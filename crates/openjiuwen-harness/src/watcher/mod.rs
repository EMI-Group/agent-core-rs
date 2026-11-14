//! Workspace file watcher (the `notify` feature): debounced change events for skills, instruction
//! files and MCP configuration.
//!
//! The module observes the harness roots a running `DeepAgent` reloads from and
//! publishes coalesced, typed events to subscribers and reload handlers. It is a
//! Rust-native addition — Python `openjiuwen.harness` has no watcher — and only
//! composes existing public harness APIs.
//!
//! Module layout:
//!
//! - `event` — [`WatchTargetKind`], [`ChangeKind`], [`WatchEvent`],
//!   [`ReloadFailure`], [`WatchFailure`] and [`WatchUpdate`]: the pure event
//!   model, including the [`ChangeKind::merge`] coalescing table.
//! - `debounce` — [`Debouncer`] and [`DebounceInput`]: one pending entry per
//!   path, a trailing window given as a [`std::time::Instant`], and monotonic
//!   sequence numbers starting at `1`.
//! - `targets` — [`WatchTarget`] and [`WatchPlan`]: workspace discovery
//!   (`skills`, agent templates, plugins, instruction files, MCP configs),
//!   `existing_only`, `validate`, and the raw-change classification the backend
//!   adapter feeds into the debouncer.
//! - `handle` — [`Subscription`] and [`WatcherHandle`]: bounded distribution of
//!   updates with drop-oldest semantics and clean, idempotent shutdown.
//! - `hub` — [`ReloadHandler`] and [`ReloadHub`]: the reload seam, dispatching an
//!   event to the handlers interested in its kind and recording their failures.
//! - `handlers` — the concrete reload bodies for skills, instruction files and
//!   MCP configuration, each with its handler adapter.
//! - `intake` — the bounded intake and dispatch queues plus the drop policy
//!   behind [`DropStage`] and [`WatcherHandle::dropped_notifications`].
//! - `backend` — the `notify`-gated OS adapter that registers the plan's watch
//!   paths, debounces raw notifications and drives the reload hub. It is private
//!   and only reached through [`WorkspaceWatcher::start`].
//!
//! The builder is the public front door: [`WatcherConfig`] holds the debounce
//! window, the broadcast capacity and the two queue capacities,
//! [`WorkspaceWatcher`] accumulates the plan and the reload handlers, and
//! [`WorkspaceWatcher::start`] turns them into a running [`WatcherHandle`].
//!
//! Reload handlers run on a dispatcher task of their own, so a slow handler
//! delays neither the debounce loop nor the publication of events; what it can
//! do is fill the bounded dispatch queue, and the resulting drops are counted and
//! announced as [`WatchUpdate::Dropped`].
//!
//! Every type above compiles and is fully usable without the `notify` feature;
//! only [`WorkspaceWatcher::start`] then reports the missing backend with
//! [`Error::Unsupported`](openjiuwen_common::error::Error::Unsupported).
//!
//! See `CONTEXT.md` in this directory for the full contract.
mod backend;
mod debounce;
mod event;
mod handle;
mod handlers;
mod hub;
mod intake;
mod targets;
#[cfg(all(test, feature = "notify"))]
mod tests;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use openjiuwen_common::config::OpenJiuwenHome;
use openjiuwen_common::error::Result;

use crate::cli::EnvSource;

pub use debounce::{DebounceInput, Debouncer, DEFAULT_MAX_PENDING};
pub use event::{
    ChangeKind, DropStage, DroppedUpdates, ReloadFailure, WatchEvent, WatchFailure,
    WatchTargetKind, WatchUpdate,
};
pub use handle::{Subscription, WatcherHandle};
pub use handlers::{
    InstructionReloadHandler, InstructionReloader, McpConfigReloadHandler, McpConfigReloader,
    SkillLibraryReloader, SkillsReloadHandler,
};
pub use hub::{FnReloadHandler, ReloadHandler, ReloadHub};
pub use targets::{WatchPlan, WatchTarget};

/// Default debounce window: a path has to stay quiet for 250 ms before its event
/// is published.
pub const DEFAULT_DEBOUNCE: Duration = Duration::from_millis(250);

/// Default capacity of the broadcast ring buffer behind
/// [`WatcherHandle::subscribe`].
pub const DEFAULT_EVENT_CAPACITY: usize = 256;

/// Default capacity of the bounded queue between the OS callback and the
/// debounce loop.
pub const DEFAULT_INTAKE_CAPACITY: usize = 1024;

/// Default capacity of the bounded queue between the debounce loop and the
/// reload dispatcher.
pub const DEFAULT_DISPATCH_CAPACITY: usize = 256;

/// Tunables of a [`WorkspaceWatcher`].
///
/// Every value is pure data, so a caller can construct, compare and reuse a
/// configuration without starting a watcher.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WatcherConfig {
    /// How long one path must stay quiet before its change is published.
    ///
    /// The window is a *trailing* debounce: every further notification for the
    /// same path moves its deadline another `debounce` into the future, and the
    /// changes coalesce through [`ChangeKind::merge`]. A longer window trades
    /// latency for fewer, more complete events; a zero window publishes as soon
    /// as the debouncer is polled.
    pub debounce: Duration,
    /// Number of updates the broadcast ring buffer keeps for a slow subscriber.
    ///
    /// Subscribers that fall behind lose the **oldest** updates and report the
    /// loss through [`Subscription::lagged_updates`]; publishing itself never
    /// blocks. A value of `0` is raised to `1`, because a broadcast channel
    /// cannot have zero capacity.
    pub event_capacity: usize,
    /// Number of raw OS notifications buffered between the `notify` callback and
    /// the debounce loop.
    ///
    /// The queue is bounded so a notification burst cannot grow memory while the
    /// loop is busy; when it is full the **newest** notification is dropped. The
    /// loss is never silent: it is counted in
    /// [`WatcherHandle::dropped_notifications`], logged at `warn` and published
    /// as [`WatchUpdate::Dropped`] with [`DropStage::Intake`]. A value of `0` is
    /// raised to `1`, because a bounded channel cannot have zero capacity.
    pub intake_capacity: usize,
    /// Number of debounced events buffered for the reload dispatcher.
    ///
    /// The reload handlers run on their own task, so a slow handler never delays
    /// debouncing or publication — it only fills this queue, and the events that
    /// no longer fit are dropped (newest first) and reported like an intake drop,
    /// with [`DropStage::Dispatch`]. Subscribers still receive those events;
    /// only their reload bodies did not run. A value of `0` is raised to `1`.
    pub dispatch_capacity: usize,
}

impl Default for WatcherConfig {
    /// Returns the 250 ms / 256 / 1024 / 256 defaults ([`DEFAULT_DEBOUNCE`],
    /// [`DEFAULT_EVENT_CAPACITY`], [`DEFAULT_INTAKE_CAPACITY`],
    /// [`DEFAULT_DISPATCH_CAPACITY`]).
    fn default() -> Self {
        Self {
            debounce: DEFAULT_DEBOUNCE,
            event_capacity: DEFAULT_EVENT_CAPACITY,
            intake_capacity: DEFAULT_INTAKE_CAPACITY,
            dispatch_capacity: DEFAULT_DISPATCH_CAPACITY,
        }
    }
}

impl WatcherConfig {
    /// Creates the default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Replaces the debounce window and returns the configuration.
    #[must_use]
    pub fn with_debounce(mut self, debounce: Duration) -> Self {
        self.debounce = debounce;
        self
    }

    /// Replaces the broadcast capacity and returns the configuration.
    #[must_use]
    pub fn with_event_capacity(mut self, event_capacity: usize) -> Self {
        self.event_capacity = event_capacity;
        self
    }

    /// Replaces the raw-intake capacity and returns the configuration.
    #[must_use]
    pub fn with_intake_capacity(mut self, intake_capacity: usize) -> Self {
        self.intake_capacity = intake_capacity;
        self
    }

    /// Replaces the reload-dispatch capacity and returns the configuration.
    #[must_use]
    pub fn with_dispatch_capacity(mut self, dispatch_capacity: usize) -> Self {
        self.dispatch_capacity = dispatch_capacity;
        self
    }
}

/// Builder of a workspace watcher.
///
/// A watcher is described by three things: its [`WatcherConfig`] (debounce
/// window, broadcast capacity), its [`WatchPlan`] (the roots to observe) and its
/// [`ReloadHub`] (what to refresh when one of them changes). Every method
/// consumes and returns the builder, so a whole watcher reads as one chain:
///
/// ```
/// use openjiuwen_harness::watcher::{WatchTargetKind, WorkspaceWatcher};
///
/// let watcher = WorkspaceWatcher::new().watch_dir(WatchTargetKind::Skills, "/tmp/skills");
/// assert_eq!(watcher.targets().len(), 1);
/// assert_eq!(watcher.targets()[0].kind(), WatchTargetKind::Skills);
/// ```
///
/// The type intentionally has no `Debug` implementation: a [`ReloadHub`] holds
/// trait objects, so there is nothing printable that would not leak handler
/// configuration. Use [`ReloadHub::handler_names`] through [`Self::start`]'s
/// logs instead.
pub struct WorkspaceWatcher {
    config: WatcherConfig,
    plan: WatchPlan,
    hub: ReloadHub,
}

impl WorkspaceWatcher {
    /// Creates a watcher with the default configuration, an empty plan and no
    /// reload handlers.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: WatcherConfig::default(),
            plan: WatchPlan::new(),
            hub: ReloadHub::new(),
        }
    }

    /// Replaces the whole configuration and returns the watcher.
    #[must_use]
    pub fn with_config(mut self, config: WatcherConfig) -> Self {
        self.config = config;
        self
    }

    /// Returns the configuration the watcher will run with.
    #[must_use]
    pub fn config(&self) -> &WatcherConfig {
        &self.config
    }

    /// Replaces the debounce window and returns the watcher.
    ///
    /// Shorthand for [`WatcherConfig::with_debounce`] on the current
    /// configuration.
    #[must_use]
    pub fn debounce(mut self, debounce: Duration) -> Self {
        self.config.debounce = debounce;
        self
    }

    /// Replaces the broadcast capacity and returns the watcher.
    ///
    /// Shorthand for [`WatcherConfig::with_event_capacity`] on the current
    /// configuration.
    #[must_use]
    pub fn event_capacity(mut self, event_capacity: usize) -> Self {
        self.config.event_capacity = event_capacity;
        self
    }

    /// Replaces the raw-intake capacity and returns the watcher.
    ///
    /// Shorthand for [`WatcherConfig::with_intake_capacity`] on the current
    /// configuration.
    #[must_use]
    pub fn intake_capacity(mut self, intake_capacity: usize) -> Self {
        self.config.intake_capacity = intake_capacity;
        self
    }

    /// Replaces the reload-dispatch capacity and returns the watcher.
    ///
    /// Shorthand for [`WatcherConfig::with_dispatch_capacity`] on the current
    /// configuration.
    #[must_use]
    pub fn dispatch_capacity(mut self, dispatch_capacity: usize) -> Self {
        self.config.dispatch_capacity = dispatch_capacity;
        self
    }

    /// Appends one watch target and returns the watcher.
    #[must_use]
    pub fn watch_target(mut self, target: WatchTarget) -> Self {
        self.plan.push(target);
        self
    }

    /// Appends a recursive directory target of `kind` and returns the watcher.
    #[must_use]
    pub fn watch_dir(self, kind: WatchTargetKind, path: impl Into<PathBuf>) -> Self {
        self.watch_target(WatchTarget::directory(kind, path))
    }

    /// Appends a single-file target of `kind` and returns the watcher.
    ///
    /// The backend watches the file's parent directory non-recursively, so the
    /// file may not exist yet and a delete→recreate is still observed.
    #[must_use]
    pub fn watch_file(self, kind: WatchTargetKind, path: impl Into<PathBuf>) -> Self {
        self.watch_target(WatchTarget::file(kind, path))
    }

    /// Appends every target of `plan` and returns the watcher.
    ///
    /// Targets keep their order and labels. The plan's [`WatchPlan::skipped`]
    /// record is not carried over, because the builder never consults it; call
    /// [`WatchPlan::existing_only`] and inspect `skipped` before handing the plan
    /// over when that record matters.
    #[must_use]
    // The plan is taken by value so a chained call can drop it; only its targets
    // are read, which is what `needless_pass_by_value` flags here. Accepting a
    // reference instead would not match the builder convention of the other
    // `watch_*` methods.
    #[allow(clippy::needless_pass_by_value)]
    pub fn watch_plan(mut self, plan: WatchPlan) -> Self {
        for target in plan.targets() {
            self.plan.push(target.clone());
        }
        self
    }

    /// Discovers the harness roots of `workspace` and `home`.
    ///
    /// Uses [`WatchPlan::workspace`] (skills, agent templates, plugins,
    /// instruction files, MCP configs) and keeps only the targets whose watch
    /// path already exists, as [`WatchPlan::existing_only`] does. Missing roots
    /// are dropped instead of failing, so the watcher covers whatever part of the
    /// workspace exists at start-up; add another watcher later to pick up a root
    /// that appears afterwards.
    #[must_use]
    pub fn for_workspace(workspace: &Path, home: &OpenJiuwenHome, env: &dyn EnvSource) -> Self {
        Self {
            config: WatcherConfig::default(),
            plan: WatchPlan::workspace(workspace, home, env).existing_only(),
            hub: ReloadHub::new(),
        }
    }

    /// Registers one reload handler and returns the watcher.
    ///
    /// Handlers run in registration order and only for the event kinds they
    /// declare, as [`ReloadHub::dispatch`] describes.
    #[must_use]
    pub fn on_reload<H: ReloadHandler + 'static>(mut self, handler: H) -> Self {
        self.hub.add_handler(Arc::new(handler));
        self
    }

    /// Returns the targets the watcher will observe, in insertion order.
    #[must_use]
    pub fn targets(&self) -> &[WatchTarget] {
        self.plan.targets()
    }

    /// Starts the watcher and returns its handle.
    ///
    /// The plan is validated first, then the OS backend is created and the
    /// watch paths are registered, and finally two tokio tasks are spawned: one
    /// owns the backend, debounces its notifications and publishes
    /// [`WatchUpdate::Event`] values on the handle's broadcast channel, and one
    /// runs the reload handlers for the published events. A slow handler
    /// therefore cannot delay debouncing or publication; if it falls behind, the
    /// bounded dispatch queue drops events and the loss is reported through
    /// [`WatcherHandle::dropped_dispatches`] and [`WatchUpdate::Dropped`].
    ///
    /// # Errors
    ///
    /// - [`Error::Config`](openjiuwen_common::error::Error::Config) when the plan
    ///   holds no target, when a directory target points at a file (or the other
    ///   way round), or when the OS backend cannot be created or configured.
    /// - [`Error::NotFound`](openjiuwen_common::error::Error::NotFound) when a
    ///   watch path disappeared before it could be registered.
    /// - [`Error::SysOp`](openjiuwen_common::error::Error::SysOp) when a watch
    ///   path cannot be inspected or registered (permission failure, exhausted
    ///   inotify watch limit, …).
    /// - [`Error::Unsupported`](openjiuwen_common::error::Error::Unsupported)
    ///   when the crate was built without the `notify` feature.
    ///
    /// Every message names the offending path and the target label.
    pub async fn start(self) -> Result<WatcherHandle> {
        backend::start(self).await
    }
}

impl Default for WorkspaceWatcher {
    /// Same as [`WorkspaceWatcher::new`].
    fn default() -> Self {
        Self::new()
    }
}
