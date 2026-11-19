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
//! [`WatcherConfig`] also parses from and renders to the single-line form
//! `debounce=250ms,event_capacity=256,intake_capacity=1024,dispatch_capacity=256`
//! (`FromStr`/`Display`, with a bare duration such as `250ms` meaning
//! `debounce=250ms`), so an environment variable or a command-line flag is one
//! `parse()` call and the canonical text round-trips.
//!
//! Reload handlers run on a dispatcher task of their own, so a slow handler
//! delays neither the debounce loop nor the publication of events; what it can
//! do is fill the bounded dispatch queue, and the resulting drops are counted and
//! announced as [`WatchUpdate::Dropped`].
//!
//! Every type above compiles and is fully usable without the `notify` feature;
//! only [`WorkspaceWatcher::start`] then reports the missing backend with
//! [`Error::Unsupported`].
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

use std::fmt;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;

use openjiuwen_common::config::OpenJiuwenHome;
use openjiuwen_common::error::{Error, Result};

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
/// Whether this build compiled the real `notify`-backed watcher backend.
///
/// `true` when `openjiuwen-harness` was built with the `notify` feature, so
/// [`WorkspaceWatcher::start`] arms an OS watcher; `false` when it was not, so
/// `start` answers [`Error::Unsupported`] before it looks at the plan.
///
/// Consumers that must branch on the backend at run time should read this
/// constant instead of their own `cfg!(feature = "notify")`: cargo unifies
/// features per crate, so a binary that enables `notify` through another
/// dependency (for example `openjiuwen-cli`, which turns it on unconditionally)
/// links the `notify`-enabled harness while its own `cfg!` still reads `false`.
pub const WATCHER_BACKEND_AVAILABLE: bool = cfg!(feature = "notify");

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

/// Keys accepted by [`WatcherConfig`]'s [`FromStr`] implementation, for error
/// messages.
const CONFIG_KEYS: &str = "`debounce`, `event_capacity`, `intake_capacity` or `dispatch_capacity`";

impl FromStr for WatcherConfig {
    type Err = Error;

    /// Parses a comma-separated `key=value` specification, or a bare duration.
    ///
    /// Accepted keys are `debounce` (a duration), `event_capacity`,
    /// `intake_capacity` and `dispatch_capacity` (unsigned integers); keys that
    /// are not named keep their [`WatcherConfig::default`] value, and `0` is
    /// accepted for every capacity because the watcher raises it to `1` where
    /// the channel is built. Surrounding whitespace around items, keys and
    /// values is ignored.
    ///
    /// Durations take an `ns`, `us`, `ms`, `s` or `m` suffix and may carry a
    /// decimal fraction (`250ms`, `1.5s`, `0.5m`); a bare number is milliseconds.
    /// A specification without any `=` is the duration-only shorthand for
    /// `debounce=…`, so a single `--debounce 250ms` flag is one `parse()` call:
    ///
    /// ```
    /// use std::str::FromStr;
    ///
    /// use openjiuwen_harness::watcher::WatcherConfig;
    ///
    /// let config = WatcherConfig::from_str("debounce=1s, intake_capacity=64").unwrap();
    /// assert_eq!(config.debounce.as_millis(), 1_000);
    /// assert_eq!(config.intake_capacity, 64);
    /// assert_eq!(config.event_capacity, WatcherConfig::default().event_capacity);
    ///
    /// assert_eq!(WatcherConfig::from_str("250ms").unwrap().debounce.as_millis(), 250);
    /// ```
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] naming the offending token when the specification is
    /// empty, an item is empty or lacks a value, a key is unknown or repeated, or
    /// a value is not a valid duration or capacity.
    fn from_str(spec: &str) -> Result<Self> {
        let spec = spec.trim();
        if spec.is_empty() {
            return Err(Error::validation(format!(
                "watcher config is empty; expected {CONFIG_KEYS}, or a bare duration such as `250ms`"
            )));
        }
        if !spec.contains('=') {
            return Ok(Self::default().with_debounce(parse_watch_duration(spec)?));
        }

        let mut config = Self::default();
        let mut seen = [false; 4];
        for item in spec.split(',') {
            let item = item.trim();
            if item.is_empty() {
                return Err(Error::validation(format!(
                    "empty item in watcher config `{spec}`; expected {CONFIG_KEYS}"
                )));
            }
            let Some((key, value)) = item.split_once('=') else {
                return Err(Error::validation(format!(
                    "watcher config item `{item}` has no `=value`; expected {CONFIG_KEYS}"
                )));
            };
            let key = key.trim();
            let value = value.trim();
            let slot = match key {
                "debounce" => 0,
                "event_capacity" => 1,
                "intake_capacity" => 2,
                "dispatch_capacity" => 3,
                other => {
                    return Err(Error::validation(format!(
                        "unknown watcher config key `{other}` in `{item}`; expected {CONFIG_KEYS}"
                    )))
                }
            };
            if std::mem::replace(&mut seen[slot], true) {
                return Err(Error::validation(format!(
                    "duplicate watcher config key `{key}` in `{spec}`"
                )));
            }
            if value.is_empty() {
                return Err(Error::validation(format!(
                    "watcher config key `{key}` has no value in `{item}`"
                )));
            }
            match slot {
                0 => config.debounce = parse_watch_duration(value)?,
                1 => config.event_capacity = parse_watch_capacity(key, value)?,
                2 => config.intake_capacity = parse_watch_capacity(key, value)?,
                _ => config.dispatch_capacity = parse_watch_capacity(key, value)?,
            }
        }
        Ok(config)
    }
}

impl fmt::Display for WatcherConfig {
    /// Writes the canonical `debounce=…,event_capacity=…,intake_capacity=…,
    /// dispatch_capacity=…` form.
    ///
    /// The output always names every key and uses a unit suffix for the
    /// debounce window, so `config.to_string().parse()` returns an equal
    /// configuration ([`FromStr`] composed with `Display` is stable).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "debounce={},event_capacity={},intake_capacity={},dispatch_capacity={}",
            format_watch_duration(self.debounce),
            self.event_capacity,
            self.intake_capacity,
            self.dispatch_capacity
        )
    }
}

/// Parses a watcher duration: `ns`/`us`/`ms`/`s`/`m` suffix (optionally with a
/// decimal fraction), or a bare integer meaning milliseconds.
fn parse_watch_duration(token: &str) -> Result<Duration> {
    let token = token.trim();
    let (number, unit_nanos) = if let Some(rest) = token.strip_suffix("ns") {
        (rest, 1_u128)
    } else if let Some(rest) = token.strip_suffix("us") {
        (rest, 1_000)
    } else if let Some(rest) = token.strip_suffix("ms") {
        (rest, 1_000_000)
    } else if let Some(rest) = token.strip_suffix('s') {
        (rest, 1_000_000_000)
    } else if let Some(rest) = token.strip_suffix('m') {
        (rest, 60_000_000_000)
    } else {
        // A bare value is a millisecond count, and only an integer one.
        if !is_digits(token) {
            return Err(invalid_duration(token));
        }
        let millis = token.parse::<u128>().map_err(|_| invalid_duration(token))?;
        let nanos = millis
            .checked_mul(1_000_000)
            .ok_or_else(|| invalid_duration(token))?;
        return nanos_to_duration(nanos, token);
    };

    let (whole, fraction) = match number.split_once('.') {
        Some((whole, fraction)) => (whole, Some(fraction)),
        None => (number, None),
    };
    if !is_digits(whole) {
        return Err(invalid_duration(token));
    }
    let fraction = match fraction {
        Some(fraction) if is_digits(fraction) => fraction,
        Some(_) => return Err(invalid_duration(token)),
        None => "",
    };
    let scale = 10_u128
        .checked_pow(u32::try_from(fraction.len()).map_err(|_| invalid_duration(token))?)
        .ok_or_else(|| invalid_duration(token))?;
    let whole = whole.parse::<u128>().map_err(|_| invalid_duration(token))?;
    let fraction = if fraction.is_empty() {
        0
    } else {
        fraction
            .parse::<u128>()
            .map_err(|_| invalid_duration(token))?
    };
    let numerator = whole
        .checked_mul(scale)
        .and_then(|value| value.checked_add(fraction))
        .ok_or_else(|| invalid_duration(token))?;
    let scaled = numerator
        .checked_mul(unit_nanos)
        .ok_or_else(|| invalid_duration(token))?;
    if scaled % scale != 0 {
        // A fraction finer than a nanosecond cannot be represented exactly.
        return Err(invalid_duration(token));
    }
    nanos_to_duration(scaled / scale, token)
}

/// Parses a watcher capacity: an unsigned integer, `0` included.
fn parse_watch_capacity(key: &str, value: &str) -> Result<usize> {
    if !is_digits(value) {
        return Err(invalid_capacity(key, value));
    }
    value
        .parse::<usize>()
        .map_err(|_| invalid_capacity(key, value))
}

/// Renders a duration in the canonical unit for its magnitude, so
/// [`parse_watch_duration`] reads it back unchanged.
fn format_watch_duration(duration: Duration) -> String {
    let nanos = duration.as_nanos();
    if nanos % 1_000_000_000 == 0 {
        format!("{}s", duration.as_secs())
    } else if nanos % 1_000_000 == 0 {
        format!("{}ms", nanos / 1_000_000)
    } else if nanos % 1_000 == 0 {
        format!("{}us", nanos / 1_000)
    } else {
        format!("{nanos}ns")
    }
}

/// Converts a nanosecond count, rejecting values [`Duration`] cannot hold.
fn nanos_to_duration(nanos: u128, token: &str) -> Result<Duration> {
    let nanos = u64::try_from(nanos).map_err(|_| invalid_duration(token))?;
    Ok(Duration::from_nanos(nanos))
}

/// True for a non-empty run of ASCII digits (no sign, no separator, no exponent).
fn is_digits(value: &str) -> bool {
    !value.is_empty() && value.bytes().all(|byte| byte.is_ascii_digit())
}

/// Error for a duration token that does not parse.
fn invalid_duration(token: &str) -> Error {
    Error::validation(format!(
        "invalid watcher duration `{token}`; expected `ns`/`us`/`ms`/`s`/`m` (for example `250ms` \
         or `1.5s`), or a bare number of milliseconds"
    ))
}

/// Error for a capacity token that does not parse.
fn invalid_capacity(key: &str, value: &str) -> Error {
    Error::validation(format!(
        "invalid watcher {key} `{value}`; expected a non-negative integer"
    ))
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
    /// - [`Error::Config`] when the plan
    ///   holds no target, when a directory target points at a file (or the other
    ///   way round), or when the OS backend cannot be created or configured.
    /// - [`Error::NotFound`] when a
    ///   watch path disappeared before it could be registered.
    /// - [`Error::SysOp`] when a watch
    ///   path cannot be inspected or registered (permission failure, exhausted
    ///   inotify watch limit, …).
    /// - [`Error::Unsupported`]
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

#[cfg(test)]
mod config_tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::{
        format_watch_duration, parse_watch_duration, WatcherConfig, DEFAULT_DEBOUNCE,
        DEFAULT_DISPATCH_CAPACITY, DEFAULT_EVENT_CAPACITY, DEFAULT_INTAKE_CAPACITY,
    };
    use std::time::Duration;

    use openjiuwen_common::error::Error;

    fn parse(spec: &str) -> WatcherConfig {
        spec.parse::<WatcherConfig>().expect("spec should parse")
    }

    fn validation_error(spec: &str) -> String {
        match spec.parse::<WatcherConfig>() {
            Ok(config) => panic!("`{spec}` should not parse, got {config}"),
            Err(Error::Validation(message)) => message,
            Err(other) => panic!("expected a validation error, got {other:?}"),
        }
    }

    #[test]
    fn parses_the_full_spec() {
        let config = parse("debounce=1s,event_capacity=8,intake_capacity=16,dispatch_capacity=32");
        assert_eq!(config.debounce, Duration::from_secs(1));
        assert_eq!(config.event_capacity, 8);
        assert_eq!(config.intake_capacity, 16);
        assert_eq!(config.dispatch_capacity, 32);
    }

    #[test]
    fn missing_keys_take_their_defaults() {
        let config = parse("intake_capacity=64");
        assert_eq!(config.debounce, DEFAULT_DEBOUNCE);
        assert_eq!(config.event_capacity, DEFAULT_EVENT_CAPACITY);
        assert_eq!(config.intake_capacity, 64);
        assert_eq!(config.dispatch_capacity, DEFAULT_DISPATCH_CAPACITY);

        let config = parse("debounce=5ms");
        assert_eq!(config.debounce, Duration::from_millis(5));
        assert_eq!(config.intake_capacity, DEFAULT_INTAKE_CAPACITY);
    }

    #[test]
    fn accepts_a_bare_duration_as_the_debounce_window() {
        let config = parse("250ms");
        assert_eq!(config.debounce, Duration::from_millis(250));
        assert_eq!(config.event_capacity, DEFAULT_EVENT_CAPACITY);
        assert_eq!(config.intake_capacity, DEFAULT_INTAKE_CAPACITY);
        assert_eq!(config.dispatch_capacity, DEFAULT_DISPATCH_CAPACITY);

        assert_eq!(parse(" 1.5s ").debounce, Duration::from_millis(1_500));
    }

    #[test]
    fn accepts_every_duration_unit_and_fraction() {
        assert_eq!(parse("7ns").debounce, Duration::from_nanos(7));
        assert_eq!(parse("1500us").debounce, Duration::from_micros(1_500));
        assert_eq!(parse("250ms").debounce, Duration::from_millis(250));
        assert_eq!(parse("1.5s").debounce, Duration::from_millis(1_500));
        assert_eq!(parse("0.5s").debounce, Duration::from_millis(500));
        assert_eq!(parse("2m").debounce, Duration::from_secs(120));
        assert_eq!(parse("0.5m").debounce, Duration::from_secs(30));
        // A bare number is milliseconds, and an integer one only.
        assert_eq!(parse("300").debounce, Duration::from_millis(300));
        assert_eq!(parse("0").debounce, Duration::ZERO);
        assert_eq!(parse("0ms").debounce, Duration::ZERO);
        assert_eq!(
            parse("debounce=1.25s").debounce,
            Duration::from_millis(1_250)
        );
    }

    #[test]
    fn trims_whitespace_around_items_keys_and_values() {
        let config = parse("  debounce = 1s , intake_capacity = 12  ");
        assert_eq!(config.debounce, Duration::from_secs(1));
        assert_eq!(config.intake_capacity, 12);
    }

    #[test]
    fn accepts_zero_capacities_because_the_watcher_clamps_them() {
        let config = parse("event_capacity=0,intake_capacity=0,dispatch_capacity=0");
        assert_eq!(config.event_capacity, 0);
        assert_eq!(config.intake_capacity, 0);
        assert_eq!(config.dispatch_capacity, 0);
    }

    #[test]
    fn display_emits_the_canonical_form() {
        assert_eq!(
            WatcherConfig::default().to_string(),
            "debounce=250ms,event_capacity=256,intake_capacity=1024,dispatch_capacity=256"
        );
        assert_eq!(
            parse("intake_capacity=64").to_string(),
            "debounce=250ms,event_capacity=256,intake_capacity=64,dispatch_capacity=256"
        );
        assert_eq!(
            WatcherConfig::default()
                .with_debounce(Duration::from_millis(1_500))
                .to_string(),
            "debounce=1500ms,event_capacity=256,intake_capacity=1024,dispatch_capacity=256"
        );
    }

    #[test]
    fn display_and_from_str_round_trip() {
        let canonical = [
            "debounce=250ms,event_capacity=256,intake_capacity=1024,dispatch_capacity=256",
            "debounce=1s,event_capacity=1,intake_capacity=2,dispatch_capacity=3",
            "debounce=7ns,event_capacity=0,intake_capacity=0,dispatch_capacity=0",
            "debounce=1500us,event_capacity=8,intake_capacity=16,dispatch_capacity=32",
            "debounce=120s,event_capacity=5,intake_capacity=5,dispatch_capacity=5",
            "debounce=0s,event_capacity=9,intake_capacity=9,dispatch_capacity=9",
        ];
        for spec in canonical {
            let config = parse(spec);
            assert_eq!(config.to_string(), spec, "canonical rendering of `{spec}`");
            assert_eq!(parse(spec).to_string(), spec, "stability of `{spec}`");
        }

        // Non-canonical spellings keep their meaning through the canonical form.
        let non_canonical = [
            "debounce=2m",
            "debounce=0.25s,event_capacity=8",
            "debounce=300",
            "1.5s",
            "debounce=0,event_capacity=0,intake_capacity=0,dispatch_capacity=0",
        ];
        for spec in non_canonical {
            let config = parse(spec);
            let rendered = config.to_string();
            let reparsed = parse(&rendered);
            assert_eq!(reparsed, config, "round-trip of `{spec}` via `{rendered}`");
            assert_eq!(reparsed.to_string(), rendered, "stable form of `{spec}`");
        }

        let default = WatcherConfig::default();
        assert_eq!(parse(&default.to_string()), default);

        for duration in [
            Duration::ZERO,
            Duration::from_nanos(1),
            Duration::from_nanos(1_500),
            Duration::from_micros(1),
            Duration::from_millis(250),
            Duration::from_secs(1),
            Duration::from_secs(3_600),
        ] {
            let rendered = format_watch_duration(duration);
            assert_eq!(
                parse_watch_duration(&rendered).expect("canonical duration parses"),
                duration,
                "round-trip of {duration:?} rendered as `{rendered}`"
            );
            assert_eq!(format_watch_duration(duration), rendered);
        }
    }

    #[test]
    fn rejects_an_empty_spec() {
        let message = validation_error("");
        assert!(message.contains("empty"), "{message}");
        assert!(validation_error("   ").contains("empty"));
    }

    #[test]
    fn rejects_an_unknown_key() {
        let message = validation_error("debounce=1s,bogus=1");
        assert!(message.contains("bogus"), "{message}");
        assert!(message.contains("unknown"), "{message}");
    }

    #[test]
    fn rejects_a_duplicate_key() {
        let message = validation_error("intake_capacity=4,intake_capacity=8");
        assert!(message.contains("duplicate"), "{message}");
        assert!(message.contains("intake_capacity"), "{message}");
    }

    #[test]
    fn rejects_an_empty_item() {
        assert!(validation_error("debounce=1s,").contains("empty item"));
        assert!(validation_error("debounce=1s,,intake_capacity=4").contains("empty item"));
    }

    #[test]
    fn rejects_an_item_without_a_value() {
        let message = validation_error("debounce=1s,250ms");
        assert!(message.contains("250ms"), "{message}");

        let message = validation_error("intake_capacity=");
        assert!(message.contains("no value"), "{message}");
        assert!(message.contains("intake_capacity"), "{message}");
    }

    #[test]
    fn rejects_an_unparseable_capacity() {
        for spec in [
            "event_capacity=abc",
            "event_capacity=-1",
            "event_capacity=1.5",
            "event_capacity=99999999999999999999999999",
            "debounce=1s,dispatch_capacity=12x",
        ] {
            let message = validation_error(spec);
            assert!(
                message.contains("non-negative integer"),
                "{spec}: {message}"
            );
        }
        assert!(validation_error("event_capacity=-1").contains("-1"));
    }

    #[test]
    fn rejects_an_unparseable_duration() {
        for spec in [
            "debounce=abc",
            "debounce=1x",
            "debounce=-5ms",
            "debounce=1.s",
            "debounce=.5s",
            "debounce=1.5",
            "1e3",
            "debounce=99999999999999999999999999s",
            "debounce=0.0000000001s",
        ] {
            let message = validation_error(spec);
            assert!(
                message.contains("invalid watcher duration"),
                "{spec}: {message}"
            );
        }
        assert!(validation_error("debounce=abc").contains("abc"));
    }
}
