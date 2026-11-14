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
//!
//! Every type above compiles and is fully usable without the `notify` feature;
//! only the OS backend and the `WorkspaceWatcher` builder are added in the next
//! round.
//!
//! See `CONTEXT.md` in this directory for the full contract.
mod debounce;
mod event;
mod handle;
mod handlers;
mod hub;
mod targets;

pub use debounce::{DebounceInput, Debouncer, DEFAULT_MAX_PENDING};
pub use event::{
    ChangeKind, ReloadFailure, WatchEvent, WatchFailure, WatchTargetKind, WatchUpdate,
};
pub use handle::{Subscription, WatcherHandle};
pub use handlers::{
    InstructionReloadHandler, InstructionReloader, McpConfigReloadHandler, McpConfigReloader,
    SkillLibraryReloader, SkillsReloadHandler,
};
pub use hub::{FnReloadHandler, ReloadHandler, ReloadHub};
pub use targets::{WatchPlan, WatchTarget};
