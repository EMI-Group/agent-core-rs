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
//!
//! Every type above compiles and is fully usable without the `notify` feature;
//! only the OS backend, subscriptions, reload handlers and the concrete reload
//! bodies are added in a later round.
//!
//! See `CONTEXT.md` in this directory for the full contract.
mod debounce;
mod event;
mod targets;

pub use debounce::{DebounceInput, Debouncer, DEFAULT_MAX_PENDING};
pub use event::{
    ChangeKind, ReloadFailure, WatchEvent, WatchFailure, WatchTargetKind, WatchUpdate,
};
pub use targets::{WatchPlan, WatchTarget};
