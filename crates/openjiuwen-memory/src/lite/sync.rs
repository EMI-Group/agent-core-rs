//! Caller-driven sync triggers (`on_session_start`, `on_interval`, `on_write`) and a
//! deterministic interval scheduler.
//!
//! Ports the deferred sync triggers of Python `core/memory/lite/`: the store exposes points
//! at which the derived state (indexes, embeddings, rollups) is reconciled with the
//! Markdown source of truth, and the caller decides when those points occur.
//!
//! # Triggers
//!
//! - `on_session_start` — reconcile once when a session begins, so a store touched by
//!   another process is current before the first recall.
//! - `on_interval` — reconcile after a configured elapsed interval, measured on an injected
//!   clock so tests are deterministic and no wall-clock ordering leaks in.
//! - `on_write` — reconcile immediately after a mutation, for callers that need the derived
//!   state to be current as soon as a write returns.
//!
//! Triggers are additive and each is independently enableable; the default configuration
//! enables none of them, so the store does nothing the caller did not ask for.
//!
//! # Scheduler
//!
//! The interval scheduler is deterministic: it computes the next due instant from the last
//! completed run plus the configured interval, reports whether a run is due for a given
//! `now`, and never depends on a timer firing promptly. A due check is idempotent within the
//! same instant, so two callers cannot schedule the same run twice.
//!
//! # No background daemon
//!
//! Nothing here starts a background loop. Reconciliation runs on the caller's task, and any
//! optional background task is explicit opt-in, lives on the caller's tokio runtime, wakes
//! no more often than the configured interval, and is cancellation-aware: it holds a
//! cancellation token, stops promptly when the token is cancelled or its handle is dropped,
//! and returns without leaving a detached task behind.
//!
//! # Errors
//!
//! A failing reconciliation surfaces the underlying typed error (filesystem, index, or
//! embedding) to the caller. A trigger that cannot run (disabled, not due, already
//! running) reports that explicitly instead of silently succeeding.
