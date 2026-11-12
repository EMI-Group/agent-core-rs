//! Dreaming: the deferred/offline memory-consolidation job queue.
//!
//! "Dreaming" is the background pass that compacts memory after the fact — merging
//! near-duplicate records, summarizing long histories, forgetting stale entries, and
//! extracting graph entities. The work is modelled as a durable queue of [`DreamJob`]s
//! so that a consolidation pass can be claimed, retried, observed, and purged
//! independently of the process that requested it.
//!
//! This module owns three things:
//! - the data model ([`DreamState`], [`DreamKind`], [`DreamTrigger`], [`DreamJob`],
//!   [`DreamOutcome`], [`DreamRunSummary`], [`DreamMetrics`]);
//! - the configuration surface ([`DreamConfig`]) for intervals, batch size, retries,
//!   retention, and the trigger thresholds;
//! - the repository seam ([`DreamRepository`]) with an in-memory backend
//!   ([`InMemoryDreamRepository`]) used by tests and single-process applications.
//!
//! The persistence backends and the scheduler are deliberately staged: this module is
//! stage 1 of the port and ships the job model, the repository contract, and the
//! in-memory implementation. The following pieces of Python's
//! `openjiuwen.core.memory.dreaming` package are **not** ported here:
//! - the scheduled/background orchestrator (`DreamingOrchestrator`: the periodic sweep
//!   loop and its busy-backoff / jitter behaviour);
//! - the LLM-driven consolidation pipeline that decides what to merge, summarize, or
//!   forget (the deterministic dedupe rules in [`crate::recall`] remain the only
//!   consolidation logic);
//! - cross-process scheduling and worker leases (the SQLite backend in stage 2 brings
//!   row-level claiming, not a distributed lock);
//! - daemon-side retention enforcement (the sweep that periodically calls
//!   [`DreamRepository::purge`] on a timer);
//! - ML-based salience/importance scoring for deciding which memories are worth
//!   consolidating.
//!
//! The deferred pieces keep their seams: a caller can already enqueue work, claim a
//! deterministic head of the queue, record an outcome, and purge finished rows.
mod config;
mod model;
mod repository;
pub use config::*;
pub use model::*;
pub use repository::*;
