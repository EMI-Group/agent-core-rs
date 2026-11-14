//! Dreaming: the deferred/offline memory-consolidation job queue.
//!
//! "Dreaming" is the background pass that compacts memory after the fact — merging
//! near-duplicate records, summarizing long histories, forgetting stale entries, and
//! extracting graph entities. The work is modelled as a durable queue of [`DreamJob`]s
//! so that a consolidation pass can be claimed, retried, observed, and purged
//! independently of the process that requested it.
//!
//! The module owns four things:
//! - the data model: [`DreamState`], [`DreamKind`], [`DreamTrigger`], [`DreamJob`], and
//!   [`DreamMetrics`], plus the [`DreamOutcome`] and [`DreamRunSummary`] a sweep reports;
//! - the configuration surface: [`DreamConfig`], covering the enable switch, interval and
//!   settle delay, batch size, retry budget, retention window, trigger thresholds, and the
//!   dedupe similarity;
//! - the repository seam: [`DreamRepository`] with a process-local
//!   [`InMemoryDreamRepository`] and, behind the `sqlite` feature, the
//!   `SqliteDreamRepository` — one `dream_jobs` table whose claim is a single atomic
//!   `UPDATE ... RETURNING` statement;
//! - the orchestration: [`DreamPlanner`] (also exported as [`DreamingEngine`]) drives one
//!   sweep, hands each claimed job to an injected [`DreamConsolidator`], and records the
//!   transition. [`NoopConsolidator`] is the unwired application's default and finishes
//!   every job as terminal `Skipped`.
//!
//! Nothing here reads a timer or a clock of its own: the caller passes `now` in, and an
//! embedding application (or a future scheduler) decides when to call
//! [`DreamPlanner::run_due`]. Cancellation and deadlines travel through
//! [`CallOptions`](crate::record::CallOptions), which the planner checks before every
//! claim and races against the consolidator.
//!
//! The following pieces of Python's `openjiuwen.core.memory.dreaming` package are **not**
//! ported:
//! - the scheduled background orchestrator with its busy-backoff/idle loop
//!   (`DreamingOrchestrator`); the planner ships the sweep, not the timer;
//! - the LLM-driven consolidation pipeline that decides what to merge, summarize, or
//!   forget ([`DreamConsolidator`] is the injection point; the deterministic dedupe rules
//!   in [`crate::recall`] remain the only consolidation logic in the crate);
//! - cross-process workers and leases: the SQLite backend claims rows atomically, which is
//!   single-statement concurrency control, not a distributed lock or a lease renewal;
//! - daemon-side retention enforcement: nothing calls [`DreamRepository::purge`] on a
//!   timer, so a caller must invoke it;
//! - ML-based salience/importance scoring for deciding which memories are worth
//!   consolidating.
mod config;
mod lease;
mod model;
mod outcome;
mod pipeline;
mod planner;
mod prompts;
mod repository;
mod retention;
mod salience;
pub use config::*;
pub use lease::*;
pub use model::*;
pub use outcome::*;
pub use pipeline::*;
pub use planner::*;
pub use prompts::*;
pub use repository::*;
pub use retention::*;
pub use salience::*;

#[cfg(feature = "sqlite")]
mod sqlite;
#[cfg(feature = "sqlite")]
pub use sqlite::*;
