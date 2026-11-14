//! Dreaming: the durable memory-consolidation queue and the workers that drain it.
//!
//! "Dreaming" is the background pass that compacts memory after the fact — merging
//! near-duplicate records, summarizing long histories, forgetting stale entries, and
//! extracting graph entities. The work is modelled as a durable queue of [`DreamJob`]s so
//! that a consolidation pass can be claimed, retried, observed, and purged independently
//! of the process that requested it.
//!
//! The module owns five layers:
//! - the data model: [`DreamState`], [`DreamKind`], [`DreamTrigger`], [`DreamJob`], and
//!   [`DreamMetrics`], plus the [`DreamOutcome`] and [`DreamRunSummary`] a sweep reports;
//! - the configuration surface: [`DreamConfig`], covering the enable switch, interval and
//!   settle delay, batch size, retry budget, retention window, trigger thresholds, and the
//!   dedupe similarity;
//! - the repository seam: [`DreamRepository`] with a process-local
//!   [`InMemoryDreamRepository`] and, behind the `sqlite` feature, the
//!   `SqliteDreamRepository` — one `dream_jobs` table (plus the `dream_job_leases` table)
//!   whose claim is a single atomic `UPDATE ... RETURNING` statement;
//! - the workers: [`DreamPlanner`] (also exported as [`DreamingEngine`]) drives one sweep,
//!   hands each claimed job to an injected [`DreamConsolidator`], and records the
//!   transition; `DreamPlanner::run_leased` does the same behind a [`DreamLeaseGuard`] so
//!   two processes cannot run the same job; [`DreamOrchestrator`] owns the idle/backoff
//!   loop and drives the planner, the retention purge, and the lease renewal;
//! - the consolidation implementations: [`LlmConsolidator`] asks a
//!   [`Model`](openjiuwen_foundation::llm::Model) which records to merge, keep, summarize,
//!   or forget and falls back to [`deterministic_plan`] with a recorded reason, and
//!   [`NoopConsolidator`] is the unwired application's default (every job finishes as
//!   terminal `Skipped`).
//!
//! Time is always a parameter: every queue, planner, retention, and salience operation
//! takes `now`, so tests and replays are reproducible. [`DreamOrchestrator::tick`] takes
//! an injected clock; only [`DreamOrchestrator::run_until_cancelled`] reads `Utc::now()`
//! for itself, and even that loop is caller-owned — this module spawns no task, starts no
//! process, and keeps no global state. Cancellation and deadlines travel through
//! [`CallOptions`](crate::record::CallOptions), which the planner checks before every
//! claim and races against the consolidator.
//!
//! What stays deferred is listed under `## Deferred` in this directory's `CONTEXT.md`:
//! ML-based salience scoring (the deterministic [`salience_score`] replaces it), a plan
//! sink backed by `LongTermMemory`, background heartbeats for a long-running
//! consolidation, and cross-process lease brokers.

mod config;
mod lease;
mod model;
mod orchestrator;
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
pub use orchestrator::*;
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
