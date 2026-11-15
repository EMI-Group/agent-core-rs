//! [`DreamPlanner`]: the timer-free orchestration over the dream queue.
//!
//! The planner owns the loop an application (or a later background scheduler) drives:
//! it claims the queue head, hands the job to an injected [`DreamConsolidator`], and
//! records the outcome back through the [`DreamRepository`]. It never decides *what* to
//! consolidate and it never reads the clock for itself — `now` is a parameter, which
//! keeps sweeps reproducible in tests and lets a caller apply its own scheduling policy.
//!
//! The split of responsibilities is deliberate:
//! - the [`DreamRepository`] owns persistence and the arm of the state machine that is
//!   pure bookkeeping (claim, attempt counting, requeue-or-fail, retention);
//! - the [`DreamConsolidator`] owns the memory mutation that a pass performs;
//! - the planner owns the sequencing, the cancellation policy, and the mapping from a
//!   consolidator result to the repository transition.
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use openjiuwen_common::error::{Error, Result};

use crate::record::CallOptions;

use super::config::DreamConfig;
use super::model::{DreamJob, DreamMetrics, DreamState};
use super::outcome::{DreamOutcome, DreamRunSummary};
use super::repository::{DreamRepository, InMemoryDreamRepository};

// The lease-aware sweep. Inherent methods declared in the child module are still public
// API on [`DreamPlanner`], and the child can use the private helpers below.
mod lease;

/// One consolidator call's result, before the repository transition is recorded.
///
/// Keeping "ask the consolidator" and "record the transition" apart is exactly what lets
/// the lease-aware sweep renew its lease in between: [`DreamPlanner::attempt`] runs the
/// two steps back to back for a trusting sweep, while
/// [`DreamPlanner::run_leased`](DreamPlanner::run_leased) renews before recording.
enum AttemptResult {
    /// The consolidator produced an outcome to record.
    Done(DreamOutcome),
    /// The consolidator, or [`CallOptions::run`] around it, failed.
    Failed(Error),
}

/// The memory-consolidation work one claimed [`DreamJob`] performs.
///
/// The consolidator owns the actual memory mutation and its own cooperative
/// cancellation: it should check `ctx` around expensive awaits (or hand it to the engine
/// call it makes) and stop when the token fires.
///
/// A correct implementation returns a *terminal* outcome — `Succeeded`, `Skipped`, or
/// `Cancelled` — or an error describing why the attempt failed. It must never touch the
/// [`DreamRepository`]: the planner records the state transition and owns attempt
/// accounting, so a consolidator cannot double-count an attempt or complete a job it does
/// not own.
#[async_trait]
pub trait DreamConsolidator: Send + Sync {
    /// Consolidates `job`'s scope and reports the terminal outcome.
    async fn consolidate(&self, job: &DreamJob, ctx: &CallOptions) -> Result<DreamOutcome>;
}

/// The consolidator an application that has not wired a pipeline gets.
///
/// It reports [`DreamState::Skipped`] with zero metrics and the note
/// `"no consolidator configured"`. That is the skip semantics, not a failure: the job
/// finishes terminally and is never retried, so an unwired application drains its queue
/// instead of accumulating retries for work nobody performs.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoopConsolidator;

#[async_trait]
impl DreamConsolidator for NoopConsolidator {
    async fn consolidate(&self, job: &DreamJob, _ctx: &CallOptions) -> Result<DreamOutcome> {
        Ok(DreamOutcome::skipped(
            job.id.clone(),
            "no consolidator configured",
        ))
    }
}

/// Note recorded when a consolidator returns an outcome that is not terminal.
const NON_TERMINAL_NOTE: &str = "consolidator returned a non-terminal dream state";

/// Drives dream jobs from the queue through a [`DreamConsolidator`].
///
/// The planner is stateless apart from its collaborators: every call to [`Self::run`] is
/// one sweep, and all durable state lives in the repository. Callers that want the
/// periodic behaviour of Python's `DreamingOrchestrator` own the timer and call
/// [`Self::run_due`] themselves.
pub struct DreamPlanner {
    /// The queue the sweep claims from.
    repository: Arc<dyn DreamRepository>,
    /// The pass performed for every claimed job.
    consolidator: Arc<dyn DreamConsolidator>,
    /// Batch size, enable switch, and the retention window.
    config: DreamConfig,
}

/// The planner under its engine-facing name; both aliases are public.
///
/// `DreamingEngine` is the name an embedding application (or a future scheduler) uses;
/// `DreamPlanner` says what it does.
pub type DreamingEngine = DreamPlanner;

impl fmt::Debug for DreamPlanner {
    /// Prints the configuration and the collaborator kinds.
    ///
    /// The repository and consolidator are trait objects without a `Debug` bound, so
    /// only their type names appear here; the job state itself is queried from the
    /// repository.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DreamPlanner")
            .field("config", &self.config)
            .field("repository", &"Arc<dyn DreamRepository>")
            .field("consolidator", &"Arc<dyn DreamConsolidator>")
            .finish()
    }
}

impl DreamPlanner {
    /// Builds a planner over `repository` and `consolidator`, validating `config`.
    ///
    /// A configuration that would busy-spin or never finish (see
    /// [`DreamConfig::validate`]) is rejected here instead of failing mid-sweep.
    pub fn new(
        repository: Arc<dyn DreamRepository>,
        consolidator: Arc<dyn DreamConsolidator>,
        config: DreamConfig,
    ) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            repository,
            consolidator,
            config,
        })
    }

    /// Builds a planner over a fresh [`InMemoryDreamRepository`].
    ///
    /// Convenience for tests and single-process applications; the jobs are not durable.
    pub fn with_in_memory(
        consolidator: Arc<dyn DreamConsolidator>,
        config: DreamConfig,
    ) -> Result<Self> {
        Self::new(
            Arc::new(InMemoryDreamRepository::new()),
            consolidator,
            config,
        )
    }

    /// The queue the planner claims from.
    #[must_use]
    pub fn repository(&self) -> &Arc<dyn DreamRepository> {
        &self.repository
    }

    /// The planner's configuration.
    #[must_use]
    pub fn config(&self) -> &DreamConfig {
        &self.config
    }

    /// Runs one sweep: at most `limit` claims at the caller-supplied clock `now`.
    ///
    /// Semantics, in order:
    ///
    /// - A disabled [`DreamConfig::enabled`] or `limit == 0` returns an empty
    ///   [`DreamRunSummary`] without claiming anything.
    /// - Cancellation is checked *before* every claim, so a cancelled sweep never claims
    ///   a job just to abandon it. A sweep cancelled while a consolidator is running
    ///   completes that job as [`DreamState::Cancelled`] and stops.
    /// - The loop claims at most `limit` times and stops early when the queue has no
    ///   eligible job (`claim_next` returns `None`).
    /// - A consolidator result of `Ok(outcome)` is stored through
    ///   [`DreamRepository::complete`]. The planner is authoritative about the job id: a
    ///   mismatched or empty `outcome.job_id` is overwritten with the claimed job's id
    ///   and logged, rather than completing the wrong row. An outcome whose state is not
    ///   terminal cannot be completed ([`DreamRepository::complete`] would reject it), so
    ///   it takes the failure path ([`DreamRepository::fail`]) instead.
    /// - A consolidator result of `Err(error)` — including the error [`CallOptions::run`]
    ///   synthesises for cancellation and for a deadline — never completes the job as
    ///   successful. [`Error::Cancelled`] (from either layer) completes it as
    ///   [`DreamState::Cancelled`] and breaks the loop, which is the cooperative shutdown
    ///   path. Every other error, [`Error::Timeout`] included, is recorded through
    ///   [`DreamRepository::fail`]; the recorded outcome carries a note saying whether the
    ///   attempt was requeued (`attempt n/max`) or exhausted the budget (terminal
    ///   failure), plus the error text.
    ///
    /// [`DreamRunSummary::scanned`] counts every claimed job; `failed` counts failed
    /// *attempts*, so it includes attempts that were rescheduled for a retry. Because the
    /// repository's retry marker is `started_at`, a requeued job is not claimable again at
    /// the same `now` — the next attempt needs a later clock, which is why retries are
    /// driven by successive `run` calls (or by a scheduler that advances time).
    ///
    /// A repository error from `claim_next`/`complete`/`fail` is returned to the caller
    /// and aborts the sweep; transitions that already committed stay committed, so the
    /// remaining jobs are simply left for the next sweep. A failed transition write never
    /// strands the claimed job: before the error propagates the planner requeues it through
    /// [`DreamRepository::fail`] (best effort, logged), so it is `Pending` again — or
    /// terminally `Failed` when the attempt budget is spent — and a later sweep can claim
    /// it.
    pub async fn run(
        &self,
        now: DateTime<Utc>,
        limit: usize,
        ctx: &CallOptions,
    ) -> Result<DreamRunSummary> {
        let mut summary = DreamRunSummary::new();
        if !self.config.enabled {
            tracing::debug!(
                target: "openjiuwen_memory::dreaming::planner",
                "dreaming is disabled; the sweep claims nothing"
            );
            return Ok(summary);
        }
        if limit == 0 {
            tracing::debug!(
                target: "openjiuwen_memory::dreaming::planner",
                "sweep limit is zero; the sweep claims nothing"
            );
            return Ok(summary);
        }
        for claimed in 0..limit {
            if ctx.check().is_err() {
                tracing::debug!(
                    target: "openjiuwen_memory::dreaming::planner",
                    claimed,
                    "the sweep was cancelled before the next claim"
                );
                break;
            }
            let Some(job) = self.repository.claim_next(now).await? else {
                break;
            };
            let stop = self.attempt(&job, ctx, &mut summary).await?;
            if stop {
                break;
            }
        }
        Ok(summary)
    }

    /// Runs one sweep with default [`CallOptions`] (no cancellation, no deadline).
    pub async fn run_once(&self, now: DateTime<Utc>, limit: usize) -> Result<DreamRunSummary> {
        self.run(now, limit, &CallOptions::new()).await
    }

    /// Runs one sweep with [`DreamConfig::batch_size`] as the claim limit.
    pub async fn run_due(&self, now: DateTime<Utc>, ctx: &CallOptions) -> Result<DreamRunSummary> {
        self.run(now, self.config.batch_size, ctx).await
    }

    /// Consolidates one claimed job and records the transition.
    ///
    /// Returns `true` when the sweep must stop (the run was cancelled). The body is the
    /// composition of [`Self::consolidate_step`] and [`Self::record_step`].
    async fn attempt(
        &self,
        job: &DreamJob,
        ctx: &CallOptions,
        summary: &mut DreamRunSummary,
    ) -> Result<bool> {
        let result = self.consolidate_step(job, ctx).await;
        self.record_step(job, result, summary).await
    }

    /// Runs the consolidator for one claimed job, applying the sweep's cancellation.
    async fn consolidate_step(&self, job: &DreamJob, ctx: &CallOptions) -> AttemptResult {
        match ctx.run(self.consolidator.consolidate(job, ctx)).await {
            Ok(Ok(outcome)) => AttemptResult::Done(outcome),
            // The consolidator's own failure, or cancellation/a deadline from
            // `CallOptions::run` (whose inner future was dropped, so there is no outcome
            // to inspect).
            Ok(Err(error)) | Err(error) => AttemptResult::Failed(error),
        }
    }

    /// Records one consolidation result and updates the summary.
    ///
    /// Returns `true` when the sweep must stop (the run was cancelled).
    async fn record_step(
        &self,
        job: &DreamJob,
        result: AttemptResult,
        summary: &mut DreamRunSummary,
    ) -> Result<bool> {
        match result {
            AttemptResult::Done(outcome) => self.record_success(job, outcome, summary).await,
            AttemptResult::Failed(error) => self.record_failure(job, &error, summary).await,
        }
    }

    /// Stores a consolidator outcome and updates the summary.
    async fn record_success(
        &self,
        job: &DreamJob,
        mut outcome: DreamOutcome,
        summary: &mut DreamRunSummary,
    ) -> Result<bool> {
        if outcome.job_id != job.id {
            tracing::warn!(
                target: "openjiuwen_memory::dreaming::planner",
                job_id = %job.id,
                outcome_job_id = %outcome.job_id,
                "consolidator reported a different job id; the planner is authoritative"
            );
            outcome.job_id = job.id.clone();
        }
        if !outcome.state.is_terminal() {
            tracing::warn!(
                target: "openjiuwen_memory::dreaming::planner",
                job_id = %job.id,
                state = outcome.state.as_str(),
                "{}",
                NON_TERMINAL_NOTE
            );
            return self
                .record_failure(
                    job,
                    &Error::validation(format!("{NON_TERMINAL_NOTE} `{}`", outcome.state)),
                    summary,
                )
                .await;
        }
        let stored = match self.repository.complete(&job.id, outcome.clone()).await {
            Ok(stored) => stored,
            Err(error) => {
                // The claim already committed and no transition did: requeue the job before
                // the error escapes, so it is not stranded in `running` forever.
                self.rescue_claimed(
                    job,
                    format!("the dream sweep could not record the transition: {error}"),
                )
                .await;
                return Err(error);
            }
        };
        tracing::debug!(
            target: "openjiuwen_memory::dreaming::planner",
            job_id = %job.id,
            state = stored.state.as_str(),
            "recorded a dream completion"
        );
        summary.record(outcome);
        Ok(false)
    }

    /// Records a failed or cancelled attempt and updates the summary.
    async fn record_failure(
        &self,
        job: &DreamJob,
        error: &Error,
        summary: &mut DreamRunSummary,
    ) -> Result<bool> {
        if matches!(error, Error::Cancelled) {
            let outcome = DreamOutcome::cancelled(job.id.clone());
            if let Err(storage_error) = self.repository.complete(&job.id, outcome.clone()).await {
                self.rescue_claimed(
                    job,
                    format!("the dream sweep could not record the cancellation: {storage_error}"),
                )
                .await;
                return Err(storage_error);
            }
            tracing::debug!(
                target: "openjiuwen_memory::dreaming::planner",
                job_id = %job.id,
                "cancelled the claimed dream job and stopped the sweep"
            );
            summary.record(outcome);
            return Ok(true);
        }
        let stored = match self.repository.fail(&job.id, &error.to_string()).await {
            Ok(stored) => stored,
            Err(storage_error) => {
                // A second, best-effort `fail` is the retry: a transient write failure is
                // repaired here, a permanent one is logged and still propagates. The note
                // keeps the attempt's own error, so the requeued job still says what went
                // wrong with the consolidation itself.
                self.rescue_claimed(
                    job,
                    format!(
                        "{error}; the dream sweep could not record the failed attempt: \
                         {storage_error}"
                    ),
                )
                .await;
                return Err(storage_error);
            }
        };
        let note = format!("{error}; {}", retry_note(&stored));
        tracing::debug!(
            target: "openjiuwen_memory::dreaming::planner",
            job_id = %job.id,
            state = stored.state.as_str(),
            attempts = stored.attempts,
            "recorded a failed dream attempt"
        );
        summary.record(DreamOutcome {
            job_id: job.id.clone(),
            state: DreamState::Failed,
            metrics: DreamMetrics::default(),
            notes: vec![note],
        });
        Ok(false)
    }
}

/// Returns a claimed job to the runnable path after a storage error, best effort.
///
/// A `complete`/`fail` write can fail *after* the claim committed, which leaves the row
/// `Running`. Nothing would ever claim it again: the claim predicate only matches
/// `state = 'pending'`, and the leased takeover needs a live lease row that the sweep
/// releases. Requeueing through [`DreamRepository::fail`] restores the job — `Pending`
/// while the attempt budget lasts, terminally `Failed` once it is spent — so a later sweep
/// picks it up instead of leaving it stuck.
///
/// The rescue is deliberately best effort: the caller still receives the original storage
/// error, and a repository that cannot write at all cannot be repaired from here. Both
/// outcomes are logged, and nothing from the rescue replaces the propagated error.
impl DreamPlanner {
    /// Requeues `job` with `note` as the recorded attempt error.
    ///
    /// `note` describes why the transition was lost; it becomes the job's `last_error` when
    /// the requeue succeeds, which is the only durable trace of the failure.
    async fn rescue_claimed(&self, job: &DreamJob, note: String) {
        match self.repository.fail(&job.id, &note).await {
            Ok(stored) => tracing::warn!(
                target: "openjiuwen_memory::dreaming::planner",
                job_id = %job.id,
                state = stored.state.as_str(),
                attempts = stored.attempts,
                %note,
                "the transition write failed; requeued the claimed job"
            ),
            Err(rescue_error) => tracing::error!(
                target: "openjiuwen_memory::dreaming::planner",
                job_id = %job.id,
                %note,
                %rescue_error,
                "the transition write failed and the job could not be requeued"
            ),
        }
    }
}

/// Describes what happened to the attempt budget after a failure: retry or terminal.
fn retry_note(job: &DreamJob) -> String {
    match job.state {
        DreamState::Pending => format!(
            "attempt {}/{} recorded and requeued; the retry needs a clock past its \
             `started_at` marker",
            job.attempts, job.max_attempts
        ),
        DreamState::Failed => format!(
            "attempt {}/{} exhausted the budget; the job is terminally failed",
            job.attempts, job.max_attempts
        ),
        other => format!("the job state is `{other}` after the failed attempt"),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::dreaming::{DreamFilter, DreamKind, DreamState, DreamTrigger};
    use crate::record::MemoryScope;
    use chrono::Duration;
    use parking_lot::Mutex;
    use std::collections::VecDeque;
    use tokio_util::sync::CancellationToken;

    /// One scripted consolidator behaviour.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Step {
        /// Succeed with zero metrics.
        Succeeded,
        /// Succeed carrying `metrics`.
        Metrics(DreamMetrics),
        /// Report the terminal skip outcome.
        Skipped,
        /// Fail with a generic memory error.
        Failed,
        /// Fail with [`Error::Cancelled`].
        Cancelled,
        /// Fail with [`Error::Timeout`].
        Timeout,
        /// Return a non-terminal (`running`) outcome.
        NonTerminal,
        /// Return a success attributed to another job id.
        WrongId,
        /// Block until released, so the test can cancel mid-consolidation.
        Block,
    }

    /// A consolidator that replays a script of [`Step`]s, then repeats `fallback`.
    #[derive(Debug)]
    struct ScriptedConsolidator {
        steps: Mutex<VecDeque<Step>>,
        fallback: Step,
        calls: Mutex<Vec<String>>,
        entered: tokio::sync::Notify,
        release: tokio::sync::Notify,
    }

    impl ScriptedConsolidator {
        fn new(steps: impl IntoIterator<Item = Step>) -> Self {
            Self {
                steps: Mutex::new(steps.into_iter().collect()),
                fallback: Step::Succeeded,
                calls: Mutex::new(Vec::new()),
                entered: tokio::sync::Notify::new(),
                release: tokio::sync::Notify::new(),
            }
        }

        fn with_fallback(steps: impl IntoIterator<Item = Step>, fallback: Step) -> Self {
            Self {
                fallback,
                ..Self::new(steps)
            }
        }

        fn calls(&self) -> Vec<String> {
            self.calls.lock().clone()
        }
    }

    #[async_trait]
    impl DreamConsolidator for ScriptedConsolidator {
        async fn consolidate(&self, job: &DreamJob, _ctx: &CallOptions) -> Result<DreamOutcome> {
            self.calls.lock().push(job.id.clone());
            let step = self.steps.lock().pop_front().unwrap_or(self.fallback);
            match step {
                Step::Succeeded => Ok(DreamOutcome::succeeded(
                    job.id.clone(),
                    DreamMetrics::default(),
                )),
                Step::Metrics(metrics) => Ok(DreamOutcome::succeeded(job.id.clone(), metrics)),
                Step::Skipped => Ok(DreamOutcome::skipped(job.id.clone(), "scripted skip")),
                Step::Failed => Err(Error::Memory("scripted failure".to_owned())),
                Step::Cancelled => Err(Error::Cancelled),
                Step::Timeout => Err(Error::Timeout("scripted timeout".to_owned())),
                Step::NonTerminal => Ok(DreamOutcome::new(job.id.clone(), DreamState::Running)),
                Step::WrongId => Ok(DreamOutcome::succeeded(
                    "somebody-else",
                    DreamMetrics::default(),
                )),
                Step::Block => {
                    self.entered.notify_one();
                    self.release.notified().await;
                    Ok(DreamOutcome::succeeded(
                        job.id.clone(),
                        DreamMetrics::default(),
                    ))
                }
            }
        }
    }

    fn ts(secs: i64) -> DateTime<Utc> {
        DateTime::from_timestamp(1_700_000_000, 0).unwrap() + Duration::seconds(secs)
    }

    fn job(id: &str, priority: i32, created_at: i64) -> DreamJob {
        DreamJob::new(
            MemoryScope::new("alice", "ws"),
            DreamKind::Consolidation,
            DreamTrigger::Manual,
            ts(created_at),
        )
        .with_id(id)
        .with_priority(priority)
    }

    fn planner(consolidator: Arc<ScriptedConsolidator>, config: DreamConfig) -> DreamPlanner {
        DreamPlanner::new(
            Arc::new(InMemoryDreamRepository::new()),
            consolidator,
            config,
        )
        .unwrap()
    }

    async fn stored(planner: &DreamPlanner, id: &str) -> DreamJob {
        planner
            .repository()
            .get(id)
            .await
            .unwrap()
            .expect("the job is stored")
    }

    #[tokio::test]
    async fn success_records_metrics_and_state() {
        let metrics = DreamMetrics {
            records_scanned: 4,
            records_merged: 2,
            records_deleted: 1,
            records_created: 3,
            tokens_saved: 512,
            duration_ms: 12,
        };
        let consolidator = Arc::new(ScriptedConsolidator::new([Step::Metrics(metrics)]));
        let planner = planner(Arc::clone(&consolidator), DreamConfig::default());
        planner
            .repository()
            .enqueue(job("job-1", 0, 0))
            .await
            .unwrap();

        let summary = planner.run_once(ts(10), 10).await.unwrap();
        assert_eq!(
            (summary.scanned, summary.succeeded, summary.failed),
            (1, 1, 0)
        );
        assert_eq!(summary.outcomes.len(), 1);
        assert_eq!(summary.outcomes[0].notes, Vec::<String>::new());
        let job = stored(&planner, "job-1").await;
        assert_eq!(job.state, DreamState::Succeeded);
        assert_eq!(job.metrics, metrics);
        assert_eq!(job.attempts, 1);
        assert!(job.last_error.is_none());
        assert_eq!(consolidator.calls(), vec!["job-1".to_owned()]);
    }

    #[tokio::test]
    async fn failures_retry_until_the_attempt_budget_is_exhausted() {
        let consolidator = Arc::new(ScriptedConsolidator::with_fallback([], Step::Failed));
        let planner = planner(
            Arc::clone(&consolidator),
            DreamConfig::default().with_max_attempts(2),
        );
        planner
            .repository()
            .enqueue(job("retry", 0, 0).with_max_attempts(2))
            .await
            .unwrap();

        let first = planner.run_once(ts(10), 10).await.unwrap();
        assert_eq!((first.scanned, first.failed), (1, 1));
        let requeued = stored(&planner, "retry").await;
        assert_eq!(requeued.state, DreamState::Pending);
        assert_eq!(requeued.attempts, 1);
        assert_eq!(
            requeued.last_error.as_deref(),
            Some("memory: scripted failure")
        );
        assert!(requeued.finished_at.is_none());
        assert!(first.outcomes[0].notes[0].contains("attempt 1/2"));
        assert!(first.outcomes[0].notes[0].contains("requeued"));

        // The retry marker is `started_at`, so the same clock is not eligible.
        let same_clock = planner.run_once(ts(10), 10).await.unwrap();
        assert!(same_clock.is_empty(), "same clock must not re-claim");
        assert_eq!(consolidator.calls().len(), 1);

        let second = planner.run_once(ts(11), 10).await.unwrap();
        assert_eq!((second.scanned, second.failed), (1, 1));
        let terminal = stored(&planner, "retry").await;
        assert_eq!(terminal.state, DreamState::Failed);
        assert_eq!(terminal.attempts, 2);
        assert!(terminal.finished_at.is_some());
        assert!(second.outcomes[0].notes[0].contains("exhausted"));

        let third = planner.run_once(ts(12), 10).await.unwrap();
        assert!(third.is_empty());
        assert_eq!(
            consolidator.calls().len(),
            2,
            "exactly max_attempts attempts"
        );
    }

    #[tokio::test]
    async fn pre_cancelled_sweep_claims_nothing() {
        let consolidator = Arc::new(ScriptedConsolidator::new([]));
        let planner = planner(Arc::clone(&consolidator), DreamConfig::default());
        planner
            .repository()
            .enqueue(job("job-1", 0, 0))
            .await
            .unwrap();

        let token = CancellationToken::new();
        token.cancel();
        let ctx = CallOptions::new().with_cancellation(token);
        let summary = planner.run_due(ts(10), &ctx).await.unwrap();
        assert!(summary.is_empty());
        assert_eq!(consolidator.calls(), Vec::<String>::new());
        let untouched = stored(&planner, "job-1").await;
        assert_eq!(untouched.state, DreamState::Pending);
        assert_eq!(untouched.attempts, 0);
    }

    #[tokio::test]
    async fn cancellation_mid_consolidation_completes_cancelled_and_stops() {
        let consolidator = Arc::new(ScriptedConsolidator::new([Step::Block]));
        let planner = Arc::new(planner(Arc::clone(&consolidator), DreamConfig::default()));
        planner
            .repository()
            .enqueue(job("first", 5, 0))
            .await
            .unwrap();
        planner
            .repository()
            .enqueue(job("second", 0, 1))
            .await
            .unwrap();

        let token = CancellationToken::new();
        let ctx = CallOptions::new().with_cancellation(token.clone());
        let running = Arc::clone(&planner);
        let handle = tokio::spawn(async move { running.run_due(ts(10), &ctx).await });

        consolidator.entered.notified().await;
        token.cancel();
        let summary = handle.await.unwrap().unwrap();
        assert_eq!(
            (summary.scanned, summary.cancelled, summary.succeeded),
            (1, 1, 0)
        );
        let cancelled = stored(&planner, "first").await;
        assert_eq!(cancelled.state, DreamState::Cancelled);
        assert!(cancelled.finished_at.is_some());
        let untouched = stored(&planner, "second").await;
        assert_eq!(untouched.state, DreamState::Pending, "the loop stopped");
        assert_eq!(untouched.attempts, 0);
        assert_eq!(consolidator.calls(), vec!["first".to_owned()]);
        consolidator.release.notify_one();
    }

    #[tokio::test]
    async fn consolidator_cancellation_error_also_stops_the_sweep() {
        let consolidator = Arc::new(ScriptedConsolidator::new([Step::Cancelled]));
        let planner = planner(Arc::clone(&consolidator), DreamConfig::default());
        planner
            .repository()
            .enqueue(job("first", 5, 0))
            .await
            .unwrap();
        planner
            .repository()
            .enqueue(job("second", 0, 1))
            .await
            .unwrap();

        let summary = planner.run_once(ts(10), 10).await.unwrap();
        assert_eq!((summary.scanned, summary.cancelled), (1, 1));
        assert_eq!(stored(&planner, "first").await.state, DreamState::Cancelled);
        assert_eq!(stored(&planner, "second").await.state, DreamState::Pending);
        assert_eq!(consolidator.calls(), vec!["first".to_owned()]);
    }

    #[tokio::test]
    async fn timeout_is_a_failed_attempt_not_a_cancellation() {
        let consolidator = Arc::new(ScriptedConsolidator::new([Step::Timeout]));
        let planner = planner(Arc::clone(&consolidator), DreamConfig::default());
        planner
            .repository()
            .enqueue(job("job-1", 0, 0))
            .await
            .unwrap();

        let summary = planner.run_once(ts(10), 10).await.unwrap();
        assert_eq!((summary.scanned, summary.failed), (1, 1));
        assert_eq!(summary.cancelled, 0);
        let job = stored(&planner, "job-1").await;
        assert_eq!(job.state, DreamState::Pending);
        assert_eq!(job.attempts, 1);
        assert!(job
            .last_error
            .as_deref()
            .unwrap_or_default()
            .contains("scripted timeout"));
    }

    #[tokio::test]
    async fn non_terminal_outcome_is_treated_as_a_failed_attempt() {
        let consolidator = Arc::new(ScriptedConsolidator::new([Step::NonTerminal]));
        let planner = planner(
            Arc::clone(&consolidator),
            DreamConfig::default().with_max_attempts(1),
        );
        planner
            .repository()
            .enqueue(job("job-1", 0, 0).with_max_attempts(1))
            .await
            .unwrap();

        let summary = planner.run_once(ts(10), 10).await.unwrap();
        assert_eq!(
            (summary.scanned, summary.failed, summary.succeeded),
            (1, 1, 0)
        );
        assert!(summary.outcomes[0].notes[0].contains("non-terminal"));
        let job = stored(&planner, "job-1").await;
        assert_eq!(job.state, DreamState::Failed);
        assert_eq!(job.attempts, 1);
        assert!(job
            .last_error
            .as_deref()
            .unwrap_or_default()
            .contains("non-terminal"));
    }

    #[tokio::test]
    async fn mismatched_outcome_job_id_is_overridden() {
        let consolidator = Arc::new(ScriptedConsolidator::new([Step::WrongId]));
        let planner = planner(Arc::clone(&consolidator), DreamConfig::default());
        planner
            .repository()
            .enqueue(job("job-1", 0, 0))
            .await
            .unwrap();

        let summary = planner.run_once(ts(10), 10).await.unwrap();
        assert_eq!((summary.scanned, summary.succeeded), (1, 1));
        assert_eq!(summary.outcomes[0].job_id, "job-1");
        assert_eq!(stored(&planner, "job-1").await.state, DreamState::Succeeded);
    }

    #[tokio::test]
    async fn consolidator_skip_outcome_is_terminal_and_not_retried() {
        let consolidator = Arc::new(ScriptedConsolidator::new([Step::Skipped]));
        let planner = planner(Arc::clone(&consolidator), DreamConfig::default());
        planner
            .repository()
            .enqueue(job("job-1", 0, 0))
            .await
            .unwrap();

        let summary = planner.run_once(ts(10), 10).await.unwrap();
        assert_eq!((summary.scanned, summary.skipped), (1, 1));
        assert_eq!(summary.failed, 0);
        let job = stored(&planner, "job-1").await;
        assert_eq!(job.state, DreamState::Skipped);
        assert!(job.metrics.is_empty());
        assert!(job.last_error.is_none());
        assert!(planner.run_once(ts(11), 10).await.unwrap().is_empty());
        assert_eq!(consolidator.calls().len(), 1);
    }

    #[tokio::test]
    async fn noop_consolidator_skips_without_retrying() {
        let planner =
            DreamPlanner::with_in_memory(Arc::new(NoopConsolidator), DreamConfig::default())
                .unwrap();
        planner
            .repository()
            .enqueue(job("job-1", 0, 0))
            .await
            .unwrap();

        let summary = planner.run_once(ts(10), 10).await.unwrap();
        assert_eq!((summary.scanned, summary.skipped), (1, 1));
        assert_eq!(
            summary.outcomes[0].notes,
            vec!["no consolidator configured".to_owned()]
        );
        let job = stored(&planner, "job-1").await;
        assert_eq!(job.state, DreamState::Skipped);
        assert!(job.metrics.is_empty());
        assert_eq!(job.attempts, 1);

        let again = planner.run_once(ts(11), 10).await.unwrap();
        assert!(again.is_empty(), "a skipped job is terminal");
        assert_eq!(stored(&planner, "job-1").await.attempts, 1);
    }

    #[tokio::test]
    async fn disabled_config_claims_nothing() {
        let consolidator = Arc::new(ScriptedConsolidator::new([]));
        let planner = planner(
            Arc::clone(&consolidator),
            DreamConfig::default().with_enabled(false),
        );
        planner
            .repository()
            .enqueue(job("job-1", 0, 0))
            .await
            .unwrap();

        let summary = planner.run_once(ts(10), 10).await.unwrap();
        assert!(summary.is_empty());
        assert_eq!(consolidator.calls(), Vec::<String>::new());
        assert_eq!(stored(&planner, "job-1").await.state, DreamState::Pending);
    }

    #[tokio::test]
    async fn zero_limit_claims_nothing() {
        let consolidator = Arc::new(ScriptedConsolidator::new([]));
        let planner = planner(Arc::clone(&consolidator), DreamConfig::default());
        planner
            .repository()
            .enqueue(job("job-1", 0, 0))
            .await
            .unwrap();

        let summary = planner.run_once(ts(10), 0).await.unwrap();
        assert!(summary.is_empty());
        assert_eq!(consolidator.calls(), Vec::<String>::new());
    }

    #[tokio::test]
    async fn limit_and_priority_order_are_honoured() {
        let consolidator = Arc::new(ScriptedConsolidator::new([]));
        let planner = planner(Arc::clone(&consolidator), DreamConfig::default());
        for job in [job("low", 0, 1), job("high", 5, 5), job("high-older", 5, 2)] {
            planner.repository().enqueue(job).await.unwrap();
        }

        let summary = planner.run_once(ts(10), 2).await.unwrap();
        assert_eq!((summary.scanned, summary.succeeded), (2, 2));
        assert_eq!(
            consolidator.calls(),
            vec!["high-older".to_owned(), "high".to_owned()]
        );

        let rest = planner.run_once(ts(20), 10).await.unwrap();
        assert_eq!(rest.scanned, 1);
        assert_eq!(consolidator.calls().last().unwrap(), "low");
    }

    #[tokio::test]
    async fn run_due_uses_the_configured_batch_size() {
        let consolidator = Arc::new(ScriptedConsolidator::new([]));
        let planner = planner(
            Arc::clone(&consolidator),
            DreamConfig::default().with_batch_size(2),
        );
        for index in 0..3 {
            planner
                .repository()
                .enqueue(job(&format!("job-{index}"), 0, index))
                .await
                .unwrap();
        }

        let summary = planner.run_due(ts(10), &CallOptions::new()).await.unwrap();
        assert_eq!(summary.scanned, 2);
        assert_eq!(consolidator.calls().len(), 2);
        assert_eq!(
            planner
                .repository()
                .list(&DreamFilter::new().with_state(DreamState::Pending))
                .await
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    fn new_rejects_an_invalid_config() {
        let consolidator: Arc<dyn DreamConsolidator> = Arc::new(NoopConsolidator);
        let error =
            DreamPlanner::with_in_memory(consolidator, DreamConfig::default().with_batch_size(0))
                .unwrap_err();
        assert!(matches!(error, Error::Validation(_)), "{error:?}");
        assert!(error.to_string().contains("batch_size"));
    }

    #[test]
    fn debug_prints_the_configuration() {
        let planner =
            DreamPlanner::with_in_memory(Arc::new(NoopConsolidator), DreamConfig::default())
                .unwrap();
        let rendered = format!("{planner:?}");
        assert!(rendered.contains("DreamPlanner"));
        assert!(rendered.contains("batch_size"));
    }
}
