//! Integration coverage for the dreaming job queue (`openjiuwen_memory::dreaming`).
//!
//! The `dreaming` submodules have their own unit tests for the data model, the
//! repository state machine, and scripted consolidator steps; this suite drives the
//! public surface the way an embedding application does: build a [`DreamPlanner`] over a
//! repository, enqueue jobs, and sweep with an explicit clock.
//!
//! Every test passes `now` in (a fixed epoch plus an offset) so nothing depends on the
//! wall clock, and the consolidators defined here are deterministic stand-ins: one
//! succeeds with non-zero metrics, one fails on every call, one blocks until the sweep
//! is cancelled. The SQLite test uses its own [`tempfile::TempDir`] and reopens the same
//! database file to prove the queue is durable.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use openjiuwen_common::error::{Error, Result};
use openjiuwen_memory::dreaming::{
    DreamConfig, DreamConsolidator, DreamFilter, DreamJob, DreamKind, DreamLease,
    DreamLeaseOptions, DreamMetrics, DreamOutcome, DreamPlanner, DreamRepository, DreamState,
    DreamTrigger, InMemoryDreamRepository, LeaseRequest, LeaseStatus, LeasedJob, NoopConsolidator,
};
use openjiuwen_memory::record::{CallOptions, MemoryScope};
use tokio_util::sync::CancellationToken;

#[cfg(feature = "sqlite")]
use openjiuwen_memory::dreaming::SqliteDreamRepository;

/// A fixed instant `offset` seconds after the shared test epoch.
fn ts(offset: i64) -> DateTime<Utc> {
    DateTime::from_timestamp(1_700_000_000 + offset, 0).expect("a valid timestamp")
}

/// The scope every job in this suite consolidates.
fn scope() -> MemoryScope {
    MemoryScope::new("alice", "workspace")
}

/// The non-zero counters [`SucceedingConsolidator`] reports.
fn success_metrics() -> DreamMetrics {
    DreamMetrics {
        records_scanned: 6,
        records_merged: 2,
        records_deleted: 1,
        records_created: 1,
        tokens_saved: 128,
        duration_ms: 7,
    }
}

/// Builds a pending job with `id` and `created_at`, default kind and trigger.
fn job(id: &str, created_at: DateTime<Utc>) -> DreamJob {
    DreamJob::new(
        scope(),
        DreamKind::Consolidation,
        DreamTrigger::Manual,
        created_at,
    )
    .with_id(id)
}

/// A consolidator that reports terminal success with fixed metrics and records the job
/// ids it was handed, so a test can assert both call count and claim order.
struct SucceedingConsolidator {
    /// Counters copied onto every successful outcome.
    metrics: DreamMetrics,
    /// Job ids in call order.
    claims: parking_lot::Mutex<Vec<String>>,
}

impl SucceedingConsolidator {
    /// Builds the consolidator with `metrics` as its reported counters.
    fn new(metrics: DreamMetrics) -> Self {
        Self {
            metrics,
            claims: parking_lot::Mutex::new(Vec::new()),
        }
    }

    /// The job ids this consolidator was called with, in call order.
    fn claims(&self) -> Vec<String> {
        self.claims.lock().clone()
    }
}

#[async_trait]
impl DreamConsolidator for SucceedingConsolidator {
    async fn consolidate(&self, job: &DreamJob, ctx: &CallOptions) -> Result<DreamOutcome> {
        self.claims.lock().push(job.id.clone());
        ctx.check()?;
        Ok(DreamOutcome::succeeded(job.id.clone(), self.metrics)
            .with_note("consolidated by the test consolidator"))
    }
}

/// A consolidator that fails every attempt, so the queue's retry budget is exercised.
struct FailingConsolidator {
    /// Message carried by the returned error and, through the planner, by `last_error`.
    message: &'static str,
    /// Number of `consolidate` calls.
    calls: AtomicUsize,
}

impl FailingConsolidator {
    /// Builds the consolidator with `message` as its failure text.
    fn new(message: &'static str) -> Self {
        Self {
            message,
            calls: AtomicUsize::new(0),
        }
    }

    /// Number of times `consolidate` was called.
    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl DreamConsolidator for FailingConsolidator {
    async fn consolidate(&self, _job: &DreamJob, _ctx: &CallOptions) -> Result<DreamOutcome> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Err(Error::memory(self.message))
    }
}

/// A consolidator that signals that it started and then never finishes, so the test can
/// cancel the sweep mid-consolidation.
struct BlockingConsolidator {
    /// Notified once `consolidate` has been entered.
    entered: tokio::sync::Notify,
}

impl BlockingConsolidator {
    /// Builds the consolidator.
    fn new() -> Self {
        Self {
            entered: tokio::sync::Notify::new(),
        }
    }

    /// Resolves once the consolidator has been entered at least once.
    async fn wait_entered(&self) {
        self.entered.notified().await;
    }
}

#[async_trait]
impl DreamConsolidator for BlockingConsolidator {
    async fn consolidate(&self, _job: &DreamJob, _ctx: &CallOptions) -> Result<DreamOutcome> {
        self.entered.notify_one();
        std::future::pending::<Result<DreamOutcome>>().await
    }
}

#[tokio::test]
async fn succeeding_consolidator_completes_the_job_and_the_queue_drains() {
    let consolidator = Arc::new(SucceedingConsolidator::new(success_metrics()));
    let shared: Arc<dyn DreamConsolidator> = consolidator.clone();
    let planner =
        DreamPlanner::with_in_memory(shared, DreamConfig::default()).expect("a valid planner");
    planner
        .repository()
        .enqueue(job("succeed-1", ts(0)))
        .await
        .expect("enqueue");

    let summary = planner.run_once(ts(10), 10).await.expect("sweep");
    assert_eq!((summary.scanned, summary.succeeded), (1, 1));
    assert_eq!((summary.failed, summary.skipped), (0, 0));
    assert_eq!(summary.cancelled, 0);
    assert_eq!(summary.outcomes.len(), 1);
    let outcome = &summary.outcomes[0];
    assert_eq!(outcome.job_id, "succeed-1");
    assert_eq!(outcome.state, DreamState::Succeeded);
    assert_eq!(outcome.metrics, success_metrics());
    assert_eq!(
        outcome.notes,
        vec!["consolidated by the test consolidator".to_owned()]
    );

    let stored = planner
        .repository()
        .get("succeed-1")
        .await
        .expect("get")
        .expect("the enqueued job");
    assert_eq!(stored.state, DreamState::Succeeded);
    assert_eq!(stored.attempts, 1);
    assert_eq!(stored.metrics, success_metrics());
    assert!(stored.finished_at.is_some(), "a terminal job is finished");
    assert!(
        stored.last_error.is_none(),
        "completion clears the previous attempt's error"
    );
    assert_eq!(consolidator.claims(), vec!["succeed-1".to_owned()]);

    let second = planner.run_once(ts(20), 10).await.expect("second sweep");
    assert!(second.is_empty(), "a finished job is never claimed again");
    assert_eq!(consolidator.claims().len(), 1);
}

#[tokio::test]
async fn failing_consolidator_exhausts_the_attempt_budget() {
    let consolidator = Arc::new(FailingConsolidator::new("consolidation exploded"));
    let shared: Arc<dyn DreamConsolidator> = consolidator.clone();
    // `DreamConfig::max_attempts` seeds new jobs; the queue itself reads the per-job
    // budget, so the job below carries the same small value explicitly.
    let planner = DreamPlanner::with_in_memory(shared, DreamConfig::default().with_max_attempts(2))
        .expect("a valid planner");
    planner
        .repository()
        .enqueue(job("fail-1", ts(0)).with_max_attempts(2))
        .await
        .expect("enqueue");

    let first = planner.run_once(ts(10), 10).await.expect("first sweep");
    assert_eq!((first.scanned, first.failed), (1, 1));
    assert_eq!(first.outcomes[0].state, DreamState::Failed);
    assert!(
        first.outcomes[0].notes[0].contains("attempt 1/2 recorded and requeued"),
        "the note says the attempt was requeued: {:?}",
        first.outcomes[0].notes
    );
    let requeued = planner
        .repository()
        .get("fail-1")
        .await
        .expect("get")
        .expect("the enqueued job");
    assert_eq!(requeued.state, DreamState::Pending);
    assert_eq!(requeued.attempts, 1);
    assert!(requeued.finished_at.is_none());
    assert!(
        requeued
            .last_error
            .as_deref()
            .is_some_and(|error| error.contains("consolidation exploded")),
        "the failed attempt's message is recorded: {:?}",
        requeued.last_error
    );
    assert!(
        requeued.started_at.is_some(),
        "the retry marker is the claim time"
    );

    let second = planner.run_once(ts(11), 10).await.expect("second sweep");
    assert_eq!((second.scanned, second.failed), (1, 1));
    assert!(
        second.outcomes[0].notes[0].contains("attempt 2/2 exhausted the budget"),
        "the note says the budget is exhausted: {:?}",
        second.outcomes[0].notes
    );
    let failed = planner
        .repository()
        .get("fail-1")
        .await
        .expect("get")
        .expect("the enqueued job");
    assert_eq!(failed.state, DreamState::Failed);
    assert_eq!(failed.attempts, 2);
    assert!(failed.finished_at.is_some());
    assert!(
        failed
            .last_error
            .as_deref()
            .is_some_and(|error| error.contains("consolidation exploded")),
        "the terminal job keeps the last error: {:?}",
        failed.last_error
    );

    let third = planner.run_once(ts(12), 10).await.expect("third sweep");
    assert!(third.is_empty(), "a terminally failed job is not retried");
    assert_eq!(consolidator.calls(), 2);
    let counts = planner
        .repository()
        .count_by_state()
        .await
        .expect("state counts");
    assert_eq!(counts[&DreamState::Failed], 1);
    assert_eq!(counts[&DreamState::Pending], 0);
}

#[tokio::test]
async fn noop_consolidator_skips_every_job_and_drains_the_queue() {
    let planner = DreamPlanner::with_in_memory(
        Arc::new(NoopConsolidator),
        DreamConfig::default().with_batch_size(2),
    )
    .expect("a valid planner");
    for id in ["noop-1", "noop-2"] {
        planner
            .repository()
            .enqueue(job(id, ts(0)))
            .await
            .expect("enqueue");
    }

    let summary = planner
        .run_due(ts(10), &CallOptions::new())
        .await
        .expect("sweep");
    assert_eq!((summary.scanned, summary.skipped), (2, 2));
    for outcome in &summary.outcomes {
        assert_eq!(outcome.state, DreamState::Skipped);
        assert!(outcome.metrics.is_empty(), "a skip does no work");
        assert_eq!(outcome.notes, vec!["no consolidator configured".to_owned()]);
    }
    for id in ["noop-1", "noop-2"] {
        let stored = planner
            .repository()
            .get(id)
            .await
            .expect("get")
            .expect("the enqueued job");
        assert_eq!(stored.state, DreamState::Skipped);
        assert!(stored.finished_at.is_some());
        assert!(stored.is_terminal());
    }
    assert!(
        planner
            .repository()
            .list(&DreamFilter::new().with_state(DreamState::Pending))
            .await
            .expect("list pending")
            .is_empty(),
        "skips drain the queue instead of accumulating retries"
    );
    assert!(planner
        .run_due(ts(20), &CallOptions::new())
        .await
        .expect("second sweep")
        .is_empty());
}

#[tokio::test]
async fn pre_cancelled_sweep_returns_an_empty_summary_and_leaves_the_job_claimable() {
    let consolidator = Arc::new(SucceedingConsolidator::new(success_metrics()));
    let shared: Arc<dyn DreamConsolidator> = consolidator.clone();
    let planner =
        DreamPlanner::with_in_memory(shared, DreamConfig::default()).expect("a valid planner");
    planner
        .repository()
        .enqueue(job("cancel-pre", ts(0)))
        .await
        .expect("enqueue");

    let token = CancellationToken::new();
    token.cancel();
    let ctx = CallOptions::new().with_cancellation(token);
    // The planner checks cancellation *before* every claim and breaks the loop, so a
    // pre-cancelled sweep is an ordinary empty summary rather than an error, and the job
    // is never moved to `Running`.
    let summary = planner
        .run_due(ts(10), &ctx)
        .await
        .expect("the sweep reports no error");
    assert!(summary.is_empty());
    assert_eq!(consolidator.claims(), Vec::<String>::new());

    let stored = planner
        .repository()
        .get("cancel-pre")
        .await
        .expect("get")
        .expect("the enqueued job");
    assert_eq!(stored.state, DreamState::Pending);
    assert_eq!(stored.attempts, 0);
    assert!(stored.started_at.is_none());

    let later = planner
        .run_once(ts(10), 10)
        .await
        .expect("uncancelled sweep");
    assert_eq!((later.scanned, later.succeeded), (1, 1));
    assert_eq!(
        planner
            .repository()
            .get("cancel-pre")
            .await
            .expect("get")
            .expect("the enqueued job")
            .state,
        DreamState::Succeeded
    );
}

#[tokio::test]
async fn cancellation_mid_consolidation_completes_the_job_as_cancelled() {
    let consolidator = Arc::new(BlockingConsolidator::new());
    let shared: Arc<dyn DreamConsolidator> = consolidator.clone();
    let planner =
        DreamPlanner::with_in_memory(shared, DreamConfig::default()).expect("a valid planner");
    planner
        .repository()
        .enqueue(job("cancel-flight", ts(0)))
        .await
        .expect("enqueue");

    let token = CancellationToken::new();
    let ctx = CallOptions::new().with_cancellation(token.clone());
    let sweep = planner.run_due(ts(10), &ctx);
    let canceller = async move {
        consolidator.wait_entered().await;
        token.cancel();
    };
    let (summary, ()) = tokio::time::timeout(Duration::from_secs(5), async move {
        tokio::join!(sweep, canceller)
    })
    .await
    .expect("the cancelled sweep must finish");
    let summary = summary.expect("a cancelled sweep still reports its outcomes");
    assert_eq!((summary.scanned, summary.cancelled), (1, 1));
    assert_eq!(summary.outcomes[0].job_id, "cancel-flight");
    assert_eq!(summary.outcomes[0].state, DreamState::Cancelled);

    let stored = planner
        .repository()
        .get("cancel-flight")
        .await
        .expect("get")
        .expect("the enqueued job");
    assert_eq!(stored.state, DreamState::Cancelled);
    assert!(stored.finished_at.is_some(), "cancellation is terminal");
    assert!(
        planner
            .run_once(ts(20), 10)
            .await
            .expect("later sweep")
            .is_empty(),
        "a cancelled job is not claimed again"
    );
}

#[tokio::test]
async fn run_once_honours_the_claim_limit_and_the_documented_ordering() {
    let consolidator = Arc::new(SucceedingConsolidator::new(success_metrics()));
    let shared: Arc<dyn DreamConsolidator> = consolidator.clone();
    let planner =
        DreamPlanner::with_in_memory(shared, DreamConfig::default()).expect("a valid planner");
    for job in [
        job("limit-low", ts(0)).with_priority(0),
        job("limit-high", ts(0)).with_priority(5),
        job("limit-mid", ts(0)).with_priority(1),
    ] {
        planner.repository().enqueue(job).await.expect("enqueue");
    }

    // The queue head is ordered `priority DESC, created_at ASC, id ASC`.
    let first = planner.run_once(ts(10), 2).await.expect("limited sweep");
    assert_eq!((first.scanned, first.succeeded), (2, 2));
    assert_eq!(
        consolidator.claims(),
        vec!["limit-high".to_owned(), "limit-mid".to_owned()]
    );
    assert_eq!(
        planner
            .repository()
            .list(&DreamFilter::new().with_state(DreamState::Pending))
            .await
            .expect("list pending")
            .len(),
        1
    );

    let rest = planner.run_once(ts(20), 10).await.expect("remainder sweep");
    assert_eq!((rest.scanned, rest.succeeded), (1, 1));
    assert_eq!(
        consolidator.claims().last().map(String::as_str),
        Some("limit-low")
    );
}

#[tokio::test]
async fn a_requeued_job_waits_for_a_later_clock() {
    let consolidator = Arc::new(FailingConsolidator::new("not today"));
    let shared: Arc<dyn DreamConsolidator> = consolidator.clone();
    let planner =
        DreamPlanner::with_in_memory(shared, DreamConfig::default()).expect("a valid planner");
    planner
        .repository()
        .enqueue(job("gate-1", ts(0)).with_max_attempts(3))
        .await
        .expect("enqueue");

    let first = planner.run_once(ts(10), 10).await.expect("first sweep");
    assert_eq!((first.scanned, first.failed), (1, 1));

    // `started_at` is the retry marker: the same clock cannot claim the requeued job.
    let same_clock = planner
        .run_once(ts(10), 10)
        .await
        .expect("same-clock sweep");
    assert!(same_clock.is_empty(), "a requeued job needs a later clock");
    assert_eq!(consolidator.calls(), 1);

    let later = planner.run_once(ts(11), 10).await.expect("later sweep");
    assert_eq!((later.scanned, later.failed), (1, 1));
    assert_eq!(consolidator.calls(), 2);
    let stored = planner
        .repository()
        .get("gate-1")
        .await
        .expect("get")
        .expect("the enqueued job");
    assert_eq!(stored.state, DreamState::Pending);
    assert_eq!(stored.attempts, 2);
}

#[cfg(feature = "sqlite")]
#[tokio::test]
async fn sqlite_repository_keeps_jobs_across_reopens() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let url = format!("sqlite:{}", dir.path().join("dreams.db").display());
    let enqueued = DreamJob::new(scope(), DreamKind::Dedupe, DreamTrigger::Schedule, ts(0))
        .with_id("sqlite-1");

    {
        let repository = Arc::new(
            SqliteDreamRepository::connect(&url)
                .await
                .expect("open the database"),
        );
        repository.enqueue(enqueued.clone()).await.expect("enqueue");
    }

    let repository: Arc<dyn DreamRepository> = Arc::new(
        SqliteDreamRepository::connect(&url)
            .await
            .expect("reopen the database"),
    );
    let pending = repository
        .get("sqlite-1")
        .await
        .expect("get")
        .expect("the persisted job");
    assert_eq!(pending.state, DreamState::Pending);
    assert_eq!(pending.kind, DreamKind::Dedupe);
    assert_eq!(pending.trigger, DreamTrigger::Schedule);
    assert_eq!(pending.scope, scope());
    assert_eq!(pending.attempts, 0);

    let consolidator = Arc::new(SucceedingConsolidator::new(success_metrics()));
    let shared: Arc<dyn DreamConsolidator> = consolidator.clone();
    let planner = DreamPlanner::new(repository, shared, DreamConfig::default()).expect("planner");
    let summary = planner.run_once(ts(10), 10).await.expect("sweep");
    assert_eq!((summary.scanned, summary.succeeded), (1, 1));
    assert_eq!(summary.outcomes[0].job_id, "sqlite-1");
    assert!(planner
        .run_once(ts(20), 10)
        .await
        .expect("second sweep")
        .is_empty());

    // A third open proves the terminal state itself is durable.
    let verified = SqliteDreamRepository::connect(&url)
        .await
        .expect("final reopen");
    let stored = verified
        .get("sqlite-1")
        .await
        .expect("get")
        .expect("the persisted job");
    assert_eq!(stored.state, DreamState::Succeeded);
    assert_eq!(stored.attempts, 1);
    assert_eq!(stored.metrics, success_metrics());
    assert!(stored.finished_at.is_some());
    assert_eq!(
        verified.count_by_state().await.expect("state counts")[&DreamState::Succeeded],
        1
    );
}

/// An [`InMemoryDreamRepository`] whose `complete` write fails a scripted number of times.
///
/// A completion write that fails *after* the claim committed is precisely the storage
/// failure that used to leave a job `running` for good, and it cannot be produced through
/// the public repository surface.
#[derive(Debug)]
struct FlakyCompletionRepository {
    /// The queue the wrapper delegates to.
    inner: InMemoryDreamRepository,
    /// How many upcoming `complete` calls must fail.
    remaining_failures: parking_lot::Mutex<usize>,
}

impl FlakyCompletionRepository {
    /// Builds a wrapper around an empty queue that fails no writes.
    fn new() -> Self {
        Self {
            inner: InMemoryDreamRepository::new(),
            remaining_failures: parking_lot::Mutex::new(0),
        }
    }

    /// Makes the next `count` completion writes fail.
    fn fail_next_completions(&self, count: usize) {
        *self.remaining_failures.lock() = count;
    }

    /// Consumes one scripted failure, returning `true` when this call must fail.
    fn take_failure(&self) -> bool {
        let mut remaining = self.remaining_failures.lock();
        if *remaining == 0 {
            return false;
        }
        *remaining -= 1;
        true
    }
}

#[async_trait]
impl DreamRepository for FlakyCompletionRepository {
    async fn enqueue(&self, job: DreamJob) -> Result<DreamJob> {
        self.inner.enqueue(job).await
    }

    async fn claim_next(&self, now: DateTime<Utc>) -> Result<Option<DreamJob>> {
        self.inner.claim_next(now).await
    }

    async fn claim_next_leased(
        &self,
        now: DateTime<Utc>,
        request: &LeaseRequest,
    ) -> Result<Option<LeasedJob>> {
        self.inner.claim_next_leased(now, request).await
    }

    async fn renew_lease(
        &self,
        job_id: &str,
        owner: &str,
        now: DateTime<Utc>,
        ttl: chrono::Duration,
    ) -> Result<LeaseStatus> {
        self.inner.renew_lease(job_id, owner, now, ttl).await
    }

    async fn release_lease(&self, job_id: &str, owner: &str) -> Result<bool> {
        self.inner.release_lease(job_id, owner).await
    }

    async fn lease(&self, job_id: &str) -> Result<Option<DreamLease>> {
        self.inner.lease(job_id).await
    }

    async fn complete(&self, job_id: &str, outcome: DreamOutcome) -> Result<DreamJob> {
        if self.take_failure() {
            return Err(Error::memory("the completion write failed"));
        }
        self.inner.complete(job_id, outcome).await
    }

    async fn fail(&self, job_id: &str, error: &str) -> Result<DreamJob> {
        self.inner.fail(job_id, error).await
    }

    async fn get(&self, job_id: &str) -> Result<Option<DreamJob>> {
        self.inner.get(job_id).await
    }

    async fn list(&self, filter: &DreamFilter) -> Result<Vec<DreamJob>> {
        self.inner.list(filter).await
    }

    async fn count_by_state(&self) -> Result<std::collections::BTreeMap<DreamState, usize>> {
        self.inner.count_by_state().await
    }

    async fn delete_finished(&self, older_than: DateTime<Utc>) -> Result<usize> {
        self.inner.delete_finished(older_than).await
    }
}

/// The leased options every lease-aware test in this suite sweeps with.
fn lease_options(owner: &str) -> DreamLeaseOptions {
    DreamLeaseOptions::new(
        owner,
        chrono::Duration::seconds(30),
        Duration::from_secs(10),
    )
}

#[tokio::test]
async fn a_failed_completion_write_requeues_the_job_instead_of_stranding_it() {
    let repository = Arc::new(FlakyCompletionRepository::new());
    repository.fail_next_completions(1);
    let consolidator = Arc::new(SucceedingConsolidator::new(success_metrics()));
    let planner = DreamPlanner::new(
        repository.clone(),
        consolidator.clone(),
        DreamConfig::default(),
    )
    .expect("a valid planner");
    repository
        .enqueue(job("stranded-1", ts(0)).with_max_attempts(3))
        .await
        .expect("enqueue");

    let error = planner
        .run_once(ts(10), 10)
        .await
        .expect_err("the completion write failed");
    assert!(matches!(error, Error::Memory(_)), "{error:?}");

    // The claim committed and the completion did not: the job must be claimable again
    // rather than parked in `running`, which no sweep can select.
    let rescued = repository
        .get("stranded-1")
        .await
        .expect("get")
        .expect("the enqueued job");
    assert_eq!(
        rescued.state,
        DreamState::Pending,
        "a failed transition write requeues the job"
    );
    assert_eq!(rescued.attempts, 1);
    assert!(rescued.finished_at.is_none());
    assert!(
        rescued
            .last_error
            .as_deref()
            .is_some_and(|error| error.contains("could not record the transition")),
        "the rescue records why the transition was lost: {:?}",
        rescued.last_error
    );

    // A later sweep, past the retry marker, claims and completes it.
    let summary = planner.run_once(ts(11), 10).await.expect("recovery sweep");
    assert_eq!((summary.scanned, summary.succeeded), (1, 1));
    let completed = repository
        .get("stranded-1")
        .await
        .expect("get")
        .expect("the enqueued job");
    assert_eq!(completed.state, DreamState::Succeeded);
    assert_eq!(completed.attempts, 2);
}

#[tokio::test]
async fn a_failed_completion_write_with_a_spent_budget_fails_the_job_explicitly() {
    let repository = Arc::new(FlakyCompletionRepository::new());
    repository.fail_next_completions(1);
    let consolidator = Arc::new(SucceedingConsolidator::new(success_metrics()));
    let planner = DreamPlanner::new(repository.clone(), consolidator, DreamConfig::default())
        .expect("a valid planner");
    repository
        .enqueue(job("spent-1", ts(0)).with_max_attempts(1))
        .await
        .expect("enqueue");

    let _ = planner
        .run_once(ts(10), 10)
        .await
        .expect_err("the completion write failed");

    let rescued = repository
        .get("spent-1")
        .await
        .expect("get")
        .expect("the enqueued job");
    assert_eq!(
        rescued.state,
        DreamState::Failed,
        "the rescue spends the exhausted budget instead of leaving the job running"
    );
    assert!(rescued.finished_at.is_some());
    assert!(planner
        .run_once(ts(11), 10)
        .await
        .expect("later sweep")
        .is_empty());
}

#[tokio::test]
async fn a_leased_sweep_requeues_the_job_and_releases_the_lease_after_a_failed_transition() {
    let repository = Arc::new(FlakyCompletionRepository::new());
    repository.fail_next_completions(1);
    let consolidator = Arc::new(SucceedingConsolidator::new(success_metrics()));
    let planner = DreamPlanner::new(
        repository.clone(),
        consolidator.clone(),
        DreamConfig::default(),
    )
    .expect("a valid planner");
    repository
        .enqueue(job("leased-1", ts(0)).with_max_attempts(3))
        .await
        .expect("enqueue");

    let error = planner
        .run_leased(ts(10), 10, &CallOptions::new(), &lease_options("worker-a"))
        .await
        .expect_err("the completion write failed");
    assert!(matches!(error, Error::Memory(_)), "{error:?}");

    let rescued = repository
        .get("leased-1")
        .await
        .expect("get")
        .expect("the enqueued job");
    assert_eq!(rescued.state, DreamState::Pending);
    assert!(
        repository
            .lease("leased-1")
            .await
            .expect("read the lease")
            .is_none(),
        "the lease is released even though the transition failed"
    );

    // The requeued job is claimable again: a pending claim never needs the stale lease.
    let summary = planner
        .run_leased(ts(11), 10, &CallOptions::new(), &lease_options("worker-b"))
        .await
        .expect("recovery sweep");
    assert_eq!((summary.scanned, summary.succeeded), (1, 1));
    assert_eq!(
        repository
            .get("leased-1")
            .await
            .expect("get")
            .expect("the enqueued job")
            .state,
        DreamState::Succeeded
    );
}

#[tokio::test]
async fn a_completion_persists_its_notes_and_a_failed_attempt_clears_them() {
    let consolidator = Arc::new(SucceedingConsolidator::new(success_metrics()));
    let shared: Arc<dyn DreamConsolidator> = consolidator;
    let planner =
        DreamPlanner::with_in_memory(shared, DreamConfig::default()).expect("a valid planner");
    planner
        .repository()
        .enqueue(job("noted-1", ts(0)))
        .await
        .expect("enqueue");

    planner.run_once(ts(10), 10).await.expect("sweep");
    let stored = planner
        .repository()
        .get("noted-1")
        .await
        .expect("get")
        .expect("the enqueued job");
    assert_eq!(stored.state, DreamState::Succeeded);
    assert_eq!(
        stored.notes,
        vec!["consolidated by the test consolidator".to_owned()],
        "the outcome's notes are durable, not just part of the sweep summary"
    );

    // A job imported with notes keeps them until a completion replaces them; a failed
    // attempt clears them, because they describe work that attempt did not do.
    let failing = Arc::new(FailingConsolidator::new("boom"));
    let shared: Arc<dyn DreamConsolidator> = failing;
    let retrying =
        DreamPlanner::with_in_memory(shared, DreamConfig::default()).expect("a valid planner");
    retrying
        .repository()
        .enqueue(
            job("noted-2", ts(0))
                .with_max_attempts(2)
                .with_notes(vec!["imported".to_owned()]),
        )
        .await
        .expect("enqueue");
    retrying.run_once(ts(10), 10).await.expect("sweep");
    let requeued = retrying
        .repository()
        .get("noted-2")
        .await
        .expect("get")
        .expect("the enqueued job");
    assert_eq!(requeued.state, DreamState::Pending);
    assert!(requeued.notes.is_empty(), "{:?}", requeued.notes);
}

#[cfg(feature = "sqlite")]
#[tokio::test]
async fn sqlite_notes_are_durable_and_an_unknown_state_is_a_typed_error() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let url = format!("sqlite:{}", dir.path().join("dreams.db").display());
    let repository: Arc<dyn DreamRepository> = Arc::new(
        SqliteDreamRepository::connect(&url)
            .await
            .expect("open the database"),
    );
    let consolidator = Arc::new(SucceedingConsolidator::new(success_metrics()));
    let shared: Arc<dyn DreamConsolidator> = consolidator;
    let planner =
        DreamPlanner::new(repository.clone(), shared, DreamConfig::default()).expect("a planner");
    repository
        .enqueue(job("sqlite-notes", ts(0)))
        .await
        .expect("enqueue");
    planner.run_once(ts(10), 10).await.expect("sweep");
    repository
        .enqueue(
            DreamJob::new(scope(), DreamKind::Dedupe, DreamTrigger::Manual, ts(1))
                .with_id("sqlite-corrupt"),
        )
        .await
        .expect("enqueue");

    // Corrupt one row's state straight in the table: no repository write produces it.
    let sqlite = SqliteDreamRepository::connect(&url)
        .await
        .expect("reopen the database");
    sqlx::query("UPDATE dream_jobs SET state = 'dormant' WHERE id = ?1")
        .bind("sqlite-corrupt")
        .execute(sqlite.pool())
        .await
        .expect("corrupt the row");

    let verified = SqliteDreamRepository::connect(&url)
        .await
        .expect("final reopen");
    let stored = verified
        .get("sqlite-notes")
        .await
        .expect("get")
        .expect("the completed job");
    assert_eq!(
        stored.notes,
        vec!["consolidated by the test consolidator".to_owned()]
    );

    // An unknown state is never reported as a runnable `pending` job: the reads and the
    // counts fail with the same typed, column-naming error.
    let read = verified
        .get("sqlite-corrupt")
        .await
        .expect_err("an unknown state cannot be decoded");
    assert!(matches!(read, Error::External(_)), "{read:?}");
    assert!(read.to_string().contains("`state`"), "{read}");
    assert!(read.to_string().contains("dormant"), "{read}");
    let counted = verified
        .count_by_state()
        .await
        .expect_err("the counts refuse the same row");
    assert!(counted.to_string().contains("`state`"), "{counted}");
}
