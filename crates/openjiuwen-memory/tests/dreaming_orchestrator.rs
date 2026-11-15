//! Integration coverage for the dream *worker* half of `openjiuwen_memory::dreaming`.
//!
//! `tests/dreaming.rs` covers the queue and planner basics (claim, retry budget, noop
//! skips, SQLite durability). This suite drives the layers above and beside it the way an
//! embedding application does:
//!
//! - [`DreamOrchestrator`] ticks with an injected clock, including retention on tick and a
//!   retention failure that must not fail the sweep;
//! - cross-worker leasing through [`DreamLeaseOptions`], `DreamPlanner::run_leased`, and
//!   [`DreamLeaseGuard`] — mutual exclusion, expiry takeover, and the old owner's silence;
//! - [`DreamRetention`] age and per-state caps with pinned finish times;
//! - the deterministic salience scorer and its typed `Unsupported` ML seam;
//! - [`LlmConsolidator`] against `ScriptedModel`: a usable answer, a prose answer that
//!   degrades to the rule plan, and a model-less configuration;
//! - `run_until_cancelled` for a pre-cancelled token and a cancellation that lands while
//!   the consolidator is still running.
//!
//! Every test pins the clock with [`ts`] and waits at most [`BOUND`]; nothing touches the
//! network or depends on wall-clock ordering. The single SQLite test uses its own
//! `tempfile::TempDir` and is gated behind the `sqlite` feature.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use chrono::{DateTime, Duration as ChronoDuration, Utc};
use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::llm::Model;
use openjiuwen_memory::dreaming::{
    deterministic_plan, rank_by_salience, salience_score, DreamCandidate, DreamConfig,
    DreamConsolidationOrigin, DreamConsolidator, DreamFilter, DreamJob, DreamKind, DreamLeaseGuard,
    DreamLeaseOptions, DreamMetrics, DreamOrchestrator, DreamOrchestratorConfig, DreamOutcome,
    DreamPlanner, DreamRepository, DreamRetention, DreamState, DreamTrigger,
    InMemoryDreamRepository, LeaseLostReason, LlmConsolidator, LlmConsolidatorConfig,
    SalienceBackend, SalienceConfig, SalienceWeights, StaticCandidateSource, RECORD_ONLY_NOTE,
};
use openjiuwen_memory::record::{CallOptions, MemoryRecord, MemoryScope, MemoryType};
use openjiuwen_test_support::{ScriptedModel, ScriptedTurn};
use parking_lot::Mutex;
use tokio_util::sync::CancellationToken;

#[cfg(feature = "sqlite")]
use openjiuwen_memory::dreaming::SqliteDreamRepository;

/// Upper bound for every await that can wait on a timer.
const BOUND: Duration = Duration::from_secs(10);

/// The shared test epoch; every instant in this suite is this plus an offset.
const EPOCH_SECS: i64 = 1_700_000_000;

/// A fixed instant `offset_secs` seconds after [`EPOCH_SECS`].
fn ts(offset_secs: i64) -> DateTime<Utc> {
    DateTime::from_timestamp(EPOCH_SECS + offset_secs, 0).expect("a valid timestamp")
}

/// The scope every job and record in this suite belongs to.
fn scope() -> MemoryScope {
    MemoryScope::new("alice", "workspace")
}

/// Builds a pending consolidation job with `id`, created at `created_at`.
fn job(id: &str, created_at: DateTime<Utc>) -> DreamJob {
    DreamJob::new(
        scope(),
        DreamKind::Consolidation,
        DreamTrigger::Manual,
        created_at,
    )
    .with_id(id)
}

/// Builds a terminal job of `state` that finished at `finished_at`.
///
/// `DreamJob::validate` requires `finished_at` (and `updated_at`) at or after the creation
/// time, so the creation time is the earlier of the two.
fn terminal_job(
    id: &str,
    state: DreamState,
    created_at: DateTime<Utc>,
    finished_at: DateTime<Utc>,
) -> DreamJob {
    DreamJob {
        state,
        attempts: 1,
        started_at: Some(finished_at),
        finished_at: Some(finished_at),
        updated_at: finished_at,
        ..DreamJob::new(
            scope(),
            DreamKind::Consolidation,
            DreamTrigger::Manual,
            created_at,
        )
    }
    .with_id(id)
}

/// The counters [`RecordingConsolidator`] reports for a successful run.
fn success_metrics() -> DreamMetrics {
    DreamMetrics {
        records_scanned: 4,
        records_merged: 2,
        records_deleted: 1,
        records_created: 1,
        tokens_saved: 96,
        duration_ms: 5,
    }
}

/// The distinct counters a takeover worker reports, so the stored row names its author.
fn takeover_metrics() -> DreamMetrics {
    DreamMetrics {
        records_scanned: 3,
        records_merged: 1,
        records_deleted: 0,
        records_created: 1,
        tokens_saved: 40,
        duration_ms: 2,
    }
}

/// Lease options for `owner` with `ttl_secs` and a one-second heartbeat (always shorter
/// than the TTL, as `DreamLeaseOptions::validate` requires).
fn lease_options(owner: &str, ttl_secs: i64) -> DreamLeaseOptions {
    DreamLeaseOptions::new(
        owner,
        ChronoDuration::seconds(ttl_secs),
        Duration::from_secs(1),
    )
}

/// Loads the stored row for `id`, panicking when it is missing.
async fn load_job(repository: &Arc<dyn DreamRepository>, id: &str) -> DreamJob {
    repository
        .get(id)
        .await
        .expect("get")
        .expect("the job is stored")
}

/// A consolidator that completes every job with fixed metrics and records the ids it saw.
struct RecordingConsolidator {
    /// Counters copied onto every successful outcome.
    metrics: DreamMetrics,
    /// Job ids in call order.
    claims: Mutex<Vec<String>>,
}

impl RecordingConsolidator {
    /// Builds the consolidator with `metrics` as its reported counters.
    fn new(metrics: DreamMetrics) -> Self {
        Self {
            metrics,
            claims: Mutex::new(Vec::new()),
        }
    }

    /// The job ids this consolidator was handed, in call order.
    fn claims(&self) -> Vec<String> {
        self.claims.lock().clone()
    }
}

#[async_trait]
impl DreamConsolidator for RecordingConsolidator {
    async fn consolidate(&self, job: &DreamJob, ctx: &CallOptions) -> Result<DreamOutcome> {
        self.claims.lock().push(job.id.clone());
        ctx.check()?;
        Ok(DreamOutcome::succeeded(job.id.clone(), self.metrics))
    }
}

/// A consolidator that signals entry, then blocks until the test releases it, so two
/// workers can be interleaved deterministically (or cancelled mid-flight).
struct BlockingConsolidator {
    /// Notified once `consolidate` is entered.
    entered: tokio::sync::Notify,
    /// Notified by the test to let the entry finish.
    release: tokio::sync::Notify,
}

impl BlockingConsolidator {
    /// Builds the blocked consolidator.
    fn new() -> Self {
        Self {
            entered: tokio::sync::Notify::new(),
            release: tokio::sync::Notify::new(),
        }
    }

    /// Resolves once `consolidate` has been entered at least once.
    async fn wait_entered(&self) {
        self.entered.notified().await;
    }

    /// Lets every entered consolidation finish.
    fn release(&self) {
        self.release.notify_one();
    }
}

#[async_trait]
impl DreamConsolidator for BlockingConsolidator {
    async fn consolidate(&self, job: &DreamJob, _ctx: &CallOptions) -> Result<DreamOutcome> {
        self.entered.notify_one();
        self.release.notified().await;
        Ok(DreamOutcome::succeeded(
            job.id.clone(),
            DreamMetrics::default(),
        ))
    }
}

/// A repository double over the in-memory backend that keeps the trait's default
/// (typed `Unsupported`) `delete_job`, so a retention purge has to fail.
#[derive(Debug, Default)]
struct NonDeletingRepository {
    /// The delegating backend.
    inner: InMemoryDreamRepository,
}

#[async_trait]
impl DreamRepository for NonDeletingRepository {
    async fn enqueue(&self, job: DreamJob) -> Result<DreamJob> {
        self.inner.enqueue(job).await
    }

    async fn claim_next(&self, now: DateTime<Utc>) -> Result<Option<DreamJob>> {
        self.inner.claim_next(now).await
    }

    async fn complete(&self, job_id: &str, outcome: DreamOutcome) -> Result<DreamJob> {
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

    async fn count_by_state(&self) -> Result<BTreeMap<DreamState, usize>> {
        self.inner.count_by_state().await
    }

    async fn delete_finished(&self, older_than: DateTime<Utc>) -> Result<usize> {
        self.inner.delete_finished(older_than).await
    }
}

/// Builds an [`InMemoryDreamRepository`] behind the trait object the workers expect.
fn in_memory() -> Arc<dyn DreamRepository> {
    Arc::new(InMemoryDreamRepository::new())
}

// --- orchestrator ticks -------------------------------------------------------------

#[tokio::test]
async fn an_orchestrator_tick_claims_and_completes_one_due_job() {
    let repository = in_memory();
    let consolidator = Arc::new(RecordingConsolidator::new(success_metrics()));
    let planner = Arc::new(
        DreamPlanner::new(
            repository.clone(),
            consolidator.clone(),
            DreamConfig::default(),
        )
        .expect("a valid planner"),
    );
    let orchestrator = DreamOrchestrator::new(
        planner,
        repository.clone(),
        DreamOrchestratorConfig::default(),
    )
    .expect("a valid orchestrator");
    repository
        .enqueue(job("tick-1", ts(0)))
        .await
        .expect("enqueue");

    let now = ts(10);
    let summary = orchestrator.tick(now).await.expect("the tick succeeds");

    assert_eq!((summary.scanned, summary.succeeded), (1, 1));
    assert_eq!(summary.outcomes[0].job_id, "tick-1");
    assert_eq!(summary.outcomes[0].metrics, success_metrics());

    let status = orchestrator.status();
    assert_eq!(status.ticks, 1);
    assert_eq!(status.last_tick_at, Some(now));
    assert_eq!(
        status.last_summary.as_ref().map(|sweep| sweep.succeeded),
        Some(1)
    );
    assert_eq!(status.consecutive_empty_sweeps, 0);
    assert!(status.last_error.is_none());
    assert!(status.last_retention.is_none());

    let stored = load_job(&repository, "tick-1").await;
    assert!(stored.is_terminal(), "a completed job is terminal");
    assert_eq!(stored.state, DreamState::Succeeded);
    assert_eq!(stored.metrics, success_metrics());
    assert!(stored.finished_at.is_some());
    assert_eq!(consolidator.claims(), vec!["tick-1".to_owned()]);

    let running = repository
        .list(&DreamFilter::new().with_state(DreamState::Running))
        .await
        .expect("list running");
    assert!(
        running.is_empty(),
        "a completed tick leaves nothing running"
    );
}

#[tokio::test]
async fn two_orchestrators_sharing_one_repository_complete_the_job_once() {
    let repository = in_memory();
    let consolidator = Arc::new(BlockingConsolidator::new());
    let planner = Arc::new(
        DreamPlanner::new(
            repository.clone(),
            consolidator.clone(),
            DreamConfig::default(),
        )
        .expect("a valid planner"),
    );
    let winner = Arc::new(
        DreamOrchestrator::new(
            planner.clone(),
            repository.clone(),
            DreamOrchestratorConfig::default().with_lease(lease_options("worker-a", 30)),
        )
        .expect("a valid orchestrator"),
    );
    let loser = DreamOrchestrator::new(
        planner,
        repository.clone(),
        DreamOrchestratorConfig::default().with_lease(lease_options("worker-b", 30)),
    )
    .expect("a valid orchestrator");
    repository
        .enqueue(job("lease-race", ts(0)))
        .await
        .expect("enqueue");

    let running = Arc::clone(&winner);
    let handle = tokio::spawn(async move { running.tick(ts(10)).await });
    consolidator.wait_entered().await;

    // The job is `Running` under worker-a's live lease, so worker-b claims nothing.
    let losing = loser.tick(ts(10)).await.expect("the second tick succeeds");
    assert!(losing.is_empty(), "the loser claims nothing");
    assert_eq!(loser.status().ticks, 1);
    assert_eq!(
        loser
            .status()
            .last_summary
            .as_ref()
            .map(|sweep| sweep.scanned),
        Some(0)
    );
    assert!(loser.status().last_error.is_none());

    consolidator.release();
    let winning = tokio::time::timeout(BOUND, handle)
        .await
        .expect("the winner finishes")
        .expect("the spawned tick does not panic")
        .expect("the winner tick succeeds");
    assert_eq!((winning.scanned, winning.succeeded), (1, 1));
    assert_eq!(winning.outcomes[0].job_id, "lease-race");

    let stored = load_job(&repository, "lease-race").await;
    assert_eq!(stored.state, DreamState::Succeeded);
    assert!(
        repository
            .lease("lease-race")
            .await
            .expect("lease lookup")
            .is_none(),
        "no lease row remains after the winning sweep"
    );
}

#[tokio::test]
async fn an_expired_lease_is_taken_over_and_the_first_owner_records_nothing() {
    let repository = in_memory();
    repository
        .enqueue(job("lease-expiry", ts(0)))
        .await
        .expect("enqueue");

    // Worker A claims the job at ts(10) with a two-second TTL, so it expires at ts(12).
    let stale = lease_options("worker-a", 2);
    let request = stale.request();
    let leased = repository
        .claim_next_leased(ts(10), &request)
        .await
        .expect("claim under a lease")
        .expect("worker-a claims the job");
    assert_eq!(leased.lease.owner, "worker-a");
    assert_eq!(leased.lease.expires_at, ts(12));
    let guard = DreamLeaseGuard::new(repository.clone(), leased.lease);

    // Worker B sweeps past the expiry and takes the job over.
    let fresh = Arc::new(RecordingConsolidator::new(takeover_metrics()));
    let planner = Arc::new(
        DreamPlanner::new(repository.clone(), fresh.clone(), DreamConfig::default())
            .expect("a valid planner"),
    );
    let summary = planner
        .run_leased(
            ts(13),
            10,
            &CallOptions::new(),
            &lease_options("worker-b", 30),
        )
        .await
        .expect("the takeover sweep succeeds");
    assert_eq!((summary.scanned, summary.succeeded), (1, 1));
    assert_eq!(summary.outcomes[0].metrics, takeover_metrics());

    let stored = load_job(&repository, "lease-expiry").await;
    assert_eq!(stored.state, DreamState::Succeeded);
    assert_eq!(
        stored.metrics,
        takeover_metrics(),
        "the row carries worker-b's outcome"
    );
    assert_eq!(stored.attempts, 2, "the takeover is a second attempt");

    // Worker A discovers the loss from its own guard, without a repository round-trip.
    let status = guard
        .renew(ts(13), ChronoDuration::seconds(2))
        .await
        .expect("renew reports the loss");
    assert!(status.is_lost(), "a locally expired lease is lost");
    assert_eq!(status.reason(), Some(LeaseLostReason::Expired));
    assert!(guard.is_lost());
    assert_eq!(guard.lost_reason(), Some(LeaseLostReason::Expired));

    // Worker A recorded nothing: the row still belongs to worker-b.
    assert_eq!(
        load_job(&repository, "lease-expiry").await.metrics,
        takeover_metrics()
    );
    assert!(
        repository
            .lease("lease-expiry")
            .await
            .expect("lease lookup")
            .is_none(),
        "worker-b released its lease"
    );
}

// --- retention ----------------------------------------------------------------------

#[tokio::test]
async fn retention_purges_expired_jobs_at_the_injected_clock_and_is_idempotent() {
    let repository = in_memory();
    for (id, state, created, finished) in [
        ("expired-succeeded", DreamState::Succeeded, ts(0), ts(0)),
        ("expired-failed", DreamState::Failed, ts(5), ts(5)),
        ("fresh-skipped", DreamState::Skipped, ts(90), ts(90)),
        ("fresh-cancelled", DreamState::Cancelled, ts(95), ts(95)),
    ] {
        repository
            .enqueue(terminal_job(id, state, created, finished))
            .await
            .expect("enqueue");
    }

    let policy = DreamRetention::from_config(&DreamConfig {
        retention_secs: 30,
        ..DreamConfig::default()
    });
    let now = ts(100);

    // The cutoff is now - 30s = ts(70): only the two ts(0)/ts(5) rows are older.
    let report = policy
        .purge_expired(repository.as_ref(), now)
        .await
        .expect("the purge succeeds");
    assert_eq!(report.deleted_expired, 2, "{report:?}");
    assert_eq!(report.deleted_over_cap, 0);
    assert_eq!(report.total(), 2);

    let counts = repository.count_by_state().await.expect("state counts");
    assert_eq!(counts[&DreamState::Succeeded], 0);
    assert_eq!(counts[&DreamState::Failed], 0);
    assert_eq!(counts[&DreamState::Skipped], 1);
    assert_eq!(counts[&DreamState::Cancelled], 1);
    assert!(repository
        .get("expired-succeeded")
        .await
        .expect("get")
        .is_none());
    assert!(repository
        .get("fresh-skipped")
        .await
        .expect("get")
        .is_some());

    // Idempotent: the same clock removes nothing more.
    let second = policy
        .purge_expired(repository.as_ref(), now)
        .await
        .expect("the second purge succeeds");
    assert!(second.is_empty(), "{second:?}");
    assert_eq!(second.total(), 0);
}

#[tokio::test]
async fn the_retention_cap_keeps_the_newest_terminal_jobs() {
    let repository = in_memory();
    for offset in 0..5 {
        repository
            .enqueue(terminal_job(
                &format!("cap-{offset}"),
                DreamState::Succeeded,
                ts(offset),
                ts(offset),
            ))
            .await
            .expect("enqueue");
    }

    // The age rule is off, so only the newest two survive the cap.
    let policy = DreamRetention::new(0).with_max_per_state(2);
    let report = policy
        .purge_expired(repository.as_ref(), ts(100))
        .await
        .expect("the capped purge succeeds");
    assert_eq!(report.deleted_expired, 0);
    assert_eq!(report.deleted_over_cap, 3, "{report:?}");
    assert_eq!(report.total(), 3);

    let remaining = repository
        .list(&DreamFilter::new().with_state(DreamState::Succeeded))
        .await
        .expect("list succeeded");
    assert_eq!(remaining.len(), 2);
    for id in ["cap-0", "cap-1", "cap-2"] {
        assert!(
            repository.get(id).await.expect("get").is_none(),
            "`{id}` is past the cap"
        );
    }
    for id in ["cap-3", "cap-4"] {
        assert!(
            repository.get(id).await.expect("get").is_some(),
            "`{id}` is among the newest"
        );
    }
}

#[tokio::test]
async fn orchestrator_retention_purges_during_a_tick() {
    let repository = in_memory();
    let consolidator = Arc::new(RecordingConsolidator::new(success_metrics()));
    let planner = Arc::new(
        DreamPlanner::new(
            repository.clone(),
            consolidator.clone(),
            DreamConfig::default(),
        )
        .expect("a valid planner"),
    );
    let policy = DreamRetention::from_config(&DreamConfig {
        retention_secs: 30,
        ..DreamConfig::default()
    });
    let config = DreamOrchestratorConfig::default()
        .with_run_retention_on_tick(true)
        .with_retention(policy);
    let orchestrator =
        DreamOrchestrator::new(planner, repository.clone(), config).expect("a valid orchestrator");
    repository
        .enqueue(terminal_job("purge-1", DreamState::Succeeded, ts(0), ts(0)))
        .await
        .expect("enqueue");

    let summary = orchestrator.tick(ts(100)).await.expect("the tick succeeds");
    assert!(summary.is_empty(), "the queue has nothing claimable");

    let status = orchestrator.status();
    assert_eq!(status.ticks, 1);
    assert!(
        status.last_summary.is_some(),
        "the sweep summary still stands"
    );
    assert!(status.last_error.is_none());
    let report = status.last_retention.expect("retention ran on the tick");
    assert_eq!(report.deleted_expired, 1);
    assert_eq!(report.total(), 1);
    assert!(repository.get("purge-1").await.expect("get").is_none());
}

#[tokio::test]
async fn a_retention_failure_does_not_fail_the_tick() {
    let repository: Arc<dyn DreamRepository> = Arc::new(NonDeletingRepository::default());
    let consolidator = Arc::new(RecordingConsolidator::new(success_metrics()));
    let planner = Arc::new(
        DreamPlanner::new(
            repository.clone(),
            consolidator.clone(),
            DreamConfig::default(),
        )
        .expect("a valid planner"),
    );
    // The cap needs `delete_job`, which this backend deliberately leaves `Unsupported`.
    let config = DreamOrchestratorConfig::default()
        .with_run_retention_on_tick(true)
        .with_retention(DreamRetention::new(0).with_max_per_state(1));
    let orchestrator =
        DreamOrchestrator::new(planner, repository.clone(), config).expect("a valid orchestrator");
    for offset in 0..2 {
        repository
            .enqueue(terminal_job(
                &format!("keep-{offset}"),
                DreamState::Succeeded,
                ts(offset),
                ts(offset),
            ))
            .await
            .expect("enqueue");
    }

    let summary = orchestrator
        .tick(ts(10))
        .await
        .expect("a retention failure never fails the tick");
    assert!(summary.is_empty());

    let status = orchestrator.status();
    assert_eq!(status.ticks, 1);
    assert!(
        status.last_summary.is_some(),
        "the sweep's summary is still recorded"
    );
    assert!(
        status.last_retention.is_none(),
        "a failed purge stores no report"
    );
    let error = status
        .last_error
        .as_deref()
        .expect("the retention failure is recorded");
    assert!(
        error.contains("does not support deleting a single job"),
        "{error}"
    );
    let remaining = repository
        .list(&DreamFilter::new().with_state(DreamState::Succeeded))
        .await
        .expect("list succeeded");
    assert_eq!(remaining.len(), 2, "nothing was deleted");
}

// --- salience -----------------------------------------------------------------------

/// Builds a record with pinned salience inputs and `id` as its identity.
fn memory_record(
    id: &str,
    kind: MemoryType,
    importance: f32,
    timestamp: DateTime<Utc>,
    access_count: u64,
) -> MemoryRecord {
    MemoryRecord {
        importance,
        timestamp,
        updated_at: timestamp,
        access_count,
        ..MemoryRecord::new(scope(), "a memory worth scoring", kind)
    }
    .with_id(id)
}

/// Asserts two `f32` values agree to within the scorer's arithmetic tolerance.
fn assert_close(actual: f32, expected: f32) {
    assert!(
        (actual - expected).abs() <= 1e-6,
        "expected {expected}, got {actual}"
    );
}

#[test]
fn salience_score_reports_every_component_and_ranks_descending() {
    let config = SalienceConfig::default()
        .with_half_life_secs(1_000)
        .with_access_saturation(10.0);
    let now = ts(1_000);
    let profile = memory_record("profile", MemoryType::UserProfile, 0.9, ts(0), 2);
    let semantic = memory_record("semantic", MemoryType::SemanticMemory, 0.5, ts(500), 0);
    let unknown = memory_record("unknown", MemoryType::Unknown, 1.5, ts(1_000), 30);

    let scored = salience_score(&profile, now, &config).expect("a valid configuration");
    assert_close(scored.importance, 0.9);
    assert_close(scored.recency, 0.5);
    assert_close(scored.access, 2.0 / 12.0);
    assert_close(scored.kind, 1.0);
    assert_close(scored.total, (0.9 + 0.5 + 2.0 / 12.0 + 1.0) / 4.0);

    let fresh = salience_score(&unknown, now, &config).expect("a valid configuration");
    assert_close(fresh.importance, 1.0);
    assert_close(fresh.recency, 1.0);
    assert_close(fresh.access, 0.75);
    assert_close(fresh.kind, 0.5);
    assert_close(fresh.total, (1.0 + 1.0 + 0.75 + 0.5) / 4.0);

    let stale = salience_score(&semantic, now, &config).expect("a valid configuration");
    assert_close(stale.recency, 0.5_f32.powf(0.5));
    assert_close(stale.access, 0.0);
    assert_close(stale.kind, 0.9);

    let ranked = rank_by_salience(&[profile, semantic, unknown], now, &config)
        .expect("a valid configuration");
    let ids: Vec<&str> = ranked.iter().map(|(id, _)| id.as_str()).collect();
    assert_eq!(ids, vec!["unknown", "profile", "semantic"]);
    for pair in ranked.windows(2) {
        assert!(
            pair[0].1.total >= pair[1].1.total,
            "the ranking descends by total"
        );
    }
}

#[test]
fn zero_weights_score_zero_and_the_ml_backend_is_unsupported() {
    let zero_weights = SalienceWeights {
        importance: 0.0,
        recency: 0.0,
        access: 0.0,
        kind: 0.0,
    };
    let config = SalienceConfig::default().with_weights(zero_weights);
    let record = memory_record("zero", MemoryType::UserProfile, 1.0, ts(0), 5);

    let scored = salience_score(&record, ts(10), &config).expect("all-zero weights are valid");
    assert!(scored.total.abs() < f32::EPSILON, "{}", scored.total);
    assert!(!scored.total.is_nan(), "a zero weight sum is never NaN");
    assert_close(scored.importance, 1.0);
    assert_close(scored.access, 5.0 / 15.0);

    let ml = config.with_backend(SalienceBackend::Ml);
    let error = salience_score(&record, ts(10), &ml).expect_err("ML salience is deferred");
    assert!(matches!(error, Error::Unsupported(_)), "{error:?}");
    assert!(error.to_string().contains("ML-based salience"), "{error}");
    let error = rank_by_salience(&[record], ts(10), &ml).expect_err("ML salience is deferred");
    assert!(matches!(error, Error::Unsupported(_)), "{error:?}");
}

// --- LLM consolidation --------------------------------------------------------------

/// A model answer the parser accepts: merge `b` into `a`, keep `c`.
const MERGE_AND_KEEP: &str = r#"{"actions":[
    {"action":"merge","target_id":"a","source_ids":["b"],"content":"alice prefers dark mode","reason":"duplicate"},
    {"action":"keep","id":"c"}
]}"#;

/// Three candidates: `a`/`b` are near-duplicates and `c` is unrelated.
fn candidates() -> Vec<DreamCandidate> {
    vec![
        DreamCandidate::new("a", "alice prefers dark mode in the editor"),
        DreamCandidate::new("b", "alice prefers dark mode"),
        DreamCandidate::new("c", "the weather is nice today"),
    ]
}

/// Builds an [`LlmConsolidator`] over [`candidates`] with the default record-only sink.
fn llm_consolidator(
    model: Option<Arc<dyn Model>>,
    config: LlmConsolidatorConfig,
) -> LlmConsolidator {
    LlmConsolidator::new(
        model,
        Arc::new(StaticCandidateSource::new(candidates())),
        config,
    )
    .expect("a valid consolidator")
}

#[tokio::test]
async fn a_valid_model_answer_is_llm_origin_and_completes_unde_graded() {
    let model = ScriptedModel::builder()
        .respond_text(MERGE_AND_KEEP)
        .respond_text(MERGE_AND_KEEP)
        .build();
    let consolidator = llm_consolidator(
        Some(Arc::new(model.clone())),
        LlmConsolidatorConfig::default(),
    );
    let pending = job("llm-valid", ts(0));

    let plan = consolidator
        .plan(&pending, &CallOptions::new())
        .await
        .expect("a plan");
    assert_eq!(plan.origin, DreamConsolidationOrigin::Llm);
    assert!(!plan.is_degraded(), "{:?}", plan.degraded);
    assert_eq!(plan.actions.len(), 2);

    let outcome = consolidator
        .consolidate(&pending, &CallOptions::new())
        .await
        .expect("an outcome");
    assert_eq!(outcome.state, DreamState::Succeeded);
    assert_eq!(outcome.job_id, "llm-valid");
    // The default sink is record-only: it observes the candidates but applies nothing, so
    // the planned merge ride on the plan, never on the metrics stored with the job.
    assert_eq!(plan.metrics.records_merged, 1);
    assert_eq!(outcome.metrics.records_scanned, 3);
    assert_eq!(outcome.metrics.records_merged, 0);
    assert_eq!(outcome.metrics.records_created, 0);
    assert_eq!(outcome.metrics.records_deleted, 0);
    assert_eq!(outcome.metrics.tokens_saved, 0);
    assert!(
        outcome
            .notes
            .iter()
            .any(|note| note.contains("merge 1 record(s) into `a`")),
        "{:?}",
        outcome.notes
    );
    assert_eq!(
        outcome.notes.last().map(String::as_str),
        Some(RECORD_ONLY_NOTE),
        "a sink that applies nothing says so"
    );
    assert!(
        !outcome
            .notes
            .iter()
            .any(|note| note.starts_with("degraded:")),
        "a model answer is never degraded: {:?}",
        outcome.notes
    );
    assert_eq!(
        model.call_count(),
        2,
        "one model call for `plan`, one for `consolidate`"
    );
}

#[tokio::test]
async fn a_malformed_model_answer_degrades_to_the_deterministic_plan() {
    let model = ScriptedModel::new([
        ScriptedTurn::text("this is prose, not json"),
        ScriptedTurn::text("still prose"),
    ]);
    // `a`/`b` share four of their seven union tokens, so the fallback merges them only
    // below the default 0.75 threshold.
    let config = LlmConsolidatorConfig::default().with_fallback_similarity(0.5);
    let consolidator = llm_consolidator(Some(Arc::new(model.clone())), config.clone());
    let pending = job("llm-degraded", ts(0));

    let outcome = consolidator
        .consolidate(&pending, &CallOptions::new())
        .await
        .expect("a degraded outcome is still an outcome");

    assert_eq!(
        outcome.state,
        DreamState::Succeeded,
        "the rule plan still merges the duplicates"
    );
    assert_eq!(model.call_count(), 2, "one call plus the configured retry");
    let degraded = outcome.notes.first().expect("a degradation note");
    assert!(degraded.starts_with("degraded: "), "{degraded}");
    assert!(degraded.contains("fell back to rules"), "{degraded}");
    assert_eq!(outcome.metrics.records_scanned, 3);
    assert_eq!(
        outcome.metrics.records_merged, 0,
        "the fallback decides the merge but the record-only sink applies none of it"
    );

    let expected = deterministic_plan("llm-degraded", &candidates(), config.fallback_similarity);
    assert_eq!(
        expected.metrics.records_merged, 1,
        "the plan decides a merge"
    );
    assert_eq!(
        &outcome.notes[1..=expected.notes().len()],
        expected.notes().as_slice()
    );
    assert_eq!(
        outcome.notes.last().map(String::as_str),
        Some(RECORD_ONLY_NOTE)
    );
}

#[tokio::test]
async fn a_consolidator_without_a_model_degrades_deterministically() {
    let config = LlmConsolidatorConfig::default().with_fallback_similarity(0.5);
    let consolidator = llm_consolidator(None, config.clone());
    let pending = job("llm-no-model", ts(0));

    let outcome = consolidator
        .consolidate(&pending, &CallOptions::new())
        .await
        .expect("a degraded outcome is still an outcome");
    assert_eq!(outcome.state, DreamState::Succeeded);
    assert_eq!(
        outcome.notes.first().map(String::as_str),
        Some("degraded: no consolidation model is configured")
    );

    let expected = deterministic_plan("llm-no-model", &candidates(), config.fallback_similarity);
    assert_eq!(
        &outcome.notes[1..=expected.notes().len()],
        expected.notes().as_slice()
    );
    assert_eq!(
        outcome.notes.last().map(String::as_str),
        Some(RECORD_ONLY_NOTE)
    );
    assert_eq!(outcome.metrics.records_merged, 0);
}

// --- the caller-driven loop ---------------------------------------------------------

#[tokio::test]
async fn run_until_cancelled_returns_cancelled_for_a_pre_cancelled_token() {
    let repository = in_memory();
    let consolidator = Arc::new(RecordingConsolidator::new(success_metrics()));
    let planner = Arc::new(
        DreamPlanner::new(
            repository.clone(),
            consolidator.clone(),
            DreamConfig::default(),
        )
        .expect("a valid planner"),
    );
    let config = DreamOrchestratorConfig::default()
        .with_initial_delay_ms(0)
        .with_poll_interval_ms(5);
    let orchestrator =
        DreamOrchestrator::new(planner, repository.clone(), config).expect("a valid orchestrator");
    repository
        .enqueue(job("cancel-pre", ts(0)))
        .await
        .expect("enqueue");

    let token = CancellationToken::new();
    token.cancel();
    let error = tokio::time::timeout(BOUND, orchestrator.run_until_cancelled(token))
        .await
        .expect("the loop returns promptly")
        .expect_err("cancellation is an error");
    assert!(matches!(error, Error::Cancelled), "{error:?}");

    assert_eq!(orchestrator.status().ticks, 0);
    assert_eq!(consolidator.claims(), Vec::<String>::new());
    let stored = load_job(&repository, "cancel-pre").await;
    assert_eq!(stored.state, DreamState::Pending);
    assert_eq!(stored.attempts, 0);
}

#[tokio::test]
async fn cancellation_mid_consolidation_leaves_no_running_job() {
    let repository = in_memory();
    let consolidator = Arc::new(BlockingConsolidator::new());
    let planner = Arc::new(
        DreamPlanner::new(
            repository.clone(),
            consolidator.clone(),
            DreamConfig::default(),
        )
        .expect("a valid planner"),
    );
    let config = DreamOrchestratorConfig::default()
        .with_initial_delay_ms(0)
        .with_poll_interval_ms(5);
    let orchestrator = Arc::new(
        DreamOrchestrator::new(planner, repository.clone(), config).expect("a valid orchestrator"),
    );
    repository
        .enqueue(job("cancel-mid", ts(0)))
        .await
        .expect("enqueue");

    let token = CancellationToken::new();
    let running = Arc::clone(&orchestrator);
    let stop = token.clone();
    let handle = tokio::spawn(async move { running.run_until_cancelled(stop).await });
    consolidator.wait_entered().await;
    token.cancel();

    let error = tokio::time::timeout(BOUND, handle)
        .await
        .expect("the loop stops promptly")
        .expect("the spawned loop does not panic")
        .expect_err("cancellation is an error");
    assert!(matches!(error, Error::Cancelled), "{error:?}");

    let stored = load_job(&repository, "cancel-mid").await;
    assert_eq!(stored.state, DreamState::Cancelled);
    assert!(stored.finished_at.is_some(), "cancellation is terminal");
    let running = repository
        .list(&DreamFilter::new().with_state(DreamState::Running))
        .await
        .expect("list running");
    assert!(running.is_empty(), "no job is left running");
    assert!(
        orchestrator.status().ticks >= 1,
        "the cancelled tick still counted"
    );
}

// --- the durable backend ------------------------------------------------------------

#[cfg(feature = "sqlite")]
#[tokio::test]
async fn a_leased_tick_drains_a_sqlite_queue_and_leaves_no_lease_row() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let url = format!("sqlite:{}", dir.path().join("orchestrator.db").display());
    let repository: Arc<dyn DreamRepository> = Arc::new(
        SqliteDreamRepository::connect(&url)
            .await
            .expect("open the database"),
    );
    let consolidator = Arc::new(RecordingConsolidator::new(success_metrics()));
    let planner = Arc::new(
        DreamPlanner::new(
            repository.clone(),
            consolidator.clone(),
            DreamConfig::default(),
        )
        .expect("a valid planner"),
    );
    let orchestrator = DreamOrchestrator::new(
        planner,
        repository.clone(),
        DreamOrchestratorConfig::default().with_lease(lease_options("worker-a", 30)),
    )
    .expect("a valid orchestrator");
    repository
        .enqueue(job("sqlite-tick", ts(0)))
        .await
        .expect("enqueue");

    let summary = orchestrator.tick(ts(10)).await.expect("a leased tick");
    assert_eq!((summary.scanned, summary.succeeded), (1, 1));

    let stored = load_job(&repository, "sqlite-tick").await;
    assert_eq!(stored.state, DreamState::Succeeded);
    assert_eq!(stored.metrics, success_metrics());
    assert_eq!(stored.attempts, 1);
    assert!(
        repository
            .lease("sqlite-tick")
            .await
            .expect("lease lookup")
            .is_none(),
        "the lease row is deleted on completion"
    );
    let counts = repository.count_by_state().await.expect("state counts");
    assert_eq!(counts[&DreamState::Succeeded], 1);
    assert_eq!(counts[&DreamState::Running], 0);
}
