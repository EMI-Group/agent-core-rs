//! Unit suite for [`DreamOrchestrator`](super::DreamOrchestrator).
//!
//! Everything here is offline and deterministic: the clock is a parameter
//! ([`super::DreamOrchestrator::tick`]), the consolidator is scripted, and the loop tests
//! use millisecond intervals plus a bounded `tokio::time::timeout` instead of real waiting.
use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;
use std::time::Duration as StdDuration;

use async_trait::async_trait;
use chrono::{DateTime, Duration as ChronoDuration, Utc};
use parking_lot::Mutex;
use tokio_util::sync::CancellationToken;

use super::*;
use crate::dreaming::{
    DreamConfig, DreamConsolidator, DreamFilter, DreamJob, DreamKind, DreamMetrics, DreamOutcome,
    DreamState, DreamTrigger, InMemoryDreamRepository, SweepLockSkipKind, DREAM_SWEEP_LOCK_NAME,
};
use crate::record::MemoryScope;
use crate::store::kv_lock::{DistributedKvLock, InProcessKvLock, LockOptions, UnsupportedKvLock};

/// Upper bound for every await in this suite.
const BOUND: StdDuration = StdDuration::from_secs(5);

/// One scripted consolidator behaviour.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Step {
    /// Succeed with zero metrics.
    Succeeded,
    /// Block until released, so a test can cancel or take over mid-consolidation.
    Block,
}

/// A consolidator that replays a script of [`Step`]s, then repeats `Succeeded`.
#[derive(Debug)]
struct ScriptedConsolidator {
    steps: Mutex<VecDeque<Step>>,
    calls: Mutex<Vec<String>>,
    entered: tokio::sync::Notify,
    release: tokio::sync::Notify,
}

impl ScriptedConsolidator {
    fn new(steps: impl IntoIterator<Item = Step>) -> Self {
        Self {
            steps: Mutex::new(steps.into_iter().collect()),
            calls: Mutex::new(Vec::new()),
            entered: tokio::sync::Notify::new(),
            release: tokio::sync::Notify::new(),
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
        // The script lock is released before the await below.
        let step = self.steps.lock().pop_front().unwrap_or(Step::Succeeded);
        if step == Step::Block {
            self.entered.notify_one();
            self.release.notified().await;
        }
        Ok(DreamOutcome::succeeded(
            job.id.clone(),
            DreamMetrics::default(),
        ))
    }
}

/// A repository double over an in-memory backend that keeps the trait's default (typed
/// `Unsupported`) lease bodies, and can optionally refuse the single-row deletion
/// retention uses.
#[derive(Debug, Default)]
struct DoubleRepository {
    inner: InMemoryDreamRepository,
    fail_delete: bool,
}

impl DoubleRepository {
    /// A plain backend: no leasing, and every default body left in place.
    fn plain() -> Self {
        Self::default()
    }

    /// A backend whose `delete_job` fails, for the retention-failure path.
    fn failing_delete() -> Self {
        Self {
            fail_delete: true,
            ..Self::default()
        }
    }
}

#[async_trait]
impl DreamRepository for DoubleRepository {
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
    async fn delete_job(&self, job_id: &str) -> Result<bool> {
        if self.fail_delete {
            return Err(Error::Memory("scripted delete failure".to_owned()));
        }
        self.inner.delete_job(job_id).await
    }
}

async fn complete_one(repository: &Arc<dyn DreamRepository>, id: &str) {
    repository.enqueue(job(id)).await.unwrap();
    repository
        .claim_next(ts(10))
        .await
        .unwrap()
        .expect("claimed");
    repository
        .complete(id, DreamOutcome::succeeded(id, DreamMetrics::default()))
        .await
        .unwrap();
}

fn ts(secs: i64) -> DateTime<Utc> {
    DateTime::from_timestamp(1_700_000_000, 0).unwrap() + ChronoDuration::seconds(secs)
}

fn job(id: &str) -> DreamJob {
    DreamJob::new(
        MemoryScope::new("alice", "ws"),
        DreamKind::Consolidation,
        DreamTrigger::Manual,
        ts(0),
    )
    .with_id(id)
    .with_max_attempts(3)
}

fn repo() -> Arc<dyn DreamRepository> {
    Arc::new(InMemoryDreamRepository::new())
}

fn lease_options(owner: &str, ttl_secs: i64) -> DreamLeaseOptions {
    DreamLeaseOptions::new(
        owner,
        ChronoDuration::seconds(ttl_secs),
        StdDuration::from_secs(1),
    )
}

fn planner(
    repository: &Arc<dyn DreamRepository>,
    consolidator: &Arc<ScriptedConsolidator>,
    config: DreamConfig,
) -> Arc<DreamPlanner> {
    Arc::new(DreamPlanner::new(repository.clone(), consolidator.clone(), config).unwrap())
}

fn orchestrator(
    planner: &Arc<DreamPlanner>,
    repository: &Arc<dyn DreamRepository>,
    config: DreamOrchestratorConfig,
) -> DreamOrchestrator {
    DreamOrchestrator::new(planner.clone(), repository.clone(), config).unwrap()
}

async fn stored(repository: &Arc<dyn DreamRepository>, id: &str) -> DreamJob {
    repository
        .get(id)
        .await
        .unwrap()
        .expect("the job is stored")
}

fn assert_validation(error: Error, needle: &str) {
    assert!(matches!(error, Error::Validation(_)), "got {error:?}");
    assert!(
        error.to_string().contains(needle),
        "`{error}` should mention `{needle}`"
    );
}

async fn wait_for_ticks(orchestrator: &DreamOrchestrator, ticks: u64) {
    tokio::time::timeout(BOUND, async {
        while orchestrator.status().ticks < ticks {
            tokio::time::sleep(StdDuration::from_millis(2)).await;
        }
    })
    .await
    .expect("the orchestrator should tick");
}

#[tokio::test]
async fn a_tick_claims_and_completes_a_due_job() {
    let repository = repo();
    let consolidator = Arc::new(ScriptedConsolidator::new([Step::Succeeded]));
    let planner = planner(&repository, &consolidator, DreamConfig::default());
    let config = DreamOrchestratorConfig::default();
    let orchestrator = orchestrator(&planner, &repository, config.clone());

    repository.enqueue(job("job-1")).await.unwrap();
    let now = ts(10);
    let summary = orchestrator.tick(now).await.unwrap();

    assert_eq!((summary.scanned, summary.succeeded), (1, 1));
    assert_eq!(
        stored(&repository, "job-1").await.state,
        DreamState::Succeeded
    );
    let status = orchestrator.status();
    assert_eq!(status.ticks, 1);
    assert_eq!(status.consecutive_empty_sweeps, 0);
    assert_eq!(status.next_interval_ms, config.poll_interval_ms);
    assert_eq!(status.last_tick_at, Some(now));
    assert_eq!(status.last_summary, Some(summary));
    assert!(status.last_retention.is_none());
    assert!(status.last_error.is_none());
}

#[tokio::test]
async fn an_empty_sweep_backs_off_up_to_the_cap() {
    let repository = repo();
    let consolidator = Arc::new(ScriptedConsolidator::new([Step::Succeeded]));
    let planner = planner(&repository, &consolidator, DreamConfig::default());
    let config = DreamOrchestratorConfig::default()
        .with_poll_interval_ms(100)
        .with_initial_delay_ms(0)
        .with_max_interval_ms(500)
        .with_backoff_multiplier(2.0)
        .with_max_backoff_steps(3);
    let orchestrator = orchestrator(&planner, &repository, config.clone());

    // Five empty sweeps: 100 -> 200 -> 400 -> 500 (cap) -> 500.
    for (index, expected) in [200_u64, 400, 500, 500, 500].into_iter().enumerate() {
        let summary = orchestrator.tick(ts(10)).await.unwrap();
        assert!(summary.is_empty());
        let status = orchestrator.status();
        assert_eq!(status.ticks as usize, index + 1);
        assert_eq!(status.consecutive_empty_sweeps as usize, index + 1);
        assert_eq!(status.next_interval_ms, expected);
        assert_eq!(
            status.next_interval_ms,
            config.interval_for(index as u32 + 1).as_millis() as u64
        );
    }

    // A sweep that claims a job resets the count and the interval.
    repository.enqueue(job("job-1")).await.unwrap();
    let summary = orchestrator.tick(ts(10)).await.unwrap();
    assert_eq!(summary.scanned, 1);
    let status = orchestrator.status();
    assert_eq!(status.consecutive_empty_sweeps, 0);
    assert_eq!(status.next_interval_ms, config.poll_interval_ms);
    assert_eq!(orchestrator.interval_for_empty_sweeps(1).as_millis(), 200);
}

#[test]
fn the_backoff_schedule_is_documented_by_interval_for() {
    let config = DreamOrchestratorConfig::default()
        .with_poll_interval_ms(100)
        .with_initial_delay_ms(0)
        .with_max_interval_ms(500)
        .with_backoff_multiplier(2.0)
        .with_max_backoff_steps(3);

    let schedule = (0..=6)
        .map(|empty| config.interval_for(empty).as_millis() as u64)
        .collect::<Vec<_>>();
    assert_eq!(schedule, vec![100, 200, 400, 500, 500, 500, 500]);
    assert_eq!(config.interval_for(0), config.poll_interval());
    assert_eq!(config.interval_for(10_000).as_millis(), 500);
    assert_eq!(config.poll_interval(), StdDuration::from_millis(100));
    assert_eq!(config.initial_delay(), StdDuration::from_millis(0));
    assert_eq!(config.max_interval(), StdDuration::from_millis(500));
}

#[test]
fn config_validation_rejects_every_out_of_range_knob() {
    let base = DreamOrchestratorConfig::default();
    assert!(base.validate().is_ok());
    let cases = [
        ("poll_interval_ms", base.clone().with_poll_interval_ms(0)),
        (
            "initial_delay_ms",
            base.clone().with_initial_delay_ms(600_001),
        ),
        (
            "max_interval_ms",
            base.clone()
                .with_poll_interval_ms(100)
                .with_initial_delay_ms(0)
                .with_max_interval_ms(50),
        ),
        (
            "backoff_multiplier",
            base.clone().with_backoff_multiplier(f32::NAN),
        ),
        (
            "backoff_multiplier",
            base.clone().with_backoff_multiplier(f32::INFINITY),
        ),
        (
            "backoff_multiplier",
            base.clone().with_backoff_multiplier(0.5),
        ),
        ("max_backoff_steps", base.clone().with_max_backoff_steps(17)),
        (
            "heartbeat",
            base.clone().with_lease(DreamLeaseOptions::new(
                "worker-a",
                ChronoDuration::seconds(10),
                StdDuration::from_secs(10),
            )),
        ),
        (
            "max_per_state",
            base.with_retention(DreamRetention::new(60).with_max_per_state(0)),
        ),
    ];
    for (needle, config) in cases {
        assert_validation(config.validate().unwrap_err(), needle);
    }
}

#[tokio::test]
async fn a_disabled_orchestrator_claims_nothing() {
    let repository = repo();
    let consolidator = Arc::new(ScriptedConsolidator::new([Step::Succeeded]));
    let planner = planner(&repository, &consolidator, DreamConfig::default());
    let orchestrator = orchestrator(
        &planner,
        &repository,
        DreamOrchestratorConfig::default().with_enabled(false),
    );

    repository.enqueue(job("job-1")).await.unwrap();
    let summary = orchestrator.tick(ts(10)).await.unwrap();
    assert!(summary.is_empty());
    assert_eq!(consolidator.calls(), Vec::<String>::new());
    let untouched = stored(&repository, "job-1").await;
    assert_eq!(untouched.state, DreamState::Pending);
    assert_eq!(untouched.attempts, 0);
    let status = orchestrator.status();
    assert_eq!(status.ticks, 0);
    assert!(status.last_tick_at.is_none());
    assert!(status.last_summary.is_none());

    let looped = orchestrator
        .run_until_cancelled(CancellationToken::new())
        .await
        .unwrap();
    assert!(looped.is_empty());
    assert_eq!(orchestrator.status().ticks, 0);
}

#[tokio::test]
async fn retention_runs_on_tick_and_is_idempotent() {
    let repository = repo();
    let consolidator = Arc::new(ScriptedConsolidator::new([Step::Succeeded]));
    let planner = planner(&repository, &consolidator, DreamConfig::default());
    let orchestrator = orchestrator(
        &planner,
        &repository,
        DreamOrchestratorConfig::default().with_retention(DreamRetention::new(60)),
    );

    complete_one(&repository, "job-1").await;
    // `complete` stamps the real clock, so the purge clock has to be past it.
    let now = Utc::now() + ChronoDuration::hours(2);

    let first = orchestrator.tick(now).await.unwrap();
    assert!(first.is_empty(), "the queue has nothing claimable");
    let report = orchestrator
        .status()
        .last_retention
        .expect("retention ran on the tick");
    assert!(report.deleted_expired >= 1, "{report:?}");
    assert_eq!(report.total(), 1);
    assert!(repository.get("job-1").await.unwrap().is_none());

    let second = orchestrator.tick(now).await.unwrap();
    assert!(second.is_empty());
    let status = orchestrator.status();
    assert!(status.last_retention.expect("retention ran").is_empty());
    assert!(status.last_error.is_none());
    assert_eq!(status.ticks, 2);
}

#[tokio::test]
async fn retention_absent_keeps_finished_jobs() {
    let repository = repo();
    let consolidator = Arc::new(ScriptedConsolidator::new([Step::Succeeded]));
    let planner = planner(&repository, &consolidator, DreamConfig::default());
    let orchestrator = orchestrator(&planner, &repository, DreamOrchestratorConfig::default());

    complete_one(&repository, "job-1").await;

    orchestrator
        .tick(Utc::now() + ChronoDuration::hours(2))
        .await
        .unwrap();
    let status = orchestrator.status();
    assert!(status.last_retention.is_none());
    assert!(status.last_error.is_none());
    assert!(repository.get("job-1").await.unwrap().is_some());
}

#[tokio::test]
async fn a_retention_error_does_not_fail_the_tick() {
    let repository: Arc<dyn DreamRepository> = Arc::new(DoubleRepository::failing_delete());
    let consolidator = Arc::new(ScriptedConsolidator::new([Step::Succeeded]));
    let planner = planner(&repository, &consolidator, DreamConfig::default());
    let orchestrator = orchestrator(
        &planner,
        &repository,
        DreamOrchestratorConfig::default()
            .with_retention(DreamRetention::new(0).with_max_per_state(1)),
    );
    for id in ["job-1", "job-2"] {
        let mut finished = job(id);
        finished.state = DreamState::Succeeded;
        finished.attempts = 1;
        finished.started_at = Some(ts(1));
        finished.finished_at = Some(ts(2));
        finished.updated_at = ts(2);
        repository.enqueue(finished).await.unwrap();
    }

    let summary = orchestrator.tick(ts(10)).await.unwrap();
    assert!(summary.is_empty());
    let status = orchestrator.status();
    assert!(
        status
            .last_error
            .as_deref()
            .is_some_and(|error| error.contains("scripted delete failure")),
        "{:?}",
        status.last_error
    );
    assert!(status.last_retention.is_none());
    assert_eq!(status.ticks, 1);
}

#[tokio::test]
async fn a_leased_tick_uses_the_leased_path() {
    let repository = repo();
    let consolidator = Arc::new(ScriptedConsolidator::new([Step::Succeeded]));
    let planner = planner(&repository, &consolidator, DreamConfig::default());
    let orchestrator = orchestrator(
        &planner,
        &repository,
        DreamOrchestratorConfig::default().with_lease(lease_options("worker-a", 30)),
    );

    repository.enqueue(job("job-1")).await.unwrap();
    let summary = orchestrator.tick(ts(10)).await.unwrap();
    assert_eq!(summary.succeeded, 1);
    assert_eq!(
        stored(&repository, "job-1").await.state,
        DreamState::Succeeded
    );
    assert!(
        repository.lease("job-1").await.unwrap().is_none(),
        "the lease must be released"
    );
    assert!(orchestrator.status().last_error.is_none());
}

#[tokio::test]
async fn a_tick_against_a_non_leasing_repository_reports_unsupported() {
    let repository: Arc<dyn DreamRepository> = Arc::new(DoubleRepository::plain());
    let consolidator = Arc::new(ScriptedConsolidator::new([Step::Succeeded]));
    let planner = planner(&repository, &consolidator, DreamConfig::default());
    let orchestrator = orchestrator(
        &planner,
        &repository,
        DreamOrchestratorConfig::default().with_lease(lease_options("worker-a", 30)),
    );
    repository.enqueue(job("job-1")).await.unwrap();

    let error = orchestrator.tick(ts(10)).await.unwrap_err();
    assert!(matches!(error, Error::Unsupported(_)), "{error:?}");
    let status = orchestrator.status();
    assert_eq!(status.ticks, 1);
    assert_eq!(status.last_tick_at, Some(ts(10)));
    assert!(status.last_error.is_some());
    assert_eq!(
        stored(&repository, "job-1").await.state,
        DreamState::Pending
    );
    assert_eq!(consolidator.calls(), Vec::<String>::new());
}

#[tokio::test]
async fn two_orchestrators_on_one_repository_only_one_runs_the_job() {
    let repository = repo();
    let consolidator = Arc::new(ScriptedConsolidator::new([Step::Block]));
    let planner = planner(&repository, &consolidator, DreamConfig::default());
    let left = Arc::new(orchestrator(
        &planner,
        &repository,
        DreamOrchestratorConfig::default().with_lease(lease_options("worker-a", 30)),
    ));
    let right = Arc::new(orchestrator(
        &planner,
        &repository,
        DreamOrchestratorConfig::default().with_lease(lease_options("worker-b", 30)),
    ));
    repository.enqueue(job("job-1")).await.unwrap();

    let now = ts(10);
    let ctx = CallOptions::new();
    let (first, second, ()) = tokio::join!(
        left.tick_with(now, &ctx),
        right.tick_with(now, &ctx),
        async {
            consolidator.entered.notified().await;
            consolidator.release.notify_one();
        },
    );
    let first = first.unwrap();
    let second = second.unwrap();

    assert_eq!(
        first.scanned + second.scanned,
        1,
        "exactly one sweep claims the job"
    );
    assert!(
        first.is_empty() || second.is_empty(),
        "the other sweep claims nothing"
    );
    assert_eq!(consolidator.calls(), vec!["job-1".to_owned()]);
    let completed = stored(&repository, "job-1").await;
    assert_eq!(completed.state, DreamState::Succeeded);
    assert!(repository.lease("job-1").await.unwrap().is_none());
}

#[tokio::test]
async fn an_expired_lease_lets_a_second_tick_take_over() {
    let repository = repo();
    let consolidator = Arc::new(ScriptedConsolidator::new([Step::Block, Step::Succeeded]));
    let planner = planner(&repository, &consolidator, DreamConfig::default());
    let stale = Arc::new(orchestrator(
        &planner,
        &repository,
        DreamOrchestratorConfig::default().with_lease(lease_options("worker-a", 2)),
    ));
    let fresh = Arc::new(orchestrator(
        &planner,
        &repository,
        DreamOrchestratorConfig::default().with_lease(lease_options("worker-b", 30)),
    ));
    repository.enqueue(job("job-1")).await.unwrap();

    let ctx = CallOptions::new();
    let running = Arc::clone(&stale);
    let handle = tokio::spawn(async move { running.tick_with(ts(10), &ctx).await });
    consolidator.entered.notified().await;

    // Past the first lease's expiry (ts(12)), so the takeover path applies.
    let second = fresh.tick(ts(13)).await.unwrap();
    assert_eq!((second.scanned, second.succeeded), (1, 1));
    assert_eq!(
        stored(&repository, "job-1").await.state,
        DreamState::Succeeded
    );

    consolidator.release.notify_one();
    let stale_summary = tokio::time::timeout(BOUND, handle)
        .await
        .expect("the stale tick finishes")
        .unwrap()
        .unwrap();
    assert!(
        stale_summary.is_empty(),
        "a lost lease must not record a transition"
    );
    let after = stored(&repository, "job-1").await;
    assert_eq!(after.state, DreamState::Succeeded);
    assert_eq!(
        after.attempts, 2,
        "the takeover counted as a second attempt"
    );
    assert!(repository.lease("job-1").await.unwrap().is_none());
}

#[tokio::test]
async fn run_until_cancelled_returns_cancelled_promptly() {
    let repository = repo();
    let consolidator = Arc::new(ScriptedConsolidator::new([Step::Succeeded]));
    let planner = planner(&repository, &consolidator, DreamConfig::default());
    let orchestrator = Arc::new(orchestrator(
        &planner,
        &repository,
        DreamOrchestratorConfig::default()
            .with_initial_delay_ms(0)
            .with_poll_interval_ms(5),
    ));
    repository.enqueue(job("job-1")).await.unwrap();

    let token = CancellationToken::new();
    let running = Arc::clone(&orchestrator);
    let stop = token.clone();
    let handle = tokio::spawn(async move { running.run_until_cancelled(stop).await });

    wait_for_ticks(&orchestrator, 1).await;
    token.cancel();
    let error = tokio::time::timeout(BOUND, handle)
        .await
        .expect("the loop stops promptly")
        .unwrap()
        .unwrap_err();
    assert!(matches!(error, Error::Cancelled), "{error:?}");
    assert_eq!(
        stored(&repository, "job-1").await.state,
        DreamState::Succeeded
    );
    assert!(orchestrator.status().ticks >= 1);
}

#[tokio::test]
async fn cancellation_mid_consolidation_leaves_no_running_job() {
    let repository = repo();
    let consolidator = Arc::new(ScriptedConsolidator::new([Step::Block]));
    let planner = planner(&repository, &consolidator, DreamConfig::default());
    let orchestrator = Arc::new(orchestrator(
        &planner,
        &repository,
        DreamOrchestratorConfig::default()
            .with_initial_delay_ms(0)
            .with_poll_interval_ms(5),
    ));
    repository.enqueue(job("job-1")).await.unwrap();

    let token = CancellationToken::new();
    let running = Arc::clone(&orchestrator);
    let stop = token.clone();
    let handle = tokio::spawn(async move { running.run_until_cancelled(stop).await });
    consolidator.entered.notified().await;
    token.cancel();

    let error = tokio::time::timeout(BOUND, handle)
        .await
        .expect("the loop stops promptly")
        .unwrap()
        .unwrap_err();
    assert!(matches!(error, Error::Cancelled), "{error:?}");
    let cancelled = stored(&repository, "job-1").await;
    assert_eq!(cancelled.state, DreamState::Cancelled);
    assert!(cancelled.finished_at.is_some());
}

#[test]
fn debug_and_status_are_inspectable() {
    let repository = repo();
    let consolidator = Arc::new(ScriptedConsolidator::new([Step::Succeeded]));
    let planner = planner(&repository, &consolidator, DreamConfig::default());
    let orchestrator = orchestrator(&planner, &repository, DreamOrchestratorConfig::default());

    let debug = format!("{orchestrator:?}");
    for needle in [
        "DreamOrchestrator",
        "poll_interval_ms",
        "initial_delay_ms",
        "max_backoff_steps",
        "status",
        "..",
    ] {
        assert!(
            debug.contains(needle),
            "`{debug}` should contain `{needle}`"
        );
    }

    let status = orchestrator.status();
    assert_eq!(status, DreamOrchestratorStatus::default());
    assert_eq!(status.ticks, 0);
    assert!(status.last_tick_at.is_none());
    let json = serde_json::to_string(&status).unwrap();
    let parsed: DreamOrchestratorStatus = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed, status);
}

#[test]
fn config_serde_round_trips_with_the_lease_in_milliseconds() {
    let config = DreamOrchestratorConfig::default()
        .with_poll_interval_ms(1_000)
        .with_retention(DreamRetention::new(30))
        .with_lease(lease_options("worker-a", 30));
    let json = serde_json::to_string(&config).unwrap();
    assert!(json.contains("\"owner\":\"worker-a\""), "{json}");
    assert!(json.contains("\"ttl\":30000"), "{json}");
    let parsed: DreamOrchestratorConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed, config);

    let partial: DreamOrchestratorConfig =
        serde_json::from_str(r#"{"poll_interval_ms": 1000}"#).unwrap();
    assert_eq!(partial.poll_interval_ms, 1_000);
    assert_eq!(partial.max_interval_ms, 600_000);
    assert!(partial.lease.is_none());
    assert!(partial.retention.is_none());
}

#[tokio::test]
async fn without_a_sweep_lock_a_tick_always_sweeps() {
    let repository = repo();
    let consolidator = Arc::new(ScriptedConsolidator::new([Step::Succeeded]));
    let planner = planner(&repository, &consolidator, DreamConfig::default());
    let orchestrator = orchestrator(&planner, &repository, DreamOrchestratorConfig::default());
    assert!(orchestrator.sweep_lock().is_none());

    repository.enqueue(job("job-1")).await.unwrap();
    let summary = orchestrator.tick(ts(10)).await.unwrap();

    assert_eq!(summary.scanned, 1);
    assert!(summary.lock_skip.is_none());
    assert!(!summary.is_lock_skip());
    let status = orchestrator.status();
    assert_eq!((status.ticks, status.lock_skips), (1, 0));
    assert!(status.last_skip_reason.is_none());
    assert_eq!(
        stored(&repository, "job-1").await.state,
        DreamState::Succeeded
    );
}

#[tokio::test]
async fn a_held_sweep_lock_skips_the_sweep_and_the_retention_pass() {
    let repository = repo();
    let consolidator = Arc::new(ScriptedConsolidator::new([Step::Succeeded]));
    let planner = planner(&repository, &consolidator, DreamConfig::default());
    let lock: Arc<dyn DistributedKvLock> = Arc::new(InProcessKvLock::new());
    let held = lock
        .acquire(DREAM_SWEEP_LOCK_NAME, &LockOptions::new("other-worker"))
        .await
        .unwrap();
    let orchestrator = orchestrator(
        &planner,
        &repository,
        DreamOrchestratorConfig::default().with_retention(DreamRetention::new(60)),
    )
    .with_sweep_lock(
        Arc::clone(&lock),
        LockOptions::new("worker-a").with_timeout(StdDuration::from_millis(50)),
    );
    assert!(orchestrator.sweep_lock().is_some());

    repository.enqueue(job("job-1")).await.unwrap();
    complete_one(&repository, "finished").await;

    // A clock past the retention window, so a tick that swept would purge `finished`.
    let now = Utc::now() + ChronoDuration::days(1);
    let summary = orchestrator.tick(now).await.unwrap();

    assert_eq!(summary.scanned, 0);
    assert!(summary.is_empty());
    assert!(summary.is_lock_skip());
    let skip = summary
        .lock_skip
        .clone()
        .expect("the summary carries the skip marker");
    assert_eq!(skip.kind, SweepLockSkipKind::Held);
    assert!(skip.is_held());
    assert_eq!(skip.lock, DREAM_SWEEP_LOCK_NAME);
    assert_eq!(skip.backend, "in-process");
    assert!(
        skip.reason.contains(DREAM_SWEEP_LOCK_NAME),
        "{}",
        skip.reason
    );

    // Nothing was claimed, consolidated, or purged, and the skip is recorded without
    // counting as a tick.
    let pending = stored(&repository, "job-1").await;
    assert_eq!(pending.state, DreamState::Pending);
    assert_eq!(pending.attempts, 0);
    assert_eq!(consolidator.calls(), Vec::<String>::new());
    assert!(
        repository.get("finished").await.unwrap().is_some(),
        "a skipped tick must not run retention"
    );
    let status = orchestrator.status();
    assert_eq!(status.lock_skips, 1);
    assert_eq!(status.ticks, 0);
    assert!(status.last_tick_at.is_none());
    assert!(status.last_summary.is_none());
    assert!(status.last_retention.is_none());
    assert!(status.last_error.is_none());
    let reason = status.last_skip_reason.expect("the reason is recorded");
    assert!(reason.contains(DREAM_SWEEP_LOCK_NAME), "{reason}");
    assert!(reason.contains("held"), "{reason}");

    // Control: once the other worker releases, the same tick sweeps, purges, and clears
    // nothing of the skip history.
    held.release();
    repository.enqueue(job("job-2")).await.unwrap();
    let summary = orchestrator.tick(now).await.unwrap();
    assert_eq!(
        (summary.scanned, summary.succeeded),
        (2, 2),
        "both jobs were still pending for the first real sweep"
    );
    assert!(!summary.is_lock_skip());
    let status = orchestrator.status();
    assert_eq!(status.ticks, 1);
    assert_eq!(status.lock_skips, 1);
    let report = status
        .last_retention
        .expect("the retention pass was live once the lock was free");
    assert!(report.deleted_expired >= 1, "{report:?}");
    assert!(repository.get("finished").await.unwrap().is_none());
}

#[tokio::test]
async fn the_sweep_lock_is_released_at_the_end_of_every_tick() {
    let repository = repo();
    let consolidator = Arc::new(ScriptedConsolidator::new([Step::Succeeded]));
    let planner = planner(&repository, &consolidator, DreamConfig::default());
    let concrete = InProcessKvLock::new();
    let lock: Arc<dyn DistributedKvLock> = Arc::new(concrete.clone());
    let orchestrator = orchestrator(&planner, &repository, DreamOrchestratorConfig::default())
        .with_sweep_lock(
            lock,
            LockOptions::new("worker-a").with_timeout(StdDuration::from_millis(50)),
        );

    repository.enqueue(job("job-1")).await.unwrap();
    let first = orchestrator.tick(ts(10)).await.unwrap();
    assert_eq!(first.scanned, 1);
    assert!(!first.is_lock_skip());
    assert!(
        concrete.held_names().is_empty(),
        "the tick releases the sweep lock before returning"
    );

    repository.enqueue(job("job-2")).await.unwrap();
    let second = orchestrator.tick(ts(11)).await.unwrap();
    assert_eq!(second.scanned, 1, "the next tick acquires the lock again");
    assert!(!second.is_lock_skip());
    assert_eq!(orchestrator.status().lock_skips, 0);
    assert!(orchestrator.status().last_skip_reason.is_none());
}

#[tokio::test]
async fn an_unsupported_lock_backend_skips_with_a_typed_reason() {
    let repository = repo();
    let consolidator = Arc::new(ScriptedConsolidator::new([Step::Succeeded]));
    let planner = planner(&repository, &consolidator, DreamConfig::default());
    let orchestrator = orchestrator(&planner, &repository, DreamOrchestratorConfig::default())
        .with_sweep_lock(
            Arc::new(UnsupportedKvLock::redis()),
            LockOptions::new("worker-a"),
        );

    repository.enqueue(job("job-1")).await.unwrap();
    let summary = orchestrator.tick(ts(10)).await.unwrap();

    assert_eq!(summary.scanned, 0);
    let skip = summary
        .lock_skip
        .expect("the summary carries the skip marker");
    assert_eq!(skip.kind, SweepLockSkipKind::Unsupported);
    assert!(skip.is_unsupported());
    assert_eq!(skip.backend, "redis");
    assert!(skip.reason.contains("redis"), "{}", skip.reason);
    assert_eq!(
        stored(&repository, "job-1").await.state,
        DreamState::Pending
    );
    let status = orchestrator.status();
    assert_eq!(status.lock_skips, 1);
    assert_eq!(status.ticks, 0);
}

#[tokio::test]
async fn unusable_sweep_lock_options_are_errors_rather_than_skips() {
    let repository = repo();
    let consolidator = Arc::new(ScriptedConsolidator::new([Step::Succeeded]));
    let planner = planner(&repository, &consolidator, DreamConfig::default());
    let orchestrator = orchestrator(&planner, &repository, DreamOrchestratorConfig::default())
        .with_sweep_lock(Arc::new(InProcessKvLock::new()), LockOptions::new(""));

    repository.enqueue(job("job-1")).await.unwrap();
    let error = orchestrator.tick(ts(10)).await.unwrap_err();
    assert_validation(error, "owner");
    assert_eq!(
        stored(&repository, "job-1").await.state,
        DreamState::Pending
    );
    let status = orchestrator.status();
    assert_eq!(status.lock_skips, 0);
    assert!(status.last_skip_reason.is_none());
}

#[tokio::test]
async fn a_disabled_orchestrator_never_touches_the_sweep_lock() {
    let repository = repo();
    let consolidator = Arc::new(ScriptedConsolidator::new([Step::Succeeded]));
    let planner = planner(&repository, &consolidator, DreamConfig::default());
    let lock: Arc<dyn DistributedKvLock> = Arc::new(InProcessKvLock::new());
    let held = lock
        .acquire(DREAM_SWEEP_LOCK_NAME, &LockOptions::new("other-worker"))
        .await
        .unwrap();
    let orchestrator = orchestrator(
        &planner,
        &repository,
        DreamOrchestratorConfig::default().with_enabled(false),
    )
    .with_sweep_lock(
        Arc::clone(&lock),
        LockOptions::new("worker-a").with_timeout(StdDuration::from_millis(50)),
    );

    repository.enqueue(job("job-1")).await.unwrap();
    let summary = orchestrator.tick(ts(10)).await.unwrap();

    assert!(summary.is_empty());
    assert!(!summary.is_lock_skip());
    let status = orchestrator.status();
    assert_eq!(status.lock_skips, 0);
    assert_eq!(status.ticks, 0);
    assert!(status.last_skip_reason.is_none());
    assert_eq!(
        stored(&repository, "job-1").await.state,
        DreamState::Pending
    );
    drop(held);
}
