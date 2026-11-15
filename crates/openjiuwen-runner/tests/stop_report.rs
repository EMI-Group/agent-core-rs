//! The typed shutdown report: `Runner::stop_with_report` and the aggregated
//! `Runner::stop` error.
//!
//! Ports the intent of Python's `Runner.stop` (a shutdown that reports what it
//! could not stop) with the failure swallowing removed: a queue that refuses to
//! stop, a queue that misses its grace period, and a dreaming loop that ignores
//! its cancellation are all *reported*, and `stop()` aggregates every one of them
//! into a single typed error naming each failed component.
//!
//! Everything here is offline and deterministic. The failing queue backends are
//! local [`MessageQueue`] implementations, the overrunning dreaming loop blocks
//! inside an injected sweep lock the orchestrator cannot race against
//! cancellation, and every await is wrapped in a [`BOUND`] timeout so a
//! regression fails instead of hanging.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use openjiuwen_common::error::{Error, ErrorKind, Result};
use openjiuwen_memory::dreaming::{
    DreamConfig, DreamConsolidator, DreamOrchestrator, DreamOrchestratorConfig, DreamPlanner,
    DreamRepository, InMemoryDreamRepository, NoopConsolidator,
};
use openjiuwen_memory::store::kv_lock::{DistributedKvLock, KvLockGuard, LockOptions};
use openjiuwen_runner::{
    ComponentStop, DreamService, MessageHandler, MessageQueue, QueueMessage, Runner, RunnerConfig,
    StopComponent, StopReport, StopStatus,
};
use openjiuwen_test_support::eventually;
use serde_json::{json, Value};

/// Upper bound for every await in this suite.
const BOUND: Duration = Duration::from_secs(10);

/// A grace period short enough to prove a bound without slowing the suite down.
const SHORT_GRACE_SECS: f64 = 0.05;

// --------------------------------------------------------------- fixtures --

/// A queue whose `stop` fails with a broker refusal.
struct FailingStopQueue;

#[async_trait]
impl MessageQueue for FailingStopQueue {
    async fn start(&self) -> Result<()> {
        Ok(())
    }
    async fn stop(&self) -> Result<()> {
        Err(Error::Transport("broker refused the shutdown".to_owned()))
    }
    fn is_started(&self) -> bool {
        true
    }
    async fn subscribe(&self, _topic: &str) -> Result<()> {
        Ok(())
    }
    async fn unsubscribe(&self, _topic: &str) -> Result<()> {
        Ok(())
    }
    async fn add_handler(&self, _topic: &str, _handler: Arc<dyn MessageHandler>) -> Result<()> {
        Ok(())
    }
    async fn publish(&self, _topic: &str, _message: QueueMessage) -> Result<()> {
        Ok(())
    }
    async fn request(
        &self,
        _topic: &str,
        _message: QueueMessage,
        _timeout: Option<Duration>,
    ) -> Result<Value> {
        Ok(Value::Null)
    }
    fn topics(&self) -> Vec<String> {
        Vec::new()
    }
    fn has_subscriber(&self, _topic: &str) -> bool {
        false
    }
}

/// A queue whose `stop` never resolves — a backend that lost its broker.
struct HangingStopQueue {
    /// How many times `stop` was called, so the test proves the bound fired on
    /// the real call rather than on a path that never reached the backend.
    stops: AtomicUsize,
}

impl HangingStopQueue {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            stops: AtomicUsize::new(0),
        })
    }

    fn stops(&self) -> usize {
        self.stops.load(Ordering::Acquire)
    }
}

#[async_trait]
impl MessageQueue for HangingStopQueue {
    async fn start(&self) -> Result<()> {
        Ok(())
    }
    async fn stop(&self) -> Result<()> {
        self.stops.fetch_add(1, Ordering::AcqRel);
        std::future::pending::<Result<()>>().await
    }
    fn is_started(&self) -> bool {
        true
    }
    async fn subscribe(&self, _topic: &str) -> Result<()> {
        Ok(())
    }
    async fn unsubscribe(&self, _topic: &str) -> Result<()> {
        Ok(())
    }
    async fn add_handler(&self, _topic: &str, _handler: Arc<dyn MessageHandler>) -> Result<()> {
        Ok(())
    }
    async fn publish(&self, _topic: &str, _message: QueueMessage) -> Result<()> {
        Ok(())
    }
    async fn request(
        &self,
        _topic: &str,
        _message: QueueMessage,
        _timeout: Option<Duration>,
    ) -> Result<Value> {
        Ok(Value::Null)
    }
    fn topics(&self) -> Vec<String> {
        Vec::new()
    }
    fn has_subscriber(&self, _topic: &str) -> bool {
        false
    }
}

/// A sweep lock no dream tick can ever acquire.
///
/// `DreamOrchestrator::open_sweep_gate` awaits the backend's `acquire` before the
/// sweep, and nothing races that future against the loop's cancellation token, so
/// a `pending` acquisition is exactly the "loop that does not observe
/// cancellation" a stop bound has to fence off.
struct BlockedSweepLock {
    /// How many acquisitions were attempted.
    attempts: AtomicUsize,
}

impl BlockedSweepLock {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            attempts: AtomicUsize::new(0),
        })
    }

    fn attempts(&self) -> usize {
        self.attempts.load(Ordering::Acquire)
    }
}

#[async_trait]
impl DistributedKvLock for BlockedSweepLock {
    fn backend(&self) -> &'static str {
        "blocked-test"
    }
    async fn acquire(&self, _name: &str, _options: &LockOptions) -> Result<KvLockGuard> {
        self.attempts.fetch_add(1, Ordering::AcqRel);
        std::future::pending::<Result<KvLockGuard>>().await
    }
}

/// A schedule that sweeps immediately and then every millisecond.
fn fast_schedule() -> DreamOrchestratorConfig {
    DreamOrchestratorConfig::default()
        .with_initial_delay_ms(0)
        .with_poll_interval_ms(1)
        .with_max_interval_ms(5)
}

/// A [`DreamService`] whose loop cannot finish a tick: the sweep lock blocks.
fn overrun_service() -> (Arc<DreamService>, Arc<BlockedSweepLock>) {
    let consolidator: Arc<dyn DreamConsolidator> = Arc::new(NoopConsolidator);
    let repository: Arc<dyn DreamRepository> = Arc::new(InMemoryDreamRepository::new());
    let planner = Arc::new(
        DreamPlanner::new(
            Arc::clone(&repository),
            consolidator,
            DreamConfig::default(),
        )
        .unwrap(),
    );
    let lock = BlockedSweepLock::new();
    let orchestrator = Arc::new(
        DreamOrchestrator::new(planner, repository, fast_schedule())
            .unwrap()
            .with_sweep_lock(
                Arc::clone(&lock) as Arc<dyn DistributedKvLock>,
                LockOptions::new("t"),
            ),
    );
    (DreamService::new(orchestrator), lock)
}

/// A service whose loop waits out its (default, two-minute) settle delay: a clean
/// stop cancels that wait immediately.
fn idle_service() -> Arc<DreamService> {
    DreamService::with_in_memory(
        Arc::new(NoopConsolidator),
        DreamConfig::default(),
        DreamOrchestratorConfig::default(),
    )
    .unwrap()
}

/// A runner configured with the short shutdown grace period the bound tests use.
fn short_grace_runner() -> Arc<Runner> {
    Runner::with_config(RunnerConfig {
        shutdown_timeout_secs: SHORT_GRACE_SECS,
        ..RunnerConfig::default()
    })
}

/// A started runner whose installed queue fails to stop.
async fn runner_with_failing_queue() -> Arc<Runner> {
    let runner = short_grace_runner();
    runner.start().await.unwrap();
    runner.set_message_queue(Arc::new(FailingStopQueue) as Arc<dyn MessageQueue>);
    runner
}

/// A started runner whose installed queue never stops, plus that queue.
async fn runner_with_hanging_queue() -> (Arc<Runner>, Arc<HangingStopQueue>) {
    let runner = short_grace_runner();
    runner.start().await.unwrap();
    let queue = HangingStopQueue::new();
    runner.set_message_queue(Arc::clone(&queue) as Arc<dyn MessageQueue>);
    (runner, queue)
}

/// A runner with a started dreaming service whose loop cannot be cancelled.
async fn runner_with_overrunning_dreaming() -> (Arc<Runner>, Arc<DreamService>) {
    let (service, lock) = overrun_service();
    let runner = short_grace_runner();
    runner.set_dreaming_service(Some(Arc::clone(&service)));
    runner.start_dreaming().await.unwrap();
    eventually(BOUND, || async { lock.attempts() > 0 })
        .await
        .expect("the loop reached the blocking sweep lock");
    (runner, service)
}

// ------------------------------------------------------------------- tests --

/// A clean stop reports every component as `Stopped`, releases the registries,
/// and answers `Ok` through the plain `stop()` — whose report the JSON of a
/// `StopReport` round-trips.
#[tokio::test]
async fn a_clean_stop_reports_every_component() {
    let runner = Runner::new();
    let service = idle_service();
    runner.set_dreaming_service(Some(Arc::clone(&service)));
    runner.start_dreaming().await.unwrap();
    runner.start().await.unwrap();

    let report = tokio::time::timeout(BOUND, runner.stop_with_report())
        .await
        .expect("stop is bounded");
    assert!(report.started, "the runner was started");
    assert_eq!(report.queue.component, StopComponent::Queue);
    assert_eq!(report.queue.status, StopStatus::Stopped);
    assert!(report.queue.error.is_none());
    assert_eq!(report.pool.status, StopStatus::Stopped);
    assert_eq!(report.dreaming.status, StopStatus::Stopped);
    assert!(report.dreaming.error.is_none());
    assert!(
        report.registries_released,
        "a started stop releases the registries"
    );
    assert!(report.is_clean());
    assert_eq!(report.failures().len(), 0);
    assert!(report.failure_summary().is_none());
    assert!(!service.is_running(), "the dreaming loop was joined");
    assert!(!runner.is_started());
    assert!(runner.checkpointer().is_none());
    assert!(runner.resource_mgr().is_empty());

    // The wire shape is the contract: snake_case, complete, and tolerant of a
    // field a later build adds.
    let json = serde_json::to_value(&report).unwrap();
    assert_eq!(json["started"], json!(true));
    assert_eq!(json["registries_released"], json!(true));
    assert_eq!(json["queue"]["status"], json!("stopped"));
    assert_eq!(json["pool"]["component"], json!("pool"));
    assert_eq!(json["dreaming"]["status"], json!("stopped"));
    let mut extended = json.clone();
    extended["future_component"] = json!({"status": "failed"});
    let decoded: StopReport = serde_json::from_value(extended).unwrap();
    assert_eq!(decoded, report);

    // `stop()` keeps its signature and its `Ok` for the same clean shutdown.
    runner.start().await.unwrap();
    let plain = tokio::time::timeout(BOUND, runner.stop())
        .await
        .expect("stop is bounded");
    plain.expect("a clean stop is Ok");
    assert!(!runner.is_started());
}

/// A never-started runner has no queue to stop: that component is `skipped`, the
/// pool and dreaming are still reported, and a repeated stop stays clean.
#[tokio::test]
async fn a_never_started_runner_reports_a_skipped_queue() {
    let runner = Runner::new();
    let report = tokio::time::timeout(BOUND, runner.stop_with_report())
        .await
        .expect("stop is bounded");
    assert!(!report.started);
    assert_eq!(report.queue.status, StopStatus::Skipped);
    assert!(report.queue.error.is_none());
    assert_eq!(report.pool.status, StopStatus::Stopped);
    assert_eq!(report.dreaming.status, StopStatus::Skipped);
    assert!(
        !report.registries_released,
        "nothing was started to release"
    );
    assert!(report.is_clean());
    assert!(
        !runner.message_queue().is_started(),
        "a skipped queue is not stopped either"
    );

    assert!(tokio::time::timeout(BOUND, runner.stop())
        .await
        .expect("stop is bounded")
        .is_ok());
    let again = tokio::time::timeout(BOUND, runner.stop_with_report())
        .await
        .expect("stop is bounded");
    assert!(again.is_clean(), "idempotent: {again:?}");
    assert_eq!(again.queue.status, StopStatus::Skipped);
    assert!(tokio::time::timeout(BOUND, runner.stop())
        .await
        .expect("stop is bounded")
        .is_ok());
}

/// A queue that refuses to stop is a failed component, and `stop()` reports it
/// with the backend's own message.
#[tokio::test]
async fn a_failing_queue_stop_is_reported_and_aggregated() {
    let runner = runner_with_failing_queue().await;
    let report = tokio::time::timeout(BOUND, runner.stop_with_report())
        .await
        .expect("stop is bounded");
    assert!(report.started);
    assert_eq!(report.queue.status, StopStatus::Failed);
    let message = report.queue.error.as_deref().expect("a reason is recorded");
    assert!(
        message.contains("broker refused the shutdown"),
        "`{message}` should carry the backend's error"
    );
    assert_eq!(report.pool.status, StopStatus::Stopped);
    assert!(!report.is_clean());
    assert_eq!(report.failures().len(), 1);
    assert!(
        report.registries_released,
        "the rest of the teardown still ran"
    );

    let runner = runner_with_failing_queue().await;
    let error = tokio::time::timeout(BOUND, runner.stop())
        .await
        .expect("stop is bounded")
        .expect_err("a failed queue stop is not Ok");
    assert_eq!(error.kind(), ErrorKind::Config, "got {error:?}");
    let text = error.to_string();
    assert!(text.contains("queue"), "`{text}` should name the queue");
    assert!(
        text.contains("broker refused the shutdown"),
        "`{text}` should carry the underlying message"
    );
}

/// A queue that never answers is bounded by the configured grace period, and the
/// expired bound is a reported failure naming it.
#[tokio::test]
async fn a_queue_that_never_stops_hits_the_configured_bound() {
    let (runner, queue) = runner_with_hanging_queue().await;
    let report = tokio::time::timeout(BOUND, runner.stop_with_report())
        .await
        .expect("the bound keeps the stop finite");
    assert_eq!(queue.stops(), 1, "the backend's stop was really called");
    assert_eq!(report.queue.status, StopStatus::Failed);
    let message = report.queue.error.as_deref().expect("a reason is recorded");
    assert!(
        message.contains("shutdown grace period"),
        "`{message}` should name the bound"
    );
    assert!(message.contains("50ms"), "`{message}` should quote it");
    assert!(!report.is_clean());

    let (runner, _queue) = runner_with_hanging_queue().await;
    let error = tokio::time::timeout(BOUND, runner.stop())
        .await
        .expect("stop is bounded")
        .expect_err("an expired grace period is not Ok");
    assert_eq!(error.kind(), ErrorKind::Config, "got {error:?}");
    let text = error.to_string();
    assert!(text.contains("queue"), "`{text}` should name the queue");
    assert!(
        text.contains("shutdown grace period"),
        "`{text}` should name the bound"
    );
}

/// A dream loop that ignores its cancellation is a failed component, and the
/// fresh pool is installed regardless.
#[tokio::test]
async fn a_dreaming_overrun_is_reported_and_the_pool_is_replaced() {
    let (runner, service) = runner_with_overrunning_dreaming().await;
    let report = tokio::time::timeout(BOUND, runner.stop_with_report())
        .await
        .expect("the bound keeps the stop finite");
    assert_eq!(report.dreaming.component, StopComponent::Dreaming);
    assert_eq!(report.dreaming.status, StopStatus::Failed);
    let message = report
        .dreaming
        .error
        .as_deref()
        .expect("a reason is recorded");
    assert!(
        message.contains("did not stop within 50ms"),
        "`{message}` should quote the configured bound"
    );
    let recorded = service
        .status()
        .last_stop_error
        .expect("the overrun is recorded on the service");
    assert_eq!(
        message, recorded,
        "the report surfaces the service's own verdict"
    );
    assert_eq!(report.pool.status, StopStatus::Stopped);
    assert_eq!(report.queue.status, StopStatus::Skipped);
    assert!(!report.is_clean());
    assert_eq!(report.failures().len(), 1);

    // The sweep still installed a fresh, open pool: a later run can be admitted.
    let pool = runner.team_pool();
    assert!(pool.is_empty(), "the swept pool holds no runs");
    assert!(!pool.is_closed(), "the fresh pool is open");
    assert!(!runner.is_started());

    // A repeat is idempotent: it stops nothing new, keeps the pool open, and
    // reports the verdict the service still carries (its next `start` clears it).
    let again = tokio::time::timeout(BOUND, runner.stop_with_report())
        .await
        .expect("stop is bounded");
    assert_eq!(again.pool.status, StopStatus::Stopped);
    assert_eq!(again.queue.status, StopStatus::Skipped);
    assert_eq!(again.dreaming.status, StopStatus::Failed);
    assert_eq!(again.dreaming.error.as_deref(), Some(message));
    assert!(!runner.team_pool().is_closed());
    assert!(!runner.is_started());

    let (runner, _service) = runner_with_overrunning_dreaming().await;
    let error = tokio::time::timeout(BOUND, runner.stop())
        .await
        .expect("stop is bounded")
        .expect_err("an unjoinable loop is not Ok");
    let text = error.to_string();
    assert!(text.contains("dreaming"), "`{text}` should name dreaming");
    assert!(
        text.contains("did not stop within 50ms"),
        "`{text}` should quote the bound"
    );
}

/// The additive `DreamService::stop_with_timeout` bound is what the runner uses:
/// a host (and this suite) can prove the overrun path in milliseconds.
#[tokio::test]
async fn stop_with_timeout_bounds_the_dreaming_join() {
    let (service, lock) = overrun_service();
    service.start().await.unwrap();
    eventually(BOUND, || async { lock.attempts() > 0 })
        .await
        .expect("the loop reached the blocking sweep lock");
    assert!(service.is_running());

    let status = tokio::time::timeout(BOUND, service.stop_with_timeout(Duration::from_millis(30)))
        .await
        .expect("the bound keeps the stop finite")
        .unwrap();
    assert!(!status.running, "the task was aborted");
    let message = status.last_stop_error.expect("the overrun is recorded");
    assert!(
        message.contains("did not stop within 30ms"),
        "`{message}` should quote the caller's bound"
    );

    // Idempotent second stop: nothing is running, the verdict stands.
    let again = tokio::time::timeout(BOUND, service.stop())
        .await
        .expect("stop is bounded")
        .unwrap();
    assert!(!again.running);
    assert_eq!(again.last_stop_error.as_deref(), Some(message.as_str()));
}

/// A stop with two broken components names both of them in the aggregated error.
#[tokio::test]
async fn a_multi_failure_stop_names_every_component() {
    let (runner, service, _lock) = multi_failure_runner().await;
    let report = tokio::time::timeout(BOUND, runner.stop_with_report())
        .await
        .expect("both bounds keep the stop finite");
    assert_eq!(report.queue.status, StopStatus::Failed);
    assert_eq!(report.dreaming.status, StopStatus::Failed);
    assert_eq!(report.pool.status, StopStatus::Stopped);
    let failures = report.failures();
    assert_eq!(failures.len(), 2);
    assert_eq!(failures[0].component, StopComponent::Queue);
    assert_eq!(failures[1].component, StopComponent::Dreaming);
    assert!(
        service.status().last_stop_error.is_some(),
        "the dreaming verdict is the service's own"
    );

    let (runner, _service, _lock) = multi_failure_runner().await;
    let error = tokio::time::timeout(BOUND, runner.stop())
        .await
        .expect("stop is bounded")
        .expect_err("two broken components are not Ok");
    assert_eq!(error.kind(), ErrorKind::Config, "got {error:?}");
    let text = error.to_string();
    for needle in [
        "queue",
        "dreaming",
        "shutdown grace period",
        "did not stop within 50ms",
    ] {
        assert!(text.contains(needle), "`{text}` should mention `{needle}`");
    }
}

/// A started runner with both a hanging queue and an overrunning dream loop.
async fn multi_failure_runner() -> (Arc<Runner>, Arc<DreamService>, Arc<BlockedSweepLock>) {
    let (service, lock) = overrun_service();
    let runner = short_grace_runner();
    runner.set_dreaming_service(Some(Arc::clone(&service)));
    runner.start_dreaming().await.unwrap();
    eventually(BOUND, || async { lock.attempts() > 0 })
        .await
        .expect("the loop reached the blocking sweep lock");
    runner.start().await.unwrap();
    let queue = HangingStopQueue::new();
    runner.set_message_queue(Arc::clone(&queue) as Arc<dyn MessageQueue>);
    (runner, service, lock)
}

/// The report's constructors are the same value the runner builds: a host can
/// assert against a hand-built expectation.
#[test]
fn a_hand_built_report_matches_the_component_vocabulary() {
    let expected = StopReport::new(true)
        .with_queue(ComponentStop::stopped(StopComponent::Queue))
        .with_pool(ComponentStop::stopped(StopComponent::Pool))
        .with_dreaming(ComponentStop::skipped(StopComponent::Dreaming))
        .with_registries_released(true);
    assert!(expected.is_clean());
    assert_eq!(expected.components()[2].component, StopComponent::Dreaming);
    assert_eq!(expected.failure_summary(), None, "skipped is not a failure");
    let failed = expected.with_dreaming(ComponentStop::failed(StopComponent::Dreaming, "boom"));
    assert_eq!(failed.failure_summary().as_deref(), Some("dreaming: boom"));
}
