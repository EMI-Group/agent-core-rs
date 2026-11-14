//! The runner-hosted dreaming driver: install, start, stop, cancellation, and
//! the typed absence contract.
//!
//! Everything here is offline and deterministic. The consolidator is a recording
//! double over an in-memory queue, the schedules use millisecond intervals, and
//! every wait is a bounded poll ([`eventually`], 10 s) rather than a wall-clock
//! sleep — the only `sleep` in this file is the observation window that proves no
//! second loop survived a stop.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use chrono::Utc;
use openjiuwen_common::error::{ErrorKind, Result};
use openjiuwen_memory::dreaming::{
    DreamJob, DreamKind, DreamMetrics, DreamOutcome, DreamRepository, DreamState, DreamTrigger,
    InMemoryDreamRepository,
};
use openjiuwen_memory::record::{CallOptions, MemoryScope};
use openjiuwen_runner::{
    DreamConfig, DreamConsolidator, DreamOrchestrator, DreamOrchestratorConfig, DreamPlanner,
    DreamService, Runner,
};
use openjiuwen_test_support::eventually;

/// Upper bound for every await in this suite.
const BOUND: Duration = Duration::from_secs(10);

/// A consolidator that records the job ids it was handed and succeeds.
#[derive(Debug, Default)]
struct RecordingConsolidator {
    calls: parking_lot::Mutex<Vec<String>>,
}

impl RecordingConsolidator {
    fn calls(&self) -> Vec<String> {
        self.calls.lock().clone()
    }
}

#[async_trait]
impl DreamConsolidator for RecordingConsolidator {
    async fn consolidate(&self, job: &DreamJob, _ctx: &CallOptions) -> Result<DreamOutcome> {
        self.calls.lock().push(job.id.clone());
        Ok(DreamOutcome::succeeded(
            job.id.clone(),
            DreamMetrics::default(),
        ))
    }
}

/// A schedule that sweeps immediately and then every millisecond.
fn fast_schedule() -> DreamOrchestratorConfig {
    DreamOrchestratorConfig::default()
        .with_initial_delay_ms(0)
        .with_poll_interval_ms(1)
        .with_max_interval_ms(5)
}

/// A schedule whose first sweep cannot come due within a test's lifetime.
fn slow_schedule() -> DreamOrchestratorConfig {
    DreamOrchestratorConfig::default()
        .with_initial_delay_ms(60_000)
        .with_poll_interval_ms(1_000)
        .with_max_interval_ms(60_000)
}

fn job(id: &str) -> DreamJob {
    DreamJob::new(
        MemoryScope::new("alice", "ws"),
        DreamKind::Consolidation,
        DreamTrigger::Manual,
        Utc::now(),
    )
    .with_id(id)
}

/// A service over an explicit in-memory queue, plus the handles a test inspects.
fn service(
    schedule: DreamOrchestratorConfig,
) -> (
    Arc<DreamService>,
    Arc<dyn DreamRepository>,
    Arc<RecordingConsolidator>,
) {
    let consolidator = Arc::new(RecordingConsolidator::default());
    let port: Arc<dyn DreamConsolidator> = consolidator.clone();
    let repository: Arc<dyn DreamRepository> = Arc::new(InMemoryDreamRepository::new());
    let planner =
        Arc::new(DreamPlanner::new(Arc::clone(&repository), port, DreamConfig::default()).unwrap());
    let orchestrator =
        Arc::new(DreamOrchestrator::new(planner, Arc::clone(&repository), schedule).unwrap());
    (DreamService::new(orchestrator), repository, consolidator)
}

async fn stored(repository: &Arc<dyn DreamRepository>, id: &str) -> DreamJob {
    repository
        .get(id)
        .await
        .unwrap()
        .expect("the job is stored")
}

/// Polls until the installed service reports at least `ticks` orchestrator ticks.
async fn wait_for_ticks(runner: &Runner, ticks: u64) {
    eventually(BOUND, || async {
        runner.dreaming_status().unwrap().orchestrator.ticks >= ticks
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn a_host_driven_tick_consolidates_every_queued_job() {
    let (service, repository, consolidator) = service(fast_schedule());
    repository.enqueue(job("job-1")).await.unwrap();
    repository.enqueue(job("job-2")).await.unwrap();

    assert!(!service.is_running(), "ticking spawns no task");
    let summary = service.tick(Utc::now()).await.unwrap();
    assert_eq!(summary.scanned, 2);
    assert_eq!(summary.succeeded, 2);
    assert_eq!(consolidator.calls().len(), 2);
    for id in ["job-1", "job-2"] {
        assert_eq!(stored(&repository, id).await.state, DreamState::Succeeded);
    }

    let status = service.status();
    assert!(!status.running);
    assert_eq!(status.orchestrator.ticks, 1);
    assert_eq!(
        status.orchestrator.last_summary.as_ref().unwrap().succeeded,
        2
    );
    assert!(
        status.orchestrator.last_tick_at.is_some(),
        "the tick stamped its clock"
    );
    assert!(status.last_stop_error.is_none());
}

#[tokio::test]
async fn the_runner_starts_and_stops_the_installed_service() {
    let (service, repository, _consolidator) = service(fast_schedule());
    repository.enqueue(job("job-1")).await.unwrap();
    let runner = Runner::new();
    assert!(runner
        .set_dreaming_service(Some(Arc::clone(&service)))
        .is_none());

    let started = runner.start_dreaming().await.unwrap();
    assert!(started.running, "the loop is live after start");
    wait_for_ticks(&runner, 1).await;
    assert_eq!(
        stored(&repository, "job-1").await.state,
        DreamState::Succeeded
    );

    let stopped = runner.stop_dreaming().await.unwrap();
    assert!(!stopped.running);
    assert!(
        stopped.last_stop_error.is_none(),
        "cancellation is the path"
    );
    assert!(!service.is_running());

    let again = runner.stop_dreaming().await.unwrap();
    assert!(!again.running, "a second stop is Ok and idempotent");
    assert!(again.last_stop_error.is_none());
}

#[tokio::test]
async fn stopping_interrupts_the_initial_delay() {
    let (service, _repository, _consolidator) = service(slow_schedule());
    let runner = Runner::new();
    runner.set_dreaming_service(Some(Arc::clone(&service)));
    let started = runner.start_dreaming().await.unwrap();
    assert!(started.running);

    let stopped = tokio::time::timeout(BOUND, runner.stop_dreaming())
        .await
        .expect("stop returns promptly while the loop waits")
        .unwrap();
    assert!(!stopped.running);
    assert_eq!(stopped.orchestrator.ticks, 0, "the first sweep never ran");
    assert!(stopped.last_stop_error.is_none());
    assert!(!service.is_running());
}

#[tokio::test]
async fn lifecycle_calls_without_a_service_are_config_errors() {
    let runner = Runner::new();
    let errors = [
        runner.start_dreaming().await.unwrap_err(),
        runner.stop_dreaming().await.unwrap_err(),
        runner.dreaming_status().unwrap_err(),
    ];
    for error in errors {
        assert_eq!(error.kind(), ErrorKind::Config, "got {error:?}");
        assert!(
            error.to_string().contains("set_dreaming_service"),
            "`{error}` should name the fix"
        );
    }
    assert!(runner.dreaming_service().is_none());
}

#[tokio::test]
async fn clearing_the_slot_returns_the_previous_service() {
    let (service, _repository, _consolidator) = service(fast_schedule());
    let runner = Runner::new();
    assert!(runner
        .set_dreaming_service(Some(Arc::clone(&service)))
        .is_none());
    assert!(runner
        .dreaming_service()
        .is_some_and(|installed| Arc::ptr_eq(&installed, &service)));

    let previous = runner
        .set_dreaming_service(None)
        .expect("the previously installed service is returned");
    assert!(Arc::ptr_eq(&previous, &service));
    assert!(runner.dreaming_service().is_none());
    assert_eq!(
        runner.dreaming_status().unwrap_err().kind(),
        ErrorKind::Config,
        "an empty slot is a typed absence, not a silent no-op"
    );
}

#[tokio::test]
async fn runner_stop_stops_a_running_dreaming_service() {
    let (service, repository, _consolidator) = service(fast_schedule());
    repository.enqueue(job("job-1")).await.unwrap();
    let runner = Runner::new();
    runner.set_dreaming_service(Some(Arc::clone(&service)));
    runner.start_dreaming().await.unwrap();
    wait_for_ticks(&runner, 1).await;

    runner.stop().await.unwrap();
    assert!(!service.is_running());
    let status = runner.dreaming_status().unwrap();
    assert!(!status.running, "the runner's stop joined the loop");
    assert!(status.orchestrator.ticks >= 1);
    assert!(status.last_stop_error.is_none());

    // Stopping again is still Ok: dreaming never makes the runner's shutdown fail,
    // and the service stays installed for a later restart.
    runner.stop().await.unwrap();
    assert!(!runner.dreaming_status().unwrap().running);
    assert!(runner.dreaming_service().is_some());
}

#[tokio::test]
async fn a_started_loop_never_spawns_a_second_one() {
    let (service, repository, _consolidator) = service(fast_schedule());
    repository.enqueue(job("job-1")).await.unwrap();
    let runner = Runner::new();
    runner.set_dreaming_service(Some(Arc::clone(&service)));
    runner.start_dreaming().await.unwrap();
    wait_for_ticks(&runner, 1).await;

    let second = runner.start_dreaming().await.unwrap();
    assert!(second.running, "the live loop is reported, not restarted");
    runner.stop_dreaming().await.unwrap();
    assert!(!service.is_running());

    let frozen = runner.dreaming_status().unwrap().orchestrator.ticks;
    // An observation window, not synchronisation: once the only loop is joined no
    // tick can advance the counter, and a leaked second loop would advance it here.
    tokio::time::sleep(Duration::from_millis(40)).await;
    assert_eq!(
        runner.dreaming_status().unwrap().orchestrator.ticks,
        frozen,
        "no second loop survived the stop"
    );
}
