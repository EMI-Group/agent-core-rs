//! The runner-hosted dreaming driver: one owned background loop over
//! [`DreamOrchestrator`].
//!
//! [`openjiuwen_memory::dreaming`] deliberately spawns nothing: a
//! [`DreamOrchestrator`] is a value that either ticks at a caller-supplied clock
//! ([`DreamOrchestrator::tick`]) or is driven by
//! [`DreamOrchestrator::run_until_cancelled`] on a task the caller created. This
//! module is that caller for the runner: a [`DreamService`] owns exactly one
//! loop task, hands it a [`CancellationToken`], and joins it in
//! [`DreamService::stop`].
//!
//! **Lifecycle.** A service is installed on a runner with
//! [`Runner::set_dreaming_service`](crate::Runner::set_dreaming_service) and
//! started/stopped through [`Runner::start_dreaming`](crate::Runner::start_dreaming)
//! and [`Runner::stop_dreaming`](crate::Runner::stop_dreaming); see
//! [`crate::runner_dreaming`] for the facade. Dreaming has its own lifecycle:
//! [`Runner::start`](crate::Runner::start) does **not** start it, and
//! [`Runner::stop`](crate::Runner::stop) does stop it (bounded, and never failing
//! the runner's own shutdown).
//!
//! **Task ownership.** `start` spawns at most one task: a second call while the
//! loop is live is a no-op that returns the current status, never a second loop.
//! The loop is joined by `stop`, which cancels the token first — cancellation is
//! the documented shutdown path, so the [`Error::Cancelled`] the loop reports is
//! mapped to `Ok`. Any *other* loop outcome (a task panic, or a loop that ended
//! early with an error) is recorded in [`DreamServiceStatus::last_stop_error`]
//! instead of being swallowed or panicking the caller.
//!
//! **A host with its own timer** calls [`DreamService::tick`], which delegates to
//! [`DreamOrchestrator::tick`] and does not touch the loop: the schedule belongs
//! to the caller, and the orchestrator records the tick either way.

use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, Utc};
use openjiuwen_common::error::Result;
use openjiuwen_memory::dreaming::{
    DreamConfig, DreamConsolidator, DreamOrchestrator, DreamOrchestratorConfig,
    DreamOrchestratorStatus, DreamPlanner, DreamRunSummary,
};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

/// `tracing` target every dreaming-service log line uses.
const LOG_TARGET: &str = "openjiuwen_runner::dreaming";

/// The longest [`DreamService::stop`] waits for the loop task to finish.
///
/// The loop observes its cancellation token before every wait and before every
/// claim, so a clean shutdown is immediate; the bound exists only so a
/// consolidator that ignores cancellation cannot block the caller forever.
pub const DEFAULT_DREAMING_STOP_TIMEOUT: Duration = Duration::from_secs(5);

/// A dreaming service's task slot and its last shutdown verdict.
#[derive(Default)]
struct ServiceState {
    /// The one loop task, while one is installed (finished or live).
    loop_task: Option<LoopTask>,
    /// Why the last `stop` did not join cleanly; cleared by the next `start`.
    last_stop_error: Option<String>,
}

/// One spawned [`DreamOrchestrator::run_until_cancelled`] loop.
struct LoopTask {
    /// Cancels the loop; `stop` fires it before joining.
    cancel: CancellationToken,
    /// The loop task itself, joined (bounded) by `stop`.
    join: JoinHandle<Result<DreamRunSummary>>,
}

impl LoopTask {
    /// Spawns the loop for `orchestrator` on the current tokio runtime.
    fn spawn(orchestrator: Arc<DreamOrchestrator>) -> Self {
        let cancel = CancellationToken::new();
        let token = cancel.clone();
        let join = tokio::spawn(async move { orchestrator.run_until_cancelled(token).await });
        Self { cancel, join }
    }

    /// Whether the loop is still running.
    fn is_live(&self) -> bool {
        !self.join.is_finished()
    }
}

/// What a [`DreamService`] is doing, and what its orchestrator has done.
///
/// One snapshot: the service-level facts ([`DreamServiceStatus::running`], the
/// last shutdown problem) next to the orchestrator's own counters
/// ([`DreamOrchestratorStatus`]) in the same value, so a host renders a single
/// view. JSON is `snake_case`, every field is optional on input (missing fields
/// default), and unknown fields are ignored.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct DreamServiceStatus {
    /// Whether the owned loop task is live; default `false`.
    ///
    /// A loop that ended on its own (a disabled schedule, or a task that
    /// panicked) reports `false` here even before anybody calls `stop`.
    pub running: bool,
    /// The orchestrator's schedule counters, last sweep, and last error.
    pub orchestrator: DreamOrchestratorStatus,
    /// Why the last [`DreamService::stop`] did not join the loop cleanly;
    /// default `None`.
    ///
    /// Only a task panic or a stop that hit [`DEFAULT_DREAMING_STOP_TIMEOUT`]
    /// (the task is then aborted) sets this; a cancelled loop — the documented
    /// shutdown path — does not. The next `start` clears it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_stop_error: Option<String>,
}

/// The runner-hosted background driver of a [`DreamOrchestrator`].
///
/// The service owns the one loop task; nothing else in the crate spawns it. Build
/// one with [`DreamService::new`] over an orchestrator the host configured (a
/// shared repository, a lease, a sweep lock, a retention policy), or with
/// [`DreamService::with_in_memory`] for the single-process shape every test uses.
///
/// ```
/// use std::sync::Arc;
/// use openjiuwen_memory::dreaming::{
///     DreamConfig, DreamOrchestrator, DreamOrchestratorConfig, DreamPlanner, NoopConsolidator,
/// };
/// use openjiuwen_runner::DreamService;
///
/// # async fn example() -> openjiuwen_common::error::Result<()> {
/// let planner = Arc::new(DreamPlanner::with_in_memory(
///     Arc::new(NoopConsolidator),
///     DreamConfig::default(),
/// )?);
/// let repository = Arc::clone(planner.repository());
/// let orchestrator = Arc::new(DreamOrchestrator::new(
///     planner,
///     repository,
///     DreamOrchestratorConfig::default(),
/// )?);
/// let service = DreamService::new(orchestrator);
/// assert!(!service.is_running());
/// service.start().await?;
/// service.stop().await?;
/// # Ok(())
/// # }
/// ```
pub struct DreamService {
    /// The orchestrator every tick sweeps through.
    orchestrator: Arc<DreamOrchestrator>,
    /// The task slot and the last shutdown verdict.
    state: Mutex<ServiceState>,
    /// Serializes `start`/`stop` so the task slot always has one writer.
    lifecycle: tokio::sync::Mutex<()>,
}

impl std::fmt::Debug for DreamService {
    /// Prints the loop state and the orchestrator (which prints its schedule and
    /// status); never a repository or consolidator handle, which have no shared
    /// printable form.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DreamService")
            .field("running", &self.is_running())
            .field("last_stop_error", &self.state.lock().last_stop_error)
            .field("orchestrator", &*self.orchestrator)
            .finish_non_exhaustive()
    }
}

impl DreamService {
    /// Wraps `orchestrator` in a stopped service (no task is spawned here).
    ///
    /// The orchestrator keeps its schedule, its planner and repository, its
    /// optional sweep lock, and its status; the service only decides when the
    /// loop runs.
    #[must_use]
    pub fn new(orchestrator: Arc<DreamOrchestrator>) -> Arc<Self> {
        Arc::new(Self {
            orchestrator,
            state: Mutex::new(ServiceState::default()),
            lifecycle: tokio::sync::Mutex::new(()),
        })
    }

    /// Builds a service over a fresh in-memory queue — the single-process shape.
    ///
    /// `config` configures the planner ([`DreamConfig`]) and `schedule` the loop
    /// ([`DreamOrchestratorConfig`]); both are validated by the memory crate, so
    /// an unusable schedule fails here instead of mid-loop. The queue is not
    /// durable and is reachable afterwards through
    /// `service.orchestrator().planner().repository()`.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`](openjiuwen_common::error::Error) when either
    /// configuration would busy-spin, never fire, or overflow.
    pub fn with_in_memory(
        consolidator: Arc<dyn DreamConsolidator>,
        config: DreamConfig,
        schedule: DreamOrchestratorConfig,
    ) -> Result<Arc<Self>> {
        let planner = Arc::new(DreamPlanner::with_in_memory(consolidator, config)?);
        let repository = Arc::clone(planner.repository());
        let orchestrator = Arc::new(DreamOrchestrator::new(planner, repository, schedule)?);
        Ok(Self::new(orchestrator))
    }

    /// The orchestrator this service drives.
    ///
    /// Use it to reach the planner and its repository, the schedule
    /// ([`DreamOrchestrator::config`]), and the status snapshot.
    #[must_use]
    pub fn orchestrator(&self) -> &Arc<DreamOrchestrator> {
        &self.orchestrator
    }

    /// Whether the owned loop task is live.
    ///
    /// A loop that finished on its own — a disabled schedule, or a task that
    /// panicked — reports `false` even before [`DreamService::stop`].
    #[must_use]
    pub fn is_running(&self) -> bool {
        self.state
            .lock()
            .loop_task
            .as_ref()
            .is_some_and(LoopTask::is_live)
    }

    /// A snapshot of the service and its orchestrator.
    ///
    /// The orchestrator status is copied out under its own lock, so the caller
    /// never holds a lock; nothing here awaits.
    #[must_use]
    pub fn status(&self) -> DreamServiceStatus {
        let (running, last_stop_error) = {
            let state = self.state.lock();
            (
                state.loop_task.as_ref().is_some_and(LoopTask::is_live),
                state.last_stop_error.clone(),
            )
        };
        DreamServiceStatus {
            running,
            orchestrator: self.orchestrator.status(),
            last_stop_error,
        }
    }

    /// Starts the background loop and returns the resulting status.
    ///
    /// Idempotent: a call while the loop is live spawns nothing and returns
    /// [`DreamService::status`]. A call after the previous loop ended on its own
    /// retires that finished task and spawns a fresh one. `start` clears
    /// [`DreamServiceStatus::last_stop_error`], because that verdict belongs to
    /// the previous run.
    ///
    /// # Errors
    ///
    /// Never today (the loop's failures are recorded in the status); the `Result`
    /// is the crate's convention for lifecycle calls.
    pub async fn start(&self) -> Result<DreamServiceStatus> {
        let _lifecycle = self.lifecycle.lock().await;
        if self.is_running() {
            tracing::debug!(
                target: LOG_TARGET,
                ticks = self.orchestrator.status().ticks,
                "the dreaming loop is already running; start is a no-op"
            );
            return Ok(self.status());
        }
        {
            let mut state = self.state.lock();
            if state.loop_task.is_some() {
                tracing::debug!(
                    target: LOG_TARGET,
                    "retiring the finished dreaming loop before starting a new one"
                );
            }
            state.loop_task = Some(LoopTask::spawn(Arc::clone(&self.orchestrator)));
            state.last_stop_error = None;
        }
        tracing::info!(target: LOG_TARGET, "the dreaming loop started");
        Ok(self.status())
    }

    /// Stops the background loop and returns the resulting status.
    ///
    /// The token is cancelled first and the task is joined with
    /// [`DEFAULT_DREAMING_STOP_TIMEOUT`] as the bound; a task still running at
    /// the bound is aborted and the reason is recorded in
    /// [`DreamServiceStatus::last_stop_error`]. Cancellation is the documented
    /// shutdown path, so the loop's [`Error::Cancelled`] answer is `Ok`, not a
    /// failure.
    ///
    /// Idempotent: a call when no loop is installed returns the current status.
    ///
    /// # Errors
    ///
    /// Never today — a failed join is reported through the status rather than as
    /// an `Err` — but the `Result` is the crate's convention for lifecycle calls.
    pub async fn stop(&self) -> Result<DreamServiceStatus> {
        let _lifecycle = self.lifecycle.lock().await;
        let Some(task) = self.take_loop_task() else {
            tracing::debug!(
                target: LOG_TARGET,
                "the dreaming loop is not running; stop is a no-op"
            );
            return Ok(self.status());
        };
        let failure = self.join_loop(task).await;
        if let Some(message) = &failure {
            tracing::warn!(
                target: LOG_TARGET,
                reason = %message,
                "the dreaming loop did not shut down cleanly"
            );
        }
        {
            let mut state = self.state.lock();
            state.last_stop_error = failure;
        }
        tracing::info!(target: LOG_TARGET, "the dreaming loop stopped");
        Ok(self.status())
    }

    /// Runs one tick at the caller's clock, for a host that owns its own timer.
    ///
    /// Delegates to [`DreamOrchestrator::tick`]: one sweep through the planner,
    /// then the optional retention pass, with the orchestrator recording the tick
    /// in its status. The service's own loop is untouched — the two are
    /// independent drivers of the same orchestrator, and a host that ticks
    /// itself usually does not call [`DreamService::start`].
    ///
    /// # Errors
    ///
    /// The orchestrator's error for a failed sweep or a lease/lock failure;
    /// a failed *retention* pass never fails the tick (it is recorded in
    /// [`DreamOrchestratorStatus::last_error`]).
    pub async fn tick(&self, now: DateTime<Utc>) -> Result<DreamRunSummary> {
        self.orchestrator.tick(now).await
    }

    /// Removes the loop task from the slot, if one is installed.
    fn take_loop_task(&self) -> Option<LoopTask> {
        self.state.lock().loop_task.take()
    }

    /// Cancels and joins `task`, returning why it did not shut down cleanly.
    ///
    /// `None` means the documented path: the loop observed the cancellation and
    /// returned [`Error::Cancelled`], or it had already returned a summary.
    async fn join_loop(&self, task: LoopTask) -> Option<String> {
        let LoopTask { cancel, mut join } = task;
        cancel.cancel();
        match tokio::time::timeout(DEFAULT_DREAMING_STOP_TIMEOUT, &mut join).await {
            Ok(Ok(Ok(summary))) => {
                tracing::debug!(
                    target: LOG_TARGET,
                    scanned = summary.scanned,
                    "the dreaming loop had already ended by itself"
                );
                None
            }
            Ok(Ok(Err(error))) if error.is_cancelled() => {
                tracing::debug!(
                    target: LOG_TARGET,
                    ticks = self.orchestrator.status().ticks,
                    "the dreaming loop stopped by cancellation"
                );
                None
            }
            Ok(Ok(Err(error))) => Some(format!("the dreaming loop failed: {error}")),
            Ok(Err(error)) => Some(format!("the dreaming loop task failed: {error}")),
            Err(_elapsed) => {
                join.abort();
                Some(format!(
                    "the dreaming loop did not stop within {}ms; the task was aborted",
                    DEFAULT_DREAMING_STOP_TIMEOUT.as_millis()
                ))
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    //! Unit suite for [`DreamService`]: loop ownership, idempotency,
    //! cancellation, and the status snapshot.
    //!
    //! Everything is offline and deterministic: the consolidator is scripted, the
    //! schedule uses millisecond intervals, and the loop's own clock is tokio's
    //! (the injected clock is tested through [`DreamService::tick`]).
    use std::sync::Arc;
    use std::time::Duration as StdDuration;

    use async_trait::async_trait;
    use openjiuwen_common::error::Error;
    use openjiuwen_memory::dreaming::{
        DreamJob, DreamKind, DreamMetrics, DreamOutcome, DreamRepository, DreamState, DreamTrigger,
        InMemoryDreamRepository,
    };
    use openjiuwen_memory::record::{CallOptions, MemoryScope};
    use openjiuwen_test_support::eventually;

    use super::*;

    /// Upper bound for every await in this suite.
    const BOUND: StdDuration = StdDuration::from_secs(10);

    /// A consolidator that records the job ids it was handed and succeeds.
    #[derive(Debug, Default)]
    struct RecordingConsolidator {
        calls: Mutex<Vec<String>>,
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

    /// A schedule whose first sweep never happens within a test's lifetime.
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

    /// A service over an explicit in-memory queue plus its recording consolidator.
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
        let planner = Arc::new(
            DreamPlanner::new(Arc::clone(&repository), port, DreamConfig::default()).unwrap(),
        );
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

    async fn wait_for_ticks(service: &DreamService, ticks: u64) {
        eventually(BOUND, || async {
            service.orchestrator().status().ticks >= ticks
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn tick_consolidates_every_queued_job_once() {
        let (service, repository, consolidator) = service(fast_schedule());
        repository.enqueue(job("job-1")).await.unwrap();
        repository.enqueue(job("job-2")).await.unwrap();

        assert!(!service.is_running(), "a fresh service owns no task");
        let summary = service.tick(Utc::now()).await.unwrap();
        assert_eq!(summary.scanned, 2);
        assert_eq!(summary.succeeded, 2);
        assert_eq!(consolidator.calls().len(), 2);
        for id in ["job-1", "job-2"] {
            assert_eq!(stored(&repository, id).await.state, DreamState::Succeeded);
        }

        let status = service.status();
        assert!(!status.running, "ticking does not start the loop");
        assert_eq!(status.orchestrator.ticks, 1);
        assert_eq!(
            status.orchestrator.last_summary.as_ref().unwrap().succeeded,
            2
        );
        assert!(status.orchestrator.last_error.is_none());
        assert!(status.last_stop_error.is_none());
    }

    #[tokio::test]
    async fn the_loop_drains_the_queue_and_stop_joins_it() {
        let (service, repository, consolidator) = service(fast_schedule());
        repository.enqueue(job("job-1")).await.unwrap();

        let started = service.start().await.unwrap();
        assert!(started.running);
        wait_for_ticks(&service, 1).await;
        assert_eq!(
            stored(&repository, "job-1").await.state,
            DreamState::Succeeded
        );

        let stopped = service.stop().await.unwrap();
        assert!(!stopped.running);
        assert!(
            stopped.last_stop_error.is_none(),
            "cancellation is the documented shutdown path"
        );
        assert!(!service.is_running());
        let calls = consolidator.calls().len();
        assert_eq!(calls, 1, "the one queued job was consolidated once");

        let again = service.stop().await.unwrap();
        assert!(!again.running, "stop is idempotent");
    }

    #[tokio::test]
    async fn a_second_start_while_running_never_spawns_a_second_loop() {
        let (service, repository, _consolidator) = service(fast_schedule());
        repository.enqueue(job("job-1")).await.unwrap();
        service.start().await.unwrap();
        wait_for_ticks(&service, 1).await;

        let second = service.start().await.unwrap();
        assert!(second.running, "the live loop is reported, not restarted");
        service.stop().await.unwrap();
        let frozen = service.orchestrator().status().ticks;
        // Nothing can advance the counter once the only loop is joined; the pause
        // is an observation window, not a synchronisation primitive.
        tokio::time::sleep(StdDuration::from_millis(40)).await;
        assert_eq!(
            service.orchestrator().status().ticks,
            frozen,
            "no second loop survived the stop"
        );
    }

    #[tokio::test]
    async fn stop_interrupts_the_initial_delay() {
        let (service, _repository, _consolidator) = service(slow_schedule());
        service.start().await.unwrap();
        let stopped = tokio::time::timeout(BOUND, service.stop())
            .await
            .expect("stop returns promptly while the loop waits")
            .unwrap();
        assert!(!stopped.running);
        assert_eq!(
            stopped.orchestrator.ticks, 0,
            "the first sweep never came due"
        );
        assert!(stopped.last_stop_error.is_none());
    }

    #[tokio::test]
    async fn a_loop_that_ended_by_itself_is_retired_by_the_next_start() {
        let (service, _repository, _consolidator) = service(fast_schedule().with_enabled(false));
        service.start().await.unwrap();
        eventually(BOUND, || async { !service.is_running() })
            .await
            .unwrap();
        assert_eq!(
            service.orchestrator().status().ticks,
            0,
            "a disabled schedule ends without sweeping"
        );

        let restarted = service.start().await.unwrap();
        assert!(restarted.last_stop_error.is_none());
        eventually(BOUND, || async { !service.is_running() })
            .await
            .unwrap();
        assert!(service.stop().await.unwrap().last_stop_error.is_none());
    }

    #[tokio::test]
    async fn with_in_memory_builds_a_usable_service() {
        let consolidator = Arc::new(RecordingConsolidator::default());
        let port: Arc<dyn DreamConsolidator> = consolidator.clone();
        let service =
            DreamService::with_in_memory(port, DreamConfig::default(), fast_schedule()).unwrap();
        let repository = Arc::clone(service.orchestrator().planner().repository());
        repository.enqueue(job("job-1")).await.unwrap();
        let summary = service.tick(Utc::now()).await.unwrap();
        assert_eq!(summary.succeeded, 1);
        assert_eq!(consolidator.calls(), vec!["job-1".to_owned()]);
    }

    #[test]
    fn with_in_memory_rejects_an_unusable_schedule() {
        let port: Arc<dyn DreamConsolidator> = Arc::new(RecordingConsolidator::default());
        let error = DreamService::with_in_memory(
            port,
            DreamConfig::default(),
            fast_schedule().with_poll_interval_ms(0),
        )
        .unwrap_err();
        assert!(matches!(error, Error::Validation(_)), "got {error:?}");
        assert!(error.to_string().contains("poll_interval_ms"));
    }

    #[test]
    fn the_status_snapshot_round_trips_and_tolerates_unknown_fields() {
        let (service, _repository, _consolidator) = service(fast_schedule());
        let status = service.status();
        assert!(!status.running);
        assert_eq!(status.orchestrator.ticks, 0);

        let json = serde_json::to_value(&status).unwrap();
        assert_eq!(json["running"], serde_json::json!(false));
        assert_eq!(json["orchestrator"]["ticks"], serde_json::json!(0));
        assert!(json.get("last_stop_error").is_none(), "{json}");
        let decoded: DreamServiceStatus = serde_json::from_value(json).unwrap();
        assert_eq!(decoded, status);

        let partial: DreamServiceStatus = serde_json::from_value(serde_json::json!({
            "running": true,
            "unknown_field": {"nested": 1},
        }))
        .unwrap();
        assert!(partial.running);
        assert_eq!(partial.orchestrator.ticks, 0);
        assert!(partial.last_stop_error.is_none());
    }

    #[test]
    fn debug_prints_the_loop_state_and_never_a_handle() {
        let (service, _repository, _consolidator) = service(fast_schedule());
        let debug = format!("{service:?}");
        for needle in ["DreamService", "running", "orchestrator", ".."] {
            assert!(
                debug.contains(needle),
                "`{debug}` should mention `{needle}`"
            );
        }
    }
}
