//! Suites for the background task scheduler.
//!
//! Everything here is offline and deterministic: every await is bounded, a
//! sleep only holds an execution open, and any hold an assertion depends on
//! outlives the step it waits for by at least 10x (>= 500 ms), while the
//! assertions read the [`TaskManager`] or a recording session, not timing.
use super::*;
use crate::controller::schema::TaskResult;
use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_session::{
    AgentSession, InteractionRequest, InteractionResponse, SessionType, StreamItem,
};
use serde_json::json;
use std::sync::atomic::AtomicUsize;

/// Session the fixtures dispatch into.
const SESSION_ID: &str = "s1";
/// Controller id the fixtures report.
const CONTROLLER_ID: &str = "ctl";
/// Task type of the general test executor.
const TASK_TYPE: &str = "test";
/// Task type of the second test executor (multi-executor suites).
const OTHER_TASK_TYPE: &str = "other";

/// Outcome a [`TestExecutor`] reports.
#[derive(Clone)]
enum Outcome {
    /// A completed result carrying `0` as its output.
    Completed(&'static str),
    /// A completed result carrying one frame (exercises `record_outputs`).
    WithFrames,
    /// A completed result carrying one frame and reasoning (exercises the
    /// exposed-reasoning chunk).
    WithFramesAndReasoning(&'static str),
    /// A completed result carrying reasoning but no frames.
    WithReasoning(&'static str),
    /// A failed result carrying the message.
    Failed(&'static str),
    /// An interrupted result.
    InputRequired,
    /// A typed error instead of a result.
    Error(&'static str),
}

/// Counters a [`TestExecutor`] publishes to its test.
#[derive(Default)]
struct ExecutorState {
    /// Executions that entered `execute`.
    executions: AtomicUsize,
    /// Executions that observed the cancellation token.
    stopped: AtomicUsize,
    /// Live executions.
    live: AtomicUsize,
    /// High-water mark of `live`.
    peak: AtomicUsize,
    /// `pause` calls.
    pauses: AtomicUsize,
    /// `cancel` calls.
    cancels: AtomicUsize,
}

/// Executor that holds a task open for `hold`, or until it is cancelled.
struct TestExecutor {
    task_type: &'static str,
    hold: Duration,
    pause_allowed: bool,
    cancel_allowed: bool,
    outcome: Outcome,
    state: Arc<ExecutorState>,
}

impl TestExecutor {
    /// Builds an executor that completes immediately with `"done"`.
    fn new(task_type: &'static str) -> Self {
        Self {
            task_type,
            hold: Duration::ZERO,
            pause_allowed: true,
            cancel_allowed: true,
            outcome: Outcome::Completed("done"),
            state: Arc::new(ExecutorState::default()),
        }
    }

    /// Holds an execution open for `hold` before it reports its outcome.
    fn with_hold(mut self, hold: Duration) -> Self {
        self.hold = hold;
        self
    }

    /// Refuses to pause.
    fn without_pause(mut self) -> Self {
        self.pause_allowed = false;
        self
    }

    /// Refuses to be cancelled.
    fn without_cancel(mut self) -> Self {
        self.cancel_allowed = false;
        self
    }

    /// Reports `outcome` instead of a plain completion.
    fn with_outcome(mut self, outcome: Outcome) -> Self {
        self.outcome = outcome;
        self
    }

    /// Returns the published counters.
    fn state(&self) -> &ExecutorState {
        &self.state
    }
}

#[async_trait]
impl TaskExecutor for TestExecutor {
    fn task_type(&self) -> &str {
        self.task_type
    }

    async fn execute(&self, _task: &Task, ctx: &TaskContext<'_>) -> Result<TaskResult> {
        let live = self.state.live.fetch_add(1, Ordering::SeqCst) + 1;
        self.state.peak.fetch_max(live, Ordering::SeqCst);
        self.state.executions.fetch_add(1, Ordering::SeqCst);
        let stopped = tokio::select! {
            () = ctx.cancellation.cancelled() => true,
            () = tokio::time::sleep(self.hold) => false,
        };
        self.state.live.fetch_sub(1, Ordering::SeqCst);
        if stopped {
            self.state.stopped.fetch_add(1, Ordering::SeqCst);
            return Ok(TaskResult::text("stopped"));
        }
        match self.outcome {
            Outcome::Completed(text) => Ok(TaskResult::text(text)),
            Outcome::WithFrames => Ok(TaskResult {
                status: TaskStatus::Completed,
                data: vec![DataFrame::text("frame")],
                ..TaskResult::default()
            }),
            Outcome::WithFramesAndReasoning(reasoning) => Ok(TaskResult {
                status: TaskStatus::Completed,
                data: vec![DataFrame::text("frame")],
                ..TaskResult::default()
            }
            .with_reasoning(reasoning)),
            Outcome::WithReasoning(reasoning) => {
                Ok(TaskResult::text("done").with_reasoning(reasoning))
            }
            Outcome::Failed(message) => Ok(TaskResult::failed(message)),
            Outcome::InputRequired => Ok(TaskResult::input_required(vec![DataFrame::text("more")])),
            Outcome::Error(message) => Err(Error::validation(message)),
        }
    }

    async fn can_pause(&self, _task: &Task, _ctx: &TaskContext<'_>) -> (bool, String) {
        if self.pause_allowed {
            (true, String::new())
        } else {
            (false, "this executor cannot pause".to_owned())
        }
    }

    async fn pause(&self, _task: &Task, _ctx: &TaskContext<'_>) -> bool {
        self.state.pauses.fetch_add(1, Ordering::SeqCst);
        true
    }

    async fn can_cancel(&self, _task: &Task, _ctx: &TaskContext<'_>) -> (bool, String) {
        if self.cancel_allowed {
            (true, String::new())
        } else {
            (false, "this executor cannot be cancelled".to_owned())
        }
    }

    async fn cancel(&self, _task: &Task, _ctx: &TaskContext<'_>) -> bool {
        self.state.cancels.fetch_add(1, Ordering::SeqCst);
        true
    }
}

/// A `Session` that records every stream frame and delegates state to an
/// [`AgentSession`].
struct RecordingSession {
    inner: AgentSession,
    items: Mutex<Vec<StreamItem>>,
}

impl RecordingSession {
    /// Builds a recording session named `session_id`.
    fn new(session_id: &str) -> Self {
        Self {
            inner: AgentSession::new(session_id),
            items: Mutex::new(Vec::new()),
        }
    }

    /// Copies the frames written through [`Session::write_stream`].
    fn items(&self) -> Vec<StreamItem> {
        self.items.lock().clone()
    }
}

#[async_trait]
impl Session for RecordingSession {
    fn id(&self) -> &str {
        Session::id(&self.inner)
    }

    fn parent_id(&self) -> Option<&str> {
        Session::parent_id(&self.inner)
    }

    fn session_type(&self) -> SessionType {
        Session::session_type(&self.inner)
    }

    async fn get_state(&self, key: &str) -> Result<Option<Value>> {
        Session::get_state(&self.inner, key).await
    }

    async fn update_state(&self, patch: Value) -> Result<()> {
        Session::update_state(&self.inner, patch).await
    }

    async fn get_component_state(&self, component_id: &str) -> Result<Option<Value>> {
        Session::get_component_state(&self.inner, component_id).await
    }

    async fn update_component_state(&self, component_id: &str, patch: Value) -> Result<()> {
        Session::update_component_state(&self.inner, component_id, patch).await
    }

    async fn write_stream(&self, item: StreamItem) -> Result<()> {
        self.items.lock().push(item);
        Ok(())
    }

    async fn request_interaction(
        &self,
        _request: InteractionRequest,
    ) -> Result<InteractionResponse> {
        Err(Error::unsupported(
            "the recording session has no interactions",
        ))
    }

    fn trace(&self) -> &dyn openjiuwen_session::TraceSink {
        Session::trace(&self.inner)
    }
}

/// A scheduler plus the handles a test inspects.
struct Fixture {
    manager: TaskManager,
    config: Arc<RwLock<ControllerConfig>>,
    registry: Arc<RwLock<TaskExecutorRegistry>>,
    scheduler: Arc<TaskSchedulerLoop>,
    sessions: Vec<Arc<RecordingSession>>,
}

impl Fixture {
    /// Builds a scheduler over one recording session named [`SESSION_ID`].
    fn new(config: ControllerConfig, executors: Vec<Arc<dyn TaskExecutor>>) -> Self {
        Self::with_sessions(
            config,
            executors,
            vec![Arc::new(RecordingSession::new(SESSION_ID))],
        )
    }

    /// Builds a scheduler resolving exactly `sessions`.
    fn with_sessions(
        config: ControllerConfig,
        executors: Vec<Arc<dyn TaskExecutor>>,
        sessions: Vec<Arc<RecordingSession>>,
    ) -> Self {
        let manager = TaskManager::with_config(config.clone());
        let mut registry = TaskExecutorRegistry::new();
        for executor in executors {
            registry.add(executor);
        }
        let registry = Arc::new(RwLock::new(registry));
        let config = Arc::new(RwLock::new(config));
        let map: BTreeMap<String, Arc<dyn Session>> = sessions
            .iter()
            .map(|session| {
                (
                    Session::id(session.as_ref()).to_owned(),
                    Arc::clone(session) as _,
                )
            })
            .collect();
        let provider: SessionProvider =
            Arc::new(move |session_id: &str| map.get(session_id).cloned());
        let scheduler = Arc::new(TaskSchedulerLoop::new(
            manager.clone(),
            Arc::clone(&registry),
            Arc::clone(&config),
            CONTROLLER_ID,
            provider,
        ));
        Self {
            manager,
            config,
            registry,
            scheduler,
            sessions,
        }
    }

    /// The single session of [`Fixture::new`].
    fn session(&self) -> &Arc<RecordingSession> {
        &self.sessions[0]
    }

    /// Frames the single session recorded.
    fn signals(&self) -> Vec<StreamItem> {
        self.session().items()
    }

    /// Adds one submitted task of `task_type` to the store.
    fn add(&self, task_id: &str, task_type: &str) {
        self.manager
            .add_task(Task::submitted(SESSION_ID, task_id, task_type, task_id))
            .expect("task is added");
    }

    /// The stored status of `task_id`.
    fn status(&self, task_id: &str) -> TaskStatus {
        self.manager
            .task(task_id)
            .map(|task| task.status)
            .expect("task is stored")
    }
}

/// Default configuration: one concurrent task, a fast scan interval.
fn config() -> ControllerConfig {
    ControllerConfig {
        max_concurrent_tasks: 1,
        schedule_interval_ms: 1,
        ..ControllerConfig::default()
    }
}

/// Configuration that admits every submitted task in one scan.
fn parallel_config() -> ControllerConfig {
    ControllerConfig {
        max_concurrent_tasks: 0,
        ..config()
    }
}

/// A shared, type-erased executor handle.
fn shared(executor: TestExecutor) -> Arc<TestExecutor> {
    Arc::new(executor)
}

/// Polls until `task_id` reports `status`, failing after five seconds.
async fn wait_for_status(fixture: &Fixture, task_id: &str, status: TaskStatus) {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    loop {
        if fixture.status(task_id) == status {
            return;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "task {task_id} never reached {status:?}: {:?}",
            fixture.manager.task(task_id)
        );
        tokio::time::sleep(IDLE_POLL_INTERVAL).await;
    }
}

/// Polls until the executor of a run entered `execute` `count` times.
///
/// Waiting for the execution (rather than for the in-flight entry) is what
/// makes a pause/cancel deterministic: the scheduler records the executor of a
/// run just before it hands the task over, so an entered execution always has
/// its hooks available.
async fn wait_for_executions(state: &ExecutorState, count: usize) {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    while state.executions.load(Ordering::SeqCst) < count {
        assert!(
            tokio::time::Instant::now() < deadline,
            "the executor only ran {} times, expected {count}",
            state.executions.load(Ordering::SeqCst)
        );
        tokio::time::sleep(IDLE_POLL_INTERVAL).await;
    }
}

/// Polls until the session recorded at least `count` frames.
async fn wait_for_frames(session: &RecordingSession, count: usize) -> Vec<StreamItem> {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    loop {
        let items = session.items();
        if items.len() >= count {
            return items;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "session recorded {} frames, expected {count}",
            items.len()
        );
        tokio::time::sleep(IDLE_POLL_INTERVAL).await;
    }
}

#[tokio::test]
async fn concurrency_gate_serialises_at_one() {
    let executor = shared(TestExecutor::new(TASK_TYPE).with_hold(Duration::from_millis(20)));
    let fixture = Fixture::new(
        config(),
        vec![Arc::clone(&executor) as Arc<dyn TaskExecutor>],
    );
    for task_id in ["t1", "t2", "t3"] {
        fixture.add(task_id, TASK_TYPE);
    }

    assert!(fixture.scheduler.start());
    fixture.scheduler.wait_until_idle().await;
    fixture.scheduler.stop().await;

    assert_eq!(executor.state().peak.load(Ordering::SeqCst), 1);
    assert_eq!(executor.state().executions.load(Ordering::SeqCst), 3);
    for task_id in ["t1", "t2", "t3"] {
        assert_eq!(fixture.status(task_id), TaskStatus::Completed);
    }
}

#[tokio::test]
async fn concurrency_gate_runs_in_parallel_when_unlimited() {
    let gated = ControllerConfig {
        max_concurrent_tasks: 0,
        schedule_interval_ms: 1,
        ..ControllerConfig::default()
    };
    // The hold has to outlive the gap between the two drivers starting (which
    // is a spawn apart) by a wide margin, or a loaded machine could observe
    // `peak == 1`. 500 ms against a sub-millisecond gap leaves ample room.
    let executor = shared(TestExecutor::new(TASK_TYPE).with_hold(Duration::from_millis(500)));
    let fixture = Fixture::new(gated, vec![Arc::clone(&executor) as Arc<dyn TaskExecutor>]);
    fixture.add("t1", TASK_TYPE);
    fixture.add("t2", TASK_TYPE);

    assert!(fixture.scheduler.start());
    fixture.scheduler.wait_until_idle().await;
    fixture.scheduler.stop().await;

    assert_eq!(executor.state().peak.load(Ordering::SeqCst), 2);
    assert_eq!(fixture.status("t1"), TaskStatus::Completed);
    assert_eq!(fixture.status("t2"), TaskStatus::Completed);
}

#[tokio::test]
async fn claim_guard_skips_a_task_that_changed_status() {
    let executor = shared(TestExecutor::new(TASK_TYPE));
    let fixture = Fixture::new(
        config(),
        vec![Arc::clone(&executor) as Arc<dyn TaskExecutor>],
    );
    fixture.add("t1", TASK_TYPE);
    fixture
        .manager
        .update_task_status("t1", TaskStatus::Canceled, false, false, None);

    fixture.scheduler.scan_once().await;

    assert_eq!(executor.state().executions.load(Ordering::SeqCst), 0);
    assert_eq!(fixture.scheduler.in_flight().len(), 0);
    assert_eq!(fixture.status("t1"), TaskStatus::Canceled);
}

#[tokio::test]
async fn timeout_fails_the_task_with_pythons_message() {
    let gated = ControllerConfig {
        task_timeout_ms: Some(20),
        ..config()
    };
    let executor = shared(TestExecutor::new(TASK_TYPE).with_hold(Duration::from_secs(2)));
    let fixture = Fixture::new(gated, vec![Arc::clone(&executor) as Arc<dyn TaskExecutor>]);
    fixture.add("t1", TASK_TYPE);

    fixture.scheduler.scan_once().await;
    wait_for_status(&fixture, "t1", TaskStatus::Failed).await;
    fixture.scheduler.wait_until_idle().await;

    let task = fixture.manager.task("t1").expect("task is stored");
    assert_eq!(
        task.error_message.as_deref(),
        Some("Task timeout after 0.02 seconds")
    );
    assert_eq!(fixture.scheduler.in_flight().len(), 0);
}

#[tokio::test]
async fn cancel_stops_a_cooperative_task() {
    let executor = shared(TestExecutor::new(TASK_TYPE).with_hold(Duration::from_secs(5)));
    let fixture = Fixture::new(
        config(),
        vec![Arc::clone(&executor) as Arc<dyn TaskExecutor>],
    );
    fixture.add("t1", TASK_TYPE);

    fixture.scheduler.scan_once().await;
    wait_for_executions(executor.state(), 1).await;
    assert_eq!(fixture.status("t1"), TaskStatus::Working);

    assert!(fixture.scheduler.cancel_task("t1").await);

    assert_eq!(fixture.status("t1"), TaskStatus::Canceled);
    assert_eq!(fixture.scheduler.in_flight().len(), 0);
    assert_eq!(executor.state().cancels.load(Ordering::SeqCst), 1);
    assert_eq!(executor.state().stopped.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn cancel_is_idempotent_and_short_circuits_submitted_tasks() {
    let executor = shared(TestExecutor::new(TASK_TYPE));
    let fixture = Fixture::new(
        config(),
        vec![Arc::clone(&executor) as Arc<dyn TaskExecutor>],
    );

    assert!(!fixture.scheduler.cancel_task("missing").await);

    fixture.add("done", TASK_TYPE);
    fixture
        .manager
        .update_task_status("done", TaskStatus::Completed, false, false, None);
    assert!(fixture.scheduler.cancel_task("done").await);
    assert_eq!(fixture.status("done"), TaskStatus::Completed);

    fixture.add("queued", TASK_TYPE);
    assert!(fixture.scheduler.cancel_task("queued").await);
    assert_eq!(fixture.status("queued"), TaskStatus::Canceled);

    fixture.scheduler.scan_once().await;
    fixture.scheduler.wait_until_idle().await;

    assert_eq!(executor.state().executions.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn cancel_is_refused_when_the_executor_says_no() {
    let executor = shared(
        TestExecutor::new(TASK_TYPE)
            .without_cancel()
            .with_hold(Duration::from_secs(5)),
    );
    let fixture = Fixture::new(
        config(),
        vec![Arc::clone(&executor) as Arc<dyn TaskExecutor>],
    );
    fixture.add("t1", TASK_TYPE);

    fixture.scheduler.scan_once().await;
    wait_for_executions(executor.state(), 1).await;

    assert!(!fixture.scheduler.cancel_task("t1").await);
    assert_eq!(fixture.status("t1"), TaskStatus::Working);
    assert_eq!(executor.state().cancels.load(Ordering::SeqCst), 0);

    fixture.scheduler.stop().await;
}

#[tokio::test]
async fn pause_is_refused_when_the_executor_cannot_pause() {
    let executor = shared(
        TestExecutor::new(TASK_TYPE)
            .without_pause()
            .with_hold(Duration::from_secs(5)),
    );
    let fixture = Fixture::new(
        config(),
        vec![Arc::clone(&executor) as Arc<dyn TaskExecutor>],
    );
    fixture.add("t1", TASK_TYPE);

    fixture.scheduler.scan_once().await;
    wait_for_executions(executor.state(), 1).await;

    assert!(!fixture.scheduler.pause_task("t1").await);
    assert_eq!(fixture.status("t1"), TaskStatus::Working);
    assert_eq!(executor.state().pauses.load(Ordering::SeqCst), 0);

    assert!(fixture.scheduler.cancel_task("t1").await);
}

#[tokio::test]
async fn pause_stops_a_cooperative_task_and_resubmitting_resumes_it() {
    // The pause has to reach the run before the hold elapses; 1 s against the
    // millisecond-scale path that observes the execution and takes the pause
    // is a 1000x margin. The resumed run sleeps the same hold once more, which
    // keeps the suite at about a second.
    let executor = shared(TestExecutor::new(TASK_TYPE).with_hold(Duration::from_secs(1)));
    let fixture = Fixture::new(
        config(),
        vec![Arc::clone(&executor) as Arc<dyn TaskExecutor>],
    );
    fixture.add("t1", TASK_TYPE);
    assert!(fixture.scheduler.start());

    wait_for_executions(executor.state(), 1).await;
    assert!(fixture.scheduler.pause_task("t1").await);

    assert_eq!(fixture.status("t1"), TaskStatus::Paused);
    assert_eq!(fixture.scheduler.in_flight().len(), 0);
    assert_eq!(executor.state().pauses.load(Ordering::SeqCst), 1);
    assert_eq!(executor.state().stopped.load(Ordering::SeqCst), 1);
    assert!(!fixture.scheduler.pause_task("t1").await, "not in flight");

    // A resume is the caller's job: flip the status back and wake the loop.
    fixture
        .manager
        .update_task_status("t1", TaskStatus::Submitted, false, false, None);
    fixture.scheduler.wake();
    wait_for_status(&fixture, "t1", TaskStatus::Completed).await;
    fixture.scheduler.stop().await;

    assert_eq!(executor.state().executions.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn missing_executor_fails_the_task_and_the_loop_keeps_running() {
    let executor = shared(TestExecutor::new(TASK_TYPE));
    let fixture = Fixture::new(
        config(),
        vec![Arc::clone(&executor) as Arc<dyn TaskExecutor>],
    );
    fixture.add("unknown", "missing");
    assert!(fixture.scheduler.start());

    wait_for_status(&fixture, "unknown", TaskStatus::Failed).await;
    let task = fixture.manager.task("unknown").expect("task is stored");
    assert_eq!(
        task.error_message.as_deref(),
        Some("task executor not found: missing")
    );
    assert!(
        fixture.scheduler.is_running(),
        "the loop survives the failure"
    );

    fixture.add("t2", TASK_TYPE);
    wait_for_status(&fixture, "t2", TaskStatus::Completed).await;
    fixture.scheduler.stop().await;
}

#[tokio::test]
async fn dependency_gate_promotes_ready_waiting_tasks() {
    let executor = shared(TestExecutor::new(TASK_TYPE));
    let failing = shared(TestExecutor::new(OTHER_TASK_TYPE).with_outcome(Outcome::Failed("boom")));
    let fixture = Fixture::new(
        parallel_config(),
        vec![
            Arc::clone(&executor) as Arc<dyn TaskExecutor>,
            Arc::clone(&failing) as Arc<dyn TaskExecutor>,
        ],
    );
    fixture.add("dep", TASK_TYPE);
    fixture
        .manager
        .add_task(
            Task::new(SESSION_ID, "waiting", TASK_TYPE)
                .with_status(TaskStatus::Waiting)
                .with_metadata(DEPENDS_ON_KEY, json!(["dep"])),
        )
        .expect("task is added");
    fixture
        .manager
        .add_task(
            Task::new(SESSION_ID, "blocked", TASK_TYPE)
                .with_status(TaskStatus::Waiting)
                .with_metadata(DEPENDS_ON_KEY, json!(["bad"])),
        )
        .expect("task is added");
    fixture
        .manager
        .add_task(Task::new(SESSION_ID, "bad", OTHER_TASK_TYPE))
        .expect("task is added");

    fixture.scheduler.scan_once().await;
    assert_eq!(fixture.status("waiting"), TaskStatus::Waiting);

    fixture.scheduler.wait_until_idle().await;
    assert_eq!(fixture.status("dep"), TaskStatus::Completed);
    assert_eq!(fixture.status("bad"), TaskStatus::Failed);

    fixture.scheduler.scan_once().await;
    fixture.scheduler.wait_until_idle().await;

    assert_eq!(fixture.status("waiting"), TaskStatus::Completed);
    assert_eq!(
        fixture.status("blocked"),
        TaskStatus::Waiting,
        "a failed dependency never promotes"
    );
    assert_eq!(executor.state().executions.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn executor_outcomes_map_onto_task_statuses() {
    let failed =
        shared(TestExecutor::new("failed").with_outcome(Outcome::Failed("executor said no")));
    let errored = shared(TestExecutor::new("errored").with_outcome(Outcome::Error("trapped")));
    let interrupted = shared(TestExecutor::new("interrupted").with_outcome(Outcome::InputRequired));
    let framed = shared(TestExecutor::new("framed").with_outcome(Outcome::WithFrames));
    let fixture = Fixture::new(
        parallel_config(),
        vec![
            Arc::clone(&failed) as Arc<dyn TaskExecutor>,
            Arc::clone(&errored) as Arc<dyn TaskExecutor>,
            Arc::clone(&interrupted) as Arc<dyn TaskExecutor>,
            Arc::clone(&framed) as Arc<dyn TaskExecutor>,
        ],
    );
    for (task_id, task_type) in [
        ("t1", "failed"),
        ("t2", "errored"),
        ("t3", "interrupted"),
        ("t4", "framed"),
    ] {
        fixture.add(task_id, task_type);
    }

    fixture.scheduler.scan_once().await;
    fixture.scheduler.wait_until_idle().await;

    let failed_task = fixture.manager.task("t1").expect("task is stored");
    assert_eq!(failed_task.status, TaskStatus::Failed);
    assert_eq!(
        failed_task.error_message.as_deref(),
        Some("executor said no")
    );
    let errored_task = fixture.manager.task("t2").expect("task is stored");
    assert_eq!(errored_task.status, TaskStatus::Failed);
    assert_eq!(errored_task.error_message.as_deref(), Some("trapped"));
    assert_eq!(fixture.status("t3"), TaskStatus::InputRequired);

    let framed_task = fixture.manager.task("t4").expect("task is stored");
    assert_eq!(framed_task.status, TaskStatus::Completed);
    assert_eq!(framed_task.outputs.len(), 1);
    let payload = framed_task.outputs[0].payload.clone().expect("payload");
    assert_eq!(payload.kind, OutputKind::TaskCompletion);
    assert!(framed_task.outputs[0].last_chunk);
}

#[tokio::test]
async fn exposed_reasoning_is_appended_before_the_terminal_chunk() {
    let executor =
        shared(TestExecutor::new(TASK_TYPE).with_outcome(Outcome::WithFramesAndReasoning("why")));
    let fixture = Fixture::new(
        ControllerConfig {
            expose_reasoning: true,
            ..config()
        },
        vec![Arc::clone(&executor) as Arc<dyn TaskExecutor>],
    );
    fixture.add("t1", TASK_TYPE);

    fixture.scheduler.scan_once().await;
    fixture.scheduler.wait_until_idle().await;

    let task = fixture.manager.task("t1").expect("task is stored");
    assert_eq!(task.status, TaskStatus::Completed);
    assert_eq!(task.outputs.len(), 2, "{:?}", task.outputs);
    let reasoning = task.outputs[0].payload.clone().expect("payload");
    assert_eq!(reasoning.kind, OutputKind::Reasoning);
    assert_eq!(reasoning.data, vec![DataFrame::text("why")]);
    assert!(!task.outputs[0].last_chunk);
    let terminal = task.outputs[1].payload.clone().expect("payload");
    assert_eq!(terminal.kind, OutputKind::TaskCompletion);
    assert_eq!(terminal.data, vec![DataFrame::text("frame")]);
    assert!(task.outputs[1].last_chunk);
}

#[tokio::test]
async fn reasoning_is_dropped_when_not_exposed() {
    let executor =
        shared(TestExecutor::new(TASK_TYPE).with_outcome(Outcome::WithFramesAndReasoning("why")));
    let fixture = Fixture::new(
        config(),
        vec![Arc::clone(&executor) as Arc<dyn TaskExecutor>],
    );
    fixture.add("t1", TASK_TYPE);

    fixture.scheduler.scan_once().await;
    fixture.scheduler.wait_until_idle().await;

    let task = fixture.manager.task("t1").expect("task is stored");
    assert_eq!(task.status, TaskStatus::Completed);
    assert_eq!(task.outputs.len(), 1, "{:?}", task.outputs);
    let payload = task.outputs[0].payload.clone().expect("payload");
    assert_eq!(payload.kind, OutputKind::TaskCompletion);
    assert_eq!(payload.data, vec![DataFrame::text("frame")]);
    assert!(task.outputs[0].last_chunk);
    assert!(
        task.outputs.iter().all(|chunk| chunk
            .payload
            .as_ref()
            .is_none_or(|payload| payload.kind != OutputKind::Reasoning)),
        "reasoning must not reach a frame: {:?}",
        task.outputs
    );
}

#[tokio::test]
async fn reasoning_without_frames_is_recorded_alone_when_exposed() {
    let exposed = Fixture::new(
        ControllerConfig {
            expose_reasoning: true,
            ..config()
        },
        vec![
            Arc::new(TestExecutor::new(TASK_TYPE).with_outcome(Outcome::WithReasoning("why")))
                as Arc<dyn TaskExecutor>,
        ],
    );
    exposed.add("t1", TASK_TYPE);
    exposed.scheduler.scan_once().await;
    exposed.scheduler.wait_until_idle().await;

    let task = exposed.manager.task("t1").expect("task is stored");
    assert_eq!(task.status, TaskStatus::Completed);
    assert_eq!(task.outputs.len(), 1, "{:?}", task.outputs);
    let payload = task.outputs[0].payload.clone().expect("payload");
    assert_eq!(payload.kind, OutputKind::Reasoning);
    assert_eq!(payload.data, vec![DataFrame::text("why")]);
    assert!(task.outputs[0].last_chunk);

    let hidden = Fixture::new(
        config(),
        vec![
            Arc::new(TestExecutor::new(TASK_TYPE).with_outcome(Outcome::WithReasoning("why")))
                as Arc<dyn TaskExecutor>,
        ],
    );
    hidden.add("t1", TASK_TYPE);
    hidden.scheduler.scan_once().await;
    hidden.scheduler.wait_until_idle().await;

    let task = hidden.manager.task("t1").expect("task is stored");
    assert_eq!(task.status, TaskStatus::Completed);
    assert_eq!(
        task.outputs.len(),
        0,
        "no data and no exposed reasoning appends nothing: {:?}",
        task.outputs
    );
}

#[tokio::test]
async fn completion_signal_is_written_once_per_idle_stretch() {
    let fixture = Fixture::new(
        config(),
        vec![shared(TestExecutor::new(TASK_TYPE)) as Arc<dyn TaskExecutor>],
    );
    fixture.add("t1", TASK_TYPE);

    fixture.scheduler.scan_once().await;
    let items = wait_for_frames(fixture.session(), 1).await;
    assert_eq!(items.len(), 1);
    assert_eq!(items[0].item_type, "controller_output");
    assert_eq!(items[0].payload["type"], json!("all_tasks_processed"));
    assert_eq!(items[0].payload["data"][0]["type"], json!("text"));
    assert_eq!(items[0].payload["data"][0]["text"], json!(COMPLETION_TEXT));

    // A second idle pass over the same stretch writes nothing.
    fixture.scheduler.ensure_completion_signal(SESSION_ID).await;
    fixture.scheduler.ensure_completion_signal(SESSION_ID).await;
    assert_eq!(fixture.signals().len(), 1);
}

#[tokio::test]
async fn completion_signal_waits_for_active_tasks() {
    let executor = shared(TestExecutor::new(TASK_TYPE).with_hold(Duration::from_secs(5)));
    let fixture = Fixture::new(
        config(),
        vec![Arc::clone(&executor) as Arc<dyn TaskExecutor>],
    );
    fixture.add("t1", TASK_TYPE);

    fixture.scheduler.scan_once().await;
    wait_for_executions(executor.state(), 1).await;

    fixture.scheduler.ensure_completion_signal(SESSION_ID).await;
    assert!(
        fixture.signals().is_empty(),
        "a running task blocks the signal"
    );

    assert!(fixture.scheduler.cancel_task("t1").await);
    fixture.scheduler.wait_until_idle().await;
    assert_eq!(
        fixture.signals().len(),
        0,
        "the driver's finalizer saw an active task, and cancel owns the status"
    );

    // The cancel cleared the idle marker, so the next pass may signal.
    fixture.scheduler.ensure_completion_signal(SESSION_ID).await;
    assert_eq!(wait_for_frames(fixture.session(), 1).await.len(), 1);
}

#[tokio::test]
async fn completion_signal_is_suppressed_by_config() {
    let gated = ControllerConfig {
        suppress_completion_signal: true,
        ..config()
    };
    let fixture = Fixture::new(
        gated,
        vec![shared(TestExecutor::new(TASK_TYPE)) as Arc<dyn TaskExecutor>],
    );
    fixture.add("t1", TASK_TYPE);

    fixture.scheduler.scan_once().await;
    fixture.scheduler.wait_until_idle().await;
    fixture.scheduler.ensure_completion_signal(SESSION_ID).await;
    tokio::time::sleep(Duration::from_millis(20)).await;

    assert_eq!(fixture.signals().len(), 0);
}

#[tokio::test]
async fn unknown_sessions_are_skipped() {
    let executor = shared(TestExecutor::new(TASK_TYPE));
    let fixture = Fixture::with_sessions(
        config(),
        vec![Arc::clone(&executor) as Arc<dyn TaskExecutor>],
        Vec::new(),
    );
    fixture.add("t1", TASK_TYPE);

    fixture.scheduler.scan_once().await;
    fixture.scheduler.ensure_completion_signal(SESSION_ID).await;

    assert_eq!(fixture.scheduler.in_flight().len(), 0);
    assert_eq!(executor.state().executions.load(Ordering::SeqCst), 0);
    assert_eq!(fixture.status("t1"), TaskStatus::Submitted);
}

#[tokio::test]
async fn in_flight_lists_running_task_ids_in_order() {
    let gated = ControllerConfig {
        max_concurrent_tasks: 0,
        schedule_interval_ms: 1,
        ..ControllerConfig::default()
    };
    // The assertions below require both runs to still be in flight when
    // `scan_once` returns; the 2 s hold dwarfs that hand-over, and `stop`
    // cancels the cooperative runs, so the margin costs nothing.
    let executor = shared(TestExecutor::new(TASK_TYPE).with_hold(Duration::from_secs(2)));
    let fixture = Fixture::new(gated, vec![Arc::clone(&executor) as Arc<dyn TaskExecutor>]);
    fixture.add("t2", TASK_TYPE);
    fixture.add("t1", TASK_TYPE);

    fixture.scheduler.scan_once().await;
    assert_eq!(fixture.scheduler.in_flight(), ["t1", "t2"]);

    fixture.scheduler.stop().await;
    assert_eq!(fixture.scheduler.in_flight().len(), 0);
    assert!(!fixture.scheduler.is_running());
}

#[tokio::test]
async fn start_and_stop_are_idempotent_and_wake_scans_immediately() {
    let gated = ControllerConfig {
        max_concurrent_tasks: 1,
        schedule_interval_ms: 5_000,
        ..ControllerConfig::default()
    };
    let fixture = Fixture::new(
        gated,
        vec![shared(TestExecutor::new(TASK_TYPE)) as Arc<dyn TaskExecutor>],
    );

    assert!(fixture.scheduler.start());
    assert!(!fixture.scheduler.start(), "already running");
    assert!(fixture.scheduler.is_running());
    assert_eq!(
        fixture.scheduler.config().read().schedule_interval_ms,
        5_000
    );

    // The submitted callback and wake() both beat the five-second interval.
    fixture.add("t1", TASK_TYPE);
    wait_for_status(&fixture, "t1", TaskStatus::Completed).await;
    fixture.add("t2", TASK_TYPE);
    fixture.scheduler.wake();
    wait_for_status(&fixture, "t2", TaskStatus::Completed).await;

    fixture.scheduler.stop().await;
    assert!(!fixture.scheduler.is_running());
    fixture.scheduler.stop().await;

    fixture.add("t3", TASK_TYPE);
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert_eq!(fixture.status("t3"), TaskStatus::Submitted);
}

#[tokio::test]
async fn task_context_cancellation_is_fresh_by_default() {
    let fixture = Fixture::new(
        config(),
        vec![shared(TestExecutor::new(TASK_TYPE)) as Arc<dyn TaskExecutor>],
    );
    let token = CancellationToken::new();
    let ctx = TaskContext::new(None, CONTROLLER_ID).with_cancellation(token.clone());
    assert!(!ctx.cancellation.is_cancelled());
    token.cancel();
    assert!(ctx.cancellation.is_cancelled());

    let plain = TaskContext::new(None, CONTROLLER_ID);
    assert!(!plain.cancellation.is_cancelled());

    assert_eq!(fixture.scheduler.registry().read().len(), 1);
    assert_eq!(
        fixture.registry.read().task_types(),
        vec![TASK_TYPE.to_owned()]
    );
    assert!(
        Arc::ptr_eq(&fixture.config, &fixture.scheduler.config()),
        "the scheduler shares the caller's configuration handle"
    );
    assert!(
        Arc::ptr_eq(&fixture.registry, &fixture.scheduler.registry()),
        "the scheduler shares the caller's registry handle"
    );
}
