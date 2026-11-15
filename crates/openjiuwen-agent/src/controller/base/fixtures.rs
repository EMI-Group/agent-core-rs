//! Test doubles shared by the controller-loop test modules.
//!
//! The accessor tests in `mod.rs`, the persistence tests in `state`, and the
//! routing tests in `tests` all wire the same controller, so the executor, the
//! event handler, the recognizer, the recording session, and the turn helpers
//! live here once.
use super::{BaseController, Controller, EventHandler, EventHandlerOutcome};
use crate::card::AgentCard;
use crate::controller::config::ControllerConfig;
use crate::controller::executors::{TaskContext, TaskExecutor};
use crate::controller::history::HistoryEntry;
use crate::controller::intent::IntentRecognizer;
use crate::controller::schema::{ControllerOutput, Event, Intent, Task, TaskResult, TaskStatus};
use async_trait::async_trait;
use openjiuwen_common::error::Result;
use openjiuwen_session::{
    AgentSession, InteractionRequest, InteractionResponse, Session, SessionType, StreamItem,
    TraceSink,
};
use parking_lot::Mutex as SyncMutex;
use serde_json::json;
use std::future::Future;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;

/// Description reported by [`StubExecutor`].
pub(crate) const STUB_DESCRIPTION: &str = "stub executor";
/// Name reported by [`StubHandler`] and [`CountingRecognizer`].
pub(crate) const STUB_NAME: &str = "stub";

/// Histories captured per dispatch or per recognition call.
pub(crate) type RecordedHistories = Arc<SyncMutex<Vec<Vec<HistoryEntry>>>>;
/// `TaskContext::expose_reasoning` flags captured per dispatch.
pub(crate) type RecordedExposure = Arc<SyncMutex<Vec<bool>>>;

/// Executor that records its calls and answers with a fixed text.
pub(crate) struct StubExecutor {
    task_type: &'static str,
    text: &'static str,
    pub(crate) calls: Arc<AtomicUsize>,
}

impl StubExecutor {
    pub(crate) fn new(task_type: &'static str, text: &'static str) -> (Self, Arc<AtomicUsize>) {
        let calls = Arc::new(AtomicUsize::new(0));
        (
            Self {
                task_type,
                text,
                calls: Arc::clone(&calls),
            },
            calls,
        )
    }
}

#[async_trait]
impl TaskExecutor for StubExecutor {
    fn task_type(&self) -> &str {
        self.task_type
    }

    fn description(&self) -> &str {
        STUB_DESCRIPTION
    }

    async fn execute(&self, _task: &Task, _ctx: &TaskContext<'_>) -> Result<TaskResult> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(TaskResult::text(self.text))
    }
}

/// Executor whose run blocks until its task is paused, cancelled, or the
/// scheduler stops, and which reports that it can be paused.
pub(crate) struct CooperativeExecutor {
    task_type: &'static str,
    pub(crate) started: Arc<AtomicUsize>,
    pub(crate) finished: Arc<AtomicUsize>,
}

impl CooperativeExecutor {
    pub(crate) fn new(task_type: &'static str) -> (Self, Arc<AtomicUsize>, Arc<AtomicUsize>) {
        let started = Arc::new(AtomicUsize::new(0));
        let finished = Arc::new(AtomicUsize::new(0));
        (
            Self {
                task_type,
                started: Arc::clone(&started),
                finished: Arc::clone(&finished),
            },
            started,
            finished,
        )
    }
}

#[async_trait]
impl TaskExecutor for CooperativeExecutor {
    fn task_type(&self) -> &str {
        self.task_type
    }

    fn description(&self) -> &str {
        STUB_DESCRIPTION
    }

    async fn execute(&self, _task: &Task, ctx: &TaskContext<'_>) -> Result<TaskResult> {
        self.started.fetch_add(1, Ordering::SeqCst);
        ctx.cancellation.cancelled().await;
        self.finished.fetch_add(1, Ordering::SeqCst);
        Ok(TaskResult::text("stopped"))
    }

    async fn can_pause(&self, _task: &Task, _ctx: &TaskContext<'_>) -> (bool, String) {
        (true, String::new())
    }

    async fn pause(&self, _task: &Task, _ctx: &TaskContext<'_>) -> bool {
        true
    }

    async fn cancel(&self, _task: &Task, _ctx: &TaskContext<'_>) -> bool {
        true
    }
}

/// Executor that records the dispatch context it was handed and answers with a
/// fixed text, optionally carrying a reasoning text.
pub(crate) struct ContextExecutor {
    task_type: &'static str,
    text: &'static str,
    reasoning: Option<&'static str>,
    pub(crate) histories: RecordedHistories,
    pub(crate) exposed: RecordedExposure,
}

impl ContextExecutor {
    pub(crate) fn new(
        task_type: &'static str,
        text: &'static str,
    ) -> (Self, RecordedHistories, RecordedExposure) {
        let histories = Arc::new(SyncMutex::new(Vec::new()));
        let exposed = Arc::new(SyncMutex::new(Vec::new()));
        (
            Self {
                task_type,
                text,
                reasoning: None,
                histories: Arc::clone(&histories),
                exposed: Arc::clone(&exposed),
            },
            histories,
            exposed,
        )
    }

    /// Reports `reasoning` on every run.
    pub(crate) fn with_reasoning(mut self, reasoning: &'static str) -> Self {
        self.reasoning = Some(reasoning);
        self
    }
}

#[async_trait]
impl TaskExecutor for ContextExecutor {
    fn task_type(&self) -> &str {
        self.task_type
    }

    fn description(&self) -> &str {
        STUB_DESCRIPTION
    }

    async fn execute(&self, _task: &Task, ctx: &TaskContext<'_>) -> Result<TaskResult> {
        self.histories.lock().push(ctx.history.clone());
        self.exposed.lock().push(ctx.expose_reasoning);
        let result = TaskResult::text(self.text);
        Ok(match self.reasoning {
            Some(reasoning) => result.with_reasoning(reasoning),
            None => result,
        })
    }
}

/// Executor whose run sleeps before it finishes, for the turn deadlines.
///
/// `started` counts every poll that entered the run and `finished` only the runs
/// that outlived the sleep, so a cancelled run leaves `finished` untouched.
pub(crate) struct StallingExecutor {
    task_type: &'static str,
    delay: Duration,
    pub(crate) started: Arc<AtomicUsize>,
    pub(crate) finished: Arc<AtomicUsize>,
}

impl StallingExecutor {
    pub(crate) fn new(
        task_type: &'static str,
        delay: Duration,
    ) -> (Self, Arc<AtomicUsize>, Arc<AtomicUsize>) {
        let started = Arc::new(AtomicUsize::new(0));
        let finished = Arc::new(AtomicUsize::new(0));
        (
            Self {
                task_type,
                delay,
                started: Arc::clone(&started),
                finished: Arc::clone(&finished),
            },
            started,
            finished,
        )
    }
}

#[async_trait]
impl TaskExecutor for StallingExecutor {
    fn task_type(&self) -> &str {
        self.task_type
    }

    fn description(&self) -> &str {
        STUB_DESCRIPTION
    }

    async fn execute(&self, _task: &Task, _ctx: &TaskContext<'_>) -> Result<TaskResult> {
        self.started.fetch_add(1, Ordering::SeqCst);
        tokio::time::sleep(self.delay).await;
        self.finished.fetch_add(1, Ordering::SeqCst);
        Ok(TaskResult::text("slow"))
    }
}

/// Handler that answers every event with one fixed outcome.
pub(crate) struct StubHandler {
    pub(crate) outcome: EventHandlerOutcome,
}

#[async_trait]
impl EventHandler for StubHandler {
    fn name(&self) -> &str {
        STUB_NAME
    }

    async fn handle(&self, _event: &Event, _ctx: &TaskContext<'_>) -> Result<EventHandlerOutcome> {
        Ok(self.outcome.clone())
    }
}

/// Recognizer that answers the first call with one intent and every later
/// call with two distinct intents.
pub(crate) struct CountingRecognizer {
    pub(crate) calls: AtomicUsize,
}

#[async_trait]
impl IntentRecognizer for CountingRecognizer {
    fn name(&self) -> &str {
        STUB_NAME
    }

    async fn recognize(&self, event: &Event) -> Result<Vec<Intent>> {
        let call = self.calls.fetch_add(1, Ordering::SeqCst);
        let count = if call == 0 { 1 } else { 2 };
        Ok((0..count)
            .map(|index| {
                Intent::create_task(event.clone(), format!("task {call}-{index}"))
                    .with_name(format!("t{call}{index}"))
            })
            .collect())
    }
}

/// Recognizer that records the history it was handed and then delegates to its
/// own plain recognition.
pub(crate) struct RecordingRecognizer {
    pub(crate) histories: RecordedHistories,
}

impl RecordingRecognizer {
    pub(crate) fn new() -> (Self, RecordedHistories) {
        let histories = Arc::new(SyncMutex::new(Vec::new()));
        (
            Self {
                histories: Arc::clone(&histories),
            },
            histories,
        )
    }
}

#[async_trait]
impl IntentRecognizer for RecordingRecognizer {
    fn name(&self) -> &str {
        STUB_NAME
    }

    async fn recognize(&self, event: &Event) -> Result<Vec<Intent>> {
        Ok(vec![Intent::default_for(event.clone())])
    }

    async fn recognize_with_history(
        &self,
        event: &Event,
        history: &[HistoryEntry],
    ) -> Result<Vec<Intent>> {
        self.histories.lock().push(history.to_vec());
        self.recognize(event).await
    }
}

/// Session that delegates every call to an [`AgentSession`] and records the
/// frames written to its stream.
pub(crate) struct RecordingSession {
    inner: AgentSession,
    pub(crate) frames: Arc<Mutex<Vec<StreamItem>>>,
}

impl RecordingSession {
    pub(crate) fn new(session_id: &str) -> Self {
        Self {
            inner: AgentSession::new(session_id),
            frames: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// The frames written so far.
    pub(crate) async fn written_frames(&self) -> Vec<StreamItem> {
        self.frames.lock().await.clone()
    }
}

#[async_trait]
impl Session for RecordingSession {
    fn id(&self) -> &str {
        self.inner.id()
    }

    fn parent_id(&self) -> Option<&str> {
        self.inner.parent_id()
    }

    fn session_type(&self) -> SessionType {
        self.inner.session_type()
    }

    async fn get_state(&self, key: &str) -> Result<Option<serde_json::Value>> {
        self.inner.get_state(key).await
    }

    async fn update_state(&self, patch: serde_json::Value) -> Result<()> {
        self.inner.update_state(patch).await
    }

    async fn get_component_state(&self, component_id: &str) -> Result<Option<serde_json::Value>> {
        self.inner.get_component_state(component_id).await
    }

    async fn update_component_state(
        &self,
        component_id: &str,
        patch: serde_json::Value,
    ) -> Result<()> {
        self.inner.update_component_state(component_id, patch).await
    }

    async fn write_stream(&self, item: StreamItem) -> Result<()> {
        self.frames.lock().await.push(item.clone());
        self.inner.write_stream(item).await
    }

    async fn request_interaction(
        &self,
        request: InteractionRequest,
    ) -> Result<InteractionResponse> {
        self.inner.request_interaction(request).await
    }

    fn trace(&self) -> &dyn TraceSink {
        self.inner.trace()
    }
}

pub(crate) fn test_session() -> Arc<dyn Session> {
    Arc::new(AgentSession::new("s1"))
}

/// A session double whose stream frames the test can assert on.
pub(crate) fn recording_session(session_id: &str) -> Arc<RecordingSession> {
    Arc::new(RecordingSession::new(session_id))
}

pub(crate) fn controller_with_executor() -> (BaseController, Arc<AtomicUsize>) {
    let controller = BaseController::new(AgentCard::new("ctrl"), ControllerConfig::new());
    let (executor, calls) = StubExecutor::new("agent", "done");
    controller.add_executor("agent", Arc::new(executor));
    (controller, calls)
}

/// A controller whose `agent` executor runs until the task is stopped.
pub(crate) fn controller_with_cooperative_executor(
) -> (BaseController, Arc<AtomicUsize>, Arc<AtomicUsize>) {
    let controller = BaseController::new(AgentCard::new("ctrl"), ControllerConfig::new());
    let (executor, started, finished) = CooperativeExecutor::new("agent");
    controller.add_executor("agent", Arc::new(executor));
    (controller, started, finished)
}

/// Waits (bounded) until `counter` is at least `count`.
pub(crate) async fn await_counter(counter: &AtomicUsize, count: usize) {
    for _ in 0..1_000 {
        if counter.load(Ordering::SeqCst) >= count {
            return;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

/// Awaits `future` under a 10 s guard, so a stuck operation fails the test
/// instead of hanging it.
pub(crate) async fn bounded<F: Future>(future: F) -> F::Output {
    tokio::time::timeout(Duration::from_secs(10), future)
        .await
        .expect("the operation finishes within the test bound")
}

/// Invokes `controller` with an empty query after queueing one control event
/// that names `task_id`.
pub(crate) async fn control_turn(
    controller: &BaseController,
    session: &Arc<dyn Session>,
    task_id: &str,
    text: &str,
) -> ControllerOutput {
    let mut event = Event::user_input(text);
    event.task_id = Some(task_id.to_owned());
    assert!(controller.push_event(event));
    Controller::invoke(controller, json!({"query": ""}), Some(Arc::clone(session)))
        .await
        .unwrap()
}

/// Adds one running (`Submitted`) task for `session_id` and returns its id.
pub(crate) fn submitted_task(controller: &BaseController, session_id: &str) -> String {
    let task = Task::new(session_id, format!("{session_id}_manual"), "agent")
        .with_status(TaskStatus::Submitted);
    let task_id = task.task_id.clone();
    controller
        .task_manager()
        .add_task(task)
        .expect("a fresh task id is unique");
    task_id
}
