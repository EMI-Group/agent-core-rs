#![allow(clippy::unwrap_used, clippy::expect_used)]
//! LLM-requested task control against the background scheduler.
//!
//! A turn's `PauseTask`/`CancelTask` intent must stop a task that is really
//! running, not just rewrite its stored status: the intent goes through
//! `BaseController::pause_task`/`cancel_task`, the executor hooks run, the run
//! is cancelled and awaited, and the recorded decision survives the driver's
//! own completion. These suites pin that route end to end.
mod common;

use async_trait::async_trait;
use openjiuwen_agent::controller::{
    BaseController, Controller, ControllerConfig, Event, LlmIntentRecognizer, Task, TaskContext,
    TaskExecutor, TaskResult, TaskStatus,
};
use openjiuwen_agent::AgentCard;
use openjiuwen_common::error::Result;
use openjiuwen_session::Session;
use openjiuwen_test_support::ScriptedModel;
use serde_json::json;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// The model answer that asks the controller to pause the event's task.
const PAUSE_INTENT: &str =
    r#"[{"intent_type":"pause_task","name":"pause","description":"pause the build"}]"#;
/// The model answer that asks the controller to cancel the event's task.
const CANCEL_INTENT: &str =
    r#"[{"intent_type":"cancel_task","name":"cancel","description":"cancel the build"}]"#;

/// An executor whose run outlives the controlling turn.
///
/// It returns a completion after a short sleep instead of waiting for its
/// cancellation token, so a manager-only status write would leave the driver
/// free to record `Completed` over it: the pause/cancel has to stop and record
/// the run itself.
struct SlowExecutor {
    started: Arc<AtomicBool>,
    pause_calls: Arc<AtomicUsize>,
    cancel_calls: Arc<AtomicUsize>,
    runs: Arc<AtomicUsize>,
}

impl SlowExecutor {
    fn new() -> Self {
        Self {
            started: Arc::new(AtomicBool::new(false)),
            pause_calls: Arc::new(AtomicUsize::new(0)),
            cancel_calls: Arc::new(AtomicUsize::new(0)),
            runs: Arc::new(AtomicUsize::new(0)),
        }
    }
}

#[async_trait]
impl TaskExecutor for SlowExecutor {
    fn task_type(&self) -> &'static str {
        "agent"
    }

    fn description(&self) -> &'static str {
        "a task that takes a moment to finish"
    }

    async fn execute(&self, _task: &Task, _ctx: &TaskContext<'_>) -> Result<TaskResult> {
        self.runs.fetch_add(1, Ordering::SeqCst);
        self.started.store(true, Ordering::SeqCst);
        tokio::time::sleep(Duration::from_millis(60)).await;
        Ok(TaskResult::completed("finished"))
    }

    async fn can_pause(&self, _task: &Task, _ctx: &TaskContext<'_>) -> (bool, String) {
        (true, String::new())
    }

    async fn pause(&self, _task: &Task, _ctx: &TaskContext<'_>) -> bool {
        self.pause_calls.fetch_add(1, Ordering::SeqCst);
        true
    }

    async fn can_cancel(&self, _task: &Task, _ctx: &TaskContext<'_>) -> (bool, String) {
        (true, String::new())
    }

    async fn cancel(&self, _task: &Task, _ctx: &TaskContext<'_>) -> bool {
        self.cancel_calls.fetch_add(1, Ordering::SeqCst);
        true
    }
}

/// Builds a controller whose scheduler is running `agent` for session `s1`.
///
/// The task is inserted into the shared store, the loop is woken, and the test
/// waits until the executor is really running it before returning.
async fn running_task(executor: Arc<SlowExecutor>) -> (BaseController, Arc<dyn Session>, String) {
    let config = ControllerConfig::new()
        // The controller must not restore the session's (empty) task snapshot
        // over the task the test inserted.
        .with_enable_task_persistence(false);
    let controller = BaseController::new(AgentCard::new("ctrl"), config);
    controller.add_executor("agent", Arc::clone(&executor) as Arc<dyn TaskExecutor>);
    controller.start();
    let session = common::session("s1");
    controller.bind_session(Arc::clone(&session)).await;
    let task = Task::new("s1", "manual", "agent");
    let task_id = task.task_id.clone();
    controller
        .task_manager()
        .add_task(task)
        .expect("a fresh task id is unique");
    assert!(controller.wake_scheduler(), "a running loop was woken");
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    while !executor.started.load(Ordering::SeqCst) {
        assert!(
            tokio::time::Instant::now() < deadline,
            "the scheduler never started the task"
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    (controller, session, task_id)
}

/// Queues `event` as the pause/cancel request and runs one turn.
///
/// The turn's own event carries an empty query, so it plans no task (a blank
/// create intent fails validation) and only the queued control event counts.
async fn control_turn(
    controller: &BaseController,
    session: &Arc<dyn Session>,
    task_id: &str,
    text: &str,
) -> openjiuwen_agent::controller::ControllerOutput {
    let mut event = Event::user_input(text);
    event.task_id = Some(task_id.to_owned());
    assert!(controller.push_event(event), "the event queue accepted it");
    Controller::invoke(controller, json!({"query": ""}), Some(Arc::clone(session)))
        .await
        .expect("the control turn succeeds")
}

#[tokio::test]
async fn an_llm_pause_intent_stops_a_running_task() {
    let executor = Arc::new(SlowExecutor::new());
    let (controller, session, task_id) = running_task(Arc::clone(&executor)).await;
    let model = ScriptedModel::builder().respond_text(PAUSE_INTENT).build();
    controller.set_intent_recognizer(Arc::new(LlmIntentRecognizer::new(Arc::new(model.clone()))));
    let output = control_turn(&controller, &session, &task_id, "pause the build").await;
    assert_eq!(model.call_count(), 1, "one recognition, not one per event");
    assert_eq!(
        executor.pause_calls.load(Ordering::SeqCst),
        1,
        "the scheduler drove the executor's pause hook: {output:?}"
    );
    assert_eq!(
        controller
            .task_manager()
            .task(&task_id)
            .expect("the task is still stored")
            .status,
        TaskStatus::Paused
    );
    // Let the driver settle on its own: a completion recorded after the pause
    // would undo it, which is exactly the regression this test pins.
    controller
        .scheduler_loop()
        .expect("the controller started a loop")
        .wait_until_idle()
        .await;
    assert_eq!(
        controller
            .task_manager()
            .task(&task_id)
            .expect("the task is still stored")
            .status,
        TaskStatus::Paused,
        "the driver must not overwrite the recorded pause"
    );
    controller.shutdown().await;
}

#[tokio::test]
async fn an_llm_cancel_intent_stops_a_running_task() {
    let executor = Arc::new(SlowExecutor::new());
    let (controller, session, task_id) = running_task(Arc::clone(&executor)).await;
    let model = ScriptedModel::builder().respond_text(CANCEL_INTENT).build();
    controller.set_intent_recognizer(Arc::new(LlmIntentRecognizer::new(Arc::new(model.clone()))));
    let output = control_turn(&controller, &session, &task_id, "cancel the build").await;
    assert_eq!(model.call_count(), 1, "one recognition, not one per event");
    assert_eq!(
        executor.cancel_calls.load(Ordering::SeqCst),
        1,
        "the scheduler drove the executor's cancel hook: {output:?}"
    );
    assert_eq!(
        controller
            .task_manager()
            .task(&task_id)
            .expect("the task is still stored")
            .status,
        TaskStatus::Canceled
    );
    controller
        .scheduler_loop()
        .expect("the controller started a loop")
        .wait_until_idle()
        .await;
    assert_eq!(
        controller
            .task_manager()
            .task(&task_id)
            .expect("the task is still stored")
            .status,
        TaskStatus::Canceled,
        "the driver must not overwrite the recorded cancel"
    );
    controller.shutdown().await;
}

#[tokio::test]
async fn a_pause_intent_for_an_unknown_task_changes_nothing() {
    let executor = Arc::new(SlowExecutor::new());
    let (controller, session, task_id) = running_task(Arc::clone(&executor)).await;
    let model = ScriptedModel::builder().respond_text(PAUSE_INTENT).build();
    controller.set_intent_recognizer(Arc::new(LlmIntentRecognizer::new(Arc::new(model.clone()))));
    let output = control_turn(&controller, &session, "nope", "pause the build").await;
    assert!(output.output.contains("not found"), "{}", output.output);
    assert_eq!(executor.pause_calls.load(Ordering::SeqCst), 0);
    // The run was never touched, so it finishes on its own.
    controller
        .scheduler_loop()
        .expect("the controller started a loop")
        .wait_until_idle()
        .await;
    assert_eq!(
        controller
            .task_manager()
            .task(&task_id)
            .expect("the running task is untouched")
            .status,
        TaskStatus::Completed
    );
    controller.shutdown().await;
}
