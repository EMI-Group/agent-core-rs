//! Routing, dispatch, aggregation, streaming, and lifecycle behaviour of a
//! controller.
//!
//! The definitions these tests build on live in `fixtures`: the stub and
//! cooperative executors, the stub event handler, the counting recognizer, the
//! recording session, and the turn helpers.
use super::fixtures::{
    await_counter, controller_with_cooperative_executor, controller_with_executor,
    recording_session, submitted_task, test_session, CountingRecognizer, StubExecutor, StubHandler,
};
use super::{BaseController, Controller, EventHandlerOutcome, TASKS_STATE_KEY};
use crate::card::AgentCard;
use crate::controller::config::{ControllerConfig, DefaultResponse};
use crate::controller::executors::{ExecutorInfo, LLMController};
use crate::controller::group::GroupRouter;
use crate::controller::reasoner::TaskControl;
use crate::controller::schema::{
    ControllerOutput, ControllerOutputChunk, DataFrame, Event, Intent, OutputKind, Task, TaskStatus,
};
use crate::controller::task_manager::TaskManager;
use async_trait::async_trait;
use futures::StreamExt;
use openjiuwen_common::error::{ErrorKind, Result};
use openjiuwen_session::{Session, StreamItem};
use openjiuwen_test_support::ScriptedModel;
use parking_lot::Mutex;
use serde_json::{json, Value};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

#[tokio::test]
async fn router_selects_between_two_executors() {
    let (controller, agent_calls) = controller_with_executor();
    let (workflow, workflow_calls) = StubExecutor::new("workflow", "workflow reply");
    controller.add_executor("workflow", Arc::new(workflow));
    let model = Arc::new(ScriptedModel::builder().respond_text("workflow").build());
    controller.set_router(Arc::new(LLMController::new(
        model.clone(),
        vec![
            ExecutorInfo::new("agent", "agent", "runs an agent"),
            ExecutorInfo::new("workflow", "workflow", "runs a workflow"),
        ],
    )));
    let output = Controller::invoke(&controller, json!({"query": "do it"}), Some(test_session()))
        .await
        .unwrap();
    assert_eq!(output.output, "workflow reply");
    assert_eq!(output.status, TaskStatus::Completed);
    assert_eq!(model.call_count(), 1);
    assert_eq!(agent_calls.load(Ordering::SeqCst), 0);
    assert_eq!(workflow_calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn single_executor_skips_the_router() {
    let (controller, calls) = controller_with_executor();
    let model = Arc::new(ScriptedModel::builder().build());
    controller.set_router(Arc::new(LLMController::new(
        model.clone(),
        vec![ExecutorInfo::new("agent", "agent", "runs an agent")],
    )));
    let output = Controller::invoke(&controller, json!({"query": "do it"}), Some(test_session()))
        .await
        .unwrap();
    assert_eq!(output.output, "done");
    assert_eq!(model.call_count(), 0);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn unregistered_default_task_type_uses_the_default_response() {
    let config = ControllerConfig::new()
        .with_default_task_type("missing")
        .with_default_response(DefaultResponse::text("fallback"));
    let controller = BaseController::new(AgentCard::new("ctrl"), config);
    let session = test_session();
    let output = Controller::invoke(
        &controller,
        json!({"query": "do it"}),
        Some(Arc::clone(&session)),
    )
    .await
    .unwrap();
    assert_eq!(output.output, "fallback");
    assert_eq!(output.status, TaskStatus::Failed);
    assert_eq!(
        output.data.as_ref().unwrap()["tasks"][0]["status"],
        json!("failed")
    );
    let tasks = controller.tasks().await;
    assert_eq!(tasks.len(), 1);
    assert_eq!(tasks[0].status, TaskStatus::Failed);
    let state = session.get_state(TASKS_STATE_KEY).await.unwrap().unwrap();
    assert_eq!(state["tasks"][0]["status"], json!("failed"));
}

#[tokio::test]
async fn stream_starts_with_processing_and_ends_with_last_chunk() {
    let (controller, _) = controller_with_executor();
    let mut stream = Controller::stream(&controller, json!({"query": "go"}), Some(test_session()));
    let mut chunks = Vec::new();
    while let Some(item) = stream.next().await {
        chunks.push(item.unwrap());
    }
    assert!(chunks.len() >= 2);
    assert_eq!(
        chunks[0].payload.as_ref().unwrap().kind,
        OutputKind::Processing
    );
    let last = chunks.last().unwrap();
    assert!(last.last_chunk);
    assert_eq!(
        last.payload.as_ref().unwrap().kind,
        OutputKind::TaskCompletion
    );
    assert_eq!(chunks.iter().filter(|chunk| chunk.last_chunk).count(), 1);
}

#[tokio::test]
async fn event_handler_intents_override_the_recognizer() {
    let (controller, _) = controller_with_executor();
    let intent =
        Intent::create_task(Event::user_input("handled"), "handled").with_name("from_handler");
    controller.set_event_handler(Arc::new(StubHandler {
        outcome: EventHandlerOutcome::Intents(vec![intent]),
    }));
    let output = Controller::invoke(&controller, json!({"query": "do it"}), Some(test_session()))
        .await
        .unwrap();
    assert_eq!(output.output, "done");
    let tasks = controller.tasks().await;
    assert_eq!(tasks.len(), 1);
    assert_eq!(tasks[0].description.as_deref(), Some("handled"));
    assert_eq!(tasks[0].metadata["intent_label"], json!("from_handler"));
}

#[tokio::test]
async fn event_handler_output_short_circuits_the_turn() {
    let (controller, calls) = controller_with_executor();
    controller.set_event_handler(Arc::new(StubHandler {
        outcome: EventHandlerOutcome::Output(Box::new(ControllerOutput::new(
            "short",
            TaskStatus::Completed,
        ))),
    }));
    let output = Controller::invoke(&controller, json!({"query": "do it"}), Some(test_session()))
        .await
        .unwrap();
    assert_eq!(output.output, "short");
    assert_eq!(controller.tasks().await.len(), 0);
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn event_handler_ignored_falls_back() {
    let config = ControllerConfig::new().with_default_response(DefaultResponse::text("fallback"));
    let controller = BaseController::new(AgentCard::new("ctrl"), config);
    controller.set_event_handler(Arc::new(StubHandler {
        outcome: EventHandlerOutcome::Ignored,
    }));
    let output = Controller::invoke(&controller, json!({"query": "do it"}), Some(test_session()))
        .await
        .unwrap();
    assert_eq!(output.output, "fallback");
    assert_eq!(controller.tasks().await.len(), 0);
    controller.clear_event_handler();
}

#[tokio::test]
async fn task_interaction_event_marks_the_task_input_required() {
    let (controller, _) = controller_with_executor();
    let session = test_session();
    Controller::invoke(
        &controller,
        json!({"query": "build it"}),
        Some(Arc::clone(&session)),
    )
    .await
    .unwrap();
    let task_id = controller.tasks().await[0].task_id.clone();
    assert!(controller.push_event(Event::task_interaction(
        task_id.clone(),
        vec![DataFrame::text("need input")],
    )));
    let output = Controller::invoke(
        &controller,
        json!({"query": ""}),
        Some(Arc::clone(&session)),
    )
    .await
    .unwrap();
    assert_eq!(output.status, TaskStatus::InputRequired);
    assert_eq!(output.output, "need input");
    assert_eq!(output.task_id.as_deref(), Some(task_id.as_str()));
    assert_eq!(
        controller.tasks().await[0].status,
        TaskStatus::InputRequired
    );
    let state = session.get_state(TASKS_STATE_KEY).await.unwrap().unwrap();
    assert_eq!(state["tasks"][0]["status"], json!("input-required"));
}

#[tokio::test]
async fn max_intents_keeps_only_the_first() {
    let controller = BaseController::with_recognizer(
        AgentCard::new("ctrl"),
        ControllerConfig::new().with_max_intents(1),
        Arc::new(CountingRecognizer {
            calls: AtomicUsize::new(1),
        }),
    );
    controller.add_executor("agent", Arc::new(StubExecutor::new("agent", "done").0));
    Controller::invoke(&controller, json!({"query": "both"}), Some(test_session()))
        .await
        .unwrap();
    assert_eq!(controller.tasks().await.len(), 1);
}

/// Router that records the calls forwarded through it.
#[derive(Default)]
struct RecordingRouter {
    group_id: Option<&'static str>,
    sent: Mutex<Vec<(String, Option<String>)>>,
    published: Mutex<Vec<Option<String>>>,
}

#[async_trait]
impl GroupRouter for RecordingRouter {
    fn group_id(&self) -> Option<&str> {
        self.group_id
    }

    async fn send_to_agent(
        &self,
        agent_id: &str,
        _event: &Event,
        session: Option<Arc<dyn Session>>,
    ) -> Result<Value> {
        self.sent.lock().push((
            agent_id.to_owned(),
            session.as_ref().map(|session| session.id().to_owned()),
        ));
        Ok(json!({"target": agent_id}))
    }

    async fn publish(
        &self,
        _event: &Event,
        session: Option<Arc<dyn Session>>,
    ) -> Result<Vec<Value>> {
        self.published
            .lock()
            .push(session.as_ref().map(|session| session.id().to_owned()));
        Ok(vec![json!({"subscriber": 1}), json!({"subscriber": 2})])
    }
}

/// Number of `all_tasks_processed` frames among `frames`.
fn sentinel_count(frames: &[StreamItem]) -> usize {
    frames
        .iter()
        .filter_map(|item| ControllerOutputChunk::from_stream_item(item).ok().flatten())
        .filter(|chunk| {
            chunk
                .payload
                .as_ref()
                .is_some_and(|payload| payload.kind == OutputKind::AllTasksProcessed)
        })
        .count()
}

/// Stores one submitted task directly on the controller's manager.
fn add_manual_task(manager: &TaskManager, session_id: &str, task_id: &str, status: TaskStatus) {
    manager
        .add_task(Task::new(session_id, task_id, "agent").with_status(status))
        .expect("the manual task id is unique");
}

#[tokio::test]
async fn started_loop_runs_a_task_added_to_the_manager() {
    let (controller, calls) = controller_with_executor();
    controller.start();
    let scheduler = controller
        .scheduler_loop()
        .expect("start builds a scheduler loop inside a runtime");
    assert!(scheduler.is_running());
    assert!(controller.is_running());
    controller.bind_session(test_session()).await;

    let task_id = submitted_task(&controller, "s1");
    scheduler.wake();
    scheduler.wait_until_idle().await;

    let task = controller
        .task_manager()
        .task(&task_id)
        .expect("the task stays stored");
    assert_eq!(task.status, TaskStatus::Completed);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    controller.shutdown().await;
}

#[tokio::test]
async fn start_is_idempotent_and_stop_clears_the_running_flag() {
    let (controller, _calls) = controller_with_executor();
    controller.start();
    let first = controller.scheduler_loop().expect("a loop is started");
    controller.start();
    let second = controller.scheduler_loop().expect("the loop is kept");
    assert!(Arc::ptr_eq(&first, &second), "start spawns one loop");

    controller.stop();
    assert!(!controller.is_running());
    assert!(
        controller.scheduler_loop().is_none(),
        "stop clears the handle so a later start builds a fresh loop"
    );
    for _ in 0..200 {
        if !first.is_running() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    assert!(!first.is_running(), "the fire-and-forget stop landed");
    // Stopping twice, and shutting down a stopped controller, stay harmless.
    controller.stop();
    controller.shutdown().await;
    assert!(!controller.is_running());
}

#[tokio::test]
async fn shutdown_stops_the_loop_and_its_in_flight_run() {
    let (controller, started, finished) = controller_with_cooperative_executor();
    controller.start();
    let scheduler = controller.scheduler_loop().expect("a loop is started");
    controller.bind_session(test_session()).await;
    let task_id = submitted_task(&controller, "s1");
    scheduler.wake();
    await_counter(&started, 1).await;
    assert_eq!(scheduler.in_flight(), vec![task_id]);

    controller.shutdown().await;
    assert!(!controller.is_running());
    assert!(controller.scheduler_loop().is_none());
    assert!(!scheduler.is_running());
    assert!(scheduler.in_flight().is_empty(), "no run stays in flight");
    assert_eq!(finished.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn pause_and_cancel_fall_back_to_the_manager() {
    let controller = BaseController::new(AgentCard::new("ctrl"), ControllerConfig::new());
    let manager = controller.task_manager();
    assert!(controller.scheduler_loop().is_none());

    add_manual_task(&manager, "s1", "t1", TaskStatus::Submitted);
    assert!(!controller.pause_task("missing").await);
    assert!(controller.pause_task("t1").await);
    assert_eq!(manager.task("t1").unwrap().status, TaskStatus::Paused);
    assert!(
        !controller.pause_task("t1").await,
        "a paused task cannot be paused again"
    );
    assert!(
        !controller.cancel_task("t1").await,
        "a paused task needs the scheduler to be cancelled"
    );

    add_manual_task(&manager, "s1", "t2", TaskStatus::Working);
    assert!(controller.pause_task("t2").await);
    assert_eq!(manager.task("t2").unwrap().status, TaskStatus::Paused);

    add_manual_task(&manager, "s1", "t3", TaskStatus::Submitted);
    assert!(controller.cancel_task("t3").await);
    assert_eq!(manager.task("t3").unwrap().status, TaskStatus::Canceled);
    assert!(
        controller.cancel_task("t3").await,
        "cancelling a terminal task is idempotent"
    );
    assert!(!controller.cancel_task("missing").await);

    add_manual_task(&manager, "s1", "t4", TaskStatus::Completed);
    assert!(
        controller.cancel_task("t4").await,
        "terminal statuses are idempotent"
    );
}

#[tokio::test]
async fn pause_cancel_and_resume_delegate_to_the_started_loop() {
    let (controller, started, finished) = controller_with_cooperative_executor();
    controller.start();
    let scheduler = controller.scheduler_loop().expect("a loop is started");
    let manager = controller.task_manager();
    controller.bind_session(test_session()).await;
    let task_id = submitted_task(&controller, "s1");
    scheduler.wake();
    await_counter(&started, 1).await;

    assert!(controller.pause_task(&task_id).await);
    assert_eq!(manager.task(&task_id).unwrap().status, TaskStatus::Paused);
    assert!(finished.load(Ordering::SeqCst) >= 1, "the run was stopped");

    let control: Arc<dyn TaskControl> = Arc::new(controller.clone());
    assert!(!control.resume("missing").await);
    add_manual_task(&manager, "s1", "done", TaskStatus::Completed);
    assert!(
        !control.resume("done").await,
        "only a paused task can be resumed"
    );
    assert!(control.resume(&task_id).await);
    assert_eq!(
        manager.task(&task_id).unwrap().status,
        TaskStatus::Submitted
    );

    scheduler.wake();
    await_counter(&started, 2).await;
    assert!(controller.cancel_task(&task_id).await);
    assert_eq!(manager.task(&task_id).unwrap().status, TaskStatus::Canceled);
    controller.shutdown().await;
}

#[tokio::test]
async fn group_routing_forwards_and_fails_without_a_group() {
    let controller = BaseController::new(AgentCard::new("ctrl"), ControllerConfig::new());
    let event = Event::user_input("hi");

    let error = controller
        .send_to_agent("a1", &event, None)
        .await
        .expect_err("no group is installed");
    assert_eq!(error.kind(), ErrorKind::Config);
    assert_eq!(
        error.message(),
        "ctrl: Cannot send_to_agent('a1'). Agent is not part of a group with a controller."
    );
    let error = controller
        .publish(&event, None)
        .await
        .expect_err("no group is installed");
    assert_eq!(error.kind(), ErrorKind::Config);
    assert_eq!(
        error.message(),
        "ctrl: Cannot publish(). Agent is not part of a group with a controller."
    );

    let recorder = Arc::new(RecordingRouter {
        group_id: Some("group-1"),
        ..RecordingRouter::default()
    });
    controller.set_group(recorder.clone());
    assert!(controller.group().is_some());

    let value = controller
        .send_to_agent("a1", &event, Some(test_session()))
        .await
        .expect("the router forwards the send");
    assert_eq!(value, json!({"target": "a1"}));
    let values = controller
        .publish(&event, Some(test_session()))
        .await
        .expect("the router forwards the publish");
    assert_eq!(values.len(), 2);
    assert_eq!(
        recorder.sent.lock().as_slice(),
        [("a1".to_owned(), Some("s1".to_owned()))]
    );
    assert_eq!(
        recorder.published.lock().as_slice(),
        [Some("s1".to_owned())]
    );

    controller.clear_group();
    assert!(controller.group().is_none());
    assert!(controller.publish(&event, None).await.is_err());

    // A nameless card is named by its id in the failure.
    let nameless = BaseController::new(AgentCard::default(), ControllerConfig::new());
    let error = nameless
        .publish(&event, None)
        .await
        .expect_err("no group is installed");
    assert!(error.message().contains(nameless.card().id()));
}

#[tokio::test]
async fn with_group_wires_the_router_builder_style() {
    let event = Event::user_input("hi");
    let recorder = Arc::new(RecordingRouter {
        group_id: Some("group-1"),
        ..RecordingRouter::default()
    });
    let controller =
        BaseController::new(AgentCard::new("ctrl"), ControllerConfig::new()).with_group(recorder);
    let router = controller
        .group_router()
        .expect("the builder injected a group");
    assert_eq!(router.group_id(), Some("group-1"));
    assert!(
        controller.group().is_some(),
        "group() and group_router() read the same router"
    );

    let value = controller
        .send_to_agent("a1", &event, Some(test_session()))
        .await
        .expect("the injected router forwards the send");
    assert_eq!(value, json!({"target": "a1"}));

    controller.clear_group();
    assert!(controller.group_router().is_none());
    let error = controller
        .send_to_agent("a1", &event, None)
        .await
        .expect_err("the cleared group is gone");
    assert_eq!(error.kind(), ErrorKind::Config);
    assert_eq!(
        error.message(),
        "ctrl: Cannot send_to_agent('a1'). Agent is not part of a group with a controller."
    );
}

#[tokio::test]
async fn reasoner_creates_a_task_the_loop_runs() {
    let (controller, calls) = controller_with_executor();
    controller.start();
    controller.bind_session(test_session()).await;

    let intent = Intent::create_task(Event::user_input("build it"), "build it");
    let ops = controller.reasoner().plan_ops(&intent);
    assert_eq!(ops.len(), 1);
    let outcomes = controller.apply_task_ops(&ops, "s1", None).await;
    assert_eq!(outcomes.len(), 1);
    assert!(outcomes[0].applied, "{}", outcomes[0].message);
    assert_eq!(outcomes[0].status, Some(TaskStatus::Submitted));
    let task_id = outcomes[0]
        .task_id
        .clone()
        .expect("a created task carries its id");
    assert!(task_id.starts_with("s1_task_"));

    let scheduler = controller.scheduler_loop().expect("a loop is started");
    scheduler.wait_until_idle().await;
    let task = controller.task_manager().task(&task_id).expect("stored");
    assert_eq!(task.status, TaskStatus::Completed);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    controller.shutdown().await;
}

#[tokio::test]
async fn completion_sentinel_is_written_once_per_idle_stretch() {
    let (controller, _calls) = controller_with_executor();
    controller.start();
    let session = recording_session("s1");
    let handle: Arc<dyn Session> = session.clone();

    let output = Controller::invoke(&controller, json!({"query": "go"}), Some(handle))
        .await
        .expect("the turn succeeds");
    assert_eq!(output.output, "done");
    assert_eq!(
        sentinel_count(&session.written_frames().await),
        1,
        "the idle session got its sentinel"
    );

    let handle: Arc<dyn Session> = session.clone();
    Controller::invoke(&controller, json!({"query": "again"}), Some(handle))
        .await
        .expect("the second turn succeeds");
    assert_eq!(
        sentinel_count(&session.written_frames().await),
        1,
        "the same idle stretch is signalled once"
    );
    controller.shutdown().await;
}
