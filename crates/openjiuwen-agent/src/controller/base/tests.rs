//! Routing, dispatch, aggregation, and streaming behaviour of a controller
//! turn.
//!
//! The definitions these tests build on live in `fixtures`: the stub executor,
//! the stub event handler, the counting recognizer, and the turn helpers.
use super::fixtures::{
    controller_with_executor, test_session, CountingRecognizer, StubExecutor, StubHandler,
};
use super::{BaseController, Controller, EventHandlerOutcome, TASKS_STATE_KEY};
use crate::card::AgentCard;
use crate::controller::config::{ControllerConfig, DefaultResponse};
use crate::controller::executors::{ExecutorInfo, LLMController};
use crate::controller::schema::{
    ControllerOutput, DataFrame, Event, Intent, OutputKind, TaskStatus,
};
use futures::StreamExt;
use openjiuwen_test_support::ScriptedModel;
use serde_json::json;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

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
