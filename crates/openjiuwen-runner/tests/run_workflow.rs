//! Workflow execution behaviour: reference resolution, session ownership,
//! callback order, error mapping, and streaming.
//!
//! Drives only the public `openjiuwen_runner::run` surface (`invoke_workflow`,
//! `stream_workflow`, `WorkflowRef`, `RunOptions`, `RunServices`).
//! Ports `tests/unit_tests/core/runner/test_runner.py` (`run_workflow`,
//! `run_workflow_streaming`, `_prepare_workflow`, `_create_workflow_session`).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::sync::Arc;

use futures::StreamExt;
use openjiuwen_runner::callback::CallbackEvent;
use openjiuwen_runner::run::{invoke_workflow, stream_workflow, RunOptions, WorkflowRef};
use openjiuwen_session::{Checkpointer, Session};
use openjiuwen_workflow::WorkflowCard;
use serde_json::json;

use common::{
    failing_workflow, passthrough_workflow, record_all, recorded_events, recorder,
    RecordingCheckpointer, RecordingSession, FAILING_COMPONENT_MESSAGE,
};

/// The result a [`passthrough_workflow`] run produces for `query`.
fn passthrough_result(query: &str) -> serde_json::Value {
    json!({"output": {"answer": query}})
}

// ---------------------------------------------------------------------------
// Plain runs
// ---------------------------------------------------------------------------

// ports tests/unit_tests/core/runner/test_runner.py::test_run_workflow
#[tokio::test]
async fn a_run_returns_the_reference_passthrough_result() {
    let services = common::services();
    let output = invoke_workflow(
        services,
        WorkflowRef::of(Arc::new(passthrough_workflow("wf"))),
        json!({"query": "hi"}),
        RunOptions::default(),
    )
    .await
    .unwrap();
    assert_eq!(output.result, passthrough_result("hi"));
}

// ports tests/unit_tests/core/runner/test_runner.py::test_run_workflow_with_id
#[tokio::test]
async fn an_id_reference_is_resolved_through_the_registry() {
    let services = common::services();
    let lookups = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let served = Arc::clone(&lookups);
    services
        .resources
        .register_workflow(
            WorkflowCard::new("registered-wf"),
            Arc::new(move || {
                served.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                passthrough_workflow("registered-wf")
            }),
        )
        .unwrap();

    let output = invoke_workflow(
        services,
        WorkflowRef::id("registered-wf"),
        json!({"query": "from the registry"}),
        RunOptions::default(),
    )
    .await
    .unwrap();

    assert_eq!(output.result, passthrough_result("from the registry"));
    assert_eq!(
        lookups.load(std::sync::atomic::Ordering::SeqCst),
        1,
        "the provider is invoked once per lookup"
    );
}

// ports tests/unit_tests/core/runner/test_runner.py::test_run_workflow_not_found
#[tokio::test]
async fn an_unregistered_id_fails_before_any_callback() {
    let services = common::services();
    let log = recorder();
    record_all(&services.callbacks, &common::workflow_events(), &log);

    let error = invoke_workflow(
        services,
        WorkflowRef::id("missing"),
        json!({"query": "hi"}),
        RunOptions::default(),
    )
    .await
    .unwrap_err();

    assert_eq!(error.kind(), openjiuwen_common::error::ErrorKind::NotFound);
    assert_eq!(
        error.to_string(),
        "not_found: workflow 'missing' is not registered"
    );
    assert!(
        recorded_events(&log).is_empty(),
        "a resolution failure fires no callback: {:?}",
        recorded_events(&log)
    );
}

// ports tests/unit_tests/core/runner/test_runner.py::test_run_workflow (callbacks)
#[tokio::test]
async fn callbacks_fire_input_before_output() {
    let services = common::services();
    let log = recorder();
    record_all(&services.callbacks, &common::workflow_events(), &log);

    invoke_workflow(
        services,
        WorkflowRef::of(Arc::new(passthrough_workflow("wf"))),
        json!({"query": "hi"}),
        RunOptions::default(),
    )
    .await
    .unwrap();

    assert_eq!(
        recorded_events(&log),
        vec![
            CallbackEvent::WorkflowInvokeInput,
            CallbackEvent::WorkflowInvokeOutput,
        ]
    );
    let entries = log.lock().clone();
    assert_eq!(
        entries.last().map(|entry| entry.component.clone()),
        Some("wf".to_owned()),
        "the events name the workflow card id"
    );
}

// ---------------------------------------------------------------------------
// Session ownership
// ---------------------------------------------------------------------------

// ports tests/unit_tests/core/runner/test_runner.py::test_run_workflow_with_session
#[tokio::test]
async fn a_caller_session_is_used_and_left_open() {
    let caller = RecordingSession::new("caller-workflow-session");
    let output = invoke_workflow(
        common::services(),
        WorkflowRef::of(Arc::new(passthrough_workflow("wf"))),
        json!({"query": "hi"}),
        RunOptions::default().with_session(Arc::clone(&caller) as Arc<dyn Session>),
    )
    .await
    .unwrap();

    assert_eq!(output.result, passthrough_result("hi"));
    assert!(
        caller.called("update_state"),
        "the engine wrote through the caller's session: {:?}",
        caller.calls()
    );
    assert!(
        !caller.called("close") && !caller.called("end_stream"),
        "the runner never closes or ends a caller-provided session: {:?}",
        caller.calls()
    );
}

/// An owned `WorkflowSession` is invisible to the test, so the session id and
/// the injected checkpointer are asserted through the checkpointer the created
/// session was built with (the only seam the public API exposes).
// ports tests/unit_tests/core/runner/test_runner.py::test_create_workflow_session
#[tokio::test]
async fn an_owned_session_uses_the_resolved_id_and_the_checkpointer() {
    let checkpointer = RecordingCheckpointer::new();
    let services =
        common::services_with_checkpointer(Arc::clone(&checkpointer) as Arc<dyn Checkpointer>);

    invoke_workflow(
        services,
        WorkflowRef::of(Arc::new(passthrough_workflow("wf"))),
        json!({"conversation_id": "conv-wf"}),
        RunOptions::default(),
    )
    .await
    .unwrap();

    assert_eq!(
        checkpointer.session_ids(),
        vec!["conv-wf".to_owned()],
        "the created session is named from `inputs.conversation_id`"
    );
    assert!(
        !checkpointer.calls_for("conv-wf").is_empty(),
        "the created session uses the runner's checkpointer"
    );
}

#[tokio::test]
async fn an_owned_session_takes_the_run_session_id_when_there_is_no_conversation() {
    let checkpointer = RecordingCheckpointer::new();
    let services =
        common::services_with_checkpointer(Arc::clone(&checkpointer) as Arc<dyn Checkpointer>);

    invoke_workflow(
        services,
        WorkflowRef::of(Arc::new(passthrough_workflow("wf"))),
        json!({}),
        RunOptions::default().with_session_id("caller-workflow-id"),
    )
    .await
    .unwrap();

    assert_eq!(
        checkpointer.session_ids(),
        vec!["caller-workflow-id".to_owned()]
    );
}

// ---------------------------------------------------------------------------
// Failures
// ---------------------------------------------------------------------------

// ports tests/unit_tests/core/runner/test_runner.py::test_run_workflow_error
#[tokio::test]
async fn a_failing_component_returns_the_error_and_fires_workflow_error() {
    let services = common::services();
    let log = recorder();
    record_all(&services.callbacks, &common::workflow_events(), &log);

    let error = invoke_workflow(
        services,
        WorkflowRef::of(Arc::new(failing_workflow("broken"))),
        json!({"query": "hi"}),
        RunOptions::default(),
    )
    .await
    .unwrap_err();

    assert!(
        error.to_string().contains(FAILING_COMPONENT_MESSAGE),
        "the component failure is reported: {error}"
    );
    assert_eq!(
        recorded_events(&log),
        vec![
            CallbackEvent::WorkflowInvokeInput,
            CallbackEvent::WorkflowError,
        ],
        "a failed run fires `WorkflowError` instead of `WorkflowInvokeOutput`"
    );
    let failure = log
        .lock()
        .iter()
        .find(|entry| entry.event == CallbackEvent::WorkflowError)
        .cloned()
        .expect("the error event was recorded");
    assert!(
        failure
            .error
            .as_deref()
            .is_some_and(|message| message.contains(FAILING_COMPONENT_MESSAGE)),
        "the error event carries the failure message: {failure:?}"
    );
    assert_eq!(failure.component, "broken");
}

// ---------------------------------------------------------------------------
// Streaming
// ---------------------------------------------------------------------------

// ports tests/unit_tests/core/runner/test_runner.py::test_run_workflow_streaming
#[tokio::test]
async fn streaming_fires_input_then_one_output_event_per_frame() {
    let services = common::services();
    let log = recorder();
    record_all(&services.callbacks, &common::workflow_events(), &log);

    let frames = stream_workflow(
        services,
        WorkflowRef::of(Arc::new(passthrough_workflow("wf"))),
        json!({"query": "hi"}),
        RunOptions::default(),
    )
    .collect::<Vec<_>>()
    .await;

    assert!(
        !frames.is_empty(),
        "a start → echo → end workflow streams at least one frame"
    );
    assert!(
        frames.iter().all(Result::is_ok),
        "no frame failed: {frames:?}"
    );

    let events = recorded_events(&log);
    assert_eq!(
        events.first(),
        Some(&CallbackEvent::WorkflowStreamInput),
        "`WorkflowStreamInput` is fired before the first frame: {events:?}"
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| **event == CallbackEvent::WorkflowStreamOutput)
            .count(),
        frames.len(),
        "every frame fires exactly one output event: {events:?}"
    );
    let indices = log
        .lock()
        .iter()
        .filter(|entry| entry.event == CallbackEvent::WorkflowStreamOutput)
        .filter_map(|entry| entry.index)
        .collect::<Vec<_>>();
    assert_eq!(
        indices,
        (0..i64::try_from(frames.len()).unwrap()).collect::<Vec<_>>(),
        "frame events are indexed from zero"
    );
}

#[tokio::test]
async fn streaming_reports_an_unregistered_id_as_its_only_item() {
    let frames = stream_workflow(
        common::services(),
        WorkflowRef::id("missing"),
        json!({}),
        RunOptions::default(),
    )
    .collect::<Vec<_>>()
    .await;

    assert_eq!(frames.len(), 1);
    assert_eq!(
        frames[0].as_ref().unwrap_err().to_string(),
        "not_found: workflow 'missing' is not registered"
    );
}

#[tokio::test]
async fn streaming_uses_a_caller_session_and_leaves_it_open() {
    let caller = RecordingSession::new("caller-stream-session");
    let frames = stream_workflow(
        common::services(),
        WorkflowRef::of(Arc::new(passthrough_workflow("wf"))),
        json!({"query": "hi"}),
        RunOptions::default().with_session(Arc::clone(&caller) as Arc<dyn Session>),
    )
    .collect::<Vec<_>>()
    .await;

    assert!(
        frames.iter().all(Result::is_ok),
        "no frame failed: {frames:?}"
    );
    assert!(
        !caller.called("close") && !caller.called("end_stream"),
        "the runner never closes a caller-provided session: {:?}",
        caller.calls()
    );
}

/// Python's third precedence step: without a conversation id and without a run
/// option, the created session takes `RunnerConfig::default_session_id`.
#[tokio::test]
async fn an_owned_session_falls_back_to_the_config_default() {
    let checkpointer = RecordingCheckpointer::new();
    let services = common::services_with_checkpointer_and_default_session(
        Arc::clone(&checkpointer) as Arc<dyn Checkpointer>,
        "workflow-default",
    );

    invoke_workflow(
        services,
        WorkflowRef::of(Arc::new(passthrough_workflow("wf"))),
        json!({}),
        RunOptions::default(),
    )
    .await
    .unwrap();

    assert_eq!(
        checkpointer.session_ids(),
        vec!["workflow-default".to_owned()]
    );
}
