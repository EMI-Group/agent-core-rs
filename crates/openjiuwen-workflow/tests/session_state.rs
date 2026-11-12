#![allow(clippy::unwrap_used, clippy::expect_used)]
#![allow(
    unreachable_pub,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    clippy::new_without_default,
    clippy::module_name_repetitions,
    clippy::unnecessary_literal_bound
)]
//! Session state, tracing, cancellation, and session-stream behavior.
//!
//! Ports of the Python workflow/session unit tests in
//! `tests/unit_tests/core/session/test_session.py`,
//! `tests/unit_tests/core/session/tracer/test_workflow_tracer.py`, and
//! `tests/unit_tests/core/workflow/test_workflow.py`, rewritten against the Rust public
//! API. Everything is offline and deterministic: state lives in an in-process
//! `WorkflowSession`, trace frames go to a `MemoryTraceSink`, and the cancellation test
//! synchronises through a `Notify` instead of sleeping.
mod common;

use std::sync::Arc;

use async_trait::async_trait;
use futures::{FutureExt, StreamExt};
use openjiuwen_common::error::{ErrorKind, Result};
use openjiuwen_session::stream::{StreamEvent, StreamItem, StreamMode};
use openjiuwen_session::{MemoryTraceSink, Session, SpanStatus};
use serde_json::{json, Value};
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;

use openjiuwen_workflow::{
    Component, ComponentAbility, ComponentCard, ComponentContext, ComponentOptions, EndComponent,
    StartComponent, Workflow, WorkflowExecutionState, WorkflowRunOptions,
};

use common::{card, session as new_session};

/// Keeps a per-component invocation counter in its own session lane and records the
/// last component id that ran in the session's global lane.
#[derive(Debug, Clone, Default)]
struct StatefulComponent;

impl StatefulComponent {
    /// Creates a stateful component.
    fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Component for StatefulComponent {
    fn card(&self) -> ComponentCard {
        ComponentCard::new("stateful", "stateful").with_abilities(self.abilities())
    }

    fn abilities(&self) -> Vec<ComponentAbility> {
        vec![ComponentAbility::Invoke]
    }

    fn component_type(&self) -> &str {
        "stateful"
    }

    async fn invoke(&self, _inputs: Value, ctx: &ComponentContext) -> Result<Value> {
        let previous = ctx.get_component_state("runs").await?;
        let runs = previous.and_then(|value| value.as_u64()).unwrap_or(0) + 1;
        ctx.update_component_state(json!({"runs": runs})).await?;
        ctx.update_state(json!({"last": ctx.component_id()}))
            .await?;
        Ok(json!({"runs": runs}))
    }
}

/// Writes a global key and returns the global lane as seen by `ComponentContext::dump_state`.
#[derive(Debug, Clone, Default)]
struct DumpStateComponent;

impl DumpStateComponent {
    /// Creates a dump-state component.
    fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Component for DumpStateComponent {
    fn card(&self) -> ComponentCard {
        ComponentCard::new("dump_state", "dump_state").with_abilities(self.abilities())
    }

    fn abilities(&self) -> Vec<ComponentAbility> {
        vec![ComponentAbility::Invoke]
    }

    fn component_type(&self) -> &str {
        "dump_state"
    }

    async fn invoke(&self, _inputs: Value, ctx: &ComponentContext) -> Result<Value> {
        ctx.update_state(json!({"dumped": true})).await?;
        let dump = ctx.dump_state().await?;
        Ok(json!({"dump": dump}))
    }
}

/// Signals `ready` when it starts, then returns `{"ready": true}`.
#[derive(Debug, Clone)]
struct SignalComponent {
    ready: Arc<Notify>,
}

impl SignalComponent {
    /// Creates a component signalling `ready` on entry.
    fn new(ready: Arc<Notify>) -> Self {
        Self { ready }
    }
}

#[async_trait]
impl Component for SignalComponent {
    fn card(&self) -> ComponentCard {
        ComponentCard::new("signal", "signal").with_abilities(self.abilities())
    }

    fn abilities(&self) -> Vec<ComponentAbility> {
        vec![ComponentAbility::Invoke]
    }

    fn component_type(&self) -> &str {
        "signal"
    }

    async fn invoke(&self, _inputs: Value, _ctx: &ComponentContext) -> Result<Value> {
        self.ready.notify_one();
        Ok(json!({"ready": true}))
    }
}

/// `start → a → b → end`, where both middle nodes are [`StatefulComponent`]s and the end
/// reports their counters.
fn stateful_chain(id: &str) -> Workflow {
    let mut workflow = Workflow::new(card(id));
    workflow
        .set_start_comp("start", StartComponent::new(), ComponentOptions::new())
        .add_workflow_comp("a", StatefulComponent::new(), ComponentOptions::new())
        .add_workflow_comp("b", StatefulComponent::new(), ComponentOptions::new())
        .set_end_comp(
            "end",
            EndComponent::new(),
            ComponentOptions::new().with_inputs_schema(json!({
                "a": "${a.runs}",
                "b": "${b.runs}",
            })),
        )
        .add_connection("start", "a")
        .add_connection("a", "b")
        .add_connection("b", "end");
    workflow
}

/// `start → ask → end`, where `ask` asks `question` through the component context.
fn asking_workflow(id: &str, question: Value) -> Workflow {
    let mut workflow = Workflow::new(card(id));
    workflow
        .set_start_comp("start", StartComponent::new(), ComponentOptions::new())
        .add_workflow_comp(
            "ask",
            common::InteractingComponent::new(question),
            ComponentOptions::new(),
        )
        .set_end_comp(
            "end",
            EndComponent::new(),
            ComponentOptions::new().with_inputs_schema(json!({"answer": "${ask.answer}"})),
        )
        .add_connection("start", "ask")
        .add_connection("ask", "end");
    workflow
}

/// `start(mid) → end` where `mid` echoes its inputs, for tracing tests.
fn traced_workflow(id: &str) -> Workflow {
    let mut workflow = Workflow::new(card(id));
    workflow
        .set_start_comp("start", StartComponent::new(), ComponentOptions::new())
        .add_workflow_comp("mid", common::EchoComponent::new(), ComponentOptions::new())
        .set_end_comp(
            "end",
            EndComponent::new(),
            ComponentOptions::new().with_inputs_schema(json!({"echo": "${mid.echo}"})),
        )
        .add_connection("start", "mid")
        .add_connection("mid", "end");
    workflow
}

/// `start → end` with a streaming end rendering `{{producer}}`.
fn streaming_end_workflow(id: &str) -> Workflow {
    let mut workflow = Workflow::new(card(id));
    workflow
        .set_start_comp("start", StartComponent::new(), ComponentOptions::new())
        .add_workflow_comp(
            "producer",
            common::StreamingProducer::new(vec![json!("a"), json!("b")]),
            ComponentOptions::new(),
        )
        .set_end_comp(
            "end",
            common::end_with_template("{{producer}}"),
            ComponentOptions::new()
                .with_stream_inputs_schema(json!({"producer": "${producer}"}))
                .streaming(),
        )
        .add_connection("start", "producer")
        .add_stream_connection("producer", "end");
    workflow
}

/// Drains the frames the session stream already buffered, without awaiting new ones.
fn ready_events(
    events: &mut futures::stream::BoxStream<'static, Result<StreamEvent>>,
) -> Vec<StreamItem> {
    let mut items = Vec::new();
    while let Some(Some(Ok(event))) = events.next().now_or_never() {
        if let StreamEvent::Item(item) = event {
            items.push(item);
        }
    }
    items
}

// port of tests/unit_tests/core/session/test_session.py::TestSession::test_basic
#[tokio::test]
async fn component_state_is_namespaced_per_component_id() {
    let workflow = stateful_chain("state-namespace");
    let session: Arc<dyn Session> = new_session("state-namespace");

    let first = workflow
        .invoke(json!({"query": "one"}), Arc::clone(&session))
        .await
        .unwrap();
    assert_eq!(first.state, WorkflowExecutionState::Completed);
    assert_eq!(first.result, json!({"output": {"a": 1, "b": 1}}));

    // The second run continues both counters: each component reads its own lane.
    let second = workflow
        .invoke(json!({"query": "two"}), Arc::clone(&session))
        .await
        .unwrap();
    assert_eq!(second.result, json!({"output": {"a": 2, "b": 2}}));

    assert_eq!(
        session.get_component_state("a").await.unwrap(),
        Some(json!({"runs": 2}))
    );
    assert_eq!(
        session.get_component_state("b").await.unwrap(),
        Some(json!({"runs": 2}))
    );
}

// port of tests/unit_tests/core/workflow/test_workflow.py::test_workflow_inputs_in_session
#[tokio::test]
async fn update_state_lands_in_the_session_global_lane() {
    let workflow = stateful_chain("state-global");
    let session: Arc<dyn Session> = new_session("state-global");

    workflow
        .invoke(json!({"query": "hello"}), Arc::clone(&session))
        .await
        .unwrap();

    // `b` runs last, so its `ctx.update_state` is the final global write.
    assert_eq!(session.get_state("last").await.unwrap(), Some(json!("b")));
    // The run inputs are merged into the global lane (Python's `commit_user_inputs`).
    assert_eq!(
        session.get_state("query").await.unwrap(),
        Some(json!("hello"))
    );
}

// port of tests/unit_tests/core/session/test_session.py::TestSession::test_get_by_schema
#[tokio::test]
async fn dump_state_returns_the_global_lane_as_one_object() {
    let mut workflow = Workflow::new(card("state-dump"));
    workflow
        .set_start_comp("start", StartComponent::new(), ComponentOptions::new())
        .add_workflow_comp("dump", DumpStateComponent::new(), ComponentOptions::new())
        .set_end_comp(
            "end",
            EndComponent::new(),
            ComponentOptions::new().with_inputs_schema(json!({"dump": "${dump.dump}"})),
        )
        .add_connection("start", "dump")
        .add_connection("dump", "end");

    let output = workflow
        .invoke(json!({"query": "hello"}), new_session("state-dump"))
        .await
        .unwrap();

    let dump = &output.result["output"]["dump"];
    assert!(dump.is_object(), "dump state must be an object: {dump}");
    assert_eq!(dump["dumped"], json!(true));
    assert_eq!(dump["query"], json!("hello"));
}

// port of tests/unit_tests/core/session/tracer/test_workflow_tracer.py::test_stream_workflow_with_trace
#[tokio::test]
async fn every_executed_component_emits_one_finished_span() {
    let session: Arc<dyn Session> = new_session("trace-finish");
    let sink = Arc::new(MemoryTraceSink::new());
    session
        .tracer()
        .expect("workflow sessions own a tracer")
        .set_sink(sink.clone());

    let output = traced_workflow("trace-finish")
        .invoke(json!({"query": "hi"}), Arc::clone(&session))
        .await
        .unwrap();
    assert_eq!(output.state, WorkflowExecutionState::Completed);

    let frames = sink.frames();
    let ids: Vec<&str> = frames
        .iter()
        .map(|frame| frame.payload.component_id.as_deref().unwrap_or_default())
        .collect();
    assert_eq!(
        ids,
        vec!["start", "mid", "end"],
        "one tracer_workflow frame per executed component"
    );
    for frame in &frames {
        assert_eq!(frame.item_type, "tracer_workflow");
        assert_eq!(frame.payload.status, SpanStatus::Finish, "{frame:?}");
        assert!(
            frame.payload.outputs.is_some(),
            "a finished span records its outputs: {frame:?}"
        );
    }
}

// port of tests/unit_tests/core/session/tracer/test_workflow_tracer.py::test_interactive_workflow_with_trace
#[tokio::test]
async fn an_interrupted_run_records_an_interrupted_span() {
    let session: Arc<dyn Session> = new_session("trace-interrupt");
    let sink = Arc::new(MemoryTraceSink::new());
    session
        .tracer()
        .expect("workflow sessions own a tracer")
        .set_sink(sink.clone());

    let output = asking_workflow("trace-interrupt", json!({"id": "ask", "question": "go?"}))
        .invoke(json!({"query": "hi"}), Arc::clone(&session))
        .await
        .unwrap();
    assert_eq!(output.state, WorkflowExecutionState::InputRequired);

    let frames = sink.frames();
    let start = frames
        .iter()
        .find(|frame| frame.payload.component_id.as_deref() == Some("start"))
        .expect("the start span is recorded");
    assert_eq!(start.payload.status, SpanStatus::Finish);

    let ask = frames
        .iter()
        .find(|frame| frame.payload.component_id.as_deref() == Some("ask"))
        .expect("the interrupted component span is recorded");
    assert_eq!(
        ask.payload.status,
        SpanStatus::Interrupted,
        "the engine marks an interaction as interrupted: {ask:?}"
    );
    assert!(
        ask.payload.error.is_some(),
        "an interrupted span records its reason"
    );
    assert!(
        frames
            .iter()
            .all(|frame| frame.payload.component_id.as_deref() != Some("end")),
        "the end node never ran"
    );
}

// port of tests/unit_tests/core/workflow/test_workflow.py::test_workflow_cancel
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_cancelled_running_workflow_fails_with_cancelled() {
    let ready = Arc::new(Notify::new());
    let mut workflow = Workflow::new(card("cancel-running"));
    workflow
        .set_start_comp(
            "start",
            SignalComponent::new(Arc::clone(&ready)),
            ComponentOptions::new(),
        )
        .add_workflow_comp(
            "wait",
            common::WaitOnCancellationComponent::new(),
            ComponentOptions::new(),
        )
        .set_end_comp("end", EndComponent::new(), ComponentOptions::new())
        .add_connection("start", "wait")
        .add_connection("wait", "end");

    let token = CancellationToken::new();
    let options = WorkflowRunOptions::default().with_cancellation(token.clone());
    let run = tokio::spawn(async move {
        workflow
            .invoke_with(
                json!({"query": "hi"}),
                new_session("cancel-running"),
                options,
            )
            .await
    });

    // The start node signals that the run is in flight; no sleep is needed.
    ready.notified().await;
    token.cancel();

    let failure = run.await.unwrap().unwrap_err();
    assert_eq!(failure.kind(), ErrorKind::Cancelled, "{failure}");
}

// port of tests/unit_tests/core/workflow/test_workflow.py::test_workflow_cancel
#[tokio::test]
async fn a_pre_cancelled_token_never_invokes_a_component() {
    let counter = common::CounterComponent::new();
    let mut workflow = Workflow::new(card("cancel-before-start"));
    workflow
        .set_start_comp("start", counter.clone(), ComponentOptions::new())
        .add_workflow_comp(
            "wait",
            common::WaitOnCancellationComponent::new(),
            ComponentOptions::new(),
        )
        .set_end_comp("end", EndComponent::new(), ComponentOptions::new())
        .add_connection("start", "wait")
        .add_connection("wait", "end");

    let token = CancellationToken::new();
    token.cancel();
    let failure = workflow
        .invoke_with(
            json!({}),
            new_session("cancel-before-start"),
            WorkflowRunOptions::default().with_cancellation(token),
        )
        .await
        .unwrap_err();

    assert_eq!(failure.kind(), ErrorKind::Cancelled, "{failure}");
    assert_eq!(counter.count(), 0, "a cancelled run must not run any node");
}

// port of tests/unit_tests/core/workflow/test_workflow.py::test_simple_stream_workflow
#[tokio::test]
async fn a_streaming_end_publishes_its_frames_on_the_session_stream() {
    let session: Arc<dyn Session> = new_session("session-stream");
    let mut events = session.subscribe();

    let output = streaming_end_workflow("session-stream")
        .invoke(json!({"query": "hi"}), Arc::clone(&session))
        .await
        .unwrap();

    // `invoke` reports the collected `end node stream` wires ...
    assert_eq!(
        output.result,
        json!([
            {"type": "end node stream", "index": 0, "payload": {"response": "a"}},
            {"type": "end node stream", "index": 1, "payload": {"response": "b"}},
        ])
    );

    // ... and the same frames went through `Session::write_stream` into the hub.
    let published: Vec<Value> = ready_events(&mut events)
        .into_iter()
        .filter(|item| item.mode == StreamMode::Output)
        .filter(|item| item.item_type == "end node stream")
        .map(|item| item.payload)
        .collect();
    assert_eq!(
        published,
        vec![json!({"response": "a"}), json!({"response": "b"})]
    );
}
