//! Behavior tests for trace spans, frames, sinks, and the tracer.
//!
//! Ports the observable contract of `openjiuwen/core/session/tracer/`
//! (v0.1.20): `span.py` (`Span`, `TraceAgentSpan`, `TraceWorkflowSpan`,
//! `SpanManager`), `handler.py` (`TraceAgentHandler`, `TraceWorkflowHandler`),
//! and `tracer.py` (`Tracer`). Python's tracer tests live in
//! `tests/unit_tests/core/session/tracer/`.
//!
//! Rust collapses Python's span classes into one [`TraceSpan`] with a kind
//! discriminator, so the tests assert the `camelCase` wire keys and the two
//! frame types (`tracer_agent`, `tracer_workflow`) instead of class identity.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;
use std::time::Duration;

use futures::stream::BoxStream;
use futures::StreamExt;
use openjiuwen_common::error::Result;
use openjiuwen_session::stream::{
    StreamEvent, StreamHub, StreamItem, StreamMode, StreamWriterManager,
};
use openjiuwen_session::{
    InvokeType, MemoryTraceSink, SpanHandle, SpanStatus, StreamTraceSink, TraceFrame,
    TraceFrameType, TracePosition, TraceSink, TraceSpan, TraceSpanKind, Tracer,
};
use serde_json::{json, Value};

/// Upper bound on how long a deterministic in-memory receive may take.
const RECV_TIMEOUT: Duration = Duration::from_secs(1);

/// Receive one event, failing the test instead of hanging when none arrives.
async fn recv(stream: &mut BoxStream<'static, Result<StreamEvent>>) -> StreamEvent {
    tokio::time::timeout(RECV_TIMEOUT, stream.next())
        .await
        .expect("a stream event arrives before the timeout")
        .expect("the stream is still open")
        .expect("the stream yields Ok events")
}

/// Receive one frame item, asserting that it is neither `End` nor late.
async fn recv_item(stream: &mut BoxStream<'static, Result<StreamEvent>>) -> StreamItem {
    match recv(stream).await {
        StreamEvent::Item(item) => item,
        StreamEvent::End => panic!("the stream ended before the expected frame"),
    }
}

/// Build a tracer recording into a fresh memory sink.
fn recording_tracer() -> (Tracer, Arc<MemoryTraceSink>) {
    let sink = Arc::new(MemoryTraceSink::new());
    let tracer = Tracer::with_sink("session-1", "trace-1", sink.clone());
    (tracer, sink)
}

/// The only recorded frame, asserting exactly one was recorded.
fn only_frame(sink: &MemoryTraceSink) -> TraceFrame {
    let frames = sink.frames();
    assert_eq!(frames.len(), 1, "exactly one frame is emitted");
    frames.into_iter().next().expect("the frame exists")
}

// ports `tests/unit_tests/core/session/tracer/test_span.py`
// (`TraceAgentSpan` start/finish with pydantic aliases).
#[test]
fn span_lifecycle_emits_one_frame_with_camel_case_keys() {
    let (tracer, sink) = recording_tracer();
    let span = tracer.start_span("answer", InvokeType::Llm);
    span.set_inputs(json!({ "question": "why?" }));
    span.set_metadata("tokens", json!(12));
    span.finish(json!({ "text": "because" }));

    assert_eq!(sink.len(), 1);
    let frame = only_frame(&sink);
    assert_eq!(frame.item_type, "tracer_agent");
    assert_eq!(frame.payload.status, SpanStatus::Finish);
    assert_eq!(frame.payload.outputs, Some(json!({ "text": "because" })));
    assert_eq!(frame.payload.inputs, Some(json!({ "question": "why?" })));
    assert_eq!(frame.payload.meta_data.get("tokens"), Some(&json!(12)));

    let wire = serde_json::to_value(&frame.payload).unwrap();
    assert_eq!(wire["traceId"], json!("trace-1"));
    assert_eq!(wire["sessionId"], json!("session-1"));
    assert_eq!(wire["invokeId"], json!(span.id()));
    assert_eq!(wire["name"], json!("answer"));
    assert_eq!(wire["invokeType"], json!("llm"));
    assert_eq!(wire["spanKind"], json!("agent"));
    assert_eq!(wire["status"], json!("finish"));
    assert!(wire["startTime"].is_u64(), "startTime is epoch millis");
    assert!(wire["endTime"].is_u64(), "endTime is epoch millis");
    assert!(wire["elapsedTime"].is_u64(), "elapsedTime is millis");
    assert_eq!(wire["childInvokes"], json!([]));
    assert!(wire.get("parentInvokeId").is_some(), "the key is stable");
    assert_eq!(wire["parentInvokeId"], Value::Null);

    // The frame itself keeps `type` as the wire key, not `item_type`.
    let frame_wire = serde_json::to_value(&frame).unwrap();
    assert_eq!(frame_wire["type"], json!("tracer_agent"));
    assert_eq!(frame_wire["payload"]["invokeId"], json!(span.id()));
}

// ports `SpanManager._refresh_parent_child_span` (parent/child bookkeeping).
#[test]
fn child_span_links_parent_and_grows_child_invokes() {
    let (tracer, sink) = recording_tracer();
    let parent = tracer.start_span("agent", InvokeType::Prompt);
    let child = tracer.start_child(&parent, "llm", InvokeType::Llm);

    assert_eq!(child.trace_id(), parent.trace_id());
    assert_eq!(child.snapshot().session_id, parent.snapshot().session_id);
    assert_eq!(child.snapshot().parent_invoke_id, Some(parent.id()));
    assert_eq!(parent.snapshot().child_invokes, vec![child.id()]);
    assert_eq!(child.snapshot().kind, TraceSpanKind::Agent);

    child.finish(Value::Null);
    parent.finish(Value::Null);

    let frames = sink.frames();
    assert_eq!(frames.len(), 2);
    assert_eq!(frames[0].payload.parent_invoke_id, Some(parent.id()));
    assert_eq!(frames[1].payload.child_invokes, vec![child.id()]);
}

// ports `TraceAgentSpan` finish/error status handling.
#[test]
fn fail_marks_the_span_as_error_with_the_message() {
    let (tracer, sink) = recording_tracer();
    let span = tracer.start_span("answer", InvokeType::Llm);
    span.fail("model unavailable");

    let frame = only_frame(&sink);
    assert_eq!(frame.payload.status, SpanStatus::Error);
    assert_eq!(frame.payload.error.as_deref(), Some("model unavailable"));
    assert!(frame.payload.end_time.is_some());
    assert!(frame.payload.elapsed_time.is_some());

    let wire = serde_json::to_value(&frame.payload).unwrap();
    assert_eq!(wire["status"], json!("error"));
    assert_eq!(wire["error"], json!("model unavailable"));
}

// ports the interrupted status produced by a workflow interrupt.
#[test]
fn interrupted_marks_the_span_and_keeps_the_reason() {
    let (tracer, sink) = recording_tracer();
    let span = tracer.start_span("answer", InvokeType::Prompt);
    span.interrupted("waiting for user input");

    let wire = serde_json::to_value(&only_frame(&sink).payload).unwrap();
    assert_eq!(wire["status"], json!("interrupted"));
    assert_eq!(wire["error"], json!("waiting for user input"));
    assert!(wire["endTime"].is_u64());
}

// ports the single-shot handler contract: one span contributes one frame.
#[test]
fn repeated_finish_emits_exactly_once() {
    let (tracer, sink) = recording_tracer();
    let span = tracer.start_span("answer", InvokeType::Llm);

    span.finish(json!({ "first": true }));
    span.finish(json!({ "second": true }));
    span.fail("ignored after finish");
    span.interrupted("also ignored");

    let frame = only_frame(&sink);
    assert_eq!(frame.payload.outputs, Some(json!({ "first": true })));
    assert_eq!(frame.payload.status, SpanStatus::Finish);
}

// ports `Tracer` span creation defaults (`name` falls back to the span kind).
#[test]
fn empty_names_and_kinds_get_defaults() {
    let (tracer, _sink) = recording_tracer();

    let agent = tracer.start_span("", InvokeType::Prompt);
    assert_eq!(agent.snapshot().name, "agent");
    assert_eq!(agent.snapshot().kind, TraceSpanKind::Agent);

    let component = tracer.start_workflow_span("node-1", "");
    let span = component.snapshot();
    assert_eq!(span.name, "component");
    assert_eq!(span.kind, TraceSpanKind::Component);
    assert_eq!(span.component_id.as_deref(), Some("node-1"));
    assert_eq!(span.workflow_id.as_deref(), Some("node-1"));
    assert_eq!(span.invoke_type, Some(InvokeType::Workflow));
}

// ports `TraceWorkflowSpan` component fields and the workflow handler frame.
#[test]
fn workflow_span_carries_component_fields_and_a_workflow_frame() {
    let (tracer, sink) = recording_tracer();
    let span = tracer.start_workflow_span("node-1", "search");
    span.finish(json!({ "hits": 2 }));

    let frame = only_frame(&sink);
    assert_eq!(frame.item_type, TraceFrameType::Workflow.as_str());
    let wire = serde_json::to_value(&frame.payload).unwrap();
    assert_eq!(wire["componentId"], json!("node-1"));
    assert_eq!(wire["workflowId"], json!("node-1"));
    assert_eq!(wire["spanKind"], json!("component"));
    assert_eq!(wire["name"], json!("search"));
}

// ports the `SpanManager` span builder helpers (`append_child_invoke_id`).
#[test]
fn span_builders_set_the_documented_fields() {
    let span = TraceSpan::new(
        "trace-9",
        "session-9",
        "invoke-9",
        "step",
        TraceSpanKind::Workflow,
    )
    .with_invoke_type(InvokeType::Chain)
    .with_parent("invoke-0")
    .with_component("comp-3", "Composer", "llm")
    .with_workflow_id("flow-2");

    assert_eq!(span.parent_invoke_id.as_deref(), Some("invoke-0"));
    assert_eq!(span.component_id.as_deref(), Some("comp-3"));
    assert_eq!(span.component_name.as_deref(), Some("Composer"));
    assert_eq!(span.component_type.as_deref(), Some("llm"));
    assert_eq!(span.workflow_id.as_deref(), Some("flow-2"));
    assert_eq!(span.invoke_type, Some(InvokeType::Chain));
    assert_eq!(span.status, SpanStatus::Start);
    assert!(span.start_time.is_some());
    assert!(span.elapsed().is_some());
}

// ports `TraceAgentSpan.elapsed_time` (end - start).
#[test]
fn elapsed_uses_the_recorded_times() {
    let span = TraceSpan {
        start_time: Some(1_000),
        end_time: Some(1_400),
        ..TraceSpan::default()
    };
    assert_eq!(span.elapsed(), Some(400));

    let out_of_order = TraceSpan {
        start_time: Some(1_400),
        end_time: Some(1_000),
        ..TraceSpan::default()
    };
    assert_eq!(out_of_order.elapsed(), Some(0), "underflow saturates");
    assert_eq!(TraceSpan::default().elapsed(), None);
}

// ports the recording handler used by Python's tracer unit tests.
#[test]
fn memory_sink_records_takes_and_tracks_position() {
    let (tracer, sink) = recording_tracer();
    assert!(sink.is_empty());
    assert_eq!(sink.len(), 0);
    assert_eq!(sink.position(), None, "nothing emitted yet");

    let first = tracer.start_span("first", InvokeType::Prompt);
    first.finish(Value::Null);
    let second = tracer.start_span("second", InvokeType::Llm);
    second.finish(Value::Null);

    assert_eq!(sink.len(), 2);
    assert!(!sink.is_empty());
    let position = sink.position().expect("a position after emission");
    assert_eq!(
        position,
        TracePosition {
            trace_id: "trace-1".to_owned(),
            last_invoke_id: Some(second.id()),
            frame_count: 2,
        }
    );

    let taken = sink.take();
    assert_eq!(taken.len(), 2);
    assert_eq!(taken[0].payload.invoke_id, first.id());
    assert!(sink.is_empty());
    assert_eq!(
        sink.position().expect("the last invoke id survives take"),
        position
    );

    sink.clear();
    assert_eq!(sink.position(), None);
}

// ports the checkpoint resume hook: `set_position` continues a trace.
#[test]
fn memory_sink_set_position_resets_the_frame_baseline() {
    let (tracer, sink) = recording_tracer();
    sink.set_position(TracePosition {
        trace_id: "trace-1".to_owned(),
        last_invoke_id: Some("invoke-before".to_owned()),
        frame_count: 5,
    });

    assert_eq!(
        sink.position(),
        Some(TracePosition {
            trace_id: "trace-1".to_owned(),
            last_invoke_id: Some("invoke-before".to_owned()),
            frame_count: 5,
        })
    );

    let span = tracer.start_span("resumed", InvokeType::Prompt);
    span.finish(Value::Null);

    let position = sink.position().expect("a position after emission");
    assert_eq!(position.frame_count, 6, "the baseline is respected");
    assert_eq!(position.last_invoke_id, Some(span.id()));
    assert_eq!(position.trace_id, "trace-1");
}

// ports `TraceAgentHandler`/`TraceWorkflowHandler`: built-in handlers write
// trace-mode frames onto the session stream.
#[tokio::test]
async fn stream_sink_publishes_agent_and_component_frames() {
    let hub = StreamHub::new();
    let mut stream = hub.subscribe();
    let writer = StreamWriterManager::new(Arc::clone(&hub));
    let tracer = Tracer::with_sink(
        "session-1",
        "trace-1",
        Arc::new(StreamTraceSink::new(writer)),
    );

    let agent = tracer.start_span("answer", InvokeType::Llm);
    agent.finish(json!({ "text": "hi" }));
    let component = tracer.start_workflow_span("node-1", "search");
    component.finish(Value::Null);

    let agent_item = recv_item(&mut stream).await;
    assert_eq!(agent_item.mode, StreamMode::Trace);
    assert_eq!(agent_item.item_type, "tracer_agent");
    assert_eq!(agent_item.index, None);
    assert_eq!(agent_item.payload["traceId"], json!("trace-1"));
    assert_eq!(agent_item.payload["invokeId"], json!(agent.id()));
    assert_eq!(agent_item.payload["status"], json!("finish"));

    let component_item = recv_item(&mut stream).await;
    assert_eq!(component_item.mode, StreamMode::Trace);
    assert_eq!(component_item.item_type, "tracer_workflow");
    assert_eq!(component_item.payload["invokeId"], json!(component.id()));
    assert_eq!(component_item.payload["componentId"], json!("node-1"));
}

// ports `TraceSchema` frame construction (`{"type", "payload"}`).
#[test]
fn to_stream_item_keeps_the_wire_shape() {
    let span = TraceSpan::new(
        "trace-1",
        "session-1",
        "invoke-1",
        "answer",
        TraceSpanKind::Agent,
    );
    let frame = TraceFrame::agent(span);

    let item = frame.to_stream_item().expect("the span serializes");
    assert_eq!(item.mode, StreamMode::Trace);
    assert_eq!(item.item_type, "tracer_agent");
    assert_eq!(item.index, None);
    assert_eq!(item.payload["invokeId"], json!("invoke-1"));

    let wire = item.to_wire();
    assert_eq!(wire["type"], json!("tracer_agent"));
    assert_eq!(wire["payload"]["invokeId"], json!("invoke-1"));
    assert!(wire.get("index").is_none(), "trace frames are unindexed");
}

// ports `Tracer()` with no handlers: frames are produced and dropped.
#[test]
fn tracer_new_installs_the_noop_sink() {
    let tracer = Tracer::new("session-1", "trace-1");
    assert_eq!(tracer.session_id(), "session-1");
    assert_eq!(tracer.trace_id(), "trace-1");
    assert!(
        format!("{:?}", tracer.sink()).contains("NoopTraceSink"),
        "the default sink is the noop sink"
    );

    // Emitting through the noop sink is a silent no-op, not an error.
    let span = tracer.start_span("answer", InvokeType::Prompt);
    span.finish(Value::Null);
    assert_eq!(span.snapshot().status, SpanStatus::Finish);

    // An empty trace id is generated, as in Python.
    let generated = Tracer::new("session-1", "");
    assert_ne!(generated.trace_id(), "");
    assert_ne!(generated.trace_id(), "session-1");
}

// ports `Tracer.init(stream_writer_manager)` replacing the frame consumer.
#[test]
fn set_sink_redirects_subsequent_frames() {
    let (tracer, first) = recording_tracer();
    let before = tracer.start_span("before", InvokeType::Prompt);

    let second = Arc::new(MemoryTraceSink::new());
    tracer.set_sink(second.clone());
    let after = tracer.start_span("after", InvokeType::Prompt);

    before.finish(Value::Null);
    after.finish(Value::Null);

    // The handle keeps the sink it started with; new handles use the new sink.
    assert_eq!(first.len(), 1);
    assert_eq!(first.frames()[0].payload.invoke_id, before.id());
    assert_eq!(second.len(), 1);
    assert_eq!(second.frames()[0].payload.invoke_id, after.id());
}

// ports `SpanManager.create_agent_span` descendents sharing one trace id.
#[test]
fn child_spans_inherit_the_parent_kind_and_ids() {
    let (tracer, sink) = recording_tracer();
    let parent = tracer.start_workflow_span("node-1", "search");
    let child = tracer.start_child(&parent, "", InvokeType::Plugin);

    let span = child.snapshot();
    assert_eq!(span.kind, TraceSpanKind::Component);
    assert_eq!(span.name, "component");
    assert_eq!(span.trace_id, "trace-1");
    assert_eq!(span.session_id, "session-1");
    assert_eq!(span.parent_invoke_id, Some(parent.id()));

    child.finish(Value::Null);
    let frame = only_frame(&sink);
    assert_eq!(frame.item_type, "tracer_workflow");
    assert_eq!(frame.payload.parent_invoke_id, Some(parent.id()));
}

// ports `Span` setters used by the handlers (inputs, outputs, status).
#[test]
fn setters_mutate_without_emitting() {
    let (tracer, sink) = recording_tracer();
    let span: SpanHandle = tracer.start_span("answer", InvokeType::Llm);

    span.set_inputs(json!({ "q": 1 }));
    span.set_outputs(json!({ "a": 2 }));
    span.set_status(SpanStatus::Running);
    span.set_metadata("model", json!("fake"));
    assert!(sink.is_empty(), "setters never emit frames");

    let snapshot = span.snapshot();
    assert_eq!(snapshot.status, SpanStatus::Running);
    assert_eq!(snapshot.inputs, Some(json!({ "q": 1 })));
    assert_eq!(snapshot.outputs, Some(json!({ "a": 2 })));
    assert_eq!(snapshot.meta_data.get("model"), Some(&json!("fake")));
    assert_eq!(snapshot.invoke_id, span.id());
    assert_eq!(snapshot.trace_id, span.trace_id());
}

// ports the idle-span elapsed computation (`Span.elapsed_time` before finish).
#[test]
fn open_spans_report_a_live_elapsed_time() {
    let (tracer, _sink) = recording_tracer();
    let span = tracer.start_span("answer", InvokeType::Llm);
    let snapshot = span.snapshot();

    assert_eq!(snapshot.end_time, None);
    assert!(snapshot.elapsed().is_some(), "an open span reports its age");
}
