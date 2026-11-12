//! Lazy-open tests of the trace-frame sink.
//!
//! The session tracer emits exactly one frame per span — the terminal frame of
//! `SpanHandle::complete` — so these tests drive `Tracer::with_sink` directly and
//! assert that the sink opens the span from that first frame. Python
//! counterpart: the `on_*_start` + `on_*_end` callback pair, which Python invokes
//! separately.
//!
//! Helpers, the exporter fixture, and the sentinel constants live beside the
//! other sink tests in the parent module.

// A test module owns its panics: a failed `unwrap`/`expect` is an assertion.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;
use std::time::{Duration, SystemTime};

use opentelemetry::trace::{SpanId, SpanKind, Status};
use serde_json::json;

use openjiuwen_core::session::tracer::{
    InvokeType, SpanStatus, TraceFrame, TraceSink, Tracer as SessionTracer,
};

use crate::otel::semconv::{
    ERROR_TYPE, GEN_AI_OPERATION_CHAT, GEN_AI_OPERATION_NAME, GEN_AI_OPERATION_RETRIEVAL,
    GEN_AI_OUTPUT_MESSAGES, GEN_AI_RETRIEVAL_DOCUMENTS, OJ_CHILD_INVOKE_IDS, OJ_ERROR,
    OJ_INVOKE_ID, OJ_PARENT_INVOKE_ID, OJ_STATUS, OJ_TRACE_ID, OJ_WORKFLOW_COMPONENT_ID,
    OJ_WORKFLOW_OUTPUTS, SPAN_STATUS_ERROR, SPAN_STATUS_FINISH, SPAN_STATUS_INTERRUPTED,
    SPAN_STATUS_RUNNING,
};

use super::{agent_start, array_attribute, attribute, attribute_values, terminal, typed_component};
use super::{Fixture, SESSION, TRACE};

/// An epoch-millisecond frame timestamp as the instant the span records.
fn epoch(millis: u64) -> SystemTime {
    SystemTime::UNIX_EPOCH + Duration::from_millis(millis)
}

#[test]
fn a_real_tracer_emits_one_span_without_a_synthetic_start_frame() {
    // test_handler.py: test_llm_start_creates_span / test_llm_end_ends_span,
    // driven through the session Tracer: the tracer emits only the terminal
    // frame (`SpanHandle::complete`), so the sink must open the span lazily.
    let fixture = Fixture::new();
    let sink: Arc<dyn TraceSink> = fixture.sink.clone();
    let tracer = SessionTracer::with_sink(SESSION, TRACE, sink);
    let span = tracer.start_span("Model", InvokeType::Llm);
    span.set_inputs(json!({"messages": [{"role": "user", "content": "hi"}]}));
    let start = span
        .snapshot()
        .start_time
        .expect("the span records its start");
    span.finish(json!({"outputs": {"content": "hello"}}));
    let end = span.snapshot().end_time.expect("the span records its end");
    assert_eq!(fixture.capture_len(), 1, "exactly one span is exported");
    let data = fixture.span("chat Model");
    assert_eq!(data.status, Status::Ok);
    assert_eq!(data.span_kind, SpanKind::Client);
    assert_eq!(
        attribute(&data, OJ_STATUS).as_deref(),
        Some(SPAN_STATUS_FINISH)
    );
    assert_eq!(attribute(&data, OJ_TRACE_ID).as_deref(), Some(TRACE));
    assert_eq!(
        attribute(&data, OJ_INVOKE_ID).as_deref(),
        Some(span.id().as_str())
    );
    assert_eq!(
        attribute(&data, GEN_AI_OPERATION_NAME).as_deref(),
        Some(GEN_AI_OPERATION_CHAT)
    );
    assert!(attribute(&data, GEN_AI_OUTPUT_MESSAGES).is_some_and(|text| text.contains("hello")));
    // The terminal-only frame carries a real duration.
    assert_eq!(data.start_time, epoch(start));
    assert_eq!(data.end_time, epoch(end));
    assert!(data.end_time >= data.start_time);
}

#[test]
fn a_failed_tracer_span_exports_an_error_span() {
    // test_handler.py: test_llm_error_marks_span_error, terminal frame only.
    let fixture = Fixture::new();
    let sink: Arc<dyn TraceSink> = fixture.sink.clone();
    let tracer = SessionTracer::with_sink(SESSION, TRACE, sink);
    let span = tracer.start_span("Model", InvokeType::Llm);
    span.fail("boom");
    let data = fixture.span("chat Model");
    assert!(matches!(data.status, Status::Error { .. }));
    assert_eq!(
        attribute(&data, OJ_STATUS).as_deref(),
        Some(SPAN_STATUS_ERROR)
    );
    let payload = attribute(&data, OJ_ERROR).expect("the error payload");
    assert!(payload.contains("boom"), "{payload}");
    assert_eq!(attribute(&data, ERROR_TYPE).as_deref(), Some("100102"));
    assert_eq!(fixture.capture_len(), 1);
}

#[test]
fn an_interrupted_tracer_span_keeps_the_span_successful() {
    // test_handler.py: the GraphInterrupt branch keeps the span successful.
    let fixture = Fixture::new();
    let sink: Arc<dyn TraceSink> = fixture.sink.clone();
    let tracer = SessionTracer::with_sink(SESSION, TRACE, sink);
    let span = tracer.start_span("Model", InvokeType::Llm);
    span.interrupted("awaiting input");
    let data = fixture.span("chat Model");
    assert_eq!(
        attribute(&data, OJ_STATUS).as_deref(),
        Some(SPAN_STATUS_INTERRUPTED)
    );
    assert!(attribute(&data, OJ_ERROR).is_none());
    assert_eq!(data.status, Status::Unset);
    assert_eq!(fixture.capture_len(), 1);
}

#[test]
fn a_traced_workflow_span_exports_one_component_span() {
    // test_handler.py: test_workflow_call_start_creates_span, terminal frame
    // only. `start_workflow_span` records the component id as the workflow id
    // and leaves the component type unset, so the component classifier falls
    // back to `component.{invoke_id}`.
    let fixture = Fixture::new();
    let sink: Arc<dyn TraceSink> = fixture.sink.clone();
    let tracer = SessionTracer::with_sink(SESSION, TRACE, sink);
    let span = tracer.start_workflow_span("comp-1", "Retriever");
    let invoke_id = span.id();
    span.finish(json!({"ok": true}));
    let data = fixture.span(&format!("component.{invoke_id}"));
    assert_eq!(data.span_kind, SpanKind::Internal);
    assert_eq!(data.status, Status::Ok);
    assert_eq!(
        attribute(&data, OJ_WORKFLOW_COMPONENT_ID).as_deref(),
        Some("comp-1")
    );
    assert_eq!(
        attribute(&data, OJ_WORKFLOW_OUTPUTS).as_deref(),
        Some("{\"ok\":true}")
    );
    assert_eq!(fixture.capture_len(), 1);
}

#[test]
fn a_lazily_opened_retrieval_component_uses_the_classifier_name() {
    // test_handler.py: test_retrieval_component_projects_the_gen_ai_attributes,
    // with the terminal frame as the first and only frame of the invoke id.
    let fixture = Fixture::new();
    let span = terminal(
        typed_component("comp-1", "KnowledgeRetrievalExecutable", "Retriever"),
        SpanStatus::Finish,
        Some(json!({"results": ["doc-1"]})),
    );
    fixture.emit(span);
    let data = fixture.span("retrieval Retriever");
    assert_eq!(
        attribute(&data, OJ_WORKFLOW_COMPONENT_ID).as_deref(),
        Some("comp-1")
    );
    assert_eq!(
        attribute(&data, GEN_AI_OPERATION_NAME).as_deref(),
        Some(GEN_AI_OPERATION_RETRIEVAL)
    );
    assert_eq!(
        array_attribute(&data, GEN_AI_RETRIEVAL_DOCUMENTS),
        Some(vec!["doc-1".to_owned()])
    );
    // `terminal(..)` stamps start 1_000 / end 1_250, so the lazy open keeps a
    // 250 ms duration instead of collapsing to the local clock.
    assert_eq!(
        data.end_time
            .duration_since(data.start_time)
            .expect("the end follows the start"),
        Duration::from_millis(250)
    );
    assert_eq!(fixture.capture_len(), 1);
}

#[test]
fn running_frames_without_a_start_frame_open_and_keep_the_span() {
    // The session tracer emits only the terminal frame, so a running frame is
    // normally the first frame of its invoke id; it must open the span and leave
    // it open for the terminal frame.
    let fixture = Fixture::new();
    let mut agent = agent_start("inv-run", "Model", InvokeType::Llm);
    agent.status = SpanStatus::Running;
    agent.inputs = Some(json!({"messages": [{"role": "user", "content": "hi"}]}));
    fixture.emit(agent.clone());
    assert_eq!(fixture.capture_len(), 0, "a running frame stays open");
    let mut done = agent;
    done.status = SpanStatus::Finish;
    done.outputs = Some(json!({"content": "hello"}));
    done.end_time = Some(1_250);
    fixture.emit(done);
    let agent_data = fixture.span("chat Model");
    assert_eq!(agent_data.status, Status::Ok);
    let statuses = attribute_values(&agent_data, OJ_STATUS);
    assert_eq!(
        statuses.first().map(String::as_str),
        Some(SPAN_STATUS_RUNNING)
    );
    assert_eq!(
        statuses.last().map(String::as_str),
        Some(SPAN_STATUS_FINISH)
    );

    let mut component = typed_component("comp-run", "StartExecutable", "step");
    component.status = SpanStatus::Running;
    fixture.emit(component.clone());
    let mut done = component;
    done.status = SpanStatus::Finish;
    done.end_time = Some(1_250);
    fixture.emit(done);
    let component_data = fixture.span("component.comp-run");
    let statuses = attribute_values(&component_data, OJ_STATUS);
    assert_eq!(
        statuses.first().map(String::as_str),
        Some(SPAN_STATUS_RUNNING)
    );
    assert_eq!(
        statuses.last().map(String::as_str),
        Some(SPAN_STATUS_FINISH)
    );
    assert_eq!(fixture.capture_len(), 2);
}

#[test]
fn terminal_frames_without_a_start_frame_open_their_span() {
    // A terminal frame is the only frame the session tracer emits, so it must
    // lazily open the span it closes — for both dimensions and every terminal
    // status, with the frame's own start/end timestamps.
    let fixture = Fixture::new();
    let cases = [
        (SpanStatus::Finish, SPAN_STATUS_FINISH),
        (SpanStatus::Error, SPAN_STATUS_ERROR),
        (SpanStatus::Interrupted, SPAN_STATUS_INTERRUPTED),
    ];
    for (index, (status, expected_status)) in cases.into_iter().enumerate() {
        let mut agent = agent_start(
            &format!("agent-{index}"),
            &format!("Model{index}"),
            InvokeType::Llm,
        );
        agent.status = status;
        agent.error = Some("boom".to_owned());
        agent.end_time = Some(1_250);
        fixture.emit(agent);
        let agent_data = fixture.span(&format!("chat Model{index}"));
        assert_eq!(
            attribute(&agent_data, OJ_STATUS).as_deref(),
            Some(expected_status)
        );
        assert_eq!(
            agent_data
                .end_time
                .duration_since(agent_data.start_time)
                .expect("the end follows the start"),
            Duration::from_millis(250)
        );

        let mut component = typed_component(&format!("wf-{index}"), "StartExecutable", "step");
        component.status = status;
        component.error = Some("boom".to_owned());
        component.end_time = Some(1_250);
        fixture.emit(component);
        let component_data = fixture.span(&format!("component.wf-{index}"));
        assert_eq!(
            attribute(&component_data, OJ_STATUS).as_deref(),
            Some(expected_status)
        );
        assert_eq!(
            attribute(&component_data, OJ_WORKFLOW_COMPONENT_ID).as_deref(),
            Some(format!("wf-{index}").as_str())
        );
    }
    assert_eq!(fixture.capture_len(), 6);
}

#[test]
fn a_traced_child_span_links_to_its_open_parent() {
    // test_span_manager.py: test_parent_context_links_the_child_to_the_open_parent.
    // The parent is opened by its start frame (the Start-then-terminal sequence);
    // the child needs no frames of its own beyond the terminal one.
    let fixture = Fixture::new();
    let sink: Arc<dyn TraceSink> = fixture.sink.clone();
    let tracer = SessionTracer::with_sink(SESSION, TRACE, sink);
    let parent = tracer.start_span("Parent", InvokeType::Llm);
    let parent_id = parent.id();
    fixture.sink.emit(TraceFrame::agent(parent.snapshot()));
    let child = tracer.start_child(&parent, "Child", InvokeType::Llm);
    let child_id = child.id();
    child.finish(json!({"content": "hi"}));
    parent.finish(json!({"content": "done"}));

    let parent_data = fixture.span("chat Parent");
    let child_data = fixture.span("chat Child");
    assert_eq!(
        child_data.parent_span_id,
        parent_data.span_context.span_id()
    );
    assert_eq!(
        child_data.span_context.trace_id(),
        parent_data.span_context.trace_id()
    );
    assert_eq!(
        attribute(&child_data, OJ_PARENT_INVOKE_ID).as_deref(),
        Some(parent_id.as_str())
    );
    assert_eq!(
        attribute(&child_data, OJ_INVOKE_ID).as_deref(),
        Some(child_id.as_str())
    );
    let children = attribute(&parent_data, OJ_CHILD_INVOKE_IDS).expect("the child ids");
    assert!(children.contains(&child_id), "{children}");
    assert_eq!(fixture.capture_len(), 2);
}

#[test]
fn a_child_finished_before_its_parent_is_still_exported() {
    // The child's terminal frame arrives before the parent has any state, so the
    // parent context cannot resolve and the child becomes a trace root; the
    // attribute-level links are still preserved and both spans are exported.
    let fixture = Fixture::new();
    let sink: Arc<dyn TraceSink> = fixture.sink.clone();
    let tracer = SessionTracer::with_sink(SESSION, TRACE, sink);
    let parent = tracer.start_span("Parent", InvokeType::Llm);
    let parent_id = parent.id();
    let child = tracer.start_child(&parent, "Child", InvokeType::Llm);
    let child_id = child.id();
    child.finish(json!({"content": "hi"}));
    parent.finish(json!({"content": "done"}));

    let parent_data = fixture.span("chat Parent");
    let child_data = fixture.span("chat Child");
    assert_eq!(child_data.parent_span_id, SpanId::INVALID);
    assert_eq!(
        attribute(&child_data, OJ_PARENT_INVOKE_ID).as_deref(),
        Some(parent_id.as_str())
    );
    let children = attribute(&parent_data, OJ_CHILD_INVOKE_IDS).expect("the child ids");
    assert!(children.contains(&child_id), "{children}");
    assert_eq!(fixture.capture_len(), 2);
}

#[test]
fn a_child_finished_after_its_parent_state_was_popped_is_still_exported() {
    // Parent-first ordering: the parent's terminal frame pops its state, so the
    // child's parent context falls back to none — the child span must still be
    // exported, with the parent invoke id preserved as an attribute.
    let fixture = Fixture::new();
    let sink: Arc<dyn TraceSink> = fixture.sink.clone();
    let tracer = SessionTracer::with_sink(SESSION, TRACE, sink);
    let parent = tracer.start_span("Parent", InvokeType::Llm);
    let parent_id = parent.id();
    let child = tracer.start_child(&parent, "Child", InvokeType::Llm);
    parent.finish(json!({"content": "done"}));
    child.finish(json!({"content": "hi"}));

    let parent_data = fixture.span("chat Parent");
    let child_data = fixture.span("chat Child");
    assert_eq!(parent_data.status, Status::Ok);
    assert_eq!(child_data.status, Status::Ok);
    assert_eq!(child_data.parent_span_id, SpanId::INVALID);
    assert_eq!(
        attribute(&child_data, OJ_PARENT_INVOKE_ID).as_deref(),
        Some(parent_id.as_str())
    );
    assert_eq!(fixture.capture_len(), 2);
}
