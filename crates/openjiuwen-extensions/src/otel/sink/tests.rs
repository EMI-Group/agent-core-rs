//! Unit tests of the trace-frame sink.
//!
//! Port of the frame-level assertions of
//! `tests/unit_tests/extensions/tracer_otel/test_handler.py`
//! (`TestOtelAgentHandler`, `TestOtelWorkflowHandler`,
//! `TestRefreshInputsAttribute`) and of `test_otel_tool_span_dedup.py`'s
//! single-span expectations, adapted to the Rust frame API: one frame per
//! lifecycle transition, `meta_data` carrying what Python passed as keyword
//! arguments, and `interrupted` as its own status instead of a `GraphInterrupt`
//! object.

// A test module owns its panics: a failed `unwrap`/`expect` is an assertion.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;

use opentelemetry::trace::{SpanId, SpanKind, Status};
use opentelemetry::{Array, Value as OtelValue};
use opentelemetry_sdk::trace::SpanData;
use serde_json::{json, Value};

use openjiuwen_core::session::tracer::{
    InvokeType, SpanStatus, TraceFrame, TraceSink, TraceSpan, TraceSpanKind,
    Tracer as SessionTracer,
};

use super::super::config::{ExporterType, OtelTracerConfig};
use super::super::handle::{init_otel_tracer, TracerHandle};
use super::super::provider::CaptureExporter;
use super::super::semconv::{
    ERROR_TYPE, GEN_AI_DATA_SOURCE_ID, GEN_AI_INPUT_MESSAGES, GEN_AI_OPERATION_CHAT,
    GEN_AI_OPERATION_EXECUTE_TOOL, GEN_AI_OPERATION_NAME, GEN_AI_OPERATION_RETRIEVAL,
    GEN_AI_OUTPUT_MESSAGES, GEN_AI_PROVIDER_NAME, GEN_AI_PROVIDER_OPENJIUWEN, GEN_AI_REQUEST_MODEL,
    GEN_AI_RESPONSE_FINISH_REASONS, GEN_AI_RETRIEVAL_DOCUMENTS, GEN_AI_RETRIEVAL_QUERY_TEXT,
    GEN_AI_RETRIEVAL_TOP_K, GEN_AI_SYSTEM_INSTRUCTIONS, GEN_AI_TOOL_CALL_ID, GEN_AI_TOOL_NAME,
    GEN_AI_TOOL_TYPE, GEN_AI_USAGE_INPUT_TOKENS, OJ_AGENT_ERROR_MESSAGE, OJ_AGENT_INPUTS,
    OJ_AGENT_INVOKE_TYPE, OJ_AGENT_NAME, OJ_AGENT_OUTPUTS, OJ_ELAPSED_TIME, OJ_END_TIME, OJ_ERROR,
    OJ_PARENT_NODE_ID, OJ_SOURCE_IDS, OJ_STATUS, OJ_STREAM_INPUTS, OJ_STREAM_OUTPUTS, OJ_TRACE_ID,
    OJ_WORKFLOW_COMPONENT_ID, OJ_WORKFLOW_COMPONENT_NAME, OJ_WORKFLOW_COMPONENT_TYPE,
    OJ_WORKFLOW_ERROR_MESSAGE, OJ_WORKFLOW_EXECUTION_ID, OJ_WORKFLOW_ID, OJ_WORKFLOW_INPUTS,
    OJ_WORKFLOW_INVOKE_DATA, OJ_WORKFLOW_NAME, OJ_WORKFLOW_OUTPUTS, OJ_WORKFLOW_VERSION,
    SPAN_STATUS_ERROR, SPAN_STATUS_FINISH, SPAN_STATUS_INTERRUPTED, SPAN_STATUS_RUNNING,
};
use super::OtelTraceSink;

/// Trace id every test frame carries.
const TRACE: &str = "trace-1";

/// Session id every test frame carries.
const SESSION: &str = "session-1";

/// Configuration with readable (unredacted) attribute values.
fn plain_config() -> OtelTracerConfig {
    OtelTracerConfig {
        redaction_enabled: false,
        ..OtelTracerConfig::default()
    }
    .with_exporter_type(ExporterType::Memory)
}

/// A sink plus the handle and exporter that observe its spans.
struct Fixture {
    /// Keeps the provider (and therefore the batch processor) alive.
    handle: TracerHandle,
    /// Span buffer shared with the provider's exporter.
    capture: Arc<CaptureExporter>,
    /// Sink under test.
    sink: Arc<OtelTraceSink>,
}

impl Fixture {
    /// Builds a fixture over `config`.
    fn build(config: OtelTracerConfig) -> Self {
        let handle = init_otel_tracer(config).expect("the memory exporter initializes");
        let capture = handle
            .capture()
            .expect("the memory exporter captures spans");
        let sink = OtelTraceSink::new(&handle).expect("a live handle builds a sink");
        Self {
            handle,
            capture,
            sink,
        }
    }

    /// A fixture with unredacted attributes.
    fn new() -> Self {
        Self::build(plain_config())
    }

    /// Exported spans, after flushing the batch processor.
    fn spans(&self) -> Vec<SpanData> {
        self.handle.force_flush();
        self.capture.spans()
    }

    /// The one exported span named `name`.
    fn span(&self, name: &str) -> SpanData {
        self.handle.force_flush();
        self.capture
            .span_named(name)
            .unwrap_or_else(|| panic!("no span named {name} in {:?}", self.capture.names()))
    }

    /// Emits the frame of `span` through the sink.
    fn emit(&self, span: TraceSpan) {
        self.sink.emit(frame_of(span));
    }
}

/// Wraps a span in the frame type its kind maps to.
fn frame_of(span: TraceSpan) -> TraceFrame {
    match span.kind {
        TraceSpanKind::Agent => TraceFrame::agent(span),
        TraceSpanKind::Workflow | TraceSpanKind::Component => TraceFrame::workflow(span),
    }
}

/// An empty frame with every base field set.
fn base_span(invoke_id: &str, name: &str, kind: TraceSpanKind) -> TraceSpan {
    TraceSpan {
        trace_id: TRACE.to_owned(),
        session_id: SESSION.to_owned(),
        invoke_id: invoke_id.to_owned(),
        name: name.to_owned(),
        kind,
        status: SpanStatus::Start,
        start_time: Some(1_000),
        ..TraceSpan::default()
    }
}

/// An agent start frame of `invoke_type`.
fn agent_start(invoke_id: &str, name: &str, invoke_type: InvokeType) -> TraceSpan {
    let mut span = base_span(invoke_id, name, TraceSpanKind::Agent);
    span.invoke_type = Some(invoke_type);
    span
}

/// A workflow component frame of the given identity.
fn component(invoke_id: &str) -> TraceSpan {
    let mut span = base_span(invoke_id, invoke_id, TraceSpanKind::Component);
    span.component_id = Some(invoke_id.to_owned());
    span.component_name = Some(invoke_id.to_owned());
    span.component_type = Some("StartExecutable".to_owned());
    span.workflow_id = Some("wf1".to_owned());
    span
}

/// A component frame of `component_type`, named after `component_name`.
fn typed_component(invoke_id: &str, component_type: &str, component_name: &str) -> TraceSpan {
    let mut span = base_span(invoke_id, component_name, TraceSpanKind::Component);
    span.component_id = Some(invoke_id.to_owned());
    span.component_name = Some(component_name.to_owned());
    span.component_type = Some(component_type.to_owned());
    span.workflow_id = Some("wf1".to_owned());
    span
}

/// The terminal frame of `span`.
fn terminal(mut span: TraceSpan, status: SpanStatus, outputs: Option<Value>) -> TraceSpan {
    span.status = status;
    span.outputs = outputs;
    span.end_time = Some(1_250);
    span.elapsed_time = Some(250);
    span
}

/// The string value of one attribute; the last occurrence wins.
fn attribute(data: &SpanData, key: &str) -> Option<String> {
    data.attributes
        .iter()
        .rev()
        .find(|pair| pair.key.as_str() == key)
        .map(|pair| pair.value.as_str().into_owned())
}

/// Every value recorded under one attribute key, in write order.
fn attribute_values(data: &SpanData, key: &str) -> Vec<String> {
    data.attributes
        .iter()
        .filter(|pair| pair.key.as_str() == key)
        .map(|pair| pair.value.as_str().into_owned())
        .collect()
}

/// The string-array value of one attribute.
fn array_attribute(data: &SpanData, key: &str) -> Option<Vec<String>> {
    data.attributes
        .iter()
        .rev()
        .find(|pair| pair.key.as_str() == key)
        .and_then(|pair| match &pair.value {
            OtelValue::Array(Array::String(items)) => {
                Some(items.iter().map(|item| item.as_str().to_owned()).collect())
            }
            _ => None,
        })
}

/// The integer value of one attribute.
fn int_attribute(data: &SpanData, key: &str) -> Option<i64> {
    data.attributes
        .iter()
        .rev()
        .find(|pair| pair.key.as_str() == key)
        .and_then(|pair| match pair.value {
            OtelValue::I64(value) => Some(value),
            _ => None,
        })
}

/// The names of the events a span carries.
fn event_names(data: &SpanData) -> Vec<String> {
    data.events
        .iter()
        .map(|event| event.name.to_string())
        .collect()
}

#[test]
fn a_sink_is_send_sync_and_debug() {
    // The session tracer shares one sink across tasks, so the bound is part of
    // the public contract.
    fn assert_traits<T: Send + Sync + std::fmt::Debug + 'static>() {}
    assert_traits::<OtelTraceSink>();
    assert_traits::<Arc<OtelTraceSink>>();
    let _ = format!(
        "{:?}",
        OtelTraceSink::new(
            &init_otel_tracer(plain_config()).expect("the memory exporter initializes"),
        )
        .expect("a live handle builds a sink")
    );
}

#[test]
fn a_real_tracer_emits_frames_the_sink_turns_into_spans() {
    // test_handler.py: test_llm_start_creates_span / test_llm_end_ends_span,
    // driven through the session Tracer instead of a direct handler call.
    let fixture = Fixture::new();
    let sink: Arc<dyn TraceSink> = fixture.sink.clone();
    let tracer = SessionTracer::with_sink(SESSION, TRACE, sink);
    let span = tracer.start_span("Model", InvokeType::Llm);
    span.set_inputs(json!({"messages": [{"role": "user", "content": "hi"}]}));
    // The frame model opens a span when the start frame arrives; the tracer
    // emits the terminal frame itself.
    fixture.sink.emit(TraceFrame::agent(span.snapshot()));
    span.finish(json!({"outputs": {"content": "hello"}}));
    let data = fixture.span("chat Model");
    assert_eq!(data.status, Status::Ok);
    assert_eq!(
        attribute(&data, OJ_STATUS).as_deref(),
        Some(SPAN_STATUS_FINISH)
    );
    assert_eq!(attribute(&data, OJ_TRACE_ID).as_deref(), Some(TRACE));
}

#[test]
fn llm_start_records_the_gen_ai_request() {
    // test_handler.py: test_llm_start_creates_span
    let fixture = Fixture::new();
    let mut span = agent_start("inv-llm", "Model", InvokeType::Llm);
    span.meta_data
        .insert("class_name".to_owned(), json!("gpt-4o"));
    span.inputs = Some(json!({"messages": [
        {"role": "system", "content": "be helpful"},
        {"role": "user", "content": "hi"},
    ]}));
    fixture.emit(span.clone());
    fixture.emit(terminal(span, SpanStatus::Finish, None));
    let data = fixture.span("chat Model");
    assert_eq!(data.span_kind, SpanKind::Client);
    assert_eq!(
        attribute(&data, GEN_AI_OPERATION_NAME).as_deref(),
        Some(GEN_AI_OPERATION_CHAT)
    );
    assert_eq!(
        attribute(&data, GEN_AI_REQUEST_MODEL).as_deref(),
        Some("gpt-4o")
    );
    assert_eq!(
        attribute(&data, GEN_AI_PROVIDER_NAME).as_deref(),
        Some(GEN_AI_PROVIDER_OPENJIUWEN)
    );
    assert!(attribute(&data, GEN_AI_SYSTEM_INSTRUCTIONS)
        .is_some_and(|text| text.contains("be helpful")));
    assert!(attribute(&data, GEN_AI_INPUT_MESSAGES).is_some_and(|text| text.contains("\"hi\"")));
    assert_eq!(
        attribute(&data, OJ_AGENT_INVOKE_TYPE).as_deref(),
        Some("llm")
    );
    assert_eq!(attribute(&data, OJ_AGENT_NAME).as_deref(), Some("Model"));
}

#[test]
fn llm_finish_records_the_response_and_output_messages() {
    // test_handler.py: test_llm_end_records_response_attributes
    let fixture = Fixture::new();
    let span = agent_start("inv-llm", "Model", InvokeType::Llm);
    fixture.emit(span.clone());
    fixture.emit(terminal(
        span,
        SpanStatus::Finish,
        Some(json!({
            "outputs": {
                "content": "hello",
                "finish_reason": "stop",
                "usage_metadata": {
                    "model_name": "gpt-4o",
                    "input_tokens": 7,
                    "output_tokens": 2,
                },
            },
        })),
    ));
    let data = fixture.span("chat Model");
    assert_eq!(data.status, Status::Ok);
    let messages = attribute(&data, GEN_AI_OUTPUT_MESSAGES).expect("output messages");
    assert!(messages.contains("\"role\":\"assistant\""), "{messages}");
    assert!(messages.contains("hello"), "{messages}");
    assert_eq!(
        array_attribute(&data, GEN_AI_RESPONSE_FINISH_REASONS),
        Some(vec!["stop".to_owned()])
    );
    assert_eq!(int_attribute(&data, GEN_AI_USAGE_INPUT_TOKENS), Some(7));
    assert!(attribute(&data, OJ_AGENT_OUTPUTS).is_some());
    assert!(attribute(&data, OJ_END_TIME).is_some());
    assert!(attribute(&data, OJ_ELAPSED_TIME).is_some());
}

#[test]
fn plugin_frames_use_the_execute_tool_shape() {
    // test_handler.py: test_plugin_start_creates_execute_tool_span
    let fixture = Fixture::new();
    let mut span = agent_start("inv-tool", "search", InvokeType::Plugin);
    span.meta_data
        .insert("tool_call_id".to_owned(), json!("call-1"));
    span.meta_data
        .insert("tool_type".to_owned(), json!("function"));
    fixture.emit(span.clone());
    fixture.emit(terminal(span, SpanStatus::Finish, None));
    let data = fixture.span("execute_tool search");
    assert_eq!(data.span_kind, SpanKind::Internal);
    assert_eq!(
        attribute(&data, GEN_AI_OPERATION_NAME).as_deref(),
        Some(GEN_AI_OPERATION_EXECUTE_TOOL)
    );
    assert_eq!(
        attribute(&data, GEN_AI_TOOL_NAME).as_deref(),
        Some("search")
    );
    assert_eq!(
        attribute(&data, GEN_AI_TOOL_CALL_ID).as_deref(),
        Some("call-1")
    );
    assert_eq!(
        attribute(&data, GEN_AI_TOOL_TYPE).as_deref(),
        Some("function")
    );
}

#[test]
fn retriever_frames_record_the_query_and_documents() {
    // test_handler.py: test_retriever_start_projects_gen_ai_attributes
    let fixture = Fixture::new();
    let mut span = agent_start("r1", "", InvokeType::Retriever);
    span.inputs = Some(json!({"query": "weather", "top_k": 3, "data_source_id": "kb"}));
    fixture.emit(span.clone());
    fixture.emit(terminal(
        span,
        SpanStatus::Finish,
        Some(json!({"results": ["doc-1", "doc-2"]})),
    ));
    let data = fixture.span("retriever.r1");
    assert_eq!(
        attribute(&data, GEN_AI_OPERATION_NAME).as_deref(),
        Some(GEN_AI_OPERATION_RETRIEVAL)
    );
    assert_eq!(int_attribute(&data, GEN_AI_RETRIEVAL_TOP_K), Some(3));
    assert_eq!(
        attribute(&data, GEN_AI_DATA_SOURCE_ID).as_deref(),
        Some("kb")
    );
    assert_eq!(
        attribute(&data, GEN_AI_RETRIEVAL_QUERY_TEXT).as_deref(),
        Some("weather")
    );
    assert_eq!(
        array_attribute(&data, GEN_AI_RETRIEVAL_DOCUMENTS),
        Some(vec!["doc-1".to_owned(), "doc-2".to_owned()])
    );
}

#[test]
fn non_llm_agent_frames_use_the_dotted_names() {
    // span_names.py parity: prompt./chain./evaluator./workflow. prefixes.
    let fixture = Fixture::new();
    let cases = [
        (InvokeType::Prompt, "prompt"),
        (InvokeType::Chain, "chain"),
        (InvokeType::Evaluator, "evaluator"),
        (InvokeType::Workflow, "workflow"),
    ];
    for (index, (invoke_type, prefix)) in cases.into_iter().enumerate() {
        let invoke_id = format!("inv-{index}");
        let span = agent_start(&invoke_id, "step", invoke_type);
        fixture.emit(span.clone());
        fixture.emit(terminal(span, SpanStatus::Finish, None));
        let data = fixture.span(&format!("{prefix}.step"));
        assert_eq!(data.span_kind, SpanKind::Internal);
        assert_eq!(
            attribute(&data, OJ_AGENT_INVOKE_TYPE).as_deref(),
            Some(invoke_type.as_str())
        );
    }
}

#[test]
fn agent_failure_marks_the_span_error() {
    // test_handler.py: test_llm_error_marks_span_error
    let fixture = Fixture::new();
    let mut span = agent_start("inv-err", "Model", InvokeType::Llm);
    fixture.emit(span.clone());
    span.status = SpanStatus::Error;
    span.error = Some("boom".to_owned());
    span.end_time = Some(1_250);
    fixture.emit(span);
    let data = fixture.span("chat Model");
    assert!(matches!(data.status, Status::Error { .. }));
    assert_eq!(
        attribute(&data, OJ_STATUS).as_deref(),
        Some(SPAN_STATUS_ERROR)
    );
    assert_eq!(
        attribute(&data, OJ_AGENT_ERROR_MESSAGE).as_deref(),
        Some("boom")
    );
    assert_eq!(attribute(&data, ERROR_TYPE).as_deref(), Some("100102"));
    let payload = attribute(&data, OJ_ERROR).expect("the error payload");
    assert!(payload.contains("\"error_code\":\"100102\""), "{payload}");
    assert!(event_names(&data).contains(&"exception".to_owned()));
}

#[test]
fn an_interrupt_is_not_an_error() {
    // test_handler.py: the GraphInterrupt branch keeps the span successful.
    let fixture = Fixture::new();
    let mut span = agent_start("inv-int", "Model", InvokeType::Llm);
    fixture.emit(span.clone());
    span.status = SpanStatus::Interrupted;
    span.error = Some("awaiting input".to_owned());
    fixture.emit(span);
    let data = fixture.span("chat Model");
    assert_eq!(
        attribute(&data, OJ_STATUS).as_deref(),
        Some(SPAN_STATUS_INTERRUPTED)
    );
    assert!(attribute(&data, OJ_ERROR).is_none());
    assert_eq!(data.status, Status::Unset);
}

#[test]
fn running_frames_update_the_status_and_inputs() {
    // test_handler.py: test_refresh_inputs_attribute (agent side).
    let fixture = Fixture::new();
    let mut span = agent_start("inv-run", "Model", InvokeType::Llm);
    span.inputs = Some(json!({"messages": [{"role": "user", "content": "first"}]}));
    fixture.emit(span.clone());
    let mut running = span.clone();
    running.status = SpanStatus::Running;
    running.inputs = Some(json!({"messages": [{"role": "user", "content": "second"}]}));
    fixture.emit(running);
    fixture.emit(terminal(span, SpanStatus::Finish, None));
    let data = fixture.span("chat Model");
    let statuses = attribute_values(&data, OJ_STATUS);
    assert!(
        statuses.contains(&SPAN_STATUS_RUNNING.to_owned()),
        "{statuses:?}"
    );
    assert_eq!(
        statuses.last().map(String::as_str),
        Some(SPAN_STATUS_FINISH)
    );
    let inputs = attribute_values(&data, OJ_AGENT_INPUTS);
    assert_eq!(inputs.len(), 2, "{inputs:?}");
    assert!(inputs.last().is_some_and(|text| text.contains("second")));
}

#[test]
fn a_child_agent_span_parents_onto_its_open_parent() {
    // test_span_manager.py: test_parent_context_links_the_child_to_the_open_parent
    let fixture = Fixture::new();
    let parent = agent_start("parent", "Parent", InvokeType::Llm);
    fixture.emit(parent.clone());
    let mut child = agent_start("child", "search", InvokeType::Plugin);
    child.parent_invoke_id = Some("parent".to_owned());
    fixture.emit(child.clone());
    fixture.emit(terminal(child, SpanStatus::Finish, None));
    fixture.emit(terminal(parent, SpanStatus::Finish, None));
    let parent_span = fixture.span("chat Parent");
    let child_span = fixture.span("execute_tool search");
    assert_eq!(
        child_span.parent_span_id,
        parent_span.span_context.span_id()
    );
    assert_eq!(
        child_span.span_context.trace_id(),
        parent_span.span_context.trace_id()
    );
}

#[test]
fn an_agent_frame_parents_onto_an_open_workflow_component() {
    // Cross-dimension parent: the LLM call of a workflow node.
    let fixture = Fixture::new();
    let component = typed_component("wf-comp", "KnowledgeRetrievalExecutable", "kb");
    fixture.emit(component.clone());
    let mut child = agent_start("llm-in-wf", "Model", InvokeType::Llm);
    child.parent_invoke_id = Some("wf-comp".to_owned());
    fixture.emit(child.clone());
    fixture.emit(terminal(child, SpanStatus::Finish, None));
    fixture.emit(terminal(component, SpanStatus::Finish, None));
    let component_span = fixture.span("retrieval kb");
    let child_span = fixture.span("chat Model");
    assert_eq!(
        child_span.parent_span_id,
        component_span.span_context.span_id()
    );
}

#[test]
fn a_workflow_frame_parents_onto_an_open_agent_span() {
    // The other cross-dimension direction: a workflow started inside an agent.
    let fixture = Fixture::new();
    let agent = agent_start("agent-1", "Agent", InvokeType::Prompt);
    fixture.emit(agent.clone());
    let mut root = base_span("wf-root", "wf", TraceSpanKind::Workflow);
    root.parent_invoke_id = Some("agent-1".to_owned());
    root.workflow_id = Some("wf1".to_owned());
    fixture.emit(root.clone());
    fixture.emit(terminal(root, SpanStatus::Finish, None));
    fixture.emit(terminal(agent, SpanStatus::Finish, None));
    let agent_span = fixture.span("prompt.Agent");
    let root_span = fixture.span("wf-root");
    assert_eq!(root_span.parent_span_id, agent_span.span_context.span_id());
}

#[test]
fn workflow_roots_are_named_after_the_invoke_id() {
    // test_handler.py: test_workflow_call_start_creates_span
    let fixture = Fixture::new();
    let mut root = base_span("wf-root", "ignored", TraceSpanKind::Workflow);
    root.workflow_id = Some("wf1".to_owned());
    root.meta_data
        .insert("workflow_name".to_owned(), json!("MyWorkflow"));
    root.meta_data
        .insert("workflow_version".to_owned(), json!("1.0"));
    root.meta_data
        .insert("workflow_id".to_owned(), json!("wf1"));
    root.meta_data
        .insert("source_ids".to_owned(), json!(["src-1"]));
    root.inputs = Some(json!({"input": "test"}));
    fixture.emit(root.clone());
    fixture.emit(terminal(
        root,
        SpanStatus::Finish,
        Some(json!({"output": "done"})),
    ));
    let data = fixture.span("wf-root");
    assert_eq!(data.span_kind, SpanKind::Internal);
    assert_eq!(data.parent_span_id, SpanId::INVALID);
    assert_eq!(attribute(&data, OJ_WORKFLOW_ID).as_deref(), Some("wf1"));
    assert_eq!(
        attribute(&data, OJ_WORKFLOW_NAME).as_deref(),
        Some("MyWorkflow")
    );
    assert_eq!(
        attribute(&data, OJ_WORKFLOW_VERSION).as_deref(),
        Some("1.0")
    );
    assert_eq!(
        attribute(&data, OJ_WORKFLOW_EXECUTION_ID).as_deref(),
        Some("wf1")
    );
    assert_eq!(
        attribute(&data, OJ_SOURCE_IDS).as_deref(),
        Some("[\"src-1\"]")
    );
    assert_eq!(attribute(&data, OJ_PARENT_NODE_ID).as_deref(), Some(""));
    assert_eq!(
        attribute(&data, OJ_WORKFLOW_INPUTS).as_deref(),
        Some("{\"input\":\"test\"}")
    );
    assert_eq!(
        attribute(&data, OJ_WORKFLOW_OUTPUTS).as_deref(),
        Some("{\"output\":\"done\"}")
    );
    assert!(attribute(&data, OJ_WORKFLOW_COMPONENT_ID).is_none());
}

#[test]
fn workflow_component_naming_covers_every_family() {
    // test_handler.py: test_llm_component_is_a_client_chat_span,
    // test_tool_component_uses_the_execute_tool_shape, and the component
    // fallback of on_call_start.
    let fixture = Fixture::new();
    let cases = [
        ("comp-llm", "LLMComponent", "m", "chat", SpanKind::Client),
        (
            "comp-tool",
            "ToolExecutable",
            "search",
            "execute_tool search",
            SpanKind::Internal,
        ),
        (
            "comp-kb",
            "KnowledgeRetrievalExecutable",
            "kb",
            "retrieval kb",
            SpanKind::Internal,
        ),
        (
            "comp-plain",
            "StartExecutable",
            "s",
            "component.comp-plain",
            SpanKind::Internal,
        ),
    ];
    for (invoke_id, component_type, component_name, expected_name, expected_kind) in cases {
        let span = typed_component(invoke_id, component_type, component_name);
        fixture.emit(span.clone());
        fixture.emit(terminal(span, SpanStatus::Finish, None));
        let data = fixture.span(expected_name);
        assert_eq!(data.span_kind, expected_kind);
        assert_eq!(
            attribute(&data, OJ_WORKFLOW_COMPONENT_ID).as_deref(),
            Some(invoke_id)
        );
        assert_eq!(
            attribute(&data, OJ_WORKFLOW_COMPONENT_TYPE).as_deref(),
            Some(component_type)
        );
        assert_eq!(
            attribute(&data, OJ_WORKFLOW_COMPONENT_NAME).as_deref(),
            Some(component_name)
        );
        if component_type == "LLMComponent" {
            assert_eq!(
                attribute(&data, GEN_AI_OPERATION_NAME).as_deref(),
                Some(GEN_AI_OPERATION_CHAT)
            );
        }
        if component_type == "ToolExecutable" {
            assert_eq!(
                attribute(&data, GEN_AI_TOOL_NAME).as_deref(),
                Some("search")
            );
        }
    }
}

#[test]
fn an_llm_component_records_its_request_model() {
    // test_handler.py: test_llm_component_is_a_client_chat_span
    let fixture = Fixture::new();
    let mut span = typed_component("comp-llm", "LLMComponent", "m");
    span.meta_data
        .insert("model_name".to_owned(), json!("gpt-4o"));
    fixture.emit(span.clone());
    fixture.emit(terminal(span, SpanStatus::Finish, None));
    let data = fixture.span("chat gpt-4o");
    assert_eq!(
        attribute(&data, GEN_AI_REQUEST_MODEL).as_deref(),
        Some("gpt-4o")
    );
}

#[test]
fn a_retrieval_component_records_its_static_facts() {
    // test_handler.py: test_retrieval_component_projects_the_gen_ai_attributes
    let fixture = Fixture::new();
    let mut span = typed_component("comp-kb", "KnowledgeRetrievalExecutable", "kb");
    span.meta_data.insert("top_k".to_owned(), json!(5));
    span.meta_data
        .insert("data_source_ids".to_owned(), json!(["kb", "web"]));
    span.meta_data
        .insert("request_model".to_owned(), json!("embed-v3"));
    span.inputs = Some(json!({"query": "weather"}));
    fixture.emit(span.clone());
    fixture.emit(terminal(
        span,
        SpanStatus::Finish,
        Some(json!({"results": ["doc-1"]})),
    ));
    let data = fixture.span("retrieval kb");
    assert_eq!(int_attribute(&data, GEN_AI_RETRIEVAL_TOP_K), Some(5));
    assert_eq!(
        array_attribute(&data, GEN_AI_DATA_SOURCE_ID),
        Some(vec!["kb".to_owned(), "web".to_owned()])
    );
    assert_eq!(
        attribute(&data, GEN_AI_REQUEST_MODEL).as_deref(),
        Some("embed-v3")
    );
    assert_eq!(
        attribute(&data, GEN_AI_RETRIEVAL_QUERY_TEXT).as_deref(),
        Some("weather")
    );
    assert_eq!(
        array_attribute(&data, GEN_AI_RETRIEVAL_DOCUMENTS),
        Some(vec!["doc-1".to_owned()])
    );
}

#[test]
fn workflow_buffers_flush_as_json_on_the_terminal_frame() {
    // test_handler.py: test_buffers_are_flushed_as_json_on_call_done, fed from
    // meta_data instead of the on_invoke/on_pre_stream/on_post_stream callbacks.
    let fixture = Fixture::new();
    let span = component("comp-1");
    fixture.emit(span.clone());
    let mut running = span.clone();
    running.status = SpanStatus::Running;
    running
        .meta_data
        .insert("on_invoke_data".to_owned(), json!({"step": 1}));
    running
        .meta_data
        .insert("stream_inputs".to_owned(), json!([{"chunk": 2}]));
    running
        .meta_data
        .insert("stream_outputs".to_owned(), json!({"chunk": 3}));
    fixture.emit(running);
    fixture.emit(terminal(span, SpanStatus::Finish, None));
    let data = fixture.span("component.comp-1");
    assert_eq!(
        attribute(&data, OJ_WORKFLOW_INVOKE_DATA).as_deref(),
        Some("[{\"step\":1}]")
    );
    assert_eq!(
        attribute(&data, OJ_STREAM_INPUTS).as_deref(),
        Some("[{\"chunk\":2}]")
    );
    assert_eq!(
        attribute(&data, OJ_STREAM_OUTPUTS).as_deref(),
        Some("[{\"chunk\":3}]")
    );
}

#[test]
fn empty_workflow_buffers_write_no_attribute() {
    // Python's _flush_buffered_data only writes a non-empty buffer.
    let fixture = Fixture::new();
    let span = component("comp-1");
    fixture.emit(span.clone());
    fixture.emit(terminal(span, SpanStatus::Finish, None));
    let data = fixture.span("component.comp-1");
    assert!(attribute(&data, OJ_WORKFLOW_INVOKE_DATA).is_none());
    assert!(attribute(&data, OJ_STREAM_INPUTS).is_none());
    assert!(attribute(&data, OJ_STREAM_OUTPUTS).is_none());
}

#[test]
fn a_running_workflow_frame_refreshes_the_inputs() {
    // test_handler.py: TestRefreshInputsAttribute, workflow side.
    let fixture = Fixture::new();
    let mut span = component("comp-1");
    span.inputs = Some(json!({"a": 1}));
    fixture.emit(span.clone());
    let mut running = span.clone();
    running.status = SpanStatus::Running;
    running.inputs = Some(json!({"a": 2}));
    fixture.emit(running);
    fixture.emit(terminal(span, SpanStatus::Finish, None));
    let data = fixture.span("component.comp-1");
    assert_eq!(
        attribute_values(&data, OJ_WORKFLOW_INPUTS),
        vec!["{\"a\":1}".to_owned(), "{\"a\":2}".to_owned()]
    );
    let statuses = attribute_values(&data, OJ_STATUS);
    assert!(
        statuses.contains(&SPAN_STATUS_RUNNING.to_owned()),
        "{statuses:?}"
    );
    assert_eq!(
        statuses.last().map(String::as_str),
        Some(SPAN_STATUS_FINISH)
    );
}

#[test]
fn a_workflow_failure_marks_the_span_error() {
    // test_handler.py: test_workflow_on_invoke_error_marks_span_error
    let fixture = Fixture::new();
    let mut span = component("comp-1");
    fixture.emit(span.clone());
    span.status = SpanStatus::Error;
    span.error = Some("node failed".to_owned());
    fixture.emit(span);
    let data = fixture.span("component.comp-1");
    assert!(matches!(data.status, Status::Error { .. }));
    assert_eq!(
        attribute(&data, OJ_STATUS).as_deref(),
        Some(SPAN_STATUS_ERROR)
    );
    assert_eq!(
        attribute(&data, OJ_WORKFLOW_ERROR_MESSAGE).as_deref(),
        Some("node failed")
    );
    assert!(attribute(&data, OJ_ERROR).is_some());
    assert!(event_names(&data).contains(&"exception".to_owned()));
}

#[test]
fn a_workflow_interrupt_keeps_the_span_successful() {
    // test_handler.py: test_interrupt_keeps_the_span_successful
    let fixture = Fixture::new();
    let mut span = component("comp-1");
    fixture.emit(span.clone());
    span.status = SpanStatus::Interrupted;
    span.error = Some("awaiting input".to_owned());
    fixture.emit(span);
    let data = fixture.span("component.comp-1");
    assert_eq!(
        attribute(&data, OJ_STATUS).as_deref(),
        Some(SPAN_STATUS_INTERRUPTED)
    );
    assert_eq!(
        attribute(&data, OJ_WORKFLOW_ERROR_MESSAGE).as_deref(),
        Some("awaiting input")
    );
    assert!(attribute(&data, OJ_ERROR).is_none());
    assert_eq!(data.status, Status::Unset);
}

#[test]
#[allow(clippy::assert_is_empty)] // the message documents which invariant failed
fn a_terminal_frame_for_an_unknown_invoke_is_dropped() {
    // Fail-open: an out-of-order frame must not panic nor create a span.
    let fixture = Fixture::new();
    for status in [
        SpanStatus::Finish,
        SpanStatus::Error,
        SpanStatus::Interrupted,
        SpanStatus::Running,
    ] {
        let mut agent = agent_start("missing", "Model", InvokeType::Llm);
        agent.status = status;
        fixture.emit(agent);

        let mut workflow = base_span("missing-wf", "wf", TraceSpanKind::Workflow);
        workflow.status = status;
        fixture.emit(workflow);
    }
    assert!(fixture.spans().is_empty());
}

#[test]
fn is_recording_follows_the_handle_state() {
    let handle = init_otel_tracer(plain_config()).expect("the memory exporter initializes");
    let live = OtelTraceSink::new(&handle).expect("a live handle builds a sink");
    assert!(live.is_recording());

    handle.shutdown();
    let after = OtelTraceSink::new(&handle).expect("a shut-down handle still builds a sink");
    assert!(!after.is_recording());
    // A non-recording sink swallows frames instead of opening spans.
    after.emit(frame_of(agent_start("inv", "Model", InvokeType::Llm)));
}

#[test]
fn with_session_records_through_its_private_provider() {
    let sink = OtelTraceSink::with_session("session-9").expect("the memory provider initializes");
    assert!(sink.is_recording());
    sink.emit(frame_of(agent_start("inv", "Model", InvokeType::Llm)));
    let mut finish = agent_start("inv", "Model", InvokeType::Llm);
    finish.status = SpanStatus::Finish;
    sink.emit(frame_of(finish));
}

#[test]
fn prompts_are_hashed_when_redaction_is_enabled() {
    // test_redaction.py: a prompt is replaced by its sha256 digest.
    let fixture =
        Fixture::build(OtelTracerConfig::default().with_exporter_type(ExporterType::Memory));
    let mut span = agent_start("inv", "Model", InvokeType::Llm);
    span.inputs = Some(json!({"messages": [{"role": "user", "content": "secret"}]}));
    fixture.emit(span.clone());
    fixture.emit(terminal(span, SpanStatus::Finish, None));
    let data = fixture.span("chat Model");
    let inputs = attribute(&data, OJ_AGENT_INPUTS).expect("redacted inputs");
    assert!(inputs.starts_with("sha256:"), "{inputs}");
    assert!(!inputs.contains("secret"), "{inputs}");
    let messages = attribute(&data, GEN_AI_INPUT_MESSAGES).expect("redacted messages");
    assert!(!messages.contains("secret"), "{messages}");
}
