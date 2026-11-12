//! Unit tests of the workflow handler.
//!
//! Port of `tests/unit_tests/extensions/tracer_otel/test_handler.py`
//! (`TestOtelWorkflowHandler`, `TestRefreshInputsAttribute`,
//! `TestWorkflowIsolation`, plus the retrieval/document assertions), adapted to
//! the Rust frame API: `parent_node_id`/`session_id`/`source_ids` travel inside
//! `metadata`, and Python's exception branch of `on_invoke` is the pair
//! `on_call_error` / `on_call_interrupted`.

// A test module owns its panics: a failed `unwrap`/`expect` is an assertion.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;

use opentelemetry::trace::{SpanId, SpanKind, Status};
use opentelemetry::{Array, Value as OtelValue};
use opentelemetry_sdk::trace::SpanData;
use serde_json::json;

use openjiuwen_common::error::{Error, ErrorKind};

use super::*;
use crate::otel::config::ExporterType;
use crate::otel::handle::init_otel_tracer;
use crate::otel::provider::CaptureExporter;
use crate::otel::semconv::{
    ERROR_TYPE, GEN_AI_RETRIEVAL_DOCUMENTS, OJ_ELAPSED_TIME, OJ_END_TIME, OJ_ERROR,
    OJ_WORKFLOW_COMPONENT_ID, OJ_WORKFLOW_COMPONENT_NAME, OJ_WORKFLOW_COMPONENT_TYPE,
    OJ_WORKFLOW_EXECUTION_ID, OJ_WORKFLOW_ID, OJ_WORKFLOW_LOOP_INDEX, OJ_WORKFLOW_LOOP_NODE_ID,
    OJ_WORKFLOW_NAME, OJ_WORKFLOW_VERSION, SPAN_STATUS_ERROR,
};

/// Configuration with readable (unredacted) attributes.
fn plain_config() -> OtelTracerConfig {
    OtelTracerConfig {
        redaction_enabled: false,
        ..OtelTracerConfig::default()
    }
    .with_exporter_type(ExporterType::Memory)
}

/// A configuration that hashes every prompt/completion payload.
fn redacting_config() -> OtelTracerConfig {
    OtelTracerConfig::default().with_exporter_type(ExporterType::Memory)
}

/// A handler plus the private provider capturing its spans.
struct Fixture {
    /// Keeps the provider (and therefore the batch processor) alive.
    handle: TracerHandle,
    /// Span buffer shared with the provider's exporter.
    capture: Arc<CaptureExporter>,
    /// Handler under test.
    handler: OtelWorkflowHandler,
}

impl Fixture {
    /// Builds a fixture over `config`, with an optional trace id.
    fn build(config: OtelTracerConfig, trace_id: Option<&str>) -> Self {
        let handle = init_otel_tracer(config.clone()).expect("the memory exporter initializes");
        let capture = handle
            .capture()
            .expect("the memory exporter captures spans");
        let tracer = handle.tracer().expect("the memory handle is active");
        let handler = OtelWorkflowHandler::new(tracer, config, trace_id.map(str::to_owned));
        Self {
            handle,
            capture,
            handler,
        }
    }

    /// A fixture with unredacted attributes and trace id `trace-1`.
    fn new() -> Self {
        Self::build(plain_config(), Some("trace-1"))
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
}

/// The string value of one attribute.
///
/// The SDK appends on re-recording, so the *last* occurrence wins: it is the
/// live value, exactly like Python's `set_attribute`.
fn attribute(data: &SpanData, key: &str) -> Option<String> {
    data.attributes
        .iter()
        .rev()
        .find(|pair| pair.key.as_str() == key)
        .map(|pair| pair.value.as_str().into_owned())
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

/// The metadata of a retrieval component span.
fn retrieval_metadata() -> Value {
    json!({
        "component_id": "kb_node",
        "component_type": "KnowledgeRetrievalExecutable",
        "component_name": "kb_node",
        "workflow_id": "wf1",
        "parent_node_id": "",
    })
}

#[tokio::test]
async fn root_span_records_the_python_attributes() {
    // test_handler.py: test_workflow_call_start_creates_span
    let fixture = Fixture::new();
    fixture
        .handler
        .on_call_start(
            "wf_root",
            &json!({
                "workflow_id": "wf1",
                "workflow_name": "MyWorkflow",
                "workflow_version": "1.0",
                "session_id": "workflow-session-123",
                "source_ids": ["src-1", "src-2"],
            }),
            Some(&json!({"input": "test"})),
        )
        .await;
    assert!(
        fixture.spans().is_empty(),
        "the span only closes on on_call_done"
    );
    fixture
        .handler
        .on_call_done("wf_root", Some(&json!({"output": "done"})))
        .await;
    let span = fixture.span("wf_root");
    assert_eq!(span.span_kind, SpanKind::Internal);
    assert_eq!(span.parent_span_id, SpanId::INVALID);
    assert_eq!(attribute(&span, OJ_WORKFLOW_ID).as_deref(), Some("wf1"));
    assert_eq!(
        attribute(&span, OJ_WORKFLOW_NAME).as_deref(),
        Some("MyWorkflow")
    );
    assert_eq!(
        attribute(&span, OJ_WORKFLOW_VERSION).as_deref(),
        Some("1.0")
    );
    assert_eq!(
        attribute(&span, OJ_WORKFLOW_EXECUTION_ID).as_deref(),
        Some("wf1"),
        "the execution id defaults to the workflow id"
    );
    assert_eq!(attribute(&span, OJ_TRACE_ID).as_deref(), Some("trace-1"));
    assert_eq!(
        attribute(&span, OJ_SESSION_ID).as_deref(),
        Some("workflow-session-123")
    );
    assert_eq!(attribute(&span, OJ_INVOKE_ID).as_deref(), Some("wf_root"));
    assert_eq!(
        attribute(&span, OJ_PARENT_NODE_ID).as_deref(),
        Some(""),
        "the parent node id is written even when empty"
    );
    assert_eq!(
        attribute(&span, OJ_SOURCE_IDS).as_deref(),
        Some(r#"["src-1","src-2"]"#)
    );
    assert_eq!(
        attribute(&span, OJ_WORKFLOW_INPUTS).as_deref(),
        Some(r#"{"input":"test"}"#)
    );
    assert_eq!(
        attribute(&span, OJ_WORKFLOW_OUTPUTS).as_deref(),
        Some(r#"{"output":"done"}"#)
    );
    assert_eq!(
        attribute(&span, OJ_STATUS).as_deref(),
        Some(SPAN_STATUS_FINISH)
    );
    assert_eq!(attribute(&span, OJ_START_TIME).map(|s| s.len()), Some(26));
    assert_eq!(attribute(&span, OJ_END_TIME).map(|s| s.len()), Some(26));
    assert!(attribute(&span, OJ_ELAPSED_TIME).is_some());
    assert_eq!(span.status, Status::Ok);
}

#[tokio::test]
async fn session_id_falls_back_to_the_injected_value() {
    // test_handler.py: test_workflow_span_omits_session_id_when_absent,
    // test_workflow_span_falls_back_to_injected_session_id
    let fixture = Fixture::new();
    fixture
        .handler
        .on_call_start("wf_root", &json!({"workflow_id": "wf1"}), None)
        .await;
    fixture.handler.on_call_done("wf_root", None).await;
    assert!(attribute(&fixture.span("wf_root"), OJ_SESSION_ID).is_none());

    let mut fixture = Fixture::new();
    fixture.handler.set_session_id("injected-session");
    fixture
        .handler
        .on_call_start("wf_root", &json!({"workflow_id": "wf1"}), None)
        .await;
    fixture.handler.on_call_done("wf_root", None).await;
    assert_eq!(
        attribute(&fixture.span("wf_root"), OJ_SESSION_ID).as_deref(),
        Some("injected-session")
    );
}

#[tokio::test]
async fn component_without_a_family_keeps_the_generic_name() {
    // test_handler.py: test_workflow_non_retrieval_component_keeps_generic_name,
    // test_workflow_non_llm_internal_span
    let fixture = Fixture::new();
    fixture
        .handler
        .on_call_start(
            "other_node",
            &json!({
                "component_id": "other_node",
                "component_type": "SomeExecutable",
                "component_name": "other_node",
                "workflow_id": "wf1",
                "parent_node_id": "",
            }),
            Some(&json!({"query": "不是检索组件"})),
        )
        .await;
    fixture
        .handler
        .on_call_done("other_node", Some(&json!({"results": ["x"]})))
        .await;
    let span = fixture.span("component.other_node");
    assert_eq!(span.span_kind, SpanKind::Internal);
    assert!(attribute(&span, GEN_AI_OPERATION_NAME).is_none());
    assert!(attribute(&span, GEN_AI_RETRIEVAL_QUERY_TEXT).is_none());
    assert!(attribute(&span, GEN_AI_RETRIEVAL_DOCUMENTS).is_none());
    assert_eq!(
        attribute(&span, OJ_WORKFLOW_COMPONENT_ID).as_deref(),
        Some("other_node")
    );
    assert_eq!(
        attribute(&span, OJ_WORKFLOW_COMPONENT_TYPE).as_deref(),
        Some("SomeExecutable")
    );
}

#[tokio::test]
async fn llm_component_is_a_client_chat_span() {
    // test_handler.py: test_workflow_component_type_mapping,
    // test_workflow_llm_compound_components_tagged_as_chat
    let fixture = Fixture::new();
    fixture
        .handler
        .on_call_start(
            "wf_root",
            &json!({"workflow_id": "wf1", "workflow_name": "WF"}),
            None,
        )
        .await;
    let components = [
        ("LLMComponent", "llm_node", Some("gpt-4o")),
        ("IntentDetectionComponent", "intent_node", None),
        ("QuestionerComponent", "questioner_node", None),
        ("LLM", "llm_substring_node", None),
    ];
    for (component_type, invoke_id, model) in components {
        let mut metadata = json!({
            "component_id": invoke_id,
            "component_type": component_type,
            "component_name": invoke_id,
            "workflow_id": "wf1",
            "parent_node_id": "",
        });
        if let Some(model) = model {
            metadata["model_name"] = json!(model);
        }
        fixture
            .handler
            .on_call_start(invoke_id, &metadata, None)
            .await;
        fixture.handler.on_call_done(invoke_id, None).await;
    }
    fixture.handler.on_call_done("wf_root", None).await;

    let spans = fixture.spans();
    assert_eq!(spans.len(), 5);
    let modelled = spans
        .iter()
        .find(|span| span.name == "chat gpt-4o")
        .expect("the model names the span");
    assert_eq!(modelled.span_kind, SpanKind::Client);
    assert_eq!(
        attribute(modelled, GEN_AI_OPERATION_NAME).as_deref(),
        Some(GEN_AI_OPERATION_CHAT)
    );
    assert_eq!(
        attribute(modelled, GEN_AI_REQUEST_MODEL).as_deref(),
        Some("gpt-4o")
    );
    let unnamed: Vec<&SpanData> = spans.iter().filter(|span| span.name == "chat").collect();
    assert_eq!(unnamed.len(), 3, "a model-less chat span drops the suffix");
    for span in unnamed {
        assert_eq!(span.span_kind, SpanKind::Client);
        assert_eq!(
            attribute(span, GEN_AI_OPERATION_NAME).as_deref(),
            Some(GEN_AI_OPERATION_CHAT)
        );
        assert!(attribute(span, GEN_AI_REQUEST_MODEL).is_none());
    }
}

#[tokio::test]
async fn tool_component_uses_the_execute_tool_shape() {
    // test_handler.py: test_workflow_tool_component_tagged_as_execute_tool,
    // test_workflow_attributes_set
    let fixture = Fixture::new();
    fixture
        .handler
        .on_call_start(
            "tool_node",
            &json!({
                "component_id": "tool_node",
                "component_type": "ToolExecutable",
                "component_name": "tool_node",
                "workflow_id": "wf1",
            }),
            None,
        )
        .await;
    fixture.handler.on_call_done("tool_node", None).await;
    let span = fixture.span("execute_tool tool_node");
    assert_eq!(span.span_kind, SpanKind::Internal);
    assert_eq!(
        attribute(&span, GEN_AI_OPERATION_NAME).as_deref(),
        Some(GEN_AI_OPERATION_EXECUTE_TOOL)
    );
    assert_eq!(
        attribute(&span, GEN_AI_TOOL_NAME).as_deref(),
        Some("tool_node")
    );
    assert_eq!(
        attribute(&span, OJ_WORKFLOW_COMPONENT_TYPE).as_deref(),
        Some("ToolExecutable")
    );
}

#[tokio::test]
async fn retrieval_component_projects_the_gen_ai_attributes() {
    // test_handler.py: test_workflow_retrieval_component_tagged_as_retrieval
    let fixture = Fixture::new();
    let mut metadata = retrieval_metadata();
    metadata["top_k"] = json!(5);
    metadata["data_source_ids"] = json!(["kb-1", "kb-2"]);
    metadata["request_model"] = json!("embedding-3");
    fixture
        .handler
        .on_call_start(
            "kb_node",
            &metadata,
            Some(&json!({"query": "什么是向量检索"})),
        )
        .await;
    fixture
        .handler
        .on_call_done(
            "kb_node",
            Some(&json!({"results": ["doc-1 内容", "doc-2 内容"]})),
        )
        .await;
    let span = fixture.span("retrieval kb_node");
    assert_eq!(span.span_kind, SpanKind::Internal);
    assert_eq!(
        attribute(&span, GEN_AI_OPERATION_NAME).as_deref(),
        Some(GEN_AI_OPERATION_RETRIEVAL)
    );
    assert_eq!(
        attribute(&span, GEN_AI_RETRIEVAL_QUERY_TEXT).as_deref(),
        Some("什么是向量检索")
    );
    assert_eq!(int_attribute(&span, GEN_AI_RETRIEVAL_TOP_K), Some(5));
    assert_eq!(
        array_attribute(&span, GEN_AI_DATA_SOURCE_ID),
        Some(vec!["kb-1".to_owned(), "kb-2".to_owned()])
    );
    assert_eq!(
        attribute(&span, GEN_AI_REQUEST_MODEL).as_deref(),
        Some("embedding-3")
    );
    assert_eq!(
        array_attribute(&span, GEN_AI_RETRIEVAL_DOCUMENTS),
        Some(vec!["doc-1 内容".to_owned(), "doc-2 内容".to_owned()])
    );
}

#[tokio::test]
async fn retrieval_documents_and_query_are_redacted() {
    // test_handler.py: test_workflow_retrieval_documents_redacted
    let fixture = Fixture::build(redacting_config(), Some("trace-1"));
    fixture
        .handler
        .on_call_start(
            "kb_node",
            &retrieval_metadata(),
            Some(&json!({"query": "什么是向量检索"})),
        )
        .await;
    fixture
        .handler
        .on_call_done(
            "kb_node",
            Some(&json!({"results": ["doc-1 内容", "doc-2 内容"]})),
        )
        .await;
    let span = fixture.span("retrieval kb_node");
    let documents = array_attribute(&span, GEN_AI_RETRIEVAL_DOCUMENTS).expect("documents recorded");
    assert_eq!(documents.len(), 2);
    assert!(documents.iter().all(|doc| doc.starts_with("sha256:")));
    for raw in ["doc-1 内容", "doc-2 内容", "什么是向量检索"] {
        assert!(
            !documents.iter().any(|doc| doc == raw),
            "raw text must never be recorded"
        );
    }
    assert!(attribute(&span, GEN_AI_RETRIEVAL_QUERY_TEXT)
        .expect("the query is recorded")
        .starts_with("sha256:"));
    assert!(attribute(&span, OJ_WORKFLOW_OUTPUTS)
        .expect("the outputs are recorded")
        .starts_with("sha256:"));
}

#[tokio::test]
async fn retrieval_query_is_refreshed_on_pre_invoke() {
    // test_handler.py: test_workflow_retrieval_query_refreshed_on_pre_invoke
    let fixture = Fixture::new();
    fixture
        .handler
        .on_call_start(
            "kb_node",
            &retrieval_metadata(),
            Some(&json!({"query": "旧查询"})),
        )
        .await;
    fixture
        .handler
        .on_pre_invoke("kb_node", Some(&json!({"query": "新查询"})))
        .await;
    fixture.handler.on_call_done("kb_node", None).await;
    let span = fixture.span("retrieval kb_node");
    assert_eq!(
        attribute(&span, GEN_AI_RETRIEVAL_QUERY_TEXT).as_deref(),
        Some("新查询")
    );
    assert_eq!(
        attribute(&span, OJ_WORKFLOW_INPUTS).as_deref(),
        Some(r#"{"query":"新查询"}"#),
        "on_pre_invoke rewrites the inputs attribute"
    );
}

#[tokio::test]
async fn retrieval_documents_are_recorded_on_post_invoke() {
    // test_handler.py: test_workflow_post_invoke projections (documents + outputs)
    let fixture = Fixture::new();
    fixture
        .handler
        .on_call_start("kb_node", &retrieval_metadata(), None)
        .await;
    fixture
        .handler
        .on_post_invoke(
            "kb_node",
            &json!({"results": ["only-doc"], "context": "ctx"}),
        )
        .await;
    fixture.handler.on_call_done("kb_node", None).await;
    let span = fixture.span("retrieval kb_node");
    assert_eq!(
        array_attribute(&span, GEN_AI_RETRIEVAL_DOCUMENTS),
        Some(vec!["only-doc".to_owned()])
    );
    assert_eq!(
        attribute(&span, OJ_WORKFLOW_OUTPUTS).as_deref(),
        Some(r#"{"results":["only-doc"],"context":"ctx"}"#)
    );
}

#[tokio::test]
async fn sub_workflow_root_parents_onto_the_host_component() {
    // test_handler.py: test_workflow_sub_workflow_parent_child
    let fixture = Fixture::new();
    fixture
        .handler
        .on_call_start(
            "wf_root",
            &json!({"workflow_id": "wf1", "workflow_name": "RootWF"}),
            None,
        )
        .await;
    fixture
        .handler
        .on_call_start(
            "llm_node",
            &json!({
                "component_id": "llm_node",
                "component_type": "LLM",
                "component_name": "llm",
                "workflow_id": "wf1",
                "parent_node_id": "",
            }),
            None,
        )
        .await;
    fixture
        .handler
        .on_call_start(
            "sub_wf_root",
            &json!({
                "workflow_id": "wf2",
                "workflow_name": "SubWF",
                "parent_node_id": "llm_node",
            }),
            None,
        )
        .await;
    fixture.handler.on_call_done("sub_wf_root", None).await;
    fixture.handler.on_call_done("llm_node", None).await;
    fixture.handler.on_call_done("wf_root", None).await;

    assert_eq!(fixture.spans().len(), 3);
    let root = fixture.span("wf_root");
    let host = fixture.span("chat");
    let sub = fixture.span("sub_wf_root");
    assert_eq!(host.parent_span_id, root.span_context.span_id());
    assert_eq!(sub.parent_span_id, host.span_context.span_id());
    assert_eq!(sub.span_context.trace_id(), root.span_context.trace_id());
}

#[tokio::test]
async fn component_inside_a_sub_workflow_parents_onto_the_host_component() {
    // test_handler.py: test_workflow_component_inside_sub_workflow_parent
    let fixture = Fixture::new();
    fixture
        .handler
        .on_call_start(
            "wf_root",
            &json!({"workflow_id": "wf1", "workflow_name": "RootWF"}),
            None,
        )
        .await;
    fixture
        .handler
        .on_call_start(
            "sub_wf_node",
            &json!({
                "component_id": "sub_wf_node",
                "component_type": "sub_workflow",
                "component_name": "sub_wf",
                "workflow_id": "wf1",
                "parent_node_id": "",
            }),
            None,
        )
        .await;
    for (invoke_id, component_type) in [("sub_wf_node.start", "Start"), ("sub_wf_node.llm", "LLM")]
    {
        fixture
            .handler
            .on_call_start(
                invoke_id,
                &json!({
                    "component_id": invoke_id,
                    "component_type": component_type,
                    "component_name": invoke_id,
                    "workflow_id": "wf2",
                    "parent_node_id": "sub_wf_node",
                }),
                None,
            )
            .await;
        fixture.handler.on_call_done(invoke_id, None).await;
    }
    fixture.handler.on_call_done("sub_wf_node", None).await;
    fixture.handler.on_call_done("wf_root", None).await;

    let root = fixture.span("wf_root");
    let host = fixture.span("component.sub_wf_node");
    let start = fixture.span("component.sub_wf_node.start");
    let llm = fixture.span("chat");
    assert_eq!(host.parent_span_id, root.span_context.span_id());
    assert_eq!(start.parent_span_id, host.span_context.span_id());
    assert_eq!(llm.parent_span_id, host.span_context.span_id());
}

#[tokio::test]
async fn a_new_workflow_id_starts_a_new_root() {
    // test_handler.py: TestWorkflowIsolation.test_different_workflow_id_cleans_stale_context
    let fixture = Fixture::new();
    fixture
        .handler
        .on_call_start(
            "wf1_root",
            &json!({"workflow_id": "wf1", "workflow_name": "Workflow 1"}),
            None,
        )
        .await;
    let first_trace = fixture
        .handler
        .manager
        .with_state("wf1_root", |state| state.span_context().trace_id())
        .expect("the first root is still open");
    fixture
        .handler
        .on_call_start(
            "wf2_root",
            &json!({"workflow_id": "wf2", "workflow_name": "Workflow 2"}),
            None,
        )
        .await;

    {
        // The guard must not outlive this block: it precedes an await point.
        let state = lock(&fixture.handler.state);
        assert!(!state.layer_roots.contains_key("wf1_root"));
        assert!(state.components.is_empty());
    }

    fixture.handler.on_call_done("wf2_root", None).await;
    let second = fixture.span("wf2_root");
    assert_eq!(second.parent_span_id, SpanId::INVALID);
    assert_ne!(second.span_context.trace_id(), first_trace);
    assert_eq!(fixture.spans().len(), 1, "the superseded root never ends");
}

#[tokio::test]
async fn buffers_are_flushed_as_json_on_call_done() {
    // test_handler.py: test_workflow_invoke_buffers_data,
    // test_workflow_stream_buffers_flushed
    let fixture = Fixture::new();
    fixture
        .handler
        .on_call_start(
            "comp1",
            &json!({
                "component_id": "comp1",
                "component_type": "Tool",
                "workflow_id": "wf1",
            }),
            None,
        )
        .await;
    fixture
        .handler
        .on_invoke("comp1", &json!({"step": 1}))
        .await;
    fixture
        .handler
        .on_invoke("comp1", &json!({"step": 2}))
        .await;
    fixture
        .handler
        .on_pre_stream("comp1", &json!({"text": "hello"}))
        .await;
    fixture
        .handler
        .on_pre_stream("comp1", &json!({"text": "world"}))
        .await;
    // A non-object chunk is dropped, like Python's `isinstance(chunk, dict)`.
    fixture
        .handler
        .on_pre_stream("comp1", &json!("not a chunk"))
        .await;
    fixture
        .handler
        .on_post_stream("comp1", &json!({"text": "response"}))
        .await;
    fixture.handler.on_call_done("comp1", None).await;

    let span = fixture.span("execute_tool comp1");
    assert_eq!(
        attribute(&span, OJ_WORKFLOW_INVOKE_DATA).as_deref(),
        Some(r#"[{"step":1},{"step":2}]"#)
    );
    assert_eq!(
        attribute(&span, OJ_STREAM_INPUTS).as_deref(),
        Some(r#"[{"text":"hello"},{"text":"world"}]"#)
    );
    assert_eq!(
        attribute(&span, OJ_STREAM_OUTPUTS).as_deref(),
        Some(r#"[{"text":"response"}]"#)
    );
}

#[tokio::test]
async fn empty_buffers_leave_the_attributes_absent() {
    // test_handler.py: _flush_buffered_data only writes non-empty buffers
    let fixture = Fixture::new();
    fixture
        .handler
        .on_call_start(
            "comp1",
            &json!({
                "component_id": "comp1",
                "component_type": "Tool",
                "workflow_id": "wf1",
            }),
            None,
        )
        .await;
    fixture
        .handler
        .on_call_done("comp1", Some(&json!({})))
        .await;

    let span = fixture.span("execute_tool comp1");
    assert!(attribute(&span, OJ_WORKFLOW_INVOKE_DATA).is_none());
    assert!(attribute(&span, OJ_STREAM_INPUTS).is_none());
    assert!(attribute(&span, OJ_STREAM_OUTPUTS).is_none());
    assert!(attribute(&span, OJ_WORKFLOW_INPUTS).is_none());
    assert_eq!(attribute(&span, OJ_WORKFLOW_OUTPUTS).as_deref(), Some("{}"));
}

#[tokio::test]
async fn inputs_are_refreshed_after_pre_invoke() {
    // test_handler.py: TestRefreshInputsAttribute.test_refresh_inputs_after_mutation
    let fixture = Fixture::new();
    fixture
        .handler
        .on_call_start(
            "node_1",
            &json!({
                "component_id": "node_1",
                "component_type": "LLM",
                "workflow_id": "wf1",
            }),
            Some(&json!({"CHAT_HISTORY": null})),
        )
        .await;
    fixture
        .handler
        .on_pre_invoke(
            "node_1",
            Some(&json!({"CHAT_HISTORY": [{"role": "user", "content": "hi"}], "selfname": "test_user"})),
        )
        .await;
    fixture
        .handler
        .on_call_done("node_1", Some(&json!({"result": "ok"})))
        .await;

    let span = fixture.span("chat");
    let inputs = attribute(&span, OJ_WORKFLOW_INPUTS).expect("inputs recorded");
    assert!(inputs.contains("CHAT_HISTORY"), "{inputs}");
    assert!(inputs.contains("test_user"), "{inputs}");
    assert!(!inputs.contains("null"), "{inputs}");
}

#[tokio::test]
async fn failure_marks_the_span_error() {
    // test_handler.py: test_workflow_invoke_error_marks_error,
    // test_workflow_invoke_error_sets_error_type
    let fixture = Fixture::new();
    fixture
        .handler
        .on_call_start("wf_root", &json!({"workflow_id": "wf1"}), None)
        .await;
    fixture
        .handler
        .on_call_error("wf_root", &Error::workflow("wf failed"))
        .await;

    let span = fixture.span("wf_root");
    assert!(matches!(span.status, Status::Error { .. }));
    assert_eq!(
        attribute(&span, OJ_STATUS).as_deref(),
        Some(SPAN_STATUS_ERROR)
    );
    assert_eq!(
        attribute(&span, OJ_WORKFLOW_ERROR_MESSAGE).as_deref(),
        Some("wf failed")
    );
    assert_eq!(
        attribute(&span, OJ_ERROR).as_deref(),
        Some(r#"{"error_code":"workflow","message":"wf failed"}"#)
    );
    assert_eq!(attribute(&span, ERROR_TYPE).as_deref(), Some("workflow"));
    assert!(event_names(&span).contains(&"exception".to_owned()));
    assert!(attribute(&span, OJ_END_TIME).is_some());
    assert!(attribute(&span, OJ_ELAPSED_TIME).is_some());
    assert!(fixture.handler.manager.is_empty());
    assert!(lock(&fixture.handler.state).layer_roots.is_empty());
}

#[tokio::test]
async fn interrupt_keeps_the_span_successful() {
    // test_handler.py: test_workflow_invoke_graph_interrupt_marks_interrupted
    let fixture = Fixture::new();
    fixture
        .handler
        .on_call_start(
            "questioner",
            &json!({
                "component_id": "questioner",
                "component_type": "Start",
                "workflow_id": "wf1",
            }),
            None,
        )
        .await;
    fixture
        .handler
        .on_call_interrupted("questioner", Some("waiting for user input"))
        .await;

    let span = fixture.span("component.questioner");
    assert_ne!(
        span.status,
        Status::Error {
            description: "waiting for user input".into()
        },
        "an interrupt is not a failure"
    );
    assert_eq!(
        attribute(&span, OJ_STATUS).as_deref(),
        Some(SPAN_STATUS_INTERRUPTED)
    );
    assert!(attribute(&span, OJ_ERROR).is_none());
    assert_eq!(
        attribute(&span, OJ_WORKFLOW_ERROR_MESSAGE).as_deref(),
        Some("waiting for user input")
    );
    assert!(attribute(&span, OJ_END_TIME).is_some());
    assert!(attribute(&span, OJ_ELAPSED_TIME).is_some());
    assert!(!event_names(&span).contains(&"exception".to_owned()));
}

#[tokio::test]
async fn stale_spans_are_cleaned_when_the_trace_id_changes() {
    // test_handler.py: TestSessionBoundary / _cleanup_stale_spans coverage
    let mut fixture = Fixture::new();
    fixture
        .handler
        .on_call_start("wf_root", &json!({"workflow_id": "wf1"}), None)
        .await;
    fixture
        .handler
        .on_call_start(
            "comp",
            &json!({
                "component_id": "comp",
                "component_type": "Start",
                "workflow_id": "wf1",
            }),
            None,
        )
        .await;
    fixture.handler.set_trace_id("trace-2");

    let names: Vec<String> = fixture
        .spans()
        .into_iter()
        .map(|span| span.name.to_string())
        .collect();
    assert!(names.contains(&"wf_root".to_owned()), "{names:?}");
    assert!(names.contains(&"component.comp".to_owned()), "{names:?}");
    let state = lock(&fixture.handler.state);
    assert!(state.layer_roots.is_empty());
    assert!(state.components.is_empty());
    assert!(state.cached_root_ctx.is_none());
    drop(state);
    assert!(fixture.handler.manager.is_empty());
}

#[tokio::test]
async fn component_span_records_the_workflow_and_loop_attributes() {
    // test_handler.py: test_workflow_attributes_set
    let fixture = Fixture::new();
    fixture
        .handler
        .on_call_start(
            "wf_root",
            &json!({"workflow_id": "wf1", "workflow_name": "MyWF", "workflow_version": "2.0"}),
            None,
        )
        .await;
    let mut metadata = json!({
        "component_id": "comp1",
        "component_type": "Tool",
        "component_name": "tool1",
        "workflow_id": "wf1",
        "parent_node_id": "",
    });
    metadata["loop_node_id"] = json!("loop-1");
    metadata["loop_index"] = json!(2);
    fixture
        .handler
        .on_call_start("comp1", &metadata, None)
        .await;
    fixture.handler.on_call_done("comp1", None).await;
    fixture.handler.on_call_done("wf_root", None).await;

    let span = fixture.span("execute_tool tool1");
    assert_eq!(attribute(&span, OJ_WORKFLOW_ID).as_deref(), Some("wf1"));
    assert_eq!(
        attribute(&span, OJ_WORKFLOW_COMPONENT_ID).as_deref(),
        Some("comp1")
    );
    assert_eq!(
        attribute(&span, OJ_WORKFLOW_COMPONENT_TYPE).as_deref(),
        Some("Tool")
    );
    assert_eq!(
        attribute(&span, OJ_WORKFLOW_COMPONENT_NAME).as_deref(),
        Some("tool1")
    );
    assert_eq!(
        attribute(&span, OJ_WORKFLOW_LOOP_NODE_ID).as_deref(),
        Some("loop-1")
    );
    assert_eq!(
        attribute(&span, OJ_WORKFLOW_LOOP_INDEX).as_deref(),
        Some("2"),
        "the loop index is recorded as text"
    );
    let root = fixture.span("wf_root");
    assert_eq!(span.parent_span_id, root.span_context.span_id());
}

#[tokio::test]
async fn interactive_inputs_are_recorded_while_the_span_is_open() {
    // test_handler.py: WorkflowHandler.on_interact (openjiuwen.interactive_inputs)
    let fixture = Fixture::new();
    fixture
        .handler
        .on_call_start(
            "comp1",
            &json!({"workflow_id": "wf1", "component_id": "comp1", "component_type": "Start"}),
            None,
        )
        .await;
    fixture
        .handler
        .on_interact("comp1", &json!({"answer": "北京"}))
        .await;
    fixture.handler.on_call_done("comp1", None).await;
    assert_eq!(
        attribute(&fixture.span("component.comp1"), OJ_INTERACTIVE_INPUTS).as_deref(),
        Some(r#"{"answer":"北京"}"#)
    );
}

#[test]
fn from_handle_requires_an_active_handle() {
    // test_handler.py: handlers are built from an initialized tracer
    let config = plain_config();
    let handle = init_otel_tracer(config).expect("the memory exporter initializes");
    let handler = OtelWorkflowHandler::from_handle(&handle).expect("an active handle builds");
    assert_eq!(handler.config.tracer_name, handle.tracer_name());
    handle.shutdown();
    let error = OtelWorkflowHandler::from_handle(&handle).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Config);
}

#[test]
fn handler_is_send_sync_and_debug() {
    fn assert_shareable<T: Send + Sync + std::fmt::Debug>() {}
    assert_shareable::<OtelWorkflowHandler>();
    let fixture = Fixture::new();
    let rendered = format!("{:?}", fixture.handler);
    assert!(rendered.contains("OtelWorkflowHandler"), "{rendered}");
    assert!(rendered.contains("HandlerState"), "{rendered}");
}
