//! Unit tests of the agent handler.
//!
//! Port of `TestOtelAgentHandler` in
//! `tests/unit_tests/extensions/tracer_otel/test_handler.py` and of
//! `test_otel_tool_span_dedup.py`, adapted to the Rust frame API: callbacks take
//! a [`TraceSpan`] plus the payloads, and the handler exports through a private
//! in-memory provider.

// A test module owns its panics: a failed `unwrap`/`expect` is an assertion.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;

use opentelemetry::trace::SpanKind;
use opentelemetry::Value as OtelValue;
use opentelemetry_sdk::trace::SpanData;
use serde_json::{json, Value};

use openjiuwen_common::error::{Error, ErrorKind};
use openjiuwen_core::session::tracer::{InvokeType, TraceSpan, TraceSpanKind};

use super::*;
use crate::otel::config::{ExporterType, OtelTracerConfig};
use crate::otel::handle::init_otel_tracer;
use crate::otel::provider::CaptureExporter;
use crate::otel::semconv::{
    ERROR_TYPE, GEN_AI_AGENT_NAME, GEN_AI_DATA_SOURCE_ID, GEN_AI_EMBEDDINGS_DIMENSION_COUNT,
    GEN_AI_INPUT_MESSAGES, GEN_AI_MEMORY_RECORD_COUNT, GEN_AI_OPERATION_NAME,
    GEN_AI_OUTPUT_MESSAGES, GEN_AI_REQUEST_MODEL, GEN_AI_RESPONSE_MODEL, GEN_AI_RETRIEVAL_TOP_K,
    GEN_AI_SYSTEM_INSTRUCTIONS, GEN_AI_TOOL_CALL_ID, GEN_AI_TOOL_NAME, GEN_AI_TOOL_TYPE,
    GEN_AI_USAGE_INPUT_TOKENS, GEN_AI_USAGE_OUTPUT_TOKENS, OJ_AGENT_ERROR_MESSAGE, OJ_AGENT_INPUTS,
    OJ_AGENT_INVOKE_TYPE, OJ_AGENT_NAME, OJ_AGENT_OUTPUTS, OJ_ELAPSED_TIME, OJ_END_TIME, OJ_ERROR,
    OJ_GEN_AI_TRACE_NAME, OJ_INVOKE_ID, OJ_SESSION_ID, OJ_START_TIME, OJ_STATUS, OJ_TRACE_ID,
    SPAN_STATUS_ERROR, SPAN_STATUS_FINISH,
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
    handler: OtelAgentHandler,
}

impl Fixture {
    /// Builds a fixture over `config`, with an optional trace id.
    fn build(config: OtelTracerConfig, trace_id: Option<&str>) -> Self {
        let handle = init_otel_tracer(config.clone()).expect("the memory exporter initializes");
        let capture = handle
            .capture()
            .expect("the memory exporter captures spans");
        let tracer = handle.tracer().expect("the memory handle is active");
        let handler = OtelAgentHandler::new(tracer, config, trace_id.map(str::to_owned));
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

    /// The exported names, after flushing the batch processor.
    fn names(&self) -> Vec<String> {
        self.handle.force_flush();
        self.capture.names()
    }

    /// The one exported span named `name`.
    fn span(&self, name: &str) -> SpanData {
        self.handle.force_flush();
        self.capture
            .span_named(name)
            .unwrap_or_else(|| panic!("no span named {name} in {:?}", self.capture.names()))
    }
}

/// A frame of an agent invocation, named by `invoke_id`.
///
/// The name stays empty on purpose: Python's tests create spans through the
/// session `SpanManager`, which leaves `name`/`invoke_type` unset so the handler
/// has to derive them.
fn agent_frame(trace_id: &str, invoke_id: &str) -> TraceSpan {
    TraceSpan::new(trace_id, "", invoke_id, "", TraceSpanKind::Agent)
}

/// A child frame `.with_parent`-linked to `parent_invoke_id`.
fn child_frame(trace_id: &str, invoke_id: &str, parent_invoke_id: &str) -> TraceSpan {
    agent_frame(trace_id, invoke_id).with_parent(parent_invoke_id)
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

/// The parsed JSON value of one attribute.
fn json_attribute(data: &SpanData, key: &str) -> Value {
    let text = attribute(data, key).unwrap_or_else(|| panic!("no attribute {key}"));
    serde_json::from_str(&text)
        .unwrap_or_else(|error| panic!("{key} is not JSON ({error}): {text}"))
}

/// The names of the events a span carries.
fn event_names(data: &SpanData) -> Vec<String> {
    data.events
        .iter()
        .map(|event| event.name.to_string())
        .collect()
}

/// Asserts that a redaction hashed the attribute value.
fn assert_hashed(text: &str, field: &str) {
    assert!(
        text.starts_with("sha256:"),
        "{field} must be hashed, got {text}"
    );
}

#[tokio::test]
async fn llm_span_matches_the_python_contract() {
    // test_handler.py: test_agent_llm_start_creates_span, test_agent_llm_end_closes_span
    let fixture = Fixture::new();
    let frame = agent_frame("trace-1", "invoke-1");
    fixture
        .handler
        .on_llm_start(&frame, &json!("hello"), &json!({"class_name": "TestModel"}))
        .await;
    assert!(
        fixture.spans().is_empty(),
        "the span only closes on on_llm_end"
    );
    fixture
        .handler
        .on_llm_end(
            &frame,
            &json!({"outputs": {"role": "assistant", "content": "world"}}),
        )
        .await;

    let span = fixture.span("chat TestModel");
    assert_eq!(span.span_kind, SpanKind::Client);
    assert_eq!(
        attribute(&span, GEN_AI_REQUEST_MODEL).as_deref(),
        Some("TestModel")
    );
    assert_eq!(
        attribute(&span, GEN_AI_OPERATION_NAME).as_deref(),
        Some("chat")
    );
    assert_eq!(
        attribute(&span, OJ_AGENT_INVOKE_TYPE).as_deref(),
        Some("llm")
    );
    assert_eq!(
        attribute(&span, OJ_AGENT_NAME).as_deref(),
        Some("TestModel")
    );
    assert_eq!(
        attribute(&span, OJ_GEN_AI_TRACE_NAME).as_deref(),
        Some("TestModel")
    );
    assert_eq!(attribute(&span, OJ_TRACE_ID).as_deref(), Some("trace-1"));
    assert_eq!(attribute(&span, OJ_INVOKE_ID).as_deref(), Some("invoke-1"));
    assert!(
        attribute(&span, OJ_SESSION_ID).is_none(),
        "the frame carries no session id"
    );
    assert_eq!(
        json_attribute(&span, GEN_AI_INPUT_MESSAGES),
        json!([{"role": "user", "parts": [{"type": "text", "content": "hello"}]}])
    );
    assert_eq!(
        json_attribute(&span, GEN_AI_OUTPUT_MESSAGES),
        json!([{"role": "assistant", "parts": [{"type": "text", "content": "world"}]}])
    );
    assert_eq!(span.status, opentelemetry::trace::Status::Ok);
    assert_eq!(
        attribute(&span, OJ_STATUS).as_deref(),
        Some(SPAN_STATUS_FINISH)
    );
    // The handler generates start/end/elapsed when the frame has none.
    assert_eq!(attribute(&span, OJ_START_TIME).map(|s| s.len()), Some(26));
    assert_eq!(attribute(&span, OJ_END_TIME).map(|s| s.len()), Some(26));
    let elapsed = attribute(&span, OJ_ELAPSED_TIME).expect("elapsed_time is written");
    assert!(
        elapsed.ends_with("ms") || elapsed.ends_with('s'),
        "{elapsed}"
    );
    assert_eq!(fixture.names(), ["chat TestModel"]);
}

#[tokio::test]
async fn llm_system_instructions_are_split_from_messages() {
    // test_handler.py: test_agent_llm_inputs_normalized_to_dict
    let fixture = Fixture::new();
    let frame = agent_frame("trace-1", "invoke-1");
    fixture
        .handler
        .on_llm_start(
            &frame,
            &json!({"inputs": [
                {"role": "system", "content": "you are helpful"},
                {"role": "user", "content": "hello"},
            ]}),
            &json!({"class_name": "M"}),
        )
        .await;
    fixture.handler.on_llm_end(&frame, &json!("world")).await;
    let span = fixture.span("chat M");
    assert_eq!(
        json_attribute(&span, GEN_AI_SYSTEM_INSTRUCTIONS),
        json!([{"type": "text", "content": "you are helpful"}])
    );
    assert_eq!(
        json_attribute(&span, GEN_AI_INPUT_MESSAGES),
        json!([{"role": "user", "parts": [{"type": "text", "content": "hello"}]}])
    );
}

#[tokio::test]
async fn llm_usage_and_cost_attributes_are_recorded() {
    // test_handler.py: test_agent_llm_usage_and_cost_attrs
    // (TestGenAiSemconvAttrs), adapted to a JSON payload.
    let fixture = Fixture::new();
    let frame = agent_frame("trace-1", "invoke-1");
    fixture
        .handler
        .on_llm_start(&frame, &json!("hi"), &json!({"class_name": "M"}))
        .await;
    fixture
        .handler
        .on_llm_end(
            &frame,
            &json!({"outputs": {
                "role": "assistant",
                "content": "world",
                "usage_metadata": {
                    "model_name": "gpt-test",
                    "input_tokens": 10,
                    "output_tokens": 4,
                    "cache_read_tokens": 2,
                    "reasoning_tokens": 1,
                    "total_cost": 0.5,
                },
            }}),
        )
        .await;
    let span = fixture.span("chat M");
    assert_eq!(
        attribute(&span, GEN_AI_RESPONSE_MODEL).as_deref(),
        Some("gpt-test")
    );
    assert_eq!(int_attribute(&span, GEN_AI_USAGE_INPUT_TOKENS), Some(10));
    assert_eq!(int_attribute(&span, GEN_AI_USAGE_OUTPUT_TOKENS), Some(4));
}

#[tokio::test]
async fn llm_request_event_is_emitted() {
    // test_handler.py: test_agent_llm_request_event (Python forwards kwargs; the
    // Rust frame carries none, so only the event name is pinned).
    let fixture = Fixture::new();
    let frame = agent_frame("trace-1", "invoke-1");
    fixture
        .handler
        .on_llm_start(&frame, &json!("hi"), &json!({"class_name": "M"}))
        .await;
    fixture.handler.on_llm_request(&frame).await;
    fixture.handler.on_llm_end(&frame, &json!("ok")).await;
    assert_eq!(event_names(&fixture.span("chat M")), ["llm.request"]);
}

#[tokio::test]
async fn llm_prompts_and_completions_are_hashed() {
    // test_handler.py: test_agent_llm_prompt_redaction
    let fixture = Fixture::build(redacting_config(), Some("trace-1"));
    let frame = agent_frame("trace-1", "invoke-1");
    fixture
        .handler
        .on_llm_start(&frame, &json!("secret data"), &json!({"class_name": "M"}))
        .await;
    fixture
        .handler
        .on_llm_end(&frame, &json!("secret response"))
        .await;
    let span = fixture.span("chat M");
    let request = json_attribute(&span, GEN_AI_INPUT_MESSAGES);
    let response = json_attribute(&span, GEN_AI_OUTPUT_MESSAGES);
    assert_hashed(
        request[0]["parts"][0]["content"].as_str().unwrap(),
        "input message",
    );
    assert_hashed(
        response[0]["parts"][0]["content"].as_str().unwrap(),
        "output message",
    );
}

#[tokio::test]
async fn llm_prompts_can_be_left_readable_while_completions_are_hashed() {
    // test_handler.py: test_agent_llm_prompt_completion_split_redaction
    let config = OtelTracerConfig {
        redaction_enabled: true,
        redact_prompts: Some(false),
        redact_completions: Some(true),
        ..OtelTracerConfig::default()
    }
    .with_exporter_type(ExporterType::Memory);
    let fixture = Fixture::build(config, Some("trace-1"));
    let frame = agent_frame("trace-1", "invoke-1");
    fixture
        .handler
        .on_llm_start(
            &frame,
            &json!("visible prompt"),
            &json!({"class_name": "M"}),
        )
        .await;
    fixture
        .handler
        .on_llm_end(&frame, &json!("secret response"))
        .await;
    let span = fixture.span("chat M");
    assert_eq!(
        json_attribute(&span, GEN_AI_INPUT_MESSAGES)[0]["parts"][0]["content"],
        json!("visible prompt")
    );
    assert_hashed(
        json_attribute(&span, GEN_AI_OUTPUT_MESSAGES)[0]["parts"][0]["content"]
            .as_str()
            .unwrap(),
        "output message",
    );
}

#[tokio::test]
async fn llm_fields_use_the_frame_values_when_present() {
    // test_handler.py: test_agent_fields_use_span_value_when_present
    let fixture = Fixture::new();
    let frame = agent_frame("trace-1", "invoke-1").with_invoke_type(InvokeType::Llm);
    let mut frame = frame;
    frame.name = "MyLLMModel".to_owned();
    fixture
        .handler
        .on_llm_start(&frame, &json!("hello"), &json!({"class_name": "TestLLM"}))
        .await;
    fixture.handler.on_llm_end(&frame, &json!("world")).await;
    let span = fixture.span("chat TestLLM");
    assert_eq!(
        attribute(&span, OJ_AGENT_INVOKE_TYPE).as_deref(),
        Some("llm")
    );
    assert_eq!(
        attribute(&span, OJ_AGENT_NAME).as_deref(),
        Some("MyLLMModel")
    );
}

#[tokio::test]
async fn llm_name_falls_back_to_an_unknown_class_name() {
    // test_handler.py: the `class_name` default of on_llm_start.
    let fixture = Fixture::new();
    let frame = agent_frame("trace-1", "invoke-1");
    fixture
        .handler
        .on_llm_start(&frame, &json!("hello"), &json!(null))
        .await;
    fixture.handler.on_llm_end(&frame, &json!("world")).await;
    let span = fixture.span("chat unknown");
    assert_eq!(
        attribute(&span, OJ_AGENT_NAME).as_deref(),
        Some(""),
        "the entity name falls back to the empty class name"
    );
}

#[tokio::test]
async fn llm_error_closes_the_span_with_the_agent_error_attributes() {
    // test_handler.py: test_agent_llm_error_marks_error
    let fixture = Fixture::new();
    let frame = agent_frame("trace-1", "invoke-1");
    fixture
        .handler
        .on_llm_start(&frame, &json!(null), &json!({"class_name": "M"}))
        .await;
    fixture
        .handler
        .on_llm_error(&frame, &Error::tool("test error"))
        .await;
    let span = fixture.span("chat M");
    assert_eq!(
        span.status,
        opentelemetry::trace::Status::Error {
            description: "test error".into()
        }
    );
    assert_eq!(
        attribute(&span, OJ_AGENT_ERROR_MESSAGE).as_deref(),
        Some("test error")
    );
    assert_eq!(
        attribute(&span, OJ_STATUS).as_deref(),
        Some(SPAN_STATUS_ERROR)
    );
    assert_eq!(
        attribute(&span, OJ_ERROR).as_deref(),
        Some(r#"{"error_code":"tool","message":"test error"}"#)
    );
    assert_eq!(attribute(&span, ERROR_TYPE).as_deref(), Some("tool"));
    assert!(event_names(&span).contains(&"exception".to_owned()));
}

#[tokio::test]
async fn unknown_invoke_ids_are_ignored_by_end_and_error() {
    // Python's `_end_and_pop`/`_mark_error_and_end` return on a missing state.
    let fixture = Fixture::new();
    let frame = agent_frame("trace-1", "never-started");
    fixture.handler.on_llm_end(&frame, &json!("output")).await;
    fixture
        .handler
        .on_llm_error(&frame, &Error::validation("boom"))
        .await;
    fixture.handler.on_llm_request(&frame).await;
    fixture
        .handler
        .on_plugin_end(&frame, &json!("output"))
        .await;
    fixture
        .handler
        .on_chain_error(&frame, &Error::validation("boom"))
        .await;
    assert!(fixture.spans().is_empty(), "no span was ever started");
}

#[tokio::test]
async fn tool_span_carries_the_genai_identity() {
    // test_otel_tool_span_dedup.py: the rail span contract, plus the
    // test_handler.py plugin assertions (name, kind, invoke type, agent name).
    let fixture = Fixture::new();
    let frame = agent_frame("trace-1", "invoke-tool");
    let info = json!({
        "class_name": "get_weather",
        "type": "plugin",
        "tool_type": "function",
        "tool_call_id": "call-1",
        "agent_name": "weather_agent",
    });
    fixture
        .handler
        .on_plugin_start(&frame, &json!({"inputs": {"city": "Beijing"}}), &info)
        .await;
    fixture
        .handler
        .on_plugin_end(&frame, &json!({"outputs": "Beijing: 22C, sunny"}))
        .await;

    let span = fixture.span("execute_tool get_weather");
    assert_eq!(span.span_kind, SpanKind::Internal);
    assert_eq!(
        attribute(&span, GEN_AI_OPERATION_NAME).as_deref(),
        Some("execute_tool")
    );
    assert_eq!(
        attribute(&span, GEN_AI_TOOL_NAME).as_deref(),
        Some("get_weather")
    );
    assert_eq!(
        attribute(&span, GEN_AI_TOOL_CALL_ID).as_deref(),
        Some("call-1")
    );
    assert_eq!(
        attribute(&span, GEN_AI_TOOL_TYPE).as_deref(),
        Some("function")
    );
    assert_eq!(
        attribute(&span, GEN_AI_AGENT_NAME).as_deref(),
        Some("weather_agent")
    );
    assert_eq!(
        attribute(&span, OJ_GEN_AI_TRACE_NAME).as_deref(),
        Some("weather_agent")
    );
    assert_eq!(
        attribute(&span, OJ_AGENT_INVOKE_TYPE).as_deref(),
        Some("plugin")
    );
    assert_eq!(
        attribute(&span, OJ_AGENT_NAME).as_deref(),
        Some("get_weather")
    );
    assert_eq!(
        json_attribute(&span, OJ_AGENT_INPUTS),
        json!({"inputs": {"city": "Beijing"}})
    );
    assert_eq!(
        json_attribute(&span, OJ_AGENT_OUTPUTS),
        json!({"outputs": "Beijing: 22C, sunny"})
    );
    assert_eq!(
        attribute(&span, OJ_STATUS).as_deref(),
        Some(SPAN_STATUS_FINISH)
    );
}

#[tokio::test]
async fn tool_span_payloads_are_redacted() {
    // test_otel_tool_span_dedup.py stores the decorator payloads via `redact()`;
    // with redaction enabled both payload attributes are hashed.
    let fixture = Fixture::build(redacting_config(), Some("trace-1"));
    let frame = agent_frame("trace-1", "invoke-tool");
    fixture
        .handler
        .on_plugin_start(
            &frame,
            &json!({"inputs": {"city": "Beijing"}}),
            &json!({"class_name": "get_weather"}),
        )
        .await;
    fixture
        .handler
        .on_plugin_end(&frame, &json!({"outputs": "22C"}))
        .await;
    let span = fixture.span("execute_tool get_weather");
    assert_hashed(
        attribute(&span, OJ_AGENT_INPUTS).as_deref().unwrap(),
        "tool inputs",
    );
    assert_hashed(
        attribute(&span, OJ_AGENT_OUTPUTS).as_deref().unwrap(),
        "tool outputs",
    );
}

#[tokio::test]
async fn tool_dedup_drops_the_decorator_twin() {
    // test_otel_tool_span_dedup.py: test_rail_and_decorator_emit_one_tool_span
    let fixture = Fixture::new();
    let rail = agent_frame("trace-1", "invoke-rail");
    let rail_info = json!({"class_name": "get_weather", "type": "plugin"});
    fixture
        .handler
        .on_plugin_start(&rail, &json!({"inputs": {"city": "Beijing"}}), &rail_info)
        .await;
    assert!(
        fixture.handler.has_open_tool_span("get_weather", "trace-1"),
        "the rail span is flagged for the dedup"
    );

    let decorator = agent_frame("trace-1", "invoke-decorator");
    let decorator_info = json!({"class_name": "get_weather", "type": "tool"});
    fixture
        .handler
        .on_plugin_start(
            &decorator,
            &json!({"inputs": {"city": "Beijing"}}),
            &decorator_info,
        )
        .await;
    // The decorator's nested end/error events are no-ops.
    fixture
        .handler
        .on_plugin_end(&decorator, &json!("22C"))
        .await;
    fixture.handler.on_plugin_end(&rail, &json!("22C")).await;

    assert_eq!(
        fixture.names(),
        ["execute_tool get_weather"],
        "exactly one span survives the two emitters"
    );
    assert!(!fixture.handler.has_open_tool_span("get_weather", "trace-1"));
}

#[tokio::test]
async fn tool_dedup_keeps_a_lone_decorator_event() {
    // test_otel_tool_span_dedup.py: test_decorator_only_still_emits_tool_span
    let fixture = Fixture::new();
    let decorator = agent_frame("trace-1", "invoke-decorator");
    fixture
        .handler
        .on_plugin_start(
            &decorator,
            &json!({"inputs": {"city": "Beijing"}}),
            &json!({"class_name": "get_weather", "type": "tool"}),
        )
        .await;
    fixture
        .handler
        .on_plugin_end(&decorator, &json!("22C"))
        .await;
    let span = fixture.span("execute_tool get_weather");
    assert!(
        attribute(&span, GEN_AI_TOOL_CALL_ID).is_none(),
        "the decorator event carries no tool call id"
    );
}

#[tokio::test]
async fn tool_dedup_scopes_by_the_handler_trace_id() {
    // test_otel_tool_span_dedup.py: same-name parallel calls stay separate; the
    // handler trace id scopes the probe (Python reads it off the frame).
    let mut fixture = Fixture::new();
    fixture.handler.set_trace_id("trace-9");
    let rail = agent_frame("trace-9", "invoke-rail");
    let rail_info = json!({"class_name": "get_weather", "type": "plugin"});
    fixture
        .handler
        .on_plugin_start(&rail, &json!(null), &rail_info)
        .await;
    let decorator = agent_frame("trace-9", "invoke-decorator");
    fixture
        .handler
        .on_plugin_start(
            &decorator,
            &json!(null),
            &json!({"class_name": "get_weather", "type": "tool"}),
        )
        .await;
    fixture.handler.on_plugin_end(&rail, &json!("22C")).await;
    fixture
        .handler
        .on_plugin_end(&decorator, &json!("22C"))
        .await;
    assert_eq!(fixture.names(), ["execute_tool get_weather"]);

    // A different handler trace id no longer matches the rail span of trace-9.
    let fixture = Fixture::new();
    let rail = agent_frame("trace-9", "invoke-rail");
    fixture
        .handler
        .on_plugin_start(&rail, &json!(null), &rail_info)
        .await;
    let decorator = agent_frame("trace-9", "invoke-decorator");
    fixture
        .handler
        .on_plugin_start(
            &decorator,
            &json!(null),
            &json!({"class_name": "get_weather", "type": "tool"}),
        )
        .await;
    fixture.handler.on_plugin_end(&rail, &json!("22C")).await;
    fixture
        .handler
        .on_plugin_end(&decorator, &json!("22C"))
        .await;
    assert_eq!(
        fixture.names(),
        ["execute_tool get_weather", "execute_tool get_weather"],
        "the dedup only matches spans of the handler's own trace"
    );
}

#[tokio::test]
async fn plugin_error_closes_one_error_span() {
    // test_otel_tool_span_dedup.py: test_failing_tool_emits_single_error_span
    let fixture = Fixture::new();
    let rail = agent_frame("trace-1", "invoke-rail");
    let rail_info = json!({"class_name": "get_weather", "type": "plugin"});
    fixture
        .handler
        .on_plugin_start(&rail, &json!(null), &rail_info)
        .await;
    let decorator = agent_frame("trace-1", "invoke-decorator");
    fixture
        .handler
        .on_plugin_start(
            &decorator,
            &json!(null),
            &json!({"class_name": "get_weather", "type": "tool"}),
        )
        .await;
    fixture
        .handler
        .on_plugin_error(&rail, &Error::tool("weather service down"))
        .await;
    fixture
        .handler
        .on_plugin_error(&decorator, &Error::tool("weather service down"))
        .await;

    assert_eq!(fixture.names(), ["execute_tool get_weather"]);
    let span = fixture.span("execute_tool get_weather");
    assert_eq!(
        attribute(&span, OJ_STATUS).as_deref(),
        Some(SPAN_STATUS_ERROR)
    );
    assert_eq!(
        attribute(&span, OJ_AGENT_ERROR_MESSAGE).as_deref(),
        Some("weather service down")
    );
    assert_eq!(
        attribute(&span, OJ_AGENT_NAME).as_deref(),
        Some("get_weather")
    );
}

#[tokio::test]
async fn retriever_span_records_top_k_and_data_source() {
    // test_handler.py: test_agent_retriever_records_top_k_and_data_source
    let fixture = Fixture::new();
    let frame = agent_frame("trace-1", "invoke-retriever");
    fixture
        .handler
        .on_retriever_start(
            &frame,
            &json!({"top_k": 5, "data_source_id": "kb-1"}),
            &json!({"class_name": "KnowledgeBase"}),
        )
        .await;
    fixture
        .handler
        .on_retriever_end(&frame, &json!({"results": ["doc-1"]}))
        .await;
    let span = fixture.span("retriever.KnowledgeBase");
    assert_eq!(int_attribute(&span, GEN_AI_RETRIEVAL_TOP_K), Some(5));
    assert_eq!(
        attribute(&span, GEN_AI_DATA_SOURCE_ID).as_deref(),
        Some("kb-1")
    );
    assert_eq!(
        attribute(&span, OJ_AGENT_INVOKE_TYPE).as_deref(),
        Some("retriever")
    );
    assert_eq!(
        json_attribute(&span, OJ_AGENT_OUTPUTS),
        json!({"results": ["doc-1"]})
    );
}

#[tokio::test]
async fn embedding_span_records_dimension_model_and_usage() {
    // test_handler.py: test_agent_embedding_start/end (issue #1833)
    let fixture = Fixture::new();
    let frame = agent_frame("trace-1", "invoke-embedding");
    fixture
        .handler
        .on_embedding_start(
            &frame,
            &json!({"dimensions": 8}),
            &json!({"model_name": "text-embedding-3-small"}),
        )
        .await;
    fixture
        .handler
        .on_embedding_end(
            &frame,
            &json!({
                "response_model": "text-embedding-3-small",
                "dimension": 8,
                "usage_input_tokens": 12,
            }),
        )
        .await;
    let span = fixture.span("embeddings text-embedding-3-small");
    assert_eq!(
        attribute(&span, GEN_AI_OPERATION_NAME).as_deref(),
        Some("embeddings")
    );
    assert_eq!(
        attribute(&span, GEN_AI_REQUEST_MODEL).as_deref(),
        Some("text-embedding-3-small")
    );
    assert_eq!(
        attribute(&span, GEN_AI_RESPONSE_MODEL).as_deref(),
        Some("text-embedding-3-small")
    );
    assert_eq!(
        int_attribute(&span, GEN_AI_EMBEDDINGS_DIMENSION_COUNT),
        Some(8)
    );
    assert_eq!(int_attribute(&span, GEN_AI_USAGE_INPUT_TOKENS), Some(12));
    assert_eq!(
        attribute(&span, OJ_AGENT_INVOKE_TYPE).as_deref(),
        Some("embedding")
    );
}

#[tokio::test]
async fn memory_span_records_the_record_count() {
    // test_handler.py: test_agent_memory_start_records_count (issue #1833)
    let fixture = Fixture::new();
    let frame = agent_frame("trace-1", "invoke-memory");
    fixture
        .handler
        .on_memory_start(
            &frame,
            &json!({"record_count": 3}),
            &json!({"class_name": "MemoryStore"}),
        )
        .await;
    fixture.handler.on_memory_end(&frame, &json!("ok")).await;
    let span = fixture.span("memory.MemoryStore");
    assert_eq!(int_attribute(&span, GEN_AI_MEMORY_RECORD_COUNT), Some(3));
    assert_eq!(
        attribute(&span, OJ_AGENT_INVOKE_TYPE).as_deref(),
        Some("memory")
    );
}

#[tokio::test]
async fn every_other_family_uses_the_python_naming_rule() {
    // test_handler.py: test_agent_non_llm_internal_span and the per-family
    // naming assertions of on_prompt/chain/evaluator/workflow.
    let fixture = Fixture::new();
    let prompt = agent_frame("trace-1", "invoke-prompt");
    fixture
        .handler
        .on_prompt_start(&prompt, &json!("in"), &json!({"class_name": "Prompt"}))
        .await;
    fixture.handler.on_prompt_end(&prompt, &json!("out")).await;

    let chain = agent_frame("trace-1", "invoke-chain");
    fixture
        .handler
        .on_chain_start(&chain, &json!("in"), &json!({"class_name": "Chain"}))
        .await;
    fixture.handler.on_chain_end(&chain, &json!("out")).await;

    let evaluator = agent_frame("trace-1", "invoke-evaluator");
    fixture
        .handler
        .on_evaluator_start(&evaluator, &json!("in"), &json!({"class_name": "Eval"}))
        .await;
    fixture
        .handler
        .on_evaluator_end(&evaluator, &json!("out"))
        .await;

    let workflow = agent_frame("trace-1", "invoke-workflow");
    fixture
        .handler
        .on_workflow_start(&workflow, &json!("in"), &json!({"class_name": "Flow"}))
        .await;
    fixture
        .handler
        .on_workflow_end(&workflow, &json!("out"))
        .await;

    assert_eq!(
        fixture.names(),
        [
            "prompt.Prompt",
            "chain.Chain",
            "evaluator.Eval",
            "workflow.Flow",
        ]
    );
    for (name, invoke_type) in [
        ("prompt.Prompt", "prompt"),
        ("chain.Chain", "chain"),
        ("evaluator.Eval", "evaluator"),
        ("workflow.Flow", "workflow"),
    ] {
        let span = fixture.span(name);
        assert_eq!(span.span_kind, SpanKind::Internal);
        assert_eq!(
            attribute(&span, OJ_AGENT_INVOKE_TYPE).as_deref(),
            Some(invoke_type)
        );
        assert_eq!(
            attribute(&span, OJ_AGENT_NAME).as_deref(),
            name.split_once('.').map(|(_, n)| n)
        );
    }
}

#[tokio::test]
async fn child_span_parents_onto_the_open_parent() {
    // test_handler.py: test_agent_parent_child_relation
    let fixture = Fixture::new();
    let parent = agent_frame("trace-1", "invoke-parent");
    fixture
        .handler
        .on_chain_start(&parent, &json!(null), &json!({"class_name": "ParentChain"}))
        .await;
    let child = child_frame("trace-1", "invoke-child", "invoke-parent");
    fixture
        .handler
        .on_plugin_start(&child, &json!(null), &json!({"class_name": "ChildTool"}))
        .await;
    fixture
        .handler
        .on_plugin_end(&child, &json!("result"))
        .await;
    fixture.handler.on_chain_end(&parent, &json!("done")).await;

    let parent_span = fixture.span("chain.ParentChain");
    let child_span = fixture.span("execute_tool ChildTool");
    assert_eq!(
        child_span.parent_span_id,
        parent_span.span_context.span_id()
    );
    assert_eq!(
        child_span.span_context.trace_id(),
        parent_span.span_context.trace_id()
    );
}

#[tokio::test]
async fn session_id_falls_back_to_the_injected_value() {
    // test_handler.py: test_agent_span_contains_session_id_when_provided,
    // test_agent_span_session_id_survives_concurrent_handler_reuse
    let mut fixture = Fixture::new();
    fixture.handler.set_session_id("injected-session");
    let frame = agent_frame("trace-1", "invoke-1");
    fixture
        .handler
        .on_llm_start(&frame, &json!("hi"), &json!({"class_name": "M"}))
        .await;
    fixture.handler.on_llm_end(&frame, &json!("ok")).await;
    assert_eq!(
        attribute(&fixture.span("chat M"), OJ_SESSION_ID).as_deref(),
        Some("injected-session")
    );

    // The frame's own session id always wins.
    let fixture = Fixture::build(plain_config(), Some("trace-1"));
    let mut owned = agent_frame("trace-1", "invoke-2");
    owned.session_id = "own-session".to_owned();
    fixture
        .handler
        .on_llm_start(&owned, &json!("hi"), &json!({"class_name": "M"}))
        .await;
    fixture.handler.on_llm_end(&owned, &json!("ok")).await;
    assert_eq!(
        attribute(&fixture.span("chat M"), OJ_SESSION_ID).as_deref(),
        Some("own-session")
    );
}

#[tokio::test]
async fn from_handle_rejects_an_inactive_handle() {
    // Python builds the handler from a live tracer; an inactive Rust handle has
    // no tracer to bind to.
    let handle = init_otel_tracer(plain_config()).expect("the memory exporter initializes");
    let handler = OtelAgentHandler::from_handle(&handle).expect("an active handle binds a tracer");
    assert!(!handler.has_open_tool_span("tool", "trace-1"));
    handle.shutdown();
    let error = OtelAgentHandler::from_handle(&handle).expect_err("a shut-down handle is inactive");
    assert_eq!(error.kind(), ErrorKind::Config);
}

#[test]
fn handler_is_send_and_sync() {
    // The session tracer may fan frames out from several tasks.
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<OtelAgentHandler>();
}
