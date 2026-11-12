//! Public-API integration tests of the `otel` feature.
//!
//! These tests drive the crate the way a host does — a `TracerHandle`, the
//! session `Tracer`/`TraceSink` seam, and the typed `OtelAgentHandler` — and
//! observe the result through the public `CaptureExporter`.
//!
//! Python counterparts: `tests/unit_tests/extensions/tracer_otel/test_handler.py`,
//! `test_otel_tool_span_dedup.py`, and the naming/status expectations of
//! `test_setup.py`. Unit tests inside `src/otel/` pin the same behavior at
//! module scope; these tests pin what callers see across the crate boundary.

#![cfg(feature = "otel")]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::Arc;

use futures::executor::block_on;
use opentelemetry::trace::{SpanKind, Status};
use opentelemetry::Key;
use opentelemetry_sdk::trace::SpanData;
use serde_json::json;

use openjiuwen_common::error::Error;
use openjiuwen_core::session::tracer::{InvokeType, TraceSink, TraceSpan, TraceSpanKind, Tracer};
use openjiuwen_extensions::otel::semconv::{
    ERROR_TYPE, GEN_AI_INPUT_MESSAGES, GEN_AI_OPERATION_CHAT, GEN_AI_OPERATION_NAME,
    GEN_AI_OUTPUT_MESSAGES, GEN_AI_PROVIDER_NAME, GEN_AI_PROVIDER_OPENJIUWEN, GEN_AI_TOOL_NAME,
    OJ_AGENT_ERROR_MESSAGE, OJ_AGENT_NAME, OJ_ERROR, OJ_SESSION_ID, OJ_STATUS, OJ_TRACE_ID,
    SPAN_STATUS_ERROR, SPAN_STATUS_FINISH,
};
use openjiuwen_extensions::otel::{
    init_otel_tracer, init_otel_tracer_with_exporter, CaptureExporter, ExporterType,
    OtelAgentHandler, OtelTraceSink, OtelTracerConfig,
};

/// A configuration that keeps payload attributes readable.
fn plain_config() -> OtelTracerConfig {
    OtelTracerConfig {
        redaction_enabled: false, // keep the attribute values readable
        ..OtelTracerConfig::default()
    }
    .with_exporter_type(ExporterType::Memory)
}

/// The live value of one exported string attribute.
///
/// The SDK appends on re-recording, so the *last* occurrence wins: that is the
/// value a caller would see through Python's `set_attribute`.
fn attribute(span: &SpanData, key: &str) -> Option<String> {
    span.attributes
        .iter()
        .rev()
        .find(|pair| pair.key.as_str() == key)
        .map(|pair| pair.value.as_str().into_owned())
}

/// The names of the events a span carries.
fn event_names(span: &SpanData) -> Vec<String> {
    span.events
        .iter()
        .map(|event| event.name.to_string())
        .collect()
}

#[test]
fn the_session_tracer_exports_one_chat_span() {
    // test_handler.py: test_llm_start_creates_span and
    // test_llm_end_records_response_attributes, driven through the session
    // `Tracer` instead of the handler callbacks.
    let config = plain_config().with_service("openjiuwen-extensions", Some("0.1.0".to_owned()));
    let handle = init_otel_tracer(config).expect("the memory exporter initializes");
    let exporter = handle.capture().expect("the memory exporter is attached");
    let sink: Arc<dyn TraceSink> =
        OtelTraceSink::new(&handle).expect("a live handle builds a sink");
    let tracer = Tracer::with_sink("session-1", "trace-1", sink);

    let span = tracer.start_span("Model", InvokeType::Llm);
    span.set_inputs(json!({"messages": [{"role": "user", "content": "hi"}]}));
    span.finish(json!({"content": "hello", "finish_reason": "stop"}));
    handle.force_flush();

    // The session tracer emits one terminal frame; the sink lazily opens and
    // closes exactly one span from it.
    assert_eq!(exporter.names(), vec!["chat Model".to_owned()]);
    let data = exporter
        .span_named("chat Model")
        .expect("the finished span is exported");
    assert_eq!(data.span_kind, SpanKind::Client);
    assert_eq!(data.status, Status::Ok);
    assert_eq!(
        attribute(&data, GEN_AI_OPERATION_NAME).as_deref(),
        Some(GEN_AI_OPERATION_CHAT)
    );
    assert_eq!(
        attribute(&data, GEN_AI_PROVIDER_NAME).as_deref(),
        Some(GEN_AI_PROVIDER_OPENJIUWEN)
    );
    assert_eq!(
        attribute(&data, OJ_STATUS).as_deref(),
        Some(SPAN_STATUS_FINISH)
    );
    assert_eq!(attribute(&data, OJ_TRACE_ID).as_deref(), Some("trace-1"));
    assert_eq!(
        attribute(&data, OJ_SESSION_ID).as_deref(),
        Some("session-1")
    );
    assert_eq!(attribute(&data, OJ_AGENT_NAME).as_deref(), Some("Model"));
    assert!(
        attribute(&data, GEN_AI_INPUT_MESSAGES).is_some_and(|text| text.contains("\"hi\"")),
        "the request messages are exported"
    );
    assert!(
        attribute(&data, GEN_AI_OUTPUT_MESSAGES).is_some_and(|text| text.contains("hello")),
        "the response messages are exported"
    );

    // The capture API also reports the resource the provider handed over.
    let resource = exporter
        .resource()
        .expect("the provider reports its resource");
    let service_name = resource.get(&Key::from_static_str("service.name"));
    assert_eq!(
        service_name.map(|value| value.as_str().into_owned()),
        Some("openjiuwen-extensions".to_owned())
    );
}

#[test]
fn a_failed_session_span_uses_the_default_error_code() {
    // test_handler.py: test_llm_error_marks_span_error, frame-level path. A
    // trace frame carries no exception object, so the failure keeps Python's
    // `OJStatusCode.WORKFLOW_EXECUTION_ERROR` code (`100102`) as `error.type`.
    let handle = init_otel_tracer(plain_config()).expect("the memory exporter initializes");
    let exporter = handle.capture().expect("the memory exporter is attached");
    let sink: Arc<dyn TraceSink> =
        OtelTraceSink::new(&handle).expect("a live handle builds a sink");
    let tracer = Tracer::with_sink("session-1", "trace-2", sink);

    let span = tracer.start_span("Model", InvokeType::Llm);
    span.fail("model exploded");
    handle.force_flush();

    let data = exporter
        .span_named("chat Model")
        .expect("the failed span is exported");
    assert!(matches!(data.status, Status::Error { .. }));
    assert_eq!(
        attribute(&data, OJ_STATUS).as_deref(),
        Some(SPAN_STATUS_ERROR)
    );
    assert_eq!(
        attribute(&data, OJ_AGENT_ERROR_MESSAGE).as_deref(),
        Some("model exploded")
    );
    assert_eq!(attribute(&data, ERROR_TYPE).as_deref(), Some("100102"));
    let payload = attribute(&data, OJ_ERROR).expect("the error payload is exported");
    assert!(payload.contains("\"error_code\":\"100102\""), "{payload}");
    assert!(payload.contains("model exploded"), "{payload}");
}

#[test]
fn the_agent_handler_dedups_the_tool_rail_and_its_decorator_twin() {
    // test_otel_tool_span_dedup.py: test_rail_and_decorator_emit_one_tool_span
    // (`bug #2046`). The handler is built over a caller-supplied
    // `CaptureExporter`, which is the documented injection seam.
    let capture = CaptureExporter::new();
    let handle = init_otel_tracer_with_exporter(plain_config(), capture.clone())
        .expect("the injected exporter initializes");
    let handler = OtelAgentHandler::from_handle(&handle).expect("the handle is active");

    let rail = TraceSpan::new("trace-1", "", "invoke-rail", "", TraceSpanKind::Agent);
    let rail_info = json!({"class_name": "get_weather", "type": "plugin"});
    block_on(handler.on_plugin_start(&rail, &json!({"city": "Beijing"}), &rail_info));
    assert!(
        handler.has_open_tool_span("get_weather", "trace-1"),
        "the rail span is flagged for the dedup"
    );

    let decorator = TraceSpan::new("trace-1", "", "invoke-decorator", "", TraceSpanKind::Agent);
    let decorator_info = json!({"class_name": "get_weather", "type": "tool"});
    block_on(handler.on_plugin_start(&decorator, &json!({"city": "Beijing"}), &decorator_info));
    // The decorator's nested end event is a no-op; the rail closes the span.
    block_on(handler.on_plugin_end(&decorator, &json!("22C")));
    block_on(handler.on_plugin_end(&rail, &json!("22C")));

    handle.force_flush();
    assert_eq!(
        capture.names(),
        ["execute_tool get_weather"],
        "exactly one span survives the two emitters"
    );
    assert!(!handler.has_open_tool_span("get_weather", "trace-1"));
    let span = capture
        .span_named("execute_tool get_weather")
        .expect("the tool span is exported");
    assert_eq!(span.span_kind, SpanKind::Internal);
    assert_eq!(
        attribute(&span, GEN_AI_TOOL_NAME).as_deref(),
        Some("get_weather")
    );
    assert_eq!(
        attribute(&span, OJ_STATUS).as_deref(),
        Some(SPAN_STATUS_FINISH)
    );
}

#[test]
fn an_llm_error_marks_the_span_error_through_the_handler() {
    // test_handler.py: test_agent_llm_error_marks_error — the typed-handler
    // error path, which carries the structured `Error` kind as `error.type`.
    let handle = init_otel_tracer(plain_config()).expect("the memory exporter initializes");
    let exporter = handle.capture().expect("the memory exporter is attached");
    let handler = OtelAgentHandler::from_handle(&handle).expect("the handle is active");
    let frame = TraceSpan::new("trace-1", "", "invoke-1", "", TraceSpanKind::Agent);

    block_on(handler.on_llm_start(&frame, &json!("hello"), &json!({"class_name": "TestModel"})));
    assert!(
        exporter.is_empty(),
        "the span only closes on the error callback"
    );
    block_on(handler.on_llm_error(&frame, &Error::tool("test error")));
    handle.force_flush();

    let span = exporter
        .span_named("chat TestModel")
        .expect("the failed LLM span is exported");
    assert_eq!(
        span.status,
        Status::Error {
            description: "test error".into()
        }
    );
    assert_eq!(attribute(&span, ERROR_TYPE).as_deref(), Some("tool"));
    assert_eq!(
        attribute(&span, OJ_STATUS).as_deref(),
        Some(SPAN_STATUS_ERROR)
    );
    assert_eq!(
        attribute(&span, OJ_AGENT_ERROR_MESSAGE).as_deref(),
        Some("test error")
    );
    assert_eq!(
        attribute(&span, OJ_ERROR).as_deref(),
        Some(r#"{"error_code":"tool","message":"test error"}"#)
    );
    assert!(event_names(&span).contains(&"exception".to_owned()));
}
