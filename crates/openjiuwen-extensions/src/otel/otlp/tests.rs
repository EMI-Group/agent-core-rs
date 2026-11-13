//! Unit tests of the OTLP/HTTP JSON exporter.
//!
//! Port of the trace-export assertions of
//! `tests/unit_tests/extensions/tracer_otel/test_setup.py` (exporter selection)
//! and of the OTLP payload checks the Python suite leaves to the SDK: here the
//! JSON body is asserted against the OTLP specification field by field.
//!
//! The transport tests drive `export()` with `futures::executor::block_on` from
//! plain `#[test]` functions. That is deliberate: the SDK's batch processor runs
//! the export future on a thread with no tokio reactor, so any reactor
//! dependency in the exporter would hang or panic here.
// A test module owns its panics: a failed `unwrap`/`expect` is an assertion.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use futures::executor::block_on;
use openjiuwen_common::error::ErrorKind;
use opentelemetry::trace::{
    Link, Span as _, SpanContext, SpanId, SpanKind, Status, TraceContextExt as _, TraceFlags,
    TraceId, TraceState, Tracer as _, TracerProvider as _,
};
use opentelemetry::Value as OtelValue;
use opentelemetry::{Array, Context, InstrumentationScope, KeyValue, StringValue};
use opentelemetry_sdk::error::OTelSdkError;
use opentelemetry_sdk::trace::{SdkTracerProvider, SpanData, SpanExporter as _};
use opentelemetry_sdk::Resource;
use serde_json::{json, Value};
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use super::json;
use super::{traces_url, OtlpHttpExporter};
use crate::otel::provider::CaptureExporter;

/// The resource the JSON assertions expect.
fn test_resource() -> Resource {
    Resource::builder_empty()
        .with_attributes([
            KeyValue::new("service.name", "otlp-test"),
            KeyValue::new("service.version", "1.0.0"),
        ])
        .build()
}

/// A valid remote span context for the span-link assertion.
fn link_context() -> SpanContext {
    SpanContext::new(
        TraceId::from_bytes([0x11; 16]),
        SpanId::from_bytes([0x22; 8]),
        TraceFlags::SAMPLED,
        true,
        TraceState::NONE,
    )
}

/// A provider that exports synchronously into a [`CaptureExporter`].
fn capture_provider() -> (SdkTracerProvider, CaptureExporter) {
    let capture = CaptureExporter::new();
    let provider = SdkTracerProvider::builder()
        .with_simple_exporter(capture.clone())
        .build();
    (provider, capture)
}

/// One finished `client {name}` span, captured through a real provider.
fn captured_span(name: &str) -> SpanData {
    let (provider, capture) = capture_provider();
    let tracer = provider.tracer("openjiuwen.test");
    let mut span = tracer
        .span_builder(name.to_owned())
        .with_kind(SpanKind::Client)
        .start(&tracer);
    span.set_status(Status::Ok);
    span.end();
    let _ = provider.force_flush();
    let mut spans = capture.spans();
    assert_eq!(spans.len(), 1);
    spans.remove(0)
}

/// Nanoseconds since the Unix epoch, the expected `*TimeUnixNano` value.
fn nanos(time: SystemTime) -> String {
    time.duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos()
        .to_string()
}

/// The `value` object of one attribute in an OTLP attribute array.
fn attribute<'a>(attributes: &'a Value, key: &str) -> &'a Value {
    attributes
        .as_array()
        .expect("attributes must be an array")
        .iter()
        .find(|pair| pair["key"] == json!(key))
        .map_or_else(
            || panic!("attribute {key} is missing from {attributes}"),
            |pair| &pair["value"],
        )
}

/// The sorted key set of a JSON object, so absent fields are asserted too.
fn sorted_keys(value: &Value) -> Vec<String> {
    let mut keys: Vec<String> = value
        .as_object()
        .expect("expected a JSON object")
        .keys()
        .cloned()
        .collect();
    keys.sort_unstable();
    keys
}

// port of tests/unit_tests/extensions/tracer_otel/test_setup.py (endpoint form)

#[test]
fn traces_url_appends_the_suffix_once() {
    assert_eq!(
        traces_url("http://localhost:4318"),
        "http://localhost:4318/v1/traces"
    );
    assert_eq!(
        traces_url("http://localhost:4318/"),
        "http://localhost:4318/v1/traces"
    );
    assert_eq!(
        traces_url("http://localhost:4318/v1/traces"),
        "http://localhost:4318/v1/traces"
    );
    assert_eq!(
        traces_url("https://collector.test/otlp/v1/traces/"),
        "https://collector.test/otlp/v1/traces"
    );
    assert_eq!(
        traces_url("https://collector.test/otlp"),
        "https://collector.test/otlp/v1/traces"
    );
}

#[test]
fn new_normalizes_the_endpoint_and_keeps_the_headers_private() {
    let exporter = OtlpHttpExporter::new(
        "http://localhost:4318/",
        BTreeMap::from([(String::from("authorization"), String::from("Bearer tok"))]),
        Duration::from_secs(5),
    )
    .unwrap();
    assert_eq!(exporter.traces_url(), "http://localhost:4318/v1/traces");
    // The header *value* must not leak through `Debug`.
    let debug = format!("{exporter:?}");
    assert!(debug.contains("authorization"), "{debug}");
    assert!(!debug.contains("Bearer tok"), "{debug}");
    assert!(debug.contains("shutdown: false"), "{debug}");
}

#[test]
fn new_rejects_invalid_endpoints_and_headers() {
    let outcome = OtlpHttpExporter::new("", BTreeMap::new(), Duration::from_secs(1));
    let error = outcome.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert_eq!(
        error.to_string(),
        "validation: the OTLP HTTP exporter requires a non-empty endpoint"
    );

    for endpoint in ["localhost:4318", "ftp://localhost:4318", "/v1/traces"] {
        let error =
            OtlpHttpExporter::new(endpoint, BTreeMap::new(), Duration::from_secs(1)).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation, "{endpoint}");
    }

    let bad_name = OtlpHttpExporter::new(
        "http://localhost:4318",
        BTreeMap::from([(String::from("bad header"), String::from("x"))]),
        Duration::from_secs(1),
    )
    .unwrap_err();
    assert_eq!(bad_name.kind(), ErrorKind::Validation);

    let bad_value = OtlpHttpExporter::new(
        "http://localhost:4318",
        BTreeMap::from([(String::from("x-token"), String::from("line\nbreak"))]),
        Duration::from_secs(1),
    )
    .unwrap_err();
    assert_eq!(bad_value.kind(), ErrorKind::Validation);
}

#[test]
fn exporter_is_send_sync_and_debug() {
    fn assert_traits<T: Send + Sync + std::fmt::Debug>() {}
    assert_traits::<OtlpHttpExporter>();
}

// port of the OTLP payload expectations of test_setup.py (span encoding)

/// The fixed start instant of the fixture span.
fn fixture_start() -> SystemTime {
    UNIX_EPOCH + Duration::from_secs(1_700_000_000) + Duration::from_nanos(123)
}

/// The fixed end instant of the fixture span.
fn fixture_end() -> SystemTime {
    fixture_start() + Duration::from_millis(7)
}

/// Builds the shared fixture: four spans across two instrumentation scopes,
/// encoded with the test resource.
///
/// Returns the encoded body and the first span's context, which is the child
/// span's parent.
fn otlp_body() -> (Value, SpanContext) {
    let (provider, capture) = capture_provider();
    let tracer = provider.tracer("openjiuwen.test");
    let worker_scope = InstrumentationScope::builder("openjiuwen.worker")
        .with_version("2.0.0")
        .with_attributes([KeyValue::new("scope.attr", "x")])
        .build();
    let worker = provider.tracer_with_scope(worker_scope);

    let start = fixture_start();
    let end = fixture_end();
    let mut first = tracer
        .span_builder("client.call")
        .with_kind(SpanKind::Client)
        .with_start_time(start)
        .with_end_time(end)
        .with_attributes([
            KeyValue::new("test.string", "value"),
            KeyValue::new("test.int", 42_i64),
            KeyValue::new("test.double", 1.5_f64),
            KeyValue::new("test.bool", true),
            KeyValue::new(
                "tags",
                OtelValue::Array(Array::String(vec![
                    StringValue::from("a"),
                    StringValue::from("b"),
                ])),
            ),
        ])
        .with_links(vec![Link::new(
            link_context(),
            vec![KeyValue::new("link.reason", "retry")],
            0,
        )])
        .start(&tracer);
    let first_context = first.span_context().clone();
    first.add_event("retry.scheduled", vec![KeyValue::new("attempt", 1_i64)]);
    first.set_status(Status::error("upstream returned 500"));
    // `end()` would stamp the wall clock; the builder's `end_time` is kept only
    // by ending with the same instant explicitly.
    first.end_with_timestamp(end);

    let mut second = tracer
        .span_builder("client.second")
        .with_kind(SpanKind::Server)
        .start(&tracer);
    second.set_status(Status::Ok);
    second.end();

    // A child of the first span: proves `parentSpanId` is emitted only when set.
    let child_context = Context::new().with_remote_span_context(first_context.clone());
    let mut child = tracer.start_with_context("client.child", &child_context);
    child.end();

    let mut worker_span = worker.start("worker.step");
    worker_span.end();

    let _ = provider.force_flush();
    let resource = test_resource();
    let body = json::encode_body(Some(&resource), &capture.spans());
    (body, first_context)
}

#[test]
fn json_body_groups_scopes_and_encodes_ids_kinds_and_timestamps() {
    let (body, first_context) = otlp_body();

    // One `resourceSpans` entry carrying the resource the provider handed over.
    let resource_spans = body["resourceSpans"].as_array().unwrap();
    assert_eq!(resource_spans.len(), 1);
    let entry = &resource_spans[0];
    let resource_attributes = &entry["resource"]["attributes"];
    assert_eq!(
        attribute(resource_attributes, "service.name"),
        &json!({"stringValue": "otlp-test"})
    );
    assert_eq!(
        attribute(resource_attributes, "service.version"),
        &json!({"stringValue": "1.0.0"})
    );

    // One `scopeSpans` entry per distinct (name, version), first-seen order.
    let scopes = entry["scopeSpans"].as_array().unwrap();
    assert_eq!(scopes.len(), 2);
    assert_eq!(scopes[0]["scope"], json!({"name": "openjiuwen.test"}));
    let spans = scopes[0]["spans"].as_array().unwrap();
    assert_eq!(spans.len(), 3);
    assert_eq!(spans[0]["name"], json!("client.call"));
    assert_eq!(spans[1]["name"], json!("client.second"));
    assert_eq!(spans[2]["name"], json!("client.child"));

    // Ids: 32/16 lowercase hex characters.
    let trace_id = spans[0]["traceId"].as_str().unwrap();
    assert_eq!(trace_id.len(), 32);
    assert!(trace_id
        .chars()
        .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
    assert_eq!(
        trace_id,
        super::json::hex(&first_context.trace_id().to_bytes())
    );
    let span_id = spans[0]["spanId"].as_str().unwrap();
    assert_eq!(span_id.len(), 16);
    assert_eq!(
        span_id,
        super::json::hex(&first_context.span_id().to_bytes())
    );

    // The root span has no parent; the child points at it.
    assert!(!spans[0].as_object().unwrap().contains_key("parentSpanId"));
    assert_eq!(
        spans[2]["parentSpanId"],
        json!(super::json::hex(&first_context.span_id().to_bytes()))
    );
    assert_eq!(spans[2]["traceId"], json!(trace_id));

    // Kinds, names, timestamps.
    assert_eq!(spans[0]["kind"], json!("SPAN_KIND_CLIENT"));
    assert_eq!(spans[1]["kind"], json!("SPAN_KIND_SERVER"));
    assert_eq!(spans[0]["startTimeUnixNano"], json!(nanos(fixture_start())));
    assert_eq!(spans[0]["endTimeUnixNano"], json!(nanos(fixture_end())));
    assert_eq!(spans[2]["kind"], json!("SPAN_KIND_INTERNAL"));

    // Every span field the exporter writes, and nothing else.
    assert_eq!(
        sorted_keys(&spans[0]),
        vec![
            "attributes",
            "droppedAttributesCount",
            "endTimeUnixNano",
            "events",
            "kind",
            "links",
            "name",
            "spanId",
            "startTimeUnixNano",
            "status",
            "traceId",
        ]
    );
    assert_eq!(spans[0]["droppedAttributesCount"], json!(0));
}

#[test]
fn json_body_encodes_attributes_events_links_and_status() {
    let (body, _) = otlp_body();
    let scopes = body["resourceSpans"][0]["scopeSpans"].as_array().unwrap();
    let spans = scopes[0]["spans"].as_array().unwrap();

    // Attribute value encodings.
    let attributes = &spans[0]["attributes"];
    assert_eq!(
        attribute(attributes, "test.string"),
        &json!({"stringValue": "value"})
    );
    assert_eq!(
        attribute(attributes, "test.int"),
        &json!({"intValue": "42"})
    );
    assert_eq!(
        attribute(attributes, "test.double"),
        &json!({"doubleValue": 1.5})
    );
    assert_eq!(
        attribute(attributes, "test.bool"),
        &json!({"boolValue": true})
    );
    assert_eq!(
        attribute(attributes, "tags"),
        &json!({"arrayValue": {"values": [{"stringValue": "a"}, {"stringValue": "b"}]}})
    );

    // One event with its timestamp string and attributes.
    let events = spans[0]["events"].as_array().unwrap();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0]["name"], json!("retry.scheduled"));
    assert!(events[0]["timeUnixNano"].is_string());
    assert_eq!(
        attribute(&events[0]["attributes"], "attempt"),
        &json!({"intValue": "1"})
    );
    assert_eq!(events[0]["droppedAttributesCount"], json!(0));

    // One link with the peer ids and attributes.
    let links = spans[0]["links"].as_array().unwrap();
    assert_eq!(links.len(), 1);
    assert_eq!(
        links[0]["traceId"],
        json!("11111111111111111111111111111111")
    );
    assert_eq!(links[0]["spanId"], json!("2222222222222222"));
    assert_eq!(
        attribute(&links[0]["attributes"], "link.reason"),
        &json!({"stringValue": "retry"})
    );

    // Status: the error description becomes the message.
    assert_eq!(
        spans[0]["status"],
        json!({"code": "STATUS_CODE_ERROR", "message": "upstream returned 500"})
    );
    assert_eq!(spans[1]["status"], json!({"code": "STATUS_CODE_OK"}));
    assert_eq!(spans[0]["events"].as_array().unwrap().len(), 1);
    assert_eq!(spans[1]["events"].as_array().unwrap().len(), 0);
    assert_eq!(spans[1]["links"].as_array().unwrap().len(), 0);

    // The second scope carries its version and scope attributes.
    assert_eq!(
        scopes[1]["scope"],
        json!({
            "name": "openjiuwen.worker",
            "version": "2.0.0",
            "attributes": [{"key": "scope.attr", "value": {"stringValue": "x"}}],
        })
    );
    assert_eq!(scopes[1]["spans"].as_array().unwrap().len(), 1);
    assert_eq!(scopes[1]["spans"][0]["name"], json!("worker.step"));
}

#[test]
fn json_body_without_a_resource_stays_valid() {
    let span = captured_span("client.orphan");
    let body = json::encode_body(None, &[span]);
    let entry = &body["resourceSpans"][0];
    assert_eq!(entry["resource"], json!({}));
    assert_eq!(
        entry["scopeSpans"][0]["spans"][0]["name"],
        json!("client.orphan")
    );
    assert_eq!(
        entry["scopeSpans"][0]["spans"][0]["status"],
        json!({"code": "STATUS_CODE_OK"})
    );
}

// transport end-to-end, driven from a plain `#[test]`

#[test]
fn export_posts_the_json_body_to_the_traces_path() {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let server = runtime.block_on(MockServer::builder().start());
    runtime.block_on(
        Mock::given(method("POST"))
            .and(path("/v1/traces"))
            .and(header("content-type", "application/json"))
            .and(header("authorization", "Bearer token-123"))
            .respond_with(ResponseTemplate::new(200))
            .mount(&server),
    );

    let mut exporter = OtlpHttpExporter::new(
        server.uri(),
        BTreeMap::from([(
            String::from("authorization"),
            String::from("Bearer token-123"),
        )]),
        Duration::from_secs(5),
    )
    .unwrap();
    exporter.set_resource(&test_resource());

    // `block_on` here runs on a plain test thread: no tokio reactor is entered.
    let outcome = block_on(exporter.export(vec![captured_span("client.call")]));
    assert!(outcome.is_ok(), "{outcome:?}");

    let requests = runtime.block_on(server.received_requests()).unwrap();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].method.as_str(), "POST");
    assert_eq!(requests[0].url.path(), "/v1/traces");
    assert_eq!(
        requests[0]
            .headers
            .get("content-type")
            .unwrap()
            .to_str()
            .unwrap(),
        "application/json"
    );
    assert_eq!(
        requests[0]
            .headers
            .get("authorization")
            .unwrap()
            .to_str()
            .unwrap(),
        "Bearer token-123"
    );
    let body: Value = serde_json::from_slice(&requests[0].body).unwrap();
    assert_eq!(
        body["resourceSpans"][0]["scopeSpans"][0]["spans"][0]["name"],
        json!("client.call")
    );
}

#[test]
fn non_2xx_responses_fail_without_panicking() {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let server = runtime.block_on(MockServer::builder().start());
    runtime.block_on(
        Mock::given(method("POST"))
            .and(path("/v1/traces"))
            .respond_with(ResponseTemplate::new(500).set_body_string("collector exploded"))
            .mount(&server),
    );

    let exporter =
        OtlpHttpExporter::new(server.uri(), BTreeMap::new(), Duration::from_secs(5)).unwrap();
    let outcome = block_on(exporter.export(vec![captured_span("client.call")]));
    match outcome {
        Err(OTelSdkError::InternalFailure(message)) => {
            assert!(message.contains("500"), "{message}");
            assert!(message.contains("collector exploded"), "{message}");
        }
        other => panic!("expected an internal failure, got {other:?}"),
    }
}

#[test]
fn a_slow_collector_times_out_within_the_client_budget() {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let server = runtime.block_on(MockServer::builder().start());
    runtime.block_on(
        Mock::given(method("POST"))
            .and(path("/v1/traces"))
            .respond_with(ResponseTemplate::new(200).set_delay(Duration::from_millis(600)))
            .mount(&server),
    );

    let mut exporter =
        OtlpHttpExporter::new(server.uri(), BTreeMap::new(), Duration::from_millis(200)).unwrap();
    exporter.set_resource(&test_resource());

    let started = Instant::now();
    let outcome = block_on(exporter.export(vec![captured_span("client.call")]));
    let elapsed = started.elapsed();
    match outcome {
        Err(OTelSdkError::InternalFailure(message)) => {
            assert!(message.contains("timed out"), "{message}");
        }
        other => panic!("expected an internal failure, got {other:?}"),
    }
    assert!(elapsed < Duration::from_secs(5), "took {elapsed:?}");
}

#[test]
fn resource_shutdown_and_post_shutdown_export() {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let server = runtime.block_on(MockServer::builder().start());
    runtime.block_on(
        Mock::given(method("POST"))
            .and(path("/v1/traces"))
            .respond_with(ResponseTemplate::new(200))
            .mount(&server),
    );

    let span = captured_span("client.call");
    let mut exporter =
        OtlpHttpExporter::new(server.uri(), BTreeMap::new(), Duration::from_secs(5)).unwrap();

    // Without `set_resource` the body carries an empty resource object.
    let outcome = block_on(exporter.export(vec![span.clone()]));
    assert!(outcome.is_ok(), "{outcome:?}");

    exporter.set_resource(&test_resource());
    let outcome = block_on(exporter.export(vec![span.clone()]));
    assert!(outcome.is_ok(), "{outcome:?}");

    let requests = runtime.block_on(server.received_requests()).unwrap();
    assert_eq!(requests.len(), 2);
    let without_resource: Value = serde_json::from_slice(&requests[0].body).unwrap();
    assert_eq!(without_resource["resourceSpans"][0]["resource"], json!({}));
    let with_resource: Value = serde_json::from_slice(&requests[1].body).unwrap();
    assert_eq!(
        attribute(
            &with_resource["resourceSpans"][0]["resource"]["attributes"],
            "service.name"
        ),
        &json!({"stringValue": "otlp-test"})
    );

    // Shutdown is idempotent.
    assert!(exporter
        .shutdown_with_timeout(Duration::from_secs(5))
        .is_ok());
    assert!(exporter
        .shutdown_with_timeout(Duration::from_secs(5))
        .is_ok());
    assert!(format!("{exporter:?}").contains("shutdown: true"));

    // Exporting after shutdown fails instead of panicking.
    match block_on(exporter.export(vec![span])) {
        Err(OTelSdkError::InternalFailure(message)) => {
            assert!(message.contains("shut down"), "{message}");
        }
        other => panic!("expected an internal failure, got {other:?}"),
    }
}
