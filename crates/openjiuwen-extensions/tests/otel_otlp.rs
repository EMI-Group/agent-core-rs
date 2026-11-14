//! Crate-boundary tests of the OTLP/HTTP span-export path.
//!
//! Python counterpart: `tests/unit_tests/extensions/tracer_otel/test_setup.py`
//! — the exporter selection of `OtelTracer.setup` plus the collector wiring
//! that Python hands to `OTLPSpanExporter`. Python's suite stops at the SDK
//! boundary; here the public surface — `init_otel_tracer`, `TracerHandle`,
//! `OtlpHttpExporter`, `OtlpProtocol`, and `env` — is driven end to end and the
//! *wire payload* is asserted, because the batch is posted to a loopback
//! `wiremock` collector rather than to a real one.
//!
//! Both OTLP/HTTP encodings are supported: `http/json`
//! (`application/json`) and `http/protobuf` (`application/x-protobuf`). This
//! file covers the JSON payload end to end, the transport selection at both the
//! environment and the serde level, and the handle wiring; the protobuf body
//! itself is decoded field by field in `tests/otel_otlp_protobuf.rs`.
//!
//! What the file pins:
//!
//! - one finished batch reaches the collector as a single `POST /v1/traces`
//!   with `content-type: application/json`, the configured headers, the
//!   `service.name`/`service.version` resource attributes, the OTLP/JSON span
//!   fields (hex ids, kinds, decimal nanosecond timestamps, typed attribute
//!   values, events, links, error status), and an idempotent shutdown;
//! - the environment precedence (`OPENJIUWEN_OTEL_*` before `OTEL_*`, the
//!   trace-specific endpoint before the generic one, explicit values first) at
//!   both the end-to-end and the pure-function level, for `http/json` and
//!   `http/protobuf` alike;
//! - every protocol spelling of `OtlpProtocol` deserializes to the right
//!   transport and serializes back to the canonical `snake_case` name;
//! - every failure mode is typed (`InternalFailure`) and bounded in time, with
//!   no tokio reactor required on the caller's thread;
//! - OTLP/gRPC stays `Error::Unsupported` instead of being downgraded, while
//!   both HTTP encodings build an active provider.
//!
//! Everything is offline and deterministic: the only endpoint is `127.0.0.1`,
//! environment mutation is serialized through [`ENV_LOCK`] and undone on drop.

#![cfg(feature = "otel")]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::collections::BTreeMap;
use std::sync::{Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

use futures::executor::block_on;
use opentelemetry::trace::{
    Link, Span as _, SpanContext, SpanKind, Status, TraceContextExt as _, Tracer as _,
};
use opentelemetry::{Array, Context, KeyValue, StringValue, Value as OtelValue};
use opentelemetry_sdk::error::OTelSdkError;
use opentelemetry_sdk::trace::SpanExporter as _;
use serde_json::{json, Value};
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use openjiuwen_common::error::ErrorKind;
use openjiuwen_extensions::otel::env;
use openjiuwen_extensions::otel::{
    init_otel_tracer, ExporterType, OtelTracerConfig, OtlpEncoding, OtlpHttpExporter, OtlpProtocol,
    TracerHandle,
};

/// Serialises the tests that read or write the OTLP process environment.
static ENV_LOCK: Mutex<()> = Mutex::new(());

/// Every variable the OTLP path resolves, plus the SDK resource variables.
const ENV_VARS: [&str; 11] = [
    env::OPENJIUWEN_OTEL_EXPORTER_OTLP_ENDPOINT,
    env::OPENJIUWEN_OTEL_EXPORTER_OTLP_HEADERS,
    env::OPENJIUWEN_OTEL_EXPORTER_OTLP_PROTOCOL,
    env::OPENJIUWEN_OTEL_SERVICE_NAME,
    env::OTEL_EXPORTER_OTLP_TRACES_ENDPOINT,
    env::OTEL_EXPORTER_OTLP_ENDPOINT,
    env::OTEL_EXPORTER_OTLP_HEADERS,
    env::OTEL_EXPORTER_OTLP_PROTOCOL,
    env::OTEL_SERVICE_NAME,
    "OTEL_RESOURCE_ATTRIBUTES",
    "OTEL_SDK_DISABLED",
];

/// Holds [`ENV_LOCK`] for one test and keeps the OTLP variables clean.
struct EnvGuard {
    /// The process-global lock; released when the guard drops.
    _lock: MutexGuard<'static, ()>,
}

impl EnvGuard {
    /// Acquires the lock, recovering it when another test panicked.
    fn new() -> Self {
        let lock = ENV_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
        clear_env();
        Self { _lock: lock }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        clear_env();
    }
}

/// Removes every variable the OTLP path resolves.
fn clear_env() {
    for name in ENV_VARS {
        std::env::remove_var(name);
    }
}

/// Sets one variable for the rest of the test.
fn set_env(name: &str, value: &str) {
    std::env::set_var(name, value);
}

/// Mounts the success response every span-post test expects exactly once.
fn mount_traces(runtime: &tokio::runtime::Runtime, server: &MockServer) {
    runtime.block_on(
        Mock::given(method("POST"))
            .and(path("/v1/traces"))
            .respond_with(ResponseTemplate::new(200))
            .mount(server),
    );
}

/// The OTLP/HTTP configuration the end-to-end test posts with.
fn otlp_config(server: &MockServer) -> OtelTracerConfig {
    OtelTracerConfig {
        exporter_type: ExporterType::Otlp,
        protocol: OtlpProtocol::Http,
        exporter_endpoint: Some(server.uri()),
        headers: BTreeMap::from([("authorization".to_owned(), "Bearer test-token".to_owned())]),
        service_name: "openjiuwen-otlp-tests".to_owned(),
        service_version: Some("9.9.9".to_owned()),
        redaction_enabled: false,
        export_timeout_ms: 5_000,
        ..OtelTracerConfig::default()
    }
}

/// What the fixture emitted, for the id assertions.
struct Emitted {
    /// Context of the root span, which the child is nested in and links to.
    root: SpanContext,
}

/// Starts the fixture spans on `handle`'s tracer: a client root with typed
/// attributes and one event, a server child with a link to the root, and one
/// span closed with an error status.
fn emit_spans(handle: &TracerHandle) -> Emitted {
    let tracer = handle.tracer().expect("an active handle exposes a tracer");

    let mut root = tracer
        .span_builder("otlp.root")
        .with_kind(SpanKind::Client)
        .with_attributes([
            KeyValue::new("test.string", "value"),
            KeyValue::new("test.int", 42_i64),
            KeyValue::new("test.double", 1.5_f64),
            KeyValue::new("test.bool", true),
            KeyValue::new(
                "test.tags",
                OtelValue::Array(Array::String(vec![
                    StringValue::from("a"),
                    StringValue::from("b"),
                ])),
            ),
        ])
        .start(&tracer);
    let root_context = root.span_context().clone();
    root.add_event(
        "tool.result",
        vec![KeyValue::new("tool.name", "get_weather")],
    );
    root.end();

    // A real child (parentSpanId) that also carries a link to its parent.
    let parent = Context::new().with_remote_span_context(root_context.clone());
    let link = Link::new(
        root_context.clone(),
        vec![KeyValue::new("link.reason", "follow-up")],
        0,
    );
    let mut child = tracer
        .span_builder("otlp.child")
        .with_kind(SpanKind::Server)
        .with_links(vec![link])
        .start_with_context(&tracer, &parent);
    child.end();

    let mut failed = tracer.span_builder("otlp.failed").start(&tracer);
    failed.set_status(Status::error("boom"));
    failed.end();

    Emitted { root: root_context }
}

/// Encodes bytes as lowercase hexadecimal, the OTLP id encoding.
fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(encoded, "{byte:02x}");
    }
    encoded
}

/// The spans of the single `scopeSpans` entry the batch produced.
fn spans_of(body: &Value) -> &Vec<Value> {
    body["resourceSpans"][0]["scopeSpans"][0]["spans"]
        .as_array()
        .unwrap_or_else(|| panic!("the body carries no spans: {body}"))
}

/// One span of the batch, by name.
fn span_named<'a>(spans: &'a [Value], name: &str) -> &'a Value {
    spans
        .iter()
        .find(|span| span["name"] == json!(name))
        .unwrap_or_else(|| panic!("no span named {name} in {spans:?}"))
}

/// The `value` object of one attribute in a bare attribute array.
fn attribute_in<'a>(attributes: &'a Value, key: &str) -> &'a Value {
    attributes
        .as_array()
        .unwrap_or_else(|| panic!("not an attribute array: {attributes}"))
        .iter()
        .find(|pair| pair["key"] == json!(key))
        .map_or_else(
            || panic!("attribute {key} is missing from {attributes}"),
            |pair| &pair["value"],
        )
}

/// The `value` object of one attribute of a span.
fn attribute<'a>(span: &'a Value, key: &str) -> &'a Value {
    attribute_in(&span["attributes"], key)
}

/// Asserts a decimal-string timestamp of the Unix-nanosecond magnitude.
fn assert_unix_nanos(value: &Value) {
    let text = value
        .as_str()
        .unwrap_or_else(|| panic!("{value} is not a decimal string"));
    assert!(
        text.chars().all(|character| character.is_ascii_digit()),
        "{text} is not all digits"
    );
    let nanos: u128 = text.parse().expect("a decimal timestamp parses");
    assert!(
        nanos > 1_000_000_000_000_000_000,
        "{text} is not Unix nanoseconds"
    );
}

/// Asserts the resource attributes the configuration asked for.
///
/// The SDK's own resource detectors add telemetry defaults on top, so the
/// assertions are containment checks, never an exact array comparison.
fn assert_resource(body: &Value) {
    let attributes = &body["resourceSpans"][0]["resource"]["attributes"];
    assert_eq!(
        attribute_in(attributes, "service.name"),
        &json!({"stringValue": "openjiuwen-otlp-tests"}),
    );
    assert_eq!(
        attribute_in(attributes, "service.version"),
        &json!({"stringValue": "9.9.9"}),
    );
}

/// Asserts that one instrumentation scope groups the whole batch.
fn assert_scope_grouping(body: &Value) {
    let scopes = body["resourceSpans"][0]["scopeSpans"]
        .as_array()
        .expect("scopeSpans is an array");
    assert_eq!(scopes.len(), 1, "one scope groups the batch");
    assert_eq!(scopes[0]["scope"]["name"], json!("openjiuwen.tracer.otel"));
    assert_eq!(spans_of(body).len(), 3, "three spans in the batch");
}

/// Asserts the root span: ids, kind, timestamps, typed attributes, status.
fn assert_root_span(body: &Value, root: &SpanContext) {
    let span = span_named(spans_of(body), "otlp.root");
    assert_eq!(span["traceId"], json!(hex(&root.trace_id().to_bytes())));
    assert_eq!(span["spanId"], json!(hex(&root.span_id().to_bytes())));
    assert_eq!(
        span["traceId"].as_str().expect("a trace id string").len(),
        32,
    );
    assert_eq!(span["spanId"].as_str().expect("a span id string").len(), 16);
    assert!(
        span.as_object()
            .expect("a span object")
            .get("parentSpanId")
            .is_none(),
        "the root span has no parent: {span}",
    );
    assert_eq!(span["kind"], json!("SPAN_KIND_CLIENT"));
    assert_unix_nanos(&span["startTimeUnixNano"]);
    assert_unix_nanos(&span["endTimeUnixNano"]);
    assert!(
        span["startTimeUnixNano"].as_str() <= span["endTimeUnixNano"].as_str(),
        "the span must not end before it starts: {span}",
    );

    // One attribute per OTLP `AnyValue` variant the SDK can produce here.
    assert_eq!(
        attribute(span, "test.string"),
        &json!({"stringValue": "value"}),
    );
    assert_eq!(attribute(span, "test.int"), &json!({"intValue": "42"}));
    assert_eq!(attribute(span, "test.double"), &json!({"doubleValue": 1.5}),);
    assert_eq!(attribute(span, "test.bool"), &json!({"boolValue": true}));
    assert_eq!(
        attribute(span, "test.tags"),
        &json!({"arrayValue": {"values": [
            {"stringValue": "a"},
            {"stringValue": "b"},
        ]}}),
    );
    assert_eq!(span["status"], json!({"code": "STATUS_CODE_UNSET"}));
}

/// Asserts the event the root span recorded.
fn assert_root_event(body: &Value) {
    let span = span_named(spans_of(body), "otlp.root");
    let events = span["events"].as_array().expect("events is an array");
    assert_eq!(events.len(), 1, "one event: {events:?}");
    assert_eq!(events[0]["name"], json!("tool.result"));
    assert_unix_nanos(&events[0]["timeUnixNano"]);
    assert_eq!(
        attribute_in(&events[0]["attributes"], "tool.name"),
        &json!({"stringValue": "get_weather"}),
    );
}

/// Asserts the child span: parent id, kind, and the link to its parent.
fn assert_child_span(body: &Value, root: &SpanContext) {
    let span = span_named(spans_of(body), "otlp.child");
    assert_eq!(span["traceId"], json!(hex(&root.trace_id().to_bytes())));
    assert_eq!(
        span["parentSpanId"],
        json!(hex(&root.span_id().to_bytes())),
        "the child points at the root",
    );
    assert_ne!(span["spanId"], json!(hex(&root.span_id().to_bytes())));
    assert_eq!(span["kind"], json!("SPAN_KIND_SERVER"));
    assert_unix_nanos(&span["startTimeUnixNano"]);

    let links = span["links"].as_array().expect("links is an array");
    assert_eq!(links.len(), 1, "one link: {links:?}");
    assert_eq!(links[0]["traceId"], json!(hex(&root.trace_id().to_bytes())));
    assert_eq!(links[0]["spanId"], json!(hex(&root.span_id().to_bytes())));
    assert_eq!(
        links[0]["traceId"].as_str().expect("a link trace id").len(),
        32,
    );
    assert_eq!(
        links[0]["spanId"].as_str().expect("a link span id").len(),
        16
    );
    assert_eq!(
        attribute_in(&links[0]["attributes"], "link.reason"),
        &json!({"stringValue": "follow-up"}),
    );
}

/// Asserts the span closed with an error status.
fn assert_failed_span(body: &Value) {
    let span = span_named(spans_of(body), "otlp.failed");
    assert_eq!(span["kind"], json!("SPAN_KIND_INTERNAL"));
    assert_eq!(
        span["status"],
        json!({"code": "STATUS_CODE_ERROR", "message": "boom"}),
    );
}

/// One `SpanData` captured through a memory provider, for the exporter tests.
fn captured_span() -> opentelemetry_sdk::trace::SpanData {
    let handle =
        init_otel_tracer(OtelTracerConfig::default().with_exporter_type(ExporterType::Memory))
            .expect("the memory exporter initializes");
    handle
        .tracer()
        .expect("an active handle exposes a tracer")
        .start("exporter.fixture")
        .end();
    handle.force_flush();
    let capture = handle.capture().expect("the memory exporter is attached");
    let mut spans = capture.spans();
    assert_eq!(spans.len(), 1, "the fixture span is captured");
    spans.remove(0)
}

#[test]
fn the_handle_posts_one_otlp_json_batch_to_the_configured_collector() {
    // test_setup.py: the exporter selection and the OTLP/JSON payload the
    // Python suite delegates to the SDK.
    let _environment = EnvGuard::new();
    let runtime = tokio::runtime::Runtime::new().expect("a tokio runtime starts");
    let server = runtime.block_on(MockServer::builder().start());
    runtime.block_on(
        Mock::given(method("POST"))
            .and(path("/v1/traces"))
            .and(header("content-type", "application/json"))
            .and(header("authorization", "Bearer test-token"))
            .respond_with(ResponseTemplate::new(200))
            .expect(1)
            .mount(&server),
    );

    let handle = init_otel_tracer(otlp_config(&server)).expect("OTLP/HTTP initializes offline");
    assert!(handle.is_active());
    assert!(
        handle.capture().is_none(),
        "the OTLP exporter exposes no capture API",
    );

    let emitted = emit_spans(&handle);
    handle.force_flush();

    let requests = runtime
        .block_on(server.received_requests())
        .expect("the mock records requests");
    assert_eq!(requests.len(), 1, "one batch, one POST");
    let request = &requests[0];
    assert_eq!(request.url.path(), "/v1/traces");
    assert_eq!(
        request
            .headers
            .get("content-type")
            .expect("a content type")
            .to_str()
            .expect("an ASCII content type"),
        "application/json",
    );
    assert_eq!(
        request
            .headers
            .get("authorization")
            .expect("the configured authorization header")
            .to_str()
            .expect("an ASCII authorization"),
        "Bearer test-token",
    );

    let body: Value = serde_json::from_slice(&request.body).expect("the body is JSON");
    assert_resource(&body);
    assert_scope_grouping(&body);
    assert_root_span(&body, &emitted.root);
    assert_root_event(&body);
    assert_child_span(&body, &emitted.root);
    assert_failed_span(&body);

    handle.shutdown();
    assert!(!handle.is_active(), "shutdown deactivates the handle");
    handle.shutdown(); // idempotent
    handle.force_flush(); // harmless after shutdown
}

#[test]
fn the_environment_endpoint_and_protocol_drive_the_export() {
    // test_setup.py: `OTEL_EXPORTER_OTLP_*` handling, end to end. The generic
    // endpoint points at a closed port on purpose: only the trace-specific
    // variable explains why the span still arrives.
    let _environment = EnvGuard::new();
    let runtime = tokio::runtime::Runtime::new().expect("a tokio runtime starts");
    let server = runtime.block_on(MockServer::builder().start());
    mount_traces(&runtime, &server);

    let endpoint = server.uri();
    set_env(env::OPENJIUWEN_OTEL_EXPORTER_OTLP_PROTOCOL, "http/json");
    set_env(env::OTEL_EXPORTER_OTLP_TRACES_ENDPOINT, &endpoint);
    set_env(env::OTEL_EXPORTER_OTLP_ENDPOINT, "http://127.0.0.1:9");

    let config = OtelTracerConfig::default()
        .with_exporter_type(ExporterType::Otlp)
        .apply_env();
    assert_eq!(config.protocol, OtlpProtocol::Http);
    assert_eq!(config.exporter_endpoint.as_deref(), Some(endpoint.as_str()));

    let handle = init_otel_tracer(config).expect("the environment selects OTLP/HTTP");
    handle
        .tracer()
        .expect("an active handle exposes a tracer")
        .start("env.span")
        .end();
    handle.force_flush();

    let requests = runtime
        .block_on(server.received_requests())
        .expect("the mock records requests");
    assert_eq!(
        requests.len(),
        1,
        "the span reached the endpoint named by the environment",
    );
    let body: Value = serde_json::from_slice(&requests[0].body).expect("the body is JSON");
    assert_eq!(spans_of(&body)[0]["name"], json!("env.span"));
    handle.shutdown();
}

#[test]
fn the_openjiuwen_endpoint_variable_beats_the_standard_one() {
    // test_setup.py: the `OPENJIUWEN_OTEL_*` precedence of the collector setup.
    let _environment = EnvGuard::new();
    let runtime = tokio::runtime::Runtime::new().expect("a tokio runtime starts");
    let server = runtime.block_on(MockServer::builder().start());
    mount_traces(&runtime, &server);

    let endpoint = server.uri();
    set_env(env::OPENJIUWEN_OTEL_EXPORTER_OTLP_PROTOCOL, "http/json");
    set_env(env::OPENJIUWEN_OTEL_EXPORTER_OTLP_ENDPOINT, &endpoint);
    set_env(env::OTEL_EXPORTER_OTLP_ENDPOINT, "http://127.0.0.1:9");

    let config = OtelTracerConfig::from_env();
    assert_eq!(config.exporter_endpoint.as_deref(), Some(endpoint.as_str()));
    assert_eq!(config.protocol, OtlpProtocol::Http);

    let handle = init_otel_tracer(config).expect("the environment selects OTLP/HTTP");
    handle
        .tracer()
        .expect("an active handle exposes a tracer")
        .start("openjiuwen.span")
        .end();
    handle.force_flush();

    let requests = runtime
        .block_on(server.received_requests())
        .expect("the mock records requests");
    assert_eq!(
        requests.len(),
        1,
        "the openJiuwen variable won over the standard one",
    );
    handle.shutdown();
}

#[test]
fn the_precedence_is_visible_at_the_pure_function_level() {
    // test_setup.py: `os.getenv(...)` precedence, lifted into `env`.
    let _environment = EnvGuard::new();

    // Nothing is set: the documented default applies.
    assert_eq!(env::resolve_endpoint(None), env::DEFAULT_OTLP_HTTP_ENDPOINT);

    set_env(env::OTEL_EXPORTER_OTLP_ENDPOINT, "http://env:4318");
    set_env(env::OTEL_EXPORTER_OTLP_PROTOCOL, "grpc");

    // An explicit endpoint outranks the environment.
    assert_eq!(
        env::resolve_endpoint(Some("http://explicit")),
        "http://explicit"
    );
    assert_eq!(env::resolve_endpoint(None), "http://env:4318");

    // An explicit `Http` protocol outranks a `grpc` variable.
    assert_eq!(
        env::resolve_protocol(OtlpProtocol::Http).expect("an explicit HTTP protocol wins"),
        OtlpProtocol::Http,
    );

    // An explicit `HttpProtobuf` protocol outranks the same conflicting
    // variable, which is what makes the configuration authoritative.
    assert_eq!(
        env::resolve_protocol(OtlpProtocol::HttpProtobuf)
            .expect("an explicit protobuf protocol wins"),
        OtlpProtocol::HttpProtobuf,
    );

    // With the default (`Grpc`) left in charge, the variable decides — and
    // `grpc` itself stays typed-unsupported, naming both HTTP encodings.
    let error = env::resolve_protocol(OtlpProtocol::Grpc).expect_err("grpc is unsupported");
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert!(
        error.message().contains("http/json"),
        "the reason names the JSON alternative: {error}",
    );
    assert!(
        error.message().contains("http/protobuf"),
        "the reason names the protobuf alternative: {error}",
    );

    // Every documented `http/protobuf` spelling selects the protobuf transport,
    // including the mixed-case and padded ones.
    for raw in [
        "http/protobuf",
        "http_protobuf",
        "http-protobuf",
        "protobuf",
        "  HTTP/Protobuf  ",
        "Protobuf",
    ] {
        assert_eq!(
            env::parse_protocol(raw).expect("protobuf is a supported encoding"),
            OtlpProtocol::HttpProtobuf,
            "value {raw:?}",
        );
    }

    // ... while the JSON spellings keep selecting the JSON transport.
    for raw in ["http", "http/json", "http_json", "json", "HTTP/JSON"] {
        assert_eq!(
            env::parse_protocol(raw).expect("JSON is a supported encoding"),
            OtlpProtocol::Http,
            "value {raw:?}",
        );
    }

    // A protobuf variable selects protobuf when the configuration keeps the
    // default, and a JSON variable keeps selecting JSON.
    set_env(env::OTEL_EXPORTER_OTLP_PROTOCOL, "http/protobuf");
    assert_eq!(
        env::resolve_protocol(OtlpProtocol::Grpc).expect("protobuf is supported"),
        OtlpProtocol::HttpProtobuf,
    );
    set_env(env::OTEL_EXPORTER_OTLP_PROTOCOL, "http/json");
    assert_eq!(
        env::resolve_protocol(OtlpProtocol::Grpc).expect("JSON is supported"),
        OtlpProtocol::Http,
    );

    // A typo is a validation error naming every accepted value.
    for raw in ["", "thrift"] {
        let error = env::parse_protocol(raw).expect_err("unknown protocols are rejected");
        assert_eq!(error.kind(), ErrorKind::Validation, "value {raw:?}");
        for accepted in [
            "http/json",
            "http",
            "http/protobuf",
            "http_protobuf",
            "protobuf",
        ] {
            assert!(
                error.message().contains(accepted),
                "the reason names {accepted:?}: {error}",
            );
        }
    }

    // The openJiuwen endpoint outranks the standard generic one.
    set_env(
        env::OPENJIUWEN_OTEL_EXPORTER_OTLP_ENDPOINT,
        "http://openjiuwen:4318",
    );
    assert_eq!(env::resolve_endpoint(None), "http://openjiuwen:4318");
}

#[test]
fn a_rejected_batch_reports_the_collector_status() {
    // No `#[tokio::test]`: the exporter owns its own runtime, so a caller
    // without a reactor still gets an answer instead of a panic.
    let runtime = tokio::runtime::Runtime::new().expect("a tokio runtime starts");
    let server = runtime.block_on(MockServer::builder().start());
    runtime.block_on(
        Mock::given(method("POST"))
            .and(path("/v1/traces"))
            .respond_with(ResponseTemplate::new(500).set_body_string("collector exploded"))
            .mount(&server),
    );

    let exporter = OtlpHttpExporter::new(server.uri(), BTreeMap::new(), Duration::from_secs(5))
        .expect("the exporter builds without I/O");
    match block_on(exporter.export(vec![captured_span()])) {
        Err(OTelSdkError::InternalFailure(message)) => {
            assert!(message.contains("500"), "the status is reported: {message}");
            assert!(
                message.contains("collector exploded"),
                "the body is reported: {message}",
            );
        }
        other => panic!("expected an internal failure, got {other:?}"),
    }
}

#[test]
fn a_slow_collector_is_cut_off_at_the_export_timeout() {
    let runtime = tokio::runtime::Runtime::new().expect("a tokio runtime starts");
    let server = runtime.block_on(MockServer::builder().start());
    runtime.block_on(
        Mock::given(method("POST"))
            .and(path("/v1/traces"))
            .respond_with(ResponseTemplate::new(200).set_delay(Duration::from_millis(600)))
            .mount(&server),
    );

    let exporter = OtlpHttpExporter::new(server.uri(), BTreeMap::new(), Duration::from_millis(150))
        .expect("the exporter builds without I/O");

    let started = Instant::now();
    match block_on(exporter.export(vec![captured_span()])) {
        Err(OTelSdkError::InternalFailure(message)) => {
            assert!(
                message.contains("timed out"),
                "unexpected message: {message}"
            );
        }
        other => panic!("expected an internal failure, got {other:?}"),
    }
    let elapsed = started.elapsed();
    assert!(elapsed < Duration::from_secs(5), "took {elapsed:?}");
}

#[test]
fn an_export_after_shutdown_is_a_typed_failure() {
    // `127.0.0.1:9` is closed, which is fine: the batch is rejected before any
    // request is attempted.
    let mut exporter = OtlpHttpExporter::new(
        "http://127.0.0.1:9",
        BTreeMap::new(),
        Duration::from_secs(1),
    )
    .expect("the exporter builds without I/O");
    assert!(exporter
        .shutdown_with_timeout(Duration::from_millis(250))
        .is_ok());
    assert!(exporter
        .shutdown_with_timeout(Duration::from_millis(250))
        .is_ok());

    match block_on(exporter.export(vec![captured_span()])) {
        Err(OTelSdkError::InternalFailure(message)) => {
            assert!(
                message.contains("shut down"),
                "unexpected message: {message}"
            );
        }
        other => panic!("expected an internal failure, got {other:?}"),
    }
}

#[test]
fn otlp_grpc_stays_typed_unsupported() {
    // test_setup.py: the protocol selection. Clearing both protocol variables
    // keeps the configuration default (`OtlpProtocol::Grpc`) in charge, and that
    // transport is rejected instead of being downgraded to one of the HTTP
    // encodings.
    let _environment = EnvGuard::new();
    let config = OtelTracerConfig {
        exporter_type: ExporterType::Otlp,
        protocol: OtlpProtocol::Grpc,
        exporter_endpoint: Some("http://127.0.0.1:9".to_owned()),
        ..OtelTracerConfig::default()
    };

    let error = init_otel_tracer(config).expect_err("OTLP/gRPC is not implemented");
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    let message = error.to_string();
    assert!(message.contains("gRPC"), "unexpected message: {message}");
    // The rejection is about gRPC *framing* (tonic/prost/HTTP2), never about a
    // missing encoding: both OTLP/HTTP encodings exist and are named.
    assert!(
        message.contains("framing"),
        "the reason is the gRPC framing: {message}",
    );
    assert!(
        message.contains("http/json"),
        "the reason names the JSON alternative: {message}",
    );
    assert!(
        message.contains("http/protobuf"),
        "the reason names the protobuf alternative: {message}",
    );
}

#[test]
fn every_protocol_spelling_deserializes_to_its_transport() {
    // test_setup.py: Python reads the protocol as a plain string; the Rust
    // configuration accepts the same spellings and normalizes them.
    for (raw, expected, canonical) in [
        ("grpc", OtlpProtocol::Grpc, "grpc"),
        ("grpc ", OtlpProtocol::Grpc, "grpc"),
        ("http", OtlpProtocol::Http, "http"),
        ("http/json", OtlpProtocol::Http, "http"),
        ("http_json", OtlpProtocol::Http, "http"),
        ("json", OtlpProtocol::Http, "http"),
        ("  HTTP/JSON  ", OtlpProtocol::Http, "http"),
        ("http/protobuf", OtlpProtocol::HttpProtobuf, "http_protobuf"),
        ("http_protobuf", OtlpProtocol::HttpProtobuf, "http_protobuf"),
        ("http-protobuf", OtlpProtocol::HttpProtobuf, "http_protobuf"),
        ("protobuf", OtlpProtocol::HttpProtobuf, "http_protobuf"),
        (
            "  HTTP/Protobuf  ",
            OtlpProtocol::HttpProtobuf,
            "http_protobuf",
        ),
    ] {
        let protocol: OtlpProtocol = serde_json::from_value(json!(raw))
            .unwrap_or_else(|error| panic!("{raw:?} must deserialize: {error}"));
        assert_eq!(protocol, expected, "value {raw:?}");

        // The canonical snake_case spelling is what the configuration writes
        // back, so a round trip is stable.
        assert_eq!(
            serde_json::to_value(protocol).expect("the protocol serializes"),
            json!(canonical),
            "value {raw:?}",
        );
        let restored: OtlpProtocol =
            serde_json::from_value(json!(canonical)).expect("the canonical name deserializes");
        assert_eq!(restored, expected, "value {raw:?}");
    }

    // An unknown value names every accepted variant.
    let error = serde_json::from_value::<OtlpProtocol>(json!("thrift"))
        .expect_err("an unknown protocol is rejected");
    let message = error.to_string();
    for accepted in ["grpc", "http", "http_protobuf"] {
        assert!(
            message.contains(accepted),
            "the error names {accepted:?}: {message}",
        );
    }
}

#[test]
fn the_otlp_http_protobuf_configuration_builds_an_active_handle() {
    // test_setup.py: the exporter selection for the protobuf encoding. No span
    // is created, so the loopback endpoint is never contacted.
    let _environment = EnvGuard::new();
    let config = OtelTracerConfig {
        exporter_type: ExporterType::Otlp,
        protocol: OtlpProtocol::HttpProtobuf,
        exporter_endpoint: Some("http://127.0.0.1:9".to_owned()),
        export_timeout_ms: 500,
        ..OtelTracerConfig::default()
    };

    let handle = init_otel_tracer(config).expect("OTLP/HTTP protobuf initializes offline");
    assert!(handle.is_active());
    assert!(
        handle.tracer().is_some(),
        "an active handle exposes a tracer"
    );
    assert!(
        handle.capture().is_none(),
        "the OTLP exporter exposes no capture API",
    );

    handle.force_flush();
    handle.shutdown();
    assert!(!handle.is_active(), "shutdown deactivates the handle");
    handle.shutdown(); // idempotent
    handle.shutdown_with_timeout(Duration::from_millis(50)); // idempotent
    handle.force_flush(); // harmless after shutdown

    // The encoding fixes the content type of every request and is visible in
    // the exporter's (credential-free) debug view.
    let exporter = OtlpHttpExporter::with_encoding(
        "http://127.0.0.1:9",
        BTreeMap::new(),
        Duration::from_secs(1),
        OtlpEncoding::Protobuf,
    )
    .expect("the protobuf exporter builds without I/O");
    assert_eq!(exporter.content_type(), "application/x-protobuf");
    assert_eq!(
        exporter.traces_url(),
        "http://127.0.0.1:9/v1/traces",
        "the traces suffix is appended",
    );
    let debug = format!("{exporter:?}");
    assert!(
        debug.contains("Protobuf"),
        "the debug view names the encoding: {debug}",
    );
    let json = OtlpHttpExporter::new(
        "http://127.0.0.1:9",
        BTreeMap::new(),
        Duration::from_secs(1),
    )
    .expect("the JSON exporter builds without I/O");
    assert_eq!(json.content_type(), "application/json");
    assert_eq!(
        OtlpEncoding::default(),
        OtlpEncoding::Json,
        "JSON stays the default encoding",
    );
}
