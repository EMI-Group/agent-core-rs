//! Crate-boundary test of the OTLP/HTTP **protobuf** (`http/protobuf`) export.
//!
//! Python counterpart: `tests/unit_tests/extensions/tracer_otel/test_setup.py`
//! — Python hands the encoding to `OTLPSpanExporter` and never inspects the
//! bytes, so the wire payload is asserted here. The batch is posted to a
//! loopback `wiremock` collector, decoded with the small hand-written protobuf
//! reader in [`support`] (no `prost`, no `opentelemetry-otlp`; the same reader
//! also serves `tests/otel_team_memory.rs`), and compared against a fixed
//! fixture whose trace/span ids have recognisable bytes.
//!
//! What the file pins:
//!
//! - `OtlpHttpExporter::with_encoding(.., OtlpEncoding::Protobuf)` posts one
//!   `POST /v1/traces` with `content-type: application/x-protobuf` and the
//!   configured headers intact;
//! - the body is an OTLP `ExportTraceServiceRequest`: the resource attributes,
//!   the instrumentation scope, and the raw 16/8-byte ids, name, kind,
//!   `fixed64` nanosecond timestamps, every `AnyValue` attribute variant, and
//!   the described error status;
//! - the reader itself understands every wire type the OTLP message set uses.
//!
//! Everything is offline and deterministic: the only peer is a loopback
//! `wiremock` server, the fixture ids and timestamps are constants, and every
//! await is bounded by the exporter's request timeout.

#![cfg(feature = "otel")]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod support;

use std::collections::BTreeMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use futures::executor::block_on;
use opentelemetry::trace::{
    Span as _, SpanContext, SpanId, SpanKind, Status, TraceFlags, TraceId, TraceState, Tracer as _,
    TracerProvider as _,
};
use opentelemetry::Value as OtelValue;
use opentelemetry::{Array, InstrumentationScope, KeyValue, StringValue};
use opentelemetry_sdk::trace::{SdkTracerProvider, SpanData, SpanExporter as _};
use opentelemetry_sdk::Resource;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use openjiuwen_extensions::otel::{CaptureExporter, OtlpEncoding, OtlpHttpExporter};

use support::{
    any_string, bytes, decode, fixed32, fixed64, key_values, messages, string, varint, Fields,
};

/// The fixture trace id, chosen so every byte is distinguishable.
const TRACE_ID: [u8; 16] = [
    0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef, 0xfe, 0xdc, 0xba, 0x98, 0x76, 0x54, 0x32, 0x10,
];

/// The fixture span id.
const SPAN_ID: [u8; 8] = [0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10, 0x11];

/// The fixture parent span id.
const PARENT_SPAN_ID: [u8; 8] = [0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28];

/// The fixture span name.
const SPAN_NAME: &str = "otlp.protobuf";

/// The fixture instrumentation scope name.
const SCOPE_NAME: &str = "openjiuwen.tracer.otlp-protobuf.test";

/// The fixture instrumentation scope version.
const SCOPE_VERSION: &str = "1.2.3";

/// The fixture `service.name` resource attribute.
const SERVICE_NAME: &str = "openjiuwen-otlp-protobuf-tests";

/// The fixture `service.version` resource attribute.
const SERVICE_VERSION: &str = "9.9.9";

/// The fixed start instant of the fixture span.
fn fixture_start() -> SystemTime {
    UNIX_EPOCH + Duration::from_secs(1_700_000_000) + Duration::from_nanos(123)
}

/// The fixed end instant of the fixture span.
fn fixture_end() -> SystemTime {
    fixture_start() + Duration::from_millis(7)
}

/// Nanoseconds since the Unix epoch, the expected `fixed64` timestamp.
fn nanos(time: SystemTime) -> u64 {
    u64::try_from(
        time.duration_since(UNIX_EPOCH)
            .expect("the fixture is after the epoch")
            .as_nanos(),
    )
    .expect("the fixture fits in u64 nanoseconds")
}

/// The resource the assertions expect.
fn fixture_resource() -> Resource {
    Resource::builder_empty()
        .with_attributes([
            KeyValue::new("service.name", SERVICE_NAME),
            KeyValue::new("service.version", SERVICE_VERSION),
        ])
        .build()
}

/// A finished span from a real provider, used only for its event and link
/// collections: `SpanEvents`/`SpanLinks` are `#[non_exhaustive]` and cannot be
/// built by a consumer, so the fixture borrows them instead.
fn seed_span() -> SpanData {
    let capture = CaptureExporter::new();
    let provider = SdkTracerProvider::builder()
        .with_simple_exporter(capture.clone())
        .build();
    let tracer = provider.tracer("openjiuwen.tracer.otlp-protobuf.test");
    tracer.start("seed").end();
    provider
        .force_flush()
        .expect("the capture exporter flushes");
    let mut spans = capture.spans();
    assert_eq!(spans.len(), 1, "the seed span is captured");
    spans.remove(0)
}

/// The exported span: fixed ids and timestamps, one attribute per `AnyValue`
/// variant, and a described error status.
fn fixture_span() -> SpanData {
    SpanData {
        span_context: SpanContext::new(
            TraceId::from_bytes(TRACE_ID),
            SpanId::from_bytes(SPAN_ID),
            TraceFlags::SAMPLED,
            false,
            TraceState::NONE,
        ),
        parent_span_id: SpanId::from_bytes(PARENT_SPAN_ID),
        parent_span_is_remote: false,
        span_kind: SpanKind::Client,
        name: SPAN_NAME.into(),
        start_time: fixture_start(),
        end_time: fixture_end(),
        attributes: vec![
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
        ],
        dropped_attributes_count: 0,
        status: Status::error("boom"),
        instrumentation_scope: InstrumentationScope::builder(SCOPE_NAME)
            .with_version(SCOPE_VERSION)
            .build(),
        // No events and no links: the seed's collections are empty.
        ..seed_span()
    }
}

#[test]
fn the_decoder_round_trips_every_wire_type() {
    // The reader is the measuring instrument for the assertions below, so its
    // own behaviour is pinned: one field per wire type, little-endian words.
    let input = [
        0x08, 0x96, 0x01, // 1: varint 150
        0x11, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, // 2: fixed64
        0x1a, 0x03, b'a', b'b', b'c', // 3: length-delimited
        0x25, 0x78, 0x56, 0x34, 0x12, // 4: fixed32
    ];

    let fields = decode(&input);
    assert_eq!(varint(&fields, 1), Some(150));
    assert_eq!(
        fixed64(&fields, 2),
        Some(u64::from_le_bytes([
            0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08
        ])),
    );
    assert_eq!(bytes(&fields, 3), Some(b"abc".as_slice()));
    assert_eq!(fixed32(&fields, 4), Some(0x1234_5678));
    assert_eq!(string(&fields, 3).as_deref(), Some("abc"));
}

/// Asserts the HTTP envelope of the single export request.
fn assert_request(request: &wiremock::Request) {
    assert_eq!(request.url.path(), "/v1/traces");
    assert_eq!(request.method.as_str(), "POST");
    assert_eq!(
        request
            .headers
            .get("content-type")
            .expect("a content type")
            .to_str()
            .expect("an ASCII content type"),
        "application/x-protobuf",
    );
    assert_eq!(
        request
            .headers
            .get("authorization")
            .expect("the configured authorization header")
            .to_str()
            .expect("an ASCII authorization"),
        "Bearer token-123",
    );
}

/// Decodes the body and asserts the resource and instrumentation-scope envelope,
/// returning the single `ScopeSpans` message the span lives in.
fn decode_envelope(body: &[u8]) -> Fields {
    // ExportTraceServiceRequest { repeated ResourceSpans resource_spans = 1 }
    let body = decode(body);
    let mut resource_spans = messages(&body, 1);
    assert_eq!(resource_spans.len(), 1, "exactly one ResourceSpans entry");
    let resource_spans = resource_spans.remove(0);

    // Resource { repeated KeyValue attributes = 1 }
    let mut resources = messages(&resource_spans, 1);
    assert_eq!(resources.len(), 1, "the resource is always present");
    let resource_attributes = key_values(&resources.remove(0), 1);
    assert_eq!(
        any_string(&resource_attributes["service.name"]),
        SERVICE_NAME,
        "the configured service.name is exported",
    );
    assert_eq!(
        any_string(&resource_attributes["service.version"]),
        SERVICE_VERSION,
    );

    // ScopeSpans { InstrumentationScope scope = 1; repeated Span spans = 2 }
    let mut scope_spans = messages(&resource_spans, 2);
    assert_eq!(scope_spans.len(), 1, "one scope groups the batch");
    let scope_spans = scope_spans.remove(0);

    let mut scopes = messages(&scope_spans, 1);
    assert_eq!(scopes.len(), 1, "one instrumentation scope");
    let scope = scopes.remove(0);
    assert_eq!(string(&scope, 1).as_deref(), Some(SCOPE_NAME));
    assert_eq!(string(&scope, 2).as_deref(), Some(SCOPE_VERSION));

    scope_spans
}

/// The single span of a `ScopeSpans` message.
fn single_span(scope_spans: &Fields) -> Fields {
    let mut spans = messages(scope_spans, 2);
    assert_eq!(spans.len(), 1, "one span in the batch");
    spans.remove(0)
}

/// Asserts the span ids, name, kind, and `fixed64` timestamps.
fn assert_span_identity(span: &Fields) {
    assert_eq!(
        bytes(span, 1),
        Some(TRACE_ID.as_slice()),
        "the trace id is the raw 16-byte id, not hex",
    );
    assert_eq!(
        bytes(span, 1).expect("a trace id").len(),
        16,
        "trace ids are 16 bytes on the wire",
    );
    assert_eq!(
        bytes(span, 2),
        Some(SPAN_ID.as_slice()),
        "the span id is the raw 8-byte id",
    );
    assert_eq!(bytes(span, 2).expect("a span id").len(), 8);
    assert_eq!(
        bytes(span, 4),
        Some(PARENT_SPAN_ID.as_slice()),
        "the fixture's parent span id survives unchanged",
    );
    assert_eq!(string(span, 5).as_deref(), Some(SPAN_NAME));
    assert_eq!(varint(span, 6), Some(3), "SPAN_KIND_CLIENT is enum value 3");
    assert_eq!(
        fixed64(span, 7),
        Some(nanos(fixture_start())),
        "start_time_unix_nano is a fixed64",
    );
    assert_eq!(
        fixed64(span, 8),
        Some(nanos(fixture_end())),
        "end_time_unix_nano is a fixed64",
    );
    assert!(
        fixed64(span, 7) <= fixed64(span, 8),
        "the span must not end before it starts",
    );
}

/// Asserts one attribute per `AnyValue` variant, including the array members.
fn assert_span_attributes(span: &Fields) {
    let attributes = key_values(span, 9);
    assert_eq!(attributes.len(), 5, "every attribute is on the wire");
    assert_eq!(any_string(&attributes["test.string"]), "value");
    assert_eq!(
        varint(&attributes["test.int"], 3),
        Some(42),
        "int_value is field 3",
    );
    assert_eq!(
        fixed64(&attributes["test.double"], 4).map(f64::from_bits),
        Some(1.5),
        "double_value is field 4",
    );
    assert_eq!(
        varint(&attributes["test.bool"], 2),
        Some(1),
        "bool_value is field 2",
    );
    let mut arrays = messages(&attributes["test.tags"], 5);
    assert_eq!(arrays.len(), 1, "array_value is field 5");
    let members: Vec<String> = messages(&arrays.remove(0), 1)
        .iter()
        .map(any_string)
        .collect();
    assert_eq!(
        members,
        ["a", "b"],
        "the string array is preserved in order"
    );
}

/// Asserts the described error status.
fn assert_error_status(span: &Fields) {
    // Status { string message = 2; StatusCode code = 3 }
    let mut statuses = messages(span, 15);
    assert_eq!(statuses.len(), 1, "the status is always present");
    let status = statuses.remove(0);
    assert_eq!(varint(&status, 3), Some(2), "STATUS_CODE_ERROR is 2");
    assert_eq!(string(&status, 2).as_deref(), Some("boom"));
}

#[test]
fn the_handle_posts_one_otlp_protobuf_batch_to_the_configured_collector() {
    // test_setup.py: the exporter selection; the payload itself is delegated to
    // the SDK there, so the protobuf body is asserted field by field here.
    let runtime = tokio::runtime::Runtime::new().expect("a tokio runtime starts");
    let server = runtime.block_on(MockServer::builder().start());
    runtime.block_on(
        Mock::given(method("POST"))
            .and(path("/v1/traces"))
            .and(header("content-type", "application/x-protobuf"))
            .and(header("authorization", "Bearer token-123"))
            .respond_with(ResponseTemplate::new(200))
            .expect(1)
            .mount(&server),
    );

    let mut exporter = OtlpHttpExporter::with_encoding(
        server.uri(),
        BTreeMap::from([("authorization".to_owned(), "Bearer token-123".to_owned())]),
        Duration::from_secs(5),
        OtlpEncoding::Protobuf,
    )
    .expect("the protobuf exporter builds without I/O");
    assert_eq!(exporter.content_type(), "application/x-protobuf");
    let debug = format!("{exporter:?}");
    assert!(
        debug.contains("Protobuf"),
        "the debug view names the encoding: {debug}",
    );
    assert!(
        !debug.contains("token-123"),
        "header values never reach the debug view: {debug}",
    );

    exporter.set_resource(&fixture_resource());
    block_on(exporter.export(vec![fixture_span()])).expect("the collector accepts the batch");

    let requests = runtime
        .block_on(server.received_requests())
        .expect("the mock records requests");
    assert_eq!(requests.len(), 1, "one batch, one POST");
    assert_request(&requests[0]);

    let scope_spans = decode_envelope(&requests[0].body);
    let span = single_span(&scope_spans);
    assert_span_identity(&span);
    assert_span_attributes(&span);
    assert_error_status(&span);

    exporter
        .shutdown_with_timeout(Duration::from_millis(250))
        .expect("the worker stops within the timeout");
}
