//! Crate-boundary test of the OTLP/HTTP **protobuf** (`http/protobuf`) export.
//!
//! Python counterpart: `tests/unit_tests/extensions/tracer_otel/test_setup.py`
//! — Python hands the encoding to `OTLPSpanExporter` and never inspects the
//! bytes, so the wire payload is asserted here. The batch is posted to a
//! loopback `wiremock` collector, decoded with the small hand-written protobuf
//! reader below (no `prost`, no `opentelemetry-otlp`), and compared against a
//! fixed fixture whose trace/span ids have recognisable bytes.
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

/// Varint wire type: `int32`, `int64`, `uint32`, `uint64`, `bool`, enums.
const WIRE_VARINT: u8 = 0;
/// 64-bit wire type: `fixed64` and `double`.
const WIRE_FIXED64: u8 = 1;
/// Length-delimited wire type: strings, bytes, and sub-messages.
const WIRE_LEN: u8 = 2;
/// 32-bit wire type: `fixed32` and `float`.
const WIRE_FIXED32: u8 = 5;

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

// ---------------------------------------------------------------------------
// A minimal protobuf reader: just enough to decode an OTLP trace request.
// ---------------------------------------------------------------------------

/// One decoded protobuf value.
#[derive(Debug, Clone, PartialEq)]
enum Wire {
    /// Wire type 0.
    Varint(u64),
    /// Wire type 1.
    Fixed64(u64),
    /// Wire type 5.
    Fixed32(u32),
    /// Wire type 2: strings, bytes, and buffered sub-messages.
    Bytes(Vec<u8>),
}

/// The decoded fields of one message, keyed by field number.
type Fields = BTreeMap<u32, Vec<Wire>>;

/// Reads one base-128 varint, advancing `offset`.
fn read_varint(input: &[u8], offset: &mut usize) -> u64 {
    let mut value = 0_u64;
    let mut shift = 0_u32;
    loop {
        let byte = *input.get(*offset).expect("truncated varint");
        *offset += 1;
        value |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return value;
        }
        shift += 7;
    }
}

/// Decodes one message into its fields.
fn decode(input: &[u8]) -> Fields {
    let mut fields = Fields::new();
    let mut offset = 0;
    while offset < input.len() {
        let key = read_varint(input, &mut offset);
        let field = u32::try_from(key >> 3).expect("field number out of range");
        let wire_type = u8::try_from(key & 0x7).expect("wire type out of range");
        let value = match wire_type {
            WIRE_VARINT => Wire::Varint(read_varint(input, &mut offset)),
            WIRE_FIXED64 => {
                let bytes: [u8; 8] = input
                    .get(offset..offset + 8)
                    .expect("truncated fixed64")
                    .try_into()
                    .expect("fixed64 is eight bytes");
                offset += 8;
                Wire::Fixed64(u64::from_le_bytes(bytes))
            }
            WIRE_FIXED32 => {
                let bytes: [u8; 4] = input
                    .get(offset..offset + 4)
                    .expect("truncated fixed32")
                    .try_into()
                    .expect("fixed32 is four bytes");
                offset += 4;
                Wire::Fixed32(u32::from_le_bytes(bytes))
            }
            WIRE_LEN => {
                let length =
                    usize::try_from(read_varint(input, &mut offset)).expect("length out of range");
                let bytes = input
                    .get(offset..offset + length)
                    .expect("truncated length-delimited field")
                    .to_vec();
                offset += length;
                Wire::Bytes(bytes)
            }
            other => panic!("unsupported wire type {other}"),
        };
        fields.entry(field).or_default().push(value);
    }
    fields
}

/// The first varint stored under `field`.
fn varint(fields: &Fields, field: u32) -> Option<u64> {
    fields.get(&field).and_then(|values| match values.first() {
        Some(Wire::Varint(value)) => Some(*value),
        _ => None,
    })
}

/// The first `fixed64`/`double` word stored under `field`.
fn fixed64(fields: &Fields, field: u32) -> Option<u64> {
    fields.get(&field).and_then(|values| match values.first() {
        Some(Wire::Fixed64(value)) => Some(*value),
        _ => None,
    })
}

/// The first `fixed32`/`float` word stored under `field`.
fn fixed32(fields: &Fields, field: u32) -> Option<u32> {
    fields.get(&field).and_then(|values| match values.first() {
        Some(Wire::Fixed32(value)) => Some(*value),
        _ => None,
    })
}

/// The first length-delimited payload stored under `field`.
fn bytes(fields: &Fields, field: u32) -> Option<&[u8]> {
    fields.get(&field).and_then(|values| match values.first() {
        Some(Wire::Bytes(value)) => Some(value.as_slice()),
        _ => None,
    })
}

/// The first string stored under `field`.
fn string(fields: &Fields, field: u32) -> Option<String> {
    bytes(fields, field).map(|bytes| String::from_utf8(bytes.to_vec()).expect("UTF-8 string"))
}

/// Every sub-message stored under `field`, in wire order.
fn messages(fields: &Fields, field: u32) -> Vec<Fields> {
    fields.get(&field).map_or_else(Vec::new, |values| {
        values
            .iter()
            .filter_map(|value| match value {
                Wire::Bytes(bytes) => Some(decode(bytes)),
                _ => None,
            })
            .collect()
    })
}

/// Decodes a repeated `KeyValue` field into attribute name → `AnyValue` fields.
fn key_values(fields: &Fields, field: u32) -> BTreeMap<String, Fields> {
    messages(fields, field)
        .iter()
        .map(|pair| {
            let key = string(pair, 1).expect("every KeyValue has a key");
            let mut values = messages(pair, 2);
            assert_eq!(values.len(), 1, "every KeyValue has one AnyValue");
            (key, values.remove(0))
        })
        .collect()
}

/// The `string_value` of an `AnyValue`.
fn any_string(value: &Fields) -> String {
    string(value, 1).expect("the AnyValue carries a string_value")
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
