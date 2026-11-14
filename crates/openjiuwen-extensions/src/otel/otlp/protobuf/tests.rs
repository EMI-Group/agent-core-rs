//! Unit tests of the OTLP/protobuf body encoder.
//!
//! Python counterpart: `tests/unit_tests/extensions/tracer_otel/test_setup.py`
//! selects the exporter but leaves the payload to the SDK, so the wire format is
//! asserted here field by field with a hand-written decoder — no `prost`, no
//! collector, no network.
//!
//! The most valuable assertion is the parity test: one batch runs through
//! `json::encode_body` and [`encode_body`], and the decoded protobuf is compared
//! against the JSON value field by field, so the two encodings cannot drift.
// A test module owns its panics: a failed `unwrap`/`expect` is an assertion.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use opentelemetry::trace::{
    Link, Span as _, SpanContext, SpanId, SpanKind, Status, TraceContextExt as _, TraceFlags,
    TraceId, TraceState, Tracer as _, TracerProvider as _,
};
use opentelemetry::Value as OtelValue;
use opentelemetry::{Array, Context, InstrumentationScope, KeyValue, StringValue};
use opentelemetry_sdk::trace::{SdkTracerProvider, SpanData};
use opentelemetry_sdk::Resource;
use serde_json::{json, Value};

use super::{
    any_value_message, encode_body, string_value_message, write_fixed32, write_tag, write_varint,
    WIRE_FIXED32, WIRE_FIXED64, WIRE_LEN, WIRE_VARINT,
};
use crate::otel::otlp::json;
use crate::otel::provider::CaptureExporter;

// ---------------------------------------------------------------------------
// A minimal protobuf reader: just enough to decode what the encoder writes.
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

/// Reads one varint, advancing `offset`.
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

/// The first `fixed32` word stored under `field`.
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

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// A provider that exports synchronously into a [`CaptureExporter`].
fn capture_provider() -> (SdkTracerProvider, CaptureExporter) {
    let capture = CaptureExporter::new();
    let provider = SdkTracerProvider::builder()
        .with_simple_exporter(capture.clone())
        .build();
    (provider, capture)
}

/// The resource the parity assertions expect.
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

/// The fixed start instant of the fixture span.
fn fixture_start() -> SystemTime {
    UNIX_EPOCH + Duration::from_secs(1_700_000_000) + Duration::from_nanos(123)
}

/// The fixed end instant of the fixture span.
fn fixture_end() -> SystemTime {
    fixture_start() + Duration::from_millis(7)
}

/// Nanoseconds since the Unix epoch, the expected `*_time_unix_nano` value.
fn nanos(time: SystemTime) -> u64 {
    u64::try_from(time.duration_since(UNIX_EPOCH).unwrap().as_nanos()).unwrap()
}

/// The shared fixture: one rich span (parent-less, event, link, error status,
/// mixed attributes) plus a child span, exported through a real provider.
///
/// Returns the batch and the first span's context, which is the child's parent.
fn fixture_batch() -> (Vec<SpanData>, SpanContext) {
    let (provider, capture) = capture_provider();
    let tracer = provider.tracer("openjiuwen.test");
    let mut span = tracer
        .span_builder("client.call")
        .with_kind(SpanKind::Client)
        .with_start_time(fixture_start())
        .with_end_time(fixture_end())
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
    let context = span.span_context().clone();
    span.add_event("retry.scheduled", vec![KeyValue::new("attempt", 1_i64)]);
    span.set_status(Status::error("upstream returned 500"));
    span.end_with_timestamp(fixture_end());

    let child_context = Context::new().with_remote_span_context(context.clone());
    let mut child = tracer.start_with_context("client.child", &child_context);
    child.end();

    let _ = provider.force_flush();
    (capture.spans(), context)
}

/// One finished span carrying exactly `attributes`.
fn span_with_attributes(attributes: Vec<KeyValue>) -> SpanData {
    let (provider, capture) = capture_provider();
    let tracer = provider.tracer("openjiuwen.test");
    let mut span = tracer
        .span_builder("values")
        .with_attributes(attributes)
        .start(&tracer);
    span.end();
    let _ = provider.force_flush();
    capture.spans().remove(0)
}

/// One finished span with `status`.
fn span_with_status(status: Status) -> SpanData {
    let (provider, capture) = capture_provider();
    let tracer = provider.tracer("openjiuwen.test");
    let mut span = tracer.span_builder("status").start(&tracer);
    span.set_status(status);
    span.end();
    let _ = provider.force_flush();
    capture.spans().remove(0)
}

// ---------------------------------------------------------------------------
// Body navigation helpers
// ---------------------------------------------------------------------------

/// The single `ResourceSpans` entry of a body.
fn resource_spans_fields(body: &[u8]) -> Fields {
    let mut entries = messages(&decode(body), 1);
    assert_eq!(entries.len(), 1, "exactly one ResourceSpans entry");
    entries.remove(0)
}

/// Every `ScopeSpans` entry of a body.
fn decoded_scopes(body: &[u8]) -> Vec<Fields> {
    messages(&resource_spans_fields(body), 2)
}

/// The spans of every `ScopeSpans` entry, flattened in wire order.
fn decoded_spans(body: &[u8]) -> Vec<Fields> {
    decoded_scopes(body)
        .iter()
        .flat_map(|scope_spans| messages(scope_spans, 2))
        .collect()
}

/// The single `InstrumentationScope` of a `ScopeSpans` entry.
fn scope_of(scope_spans: &Fields) -> Fields {
    let mut scopes = messages(scope_spans, 1);
    assert_eq!(scopes.len(), 1);
    scopes.remove(0)
}

/// The JSON encoder's view of a decoded `KeyValue` list, keyed by attribute
/// name, so the two encodings can be compared directly.
fn attributes_map(pairs: &[Fields]) -> BTreeMap<String, Value> {
    pairs
        .iter()
        .map(|pair| {
            let key = string(pair, 1).expect("every KeyValue has a key");
            let mut values = messages(pair, 2);
            assert_eq!(values.len(), 1, "every KeyValue has one value");
            (key, any_value_json(&values.remove(0)))
        })
        .collect()
}

/// The JSON encoder's view of a JSON attribute array, keyed by attribute name.
fn json_attributes_map(attributes: &Value) -> BTreeMap<String, Value> {
    attributes
        .as_array()
        .expect("attributes must be an array")
        .iter()
        .map(|pair| {
            let key = pair["key"].as_str().expect("key").to_owned();
            (key, pair["value"].clone())
        })
        .collect()
}

/// The single attribute value stored under `key`.
fn attribute(pairs: &[Fields], key: &str) -> Value {
    attributes_map(pairs)
        .remove(key)
        .unwrap_or_else(|| panic!("attribute {key} is missing"))
}

/// Renders a decoded `AnyValue` the way the JSON encoder writes it.
fn any_value_json(value: &Fields) -> Value {
    if let Some(text) = string(value, 1) {
        return json!({ "stringValue": text });
    }
    if let Some(flag) = varint(value, 2) {
        return json!({ "boolValue": flag != 0 });
    }
    if let Some(number) = varint(value, 3) {
        // `int64` is two's complement on the wire.
        return json!({ "intValue": i64::from_ne_bytes(number.to_ne_bytes()).to_string() });
    }
    if let Some(bits) = fixed64(value, 4) {
        return json!({ "doubleValue": f64::from_bits(bits) });
    }
    if let Some(array) = messages(value, 5).first() {
        let values: Vec<Value> = messages(array, 1).iter().map(any_value_json).collect();
        return json!({ "arrayValue": { "values": values } });
    }
    json!({})
}

/// The JSON encoder's `kind` spelling for a decoded `SpanKind` value.
fn span_kind_name(value: u64) -> &'static str {
    match value {
        1 => "SPAN_KIND_INTERNAL",
        2 => "SPAN_KIND_SERVER",
        3 => "SPAN_KIND_CLIENT",
        4 => "SPAN_KIND_PRODUCER",
        5 => "SPAN_KIND_CONSUMER",
        _ => "SPAN_KIND_UNSPECIFIED",
    }
}

/// The JSON encoder's `status.code` spelling for a decoded `StatusCode` value.
fn status_code_name(value: u64) -> &'static str {
    match value {
        1 => "STATUS_CODE_OK",
        2 => "STATUS_CODE_ERROR",
        _ => "STATUS_CODE_UNSET",
    }
}

// ---------------------------------------------------------------------------
// Writer primitives
// ---------------------------------------------------------------------------

#[test]
fn varints_encode_the_unsigned_edges() {
    let cases: [(u64, Vec<u8>); 6] = [
        (0, vec![0x00]),
        (1, vec![0x01]),
        // The last single-byte value.
        (127, vec![0x7f]),
        // The first two-byte value.
        (128, vec![0x80, 0x01]),
        (300, vec![0xac, 0x02]),
        // A full 64-bit value: nine continuation bytes plus the top group.
        (u64::MAX, [vec![0xff; 9], vec![0x01]].concat()),
    ];
    for (value, expected) in cases {
        let mut buffer = Vec::new();
        write_varint(&mut buffer, value);
        assert_eq!(buffer, expected, "varint of {value}");
        let mut offset = 0;
        assert_eq!(read_varint(&buffer, &mut offset), value);
        assert_eq!(offset, buffer.len());
    }

    // `i64::MAX` is the same 64-bit width: 63 set bits fill nine 7-bit groups.
    let signed_max = u64::try_from(i64::MAX).unwrap();
    let mut buffer = Vec::new();
    write_varint(&mut buffer, signed_max);
    assert_eq!(buffer, [vec![0xff; 8], vec![0x7f]].concat());
    let mut offset = 0;
    assert_eq!(read_varint(&buffer, &mut offset), signed_max);
}

#[test]
fn tags_combine_field_number_and_wire_type() {
    let mut buffer = Vec::new();
    write_tag(&mut buffer, 1, WIRE_VARINT);
    assert_eq!(buffer, [0x08]);

    // A Span's `status` is field 15, wire type 2: 15 << 3 | 2 = 122.
    let mut buffer = Vec::new();
    write_tag(&mut buffer, 15, WIRE_LEN);
    assert_eq!(buffer, [0x7a]);

    // A Span's `flags` is field 16, wire type 5: 16 << 3 | 5 = 133.
    let mut buffer = Vec::new();
    write_tag(&mut buffer, 16, WIRE_FIXED32);
    assert_eq!(buffer, [0x85, 0x01]);
}

#[test]
fn fixed_width_words_are_little_endian() {
    let mut buffer = Vec::new();
    write_fixed32(&mut buffer, 0x1234_5678);
    assert_eq!(buffer, [0x78, 0x56, 0x34, 0x12]);
    let decoded = decode(&{
        let mut framed = Vec::new();
        write_tag(&mut framed, 3, WIRE_FIXED32);
        framed.extend_from_slice(&buffer);
        framed
    });
    assert_eq!(fixed32(&decoded, 3), Some(0x1234_5678));

    // `double` is the IEEE-754 bit pattern in the same little-endian word.
    let doubled = decode(&any_value_message(&OtelValue::F64(-2.25)));
    assert_eq!(
        fixed64(&doubled, 4),
        Some((-2.25_f64).to_bits()),
        "negative doubles keep their sign bit"
    );
}

// ---------------------------------------------------------------------------
// Span encoding
// ---------------------------------------------------------------------------

#[test]
fn one_span_encodes_every_field() {
    let (spans, context) = fixture_batch();
    let body = encode_body(Some(&test_resource()), &spans);
    let resource_spans = resource_spans_fields(&body);

    // The resource attributes ride on field 1 of the single ResourceSpans.
    let mut resources = messages(&resource_spans, 1);
    assert_eq!(resources.len(), 1);
    let resource_attributes = attributes_map(&messages(&resources.remove(0), 1));
    assert_eq!(
        resource_attributes.get("service.name"),
        Some(&json!({"stringValue": "otlp-test"}))
    );
    assert_eq!(
        resource_attributes.get("service.version"),
        Some(&json!({"stringValue": "1.0.0"}))
    );

    let scopes = messages(&resource_spans, 2);
    assert_eq!(scopes.len(), 1);
    let scope = scope_of(&scopes[0]);
    assert_eq!(string(&scope, 1).as_deref(), Some("openjiuwen.test"));
    assert_eq!(string(&scope, 2), None, "the tracer scope has no version");

    let decoded = messages(&scopes[0], 2);
    assert_eq!(decoded.len(), 2);
    let span = &decoded[0];

    // Ids are the raw bytes, and a root span carries no `parent_span_id`.
    assert_eq!(
        bytes(span, 1),
        Some(context.trace_id().to_bytes().as_slice())
    );
    assert_eq!(
        bytes(span, 2),
        Some(context.span_id().to_bytes().as_slice())
    );
    assert_eq!(bytes(span, 4), None, "the first span is a root");
    assert_eq!(string(span, 5).as_deref(), Some("client.call"));
    assert_eq!(varint(span, 6), Some(3), "SPAN_KIND_CLIENT");
    assert_eq!(fixed64(span, 7), Some(nanos(fixture_start())));
    assert_eq!(fixed64(span, 8), Some(nanos(fixture_end())));
    assert_eq!(varint(span, 10), None, "dropped_attributes_count is 0");

    let attributes = messages(span, 9);
    assert_eq!(attributes.len(), 5);
    assert_eq!(
        attribute(&attributes, "test.string"),
        json!({"stringValue": "value"})
    );
    assert_eq!(
        attribute(&attributes, "test.int"),
        json!({"intValue": "42"})
    );
    assert_eq!(
        attribute(&attributes, "test.double"),
        json!({"doubleValue": 1.5})
    );
    assert_eq!(
        attribute(&attributes, "test.bool"),
        json!({"boolValue": true})
    );
    assert_eq!(
        attribute(&attributes, "tags"),
        json!({"arrayValue": {"values": [{"stringValue": "a"}, {"stringValue": "b"}]}})
    );

    let events = messages(span, 11);
    assert_eq!(events.len(), 1);
    assert!(fixed64(&events[0], 1).unwrap_or(0) > 0, "event timestamp");
    assert_eq!(string(&events[0], 2).as_deref(), Some("retry.scheduled"));
    assert_eq!(
        attribute(&messages(&events[0], 3), "attempt"),
        json!({"intValue": "1"})
    );
    assert_eq!(varint(&events[0], 4), None, "dropped_attributes_count is 0");

    let links = messages(span, 13);
    assert_eq!(links.len(), 1);
    assert_eq!(
        bytes(&links[0], 1),
        Some(link_context().trace_id().to_bytes().as_slice())
    );
    assert_eq!(
        bytes(&links[0], 2),
        Some(link_context().span_id().to_bytes().as_slice())
    );
    assert_eq!(
        attribute(&messages(&links[0], 4), "link.reason"),
        json!({"stringValue": "retry"})
    );
    assert_eq!(varint(&links[0], 5), None, "dropped_attributes_count is 0");

    let statuses = messages(span, 15);
    assert_eq!(statuses.len(), 1, "status is always present");
    assert_eq!(varint(&statuses[0], 3), Some(2), "STATUS_CODE_ERROR");
    assert_eq!(
        string(&statuses[0], 2).as_deref(),
        Some("upstream returned 500")
    );

    // `trace_state`, the dropped _events/_links counts, and `flags` have no
    // counterpart in the JSON body, so both encodings omit them.
    for field in [3, 12, 14, 16] {
        assert!(span.get(&field).is_none(), "field {field} must be omitted");
    }

    // The child span points at its parent; its status is the proto3 default.
    assert_child_span(&decoded[1], &context);
}

/// The child span of the fixture batch names its parent, and its status is the
/// proto3 default, so both status members are omitted.
fn assert_child_span(child: &Fields, parent: &SpanContext) {
    assert_eq!(
        bytes(child, 4),
        Some(parent.span_id().to_bytes().as_slice())
    );
    assert_eq!(string(child, 5).as_deref(), Some("client.child"));
    let child_status = messages(child, 15).remove(0);
    assert_eq!(varint(&child_status, 3), None, "STATUS_CODE_UNSET is 0");
    assert_eq!(string(&child_status, 2), None);
}

#[test]
fn any_values_encode_every_scalar_variant() {
    let span = span_with_attributes(vec![
        KeyValue::new("bool.true", true),
        KeyValue::new("bool.false", false),
        KeyValue::new("int.min", i64::MIN),
        KeyValue::new("int.max", i64::MAX),
        KeyValue::new("double.negative", -2.25_f64),
        KeyValue::new("double.fractional", 0.5_f64),
        KeyValue::new("double.zero", 0.0_f64),
        KeyValue::new("string.text", "value"),
        KeyValue::new("string.empty", ""),
        KeyValue::new("array.mixed", OtelValue::Array(Array::I64(vec![1, -1, 0]))),
        KeyValue::new("array.empty", OtelValue::Array(Array::String(Vec::new()))),
    ]);
    let body = encode_body(None, std::slice::from_ref(&span));
    let attributes = messages(&decoded_spans(&body)[0], 9);
    assert_eq!(attributes.len(), 11);

    assert_eq!(
        attribute(&attributes, "bool.true"),
        json!({"boolValue": true})
    );
    assert_eq!(
        attribute(&attributes, "bool.false"),
        json!({"boolValue": false}),
        "oneof members carry explicit presence"
    );
    assert_eq!(
        attribute(&attributes, "int.min"),
        json!({"intValue": i64::MIN.to_string()})
    );
    assert_eq!(
        attribute(&attributes, "int.max"),
        json!({"intValue": i64::MAX.to_string()})
    );
    assert_eq!(
        attribute(&attributes, "double.negative"),
        json!({"doubleValue": -2.25})
    );
    assert_eq!(
        attribute(&attributes, "double.fractional"),
        json!({"doubleValue": 0.5})
    );
    assert_eq!(
        attribute(&attributes, "double.zero"),
        json!({"doubleValue": 0.0})
    );
    assert_eq!(
        attribute(&attributes, "string.text"),
        json!({"stringValue": "value"})
    );
    assert_eq!(
        attribute(&attributes, "string.empty"),
        json!({"stringValue": ""})
    );
    assert_eq!(
        attribute(&attributes, "array.mixed"),
        json!({"arrayValue": {"values": [
            {"intValue": "1"},
            {"intValue": "-1"},
            {"intValue": "0"},
        ]}})
    );
    assert_eq!(
        attribute(&attributes, "array.empty"),
        json!({"arrayValue": {"values": []}})
    );
}

#[test]
fn known_any_values_never_take_the_text_fallback() {
    // `opentelemetry::Value` is `#[non_exhaustive]`, so a variant newer than
    // this crate cannot be constructed in a test at all; the fallback arm is
    // therefore unreachable here, and the JSON encoder carries the same
    // unreachable arm. What *is* assertable: the fallback's encoder (wire field
    // 1, the `string_value` member) and the fact that no known variant reaches
    // it.
    let fallback = decode(&string_value_message("unknown.variant"));
    assert_eq!(string(&fallback, 1).as_deref(), Some("unknown.variant"));
    assert_eq!(varint(&fallback, 2), None);
    assert_eq!(fixed64(&fallback, 4), None);

    for value in [
        OtelValue::Bool(false),
        OtelValue::I64(0),
        OtelValue::F64(0.0),
        OtelValue::Array(Array::Bool(Vec::new())),
    ] {
        let decoded = decode(&any_value_message(&value));
        assert_eq!(
            string(&decoded, 1),
            None,
            "{value:?} must not use the string member"
        );
    }
}

#[test]
fn status_codes_and_messages_map_to_the_enum() {
    let cases: [(Status, Option<u64>, Option<&str>); 4] = [
        (Status::Unset, None, None),
        (Status::Ok, Some(1), None),
        (Status::error(""), Some(2), None),
        (
            Status::error("upstream returned 500"),
            Some(2),
            Some("upstream returned 500"),
        ),
    ];
    for (status, code, message) in cases {
        let span = span_with_status(status);
        let body = encode_body(None, std::slice::from_ref(&span));
        let statuses = messages(&decoded_spans(&body)[0], 15);
        assert_eq!(statuses.len(), 1, "status is always present");
        assert_eq!(varint(&statuses[0], 3), code, "status code");
        assert_eq!(
            string(&statuses[0], 2).as_deref(),
            message,
            "status message"
        );
    }
}

#[test]
fn every_span_kind_has_its_wire_value() {
    let cases = [
        (SpanKind::Internal, 1_u64),
        (SpanKind::Server, 2),
        (SpanKind::Client, 3),
        (SpanKind::Producer, 4),
        (SpanKind::Consumer, 5),
    ];
    for (kind, expected) in cases {
        let (provider, capture) = capture_provider();
        let tracer = provider.tracer("openjiuwen.test");
        let mut span = tracer
            .span_builder("kind")
            .with_kind(kind.clone())
            .start(&tracer);
        span.end();
        let _ = provider.force_flush();
        let batch = capture.spans();
        let body = encode_body(None, &batch);
        assert_eq!(
            varint(&decoded_spans(&body)[0], 6),
            Some(expected),
            "{kind:?}"
        );
    }
}

#[test]
fn scopes_group_spans_in_first_seen_order() {
    let (provider, capture) = capture_provider();
    let plain = provider.tracer("openjiuwen.test");
    let versioned = provider.tracer_with_scope(
        InstrumentationScope::builder("openjiuwen.worker")
            .with_version("2.0.0")
            .build(),
    );
    for (tracer, name) in [
        (&plain, "client.a"),
        (&versioned, "worker.b"),
        (&plain, "client.c"),
        (&versioned, "worker.d"),
    ] {
        let mut span = tracer.span_builder(name.to_owned()).start(tracer);
        span.end();
    }
    let _ = provider.force_flush();
    let spans = capture.spans();

    let body = encode_body(None, &spans);
    let scopes = decoded_scopes(&body);
    assert_eq!(scopes.len(), 2);

    let first = scope_of(&scopes[0]);
    assert_eq!(string(&first, 1).as_deref(), Some("openjiuwen.test"));
    assert_eq!(string(&first, 2), None);
    let first_spans = messages(&scopes[0], 2);
    assert_eq!(first_spans.len(), 2);
    assert_eq!(string(&first_spans[0], 5).as_deref(), Some("client.a"));
    assert_eq!(string(&first_spans[1], 5).as_deref(), Some("client.c"));

    let second = scope_of(&scopes[1]);
    assert_eq!(string(&second, 1).as_deref(), Some("openjiuwen.worker"));
    assert_eq!(string(&second, 2).as_deref(), Some("2.0.0"));
    let second_spans = messages(&scopes[1], 2);
    assert_eq!(second_spans.len(), 2);
    assert_eq!(string(&second_spans[0], 5).as_deref(), Some("worker.b"));
    assert_eq!(string(&second_spans[1], 5).as_deref(), Some("worker.d"));

    // The JSON encoder consumes the same grouping and sees the same scopes.
    let json_body = json::encode_body(None, &spans);
    let json_scopes = json_body["resourceSpans"][0]["scopeSpans"]
        .as_array()
        .expect("scopeSpans must be an array");
    assert_eq!(json_scopes.len(), 2);
    assert_eq!(
        json_scopes[0]["scope"],
        json!({"name": "openjiuwen.test"}),
        "shared grouping keeps first-seen order"
    );
    assert_eq!(
        json_scopes[1]["scope"],
        json!({"name": "openjiuwen.worker", "version": "2.0.0"})
    );
}

#[test]
fn an_empty_batch_keeps_the_request_shape() {
    let body = encode_body(None, &[]);
    let resource_spans = resource_spans_fields(&body);

    let mut resources = messages(&resource_spans, 1);
    assert_eq!(resources.len(), 1, "the resource message is always present");
    assert!(
        messages(&resources.remove(0), 1).is_empty(),
        "a resource-less export carries no attributes"
    );
    assert!(
        messages(&resource_spans, 2).is_empty(),
        "no spans means no scopeSpans"
    );
}

// ---------------------------------------------------------------------------
// Encoding parity: the same batch through both encoders
// ---------------------------------------------------------------------------

#[test]
fn json_and_protobuf_describe_the_same_batch() {
    // Python: tests/unit_tests/extensions/tracer_otel/test_setup.py pins the
    // exporter selection; payload parity between the two OTLP encodings (which
    // the Python suite delegates to the SDK) is pinned here.
    let (spans, context) = fixture_batch();
    let resource = test_resource();
    let json_body = json::encode_body(Some(&resource), &spans);
    let protobuf_body = encode_body(Some(&resource), &spans);

    let resource_spans = resource_spans_fields(&protobuf_body);
    let mut resources = messages(&resource_spans, 1);
    assert_eq!(resources.len(), 1);
    assert_eq!(
        attributes_map(&messages(&resources.remove(0), 1)),
        json_attributes_map(&json_body["resourceSpans"][0]["resource"]["attributes"])
    );

    let scopes = messages(&resource_spans, 2);
    let json_scopes = json_body["resourceSpans"][0]["scopeSpans"]
        .as_array()
        .expect("scopeSpans must be an array");
    assert_eq!(scopes.len(), json_scopes.len());

    let decoded = messages(&scopes[0], 2);
    let json_spans = json_scopes[0]["spans"]
        .as_array()
        .expect("spans must be an array");
    assert_eq!(decoded.len(), json_spans.len());

    // The first JSON span is the fixture root, whose id the batch returned.
    assert_eq!(
        context.trace_id().to_bytes().as_slice(),
        bytes(&decoded[0], 1).expect("trace id")
    );

    for (span, encoded) in decoded.iter().zip(json_spans) {
        assert_span_matches_json(span, encoded);
        assert_events_match_json(span, encoded);
        assert_links_match_json(span, encoded);
    }
}

/// Asserts one decoded span's id, name, kind, timestamps, attributes and status
/// against the JSON encoder's span object.
fn assert_span_matches_json(span: &Fields, encoded: &Value) {
    assert_eq!(
        json::hex(bytes(span, 1).expect("trace id")),
        encoded["traceId"].as_str().expect("traceId")
    );
    assert_eq!(
        json::hex(bytes(span, 2).expect("span id")),
        encoded["spanId"].as_str().expect("spanId")
    );
    match bytes(span, 4) {
        Some(parent) => assert_eq!(
            json::hex(parent),
            encoded["parentSpanId"].as_str().expect("parentSpanId")
        ),
        None => assert!(
            encoded.get("parentSpanId").is_none(),
            "a root span has no parentSpanId in either encoding"
        ),
    }
    assert_eq!(string(span, 5), encoded["name"].as_str().map(str::to_owned));
    assert_eq!(
        span_kind_name(varint(span, 6).unwrap_or(0)),
        encoded["kind"].as_str().expect("kind")
    );
    assert_eq!(
        fixed64(span, 7).unwrap_or(0).to_string(),
        encoded["startTimeUnixNano"]
            .as_str()
            .expect("startTimeUnixNano")
    );
    assert_eq!(
        fixed64(span, 8).unwrap_or(0).to_string(),
        encoded["endTimeUnixNano"]
            .as_str()
            .expect("endTimeUnixNano")
    );
    assert_eq!(
        attributes_map(&messages(span, 9)),
        json_attributes_map(&encoded["attributes"])
    );

    let statuses = messages(span, 15);
    assert_eq!(statuses.len(), 1);
    assert_eq!(
        status_code_name(varint(&statuses[0], 3).unwrap_or(0)),
        encoded["status"]["code"].as_str().expect("status code")
    );
    assert_eq!(
        string(&statuses[0], 2),
        encoded["status"]["message"].as_str().map(str::to_owned)
    );
}

/// Asserts one decoded span's events against the JSON span's `events` array.
fn assert_events_match_json(span: &Fields, encoded: &Value) {
    let events = messages(span, 11);
    let json_events = encoded["events"].as_array().expect("events");
    assert_eq!(events.len(), json_events.len());
    for (event, expected) in events.iter().zip(json_events) {
        assert_eq!(
            fixed64(event, 1).unwrap_or(0).to_string(),
            expected["timeUnixNano"].as_str().expect("timeUnixNano")
        );
        assert_eq!(
            string(event, 2),
            expected["name"].as_str().map(str::to_owned)
        );
        assert_eq!(
            attributes_map(&messages(event, 3)),
            json_attributes_map(&expected["attributes"])
        );
    }
}

/// Asserts one decoded span's links against the JSON span's `links` array.
fn assert_links_match_json(span: &Fields, encoded: &Value) {
    let links = messages(span, 13);
    let json_links = encoded["links"].as_array().expect("links");
    assert_eq!(links.len(), json_links.len());
    for (link, expected) in links.iter().zip(json_links) {
        assert_eq!(
            json::hex(bytes(link, 1).expect("link trace id")),
            expected["traceId"].as_str().expect("traceId")
        );
        assert_eq!(
            json::hex(bytes(link, 2).expect("link span id")),
            expected["spanId"].as_str().expect("spanId")
        );
        assert_eq!(
            attributes_map(&messages(link, 4)),
            json_attributes_map(&expected["attributes"])
        );
    }
}
