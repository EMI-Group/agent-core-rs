//! OTLP/protobuf body encoder.
//!
//! Serializes an exported `SpanData` batch into the protobuf wire format of
//! `ExportTraceServiceRequest` — the body of an `http/protobuf` OTLP export.
//! The encoder is hand-written: `prost`, `tonic` and `opentelemetry-otlp` are
//! not available to an offline build, and the message set is small enough that
//! a few hundred lines of writer helpers cover it exactly.
//!
//! # Relationship to the JSON encoder
//!
//! Both encoders describe the same request and share the span grouping from
//! `model.rs`, so the scopes, their order, and the spans inside them match.
//! Deliberate choices mirror `json.rs`:
//!
//! - ids are the raw 16/8 bytes here and lowercase hex there;
//! - a pre-epoch or unrepresentable `SystemTime` becomes `0` nanoseconds;
//! - `trace_state`, `dropped_events_count`, `dropped_links_count`, and the
//!   span/link `flags` are omitted from **both** encodings (the JSON body emits
//!   no counterpart for them).
//!
//! Proto3 default scalars (empty strings/bytes, zero varints and fixed words)
//! are skipped, exactly as a `prost`-generated encoder would. The exception is
//! `AnyValue`, whose variants form a `oneof`: oneof members carry explicit
//! presence, so an empty string or a `0.0` double is still written — a decoder
//! must see which variant was selected, just as the JSON body always names it.

use std::time::{SystemTime, UNIX_EPOCH};

use opentelemetry::trace::{Event, Link, SpanId, SpanKind, Status};
use opentelemetry::{Array, InstrumentationScope, Value as OtelValue};
use opentelemetry_sdk::trace::SpanData;
use opentelemetry_sdk::Resource;

use super::model::{scope_groups, ScopeGroup};

#[cfg(test)]
mod tests;

/// Varint wire type: `int32`, `int64`, `uint32`, `uint64`, `bool`, enums.
const WIRE_VARINT: u8 = 0;
/// 64-bit wire type: `fixed64` and `double`.
const WIRE_FIXED64: u8 = 1;
/// Length-delimited wire type: strings, bytes, and sub-messages.
const WIRE_LEN: u8 = 2;
/// 32-bit wire type: `fixed32` and `float`.
///
/// Only the trace `flags` fields use it and those are not emitted, so the
/// writer is exercised by the unit tests alone.
#[cfg(test)]
const WIRE_FIXED32: u8 = 5;

/// The `StatusCode` value for a successful span.
const STATUS_CODE_OK: u64 = 1;
/// The `StatusCode` value for a failed span.
const STATUS_CODE_ERROR: u64 = 2;

/// Encodes one export batch as the OTLP/protobuf request body.
///
/// Every span goes into a single `ResourceSpans` entry, grouped by
/// instrumentation scope exactly like the JSON body: one `ScopeSpans` entry per
/// distinct `(name, version)` pair, first-seen order, span order preserved
/// inside a scope. The `resource` sub-message is always present (empty when the
/// provider set none), matching the JSON body's empty `resource` object.
pub(super) fn encode_body(resource: Option<&Resource>, spans: &[SpanData]) -> Vec<u8> {
    let mut resource_spans = Vec::new();
    write_message(&mut resource_spans, 1, &resource_message(resource));
    for group in scope_groups(spans) {
        write_message(&mut resource_spans, 2, &scope_spans_message(&group));
    }
    let mut body = Vec::new();
    write_message(&mut body, 1, &resource_spans);
    body
}

/// Encodes the `Resource` sub-message: attributes only.
fn resource_message(resource: Option<&Resource>) -> Vec<u8> {
    let mut message = Vec::new();
    if let Some(resource) = resource {
        for (key, value) in resource {
            write_message(&mut message, 1, &key_value_message(key.as_str(), value));
        }
    }
    message
}

/// Encodes one `ScopeSpans` entry from the shared scope grouping.
fn scope_spans_message(group: &ScopeGroup<'_>) -> Vec<u8> {
    let mut message = Vec::new();
    write_message(&mut message, 1, &scope_message(group.scope));
    for span in &group.spans {
        write_message(&mut message, 2, &span_message(span));
    }
    message
}

/// Encodes an `InstrumentationScope`: name, optional version, optional
/// attributes.
fn scope_message(scope: &InstrumentationScope) -> Vec<u8> {
    let mut message = Vec::new();
    write_string(&mut message, 1, scope.name());
    if let Some(version) = scope.version() {
        write_string(&mut message, 2, version);
    }
    for pair in scope.attributes() {
        write_message(
            &mut message,
            3,
            &key_value_message(pair.key.as_str(), &pair.value),
        );
    }
    message
}

/// Encodes one exported `Span`.
fn span_message(span: &SpanData) -> Vec<u8> {
    let mut message = Vec::new();
    // 1 `trace_id`, 2 `span_id`: the raw bytes, never hex.
    write_len_field(&mut message, 1, &span.span_context.trace_id().to_bytes());
    write_len_field(&mut message, 2, &span.span_context.span_id().to_bytes());
    // 3 `trace_state` is omitted, as in the JSON body.
    if span.parent_span_id != SpanId::INVALID {
        write_len_field(&mut message, 4, &span.parent_span_id.to_bytes());
    }
    write_string(&mut message, 5, span.name.as_ref());
    write_uint(&mut message, 6, span_kind_value(&span.span_kind));
    write_fixed64_field(&mut message, 7, unix_nanos(span.start_time));
    write_fixed64_field(&mut message, 8, unix_nanos(span.end_time));
    for pair in &span.attributes {
        write_message(
            &mut message,
            9,
            &key_value_message(pair.key.as_str(), &pair.value),
        );
    }
    write_uint(&mut message, 10, u64::from(span.dropped_attributes_count));
    for event in &span.events.events {
        write_message(&mut message, 11, &event_message(event));
    }
    // 12 `dropped_events_count` is omitted, as in the JSON body.
    for link in &span.links.links {
        write_message(&mut message, 13, &link_message(link));
    }
    // 14 `dropped_links_count` is omitted, as in the JSON body.
    write_message(&mut message, 15, &status_message(&span.status));
    // 16 `flags` is omitted, as in the JSON body.
    message
}

/// Encodes one span `Event`.
fn event_message(event: &Event) -> Vec<u8> {
    let mut message = Vec::new();
    write_fixed64_field(&mut message, 1, unix_nanos(event.timestamp));
    write_string(&mut message, 2, event.name.as_ref());
    for pair in &event.attributes {
        write_message(
            &mut message,
            3,
            &key_value_message(pair.key.as_str(), &pair.value),
        );
    }
    write_uint(&mut message, 4, u64::from(event.dropped_attributes_count));
    message
}

/// Encodes one `Link`.
fn link_message(link: &Link) -> Vec<u8> {
    let mut message = Vec::new();
    write_len_field(&mut message, 1, &link.span_context.trace_id().to_bytes());
    write_len_field(&mut message, 2, &link.span_context.span_id().to_bytes());
    // 3 `trace_state` is omitted, as in the JSON body.
    for pair in &link.attributes {
        write_message(
            &mut message,
            4,
            &key_value_message(pair.key.as_str(), &pair.value),
        );
    }
    write_uint(&mut message, 5, u64::from(link.dropped_attributes_count));
    // 6 `flags` is omitted, as in the JSON body.
    message
}

/// Encodes the span `Status`; `message` is only written for a described error,
/// like the JSON encoder.
fn status_message(status: &Status) -> Vec<u8> {
    let mut message = Vec::new();
    match status {
        // `STATUS_CODE_UNSET` is zero, the proto3 default.
        Status::Unset => {}
        Status::Ok => write_uint(&mut message, 3, STATUS_CODE_OK),
        Status::Error { description } => {
            write_uint(&mut message, 3, STATUS_CODE_ERROR);
            // 2 `message`.
            write_string(&mut message, 2, description.as_ref());
        }
    }
    message
}

/// Encodes one `KeyValue`.
fn key_value_message(key: &str, value: &OtelValue) -> Vec<u8> {
    let mut message = Vec::new();
    write_string(&mut message, 1, key);
    write_message(&mut message, 2, &any_value_message(value));
    message
}

/// Encodes one `AnyValue`.
///
/// `opentelemetry::Value` is `#[non_exhaustive]`, so a variant this crate does
/// not know has no wire field of its own; like the JSON encoder, it falls back
/// to its `Display` text as `string_value` instead of dropping the attribute.
/// (`Value` has no bytes or key-value-list variant, so wire fields 6 and 7 of
/// the `AnyValue` oneof are never written.)
fn any_value_message(value: &OtelValue) -> Vec<u8> {
    match value {
        OtelValue::Bool(flag) => {
            let mut message = Vec::new();
            write_bool_field(&mut message, 2, *flag);
            message
        }
        OtelValue::I64(number) => {
            let mut message = Vec::new();
            write_int64_field(&mut message, 3, *number);
            message
        }
        OtelValue::F64(number) => {
            let mut message = Vec::new();
            write_double_field(&mut message, 4, *number);
            message
        }
        OtelValue::String(text) => string_value_message(text.as_str()),
        OtelValue::Array(array) => {
            let mut message = Vec::new();
            write_message(&mut message, 5, &array_value_message(array));
            message
        }
        other => string_value_message(&other.to_string()),
    }
}

/// Encodes the `AnyValue.string_value` variant; the encoder for the textual
/// fallback is the same one.
fn string_value_message(text: &str) -> Vec<u8> {
    let mut message = Vec::new();
    write_len_field(&mut message, 1, text.as_bytes());
    message
}

/// Encodes an `Array` as `ArrayValue { values: [AnyValue, …] }`.
///
/// `Array` is `#[non_exhaustive]` too: an unknown variant yields an empty
/// `values` list, matching the JSON encoder's empty `values` array.
fn array_value_message(array: &Array) -> Vec<u8> {
    let values: Vec<OtelValue> = match array {
        Array::Bool(flags) => flags.iter().map(|flag| OtelValue::Bool(*flag)).collect(),
        Array::I64(numbers) => numbers
            .iter()
            .map(|number| OtelValue::I64(*number))
            .collect(),
        Array::F64(numbers) => numbers
            .iter()
            .map(|number| OtelValue::F64(*number))
            .collect(),
        Array::String(texts) => texts
            .iter()
            .map(|text| OtelValue::String(text.clone()))
            .collect(),
        _ => Vec::new(),
    };
    let mut message = Vec::new();
    for value in &values {
        write_message(&mut message, 1, &any_value_message(value));
    }
    message
}

/// The `SpanKind` enum value.
fn span_kind_value(kind: &SpanKind) -> u64 {
    match kind {
        SpanKind::Internal => 1,
        SpanKind::Server => 2,
        SpanKind::Client => 3,
        SpanKind::Producer => 4,
        SpanKind::Consumer => 5,
    }
}

/// Nanoseconds since the Unix epoch as a `fixed64`; pre-epoch or
/// unrepresentable times are zero, matching the JSON encoder's `"0"`.
fn unix_nanos(time: SystemTime) -> u64 {
    time.duration_since(UNIX_EPOCH)
        .ok()
        .and_then(|duration| u64::try_from(duration.as_nanos()).ok())
        .unwrap_or(0)
}

/// Appends `value` as a base-128 varint.
fn write_varint(buffer: &mut Vec<u8>, mut value: u64) {
    while value > 0x7f {
        buffer.push(low_seven_bits(value) | 0x80);
        value >>= 7;
    }
    buffer.push(low_seven_bits(value));
}

/// The low seven bits of one varint group, which always fit in a byte.
fn low_seven_bits(value: u64) -> u8 {
    u8::try_from(value & 0x7f).unwrap_or(0)
}

/// Appends a field tag: `(field_number << 3) | wire_type`.
fn write_tag(buffer: &mut Vec<u8>, field: u32, wire_type: u8) {
    write_varint(buffer, (u64::from(field) << 3) | u64::from(wire_type));
}

/// Appends a little-endian 64-bit word (`fixed64`/`double`).
fn write_fixed64(buffer: &mut Vec<u8>, value: u64) {
    buffer.extend_from_slice(&value.to_le_bytes());
}

/// Appends a little-endian 32-bit word (`fixed32`/`float`).
#[cfg(test)]
fn write_fixed32(buffer: &mut Vec<u8>, value: u32) {
    buffer.extend_from_slice(&value.to_le_bytes());
}

/// Appends a length-delimited value: a string, bytes, or a buffered
/// sub-message.
///
/// Always emitted, even when empty: message fields and `oneof` members keep
/// explicit presence, so "absent" and "present but empty" must stay
/// distinguishable on the wire.
fn write_len_field(buffer: &mut Vec<u8>, field: u32, value: &[u8]) {
    write_tag(buffer, field, WIRE_LEN);
    write_varint(buffer, value.len() as u64);
    buffer.extend_from_slice(value);
}

/// Appends a string field, skipping the empty proto3 default.
fn write_string(buffer: &mut Vec<u8>, field: u32, value: &str) {
    if !value.is_empty() {
        write_len_field(buffer, field, value.as_bytes());
    }
}

/// Appends a buffered sub-message with its computed length.
fn write_message(buffer: &mut Vec<u8>, field: u32, message: &[u8]) {
    write_len_field(buffer, field, message);
}

/// Appends a varint field, skipping the zero proto3 default.
fn write_uint(buffer: &mut Vec<u8>, field: u32, value: u64) {
    if value != 0 {
        write_tag(buffer, field, WIRE_VARINT);
        write_varint(buffer, value);
    }
}

/// Appends a `fixed64` field, skipping the zero proto3 default.
fn write_fixed64_field(buffer: &mut Vec<u8>, field: u32, value: u64) {
    if value != 0 {
        write_tag(buffer, field, WIRE_FIXED64);
        write_fixed64(buffer, value);
    }
}

/// Appends a `oneof` bool member: explicit presence, so `false` is written.
fn write_bool_field(buffer: &mut Vec<u8>, field: u32, value: bool) {
    write_tag(buffer, field, WIRE_VARINT);
    write_varint(buffer, u64::from(value));
}

/// Appends a `oneof` `int64` member: explicit presence, so `0` is written.
fn write_int64_field(buffer: &mut Vec<u8>, field: u32, value: i64) {
    write_tag(buffer, field, WIRE_VARINT);
    // `int64` is a two's-complement varint, not zig-zag, so the bit pattern is
    // reinterpreted instead of converted.
    write_varint(buffer, u64::from_ne_bytes(value.to_ne_bytes()));
}

/// Appends a `oneof` `double` member: explicit presence, so `0.0` is written.
fn write_double_field(buffer: &mut Vec<u8>, field: u32, value: f64) {
    write_tag(buffer, field, WIRE_FIXED64);
    write_fixed64(buffer, value.to_bits());
}
