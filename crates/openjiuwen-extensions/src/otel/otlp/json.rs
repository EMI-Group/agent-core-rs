//! OTLP/JSON body encoder.
//!
//! Serializes an exported [`SpanData`] batch into the OTLP/HTTP JSON encoding of
//! `ExportTraceServiceRequest`: field names are the specification's
//! lowerCamelCase names, ids are lowercase hex, and nanosecond timestamps are
//! decimal strings (JSON numbers cannot carry 64-bit nanoseconds losslessly).
//!
//! The encoder is pure — it takes the resource the provider handed to the
//! exporter plus the batch, and returns a [`Value`] — so the exporter and the
//! tests observe exactly the payload that goes over the wire.
//!
//! The scope grouping comes from `model.rs`, shared with the protobuf encoder.

use std::time::{SystemTime, UNIX_EPOCH};

use opentelemetry::trace::{Event, SpanId, SpanKind, Status};
use opentelemetry::{Array, InstrumentationScope, KeyValue, Value as OtelValue};
use opentelemetry_sdk::trace::{SpanData, SpanEvents, SpanLinks};
use opentelemetry_sdk::Resource;
use serde_json::{Map, Value};

use super::model::{scope_groups, ScopeGroup};

/// Encodes one export batch as the OTLP/JSON request body.
///
/// Every span goes into a single `resourceSpans` entry, grouped by
/// instrumentation scope: one `scopeSpans` entry per distinct
/// `(name, version)` pair, in first-seen order, keeping the span order inside a
/// scope. Without a resource the `resource` object is emitted empty instead of
/// dropping the key, so the body shape never depends on the resource.
pub(crate) fn encode_body(resource: Option<&Resource>, spans: &[SpanData]) -> Value {
    let mut root = Map::new();
    root.insert(
        "resourceSpans".to_owned(),
        Value::Array(vec![resource_spans(resource, spans)]),
    );
    Value::Object(root)
}

/// Encodes the `resourceSpans` entry shared by every span in the batch.
fn resource_spans(resource: Option<&Resource>, spans: &[SpanData]) -> Value {
    let mut object = Map::new();
    object.insert("resource".to_owned(), resource_json(resource));
    object.insert(
        "scopeSpans".to_owned(),
        Value::Array(scope_groups(spans).iter().map(scope_group_json).collect()),
    );
    Value::Object(object)
}

/// The resource object: `attributes` only when the resource carries any.
fn resource_json(resource: Option<&Resource>) -> Value {
    let Some(resource) = resource else {
        return Value::Object(Map::new());
    };
    let attributes: Vec<Value> = resource
        .iter()
        .map(|(key, value)| key_value_json(key.as_str(), value))
        .collect();
    let mut object = Map::new();
    if !attributes.is_empty() {
        object.insert("attributes".to_owned(), Value::Array(attributes));
    }
    Value::Object(object)
}

/// Encodes one `ScopeSpans` entry from the shared scope grouping.
fn scope_group_json(group: &ScopeGroup<'_>) -> Value {
    let mut object = Map::new();
    object.insert("scope".to_owned(), scope_json(group.scope));
    object.insert(
        "spans".to_owned(),
        Value::Array(group.spans.iter().map(|span| span_json(span)).collect()),
    );
    Value::Object(object)
}

/// Encodes an instrumentation scope: name, optional version, optional
/// attributes.
fn scope_json(scope: &InstrumentationScope) -> Value {
    let mut object = Map::new();
    object.insert("name".to_owned(), Value::String(scope.name().to_owned()));
    if let Some(version) = scope.version() {
        object.insert("version".to_owned(), Value::String(version.to_owned()));
    }
    let attributes: Vec<Value> = scope
        .attributes()
        .map(|pair| key_value_json(pair.key.as_str(), &pair.value))
        .collect();
    if !attributes.is_empty() {
        object.insert("attributes".to_owned(), Value::Array(attributes));
    }
    Value::Object(object)
}

/// Encodes one exported span.
fn span_json(span: &SpanData) -> Value {
    let mut object = Map::new();
    object.insert(
        "traceId".to_owned(),
        Value::String(hex(&span.span_context.trace_id().to_bytes())),
    );
    object.insert(
        "spanId".to_owned(),
        Value::String(hex(&span.span_context.span_id().to_bytes())),
    );
    if span.parent_span_id != SpanId::INVALID {
        object.insert(
            "parentSpanId".to_owned(),
            Value::String(hex(&span.parent_span_id.to_bytes())),
        );
    }
    object.insert("name".to_owned(), Value::String(span.name.to_string()));
    object.insert(
        "kind".to_owned(),
        Value::String(span_kind_name(&span.span_kind).to_owned()),
    );
    object.insert(
        "startTimeUnixNano".to_owned(),
        Value::String(unix_nanos(span.start_time)),
    );
    object.insert(
        "endTimeUnixNano".to_owned(),
        Value::String(unix_nanos(span.end_time)),
    );
    object.insert(
        "attributes".to_owned(),
        Value::Array(attributes_json(&span.attributes)),
    );
    object.insert(
        "droppedAttributesCount".to_owned(),
        Value::from(span.dropped_attributes_count),
    );
    object.insert("events".to_owned(), events_json(&span.events));
    object.insert("links".to_owned(), links_json(&span.links));
    object.insert("status".to_owned(), status_json(&span.status));
    Value::Object(object)
}

/// Encodes an attribute list as OTLP `KeyValue` objects.
fn attributes_json(attributes: &[KeyValue]) -> Vec<Value> {
    attributes
        .iter()
        .map(|pair| key_value_json(pair.key.as_str(), &pair.value))
        .collect()
}

/// Encodes one `KeyValue` object.
fn key_value_json(key: &str, value: &OtelValue) -> Value {
    let mut object = Map::new();
    object.insert("key".to_owned(), Value::String(key.to_owned()));
    object.insert("value".to_owned(), any_value_json(value));
    Value::Object(object)
}

/// Encodes one `AnyValue`.
///
/// `opentelemetry::Value` is `#[non_exhaustive]`, so an unknown variant falls
/// back to its `Display` text as a `stringValue` instead of dropping the
/// attribute.
fn any_value_json(value: &OtelValue) -> Value {
    let mut object = Map::new();
    match value {
        OtelValue::Bool(flag) => {
            object.insert("boolValue".to_owned(), Value::Bool(*flag));
        }
        OtelValue::I64(number) => {
            object.insert("intValue".to_owned(), Value::String(number.to_string()));
        }
        OtelValue::F64(number) => {
            object.insert(
                "doubleValue".to_owned(),
                serde_json::Number::from_f64(*number).map_or(Value::Null, Value::Number),
            );
        }
        OtelValue::String(text) => {
            object.insert("stringValue".to_owned(), Value::String(text.to_string()));
        }
        OtelValue::Array(array) => {
            object.insert("arrayValue".to_owned(), array_json(array));
        }
        other => {
            object.insert("stringValue".to_owned(), Value::String(other.to_string()));
        }
    }
    Value::Object(object)
}

/// Encodes an `Array` as `{"values": [AnyValue, …]}`.
fn array_json(array: &Array) -> Value {
    let values: Vec<Value> = match array {
        Array::Bool(items) => items
            .iter()
            .map(|flag| any_value_json(&OtelValue::Bool(*flag)))
            .collect(),
        Array::I64(items) => items
            .iter()
            .map(|number| any_value_json(&OtelValue::I64(*number)))
            .collect(),
        Array::F64(items) => items
            .iter()
            .map(|number| any_value_json(&OtelValue::F64(*number)))
            .collect(),
        Array::String(items) => items
            .iter()
            .map(|text| any_value_json(&OtelValue::String(text.clone())))
            .collect(),
        // `Array` is `#[non_exhaustive]`; an unknown variant has no encoding.
        _ => Vec::new(),
    };
    let mut object = Map::new();
    object.insert("values".to_owned(), Value::Array(values));
    Value::Object(object)
}

/// Encodes the span events.
fn events_json(events: &SpanEvents) -> Value {
    Value::Array(events.events.iter().map(event_json).collect())
}

/// Encodes one span event.
fn event_json(event: &Event) -> Value {
    let mut object = Map::new();
    object.insert(
        "timeUnixNano".to_owned(),
        Value::String(unix_nanos(event.timestamp)),
    );
    object.insert("name".to_owned(), Value::String(event.name.to_string()));
    object.insert(
        "attributes".to_owned(),
        Value::Array(attributes_json(&event.attributes)),
    );
    object.insert(
        "droppedAttributesCount".to_owned(),
        Value::from(event.dropped_attributes_count),
    );
    Value::Object(object)
}

/// Encodes the span links.
fn links_json(links: &SpanLinks) -> Value {
    Value::Array(links.links.iter().map(link_json).collect())
}

/// Encodes one span link.
fn link_json(link: &opentelemetry::trace::Link) -> Value {
    let mut object = Map::new();
    object.insert(
        "traceId".to_owned(),
        Value::String(hex(&link.span_context.trace_id().to_bytes())),
    );
    object.insert(
        "spanId".to_owned(),
        Value::String(hex(&link.span_context.span_id().to_bytes())),
    );
    object.insert(
        "attributes".to_owned(),
        Value::Array(attributes_json(&link.attributes)),
    );
    object.insert(
        "droppedAttributesCount".to_owned(),
        Value::from(link.dropped_attributes_count),
    );
    Value::Object(object)
}

/// Encodes the span status; `message` is only emitted for a described error.
fn status_json(status: &Status) -> Value {
    let mut object = Map::new();
    match status {
        Status::Unset => {
            object.insert("code".to_owned(), Value::String(STATUS_UNSET.to_owned()));
        }
        Status::Ok => {
            object.insert("code".to_owned(), Value::String(STATUS_OK.to_owned()));
        }
        Status::Error { description } => {
            object.insert("code".to_owned(), Value::String(STATUS_ERROR.to_owned()));
            if !description.is_empty() {
                object.insert("message".to_owned(), Value::String(description.to_string()));
            }
        }
    }
    Value::Object(object)
}

/// The `STATUS_CODE_UNSET` wire value.
const STATUS_UNSET: &str = "STATUS_CODE_UNSET";
/// The `STATUS_CODE_OK` wire value.
const STATUS_OK: &str = "STATUS_CODE_OK";
/// The `STATUS_CODE_ERROR` wire value.
const STATUS_ERROR: &str = "STATUS_CODE_ERROR";

/// The `SpanKind` wire value.
fn span_kind_name(kind: &SpanKind) -> &'static str {
    match kind {
        SpanKind::Internal => "SPAN_KIND_INTERNAL",
        SpanKind::Server => "SPAN_KIND_SERVER",
        SpanKind::Client => "SPAN_KIND_CLIENT",
        SpanKind::Producer => "SPAN_KIND_PRODUCER",
        SpanKind::Consumer => "SPAN_KIND_CONSUMER",
    }
}

/// Encodes bytes as lowercase hexadecimal.
pub(super) fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        // Writing into a `String` cannot fail.
        let _ = write!(encoded, "{byte:02x}");
    }
    encoded
}

/// Nanoseconds since the Unix epoch as a decimal string; pre-epoch is `0`.
fn unix_nanos(time: SystemTime) -> String {
    time.duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_nanos())
        .to_string()
}
