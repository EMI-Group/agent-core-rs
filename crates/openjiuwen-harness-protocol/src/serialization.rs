//! Shared event wire codec (schema `"1"`).
//!
//! Mirrors the Python `openjiuwen.harness_protocol.serialization` module (v0.1.20): the single
//! source of truth for the event envelope wire shape, the payload key sets, and the schema-version
//! gate that turns a newer-schema event into an `UnknownEvent`.
//!
//! Encode always writes exactly twelve envelope keys, in this order, with every key present
//! (`null` for an absent optional id, `[]` for empty `causation_ids`):
//! `schema_version`, `event_type`, `sequence`, `timestamp`, `host_session_id`, `agent_id`,
//! `provider_session_id`, `turn_id`, `item_id`, `correlation_id`, `causation_ids`, `payload`.
//! The payload value is the payload struct's own serde shape (each payload struct already keeps
//! every key present and never skips one); an `UnknownEvent` re-encodes its raw payload verbatim.
//!
//! Decode reads the two discriminators first, rejects a malformed envelope, and then dispatches:
//! a `schema_version` other than [`EVENT_WIRE_SCHEMA_VERSION`] becomes an `UnknownEvent` *before*
//! any payload parsing, so a future schema is never partially parsed; an unknown `event_type`
//! under schema `"1"` becomes an `UnknownEvent` preserving the payload. Unknown extra envelope keys
//! are ignored (forward compatible), and unknown extra keys inside a *known* payload are ignored
//! by serde's default tolerance. The decoded envelope is validated.
//!
//! # Error wrapping
//!
//! Step 1 of Python's decode (the `event_type`/`schema_version` discriminators) and the
//! `payload` object check raise their message unwrapped. Every other envelope reader failure is
//! wrapped as `Protocol("invalid harness event wire object: {detail}")`, exactly like Python's
//! single `try` block. Validation failures of the assembled envelope propagate unwrapped, because
//! Python raises the same `ValueError` text from the dataclass.
//!
//! The `HarnessEvent` serde impls delegate here, so `serde_json::to_value(&event)` equals
//! `json_object_to_value(&harness_event_to_dict(&event)?)` and
//! `harness_event_from_dict(w)` equals `serde_json::from_value::<HarnessEvent>(json_object_to_value(&w))`
//! for every well-formed wire object.

use std::fmt::Display;

use serde_json::{Map, Number};

use crate::errors::{HarnessError, HarnessResult};
use crate::events::envelope::HarnessEvent;
use crate::events::payloads::{
    DiagnosticEvent, HarnessEventPayload, HookObservedEvent, ItemLifecycleEvent, ModelRequestEvent,
    OutputEvent, ProviderEvent, StateChangedEvent, TurnLifecycleEvent, UnknownEvent,
    UsageUpdatedEvent,
};
use crate::json::{json_object_from_value, json_object_to_value, JsonObject, JsonValue};
use crate::validation::Validate;

/// Wire schema version of the event envelope understood by this crate.
///
/// Known payloads encode with this version; a decoded event carrying any other version becomes an
/// `UnknownEvent` and is re-encoded losslessly with the version it arrived with.
pub const EVENT_WIRE_SCHEMA_VERSION: &str = "1";

/// Encodes `event` into a JSON object with the DESIGN key order.
///
/// # Errors
///
/// Returns [`HarnessError::Validation`] with `"event timestamp must be a finite Unix timestamp"`
/// when `timestamp` is not representable as a JSON number, and
/// [`HarnessError::Protocol`] when a payload fails to serialize.
pub fn harness_event_to_dict(event: &HarnessEvent) -> HarnessResult<JsonObject> {
    let value = encode_event(event)?;
    // `encode_event` always builds an object; the error below is defensive only.
    json_object_from_value(value)
        .ok_or_else(|| HarnessError::protocol("harness event must encode as a JSON object"))
}

/// Decodes an event from an already-parsed JSON object, validating it.
///
/// # Errors
///
/// Returns [`HarnessError::Protocol`] for a malformed envelope or payload (with Python's message
/// wording), and the envelope's [`HarnessError::Validation`] when an invariant is violated.
// DESIGN §6 fixes the owned-dict signature, mirroring Python's `harness_event_from_dict(data)`.
#[allow(clippy::needless_pass_by_value)]
pub fn harness_event_from_dict(data: JsonObject) -> HarnessResult<HarnessEvent> {
    decode_event(&data)
}

/// Encodes `event` as compact JSON text with the DESIGN key order.
///
/// The text is structurally equal to Python's `json.dumps(harness_event_to_dict(event))`
/// (separators and key order match); byte equality across languages is not guaranteed.
///
/// # Errors
///
/// The same errors as [`harness_event_to_dict`], plus [`HarnessError::Protocol`] when the encoded
/// value cannot be rendered as JSON text.
pub fn harness_event_to_json_string(event: &HarnessEvent) -> HarnessResult<String> {
    Ok(serde_json::to_string(&encode_event(event)?)?)
}

/// Decodes an event from JSON text, validating it.
///
/// The Rust API also accepts text, which Python only exposed through its dict form: a top-level
/// JSON value that is not an object is rejected with
/// `"harness event JSON must be a JSON object"`.
///
/// # Errors
///
/// Returns [`HarnessError::Protocol`] when `text` is not JSON, when the top-level value is not an
/// object, for a malformed envelope or payload, and the envelope's
/// [`HarnessError::Validation`] when an invariant is violated.
pub fn harness_event_from_json_str(text: &str) -> HarnessResult<HarnessEvent> {
    let value: JsonValue = serde_json::from_str(text)?;
    let object = json_object_from_value(value)
        .ok_or_else(|| HarnessError::protocol("harness event JSON must be a JSON object"))?;
    decode_event(&object)
}

/// Builds the wire JSON value for `event`; the codec entry point used by the manual `Serialize`.
///
/// # Errors
///
/// Returns [`HarnessError::Validation`] with `"event timestamp must be a finite Unix timestamp"`
/// when `timestamp` is not representable as a JSON number. A non-finite timestamp cannot be
/// represented in JSON at all; Python never encodes such an event because its constructor already
/// rejects it, so this mirrors the envelope rule rather than adding a new one.
/// Returns [`HarnessError::Protocol`] when a payload fails to serialize.
pub(crate) fn encode_event(event: &HarnessEvent) -> HarnessResult<JsonValue> {
    let timestamp = Number::from_f64(event.timestamp).ok_or_else(|| {
        HarnessError::validation("event timestamp must be a finite Unix timestamp")
    })?;
    let payload = encode_payload(&event.event)?;

    let mut object = Map::new();
    object.insert(
        "schema_version".to_owned(),
        JsonValue::String(event.event.wire_schema_version().to_owned()),
    );
    object.insert(
        "event_type".to_owned(),
        JsonValue::String(event.event.wire_event_type().to_owned()),
    );
    object.insert("sequence".to_owned(), JsonValue::from(event.sequence));
    object.insert("timestamp".to_owned(), JsonValue::Number(timestamp));
    object.insert(
        "host_session_id".to_owned(),
        JsonValue::String(event.host_session_id.clone()),
    );
    object.insert(
        "agent_id".to_owned(),
        JsonValue::String(event.agent_id.clone()),
    );
    object.insert(
        "provider_session_id".to_owned(),
        optional_string_value(event.provider_session_id.as_deref()),
    );
    object.insert(
        "turn_id".to_owned(),
        optional_string_value(event.turn_id.as_deref()),
    );
    object.insert(
        "item_id".to_owned(),
        optional_string_value(event.item_id.as_deref()),
    );
    object.insert(
        "correlation_id".to_owned(),
        optional_string_value(event.correlation_id.as_deref()),
    );
    object.insert(
        "causation_ids".to_owned(),
        JsonValue::Array(
            event
                .causation_ids
                .iter()
                .cloned()
                .map(JsonValue::String)
                .collect(),
        ),
    );
    object.insert("payload".to_owned(), payload);
    Ok(JsonValue::Object(object))
}

/// Decodes an event from a wire object, validating it; the codec entry point used by the manual
/// `Deserialize`.
///
/// # Errors
///
/// See [`harness_event_from_dict`].
pub(crate) fn decode_event(data: &JsonObject) -> HarnessResult<HarnessEvent> {
    // DESIGN §6 step 1: the discriminators are read first and unwrapped.
    let event_type = required_string(data, "event_type").map_err(HarnessError::Protocol)?;
    let schema_version = required_string(data, "schema_version").map_err(HarnessError::Protocol)?;
    // DESIGN §6: a missing or non-object `payload` is unwrapped too.
    let payload = payload_object(data)?;
    // DESIGN §6 steps 4-6: everything else is wrapped into one protocol message.
    let causation_ids = causation_ids(data).map_err(invalid)?;
    let sequence = required_u64(data, "sequence").map_err(invalid)?;
    let timestamp = required_f64(data, "timestamp").map_err(invalid)?;
    let host_session_id = required_string(data, "host_session_id").map_err(invalid)?;
    let agent_id = required_string(data, "agent_id").map_err(invalid)?;
    let provider_session_id = optional_string(data, "provider_session_id").map_err(invalid)?;
    let turn_id = optional_string(data, "turn_id").map_err(invalid)?;
    let item_id = optional_string(data, "item_id").map_err(invalid)?;
    let correlation_id = optional_string(data, "correlation_id").map_err(invalid)?;

    let event = HarnessEvent {
        sequence,
        timestamp,
        event: decode_payload(&event_type, &schema_version, payload)?,
        host_session_id,
        agent_id,
        provider_session_id,
        turn_id,
        item_id,
        correlation_id,
        causation_ids,
    };
    // DESIGN §6 step 6: validation errors propagate unwrapped.
    event.validate()?;
    Ok(event)
}

/// Wraps a reader detail in Python's single envelope-decoding protocol message.
fn invalid(detail: impl Display) -> HarnessError {
    HarnessError::protocol(format!("invalid harness event wire object: {detail}"))
}

/// Encodes the payload value for its variant.
fn encode_payload(event: &HarnessEventPayload) -> HarnessResult<JsonValue> {
    let value = match event {
        HarnessEventPayload::Output(payload) => serde_json::to_value(payload)?,
        HarnessEventPayload::ItemLifecycle(payload) => serde_json::to_value(payload)?,
        HarnessEventPayload::UsageUpdated(payload) => serde_json::to_value(payload)?,
        HarnessEventPayload::ModelRequest(payload) => serde_json::to_value(payload)?,
        HarnessEventPayload::StateChanged(payload) => serde_json::to_value(payload)?,
        HarnessEventPayload::TurnLifecycle(payload) => serde_json::to_value(payload)?,
        HarnessEventPayload::HookObserved(payload) => serde_json::to_value(payload)?,
        HarnessEventPayload::Diagnostic(payload) => serde_json::to_value(payload)?,
        HarnessEventPayload::Provider(payload) => serde_json::to_value(payload)?,
        // The unknown event's payload is preserved verbatim so a re-encode is lossless.
        HarnessEventPayload::Unknown(payload) => json_object_to_value(&payload.payload),
    };
    Ok(value)
}

/// Dispatches a wire payload to its type, applying the schema gate before any parsing.
fn decode_payload(
    event_type: &str,
    schema_version: &str,
    payload: JsonObject,
) -> HarnessResult<HarnessEventPayload> {
    // Schema gate first: never partially parse a newer schema.
    if schema_version != EVENT_WIRE_SCHEMA_VERSION {
        return Ok(UnknownEvent {
            event_type: event_type.to_owned(),
            schema_version: schema_version.to_owned(),
            payload,
        }
        .into());
    }
    let value = json_object_to_value(&payload);
    let result = match event_type {
        "output" => serde_json::from_value::<OutputEvent>(value).map(HarnessEventPayload::Output),
        "item_lifecycle" => serde_json::from_value::<ItemLifecycleEvent>(value)
            .map(HarnessEventPayload::ItemLifecycle),
        "usage_updated" => serde_json::from_value::<UsageUpdatedEvent>(value)
            .map(HarnessEventPayload::UsageUpdated),
        "model_request" => serde_json::from_value::<ModelRequestEvent>(value)
            .map(HarnessEventPayload::ModelRequest),
        "state_changed" => serde_json::from_value::<StateChangedEvent>(value)
            .map(HarnessEventPayload::StateChanged),
        "turn_lifecycle" => serde_json::from_value::<TurnLifecycleEvent>(value)
            .map(HarnessEventPayload::TurnLifecycle),
        "hook_observed" => serde_json::from_value::<HookObservedEvent>(value)
            .map(HarnessEventPayload::HookObserved),
        "diagnostic" => {
            serde_json::from_value::<DiagnosticEvent>(value).map(HarnessEventPayload::Diagnostic)
        }
        "provider" => {
            serde_json::from_value::<ProviderEvent>(value).map(HarnessEventPayload::Provider)
        }
        _ => Ok(UnknownEvent {
            event_type: event_type.to_owned(),
            schema_version: schema_version.to_owned(),
            payload,
        }
        .into()),
    };
    result.map_err(invalid)
}

/// Maps an optional string field to its wire value.
fn optional_string_value(value: Option<&str>) -> JsonValue {
    value.map_or(JsonValue::Null, |text| JsonValue::String(text.to_owned()))
}

/// Reads the `payload` object; a missing or non-object value is Python's unwrapped error.
fn payload_object(data: &JsonObject) -> HarnessResult<JsonObject> {
    match data.get("payload") {
        Some(JsonValue::Object(object)) => Ok(object
            .iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect()),
        _ => Err(HarnessError::protocol("payload must be a JSON object")),
    }
}

/// Reads `causation_ids`: absent means `[]`, otherwise a JSON array of non-empty strings.
fn causation_ids(data: &JsonObject) -> Result<Vec<String>, String> {
    let Some(value) = data.get("causation_ids") else {
        return Ok(Vec::new());
    };
    let JsonValue::Array(values) = value else {
        return Err("causation_ids must be a JSON array".to_owned());
    };
    values
        .iter()
        .map(|entry| match entry {
            JsonValue::String(text) if !text.is_empty() => Ok(text.clone()),
            _ => Err("causation_id must be a non-empty string".to_owned()),
        })
        .collect()
}

/// Reads a required non-empty string field.
fn required_string(data: &JsonObject, field: &str) -> Result<String, String> {
    match data.get(field) {
        Some(JsonValue::String(text)) if !text.is_empty() => Ok(text.clone()),
        _ => Err(format!("{field} must be a non-empty string")),
    }
}

/// Reads an optional string field: absent or `null` are `None`, anything else must be a non-empty
/// string.
fn optional_string(data: &JsonObject, field: &str) -> Result<Option<String>, String> {
    match data.get(field) {
        None | Some(JsonValue::Null) => Ok(None),
        Some(_) => required_string(data, field).map(Some),
    }
}

/// Reads a required JSON integer field (booleans, negatives, and non-integral numbers fail).
fn required_u64(data: &JsonObject, field: &str) -> Result<u64, String> {
    data.get(field)
        .and_then(JsonValue::as_u64)
        .ok_or_else(|| format!("{field} must be an integer"))
}

/// Reads a required JSON number field as `f64` (booleans fail, integers are accepted).
fn required_f64(data: &JsonObject, field: &str) -> Result<f64, String> {
    data.get(field)
        .and_then(JsonValue::as_f64)
        .ok_or_else(|| format!("{field} must be a number"))
}

#[cfg(test)]
mod tests {
    // Tests assert by panicking on failure, which is the intent of a test.
    #![allow(clippy::expect_used, clippy::unwrap_used)]

    use serde_json::json;

    use super::{
        harness_event_from_dict, harness_event_from_json_str, harness_event_to_dict,
        harness_event_to_json_string, EVENT_WIRE_SCHEMA_VERSION,
    };
    use crate::events::envelope::HarnessEvent;
    use crate::events::payloads::{HarnessEventPayload, OutputEvent, UnknownEvent};
    use crate::json::{json_object_from_value, JsonObject, JsonValue};
    use crate::{OutputChannel, OutputKind, OutputOperation};

    /// The twelve envelope keys in DESIGN §6 order.
    const ENVELOPE_KEYS: [&str; 12] = [
        "schema_version",
        "event_type",
        "sequence",
        "timestamp",
        "host_session_id",
        "agent_id",
        "provider_session_id",
        "turn_id",
        "item_id",
        "correlation_id",
        "causation_ids",
        "payload",
    ];

    fn sample_event() -> HarnessEvent {
        HarnessEvent::new(
            7,
            1.5,
            OutputEvent {
                output_id: "out-1".to_owned(),
                kind: OutputKind::Text,
                content: json!("hi"),
                operation: OutputOperation::Snapshot,
                channel: OutputChannel::Answer,
                content_index: 0,
                content_type: None,
                data: JsonObject::new(),
            },
            "host-1",
            "agent-1",
        )
        .with_turn_id("turn-1")
        .with_causation_ids(vec!["evt-1".to_owned(), "evt-2".to_owned()])
    }

    fn wire_object(value: JsonValue) -> JsonObject {
        json_object_from_value(value).expect("object")
    }

    #[test]
    fn json_string_keeps_the_twelve_envelope_keys_in_design_order() {
        let text = harness_event_to_json_string(&sample_event()).expect("encodes");
        let parsed: JsonValue = serde_json::from_str(&text).expect("valid JSON");
        let keys: Vec<&str> = parsed
            .as_object()
            .expect("object")
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(keys, ENVELOPE_KEYS);
        assert_eq!(
            text,
            r#"{"schema_version":"1","event_type":"output","sequence":7,"timestamp":1.5,"host_session_id":"host-1","agent_id":"agent-1","provider_session_id":null,"turn_id":"turn-1","item_id":null,"correlation_id":null,"causation_ids":["evt-1","evt-2"],"payload":{"output_id":"out-1","kind":"text","content":"hi","operation":"snapshot","channel":"answer","content_index":0,"content_type":null,"data":{}}}"#
        );
    }

    #[test]
    fn dict_and_json_string_round_trip() {
        let event = sample_event();

        let dict = harness_event_to_dict(&event).expect("encodes");
        let from_dict = harness_event_from_dict(dict).expect("decodes");
        assert_eq!(from_dict, event);

        let text = harness_event_to_json_string(&event).expect("encodes");
        let from_text = harness_event_from_json_str(&text).expect("decodes");
        assert_eq!(from_text, event);

        // The manual serde impls delegate to the same codec.
        let value = serde_json::to_value(&event).expect("serializes");
        assert_eq!(
            value,
            crate::json::json_object_to_value(&harness_event_to_dict(&event).expect("encodes"))
        );
        let decoded: HarnessEvent = serde_json::from_value(value).expect("deserializes");
        assert_eq!(decoded, event);
    }

    #[test]
    fn newer_schema_re_encodes_losslessly() {
        let wire = json!({
            "schema_version": "2",
            "event_type": "future.event",
            "sequence": 3,
            "timestamp": 2.0,
            "host_session_id": "host-1",
            "agent_id": "agent-1",
            "provider_session_id": null,
            "turn_id": null,
            "item_id": null,
            "correlation_id": null,
            "causation_ids": [],
            "payload": {"alpha": 1},
        });
        let event = harness_event_from_dict(wire_object(wire)).expect("decodes");
        let HarnessEventPayload::Unknown(unknown) = &event.event else {
            panic!("a newer schema must not be parsed");
        };
        assert_eq!(
            unknown,
            &UnknownEvent {
                event_type: "future.event".to_owned(),
                schema_version: "2".to_owned(),
                payload: wire_object(json!({"alpha": 1})),
            }
        );

        let re_encoded = harness_event_to_json_string(&event).expect("re-encodes");
        let parsed: JsonValue = serde_json::from_str(&re_encoded).expect("valid JSON");
        let keys: Vec<&str> = parsed
            .as_object()
            .expect("object")
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(keys, ENVELOPE_KEYS);
        assert_eq!(parsed["schema_version"], json!("2"));
        assert_eq!(parsed["event_type"], json!("future.event"));
        assert_eq!(parsed["payload"], json!({"alpha": 1}));
    }

    #[test]
    fn known_payload_tolerates_unknown_keys() {
        let wire = json!({
            "schema_version": EVENT_WIRE_SCHEMA_VERSION,
            "event_type": "output",
            "sequence": 1,
            "timestamp": 0,
            "host_session_id": "host-1",
            "agent_id": "agent-1",
            "provider_session_id": null,
            "turn_id": null,
            "item_id": null,
            "correlation_id": null,
            "causation_ids": [],
            "payload": {
                "output_id": "out-1",
                "kind": "text",
                "content": null,
                "future_key": true,
            },
            "future_envelope_key": true,
        });
        let event =
            harness_event_from_json_str(&wire.to_string()).expect("unknown keys are tolerated");
        let HarnessEventPayload::Output(output) = &event.event else {
            panic!("known event_type must dispatch");
        };
        assert_eq!(output.output_id, "out-1");
        assert_eq!(output.operation, OutputOperation::Snapshot);

        let dict = harness_event_to_dict(&event).expect("encodes");
        assert!(
            !dict.contains_key("future_envelope_key"),
            "unknown envelope keys are dropped on re-encode"
        );
        let payload = dict["payload"].as_object().expect("payload object");
        assert!(!payload.contains_key("future_key"));
    }
}
