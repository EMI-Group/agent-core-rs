//! JSON aliases and conversion helpers.
//!
//! [`JsonValue`] is [`serde_json::Value`]; [`JsonObject`] is a
//! [`std::collections::BTreeMap`] so key order is
//! deterministic (the workspace builds `serde_json` with `preserve_order`, so conversions go
//! through an explicit `collect` rather than reusing serde's map type).
//!
//! Python's `freeze_json_value`/`json_value_to_builtin` have no Rust analogue: [`JsonValue`] is
//! already owned and JSON-compatible, so deep immutability is replaced by value semantics.
//!
//! Decoding follows the crate-wide serde rules: unknown JSON fields are tolerated
//! (serde's default), and a missing required field is an error — no container-level default is
//! applied to models with required fields.

use serde::de::DeserializeOwned;

use crate::errors::HarnessResult;
use crate::validation::Validate;

/// Any JSON value.
pub type JsonValue = serde_json::Value;

/// A JSON object with deterministic key order.
pub type JsonObject = std::collections::BTreeMap<String, JsonValue>;

/// Returns the object contents when `value` is a JSON object, otherwise [`None`].
#[must_use]
pub fn json_object_from_value(value: JsonValue) -> Option<JsonObject> {
    match value {
        JsonValue::Object(object) => Some(object.into_iter().collect()),
        _ => None,
    }
}

/// Converts an object into a JSON value, preserving the map's (sorted) key order.
#[must_use]
pub fn json_object_to_value(object: &JsonObject) -> JsonValue {
    JsonValue::Object(
        object
            .iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect(),
    )
}

/// Returns a new empty JSON object.
#[must_use]
pub fn empty_json_object() -> JsonObject {
    JsonObject::new()
}

/// Deserializes `T` from a JSON value and then runs its [`Validate`] rules.
///
/// Unknown JSON fields are tolerated; a missing required field is a deserialization error.
///
/// # Errors
///
/// Returns [`HarnessError::Protocol`](crate::HarnessError::Protocol) when the value does not
/// match `T`'s serde shape, or the model's own validation error when an invariant fails.
pub fn from_value_validated<T>(value: JsonValue) -> HarnessResult<T>
where
    T: DeserializeOwned + Validate,
{
    let parsed: T = serde_json::from_value(value)?;
    parsed.validate()?;
    Ok(parsed)
}

/// Deserializes `T` from JSON text and then runs its [`Validate`] rules.
///
/// Unknown JSON fields are tolerated; a missing required field is a deserialization error.
///
/// # Errors
///
/// Returns [`HarnessError::Protocol`](crate::HarnessError::Protocol) when the text does not match
/// `T`'s serde shape, or the model's own validation error when an invariant fails.
pub fn from_str_validated<T>(text: &str) -> HarnessResult<T>
where
    T: DeserializeOwned + Validate,
{
    let parsed: T = serde_json::from_str(text)?;
    parsed.validate()?;
    Ok(parsed)
}

#[cfg(test)]
mod tests {
    // Tests assert by panicking on failure, which is the intent of a test.
    #![allow(clippy::expect_used, clippy::unwrap_used)]

    use super::{
        empty_json_object, from_str_validated, json_object_from_value, json_object_to_value,
        JsonValue,
    };
    use crate::{HarnessErrorKind, HarnessResult, Validate};
    use serde::Deserialize;

    #[test]
    fn from_value_rejects_non_objects() {
        assert!(json_object_from_value(JsonValue::Null).is_none());
        assert!(json_object_from_value(JsonValue::Array(Vec::new())).is_none());
        assert!(json_object_from_value(JsonValue::String("x".to_owned())).is_none());
        assert!(json_object_from_value(JsonValue::Number(1.into())).is_none());
    }

    #[test]
    fn object_value_round_trip_is_stable() {
        let text = r#"{"b": 2, "a": {"c": [1, true, null]}}"#;
        let value: JsonValue = serde_json::from_str(text).expect("valid JSON");
        let object = json_object_from_value(value).expect("object");
        let keys: Vec<&str> = object.keys().map(String::as_str).collect();
        assert_eq!(keys, ["a", "b"], "BTreeMap keeps sorted key order");

        let value = json_object_to_value(&object);
        assert_eq!(
            serde_json::to_string(&value).expect("serializes"),
            r#"{"a":{"c":[1,true,null]},"b":2}"#
        );
        assert!(empty_json_object().is_empty());
    }

    #[derive(Debug, Deserialize)]
    struct Message {
        text: String,
    }

    impl Validate for Message {
        fn validate(&self) -> HarnessResult<()> {
            if self.text.is_empty() {
                return Err(crate::HarnessError::validation("text must not be empty"));
            }
            Ok(())
        }
    }

    #[test]
    fn decode_path_runs_validation() {
        let message: Message = from_str_validated(r#"{"text": "hi"}"#).expect("valid message");
        assert_eq!(message.text, "hi");

        let error = from_str_validated::<Message>(r#"{"text": ""}"#).expect_err("empty text");
        assert_eq!(error.kind(), HarnessErrorKind::Validation);

        let error = from_str_validated::<Message>("{}").expect_err("missing required field");
        assert_eq!(error.kind(), HarnessErrorKind::Protocol);
    }
}
