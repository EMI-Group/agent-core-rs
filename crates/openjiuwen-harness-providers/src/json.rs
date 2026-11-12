//! JSON projections shared by the registry, provider adapters and the IO adapter.
//!
//! [`to_json_safe`] is Python's `to_json_safe` analogue; [`to_json_object`] is Python's
//! `to_json_object` analogue. Both keep the JSON vocabulary of
//! [`openjiuwen_harness_protocol`]: [`JsonValue`] is [`serde_json::Value`] and [`JsonObject`] is a
//! [`std::collections::BTreeMap`] with deterministic key order.

use openjiuwen_harness_protocol::{JsonObject, JsonValue};

/// Returns a JSON-safe copy of `value`.
///
/// In Python this walks the payload to coerce dataclasses, enums and sets into builtins. Rust
/// values are already JSON-compatible by construction, so this is an identity that clones the
/// value (documented deviation; no conversion is ever needed).
#[must_use]
pub fn to_json_safe(value: &JsonValue) -> JsonValue {
    value.clone()
}

/// Returns `value` as a [`JsonObject`].
///
/// A JSON object is cloned into a [`JsonObject`] (sorted, deterministic key order). Any other
/// value — scalar, array or null — is wrapped as `{"value": <cloned value>}`, mirroring Python's
/// non-object fallback.
#[must_use]
pub fn to_json_object(value: &JsonValue) -> JsonObject {
    match value {
        JsonValue::Object(object) => object
            .iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect(),
        other => {
            let mut wrapped = JsonObject::new();
            wrapped.insert("value".to_owned(), other.clone());
            wrapped
        }
    }
}

#[cfg(test)]
mod tests {
    // Tests assert by panicking on failure, which is the intent of a test.
    #![allow(clippy::expect_used, clippy::unwrap_used)]

    use openjiuwen_harness_protocol::JsonValue;
    use serde_json::json;

    use super::{to_json_object, to_json_safe};

    #[test]
    fn to_json_safe_is_an_identity_clone() {
        let value = json!({"a": [1, {"b": null}], "c": true});
        assert_eq!(to_json_safe(&value), value);
    }

    #[test]
    fn to_json_safe_handles_scalars() {
        assert_eq!(
            to_json_safe(&json!("text")),
            JsonValue::String("text".to_owned())
        );
        assert_eq!(to_json_safe(&JsonValue::Null), JsonValue::Null);
    }

    #[test]
    fn to_json_object_passes_objects_through() {
        let object = to_json_object(&json!({"zeta": 1, "alpha": "two"}));
        let keys: Vec<&str> = object.keys().map(String::as_str).collect();
        assert_eq!(keys, ["alpha", "zeta"]);
        assert_eq!(object.get("alpha"), Some(&json!("two")));
    }

    #[test]
    fn to_json_object_wraps_scalars_under_value() {
        assert_eq!(
            to_json_object(&json!("text")).get("value"),
            Some(&json!("text"))
        );
        assert_eq!(to_json_object(&json!(7)).get("value"), Some(&json!(7)));
        assert_eq!(
            to_json_object(&JsonValue::Null).get("value"),
            Some(&JsonValue::Null)
        );
    }

    #[test]
    fn to_json_object_wraps_arrays_under_value() {
        let object = to_json_object(&json!([1, 2, 3]));
        assert_eq!(object.len(), 1);
        assert_eq!(object.get("value"), Some(&json!([1, 2, 3])));
    }

    #[test]
    fn to_json_object_preserves_nested_values() {
        let object = to_json_object(&json!({"outer": {"inner": [true, {"deep": null}]}}));
        assert_eq!(
            object.get("outer"),
            Some(&json!({"inner": [true, {"deep": null}]}))
        );
    }
}
