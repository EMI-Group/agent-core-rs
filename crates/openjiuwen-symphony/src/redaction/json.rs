//! Recursive JSON redaction.
//!
//! Ports `redact_sensitive_json` and `redact_private_json` from
//! `openjiuwen.symphony.models._redaction` and
//! `openjiuwen.symphony.flow.privacy`. Recursion preserves object key order,
//! leaves every non-string scalar untouched, and stops at a fixed depth so a
//! pathological document cannot exhaust the stack.

use serde_json::{Map, Value};

use super::names::{is_sensitive_name, normalize_sensitive_name};
use super::scanner::redact_sensitive_text;
use super::REDACTED;

/// The maximum container depth the JSON redactor descends into.
///
/// Mirrors Python's `_MAX_REDACTION_DEPTH`.
const MAX_REDACTION_DEPTH: usize = 64;

/// Recursively redacts sensitive mapping values from JSON-like data.
///
/// A key that [`is_sensitive_name`] recognizes has its whole value replaced by
/// `"<redacted>"`; string leaves elsewhere are scanned by
/// [`redact_sensitive_text`]; objects and arrays are rebuilt in their original
/// key order. Containers nested deeper than 64 levels become `"<redacted>"`.
///
/// ```
/// use openjiuwen_symphony::redaction::redact_sensitive_json;
///
/// let value = serde_json::json!({"api_key": "abc", "note": "token: xyz"});
/// let redacted = redact_sensitive_json(&value);
/// assert_eq!(redacted["api_key"], "<redacted>");
/// assert_eq!(redacted["note"], "token=<redacted>");
/// ```
#[must_use]
pub fn redact_sensitive_json(value: &Value) -> Value {
    redact_at(value, &[], 0)
}

/// Recursively redacts sensitive and caller-named keys from JSON-like data.
///
/// Behaves like [`redact_sensitive_json`] and additionally replaces the value of
/// any key whose [`normalize_sensitive_name`] form appears in `private_keys`.
/// Names are compared after normalization, so `API-Key` and `api_key` match the
/// same entry.
///
/// ```
/// use openjiuwen_symphony::redaction::redact_private_json;
///
/// let value = serde_json::json!({"user_email": "a@b.c", "api_key": "abc"});
/// let redacted = redact_private_json(&value, &["user_email"]);
/// assert_eq!(redacted["user_email"], "<redacted>");
/// assert_eq!(redacted["api_key"], "<redacted>");
/// ```
#[must_use]
pub fn redact_private_json(value: &Value, private_keys: &[&str]) -> Value {
    let normalized: Vec<String> = private_keys
        .iter()
        .map(|key| normalize_sensitive_name(key))
        .filter(|key| !key.is_empty())
        .collect();
    redact_at(value, &normalized, 0)
}

/// Redacts one JSON node at `depth`.
fn redact_at(value: &Value, private_keys: &[String], depth: usize) -> Value {
    match value {
        Value::Object(map) => {
            if depth >= MAX_REDACTION_DEPTH {
                return Value::String(REDACTED.to_string());
            }
            let mut redacted = Map::with_capacity(map.len());
            for (key, item) in map {
                let replacement = if is_private_key(key, private_keys) {
                    Value::String(REDACTED.to_string())
                } else {
                    redact_at(item, private_keys, depth + 1)
                };
                redacted.insert(key.clone(), replacement);
            }
            Value::Object(redacted)
        }
        Value::Array(items) => {
            if depth >= MAX_REDACTION_DEPTH {
                return Value::String(REDACTED.to_string());
            }
            Value::Array(
                items
                    .iter()
                    .map(|item| redact_at(item, private_keys, depth + 1))
                    .collect(),
            )
        }
        Value::String(text) => Value::String(redact_sensitive_text(text)),
        other => other.clone(),
    }
}

/// Reports whether `key` names credential material or a caller-private field.
fn is_private_key(key: &str, private_keys: &[String]) -> bool {
    if is_sensitive_name(key) {
        return true;
    }
    if private_keys.is_empty() {
        return false;
    }
    private_keys.contains(&normalize_sensitive_name(key))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn sensitive_keys_are_replaced_whole() {
        let value = json!({
            "api_key": "sk-live-0123456789abcdef",
            "nested": {"client_secret": {"inner": 1}},
            "count": 3,
            "flag": true,
            "nothing": null,
        });
        let redacted = redact_sensitive_json(&value);
        assert_eq!(redacted["api_key"], json!("<redacted>"));
        assert_eq!(redacted["nested"]["client_secret"], json!("<redacted>"));
        assert_eq!(redacted["count"], json!(3));
        assert_eq!(redacted["flag"], json!(true));
        assert_eq!(redacted["nothing"], Value::Null);
        assert_eq!(redacted["nested"]["client_secret"].as_object(), None);
    }

    #[test]
    fn string_leaves_are_scanned_and_arrays_recurse() {
        let value = json!({
            "items": ["token: abc", {"password": "x"}, 7, ["fine"]],
            "note": "password = hunter2",
        });
        let redacted = redact_sensitive_json(&value);
        assert_eq!(redacted["note"], json!("password=<redacted>"));
        assert_eq!(redacted["items"][1]["password"], json!("<redacted>"));
        assert_eq!(redacted["items"][0], json!("token=<redacted>"));
        assert_eq!(redacted["items"][2], json!(7));
        assert_eq!(redacted["items"][3][0], json!("fine"));
    }

    #[test]
    fn object_key_order_and_names_are_preserved() {
        let text = r#"{"z": 1, "api_key": "x", "a": 2, "m": 3}"#;
        let value: Value = serde_json::from_str(text).expect("object");
        let redacted = redact_sensitive_json(&value);
        let keys: Vec<&str> = redacted
            .as_object()
            .expect("object")
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(keys, ["z", "api_key", "a", "m"]);
        assert_eq!(redacted["z"], json!(1));
    }

    #[test]
    fn private_keys_are_matched_after_normalization() {
        let value = json!({"API-Key": "abc", "userEmail": "a@b.c", "note": "ok"});
        let redacted = redact_private_json(&value, &["api_key", "user_email"]);
        assert_eq!(redacted["API-Key"], json!("<redacted>"));
        assert_eq!(redacted["userEmail"], json!("<redacted>"));
        assert_eq!(redacted["note"], json!("ok"));
        // Without the caller's names only the conventionally sensitive key goes.
        let plain = redact_sensitive_json(&value);
        assert_eq!(plain["API-Key"], json!("<redacted>"));
        assert_eq!(plain["userEmail"], json!("a@b.c"));
    }

    #[test]
    fn the_depth_cap_replaces_deep_values() {
        let mut value = json!("leaf");
        for _ in 0..(MAX_REDACTION_DEPTH + 4) {
            value = Value::Array(vec![value]);
        }
        let redacted = redact_sensitive_json(&value);
        assert!(redacted.to_string().contains("<redacted>"));
        let mut cursor = &redacted;
        let mut depth = 0;
        while let Some(items) = cursor.as_array() {
            cursor = &items[0];
            depth += 1;
        }
        assert_eq!(cursor, &json!("<redacted>"));
        assert_eq!(depth, MAX_REDACTION_DEPTH);
    }

    #[test]
    fn scalars_and_empty_containers_round_trip() {
        assert_eq!(redact_sensitive_json(&json!(5)), json!(5));
        assert_eq!(redact_sensitive_json(&json!(null)), Value::Null);
        assert_eq!(redact_sensitive_json(&json!([])), json!([]));
        assert_eq!(redact_sensitive_json(&json!({})), json!({}));
    }

    #[test]
    fn redaction_is_idempotent() {
        let value = json!({"note": "api_key: sk-live-0123456789abcdef", "list": ["a=b"]});
        let once = redact_sensitive_json(&value);
        assert_eq!(redact_sensitive_json(&once), once);
    }
}
