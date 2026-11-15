//! Strict JSON mapping helpers shared by the provider factory and the vendor configuration
//! decoders.
//!
//! Python's `from_mapping` constructors validate field types and reject unknown keys; Rust's serde
//! decoders tolerate unknown keys, so the strict path goes through these helpers instead. Every
//! helper accepts an absent key or a JSON `null` as "not configured" and never coerces types:
//! a boolean is not a number, a number is not a string, and a negative number is not a
//! non-negative one. Error messages use the Python wording (`"{key} must be a string"`,
//! `"unknown Codex configuration fields: foo, zeta"`).

// These helpers are reached only through the feature-gated configuration decoders (`vendor` for the
// Claude Code, Codex and DSH configurations, `native-v2` for the team provider), so a build with
// those features compiled out leaves some of them unused.
#![allow(dead_code)]

use std::collections::BTreeMap;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_harness_protocol::{JsonObject, JsonValue};

/// Returns `value` as a [`JsonObject`].
///
/// # Errors
///
/// Returns [`Error::Validation`] with `"{what} must be a JSON object"` when `value` is not a JSON
/// object.
pub(crate) fn expect_object(value: &JsonValue, what: &str) -> Result<JsonObject> {
    match value.as_object() {
        Some(object) => Ok(object
            .iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect()),
        None => Err(Error::validation(format!("{what} must be a JSON object"))),
    }
}

/// Rejects keys of `object` that are not listed in `allowed`.
///
/// The unknown keys are reported sorted and comma-separated after `prefix`, mirroring Python's
/// `f"{prefix}: {', '.join(sorted(unknown))}"` (for example
/// `"unknown Codex configuration fields: foo, zeta"`).
///
/// # Errors
///
/// Returns [`Error::Validation`] when at least one key is unknown.
pub(crate) fn reject_unknown_keys(
    object: &JsonObject,
    allowed: &[&str],
    prefix: &str,
) -> Result<()> {
    let mut unknown: Vec<&str> = object
        .keys()
        .filter(|key| !allowed.contains(&key.as_str()))
        .map(String::as_str)
        .collect();
    if unknown.is_empty() {
        return Ok(());
    }
    unknown.sort_unstable();
    Err(Error::validation(format!(
        "{prefix}: {}",
        unknown.join(", ")
    )))
}

/// Returns the string value of `key`.
///
/// # Errors
///
/// Returns [`Error::Validation`] with `"{key} must be a string"` when the value is not a string.
pub(crate) fn optional_str(object: &JsonObject, key: &str) -> Result<Option<String>> {
    match configured(object, key) {
        None => Ok(None),
        Some(value) => match value.as_str() {
            Some(text) => Ok(Some(text.to_owned())),
            None => Err(invalid(key, "a string")),
        },
    }
}

/// Returns the boolean value of `key`.
///
/// # Errors
///
/// Returns [`Error::Validation`] with `"{key} must be a boolean"` when the value is not a boolean.
pub(crate) fn optional_bool(object: &JsonObject, key: &str) -> Result<Option<bool>> {
    match configured(object, key) {
        None => Ok(None),
        Some(value) => match value.as_bool() {
            Some(flag) => Ok(Some(flag)),
            None => Err(invalid(key, "a boolean")),
        },
    }
}

/// Returns the non-negative integer value of `key`.
///
/// # Errors
///
/// Returns [`Error::Validation`] with `"{key} must be a non-negative integer"` when the value is a
/// boolean, a float, a negative number, or a number that does not fit in `u64`.
pub(crate) fn optional_u64(object: &JsonObject, key: &str) -> Result<Option<u64>> {
    match configured(object, key) {
        None => Ok(None),
        Some(value) => match value.as_u64() {
            Some(number) => Ok(Some(number)),
            None => Err(invalid(key, "a non-negative integer")),
        },
    }
}

/// Returns the non-negative integer value of `key`, checked against `u32`.
///
/// # Errors
///
/// Returns [`Error::Validation`] with `"{key} must be a non-negative integer"` when the value is not
/// an integer of `0..=u32::MAX` (booleans, floats and negatives are rejected).
pub(crate) fn optional_u32(object: &JsonObject, key: &str) -> Result<Option<u32>> {
    match configured(object, key) {
        None => Ok(None),
        Some(value) => match value.as_u64().and_then(|number| u32::try_from(number).ok()) {
            Some(number) => Ok(Some(number)),
            None => Err(invalid(key, "a non-negative integer")),
        },
    }
}

/// Returns the non-negative finite number value of `key`.
///
/// JSON integers decode as `f64`. Booleans, negative numbers and non-finite numbers are rejected.
///
/// # Errors
///
/// Returns [`Error::Validation`] with `"{key} must be a non-negative number"` when the value is not
/// a non-negative finite number.
pub(crate) fn optional_f64(object: &JsonObject, key: &str) -> Result<Option<f64>> {
    match configured(object, key) {
        None => Ok(None),
        Some(value) => match value.as_f64() {
            Some(number) if number.is_finite() && number >= 0.0 => Ok(Some(number)),
            _ => Err(invalid(key, "a non-negative number")),
        },
    }
}

/// Returns the string-array value of `key`.
///
/// # Errors
///
/// Returns [`Error::Validation`] with `"{key} must be an array of strings"` when the value is not an
/// array whose elements are all strings.
pub(crate) fn optional_string_list(object: &JsonObject, key: &str) -> Result<Option<Vec<String>>> {
    match configured(object, key) {
        None => Ok(None),
        Some(value) => {
            let Some(items) = value.as_array() else {
                return Err(invalid(key, "an array of strings"));
            };
            let mut texts = Vec::with_capacity(items.len());
            for item in items {
                let Some(text) = item.as_str() else {
                    return Err(invalid(key, "an array of strings"));
                };
                texts.push(text.to_owned());
            }
            Ok(Some(texts))
        }
    }
}

/// Returns the string-to-string map value of `key`.
///
/// # Errors
///
/// Returns [`Error::Validation`] with `"{key} must be an object of string values"` when the value is
/// not an object whose values are all strings.
pub(crate) fn optional_string_map(
    object: &JsonObject,
    key: &str,
) -> Result<Option<BTreeMap<String, String>>> {
    match configured(object, key) {
        None => Ok(None),
        Some(value) => {
            let Some(entries) = value.as_object() else {
                return Err(invalid(key, "an object of string values"));
            };
            let mut mapped = BTreeMap::new();
            for (name, entry) in entries {
                let Some(text) = entry.as_str() else {
                    return Err(invalid(key, "an object of string values"));
                };
                mapped.insert(name.clone(), text.to_owned());
            }
            Ok(Some(mapped))
        }
    }
}

/// Returns the configured value of `key`, treating an absent key and JSON `null` alike.
fn configured<'a>(object: &'a JsonObject, key: &str) -> Option<&'a JsonValue> {
    match object.get(key) {
        None | Some(JsonValue::Null) => None,
        Some(value) => Some(value),
    }
}

/// Builds the validation error for a key with the wrong type.
fn invalid(key: &str, expected: &str) -> Error {
    Error::validation(format!("{key} must be {expected}"))
}

#[cfg(test)]
mod tests {
    // Tests assert by panicking on failure, which is the intent of a test.
    #![allow(clippy::expect_used, clippy::unwrap_used)]

    use std::collections::BTreeMap;

    use openjiuwen_common::error::Error;
    use openjiuwen_harness_protocol::JsonValue;
    use serde_json::json;

    use super::{
        expect_object, optional_bool, optional_f64, optional_str, optional_string_list,
        optional_string_map, optional_u32, optional_u64, reject_unknown_keys,
    };

    fn object(value: &JsonValue) -> openjiuwen_harness_protocol::JsonObject {
        expect_object(value, "configuration").expect("valid object")
    }

    fn message(error: &Error) -> String {
        error.to_string()
    }

    #[test]
    fn expect_object_accepts_objects_sorted_by_key() {
        let parsed = object(&json!({"zeta": 1, "alpha": 2}));
        let keys: Vec<&str> = parsed.keys().map(String::as_str).collect();
        assert_eq!(keys, ["alpha", "zeta"]);
    }

    #[test]
    fn expect_object_rejects_non_objects_with_the_what_prefix() {
        for value in [
            json!(1),
            json!("x"),
            json!([1]),
            JsonValue::Null,
            json!(true),
        ] {
            let error = expect_object(&value, "Harness configuration").unwrap_err();
            assert!(matches!(error, Error::Validation(_)));
            assert_eq!(
                message(&error),
                "validation: Harness configuration must be a JSON object"
            );
        }
    }

    #[test]
    fn reject_unknown_keys_accepts_allowed_and_empty_objects() {
        let empty = object(&json!({}));
        reject_unknown_keys(&empty, &["model"], "unknown configuration fields").expect("no keys");
        let complete = object(&json!({"model": "m", "api_key": "k"}));
        reject_unknown_keys(
            &complete,
            &["model", "api_key"],
            "unknown configuration fields",
        )
        .expect("all allowed");
    }

    #[test]
    fn reject_unknown_keys_reports_unknown_keys_sorted() {
        let incomplete = object(&json!({"zeta": 1, "foo": 2, "model": "m"}));
        let error = reject_unknown_keys(
            &incomplete,
            &["model"],
            "unknown Codex configuration fields",
        )
        .unwrap_err();
        assert!(matches!(error, Error::Validation(_)));
        assert_eq!(
            message(&error),
            "validation: unknown Codex configuration fields: foo, zeta"
        );
    }

    #[test]
    fn optional_str_handles_absent_null_and_valid_values() {
        let empty = object(&json!({}));
        assert_eq!(optional_str(&empty, "model").expect("absent"), None);
        let null = object(&json!({"model": null}));
        assert_eq!(optional_str(&null, "model").expect("null"), None);
        let valid = object(&json!({"model": "gpt-5"}));
        assert_eq!(
            optional_str(&valid, "model").expect("string"),
            Some("gpt-5".to_owned())
        );
    }

    #[test]
    fn optional_str_rejects_non_strings() {
        for value in [json!(1), json!(true), json!(["m"]), json!({"m": 1})] {
            let config = object(&json!({"model": value}));
            let error = optional_str(&config, "model").unwrap_err();
            assert_eq!(message(&error), "validation: model must be a string");
        }
    }

    #[test]
    fn optional_bool_handles_absent_null_and_valid_values() {
        let empty = object(&json!({}));
        assert_eq!(optional_bool(&empty, "flag").expect("absent"), None);
        let null = object(&json!({"flag": null}));
        assert_eq!(optional_bool(&null, "flag").expect("null"), None);
        let valid = object(&json!({"flag": false}));
        assert_eq!(optional_bool(&valid, "flag").expect("bool"), Some(false));
    }

    #[test]
    fn optional_bool_rejects_non_booleans_without_coercion() {
        for value in [json!(1), json!(0), json!("true")] {
            let config = object(&json!({"flag": value}));
            let error = optional_bool(&config, "flag").unwrap_err();
            assert_eq!(message(&error), "validation: flag must be a boolean");
        }
    }

    #[test]
    fn optional_u64_handles_absent_null_and_valid_values() {
        let empty = object(&json!({}));
        assert_eq!(optional_u64(&empty, "limit").expect("absent"), None);
        let null = object(&json!({"limit": null}));
        assert_eq!(optional_u64(&null, "limit").expect("null"), None);
        let valid = object(&json!({"limit": 0}));
        assert_eq!(optional_u64(&valid, "limit").expect("integer"), Some(0));
        let large = object(&json!({"limit": 9_007_199_254_740_993_u64}));
        assert_eq!(
            optional_u64(&large, "limit").expect("u64"),
            Some(9_007_199_254_740_993)
        );
    }

    #[test]
    fn optional_u64_rejects_negative_bool_and_float() {
        for value in [json!(-1), json!(true), json!(1.5), json!("5")] {
            let config = object(&json!({"limit": value}));
            let error = optional_u64(&config, "limit").unwrap_err();
            assert_eq!(
                message(&error),
                "validation: limit must be a non-negative integer"
            );
        }
    }

    #[test]
    fn optional_u32_handles_absent_null_and_valid_values() {
        let empty = object(&json!({}));
        assert_eq!(optional_u32(&empty, "capacity").expect("absent"), None);
        let null = object(&json!({"capacity": null}));
        assert_eq!(optional_u32(&null, "capacity").expect("null"), None);
        let valid = object(&json!({"capacity": u32::MAX}));
        assert_eq!(
            optional_u32(&valid, "capacity").expect("u32"),
            Some(u32::MAX)
        );
    }

    #[test]
    fn optional_u32_rejects_overflow_negative_bool_and_float() {
        let overflow = u64::from(u32::MAX) + 1;
        for value in [json!(overflow), json!(-1), json!(true), json!(2.0)] {
            let config = object(&json!({"capacity": value}));
            let error = optional_u32(&config, "capacity").unwrap_err();
            assert_eq!(
                message(&error),
                "validation: capacity must be a non-negative integer"
            );
        }
    }

    #[test]
    fn optional_f64_handles_absent_null_and_valid_values() {
        let empty = object(&json!({}));
        assert_eq!(optional_f64(&empty, "temperature").expect("absent"), None);
        let null = object(&json!({"temperature": null}));
        assert_eq!(optional_f64(&null, "temperature").expect("null"), None);
        let valid = object(&json!({"temperature": 0.25}));
        assert_eq!(
            optional_f64(&valid, "temperature").expect("float"),
            Some(0.25)
        );
        let integer = object(&json!({"temperature": 1}));
        assert_eq!(
            optional_f64(&integer, "temperature").expect("int"),
            Some(1.0)
        );
    }

    #[test]
    fn optional_f64_rejects_negative_bool_and_string() {
        for value in [json!(-0.5), json!(true), json!("0.5"), json!([1.0])] {
            let config = object(&json!({"temperature": value}));
            let error = optional_f64(&config, "temperature").unwrap_err();
            assert_eq!(
                message(&error),
                "validation: temperature must be a non-negative number"
            );
        }
    }

    #[test]
    fn optional_string_list_handles_absent_null_and_valid_values() {
        let empty = object(&json!({}));
        assert_eq!(optional_string_list(&empty, "args").expect("absent"), None);
        let null = object(&json!({"args": null}));
        assert_eq!(optional_string_list(&null, "args").expect("null"), None);
        let valid = object(&json!({"args": ["--a", "b"]}));
        assert_eq!(
            optional_string_list(&valid, "args").expect("list"),
            Some(vec!["--a".to_owned(), "b".to_owned()])
        );
        let empty_list = object(&json!({"args": []}));
        assert_eq!(
            optional_string_list(&empty_list, "args").expect("empty list"),
            Some(Vec::new())
        );
    }

    #[test]
    fn optional_string_list_rejects_non_string_elements_and_scalars() {
        for value in [json!("--a"), json!(["--a", 2]), json!({"a": "b"}), json!(3)] {
            let config = object(&json!({"args": value}));
            let error = optional_string_list(&config, "args").unwrap_err();
            assert_eq!(
                message(&error),
                "validation: args must be an array of strings"
            );
        }
    }

    #[test]
    fn optional_string_map_handles_absent_null_and_valid_values() {
        let empty = object(&json!({}));
        assert_eq!(optional_string_map(&empty, "env").expect("absent"), None);
        let null = object(&json!({"env": null}));
        assert_eq!(optional_string_map(&null, "env").expect("null"), None);
        let valid = object(&json!({"env": {"ZED": "1", "ALPHA": "2"}}));
        let mapped = optional_string_map(&valid, "env")
            .expect("map")
            .expect("configured");
        let expected: BTreeMap<String, String> = [
            ("ALPHA".to_owned(), "2".to_owned()),
            ("ZED".to_owned(), "1".to_owned()),
        ]
        .into_iter()
        .collect();
        assert_eq!(mapped, expected);
    }

    #[test]
    fn optional_string_map_rejects_non_string_values_and_non_objects() {
        for value in [json!({"A": 1}), json!(["A"]), json!("A=1")] {
            let config = object(&json!({"env": value}));
            let error = optional_string_map(&config, "env").unwrap_err();
            assert_eq!(
                message(&error),
                "validation: env must be an object of string values"
            );
        }
    }
}
