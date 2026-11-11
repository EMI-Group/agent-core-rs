//! JSON assertions with readable diffs.
//!
//! These helpers exist for test code: they report a mismatch by panicking with
//! the offending JSON-pointer path plus pretty-printed `expected:` and `actual:`
//! blocks, which is more useful than an `assert_eq!` dump of two `Value`s.
//!
//! # Why panicking is allowed here
//!
//! Every other module of this crate forbids `unwrap`/`expect`/`panic!`. This
//! module is the deliberate exception: an assertion helper has no meaningful way
//! to report a mismatch other than aborting the test, and a `Result` return
//! would be silently ignored by callers. The allows below are scoped to this
//! file only; the panics carry all diagnostics (path, expected, actual).
#![allow(clippy::panic, clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeSet;
use std::fmt::Debug;

use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::Value;

/// Asserts that two JSON values are equal, ignoring object key order.
///
/// Panics with the first differing path and pretty-printed `expected:` and
/// `actual:` blocks.
pub fn assert_json_eq(actual: &Value, expected: &Value) {
    if let Some(path) = first_difference(actual, expected) {
        panic!(
            "JSON values differ at `{}`\nexpected:\n{}\nactual:\n{}",
            display_path(&path),
            json_pretty(expected),
            json_pretty(actual)
        );
    }
}

/// Asserts that `actual` contains everything `expected` describes.
///
/// Objects match when every key of `expected` is present in `actual` and its
/// value is a subset; extra keys in `actual` are allowed. Arrays must have the
/// same length and match element-wise (a subset check per element). Anything
/// else must be equal. Panics naming the offending path.
pub fn assert_json_subset(actual: &Value, expected: &Value) {
    if let Some(path) = subset_difference(actual, expected, "") {
        panic!(
            "JSON subset mismatch at `{}`\nexpected (subset):\n{}\nactual:\n{}",
            display_path(&path),
            json_pretty(expected),
            json_pretty(actual)
        );
    }
}

/// Asserts that the value behind `pointer` equals `expected`.
///
/// Panics — naming the pointer and both values — when the pointer does not
/// resolve or the values differ.
pub fn assert_json_pointer_eq(actual: &Value, pointer: &str, expected: &Value) {
    let Some(found) = json_at(actual, pointer) else {
        panic!(
            "JSON pointer `{pointer}` did not resolve in:\n{}",
            json_pretty(actual)
        );
    };
    if let Some(path) = first_difference(found, expected) {
        panic!(
            "JSON value at pointer `{pointer}` differs at `{}`\nexpected:\n{}\nactual:\n{}",
            display_path(&path),
            json_pretty(expected),
            json_pretty(found)
        );
    }
}

/// Asserts that `value` survives a JSON serialize/deserialize round trip.
///
/// This is the check every public data type of the rewrite must pass; the panic
/// message shows the serialized JSON so a lossy field is easy to spot.
pub fn assert_json_roundtrip<T>(value: &T)
where
    T: Serialize + DeserializeOwned + PartialEq + Debug,
{
    let text = match serde_json::to_string_pretty(value) {
        Ok(text) => text,
        Err(error) => panic!("value could not be serialized: {error}\nvalue: {value:?}"),
    };
    let decoded: T = match serde_json::from_str(&text) {
        Ok(decoded) => decoded,
        Err(error) => panic!("serialized JSON could not be deserialized: {error}\nJSON:\n{text}"),
    };
    assert!(
        decoded == *value,
        "value did not round-trip through JSON\nJSON:\n{text}\nexpected: {value:?}\nactual: {decoded:?}"
    );
}

/// Resolves an RFC 6901 JSON pointer, if it names a value.
///
/// The empty pointer resolves to `value` itself; `"/a/0/b"` walks objects and
/// arrays; a pointer that is malformed or names a missing member yields `None`.
pub fn json_at<'a>(value: &'a Value, pointer: &str) -> Option<&'a Value> {
    value.pointer(pointer)
}

/// Pretty-prints a JSON value for assertion messages.
///
/// Falls back to the compact rendering when pretty-printing fails (only a
/// non-string map key can trigger that, which `Value` cannot hold).
pub fn json_pretty(value: &Value) -> String {
    serde_json::to_string_pretty(value).unwrap_or_else(|_| value.to_string())
}

/// Returns the path of the first difference between two JSON values.
///
/// Object key order never matters: keys are compared in sorted order. A missing
/// key is reported at the key's path, an array length mismatch at the array's
/// path, and a type or scalar mismatch at the scalar's path. Equal values yield
/// `None`, and the root path is the empty string.
pub fn first_difference(actual: &Value, expected: &Value) -> Option<String> {
    difference(actual, expected, "")
}

/// Recursive core of [`first_difference`].
fn difference(actual: &Value, expected: &Value, path: &str) -> Option<String> {
    match (actual, expected) {
        (Value::Object(actual), Value::Object(expected)) => {
            let keys: BTreeSet<&String> = actual.keys().chain(expected.keys()).collect();
            for key in keys {
                let child = format!("{path}/{}", escape_token(key));
                match (actual.get(key), expected.get(key)) {
                    (Some(actual), Some(expected)) => {
                        if let Some(found) = difference(actual, expected, &child) {
                            return Some(found);
                        }
                    }
                    // Missing on either side: the key path itself is the diff.
                    _ => return Some(child),
                }
            }
            None
        }
        (Value::Array(actual), Value::Array(expected)) => {
            if actual.len() != expected.len() {
                return Some(path.to_owned());
            }
            for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                if let Some(found) = difference(actual, expected, &format!("{path}/{index}")) {
                    return Some(found);
                }
            }
            None
        }
        _ => {
            if actual == expected {
                None
            } else {
                Some(path.to_owned())
            }
        }
    }
}

/// Recursive core of [`assert_json_subset`], returning the first offending path.
fn subset_difference(actual: &Value, expected: &Value, path: &str) -> Option<String> {
    match (actual, expected) {
        (Value::Object(actual), Value::Object(expected)) => {
            // Sorted for deterministic failure paths.
            let keys: BTreeSet<&String> = expected.keys().collect();
            for key in keys {
                let child = format!("{path}/{}", escape_token(key));
                match (actual.get(key.as_str()), expected.get(key.as_str())) {
                    (Some(actual), Some(expected)) => {
                        if let Some(found) = subset_difference(actual, expected, &child) {
                            return Some(found);
                        }
                    }
                    _ => return Some(child),
                }
            }
            None
        }
        (Value::Array(actual), Value::Array(expected)) => {
            if actual.len() != expected.len() {
                return Some(path.to_owned());
            }
            for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                if let Some(found) = subset_difference(actual, expected, &format!("{path}/{index}"))
                {
                    return Some(found);
                }
            }
            None
        }
        _ => {
            if actual == expected {
                None
            } else {
                Some(path.to_owned())
            }
        }
    }
}

/// Escapes one JSON-pointer reference token (`~` and `/`).
fn escape_token(token: &str) -> String {
    token.replace('~', "~0").replace('/', "~1")
}

/// Renders a diff path for a panic message; the root path shows as `/`.
fn display_path(path: &str) -> &str {
    if path.is_empty() {
        "/"
    } else {
        path
    }
}

#[cfg(test)]
mod tests {
    use std::panic::{catch_unwind, AssertUnwindSafe};

    use serde::Deserialize;
    use serde_json::json;

    use super::*;

    fn catches<F: FnOnce()>(body: F) -> bool {
        catch_unwind(AssertUnwindSafe(body)).is_err()
    }

    #[derive(Debug, PartialEq, Serialize, Deserialize)]
    struct Sample {
        name: String,
        count: usize,
        tags: Vec<String>,
    }

    #[test]
    fn equal_values_have_no_difference() {
        let left = json!({"a": 1, "b": [true, null]});
        let right = json!({"b": [true, null], "a": 1});

        assert_eq!(first_difference(&left, &right), None);
        assert_json_eq(&left, &right);
    }

    #[test]
    fn nested_object_value_mismatch_reports_the_path() {
        let actual = json!({"choices": [{"message": {"content": "actual"}}]});
        let expected = json!({"choices": [{"message": {"content": "expected"}}]});

        assert_eq!(
            first_difference(&actual, &expected).as_deref(),
            Some("/choices/0/message/content")
        );
    }

    #[test]
    fn missing_keys_report_the_key_path() {
        let actual = json!({"a": 1});
        let expected = json!({"a": 1, "b": 2});

        assert_eq!(first_difference(&actual, &expected).as_deref(), Some("/b"));
        assert_eq!(first_difference(&expected, &actual).as_deref(), Some("/b"));
    }

    #[test]
    fn array_length_mismatch_reports_the_array_path() {
        let actual = json!({"items": [1, 2]});
        let expected = json!({"items": [1, 2, 3]});

        assert_eq!(
            first_difference(&actual, &expected).as_deref(),
            Some("/items")
        );
    }

    #[test]
    fn array_element_mismatch_reports_the_index() {
        let actual = json!([1, 2, 3]);
        let expected = json!([1, 9, 3]);

        assert_eq!(first_difference(&actual, &expected).as_deref(), Some("/1"));
    }

    #[test]
    fn root_type_mismatch_reports_the_root_path() {
        let actual = json!({"a": 1});
        let expected = json!("text");

        assert_eq!(first_difference(&actual, &expected).as_deref(), Some(""));
    }

    #[test]
    fn pointer_tokens_are_escaped() {
        let actual = json!({"a/b": 1});
        let expected = json!({"a/b": 2});

        assert_eq!(
            first_difference(&actual, &expected).as_deref(),
            Some("/a~1b")
        );
    }

    #[test]
    fn assert_json_eq_passes_and_fails() {
        assert_json_eq(&json!({"a": [1]}), &json!({"a": [1]}));

        assert!(catches(|| {
            assert_json_eq(&json!({"a": 1}), &json!({"a": 2}));
        }));
    }

    #[test]
    fn subset_allows_extra_actual_keys() {
        let actual = json!({"a": 1, "b": 2, "c": {"d": 3, "e": 4}});
        let expected = json!({"a": 1, "c": {"d": 3}});

        assert_json_subset(&actual, &expected);
    }

    #[test]
    fn subset_rejects_missing_keys_and_different_values() {
        let actual = json!({"a": 1});

        assert!(catches(|| {
            assert_json_subset(&actual, &json!({"b": 1}));
        }));
        assert!(catches(|| {
            assert_json_subset(&actual, &json!({"a": 2}));
        }));
    }

    #[test]
    fn subset_requires_equal_array_lengths() {
        assert!(catches(|| {
            assert_json_subset(&json!([1, 2, 3]), &json!([1, 2]));
        }));
        assert!(catches(|| {
            assert_json_subset(&json!([1, 2]), &json!([1, 9]));
        }));
    }

    #[test]
    fn pointer_assertions_pass_and_fail() {
        let actual = json!({"choices": [{"message": {"content": "hi"}}]});

        assert_json_pointer_eq(&actual, "/choices/0/message/content", &json!("hi"));
        assert!(catches(|| {
            assert_json_pointer_eq(&actual, "/choices/0/message/content", &json!("bye"));
        }));
        assert!(catches(|| {
            assert_json_pointer_eq(&actual, "/choices/9/message", &json!("hi"));
        }));
    }

    #[test]
    fn json_at_resolves_valid_and_invalid_pointers() {
        let value = json!({"a": {"b": [10, 20]}});

        assert_eq!(json_at(&value, ""), Some(&value));
        assert_eq!(json_at(&value, "/a/b/1"), Some(&json!(20)));
        assert_eq!(json_at(&value, "/a/missing"), None);
        assert_eq!(json_at(&value, "a/b"), None);
    }

    #[test]
    fn json_pretty_is_indented() {
        let text = json_pretty(&json!({"a": 1, "b": [1, 2]}));

        assert!(text.contains('\n'), "pretty output is multi-line: {text}");
        assert!(text.contains("\"a\": 1"));
    }

    #[test]
    fn roundtrip_accepts_a_matching_struct_and_rejects_a_lossy_one() {
        let sample = Sample {
            name: "agent".to_owned(),
            count: 2,
            tags: vec!["a".to_owned()],
        };
        assert_json_roundtrip(&sample);

        // `serde(skip)` on a field means the value does not survive the trip.
        #[derive(Debug, PartialEq, Serialize, Deserialize)]
        struct Lossy {
            kept: u32,
            #[serde(skip)]
            dropped: u32,
        }
        assert!(catches(|| {
            assert_json_roundtrip(&Lossy {
                kept: 1,
                dropped: 2,
            });
        }));
    }
}
