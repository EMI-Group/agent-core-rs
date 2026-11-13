//! Newline-delimited JSON helpers shared by the vendor CLI transports.
//!
//! A vendor CLI speaks one JSON value per line. This module holds the pure half of that protocol:
//! parsing one line, encoding one line, and the typed object accessors every provider decoder
//! needs (string, object, array, bool, integer, float, string list) plus [`snippet`], which keeps
//! error messages short when a CLI emits a huge or binary line.
//!
//! The helpers never panic and never coerce: a value of the wrong JSON type reads as [`None`], so
//! a provider decoder can apply its own wording instead of inheriting a parse failure.

use openjiuwen_harness_protocol::{HarnessError, HarnessResult, JsonObject, JsonValue};

/// Label used by [`parse_line`] when the caller does not name its provider.
const DEFAULT_LABEL: &str = "harness";

/// Maximum number of characters a [`snippet`] keeps by default in an error message.
const DEFAULT_SNIPPET_CHARS: usize = 200;

/// Parses one NDJSON line into a JSON value.
///
/// Surrounding whitespace is ignored, so a trailing `\r` or an indented line parses correctly.
/// Use [`parse_line_labelled`] to name the provider in the error message.
///
/// # Errors
///
/// Returns [`HarnessError::Protocol`] with `invalid harness JSON line: {snippet}` when the line is
/// not valid JSON.
pub fn parse_line(line: &str) -> HarnessResult<JsonValue> {
    parse_line_labelled(DEFAULT_LABEL, line)
}

/// Parses one NDJSON line, naming `label` in the error message.
///
/// # Errors
///
/// Returns [`HarnessError::Protocol`] with `invalid {label} JSON line: {snippet}` when the line is
/// not valid JSON.
pub fn parse_line_labelled(label: &str, line: &str) -> HarnessResult<JsonValue> {
    let trimmed = line.trim();
    serde_json::from_str(trimmed).map_err(|_| {
        HarnessError::protocol(format!(
            "invalid {label} JSON line: {}",
            snippet(trimmed, DEFAULT_SNIPPET_CHARS)
        ))
    })
}

/// Encodes `value` as one compact JSON line, without the trailing newline.
///
/// A [`JsonValue`] is always JSON-encodable, so the fallback is unreachable; it exists so the
/// helper stays panic-free.
#[must_use]
pub fn encode_line(value: &JsonValue) -> String {
    match serde_json::to_string(value) {
        Ok(line) => line,
        Err(error) => {
            tracing::warn!(%error, "failed to encode a vendor CLI JSON line");
            "null".to_owned()
        }
    }
}

/// Reads a string member, returning [`None`] when it is absent or not a string.
#[must_use]
pub fn get_str(object: &JsonObject, key: &str) -> Option<String> {
    object
        .get(key)
        .and_then(JsonValue::as_str)
        .map(str::to_owned)
}

/// Reads an object member as an owned [`JsonObject`].
///
/// The result is a copy rather than a borrow: [`JsonObject`] is a `BTreeMap`, while serde stores an
/// object inside a [`JsonValue`] as a `serde_json::Map`, so no borrowed view can be handed out.
/// Returns [`None`] when the member is absent or not an object.
#[must_use]
pub fn get_object(object: &JsonObject, key: &str) -> Option<JsonObject> {
    object
        .get(key)
        .and_then(JsonValue::as_object)
        .map(|member| member.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
}

/// Reads an array member, returning [`None`] when it is absent or not an array.
#[must_use]
pub fn get_array<'a>(object: &'a JsonObject, key: &str) -> Option<&'a Vec<JsonValue>> {
    object.get(key).and_then(JsonValue::as_array)
}

/// Reads a boolean member, returning [`None`] when it is absent or not a boolean.
#[must_use]
pub fn get_bool(object: &JsonObject, key: &str) -> Option<bool> {
    object.get(key).and_then(JsonValue::as_bool)
}

/// Reads an unsigned integer member, returning [`None`] when it is absent or not a non-negative
/// integer.
#[must_use]
pub fn get_u64(object: &JsonObject, key: &str) -> Option<u64> {
    object.get(key).and_then(JsonValue::as_u64)
}

/// Reads a numeric member as `f64`, returning [`None`] when it is absent or not a number.
#[must_use]
pub fn get_f64(object: &JsonObject, key: &str) -> Option<f64> {
    object.get(key).and_then(JsonValue::as_f64)
}

/// Reads an array member as a list of strings, skipping every non-string element.
///
/// Non-string elements are skipped instead of failing: vendor CLIs occasionally emit a `null` or a
/// nested object where the schema announces a string, and dropping it keeps the rest of the list.
#[must_use]
pub fn string_list(object: &JsonObject, key: &str) -> Vec<String> {
    get_array(object, key)
        .map(|values| {
            values
                .iter()
                .filter_map(JsonValue::as_str)
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default()
}

/// Truncates `text` to at most `max` characters, appending `…` when anything was cut.
///
/// Truncation counts characters, not bytes, so the result stays valid UTF-8 for any input. A `max`
/// of zero yields the ellipsis alone.
#[must_use]
pub fn snippet(text: &str, max: usize) -> String {
    let mut chars = text.chars();
    let mut kept = String::new();
    for _ in 0..max {
        match chars.next() {
            Some(character) => kept.push(character),
            None => return kept,
        }
    }
    if chars.next().is_some() {
        kept.push('…');
    }
    kept
}

#[cfg(test)]
mod tests {
    // Tests assert by panicking on failure, which is the intent of a test.
    #![allow(clippy::expect_used, clippy::unwrap_used)]

    use openjiuwen_harness_protocol::HarnessErrorKind;
    use serde_json::json;

    use super::{
        encode_line, get_array, get_bool, get_f64, get_object, get_str, get_u64, parse_line,
        parse_line_labelled, snippet, string_list,
    };

    /// Object used by the accessor tests.
    fn sample() -> openjiuwen_harness_protocol::JsonObject {
        serde_json::from_value(json!({
            "text": "hello",
            "count": 3,
            "ratio": 1.5,
            "flag": true,
            "inner": {"nested": 1},
            "items": ["a", 2, "b", null],
            "number": "not a number",
        }))
        .expect("a JSON object")
    }

    #[test]
    fn parse_line_accepts_surrounding_whitespace() {
        assert_eq!(
            parse_line("  {\"type\": \"result\"}\r\n").expect("valid JSON"),
            json!({"type": "result"})
        );
        assert_eq!(parse_line("null").expect("valid JSON"), json!(null));
    }

    #[test]
    fn parse_line_reports_the_snippet_of_an_invalid_line() {
        let error = parse_line("not json at all").expect_err("invalid JSON");
        assert_eq!(error.kind(), HarnessErrorKind::Protocol);
        assert_eq!(
            error.message(),
            "invalid harness JSON line: not json at all"
        );
    }

    #[test]
    fn parse_line_labelled_names_the_provider() {
        let error = parse_line_labelled("Codex", "{").expect_err("invalid JSON");
        assert_eq!(error.message(), "invalid Codex JSON line: {");
    }

    #[test]
    fn parse_line_truncates_a_long_invalid_line() {
        let line = "x".repeat(500);
        let error = parse_line(&line).expect_err("invalid JSON");
        assert!(error.message().ends_with('…'));
        assert_eq!(
            error.message().chars().count(),
            "invalid harness JSON line: ".len() + 201
        );
    }

    #[test]
    fn encode_line_is_compact_and_round_trips() {
        let value = json!({"a": 1, "b": [true, null]});
        let line = encode_line(&value);
        assert_eq!(line, "{\"a\":1,\"b\":[true,null]}");
        assert_eq!(parse_line(&line).expect("valid JSON"), value);
    }

    #[test]
    fn accessors_read_typed_members_and_reject_others() {
        let object = sample();
        assert_eq!(get_str(&object, "text").as_deref(), Some("hello"));
        assert_eq!(get_str(&object, "count"), None);
        assert_eq!(get_bool(&object, "flag"), Some(true));
        assert_eq!(get_u64(&object, "count"), Some(3));
        assert_eq!(get_u64(&object, "ratio"), None);
        assert_eq!(get_f64(&object, "ratio"), Some(1.5));
        assert_eq!(get_f64(&object, "text"), None);
        assert_eq!(
            get_object(&object, "inner"),
            Some(serde_json::from_value(json!({"nested": 1})).expect("a JSON object"))
        );
        assert_eq!(get_object(&object, "text"), None);
        assert_eq!(get_array(&object, "items").map(Vec::len), Some(4));
        assert_eq!(get_array(&object, "text"), None);
    }

    #[test]
    fn string_list_skips_non_string_elements() {
        let object = sample();
        assert_eq!(
            string_list(&object, "items"),
            vec!["a".to_owned(), "b".to_owned()]
        );
        assert_eq!(string_list(&object, "text"), Vec::<String>::new());
        assert_eq!(string_list(&object, "absent"), Vec::<String>::new());
    }

    #[test]
    fn snippet_keeps_short_text_and_marks_truncation() {
        assert_eq!(snippet("abc", 10), "abc");
        assert_eq!(snippet("abcdef", 6), "abcdef");
        assert_eq!(snippet("abcdef", 3), "abc…");
        assert_eq!(snippet("abcdef", 0), "…");
        assert_eq!(snippet("", 0), "");
    }

    #[test]
    fn snippet_counts_characters_not_bytes() {
        assert_eq!(snippet("日本語テキスト", 3), "日本語…");
    }
}
