//! Deterministic helpers ported from `agent_builder/utils/utils.py`.
//!
//! The Python module mixes pure text/dict helpers with filesystem access; this
//! port keeps only the deterministic, model-free ones:
//!
//! - [`crate::agent_builder::utils::extract_json_from_text`] — the body of the
//!   first fenced code block, or the input when there is no fence.
//! - [`crate::agent_builder::utils::format_dialog_history`] and
//!   [`crate::agent_builder::utils::format_dialog_history_with`] — `"role: text"`
//!   entries joined by a separator.
//! - [`crate::agent_builder::utils::merge_dict_lists`] — first-wins merge keyed
//!   on one truthy field.
//! - [`crate::agent_builder::utils::validate_session_id`] and
//!   [`crate::agent_builder::utils::is_valid_session_id`] — the session-id
//!   character class.
//!
//! Python's `safe_json_loads`, `deep_merge_dict`, and `load_json_file` are not
//! ported: `serde_json` covers safe parsing at the call site, and file loading
//! belongs to the caller (see the directory `CONTEXT.md`).
//!
//! Fence extraction is deliberately *not* [`crate::util::extract_json_fence`]:
//! that helper parses a `json`-tagged fence into a [`serde_json::Value`] and
//! ignores untagged fences, while Python's `extract_json_from_text` returns the
//! raw body of the first fence of either kind.

use openjiuwen_common::error::{Error, Result};
use regex::Regex;
use serde_json::{Map, Value};

use super::history::DialogueMessage;

/// Python's `JSON_EXTRACT_PATTERN` (`agent_builder/utils/constants.py`).
const JSON_EXTRACT_PATTERN: &str = r"```(?:json)?\s*([\s\S]*?)\s*```";

/// Python's `validate_session_id` pattern.
const SESSION_ID_PATTERN: &str = r"^[a-zA-Z0-9_-]+$";

/// The role rendered for a message whose role is missing or blank.
const UNKNOWN_ROLE: &str = "unknown";

/// Extracts the body of the first fenced code block in `text`.
///
/// Mirrors Python's `extract_json_from_text`: the pattern accepts both a
/// `json`-tagged fence and an untagged one, the first match wins, and the
/// whitespace the pattern allows around the body is trimmed. Text without a
/// fence — including the empty string — is returned unchanged, so an unfenced
/// answer still reaches the caller's parser.
///
/// # Example
///
/// ````
/// use openjiuwen_dev_tools::agent_builder::utils::extract_json_from_text;
///
/// let answer = "```json\n{\"key\": \"value\"}\n```";
/// assert_eq!(extract_json_from_text(answer), "{\"key\": \"value\"}");
/// assert_eq!(extract_json_from_text("not fenced"), "not fenced");
/// ````
#[must_use]
pub fn extract_json_from_text(text: &str) -> String {
    if text.is_empty() {
        return String::new();
    }
    // The pattern is a compile-time constant; a failed compilation would mean a
    // broken build rather than bad input, so the fallback is "no fence found".
    let Ok(regex) = Regex::new(JSON_EXTRACT_PATTERN) else {
        return text.to_string();
    };
    regex
        .captures(text)
        .and_then(|captures| captures.get(1))
        .map_or_else(|| text.to_string(), |body| body.as_str().to_string())
}

/// Formats `history` as `"role: content"` entries joined by newlines.
///
/// This is [`format_dialog_history_with`] with Python's default separator.
#[must_use]
pub fn format_dialog_history(history: &[DialogueMessage]) -> String {
    format_dialog_history_with(history, "\n")
}

/// Formats `history` as `"role: content"` entries joined by `separator`.
///
/// A missing or blank role renders as `unknown`, matching Python's
/// `msg.get("role", "unknown")` for the empty roles this port can represent;
/// the content is copied verbatim. An empty history yields the empty string.
#[must_use]
pub fn format_dialog_history_with(history: &[DialogueMessage], separator: &str) -> String {
    let lines: Vec<String> = history
        .iter()
        .map(|message| {
            let role = if message.role.trim().is_empty() {
                UNKNOWN_ROLE
            } else {
                message.role.as_str()
            };
            format!("{role}: {}", message.content)
        })
        .collect();
    lines.join(separator)
}

/// Merges `new_items` into `existing`, keeping the first occurrence of each key.
///
/// Exactly Python's `merge_dict_lists`: every existing item is kept in order —
/// even when it lacks `unique_key` or the value is falsy, because Python never
/// removes duplicates already present in `existing` — and each new item is
/// appended only when its `unique_key` value is truthy and not already seen
/// among the existing truthy keys or the previously appended ones.
///
/// Truthiness follows Python: a missing key, `null`, `false`, a blank string,
/// zero, an empty array, and an empty object are all falsy.
#[must_use]
pub fn merge_dict_lists(
    existing: &[Map<String, Value>],
    new_items: &[Map<String, Value>],
    unique_key: &str,
) -> Vec<Map<String, Value>> {
    let mut merged = existing.to_vec();
    if new_items.is_empty() {
        return merged;
    }
    let mut seen: Vec<Value> = existing
        .iter()
        .filter_map(|item| item.get(unique_key))
        .filter(|value| is_truthy(value))
        .cloned()
        .collect();
    for item in new_items {
        let Some(key) = item.get(unique_key) else {
            continue;
        };
        if !is_truthy(key) || seen.contains(key) {
            continue;
        }
        seen.push(key.clone());
        merged.push(item.clone());
    }
    merged
}

/// Validates a builder session id.
///
/// Session ids may contain only ASCII letters, digits, underscores, and
/// hyphens, and must be non-empty — Python's `^[a-zA-Z0-9_-]+$`.
///
/// # Errors
///
/// Returns [`Error::Validation`] naming the rejected id for an empty id or one
/// containing any other character.
///
/// # Example
///
/// ```
/// use openjiuwen_dev_tools::agent_builder::utils::validate_session_id;
///
/// assert!(validate_session_id("session_123").is_ok());
/// let error = validate_session_id("session@123").expect_err("rejected");
/// assert!(error.to_string().contains("session@123"));
/// ```
pub fn validate_session_id(session_id: &str) -> Result<()> {
    if is_valid_session_id(session_id) {
        Ok(())
    } else {
        Err(Error::validation(format!(
            "invalid session id: {session_id}"
        )))
    }
}

/// Whether `session_id` matches Python's session-id character class.
///
/// The infallible half of [`validate_session_id`], for callers that only need
/// the predicate.
#[must_use]
pub fn is_valid_session_id(session_id: &str) -> bool {
    !session_id.is_empty()
        && Regex::new(SESSION_ID_PATTERN).is_ok_and(|regex| regex.is_match(session_id))
}

/// Whether `value` counts as a present key, using Python truthiness.
fn is_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(flag) => *flag,
        Value::String(text) => !text.trim().is_empty(),
        Value::Number(number) => match number.as_i64() {
            Some(integer) => integer != 0,
            None => number
                .as_f64()
                .is_some_and(|float| float.abs() > f64::EPSILON),
        },
        Value::Array(items) => !items.is_empty(),
        Value::Object(entries) => !entries.is_empty(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Builds one dialogue turn.
    fn message(role: &str, content: &str) -> DialogueMessage {
        DialogueMessage {
            role: role.to_string(),
            content: content.to_string(),
        }
    }

    /// Converts a JSON literal into the object type the merge helper takes.
    fn object(value: Value) -> Map<String, Value> {
        match value {
            Value::Object(entries) => entries,
            other => panic!("expected a JSON object, got {other}"),
        }
    }

    #[test]
    fn extracts_json_tagged_fence() {
        let text = "Here is the document:\n```json\n{\"key\": \"value\"}\n```\nthanks";
        assert_eq!(extract_json_from_text(text), "{\"key\": \"value\"}");
    }

    #[test]
    fn extracts_untagged_fence() {
        assert_eq!(extract_json_from_text("```\n{\"a\": 1}\n```"), "{\"a\": 1}");
        assert_eq!(
            extract_json_from_text("prefix ```\n[1, 2]\n``` suffix"),
            "[1, 2]"
        );
    }

    #[test]
    fn keeps_multiline_body_without_surrounding_whitespace() {
        let text = "```json\n{\n  \"a\": 1\n}\n```";
        assert_eq!(extract_json_from_text(text), "{\n  \"a\": 1\n}");
    }

    #[test]
    fn non_json_info_string_is_included_in_the_body() {
        // Python's pattern does not exclude other info strings: `json` is
        // optional, so the info string becomes part of the captured body.
        assert_eq!(
            extract_json_from_text("```python\nprint(1)\n```"),
            "python\nprint(1)"
        );
    }

    #[test]
    fn first_fence_wins() {
        let text = "```json\n{\"a\": 1}\n```\nthen\n```json\n{\"b\": 2}\n```";
        assert_eq!(extract_json_from_text(text), "{\"a\": 1}");
    }

    #[test]
    fn text_without_a_fence_is_returned_unchanged() {
        assert_eq!(extract_json_from_text("{\"a\": 1}"), "{\"a\": 1}");
        assert_eq!(
            extract_json_from_text("no fences at all"),
            "no fences at all"
        );
        assert_eq!(extract_json_from_text("``"), "``");
        // An unclosed fence has no body to return.
        assert_eq!(
            extract_json_from_text("```json\n{\"a\": 1}"),
            "```json\n{\"a\": 1}"
        );
    }

    #[test]
    fn empty_text_stays_empty() {
        assert_eq!(extract_json_from_text(""), "");
    }

    #[test]
    fn formats_history_with_default_separator() {
        let history = [message("user", "Hello"), message("assistant", "Hi there")];
        assert_eq!(
            format_dialog_history(&history),
            "user: Hello\nassistant: Hi there"
        );
    }

    #[test]
    fn formats_history_with_custom_separator() {
        let history = [message("user", "a"), message("assistant", "b")];
        assert_eq!(
            format_dialog_history_with(&history, " | "),
            "user: a | assistant: b"
        );
        assert_eq!(
            format_dialog_history_with(&history, ""),
            "user: aassistant: b"
        );
    }

    #[test]
    fn blank_or_missing_role_renders_unknown() {
        let history = [
            message("", "no role"),
            message("   ", "blank role"),
            message("user", "normal"),
        ];
        assert_eq!(
            format_dialog_history(&history),
            "unknown: no role\nunknown: blank role\nuser: normal"
        );
    }

    #[test]
    fn empty_history_formats_to_empty_string() {
        assert_eq!(format_dialog_history(&[]), "");
        assert_eq!(format_dialog_history_with(&[], " | "), "");
    }

    #[test]
    fn merge_keeps_existing_order_and_skips_known_keys() {
        let existing = vec![
            object(json!({"id": "1", "name": "A"})),
            object(json!({"id": "2", "name": "B"})),
        ];
        let new_items = vec![
            object(json!({"id": "2", "name": "B2"})),
            object(json!({"id": "3", "name": "C"})),
        ];
        assert_eq!(
            merge_dict_lists(&existing, &new_items, "id"),
            vec![
                object(json!({"id": "1", "name": "A"})),
                object(json!({"id": "2", "name": "B"})),
                object(json!({"id": "3", "name": "C"})),
            ]
        );
    }

    #[test]
    fn merge_keeps_existing_items_without_a_truthy_key() {
        let existing = vec![
            object(json!({"name": "no key"})),
            object(json!({"id": "", "name": "blank"})),
            object(json!({"id": null, "name": "null"})),
            object(json!({"id": 0, "name": "zero"})),
            object(json!({"id": false, "name": "false"})),
            object(json!({"id": [], "name": "array"})),
            object(json!({"id": {}, "name": "object"})),
            object(json!({"id": "kept"})),
        ];
        let new_items = vec![object(json!({"id": "new"}))];
        let merged = merge_dict_lists(&existing, &new_items, "id");
        assert_eq!(merged.len(), existing.len() + 1);
        assert_eq!(&merged[..existing.len()], existing.as_slice());
        assert_eq!(merged.last(), Some(&object(json!({"id": "new"}))));
    }

    #[test]
    fn merge_drops_new_items_with_falsy_keys() {
        let new_items = vec![
            object(json!({"name": "missing key"})),
            object(json!({"id": "", "name": "blank"})),
            object(json!({"id": "   ", "name": "spaces"})),
            object(json!({"id": null})),
            object(json!({"id": false})),
            object(json!({"id": 0})),
            object(json!({"id": 0.0})),
            object(json!({"id": []})),
            object(json!({"id": {}})),
            object(json!({"id": "kept"})),
            object(json!({"id": true, "name": "bool"})),
            object(json!({"id": 7, "name": "int"})),
        ];
        let merged = merge_dict_lists(&[], &new_items, "id");
        assert_eq!(
            merged,
            vec![
                object(json!({"id": "kept"})),
                object(json!({"id": true, "name": "bool"})),
                object(json!({"id": 7, "name": "int"})),
            ]
        );
    }

    #[test]
    fn merge_uses_the_requested_unique_key() {
        let existing = vec![object(json!({"resource_id": "r1", "name": "A"}))];
        let new_items = vec![
            object(json!({"resource_id": "r1", "name": "A2"})),
            object(json!({"resource_id": "r2", "name": "B"})),
        ];
        assert_eq!(
            merge_dict_lists(&existing, &new_items, "resource_id").len(),
            2
        );

        // The default Python key is `resource_id`, so `id` is not considered.
        let by_id = vec![object(json!({"id": "r1"}))];
        assert_eq!(merge_dict_lists(&existing, &by_id, "id").len(), 2);
    }

    #[test]
    fn merge_does_not_deduplicate_existing_or_new_aliases() {
        let existing = vec![
            object(json!({"id": "dup", "name": "first"})),
            object(json!({"id": "dup", "name": "second"})),
        ];
        let new_items = vec![
            object(json!({"id": "dup", "name": "third"})),
            object(json!({"id": "fresh", "name": "fourth"})),
            object(json!({"id": "fresh", "name": "fifth"})),
        ];
        assert_eq!(
            merge_dict_lists(&existing, &new_items, "id"),
            vec![
                object(json!({"id": "dup", "name": "first"})),
                object(json!({"id": "dup", "name": "second"})),
                object(json!({"id": "fresh", "name": "fourth"})),
            ]
        );
    }

    #[test]
    fn merge_with_no_new_items_returns_the_existing_list() {
        let existing = vec![object(json!({"id": "1"}))];
        assert_eq!(merge_dict_lists(&existing, &[], "id"), existing);
        assert_eq!(merge_dict_lists(&[], &[], "id").len(), 0);
    }

    #[test]
    fn valid_session_ids_are_accepted() {
        for session_id in ["session_123", "ABC-9_x", "a", "0", "--__"] {
            assert!(
                is_valid_session_id(session_id),
                "{session_id} should be valid"
            );
            assert!(validate_session_id(session_id).is_ok());
        }
    }

    #[test]
    fn invalid_session_ids_are_rejected_and_named() {
        for session_id in [
            "",
            "session 123",
            "session@123",
            "会话",
            "a.b",
            "a/b",
            "a\nb",
        ] {
            assert!(
                !is_valid_session_id(session_id),
                "{session_id} should be invalid"
            );
            let error = validate_session_id(session_id).expect_err("rejected");
            assert!(matches!(error, Error::Validation(_)), "{error:?}");
            assert!(error.to_string().contains(session_id));
            assert!(error
                .to_string()
                .starts_with("validation: invalid session id:"));
        }
    }
}
