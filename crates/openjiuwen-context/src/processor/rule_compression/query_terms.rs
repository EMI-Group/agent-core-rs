//! Relevance terms extracted from the user request and the triggering tool call.
//!
//! Ports `rule_compression/query_terms.py`: collect lower-case
//! `[A-Za-z0-9_]{3,}` tokens from the latest user message, the tool name and the
//! tool-call arguments, skipping path-valued and non-user-input argument keys.
//! Compressors use the set to rank diffs, search hits and source-file regions.
//!
//! CJK text contributes only its ASCII terms, on purpose: the Python
//! implementation tokenises with the same ASCII class, and inventing per-script
//! segmentation would make the ranking non-portable.
use serde_json::Value;
use std::collections::BTreeSet;
/// Argument keys whose value is a path rather than user intent.
const PATH_ARGUMENT_KEYS: [&str; 6] =
    ["file", "file_path", "filepath", "filename", "path", "paths"];
/// Argument keys that never carry user intent.
const NON_USER_INPUT_ARGUMENT_KEYS: [&str; 17] = [
    "trusted_dirs",
    "preferred_response_language",
    "files_updated_by_user",
    "timezone",
    "timestamp",
    "encoding",
    "offset",
    "limit",
    "start",
    "end",
    "channel",
    "channels",
    "frontend",
    "session_id",
    "context_id",
    "request_id",
    "recursion_limit",
];
/// Extracts the stable lower-case terms used for relevance scoring.
///
/// `tool_arguments` is the raw JSON string of the call's arguments; a string that
/// is not valid JSON contributes its own ASCII terms.
#[must_use]
pub fn extract_query_terms(
    user_content: &str,
    tool_name: Option<&str>,
    tool_arguments: Option<&str>,
) -> BTreeSet<String> {
    let mut terms = BTreeSet::new();
    add_terms(&mut terms, user_content);
    if let Some(name) = tool_name {
        add_terms(&mut terms, name);
    }
    let parsed = tool_arguments.map(parse_json_or_string);
    if let Some(value) = parsed.as_ref() {
        add_terms_from_value(&mut terms, value);
    }
    terms
}
/// Parses `raw` as JSON, falling back to the raw string value.
fn parse_json_or_string(raw: &str) -> Value {
    serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.to_owned()))
}
/// Adds every `[A-Za-z0-9_]{3,}` run in `value` that is not path-valued.
fn add_terms(terms: &mut BTreeSet<String>, value: &str) {
    let bytes = value.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        if !is_term_byte(bytes[index]) {
            index += 1;
            continue;
        }
        let start = index;
        while index < bytes.len() && is_term_byte(bytes[index]) {
            index += 1;
        }
        if index - start < 3 {
            continue;
        }
        // A term glued to a path separator is a path fragment, not a word.
        let path_prefixed = start > 0 && matches!(bytes[start - 1], b'/' | b'\\');
        if path_prefixed {
            continue;
        }
        if let Ok(term) = std::str::from_utf8(&bytes[start..index]) {
            terms.insert(term.to_ascii_lowercase());
        }
    }
}
/// Whether `byte` belongs to the term alphabet.
fn is_term_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
}
/// Walks a parsed argument value, collecting terms.
fn add_terms_from_value(terms: &mut BTreeSet<String>, value: &Value) {
    match value {
        Value::Null => {}
        Value::String(text) => add_terms(terms, text),
        Value::Object(map) => {
            for (key, item) in map {
                add_terms(terms, key);
                let lowered = key.to_lowercase();
                if PATH_ARGUMENT_KEYS.contains(&lowered.as_str())
                    || NON_USER_INPUT_ARGUMENT_KEYS.contains(&lowered.as_str())
                {
                    continue;
                }
                add_terms_from_value(terms, item);
            }
        }
        Value::Array(items) => {
            for item in items {
                add_terms_from_value(terms, item);
            }
        }
        Value::Bool(flag) => add_terms(terms, if *flag { "true" } else { "false" }),
        Value::Number(number) => add_terms(terms, &number.to_string()),
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn user_request_and_tool_name_contribute_terms() {
        let terms = extract_query_terms("Why does the PAYOUT overflow?", Some("grep_tool"), None);
        assert!(terms.contains("why"));
        assert!(terms.contains("does"));
        assert!(terms.contains("the"));
        assert!(terms.contains("payout"));
        assert!(terms.contains("overflow"));
        assert!(terms.contains("grep_tool"));
        assert!(!terms.contains("pa"));
    }
    #[test]
    fn path_and_paging_arguments_are_skipped() {
        let arguments = r#"{"path": "src/payment_router.rs", "offset": 42,
            "query": "payment timeout"}"#;
        let terms = extract_query_terms("", None, Some(arguments));
        assert!(terms.contains("query"));
        assert!(terms.contains("payment"));
        assert!(terms.contains("timeout"));
        assert!(!terms.contains("payment_router"));
        assert!(!terms.contains("42"));
        // Python adds the *key* before skipping its value, so `offset` is a term.
        assert!(terms.contains("offset"));
    }
    #[test]
    fn invalid_json_contributes_its_raw_terms() {
        let terms = extract_query_terms("", None, Some("grep_payment_timeout"));
        assert!(terms.contains("grep_payment_timeout"));
    }
}
