//! Parsing of the raw JSON text of a model-requested tool call.
//!
//! Python parity: `openjiuwen/core/single_agent/utils.py`'s argument parsing
//! hook. Providers usually send `arguments` as a JSON object string, but may
//! omit it entirely for argument-less tools; streaming providers may truncate
//! the text or wrap it in a Markdown fence.
//!
//! Parsing therefore has three stages: strip an optional code fence, try
//! `serde_json`, then apply a bounded repair (trailing commas, unterminated
//! strings, missing closing brackets) and retry. Anything still invalid is a
//! [`Error::Tool`] so the model receives a tool message it can react to.
use openjiuwen_common::error::{Error, Result};
use serde_json::Value;

/// Maximum number of repair-and-retry passes before giving up.
const MAX_REPAIR_PASSES: usize = 3;

/// Parses the raw `arguments` text of a [`ToolCall`].
///
/// Empty or whitespace-only input means "no arguments" and yields an empty
/// object (`{}`), matching how argument-less tools are invoked. Markdown code
/// fences are stripped and mildly malformed JSON is repaired before failing.
///
/// [`ToolCall`]: openjiuwen_foundation::ToolCall
///
/// # Errors
///
/// Returns [`Error::Tool`] when `raw` is not, and cannot be repaired into,
/// valid JSON.
#[must_use = "the parsed arguments are the invocation input"]
pub fn parse_tool_arguments(raw: &str) -> Result<Value> {
    let text = strip_code_fence(raw);
    if text.trim().is_empty() {
        return Ok(serde_json::json!({}));
    }
    let mut candidate = text.trim().to_owned();
    let mut last_error = None;
    for pass in 0..=MAX_REPAIR_PASSES {
        match serde_json::from_str::<Value>(&candidate) {
            Ok(value) => return Ok(value),
            Err(error) => last_error = Some(error),
        }
        if pass == MAX_REPAIR_PASSES {
            break;
        }
        let repaired = repair_once(&candidate);
        if repaired == candidate {
            // No repair was possible, so further passes cannot help.
            break;
        }
        candidate = repaired;
    }
    let source =
        last_error.map_or_else(|| "unrecognized JSON".to_owned(), |error| error.to_string());
    Err(Error::tool(format!(
        "Invalid tool arguments JSON: {source}"
    )))
}

/// Removes one leading Markdown code fence (```` ```json ````) and one trailing
/// fence, then trims the result.
fn strip_code_fence(raw: &str) -> String {
    let trimmed = raw.trim();
    let Some(after_open) = trimmed.strip_prefix("```") else {
        return trimmed.to_owned();
    };
    let body = match after_open.find('\n') {
        // Drop the info string (`json`, `JSON`, …) up to the first newline.
        Some(index) => &after_open[index + 1..],
        // Single-line fence: drop the info string itself.
        None => after_open.trim_start_matches(|ch: char| ch.is_alphanumeric()),
    };
    let body = body.trim_end();
    let body = body.strip_suffix("```").unwrap_or(body);
    body.trim().to_owned()
}

/// Applies one repair pass and returns the repaired text.
///
/// The pass: drops commas that precede a closing bracket, closes an
/// unterminated string, and appends the missing closing brackets in reverse
/// open order. A pass that cannot change the text returns it unchanged, which
/// tells the caller to stop retrying.
fn repair_once(text: &str) -> String {
    let mut repaired: Vec<char> = Vec::with_capacity(text.len() + 4);
    let mut open_brackets: Vec<char> = Vec::new();
    let mut in_string = false;
    let mut escaped = false;
    // Index in `repaired` of a comma whose value may not have arrived yet.
    let mut pending_comma: Option<usize> = None;
    for ch in text.chars() {
        if in_string {
            repaired.push(ch);
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }
        match ch {
            '"' => {
                pending_comma = None;
                in_string = true;
                repaired.push(ch);
            }
            '{' | '[' => {
                pending_comma = None;
                open_brackets.push(ch);
                repaired.push(ch);
            }
            '}' | ']' => {
                if let Some(index) = pending_comma.take() {
                    repaired.remove(index);
                }
                open_brackets.pop();
                repaired.push(ch);
            }
            ',' => {
                pending_comma = Some(repaired.len());
                repaired.push(ch);
            }
            ch if ch.is_whitespace() => repaired.push(ch),
            _ => {
                pending_comma = None;
                repaired.push(ch);
            }
        }
    }
    if in_string {
        // The trailing backslash escapes nothing once the string is closed, so
        // it has to go before the synthetic quote.
        if escaped {
            repaired.pop();
        }
        repaired.push('"');
    }
    while let Some(open) = open_brackets.pop() {
        repaired.push(if open == '{' { '}' } else { ']' });
    }
    repaired
        .into_iter()
        .collect::<String>()
        .trim_end()
        .to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input_is_an_empty_object() {
        assert_eq!(parse_tool_arguments("").unwrap(), serde_json::json!({}));
    }

    #[test]
    fn whitespace_input_is_an_empty_object() {
        assert_eq!(
            parse_tool_arguments("  \n\t ").unwrap(),
            serde_json::json!({})
        );
    }

    #[test]
    fn valid_object_round_trips() {
        assert_eq!(
            parse_tool_arguments(r#"{"city": "Paris", "days": 3}"#).unwrap(),
            serde_json::json!({"city": "Paris", "days": 3})
        );
    }

    #[test]
    fn fenced_object_is_unwrapped() {
        let raw = "```json\n{\"city\": \"Paris\"}\n```";
        assert_eq!(
            parse_tool_arguments(raw).unwrap(),
            serde_json::json!({"city": "Paris"})
        );
        let raw = "```\n[1, 2]\n```";
        assert_eq!(
            parse_tool_arguments(raw).unwrap(),
            serde_json::json!([1, 2])
        );
    }

    #[test]
    fn truncated_array_is_closed() {
        assert_eq!(
            parse_tool_arguments(r#"{"a": [1, 2"#).unwrap(),
            serde_json::json!({"a": [1, 2]})
        );
    }

    #[test]
    fn truncated_nested_objects_are_closed_in_reverse_order() {
        assert_eq!(
            parse_tool_arguments(r#"{"a": {"b": 1"#).unwrap(),
            serde_json::json!({"a": {"b": 1}})
        );
    }

    #[test]
    fn trailing_commas_are_dropped() {
        assert_eq!(
            parse_tool_arguments(r#"{"a": 1,}"#).unwrap(),
            serde_json::json!({"a": 1})
        );
        assert_eq!(
            parse_tool_arguments(r#"{"a": [1, 2, ], }"#).unwrap(),
            serde_json::json!({"a": [1, 2]})
        );
    }

    #[test]
    fn unterminated_string_is_closed() {
        assert_eq!(
            parse_tool_arguments(r#"{"name": "Paris"#).unwrap(),
            serde_json::json!({"name": "Paris"})
        );
        assert_eq!(
            parse_tool_arguments("{\"name\": \"Paris\\").unwrap(),
            serde_json::json!({"name": "Paris"})
        );
    }

    #[test]
    fn plain_text_is_a_tool_error() {
        let error = parse_tool_arguments("not json").unwrap_err();
        assert!(matches!(error, Error::Tool(_)), "unexpected error: {error}");
        assert!(error.message().contains("Invalid tool arguments JSON"));
    }

    #[test]
    fn scalar_json_passes_through() {
        assert_eq!(
            parse_tool_arguments(r#""x""#).unwrap(),
            serde_json::json!("x")
        );
        assert_eq!(parse_tool_arguments("3").unwrap(), serde_json::json!(3));
    }
}
