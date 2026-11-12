//! Parsing of the raw JSON text of a model-requested tool call.
//!
//! Python parity: `openjiuwen/core/single_agent/utils.py`'s argument parsing
//! hook. Providers usually send `arguments` as a JSON object string, but may
//! omit it entirely for argument-less tools.
use openjiuwen_common::error::{Error, Result};
use serde_json::Value;

/// Parses the raw `arguments` text of a [`ToolCall`].
///
/// Empty or whitespace-only input means "no arguments" and yields an empty
/// object (`{}`), matching how argument-less tools are invoked. Any other input
/// must be valid JSON.
///
/// [`ToolCall`]: openjiuwen_foundation::ToolCall
///
/// # Errors
///
/// Returns [`Error::Tool`] when `raw` is not valid JSON.
///
// TODO: repair truncated JSON (bracket and quote balancing) before failing; the
// current implementation only accepts well-formed JSON.
#[must_use = "the parsed arguments are the invocation input"]
pub fn parse_tool_arguments(raw: &str) -> Result<Value> {
    if raw.trim().is_empty() {
        return Ok(serde_json::json!({}));
    }
    serde_json::from_str::<Value>(raw)
        .map_err(|error| Error::tool(format!("invalid tool arguments: {error}")))
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
    fn invalid_json_is_a_tool_error() {
        let error = parse_tool_arguments(r#"{"city": "#).unwrap_err();
        assert!(matches!(error, Error::Tool(_)), "unexpected error: {error}");
        assert!(error.message().contains("invalid tool arguments"));
    }
}
