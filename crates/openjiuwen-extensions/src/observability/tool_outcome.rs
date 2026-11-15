//! Shared reading of how one tool call actually ended.
//!
//! Port of `openjiuwen/extensions/observability/tool_outcome.py` (v0.1.20).
//!
//! A tool call can fail in two unrelated ways, and the span has to record both
//! the same way or the trajectory disagrees with the conversation the model
//! saw:
//!
//! - it **raises** — the ability manager catches the exception and still hands
//!   the model a `ToolMessage` describing it, so the span needs an output too,
//!   not only an exception event;
//! - it **returns** `ToolOutput(success=False)` — the far more common shape for
//!   the built-in tools (`bash`, `grep`, `glob`, `read_file`, `write_file`).
//!   Nothing raises, so a status derived from exceptions alone reports OK on a
//!   failed call.
//!
//! Both the harness rail (which owns the authoritative tool span) and the
//! low-level callback handler (which owns it in team mode) read the outcome
//! through this module, so the two paths cannot drift apart.
//!
//! # Python input mapping
//!
//! Python accepts *whatever the ability returned* and reads `success`/`error`
//! off a mapping or an attribute. Rust callers already hold the tool result as
//! JSON ([`tool_failure_reason`]), while a raised call is modelled by plain
//! arguments ([`tool_result_for_exception`]): the exception's optional
//! `tool_message.content` becomes `tool_message_content`, and the exception's
//! own text becomes `error`.
//! No crate-local exception type is invented, because the kernel reports tool
//! failures as `openjiuwen_common::error::Error` values rather than as Python
//! exceptions.
use serde_json::Value;
/// `error.type` for a call that returned a failing result instead of raising.
///
/// Exceptions keep reporting their own class name.
pub const TOOL_REPORTED_FAILURE: &str = "ToolReportedFailure";
/// Reason text used when a failing result carries no `error` of its own.
pub const DEFAULT_FAILURE_REASON: &str = "tool reported failure";
/// Prefix of the tool result the ability manager builds for a raised call.
pub const EXCEPTION_RESULT_PREFIX: &str = "Ability execution error: ";
/// Returns the failure a tool reported in its own result.
///
/// Only an explicit `"success": false` counts.
/// A result that carries no `success` field at all (workflow outputs, raw MCP
/// payloads, plain strings) is left alone rather than guessed at, and so is any
/// other `success` value.
/// The reason is the trimmed `error` text, or [`DEFAULT_FAILURE_REASON`] when
/// that is blank; a non-string `error` is rendered as compact JSON, matching the
/// `str(error)` fallback Python applies.
#[must_use]
pub fn tool_failure_reason(output: &Value) -> Option<String> {
    if output.get("success") != Some(&Value::Bool(false)) {
        return None;
    }
    let reason = match output.get("error") {
        Some(Value::String(error)) => error.clone(),
        Some(Value::Null) | None => String::new(),
        Some(other) => other.to_string(),
    };
    let trimmed = reason.trim();
    if trimmed.is_empty() {
        return Some(DEFAULT_FAILURE_REASON.to_owned());
    }
    Some(trimmed.to_owned())
}
/// Rebuilds the tool result the model was handed for a raised call.
///
/// `AbilityManager` turns a raised tool call into a `ToolMessage` before the
/// model ever sees it, preferring the exception's own `tool_message` when it
/// carries one.
/// Mirroring that here keeps the recorded output equal to the conversation
/// content instead of merely similar to it: a non-empty
/// `tool_message_content` is returned verbatim, and otherwise the result is
/// [`EXCEPTION_RESULT_PREFIX`] followed by `error`.
#[must_use]
pub fn tool_result_for_exception(tool_message_content: Option<&str>, error: &str) -> String {
    match tool_message_content {
        Some(content) if !content.is_empty() => content.to_owned(),
        _ => format!("{EXCEPTION_RESULT_PREFIX}{error}"),
    }
}
#[cfg(test)]
mod tests {
    //! Port of the `tool_outcome.py` assertions of the observability suites.
    //!
    //! A test module owns its panics: a failed `unwrap`/`expect` is an assertion.
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::{
        tool_failure_reason, tool_result_for_exception, DEFAULT_FAILURE_REASON,
        EXCEPTION_RESULT_PREFIX, TOOL_REPORTED_FAILURE,
    };
    use serde_json::json;
    #[test]
    fn constants_name_the_wire_values() {
        assert_eq!(TOOL_REPORTED_FAILURE, "ToolReportedFailure");
        assert_eq!(DEFAULT_FAILURE_REASON, "tool reported failure");
        assert_eq!(EXCEPTION_RESULT_PREFIX, "Ability execution error: ");
    }
    #[test]
    fn an_explicit_failure_reports_its_reason() {
        let output = json!({"success": false, "error": "  bash exited with 1  "});
        assert_eq!(
            tool_failure_reason(&output),
            Some("bash exited with 1".to_owned())
        );
    }
    #[test]
    fn a_failure_without_a_reason_uses_the_default() {
        for output in [
            json!({"success": false}),
            json!({"success": false, "error": null}),
            json!({"success": false, "error": "   "}),
            json!({"success": false, "error": ""}),
        ] {
            assert_eq!(
                tool_failure_reason(&output),
                Some(DEFAULT_FAILURE_REASON.to_owned()),
                "{output}"
            );
        }
    }
    #[test]
    fn a_non_string_error_is_rendered_as_compact_json() {
        let output = json!({"success": false, "error": {"code": 2, "detail": "bad"}});
        assert_eq!(
            tool_failure_reason(&output),
            Some(r#"{"code":2,"detail":"bad"}"#.to_owned())
        );
    }
    #[test]
    fn only_an_explicit_false_counts_as_a_failure() {
        for output in [
            json!({"success": true}),
            json!({"error": "exited with 1"}),
            json!({"success": "false", "error": "no"}),
            json!({"success": 0, "error": "no"}),
            json!([{"success": false}]),
            json!("success=false"),
            json!(null),
        ] {
            assert_eq!(tool_failure_reason(&output), None, "{output}");
        }
    }
    #[test]
    fn a_raising_call_reports_the_tool_message_the_model_saw() {
        assert_eq!(
            tool_result_for_exception(Some("Error: no such file"), "no such file"),
            "Error: no such file"
        );
    }
    #[test]
    fn a_raising_call_without_a_tool_message_gets_the_prefix() {
        assert_eq!(
            tool_result_for_exception(None, "KeyError: 'path'"),
            "Ability execution error: KeyError: 'path'"
        );
        // Python's `if content:` is truthiness, so an empty content falls through.
        assert_eq!(
            tool_result_for_exception(Some(""), "boom"),
            "Ability execution error: boom"
        );
    }
}
