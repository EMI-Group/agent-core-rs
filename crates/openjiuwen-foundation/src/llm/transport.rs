//! Transport helpers shared by the OpenAI-compatible and Anthropic clients.
//!
//! The two clients differ in their wire bodies and stream events but share the
//! same URL joining, status-to-error mapping, and log-safe error bodies. Keeping
//! those three in one place guarantees that neither client can leak a credential
//! or report a timeout as a generic provider failure.
//!
//! Everything here is `pub(crate)`: it is an implementation seam between
//! [`crate::llm::openai`], [`crate::llm::anthropic`], and their tests.

// The two clients consume these helpers; while they are still stubs the crate
// cannot see the call sites, so the dead-code lint is silenced here rather than
// by duplicating the helpers into each client.
#![allow(dead_code)]

use openjiuwen_common::error::Error;
use openjiuwen_common::security::redact_str;

/// Maximum number of characters of a provider body kept in an error message.
const MAX_BODY_CHARS: usize = 1000;

/// Appended to a body that had to be truncated.
const TRUNCATION_MARKER: char = '…';

/// Joins a base URL and an endpoint path with exactly one `/` between them.
///
/// The base is trimmed and its trailing slashes are removed; the path is trimmed
/// and all leading slashes are removed. A base with no path yields the base
/// alone, and an empty base yields `/<path>`, so both clients can pass a
/// configured `api_base` through unchanged.
#[must_use]
pub(crate) fn join_url(base: &str, path: &str) -> String {
    let base = base.trim().trim_end_matches('/');
    let path = path.trim().trim_start_matches('/');

    if path.is_empty() {
        return base.to_owned();
    }
    if base.is_empty() {
        return format!("/{path}");
    }
    format!("{base}/{path}")
}

/// Maps a non-success HTTP status to the error kind the SDK promises callers.
///
/// `401`/`403` become [`Error::Config`] (an authentication problem, which no
/// retry can fix), `408`/`504` become [`Error::Timeout`], and every other status
/// becomes [`Error::Llm`]. The response body is passed through
/// [`redacted_body`] before it reaches the message, so a provider that echoes the
/// request (including its `Authorization` header) cannot leak it into a log.
pub(crate) fn error_for_status(status: u16, url: &str, body: &str) -> Error {
    let detail = redacted_body(body);
    let suffix = if detail.is_empty() {
        String::new()
    } else {
        format!(": {detail}")
    };

    match status {
        401 | 403 => Error::config(format!(
            "authentication failed (HTTP {status}) for {url}{suffix}"
        )),
        408 | 504 => Error::timeout(format!(
            "request timed out (HTTP {status}) for {url}{suffix}"
        )),
        _ => Error::llm(format!(
            "provider request failed with HTTP {status} for {url}{suffix}"
        )),
    }
}

/// Renders a provider response body safe to put in an error message.
///
/// The body is redacted through [`redact_str`] (a non-empty body becomes the
/// shared `<REDACTED>` placeholder) and then truncated to [`MAX_BODY_CHARS`]
/// characters on a character boundary, with [`TRUNCATION_MARKER`] appended when
/// anything was cut. Truncation therefore never splits a multi-byte character.
#[must_use]
pub(crate) fn redacted_body(body: &str) -> String {
    truncate_chars(&redact_str(body), MAX_BODY_CHARS)
}

/// Truncates `text` to at most `max` characters, appending the marker when cut.
fn truncate_chars(text: &str, max: usize) -> String {
    let mut truncated: String = text.chars().take(max).collect();
    if truncated.chars().count() < text.chars().count() {
        truncated.push(TRUNCATION_MARKER);
    }
    truncated
}

#[cfg(test)]
mod tests {
    use super::*;
    use openjiuwen_common::error::ErrorKind;
    use openjiuwen_common::security::REDACTED;

    #[test]
    fn join_url_puts_exactly_one_slash_between_base_and_path() {
        assert_eq!(
            join_url("https://api.openai.com/v1", "/chat/completions"),
            "https://api.openai.com/v1/chat/completions"
        );
        assert_eq!(
            join_url("https://api.openai.com/v1/", "chat/completions"),
            "https://api.openai.com/v1/chat/completions"
        );
        assert_eq!(
            join_url("  https://api.openai.com/v1///  ", "///chat/completions"),
            "https://api.openai.com/v1/chat/completions"
        );
        assert_eq!(
            join_url("https://api.anthropic.com", "/v1/messages"),
            "https://api.anthropic.com/v1/messages"
        );
    }

    #[test]
    fn join_url_handles_empty_parts() {
        assert_eq!(
            join_url("https://api.openai.com/v1", ""),
            "https://api.openai.com/v1"
        );
        assert_eq!(
            join_url("https://api.openai.com/v1/", "  "),
            "https://api.openai.com/v1"
        );
        assert_eq!(join_url("", "/chat/completions"), "/chat/completions");
        assert_eq!(join_url("   ", "chat/completions"), "/chat/completions");
        assert_eq!(join_url("", ""), "");
    }

    #[test]
    fn unauthorized_statuses_are_config_errors() {
        for status in [401, 403] {
            let error = error_for_status(status, "https://api.example/v1/chat", "nope");
            assert_eq!(error.kind(), ErrorKind::Config, "status {status}");
            let message = error.to_string();
            assert!(
                message.contains("authentication failed"),
                "message: {message}"
            );
            assert!(
                message.contains("https://api.example/v1/chat"),
                "message: {message}"
            );
        }
    }

    #[test]
    fn timeout_statuses_are_timeout_errors() {
        for status in [408, 504] {
            let error = error_for_status(status, "https://api.example/v1/chat", "");
            assert_eq!(error.kind(), ErrorKind::Timeout, "status {status}");
            let message = error.to_string();
            assert!(message.contains("timed out"), "message: {message}");
            assert!(
                message.contains("https://api.example/v1/chat"),
                "message: {message}"
            );
        }
    }

    #[test]
    fn other_statuses_are_llm_errors() {
        for status in [400, 404, 429, 500, 502, 503] {
            let error = error_for_status(status, "https://api.example/v1/chat", "boom");
            assert_eq!(error.kind(), ErrorKind::Llm, "status {status}");
            let message = error.to_string();
            assert!(message.contains(&status.to_string()), "message: {message}");
            assert!(
                message.contains("https://api.example/v1/chat"),
                "message: {message}"
            );
            assert!(message.contains(REDACTED), "message: {message}");
        }
    }

    #[test]
    fn error_bodies_are_redacted() {
        let body = r#"{"error":"invalid api key sk-live-secret"}"#;
        let error = error_for_status(401, "https://api.example/v1/chat", body);
        let message = error.to_string();

        assert!(!message.contains("sk-live-secret"), "message: {message}");
        assert!(!message.contains("invalid api key"), "message: {message}");
        assert!(message.contains(REDACTED), "message: {message}");
    }

    #[test]
    fn empty_bodies_leave_no_trailing_separator() {
        let error = error_for_status(500, "https://api.example/v1/chat", "");
        assert_eq!(
            error.to_string(),
            "llm: provider request failed with HTTP 500 for https://api.example/v1/chat"
        );
    }

    #[test]
    fn redacted_body_masks_any_non_empty_body() {
        assert_eq!(redacted_body(""), "");
        assert_eq!(redacted_body("   "), REDACTED);
        assert_eq!(redacted_body("{\"error\": \"secret\"}"), REDACTED);
    }

    #[test]
    fn truncate_chars_keeps_char_boundaries_and_marks_the_cut() {
        assert_eq!(truncate_chars("short", 10), "short");
        assert_eq!(truncate_chars("exactly", 7), "exactly");

        let truncated = truncate_chars("abcdefgh", 4);
        assert_eq!(truncated, "abcd…");
        assert!(truncated.ends_with(TRUNCATION_MARKER));

        // Multi-byte characters are never split: each counts as one character.
        let japanese = "模型响应错误信息";
        let cut = truncate_chars(japanese, 3);
        assert_eq!(cut, "模型响…");
        assert_eq!(cut.chars().count(), 4);
    }

    #[test]
    fn redacted_body_truncation_is_bounded_by_the_configured_limit() {
        // The redaction placeholder is shorter than the limit, so the bound is
        // enforced by `truncate_chars`, exercised above; assert the wiring here.
        let body = "x".repeat(MAX_BODY_CHARS * 2);
        let rendered = redacted_body(&body);
        assert_eq!(rendered, REDACTED);
        assert!(rendered.chars().count() <= MAX_BODY_CHARS);
    }
}
