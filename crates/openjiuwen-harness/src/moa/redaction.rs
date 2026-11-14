//! Credential redaction for text handed to reference models and verifiers.
//!
//! Ports `openjiuwen/harness/moa/verifier/redaction.py`. Tool arguments and tool results
//! are the likeliest carriers of API keys and tokens, so every transcript, problem and
//! step summary is redacted before it leaves the process.
//!
//! Two passes, in the Python order: the first removes the *value* of a JSON-shaped secret
//! while keeping the JSON syntax intact; the second removes bare credential forms
//! (`Bearer …`, `api_key=…`, `token: …`, `sk-…`).
use std::sync::LazyLock;

use regex::Regex;

/// JSON-shaped secrets: `"api_key": "…"`, `"token": "…"`, `"password": "…"`.
///
/// The value pattern consumes escapes (`\\.`) so a quoted secret with an escaped quote is
/// removed as a whole instead of being cut at the first escape.
#[allow(clippy::expect_used)] // the patterns are compile-time constants; a typo fails loudly at first use
static JSON_SECRET: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(?i)("(?:api[_-]?key|token|password)"\s*:\s*)"(?:\\.|[^"\\])*""#)
        .expect("JSON secret pattern is valid")
});

/// Bare credential forms: bearer tokens, `key=value` / `key: value` assignments, `sk-` keys.
#[allow(clippy::expect_used)] // the patterns are compile-time constants; a typo fails loudly at first use
static BARE_SECRET: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)(bearer\s+|(?:api[_-]?key|token|password)\s*[:=]\s*|sk-)[A-Za-z0-9_./+=-]+")
        .expect("bare secret pattern is valid")
});

/// Replaces `[redacted]`, the marker both passes leave behind.
const REDACTED: &str = "[redacted]";

/// Removes common credential forms from `value`.
///
/// Returns `value` unchanged (as a copy) when it contains no credential form. The result
/// is always plain text: no error path, no I/O, no logging of the removed content.
///
/// # Examples
///
/// ```
/// use openjiuwen_harness::moa::redact;
///
/// assert_eq!(
///     redact("{\"api_key\": \"sk-live-123\"}"),
///     "{\"api_key\": \"[redacted]\"}"
/// );
/// assert_eq!(redact("Authorization: Bearer abc.def"), "Authorization: [redacted]");
/// assert_eq!(redact("the build is broken"), "the build is broken");
/// ```
#[must_use]
pub fn redact(value: &str) -> String {
    let json_safe = JSON_SECRET.replace_all(value, format!("$1\"{REDACTED}\""));
    BARE_SECRET.replace_all(&json_safe, REDACTED).into_owned()
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    #[test]
    fn removes_json_shaped_secret_values_but_keeps_json() {
        let redacted = redact(r#"{"api_key": "abcd-1234", "model": "gpt"}"#);
        assert_eq!(
            redacted,
            "{\"api_key\": \"[redacted]\", \"model\": \"gpt\"}"
        );

        let redacted = redact(r#"{"token":"x1","password": "p a s s"}"#);
        assert_eq!(
            redacted,
            "{\"token\":\"[redacted]\",\"password\": \"[redacted]\"}"
        );

        // Escaped quotes inside the value stay part of the removed secret.
        let redacted = redact(r#"{"api-key": "before \"quoted\" after"}"#);
        assert_eq!(redacted, "{\"api-key\": \"[redacted]\"}");
    }

    #[test]
    fn removes_bearer_tokens() {
        assert_eq!(
            redact("Authorization: Bearer eyJhbGciOi.abc-123_x"),
            "Authorization: [redacted]"
        );
        assert_eq!(redact("bearer TOKEN123"), "[redacted]");
    }

    #[test]
    fn removes_assignments_and_sk_keys() {
        assert_eq!(redact("api_key=abc123"), "[redacted]");
        assert_eq!(redact("api-key = abc123"), "[redacted]");
        assert_eq!(redact("token: abc/def+ghi="), "[redacted]");
        assert_eq!(redact("password: hunter2"), "[redacted]");
        assert_eq!(redact("key sk-abc123def"), "key [redacted]");
        assert_eq!(redact("use sk-proj-abc.def/ghi"), "use [redacted]");
    }

    #[test]
    fn leaves_ordinary_prose_untouched() {
        for text in [
            "the build is broken, fix src/main.rs",
            "a tokenizer is not a token here",
            "{}",
            "",
            "<reference id=\"reference-1\">x</reference>",
        ] {
            assert_eq!(redact(text), text);
        }
    }

    #[test]
    fn redacts_every_occurrence() {
        let redacted = redact("api_key=a and api_key=b");
        assert_eq!(redacted, "[redacted] and [redacted]");
    }
}
