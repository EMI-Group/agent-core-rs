//! Redaction helpers for structured logs, URLs, and HTTP headers.
//!
//! Port of Python `openjiuwen/core/common/logging/events.py::sanitize_event_for_logging`,
//! `utils/url_utils.py::redact_url_password`, and `utils/header_utils.py::sanitize_headers`.
#![allow(clippy::module_name_repetitions)]

use std::collections::BTreeMap;

use serde_json::{Map, Value};
use url::Url;

/// Placeholder substituted for every redacted value (`"<REDACTED>"`).
///
/// Re-exported from [`crate::constants::REDACTED`] so `security` and `logging`
/// share exactly one placeholder.
pub use crate::constants::REDACTED;

/// Field names whose values are replaced by [`REDACTED`] when logging events.
///
/// Python `logging/events.py::sanitize_event_for_logging` default list, in the
/// same order.
pub const DEFAULT_SENSITIVE_FIELDS: &[&str] = &[
    "messages",
    "response_content",
    "input_content",
    "query",
    "arguments",
    "result",
    "message_content",
    "tool_calls",
    "input_data",
    "output_data",
    "retrieved_memories",
];

/// Headers that must never be forwarded verbatim (Python `header_utils.PROTECTED_HEADERS`).
const PROTECTED_HEADERS: &[&str] = &[
    "host",
    "content-length",
    "transfer-encoding",
    "connection",
    "authorization",
];

/// Returns a copy of `value` with every sensitive field redacted, at any depth.
///
/// Every object key that exactly matches an entry of `sensitive` has its value
/// replaced with [`REDACTED`]; the walk descends into nested objects and arrays.
/// `None` selects [`DEFAULT_SENSITIVE_FIELDS`].
///
/// ```
/// use serde_json::json;
/// use openjiuwen_common::security::redact_json;
///
/// let logged = redact_json(&json!({"query": "secret", "nested": {"result": 1}}), None);
/// assert_eq!(logged["query"], "<REDACTED>");
/// assert_eq!(logged["nested"]["result"], "<REDACTED>");
/// ```
#[must_use]
pub fn redact_json(value: &Value, sensitive: Option<&[&str]>) -> Value {
    redact_value(value, sensitive.unwrap_or(DEFAULT_SENSITIVE_FIELDS))
}

fn redact_value(value: &Value, sensitive: &[&str]) -> Value {
    match value {
        Value::Object(map) => {
            let mut redacted = Map::with_capacity(map.len());
            for (key, child) in map {
                let entry = if sensitive.contains(&key.as_str()) {
                    Value::String(REDACTED.to_owned())
                } else {
                    redact_value(child, sensitive)
                };
                redacted.insert(key.clone(), entry);
            }
            Value::Object(redacted)
        }
        Value::Array(items) => Value::Array(
            items
                .iter()
                .map(|item| redact_value(item, sensitive))
                .collect(),
        ),
        other => other.clone(),
    }
}

/// Redacts a free-form string value: empty stays empty, anything else becomes [`REDACTED`].
#[must_use]
pub fn redact_str(value: &str) -> String {
    if value.is_empty() {
        String::new()
    } else {
        REDACTED.to_owned()
    }
}

/// Masks the password of a URL so the URL can be logged safely.
///
/// Port of Python `utils/url_utils.redact_url_password`: a blank string, an
/// unparseable URL, or a URL without a password is returned unchanged; otherwise
/// only the password is replaced with `***`, preserving scheme, user name, host,
/// port, path, query, and fragment.
///
/// ```
/// use openjiuwen_common::security::redact_url_password;
///
/// assert_eq!(
///     redact_url_password("redis://user:secret@host:6379/0"),
///     "redis://user:***@host:6379/0"
/// );
/// ```
#[must_use]
pub fn redact_url_password(url: &str) -> String {
    if url.trim().is_empty() {
        return url.to_owned();
    }

    let Ok(parsed) = Url::parse(url) else {
        return url.to_owned();
    };
    match parsed.password() {
        Some(password) if !password.is_empty() => {}
        _ => return url.to_owned(),
    }

    let Some(scheme_end) = url.find("://") else {
        return url.to_owned();
    };
    let authority_start = scheme_end + 3;
    let rest = &url[authority_start..];
    let authority_end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
    let authority = &rest[..authority_end];
    let Some(at) = authority.rfind('@') else {
        return url.to_owned();
    };
    let userinfo = &authority[..at];
    let host_and_port = &authority[at + 1..];
    let Some(colon) = userinfo.find(':') else {
        return url.to_owned();
    };

    let prefix = &url[..authority_start];
    let username = &userinfo[..colon];
    let suffix = &rest[authority_end..];
    format!("{prefix}{username}:***@{host_and_port}{suffix}")
}

/// Drops blank and protected headers, preserving the casing of the rest.
///
/// Port of Python `utils/header_utils.sanitize_headers`: keys and values that are
/// blank after trimming are dropped, protected headers (`host`, `content-length`,
/// `transfer-encoding`, `connection`, `authorization`) are dropped
/// case-insensitively, and every other key keeps its original casing.
#[must_use]
pub fn sanitize_headers(headers: &BTreeMap<String, String>) -> BTreeMap<String, String> {
    headers
        .iter()
        .filter_map(|(key, value)| {
            let key = key.trim();
            if key.is_empty() || PROTECTED_HEADERS.contains(&key.to_ascii_lowercase().as_str()) {
                return None;
            }
            if value.trim().is_empty() {
                return None;
            }
            Some((key.to_owned(), value.clone()))
        })
        .collect()
}

/// Compares two byte slices in constant time with respect to their contents.
///
/// A length mismatch returns `false` immediately (the length of a secret is not
/// treated as sensitive). Otherwise every byte pair is folded into an accumulator
/// and the accumulator is compared to zero through [`std::hint::black_box`], so
/// the comparison does not short-circuit on the first differing byte.
#[must_use]
pub fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }

    let mut accumulator = 0_u8;
    for (left, right) in a.iter().zip(b.iter()) {
        accumulator |= left ^ right;
    }
    std::hint::black_box(accumulator) == 0
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use serde_json::json;

    use super::*;

    // port of tests/unit_tests/core/common/logging/test_events.py (sanitize_event_for_logging)
    #[test]
    fn redacts_nested_json_with_default_fields() {
        let value = json!({
            "message": "kept",
            "query": "secret query",
            "nested": {"result": 42, "fine": true},
            "items": [{"arguments": {"a": 1}}, {"fine": "ok"}],
        });

        let redacted = redact_json(&value, None);

        assert_eq!(redacted["message"], json!("kept"));
        assert_eq!(redacted["query"], json!(REDACTED));
        assert_eq!(redacted["nested"]["result"], json!(REDACTED));
        assert_eq!(redacted["nested"]["fine"], json!(true));
        assert_eq!(redacted["items"][0]["arguments"], json!(REDACTED));
        assert_eq!(redacted["items"][1]["fine"], json!("ok"));
    }

    #[test]
    fn redacts_only_the_requested_fields() {
        let value = json!({"query": "q", "result": "r", "other": "o"});

        let redacted = redact_json(&value, Some(&["query"]));

        assert_eq!(redacted["query"], json!(REDACTED));
        assert_eq!(redacted["result"], json!("r"));
        assert_eq!(redacted["other"], json!("o"));
    }

    #[test]
    fn redacts_default_field_names_exactly() {
        assert_eq!(
            DEFAULT_SENSITIVE_FIELDS,
            &[
                "messages",
                "response_content",
                "input_content",
                "query",
                "arguments",
                "result",
                "message_content",
                "tool_calls",
                "input_data",
                "output_data",
                "retrieved_memories",
            ]
        );
        assert_eq!(REDACTED, "<REDACTED>");

        // Matching is exact and case-sensitive.
        let redacted = redact_json(&json!({"Query": "q", "query": "q"}), None);
        assert_eq!(redacted["Query"], json!("q"));
        assert_eq!(redacted["query"], json!(REDACTED));
    }

    #[test]
    fn redacts_strings() {
        assert_eq!(redact_str(""), "");
        assert_eq!(redact_str("anything"), REDACTED);
    }

    // port of tests/unit_tests/core/common/utils/test_url_utils.py
    #[test]
    fn masks_url_password_with_username() {
        assert_eq!(
            redact_url_password("redis://user:secret@host:6379/0"),
            "redis://user:***@host:6379/0"
        );
        assert_eq!(
            redact_url_password("mysql://root:password123@localhost:3306/mydb"),
            "mysql://root:***@localhost:3306/mydb"
        );
        assert_eq!(
            redact_url_password("postgresql://admin:secretpass@db.example.com:5432/production"),
            "postgresql://admin:***@db.example.com:5432/production"
        );
    }

    #[test]
    fn masks_password_only_userinfo() {
        assert_eq!(
            redact_url_password("redis://:secret@host:6379/0"),
            "redis://:***@host:6379/0"
        );
        assert_eq!(
            redact_url_password("redis://:My%23SecretPwd@127.0.0.1:6379/0"),
            "redis://:***@127.0.0.1:6379/0"
        );
        assert_eq!(
            redact_url_password("redis://:password@localhost/0"),
            "redis://:***@localhost/0"
        );
    }

    #[test]
    fn preserves_non_password_parts() {
        assert_eq!(
            redact_url_password("redis://:secret@host:6379/0?ssl=true#frag"),
            "redis://:***@host:6379/0?ssl=true#frag"
        );
        assert_eq!(
            redact_url_password("https://user:secret@example.com:8443/a/b"),
            "https://user:***@example.com:8443/a/b"
        );
    }

    #[test]
    fn leaves_urls_without_password_unchanged() {
        for url in [
            "",
            "   ",
            "redis://host:6379/0",
            "redis://localhost:6379/1",
            "redis://user@host:6379/0",
            "not a valid url at all",
        ] {
            assert_eq!(redact_url_password(url), url);
        }
    }

    // port of tests/unit_tests/core/common/utils/test_header_utils.py
    #[test]
    fn sanitize_headers_drops_protected_and_blank_entries() {
        let headers = BTreeMap::from([
            ("Content-Type".to_owned(), "application/json".to_owned()),
            ("Authorization".to_owned(), "Bearer x".to_owned()),
            ("authorization".to_owned(), "Bearer y".to_owned()),
            ("HOST".to_owned(), "example.com".to_owned()),
            ("Content-Length".to_owned(), "10".to_owned()),
            ("Transfer-Encoding".to_owned(), "chunked".to_owned()),
            ("Connection".to_owned(), "keep-alive".to_owned()),
            ("X-Empty".to_owned(), "   ".to_owned()),
            ("  ".to_owned(), "value".to_owned()),
            ("X-Custom".to_owned(), "  spaced  ".to_owned()),
        ]);

        let sanitized = sanitize_headers(&headers);

        assert_eq!(
            sanitized,
            BTreeMap::from([
                ("Content-Type".to_owned(), "application/json".to_owned()),
                ("X-Custom".to_owned(), "  spaced  ".to_owned()),
            ])
        );
    }

    #[test]
    fn sanitize_headers_on_empty_input() {
        assert!(sanitize_headers(&BTreeMap::new()).is_empty());
    }

    #[test]
    fn constant_time_eq_matches_normal_equality() {
        assert!(constant_time_eq(b"", b""));
        assert!(constant_time_eq(b"secret", b"secret"));
        assert!(!constant_time_eq(b"secret", b"secreT"));
        assert!(!constant_time_eq(b"secret", b"secret "));
        assert!(!constant_time_eq(b"", b"a"));
    }
}
