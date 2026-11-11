//! String helpers: log-message sanitization, char-safe truncation, log filename
//! formatting, and validation-error rendering.
//!
//! Ports Python `logging/base_impl._sanitize_message`,
//! `security/exception_utils.ExceptionUtils.format_validation_error`, and the
//! log-file naming conventions of the logging backends.

use std::fmt::Write as _;
use std::path::Path;

use chrono::{DateTime, Utc};

use crate::utils::id::now_utc;

/// Escapes control characters in a plain log message.
///
/// `\r`, `\n`, `\t`, backspace, vertical tab, form feed, and NUL become their
/// two-character escape sequences; every other character below `0x20` or equal
/// to `0x7f` becomes `\xNN` (lowercase, two digits). All other characters,
/// including non-ASCII UTF-8, pass through unchanged.
///
/// ```
/// use openjiuwen_common::utils::string::sanitize_message;
///
/// assert_eq!(sanitize_message("a\nb"), "a\\nb");
/// assert_eq!(sanitize_message("a\u{1f}b"), "a\\x1fb");
/// ```
#[must_use]
pub fn sanitize_message(message: &str) -> String {
    let mut sanitized = String::with_capacity(message.len());
    for ch in message.chars() {
        match ch {
            '\r' => sanitized.push_str("\\r"),
            '\n' => sanitized.push_str("\\n"),
            '\t' => sanitized.push_str("\\t"),
            '\u{0008}' => sanitized.push_str("\\b"),
            '\u{000b}' => sanitized.push_str("\\v"),
            '\u{000c}' => sanitized.push_str("\\f"),
            '\0' => sanitized.push_str("\\0"),
            ch if (ch as u32) < 0x20 || (ch as u32) == 0x7f => {
                let _ = write!(sanitized, "\\x{:02x}", ch as u32);
            }
            ch => sanitized.push(ch),
        }
    }
    sanitized
}

/// Truncates `value` to at most `max_chars` characters without an ellipsis.
///
/// Truncation respects UTF-8 char boundaries; an already-short input is
/// returned unchanged.
///
/// ```
/// use openjiuwen_common::utils::string::truncate_chars;
///
/// assert_eq!(truncate_chars("héllo", 2), "hé");
/// assert_eq!(truncate_chars("hi", 10), "hi");
/// ```
#[must_use]
pub fn truncate_chars(value: &str, max_chars: usize) -> String {
    match value.char_indices().nth(max_chars) {
        Some((boundary, _)) => value[..boundary].to_string(),
        None => value.to_string(),
    }
}

/// Expands a log-filename pattern using the current UTC time.
///
/// Supported placeholders:
///
/// - `{name}` — the base file stem;
/// - `{ext}` — the base extension without a dot (default `log`);
/// - `{pid}` — the current process id;
/// - `{timestamp=%Y%m%d%H%M%S}`, `{date=%Y%m%d}`, `{time=%H%M%S}`,
///   `{datetime=%Y-%m-%d_%H-%M-%S}` — the text after `=` is a chrono strftime
///   pattern; the shown default applies when no pattern is supplied.
///
/// Unknown placeholders expand to the base file stem. If the expanded result
/// has no extension, `.{ext}` is appended; plain text without placeholders is
/// treated the same way.
///
/// ```
/// use openjiuwen_common::utils::string::format_log_filename;
///
/// assert_eq!(format_log_filename("app.log", "{name}"), "app.log");
/// assert_eq!(format_log_filename("app", "server"), "server.log");
/// ```
#[must_use]
pub fn format_log_filename(base: &str, pattern: &str) -> String {
    let base_path = Path::new(base);
    let stem = base_path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or(base);
    let extension = base_path
        .extension()
        .and_then(|ext| ext.to_str())
        .filter(|ext| !ext.is_empty())
        .unwrap_or("log");

    let now = now_utc();
    let mut rendered = String::with_capacity(pattern.len() + 16);
    let mut position = 0;
    while position < pattern.len() {
        let remaining = &pattern[position..];
        let Some(open) = remaining.find('{') else {
            rendered.push_str(remaining);
            break;
        };

        rendered.push_str(&remaining[..open]);
        let token_start = position + open + 1;
        if let Some(close) = pattern[token_start..].find('}') {
            let token = &pattern[token_start..token_start + close];
            rendered.push_str(&expand_placeholder(token, stem, extension, &now));
            position = token_start + close + 1;
        } else {
            rendered.push_str(&pattern[position + open..]);
            break;
        }
    }

    if Path::new(&rendered).extension().is_some() {
        rendered
    } else {
        format!("{rendered}.{extension}")
    }
}

/// Renders `(location, message)` validation errors as newline-separated lines.
///
/// This is a port of Python
/// `security.exception_utils.ExceptionUtils.format_validation_error`, which
/// emits one `"<location>: <message>"` line per error.
///
/// ```
/// use openjiuwen_common::utils::string::format_validation_errors;
///
/// let errors = vec![
///     ("body.name".to_string(), "field required".to_string()),
///     ("body.age".to_string(), "must be positive".to_string()),
/// ];
/// assert_eq!(
///     format_validation_errors(errors),
///     "body.name: field required\nbody.age: must be positive"
/// );
/// ```
#[must_use]
pub fn format_validation_errors<I>(errors: I) -> String
where
    I: IntoIterator<Item = (String, String)>,
{
    errors
        .into_iter()
        .map(|(location, message)| format!("{location}: {message}"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Expands one `{...}` placeholder token for [`format_log_filename`].
fn expand_placeholder(token: &str, stem: &str, extension: &str, now: &DateTime<Utc>) -> String {
    let (key, format) = match token.split_once('=') {
        Some((key, format)) => (key.trim(), Some(format)),
        None => (token.trim(), None),
    };

    match key {
        "ext" => extension.to_string(),
        "pid" => std::process::id().to_string(),
        "timestamp" => now.format(format.unwrap_or("%Y%m%d%H%M%S")).to_string(),
        "date" => now.format(format.unwrap_or("%Y%m%d")).to_string(),
        "time" => now.format(format.unwrap_or("%H%M%S")).to_string(),
        "datetime" => now
            .format(format.unwrap_or("%Y-%m-%d_%H-%M-%S"))
            .to_string(),
        // `name` and unknown placeholders both expand to the base file stem.
        _ => stem.to_string(),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    // port of tests/unit_tests/core/common/logging/test_base_impl.py

    #[test]
    fn sanitize_escapes_named_control_characters() {
        assert_eq!(sanitize_message("a\rb"), "a\\rb");
        assert_eq!(sanitize_message("a\nb"), "a\\nb");
        assert_eq!(sanitize_message("a\tb"), "a\\tb");
        assert_eq!(sanitize_message("a\u{0008}b"), "a\\bb");
        assert_eq!(sanitize_message("a\u{000b}b"), "a\\vb");
        assert_eq!(sanitize_message("a\u{000c}b"), "a\\fb");
        assert_eq!(sanitize_message("a\0b"), "a\\0b");
    }

    #[test]
    fn sanitize_escapes_other_control_characters_as_hex() {
        assert_eq!(sanitize_message("a\u{1f}b"), "a\\x1fb");
        assert_eq!(sanitize_message("\u{01}"), "\\x01");
        assert_eq!(sanitize_message("\u{7f}"), "\\x7f");
        assert_eq!(sanitize_message("\u{0002}\u{0003}"), "\\x02\\x03");
    }

    #[test]
    fn sanitize_preserves_regular_and_utf8_characters() {
        assert_eq!(sanitize_message("plain text"), "plain text");
        assert_eq!(sanitize_message("héllo 世界"), "héllo 世界");
        assert_eq!(sanitize_message(""), "");
        assert_eq!(sanitize_message("emoji 🚀"), "emoji 🚀");
    }

    #[test]
    fn truncate_respects_char_boundaries() {
        assert_eq!(truncate_chars("héllo", 2), "hé");
        assert_eq!(truncate_chars("héllo", 0), "");
        assert_eq!(truncate_chars("héllo", 5), "héllo");
        assert_eq!(truncate_chars("héllo", 99), "héllo");
        assert_eq!(truncate_chars("世界", 1), "世");
        assert_eq!(truncate_chars("", 3), "");
    }

    #[test]
    fn format_log_filename_expands_named_placeholders() {
        let name = std::process::id().to_string();
        assert_eq!(
            format_log_filename("app.log", "{pid}"),
            format!("{name}.log")
        );
        assert_eq!(
            format_log_filename("app.log", "app-{pid}"),
            format!("app-{name}.log")
        );
        assert_eq!(format_log_filename("app.log", "{ext}"), "log.log");
        assert_eq!(format_log_filename("app.txt", "{name}.txt"), "app.txt");
        assert_eq!(format_log_filename("app.txt", "{name}"), "app.txt");
        assert_eq!(format_log_filename("app.log", "{ext}.out"), "log.out");
    }

    #[test]
    fn format_log_filename_expands_time_placeholders() {
        let date = format_log_filename("app", "{date}");
        let stamp = date
            .strip_suffix(".log")
            .expect("date placeholder should keep the default extension");
        assert_eq!(stamp.len(), 8);
        assert!(stamp.bytes().all(|byte| byte.is_ascii_digit()));

        let timestamp = format_log_filename("app", "run-{timestamp=%Y%m%d%H%M%S}");
        let stamp = timestamp
            .strip_prefix("run-")
            .and_then(|rest| rest.strip_suffix(".log"))
            .expect("timestamp placeholder should be rendered");
        assert_eq!(stamp.len(), 14);
        assert!(stamp.bytes().all(|byte| byte.is_ascii_digit()));

        let time = format_log_filename("app", "{time=%H-%M}");
        let stamp = time
            .strip_suffix(".log")
            .expect("time placeholder should keep the default extension");
        assert_eq!(stamp.len(), 5);
        assert_eq!(&stamp[2..3], "-");

        let datetime = format_log_filename("app", "{datetime}");
        let stamp = datetime
            .strip_suffix(".log")
            .expect("datetime placeholder should keep the default extension");
        assert_eq!(stamp.len(), 19);
        assert_eq!(&stamp[4..5], "-");
        assert_eq!(&stamp[10..11], "_");
    }

    #[test]
    fn format_log_filename_falls_back_to_the_file_stem() {
        assert_eq!(format_log_filename("app.log", "{unknown}"), "app.log");
        assert_eq!(format_log_filename("app.log", "{unknown=.txt}"), "app.log");
        assert_eq!(
            format_log_filename("app.log", "x-{unknown}-y"),
            "x-app-y.log"
        );
    }

    #[test]
    fn format_log_filename_handles_plain_text_and_extensionless_results() {
        assert_eq!(format_log_filename("app.log", "server"), "server.log");
        assert_eq!(format_log_filename("app.log", "server.log"), "server.log");
        assert_eq!(format_log_filename("app.log", "server.txt"), "server.txt");
        assert_eq!(format_log_filename("app", ""), ".log");
        assert_eq!(format_log_filename("app.log", "{name}-"), "app-.log");
    }

    #[test]
    fn format_log_filename_keeps_unbalanced_braces() {
        assert_eq!(format_log_filename("app.log", "a{b"), "a{b.log");
        assert_eq!(format_log_filename("app.log", "a}b"), "a}b.log");
    }

    #[test]
    fn validation_errors_join_with_newlines() {
        let errors = vec![
            ("body.name".to_string(), "field required".to_string()),
            ("body.age".to_string(), "must be positive".to_string()),
        ];
        assert_eq!(
            format_validation_errors(errors),
            "body.name: field required\nbody.age: must be positive"
        );
        assert_eq!(format_validation_errors(Vec::new()), "");
        assert_eq!(
            format_validation_errors([("x".to_string(), "bad".to_string())]),
            "x: bad"
        );
    }
}
