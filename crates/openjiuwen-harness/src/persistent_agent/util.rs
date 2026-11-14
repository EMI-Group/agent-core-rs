//! Small, dependency-free helpers shared by the persistent-agent modules.
//!
//! The persistent-agent records are identified by content: event ids, operation ids and entry
//! digests are all derived from canonical JSON. This module is the single place where that
//! rendering lives, together with the percent-encoding used by session keys and the atomic file
//! helpers used by every store.
use std::fmt;
use std::io;
use std::path::{Path, PathBuf};

use openjiuwen_common::error::{Error, Result};
use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::Value;

/// Maximum length, in Unicode scalar values, of an identifier.
pub const IDENTIFIER_MAX_CHARS: usize = 512;

/// Renders `value` as canonical JSON: object keys sorted, no insignificant whitespace and
/// non-ASCII characters emitted as raw UTF-8.
///
/// The crate's `serde_json` build enables the `preserve_order` feature, so map iteration order
/// follows insertion order; this function therefore emits object keys through an explicitly
/// sorted list instead of relying on the map.
#[must_use]
pub fn canonical_json(value: &Value) -> String {
    let mut rendered = String::new();
    write_canonical(value, &mut rendered);
    rendered
}

/// Appends the canonical rendering of `value` to `output`.
fn write_canonical(value: &Value, output: &mut String) {
    match value {
        Value::Null => output.push_str("null"),
        Value::Bool(flag) => output.push_str(if *flag { "true" } else { "false" }),
        Value::Number(number) => output.push_str(&number.to_string()),
        Value::String(text) => output.push_str(&encode_string(text)),
        Value::Array(items) => {
            output.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    output.push(',');
                }
                write_canonical(item, output);
            }
            output.push(']');
        }
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort_unstable();
            output.push('{');
            for (index, key) in keys.into_iter().enumerate() {
                if index > 0 {
                    output.push(',');
                }
                output.push_str(&encode_string(key));
                output.push(':');
                if let Some(item) = map.get(key) {
                    write_canonical(item, output);
                }
            }
            output.push('}');
        }
    }
}

/// Encodes a JSON string with the usual escapes; non-ASCII characters stay raw.
fn encode_string(text: &str) -> String {
    // Serializing a `&str` never fails; the fallback only keeps the function total.
    serde_json::to_string(text).unwrap_or_else(|_| "\"\"".to_string())
}

/// Returns the lowercase hex SHA-256 digest of the canonical JSON of `value`.
#[must_use]
pub fn digest_hex(value: &Value) -> String {
    sha256_hex_str(&canonical_json(value))
}

/// Returns the lowercase hex SHA-256 digest of `text`.
#[must_use]
pub fn sha256_hex_str(text: &str) -> String {
    openjiuwen_sysop::util::sha256_hex(text.as_bytes())
}

/// Percent-encodes every byte that is not an RFC 3986 unreserved character.
///
/// The unreserved set (`A-Z`, `a-z`, `0-9`, `-`, `.`, `_`, `~`) passes through unchanged; every
/// other UTF-8 byte becomes an uppercase `%XX` escape, matching Python's `quote(value, safe="")`.
#[must_use]
pub fn percent_encode_identifier(value: &str) -> String {
    let mut encoded = String::with_capacity(value.len());
    for byte in value.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~') {
            encoded.push(char::from(byte));
        } else {
            const HEX: &[u8; 16] = b"0123456789ABCDEF";
            encoded.push('%');
            encoded.push(char::from(HEX[usize::from(byte >> 4)]));
            encoded.push(char::from(HEX[usize::from(byte & 0x0F)]));
        }
    }
    encoded
}

/// Returns whether `value` is a usable identifier.
///
/// An identifier has between 1 and [`IDENTIFIER_MAX_CHARS`] Unicode scalar values and contains
/// neither whitespace nor control characters.
#[must_use]
pub fn is_valid_identifier(value: &str) -> bool {
    let count = value.chars().count();
    count > 0
        && count <= IDENTIFIER_MAX_CHARS
        && !value
            .chars()
            .any(|ch| ch.is_whitespace() || ch.is_control())
}

/// Writes `bytes` to `path` atomically: parent directories are created, the bytes go to
/// `<path>.tmp`, and that file is renamed over `path`.
///
/// # Errors
/// `Error::Session` when a directory, the temporary file or the rename cannot be completed;
/// the message names the target path.
pub async fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let fail = |detail: &dyn fmt::Display| {
        Error::session(format!("failed to write {}: {detail}", path.display()))
    };
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|error| fail(&error))?;
        }
    }
    let temporary = temporary_path(path);
    tokio::fs::write(&temporary, bytes)
        .await
        .map_err(|error| fail(&error))?;
    if let Err(error) = tokio::fs::rename(&temporary, path).await {
        if let Err(cleanup) = tokio::fs::remove_file(&temporary).await {
            tracing::debug!(
                path = %temporary.display(),
                error = %cleanup,
                "failed to remove the temporary persistent-agent file"
            );
        }
        return Err(fail(&error));
    }
    Ok(())
}

/// Returns the sibling temporary path used by [`atomic_write`].
fn temporary_path(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_os_string();
    name.push(".tmp");
    PathBuf::from(name)
}

/// Serializes `value` as pretty JSON with a trailing newline and writes it atomically.
///
/// # Errors
/// `Error::Validation` when `value` cannot be encoded, or `Error::Session` on IO failure.
pub async fn write_json_atomic<T>(path: &Path, value: &T) -> Result<()>
where
    T: Serialize + Sync,
{
    let encoded = serde_json::to_vec_pretty(value).map_err(|error| {
        Error::validation(format!(
            "failed to encode JSON for {}: {error}",
            path.display()
        ))
    })?;
    let mut bytes = encoded;
    bytes.push(b'\n');
    atomic_write(path, &bytes).await
}

/// Writes `bytes` to `path` atomically from a blocking context.
///
/// This is the [`atomic_write`] counterpart used by callers that already run on
/// [`tokio::task::spawn_blocking`].
///
/// # Errors
/// The same errors as [`atomic_write`].
pub(crate) fn atomic_write_blocking(path: &Path, bytes: &[u8]) -> Result<()> {
    let fail = |detail: &dyn fmt::Display| {
        Error::session(format!("failed to write {}: {detail}", path.display()))
    };
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).map_err(|error| fail(&error))?;
        }
    }
    let temporary = temporary_path(path);
    std::fs::write(&temporary, bytes).map_err(|error| fail(&error))?;
    if let Err(error) = std::fs::rename(&temporary, path) {
        if let Err(cleanup) = std::fs::remove_file(&temporary) {
            tracing::debug!(
                path = %temporary.display(),
                error = %cleanup,
                "failed to remove the temporary persistent-agent file"
            );
        }
        return Err(fail(&error));
    }
    Ok(())
}

/// Serializes `value` as pretty JSON with a trailing newline and writes it atomically from a
/// blocking context.
///
/// This is the [`write_json_atomic`] counterpart used by callers that already run on
/// [`tokio::task::spawn_blocking`].
///
/// # Errors
/// The same errors as [`write_json_atomic`].
pub(crate) fn write_json_atomic_blocking<T>(path: &Path, value: &T) -> Result<()>
where
    T: Serialize,
{
    let encoded = serde_json::to_vec_pretty(value).map_err(|error| {
        Error::validation(format!(
            "failed to encode JSON for {}: {error}",
            path.display()
        ))
    })?;
    let mut bytes = encoded;
    bytes.push(b'\n');
    atomic_write_blocking(path, &bytes)
}

/// Reads and decodes a JSON file.
///
/// A missing file yields `Ok(None)`; any other failure is reported.
///
/// # Errors
/// `Error::Session` when the file cannot be read, or `Error::Validation` when its contents are
/// not valid JSON. Both messages name `path`.
pub async fn read_json<T>(path: &Path) -> Result<Option<T>>
where
    T: DeserializeOwned,
{
    let text = match tokio::fs::read_to_string(path).await {
        Ok(text) => text,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(Error::session(format!(
                "failed to read {}: {error}",
                path.display()
            )));
        }
    };
    serde_json::from_str(&text)
        .map(Some)
        .map_err(|error| Error::validation(format!("invalid JSON in {}: {error}", path.display())))
}

/// Reads and decodes a JSON file from a blocking context.
///
/// This is the [`read_json`] counterpart used by callers that already run on
/// [`tokio::task::spawn_blocking`].
///
/// # Errors
/// The same errors as [`read_json`].
pub(crate) fn read_json_blocking<T>(path: &Path) -> Result<Option<T>>
where
    T: DeserializeOwned,
{
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(Error::session(format!(
                "failed to read {}: {error}",
                path.display()
            )));
        }
    };
    serde_json::from_str(&text)
        .map(Some)
        .map_err(|error| Error::validation(format!("invalid JSON in {}: {error}", path.display())))
}

/// Truncates `text` to at most `max_chars` Unicode scalar values.
///
/// An ellipsis (`…`) is appended only when something was dropped, so an exact-length input is
/// returned unchanged and truncation never splits a character. The ellipsis is part of the budget,
/// never extra: a truncated result is exactly `max_chars` characters long, `max_chars == 0` yields
/// an empty string, and `max_chars == 1` yields the ellipsis alone.
#[must_use]
pub fn truncate_chars(text: &str, max_chars: usize) -> String {
    let mut chars = text.chars();
    let mut truncated: String = chars.by_ref().take(max_chars).collect();
    if chars.next().is_some() && !truncated.is_empty() {
        truncated.pop();
        truncated.push('…');
    }
    truncated
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use serde_json::json;

    use super::{
        atomic_write, canonical_json, digest_hex, is_valid_identifier, percent_encode_identifier,
        read_json, sha256_hex_str, truncate_chars, write_json_atomic, IDENTIFIER_MAX_CHARS,
    };

    #[test]
    fn canonical_json_sorts_nested_object_keys() {
        let value = json!({"b": 1, "a": {"d": 2, "c": [3, {"z": 0, "y": 1}]}});
        assert_eq!(
            canonical_json(&value),
            r#"{"a":{"c":[3,{"y":1,"z":0}],"d":2},"b":1}"#
        );
    }

    #[test]
    fn canonical_json_keeps_non_ascii_raw_and_escapes_strings() {
        let value = json!({"text": "héllo\n世界", "empty": "", "flag": true, "none": null});
        let rendered = canonical_json(&value);
        assert!(rendered.contains("héllo"));
        assert!(rendered.contains("世界"));
        assert!(rendered.contains("\\n"));
        assert!(!rendered.contains("\\u"));
        assert!(rendered.contains("\"flag\":true"));
        assert!(rendered.contains("\"none\":null"));
    }

    #[test]
    fn canonical_json_is_independent_of_insertion_order() {
        let first: serde_json::Value = serde_json::from_str(r#"{"b": 2, "a": 1}"#).unwrap();
        let second: serde_json::Value = serde_json::from_str(r#"{"a": 1, "b": 2}"#).unwrap();
        assert_eq!(canonical_json(&first), canonical_json(&second));
        assert_eq!(digest_hex(&first), digest_hex(&second));
        assert_eq!(digest_hex(&first).len(), 64);
        assert!(digest_hex(&first).chars().all(|ch| ch.is_ascii_hexdigit()));
    }

    #[test]
    fn digest_of_known_value_is_stable() {
        let value = json!({"kind": "note", "text": "hello"});
        assert_eq!(
            digest_hex(&value),
            sha256_hex_str(r#"{"kind":"note","text":"hello"}"#)
        );
    }

    #[test]
    fn sha256_of_empty_string_matches_the_known_vector() {
        assert_eq!(
            sha256_hex_str(""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn percent_encoding_matches_python_quote() {
        assert_eq!(percent_encode_identifier("a:b"), "a%3Ab");
        assert_eq!(percent_encode_identifier("s/1"), "s%2F1");
        assert_eq!(percent_encode_identifier("é"), "%C3%A9");
        assert_eq!(percent_encode_identifier("a-b_c.d~e"), "a-b_c.d~e");
        assert_eq!(percent_encode_identifier("100%"), "100%25");
        assert_eq!(percent_encode_identifier(""), "");
    }

    #[test]
    fn identifier_validation_checks_length_and_characters() {
        assert!(is_valid_identifier("agent_1"));
        assert!(is_valid_identifier(&"a".repeat(IDENTIFIER_MAX_CHARS)));
        assert!(!is_valid_identifier(&"a".repeat(IDENTIFIER_MAX_CHARS + 1)));
        assert!(!is_valid_identifier(""));
        assert!(!is_valid_identifier("has space"));
        assert!(!is_valid_identifier("has\ttab"));
        assert!(!is_valid_identifier("has\nnewline"));
        assert!(!is_valid_identifier("control\u{7f}"));
        assert!(!is_valid_identifier("non\u{00a0}breaking"));
        // A zero-width space is a format character, not Unicode whitespace.
        assert!(is_valid_identifier("zero\u{200b}width"));
    }

    #[test]
    fn truncate_chars_adds_an_ellipsis_only_when_truncated() {
        assert_eq!(truncate_chars("abcdef", 3), "ab…");
        assert_eq!(truncate_chars("abc", 3), "abc");
        assert_eq!(truncate_chars("abc", 9), "abc");
        assert_eq!(truncate_chars("", 3), "");
        assert_eq!(truncate_chars("héllo", 2), "h…");
        assert_eq!(truncate_chars("日本", 1), "…");
    }

    #[test]
    fn truncate_chars_never_exceeds_the_budget() {
        for (text, max_chars) in [
            ("abcdef", 0_usize),
            ("abcdef", 1),
            ("abcdef", 2),
            ("abcdef", 5),
            ("abcdef", 6),
            ("abcdef", 7),
            ("héllo", 3),
            ("世界你好", 4),
        ] {
            let truncated = truncate_chars(text, max_chars);
            assert!(
                truncated.chars().count() <= max_chars,
                "`{text}` truncated to {max_chars} chars produced `{truncated}`"
            );
        }
        assert_eq!(truncate_chars("abcdef", 0), "");
        assert_eq!(truncate_chars("abcdef", 1), "…");
        assert_eq!(truncate_chars("abcdef", 2), "a…");
        assert_eq!(truncate_chars("abcdef", 5), "abcd…");
        assert_eq!(truncate_chars("abcdef", 6), "abcdef");
        assert_eq!(truncate_chars("abcdef", 7), "abcdef");
        assert_eq!(truncate_chars("世界你好", 3), "世界…");
    }

    #[tokio::test]
    async fn atomic_write_replaces_an_existing_file_and_leaves_no_temporary() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("nested").join("record.json");
        atomic_write(&path, b"old").await.unwrap();
        assert_eq!(tokio::fs::read(&path).await.unwrap(), b"old");
        atomic_write(&path, b"new").await.unwrap();
        assert_eq!(tokio::fs::read(&path).await.unwrap(), b"new");
        assert!(!dir.path().join("nested").join("record.json.tmp").exists());
    }

    #[tokio::test]
    async fn json_round_trip_and_missing_file() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("state.json");
        assert!(read_json::<serde_json::Value>(&path)
            .await
            .unwrap()
            .is_none());
        let value = json!({"b": 2, "a": 1});
        write_json_atomic(&path, &value).await.unwrap();
        assert_eq!(
            read_json::<serde_json::Value>(&path).await.unwrap(),
            Some(value)
        );
    }

    #[tokio::test]
    async fn malformed_json_names_the_path() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("broken.json");
        tokio::fs::write(&path, b"{not json").await.unwrap();
        let error = read_json::<serde_json::Value>(&path).await.unwrap_err();
        assert!(matches!(
            error,
            openjiuwen_common::error::Error::Validation(_)
        ));
        assert!(error.to_string().contains("broken.json"));
    }
}
