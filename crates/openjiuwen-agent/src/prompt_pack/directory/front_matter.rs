//! YAML front matter of a prompt file and the `baseline_sha256` evolution rule.
//!
//! Ports `openjiuwen/agent_teams/team_workspace/frontmatter.py`: every evolved
//! prompt file is `YAML front matter + body`, where the front matter carries
//! `kind` / `name` / `language` and the write-time `baseline_sha256`. Reading
//! degrades gracefully — a document that does not start with a `---` line is a
//! hand-written body with empty metadata, and a front matter block without a
//! closing `---` line is treated the same way. Malformed YAML and a non-mapping
//! root are validation errors, because callers must not treat such a body as a
//! prompt value.
//!
//! Newlines are normalized to `\n` first (Python reads prompt files in text
//! mode, which does the same), and the body after the closing delimiter is kept
//! byte-faithfully so [`body_sha256`] round-trips with the hash stamped at write
//! time.
use super::sha256::sha256_hex;
use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::BTreeMap;

/// Priority a prompt-file section gets when its front matter does not set one.
pub const DEFAULT_PROMPT_PRIORITY: i32 = 100;

/// The YAML front matter of one prompt file.
///
/// Every field is optional; keys without a structured counterpart (including
/// future front-matter fields) are kept in [`PromptFileMeta::extra`] so a newer
/// pack can still be read. `evolved` is the *declared* write-time flag, which
/// the Python reference never trusts at read time; call [`is_evolved`] to decide
/// from the body hash instead.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct PromptFileMeta {
    /// Section name declared by the file.
    pub name: Option<String>,
    /// Prompt language declared by the file (`cn`, `en`, or another token).
    pub language: Option<String>,
    /// Ordering key; lower priorities render first.
    pub priority: Option<i32>,
    /// Asset kind (`prompt`, `identity`, `tool`, ...).
    pub kind: Option<String>,
    /// Write-time evolution flag; informational only.
    pub evolved: Option<bool>,
    /// SHA-256 of the body stamped when the file was assembled.
    pub baseline_sha256: Option<String>,
    /// Last update timestamp as written by the assembler.
    pub updated_at: Option<String>,
    /// Front-matter keys without a structured field.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// Returns the lowercase hex SHA-256 digest of `body`.
///
/// The body is hashed as UTF-8, matching Python's `body_sha256`.
#[must_use]
pub fn body_sha256(body: &str) -> String {
    sha256_hex(body.as_bytes())
}

/// Whether `body` has diverged from the baseline recorded in `meta`.
///
/// A file without `baseline_sha256` (hand-written, or no front matter at all)
/// is always evolved: its body wins. Otherwise the body hash must equal the
/// stamped baseline.
#[must_use]
pub fn is_evolved(meta: &PromptFileMeta, body: &str) -> bool {
    meta.baseline_sha256
        .as_deref()
        .is_none_or(|baseline| body_sha256(body) != baseline)
}

/// Splits a prompt document into its front matter and its body.
///
/// A document that does not start with a `---` line, or whose opening `---` is
/// never closed, is a hand-written body: the metadata is empty and the body is
/// the whole (newline-normalized) text.
///
/// # Errors
///
/// Returns [`Error::Validation`] when the block is not valid YAML, is not a
/// mapping at the root, or gives a known field the wrong type (including an
/// out-of-range or non-numeric `priority`).
pub fn read_frontmatter(text: &str) -> Result<(PromptFileMeta, String)> {
    read_frontmatter_from(text, "prompt document")
}

/// [`read_frontmatter`] naming `source` in every error message.
pub(crate) fn read_frontmatter_from(text: &str, source: &str) -> Result<(PromptFileMeta, String)> {
    let text = normalize_newlines(text);
    let Some((yaml, body)) = split_front_matter(&text) else {
        return Ok((PromptFileMeta::default(), text));
    };
    let meta = parse_meta(yaml, source)?;
    Ok((meta, body.to_owned()))
}

/// Splits `---`-delimited front matter from the body without interpreting it.
///
/// Returns `None` when the text does not open with a `---` line or has no
/// closing `---` line. The returned body starts right after the closing line's
/// newline, so its trailing newlines survive.
fn split_front_matter(text: &str) -> Option<(&str, &str)> {
    let mut lines = text.split_inclusive('\n');
    let opening = lines.next()?;
    if opening.trim() != "---" {
        return None;
    }
    let yaml_start = opening.len();
    let mut line_start = yaml_start;
    while line_start <= text.len() {
        let line_end = text[line_start..]
            .find('\n')
            .map_or(text.len(), |offset| line_start + offset);
        if text[line_start..line_end].trim() == "---" {
            let yaml_end = line_start.saturating_sub(1).max(yaml_start);
            let body_start = (line_end + 1).min(text.len());
            return Some((&text[yaml_start..yaml_end], &text[body_start..]));
        }
        if line_end == text.len() {
            break;
        }
        line_start = line_end + 1;
    }
    None
}

/// Parses the YAML block into [`PromptFileMeta`].
fn parse_meta(yaml: &str, source: &str) -> Result<PromptFileMeta> {
    if yaml.trim().is_empty() {
        return Ok(PromptFileMeta::default());
    }
    let value: Value = serde_yaml_ng::from_str(yaml).map_err(|error| {
        Error::validation(format!(
            "malformed prompt front matter in {source}: {error}"
        ))
    })?;
    let Value::Object(map) = value else {
        return Err(Error::validation(format!(
            "prompt front matter in {source} is not a YAML mapping"
        )));
    };
    build_meta(map, source)
}

/// Extracts the structured fields from an already-parsed YAML mapping.
fn build_meta(mut map: Map<String, Value>, source: &str) -> Result<PromptFileMeta> {
    Ok(PromptFileMeta {
        name: take_string(&mut map, "name", source)?,
        language: take_string(&mut map, "language", source)?,
        priority: take_priority(&mut map, source)?,
        kind: take_string(&mut map, "kind", source)?,
        evolved: take_bool(&mut map, "evolved", source)?,
        baseline_sha256: take_string(&mut map, "baseline_sha256", source)?,
        updated_at: take_string(&mut map, "updated_at", source)?,
        extra: map.into_iter().collect(),
    })
}

/// Removes `key` as a string; `null` counts as absent.
fn take_string(map: &mut Map<String, Value>, key: &str, source: &str) -> Result<Option<String>> {
    match map.remove(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(value)) => Ok(Some(value)),
        Some(other) => Err(Error::validation(format!(
            "prompt front matter in {source}: `{key}` must be a string, got {other}"
        ))),
    }
}

/// Removes `key` as a 32-bit integer, accepting a decimal string too.
fn take_priority(map: &mut Map<String, Value>, source: &str) -> Result<Option<i32>> {
    match map.remove("priority") {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Number(number)) => number
            .as_i64()
            .and_then(|value| i32::try_from(value).ok())
            .map(Some)
            .ok_or_else(|| invalid_priority(source, &number.to_string())),
        Some(Value::String(text)) => text
            .trim()
            .parse::<i32>()
            .map(Some)
            .map_err(|_| invalid_priority(source, &text)),
        Some(other) => Err(invalid_priority(source, &other.to_string())),
    }
}

/// The typed error for a priority that is not a 32-bit integer.
fn invalid_priority(source: &str, value: &str) -> Error {
    Error::validation(format!(
        "invalid `priority` in prompt front matter in {source}: expected an integer, got {value:?}"
    ))
}

/// Removes `key` as a boolean, accepting `true`/`false` strings too.
fn take_bool(map: &mut Map<String, Value>, key: &str, source: &str) -> Result<Option<bool>> {
    match map.remove(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Bool(value)) => Ok(Some(value)),
        Some(Value::String(text)) => match text.trim().to_ascii_lowercase().as_str() {
            "true" => Ok(Some(true)),
            "false" => Ok(Some(false)),
            _ => Err(Error::validation(format!(
                "prompt front matter in {source}: `{key}` must be a boolean, got {text:?}"
            ))),
        },
        Some(other) => Err(Error::validation(format!(
            "prompt front matter in {source}: `{key}` must be a boolean, got {other}"
        ))),
    }
}

/// Normalizes CRLF and lone CR line endings to `\n`.
fn normalize_newlines(text: &str) -> String {
    if !text.contains('\r') {
        return text.to_owned();
    }
    text.replace("\r\n", "\n").replace('\r', "\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use openjiuwen_common::error::ErrorKind;

    #[test]
    fn front_matter_parses_every_structured_field() {
        let text = "---\n\
kind: prompt\n\
name: leader_policy\n\
language: en\n\
priority: 30\n\
baseline_sha256: deadbeef\n\
evolved: false\n\
updated_at: 2026-01-01T00:00:00Z\n\
future_key: 7\n\
---\n\
body text\n";
        let (meta, body) = read_frontmatter(text).expect("the document parses");
        assert_eq!(meta.kind.as_deref(), Some("prompt"));
        assert_eq!(meta.name.as_deref(), Some("leader_policy"));
        assert_eq!(meta.language.as_deref(), Some("en"));
        assert_eq!(meta.priority, Some(30));
        assert_eq!(meta.baseline_sha256.as_deref(), Some("deadbeef"));
        assert_eq!(meta.evolved, Some(false));
        assert_eq!(meta.updated_at.as_deref(), Some("2026-01-01T00:00:00Z"));
        assert_eq!(meta.extra["future_key"], Value::from(7));
        assert_eq!(body, "body text\n");
    }

    #[test]
    fn documents_without_front_matter_are_hand_written_bodies() {
        let text = "just a prompt\nwith two lines\n";
        let (meta, body) = read_frontmatter(text).expect("no front matter is fine");
        assert_eq!(meta, PromptFileMeta::default());
        assert_eq!(body, text);

        let unclosed = "---\nname: a\nbody without a closing fence\n";
        let (meta, body) = read_frontmatter(unclosed).expect("an unclosed fence is a body");
        assert_eq!(meta, PromptFileMeta::default());
        assert_eq!(body, unclosed);
    }

    #[test]
    fn empty_front_matter_yields_empty_metadata() {
        let (meta, body) = read_frontmatter("---\n---\n\nbody\n").expect("the empty block parses");
        assert_eq!(meta, PromptFileMeta::default());
        assert_eq!(body, "\nbody\n");
    }

    #[test]
    fn the_body_is_byte_faithful_after_the_closing_fence() {
        let body = "line one\n\nline three\n\n";
        let text = format!("---\nname: a\n---\n{body}");
        let (_, parsed) = read_frontmatter(&text).expect("the document parses");
        assert_eq!(parsed, body);
        assert_eq!(body_sha256(&parsed), body_sha256(body));
    }

    #[test]
    fn crlf_and_leading_bom_are_handled() {
        let text = "\u{feff}---\r\nname: a\r\n---\r\nbody\r\n";
        // The BOM means the document no longer opens with `---`, so it is a body.
        let (meta, body) = read_frontmatter(text).expect("a BOM does not fail the read");
        assert_eq!(meta, PromptFileMeta::default());
        assert_eq!(body, "\u{feff}---\nname: a\n---\nbody\n");

        let (meta, body) =
            read_frontmatter("---\r\nname: a\r\n---\r\nline\r\n").expect("CRLF parses");
        assert_eq!(meta.name.as_deref(), Some("a"));
        assert_eq!(body, "line\n");
    }

    #[test]
    fn malformed_yaml_and_non_mapping_roots_are_validation_errors() {
        let error = read_frontmatter("---\nname: [unclosed\n---\nbody\n")
            .expect_err("malformed YAML is rejected");
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("malformed"), "{error}");

        let error =
            read_frontmatter("---\n- a\n- b\n---\nbody\n").expect_err("a list root is rejected");
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("mapping"), "{error}");
    }

    #[test]
    fn an_invalid_priority_is_a_typed_error() {
        let error =
            read_frontmatter("---\npriority: high\n---\n").expect_err("text is not a priority");
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("priority"), "{error}");

        let error = read_frontmatter("---\npriority: 999999999999\n---\n")
            .expect_err("an out-of-range priority is rejected");
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("priority"), "{error}");

        let (meta, _) = read_frontmatter("---\npriority: \"20\"\n---\n").expect("a numeric string");
        assert_eq!(meta.priority, Some(20));
    }

    #[test]
    fn known_fields_with_the_wrong_type_are_validation_errors() {
        let error =
            read_frontmatter("---\nname: 3\n---\n").expect_err("a numeric name is rejected");
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("`name`"), "{error}");

        let error =
            read_frontmatter("---\nevolved: sometimes\n---\n").expect_err("a bad flag is rejected");
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("`evolved`"), "{error}");
    }

    #[test]
    fn evolution_is_decided_by_the_body_hash() {
        let body = "the prompt body\n";
        let matching = format!("---\nbaseline_sha256: {}\n---\n{body}", body_sha256(body));
        let (meta, parsed) = read_frontmatter(&matching).expect("the seeded file parses");
        assert!(!is_evolved(&meta, &parsed));

        let diverged = "---\nbaseline_sha256: deadbeef\n---\nthe prompt body\n";
        let (meta, parsed) = read_frontmatter(diverged).expect("the file parses");
        assert!(is_evolved(&meta, &parsed));

        let hand_written = "---\nkind: prompt\n---\nthe prompt body\n";
        let (meta, parsed) = read_frontmatter(hand_written).expect("the file parses");
        assert!(is_evolved(&meta, &parsed));
    }
}
