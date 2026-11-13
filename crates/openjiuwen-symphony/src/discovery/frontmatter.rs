//! `SKILL.md` front-matter parsing.
//!
//! Ports the front-matter half of Python `openjiuwen.symphony.shared.fingerprint.scanner`
//! (`split_front_matter` plus the field readers). A skill document is an optional YAML
//! mapping delimited by a first line holding exactly `---` and a closing line holding
//! `---` or `...`; everything after the closing line is the markdown body.
//!
//! Parsing is total — no input panics — and reports its outcome as one of three
//! statuses:
//!
//! - [`FrontMatterStatus::Ok`] — the delimiters were found and closed. The YAML may
//!   still be unusable (malformed, non-mapping, null); in that case `metadata` is
//!   `None` and `error` explains why, which the scanner turns into a warning.
//! - [`FrontMatterStatus::Missing`] — the document does not open with `---`. The whole
//!   document is the body and `metadata` is `None` (Python: a warning that still yields
//!   a descriptor).
//! - [`FrontMatterStatus::Unterminated`] — the document opens with `---` but never
//!   closes the mapping. Python treats this as fatal for that skill; the scanner emits
//!   an error diagnostic and produces no descriptor for it.

use serde_json::{Map, Value};

use crate::discovery::models::dedupe_ignore_case;
use crate::model::{CapabilityIo, DEFAULT_IO_TYPE};

/// The front-matter delimiter line.
pub const DELIMITER: &str = "---";
/// The alternate closing delimiter line.
pub const ALTERNATE_DELIMITER: &str = "...";

/// The outcome of locating front matter in a markdown document.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum FrontMatterStatus {
    /// Both delimiters were found; `metadata` holds the parsed mapping (or is empty).
    Ok,
    /// The document does not start with [`DELIMITER`]; the whole document is the body.
    #[default]
    Missing,
    /// The document starts with [`DELIMITER`] but has no closing delimiter.
    Unterminated,
}

impl FrontMatterStatus {
    /// Whether the front-matter delimiters were found and closed.
    #[must_use]
    pub fn is_ok(self) -> bool {
        matches!(self, Self::Ok)
    }

    /// A stable `snake_case` label for diagnostics.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ok => "ok",
            Self::Missing => "missing",
            Self::Unterminated => "unterminated",
        }
    }
}

/// The parsed front matter of one markdown document.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct FrontMatterOutcome {
    /// Whether the delimiters were found, missing, or unterminated.
    pub status: FrontMatterStatus,
    /// The YAML mapping as JSON, when the front matter parsed into one.
    pub metadata: Option<Value>,
    /// The markdown body (everything after the closing delimiter); empty for
    /// [`FrontMatterStatus::Unterminated`] and the whole document when missing.
    pub body: String,
    /// Why the mapping is unusable, when it is; the scanner reports this as a warning.
    pub error: Option<String>,
}

impl FrontMatterOutcome {
    /// Builds an outcome for a document without front matter.
    #[must_use]
    pub fn missing(body: impl Into<String>) -> Self {
        Self {
            status: FrontMatterStatus::Missing,
            metadata: None,
            body: body.into(),
            error: None,
        }
    }

    /// Whether the delimiters were found and closed (malformed YAML still counts).
    #[must_use]
    pub fn fences_ok(&self) -> bool {
        self.status.is_ok()
    }

    /// The metadata mapping, when the front matter parsed into one.
    #[must_use]
    pub fn mapping(&self) -> Option<&Map<String, Value>> {
        self.metadata.as_ref()?.as_object()
    }

    /// Reads a metadata field, whatever its JSON type.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.mapping()?.get(key)
    }

    /// Reads a metadata field as a non-blank, trimmed string.
    #[must_use]
    pub fn get_str(&self, key: &str) -> Option<&str> {
        let value = self.get(key)?.as_str()?.trim();
        (!value.is_empty()).then_some(value)
    }

    /// Reads the first non-blank string among `keys`.
    #[must_use]
    pub fn first_str(&self, keys: &[&str]) -> Option<&str> {
        keys.iter().find_map(|key| self.get_str(key))
    }

    /// Reads a metadata field as a boolean, also accepting `"true"`/`"false"`
    /// (and `"yes"`/`"no"`, `"1"`/`"0"`) strings.
    #[must_use]
    pub fn get_bool(&self, key: &str) -> Option<bool> {
        boolean_value(self.get(key)?)
    }
}

/// Interprets a JSON value as a boolean, tolerating the common YAML spellings.
#[must_use]
pub fn boolean_value(value: &Value) -> Option<bool> {
    match value {
        Value::Bool(flag) => Some(*flag),
        Value::String(text) => match text.trim().to_ascii_lowercase().as_str() {
            "true" | "yes" | "on" | "1" => Some(true),
            "false" | "no" | "off" | "0" => Some(false),
            _ => None,
        },
        _ => None,
    }
}

/// Parses the front matter and body of a markdown document.
///
/// The first line must be exactly [`DELIMITER`] (a trailing `\r` from CRLF input is
/// tolerated). The mapping ends at the first line holding [`DELIMITER`] or
/// [`ALTERNATE_DELIMITER`]. The YAML is parsed with `serde_yaml_ng` and converted to
/// JSON; a null or non-mapping document, or one whose mapping is not JSON-compatible,
/// yields `metadata = None` with an `error` note instead of a failure.
///
/// ```
/// use openjiuwen_symphony::discovery::frontmatter::parse_front_matter;
///
/// let outcome = parse_front_matter("---\nname: pdf-export\ntags: [pdf]\n---\nBody");
/// assert!(outcome.fences_ok());
/// assert_eq!(outcome.get_str("name"), Some("pdf-export"));
/// assert_eq!(outcome.body, "Body");
/// ```
#[must_use]
pub fn parse_front_matter(raw: &str) -> FrontMatterOutcome {
    let lines: Vec<&str> = raw
        .split('\n')
        .map(|line| line.strip_suffix('\r').unwrap_or(line))
        .collect();

    let Some(first) = lines.first() else {
        return FrontMatterOutcome::missing(String::new());
    };
    if *first != DELIMITER {
        return FrontMatterOutcome::missing(raw.to_string());
    }

    let closing = lines.iter().enumerate().skip(1).find_map(|(index, line)| {
        (*line == DELIMITER || *line == ALTERNATE_DELIMITER).then_some(index)
    });
    let Some(closing) = closing else {
        return FrontMatterOutcome {
            status: FrontMatterStatus::Unterminated,
            metadata: None,
            body: String::new(),
            error: Some(format!(
                "unterminated front matter: missing a closing `{DELIMITER}` line"
            )),
        };
    };

    let yaml = lines[1..closing].join("\n");
    let body = lines[closing + 1..].join("\n");
    let (metadata, error) = parse_yaml_mapping(&yaml);
    FrontMatterOutcome {
        status: FrontMatterStatus::Ok,
        metadata,
        body,
        error,
    }
}

/// Parses the YAML mapping text into JSON, tolerating null and non-mapping documents.
///
/// Returns the mapping (or `None`) plus an explanatory error when the mapping could not
/// be used.
fn parse_yaml_mapping(yaml: &str) -> (Option<Value>, Option<String>) {
    if yaml.trim().is_empty() {
        return (None, None);
    }
    let parsed: serde_yaml_ng::Value = match serde_yaml_ng::from_str(yaml) {
        Ok(parsed) => parsed,
        Err(error) => {
            return (
                None,
                Some(format!("invalid front matter YAML: {error}").replace('\n', " ")),
            )
        }
    };
    match parsed {
        serde_yaml_ng::Value::Null => (None, None),
        serde_yaml_ng::Value::Mapping(mapping) => match serde_json::to_value(mapping) {
            Ok(Value::Object(object)) => (Some(Value::Object(object)), None),
            Ok(_) => (
                None,
                Some("front matter is not a YAML mapping of string keys".to_string()),
            ),
            Err(error) => (
                None,
                Some(format!("front matter is not JSON-compatible: {error}")),
            ),
        },
        _ => (None, Some("front matter is not a YAML mapping".to_string())),
    }
}

/// Normalizes a `capability_type` value against the Python scanner's aliases.
///
/// `skills` collapses to `skill`; `subagent`, `sub-agent`, and `sub_agent` collapse to
/// `agent`; any other value is trimmed and lowercased (with `-` folded to `_`), and a
/// blank value becomes `skill`.
///
/// ```
/// use openjiuwen_symphony::discovery::frontmatter::normalize_capability_type;
///
/// assert_eq!(normalize_capability_type("skills"), "skill");
/// assert_eq!(normalize_capability_type("sub-agent"), "agent");
/// assert_eq!(normalize_capability_type(""), "skill");
/// ```
#[must_use]
pub fn normalize_capability_type(raw: &str) -> String {
    let normalized = raw.trim().to_ascii_lowercase().replace('-', "_");
    match normalized.as_str() {
        "" | "skill" | "skills" => "skill".to_string(),
        "subagent" | "sub_agent" => "agent".to_string(),
        other => other.to_string(),
    }
}

/// Parses a `tags`/`aliases` field: a list of strings or one delimited string.
///
/// A string is split on `,` and `;`; list items are used as-is. Entries are trimmed,
/// blank entries are dropped, and duplicates collapse case-insensitively (the first
/// spelling wins), matching Python's tag normalization.
#[must_use]
pub fn tag_values(value: &Value) -> Vec<String> {
    let mut values: Vec<String> = Vec::new();
    match value {
        Value::String(text) => {
            values.extend(
                text.split([',', ';'])
                    .filter(|entry| !entry.trim().is_empty())
                    .map(ToString::to_string),
            );
        }
        Value::Array(items) => {
            for item in items {
                match item {
                    Value::String(text) => values.push(text.clone()),
                    Value::Number(number) => values.push(number.to_string()),
                    Value::Bool(flag) => values.push(flag.to_string()),
                    _ => {}
                }
            }
        }
        Value::Number(number) => values.push(number.to_string()),
        _ => {}
    }
    dedupe_ignore_case(values)
}

/// Parses an `inputs`/`outputs` field into typed [`CapabilityIo`] values.
///
/// A list of objects (or a single object) with `name`, `type`, `description`,
/// `required`, and `default` is understood; a bare string becomes a named IO of type
/// [`DEFAULT_IO_TYPE`]. Entries without a usable name are dropped.
#[must_use]
pub fn io_values(value: &Value) -> Vec<CapabilityIo> {
    match value {
        Value::Array(items) => items.iter().filter_map(io_from_value).collect(),
        Value::Object(_) => io_from_value(value).into_iter().collect(),
        _ => Vec::new(),
    }
}

/// Converts one `inputs`/`outputs` entry into a [`CapabilityIo`].
fn io_from_value(value: &Value) -> Option<CapabilityIo> {
    match value {
        Value::String(name) => {
            let name = name.trim();
            (!name.is_empty()).then(|| CapabilityIo::new(name, DEFAULT_IO_TYPE))
        }
        Value::Object(mapping) => {
            let name = mapping
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .trim();
            if name.is_empty() {
                return None;
            }
            let io_type = mapping
                .get("type")
                .or_else(|| mapping.get("io_type"))
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|io_type| !io_type.is_empty())
                .unwrap_or(DEFAULT_IO_TYPE);
            let description = mapping
                .get("description")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .trim();
            let mut io = CapabilityIo::new(name, io_type).with_description(description);
            if let Some(required) = mapping.get("required").and_then(boolean_value) {
                io = io.with_required(required);
            }
            if let Some(default) = mapping.get("default") {
                if !default.is_null() {
                    io = io.with_default(default.clone());
                }
            }
            Some(io)
        }
        _ => None,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_mapping_and_body() {
        let outcome = parse_front_matter("---\nname: alpha\ntags: [a, b]\n---\n\nBody text\n");
        assert_eq!(outcome.status, FrontMatterStatus::Ok);
        assert_eq!(outcome.get_str("name"), Some("alpha"));
        assert_eq!(outcome.body, "\nBody text\n");
        assert!(outcome.error.is_none());
    }

    #[test]
    fn accepts_both_closing_delimiters_and_crlf() {
        let dots = parse_front_matter("---\r\nname: alpha\r\n...\r\nBody");
        assert!(dots.fences_ok());
        assert_eq!(dots.get_str("name"), Some("alpha"));
        assert_eq!(dots.body, "Body");
    }

    #[test]
    fn missing_front_matter_keeps_the_whole_document() {
        let outcome = parse_front_matter("# Title\n\nText");
        assert_eq!(outcome.status, FrontMatterStatus::Missing);
        assert_eq!(outcome.body, "# Title\n\nText");
        assert!(outcome.metadata.is_none());
    }

    #[test]
    fn unterminated_front_matter_is_reported() {
        let outcome = parse_front_matter("---\nname: alpha\nBody");
        assert_eq!(outcome.status, FrontMatterStatus::Unterminated);
        assert!(outcome.error.is_some());
        assert_eq!(outcome.body, "");
        assert!(outcome.metadata.is_none());
    }

    #[test]
    fn malformed_yaml_is_tolerated_with_an_error_note() {
        let outcome = parse_front_matter("---\nname: [unclosed\n---\nBody");
        assert!(outcome.fences_ok());
        assert!(outcome.metadata.is_none());
        assert!(outcome.error.is_some());
        assert_eq!(outcome.body, "Body");
    }

    #[test]
    fn non_mapping_and_null_documents_are_tolerated() {
        let scalar = parse_front_matter("---\n- a\n- b\n---\nBody");
        assert!(scalar.fences_ok());
        assert!(scalar.metadata.is_none());
        assert!(scalar.error.is_some());

        let empty = parse_front_matter("---\n---\nBody");
        assert!(empty.fences_ok());
        assert!(empty.metadata.is_none());
        assert!(empty.error.is_none());

        let null = parse_front_matter("---\nnull\n---\nBody");
        assert!(null.metadata.is_none());
        assert!(null.error.is_none());
    }

    #[test]
    fn boolean_and_numeric_field_readers_are_lenient() {
        let outcome = parse_front_matter("---\nflag: 'false'\nnum: 3\n---\n");
        assert_eq!(outcome.get_bool("flag"), Some(false));
        assert_eq!(outcome.get_str("flag"), Some("false"));
        assert_eq!(outcome.get_str("num"), None);
        assert_eq!(outcome.get_bool("num"), None);
    }

    #[test]
    fn tags_and_io_parsers_cover_both_shapes() {
        let outcome = parse_front_matter(
            "---\ntags: 'Pdf, pdf ;Docs'\naliases:\n  - Alpha\n  - alpha\n---\n",
        );
        assert_eq!(tag_values(outcome.get("tags").unwrap()), ["Pdf", "Docs"]);
        assert_eq!(tag_values(outcome.get("aliases").unwrap()), ["Alpha"]);

        let inputs = serde_json::json!([
            "raw",
            {"name": "path", "type": "string", "description": "input file", "required": true, "default": "a.txt"},
            {"description": "nameless"}
        ]);
        let parsed = io_values(&inputs);
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].name, "raw");
        assert_eq!(parsed[0].io_type(), DEFAULT_IO_TYPE);
        assert_eq!(parsed[1].name, "path");
        assert_eq!(parsed[1].io_type(), "string");
        assert_eq!(parsed[1].required, Some(true));
        assert_eq!(
            parsed[1].default.as_ref().and_then(Value::as_str),
            Some("a.txt")
        );
    }

    #[test]
    fn capability_type_aliases_are_normalized() {
        assert_eq!(normalize_capability_type("skills"), "skill");
        assert_eq!(normalize_capability_type("SKILL"), "skill");
        assert_eq!(normalize_capability_type("subagent"), "agent");
        assert_eq!(normalize_capability_type("sub-agent"), "agent");
        assert_eq!(normalize_capability_type("sub_agent"), "agent");
        assert_eq!(normalize_capability_type("workflow"), "workflow");
    }
}
