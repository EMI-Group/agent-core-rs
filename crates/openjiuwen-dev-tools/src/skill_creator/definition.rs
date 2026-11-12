//! Skill definitions, YAML frontmatter, validation, and markdown conversion.
//!
//! A skill is a portable directory (`<name>/SKILL.md`) whose document opens with
//! a YAML frontmatter block and continues with a markdown instruction body. This
//! module owns that document format as pure data plus deterministic transforms:
//!
//! - [`SkillDefinition`] — the full skill: identity, instructions, metadata, resources.
//! - [`SkillResource`] — one text file shipped beside the document.
//! - [`SkillFrontmatter`] — the machine-readable header.
//! - [`parse_markdown`] / [`render_markdown`] — the document codec.
//!
//! The name rule matches the harness (`_name` in the Python `harness_providers`,
//! `^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$`), so a name is always usable as a single
//! directory component.
//!
//! Violations are [`Error::validation`](openjiuwen_common::error::Error::validation)
//! failures: Python's "warn and degrade" behavior is replaced by strict checks.

use std::collections::{BTreeMap, BTreeSet};

use openjiuwen_common::error::{Error, Result};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// The portable skill-name pattern, shared with the Python harness.
///
/// The first character is an ASCII letter or digit; up to 127 further ASCII
/// letters, digits, dots, underscores, or hyphens may follow.
pub const NAME_PATTERN: &str = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$";

/// The longest accepted skill name, in bytes (the pattern's `1 + 127`).
pub const MAX_NAME_LENGTH: usize = 128;

/// The frontmatter delimiter line.
const DELIMITER: &str = "---";

/// One text file shipped inside a skill bundle.
///
/// Binary assets are out of scope for the Rust port: content is text, and a
/// non-UTF-8 file found on disk is skipped when reading a bundle.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct SkillResource {
    /// Relative POSIX path inside the skill directory, e.g. `references/guide.md`.
    pub path: String,
    /// Text content written verbatim to [`path`](Self::path).
    pub content: String,
    /// Optional media type, informational only; it is never written to disk.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub media_type: Option<String>,
}

impl SkillResource {
    /// Creates a text resource at `path` with `content`.
    #[must_use]
    pub fn new(path: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            content: content.into(),
            media_type: None,
        }
    }

    /// Sets the informational media type.
    #[must_use]
    pub fn with_media_type(mut self, media_type: impl Into<String>) -> Self {
        self.media_type = Some(media_type.into());
        self
    }

    /// Checks that the resource path is a safe relative POSIX path.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`](openjiuwen_common::error::Error::validation)
    /// when the path is empty, absolute, contains a backslash or a NUL byte, has
    /// an empty, `.` or `..` component, or names the bundle document itself.
    pub fn validate(&self) -> Result<()> {
        validate_resource_path(&self.path)?;
        if self
            .media_type
            .as_deref()
            .is_some_and(|media_type| media_type.trim().is_empty())
        {
            return Err(Error::validation(format!(
                "skill resource {:?}: media type must not be blank",
                self.path
            )));
        }
        Ok(())
    }
}

/// The machine-readable header of a `SKILL.md` document.
///
/// Serialized as a YAML mapping: `name`, `description`, an optional `version`,
/// then every [`extra`](Self::extra) entry flattened at the top level.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct SkillFrontmatter {
    /// Portable skill name; also the bundle directory name.
    pub name: String,
    /// One-line description of what the skill does.
    pub description: String,
    /// Optional version string; omitted from YAML when `None`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    /// `metadata` keys flattened into the frontmatter mapping.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// A complete skill: identity, instructions, metadata, and text resources.
///
/// Python parity: the `SkillDefinition`/`SkillMetadata` pair of
/// `openjiuwen.dev_tools.skill_creator`. Validation is explicit
/// ([`validate`](Self::validate)); [`to_markdown`](Self::to_markdown) and
/// [`crate::skill_creator::SkillBundle::write`] run it before producing output.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct SkillDefinition {
    /// Portable skill name (`NAME_PATTERN`).
    pub name: String,
    /// One-line description of what the skill does.
    pub description: String,
    /// Markdown instruction body written after the frontmatter.
    pub instructions: String,
    /// Optional version string.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    /// Free-form metadata, flattened into the frontmatter.
    pub metadata: BTreeMap<String, Value>,
    /// Text resources shipped beside `SKILL.md`; order is preserved on write.
    pub resources: Vec<SkillResource>,
}

impl SkillDefinition {
    /// Creates a skill with a name, description, and instruction body.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        instructions: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            instructions: instructions.into(),
            version: None,
            metadata: BTreeMap::new(),
            resources: Vec::new(),
        }
    }

    /// Validates the name, the non-blank fields, and every resource path.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`](openjiuwen_common::error::Error::validation)
    /// for an unusable name, a blank description or instruction body, an invalid
    /// resource path, or two resources sharing a path.
    pub fn validate(&self) -> Result<()> {
        validate_name(&self.name)?;
        if self.description.trim().is_empty() {
            return Err(Error::validation("skill description must not be blank"));
        }
        if self.instructions.trim().is_empty() {
            return Err(Error::validation("skill instructions must not be blank"));
        }
        let mut seen = BTreeSet::new();
        for resource in &self.resources {
            resource.validate()?;
            if !seen.insert(resource.path.as_str()) {
                return Err(Error::validation(format!(
                    "duplicate skill resource path {:?}",
                    resource.path
                )));
            }
        }
        Ok(())
    }

    /// The directory name of this skill's bundle, which is always
    /// [`name`](Self::name).
    #[must_use]
    pub fn skill_dir_name(&self) -> &str {
        &self.name
    }

    /// Appends a resource.
    #[must_use]
    pub fn with_resource(mut self, resource: SkillResource) -> Self {
        self.resources.push(resource);
        self
    }

    /// Inserts one metadata entry.
    #[must_use]
    pub fn with_metadata(mut self, key: impl Into<String>, value: Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Sets the version string.
    #[must_use]
    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.version = Some(version.into());
        self
    }

    /// Projects this skill into its frontmatter header.
    #[must_use]
    pub fn frontmatter(&self) -> SkillFrontmatter {
        SkillFrontmatter {
            name: self.name.clone(),
            description: self.description.clone(),
            version: self.version.clone(),
            extra: self.metadata.clone(),
        }
    }

    /// Renders the `SKILL.md` document for this skill.
    ///
    /// # Errors
    ///
    /// Returns whatever [`validate`](Self::validate) or [`render_markdown`]
    /// rejects, plus a metadata key that collides with a reserved field.
    pub fn to_markdown(&self) -> Result<String> {
        self.validate()?;
        render_markdown(&self.frontmatter(), &self.instructions)
    }

    /// Parses a `SKILL.md` document into a skill with no resources.
    ///
    /// # Errors
    ///
    /// Returns whatever [`parse_markdown`] or [`validate`](Self::validate) rejects.
    pub fn from_markdown(text: &str) -> Result<Self> {
        let (frontmatter, body) = parse_markdown(text)?;
        let definition = Self {
            name: frontmatter.name,
            description: frontmatter.description,
            instructions: body,
            version: frontmatter.version,
            metadata: frontmatter.extra,
            resources: Vec::new(),
        };
        definition.validate()?;
        Ok(definition)
    }
}

/// Parses a `SKILL.md` document into its frontmatter and instruction body.
///
/// The document must open with a line holding only `---`, then a YAML mapping,
/// then a closing `---` line; everything after it (trimmed) is the body.
///
/// # Errors
///
/// Returns [`Error::validation`](openjiuwen_common::error::Error::validation) when
/// the document has no frontmatter, the frontmatter is unterminated or not a YAML
/// mapping, `name` or `description` is missing or not a non-blank string, or the
/// name does not match [`NAME_PATTERN`].
pub fn parse_markdown(text: &str) -> Result<(SkillFrontmatter, String)> {
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    let lines: Vec<&str> = text
        .split('\n')
        .map(|line| line.trim_end_matches('\r'))
        .collect();

    let opens_document = lines.first().is_some_and(|line| line.trim() == DELIMITER);
    if !opens_document {
        return Err(Error::validation(format!(
            "skill markdown must start with a '{DELIMITER}' frontmatter delimiter"
        )));
    }
    let close = lines
        .iter()
        .skip(1)
        .position(|line| line.trim() == DELIMITER)
        .map(|offset| offset + 1);
    let Some(close) = close else {
        return Err(Error::validation(format!(
            "unterminated skill frontmatter: missing closing '{DELIMITER}'"
        )));
    };

    let yaml = lines[1..close].join("\n");
    let body = lines[close + 1..].join("\n").trim().to_string();
    if yaml.trim().is_empty() {
        return Err(Error::validation(
            "skill frontmatter must be a YAML mapping",
        ));
    }

    let parsed: serde_yaml_ng::Value = serde_yaml_ng::from_str(&yaml)
        .map_err(|error| Error::validation(format!("invalid skill frontmatter YAML: {error}")))?;
    let serde_yaml_ng::Value::Mapping(mapping) = parsed else {
        return Err(Error::validation(
            "skill frontmatter must be a YAML mapping",
        ));
    };

    let name = required_string(&mapping, "name")?;
    validate_name(&name)?;
    let description = required_string(&mapping, "description")?;
    let version = optional_string(&mapping, "version")?;

    let mut extra = BTreeMap::new();
    for (key, value) in &mapping {
        let Some(key) = key.as_str() else {
            return Err(Error::validation("skill frontmatter keys must be strings"));
        };
        if is_reserved_key(key) {
            continue;
        }
        let value = serde_json::to_value(value).map_err(|error| {
            Error::validation(format!(
                "unsupported skill frontmatter value for '{key}': {error}"
            ))
        })?;
        extra.insert(key.to_owned(), value);
    }

    Ok((
        SkillFrontmatter {
            name,
            description,
            version,
            extra,
        },
        body,
    ))
}

/// Renders a `SKILL.md` document from a frontmatter header and an instruction body.
///
/// The layout is `---\n<yaml mapping>---\n\n<body>`: the YAML serializer's
/// trailing newline supplies the newline before the closing delimiter.
///
/// # Errors
///
/// Returns [`Error::validation`](openjiuwen_common::error::Error::validation) for
/// an invalid name, a blank description or body, a metadata key that collides
/// with a reserved field, or a YAML serialization failure.
pub fn render_markdown(frontmatter: &SkillFrontmatter, body: &str) -> Result<String> {
    validate_name(&frontmatter.name)?;
    if frontmatter.description.trim().is_empty() {
        return Err(Error::validation("skill description must not be blank"));
    }
    if body.trim().is_empty() {
        return Err(Error::validation("skill instructions must not be blank"));
    }
    for key in frontmatter.extra.keys() {
        if is_reserved_key(key) {
            return Err(Error::validation(format!(
                "skill metadata key '{key}' collides with a reserved frontmatter field"
            )));
        }
    }

    let yaml = serde_yaml_ng::to_string(frontmatter).map_err(|error| {
        Error::validation(format!("failed to serialize skill frontmatter: {error}"))
    })?;
    let yaml = yaml.trim_end();
    Ok(format!("{DELIMITER}\n{yaml}\n{DELIMITER}\n\n{body}"))
}

/// Checks a skill name against [`NAME_PATTERN`].
fn validate_name(name: &str) -> Result<()> {
    let bytes = name.as_bytes();
    let valid = !bytes.is_empty()
        && bytes.len() <= MAX_NAME_LENGTH
        && bytes[0].is_ascii_alphanumeric()
        && bytes[1..]
            .iter()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'));
    if valid {
        Ok(())
    } else {
        Err(Error::validation(format!(
            "invalid skill name {name:?}: expected {NAME_PATTERN}"
        )))
    }
}

/// Checks a resource path: relative, `/`-separated, non-escaping.
fn validate_resource_path(path: &str) -> Result<()> {
    if path.trim().is_empty() {
        return Err(Error::validation("skill resource path must not be empty"));
    }
    if path.starts_with('/') {
        return Err(Error::validation(format!(
            "skill resource path {path:?} must be relative"
        )));
    }
    if path.contains('\\') {
        return Err(Error::validation(format!(
            "skill resource path {path:?} must use '/' separators"
        )));
    }
    if path.contains('\0') {
        return Err(Error::validation(format!(
            "skill resource path {path:?} must not contain NUL"
        )));
    }
    if path.eq_ignore_ascii_case("SKILL.md") {
        return Err(Error::validation(
            "skill resource path must not name the bundle document SKILL.md",
        ));
    }
    for component in path.split('/') {
        if component.is_empty() {
            return Err(Error::validation(format!(
                "skill resource path {path:?} must not contain empty components"
            )));
        }
        if component == "." || component == ".." {
            return Err(Error::validation(format!(
                "skill resource path {path:?} must not contain '.' or '..' components"
            )));
        }
    }
    Ok(())
}

/// Whether `key` names a structured frontmatter field rather than metadata.
fn is_reserved_key(key: &str) -> bool {
    matches!(key, "name" | "description" | "version")
}

/// Reads a required, non-blank string field from the parsed mapping.
fn required_string(mapping: &serde_yaml_ng::Mapping, key: &str) -> Result<String> {
    match mapping.get(key) {
        Some(serde_yaml_ng::Value::String(value)) if !value.trim().is_empty() => Ok(value.clone()),
        Some(serde_yaml_ng::Value::String(_)) => Err(Error::validation(format!(
            "skill frontmatter field '{key}' must not be blank"
        ))),
        Some(_) => Err(Error::validation(format!(
            "skill frontmatter field '{key}' must be a string"
        ))),
        None => Err(Error::validation(format!(
            "skill frontmatter is missing required field '{key}'"
        ))),
    }
}

/// Reads an optional version field.
///
/// A missing or `null` value is `None`; a YAML scalar is stringified, so both
/// `version: 2.0` and `version: "2.0"` yield `"2.0"`.
fn optional_string(mapping: &serde_yaml_ng::Mapping, key: &str) -> Result<Option<String>> {
    match mapping.get(key) {
        None | Some(serde_yaml_ng::Value::Null) => Ok(None),
        Some(serde_yaml_ng::Value::String(value)) => Ok(Some(value.clone())),
        Some(serde_yaml_ng::Value::Number(value)) => Ok(Some(value.to_string())),
        Some(_) => Err(Error::validation(format!(
            "skill frontmatter field '{key}' must be a string"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn definition() -> SkillDefinition {
        SkillDefinition::new(
            "demo-skill",
            "A demo skill",
            "# Instructions\n\nDo the thing.",
        )
    }

    #[test]
    fn valid_names_are_accepted() {
        for name in [
            "my-skill",
            "A1",
            "a",
            "a.b_c-d",
            &"a".repeat(MAX_NAME_LENGTH),
        ] {
            assert!(validate_name(name).is_ok(), "{name:?} should be valid");
        }
    }

    #[test]
    fn invalid_names_are_rejected() {
        let too_long = "a".repeat(MAX_NAME_LENGTH + 1);
        for name in ["", "-x", ".x", "a b", "技能", "a/b", &too_long] {
            assert!(validate_name(name).is_err(), "{name:?} should be invalid");
        }
    }

    #[test]
    fn definition_requires_identity_and_body() {
        let mut skill = definition();
        skill.description = "   ".to_owned();
        assert!(skill.validate().is_err());

        let mut skill = definition();
        skill.instructions = "\n\t ".to_owned();
        assert!(skill.validate().is_err());

        let mut skill = definition();
        skill.name = "bad name".to_owned();
        assert!(skill.validate().is_err());

        assert!(definition().validate().is_ok());
        assert_eq!(definition().skill_dir_name(), "demo-skill");
    }

    #[test]
    fn resource_paths_are_validated() {
        for path in ["references/guide.md", "a/b/c.txt", "notes.md"] {
            assert!(
                SkillResource::new(path, "text").validate().is_ok(),
                "{path:?} should be valid"
            );
        }
        for path in [
            "",
            "   ",
            "/absolute.md",
            "../escape.md",
            "a/../b.md",
            "./relative.md",
            "a//b.md",
            "a/",
            "windows\\path.md",
            "SKILL.md",
            "skill.md",
        ] {
            assert!(
                SkillResource::new(path, "text").validate().is_err(),
                "{path:?} should be invalid"
            );
        }
    }

    #[test]
    fn duplicate_resource_paths_are_rejected() {
        let skill = definition()
            .with_resource(SkillResource::new("references/guide.md", "one"))
            .with_resource(SkillResource::new("references/guide.md", "two"));
        let error = skill.validate().expect_err("duplicate path");
        assert!(error.to_string().contains("duplicate"), "{error}");
    }

    #[test]
    fn blank_media_type_is_rejected() {
        let resource = SkillResource::new("notes.md", "x").with_media_type("  ");
        assert!(resource.validate().is_err());
    }

    #[test]
    fn markdown_round_trip_keeps_everything() {
        let skill = definition()
            .with_version("1.2.3")
            .with_metadata("author", json!("alice"))
            .with_metadata("tags", json!(["docs", "demo"]));

        let markdown = skill.to_markdown().expect("renders");
        assert!(markdown.starts_with("---\nname: demo-skill\n"));
        assert!(markdown.contains("version: 1.2.3"));
        assert!(markdown.contains("author: alice"));
        assert!(markdown.ends_with("\n---\n\n# Instructions\n\nDo the thing."));

        let parsed = SkillDefinition::from_markdown(&markdown).expect("parses");
        assert_eq!(parsed, skill);
    }

    #[test]
    fn null_version_is_omitted_from_yaml() {
        let markdown = definition().to_markdown().expect("renders");
        assert!(!markdown.contains("version:"), "{markdown}");
    }

    #[test]
    fn multi_line_instructions_round_trip() {
        let body = "## Step 1\n\nDo this.\n\n## Step 2\n\nDo that.\n  - nested";
        let skill = SkillDefinition::new("multi", "Multi line", body);
        let markdown = skill.to_markdown().expect("renders");
        let parsed = SkillDefinition::from_markdown(&markdown).expect("parses");
        assert_eq!(parsed.instructions, body);
    }

    #[test]
    fn parse_preserves_unknown_frontmatter_fields() {
        let markdown =
            "---\nname: demo\ndescription: Demo\nversion: 2.0\ncustom:\n  nested: 1\n---\n\nBody";
        let (frontmatter, body) = parse_markdown(markdown).expect("parses");
        assert_eq!(frontmatter.name, "demo");
        assert_eq!(frontmatter.description, "Demo");
        assert_eq!(frontmatter.version.as_deref(), Some("2.0"));
        assert_eq!(frontmatter.extra.get("custom"), Some(&json!({"nested": 1})));
        assert_eq!(body, "Body");
    }

    #[test]
    fn numeric_version_is_stringified() {
        let (frontmatter, _) =
            parse_markdown("---\nname: demo\ndescription: Demo\nversion: 2.0\n---\nbody")
                .expect("parses");
        assert_eq!(frontmatter.version.as_deref(), Some("2.0"));
    }

    #[test]
    fn parse_rejects_structural_errors() {
        let cases = [
            "# no frontmatter\nbody",
            "---\nname: demo\ndescription: Demo\nbody without close",
            "---\n- a\n- b\n---\nbody",
            "---\nname: demo\n---\nbody",
            "---\ndescription: Demo\n---\nbody",
            "---\nname: demo\ndescription: Demo\nversion: [3]\n---\nbody",
            "---\nname: bad name\ndescription: Demo\n---\nbody",
            "---\nname: demo\ndescription: '   '\n---\nbody",
            "---\n---\nbody",
            "---\nname: demo\ndescription: Demo\n{unclosed\n---\nbody",
        ];
        for case in cases {
            let error = parse_markdown(case).expect_err("should be rejected");
            assert!(
                error.to_string().starts_with("validation:"),
                "{case}: {error}"
            );
        }
    }

    #[test]
    fn parse_accepts_crlf_and_bom() {
        let markdown = "\u{feff}---\r\nname: demo\r\ndescription: Demo\r\n---\r\n\r\nBody\r\n";
        let (frontmatter, body) = parse_markdown(markdown).expect("parses");
        assert_eq!(frontmatter.name, "demo");
        assert_eq!(body, "Body");
    }

    #[test]
    fn render_rejects_reserved_metadata_keys() {
        let frontmatter = SkillFrontmatter {
            name: "demo".to_owned(),
            description: "Demo".to_owned(),
            version: None,
            extra: BTreeMap::from([("name".to_owned(), json!("other"))]),
        };
        let error = render_markdown(&frontmatter, "body").expect_err("reserved key");
        assert!(error.to_string().contains("reserved"), "{error}");
    }

    #[test]
    fn render_rejects_blank_parts() {
        let mut frontmatter = SkillFrontmatter {
            name: "demo".to_owned(),
            description: "Demo".to_owned(),
            version: None,
            extra: BTreeMap::new(),
        };
        assert!(render_markdown(&frontmatter, " ").is_err());
        frontmatter.description = String::new();
        assert!(render_markdown(&frontmatter, "body").is_err());
    }

    #[test]
    fn definition_json_round_trip_tolerates_unknown_fields() {
        let json = json!({
            "name": "demo",
            "description": "Demo",
            "instructions": "body",
            "version": "1.0",
            "metadata": {"author": "alice"},
            "resources": [{"path": "notes.md", "content": "hi", "media_type": "text/markdown"}],
            "unknown_field": {"ignored": true}
        });
        let skill: SkillDefinition = serde_json::from_value(json).expect("deserializes");
        assert_eq!(skill.resources.len(), 1);
        assert_eq!(
            skill.resources[0].media_type.as_deref(),
            Some("text/markdown")
        );
        let round_tripped: SkillDefinition =
            serde_json::from_value(serde_json::to_value(&skill).expect("serializes"))
                .expect("deserializes");
        assert_eq!(round_tripped, skill);
    }

    #[test]
    fn definition_schema_covers_public_fields() {
        let schema = schemars::schema_for!(SkillDefinition);
        let text = serde_json::to_string(&schema).expect("schema serializes");
        for field in [
            "name",
            "description",
            "instructions",
            "metadata",
            "resources",
        ] {
            assert!(text.contains(field), "schema should mention {field}");
        }
    }

    #[test]
    fn empty_json_object_deserializes_with_defaults() {
        let skill: SkillDefinition = serde_json::from_value(json!({})).expect("deserializes");
        assert_eq!(skill, SkillDefinition::default());
        assert!(skill.validate().is_err());
    }
}
