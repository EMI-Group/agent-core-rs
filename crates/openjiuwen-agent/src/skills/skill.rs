//! One skill: front-matter metadata plus the Markdown body the model reads.
//!
//! Ports the `Skill` schema of
//! `openjiuwen/core/single_agent/skills/skill_manager.py`, extended with the
//! Markdown parsing Python leaves to pydantic/YAML: a `SKILL.md` document is a
//! YAML front matter block delimited by `---` lines followed by the body. A
//! document without front matter takes its name from its first `# ` heading.
use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

/// Front-matter keys with a structured counterpart in [`Skill`].
const RESERVED_FRONT_MATTER_KEYS: [&str; 4] = ["name", "description", "version", "keywords"];

/// A skill the agent can advertise to the model.
///
/// `name` and `description` are the only fields the catalog needs; `body` is
/// the instruction text the model reads through the `read_file` tool, `path` is
/// the file the skill was loaded from (set by [`crate::skills::SkillManager`]),
/// and `metadata` keeps every front-matter key without a structured field.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct Skill {
    /// Stable skill name, unique within a [`crate::skills::SkillManager`].
    pub name: String,
    /// One-line summary used by the catalog and the `matches` check.
    pub description: String,
    /// Optional skill version from the front matter.
    pub version: Option<String>,
    /// Keywords that make the skill match a query.
    pub keywords: Vec<String>,
    /// File the skill was loaded from, when it came from disk.
    pub path: Option<String>,
    /// Instruction body shown when the model reads the skill document.
    pub body: String,
    /// Front-matter keys without a structured field, kept as JSON values.
    pub metadata: BTreeMap<String, Value>,
}

/// The YAML front matter of a skill document.
///
/// Deserialization requires `name` and `description` (the name must also be
/// non-blank; the description may be empty); every other key is collected into
/// [`SkillFrontMatter::metadata`] so [`Skill::to_markdown`] can write it back.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct SkillFrontMatter {
    /// Skill name.
    pub name: String,
    /// One-line skill description.
    pub description: String,
    /// Optional skill version.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    /// Keywords that make the skill match a query.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub keywords: Vec<String>,
    /// Front-matter keys without a structured field.
    #[serde(flatten)]
    pub metadata: BTreeMap<String, Value>,
}

impl Skill {
    /// Builds a skill with `name` and `description` and an empty body.
    #[must_use]
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            ..Self::default()
        }
    }

    /// Replaces the instruction body.
    #[must_use]
    pub fn with_body(mut self, body: impl Into<String>) -> Self {
        self.body = body.into();
        self
    }

    /// Replaces the keywords.
    #[must_use]
    pub fn with_keywords<I, S>(mut self, keywords: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.keywords = keywords.into_iter().map(Into::into).collect();
        self
    }

    /// Inserts one metadata entry; a later insert with the same key replaces the
    /// previous value.
    #[must_use]
    pub fn with_metadata(mut self, key: impl Into<String>, value: Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Sets the file the skill was loaded from.
    #[must_use]
    pub fn with_path(mut self, path: impl Into<String>) -> Self {
        self.path = Some(path.into());
        self
    }

    /// Whether `query` refers to this skill.
    ///
    /// The check is case-insensitive: the whole trimmed query and then each of
    /// its alphanumeric terms (two characters or more) are matched as substrings
    /// of the name, the description, and each keyword. An empty query never
    /// matches.
    #[must_use]
    pub fn matches(&self, query: &str) -> bool {
        let query = query.trim().to_lowercase();
        if query.is_empty() {
            return false;
        }
        if self.mentions(&query) {
            return true;
        }
        query
            .split(|character: char| !character.is_alphanumeric())
            .filter(|term| term.chars().count() >= 2)
            .any(|term| self.mentions(term))
    }

    /// Renders the skill as the Markdown block the model sees: the `# name`
    /// heading, the description, and then the body, blank-line separated.
    ///
    /// Empty description and body parts are omitted.
    #[must_use]
    pub fn render(&self) -> String {
        let name = format!("# {}", self.name.trim());
        let description = self.description.trim();
        let body = self.body.trim();
        let mut parts = vec![name.as_str()];
        if !description.is_empty() {
            parts.push(description);
        }
        if !body.is_empty() {
            parts.push(body);
        }
        parts.join("\n\n")
    }

    /// Parses a skill document, using `"markdown document"` as the source label
    /// in error messages.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when a front matter block is unterminated,
    /// its YAML is malformed, its `name`/`description` field is missing, or its
    /// `name` is blank; and when the document has neither front matter nor a
    /// `# ` heading.
    pub fn from_markdown(markdown: &str) -> Result<Self> {
        Self::parse_markdown(markdown, "markdown document")
    }

    /// Serializes the skill back into a `SKILL.md` document.
    ///
    /// The layout is `---\n<yaml>\n---\n\n<body>`; `path` is not written, and a
    /// metadata key that collides with a structured front-matter field is
    /// skipped with a warning.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the front matter cannot be serialized
    /// to YAML.
    pub fn to_markdown(&self) -> Result<String> {
        let mut front = SkillFrontMatter {
            name: self.name.clone(),
            description: self.description.clone(),
            version: self.version.clone(),
            keywords: self.keywords.clone(),
            metadata: BTreeMap::new(),
        };
        for (key, value) in &self.metadata {
            if RESERVED_FRONT_MATTER_KEYS.contains(&key.as_str()) {
                tracing::warn!(
                    target: "openjiuwen_agent::skills",
                    name = %self.name,
                    key = %key,
                    "skill metadata key collides with a structured front matter field; skipped"
                );
                continue;
            }
            front.metadata.insert(key.clone(), value.clone());
        }
        let yaml = serde_yaml_ng::to_string(&front).map_err(|error| {
            Error::validation(format!(
                "failed to serialize skill front matter for {:?}: {error}",
                self.name
            ))
        })?;
        let mut rendered = String::from("---\n");
        rendered.push_str(yaml.trim_end());
        rendered.push_str("\n---\n");
        let body = self.body.trim();
        if !body.is_empty() {
            rendered.push('\n');
            rendered.push_str(body);
            rendered.push('\n');
        }
        Ok(rendered)
    }

    /// Parses a skill document, naming `source` in every error message.
    ///
    /// Carriage-return line endings and a leading byte-order mark are
    /// normalized, and name/description/body are trimmed.
    pub(crate) fn parse_markdown(markdown: &str, source: &str) -> Result<Self> {
        let text = markdown
            .trim_start_matches('\u{feff}')
            .replace("\r\n", "\n");
        let text = text.trim_start();
        if text.starts_with("---\n") {
            return Self::parse_front_matter(text, source);
        }
        Self::parse_heading(text, source)
    }

    /// Parses a document whose front matter block was detected.
    fn parse_front_matter(text: &str, source: &str) -> Result<Self> {
        let Some((yaml, body)) = split_front_matter(text) else {
            return Err(Error::validation(format!(
                "unterminated skill front matter in {source}: missing closing `---`"
            )));
        };
        let front: SkillFrontMatter = serde_yaml_ng::from_str(yaml).map_err(|error| {
            Error::validation(format!("invalid skill front matter in {source}: {error}"))
        })?;
        if front.name.trim().is_empty() {
            return Err(Error::validation(format!(
                "skill front matter in {source} has a blank `name` field"
            )));
        }
        Ok(Self {
            name: front.name.trim().to_owned(),
            description: front.description.trim().to_owned(),
            version: front.version,
            keywords: front.keywords,
            path: None,
            body: body.trim().to_owned(),
            metadata: front.metadata,
        })
    }

    /// Parses a document without front matter, naming it after its `# ` heading.
    fn parse_heading(text: &str, source: &str) -> Result<Self> {
        let mut offset = 0;
        for line in text.split_inclusive('\n') {
            if let Some(name) = line.trim_end().strip_prefix("# ") {
                let name = name.trim();
                if !name.is_empty() {
                    let body = text[offset + line.len()..].trim();
                    return Ok(Self {
                        name: name.to_owned(),
                        body: body.to_owned(),
                        ..Self::default()
                    });
                }
            }
            offset += line.len();
        }
        Err(Error::validation(format!(
            "skill document {source} has neither front matter nor a `# name` heading"
        )))
    }

    /// Whether the lowercase `needle` occurs in the name, description, or a
    /// keyword.
    fn mentions(&self, needle: &str) -> bool {
        self.name.to_lowercase().contains(needle)
            || self.description.to_lowercase().contains(needle)
            || self
                .keywords
                .iter()
                .any(|keyword| keyword.to_lowercase().contains(needle))
    }
}

/// Splits a document starting with `---\n` into its YAML block and its body.
///
/// Returns `None` when the closing `---` line is missing.
fn split_front_matter(text: &str) -> Option<(&str, &str)> {
    let rest = text.strip_prefix("---\n")?;
    let mut offset = 0;
    for line in rest.split_inclusive('\n') {
        if line.trim_end_matches('\n').trim_end() == "---" {
            return Some((&rest[..offset], &rest[offset + line.len()..]));
        }
        offset += line.len();
    }
    None
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]
mod tests {
    use super::*;
    use serde_json::json;

    fn deploy() -> Skill {
        Skill::new("deploy", "Deploy the service")
            .with_body("Run the deploy script.")
            .with_keywords(["release", "cd"])
            .with_path("skills/deploy/SKILL.md")
    }

    #[test]
    fn serde_round_trips_with_snake_case_fields() {
        let skill = deploy().with_metadata("owner", json!("platform"));
        let value = serde_json::to_value(&skill).unwrap();
        assert_eq!(value["name"], json!("deploy"));
        assert_eq!(value["description"], json!("Deploy the service"));
        assert_eq!(value["path"], json!("skills/deploy/SKILL.md"));
        assert_eq!(value["keywords"], json!(["release", "cd"]));
        assert_eq!(value["metadata"]["owner"], json!("platform"));
        let restored: Skill = serde_json::from_value(value).unwrap();
        assert_eq!(restored, skill);
        // Unknown keys are tolerated and defaulted fields are filled in.
        let sparse: Skill = serde_json::from_value(json!({"name": "x", "extra": 1})).unwrap();
        assert_eq!(sparse.name, "x");
        assert!(sparse.description.is_empty());
        assert!(sparse.body.is_empty());
        assert!(sparse.keywords.is_empty());
        assert!(sparse.metadata.is_empty());
    }

    #[test]
    fn front_matter_parses_every_field() {
        let markdown = "---\nname: deploy\ndescription: Deploy the service\nversion: 1.2.0\nkeywords:\n  - release\n  - cd\nowner: platform\n---\n\nRun the deploy script.\n";
        let skill = Skill::from_markdown(markdown).unwrap();
        assert_eq!(skill.name, "deploy");
        assert_eq!(skill.description, "Deploy the service");
        assert_eq!(skill.version.as_deref(), Some("1.2.0"));
        assert_eq!(skill.keywords, vec!["release", "cd"]);
        assert_eq!(skill.body, "Run the deploy script.");
        assert_eq!(skill.metadata["owner"], json!("platform"));
        assert!(skill.path.is_none());
    }

    #[test]
    fn front_matter_defaults_optional_fields() {
        let skill = Skill::from_markdown("---\nname: a\ndescription: b\n---\nbody").unwrap();
        assert!(skill.version.is_none());
        assert!(skill.keywords.is_empty());
        assert!(skill.metadata.is_empty());
        assert_eq!(skill.body, "body");
    }

    #[test]
    fn missing_name_is_a_validation_error() {
        let error = Skill::from_markdown("---\ndescription: b\n---\nbody").unwrap_err();
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Validation
        );
        assert!(error.to_string().contains("`name`"), "{error}");
    }

    #[test]
    fn missing_description_is_a_validation_error() {
        let error = Skill::from_markdown("---\nname: a\n---\nbody").unwrap_err();
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Validation
        );
        assert!(error.to_string().contains("`description`"), "{error}");
    }

    #[test]
    fn a_blank_name_is_a_validation_error() {
        let error =
            Skill::from_markdown("---\nname: \"  \"\ndescription: b\n---\nbody").unwrap_err();
        assert!(error.to_string().contains("blank `name`"), "{error}");
    }

    #[test]
    fn an_empty_description_is_allowed() {
        let skill = Skill::from_markdown("---\nname: a\ndescription: ''\n---\nbody").unwrap();
        assert_eq!(skill.name, "a");
        assert!(skill.description.is_empty());
        assert_eq!(skill.body, "body");
    }

    #[test]
    fn malformed_yaml_is_a_validation_error() {
        let error = Skill::from_markdown("---\nname: [unterminated\n---\nbody").unwrap_err();
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Validation
        );
        assert!(
            error.to_string().contains("markdown document"),
            "the message names its source: {error}"
        );
        assert!(error.to_string().contains("invalid skill front matter"));
    }

    #[test]
    fn unterminated_front_matter_is_a_validation_error() {
        let error = Skill::from_markdown("---\nname: a\ndescription: b\nbody").unwrap_err();
        assert!(error.to_string().contains("unterminated"), "{error}");
    }

    #[test]
    fn heading_names_a_document_without_front_matter() {
        let skill = Skill::from_markdown("# Deploy\n\nRun the script.\n").unwrap();
        assert_eq!(skill.name, "Deploy");
        assert!(skill.description.is_empty());
        assert_eq!(skill.body, "Run the script.");
        assert!(skill.path.is_none());
    }

    #[test]
    fn heading_without_body_is_allowed() {
        let skill = Skill::from_markdown("# Deploy").unwrap();
        assert_eq!(skill.name, "Deploy");
        assert!(skill.body.is_empty());
    }

    #[test]
    fn document_without_heading_or_front_matter_is_rejected() {
        let error = Skill::from_markdown("just prose\n").unwrap_err();
        assert!(
            error.to_string().contains("neither front matter"),
            "{error}"
        );
    }

    #[test]
    fn carriage_returns_and_byte_order_marks_are_normalized() {
        let markdown = "\u{feff}---\r\nname: a\r\ndescription: b\r\n---\r\n\r\nbody\r\n";
        let skill = Skill::from_markdown(markdown).unwrap();
        assert_eq!(skill.name, "a");
        assert_eq!(skill.description, "b");
        assert_eq!(skill.body, "body");
    }

    #[test]
    fn front_matter_body_excludes_the_block_and_is_trimmed() {
        let markdown = "---\nname: a\ndescription: b\n---\n\n\n  indented body  \n\n";
        let skill = Skill::from_markdown(markdown).unwrap();
        assert_eq!(skill.body, "indented body");
    }

    #[test]
    fn to_markdown_round_trips() {
        let mut skill = deploy()
            .with_metadata("owner", json!("platform"))
            .with_metadata("stars", json!(4));
        skill.version = Some("0.1.0".to_owned());
        let markdown = skill.to_markdown().unwrap();
        assert!(markdown.starts_with("---\n"));
        assert!(markdown.contains("name: deploy"));
        assert!(markdown.ends_with("Run the deploy script.\n"));
        // `path` records where the skill came from and is not part of the document.
        let mut expected = skill.clone();
        expected.path = None;
        assert_eq!(Skill::from_markdown(&markdown).unwrap(), expected);
    }

    #[test]
    fn to_markdown_round_trips_a_heading_skill() {
        let skill = Skill::from_markdown("# Deploy\n\nRun it.\n").unwrap();
        let markdown = skill.to_markdown().unwrap();
        assert!(markdown.contains("name: Deploy"));
        assert!(markdown.contains("description: ''"));
        assert_eq!(Skill::from_markdown(&markdown).unwrap(), skill);
    }

    #[test]
    fn to_markdown_skips_reserved_metadata_keys() {
        let skill = deploy().with_metadata("name", json!("shadow"));
        let markdown = skill.to_markdown().unwrap();
        assert!(markdown.contains("name: deploy"));
        assert!(!markdown.contains("shadow"));
    }

    #[test]
    fn render_joins_heading_description_and_body() {
        let rendered = deploy().render();
        assert_eq!(
            rendered,
            "# deploy\n\nDeploy the service\n\nRun the deploy script."
        );
        let bare = Skill::new("bare", "");
        assert_eq!(bare.render(), "# bare");
    }

    #[test]
    fn matches_is_case_insensitive_across_name_description_and_keywords() {
        let skill = deploy();
        assert!(skill.matches("deploy"));
        assert!(skill.matches("DEPLOY"));
        assert!(skill.matches("service"));
        assert!(skill.matches("release"));
        assert!(skill.matches("please deploy the service"));
        assert!(!skill.matches("rollback"));
        assert!(!skill.matches(""));
        assert!(!skill.matches("   "));
    }
}
