//! [`CodingFact`] and [`CodingFactType`]: the lite coding-memory data model and its
//! record encoding.
//!
//! A coding fact is stored as an ordinary [`MemoryRecord`] of kind
//! [`MemoryType::Coding`], so it inherits the engine's scoping, persistence, dedupe,
//! and recall behavior. The fact fields that have no record column live in the
//! record's metadata under the keys exposed as `METADATA_*` constants:
//! `fact_type`, `name`, `description`, `tags`, `path` (only when set), and
//! `coding_fact` (always `true`, the marker that makes a record decodable as a fact).
//!
//! [`CodingFact::to_record`] and [`CodingFact::from_record`] are inverses for every
//! fact that came from [`CodingFact::new`], and [`CodingFact::from_record`] is
//! tolerant: unknown fact types fall back to [`CodingFactType::Project`] and malformed
//! optional fields decode to their defaults instead of failing.
use chrono::{DateTime, Utc};
use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::str::FromStr;

use crate::record::{MemoryRecord, MemoryScope, MemoryType};

/// Metadata key holding the fact type name (`"user"`, `"feedback"`, ...).
pub(crate) const METADATA_FACT_TYPE: &str = "fact_type";
/// Metadata key holding the fact's unique name.
pub(crate) const METADATA_NAME: &str = "name";
/// Metadata key holding the one-line description.
pub(crate) const METADATA_DESCRIPTION: &str = "description";
/// Metadata key holding the tags array (always present, possibly empty).
pub(crate) const METADATA_TAGS: &str = "tags";
/// Metadata key holding the optional source path.
pub(crate) const METADATA_PATH: &str = "path";
/// Metadata key marking a record as a coding fact (`true`).
pub(crate) const METADATA_CODING_FACT: &str = "coding_fact";

/// Maximum accepted fact-name length, in characters.
const MAX_NAME_CHARS: usize = 128;

/// The category of a coding fact.
///
/// The names match the Python lite-memory frontmatter types (`user`, `feedback`,
/// `project`, `reference`); [`CodingFactType::Project`] is the default so callers that
/// do not classify a fact still get an usable memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum CodingFactType {
    /// Durable facts about the user (preferences, identity, constraints).
    User,
    /// Corrections and guidance the user gave about how to work.
    Feedback,
    /// Knowledge about the project itself (structure, commands, conventions).
    #[default]
    Project,
    /// Pointers to external material (documents, APIs, repositories).
    Reference,
}

impl CodingFactType {
    /// Every fact type, in index order (`user`, `feedback`, `project`, `reference`).
    pub const ALL: [Self; 4] = [Self::User, Self::Feedback, Self::Project, Self::Reference];

    /// The `snake_case` name of this fact type, as stored in metadata and serialized.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::User => "user",
            Self::Feedback => "feedback",
            Self::Project => "project",
            Self::Reference => "reference",
        }
    }
}

impl FromStr for CodingFactType {
    /// A validation error listing the accepted fact-type names.
    type Err = Error;

    /// Parses a fact-type name, trimming and ASCII-lowercasing it first.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] naming the offending value when it is not one of
    /// `user`, `feedback`, `project`, or `reference`.
    fn from_str(raw: &str) -> Result<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "user" => Ok(Self::User),
            "feedback" => Ok(Self::Feedback),
            "project" => Ok(Self::Project),
            "reference" => Ok(Self::Reference),
            _ => Err(Error::validation(format!(
                "unknown coding fact type \"{raw}\"; expected one of \"user\", \"feedback\", \
                 \"project\", or \"reference\""
            ))),
        }
    }
}

/// A typed coding fact owned by one [`MemoryScope`].
///
/// `name` is the lookup key inside the scope; the engine keeps one record per
/// `(scope, name)`. `content` is the fact body (Markdown), `description` is the
/// one-line summary the rendered index shows, and `path` is an optional source
/// location used for citations.
///
/// JSON is flat and `snake_case`: `fact_type` serializes as `"type"` (and accepts the
/// `"fact_type"` alias), and the scope fields are flattened onto the top level.
/// Missing fields fall back to the defaults below.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case", default)]
pub struct CodingFact {
    /// Unique name of the fact inside its scope (at most 128 characters, no path
    /// separators).
    pub name: String,
    /// One-line summary used by the rendered index.
    pub description: String,
    /// The fact category; JSON key `"type"` (accepts `"fact_type"`).
    #[serde(rename = "type", alias = "fact_type")]
    pub fact_type: CodingFactType,
    /// Owning user/scope (and optional session/agent), flattened into the JSON object.
    #[serde(flatten)]
    pub scope: MemoryScope,
    /// Markdown body of the fact.
    pub content: String,
    /// Free-form tags stored alongside the fact.
    pub tags: Vec<String>,
    /// Optional source path, used to render the `path#L1-L<n>` citation
    /// (see [`CodingMemory::citation`](crate::lite::CodingMemory::citation)).
    pub path: Option<String>,
    /// Last modification time of the fact.
    pub updated_at: DateTime<Utc>,
}

impl CodingFact {
    /// Builds a fact in `scope` with the current time, no tags, and no path.
    ///
    /// `description` starts empty and `fact_type` is [`CodingFactType::Project`]; use
    /// the builders to set them.
    #[must_use]
    pub fn new(scope: &MemoryScope, name: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: String::new(),
            fact_type: CodingFactType::Project,
            scope: scope.clone(),
            content: content.into(),
            tags: Vec::new(),
            path: None,
            updated_at: Utc::now(),
        }
    }

    /// Returns the fact with `description` set.
    #[must_use]
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    /// Returns the fact with `fact_type` set.
    #[must_use]
    pub fn with_type(mut self, fact_type: CodingFactType) -> Self {
        self.fact_type = fact_type;
        self
    }

    /// Returns the fact with `tags` set.
    #[must_use]
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    /// Returns the fact with `path` set.
    #[must_use]
    pub fn with_path(mut self, path: impl Into<String>) -> Self {
        self.path = Some(path.into());
        self
    }

    /// Validates the fact's name and scope.
    ///
    /// The content is deliberately not validated here: the engine rejects blank
    /// content with its own [`Error::Validation`], and an empty body is a legal
    /// intermediate state for a fact that is filled in later.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] naming the offending field when `name` is empty,
    /// longer than 128 characters, or contains `/` or `\`, and propagates
    /// [`MemoryScope::validate`] for an invalid scope.
    pub fn validate(&self) -> Result<()> {
        if self.name.trim().is_empty() {
            return Err(Error::validation(format!(
                "coding fact `name` must not be empty (got \"{}\")",
                self.name
            )));
        }
        let length = self.name.chars().count();
        if length > MAX_NAME_CHARS {
            return Err(Error::validation(format!(
                "coding fact `name` must be at most {MAX_NAME_CHARS} characters (got {length} in \
                 \"{}\")",
                self.name
            )));
        }
        if self.name.contains('/') || self.name.contains('\\') {
            return Err(Error::validation(format!(
                "coding fact `name` must not contain a path separator (got \"{}\")",
                self.name
            )));
        }
        self.scope.validate()
    }

    /// Returns the metadata map stored on the record, using `fact_type` as the stored
    /// type name.
    ///
    /// `fact_type` is a parameter because [`super::CodingMemory::write`] resolves it
    /// against [`crate::config::CodingMemoryConfig::fact_types`] before persisting;
    /// [`CodingFact::to_record`] passes the fact's own canonical name. `tags` is always
    /// present (even empty) and `path` is present only when set.
    pub(crate) fn metadata_map(&self, fact_type: &str) -> BTreeMap<String, Value> {
        let mut metadata = BTreeMap::new();
        metadata.insert(
            METADATA_FACT_TYPE.to_owned(),
            Value::String(fact_type.to_owned()),
        );
        metadata.insert(METADATA_NAME.to_owned(), Value::String(self.name.clone()));
        metadata.insert(
            METADATA_DESCRIPTION.to_owned(),
            Value::String(self.description.clone()),
        );
        metadata.insert(
            METADATA_TAGS.to_owned(),
            Value::Array(self.tags.iter().cloned().map(Value::String).collect()),
        );
        if let Some(path) = &self.path {
            metadata.insert(METADATA_PATH.to_owned(), Value::String(path.clone()));
        }
        metadata.insert(METADATA_CODING_FACT.to_owned(), Value::Bool(true));
        metadata
    }

    /// Encodes the fact as a [`MemoryType::Coding`] memory record.
    ///
    /// The generated 24-hex id is kept so the engine can persist the record; both
    /// timestamps are set to [`CodingFact::updated_at`].
    #[must_use]
    pub fn to_record(&self) -> MemoryRecord {
        let mut record =
            MemoryRecord::new(self.scope.clone(), self.content.clone(), MemoryType::Coding)
                .with_timestamp(self.updated_at);
        record.updated_at = self.updated_at;
        record.metadata = self.metadata_map(self.fact_type.as_str());
        record
    }

    /// Decodes a stored record, or `None` when it is not a coding fact.
    ///
    /// A record qualifies when its kind is [`MemoryType::Coding`], its `coding_fact`
    /// metadata is exactly `true`, and its `name` metadata is a non-blank string.
    /// Decoding is otherwise tolerant: an unknown `fact_type` becomes
    /// [`CodingFactType::Project`], non-string tags are dropped, and a missing or
    /// null `path` becomes `None`.
    #[must_use]
    pub fn from_record(record: &MemoryRecord) -> Option<Self> {
        if record.kind != MemoryType::Coding {
            return None;
        }
        if record.metadata.get(METADATA_CODING_FACT) != Some(&Value::Bool(true)) {
            return None;
        }
        let name = record.metadata_str(METADATA_NAME)?;
        if name.trim().is_empty() {
            return None;
        }
        let tags = record
            .metadata
            .get(METADATA_TAGS)
            .and_then(Value::as_array)
            .map(|tags| {
                tags.iter()
                    .filter_map(Value::as_str)
                    .map(str::to_owned)
                    .collect()
            })
            .unwrap_or_default();
        let fact_type = record
            .metadata_str(METADATA_FACT_TYPE)
            .and_then(|raw| raw.parse().ok())
            .unwrap_or_default();
        Some(Self {
            name: name.to_owned(),
            description: record
                .metadata_str(METADATA_DESCRIPTION)
                .unwrap_or_default()
                .to_owned(),
            fact_type,
            scope: record.scope.clone(),
            content: record.content.clone(),
            tags,
            path: record
                .metadata
                .get(METADATA_PATH)
                .and_then(Value::as_str)
                .map(str::to_owned),
            updated_at: record.updated_at,
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn fixed_time() -> DateTime<Utc> {
        DateTime::from_timestamp(1_700_000_000, 0).unwrap()
    }

    fn scope() -> MemoryScope {
        MemoryScope::new("alice", "ws")
    }

    fn sample_fact() -> CodingFact {
        let mut fact = CodingFact::new(&scope(), "verify-build", "Run `cargo test`.")
            .with_description("Run the test suite")
            .with_type(CodingFactType::Feedback)
            .with_tags(vec!["build".to_owned(), "ci".to_owned()])
            .with_path("docs/BUILD.md");
        fact.updated_at = fixed_time();
        fact
    }

    #[test]
    fn fact_type_serde_uses_snake_case_names() {
        for (fact_type, name) in [
            (CodingFactType::User, "user"),
            (CodingFactType::Feedback, "feedback"),
            (CodingFactType::Project, "project"),
            (CodingFactType::Reference, "reference"),
        ] {
            let json = serde_json::to_value(fact_type).unwrap();
            assert_eq!(json, serde_json::json!(name));
            let restored: CodingFactType = serde_json::from_value(json).unwrap();
            assert_eq!(restored, fact_type);
        }
    }

    #[test]
    fn fact_type_all_is_ordered_and_as_str_matches() {
        assert_eq!(
            CodingFactType::ALL.map(CodingFactType::as_str),
            ["user", "feedback", "project", "reference"]
        );
        assert_eq!(CodingFactType::default(), CodingFactType::Project);
    }

    #[test]
    fn fact_type_from_str_trims_and_lowercases() {
        for (raw, expected) in [
            ("user", CodingFactType::User),
            (" Feedback ", CodingFactType::Feedback),
            ("PROJECT", CodingFactType::Project),
            ("  reference\n", CodingFactType::Reference),
        ] {
            assert_eq!(raw.parse::<CodingFactType>().unwrap(), expected);
        }
    }

    #[test]
    fn fact_type_from_str_rejects_unknown_values() {
        let error = "note".parse::<CodingFactType>().unwrap_err();
        assert!(matches!(error, Error::Validation(_)), "{error:?}");
        let message = error.to_string();
        assert!(message.contains("note"), "{message}");
        for accepted in ["user", "feedback", "project", "reference"] {
            assert!(message.contains(accepted), "{message}");
        }
        assert!("".parse::<CodingFactType>().is_err());
    }

    #[test]
    fn coding_fact_json_round_trip_is_flat_and_uses_type() {
        let fact = sample_fact();
        let json = serde_json::to_value(&fact).unwrap();
        assert_eq!(json["type"], serde_json::json!("feedback"));
        assert!(json.get("fact_type").is_none());
        assert_eq!(json["name"], serde_json::json!("verify-build"));
        assert_eq!(json["description"], serde_json::json!("Run the test suite"));
        assert_eq!(json["content"], serde_json::json!("Run `cargo test`."));
        assert_eq!(json["tags"], serde_json::json!(["build", "ci"]));
        assert_eq!(json["path"], serde_json::json!("docs/BUILD.md"));
        assert_eq!(json["user_id"], serde_json::json!("alice"));
        assert_eq!(json["scope_id"], serde_json::json!("ws"));
        assert!(json.get("scope").is_none());
        let restored: CodingFact = serde_json::from_value(json).unwrap();
        assert_eq!(restored, fact);
    }

    #[test]
    fn coding_fact_json_accepts_the_fact_type_alias_and_defaults() {
        let fact: CodingFact = serde_json::from_value(serde_json::json!({
            "name": "build",
            "fact_type": "feedback",
            "content": "cargo build",
            "user_id": "alice",
            "scope_id": "ws",
        }))
        .unwrap();
        assert_eq!(fact.fact_type, CodingFactType::Feedback);
        assert_eq!(fact.description, "");
        assert_eq!(fact.tags, Vec::<String>::new());
        assert!(fact.path.is_none());
        assert_eq!(fact.updated_at, DateTime::<Utc>::default());
    }

    #[test]
    fn coding_fact_to_record_round_trips_for_every_type_and_path_state() {
        for fact_type in CodingFactType::ALL {
            for path in [None, Some("src/lib.rs")] {
                let mut fact = CodingFact::new(&scope(), "build-command", "cargo build")
                    .with_description("How to build")
                    .with_type(fact_type)
                    .with_tags(vec!["build".to_owned()]);
                if let Some(path) = path {
                    fact = fact.with_path(path);
                }
                fact.updated_at = fixed_time();

                let record = fact.to_record();
                assert_eq!(record.kind, MemoryType::Coding);
                assert_eq!(record.content, "cargo build");
                assert_eq!(record.id.len(), 24);
                assert!(record
                    .id
                    .chars()
                    .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
                assert_eq!(record.timestamp, fixed_time());
                assert_eq!(record.updated_at, fixed_time());
                assert_eq!(record.scope, scope());
                assert_eq!(
                    record.metadata.get(METADATA_CODING_FACT),
                    Some(&Value::Bool(true))
                );
                assert_eq!(record.metadata_str(METADATA_NAME), Some("build-command"));
                assert_eq!(
                    record.metadata_str(METADATA_DESCRIPTION),
                    Some("How to build")
                );
                assert_eq!(
                    record.metadata_str(METADATA_FACT_TYPE),
                    Some(fact_type.as_str())
                );
                assert_eq!(
                    record.metadata.get(METADATA_TAGS),
                    Some(&serde_json::json!(["build"]))
                );
                match path {
                    Some(path) => assert_eq!(record.metadata_str(METADATA_PATH), Some(path)),
                    None => assert!(!record.metadata.contains_key(METADATA_PATH)),
                }

                let restored = CodingFact::from_record(&record).expect("decodable fact");
                assert_eq!(restored, fact);
            }
        }
    }

    #[test]
    fn empty_tags_are_always_stored() {
        let record = CodingFact::new(&scope(), "n", "body").to_record();
        assert_eq!(
            record.metadata.get(METADATA_TAGS),
            Some(&serde_json::json!([]))
        );
        assert_eq!(
            CodingFact::from_record(&record).unwrap().tags,
            Vec::<String>::new()
        );
    }

    #[test]
    fn from_record_rejects_records_that_are_not_coding_facts() {
        let record = CodingFact::new(&scope(), "n", "body").to_record();

        let mut wrong_kind = record.clone();
        wrong_kind.kind = MemoryType::SemanticMemory;
        assert!(CodingFact::from_record(&wrong_kind).is_none());

        let mut missing_marker = record.clone();
        missing_marker.metadata.remove(METADATA_CODING_FACT);
        assert!(CodingFact::from_record(&missing_marker).is_none());

        let mut marker_false = record.clone();
        marker_false
            .metadata
            .insert(METADATA_CODING_FACT.to_owned(), Value::Bool(false));
        assert!(CodingFact::from_record(&marker_false).is_none());

        let mut marker_not_a_bool = record.clone();
        marker_not_a_bool
            .metadata
            .insert(METADATA_CODING_FACT.to_owned(), Value::from(1));
        assert!(CodingFact::from_record(&marker_not_a_bool).is_none());

        let mut missing_name = record.clone();
        missing_name.metadata.remove(METADATA_NAME);
        assert!(CodingFact::from_record(&missing_name).is_none());

        for name in ["", "   "] {
            let mut blank_name = record.clone();
            blank_name
                .metadata
                .insert(METADATA_NAME.to_owned(), Value::String(name.to_owned()));
            assert!(CodingFact::from_record(&blank_name).is_none());
        }
    }

    #[test]
    fn unknown_or_missing_fact_type_falls_back_to_project() {
        let mut record = CodingFact::new(&scope(), "n", "body").to_record();
        record.metadata.insert(
            METADATA_FACT_TYPE.to_owned(),
            Value::String("note".to_owned()),
        );
        assert_eq!(
            CodingFact::from_record(&record).unwrap().fact_type,
            CodingFactType::Project
        );
        record.metadata.remove(METADATA_FACT_TYPE);
        assert_eq!(
            CodingFact::from_record(&record).unwrap().fact_type,
            CodingFactType::Project
        );
    }

    #[test]
    fn malformed_tags_and_path_decode_tolerantly() {
        let mut record = CodingFact::new(&scope(), "n", "body").to_record();
        record.metadata.insert(
            METADATA_TAGS.to_owned(),
            serde_json::json!(["keep", 7, null, "also"]),
        );
        record
            .metadata
            .insert(METADATA_PATH.to_owned(), Value::Null);
        let fact = CodingFact::from_record(&record).unwrap();
        assert_eq!(fact.tags, vec!["keep".to_owned(), "also".to_owned()]);
        assert!(fact.path.is_none());

        record.metadata.insert(
            METADATA_TAGS.to_owned(),
            Value::String("not-an-array".to_owned()),
        );
        assert_eq!(
            CodingFact::from_record(&record).unwrap().tags,
            Vec::<String>::new()
        );
    }

    #[test]
    fn validate_checks_the_name_matrix() {
        let mut fact = CodingFact::new(&scope(), "ok", "body");
        assert!(fact.validate().is_ok());

        fact.name = "a".repeat(128);
        assert!(fact.validate().is_ok());

        fact.name = "a".repeat(129);
        let message = fact.validate().unwrap_err().to_string();
        assert!(message.contains("name"), "{message}");

        fact.name = "a/b".to_owned();
        assert!(fact.validate().unwrap_err().to_string().contains("name"));

        fact.name = "a\\b".to_owned();
        assert!(fact.validate().unwrap_err().to_string().contains("name"));

        fact.name = "   ".to_owned();
        assert!(fact.validate().unwrap_err().to_string().contains("name"));

        fact.name = String::new();
        assert!(fact.validate().unwrap_err().to_string().contains("name"));
    }

    #[test]
    fn validate_accepts_empty_content_and_rejects_an_invalid_scope() {
        assert!(CodingFact::new(&scope(), "n", "").validate().is_ok());
        let fact = CodingFact::new(&MemoryScope::new("", "ws"), "n", "body");
        let message = fact.validate().unwrap_err().to_string();
        assert!(message.contains("user_id"), "{message}");
    }

    #[test]
    fn new_sets_defaults_and_builders_chain() {
        let fact = CodingFact::new(&scope(), "name", "body");
        assert_eq!(fact.name, "name");
        assert_eq!(fact.content, "body");
        assert_eq!(fact.description, "");
        assert_eq!(fact.fact_type, CodingFactType::Project);
        assert_eq!(fact.tags, Vec::<String>::new());
        assert!(fact.path.is_none());
        assert_eq!(fact.scope, scope());

        let built = fact
            .with_description("d")
            .with_type(CodingFactType::Reference)
            .with_tags(vec!["t".to_owned()])
            .with_path("docs/REF.md");
        assert_eq!(built.description, "d");
        assert_eq!(built.fact_type, CodingFactType::Reference);
        assert_eq!(built.tags, vec!["t".to_owned()]);
        assert_eq!(built.path.as_deref(), Some("docs/REF.md"));
    }
}
