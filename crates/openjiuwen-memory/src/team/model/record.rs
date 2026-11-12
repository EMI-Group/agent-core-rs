//! [`TeamMemoryRecord`] and team-scoped record ids.
//!
//! A record is a plain `serde` value: `snake_case` JSON, tolerant of unknown fields,
//! optional fields omitted when absent, and `metadata` always present as an object.
//! `kind` and `visibility` serialize as the strings documented in [`super::kind`].
use super::default_importance;
use super::empty_object;
use super::kind::{TeamMemoryKind, TeamVisibility};
use super::scope::TeamMemoryScope;
use super::validate::{
    normalize_tags, validate_component, validate_content, validate_importance, validate_tags,
    validate_title,
};
use super::{DEFAULT_IMPORTANCE, MAX_RECORD_ID_CHARS, MAX_SCOPE_ID_CHARS};
use chrono::{DateTime, Utc};
use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
/// Returns the current UTC time; the default of both timestamps.
fn utc_now() -> DateTime<Utc> {
    Utc::now()
}
/// A stored team memory record.
///
/// The record carries its whole scope so it can be persisted, filtered, and rendered
/// without consulting a store. `created_at`/`updated_at` are RFC 3339, `importance`
/// lives in `0.0..=1.0` and defaults to `0.5`, and an expired record (`expires_at` at
/// or before "now") is hidden from reads unless explicitly requested.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct TeamMemoryRecord {
    /// Unique record id (24 lowercase hex characters, see [`new_team_memory_id`]).
    pub id: String,
    /// The team that owns the record.
    pub team_id: String,
    /// The member the record belongs to; absent means team-wide.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub member_id: Option<String>,
    /// The coordination session the record belongs to, when session-scoped.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    /// The record kind.
    pub kind: TeamMemoryKind,
    /// The memory text.
    pub content: String,
    /// Optional short title.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// Relative importance in `0.0..=1.0`; defaults to `0.5`.
    #[serde(default = "default_importance")]
    pub importance: f32,
    /// Normalized, unique, lowercase tags.
    pub tags: Vec<String>,
    /// Free-form structured fields; always a JSON object, always serialized.
    #[serde(default = "empty_object")]
    pub metadata: Value,
    /// Creation time; defaults to "now".
    pub created_at: DateTime<Utc>,
    /// Last modification time; defaults to "now".
    pub updated_at: DateTime<Utc>,
    /// Optional provenance marker (leader extraction, member id, import).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    /// Who may read the record.
    pub visibility: TeamVisibility,
    /// Optional expiry; the record is hidden once `now >= expires_at`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<DateTime<Utc>>,
}
impl Default for TeamMemoryRecord {
    /// An invalid-but-constructible record: empty ids/content, default kind and
    /// visibility, importance `0.5`, object metadata, and now timestamps.
    fn default() -> Self {
        let now = utc_now();
        Self {
            id: String::new(),
            team_id: String::new(),
            member_id: None,
            session_id: None,
            kind: TeamMemoryKind::default(),
            content: String::new(),
            title: None,
            importance: DEFAULT_IMPORTANCE,
            tags: Vec::new(),
            metadata: empty_object(),
            created_at: now,
            updated_at: now,
            source: None,
            visibility: TeamVisibility::default(),
            expires_at: None,
        }
    }
}
impl TeamMemoryRecord {
    /// Builds a record with a generated id, the current time, and importance `0.5`.
    ///
    /// The scope's `team_id`/`member_id`/`session_id` are copied onto the record.
    #[must_use]
    pub fn new(scope: &TeamMemoryScope, content: impl Into<String>, kind: TeamMemoryKind) -> Self {
        let now = utc_now();
        Self {
            id: new_team_memory_id(&scope.team_id, now),
            team_id: scope.team_id.clone(),
            member_id: scope.member_id.clone(),
            session_id: scope.session_id.clone(),
            kind,
            content: content.into(),
            title: None,
            importance: DEFAULT_IMPORTANCE,
            tags: Vec::new(),
            metadata: empty_object(),
            created_at: now,
            updated_at: now,
            source: None,
            visibility: TeamVisibility::Team,
            expires_at: None,
        }
    }
    /// Replaces the record id.
    #[must_use]
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = id.into();
        self
    }
    /// Sets the record title.
    #[must_use]
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }
    /// Sets the record importance (validated on [`TeamMemoryRecord::validate`]).
    #[must_use]
    pub fn with_importance(mut self, importance: f32) -> Self {
        self.importance = importance;
        self
    }
    /// Replaces the tags, normalizing them via [`super::normalize_tags`].
    #[must_use]
    pub fn with_tags<I, S>(mut self, tags: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.tags = normalize_tags(tags.into_iter().map(Into::into).collect());
        self
    }
    /// Inserts one metadata key, recreating the object if it was not one.
    #[must_use]
    pub fn with_metadata(mut self, key: impl Into<String>, value: Value) -> Self {
        if !self.metadata.is_object() {
            self.metadata = empty_object();
        }
        if let Some(object) = self.metadata.as_object_mut() {
            object.insert(key.into(), value);
        }
        self
    }
    /// Sets the provenance marker.
    #[must_use]
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }
    /// Sets the visibility.
    #[must_use]
    pub fn with_visibility(mut self, visibility: TeamVisibility) -> Self {
        self.visibility = visibility;
        self
    }
    /// Sets the expiry time.
    #[must_use]
    pub fn with_expires_at(mut self, expires_at: DateTime<Utc>) -> Self {
        self.expires_at = Some(expires_at);
        self
    }
    /// Returns `true` when `now` is at or past `expires_at`; records without an expiry
    /// never expire.
    #[must_use]
    pub fn is_expired(&self, now: DateTime<Utc>) -> bool {
        self.expires_at.is_some_and(|expires_at| now >= expires_at)
    }
    /// The metadata object, or `None` when `metadata` is not a JSON object.
    #[must_use]
    pub fn metadata_object(&self) -> Option<&Map<String, Value>> {
        self.metadata.as_object()
    }
    /// The string value of one metadata key, if present and a string.
    #[must_use]
    pub fn metadata_str(&self, key: &str) -> Option<&str> {
        self.metadata_object()?.get(key)?.as_str()
    }
    /// Validates every field the record carries.
    ///
    /// Ids go through the shared component rules, content/title must be non-blank and
    /// within their limits, importance must be finite and in `0.0..=1.0`, tags must be
    /// normalized and unique, metadata must be an object, and visibility must be
    /// valid.
    pub fn validate(&self) -> Result<()> {
        validate_component("id", &self.id, MAX_RECORD_ID_CHARS)?;
        validate_component("team_id", &self.team_id, MAX_SCOPE_ID_CHARS)?;
        if let Some(member_id) = &self.member_id {
            validate_component("member_id", member_id, MAX_SCOPE_ID_CHARS)?;
        }
        if let Some(session_id) = &self.session_id {
            validate_component("session_id", session_id, MAX_SCOPE_ID_CHARS)?;
        }
        validate_content(&self.content)?;
        if let Some(title) = &self.title {
            validate_title(title)?;
        }
        validate_importance(self.importance)?;
        validate_tags(&self.tags)?;
        if !self.metadata.is_object() {
            return Err(Error::validation(
                "team memory `metadata` must be a JSON object",
            ));
        }
        self.visibility.validate()
    }
}
/// Generates a team memory record id.
///
/// Delegates to [`crate::record::new_memory_id`]: 24 lowercase hex characters, laid
/// out as a 6-byte big-endian millisecond timestamp, 3 random bytes, and a 3-byte
/// FNV-1a hash of `team_id`, so ids sort by creation time and stay team-specific.
#[must_use]
pub fn new_team_memory_id(team_id: &str, now: DateTime<Utc>) -> String {
    crate::record::new_memory_id(team_id, now)
}
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    fn fixed_time() -> DateTime<Utc> {
        DateTime::from_timestamp(1_700_000_000, 0).unwrap()
    }
    fn sample_record() -> TeamMemoryRecord {
        TeamMemoryRecord::new(
            &TeamMemoryScope::member("review-team", "alice"),
            "some content",
            TeamMemoryKind::Decision,
        )
        .with_id("0123456789abcdef01234567")
        .with_title("Storage choice")
        .with_importance(0.8)
        .with_tags(["Storage", "sqlite"])
        .with_metadata("issue", serde_json::json!(42))
        .with_source("leader-extraction")
        .with_visibility(TeamVisibility::members(["alice", "bob"]))
    }
    #[test]
    fn new_copies_scope_and_defaults() {
        let scope = TeamMemoryScope::member("review-team", "alice").with_session("s1");
        let record = TeamMemoryRecord::new(&scope, "content", TeamMemoryKind::Task);
        assert_eq!(record.team_id, "review-team");
        assert_eq!(record.member_id.as_deref(), Some("alice"));
        assert_eq!(record.session_id.as_deref(), Some("s1"));
        assert_eq!(record.kind, TeamMemoryKind::Task);
        assert_eq!(record.content, "content");
        assert_eq!(record.id.len(), 24);
        assert!((record.importance - DEFAULT_IMPORTANCE).abs() < f32::EPSILON);
        assert_eq!(record.tags.len(), 0);
        assert_eq!(record.metadata, serde_json::json!({}));
        assert_eq!(record.visibility, TeamVisibility::Team);
        assert_eq!(record.created_at, record.updated_at);
        assert!(record.validate().is_ok());
    }
    #[test]
    fn serde_json_shape_and_round_trip() {
        let mut record = sample_record();
        record.created_at = fixed_time();
        record.updated_at = fixed_time();
        let json = serde_json::to_value(&record).unwrap();
        assert_eq!(json["id"], serde_json::json!("0123456789abcdef01234567"));
        assert_eq!(json["team_id"], serde_json::json!("review-team"));
        assert_eq!(json["member_id"], serde_json::json!("alice"));
        assert_eq!(json["kind"], serde_json::json!("decision"));
        assert_eq!(json["content"], serde_json::json!("some content"));
        assert_eq!(json["title"], serde_json::json!("Storage choice"));
        assert_eq!(json["importance"], serde_json::json!(0.8_f32));
        assert_eq!(json["tags"], serde_json::json!(["storage", "sqlite"]));
        assert_eq!(json["metadata"]["issue"], serde_json::json!(42));
        assert_eq!(
            json["visibility"],
            serde_json::json!({"members": ["alice", "bob"]})
        );
        assert_eq!(json["source"], serde_json::json!("leader-extraction"));
        assert!(json.get("session_id").is_none());
        assert!(json.get("expires_at").is_none());
        let restored: TeamMemoryRecord = serde_json::from_value(json).unwrap();
        assert_eq!(restored, record);
    }
    #[test]
    fn defaults_are_applied_when_fields_are_missing() {
        let record: TeamMemoryRecord = serde_json::from_value(serde_json::json!({})).unwrap();
        assert_eq!(record.id, "");
        assert_eq!(record.team_id, "");
        assert_eq!(record.kind, TeamMemoryKind::SharedFact);
        assert_eq!(record.visibility, TeamVisibility::Team);
        assert!((record.importance - DEFAULT_IMPORTANCE).abs() < f32::EPSILON);
        assert_eq!(record.metadata, serde_json::json!({}));
        assert_eq!(TeamMemoryRecord::default().kind, TeamMemoryKind::SharedFact);
        assert!(TeamMemoryRecord::default().validate().is_err());
    }
    #[test]
    fn validate_rejects_bad_ids() {
        assert!(sample_record().validate().is_ok());
        let mut record = sample_record();
        record.id = String::new();
        assert!(record.validate().unwrap_err().to_string().contains("`id`"));
        let mut record = sample_record();
        record.id = "a/b".to_owned();
        assert!(record.validate().unwrap_err().to_string().contains("`id`"));
        let mut record = sample_record();
        record.id = "x".repeat(MAX_RECORD_ID_CHARS + 1);
        assert!(record
            .validate()
            .unwrap_err()
            .to_string()
            .contains("at most 128"));
        let mut record = sample_record();
        record.team_id = "a\\b".to_owned();
        assert!(record
            .validate()
            .unwrap_err()
            .to_string()
            .contains("team_id"));
        let mut record = sample_record();
        record.member_id = Some("nul\0id".to_owned());
        assert!(record.validate().unwrap_err().to_string().contains("NUL"));
        let mut record = sample_record();
        record.session_id = Some(String::new());
        assert!(record
            .validate()
            .unwrap_err()
            .to_string()
            .contains("session_id"));
    }
    #[test]
    fn validate_rejects_bad_content_title_and_importance() {
        let mut record = sample_record();
        record.content = "   ".to_owned();
        assert!(record
            .validate()
            .unwrap_err()
            .to_string()
            .contains("content"));
        let mut record = sample_record();
        record.content = "x".repeat(super::super::MAX_CONTENT_CHARS + 1);
        assert!(record
            .validate()
            .unwrap_err()
            .to_string()
            .contains("at most"));
        let mut record = sample_record();
        record.title = Some("  ".to_owned());
        assert!(record.validate().unwrap_err().to_string().contains("title"));
        let mut record = sample_record();
        record.title = Some("x".repeat(super::super::MAX_TITLE_CHARS + 1));
        assert!(record
            .validate()
            .unwrap_err()
            .to_string()
            .contains("at most"));
        let mut record = sample_record();
        record.importance = 1.5;
        assert!(record
            .validate()
            .unwrap_err()
            .to_string()
            .contains("importance"));
        let mut record = sample_record();
        record.importance = f32::NAN;
        assert!(record
            .validate()
            .unwrap_err()
            .to_string()
            .contains("finite"));
    }
    #[test]
    fn validate_rejects_bad_tags_metadata_and_visibility() {
        let mut record = sample_record();
        record.tags = vec!["Storage".to_owned(), "storage".to_owned()];
        assert!(record
            .validate()
            .unwrap_err()
            .to_string()
            .contains("duplicate"));
        let mut record = sample_record();
        record.metadata = serde_json::json!("not-an-object");
        assert!(record
            .validate()
            .unwrap_err()
            .to_string()
            .contains("JSON object"));
        let mut record = sample_record();
        record.visibility = TeamVisibility::Members(Vec::new());
        assert!(record
            .validate()
            .unwrap_err()
            .to_string()
            .contains("at least one member"));
    }
    #[test]
    fn is_expired_uses_an_inclusive_boundary() {
        let record = sample_record();
        assert!(!record.is_expired(fixed_time()));
        let record = record.with_expires_at(fixed_time());
        let before = DateTime::from_timestamp(1_699_999_999, 0).unwrap();
        let at = fixed_time();
        let after = DateTime::from_timestamp(1_700_000_001, 0).unwrap();
        assert!(!record.is_expired(before));
        assert!(record.is_expired(at));
        assert!(record.is_expired(after));
    }
    #[test]
    fn metadata_helpers() {
        let record = sample_record();
        assert!(record.metadata_object().is_some());
        assert_eq!(record.metadata_str("issue"), None);
        let record = record.with_metadata("note", serde_json::json!("hello"));
        assert_eq!(record.metadata_str("note"), Some("hello"));
        assert_eq!(record.metadata_str("absent"), None);
        let record = TeamMemoryRecord::default().with_metadata("k", serde_json::json!(1));
        assert_eq!(record.metadata_str("k"), None);
        assert_eq!(record.metadata["k"], serde_json::json!(1));
    }
    #[test]
    fn new_team_memory_id_is_time_ordered_and_unique() {
        let earlier = DateTime::from_timestamp(1_700_000_000, 0).unwrap();
        let later = DateTime::from_timestamp(1_700_000_001, 0).unwrap();
        let first = new_team_memory_id("review-team", earlier);
        let second = new_team_memory_id("review-team", later);
        assert_eq!(first.len(), 24);
        assert!(first
            .chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
        assert!(first < second);
        let ids: std::collections::HashSet<String> = (0..16)
            .map(|_| new_team_memory_id("review-team", earlier))
            .collect();
        assert_eq!(ids.len(), 16);
        let alice = new_team_memory_id("team-a", earlier);
        let bob = new_team_memory_id("team-b", earlier);
        assert_eq!(alice[..12], bob[..12]);
        assert_ne!(alice[12..], bob[12..]);
    }
}
