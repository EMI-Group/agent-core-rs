//! [`NewTeamMemory`] and [`TeamMemoryUpdate`]: the write inputs of team memory.
//!
//! [`NewTeamMemory`] is the unnormalized create input; [`NewTeamMemory::to_record`]
//! validates it, generates an id when absent, clamps importance, normalizes tags and
//! visibility, stamps the timestamps, and returns a validated
//! [`TeamMemoryRecord`]. [`TeamMemoryUpdate`] is a partial update applied in place by
//! [`TeamMemoryUpdate::apply`].
use super::empty_object;
use super::kind::{TeamMemoryKind, TeamVisibility};
use super::record::{new_team_memory_id, TeamMemoryRecord};
use super::validate::{
    normalize_members, normalize_tags, validate_component, validate_content, validate_tag_entries,
    validate_title,
};
use super::{DEFAULT_IMPORTANCE, MAX_RECORD_ID_CHARS, MAX_SCOPE_ID_CHARS};
use chrono::{DateTime, Utc};
use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
/// Normalizes the member list of a [`TeamVisibility::Members`] visibility; other
/// visibilities pass through unchanged.
fn normalize_visibility(visibility: &TeamVisibility) -> TeamVisibility {
    match visibility {
        TeamVisibility::Members(members) => {
            TeamVisibility::Members(normalize_members(members.clone()))
        }
        other => other.clone(),
    }
}
/// A create request for one team memory record.
///
/// Every field except `team_id`/`content`/`kind` is optional; [`Self::validate`] checks
/// caller input as supplied, while [`Self::to_record`] normalizes and re-validates the
/// resulting record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct NewTeamMemory {
    /// Explicit record id; generated from `team_id` and "now" when absent.
    pub id: Option<String>,
    /// The team that owns the record.
    pub team_id: String,
    /// The member the record belongs to; absent means team-wide.
    pub member_id: Option<String>,
    /// The coordination session the record belongs to, when session-scoped.
    pub session_id: Option<String>,
    /// The record kind.
    pub kind: TeamMemoryKind,
    /// The memory text.
    pub content: String,
    /// Optional short title.
    pub title: Option<String>,
    /// Requested importance; defaults to the caller-supplied fallback on
    /// materialization.
    pub importance: Option<f32>,
    /// Unnormalized tags.
    pub tags: Vec<String>,
    /// Free-form structured fields; must be a JSON object.
    pub metadata: Value,
    /// Optional provenance marker.
    pub source: Option<String>,
    /// Who may read the record.
    pub visibility: TeamVisibility,
    /// Optional expiry time.
    pub expires_at: Option<DateTime<Utc>>,
}
impl Default for NewTeamMemory {
    /// An empty create request; [`NewTeamMemory::validate`] rejects it.
    fn default() -> Self {
        Self {
            id: None,
            team_id: String::new(),
            member_id: None,
            session_id: None,
            kind: TeamMemoryKind::default(),
            content: String::new(),
            title: None,
            importance: None,
            tags: Vec::new(),
            metadata: empty_object(),
            source: None,
            visibility: TeamVisibility::default(),
            expires_at: None,
        }
    }
}
impl NewTeamMemory {
    /// Builds a create request for `content` of `kind` in `team_id`.
    #[must_use]
    pub fn new(
        team_id: impl Into<String>,
        content: impl Into<String>,
        kind: TeamMemoryKind,
    ) -> Self {
        Self {
            team_id: team_id.into(),
            content: content.into(),
            kind,
            ..Self::default()
        }
    }
    /// Builds a [`TeamMemoryKind::SharedFact`] request.
    #[must_use]
    pub fn shared_fact(team_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self::new(team_id, content, TeamMemoryKind::SharedFact)
    }
    /// Builds a [`TeamMemoryKind::MemberProfile`] request.
    #[must_use]
    pub fn member_profile(team_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self::new(team_id, content, TeamMemoryKind::MemberProfile)
    }
    /// Builds a [`TeamMemoryKind::Decision`] request.
    #[must_use]
    pub fn decision(team_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self::new(team_id, content, TeamMemoryKind::Decision)
    }
    /// Builds a [`TeamMemoryKind::Task`] request.
    #[must_use]
    pub fn task(team_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self::new(team_id, content, TeamMemoryKind::Task)
    }
    /// Builds a [`TeamMemoryKind::Artifact`] request.
    #[must_use]
    pub fn artifact(team_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self::new(team_id, content, TeamMemoryKind::Artifact)
    }
    /// Builds a [`TeamMemoryKind::HandoffNote`] request.
    #[must_use]
    pub fn handoff_note(team_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self::new(team_id, content, TeamMemoryKind::HandoffNote)
    }
    /// Sets an explicit record id.
    #[must_use]
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }
    /// Sets the owning member.
    #[must_use]
    pub fn with_member(mut self, member_id: impl Into<String>) -> Self {
        self.member_id = Some(member_id.into());
        self
    }
    /// Sets the owning session.
    #[must_use]
    pub fn with_session(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }
    /// Sets the title.
    #[must_use]
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }
    /// Sets the requested importance.
    #[must_use]
    pub fn with_importance(mut self, importance: f32) -> Self {
        self.importance = Some(importance);
        self
    }
    /// Appends one tag (normalized on materialization).
    #[must_use]
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
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
    /// Validates the caller input as supplied.
    ///
    /// Ids go through the shared component rules, content/title must be non-blank and
    /// within their limits, a supplied importance must be finite (it is clamped on
    /// materialization), tags must be valid entries (duplicates allowed until
    /// normalization), metadata must be an object, and visibility must be valid.
    pub fn validate(&self) -> Result<()> {
        if let Some(id) = &self.id {
            validate_component("id", id, MAX_RECORD_ID_CHARS)?;
        }
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
        if let Some(importance) = self.importance {
            if !importance.is_finite() {
                return Err(Error::validation(format!(
                    "team memory `importance` must be finite (got {importance})"
                )));
            }
        }
        validate_tag_entries(&self.tags)?;
        if !self.metadata.is_object() {
            return Err(Error::validation(
                "team memory `metadata` must be a JSON object",
            ));
        }
        self.visibility.validate()
    }
    /// Materializes a validated [`TeamMemoryRecord`] at `now`.
    ///
    /// A missing id is generated, the title is trimmed, importance is clamped into
    /// `0.0..=1.0` (falling back to `default_importance`, clamped, or
    /// [`DEFAULT_IMPORTANCE`] when that fallback is not finite), tags and the
    /// `Members` visibility list are normalized, and both timestamps are `now`.
    pub fn to_record(
        &self,
        now: DateTime<Utc>,
        default_importance: f32,
    ) -> Result<TeamMemoryRecord> {
        self.validate()?;
        let record = TeamMemoryRecord {
            id: self
                .id
                .clone()
                .unwrap_or_else(|| new_team_memory_id(&self.team_id, now)),
            team_id: self.team_id.clone(),
            member_id: self.member_id.clone(),
            session_id: self.session_id.clone(),
            kind: self.kind.clone(),
            content: self.content.clone(),
            title: self.title.as_deref().map(str::trim).map(str::to_owned),
            importance: self.resolve_importance(default_importance),
            tags: normalize_tags(self.tags.clone()),
            // The embedding is derived state: only the facade that owns an embedder
            // writes one, after the record is materialized.
            embedding: None,
            metadata: self.metadata.clone(),
            created_at: now,
            updated_at: now,
            source: self.source.clone(),
            visibility: normalize_visibility(&self.visibility),
            expires_at: self.expires_at,
        };
        record.validate()?;
        Ok(record)
    }
    /// Resolves the importance to store: a finite supplied value clamped into
    /// `0.0..=1.0`, otherwise the (clamped) fallback, otherwise [`DEFAULT_IMPORTANCE`].
    fn resolve_importance(&self, fallback: f32) -> f32 {
        let chosen = match self.importance {
            Some(value) if value.is_finite() => value,
            Some(_) | None => {
                if fallback.is_finite() {
                    fallback
                } else {
                    DEFAULT_IMPORTANCE
                }
            }
        };
        chosen.clamp(0.0, 1.0)
    }
}
/// A partial update of an existing team memory record.
///
/// Only set fields are applied; the metadata map is merged key-by-key (incoming
/// wins). [`Self::validate`] checks the supplied values; [`Self::apply`] clamps,
/// normalizes, stamps `updated_at`, and re-validates the record.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct TeamMemoryUpdate {
    /// Replacement content.
    pub content: Option<String>,
    /// Replacement kind.
    pub kind: Option<TeamMemoryKind>,
    /// Replacement title; a blank title clears it.
    pub title: Option<String>,
    /// Replacement importance (clamped on apply).
    pub importance: Option<f32>,
    /// Replacement tag list (normalized on apply).
    pub tags: Option<Vec<String>>,
    /// Metadata entries merged into the record.
    pub metadata: BTreeMap<String, Value>,
    /// Replacement provenance marker.
    pub source: Option<String>,
    /// Replacement visibility (member list normalized on apply).
    pub visibility: Option<TeamVisibility>,
    /// Replacement expiry; absent leaves the current value untouched.
    pub expires_at: Option<DateTime<Utc>>,
}
impl TeamMemoryUpdate {
    /// Sets the content.
    #[must_use]
    pub fn content(mut self, content: impl Into<String>) -> Self {
        self.content = Some(content.into());
        self
    }
    /// Sets the kind.
    #[must_use]
    pub fn with_kind(mut self, kind: TeamMemoryKind) -> Self {
        self.kind = Some(kind);
        self
    }
    /// Sets the title.
    #[must_use]
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }
    /// Sets the importance.
    #[must_use]
    pub fn with_importance(mut self, importance: f32) -> Self {
        self.importance = Some(importance);
        self
    }
    /// Replaces the tags.
    #[must_use]
    pub fn with_tags<I, S>(mut self, tags: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.tags = Some(tags.into_iter().map(Into::into).collect());
        self
    }
    /// Inserts one metadata key.
    #[must_use]
    pub fn with_metadata(mut self, key: impl Into<String>, value: Value) -> Self {
        self.metadata.insert(key.into(), value);
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
        self.visibility = Some(visibility);
        self
    }
    /// Sets the expiry.
    #[must_use]
    pub fn with_expires_at(mut self, expires_at: DateTime<Utc>) -> Self {
        self.expires_at = Some(expires_at);
        self
    }
    /// Returns `true` when the update would change nothing.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.content.is_none()
            && self.kind.is_none()
            && self.title.is_none()
            && self.importance.is_none()
            && self.tags.is_none()
            && self.metadata.is_empty()
            && self.source.is_none()
            && self.visibility.is_none()
            && self.expires_at.is_none()
    }
    /// Validates the supplied values.
    ///
    /// Content and title must be valid when present, importance only has to be finite
    /// (it is clamped on apply), tags must be valid entries, and visibility must be
    /// valid.
    pub fn validate(&self) -> Result<()> {
        if let Some(content) = &self.content {
            validate_content(content)?;
        }
        if let Some(title) = &self.title {
            validate_title(title)?;
        }
        if let Some(importance) = self.importance {
            if !importance.is_finite() {
                return Err(Error::validation(format!(
                    "team memory `importance` must be finite (got {importance})"
                )));
            }
        }
        if let Some(tags) = &self.tags {
            validate_tag_entries(tags)?;
        }
        if let Some(visibility) = &self.visibility {
            visibility.validate()?;
        }
        Ok(())
    }
    /// Applies the update to `record` and re-validates it.
    ///
    /// Only set fields change; the title is trimmed and a blank title clears it,
    /// importance is clamped into `0.0..=1.0`, tags and the `Members` list are
    /// normalized, metadata is merged key-by-key (a non-object metadata value is
    /// replaced with `{}` first), and `updated_at` becomes `now`.
    pub fn apply(&self, record: &mut TeamMemoryRecord, now: DateTime<Utc>) -> Result<()> {
        if let Some(content) = &self.content {
            record.content.clone_from(content);
        }
        if let Some(kind) = &self.kind {
            record.kind = kind.clone();
        }
        if let Some(title) = &self.title {
            let trimmed = title.trim();
            record.title = (!trimmed.is_empty()).then(|| trimmed.to_owned());
        }
        if let Some(importance) = self.importance {
            if importance.is_finite() {
                record.importance = importance.clamp(0.0, 1.0);
            }
        }
        if let Some(tags) = &self.tags {
            record.tags = normalize_tags(tags.clone());
        }
        if !self.metadata.is_empty() {
            if !record.metadata.is_object() {
                record.metadata = empty_object();
            }
            if let Some(object) = record.metadata.as_object_mut() {
                for (key, value) in &self.metadata {
                    object.insert(key.clone(), value.clone());
                }
            }
        }
        if let Some(source) = &self.source {
            record.source = Some(source.clone());
        }
        if let Some(visibility) = &self.visibility {
            record.visibility = normalize_visibility(visibility);
        }
        if let Some(expires_at) = self.expires_at {
            record.expires_at = Some(expires_at);
        }
        record.updated_at = now;
        record.validate()
    }
}
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::team::model::TeamMemoryScope;
    fn fixed_time() -> DateTime<Utc> {
        DateTime::from_timestamp(1_700_000_000, 0).unwrap()
    }
    #[test]
    fn constructors_pick_the_declared_kind() {
        assert_eq!(
            NewTeamMemory::new("team", "c", TeamMemoryKind::Task).kind,
            TeamMemoryKind::Task
        );
        assert_eq!(
            NewTeamMemory::shared_fact("team", "c").kind,
            TeamMemoryKind::SharedFact
        );
        assert_eq!(
            NewTeamMemory::member_profile("team", "c").kind,
            TeamMemoryKind::MemberProfile
        );
        assert_eq!(
            NewTeamMemory::decision("team", "c").kind,
            TeamMemoryKind::Decision
        );
        assert_eq!(NewTeamMemory::task("team", "c").kind, TeamMemoryKind::Task);
        assert_eq!(
            NewTeamMemory::artifact("team", "c").kind,
            TeamMemoryKind::Artifact
        );
        assert_eq!(
            NewTeamMemory::handoff_note("team", "c").kind,
            TeamMemoryKind::HandoffNote
        );
        assert!(NewTeamMemory::default().validate().is_err());
    }
    #[test]
    fn materialization_normalizes_and_defaults() {
        let input = NewTeamMemory::decision("review-team", "Use SQLite.")
            .with_member("alice")
            .with_session("s1")
            .with_title("  Storage choice  ")
            .with_importance(1.7)
            .with_tag(" Storage ")
            .with_tag("storage")
            .with_tag("sqlite")
            .with_metadata("issue", serde_json::json!(42))
            .with_source("leader")
            .with_visibility(TeamVisibility::members([" alice ", "alice", "bob"]));
        let record = input.to_record(fixed_time(), 0.5).unwrap();
        assert_eq!(record.id.len(), 24);
        assert_eq!(record.team_id, "review-team");
        assert_eq!(record.member_id.as_deref(), Some("alice"));
        assert_eq!(record.session_id.as_deref(), Some("s1"));
        assert_eq!(record.title.as_deref(), Some("Storage choice"));
        assert!((record.importance - 1.0).abs() < f32::EPSILON);
        assert_eq!(record.tags, vec!["storage".to_owned(), "sqlite".to_owned()]);
        assert_eq!(record.metadata["issue"], serde_json::json!(42));
        assert_eq!(record.source.as_deref(), Some("leader"));
        assert_eq!(
            record.visibility,
            TeamVisibility::Members(vec!["alice".to_owned(), "bob".to_owned()])
        );
        assert_eq!(record.created_at, fixed_time());
        assert_eq!(record.updated_at, fixed_time());
        assert!(record.validate().is_ok());
    }
    #[test]
    fn materialization_uses_the_fallback_importance() {
        let input = NewTeamMemory::shared_fact("team", "c");
        let record = input.to_record(fixed_time(), 0.9).unwrap();
        assert!((record.importance - 0.9).abs() < f32::EPSILON);
        let record = input.to_record(fixed_time(), 3.0).unwrap();
        assert!((record.importance - 1.0).abs() < f32::EPSILON);
        let record = input.to_record(fixed_time(), f32::NAN).unwrap();
        assert!((record.importance - DEFAULT_IMPORTANCE).abs() < f32::EPSILON);
    }
    #[test]
    fn materialization_keeps_an_explicit_id() {
        let input = NewTeamMemory::task("team", "c").with_id("0123456789abcdef01234567");
        let record = input.to_record(fixed_time(), 0.5).unwrap();
        assert_eq!(record.id, "0123456789abcdef01234567");
    }
    #[test]
    fn validate_rejects_bad_input() {
        assert!(NewTeamMemory::shared_fact("", "c").validate().is_err());
        assert!(NewTeamMemory::shared_fact("a/b", "c")
            .validate()
            .unwrap_err()
            .to_string()
            .contains("team_id"));
        assert!(NewTeamMemory::shared_fact("team", "  ")
            .validate()
            .unwrap_err()
            .to_string()
            .contains("content"));
        assert!(NewTeamMemory::shared_fact("team", "c")
            .with_title(" ")
            .validate()
            .unwrap_err()
            .to_string()
            .contains("title"));
        assert!(NewTeamMemory::shared_fact("team", "c")
            .with_importance(f32::NAN)
            .validate()
            .unwrap_err()
            .to_string()
            .contains("finite"));
        assert!(NewTeamMemory::shared_fact("team", "c")
            .with_tag("")
            .validate()
            .unwrap_err()
            .to_string()
            .contains("empty entries"));
        assert!(NewTeamMemory::shared_fact("team", "c")
            .with_id("a/b")
            .validate()
            .unwrap_err()
            .to_string()
            .contains("`id`"));
        assert!(NewTeamMemory::shared_fact("team", "c")
            .with_member("")
            .validate()
            .unwrap_err()
            .to_string()
            .contains("member_id"));
        assert!(NewTeamMemory::shared_fact("team", "c")
            .with_session("s/1")
            .validate()
            .unwrap_err()
            .to_string()
            .contains("session_id"));
        let mut input = NewTeamMemory::shared_fact("team", "c");
        input.metadata = serde_json::json!([1, 2]);
        assert!(input
            .validate()
            .unwrap_err()
            .to_string()
            .contains("JSON object"));
        assert!(NewTeamMemory::shared_fact("team", "c")
            .with_visibility(TeamVisibility::Members(Vec::new()))
            .validate()
            .unwrap_err()
            .to_string()
            .contains("at least one member"));
    }
    #[test]
    fn update_applies_every_field() {
        let mut record = TeamMemoryRecord::new(
            &TeamMemoryScope::member("team", "alice"),
            "old",
            TeamMemoryKind::Task,
        )
        .with_id("0123456789abcdef01234567")
        .with_metadata("keep", serde_json::json!(1))
        .with_metadata("replace", serde_json::json!("before"));
        let later = DateTime::from_timestamp(1_700_000_100, 0).unwrap();
        let update = TeamMemoryUpdate::default()
            .content("new")
            .with_kind(TeamMemoryKind::Decision)
            .with_title("  Chosen  ")
            .with_importance(1.6)
            .with_tags([" Storage ", "storage", "sqlite"])
            .with_metadata("replace", serde_json::json!("after"))
            .with_metadata("added", serde_json::json!(true))
            .with_source("leader")
            .with_visibility(TeamVisibility::members(["alice", "alice", "bob"]))
            .with_expires_at(later);
        update.validate().unwrap();
        update.apply(&mut record, later).unwrap();
        assert_eq!(record.content, "new");
        assert_eq!(record.kind, TeamMemoryKind::Decision);
        assert_eq!(record.title.as_deref(), Some("Chosen"));
        assert!((record.importance - 1.0).abs() < f32::EPSILON);
        assert_eq!(record.tags, vec!["storage".to_owned(), "sqlite".to_owned()]);
        assert_eq!(record.metadata["keep"], serde_json::json!(1));
        assert_eq!(record.metadata["replace"], serde_json::json!("after"));
        assert_eq!(record.metadata["added"], serde_json::json!(true));
        assert_eq!(record.source.as_deref(), Some("leader"));
        assert_eq!(
            record.visibility,
            TeamVisibility::Members(vec!["alice".to_owned(), "bob".to_owned()])
        );
        assert_eq!(record.expires_at, Some(later));
        assert_eq!(record.updated_at, later);
    }
    #[test]
    fn blank_title_clears_and_low_importance_clamps() {
        let mut record = TeamMemoryRecord::new(
            &TeamMemoryScope::new("team"),
            "content",
            TeamMemoryKind::Task,
        )
        .with_id("0123456789abcdef01234567")
        .with_title("Existing")
        .with_importance(0.8);
        let update = TeamMemoryUpdate::default()
            .with_title("   ")
            .with_importance(-0.5);
        update.apply(&mut record, fixed_time()).unwrap();
        assert_eq!(record.title, None);
        assert!(record.importance.abs() < f32::EPSILON);
    }
    #[test]
    fn update_replaces_non_object_metadata_before_merging() {
        let mut record = TeamMemoryRecord::new(
            &TeamMemoryScope::new("team"),
            "content",
            TeamMemoryKind::Task,
        )
        .with_id("0123456789abcdef01234567");
        record.metadata = serde_json::json!("broken");
        let update = TeamMemoryUpdate::default().with_metadata("k", serde_json::json!(1));
        update.apply(&mut record, fixed_time()).unwrap();
        assert_eq!(record.metadata, serde_json::json!({"k": 1}));
    }
    #[test]
    fn empty_update_is_a_no_op_except_the_timestamp() {
        let mut record = TeamMemoryRecord::new(
            &TeamMemoryScope::new("team"),
            "content",
            TeamMemoryKind::Task,
        )
        .with_id("0123456789abcdef01234567")
        .with_tags(["a"]);
        let before = record.clone();
        let update = TeamMemoryUpdate::default();
        assert!(update.is_empty());
        let later = DateTime::from_timestamp(1_700_000_100, 0).unwrap();
        update.apply(&mut record, later).unwrap();
        assert_eq!(record.content, before.content);
        assert_eq!(record.kind, before.kind);
        assert_eq!(record.title, before.title);
        assert_eq!(record.importance, before.importance);
        assert_eq!(record.tags, before.tags);
        assert_eq!(record.visibility, before.visibility);
        assert_eq!(record.updated_at, later);
    }
    #[test]
    fn is_empty_matrix() {
        assert!(TeamMemoryUpdate::default().is_empty());
        let cases = [
            TeamMemoryUpdate::default().content("x"),
            TeamMemoryUpdate::default().with_kind(TeamMemoryKind::Task),
            TeamMemoryUpdate::default().with_title("t"),
            TeamMemoryUpdate::default().with_importance(0.1),
            TeamMemoryUpdate::default().with_tags(["a"]),
            TeamMemoryUpdate::default().with_metadata("k", serde_json::json!(1)),
            TeamMemoryUpdate::default().with_source("s"),
            TeamMemoryUpdate::default().with_visibility(TeamVisibility::Private),
            TeamMemoryUpdate::default().with_expires_at(fixed_time()),
        ];
        for update in cases {
            assert!(!update.is_empty());
        }
    }
    #[test]
    fn update_validate_rejects_bad_values() {
        assert!(TeamMemoryUpdate::default().content(" ").validate().is_err());
        assert!(TeamMemoryUpdate::default()
            .with_title(" ")
            .validate()
            .is_err());
        assert!(TeamMemoryUpdate::default()
            .with_importance(f32::INFINITY)
            .validate()
            .unwrap_err()
            .to_string()
            .contains("finite"));
        assert!(TeamMemoryUpdate::default()
            .with_tags([""])
            .validate()
            .unwrap_err()
            .to_string()
            .contains("empty entries"));
        assert!(TeamMemoryUpdate::default()
            .with_visibility(TeamVisibility::Members(Vec::new()))
            .validate()
            .is_err());
        assert!(TeamMemoryUpdate::default()
            .with_importance(5.0)
            .validate()
            .is_ok());
    }
    #[test]
    fn serde_round_trip_and_defaults() {
        let input = NewTeamMemory::decision("team", "content")
            .with_member("alice")
            .with_importance(0.7)
            .with_tag("a")
            .with_metadata("k", serde_json::json!(1));
        let json = serde_json::to_value(&input).unwrap();
        assert_eq!(json["id"], serde_json::Value::Null);
        assert_eq!(json["kind"], serde_json::json!("decision"));
        assert_eq!(json["visibility"], serde_json::json!("team"));
        assert_eq!(json["metadata"]["k"], serde_json::json!(1));
        let restored: NewTeamMemory = serde_json::from_value(json).unwrap();
        assert_eq!(restored, input);
        let restored: NewTeamMemory = serde_json::from_value(serde_json::json!({})).unwrap();
        assert_eq!(restored, NewTeamMemory::default());
        let update = TeamMemoryUpdate::default()
            .content("c")
            .with_tags(["a"])
            .with_metadata("k", serde_json::json!(1));
        let json = serde_json::to_value(&update).unwrap();
        assert_eq!(json["content"], serde_json::json!("c"));
        assert_eq!(json["metadata"]["k"], serde_json::json!(1));
        let restored: TeamMemoryUpdate = serde_json::from_value(json).unwrap();
        assert_eq!(restored, update);
        let restored: TeamMemoryUpdate = serde_json::from_value(serde_json::json!({})).unwrap();
        assert_eq!(restored, TeamMemoryUpdate::default());
    }
}
