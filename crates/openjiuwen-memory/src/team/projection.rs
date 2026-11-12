//! Projection of a [`TeamMemoryRecord`] onto the unified
//! [`MemoryRecord`], and back.
//!
//! # This is a projection, not a store
//!
//! A team memory record lives in a [`TeamMemoryStore`](super::store::TeamMemoryStore); the
//! projected [`MemoryRecord`] is a *view* produced so that team memory can flow through the
//! same recall, prompt, and context paths as long-term memory. Nothing here writes to the
//! long-term memory engine, and a projected record is only authoritative once it is fed
//! back through [`TeamMemoryRecord::from_memory_record`]: the team store stays the single
//! source of truth.
//!
//! # Mapping
//!
//! | Team field | `MemoryRecord` field |
//! | --- | --- |
//! | `id`, `content`, `importance`, `source` | copied unchanged |
//! | `created_at` | `timestamp` |
//! | `updated_at` | `updated_at` (`last_access = None`, `access_count = 0`, `embedding = None`) |
//! | `team_id` | `scope.scope_id` |
//! | `member_id` | `scope.user_id` (falling back to `team_id` for team-wide records) |
//! | `session_id` | `scope.session_id` (`agent_id` is always `None`) |
//! | `kind` | [`MemoryType`]: `member_profile` → [`MemoryType::UserProfile`], everything else → [`MemoryType::SemanticMemory`] |
//! | `metadata` | the record's own entries (minus the reserved key) |
//! | everything else | a reserved [`TEAM_MEMORY_METADATA_KEY`] object |
//!
//! The reserved object holds `team_id`, `member_id`, `session_id`, `kind`, `title`, `tags`,
//! `visibility`, and `expires_at`, and is the marker [`TeamMemoryRecord::from_memory_record`]
//! requires: a [`MemoryRecord`] without it is rejected rather than silently treated as a
//! team memory.
//!
//! Round-tripping is lossless for every kind and visibility, including `source`,
//! `expires_at`, and both timestamps.
use super::model::{TeamMemoryKind, TeamMemoryRecord, TeamVisibility};
use crate::record::{MemoryRecord, MemoryScope, MemoryType};
use chrono::{DateTime, Utc};
use openjiuwen_common::error::{Error, Result};
use serde_json::{Map, Value};
use std::collections::BTreeMap;
/// Metadata key holding the team memory scope of a projected record.
pub const TEAM_MEMORY_METADATA_KEY: &str = "team_memory";
impl TeamMemoryRecord {
    /// Projects this team memory onto a [`MemoryRecord`] for unified recall.
    ///
    /// The result is a valid [`MemoryRecord`] (see [`MemoryRecord::validate`]) carrying the
    /// reserved [`TEAM_MEMORY_METADATA_KEY`] object; it is a view, so changing it does not
    /// change the team store. No embedding is computed (`embedding = None`) and no access
    /// is recorded (`last_access = None`, `access_count = 0`).
    #[must_use]
    pub fn to_memory_record(&self) -> MemoryRecord {
        let mut metadata: BTreeMap<String, Value> = self
            .metadata
            .as_object()
            .map(|object| {
                object
                    .iter()
                    .filter(|(key, _)| key.as_str() != TEAM_MEMORY_METADATA_KEY)
                    .map(|(key, value)| (key.clone(), value.clone()))
                    .collect()
            })
            .unwrap_or_default();
        metadata.insert(
            TEAM_MEMORY_METADATA_KEY.to_owned(),
            self.reserved_metadata(),
        );
        MemoryRecord {
            id: self.id.clone(),
            content: self.content.clone(),
            kind: memory_type(&self.kind),
            importance: self.importance,
            timestamp: self.created_at,
            updated_at: self.updated_at,
            last_access: None,
            access_count: 0,
            metadata,
            embedding: None,
            source: self.source.clone(),
            scope: MemoryScope {
                user_id: self
                    .member_id
                    .clone()
                    .unwrap_or_else(|| self.team_id.clone()),
                scope_id: self.team_id.clone(),
                session_id: self.session_id.clone(),
                agent_id: None,
            },
        }
    }
    /// Rebuilds a team memory record from a projected [`MemoryRecord`].
    ///
    /// The record must carry the reserved [`TEAM_MEMORY_METADATA_KEY`] object, otherwise
    /// the error is [`Error::Validation`] saying it is not a projected team memory. Fields
    /// are taken from that object (`team_id` is required and non-empty; `member_id`,
    /// `session_id`, `title`, and `expires_at` may be absent or `null`; `kind` is parsed
    /// tolerantly; `tags` and `visibility` are decoded strictly), while `content`,
    /// `importance`, `source`, `timestamp`, and `updated_at` come from the record itself.
    /// The result is validated before it is returned.
    pub fn from_memory_record(record: &MemoryRecord) -> Result<Self> {
        let reserved = record
            .metadata
            .get(TEAM_MEMORY_METADATA_KEY)
            .ok_or_else(|| {
                Error::validation(format!(
                    "memory record `{}` is not a projected team memory (missing `{TEAM_MEMORY_METADATA_KEY}` metadata)",
                    record.id
                ))
            })?;
        let object = reserved.as_object().ok_or_else(|| {
            Error::validation(format!(
                "memory record `{}` is not a projected team memory (`{TEAM_MEMORY_METADATA_KEY}` metadata is not a JSON object)",
                record.id
            ))
        })?;
        let team_id = required_string(object, "team_id")?;
        let metadata: Map<String, Value> = record
            .metadata
            .iter()
            .filter(|(key, _)| key.as_str() != TEAM_MEMORY_METADATA_KEY)
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect();
        let projected = Self {
            id: record.id.clone(),
            team_id,
            member_id: optional_string(object, "member_id")?,
            session_id: optional_string(object, "session_id")?,
            kind: TeamMemoryKind::parse(
                object
                    .get("kind")
                    .and_then(Value::as_str)
                    .unwrap_or_default(),
            ),
            content: record.content.clone(),
            title: optional_string(object, "title")?,
            importance: record.importance,
            tags: tag_list(object)?,
            metadata: Value::Object(metadata),
            created_at: record.timestamp,
            updated_at: record.updated_at,
            source: record.source.clone(),
            visibility: visibility(object)?,
            expires_at: optional_timestamp(object, "expires_at")?,
        };
        projected.validate()?;
        Ok(projected)
    }
    /// The reserved [`TEAM_MEMORY_METADATA_KEY`] object of this record.
    ///
    /// The `visibility` value is the serde JSON of [`TeamVisibility`] (`"team"`,
    /// `"private"`, or `{"members": [...]}`) and `expires_at` is RFC 3339 or `null`.
    fn reserved_metadata(&self) -> Value {
        let mut object = Map::new();
        object.insert("team_id".to_owned(), Value::String(self.team_id.clone()));
        object.insert(
            "member_id".to_owned(),
            optional_value(self.member_id.as_ref()),
        );
        object.insert(
            "session_id".to_owned(),
            optional_value(self.session_id.as_ref()),
        );
        object.insert(
            "kind".to_owned(),
            Value::String(self.kind.as_str().to_owned()),
        );
        object.insert("title".to_owned(), optional_value(self.title.as_ref()));
        object.insert(
            "tags".to_owned(),
            Value::Array(self.tags.iter().cloned().map(Value::String).collect()),
        );
        object.insert("visibility".to_owned(), visibility_json(&self.visibility));
        object.insert(
            "expires_at".to_owned(),
            self.expires_at.map_or(Value::Null, |expires_at| {
                Value::String(expires_at.to_rfc3339())
            }),
        );
        Value::Object(object)
    }
}
/// The [`MemoryType`] a team memory kind projects onto.
fn memory_type(kind: &TeamMemoryKind) -> MemoryType {
    match kind {
        TeamMemoryKind::MemberProfile => MemoryType::UserProfile,
        TeamMemoryKind::SharedFact
        | TeamMemoryKind::Decision
        | TeamMemoryKind::Task
        | TeamMemoryKind::Artifact
        | TeamMemoryKind::HandoffNote
        | TeamMemoryKind::Custom(_) => MemoryType::SemanticMemory,
    }
}
/// The serde JSON of a [`TeamVisibility`].
///
/// Built by hand (instead of `serde_json::to_value`) so the projection is infallible; the
/// `visibility_json` test pins it to serde's own output.
fn visibility_json(visibility: &TeamVisibility) -> Value {
    match visibility {
        TeamVisibility::Team => Value::String("team".to_owned()),
        TeamVisibility::Private => Value::String("private".to_owned()),
        TeamVisibility::Members(members) => {
            let mut object = Map::new();
            object.insert(
                "members".to_owned(),
                Value::Array(members.iter().cloned().map(Value::String).collect()),
            );
            Value::Object(object)
        }
    }
}
/// JSON `null` for an absent value, the string itself otherwise.
fn optional_value(value: Option<&String>) -> Value {
    value.map_or(Value::Null, |value| Value::String(value.clone()))
}
/// Reads a required, non-empty string field of the reserved object.
fn required_string(object: &Map<String, Value>, key: &str) -> Result<String> {
    match object.get(key) {
        Some(Value::String(value)) if !value.trim().is_empty() => Ok(value.clone()),
        Some(Value::String(_)) => Err(Error::validation(format!(
            "team memory projection `{key}` must not be empty"
        ))),
        Some(_) => Err(Error::validation(format!(
            "team memory projection `{key}` must be a string"
        ))),
        None => Err(Error::validation(format!(
            "team memory projection is missing `{key}`"
        ))),
    }
}
/// Reads an optional string field of the reserved object; `null`/absent is `None`.
fn optional_string(object: &Map<String, Value>, key: &str) -> Result<Option<String>> {
    match object.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(value)) => Ok(Some(value.clone())),
        Some(_) => Err(Error::validation(format!(
            "team memory projection `{key}` must be a string or null"
        ))),
    }
}
/// Reads the `tags` array of the reserved object; `null`/absent is an empty list.
fn tag_list(object: &Map<String, Value>) -> Result<Vec<String>> {
    match object.get("tags") {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(Value::Array(items)) => items
            .iter()
            .map(|item| match item {
                Value::String(tag) => Ok(tag.clone()),
                other => Err(Error::validation(format!(
                    "team memory projection `tags` entries must be strings (got {other})"
                ))),
            })
            .collect(),
        Some(other) => Err(Error::validation(format!(
            "team memory projection `tags` must be an array of strings (got {other})"
        ))),
    }
}
/// Reads the `visibility` field of the reserved object; absent is [`TeamVisibility::Team`].
fn visibility(object: &Map<String, Value>) -> Result<TeamVisibility> {
    match object.get("visibility") {
        None | Some(Value::Null) => Ok(TeamVisibility::default()),
        Some(value) => serde_json::from_value(value.clone()).map_err(|error| {
            Error::validation(format!(
                "team memory projection `visibility` is invalid: {error}"
            ))
        }),
    }
}
/// Reads an optional RFC 3339 timestamp field of the reserved object.
fn optional_timestamp(object: &Map<String, Value>, key: &str) -> Result<Option<DateTime<Utc>>> {
    match object.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(raw)) => DateTime::parse_from_rfc3339(raw)
            .map(|parsed| Some(parsed.with_timezone(&Utc)))
            .map_err(|error| {
                Error::validation(format!(
                    "team memory projection `{key}` is not an RFC 3339 timestamp: {error}"
                ))
            }),
        Some(other) => Err(Error::validation(format!(
            "team memory projection `{key}` must be an RFC 3339 string or null (got {other})"
        ))),
    }
}
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]
mod tests {
    use super::*;
    use crate::team::model::TeamMemoryScope;
    fn at(seconds: i64) -> DateTime<Utc> {
        DateTime::from_timestamp(seconds, 0).unwrap()
    }
    fn sample(kind: TeamMemoryKind, visibility: TeamVisibility) -> TeamMemoryRecord {
        let mut record = TeamMemoryRecord::new(
            &TeamMemoryScope::member("review-team", "alice").with_session("s1"),
            "use sqlite for the shared store",
            kind,
        )
        .with_id("0123456789abcdef01234567")
        .with_title("Storage choice")
        .with_importance(0.8)
        .with_tags(["Storage", "sqlite"])
        .with_metadata("issue", serde_json::json!(42))
        .with_source("leader-extraction")
        .with_visibility(visibility)
        .with_expires_at(at(1_800_000_000));
        record.created_at = at(1_700_000_000);
        record.updated_at = at(1_700_000_050);
        record
    }
    fn all_kinds() -> Vec<TeamMemoryKind> {
        let mut kinds = TeamMemoryKind::BUILTIN.to_vec();
        kinds.push(TeamMemoryKind::Custom("release_check".to_owned()));
        kinds
    }
    fn all_visibilities() -> Vec<TeamVisibility> {
        vec![
            TeamVisibility::Team,
            TeamVisibility::Private,
            TeamVisibility::members(["alice", "bob"]),
        ]
    }
    #[test]
    fn round_trip_is_lossless_for_every_kind_and_visibility() {
        for kind in all_kinds() {
            for visibility in all_visibilities() {
                let record = sample(kind.clone(), visibility.clone());
                let projected = record.to_memory_record();
                assert!(
                    projected.validate().is_ok(),
                    "{kind:?} {visibility:?}: {:?}",
                    projected.validate()
                );
                let restored = TeamMemoryRecord::from_memory_record(&projected).unwrap();
                assert_eq!(restored, record, "{kind:?} {visibility:?}");
            }
        }
    }
    #[test]
    fn round_trip_without_member_session_title_source_or_expiry() {
        let mut record = TeamMemoryRecord::new(
            &TeamMemoryScope::new("review-team"),
            "team-wide fact",
            TeamMemoryKind::SharedFact,
        )
        .with_id("0123456789abcdef01234567");
        record.created_at = at(1_700_000_000);
        record.updated_at = at(1_700_000_000);
        let projected = record.to_memory_record();
        assert_eq!(projected.scope.user_id, "review-team");
        assert_eq!(projected.scope.session_id, None);
        let restored = TeamMemoryRecord::from_memory_record(&projected).unwrap();
        assert_eq!(restored, record);
        let reserved = projected.metadata[TEAM_MEMORY_METADATA_KEY].clone();
        assert_eq!(reserved["member_id"], Value::Null);
        assert_eq!(reserved["session_id"], Value::Null);
        assert_eq!(reserved["title"], Value::Null);
        assert_eq!(reserved["expires_at"], Value::Null);
    }
    #[test]
    fn scope_and_kind_mapping() {
        let record = sample(TeamMemoryKind::MemberProfile, TeamVisibility::Team);
        let projected = record.to_memory_record();
        assert_eq!(projected.scope.user_id, "alice");
        assert_eq!(projected.scope.scope_id, "review-team");
        assert_eq!(projected.scope.session_id.as_deref(), Some("s1"));
        assert_eq!(projected.scope.agent_id, None);
        assert_eq!(projected.kind, MemoryType::UserProfile);
        assert_eq!(projected.id, record.id);
        assert_eq!(projected.content, record.content);
        assert!((projected.importance - 0.8).abs() < f32::EPSILON);
        assert_eq!(projected.timestamp, record.created_at);
        assert_eq!(projected.updated_at, record.updated_at);
        assert_eq!(projected.last_access, None);
        assert_eq!(projected.access_count, 0);
        assert_eq!(projected.embedding, None);
        assert_eq!(projected.source.as_deref(), Some("leader-extraction"));
        for kind in TeamMemoryKind::BUILTIN {
            let expected = if kind == TeamMemoryKind::MemberProfile {
                MemoryType::UserProfile
            } else {
                MemoryType::SemanticMemory
            };
            assert_eq!(
                sample(kind.clone(), TeamVisibility::Team)
                    .to_memory_record()
                    .kind,
                expected,
                "{kind:?}"
            );
        }
        assert_eq!(
            sample(
                TeamMemoryKind::Custom("release_check".to_owned()),
                TeamVisibility::Team
            )
            .to_memory_record()
            .kind,
            MemoryType::SemanticMemory
        );
    }
    #[test]
    fn metadata_is_preserved_and_the_reserved_key_is_not_duplicated() {
        let record = sample(TeamMemoryKind::Decision, TeamVisibility::Team)
            .with_metadata("owner", serde_json::json!({"name": "alice"}));
        let projected = record.to_memory_record();
        assert_eq!(projected.metadata["issue"], serde_json::json!(42));
        assert_eq!(
            projected.metadata["owner"],
            serde_json::json!({"name": "alice"})
        );
        assert!(projected.metadata.contains_key(TEAM_MEMORY_METADATA_KEY));
        let restored = TeamMemoryRecord::from_memory_record(&projected).unwrap();
        assert_eq!(restored.metadata["issue"], serde_json::json!(42));
        assert_eq!(
            restored.metadata["owner"],
            serde_json::json!({"name": "alice"})
        );
        assert!(restored
            .metadata_object()
            .unwrap()
            .get(TEAM_MEMORY_METADATA_KEY)
            .is_none());
        let mut with_reserved = record.clone();
        with_reserved.metadata = serde_json::json!({"team_memory": {"bogus": true}, "keep": 1});
        let projected = with_reserved.to_memory_record();
        assert_eq!(projected.metadata["keep"], serde_json::json!(1));
        let reserved = projected.metadata[TEAM_MEMORY_METADATA_KEY]
            .as_object()
            .unwrap();
        assert_eq!(reserved["team_id"], serde_json::json!("review-team"));
        let restored = TeamMemoryRecord::from_memory_record(&projected).unwrap();
        assert_eq!(restored.metadata, serde_json::json!({"keep": 1}));
    }
    #[test]
    fn reserved_metadata_json_shape() {
        let record = sample(
            TeamMemoryKind::Decision,
            TeamVisibility::members(["alice", "bob"]),
        );
        let projected = record.to_memory_record();
        let reserved = &projected.metadata[TEAM_MEMORY_METADATA_KEY];
        assert_eq!(
            reserved,
            &serde_json::json!({
                "team_id": "review-team",
                "member_id": "alice",
                "session_id": "s1",
                "kind": "decision",
                "title": "Storage choice",
                "tags": ["storage", "sqlite"],
                "visibility": {"members": ["alice", "bob"]},
                "expires_at": "2027-01-15T08:00:00+00:00",
            })
        );
        let json = serde_json::to_value(&projected).unwrap();
        assert_eq!(
            json["fields"][TEAM_MEMORY_METADATA_KEY]["kind"],
            serde_json::json!("decision")
        );
        assert_eq!(json["type"], serde_json::json!("semantic_memory"));
        assert_eq!(json["scope_id"], serde_json::json!("review-team"));
        assert_eq!(json["user_id"], serde_json::json!("alice"));
    }
    #[test]
    fn visibility_json_matches_serde() {
        for visibility in all_visibilities() {
            assert_eq!(
                visibility_json(&visibility),
                serde_json::to_value(&visibility).unwrap(),
                "{visibility:?}"
            );
        }
    }
    #[test]
    fn plain_memory_records_are_rejected() {
        let record = MemoryRecord::new(
            MemoryScope::new("alice", "review-team"),
            "content",
            MemoryType::SemanticMemory,
        );
        let error = TeamMemoryRecord::from_memory_record(&record).unwrap_err();
        assert!(matches!(error, Error::Validation(_)));
        assert!(error.to_string().contains("is not a projected team memory"));
        let mut wrong_type = record.clone();
        wrong_type.metadata.insert(
            TEAM_MEMORY_METADATA_KEY.to_owned(),
            serde_json::json!("not-an-object"),
        );
        let error = TeamMemoryRecord::from_memory_record(&wrong_type).unwrap_err();
        assert!(error.to_string().contains("is not a JSON object"));
    }
    /// One mutation applied to the reserved metadata object.
    type ReservedMutation = Box<dyn Fn(&mut Map<String, Value>)>;
    #[test]
    fn malformed_reserved_metadata_is_rejected() {
        let projected = sample(TeamMemoryKind::Decision, TeamVisibility::Team).to_memory_record();
        let with_reserved = |mutate: &dyn Fn(&mut Map<String, Value>)| {
            let mut record = projected.clone();
            let object = record
                .metadata
                .get_mut(TEAM_MEMORY_METADATA_KEY)
                .unwrap()
                .as_object_mut()
                .unwrap();
            mutate(object);
            record
        };
        let cases: Vec<ReservedMutation> = vec![
            Box::new(|object| {
                object.remove("team_id");
            }),
            Box::new(|object| {
                object.insert("team_id".to_owned(), Value::String(String::new()));
            }),
            Box::new(|object| {
                object.insert("team_id".to_owned(), Value::from(7));
            }),
            Box::new(|object| {
                object.insert("member_id".to_owned(), Value::from(7));
            }),
            Box::new(|object| {
                object.insert("tags".to_owned(), Value::String("storage".to_owned()));
            }),
            Box::new(|object| {
                object.insert("tags".to_owned(), Value::Array(vec![Value::from(1)]));
            }),
            Box::new(|object| {
                object.insert("visibility".to_owned(), Value::String("shared".to_owned()));
            }),
            Box::new(|object| {
                object.insert(
                    "expires_at".to_owned(),
                    Value::String("next tuesday".to_owned()),
                );
            }),
            Box::new(|object| {
                object.insert("expires_at".to_owned(), Value::from(7));
            }),
        ];
        for mutate in cases {
            let record = with_reserved(mutate.as_ref());
            let error = TeamMemoryRecord::from_memory_record(&record).unwrap_err();
            assert!(matches!(error, Error::Validation(_)), "{error}");
        }
        let mut bad_kind = projected.clone();
        bad_kind.metadata.get_mut(TEAM_MEMORY_METADATA_KEY).unwrap()["kind"] = Value::from(7);
        let restored = TeamMemoryRecord::from_memory_record(&bad_kind).unwrap();
        assert_eq!(restored.kind, TeamMemoryKind::SharedFact);
    }
    #[test]
    fn missing_optional_reserved_fields_fall_back_to_defaults() {
        let projected = sample(TeamMemoryKind::Decision, TeamVisibility::Team).to_memory_record();
        let mut record = projected.clone();
        let object = record
            .metadata
            .get_mut(TEAM_MEMORY_METADATA_KEY)
            .unwrap()
            .as_object_mut()
            .unwrap();
        object.remove("tags");
        object.remove("visibility");
        object.remove("kind");
        object.remove("expires_at");
        object.remove("member_id");
        object.remove("session_id");
        object.remove("title");
        let restored = TeamMemoryRecord::from_memory_record(&record).unwrap();
        assert_eq!(restored.kind, TeamMemoryKind::SharedFact);
        assert_eq!(restored.visibility, TeamVisibility::Team);
        assert!(restored.tags.is_empty());
        assert_eq!(restored.expires_at, None);
        assert_eq!(restored.member_id, None);
        assert_eq!(restored.session_id, None);
        assert_eq!(restored.title, None);
        assert_eq!(restored.team_id, "review-team");
    }
}
