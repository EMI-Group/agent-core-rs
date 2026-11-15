//! Result types of the team memory read path.
//!
//! [`TeamMemoryStats`] aggregates a filtered record set and [`TeamSearchHit`] carries
//! one scored record. Both are plain `serde` values so callers can log, persist, or
//! forward them.
use super::record::TeamMemoryRecord;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
/// Aggregate statistics of a filtered team memory record set.
///
/// `by_member` uses [`super::TEAM_WIDE_MEMBER_KEY`] for records without a member, and
/// `oldest`/`newest` are the creation times of the oldest and newest matching records.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct TeamMemoryStats {
    /// Number of matching records.
    pub total: usize,
    /// Record count per kind, keyed by [`super::TeamMemoryKind::as_str`].
    pub by_kind: BTreeMap<String, usize>,
    /// Record count per author member, keyed by member id or
    /// [`super::TEAM_WIDE_MEMBER_KEY`].
    pub by_member: BTreeMap<String, usize>,
    /// Number of matching records that have expired.
    pub expired: usize,
    /// Creation time of the oldest matching record.
    pub oldest: Option<DateTime<Utc>>,
    /// Creation time of the newest matching record.
    pub newest: Option<DateTime<Utc>>,
}
/// One scored team memory search result.
///
/// The record is flattened onto the top level (so a hit has the same JSON fields as a
/// record plus the score components), matching `RecallHit` in [`crate::record`].
/// `score` is the blended score the caller filtered on; `similarity`, `keyword`,
/// `importance`, and `recency` are the exact `0.0..=1.0` components it was computed from.
/// `similarity` stays `0.0` whenever the search had no query embedding (or the record has
/// none), which is the "no semantic signal" value.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TeamSearchHit {
    /// The recalled record, flattened into this object.
    #[serde(flatten)]
    pub record: TeamMemoryRecord,
    /// Final blended score in `0.0..=1.0`.
    pub score: f32,
    /// Normalized semantic similarity component in `0.0..=1.0`.
    #[serde(default)]
    pub similarity: f32,
    /// Normalized keyword component.
    pub keyword: f32,
    /// The record's importance component.
    pub importance: f32,
    /// Exponential recency component.
    pub recency: f32,
}
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::team::model::{TeamMemoryKind, TeamMemoryScope, TEAM_WIDE_MEMBER_KEY};
    fn hit(score: f32) -> TeamSearchHit {
        let mut record = TeamMemoryRecord::new(
            &TeamMemoryScope::new("review-team"),
            "some content",
            TeamMemoryKind::SharedFact,
        )
        .with_id("0123456789abcdef01234567");
        record.created_at = DateTime::from_timestamp(1_700_000_000, 0).unwrap();
        record.updated_at = record.created_at;
        TeamSearchHit {
            record,
            score,
            similarity: 0.0,
            keyword: 0.4,
            importance: 0.5,
            recency: 0.9,
        }
    }
    #[test]
    fn stats_defaults_and_round_trip() {
        let empty = TeamMemoryStats::default();
        assert_eq!(empty.total, 0);
        assert!(empty.by_kind.is_empty());
        assert!(empty.by_member.is_empty());
        assert_eq!(empty.expired, 0);
        assert!(empty.oldest.is_none());
        assert!(empty.newest.is_none());
        let stats = TeamMemoryStats {
            total: 3,
            by_kind: BTreeMap::from([("decision".to_owned(), 2), ("task".to_owned(), 1)]),
            by_member: BTreeMap::from([
                ("alice".to_owned(), 2),
                (TEAM_WIDE_MEMBER_KEY.to_owned(), 1),
            ]),
            expired: 1,
            oldest: Some(DateTime::from_timestamp(1_700_000_000, 0).unwrap()),
            newest: Some(DateTime::from_timestamp(1_700_000_100, 0).unwrap()),
        };
        let json = serde_json::to_value(&stats).unwrap();
        assert_eq!(json["total"], serde_json::json!(3));
        assert_eq!(json["by_kind"]["decision"], serde_json::json!(2));
        assert_eq!(
            json["by_member"][TEAM_WIDE_MEMBER_KEY],
            serde_json::json!(1)
        );
        assert_eq!(json["expired"], serde_json::json!(1));
        let restored: TeamMemoryStats = serde_json::from_value(json).unwrap();
        assert_eq!(restored, stats);
    }
    #[test]
    fn search_hit_flattens_the_record() {
        let hit = hit(0.75);
        let json = serde_json::to_value(&hit).unwrap();
        assert_eq!(json["id"], serde_json::json!("0123456789abcdef01234567"));
        assert_eq!(json["team_id"], serde_json::json!("review-team"));
        assert_eq!(json["content"], serde_json::json!("some content"));
        assert_eq!(json["kind"], serde_json::json!("shared_fact"));
        assert_eq!(json["score"], serde_json::json!(0.75));
        assert_eq!(json["similarity"], serde_json::json!(0.0_f32));
        assert_eq!(json["keyword"], serde_json::json!(0.4_f32));
        assert_eq!(json["importance"], serde_json::json!(0.5_f32));
        assert_eq!(json["recency"], serde_json::json!(0.9_f32));
        let restored: TeamSearchHit = serde_json::from_value(json).unwrap();
        assert_eq!(restored, hit);
    }
}
