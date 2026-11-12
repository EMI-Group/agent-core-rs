//! [`TeamMemoryFilter`]: the predicate used to list, clear, and count team memories.
//!
//! A filter is an AND of the fields that are set; empty `kinds`/`tags` mean "any" and
//! `team_id` is the only required field. Expired records are hidden unless
//! `include_expired` is set.
use super::kind::{TeamMemoryKind, TeamVisibility};
use super::record::TeamMemoryRecord;
use super::validate::{contains_all_tags, validate_component};
use super::MAX_SCOPE_ID_CHARS;
use chrono::{DateTime, Utc};
use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};
/// A conjunction of team memory record predicates.
///
/// [`TeamMemoryFilter::for_team`] builds the common "everything in this team" filter;
/// builder methods add narrowings. Time bounds are exclusive, `min_importance` is
/// inclusive, and `text_contains` is a case-insensitive substring matched against the
/// content or the title.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct TeamMemoryFilter {
    /// Required owning team id.
    pub team_id: String,
    /// Required author member id (team-wide records do not match).
    pub member_id: Option<String>,
    /// Required session id (records without a session do not match).
    pub session_id: Option<String>,
    /// Accepted kinds; empty means every kind.
    pub kinds: Vec<TeamMemoryKind>,
    /// Tags that must all be present (case-insensitive); empty means no constraint.
    pub tags: Vec<String>,
    /// Required visibility (exact match).
    pub visibility: Option<TeamVisibility>,
    /// Case-insensitive substring that must appear in the content or the title.
    pub text_contains: Option<String>,
    /// Minimum inclusive importance.
    pub min_importance: Option<f32>,
    /// Exclusive lower bound on `created_at`.
    pub created_after: Option<DateTime<Utc>>,
    /// Exclusive upper bound on `created_at`.
    pub created_before: Option<DateTime<Utc>>,
    /// Exclusive lower bound on `updated_at`.
    pub updated_after: Option<DateTime<Utc>>,
    /// Exclusive upper bound on `updated_at`.
    pub updated_before: Option<DateTime<Utc>>,
    /// Whether expired records should be returned; `false` hides them.
    pub include_expired: bool,
    /// Optional maximum number of records to return.
    pub limit: Option<usize>,
    /// Optional number of records to skip.
    pub offset: Option<usize>,
}
impl TeamMemoryFilter {
    /// Builds a filter that matches every record in `team_id`.
    #[must_use]
    pub fn for_team(team_id: impl Into<String>) -> Self {
        Self {
            team_id: team_id.into(),
            ..Self::default()
        }
    }
    /// Accepts one more kind.
    #[must_use]
    pub fn kind(mut self, kind: TeamMemoryKind) -> Self {
        self.kinds.push(kind);
        self
    }
    /// Restricts the filter to one author member.
    #[must_use]
    pub fn member(mut self, member_id: impl Into<String>) -> Self {
        self.member_id = Some(member_id.into());
        self
    }
    /// Caps the number of returned records.
    #[must_use]
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }
    /// Skips `offset` matching records.
    #[must_use]
    pub fn with_offset(mut self, offset: usize) -> Self {
        self.offset = Some(offset);
        self
    }
    /// Validates the filter: `team_id` follows the component rules and `limit` may not
    /// be zero.
    pub fn validate(&self) -> Result<()> {
        validate_component("team_id", &self.team_id, MAX_SCOPE_ID_CHARS)?;
        if self.limit == Some(0) {
            return Err(Error::validation(
                "team memory `filter.limit` must be greater than zero",
            ));
        }
        Ok(())
    }
    /// Returns `true` when `record` satisfies every set clause.
    ///
    /// The team must match; member/session/kind/tag/visibility constraints apply when
    /// set; `text_contains` looks at content and title; time bounds are exclusive; and
    /// expired records are hidden unless `include_expired` is set. `limit`/`offset` are
    /// applied by the caller, not here.
    #[must_use]
    pub fn matches(&self, record: &TeamMemoryRecord, now: DateTime<Utc>) -> bool {
        if record.team_id != self.team_id {
            return false;
        }
        if let Some(member) = &self.member_id {
            if record.member_id.as_ref() != Some(member) {
                return false;
            }
        }
        if let Some(session) = &self.session_id {
            if record.session_id.as_ref() != Some(session) {
                return false;
            }
        }
        if !self.kinds.is_empty() && !self.kinds.contains(&record.kind) {
            return false;
        }
        if !self.tags.is_empty() && !contains_all_tags(&record.tags, &self.tags) {
            return false;
        }
        if let Some(visibility) = &self.visibility {
            if &record.visibility != visibility {
                return false;
            }
        }
        if let Some(needle) = &self.text_contains {
            let needle = needle.to_lowercase();
            let in_content = record.content.to_lowercase().contains(&needle);
            let in_title = record
                .title
                .as_ref()
                .is_some_and(|title| title.to_lowercase().contains(&needle));
            if !in_content && !in_title {
                return false;
            }
        }
        if let Some(min) = self.min_importance {
            if record.importance < min {
                return false;
            }
        }
        if let Some(after) = self.created_after {
            if record.created_at <= after {
                return false;
            }
        }
        if let Some(before) = self.created_before {
            if record.created_at >= before {
                return false;
            }
        }
        if let Some(after) = self.updated_after {
            if record.updated_at <= after {
                return false;
            }
        }
        if let Some(before) = self.updated_before {
            if record.updated_at >= before {
                return false;
            }
        }
        if !self.include_expired && record.is_expired(now) {
            return false;
        }
        true
    }
}
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::team::model::TeamMemoryScope;
    fn at(seconds: i64) -> DateTime<Utc> {
        DateTime::from_timestamp(seconds, 0).unwrap()
    }
    fn record() -> TeamMemoryRecord {
        let mut record = TeamMemoryRecord::new(
            &TeamMemoryScope::member("review-team", "alice").with_session("s1"),
            "Use SQLite for the shared store",
            TeamMemoryKind::Decision,
        )
        .with_id("0123456789abcdef01234567")
        .with_title("Storage choice")
        .with_importance(0.8)
        .with_tags(["Storage", "sqlite"])
        .with_visibility(TeamVisibility::Private);
        record.created_at = at(1_700_000_000);
        record.updated_at = at(1_700_000_050);
        record
    }
    #[test]
    fn for_team_matches_every_record_of_the_team() {
        let filter = TeamMemoryFilter::for_team("review-team");
        assert_eq!(filter.kinds, Vec::new());
        assert_eq!(filter.limit, None);
        assert!(!filter.include_expired);
        assert!(filter.matches(&record(), at(1_700_000_100)));
        let filter = TeamMemoryFilter::for_team("other-team");
        assert!(!filter.matches(&record(), at(1_700_000_100)));
    }
    #[test]
    fn member_and_session_require_an_exact_match() {
        let filter = TeamMemoryFilter::for_team("review-team").member("alice");
        assert!(filter.matches(&record(), at(1_700_000_100)));
        assert!(!TeamMemoryFilter::for_team("review-team")
            .member("bob")
            .matches(&record(), at(1_700_000_100)));
        let mut filter = TeamMemoryFilter::for_team("review-team");
        filter.session_id = Some("s1".to_owned());
        assert!(filter.matches(&record(), at(1_700_000_100)));
        filter.session_id = Some("s2".to_owned());
        assert!(!filter.matches(&record(), at(1_700_000_100)));
    }
    #[test]
    fn kind_accepts_any_of_the_listed_kinds() {
        let filter = TeamMemoryFilter::for_team("review-team").kind(TeamMemoryKind::Task);
        assert!(!filter.matches(&record(), at(1_700_000_100)));
        let filter = filter.kind(TeamMemoryKind::Decision);
        assert!(filter.matches(&record(), at(1_700_000_100)));
    }
    #[test]
    fn tags_must_all_be_present_case_insensitively() {
        let mut filter = TeamMemoryFilter::for_team("review-team");
        filter.tags = vec!["STORAGE".to_owned()];
        assert!(filter.matches(&record(), at(1_700_000_100)));
        filter.tags = vec!["storage".to_owned(), "sqlite".to_owned()];
        assert!(filter.matches(&record(), at(1_700_000_100)));
        filter.tags.push("redis".to_owned());
        assert!(!filter.matches(&record(), at(1_700_000_100)));
    }
    #[test]
    fn visibility_must_match_exactly() {
        let mut filter = TeamMemoryFilter::for_team("review-team");
        filter.visibility = Some(TeamVisibility::Private);
        assert!(filter.matches(&record(), at(1_700_000_100)));
        filter.visibility = Some(TeamVisibility::Team);
        assert!(!filter.matches(&record(), at(1_700_000_100)));
        filter.visibility = Some(TeamVisibility::members(["alice"]));
        assert!(!filter.matches(&record(), at(1_700_000_100)));
    }
    #[test]
    fn text_contains_matches_content_or_title() {
        let mut filter = TeamMemoryFilter::for_team("review-team");
        filter.text_contains = Some("sqlite".to_owned());
        assert!(filter.matches(&record(), at(1_700_000_100)));
        filter.text_contains = Some("STORAGE CHOICE".to_owned());
        assert!(filter.matches(&record(), at(1_700_000_100)));
        filter.text_contains = Some("redis".to_owned());
        assert!(!filter.matches(&record(), at(1_700_000_100)));
    }
    #[test]
    fn min_importance_is_inclusive() {
        let mut filter = TeamMemoryFilter::for_team("review-team");
        filter.min_importance = Some(0.8);
        assert!(filter.matches(&record(), at(1_700_000_100)));
        filter.min_importance = Some(0.81);
        assert!(!filter.matches(&record(), at(1_700_000_100)));
    }
    #[test]
    fn time_bounds_are_exclusive() {
        let created = at(1_700_000_000);
        let updated = at(1_700_000_050);
        let mut filter = TeamMemoryFilter::for_team("review-team");
        filter.created_after = Some(at(1_699_999_999));
        assert!(filter.matches(&record(), at(1_700_000_100)));
        filter.created_after = Some(created);
        assert!(!filter.matches(&record(), at(1_700_000_100)));
        let mut filter = TeamMemoryFilter::for_team("review-team");
        filter.created_before = Some(at(1_700_000_001));
        assert!(filter.matches(&record(), at(1_700_000_100)));
        filter.created_before = Some(created);
        assert!(!filter.matches(&record(), at(1_700_000_100)));
        let mut filter = TeamMemoryFilter::for_team("review-team");
        filter.updated_after = Some(at(1_700_000_049));
        assert!(filter.matches(&record(), at(1_700_000_200)));
        filter.updated_after = Some(updated);
        assert!(!filter.matches(&record(), at(1_700_000_200)));
        let mut filter = TeamMemoryFilter::for_team("review-team");
        filter.updated_before = Some(at(1_700_000_051));
        assert!(filter.matches(&record(), at(1_700_000_100)));
        filter.updated_before = Some(updated);
        assert!(!filter.matches(&record(), at(1_700_000_100)));
    }
    #[test]
    fn expired_records_are_hidden_unless_requested() {
        let record = record().with_expires_at(at(1_700_000_100));
        let filter = TeamMemoryFilter::for_team("review-team");
        assert!(!filter.matches(&record, at(1_700_000_100)));
        assert!(!filter.matches(&record, at(1_700_000_200)));
        assert!(filter.matches(&record, at(1_699_999_999)));
        let mut filter = filter;
        filter.include_expired = true;
        assert!(filter.matches(&record, at(1_700_000_200)));
    }
    #[test]
    fn validate_checks_team_and_limit() {
        assert!(TeamMemoryFilter::for_team("review-team").validate().is_ok());
        assert!(TeamMemoryFilter::default().validate().is_err());
        assert!(TeamMemoryFilter::for_team("a/b")
            .validate()
            .unwrap_err()
            .to_string()
            .contains("team_id"));
        assert!(TeamMemoryFilter::for_team("review-team")
            .with_limit(0)
            .validate()
            .unwrap_err()
            .to_string()
            .contains("limit"));
        assert!(TeamMemoryFilter::for_team("review-team")
            .with_limit(10)
            .validate()
            .is_ok());
    }
    #[test]
    fn serde_round_trip_and_defaults() {
        let filter = TeamMemoryFilter::for_team("review-team")
            .member("alice")
            .kind(TeamMemoryKind::Decision)
            .with_limit(5)
            .with_offset(2);
        let json = serde_json::to_value(&filter).unwrap();
        assert_eq!(json["team_id"], serde_json::json!("review-team"));
        assert_eq!(json["kinds"], serde_json::json!(["decision"]));
        assert_eq!(json["limit"], serde_json::json!(5));
        assert_eq!(json["include_expired"], serde_json::json!(false));
        let restored: TeamMemoryFilter = serde_json::from_value(json).unwrap();
        assert_eq!(restored, filter);
        let restored: TeamMemoryFilter = serde_json::from_value(serde_json::json!({})).unwrap();
        assert_eq!(restored, TeamMemoryFilter::default());
    }
}
