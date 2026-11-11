//! [`MemoryFilter`]: the predicate used by list, clear, and stats.
//!
//! A filter is an AND of the fields that are set; empty `kinds`/`ids` vectors mean
//! "any kind"/"any id" so a default filter matches every record.
use super::kind::MemoryType;
use super::record::MemoryRecord;
use super::scope::MemoryScope;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
/// A conjunction of record predicates.
///
/// `for_scope` builds the common "everything in this scope" filter; builder methods add
/// narrowings. Scope fields set on the filter must equal the record's fields, while
/// unset fields impose no constraint.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct MemoryFilter {
    /// Required owner user id.
    pub user_id: Option<String>,
    /// Required owner scope id.
    pub scope_id: Option<String>,
    /// Required session id (records without a session do not match).
    pub session_id: Option<String>,
    /// Required agent id (records without an agent do not match).
    pub agent_id: Option<String>,
    /// Accepted kinds; empty means every kind.
    pub kinds: Vec<MemoryType>,
    /// Accepted record ids; empty means every id.
    pub ids: Vec<String>,
    /// Case-insensitive substring that must appear in the content.
    pub text_contains: Option<String>,
    /// Minimum inclusive importance.
    pub min_importance: Option<f32>,
    /// Exclusive lower bound on `timestamp`.
    pub created_after: Option<DateTime<Utc>>,
    /// Exclusive upper bound on `timestamp`.
    pub created_before: Option<DateTime<Utc>>,
}
impl MemoryFilter {
    /// Builds an empty filter that matches every record.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
    /// Builds a filter that matches everything inside `scope`.
    ///
    /// All four scope fields are copied, so a session- or agent-bound scope only
    /// selects records bound to the same session or agent.
    #[must_use]
    pub fn for_scope(scope: &MemoryScope) -> Self {
        Self {
            user_id: Some(scope.user_id.clone()),
            scope_id: Some(scope.scope_id.clone()),
            session_id: scope.session_id.clone(),
            agent_id: scope.agent_id.clone(),
            ..Self::default()
        }
    }
    /// Returns the filter with one more accepted kind.
    #[must_use]
    pub fn kind(mut self, kind: MemoryType) -> Self {
        self.kinds.push(kind);
        self
    }
    /// Returns `true` when `record` satisfies every set clause.
    #[must_use]
    pub fn matches(&self, record: &MemoryRecord) -> bool {
        if let Some(user_id) = &self.user_id {
            if &record.scope.user_id != user_id {
                return false;
            }
        }
        if let Some(scope_id) = &self.scope_id {
            if &record.scope.scope_id != scope_id {
                return false;
            }
        }
        if let Some(session_id) = &self.session_id {
            if record.scope.session_id.as_ref() != Some(session_id) {
                return false;
            }
        }
        if let Some(agent_id) = &self.agent_id {
            if record.scope.agent_id.as_ref() != Some(agent_id) {
                return false;
            }
        }
        if !self.kinds.is_empty() && !self.kinds.contains(&record.kind) {
            return false;
        }
        if !self.ids.is_empty() && !self.ids.contains(&record.id) {
            return false;
        }
        if let Some(needle) = &self.text_contains {
            if !record
                .content
                .to_lowercase()
                .contains(&needle.to_lowercase())
            {
                return false;
            }
        }
        if let Some(min_importance) = self.min_importance {
            if record.importance < min_importance {
                return false;
            }
        }
        if let Some(created_after) = self.created_after {
            if record.timestamp <= created_after {
                return false;
            }
        }
        if let Some(created_before) = self.created_before {
            if record.timestamp >= created_before {
                return false;
            }
        }
        true
    }
    /// Returns `true` when no clause is set, i.e. the filter matches everything.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.user_id.is_none()
            && self.scope_id.is_none()
            && self.session_id.is_none()
            && self.agent_id.is_none()
            && self.kinds.is_empty()
            && self.ids.is_empty()
            && self.text_contains.is_none()
            && self.min_importance.is_none()
            && self.created_after.is_none()
            && self.created_before.is_none()
    }
    /// Returns the `(user_id, scope_id)` prefix when both ids are set.
    ///
    /// The prefix enables a targeted storage scan instead of a full-collection listing.
    #[must_use]
    pub fn scope_prefix(&self) -> Option<(&str, &str)> {
        match (&self.user_id, &self.scope_id) {
            (Some(user_id), Some(scope_id)) => Some((user_id.as_str(), scope_id.as_str())),
            _ => None,
        }
    }
}
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::record::MemoryRecord;
    fn record() -> MemoryRecord {
        MemoryRecord::new(
            MemoryScope::new("alice", "ws").session("s1").agent("a1"),
            "Alice Prefers Dark Mode",
            MemoryType::SemanticMemory,
        )
        .with_id("id-1")
        .with_importance(0.6)
        .with_timestamp(DateTime::from_timestamp(1_700_000_000, 0).unwrap())
    }
    #[test]
    fn empty_filter_matches_everything() {
        let filter = MemoryFilter::new();
        assert!(filter.is_empty());
        assert!(filter.matches(&record()));
        assert_eq!(filter.scope_prefix(), None);
        assert_eq!(filter, MemoryFilter::default());
    }
    #[test]
    fn for_scope_copies_all_scope_fields() {
        let scope = MemoryScope::new("alice", "ws").session("s1").agent("a1");
        let filter = MemoryFilter::for_scope(&scope);
        assert_eq!(filter.user_id.as_deref(), Some("alice"));
        assert_eq!(filter.scope_id.as_deref(), Some("ws"));
        assert_eq!(filter.session_id.as_deref(), Some("s1"));
        assert_eq!(filter.agent_id.as_deref(), Some("a1"));
        assert!(!filter.is_empty());
        assert_eq!(filter.scope_prefix(), Some(("alice", "ws")));
        assert!(filter.matches(&record()));
    }
    #[test]
    fn scope_mismatch_fails() {
        assert!(!MemoryFilter::for_scope(&MemoryScope::new("bob", "ws")).matches(&record()));
        assert!(!MemoryFilter::for_scope(&MemoryScope::new("alice", "other")).matches(&record()));
        assert!(
            !MemoryFilter::for_scope(&MemoryScope::new("alice", "ws").session("s2"))
                .matches(&record())
        );
        assert!(
            !MemoryFilter::for_scope(&MemoryScope::new("alice", "ws").agent("a2"))
                .matches(&record())
        );
    }
    #[test]
    fn kinds_and_ids_clauses() {
        let filter = MemoryFilter::new().kind(MemoryType::SemanticMemory);
        assert!(filter.matches(&record()));
        let filter = MemoryFilter::new().kind(MemoryType::EpisodicMemory);
        assert!(!filter.matches(&record()));
        let filter = MemoryFilter::default()
            .kind(MemoryType::EpisodicMemory)
            .kind(MemoryType::SemanticMemory);
        assert!(filter.matches(&record()));
        let mut filter = MemoryFilter::new();
        filter.ids = vec!["id-1".to_owned()];
        assert!(filter.matches(&record()));
        filter.ids = vec!["id-2".to_owned()];
        assert!(!filter.matches(&record()));
    }
    #[test]
    fn text_contains_is_case_insensitive_substring() {
        let mut filter = MemoryFilter::new();
        filter.text_contains = Some("prefers dark".to_owned());
        assert!(filter.matches(&record()));
        filter.text_contains = Some("PREFERS DARK".to_owned());
        assert!(filter.matches(&record()));
        filter.text_contains = Some("light".to_owned());
        assert!(!filter.matches(&record()));
    }
    #[test]
    fn importance_and_time_bounds() {
        let mut filter = MemoryFilter::new();
        filter.min_importance = Some(0.5);
        assert!(filter.matches(&record()));
        filter.min_importance = Some(0.7);
        assert!(!filter.matches(&record()));
        let timestamp = record().timestamp;
        let mut filter = MemoryFilter::new();
        filter.created_after = Some(timestamp - chrono::Duration::seconds(1));
        assert!(filter.matches(&record()));
        filter.created_after = Some(timestamp);
        assert!(!filter.matches(&record()));
        let mut filter = MemoryFilter::new();
        filter.created_before = Some(timestamp + chrono::Duration::seconds(1));
        assert!(filter.matches(&record()));
        filter.created_before = Some(timestamp);
        assert!(!filter.matches(&record()));
    }
    #[test]
    fn scope_prefix_requires_both_ids() {
        let mut filter = MemoryFilter::new();
        filter.user_id = Some("alice".to_owned());
        assert_eq!(filter.scope_prefix(), None);
        filter.scope_id = Some("ws".to_owned());
        assert_eq!(filter.scope_prefix(), Some(("alice", "ws")));
    }
    #[test]
    fn serde_round_trip() {
        let filter =
            MemoryFilter::for_scope(&MemoryScope::new("alice", "ws")).kind(MemoryType::Coding);
        let json = serde_json::to_value(&filter).unwrap();
        assert_eq!(json["kinds"], serde_json::json!(["coding"]));
        let restored: MemoryFilter = serde_json::from_value(json).unwrap();
        assert_eq!(restored, filter);
        let restored: MemoryFilter = serde_json::from_value(serde_json::json!({})).unwrap();
        assert!(restored.is_empty());
    }
}
