//! [`TeamMemoryScope`]: the team / member / session tuple that owns a team memory and
//! doubles as the read view used to query it.
//!
//! Mapping to the Python `openjiuwen.core.memory` team paths: `team_id` corresponds to
//! the Python team scope id (the team name), `member_id` to a member's user id, and
//! `session_id` to the coordination session that produced the memory. A scope with no
//! `member_id` is the team-wide (leader) view; a scope with one is that member's view.
use super::kind::TeamVisibility;
use super::record::TeamMemoryRecord;
use super::validate::validate_component;
use super::MAX_SCOPE_ID_CHARS;
use openjiuwen_common::error::Result;
use serde::{Deserialize, Serialize};
/// The ownership tuple of a team memory record, used both as a write target and as a
/// query view.
///
/// `team_id` is required; `member_id`/`session_id` narrow the scope. JSON is flat and
/// `snake_case`; absent optional ids are omitted and a missing `team_id` yields an
/// invalid-but-constructible empty scope that [`TeamMemoryScope::validate`] rejects.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct TeamMemoryScope {
    /// The team that owns the memory (Python team scope id / team name).
    pub team_id: String,
    /// The member the memory belongs to, when member-scoped; absent is team-wide.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub member_id: Option<String>,
    /// The coordination session the memory belongs to, when session-scoped.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
}
impl TeamMemoryScope {
    /// Builds a team-wide scope (every member, every visibility).
    #[must_use]
    pub fn new(team_id: impl Into<String>) -> Self {
        Self {
            team_id: team_id.into(),
            member_id: None,
            session_id: None,
        }
    }
    /// Builds the scope of one member inside a team.
    #[must_use]
    pub fn member(team_id: impl Into<String>, member_id: impl Into<String>) -> Self {
        Self {
            team_id: team_id.into(),
            member_id: Some(member_id.into()),
            session_id: None,
        }
    }
    /// Returns the same scope bound to `member_id`.
    #[must_use]
    pub fn with_member(mut self, member_id: impl Into<String>) -> Self {
        self.member_id = Some(member_id.into());
        self
    }
    /// Returns the same scope bound to `session_id`.
    #[must_use]
    pub fn with_session(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }
    /// Validates `team_id` and any present `member_id`/`session_id` with the shared
    /// component rules (non-blank, no NUL/`/`/`\`, no control characters, at most 128
    /// characters).
    pub fn validate(&self) -> Result<()> {
        validate_component("team_id", &self.team_id, MAX_SCOPE_ID_CHARS)?;
        if let Some(member_id) = &self.member_id {
            validate_component("member_id", member_id, MAX_SCOPE_ID_CHARS)?;
        }
        if let Some(session_id) = &self.session_id {
            validate_component("session_id", session_id, MAX_SCOPE_ID_CHARS)?;
        }
        Ok(())
    }
    /// Returns `true` when this scope is allowed to read `record`.
    ///
    /// The team must always match. When the scope names a session, a record from a
    /// different session is rejected while session-less records still pass. A
    /// team-wide scope (`member_id` absent) is the leader view and allows every
    /// visibility; a member scope allows [`TeamVisibility::Team`], its own
    /// [`TeamVisibility::Private`] records, and [`TeamVisibility::Members`] lists that
    /// name it (or whose author is the member).
    #[must_use]
    pub fn allows(&self, record: &TeamMemoryRecord) -> bool {
        if record.team_id != self.team_id {
            return false;
        }
        if let Some(session_id) = &self.session_id {
            if let Some(record_session) = &record.session_id {
                if record_session != session_id {
                    return false;
                }
            }
        }
        let Some(member) = self.member_id.as_deref() else {
            return true;
        };
        match &record.visibility {
            TeamVisibility::Team => true,
            TeamVisibility::Private => record.member_id.as_deref() == Some(member),
            TeamVisibility::Members(members) => {
                record.member_id.as_deref() == Some(member)
                    || members.iter().any(|candidate| candidate == member)
            }
        }
    }
}
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::team::model::TeamMemoryKind;
    fn record(team: &str, member: Option<&str>, visibility: TeamVisibility) -> TeamMemoryRecord {
        let mut record = TeamMemoryRecord::new(
            &TeamMemoryScope::new(team),
            "some content",
            TeamMemoryKind::SharedFact,
        );
        record.member_id = member.map(str::to_owned);
        record.visibility = visibility;
        record
    }
    #[test]
    fn constructors_and_builders() {
        let team = TeamMemoryScope::new("review-team");
        assert_eq!(team.team_id, "review-team");
        assert_eq!(team.member_id, None);
        assert_eq!(team.session_id, None);
        let member = TeamMemoryScope::member("review-team", "alice").with_session("s1");
        assert_eq!(member.member_id.as_deref(), Some("alice"));
        assert_eq!(member.session_id.as_deref(), Some("s1"));
        let member = TeamMemoryScope::new("review-team").with_member("bob");
        assert_eq!(member.member_id.as_deref(), Some("bob"));
        assert_eq!(TeamMemoryScope::default().team_id, "");
    }
    #[test]
    fn validate_accepts_and_rejects_ids() {
        assert!(TeamMemoryScope::member("team", "alice").validate().is_ok());
        assert!(TeamMemoryScope::default().validate().is_err());
        assert!(TeamMemoryScope::new("a/b")
            .validate()
            .unwrap_err()
            .to_string()
            .contains("team_id"));
        assert!(TeamMemoryScope::new("a\\b")
            .validate()
            .unwrap_err()
            .to_string()
            .contains("backslash"));
        assert!(TeamMemoryScope::new("a\0b")
            .validate()
            .unwrap_err()
            .to_string()
            .contains("NUL"));
        assert!(TeamMemoryScope::new("a\tb")
            .validate()
            .unwrap_err()
            .to_string()
            .contains("control characters"));
        assert!(TeamMemoryScope::member("team", "")
            .validate()
            .unwrap_err()
            .to_string()
            .contains("member_id"));
        assert!(TeamMemoryScope::new("team")
            .with_session("s/1")
            .validate()
            .unwrap_err()
            .to_string()
            .contains("session_id"));
        let long = "x".repeat(MAX_SCOPE_ID_CHARS + 1);
        assert!(TeamMemoryScope::new(&long)
            .validate()
            .unwrap_err()
            .to_string()
            .contains("at most 128"));
        assert!(TeamMemoryScope::new("x".repeat(MAX_SCOPE_ID_CHARS))
            .validate()
            .is_ok());
    }
    #[test]
    fn team_wide_view_allows_every_visibility() {
        let scope = TeamMemoryScope::new("team");
        assert!(scope.allows(&record("team", Some("alice"), TeamVisibility::Private)));
        assert!(scope.allows(&record("team", Some("alice"), TeamVisibility::Team)));
        assert!(scope.allows(&record(
            "team",
            Some("alice"),
            TeamVisibility::Members(vec!["bob".to_owned()])
        )));
        assert!(!scope.allows(&record("other", None, TeamVisibility::Team)));
    }
    #[test]
    fn member_view_applies_visibility_rules() {
        let scope = TeamMemoryScope::member("team", "alice");
        assert!(scope.allows(&record("team", Some("bob"), TeamVisibility::Team)));
        assert!(scope.allows(&record("team", Some("alice"), TeamVisibility::Private)));
        assert!(!scope.allows(&record("team", Some("bob"), TeamVisibility::Private)));
        assert!(!scope.allows(&record("team", None, TeamVisibility::Private)));
        assert!(scope.allows(&record(
            "team",
            Some("bob"),
            TeamVisibility::Members(vec!["alice".to_owned()])
        )));
        assert!(scope.allows(&record(
            "team",
            Some("alice"),
            TeamVisibility::Members(vec!["bob".to_owned()])
        )));
        assert!(!scope.allows(&record(
            "team",
            Some("bob"),
            TeamVisibility::Members(vec!["carol".to_owned()])
        )));
    }
    #[test]
    fn session_scoped_view_filters_other_sessions_but_keeps_sessionless_records() {
        let scope = TeamMemoryScope::new("team").with_session("s1");
        let mut same = record("team", None, TeamVisibility::Team);
        same.session_id = Some("s1".to_owned());
        let mut other = record("team", None, TeamVisibility::Team);
        other.session_id = Some("s2".to_owned());
        assert!(scope.allows(&same));
        assert!(!scope.allows(&other));
        assert!(scope.allows(&record("team", None, TeamVisibility::Team)));
    }
    #[test]
    fn serde_omits_absent_optionals() {
        let scope = TeamMemoryScope::new("team");
        assert_eq!(
            serde_json::to_value(&scope).unwrap(),
            serde_json::json!({"team_id": "team"})
        );
        let restored: TeamMemoryScope = serde_json::from_value(serde_json::json!({})).unwrap();
        assert_eq!(restored, TeamMemoryScope::default());
        let restored: TeamMemoryScope = serde_json::from_value(serde_json::json!({
            "team_id": "team", "member_id": "alice", "session_id": "s1"
        }))
        .unwrap();
        assert_eq!(
            restored,
            TeamMemoryScope::member("team", "alice").with_session("s1")
        );
    }
}
