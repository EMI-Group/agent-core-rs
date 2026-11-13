//! Where an inbound payload goes, and how a delivery attempt ended.
//!
//! A [`RunTarget`] names a live run by team name and/or session id — the same
//! pair Python's `interact_agent_team(payload, team_name=..., session_id=...)`
//! addresses. A target with both fields unset matches nothing ("missing
//! target"), so an accidentally empty target never broadcasts to every run.
//!
//! [`DeliverResult`] is Python's `DeliverResult`: it distinguishes a missing
//! target from a run whose inbox no longer exists and from a closed or full
//! inbox.

use serde::{Deserialize, Serialize};

use super::bridge::ActiveRunInfo;

/// Which live run an inbound payload is addressed to.
///
/// A target matches a run when every field it sets equals the run's field and
/// at least one field is set; see [`RunTarget::matches`]. Both fields are
/// `Option` because a host may address a run by session only (an agent run has
/// no team) or by team only (the team's own runtime).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct RunTarget {
    /// Team name of the addressed run, when the host knows it.
    pub team_name: Option<String>,
    /// Session id of the addressed run, when the host knows it.
    pub session_id: Option<String>,
}

impl RunTarget {
    /// An empty target: matches nothing and yields
    /// [`DeliverResult::MissingTarget`] on delivery.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Addresses a run by team name.
    #[must_use]
    pub fn for_team(name: impl Into<String>) -> Self {
        Self {
            team_name: Some(name.into()),
            session_id: None,
        }
    }

    /// Addresses a run by session id.
    #[must_use]
    pub fn for_session(id: impl Into<String>) -> Self {
        Self {
            team_name: None,
            session_id: Some(id.into()),
        }
    }

    /// Whether `info` is the run this target addresses.
    ///
    /// Both target fields unset matches nothing. A set field must equal the
    /// run's corresponding field; a run without that field never matches.
    #[must_use]
    pub fn matches(&self, info: &ActiveRunInfo) -> bool {
        if self.team_name.is_none() && self.session_id.is_none() {
            return false;
        }
        self.team_name
            .as_deref()
            .is_none_or(|name| info.team_name.as_deref() == Some(name))
            && self
                .session_id
                .as_deref()
                .is_none_or(|id| info.session_id.as_deref() == Some(id))
    }
}

/// How a payload delivery attempt ended.
///
/// Python's `DeliverResult`; [`DeliverResult::success`] is `true` only for
/// [`DeliverResult::Accepted`], so a caller can branch once instead of matching
/// every failure kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeliverResult {
    /// The payload was queued in the matching run's inbox.
    Accepted,
    /// No registered run matched the target (including an empty target).
    MissingTarget,
    /// The matching run has no inbox: it was never created, or a failed delivery
    /// dropped the dead sender.
    NoInbox,
    /// The matching run's inbox is closed (its receiver was dropped) or full
    /// ([`INBOX_CAPACITY`](super::INBOX_CAPACITY) unread payloads).
    Closed,
}

impl DeliverResult {
    /// Whether the payload was accepted.
    #[must_use]
    pub fn success(&self) -> bool {
        matches!(self, Self::Accepted)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use crate::monitor::RunPhase;

    fn info(team: Option<&str>, session: Option<&str>) -> ActiveRunInfo {
        ActiveRunInfo {
            run_id: "run-1".to_string(),
            label: "assistant".to_string(),
            team_name: team.map(str::to_string),
            session_id: session.map(str::to_string),
            state: RunPhase::Queued,
            gate_closed: false,
            queued: 0,
        }
    }

    #[test]
    fn an_empty_target_matches_nothing() {
        assert!(!RunTarget::new().matches(&info(None, None)));
        assert!(!RunTarget::new().matches(&info(Some("crew"), Some("s-1"))));
        assert_eq!(RunTarget::new(), RunTarget::default());
    }

    #[test]
    fn a_set_field_must_equal_the_run() {
        let target = RunTarget::for_team("crew");
        assert!(target.matches(&info(Some("crew"), Some("s-1"))));
        assert!(!target.matches(&info(Some("other"), Some("s-1"))));
        assert!(!target.matches(&info(None, Some("s-1"))));

        let target = RunTarget::for_session("s-1");
        assert!(target.matches(&info(None, Some("s-1"))));
        assert!(target.matches(&info(Some("crew"), Some("s-1"))));
        assert!(!target.matches(&info(Some("crew"), Some("s-2"))));
        assert!(!target.matches(&info(Some("crew"), None)));
    }

    #[test]
    fn both_fields_set_require_both_to_match() {
        let target = RunTarget {
            team_name: Some("crew".to_string()),
            session_id: Some("s-1".to_string()),
        };
        assert!(target.matches(&info(Some("crew"), Some("s-1"))));
        assert!(!target.matches(&info(Some("crew"), Some("s-2"))));
        assert!(!target.matches(&info(Some("other"), Some("s-1"))));
    }

    #[test]
    fn only_accepted_is_success() {
        assert!(DeliverResult::Accepted.success());
        for result in [
            DeliverResult::MissingTarget,
            DeliverResult::NoInbox,
            DeliverResult::Closed,
        ] {
            assert!(!result.success());
        }
    }

    #[test]
    fn target_round_trips_through_json() {
        let target = RunTarget {
            team_name: Some("crew".to_string()),
            session_id: Some("s-1".to_string()),
        };
        let value = serde_json::to_value(&target).unwrap();
        assert_eq!(
            value,
            serde_json::json!({"team_name": "crew", "session_id": "s-1"})
        );
        assert_eq!(serde_json::from_value::<RunTarget>(value).unwrap(), target);
        assert_eq!(
            serde_json::from_value::<RunTarget>(serde_json::json!({"future": true})).unwrap(),
            RunTarget::default()
        );
    }
}
