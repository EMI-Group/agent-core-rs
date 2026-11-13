//! The identity of one pooled team run.
//!
//! Python's `TeamRuntimePool` keys a live runtime by `team_name` alone and
//! tears a stale entry down when the session changes; the Rust port keys by
//! `(team_name, session_id)` instead, so two sessions of the same team are
//! **two** independent pooled runs rather than a teardown plus a rebuild. That
//! is the in-process equivalent of Python's "switching sessions is a hard
//! boundary", without the destructive hop.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::run::TeamRef;

/// The key of one pooled team run: a team name and the session it runs in.
///
/// Serialized with `snake_case` field names so a host can persist or ship a
/// key; the natural order is `(team_name, session_id)`, which is also the order
/// [`crate::pool::TeamPool::list_active`] reports.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct PoolKey {
    /// The team the run belongs to.
    pub team_name: String,
    /// The session the run is bound to.
    pub session_id: String,
}

impl PoolKey {
    /// Builds a key from its two parts.
    #[must_use]
    pub fn new(team_name: impl Into<String>, session_id: impl Into<String>) -> Self {
        Self {
            team_name: team_name.into(),
            session_id: session_id.into(),
        }
    }

    /// Derives the key of `team` in `session_id`.
    ///
    /// An id-based reference uses the id verbatim and is **never** resolved, so
    /// deriving a key cannot instantiate a team through a registry provider.
    /// A live team reports `card().id`, falling back to `card().name` when the
    /// id is empty.
    #[must_use]
    pub fn for_team(team: &TeamRef, session_id: impl Into<String>) -> Self {
        let team_name = match team {
            TeamRef::Id(id) => id.clone(),
            TeamRef::Team(team) => {
                let card = team.card();
                if card.id.is_empty() {
                    card.name
                } else {
                    card.id
                }
            }
        };
        Self::new(team_name, session_id)
    }
}

impl fmt::Display for PoolKey {
    /// Renders the key as `team@session`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}@{}", self.team_name, self.session_id)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::sync::Arc;

    use openjiuwen_multiagent::{handoff_team, HandoffTeamConfig, Team, TeamCard};

    use super::*;

    /// A real (strategy-bearing) team whose only interesting part is its card.
    fn live_team(card: TeamCard) -> Arc<dyn Team> {
        Arc::new(handoff_team(card, HandoffTeamConfig::default()))
    }

    #[test]
    fn new_and_display_use_team_at_session() {
        let key = PoolKey::new("crew", "s-1");
        assert_eq!(key.team_name, "crew");
        assert_eq!(key.session_id, "s-1");
        assert_eq!(key.to_string(), "crew@s-1");
    }

    #[test]
    fn an_id_reference_keeps_the_id_verbatim() {
        let key = PoolKey::for_team(&TeamRef::id("crew"), "s-1");
        assert_eq!(key, PoolKey::new("crew", "s-1"));
        // A reference is never resolved: an unregistered id still derives a key.
        assert_eq!(
            PoolKey::for_team(&TeamRef::from("unregistered"), "s-2"),
            PoolKey::new("unregistered", "s-2")
        );
    }

    #[test]
    fn a_live_team_prefers_its_card_id() {
        let team = live_team(TeamCard::new("display-name").with_id("crew-id"));
        assert_eq!(
            PoolKey::for_team(&TeamRef::of(team), "s-1"),
            PoolKey::new("crew-id", "s-1")
        );
    }

    #[test]
    fn a_live_team_without_an_id_falls_back_to_its_name() {
        let team = live_team(TeamCard::new("crew").with_id(""));
        assert_eq!(
            PoolKey::for_team(&TeamRef::of(team), "s-1"),
            PoolKey::new("crew", "s-1")
        );
    }

    #[test]
    fn keys_order_by_team_then_session() {
        let mut keys = vec![
            PoolKey::new("crew-b", "s-1"),
            PoolKey::new("crew-a", "s-2"),
            PoolKey::new("crew-a", "s-1"),
        ];
        keys.sort();
        assert_eq!(
            keys,
            vec![
                PoolKey::new("crew-a", "s-1"),
                PoolKey::new("crew-a", "s-2"),
                PoolKey::new("crew-b", "s-1"),
            ]
        );
    }

    #[test]
    fn keys_round_trip_through_json() {
        let key = PoolKey::new("crew", "s-1");
        let value = serde_json::to_value(&key).unwrap();
        assert_eq!(
            value,
            serde_json::json!({"team_name": "crew", "session_id": "s-1"})
        );
        assert_eq!(serde_json::from_value::<PoolKey>(value).unwrap(), key);
        assert!(
            serde_json::from_value::<PoolKey>(serde_json::json!({"team_name": "future"})).is_err(),
            "both fields are required"
        );
    }
}
