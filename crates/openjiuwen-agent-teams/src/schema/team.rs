//! Team-scoped runtime types: the member roles a team addresses and the runtime mirrors of a team
//! spec.
//!
//! Python parity source: `openjiuwen/agent_teams/schema/team.py` (`TeamRole`, `TeamSpec`,
//! `TeamRuntimeContext`, `TeamSnapshot`).
//!
//! Only [`TeamRole`] lives here today: it is the vocabulary shared by the roster specs in
//! [`super::member_spec`] and by runtime dispatch, so it is defined first. The remaining runtime
//! mirrors (`TeamSpec`, `TeamRuntimeContext`, `TeamSnapshot`) are added by the node that owns
//! them.

use serde::{Deserialize, Serialize};

/// The role a team member plays.
///
/// Wire values are the Python `TeamRole` strings, so a role persisted by the Python runtime loads
/// unchanged. [`TeamRole::Leader`] is the default (Python's `TeamRuntimeContext` defaults to the
/// leader role).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TeamRole {
    /// Drives the team: dispatches tasks, reviews submissions and talks to the user.
    Leader,
    /// Autonomous member that polls the mailbox and claims tasks from the board.
    Teammate,
    /// Avatar channel for a real human joining the team.
    HumanAgent,
    /// Directed human who is addressed but never polls the board.
    PassiveHuman,
    /// Local teammate paired with a remote agent reachable over a text protocol.
    BridgeAgent,
    /// Organisation-layer worker.
    Worker,
    /// Member whose work is performed by an external CLI process.
    ExternalCli,
}

impl Default for TeamRole {
    /// Python default: [`TeamRole::Leader`].
    fn default() -> Self {
        Self::Leader
    }
}

impl TeamRole {
    /// Returns the Python wire string of this role.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Leader => "leader",
            Self::Teammate => "teammate",
            Self::HumanAgent => "human_agent",
            Self::PassiveHuman => "passive_human",
            Self::BridgeAgent => "bridge_agent",
            Self::Worker => "worker",
            Self::ExternalCli => "external_cli",
        }
    }

    /// Returns `true` for roles that participate in the coordination loop as autonomous members.
    ///
    /// Python: `TeamRole.is_coordinated_member` — teammates and external CLI members both poll the
    /// mailbox, claim tasks and self-nudge, so gates should ask this instead of comparing against
    /// [`TeamRole::Teammate`].
    #[must_use]
    pub const fn is_coordinated_member(self) -> bool {
        matches!(self, Self::Teammate | Self::ExternalCli)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use serde_json::json;

    use super::*;

    /// Every role paired with its pinned wire string.
    const ROLE_WIRE_VALUES: [(TeamRole, &str); 7] = [
        (TeamRole::Leader, "leader"),
        (TeamRole::Teammate, "teammate"),
        (TeamRole::HumanAgent, "human_agent"),
        (TeamRole::PassiveHuman, "passive_human"),
        (TeamRole::BridgeAgent, "bridge_agent"),
        (TeamRole::Worker, "worker"),
        (TeamRole::ExternalCli, "external_cli"),
    ];

    #[test]
    fn team_role_wire_values_match_python() {
        for (role, wire) in ROLE_WIRE_VALUES {
            assert_eq!(role.as_str(), wire, "{role:?}");
            assert_eq!(serde_json::to_value(role).unwrap(), json!(wire), "{role:?}");
            assert_eq!(
                serde_json::from_value::<TeamRole>(json!(wire)).unwrap(),
                role,
                "{wire}"
            );
        }
    }

    #[test]
    fn team_role_default_is_leader() {
        assert_eq!(TeamRole::default(), TeamRole::Leader);
        let parsed: TeamRole = serde_json::from_value(json!("leader")).unwrap();
        assert_eq!(parsed, TeamRole::default());
    }

    #[test]
    fn coordinated_members_are_teammates_and_external_cli() {
        for (role, _) in ROLE_WIRE_VALUES {
            assert_eq!(
                role.is_coordinated_member(),
                matches!(role, TeamRole::Teammate | TeamRole::ExternalCli),
                "{role:?}"
            );
        }
    }

    #[test]
    fn unknown_role_is_rejected() {
        assert!(serde_json::from_value::<TeamRole>(json!("supervisor")).is_err());
    }
}
