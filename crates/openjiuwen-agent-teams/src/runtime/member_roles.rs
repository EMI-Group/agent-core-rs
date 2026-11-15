//! Role-driven member capabilities: which [`TeamRole`](crate::schema::TeamRole) values own a local
//! harness and which cannot be spawned at all.
//!
//! Python parity source: `openjiuwen/agent_teams/spawn/inprocess_spawn.py`, which has no role gate
//! of its own — the role Python assigns a member is what decides whether that member has a brain.
//! Python's `TeamRole` semantics are:
//!
//! * `TEAMMATE` and `EXTERNAL_CLI` are coordinated members;
//! * `BRIDGE_AGENT` is a full local teammate paired with a remote agent behind a text protocol;
//! * `WORKER` is a swarmflow single-shot stateless member (roster identity plus one `DeepAgent`
//!   turn, never polling the mailbox);
//! * `HUMAN_AGENT` is an avatar with a local harness that stays idle until the user drives it;
//! * `PASSIVE_HUMAN` has **no** harness at all, so no startup, restart or recovery path may spawn a
//!   runtime for it (only tool-call passthrough executes under its identity).
//!
//! [`has_local_harness`] answers that question for every role and [`in_process_refusal`] is the
//! typed error a spawn path returns for a role that must not get a runtime.

use openjiuwen_common::error::Error;

use crate::schema::TeamRole;

/// Returns `true` when `role` owns a local harness and may therefore be spawned in-process.
///
/// True for [`TeamRole::Leader`], [`TeamRole::Teammate`], [`TeamRole::HumanAgent`],
/// [`TeamRole::BridgeAgent`], [`TeamRole::Worker`] and [`TeamRole::ExternalCli`]; false only for
/// [`TeamRole::PassiveHuman`], which Python gives no harness at all.
///
/// The list is spelled out rather than expressed as "anything but a passive human" so a future
/// role has to be classified deliberately instead of inheriting a harness by accident.
#[must_use]
pub(crate) const fn has_local_harness(role: TeamRole) -> bool {
    matches!(
        role,
        TeamRole::Leader
            | TeamRole::Teammate
            | TeamRole::HumanAgent
            | TeamRole::BridgeAgent
            | TeamRole::Worker
            | TeamRole::ExternalCli
    )
}

/// Builds the refusal for spawning `member_name`, whose `role` has no local harness.
///
/// Always [`Error::Unsupported`]: Python's `TeamRole.PASSIVE_HUMAN` (the only role without a
/// harness) has no startup, restart or recovery path that spawns a runtime, so the typed error is
/// the parity-correct answer instead of a silent no-op. The message names the member, its role and
/// that rule; the Python enum name is derived from the wire string (`passive_human` →
/// `TeamRole.PASSIVE_HUMAN`), so the message stays role-generic.
pub(crate) fn in_process_refusal(member_name: &str, role: TeamRole) -> Error {
    Error::unsupported(format!(
        "member '{member_name}' has role '{}'; Python's TeamRole.{} has no local harness, so no \
         startup, restart or recovery path may spawn a runtime for it",
        role.as_str(),
        role.as_str().to_uppercase()
    ))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use openjiuwen_common::error::ErrorKind;

    use super::*;

    /// Every variant, spelled out so adding one has to extend this list.
    const ROLES: [TeamRole; 7] = [
        TeamRole::Leader,
        TeamRole::Teammate,
        TeamRole::HumanAgent,
        TeamRole::PassiveHuman,
        TeamRole::BridgeAgent,
        TeamRole::Worker,
        TeamRole::ExternalCli,
    ];

    /// The expected capability per role, exhaustive over [`TeamRole`].
    ///
    /// A new variant makes this `match` non-exhaustive, so the suite fails to compile until the
    /// role's capability is pinned here — which is the point of the explicit list.
    const fn expected(role: TeamRole) -> bool {
        match role {
            TeamRole::Leader
            | TeamRole::Teammate
            | TeamRole::HumanAgent
            | TeamRole::BridgeAgent
            | TeamRole::Worker
            | TeamRole::ExternalCli => true,
            TeamRole::PassiveHuman => false,
        }
    }

    #[test]
    fn has_local_harness_pins_every_role() {
        for role in ROLES {
            assert_eq!(has_local_harness(role), expected(role), "{}", role.as_str());
        }
        assert!(
            !has_local_harness(TeamRole::PassiveHuman),
            "only a passive human is harness-less"
        );
    }

    #[test]
    fn in_process_refusal_names_a_passive_human() {
        let error = in_process_refusal("alice", TeamRole::PassiveHuman);
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        assert_eq!(
            error.message(),
            "member 'alice' has role 'passive_human'; Python's TeamRole.PASSIVE_HUMAN has no \
             local harness, so no startup, restart or recovery path may spawn a runtime for it"
        );
    }

    #[test]
    fn in_process_refusal_derives_the_python_name_from_the_wire_string() {
        let error = in_process_refusal("alice", TeamRole::HumanAgent);
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        assert_eq!(
            error.message(),
            "member 'alice' has role 'human_agent'; Python's TeamRole.HUMAN_AGENT has no local \
             harness, so no startup, restart or recovery path may spawn a runtime for it"
        );
    }
}
