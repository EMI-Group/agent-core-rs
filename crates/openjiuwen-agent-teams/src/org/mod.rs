//! Organisation layer: the cross-team chart and its delegation rules, the organisation task pool
//! and leader mailbox, the swarmflow lane runner, and fork/compact of member context.
//!
//! The layer sits *above* [`crate::runtime`]: it never spawns, pauses or re-dispatches a member —
//! member lifecycle, permissions and dispatch stay inside one team's runtime. What it adds is
//! cross-team coordination:
//!
//! ```text
//! roles        OrganizationSpec ──▶ OrgChart (teams, members, reporting lines)
//! delegation   DelegationRules  ──▶ DelegationRouter ──▶ Allowed | Refused
//! task_pool    OrgTaskPool    ──▶ cross-team tasks over crate::storage::TeamStore
//! mailbox      OrgMailbox     ──▶ leader-to-leader mail + receipts + system notices
//! swarmflow    SwarmflowPlan  ──▶ bounded fan-out/fan-in over SwarmLane
//! fork_compact ForkedMemberSpec / CompactedTranscript (pure, deterministic)
//! ```
//!
//! Everything here is in-process and deterministic: no LLM call, no network, no wall-clock
//! ordering, and no process-global state. The durable half (task pool and mailbox) writes through
//! the existing [`crate::storage::TeamStore`] seam under a reserved organisation namespace instead
//! of introducing a second database.
//!
//! Python parity sources: `openjiuwen/agent_teams/organization/` (`schema.py`, `task_pool.py`,
//! `message_service.py`), `fork.py`, `fork_compact.py` and `workflow/`
//! (`swarmflow_infra.py`, `schema.py`, `runner.py`).
//!
//! | Module | Responsibility |
//! | --- | --- |
//! | [`roles`] | `OrgRole`, `OrgTeam`, `OrgMember`, `OrganizationSpec`, `OrgChart` and hierarchy queries |
//! | [`delegation`] | `DelegationRules`, `DelegationRequest`, `DelegationRouter`, `DelegationDecision` |
//! | [`task_pool`] | `OrgTaskPool`: cross-team task create/delegate/claim/start/complete/fail/review |
//! | [`mailbox`] | `OrgMailbox`: leader-to-leader messages, receipts, system notifications |
//! | [`swarmflow`] | `SwarmflowPlan`, `SwarmLane`, `SwarmflowRunner`, `SwarmflowOutcome` |
//! | [`fork_compact`] | `fork_member_context` (member fork) and `compact_transcript` (context compaction) |

use openjiuwen_common::error::{Error, Result};

pub mod delegation;
pub mod fork_compact;
pub mod mailbox;
pub mod roles;
pub mod swarmflow;
pub mod task_pool;

// Flat re-exports of every submodule are added once their public items land (`pub use roles::*;`
// …), so a host can reach the whole layer as `openjiuwen_agent_teams::org::OrgChart`.

/// Maximum length of an organisation id in characters.
pub const MAX_ORG_ID_LEN: usize = 96;

/// Characters an organisation id may contain (Python `validate_organization_id`).
pub const ORG_ID_ALLOWED_CHARS: &str = "A-Za-z0-9_.-";

/// Prefix of the store namespace that holds organisation rows.
///
/// Organisation tasks and messages are stored through [`crate::storage::TeamStore`] under
/// `org::<organization_id>`; a real team name never contains `:`, so the namespace cannot collide
/// with a team.
pub const ORG_NAMESPACE_PREFIX: &str = "org::";

/// Sender stamped on organisation system notifications (Python `__organization__`).
pub const ORG_SYSTEM_SENDER: &str = "__organization__";

/// Value of [`crate::storage::MessageRecord::protocol`] for an organisation message.
pub const ORG_MESSAGE_PROTOCOL: &str = "org";

/// Separator between a team and a member in a qualified member reference.
pub const MEMBER_QUALIFIER: &str = "::";

/// Returns `true` when `id` is a syntactically valid organisation id.
///
/// An id is 1–96 characters drawn from `[A-Za-z0-9_.-]`. The id doubles as a path segment in
/// Python's organisation workspace, which is why dots and dashes are allowed but slashes are not.
///
/// ```
/// use openjiuwen_agent_teams::org::{is_valid_org_id, org_namespace};
/// assert!(is_valid_org_id("acme.core-1"));
/// assert!(!is_valid_org_id("acme/core"));
/// assert_eq!(org_namespace("acme"), "org::acme");
/// ```
#[must_use]
pub fn is_valid_org_id(id: &str) -> bool {
    !id.is_empty()
        && id.chars().count() <= MAX_ORG_ID_LEN
        && id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '-'))
}

/// Validates an organisation id.
///
/// # Errors
///
/// Returns [`Error::validation`] when [`is_valid_org_id`] rejects `id`.
pub fn validate_org_id(id: &str) -> Result<()> {
    if is_valid_org_id(id) {
        Ok(())
    } else {
        Err(Error::validation(format!(
            "invalid organization id '{id}': expected 1..={MAX_ORG_ID_LEN} characters of \
             [{ORG_ID_ALLOWED_CHARS}]"
        )))
    }
}

/// Returns the store namespace that holds every organisation row of `organization_id`.
#[must_use]
pub fn org_namespace(organization_id: &str) -> String {
    format!("{ORG_NAMESPACE_PREFIX}{organization_id}")
}

/// Returns `true` when `team_name` is an organisation namespace rather than a team.
#[must_use]
pub fn is_org_namespace(team_name: &str) -> bool {
    team_name.starts_with(ORG_NAMESPACE_PREFIX)
}

/// Returns the organisation id encoded in `team_name`, when it is a namespace.
#[must_use]
pub fn org_namespace_id(team_name: &str) -> Option<&str> {
    team_name.strip_prefix(ORG_NAMESPACE_PREFIX)
}

/// Renders a team-qualified member reference (`<team>::<member>`).
#[must_use]
pub fn qualify_member(team: &str, member: &str) -> String {
    format!("{team}{MEMBER_QUALIFIER}{member}")
}

/// Splits a team-qualified member reference into its team and member.
#[must_use]
pub fn split_qualified(value: &str) -> Option<(&str, &str)> {
    let (team, member) = value.split_once(MEMBER_QUALIFIER)?;
    if team.is_empty() || member.is_empty() {
        return None;
    }
    Some((team, member))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn org_ids_follow_the_python_grammar() {
        assert!(is_valid_org_id("acme"));
        assert!(is_valid_org_id("acme.core-1_2"));
        assert!(is_valid_org_id(&"a".repeat(MAX_ORG_ID_LEN)));
        assert!(!is_valid_org_id(""));
        assert!(!is_valid_org_id(&"a".repeat(MAX_ORG_ID_LEN + 1)));
        assert!(!is_valid_org_id("acme/core"));
        assert!(!is_valid_org_id("acme core"));
        assert!(!is_valid_org_id("acme:core"));
    }

    #[test]
    fn validate_org_id_reports_the_python_message() {
        assert!(validate_org_id("acme").is_ok());
        let error = validate_org_id("../etc").unwrap_err();
        assert!(error.to_string().contains("invalid organization id"));
    }

    #[test]
    fn namespaces_round_trip() {
        let namespace = org_namespace("acme");
        assert_eq!(namespace, "org::acme");
        assert!(is_org_namespace(&namespace));
        assert_eq!(org_namespace_id(&namespace), Some("acme"));
        assert!(!is_org_namespace("acme"));
        assert_eq!(org_namespace_id("acme"), None);
    }

    #[test]
    fn qualified_member_references_round_trip() {
        assert_eq!(qualify_member("alpha", "dev-1"), "alpha::dev-1");
        assert_eq!(split_qualified("alpha::dev-1"), Some(("alpha", "dev-1")));
        assert_eq!(split_qualified("dev-1"), None);
        assert_eq!(split_qualified("::dev-1"), None);
        assert_eq!(split_qualified("alpha::"), None);
    }
}
