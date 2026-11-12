//! Reserved member names, defaults, and the `@mention` grammar.
//!
//! Python parity source: `openjiuwen/agent_teams/constants.py` and
//! `openjiuwen/agent_teams/interaction/router.py::parse_mention`.
//!
//! These values are wire-visible: they appear in prompts, in persisted team rows and in the
//! `send_message` tool contract, so they must never drift.

/// Default member name of the team leader.
pub const DEFAULT_LEADER_MEMBER_NAME: &str = "team_leader";
/// Member name reserved for the human-agent avatar channel.
pub const HUMAN_AGENT_MEMBER_NAME: &str = "human_agent";
/// Pseudo member name that addresses the end user.
pub const USER_PSEUDO_MEMBER_NAME: &str = "user";
/// Wildcard target used by `send_message` for a broadcast.
pub const BROADCAST_TARGET: &str = "*";
/// Member names a roster may not use (Python `RESERVED_MEMBER_NAMES`).
pub const RESERVED_MEMBER_NAMES: [&str; 3] = [
    USER_PSEUDO_MEMBER_NAME,
    DEFAULT_LEADER_MEMBER_NAME,
    HUMAN_AGENT_MEMBER_NAME,
];
/// Default team name produced by [`crate::schema::TeamAgentSpec`].
pub const DEFAULT_TEAM_NAME: &str = "agent_team";
/// Default leader display name.
pub const DEFAULT_LEADER_DISPLAY_NAME: &str = "Team Leader";
/// Environment variable carrying a serialized team join descriptor.
pub const TEAM_JOIN_ENV: &str = "OPENJIUWEN_TEAM_JOIN";
/// Environment variable holding the openJiuwen home directory.
pub const OPENJIUWEN_HOME_ENV: &str = "OPENJIUWEN_HOME";
/// Maximum characters accepted by `send_message` on any route (Python `MAX_CONTENT_CHARS`).
pub const MAX_MESSAGE_CONTENT_CHARS: usize = 20_000;
/// Language used when neither the team nor the member declares one.
pub const DEFAULT_LANGUAGE: &str = "cn";

/// Returns `true` when `name` is reserved and therefore illegal for a roster member.
///
/// ```
/// use openjiuwen_agent_teams::{is_reserved_name, parse_mention};
/// assert!(is_reserved_name("user"));
/// assert!(!is_reserved_name("dev-1"));
/// assert_eq!(parse_mention("@dev-1 please start"), Some(("dev-1", "please start")));
/// ```
#[must_use]
pub fn is_reserved_name(name: &str) -> bool {
    RESERVED_MEMBER_NAMES.contains(&name)
}

/// Splits an `@member body` mention into its target and body.
///
/// Mirrors Python's `parse_mention` regular expression `^@(\S+)\s+([\s\S]+)$`: the target is the
/// first whitespace-delimited token after `@`, and the body keeps all following text verbatim
/// (including newlines). A bare `@name` with no body, or text that does not start with `@`,
/// yields `None`.
#[must_use]
pub fn parse_mention(text: &str) -> Option<(&str, &str)> {
    let rest = text.strip_prefix('@')?;
    let mut chars = rest.char_indices();
    let first_ws = rest.find(char::is_whitespace)?;
    // The first token must be non-empty; `@ body` is not a mention.
    let target = &rest[..first_ws];
    if target.is_empty() {
        return None;
    }
    let _ = chars.next();
    let body = rest[first_ws..].trim_start_matches(char::is_whitespace);
    if body.is_empty() {
        return None;
    }
    Some((target, body))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mention_parsing_matches_python() {
        assert_eq!(
            parse_mention("@dev-1 please start task 123"),
            Some(("dev-1", "please start task 123"))
        );
        assert_eq!(parse_mention("@dev-1 line one\nline two"), Some(("dev-1", "line one\nline two")));
        assert_eq!(parse_mention("@dev-1"), None);
        assert_eq!(parse_mention("@ dev-1 body"), None);
        assert_eq!(parse_mention("dev-1 body"), None);
    }

    #[test]
    fn reserved_names_match_python() {
        assert!(is_reserved_name("user"));
        assert!(is_reserved_name("team_leader"));
        assert!(is_reserved_name("human_agent"));
        assert!(!is_reserved_name("leader"));
    }
}
