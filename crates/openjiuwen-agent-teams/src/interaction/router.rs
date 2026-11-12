//! Top-layer parser for free-form `interact` text.
//!
//! Python parity source: `openjiuwen/agent_teams/interaction/router.py` (v0.1.20).
//! The runtime keeps channel and mention parsing in exactly one place — this module — so the
//! inboxes and the dispatch switch consume already-typed routing decisions and never re-parse a
//! body.
//!
//! # Grammar
//!
//! ```text
//! input     := channel? recipients? body
//! channel   := "# " | "$" name (" " | "@")
//! recipients := ("@" name " ")+
//! body      := <remaining text after consuming channel and recipients>
//! ```
//!
//! | Input | Route |
//! | --- | --- |
//! | `# body` | [`Route::GodView`] — body lands on the leader's `DeepAgent` |
//! | `$name body` | [`Route::HumanAgent`] — drives that avatar's `DeepAgent` |
//! | `@member body` | [`Route::Member`] — point-to-point to one roster member |
//! | `@a @b body` | [`Route::Member`] with `targets = [a, b]` — one bus message per recipient |
//! | `@all body` / `@* body` | [`Route::Broadcast`] |
//! | bare text | [`Route::Broadcast`] |
//!
//! Whitespace separates every prefix from its body — `#hashtag` and `$variable` are content, not
//! channel markers, because they lack the trailing separator the grammar requires.
//!
//! # Roster resolution
//!
//! Parsing is pure syntax; [`resolve_targets`] applies the roster. Unknown `@` names are dropped
//! from the target list, and when every mentioned name is unknown the result is a broadcast
//! carrying the original text, so the user's `@ghost` is preserved as prose instead of
//! disappearing. An unknown `$name` channel is likewise unrouted and becomes a broadcast.
//!
//! # Example
//!
//! ```
//! use openjiuwen_agent_teams::interaction::{parse_interact_str, Route};
//!
//! let roster = vec!["dev-1".to_owned(), "dev-2".to_owned()];
//! assert_eq!(
//!     parse_interact_str("@dev-1 please start", &roster),
//!     Route::Member {
//!         targets: vec!["dev-1".to_owned()],
//!         body: "please start".to_owned(),
//!     }
//! );
//! assert_eq!(
//!     parse_interact_str("# status?", &roster),
//!     Route::GodView {
//!         body: "status?".to_owned(),
//!     }
//! );
//! ```
use serde::{Deserialize, Serialize};

use crate::constants::parse_mention;

/// The `# ` god-view channel prefix.
const GOD_VIEW_PREFIX: &str = "# ";
/// Reserved mention targets that mean "team-wide broadcast".
const BROADCAST_TARGETS: [&str; 2] = ["all", "*"];

/// A parsed, roster-resolved interact routing decision.
///
/// Serialized with a `kind` tag (`god_view`, `human_agent`, `member`, `broadcast`) so a host
/// application can forward the decision without re-parsing the body.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Route {
    /// Speak directly to the leader's `DeepAgent`.
    GodView {
        /// Message body after the `# ` prefix.
        body: String,
    },
    /// Drive one registered human agent's avatar loop.
    HumanAgent {
        /// Human-agent member name from the `$name` prefix.
        name: String,
        /// Message body.
        body: String,
    },
    /// Point-to-point bus traffic to one or more roster members.
    Member {
        /// Resolved recipient names, in mention order and deduplicated.
        targets: Vec<String>,
        /// Message body shared by every recipient.
        body: String,
    },
    /// Team-wide broadcast.
    Broadcast {
        /// Message body, or the original text when an all-unknown mention list was folded back.
        body: String,
    },
}

/// Parses one `interact` input into a routing decision, applying `roster` to `@` mentions.
///
/// `roster` holds the team's addressable member names. Parsing is otherwise pure; see the
/// module docs for the grammar and the fold-back rule for unknown mentions.
#[must_use]
pub fn parse_interact_str(text: &str, roster: &[String]) -> Route {
    let mut rest = text;
    let mut human_agent: Option<&str> = None;
    let mut explicit_god_view = false;

    if let Some(stripped) = rest.strip_prefix(GOD_VIEW_PREFIX) {
        rest = stripped.trim_start_matches(char::is_whitespace);
        explicit_god_view = true;
    } else if let Some((name, body)) = split_human_agent_prefix(rest) {
        human_agent = Some(name);
        rest = body;
    }

    // Leading mention segments: `@a @b body`. `parse_mention` requires a non-empty body, so a
    // dangling `@name` at the end of the input stays prose.
    let mut mentions: Vec<&str> = Vec::new();
    while let Some((target, body)) = parse_mention(rest) {
        mentions.push(target);
        rest = body;
    }
    let body = rest.to_owned();

    // A broadcast token supersedes every other listed recipient: broadcast already reaches
    // each of them.
    if mentions
        .iter()
        .any(|target| BROADCAST_TARGETS.contains(target))
    {
        return Route::Broadcast { body };
    }

    let mentions: Vec<String> = mentions.into_iter().map(str::to_owned).collect();
    let (resolved, unknown) = resolve_targets(&mentions, roster);

    if let Some(name) = human_agent {
        // A `$name` channel names a member: when no roster row matches, there is nobody to
        // drive, so the whole input is unrouted and folds back into a broadcast.
        if !roster.iter().any(|member| member == name) {
            return Route::Broadcast {
                body: text.to_owned(),
            };
        }
        if mentions.is_empty() {
            return Route::HumanAgent {
                name: name.to_owned(),
                body,
            };
        }
        if resolved.is_empty() {
            // Every recipient is unknown: fold the original text back into a broadcast so no
            // mention is lost as prose.
            return Route::Broadcast {
                body: text.to_owned(),
            };
        }
        // Known recipients win; unknown ones are dropped from the target list.
        return Route::Member {
            targets: resolved,
            body,
        };
    }

    if mentions.is_empty() {
        if explicit_god_view {
            return Route::GodView { body };
        }
        return Route::Broadcast { body };
    }
    if resolved.is_empty() {
        // Every mention was prose: preserve the original text verbatim.
        return Route::Broadcast {
            body: text.to_owned(),
        };
    }
    if !unknown.is_empty() {
        tracing::debug!(
            dropped = unknown.len(),
            "interact: dropping mentions that match no roster member"
        );
    }
    Route::Member {
        targets: resolved,
        body,
    }
}

/// Splits `$name`, `$name body` and `$name@target body` channel prefixes.
///
/// Mirrors Python's `^\$([^\s@]+)(?:\s+|(?=@))([\s\S]*)$`: the name runs up to the first
/// whitespace or `@`, and that separator is consumed. `$name` with no separator, `$` alone and
/// `$@target` are not channel prefixes.
fn split_human_agent_prefix(text: &str) -> Option<(&str, &str)> {
    let rest = text.strip_prefix('$')?;
    let name_end = rest.find(|c: char| c.is_whitespace() || c == '@')?;
    let name = &rest[..name_end];
    if name.is_empty() {
        return None;
    }
    let body = rest[name_end..].trim_start_matches(char::is_whitespace);
    Some((name, body))
}

/// Splits `mentions` into those present in `roster` and those that are not.
///
/// Both lists preserve mention order and drop duplicates. Broadcast tokens (`all`, `*`) are
/// handled by [`parse_interact_str`] before resolution, so they are treated as ordinary names
/// here; this function is pure roster membership.
#[must_use]
pub fn resolve_targets(mentions: &[String], roster: &[String]) -> (Vec<String>, Vec<String>) {
    let mut resolved = Vec::new();
    let mut unknown = Vec::new();
    for mention in mentions {
        if roster.iter().any(|member| member == mention) {
            if !resolved.contains(mention) {
                resolved.push(mention.clone());
            }
        } else if !unknown.contains(mention) {
            unknown.push(mention.clone());
        }
    }
    (resolved, unknown)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn roster() -> Vec<String> {
        vec!["dev-1".to_owned(), "dev-2".to_owned(), "alice".to_owned()]
    }

    fn member(targets: &[&str], body: &str) -> Route {
        Route::Member {
            targets: targets.iter().map(|target| (*target).to_owned()).collect(),
            body: body.to_owned(),
        }
    }

    fn broadcast(body: &str) -> Route {
        Route::Broadcast {
            body: body.to_owned(),
        }
    }

    fn god_view(body: &str) -> Route {
        Route::GodView {
            body: body.to_owned(),
        }
    }

    fn human_agent(name: &str, body: &str) -> Route {
        Route::HumanAgent {
            name: name.to_owned(),
            body: body.to_owned(),
        }
    }

    #[test]
    fn hash_prefix_is_god_view() {
        assert_eq!(
            parse_interact_str("# status?", &roster()),
            god_view("status?")
        );
        assert_eq!(
            parse_interact_str("#   status?", &roster()),
            god_view("status?")
        );
        assert_eq!(parse_interact_str("# ", &roster()), god_view(""));
    }

    #[test]
    fn hash_without_space_is_bare_text() {
        assert_eq!(
            parse_interact_str("#hashtag", &roster()),
            broadcast("#hashtag")
        );
    }

    #[test]
    fn dollar_prefix_drives_human_agent() {
        assert_eq!(
            parse_interact_str("$alice hello there", &roster()),
            human_agent("alice", "hello there")
        );
        assert_eq!(
            parse_interact_str("$alice   hello", &roster()),
            human_agent("alice", "hello")
        );
        assert_eq!(
            parse_interact_str("$alice ", &roster()),
            human_agent("alice", "")
        );
    }

    #[test]
    fn dollar_without_separator_is_bare_text() {
        assert_eq!(
            parse_interact_str("$variable", &roster()),
            broadcast("$variable")
        );
    }

    #[test]
    fn at_prefix_routes_to_member() {
        assert_eq!(
            parse_interact_str("@dev-1 start the task", &roster()),
            member(&["dev-1"], "start the task")
        );
    }

    #[test]
    fn multiple_leading_mentions_fan_out_one_body() {
        assert_eq!(
            parse_interact_str("@dev-1 @dev-2 standup now", &roster()),
            member(&["dev-1", "dev-2"], "standup now")
        );
    }

    #[test]
    fn broadcast_tokens_route_to_broadcast() {
        assert_eq!(
            parse_interact_str("@all standup", &roster()),
            broadcast("standup")
        );
        assert_eq!(
            parse_interact_str("@* standup", &roster()),
            broadcast("standup")
        );
        assert_eq!(
            parse_interact_str("@all @dev-1 standup", &roster()),
            broadcast("standup")
        );
        assert_eq!(
            parse_interact_str("$alice @all standup", &roster()),
            broadcast("standup")
        );
    }

    #[test]
    fn bare_text_is_broadcast() {
        assert_eq!(
            parse_interact_str("hello team", &roster()),
            broadcast("hello team")
        );
    }

    #[test]
    fn empty_body_routes_are_deterministic() {
        assert_eq!(parse_interact_str("", &roster()), broadcast(""));
        assert_eq!(parse_interact_str("   ", &roster()), broadcast("   "));
    }

    #[test]
    fn unknown_mentions_are_dropped_when_some_resolve() {
        assert_eq!(
            parse_interact_str("@ghost @dev-1 hello", &roster()),
            member(&["dev-1"], "hello")
        );
    }

    #[test]
    fn all_unknown_mentions_fold_back_into_broadcast_with_original_text() {
        assert_eq!(
            parse_interact_str("@ghost hello", &roster()),
            broadcast("@ghost hello")
        );
        assert_eq!(parse_interact_str("@ghost", &roster()), broadcast("@ghost"));
    }

    #[test]
    fn unknown_human_agent_name_folds_back_into_broadcast() {
        assert_eq!(
            parse_interact_str("$ghost hello", &roster()),
            broadcast("$ghost hello")
        );
    }

    #[test]
    fn known_human_agent_with_recipient_becomes_member() {
        assert_eq!(
            parse_interact_str("$alice@dev-1 hello", &roster()),
            member(&["dev-1"], "hello")
        );
    }

    #[test]
    fn known_human_agent_with_only_unknown_recipients_folds_back() {
        assert_eq!(
            parse_interact_str("$alice@ghost hello", &roster()),
            broadcast("$alice@ghost hello")
        );
    }

    #[test]
    fn resolve_targets_preserves_order_and_deduplicates() {
        let mentions = vec![
            "dev-2".to_owned(),
            "dev-1".to_owned(),
            "dev-2".to_owned(),
            "ghost".to_owned(),
            "ghost".to_owned(),
        ];
        let (resolved, unknown) = resolve_targets(&mentions, &roster());
        assert_eq!(resolved, vec!["dev-2".to_owned(), "dev-1".to_owned()]);
        assert_eq!(unknown, vec!["ghost".to_owned()]);
    }

    #[test]
    fn route_round_trips_through_json() {
        let route = member(&["dev-1", "dev-2"], "hi");
        let encoded = serde_json::to_string(&route).unwrap();
        let decoded: Route = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, route);
    }
}
