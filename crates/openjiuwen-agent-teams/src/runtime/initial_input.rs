//! Caller request parsing: one `Value` in, a leader query and pre-routed deliveries out.
//!
//! When a caller runs a team it hands the runtime a single opaque `inputs` value (a plain string
//! or an object carrying `query`/`message`/`input`). Before the leader's `DeepAgent` ever sees the
//! text, the runtime decides whether the request is something to *say* to the leader or something
//! to *deliver* to a specific member — a human-agent avatar (`$name @member body`) or a roster
//! member mentioned directly (`@member body`).
//!
//! Python parity source: `openjiuwen/agent_teams/agent/team_agent.py`
//! (`_prepare_initial_input` / `_initial_leader_route_payloads`, v0.1.20), re-expressed as a pure
//! function over the roster so it is testable without a team, a store or a session.
//!
//! The parsing rules are pinned by `src/runtime/CONTEXT.md`:
//!
//! | Query | `leader_query` | `deliveries` |
//! | --- | --- | --- |
//! | `$human @member body` (member on the roster) | `None` | one `HumanAgent` payload |
//! | `$human body` | `None` | one `HumanAgent` payload with `target: None` |
//! | `$ghost @member body`, `$human @ghost body` | the query verbatim | none |
//! | `@member body` (member on the roster) | `None` | one `HumanAgent` payload, sender `user` |
//! | `@ghost body`, `@all body`, bare `@member` | the query verbatim | none |
//! | plain text, `# body`, `@all body` | the query verbatim | none |
//! | whitespace only | `None` | none |
//!
//! "Verbatim" means the query as supplied (outer whitespace trimmed), channel prefix included, so
//! a request the runtime cannot route still reaches the leader unchanged instead of being dropped.
//!
//! # Example
//!
//! ```
//! use serde_json::Value;
//! use openjiuwen_agent_teams::runtime::initial_input::parse_initial_input;
//!
//! let roster = vec!["ai-counter-1".to_owned()];
//! let input = parse_initial_input(
//!     &Value::from("$human-counter-1 @ai-counter-1 body"),
//!     &roster,
//! );
//! assert_eq!(input.leader_query, None);
//! assert_eq!(input.deliveries.len(), 1);
//!
//! let input = parse_initial_input(&Value::from("plan the sprint"), &roster);
//! assert_eq!(input.leader_query.as_deref(), Some("plan the sprint"));
//! assert!(input.deliveries.is_empty());
//! ```
use serde_json::Value;

use crate::constants::{parse_mention, BROADCAST_TARGET, USER_PSEUDO_MEMBER_NAME};
use crate::interaction::{HumanAgentMessage, InteractPayload};

/// Mention target that means "everyone on the team", alongside [`BROADCAST_TARGET`].
const ALL_TARGET: &str = "all";

/// A caller request resolved into leader input and pre-routed member deliveries.
///
/// Exactly one half is normally populated: a routable delivery sets `leader_query` to `None` so
/// the leader is not asked to re-dispatch text it never sees, while an unroutable request keeps
/// the whole query for the leader and delivers nothing.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct InitialInput {
    /// Text to send to the team leader, as supplied apart from outer whitespace.
    pub leader_query: Option<String>,
    /// Payloads to deliver directly, without involving the leader.
    pub deliveries: Vec<InteractPayload>,
}

/// Parses a caller's `inputs` value into a leader query and pre-routed deliveries.
///
/// `roster` holds the team's addressable member names; a `$`/`@` route only becomes a delivery
/// when its target is on that roster and is not a broadcast token. An empty roster therefore
/// sends every request to the leader, and an unknown target, a broadcast token or unparsable
/// mention text falls back to the leader query verbatim.
///
/// Returns an empty [`InitialInput`] when the value carries no text at all (missing keys, non-string
/// values, `Value::Null`, numbers) or when the extracted text trims to nothing.
#[must_use]
pub fn parse_initial_input(input: &Value, roster: &[String]) -> InitialInput {
    let Some(query) = extract_query(input) else {
        return InitialInput::default();
    };
    let query = query.trim();
    if query.is_empty() {
        return InitialInput::default();
    }

    if let Some(message) = parse_human_agent_route(query, roster) {
        return delivery(message);
    }
    if let Some(message) = parse_member_route(query, roster) {
        return delivery(message);
    }
    InitialInput {
        leader_query: Some(query.to_owned()),
        deliveries: Vec::new(),
    }
}

/// Wraps one routed [`HumanAgentMessage`] as the sole delivery of an [`InitialInput`].
fn delivery(message: HumanAgentMessage) -> InitialInput {
    InitialInput {
        leader_query: None,
        deliveries: vec![InteractPayload::HumanAgent(message)],
    }
}

/// Extracts the caller's text from an `inputs` value.
///
/// A string value is the query. An object is probed in contract order: a non-empty `query`, then
/// a non-empty `message`, then a non-empty string `input`, then `input.query` when `input` is an
/// object. Everything else yields `None`.
fn extract_query(input: &Value) -> Option<String> {
    match input {
        Value::String(text) => Some(text.clone()),
        Value::Object(map) => {
            for key in ["query", "message"] {
                if let Some(text) = map.get(key).and_then(Value::as_str) {
                    if !text.is_empty() {
                        return Some(text.to_owned());
                    }
                }
            }
            match map.get("input") {
                Some(Value::String(text)) if !text.is_empty() => Some(text.clone()),
                Some(Value::Object(inner)) => inner
                    .get("query")
                    .and_then(Value::as_str)
                    .filter(|text| !text.is_empty())
                    .map(str::to_owned),
                _ => None,
            }
        }
        _ => None,
    }
}

/// Whether `target` names one member of `roster` (broadcast tokens never do).
fn is_addressable(target: &str, roster: &[String]) -> bool {
    target != BROADCAST_TARGET
        && target != ALL_TARGET
        && roster.iter().any(|member| member == target)
}

/// Routes `$human_name [@target] body` as a human-agent delivery.
///
/// The name is the first whitespace-delimited token after `$`. A following `@` opens an optional
/// recipient, which must be on the roster; any other remaining text is the body driving the human
/// agent's own avatar loop. Returns `None` for everything the leader must see instead: a bare `$`,
/// a recipient that is unknown or a broadcast token, or an empty body.
fn parse_human_agent_route(query: &str, roster: &[String]) -> Option<HumanAgentMessage> {
    let rest = query.strip_prefix('$')?.trim_start();
    let human_name = rest.split_whitespace().next()?;
    let remainder = rest[human_name.len()..].trim_start();

    if remainder.starts_with('@') {
        let (target, body) = parse_mention(remainder)?;
        if !is_addressable(target, roster) {
            return None;
        }
        return Some(HumanAgentMessage {
            human_name: human_name.to_owned(),
            target: Some(target.to_owned()),
            body: body.to_owned(),
            sender: Some(human_name.to_owned()),
        });
    }

    if remainder.is_empty() {
        return None;
    }
    Some(HumanAgentMessage {
        human_name: human_name.to_owned(),
        target: None,
        body: remainder.to_owned(),
        sender: Some(human_name.to_owned()),
    })
}

/// Routes `@member body` as a delivery from the `user` pseudo-member.
///
/// Returns `None` when the mention does not parse (bare `@member`), when the target is a broadcast
/// token, or when the target is not on `roster` — the leader then sees the query verbatim.
fn parse_member_route(query: &str, roster: &[String]) -> Option<HumanAgentMessage> {
    if !query.starts_with('@') {
        return None;
    }
    let (target, body) = parse_mention(query)?;
    if !is_addressable(target, roster) {
        return None;
    }
    Some(HumanAgentMessage {
        human_name: USER_PSEUDO_MEMBER_NAME.to_owned(),
        target: Some(target.to_owned()),
        body: body.to_owned(),
        sender: Some(USER_PSEUDO_MEMBER_NAME.to_owned()),
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use serde_json::json;

    fn roster() -> Vec<String> {
        vec!["dev-1".to_owned(), "ai-counter-1".to_owned()]
    }

    /// The single human-agent payload of `input`, panicking when the shape is wrong.
    fn only_delivery(input: &InitialInput) -> HumanAgentMessage {
        assert_eq!(input.deliveries.len(), 1, "expected one delivery");
        match input.deliveries.first() {
            Some(InteractPayload::HumanAgent(message)) => message.clone(),
            other => panic!("expected a human_agent payload, got {other:?}"),
        }
    }

    fn assert_leader(query: &str, input: &InitialInput) {
        assert_eq!(input.leader_query.as_deref(), Some(query));
        assert!(input.deliveries.is_empty(), "expected no deliveries");
    }

    #[test]
    fn pinned_parity_example_routes_to_one_human_agent_delivery() {
        let input = parse_initial_input(
            &Value::from("$human-counter-1 @ai-counter-1 body"),
            &roster(),
        );
        assert_eq!(input.leader_query, None);
        assert_eq!(
            only_delivery(&input),
            HumanAgentMessage {
                human_name: "human-counter-1".to_owned(),
                target: Some("ai-counter-1".to_owned()),
                body: "body".to_owned(),
                sender: Some("human-counter-1".to_owned()),
            }
        );
    }

    #[test]
    fn human_agent_without_recipient_has_no_target() {
        let input = parse_initial_input(&Value::from("$alice hello"), &roster());
        assert_eq!(input.leader_query, None);
        assert_eq!(
            only_delivery(&input),
            HumanAgentMessage {
                human_name: "alice".to_owned(),
                target: None,
                body: "hello".to_owned(),
                sender: Some("alice".to_owned()),
            }
        );
    }

    #[test]
    fn human_agent_body_keeps_inner_whitespace_and_newlines() {
        let input = parse_initial_input(&Value::from("$alice line one\nline two"), &roster());
        assert_eq!(only_delivery(&input).body, "line one\nline two");
    }

    #[test]
    fn bare_mention_from_user_routes_to_that_member() {
        let input = parse_initial_input(&Value::from("@dev-1 please start"), &roster());
        assert_eq!(input.leader_query, None);
        assert_eq!(
            only_delivery(&input),
            HumanAgentMessage {
                human_name: USER_PSEUDO_MEMBER_NAME.to_owned(),
                target: Some("dev-1".to_owned()),
                body: "please start".to_owned(),
                sender: Some(USER_PSEUDO_MEMBER_NAME.to_owned()),
            }
        );
        assert_eq!(only_delivery(&input).sender.as_deref(), Some("user"));
    }

    #[test]
    fn unknown_mention_goes_to_the_leader() {
        let input = parse_initial_input(&Value::from("@ghost body"), &["dev-1".to_owned()]);
        assert_leader("@ghost body", &input);
    }

    #[test]
    fn broadcast_mentions_go_to_the_leader() {
        for query in ["@all body", "@* body", "@all"] {
            let input = parse_initial_input(&Value::from(query), &roster());
            assert_leader(query, &input);
        }
    }

    #[test]
    fn unknown_human_agent_recipient_goes_to_the_leader() {
        let input = parse_initial_input(&Value::from("$alice @ghost body"), &roster());
        assert_leader("$alice @ghost body", &input);
    }

    #[test]
    fn unknown_human_agent_channel_goes_to_the_leader() {
        // A `$name` channel is not roster-checked (the human agent need not be a row), only its
        // optional recipient is; a `$` with no name, and a `$name` with no body, stay with the
        // leader.
        let input = parse_initial_input(&Value::from("$ hi"), &roster());
        assert_leader("$ hi", &input);
        let input = parse_initial_input(&Value::from("$alice"), &roster());
        assert_leader("$alice", &input);
    }

    #[test]
    fn human_agent_broadcast_recipient_goes_to_the_leader() {
        let input = parse_initial_input(&Value::from("$alice @* body"), &roster());
        assert_leader("$alice @* body", &input);
    }

    #[test]
    fn plain_text_goes_to_the_leader() {
        let input = parse_initial_input(&Value::from("plan the sprint"), &roster());
        assert_leader("plan the sprint", &input);
    }

    #[test]
    fn god_view_prefix_goes_to_the_leader_verbatim() {
        let input = parse_initial_input(&Value::from("# status?"), &roster());
        assert_leader("# status?", &input);
    }

    #[test]
    fn outer_whitespace_is_trimmed() {
        let input = parse_initial_input(&Value::from("  @dev-1   start  "), &roster());
        assert_eq!(only_delivery(&input).body, "start");
        let input = parse_initial_input(&Value::from("  # status?  "), &roster());
        assert_leader("# status?", &input);
    }

    #[test]
    fn empty_roster_routes_everything_to_the_leader() {
        for query in ["@dev-1 body", "$alice @dev-1 body"] {
            let input = parse_initial_input(&Value::from(query), &[]);
            assert_leader(query, &input);
        }
        // Without a recipient the human agent's own loop needs no roster row.
        let input = parse_initial_input(&Value::from("$alice body"), &[]);
        assert_eq!(only_delivery(&input).human_name, "alice");
    }

    #[test]
    fn object_query_forms_are_probed_in_contract_order() {
        for value in [
            json!({"query": "@dev-1 body"}),
            json!({"message": "@dev-1 body"}),
            json!({"query": "", "message": "@dev-1 body"}),
            json!({"input": "@dev-1 body"}),
            json!({"input": {"query": "@dev-1 body"}}),
            json!({"query": "@dev-1 body", "message": "ignored"}),
        ] {
            let input = parse_initial_input(&value, &roster());
            assert_eq!(
                only_delivery(&input),
                HumanAgentMessage {
                    human_name: USER_PSEUDO_MEMBER_NAME.to_owned(),
                    target: Some("dev-1".to_owned()),
                    body: "body".to_owned(),
                    sender: Some(USER_PSEUDO_MEMBER_NAME.to_owned()),
                },
                "value {value}"
            );
            assert_eq!(input.leader_query, None, "value {value}");
        }
    }

    #[test]
    fn object_forms_carry_the_leader_query() {
        for value in [
            json!({"query": "plan"}),
            json!({"message": "plan"}),
            json!({"input": "plan"}),
            json!({"input": {"query": "plan"}}),
        ] {
            let input = parse_initial_input(&value, &roster());
            assert_leader("plan", &input);
        }
    }

    #[test]
    fn values_without_text_yield_an_empty_result() {
        for value in [
            json!({}),
            json!({"query": ""}),
            json!({"query": null}),
            json!({"query": 7}),
            json!({"input": {}}),
            json!({"input": {"query": ""}}),
            json!({"input": 7}),
            Value::Null,
            json!(42),
            json!(true),
            json!([]),
            Value::from(""),
            Value::from("   "),
        ] {
            let input = parse_initial_input(&value, &roster());
            assert_eq!(input, InitialInput::default(), "value {value}");
        }
    }

    #[test]
    fn initial_input_default_is_empty() {
        let input = InitialInput::default();
        assert_eq!(input.leader_query, None);
        assert_eq!(input.deliveries.len(), 0);
    }
}
