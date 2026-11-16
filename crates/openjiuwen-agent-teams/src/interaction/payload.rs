//! Typed interact payloads and the uniform delivery outcome.
//!
//! Python parity source: `openjiuwen/agent_teams/interaction/payload.py` (v0.1.20).
//! [`InteractPayload`] is the discriminated union the runtime accepts from every external
//! channel; [`DeliverResult`] is what the runtime hands back so callers never special-case a
//! channel's return type.
//!
//! The wire discriminator is `kind` (Python's payload classes are not serde types, so the tag
//! is a Rust-side detail pinned here): `god_view`, `operator`, `human_agent`,
//! `human_agent_tool_call`, `group_chat`, `external_team`.
//!
//! Human-agent channels are the only ones gated by HITL — see
//! [`InteractPayload::requires_hitt`]. Everything else is accepted whenever the team is active.
//!
//! # Example
//!
//! ```
//! use openjiuwen_agent_teams::interaction::{DeliverResult, GodViewMessage, InteractPayload};
//!
//! let payload = InteractPayload::GodView(GodViewMessage {
//!     body: "please summarise the board".to_owned(),
//!     sender: None,
//! });
//! assert_eq!(payload.kind(), "god_view");
//! assert!(!payload.requires_hitt());
//! assert_eq!(
//!     DeliverResult::gate_closed().reason.as_deref(),
//!     Some("gate_closed")
//! );
//! ```
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Speak directly to the team leader's `DeepAgent` (the historical `invoke` channel).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct GodViewMessage {
    /// Message content delivered to the leader.
    pub body: String,
    /// Identity of the external caller when the host application knows it.
    pub sender: Option<String>,
}

/// Speak as the external user, optionally addressing one member.
///
/// An operator message is the `#`/bare-text channel: `target = None` broadcasts to the team,
/// while `target = Some(member)` posts a point-to-point bus message from the `user`
/// pseudo-member.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct OperatorMessage {
    /// Member the message is addressed to; `None` broadcasts to the team.
    pub target: Option<String>,
    /// Message content.
    pub body: String,
    /// Identity of the external caller when the host application knows it.
    pub sender: Option<String>,
}

/// Speak as a registered human-agent (avatar) team member.
///
/// Routes through the human-agent inbox and is gated by HITL; a passive human member uses
/// [`HumanAgentToolCall`] for structured actions instead.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct HumanAgentMessage {
    /// Member name of the human agent the message belongs to.
    pub human_name: String,
    /// Target member; `None` drives the human agent's own avatar loop.
    pub target: Option<String>,
    /// Message content.
    pub body: String,
    /// Member name of the human agent speaking, when the caller supplied it.
    pub sender: Option<String>,
}

/// Relay one team tool call on behalf of a passive human member.
///
/// A passive human has no avatar, so the runtime executes the call verbatim under
/// `human_name`'s identity (assignee locks, reviewer guards and sender-bound message rows all
/// still apply). This channel is gated by HITL.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct HumanAgentToolCall {
    /// Member name of the passive human member acting.
    pub human_name: String,
    /// Team tool name, e.g. `member_complete_task`.
    pub tool_name: String,
    /// Tool arguments, matching the tool's declared input schema one-to-one.
    pub tool_args: Value,
}

/// Public group-chat input; an empty `mentions` list only archives the message.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct GroupChatMessage {
    /// Message content.
    pub body: String,
    /// Identity of the external caller when the host application knows it.
    pub sender: Option<String>,
    /// Members the message explicitly notifies; empty archives without notification.
    pub mentions: Vec<String>,
}

/// A standard team event received through the external interact channel.
///
/// Data-only: this build ships no distributed transport, so the runtime rejects a payload of this
/// kind as `unsupported:external_team_transport` — `DeliverResult::unsupported` with that token,
/// applied by `TeamAgent::deliver` in `src/runtime/team_agent/lifecycle.rs`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ExternalTeamEvent {
    /// Raw team event envelope received from the external transport.
    pub event: Value,
}

/// Discriminated union of the interact payload shapes the runtime accepts.
///
/// Serialized with a `kind` tag; every variant carries a single struct payload, so the tag and
/// the struct's fields sit at the same JSON level.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum InteractPayload {
    /// Speak to the leader's `DeepAgent`.
    GodView(GodViewMessage),
    /// Speak as the external user, optionally to one member.
    Operator(OperatorMessage),
    /// Speak as a registered human-agent member.
    HumanAgent(HumanAgentMessage),
    /// Relay one tool call as a passive human member.
    HumanAgentToolCall(HumanAgentToolCall),
    /// Public group-chat input.
    GroupChat(GroupChatMessage),
    /// External team event from a distributed transport.
    ExternalTeam(ExternalTeamEvent),
}

impl InteractPayload {
    /// Returns this payload's wire discriminator (`kind`).
    ///
    /// The values match the serde tag exactly, so a payload round-tripped through JSON keeps
    /// its kind.
    #[must_use]
    pub const fn kind(&self) -> &'static str {
        match self {
            Self::GodView(_) => "god_view",
            Self::Operator(_) => "operator",
            Self::HumanAgent(_) => "human_agent",
            Self::HumanAgentToolCall(_) => "human_agent_tool_call",
            Self::GroupChat(_) => "group_chat",
            Self::ExternalTeam(_) => "external_team",
        }
    }

    /// Whether delivering this payload requires HITL to be enabled on the team.
    ///
    /// Only the two direct-control channels (a human avatar speaking, or a passive human
    /// relaying a tool call) need a registered human-agent member; every other channel works
    /// on a team without one.
    #[must_use]
    pub const fn requires_hitt(&self) -> bool {
        matches!(self, Self::HumanAgent(_) | Self::HumanAgentToolCall(_))
    }
}

/// Outcome of a single payload delivery.
///
/// On success `ok` is `true` and `message_id` carries the id assigned to the bus message
/// (`None` for channels that produce no message, e.g. driving the leader or queueing at a
/// gate).
///
/// On failure `ok` is `false` and `reason` is a short, stable token the caller may surface:
/// `not_active` and `gate_closed` are pinned by the gate contract; other channels use
/// `unsupported:<capability>`, `unknown_member:<target>`, `send_failed:<target>` and similar
/// tokens ported from Python.
///
/// `output` and `data` are filled only by a successfully relayed [`HumanAgentToolCall`]: the
/// tool's model-facing result text and its raw data payload, so an external protocol can show
/// the human the effect of the call synchronously.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct DeliverResult {
    /// Whether the payload was delivered.
    pub ok: bool,
    /// Id assigned to the delivered bus message, when the channel produced one.
    pub message_id: Option<String>,
    /// Stable failure token when `ok` is `false`.
    pub reason: Option<String>,
    /// Tool result text when the payload was a relayed tool call.
    pub output: Value,
    /// Raw tool data when the payload was a relayed tool call.
    pub data: Value,
}

impl DeliverResult {
    /// Builds a success, optionally carrying the delivered message id.
    #[must_use]
    pub fn ok(message_id: Option<String>) -> Self {
        Self {
            ok: true,
            message_id,
            ..Self::default()
        }
    }

    /// Builds a failure carrying a stable reason token.
    #[must_use]
    pub fn failed(reason: impl Into<String>) -> Self {
        Self {
            ok: false,
            reason: Some(reason.into()),
            ..Self::default()
        }
    }

    /// The team is not running an active run cycle, so the payload was not accepted.
    #[must_use]
    pub fn not_active() -> Self {
        Self::failed("not_active")
    }

    /// The team's interaction gate is closed (the run cycle is exiting), so the payload was
    /// rejected.
    #[must_use]
    pub fn gate_closed() -> Self {
        Self::failed("gate_closed")
    }

    /// The channel needs a capability this build does not provide.
    ///
    /// The reason token is `unsupported:<reason>` so callers can pattern-match the prefix.
    #[must_use]
    pub fn unsupported(reason: impl Into<String>) -> Self {
        Self::failed(format!("unsupported:{}", reason.into()))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use serde_json::json;

    /// One instance of every payload variant, used by the round-trip test.
    fn all_payloads() -> Vec<InteractPayload> {
        vec![
            InteractPayload::GodView(GodViewMessage {
                body: "hello leader".to_owned(),
                sender: Some("ops".to_owned()),
            }),
            InteractPayload::Operator(OperatorMessage {
                target: Some("dev-1".to_owned()),
                body: "start".to_owned(),
                sender: None,
            }),
            InteractPayload::HumanAgent(HumanAgentMessage {
                human_name: "human_agent".to_owned(),
                target: None,
                body: "on it".to_owned(),
                sender: Some("alice".to_owned()),
            }),
            InteractPayload::HumanAgentToolCall(HumanAgentToolCall {
                human_name: "passive-1".to_owned(),
                tool_name: "member_complete_task".to_owned(),
                tool_args: json!({"task_id": "t1"}),
            }),
            InteractPayload::GroupChat(GroupChatMessage {
                body: "standup in 5".to_owned(),
                sender: Some("pm".to_owned()),
                mentions: vec!["dev-1".to_owned()],
            }),
            InteractPayload::ExternalTeam(ExternalTeamEvent {
                event: json!({"event_type": "team_completed", "payload": {}}),
            }),
        ]
    }

    #[test]
    fn kind_matches_serde_tag_for_every_variant() {
        for payload in all_payloads() {
            let encoded = serde_json::to_value(&payload).unwrap();
            assert_eq!(
                encoded.get("kind").and_then(Value::as_str),
                Some(payload.kind())
            );
        }
    }

    #[test]
    fn payloads_round_trip_through_json() {
        for payload in all_payloads() {
            let encoded = serde_json::to_string(&payload).unwrap();
            let decoded: InteractPayload = serde_json::from_str(&encoded).unwrap();
            assert_eq!(decoded, payload);
        }
    }

    #[test]
    fn unknown_fields_are_tolerated() {
        let decoded: InteractPayload = serde_json::from_value(json!({
            "kind": "god_view",
            "body": "hello",
            "future_field": 42,
        }))
        .unwrap();
        assert_eq!(
            decoded,
            InteractPayload::GodView(GodViewMessage {
                body: "hello".to_owned(),
                sender: None,
            })
        );
    }

    #[test]
    fn human_agent_channels_require_hitt() {
        for payload in all_payloads() {
            let expected = matches!(
                payload,
                InteractPayload::HumanAgent(_) | InteractPayload::HumanAgentToolCall(_)
            );
            assert_eq!(payload.requires_hitt(), expected, "kind {}", payload.kind());
        }
    }

    #[test]
    fn deliver_result_constructors_pin_reason_tokens() {
        let ok = DeliverResult::ok(Some("msg_1".to_owned()));
        assert!(ok.ok);
        assert_eq!(ok.message_id.as_deref(), Some("msg_1"));
        assert_eq!(ok.reason, None);
        assert_eq!(ok.output, Value::Null);
        assert_eq!(ok.data, Value::Null);

        assert_eq!(
            DeliverResult::not_active().reason.as_deref(),
            Some("not_active")
        );
        assert_eq!(
            DeliverResult::gate_closed().reason.as_deref(),
            Some("gate_closed")
        );
        assert_eq!(
            DeliverResult::unsupported("pyzmq").reason.as_deref(),
            Some("unsupported:pyzmq")
        );
        assert!(!DeliverResult::failed("send_failed:dev-1").ok);
    }

    #[test]
    fn deliver_result_round_trips_through_json() {
        let result = DeliverResult {
            ok: true,
            message_id: Some("msg_1".to_owned()),
            reason: None,
            output: json!("done"),
            data: json!({"status": "completed"}),
        };
        let encoded = serde_json::to_string(&result).unwrap();
        let decoded: DeliverResult = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, result);
    }
}
