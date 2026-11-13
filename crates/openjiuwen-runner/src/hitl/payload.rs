//! The inbound human payload a host delivers to an active run.
//!
//! [`InteractKind`] is the Rust form of Python's four interaction message
//! classes (`GodViewMessage`, `OperatorMessage`, `HumanAgentMessage`,
//! `GroupChatMessage`) plus a plain [`InteractKind::Text`] shorthand for a bare
//! string, which is what Python's `interact_agent_team` accepts as a god-view
//! message.
//!
//! [`InteractPayload`] is deliberately open: `body` carries the channel-specific
//! JSON verbatim, `text` the human-visible rendering, `sender` who spoke, and
//! `correlation_id` ties a reply to an earlier message. The runner does not
//! interpret the body; the run path wakes with it.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// The channel an inbound human payload arrives on.
///
/// Serialized as `snake_case` (`god_view`, `operator`, `human_agent`,
/// `group_chat`, `text`) so a host UI can round-trip a payload through JSON.
/// [`InteractKind::Text`] is the default: a bare message needs no channel
/// context.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InteractKind {
    /// The omnipotent god-view channel: a human observes and steers the run.
    GodView,
    /// An operator instruction addressed to the running team/agent.
    Operator,
    /// A message from a human agent participating in the team.
    HumanAgent,
    /// A group-chat message broadcast to every participant.
    GroupChat,
    /// A plain text message without a channel-specific meaning.
    #[default]
    Text,
}

/// One inbound human payload for an active run.
///
/// Every field is optional except the channel: `body` is the channel-specific
/// JSON, `text` the human-visible rendering, `sender` the author, and
/// `correlation_id` links a reply to an earlier message. Serialized with
/// `snake_case` field names, missing fields defaulted, and unknown fields
/// tolerated, so a larger wire envelope can carry a payload verbatim.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct InteractPayload {
    /// The channel this payload arrived on.
    pub kind: InteractKind,
    /// Channel-specific payload; the hub and the run pass it through verbatim.
    pub body: Value,
    /// Human-visible text of the message, when the host rendered one.
    pub text: Option<String>,
    /// Author of the message (`"alice"`, `"operator-7"`, ...), when known.
    pub sender: Option<String>,
    /// Id of the message this one replies to, when it is a reply.
    pub correlation_id: Option<String>,
}

impl InteractPayload {
    /// Builds an empty payload on `kind` with a null body and no text.
    #[must_use]
    pub fn new(kind: InteractKind) -> Self {
        Self {
            kind,
            ..Self::default()
        }
    }

    /// Builds a [`InteractKind::Text`] payload carrying `text`.
    #[must_use]
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            kind: InteractKind::Text,
            text: Some(text.into()),
            ..Self::default()
        }
    }

    /// Builds a payload on `kind` whose body is `body`.
    #[must_use]
    pub fn body(kind: InteractKind, body: Value) -> Self {
        Self {
            kind,
            body,
            ..Self::default()
        }
    }

    /// Sets the author.
    #[must_use]
    pub fn with_sender(mut self, sender: impl Into<String>) -> Self {
        self.sender = Some(sender.into());
        self
    }

    /// Sets the correlation id (the message this one replies to).
    #[must_use]
    pub fn with_correlation(mut self, correlation_id: impl Into<String>) -> Self {
        self.correlation_id = Some(correlation_id.into());
        self
    }

    /// Sets the human-visible text.
    #[must_use]
    pub fn with_text(mut self, text: impl Into<String>) -> Self {
        self.text = Some(text.into());
        self
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use serde_json::json;

    #[test]
    fn kinds_serialize_as_snake_case() {
        assert_eq!(
            serde_json::to_value(InteractKind::GodView).unwrap(),
            json!("god_view")
        );
        assert_eq!(
            serde_json::to_value(InteractKind::HumanAgent).unwrap(),
            json!("human_agent")
        );
        assert_eq!(
            serde_json::to_value(InteractKind::GroupChat).unwrap(),
            json!("group_chat")
        );
        assert_eq!(
            serde_json::to_value(InteractKind::Text).unwrap(),
            json!("text")
        );
        assert_eq!(InteractKind::default(), InteractKind::Text);
        assert_eq!(
            serde_json::from_value::<InteractKind>(json!("operator")).unwrap(),
            InteractKind::Operator
        );
    }

    #[test]
    fn payload_json_has_the_documented_shape() {
        let value = serde_json::to_value(InteractPayload::text("hello")).unwrap();
        assert_eq!(
            value,
            json!({
                "kind": "text",
                "body": null,
                "text": "hello",
                "sender": null,
                "correlation_id": null,
            })
        );

        let empty = serde_json::to_value(InteractPayload::new(InteractKind::Operator)).unwrap();
        assert_eq!(empty["kind"], json!("operator"));
        assert_eq!(empty["body"], Value::Null);
        assert_eq!(empty["text"], Value::Null);
        assert_eq!(empty["sender"], Value::Null);
        assert_eq!(empty["correlation_id"], Value::Null);
    }

    #[test]
    fn payload_defaults_missing_fields_and_tolerates_unknown_ones() {
        let parsed: InteractPayload =
            serde_json::from_value(json!({"kind": "group_chat", "future": 1})).unwrap();
        assert_eq!(parsed.kind, InteractKind::GroupChat);
        assert_eq!(parsed.body, Value::Null);
        assert!(parsed.text.is_none());

        let bare: InteractPayload = serde_json::from_value(json!({})).unwrap();
        assert_eq!(bare, InteractPayload::default());
        assert_eq!(bare.kind, InteractKind::Text);
    }

    #[test]
    fn builders_attach_the_body_sender_correlation_and_text() {
        let payload = InteractPayload::body(InteractKind::Operator, json!({"command": "pause"}))
            .with_sender("alice")
            .with_correlation("msg-1")
            .with_text("please pause");

        assert_eq!(payload.kind, InteractKind::Operator);
        assert_eq!(payload.body, json!({"command": "pause"}));
        assert_eq!(payload.sender.as_deref(), Some("alice"));
        assert_eq!(payload.correlation_id.as_deref(), Some("msg-1"));
        assert_eq!(payload.text.as_deref(), Some("please pause"));

        let original = payload.clone();
        let text = serde_json::to_string(&original).unwrap();
        assert_eq!(
            serde_json::from_str::<InteractPayload>(&text).unwrap(),
            original
        );
    }
}
