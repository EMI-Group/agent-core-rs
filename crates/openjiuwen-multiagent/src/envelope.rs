//! Message envelopes carried by the team message bus.
//!
//! Mirrors the Python `core.multi_agent.message_bus.MessageEnvelope` frozen dataclass: a JSON
//! payload plus routing metadata. A `Some` [`recipient`](MessageEnvelope::recipient) marks a
//! point-to-point message, a `Some` [`topic_id`](MessageEnvelope::topic_id) marks a pub/sub
//! message.

use std::collections::BTreeMap;

use openjiuwen_common::utils::id::uuid_v4;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// A transport envelope: one message payload plus its routing metadata.
///
/// The type round-trips through JSON, tolerates unknown fields, and treats every field as
/// optional on input. A missing field is filled from [`Default`], which allocates a fresh
/// message id, so an empty JSON object still yields a usable envelope.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct MessageEnvelope {
    /// Unique message id, a hyphenated UUID v4.
    ///
    /// Generated on construction; used for tracing and correlation.
    pub message_id: String,
    /// The JSON payload.
    pub message: Value,
    /// Optional sender agent id, used for message tracing.
    pub sender: Option<String>,
    /// Optional recipient agent id; `Some` marks a point-to-point message.
    pub recipient: Option<String>,
    /// Optional pub/sub topic id; `Some` marks a pub/sub message.
    pub topic_id: Option<String>,
    /// Optional team session id used to select the session on delivery.
    pub session_id: Option<String>,
    /// Free-form metadata carried alongside the payload.
    ///
    /// An explicit `null` is read as an empty map.
    #[serde(default, deserialize_with = "deserialize_metadata")]
    pub metadata: BTreeMap<String, Value>,
}

impl Default for MessageEnvelope {
    fn default() -> Self {
        Self {
            message_id: uuid_v4(),
            message: Value::Null,
            sender: None,
            recipient: None,
            topic_id: None,
            session_id: None,
            metadata: BTreeMap::new(),
        }
    }
}

impl MessageEnvelope {
    /// Creates an envelope around `message` with a fresh id and no routing information.
    #[must_use]
    pub fn new(message: Value) -> Self {
        Self {
            message,
            ..Self::default()
        }
    }

    /// Creates a point-to-point envelope from `sender` to `recipient`.
    #[must_use]
    pub fn p2p(message: Value, sender: impl Into<String>, recipient: impl Into<String>) -> Self {
        Self {
            sender: Some(sender.into()),
            recipient: Some(recipient.into()),
            ..Self::new(message)
        }
    }

    /// Creates a pub/sub envelope published by `sender` on `topic_id`.
    #[must_use]
    pub fn pubsub(message: Value, sender: impl Into<String>, topic_id: impl Into<String>) -> Self {
        Self {
            sender: Some(sender.into()),
            topic_id: Some(topic_id.into()),
            ..Self::new(message)
        }
    }

    /// Returns `true` when the envelope is addressed to a single recipient.
    #[must_use]
    pub const fn is_p2p(&self) -> bool {
        self.recipient.is_some()
    }

    /// Returns `true` when the envelope targets a pub/sub topic.
    #[must_use]
    pub const fn is_pubsub(&self) -> bool {
        self.topic_id.is_some()
    }

    /// Sets the team session id used to resolve the delivery session.
    #[must_use]
    pub fn with_session(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    /// Adds one metadata entry.
    #[must_use]
    pub fn with_metadata(mut self, key: impl Into<String>, value: Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Extends the metadata with every entry of `metadata`.
    #[must_use]
    pub fn with_metadata_map(mut self, metadata: BTreeMap<String, Value>) -> Self {
        self.metadata.extend(metadata);
        self
    }
}

/// Reads an optional metadata map, treating `null` as empty.
fn deserialize_metadata<'de, D>(deserializer: D) -> Result<BTreeMap<String, Value>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Ok(Option::<BTreeMap<String, Value>>::deserialize(deserializer)?.unwrap_or_default())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use serde_json::json;

    fn assert_uuid_v4_hyphenated(id: &str) {
        assert_eq!(id.len(), 36, "uuid must be 36 chars: {id}");
        let bytes = id.as_bytes();
        for idx in [8, 13, 18, 23] {
            assert_eq!(bytes[idx], b'-', "missing hyphen at {idx}: {id}");
        }
        assert_eq!(bytes[14], b'4', "uuid must be version 4: {id}");
        assert!(
            matches!(bytes[19], b'8' | b'9' | b'a' | b'b'),
            "uuid must be RFC 4122 variant: {id}"
        );
    }

    #[test]
    fn new_has_fresh_id_and_no_routing() {
        let first = MessageEnvelope::new(json!({"hello": "world"}));
        let second = MessageEnvelope::new(Value::Null);
        assert_uuid_v4_hyphenated(&first.message_id);
        assert_uuid_v4_hyphenated(&second.message_id);
        assert_ne!(first.message_id, second.message_id);
        assert_eq!(first.message, json!({"hello": "world"}));
        assert_eq!(first.sender, None);
        assert_eq!(first.recipient, None);
        assert_eq!(first.topic_id, None);
        assert_eq!(first.session_id, None);
        assert!(first.metadata.is_empty());
        assert!(!first.is_p2p());
        assert!(!first.is_pubsub());
    }

    #[test]
    fn default_uses_fresh_uuid() {
        let first = MessageEnvelope::default();
        let second = MessageEnvelope::default();
        assert_uuid_v4_hyphenated(&first.message_id);
        assert_ne!(first.message_id, second.message_id);
        assert_eq!(first.message, Value::Null);
        assert!(first.metadata.is_empty());
    }

    #[test]
    fn p2p_sets_sender_and_recipient_only() {
        let envelope = MessageEnvelope::p2p(json!(1), "alice", "bob");
        assert_eq!(envelope.sender.as_deref(), Some("alice"));
        assert_eq!(envelope.recipient.as_deref(), Some("bob"));
        assert_eq!(envelope.topic_id, None);
        assert_eq!(envelope.session_id, None);
        assert_eq!(envelope.message, json!(1));
        assert_uuid_v4_hyphenated(&envelope.message_id);
        assert!(envelope.is_p2p());
        assert!(!envelope.is_pubsub());
    }

    #[test]
    fn pubsub_sets_sender_and_topic_only() {
        let envelope = MessageEnvelope::pubsub(json!("payload"), "alice", "team__pubsub__");
        assert_eq!(envelope.sender.as_deref(), Some("alice"));
        assert_eq!(envelope.topic_id.as_deref(), Some("team__pubsub__"));
        assert_eq!(envelope.recipient, None);
        assert_eq!(envelope.message, json!("payload"));
        assert!(envelope.is_pubsub());
        assert!(!envelope.is_p2p());
    }

    #[test]
    fn builders_set_session_and_metadata() {
        let envelope = MessageEnvelope::p2p(json!({"a": 1}), "alice", "bob")
            .with_session("session-1")
            .with_metadata("trace", json!("t"))
            .with_metadata_map(BTreeMap::from([("k".to_owned(), json!(2))]));
        assert_eq!(envelope.session_id.as_deref(), Some("session-1"));
        assert_eq!(envelope.metadata.get("trace"), Some(&json!("t")));
        assert_eq!(envelope.metadata.get("k"), Some(&json!(2)));
        assert_eq!(envelope.metadata.len(), 2);
    }

    #[test]
    fn serde_round_trip() {
        let envelope = MessageEnvelope::p2p(json!({"nested": [1, 2, 3]}), "alice", "bob")
            .with_session("s-1")
            .with_metadata("trace", json!({"id": "abc"}));
        let encoded = serde_json::to_string(&envelope).expect("serialize");
        let decoded: MessageEnvelope = serde_json::from_str(&encoded).expect("deserialize");
        assert_eq!(decoded, envelope);
        assert_eq!(decoded.message_id, envelope.message_id);
    }

    #[test]
    fn empty_object_deserializes_to_default_with_id() {
        let decoded: MessageEnvelope = serde_json::from_str("{}").expect("deserialize");
        assert_ne!(decoded.message_id, "");
        assert_uuid_v4_hyphenated(&decoded.message_id);
        assert_eq!(decoded.message, Value::Null);
        assert_eq!(decoded.sender, None);
        assert_eq!(decoded.recipient, None);
        assert_eq!(decoded.topic_id, None);
        assert_eq!(decoded.session_id, None);
        assert!(decoded.metadata.is_empty());
    }

    #[test]
    fn unknown_fields_are_tolerated() {
        let decoded: MessageEnvelope = serde_json::from_str(
            r#"{"message_id":"m-1","message":{"x":1},"recipient":"bob","unknown":{"a":1}}"#,
        )
        .expect("deserialize");
        assert_eq!(decoded.message_id, "m-1");
        assert_eq!(decoded.message, json!({"x": 1}));
        assert_eq!(decoded.recipient.as_deref(), Some("bob"));
    }

    #[test]
    fn null_optional_fields_are_tolerated() {
        let decoded: MessageEnvelope = serde_json::from_str(
            r#"{"message_id":"m-2","message":null,"sender":null,"recipient":null,
                "topic_id":null,"session_id":null,"metadata":null}"#,
        )
        .expect("deserialize");
        assert_eq!(decoded.message_id, "m-2");
        assert_eq!(decoded.message, Value::Null);
        assert_eq!(decoded.sender, None);
        assert_eq!(decoded.recipient, None);
        assert_eq!(decoded.topic_id, None);
        assert_eq!(decoded.session_id, None);
        assert!(decoded.metadata.is_empty());
    }

    #[test]
    fn omitted_optional_fields_keep_the_default_id() {
        let decoded: MessageEnvelope =
            serde_json::from_str(r#"{"recipient":"bob"}"#).expect("deserialize");
        assert_uuid_v4_hyphenated(&decoded.message_id);
        assert_eq!(decoded.recipient.as_deref(), Some("bob"));
        assert_eq!(decoded.topic_id, None);
        assert!(decoded.is_p2p());
    }
}
