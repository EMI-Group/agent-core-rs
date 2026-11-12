//! The queue envelope and the handler seam.
//!
//! Python parity: `QueueMessage` and `AsyncMessageHandler` in
//! `openjiuwen/core/runner/message_queue_base.py`. The Python subclasses
//! `InvokeQueueMessage` / `StreamQueueMessage` carried a live `asyncio.Future`;
//! in this port the reply channel belongs to the transport
//! ([`crate::mq::InMemoryMessageQueue::request`]) and the envelope stays plain
//! serializable data, so a broker backend can ship it over the wire unchanged.

use openjiuwen_common::error::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// The `error_code` of a message that carries no error (`StatusCode.SUCCESS`).
pub const SUCCESS: i32 = 0;

/// One message travelling through the queue.
///
/// The payload is an opaque [`Value`]: the queue never inspects it, and the
/// handler interprets it. `error_code` / `error_msg` mirror Python's
/// `StatusCode` pair (`SUCCESS` = `0`); [`is_error`](Self::is_error) is the only
/// interpretation the queue itself makes.
///
/// ```
/// use openjiuwen_runner::mq::{QueueMessage, SUCCESS};
/// use serde_json::json;
///
/// let message = QueueMessage::new(json!({"query": "hi"}));
/// assert!(!message.is_error());
/// assert!(!message.message_id.is_empty());
/// assert_eq!(message.error_code, SUCCESS);
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct QueueMessage {
    /// Unique id of the message; assigned by [`new`](Self::new), empty by default.
    pub message_id: String,
    /// Opaque message body (`None`/`null` serializes as `null`, Python parity).
    pub payload: Value,
    /// `SUCCESS` (`0`) for a normal message, a status code otherwise.
    pub error_code: i32,
    /// Human-readable error description; empty when [`error_code`](Self::error_code)
    /// is [`SUCCESS`].
    pub error_msg: String,
}

impl Default for QueueMessage {
    /// The Python defaults: no id, null payload, [`SUCCESS`], empty message.
    fn default() -> Self {
        Self {
            message_id: String::new(),
            payload: Value::Null,
            error_code: SUCCESS,
            error_msg: String::new(),
        }
    }
}

impl QueueMessage {
    /// Builds a message for `payload` with a fresh UUID v4 `message_id`.
    #[must_use]
    pub fn new(payload: Value) -> Self {
        Self {
            message_id: openjiuwen_common::utils::id::uuid_v4(),
            payload,
            ..Self::default()
        }
    }

    /// Overrides the message id (a caller-supplied correlation id).
    #[must_use]
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.message_id = id.into();
        self
    }

    /// Marks the message as failed with `code` and `message`.
    ///
    /// The payload is kept: Python's transport put the original request body in
    /// the reply envelope and described the failure in the status fields.
    #[must_use]
    pub fn with_error(mut self, code: i32, message: impl Into<String>) -> Self {
        self.error_code = code;
        self.error_msg = message.into();
        self
    }

    /// Returns `true` when [`error_code`](Self::error_code) is not [`SUCCESS`].
    #[must_use]
    pub fn is_error(&self) -> bool {
        self.error_code != SUCCESS
    }
}

/// Handles one delivery of a topic.
///
/// Implementations are shared across the topic worker and the callers of
/// [`MessageQueue::request`](crate::mq::MessageQueue::request), hence
/// `Send + Sync`. The queue holds at most one handler per topic and invokes it
/// from the topic's single worker task, so calls are serialized per topic.
#[async_trait::async_trait]
pub trait MessageHandler: Send + Sync {
    /// Handles one delivery and returns the request/reply response.
    ///
    /// Return [`Value::Null`] for a publish/subscribe delivery: its response is
    /// discarded. A returned error is logged and dropped on the publish path and
    /// forwarded unchanged to the `request` caller on the request path.
    ///
    /// # Errors
    ///
    /// Whatever the handler implementation fails with; the queue never wraps or
    /// rewrites it.
    async fn handle(&self, message: QueueMessage) -> Result<Value>;
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]

    use super::*;
    use serde_json::json;

    #[test]
    fn default_is_an_empty_success_message() {
        let message = QueueMessage::default();
        assert!(message.message_id.is_empty());
        assert_eq!(message.payload, Value::Null);
        assert_eq!(message.error_code, SUCCESS);
        assert!(message.error_msg.is_empty());
        assert!(!message.is_error());
    }

    #[test]
    fn new_assigns_unique_uuid_ids() {
        let first = QueueMessage::new(json!({"seq": 1}));
        let second = QueueMessage::new(json!({"seq": 2}));
        assert_ne!(first.message_id, second.message_id);
        assert_eq!(first.message_id.len(), 36, "hyphenated uuid v4");
        assert_eq!(first.payload, json!({"seq": 1}));
        assert!(!first.is_error());
    }

    #[test]
    fn builders_override_id_and_error() {
        let message = QueueMessage::new(json!({"query": "hi"}))
            .with_id("corr-1")
            .with_error(500, "boom");
        assert_eq!(message.message_id, "corr-1");
        assert_eq!(message.error_code, 500);
        assert_eq!(message.error_msg, "boom");
        assert!(message.is_error());
        assert_eq!(message.payload, json!({"query": "hi"}), "payload is kept");
    }

    #[test]
    fn message_round_trips_through_json() {
        let message = QueueMessage::new(json!({"items": [1, 2]})).with_id("m-1");
        let text = serde_json::to_string(&message).expect("serialize");
        let back: QueueMessage = serde_json::from_str(&text).expect("deserialize");
        assert_eq!(back, message);
    }

    #[test]
    fn unknown_fields_are_tolerated_and_missing_fields_default() {
        let value = json!({"message_id": "m-1", "payload": 7, "future_field": true});
        let message: QueueMessage = serde_json::from_value(value).expect("deserialize");
        assert_eq!(message.message_id, "m-1");
        assert_eq!(message.payload, json!(7));
        assert_eq!(message.error_code, SUCCESS);
        assert!(message.error_msg.is_empty());
    }
}
