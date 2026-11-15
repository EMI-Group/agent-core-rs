//! Steering: user messages injected into a running agent at safe points.
//!
//! Ports the steering-queue half of Python's `ReActAgent`: a host pushes a
//! message while a run is in flight and the loop drains it between steps
//! (`AgentCallbackContext::push_steering` / `drain_steering` in
//! `openjiuwen/core/single_agent/rail/base.py`, consumed by
//! `_drain_steering_batch` in `react_agent.py`).
//!
//! The queue is a plain, infallible, thread-safe FIFO behind an [`Arc`], so a
//! host thread, a rail, or a tool can push while the loop is awaiting a model
//! call. The loop itself is wired elsewhere; it drains the backlog through
//! [`SteeringQueue::drain_limit`] at its steering-drain point, so a
//! rail-supplied cap is honoured. [`SteeringQueue::drain_prompt`] remains the
//! standalone helper that renders a drained backlog into one text block.
//!
//! ```
//! use openjiuwen_agent::steering::{SteeringMessage, SteeringQueue};
//!
//! let queue = SteeringQueue::new();
//! queue.push(SteeringMessage::new("stop after this step").with_source("host"));
//! assert_eq!(queue.len(), 1);
//! let prompt = queue.drain_prompt().expect("queued message");
//! assert_eq!(prompt, "stop after this step");
//! assert!(queue.is_empty());
//! ```

use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;

use openjiuwen_foundation::BaseMessage;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Metadata key under which a [`SteeringMessage`]'s annotations travel on the
/// [`BaseMessage`] built by [`SteeringMessage::to_message`].
pub const STEERING_METADATA_KEY: &str = "_openjiuwen_steering";

/// One steering message: text a host wants the running agent to see next.
///
/// Serializes with the `snake_case` field names `text`, `source`, and
/// `metadata`, and tolerates unknown fields. `source` names the origin of the
/// message (`"host"`, `"rail"`, a teammate id, …); `metadata` keeps arbitrary
/// machine-readable annotations that are forwarded onto the produced
/// [`BaseMessage`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct SteeringMessage {
    /// The instruction text; the only field the model always sees.
    pub text: String,
    /// Optional origin tag, used for provenance and diagnostics.
    pub source: Option<String>,
    /// Free-form annotations forwarded into the produced message's metadata.
    pub metadata: BTreeMap<String, Value>,
}

impl SteeringMessage {
    /// Creates a message with the given text and no source or metadata.
    #[must_use]
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            ..Self::default()
        }
    }

    /// Sets [`SteeringMessage::source`].
    #[must_use]
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    /// Inserts one [`SteeringMessage::metadata`] entry, replacing a previous
    /// value under the same key.
    #[must_use]
    pub fn with_metadata(mut self, key: impl Into<String>, value: Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Whether the text is blank, i.e. empty or whitespace-only.
    ///
    /// Blank messages carry no instruction, so the queue drops them.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.text.trim().is_empty()
    }

    /// Renders the message as a [`openjiuwen_foundation::Role::User`] message.
    ///
    /// Non-empty [`SteeringMessage::metadata`] is attached under the
    /// [`STEERING_METADATA_KEY`] metadata key as a JSON object, so the loop's
    /// provenance markers stay machine-readable.
    #[must_use]
    pub fn to_message(&self) -> BaseMessage {
        let message = BaseMessage::user(self.text.clone());
        if self.metadata.is_empty() {
            return message;
        }
        let object = self
            .metadata
            .iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect();
        message.with_metadata(STEERING_METADATA_KEY, Value::Object(object))
    }
}

/// Thread-safe FIFO of [`SteeringMessage`]s shared with the running loop.
///
/// Cloning a queue is cheap and shares the same backlog: the host keeps one
/// clone to push while the loop holds another to drain. Blank messages are
/// dropped on ingress (and skipped defensively on egress), so `len` only counts
/// meaningful entries.
#[derive(Debug, Clone, Default)]
pub struct SteeringQueue {
    inner: Arc<Mutex<VecDeque<SteeringMessage>>>,
}

impl SteeringQueue {
    /// Creates an empty queue.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Pushes one message and returns the resulting queue length.
    ///
    /// A blank message ([`SteeringMessage::is_empty`]) is dropped and leaves the
    /// length unchanged.
    #[allow(clippy::must_use_candidate)]
    pub fn push(&self, message: SteeringMessage) -> usize {
        let mut queue = self.inner.lock();
        if !message.is_empty() {
            queue.push_back(message);
        }
        queue.len()
    }

    /// Pushes a plain text message and returns the resulting queue length.
    #[allow(clippy::must_use_candidate)]
    pub fn push_text(&self, text: impl Into<String>) -> usize {
        self.push(SteeringMessage::new(text))
    }

    /// Removes and returns every queued message, oldest first.
    ///
    /// The queue is emptied; blank messages are skipped.
    #[must_use]
    pub fn drain(&self) -> Vec<SteeringMessage> {
        let mut queue = self.inner.lock();
        queue
            .drain(..)
            .filter(|message| !message.is_empty())
            .collect()
    }

    /// Removes and returns at most `limit` queued messages, oldest first.
    ///
    /// `None` drains the whole backlog, exactly like [`SteeringQueue::drain`].
    /// `Some(n)` takes at most `max(1, n)` messages, so `Some(0)` still takes one
    /// and a non-empty queue always makes progress (Python's
    /// `drain_steering(limit)` computes `take = max(1, limit)`).
    ///
    /// Blank messages ([`SteeringMessage::is_empty`]) are skipped and do not
    /// count against the limit. Any message beyond the cap stays queued, in
    /// order, for the next [`SteeringQueue::drain`] or `drain_limit` call.
    #[must_use]
    pub fn drain_limit(&self, limit: Option<usize>) -> Vec<SteeringMessage> {
        let take = limit.map_or(usize::MAX, |n| n.max(1));
        let mut queue = self.inner.lock();
        let mut drained = Vec::new();
        while drained.len() < take {
            let Some(message) = queue.pop_front() else {
                break;
            };
            if !message.is_empty() {
                drained.push(message);
            }
        }
        drained
    }

    /// Drains the backlog as [`BaseMessage`]s ready to append to the window.
    #[must_use]
    pub fn drain_messages(&self) -> Vec<BaseMessage> {
        self.drain()
            .iter()
            .map(SteeringMessage::to_message)
            .collect()
    }

    /// Peeks at the backlog without removing anything, oldest first.
    #[must_use]
    pub fn snapshot(&self) -> Vec<SteeringMessage> {
        self.inner.lock().iter().cloned().collect()
    }

    /// Number of queued messages.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.lock().len()
    }

    /// Whether the queue holds no messages.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.lock().is_empty()
    }

    /// Drops every queued message.
    pub fn clear(&self) {
        self.inner.lock().clear();
    }

    /// Drains the backlog into one text block for the model window.
    ///
    /// Messages are rendered oldest first, one per line, joined by a blank line
    /// (`"\n\n"`); blank messages are ignored. Returns `None` when nothing was
    /// queued, so the caller can skip the injection entirely. This is a
    /// standalone render helper over [`SteeringQueue::drain`]; the `ReAct` loop
    /// drains through [`SteeringQueue::drain_limit`] instead.
    #[must_use]
    pub fn drain_prompt(&self) -> Option<String> {
        let texts: Vec<String> = self
            .drain()
            .into_iter()
            .map(|message| message.text)
            .filter(|text| !text.trim().is_empty())
            .collect();
        if texts.is_empty() {
            None
        } else {
            Some(texts.join("\n\n"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use openjiuwen_foundation::Role;

    #[test]
    fn drain_is_fifo() {
        let queue = SteeringQueue::new();
        queue.push_text("first");
        queue.push_text("second");
        queue.push_text("third");

        let drained: Vec<String> = queue.drain().into_iter().map(|m| m.text).collect();
        assert_eq!(drained, vec!["first", "second", "third"]);
    }

    #[test]
    fn drain_empties_the_queue() {
        let queue = SteeringQueue::new();
        queue.push_text("only");
        assert_eq!(queue.drain().len(), 1);
        assert!(queue.is_empty());
        assert_eq!(queue.len(), 0);
        assert_eq!(queue.drain(), []);
        assert!(queue.drain_prompt().is_none());
    }

    #[test]
    fn blank_messages_are_ignored() {
        let queue = SteeringQueue::new();
        assert_eq!(queue.push_text(""), 0);
        assert_eq!(queue.push_text("   "), 0);
        assert_eq!(queue.push_text("\n\t "), 0);
        assert_eq!(queue.push(SteeringMessage::default()), 0);
        assert!(queue.is_empty());

        assert_eq!(queue.push_text("real"), 1);
        assert_eq!(queue.snapshot().len(), 1);
        assert!(SteeringMessage::new("").is_empty());
        assert!(SteeringMessage::new(" \n ").is_empty());
        assert!(!SteeringMessage::new("hi").is_empty());
    }

    #[test]
    fn snapshot_does_not_consume() {
        let queue = SteeringQueue::new();
        queue.push_text("a");
        queue.push_text("b");

        let first = queue.snapshot();
        let second = queue.snapshot();
        assert_eq!(first, second);
        assert_eq!(first.len(), 2);
        assert_eq!(queue.len(), 2);

        let drained = queue.drain();
        assert_eq!(drained, first);
        assert_eq!(queue.snapshot(), []);
    }

    #[test]
    fn clones_share_state_across_threads() {
        let queue = SteeringQueue::new();
        let handles: Vec<_> = (0..4)
            .map(|worker| {
                let queue = queue.clone();
                std::thread::spawn(move || {
                    for index in 0..25 {
                        queue.push_text(format!("worker {worker} item {index}"));
                    }
                })
            })
            .collect();
        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(queue.len(), 100);
        let drained = queue.drain();
        assert_eq!(drained.len(), 100);
        assert!(queue.is_empty());

        // Every worker's messages survived, each worker preserving its order.
        for worker in 0..4 {
            let observed: Vec<&SteeringMessage> = drained
                .iter()
                .filter(|m| m.text.starts_with(&format!("worker {worker} ")))
                .collect();
            assert_eq!(observed.len(), 25);
            for (index, message) in observed.iter().enumerate() {
                assert_eq!(message.text, format!("worker {worker} item {index}"));
            }
        }
    }

    #[test]
    fn to_message_is_a_user_message_with_metadata() {
        let message = SteeringMessage::new("please stop")
            .with_source("host")
            .with_metadata("kind", Value::String("halt".into()))
            .with_metadata("priority", Value::from(3));

        let base = message.to_message();
        assert_eq!(base.role, Role::User);
        assert_eq!(base.text(), "please stop");

        let steering = base
            .metadata
            .get(STEERING_METADATA_KEY)
            .expect("metadata attached");
        assert_eq!(steering.get("kind"), Some(&Value::String("halt".into())));
        assert_eq!(steering.get("priority"), Some(&Value::from(3)));

        let plain = SteeringMessage::new("no metadata").to_message();
        assert_eq!(plain.role, Role::User);
        assert!(plain.metadata.is_empty());

        let messages = {
            let queue = SteeringQueue::new();
            queue.push(message);
            queue.drain_messages()
        };
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, Role::User);
        assert!(messages[0].metadata.contains_key(STEERING_METADATA_KEY));
    }

    #[test]
    fn drain_prompt_joins_messages() {
        let queue = SteeringQueue::new();
        queue.push_text("first instruction");
        queue.push(SteeringMessage::new("").with_source("rail"));
        queue.push_text("second instruction");

        assert_eq!(
            queue.drain_prompt().as_deref(),
            Some("first instruction\n\nsecond instruction")
        );
        assert!(queue.is_empty());
        assert_eq!(queue.drain_prompt(), None);
    }

    #[test]
    fn drain_limit_none_takes_the_whole_backlog() {
        let queue = SteeringQueue::new();
        queue.push_text("first");
        queue.push_text("second");
        queue.push_text("third");

        let drained: Vec<String> = queue
            .drain_limit(None)
            .into_iter()
            .map(|m| m.text)
            .collect();
        assert_eq!(drained, vec!["first", "second", "third"]);
        assert!(queue.is_empty());
    }

    #[test]
    fn drain_limit_zero_still_takes_one() {
        let queue = SteeringQueue::new();
        queue.push_text("oldest");
        queue.push_text("second");
        queue.push_text("third");

        let drained: Vec<String> = queue
            .drain_limit(Some(0))
            .into_iter()
            .map(|m| m.text)
            .collect();
        assert_eq!(drained, vec!["oldest"]);
        assert_eq!(queue.len(), 2);
        assert_eq!(
            queue
                .snapshot()
                .into_iter()
                .map(|m| m.text)
                .collect::<Vec<_>>(),
            vec!["second", "third"]
        );
    }

    #[test]
    fn drain_limit_caps_and_keeps_the_rest_in_order() {
        let queue = SteeringQueue::new();
        queue.push_text("first");
        queue.push_text("second");
        queue.push_text("third");

        let first: Vec<String> = queue
            .drain_limit(Some(2))
            .into_iter()
            .map(|m| m.text)
            .collect();
        assert_eq!(first, vec!["first", "second"]);
        assert_eq!(queue.len(), 1);

        let second: Vec<String> = queue
            .drain_limit(Some(2))
            .into_iter()
            .map(|m| m.text)
            .collect();
        assert_eq!(second, vec!["third"]);
        assert!(queue.is_empty());
        assert_eq!(queue.drain_limit(Some(2)).len(), 0);
    }

    #[test]
    fn drain_limit_oversized_takes_everything() {
        let queue = SteeringQueue::new();
        queue.push_text("a");
        queue.push_text("b");

        let drained: Vec<String> = queue
            .drain_limit(Some(10))
            .into_iter()
            .map(|m| m.text)
            .collect();
        assert_eq!(drained, vec!["a", "b"]);
        assert!(queue.is_empty());
        assert_eq!(queue.len(), 0);
    }

    #[test]
    fn drain_limit_skips_blanks_without_consuming_the_limit() {
        let queue = SteeringQueue::new();
        // Blank messages never enter the backlog, so they cannot consume the cap.
        assert_eq!(queue.push_text("   "), 0);
        queue.push_text("first");
        queue.push(SteeringMessage::default());
        queue.push_text("second");

        let drained: Vec<String> = queue
            .drain_limit(Some(2))
            .into_iter()
            .map(|m| m.text)
            .collect();
        assert_eq!(drained, vec!["first", "second"]);
        assert!(queue.is_empty());
    }

    #[test]
    fn clear_drops_everything() {
        let queue = SteeringQueue::new();
        queue.push_text("a");
        queue.push_text("b");
        queue.clear();
        assert!(queue.is_empty());
        assert_eq!(queue.len(), 0);
        assert_eq!(queue.snapshot(), []);
    }

    #[test]
    fn steering_message_serde_round_trip() {
        let message = SteeringMessage::new("hello")
            .with_source("rail")
            .with_metadata("k", Value::Bool(true));
        let json = serde_json::to_value(&message).unwrap();
        assert_eq!(json["text"], Value::String("hello".into()));
        assert_eq!(json["source"], Value::String("rail".into()));
        assert_eq!(json["metadata"]["k"], Value::Bool(true));

        let back: SteeringMessage = serde_json::from_value(json).unwrap();
        assert_eq!(back, message);

        // Unknown fields are tolerated and missing fields default.
        let sparse: SteeringMessage = serde_json::from_value(serde_json::json!({
            "text": "only text",
            "unknown": 42,
        }))
        .unwrap();
        assert_eq!(sparse, SteeringMessage::new("only text"));
    }
}
