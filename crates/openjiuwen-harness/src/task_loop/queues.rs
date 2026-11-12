//! Queued user input and the shared follow-up/steering queues of the task loop.
//!
//! Ports `openjiuwen/harness/task_loop/loop_queues.py`. Python shares one
//! `asyncio.Queue` object per queue by identity between the agent, the event
//! handler and the task-loop controller; the Rust port keeps the same sharing
//! semantics with `Arc<parking_lot::Mutex<VecDeque<_>>>` clones.
//!
//! Two planes exist:
//!
//! - the **follow-up** queue holds inputs that run as their own round after the
//!   current round finishes;
//! - the **steering** queue holds inputs the executor drains *before each model
//!   call* inside a running round.

use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// One user input waiting to be consumed by a task-loop round.
///
/// Python parity: the `(content, task_id, metadata)` tuples pushed onto
/// `LoopQueues`; the Rust port carries one named type instead of bare tuples.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct QueuedInput {
    /// The user message text.
    content: String,
    /// Optional task id the input belongs to.
    task_id: Option<String>,
    /// Free-form metadata carried alongside the input.
    metadata: BTreeMap<String, Value>,
}

impl QueuedInput {
    /// Builds a queued input with no task id and no metadata.
    #[must_use]
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            task_id: None,
            metadata: BTreeMap::new(),
        }
    }

    /// Returns this input with `task_id` replaced.
    #[must_use]
    pub fn with_task_id(mut self, task_id: Option<String>) -> Self {
        self.task_id = task_id;
        self
    }

    /// Returns this input with `metadata` replaced.
    #[must_use]
    pub fn with_metadata(mut self, metadata: BTreeMap<String, Value>) -> Self {
        self.metadata = metadata;
        self
    }

    /// The user message text.
    #[must_use]
    pub fn content(&self) -> &str {
        &self.content
    }

    /// The task id this input belongs to, when one was supplied.
    #[must_use]
    pub fn task_id(&self) -> Option<&str> {
        self.task_id.as_deref()
    }

    /// Free-form metadata carried alongside the input.
    #[must_use]
    pub fn metadata(&self) -> &BTreeMap<String, Value> {
        &self.metadata
    }
}

impl From<String> for QueuedInput {
    fn from(content: String) -> Self {
        Self::new(content)
    }
}

impl From<&str> for QueuedInput {
    fn from(content: &str) -> Self {
        Self::new(content)
    }
}

/// Shared FIFO follow-up and steering queues.
///
/// `Clone` shares the underlying queues by identity (Python parity); a fresh
/// [`Default`] value creates two independent empty queues.
#[derive(Debug, Clone, Default)]
pub struct LoopQueues {
    follow_up: Arc<Mutex<VecDeque<QueuedInput>>>,
    steering: Arc<Mutex<VecDeque<QueuedInput>>>,
}

impl LoopQueues {
    /// Appends one input to the follow-up queue.
    pub fn push_follow_up(&self, input: impl Into<QueuedInput>) {
        self.follow_up.lock().push_back(input.into());
    }

    /// Appends one input to the steering queue.
    pub fn push_steer(&self, input: impl Into<QueuedInput>) {
        self.steering.lock().push_back(input.into());
    }

    /// Removes and returns the oldest follow-up input.
    #[must_use]
    pub fn pop_follow_up(&self) -> Option<QueuedInput> {
        self.follow_up.lock().pop_front()
    }

    /// Removes and returns the oldest steering input.
    #[must_use]
    pub fn pop_steer(&self) -> Option<QueuedInput> {
        self.steering.lock().pop_front()
    }

    /// Removes and returns every queued follow-up input, oldest first.
    #[must_use]
    pub fn drain_follow_up(&self) -> Vec<QueuedInput> {
        self.follow_up.lock().drain(..).collect()
    }

    /// Removes and returns every queued steering input, oldest first.
    #[must_use]
    pub fn drain_steering(&self) -> Vec<QueuedInput> {
        self.steering.lock().drain(..).collect()
    }

    /// Returns the contents of the follow-up queue without removing them.
    ///
    /// Used to snapshot `DeepAgentState::pending_follow_ups` after a round; the
    /// queue itself is the authoritative data plane.
    #[must_use]
    pub fn follow_up_contents(&self) -> Vec<String> {
        self.follow_up
            .lock()
            .iter()
            .map(|input| input.content().to_string())
            .collect()
    }

    /// Whether a follow-up input is queued.
    #[must_use]
    pub fn has_follow_up(&self) -> bool {
        !self.follow_up.lock().is_empty()
    }

    /// Whether a steering input is queued.
    #[must_use]
    pub fn has_steering(&self) -> bool {
        !self.steering.lock().is_empty()
    }

    /// Whether both queues are empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.follow_up.lock().is_empty() && self.steering.lock().is_empty()
    }

    /// Number of queued follow-up inputs.
    #[must_use]
    pub fn follow_up_len(&self) -> usize {
        self.follow_up.lock().len()
    }

    /// Number of queued steering inputs.
    #[must_use]
    pub fn steering_len(&self) -> usize {
        self.steering.lock().len()
    }

    /// Drops every queued input on both planes.
    pub fn clear(&self) {
        self.follow_up.lock().clear();
        self.steering.lock().clear();
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::{LoopQueues, QueuedInput};
    use serde_json::json;
    use std::collections::BTreeMap;

    #[test]
    fn queued_input_builders_and_accessors() {
        let mut metadata = BTreeMap::new();
        metadata.insert("origin".to_string(), json!("host"));
        let input = QueuedInput::new("hello")
            .with_task_id(Some("task-1".to_string()))
            .with_metadata(metadata.clone());
        assert_eq!(input.content(), "hello");
        assert_eq!(input.task_id(), Some("task-1"));
        assert_eq!(input.metadata(), &metadata);

        let bare = QueuedInput::new("bare");
        assert_eq!(bare.task_id(), None);
        assert!(bare.metadata().is_empty());
    }

    #[test]
    fn queued_input_conversions_and_serde_round_trip() {
        let from_string: QueuedInput = String::from("a").into();
        let from_str: QueuedInput = "a".into();
        assert_eq!(from_string, from_str);

        let input = QueuedInput::new("round trip").with_task_id(Some("t".to_string()));
        let encoded = serde_json::to_string(&input).expect("serialize queued input");
        let decoded: QueuedInput = serde_json::from_str(&encoded).expect("deserialize");
        assert_eq!(decoded, input);
        let tolerant: QueuedInput = serde_json::from_str("{}").expect("defaults");
        assert_eq!(tolerant, QueuedInput::default());
    }

    #[test]
    fn follow_up_queue_is_fifo() {
        let queues = LoopQueues::default();
        queues.push_follow_up("first");
        queues.push_follow_up(QueuedInput::new("second"));
        assert_eq!(queues.follow_up_len(), 2);
        assert!(queues.has_follow_up());
        assert!(!queues.has_steering());
        assert!(!queues.is_empty());

        assert_eq!(
            queues.pop_follow_up().map(|i| i.content().to_string()),
            Some("first".to_string())
        );
        assert_eq!(
            queues.pop_follow_up().map(|i| i.content().to_string()),
            Some("second".to_string())
        );
        assert!(queues.pop_follow_up().is_none());
        assert!(queues.is_empty());
    }

    #[test]
    fn steering_queue_is_fifo_and_independent() {
        let queues = LoopQueues::default();
        queues.push_steer("s1");
        queues.push_steer("s2");
        queues.push_follow_up("f1");
        assert_eq!(queues.steering_len(), 2);
        assert_eq!(queues.follow_up_len(), 1);
        assert_eq!(queues.drain_steering().len(), 2);
        assert!(!queues.has_steering());
        assert_eq!(queues.drain_follow_up().len(), 1);
        assert!(queues.is_empty());
    }

    #[test]
    fn clone_shares_both_queues() {
        let queues = LoopQueues::default();
        let clone = queues.clone();
        clone.push_follow_up("shared follow-up");
        clone.push_steer("shared steer");
        assert_eq!(queues.follow_up_len(), 1);
        assert_eq!(queues.steering_len(), 1);
        assert_eq!(
            queues.pop_follow_up().map(|i| i.content().to_string()),
            Some("shared follow-up".to_string())
        );
        assert!(clone.pop_steer().is_some());
    }

    #[test]
    fn default_queues_are_unrelated() {
        let first = LoopQueues::default();
        let second = LoopQueues::default();
        first.push_follow_up("only first");
        assert_eq!(first.follow_up_len(), 1);
        assert_eq!(second.follow_up_len(), 0);
    }

    #[test]
    fn follow_up_contents_snapshots_without_draining() {
        let queues = LoopQueues::default();
        queues.push_follow_up("a");
        queues.push_follow_up("b");
        assert_eq!(
            queues.follow_up_contents(),
            vec!["a".to_string(), "b".to_string()]
        );
        assert_eq!(queues.follow_up_len(), 2);
        queues.clear();
        assert!(queues.follow_up_contents().is_empty());
    }
}
