//! Priority event queue of the outer task loop.
//!
//! Ports the dormant Python design in `openjiuwen/harness/schema/loop_event.py`
//! (the `DeepLoopEvent` type has no Python consumer). Events are ordered by
//! `(priority, seq)`: the lowest priority value is dequeued first, and `seq`
//! provides FIFO order inside one priority bucket.
//!
//! `Abort` (priority `0`) outranks `Steer` (`1`), which outranks `Followup`
//! (`10`). A bounded queue evicts its least urgent queued event when full.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;
use serde_json::Value;

use crate::schema::{create_loop_event, DeepLoopEvent, DeepLoopEventType};

/// Default queue capacity; `0` means unbounded (Python parity).
pub const DEFAULT_EVENT_CAPACITY: usize = 0;

/// Priority queue of [`DeepLoopEvent`]s, ordered by `(priority, seq)`.
///
/// `Clone` shares the queue and the sequence counter by identity.
#[derive(Debug, Clone)]
pub struct EventManager {
    queue: Arc<Mutex<Vec<DeepLoopEvent>>>,
    capacity: usize,
    seq: Arc<AtomicU64>,
}

impl Default for EventManager {
    fn default() -> Self {
        Self::new(DEFAULT_EVENT_CAPACITY)
    }
}

impl EventManager {
    /// Builds a manager with the given capacity; `0` means unbounded.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            queue: Arc::new(Mutex::new(Vec::new())),
            capacity,
            seq: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Builder form of [`Self::new`].
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self::new(capacity)
    }

    /// The configured capacity (`0` = unbounded).
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Enqueues a pre-built event, returning the event rejected by a full queue.
    ///
    /// A full queue evicts its least urgent event (largest `(priority, seq)`).
    /// When the incoming event is itself the least urgent, it is rejected and
    /// returned instead, so the caller can observe the drop.
    #[must_use]
    pub fn publish(&self, event: DeepLoopEvent) -> Option<DeepLoopEvent> {
        let mut queue = self.queue.lock();
        if self.capacity > 0 && queue.len() >= self.capacity {
            let incoming_is_least_urgent = queue.last().is_none_or(|queued| queued <= &event);
            if incoming_is_least_urgent {
                return Some(event);
            }
            let dropped = queue.pop();
            insert_sorted(&mut queue, event);
            return dropped;
        }
        insert_sorted(&mut queue, event);
        None
    }

    /// Builds, stamps and enqueues an event, returning the event a full queue
    /// rejected.
    ///
    /// The `seq` value is taken from the manager's shared monotonic counter;
    /// `priority = None` uses the event type's default priority.
    #[must_use]
    pub fn publish_event(
        &self,
        event_type: DeepLoopEventType,
        content: impl Into<String>,
        task_id: Option<String>,
        metadata: Option<BTreeMap<String, Value>>,
        priority: Option<i32>,
    ) -> Option<DeepLoopEvent> {
        let event = create_loop_event(
            self.next_seq(),
            event_type,
            content,
            task_id,
            metadata,
            priority,
        );
        self.publish(event)
    }

    /// Removes and returns the most urgent queued event.
    #[must_use]
    pub fn pop(&self) -> Option<DeepLoopEvent> {
        let mut queue = self.queue.lock();
        if queue.is_empty() {
            return None;
        }
        Some(queue.remove(0))
    }

    /// Returns the most urgent queued event without removing it.
    #[must_use]
    pub fn peek(&self) -> Option<DeepLoopEvent> {
        self.queue.lock().first().cloned()
    }

    /// Removes and returns every queued event in dequeue order.
    #[must_use]
    pub fn drain(&self) -> Vec<DeepLoopEvent> {
        let mut queue = self.queue.lock();
        std::mem::take(&mut *queue)
    }

    /// Number of queued events.
    #[must_use]
    pub fn len(&self) -> usize {
        self.queue.lock().len()
    }

    /// Whether the queue is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.queue.lock().is_empty()
    }

    /// Drops every queued event; the sequence counter keeps advancing.
    pub fn clear(&self) {
        self.queue.lock().clear();
    }

    /// Takes the next monotonic sequence number from the shared counter.
    #[must_use]
    pub fn next_seq(&self) -> u64 {
        self.seq.fetch_add(1, Ordering::SeqCst)
    }
}

/// Inserts `event` at its ordered position (ascending `(priority, seq)`).
fn insert_sorted(queue: &mut Vec<DeepLoopEvent>, event: DeepLoopEvent) {
    let position = queue.partition_point(|queued| queued <= &event);
    queue.insert(position, event);
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::{EventManager, DEFAULT_EVENT_CAPACITY};
    use crate::schema::{create_loop_event, DeepLoopEventType};
    use serde_json::json;

    /// Reads the private event payload through its serde representation.
    fn contents(manager: &EventManager) -> Vec<String> {
        manager
            .drain()
            .iter()
            .map(|event| {
                serde_json::to_value(event).expect("serialize event")["content"]
                    .as_str()
                    .expect("content string")
                    .to_string()
            })
            .collect()
    }

    fn priorities(events: &[crate::schema::DeepLoopEvent]) -> Vec<i64> {
        events
            .iter()
            .map(|event| {
                serde_json::to_value(event).expect("serialize event")["priority"]
                    .as_i64()
                    .expect("priority")
            })
            .collect()
    }

    #[test]
    fn default_capacity_is_unbounded() {
        let manager = EventManager::default();
        assert_eq!(manager.capacity(), DEFAULT_EVENT_CAPACITY);
        for index in 0..64 {
            assert!(manager
                .publish_event(
                    DeepLoopEventType::Followup,
                    format!("event-{index}"),
                    None,
                    None,
                    None,
                )
                .is_none());
        }
        assert_eq!(manager.len(), 64);
    }

    #[test]
    fn pops_in_priority_then_sequence_order() {
        let manager = EventManager::new(0);
        let _ = manager.publish_event(DeepLoopEventType::Followup, "f1", None, None, None);
        let _ = manager.publish_event(DeepLoopEventType::Steer, "s1", None, None, None);
        let _ = manager.publish_event(DeepLoopEventType::Abort, "a1", None, None, None);
        let _ = manager.publish_event(DeepLoopEventType::Followup, "f2", None, None, None);

        assert_eq!(manager.len(), 4);
        assert_eq!(
            manager.peek().map(
                |event| serde_json::to_value(event).expect("event")["content"]
                    .as_str()
                    .expect("content")
                    .to_string()
            ),
            Some("a1".to_string())
        );
        assert_eq!(contents(&manager), vec!["a1", "s1", "f1", "f2"]);
        assert!(manager.pop().is_none());
        assert!(manager.is_empty());
    }

    #[test]
    fn custom_priorities_override_type_defaults() {
        let manager = EventManager::new(0);
        let _ = manager.publish_event(DeepLoopEventType::Followup, "low", None, None, Some(20));
        let _ = manager.publish_event(DeepLoopEventType::Followup, "high", None, None, Some(-5));
        let _ = manager.publish_event(DeepLoopEventType::Abort, "mid", None, None, Some(5));
        let events = manager.drain();
        assert_eq!(priorities(&events), vec![-5, 5, 20]);
    }

    #[test]
    fn publish_event_stamps_monotonic_sequence() {
        let manager = EventManager::new(0);
        assert_eq!(manager.next_seq(), 0);
        assert_eq!(manager.next_seq(), 1);
        let clone = manager.clone();
        assert_eq!(clone.next_seq(), 2);
        // The counter is shared between clones.
        assert_eq!(manager.next_seq(), 3);
    }

    #[test]
    fn bounded_queue_evicts_least_urgent_event() {
        let manager = EventManager::new(2);
        let _ = manager.publish_event(DeepLoopEventType::Abort, "a", None, None, None);
        let _ = manager.publish_event(DeepLoopEventType::Followup, "f", None, None, None);
        assert_eq!(manager.len(), 2);

        let rejected = manager.publish_event(DeepLoopEventType::Steer, "s", None, None, None);
        // Steer outranks the follow-up, so the follow-up is evicted.
        let dropped = rejected.expect("the follow-up should be evicted");
        let dropped_value = serde_json::to_value(&dropped).expect("event");
        assert_eq!(dropped_value["content"], json!("f"));
        assert_eq!(manager.len(), 2);
        assert_eq!(contents(&manager), vec!["a", "s"]);
    }

    #[test]
    fn bounded_queue_rejects_least_urgent_incoming_event() {
        let manager = EventManager::new(1);
        let _ = manager.publish_event(DeepLoopEventType::Abort, "a", None, None, None);
        let rejected = manager.publish_event(DeepLoopEventType::Followup, "f", None, None, None);
        let rejected = rejected.expect("the incoming follow-up is least urgent");
        let value = serde_json::to_value(&rejected).expect("event");
        assert_eq!(value["content"], json!("f"));
        assert_eq!(manager.len(), 1);
        assert_eq!(contents(&manager), vec!["a"]);
    }

    #[test]
    fn same_priority_keeps_oldest_event() {
        let manager = EventManager::new(1);
        let _ = manager.publish_event(DeepLoopEventType::Steer, "first", None, None, None);
        let rejected = manager.publish_event(DeepLoopEventType::Steer, "second", None, None, None);
        let value =
            serde_json::to_value(rejected.expect("latest steer is rejected")).expect("event");
        assert_eq!(value["content"], json!("second"));
        assert_eq!(contents(&manager), vec!["first"]);
    }

    #[test]
    fn clear_and_drain_reset_the_queue_but_not_the_counter() {
        let manager = EventManager::new(0);
        let _ = manager.publish_event(DeepLoopEventType::Followup, "a", None, None, None);
        let _ = manager.publish_event(DeepLoopEventType::Followup, "b", None, None, None);
        manager.clear();
        assert!(manager.is_empty());
        assert_eq!(manager.next_seq(), 2);

        let _ = manager.publish_event(DeepLoopEventType::Followup, "c", None, None, None);
        assert_eq!(manager.drain().len(), 1);
        assert_eq!(manager.drain(), Vec::new());
    }

    #[test]
    fn publish_accepts_prebuilt_events_and_metadata() {
        let manager = EventManager::new(0);
        let mut metadata = std::collections::BTreeMap::new();
        metadata.insert("origin".to_string(), json!("host"));
        let _ = manager.publish_event(
            DeepLoopEventType::Steer,
            "steer",
            Some("task-7".to_string()),
            Some(metadata),
            Some(0),
        );
        let event = manager.pop().expect("queued event");
        let value = serde_json::to_value(&event).expect("event");
        assert_eq!(value["priority"], json!(0));
        assert_eq!(value["task_id"], json!("task-7"));
        assert_eq!(value["metadata"]["origin"], json!("host"));
        assert_eq!(value["event_type"], json!("steer"));

        assert!(manager
            .publish(create_loop_event(
                9,
                DeepLoopEventType::Abort,
                "a",
                None,
                None,
                None
            ))
            .is_none());
        assert_eq!(manager.len(), 1);
    }

    #[test]
    fn clone_shares_queue_and_capacity() {
        let manager = EventManager::with_capacity(1);
        let clone = manager.clone();
        let _ = clone.publish_event(DeepLoopEventType::Abort, "a", None, None, None);
        assert_eq!(manager.len(), 1);
        assert_eq!(clone.capacity(), 1);
        let dropped = manager.publish_event(DeepLoopEventType::Steer, "s", None, None, None);
        assert!(dropped.is_some());
        assert_eq!(manager.len(), 1);
    }
}
