//! Loop event schemas for the `DeepAgent` outer task-loop.
//!
//! Ports `openjiuwen/harness/schema/loop_event.py`: the event types consumed by
//! the outer task-loop, their default queue priorities, and the
//! [`DeepLoopEvent`] payload queued for the loop.
//! Ordering follows the Python `dataclass(order=True)` semantics: two events are
//! compared by `(priority, seq)` only, so the remaining payload fields never
//! affect queue order.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::sync::OnceLock;
use std::time::Instant;

/// Monotonic clock anchor for [`DeepLoopEvent::created_at`].
///
/// The process start is the zero point, mirroring Python's `time.monotonic`.
static ANCHOR: OnceLock<Instant> = OnceLock::new();

/// Event types consumed by the outer task-loop.
///
/// The serde wire names are `followup`, `steer` and `abort`.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
    Default,
    Serialize,
    Deserialize,
    JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum DeepLoopEventType {
    /// A follow-up message appended after the current round.
    #[default]
    Followup,
    /// A steering message injected into the running round.
    Steer,
    /// A request to abort the running round.
    Abort,
}

impl DeepLoopEventType {
    /// Return the default queue priority for this event type.
    ///
    /// Lower values are dequeued first: `Abort` (`0`) before `Steer` (`1`)
    /// before `Followup` (`10`).
    #[must_use]
    pub fn default_priority(self) -> i32 {
        default_event_priority(self)
    }
}

/// Return the default queue priority for a deep loop event type.
///
/// Lower is higher priority: `Abort` = `0`, `Steer` = `1`, `Followup` = `10`.
/// Unknown or custom types fall back to `10` in the Python implementation; the
/// Rust enum is closed, so every variant is mapped explicitly.
#[must_use]
pub fn default_event_priority(event_type: DeepLoopEventType) -> i32 {
    match event_type {
        DeepLoopEventType::Abort => 0,
        DeepLoopEventType::Steer => 1,
        DeepLoopEventType::Followup => 10,
    }
}

/// A queued outer-loop event.
///
/// The `priority` and `seq` fields are the ordering keys for the loop queue:
/// `priority` decides first (lower is higher priority) and `seq` provides FIFO
/// order within a priority bucket.
/// Equality and ordering deliberately ignore the payload fields.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct DeepLoopEvent {
    /// Queue priority; lower is dequeued first.
    ///
    /// Required on the wire so a missing priority is rejected.
    priority: i32,
    /// Monotonic enqueue sequence number; FIFO within a priority.
    ///
    /// Required on the wire so a missing sequence number is rejected.
    seq: u64,
    /// Monotonic seconds since process start when the event was created.
    #[serde(default = "default_created_at")]
    created_at: f64,
    /// Unique event identifier (hyphenated UUID v4 string).
    #[serde(default = "default_event_id")]
    event_id: String,
    /// Kind of loop event.
    #[serde(default)]
    event_type: DeepLoopEventType,
    /// Human-readable payload content.
    #[serde(default)]
    content: String,
    /// Optional task id the event relates to.
    #[serde(default)]
    task_id: Option<String>,
    /// Free-form metadata carried alongside the event.
    #[serde(default)]
    metadata: BTreeMap<String, Value>,
}

/// Default [`DeepLoopEvent::created_at`] value: monotonic seconds since the
/// process-anchored start instant.
fn default_created_at() -> f64 {
    ANCHOR.get_or_init(Instant::now).elapsed().as_secs_f64()
}

/// Default [`DeepLoopEvent::event_id`] value: a fresh hyphenated UUID v4.
fn default_event_id() -> String {
    uuid::Uuid::new_v4().to_string()
}

impl DeepLoopEvent {
    /// Build an event from its parts, generating a fresh `created_at` and
    /// `event_id`.
    fn from_parts(
        priority: i32,
        seq: u64,
        event_type: DeepLoopEventType,
        content: String,
        task_id: Option<String>,
        metadata: BTreeMap<String, Value>,
    ) -> Self {
        Self {
            priority,
            seq,
            created_at: default_created_at(),
            event_id: default_event_id(),
            event_type,
            content,
            task_id,
            metadata,
        }
    }
}

impl Default for DeepLoopEvent {
    /// Build an empty follow-up event at sequence `0` with the follow-up
    /// default priority.
    fn default() -> Self {
        Self::from_parts(
            default_event_priority(DeepLoopEventType::Followup),
            0,
            DeepLoopEventType::Followup,
            String::new(),
            None,
            BTreeMap::new(),
        )
    }
}

impl PartialEq for DeepLoopEvent {
    /// Compare the queue keys `(priority, seq)` only.
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority && self.seq == other.seq
    }
}

impl Eq for DeepLoopEvent {}

impl PartialOrd for DeepLoopEvent {
    /// Compare the queue keys `(priority, seq)` only.
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DeepLoopEvent {
    /// Order by `priority` first, then by `seq`.
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority
            .cmp(&other.priority)
            .then_with(|| self.seq.cmp(&other.seq))
    }
}

/// Build a [`DeepLoopEvent`] with the type's default priority when `priority`
/// is `None`.
///
/// A supplied `priority` is honoured verbatim, including `Some(0)`.
/// `None` metadata is treated as an empty map.
pub fn create_loop_event(
    seq: u64,
    event_type: DeepLoopEventType,
    content: impl Into<String>,
    task_id: Option<String>,
    metadata: Option<BTreeMap<String, Value>>,
    priority: Option<i32>,
) -> DeepLoopEvent {
    DeepLoopEvent::from_parts(
        priority.unwrap_or_else(|| default_event_priority(event_type)),
        seq,
        event_type,
        content.into(),
        task_id,
        metadata.unwrap_or_default(),
    )
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]

    use super::{create_loop_event, default_event_priority, DeepLoopEvent, DeepLoopEventType};
    use serde_json::json;
    use std::cmp::Ordering;
    use std::collections::BTreeMap;

    #[test]
    fn event_type_wire_names_round_trip() {
        for (event_type, wire) in [
            (DeepLoopEventType::Followup, "\"followup\""),
            (DeepLoopEventType::Steer, "\"steer\""),
            (DeepLoopEventType::Abort, "\"abort\""),
        ] {
            let encoded = serde_json::to_string(&event_type).expect("serialize event type");
            assert_eq!(encoded, wire);
            let decoded: DeepLoopEventType =
                serde_json::from_str(&encoded).expect("deserialize event type");
            assert_eq!(decoded, event_type);
        }
        assert_eq!(DeepLoopEventType::default(), DeepLoopEventType::Followup);
    }

    #[test]
    fn default_priority_map_and_method() {
        assert_eq!(default_event_priority(DeepLoopEventType::Abort), 0);
        assert_eq!(default_event_priority(DeepLoopEventType::Steer), 1);
        assert_eq!(default_event_priority(DeepLoopEventType::Followup), 10);

        assert_eq!(DeepLoopEventType::Abort.default_priority(), 0);
        assert_eq!(DeepLoopEventType::Steer.default_priority(), 1);
        assert_eq!(DeepLoopEventType::Followup.default_priority(), 10);
    }

    #[test]
    fn sorting_orders_by_priority_then_seq() {
        let mut events = [
            create_loop_event(1, DeepLoopEventType::Followup, "f1", None, None, None),
            create_loop_event(2, DeepLoopEventType::Abort, "a2", None, None, None),
            create_loop_event(3, DeepLoopEventType::Steer, "s3", None, None, None),
            create_loop_event(4, DeepLoopEventType::Followup, "f4", None, None, None),
            create_loop_event(5, DeepLoopEventType::Abort, "a5", None, None, None),
            create_loop_event(6, DeepLoopEventType::Steer, "s6", None, None, None),
        ];
        events.sort();

        let ordered: Vec<&str> = events.iter().map(|event| event.content.as_str()).collect();
        assert_eq!(ordered, ["a2", "a5", "s3", "s6", "f1", "f4"]);
    }

    #[test]
    fn payload_does_not_affect_equality_or_ordering() {
        let first = create_loop_event(3, DeepLoopEventType::Steer, "first", None, None, Some(1));
        let second = create_loop_event(
            3,
            DeepLoopEventType::Steer,
            "second",
            Some("task-9".to_string()),
            None,
            Some(1),
        );

        assert_eq!(first, second);
        assert_eq!(first.cmp(&second), Ordering::Equal);
        assert_eq!(first.partial_cmp(&second), Some(Ordering::Equal));
        assert_ne!(first.content, second.content);
    }

    #[test]
    fn create_loop_event_honours_explicit_zero_priority() {
        let event = create_loop_event(1, DeepLoopEventType::Followup, "zero", None, None, Some(0));
        assert_eq!(event.priority, 0);
    }

    #[test]
    fn create_loop_event_uses_type_default_priority() {
        assert_eq!(
            create_loop_event(1, DeepLoopEventType::Abort, "abort", None, None, None).priority,
            0
        );
        assert_eq!(
            create_loop_event(1, DeepLoopEventType::Steer, "steer", None, None, None).priority,
            1
        );
        assert_eq!(
            create_loop_event(1, DeepLoopEventType::Followup, "follow", None, None, None).priority,
            10
        );
    }

    #[test]
    fn create_loop_event_defaults_metadata_and_carries_task_id() {
        let event = create_loop_event(
            7,
            DeepLoopEventType::Steer,
            String::from("steer now"),
            Some("task-1".to_string()),
            None,
            None,
        );

        assert_eq!(event.seq, 7);
        assert_eq!(event.content, "steer now");
        assert_eq!(event.task_id.as_deref(), Some("task-1"));
        assert!(event.metadata.is_empty());
        assert!(event.created_at >= 0.0);
        assert!(!event.event_id.is_empty());

        let mut metadata = BTreeMap::new();
        metadata.insert("key".to_string(), json!("value"));
        let with_metadata = create_loop_event(
            8,
            DeepLoopEventType::Followup,
            "meta",
            None,
            Some(metadata),
            None,
        );
        assert_eq!(with_metadata.metadata.get("key"), Some(&json!("value")));
    }

    #[test]
    fn default_event_values() {
        let event = DeepLoopEvent::default();

        assert_eq!(event.priority, 10);
        assert_eq!(event.seq, 0);
        assert_eq!(event.event_type, DeepLoopEventType::Followup);
        assert!(event.content.is_empty());
        assert!(event.task_id.is_none());
        assert!(event.metadata.is_empty());
        assert!(event.created_at >= 0.0);
        assert!(!event.event_id.is_empty());
    }

    #[test]
    fn empty_json_object_is_rejected() {
        assert!(serde_json::from_str::<DeepLoopEvent>("{}").is_err());
    }

    #[test]
    fn json_round_trip_keeps_priority_and_seq() {
        let event = create_loop_event(
            42,
            DeepLoopEventType::Abort,
            "stop",
            Some("task-2".to_string()),
            None,
            Some(0),
        );
        assert!(serde_json::from_str::<DeepLoopEvent>("{\"priority\": 0}").is_err());

        let encoded = serde_json::to_string(&event).expect("serialize event");
        let decoded: DeepLoopEvent = serde_json::from_str(&encoded).expect("deserialize event");

        assert_eq!(decoded.priority, 0);
        assert_eq!(decoded.seq, 42);
        assert_eq!(decoded.event_type, DeepLoopEventType::Abort);
        assert_eq!(decoded.content, "stop");
        assert_eq!(decoded.task_id.as_deref(), Some("task-2"));
        assert_eq!(decoded.event_id, event.event_id);
        assert!((decoded.created_at - event.created_at).abs() < f64::EPSILON);
        assert_eq!(decoded, event);
    }
}
