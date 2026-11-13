//! The transition event published on the monitor's broadcast channel.
//!
//! One [`RunEvent`] is published per state transition — `begin` publishes the
//! initial [`RunPhase::Queued`] event, `update` publishes the phase it was given,
//! and `finish` publishes the outcome phase — and the same event is appended to
//! the run's bounded history. The event carries the fields a UI needs, so a
//! subscriber never has to look the status up to render a transition.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::status::{RunKind, RunPhase};

/// One run state transition, as published by
/// [`RunMonitor::subscribe`](crate::monitor::RunMonitor::subscribe).
///
/// Events are ordered per run as the transitions happened, but their `at_ms`
/// stamps are not a global order: two runs can share a millisecond, and only
/// [`RunMonitor::list_active`](crate::monitor::RunMonitor::list_active) promises a
/// deterministic ordering (by `(started_at, run_id)`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct RunEvent {
    /// The run this event belongs to.
    pub run_id: String,
    /// Which entry point the run executes.
    pub kind: RunKind,
    /// Human-readable component label (the card name).
    pub label: String,
    /// The phase this transition entered.
    pub phase: RunPhase,
    /// Session the run is bound to, once resolved.
    pub session_id: Option<String>,
    /// Epoch milliseconds of the transition.
    pub at_ms: i64,
    /// Opaque caller detail of this transition.
    pub detail: Value,
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]

    use super::*;
    use serde_json::json;

    #[test]
    fn event_json_has_the_documented_shape() {
        let event = RunEvent {
            run_id: "run-1".to_string(),
            kind: RunKind::Agent,
            label: "assistant".to_string(),
            phase: RunPhase::Completed,
            session_id: None,
            at_ms: 1_700_000_000_000,
            detail: json!({"answer": 42}),
        };
        let value = serde_json::to_value(&event).unwrap();
        assert_eq!(value["run_id"], json!("run-1"));
        assert_eq!(value["kind"], json!("agent"));
        assert_eq!(value["label"], json!("assistant"));
        assert_eq!(value["phase"], json!("completed"));
        assert_eq!(value["session_id"], Value::Null);
        assert_eq!(value["at_ms"], json!(1_700_000_000_000_i64));
        assert_eq!(value["detail"], json!({"answer": 42}));

        let back: RunEvent = serde_json::from_value(value).unwrap();
        assert_eq!(back, event);
    }

    #[test]
    fn event_default_is_the_queued_agent_event() {
        let event = RunEvent::default();
        assert_eq!(event.phase, RunPhase::Queued);
        assert_eq!(event.kind, RunKind::Agent);
        assert!(event.run_id.is_empty());
        assert_eq!(event.at_ms, 0);
        assert_eq!(event.detail, Value::Null);
    }
}
