//! Ordered execution events emitted by a run.
//!
//! Events are the streaming contract: they arrive in a fixed order (superstep
//! boundary, then nodes in declaration order, then edges) regardless of which
//! concurrent node finished first. The payload of an event is always a JSON
//! object, because the session stream hub rejects `null` payloads.

use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

/// What happened in a run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GraphEventKind {
    /// A superstep started; the payload lists the ready node ids.
    SuperstepStarted,
    /// A ready node is about to run.
    NodeStarted,
    /// A node completed successfully.
    NodeCompleted,
    /// A node failed (after its final attempt).
    NodeFailed,
    /// A routing edge activated a target.
    EdgeTraversed,
    /// Every ready node of the superstep completed and routing was applied.
    SuperstepCompleted,
    /// The run completed normally.
    GraphCompleted,
    /// The run halted on a [`crate::NodeOutcome::Halt`].
    GraphHalted,
    /// The run failed.
    GraphFailed,
}

impl GraphEventKind {
    /// The `snake_case` wire name of the kind.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SuperstepStarted => "superstep_started",
            Self::NodeStarted => "node_started",
            Self::NodeCompleted => "node_completed",
            Self::NodeFailed => "node_failed",
            Self::EdgeTraversed => "edge_traversed",
            Self::SuperstepCompleted => "superstep_completed",
            Self::GraphCompleted => "graph_completed",
            Self::GraphHalted => "graph_halted",
            Self::GraphFailed => "graph_failed",
        }
    }
}

impl std::fmt::Display for GraphEventKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One ordered execution event.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct GraphEvent {
    /// What happened.
    pub kind: GraphEventKind,
    /// The node the event belongs to, when it belongs to one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub node_id: Option<String>,
    /// The superstep the event was emitted in (1-based; `0` outside a run).
    pub superstep: u32,
    /// Event details; always a JSON object.
    pub payload: Value,
}

impl GraphEvent {
    /// Builds an event, normalizing the payload to an object.
    #[must_use]
    pub fn new(
        kind: GraphEventKind,
        node_id: Option<String>,
        superstep: u32,
        payload: Value,
    ) -> Self {
        Self {
            kind,
            node_id,
            superstep,
            payload: payload_object(payload),
        }
    }

    /// The wire projection of the event:
    /// `{"kind", "node_id"?, "superstep", "payload"}`.
    #[must_use]
    pub fn to_wire(&self) -> Value {
        let mut map = Map::new();
        map.insert(
            "kind".to_string(),
            Value::String(self.kind.as_str().to_string()),
        );
        if let Some(node_id) = &self.node_id {
            map.insert("node_id".to_string(), Value::String(node_id.clone()));
        }
        map.insert("superstep".to_string(), Value::from(self.superstep));
        map.insert("payload".to_string(), payload_object(self.payload.clone()));
        Value::Object(map)
    }
}

/// Wraps any non-object payload as `{"value": …}` so it never reaches a stream
/// as `null`.
fn payload_object(payload: Value) -> Value {
    match payload {
        Value::Object(map) => Value::Object(map),
        other => {
            let mut map = Map::new();
            map.insert("value".to_string(), other);
            Value::Object(map)
        }
    }
}

/// Builds the standard `{"error": …, "kind": …}` payload of a failure event.
pub(crate) fn error_payload(error: &openjiuwen_common::error::Error) -> Value {
    json!({
        "error": error.message().into_owned(),
        "kind": error.kind().as_str(),
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]
mod tests {
    use super::*;

    #[test]
    fn wire_shape_and_object_payload_invariant() {
        let event = GraphEvent::new(
            GraphEventKind::NodeStarted,
            Some("a".to_string()),
            2,
            Value::Null,
        );
        let wire = event.to_wire();
        assert_eq!(wire["kind"], "node_started");
        assert_eq!(wire["node_id"], "a");
        assert_eq!(wire["superstep"], 2);
        assert_eq!(wire["payload"], json!({"value": null}));
        assert!(wire["payload"].is_object());

        let scalar = GraphEvent::new(GraphEventKind::GraphCompleted, None, 1, json!(7));
        assert_eq!(scalar.to_wire()["payload"], json!({"value": 7}));
    }

    #[test]
    fn object_payloads_pass_through() {
        let event = GraphEvent::new(
            GraphEventKind::EdgeTraversed,
            Some("a".to_string()),
            1,
            json!({"target": "b"}),
        );
        assert_eq!(event.payload, json!({"target": "b"}));
        assert_eq!(event.to_wire()["payload"]["target"], "b");
    }

    #[test]
    fn event_round_trips() {
        let event = GraphEvent::new(GraphEventKind::GraphFailed, None, 3, json!({"x": 1}));
        let text = serde_json::to_string(&event).unwrap();
        let decoded: GraphEvent = serde_json::from_str(&text).unwrap();
        assert_eq!(decoded, event);
        assert_eq!(GraphEventKind::GraphHalted.as_str(), "graph_halted");
        assert_eq!(
            GraphEventKind::SuperstepStarted.to_string(),
            "superstep_started"
        );
    }

    #[test]
    fn missing_node_id_is_omitted_from_serialization() {
        let event = GraphEvent::new(GraphEventKind::GraphCompleted, None, 1, json!({}));
        let wire = event.to_wire();
        assert!(wire.get("node_id").is_none());
        let serialized = serde_json::to_value(&event).unwrap();
        assert!(serialized.get("node_id").is_none());
    }

    #[test]
    fn failure_payload_carries_kind() {
        let error = openjiuwen_common::error::Error::timeout("slow node");
        let payload = error_payload(&error);
        assert_eq!(payload["kind"], "timeout");
        assert_eq!(payload["error"], "slow node");
    }
}
