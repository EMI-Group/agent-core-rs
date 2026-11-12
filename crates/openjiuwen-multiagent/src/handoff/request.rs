//! The drive message published between container agents.
//!
//! Python parity: `HandoffRequest` in
//! `openjiuwen/core/multi_agent/teams/handoff/handoff_request.py`. Python's
//! dataclass also carries the team `session` object; Rust keeps the request pure
//! JSON data and lets the container look the team session up through the bus
//! envelope's session id, so the request stays serializable.
//!
//! The request is the payload of the pub/sub message delivered to
//! `container_{agent_id}`: [`HandoffRequest::input_message`] is what the next hop
//! consumes, and [`HandoffRequest::history`] accumulates the output of every hop
//! already taken so a resumed chain keeps its context.

use openjiuwen_common::error::{Error, Result};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// One completed hop of a handoff chain.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct HandoffHistoryEntry {
    /// Id of the agent that produced `output`.
    pub agent: String,
    /// The agent's output value for that hop.
    pub output: Value,
}

/// The payload published to the container endpoint of the next hop.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct HandoffRequest {
    /// Input the next agent receives.
    pub input_message: Value,
    /// Outputs of the hops already taken, oldest first.
    pub history: Vec<HandoffHistoryEntry>,
}

impl HandoffRequest {
    /// Builds a request with the given input and accumulated history.
    #[must_use]
    pub fn new(input_message: Value, history: Vec<HandoffHistoryEntry>) -> Self {
        Self {
            input_message,
            history,
        }
    }

    /// Serializes the request to its wire shape.
    ///
    /// The result always serializes: it is built from already-valid JSON values,
    /// and an impossible serialization failure degrades to `null` with a warning
    /// instead of panicking.
    #[must_use]
    pub fn to_value(&self) -> Value {
        match serde_json::to_value(self) {
            Ok(value) => value,
            Err(error) => {
                tracing::warn!(%error, "handoff request could not be serialized");
                Value::Null
            }
        }
    }

    /// Parses a request from its wire shape.
    ///
    /// An absent or `null` `history` means an empty history (the first hop),
    /// matching the Python default. Unknown keys are ignored.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::ErrorKind::Validation`] when `value`
    /// is not a JSON object or when `history` is not an array of history
    /// entries.
    pub fn from_value(value: &Value) -> Result<Self> {
        let object = value
            .as_object()
            .ok_or_else(|| Error::validation("handoff request must be a JSON object"))?;
        let input_message = object.get("input_message").cloned().unwrap_or(Value::Null);
        let history = match object.get("history") {
            None | Some(Value::Null) => Vec::new(),
            Some(history) => serde_json::from_value(history.clone())
                .map_err(|error| Error::validation(format!("invalid handoff history: {error}")))?,
        };
        Ok(Self {
            input_message,
            history,
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use openjiuwen_common::error::ErrorKind;
    use serde_json::json;

    use super::*;

    #[test]
    fn round_trips_through_the_wire_shape() {
        let request = HandoffRequest::new(
            json!({"query": "hi"}),
            vec![HandoffHistoryEntry {
                agent: "a".to_owned(),
                output: json!({"answer": 1}),
            }],
        );
        let value = request.to_value();
        assert_eq!(
            value,
            json!({
                "input_message": {"query": "hi"},
                "history": [{"agent": "a", "output": {"answer": 1}}]
            })
        );
        assert_eq!(HandoffRequest::from_value(&value).unwrap(), request);
    }

    #[test]
    fn missing_and_null_history_are_empty() {
        let absent = HandoffRequest::from_value(&json!({"input_message": "x"})).unwrap();
        assert!(absent.history.is_empty(), "absent history is empty");
        assert_eq!(absent.input_message, json!("x"));

        let null =
            HandoffRequest::from_value(&json!({"input_message": null, "history": null})).unwrap();
        assert!(null.history.is_empty(), "null history is empty");
        assert_eq!(null.input_message, Value::Null);
    }

    #[test]
    fn unknown_keys_are_tolerated() {
        let value = json!({
            "input_message": {"a": 1},
            "history": [],
            "session": {"session_id": "ignored"},
            "surprise": true
        });
        let request = HandoffRequest::from_value(&value).unwrap();
        assert_eq!(request.input_message, json!({"a": 1}));
        assert!(request.history.is_empty(), "history stays empty");
    }

    #[test]
    fn non_object_input_is_a_validation_error() {
        let error = HandoffRequest::from_value(&json!(["not", "an", "object"])).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        let error = HandoffRequest::from_value(&Value::Null).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
    }

    #[test]
    fn wrong_typed_history_is_a_validation_error() {
        let error = HandoffRequest::from_value(&json!({
            "input_message": "x",
            "history": 3
        }))
        .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.message().contains("invalid handoff history"));

        let error = HandoffRequest::from_value(&json!({
            "input_message": "x",
            "history": [{"agent": 7, "output": {}}]
        }))
        .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
    }

    #[test]
    fn defaults_are_empty() {
        let request = HandoffRequest::default();
        assert_eq!(request.input_message, Value::Null);
        assert!(request.history.is_empty(), "default history is empty");
        assert_eq!(
            request.to_value(),
            json!({"input_message": null, "history": []})
        );
    }
}
