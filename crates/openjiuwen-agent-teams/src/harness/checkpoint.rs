//! JSON checkpoint value of a team member's native harness.
//!
//! Python parity source: `openjiuwen/agent_teams/harness/checkpoint.py` (`NativeCheckpoint`).
//! Python is a pydantic model serialized with `model_dump`/`model_validate`; the Rust port is a
//! plain `serde` struct so a persisted member survives a release bump: unknown fields are
//! tolerated on decode, every non-identity field has a default, and nothing executable is stored
//! — no pickles, no handles, no configuration objects.
//!
//! This is the *payload* a team embeds in its own persistence; it is not the protocol-level
//! [`openjiuwen_harness_protocol::HarnessCheckpoint`] envelope (which carries provider identity
//! and a sequence number around an opaque payload).
use openjiuwen_common::error::{Error, Result};
use openjiuwen_harness_protocol::HarnessState;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
/// One member's portable checkpoint: session identity, harness state and accumulated output.
///
/// Wire keys are `snake_case`. `session_id` and `card_id` are required; `cwd`, `state`,
/// `turn_id` and `outputs` default when a persisted document predates them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct NativeCheckpoint {
    /// Host session the checkpoint belongs to (`"<team_session>:<member_name>"` by convention).
    pub session_id: String,
    /// Identifier of the agent card the member ran with; a changed card invalidates the state.
    pub card_id: String,
    /// Working directory the member ran in, when it was pinned.
    #[serde(default)]
    pub cwd: Option<String>,
    /// Harness lifecycle state captured when the checkpoint was written.
    #[serde(default = "default_state")]
    pub state: HarnessState,
    /// Turn the member was working on, when the checkpoint was taken mid-turn.
    #[serde(default)]
    pub turn_id: Option<String>,
    /// Accumulated member output (answer parts, final output, tool blocks, …), opaque to this
    /// layer and preserved verbatim.
    #[serde(default = "default_outputs")]
    pub outputs: Value,
}
/// Default captured state: an idle harness.
fn default_state() -> HarnessState {
    HarnessState::Idle
}
/// Default output container: an empty JSON object, matching Python's `{}`.
fn default_outputs() -> Value {
    Value::Object(Map::new())
}
impl NativeCheckpoint {
    /// Builds an empty idle checkpoint for `session_id`/`card_id`.
    #[must_use]
    pub fn new(session_id: impl Into<String>, card_id: impl Into<String>) -> Self {
        Self {
            session_id: session_id.into(),
            card_id: card_id.into(),
            cwd: None,
            state: HarnessState::Idle,
            turn_id: None,
            outputs: default_outputs(),
        }
    }
    /// Sets the pinned working directory.
    #[must_use]
    pub fn with_cwd(mut self, cwd: impl Into<String>) -> Self {
        self.cwd = Some(cwd.into());
        self
    }
    /// Sets the captured harness state.
    #[must_use]
    pub fn with_state(mut self, state: HarnessState) -> Self {
        self.state = state;
        self
    }
    /// Sets the in-flight turn identifier.
    #[must_use]
    pub fn with_turn_id(mut self, turn_id: impl Into<String>) -> Self {
        self.turn_id = Some(turn_id.into());
        self
    }
    /// Replaces the accumulated output value.
    #[must_use]
    pub fn with_outputs(mut self, outputs: Value) -> Self {
        self.outputs = outputs;
        self
    }
    /// Serializes the checkpoint to a JSON value.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::ErrorKind::Validation`] when the value cannot be
    /// encoded (impossible for this plain-data struct, kept for a fallible seam).
    pub fn to_json(&self) -> Result<Value> {
        serde_json::to_value(self)
            .map_err(|error| Error::validation(format!("cannot encode native checkpoint: {error}")))
    }
    /// Decodes a checkpoint from a JSON value, tolerating unknown fields.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::ErrorKind::Validation`] when the value is not a valid
    /// checkpoint document (missing `session_id`/`card_id`, wrong types, …).
    pub fn from_json(value: Value) -> Result<Self> {
        serde_json::from_value(value)
            .map_err(|error| Error::validation(format!("invalid native checkpoint: {error}")))
    }
    /// Checks that this checkpoint may be restored into `session_id`/`card_id`.
    ///
    /// The messages reproduce the Python adapter's rejection wording so an operator can tell a
    /// session change from a card change.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::ErrorKind::Validation`] with
    /// `checkpoint belongs to session '<x>'` when the session differs, or
    /// `checkpoint belongs to card '<x>'` when the card differs.
    pub fn validate_compatible(&self, session_id: &str, card_id: &str) -> Result<()> {
        if self.session_id != session_id {
            return Err(Error::validation(format!(
                "checkpoint belongs to session '{}'",
                self.session_id
            )));
        }
        if self.card_id != card_id {
            return Err(Error::validation(format!(
                "checkpoint belongs to card '{}'",
                self.card_id
            )));
        }
        Ok(())
    }
}
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use openjiuwen_common::error::ErrorKind;
    use serde_json::json;
    #[test]
    fn round_trip_preserves_every_field() {
        let checkpoint = NativeCheckpoint::new("team:dev-1", "card-1")
            .with_cwd("/work/dev-1")
            .with_state(HarnessState::Paused)
            .with_turn_id("turn-7")
            .with_outputs(json!({"answer_parts": ["hi"], "tool_blocks": []}));
        let encoded = checkpoint.to_json().unwrap();
        assert_eq!(encoded["session_id"], json!("team:dev-1"));
        assert_eq!(encoded["state"], json!("paused"));
        assert_eq!(encoded["turn_id"], json!("turn-7"));
        let decoded = NativeCheckpoint::from_json(encoded).unwrap();
        assert_eq!(decoded, checkpoint);
    }
    #[test]
    fn unknown_fields_are_tolerated() {
        let decoded = NativeCheckpoint::from_json(json!({
            "session_id": "team:dev-1",
            "card_id": "card-1",
            "legacy_field": {"any": "thing"},
        }))
        .unwrap();
        assert_eq!(decoded.session_id, "team:dev-1");
        assert_eq!(decoded.card_id, "card-1");
        assert_eq!(decoded.state, HarnessState::Idle);
        assert_eq!(decoded.outputs, json!({}));
        assert_eq!(decoded.cwd, None);
        assert_eq!(decoded.turn_id, None);
    }
    #[test]
    fn missing_identity_fields_are_rejected() {
        let error = NativeCheckpoint::from_json(json!({"card_id": "card-1"})).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("invalid native checkpoint"));
    }
    #[test]
    fn incompatibility_reports_session_then_card() {
        let checkpoint = NativeCheckpoint::new("team:dev-1", "card-1");
        let error = checkpoint
            .validate_compatible("team:dev-2", "card-1")
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert_eq!(
            error.to_string(),
            "validation: checkpoint belongs to session 'team:dev-1'"
        );
        let error = checkpoint
            .validate_compatible("team:dev-1", "card-2")
            .unwrap_err();
        assert_eq!(
            error.to_string(),
            "validation: checkpoint belongs to card 'card-1'"
        );
        checkpoint
            .validate_compatible("team:dev-1", "card-1")
            .unwrap();
    }
}
