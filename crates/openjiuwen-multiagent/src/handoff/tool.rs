//! The `transfer_to_<id>` tool injected into a member agent.
//!
//! Python parity: `HandoffTool` in
//! `openjiuwen/core/multi_agent/teams/handoff/handoff_tool.py`. The tool is how
//! a model asks for a handoff: it never performs the transfer itself, it returns
//! the signal payload ([`HANDOFF_TARGET_KEY`] plus optional message and reason)
//! that [`extract_handoff_signal`](super::extract_handoff_signal) later reads.
//!
//! The tool never fails: malformed model arguments degrade to an empty reason
//! and message, so a handoff request is never lost to argument validation.

use async_trait::async_trait;
use openjiuwen_common::error::Result;
use openjiuwen_foundation::tool::{Tool, ToolCard, ToolContext};
use serde_json::{json, Value};

use super::{HANDOFF_MESSAGE_KEY, HANDOFF_REASON_KEY, HANDOFF_TARGET_KEY};

/// A tool that signals control transfer to one target agent.
///
/// Python parity: the tool name is `transfer_to_{target_id}`, the card id equals
/// the name, and the description appends the target's own description when one
/// is provided.
pub struct HandoffTool {
    /// The model-facing card.
    card: ToolCard,
    /// Id of the agent a call transfers control to.
    target_id: String,
}

impl std::fmt::Debug for HandoffTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HandoffTool")
            .field("name", &self.card.name)
            .field("target_id", &self.target_id)
            .finish_non_exhaustive()
    }
}

impl HandoffTool {
    /// Builds the transfer tool for `target_id`.
    ///
    /// `target_description` is appended to the model-facing description when it
    /// is not empty; an empty description leaves the generic text alone.
    #[must_use]
    pub fn new(target_id: impl Into<String>, target_description: &str) -> Self {
        let target_id = target_id.into();
        let name = format!("transfer_to_{target_id}");
        let mut description = format!("Transfer the current task to {target_id} for processing.");
        if !target_description.is_empty() {
            description.push(' ');
            description.push_str(target_description);
        }
        let card = ToolCard::new(name.clone(), description, Self::input_params())
            .with_id(name)
            .with_parallel_safe(true);
        Self { card, target_id }
    }

    /// The exact JSON Schema of the tool arguments.
    fn input_params() -> Value {
        json!({
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Reason for handoff: briefly explain why the task is being transferred."
                },
                "message": {
                    "type": "string",
                    "description": "Context information passed to the next agent (optional)."
                }
            },
            "required": ["reason"]
        })
    }

    /// Returns the tool name (`transfer_to_{target_id}`).
    #[must_use]
    pub fn name(&self) -> &str {
        &self.card.name
    }

    /// Returns the id of the agent this tool transfers control to.
    #[must_use]
    pub fn target_id(&self) -> &str {
        &self.target_id
    }

    /// Normalizes the model's arguments into the signal payload.
    ///
    /// A string argument is parsed as JSON and, when that fails, treated as the
    /// reason itself; a non-object argument becomes an empty argument object.
    /// Only string `message`/`reason` values are kept.
    fn payload(&self, inputs: Value) -> Value {
        let inputs = match inputs {
            Value::String(text) => {
                serde_json::from_str(&text).unwrap_or_else(|_| json!({"reason": text}))
            }
            other => other,
        };
        let (message, reason) = match inputs.as_object() {
            Some(object) => (
                object.get("message").and_then(Value::as_str).unwrap_or(""),
                object.get("reason").and_then(Value::as_str).unwrap_or(""),
            ),
            None => ("", ""),
        };
        json!({
            HANDOFF_TARGET_KEY: self.target_id,
            HANDOFF_MESSAGE_KEY: message,
            HANDOFF_REASON_KEY: reason
        })
    }
}

#[async_trait]
impl Tool for HandoffTool {
    fn card(&self) -> &ToolCard {
        &self.card
    }

    /// Returns the handoff signal; this tool has no failure mode.
    async fn invoke(&self, inputs: Value, _ctx: &ToolContext) -> Result<Value> {
        Ok(self.payload(inputs))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use serde_json::json;

    use super::*;

    #[tokio::test]
    async fn card_matches_the_python_shape() {
        let tool = HandoffTool::new("b", "");
        assert_eq!(tool.name(), "transfer_to_b");
        assert_eq!(tool.target_id(), "b");

        let card = tool.card();
        assert_eq!(card.id, "transfer_to_b");
        assert_eq!(card.name, "transfer_to_b");
        assert_eq!(
            card.description,
            "Transfer the current task to b for processing."
        );
        assert!(card.parallel_safe);
        assert_eq!(
            card.input_params,
            json!({
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Reason for handoff: briefly explain why the task is being transferred."
                    },
                    "message": {
                        "type": "string",
                        "description": "Context information passed to the next agent (optional)."
                    }
                },
                "required": ["reason"]
            })
        );
    }

    #[tokio::test]
    async fn target_description_is_appended_when_present() {
        let tool = HandoffTool::new("b", "Handles billing questions.");
        assert_eq!(
            tool.card().description,
            "Transfer the current task to b for processing. Handles billing questions."
        );
    }

    #[tokio::test]
    async fn invoke_with_an_object_returns_the_signal() {
        let tool = HandoffTool::new("b", "");
        let value = tool
            .invoke(
                json!({"reason": "busy", "message": "context"}),
                &ToolContext::new(),
            )
            .await
            .unwrap();
        assert_eq!(
            value,
            json!({
                "__handoff_to__": "b",
                "__handoff_message__": "context",
                "__handoff_reason__": "busy"
            })
        );
    }

    #[tokio::test]
    async fn invoke_parses_a_json_string() {
        let tool = HandoffTool::new("b", "");
        let value = tool
            .invoke(
                json!("{\"reason\": \"r\", \"message\": \"m\"}"),
                &ToolContext::new(),
            )
            .await
            .unwrap();
        assert_eq!(value["__handoff_reason__"], "r");
        assert_eq!(value["__handoff_message__"], "m");
    }

    #[tokio::test]
    async fn plain_strings_become_the_reason() {
        let tool = HandoffTool::new("b", "");
        let value = tool
            .invoke(json!("the user needs billing"), &ToolContext::new())
            .await
            .unwrap();
        assert_eq!(value["__handoff_reason__"], "the user needs billing");
        assert_eq!(value["__handoff_message__"], "");
    }

    #[tokio::test]
    async fn non_object_arguments_degrade_to_empty_fields() {
        let tool = HandoffTool::new("b", "");
        for inputs in [json!([1, 2]), json!(7), Value::Null, json!(true)] {
            let value = tool.invoke(inputs, &ToolContext::new()).await.unwrap();
            assert_eq!(value["__handoff_to__"], "b");
            assert_eq!(value["__handoff_reason__"], "");
            assert_eq!(value["__handoff_message__"], "");
        }
    }

    #[tokio::test]
    async fn non_string_fields_are_dropped_and_garbage_never_errors() {
        let tool = HandoffTool::new("b", "");
        let value = tool
            .invoke(
                json!({"reason": 3, "message": ["x"], "other": 1}),
                &ToolContext::new(),
            )
            .await
            .unwrap();
        assert_eq!(
            value,
            json!({
                "__handoff_to__": "b",
                "__handoff_message__": "",
                "__handoff_reason__": ""
            })
        );

        // A broken JSON string falls back to using the whole text as the reason.
        let value = tool
            .invoke(json!("{\"reason\": "), &ToolContext::new())
            .await
            .unwrap();
        assert_eq!(value["__handoff_reason__"], "{\"reason\": ");
    }
}
