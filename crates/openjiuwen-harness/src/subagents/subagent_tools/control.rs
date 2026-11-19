//! Port of the steer/close/revive half of `openjiuwen/harness/tools/subagent/subagent_tools.py`:
//! the `subagent_send_input`, `subagent_close` and `subagent_resume` tools.

use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_common::error::Result;
use openjiuwen_foundation::tool::{Tool, ToolCard, ToolContext};
use serde_json::{json, Value};

use super::super::runtime::SubAgentRuntime;
use super::super::task_tool::description_for;
use super::super::RuntimeSubAgentSpawner;
use super::util::{new_task_id, non_empty, render_payload};
use crate::tools::{tool_card, ToolPayload};

subagent_tool!(
    /// Sends follow-up input to an existing sub-agent instance.
    SubagentSendInputTool,
    "subagent_send_input",
    json!({
        "type": "object",
        "properties": {
            "subagent_id": {
                "type": "string",
                "description": "Target subagent id."
            },
            "query": {
                "type": "string",
                "description": "Follow-up task prompt for the subagent."
            },
            "interrupt": {
                "type": "boolean",
                "default": false,
                "description": "Cancel the active turn before enqueueing the new input."
            }
        },
        "required": ["subagent_id", "query"]
    })
);

#[async_trait]
impl Tool for SubagentSendInputTool {
    fn card(&self) -> &ToolCard {
        &self.card
    }

    async fn invoke(&self, inputs: Value, _ctx: &ToolContext) -> Result<Value> {
        let Some(subagent_id) = non_empty(&inputs, "subagent_id") else {
            return Ok(ToolPayload::fail("'subagent_id' is required").to_json());
        };
        let Some(query) = non_empty(&inputs, "query") else {
            return Ok(ToolPayload::fail("'query' is required").to_json());
        };
        let interrupt = match inputs.get("interrupt") {
            None => false,
            Some(Value::Bool(interrupt)) => *interrupt,
            Some(_) => return Ok(ToolPayload::fail("'interrupt' must be a boolean").to_json()),
        };
        if let Err(error) = self
            .runtime()
            .send_input(&subagent_id, &query, interrupt)
            .await
        {
            return Ok(ToolPayload::fail(error.to_string()).to_json());
        }
        let status = self.runtime().get_status(&subagent_id).as_str();
        Ok(ToolPayload::ok(json!({
            "subagent_id": subagent_id,
            "task_id": new_task_id(),
            "status": status,
        }))
        .to_json())
    }

    fn render_for_llm(&self, output: &Value) -> String {
        render_payload(output, &["subagent_id", "task_id", "status"])
    }
}

subagent_tool!(
    /// Closes an idle sub-agent instance and releases its capacity slot.
    SubagentCloseTool,
    "subagent_close",
    json!({
        "type": "object",
        "properties": {
            "subagent_id": {
                "type": "string",
                "description": "Subagent id to close (from subagent_spawn)."
            }
        },
        "required": ["subagent_id"]
    })
);

#[async_trait]
impl Tool for SubagentCloseTool {
    fn card(&self) -> &ToolCard {
        &self.card
    }

    async fn invoke(&self, inputs: Value, _ctx: &ToolContext) -> Result<Value> {
        let Some(subagent_id) = non_empty(&inputs, "subagent_id") else {
            return Ok(ToolPayload::fail("'subagent_id' is required").to_json());
        };
        match self.runtime().close(&subagent_id).await {
            Ok(previous) => Ok(ToolPayload::ok(json!({
                "subagent_id": subagent_id,
                "previous_status": previous.as_str(),
            }))
            .to_json()),
            Err(error) => Ok(ToolPayload::fail(error.to_string()).to_json()),
        }
    }

    fn render_for_llm(&self, output: &Value) -> String {
        render_payload(output, &["subagent_id", "previous_status"])
    }
}

subagent_tool!(
    /// Restores a closed sub-agent from its checkpoint without dispatching work.
    SubagentResumeTool,
    "subagent_resume",
    json!({
        "type": "object",
        "properties": {
            "subagent_id": {
                "type": "string",
                "description": "Subagent id to restore from checkpointer."
            }
        },
        "required": ["subagent_id"]
    })
);

#[async_trait]
impl Tool for SubagentResumeTool {
    fn card(&self) -> &ToolCard {
        &self.card
    }

    async fn invoke(&self, inputs: Value, _ctx: &ToolContext) -> Result<Value> {
        let Some(subagent_id) = non_empty(&inputs, "subagent_id") else {
            return Ok(ToolPayload::fail("'subagent_id' is required").to_json());
        };
        match self.runtime().resume(&subagent_id).await {
            Ok(report) => {
                let mut data = json!({
                    "subagent_id": subagent_id,
                    "status": report.status.as_str(),
                    "restored": report.restored,
                });
                if let Some(message) = report.message {
                    data["message"] = Value::String(message);
                }
                Ok(ToolPayload::ok(data).to_json())
            }
            Err(error) => Ok(ToolPayload::fail(error.to_string()).to_json()),
        }
    }

    fn render_for_llm(&self, output: &Value) -> String {
        render_payload(output, &["subagent_id", "status", "restored", "message"])
    }
}
