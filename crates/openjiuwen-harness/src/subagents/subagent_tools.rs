//! Port of `openjiuwen/harness/tools/subagent/subagent_tools.py`: the persistent
//! `subagent_*` tools.
//!
//! These six tools are the non-blocking counterpart of [`super::task_tool::TaskTool`]:
//! `subagent_spawn` registers an instance and returns immediately, `subagent_wait` blocks
//! until the requested instances reach a final status, and the remaining tools list,
//! steer, close and revive instances.
//!
//! Every expected failure is a [`ToolPayload::fail`] envelope (`success = false` plus a
//! Python-derived `error` message), never an `Err`; only an unexpected internal error
//! escapes as `Err`.
//!
//! The steer/close/revive tools live in [`control`]; the parsing and rendering helpers shared
//! by every tool live in `util`.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use openjiuwen_common::error::Result;
use openjiuwen_foundation::tool::{Tool, ToolCard, ToolContext};
use serde_json::{json, Map, Value};

pub use control::{SubagentCloseTool, SubagentResumeTool, SubagentSendInputTool};
use util::{
    new_task_id, non_empty, parse_subagent_ids, payload_error, render_payload,
    render_subagent_rows, resolve_timeout_ms,
};

use super::runtime::{SubAgentRuntime, SubAgentSpawnRequest, SubagentStatus};
use super::task_tool::description_for;
use crate::tools::{tool_card, ToolPayload};

/// Default `subagent_wait` deadline (Python `WAIT_TIMEOUT_MS_DEFAULT`), in milliseconds.
const DEFAULT_WAIT_TIMEOUT_MS: u64 = 2_400_000;

/// Smallest accepted `subagent_wait` deadline, in milliseconds.
const MIN_WAIT_TIMEOUT_MS: i64 = 10_000;

/// Largest accepted `subagent_wait` deadline, in milliseconds.
const MAX_WAIT_TIMEOUT_MS: i64 = 3_600_000;

/// The Python message for an incomplete `subagent_spawn` payload.
const SPAWN_REQUIRED_MESSAGE: &str =
    "'subagent_type', 'task_description', 'display_name', and 'role' are required";

/// The fields of a live/closed sub-agent row, in Python order.
const SUBAGENT_ROW_KEYS: [&str; 9] = [
    "subagent_id",
    "display_name",
    "subagent_type",
    "role",
    "status",
    "turn_outcome",
    "closed_reason",
    "error",
    "task_description",
];

/// Builds the six runtime sub-agent tools in Python order.
///
/// Order: `subagent_spawn`, `subagent_wait`, `subagent_list`, `subagent_send_input`,
/// `subagent_close`, `subagent_resume`.
#[must_use]
pub fn subagent_tools(runtime: Arc<SubAgentRuntime>) -> Vec<Arc<dyn Tool>> {
    vec![
        Arc::new(SubagentSpawnTool::new(Arc::clone(&runtime))),
        Arc::new(SubagentWaitTool::new(Arc::clone(&runtime))),
        Arc::new(SubagentListTool::new(Arc::clone(&runtime))),
        Arc::new(SubagentSendInputTool::new(Arc::clone(&runtime))),
        Arc::new(SubagentCloseTool::new(Arc::clone(&runtime))),
        Arc::new(SubagentResumeTool::new(runtime)),
    ]
}

/// Declares one persistent sub-agent tool over [`SubAgentRuntime`].
///
/// The generated type owns the runtime and its model-facing card, plus the shared
/// `new`/`card_for`/`with_language` surface; each tool supplies only its name and schema.
macro_rules! subagent_tool {
    ($(#[$meta:meta])* $tool:ident, $name:literal, $schema:expr) => {
        $(#[$meta])*
        pub struct $tool {
            runtime: Arc<SubAgentRuntime>,
            card: ToolCard,
        }

        impl $tool {
            /// Builds the tool over `runtime` in the default language.
            #[must_use]
            pub fn new(runtime: Arc<SubAgentRuntime>) -> Self {
                Self {
                    runtime,
                    card: Self::card_for(crate::prompts::DEFAULT_LANGUAGE),
                }
            }

            /// Builds the model-facing card in `language`.
            #[must_use]
            pub fn card_for(language: &str) -> ToolCard {
                tool_card($name, description_for($name, language), $schema, true)
            }

            /// Returns this tool with its card rebuilt in `language`.
            #[must_use]
            pub fn with_language(mut self, language: &str) -> Self {
                self.card = Self::card_for(language);
                self
            }
        }
    };
}

// `mod control;` must stay below the `subagent_tool!` definition: `macro_rules!` macros are
// only in scope for child modules declared after them.
mod control;

subagent_tool!(
    /// Spawns a persistent sub-agent and enqueues its first turn without blocking.
    SubagentSpawnTool,
    "subagent_spawn",
    json!({
        "type": "object",
        "properties": {
            "subagent_type": {
                "type": "string",
                "description": "Registered subagent type name."
            },
            "task_description": {
                "type": "string",
                "description": "First-turn task prompt for the subagent."
            },
            "display_name": {
                "type": "string",
                "description": "Short human-readable name shown in the UI (e.g. 'API Researcher')."
            },
            "role": {
                "type": "string",
                "description": "One-line role for this subtask (e.g. 'Review auth module changes')."
            }
        },
        "required": ["subagent_type", "task_description", "display_name", "role"]
    })
);

#[async_trait]
impl Tool for SubagentSpawnTool {
    fn card(&self) -> &ToolCard {
        &self.card
    }

    async fn invoke(&self, inputs: Value, _ctx: &ToolContext) -> Result<Value> {
        let (Some(subagent_type), Some(task_description), Some(display_name), Some(role)) = (
            non_empty(&inputs, "subagent_type"),
            non_empty(&inputs, "task_description"),
            non_empty(&inputs, "display_name"),
            non_empty(&inputs, "role"),
        ) else {
            return Ok(ToolPayload::fail(SPAWN_REQUIRED_MESSAGE).to_json());
        };
        let request = SubAgentSpawnRequest {
            subagent_type,
            task_description,
            display_name: Some(display_name),
            role: Some(role),
            subagent_id: None,
            parent_subagent_id: None,
        };
        match self.runtime.spawn(request).await {
            Ok(id) => {
                let status = self.runtime.get_status(&id).as_str();
                Ok(ToolPayload::ok(json!({
                    "subagent_id": id,
                    "sub_session_id": id,
                    "task_id": new_task_id(),
                    "status": status,
                }))
                .to_json())
            }
            Err(error) => Ok(ToolPayload::fail(error.to_string()).to_json()),
        }
    }

    fn render_for_llm(&self, output: &Value) -> String {
        render_payload(
            output,
            &["subagent_id", "sub_session_id", "task_id", "status"],
        )
    }
}

subagent_tool!(
    /// Blocks until the requested sub-agents reach a final status.
    SubagentWaitTool,
    "subagent_wait",
    json!({
        "type": "object",
        "properties": {
            "subagent_ids": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "description": "Subagent ids to wait for."
            },
            "timeout_ms": {
                "type": "integer",
                "default": 2_400_000,
                "description": "Wait deadline in milliseconds; clamped to 10000..=3600000."
            }
        },
        "required": ["subagent_ids"]
    })
);

#[async_trait]
impl Tool for SubagentWaitTool {
    fn card(&self) -> &ToolCard {
        &self.card
    }

    async fn invoke(&self, inputs: Value, _ctx: &ToolContext) -> Result<Value> {
        let ids = parse_subagent_ids(&inputs);
        if ids.is_empty() {
            return Ok(ToolPayload::fail("'subagent_ids' must be a non-empty list").to_json());
        }
        let timeout_ms = match resolve_timeout_ms(&inputs) {
            Ok(timeout_ms) => timeout_ms,
            Err(message) => return Ok(ToolPayload::fail(message).to_json()),
        };
        let outcomes = self
            .runtime
            .wait(&ids, Duration::from_millis(timeout_ms))
            .await;
        let mut statuses = Map::new();
        let mut results = Map::new();
        let mut timed_out = false;
        for outcome in outcomes {
            statuses.insert(
                outcome.subagent_id.clone(),
                Value::String(outcome.status.as_str().to_owned()),
            );
            if outcome.status == SubagentStatus::Completed {
                results.insert(
                    outcome.subagent_id,
                    Value::String(outcome.output.unwrap_or_default()),
                );
            } else if !outcome.is_final() {
                timed_out = true;
            }
        }
        Ok(ToolPayload::ok(json!({
            "statuses": statuses,
            "results": results,
            "output_files": {},
            "timed_out": timed_out,
        }))
        .to_json())
    }

    fn render_for_llm(&self, output: &Value) -> String {
        if let Some(error) = payload_error(output) {
            return error.to_owned();
        }
        let Some(data) = output.get("data") else {
            return String::new();
        };
        let results = data.get("results").and_then(Value::as_object);
        let mut sections = Vec::new();
        if let Some(statuses) = data.get("statuses").and_then(Value::as_object) {
            for (id, status) in statuses {
                let status = status.as_str().unwrap_or_default();
                let result = results
                    .and_then(|results| results.get(id))
                    .and_then(Value::as_str)
                    .filter(|text| !text.is_empty())
                    .unwrap_or("(no result)");
                sections.push(format!(
                    "subagent_id: {id}\nstatus: {status}\nresult:\n{result}"
                ));
            }
        }
        if data.get("timed_out").and_then(Value::as_bool) == Some(true) {
            sections.push("Timed out before every subagent reached a final status.".to_owned());
        }
        sections.join("\n\n")
    }
}

subagent_tool!(
    /// Lists the live and closed sub-agents of the parent session with capacity usage.
    SubagentListTool,
    "subagent_list",
    json!({"type": "object", "properties": {}})
);

#[async_trait]
impl Tool for SubagentListTool {
    fn card(&self) -> &ToolCard {
        &self.card
    }

    async fn invoke(&self, _inputs: Value, _ctx: &ToolContext) -> Result<Value> {
        Ok(ToolPayload::ok(self.runtime.describe_list()).to_json())
    }

    fn render_for_llm(&self, output: &Value) -> String {
        if let Some(error) = payload_error(output) {
            return error.to_owned();
        }
        let Some(data) = output.get("data") else {
            return String::new();
        };
        let capacity = data.get("capacity");
        let used = capacity
            .and_then(|capacity| capacity.get("used"))
            .and_then(Value::as_u64)
            .unwrap_or_default();
        let max = capacity
            .and_then(|capacity| capacity.get("max"))
            .and_then(Value::as_u64)
            .unwrap_or_default();
        let live = data
            .get("live_subagents")
            .and_then(Value::as_array)
            .map_or(&[][..], Vec::as_slice);
        let closed = data
            .get("closed_subagents")
            .and_then(Value::as_array)
            .map_or(&[][..], Vec::as_slice);
        format!(
            "Capacity: {used}/{max} subagents in use.\n\n{}\n\n{}",
            render_subagent_rows("Live subagents", live),
            render_subagent_rows("Closed subagents (resumable)", closed)
        )
    }
}

mod util;

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests;
