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
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use openjiuwen_common::error::Result;
use openjiuwen_foundation::tool::{Tool, ToolCard, ToolContext};
use serde_json::{json, Map, Value};

use super::runtime::{SubAgentRuntime, SubAgentSpawnRequest, SubagentStatus};
use super::task_tool::description_for;
use crate::tools::{compact_json, render_fields, tool_card, ToolPayload};

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
            .runtime
            .send_input(&subagent_id, &query, interrupt)
            .await
        {
            return Ok(ToolPayload::fail(error.to_string()).to_json());
        }
        let status = self.runtime.get_status(&subagent_id).as_str();
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
        match self.runtime.close(&subagent_id).await {
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
        match self.runtime.resume(&subagent_id).await {
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

/// Returns the trimmed, non-empty string value of `key`, if any.
fn non_empty(inputs: &Value, key: &str) -> Option<String> {
    inputs
        .get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
}

/// Normalizes `subagent_ids`, accepting a singular `subagent_id` as a fallback.
fn parse_subagent_ids(inputs: &Value) -> Vec<String> {
    let raw = match inputs.get("subagent_ids") {
        Some(Value::Null) | None => inputs.get("subagent_id"),
        Some(value) => Some(value),
    };
    match raw {
        Some(Value::String(id)) => normalize_ids(std::iter::once(id.as_str())),
        Some(Value::Array(items)) => normalize_ids(items.iter().filter_map(Value::as_str)),
        _ => Vec::new(),
    }
}

/// Trims IDs, drops blanks, and keeps input order.
fn normalize_ids<'a>(ids: impl Iterator<Item = &'a str>) -> Vec<String> {
    ids.map(str::trim)
        .filter(|id| !id.is_empty())
        .map(str::to_owned)
        .collect()
}

/// Resolves and clamps the `subagent_wait` deadline.
///
/// An absent deadline keeps [`DEFAULT_WAIT_TIMEOUT_MS`]; a present but non-integer value is
/// a failed payload, matching Python's `isinstance(timeout_ms, int)` check.
fn resolve_timeout_ms(inputs: &Value) -> std::result::Result<u64, &'static str> {
    match inputs.get("timeout_ms") {
        None => Ok(DEFAULT_WAIT_TIMEOUT_MS),
        Some(value) => value
            .as_i64()
            .map(|timeout_ms| {
                timeout_ms
                    .clamp(MIN_WAIT_TIMEOUT_MS, MAX_WAIT_TIMEOUT_MS)
                    .unsigned_abs()
            })
            .ok_or("'timeout_ms' must be an integer"),
    }
}

/// Returns a new Python-style `task_id` (32 hex characters).
fn new_task_id() -> String {
    uuid::Uuid::new_v4().simple().to_string()
}

/// Returns the failure message of a payload, or `None` on success.
fn payload_error(output: &Value) -> Option<&str> {
    if output.get("success").and_then(Value::as_bool) == Some(false) {
        output.get("error").and_then(Value::as_str)
    } else {
        None
    }
}

/// Renders the listed fields of a successful payload in Python `render_fields` style.
fn render_payload(output: &Value, keys: &[&str]) -> String {
    if let Some(error) = payload_error(output) {
        return error.to_owned();
    }
    let Some(data) = output.get("data").and_then(Value::as_object) else {
        return String::new();
    };
    let fields: Vec<(&str, Value)> = keys
        .iter()
        .filter_map(|key| data.get(*key).map(|value| (*key, value.clone())))
        .collect();
    render_fields(&fields)
}

/// Renders one live/closed sub-agent list, Python `_render_subagent_rows` style.
fn render_subagent_rows(title: &str, rows: &[Value]) -> String {
    if rows.is_empty() {
        return format!("{title}: none");
    }
    let lines: Vec<String> = rows
        .iter()
        .map(|row| {
            let fields: Vec<String> = SUBAGENT_ROW_KEYS
                .iter()
                .filter_map(|key| {
                    let value = row.get(*key)?;
                    let text = match value {
                        Value::Null => return None,
                        Value::String(text) if text.is_empty() => return None,
                        Value::String(text) => text.clone(),
                        other => compact_json(other),
                    };
                    Some(format!("{key}: {text}"))
                })
                .collect();
            format!("- {}", fields.join(" | "))
        })
        .collect();
    format!("{title}:\n{}", lines.join("\n"))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::subagents::test_support::{build_runtime, echo_runtime, Behavior, FakeFactory};

    async fn call(tool: &impl Tool, inputs: Value) -> Value {
        tool.invoke(inputs, &ToolContext::new()).await.unwrap()
    }

    fn sorted(values: &[&str]) -> Vec<String> {
        let mut values: Vec<String> = values.iter().map(|value| (*value).to_owned()).collect();
        values.sort();
        values
    }

    fn sorted_strings(value: &Value) -> Vec<String> {
        let mut values: Vec<String> = value
            .as_array()
            .map(|items| {
                items
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_owned)
                    .collect()
            })
            .unwrap_or_default();
        values.sort();
        values
    }

    fn property_keys(card: &ToolCard) -> Vec<String> {
        let mut keys: Vec<String> = card.input_params["properties"]
            .as_object()
            .map(|properties| properties.keys().cloned().collect())
            .unwrap_or_default();
        keys.sort();
        keys
    }

    async fn spawn_one(runtime: &Arc<SubAgentRuntime>) -> (String, String) {
        let tool = SubagentSpawnTool::new(Arc::clone(runtime));
        let spawned = call(
            &tool,
            json!({
                "subagent_type": "general-purpose",
                "task_description": "task",
                "display_name": "Worker",
                "role": "tester",
            }),
        )
        .await;
        (
            spawned["data"]["subagent_id"].as_str().unwrap().to_owned(),
            spawned["data"]["task_id"].as_str().unwrap().to_owned(),
        )
    }

    async fn wait_one(runtime: &Arc<SubAgentRuntime>, id: &str) -> Value {
        let tool = SubagentWaitTool::new(Arc::clone(runtime));
        call(&tool, json!({"subagent_ids": [id]})).await
    }

    #[test]
    fn cards_expose_the_python_names_and_schemas() {
        let cases: Vec<(ToolCard, &str, &[&str], &[&str])> = vec![
            (
                SubagentSpawnTool::card_for("en"),
                "subagent_spawn",
                &["subagent_type", "task_description", "display_name", "role"],
                &["subagent_type", "task_description", "display_name", "role"],
            ),
            (
                SubagentWaitTool::card_for("en"),
                "subagent_wait",
                &["subagent_ids"],
                &["subagent_ids", "timeout_ms"],
            ),
            (SubagentListTool::card_for("en"), "subagent_list", &[], &[]),
            (
                SubagentSendInputTool::card_for("en"),
                "subagent_send_input",
                &["subagent_id", "query"],
                &["subagent_id", "query", "interrupt"],
            ),
            (
                SubagentCloseTool::card_for("en"),
                "subagent_close",
                &["subagent_id"],
                &["subagent_id"],
            ),
            (
                SubagentResumeTool::card_for("en"),
                "subagent_resume",
                &["subagent_id"],
                &["subagent_id"],
            ),
        ];
        for (card, name, required, properties) in cases {
            assert_eq!(card.name, name);
            assert_ne!(card.description, "", "{name} has no description");
            assert_eq!(card.input_params["type"], "object", "{name}");
            assert_eq!(
                sorted_strings(&card.input_params["required"]),
                sorted(required),
                "{name}"
            );
            assert_eq!(property_keys(&card), sorted(properties), "{name}");
        }
        let wait = SubagentWaitTool::card_for("en");
        assert_eq!(
            wait.input_params["properties"]["subagent_ids"]["items"]["type"],
            "string"
        );
        assert_eq!(
            wait.input_params["properties"]["subagent_ids"]["minItems"],
            1
        );
        assert_eq!(
            wait.input_params["properties"]["timeout_ms"]["default"],
            2_400_000
        );
        assert_eq!(
            SubagentSendInputTool::card_for("en").input_params["properties"]["interrupt"]
                ["default"],
            false
        );
    }

    #[tokio::test]
    async fn subagent_tools_returns_the_python_order() {
        let (runtime, _factory) = echo_runtime();
        let tools = subagent_tools(runtime);
        let names: Vec<&str> = tools.iter().map(|tool| tool.card().name.as_str()).collect();
        assert_eq!(
            names,
            vec![
                "subagent_spawn",
                "subagent_wait",
                "subagent_list",
                "subagent_send_input",
                "subagent_close",
                "subagent_resume",
            ]
        );
        for tool in &tools {
            assert_ne!(tool.card().description, "");
        }
        let listed = tools[2]
            .invoke(json!({}), &ToolContext::new())
            .await
            .unwrap();
        assert_eq!(listed["success"], true);
    }

    #[tokio::test]
    async fn spawn_returns_ids_and_initial_status() {
        let (runtime, factory) = echo_runtime();
        let tool = SubagentSpawnTool::new(Arc::clone(&runtime));
        let output = call(
            &tool,
            json!({
                "subagent_type": "general-purpose",
                "task_description": "task",
                "display_name": "Worker",
                "role": "tester",
            }),
        )
        .await;
        assert_eq!(output["success"], true);
        let data = &output["data"];
        let id = data["subagent_id"].as_str().unwrap();
        assert_eq!(data["sub_session_id"], id);
        assert_eq!(data["task_id"].as_str().unwrap().len(), 32);
        assert!(matches!(
            data["status"].as_str().unwrap(),
            "pending_init" | "running"
        ));
        let recorded = factory.recorded();
        assert_eq!(recorded.len(), 1);
        assert_eq!(recorded[0].1, id);
        assert_eq!(recorded[0].0.display_name.as_deref(), Some("Worker"));
        assert_eq!(recorded[0].0.role.as_deref(), Some("tester"));
        assert_eq!(runtime.capacity().0, 1);
    }

    #[tokio::test]
    async fn spawn_rejects_incomplete_payloads_and_unknown_types() {
        let (runtime, _factory) = echo_runtime();
        let tool = SubagentSpawnTool::new(runtime);
        let incomplete = call(
            &tool,
            json!({
                "subagent_type": "general-purpose",
                "task_description": "task",
                "display_name": "Worker",
            }),
        )
        .await;
        assert_eq!(incomplete["success"], false);
        assert_eq!(incomplete["error"], SPAWN_REQUIRED_MESSAGE);

        let unknown = call(
            &tool,
            json!({
                "subagent_type": "nope",
                "task_description": "task",
                "display_name": "Worker",
                "role": "tester",
            }),
        )
        .await;
        assert_eq!(unknown["success"], false);
        assert!(unknown["error"]
            .as_str()
            .unwrap()
            .contains("subagent type not found"));
    }

    #[tokio::test]
    async fn wait_returns_statuses_and_results_after_a_spawn() {
        let (runtime, _factory) = echo_runtime();
        let (id, _task_id) = spawn_one(&runtime).await;
        let tool = SubagentWaitTool::new(Arc::clone(&runtime));
        // `timeout_ms` 1 exercises the minimum clamp without waiting ten seconds.
        let output = call(
            &tool,
            json!({"subagent_ids": [id.as_str()], "timeout_ms": 1}),
        )
        .await;
        assert_eq!(output["success"], true);
        let data = &output["data"];
        assert_eq!(data["statuses"][id.as_str()], "completed");
        assert_eq!(data["results"][id.as_str()], "echo:task");
        assert_eq!(data["output_files"], json!({}));
        assert_eq!(data["timed_out"], false);
    }

    #[tokio::test]
    async fn wait_collects_a_blocked_subagent_once_the_gate_opens() {
        let gate = Arc::new(tokio::sync::Notify::new());
        let factory = FakeFactory::new(Behavior::Blocked(Arc::clone(&gate)));
        let runtime = build_runtime(factory);
        let (id, _task_id) = spawn_one(&runtime).await;
        gate.notify_one();
        let output = wait_one(&runtime, &id).await;
        assert_eq!(output["data"]["statuses"][id.as_str()], "completed");
        assert_eq!(output["data"]["timed_out"], false);
    }

    #[tokio::test]
    async fn wait_rejects_an_empty_list_and_a_non_integer_timeout() {
        let (runtime, _factory) = echo_runtime();
        let wait = SubagentWaitTool::new(runtime);

        for payload in [json!({"subagent_ids": []}), json!({"subagent_ids": ["  "]})] {
            let output = call(&wait, payload).await;
            assert_eq!(output["success"], false);
            assert_eq!(output["error"], "'subagent_ids' must be a non-empty list");
        }

        let fractional = call(&wait, json!({"subagent_ids": ["x"], "timeout_ms": 1.5})).await;
        assert_eq!(fractional["success"], false);
        assert_eq!(fractional["error"], "'timeout_ms' must be an integer");
    }

    #[test]
    fn timeout_resolution_clamps_to_the_python_bounds() {
        assert_eq!(
            resolve_timeout_ms(&json!({})).ok(),
            Some(DEFAULT_WAIT_TIMEOUT_MS)
        );
        assert_eq!(
            resolve_timeout_ms(&json!({"timeout_ms": 1})).ok(),
            Some(10_000)
        );
        assert_eq!(
            resolve_timeout_ms(&json!({"timeout_ms": 9_999_999})).ok(),
            Some(3_600_000)
        );
        assert!(resolve_timeout_ms(&json!({"timeout_ms": 1.5})).is_err());
        assert!(resolve_timeout_ms(&json!({"timeout_ms": "1000"})).is_err());
    }

    #[tokio::test]
    async fn list_reports_capacity_and_subagent_buckets() {
        let (runtime, _factory) = echo_runtime();
        let (id, _task_id) = spawn_one(&runtime).await;
        wait_one(&runtime, &id).await;

        let list = SubagentListTool::new(runtime);
        let output = call(&list, json!({})).await;
        assert_eq!(output["success"], true);
        let data = &output["data"];
        for key in [
            "capacity",
            "subagents",
            "live_subagents",
            "closed_subagents",
            "summary",
        ] {
            assert!(data.get(key).is_some(), "missing {key}");
        }
        assert_eq!(data["capacity"]["used"], 1);
        assert_eq!(data["capacity"]["max"], 10);
        assert_eq!(data["summary"]["live_count"], 1);
        assert_eq!(data["summary"]["closed_count"], 0);
        assert!(list
            .render_for_llm(&output)
            .starts_with("Capacity: 1/10 subagents in use."));
    }

    #[tokio::test]
    async fn send_input_queues_a_new_task_on_a_completed_subagent() {
        let (runtime, _factory) = echo_runtime();
        let (id, first_task) = spawn_one(&runtime).await;
        wait_one(&runtime, &id).await;

        let send = SubagentSendInputTool::new(runtime);
        let output = call(&send, json!({"subagent_id": id.as_str(), "query": "again"})).await;
        assert_eq!(output["success"], true);
        let data = &output["data"];
        assert_eq!(data["subagent_id"], id);
        assert_ne!(data["task_id"].as_str().unwrap(), first_task);
        assert_eq!(data["task_id"].as_str().unwrap().len(), 32);
        assert!(matches!(
            data["status"].as_str().unwrap(),
            "pending_init" | "running" | "completed"
        ));
    }

    #[tokio::test]
    async fn send_input_rejects_missing_fields_and_a_closed_target() {
        let (runtime, _factory) = echo_runtime();
        let (id, _task_id) = spawn_one(&runtime).await;
        wait_one(&runtime, &id).await;
        let close = SubagentCloseTool::new(Arc::clone(&runtime));
        call(&close, json!({"subagent_id": id.as_str()})).await;

        let send = SubagentSendInputTool::new(runtime);
        let missing_id = call(&send, json!({"query": "again"})).await;
        assert_eq!(missing_id["success"], false);
        assert_eq!(missing_id["error"], "'subagent_id' is required");

        let missing_query = call(&send, json!({"subagent_id": id.as_str()})).await;
        assert_eq!(missing_query["success"], false);
        assert_eq!(missing_query["error"], "'query' is required");

        let bad_interrupt = call(
            &send,
            json!({"subagent_id": id.as_str(), "query": "again", "interrupt": "yes"}),
        )
        .await;
        assert_eq!(bad_interrupt["success"], false);
        assert_eq!(bad_interrupt["error"], "'interrupt' must be a boolean");

        let closed = call(&send, json!({"subagent_id": id.as_str(), "query": "again"})).await;
        assert_eq!(closed["success"], false);
        assert!(closed["error"]
            .as_str()
            .unwrap()
            .contains("subagent closed or not found"));
    }

    #[tokio::test]
    async fn close_is_idempotent_and_resume_restores_the_instance() {
        let (runtime, _factory) = echo_runtime();
        let (id, _task_id) = spawn_one(&runtime).await;
        wait_one(&runtime, &id).await;

        let close = SubagentCloseTool::new(Arc::clone(&runtime));
        let missing = call(&close, json!({})).await;
        assert_eq!(missing["success"], false);
        assert_eq!(missing["error"], "'subagent_id' is required");
        let first = call(&close, json!({"subagent_id": id.as_str()})).await;
        assert_eq!(first["success"], true);
        assert_eq!(first["data"]["previous_status"], "completed");
        let second = call(&close, json!({"subagent_id": id.as_str()})).await;
        assert_eq!(second["success"], true);
        assert_eq!(second["data"]["previous_status"], "closed");

        let resume = SubagentResumeTool::new(runtime);
        let blank = call(&resume, json!({"subagent_id": "  "})).await;
        assert_eq!(blank["success"], false);
        assert_eq!(blank["error"], "'subagent_id' is required");
        let restored = call(&resume, json!({"subagent_id": id.as_str()})).await;
        assert_eq!(restored["success"], true);
        assert_eq!(restored["data"]["restored"], true);
        assert_eq!(restored["data"]["status"], "completed");

        let unknown = call(&resume, json!({"subagent_id": "missing"})).await;
        assert_eq!(unknown["success"], false);
        assert!(unknown["error"]
            .as_str()
            .unwrap()
            .contains("subagent not found"));
    }

    #[test]
    fn renderers_describe_payloads_and_errors() {
        let (runtime, _factory) = echo_runtime();
        let spawn = SubagentSpawnTool::new(Arc::clone(&runtime));
        assert_eq!(
            spawn.render_for_llm(&json!({
                "success": true,
                "data": {
                    "subagent_id": "sub-1",
                    "sub_session_id": "sub-1",
                    "task_id": "abc",
                    "status": "pending_init",
                },
                "error": null,
            })),
            "subagent_id: sub-1\nsub_session_id: sub-1\ntask_id: abc\nstatus: pending_init"
        );
        assert_eq!(
            spawn.render_for_llm(&json!({"success": false, "data": null, "error": "boom"})),
            "boom"
        );
    }
}
