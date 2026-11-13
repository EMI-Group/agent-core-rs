//! Port of `openjiuwen/harness/tools/subagent/task_tool.py`: the `task_tool`.
//!
//! [`TaskTool`] is the synchronous delegation path: it spawns one sub-agent instance,
//! waits for its first turn, and returns the terminal answer. The persistent
//! `subagent_*` tools in [`super::subagent_tools`] are the non-blocking counterpart.
//!
//! Python parity: the 720 s delegation ceiling and the
//! `OPENJIUWEN_SUBAGENT_TASK_TIMEOUT_S` override (a non-positive or non-numeric value
//! keeps the default). Browser/CUA resume rules, observability hooks, and the streaming
//! execution path are deferred.
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use openjiuwen_common::error::Result;
use openjiuwen_foundation::tool::{Tool, ToolCard, ToolContext};
use serde_json::{json, Value};

use super::runtime::{SubAgentRuntime, SubAgentSpawnRequest, SubagentStatus};
use crate::tools::{tool_card, ToolPayload};

/// Default delegation ceiling in seconds (Python `DEFAULT_SUBAGENT_TASK_TIMEOUT_S`).
const DEFAULT_TASK_TIMEOUT: Duration = Duration::from_secs(720);

/// Environment variable overriding the delegation ceiling.
const TIMEOUT_ENV: &str = "OPENJIUWEN_SUBAGENT_TASK_TIMEOUT_S";

/// Largest accepted override, guarding [`Duration::try_from_secs_f64`].
const MAX_TIMEOUT_SECONDS: f64 = 1_000_000_000.0;

/// Delegates one task to a sub-agent and returns its terminal answer.
///
/// Python parity: `TaskTool`. The instance is ephemeral from the tool's point of view,
/// although the runtime keeps it alive until the parent closes it.
pub struct TaskTool {
    /// The runtime the delegated instance is spawned into.
    runtime: Arc<SubAgentRuntime>,
    /// The model-facing card built in the default language.
    card: ToolCard,
    /// How long the first turn may run before the tool reports failure.
    timeout: Duration,
}

impl TaskTool {
    /// Builds the tool over `runtime` in the default language.
    ///
    /// The delegation ceiling is read from `TIMEOUT_ENV` on construction.
    #[must_use]
    pub fn new(runtime: Arc<SubAgentRuntime>) -> Self {
        let raw = std::env::var(TIMEOUT_ENV).ok();
        let timeout = resolve_timeout(raw.as_deref());
        Self {
            runtime,
            card: Self::card_for(crate::prompts::DEFAULT_LANGUAGE),
            timeout,
        }
    }

    /// Returns this tool with `timeout` as its delegation ceiling.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Builds the model-facing card in `language`.
    ///
    /// The schema is Python's: `subagent_type` and `task_description` are required and
    /// `resume_task_id` is optional.
    #[must_use]
    pub fn card_for(language: &str) -> ToolCard {
        tool_card(
            "task_tool",
            description_for("task_tool", language),
            json!({
                "type": "object",
                "properties": {
                    "subagent_type": {
                        "type": "string",
                        "description": "Registered subagent type name."
                    },
                    "task_description": {
                        "type": "string",
                        "description": "Complete task prompt for the subagent; it starts with no memory of this conversation."
                    },
                    "resume_task_id": {
                        "type": "string",
                        "description": "The resume_task_id returned by a previous task_tool call to continue that subagent; omit it for a new task."
                    }
                },
                "required": ["subagent_type", "task_description"]
            }),
            true,
        )
    }

    /// Returns the configured delegation ceiling.
    #[must_use]
    pub fn timeout(&self) -> Duration {
        self.timeout
    }
}

#[async_trait]
impl Tool for TaskTool {
    fn card(&self) -> &ToolCard {
        &self.card
    }

    async fn invoke(&self, inputs: Value, _ctx: &ToolContext) -> Result<Value> {
        let subagent_type = inputs
            .get("subagent_type")
            .and_then(Value::as_str)
            .map_or("", str::trim);
        let task_description = inputs
            .get("task_description")
            .and_then(Value::as_str)
            .map_or("", str::trim);
        if subagent_type.is_empty() || task_description.is_empty() {
            return Ok(
                ToolPayload::fail("'subagent_type' and 'task_description' are required").to_json(),
            );
        }
        if !self.runtime.registry().contains(subagent_type) {
            return Ok(ToolPayload::fail(format!(
                "Subagent type '{subagent_type}' is not available through task_tool"
            ))
            .to_json());
        }
        let resume_task_id = inputs
            .get("resume_task_id")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|id| !id.is_empty())
            .map(str::to_owned);
        let request = SubAgentSpawnRequest {
            subagent_type: subagent_type.to_owned(),
            task_description: task_description.to_owned(),
            display_name: None,
            role: None,
            subagent_id: resume_task_id,
            parent_subagent_id: None,
        };
        let id = match self.runtime.spawn(request).await {
            Ok(id) => id,
            Err(error) => return Ok(ToolPayload::fail(error.to_string()).to_json()),
        };
        let Some(outcome) = self
            .runtime
            .wait(std::slice::from_ref(&id), self.timeout)
            .await
            .pop()
        else {
            return Ok(ToolPayload::fail(format!(
                "Subagent {subagent_type} execution failed: no result"
            ))
            .to_json());
        };
        if outcome.status == SubagentStatus::Completed {
            return Ok(ToolPayload::ok(json!({
                "output": outcome.output.unwrap_or_default(),
                "agent_id": outcome.agent_id,
                "subagent_type": outcome.subagent_type,
            }))
            .to_json());
        }
        let reason = outcome
            .error
            .unwrap_or_else(|| outcome.status.as_str().to_owned());
        Ok(ToolPayload::fail(format!(
            "Subagent {} execution failed: {reason}",
            outcome.subagent_type
        ))
        .to_json())
    }

    fn render_for_llm(&self, output: &Value) -> String {
        if let Some(text) = output
            .get("data")
            .and_then(|data| data.get("output"))
            .and_then(Value::as_str)
            .filter(|text| !text.is_empty())
        {
            return text.to_owned();
        }
        output.get("error").and_then(Value::as_str).map_or_else(
            || "Subagent finished without output.".to_owned(),
            str::to_owned,
        )
    }
}

/// Returns the model-facing description of `name` in `language`.
///
/// The coding-tool description table is consulted first; the prompt tool table supplies the
/// bilingual text of the Python sub-agent tools. Rust-port extensions that have no Python row
/// (currently `subagent_transcript`) fall back to [`extension_description`]. Unknown names
/// yield an empty string instead of panicking.
pub(crate) fn description_for(name: &str, language: &str) -> &'static str {
    let description = crate::tools::descriptions::description(name, language);
    if !description.is_empty() {
        return description;
    }
    crate::prompts::description(name, language)
        .unwrap_or_else(|| extension_description(name, language))
}

/// Descriptions of the model-facing tools that exist only in the Rust port.
///
/// Each row is `(name, chinese, english)`; the Python agent never sees these names, so they
/// cannot live in the prompt description table that mirrors `openjiuwen.prompts`.
const EXTENSION_TOOL_DESCRIPTIONS: [(&str, &str, &str); 1] = [(
    "subagent_transcript",
    "读取指定子代理的有界执行记录；支持 after_seq 分页与 limit 截断，返回条目、摘要以及是否还有更多。",
    "Read the bounded transcript of a subagent instance; supports after_seq pagination and a limit, and reports the entries, the summary, and whether more remain.",
)];

/// Returns the description of the Rust-only tool `name`.
///
/// `language` is trimmed and lower-cased, so `en`/`en-*` select the English text and anything
/// else the Chinese one, matching the prompt description table. Unknown names yield `""`.
fn extension_description(name: &str, language: &str) -> &'static str {
    let Some((_, chinese, english)) = EXTENSION_TOOL_DESCRIPTIONS
        .iter()
        .find(|(tool, _, _)| *tool == name)
    else {
        return "";
    };
    let key = language.trim().to_lowercase();
    if key == "en" || key.starts_with("en-") {
        english
    } else {
        chinese
    }
}

/// Resolves the delegation ceiling from an explicit environment value.
///
/// An absent, blank, non-numeric or non-positive value keeps the Python default; absurd
/// finite values are clamped so conversion to a [`Duration`] cannot panic.
fn resolve_timeout(raw: Option<&str>) -> Duration {
    raw.and_then(parse_timeout).unwrap_or(DEFAULT_TASK_TIMEOUT)
}

/// Parses a positive number of seconds, clamping absurd finite values.
fn parse_timeout(raw: &str) -> Option<Duration> {
    let seconds = raw.trim().parse::<f64>().ok()?;
    if !seconds.is_finite() || seconds <= 0.0 {
        return None;
    }
    Duration::try_from_secs_f64(seconds.min(MAX_TIMEOUT_SECONDS)).ok()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::subagents::test_support::echo_runtime;

    async fn invoke(tool: &TaskTool, inputs: Value) -> Value {
        tool.invoke(inputs, &ToolContext::new()).await.unwrap()
    }

    #[test]
    fn card_exposes_the_python_contract() {
        let card = TaskTool::card_for("en");
        assert_eq!(card.name, "task_tool");
        assert_ne!(card.description, "");
        assert_eq!(card.input_params["type"], "object");
        let mut required: Vec<String> = card.input_params["required"]
            .as_array()
            .unwrap()
            .iter()
            .map(|name| name.as_str().unwrap().to_owned())
            .collect();
        required.sort();
        assert_eq!(
            required,
            vec!["subagent_type".to_owned(), "task_description".to_owned()]
        );
        let properties = card.input_params["properties"].as_object().unwrap();
        let mut keys: Vec<&str> = properties.keys().map(String::as_str).collect();
        keys.sort_unstable();
        assert_eq!(
            keys,
            vec!["resume_task_id", "subagent_type", "task_description"]
        );
        for name in ["subagent_type", "task_description", "resume_task_id"] {
            assert_eq!(properties[name]["type"], "string", "{name}");
            assert_ne!(properties[name]["description"].as_str().unwrap(), "");
        }
        assert_ne!(
            TaskTool::card_for("en").description,
            TaskTool::card_for("cn").description
        );
    }

    #[tokio::test]
    async fn task_tool_runs_the_subagent_to_completion() {
        let (runtime, _factory) = echo_runtime();
        let tool = TaskTool::new(Arc::clone(&runtime));
        let output = invoke(
            &tool,
            json!({
                "subagent_type": "general-purpose",
                "task_description": "hello"
            }),
        )
        .await;
        assert_eq!(output["success"], true);
        let data = &output["data"];
        assert_eq!(data["output"], "echo:hello");
        assert_eq!(data["subagent_type"], "general-purpose");
        assert_ne!(data["agent_id"].as_str().unwrap(), "");
        assert_eq!(runtime.capacity().0, 1);
    }

    #[tokio::test]
    async fn unknown_type_is_rejected_without_spawning() {
        let (runtime, _factory) = echo_runtime();
        let tool = TaskTool::new(Arc::clone(&runtime));
        let output = invoke(
            &tool,
            json!({"subagent_type": "nope", "task_description": "hello"}),
        )
        .await;
        assert_eq!(output["success"], false);
        assert_eq!(
            output["error"],
            "Subagent type 'nope' is not available through task_tool"
        );
        assert_eq!(runtime.capacity().0, 0);
    }

    #[tokio::test]
    async fn missing_task_description_is_rejected() {
        let (runtime, _factory) = echo_runtime();
        let tool = TaskTool::new(runtime);
        let output = invoke(&tool, json!({"subagent_type": "general-purpose"})).await;
        assert_eq!(output["success"], false);
        assert_eq!(
            output["error"],
            "'subagent_type' and 'task_description' are required"
        );
    }

    #[tokio::test]
    async fn an_errored_turn_is_reported_as_a_failed_payload() {
        let factory = crate::subagents::test_support::FakeFactory::new(
            crate::subagents::test_support::Behavior::ErrorResult,
        );
        let runtime = crate::subagents::test_support::build_runtime(Arc::clone(&factory));
        let tool = TaskTool::new(runtime);
        let output = invoke(
            &tool,
            json!({"subagent_type": "general-purpose", "task_description": "boom"}),
        )
        .await;
        assert_eq!(output["success"], false);
        assert_eq!(
            output["error"],
            "Subagent general-purpose execution failed: model failed"
        );
    }

    #[test]
    fn timeout_resolution_prefers_a_valid_environment_value() {
        assert_eq!(resolve_timeout(None), Duration::from_secs(720));
        assert_eq!(resolve_timeout(Some("12.5")), Duration::from_secs_f64(12.5));
        assert_eq!(resolve_timeout(Some("  ")), Duration::from_secs(720));
        assert_eq!(resolve_timeout(Some("abc")), Duration::from_secs(720));
        assert_eq!(resolve_timeout(Some("0")), Duration::from_secs(720));
        assert_eq!(resolve_timeout(Some("-4")), Duration::from_secs(720));
        assert_eq!(resolve_timeout(Some("inf")), Duration::from_secs(720));
        assert_eq!(
            resolve_timeout(Some("1e300")),
            Duration::from_secs(1_000_000_000)
        );
    }

    #[test]
    fn environment_override_is_read_by_new() {
        let (runtime, _factory) = echo_runtime();
        let previous = std::env::var(TIMEOUT_ENV).ok();
        std::env::set_var(TIMEOUT_ENV, "5");
        let timeout = TaskTool::new(runtime).timeout();
        match previous {
            Some(value) => std::env::set_var(TIMEOUT_ENV, value),
            None => std::env::remove_var(TIMEOUT_ENV),
        }
        assert_eq!(timeout, Duration::from_secs(5));
    }

    #[test]
    fn with_timeout_overrides_the_configured_ceiling() {
        let (runtime, _factory) = echo_runtime();
        let tool = TaskTool::new(runtime).with_timeout(Duration::from_secs(3));
        assert_eq!(tool.timeout(), Duration::from_secs(3));
    }

    #[test]
    fn render_returns_the_answer_or_a_fallback() {
        let (runtime, _factory) = echo_runtime();
        let tool = TaskTool::new(runtime);
        assert_eq!(
            tool.render_for_llm(
                &json!({"success": true, "data": {"output": "done"}, "error": null})
            ),
            "done"
        );
        assert_eq!(
            tool.render_for_llm(&json!({"success": true, "data": {"output": ""}, "error": null})),
            "Subagent finished without output."
        );
        assert_eq!(
            tool.render_for_llm(&json!({"success": false, "data": null, "error": "boom"})),
            "boom"
        );
    }
}
