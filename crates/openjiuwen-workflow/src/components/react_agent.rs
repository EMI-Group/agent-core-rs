//! Built-in `react_agent` workflow component.
//!
//! Ports the Python package `openjiuwen/core/workflow/components/llm/react/`:
//!
//! - `react_component.py` — the workflow node wrapper this module's
//!   [`ReActAgentComponent`] replaces;
//! - `react_config.py` — [`ReActAgentConfig`];
//! - `react_executable.py` — the `invoke` that forwards the resolved inputs to the
//!   agent unchanged and turns an agent exception into
//!   `{"output": "Error in ReAct execution: <e>", "result_type": "error"}`
//!   instead of raising;
//! - `openjiuwen/core/single_agent/agents/react_agent.py` — the loop semantics
//!   carried by [`ModelReActRunner`]: `max_iterations` (default 5), the
//!   `on_max_iterations` policies (`"error"`, `"final_answer"`,
//!   `"return_observations"`), the system/user prompt pair, and the tool-calling
//!   loop that feeds tool results back to the model.
//!
//! The component owns no agent runtime. It talks to an [`AgentRunner`], a trait
//! seam that `openjiuwen-agent` and `openjiuwen-harness` adapters implement; this
//! crate stays in layer 2 of the workspace and must not depend on those crates.
//! [`ModelReActRunner`] is the bundled default: a bounded tool-calling loop over
//! [`openjiuwen_foundation::Model`], enough to run a `ReAct` node offline.
//!
//! # The pinned output contract
//!
//! `invoke` always returns a JSON object (Python's data mapping, never a raised
//! error):
//!
//! | Runner result | Node output |
//! | --- | --- |
//! | `Ok` object with a string `result_type` | the value unchanged |
//! | `Ok` object with an `output` key | `{"output": <value["output"]>, "result_type": "answer"}` |
//! | any other `Ok` value | `{"output": <value>, "result_type": "answer"}` |
//! | `Err` carrying a workflow interrupt | `{"output": <interaction.value>, "result_type": "interrupt"}` |
//! | any other `Err` | `{"output": "Error in ReAct execution: <reason>", "result_type": "error"}` |
//!
//! The failure row is Python's observable mapping, not a swallowed failure: the
//! runner's error is logged at `warn` level with the component and workflow ids,
//! and the run completes so downstream nodes can see the reason. `<reason>` is
//! the error's payload ([`openjiuwen_common::error::Error::message`]) — Python's
//! `str(e)` — because the Rust `Display` form would prepend the error kind
//! (`"workflow: boom"`), and a context-chained error already folds its context
//! into the payload.
//!
//! A workflow interrupt is data too. [`ComponentContext::interact`] has already
//! written the `__interaction__` frame when the runner returns the interrupt
//! error, so the node reports the question as its output and the run completes
//! with `COMPLETED` instead of halting with `INPUT_REQUIRED`.
//!
//! # Deviations from Python
//!
//! - **Interrupts are mapped, not propagated.** Python lets a HITL question halt
//!   the node and resume it later; here the question becomes
//!   `result_type: "interrupt"` output data (see above).
//! - **`on_max_iterations = "final_answer"` returns the last assistant text.**
//!   Python makes one extra tool-free model call and returns a rich dict with
//!   `termination_reason`, `iterations_used`, and `truncated_by_max_iterations`;
//!   this runner returns the most recent non-empty assistant text (or `""`).
//! - **`"return_observations"` returns a plain array** of
//!   `{"tool_call_id", "tool_name", "content"}` records instead of Python's
//!   finisher dict.
//! - **`"error"` reports `Max iterations reached without completion`** as an
//!   `Error::llm` failure, which the component renders through the error row.
//! - **Streaming is not reproduced.** The abilities stay `[Invoke]`, so a `ReAct`
//!   node produces exactly one output value (and therefore one node frame) per
//!   run; Python's per-chunk `llm_output` content flattening has no counterpart.
//! - **Tool calls run sequentially in call order**, one tool message per call.
//!   A failed or unknown tool is an observation the model can react to; it never
//!   aborts the loop.
//! - **The agent runtime is deferred.** Python's context engine, rails, skills,
//!   KV cache, and interrupt/resume machinery are not reproduced; a caller that
//!   needs them implements [`AgentRunner`] over the crates that provide them.
//! - **Runtime handles do not serialize.** [`ReActAgentConfig::model`],
//!   [`ReActAgentConfig::tools`], and [`ReActAgentConfig::request`] are
//!   `#[serde(skip)]`; the JSON form carries only the scalar configuration.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::{
    BaseMessage, Model, ModelRequestConfig, PromptTemplate, TemplateContent, Tool, ToolCall,
    ToolContext,
};

use crate::ability::ComponentAbility;
use crate::component::{Component, ComponentCard};
use crate::context::ComponentContext;
use crate::interrupt::find_interrupt;

/// The max-iteration policy that reports a failure.
pub const ON_MAX_ITERATIONS_ERROR: &str = "error";
/// The max-iteration policy that answers with the last assistant text.
pub const ON_MAX_ITERATIONS_FINAL_ANSWER: &str = "final_answer";
/// The max-iteration policy that answers with the collected observations.
pub const ON_MAX_ITERATIONS_RETURN_OBSERVATIONS: &str = "return_observations";

/// The three legal [`ReActAgentConfig::on_max_iterations`] values.
const MAX_ITERATION_POLICIES: [&str; 3] = [
    ON_MAX_ITERATIONS_ERROR,
    ON_MAX_ITERATIONS_FINAL_ANSWER,
    ON_MAX_ITERATIONS_RETURN_OBSERVATIONS,
];

/// The runner seam behind [`ReActAgentComponent`].
///
/// An implementation runs one `ReAct` turn for a workflow node; the returned JSON
/// is the node output (after the component's normalization). The
/// `openjiuwen-agent` `ReActAgent` and the `openjiuwen-harness` `DeepAgent` both
/// adapt to this trait through thin wrappers in their own crates — this crate
/// must not depend on them, so the seam lives here.
///
/// `ctx` exposes the run's session, cancellation token, frame writer, and HITL
/// interactions, so a runner can stream frames, ask a question, or honor
/// cancellation without owning any of that state.
#[async_trait]
pub trait AgentRunner: Send + Sync {
    /// Runs the agent for one workflow node invocation.
    ///
    /// # Errors
    ///
    /// Any failure is mapped to data by the component: an interrupt becomes
    /// `result_type: "interrupt"`, everything else `result_type: "error"`.
    async fn invoke(&self, inputs: Value, ctx: &ComponentContext) -> Result<Value>;
}

/// Configuration of a `ReAct` loop, mirroring Python's `ReActAgentConfig`.
///
/// The scalar fields round-trip through JSON (unknown fields are ignored); the
/// runtime handles (`model`, `tools`, `request`) do not serialize and are set
/// through the chainable builders.
#[derive(Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ReActAgentConfig {
    /// Maximum model rounds; Python's default is 5 and zero is rejected.
    pub max_iterations: u32,
    /// What to report when the budget runs out: `"error"`,
    /// `"final_answer"`, or `"return_observations"`.
    pub on_max_iterations: String,
    /// System prompt template (`{{name}}` placeholders); an empty string sends
    /// no system message.
    pub system_prompt_template: String,
    /// User prompt template (`{{name}}` placeholders).
    pub user_prompt_template: String,
    /// Logical model id carried for configuration parity.
    pub model_id: String,
    /// The model the default runner calls; runtime-only, never serialized.
    #[serde(skip)]
    pub model: Option<Arc<dyn Model>>,
    /// The tools the default runner may call; runtime-only, never serialized.
    #[serde(skip)]
    pub tools: Vec<Arc<dyn Tool>>,
    /// Request settings sent with every model call; runtime-only, never
    /// serialized.
    #[serde(skip)]
    pub request: ModelRequestConfig,
}

impl fmt::Debug for ReActAgentConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ReActAgentConfig")
            .field("max_iterations", &self.max_iterations)
            .field("on_max_iterations", &self.on_max_iterations)
            .field("system_prompt_template", &self.system_prompt_template)
            .field("user_prompt_template", &self.user_prompt_template)
            .field("model_id", &self.model_id)
            .field("model", &self.model.as_deref().map(Model::model_name))
            .field(
                "tools",
                &self
                    .tools
                    .iter()
                    .map(|tool| tool.card().name.clone())
                    .collect::<Vec<_>>(),
            )
            .field("request", &self.request)
            .finish()
    }
}

impl Default for ReActAgentConfig {
    /// The Python defaults: five iterations, the `"error"` policy, no system
    /// prompt, the `"{{query}}"` user prompt, an empty model id, and no runtime
    /// handles.
    fn default() -> Self {
        Self {
            max_iterations: 5,
            on_max_iterations: ON_MAX_ITERATIONS_ERROR.to_owned(),
            system_prompt_template: String::new(),
            user_prompt_template: "{{query}}".to_owned(),
            model_id: String::new(),
            model: None,
            tools: Vec::new(),
            request: ModelRequestConfig::default(),
        }
    }
}

impl ReActAgentConfig {
    /// Sets [`ReActAgentConfig::max_iterations`].
    #[must_use]
    pub fn with_max_iterations(mut self, max_iterations: u32) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Sets [`ReActAgentConfig::on_max_iterations`].
    #[must_use]
    pub fn with_on_max_iterations(mut self, policy: impl Into<String>) -> Self {
        self.on_max_iterations = policy.into();
        self
    }

    /// Sets [`ReActAgentConfig::system_prompt_template`].
    #[must_use]
    pub fn with_system_prompt(mut self, template: impl Into<String>) -> Self {
        self.system_prompt_template = template.into();
        self
    }

    /// Sets [`ReActAgentConfig::user_prompt_template`].
    #[must_use]
    pub fn with_user_prompt(mut self, template: impl Into<String>) -> Self {
        self.user_prompt_template = template.into();
        self
    }

    /// Sets [`ReActAgentConfig::model_id`].
    #[must_use]
    pub fn with_model_id(mut self, model_id: impl Into<String>) -> Self {
        self.model_id = model_id.into();
        self
    }

    /// Sets [`ReActAgentConfig::model`].
    #[must_use]
    pub fn with_model(mut self, model: Arc<dyn Model>) -> Self {
        self.model = Some(model);
        self
    }

    /// Replaces [`ReActAgentConfig::tools`].
    #[must_use]
    pub fn with_tools(mut self, tools: Vec<Arc<dyn Tool>>) -> Self {
        self.tools = tools;
        self
    }

    /// Replaces [`ReActAgentConfig::request`].
    #[must_use]
    pub fn with_request(mut self, request: ModelRequestConfig) -> Self {
        self.request = request;
        self
    }

    /// Checks the scalar configuration.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when `on_max_iterations` is not one of
    /// `"error"`, `"final_answer"`, `"return_observations"`, or when
    /// `max_iterations` is zero.
    pub fn validate(&self) -> Result<()> {
        if !MAX_ITERATION_POLICIES.contains(&self.on_max_iterations.as_str()) {
            return Err(Error::validation(format!(
                "invalid on_max_iterations '{}': expected one of {}",
                self.on_max_iterations,
                MAX_ITERATION_POLICIES.join(", ")
            )));
        }
        if self.max_iterations == 0 {
            return Err(Error::validation("max_iterations must be at least 1"));
        }
        Ok(())
    }
}

/// The built-in `react_agent` component.
///
/// `invoke` forwards the resolved inputs to its [`AgentRunner`] and normalizes
/// the answer into the `{"output", "result_type"}` shape documented at module
/// level.
pub struct ReActAgentComponent {
    runner: Arc<dyn AgentRunner>,
}

impl fmt::Debug for ReActAgentComponent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ReActAgentComponent")
            .field("runner", &"dyn AgentRunner")
            .finish()
    }
}

impl ReActAgentComponent {
    /// Wraps `runner`.
    #[must_use]
    pub fn new(runner: Arc<dyn AgentRunner>) -> Self {
        Self { runner }
    }

    /// The runner this component delegates to.
    #[must_use]
    pub fn runner(&self) -> &Arc<dyn AgentRunner> {
        &self.runner
    }

    /// Builds a component over the bundled [`ModelReActRunner`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when `config` is invalid or carries no
    /// model (`"react_agent component requires a model"`).
    pub fn from_model(config: ReActAgentConfig) -> Result<Self> {
        config.validate()?;
        if config.model.is_none() {
            return Err(Error::validation("react_agent component requires a model"));
        }
        Ok(Self {
            runner: Arc::new(ModelReActRunner::new(config)?),
        })
    }
}

/// Normalizes a runner answer into `{"output", "result_type"}`.
///
/// See the module docs for the exact rules; an already-shaped object (a string
/// `result_type`) passes through untouched.
fn normalize_success(value: Value) -> Value {
    match value {
        Value::Object(mut map) => {
            if map.get("result_type").is_some_and(Value::is_string) {
                return Value::Object(map);
            }
            let output = map.remove("output").unwrap_or(Value::Object(map));
            json!({"output": output, "result_type": "answer"})
        }
        other => json!({"output": other, "result_type": "answer"}),
    }
}

/// The reason string reported for a failed run.
///
/// This is the error payload (Python's `str(e)`), not the `Display` form: the
/// latter would prefix the Rust error kind (`"workflow: boom"`).
fn error_reason(error: &Error) -> String {
    error.message().into_owned()
}

#[async_trait]
#[allow(clippy::unnecessary_literal_bound)]
impl Component for ReActAgentComponent {
    fn card(&self) -> ComponentCard {
        ComponentCard::new("react_agent", "react_agent")
            .with_description("runs a ReAct agent for one workflow node")
            .with_abilities(vec![ComponentAbility::Invoke])
    }

    fn abilities(&self) -> Vec<ComponentAbility> {
        vec![ComponentAbility::Invoke]
    }

    fn component_type(&self) -> &str {
        "react_agent"
    }

    async fn invoke(&self, inputs: Value, ctx: &ComponentContext) -> Result<Value> {
        match self.runner.invoke(inputs, ctx).await {
            Ok(value) => Ok(normalize_success(value)),
            Err(error) => {
                if let Some(interaction) = find_interrupt(&error) {
                    // The `__interaction__` frame is already written; report the
                    // question as data so the run completes instead of halting.
                    return Ok(json!({
                        "output": interaction.value.clone(),
                        "result_type": "interrupt",
                    }));
                }
                let reason = error_reason(&error);
                tracing::warn!(
                    component = ctx.component_id(),
                    workflow = ctx.workflow_id(),
                    error = %error,
                    "react agent runner failed"
                );
                Ok(json!({
                    "output": format!("Error in ReAct execution: {reason}"),
                    "result_type": "error",
                }))
            }
        }
    }
}

/// The bundled [`AgentRunner`]: a bounded tool-calling loop over a [`Model`].
///
/// Each round renders the prompts, calls the model, and — when the assistant
/// asks for tools — runs them in call order, appending one tool message per call
/// before the next round. A round without tool calls is the final answer.
pub struct ModelReActRunner {
    config: ReActAgentConfig,
    model: Arc<dyn Model>,
}

impl fmt::Debug for ModelReActRunner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ModelReActRunner")
            .field("model", &self.model.model_name())
            .field("config", &self.config)
            .finish()
    }
}

impl ModelReActRunner {
    /// Builds a runner from `config`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when `config` is invalid or carries no
    /// model (`"react runner requires a model"`).
    pub fn new(config: ReActAgentConfig) -> Result<Self> {
        config.validate()?;
        let Some(model) = config.model.clone() else {
            return Err(Error::validation("react runner requires a model"));
        };
        Ok(Self { config, model })
    }

    /// The configuration this runner was built from.
    #[must_use]
    pub fn config(&self) -> &ReActAgentConfig {
        &self.config
    }

    /// Builds `[system?, user]` from the configured templates.
    fn build_messages(&self, inputs: &Value) -> Vec<BaseMessage> {
        let values = render_values(inputs);
        let mut messages = Vec::new();
        if !self.config.system_prompt_template.is_empty() {
            messages.push(BaseMessage::system(render_text(
                &self.config.system_prompt_template,
                &values,
            )));
        }
        messages.push(BaseMessage::user(render_text(
            &self.config.user_prompt_template,
            &values,
        )));
        messages
    }

    /// The request sent with every model call, carrying the tool definitions.
    fn request(&self) -> ModelRequestConfig {
        let request = self.config.request.clone();
        if self.config.tools.is_empty() {
            return request;
        }
        request.with_tools(
            self.config
                .tools
                .iter()
                .map(|tool| tool.card().openai_tool())
                .collect(),
        )
    }

    /// Finds a configured tool by card name, then by card id.
    fn find_tool(&self, name: &str) -> Option<&Arc<dyn Tool>> {
        self.config.tools.iter().find(|tool| {
            let card = tool.card();
            card.name == name || card.id == name
        })
    }

    /// Runs one tool call and returns its model-facing text plus its observation.
    ///
    /// Unknown tools, malformed arguments, and tool failures are text the model
    /// can react to; they never abort the loop.
    async fn run_tool(&self, call: &ToolCall, ctx: &ComponentContext) -> (String, Value) {
        let name = call.name.as_str();
        let content = match self.find_tool(name) {
            None => format!("Error: tool '{name}' is not registered"),
            Some(tool) => match call.arguments_value() {
                Err(error) => format!("Error: invalid arguments for tool '{name}': {error}"),
                Ok(arguments) => {
                    let tool_ctx = ToolContext {
                        session_id: Some(ctx.session().id().to_owned()),
                        metadata: ctx.metadata().clone(),
                        cancellation: ctx.cancellation().clone(),
                        ..ToolContext::default()
                    };
                    match tool.invoke(arguments, &tool_ctx).await {
                        Ok(output) => tool.render_for_llm(&output),
                        Err(error) => format!("Error: {error}"),
                    }
                }
            },
        };
        let observation = json!({
            "tool_call_id": call.id.clone().unwrap_or_default(),
            "tool_name": name,
            "content": content.clone(),
        });
        (content, observation)
    }
}

#[async_trait]
impl AgentRunner for ModelReActRunner {
    async fn invoke(&self, inputs: Value, ctx: &ComponentContext) -> Result<Value> {
        let mut messages = self.build_messages(&inputs);
        let request = self.request();
        let mut observations: Vec<Value> = Vec::new();
        let mut last_text = String::new();

        for _ in 0..self.config.max_iterations {
            ctx.check_cancelled()?;
            let response = self.model.invoke(&messages, &request).await?;
            let assistant = response
                .message()
                .ok_or_else(|| Error::llm("model returned no message"))?;
            let text = assistant.text().into_owned();
            if assistant.tool_calls.is_empty() {
                return Ok(Value::String(text));
            }
            if !text.is_empty() {
                last_text = text;
            }
            let calls = assistant.tool_calls.clone();
            messages.push(assistant.clone());
            for call in &calls {
                let (content, observation) = self.run_tool(call, ctx).await;
                observations.push(observation);
                messages.push(BaseMessage::tool(
                    call.id.clone().unwrap_or_default(),
                    content,
                ));
            }
        }

        match self.config.on_max_iterations.as_str() {
            ON_MAX_ITERATIONS_FINAL_ANSWER => Ok(Value::String(last_text)),
            ON_MAX_ITERATIONS_RETURN_OBSERVATIONS => Ok(Value::Array(observations)),
            // `"error"` is the validated default; any other value was rejected by
            // `ReActAgentConfig::validate`, so this arm is the same failure path.
            _ => Err(Error::llm("Max iterations reached without completion")),
        }
    }
}

/// The rendering values of `inputs`: its top-level entries plus `"inputs"`.
fn render_values(inputs: &Value) -> BTreeMap<String, Value> {
    let mut values = BTreeMap::new();
    if let Value::Object(object) = inputs {
        values.extend(
            object
                .iter()
                .map(|(key, value)| (key.clone(), value.clone())),
        );
    }
    values.insert("inputs".to_owned(), inputs.clone());
    values
}

/// Renders `{{name}}` placeholders in one template string.
///
/// An unresolvable placeholder is kept verbatim by [`PromptTemplate`], matching
/// Python's partial rendering.
fn render_text(template: &str, values: &BTreeMap<String, Value>) -> String {
    match PromptTemplate::new(template).format(values).content {
        TemplateContent::Text(text) => text,
        // `PromptTemplate::new` always builds text content; join defensively so a
        // future change cannot panic here.
        TemplateContent::Messages(messages) => messages
            .iter()
            .map(|message| message.text().into_owned())
            .collect::<String>(),
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::needless_pass_by_value,
    clippy::unnecessary_literal_bound
)]
mod tests {
    use std::sync::Arc;

    use async_trait::async_trait;
    use openjiuwen_common::error::{Error, ErrorKind, Result};
    use openjiuwen_foundation::Model;
    use openjiuwen_session::{Session, WorkflowSession};
    use openjiuwen_test_support::ScriptedModel;
    use parking_lot::Mutex;
    use serde_json::{json, Value};

    use super::*;
    use crate::interrupt::{interrupt_error, WorkflowInteraction};

    fn context() -> ComponentContext {
        let session: Arc<dyn Session> = Arc::new(WorkflowSession::new("react-tests"));
        ComponentContext::new(session, "react", "wf")
    }

    /// What a [`FakeRunner`] does when invoked.
    enum Behavior {
        Answer(Value),
        Fail(fn() -> Error),
    }

    /// An [`AgentRunner`] whose answer is fixed by the test.
    struct FakeRunner {
        behavior: Behavior,
        seen: Arc<Mutex<Vec<Value>>>,
    }

    impl FakeRunner {
        fn answer(value: Value) -> Arc<Self> {
            Arc::new(Self {
                behavior: Behavior::Answer(value),
                seen: Arc::new(Mutex::new(Vec::new())),
            })
        }

        fn fail(error: fn() -> Error) -> Arc<Self> {
            Arc::new(Self {
                behavior: Behavior::Fail(error),
                seen: Arc::new(Mutex::new(Vec::new())),
            })
        }
    }

    #[async_trait]
    impl AgentRunner for FakeRunner {
        async fn invoke(&self, inputs: Value, _ctx: &ComponentContext) -> Result<Value> {
            self.seen.lock().push(inputs);
            match &self.behavior {
                Behavior::Answer(value) => Ok(value.clone()),
                Behavior::Fail(error) => Err(error()),
            }
        }
    }

    fn component(runner: Arc<FakeRunner>) -> ReActAgentComponent {
        ReActAgentComponent::new(runner)
    }

    fn model() -> Arc<dyn Model> {
        Arc::new(ScriptedModel::text("x"))
    }

    #[test]
    fn identity_is_react_agent_with_invoke_only() {
        let component = component(FakeRunner::answer(json!("done")));
        let card = component.card();
        assert_eq!(card.id, "react_agent");
        assert_eq!(card.name, "react_agent");
        assert_eq!(card.component_type, "react_agent");
        assert_ne!(card.description, "");
        assert_eq!(card.abilities, vec![ComponentAbility::Invoke]);
        assert_eq!(component.component_type(), "react_agent");
        assert_eq!(component.abilities(), vec![ComponentAbility::Invoke]);
    }

    #[tokio::test]
    async fn inputs_are_forwarded_unchanged() {
        let runner = FakeRunner::answer(json!("done"));
        let seen = Arc::clone(&runner.seen);
        component(runner)
            .invoke(json!({"query": "hi"}), &context())
            .await
            .unwrap();
        assert_eq!(*seen.lock(), vec![json!({"query": "hi"})]);
    }

    #[tokio::test]
    async fn a_scalar_answer_becomes_the_output() {
        let output = component(FakeRunner::answer(json!("done")))
            .invoke(json!({}), &context())
            .await
            .unwrap();
        assert_eq!(output, json!({"output": "done", "result_type": "answer"}));
    }

    #[tokio::test]
    async fn an_output_key_is_unwrapped() {
        let output = component(FakeRunner::answer(json!({"output": {"a": 1}, "extra": 2})))
            .invoke(json!({}), &context())
            .await
            .unwrap();
        assert_eq!(output, json!({"output": {"a": 1}, "result_type": "answer"}));
    }

    #[tokio::test]
    async fn an_already_shaped_answer_passes_through() {
        let shaped = json!({"output": "x", "result_type": "custom"});
        let output = component(FakeRunner::answer(shaped.clone()))
            .invoke(json!({}), &context())
            .await
            .unwrap();
        assert_eq!(output, shaped);
    }

    #[tokio::test]
    async fn a_failure_becomes_error_data() {
        let output = component(FakeRunner::fail(|| Error::workflow("boom")))
            .invoke(json!({}), &context())
            .await
            .unwrap();
        assert_eq!(
            output,
            json!({"output": "Error in ReAct execution: boom", "result_type": "error"})
        );
    }

    /// The error a HITL question produces after the frame is written.
    fn interrupt() -> Error {
        interrupt_error(WorkflowInteraction::new(
            "q",
            0,
            json!({"id": "q", "text": "continue?"}),
        ))
    }

    #[tokio::test]
    async fn an_interrupt_becomes_the_question_as_data() {
        let output = component(FakeRunner::fail(interrupt))
            .invoke(json!({}), &context())
            .await
            .unwrap();
        assert_eq!(
            output,
            json!({
                "output": {"id": "q", "text": "continue?"},
                "result_type": "interrupt",
            })
        );
    }

    #[test]
    fn config_defaults_match_python() {
        let config = ReActAgentConfig::default();
        assert_eq!(config.max_iterations, 5);
        assert_eq!(config.on_max_iterations, "error");
        assert_eq!(config.system_prompt_template, "");
        assert_eq!(config.user_prompt_template, "{{query}}");
        assert_eq!(config.model_id, "");
        assert!(config.model.is_none());
        assert!(config.tools.is_empty());
        assert_eq!(config.request, ModelRequestConfig::default());
        assert!(config.validate().is_ok());
    }

    #[test]
    fn config_builders_are_chainable() {
        let config = ReActAgentConfig::default()
            .with_max_iterations(2)
            .with_on_max_iterations("return_observations")
            .with_system_prompt("You are helpful")
            .with_user_prompt("{{inputs.query}}")
            .with_model_id("m")
            .with_model(model())
            .with_request(ModelRequestConfig::new("gpt"));
        assert_eq!(config.max_iterations, 2);
        assert_eq!(config.on_max_iterations, "return_observations");
        assert_eq!(config.system_prompt_template, "You are helpful");
        assert_eq!(config.user_prompt_template, "{{inputs.query}}");
        assert_eq!(config.model_id, "m");
        assert!(config.model.is_some());
        assert_eq!(config.request.model_name, "gpt");
        assert!(config.validate().is_ok());
    }

    #[test]
    fn config_json_round_trips_and_ignores_unknown_fields() {
        let parsed: ReActAgentConfig = serde_json::from_value(json!({
            "max_iterations": 3,
            "on_max_iterations": "final_answer",
            "model_id": "m",
            "future_field": {"ignored": true},
        }))
        .unwrap();
        assert_eq!(parsed.max_iterations, 3);
        assert_eq!(parsed.on_max_iterations, "final_answer");
        assert_eq!(parsed.model_id, "m");
        // Omitted fields fall back to the documented defaults.
        assert_eq!(parsed.system_prompt_template, "");
        assert_eq!(parsed.user_prompt_template, "{{query}}");
        assert!(parsed.model.is_none());
        assert!(parsed.tools.is_empty());

        let parsed: ReActAgentConfig = serde_json::from_value(json!({
            "model": {"provider": "openai"},
            "tools": [{"name": "t"}],
        }))
        .unwrap();
        assert!(parsed.model.is_none(), "model handles never deserialize");
        assert!(parsed.tools.is_empty(), "tool handles never deserialize");

        let wire = serde_json::to_value(&parsed).unwrap();
        assert_eq!(wire["max_iterations"], json!(5));
        assert_eq!(wire["user_prompt_template"], json!("{{query}}"));
        assert!(wire.get("model").is_none());
        assert!(wire.get("tools").is_none());
        assert!(wire.get("request").is_none());
        let round_tripped: ReActAgentConfig = serde_json::from_value(wire).unwrap();
        assert_eq!(round_tripped.max_iterations, 5);
    }

    #[test]
    fn an_unknown_max_iteration_policy_is_rejected() {
        let error = ReActAgentConfig::default()
            .with_on_max_iterations("explode")
            .validate()
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert_eq!(
            error.message(),
            "invalid on_max_iterations 'explode': expected one of error, final_answer, return_observations"
        );
    }

    #[test]
    fn a_zero_iteration_budget_is_rejected() {
        let error = ReActAgentConfig::default()
            .with_max_iterations(0)
            .validate()
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert_eq!(error.message(), "max_iterations must be at least 1");
    }

    #[test]
    fn from_model_requires_a_model() {
        let error = ReActAgentComponent::from_model(ReActAgentConfig::default()).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert_eq!(error.message(), "react_agent component requires a model");

        let component =
            ReActAgentComponent::from_model(ReActAgentConfig::default().with_model(model()))
                .unwrap();
        assert_eq!(component.component_type(), "react_agent");
    }

    #[test]
    fn the_model_runner_requires_a_model() {
        let error = ModelReActRunner::new(ReActAgentConfig::default()).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert_eq!(error.message(), "react runner requires a model");
    }
}
