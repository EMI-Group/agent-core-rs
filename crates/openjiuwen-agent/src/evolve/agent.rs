//! [`ReActAgentEvolve`]: the `ReActAgent` execution path plus tunable
//! operators.
//!
//! Ports `openjiuwen/core/single_agent/agents/react_agent_evolve.py`. The
//! agent delegates every call to an inner [`ReActAgent`]; the operators are
//! parameter handles whose updates are pushed back into that inner agent by
//! [`ParameterCallback`]s, so an external optimiser can tune a live agent
//! without owning a second loop implementation.
use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::BoxStream;
use openjiuwen_common::error::Result;
use openjiuwen_foundation::{BaseMessage, Model, Tool};
use openjiuwen_session::Session;
use parking_lot::RwLock;
use serde_json::Value;

use super::description_override::DescriptionOverrideTool;
use crate::abilities::AddAbilityResult;
use crate::agent::Agent;
use crate::callbacks::CallbackFramework;
use crate::card::AgentCard;
use crate::operator::{
    ApplyResult, LLMCallOperator, Operator, ParameterCallback, ToolCallOperator, TunableSpec,
    UpdateValue, DEFAULT_USER_PROMPT, TOOL_DESCRIPTION_TARGET,
};
use crate::output::{AgentChunk, AgentOutput};
use crate::policy::ClientToolRegistry;
use crate::rails::AgentRail;
use crate::react::{ReActAgent, ReActAgentConfig};

/// Identifier of the prompt operator (Python's `"react_llm"`).
pub const LLM_OPERATOR_ID: &str = "react_llm";

/// Identifier of the tool-description operator (Python's `"react_tool"`).
pub const TOOL_OPERATOR_ID: &str = "react_tool";

/// A type-erased operator handle keyed by [`Operator::operator_id`].
pub type SharedOperator = Arc<RwLock<Box<dyn Operator>>>;

/// A shared, typed handle over the prompt operator.
pub type SharedLlmOperator = Arc<RwLock<LLMCallOperator>>;

/// A shared, typed handle over the tool-description operator.
pub type SharedToolOperator = Arc<RwLock<ToolCallOperator>>;

/// The operator set of one evolve agent.
struct Operators {
    /// Typed handle over the prompt operator.
    llm: SharedLlmOperator,
    /// Typed handle over the tool-description operator.
    tool: SharedToolOperator,
    /// Type-erased handles keyed by operator id.
    shared: BTreeMap<String, SharedOperator>,
}

/// A type-erased handle forwarding to the typed operator behind `inner`.
///
/// This keeps [`ReActAgentEvolve::get_operators`] and the typed accessors on
/// the same state: both observe (and mutate) the one operator value.
struct OperatorHandle<O: Operator> {
    /// Captured id, because [`Operator::operator_id`] returns a borrow.
    operator_id: String,
    /// The typed operator shared with the evolve agent.
    inner: Arc<RwLock<O>>,
}

impl<O: Operator> OperatorHandle<O> {
    /// Builds a handle for `operator_id` over `inner`.
    fn new(operator_id: &str, inner: Arc<RwLock<O>>) -> Self {
        Self {
            operator_id: operator_id.to_owned(),
            inner,
        }
    }
}

#[async_trait]
impl<O: Operator + 'static> Operator for OperatorHandle<O> {
    fn operator_id(&self) -> &str {
        &self.operator_id
    }

    fn get_tunables(&self) -> Vec<TunableSpec> {
        self.inner.read().get_tunables()
    }

    fn get_state(&self) -> Value {
        self.inner.read().get_state()
    }

    fn set_parameter(&mut self, target: &str, value: Value) -> Result<ApplyResult> {
        self.inner.write().set_parameter(target, value)
    }

    fn load_state(&mut self, state: Value) -> Result<()> {
        self.inner.write().load_state(state)
    }

    fn apply_update(&mut self, target: &str, update: UpdateValue) -> Result<ApplyResult> {
        self.inner.write().apply_update(target, update)
    }
}

/// A [`ReActAgent`] with tunable operators for self-evolution training.
///
/// Execution is fully inherited: [`ReActAgentEvolve::react`] exposes the inner
/// agent, and [`AgentCard`] / [`Agent::invoke`] / [`Agent::stream`] delegate to
/// it. The operators returned by [`ReActAgentEvolve::get_operators`] contain
/// the tunable prompt and tool descriptions and keep the inner agent in sync:
///
/// ```
/// use openjiuwen_agent::evolve::ReActAgentEvolve;
/// use openjiuwen_agent::operator::Operator;
/// use openjiuwen_agent::AgentCard;
/// use serde_json::json;
///
/// let agent = ReActAgentEvolve::new(AgentCard::new("evolve"));
/// let operators = agent.get_operators();
/// assert_eq!(operators.keys().cloned().collect::<Vec<_>>(), ["react_llm", "react_tool"]);
/// let llm = operators["react_llm"].clone();
/// llm.write().set_parameter("system_prompt", json!("be terse")).unwrap();
/// assert_eq!(agent.react().config().prompt_template[0].text(), "be terse");
/// ```
pub struct ReActAgentEvolve {
    /// The inner agent that owns the loop, model, tools, rails, and session.
    agent: Arc<ReActAgent>,
    /// The current operator set, rebuilt by [`ReActAgentEvolve::configure`].
    operators: RwLock<Operators>,
}

impl ReActAgentEvolve {
    /// Creates an evolve agent around `card` with the default configuration and
    /// no model or tools.
    ///
    /// Both operators are built immediately, so an optimiser may already read
    /// (and, after mounting tools and configuring, tune) their state.
    #[must_use]
    pub fn new(card: AgentCard) -> Self {
        let agent = Arc::new(ReActAgent::new(card));
        let operators = build_operators(&agent);
        Self {
            agent,
            operators: RwLock::new(operators),
        }
    }

    /// Returns the inner agent that executes every invocation.
    #[must_use]
    pub fn react(&self) -> &Arc<ReActAgent> {
        &self.agent
    }

    /// Replaces the configuration, then rebuilds the operators.
    ///
    /// Mirrors Python's `configure`: the parent configuration (and with it the
    /// prompt builder, context engine, and request configuration) is replaced,
    /// and fresh operators are seeded from the resulting prompt template and
    /// mounted tool descriptions. Mounted tools survive reconfiguration.
    #[must_use]
    pub fn configure(&self, config: ReActAgentConfig) -> &Self {
        self.agent.configure(config);
        *self.operators.write() = build_operators(&self.agent);
        self
    }

    /// Sets the model used for subsequent invocations.
    pub fn set_model(&self, model: Arc<dyn Model>) {
        self.agent.set_model(model);
    }

    /// Mounts `tool` on the inner agent.
    ///
    /// The tool operator is not refreshed, matching Python: a later
    /// [`ReActAgentEvolve::configure`] (or an explicit
    /// [`ReActAgentEvolve::get_operators`] description update) re-reads the
    /// mounted descriptions.
    pub fn add_tool(&self, tool: Arc<dyn Tool>) -> AddAbilityResult {
        self.agent.add_tool(tool)
    }

    /// Mounts every tool in `tools` on the inner agent.
    pub fn add_tools(
        &self,
        tools: impl IntoIterator<Item = Arc<dyn Tool>>,
    ) -> Vec<AddAbilityResult> {
        self.agent.add_tools(tools)
    }

    /// Removes and returns the tool named `name` from the inner agent.
    pub fn remove_tool(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.agent.remove_tool(name)
    }

    /// Adds or replaces a rail on the inner agent.
    pub fn register_rail(&self, rail: Arc<dyn AgentRail>) {
        self.agent.register_rail(rail);
    }

    /// Replaces the inner agent's registry of client-executed tool handlers.
    pub fn set_client_tools(&self, tools: Arc<ClientToolRegistry>) {
        self.agent.set_client_tools(tools);
    }

    /// Replaces the inner agent's callback registry.
    pub fn set_callback_framework(&self, framework: Arc<CallbackFramework>) {
        self.agent.set_callback_framework(framework);
    }

    /// Returns the current operators in operator-id order
    /// (`react_llm`, `react_tool`).
    #[must_use]
    pub fn operators(&self) -> Vec<SharedOperator> {
        self.operators.read().shared.values().cloned().collect()
    }

    /// Returns the operators keyed by [`Operator::operator_id`], the primary
    /// interface of the self-evolution framework.
    #[must_use]
    pub fn get_operators(&self) -> BTreeMap<String, SharedOperator> {
        self.operators.read().shared.clone()
    }

    /// Returns the type-erased prompt operator (`"react_llm"`).
    ///
    /// The operator always exists; the `Option` keeps the Python-shaped
    /// accessor total when a build ever omits it.
    #[must_use]
    pub fn llm_operator(&self) -> Option<SharedOperator> {
        self.operators.read().shared.get(LLM_OPERATOR_ID).cloned()
    }

    /// Returns the type-erased tool-description operator (`"react_tool"`).
    #[must_use]
    pub fn tool_operator(&self) -> Option<SharedOperator> {
        self.operators.read().shared.get(TOOL_OPERATOR_ID).cloned()
    }

    /// Returns the typed prompt operator for state inspection.
    #[must_use]
    pub fn llm_call_operator(&self) -> SharedLlmOperator {
        Arc::clone(&self.operators.read().llm)
    }

    /// Returns the typed tool-description operator for state inspection.
    #[must_use]
    pub fn tool_call_operator(&self) -> SharedToolOperator {
        Arc::clone(&self.operators.read().tool)
    }
}

impl fmt::Debug for ReActAgentEvolve {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ReActAgentEvolve")
            .field("card_id", &self.agent.card().id)
            .field("operators", &[LLM_OPERATOR_ID, TOOL_OPERATOR_ID])
            .finish()
    }
}

#[async_trait]
impl Agent for ReActAgentEvolve {
    fn card(&self) -> &AgentCard {
        self.agent.card()
    }

    async fn invoke(
        &self,
        inputs: Value,
        session: Option<Arc<dyn Session>>,
    ) -> Result<AgentOutput> {
        self.agent.invoke(inputs, session).await
    }

    fn stream(
        &self,
        inputs: Value,
        session: Option<Arc<dyn Session>>,
    ) -> BoxStream<'static, Result<AgentChunk>> {
        self.agent.stream(inputs, session)
    }
}

/// Builds both operators from the inner agent's current configuration and
/// mounted tools.
fn build_operators(agent: &Arc<ReActAgent>) -> Operators {
    let config = agent.config();
    let llm: SharedLlmOperator = Arc::new(RwLock::new(
        LLMCallOperator::new(LLM_OPERATOR_ID)
            .with_system_prompt(config.prompt_template)
            .with_user_prompt(DEFAULT_USER_PROMPT)
            .with_freeze_system_prompt(false)
            .with_freeze_user_prompt(true)
            .with_callback(llm_parameter_callback(Arc::clone(agent))),
    ));
    let tool: SharedToolOperator = Arc::new(RwLock::new(
        ToolCallOperator::new(TOOL_OPERATOR_ID)
            .with_descriptions(tool_descriptions(agent))
            .with_callback(tool_parameter_callback(Arc::clone(agent))),
    ));
    let mut shared: BTreeMap<String, SharedOperator> = BTreeMap::new();
    shared.insert(
        LLM_OPERATOR_ID.to_owned(),
        Arc::new(RwLock::new(
            Box::new(OperatorHandle::new(LLM_OPERATOR_ID, Arc::clone(&llm))) as Box<dyn Operator>,
        )),
    );
    shared.insert(
        TOOL_OPERATOR_ID.to_owned(),
        Arc::new(RwLock::new(
            Box::new(OperatorHandle::new(TOOL_OPERATOR_ID, Arc::clone(&tool))) as Box<dyn Operator>,
        )),
    );
    Operators { llm, tool, shared }
}

/// Collects the mounted `tool name -> description` map (unnamed tools are
/// skipped, mirroring Python's `if tool_info.name`).
fn tool_descriptions(agent: &ReActAgent) -> BTreeMap<String, String> {
    agent
        .tools()
        .into_iter()
        .filter(|card| !card.name.is_empty())
        .map(|card| (card.name, card.description))
        .collect()
}

/// Builds the prompt callback that rewrites the inner agent's
/// `prompt_template`.
fn llm_parameter_callback(agent: Arc<ReActAgent>) -> ParameterCallback {
    Arc::new(move |target: &str, value: &Value| {
        if target != "system_prompt" {
            return;
        }
        let Some(template) = prompt_template_from(value) else {
            tracing::warn!(
                parameter = target,
                "ignoring malformed system prompt update for ReActAgentEvolve"
            );
            return;
        };
        let mut config = agent.config();
        if config.prompt_template == template {
            return;
        }
        config.prompt_template = template;
        agent.configure(config);
    })
}

/// Builds the tool callback that rewrites the mounted tools' card
/// descriptions.
fn tool_parameter_callback(agent: Arc<ReActAgent>) -> ParameterCallback {
    Arc::new(move |target: &str, value: &Value| {
        if target != TOOL_DESCRIPTION_TARGET {
            return;
        }
        match serde_json::from_value::<BTreeMap<String, String>>(value.clone()) {
            Ok(descriptions) => sync_tool_descriptions(&agent, &descriptions),
            Err(error) => tracing::warn!(
                parameter = target,
                %error,
                "ignoring malformed tool description update for ReActAgentEvolve"
            ),
        }
    })
}

/// Parses an operator `system_prompt` value into messages.
///
/// Accepts a plain string (one system message), a list of message objects
/// (the `Vec<BaseMessage>` shape the operator stores), or a list of strings
/// (one system message each). Anything else is malformed.
fn prompt_template_from(value: &Value) -> Option<Vec<BaseMessage>> {
    match value {
        Value::String(text) => Some(vec![BaseMessage::system(text.clone())]),
        Value::Array(items) => items
            .iter()
            .map(|item| match item {
                Value::String(text) => Some(BaseMessage::system(text.clone())),
                other => serde_json::from_value::<BaseMessage>(other.clone()).ok(),
            })
            .collect(),
        _ => None,
    }
}

/// Applies `descriptions` to the mounted tools, preserving their order and
/// every card field except the description.
///
/// `ReActAgent` exposes no card mutation, so this takes every mounted tool out
/// (capturing the card order first), re-wraps only the tools whose description
/// actually changed in a [`DescriptionOverrideTool`], and mounts them back in
/// the original order. Python deviates here by assigning
/// `ToolCard.description` in place.
fn sync_tool_descriptions(agent: &ReActAgent, descriptions: &BTreeMap<String, String>) {
    let cards = agent.tools();
    if !cards.iter().any(|card| {
        descriptions
            .get(&card.name)
            .is_some_and(|description| description != &card.description)
    }) {
        return;
    }
    let mut taken = Vec::with_capacity(cards.len());
    for card in cards {
        if let Some(tool) = agent.remove_tool(&card.name) {
            taken.push((card, tool));
        } else {
            tracing::warn!(
                tool = %card.name,
                "mounted tool was missing during description sync"
            );
        }
    }
    for (card, tool) in taken {
        let changed = descriptions
            .get(&card.name)
            .filter(|description| *description != &card.description);
        match changed {
            Some(description) => {
                agent.add_tool(Arc::new(DescriptionOverrideTool::new(
                    tool,
                    description.clone(),
                )));
            }
            None => {
                agent.add_tool(tool);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::assert_is_empty
    )]

    use super::*;
    use openjiuwen_foundation::{FunctionTool, Role, ToolCard, ToolContext};
    use openjiuwen_test_support::{MockCompletion, ScriptedModel};
    use parking_lot::Mutex;
    use serde_json::json;

    type Log = Arc<Mutex<Vec<Value>>>;

    fn shared(model: &ScriptedModel) -> Arc<dyn Model> {
        Arc::new(model.clone())
    }

    /// An echo tool that records its parsed inputs and keeps an inspectable
    /// card.
    fn echo_tool(log: &Log) -> (Arc<dyn Tool>, ToolCard) {
        let log = Arc::clone(log);
        let tool = FunctionTool::builder()
            .name("echo")
            .description("echoes its input")
            .input_schema(json!({"type": "object"}))
            .handler(move |inputs: Value, _ctx: ToolContext| {
                let log = Arc::clone(&log);
                async move {
                    log.lock().push(inputs.clone());
                    Ok(inputs)
                }
            })
            .build()
            .expect("the echo tool fixture builds");
        let card = tool.card().clone();
        (Arc::new(tool), card)
    }

    fn system_config(prompt: &str, max_iterations: usize) -> ReActAgentConfig {
        let mut config = ReActAgentConfig::new();
        config.prompt_template = vec![BaseMessage::system(prompt)];
        config.max_iterations = max_iterations;
        config
    }

    /// Creates an evolve agent with `config` and `model` (no tools).
    fn evolve(config: ReActAgentConfig, model: Arc<dyn Model>) -> ReActAgentEvolve {
        let agent = ReActAgentEvolve::new(AgentCard::new("evolve-test"));
        agent.set_model(model);
        let _ = agent.configure(config);
        agent
    }

    /// Creates an evolve agent with the echo tool mounted before the
    /// configuration (and therefore before the operators are seeded).
    fn evolve_with_echo(
        config: ReActAgentConfig,
        model: Arc<dyn Model>,
    ) -> (ReActAgentEvolve, Log, ToolCard) {
        let agent = ReActAgentEvolve::new(AgentCard::new("evolve-test"));
        let log: Log = Arc::new(Mutex::new(Vec::new()));
        let (tool, card) = echo_tool(&log);
        agent.add_tool(tool);
        agent.set_model(model);
        let _ = agent.configure(config);
        (agent, log, card)
    }

    #[test]
    fn new_builds_both_operators_keyed_by_id() {
        let agent = ReActAgentEvolve::new(AgentCard::new("evolve-test"));
        assert_eq!(
            agent.get_operators().keys().cloned().collect::<Vec<_>>(),
            [LLM_OPERATOR_ID.to_owned(), TOOL_OPERATOR_ID.to_owned()]
        );
        assert_eq!(agent.operators().len(), 2);
        assert_eq!(
            agent.llm_operator().unwrap().read().operator_id(),
            LLM_OPERATOR_ID
        );
        assert_eq!(
            agent.tool_operator().unwrap().read().operator_id(),
            TOOL_OPERATOR_ID
        );
        let llm = agent.llm_call_operator();
        assert_eq!(llm.read().user_prompt(), "{{query}}");
        assert!(llm.read().freeze_user_prompt());
        assert!(!llm.read().freeze_system_prompt());
    }

    #[test]
    fn new_seeds_operators_from_config_and_mounted_tools() {
        let model = Arc::new(ScriptedModel::text("unused")) as Arc<dyn Model>;
        let (agent, _log, _card) = evolve_with_echo(system_config("be terse", 7), model);

        let llm = agent.llm_operator().unwrap();
        assert_eq!(
            llm.read().get_state()["system_prompt"],
            json!([{"role": "system", "content": "be terse"}])
        );
        let tool = agent.tool_operator().unwrap();
        assert_eq!(
            tool.read().get_state()[TOOL_DESCRIPTION_TARGET],
            json!({"echo": "echoes its input"})
        );
        assert_eq!(agent.react().config().max_iterations, 7);
    }

    #[test]
    fn system_prompt_update_syncs_the_live_configuration() {
        let model = ScriptedModel::text("done");
        let agent = evolve(system_config("old prompt", 7), shared(&model));

        let result = agent
            .llm_operator()
            .unwrap()
            .write()
            .set_parameter(
                "system_prompt",
                json!([{"role": "system", "content": "new prompt"}]),
            )
            .unwrap();

        assert!(result.applied);
        let config = agent.react().config();
        assert_eq!(config.prompt_template.len(), 1);
        assert_eq!(config.prompt_template[0].text(), "new prompt");
        // The rest of the configuration is untouched.
        assert_eq!(config.max_iterations, 7);
    }

    #[tokio::test]
    async fn system_prompt_update_reaches_the_next_model_call() {
        let model = ScriptedModel::text("done");
        let agent = evolve(system_config("old prompt", 5), shared(&model));
        agent
            .llm_operator()
            .unwrap()
            .write()
            .set_parameter("system_prompt", json!("new prompt"))
            .unwrap();

        let output = agent.invoke(json!("hi"), None).await.unwrap();
        assert_eq!(output.result_type, crate::output::ResultType::Answer);
        assert_eq!(output.output, "done");
        let call = model.last_call().unwrap();
        let system_texts: Vec<String> = call
            .messages
            .iter()
            .filter(|message| message.role == Role::System)
            .map(|message| message.text().into_owned())
            .collect();
        assert_eq!(system_texts, ["new prompt"]);
    }

    #[test]
    fn frozen_user_prompt_update_is_skipped() {
        let model = ScriptedModel::text("done");
        let agent = evolve(system_config("keep me", 5), shared(&model));

        let result = agent
            .llm_operator()
            .unwrap()
            .write()
            .set_parameter("user_prompt", json!("always this"))
            .unwrap();

        assert!(!result.applied);
        assert_eq!(result.reason, "frozen");
        let config = agent.react().config();
        assert_eq!(config.prompt_template[0].text(), "keep me");
        assert_eq!(agent.llm_call_operator().read().user_prompt(), "{{query}}");
    }

    #[test]
    fn malformed_system_prompt_is_ignored() {
        let model = ScriptedModel::text("done");
        let agent = evolve(system_config("keep me", 5), shared(&model));

        agent
            .llm_operator()
            .unwrap()
            .write()
            .set_parameter("system_prompt", json!(42))
            .unwrap();

        assert_eq!(agent.react().config().prompt_template[0].text(), "keep me");
    }

    #[test]
    fn tool_description_update_preserves_every_other_card_field() {
        let model = ScriptedModel::text("done");
        let (agent, _log, original) = evolve_with_echo(ReActAgentConfig::new(), shared(&model));

        let result = agent
            .tool_operator()
            .unwrap()
            .write()
            .set_parameter("tool_description", json!({"echo": "new description"}))
            .unwrap();

        assert!(result.applied);
        let cards = agent.react().tools();
        assert_eq!(cards.len(), 1);
        assert_eq!(cards[0].description, "new description");
        assert_eq!(cards[0].id, original.id);
        assert_eq!(cards[0].name, original.name);
        assert_eq!(cards[0].input_params, original.input_params);
        assert_eq!(cards[0].parallel_safe, original.parallel_safe);
        assert_eq!(
            agent.react().tools()[0].description,
            agent
                .tool_call_operator()
                .read()
                .description("echo")
                .unwrap()
        );
    }

    #[tokio::test]
    async fn tuned_tool_description_reaches_the_model_and_still_executes() {
        let model = ScriptedModel::builder()
            .completion(MockCompletion::tool_call("echo", json!({"value": "hi"})))
            .respond_text("final answer")
            .build();
        let (agent, log, _card) = evolve_with_echo(ReActAgentConfig::new(), shared(&model));
        agent
            .tool_operator()
            .unwrap()
            .write()
            .set_parameter("tool_description", json!({"echo": "a tuned description"}))
            .unwrap();

        let output = agent.invoke(json!("hi"), None).await.unwrap();
        assert_eq!(output.result_type, crate::output::ResultType::Answer);
        assert_eq!(output.output, "final answer");
        // The tool still executes through the overriding wrapper.
        assert_eq!(log.lock().clone(), vec![json!({"value": "hi"})]);
        let tools = model.calls()[0].request.tools.clone().unwrap();
        assert!(tools.to_string().contains("a tuned description"), "{tools}");
    }

    #[test]
    fn llm_load_state_restores_and_syncs_a_previous_prompt() {
        let model = ScriptedModel::text("done");
        let agent = evolve(system_config("original prompt", 5), shared(&model));
        let saved = agent.llm_call_operator().read().get_state();

        agent
            .llm_operator()
            .unwrap()
            .write()
            .set_parameter("system_prompt", json!("changed prompt"))
            .unwrap();
        assert_eq!(
            agent.react().config().prompt_template[0].text(),
            "changed prompt"
        );

        agent
            .llm_operator()
            .unwrap()
            .write()
            .load_state(saved.clone())
            .unwrap();

        assert_eq!(
            agent.react().config().prompt_template[0].text(),
            "original prompt"
        );
        assert_eq!(agent.llm_call_operator().read().get_state(), saved);
    }

    #[test]
    fn tool_load_state_restores_and_syncs_previous_descriptions() {
        let model = ScriptedModel::text("done");
        let (agent, _log, _card) = evolve_with_echo(ReActAgentConfig::new(), shared(&model));
        let saved = agent.tool_call_operator().read().get_state();

        agent
            .tool_operator()
            .unwrap()
            .write()
            .set_parameter("tool_description", json!({"echo": "changed description"}))
            .unwrap();
        assert_eq!(agent.react().tools()[0].description, "changed description");

        agent
            .tool_operator()
            .unwrap()
            .write()
            .load_state(saved.clone())
            .unwrap();

        assert_eq!(agent.react().tools()[0].description, "echoes its input");
        assert_eq!(agent.tool_call_operator().read().get_state(), saved);
    }

    #[test]
    fn description_state_round_trips_through_serde() {
        let model = ScriptedModel::text("done");
        let agent = evolve(ReActAgentConfig::new(), shared(&model));
        let saved = agent.llm_call_operator().read().get_state();
        let encoded = serde_json::to_string(&saved).unwrap();
        let decoded: Value = serde_json::from_str(&encoded).unwrap();

        agent
            .llm_operator()
            .unwrap()
            .write()
            .load_state(decoded)
            .unwrap();

        assert_eq!(agent.llm_call_operator().read().get_state(), saved);
    }

    #[tokio::test]
    async fn invoke_delegates_to_the_inner_agent() {
        let model = ScriptedModel::text("inner answer");
        let agent = evolve(ReActAgentConfig::new(), shared(&model));

        let output = agent.invoke(json!("hi"), None).await.unwrap();

        assert_eq!(output.result_type, crate::output::ResultType::Answer);
        assert_eq!(output.output, "inner answer");
        assert_eq!(model.call_count(), 1);
        assert_eq!(agent.card().name, "evolve-test");
    }
}
