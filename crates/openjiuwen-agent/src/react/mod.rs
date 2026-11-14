//! The `ReAct` model/tool loop, its configuration, and the [`ReActAgent`] shell.
//!
//! [`ReActAgent`] owns the mutable runtime state of one agent — configuration,
//! model, mounted tools, rails, prompt builder, and context engine — behind
//! interior mutability, so a single value can be shared as `Arc<dyn Agent>`.
//! The iteration loop lives in `agent_loop`, one model call in [`model_step`],
//! pause persistence and rendering in `pause`, interrupt replay and turn
//! seeding in `resume`, and identity prompt rendering in `prompt`.
//!
//! The implemented loop semantics match Python's
//! `openjiuwen/core/single_agent/agents/react_agent.py::_inner_invoke`:
//!
//! 1. Iterate `start..=limit`, where `start` is `1` for a fresh turn and one
//!    past the interrupted iteration for a resumed one; a host limit
//!    (`_host_max_iterations`) and the configured hard limit may lower the
//!    maximum.
//! 2. Build the system prompt for the iteration and prepare the model window
//!    through [`ContextEngine::prepare`](openjiuwen_context::ContextEngine).
//! 3. Fire `BeforeModelCall`, call the model, append the assistant message, and
//!    fire `AfterModelCall`.
//! 4. Execute the model's tool calls in model order, batched by `parallel_safe`,
//!    firing `BeforeToolCall`/`AfterToolCall` around each call, and append the
//!    tool messages.
//! 5. Fire `AfterReactIteration`; with no tool calls the turn terminates with an
//!    answer.
//! 6. Exhausting the budget applies [`MaxIterationsPolicy`] and records
//!    `termination_reason = "max_iterations"`; exceeding
//!    [`ReActAgentConfig::max_total_tokens`] records `"token_budget"`.
//! 7. Rails may `force_finish`, request a model `retry`, `model_continue`, or
//!    `skip_tool`; an interrupt pauses the loop into an output carrying state
//!    frames and interrupt ids that a later invocation resumes.
//!
//! The loop also drives the agent's side seams, all of which are no-ops in
//! their defaults: `hooks` drains steering messages at the top of every
//! iteration, fires the six lifecycle callbacks (logging their errors instead
//! of failing the run), and runs the optional final-request check; a client
//! tool call pauses the turn as a `client_tool` interrupt instead of executing
//! locally; and `frames` publishes a `context.usage` frame after each
//! preparation plus an `llm_route` frame when a rail swapped the model.
//! `exposure` owns the model-facing projection computed once per run: tools
//! the exposure policy or MCP allowlist denies never reach the model, while
//! execution still sees the full tool map.
//!
//! [`Agent::stream`] runs the same loop in a spawned task and yields
//! [`AgentChunk`] frames: model deltas, one usage frame per call, context and
//! route frames, the `__interaction__` frames of a pause, and one final `answer`
//! frame.
pub(crate) mod agent_loop;
pub mod config;
pub(crate) mod exposure;
pub(crate) mod frames;
pub(crate) mod hooks;
pub mod model_step;
pub(crate) mod pause;
pub mod prompt;
pub(crate) mod resume;
#[cfg(test)]
mod wiring;

pub use config::{MaxIterationsPolicy, ReActAgentConfig};

use crate::abilities::{AbilityManager, AddAbilityResult};
use crate::agent::Agent;
use crate::callbacks::{AgentCallback, AgentCallbackEvent, CallbackFramework};
use crate::card::AgentCard;
use crate::memory_provider::{provider_tools, ProviderMemoryRail, PROVIDER_RAIL_NAME};
use crate::output::{AgentChunk, AgentOutput};
use crate::policy::{
    ClientToolRegistry, FinalRequestCheck, McpToolAllowlist, ToolExposurePolicy,
    ToolResiliencePolicy,
};
use crate::prompt::{PromptSection, SystemPromptBuilder, IDENTITY_PRIORITY};
use crate::rails::{AgentRail, RailRegistry};
use crate::steering::SteeringQueue;
use agent_loop::AgentRuntime;
use futures::stream::{BoxStream, StreamExt};
use openjiuwen_common::error::{Error, ErrorKind, Result};
use openjiuwen_context::{ContextConfig, ContextEngine};
use openjiuwen_foundation::{Model, ModelRequestConfig, Tool, ToolCard};
use openjiuwen_memory::{ConversationTurn, MemoryProvider};
use openjiuwen_session::Session;
use serde_json::Value;
use std::fmt;
use std::sync::Arc;

/// The `ReAct` agent: configuration plus the runtime state of one agent.
///
/// Construct it with [`ReActAgent::new`], feed it a model through
/// [`ReActAgent::set_model`], and mount tools through [`ReActAgent::add_tool`].
/// All mutable state uses `parking_lot::RwLock` so the value can be shared as
/// `Arc<dyn Agent>` without a global registry.
///
/// # Loop contract
///
/// The loop follows Python's `_inner_invoke`:
///
/// - iterations run from `1` (or one past the interrupted iteration of a
///   resumed turn) to the effective limit ([`ReActAgentConfig::max_iterations`],
///   optionally lowered by `_host_max_iterations` and
///   [`ReActAgentConfig::max_iterations_hard_limit`]);
/// - the system prompt is rebuilt each iteration and the model window comes
///   from [`ContextEngine::prepare`](openjiuwen_context::ContextEngine);
/// - rails fire around the model step (`BeforeModelCall`/`AfterModelCall`),
///   every tool call (`BeforeToolCall`/`AfterToolCall`), and each iteration
///   (`AfterReactIteration`), and may `force_finish`, request a model `retry`,
///   `model_continue`, or `skip_tool`;
/// - tool calls execute in model order; consecutive parallel-safe tools run as
///   one batch and a non-parallel-safe tool is a barrier;
/// - exhaustion applies [`ReActAgentConfig::on_max_iterations`] and records
///   `termination_reason = "max_iterations"`, while exceeding
///   [`ReActAgentConfig::max_total_tokens`] records `"token_budget"`;
/// - an interrupt is an output (`result_type = "interrupt"`) whose state is
///   persisted under `__react_agent_interruption__` and resumed by a later
///   invocation whose query is an interactive payload;
/// - a model call always streams (a run is always session-scoped), so the
///   session receives `llm_output`/`llm_reasoning`/`llm_usage` frames as the
///   provider produces them.
pub struct ReActAgent {
    card: AgentCard,
    config: parking_lot::RwLock<ReActAgentConfig>,
    model: parking_lot::RwLock<Option<Arc<dyn Model>>>,
    request_config: parking_lot::RwLock<Option<ModelRequestConfig>>,
    abilities: parking_lot::RwLock<AbilityManager>,
    rails: parking_lot::RwLock<Arc<RailRegistry>>,
    prompt_builder: parking_lot::RwLock<SystemPromptBuilder>,
    engine: parking_lot::RwLock<Arc<ContextEngine>>,
    callbacks: parking_lot::RwLock<Arc<CallbackFramework>>,
    steering: SteeringQueue,
    exposure: parking_lot::RwLock<ToolExposurePolicy>,
    mcp_allowlist: parking_lot::RwLock<McpToolAllowlist>,
    client_tools: parking_lot::RwLock<Arc<ClientToolRegistry>>,
    resilience: parking_lot::RwLock<ToolResiliencePolicy>,
    final_check: parking_lot::RwLock<Option<Arc<dyn FinalRequestCheck>>>,
    memory_provider: parking_lot::RwLock<Option<Arc<dyn MemoryProvider>>>,
}

impl ReActAgent {
    /// Creates an agent around `card` with the default configuration, no model,
    /// nobody's tools, no rails, and a default context engine.
    ///
    /// The policy, callback, and steering seams start in their
    /// behavior-preserving defaults: an empty callback registry, an empty
    /// steering queue, allow-all tool exposure with no MCP restriction, no
    /// client tools, an empty resilience policy (single attempt, no per-tool
    /// timeout), and no final-request check.
    #[must_use]
    pub fn new(card: AgentCard) -> Self {
        let abilities = AbilityManager::new(&card.id);
        Self {
            card,
            config: parking_lot::RwLock::new(ReActAgentConfig::default()),
            model: parking_lot::RwLock::new(None),
            request_config: parking_lot::RwLock::new(None),
            abilities: parking_lot::RwLock::new(abilities),
            rails: parking_lot::RwLock::new(Arc::new(RailRegistry::new())),
            prompt_builder: parking_lot::RwLock::new(SystemPromptBuilder::new()),
            engine: parking_lot::RwLock::new(Arc::new(
                ContextEngine::new(ContextConfig::default()),
            )),
            callbacks: parking_lot::RwLock::new(Arc::new(CallbackFramework::new())),
            steering: SteeringQueue::new(),
            exposure: parking_lot::RwLock::new(ToolExposurePolicy::allow_all()),
            mcp_allowlist: parking_lot::RwLock::new(McpToolAllowlist::new()),
            client_tools: parking_lot::RwLock::new(Arc::new(ClientToolRegistry::new())),
            resilience: parking_lot::RwLock::new(ToolResiliencePolicy::new()),
            final_check: parking_lot::RwLock::new(None),
            memory_provider: parking_lot::RwLock::new(None),
        }
    }

    /// Replaces the configuration and rebuilds the derived state.
    ///
    /// The request configuration follows
    /// [`ReActAgentConfig::model_config`], the context engine is rebuilt from
    /// [`ReActAgentConfig::context_engine_config`], and the prompt builder is
    /// rebuilt from the system-role entries of
    /// [`ReActAgentConfig::prompt_template`]: their rendered contents are
    /// joined with a blank line into one identity section.
    pub fn configure(&self, config: ReActAgentConfig) {
        let identity = config
            .prompt_template
            .iter()
            .filter(|message| message.role.is_system())
            .map(|message| message.text().into_owned())
            .collect::<Vec<_>>()
            .join("\n\n");
        let mut builder = SystemPromptBuilder::new();
        if !identity.is_empty() {
            builder.add_section(PromptSection::new("identity", identity, IDENTITY_PRIORITY));
        }
        self.request_config.write().clone_from(&config.model_config);
        *self.engine.write() = Arc::new(ContextEngine::new(config.context_engine_config.clone()));
        *self.prompt_builder.write() = builder;
        *self.config.write() = config;
    }

    /// Returns a copy of the current configuration.
    #[must_use]
    pub fn config(&self) -> ReActAgentConfig {
        self.config.read().clone()
    }

    /// Sets the model used for subsequent invocations.
    pub fn set_model(&self, model: Arc<dyn Model>) {
        *self.model.write() = Some(model);
    }

    /// Returns the configured model.
    ///
    /// # Errors
    ///
    /// [`Error::Config`] when no model is configured.
    pub fn model(&self) -> Result<Arc<dyn Model>> {
        self.model.read().clone().ok_or_else(|| {
            Error::config(
                "model is not configured; call ReActAgent::set_model \
                 (model_client_config is required to build one)",
            )
        })
    }

    /// Mounts `tool`, replacing a tool of the same name.
    pub fn add_tool(&self, tool: Arc<dyn Tool>) -> AddAbilityResult {
        self.abilities.write().add_tool(tool)
    }

    /// Mounts every tool in `tools`, returning one result per tool.
    pub fn add_tools(
        &self,
        tools: impl IntoIterator<Item = Arc<dyn Tool>>,
    ) -> Vec<AddAbilityResult> {
        let mut abilities = self.abilities.write();
        tools
            .into_iter()
            .map(|tool| abilities.add_tool(tool))
            .collect()
    }

    /// Removes and returns the tool named `name`.
    pub fn remove_tool(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.abilities.write().remove_tool(name)
    }

    /// Returns the model-visible tool cards in mount order.
    #[must_use]
    pub fn tools(&self) -> Vec<ToolCard> {
        self.abilities.read().cards()
    }

    /// Adds or replaces a rail.
    ///
    /// Do not call this while a running loop holds the [`ReActAgent::rails`]
    /// `Arc`: the registry is copy-on-write, so the running loop keeps the
    /// registry snapshot it started with.
    pub fn register_rail(&self, rail: Arc<dyn AgentRail>) {
        let mut guard = self.rails.write();
        let mut registry = Arc::unwrap_or_clone(Arc::clone(&guard));
        registry.register(rail);
        *guard = Arc::new(registry);
    }

    /// Removes the rail named `name`.
    ///
    /// Like [`ReActAgent::register_rail`], this replaces the registry snapshot
    /// rather than mutating one a running loop may hold.
    pub fn unregister_rail(&self, name: &str) {
        let mut guard = self.rails.write();
        let mut registry = Arc::unwrap_or_clone(Arc::clone(&guard));
        registry.unregister(name);
        *guard = Arc::new(registry);
    }

    /// Returns the current rail registry snapshot.
    #[must_use]
    pub fn rails(&self) -> Arc<RailRegistry> {
        Arc::clone(&*self.rails.read())
    }

    /// Returns a copy of the current system prompt builder.
    #[must_use]
    pub fn prompt_builder(&self) -> SystemPromptBuilder {
        self.prompt_builder.read().clone()
    }

    /// Returns the context engine used to prepare the model window.
    #[must_use]
    pub fn context_engine(&self) -> Arc<ContextEngine> {
        Arc::clone(&*self.engine.read())
    }

    /// Replaces the whole callback registry.
    ///
    /// A running loop keeps the framework snapshot it started with, so a
    /// replacement affects later invocations.
    pub fn set_callback_framework(&self, framework: Arc<CallbackFramework>) {
        *self.callbacks.write() = framework;
    }

    /// Returns the current callback registry snapshot; empty by default.
    #[must_use]
    pub fn callbacks(&self) -> Arc<CallbackFramework> {
        Arc::clone(&*self.callbacks.read())
    }

    /// Registers `callback` for `event`, adding to the existing registry.
    ///
    /// The registry is copy-on-write like [`ReActAgent::register_rail`], so a
    /// running loop keeps the framework snapshot it started with and the
    /// callback fires on later invocations. [`CallbackFramework`] is not `Clone`
    /// (its handlers are trait objects), so the replacement snapshot is rebuilt
    /// by re-registering the existing handlers in their current order.
    pub fn register_callback(&self, event: AgentCallbackEvent, callback: Arc<dyn AgentCallback>) {
        let mut guard = self.callbacks.write();
        let mut framework = CallbackFramework::new();
        for existing in AgentCallbackEvent::ALL {
            for registered in guard.handlers(existing) {
                framework.register(existing, registered);
            }
        }
        framework.register(event, callback);
        *guard = Arc::new(framework);
    }

    /// Returns the steering queue; the clone shares this agent's backlog.
    #[must_use]
    pub fn steering(&self) -> SteeringQueue {
        self.steering.clone()
    }

    /// Pushes one steering message and returns the resulting queue length.
    ///
    /// A blank text is dropped and leaves the length unchanged. This delegates
    /// to [`SteeringQueue::push_text`] on [`ReActAgent::steering`].
    pub fn steer(&self, text: impl Into<String>) -> usize {
        self.steering.push_text(text)
    }

    /// Replaces the tool exposure policy; allow-all by default.
    pub fn set_tool_exposure_policy(&self, policy: ToolExposurePolicy) {
        *self.exposure.write() = policy;
    }

    /// Returns a copy of the current tool exposure policy.
    #[must_use]
    pub fn tool_exposure_policy(&self) -> ToolExposurePolicy {
        self.exposure.read().clone()
    }

    /// Replaces the MCP tool allowlist; an empty allowlist restricts nothing.
    pub fn set_mcp_tool_allowlist(&self, allowlist: McpToolAllowlist) {
        *self.mcp_allowlist.write() = allowlist;
    }

    /// Returns a copy of the current MCP tool allowlist.
    #[must_use]
    pub fn mcp_tool_allowlist(&self) -> McpToolAllowlist {
        self.mcp_allowlist.read().clone()
    }

    /// Replaces the registry of client-executed tool handlers.
    pub fn set_client_tools(&self, tools: Arc<ClientToolRegistry>) {
        *self.client_tools.write() = tools;
    }

    /// Returns the current client tool registry; empty by default.
    #[must_use]
    pub fn client_tools(&self) -> Arc<ClientToolRegistry> {
        Arc::clone(&*self.client_tools.read())
    }

    /// Replaces the per-tool resilience policy; empty by default, which keeps
    /// the single-attempt, no-per-tool-timeout behavior.
    pub fn set_tool_resilience(&self, policy: ToolResiliencePolicy) {
        *self.resilience.write() = policy;
    }

    /// Returns a copy of the current resilience policy.
    #[must_use]
    pub fn tool_resilience(&self) -> ToolResiliencePolicy {
        self.resilience.read().clone()
    }

    /// Installs the check run before the final model request of a round.
    pub fn set_final_request_check(&self, check: Arc<dyn FinalRequestCheck>) {
        *self.final_check.write() = Some(check);
    }

    /// Returns the installed final-request check; `None` (no check) by default.
    #[must_use]
    pub fn final_request_check(&self) -> Option<Arc<dyn FinalRequestCheck>> {
        self.final_check.read().clone()
    }

    /// Installs an external long-term-memory provider.
    ///
    /// Three things happen, in this order:
    ///
    /// 1. the provider is stored (replacing a previously installed one);
    /// 2. a [`ProviderMemoryRail`] over it is registered under the fixed rail name
    ///    `"memory_provider"`, replacing any rail already registered under it, so
    ///    prompt block, prefetch, and turn write-back take effect on the next
    ///    invocation;
    /// 3. the provider's advertised tool schemas are mounted with
    ///    [`ReActAgent::add_tools`] ([`provider_tools`]).
    ///
    /// Mounting does **not** depend on [`MemoryProvider::is_available`]: an
    /// unavailable provider is still stored, its rail and tools are still
    /// mounted, and it reports its own behaviour through them (a fail-soft
    /// `prefetch`, `{"error": …}` tool payloads). The only difference is one
    /// `tracing::warn!` naming the provider, so the misconfiguration is visible
    /// without silently disabling the seam.
    ///
    /// Like [`ReActAgent::register_rail`], a running loop keeps the registry
    /// snapshot it started with.
    pub fn set_memory_provider(&self, provider: Arc<dyn MemoryProvider>) {
        if !provider.is_available() {
            tracing::warn!(
                target: "openjiuwen_agent::memory_provider",
                provider = %provider.name(),
                "memory provider reports itself unavailable; mounting its rail and tools anyway"
            );
        }
        *self.memory_provider.write() = Some(Arc::clone(&provider));
        self.register_rail(Arc::new(ProviderMemoryRail::new(Arc::clone(&provider))));
        self.add_tools(provider_tools(provider));
    }

    /// Returns the installed external memory provider, if any.
    #[must_use]
    pub fn memory_provider(&self) -> Option<Arc<dyn MemoryProvider>> {
        self.memory_provider.read().clone()
    }

    /// Uninstalls the external memory provider.
    ///
    /// The `"memory_provider"` rail is dropped and later runs neither prefetch,
    /// nor receive the provider's prompt block, nor write turns back. Tools the
    /// provider mounted stay mounted: they are ordinary agent tools by then, and
    /// removed tools the caller may have renamed or reused would be surprising —
    /// drop them with [`ReActAgent::remove_tool`] for each name in
    /// `crate::memory_provider::provider_tool_cards` when that is what the
    /// caller wants.
    pub fn clear_memory_provider(&self) {
        *self.memory_provider.write() = None;
        self.unregister_rail(PROVIDER_RAIL_NAME);
    }

    /// Finalizes the conversation with the external memory provider.
    ///
    /// Runs [`MemoryProvider::on_session_end`] on the installed provider and
    /// propagates its error; with no provider installed it is a successful
    /// no-op. The loop calls this itself for a run whose session it created, so
    /// a caller only needs it after an invocation that reused a supplied
    /// session.
    ///
    /// # Errors
    ///
    /// The provider's own error when the finalization write fails.
    pub async fn end_memory_session(&self, messages: &[ConversationTurn]) -> Result<()> {
        let provider = self.memory_provider.read().clone();
        match provider {
            Some(provider) => provider.on_session_end(messages).await,
            None => Ok(()),
        }
    }
}

// The model, rails, and context engine are trait objects or lock guards, so the
// debug view summarizes them instead of printing their internals.
#[allow(clippy::missing_fields_in_debug)]
impl fmt::Debug for ReActAgent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ReActAgent")
            .field("card_id", &self.card.id)
            .field("max_iterations", &self.config.read().max_iterations)
            .field("model_configured", &self.model.read().is_some())
            .field("tools", &self.abilities.read().len())
            .finish()
    }
}

#[async_trait::async_trait]
impl Agent for ReActAgent {
    fn card(&self) -> &AgentCard {
        &self.card
    }

    async fn invoke(
        &self,
        inputs: Value,
        session: Option<Arc<dyn Session>>,
    ) -> Result<AgentOutput> {
        let input = crate::input::AgentInput::parse(&inputs)?;
        let run = crate::run::resolve_session(
            session,
            &self.card.id,
            Some(&self.card.name),
            input.conversation_id.as_deref(),
            input.parent_session_id.as_deref(),
        )
        .await?;
        let owned = run.is_owned();
        let result = agent_loop::run(self, input, Arc::clone(run.session()), None, owned).await;
        run.finish().await;
        result
    }

    /// Runs the loop in a spawned task and yields its frames.
    ///
    /// The agent is snapshotted synchronously (so the returned stream owns
    /// everything it needs), the session is resolved inside the task, and every
    /// frame is forwarded through an unbounded channel: model deltas, one
    /// `llm_usage` frame per call, `__interaction__` frames when the run pauses,
    /// and one final `answer` frame carrying the [`AgentOutput`].
    ///
    /// Errors surface two ways, matching Python's streaming contract: invalid
    /// input, an unconfigured model, and cancellation yield one `Err` item,
    /// while any other loop failure yields a final `answer` frame whose payload
    /// has `result_type = "error"`.
    fn stream(
        &self,
        inputs: Value,
        session: Option<Arc<dyn Session>>,
    ) -> BoxStream<'static, Result<AgentChunk>> {
        let input = match crate::input::AgentInput::parse(&inputs) {
            Ok(input) => input,
            Err(error) => return futures::stream::once(async move { Err(error) }).boxed(),
        };
        let runtime = match AgentRuntime::snapshot(self) {
            Ok(runtime) => runtime,
            Err(error) => return futures::stream::once(async move { Err(error) }).boxed(),
        };
        let agent_id = self.card.id.clone();
        let agent_name = self.card.name.clone();
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Result<AgentChunk>>();
        tokio::spawn(async move {
            // The loop publishes plain chunks; a forwarding task wraps each one
            // in `Ok` so the returned stream can also carry terminal `Err` items.
            let (frame_tx, mut frame_rx) = tokio::sync::mpsc::unbounded_channel::<AgentChunk>();
            let frames = tx.clone();
            let forwarder = tokio::spawn(async move {
                while let Some(chunk) = frame_rx.recv().await {
                    if frames.send(Ok(chunk)).is_err() {
                        break;
                    }
                }
            });
            let run = match crate::run::resolve_session(
                session,
                &agent_id,
                Some(&agent_name),
                input.conversation_id.as_deref(),
                input.parent_session_id.as_deref(),
            )
            .await
            {
                Ok(run) => run,
                Err(error) => {
                    drop(frame_tx);
                    let _ = forwarder.await;
                    let _ = tx.send(Err(error));
                    return;
                }
            };
            let owned = run.is_owned();
            let result = agent_loop::run_with(
                &runtime,
                input,
                Arc::clone(run.session()),
                Some(frame_tx),
                owned,
            )
            .await;
            // Every frame the loop produced reaches the caller before the
            // terminal item: the forwarder ends once the loop dropped its sender.
            let _ = forwarder.await;
            if let Err(error) = result {
                if error.kind() == ErrorKind::Cancelled {
                    let _ = tx.send(Err(error));
                } else {
                    let failed = AgentOutput::error(error.to_string());
                    let _ = tx.send(Ok(AgentChunk::answer(&failed)));
                }
            }
            run.finish().await;
            drop(tx);
        });
        futures::stream::unfold(rx, |mut rx| async move {
            rx.recv().await.map(|item| (item, rx))
        })
        .boxed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;
    use openjiuwen_common::error::ErrorKind;
    use openjiuwen_foundation::BaseMessage;
    use openjiuwen_test_support::ScriptedModel;

    fn agent() -> ReActAgent {
        ReActAgent::new(AgentCard::new("react"))
    }

    /// A tool that echoes its input, used to exercise mounting.
    struct Echo(openjiuwen_foundation::ToolCard);

    #[async_trait::async_trait]
    impl Tool for Echo {
        fn card(&self) -> &ToolCard {
            &self.0
        }

        async fn invoke(
            &self,
            input: Value,
            _ctx: &openjiuwen_foundation::ToolContext,
        ) -> Result<Value> {
            Ok(input)
        }
    }

    #[test]
    fn new_uses_the_default_configuration() {
        let agent = agent();
        assert_eq!(agent.config().max_iterations, 5);
        assert_eq!(agent.tools().len(), 0);
        assert_eq!(agent.rails().len(), 0);
        assert_eq!(agent.prompt_builder().build(), "");
    }

    #[test]
    fn configure_builds_identity_from_system_messages_only() {
        let agent = agent();
        let config = ReActAgentConfig {
            prompt_template: vec![
                BaseMessage::system("you are terse"),
                BaseMessage::user("ignored"),
            ],
            ..ReActAgentConfig::new()
        };
        agent.configure(config);
        assert_eq!(agent.prompt_builder().build(), "you are terse");
    }

    #[test]
    fn configure_joins_multiple_system_messages_with_a_blank_line() {
        let agent = agent();
        let config = ReActAgentConfig {
            prompt_template: vec![BaseMessage::system("first"), BaseMessage::system("second")],
            ..ReActAgentConfig::new()
        };
        agent.configure(config);
        assert_eq!(agent.prompt_builder().build(), "first\n\nsecond");
    }

    #[test]
    fn configure_rebuilds_derived_state() {
        let agent = agent();
        let mut config = ReActAgentConfig::new();
        config.configure_max_iterations(9);
        config.model_config = Some(ModelRequestConfig::new("gpt-4o"));
        agent.configure(config);
        assert_eq!(agent.config().max_iterations, 9);
    }

    #[test]
    fn model_is_unconfigured_by_default() {
        let agent = agent();
        // `unwrap_err` would require the Ok type (`Arc<dyn Model>`) to be Debug,
        // so project to the error first.
        let error = agent.model().err().unwrap();
        assert_eq!(error.kind(), ErrorKind::Config);
    }

    #[test]
    fn set_model_makes_the_model_available() {
        let agent = agent();
        let model: Arc<dyn Model> = Arc::new(ScriptedModel::text("hi"));
        agent.set_model(Arc::clone(&model));
        assert!(agent.model().is_ok());
    }

    #[test]
    fn add_tool_mounts_it_and_lists_the_card() {
        let tool_card = openjiuwen_foundation::ToolCard::new("echo", "echoes", Value::Null);
        let agent = agent();
        let result = agent.add_tool(Arc::new(Echo(tool_card)));
        assert!(result.added);
        assert_eq!(result.reason, "added_tool");
        let names: Vec<String> = agent.tools().into_iter().map(|card| card.name).collect();
        assert_eq!(names, vec!["echo".to_owned()]);
    }

    #[test]
    fn register_then_unregister_rail() {
        struct Noop;
        #[async_trait::async_trait]
        impl AgentRail for Noop {
            fn name(&self) -> &'static str {
                "noop"
            }
        }
        let agent = agent();
        agent.register_rail(Arc::new(Noop));
        assert_eq!(agent.rails().len(), 1);
        agent.unregister_rail("noop");
        assert_eq!(agent.rails().len(), 0);
    }

    #[tokio::test]
    async fn invoke_rejects_invalid_input() {
        let agent = agent();
        let error = agent.invoke(Value::Null, None).await.unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
    }

    #[tokio::test]
    async fn invoke_without_a_model_reports_a_configuration_error() {
        let agent = agent();
        let error = agent
            .invoke(serde_json::json!("hi"), None)
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Config);
    }

    #[tokio::test]
    async fn stream_without_a_model_yields_exactly_one_error() {
        let agent = agent();
        let frames: Vec<Result<AgentChunk>> =
            agent.stream(serde_json::json!("hi"), None).collect().await;
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].as_ref().unwrap_err().kind(), ErrorKind::Config);
    }

    #[tokio::test]
    async fn invoke_answers_with_the_configured_model() {
        let agent = agent();
        let model: Arc<dyn Model> = Arc::new(ScriptedModel::text("hi back"));
        agent.set_model(model);
        let output = agent
            .invoke(Value::String("hello".to_owned()), None)
            .await
            .unwrap();
        assert_eq!(output.result_type, crate::output::ResultType::Answer);
        assert_eq!(output.output, "hi back");
    }

    #[tokio::test]
    async fn stream_yields_model_deltas_and_one_answer_frame() {
        let agent = agent();
        let model: Arc<dyn Model> =
            Arc::new(ScriptedModel::builder().stream_text("hi back").build());
        agent.set_model(model);
        let frames: Vec<AgentChunk> = agent
            .stream(Value::String("hello".to_owned()), None)
            .map(|frame| frame.unwrap())
            .collect()
            .await;
        let answers = frames
            .iter()
            .filter(|frame| frame.kind == crate::output::ANSWER)
            .collect::<Vec<_>>();
        assert_eq!(answers.len(), 1);
        assert_eq!(answers[0].payload["output"], "hi back");
        assert_eq!(answers[0].payload["result_type"], "answer");
        assert!(frames
            .iter()
            .any(|frame| frame.kind == crate::output::LLM_OUTPUT));
    }

    #[test]
    fn policy_callback_and_steering_seams_start_in_their_defaults() {
        let agent = agent();
        assert!(!agent.callbacks().has_hooks(AgentCallbackEvent::InvokeStart));
        assert!(agent.steering().is_empty());
        assert_eq!(
            agent.tool_exposure_policy(),
            ToolExposurePolicy::allow_all()
        );
        assert!(agent.mcp_tool_allowlist().is_empty());
        assert_eq!(agent.client_tools().names(), Vec::<String>::new());
        assert!(agent.tool_resilience().is_empty());
        assert!(agent.final_request_check().is_none());
    }

    #[test]
    fn register_callback_populates_the_registry() {
        struct Named;

        #[async_trait::async_trait]
        impl AgentCallback for Named {
            fn name(&self) -> &'static str {
                "named"
            }
        }

        let agent = agent();
        let before = agent.callbacks();
        assert!(!before.has_hooks(AgentCallbackEvent::StepStart));

        agent.register_callback(AgentCallbackEvent::StepStart, Arc::new(Named));
        assert!(agent.callbacks().has_hooks(AgentCallbackEvent::StepStart));
        // Copy-on-write: a snapshot taken before registration is unaffected and
        // only the event that was subscribed to gains a handler.
        assert!(!before.has_hooks(AgentCallbackEvent::StepStart));
        assert!(!agent.callbacks().has_hooks(AgentCallbackEvent::InvokeStart));

        agent.set_callback_framework(Arc::new(CallbackFramework::new()));
        assert!(!agent.callbacks().has_hooks(AgentCallbackEvent::StepStart));
    }

    #[test]
    fn policy_setters_round_trip() {
        struct EchoAnswer;

        #[async_trait::async_trait]
        impl crate::policy::ClientToolHandler for EchoAnswer {
            async fn invoke(
                &self,
                _call: &openjiuwen_foundation::ToolCall,
                answer: &Value,
            ) -> Result<Value> {
                Ok(answer.clone())
            }
        }

        struct AllowEveryRequest;

        #[async_trait::async_trait]
        impl FinalRequestCheck for AllowEveryRequest {
            async fn check(
                &self,
                _ctx: &crate::policy::FinalRequestContext<'_>,
            ) -> Result<crate::policy::FinalRequestDecision> {
                Ok(crate::policy::FinalRequestDecision::Allow)
            }
        }

        let agent = agent();

        // Exposure policy.
        let policy = ToolExposurePolicy::allowlist(["echo"]);
        agent.set_tool_exposure_policy(policy.clone());
        assert_eq!(agent.tool_exposure_policy(), policy);
        let cards = vec![
            ToolCard::new("echo", "echoes", Value::Null),
            ToolCard::new("other", "other", Value::Null),
        ];
        let visible = agent.tool_exposure_policy().filter_cards(&cards);
        assert_eq!(visible.len(), 1);
        assert_eq!(visible[0].name, "echo");

        // MCP allowlist.
        let mut allowlist = McpToolAllowlist::new();
        allowlist.allow("files", "read_file");
        agent.set_mcp_tool_allowlist(allowlist.clone());
        assert_eq!(agent.mcp_tool_allowlist(), allowlist);
        assert!(agent.mcp_tool_allowlist().allows("files", "read_file"));
        assert!(!agent.mcp_tool_allowlist().allows("files", "write_file"));

        // Steering: the clone shares the agent's backlog.
        let shared = agent.steering();
        assert_eq!(agent.steer("stop after this step"), 1);
        assert_eq!(agent.steering().len(), 1);
        assert_eq!(shared.len(), 1);

        // Client tools.
        let registry = Arc::new(ClientToolRegistry::new());
        registry.register("pick", Arc::new(EchoAnswer));
        agent.set_client_tools(Arc::clone(&registry));
        assert!(agent.client_tools().is_client_tool("pick"));
        assert_eq!(agent.client_tools().names(), vec!["pick".to_owned()]);

        // Resilience.
        let mut resilience = ToolResiliencePolicy::new();
        resilience.set_default(crate::policy::ToolResilience::new().with_attempts(3));
        agent.set_tool_resilience(resilience.clone());
        assert_eq!(agent.tool_resilience(), resilience);
        assert_eq!(agent.tool_resilience().for_tool("echo").attempts(), 3);

        // Final request check.
        agent.set_final_request_check(Arc::new(AllowEveryRequest));
        assert!(agent.final_request_check().is_some());
    }
}
