//! The `ReAct` model/tool loop, its configuration, and the [`ReActAgent`] shell.
//!
//! [`ReActAgent`] owns the mutable runtime state of one agent — configuration,
//! model, mounted tools, rails, prompt builder, and context engine — behind
//! interior mutability, so a single value can be shared as `Arc<dyn Agent>`.
//! The iteration loop lives in `agent_loop`, one model call in [`model_step`],
//! interrupt replay in `resume`, and identity prompt rendering in `prompt`.
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
//! [`Agent::stream`] runs the same loop in a spawned task and yields
//! [`AgentChunk`] frames: model deltas, one usage frame per call, the
//! `__interaction__` frames of a pause, and one final `answer` frame.
pub(crate) mod agent_loop;
pub mod config;
pub mod model_step;
pub(crate) mod prompt;
pub(crate) mod resume;

pub use config::{MaxIterationsPolicy, ReActAgentConfig};

use crate::abilities::{AbilityManager, AddAbilityResult};
use crate::agent::Agent;
use crate::card::AgentCard;
use crate::output::{AgentChunk, AgentOutput};
use crate::prompt::{PromptSection, SystemPromptBuilder, IDENTITY_PRIORITY};
use crate::rails::{AgentRail, RailRegistry};
use agent_loop::AgentRuntime;
use futures::stream::{BoxStream, StreamExt};
use openjiuwen_common::error::{Error, ErrorKind, Result};
use openjiuwen_context::{ContextConfig, ContextEngine};
use openjiuwen_foundation::{Model, ModelRequestConfig, Tool, ToolCard};
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
}

impl ReActAgent {
    /// Creates an agent around `card` with the default configuration, no model,
    /// nobody's tools, no rails, and a default context engine.
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
        let result = agent_loop::run(self, input, Arc::clone(run.session()), None).await;
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
            let result =
                agent_loop::run_with(&runtime, input, Arc::clone(run.session()), Some(frame_tx))
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
}
