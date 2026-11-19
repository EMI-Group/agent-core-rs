//! `DeepAgent` facade, builder, sub-agent factory, session state persistence,
//! and round execution.
//!
//! [`DeepAgent`] is the harness product framework: it composes the kernel
//! [`ReActAgent`] with a workspace, a permission engine, coding tools, harness
//! rails and (optionally) sub-agents, then drives it either as one `ReAct` turn
//! or as the outer task loop. It never re-implements the model/tool loop.
//!
//! Module layout:
//!
//! - `mod.rs` (this file) — the [`DeepAgent`] facade, its [`Agent`] impl, the
//!   internal `Driver` used by spawned stream tasks, and follow-up/steer,
//!   mode and state entry points
//! - `builder` — [`DeepAgentBuilder`], prompt assembly, react-config
//!   application and lazy model resolution
//! - `factory` — [`resolve_deep_agent_parts`], [`apply_deep_agent_parts`] and
//!   [`create_deep_agent`]: the Python factory surface (default rails, base
//!   config merge, one-call assembly)
//! - `prompt_rail` — [`PromptRail`] and [`PromptState`]: system-prompt
//!   assembly plus steering injection before every model call
//! - `rail_events` — the rail event channels (`harness.goal_events`,
//!   `harness.events.*`), the [`RailEventDrain`] rail and the [`RailEventBus`]
//!   that forwards drained events to the invocation stream
//! - `round` — input normalization, [`RoundState`] and the task-loop
//!   [`RoundExecutorHandle`]
//! - `seed` — [`DeepAgent::seed_history`] and [`seed_state`]: pushing a
//!   persisted transcript into the kernel `"main"` conversation context and a
//!   host-built `deepagent` state payload into a session
//! - `state_store` — the session-persisted [`DeepAgentState`]
//! - `invocation_log` — the bounded per-invocation JSONL journal behind
//!   [`DeepAgentConfig::invocation_log_dir`], written from `Driver::run_resolved`
//! - `resilience` — the per-tool retry budget behind
//!   [`DeepAgentConfig::enable_tool_resilience_rail`], installed on the kernel agent
//! - `subagent_factory` — the default [`crate::subagents::SubAgentFactory`]
//!   and sub-agent tool wiring
//! - `tests` — offline unit and integration suite
//!
//! ## State sink (sync → async bridge)
//!
//! The task-loop controller invokes its `StateSink` synchronously after every
//! round, while persisting to a session is async. `Driver::run_task_loop`
//! therefore bridges the two with an unbounded channel: the sink forwards a
//! state clone, a background task writes it to the session, and after the loop
//! the controller's sink is replaced with a no-op so the channel closes and the
//! writer task can be awaited.
//!
//! ## Lazy wiring
//!
//! `new`/`configure` build a usable agent without a model; the first
//! invocation claims the wiring decision through an atomic compare-exchange,
//! resolves the model through `builder::ensure_agent_wired`, and marks the
//! agent wired. A failed resolution releases the claim so a later invocation can
//! retry.
//!
//! Size note: ~1095 lines of implementation plus `tests.rs`; the facade, its `Agent`
//! impl, the `RunningGuard` and the streaming `Driver` share the agent's private
//! wiring, so they stay in one file.

mod builder;
mod factory;
mod invocation_log;
mod prompt_rail;
mod rail_events;
mod resilience;
mod round;
mod seed;
mod state_store;
mod subagent_factory;
#[cfg(test)]
mod tests;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use futures::stream::BoxStream;
use futures::StreamExt;
use openjiuwen_agent::usage::MAIN_CONTEXT_ID;
use openjiuwen_agent::{
    resolve_session, Agent, AgentCard, AgentChunk, AgentOutput, ReActAgent, ResultType,
};
use openjiuwen_common::error::{Error, Result};
use openjiuwen_context::ModelContext;
use openjiuwen_foundation::llm::Model;
use openjiuwen_foundation::tool::{Tool, ToolCard};
use openjiuwen_session::Session;
use parking_lot::{Mutex, RwLock};
use serde_json::Value;
use tokio::sync::mpsc::Sender;
use tokio_stream::wrappers::ReceiverStream;
use tokio_util::sync::CancellationToken;

use crate::permission::{PermissionEngine, PermissionHost};
use crate::persistent_agent::{
    ActiveExecution, ExecutionRecorder, FinalStatus as RecordStatus, PersistentRunConfig,
};
use crate::prompts::{mode_instructions, SectionName};
use crate::rails::{HarnessRail, HarnessRailAdapter, HarnessRailRegistry, SubAgentSpawner};
use crate::schema::{
    AgentMode, DeepAgentConfig, DeepAgentState, InputDispatchMode, InteractionEvent,
    InteractionPhase, PlanModeState, SendInputRequest,
};
use crate::subagents::SubAgentRuntime;
use crate::task_loop::{
    DefaultTaskLoopEventHandler, LoopCoordinator, LoopQueues, StateSink, TaskLoopConfig,
    TaskLoopController,
};

pub use builder::DeepAgentBuilder;
pub use factory::{
    apply_deep_agent_parts, create_deep_agent, resolve_deep_agent_parts, DeepAgentParts,
};
pub use prompt_rail::{PromptRail, PromptState};
pub use rail_events::{
    drain_rail_events, RailEventBus, RailEventDrain, DEFAULT_RAIL_EVENT_CAPACITY,
    RAIL_EVENTS_EXTRA_PREFIX, RAIL_EVENT_DRAIN_NAME, RAIL_EVENT_DRAIN_PRIORITY,
};
pub use round::{normalize_inputs, RoundExecutorHandle, RoundState};
pub use state_store::{clear_state, load_state, save_state, seed_state, SESSION_STATE_KEY};
pub use subagent_factory::DeepAgentSubAgentFactory;

/// Error message of a second concurrent invocation.
const RUNNING_MESSAGE: &str =
    "DeepAgent invocation already running: only one invocation at a time is supported.";

/// Buffer of the public invocation stream.
///
/// Bounded so a consumer that stops polling applies backpressure instead of letting the run
/// buffer frames without limit; dropping the stream cancels the run.
const STREAM_CHANNEL_CAPACITY: usize = 64;

/// Error message of an explicit state entry point without a session.
const NO_SESSION_MESSAGE: &str = "no session bound to this DeepAgent";

/// Python `RuntimeError("interaction_terminated")` text of a stopped agent.
const TERMINATED_MESSAGE: &str = "interaction_terminated";

/// Error message of `send_input` without usable user text.
const SEND_INPUT_QUERY_MESSAGE: &str = "send_input requires a non-empty 'query'";

/// Upper bound of the configured completion timeout, guarding `Duration` conversion.
const MAX_COMPLETION_TIMEOUT_SECS: f64 = 1_000_000.0;

/// The `DeepAgent` task-loop coding agent.
///
/// All fields are private: the public surface is the documented methods plus
/// the [`Agent`] impl. Cloning is not supported; share it as
/// `Arc<DeepAgent>` (the builder returns one).
pub struct DeepAgent {
    /// Identity card of the agent.
    card: AgentCard,
    /// Data configuration; swapping it keeps live handles untouched.
    config: Arc<RwLock<DeepAgentConfig>>,
    /// Kernel agent executing model/tool turns.
    inner: Arc<ReActAgent>,
    /// Harness rails fired around the inner turns.
    rails: HarnessRailRegistry,
    /// Shared system-prompt state (also owned by the prompt rail).
    prompt: Arc<PromptState>,
    /// Model-facing cards of the currently mounted tools.
    tool_cards: Arc<RwLock<Vec<ToolCard>>>,
    /// Follow-up and steering queues shared with the loop and the prompt rail.
    queues: LoopQueues,
    /// Task-loop controller, present once task-loop mode has been used.
    controller: Arc<Mutex<Option<TaskLoopController>>>,
    /// Permission engine, when permissions are enabled.
    permission: Option<Arc<PermissionEngine>>,
    /// Host answering human-in-the-loop permission confirmations.
    permission_host: Arc<dyn PermissionHost>,
    /// Sub-agent runtime, when sub-agents are mounted.
    subagent_runtime: Arc<RwLock<Option<Arc<SubAgentRuntime>>>>,
    /// Delegation seam wired into the default subagent rail, when the builder registered it.
    subagent_spawner: Option<Arc<dyn SubAgentSpawner>>,
    /// Resolved model, supplied by the builder or built lazily.
    model: Arc<RwLock<Option<Arc<dyn Model>>>>,
    /// Whether the lazy-wiring decision has been claimed.
    initialized: Arc<AtomicBool>,
    /// Whether the inner agent has been wired with config, tools and rails.
    wired: Arc<AtomicBool>,
    /// Single-run guard: at most one invocation at a time per agent.
    running: Arc<AtomicBool>,
    /// Cancellation token of the current invocation.
    cancelled: Arc<Mutex<CancellationToken>>,
    /// Last round output and cumulative token usage.
    state: Arc<RoundState>,
    /// Events queued by rails, drained by [`RailEventDrain`] after every firing point.
    rail_events: Arc<RailEventBus>,
    /// Plan-mode bookkeeping.
    plan_mode: Arc<RwLock<PlanModeState>>,
    /// Session of the most recent invocation, when one was supplied.
    last_session: Arc<Mutex<Option<Arc<dyn Session>>>>,
    /// Last persisted state, merged with the live plan mode and follow-ups.
    last_state: Arc<Mutex<DeepAgentState>>,
}

impl DeepAgent {
    /// Creates an unwired `DeepAgent` with the default configuration.
    ///
    /// The prompt already carries the generic identity, safety, workspace and
    /// tool sections and the prompt rail is registered; no model is configured.
    /// An invocation therefore needs either [`DeepAgent::configure`] with a
    /// usable model identity or an agent built through [`DeepAgentBuilder`].
    ///
    /// Never panics and performs no IO.
    #[must_use]
    pub fn new(card: AgentCard) -> Self {
        let config = DeepAgentConfig::default().with_card(card.clone());
        let prompt = builder::build_prompt_state(&config, &[]);
        let queues = LoopQueues::default();
        let rails = HarnessRailRegistry::new();
        rails.register(Arc::new(PromptRail::new(prompt.clone(), queues.clone())));
        let rail_events = Arc::new(RailEventBus::default());
        rails.register(Arc::new(RailEventDrain::new(rail_events.clone())));
        let inner = Arc::new(ReActAgent::new(card.clone()));
        builder::apply_react_config(&inner, &config, &prompt.system_text());
        for rail in rails.to_agent_rails() {
            inner.register_rail(rail);
        }
        Self {
            card,
            config: Arc::new(RwLock::new(config)),
            inner,
            rails,
            prompt,
            tool_cards: Arc::new(RwLock::new(Vec::new())),
            queues,
            controller: Arc::new(Mutex::new(None)),
            permission: None,
            permission_host: Arc::new(crate::permission::NoopPermissionHost),
            subagent_runtime: Arc::new(RwLock::new(None)),
            subagent_spawner: None,
            model: Arc::new(RwLock::new(None)),
            initialized: Arc::new(AtomicBool::new(false)),
            wired: Arc::new(AtomicBool::new(false)),
            running: Arc::new(AtomicBool::new(false)),
            cancelled: Arc::new(Mutex::new(CancellationToken::new())),
            state: RoundState::new(),
            rail_events,
            plan_mode: Arc::new(RwLock::new(PlanModeState::default())),
            last_session: Arc::new(Mutex::new(None)),
            last_state: Arc::new(Mutex::new(DeepAgentState::default())),
        }
    }

    /// Replaces the data configuration.
    ///
    /// The configuration is validated first: an invalid one is rejected with the
    /// error [`DeepAgentConfig::validate`] reports (the Python `__post_init__`
    /// message text) and nothing is installed, so the previously configured agent
    /// keeps its prompt, kernel settings, permission policy and tool set.
    ///
    /// A valid configuration replaces the stored one, rebuilds the prompt sections
    /// from it (plus the mounted tool cards), re-applies the inner kernel config and
    /// the tool retry budget, and refreshes the permission policy when a permission
    /// engine exists and the new section is enabled. Tools are never re-mounted.
    ///
    /// # Errors
    ///
    /// Returns [`Error::config`] when [`DeepAgentConfig::validate`] rejects `config`.
    pub fn configure(&self, config: DeepAgentConfig) -> Result<()> {
        config.validate()?;
        if let Some(section) = config
            .permissions
            .as_ref()
            .filter(|section| section.enabled)
        {
            if let Some(engine) = &self.permission {
                engine.update_permissions(section.clone());
            }
        }
        let cards = self.tool_cards.read().clone();
        builder::replace_prompt_state(&self.prompt, &config, &cards);
        builder::apply_react_config(&self.inner, &config, &self.prompt.system_text());
        resilience::apply(&self.inner, &config, &cards);
        if !self.initialized.load(Ordering::Acquire) {
            self.wired.store(false, Ordering::Release);
        }
        *self.config.write() = config;
        Ok(())
    }

    /// Returns a copy of the data configuration.
    #[must_use]
    pub fn config(&self) -> DeepAgentConfig {
        self.config.read().clone()
    }
    /// Returns the cards of the tools currently mounted on this agent.
    ///
    /// The cards are what the model sees (and what the tools prompt section is built
    /// from), so a host can assert the mounted surface without reaching into the kernel
    /// agent: the twenty core tools, whatever
    /// [`DeepAgentConfig::enable_task_planning`] and the sub-agent mode add, and the
    /// caller-supplied tools. Config gating happens at build time, so the set never
    /// changes behind a caller's back except through
    /// [`DeepAgentBuilder`].
    #[must_use]
    pub fn tool_cards(&self) -> Vec<ToolCard> {
        self.tool_cards.read().clone()
    }

    /// Wires the agent and resolves its model when that has not happened yet.
    ///
    /// Idempotent: once the agent is wired and a model is available this is a
    /// no-op. Concurrent first calls race on `Self::initialized`, so only one
    /// performs the work; a failure releases the claim for a later retry.
    ///
    /// # Errors
    ///
    /// Returns [`Error::config`] when no model was supplied and the configured
    /// model identity is unusable.
    pub async fn ensure_initialized(&self) -> Result<()> {
        if self.wired.load(Ordering::Acquire) && self.model.read().is_some() {
            return Ok(());
        }
        if self
            .initialized
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Ok(());
        }
        let driver = self.driver();
        match builder::ensure_agent_wired(&driver).await {
            Ok(()) => {
                self.wired.store(true, Ordering::Release);
                Ok(())
            }
            Err(error) => {
                self.initialized.store(false, Ordering::Release);
                Err(error)
            }
        }
    }

    /// Queues a follow-up user message for the next round (or next invocation).
    pub fn follow_up(&self, message: impl Into<String>, task_id: Option<String>) {
        let input = crate::task_loop::QueuedInput::from(message.into()).with_task_id(task_id);
        match self.controller.lock().as_ref() {
            Some(controller) => controller.push_follow_up(input),
            None => self.queues.push_follow_up(input),
        }
    }

    /// Queues a steering input for the next model call.
    pub fn steer(&self, message: impl Into<String>) {
        let input = crate::task_loop::QueuedInput::from(message.into());
        match self.controller.lock().as_ref() {
            Some(controller) => controller.push_steer(input),
            None => self.queues.push_steer(input),
        }
    }

    /// Aborts the in-flight invocation.
    ///
    /// Synchronous and idempotent: the cancellation token is cancelled and the
    /// controller (when one exists) is asked to stop after the current round.
    pub fn abort(&self) {
        self.cancelled.lock().cancel();
        if let Some(controller) = self.controller.lock().as_ref() {
            controller.request_abort();
        }
    }

    /// Re-opens a stopped agent.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`] with `interaction_terminated` once the
    /// agent has been stopped.
    pub fn start(&self) -> Result<()> {
        if self.cancelled.lock().is_cancelled() {
            return Err(Error::validation(TERMINATED_MESSAGE));
        }
        Ok(())
    }

    /// Stops the agent: the current invocation is cancelled and the loop is
    /// terminated.
    ///
    /// # Errors
    ///
    /// Currently infallible; the signature mirrors the Python `stop`.
    pub fn stop(&self) -> Result<()> {
        self.cancelled.lock().cancel();
        if let Some(controller) = self.controller.lock().as_ref() {
            controller.cancel();
        }
        Ok(())
    }

    /// Dispatches one host input through the follow-up or steering queue.
    ///
    /// `mode` decides the destination; `None` queues a follow-up while idle and
    /// steers while an invocation is running.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`] when `inputs["query"]` is missing or empty.
    #[allow(clippy::needless_pass_by_value)] // part of the fixed public surface
    pub fn send_input(&self, request: SendInputRequest) -> Result<()> {
        let content = request
            .inputs
            .get("query")
            .and_then(Value::as_str)
            .filter(|query| !query.is_empty())
            .ok_or_else(|| Error::validation(SEND_INPUT_QUERY_MESSAGE))?;
        let mode = request.mode.unwrap_or_else(|| {
            if self.running.load(Ordering::Acquire) {
                InputDispatchMode::Steer
            } else {
                InputDispatchMode::FollowUp
            }
        });
        match mode {
            InputDispatchMode::FollowUp => self.follow_up(content.to_string(), None),
            InputDispatchMode::Steer => self.steer(content.to_string()),
        }
        Ok(())
    }

    /// Returns the lifecycle phase of the agent.
    ///
    /// A cancelled token reports [`InteractionPhase::Terminated`]; otherwise the
    /// controller's phase wins and an agent without a controller is idle.
    #[must_use]
    pub fn phase(&self) -> InteractionPhase {
        if self.cancelled.lock().is_cancelled() {
            return InteractionPhase::Terminated;
        }
        self.controller
            .lock()
            .as_ref()
            .map_or(InteractionPhase::Idle, TaskLoopController::phase)
    }

    /// Switches the agent between normal and plan mode.
    ///
    /// Updates the configured default mode, the plan-mode bookkeeping, and the
    /// mode-instructions prompt section (present only in plan mode).
    ///
    /// # Errors
    ///
    /// Returns an error when the prompt builder rejects the section.
    pub fn switch_mode(&self, mode: AgentMode) -> Result<()> {
        let previous = self.config.read().default_mode;
        {
            let mut plan = self.plan_mode.write();
            if mode == AgentMode::Plan && previous != AgentMode::Plan {
                plan.pre_plan_mode = Some(previous.as_str().to_string());
            }
            plan.mode = mode.as_str().to_string();
        }
        self.config.write().default_mode = mode;
        self.prompt.update(|prompt| {
            prompt.remove_section(SectionName::MODE_INSTRUCTIONS);
            if mode == AgentMode::Plan {
                if let Err(error) = prompt.add_section(mode_instructions(mode)) {
                    tracing::warn!(error = %error, "failed to add the mode instructions section");
                }
            }
        });
        let config = self.config.read().clone();
        builder::apply_react_config(&self.inner, &config, &self.prompt.system_text());
        Ok(())
    }

    /// Persists the current agent state on the last used session.
    ///
    /// The persisted payload merges the last state with the live plan mode and
    /// the queued follow-ups.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`] without a session and [`Error::session`]
    /// when the session write fails.
    pub async fn save_state(&self) -> Result<()> {
        let session = self
            .last_session
            .lock()
            .clone()
            .ok_or_else(|| Error::validation(NO_SESSION_MESSAGE))?;
        self.driver().persist_state(&session).await
    }

    /// Loads the persisted agent state from the last used session.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`] without a session and [`Error::session`]
    /// when the session read fails.
    pub async fn load_state(&self) -> Result<DeepAgentState> {
        let session = self
            .last_session
            .lock()
            .clone()
            .ok_or_else(|| Error::validation(NO_SESSION_MESSAGE))?;
        state_store::load_state(session.as_ref()).await
    }

    /// Clears the persisted agent state on the last used session.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`] without a session and [`Error::session`]
    /// when the session write fails.
    pub async fn clear_state(&self) -> Result<()> {
        let session = self
            .last_session
            .lock()
            .clone()
            .ok_or_else(|| Error::validation(NO_SESSION_MESSAGE))?;
        state_store::clear_state(session.as_ref()).await
    }

    /// Registers a harness rail on both the harness registry and the inner agent.
    pub fn add_rail(&self, rail: Arc<dyn HarnessRail>) {
        self.rails.register(rail.clone());
        self.inner
            .register_rail(Arc::new(HarnessRailAdapter::new(rail)));
    }

    /// Registers a kernel rail on the inner agent.
    pub fn register_rail(&self, rail: Arc<dyn openjiuwen_agent::AgentRail>) {
        self.inner.register_rail(rail);
    }

    /// Returns the harness rail registry.
    #[must_use]
    pub fn rails(&self) -> &HarnessRailRegistry {
        &self.rails
    }

    /// Returns the production sub-agent seam wired into the default subagent rail.
    ///
    /// [`DeepAgentBuilder`] registers a
    /// [`RuntimeSubAgentSpawner`](crate::subagents::RuntimeSubAgentSpawner) over the sub-agent
    /// runtime it mounts, so hosts can delegate through [`SubAgentSpawner`] without owning the
    /// runtime.
    ///
    /// `None` when sub-agents are not mounted (`subagents` is empty and
    /// `add_general_purpose_agent` is off), and when the caller supplied its own
    /// `subagent` rail — a caller rail always wins.
    #[must_use]
    pub fn subagent_spawner(&self) -> Option<Arc<dyn SubAgentSpawner>> {
        self.subagent_spawner.clone()
    }

    /// Returns and removes the interaction events rails have queued so far.
    ///
    /// [`RailEventDrain`] moves every `InteractionEvent` a rail queued under
    /// `harness.goal_events` or a [`RAIL_EVENTS_EXTRA_PREFIX`] key into the
    /// agent's buffer after each rail firing point. A streaming caller observes
    /// the same events as `__interaction__` frames; this entry point serves
    /// hosts that invoke without a stream. Events survive across rounds and
    /// invocations until they are collected, and the buffer keeps at most
    /// [`DEFAULT_RAIL_EVENT_CAPACITY`] events, dropping the oldest first.
    #[must_use]
    pub fn drain_rail_events(&self) -> Vec<InteractionEvent> {
        self.rail_events.drain()
    }

    /// Mounts additional tools and refreshes the tools prompt section.
    ///
    /// Used by [`apply_deep_agent_parts`] (and upper layers that add tools after
    /// construction); the mounted cards are appended so the prompt always
    /// reflects the live tool set.
    #[allow(clippy::needless_pass_by_value)] // the parts carry an owned `Vec`
    pub(crate) fn apply_extra_tools(&self, tools: Vec<Arc<dyn Tool>>) {
        for tool in &tools {
            self.inner.add_tool(tool.clone());
            self.tool_cards.write().push(tool.card().clone());
        }
        let cards = self.tool_cards.read().clone();
        self.prompt.update(|prompt| {
            prompt.remove_section(SectionName::TOOLS);
            if let Err(error) = prompt.add_section(crate::prompts::tools_section(&cards)) {
                tracing::warn!(error = %error, "failed to refresh the tools prompt section");
            }
        });
        let config = self.config.read().clone();
        builder::apply_react_config(&self.inner, &config, &self.prompt.system_text());
        resilience::apply(&self.inner, &config, &cards);
    }

    /// Builds the internal driver used by invocation and stream tasks.
    pub(crate) fn driver(&self) -> Driver {
        Driver {
            card: self.card.clone(),
            config: self.config.clone(),
            inner: self.inner.clone(),
            rails: self.rails.clone(),
            prompt: self.prompt.clone(),
            tool_cards: self.tool_cards.clone(),
            queues: self.queues.clone(),
            controller: self.controller.clone(),
            permission: self.permission.clone(),
            permission_host: self.permission_host.clone(),
            subagent_runtime: self.subagent_runtime.clone(),
            model: self.model.clone(),
            initialized: self.initialized.clone(),
            wired: self.wired.clone(),
            cancelled: self.cancelled.clone(),
            state: self.state.clone(),
            rail_events: self.rail_events.clone(),
            plan_mode: self.plan_mode.clone(),
            last_session: self.last_session.clone(),
            last_state: self.last_state.clone(),
        }
    }
}

#[async_trait::async_trait]
impl Agent for DeepAgent {
    fn card(&self) -> &AgentCard {
        &self.card
    }

    /// Runs one invocation to completion.
    ///
    /// Exactly one invocation may run per agent; a second concurrent call fails
    /// with [`Error::config`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::config`] when an invocation is already running or the
    /// model cannot be resolved, and [`Error::validation`] for invalid inputs or
    /// a cancelled invocation.
    async fn invoke(
        &self,
        inputs: Value,
        session: Option<Arc<dyn Session>>,
    ) -> Result<AgentOutput> {
        let guard = RunningGuard::acquire(self.running.clone())?;
        let driver = self.driver();
        let cancel = driver.cancel_token();
        let output = driver.invoke_inner(inputs, session, None, cancel).await;
        drop(guard);
        output
    }

    /// Runs one invocation, yielding stream frames as they are produced.
    ///
    /// The channel is bounded: a consumer that stops polling applies backpressure, and dropping
    /// the stream cancels the in-flight turn so it stops burning tokens and releases the
    /// single-run slot. The terminal `answer` frame is emitted for a completed run; interrupt
    /// runs forward their interaction frames only.
    fn stream(
        &self,
        inputs: Value,
        session: Option<Arc<dyn Session>>,
    ) -> BoxStream<'static, Result<AgentChunk>> {
        let running = self.running.clone();
        let guard = match RunningGuard::acquire(running) {
            Ok(guard) => guard,
            Err(error) => return futures::stream::once(async move { Err(error) }).boxed(),
        };
        let (tx, rx) = tokio::sync::mpsc::channel(STREAM_CHANNEL_CAPACITY);
        let driver = self.driver();
        // The token is claimed before the task starts, so a consumer that drops the stream
        // immediately still cancels the exact turn that is about to run.
        let cancel = driver.cancel_token();
        tokio::spawn(async move {
            let _guard = guard;
            let sender = Some(tx.clone());
            let mut invocation =
                Box::pin(driver.invoke_inner(inputs, session, sender, cancel.clone()));
            loop {
                tokio::select! {
                    biased;
                    _ = tx.closed() => {
                        // Nobody is reading: cancel the turn and let it unwind so the
                        // running slot is free for the next invocation.
                        cancel.cancel();
                        let _ = invocation.as_mut().await;
                        break;
                    }
                    result = &mut invocation => {
                        match result {
                            Ok(output) => {
                                if !output.is_interrupt() {
                                    let _ = tx.send(Ok(AgentChunk::answer(&output))).await;
                                }
                            }
                            Err(error) => {
                                let _ = tx.send(Err(error)).await;
                            }
                        }
                        break;
                    }
                }
            }
        });
        ReceiverStream::new(rx).boxed()
    }
}

/// Releases the single-run guard when it leaves scope.
struct RunningGuard {
    /// The shared flag to clear.
    flag: Arc<AtomicBool>,
}

impl RunningGuard {
    /// Claims the single-run slot, or reports the concurrent invocation.
    fn acquire(flag: Arc<AtomicBool>) -> Result<Self> {
        flag.compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| Error::config(RUNNING_MESSAGE))?;
        Ok(Self { flag })
    }
}

impl Drop for RunningGuard {
    fn drop(&mut self) {
        self.flag.store(false, Ordering::Release);
    }
}

/// Cloneable set of handles one invocation needs, independent of [`DeepAgent`].
///
/// Spawned stream tasks own a `Driver` instead of an `Arc<DeepAgent>`, so the
/// dispatch does not require the agent to be self-referential. Every field is a
/// cheap clone of an `Arc` (or the card), and no lock is held by the driver
/// itself.
#[allow(dead_code)]
pub(crate) struct Driver {
    /// Identity card of the owning agent.
    pub(crate) card: AgentCard,
    /// Data configuration.
    pub(crate) config: Arc<RwLock<DeepAgentConfig>>,
    /// Kernel agent executing the turns.
    pub(crate) inner: Arc<ReActAgent>,
    /// Harness rails.
    pub(crate) rails: HarnessRailRegistry,
    /// System-prompt state.
    pub(crate) prompt: Arc<PromptState>,
    /// Mounted tool cards.
    pub(crate) tool_cards: Arc<RwLock<Vec<ToolCard>>>,
    /// Follow-up and steering queues.
    pub(crate) queues: LoopQueues,
    /// Task-loop controller slot.
    pub(crate) controller: Arc<Mutex<Option<TaskLoopController>>>,
    /// Permission engine.
    pub(crate) permission: Option<Arc<PermissionEngine>>,
    /// Permission host.
    pub(crate) permission_host: Arc<dyn PermissionHost>,
    /// Sub-agent runtime slot.
    pub(crate) subagent_runtime: Arc<RwLock<Option<Arc<SubAgentRuntime>>>>,
    /// Resolved model slot.
    pub(crate) model: Arc<RwLock<Option<Arc<dyn Model>>>>,
    /// Lazy-wiring claim.
    pub(crate) initialized: Arc<AtomicBool>,
    /// Wiring flag.
    pub(crate) wired: Arc<AtomicBool>,
    /// Cancellation token.
    pub(crate) cancelled: Arc<Mutex<CancellationToken>>,
    /// Last round output and token usage.
    pub(crate) state: Arc<RoundState>,
    /// Events queued by rails and drained after every firing point.
    pub(crate) rail_events: Arc<RailEventBus>,
    /// Plan-mode bookkeeping.
    pub(crate) plan_mode: Arc<RwLock<PlanModeState>>,
    /// Session of the most recent invocation.
    pub(crate) last_session: Arc<Mutex<Option<Arc<dyn Session>>>>,
    /// Last persisted state.
    pub(crate) last_state: Arc<Mutex<DeepAgentState>>,
}

impl Driver {
    /// Ensures the agent is wired, resolving the model when needed.
    ///
    /// # Errors
    ///
    /// Propagates [`builder::ensure_agent_wired`] errors.
    pub(crate) async fn ensure_wired(&self) -> Result<()> {
        builder::ensure_agent_wired(self).await
    }

    /// Runs one whole invocation: normalize, execute, persist, return.
    ///
    /// `stream_tx` forwards stream frames when the caller wants them; the rail
    /// event bus forwards its drained events to the same channel for the
    /// duration of the invocation and drops the sender afterwards.
    ///
    /// # Errors
    ///
    /// Propagates input validation, model-resolution, session-read and
    /// round-execution errors.
    pub(crate) async fn invoke_inner(
        &self,
        inputs: Value,
        session: Option<Arc<dyn Session>>,
        stream_tx: Option<Sender<Result<AgentChunk>>>,
        cancel: CancellationToken,
    ) -> Result<AgentOutput> {
        self.rail_events.set_stream(stream_tx.clone());
        let result = self
            .invoke_streaming(inputs, session, stream_tx, cancel)
            .await;
        self.rail_events.set_stream(None);
        result
    }

    /// Body of [`Self::invoke_inner`] with the stream already installed.
    async fn invoke_streaming(
        &self,
        inputs: Value,
        session: Option<Arc<dyn Session>>,
        stream_tx: Option<Sender<Result<AgentChunk>>>,
        cancel: CancellationToken,
    ) -> Result<AgentOutput> {
        self.ensure_wired().await?;
        let normalized = normalize_inputs(&inputs)?;
        let conversation_id = normalized.get("conversation_id").and_then(Value::as_str);
        let parent_session_id = normalized.get("parent_session_id").and_then(Value::as_str);
        let run = resolve_session(
            session,
            &self.card.id,
            Some(&self.card.name),
            conversation_id,
            parent_session_id,
        )
        .await?;
        let session = run.session().clone();
        let result = self
            .run_resolved(&normalized, &session, stream_tx, cancel)
            .await;
        run.finish().await;
        result
    }

    /// Runs an invocation on the resolved session: reset, execute, persist.
    ///
    /// # Errors
    ///
    /// Propagates session-read and round-execution errors; a persistence
    /// failure is logged, never propagated.
    async fn run_resolved(
        &self,
        normalized: &Value,
        session: &Arc<dyn Session>,
        stream_tx: Option<Sender<Result<AgentChunk>>>,
        cancel: CancellationToken,
    ) -> Result<AgentOutput> {
        *self.last_session.lock() = Some(session.clone());
        self.state.reset();
        let loaded = state_store::load_state(session.as_ref()).await?;
        *self.last_state.lock() = loaded.clone();

        let config = self.config.read().clone();
        let interactive = normalized.get("query").is_some_and(Value::is_object);
        let task_loop =
            config.enable_task_loop && config.max_iterations_hard_limit.is_none() && !interactive;
        let recording = self.begin_recording(normalized, session).await;
        let journal = invocation_log::InvocationJournal::open(
            &config,
            &self.card.id,
            &self.card.name,
            session.id(),
            &recorded_query(normalized),
        )
        .await;
        let outcome = if task_loop {
            self.run_task_loop(normalized, session.clone(), loaded, stream_tx, cancel)
                .await
        } else {
            self.round_handle(Some(session.clone()), stream_tx, cancel)
                .with_timeout(completion_timeout(config.completion_timeout_secs))
                .execute(normalized)
                .await
        };
        if let Some(journal) = &journal {
            match &outcome {
                Ok(output) => journal.finish(output).await,
                Err(error) => journal.finish_error(&error.to_string()).await,
            }
        }
        let output = outcome?;

        if let Err(error) = self.persist_state(session).await {
            // Deviation from Python (which propagates): an answer must not be
            // lost to a persistence hiccup.
            tracing::warn!(error = %error, "failed to persist the DeepAgent state");
        }
        if let Some(active) = recording {
            self.finish_recording(&active, session, &output).await;
        }
        Ok(output)
    }

    /// Opens this turn's persistent-agent execution journal, when configured.
    ///
    /// Returns `None` when [`DeepAgentConfig::persistent_agent`] is off, and also when the journal
    /// cannot be opened: recording is an observation of the turn, never a precondition for it, so
    /// a failure is logged and the turn runs unrecorded (the same policy
    /// [`Self::persist_state`] follows).
    async fn begin_recording(
        &self,
        normalized: &Value,
        session: &Arc<dyn Session>,
    ) -> Option<ActiveExecution> {
        let config = self.config.read().clone();
        if !config.persistent_agent {
            return None;
        }
        let run = PersistentRunConfig::new(self.card.id.clone(), session.id());
        let recorder = ExecutionRecorder::for_workspace(&config.workspace);
        match recorder.begin(&run, &recorded_query(normalized)).await {
            Ok(active) => Some(active),
            Err(error) => {
                tracing::warn!(error = %error, "failed to open the persistent-agent journal");
                None
            }
        }
    }

    /// Records the turn's outcome and rewrites the persisted session record.
    ///
    /// Both writes are best effort: a failure is logged and never propagated, because the answer
    /// the caller already holds must not be replaced by a recording error.
    async fn finish_recording(
        &self,
        active: &ActiveExecution,
        session: &Arc<dyn Session>,
        output: &AgentOutput,
    ) {
        let status = if matches!(output.result_type, ResultType::Answer) {
            RecordStatus::Completed
        } else {
            RecordStatus::Failed
        };
        if let Err(error) = active.finish(status, Some(&output.output)).await {
            tracing::warn!(error = %error, "failed to finalise the persistent-agent journal");
        }
        let messages = match self
            .inner
            .context_engine()
            .get_context(session.id(), MAIN_CONTEXT_ID)
            .await
        {
            Some(context) => context.messages().await,
            None => Vec::new(),
        };
        let state = serde_json::to_value(self.last_state.lock().clone()).ok();
        if let Err(error) = active.save_session(messages, state).await {
            tracing::warn!(error = %error, "failed to save the persistent-agent session record");
        }
    }

    /// Executes `normalized` through the outer task loop.
    async fn run_task_loop(
        &self,
        normalized: &Value,
        session: Arc<dyn Session>,
        loaded: DeepAgentState,
        stream_tx: Option<Sender<Result<AgentChunk>>>,
        cancel: CancellationToken,
    ) -> Result<AgentOutput> {
        let (state_tx, mut state_rx) = tokio::sync::mpsc::unbounded_channel::<DeepAgentState>();
        let sink: StateSink = Arc::new(move |state: &DeepAgentState| {
            state_tx
                .send(state.clone())
                .map_err(|_| Error::session("DeepAgent state sink closed"))
        });
        let writer = tokio::spawn({
            let session = session.clone();
            async move {
                while let Some(state) = state_rx.recv().await {
                    if let Err(error) = state_store::save_state(session.as_ref(), &state).await {
                        tracing::warn!(error = %error, "failed to persist the DeepAgent state");
                    }
                }
            }
        });

        let existing = self.controller.lock().take();
        let mut controller = match existing {
            Some(mut controller) if reusable(&controller) => {
                controller.set_state(loaded);
                controller.set_state_sink(sink);
                controller
            }
            _ => self.make_controller(loaded, sink),
        };

        let inputs = normalized
            .as_object()
            .map(|map| {
                map.iter()
                    .map(|(key, value)| (key.clone(), value.clone()))
                    .collect()
            })
            .unwrap_or_default();
        controller.submit_round(inputs, false, None);
        let handle = self.round_handle(Some(session.clone()), stream_tx, cancel);
        let report = controller.run_until_idle(&handle).await;
        *self.last_state.lock() = controller.state();

        // Close the sink so the writer task observes the end of the channel and
        // can be awaited, then publish the controller for later invocations.
        let closed: StateSink = Arc::new(|_state: &DeepAgentState| Ok(()));
        controller.set_state_sink(closed);
        *self.controller.lock() = Some(controller);
        if let Err(error) = writer.await {
            tracing::warn!(error = %error, "DeepAgent state writer task failed");
        }
        Ok(report_output(&report, &handle))
    }

    /// Builds a fresh task-loop controller with the configured budgets.
    fn make_controller(&self, state: DeepAgentState, sink: StateSink) -> TaskLoopController {
        let config = self.config.read().clone();
        TaskLoopController::new(
            Box::new(DefaultTaskLoopEventHandler::new().with_queues(self.queues.clone())),
            LoopCoordinator::with_defaults(&config.task_loop_no_progress_guard),
            self.queues.clone(),
            TaskLoopConfig::default()
                .with_completion_timeout(completion_timeout(config.completion_timeout_secs)),
        )
        .with_state(state)
        .with_state_sink(sink)
    }

    /// Builds one round handle, forwarding frames when `stream_tx` is present.
    fn round_handle(
        &self,
        session: Option<Arc<dyn Session>>,
        stream_tx: Option<Sender<Result<AgentChunk>>>,
        cancel: CancellationToken,
    ) -> RoundExecutorHandle {
        let handle = RoundExecutorHandle::new(
            self.inner.clone(),
            self.rails.clone(),
            self.queues.clone(),
            cancel,
            session,
            self.state.clone(),
        )
        .with_rail_events(self.rail_events.clone());
        match stream_tx {
            Some(tx) => handle.with_stream_sender(tx),
            None => handle,
        }
    }

    /// Returns the current cancellation token, replacing a cancelled one.
    fn cancel_token(&self) -> CancellationToken {
        let mut guard = self.cancelled.lock();
        if guard.is_cancelled() {
            *guard = CancellationToken::new();
        }
        guard.clone()
    }

    /// Adds the live plan mode and queued follow-ups to `base`.
    pub(crate) fn collect_state(&self, mut base: DeepAgentState) -> DeepAgentState {
        base.plan_mode = self.plan_mode.read().clone();
        base.pending_follow_ups = self.queues.follow_up_contents();
        base
    }

    /// Persists the collected state on `session`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::session`] when the session write fails.
    pub(crate) async fn persist_state(&self, session: &Arc<dyn Session>) -> Result<()> {
        let base = self.last_state.lock().clone();
        let state = self.collect_state(base);
        *self.last_state.lock() = state.clone();
        state_store::save_state(session.as_ref(), &state).await
    }
}

/// Returns the goal text a persistent-agent journal records for `normalized`.
///
/// [`normalize_inputs`] leaves the query as either the user's string or an interactive object (an
/// interruption reply). A string is recorded verbatim; an object is rendered from its first
/// textual field, falling back to its compact JSON so an interruption reply still journals
/// something a reader can identify.
fn recorded_query(normalized: &Value) -> String {
    match normalized.get("query") {
        Some(Value::String(text)) => text.clone(),
        Some(Value::Object(map)) => {
            for key in ["content", "text", "query"] {
                if let Some(text) = map.get(key).and_then(Value::as_str) {
                    return text.to_string();
                }
            }
            Value::Object(map.clone()).to_string()
        }
        Some(other) => other.to_string(),
        None => String::new(),
    }
}

/// Whether a stored controller can serve another invocation.
///
/// Aborted, cancelled and terminated controllers carry latched state (the
/// coordinator's abort flag, the cancel token, the terminated phase) that would
/// poison the next round, so only a clean controller is reused.
fn reusable(controller: &TaskLoopController) -> bool {
    !controller.is_aborted()
        && !controller.is_cancelled()
        && controller.phase() != InteractionPhase::Terminated
}

/// Maps a configured completion timeout onto a valid [`Duration`].
fn completion_timeout(seconds: f64) -> Duration {
    if seconds.is_finite() && seconds > 0.0 {
        Duration::from_secs_f64(seconds.min(MAX_COMPLETION_TIMEOUT_SECS))
    } else {
        Duration::ZERO
    }
}

/// Projects a finished task loop onto an [`AgentOutput`].
fn report_output(
    report: &crate::task_loop::LoopReport,
    handle: &RoundExecutorHandle,
) -> AgentOutput {
    if let Some(output) = handle.last_output() {
        return output;
    }
    if let Some(outcome) = &report.last_outcome {
        if let Some(code) = &outcome.error_code {
            return AgentOutput::error(format!(
                "{code}: {}",
                outcome.error_message.clone().unwrap_or_default()
            ));
        }
        return AgentOutput::answer("");
    }
    if report.aborted {
        return AgentOutput::error("aborted");
    }
    AgentOutput::answer("")
}
