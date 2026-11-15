//! One invoke of the `ReAct` loop: runtime snapshot, interruption load/resume,
//! the iteration loop, rails, and termination.
//!
//! The loop drives the whole turn from [`run`]:
//!
//! - snapshot the agent into an owned [`AgentRuntime`] (config, model, request
//!   config, rails, context engine, tools, policy seams) so a streaming
//!   invocation can move it into a spawned task;
//! - load any persisted [`InterruptionState`](crate::interrupt::InterruptionState)
//!   and replay the interrupted tool calls, or seed a fresh turn with the user
//!   query;
//! - for each iteration in `start..=limit`, drain the steering backlog, rebuild
//!   the system prompt, prepare the window with `ContextEngine::prepare`, emit a
//!   `context.usage` frame, fire `BeforeModelCall`, call the model through
//!   [`call_model_streaming`](crate::react::model_step::call_model_streaming),
//!   and fire `AfterModelCall`; the call itself is wrapped in best-effort
//!   KV-cache affinity ([`kv_cache`]): the request scope is prepared, its
//!   inference admitted, its prefix recorded, and the provider's input tokens
//!   reported afterwards, without ever changing the run's outcome;
//! - execute the model's tool calls in model order through
//!   [`execute_calls_with`](crate::abilities::execute::execute_calls_with),
//!   firing `BeforeToolCall`/`AfterToolCall` around each call and appending the
//!   resulting tool messages with `ContextEngine::observe`;
//! - fire `AfterReactIteration` after each iteration;
//! - honour rail control flags: `force_finish` returns immediately, `retry`
//!   repeats the model call once, `model_continue` keeps iterating, `skip_tool`
//!   substitutes the tool result without executing it, and `interrupts` pause the
//!   run once every call of the turn has been asked, so one turn surfaces all of
//!   its pending requests together and executes no tool;
//! - pause a call the connected client must run as a `client_tool` interrupt
//!   instead of executing it locally;
//! - persist contexts with `ContextEngine::save_contexts` and interruption state
//!   under `__react_agent_interruption__`;
//! - terminate with `termination_reason` `max_iterations` or `token_budget`.
mod kv_cache;

use crate::abilities::execute::execute_calls_with;
use crate::abilities::{ToolCallResult, ToolExecutionOptions};
use crate::callbacks::CallbackFramework;
use crate::input::AgentInput;
use crate::interrupt::{InterruptKind, InterruptRequest};
use crate::memory_provider::{finalize_session, injected_prompt_block};
use crate::output::{AgentChunk, AgentOutput, Observation};
use crate::policy::{
    ClientToolRegistry, FinalRequestCheck, McpToolAllowlist, ToolExposurePolicy,
    ToolResiliencePolicy,
};
use crate::rails::memory::MEMORY_VARIABLES_KEY;
use crate::rails::{RailContext, RailEvent, RailRegistry};
use crate::react::model_step::{call_model, call_model_streaming, DeltaSink};
use crate::react::pause::pause_output;
use crate::react::prompt::build_system_with_block;
use crate::react::resume::{resume_or_seed, ResumeOutcome};
use crate::react::{exposure, frames, hooks};
use crate::react::{MaxIterationsPolicy, ReActAgent, ReActAgentConfig};
use crate::steering::SteeringQueue;
use crate::usage::{self, UsageTelemetry};
use indexmap::IndexMap;
use openjiuwen_common::error::Result;
use openjiuwen_context::{ContextEngine, ConversationContext, PrepareOptions};
use openjiuwen_foundation::{
    BaseMessage, Model, ModelRequestConfig, Tool, ToolCall, ToolCard, Usage,
};
use openjiuwen_memory::{ConversationTurn, MemoryProvider};
use openjiuwen_session::{InvokeType, Session};
use serde_json::Value;
use std::collections::BTreeMap;
use std::sync::Arc;
use tokio::sync::mpsc::UnboundedSender;

/// Message of the `result_type: "error"` output produced by an empty model turn.
const EMPTY_RESPONSE_MSG: &str = "模型未返回有效内容（空响应），请重试或检查上下文。";
/// Notice injected before retrying one empty model turn.
const EMPTY_RESPONSE_NOTICE: &str =
    "Your previous response was empty. Reply with a final answer or call a tool.";
/// Message of the output produced when the iteration budget is exhausted.
const MAX_ITERATIONS_MSG: &str = "Max iterations reached without completion";
/// Message of the output produced when the token budget is exceeded.
const TOKEN_BUDGET_MSG: &str = "Token budget exceeded";
/// Nudge the `FinalAnswer` policy appends before its last tool-free model call.
const FINAL_ANSWER_NUDGE: &str = "Provide your best final answer now without calling any tools.";
/// Metadata key marking the injected empty-response notice.
const EMPTY_RESPONSE_SOURCE_KEY: &str = "_openjiuwen_message_source_kind";
/// The `phase` a snapshot of a completed model call reports.
const POST_CALL_PHASE: &str = "post_call";

/// Owned snapshot of everything one run needs from an [`ReActAgent`].
///
/// [`crate::react::ReActAgent::stream`] returns a `BoxStream<'static, _>`, so the
/// run must not borrow the agent: the snapshot is built synchronously (no lock
/// guard is held across an `await`) and moved into the spawned task.
pub(crate) struct AgentRuntime {
    /// Id of the agent's card; the rail `agent_id`.
    pub(crate) card_id: String,
    /// Name of the agent's card; the root span name.
    pub(crate) card_name: String,
    /// Configuration the run follows.
    pub(crate) config: ReActAgentConfig,
    /// Model every iteration calls.
    pub(crate) model: Arc<dyn Model>,
    /// Request-level model configuration, with the model name resolved.
    pub(crate) request_config: ModelRequestConfig,
    /// Rails fired around the loop's steps.
    pub(crate) rails: Arc<RailRegistry>,
    /// Context engine owning the conversation window.
    pub(crate) engine: Arc<ContextEngine>,
    /// Mounted tools keyed by name, in model-visible order.
    pub(crate) tools: IndexMap<String, Arc<dyn Tool>>,
    /// Cards of every mounted tool, in model-visible order.
    pub(crate) tool_cards: Vec<ToolCard>,
    /// Model-facing schemas after the exposure policy and MCP allowlist.
    pub(crate) visible_tool_info: Vec<Value>,
    /// Model-facing cards after the exposure policy and MCP allowlist.
    pub(crate) visible_cards: Vec<ToolCard>,
    /// Lifecycle callbacks observing this run.
    pub(crate) callbacks: Arc<CallbackFramework>,
    /// Steering backlog drained between iterations.
    pub(crate) steering: SteeringQueue,
    /// Policy deciding which tools the model may see.
    pub(crate) exposure: ToolExposurePolicy,
    /// Per-server MCP tool allowlist.
    pub(crate) mcp_allowlist: McpToolAllowlist,
    /// Handlers for tool calls the connected client executes.
    pub(crate) client_tools: Arc<ClientToolRegistry>,
    /// Per-tool resilience budget applied to execution.
    pub(crate) resilience: ToolResiliencePolicy,
    /// Check run before a final model request, when installed.
    pub(crate) final_check: Option<Arc<dyn FinalRequestCheck>>,
    /// KV-cache hook the model calls record their prefix affinity through.
    pub(crate) kv_cache_hook: Option<Arc<crate::kv_cache::KVCacheModelCallHook>>,
    /// External long-term-memory provider, when the agent has one installed.
    pub(crate) memory_provider: Option<Arc<dyn MemoryProvider>>,
    /// Buffer the post-call usage snapshots; shared with the agent.
    pub(crate) usage_telemetry: Arc<UsageTelemetry>,
}

impl AgentRuntime {
    /// Copies the agent's runtime state into an owned value.
    ///
    /// Every lock is taken and released synchronously, so the returned value is
    /// independent of later agent mutations. The model-facing tool projection
    /// (`visible_tool_info`/`visible_cards`) is computed here, once per run.
    ///
    /// # Errors
    ///
    /// [`Error::Config`] when the agent has no model configured.
    pub(crate) fn snapshot(agent: &ReActAgent) -> Result<Self> {
        let model = agent.model()?;
        let config = agent.config();
        let rails = agent.rails();
        let engine = agent.context_engine();
        let (tools, tool_cards, tool_info) = {
            let abilities = agent.abilities.read();
            let cards = abilities.cards();
            let mut tools: IndexMap<String, Arc<dyn Tool>> = IndexMap::with_capacity(cards.len());
            for card in &cards {
                if let Some(tool) = abilities.get(&card.name) {
                    tools.insert(card.name.clone(), tool);
                }
            }
            (tools, cards, abilities.tool_info())
        };
        let exposure = agent.tool_exposure_policy();
        let mcp_allowlist = agent.mcp_tool_allowlist();
        let mut request_config = agent.request_config.read().clone().unwrap_or_default();
        if request_config.model_name.is_empty() {
            request_config.model_name.clone_from(&config.model_name);
        }
        let mut runtime = Self {
            card_id: agent.card.id.clone(),
            card_name: agent.card.name.clone(),
            config,
            model,
            request_config,
            rails,
            engine,
            tools,
            tool_cards,
            visible_tool_info: Vec::new(),
            visible_cards: Vec::new(),
            callbacks: agent.callbacks(),
            steering: agent.steering(),
            exposure,
            mcp_allowlist,
            client_tools: agent.client_tools(),
            resilience: agent.tool_resilience(),
            final_check: agent.final_request_check(),
            kv_cache_hook: agent.kv_cache_hook(),
            memory_provider: agent.memory_provider(),
            usage_telemetry: agent.usage_telemetry(),
        };
        let (visible_tool_info, visible_cards) = exposure::visible_tools(
            &tool_info,
            &runtime.tool_cards,
            &runtime.exposure,
            &runtime.mcp_allowlist,
        );
        runtime.visible_tool_info = visible_tool_info;
        runtime.visible_cards = visible_cards;
        Ok(runtime)
    }
}

/// One successful model call's usage reporting.
///
/// Grouping the loop-invariant parts keeps [`UsageCall::report`] a one-argument
/// call at both firing points (an iteration's streamed call and the
/// `FinalAnswer` finalization call), so neither has to rebuild the identity of
/// the call by hand.
struct UsageCall<'a> {
    /// The run's snapshot: the engine and the telemetry buffer.
    runtime: &'a AgentRuntime,
    /// The run's session, which supplies the session id.
    session: &'a Arc<dyn Session>,
    /// The stream publisher, when the caller streams.
    publisher: Option<&'a UnboundedSender<AgentChunk>>,
    /// The iteration the call belongs to.
    iteration: usize,
    /// The request the call was made with; it names the model.
    request: &'a ModelRequestConfig,
    /// The conversation window the call was measured against.
    context: &'a ConversationContext,
    /// The window fields the post-call frame repeats.
    window: frames::WindowSummary,
}

impl UsageCall<'_> {
    /// Records the provider's accounting and publishes the post-call snapshot.
    ///
    /// The KV-cache aggregator entry is written under the deterministic
    /// [`usage::request_id`] of this call, so a retried call updates it instead
    /// of double counting. The post-call snapshot is only produced when an
    /// analyzer is installed; both steps are best effort, because telemetry must
    /// never change a run's outcome: an error is logged at debug level and the
    /// frame is simply not emitted.
    ///
    /// The engine's `usage_snapshot` measures the window, so the snapshot it
    /// returns is labelled `pre_call` and carries no request id; the agent
    /// stamps both (`phase = "post_call"`, `request_id = "<session>#<iteration>"`)
    /// so a drained snapshot names the call it belongs to, exactly like the
    /// frame that carries it.
    async fn report(&self, usage: &Usage) {
        let scope = usage::cache_scope_key(
            self.session.id(),
            self.runtime.model.provider(),
            self.request,
        );
        let request_id = usage::request_id(self.session.id(), self.iteration);
        let reported = usage::provider_usage_value(usage);
        if let Err(error) = self
            .runtime
            .engine
            .record_provider_usage(scope, &request_id, &reported)
        {
            tracing::debug!(%error, "provider usage was not recorded in the KV-cache rollup");
        }
        match self
            .runtime
            .engine
            .usage_snapshot(self.context, Some(usage.input_tokens))
            .await
        {
            Ok(mut snapshot) => {
                // The engine measures a window, so its own snapshot is always
                // labelled `pre_call` and carries no request id. This call has
                // already happened, so the agent stamps the phase the frame
                // publishes and the id the KV-cache entry was recorded under.
                POST_CALL_PHASE.clone_into(&mut snapshot.phase);
                snapshot.request_id.clone_from(&request_id);
                self.runtime.usage_telemetry.record(snapshot.clone());
                frames::context_usage_post_call(
                    self.publisher,
                    self.session,
                    self.iteration,
                    usage,
                    &snapshot,
                    self.window,
                )
                .await;
            }
            Err(error) => {
                tracing::debug!(%error, "no post-call context usage snapshot was produced");
            }
        }
    }
}

/// Runs one invoke of the `ReAct` loop to completion.
///
/// `agent` supplies the model, tools, rails, and context engine; `input` is the
/// parsed caller payload (a text query or an interactive resume); `session` is
/// the resolved session the run reads state from and writes histories to;
/// `publisher`, when present, receives each [`AgentChunk`] as it is produced; and
/// `owned` records whether this run created `session` (see
/// [`AgentRun`](crate::AgentRun)), which decides whether the external memory
/// provider finalizes the conversation at the end.
///
/// # Errors
///
/// Returns an error when the invocation cannot run at all (unconfigured model,
/// invalid input) or when a rail aborts the run; recoverable model and tool
/// failures are reported inside the returned [`AgentOutput`].
pub(crate) async fn run(
    agent: &ReActAgent,
    input: AgentInput,
    session: Arc<dyn Session>,
    publisher: Option<UnboundedSender<AgentChunk>>,
    owned: bool,
) -> Result<AgentOutput> {
    let runtime = AgentRuntime::snapshot(agent)?;
    run_with(&runtime, input, session, publisher, owned).await
}

/// Runs one invoke against an already snapshotted runtime.
///
/// The wrapper owns the invocation-wide concerns: the root span, the single
/// [`RailContext`] shared by every firing point, and the unconditional
/// finalization (`AfterInvoke` rails, context persistence, the final `answer`
/// frame, the external memory provider's session end, and the span status) that
/// must happen even when the loop fails.
///
/// `owned` marks a session this run created: only then does the memory provider
/// receive [`on_session_end`](openjiuwen_memory::MemoryProvider::on_session_end),
/// and never for an interrupt pause, which ends no conversation.
///
/// # Errors
///
/// Returns the loop's error unchanged; finalization failures are logged only.
pub(crate) async fn run_with(
    runtime: &AgentRuntime,
    input: AgentInput,
    session: Arc<dyn Session>,
    publisher: Option<UnboundedSender<AgentChunk>>,
    owned: bool,
) -> Result<AgentOutput> {
    let span = session.tracer().map(|tracer| {
        let span = tracer.start_span(runtime.card_name.clone(), InvokeType::Chain);
        span.set_inputs(serde_json::to_value(&input).unwrap_or(Value::Null));
        span
    });
    let mut ctx = RailContext::new(RailEvent::BeforeInvoke)
        .with_session(Arc::clone(&session))
        .with_agent_id(runtime.card_id.clone())
        .with_input(input.clone());

    // The run's text turns, forwarded to the memory provider when this run owns
    // the session.
    let mut turns: Vec<ConversationTurn> = Vec::new();
    let result = run_inner(
        runtime,
        &input,
        &session,
        publisher.as_ref(),
        &mut ctx,
        &mut turns,
    )
    .await;

    let output = match &result {
        Ok(output) => output.clone(),
        Err(error) => AgentOutput::error(error.to_string()),
    };
    reset_ctx(&mut ctx, RailEvent::AfterInvoke);
    ctx.output = Some(output.clone());
    if let Err(error) = runtime.rails.fire(&mut ctx).await {
        tracing::warn!(%error, "AfterInvoke rail failed");
    }
    if let Err(error) = runtime.engine.save_contexts(&*session).await {
        tracing::warn!(%error, "failed to persist the conversation context");
    }
    // A successful run publishes its terminal `answer` frame here; a failed one
    // leaves that to the streaming caller, which turns the error into either a
    // terminal `answer` frame or a terminal `Err` item, so the stream never
    // carries two terminal frames.
    if result.is_ok() {
        let chunk = AgentChunk::answer(&output);
        if let Some(publisher) = publisher.as_ref() {
            let _ = publisher.send(chunk.clone());
        }
        if let Err(error) = session.write_stream(chunk.to_stream_item()).await {
            tracing::warn!(%error, "failed to write the answer frame to the session");
        }
    }
    // The invocation-end callback observes both successful and failed runs.
    hooks::invoke_end(runtime, &session, &output).await;
    // An owned session ends here, so the external memory provider finalizes the
    // conversation; a borrowed session stays the caller's to finalize. Best
    // effort: a provider failure never changes the run's output.
    finalize_session(runtime.memory_provider.as_ref(), &turns, owned, &output).await;
    if let Some(span) = span {
        match &result {
            Ok(output) => span.finish(output.to_value()),
            Err(error) => span.fail(error.to_string()),
        }
    }
    result
}

/// Clears the event-specific fields of `ctx` and switches it to `event`.
///
/// `extra`, `session`, `agent_id`, `input`, `iteration`, and `output` survive:
/// they describe the invocation, not one firing point.
fn reset_ctx(ctx: &mut RailContext, event: RailEvent) {
    ctx.event = event;
    ctx.messages.clear();
    ctx.tools.clear();
    ctx.response = None;
    ctx.tool_call = None;
    ctx.tool_result = None;
    ctx.force_finish = None;
    ctx.model_continue = false;
    ctx.retry = None;
    ctx.skip_tool = None;
    ctx.interrupts.clear();
    ctx.model_override = None;
}

/// Attaches the iteration count and observations a force-finished output lacks.
fn with_progress(
    mut output: AgentOutput,
    used: usize,
    observations: &[Observation],
) -> AgentOutput {
    if output.iterations_used.is_none() {
        output.iterations_used = Some(used);
    }
    if output.observations.is_empty() {
        output.observations = observations.to_vec();
    }
    output
}

/// Builds the model request for one call: the run's request config with the
/// prepared tool schemas attached.
fn request_for(runtime: &AgentRuntime, tools: &[Value]) -> ModelRequestConfig {
    let mut request = runtime.request_config.clone();
    if request.model_name.is_empty() {
        request.model_name.clone_from(&runtime.config.model_name);
    }
    request.with_tools(tools.to_vec())
}

/// The render fields a `BeforeInvoke` rail published under the
/// `MEMORY_VARIABLES_KEY` extra key, if any.
///
/// The loop hands them to the prompt renderer on every system-prompt build, so
/// `{{sys_long_term_memory}}`/`{{sys_memory_variables}}` resolve in the first
/// and in later iterations alike; `extra` survives `reset_ctx`.
fn memory_render_fields(ctx: &RailContext) -> Option<&serde_json::Map<String, Value>> {
    ctx.extra
        .get(MEMORY_VARIABLES_KEY)
        .and_then(Value::as_object)
}

/// Builds the system message of one iteration: the rendered identity plus the
/// external memory provider's prompt block, when a rail published one.
fn system_message(runtime: &AgentRuntime, input: &AgentInput, ctx: &RailContext) -> BaseMessage {
    build_system_with_block(
        &runtime.config,
        input,
        memory_render_fields(ctx),
        injected_prompt_block(ctx),
    )
}

/// Drives the iteration loop and turns its termination into an [`AgentOutput`].
///
/// `turns` collects the run's text turns (the user query and every assistant
/// reply, in order) for the external memory provider's session finalization.
#[allow(clippy::too_many_lines)]
async fn run_inner(
    runtime: &AgentRuntime,
    input: &AgentInput,
    session: &Arc<dyn Session>,
    publisher: Option<&UnboundedSender<AgentChunk>>,
    ctx: &mut RailContext,
    turns: &mut Vec<ConversationTurn>,
) -> Result<AgentOutput> {
    let config = &runtime.config;
    let context = runtime
        .engine
        .create_context(usage::MAIN_CONTEXT_ID, Some(Arc::clone(session)))
        .await?;

    reset_ctx(ctx, RailEvent::BeforeInvoke);
    runtime.rails.fire(ctx).await?;
    if let Some(output) = ctx.take_force_finish() {
        return Ok(output);
    }

    let tool_options = ToolExecutionOptions {
        parallel: config.parallel_tool_calls,
        session_id: Some(session.id().to_owned()),
        ..ToolExecutionOptions::default()
    };

    let (start, original_query, mut observations) = match resume_or_seed(
        runtime,
        input,
        session,
        publisher,
        &tool_options,
        ctx,
        &context,
    )
    .await?
    {
        ResumeOutcome::Start {
            start,
            original_query,
            observations,
        } => (start, original_query, observations),
        ResumeOutcome::Paused(output) => return Ok(*output),
    };

    let host_limit = input.max_iterations.unwrap_or(usize::MAX);
    let hard_limit = config.max_iterations_hard_limit.unwrap_or(usize::MAX);
    let limit = config.max_iterations.max(1).min(host_limit).min(hard_limit);
    let mut used = start.saturating_sub(1);
    let mut total_usage = Usage::default();
    let mut empty_notice_sent = false;

    turns.push(ConversationTurn::new("user", original_query.clone()));

    hooks::invoke_start(runtime, session, &original_query, limit).await;

    for iteration in start..=limit {
        used = iteration;
        // Steering pushed before the run lands in the first request; one pushed
        // mid-run lands in the next iteration's window.
        hooks::drain_steering(runtime, ctx, &context).await?;
        let system = system_message(runtime, input, ctx);
        let prepared = runtime
            .engine
            .prepare(
                &context,
                &PrepareOptions {
                    system_messages: vec![system],
                    tools: runtime.visible_tool_info.clone(),
                    ..PrepareOptions::default()
                },
            )
            .await?;
        let window = frames::WindowSummary::of(&prepared);
        frames::context_usage(publisher, session, iteration, &prepared).await;

        reset_ctx(ctx, RailEvent::BeforeModelCall);
        ctx.iteration = iteration;
        ctx.messages.clone_from(&prepared.messages);
        ctx.tools.clone_from(&runtime.visible_cards);
        runtime.rails.fire(ctx).await?;
        if let Some(output) = ctx.take_force_finish() {
            return Ok(with_progress(output, used, &observations));
        }
        // Capture the possibly rail-mutated window before the retry loop clears `ctx.messages`.
        let messages = ctx.messages.clone();
        let mut model = ctx
            .model_override
            .take()
            .unwrap_or_else(|| Arc::clone(&runtime.model));
        let request = request_for(runtime, &prepared.tools);
        hooks::step_start(
            runtime,
            session,
            iteration,
            &messages,
            &runtime.visible_cards,
        )
        .await;

        // At most two attempts: one retry a rail asks for after an exception.
        let mut attempt = 0;
        let reply = loop {
            attempt += 1;
            if !Arc::ptr_eq(&model, &runtime.model) {
                // A rail swapped the model for this call; the primary model
                // emits no route frame.
                frames::llm_route(
                    publisher,
                    session,
                    &request,
                    "backup",
                    Some("model_override"),
                )
                .await;
            }
            let mut sink = DeltaSink::new(publisher, Some(session));
            // Best-effort prefix affinity: prepared per attempt, so a retried
            // call records its own scope and never leaves a lease dangling.
            let cache = kv_cache::prepare(runtime, session, &messages, &request).await;
            match call_model_streaming(&model, &messages, request.clone(), &mut sink).await {
                Ok(reply) => {
                    kv_cache::finish(cache, true, Some(reply.usage.input_tokens)).await;
                    break reply;
                }
                Err(error) => {
                    kv_cache::finish(cache, false, None).await;
                    reset_ctx(ctx, RailEvent::OnModelException);
                    ctx.iteration = iteration;
                    ctx.extra
                        .insert("error".to_owned(), Value::String(error.to_string()));
                    runtime.rails.fire(ctx).await?;
                    if let Some(override_model) = ctx.model_override.take() {
                        model = override_model;
                    }
                    let retry = ctx.take_retry();
                    if attempt >= 2 {
                        return Err(error);
                    }
                    let Some(retry) = retry else {
                        return Err(error);
                    };
                    if let Some(delay) = retry.delay {
                        tokio::time::sleep(delay).await;
                    }
                }
            }
        };
        total_usage.add(&reply.usage);
        UsageCall {
            runtime,
            session,
            publisher,
            iteration,
            request: &request,
            context: &context,
            window,
        }
        .report(&reply.usage)
        .await;

        reset_ctx(ctx, RailEvent::AfterModelCall);
        ctx.iteration = iteration;
        ctx.response = Some(reply.clone());
        runtime.rails.fire(ctx).await?;
        if let Some(output) = ctx.take_force_finish() {
            return Ok(with_progress(output, used, &observations));
        }
        let model_continue = ctx.take_model_continue();

        runtime
            .engine
            .observe(&context, reply.message.clone())
            .await?;
        let raw = reply.message.text().into_owned();
        let calls = reply.message.tool_calls.clone();
        hooks::step_end(runtime, session, iteration, &reply, calls.len()).await;
        if !raw.trim().is_empty() {
            turns.push(ConversationTurn::new("assistant", raw.clone()));
        }

        if calls.is_empty() {
            if model_continue {
                continue;
            }
            match hooks::check_final_request(
                runtime,
                iteration,
                &raw,
                &messages,
                &runtime.visible_cards,
            )
            .await?
            {
                hooks::FinalCheckOutcome::Allow => {}
                hooks::FinalCheckOutcome::Finish(output) => {
                    return Ok(with_progress(*output, used, &observations));
                }
                hooks::FinalCheckOutcome::Reject { reason, inject } => {
                    if iteration < limit {
                        let notice = inject.unwrap_or(reason);
                        runtime
                            .engine
                            .observe(&context, BaseMessage::user(notice))
                            .await?;
                        continue;
                    }
                    // At the limit a rejection falls through to the regular
                    // termination of this round.
                }
            }
            if raw.trim().is_empty() {
                if !empty_notice_sent && iteration < limit {
                    empty_notice_sent = true;
                    let notice = BaseMessage::user(EMPTY_RESPONSE_NOTICE).with_metadata(
                        EMPTY_RESPONSE_SOURCE_KEY,
                        Value::String("empty_response".to_owned()),
                    );
                    runtime.engine.observe(&context, notice).await?;
                    continue;
                }
                let mut output = AgentOutput::error(EMPTY_RESPONSE_MSG)
                    .with_reasoning_present(reply.reasoning_present)
                    .with_iterations_used(used)
                    .with_observations(observations.clone());
                if let Some(reason) = &reply.finish_reason {
                    output.finish_reason = Some(reason.as_str().to_owned());
                }
                return Ok(output);
            }
            let mut output = AgentOutput::answer(raw)
                .with_reasoning_present(reply.reasoning_present)
                .with_iterations_used(used)
                .with_observations(observations.clone());
            if let Some(reason) = &reply.finish_reason {
                output.finish_reason = Some(reason.as_str().to_owned());
            }
            return Ok(output);
        }

        let mut results: Vec<Option<ToolCallResult>> = vec![None; calls.len()];
        let mut queued: Vec<(usize, ToolCall)> = Vec::new();
        // Requests recorded by the turn's tool-call hooks. Every call is asked
        // (in model order) before the turn pauses, so one model turn with two
        // matching calls surfaces both requests at once.
        let mut interrupted: Vec<InterruptRequest> = Vec::new();
        for (index, call) in calls.iter().enumerate() {
            reset_ctx(ctx, RailEvent::BeforeToolCall);
            ctx.iteration = iteration;
            ctx.tool_call = Some(call.clone());
            runtime.rails.fire(ctx).await?;
            if let Some(output) = ctx.take_force_finish() {
                return Ok(with_progress(output, used, &observations));
            }
            // `reset_ctx` clears `ctx.interrupts` per call, so drain now.
            let recorded = std::mem::take(&mut ctx.interrupts);
            if !recorded.is_empty() {
                interrupted.extend(recorded);
                continue;
            }
            if let Some(text) = ctx.take_skip_tool() {
                hooks::tool_call_start(runtime, session, iteration, call).await;
                let output = Value::String(text.clone());
                results[index] = Some(ToolCallResult::success(call.clone(), output, text));
                continue;
            }
            if exposure::is_client_tool(&runtime.client_tools, &runtime.tool_cards, &call.name) {
                // The connected client runs this tool: pause the turn instead
                // of executing it locally. A rail-recorded request or a
                // `skip_tool` wins because both are handled above.
                interrupted.push(InterruptRequest::for_tool_call(
                    InterruptKind::ClientTool,
                    runtime.card_id.clone(),
                    "Please run this tool and return its result",
                    call,
                ));
                continue;
            }
            queued.push((index, call.clone()));
        }
        if !interrupted.is_empty() {
            // No tool of this turn runs while a request is pending.
            let answers = BTreeMap::new();
            return pause_output(
                session,
                publisher,
                iteration,
                &original_query,
                Some(reply.message.clone()),
                &interrupted,
                &answers,
                observations,
            )
            .await;
        }
        if !queued.is_empty() {
            let calls_to_run = queued
                .iter()
                .map(|(_, call)| call.clone())
                .collect::<Vec<_>>();
            for call in &calls_to_run {
                hooks::tool_call_start(runtime, session, iteration, call).await;
            }
            let executed = execute_calls_with(
                &runtime.tools,
                &calls_to_run,
                &tool_options,
                &runtime.resilience,
            )
            .await;
            for ((index, _), result) in queued.into_iter().zip(executed) {
                results[index] = Some(result);
            }
        }

        for (index, call) in calls.iter().enumerate() {
            let Some(mut result) = results[index].take() else {
                continue;
            };
            reset_ctx(ctx, RailEvent::AfterToolCall);
            ctx.iteration = iteration;
            ctx.tool_call = Some(call.clone());
            ctx.tool_result = Some(result.clone());
            runtime.rails.fire(ctx).await?;
            if let Some(output) = ctx.take_force_finish() {
                return Ok(with_progress(output, used, &observations));
            }
            if let Some(text) = ctx.take_skip_tool() {
                result.output = Some(Value::String(text.clone()));
                result.text = text;
                result.success = true;
                // The substituted text has no multimodal payload behind it.
                result.parts.clear();
            }
            hooks::tool_call_end(runtime, session, iteration, &result).await;
            runtime.engine.observe(&context, result.message()).await?;
            observations.push(Observation {
                content: result.text,
                name: Some(call.name.clone()),
                tool_call_id: call.id.clone(),
                parts: result.parts,
            });
        }

        if let Some(budget) = config.max_total_tokens {
            let total = total_usage.total_tokens.max(
                total_usage
                    .input_tokens
                    .saturating_add(total_usage.output_tokens),
            );
            if total > u64::try_from(budget).unwrap_or(u64::MAX) {
                return Ok(AgentOutput::error(TOKEN_BUDGET_MSG)
                    .with_termination_reason("token_budget")
                    .with_iterations_used(used)
                    .with_observations(observations.clone()));
            }
        }

        reset_ctx(ctx, RailEvent::AfterReactIteration);
        ctx.iteration = iteration;
        runtime.rails.fire(ctx).await?;
        if let Some(output) = ctx.take_force_finish() {
            return Ok(with_progress(output, used, &observations));
        }
    }

    let mut output = AgentOutput::error(MAX_ITERATIONS_MSG)
        .with_termination_reason("max_iterations")
        .with_iterations_used(used)
        .with_observations(observations.clone());
    output.truncated_by_max_iterations = true;
    if config.on_max_iterations != MaxIterationsPolicy::FinalAnswer {
        return Ok(output);
    }

    let system = system_message(runtime, input, ctx);
    let prepared = runtime
        .engine
        .prepare(
            &context,
            &PrepareOptions {
                system_messages: vec![system],
                ..PrepareOptions::default()
            },
        )
        .await?;
    let window = frames::WindowSummary::of(&prepared);
    let mut messages = prepared.messages;
    messages.push(BaseMessage::user(FINAL_ANSWER_NUDGE));
    let request = request_for(runtime, &[]);
    let cache = kv_cache::prepare(runtime, session, &messages, &request).await;
    match call_model(&runtime.model, &messages, request.clone()).await {
        Ok(reply) => {
            kv_cache::finish(cache, true, Some(reply.usage.input_tokens)).await;
            UsageCall {
                runtime,
                session,
                publisher,
                iteration: used,
                request: &request,
                context: &context,
                window,
            }
            .report(&reply.usage)
            .await;
            let text = reply.message.text().into_owned();
            if !text.trim().is_empty() && reply.message.tool_calls.is_empty() {
                turns.push(ConversationTurn::new("assistant", text.clone()));
                let mut answer = AgentOutput::answer(text)
                    .with_reasoning_present(reply.reasoning_present)
                    .with_iterations_used(used)
                    .with_observations(observations);
                if let Some(reason) = &reply.finish_reason {
                    answer.finish_reason = Some(reason.as_str().to_owned());
                }
                return Ok(answer);
            }
            output.finalization_error = Some("model did not produce a final answer".to_owned());
            Ok(output)
        }
        Err(error) => {
            kv_cache::finish(cache, false, None).await;
            output.finalization_error = Some(error.to_string());
            Ok(output)
        }
    }
}

#[cfg(test)]
mod tests;
