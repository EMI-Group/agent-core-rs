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
//!   and fire `AfterModelCall`;
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
use crate::abilities::execute::execute_calls_with;
use crate::abilities::{ToolCallResult, ToolExecutionOptions};
use crate::callbacks::CallbackFramework;
use crate::input::AgentInput;
use crate::interrupt::{InterruptKind, InterruptRequest};
use crate::output::{AgentChunk, AgentOutput, Observation};
use crate::policy::{
    ClientToolRegistry, FinalRequestCheck, McpToolAllowlist, ToolExposurePolicy,
    ToolResiliencePolicy,
};
use crate::rails::{RailContext, RailEvent, RailRegistry};
use crate::react::model_step::{call_model, call_model_streaming, DeltaSink};
use crate::react::pause::pause_output;
use crate::react::prompt::build_system;
use crate::react::resume::{resume_or_seed, ResumeOutcome};
use crate::react::{exposure, frames, hooks};
use crate::react::{MaxIterationsPolicy, ReActAgent, ReActAgentConfig};
use crate::steering::SteeringQueue;
use indexmap::IndexMap;
use openjiuwen_common::error::Result;
use openjiuwen_context::{ContextEngine, PrepareOptions};
use openjiuwen_foundation::{
    BaseMessage, Model, ModelRequestConfig, Tool, ToolCall, ToolCard, Usage,
};
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

/// Runs one invoke of the `ReAct` loop to completion.
///
/// `agent` supplies the model, tools, rails, and context engine; `input` is the
/// parsed caller payload (a text query or an interactive resume); `session` is
/// the resolved session the run reads state from and writes histories to; and
/// `publisher`, when present, receives each [`AgentChunk`] as it is produced.
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
) -> Result<AgentOutput> {
    let runtime = AgentRuntime::snapshot(agent)?;
    run_with(&runtime, input, session, publisher).await
}

/// Runs one invoke against an already snapshotted runtime.
///
/// The wrapper owns the invocation-wide concerns: the root span, the single
/// [`RailContext`] shared by every firing point, and the unconditional
/// finalization (`AfterInvoke` rails, context persistence, the final `answer`
/// frame, and the span status) that must happen even when the loop fails.
///
/// # Errors
///
/// Returns the loop's error unchanged; finalization failures are logged only.
pub(crate) async fn run_with(
    runtime: &AgentRuntime,
    input: AgentInput,
    session: Arc<dyn Session>,
    publisher: Option<UnboundedSender<AgentChunk>>,
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

    let result = run_inner(runtime, &input, &session, publisher.as_ref(), &mut ctx).await;

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

/// Drives the iteration loop and turns its termination into an [`AgentOutput`].
#[allow(clippy::too_many_lines)]
async fn run_inner(
    runtime: &AgentRuntime,
    input: &AgentInput,
    session: &Arc<dyn Session>,
    publisher: Option<&UnboundedSender<AgentChunk>>,
    ctx: &mut RailContext,
) -> Result<AgentOutput> {
    let config = &runtime.config;
    let context = runtime
        .engine
        .create_context("main", Some(Arc::clone(session)))
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

    hooks::invoke_start(runtime, session, &original_query, limit).await;

    for iteration in start..=limit {
        used = iteration;
        // Steering pushed before the run lands in the first request; one pushed
        // mid-run lands in the next iteration's window.
        hooks::drain_steering(runtime, &context).await?;
        let system = build_system(config, input);
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
            match call_model_streaming(&model, &messages, request.clone(), &mut sink).await {
                Ok(reply) => break reply,
                Err(error) => {
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
            }
            hooks::tool_call_end(runtime, session, iteration, &result).await;
            runtime.engine.observe(&context, result.message()).await?;
            observations.push(Observation {
                content: result.text,
                name: Some(call.name.clone()),
                tool_call_id: call.id.clone(),
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

    let system = build_system(config, input);
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
    let mut messages = prepared.messages;
    messages.push(BaseMessage::user(FINAL_ANSWER_NUDGE));
    match call_model(&runtime.model, &messages, request_for(runtime, &[])).await {
        Ok(reply) => {
            let text = reply.message.text().into_owned();
            if !text.trim().is_empty() && reply.message.tool_calls.is_empty() {
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
            output.finalization_error = Some(error.to_string());
            Ok(output)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::Agent;
    use crate::card::AgentCard;
    use crate::interrupt::InterruptionState;
    use crate::output::ResultType;
    use crate::rails::{AskUserRail, ConfirmInterruptRail};
    use openjiuwen_context::ContextConfig;
    use openjiuwen_foundation::ToolContext;
    use openjiuwen_session::AgentSession;
    use openjiuwen_test_support::{MockCompletion, ScriptedModel};

    #[test]
    fn request_for_inherits_the_configured_model_name() {
        let runtime_request = ModelRequestConfig::default();
        let mut config = ReActAgentConfig::new();
        config.model_name = "gpt-4o".to_owned();
        let runtime = AgentRuntime {
            card_id: "agent".to_owned(),
            card_name: "Agent".to_owned(),
            config,
            model: Arc::new(openjiuwen_test_support::ScriptedModel::text("x")),
            request_config: runtime_request,
            rails: Arc::new(RailRegistry::new()),
            engine: Arc::new(ContextEngine::new(ContextConfig::default())),
            tools: IndexMap::new(),
            tool_cards: Vec::new(),
            visible_tool_info: Vec::new(),
            visible_cards: Vec::new(),
            callbacks: Arc::new(crate::callbacks::CallbackFramework::new()),
            steering: crate::steering::SteeringQueue::new(),
            exposure: crate::policy::ToolExposurePolicy::allow_all(),
            mcp_allowlist: crate::policy::McpToolAllowlist::new(),
            client_tools: Arc::new(crate::policy::ClientToolRegistry::new()),
            resilience: crate::policy::ToolResiliencePolicy::new(),
            final_check: None,
        };
        assert_eq!(request_for(&runtime, &[]).model_name, "gpt-4o");
    }

    #[test]
    fn reset_ctx_clears_event_fields_but_keeps_the_invocation_ones() {
        let mut ctx = RailContext::new(RailEvent::BeforeInvoke)
            .with_agent_id("agent")
            .with_input(AgentInput::from_text("hi"))
            .with_iteration(3);
        ctx.extra.insert("kept".to_owned(), Value::Bool(true));
        ctx.messages.push(BaseMessage::user("m"));
        ctx.response = Some(crate::react::model_step::ModelReply::default());
        ctx.request_retry(None);
        ctx.skip_tool_with("skip");

        reset_ctx(&mut ctx, RailEvent::AfterModelCall);

        assert_eq!(ctx.event, RailEvent::AfterModelCall);
        assert_eq!(ctx.agent_id, "agent");
        assert_eq!(ctx.iteration, 3);
        assert!(ctx.extra.contains_key("kept"));
        assert_eq!(ctx.messages.len(), 0);
        assert!(ctx.response.is_none());
        assert!(ctx.retry.is_none());
        assert!(ctx.skip_tool.is_none());
    }

    #[test]
    fn with_progress_keeps_existing_values() {
        let observations = vec![Observation {
            content: "out".to_owned(),
            ..Observation::default()
        }];
        let output = with_progress(AgentOutput::answer("done"), 2, &observations);
        assert_eq!(output.iterations_used, Some(2));
        assert_eq!(output.observations, observations);
        let output = with_progress(
            AgentOutput::answer("done").with_iterations_used(9),
            2,
            &observations,
        );
        assert_eq!(output.iterations_used, Some(9));
    }

    /// Tool that echoes its arguments, used by the loop tests below.
    struct Echo(ToolCard);

    #[async_trait::async_trait]
    impl Tool for Echo {
        fn card(&self) -> &ToolCard {
            &self.0
        }

        async fn invoke(&self, inputs: Value, _ctx: &ToolContext) -> Result<Value> {
            Ok(serde_json::json!({"echoed": inputs}))
        }
    }

    fn agent_with(model: ScriptedModel) -> ReActAgent {
        let agent = ReActAgent::new(AgentCard::new("react"));
        let model: Arc<dyn Model> = Arc::new(model);
        agent.set_model(model);
        agent.add_tool(Arc::new(Echo(ToolCard::new(
            "echo",
            "echoes its arguments",
            serde_json::json!({"type": "object"}),
        ))));
        agent
    }

    #[tokio::test]
    async fn tool_calls_execute_and_feed_the_answer() {
        let model = ScriptedModel::builder()
            .completion(MockCompletion::tool_call(
                "echo",
                serde_json::json!({"x": 1}),
            ))
            .respond_text("done")
            .build();
        let agent = agent_with(model);
        let output = agent
            .invoke(Value::String("hi".to_owned()), None)
            .await
            .unwrap();
        assert_eq!(output.result_type, ResultType::Answer);
        assert_eq!(output.output, "done");
        assert_eq!(output.iterations_used, Some(2));
        assert_eq!(output.observations.len(), 1);
        assert_eq!(output.observations[0].name.as_deref(), Some("echo"));
        assert_eq!(
            output.observations[0].tool_call_id.as_deref(),
            Some("call_echo")
        );
    }

    #[tokio::test]
    async fn empty_response_is_retried_once() {
        let model = ScriptedModel::builder()
            .respond_text("")
            .respond_text("second attempt")
            .build();
        let agent = agent_with(model);
        let output = agent
            .invoke(Value::String("hi".to_owned()), None)
            .await
            .unwrap();
        assert_eq!(output.result_type, ResultType::Answer);
        assert_eq!(output.output, "second attempt");
        assert_eq!(output.iterations_used, Some(2));
    }

    #[tokio::test]
    async fn iteration_limit_reports_max_iterations() {
        let model = ScriptedModel::builder()
            .completion(MockCompletion::tool_call("echo", serde_json::json!({})))
            .completion(MockCompletion::tool_call("echo", serde_json::json!({})))
            .build();
        let agent = agent_with(model);
        agent.configure(ReActAgentConfig {
            max_iterations: 2,
            ..ReActAgentConfig::new()
        });
        let output = agent
            .invoke(Value::String("hi".to_owned()), None)
            .await
            .unwrap();
        assert_eq!(output.result_type, ResultType::Error);
        assert_eq!(output.output, MAX_ITERATIONS_MSG);
        assert_eq!(output.termination_reason.as_deref(), Some("max_iterations"));
        assert!(output.truncated_by_max_iterations);
        assert_eq!(output.iterations_used, Some(2));
        assert_eq!(output.observations.len(), 2);
    }

    #[tokio::test]
    async fn final_answer_policy_makes_one_tool_free_call() {
        let model = ScriptedModel::builder()
            .completion(MockCompletion::tool_call("echo", serde_json::json!({})))
            .respond_text("final answer")
            .build();
        let agent = agent_with(model.clone());
        agent.configure(ReActAgentConfig {
            max_iterations: 1,
            on_max_iterations: MaxIterationsPolicy::FinalAnswer,
            ..ReActAgentConfig::new()
        });
        let output = agent
            .invoke(Value::String("hi".to_owned()), None)
            .await
            .unwrap();
        assert_eq!(output.result_type, ResultType::Answer);
        assert_eq!(output.output, "final answer");
        assert!(output.finalization_error.is_none());
        let last = model
            .last_call()
            .expect("the finalization call is recorded");
        assert!(last.request.tools.is_none());
    }

    #[tokio::test]
    async fn confirm_interrupt_pauses_and_resumes() {
        let model = ScriptedModel::builder()
            .completion(MockCompletion::tool_call(
                "echo",
                serde_json::json!({"x": 1}),
            ))
            .respond_text("approved")
            .build();
        let agent = agent_with(model);
        agent.register_rail(Arc::new(ConfirmInterruptRail::new()));
        let session: Arc<dyn Session> = Arc::new(AgentSession::new("s1"));

        let paused = agent
            .invoke(
                Value::String("do it".to_owned()),
                Some(Arc::clone(&session)),
            )
            .await
            .unwrap();
        assert!(paused.is_interrupt());
        assert_eq!(paused.interrupt_ids.len(), 1);
        assert_eq!(paused.state.len(), 1);
        let interrupt_id = paused.interrupt_ids[0].clone();

        let resumed = agent
            .invoke(
                serde_json::json!({
                    "query": {"interrupt_answers": {interrupt_id.clone(): {"approved": true}}},
                }),
                Some(Arc::clone(&session)),
            )
            .await
            .unwrap();
        assert_eq!(resumed.result_type, ResultType::Answer);
        assert_eq!(resumed.output, "approved");
        assert_eq!(resumed.iterations_used, Some(2));
        assert_eq!(resumed.observations.len(), 1);
        assert_eq!(
            InterruptionState::load(&*session).await.unwrap(),
            None,
            "a resumed turn clears the persisted interruption"
        );
    }

    #[tokio::test]
    async fn question_rail_answers_become_the_tool_result() {
        let model = ScriptedModel::builder()
            .completion(MockCompletion::tool_call("echo", serde_json::json!({})))
            .respond_text("settled")
            .build();
        let agent = agent_with(model);
        agent.register_rail(Arc::new(AskUserRail::new()));
        let session: Arc<dyn Session> = Arc::new(AgentSession::new("s2"));

        let paused = agent
            .invoke(Value::String("ask".to_owned()), Some(Arc::clone(&session)))
            .await
            .unwrap();
        assert!(paused.is_interrupt());
        let resume = serde_json::json!({
            "query": {"user_inputs": {"call_echo": "Paris"}},
        });
        let resumed = agent
            .invoke(resume, Some(Arc::clone(&session)))
            .await
            .unwrap();
        assert_eq!(resumed.result_type, ResultType::Answer);
        assert_eq!(resumed.output, "settled");
        assert_eq!(resumed.observations[0].content, "Paris");
    }
}
