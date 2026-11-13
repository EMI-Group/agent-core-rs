//! Loop hooks: lifecycle callbacks, steering drain, and the final-request check.
//!
//! These are the side seams of one `ReAct` iteration. They never decide the
//! loop's control flow on their own: [`drain_steering`] appends user messages,
//! the callback helpers observe events, and [`check_final_request`] answers what
//! the loop should do with a pending final request.
//!
//! ## Callback errors never abort the run
//!
//! A callback is host code, and Python's callback manager only logs a failing
//! handler. Every helper here therefore fires the matching
//! [`AgentCallbackEvent`](crate::callbacks::AgentCallbackEvent) when (and only
//! when) the registry has hooks for it, and turns an error into a
//! `tracing::warn!` instead of an `Err`: a broken observer must not fail a
//! model/tool run. The payload is not built at all when no callback subscribed.
//!
//! ## Steering is rail-free
//!
//! The loop drains the steering backlog at the top of every iteration, so a
//! message pushed before the run lands in the first request and one pushed
//! mid-run lands in the next iteration's window. No rail event is fired for
//! steering: [`RailEvent`](crate::rails::RailEvent) has no `BeforeSteeringDrain`
//! variant, and the read-only `openjiuwen-harness` crate matches that enum
//! exhaustively, so adding one would break it. The whole backlog is drained as
//! one batch of `user` messages.
use crate::abilities::ToolCallResult;
use crate::callbacks::{
    AgentCallbackEvent, CallbackPayload, InvokeEndPayload, InvokeStartPayload, StepEndPayload,
    StepStartPayload, ToolCallEndPayload, ToolCallPayload,
};
use crate::output::AgentOutput;
use crate::policy::{FinalRequestContext, FinalRequestDecision};
use crate::react::agent_loop::AgentRuntime;
use crate::react::model_step::ModelReply;
use openjiuwen_common::error::Result;
use openjiuwen_context::ConversationContext;
use openjiuwen_foundation::{BaseMessage, ToolCall, ToolCard};
use openjiuwen_session::Session;
use std::sync::Arc;

/// Fires `payload`'s hook, logging (never propagating) a callback failure.
async fn fire(runtime: &AgentRuntime, event: AgentCallbackEvent, payload: &CallbackPayload) {
    if let Err(error) = runtime.callbacks.invoke(payload).await {
        tracing::warn!(%error, event = event.as_str(), "agent callback failed");
    }
}

/// Fires [`AgentCallbackEvent::InvokeStart`] once per run.
pub(crate) async fn invoke_start(
    runtime: &AgentRuntime,
    session: &Arc<dyn Session>,
    query: &str,
    max_iterations: usize,
) {
    let event = AgentCallbackEvent::InvokeStart;
    if !runtime.callbacks.has_hooks(event) {
        return;
    }
    let payload = CallbackPayload::InvokeStart(InvokeStartPayload {
        agent_id: runtime.card_id.clone(),
        session_id: Some(session.id().to_owned()),
        query: query.to_owned(),
        max_iterations,
    });
    fire(runtime, event, &payload).await;
}

/// Fires [`AgentCallbackEvent::StepStart`] for one iteration's model request.
pub(crate) async fn step_start(
    runtime: &AgentRuntime,
    session: &Arc<dyn Session>,
    iteration: usize,
    messages: &[BaseMessage],
    tools: &[ToolCard],
) {
    let event = AgentCallbackEvent::StepStart;
    if !runtime.callbacks.has_hooks(event) {
        return;
    }
    let payload = CallbackPayload::StepStart(StepStartPayload {
        agent_id: runtime.card_id.clone(),
        session_id: Some(session.id().to_owned()),
        iteration,
        messages: messages.to_vec(),
        tools: tools.to_vec(),
    });
    fire(runtime, event, &payload).await;
}

/// Fires [`AgentCallbackEvent::StepEnd`] after one iteration's model reply.
pub(crate) async fn step_end(
    runtime: &AgentRuntime,
    session: &Arc<dyn Session>,
    iteration: usize,
    reply: &ModelReply,
    tool_call_count: usize,
) {
    let event = AgentCallbackEvent::StepEnd;
    if !runtime.callbacks.has_hooks(event) {
        return;
    }
    let payload = CallbackPayload::StepEnd(StepEndPayload {
        agent_id: runtime.card_id.clone(),
        session_id: Some(session.id().to_owned()),
        iteration,
        output_text: reply.message.text().into_owned(),
        tool_call_count,
        usage: reply.usage.clone(),
        finish_reason: reply
            .finish_reason
            .as_ref()
            .map(|reason| reason.as_str().to_owned()),
    });
    fire(runtime, event, &payload).await;
}

/// Fires [`AgentCallbackEvent::ToolCallStart`] for a call about to execute.
pub(crate) async fn tool_call_start(
    runtime: &AgentRuntime,
    session: &Arc<dyn Session>,
    iteration: usize,
    call: &ToolCall,
) {
    let event = AgentCallbackEvent::ToolCallStart;
    if !runtime.callbacks.has_hooks(event) {
        return;
    }
    let payload = CallbackPayload::ToolCallStart(ToolCallPayload {
        agent_id: runtime.card_id.clone(),
        session_id: Some(session.id().to_owned()),
        iteration,
        call: call.clone(),
    });
    fire(runtime, event, &payload).await;
}

/// Fires [`AgentCallbackEvent::ToolCallEnd`] for one produced tool result.
pub(crate) async fn tool_call_end(
    runtime: &AgentRuntime,
    session: &Arc<dyn Session>,
    iteration: usize,
    result: &ToolCallResult,
) {
    let event = AgentCallbackEvent::ToolCallEnd;
    if !runtime.callbacks.has_hooks(event) {
        return;
    }
    let payload = CallbackPayload::ToolCallEnd(ToolCallEndPayload {
        agent_id: runtime.card_id.clone(),
        session_id: Some(session.id().to_owned()),
        iteration,
        call: result.call.clone(),
        text: result.text.clone(),
        success: result.success,
        error: result.error.clone(),
    });
    fire(runtime, event, &payload).await;
}

/// Fires [`AgentCallbackEvent::InvokeEnd`] with the terminal output.
pub(crate) async fn invoke_end(
    runtime: &AgentRuntime,
    session: &Arc<dyn Session>,
    output: &AgentOutput,
) {
    let event = AgentCallbackEvent::InvokeEnd;
    if !runtime.callbacks.has_hooks(event) {
        return;
    }
    let payload = CallbackPayload::InvokeEnd(InvokeEndPayload {
        agent_id: runtime.card_id.clone(),
        session_id: Some(session.id().to_owned()),
        output: output.clone(),
    });
    fire(runtime, event, &payload).await;
}

/// Drains the steering backlog into the conversation window.
///
/// The whole backlog is appended as one batch of `user` messages, so the next
/// model request of the run observes every message pushed since the previous
/// iteration. See the module docs for why no rail event is fired.
///
/// # Errors
///
/// Returns the context engine's error when a message cannot be observed.
pub(crate) async fn drain_steering(
    runtime: &AgentRuntime,
    context: &Arc<ConversationContext>,
) -> Result<()> {
    for message in runtime.steering.drain_messages() {
        runtime.engine.observe(context, message).await?;
    }
    Ok(())
}

/// What the loop does with a pending final model request.
#[derive(Debug)]
pub(crate) enum FinalCheckOutcome {
    /// Let the model call go ahead.
    Allow,
    /// Do not call the model; append `inject` (else the reason) and iterate.
    Reject {
        /// Human-readable reason, used when no notice is injected.
        reason: String,
        /// Notice the loop appends as a user message.
        inject: Option<String>,
    },
    /// Return this output instead of the model's answer.
    Finish(Box<AgentOutput>),
}

/// Runs the configured final-request check, if any.
///
/// No configured check allows the request, which keeps the loop's behaviour
/// identical to a run without the seam. A configured check receives the pending
/// request and decides whether to allow it, reject it (with an optional notice
/// the loop injects and iterates on), or finish the round with its own output.
///
/// # Errors
///
/// Returns the check's own error, which aborts the run.
pub(crate) async fn check_final_request(
    runtime: &AgentRuntime,
    iteration: usize,
    text: &str,
    messages: &[BaseMessage],
    tools: &[ToolCard],
) -> Result<FinalCheckOutcome> {
    let Some(check) = runtime.final_check.as_ref() else {
        return Ok(FinalCheckOutcome::Allow);
    };
    let context = FinalRequestContext::new(&runtime.card_id, iteration, text, messages, tools);
    Ok(match check.check(&context).await? {
        FinalRequestDecision::Allow => FinalCheckOutcome::Allow,
        FinalRequestDecision::Reject { reason, inject } => {
            FinalCheckOutcome::Reject { reason, inject }
        }
        FinalRequestDecision::Finish(output) => FinalCheckOutcome::Finish(output),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::callbacks::{AgentCallback, CallbackFramework};
    use crate::policy::ClientToolRegistry;
    use crate::rails::RailRegistry;
    use crate::steering::SteeringQueue;
    use openjiuwen_common::error::Error;
    use openjiuwen_context::{ContextConfig, ContextEngine};
    use openjiuwen_foundation::Model;
    use openjiuwen_session::AgentSession;
    use openjiuwen_test_support::ScriptedModel;
    use serde_json::Value;

    fn runtime(callbacks: Arc<CallbackFramework>) -> AgentRuntime {
        let model: Arc<dyn Model> = Arc::new(ScriptedModel::text("x"));
        AgentRuntime {
            card_id: "agent".to_owned(),
            card_name: "Agent".to_owned(),
            config: crate::react::ReActAgentConfig::new(),
            model,
            request_config: openjiuwen_foundation::ModelRequestConfig::default(),
            rails: Arc::new(RailRegistry::new()),
            engine: Arc::new(ContextEngine::new(ContextConfig::default())),
            tools: indexmap::IndexMap::new(),
            tool_cards: Vec::new(),
            callbacks,
            steering: SteeringQueue::new(),
            exposure: crate::policy::ToolExposurePolicy::allow_all(),
            mcp_allowlist: crate::policy::McpToolAllowlist::new(),
            client_tools: Arc::new(ClientToolRegistry::new()),
            resilience: crate::policy::ToolResiliencePolicy::new(),
            final_check: None,
            visible_tool_info: Vec::new(),
            visible_cards: Vec::new(),
        }
    }

    /// A callback whose hooks always fail.
    struct Failing;

    #[async_trait::async_trait]
    impl AgentCallback for Failing {
        fn name(&self) -> &'static str {
            "failing"
        }

        async fn on_invoke_start(&self, _payload: &InvokeStartPayload) -> Result<()> {
            Err(Error::validation("boom"))
        }

        async fn on_invoke_end(&self, _payload: &InvokeEndPayload) -> Result<()> {
            Err(Error::validation("boom"))
        }
    }

    #[tokio::test]
    async fn a_failing_callback_does_not_fail_the_hook() {
        let mut framework = CallbackFramework::new();
        framework.register(AgentCallbackEvent::InvokeStart, Arc::new(Failing));
        framework.register(AgentCallbackEvent::InvokeEnd, Arc::new(Failing));
        let runtime = runtime(Arc::new(framework));
        let session: Arc<dyn Session> = Arc::new(AgentSession::new("s"));
        let output = AgentOutput::answer("done");

        // Returning normally is the contract: callback errors are logged only.
        invoke_start(&runtime, &session, "hi", 5).await;
        invoke_end(&runtime, &session, &output).await;
    }

    #[tokio::test]
    async fn an_empty_registry_skips_every_hook() {
        let runtime = runtime(Arc::new(CallbackFramework::new()));
        let session: Arc<dyn Session> = Arc::new(AgentSession::new("s"));
        let reply = ModelReply::default();
        invoke_start(&runtime, &session, "hi", 5).await;
        step_start(&runtime, &session, 1, &[], &[]).await;
        step_end(&runtime, &session, 1, &reply, 0).await;
        tool_call_start(&runtime, &session, 1, &ToolCall::new("echo")).await;
        let result =
            ToolCallResult::success(ToolCall::new("echo"), Value::Null, "no output".to_owned());
        tool_call_end(&runtime, &session, 1, &result).await;
        invoke_end(&runtime, &session, &AgentOutput::answer("done")).await;
    }

    #[tokio::test]
    async fn no_configured_check_allows_the_request() {
        let runtime = runtime(Arc::new(CallbackFramework::new()));
        let outcome = check_final_request(&runtime, 1, "answer", &[], &[])
            .await
            .expect("no check");
        assert!(matches!(outcome, FinalCheckOutcome::Allow));
    }
}
