//! Resume of an interrupted `ReAct` turn and seeding of a fresh one.
//!
//! [`resume_or_seed`] is the loop's entry point: it replays a persisted
//! interruption when the session has one, re-pausing the run when the caller
//! left requests unanswered, and otherwise observes the user query and starts a
//! fresh turn. Both branches seed the host's auto-confirm decision into the rail
//! scratch space.
//!
//! A turn paused on an interrupt persists its pending requests; resuming matches
//! the caller's [`InteractiveInput`] answers against them with
//! [`match_answers`](crate::interrupt::match_answers) (tool-call id first, then
//! interrupt id) and turns each answer into a tool result:
//!
//! - a client tool (`client_tool` kind, or a name the
//!   [`ClientToolRegistry`] knows) hands the answer to its registered handler and
//!   renders the handler's value, or the handler's error as a failed result;
//! - an approved server tool is re-executed with the run's resilience budget;
//! - a rejected call yields the rejection-feedback tool message;
//! - an ask-user answer becomes the tool content.
use crate::abilities::execute::execute_calls_with;
use crate::abilities::{ToolCallResult, ToolExecutionOptions};
use crate::input::AgentInput;
use crate::interrupt::{
    answer_text, match_answers, parse_approval, reject_feedback, InterruptKind, InterruptRequest,
    InterruptionState, AUTO_CONFIRM_STATE_KEY,
};
use crate::output::{AgentChunk, AgentOutput, Observation};
use crate::policy::{ClientToolRegistry, ToolResiliencePolicy};
use crate::rails::RailContext;
use crate::react::agent_loop::AgentRuntime;
use crate::react::pause::pause_output;
use indexmap::IndexMap;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_context::ConversationContext;
use openjiuwen_foundation::{BaseMessage, Tool, ToolCall};
use openjiuwen_session::{InteractiveInput, Session};
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::mpsc::UnboundedSender;

/// `RailContext::extra` key carrying the host's auto-confirm decision.
const AUTO_CONFIRM_EXTRA_KEY: &str = "_interrupt_auto_confirm";
/// `RailContext::extra` key carrying the answers matched during a resume.
const RESUME_ANSWERS_EXTRA_KEY: &str = "_resume_interrupt_answers";

/// Iteration a resumed run restarts from after replaying the interrupted calls.
///
/// Kept as a documented starting point only: the loop computes
/// `state.iteration + 1` itself and does not read this constant.
#[allow(dead_code)] // used by the loop's resume path
pub(crate) const RESUME_START_ITERATION: usize = 1;

/// How a run starts: a fresh turn or a resumed interruption.
pub(crate) enum ResumeOutcome {
    /// Start the iteration loop at `start` with `original_query` in hand.
    Start {
        /// First iteration to run.
        start: usize,
        /// The user query that started the turn.
        original_query: String,
        /// Observations replayed from the interrupted tool calls.
        observations: Vec<Observation>,
    },
    /// The interruption is still unanswered; the run pauses again.
    Paused(Box<AgentOutput>),
}

/// Seeds the conversation: replays a pending interruption or observes the query.
///
/// A resumed turn clears the persisted state, seeds the rail scratch space with
/// the auto-confirm decision and the matched answers, replays the interrupted
/// tool calls, and starts at the iteration after the interruption. Requests the
/// caller did not answer pause the run again. A fresh turn also seeds the
/// auto-confirm decision, so a host can pre-approve tools before the first call.
pub(crate) async fn resume_or_seed(
    runtime: &AgentRuntime,
    input: &AgentInput,
    session: &Arc<dyn Session>,
    publisher: Option<&UnboundedSender<AgentChunk>>,
    tool_options: &ToolExecutionOptions,
    ctx: &mut RailContext,
    context: &Arc<ConversationContext>,
) -> Result<ResumeOutcome> {
    let Some(state) = InterruptionState::load(&**session).await? else {
        let Some(text) = input.query_text() else {
            return Err(Error::validation(
                "interactive input requires a pending interruption to resume",
            ));
        };
        seed_auto_confirm(session, ctx).await?;
        runtime
            .engine
            .observe(context, BaseMessage::user(text.to_owned()))
            .await?;
        return Ok(ResumeOutcome::Start {
            start: 1,
            original_query: text.to_owned(),
            observations: Vec::new(),
        });
    };

    InterruptionState::clear(&**session).await?;
    seed_auto_confirm(session, ctx).await?;
    let interactive = input.interactive().cloned().unwrap_or_default();
    let pending = state
        .requests
        .iter()
        .map(|(call_id, request)| (call_id.clone(), request.interrupt_id.clone()))
        .collect::<Vec<_>>();
    let matched = match_answers(&interactive, &pending);
    ctx.extra.insert(
        RESUME_ANSWERS_EXTRA_KEY.to_owned(),
        serde_json::to_value(&matched.answers)?,
    );

    let (results, unanswered) = replay_interrupted(
        &state,
        &interactive,
        &runtime.tools,
        tool_options,
        &runtime.resilience,
        &runtime.client_tools,
    )
    .await?;
    let mut observations = Vec::with_capacity(results.len());
    for result in results {
        runtime.engine.observe(context, result.message()).await?;
        observations.push(Observation {
            content: result.text,
            name: Some(result.call.name),
            tool_call_id: result.call.id,
            parts: result.parts,
        });
    }
    if !unanswered.is_empty() {
        let paused = pause_output(
            session,
            publisher,
            state.iteration,
            &state.original_query,
            state.assistant_message,
            &unanswered,
            &matched.answers,
            observations,
        )
        .await?;
        return Ok(ResumeOutcome::Paused(Box::new(paused)));
    }
    Ok(ResumeOutcome::Start {
        start: state.iteration + 1,
        original_query: state.original_query,
        observations,
    })
}

/// Seeds `ctx` with the host's auto-confirm decision, when the session has one.
///
/// Python's `BaseInterruptRail.before_tool_call` reads the decision from the
/// session on every turn, so both the fresh and the resumed branch seed it.
async fn seed_auto_confirm(session: &Arc<dyn Session>, ctx: &mut RailContext) -> Result<()> {
    if let Some(value) = session.get_state(AUTO_CONFIRM_STATE_KEY).await? {
        if !value.is_null() {
            ctx.extra.insert(AUTO_CONFIRM_EXTRA_KEY.to_owned(), value);
        }
    }
    Ok(())
}

/// Replays the interrupted tool calls of `state` using `interactive` answers.
///
/// Each entry of [`InterruptionState::requests`] is matched to an answer:
///
/// - a client tool (kind [`InterruptKind::ClientTool`], or a name registered in
///   `client_tools`) is completed through the handler: a value becomes a
///   successful result whose text is the value's string (a `Value::String`) or
///   its JSON rendering, and a handler error becomes a failed result;
/// - an approval (`parse_approval` is `Some(true)`) re-executes exactly that
///   call through `execute_calls_with` using `tools` and `resilience`;
/// - a rejection (`Some(false)`) becomes a failed result carrying
///   [`reject_feedback`] as its text;
/// - any other answer (an ask-user reply) becomes a successful result whose
///   output and text are [`answer_text`].
///
/// Requests with no matched answer produce no result and are returned as the
/// second tuple element, in request order, so the loop can re-pause.
///
/// # Errors
///
/// The signature keeps a `Result` so later rounds can surface matching or
/// execution errors; the current replay turns every tool and handler outcome
/// into a [`ToolCallResult`].
#[allow(dead_code)] // used by the loop's resume path
pub(crate) async fn replay_interrupted(
    state: &InterruptionState,
    interactive: &InteractiveInput,
    tools: &IndexMap<String, Arc<dyn Tool>>,
    options: &ToolExecutionOptions,
    resilience: &ToolResiliencePolicy,
    client_tools: &Arc<ClientToolRegistry>,
) -> Result<(Vec<ToolCallResult>, Vec<InterruptRequest>)> {
    let pending = state
        .requests
        .iter()
        .map(|(call_id, request)| (call_id.clone(), request.interrupt_id.clone()))
        .collect::<Vec<_>>();
    let matched = match_answers(interactive, &pending);

    let mut results = Vec::with_capacity(state.requests.len());
    let mut unanswered = Vec::new();
    for (call_id, request) in &state.requests {
        let call = ToolCall {
            id: Some(if request.tool_call_id.is_empty() {
                call_id.clone()
            } else {
                request.tool_call_id.clone()
            }),
            name: request.tool_name.clone(),
            arguments: request.tool_args.clone(),
            index: request.index,
            ..ToolCall::default()
        };
        let Some(answer) = matched.answers.get(call_id) else {
            unanswered.push(request.clone());
            continue;
        };
        if request.kind == InterruptKind::ClientTool
            || client_tools.is_client_tool(&request.tool_name)
        {
            match client_tools.invoke(&call, answer).await {
                Ok(value) => {
                    let text = text_of(&value);
                    results.push(ToolCallResult::success(call, value, text));
                }
                Err(error) => results.push(ToolCallResult::failure(call, error.to_string())),
            }
            continue;
        }
        match parse_approval(answer) {
            Some(true) => {
                let executed =
                    execute_calls_with(tools, std::slice::from_ref(&call), options, resilience)
                        .await;
                if let Some(result) = executed.into_iter().next() {
                    results.push(result);
                } else {
                    let name = call.name.clone();
                    results.push(ToolCallResult::failure(
                        call,
                        format!("tool '{name}' produced no result"),
                    ));
                }
            }
            Some(false) => results.push(ToolCallResult::failure(call, reject_feedback(answer))),
            None => {
                let text = answer_text(answer);
                results.push(ToolCallResult::success(
                    call,
                    Value::String(text.clone()),
                    text,
                ));
            }
        }
    }
    Ok((results, unanswered))
}

/// The model-visible text of a client tool's value.
///
/// A string value is taken verbatim so a client returning `"ok"` does not make
/// the model read `"\"ok\""`; every other value is rendered as JSON.
fn text_of(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card::AgentCard;
    use crate::interrupt::{InterruptKind, InterruptRequest};
    use crate::policy::ClientToolHandler;
    use crate::rails::{RailContext, RailEvent};
    use crate::react::agent_loop::AgentRuntime;
    use crate::react::ReActAgent;
    use async_trait::async_trait;
    use openjiuwen_common::error::Error;
    use openjiuwen_foundation::{Model, Tool, ToolCard, ToolContext};
    use openjiuwen_session::AgentSession;
    use openjiuwen_test_support::ScriptedModel;
    use serde_json::json;
    use std::collections::BTreeMap;
    use std::sync::Arc;

    struct EchoTool {
        card: ToolCard,
    }

    #[async_trait]
    impl Tool for EchoTool {
        fn card(&self) -> &ToolCard {
            &self.card
        }

        async fn invoke(&self, inputs: Value, _ctx: &ToolContext) -> Result<Value> {
            Ok(json!({"echoed": inputs}))
        }
    }

    fn tools() -> IndexMap<String, Arc<dyn Tool>> {
        let tool: Arc<dyn Tool> = Arc::new(EchoTool {
            card: ToolCard::new("shell", "runs a shell command", json!({"type": "object"}))
                .with_id("shell-1"),
        });
        let mut tools = IndexMap::new();
        tools.insert("shell".to_owned(), tool);
        tools
    }

    fn request(call_id: &str, tool_name: &str, arguments: &str) -> InterruptRequest {
        InterruptRequest {
            tool_name: tool_name.to_owned(),
            tool_call_id: call_id.to_owned(),
            tool_args: arguments.to_owned(),
            index: Some(0),
            ..InterruptRequest::new(InterruptKind::Confirm, "agent", "ok?")
        }
    }

    fn state(requests: Vec<(&str, InterruptRequest)>) -> InterruptionState {
        InterruptionState {
            iteration: 1,
            original_query: "q".to_owned(),
            requests: requests
                .into_iter()
                .map(|(call_id, request)| (call_id.to_owned(), request))
                .collect(),
            ..InterruptionState::default()
        }
    }

    fn interactive_with(entries: Vec<(&str, Value)>) -> InteractiveInput {
        InteractiveInput {
            user_inputs: entries
                .into_iter()
                .map(|(key, value)| (key.to_owned(), value))
                .collect::<BTreeMap<_, _>>(),
            ..InteractiveInput::default()
        }
    }

    /// Replays `state` against an empty client-tool registry.
    async fn replay(
        state: &InterruptionState,
        interactive: &InteractiveInput,
    ) -> (Vec<ToolCallResult>, Vec<InterruptRequest>) {
        replay_interrupted(
            state,
            interactive,
            &tools(),
            &ToolExecutionOptions::default(),
            &ToolResiliencePolicy::default(),
            &Arc::new(ClientToolRegistry::new()),
        )
        .await
        .unwrap()
    }

    #[tokio::test]
    async fn approved_call_re_executes_the_mounted_tool() {
        let state = state(vec![(
            "call-1",
            request("call-1", "shell", r#"{"command":"ls"}"#),
        )]);
        let interactive = interactive_with(vec![("call-1", json!({"approved": true}))]);
        let (results, unanswered) = replay(&state, &interactive).await;
        assert_eq!(unanswered, Vec::<InterruptRequest>::new());
        assert_eq!(results.len(), 1);
        assert!(results[0].success);
        assert_eq!(results[0].call.id.as_deref(), Some("call-1"));
        assert_eq!(
            results[0].output,
            Some(json!({"echoed": {"command": "ls"}}))
        );
    }

    #[tokio::test]
    async fn rejected_call_yields_the_rejection_feedback() {
        let state = state(vec![(
            "call-1",
            request("call-1", "shell", r#"{"command":"rm"}"#),
        )]);
        let interactive = interactive_with(vec![(
            "call-1",
            json!({"approved": false, "feedback": "nope"}),
        )]);
        let (results, unanswered) = replay(&state, &interactive).await;
        assert_eq!(unanswered, Vec::<InterruptRequest>::new());
        assert_eq!(results.len(), 1);
        assert!(!results[0].success);
        assert_eq!(results[0].text, "nope");
        assert_eq!(results[0].error.as_deref(), Some("nope"));
    }

    #[tokio::test]
    async fn ask_user_answer_becomes_the_result_text() {
        let state = state(vec![(
            "call-1",
            InterruptRequest {
                kind: InterruptKind::AskUser,
                ..request("call-1", "ask", "{}")
            },
        )]);
        let interactive = interactive_with(vec![("call-1", json!("Paris"))]);
        let (results, unanswered) = replay(&state, &interactive).await;
        assert_eq!(unanswered, Vec::<InterruptRequest>::new());
        assert_eq!(results.len(), 1);
        assert!(results[0].success);
        assert_eq!(results[0].text, "Paris");
        assert_eq!(results[0].output, Some(json!("Paris")));
    }

    #[tokio::test]
    async fn unanswered_request_is_returned_without_a_result() {
        let state = state(vec![
            ("call-1", request("call-1", "shell", "{}")),
            ("call-2", request("call-2", "shell", "{}")),
        ]);
        let interactive = interactive_with(vec![("call-1", json!({"approved": true}))]);
        let (results, unanswered) = replay(&state, &interactive).await;
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].call.id.as_deref(), Some("call-1"));
        assert_eq!(unanswered.len(), 1);
        assert_eq!(unanswered[0].tool_call_id, "call-2");
    }

    #[tokio::test]
    async fn answer_keyed_by_interrupt_id_is_matched() {
        let request = request("call-1", "shell", "{}");
        let interrupt_id = request.interrupt_id.clone();
        let state = state(vec![("call-1", request)]);
        let interactive = InteractiveInput {
            interrupt_answers: BTreeMap::from([(interrupt_id, json!("yes"))]),
            ..InteractiveInput::default()
        };
        let (results, unanswered) = replay(&state, &interactive).await;
        assert_eq!(unanswered, Vec::<InterruptRequest>::new());
        assert_eq!(results.len(), 1);
        assert!(results[0].success);
    }

    /// Completes a client call with the answer, or fails when `fail` is set.
    struct ClientHandler {
        fail: bool,
    }

    #[async_trait]
    impl ClientToolHandler for ClientHandler {
        async fn invoke(&self, _call: &ToolCall, answer: &Value) -> Result<Value> {
            if self.fail {
                Err(Error::tool("client boom"))
            } else {
                Ok(answer.clone())
            }
        }
    }

    fn client_registry(fail: bool) -> Arc<ClientToolRegistry> {
        let registry = ClientToolRegistry::new();
        registry.register("run_on_client", Arc::new(ClientHandler { fail }));
        Arc::new(registry)
    }

    async fn replay_client(
        state: &InterruptionState,
        interactive: &InteractiveInput,
        registry: &Arc<ClientToolRegistry>,
    ) -> (Vec<ToolCallResult>, Vec<InterruptRequest>) {
        replay_interrupted(
            state,
            interactive,
            &tools(),
            &ToolExecutionOptions::default(),
            &ToolResiliencePolicy::default(),
            registry,
        )
        .await
        .unwrap()
    }

    #[tokio::test]
    async fn a_client_tool_answer_is_completed_by_its_handler() {
        let state = state(vec![(
            "call-1",
            InterruptRequest {
                kind: InterruptKind::ClientTool,
                ..request("call-1", "run_on_client", "{}")
            },
        )]);
        let interactive = interactive_with(vec![("call-1", json!({"stdout": "ok"}))]);
        let (results, unanswered) =
            replay_client(&state, &interactive, &client_registry(false)).await;
        assert_eq!(unanswered, Vec::<InterruptRequest>::new());
        assert_eq!(results.len(), 1);
        assert!(results[0].success);
        assert_eq!(results[0].output, Some(json!({"stdout": "ok"})));
        assert_eq!(results[0].text, r#"{"stdout":"ok"}"#);
    }

    #[tokio::test]
    async fn a_failing_client_handler_yields_a_failed_result() {
        let state = state(vec![(
            "call-1",
            InterruptRequest {
                kind: InterruptKind::ClientTool,
                ..request("call-1", "run_on_client", "{}")
            },
        )]);
        let interactive = interactive_with(vec![("call-1", json!("ok"))]);
        let (results, _) = replay_client(&state, &interactive, &client_registry(true)).await;
        assert_eq!(results.len(), 1);
        assert!(!results[0].success);
        assert!(results[0]
            .error
            .as_deref()
            .is_some_and(|error| error.contains("client boom")));
    }

    /// Answers with a text part plus an image part, like a screenshot tool.
    struct ShotTool {
        card: ToolCard,
    }

    #[async_trait]
    impl Tool for ShotTool {
        fn card(&self) -> &ToolCard {
            &self.card
        }

        async fn invoke(&self, _inputs: Value, _ctx: &ToolContext) -> Result<Value> {
            Ok(json!({"content": [
                {"type": "text", "text": "ok"},
                {"type": "image", "image_url": {"url": "data:image/png;base64,AAA"}},
            ]}))
        }
    }

    /// An agent owning `ShotTool`, snapshot into a runtime for `resume_or_seed`.
    fn shot_runtime() -> AgentRuntime {
        let agent = ReActAgent::new(AgentCard::new("react"));
        let model: Arc<dyn Model> = Arc::new(ScriptedModel::text("done"));
        agent.set_model(model);
        agent.add_tool(Arc::new(ShotTool {
            card: ToolCard::new("shot", "captures the screen", json!({"type": "object"}))
                .with_id("shot-1"),
        }));
        AgentRuntime::snapshot(&agent).unwrap()
    }

    #[tokio::test]
    async fn a_replayed_result_keeps_its_parts_in_the_observation() {
        let runtime = shot_runtime();
        let session: Arc<dyn Session> = Arc::new(AgentSession::new("s1"));
        state(vec![("call-1", request("call-1", "shot", "{}"))])
            .save(&*session)
            .await
            .unwrap();
        let input =
            AgentInput::parse(&json!({"query": {"user_inputs": {"call-1": {"approved": true}}}}))
                .unwrap();
        let context = runtime
            .engine
            .create_context("main", Some(Arc::clone(&session)))
            .await
            .unwrap();
        let mut ctx = RailContext::new(RailEvent::BeforeInvoke).with_agent_id("agent");

        let outcome = resume_or_seed(
            &runtime,
            &input,
            &session,
            None,
            &ToolExecutionOptions::default(),
            &mut ctx,
            &context,
        )
        .await
        .unwrap();

        let ResumeOutcome::Start { observations, .. } = outcome else {
            panic!("the approved call must replay");
        };
        assert_eq!(observations.len(), 1);
        assert!(observations[0].content.contains("ok"));
        assert_eq!(observations[0].parts.len(), 2);
        assert_eq!(
            observations[0].parts[0],
            openjiuwen_foundation::ContentPart::Text {
                text: "ok".to_owned()
            }
        );
        assert_eq!(
            observations[0].parts[1]
                .image_url()
                .map(|image| image.url.as_str()),
            Some("data:image/png;base64,AAA")
        );
    }
}
