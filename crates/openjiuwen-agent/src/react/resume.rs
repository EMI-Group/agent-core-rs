//! Replay of an interrupted `ReAct` turn.
//!
//! A turn paused on an interrupt persists its pending requests; resuming matches
//! the caller's [`InteractiveInput`] answers against them with
//! [`match_answers`](crate::interrupt::match_answers) (tool-call id first, then
//! interrupt id) and turns each answer into a tool result: an approved call is
//! re-executed, a rejected call yields the rejection-feedback tool message, and
//! an ask-user answer becomes the tool content.
use crate::abilities::{AbilityManager, ToolCallResult, ToolExecutionOptions};
use crate::interrupt::{
    answer_text, match_answers, parse_approval, reject_feedback, InterruptRequest,
    InterruptionState,
};
use openjiuwen_common::error::Result;
use openjiuwen_foundation::ToolCall;
use openjiuwen_session::InteractiveInput;
use serde_json::Value;

/// Iteration a resumed run restarts from after replaying the interrupted calls.
///
/// Kept as a documented starting point only: the loop computes
/// `state.iteration + 1` itself and does not read this constant.
#[allow(dead_code)] // used by the loop's resume path
pub(crate) const RESUME_START_ITERATION: usize = 1;

/// Replays the interrupted tool calls of `state` using `interactive` answers.
///
/// Each entry of [`InterruptionState::requests`] is matched to an answer:
///
/// - an approval (`parse_approval` is `Some(true)`) re-executes exactly that
///   call through `abilities`;
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
/// execution errors; the current replay turns every tool outcome into a
/// [`ToolCallResult`].
#[allow(dead_code)] // used by the loop's resume path
pub(crate) async fn replay_interrupted(
    state: &InterruptionState,
    interactive: &InteractiveInput,
    abilities: &AbilityManager,
    options: &ToolExecutionOptions,
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
        match parse_approval(answer) {
            Some(true) => {
                let executed = abilities
                    .execute(std::slice::from_ref(&call), options)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interrupt::{InterruptKind, InterruptRequest};
    use async_trait::async_trait;
    use openjiuwen_foundation::{Tool, ToolCard, ToolContext};
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

    fn abilities() -> AbilityManager {
        let mut manager = AbilityManager::new("agent");
        let tool: Arc<dyn Tool> = Arc::new(EchoTool {
            card: ToolCard::new("shell", "runs a shell command", json!({"type": "object"}))
                .with_id("shell-1"),
        });
        let added = manager.add_tool(tool);
        assert!(added.added);
        manager
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

    #[tokio::test]
    async fn approved_call_re_executes_the_mounted_tool() {
        let state = state(vec![(
            "call-1",
            request("call-1", "shell", r#"{"command":"ls"}"#),
        )]);
        let interactive = interactive_with(vec![("call-1", json!({"approved": true}))]);
        let (results, unanswered) = replay_interrupted(
            &state,
            &interactive,
            &abilities(),
            &ToolExecutionOptions::default(),
        )
        .await
        .unwrap();
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
        let (results, unanswered) = replay_interrupted(
            &state,
            &interactive,
            &abilities(),
            &ToolExecutionOptions::default(),
        )
        .await
        .unwrap();
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
        let (results, unanswered) = replay_interrupted(
            &state,
            &interactive,
            &abilities(),
            &ToolExecutionOptions::default(),
        )
        .await
        .unwrap();
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
        let (results, unanswered) = replay_interrupted(
            &state,
            &interactive,
            &abilities(),
            &ToolExecutionOptions::default(),
        )
        .await
        .unwrap();
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
        let (results, unanswered) = replay_interrupted(
            &state,
            &interactive,
            &abilities(),
            &ToolExecutionOptions::default(),
        )
        .await
        .unwrap();
        assert_eq!(unanswered, Vec::<InterruptRequest>::new());
        assert_eq!(results.len(), 1);
        assert!(results[0].success);
    }
}
