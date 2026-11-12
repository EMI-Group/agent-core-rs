#![allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]
//! Integration tests for the implemented `ReAct` loop.
//!
//! The suite pins the loop's observable contract against a `ScriptedModel`:
//! a tool-free answer, tool rounds (single, parallel, failing) feeding tool
//! messages back to the model, the three max-iteration policies, the
//! empty-response retry, prompt rendering, and the input/config error paths.
//! Everything is offline and deterministic: tools are in-process closures and
//! assertions read the recorded call history instead of timing anything.
mod common;

use std::sync::Arc;

use openjiuwen_agent::{
    Agent, AgentCard, MaxIterationsPolicy, ReActAgent, ReActAgentConfig, ResultType,
};
use openjiuwen_common::error::ErrorKind;
use openjiuwen_foundation::{BaseMessage, Model, Role};
use openjiuwen_test_support::{MockCompletion, MockToolCall, ScriptedModel};
use serde_json::{json, Value};

/// Hands a clone of `model` to the agent while the test keeps the original for
/// inspecting the recorded calls.
fn shared(model: &ScriptedModel) -> Arc<dyn Model> {
    Arc::new(model.clone())
}

/// Collects the text of every message in `messages` that has `role`.
fn texts_with_role(messages: &[BaseMessage], role: &Role) -> Vec<String> {
    messages
        .iter()
        .filter(|message| &message.role == role)
        .map(|message| message.text().into_owned())
        .collect()
}

#[tokio::test]
async fn no_tool_turn_answers_with_the_model_text() {
    let model = ScriptedModel::builder().respond_text("hello back").build();
    let agent = common::agent(ReActAgentConfig::new(), shared(&model));

    let output = agent.invoke(json!("hi"), None).await.unwrap();

    assert_eq!(output.result_type, ResultType::Answer);
    assert_eq!(output.output, "hello back");
    assert_eq!(model.call_count(), 1);
}

#[tokio::test]
async fn tool_round_feeds_the_tool_message_back_to_the_model() {
    let model = ScriptedModel::builder()
        .completion(MockCompletion::tool_call("echo", json!({"value": "hi"})))
        .respond_text("final answer")
        .build();
    let log = common::log();
    let agent = common::agent(ReActAgentConfig::new(), shared(&model));
    agent.add_tool(common::recording_tool("echo", &log));

    let output = agent.invoke(json!("hi"), None).await.unwrap();

    assert_eq!(output.result_type, ResultType::Answer);
    assert_eq!(output.output, "final answer");
    assert_eq!(model.call_count(), 2);
    let calls = model.calls();
    let tool_texts = texts_with_role(&calls[1].messages, &Role::Tool);
    assert_eq!(tool_texts.len(), 1, "{:?}", calls[1].messages);
    assert!(tool_texts[0].contains("tool-output"), "{}", tool_texts[0]);
    assert_eq!(log.lock().clone(), vec![json!({"value": "hi"})]);
}

#[tokio::test]
async fn parallel_tool_calls_both_run_and_feed_the_answer() {
    let model = ScriptedModel::builder()
        .completion(MockCompletion::tool_calls(vec![
            MockToolCall::new("alpha", json!({"n": 1})),
            MockToolCall::new("beta", json!({"n": 2})),
        ]))
        .respond_text("done")
        .build();
    let alpha_log = common::log();
    let beta_log = common::log();
    let agent = common::agent(ReActAgentConfig::new(), shared(&model));
    agent.add_tools(vec![
        common::recording_tool("alpha", &alpha_log),
        common::recording_tool("beta", &beta_log),
    ]);

    let output = agent.invoke(json!("hi"), None).await.unwrap();

    assert_eq!(output.result_type, ResultType::Answer);
    assert_eq!(output.output, "done");
    assert_eq!(alpha_log.lock().clone(), vec![json!({"n": 1})]);
    assert_eq!(beta_log.lock().clone(), vec![json!({"n": 2})]);
    let calls = model.calls();
    let tool_texts = texts_with_role(&calls[1].messages, &Role::Tool);
    assert_eq!(tool_texts.len(), 2, "{:?}", calls[1].messages);
}

#[tokio::test]
async fn a_tool_error_becomes_a_tool_message() {
    let model = ScriptedModel::builder()
        .completion(MockCompletion::tool_call("bad", json!({})))
        .respond_text("recovered")
        .build();
    let agent = common::agent(ReActAgentConfig::new(), shared(&model));
    agent.add_tool(common::failing_tool("bad"));

    let output = agent.invoke(json!("hi"), None).await.unwrap();

    assert_eq!(output.result_type, ResultType::Answer);
    assert_eq!(output.output, "recovered");
    assert_eq!(model.call_count(), 2);
    let calls = model.calls();
    let tool_texts = texts_with_role(&calls[1].messages, &Role::Tool);
    assert!(
        tool_texts.iter().any(|text| text.contains("boom")),
        "{tool_texts:?}"
    );
}

/// Builds the always-calls-a-tool agent used by the policy tests.
fn exhausted_agent(policy: MaxIterationsPolicy) -> (ScriptedModel, ReActAgent) {
    let model = ScriptedModel::builder()
        .completion(MockCompletion::tool_call("echo", json!({})))
        .respond_text("unused")
        .build();
    let config = ReActAgentConfig {
        max_iterations: 1,
        on_max_iterations: policy,
        ..ReActAgentConfig::new()
    };
    let agent = common::agent(config, shared(&model));
    agent.add_tool(common::recording_tool("echo", &common::log()));
    (model, agent)
}

#[tokio::test]
async fn exhausted_budget_with_the_error_policy_reports_max_iterations() {
    let (model, agent) = exhausted_agent(MaxIterationsPolicy::Error);

    let output = agent.invoke(json!("hi"), None).await.unwrap();

    assert_eq!(output.result_type, ResultType::Error);
    assert_eq!(output.termination_reason.as_deref(), Some("max_iterations"));
    assert_eq!(output.iterations_used, Some(1));
    assert!(output.truncated_by_max_iterations);
    assert_eq!(model.call_count(), 1);
}

#[tokio::test]
async fn exhausted_budget_with_the_return_observations_policy_keeps_the_tool_results() {
    let (_model, agent) = exhausted_agent(MaxIterationsPolicy::ReturnObservations);

    let output = agent.invoke(json!("hi"), None).await.unwrap();

    assert_eq!(output.result_type, ResultType::Error);
    assert_eq!(output.termination_reason.as_deref(), Some("max_iterations"));
    assert!(output.truncated_by_max_iterations);
    assert!(!output.observations.is_empty());
    assert_eq!(output.observations[0].name.as_deref(), Some("echo"));
}

#[tokio::test]
async fn the_final_answer_policy_returns_its_text_and_offers_no_tools() {
    let model = ScriptedModel::builder()
        .completion(MockCompletion::tool_call("echo", json!({})))
        .respond_text("best effort")
        .build();
    let config = ReActAgentConfig {
        max_iterations: 1,
        on_max_iterations: MaxIterationsPolicy::FinalAnswer,
        ..ReActAgentConfig::new()
    };
    let agent = common::agent(config, shared(&model));
    let log = common::log();
    agent.add_tool(common::recording_tool("echo", &log));

    let output = agent.invoke(json!("hi"), None).await.unwrap();

    assert_eq!(model.call_count(), 2);
    assert_eq!(output.result_type, ResultType::Answer);
    assert_eq!(output.output, "best effort");
    let last = model
        .last_call()
        .expect("the finalization call is recorded");
    assert!(last.request.tools.is_none(), "{:?}", last.request.tools);
}

#[tokio::test]
async fn an_empty_response_is_retried_once_after_a_notice() {
    let model = ScriptedModel::builder()
        .respond_text("")
        .respond_text("second try")
        .build();
    let agent = common::agent(ReActAgentConfig::new(), shared(&model));

    let output = agent.invoke(json!("hi"), None).await.unwrap();

    assert_eq!(output.result_type, ResultType::Answer);
    assert_eq!(output.output, "second try");
    assert_eq!(model.call_count(), 2);
    let flagged = model
        .calls()
        .iter()
        .flat_map(|call| call.messages.iter())
        .any(|message| {
            message
                .metadata
                .get("_openjiuwen_message_source_kind")
                .and_then(Value::as_str)
                == Some("empty_response")
        });
    assert!(flagged, "the retry notice carries its source metadata");
}

#[tokio::test]
async fn the_system_prompt_renders_the_query_placeholder() {
    let model = ScriptedModel::builder().respond_text("ok").build();
    let config = ReActAgentConfig {
        prompt_template: vec![BaseMessage::system("You answer for {{query}}.")],
        ..ReActAgentConfig::new()
    };
    let agent = common::agent(config, shared(&model));

    agent.invoke(json!("what?"), None).await.unwrap();

    let messages = model.last_call().expect("one model call").messages;
    assert!(
        messages.iter().any(|message| {
            message.role.is_system() && message.text().contains("You answer for what?.")
        }),
        "{messages:?}"
    );
}

#[tokio::test]
async fn a_string_input_is_a_text_query() {
    let model = ScriptedModel::text("ok");
    let agent = common::agent(ReActAgentConfig::new(), shared(&model));

    let output = agent.invoke(json!("hello"), None).await.unwrap();

    assert_eq!(output.result_type, ResultType::Answer);
    assert_eq!(output.output, "ok");
}

#[tokio::test]
async fn a_payload_without_a_query_is_a_validation_error() {
    let model = ScriptedModel::text("unused");
    let agent = common::agent(ReActAgentConfig::new(), shared(&model));

    let error = agent
        .invoke(json!({"conversation_id": "c1"}), None)
        .await
        .unwrap_err();

    assert_eq!(error.kind(), ErrorKind::Validation);
    assert_eq!(model.call_count(), 0);
}

#[tokio::test]
async fn an_unconfigured_model_is_a_config_error() {
    let agent = ReActAgent::new(AgentCard::new("react-test"));

    let error = agent.invoke(json!("hi"), None).await.unwrap_err();

    assert_eq!(error.kind(), ErrorKind::Config);
}
