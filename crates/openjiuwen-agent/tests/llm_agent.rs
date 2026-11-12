#![allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]
//! Integration tests for `LlmAgent`, the configured `ReActAgent` composition.
//!
//! The suite pins the Python-compatible contract of `openjiuwen.core.application.llm_agent`:
//! the configuration mapping (`constrain.max_iteration` → the `ReAct` iteration
//! budget, `constrain.reserved_max_chat_rounds` → the context window rounds, an
//! empty provider → `"openai"`), invoke/stream answer pass-through with the
//! completed output filtered to `{output, result_type}`, prompt-template
//! replacement, and interrupt pass-through. Everything is offline and
//! deterministic: the model is a `ScriptedModel` and fixtures are in-process
//! closures.
mod common;

use std::sync::Arc;

use futures::StreamExt;
use openjiuwen_agent::output::{ANSWER, LLM_OUTPUT};
use openjiuwen_agent::{
    Agent, AgentChunk, AskUserRail, ConstrainConfig, LlmAgent, LlmAgentConfig, ModelConfig,
    ModelInfo, ResultType,
};
use openjiuwen_foundation::{BaseMessage, Model, Role};
use openjiuwen_test_support::{MockCompletion, ScriptedModel};
use serde_json::json;

/// Builds a `ModelConfig` for `provider` and `api_base` with a fixed model name.
fn model_config(provider: &str, api_base: &str) -> ModelConfig {
    ModelConfig {
        model_provider: provider.to_owned(),
        model_info: ModelInfo {
            api_base: api_base.to_owned(),
            model_name: "test-model".to_owned(),
            streaming: true,
            ..ModelInfo::default()
        },
    }
}

/// Builds a model-configured `LlmAgent` with the default budgets.
fn configured_agent(id: &str) -> LlmAgent {
    LlmAgent::new(LlmAgentConfig::new(id).with_model(model_config("openai", "")))
        .expect("the LLM agent fixture builds")
}

/// Hands a clone of `model` to `agent` while the test keeps the original for
/// inspecting the recorded calls.
fn set_model(agent: &LlmAgent, model: &ScriptedModel) {
    agent
        .react()
        .set_model(Arc::new(model.clone()) as Arc<dyn Model>);
}

/// Concatenates the `content` of every frame of `kind`.
fn content_of(frames: &[AgentChunk], kind: &str) -> String {
    frames
        .iter()
        .filter(|frame| frame.kind == kind)
        .filter_map(|frame| {
            frame
                .payload
                .get("content")
                .and_then(|value| value.as_str())
        })
        .collect()
}

#[test]
fn config_maps_onto_the_react_agent() {
    let config = LlmAgentConfig::new("agent-1")
        .with_model(ModelConfig {
            model_provider: "anthropic".to_owned(),
            model_info: ModelInfo {
                api_base: "http://x".to_owned(),
                model_name: "test-model".to_owned(),
                streaming: true,
                ..ModelInfo::default()
            },
        })
        .with_constrain(ConstrainConfig {
            reserved_max_chat_rounds: 4,
            max_iteration: 3,
        });

    let agent = LlmAgent::new(config).expect("the LLM agent fixture builds");

    assert_eq!(agent.card().id, "agent-1");
    let react = agent.react().config();
    assert_eq!(react.max_iterations, 3);
    assert_eq!(react.context_engine_config.default_window_rounds, Some(4));
    assert_eq!(react.model_name, "test-model");
    assert_eq!(react.model_provider, "anthropic");
    assert_eq!(react.api_base.as_deref(), Some("http://x"));
}

#[test]
fn an_empty_provider_falls_back_to_openai() {
    let agent = LlmAgent::new(LlmAgentConfig::new("agent-2").with_model(model_config("", "")))
        .expect("the LLM agent fixture builds");

    assert_eq!(agent.react().config().model_provider, "openai");
    // The default budgets survive: 10 rounds and 5 iterations.
    assert_eq!(
        agent
            .react()
            .config()
            .context_engine_config
            .default_window_rounds,
        Some(10)
    );
    assert_eq!(agent.react().config().max_iterations, 5);
}

#[tokio::test]
async fn invoke_returns_the_scripted_answer() {
    let model = ScriptedModel::builder().respond_text("hello back").build();
    let agent = configured_agent("agent-1");
    set_model(&agent, &model);

    let output = agent.invoke(json!("hi"), None).await.unwrap();

    assert_eq!(output.result_type, ResultType::Answer);
    assert_eq!(output.output, "hello back");
    assert_eq!(model.call_count(), 1);
}

#[tokio::test]
async fn answers_are_filtered_to_output_and_result_type() {
    let model = ScriptedModel::builder().respond_text("hello back").build();
    let agent = configured_agent("agent-1");
    set_model(&agent, &model);

    let output = agent.invoke(json!("hi"), None).await.unwrap();

    // The completed output drops every ReAct-only field.
    assert_eq!(output.output, "hello back");
    assert_eq!(output.result_type, ResultType::Answer);
    assert!(output.observations.is_empty());
    assert!(output.iterations_used.is_none());
    assert!(output.finish_reason.is_none());
    assert!(output.termination_reason.is_none());
    assert!(!output.truncated_by_max_iterations);
    assert!(output.state.is_empty());
    assert!(output.interrupt_ids.is_empty());
    let wire = serde_json::to_value(&output).unwrap();
    assert!(wire.get("observations").is_none(), "{wire}");
    assert!(wire.get("iterations_used").is_none(), "{wire}");
    assert_eq!(wire["output"], "hello back");
    assert_eq!(wire["result_type"], "answer");
}

#[tokio::test]
async fn stream_yields_llm_output_and_one_terminal_answer() {
    let model = ScriptedModel::builder().stream_text("hello world").build();
    let agent = configured_agent("agent-1");
    set_model(&agent, &model);

    let frames: Vec<AgentChunk> = agent
        .stream(json!("hi"), Some(common::session("s1")))
        .map(|frame| frame.unwrap())
        .collect()
        .await;

    assert!(
        frames.iter().any(|frame| frame.kind == LLM_OUTPUT),
        "no llm_output frame was published: {frames:?}"
    );
    assert_eq!(content_of(&frames, LLM_OUTPUT), "hello world");
    let answer_frames = frames
        .iter()
        .filter(|frame| frame.kind == ANSWER)
        .collect::<Vec<_>>();
    assert_eq!(answer_frames.len(), 1, "{frames:?}");
    let last = frames.last().expect("the stream has a terminal frame");
    assert_eq!(last.kind, ANSWER);
    assert_eq!(last.payload["output"], "hello world");
    assert_eq!(last.payload["result_type"], "answer");
}

#[tokio::test]
async fn set_prompt_template_changes_the_system_prompt_seen_by_the_model() {
    let model = ScriptedModel::builder().respond_text("ok").build();
    let agent = configured_agent("agent-1");
    set_model(&agent, &model);

    agent.set_prompt_template(vec![BaseMessage::system("be terse")]);
    agent.invoke(json!("hi"), None).await.unwrap();

    assert_eq!(agent.config().prompt_template.len(), 1);
    let messages = model.last_call().expect("one model call").messages;
    assert!(
        messages
            .iter()
            .any(|message| message.role == Role::System && message.text() == "be terse"),
        "{messages:?}"
    );
    // The query still reaches the model as its own user message.
    assert!(
        messages
            .iter()
            .any(|message| message.role.is_user() && message.text().contains("hi")),
        "{messages:?}"
    );
}

#[tokio::test]
async fn an_interrupted_output_passes_through_unchanged() {
    let model = ScriptedModel::builder()
        .completion(MockCompletion::tool_call(
            "echo",
            json!({"question": "which?"}),
        ))
        .respond_text("after answer")
        .build();
    let agent = configured_agent("agent-1");
    set_model(&agent, &model);
    agent
        .react()
        .add_tool(common::recording_tool("echo", &common::log()));
    agent
        .react()
        .register_rail(Arc::new(AskUserRail::for_tools(["echo"])));

    let output = agent.invoke(json!("hi"), None).await.unwrap();

    assert!(output.is_interrupt());
    assert_eq!(output.result_type, ResultType::Interrupt);
    assert!(!output.interrupt_ids.is_empty(), "{output:?}");
    assert_eq!(output.state.len(), 1, "{output:?}");
    assert_eq!(output.state[0].kind, openjiuwen_agent::output::INTERACTION);
    // The paused run consumed exactly one model call and never finalized.
    assert_eq!(model.call_count(), 1);
}
