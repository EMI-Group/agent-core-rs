#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Kernel-flow scenario: model → tool → session → workflow → runner.
//!
//! This binary is the cross-crate seam between the kernel layers, so it drives
//! the real `ReActAgent`, the real session/checkpointer pair, the real workflow
//! engine and the real `Runner` facade together instead of each crate's own
//! tests. Every scenario is offline and deterministic: models are
//! `ScriptedModel`s, sessions and checkpointers are in-process, no global
//! `Runner` or shared checkpointer is used, and nothing reads the clock.
//! Assertions are made against recorded evidence — the model's call history,
//! the collected `AgentChunk` frames, the checkpoint store — never against
//! timing, timestamps, durations or generated ids.

// The shared helper module's `match Err(_)` arms are part of the deadline
// contract (`tokio::time::timeout` only ever yields `Elapsed`), and the allow is
// scoped to the module so this scenario's own code stays fully linted.
#[allow(clippy::match_wild_err_arm)]
mod support;

use std::sync::Arc;

use openjiuwen::core::foundation::{FunctionTool, Model, Tool};
use openjiuwen::core::runner::{RunOptions, Runner};
use openjiuwen::core::session::AgentSessionOptions;
use openjiuwen::core::single_agent::output::{ANSWER, LLM_OUTPUT};
use openjiuwen::core::single_agent::{
    Agent, AgentCard, AgentChunk, ReActAgent, ReActAgentConfig, ResultType,
};
use openjiuwen::core::workflow::{
    ComponentOptions, EndComponent, LlmCompConfig, LlmComponent, StartComponent, Workflow,
    WorkflowCard, WorkflowExecutionState,
};
use openjiuwen_test_support::{
    agent_session, agent_session_with, memory_checkpointer, workflow_session, MockCompletion,
    ScriptedModel,
};
use serde_json::{json, Value};

use support::{collect_bounded, within};

/// Hands a clone of `model` to an agent or component while the test keeps the
/// original handle for inspecting the recorded calls.
fn shared(model: &ScriptedModel) -> Arc<dyn Model> {
    Arc::new(model.clone())
}

/// Builds the `echo` tool: it answers with the arguments it was given, so the
/// tool message the model reads back is the rendered call arguments.
fn echo_tool() -> Arc<dyn Tool> {
    FunctionTool::builder()
        .name("echo")
        .description("echoes its arguments")
        .input_schema(json!({"type": "object"}))
        .handler(|args: Value, _ctx| async move { Ok(args) })
        .build()
        .expect("the echo tool fixture builds")
        .boxed()
}

/// Builds a `ReAct` agent with default configuration, driven by `model`.
fn react_agent(model: Arc<dyn Model>) -> ReActAgent {
    let agent = ReActAgent::new(AgentCard::new("react").with_id("react"));
    agent.configure(ReActAgentConfig::new());
    agent.set_model(model);
    agent
}

/// Concatenates the `content` of every frame of `kind`, in order.
fn content_of(frames: &[AgentChunk], kind: &str) -> String {
    frames
        .iter()
        .filter(|frame| frame.kind == kind)
        .filter_map(|frame| frame.payload.get("content").and_then(Value::as_str))
        .collect()
}

/// Asserts the terminal-`answer` frame contract shared by both streaming
/// scenarios: exactly one `answer` frame, last, carrying `expected`.
fn assert_terminal_answer(frames: &[AgentChunk], expected: &str) {
    let answers = frames
        .iter()
        .filter(|frame| frame.kind == ANSWER)
        .collect::<Vec<_>>();
    assert_eq!(answers.len(), 1, "exactly one answer frame: {frames:?}");
    let last = frames.last().expect("the stream has a terminal frame");
    assert_eq!(last.kind, ANSWER, "the answer frame is last: {frames:?}");
    assert_eq!(last.payload["output"], expected);
    assert_eq!(last.payload["result_type"], "answer");
}

/// `start → llm → end` where the LLM renders `{{query}}` into its answer and the
/// end node reports it under `answer`.
fn linear_workflow(model: Arc<dyn Model>) -> Workflow {
    let mut workflow = Workflow::new(WorkflowCard::new("kernel-flow"));
    workflow
        .set_start_comp(
            "start",
            StartComponent::new(),
            ComponentOptions::new().with_inputs_schema(json!({"query": "${inputs.query}"})),
        )
        .add_workflow_comp(
            "llm",
            LlmComponent::new(LlmCompConfig {
                model: Some(model),
                user_prompt_template: Some("Answer: {{query}}".into()),
                ..Default::default()
            })
            .unwrap(),
            ComponentOptions::new().with_inputs_schema(json!({"query": "${start.query}"})),
        )
        .set_end_comp(
            "end",
            EndComponent::new(),
            ComponentOptions::new().with_inputs_schema(json!({"answer": "${llm.result}"})),
        )
        .add_connection("start", "llm")
        .add_connection("llm", "end");
    workflow
}

// A tool round proves the loop closes: the model asks for a tool, the tool's
// result is rendered as a `tool` message, and that message is fed back into the
// second model call together with the call id it answers.
#[tokio::test]
async fn react_tool_round_feeds_the_tool_message_back_to_the_model() {
    let model = ScriptedModel::builder()
        .completion(MockCompletion::tool_call("echo", json!({"value": "hi"})))
        .respond_text("final answer")
        .build();
    let agent = react_agent(shared(&model));
    agent.add_tool(echo_tool());

    let output = within(
        "the react tool round",
        agent.invoke(json!("hi"), Some(agent_session("react-tool"))),
    )
    .await
    .unwrap();

    assert_eq!(output.result_type, ResultType::Answer);
    assert_eq!(output.output, "final answer");
    assert_eq!(model.call_count(), 2, "one tool turn plus the answer turn");
    assert_eq!(model.remaining(), 0, "the whole script was consumed");

    let calls = model.calls();
    let tool_messages = calls[1]
        .messages
        .iter()
        .filter(|message| message.role.is_tool())
        .collect::<Vec<_>>();
    assert_eq!(
        tool_messages.len(),
        1,
        "one tool message was fed back: {:?}",
        calls[1].messages
    );
    let tool_message = tool_messages[0];
    assert_eq!(tool_message.tool_call_id.as_deref(), Some("call_echo"));
    assert!(
        tool_message.text().contains("hi"),
        "the tool message carries the echoed arguments: {}",
        tool_message.text()
    );
}

// The agent's own stream: model deltas arrive as `llm_output` frames and the run
// ends with exactly one `answer` frame carrying the assembled output.
#[tokio::test]
async fn agent_stream_yields_llm_output_and_answer_frames() {
    let model = ScriptedModel::builder().stream_text("hello world").build();
    let agent = react_agent(shared(&model));

    let frames = collect_bounded(
        agent.stream(json!("hi"), Some(agent_session("agent-stream"))),
        64,
    )
    .await
    .into_iter()
    .map(|frame| frame.unwrap())
    .collect::<Vec<_>>();

    assert_eq!(content_of(&frames, LLM_OUTPUT), "hello world");
    assert_terminal_answer(&frames, "hello world");
}

// A checkpoint is a real round trip through a shared store: a fresh session with
// the same id starts empty, `resume` finds the snapshot, and both the global and
// the component lanes come back exactly as saved.
#[tokio::test]
async fn session_state_survives_a_checkpoint_round_trip() {
    let cp = memory_checkpointer();
    let first = agent_session_with(AgentSessionOptions {
        session_id: Some("cp-session".into()),
        checkpointer: Some(Arc::clone(&cp)),
        ..Default::default()
    });
    first
        .update_state(json!({"counter": 1, "nested": {"label": "kept"}}))
        .await
        .unwrap();
    first
        .update_component_state("node", json!({"value": 2}))
        .await
        .unwrap();
    let checkpoint_id = first.save_checkpoint(Some("cp")).await.unwrap();
    assert_eq!(checkpoint_id, "cp");

    let second = agent_session_with(AgentSessionOptions {
        session_id: Some("cp-session".into()),
        checkpointer: Some(Arc::clone(&cp)),
        ..Default::default()
    });
    assert_eq!(
        second.get_state("counter").await.unwrap(),
        None,
        "a fresh session starts from an empty global lane"
    );

    assert!(
        second.resume("cp").await.unwrap(),
        "the shared checkpointer holds the snapshot"
    );
    assert_eq!(second.get_state("counter").await.unwrap(), Some(json!(1)));
    let nested = second.get_state("nested").await.unwrap().unwrap();
    assert_eq!(nested["label"], "kept");
    assert_eq!(
        second.get_component_state("node").await.unwrap(),
        Some(json!({"value": 2}))
    );
}

// `${node.field}` resolution across two hops: the start node exposes the run
// inputs as `query`, the LLM node renders `{{query}}` from that resolved value,
// and the end node reports `${llm.result}`.
#[tokio::test]
async fn workflow_resolves_node_field_references_end_to_end() {
    let model = ScriptedModel::builder()
        .respond_text("the resolved answer")
        .build();
    let workflow = linear_workflow(shared(&model));

    let output = within(
        "the reference workflow run",
        workflow.invoke(json!({"query": "hello"}), workflow_session("workflow-refs")),
    )
    .await
    .unwrap();

    assert_eq!(output.state, WorkflowExecutionState::Completed);
    assert_eq!(
        output.result,
        json!({"output": {"answer": "the resolved answer"}})
    );
    let rendered = output.result.to_string();
    assert!(
        !rendered.contains("${"),
        "no reference is left unresolved: {rendered}"
    );
    assert_eq!(model.call_count(), 1, "the llm node ran exactly once");
    let call = model.last_call().expect("the llm node called the model");
    assert_eq!(call.last_user_text().as_deref(), Some("Answer: hello"));
}

// The `Runner` facade resolves both registry kinds and drives the same real
// implementations the direct calls above exercised.
#[tokio::test]
async fn local_runner_runs_an_agent_and_a_workflow() {
    let runner = Runner::new();

    let agent_model = ScriptedModel::builder()
        .respond_text("agent answer")
        .build();
    let agent: Arc<dyn Agent> = Arc::new(react_agent(shared(&agent_model)));
    runner
        .resource_mgr()
        .register_agent(
            AgentCard::new("react").with_id("react"),
            Arc::new(move || Arc::clone(&agent)),
        )
        .unwrap();

    let workflow_model = ScriptedModel::builder()
        .respond_text("workflow answer")
        .build();
    let workflow = linear_workflow(shared(&workflow_model));
    runner
        .resource_mgr()
        .register_workflow(
            WorkflowCard::new("kernel-flow"),
            Arc::new(move || workflow.clone()),
        )
        .unwrap();

    let agent_output = runner
        .run_agent(
            "react",
            json!("hi"),
            RunOptions::default().with_session(agent_session("runner-agent")),
        )
        .await
        .unwrap();
    assert_eq!(agent_output.result_type, ResultType::Answer);
    assert_eq!(agent_output.output, "agent answer");
    assert_eq!(agent_model.call_count(), 1);

    let workflow_output = runner
        .run_workflow(
            "kernel-flow",
            json!({"query": "hello"}),
            RunOptions::default().with_session(workflow_session("runner-workflow")),
        )
        .await
        .unwrap();
    assert_eq!(
        workflow_output.result,
        json!({"output": {"answer": "workflow answer"}})
    );
    assert_eq!(workflow_model.call_count(), 1);

    // `stop` clears the registries, so it runs only once every assertion is done.
    runner.stop().await.unwrap();
}

// Streaming through the facade keeps the agent's frame contract: the model
// deltas are `llm_output` frames and the run ends with one `answer` frame.
#[tokio::test]
async fn local_runner_streams_agent_frames() {
    let runner = Runner::new();
    let model = ScriptedModel::builder()
        .stream_text("streamed answer")
        .build();
    let agent: Arc<dyn Agent> = Arc::new(react_agent(shared(&model)));
    runner
        .resource_mgr()
        .register_agent(
            AgentCard::new("react").with_id("react"),
            Arc::new(move || Arc::clone(&agent)),
        )
        .unwrap();

    let frames = collect_bounded(
        runner.run_agent_streaming(
            "react",
            json!("hi"),
            RunOptions::default().with_session(agent_session("runner-stream")),
        ),
        64,
    )
    .await
    .into_iter()
    .map(|frame| frame.unwrap())
    .collect::<Vec<_>>();

    assert_eq!(content_of(&frames, LLM_OUTPUT), "streamed answer");
    assert_terminal_answer(&frames, "streamed answer");
    assert_eq!(model.call_count(), 1);

    runner.stop().await.unwrap();
}
