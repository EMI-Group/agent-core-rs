#![allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]
//! Integration tests for [`WorkflowAgent`].
//!
//! The agent selects one of its registered workflows and projects the
//! workflow result onto `AgentOutput` / `AgentChunk`. Everything here is offline
//! and deterministic: workflows are in-process graphs, the only model is a
//! `ScriptedModel`, and sessions are `AgentSession`s.
//!
//! Notes on the contract exercised below:
//!
//! - Completed runs expose the structured result in `data` and
//!   `workflow_output`, and its JSON text in `output`.
//! - `stream` forwards every workflow frame verbatim and appends exactly one
//!   terminal `answer` chunk.
//! - A `Workflow` interruption is an agent `interrupt` (frames in `state`,
//!   ids in `interrupt_ids`); the engine writes the `__interaction__` frame to
//!   the shared session hub itself.
//! - Resume needs a checkpointer on the supplied agent session, which the child
//!   workflow session inherits through its `parent`.
mod common;

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures::StreamExt;
use openjiuwen_agent::output::{ANSWER, WORKFLOW_FINAL};
use openjiuwen_agent::{Agent, ResultType, WorkflowAgent, WorkflowAgentConfig};
use openjiuwen_common::error::{ErrorKind, Result};
use openjiuwen_foundation::Model;
use openjiuwen_session::{
    AgentSession, AgentSessionOptions, Checkpointer, MemoryCheckpointer, Session, StreamEvent,
};
use openjiuwen_test_support::ScriptedModel;
use openjiuwen_workflow::{
    Component, ComponentAbility, ComponentCard, ComponentContext, ComponentOptions, EndComponent,
    StartComponent, Workflow, WorkflowCard, WorkflowExecutionState,
};
use serde_json::{json, Value};

/// An agent whose card identity comes from `id`.
fn agent(id: &str) -> WorkflowAgent {
    WorkflowAgent::new(WorkflowAgentConfig::new(id))
}

/// `start → end`; the end node's value is `{output_key: <start.query>}`.
///
/// The end component's `outputs_schema` projects its `{"output": …}` batch
/// result into a flat object, which is the workflow result.
fn echo_workflow(id: &str, output_key: &str) -> Arc<Workflow> {
    let mut workflow = Workflow::new(WorkflowCard::new(id));
    workflow
        .set_start_comp(
            "start",
            StartComponent::new(),
            ComponentOptions::new().with_inputs_schema(json!({"query": "${inputs.query}"})),
        )
        .set_end_comp(
            "end",
            EndComponent::new(),
            ComponentOptions::new()
                .with_inputs_schema(json!({"answer": "${start.query}"}))
                .with_outputs_schema(json!({output_key: "${output.output.answer}"})),
        )
        .add_connection("start", "end");
    Arc::new(workflow)
}

/// [`echo_workflow`] with a description used by intent detection.
fn described_workflow(id: &str, description: &str, output_key: &str) -> Arc<Workflow> {
    let mut card = WorkflowCard::new(id);
    description.clone_into(&mut card.description);
    let mut workflow = Workflow::new(card);
    workflow
        .set_start_comp(
            "start",
            StartComponent::new(),
            ComponentOptions::new().with_inputs_schema(json!({"query": "${inputs.query}"})),
        )
        .set_end_comp(
            "end",
            EndComponent::new(),
            ComponentOptions::new()
                .with_inputs_schema(json!({"answer": "${start.query}"}))
                .with_outputs_schema(json!({output_key: "${output.output.answer}"})),
        )
        .add_connection("start", "end");
    Arc::new(workflow)
}

/// Hands a clone of `model` to the agent while the test keeps the original.
fn shared(model: &ScriptedModel) -> Arc<dyn Model> {
    Arc::new(model.clone())
}

#[tokio::test]
async fn invoke_runs_the_single_workflow() {
    let agent = agent("wf-single");
    agent.add_workflows([echo_workflow("echo", "answer")]);

    let output = agent.invoke(json!({"query": "hi"}), None).await.unwrap();

    assert_eq!(output.result_type, ResultType::Answer);
    assert_eq!(output.data.as_ref().unwrap()["answer"], json!("hi"));
    assert_eq!(
        output.workflow_output.as_ref().unwrap().state,
        WorkflowExecutionState::Completed
    );
    let parsed: Value = serde_json::from_str(&output.output).unwrap();
    assert_eq!(parsed["answer"], json!("hi"));
}

#[tokio::test]
async fn stream_forwards_the_final_frame_and_one_terminal_answer() {
    let agent = agent("wf-stream");
    agent.add_workflows([echo_workflow("echo", "answer")]);
    let session = common::session("wf-stream-agent");
    // Subscribe before the run so the engine's frames are not missed.
    let mut events = session.subscribe();

    let chunks = agent
        .stream(json!({"query": "hi"}), Some(Arc::clone(&session)))
        .map(|chunk| chunk.unwrap())
        .collect::<Vec<_>>()
        .await;

    let finals = chunks
        .iter()
        .filter(|chunk| chunk.kind == WORKFLOW_FINAL)
        .collect::<Vec<_>>();
    assert_eq!(finals.len(), 1, "{chunks:?}");
    assert_eq!(finals[0].payload, json!({"answer": "hi"}));

    let answers = chunks
        .iter()
        .filter(|chunk| chunk.kind == ANSWER)
        .collect::<Vec<_>>();
    assert_eq!(answers.len(), 1, "exactly one terminal answer: {chunks:?}");
    assert_eq!(chunks.last().unwrap().kind, ANSWER, "answer is terminal");
    let terminal = &answers[0].payload;
    assert_eq!(terminal["result_type"], "answer");
    assert!(
        terminal["output"].as_str().unwrap().contains("hi"),
        "{terminal}"
    );
    assert_eq!(
        terminal["workflow_execution_state"]["state"],
        json!("COMPLETED")
    );

    // The engine writes its own frames to the session hub (shared with the
    // child workflow session) as it runs. The terminal `workflow_final` frame
    // is synthesized by the compiled workflow's stream and is not re-written to
    // the hub, so the drain below looks for the engine's graph frames instead.
    let mut kinds = Vec::new();
    let deadline = tokio::time::Instant::now() + Duration::from_millis(500);
    loop {
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        if remaining.is_zero() {
            break;
        }
        match tokio::time::timeout(remaining, events.next()).await {
            Ok(Some(Ok(StreamEvent::Item(item)))) => {
                let engine_frame = item.item_type == "graph_event";
                kinds.push(item.item_type);
                if engine_frame {
                    break;
                }
            }
            Ok(Some(Ok(StreamEvent::End)) | None) | Err(_) => break,
            Ok(Some(Err(error))) => panic!("session stream failed: {error}"),
        }
    }
    assert!(
        kinds.iter().any(|kind| kind == "graph_event"),
        "the engine's frames never reached the session hub: {kinds:?}"
    );
}

#[tokio::test]
async fn selection_drives_the_second_workflow_and_falls_back_to_the_default_response() {
    let model = ScriptedModel::text(r#"{"result": 2}"#);
    let agent = agent("wf-select");
    agent.add_workflows([
        described_workflow("first", "first", "answer"),
        described_workflow("second", "second", "second"),
    ]);
    agent.set_model(shared(&model));

    let output = agent.invoke(json!({"query": "hi"}), None).await.unwrap();

    assert_eq!(output.result_type, ResultType::Answer);
    let data = output.data.as_ref().unwrap();
    assert_eq!(data["second"], json!("hi"), "{data}");
    assert!(data.get("answer").is_none(), "selected the first workflow");
    assert!(model.call_count() >= 1, "intent detection never ran");

    let mut config = WorkflowAgentConfig::new("wf-default");
    config.default_response.text = Some("fallback".to_owned());
    let fallback_agent = WorkflowAgent::new(config);
    fallback_agent.add_workflows([
        described_workflow("first", "first", "answer"),
        described_workflow("second", "second", "second"),
    ]);
    let garbage = ScriptedModel::text("no idea");
    fallback_agent.set_model(shared(&garbage));

    let output = fallback_agent
        .invoke(json!({"query": "hi"}), None)
        .await
        .unwrap();

    assert_eq!(output.result_type, ResultType::Answer);
    assert_eq!(output.output, "fallback");
    assert_eq!(output.extra["status"], json!("default_response"));
    assert_eq!(
        output.data.as_ref().unwrap()["status"],
        json!("default_response")
    );
}

/// Asks the human a question through `ComponentContext::interact`.
#[derive(Debug, Clone)]
struct InteractingComponent {
    question: Value,
}

impl InteractingComponent {
    fn new(question: Value) -> Self {
        Self { question }
    }
}

#[async_trait]
impl Component for InteractingComponent {
    fn card(&self) -> ComponentCard {
        ComponentCard::new("ask", "ask").with_abilities(self.abilities())
    }

    fn abilities(&self) -> Vec<ComponentAbility> {
        vec![ComponentAbility::Invoke]
    }

    fn component_type(&self) -> &'static str {
        "ask"
    }

    async fn invoke(&self, _inputs: Value, ctx: &ComponentContext) -> Result<Value> {
        let answer = ctx.interact(self.question.clone()).await?;
        Ok(json!({"answer": answer}))
    }
}

/// `start → ask → end`; the end node flattens the answer.
fn asking_workflow(id: &str) -> Arc<Workflow> {
    let mut workflow = Workflow::new(WorkflowCard::new(id));
    workflow
        .set_start_comp(
            "start",
            StartComponent::new(),
            ComponentOptions::new().with_inputs_schema(json!({"query": "${inputs.query}"})),
        )
        .add_workflow_comp(
            "ask",
            InteractingComponent::new(json!({"id": "ask", "question": "go?"})),
            ComponentOptions::new(),
        )
        .set_end_comp(
            "end",
            EndComponent::new(),
            ComponentOptions::new()
                .with_inputs_schema(json!({"answer": "${ask.answer}"}))
                .with_outputs_schema(json!({"answer": "${output.output.answer}"})),
        )
        .add_connection("start", "ask")
        .add_connection("ask", "end");
    Arc::new(workflow)
}

/// An `AgentSession` whose snapshots go to `checkpointer`.
fn checkpointed_session(id: &str, checkpointer: MemoryCheckpointer) -> Arc<dyn Session> {
    Arc::new(AgentSession::with_options(AgentSessionOptions {
        session_id: Some(id.to_owned()),
        checkpointer: Some(Arc::new(checkpointer) as Arc<dyn Checkpointer>),
        ..AgentSessionOptions::default()
    }))
}

#[tokio::test]
async fn an_interrupt_reports_input_required_and_resumes_with_the_answer() {
    let agent = agent("wf-hitl");
    agent.add_workflows([asking_workflow("ask")]);
    let session = checkpointed_session("hitl-agent", MemoryCheckpointer::new());

    let paused = agent
        .invoke(json!({"query": "hi"}), Some(Arc::clone(&session)))
        .await
        .unwrap();

    assert_eq!(paused.result_type, ResultType::Interrupt, "{paused:?}");
    assert_eq!(
        paused.workflow_output.as_ref().unwrap().state,
        WorkflowExecutionState::InputRequired
    );
    let frames = paused
        .state
        .iter()
        .filter(|frame| frame.kind == "__interaction__")
        .collect::<Vec<_>>();
    assert_eq!(frames.len(), 1, "{:?}", paused.state);
    assert_eq!(frames[0].index, 0);
    assert_eq!(frames[0].payload["id"], "ask");
    assert_eq!(paused.interrupt_ids, ["ask"]);
    // The workflow projection reports the interaction id as the interrupt id;
    // `AgentOutput::interrupt` leaves the component-id list empty.
    assert!(
        paused.component_ids.is_empty(),
        "{:?}",
        paused.component_ids
    );

    let resumed = agent
        .invoke(
            json!({"query": {"user_inputs": {"ask": "yes"}}}),
            Some(Arc::clone(&session)),
        )
        .await
        .unwrap();

    assert_eq!(resumed.result_type, ResultType::Answer, "{resumed:?}");
    assert_eq!(resumed.data.as_ref().unwrap()["answer"], json!("yes"));
    assert_eq!(
        resumed.workflow_output.as_ref().unwrap().state,
        WorkflowExecutionState::Completed
    );
}

#[tokio::test]
async fn invoke_without_workflows_is_a_workflow_error() {
    let agent = agent("wf-empty");

    let error = agent
        .invoke(json!({"query": "hi"}), None)
        .await
        .unwrap_err();

    assert_eq!(error.kind(), ErrorKind::Workflow);
    assert!(error.to_string().contains("No workflows configured"));
}
