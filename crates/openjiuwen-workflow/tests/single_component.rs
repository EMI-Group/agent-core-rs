#![allow(clippy::unwrap_used, clippy::expect_used)]
#![allow(
    unreachable_pub,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    clippy::new_without_default,
    clippy::module_name_repetitions,
    clippy::unnecessary_literal_bound
)]
//! `execute_single_component`: one component driven outside a workflow graph.
//!
//! Ports `tests/unit_tests/core/workflow/test_workflow.py::test_single_component_execution`
//! and `::test_executor_single_interupt_component`, plus the `generate_workflow_key` and
//! `WorkflowChunkType` exports of Python's `openjiuwen.core.workflow` package.
//!
//! Everything is offline and deterministic: an in-process `WorkflowSession`, no sleeps,
//! no network, no wall-clock reads. Each suite names the Python test it mirrors where one
//! exists.

use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_common::error::{Error, ErrorKind, Result};
use openjiuwen_session::interaction::INTERACTION_FRAME_TYPE;
use openjiuwen_session::{InteractiveInput, Session, WorkflowSession};
use serde_json::{json, Value};

use openjiuwen_workflow::{
    execute_single_component, execute_single_component_with, generate_workflow_key, Component,
    ComponentCard, ComponentContext, SingleComponentOptions, WorkflowChunkType,
    WorkflowExecutionState,
};

/// A fresh in-process session with no checkpointer.
fn session() -> Arc<dyn Session> {
    Arc::new(WorkflowSession::new("single-component-tests"))
}

/// An `InteractiveInput` carrying one interrupt-addressed answer.
fn seeded(id: &str, value: Value) -> InteractiveInput {
    let mut input = InteractiveInput::new();
    input.answer(id, value);
    input
}

/// Greets the `q` input, so template resolution is observable in the output.
struct GreetingComponent;

#[async_trait]
impl Component for GreetingComponent {
    fn card(&self) -> ComponentCard {
        ComponentCard::new("work", "greeting")
    }

    fn component_type(&self) -> &str {
        "greeting"
    }

    async fn invoke(&self, inputs: Value, _ctx: &ComponentContext) -> Result<Value> {
        let name = inputs.get("q").and_then(Value::as_str).unwrap_or_default();
        Ok(json!({"greeting": format!("hi {name}")}))
    }
}

/// Returns its inputs unchanged, so a pass-through call is observable.
struct EchoComponent;

#[async_trait]
impl Component for EchoComponent {
    fn card(&self) -> ComponentCard {
        ComponentCard::new("echo", "echo")
    }

    fn component_type(&self) -> &str {
        "echo"
    }

    async fn invoke(&self, inputs: Value, _ctx: &ComponentContext) -> Result<Value> {
        Ok(json!({"echo": inputs}))
    }
}

/// Asks one question before answering, the Python interrupt fixture.
struct AskingComponent;

#[async_trait]
impl Component for AskingComponent {
    fn card(&self) -> ComponentCard {
        ComponentCard::new("ask", "asking")
    }

    fn component_type(&self) -> &str {
        "asking"
    }

    async fn invoke(&self, _inputs: Value, ctx: &ComponentContext) -> Result<Value> {
        let answer = ctx
            .interact(json!({"id": "q1", "text": "continue?"}))
            .await?;
        Ok(json!({"answer": answer}))
    }
}

/// Fails its invocation.
struct FailingComponent;

#[async_trait]
impl Component for FailingComponent {
    fn card(&self) -> ComponentCard {
        ComponentCard::new("failing", "failing")
    }

    fn component_type(&self) -> &str {
        "failing"
    }

    async fn invoke(&self, _inputs: Value, _ctx: &ComponentContext) -> Result<Value> {
        Err(Error::workflow("boom"))
    }
}

// port of tests/unit_tests/core/workflow/test_workflow.py::test_single_component_execution
// (Python resolves `{"result": "${result}"}` against the node's own outputs; this crate's
// convention is `${output.field}` / `${<component_id>.field}` instead).
#[tokio::test]
async fn single_component_resolves_inputs_and_projects_outputs() {
    let output = execute_single_component_with(
        "work",
        Arc::new(GreetingComponent),
        json!({"source": {"query": "hello"}, "unused": true}),
        session(),
        SingleComponentOptions::new()
            .with_inputs_schema(json!({"q": "${source.query}"}))
            .with_outputs_schema(json!({
                "answer": "${output.greeting}",
                "seen": "${work.greeting}",
            })),
    )
    .await
    .unwrap();

    assert_eq!(
        output.result,
        json!({"answer": "hi hello", "seen": "hi hello"})
    );
    assert_eq!(output.state, WorkflowExecutionState::Completed);
}

#[tokio::test]
async fn single_component_without_schemas_passes_inputs_through() {
    let inputs = json!({"a": 1, "nested": {"b": [1, 2]}});
    let output =
        execute_single_component("echo", Arc::new(EchoComponent), inputs.clone(), session())
            .await
            .unwrap();

    assert_eq!(output.result, json!({"echo": inputs}));
    assert_eq!(output.state, WorkflowExecutionState::Completed);
}

// port of tests/unit_tests/core/workflow/test_workflow.py::test_executor_single_interupt_component
#[tokio::test]
async fn single_component_interrupt_reports_the_question() {
    let output = execute_single_component(
        "ask",
        Arc::new(AskingComponent),
        json!({"a": "input"}),
        session(),
    )
    .await
    .unwrap();

    assert_eq!(output.state, WorkflowExecutionState::InputRequired);
    assert_eq!(
        output.result,
        json!([{
            "type": INTERACTION_FRAME_TYPE,
            "index": 0,
            "payload": {
                "id": "q1",
                "index": 0,
                "question": {"id": "q1", "text": "continue?"},
            },
        }])
    );
}

// port of tests/unit_tests/core/workflow/test_workflow.py::test_executor_single_interupt_component
// (second call: the seeded `InteractiveInput` answers the question)
#[tokio::test]
async fn single_component_seeded_answer_completes_the_call() {
    let output = execute_single_component_with(
        "ask",
        Arc::new(AskingComponent),
        json!({"a": "input"}),
        session(),
        SingleComponentOptions::new().with_interactive(seeded("q1", json!("yes"))),
    )
    .await
    .unwrap();

    assert_eq!(output.state, WorkflowExecutionState::Completed);
    assert_eq!(output.result, json!({"answer": "yes"}));
}

#[tokio::test]
async fn single_component_error_propagates() {
    let error =
        execute_single_component("failing", Arc::new(FailingComponent), json!({}), session())
            .await
            .unwrap_err();

    assert_eq!(error.kind(), ErrorKind::Workflow);
    assert!(
        error.message().contains("boom"),
        "unexpected message: {}",
        error.message()
    );
}

#[test]
fn generate_workflow_key_matches_python() {
    // Python: openjiuwen/core/workflow/base.py::generate_workflow_key
    assert_eq!(generate_workflow_key("wf", "1.0.0"), "wf_1.0.0");
    assert_eq!(generate_workflow_key("wf", ""), "wf_");
}

#[test]
fn chunk_type_wire_values_match_python() {
    // Python: openjiuwen/core/workflow/base.py::WorkflowChunkType
    assert_eq!(
        serde_json::to_value(WorkflowChunkType::Interaction).unwrap(),
        json!("interaction")
    );
    assert_eq!(
        serde_json::to_value(WorkflowChunkType::Output).unwrap(),
        json!("output")
    );
    assert_eq!(
        serde_json::to_value(WorkflowChunkType::Error).unwrap(),
        json!("error")
    );
    assert_eq!(WorkflowChunkType::Interaction.to_string(), "interaction");
    assert_eq!(WorkflowChunkType::Error.as_str(), "error");

    let parsed: WorkflowChunkType = serde_json::from_value(json!("output")).unwrap();
    assert_eq!(parsed, WorkflowChunkType::Output);
}
