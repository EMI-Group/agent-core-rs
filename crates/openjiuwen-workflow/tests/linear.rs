#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Linear workflows: `start → … → end` with a scripted model, session state, cancellation,
//! build failures, and concurrent runs.
//!
//! These are ports of the Python unit tests in
//! `tests/unit_tests/core/workflow/test_workflow.py`, rewritten against the Rust public
//! API (`Workflow`, `ComponentOptions`, `WorkflowOutput`). Every test is offline and
//! deterministic: the only model is `ScriptedModel` and sessions are in-process
//! `WorkflowSession`s.
mod common;

use std::sync::Arc;

use openjiuwen_common::error::ErrorKind;
use openjiuwen_session::Session;
use openjiuwen_test_support::ScriptedModel;
use serde_json::json;
use tokio_util::sync::CancellationToken;

use openjiuwen_workflow::{
    ComponentOptions, EndComponent, StartComponent, Workflow, WorkflowExecutionState,
    WorkflowRunOptions,
};

use common::{card, llm_component, session};

/// `start → llm → end`, where the LLM answers from `model` and the end node reports the
/// answer under `answer`.
fn linear_workflow(model: Arc<dyn openjiuwen_foundation::Model>, user: &str) -> Workflow {
    let mut workflow = Workflow::new(card("linear"));
    workflow
        .set_start_comp(
            "start",
            StartComponent::new(),
            ComponentOptions::new().with_inputs_schema(json!({"query": "${inputs.query}"})),
        )
        .add_workflow_comp(
            "llm",
            llm_component(model, user),
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

// port of tests/unit_tests/core/workflow/test_workflow.py::test_simple_workflow
#[tokio::test]
async fn start_llm_end_runs_to_completion() {
    let model = Arc::new(ScriptedModel::builder().respond_text("hello there").build());
    let workflow = linear_workflow(model.clone(), "Answer the query: {{query}}");

    let output = workflow
        .invoke(json!({"query": "hello"}), session("linear-simple"))
        .await
        .unwrap();

    assert_eq!(output.state, WorkflowExecutionState::Completed);
    assert_eq!(output.result, json!({"output": {"answer": "hello there"}}));
    assert_eq!(
        model.call_count(),
        1,
        "the model must be called exactly once"
    );
    let call = model.last_call().unwrap();
    assert!(
        call.last_user_text().unwrap().contains("hello"),
        "unexpected user text: {:?}",
        call.last_user_text()
    );
}

// port of tests/unit_tests/core/workflow/test_workflow.py::test_workflow_inputs_in_session
#[tokio::test]
async fn run_inputs_land_in_the_session_global_lane() {
    let model = Arc::new(ScriptedModel::builder().respond_text("done").build());
    let workflow = linear_workflow(model, "{{query}}");
    let session: Arc<dyn Session> = session("linear-session");

    workflow
        .invoke(json!({"query": "hello"}), Arc::clone(&session))
        .await
        .unwrap();

    // Python's `commit_user_inputs`: the object inputs are merged into the global lane.
    assert_eq!(
        session.get_state("query").await.unwrap(),
        Some(json!("hello"))
    );
}

// port of tests/unit_tests/core/workflow/test_workflow.py::test_three_node_workflow
#[tokio::test]
async fn a_middle_component_composes_into_the_llm_and_the_result() {
    let model = Arc::new(ScriptedModel::builder().respond_text("composed").build());
    let middle = common::FixedComponent::new(json!({"question": "mid says hi"}));

    let mut workflow = Workflow::new(card("composed"));
    workflow
        .set_start_comp(
            "start",
            StartComponent::new(),
            ComponentOptions::new().with_inputs_schema(json!({"query": "${inputs.query}"})),
        )
        .add_workflow_comp(
            "mid",
            middle.clone(),
            ComponentOptions::new().with_inputs_schema(json!({"query": "${start.query}"})),
        )
        .add_workflow_comp(
            "llm",
            llm_component(model.clone(), "{{query}}"),
            ComponentOptions::new().with_inputs_schema(json!({"query": "${mid.question}"})),
        )
        .set_end_comp(
            "end",
            EndComponent::new(),
            ComponentOptions::new().with_inputs_schema(json!({
                "answer": "${llm.result}",
                "mid": "${mid.question}",
            })),
        )
        .add_connection("start", "mid")
        .add_connection("mid", "llm")
        .add_connection("llm", "end");

    let output = workflow
        .invoke(json!({"query": "hello"}), session("linear-middle"))
        .await
        .unwrap();

    assert_eq!(middle.calls(), vec![json!({"query": "hello"})]);
    assert_eq!(
        model.last_call().unwrap().last_user_text().unwrap(),
        "mid says hi",
        "the llm must receive the middle component's output"
    );
    assert_eq!(
        output.result,
        json!({"output": {"answer": "composed", "mid": "mid says hi"}})
    );
}

// port of tests/unit_tests/core/workflow/test_workflow.py::test_workflow_cancelled
#[tokio::test]
async fn a_cancelled_token_fails_the_run_before_any_node_runs() {
    let model = Arc::new(ScriptedModel::builder().respond_text("never").build());
    let workflow = linear_workflow(model.clone(), "{{query}}");

    let cancellation = CancellationToken::new();
    cancellation.cancel();
    let options = WorkflowRunOptions {
        cancellation,
        ..WorkflowRunOptions::default()
    };
    let failure = workflow
        .invoke_with(json!({"query": "hello"}), session("linear-cancel"), options)
        .await
        .unwrap_err();

    assert_eq!(failure.kind(), ErrorKind::Cancelled);
    assert_eq!(
        model.call_count(),
        0,
        "a cancelled run must not reach the model"
    );
}

// port of tests/unit_tests/core/workflow/test_workflow.py::test_invalid_workflow
#[tokio::test]
async fn a_workflow_without_endpoints_fails_to_build() {
    let workflow = Workflow::new(card("incomplete"));
    let failure = workflow.build().unwrap_err();

    assert_eq!(failure.kind(), ErrorKind::Workflow);
    let message = failure.to_string();
    assert!(message.contains("no start component"), "{message}");
    assert!(message.contains("no end component"), "{message}");
}

// port of tests/unit_tests/core/workflow/test_workflow.py::test_concurrent_invocations
#[tokio::test]
async fn one_workflow_runs_concurrently() {
    let model = Arc::new(
        ScriptedModel::builder()
            .respond_text("first")
            .respond_text("second")
            .build(),
    );
    let workflow = linear_workflow(model.clone(), "{{query}}");

    let (first, second) = tokio::join!(
        workflow.invoke(json!({"query": "one"}), session("linear-a")),
        workflow.invoke(json!({"query": "two"}), session("linear-b")),
    );

    let first = first.unwrap();
    let second = second.unwrap();
    assert_eq!(first.state, WorkflowExecutionState::Completed);
    assert_eq!(second.state, WorkflowExecutionState::Completed);
    assert_eq!(model.call_count(), 2);

    // The model queue is shared, so the interleaving decides which run sees which turn;
    // the set of answers is what the builder's immutability guarantees.
    let answers = [
        first.result["output"]["answer"].clone(),
        second.result["output"]["answer"].clone(),
    ];
    assert!(answers.contains(&json!("first")), "{answers:?}");
    assert!(answers.contains(&json!("second")), "{answers:?}");
}
