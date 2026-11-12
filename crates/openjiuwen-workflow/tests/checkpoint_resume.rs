#![allow(clippy::unwrap_used, clippy::expect_used)]
#![allow(
    unreachable_pub,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    clippy::new_without_default,
    clippy::module_name_repetitions,
    clippy::unnecessary_literal_bound
)]
//! Checkpoint-backed resume: failure recovery, human-in-the-loop, and snapshot lifecycle.
//!
//! Ports of the Python unit tests in
//! `tests/unit_tests/core/workflow/test_workflow_with_interrupt.py` and
//! `tests/unit_tests/core/session/checkpointer/test_inmemory_checkpointer.py`, rewritten
//! against the Rust public API with `MemoryCheckpointer` as the store. Everything is
//! offline and deterministic: no sleeps, no network, no stdout.
mod common;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_common::error::{Error, ErrorKind, Result};
use openjiuwen_session::{Checkpointer, InteractiveInput, MemoryCheckpointer, Session};
use serde_json::{json, Value};

use openjiuwen_workflow::{
    Component, ComponentAbility, ComponentCard, ComponentContext, ComponentOptions, EndComponent,
    StartComponent, Workflow, WorkflowExecutionState, WorkflowRunOptions,
};

use common::{card, session_with_checkpointer};

/// Fails its first invocation and answers `{"call": n}` afterwards.
#[derive(Debug, Clone, Default)]
struct FailOnceComponent {
    calls: Arc<AtomicUsize>,
}

impl FailOnceComponent {
    /// Creates a component that fails exactly once.
    fn new() -> Self {
        Self::default()
    }

    /// How many times the component ran, successful or not.
    fn count(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl Component for FailOnceComponent {
    fn card(&self) -> ComponentCard {
        ComponentCard::new("fail_once", "fail_once").with_abilities(self.abilities())
    }

    fn abilities(&self) -> Vec<ComponentAbility> {
        vec![ComponentAbility::Invoke]
    }

    fn component_type(&self) -> &str {
        "fail_once"
    }

    async fn invoke(&self, _inputs: Value, _ctx: &ComponentContext) -> Result<Value> {
        let attempt = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
        if attempt == 1 {
            return Err(Error::workflow(format!("attempt {attempt} failed")));
        }
        Ok(json!({"call": attempt}))
    }
}

/// `start → mid → boom → end`; `mid` records its calls, `boom` is the fallible node.
fn failing_workflow(id: &str, mid: &common::FixedComponent, boom: &FailOnceComponent) -> Workflow {
    let mut workflow = Workflow::new(card(id));
    workflow
        .set_start_comp("start", StartComponent::new(), ComponentOptions::new())
        .add_workflow_comp("mid", mid.clone(), ComponentOptions::new())
        .add_workflow_comp("boom", boom.clone(), ComponentOptions::new())
        .set_end_comp(
            "end",
            EndComponent::new(),
            ComponentOptions::new().with_inputs_schema(json!({
                "mid": "${mid.ready}",
                "boom": "${boom.call}",
            })),
        )
        .add_connection("start", "mid")
        .add_connection("mid", "boom")
        .add_connection("boom", "end");
    workflow
}

/// `start → ask → end`; `ask` asks the human and reports the answer under `answer`.
fn asking_workflow(id: &str) -> Workflow {
    let mut workflow = Workflow::new(card(id));
    workflow
        .set_start_comp("start", StartComponent::new(), ComponentOptions::new())
        .add_workflow_comp(
            "ask",
            common::InteractingComponent::new(json!({"id": "ask", "question": "go?"})),
            ComponentOptions::new(),
        )
        .set_end_comp(
            "end",
            EndComponent::new(),
            ComponentOptions::new().with_inputs_schema(json!({"answer": "${ask.answer}"})),
        )
        .add_connection("start", "ask")
        .add_connection("ask", "end");
    workflow
}

/// `start(counter) → ask → end`, so the start count proves nothing was replayed twice.
fn counted_asking_workflow(id: &str, counter: &common::CounterComponent) -> Workflow {
    let mut workflow = Workflow::new(card(id));
    workflow
        .set_start_comp("start", counter.clone(), ComponentOptions::new())
        .add_workflow_comp(
            "ask",
            common::InteractingComponent::new(json!({"id": "ask", "question": "go?"})),
            ComponentOptions::new(),
        )
        .set_end_comp(
            "end",
            EndComponent::new(),
            ComponentOptions::new().with_inputs_schema(json!({"answer": "${ask.answer}"})),
        )
        .add_connection("start", "ask")
        .add_connection("ask", "end");
    workflow
}

/// A session checkpointed through `checkpointer`.
fn checkpointed(id: &str, checkpointer: &MemoryCheckpointer) -> Arc<dyn Session> {
    session_with_checkpointer(id, Arc::new(checkpointer.clone()) as Arc<dyn Checkpointer>)
}

/// An answering `InteractiveInput` for the `ask` question.
fn answer_yes() -> InteractiveInput {
    let mut interactive = InteractiveInput::new();
    interactive.answer("ask", json!("yes"));
    interactive
}

// port of tests/unit_tests/core/workflow/test_workflow.py::test_two_iterative_node_and_recover_each
#[tokio::test]
async fn a_resumed_run_does_not_re_execute_completed_nodes() {
    let checkpointer = MemoryCheckpointer::new();
    let session = checkpointed("resume-failure", &checkpointer);
    let mid = common::FixedComponent::new(json!({"ready": true}));
    let boom = FailOnceComponent::new();
    let workflow = failing_workflow("resume-failure", &mid, &boom);

    let failure = workflow
        .invoke(json!({"query": "hi"}), Arc::clone(&session))
        .await
        .unwrap_err();
    assert_eq!(failure.kind(), ErrorKind::Workflow, "{failure}");
    assert_eq!(mid.call_count(), 1);
    assert_eq!(boom.count(), 1);
    assert_eq!(
        checkpointer.len(session.id()),
        1,
        "a failed run keeps its pre-superstep snapshot"
    );

    let resumed = workflow
        .invoke_with(
            json!({}),
            Arc::clone(&session),
            WorkflowRunOptions::default().with_resume(true),
        )
        .await
        .unwrap();

    assert_eq!(resumed.state, WorkflowExecutionState::Completed);
    assert_eq!(
        mid.call_count(),
        1,
        "the resumed run must not re-execute completed nodes"
    );
    assert_eq!(boom.count(), 2, "only the failing node is retried");
    assert_eq!(
        resumed.result,
        json!({"output": {"mid": true, "boom": 2}}),
        "the resumed run keeps the completed node's output"
    );
    assert_eq!(checkpointer.len(session.id()), 0);
}

// port of tests/unit_tests/core/workflow/test_workflow_with_interrupt.py::test_simple_interactive_workflow_checkpointer
#[tokio::test]
async fn an_interrupt_reports_input_required_and_resumes_with_the_answer() {
    let checkpointer = MemoryCheckpointer::new();
    let session = checkpointed("hitl", &checkpointer);
    let workflow = asking_workflow("hitl");

    let first = workflow
        .invoke(json!({"query": "hi"}), Arc::clone(&session))
        .await
        .unwrap();
    assert_eq!(first.state, WorkflowExecutionState::InputRequired);
    assert_eq!(
        first.result,
        json!([{
            "type": "__interaction__",
            "index": 0,
            "payload": {"id": "ask", "index": 0, "question": {"id": "ask", "question": "go?"}}
        }])
    );
    assert_eq!(checkpointer.len(session.id()), 1);

    let resumed = workflow
        .resume(answer_yes(), Arc::clone(&session))
        .await
        .unwrap();

    assert_eq!(resumed.state, WorkflowExecutionState::Completed);
    assert_eq!(resumed.result, json!({"output": {"answer": "yes"}}));
    assert_eq!(
        checkpointer.len(session.id()),
        0,
        "the resumed run releases its graph snapshot"
    );
}

// port of tests/unit_tests/core/workflow/test_workflow_with_interrupt.py::test_simple_interactive_workflow
#[tokio::test]
async fn resume_without_a_checkpointer_is_a_workflow_error() {
    let workflow = asking_workflow("hitl-no-checkpointer");

    let failure = workflow
        .resume(answer_yes(), common::session("hitl-no-checkpointer"))
        .await
        .unwrap_err();

    assert_eq!(failure.kind(), ErrorKind::Workflow);
    assert_eq!(
        failure.message(),
        "interactive resume requested without a checkpointer"
    );
}

// port of tests/unit_tests/core/workflow/test_workflow_with_interrupt.py::test_simple_interactive_workflow_clear_checkpointer
#[tokio::test]
async fn a_completed_run_releases_its_snapshot_and_an_interrupt_keeps_one() {
    let checkpointer = MemoryCheckpointer::new();
    let session = checkpointed("snapshot-lifecycle", &checkpointer);

    let mut completed_workflow = Workflow::new(card("snapshot-lifecycle"));
    completed_workflow
        .set_start_comp("start", StartComponent::new(), ComponentOptions::new())
        .set_end_comp("end", EndComponent::new(), ComponentOptions::new())
        .add_connection("start", "end");
    let output = completed_workflow
        .invoke(json!({"query": "done"}), Arc::clone(&session))
        .await
        .unwrap();
    assert_eq!(output.state, WorkflowExecutionState::Completed);
    assert!(
        checkpointer.list(session.id()).await.unwrap().is_empty(),
        "a completed run releases its graph snapshot"
    );
    assert_eq!(checkpointer.len(session.id()), 0);

    let interrupted = asking_workflow("snapshot-lifecycle")
        .invoke(json!({"query": "hi"}), Arc::clone(&session))
        .await
        .unwrap();
    assert_eq!(interrupted.state, WorkflowExecutionState::InputRequired);
    assert_eq!(
        checkpointer.list(session.id()).await.unwrap().len(),
        1,
        "an interrupted run keeps exactly one pre-superstep snapshot"
    );
}

// port of tests/unit_tests/core/workflow/test_workflow_with_interrupt.py::test_simple_interactive_workflow_raw_input
#[tokio::test]
async fn seeded_answers_complete_a_fresh_run_without_a_snapshot() {
    let checkpointer = MemoryCheckpointer::new();
    let session = checkpointed("seeded-answers", &checkpointer);

    let output = asking_workflow("seeded-answers")
        .invoke_with(
            json!({"query": "hi"}),
            Arc::clone(&session),
            WorkflowRunOptions::default().with_interactive(answer_yes()),
        )
        .await
        .unwrap();

    assert_eq!(output.state, WorkflowExecutionState::Completed);
    assert_eq!(output.result, json!({"output": {"answer": "yes"}}));
    assert_eq!(
        checkpointer.len(session.id()),
        0,
        "a seeded run never stores a snapshot"
    );
}

// port of tests/unit_tests/core/workflow/test_workflow_with_interrupt.py::test_simple_interactive_workflow_checkpointer
#[tokio::test]
async fn a_resumed_hitl_run_replays_only_the_interrupted_superstep() {
    let checkpointer = MemoryCheckpointer::new();
    let session = checkpointed("hitl-superstep", &checkpointer);
    let counter = common::CounterComponent::new();
    let workflow = counted_asking_workflow("hitl-superstep", &counter);

    let first = workflow
        .invoke(json!({"query": "hi"}), Arc::clone(&session))
        .await
        .unwrap();
    assert_eq!(first.state, WorkflowExecutionState::InputRequired);
    assert_eq!(
        counter.count(),
        1,
        "the start node ran once before the halt"
    );

    let resumed = workflow
        .resume(answer_yes(), Arc::clone(&session))
        .await
        .unwrap();

    assert_eq!(resumed.state, WorkflowExecutionState::Completed);
    assert_eq!(resumed.result, json!({"output": {"answer": "yes"}}));
    assert_eq!(
        counter.count(),
        1,
        "resume replays the interrupted superstep, not the completed start node"
    );
}
