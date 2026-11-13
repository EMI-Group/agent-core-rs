#![allow(clippy::unwrap_used, clippy::expect_used)]
#![allow(
    unreachable_pub,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    clippy::new_without_default,
    clippy::module_name_repetitions,
    clippy::unnecessary_literal_bound
)]
//! Nested workflows: `SubWorkflowComponent` drives a compiled child workflow as one
//! component of a parent workflow.
//!
//! The tests pin the contract the Python component has
//! (`tests/unit_tests/core/test_sub_workflow_comp.py`): the child's end output is returned
//! unchanged, a fresh child session keeps node state lanes apart even when parent and
//! child reuse the same ids, the nesting limit is enforced, child frames are forwarded as
//! wires, the optional stream cache answers later `invoke` calls, and a child
//! human-in-the-loop question surfaces as the parent's `INPUT_REQUIRED` interaction —
//! whose answer a parent `resume` forwards into the child run (by bare or dotted
//! interaction id), so the resumed child finishes instead of asking again.
//!
//! Everything is offline and deterministic: no sleeps, no network, no wall-clock reads.
mod common;

use std::sync::Arc;

use futures::StreamExt;
use openjiuwen_common::error::ErrorKind;
use openjiuwen_session::interaction::INTERACTION_FRAME_TYPE;
use openjiuwen_session::{Checkpointer, InteractiveInput, MemoryCheckpointer};
use serde_json::{json, Value};

use openjiuwen_workflow::components::questioner::{QuestionerComponent, QuestionerConfig};
use openjiuwen_workflow::components::sub_workflow::{SubWorkflowComponent, NESTED_ANSWERS_KEY};
use openjiuwen_workflow::{
    Component, ComponentContext, ComponentOptions, EndComponent, StartComponent, Workflow,
    WorkflowCard, WorkflowExecutionState, END_NODE_STREAM,
};

use common::{
    card, end_with_template, session, session_with_checkpointer, FixedComponent,
    InteractingComponent, StreamingProducer,
};

/// A child workflow ending with a batch `end` that projects `work.answer`.
fn linear_child(work: FixedComponent) -> Workflow {
    let mut child = Workflow::new(WorkflowCard::new("child"));
    child
        .set_start_comp("start", StartComponent::new(), ComponentOptions::new())
        .add_workflow_comp("work", work, ComponentOptions::new())
        .set_end_comp(
            "end",
            EndComponent::new(),
            ComponentOptions::new().with_inputs_schema(json!({"answer": "${work.answer}"})),
        )
        .add_connection("start", "work")
        .add_connection("work", "end");
    child
}

// port of tests/unit_tests/core/component/test_sub_workflow_comp.py::test_sub_invoke
#[tokio::test]
async fn a_parent_invokes_a_linear_child_and_reads_its_output() {
    let work = FixedComponent::new(json!({"answer": "child"}));
    let sub = SubWorkflowComponent::new(&linear_child(work.clone())).unwrap();

    let mut parent = Workflow::new(card("parent"));
    parent
        .set_start_comp("start", StartComponent::new(), ComponentOptions::new())
        .add_workflow_comp("sub", sub, ComponentOptions::new())
        .set_end_comp(
            "end",
            EndComponent::new(),
            ComponentOptions::new().with_inputs_schema(json!({"answer": "${sub.output.answer}"})),
        )
        .add_connection("start", "sub")
        .add_connection("sub", "end");

    let output = parent
        .invoke(json!({"query": "hi"}), session("nested-linear"))
        .await
        .unwrap();

    assert_eq!(output.state, WorkflowExecutionState::Completed);
    // The child end yielded `{"output": {"answer": "child"}}`; the parent end read it back
    // through `${sub.output.answer}`.
    assert_eq!(output.result, json!({"output": {"answer": "child"}}));
    assert_eq!(work.call_count(), 1);
}

// port of tests/unit_tests/core/component/test_sub_workflow_comp.py::test_sub_invoke
#[tokio::test]
async fn parent_and_child_may_share_component_ids() {
    let work = FixedComponent::new(json!({"answer": "child"}));
    // The child uses the ids `start`, `work`, `end`; so does the parent, whose sub node
    // is also called `work`.
    let mut child = Workflow::new(WorkflowCard::new("child"));
    child
        .set_start_comp("start", StartComponent::new(), ComponentOptions::new())
        .add_workflow_comp("work", work.clone(), ComponentOptions::new())
        .set_end_comp(
            "end",
            EndComponent::new(),
            ComponentOptions::new().with_inputs_schema(json!({"answer": "${work.answer}"})),
        )
        .add_connection("start", "work")
        .add_connection("work", "end");
    let sub = SubWorkflowComponent::new(&child).unwrap();

    let mut parent = Workflow::new(card("parent"));
    parent
        .set_start_comp("start", StartComponent::new(), ComponentOptions::new())
        .add_workflow_comp("work", sub, ComponentOptions::new())
        .set_end_comp(
            "end",
            EndComponent::new(),
            ComponentOptions::new().with_inputs_schema(json!({"answer": "${work.output.answer}"})),
        )
        .add_connection("start", "work")
        .add_connection("work", "end");

    let output = parent
        .invoke(json!({"query": "hi"}), session("nested-colliding-ids"))
        .await
        .unwrap();

    assert_eq!(output.state, WorkflowExecutionState::Completed);
    assert_eq!(output.result, json!({"output": {"answer": "child"}}));
    assert_eq!(
        work.call_count(),
        1,
        "the child runs once; its state lanes are its own, not the parent's"
    );
}

// port of tests/unit_tests/core/component/test_sub_workflow_comp.py::test_max_nesting_depth
#[tokio::test]
async fn exceeding_the_nesting_limit_is_a_compile_error() {
    let work = FixedComponent::new(json!({"answer": "child"}));
    let sub = SubWorkflowComponent::new(&linear_child(work))
        .unwrap()
        .with_max_nesting_depth(0);

    let mut parent = Workflow::new(card("parent"));
    parent
        .set_start_comp("start", StartComponent::new(), ComponentOptions::new())
        .add_workflow_comp("sub", sub, ComponentOptions::new())
        .set_end_comp("end", EndComponent::new(), ComponentOptions::new())
        .add_connection("start", "sub")
        .add_connection("sub", "end");

    let failure = parent
        .invoke(json!({}), session("nested-depth"))
        .await
        .unwrap_err();

    assert_eq!(failure.kind(), ErrorKind::Workflow);
    let message = failure.to_string();
    assert!(message.contains("[100100]"), "{message}");
    assert!(
        message.contains("nesting hierarchy is too big"),
        "{message}"
    );
    assert!(message.contains("must <= 0"), "{message}");
}

// port of tests/unit_tests/core/component/test_sub_workflow_comp.py::test_sub_stream
#[tokio::test]
async fn a_parent_stream_forwards_the_child_end_frames_as_wires() {
    let component = Arc::new(SubWorkflowComponent::new(&streaming_child()).unwrap());
    let ctx = Arc::new(ComponentContext::new(
        session("nested-stream"),
        "sub",
        "parent",
    ));

    let frames: Vec<Value> = component
        .stream(json!({}), ctx)
        .map(|frame| frame.expect("the sub workflow stream failed"))
        .collect()
        .await;

    assert_eq!(
        frames,
        vec![
            json!({"type": END_NODE_STREAM, "index": 0, "payload": {"response": "a"}}),
            json!({"type": END_NODE_STREAM, "index": 1, "payload": {"response": "b"}}),
        ],
        "the child's terminal workflow_final frame is the parent stream's business"
    );
}

// port of tests/unit_tests/core/component/test_sub_workflow_comp.py::test_cache_stream
#[tokio::test]
async fn caching_stream_frames_answers_a_later_invoke() {
    let component = Arc::new(
        SubWorkflowComponent::new(&streaming_child())
            .unwrap()
            .with_cache_stream(true),
    );
    let ctx = Arc::new(ComponentContext::new(
        session("nested-cache-stream"),
        "sub",
        "parent",
    ));
    let frames: Vec<Value> = Arc::clone(&component)
        .stream(json!({}), ctx)
        .map(|frame| frame.expect("the sub workflow stream failed"))
        .collect()
        .await;
    assert_eq!(frames.len(), 2);

    assert_eq!(
        component.stream_output(),
        Some(json!({"response": "ab"})),
        "string payloads under the same key concatenate"
    );

    let invoke_ctx = ComponentContext::new(session("nested-cache-invoke"), "sub", "parent");
    let output = component.invoke(json!({}), &invoke_ctx).await.unwrap();
    // A real child run of this streaming workflow would answer the collected wire array
    // (see `tests/streaming.rs`), so the merged object can only come from the cache.
    assert_eq!(output, json!({"output": {"response": "ab"}}));
}

// port of tests/unit_tests/core/component/test_sub_workflow_comp.py::test_sub_interact
#[tokio::test]
async fn a_child_question_halts_the_parent_for_input() {
    let mut child = Workflow::new(WorkflowCard::new("child"));
    child
        .set_start_comp("start", StartComponent::new(), ComponentOptions::new())
        .add_workflow_comp(
            "ask",
            InteractingComponent::new(json!({"id": "ask", "text": "continue?"})),
            ComponentOptions::new(),
        )
        .set_end_comp("end", EndComponent::new(), ComponentOptions::new())
        .add_connection("start", "ask")
        .add_connection("ask", "end");
    let sub = SubWorkflowComponent::new(&child).unwrap();

    let mut parent = Workflow::new(card("parent"));
    parent
        .set_start_comp("start", StartComponent::new(), ComponentOptions::new())
        .add_workflow_comp("sub", sub, ComponentOptions::new())
        .set_end_comp("end", EndComponent::new(), ComponentOptions::new())
        .add_connection("start", "sub")
        .add_connection("sub", "end");

    let output = parent
        .invoke(json!({}), session("nested-interact"))
        .await
        .unwrap();

    assert_eq!(output.state, WorkflowExecutionState::InputRequired);
    let wires = output.result.as_array().expect("interaction wire array");
    let interaction = wires
        .iter()
        .find(|wire| wire["type"] == json!(INTERACTION_FRAME_TYPE))
        .unwrap_or_else(|| panic!("no interaction wire in {wires:?}"));
    assert_eq!(interaction["payload"]["id"], json!("ask"));
    assert_eq!(
        interaction["payload"]["question"],
        json!({"id": "ask", "text": "continue?"})
    );
}

/// The child workflow the streaming tests drive: `start → p → end`, where `p` is a
/// `Stream` producer and `end` renders `{{out}}` over the producer's buffered array.
fn streaming_child() -> Workflow {
    let mut child = Workflow::new(WorkflowCard::new("child-stream"));
    child
        .set_start_comp("start", StartComponent::new(), ComponentOptions::new())
        .add_workflow_comp(
            "p",
            StreamingProducer::new(vec![json!("a"), json!("b")]),
            ComponentOptions::new(),
        )
        .set_end_comp(
            "end",
            end_with_template("{{out}}"),
            // The `Transform` ability resolves `stream_inputs_schema`; declaring the same
            // key in `inputs_schema` too is a schema error (duplicate keys).
            ComponentOptions::new()
                .with_stream_inputs_schema(json!({"out": "${p}"}))
                .streaming(),
        )
        .add_connection("start", "p")
        .add_stream_connection("p", "end");
    child
}

/// The parent's sub-workflow node id; a dotted child id is addressed as `sub.<child>`.
const SUB_NODE: &str = "sub";

/// A checkpointed session through `checkpointer`.
fn checkpointed(
    id: &str,
    checkpointer: &MemoryCheckpointer,
) -> Arc<dyn openjiuwen_session::Session> {
    session_with_checkpointer(id, Arc::new(checkpointer.clone()) as Arc<dyn Checkpointer>)
}

/// One node-addressed resume answer for the interaction `id`.
fn answer(id: &str, value: Value) -> InteractiveInput {
    let mut interactive = InteractiveInput::new();
    interactive.update(id, value);
    interactive
}

/// A `question_content`-mode questioner: it asks `question` and then finishes.
fn questioner(id: &str, question: &str) -> QuestionerComponent {
    QuestionerComponent::new(
        QuestionerConfig::default()
            .with_question_content(question)
            .with_extract_fields_from_response(false),
    )
    .unwrap_or_else(|error| panic!("the questioner config of '{id}' is valid: {error}"))
}

/// The question of the single `__interaction__` wire an interrupted run reports.
fn only_question(output: &openjiuwen_workflow::WorkflowOutput) -> (String, Value) {
    let wires = output.result.as_array().expect("interaction wire array");
    let interaction = wires
        .iter()
        .find(|wire| wire["type"] == json!(INTERACTION_FRAME_TYPE))
        .unwrap_or_else(|| panic!("no interaction wire in {wires:?}"));
    (
        interaction["payload"]["id"]
            .as_str()
            .expect("the interaction carries an id")
            .to_owned(),
        interaction["payload"]["question"].clone(),
    )
}

/// `start → questioner → end`: the child asks one question and reports it.
fn questioner_child() -> Workflow {
    let mut child = Workflow::new(WorkflowCard::new("child"));
    child
        .set_start_comp("start", StartComponent::new(), ComponentOptions::new())
        .add_workflow_comp(
            "questioner",
            questioner("questioner", "which city?"),
            ComponentOptions::new(),
        )
        .set_end_comp(
            "end",
            EndComponent::new(),
            ComponentOptions::new()
                .with_inputs_schema(json!({"question": "${questioner.question}"})),
        )
        .add_connection("start", "questioner")
        .add_connection("questioner", "end");
    child
}

/// `start → sub → end`, where `sub` runs `child` and the parent end projects its question.
fn questioner_parent(child: &Workflow) -> Workflow {
    let sub = SubWorkflowComponent::new(child).expect("the child workflow compiles");
    let mut parent = Workflow::new(card("parent"));
    parent
        .set_start_comp("start", StartComponent::new(), ComponentOptions::new())
        .add_workflow_comp(SUB_NODE, sub, ComponentOptions::new())
        .set_end_comp(
            "end",
            EndComponent::new(),
            ComponentOptions::new()
                .with_inputs_schema(json!({"question": "${sub.output.question}"})),
        )
        .add_connection("start", SUB_NODE)
        .add_connection(SUB_NODE, "end");
    parent
}

/// The answers recorded in the parent session's sub-workflow lane, by interaction id.
async fn recorded_answers(session: &Arc<dyn openjiuwen_session::Session>) -> Value {
    session
        .get_component_state(SUB_NODE)
        .await
        .expect("the sub-workflow lane is readable")
        .and_then(|lane| lane.get(NESTED_ANSWERS_KEY).cloned())
        .unwrap_or(Value::Null)
}

/// The parent's first `invoke` halts on the child question; the resume answering it by the
/// child's own interaction id no longer deadlocks: the forwarded answer is consumed by the
/// child questioner in the same run, so the run completes.
#[tokio::test]
async fn a_resume_forwards_the_parents_answer_into_the_child_questioner() {
    let checkpointer = MemoryCheckpointer::new();
    let session = checkpointed("nested-resume-bare", &checkpointer);
    let parent = questioner_parent(&questioner_child());

    let first = parent
        .invoke(json!({"query": "hi"}), Arc::clone(&session))
        .await
        .unwrap();
    assert_eq!(first.state, WorkflowExecutionState::InputRequired);
    assert_eq!(
        only_question(&first),
        ("questioner".to_owned(), json!("which city?"))
    );
    assert_eq!(recorded_answers(&session).await, json!({}));

    let resumed = parent
        .resume(
            answer("questioner", json!("hangzhou")),
            Arc::clone(&session),
        )
        .await
        .unwrap();
    assert_eq!(
        resumed.state,
        WorkflowExecutionState::Completed,
        "a forwarded answer must not make the child ask again"
    );
    assert_eq!(
        resumed.result,
        json!({"output": {"question": "which city?"}})
    );
    assert_eq!(
        recorded_answers(&session).await,
        json!({"questioner": ["hangzhou"]}),
        "the forwarded answer is recorded for a later resume"
    );
    assert_eq!(
        checkpointer.len(session.id()),
        0,
        "the completed parent and child runs release their snapshots"
    );
}

/// Python addresses a nested interaction by its dotted node path, so a resume answering
/// `"<sub node>.<child id>"` must reach the same child queue.
#[tokio::test]
async fn a_dotted_interaction_id_answers_the_child_question() {
    let checkpointer = MemoryCheckpointer::new();
    let session = checkpointed("nested-resume-dotted", &checkpointer);
    let parent = questioner_parent(&questioner_child());

    let first = parent
        .invoke(json!({"query": "hi"}), Arc::clone(&session))
        .await
        .unwrap();
    assert_eq!(first.state, WorkflowExecutionState::InputRequired);

    let resumed = parent
        .resume(
            answer("sub.questioner", json!("hangzhou")),
            Arc::clone(&session),
        )
        .await
        .unwrap();
    assert_eq!(
        resumed.state,
        WorkflowExecutionState::Completed,
        "the dotted id is folded onto the bare child id"
    );
    assert_eq!(
        recorded_answers(&session).await,
        json!({"questioner": ["hangzhou"]}),
        "the dotted id is recorded under the bare child id, not twice"
    );
}

/// Two child questions across two resumes: the second resume completes because the answer
/// the child consumed first is replayed from the recorded history.
///
/// A single `QuestionerConfig` cannot make the child ask twice in one run — a fresh child
/// session restarts the questioner round, so each child run asks at most one question per
/// questioner node — so the child has two human-in-the-loop nodes instead. The child's
/// answers reach the parent's result through the generic `interact` seam.
#[tokio::test]
async fn a_second_resume_replays_the_first_answer_to_the_child() {
    let checkpointer = MemoryCheckpointer::new();
    let session = checkpointed("nested-resume-replay", &checkpointer);

    let mut child = Workflow::new(WorkflowCard::new("child"));
    child
        .set_start_comp("start", StartComponent::new(), ComponentOptions::new())
        .add_workflow_comp(
            "q1",
            InteractingComponent::new(json!({"id": "q1", "text": "first?"})),
            ComponentOptions::new(),
        )
        .add_workflow_comp(
            "q2",
            InteractingComponent::new(json!({"id": "q2", "text": "second?"})),
            ComponentOptions::new(),
        )
        .set_end_comp(
            "end",
            EndComponent::new(),
            ComponentOptions::new().with_inputs_schema(json!({
                "first": "${q1.answer}",
                "second": "${q2.answer}",
            })),
        )
        .add_connection("start", "q1")
        .add_connection("q1", "q2")
        .add_connection("q2", "end");
    let sub = SubWorkflowComponent::new(&child).expect("the child workflow compiles");

    let mut parent = Workflow::new(card("parent"));
    parent
        .set_start_comp("start", StartComponent::new(), ComponentOptions::new())
        .add_workflow_comp(SUB_NODE, sub, ComponentOptions::new())
        .set_end_comp(
            "end",
            EndComponent::new(),
            ComponentOptions::new().with_inputs_schema(json!({
                "first": "${sub.output.first}",
                "second": "${sub.output.second}",
            })),
        )
        .add_connection("start", SUB_NODE)
        .add_connection(SUB_NODE, "end");

    let first = parent
        .invoke(json!({"query": "hi"}), Arc::clone(&session))
        .await
        .unwrap();
    assert_eq!(first.state, WorkflowExecutionState::InputRequired);
    assert_eq!(
        only_question(&first),
        ("q1".to_owned(), json!({"id": "q1", "text": "first?"}))
    );

    let second = parent
        .resume(answer("q1", json!("one")), Arc::clone(&session))
        .await
        .unwrap();
    assert_eq!(second.state, WorkflowExecutionState::InputRequired);
    assert_eq!(
        only_question(&second),
        ("q2".to_owned(), json!({"id": "q2", "text": "second?"})),
        "the child consumed the first answer and asked its second question"
    );
    assert_eq!(
        recorded_answers(&session).await,
        json!({"q1": ["one"]}),
        "the first answer is recorded before the second question halts the run"
    );

    let third = parent
        .resume(answer("q2", json!("two")), Arc::clone(&session))
        .await
        .unwrap();
    assert_eq!(third.state, WorkflowExecutionState::Completed);
    assert_eq!(
        third.result,
        json!({"output": {"first": "one", "second": "two"}}),
        "the replayed child re-consumed the recorded first answer"
    );
    assert_eq!(
        recorded_answers(&session).await,
        json!({"q1": ["one"], "q2": ["two"]}),
        "the history accumulates one answer per interaction id"
    );
}
