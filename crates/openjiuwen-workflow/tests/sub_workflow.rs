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
//! human-in-the-loop question surfaces as the parent's `INPUT_REQUIRED` interaction.
//!
//! Everything is offline and deterministic: no sleeps, no network, no wall-clock reads.
mod common;

use std::sync::Arc;

use futures::StreamExt;
use openjiuwen_common::error::ErrorKind;
use openjiuwen_session::interaction::INTERACTION_FRAME_TYPE;
use serde_json::{json, Value};

use openjiuwen_workflow::components::sub_workflow::SubWorkflowComponent;
use openjiuwen_workflow::{
    Component, ComponentContext, ComponentOptions, EndComponent, StartComponent, Workflow,
    WorkflowCard, WorkflowExecutionState, END_NODE_STREAM,
};

use common::{
    card, end_with_template, session, FixedComponent, InteractingComponent, StreamingProducer,
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
