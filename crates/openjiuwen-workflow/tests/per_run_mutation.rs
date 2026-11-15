#![allow(clippy::unwrap_used, clippy::expect_used)]
#![allow(
    unreachable_pub,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    clippy::new_without_default,
    clippy::module_name_repetitions,
    clippy::unnecessary_literal_bound
)]
//! Per-run topology mutation: Python mutates a workflow's graph between runs, and this crate
//! expresses that as *a mutated copy compiled for one run*.
//!
//! [`CompiledWorkflow::mutated`] clones the recorded definition, applies a closure,
//! re-validates, and compiles the result, so `mutate` runs before the run starts (it is never
//! a mid-execution change). `invoke_mutated` / `stream_mutated` build that copy and run it
//! once; the receiver keeps its own topology, its own spec, and its own ability selection.
//! Everything here is offline and deterministic.
mod common;

use std::sync::Arc;

use futures::StreamExt;
use openjiuwen_common::error::{ErrorKind, Result};
use openjiuwen_session::Session;
use serde_json::{json, Value};

use openjiuwen_workflow::engine::{EdgeKind, NodeConfig, StreamDelivery, WorkflowDefinition};
use openjiuwen_workflow::{
    ComponentAbility, ComponentOptions, EndComponent, EndConfig, StartComponent, Workflow,
    WorkflowCard, WorkflowExecutionState, WorkflowRunOptions, END_NODE_STREAM, WORKFLOW_FINAL,
};

use common::{card, end_with_template, session, EchoComponent, StreamingProducer};

/// `start → echo → end`, where `end` is a flat (template-less) end component.
fn linear() -> Workflow {
    let mut workflow = Workflow::new(card("mutation"));
    workflow
        .set_start_comp("start", StartComponent::new(), ComponentOptions::new())
        .add_workflow_comp("echo", EchoComponent::new(), ComponentOptions::new())
        .set_end_comp("end", EndComponent::new(), ComponentOptions::new())
        .add_connection("start", "echo")
        .add_connection("echo", "end");
    workflow
}

/// A template end component for `template`, declaring the abilities a mutation needs.
fn template_end(template: &str, abilities: Vec<ComponentAbility>) -> (EndComponent, NodeConfig) {
    (
        EndComponent::with_config(EndConfig::new(template)),
        NodeConfig {
            abilities: Some(abilities),
            ..NodeConfig::default()
        },
    )
}

/// A `Stream` end that reads the producer's first frame through `{{first}}`.
///
/// A stream edge requires the target to support `Collect`/`Transform` (validation
/// `[100015]`), and the streamed inputs come from the producer's buffered frames, so the
/// node declares `Transform` and both schema pairs — the same shape
/// `ComponentOptions::streaming()` produces.
fn streaming_end(template: &str) -> (EndComponent, NodeConfig) {
    let (end, mut node) = template_end(
        template,
        vec![
            ComponentAbility::Invoke,
            ComponentAbility::Stream,
            ComponentAbility::Transform,
        ],
    );
    node.inputs_schema = Some(json!({
        "first": "${producer[0].delta}",
        "second": "${producer[1].delta}",
    }));
    node.stream_inputs_schema = Some(json!({"whole": "${producer}"}));
    node.wait_for_all = Some(true);
    (end, node)
}

/// `start → producer → end`, where `end` streams `{{first}}{{second}}`.
fn streaming() -> Workflow {
    let mut workflow = Workflow::new(card("mutation-stream"));
    workflow
        .set_start_comp("start", StartComponent::new(), ComponentOptions::new())
        .add_workflow_comp(
            "producer",
            StreamingProducer::new(vec![json!({"delta": "a"}), json!({"delta": "b"})]),
            ComponentOptions::new(),
        )
        .set_end_comp(
            "end",
            end_with_template("{{first}}{{second}}"),
            ComponentOptions::new()
                .with_inputs_schema(json!({
                    "first": "${producer[0].delta}",
                    "second": "${producer[1].delta}",
                }))
                .with_stream_inputs_schema(json!({"whole": "${producer}"}))
                .streaming(),
        )
        .add_connection("start", "producer")
        .add_stream_connection("producer", "end");
    workflow
}

/// The whole state an error-free flat end node reports under `output`.
fn end_state(result: &Value) -> &serde_json::Map<String, Value> {
    result
        .get("output")
        .and_then(Value::as_object)
        .unwrap_or_else(|| panic!("expected a flat end projection, got {result}"))
}

// port of tests/unit_tests/core/workflow/test_workflow.py::test_workflow_mutate_for_one_run
#[tokio::test]
async fn a_mutated_end_template_applies_to_one_run_only() {
    let compiled = linear().build().unwrap();

    let mutated = compiled
        .invoke_mutated(
            json!({"query": "hi"}),
            session("mutation-end"),
            WorkflowRunOptions::default(),
            |definition| {
                let (end, node) = template_end("{{start.query}}!", vec![ComponentAbility::Invoke]);
                definition.insert_component("end", Arc::new(end), node);
            },
        )
        .await
        .unwrap();
    assert_eq!(mutated.state, WorkflowExecutionState::Completed);
    assert_eq!(mutated.result, json!({"response": "hi!"}));

    // The original keeps its own topology: two runs both report the flat end projection and
    // never the mutated template response.
    for name in ["mutation-original-1", "mutation-original-2"] {
        let output = compiled
            .invoke(
                json!({"query": "hi"}),
                session(name),
                WorkflowRunOptions::default(),
            )
            .await
            .unwrap();
        assert!(
            output.result.get("response").is_none(),
            "the original end component is unchanged: {}",
            output.result
        );
        assert_eq!(
            end_state(&output.result).get("query"),
            Some(&json!("hi")),
            "the original run still carries its inputs: {}",
            output.result
        );
        assert_eq!(compiled.end_comp_id(), "end");
        assert!(!compiled.spec().comp_configs.contains_key("extra"));
    }
}

// port of tests/unit_tests/core/workflow/test_workflow.py::test_workflow_mutate_insert_node
#[tokio::test]
async fn mutation_can_insert_a_node_and_retarget_an_edge() {
    let compiled = linear().build().unwrap();
    let original = compiled.definition().clone();

    let output = compiled
        .invoke_mutated(
            json!({"query": "hi"}),
            session("mutation-insert"),
            WorkflowRunOptions::default(),
            |definition: &mut WorkflowDefinition| {
                definition.insert_component(
                    "extra",
                    Arc::new(EchoComponent::new()),
                    NodeConfig::default(),
                );
                assert!(
                    definition.retarget_edge("echo", "end", "extra"),
                    "the echo → end edge must exist"
                );
                definition.add_edge(
                    vec!["extra".to_owned()],
                    "end".to_owned(),
                    EdgeKind::Regular,
                );
            },
        )
        .await
        .unwrap();

    let state = end_state(&output.result);
    assert!(state.contains_key("extra"), "{state:?}");
    assert!(state.contains_key("echo"), "{state:?}");

    // The mutated copy reports its own spec; the original's definition is untouched.
    let mutated = compiled
        .mutated(|definition| {
            definition.insert_component(
                "extra",
                Arc::new(EchoComponent::new()),
                NodeConfig::default(),
            );
            assert!(definition.retarget_edge("echo", "end", "extra"));
            definition.add_edge(
                vec!["extra".to_owned()],
                "end".to_owned(),
                EdgeKind::Regular,
            );
        })
        .unwrap();
    assert_eq!(
        mutated.spec().edges.get("echo"),
        Some(&vec!["extra".to_owned()])
    );
    assert_eq!(
        mutated.spec().edges.get("extra"),
        Some(&vec!["end".to_owned()])
    );
    assert!(mutated.spec().comp_configs.contains_key("extra"));
    assert_eq!(mutated.ability("extra"), Some(ComponentAbility::Invoke));
    assert_eq!(mutated.card().id, "mutation");
    assert_eq!(mutated.end_comp_id(), "end");
    assert!(
        !compiled.spec().comp_configs.contains_key("extra"),
        "the original spec never gains the mutated node"
    );
    assert_eq!(
        compiled.definition().component_ids(),
        original.component_ids(),
        "the original definition is untouched"
    );
}

// port of tests/unit_tests/core/workflow/test_workflow.py::test_workflow_mutate_remove_node
#[tokio::test]
async fn mutation_can_remove_a_node_and_rewire_the_edges() {
    let compiled = linear().build().unwrap();

    let output = compiled
        .invoke_mutated(
            json!({"query": "hi"}),
            session("mutation-remove"),
            WorkflowRunOptions::default(),
            |definition| {
                assert!(
                    definition.retarget_edge("start", "echo", "end"),
                    "the start → echo edge must exist"
                );
                definition.remove_component("echo");
            },
        )
        .await
        .unwrap();

    let state = end_state(&output.result);
    assert!(
        !state.contains_key("echo"),
        "the removed node never ran: {state:?}"
    );
    assert_eq!(state.get("query"), Some(&json!("hi")));
    assert_eq!(
        compiled.definition().component_ids(),
        vec!["start", "echo", "end"],
        "the original still has the removed node"
    );
}

#[tokio::test]
async fn an_invalid_mutation_reports_the_coded_compile_error() {
    let compiled = linear().build().unwrap();

    let failure = compiled
        .invoke_mutated(
            json!({"query": "hi"}),
            session("mutation-invalid"),
            WorkflowRunOptions::default(),
            |definition| definition.end = None,
        )
        .await
        .expect_err("a workflow without an end component must not run");
    assert_eq!(failure.kind(), ErrorKind::Workflow);
    assert!(
        failure.message().contains("[100100]"),
        "the compile error carries Python's code:\n{failure}"
    );
    assert!(failure.message().contains("no end component"), "{failure}");

    // A dangling edge is also a coded compile error, and it fails before the run starts.
    let failure = compiled
        .mutated(|definition| {
            definition.add_edge(
                vec!["start".to_owned()],
                "ghost".to_owned(),
                EdgeKind::Regular,
            );
        })
        .expect_err("an edge to an unregistered component must not compile");
    assert_eq!(failure.kind(), ErrorKind::Workflow);
    assert!(
        failure.message().contains("[100100]"),
        "the edge error is a coded compile error:\n{failure}"
    );
    assert!(failure.message().contains("ghost"), "{failure}");
}

#[tokio::test]
async fn a_mutated_run_rejects_interleaved_delivery_like_any_other_run() {
    let compiled = linear().build().unwrap();

    let failure = compiled
        .invoke_mutated(
            json!({"query": "hi"}),
            session("mutation-interleaved"),
            WorkflowRunOptions::default().with_stream_delivery(StreamDelivery::Interleaved),
            |_definition| {},
        )
        .await
        .expect_err("Interleaved cannot be served by this engine");
    assert_eq!(failure.kind(), ErrorKind::Unsupported);
    assert!(
        failure.message().contains("StreamDelivery::Buffered"),
        "{failure}"
    );
}

#[tokio::test]
async fn stream_mutated_yields_frames_from_the_mutated_topology() {
    let compiled = streaming().build().unwrap();

    let (end, node) = streaming_end("mutated {{first}}");
    let frames: Vec<_> = compiled
        .stream_mutated(
            json!({}),
            session("mutation-stream"),
            WorkflowRunOptions::default(),
            |definition| {
                definition.insert_component("end", Arc::new(end), node);
            },
        )
        .map(|item: Result<openjiuwen_session::stream::StreamItem>| {
            item.expect("the mutated stream yielded an error")
        })
        .collect()
        .await;

    let end_frames: Vec<&Value> = frames
        .iter()
        .filter(|frame| frame.item_type == END_NODE_STREAM)
        .map(|frame| &frame.payload)
        .collect();
    assert_eq!(
        end_frames,
        vec![&json!({"response": "mutated "}), &json!({"response": "a"})],
        "the mutated template renders through the mutated end node: {frames:?}"
    );
    let final_frame = frames.last().expect("the stream ends with the final frame");
    assert_eq!(final_frame.item_type, WORKFLOW_FINAL);
    assert_eq!(
        final_frame.payload.as_array().map(Vec::len),
        Some(end_frames.len()),
        "the final frame collects the mutated end wires: {final_frame:?}"
    );

    // The original template still renders on the original workflow.
    let original: Vec<_> = compiled
        .stream(
            json!({}),
            session("mutation-stream-original"),
            WorkflowRunOptions::default(),
        )
        .map(|item| item.expect("the original stream yielded an error"))
        .collect()
        .await;
    let original_end: Vec<&Value> = original
        .iter()
        .filter(|frame| frame.item_type == END_NODE_STREAM)
        .map(|frame| &frame.payload)
        .collect();
    assert_eq!(
        original_end,
        vec![&json!({"response": "a"}), &json!({"response": "b"})],
        "{original:?}"
    );
}

#[tokio::test]
async fn mutation_leaves_the_receiver_runnable_concurrently() {
    let compiled = linear().build().unwrap();
    let session_a: Arc<dyn Session> = session("mutation-concurrent-a");
    let session_b: Arc<dyn Session> = session("mutation-concurrent-b");

    let mutated = compiled.invoke_mutated(
        json!({"query": "mutated"}),
        Arc::clone(&session_a),
        WorkflowRunOptions::default(),
        |definition| {
            let (end, node) = template_end("{{start.query}}!", vec![ComponentAbility::Invoke]);
            definition.insert_component("end", Arc::new(end), node);
        },
    );
    let original = compiled.invoke(
        json!({"query": "original"}),
        session_b,
        WorkflowRunOptions::default(),
    );
    let (mutated, original) = tokio::join!(mutated, original);

    assert_eq!(mutated.unwrap().result, json!({"response": "mutated!"}));
    let original = original.unwrap();
    assert!(
        original.result.get("response").is_none(),
        "{}",
        original.result
    );
    assert_eq!(
        end_state(&original.result).get("query"),
        Some(&json!("original"))
    );
}

#[test]
fn mutation_helpers_are_state_aware() {
    let mut definition = linear().build().unwrap().definition().clone();
    let original_ids: Vec<String> = definition
        .component_ids()
        .into_iter()
        .map(str::to_owned)
        .collect();
    let workflow_card = WorkflowCard::new("mutation");

    // A retarget that matches nothing is reported instead of silently ignored.
    assert!(!definition.retarget_edge("start", "end", "elsewhere"));

    // `remove_component` drops the node, its edges, and a start/end choice naming it.
    definition.remove_component("echo");
    assert_eq!(definition.component_ids(), vec!["start", "end"]);
    assert!(definition.edges.iter().all(|edge| !edge
        .sources
        .iter()
        .any(|source| source == "echo")
        && edge.target != "echo"));

    definition.remove_component("end");
    assert_eq!(definition.end, None);
    assert_eq!(
        definition.edges.len(),
        0,
        "every edge is dropped: {:?}",
        definition.edges
    );
    assert!(definition.component_mut("end").is_none());
    assert!(definition.component_mut("start").is_some());

    // `remove_edges_to` keeps the edges leaving the node.
    let mut restored = linear().build().unwrap().definition().clone();
    assert_eq!(
        restored
            .component_ids()
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>(),
        original_ids
    );
    restored.remove_edges_to("echo");
    assert!(restored.edges.iter().all(|edge| edge.target != "echo"));
    assert!(restored
        .edges
        .iter()
        .any(|edge| edge.sources.iter().any(|source| source == "echo")));
    assert_eq!(restored.start, Some("start".to_owned()));

    // The card is untouched by mutation helpers.
    assert_eq!(restored.card, workflow_card);
    assert_eq!(restored.card.id, "mutation");
}
