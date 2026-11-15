#![allow(clippy::unwrap_used, clippy::expect_used)]
#![allow(
    unreachable_pub,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    clippy::new_without_default,
    clippy::module_name_repetitions,
    clippy::unnecessary_literal_bound,
    clippy::float_cmp
)]
//! Build-time validation through the public builder API.
//!
//! `Workflow::build()` (and therefore `invoke`, `stream`, and `to_mermaid`) collects every
//! problem it can see and reports them as ONE [`ErrorKind::Workflow`] error whose message
//! is the `"; "`-joined list of Python-coded `"[<code>] <message>"` problems. These tests
//! pin that contract: the start/end shape, component ids, edge endpoints, streaming
//! capabilities, `wait_for_all`, schema overlap, and the `ComponentOptions` projection into
//! `WorkflowSpec`.
//!
//! Ports of `tests/unit_tests/core/workflow/test_workflow.py` (`test_invalid_workflow`,
//! `test_workflow_with_invalid_connection`) and the workflow-config validation cases of
//! `tests/unit_tests/core/workflow/test_workflow_config.py`. Every test is offline and
//! deterministic.
mod common;

use std::sync::Arc;

use futures::StreamExt;
use openjiuwen_common::error::ErrorKind;
use openjiuwen_graph::{FnRouter, Router, RouterContext};
use serde_json::json;

use openjiuwen_workflow::{
    Component, ComponentAbility, ComponentOptions, EndComponent, LoopBound, LoopComponent,
    LoopConfig, LoopGroupComponent, LoopGroupConfig, LoopType, StartComponent, Workflow,
};

use common::{card, session, CollectorComponent, EchoComponent, StreamingProducer};

/// Compiles `workflow`, asserts it failed as a workflow error, and returns its message.
fn build_error(workflow: &Workflow) -> String {
    let error = workflow.build().unwrap_err();
    assert_eq!(
        error.kind(),
        ErrorKind::Workflow,
        "unexpected error: {error}"
    );
    error.message().into_owned()
}

/// The smallest valid workflow: `start → end`.
fn minimal() -> Workflow {
    let mut workflow = Workflow::new(card("minimal"));
    workflow
        .set_start_comp("start", StartComponent::new(), ComponentOptions::new())
        .set_end_comp("end", EndComponent::new(), ComponentOptions::new())
        .add_connection("start", "end");
    workflow
}

/// A graph router with no targets, enough to declare a conditional edge.
fn empty_router() -> Arc<dyn Router> {
    Arc::new(FnRouter::new("empty", |_ctx: &RouterContext| {
        Box::pin(async { Ok(Vec::new()) })
    }))
}

// port of tests/unit_tests/core/workflow/test_workflow.py::test_invalid_workflow
#[tokio::test]
async fn missing_start_and_end_are_reported() {
    let empty = Workflow::new(card("incomplete"));
    let message = build_error(&empty);
    assert!(message.contains("no start component"), "{message}");
    assert!(message.contains("no end component"), "{message}");

    let mut start_only = Workflow::new(card("start-only"));
    start_only.set_start_comp("start", StartComponent::new(), ComponentOptions::new());
    let message = build_error(&start_only);
    assert!(message.contains("no end component"), "{message}");
    assert!(!message.contains("no start component"), "{message}");

    let mut end_only = Workflow::new(card("end-only"));
    end_only.set_end_comp("end", EndComponent::new(), ComponentOptions::new());
    let message = build_error(&end_only);
    assert!(message.contains("no start component"), "{message}");
    assert!(!message.contains("no end component"), "{message}");
}

// port of tests/unit_tests/core/workflow/test_workflow.py::test_invalid_workflow
#[tokio::test]
async fn the_start_and_end_components_must_differ() {
    let mut workflow = Workflow::new(card("shared-endpoints"));
    workflow
        .set_start_comp("shared", StartComponent::new(), ComponentOptions::new())
        .set_end_comp("shared", EndComponent::new(), ComponentOptions::new());

    let message = build_error(&workflow);
    assert!(
        message.contains("the start component 'shared' is also the end component"),
        "{message}"
    );
}

// port of tests/unit_tests/core/workflow/test_workflow_config.py::test_component_id_validation
#[tokio::test]
async fn invalid_component_ids_carry_code_100010() {
    let mut workflow = minimal();
    workflow.add_workflow_comp("bad id!", EchoComponent::new(), ComponentOptions::new());
    let message = build_error(&workflow);
    assert!(message.contains("[100010]"), "{message}");
    assert!(message.contains("bad id!"), "{message}");

    let mut workflow = minimal();
    workflow.add_workflow_comp("", EchoComponent::new(), ComponentOptions::new());
    let message = build_error(&workflow);
    assert!(message.contains("[100010]"), "{message}");
    assert!(message.contains("is None or empty"), "{message}");
}

// port of tests/unit_tests/core/workflow/test_workflow.py::test_duplicate_component_id
#[tokio::test]
async fn a_duplicate_component_id_is_recorded_and_reported() {
    let mut workflow = minimal();
    workflow.add_workflow_comp("dup", EchoComponent::new(), ComponentOptions::new());
    workflow.add_workflow_comp("dup", EchoComponent::new(), ComponentOptions::new());

    assert_eq!(
        workflow
            .component_ids()
            .iter()
            .filter(|id| **id == "dup")
            .count(),
        1,
        "the first registration wins"
    );
    let message = build_error(&workflow);
    assert!(
        message.contains("component id 'dup' is already registered"),
        "{message}"
    );
}

// port of tests/unit_tests/core/workflow/test_workflow.py::test_workflow_with_invalid_connection
#[tokio::test]
async fn edges_to_unknown_components_name_the_endpoint() {
    let mut workflow = minimal();
    workflow.add_connection("start", "ghost");
    let message = build_error(&workflow);
    assert!(
        message.contains("edge target 'ghost' is not registered"),
        "{message}"
    );

    let mut workflow = minimal();
    workflow.add_connection("ghost", "end");
    let message = build_error(&workflow);
    assert!(
        message.contains("edge source 'ghost' is not registered"),
        "{message}"
    );

    let mut workflow = minimal();
    workflow.add_conditional_connection("start", "ghost", empty_router());
    let message = build_error(&workflow);
    assert!(
        message.contains("conditional edge target 'ghost' is not registered"),
        "{message}"
    );
}

// port of tests/unit_tests/core/workflow/test_workflow_with_comp_stream.py::test_stream_edge_ability
#[tokio::test]
async fn stream_edges_require_streaming_abilities_on_both_ends() {
    // The source only implements `invoke`.
    let mut workflow = Workflow::new(card("stream-source"));
    workflow
        .set_start_comp("start", StartComponent::new(), ComponentOptions::new())
        .add_workflow_comp("echo", EchoComponent::new(), ComponentOptions::new())
        .set_end_comp("end", EndComponent::new(), ComponentOptions::new())
        .add_connection("start", "echo")
        .add_stream_connection("echo", "end");

    let message = build_error(&workflow);
    assert!(message.contains("[100015]"), "{message}");
    assert!(
        message.contains("does not support 'stream' or 'transform'"),
        "{message}"
    );

    // The target implements neither `collect` nor `transform`.
    let mut workflow = Workflow::new(card("stream-target"));
    workflow
        .set_start_comp("start", StartComponent::new(), ComponentOptions::new())
        .add_workflow_comp(
            "producer",
            StreamingProducer::new(vec![json!("frame")]),
            ComponentOptions::new(),
        )
        .add_workflow_comp("echo", EchoComponent::new(), ComponentOptions::new())
        .set_end_comp("end", EndComponent::new(), ComponentOptions::new())
        .add_connection("start", "producer")
        .add_stream_connection("producer", "echo")
        .add_connection("echo", "end");

    let message = build_error(&workflow);
    assert!(message.contains("[100011]"), "{message}");
    assert!(
        message.contains("a stream target must support 'collect' or 'transform'"),
        "{message}"
    );
}

// port of tests/unit_tests/core/workflow/test_workflow_config.py::test_declared_ability_validation
#[tokio::test]
async fn a_declared_ability_the_component_lacks_is_rejected() {
    let mut workflow = minimal();
    workflow.add_workflow_comp(
        "echo",
        EchoComponent::new(),
        ComponentOptions::new().with_abilities(vec![ComponentAbility::Transform]),
    );
    workflow
        .add_connection("start", "echo")
        .add_connection("echo", "end");

    let message = build_error(&workflow);
    assert!(message.contains("[100011]"), "{message}");
    assert!(
        message.contains("the component does not implement the declared ability"),
        "{message}"
    );
    assert!(message.contains("ability=transform"), "{message}");
}

// port of tests/unit_tests/core/workflow/test_workflow.py::test_workflow_with_wait_for_all
#[tokio::test]
async fn collect_nodes_may_not_disable_wait_for_all() {
    let mut workflow = Workflow::new(card("wait-for-all"));
    workflow
        .set_start_comp("start", StartComponent::new(), ComponentOptions::new())
        .add_workflow_comp(
            "producer",
            StreamingProducer::new(vec![json!("frame")]),
            ComponentOptions::new(),
        )
        .add_workflow_comp(
            "collector",
            CollectorComponent::new(),
            ComponentOptions::new().with_wait_for_all(false),
        )
        .set_end_comp("end", EndComponent::new(), ComponentOptions::new())
        .add_connection("start", "producer")
        .add_stream_connection("producer", "collector")
        .add_connection("collector", "end");

    let message = build_error(&workflow);
    assert!(message.contains("[100011]"), "{message}");
    assert!(
        message.contains("must set 'wait_for_all' to True"),
        "{message}"
    );
}

// port of tests/unit_tests/core/workflow/test_workflow_config.py::test_schema_key_overlap
#[tokio::test]
async fn batch_and_stream_schemas_may_not_share_a_key() {
    let mut workflow = minimal();
    workflow.add_workflow_comp(
        "echo",
        EchoComponent::new(),
        ComponentOptions::new()
            .with_inputs_schema(json!({"query": "${start.query}"}))
            .with_stream_inputs_schema(json!({"query": "${start.query}"})),
    );
    workflow
        .add_connection("start", "echo")
        .add_connection("echo", "end");

    let message = build_error(&workflow);
    assert!(message.contains("[100014]"), "{message}");
    assert!(
        message.contains("duplicate key both exist in inputs_schema with stream_inputs_schema"),
        "{message}"
    );
    assert!(message.contains("key=query"), "{message}");
}

// port of tests/unit_tests/core/workflow/test_workflow.py::test_end_component_has_no_successors
#[tokio::test]
async fn the_end_component_may_not_have_successors() {
    let mut workflow = minimal();
    workflow.add_workflow_comp("after", EchoComponent::new(), ComponentOptions::new());
    workflow.add_connection("end", "after");

    let message = build_error(&workflow);
    assert!(
        message.contains("the end component 'end' has an outgoing regular edge to 'after'"),
        "{message}"
    );
}

// port of tests/unit_tests/core/workflow/test_workflow.py::test_invalid_workflow
#[tokio::test]
async fn every_problem_is_joined_into_one_workflow_error() {
    let mut workflow = Workflow::new(card("many-problems"));
    workflow
        .set_start_comp("start", StartComponent::new(), ComponentOptions::new())
        .add_workflow_comp("bad id!", EchoComponent::new(), ComponentOptions::new())
        .add_connection("start", "ghost");

    let error = workflow.build().unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Workflow);
    let message = error.message().into_owned();
    let parts: Vec<&str> = message.split("; ").collect();

    assert_eq!(parts.len(), 3, "{message}");
    assert!(
        parts[0].starts_with("[100100]") && parts[0].contains("no end component"),
        "{message}"
    );
    assert!(parts[1].starts_with("[100010]"), "{message}");
    assert!(
        parts[2].starts_with("[100100]") && parts[2].contains("edge target 'ghost'"),
        "{message}"
    );
}

// port of tests/unit_tests/core/workflow/test_workflow_config.py::test_node_spec_defaults
#[tokio::test]
async fn component_options_surface_in_the_spec() {
    let mut workflow = Workflow::new(card("options"));
    workflow
        .set_start_comp("start", StartComponent::new(), ComponentOptions::new())
        .add_workflow_comp(
            "mid",
            EchoComponent::new(),
            ComponentOptions::new()
                .with_timeout_ms(1500)
                .with_max_retries(2),
        )
        .set_end_comp("end", EndComponent::new(), ComponentOptions::new())
        .add_connection("start", "mid")
        .add_connection("mid", "end");

    let spec = workflow.spec();
    let start = &spec.comp_configs["start"];
    assert!(
        (start.timeout + 1.0).abs() < f64::EPSILON,
        "an unset timeout is -1.0 seconds, got {}",
        start.timeout
    );
    assert_eq!(start.max_retries, 0);

    let mid = &spec.comp_configs["mid"];
    assert!(
        (mid.timeout - 1.5).abs() < f64::EPSILON,
        "1500 ms is 1.5 seconds, got {}",
        mid.timeout
    );
    assert_eq!(mid.max_retries, 2);
    assert!(workflow.build().is_ok(), "the options are valid");
}

// port of tests/unit_tests/core/workflow/test_workflow.py::test_simple_workflow
#[tokio::test]
async fn a_minimal_workflow_builds() {
    let compiled = minimal().build().unwrap();

    assert_eq!(compiled.start_comp_id(), "start");
    assert_eq!(compiled.end_comp_id(), "end");
    assert!(!compiled.is_streaming(), "the end node does not stream");
}

// port of tests/unit_tests/core/workflow/test_workflow.py::test_invalid_workflow_stream
#[tokio::test]
async fn streaming_an_invalid_workflow_yields_one_compile_error() {
    let workflow = Workflow::new(card("invalid-stream"));
    let items: Vec<_> = workflow
        .stream(json!({}), session("invalid-stream"))
        .collect()
        .await;

    assert_eq!(items.len(), 1, "the compile error is the only item");
    let failure = items.into_iter().next().unwrap().unwrap_err();
    assert_eq!(failure.kind(), ErrorKind::Workflow);
    assert!(
        failure.message().contains("no start component"),
        "{failure}"
    );
}

/// `start → loop → end` around `component`, the shape every loop build test uses.
fn loop_shaped(id: &str, component: impl Component + 'static) -> Workflow {
    let mut workflow = Workflow::new(card(id));
    workflow
        .set_start_comp("start", StartComponent::new(), ComponentOptions::new())
        .add_workflow_comp("loop", component, ComponentOptions::new())
        .set_end_comp("end", EndComponent::new(), ComponentOptions::new())
        .add_connection("start", "loop")
        .add_connection("loop", "end");
    workflow
}

/// The loop body used by the group cases.
fn echo_body() -> Vec<Arc<dyn Component>> {
    vec![Arc::new(EchoComponent::new()) as Arc<dyn Component>]
}

// port of tests/unit_tests/core/component/test_loop_comp.py::test_loop_number
#[tokio::test]
async fn a_loop_without_its_bound_parameter_fails_at_build_time() {
    // Python rejects the config with `COMPONENT_LOOP_PARAM_INVALID` (`101030`) while
    // parsing it; `Component::validate_config` moves that check into `Workflow::build`,
    // so no run starts on a loop that cannot resolve its bound.
    let failure = build_error(&loop_shaped(
        "loop-missing-bound",
        LoopComponent::new(
            LoopConfig::new(LoopType::Number),
            vec![Arc::new(EchoComponent::new())],
        ),
    ));
    assert!(failure.contains("[101030]"), "{failure}");
    assert!(
        failure.contains("loop_number variable not found"),
        "{failure}"
    );
}

#[tokio::test]
async fn a_group_over_its_iteration_cap_fails_at_build_time() {
    let workflow = loop_shaped(
        "group-over-cap",
        LoopGroupComponent::new(
            LoopGroupConfig::new()
                .with_loop_number(5)
                .with_max_iterations(2),
            echo_body(),
        ),
    );
    let failure = build_error(&workflow);
    assert!(failure.contains("[101030]"), "{failure}");
    assert!(failure.contains("maximum limit 2"), "{failure}");
    // `stream` reuses the same compile step, so it fails the same way.
    let items: Vec<_> = workflow
        .stream(json!({}), session("group-over-cap"))
        .collect()
        .await;
    assert_eq!(items.len(), 1);
    assert!(items.into_iter().next().unwrap().is_err());
}

#[tokio::test]
async fn a_group_without_a_terminator_fails_at_build_time() {
    // A break-only container needs a `LoopBreakComponent` in its body to ever stop.
    let failure = build_error(&loop_shaped(
        "group-no-terminator",
        LoopGroupComponent::new(
            LoopGroupConfig::new()
                .with_loop_type(LoopBound::UntilBreak)
                .with_max_iterations(10),
            echo_body(),
        ),
    ));
    assert!(failure.contains("[101030]"), "{failure}");
    assert!(failure.contains("cannot terminate"), "{failure}");
}

#[tokio::test]
async fn a_valid_bounded_group_compiles_and_runs() {
    let workflow = loop_shaped(
        "group-valid",
        LoopGroupComponent::new(LoopGroupConfig::new().with_loop_number(2), echo_body()),
    );
    let compiled = workflow.build().expect("the bound is valid");
    assert!(compiled.ability("loop").is_some());
    let output = workflow
        .invoke(json!({"q": 1}), session("group-valid"))
        .await
        .expect("the run completes");
    assert_eq!(output.result["output"]["loop_count"], json!(2));
}
