#![allow(clippy::unwrap_used, clippy::expect_used)]
#![allow(
    unreachable_pub,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    clippy::new_without_default,
    clippy::module_name_repetitions,
    clippy::unnecessary_literal_bound
)]
//! Fan-in: Python's `wait_for_all` barrier versus first-arrival activation.
//!
//! A regular/stream/barrier edge compiles to a CNF channel, so a node with several
//! inbound edges waits for every predecessor. A node registered with
//! `wait_for_all = false` and at least two distinct such predecessors is instead a
//! first-arrival node: its channel keeps no group, so it runs as soon as any predecessor
//! arrives and runs again when a later one does, resolving a not-yet-produced key to
//! `null`.
//!
//! Ports of `tests/unit_tests/core/workflow/test_workflow.py::test_workflow_with_wait_for_all`.
//! The BSP superstep boundary supplies the ordering these tests rely on — a producer in a
//! later layer arrives one superstep later — so nothing here sleeps or polls.
mod common;

use std::sync::Arc;

use openjiuwen_common::error::ErrorKind;
use openjiuwen_graph::{FnRouter, Router, RouterContext};
use serde_json::{json, Value};

use openjiuwen_workflow::{
    ComponentAbility, ComponentOptions, EndComponent, StartComponent, Workflow,
    WorkflowExecutionState,
};

use common::{card, session, CollectorComponent, FixedComponent};

/// `start → {b, c, d} → join` plus `start → a → a2 → join`.
///
/// `b`, `c`, and `d` arrive in the second superstep; `a2` (one layer deeper) arrives in
/// the third. That difference is what separates "waits for everyone" from "fires on the
/// first arrival".
struct LayeredFanIn {
    workflow: Workflow,
    join: FixedComponent,
    a2: FixedComponent,
}

fn layered_fan_in(wait_for_all: Option<bool>) -> LayeredFanIn {
    let a2 = FixedComponent::new(json!({"value": "a2"}));
    let join = FixedComponent::new(json!({"joined": true}));
    let mut options = ComponentOptions::new().with_inputs_schema(json!({
        "a2": "${a2.value}",
        "b": "${b.value}",
        "c": "${c.value}",
        "d": "${d.value}",
    }));
    if let Some(wait_for_all) = wait_for_all {
        options = options.with_wait_for_all(wait_for_all);
    }
    let mut workflow = Workflow::new(card("fan-in-layered"));
    workflow
        .set_start_comp(
            "start",
            StartComponent::new(),
            ComponentOptions::new().with_inputs_schema(json!({"query": "${inputs.query}"})),
        )
        .add_workflow_comp(
            "a",
            FixedComponent::new(json!({"value": "a"})),
            ComponentOptions::new(),
        )
        .add_workflow_comp("a2", a2.clone(), ComponentOptions::new())
        .add_workflow_comp(
            "b",
            FixedComponent::new(json!({"value": "b"})),
            ComponentOptions::new(),
        )
        .add_workflow_comp(
            "c",
            FixedComponent::new(json!({"value": "c"})),
            ComponentOptions::new(),
        )
        .add_workflow_comp(
            "d",
            FixedComponent::new(json!({"value": "d"})),
            ComponentOptions::new(),
        )
        .add_workflow_comp("join", join.clone(), options)
        .set_end_comp(
            "end",
            EndComponent::new(),
            ComponentOptions::new().with_inputs_schema(json!({"join": "${join.joined}"})),
        )
        .add_connection("start", "a")
        .add_connection("a", "a2")
        .add_connection("start", "b")
        .add_connection("start", "c")
        .add_connection("start", "d")
        .add_connection("a2", "join")
        .add_connection("b", "join")
        .add_connection("c", "join")
        .add_connection("d", "join")
        .add_connection("join", "end");
    LayeredFanIn { workflow, join, a2 }
}

// port of tests/unit_tests/core/workflow/test_workflow.py::test_workflow_with_wait_for_all
#[tokio::test]
async fn the_default_fan_in_waits_for_every_predecessor() {
    let LayeredFanIn { workflow, join, a2 } = layered_fan_in(None);
    let output = workflow
        .invoke(json!({"query": "hi"}), session("fan-in-cnf"))
        .await
        .unwrap();

    assert_eq!(output.state, WorkflowExecutionState::Completed);
    assert_eq!(
        a2.call_count(),
        1,
        "the deep producer runs exactly once per run"
    );
    assert_eq!(
        join.call_count(),
        1,
        "a barrier join runs once, after every predecessor"
    );
    assert_eq!(
        join.calls(),
        vec![json!({"a2": "a2", "b": "b", "c": "c", "d": "d"})]
    );
    assert_eq!(output.result, json!({"output": {"join": true}}));
}

#[tokio::test]
async fn wait_for_all_true_behaves_like_the_default() {
    let LayeredFanIn { workflow, join, .. } = layered_fan_in(Some(true));
    let output = workflow
        .invoke(json!({"query": "hi"}), session("fan-in-cnf-explicit"))
        .await
        .unwrap();

    assert_eq!(output.state, WorkflowExecutionState::Completed);
    assert_eq!(join.call_count(), 1);
    assert_eq!(
        join.calls(),
        vec![json!({"a2": "a2", "b": "b", "c": "c", "d": "d"})]
    );
}

#[tokio::test]
async fn first_arrival_fires_on_each_arriving_predecessor() {
    let LayeredFanIn { workflow, join, .. } = layered_fan_in(Some(false));
    let output = workflow
        .invoke(json!({"query": "hi"}), session("fan-in-first"))
        .await
        .unwrap();

    assert_eq!(output.state, WorkflowExecutionState::Completed);
    let calls = join.calls();
    assert_eq!(
        calls.len(),
        2,
        "the join fires once for the b/c/d layer and again for a2: {calls:?}"
    );
    assert_eq!(
        calls[0],
        json!({"a2": null, "b": "b", "c": "c", "d": "d"}),
        "the first arrival resolves the pending producer's key to null"
    );
    assert_eq!(
        calls[1],
        json!({"a2": "a2", "b": "b", "c": "c", "d": "d"}),
        "the later arrival sees the completed state"
    );
    assert_eq!(output.result, json!({"output": {"join": true}}));
}

/// A router selecting the branch named by the workflow's `which` input.
fn pick_branch() -> Arc<dyn Router> {
    Arc::new(FnRouter::new("pick_branch", |ctx: &RouterContext| {
        let which: Option<Value> = ctx
            .state()
            .get("start")
            .and_then(|start| start.get("which"))
            .cloned();
        Box::pin(async move {
            let target = if which == Some(json!("a")) { "a" } else { "b" };
            Ok(vec![target.to_string()])
        })
    }))
}

/// `start --(conditional)--> {a, b}`, `{a, b} → join`, `join → end`.
fn alternative_branches(wait_for_all: Option<bool>) -> (Workflow, FixedComponent) {
    let join = FixedComponent::new(json!({"joined": true}));
    let mut options =
        ComponentOptions::new().with_inputs_schema(json!({"a": "${a.value}", "b": "${b.value}"}));
    if let Some(wait_for_all) = wait_for_all {
        options = options.with_wait_for_all(wait_for_all);
    }
    let router = pick_branch();
    let mut workflow = Workflow::new(card("fan-in-alternative"));
    workflow
        .set_start_comp(
            "start",
            StartComponent::new(),
            ComponentOptions::new().with_inputs_schema(json!({"which": "${inputs.which}"})),
        )
        .add_workflow_comp(
            "a",
            FixedComponent::new(json!({"value": "a"})),
            ComponentOptions::new(),
        )
        .add_workflow_comp(
            "b",
            FixedComponent::new(json!({"value": "b"})),
            ComponentOptions::new(),
        )
        .add_workflow_comp("join", join.clone(), options)
        .set_end_comp(
            "end",
            EndComponent::new(),
            ComponentOptions::new().with_inputs_schema(json!({"join": "${join.joined}"})),
        )
        .add_conditional_connection("start", "a", Arc::clone(&router))
        .add_conditional_connection("start", "b", router)
        .add_connection("a", "join")
        .add_connection("b", "join")
        .add_connection("join", "end");
    (workflow, join)
}

#[tokio::test]
async fn first_arrival_joins_only_the_branch_that_ran() {
    let (workflow, join) = alternative_branches(Some(false));
    let output = workflow
        .invoke(json!({"which": "a"}), session("fan-in-alternative-first"))
        .await
        .unwrap();

    assert_eq!(output.state, WorkflowExecutionState::Completed);
    assert_eq!(
        join.calls(),
        vec![json!({"a": "a", "b": null})],
        "the join fires once for the selected branch instead of waiting for both"
    );
}

#[tokio::test]
async fn a_barrier_join_with_wait_for_all_false_never_runs_a_partial_branch() {
    // The CNF baseline for the same topology: the unselected branch never arrives, so the
    // join keeps waiting and the end node never runs.
    let (workflow, join) = alternative_branches(None);
    let output = workflow
        .invoke(json!({"which": "a"}), session("fan-in-alternative-cnf"))
        .await
        .unwrap();

    assert_eq!(output.state, WorkflowExecutionState::Completed);
    assert_eq!(
        join.call_count(),
        0,
        "a barrier fan-in over two mutually exclusive branches never becomes ready"
    );
    assert_eq!(output.result, Value::Null);
}

#[tokio::test]
async fn a_single_source_node_ignores_wait_for_all_false() {
    let solo = FixedComponent::new(json!({"value": "solo"}));
    let mut workflow = Workflow::new(card("fan-in-single"));
    workflow
        .set_start_comp(
            "start",
            StartComponent::new(),
            ComponentOptions::new().with_inputs_schema(json!({"query": "${inputs.query}"})),
        )
        .add_workflow_comp(
            "solo",
            solo.clone(),
            ComponentOptions::new()
                .with_inputs_schema(json!({"query": "${start.query}"}))
                .with_wait_for_all(false),
        )
        .set_end_comp(
            "end",
            EndComponent::new(),
            ComponentOptions::new().with_inputs_schema(json!({"solo": "${solo.value}"})),
        )
        .add_connection("start", "solo")
        .add_connection("solo", "end");

    let output = workflow
        .invoke(json!({"query": "hi"}), session("fan-in-single"))
        .await
        .unwrap();
    assert_eq!(output.state, WorkflowExecutionState::Completed);
    assert_eq!(solo.calls(), vec![json!({"query": "hi"})]);
    assert_eq!(output.result, json!({"output": {"solo": "solo"}}));
}

#[tokio::test]
async fn a_barrier_form_fan_in_also_fires_on_the_first_arrival() {
    let join = FixedComponent::new(json!({"joined": true}));
    let mut workflow = Workflow::new(card("fan-in-barrier"));
    workflow
        .set_start_comp(
            "start",
            StartComponent::new(),
            ComponentOptions::new().with_inputs_schema(json!({"query": "${inputs.query}"})),
        )
        .add_workflow_comp(
            "a",
            FixedComponent::new(json!({"value": "a"})),
            ComponentOptions::new(),
        )
        .add_workflow_comp(
            "b",
            FixedComponent::new(json!({"value": "b"})),
            ComponentOptions::new(),
        )
        .add_workflow_comp(
            "b2",
            FixedComponent::new(json!({"value": "b2"})),
            ComponentOptions::new(),
        )
        .add_workflow_comp(
            "join",
            join.clone(),
            ComponentOptions::new()
                .with_inputs_schema(json!({"a": "${a.value}", "b2": "${b2.value}"}))
                .with_wait_for_all(false),
        )
        .set_end_comp(
            "end",
            EndComponent::new(),
            ComponentOptions::new().with_inputs_schema(json!({"join": "${join.joined}"})),
        )
        .add_connection("start", "a")
        .add_connection("start", "b")
        .add_connection("b", "b2")
        .add_connections(&["a", "b2"], "join")
        .add_connection("join", "end");

    let output = workflow
        .invoke(json!({"query": "hi"}), session("fan-in-barrier-first"))
        .await
        .unwrap();
    assert_eq!(output.state, WorkflowExecutionState::Completed);
    assert_eq!(
        join.calls(),
        vec![json!({"a": "a", "b2": null}), json!({"a": "a", "b2": "b2"}),],
        "the barrier edge's sources activate the join independently"
    );
}

#[tokio::test]
async fn a_conditional_source_keeps_its_routing_when_it_also_feeds_a_first_arrival_node() {
    let join = FixedComponent::new(json!({"joined": true}));
    let router = pick_branch();
    let mut workflow = Workflow::new(card("fan-in-mixed"));
    workflow
        .set_start_comp(
            "start",
            StartComponent::new(),
            ComponentOptions::new().with_inputs_schema(json!({"which": "${inputs.which}"})),
        )
        .add_workflow_comp(
            "a",
            FixedComponent::new(json!({"value": "a"})),
            ComponentOptions::new(),
        )
        .add_workflow_comp(
            "b",
            FixedComponent::new(json!({"value": "b"})),
            ComponentOptions::new(),
        )
        .add_workflow_comp(
            "join",
            join.clone(),
            ComponentOptions::new()
                .with_inputs_schema(json!({"a": "${a.value}"}))
                .with_wait_for_all(false),
        )
        .set_end_comp(
            "end",
            EndComponent::new(),
            ComponentOptions::new().with_inputs_schema(json!({"a": "${a.value}"})),
        )
        .add_conditional_connection("start", "a", Arc::clone(&router))
        .add_conditional_connection("start", "b", router)
        .add_connection("start", "join")
        .add_connection("a", "join")
        .add_connection("b", "join")
        .add_connection("join", "end");

    let output = workflow
        .invoke(json!({"which": "a"}), session("fan-in-mixed"))
        .await
        .unwrap();
    assert_eq!(output.state, WorkflowExecutionState::Completed);
    assert_eq!(
        join.calls(),
        vec![json!({"a": null}), json!({"a": "a"})],
        "start always reaches the join, while the selected branch arrives later"
    );
}

#[tokio::test]
async fn a_collect_node_still_rejects_wait_for_all_false() {
    let mut workflow = Workflow::new(card("fan-in-collect"));
    workflow
        .set_start_comp("start", StartComponent::new(), ComponentOptions::new())
        .add_workflow_comp(
            "collect",
            CollectorComponent::new(),
            ComponentOptions::new()
                .with_abilities(vec![ComponentAbility::Collect])
                .with_wait_for_all(false),
        )
        .set_end_comp("end", EndComponent::new(), ComponentOptions::new())
        .add_connection("start", "collect")
        .add_connection("collect", "end");

    let error = workflow.build().unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Workflow, "unexpected: {error}");
    let message = error.message().into_owned();
    assert!(message.contains("[100011]"), "{message}");
    assert!(
        message.contains("must set 'wait_for_all' to True"),
        "{message}"
    );
}
/// The Mermaid node number the renderer assigned to `id`.
///
/// A rendered node is `node_<n>["<id>"]`, or `node_<n>("<id>")` for the entry/exit shapes.
fn mermaid_node(mermaid: &str, id: &str) -> usize {
    let shapes = [format!("[\"{id}\"]"), format!("(\"{id}\")")];
    let line = mermaid
        .lines()
        .find(|line| shapes.iter().any(|shape| line.contains(shape)))
        .unwrap_or_else(|| panic!("no node '{id}' in:\n{mermaid}"));
    line.trim()
        .strip_prefix("node_")
        .and_then(|rest| {
            let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
            digits.parse().ok()
        })
        .unwrap_or_else(|| panic!("no node number in '{line}'"))
}

#[test]
fn to_mermaid_draws_every_predecessor_of_a_first_arrival_node() {
    let LayeredFanIn { workflow, .. } = layered_fan_in(Some(false));
    let mermaid = workflow.to_mermaid().unwrap();
    let join = mermaid_node(&mermaid, "join");
    for source in ["a2", "b", "c", "d"] {
        let from = mermaid_node(&mermaid, source);
        let edge = format!("node_{from} --> node_{join}");
        assert!(
            mermaid.contains(&edge),
            "expected '{edge}' (render-only compiles keep the recorded CNF edges):\n{mermaid}"
        );
    }
}

#[tokio::test]
async fn rendering_a_first_arrival_workflow_leaves_the_run_path_unchanged() {
    let LayeredFanIn { workflow, join, .. } = layered_fan_in(Some(false));
    // The render path compiles the recorded topology; it must not affect what `invoke`
    // compiles afterwards.
    let mermaid = workflow.to_mermaid().unwrap();
    assert!(!mermaid.contains("could not be compiled"), "{mermaid}");
    let output = workflow
        .invoke(json!({"query": "hi"}), session("fan-in-first-rendered"))
        .await
        .unwrap();
    assert_eq!(output.state, WorkflowExecutionState::Completed);
    let calls = join.calls();
    assert_eq!(
        calls.len(),
        2,
        "the join still fires once per arriving layer: {calls:?}"
    );
    assert_eq!(calls[0], json!({"a2": null, "b": "b", "c": "c", "d": "d"}));
    assert_eq!(calls[1], json!({"a2": "a2", "b": "b", "c": "c", "d": "d"}));
}
