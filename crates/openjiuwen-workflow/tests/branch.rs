#![allow(clippy::unwrap_used, clippy::expect_used)]
#![allow(
    unreachable_pub,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    clippy::new_without_default,
    clippy::module_name_repetitions,
    clippy::unnecessary_literal_bound
)]
//! Conditional fan-out: `BranchComponent` + `BranchRouter` and graph-level routers.
//!
//! A branch component produces no data; the engine asks `Component::route` which
//! successors to activate, and the graph activates exactly those. These tests pin the
//! selected-target rule, the Rust-only default target, the `101021` no-match error, custom
//! `openjiuwen_graph::Router` fan-out, undeclared targets, router unions, and empty routes.
//!
//! Ports of `tests/unit_tests/core/workflow/test_workflow.py::test_workflow_with_branch`
//! and `::test_simple_workflow_with_branch_condition`, plus the branch-router cases of
//! `tests/unit_tests/core/component/test_branch_comp.py`. Every test is offline and
//! deterministic: counters are atomics read after the run, never sleeps.
mod common;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use openjiuwen_common::error::ErrorKind;
use openjiuwen_graph::{FnRouter, Router, RouterContext};
use serde_json::{json, Value};

use openjiuwen_workflow::{
    Branch, BranchComponent, BranchRouter, ComponentOptions, Condition, EndComponent,
    StartComponent, Workflow, WorkflowExecutionState,
};

use common::{card, session, CounterComponent};

/// A graph router that answers `targets` and counts how often it was consulted.
fn counting_router(targets: Vec<String>, calls: Arc<AtomicUsize>) -> Arc<dyn Router> {
    Arc::new(FnRouter::new("counting", move |_ctx: &RouterContext| {
        let targets = targets.clone();
        let calls = Arc::clone(&calls);
        Box::pin(async move {
            calls.fetch_add(1, Ordering::SeqCst);
            Ok(targets)
        })
    }))
}

/// `start → branch → {a, b}`, where only `end_source` has an edge to `end`.
///
/// Both targets are declared as successors of the branch so the graph has a channel for
/// each; the branch's own `route` decides which one actually runs.
fn branch_workflow(
    router: BranchRouter,
    end_source: &str,
) -> (Workflow, CounterComponent, CounterComponent) {
    let a = CounterComponent::new();
    let b = CounterComponent::new();
    let mut workflow = Workflow::new(card("branch"));
    workflow
        .set_start_comp(
            "start",
            StartComponent::new(),
            ComponentOptions::new().with_inputs_schema(json!({"which": "${inputs.which}"})),
        )
        .add_workflow_comp(
            "branch",
            BranchComponent::new(router),
            ComponentOptions::new(),
        )
        .add_workflow_comp("a", a.clone(), ComponentOptions::new())
        .add_workflow_comp("b", b.clone(), ComponentOptions::new())
        .set_end_comp(
            "end",
            EndComponent::new(),
            ComponentOptions::new()
                .with_inputs_schema(json!({"a": "${a.count}", "b": "${b.count}"})),
        )
        .add_connection("start", "branch")
        .add_connection("branch", "a")
        .add_connection("branch", "b")
        .add_connection(end_source, "end");
    (workflow, a, b)
}

/// A router selecting `a` on `which == "a"` and `b` on `which == "b"`.
fn two_branch_router() -> BranchRouter {
    let mut router = BranchRouter::new();
    let _ = router
        .add_branch(Branch::new(
            Condition::new("${start.which} == 'a'").unwrap(),
            vec!["a".to_owned()],
        ))
        .add_branch(Branch::new(
            Condition::new("${start.which} == 'b'").unwrap(),
            vec!["b".to_owned()],
        ));
    router
}

// port of tests/unit_tests/core/workflow/test_workflow.py::test_workflow_with_branch
#[tokio::test]
async fn a_branch_runs_only_the_selected_target() {
    let (workflow, a, b) = branch_workflow(two_branch_router(), "a");
    let output = workflow
        .invoke(json!({"which": "a"}), session("branch-a"))
        .await
        .unwrap();
    assert_eq!(output.state, WorkflowExecutionState::Completed);
    assert_eq!(a.count(), 1, "the selected target ran once");
    assert_eq!(b.count(), 0, "the unselected target never ran");
    assert_eq!(output.result, json!({"output": {"a": 1}}));

    let (workflow, a, b) = branch_workflow(two_branch_router(), "b");
    let output = workflow
        .invoke(json!({"which": "b"}), session("branch-b"))
        .await
        .unwrap();
    assert_eq!(a.count(), 0);
    assert_eq!(b.count(), 1);
    assert_eq!(output.result, json!({"output": {"b": 1}}));

    // GAP: Python's `test_workflow_with_branch` declares `a -> end` and `b -> end` in the
    // same workflow (its barrier passes as soon as one predecessor completes). The Rust
    // graph compiles one CNF group per incoming edge, so `end` waits for BOTH targets and
    // never runs when only one branch is taken. The port therefore wires only the selected
    // target to `end`; the deviation is pinned here so a future OR-barrier fix is visible.
    let (mut both, a, b) = branch_workflow(two_branch_router(), "a");
    both.add_connection("b", "end");
    let output = both
        .invoke(json!({"which": "a"}), session("branch-both"))
        .await
        .unwrap();
    assert_eq!(output.state, WorkflowExecutionState::Completed);
    assert_eq!(a.count(), 1);
    assert_eq!(b.count(), 0);
    assert_eq!(
        output.result,
        Value::Null,
        "with both targets wired to `end` the CNF barrier still waits for `b`"
    );
}

// port of tests/unit_tests/core/workflow/test_workflow.py::test_simple_workflow_with_branch_condition
#[tokio::test]
async fn the_default_target_handles_a_missing_match() {
    let mut router = BranchRouter::new();
    let _ = router
        .add_branch(Branch::new(
            Condition::new("${start.which} == 'a'").unwrap(),
            vec!["a".to_owned()],
        ))
        .default_target(vec!["b".to_owned()]);

    let (workflow, a, b) = branch_workflow(router, "b");
    let output = workflow
        .invoke(json!({"which": "zzz"}), session("branch-default"))
        .await
        .unwrap();

    assert_eq!(output.state, WorkflowExecutionState::Completed);
    assert_eq!(a.count(), 0);
    assert_eq!(b.count(), 1, "the default target receives the flow");
    assert_eq!(output.result, json!({"output": {"b": 1}}));
}

// port of tests/unit_tests/core/component/test_branch_comp.py::test_branch_no_match
#[tokio::test]
async fn no_match_without_a_default_is_code_101021() {
    let mut router = BranchRouter::new();
    let _ = router.add_branch(Branch::new(
        Condition::new("${start.which} == 'a'").unwrap(),
        vec!["a".to_owned()],
    ));

    let (workflow, a, b) = branch_workflow(router, "a");
    let failure = workflow
        .invoke(json!({"which": "zzz"}), session("branch-no-match"))
        .await
        .unwrap_err();

    assert_eq!(failure.kind(), ErrorKind::Workflow);
    assert!(failure.message().contains("[101021]"), "{failure}");
    assert_eq!(a.count(), 0);
    assert_eq!(b.count(), 0);
}

// port of tests/unit_tests/core/workflow/test_workflow.py::test_simple_workflow_with_branch_condition
#[tokio::test]
async fn a_custom_graph_router_selects_a_subset_of_the_declared_targets() {
    let a = CounterComponent::new();
    let b = CounterComponent::new();
    let calls = Arc::new(AtomicUsize::new(0));
    let router = counting_router(vec!["a".to_owned()], Arc::clone(&calls));

    let mut workflow = Workflow::new(card("custom-router"));
    workflow
        .set_start_comp("start", StartComponent::new(), ComponentOptions::new())
        .add_workflow_comp(
            "parent",
            common::EchoComponent::new(),
            ComponentOptions::new(),
        )
        .add_workflow_comp("a", a.clone(), ComponentOptions::new())
        .add_workflow_comp("b", b.clone(), ComponentOptions::new())
        .set_end_comp(
            "end",
            EndComponent::new(),
            ComponentOptions::new().with_inputs_schema(json!({"a": "${a.count}"})),
        )
        .add_connection("start", "parent")
        .add_conditional_connection("parent", "a", Arc::clone(&router))
        .add_conditional_connection("parent", "b", router)
        .add_connection("a", "end");

    let output = workflow
        .invoke(json!({}), session("custom-router"))
        .await
        .unwrap();

    assert_eq!(a.count(), 1, "the routed target ran once");
    assert_eq!(b.count(), 0, "the unrouted target never ran");
    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "the combined router consults one registration per declared target"
    );
    assert_eq!(output.result, json!({"output": {"a": 1}}));
}

// port of tests/unit_tests/core/workflow/test_workflow.py::test_workflow_invalid_router_target
#[tokio::test]
async fn an_undeclared_router_target_is_a_graph_error() {
    let a = CounterComponent::new();
    let calls = Arc::new(AtomicUsize::new(0));
    let router = counting_router(vec!["ghost".to_owned()], calls);

    let mut workflow = Workflow::new(card("undeclared"));
    workflow
        .set_start_comp("start", StartComponent::new(), ComponentOptions::new())
        .add_workflow_comp("a", a.clone(), ComponentOptions::new())
        .set_end_comp("end", EndComponent::new(), ComponentOptions::new())
        .add_conditional_connection("start", "a", router)
        .add_connection("a", "end");

    let failure = workflow
        .invoke(json!({}), session("undeclared"))
        .await
        .unwrap_err();

    assert_eq!(failure.kind(), ErrorKind::Graph);
    assert!(
        failure.message().contains("undeclared target 'ghost'"),
        "{failure}"
    );
    assert_eq!(a.count(), 0);
}

// port of tests/unit_tests/core/workflow/test_workflow.py::test_simple_workflow_with_branch_condition
#[tokio::test]
async fn agreeing_routers_activate_a_target_once() {
    let a = CounterComponent::new();
    let b = CounterComponent::new();
    let first_calls = Arc::new(AtomicUsize::new(0));
    let second_calls = Arc::new(AtomicUsize::new(0));
    let first = counting_router(vec!["a".to_owned()], Arc::clone(&first_calls));
    let second = counting_router(vec!["a".to_owned()], Arc::clone(&second_calls));

    let mut workflow = Workflow::new(card("agreeing-routers"));
    workflow
        .set_start_comp("start", StartComponent::new(), ComponentOptions::new())
        .add_workflow_comp("a", a.clone(), ComponentOptions::new())
        .add_workflow_comp("b", b.clone(), ComponentOptions::new())
        .set_end_comp(
            "end",
            EndComponent::new(),
            ComponentOptions::new().with_inputs_schema(json!({"a": "${a.count}"})),
        )
        .add_conditional_connection("start", "a", first)
        .add_conditional_connection("start", "a", second)
        .add_connection("a", "end");

    let output = workflow
        .invoke(json!({}), session("agreeing-routers"))
        .await
        .unwrap();

    assert_eq!(first_calls.load(Ordering::SeqCst), 1);
    assert_eq!(second_calls.load(Ordering::SeqCst), 1);
    assert_eq!(a.count(), 1, "the union of the routers activates 'a' once");
    assert_eq!(b.count(), 0);
    assert_eq!(output.result, json!({"output": {"a": 1}}));
}

// port of tests/unit_tests/core/component/test_branch_comp.py::test_branch_empty_targets
#[tokio::test]
async fn an_empty_route_ends_the_run_without_running_a_successor() {
    let mut router = BranchRouter::new();
    let _ = router
        .add_branch(Branch::new(
            Condition::new("${start.which} == 'a'").unwrap(),
            vec!["a".to_owned()],
        ))
        .default_target(Vec::new());

    let branch = BranchComponent::new(router);
    let adapter = branch.router_adapter();
    let a = CounterComponent::new();
    let mut workflow = Workflow::new(card("empty-route"));
    workflow
        .set_start_comp(
            "start",
            StartComponent::new(),
            ComponentOptions::new().with_inputs_schema(json!({"which": "${inputs.which}"})),
        )
        .add_workflow_comp("branch", branch, ComponentOptions::new())
        .add_workflow_comp("a", a.clone(), ComponentOptions::new())
        .set_end_comp("end", EndComponent::new(), ComponentOptions::new())
        .add_connection("start", "branch")
        .add_conditional_connection("branch", "a", adapter);

    let output = workflow
        .invoke(json!({"which": "zzz"}), session("empty-route"))
        .await
        .unwrap();

    assert_eq!(output.state, WorkflowExecutionState::Completed);
    assert_eq!(a.count(), 0, "no successor was activated");
    assert_eq!(
        output.result,
        Value::Null,
        "the end node is unreachable without an activated successor"
    );
}
