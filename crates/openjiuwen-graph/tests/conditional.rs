//! Conditional routing behaviour.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]

mod common;

use serde_json::json;

use openjiuwen_graph::{
    CompiledGraph, ExecutionOptions, FnRouter, GraphBuilder, NodeOutcome, RouterContext, SyncNode,
};
use openjiuwen_graph::{GraphContext, GraphState};

fn branch_graph() -> CompiledGraph {
    let router = FnRouter::new("branch", |ctx: &RouterContext| {
        let go = ctx
            .state()
            .get("go")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
        Box::pin(async move {
            Ok(if go {
                vec!["yes".to_string()]
            } else {
                vec!["no".to_string()]
            })
        })
    });
    GraphBuilder::new("branch")
        .node("check", common::noop("check"))
        .node("yes", common::marker("yes"))
        .node("no", common::marker("no"))
        .conditional_edges_to("check", ["yes", "no"], router)
        .compile()
        .unwrap()
}

#[tokio::test]
async fn router_branches_on_state() {
    let graph = branch_graph();
    let taken = graph
        .run(
            json!({"go": true}),
            common::session(),
            ExecutionOptions::default(),
        )
        .await
        .unwrap();
    assert_eq!(taken.visited, vec!["check".to_string(), "yes".to_string()]);
    assert_eq!(taken.state.get("ran_yes"), Some(&json!(true)));
    assert_eq!(taken.state.get("ran_no"), None);

    let skipped = graph
        .run(
            json!({"go": false}),
            common::session(),
            ExecutionOptions::default(),
        )
        .await
        .unwrap();
    assert_eq!(skipped.visited, vec!["check".to_string(), "no".to_string()]);
    assert_eq!(skipped.state.get("ran_no"), Some(&json!(true)));
}

#[tokio::test]
async fn multiple_conditionals_union_their_targets() {
    let graph = GraphBuilder::new("union")
        .node("source", common::noop("source"))
        .node("a", common::marker("a"))
        .node("b", common::marker("b"))
        .entry("source")
        .conditional_edges(
            "source",
            FnRouter::new("to_a", |_ctx: &RouterContext| {
                Box::pin(async { Ok(vec!["a".to_string()]) })
            }),
        )
        .conditional_edges(
            "source",
            FnRouter::new("to_b", |_ctx: &RouterContext| {
                Box::pin(async { Ok(vec!["b".to_string()]) })
            }),
        )
        .compile()
        .unwrap();
    let result = graph
        .run(json!({}), common::session(), ExecutionOptions::default())
        .await
        .unwrap();
    assert_eq!(
        result.visited,
        vec!["source".to_string(), "a".to_string(), "b".to_string()]
    );
    assert_eq!(result.state.get("ran_a"), Some(&json!(true)));
    assert_eq!(result.state.get("ran_b"), Some(&json!(true)));
}

#[tokio::test]
async fn static_edges_and_routers_both_fire() {
    let graph = GraphBuilder::new("mixed")
        .node("source", common::noop("source"))
        .node("always", common::marker("always"))
        .node("sometimes", common::marker("sometimes"))
        .edge("source", "always")
        .conditional_edges_to(
            "source",
            ["sometimes"],
            FnRouter::new("sometimes", |_ctx: &RouterContext| {
                Box::pin(async { Ok(vec!["sometimes".to_string()]) })
            }),
        )
        .compile()
        .unwrap();
    let result = graph
        .run(json!({}), common::session(), ExecutionOptions::default())
        .await
        .unwrap();
    assert_eq!(
        result.visited,
        vec![
            "source".to_string(),
            "always".to_string(),
            "sometimes".to_string()
        ]
    );
}

#[tokio::test]
async fn router_can_route_to_a_subset_with_route_outcome() {
    let graph = GraphBuilder::new("subset")
        .node(
            "source",
            SyncNode::new("source", |_state: &mut GraphState, _ctx: &GraphContext| {
                Ok(NodeOutcome::route(["b"]))
            }),
        )
        .node("a", common::marker("a"))
        .node("b", common::marker("b"))
        .edge("source", "a")
        .edge("source", "b")
        .compile()
        .unwrap();
    let result = graph
        .run(json!({}), common::session(), ExecutionOptions::default())
        .await
        .unwrap();
    assert_eq!(result.visited, vec!["source".to_string(), "b".to_string()]);
    assert_eq!(result.state.get("ran_a"), None);
}

#[tokio::test]
async fn empty_route_ends_the_branch() {
    let graph = GraphBuilder::new("dead-end")
        .node("source", common::noop("source"))
        .node("a", common::marker("a"))
        .conditional_edges_to(
            "source",
            ["a"],
            FnRouter::new("nowhere", |_ctx: &RouterContext| {
                Box::pin(async { Ok(Vec::new()) })
            }),
        )
        .compile()
        .unwrap();
    let result = graph
        .run(json!({}), common::session(), ExecutionOptions::default())
        .await
        .unwrap();
    assert_eq!(result.visited, vec!["source".to_string()]);
    assert_eq!(result.supersteps, 1);
}

#[tokio::test]
async fn empty_route_outcome_falls_back_to_routers() {
    let graph = GraphBuilder::new("fallback")
        .node(
            "source",
            SyncNode::new("source", |_state: &mut GraphState, _ctx: &GraphContext| {
                Ok(NodeOutcome::Route {
                    targets: Vec::new(),
                })
            }),
        )
        .node("fallback", common::marker("fallback"))
        .edge("source", "fallback")
        .compile()
        .unwrap();
    let result = graph
        .run(json!({}), common::session(), ExecutionOptions::default())
        .await
        .unwrap();
    assert_eq!(result.state.get("ran_fallback"), Some(&json!(true)));
}

#[tokio::test]
async fn router_sees_state_written_by_its_node() {
    let graph = GraphBuilder::new("stateful")
        .node("source", common::counting("source"))
        .node("next", common::marker("next"))
        .conditional_edges_to(
            "source",
            ["next"],
            FnRouter::new("count", |ctx: &RouterContext| {
                let count = ctx
                    .state()
                    .get("count")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0);
                Box::pin(async move {
                    Ok(if count == 1 {
                        vec!["next".to_string()]
                    } else {
                        Vec::new()
                    })
                })
            }),
        )
        .compile()
        .unwrap();
    let result = graph
        .run(json!({}), common::session(), ExecutionOptions::default())
        .await
        .unwrap();
    assert_eq!(
        result.visited,
        vec!["source".to_string(), "next".to_string()]
    );
}

#[tokio::test]
async fn router_context_exposes_session_and_metadata() {
    let graph = GraphBuilder::new("meta")
        .node("source", common::noop("source"))
        .node("next", common::marker("next"))
        .metadata("mode", json!("test"))
        .conditional_edges_to(
            "source",
            ["next"],
            FnRouter::new("inspect", |ctx: &RouterContext| {
                let ok = ctx.source() == "source"
                    && ctx.superstep() == 1
                    && ctx.session().id() == "s1"
                    && ctx.metadata().get("mode") == Some(&json!("test"));
                Box::pin(async move {
                    Ok(if ok {
                        vec!["next".to_string()]
                    } else {
                        Vec::new()
                    })
                })
            }),
        )
        .compile()
        .unwrap();
    let result = graph
        .run(json!({}), common::session(), ExecutionOptions::default())
        .await
        .unwrap();
    assert_eq!(result.state.get("ran_next"), Some(&json!(true)));
}
