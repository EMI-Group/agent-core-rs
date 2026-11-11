//! Compile-time and run-time validation.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]

mod common;

use serde_json::json;

use openjiuwen_common::error::ErrorKind;
use openjiuwen_graph::{
    ExecutionOptions, FnRouter, GraphBuilder, GraphContext, GraphState, NodeOutcome, RouterContext,
    SyncNode,
};

#[test]
fn empty_graph_is_rejected() {
    let error = GraphBuilder::new("empty").compile().unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Graph);
    assert!(error.message().contains("graph `empty` has no nodes"));
}

#[test]
fn duplicate_node_ids_are_rejected() {
    let error = GraphBuilder::new("dup")
        .node("a", common::noop("a"))
        .node("a", common::noop("a"))
        .compile()
        .unwrap_err();
    assert!(error.message().contains("duplicate node id `a`"));
}

#[test]
fn unknown_edge_endpoints_are_rejected() {
    let target = GraphBuilder::new("edges")
        .node("a", common::noop("a"))
        .edge("a", "ghost")
        .compile()
        .unwrap_err();
    assert!(target
        .message()
        .contains("edge target `ghost` is not a node"));

    let source = GraphBuilder::new("edges")
        .node("a", common::noop("a"))
        .edge("ghost", "a")
        .compile()
        .unwrap_err();
    assert!(source
        .message()
        .contains("edge source `ghost` is not a node"));
}

#[test]
fn unknown_router_source_is_rejected() {
    let error = GraphBuilder::new("routers")
        .node("a", common::noop("a"))
        .conditional_edges(
            "ghost",
            FnRouter::new("ghost-router", |_ctx: &RouterContext| {
                Box::pin(async { Ok(Vec::new()) })
            }),
        )
        .compile()
        .unwrap_err();
    // The spec reports the unknown edge source first; either way the router
    // never compiles against a node that does not exist.
    assert!(error.message().contains("`ghost` is not a node"));
}

#[test]
fn missing_entry_is_rejected() {
    let error = GraphBuilder::new("cycle")
        .node("a", common::noop("a"))
        .node("b", common::noop("b"))
        .edge("a", "b")
        .edge("b", "a")
        .compile()
        .unwrap_err();
    assert!(error.message().contains("graph has no entry nodes"));
}

#[test]
fn unknown_entry_is_rejected() {
    let error = GraphBuilder::new("entry")
        .node("a", common::noop("a"))
        .entry("ghost")
        .compile()
        .unwrap_err();
    assert!(error.message().contains("entry `ghost` is not a node"));
}

#[test]
fn empty_barrier_is_rejected() {
    let error = GraphBuilder::new("barrier")
        .node("a", common::noop("a"))
        .barrier_edge(Vec::<String>::new(), "a")
        .compile()
        .unwrap_err();
    assert!(error
        .message()
        .contains("barrier edge to `a` has no sources"));
}

#[test]
fn every_problem_is_reported_in_one_error() {
    let error = GraphBuilder::new("many")
        .node("a", common::noop("a"))
        .node("a", common::noop("a"))
        .edge("a", "ghost")
        .compile()
        .unwrap_err();
    let message = error.message();
    assert!(message.contains("duplicate node id `a`"), "{message}");
    assert!(
        message.contains("edge target `ghost` is not a node"),
        "{message}"
    );
    assert!(message.contains("; "));
}

#[tokio::test]
async fn route_outcome_to_an_unknown_node_fails_the_run() {
    let graph = GraphBuilder::new("route")
        .node(
            "a",
            SyncNode::new("a", |_state: &mut GraphState, _ctx: &GraphContext| {
                Ok(NodeOutcome::route(["ghost"]))
            }),
        )
        .compile()
        .unwrap();
    let error = graph
        .run(json!({}), common::session(), ExecutionOptions::default())
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Graph);
    assert!(
        error.message().contains("unknown node `ghost`"),
        "{}",
        error.message()
    );
}

#[tokio::test]
async fn router_returning_an_unknown_node_fails_the_run() {
    let graph = GraphBuilder::new("route")
        .node("a", common::noop("a"))
        .node("b", common::noop("b"))
        .conditional_edges_to(
            "a",
            ["b"],
            FnRouter::new("bad", |_ctx: &RouterContext| {
                Box::pin(async { Ok(vec!["ghost".to_string()]) })
            }),
        )
        .compile()
        .unwrap();
    let error = graph
        .run(json!({}), common::session(), ExecutionOptions::default())
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Graph);
    assert!(
        error.message().contains("unknown node `ghost`"),
        "{}",
        error.message()
    );
}
