//! Parallel fan-out and barrier fan-in behaviour.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]

mod common;

use serde_json::json;

use openjiuwen_graph::SyncNode;
use openjiuwen_graph::{ExecutionOptions, GraphBuilder, GraphEventKind};
use openjiuwen_graph::{GraphContext, GraphState, NodeOutcome};

fn diamond() -> openjiuwen_graph::CompiledGraph {
    GraphBuilder::new("diamond")
        .node("start", common::counting("start"))
        .node("a", common::marker("a"))
        .node("b", common::marker("b"))
        .node("join", common::marker("join"))
        .node("end", common::counting("end"))
        .edge("start", "a")
        .edge("start", "b")
        .barrier_edge(["a", "b"], "join")
        .edge("join", "end")
        .exit("end")
        .compile()
        .unwrap()
}

#[tokio::test]
async fn fan_out_runs_in_parallel_and_barrier_joins() {
    let result = diamond()
        .run(json!({}), common::session(), ExecutionOptions::default())
        .await
        .unwrap();

    assert_eq!(
        result.visited,
        vec![
            "start".to_string(),
            "a".to_string(),
            "b".to_string(),
            "join".to_string(),
            "end".to_string()
        ]
    );
    assert_eq!(result.supersteps, 4);
    assert_eq!(result.state.get("ran_a"), Some(&json!(true)));
    assert_eq!(result.state.get("ran_b"), Some(&json!(true)));
    assert_eq!(result.state.get("ran_join"), Some(&json!(true)));
    assert_eq!(result.state.get("count"), Some(&json!(2)));
}

#[tokio::test]
async fn barrier_waits_for_every_source_in_the_same_superstep() {
    use futures::StreamExt;

    let events: Vec<_> = diamond()
        .stream(json!({}), common::session())
        .map(|event| event.unwrap())
        .collect()
        .await;

    let join_superstep = events
        .iter()
        .find(|event| {
            event.kind == GraphEventKind::NodeStarted && event.node_id.as_deref() == Some("join")
        })
        .map(|event| event.superstep)
        .unwrap();
    assert_eq!(join_superstep, 3);
}

#[tokio::test]
async fn declaration_order_wins_on_scalar_conflicts() {
    let graph = GraphBuilder::new("conflict")
        .node(
            "a",
            SyncNode::new("a", |state: &mut GraphState, _ctx: &GraphContext| {
                state.set("who", json!("a"));
                Ok(NodeOutcome::Continue)
            }),
        )
        .node(
            "b",
            SyncNode::new("b", |state: &mut GraphState, _ctx: &GraphContext| {
                state.set("who", json!("b"));
                Ok(NodeOutcome::Continue)
            }),
        )
        .barrier_edge(["a", "b"], "join")
        .node("join", common::noop("join"))
        .compile()
        .unwrap();

    let result = graph
        .run(json!({}), common::session(), ExecutionOptions::default())
        .await
        .unwrap();
    assert_eq!(result.state.get("who"), Some(&json!("b")));
}

#[tokio::test]
async fn parallel_runs_are_deterministic() {
    let graph = diamond();
    let first = graph
        .run(json!({}), common::session(), ExecutionOptions::default())
        .await
        .unwrap();
    for _ in 0..5 {
        let next = graph
            .run(json!({}), common::session(), ExecutionOptions::default())
            .await
            .unwrap();
        assert_eq!(next.state, first.state);
        assert_eq!(next.visited, first.visited);
        assert_eq!(next.supersteps, first.supersteps);
    }
}

#[tokio::test]
async fn fan_out_to_several_targets_from_one_node() {
    let graph = GraphBuilder::new("fan")
        .node("start", common::noop("start"))
        .node("a", common::marker("a"))
        .node("b", common::marker("b"))
        .edge("start", "a")
        .edge("start", "b")
        .compile()
        .unwrap();
    let result = graph
        .run(json!({}), common::session(), ExecutionOptions::default())
        .await
        .unwrap();
    assert_eq!(
        result.visited,
        vec!["start".to_string(), "a".to_string(), "b".to_string()]
    );
    assert_eq!(result.supersteps, 2);
}
