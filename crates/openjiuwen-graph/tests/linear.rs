//! Linear graph behaviour.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]

mod common;

use serde_json::{json, Value};

use openjiuwen_graph::{ExecutionOptions, GraphBuilder};

#[tokio::test]
async fn linear_chain_threads_state() {
    let graph = GraphBuilder::new("linear")
        .node("a", common::counting("a"))
        .node("b", common::counting("b"))
        .node("c", common::counting("c"))
        .edge("a", "b")
        .edge("b", "c")
        .exit("c")
        .compile()
        .unwrap();

    let result = graph
        .run(json!({}), common::session(), ExecutionOptions::default())
        .await
        .unwrap();

    assert_eq!(result.state.get("count"), Some(&json!(3)));
    assert_eq!(
        result.visited,
        vec!["a".to_string(), "b".to_string(), "c".to_string()]
    );
    assert_eq!(result.supersteps, 3);
    assert_eq!(result.graph_id, "linear");
    assert!(!result.resumed);
    assert!(result.halted.is_none());
}

#[tokio::test]
async fn input_seeds_the_state_and_invoke_returns_it() {
    let graph = GraphBuilder::new("linear")
        .node("a", common::counting("a"))
        .compile()
        .unwrap();

    let value = graph
        .invoke(json!({"count": 41, "keep": "yes"}), common::session())
        .await
        .unwrap();
    assert_eq!(value, json!({"count": 42, "keep": "yes"}));
}

#[tokio::test]
async fn linear_chain_runs_one_node_per_superstep() {
    use futures::StreamExt;

    let graph = GraphBuilder::new("linear")
        .node("a", common::noop("a"))
        .node("b", common::noop("b"))
        .edge("a", "b")
        .compile()
        .unwrap();

    let events: Vec<_> = graph
        .stream(json!({}), common::session())
        .map(|event| event.unwrap())
        .collect()
        .await;

    let supersteps: Vec<u32> = events
        .iter()
        .filter(|event| event.kind == openjiuwen_graph::GraphEventKind::SuperstepStarted)
        .map(|event| event.superstep)
        .collect();
    assert_eq!(supersteps, vec![1, 2]);

    let started: Vec<String> = events
        .iter()
        .filter(|event| event.kind == openjiuwen_graph::GraphEventKind::NodeStarted)
        .filter_map(|event| event.node_id.clone())
        .collect();
    assert_eq!(started, vec!["a".to_string(), "b".to_string()]);
}

#[tokio::test]
async fn isolated_node_runs_once() {
    let graph = GraphBuilder::new("single")
        .node("only", common::counting("only"))
        .compile()
        .unwrap();
    let result = graph
        .run(
            Value::Object(serde_json::Map::new()),
            common::session(),
            ExecutionOptions::default(),
        )
        .await
        .unwrap();
    assert_eq!(result.supersteps, 1);
    assert_eq!(result.visited, vec!["only".to_string()]);
    assert_eq!(result.state.get("count"), Some(&json!(1)));
}
