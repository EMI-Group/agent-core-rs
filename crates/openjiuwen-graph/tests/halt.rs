//! Halt-as-data behaviour.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]

mod common;

use serde_json::json;

use openjiuwen_graph::{ExecutionOptions, GraphBuilder, GraphState, NodeOutcome, SyncNode};

#[tokio::test]
async fn halt_ends_the_run_after_the_superstep() {
    let graph = GraphBuilder::new("halt")
        .node("start", common::noop("start"))
        .node(
            "halt",
            SyncNode::new(
                "halt",
                |_state: &mut GraphState, _ctx: &openjiuwen_graph::GraphContext| {
                    Ok(NodeOutcome::halt(Some(json!({"reason": "enough"}))))
                },
            ),
        )
        .node("sibling", common::marker("sibling"))
        .node("after", common::marker("after"))
        .edge("start", "halt")
        .edge("start", "sibling")
        .edge("halt", "after")
        .compile()
        .unwrap();

    let result = graph
        .run(json!({}), common::session(), ExecutionOptions::default())
        .await
        .unwrap();

    let halt = result.halted.clone().unwrap();
    assert_eq!(halt.node_id, "halt");
    assert_eq!(halt.superstep, 2);
    assert_eq!(halt.value, Some(json!({"reason": "enough"})));

    // Siblings of the halting node still complete, downstream nodes do not run.
    assert_eq!(result.state.get("ran_sibling"), Some(&json!(true)));
    assert_eq!(result.state.get("ran_after"), None);
    assert_eq!(
        result.visited,
        vec![
            "start".to_string(),
            "halt".to_string(),
            "sibling".to_string()
        ]
    );
    assert_eq!(result.supersteps, 2);
}

#[tokio::test]
async fn halt_without_a_value_is_preserved() {
    let graph = GraphBuilder::new("halt")
        .node(
            "only",
            SyncNode::new(
                "only",
                |_state: &mut GraphState, _ctx: &openjiuwen_graph::GraphContext| {
                    Ok(NodeOutcome::halt(None))
                },
            ),
        )
        .compile()
        .unwrap();
    let result = graph
        .run(json!({}), common::session(), ExecutionOptions::default())
        .await
        .unwrap();
    let halt = result.halted.unwrap();
    assert_eq!(halt.node_id, "only");
    assert_eq!(halt.value, None);
    assert_eq!(result.supersteps, 1);
}

#[tokio::test]
async fn halt_emits_graph_halted_before_graph_completed() {
    use futures::StreamExt;

    let graph = GraphBuilder::new("halt")
        .node(
            "only",
            SyncNode::new(
                "only",
                |_state: &mut GraphState, _ctx: &openjiuwen_graph::GraphContext| {
                    Ok(NodeOutcome::halt(Some(json!(7))))
                },
            ),
        )
        .compile()
        .unwrap();
    let events: Vec<_> = graph
        .stream(json!({}), common::session())
        .map(|event| event.unwrap())
        .collect()
        .await;
    let kinds: Vec<openjiuwen_graph::GraphEventKind> =
        events.iter().map(|event| event.kind).collect();
    let halted_at = kinds
        .iter()
        .position(|kind| *kind == openjiuwen_graph::GraphEventKind::GraphHalted)
        .unwrap();
    let completed_at = kinds
        .iter()
        .position(|kind| *kind == openjiuwen_graph::GraphEventKind::GraphCompleted)
        .unwrap();
    assert!(halted_at < completed_at);
    assert_eq!(events[completed_at].payload["halted"], json!(true));
}

#[tokio::test]
async fn halt_is_not_retried() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    let attempts = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&attempts);
    let graph = GraphBuilder::new("halt")
        .node_with(
            "only",
            SyncNode::new(
                "only",
                move |_state: &mut GraphState, _ctx: &openjiuwen_graph::GraphContext| {
                    counter.fetch_add(1, Ordering::SeqCst);
                    Ok(NodeOutcome::halt(None))
                },
            ),
            openjiuwen_graph::NodeOptions {
                timeout_ms: None,
                max_retries: 3,
            },
        )
        .compile()
        .unwrap();
    graph
        .run(json!({}), common::session(), ExecutionOptions::default())
        .await
        .unwrap();
    assert_eq!(attempts.load(Ordering::SeqCst), 1);
}
