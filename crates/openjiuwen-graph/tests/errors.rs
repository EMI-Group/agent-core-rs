//! Node failure, retry, and timeout behaviour.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]

mod common;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use serde_json::json;

use openjiuwen_common::error::Error;
use openjiuwen_graph::{
    CompileOptions, ExecutionOptions, FnNode, GraphBuilder, GraphContext, GraphState, NodeOptions,
    NodeOutcome, SyncNode,
};

#[tokio::test]
async fn node_errors_carry_node_id_and_superstep() {
    let graph = GraphBuilder::new("errors")
        .node("start", common::noop("start"))
        .node(
            "boom",
            SyncNode::new("boom", |_state: &mut GraphState, _ctx: &GraphContext| {
                Err(Error::validation("kaboom"))
            }),
        )
        .edge("start", "boom")
        .compile()
        .unwrap();
    let error = graph
        .run(json!({}), common::session(), ExecutionOptions::default())
        .await
        .unwrap_err();
    assert_eq!(
        error.kind(),
        openjiuwen_common::error::ErrorKind::Validation
    );
    let message = error.message();
    assert!(
        message.contains("node `boom` failed at superstep 2"),
        "{message}"
    );
    assert!(message.contains("kaboom"));
}

#[tokio::test]
async fn failing_node_is_retried_max_retries_plus_one_times() {
    let attempts = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&attempts);
    let graph = GraphBuilder::new("retry")
        .node_with(
            "flaky",
            SyncNode::new(
                "flaky",
                move |_state: &mut GraphState, _ctx: &GraphContext| {
                    if counter.fetch_add(1, Ordering::SeqCst) < 2 {
                        Err(Error::validation("flaky"))
                    } else {
                        Ok(NodeOutcome::Continue)
                    }
                },
            ),
            NodeOptions {
                timeout_ms: None,
                max_retries: 2,
            },
        )
        .compile()
        .unwrap();
    let result = graph
        .run(json!({}), common::session(), ExecutionOptions::default())
        .await
        .unwrap();
    assert_eq!(attempts.load(Ordering::SeqCst), 3);
    assert_eq!(result.visited, vec!["flaky".to_string()]);
}

#[tokio::test]
async fn exhausted_retries_propagate_the_original_error() {
    let attempts = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&attempts);
    let graph = GraphBuilder::new("retry")
        .node_with(
            "boom",
            SyncNode::new(
                "boom",
                move |_state: &mut GraphState, _ctx: &GraphContext| {
                    counter.fetch_add(1, Ordering::SeqCst);
                    Err(Error::validation("still broken"))
                },
            ),
            NodeOptions {
                timeout_ms: None,
                max_retries: 1,
            },
        )
        .compile()
        .unwrap();
    let error = graph
        .run(json!({}), common::session(), ExecutionOptions::default())
        .await
        .unwrap_err();
    assert_eq!(attempts.load(Ordering::SeqCst), 2);
    assert_eq!(
        error.kind(),
        openjiuwen_common::error::ErrorKind::Validation
    );
    assert!(error.message().contains("still broken"));
}

fn slow_node() -> FnNode<
    impl for<'a> Fn(&'a mut GraphState, &'a GraphContext) -> openjiuwen_graph::BoxNodeFuture<'a>,
> {
    FnNode::new("slow", |_state: &mut GraphState, _ctx: &GraphContext| {
        Box::pin(async {
            tokio::time::sleep(Duration::from_millis(200)).await;
            Ok(NodeOutcome::Continue)
        })
    })
}

#[tokio::test]
async fn execution_timeout_overrides_node_options() {
    let graph = GraphBuilder::new("slow")
        .node_with(
            "slow",
            slow_node(),
            NodeOptions {
                timeout_ms: Some(1_000),
                max_retries: 0,
            },
        )
        .compile()
        .unwrap();
    let error = graph
        .run(
            json!({}),
            common::session(),
            ExecutionOptions {
                node_timeout_ms: Some(5),
                ..ExecutionOptions::default()
            },
        )
        .await
        .unwrap_err();
    assert_eq!(error.kind(), openjiuwen_common::error::ErrorKind::Timeout);
    assert!(error.message().contains("timed out"));
    assert!(error.message().contains("superstep 1"));
}

#[tokio::test]
async fn compile_timeout_overrides_node_options() {
    let graph = GraphBuilder::new("slow")
        .node_with(
            "slow",
            slow_node(),
            NodeOptions {
                timeout_ms: Some(1_000),
                max_retries: 0,
            },
        )
        .compile_with(CompileOptions {
            node_timeout_ms: Some(5),
            ..CompileOptions::default()
        })
        .unwrap();
    let error = graph
        .run(json!({}), common::session(), ExecutionOptions::default())
        .await
        .unwrap_err();
    assert_eq!(error.kind(), openjiuwen_common::error::ErrorKind::Timeout);
}

#[tokio::test]
async fn node_option_timeout_applies_when_nothing_overrides_it() {
    let graph = GraphBuilder::new("slow")
        .node_with(
            "slow",
            slow_node(),
            NodeOptions {
                timeout_ms: Some(5),
                max_retries: 0,
            },
        )
        .compile()
        .unwrap();
    let error = graph
        .run(json!({}), common::session(), ExecutionOptions::default())
        .await
        .unwrap_err();
    assert_eq!(error.kind(), openjiuwen_common::error::ErrorKind::Timeout);
}

#[tokio::test]
async fn failing_runs_emit_node_failed_then_graph_failed() {
    use futures::StreamExt;

    let graph = GraphBuilder::new("errors")
        .node(
            "boom",
            SyncNode::new("boom", |_state: &mut GraphState, _ctx: &GraphContext| {
                Err(Error::timeout("too slow"))
            }),
        )
        .compile()
        .unwrap();
    let collected: Vec<_> = graph.stream(json!({}), common::session()).collect().await;
    let events: Vec<_> = collected
        .iter()
        .filter_map(|item| item.as_ref().ok())
        .map(|event| event.kind)
        .collect();
    let failed_at = events
        .iter()
        .position(|kind| *kind == openjiuwen_graph::GraphEventKind::NodeFailed)
        .unwrap();
    let graph_failed_at = events
        .iter()
        .position(|kind| *kind == openjiuwen_graph::GraphEventKind::GraphFailed)
        .unwrap();
    assert!(failed_at < graph_failed_at);
    assert_eq!(collected.iter().filter(|item| item.is_err()).count(), 1);
}
