//! Cancellation behaviour.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]

mod common;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use serde_json::json;
use tokio_util::sync::CancellationToken;

use openjiuwen_common::error::ErrorKind;
use openjiuwen_graph::{
    ExecutionOptions, FnNode, GraphBuilder, GraphContext, GraphState, NodeOptions, NodeOutcome,
};

fn slow_graph(attempts: Arc<AtomicUsize>) -> openjiuwen_graph::CompiledGraph {
    let node = FnNode::new(
        "slow",
        move |_state: &mut GraphState, _ctx: &GraphContext| {
            let attempts = Arc::clone(&attempts);
            Box::pin(async move {
                attempts.fetch_add(1, Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis(500)).await;
                Ok(NodeOutcome::Continue)
            })
        },
    );
    GraphBuilder::new("cancel")
        .node_with(
            "slow",
            node,
            NodeOptions {
                timeout_ms: None,
                max_retries: 3,
            },
        )
        .compile()
        .unwrap()
}

#[tokio::test]
async fn cancellation_before_the_run_fails_it() {
    let token = CancellationToken::new();
    token.cancel();
    let graph = GraphBuilder::new("cancel")
        .node("a", common::noop("a"))
        .compile()
        .unwrap();
    let error = graph
        .run(
            json!({}),
            common::session(),
            ExecutionOptions {
                cancellation: token,
                ..ExecutionOptions::default()
            },
        )
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Cancelled);
}

#[tokio::test]
async fn cancellation_during_a_superstep_drops_the_run() {
    let attempts = Arc::new(AtomicUsize::new(0));
    let graph = slow_graph(Arc::clone(&attempts));
    let token = CancellationToken::new();
    let handle = {
        let graph = graph.clone();
        let token = token.clone();
        tokio::spawn(async move {
            graph
                .run(
                    json!({}),
                    common::session(),
                    ExecutionOptions {
                        cancellation: token,
                        ..ExecutionOptions::default()
                    },
                )
                .await
        })
    };
    tokio::time::sleep(Duration::from_millis(20)).await;
    token.cancel();
    let error = handle.await.unwrap().unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Cancelled);
    // A cancelled attempt is never retried.
    assert_eq!(attempts.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn cancellation_between_supersteps_stops_the_run() {
    use futures::StreamExt;

    let graph = GraphBuilder::new("cancel")
        .node("a", common::noop("a"))
        .node("b", common::noop("b"))
        .edge("a", "b")
        .compile()
        .unwrap();
    let token = CancellationToken::new();
    let mut events = graph.stream_with(
        json!({}),
        common::session(),
        ExecutionOptions {
            cancellation: token.clone(),
            write_session_trace: false,
            ..ExecutionOptions::default()
        },
    );
    // Cancel before the first event is consumed; the engine sees it before
    // (or during) superstep 1 and yields exactly one error.
    token.cancel();
    let mut errors = 0;
    while let Some(item) = events.next().await {
        if let Err(error) = item {
            assert_eq!(error.kind(), ErrorKind::Cancelled);
            errors += 1;
        }
    }
    assert_eq!(errors, 1);
}

#[tokio::test]
async fn cancelled_runs_emit_graph_failed() {
    use futures::StreamExt;

    let token = CancellationToken::new();
    let graph = GraphBuilder::new("cancel")
        .node("a", common::noop("a"))
        .compile()
        .unwrap();
    token.cancel();
    let items: Vec<_> = graph
        .stream_with(
            json!({}),
            common::session(),
            ExecutionOptions {
                cancellation: token,
                write_session_trace: false,
                ..ExecutionOptions::default()
            },
        )
        .collect()
        .await;
    let kinds: Vec<_> = items
        .iter()
        .filter_map(|item| item.as_ref().ok())
        .map(|event| event.kind)
        .collect();
    assert!(kinds.contains(&openjiuwen_graph::GraphEventKind::GraphFailed));
    assert_eq!(items.iter().filter(|item| item.is_err()).count(), 1);
}
