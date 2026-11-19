//! Snapshot persistence and resume behaviour.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]

mod common;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use serde_json::json;

use openjiuwen_graph::{
    CheckpointerStore, CompileOptions, ExecutionOptions, GraphBuilder, GraphContext, GraphState,
    GraphStore, InMemoryGraphStore, NodeOutcome, SyncNode,
};
use openjiuwen_session::{CheckpointNamespace, Checkpointer, MemoryCheckpointer};

fn chain() -> GraphBuilder {
    GraphBuilder::new("chain")
        .node("a", common::counting("a"))
        .node("b", common::counting("b"))
        .node("c", common::counting("c"))
        .edge("a", "b")
        .edge("b", "c")
        .exit("c")
}

#[tokio::test]
async fn snapshot_is_saved_after_every_superstep() {
    let store = Arc::new(InMemoryGraphStore::new());
    let graph = chain()
        .compile_with(CompileOptions {
            store: Some(Arc::clone(&store) as Arc<dyn GraphStore>),
            ..CompileOptions::default()
        })
        .unwrap();
    let result = graph
        .run(json!({}), common::session(), ExecutionOptions::default())
        .await
        .unwrap();
    assert_eq!(result.supersteps, 3);

    let snapshot = store.get("s1", "chain").await.unwrap().unwrap();
    assert_eq!(snapshot.graph_id, "chain");
    assert_eq!(snapshot.namespace, "chain");
    assert_eq!(snapshot.superstep, 3);
    assert_eq!(snapshot.state.get("count"), Some(&json!(3)));
    assert!(snapshot.is_resumable());
}

#[tokio::test]
async fn checkpoint_namespace_defaults_to_the_graph_id() {
    let store = Arc::new(InMemoryGraphStore::new());
    let graph = chain()
        .compile_with(CompileOptions {
            store: Some(Arc::clone(&store) as Arc<dyn GraphStore>),
            namespace: Some("custom".to_string()),
            ..CompileOptions::default()
        })
        .unwrap();
    graph
        .run(json!({}), common::session(), ExecutionOptions::default())
        .await
        .unwrap();
    assert!(store.get("s1", "chain").await.unwrap().is_none());
    assert!(store.get("s1", "custom").await.unwrap().is_some());
}

#[tokio::test]
async fn failing_runs_save_the_last_completed_boundary() {
    let store = Arc::new(InMemoryGraphStore::new());
    let graph = GraphBuilder::new("chain")
        .node("a", common::counting("a"))
        .node(
            "b",
            SyncNode::new("b", |_state: &mut GraphState, _ctx: &GraphContext| {
                Err(openjiuwen_common::error::Error::validation("broken"))
            }),
        )
        .edge("a", "b")
        .compile_with(CompileOptions {
            store: Some(Arc::clone(&store) as Arc<dyn GraphStore>),
            ..CompileOptions::default()
        })
        .unwrap();
    let _ = graph
        .run(json!({}), common::session(), ExecutionOptions::default())
        .await
        .unwrap_err();
    let snapshot = store.get("s1", "chain").await.unwrap().unwrap();
    assert_eq!(snapshot.superstep, 1);
    assert_eq!(snapshot.state.get("count"), Some(&json!(1)));
}

#[tokio::test]
async fn a_first_superstep_failure_keeps_the_initial_boundary() {
    // The run's initial boundary is captured before the first superstep, so a
    // failure inside that superstep still leaves a resume point holding the
    // input state and the seeded entry activation.
    let store = Arc::new(InMemoryGraphStore::new());
    let graph = GraphBuilder::new("chain")
        .node(
            "a",
            SyncNode::new("a", |_state: &mut GraphState, _ctx: &GraphContext| {
                Err(openjiuwen_common::error::Error::validation("broken"))
            }),
        )
        .compile_with(CompileOptions {
            store: Some(Arc::clone(&store) as Arc<dyn GraphStore>),
            ..CompileOptions::default()
        })
        .unwrap();
    let _ = graph
        .run(
            json!({"seed": 1}),
            common::session(),
            ExecutionOptions::default(),
        )
        .await
        .unwrap_err();
    let snapshot = store.get("s1", "chain").await.unwrap().unwrap();
    assert_eq!(snapshot.superstep, 0);
    assert_eq!(snapshot.state.get("seed"), Some(&json!(1)));
    assert!(!snapshot.is_resumable());
}

#[tokio::test]
async fn resume_continues_without_repeating_completed_supersteps() {
    let store = Arc::new(InMemoryGraphStore::new());
    let fail_once = Arc::new(AtomicBool::new(true));
    let flag = Arc::clone(&fail_once);
    let graph = GraphBuilder::new("chain")
        .node("a", common::counting("a"))
        .node(
            "b",
            SyncNode::new("b", move |state: &mut GraphState, _ctx: &GraphContext| {
                if flag.swap(false, Ordering::SeqCst) {
                    return Err(openjiuwen_common::error::Error::validation("first attempt"));
                }
                let count = state
                    .get("count")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0);
                state.set("count", json!(count + 1));
                Ok(NodeOutcome::Continue)
            }),
        )
        .node("c", common::counting("c"))
        .edge("a", "b")
        .edge("b", "c")
        .compile_with(CompileOptions {
            store: Some(Arc::clone(&store) as Arc<dyn GraphStore>),
            ..CompileOptions::default()
        })
        .unwrap();

    let _ = graph
        .run(json!({}), common::session(), ExecutionOptions::default())
        .await
        .unwrap_err();

    let result = graph
        .run(
            json!({"count": 999}),
            common::session(),
            ExecutionOptions {
                resume: true,
                ..ExecutionOptions::default()
            },
        )
        .await
        .unwrap();
    assert!(result.resumed);
    // The resumed run skips `a` and ignores the non-null input value.
    assert_eq!(result.visited, vec!["b".to_string(), "c".to_string()]);
    assert_eq!(result.supersteps, 3);
    assert_eq!(result.state.get("count"), Some(&json!(3)));
}

#[tokio::test]
async fn resume_without_a_snapshot_starts_fresh() {
    let store = Arc::new(InMemoryGraphStore::new());
    let graph = chain()
        .compile_with(CompileOptions {
            store: Some(Arc::clone(&store) as Arc<dyn GraphStore>),
            ..CompileOptions::default()
        })
        .unwrap();
    let result = graph
        .run(
            json!({}),
            common::session(),
            ExecutionOptions {
                resume: true,
                ..ExecutionOptions::default()
            },
        )
        .await
        .unwrap();
    assert!(!result.resumed);
    assert_eq!(result.visited.len(), 3);
    assert_eq!(result.supersteps, 3);
}

#[tokio::test]
async fn in_memory_store_deletes_per_namespace_and_per_session() {
    let store = InMemoryGraphStore::new();
    let snapshot = openjiuwen_graph::GraphSnapshot {
        graph_id: "chain".to_string(),
        namespace: "chain".to_string(),
        superstep: 1,
        state: GraphState::new(),
        channels: std::collections::BTreeMap::new(),
        buffer: Vec::new(),
    };
    store.save("s1", "chain", &snapshot).await.unwrap();
    store.save("s2", "chain", &snapshot).await.unwrap();
    assert_eq!(store.len(), 2);

    store.delete("s1", Some("chain")).await.unwrap();
    assert!(store.get("s1", "chain").await.unwrap().is_none());
    assert!(store.get("s2", "chain").await.unwrap().is_some());

    store.delete("s2", None).await.unwrap();
    assert!(store.is_empty());
}

#[tokio::test]
async fn checkpointer_store_round_trips_through_the_session_checkpointer() {
    let checkpointer = Arc::new(MemoryCheckpointer::new());
    let graph = chain()
        .compile_with(CompileOptions {
            checkpointer: Some(Arc::clone(&checkpointer) as Arc<dyn Checkpointer>),
            ..CompileOptions::default()
        })
        .unwrap();
    graph
        .run(json!({}), common::session(), ExecutionOptions::default())
        .await
        .unwrap();

    let checkpoint = checkpointer
        .load("s1", "chain")
        .await
        .unwrap()
        .expect("snapshot checkpoint");
    assert_eq!(checkpoint.namespace, CheckpointNamespace::WorkflowGraph);
    assert_eq!(checkpoint.entity_id, "chain");
    assert_eq!(checkpoint.checkpoint_id, "chain");

    let store = CheckpointerStore::new(Arc::clone(&checkpointer) as Arc<dyn Checkpointer>, "chain");
    let snapshot = store.get("s1", "chain").await.unwrap().unwrap();
    assert_eq!(snapshot.graph_id, "chain");
    assert_eq!(snapshot.superstep, 3);
    assert_eq!(snapshot.state.get("count"), Some(&json!(3)));
}

#[tokio::test]
async fn checkpointer_store_releases_a_session() {
    let checkpointer = Arc::new(MemoryCheckpointer::new());
    let store = CheckpointerStore::new(Arc::clone(&checkpointer) as Arc<dyn Checkpointer>, "chain");
    let graph = chain()
        .compile_with(CompileOptions {
            store: Some(Arc::new(store) as Arc<dyn GraphStore>),
            ..CompileOptions::default()
        })
        .unwrap();
    let session = common::session();
    graph
        .run(json!({}), Arc::clone(&session), ExecutionOptions::default())
        .await
        .unwrap();
    assert!(checkpointer.session_exists("s1").await.unwrap());

    let store = CheckpointerStore::new(Arc::clone(&checkpointer) as Arc<dyn Checkpointer>, "chain");
    GraphStore::delete(&store, "s1", None).await.unwrap();
    assert!(store.get("s1", "chain").await.unwrap().is_none());
}
