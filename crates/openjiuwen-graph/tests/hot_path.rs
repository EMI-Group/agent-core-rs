//! The per-superstep hot path: what a run copies, persists, and returns.
//!
//! The executor shares the run state with routers through an `Arc`, merges node
//! results copy-on-write, and builds a boundary snapshot only when a store is
//! attached. These tests pin the observable side of that: the exact output of a
//! parallel run, and exactly one persisted snapshot per durable boundary.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]

mod common;

use std::sync::{Arc, Mutex, MutexGuard};

use serde_json::{json, Value};

use openjiuwen_common::error::Result;
use openjiuwen_graph::{
    CompileOptions, ExecutionOptions, FnRouter, GraphBuilder, GraphContext, GraphSnapshot,
    GraphState, GraphStore, RouterContext, SyncNode,
};

/// A store that records every saved boundary instead of replacing it.
#[derive(Debug, Default)]
struct CountingStore {
    saves: Mutex<Vec<GraphSnapshot>>,
}

impl CountingStore {
    fn saved(&self) -> MutexGuard<'_, Vec<GraphSnapshot>> {
        self.saves
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn as_store(self: &Arc<Self>) -> Arc<dyn GraphStore> {
        Arc::clone(self) as Arc<dyn GraphStore>
    }
}

#[async_trait::async_trait]
impl GraphStore for CountingStore {
    async fn get(&self, _session_id: &str, _namespace: &str) -> Result<Option<GraphSnapshot>> {
        Ok(None)
    }

    async fn save(
        &self,
        _session_id: &str,
        _namespace: &str,
        snapshot: &GraphSnapshot,
    ) -> Result<()> {
        self.saved().push(snapshot.clone());
        Ok(())
    }

    async fn delete(&self, _session_id: &str, _namespace: Option<&str>) -> Result<()> {
        self.saved().clear();
        Ok(())
    }
}

/// Three nodes, two supersteps: `a`, then `b` and `c` in parallel.
fn parallel() -> GraphBuilder {
    GraphBuilder::new("g")
        .node("a", common::counting("a"))
        .node("b", common::counting("b"))
        .node("c", common::counting("c"))
        .edge("a", "b")
        .edge("a", "c")
}

#[tokio::test]
async fn a_three_node_two_superstep_run_keeps_its_output() {
    let graph = parallel().compile().unwrap();
    let result = graph
        .run(json!({}), common::session(), ExecutionOptions::default())
        .await
        .unwrap();
    assert_eq!(result.graph_id, "g");
    assert_eq!(result.supersteps, 2);
    assert_eq!(
        result.visited,
        vec!["a".to_string(), "b".to_string(), "c".to_string()]
    );
    assert!(result.halted.is_none());
    assert!(!result.resumed);
    // `b` and `c` both start from the pre-superstep state and write `count: 2`.
    assert_eq!(result.state.as_value(), json!({"count": 2}));
}

#[tokio::test]
async fn a_router_reads_the_state_the_superstep_merged() {
    let seen: Arc<Mutex<Vec<Value>>> = Arc::default();
    let recorder = Arc::clone(&seen);
    let router = FnRouter::new("fan", move |ctx: &RouterContext| {
        let observed = ctx.state().as_value();
        let recorder = Arc::clone(&recorder);
        Box::pin(async move {
            recorder.lock().unwrap().push(observed);
            Ok(vec!["b".to_string(), "c".to_string()])
        })
    });
    let graph = GraphBuilder::new("g")
        .node("a", common::counting("a"))
        .node("b", common::counting("b"))
        .node("c", common::counting("c"))
        .entry("a")
        .conditional_edges("a", router)
        .compile()
        .unwrap();
    let result = graph
        .run(json!({}), common::session(), ExecutionOptions::default())
        .await
        .unwrap();
    assert_eq!(result.supersteps, 2);
    assert_eq!(
        result.visited,
        vec!["a".to_string(), "b".to_string(), "c".to_string()]
    );
    assert_eq!(result.state.as_value(), json!({"count": 2}));
    // The router sees `a`'s merged write, before `b` and `c` run.
    assert_eq!(*seen.lock().unwrap(), vec![json!({"count": 1})]);
}

#[tokio::test]
async fn a_stored_run_saves_exactly_one_boundary_per_superstep() {
    let store = Arc::new(CountingStore::default());
    let graph = parallel()
        .compile_with(CompileOptions {
            store: Some(store.as_store()),
            ..CompileOptions::default()
        })
        .unwrap();
    let result = graph
        .run(json!({}), common::session(), ExecutionOptions::default())
        .await
        .unwrap();

    let saved = store.saved();
    assert_eq!(
        saved
            .iter()
            .map(|snapshot| snapshot.superstep)
            .collect::<Vec<_>>(),
        vec![1, 2],
        "one snapshot per durable boundary"
    );
    assert_eq!(saved.len(), result.supersteps as usize);
    assert_eq!(saved[1].namespace, "g");
    assert_eq!(saved[1].state.as_value(), json!({"count": 2}));
    assert!(saved[1].is_resumable());
}

#[tokio::test]
async fn a_failing_run_persists_the_last_completed_boundary() {
    let store = Arc::new(CountingStore::default());
    let graph = GraphBuilder::new("g")
        .node("a", common::counting("a"))
        .node(
            "b",
            SyncNode::new("b", |_state: &mut GraphState, _ctx: &GraphContext| {
                Err(openjiuwen_common::error::Error::validation("broken"))
            }),
        )
        .edge("a", "b")
        .compile_with(CompileOptions {
            store: Some(store.as_store()),
            ..CompileOptions::default()
        })
        .unwrap();
    let _ = graph
        .run(json!({}), common::session(), ExecutionOptions::default())
        .await
        .unwrap_err();

    let saved = store.saved();
    // Superstep 1 saves its boundary; the failure then re-saves that same
    // boundary as the run's resume point.
    assert_eq!(
        saved
            .iter()
            .map(|snapshot| snapshot.superstep)
            .collect::<Vec<_>>(),
        vec![1, 1]
    );
    assert_eq!(saved[1].state.as_value(), json!({"count": 1}));
}

#[tokio::test]
async fn a_stored_run_returns_the_same_state_as_a_store_less_one() {
    let plain = parallel().compile().unwrap();
    let stored = parallel()
        .compile_with(CompileOptions {
            store: Some(Arc::new(CountingStore::default()).as_store()),
            ..CompileOptions::default()
        })
        .unwrap();
    let input = json!({"count": 41});
    let plain = plain
        .run(
            input.clone(),
            common::session(),
            ExecutionOptions::default(),
        )
        .await
        .unwrap();
    let stored = stored
        .run(input, common::session(), ExecutionOptions::default())
        .await
        .unwrap();
    assert_eq!(stored.state.as_value(), plain.state.as_value());
    assert_eq!(stored.state.as_value(), json!({"count": 43}));
    assert_eq!(stored.visited, plain.visited);
    assert_eq!(stored.supersteps, plain.supersteps);
}
