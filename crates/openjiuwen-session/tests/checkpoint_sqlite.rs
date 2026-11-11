//! Behavior tests for the `SQLite` checkpoint backend (feature `sqlite`).
//!
//! Ports the storage contract of `openjiuwen/core/session/checkpointer/persistence.py`
//! (v0.1.20), whose Python tests live in
//! `tests/unit_tests/core/session/checkpointer/test_persistence_storage.py`. The
//! shared suite in `./support/mod.rs` runs against both an in-memory and a
//! file-backed database; reopening and factory selection are tested here.
#![cfg(feature = "sqlite")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod support;

use std::collections::BTreeMap;
use std::sync::Arc;

use openjiuwen_common::error::ErrorKind;
use openjiuwen_session::checkpoint::{
    create_checkpointer, Checkpoint, CheckpointNamespace, Checkpointer, CheckpointerConfig,
    CheckpointerKind,
};
use openjiuwen_session::tracer::TracePosition;
use openjiuwen_session::SqliteCheckpointer;
use serde_json::json;

use support::{run_backend_contract, ListOrder};

#[tokio::test]
async fn in_memory_backend_satisfies_the_contract() {
    let checkpointer = SqliteCheckpointer::in_memory().await.unwrap();
    run_backend_contract(
        "sqlite-in-memory",
        Arc::new(checkpointer),
        ListOrder::NewestFirst,
    )
    .await;
}

#[tokio::test]
async fn file_backend_satisfies_the_contract() {
    let directory = tempfile::tempdir().unwrap();
    let checkpointer = SqliteCheckpointer::open(directory.path().join("checkpoints.db"))
        .await
        .unwrap();
    run_backend_contract(
        "sqlite-file",
        Arc::new(checkpointer),
        ListOrder::NewestFirst,
    )
    .await;
}

#[tokio::test]
async fn in_memory_backends_are_isolated() {
    let first = SqliteCheckpointer::in_memory().await.unwrap();
    let second = SqliteCheckpointer::in_memory().await.unwrap();
    let checkpoint_id = first
        .save(Checkpoint::new(
            "isolated-session",
            CheckpointNamespace::Agent,
            "agent-1",
        ))
        .await
        .unwrap();
    assert!(first
        .load("isolated-session", &checkpoint_id)
        .await
        .unwrap()
        .is_some());
    assert!(!second.session_exists("isolated-session").await.unwrap());
}

#[tokio::test]
async fn checkpoints_survive_reopening_the_database() {
    let directory = tempfile::tempdir().unwrap();
    // the nested parent directory is created by `open`.
    let path = directory.path().join("nested").join("checkpoints.db");
    let expected = Checkpoint::new("reopen-session", CheckpointNamespace::Workflow, "flow-1")
        .with_global_state(json!({ "counter": 7 }))
        .with_component_state(json!({ "node_a": { "value": 8 } }))
        .with_updates(json!({ "staged": ["a"] }))
        .with_trace(Some(TracePosition {
            trace_id: "trace-7".to_string(),
            last_invoke_id: Some("invoke-9".to_string()),
            frame_count: 11,
        }))
        .with_metadata(BTreeMap::from([("reason".to_string(), json!("resume"))]));
    let checkpoint_id = {
        let checkpointer = SqliteCheckpointer::open(&path).await.unwrap();
        checkpointer.save(expected.clone()).await.unwrap()
    };

    // dropping the first pool leaves only the file; a second open reads it back.
    let reopened = SqliteCheckpointer::open(&path).await.unwrap();
    let loaded = reopened
        .load("reopen-session", &checkpoint_id)
        .await
        .unwrap()
        .expect("the saved checkpoint is loadable after reopening");
    assert_eq!(loaded.session_id, expected.session_id);
    assert_eq!(loaded.checkpoint_id, checkpoint_id);
    assert_eq!(loaded.namespace, expected.namespace);
    assert_eq!(loaded.entity_id, expected.entity_id);
    assert_eq!(loaded.created_at, expected.created_at);
    assert_eq!(loaded.global_state, expected.global_state);
    assert_eq!(loaded.component_state, expected.component_state);
    assert_eq!(loaded.updates, expected.updates);
    assert_eq!(loaded.trace, expected.trace);
    assert_eq!(loaded.metadata, expected.metadata);
    let latest = reopened
        .latest("reopen-session")
        .await
        .unwrap()
        .expect("the session has a checkpoint");
    assert_eq!(latest.checkpoint_id, checkpoint_id);
}

#[tokio::test]
async fn factory_builds_the_sqlite_backend() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("factory.db");
    let config = CheckpointerConfig {
        kind: CheckpointerKind::Sqlite,
        path: Some(path.clone()),
        extra: BTreeMap::new(),
    };
    let checkpointer = create_checkpointer(&config).await.unwrap();
    let checkpoint_id = checkpointer
        .save(Checkpoint::new(
            "factory-sqlite-session",
            CheckpointNamespace::Agent,
            "agent-1",
        ))
        .await
        .unwrap();
    assert!(
        tokio::fs::metadata(&path).await.is_ok(),
        "the factory creates the database file"
    );

    // a second backend on the same file sees the factory's checkpoint.
    let reopened = SqliteCheckpointer::open(&path).await.unwrap();
    assert!(reopened
        .load("factory-sqlite-session", &checkpoint_id)
        .await
        .unwrap()
        .is_some());
}

#[tokio::test]
async fn factory_requires_a_sqlite_path() {
    let config = CheckpointerConfig {
        kind: CheckpointerKind::Sqlite,
        ..CheckpointerConfig::default()
    };
    let error = create_checkpointer(&config).await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
}
