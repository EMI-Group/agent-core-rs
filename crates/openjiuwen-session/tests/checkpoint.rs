//! Behavior tests for checkpoint persistence.
//!
//! Ports the storage contract of `openjiuwen/core/session/checkpointer/`
//! (v0.1.20), whose Python tests live in
//! `tests/unit_tests/core/session/checkpointer/` (`test_inmemory_checkpointer.py`,
//! `test_inmemory_storage.py`, `test_persistence_storage.py`). Rust checkpoints
//! are JSON documents, so these tests assert the JSON field names rather than
//! pickle payloads.
//!
//! The shared contract is one generic async suite run against every backend
//! (`./support/mod.rs`); backend-specific behavior (on-disk layout, factory
//! selection) is tested here.
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod support;

use std::collections::BTreeMap;
use std::sync::Arc;

use openjiuwen_common::error::ErrorKind;
use openjiuwen_session::checkpoint::{
    build_key, build_key_with_namespace, create_checkpointer, Checkpoint, CheckpointNamespace,
    Checkpointer, CheckpointerConfig, CheckpointerKind, FileCheckpointer, MemoryCheckpointer,
    SESSION_NAMESPACE_AGENT, SESSION_NAMESPACE_AGENT_TEAM, SESSION_NAMESPACE_WORKFLOW,
    WORKFLOW_NAMESPACE_GRAPH,
};
use serde_json::{json, Value};

use support::{run_backend_contract, ListOrder};

#[tokio::test]
async fn memory_backend_satisfies_the_contract() {
    // ports tests/unit_tests/core/session/checkpointer/test_inmemory_checkpointer.py
    let backend = MemoryCheckpointer::new();
    assert_eq!(backend.len("session-1"), 0);
    assert!(backend.is_empty());
    run_backend_contract("memory", Arc::new(backend.clone()), ListOrder::OldestFirst).await;
    assert_eq!(backend.len("session-1"), 1);
    backend.clear();
    assert!(backend.is_empty());
}

#[tokio::test]
async fn shared_memory_checkpointer_is_process_wide() {
    let first = MemoryCheckpointer::shared();
    let second = MemoryCheckpointer::shared();
    let id = first
        .save(Checkpoint::new(
            "shared-session",
            CheckpointNamespace::Agent,
            "agent-1",
        ))
        .await
        .unwrap();
    assert!(second.load("shared-session", &id).await.unwrap().is_some());
    first.release("shared-session").await.unwrap();
}

#[tokio::test]
async fn file_backend_satisfies_the_contract() {
    // ports tests/unit_tests/core/session/checkpointer/test_persistence_storage.py
    let directory = tempfile::tempdir().unwrap();
    run_backend_contract(
        "file",
        Arc::new(FileCheckpointer::new(directory.path())),
        ListOrder::OldestFirst,
    )
    .await;
    assert!(!directory.path().join("session-release").exists());
}

#[tokio::test]
async fn file_backend_writes_json_documents() {
    let directory = tempfile::tempdir().unwrap();
    let checkpointer = FileCheckpointer::new(directory.path());
    let checkpoint = Checkpoint::new("file-session", CheckpointNamespace::Agent, "agent-1")
        .with_global_state(json!({ "a": 1 }));
    let checkpoint_id = checkpointer.save(checkpoint).await.unwrap();
    let session_dir = directory.path().join("file-session");
    let path = session_dir.join(format!("{checkpoint_id}.json"));
    let text = tokio::fs::read_to_string(&path).await.unwrap();
    let value: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(value["session_id"], json!("file-session"));
    assert_eq!(value["checkpoint_id"], json!(checkpoint_id));
    assert_eq!(value["namespace"], json!("agent"));
    assert_eq!(value["global_state"], json!({ "a": 1 }));
    assert!(value.get("metadata").is_some());

    // the atomic write leaves no temporary file behind.
    let mut entries = tokio::fs::read_dir(&session_dir).await.unwrap();
    while let Some(entry) = entries.next_entry().await.unwrap() {
        assert_eq!(
            entry.path().extension().and_then(|ext| ext.to_str()),
            Some("json"),
            "only final JSON documents remain: {:?}",
            entry.path()
        );
    }
}

#[tokio::test]
async fn file_backend_reports_corrupt_json_as_session_error() {
    let directory = tempfile::tempdir().unwrap();
    let session_dir = directory.path().join("corrupt-session");
    tokio::fs::create_dir_all(&session_dir).await.unwrap();
    tokio::fs::write(session_dir.join("cp-1.json"), b"{ not json")
        .await
        .unwrap();
    let checkpointer = FileCheckpointer::new(directory.path());
    let error = checkpointer
        .load("corrupt-session", "cp-1")
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Session);
    let error = checkpointer.list("corrupt-session").await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Session);
}

#[tokio::test]
async fn factory_builds_the_memory_backend() {
    let config = CheckpointerConfig {
        kind: CheckpointerKind::InMemory,
        ..CheckpointerConfig::default()
    };
    let checkpointer = create_checkpointer(&config).await.unwrap();
    let id = checkpointer
        .save(Checkpoint::new(
            "factory-session",
            CheckpointNamespace::Agent,
            "agent-1",
        ))
        .await
        .unwrap();
    assert!(checkpointer
        .session_exists("factory-session")
        .await
        .unwrap());
    assert!(checkpointer
        .load("factory-session", &id)
        .await
        .unwrap()
        .is_some());
}

#[tokio::test]
async fn factory_builds_the_file_backend() {
    let directory = tempfile::tempdir().unwrap();
    let config = CheckpointerConfig {
        kind: CheckpointerKind::File,
        path: Some(directory.path().to_path_buf()),
        extra: BTreeMap::new(),
    };
    let checkpointer = create_checkpointer(&config).await.unwrap();
    checkpointer
        .save(Checkpoint::new(
            "factory-file-session",
            CheckpointNamespace::Workflow,
            "flow-1",
        ))
        .await
        .unwrap();
    assert!(directory
        .path()
        .join("factory-file-session")
        .read_dir()
        .unwrap()
        .next()
        .is_some());
}

#[tokio::test]
async fn factory_requires_a_file_path() {
    let config = CheckpointerConfig {
        kind: CheckpointerKind::File,
        ..CheckpointerConfig::default()
    };
    let error = create_checkpointer(&config).await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
}

#[cfg(not(feature = "sqlite"))]
#[tokio::test]
async fn factory_rejects_sqlite_without_the_feature() {
    let directory = tempfile::tempdir().unwrap();
    let config = CheckpointerConfig {
        kind: CheckpointerKind::Sqlite,
        path: Some(directory.path().join("checkpoints.db")),
        extra: BTreeMap::new(),
    };
    let error = create_checkpointer(&config).await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Unsupported);
}

#[test]
fn namespaces_and_keys_match_python() {
    // ports openjiuwen/core/session/checkpointer/base.py namespace constants.
    assert_eq!(CheckpointNamespace::default(), CheckpointNamespace::Agent);
    assert_eq!(CheckpointNamespace::Agent.as_str(), SESSION_NAMESPACE_AGENT);
    assert_eq!(
        CheckpointNamespace::AgentTeam.as_str(),
        SESSION_NAMESPACE_AGENT_TEAM
    );
    assert_eq!(
        CheckpointNamespace::Workflow.as_str(),
        SESSION_NAMESPACE_WORKFLOW
    );
    assert_eq!(
        CheckpointNamespace::WorkflowGraph.as_str(),
        WORKFLOW_NAMESPACE_GRAPH
    );
    assert_eq!(
        serde_json::to_value(CheckpointNamespace::AgentTeam).unwrap(),
        json!("agent-team")
    );
    assert_eq!(
        serde_json::to_value(CheckpointNamespace::WorkflowGraph).unwrap(),
        json!("workflow-graph")
    );
    assert_eq!(
        serde_json::from_value::<CheckpointNamespace>(json!("workflow")).unwrap(),
        CheckpointNamespace::Workflow
    );

    assert_eq!(build_key(&["s1", "agent", "a1"]), "s1:agent:a1");
    assert_eq!(
        build_key_with_namespace("s1", "agent", "a1", &["state", "blob"]),
        "s1:agent:a1:state:blob"
    );
    assert_eq!(
        build_key_with_namespace("s1", "workflow", "", &["state"]),
        "s1:workflow:state"
    );
    assert_eq!(
        build_key_with_namespace("s1", "agent", "a1", &[]),
        "s1:agent:a1"
    );
}
