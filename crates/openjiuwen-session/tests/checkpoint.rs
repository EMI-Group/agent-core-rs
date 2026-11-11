//! Behavior tests for checkpoint persistence.
//!
//! Ports the storage contract of `openjiuwen/core/session/checkpointer/`
//! (v0.1.20), whose Python tests live in
//! `tests/unit_tests/core/session/checkpointer/` (`test_inmemory_checkpointer.py`,
//! `test_inmemory_storage.py`, `test_persistence_storage.py`). Rust checkpoints
//! are JSON documents, so these tests assert the JSON field names rather than
//! pickle payloads.
//!
//! The shared contract is one generic async suite run against every backend;
//! backend-specific behavior (on-disk layout, factory selection) is tested
//! separately.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;
use std::sync::Arc;

use openjiuwen_common::error::{Error, ErrorKind};
use openjiuwen_session::checkpoint::{
    build_key, build_key_with_namespace, create_checkpointer, Checkpoint, CheckpointMeta,
    CheckpointNamespace, Checkpointer, CheckpointerConfig, CheckpointerKind, FileCheckpointer,
    MemoryCheckpointer, SESSION_NAMESPACE_AGENT, SESSION_NAMESPACE_AGENT_TEAM,
    SESSION_NAMESPACE_WORKFLOW, WORKFLOW_NAMESPACE_GRAPH,
};
use openjiuwen_session::tracer::TracePosition;
use serde_json::{json, Value};

/// Returns the metadata map used by round-trip assertions.
fn metadata() -> BTreeMap<String, Value> {
    BTreeMap::from([
        ("reason".to_string(), json!("interrupt")),
        ("attempt".to_string(), json!(3)),
    ])
}

/// Returns the trace position used by round-trip assertions.
fn trace_position() -> TracePosition {
    TracePosition {
        trace_id: "trace-1".to_string(),
        last_invoke_id: Some("invoke-2".to_string()),
        frame_count: 7,
    }
}

/// Builds a checkpoint with an explicit id and creation time.
fn checkpoint_at(session_id: &str, checkpoint_id: &str, created_at: &str) -> Checkpoint {
    let mut checkpoint = Checkpoint::new(session_id, CheckpointNamespace::Workflow, "flow-1")
        .with_checkpoint_id(checkpoint_id)
        .with_global_state(json!({ "counter": 1 }));
    checkpoint.created_at = created_at.to_string();
    checkpoint
}

/// Asserts that `result` failed with [`ErrorKind::Validation`].
fn assert_validation<T>(result: openjiuwen_common::error::Result<T>) {
    let error = result.err().expect("the call must fail");
    assert_eq!(error.kind(), ErrorKind::Validation, "unexpected: {error}");
    assert!(matches!(error, Error::Validation(_)));
}

/// Collects the checkpoint ids of a metadata list, preserving order.
fn ids(metas: &[CheckpointMeta]) -> Vec<String> {
    metas
        .iter()
        .map(|meta| meta.checkpoint_id.clone())
        .collect()
}

/// The storage contract every backend must satisfy.
async fn exercise_storage_contract(checkpointer: Arc<dyn Checkpointer>) {
    let checkpointer = checkpointer.as_ref();
    exercise_save_and_load(checkpointer).await;
    exercise_ordering(checkpointer).await;
    exercise_delete(checkpointer).await;
    exercise_release(checkpointer).await;
    exercise_invalid_ids(checkpointer).await;
}

/// Saves a generated id, loads it back, and checks missing lookups.
async fn exercise_save_and_load(checkpointer: &dyn Checkpointer) {
    let session_id = "session-1";
    assert!(
        !checkpointer.session_exists(session_id).await.unwrap(),
        "a fresh session has no checkpoints"
    );
    assert_eq!(
        checkpointer.list(session_id).await.unwrap(),
        Vec::<CheckpointMeta>::new()
    );

    // save fills a generated id and returns it.
    let checkpoint = Checkpoint::new(session_id, CheckpointNamespace::Workflow, "flow-1")
        .with_global_state(json!({ "counter": 1 }))
        .with_component_state(json!({ "node_a": { "value": 2 } }))
        .with_updates(json!({ "staged": [1, 2] }))
        .with_trace(Some(trace_position()))
        .with_metadata(metadata());
    assert_eq!(checkpoint.checkpoint_id, "");
    assert_ne!(checkpoint.created_at, "");
    let generated = checkpointer.save(checkpoint).await.unwrap();
    assert_ne!(generated, "", "save generates a checkpoint id");
    assert!(checkpointer.session_exists(session_id).await.unwrap());

    // load round-trips the whole document.
    let loaded = checkpointer
        .load(session_id, &generated)
        .await
        .unwrap()
        .expect("the saved checkpoint is loadable");
    assert_eq!(loaded.session_id, session_id);
    assert_eq!(loaded.checkpoint_id, generated);
    assert_eq!(loaded.namespace, CheckpointNamespace::Workflow);
    assert_eq!(loaded.entity_id, "flow-1");
    assert_eq!(loaded.global_state, json!({ "counter": 1 }));
    assert_eq!(loaded.component_state, json!({ "node_a": { "value": 2 } }));
    assert_eq!(loaded.updates, json!({ "staged": [1, 2] }));
    assert_eq!(loaded.trace, Some(trace_position()));
    assert_eq!(loaded.metadata, metadata());

    // missing checkpoints and sessions load as None.
    assert!(checkpointer
        .load(session_id, "does-not-exist")
        .await
        .unwrap()
        .is_none());
    assert!(checkpointer
        .load("unknown-session", &generated)
        .await
        .unwrap()
        .is_none());
    assert!(checkpointer
        .latest("unknown-session")
        .await
        .unwrap()
        .is_none());
}

/// Orders by `(created_at, checkpoint_id)` ascending and resolves `latest`.
async fn exercise_ordering(checkpointer: &dyn Checkpointer) {
    let session_id = "session-order";
    checkpointer
        .save(checkpoint_at(
            session_id,
            "cp-older",
            "2024-01-01T00:00:00.000Z",
        ))
        .await
        .unwrap();
    checkpointer
        .save(checkpoint_at(
            session_id,
            "cp-newer",
            "2025-01-01T00:00:00.000Z",
        ))
        .await
        .unwrap();
    let metas = checkpointer.list(session_id).await.unwrap();
    assert_eq!(ids(&metas), vec!["cp-older", "cp-newer"]);
    assert_eq!(metas[1].namespace, CheckpointNamespace::Workflow);
    assert_eq!(metas[1].entity_id, "flow-1");
    assert_eq!(metas[1].created_at, "2025-01-01T00:00:00.000Z");
    let latest = checkpointer
        .latest(session_id)
        .await
        .unwrap()
        .expect("the session has checkpoints");
    assert_eq!(latest.checkpoint_id, "cp-newer");

    // overwriting the same id is allowed.
    let replaced = checkpointer
        .save(checkpoint_at(
            session_id,
            "cp-newer",
            "2025-01-01T00:00:00.000Z",
        ))
        .await
        .unwrap();
    assert_eq!(replaced, "cp-newer");
    assert_eq!(checkpointer.list(session_id).await.unwrap().len(), 2);
}

/// Deletes one checkpoint and clears session existence with the last one.
async fn exercise_delete(checkpointer: &dyn Checkpointer) {
    let session_id = "session-delete";
    checkpointer
        .save(checkpoint_at(
            session_id,
            "cp-a",
            "2024-01-01T00:00:00.000Z",
        ))
        .await
        .unwrap();
    checkpointer
        .save(checkpoint_at(
            session_id,
            "cp-b",
            "2024-01-02T00:00:00.000Z",
        ))
        .await
        .unwrap();
    checkpointer.delete(session_id, "cp-a").await.unwrap();
    assert!(checkpointer
        .load(session_id, "cp-a")
        .await
        .unwrap()
        .is_none());
    assert!(checkpointer.session_exists(session_id).await.unwrap());
    checkpointer.delete(session_id, "cp-a").await.unwrap();
    checkpointer.delete(session_id, "missing").await.unwrap();
    checkpointer.delete(session_id, "cp-b").await.unwrap();
    assert!(
        !checkpointer.session_exists(session_id).await.unwrap(),
        "deleting the last checkpoint clears session existence"
    );
}

/// Releases every checkpoint of a session; releasing twice is a no-op.
async fn exercise_release(checkpointer: &dyn Checkpointer) {
    let session_id = "session-release";
    checkpointer
        .save(checkpoint_at(
            session_id,
            "cp-1",
            "2024-01-01T00:00:00.000Z",
        ))
        .await
        .unwrap();
    checkpointer
        .save(checkpoint_at(
            session_id,
            "cp-2",
            "2024-01-02T00:00:00.000Z",
        ))
        .await
        .unwrap();
    checkpointer.release(session_id).await.unwrap();
    assert!(!checkpointer.session_exists(session_id).await.unwrap());
    assert_eq!(
        checkpointer.list(session_id).await.unwrap(),
        Vec::<CheckpointMeta>::new()
    );
    assert!(checkpointer
        .load(session_id, "cp-1")
        .await
        .unwrap()
        .is_none());
    checkpointer.release(session_id).await.unwrap();
}

/// Rejects malformed ids with validation errors, never panics.
async fn exercise_invalid_ids(checkpointer: &dyn Checkpointer) {
    let session_id = "invalid-session";
    assert_validation(
        checkpointer
            .save(Checkpoint::new(
                "../escape",
                CheckpointNamespace::Agent,
                "agent-1",
            ))
            .await,
    );
    assert_validation(
        checkpointer
            .save(Checkpoint::new(
                "a/b",
                CheckpointNamespace::Agent,
                "agent-1",
            ))
            .await,
    );
    assert_validation(
        checkpointer
            .save(Checkpoint::new(
                String::new(),
                CheckpointNamespace::Agent,
                "agent-1",
            ))
            .await,
    );
    assert_validation(
        checkpointer
            .save(
                Checkpoint::new(session_id, CheckpointNamespace::Agent, "agent-1")
                    .with_checkpoint_id("../escape"),
            )
            .await,
    );
    assert_validation(
        checkpointer
            .save(
                Checkpoint::new(session_id, CheckpointNamespace::Agent, "agent-1")
                    .with_checkpoint_id("a/b"),
            )
            .await,
    );
    assert_validation(checkpointer.load(session_id, "a/b").await);
    assert_validation(checkpointer.load(session_id, "").await);
    assert_validation(checkpointer.delete(session_id, "").await);
    assert_validation(checkpointer.list("../escape").await);
    assert_validation(checkpointer.release("../escape").await);
    assert_validation(checkpointer.session_exists("a/b").await);
}

#[tokio::test]
async fn memory_backend_satisfies_the_contract() {
    // ports tests/unit_tests/core/session/checkpointer/test_inmemory_checkpointer.py
    let backend = MemoryCheckpointer::new();
    assert_eq!(backend.len("session-1"), 0);
    assert!(backend.is_empty());
    exercise_storage_contract(Arc::new(backend.clone())).await;
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
    exercise_storage_contract(Arc::new(FileCheckpointer::new(directory.path()))).await;
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
