//! The shared behavior contract every [`Checkpointer`] backend must satisfy.
//!
//! Ports the storage contract of `openjiuwen/core/session/checkpointer/`
//! (v0.1.20), whose Python tests live in
//! `tests/unit_tests/core/session/checkpointer/` (`test_inmemory_checkpointer.py`,
//! `test_inmemory_storage.py`, `test_persistence_storage.py`). Rust checkpoints
//! are JSON documents, so this suite asserts the JSON field names rather than
//! pickle payloads.
//!
//! The suite is included with `mod support;` from each backend's integration
//! test; backend-specific behavior (on-disk layout, database files, factory
//! selection) is tested in those files.
#![allow(dead_code)]

use std::collections::BTreeMap;
use std::sync::Arc;

use openjiuwen_common::error::{Error, ErrorKind};
use openjiuwen_session::checkpoint::{
    Checkpoint, CheckpointMeta, CheckpointNamespace, Checkpointer,
};
use openjiuwen_session::tracer::TracePosition;
use serde_json::{json, Value};

/// The order a backend's [`Checkpointer::list`] returns checkpoints in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ListOrder {
    /// `(created_at, checkpoint_id)` ascending (`MemoryCheckpointer`, `FileCheckpointer`).
    OldestFirst,
    /// `(created_at, checkpoint_id)` descending (`SqliteCheckpointer`).
    NewestFirst,
}

/// Runs the whole storage contract against `checkpointer`.
///
/// `backend` labels assertion messages, and `order` is the ordering the
/// backend's [`Checkpointer::list`] promises.
pub(crate) async fn run_backend_contract(
    backend: &str,
    checkpointer: Arc<dyn Checkpointer>,
    order: ListOrder,
) {
    let checkpointer = checkpointer.as_ref();
    exercise_save_and_load(backend, checkpointer).await;
    exercise_ordering(backend, checkpointer, order).await;
    exercise_delete(backend, checkpointer).await;
    exercise_release(backend, checkpointer).await;
    exercise_invalid_ids(backend, checkpointer).await;
}

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
fn assert_validation<T>(backend: &str, result: openjiuwen_common::error::Result<T>) {
    let error = result.err().expect("the call must fail");
    assert_eq!(
        error.kind(),
        ErrorKind::Validation,
        "{backend}: unexpected error: {error}"
    );
    assert!(matches!(error, Error::Validation(_)));
}

/// Collects the checkpoint ids of a metadata list, preserving order.
fn ids(metas: &[CheckpointMeta]) -> Vec<String> {
    metas
        .iter()
        .map(|meta| meta.checkpoint_id.clone())
        .collect()
}

/// Saves a generated id, loads it back, and checks missing lookups.
async fn exercise_save_and_load(backend: &str, checkpointer: &dyn Checkpointer) {
    let session_id = "session-1";
    assert!(
        !checkpointer.session_exists(session_id).await.unwrap(),
        "{backend}: a fresh session has no checkpoints"
    );
    assert_eq!(
        checkpointer.list(session_id).await.unwrap(),
        Vec::<CheckpointMeta>::new(),
        "{backend}: a fresh session lists no checkpoints"
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
    assert_ne!(generated, "", "{backend}: save generates a checkpoint id");
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

/// Orders by `(created_at, checkpoint_id)` as `order` promises and resolves
/// `latest`.
async fn exercise_ordering(backend: &str, checkpointer: &dyn Checkpointer, order: ListOrder) {
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
    let expected = match order {
        ListOrder::OldestFirst => vec!["cp-older", "cp-newer"],
        ListOrder::NewestFirst => vec!["cp-newer", "cp-older"],
    };
    assert_eq!(ids(&metas), expected, "{backend}: list order");
    let newer = metas
        .iter()
        .find(|meta| meta.checkpoint_id == "cp-newer")
        .expect("the newer checkpoint is listed");
    assert_eq!(newer.namespace, CheckpointNamespace::Workflow);
    assert_eq!(newer.entity_id, "flow-1");
    assert_eq!(newer.created_at, "2025-01-01T00:00:00.000Z");
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
async fn exercise_delete(backend: &str, checkpointer: &dyn Checkpointer) {
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
        "{backend}: deleting the last checkpoint clears session existence"
    );
}

/// Releases every checkpoint of a session; releasing twice is a no-op.
async fn exercise_release(backend: &str, checkpointer: &dyn Checkpointer) {
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
        Vec::<CheckpointMeta>::new(),
        "{backend}: release empties the session"
    );
    assert!(checkpointer
        .load(session_id, "cp-1")
        .await
        .unwrap()
        .is_none());
    checkpointer.release(session_id).await.unwrap();
}

/// Rejects malformed ids with validation errors, never panics.
async fn exercise_invalid_ids(backend: &str, checkpointer: &dyn Checkpointer) {
    let session_id = "invalid-session";
    assert_validation(
        backend,
        checkpointer
            .save(Checkpoint::new(
                "../escape",
                CheckpointNamespace::Agent,
                "agent-1",
            ))
            .await,
    );
    assert_validation(
        backend,
        checkpointer
            .save(Checkpoint::new(
                "a/b",
                CheckpointNamespace::Agent,
                "agent-1",
            ))
            .await,
    );
    assert_validation(
        backend,
        checkpointer
            .save(Checkpoint::new(
                String::new(),
                CheckpointNamespace::Agent,
                "agent-1",
            ))
            .await,
    );
    assert_validation(
        backend,
        checkpointer
            .save(
                Checkpoint::new(session_id, CheckpointNamespace::Agent, "agent-1")
                    .with_checkpoint_id("../escape"),
            )
            .await,
    );
    assert_validation(
        backend,
        checkpointer
            .save(
                Checkpoint::new(session_id, CheckpointNamespace::Agent, "agent-1")
                    .with_checkpoint_id("a/b"),
            )
            .await,
    );
    assert_validation(backend, checkpointer.load(session_id, "a/b").await);
    assert_validation(backend, checkpointer.load(session_id, "").await);
    assert_validation(backend, checkpointer.delete(session_id, "").await);
    assert_validation(backend, checkpointer.list("../escape").await);
    assert_validation(backend, checkpointer.release("../escape").await);
    assert_validation(backend, checkpointer.session_exists("a/b").await);
}
