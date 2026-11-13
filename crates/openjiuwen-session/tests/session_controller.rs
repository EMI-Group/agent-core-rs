//! Behavior tests for the disk session store ([`DiskSessionStore`]) and the
//! per-agent cache ([`SessionController`]).
//!
//! Ports the storage behavior of `openjiuwen/core/session/session_controller/`
//! (v0.1.20): the on-disk layout
//! `<base>/<agent>/sessions/{sessions.json,<session_id>/state.data}`, scope
//! isolation, the active-session marker, and the controller cache. The suite is
//! offline and deterministic and uses a temporary root per test.
#![allow(clippy::unwrap_used, clippy::expect_used)]
use std::path::Path;

use openjiuwen_common::error::ErrorKind;
use openjiuwen_session::session::{Session, SessionSnapshot, SessionType};
use openjiuwen_session::session_controller::{DiskSessionStore, SessionController, SessionScope};
use openjiuwen_session::sessions::{AgentSession, WorkflowSession};
use openjiuwen_session::tracer::InvokeType;
use serde_json::{json, Value};

/// Recursively collects every file name under `dir`.
fn file_names(dir: &Path) -> Vec<String> {
    let mut names = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(current) = stack.pop() {
        let entries = std::fs::read_dir(&current).expect("the directory is readable");
        for entry in entries {
            let entry = entry.expect("the entry is readable");
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if let Some(name) = path.file_name().and_then(|name| name.to_str()) {
                names.push(name.to_string());
            }
        }
    }
    names
}

async fn save_json(
    store: &DiskSessionStore,
    agent_id: &str,
    scope: &SessionScope,
    session_id: &str,
    global: Value,
) {
    let mut snapshot = SessionSnapshot::new(session_id, SessionType::Workflow);
    snapshot.global = global;
    store
        .save_snapshot(agent_id, scope, session_id, snapshot, None)
        .await
        .expect("the snapshot is saved");
}

#[tokio::test]
async fn workflow_session_round_trips_through_disk() {
    let temp = tempfile::tempdir().unwrap();
    let store = DiskSessionStore::new(temp.path());
    let scope = SessionScope::direct("user-1");

    let session = WorkflowSession::new("wf-1");
    session
        .update_state(json!({"user": {"name": "Ada"}, "count": 1}))
        .await
        .unwrap();
    session
        .update_component_state("comp.1", json!({"status": "ok"}))
        .await
        .unwrap();
    {
        let tracer = session.tracer().expect("a workflow session owns a tracer");
        let span = tracer.start_span("run", InvokeType::Workflow);
        span.finish(json!({"done": true}));
    }

    let meta = store
        .save_from_session("agent-a", &scope, &session, Some("first"))
        .await
        .unwrap();
    assert_eq!(meta.session_id, "wf-1");
    assert_eq!(meta.version, 1);
    assert!(meta.is_active);
    assert_eq!(meta.title.as_deref(), Some("first"));
    assert!(store.exists("agent-a", "wf-1").await.unwrap());

    // The state document is exactly `{"meta": …, "data": <snapshot>}`.
    let raw: Value =
        serde_json::from_slice(&std::fs::read(store.state_path("agent-a", "wf-1")).unwrap())
            .unwrap();
    assert!(raw.get("meta").is_some());
    assert_eq!(raw["data"]["session_id"], json!("wf-1"));

    let record = store.load("agent-a", "wf-1").await.unwrap().unwrap();
    assert_eq!(record.snapshot.global["user"]["name"], json!("Ada"));
    let expected_trace = record.snapshot.trace.clone();
    assert!(expected_trace.is_some());

    let restored = WorkflowSession::new("wf-1");
    assert!(store
        .load_into_session("agent-a", "wf-1", &restored)
        .await
        .unwrap());
    assert_eq!(
        restored.get_state("user.name").await.unwrap(),
        Some(json!("Ada"))
    );
    assert_eq!(restored.get_state("count").await.unwrap(), Some(json!(1)));
    assert_eq!(
        restored.get_component_state("comp.1").await.unwrap(),
        Some(json!({"status": "ok"}))
    );
    assert_eq!(restored.trace().position(), expected_trace);

    // A fresh session of a different kind starts empty.
    assert!(!store
        .load_into_session("agent-a", "missing", &restored)
        .await
        .unwrap());
}

#[tokio::test]
async fn agent_session_round_trips_through_disk() {
    let temp = tempfile::tempdir().unwrap();
    let store = DiskSessionStore::new(temp.path());

    let session = AgentSession::new("agent-session-1");
    session.update_state(json!({"turn": 3})).await.unwrap();
    session
        .update_component_state("tool", json!({"calls": 2}))
        .await
        .unwrap();
    let meta = store
        .save_from_session("agent-a", &SessionScope::main(), &session, None)
        .await
        .unwrap();
    assert_eq!(meta.session_id, "agent-session-1");

    let restored = AgentSession::new("agent-session-1");
    assert!(store
        .load_into_session("agent-a", "agent-session-1", &restored)
        .await
        .unwrap());
    assert_eq!(restored.get_state("turn").await.unwrap(), Some(json!(3)));
    assert_eq!(
        restored.get_component_state("tool").await.unwrap(),
        Some(json!({"calls": 2}))
    );
    assert_eq!(restored.session_type(), SessionType::Agent);
}

#[tokio::test]
async fn index_lists_scopes_newest_first_and_tracks_the_active_session() {
    let temp = tempfile::tempdir().unwrap();
    let store = DiskSessionStore::new(temp.path());
    let main = SessionScope::main();
    let direct = SessionScope::direct("u1");

    save_json(&store, "agent", &main, "s1", json!({"step": 1})).await;
    save_json(&store, "agent", &direct, "s2", json!({"step": 2})).await;

    let all = store.list("agent", None).await.unwrap();
    let ids: Vec<&str> = all.iter().map(|meta| meta.session_id.as_str()).collect();
    assert_eq!(ids, ["s2", "s1"]);

    let main_sessions = store.list("agent", Some(&main)).await.unwrap();
    assert_eq!(main_sessions.len(), 1);
    assert_eq!(main_sessions[0].session_id, "s1");

    // Saving an existing session bumps its version and makes it active again.
    let updated = store
        .save_snapshot(
            "agent",
            &main,
            "s1",
            SessionSnapshot::new("s1", SessionType::Workflow),
            None,
        )
        .await
        .unwrap();
    assert_eq!(updated.version, 2);
    let main_sessions = store.list("agent", Some(&main)).await.unwrap();
    assert!(main_sessions[0].is_active);

    // A newer session in the same scope deactivates the previous one.
    save_json(&store, "agent", &main, "s3", json!({"step": 3})).await;
    let main_sessions = store.list("agent", Some(&main)).await.unwrap();
    let ids: Vec<&str> = main_sessions
        .iter()
        .map(|meta| meta.session_id.as_str())
        .collect();
    assert_eq!(ids, ["s3", "s1"]);
    assert!(main_sessions[0].is_active);
    assert!(!main_sessions[1].is_active);

    let index = store.read_index("agent").await.unwrap();
    assert_eq!(index.agent_id, "agent");
    assert!(index.scopes.contains_key("agent:agent:main"));
    assert!(index.scopes.contains_key("agent:agent:main:direct:u1"));
    assert_eq!(
        index.scopes["agent:agent:main"].active_session.as_deref(),
        Some("s3")
    );
    assert!(store.sessions_index_path("agent").is_file());
}

#[tokio::test]
async fn list_agents_scans_the_base_directory() {
    let temp = tempfile::tempdir().unwrap();
    let store = DiskSessionStore::new(temp.path());
    save_json(&store, "alpha", &SessionScope::main(), "s1", json!({})).await;
    save_json(&store, "beta", &SessionScope::main(), "s2", json!({})).await;
    std::fs::create_dir_all(temp.path().join("unrelated")).unwrap();

    assert_eq!(
        store.list_agents().await.unwrap(),
        vec!["alpha".to_string(), "beta".to_string()]
    );
    let missing = DiskSessionStore::new(temp.path().join("missing"));
    assert_eq!(missing.list_agents().await.unwrap().len(), 0);
}

#[tokio::test]
async fn delete_removes_the_directory_and_index_entry() {
    let temp = tempfile::tempdir().unwrap();
    let store = DiskSessionStore::new(temp.path());
    let scope = SessionScope::group("g1");
    save_json(&store, "agent", &scope, "s1", json!({"a": 1})).await;

    let dir = store.session_dir("agent", "s1");
    assert!(dir.is_dir());
    assert!(store.delete("agent", "s1").await.unwrap());
    assert!(!dir.exists());
    assert!(!store.delete("agent", "s1").await.unwrap());

    let index = store.read_index("agent").await.unwrap();
    assert_eq!(index.scopes.len(), 0);
}

#[tokio::test]
async fn invalid_ids_are_rejected() {
    let temp = tempfile::tempdir().unwrap();
    let store = DiskSessionStore::new(temp.path());
    let scope = SessionScope::main();

    let error = store
        .save_snapshot(
            "agent",
            &scope,
            "../escape",
            SessionSnapshot::new("../escape", SessionType::Workflow),
            None,
        )
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);

    let error = store.load("agent", "../escape").await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);

    let error = store.load("", "s1").await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);

    let error = store.list("../agent", None).await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
}

#[tokio::test]
async fn atomic_writes_leave_no_temporary_files() {
    let temp = tempfile::tempdir().unwrap();
    let store = DiskSessionStore::new(temp.path());
    save_json(
        &store,
        "agent",
        &SessionScope::main(),
        "s1",
        json!({"a": 1}),
    )
    .await;
    store
        .history()
        .commit(
            "agent",
            "s1",
            &SessionSnapshot::new("s1", SessionType::Workflow),
            None,
        )
        .await
        .unwrap();

    let names = file_names(temp.path());
    assert_ne!(names.len(), 0);
    assert!(
        names.iter().all(|name| !name.contains(".tmp")),
        "temporary files remain: {names:?}"
    );
}

#[tokio::test]
async fn corrupt_documents_are_session_errors() {
    let temp = tempfile::tempdir().unwrap();
    let store = DiskSessionStore::new(temp.path());
    save_json(
        &store,
        "agent",
        &SessionScope::main(),
        "s1",
        json!({"a": 1}),
    )
    .await;

    let state_path = store.state_path("agent", "s1");
    tokio::fs::write(&state_path, b"{not json").await.unwrap();
    let error = store.load("agent", "s1").await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Session);
    assert!(error.to_string().contains("state.data"));

    let index_path = store.sessions_index_path("agent");
    tokio::fs::write(&index_path, b"oops").await.unwrap();
    let error = store.list("agent", None).await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Session);
    assert!(error.to_string().contains("sessions.json"));
}

#[tokio::test]
async fn controller_caches_reads_and_flush_persists_them() {
    let temp = tempfile::tempdir().unwrap();
    let store = DiskSessionStore::new(temp.path());
    let controller = SessionController::new(store.clone(), "agent");
    let scope = SessionScope::custom("tenant");

    let meta = controller.create(&scope, "s1").await.unwrap();
    assert_eq!(meta.version, 1);
    assert_eq!(meta.session_id, "s1");
    let cached = controller.get("s1").await.unwrap().unwrap();
    assert_eq!(cached.snapshot.session_id, "s1");
    assert_eq!(cached.snapshot.kind, SessionType::Agent);

    // A concurrent store write must not be visible through the cache.
    let mut tampered = SessionSnapshot::new("s1", SessionType::Workflow);
    tampered.global = json!({"tampered": true});
    store
        .save_snapshot("agent", &scope, "s1", tampered, None)
        .await
        .unwrap();
    let cached = controller.get("s1").await.unwrap().unwrap();
    assert!(cached.snapshot.global.get("tampered").is_none());

    // The cache-only listing still reports the session.
    let listed = controller.list(None).await.unwrap();
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].session_id, "s1");

    controller.flush().await.unwrap();
    let on_disk = store.load("agent", "s1").await.unwrap().unwrap();
    assert!(on_disk.snapshot.global.get("tampered").is_none());
    assert_eq!(on_disk.meta.version, 3);

    assert!(controller.delete("s1").await.unwrap());
    assert!(controller.get("s1").await.unwrap().is_none());
    assert_eq!(controller.list(Some(&scope)).await.unwrap().len(), 0);
}

#[tokio::test]
async fn controller_saves_live_sessions_into_the_scope() {
    let temp = tempfile::tempdir().unwrap();
    let store = DiskSessionStore::new(temp.path());
    let controller = SessionController::new(store.clone(), "agent");
    let scope = SessionScope::group_user("g1", "u1");

    let session = WorkflowSession::new("wf-7");
    session.update_state(json!({"step": 7})).await.unwrap();
    let meta = controller
        .save(&session, &scope, Some("seven"))
        .await
        .unwrap();
    assert_eq!(meta.session_id, "wf-7");
    assert_eq!(meta.title.as_deref(), Some("seven"));

    let listed = controller.list(Some(&scope)).await.unwrap();
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].session_id, "wf-7");
    let index = controller.store().read_index("agent").await.unwrap();
    assert!(index
        .scopes
        .contains_key("agent:agent:main:group:g1:user:u1"));

    // `upsert` keeps the record in the scope it already belongs to.
    let mut record = controller.get("wf-7").await.unwrap().unwrap();
    record.snapshot.global = json!({"step": 8});
    let meta = controller.upsert(record).await.unwrap();
    assert_eq!(meta.version, 2);
    let reloaded = store.load("agent", "wf-7").await.unwrap().unwrap();
    assert_eq!(reloaded.snapshot.global, json!({"step": 8}));
}
