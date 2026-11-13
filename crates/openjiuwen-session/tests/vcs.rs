//! Behavior tests for the local version history ([`SessionHistory`]) and the
//! JSON diff helpers.
//!
//! Ports the local, JSON-only subset of `openjiuwen/core/session/vcs/`
//! (v0.1.20): sequential commits with parent links, listing/loading, snapshot
//! restore, delete/clear, concurrent commits, and snapshot diffing. The suite is
//! offline and deterministic and uses a temporary root per test.
#![allow(clippy::unwrap_used, clippy::expect_used)]
use openjiuwen_common::error::ErrorKind;
use openjiuwen_session::session::{Session, SessionSnapshot, SessionType};
use openjiuwen_session::session_controller::DiskSessionStore;
use openjiuwen_session::sessions::WorkflowSession;
use openjiuwen_session::vcs::{ChangeKind, SessionHistory, VersionId};
use serde_json::{json, Value};

fn snapshot(session_id: &str, global: Value) -> SessionSnapshot {
    let mut snapshot = SessionSnapshot::new(session_id, SessionType::Workflow);
    snapshot.global = global;
    snapshot
}

#[tokio::test]
async fn commits_get_sequential_ids_and_parent_links() {
    let temp = tempfile::tempdir().unwrap();
    let store = DiskSessionStore::new(temp.path());
    let history: SessionHistory = store.history();

    let first = history
        .commit(
            "agent",
            "s1",
            &snapshot("s1", json!({"count": 1})),
            Some("first"),
        )
        .await
        .unwrap();
    assert_eq!(first.id, 1);
    assert_eq!(first.parent, None);
    assert_eq!(first.label.as_deref(), Some("first"));
    assert_ne!(first.created_at.len(), 0);
    assert_eq!(VersionId(first.id).to_string(), "v000001");

    let second = history
        .commit("agent", "s1", &snapshot("s1", json!({"count": 2})), None)
        .await
        .unwrap();
    assert_eq!(second.id, 2);
    assert_eq!(second.parent, Some(1));

    let third = history
        .commit("agent", "s1", &snapshot("s1", json!({"count": 3})), None)
        .await
        .unwrap();
    assert_eq!(third.id, 3);
    assert_eq!(third.parent, Some(2));

    let listed = history.list("agent", "s1").await.unwrap();
    let ids: Vec<u64> = listed.iter().map(|meta| meta.id).collect();
    assert_eq!(ids, [1, 2, 3]);
    assert_eq!(listed[2].parent, Some(2));

    let latest = history.latest("agent", "s1").await.unwrap().unwrap();
    assert_eq!(latest.id, 3);
    assert_eq!(latest.snapshot.global["count"], json!(3));

    let loaded = history
        .load("agent", "s1", VersionId(2))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(loaded.id, 2);
    assert_eq!(loaded.snapshot.global["count"], json!(2));

    // Version documents use the padded file names under history/versions.
    assert!(history.version_path("agent", "s1", VersionId(1)).is_file());
    assert!(history.index_path("agent", "s1").is_file());
    let file_name = history
        .version_path("agent", "s1", VersionId(1))
        .file_name()
        .unwrap()
        .to_string_lossy()
        .into_owned();
    assert_eq!(file_name, "v000001.json");
}

#[tokio::test]
async fn diff_reports_nested_and_array_changes() {
    let temp = tempfile::tempdir().unwrap();
    let store = DiskSessionStore::new(temp.path());
    let history = store.history();

    let mut first = snapshot(
        "s1",
        json!({
            "user": {"name": "Ada", "age": 36},
            "tags": ["a", "b"],
            "removed": "gone",
        }),
    );
    first.components.insert("comp".to_string(), json!({"x": 1}));
    first.metadata.insert("owner".to_string(), json!("ada"));
    history.commit("agent", "s1", &first, None).await.unwrap();

    let mut second = snapshot(
        "s1",
        json!({
            "user": {"name": "Ada", "age": 37},
            "tags": ["a", "c", "d"],
            "added": true,
        }),
    );
    second
        .components
        .insert("comp".to_string(), json!({"x": 2}));
    second.metadata.insert("owner".to_string(), json!("grace"));
    history.commit("agent", "s1", &second, None).await.unwrap();

    let changes = history
        .diff("agent", "s1", VersionId(1), VersionId(2))
        .await
        .unwrap();
    let find = |path: &str| {
        changes
            .iter()
            .find(|change| change.path == path)
            .unwrap_or_else(|| panic!("no change reported at {path:?}: {changes:?}"))
    };

    assert_eq!(find("global.user.age").kind, ChangeKind::Changed);
    assert_eq!(find("global.user.age").before, Some(json!(36)));
    assert_eq!(find("global.user.age").after, Some(json!(37)));
    assert_eq!(find("global.tags[1]").kind, ChangeKind::Changed);
    assert_eq!(find("global.tags[2]").kind, ChangeKind::Added);
    assert_eq!(find("global.tags[2]").after, Some(json!("d")));
    assert_eq!(find("global.removed").kind, ChangeKind::Removed);
    assert_eq!(find("global.removed").before, Some(json!("gone")));
    assert_eq!(find("global.added").kind, ChangeKind::Added);
    assert_eq!(find("components.comp.x").kind, ChangeKind::Changed);
    assert_eq!(find("metadata.owner").kind, ChangeKind::Changed);

    // Unchanged values are not reported.
    assert!(changes
        .iter()
        .all(|change| change.path != "global.user.name"));

    // The missing version is a storage error.
    let error = history
        .diff("agent", "s1", VersionId(1), VersionId(9))
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Session);
}

#[tokio::test]
async fn restore_applies_an_old_snapshot_without_committing() {
    let temp = tempfile::tempdir().unwrap();
    let store = DiskSessionStore::new(temp.path());
    let history = store.history();

    let session = WorkflowSession::new("s1");
    session.update_state(json!({"count": 1})).await.unwrap();
    let first = history
        .commit_from_session("agent", &session, Some("one"))
        .await
        .unwrap();
    assert_eq!(first.id, 1);
    let stored_first = history
        .load("agent", "s1", VersionId(1))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(stored_first.snapshot.global["count"], json!(1));

    session.update_state(json!({"count": 2})).await.unwrap();
    let second = history
        .commit_from_session("agent", &session, None)
        .await
        .unwrap();
    assert_eq!(second.id, 2);
    assert_eq!(second.parent, Some(1));
    let stored_second = history
        .load("agent", "s1", VersionId(2))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(stored_second.snapshot.global["count"], json!(2));

    assert!(history
        .restore("agent", "s1", VersionId(1), &session)
        .await
        .unwrap());
    assert_eq!(session.get_state("count").await.unwrap(), Some(json!(1)));

    // Restoring is not a commit.
    assert_eq!(history.list("agent", "s1").await.unwrap().len(), 2);

    // A missing version leaves the session untouched.
    assert!(!history
        .restore("agent", "s1", VersionId(9), &session)
        .await
        .unwrap());
    assert_eq!(session.get_state("count").await.unwrap(), Some(json!(1)));
}

#[tokio::test]
async fn delete_and_clear_prune_the_history() {
    let temp = tempfile::tempdir().unwrap();
    let store = DiskSessionStore::new(temp.path());
    let history = store.history();
    for index in 1..=3u64 {
        history
            .commit(
                "agent",
                "s1",
                &snapshot("s1", json!({"count": index})),
                None,
            )
            .await
            .unwrap();
    }

    assert!(history.delete("agent", "s1", VersionId(2)).await.unwrap());
    assert!(!history.delete("agent", "s1", VersionId(2)).await.unwrap());
    let ids: Vec<u64> = history
        .list("agent", "s1")
        .await
        .unwrap()
        .iter()
        .map(|meta| meta.id)
        .collect();
    assert_eq!(ids, [1, 3]);

    history.clear("agent", "s1").await.unwrap();
    assert_eq!(history.list("agent", "s1").await.unwrap().len(), 0);
    assert!(history.latest("agent", "s1").await.unwrap().is_none());
    // Clearing an absent history is a no-op.
    history.clear("agent", "s1").await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_commits_get_unique_sequential_ids() {
    let temp = tempfile::tempdir().unwrap();
    let store = DiskSessionStore::new(temp.path());
    let history = store.history();

    let mut handles = Vec::new();
    for index in 0..8u64 {
        let history = history.clone();
        let snapshot = snapshot("s1", json!({"index": index}));
        handles.push(tokio::spawn(async move {
            history.commit("agent", "s1", &snapshot, None).await
        }));
    }
    for handle in handles {
        handle.await.unwrap().unwrap();
    }

    let listed = history.list("agent", "s1").await.unwrap();
    let ids: Vec<u64> = listed.iter().map(|meta| meta.id).collect();
    assert_eq!(ids, (1..=8).collect::<Vec<u64>>());
    for meta in &listed {
        let expected_parent = if meta.id == 1 {
            None
        } else {
            Some(meta.id - 1)
        };
        assert_eq!(
            meta.parent, expected_parent,
            "parent of version {}",
            meta.id
        );
    }
}

#[test]
fn version_ids_round_trip() {
    assert_eq!(VersionId(7).to_string(), "v000007");
    assert_eq!("v000007".parse::<VersionId>().unwrap(), VersionId(7));
    assert_eq!("7".parse::<VersionId>().unwrap(), VersionId(7));
    assert_eq!("v7".parse::<VersionId>().unwrap(), VersionId(7));
    assert!("nope".parse::<VersionId>().is_err());
    assert!("v0".parse::<VersionId>().is_err());
}

#[tokio::test]
async fn invalid_ids_are_rejected() {
    let temp = tempfile::tempdir().unwrap();
    let store = DiskSessionStore::new(temp.path());
    let history = store.history();

    let error = history
        .commit(
            "agent",
            "../escape",
            &snapshot("../escape", json!({})),
            None,
        )
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);

    let error = history.list("../agent", "s1").await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
}
