//! Behavior tests for the local version history ([`SessionHistory`]), its refs,
//! rewinds, forks, and the JSON diff helpers.
//!
//! Ports the local, JSON-only subset of `openjiuwen/core/session/vcs/`
//! (v0.1.20): sequential commits with parent links, listing/loading, snapshot
//! restore, delete/clear, concurrent commits, and snapshot diffing, plus the
//! local refs document, `rewind` (soft and hard) and `fork`. The suite is
//! offline and deterministic and uses a temporary root per test.
#![allow(clippy::unwrap_used, clippy::expect_used)]
use openjiuwen_common::error::ErrorKind;
use openjiuwen_session::session::{Session, SessionSnapshot, SessionType};
use openjiuwen_session::session_controller::{DiskSessionStore, SessionScope};
use openjiuwen_session::sessions::WorkflowSession;
use openjiuwen_session::vcs::{ChangeKind, RefName, SessionHistory, VersionId, HEAD_REF, MAIN_REF};
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

// ports openjiuwen/core/session/vcs/manager.py `VersioningManager.append` /
// `put_head`: the local port keeps that head pointer in a refs document beside
// the version files.
#[tokio::test]
async fn commits_create_and_advance_the_head_and_main_refs() {
    let temp = tempfile::tempdir().unwrap();
    let store = DiskSessionStore::new(temp.path());
    let history = store.history();

    // Nothing to point at before the first commit.
    assert!(history.head("agent", "s1").await.unwrap().is_none());
    assert!(history.refs("agent", "s1").await.unwrap().is_empty());
    assert!(!history.refs_path("agent", "s1").is_file());

    history
        .commit(
            "agent",
            "s1",
            &snapshot("s1", json!({"count": 1})),
            Some("first"),
        )
        .await
        .unwrap();
    let refs = history.refs("agent", "s1").await.unwrap();
    assert_eq!(refs.len(), 2);
    assert_eq!(refs.head().unwrap().target, VersionId(1));
    assert_eq!(refs.main().unwrap().target, VersionId(1));
    assert_eq!(
        refs.get(&RefName::new(HEAD_REF).unwrap()).unwrap().target,
        VersionId(1)
    );
    assert_eq!(
        refs.get(&RefName::new(MAIN_REF).unwrap()).unwrap().target,
        VersionId(1)
    );
    assert!(history.refs_path("agent", "s1").is_file());
    let first_head = refs.head().unwrap().clone();

    history
        .commit("agent", "s1", &snapshot("s1", json!({"count": 2})), None)
        .await
        .unwrap();
    let refs = history.refs("agent", "s1").await.unwrap();
    assert_eq!(refs.head().unwrap().target, VersionId(2));
    assert_eq!(refs.main().unwrap().target, VersionId(2));
    // Moving a ref keeps created_at and moves updated_at forward.
    assert_eq!(refs.head().unwrap().created_at, first_head.created_at);
    assert!(refs.head().unwrap().updated_at > first_head.updated_at);
    assert_eq!(history.head("agent", "s1").await.unwrap().unwrap().id, 2);
    assert_eq!(
        history
            .resolve_ref("agent", "s1", MAIN_REF)
            .await
            .unwrap()
            .unwrap()
            .id,
        2
    );
}

#[tokio::test]
async fn refs_round_trip_through_a_fresh_history() {
    let temp = tempfile::tempdir().unwrap();
    let store = DiskSessionStore::new(temp.path());
    let history = store.history();
    history
        .commit("agent", "s1", &snapshot("s1", json!({"count": 1})), None)
        .await
        .unwrap();
    history
        .commit("agent", "s1", &snapshot("s1", json!({"count": 2})), None)
        .await
        .unwrap();

    let entry = history
        .set_ref("agent", "s1", "before-refactor", VersionId(1))
        .await
        .unwrap();
    assert_eq!(entry.name.as_str(), "before-refactor");
    assert_eq!(entry.target, VersionId(1));
    assert_eq!(entry.created_at, entry.updated_at);

    // A history over the same root sees the same refs.
    let fresh = DiskSessionStore::new(temp.path()).history();
    let listed = fresh.list_refs("agent", "s1").await.unwrap();
    let names: Vec<&str> = listed.iter().map(|entry| entry.name.as_str()).collect();
    assert_eq!(names, ["HEAD", "before-refactor", "main"]);
    let resolved = fresh
        .resolve_ref("agent", "s1", "before-refactor")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(resolved.id, 1);
    assert_eq!(resolved.parent, None);
    assert!(fresh
        .resolve_ref("agent", "s1", "missing")
        .await
        .unwrap()
        .is_none());

    // Moving a ref keeps created_at and bumps updated_at (see refs::Refs::set).
    let moved = fresh
        .set_ref("agent", "s1", "before-refactor", VersionId(2))
        .await
        .unwrap();
    assert_eq!(moved.created_at, entry.created_at);
    assert!(moved.updated_at > entry.updated_at);
    assert_eq!(moved.target, VersionId(2));

    assert!(fresh
        .delete_ref("agent", "s1", "before-refactor")
        .await
        .unwrap());
    assert!(!fresh
        .delete_ref("agent", "s1", "before-refactor")
        .await
        .unwrap());
    assert!(fresh
        .resolve_ref("agent", "s1", "before-refactor")
        .await
        .unwrap()
        .is_none());

    // A ref may not point at a version that does not exist.
    let error = fresh
        .set_ref("agent", "s1", "ghost", VersionId(9))
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(format!("{error}").contains("v000009"));

    // Ref names are validated, so a bad name never reaches the store.
    assert!(RefName::new("HEAD").is_ok());
    assert_eq!(RefName::head().as_str(), HEAD_REF);
    assert_eq!(RefName::main().as_str(), MAIN_REF);
    assert!("release-1".parse::<RefName>().is_ok());
    for name in ["", "a b", "..", "a/b"] {
        let error = fresh
            .set_ref("agent", "s1", name, VersionId(1))
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation, "name {name:?}");
        assert!(name.parse::<RefName>().is_err(), "name {name:?}");
    }

    // Only the two automatic refs remain.
    assert_eq!(fresh.refs("agent", "s1").await.unwrap().len(), 2);
}

#[tokio::test]
async fn head_falls_back_to_latest_without_a_refs_document() {
    let temp = tempfile::tempdir().unwrap();
    let store = DiskSessionStore::new(temp.path());
    let history = store.history();
    history
        .commit("agent", "s1", &snapshot("s1", json!({"count": 1})), None)
        .await
        .unwrap();
    history
        .commit("agent", "s1", &snapshot("s1", json!({"count": 2})), None)
        .await
        .unwrap();

    // A history written before refs existed still has a head.
    tokio::fs::remove_file(history.refs_path("agent", "s1"))
        .await
        .unwrap();
    assert!(history.refs("agent", "s1").await.unwrap().is_empty());
    let head = history.head("agent", "s1").await.unwrap().unwrap();
    assert_eq!(head.id, 2);
    assert_eq!(history.head("agent", "unknown").await.unwrap(), None);

    // A corrupt refs document is tolerated the same way (it is metadata only).
    tokio::fs::write(history.refs_path("agent", "s1"), b"{ not json")
        .await
        .unwrap();
    assert!(history.refs("agent", "s1").await.unwrap().is_empty());
    assert_eq!(history.head("agent", "s1").await.unwrap().unwrap().id, 2);
}

/// The version ids of a session's history, ascending.
async fn history_ids(history: &SessionHistory, session_id: &str) -> Vec<u64> {
    history
        .list("agent", session_id)
        .await
        .unwrap()
        .iter()
        .map(|meta| meta.id)
        .collect()
}

/// A live session with three versions holding `count` 1, 2 and 3.
async fn three_version_history() -> (tempfile::TempDir, SessionHistory, WorkflowSession) {
    let temp = tempfile::tempdir().unwrap();
    let store = DiskSessionStore::new(temp.path());
    let history = store.history();
    let session = WorkflowSession::new("s1");
    session.update_state(json!({"count": 1})).await.unwrap();
    history
        .commit_from_session("agent", &session, Some("one"))
        .await
        .unwrap();
    session.update_state(json!({"count": 2})).await.unwrap();
    history
        .commit_from_session("agent", &session, None)
        .await
        .unwrap();
    session.update_state(json!({"count": 3})).await.unwrap();
    history
        .commit_from_session("agent", &session, None)
        .await
        .unwrap();
    (temp, history, session)
}

// ports tests/unit_tests/core/session/vcs/test_rewind_overwrite.py: a rewind
// reloads the live state from the target version and moves the head back.
#[tokio::test]
async fn rewind_restores_the_live_session_and_moves_the_head() {
    let (_temp, history, session) = three_version_history().await;

    let restored = history
        .rewind("agent", "s1", Some(VersionId(1)), &session)
        .await
        .unwrap();
    assert_eq!(restored.id, 1);
    assert_eq!(restored.label.as_deref(), Some("one"));
    assert_eq!(session.get_state("count").await.unwrap(), Some(json!(1)));
    assert_eq!(history.head("agent", "s1").await.unwrap().unwrap().id, 1);
    assert_eq!(
        history
            .resolve_ref("agent", "s1", "HEAD")
            .await
            .unwrap()
            .unwrap()
            .id,
        1
    );

    // The soft form keeps the newer versions, and `main` still tracks the
    // latest commit.
    assert_eq!(history_ids(&history, "s1").await, [1, 2, 3]);
    assert_eq!(
        history
            .resolve_ref("agent", "s1", "main")
            .await
            .unwrap()
            .unwrap()
            .id,
        3
    );

    // Rewinding to the current HEAD is idempotent, explicit or implicit.
    for target in [Some(VersionId(1)), None] {
        let again = history
            .rewind("agent", "s1", target, &session)
            .await
            .unwrap();
        assert_eq!(again.id, 1);
        assert_eq!(session.get_state("count").await.unwrap(), Some(json!(1)));
        assert_eq!(history_ids(&history, "s1").await, [1, 2, 3]);
    }

    // A missing version is a storage error and leaves the history alone.
    let error = history
        .rewind("agent", "s1", Some(VersionId(9)), &session)
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Session);
    assert_eq!(history.head("agent", "s1").await.unwrap().unwrap().id, 1);
    assert_eq!(history_ids(&history, "s1").await, [1, 2, 3]);

    // An empty history has nothing to rewind to.
    let error = history
        .rewind("agent", "empty", None, &session)
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Session);
}

// ports the `truncate(after_event_id=…)` half of Python's overwrite rewind: the
// hard form deletes the newer versions and re-points `main`.
#[tokio::test]
async fn rewind_hard_truncates_the_newer_versions() {
    let (_temp, history, session) = three_version_history().await;

    let truncated = history
        .rewind_hard("agent", "s1", Some(VersionId(1)), &session)
        .await
        .unwrap();
    assert_eq!(truncated.id, 1);
    assert_eq!(session.get_state("count").await.unwrap(), Some(json!(1)));
    assert_eq!(history_ids(&history, "s1").await, [1]);

    // The version documents are physically gone, the index is rewritten.
    let versions_dir = history.versions_dir("agent", "s1");
    assert!(history.version_path("agent", "s1", VersionId(1)).is_file());
    assert!(!history.version_path("agent", "s1", VersionId(2)).is_file());
    assert!(!history.version_path("agent", "s1", VersionId(3)).is_file());
    let leftover: Vec<String> = std::fs::read_dir(&versions_dir)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    assert_eq!(leftover, ["v000001.json"]);

    // Both refs point at the target now.
    for name in ["HEAD", "main"] {
        assert_eq!(
            history
                .resolve_ref("agent", "s1", name)
                .await
                .unwrap()
                .unwrap()
                .id,
            1,
            "ref {name}"
        );
    }

    // A hard rewind to HEAD is idempotent as well, and the discarded versions
    // are gone for good.
    let again = history
        .rewind_hard("agent", "s1", None, &session)
        .await
        .unwrap();
    assert_eq!(again.id, 1);
    let error = history
        .rewind_hard("agent", "s1", Some(VersionId(3)), &session)
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Session);
    assert_eq!(history_ids(&history, "s1").await, [1]);
}

/// A stored source session with two versions of `count` 1 and 2.
async fn two_version_source() -> (tempfile::TempDir, SessionHistory, SessionScope) {
    let temp = tempfile::tempdir().unwrap();
    let store = DiskSessionStore::new(temp.path());
    let history = store.history();
    let scope = SessionScope::direct("u1");
    let session = WorkflowSession::new("s1");
    session.update_state(json!({"count": 1})).await.unwrap();
    store
        .save_from_session("agent", &scope, &session, Some("source"))
        .await
        .unwrap();
    history
        .commit_from_session("agent", &session, Some("one"))
        .await
        .unwrap();
    session.update_state(json!({"count": 2})).await.unwrap();
    store
        .save_from_session("agent", &scope, &session, Some("source"))
        .await
        .unwrap();
    history
        .commit_from_session("agent", &session, None)
        .await
        .unwrap();
    (temp, history, scope)
}

// ports tests/unit_tests/core/session/vcs/test_fork.py: a fork seeds a
// brand-new session from the source state at the chosen version and leaves the
// source session, its history, and its refs intact.
#[tokio::test]
async fn fork_creates_a_new_session_with_its_own_history() {
    let (temp, history, scope) = two_version_source().await;
    let store = DiskSessionStore::new(temp.path());

    let first = history
        .fork("agent", "s1", Some(VersionId(1)), "s2", Some("fork"))
        .await
        .unwrap();
    assert_eq!(first.id, 1);
    assert_eq!(first.parent, None);
    assert_eq!(first.label.as_deref(), Some("fork of s1@v000001"));

    // The forked session is a real record in the source session's scope.
    assert!(store.exists("agent", "s2").await.unwrap());
    let record = store.load("agent", "s2").await.unwrap().unwrap();
    assert_eq!(record.snapshot.session_id, "s2");
    assert_eq!(record.snapshot.global["count"], json!(1));
    assert_eq!(record.meta.title.as_deref(), Some("fork"));
    let scoped: Vec<String> = store
        .list("agent", Some(&scope))
        .await
        .unwrap()
        .into_iter()
        .map(|meta| meta.session_id)
        .collect();
    assert!(scoped.contains(&"s1".to_string()));
    assert!(scoped.contains(&"s2".to_string()));

    // The fork's history holds exactly one version, and HEAD/main point at it.
    let listed = history.list("agent", "s2").await.unwrap();
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].label.as_deref(), Some("fork of s1@v000001"));
    assert_eq!(listed[0].parent, None);
    let head = history.head("agent", "s2").await.unwrap().unwrap();
    assert_eq!(head.id, 1);
    assert_eq!(head.label.as_deref(), Some("fork of s1@v000001"));
    assert_eq!(
        history
            .resolve_ref("agent", "s2", "main")
            .await
            .unwrap()
            .unwrap()
            .id,
        1
    );

    // Committing on the fork does not touch the source history.
    history
        .commit("agent", "s2", &snapshot("s2", json!({"count": 9})), None)
        .await
        .unwrap();
    assert_eq!(history_ids(&history, "s2").await, [1, 2]);
    assert_eq!(history_ids(&history, "s1").await, [1, 2]);
    assert_eq!(history.head("agent", "s1").await.unwrap().unwrap().id, 2);

    // Forking without `from` clones the source's HEAD, and the source lineage
    // stays its own.
    let tail = history.fork("agent", "s2", None, "s3", None).await.unwrap();
    assert_eq!(tail.id, 1);
    assert_eq!(tail.parent, None);
    assert_eq!(tail.label.as_deref(), Some("fork of s2@v000002"));
    assert_eq!(
        store
            .load("agent", "s3")
            .await
            .unwrap()
            .unwrap()
            .snapshot
            .global["count"],
        json!(9)
    );
}

#[tokio::test]
async fn fork_rejects_bad_targets_and_unknown_sources() {
    let (temp, history, _scope) = two_version_source().await;
    let store = DiskSessionStore::new(temp.path());

    // A fork needs a fresh, valid id.
    let error = history
        .fork("agent", "s1", Some(VersionId(1)), "s1", None)
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    let error = history
        .fork("agent", "s1", Some(VersionId(1)), "../escape", None)
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);

    // An existing destination is rejected.
    history
        .fork("agent", "s1", Some(VersionId(1)), "s2", None)
        .await
        .unwrap();
    let error = history
        .fork("agent", "s1", Some(VersionId(1)), "s2", None)
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(format!("{error}").contains("s2"));
    // The existing forked record was not overwritten by the rejected fork.
    assert_eq!(history_ids(&history, "s2").await, [1]);

    // An unknown source is rejected instead of resolving to another scope.
    for from in [None, Some(VersionId(1))] {
        let error = history
            .fork("agent", "ghost", from, "s4", None)
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation, "from {from:?}");
        assert!(format!("{error}").contains("ghost"));
    }
    assert!(!store.exists("agent", "s4").await.unwrap());

    // A missing source version is a storage error.
    let error = history
        .fork("agent", "s1", Some(VersionId(9)), "s5", None)
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Session);
    assert!(!store.exists("agent", "s5").await.unwrap());

    // The source session, its history, and its record are untouched.
    assert_eq!(history_ids(&history, "s1").await, [1, 2]);
    assert_eq!(history.head("agent", "s1").await.unwrap().unwrap().id, 2);
    assert_eq!(
        store
            .load("agent", "s1")
            .await
            .unwrap()
            .unwrap()
            .snapshot
            .global["count"],
        json!(2)
    );
}
