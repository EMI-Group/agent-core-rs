//! Behavior tests for the session link graph ([`SessionLinks`]) and the store's
//! link + scope plumbing.
//!
//! Ports the relationship half of
//! `openjiuwen/core/session/session_controller/` (v0.1.20) — Python's
//! `ChainSession` downstream links — onto the local `links.json` layout. The
//! suite is offline and deterministic: temporary roots, no sleeps, no network.
#![allow(clippy::unwrap_used, clippy::expect_used)]
use openjiuwen_common::error::ErrorKind;
use openjiuwen_session::session::{SessionSnapshot, SessionType};
use openjiuwen_session::session_controller::{
    DiskSessionStore, LinkKind, LinksIndex, SessionController, SessionLink, SessionLinks,
    SessionMeta, SessionRecord, SessionScope,
};
use openjiuwen_session::sessions::WorkflowSession;
use serde_json::json;

/// Builds a record for `session_id` with a minimal agent snapshot.
fn record(session_id: &str) -> SessionRecord {
    SessionRecord::new(
        SessionMeta::new(session_id),
        SessionSnapshot::new(session_id, SessionType::Agent),
    )
}

/// Fills `store` with the traversal fixture: `root -> child -> grandchild`, plus
/// a fork of `child` (a second level-2 node next to `grandchild`).
async fn seed_fixture(store: &DiskSessionStore) {
    store.link_parent("agent", "child", "root").await.unwrap();
    store
        .link_parent("agent", "grandchild", "child")
        .await
        .unwrap();
    store.link_fork("agent", "branch", "child").await.unwrap();
}

#[tokio::test]
async fn links_round_trip_through_a_fresh_store() {
    let temp = tempfile::tempdir().unwrap();
    let store = DiskSessionStore::new(temp.path());
    store.link_parent("agent", "child", "root").await.unwrap();
    store.link_fork("agent", "branch", "child").await.unwrap();

    let path = store.links_path("agent");
    assert_eq!(
        path,
        temp.path().join("agent").join("links.json"),
        "links live at <base>/<agent_id>/links.json"
    );
    let raw: serde_json::Value =
        serde_json::from_slice(&tokio::fs::read(&path).await.unwrap()).unwrap();
    assert_eq!(raw["agent_id"], "agent");
    assert_eq!(raw["links"]["child"]["parent"], "root");
    assert_eq!(raw["links"]["branch"]["forked_from"], "child");
    assert!(raw["links"]["child"]["created_at"].is_string());
    assert!(
        !tokio::fs::read_dir(path.parent().unwrap())
            .await
            .unwrap()
            .next_entry()
            .await
            .unwrap()
            .unwrap()
            .file_name()
            .to_string_lossy()
            .contains(".tmp"),
        "the atomic write leaves no temporary file"
    );

    // A fresh store over the same base reads the same graph back.
    let reopened = DiskSessionStore::new(temp.path());
    let links = reopened.read_links("agent").await.unwrap();
    assert_eq!(links.len(), 3);
    assert_eq!(links.parent_of("child"), Some("root"));
    assert_eq!(links.parent_of("branch"), None, "a fork has no parent link");
    assert_eq!(links.get("branch").unwrap().kind(), Some(LinkKind::Fork));
    assert_eq!(links.forks_of("child"), vec!["branch".to_string()]);
    assert_eq!(links.children_of("root"), vec!["child".to_string()]);
    assert_eq!(
        links.get("child").unwrap().created_at,
        raw["links"]["child"]["created_at"]
            .as_str()
            .unwrap()
            .to_string()
    );
    assert_eq!(
        reopened.parent_of("agent", "child").await.unwrap(),
        Some("root".to_string())
    );
    assert_eq!(
        reopened.children_of("agent", "root").await.unwrap(),
        ["child"]
    );
}

#[tokio::test]
async fn traversal_reports_the_chain_forks_and_depth() {
    let temp = tempfile::tempdir().unwrap();
    let store = DiskSessionStore::new(temp.path());
    seed_fixture(&store).await;

    let links = store.read_links("agent").await.unwrap();
    assert_eq!(links.parent_of("root"), None);
    assert_eq!(links.parent_of("child"), Some("root"));
    assert_eq!(links.parent_of("grandchild"), Some("child"));
    assert_eq!(links.children_of("root"), vec!["child".to_string()]);
    assert_eq!(links.children_of("child"), vec!["grandchild".to_string()]);
    assert_eq!(links.forks_of("child"), vec!["branch".to_string()]);
    assert_eq!(links.forks_of("grandchild").len(), 0);

    assert_eq!(
        links.chain("grandchild").unwrap(),
        ["root", "child", "grandchild"]
    );
    assert_eq!(links.chain("branch").unwrap(), ["root", "child", "branch"]);
    assert_eq!(links.chain("root").unwrap(), ["root"]);
    assert_eq!(links.ancestors("grandchild").unwrap(), ["root", "child"]);
    assert_eq!(links.ancestors("root").unwrap().len(), 0);
    assert_eq!(
        links.descendants("root").unwrap(),
        ["child", "grandchild", "branch"]
    );
    assert_eq!(links.descendants("grandchild").unwrap().len(), 0);
    assert_eq!(links.depth("root").unwrap(), 0);
    assert_eq!(links.depth("child").unwrap(), 1);
    assert_eq!(links.depth("branch").unwrap(), 2);
    assert_eq!(links.depth("grandchild").unwrap(), 2);
    assert_eq!(links.roots(), vec!["root".to_string()]);
    assert_eq!(
        links.leaves(),
        vec!["branch".to_string(), "grandchild".to_string()]
    );

    // Unknown sessions traverse to empty results rather than erroring.
    assert_eq!(links.chain("nope").unwrap().len(), 0);
    assert_eq!(links.ancestors("nope").unwrap().len(), 0);
    assert_eq!(links.descendants("nope").unwrap().len(), 0);
    assert_eq!(links.depth("nope").unwrap(), 0);

    // The store wrappers expose the same traversal over the persisted graph.
    assert_eq!(
        store.chain("agent", "grandchild").await.unwrap(),
        ["root", "child", "grandchild"]
    );
    assert_eq!(
        store.ancestors("agent", "grandchild").await.unwrap(),
        ["root", "child"]
    );
    assert_eq!(
        store.descendants("agent", "child").await.unwrap(),
        ["grandchild", "branch"]
    );
}

#[tokio::test]
async fn a_stored_cycle_is_a_session_error_not_a_hang() {
    let temp = tempfile::tempdir().unwrap();
    let store = DiskSessionStore::new(temp.path());
    let path = store.links_path("agent");
    tokio::fs::create_dir_all(path.parent().unwrap())
        .await
        .unwrap();
    let document = json!({
        "agent_id": "agent",
        "updated_at": 1.0,
        "links": {
            "a": {"session_id": "a", "parent": "b", "children": []},
            "b": {"session_id": "b", "parent": "a", "children": ["a"]},
        },
    });
    tokio::fs::write(&path, serde_json::to_vec(&document).unwrap())
        .await
        .unwrap();

    // The document is readable as stored; traversal reports the cycle.
    let links = store.read_links("agent").await.unwrap();
    assert_eq!(links.len(), 2);
    for result in [
        links.chain("a"),
        links.chain("b"),
        links.ancestors("b"),
        links.descendants("a"),
        links.check_acyclic().map(|()| Vec::new()),
    ] {
        let error = result.unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Session);
        assert!(error.to_string().contains("cycle"), "{error}");
    }
    assert_eq!(links.depth("a").unwrap_err().kind(), ErrorKind::Session);

    let error = store.chain("agent", "a").await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Session);
    assert!(error.to_string().contains("cycle at session"), "{error}");
}

#[tokio::test]
async fn self_links_and_cycle_creating_links_are_validation_errors() {
    let temp = tempfile::tempdir().unwrap();
    let store = DiskSessionStore::new(temp.path());
    let mut links = SessionLinks::new();

    let error = links
        .link(SessionLink::new("a", LinkKind::Parent, "a"))
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    let error = links
        .link(SessionLink::new("a", LinkKind::Fork, "a"))
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);

    links.record_parent("b", "a").unwrap();
    links.record_parent("c", "b").unwrap();
    let error = links.record_fork("a", "c").unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("cycle"), "{error}");

    // The same guard runs through the store, over the persisted graph.
    store.link_parent("agent", "b", "a").await.unwrap();
    store.link_parent("agent", "c", "b").await.unwrap();
    let error = store.link_parent("agent", "a", "c").await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert_eq!(
        store.parent_of("agent", "a").await.unwrap(),
        None,
        "a rejected link must not be persisted"
    );
}

#[tokio::test]
async fn delete_drops_exactly_one_link_entry() {
    let temp = tempfile::tempdir().unwrap();
    let store = DiskSessionStore::new(temp.path());
    let scope = SessionScope::main();
    for session_id in ["root", "mid", "leaf"] {
        store
            .save_snapshot(
                "agent",
                &scope,
                session_id,
                SessionSnapshot::new(session_id, SessionType::Agent),
                None,
            )
            .await
            .unwrap();
    }
    store.link_parent("agent", "mid", "root").await.unwrap();
    store.link_parent("agent", "leaf", "mid").await.unwrap();
    store.link_fork("agent", "branch", "mid").await.unwrap();

    assert!(store.delete("agent", "mid").await.unwrap());
    let links = store.read_links("agent").await.unwrap();
    assert!(
        !links.contains("mid"),
        "the deleted session's entry is gone"
    );
    assert_eq!(links.len(), 3, "only that entry is dropped");
    assert!(links.contains("root"));
    assert!(links.contains("leaf"));
    assert!(links.contains("branch"));
    // `leaf` was `mid`'s child and `branch` its fork: both keep their entry and
    // lose the lineage pointer (documented on `SessionLinks::remove`).
    assert_eq!(links.get("leaf").unwrap().parent, None);
    assert_eq!(links.get("branch").unwrap().forked_from, None);
    assert_eq!(links.children_of("root").len(), 0);

    // Deleting an unknown session stays a `false` no-op.
    assert!(!store.delete("agent", "mid").await.unwrap());
    assert_eq!(store.read_links("agent").await.unwrap().len(), 3);
}

#[tokio::test]
async fn invalid_ids_and_corrupt_documents_are_typed_errors() {
    let temp = tempfile::tempdir().unwrap();
    let store = DiskSessionStore::new(temp.path());

    for agent_id in ["", "../x"] {
        assert_eq!(
            store.read_links(agent_id).await.unwrap_err().kind(),
            ErrorKind::Validation
        );
        assert_eq!(
            store
                .link_parent(agent_id, "child", "root")
                .await
                .unwrap_err()
                .kind(),
            ErrorKind::Validation
        );
    }
    for session_id in ["", "../x"] {
        assert_eq!(
            store
                .link_parent("agent", session_id, "root")
                .await
                .unwrap_err()
                .kind(),
            ErrorKind::Validation
        );
        assert_eq!(
            store
                .link_fork("agent", "child", session_id)
                .await
                .unwrap_err()
                .kind(),
            ErrorKind::Validation
        );
    }

    let path = store.links_path("agent");
    tokio::fs::create_dir_all(path.parent().unwrap())
        .await
        .unwrap();
    tokio::fs::write(&path, b"{not json").await.unwrap();
    let error = store.read_links("agent").await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Session);
    assert!(error.to_string().contains("links.json"), "{error}");
    let error = store.chain("agent", "a").await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Session);
}

#[test]
fn link_document_round_trips_through_its_index_type() {
    let mut links = SessionLinks::new();
    links.record_parent("child", "root").unwrap();
    links.record_fork("branch", "child").unwrap();
    let document = LinksIndex::new("agent", links.clone());
    assert_eq!(document.agent_id, "agent");
    assert!(document.updated_at > 0.0);
    let restored: LinksIndex =
        serde_json::from_value(serde_json::to_value(&document).unwrap()).unwrap();
    assert_eq!(restored, document);
    assert_eq!(restored.into_links(), links);
}

#[tokio::test]
async fn a_session_without_a_scope_is_a_validation_error() {
    let temp = tempfile::tempdir().unwrap();
    let store = DiskSessionStore::new(temp.path());
    let controller = SessionController::new(store.clone(), "agent");

    let error = controller.upsert(record("unknown")).await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("unknown"), "{error}");
    assert!(error.to_string().contains("explicit scope"), "{error}");

    // `upsert_in` is the explicit-scope alternative: it files the record there.
    let scope = SessionScope::group("g1");
    let meta = controller.upsert_in(&scope, record("fresh")).await.unwrap();
    assert_eq!(meta.session_id, "fresh");
    assert_eq!(meta.version, 1);
    assert_eq!(controller.list(Some(&scope)).await.unwrap().len(), 1);
    assert_eq!(
        controller
            .get("fresh")
            .await
            .unwrap()
            .unwrap()
            .snapshot
            .session_id,
        "fresh"
    );

    // A known session keeps its scope through `upsert`.
    let mut known = controller.get("fresh").await.unwrap().unwrap();
    known.snapshot.global = json!({"step": 2});
    let meta = controller.upsert(known).await.unwrap();
    assert_eq!(meta.version, 2);
    let index = store.read_index("agent").await.unwrap();
    assert!(index.scopes.contains_key("agent:agent:main:group:g1"));
}

#[tokio::test]
async fn flush_requires_a_known_scope_but_save_in_does_not() {
    let temp = tempfile::tempdir().unwrap();
    let store = DiskSessionStore::new(temp.path());
    let scope = SessionScope::custom("tenant");
    let controller = SessionController::new(store.clone(), "agent");
    controller.create(&scope, "known").await.unwrap();
    controller.flush().await.unwrap();

    // A controller over an agent whose index knows nothing must not silently file
    // an upserted record under `main`.
    let empty = SessionController::new(store.clone(), "other");
    let error = empty.upsert(record("orphan")).await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);

    // `save`/`save_in` take an explicit scope and succeed for a new session.
    let session = WorkflowSession::new("wf-1");
    let meta = controller
        .save_in(&session, &scope, Some("one"))
        .await
        .unwrap();
    assert_eq!(meta.title.as_deref(), Some("one"));
    assert_eq!(meta.version, 1);
    assert_eq!(
        store
            .list("agent", Some(&scope))
            .await
            .unwrap()
            .iter()
            .map(|meta| meta.session_id.as_str())
            .collect::<Vec<_>>(),
        ["wf-1", "known"]
    );
    let meta = controller.save(&session, &scope, None).await.unwrap();
    assert_eq!(meta.version, 2);
}

#[tokio::test]
async fn an_unparseable_scope_key_is_a_session_error() {
    let temp = tempfile::tempdir().unwrap();
    let store = DiskSessionStore::new(temp.path());
    let controller = SessionController::new(store.clone(), "agent");
    controller
        .create(&SessionScope::main(), "s1")
        .await
        .unwrap();

    // Rewrite the index so `s1` sits under a key that is not a scope key.
    let mut index = store.read_index("agent").await.unwrap();
    let scoped = index.scopes.remove("agent:agent:main").unwrap();
    index.scopes.insert("not-a-scope-key".to_string(), scoped);
    store.write_index("agent", &index).await.unwrap();

    for error in [
        controller.upsert(record("s1")).await.unwrap_err(),
        controller.flush().await.unwrap_err(),
    ] {
        assert_eq!(error.kind(), ErrorKind::Session);
        assert!(error.to_string().contains("not-a-scope-key"), "{error}");
    }
}

#[tokio::test]
async fn concurrent_link_records_do_not_lose_updates() {
    let temp = tempfile::tempdir().unwrap();
    let store = DiskSessionStore::new(temp.path());
    let mut tasks = Vec::new();
    for index in 0..12 {
        let store = store.clone();
        tasks.push(tokio::spawn(async move {
            store
                .link_parent("agent", &format!("child-{index:02}"), "root")
                .await
                .unwrap();
        }));
    }
    for task in tasks {
        task.await.unwrap();
    }

    let links = store.read_links("agent").await.unwrap();
    assert_eq!(links.len(), 13, "every recorded child survived");
    let mut children = links.children_of("root").to_vec();
    assert_eq!(children.len(), 12);
    children.sort();
    let expected: Vec<String> = (0..12).map(|index| format!("child-{index:02}")).collect();
    assert_eq!(children, expected);
}
