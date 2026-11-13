//! `open_store` dispatch: the `StorageSpec` keys, their precedence, laziness and the deferrals.

use std::path::Path;

use openjiuwen_common::error::{Error, ErrorKind};
use serde_json::json;

use crate::schema::{StorageSpec, TeamRole};
use crate::storage::records::{MemberRecord, TeamInfoRecord};
use crate::storage::{open_store, SqliteTeamStore, TeamStore};

use super::support::{path_value, spec};

/// Opens `spec` and returns the error it must be rejected with.
fn rejected(spec: &StorageSpec) -> Error {
    match open_store(spec) {
        Ok(_) => panic!("the storage spec must be rejected"),
        Err(error) => error,
    }
}

fn sample_member() -> MemberRecord {
    MemberRecord::new("alpha", "dev-1", TeamRole::Teammate)
}

fn uri_for(path: &Path) -> String {
    format!("sqlite://{}", path.display())
}

#[tokio::test]
async fn memory_spec_selects_the_in_memory_store() {
    let store = open_store(&spec("memory", &[])).unwrap();
    assert_eq!(store.name(), "memory");
    store.insert_member(sample_member()).await.unwrap();
    assert_eq!(store.list_members("alpha").await.unwrap().len(), 1);
    // `open_store` is not a singleton: a second call is an independent, empty backend.
    let second = open_store(&spec("", &[])).unwrap();
    assert_eq!(second.name(), "memory");
    assert_eq!(second.list_members("alpha").await.unwrap().len(), 0);
}

#[tokio::test]
async fn sqlite_memory_path_is_usable() {
    let store = open_store(&spec("sqlite", &[("path", json!(":memory:"))])).unwrap();
    assert_eq!(store.name(), "sqlite");
    store.insert_member(sample_member()).await.unwrap();
    assert_eq!(store.list_members("alpha").await.unwrap().len(), 1);
}

#[tokio::test]
async fn sqlite_file_path_persists_across_open_store_calls() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("team.db");
    let store = open_store(&spec("sqlite", &[("path", path_value(&path))])).unwrap();
    store
        .insert_member(MemberRecord::new("alpha", "dev-1", TeamRole::Teammate))
        .await
        .unwrap();
    store
        .upsert_team(TeamInfoRecord::new("alpha"))
        .await
        .unwrap();
    drop(store);

    let reopened = open_store(&spec("sqlite", &[("path", path_value(&path))])).unwrap();
    let members = reopened.list_members("alpha").await.unwrap();
    assert_eq!(members.len(), 1);
    assert_eq!(members[0].member_name, "dev-1");
    assert!(reopened.team("alpha").await.unwrap().is_some());
}

#[tokio::test]
async fn sqlite_accepts_memory_urls_database_keys_and_absolute_urls() {
    // A `sqlite:` URL naming `:memory:` selects the private in-memory database.
    let memory = open_store(&spec("sqlite", &[("url", json!("sqlite::memory:"))])).unwrap();
    assert_eq!(memory.name(), "sqlite");
    memory.insert_member(sample_member()).await.unwrap();
    assert_eq!(memory.list_members("alpha").await.unwrap().len(), 1);

    // The `database` key is accepted as well.
    let dir = tempfile::tempdir().unwrap();
    let database_path = dir.path().join("database-key.db");
    let database =
        open_store(&spec("sqlite", &[("database", path_value(&database_path))])).unwrap();
    assert_eq!(database.name(), "sqlite");
    database.insert_member(sample_member()).await.unwrap();
    assert!(database_path.exists());

    // An absolute filesystem path as a `sqlx` URL.
    let url_path = dir.path().join("url.db");
    let from_url = open_store(&spec("sqlite", &[("url", json!(uri_for(&url_path)))])).unwrap();
    from_url.insert_member(sample_member()).await.unwrap();
    assert!(url_path.exists());
    assert_eq!(from_url.list_members("alpha").await.unwrap().len(), 1);
}

#[tokio::test]
async fn sqlite_path_wins_over_url() {
    let dir = tempfile::tempdir().unwrap();
    let chosen = dir.path().join("chosen.db");
    let ignored = dir.path().join("ignored.db");
    let store = open_store(&spec(
        "sqlite",
        &[
            ("path", path_value(&chosen)),
            ("url", json!(uri_for(&ignored))),
        ],
    ))
    .unwrap();
    store.insert_member(sample_member()).await.unwrap();
    assert!(chosen.exists());
    assert!(!ignored.exists(), "the url must not be connected to");
}

#[test]
fn sqlite_rejects_missing_and_malformed_params() {
    let missing = rejected(&spec("sqlite", &[]));
    assert_eq!(missing.kind(), ErrorKind::Config);
    assert!(missing
        .to_string()
        .contains("requires a 'path' or 'url' parameter"));

    let empty = rejected(&spec("sqlite", &[("path", json!(""))]));
    assert_eq!(empty.kind(), ErrorKind::Validation);
    assert!(empty.to_string().contains("non-empty"));

    let non_string = rejected(&spec("sqlite", &[("path", json!(1))]));
    assert_eq!(non_string.kind(), ErrorKind::Validation);
    assert!(non_string.to_string().contains("must be a string"));

    let non_string_url = rejected(&spec("sqlite", &[("url", json!(true))]));
    assert_eq!(non_string_url.kind(), ErrorKind::Validation);
    assert!(non_string_url.to_string().contains("must be a string"));
}

#[tokio::test]
async fn deferred_backends_stay_typed() {
    for kind in ["postgresql", "mysql"] {
        let err = rejected(&spec(kind, &[]));
        assert_eq!(err.kind(), ErrorKind::Unsupported);
        assert!(
            err.to_string().contains("driver is not available offline"),
            "{err}"
        );
    }
    let err = rejected(&spec("pulsar", &[]));
    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert!(err.to_string().contains("unknown storage backend 'pulsar'"));
}

#[tokio::test]
async fn open_store_connects_lazily() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("lazy").join("team.db");
    let store = open_store(&spec("sqlite", &[("path", path_value(&path))])).unwrap();
    // The synchronous seam only records the target: no directory, no file, no connection.
    assert!(!path.exists());
    assert!(!path.parent().unwrap().exists());

    store.insert_member(sample_member()).await.unwrap();
    assert!(path.exists());
    assert!(path.parent().unwrap().exists());
    assert_eq!(store.list_members("alpha").await.unwrap().len(), 1);
}

#[tokio::test]
async fn two_in_memory_sqlite_stores_are_independent() {
    let first = SqliteTeamStore::in_memory().await.unwrap();
    let second = SqliteTeamStore::in_memory().await.unwrap();
    first.insert_member(sample_member()).await.unwrap();
    assert_eq!(first.list_members("alpha").await.unwrap().len(), 1);
    assert_eq!(second.list_members("alpha").await.unwrap().len(), 0);
    assert_eq!(second.message_count("alpha", "dev-1").await.unwrap(), 0);
}
