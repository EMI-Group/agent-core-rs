// ports the SQLite adaptation of tests/unit_tests/core/foundation/store/*
// (the shared contracts plus the `SQLite`-specific prefix, reopen, and
// collection-lifecycle behavior)
#![allow(clippy::unwrap_used, clippy::expect_used)]
#![cfg(feature = "sqlite")]

//! The storage contracts against the `SQLite` backends (feature `sqlite`).
//!
//! Every test opens a fresh database, so the shared contract helpers run against
//! an empty store exactly like the in-memory and local-filesystem suites.

use std::time::Duration;

use tempfile::TempDir;

use crate::store::{
    Document, DocumentStore, KvStore, KvValue, Query, SqliteDocumentStore, SqliteKvStore,
};

#[tokio::test]
async fn sqlite_kv_satisfies_the_kv_contract() {
    let store = SqliteKvStore::connect("sqlite::memory:").await.unwrap();
    super::kv::assert_kv_contract(&store).await;
}

#[tokio::test]
async fn sqlite_document_satisfies_the_document_contract() {
    let store = SqliteDocumentStore::connect("sqlite::memory:")
        .await
        .unwrap();
    super::document::assert_document_contract(&store).await;
}

#[tokio::test]
async fn sqlite_kv_prefix_scans_treat_wildcards_literally() {
    let store = SqliteKvStore::connect("sqlite::memory:").await.unwrap();
    for key in ["like:a%b", "like:a_b", "like:axb", "like:a"] {
        store.set(key, KvValue::text(key)).await.unwrap();
    }

    let percent = store.get_by_prefix("like:a%").await.unwrap();
    assert_eq!(
        percent.keys().cloned().collect::<Vec<String>>(),
        vec!["like:a%b".to_string()]
    );
    let underscore = store.get_by_prefix("like:a_").await.unwrap();
    assert_eq!(
        underscore.keys().cloned().collect::<Vec<String>>(),
        vec!["like:a_b".to_string()]
    );
    assert_eq!(store.get_by_prefix("like:").await.unwrap().len(), 4);
}

#[tokio::test]
async fn sqlite_kv_reopens_a_file_database_and_shares_its_pool() {
    let directory = TempDir::new().unwrap();
    let path = directory.path().join("kv.db");
    let url = format!("sqlite:{}", path.display());

    {
        let store = SqliteKvStore::connect(&url).await.unwrap();
        store.set("text", KvValue::text("hello")).await.unwrap();
        store
            .set("bytes", KvValue::bytes(vec![0x00, 0xff, 0xfe]))
            .await
            .unwrap();
        assert!(store
            .exclusive_set(
                "ttl",
                KvValue::text("temporary"),
                Some(Duration::from_millis(40))
            )
            .await
            .unwrap());

        // A store wrapping the same pool sees the committed rows immediately.
        let shared = SqliteKvStore::from_pool(store.pool().clone())
            .await
            .unwrap();
        assert_eq!(
            shared.get("text").await.unwrap(),
            Some(KvValue::text("hello"))
        );
        assert_eq!(shared.len().await.unwrap(), 3);
    }

    let reopened = SqliteKvStore::connect(&url).await.unwrap();
    assert_eq!(
        reopened.get("text").await.unwrap(),
        Some(KvValue::text("hello"))
    );
    assert_eq!(
        reopened.get("bytes").await.unwrap(),
        Some(KvValue::bytes(vec![0x00, 0xff, 0xfe]))
    );
    // The TTL is an absolute deadline stored in the database, so it still expires.
    tokio::time::sleep(Duration::from_millis(60)).await;
    assert!(!reopened.exists("ttl").await.unwrap());
}

#[tokio::test]
async fn sqlite_document_collection_lifecycle() {
    let store = SqliteDocumentStore::connect("sqlite::memory:")
        .await
        .unwrap();

    store.create_collection("lifecycle").await.unwrap();
    store.create_collection("lifecycle").await.unwrap();
    assert_eq!(
        store.list_collections().await.unwrap(),
        vec!["lifecycle".to_string()]
    );

    store
        .upsert("lifecycle", vec![Document::new("a").with_field("n", 1)])
        .await
        .unwrap();
    assert_eq!(
        store.count("lifecycle", &Query::default()).await.unwrap(),
        1
    );

    store.drop_collection("lifecycle").await.unwrap();
    assert_eq!(store.list_collections().await.unwrap().len(), 0);
    assert!(store.get("lifecycle", "a").await.unwrap().is_none());
    assert_eq!(
        store.count("lifecycle", &Query::default()).await.unwrap(),
        0
    );
    // Dropping a missing collection is a no-op.
    store.drop_collection("lifecycle").await.unwrap();
}
