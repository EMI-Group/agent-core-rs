// ports the SQLite adaptation of tests/unit_tests/core/foundation/store/*
// (the shared contracts plus the `SQLite`-specific prefix, reopen, and
// collection-lifecycle behavior)
#![allow(clippy::unwrap_used, clippy::expect_used)]
#![cfg(feature = "sqlite")]

//! The storage contracts against the `SQLite` backends (feature `sqlite`).
//!
//! Every test opens a fresh database, so the shared contract helpers run against
//! an empty store exactly like the in-memory and local-filesystem suites.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

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
        // A generous TTL keeps the lease live for the whole assertion sequence: a
        // sub-second window could elapse between this write and the count below when
        // the suite runs in parallel.
        assert!(store
            .exclusive_set(
                "ttl",
                KvValue::text("temporary"),
                Some(Duration::from_secs(60))
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

        // Expire the row by backdating its absolute deadline instead of sleeping
        // through a real-time TTL window: `pool()` is the documented companion-query
        // hook and the elapsed-TTL path is identical once `expiry_ms` is in the past.
        assert_eq!(backdate_expiry(&shared, "ttl").await, 1);
        assert_eq!(shared.get("ttl").await.unwrap(), None);
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
    // The TTL is an absolute deadline stored in the database, so the backdated row is
    // still expired after a reopen even though the lease was written moments ago.
    assert!(!reopened.exists("ttl").await.unwrap());
}

/// Moves one row's absolute TTL deadline into the past and returns the rows touched.
///
/// This is the deterministic stand-in for waiting out a real TTL: `expiry_ms` is
/// stored in the database, so the expired-row path is the same either way.
async fn backdate_expiry(store: &SqliteKvStore, key: &str) -> u64 {
    let now_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis();
    sqlx::query("UPDATE kv_store SET expiry_ms = ?1 WHERE k = ?2")
        .bind(i64::try_from(now_ms).unwrap_or(i64::MAX) - 1_000)
        .bind(key)
        .execute(store.pool())
        .await
        .unwrap()
        .rows_affected()
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
