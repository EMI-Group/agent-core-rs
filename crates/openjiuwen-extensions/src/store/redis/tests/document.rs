//! Unit tests for the live `Redis` document backend.
//!
//! Python counterpart: none — `openjiuwen/extensions/store` has no document
//! backend (its `db/` family is GaussDB), so these suites verify the Rust-side
//! extension against the kernel `InMemoryDocumentStore` reference instead.

use super::*;
use crate::redis::testsupport::{FakeOptions, FakeRedis};
use openjiuwen_common::error::ErrorKind;
use openjiuwen_core::foundation::store::{
    Document, DocumentStore, InMemoryDocumentStore, Query, SortOrder,
};
use serde_json::json;

/// A configuration carrying only `url`.
fn url_config(url: &str) -> RedisStoreConfig {
    RedisStoreConfig {
        url: Some(url.to_string()),
        ..RedisStoreConfig::default()
    }
}

/// A document store bound to a running fake server (the client connects lazily).
fn store_for(fake: &FakeRedis) -> RedisDocumentStore {
    RedisDocumentStore::new(url_config(&fake.url())).unwrap()
}

/// A document with a name and an age, the fixture the query tests use.
fn person(id: &str, name: &str, age: i64) -> Document {
    Document::new(id).with_field("name", name).with_field("age", age)
}

/// The ids of `documents`, in order.
fn ids(documents: &[Document]) -> Vec<String> {
    documents
        .iter()
        .map(|document| document.id.clone())
        .collect()
}

#[tokio::test]
async fn document_store_creates_lists_and_drops_collections() {
    let fake = FakeRedis::start().await;
    let store = store_for(&fake);
    assert_eq!(store.list_collections().await.unwrap(), Vec::<String>::new());

    store.create_collection("beta").await.unwrap();
    store.create_collection("alpha").await.unwrap();
    // Creating an existing collection is idempotent.
    store.create_collection("alpha").await.unwrap();
    assert_eq!(
        store.list_collections().await.unwrap(),
        vec!["alpha".to_string(), "beta".to_string()]
    );

    // `upsert` registers the collection it writes into.
    store
        .upsert("gamma", vec![person("g", "Gina", 30)])
        .await
        .unwrap();
    assert_eq!(
        store.list_collections().await.unwrap(),
        vec![
            "alpha".to_string(),
            "beta".to_string(),
            "gamma".to_string()
        ]
    );

    store.drop_collection("beta").await.unwrap();
    // Dropping a missing collection is a no-op.
    store.drop_collection("beta").await.unwrap();
    assert_eq!(
        store.list_collections().await.unwrap(),
        vec!["alpha".to_string(), "gamma".to_string()]
    );

    store.drop_collection("gamma").await.unwrap();
    assert!(store.get("gamma", "g").await.unwrap().is_none());
    assert!(
        fake.state()
            .keys()
            .iter()
            .all(|key| !key.contains("gamma")),
        "the dropped collection keeps keys: {:?}",
        fake.state().keys()
    );
    fake.shutdown().await;
}

#[tokio::test]
async fn document_store_upserts_gets_and_replaces_documents() {
    let fake = FakeRedis::start().await;
    let store = store_for(&fake);
    let nested = Document::new("nested")
        .with_field("meta", json!({"owner": "ada", "tags": ["rust", "store"]}))
        .with_field("ratio", 0.5)
        .with_field("count", 9_007_199_254_740_991_i64)
        .with_field("flag", true)
        .with_field("nil", serde_json::Value::Null);
    let documents = vec![nested.clone(), person("a", "Ada", 36)];
    assert_eq!(store.upsert("docs", documents).await.unwrap(), 2);

    // Every JSON shape survives the round trip unchanged.
    assert_eq!(store.get("docs", "nested").await.unwrap(), Some(nested));
    let ada = store.get("docs", "a").await.unwrap().expect("stored");
    assert_eq!(ada.field("name"), Some(&json!("Ada")));
    assert_eq!(ada.field("age"), Some(&json!(36)));

    // A second upsert with the same id replaces the whole document.
    assert_eq!(
        store
            .upsert("docs", vec![Document::new("a").with_field("age", 37)])
            .await
            .unwrap(),
        1
    );
    let replaced = store.get("docs", "a").await.unwrap().expect("replaced");
    assert_eq!(replaced.field("age"), Some(&json!(37)));
    assert_eq!(replaced.field("name"), None);

    // An empty upsert writes nothing but still registers the collection.
    assert_eq!(store.upsert("empty", Vec::new()).await.unwrap(), 0);
    assert!(store
        .list_collections()
        .await
        .unwrap()
        .contains(&"empty".to_string()));
    assert!(store.get("docs", "missing").await.unwrap().is_none());
    fake.shutdown().await;
}

#[tokio::test]
async fn document_store_query_matches_the_reference_window() {
    let fake = FakeRedis::start().await;
    let store = store_for(&fake);
    let reference = InMemoryDocumentStore::new();
    let documents: Vec<Document> = (0..25)
        .map(|index| {
            person(
                &format!("doc-{index:02}"),
                &format!("name-{index:02}"),
                i64::from(index % 5) * 10,
            )
        })
        .collect();
    store.upsert("paged", documents.clone()).await.unwrap();
    reference.upsert("paged", documents).await.unwrap();

    let queries = [
        Query {
            order_by: vec![("id".to_string(), SortOrder::Asc)],
            offset: 5,
            limit: Some(10),
            ..Query::default()
        },
        Query {
            order_by: vec![("id".to_string(), SortOrder::Asc)],
            offset: 20,
            limit: Some(10),
            ..Query::default()
        },
        Query {
            filter: json!({"age": {"$gte": 20}}),
            order_by: vec![
                ("age".to_string(), SortOrder::Desc),
                ("id".to_string(), SortOrder::Asc),
            ],
            offset: 2,
            limit: Some(7),
            ..Query::default()
        },
        Query {
            filter: json!({"name": "name-13"}),
            ..Query::default()
        },
        Query {
            order_by: vec![("age".to_string(), SortOrder::Asc)],
            ..Query::default()
        },
    ];
    for query in &queries {
        let expected = reference.query("paged", query).await.unwrap();
        let actual = store.query("paged", query).await.unwrap();
        assert_eq!(ids(&actual), ids(&expected), "query {query:?}");
    }
    fake.shutdown().await;
}

#[tokio::test]
async fn document_store_count_ignores_paging() {
    let fake = FakeRedis::start().await;
    let store = store_for(&fake);
    let documents: Vec<Document> = (0..6)
        .map(|index| person(&format!("doc-{index}"), "Ada", i64::from(index) * 10))
        .collect();
    store.upsert("counts", documents).await.unwrap();

    assert_eq!(
        store.count("counts", &Query::default()).await.unwrap(),
        6,
        "every document is counted"
    );
    let paged = Query {
        filter: json!({"age": {"$gte": 30}}),
        limit: Some(1),
        offset: 4,
        order_by: vec![("age".to_string(), SortOrder::Desc)],
    };
    assert_eq!(
        store.count("counts", &paged).await.unwrap(),
        3,
        "`count` ignores limit, offset and ordering"
    );
    fake.shutdown().await;
}

#[tokio::test]
async fn document_store_delete_removes_the_matching_documents() {
    let fake = FakeRedis::start().await;
    let store = store_for(&fake);
    let documents = vec![
        person("a", "Ada", 36),
        person("b", "Bob", 24),
        person("c", "Cid", 36),
    ];
    store.upsert("deleting", documents).await.unwrap();

    // Paging and ordering are ignored by `delete`: every match goes.
    let query = Query {
        filter: json!({"age": {"$gte": 30}}),
        limit: Some(1),
        offset: 1,
        order_by: vec![("age".to_string(), SortOrder::Desc)],
    };
    assert_eq!(store.delete("deleting", &query).await.unwrap(), 2);
    assert_eq!(store.delete("deleting", &query).await.unwrap(), 0);
    assert!(store.get("deleting", "a").await.unwrap().is_none());
    assert!(store.get("deleting", "b").await.unwrap().is_some());
    assert_eq!(
        store.count("deleting", &Query::default()).await.unwrap(),
        1
    );

    assert_eq!(
        store
            .delete("deleting", &Query::default())
            .await
            .unwrap(),
        1
    );
    assert_eq!(
        store.count("deleting", &Query::default()).await.unwrap(),
        0
    );
    // The id set shrank with the documents, so the collection is empty, not gone.
    assert!(store
        .list_collections()
        .await
        .unwrap()
        .contains(&"deleting".to_string()));
    fake.shutdown().await;
}

#[tokio::test]
async fn document_store_missing_collections_read_like_empty() {
    let fake = FakeRedis::start().await;
    let store = store_for(&fake);
    assert!(store.get("absent", "a").await.unwrap().is_none());
    assert_eq!(
        store.query("absent", &Query::default()).await.unwrap(),
        Vec::<Document>::new()
    );
    assert_eq!(
        store.count("absent", &Query::default()).await.unwrap(),
        0
    );
    assert_eq!(
        store.delete("absent", &Query::default()).await.unwrap(),
        0
    );
    assert!(!store
        .list_collections()
        .await
        .unwrap()
        .contains(&"absent".to_string()));
    fake.shutdown().await;
}

#[tokio::test]
async fn document_store_rejects_invalid_collection_names() {
    let fake = FakeRedis::start().await;
    let store = store_for(&fake);
    for name in ["", "a/b", "a\\b", "..", ".", "bad\0name"] {
        for error in [
            store.create_collection(name).await.expect_err("create"),
            store.upsert(name, Vec::new()).await.expect_err("upsert"),
            store.get(name, "a").await.expect_err("get"),
            store
                .query(name, &Query::default())
                .await
                .expect_err("query"),
            store
                .count(name, &Query::default())
                .await
                .expect_err("count"),
            store
                .delete(name, &Query::default())
                .await
                .expect_err("delete"),
            store.drop_collection(name).await.expect_err("drop"),
        ] {
            assert_eq!(error.kind(), ErrorKind::Validation, "name {name:?}");
            assert!(
                error.message().contains("invalid collection name"),
                "unexpected: {error}"
            );
        }
    }
    fake.shutdown().await;
}

#[tokio::test]
async fn document_store_keys_do_not_cross_talk() {
    let fake = FakeRedis::start().await;
    let store = store_for(&fake);
    // A `:` in a collection name must not blur the collection/id boundary.
    store
        .upsert("a:b", vec![Document::new("c").with_field("v", 1)])
        .await
        .unwrap();
    store
        .upsert("a", vec![Document::new("b:c").with_field("v", 2)])
        .await
        .unwrap();
    assert_eq!(
        store.get("a:b", "c").await.unwrap().expect("outer").field("v"),
        Some(&json!(1))
    );
    assert_eq!(
        store
            .get("a", "b:c")
            .await
            .unwrap()
            .expect("inner")
            .field("v"),
        Some(&json!(2))
    );
    assert!(store.get("a:b", "b:c").await.unwrap().is_none());

    // Glob metacharacters in a collection name or an id never leak into a scan.
    store
        .upsert(
            "glob*a",
            vec![Document::new("id?[1]").with_field("v", 3)],
        )
        .await
        .unwrap();
    assert!(store.get("glob*a", "id?[1]").await.unwrap().is_some());
    assert!(store.get("globXa", "id?[1]").await.unwrap().is_none());
    assert!(store.get("glob*a", "idX[1]").await.unwrap().is_none());

    store.drop_collection("glob*a").await.unwrap();
    assert!(store.get("glob*a", "id?[1]").await.unwrap().is_none());
    assert!(store.get("a", "b:c").await.unwrap().is_some());
    fake.shutdown().await;
}

#[tokio::test]
async fn document_store_tolerates_a_dangling_id_and_reports_corruption() {
    let fake = FakeRedis::start().await;
    let store = store_for(&fake);
    store
        .upsert("docs", vec![person("a", "Ada", 36)])
        .await
        .unwrap();

    // A second client plays the role of an interrupted writer: the id set keeps
    // `a` while its value is gone.
    let client = crate::redis::Client::from_url(&fake.url()).await.unwrap();
    client
        .command_str(&["DEL", "openjiuwen:doc:doc:docs:a"])
        .await
        .unwrap();

    // A vanished value is skipped, so the collection simply reads as empty.
    assert_eq!(
        store.query("docs", &Query::default()).await.unwrap(),
        Vec::<Document>::new()
    );
    assert!(store.get("docs", "a").await.unwrap().is_none());

    // A listed id whose payload is corrupt surfaces as a transport error naming
    // its key instead of being silently dropped.
    client
        .command_str(&["SADD", "openjiuwen:doc:ids:docs", "b"])
        .await
        .unwrap();
    client
        .command_str(&["SET", "openjiuwen:doc:doc:docs:b", "not json"])
        .await
        .unwrap();
    let error = store
        .query("docs", &Query::default())
        .await
        .expect_err("corrupt document");
    assert_eq!(error.kind(), ErrorKind::Transport);
    assert!(
        error
            .message()
            .contains("openjiuwen:doc:doc:docs:b"),
        "unexpected: {error}"
    );
    fake.shutdown().await;
}

#[tokio::test]
async fn document_store_connect_validates_pings_and_reports_unreachable_hosts() {
    let fake = FakeRedis::start().await;
    let store = RedisDocumentStore::connect(url_config(&fake.url()))
        .await
        .expect("connect to the fake server");
    store.create_collection("live").await.unwrap();
    assert_eq!(
        store.list_collections().await.unwrap(),
        vec!["live".to_string()]
    );
    assert_eq!(
        RedisDocumentStore::new(url_config(&fake.url()))
            .unwrap()
            .config()
            .url
            .as_deref(),
        Some(fake.url().as_str())
    );
    fake.shutdown().await;

    // An invalid scheme is rejected before any I/O.
    let error = RedisDocumentStore::new(url_config("http://localhost:6379")).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains("redis://"), "unexpected: {error}");

    // A missing url and connection args is the store-level validation error.
    let error = RedisDocumentStore::new(RedisStoreConfig::default()).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains("url"), "unexpected: {error}");

    // An unreachable host fails the eager connect.
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    drop(listener);
    let error = RedisDocumentStore::connect(url_config(&format!("redis://{addr}/0")))
        .await
        .err()
        .expect("a closed port must fail the eager connect");
    assert_eq!(error.kind(), ErrorKind::Transport);

    // A server that rejects `PING` surfaces as a transport error.
    let strict = FakeRedis::start_with(FakeOptions::auth_required(None, "secret")).await;
    let error = RedisDocumentStore::connect(url_config(&strict.url()))
        .await
        .err()
        .expect("an unauthenticated PING must fail");
    assert_eq!(error.kind(), ErrorKind::Transport);
    assert!(
        error.message().to_ascii_uppercase().contains("NOAUTH"),
        "unexpected: {error}"
    );
    strict.shutdown().await;

    // TLS stays deferred.
    let error = RedisDocumentStore::connect(url_config("rediss://localhost:6379"))
        .await
        .err()
        .expect("TLS is not supported");
    assert_eq!(error.kind(), ErrorKind::Unsupported);
}
