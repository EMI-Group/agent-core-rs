// ports tests/unit_tests/core/foundation/store/test_in_memory_db_store.py
// (adapted to the Rust `DocumentStore` contract shared by every backend)
#![allow(clippy::unwrap_used, clippy::expect_used)]

//! The [`crate::store::DocumentStore`] contract against every document backend.
//!
//! Covers collection lifecycle, upsert/replace, the shared filter language,
//! ordering and paging, counts and deletes, and the missing-collection rules.
//! File persistence is verified separately because only the local-filesystem
//! backend can be reopened.

use serde_json::{json, Value};
use tempfile::TempDir;

use crate::store::{
    Document, DocumentStore, InMemoryDocumentStore, LocalFileDocumentStore, Query, SortOrder,
};

#[tokio::test]
async fn in_memory_satisfies_the_document_contract() {
    assert_document_contract(&InMemoryDocumentStore::new()).await;
}

#[tokio::test]
async fn local_file_satisfies_the_document_contract() {
    let directory = TempDir::new().unwrap();
    let store = LocalFileDocumentStore::new(directory.path()).await.unwrap();
    assert_document_contract(&store).await;
}

#[tokio::test]
async fn local_file_persists_documents_and_empty_collections_across_a_reopen() {
    let directory = TempDir::new().unwrap();
    let root = directory.path().join("documents");
    {
        let store = LocalFileDocumentStore::new(&root).await.unwrap();
        store.create_collection("empty").await.unwrap();
        store
            .upsert(
                "docs",
                vec![
                    Document::new("a").with_field("name", "Ada"),
                    Document::new("b")
                        .with_field("name", "Bob")
                        .with_field("meta", json!({"owner": "bob"})),
                ],
            )
            .await
            .unwrap();
    }

    let reopened = LocalFileDocumentStore::new(&root).await.unwrap();
    assert_eq!(
        reopened.list_collections().await.unwrap(),
        vec!["docs".to_string(), "empty".to_string()]
    );
    let ada = reopened.get("docs", "a").await.unwrap().unwrap();
    assert_eq!(ada.field("name"), Some(&json!("Ada")));
    let owner = reopened
        .get("docs", "b")
        .await
        .unwrap()
        .unwrap()
        .field("meta")
        .cloned()
        .unwrap();
    assert_eq!(owner, json!({"owner": "bob"}));
    assert_eq!(reopened.count("docs", &Query::default()).await.unwrap(), 2);
}

/// A document value round-trips through its flat JSON view.
#[test]
fn document_value_round_trips() {
    let document = Document::new("a")
        .with_field("age", 36)
        .with_field("nested", json!({"list": [1, 2, 3]}))
        .with_field("nil", Value::Null);
    assert_eq!(
        document.to_value(),
        json!({"id": "a", "age": 36, "nested": {"list": [1, 2, 3]}, "nil": null})
    );
    assert_eq!(Document::from_value(document.to_value()).unwrap(), document);
}

/// Verifies the whole [`DocumentStore`] contract against one open store.
pub(crate) async fn assert_document_contract(store: &dyn DocumentStore) {
    missing_collections(store).await;
    collection_lifecycle(store).await;
    upsert_get_and_replace(store).await;
    filters(store).await;
    ordering_and_paging(store).await;
    counting_and_deleting(store).await;
}

/// The fixture written by [`seed`]: ages 36, 24, 36 with an owner per document.
fn sample() -> Vec<Document> {
    vec![
        Document::new("a")
            .with_field("name", "Ada")
            .with_field("age", 36)
            .with_field("meta", json!({"owner": "ada"}))
            .with_field("tags", json!(["rust", "store"])),
        Document::new("b")
            .with_field("name", "Bob")
            .with_field("age", 24)
            .with_field("meta", json!({"owner": "bob"}))
            .with_field("tags", json!(["rust"])),
        Document::new("c")
            .with_field("name", "Cid")
            .with_field("age", 36)
            .with_field("meta", json!({"owner": "ada"}))
            .with_field("tags", json!([])),
    ]
}

/// Writes [`sample`] into `collection` (which must be empty) and returns it.
async fn seed(store: &dyn DocumentStore, collection: &str) -> Vec<Document> {
    let documents = sample();
    let written = store.upsert(collection, documents.clone()).await.unwrap();
    assert_eq!(written, 3);
    documents
}

/// A query with only a filter.
fn filtered(filter: Value) -> Query {
    Query {
        filter,
        ..Query::default()
    }
}

/// The ids of `documents`, in order.
fn ids(documents: &[Document]) -> Vec<&str> {
    documents
        .iter()
        .map(|document| document.id.as_str())
        .collect()
}

async fn missing_collections(store: &dyn DocumentStore) {
    let query = Query::default();
    assert!(store.get("absent", "a").await.unwrap().is_none());
    assert_eq!(
        store.query("absent", &query).await.unwrap(),
        Vec::<Document>::new()
    );
    assert_eq!(store.count("absent", &query).await.unwrap(), 0);
    assert_eq!(store.delete("absent", &query).await.unwrap(), 0);
    assert!(!store
        .list_collections()
        .await
        .unwrap()
        .contains(&"absent".to_string()));
}

async fn collection_lifecycle(store: &dyn DocumentStore) {
    store.create_collection("lifecycle").await.unwrap();
    store.create_collection("lifecycle").await.unwrap();

    let names = store.list_collections().await.unwrap();
    assert!(names.contains(&"lifecycle".to_string()));
    assert!(
        names.windows(2).all(|pair| pair[0] < pair[1]),
        "list_collections is sorted: {names:?}"
    );

    store.drop_collection("lifecycle").await.unwrap();
    store.drop_collection("lifecycle").await.unwrap();
    assert!(!store
        .list_collections()
        .await
        .unwrap()
        .contains(&"lifecycle".to_string()));
}

async fn upsert_get_and_replace(store: &dyn DocumentStore) {
    seed(store, "upsert").await;

    let mut names = store.list_collections().await.unwrap();
    names.retain(|name| name == "upsert");
    assert_eq!(names, vec!["upsert".to_string()], "upsert auto-creates");

    let bob = store.get("upsert", "b").await.unwrap().unwrap();
    assert_eq!(bob.field("name"), Some(&json!("Bob")));
    assert_eq!(bob.field("age"), Some(&json!(24)));
    assert!(store.get("upsert", "missing").await.unwrap().is_none());

    // Upsert by id replaces the whole document, it does not merge fields.
    store
        .upsert(
            "upsert",
            vec![Document::new("b").with_field("name", "Robert")],
        )
        .await
        .unwrap();
    let replaced = store.get("upsert", "b").await.unwrap().unwrap();
    assert_eq!(replaced.field("name"), Some(&json!("Robert")));
    assert_eq!(replaced.field("age"), None);
    assert_eq!(store.count("upsert", &Query::default()).await.unwrap(), 3);
}

async fn filters(store: &dyn DocumentStore) {
    seed(store, "filters").await;

    let equality = store
        .query("filters", &filtered(json!({"name": "Bob"})))
        .await
        .unwrap();
    assert_eq!(ids(&equality), vec!["b"]);

    let greater = store
        .query("filters", &filtered(json!({"age": {"$gt": 24}})))
        .await
        .unwrap();
    assert_eq!(ids(&greater), vec!["a", "c"]);

    let members = store
        .query("filters", &filtered(json!({"age": {"$in": [24, 99]}})))
        .await
        .unwrap();
    assert_eq!(ids(&members), vec!["b"]);

    let contains = store
        .query(
            "filters",
            &filtered(json!({"tags": {"$contains": ["rust"]}})),
        )
        .await
        .unwrap();
    assert_eq!(ids(&contains), vec!["a", "b"]);

    let dotted = store
        .query("filters", &filtered(json!({"meta.owner": "ada"})))
        .await
        .unwrap();
    assert_eq!(ids(&dotted), vec!["a", "c"]);

    // The document id is addressable like any other field.
    let by_id = store
        .query("filters", &filtered(json!({"id": "c"})))
        .await
        .unwrap();
    assert_eq!(ids(&by_id), vec!["c"]);

    let none = store
        .query("filters", &filtered(json!({"age": {"$gt": 1000}})))
        .await
        .unwrap();
    assert_eq!(none.len(), 0);
}

async fn ordering_and_paging(store: &dyn DocumentStore) {
    seed(store, "ordering").await;

    let by_id = store
        .query(
            "ordering",
            &Query {
                order_by: vec![("id".to_string(), SortOrder::Asc)],
                ..Query::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(ids(&by_id), vec!["a", "b", "c"]);

    let ascending = store
        .query(
            "ordering",
            &Query {
                order_by: vec![
                    ("age".to_string(), SortOrder::Asc),
                    ("id".to_string(), SortOrder::Asc),
                ],
                ..Query::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(ids(&ascending), vec!["b", "a", "c"]);

    let descending = store
        .query(
            "ordering",
            &Query {
                order_by: vec![
                    ("age".to_string(), SortOrder::Desc),
                    ("id".to_string(), SortOrder::Asc),
                ],
                ..Query::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(ids(&descending), vec!["a", "c", "b"]);

    // Every criterion participates, in priority order.
    let by_age_then_name = store
        .query(
            "ordering",
            &Query {
                order_by: vec![
                    ("age".to_string(), SortOrder::Desc),
                    ("name".to_string(), SortOrder::Desc),
                ],
                ..Query::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(ids(&by_age_then_name), vec!["c", "a", "b"]);

    // Offset is applied before the limit.
    let paged = store
        .query(
            "ordering",
            &Query {
                offset: 1,
                limit: Some(1),
                order_by: vec![("id".to_string(), SortOrder::Asc)],
                ..Query::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(ids(&paged), vec!["b"]);

    let past_the_end = store
        .query(
            "ordering",
            &Query {
                offset: 10,
                ..Query::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(past_the_end.len(), 0);
}

async fn counting_and_deleting(store: &dyn DocumentStore) {
    seed(store, "counting").await;

    assert_eq!(
        store
            .count("counting", &filtered(json!({"age": 36})))
            .await
            .unwrap(),
        2
    );

    // `count` uses only the filter: paging and ordering are ignored.
    let paged = Query {
        filter: json!({"age": {"$gte": 30}}),
        limit: Some(1),
        offset: 5,
        order_by: vec![("age".to_string(), SortOrder::Desc)],
    };
    assert_eq!(store.count("counting", &paged).await.unwrap(), 2);

    // `delete` likewise ignores paging and ordering.
    assert_eq!(store.delete("counting", &paged).await.unwrap(), 2);
    assert_eq!(store.delete("counting", &paged).await.unwrap(), 0);
    assert_eq!(store.count("counting", &Query::default()).await.unwrap(), 1);
    assert!(store.get("counting", "a").await.unwrap().is_none());
    assert!(store.get("counting", "b").await.unwrap().is_some());

    assert_eq!(
        store.delete("counting", &Query::default()).await.unwrap(),
        1
    );
    assert_eq!(store.count("counting", &Query::default()).await.unwrap(), 0);
}
