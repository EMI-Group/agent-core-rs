// ports tests/unit_tests/core/foundation/store/test_in_memory_vector_store.py
// (adapted to the Rust `VectorStore` contract shared by every backend)
#![allow(clippy::unwrap_used, clippy::expect_used)]

//! The [`crate::store::VectorStore`] contract against every vector backend.
//!
//! Covers schema validation, collection lifecycle, insert-only document adds,
//! deterministic similarity search, `top_k` and filters, deletion counts, the
//! missing-collection rules, and snapshot persistence across a reopen.

use serde_json::{json, Map, Value};
use tempfile::TempDir;

use openjiuwen_common::error::{ErrorKind, Result};

use crate::store::{
    CollectionSchema, FieldSchema, InMemoryVectorStore, LocalFileVectorStore, VectorDataType,
    VectorQuery, VectorSearchResult, VectorStore,
};

#[tokio::test]
async fn in_memory_satisfies_the_vector_contract() {
    assert_vector_contract(&InMemoryVectorStore::new()).await;
}

#[tokio::test]
async fn local_file_satisfies_the_vector_contract() {
    let directory = TempDir::new().unwrap();
    let store = LocalFileVectorStore::new(directory.path()).await.unwrap();
    assert_vector_contract(&store).await;
}

#[tokio::test]
async fn local_file_persists_schemas_and_documents_across_a_reopen() {
    let directory = TempDir::new().unwrap();
    let root = directory.path().join("vector");
    {
        let store = LocalFileVectorStore::new(&root).await.unwrap();
        store.create_collection("docs", schema(2)).await.unwrap();
        store.create_collection("empty", schema(2)).await.unwrap();
        store
            .add_docs(
                "docs",
                vec![vector_doc("a", &[1.0, 0.0]), vector_doc("b", &[0.0, 1.0])],
            )
            .await
            .unwrap();
    }

    let reopened = LocalFileVectorStore::new(&root).await.unwrap();
    assert_eq!(reopened.get_schema("docs").await.unwrap(), Some(schema(2)));
    assert!(reopened.collection_exists("empty").await.unwrap());
    let hits = reopened
        .search("docs", &VectorQuery::new(vec![1.0, 0.0]))
        .await
        .unwrap();
    assert_eq!(hit_ids(&hits), vec!["a", "b"]);
    assert_score(&hits[0], 1.0);
    assert_score(&hits[1], 0.5);
}

/// Verifies the whole [`VectorStore`] contract against one open store.
pub(crate) async fn assert_vector_contract(store: &dyn VectorStore) {
    schema_validation(store).await;
    lifecycle(store).await;
    insert_validation(store).await;
    search_scores(store).await;
    top_k_and_filters(store).await;
    deletion(store).await;
    missing_collection(store).await;
}

/// A two-field schema: a string primary key plus one vector field.
fn schema(dim: usize) -> CollectionSchema {
    CollectionSchema::new(vec![
        FieldSchema {
            is_primary: true,
            ..FieldSchema::new("id", VectorDataType::Varchar)
        },
        FieldSchema {
            dim: Some(dim),
            ..FieldSchema::new("embedding", VectorDataType::FloatVector)
        },
    ])
}

/// [`schema`] plus a scalar `kind` field, used by the filter tests.
fn schema_with_kind(dim: usize) -> CollectionSchema {
    let mut fields = schema(dim).fields;
    fields.push(FieldSchema::new("kind", VectorDataType::Varchar));
    CollectionSchema::new(fields)
}

/// A document with a primary key and an embedding.
fn vector_doc(id: &str, embedding: &[f32]) -> Map<String, Value> {
    let mut doc = Map::new();
    doc.insert("id".to_string(), Value::String(id.to_string()));
    doc.insert("embedding".to_string(), vector_value(embedding));
    doc
}

/// A document with a primary key, an embedding, and a `kind` label.
fn kind_vector_doc(id: &str, embedding: &[f32], kind: &str) -> Map<String, Value> {
    let mut doc = vector_doc(id, embedding);
    doc.insert("kind".to_string(), Value::String(kind.to_string()));
    doc
}

/// The JSON form of an embedding.
fn vector_value(embedding: &[f32]) -> Value {
    Value::Array(embedding.iter().map(|value| Value::from(*value)).collect())
}

/// The primary keys of `hits`, in result order.
fn hit_ids(hits: &[VectorSearchResult]) -> Vec<String> {
    hits.iter()
        .map(|hit| {
            hit.fields
                .get("id")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string()
        })
        .collect()
}

/// Asserts that a hit carries `expected`, and that its score is normalized.
fn assert_score(hit: &VectorSearchResult, expected: f32) {
    assert!(hit.score.is_finite(), "score must be finite: {}", hit.score);
    assert!(
        (0.0..=1.0).contains(&hit.score),
        "score must lie in [0, 1]: {}",
        hit.score
    );
    assert!(
        (hit.score - expected).abs() < 1e-6,
        "expected score {expected}, got {}",
        hit.score
    );
}

/// Asserts that a store operation failed with [`ErrorKind::Validation`].
fn assert_validation<T: std::fmt::Debug>(result: Result<T>, context: &str) {
    let error = result.expect_err(context);
    assert_eq!(error.kind(), ErrorKind::Validation, "{context}");
}

/// Asserts that a store operation failed with [`ErrorKind::NotFound`].
fn assert_not_found<T: std::fmt::Debug>(result: Result<T>, context: &str) {
    let error = result.expect_err(context);
    assert_eq!(error.kind(), ErrorKind::NotFound, "{context}");
}

async fn schema_validation(store: &dyn VectorStore) {
    let invalid = [
        (
            "invalid-no-primary",
            CollectionSchema::new(vec![FieldSchema {
                dim: Some(2),
                ..FieldSchema::new("embedding", VectorDataType::FloatVector)
            }]),
        ),
        (
            "invalid-two-primaries",
            CollectionSchema::new(vec![
                FieldSchema {
                    is_primary: true,
                    ..FieldSchema::new("first", VectorDataType::Varchar)
                },
                FieldSchema {
                    is_primary: true,
                    ..FieldSchema::new("second", VectorDataType::Varchar)
                },
                FieldSchema {
                    dim: Some(2),
                    ..FieldSchema::new("embedding", VectorDataType::FloatVector)
                },
            ]),
        ),
        (
            "invalid-no-vector",
            CollectionSchema::new(vec![FieldSchema {
                is_primary: true,
                ..FieldSchema::new("id", VectorDataType::Varchar)
            }]),
        ),
        (
            "invalid-missing-dim",
            CollectionSchema::new(vec![
                FieldSchema {
                    is_primary: true,
                    ..FieldSchema::new("id", VectorDataType::Varchar)
                },
                FieldSchema::new("embedding", VectorDataType::FloatVector),
            ]),
        ),
        (
            "invalid-zero-dim",
            CollectionSchema::new(vec![
                FieldSchema {
                    is_primary: true,
                    ..FieldSchema::new("id", VectorDataType::Varchar)
                },
                FieldSchema {
                    dim: Some(0),
                    ..FieldSchema::new("embedding", VectorDataType::FloatVector)
                },
            ]),
        ),
        (
            "invalid-duplicate-field",
            CollectionSchema::new(vec![
                FieldSchema {
                    is_primary: true,
                    ..FieldSchema::new("id", VectorDataType::Varchar)
                },
                FieldSchema::new("id", VectorDataType::Varchar),
                FieldSchema {
                    dim: Some(2),
                    ..FieldSchema::new("embedding", VectorDataType::FloatVector)
                },
            ]),
        ),
    ];

    for (name, schema) in invalid {
        assert_validation(store.create_collection(name, schema).await, name);
        assert!(
            !store.collection_exists(name).await.unwrap(),
            "an invalid schema must not create {name:?}"
        );
    }
}

async fn lifecycle(store: &dyn VectorStore) {
    let declared = schema(2);
    store
        .create_collection("lifecycle", declared.clone())
        .await
        .unwrap();
    assert!(store.collection_exists("lifecycle").await.unwrap());
    assert_eq!(
        store.get_schema("lifecycle").await.unwrap(),
        Some(declared.clone())
    );
    assert!(store
        .list_collection_names()
        .await
        .unwrap()
        .contains(&"lifecycle".to_string()));

    assert_validation(
        store.create_collection("lifecycle", declared).await,
        "duplicate collection name",
    );

    store.delete_collection("lifecycle").await.unwrap();
    assert!(!store.collection_exists("lifecycle").await.unwrap());
    assert_eq!(store.get_schema("lifecycle").await.unwrap(), None);
    // Deleting a missing collection is a no-op.
    store.delete_collection("lifecycle").await.unwrap();
    assert!(!store
        .list_collection_names()
        .await
        .unwrap()
        .contains(&"lifecycle".to_string()));
}

async fn insert_validation(store: &dyn VectorStore) {
    store.create_collection("inserts", schema(2)).await.unwrap();

    assert_eq!(
        store
            .add_docs(
                "inserts",
                vec![vector_doc("a", &[1.0, 0.0]), vector_doc("b", &[0.0, 1.0])]
            )
            .await
            .unwrap(),
        2
    );

    // A repeated primary key inside one batch fails and inserts nothing.
    assert_validation(
        store
            .add_docs(
                "inserts",
                vec![
                    vector_doc("dup", &[1.0, 0.0]),
                    vector_doc("dup", &[0.0, 1.0]),
                ],
            )
            .await,
        "duplicate primary key in a batch",
    );
    // Proof that the failed batch left the collection unchanged.
    assert_eq!(
        store
            .add_docs("inserts", vec![vector_doc("dup", &[1.0, 0.0])])
            .await
            .unwrap(),
        1
    );

    // A primary key already stored fails too.
    assert_validation(
        store
            .add_docs("inserts", vec![vector_doc("dup", &[0.0, 1.0])])
            .await,
        "duplicate primary key across add_docs calls",
    );

    // Wrong dimension and a missing vector field are rejected.
    assert_validation(
        store
            .add_docs("inserts", vec![vector_doc("wide", &[1.0, 0.0, 0.0])])
            .await,
        "document vector dimension",
    );
    let mut missing = Map::new();
    missing.insert("id".to_string(), json!("no-vector"));
    assert_validation(
        store.add_docs("inserts", vec![missing]).await,
        "missing vector field",
    );

    // The query vector is validated against the schema dimension too.
    assert_validation(
        store.search("inserts", &VectorQuery::new(vec![1.0])).await,
        "query vector dimension",
    );
}

async fn search_scores(store: &dyn VectorStore) {
    store.create_collection("scores", schema(2)).await.unwrap();
    store
        .add_docs(
            "scores",
            vec![
                vector_doc("a", &[1.0, 0.0]),
                vector_doc("b", &[0.0, 1.0]),
                vector_doc("c", &[-1.0, 0.0]),
            ],
        )
        .await
        .unwrap();

    let hits = store
        .search("scores", &VectorQuery::new(vec![1.0, 0.0]))
        .await
        .unwrap();
    assert_eq!(hit_ids(&hits), vec!["a", "b", "c"]);
    assert_score(&hits[0], 1.0);
    assert_score(&hits[1], 0.5);
    assert_score(&hits[2], 0.0);

    // Equal scores break ties by primary key, so the order is deterministic.
    store
        .add_docs(
            "scores",
            vec![
                vector_doc("tie-b", &[1.0, 0.0]),
                vector_doc("tie-a", &[1.0, 0.0]),
            ],
        )
        .await
        .unwrap();
    let tied = store
        .search("scores", &VectorQuery::new(vec![1.0, 0.0]).with_top_k(2))
        .await
        .unwrap();
    assert_eq!(hit_ids(&tied), vec!["a", "tie-a"]);
}

async fn top_k_and_filters(store: &dyn VectorStore) {
    store
        .create_collection("topk", schema_with_kind(2))
        .await
        .unwrap();
    let documents: Vec<Map<String, Value>> = (0..7)
        .map(|index| {
            let kind = if index % 2 == 0 { "even" } else { "odd" };
            kind_vector_doc(&format!("i{index}"), &[1.0, 0.0], kind)
        })
        .collect();
    assert_eq!(store.add_docs("topk", documents).await.unwrap(), 7);

    // `VectorQuery`'s default top_k is 5.
    let default_hits = store
        .search("topk", &VectorQuery::new(vec![1.0, 0.0]))
        .await
        .unwrap();
    assert_eq!(hit_ids(&default_hits), vec!["i0", "i1", "i2", "i3", "i4"]);

    let limited = store
        .search("topk", &VectorQuery::new(vec![1.0, 0.0]).with_top_k(2))
        .await
        .unwrap();
    assert_eq!(hit_ids(&limited), vec!["i0", "i1"]);

    let evens = store
        .search(
            "topk",
            &VectorQuery::new(vec![1.0, 0.0]).with_filters(json!({"kind": "even"})),
        )
        .await
        .unwrap();
    assert_eq!(hit_ids(&evens), vec!["i0", "i2", "i4", "i6"]);
    assert!(evens
        .iter()
        .all(|hit| hit.fields.get("kind") == Some(&json!("even"))));

    let unmatched = store
        .search(
            "topk",
            &VectorQuery::new(vec![1.0, 0.0]).with_filters(json!({"kind": "absent"})),
        )
        .await
        .unwrap();
    assert_eq!(unmatched.len(), 0);
}

async fn deletion(store: &dyn VectorStore) {
    store
        .create_collection("deletes", schema_with_kind(2))
        .await
        .unwrap();
    store
        .add_docs(
            "deletes",
            vec![
                kind_vector_doc("a", &[1.0, 0.0], "keep"),
                kind_vector_doc("b", &[0.0, 1.0], "drop"),
                kind_vector_doc("c", &[-1.0, 0.0], "drop"),
            ],
        )
        .await
        .unwrap();

    // A repeated id is removed once, so it counts once; unknown ids are ignored.
    let ids = vec!["a".to_string(), "a".to_string(), "missing".to_string()];
    assert_eq!(store.delete_docs_by_ids("deletes", &ids).await.unwrap(), 1);
    assert_eq!(
        store
            .delete_docs_by_ids("deletes", &["missing".to_string()])
            .await
            .unwrap(),
        0
    );
    assert_eq!(
        store
            .delete_docs_by_filters("deletes", &json!({"kind": "drop"}))
            .await
            .unwrap(),
        2
    );
    assert_eq!(
        store
            .search("deletes", &VectorQuery::new(vec![1.0, 0.0]))
            .await
            .unwrap()
            .len(),
        0
    );

    // A `null` filter matches every remaining document.
    store
        .add_docs("deletes", vec![vector_doc("only", &[1.0, 0.0])])
        .await
        .unwrap();
    assert_eq!(
        store
            .delete_docs_by_filters("deletes", &Value::Null)
            .await
            .unwrap(),
        1
    );
}

async fn missing_collection(store: &dyn VectorStore) {
    assert!(!store.collection_exists("absent").await.unwrap());
    assert_eq!(store.get_schema("absent").await.unwrap(), None);
    store.delete_collection("absent").await.unwrap();

    assert_not_found(
        store
            .add_docs("absent", vec![vector_doc("a", &[1.0, 0.0])])
            .await,
        "add_docs",
    );
    assert_not_found(
        store
            .search("absent", &VectorQuery::new(vec![1.0, 0.0]))
            .await,
        "search",
    );
    assert_not_found(
        store.delete_docs_by_ids("absent", &["a".to_string()]).await,
        "delete_docs_by_ids",
    );
    assert_not_found(
        store.delete_docs_by_filters("absent", &Value::Null).await,
        "delete_docs_by_filters",
    );
    assert!(!store
        .list_collection_names()
        .await
        .unwrap()
        .contains(&"absent".to_string()));
}
