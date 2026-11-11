// ports tests/unit_tests/core/foundation/store/test_in_memory_graph_store.py and
// tests/unit_tests/core/foundation/store/test_in_memory_object_store.py
// (adapted to the Rust `GraphStore` / `ObjectStore` contracts)
#![allow(clippy::unwrap_used, clippy::expect_used)]

//! The [`crate::store::GraphStore`] and [`crate::store::ObjectStore`] contracts
//! against every backend.
//!
//! Graph coverage: upserts and replacement by id, lookup, label and property
//! queries, directional neighbors, cascade deletion, counts, and persistence.
//! Object coverage: bucket lifecycle, put/get/exists/delete, prefix listing with
//! a limit, failed operations on a missing bucket, bucket deletion, persistence,
//! and (local backend only) rejection of keys that could escape the root.

use bytes::Bytes;
use serde_json::{json, Map, Value};
use tempfile::TempDir;

use openjiuwen_common::error::{ErrorKind, Result};

use crate::store::{
    Direction, GraphEdge, GraphNode, GraphQuery, GraphStore, InMemoryGraphStore,
    InMemoryObjectStore, LocalFileGraphStore, LocalObjectStore, ObjectMeta, ObjectStore,
};

/// A payload that is not valid UTF-8, to prove bytes survive a round-trip.
const BINARY: [u8; 4] = [0x00, 0xff, 0xfe, 0x80];

#[tokio::test]
async fn in_memory_satisfies_the_graph_contract() {
    assert_graph_contract(&InMemoryGraphStore::new()).await;
}

#[tokio::test]
async fn local_file_satisfies_the_graph_contract() {
    let directory = TempDir::new().unwrap();
    let store = LocalFileGraphStore::new(directory.path()).await.unwrap();
    assert_graph_contract(&store).await;
}

#[tokio::test]
async fn in_memory_satisfies_the_object_contract() {
    assert_object_contract(&InMemoryObjectStore::new()).await;
}

#[tokio::test]
async fn local_object_satisfies_the_object_contract() {
    let directory = TempDir::new().unwrap();
    let store = LocalObjectStore::new(directory.path()).await.unwrap();
    assert_object_contract(&store).await;
}

#[tokio::test]
async fn local_file_graph_persists_across_a_reopen() {
    let directory = TempDir::new().unwrap();
    let root = directory.path().join("graph");
    {
        let store = LocalFileGraphStore::new(&root).await.unwrap();
        store
            .upsert_nodes(vec![
                node("a", &["Person"], json!({"age": 30})),
                node("b", &["Person"], json!({})),
            ])
            .await
            .unwrap();
        store
            .upsert_edges(vec![edge("e1", "a", "b", "KNOWS", json!({"since": 2020}))])
            .await
            .unwrap();
    }

    let reopened = LocalFileGraphStore::new(&root).await.unwrap();
    assert_eq!(reopened.len().await.unwrap(), (2, 1));
    let stored = reopened.get_node("a").await.unwrap().unwrap();
    assert_eq!(stored.properties.get("age"), Some(&json!(30)));
    let stored_edge = reopened.get_edge("e1").await.unwrap().unwrap();
    assert_eq!(
        (stored_edge.from.as_str(), stored_edge.to.as_str()),
        ("a", "b")
    );
    assert_eq!(stored_edge.label, "KNOWS");
    let adjacent = reopened
        .neighbors("a", Direction::Outgoing, None)
        .await
        .unwrap();
    assert_eq!(node_ids(&adjacent), vec!["b"]);
}

#[tokio::test]
async fn local_object_persists_nested_keys_across_a_reopen() {
    let directory = TempDir::new().unwrap();
    let root = directory.path().join("objects");
    {
        let store = LocalObjectStore::new(&root).await.unwrap();
        store.create_bucket("bucket").await.unwrap();
        store
            .put("bucket", "nested/deep.bin", Bytes::from(BINARY.to_vec()))
            .await
            .unwrap();
        store
            .put("bucket", "top.txt", Bytes::from_static(b"top"))
            .await
            .unwrap();
    }

    let reopened = LocalObjectStore::new(&root).await.unwrap();
    assert_eq!(reopened.list_buckets().await.unwrap(), vec!["bucket"]);
    assert_eq!(
        reopened.get("bucket", "nested/deep.bin").await.unwrap(),
        Some(Bytes::from(BINARY.to_vec()))
    );
    let keys: Vec<String> = reopened
        .list("bucket", "", None)
        .await
        .unwrap()
        .into_iter()
        .map(|meta| meta.key)
        .collect();
    assert_eq!(keys, vec!["nested/deep.bin", "top.txt"]);
}

#[tokio::test]
async fn local_object_rejects_keys_that_escape_the_root() {
    let directory = TempDir::new().unwrap();
    let store = LocalObjectStore::new(directory.path()).await.unwrap();
    store.create_bucket("bucket").await.unwrap();

    // The in-memory backend accepts arbitrary keys; the local backend must not
    // turn an escaping key into a path outside its root.
    for key in [
        "",
        "/absolute.txt",
        "..",
        "../escape.txt",
        "a/../b",
        "a//b",
        "a\\b",
    ] {
        assert_validation(
            store.put("bucket", key, Bytes::from_static(b"x")).await,
            key,
        );
        assert_validation(store.get("bucket", key).await, key);
        assert_validation(store.exists("bucket", key).await, key);
    }
    assert_eq!(store.list("bucket", "", None).await.unwrap().len(), 0);

    for bucket in ["", ".", "..", "a/b", "a\\b"] {
        assert_validation(store.create_bucket(bucket).await, bucket);
    }
    assert_eq!(store.list_buckets().await.unwrap(), vec!["bucket"]);
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

/// A node built from JSON properties.
fn node(id: &str, labels: &[&str], properties: Value) -> GraphNode {
    GraphNode {
        id: id.to_string(),
        labels: labels.iter().map(|label| (*label).to_string()).collect(),
        properties: object(properties),
    }
}

/// An edge built from JSON properties.
fn edge(id: &str, from: &str, to: &str, label: &str, properties: Value) -> GraphEdge {
    GraphEdge {
        id: id.to_string(),
        from: from.to_string(),
        to: to.to_string(),
        label: label.to_string(),
        properties: object(properties),
    }
}

/// The object form of a JSON properties value.
fn object(value: Value) -> Map<String, Value> {
    value
        .as_object()
        .cloned()
        .expect("properties are an object")
}

/// The ids of `nodes`, in order.
fn node_ids(nodes: &[GraphNode]) -> Vec<String> {
    nodes.iter().map(|node| node.id.clone()).collect()
}

/// The keys of `metas`, in order.
fn meta_keys(metas: &[ObjectMeta]) -> Vec<String> {
    metas.iter().map(|meta| meta.key.clone()).collect()
}

/// Verifies the whole [`GraphStore`] contract against one open store.
pub(crate) async fn assert_graph_contract(store: &dyn GraphStore) {
    node_upsert_and_replace(store).await;
    edge_upsert_and_replace(store).await;
    query_labels_any(store).await;
    query_filters_and_limits(store).await;
    neighbors_directions(store).await;
    neighbors_label_filter(store).await;
    node_deletion_cascades(store).await;
    missing_ids(store).await;
}

async fn node_upsert_and_replace(store: &dyn GraphStore) {
    assert_eq!(
        store
            .upsert_nodes(vec![node("a", &["Person"], json!({"age": 30}))])
            .await
            .unwrap(),
        1
    );
    let stored = store.get_node("a").await.unwrap().unwrap();
    assert_eq!(stored.labels, vec!["Person".to_string()]);
    assert_eq!(stored.properties.get("age"), Some(&json!(30)));

    // Same id: labels and properties are replaced, not merged.
    assert_eq!(
        store
            .upsert_nodes(vec![node("a", &["Person", "Employee"], json!({"age": 31}))])
            .await
            .unwrap(),
        1
    );
    let replaced = store.get_node("a").await.unwrap().unwrap();
    assert_eq!(replaced.labels.len(), 2);
    assert_eq!(replaced.properties.get("age"), Some(&json!(31)));
    assert_eq!(store.len().await.unwrap(), (1, 0));
    assert!(store.get_node("missing").await.unwrap().is_none());

    assert_eq!(store.delete_nodes(&["a".to_string()]).await.unwrap(), 1);
    assert_eq!(store.len().await.unwrap(), (0, 0));
}

async fn edge_upsert_and_replace(store: &dyn GraphStore) {
    store
        .upsert_nodes(vec![
            node("a", &["Person"], json!({})),
            node("b", &["Person"], json!({})),
        ])
        .await
        .unwrap();
    assert_eq!(
        store
            .upsert_edges(vec![edge("e1", "a", "b", "KNOWS", json!({"since": 2020}))])
            .await
            .unwrap(),
        1
    );
    let stored = store.get_edge("e1").await.unwrap().unwrap();
    assert_eq!(stored.from, "a");
    assert_eq!(stored.to, "b");
    assert_eq!(stored.label, "KNOWS");
    assert_eq!(stored.properties.get("since"), Some(&json!(2020)));

    assert_eq!(
        store
            .upsert_edges(vec![edge("e1", "b", "a", "LIKES", json!({"since": 2021}))])
            .await
            .unwrap(),
        1
    );
    let replaced = store.get_edge("e1").await.unwrap().unwrap();
    assert_eq!((replaced.from.as_str(), replaced.to.as_str()), ("b", "a"));
    assert_eq!(replaced.label, "LIKES");
    assert_eq!(store.len().await.unwrap(), (2, 1));
    assert!(store.get_edge("missing").await.unwrap().is_none());

    assert_eq!(
        store
            .delete_nodes(&["a".to_string(), "b".to_string()])
            .await
            .unwrap(),
        2
    );
    assert_eq!(store.len().await.unwrap(), (0, 0));
}

async fn query_labels_any(store: &dyn GraphStore) {
    store
        .upsert_nodes(vec![
            node("a", &["Person"], json!({})),
            node("b", &["Employee", "Person"], json!({})),
            node("c", &["Document"], json!({})),
        ])
        .await
        .unwrap();

    let people = store
        .query_nodes(&GraphQuery {
            labels: vec!["Person".to_string()],
            ..GraphQuery::default()
        })
        .await
        .unwrap();
    assert_eq!(node_ids(&people), vec!["a", "b"]);

    let either = store
        .query_nodes(&GraphQuery {
            labels: vec!["Employee".to_string(), "Document".to_string()],
            ..GraphQuery::default()
        })
        .await
        .unwrap();
    assert_eq!(node_ids(&either), vec!["b", "c"]);

    let all = store.query_nodes(&GraphQuery::default()).await.unwrap();
    assert_eq!(node_ids(&all), vec!["a", "b", "c"]);

    let none = store
        .query_nodes(&GraphQuery {
            labels: vec!["Unknown".to_string()],
            ..GraphQuery::default()
        })
        .await
        .unwrap();
    assert_eq!(none.len(), 0);

    assert_eq!(
        store
            .delete_nodes(&["a".to_string(), "b".to_string(), "c".to_string()])
            .await
            .unwrap(),
        3
    );
}

async fn query_filters_and_limits(store: &dyn GraphStore) {
    store
        .upsert_nodes(vec![
            node(
                "c",
                &["Person"],
                json!({"age": 60, "city": "Berlin", "meta": {"score": 9}}),
            ),
            node(
                "a",
                &["Person"],
                json!({"age": 30, "city": "Berlin", "meta": {"score": 7}}),
            ),
            node(
                "b",
                &["Person"],
                json!({"age": 45, "city": "Paris", "meta": {"score": 3}}),
            ),
        ])
        .await
        .unwrap();

    let older = store
        .query_nodes(&GraphQuery {
            filter: json!({"age": {"$gt": 40}}),
            ..GraphQuery::default()
        })
        .await
        .unwrap();
    assert_eq!(node_ids(&older), vec!["b", "c"]);

    let berlin_young = store
        .query_nodes(&GraphQuery {
            filter: json!({"city": {"$in": ["Berlin"]}, "age": {"$lte": 30}}),
            ..GraphQuery::default()
        })
        .await
        .unwrap();
    assert_eq!(node_ids(&berlin_young), vec!["a"]);

    let dotted = store
        .query_nodes(&GraphQuery {
            filter: json!({"meta.score": {"$gte": 7}}),
            ..GraphQuery::default()
        })
        .await
        .unwrap();
    assert_eq!(node_ids(&dotted), vec!["a", "c"]);

    let by_id = store
        .query_nodes(&GraphQuery {
            filter: json!({"id": {"$in": ["b"]}}),
            ..GraphQuery::default()
        })
        .await
        .unwrap();
    assert_eq!(node_ids(&by_id), vec!["b"]);

    let by_label = store
        .query_nodes(&GraphQuery {
            filter: json!({"labels": {"$contains": ["Person"]}}),
            ..GraphQuery::default()
        })
        .await
        .unwrap();
    assert_eq!(node_ids(&by_label), vec!["a", "b", "c"]);

    // Results are always sorted by id, and the limit applies after sorting.
    let limited = store
        .query_nodes(&GraphQuery {
            limit: Some(2),
            ..GraphQuery::default()
        })
        .await
        .unwrap();
    assert_eq!(node_ids(&limited), vec!["a", "b"]);
    let empty = store
        .query_nodes(&GraphQuery {
            limit: Some(0),
            ..GraphQuery::default()
        })
        .await
        .unwrap();
    assert_eq!(empty.len(), 0);

    assert_eq!(
        store
            .delete_nodes(&["a".to_string(), "b".to_string(), "c".to_string()])
            .await
            .unwrap(),
        3
    );
}

/// The fixture of the traversal tests: `a` reaches `b` and `d`, `c` reaches `a`.
async fn seed_traversal(store: &dyn GraphStore) {
    store
        .upsert_nodes(vec![
            node("a", &["N"], json!({})),
            node("b", &["N"], json!({})),
            node("c", &["N"], json!({})),
            node("d", &["N"], json!({})),
        ])
        .await
        .unwrap();
    store
        .upsert_edges(vec![
            edge("e1", "a", "b", "KNOWS", json!({})),
            edge("e2", "c", "a", "MENTIONS", json!({})),
            edge("e3", "a", "d", "KNOWS", json!({})),
            // A self-loop must yield the node once, not twice.
            edge("e4", "a", "a", "SELF", json!({})),
        ])
        .await
        .unwrap();
}

async fn neighbors_directions(store: &dyn GraphStore) {
    seed_traversal(store).await;

    let outgoing = store
        .neighbors("a", Direction::Outgoing, None)
        .await
        .unwrap();
    assert_eq!(node_ids(&outgoing), vec!["a", "b", "d"]);
    let incoming = store
        .neighbors("a", Direction::Incoming, None)
        .await
        .unwrap();
    assert_eq!(node_ids(&incoming), vec!["a", "c"]);
    let both = store.neighbors("a", Direction::Both, None).await.unwrap();
    assert_eq!(node_ids(&both), vec!["a", "b", "c", "d"]);

    // Traversal is directional.
    assert_eq!(
        node_ids(
            &store
                .neighbors("b", Direction::Incoming, None)
                .await
                .unwrap()
        ),
        vec!["a"]
    );
    assert_eq!(
        store
            .neighbors("b", Direction::Outgoing, None)
            .await
            .unwrap()
            .len(),
        0
    );
    // A missing node has no neighbors.
    assert_eq!(
        store
            .neighbors("missing", Direction::Both, None)
            .await
            .unwrap()
            .len(),
        0
    );

    assert_eq!(
        store
            .delete_nodes(&[
                "a".to_string(),
                "b".to_string(),
                "c".to_string(),
                "d".to_string()
            ])
            .await
            .unwrap(),
        4
    );
    assert_eq!(store.len().await.unwrap(), (0, 0));
}

async fn neighbors_label_filter(store: &dyn GraphStore) {
    store
        .upsert_nodes(vec![
            node("a", &["N"], json!({})),
            node("b", &["N"], json!({})),
            node("c", &["N"], json!({})),
        ])
        .await
        .unwrap();
    store
        .upsert_edges(vec![
            edge("e1", "a", "b", "KNOWS", json!({})),
            edge("e2", "a", "c", "MENTIONS", json!({})),
        ])
        .await
        .unwrap();

    let knows = vec!["KNOWS".to_string()];
    assert_eq!(
        node_ids(
            &store
                .neighbors("a", Direction::Outgoing, Some(&knows))
                .await
                .unwrap()
        ),
        vec!["b"]
    );
    let both_labels = vec!["KNOWS".to_string(), "MENTIONS".to_string()];
    assert_eq!(
        node_ids(
            &store
                .neighbors("a", Direction::Outgoing, Some(&both_labels))
                .await
                .unwrap()
        ),
        vec!["b", "c"]
    );
    // An empty label list matches no edge at all.
    assert_eq!(
        store
            .neighbors("a", Direction::Outgoing, Some(&[]))
            .await
            .unwrap()
            .len(),
        0
    );

    assert_eq!(
        store
            .delete_nodes(&["a".to_string(), "b".to_string(), "c".to_string()])
            .await
            .unwrap(),
        3
    );
}

async fn node_deletion_cascades(store: &dyn GraphStore) {
    store
        .upsert_nodes(vec![
            node("a", &["N"], json!({})),
            node("b", &["N"], json!({})),
        ])
        .await
        .unwrap();
    store
        .upsert_edges(vec![
            edge("e1", "a", "b", "KNOWS", json!({})),
            // Dangling edge: `c` has no node.
            edge("e2", "b", "c", "MENTIONS", json!({})),
        ])
        .await
        .unwrap();

    // A missing id is not counted, but its incident edges are still cleaned up.
    assert_eq!(
        store
            .delete_nodes(&["a".to_string(), "missing".to_string()])
            .await
            .unwrap(),
        1
    );
    assert_eq!(store.len().await.unwrap(), (1, 1));
    assert!(store.get_edge("e1").await.unwrap().is_none());
    assert!(store.get_edge("e2").await.unwrap().is_some());

    // Repeated edge ids are deleted once, so they count once.
    let doomed = vec!["e2".to_string(), "e2".to_string(), "missing".to_string()];
    assert_eq!(store.delete_edges(&doomed).await.unwrap(), 1);
    assert_eq!(store.len().await.unwrap(), (1, 0));

    // Deleting an id with no node still removes the edges dangling off it.
    assert_eq!(store.delete_nodes(&["c".to_string()]).await.unwrap(), 0);
    assert_eq!(store.delete_nodes(&["b".to_string()]).await.unwrap(), 1);
    assert_eq!(store.len().await.unwrap(), (0, 0));
}

async fn missing_ids(store: &dyn GraphStore) {
    assert_eq!(store.len().await.unwrap(), (0, 0));
    assert!(store.get_node("nope").await.unwrap().is_none());
    assert!(store.get_edge("nope").await.unwrap().is_none());
    assert_eq!(store.delete_nodes(&["nope".to_string()]).await.unwrap(), 0);
    assert_eq!(store.delete_edges(&["nope".to_string()]).await.unwrap(), 0);
    assert_eq!(
        store
            .query_nodes(&GraphQuery::default())
            .await
            .unwrap()
            .len(),
        0
    );
}

/// Verifies the whole [`ObjectStore`] contract against one open store.
pub(crate) async fn assert_object_contract(store: &dyn ObjectStore) {
    bucket_lifecycle(store).await;
    object_round_trip(store).await;
    prefix_listing(store).await;
    missing_bucket(store).await;
    bucket_deletion(store).await;
}

async fn bucket_lifecycle(store: &dyn ObjectStore) {
    for bucket in ["zebra", "alpha", "mango"] {
        store.create_bucket(bucket).await.unwrap();
    }
    // Creating an existing bucket is a successful no-op.
    store.create_bucket("alpha").await.unwrap();
    assert_eq!(
        store.list_buckets().await.unwrap(),
        vec!["alpha", "mango", "zebra"]
    );
}

async fn object_round_trip(store: &dyn ObjectStore) {
    let meta = store
        .put("alpha", "notes.txt", Bytes::from_static(b"hello"))
        .await
        .unwrap();
    assert_eq!(meta.key, "notes.txt");
    assert_eq!(meta.size, 5);
    assert!(store.exists("alpha", "notes.txt").await.unwrap());

    // Overwriting replaces the payload and its metadata.
    let replaced = store
        .put("alpha", "notes.txt", Bytes::from(BINARY.to_vec()))
        .await
        .unwrap();
    assert_eq!(replaced.size, BINARY.len() as u64);
    assert_eq!(
        store.get("alpha", "notes.txt").await.unwrap(),
        Some(Bytes::from(BINARY.to_vec()))
    );

    // Nested keys are ordinary keys.
    store
        .put("alpha", "nested/deep.bin", Bytes::from(BINARY.to_vec()))
        .await
        .unwrap();
    assert_eq!(
        store.get("alpha", "nested/deep.bin").await.unwrap(),
        Some(Bytes::from(BINARY.to_vec()))
    );

    store.delete("alpha", "notes.txt").await.unwrap();
    assert!(!store.exists("alpha", "notes.txt").await.unwrap());
    assert_eq!(store.get("alpha", "notes.txt").await.unwrap(), None);
    // Deleting a missing object is a no-op.
    store.delete("alpha", "notes.txt").await.unwrap();
}

async fn prefix_listing(store: &dyn ObjectStore) {
    for (key, size) in [
        ("b.txt", 1_usize),
        ("a/one.txt", 2),
        ("a/two.txt", 3),
        ("c.txt", 4),
    ] {
        store
            .put("mango", key, Bytes::from(vec![b'x'; size]))
            .await
            .unwrap();
    }

    let all = store.list("mango", "", None).await.unwrap();
    assert_eq!(
        meta_keys(&all),
        vec!["a/one.txt", "a/two.txt", "b.txt", "c.txt"]
    );
    assert_eq!(all[1].size, 3);

    let nested = store.list("mango", "a/", None).await.unwrap();
    assert_eq!(meta_keys(&nested), vec!["a/one.txt", "a/two.txt"]);

    // The limit applies after the keys are sorted.
    let limited = store.list("mango", "", Some(2)).await.unwrap();
    assert_eq!(meta_keys(&limited), vec!["a/one.txt", "a/two.txt"]);

    assert_eq!(store.list("mango", "zzz", None).await.unwrap().len(), 0);
}

async fn missing_bucket(store: &dyn ObjectStore) {
    assert_not_found(
        store.put("absent", "k", Bytes::from_static(b"x")).await,
        "put",
    );
    assert_not_found(store.get("absent", "k").await, "get");
    assert_not_found(store.exists("absent", "k").await, "exists");
    assert_not_found(store.delete("absent", "k").await, "delete");
    assert_not_found(store.list("absent", "", None).await, "list");
}

async fn bucket_deletion(store: &dyn ObjectStore) {
    store.create_bucket("temp").await.unwrap();
    store
        .put("temp", "k.txt", Bytes::from_static(b"v"))
        .await
        .unwrap();
    // Empty buckets are deleted as readily as populated ones.
    store.delete_bucket("zebra").await.unwrap();
    store.delete_bucket("temp").await.unwrap();
    assert!(!store
        .list_buckets()
        .await
        .unwrap()
        .contains(&"temp".to_string()));
    // Deleting a missing bucket is a no-op.
    store.delete_bucket("temp").await.unwrap();

    // Recreating a bucket yields an empty one: its old objects are gone.
    store.create_bucket("temp").await.unwrap();
    assert_eq!(store.list("temp", "", None).await.unwrap().len(), 0);
    assert_eq!(
        store.list_buckets().await.unwrap(),
        vec!["alpha", "mango", "temp"]
    );
}
