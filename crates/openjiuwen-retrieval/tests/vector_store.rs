#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Integration tests for the vector store surface: records, queries, filters,
//! the in-memory store, JSON persistence, and the provider registry.
//!
//! The in-memory store and its registry are Rust additions required by the
//! offline retrieval pipeline, so these tests assert their documented behavior
//! (insertion order, `(cos + 1) / 2` scores, inclusive thresholds) rather than
//! porting a Python test file. Everything runs offline and deterministically.

use openjiuwen_common::error::ErrorKind;
use openjiuwen_retrieval::{
    cosine_similarity, normalize_cosine, InMemoryVectorStore, Metadata, MetadataFilter,
    SearchResult, VectorQuery, VectorRecord, VectorStore, VectorStoreConfig, VectorStoreRegistry,
    DEFAULT_DISTANCE_METRIC, DOC_ID_FIELD,
};
use serde_json::{json, Value};

/// Absolute tolerance for score comparisons (scores are exact or 0.5/1.0 based).
const TOLERANCE: f32 = 1e-5;

/// Builds a record with deterministic text and no metadata or document ids.
fn record(id: &str, vector: Vec<f32>) -> VectorRecord {
    VectorRecord {
        id: id.to_owned(),
        text: format!("text-{id}"),
        vector,
        ..VectorRecord::default()
    }
}

/// Builds a metadata map from JSON values.
fn metadata(entries: &[(&str, Value)]) -> Metadata {
    entries
        .iter()
        .map(|(key, value)| ((*key).to_owned(), value.clone()))
        .collect()
}

/// Builds a store configuration with all four fields set explicitly.
///
/// `database_name` is blank because the in-memory store has no databases; the
/// external-provider fixtures below set it themselves.
fn config(provider: &str, collection: &str) -> VectorStoreConfig {
    VectorStoreConfig {
        store_provider: provider.to_owned(),
        database_name: String::new(),
        collection_name: collection.to_owned(),
        distance_metric: DEFAULT_DISTANCE_METRIC.to_owned(),
    }
}

/// Extracts record ids in list order.
fn record_ids(records: &[VectorRecord]) -> Vec<String> {
    records.iter().map(|record| record.id.clone()).collect()
}

/// Extracts hit ids in result order.
fn hit_ids(hits: &[SearchResult]) -> Vec<String> {
    hits.iter().map(|hit| hit.id.clone()).collect()
}

/// A store populated with the four-quadrant vectors used by the search tests.
async fn quadrant_store() -> InMemoryVectorStore {
    let store = InMemoryVectorStore::new();
    store
        .upsert(vec![
            record("a", vec![1.0, 0.0]),
            record("b", vec![1.0, 1.0]),
            record("c", vec![0.0, 1.0]),
            record("d", vec![-1.0, 0.0]),
        ])
        .await
        .unwrap();
    store
}

#[tokio::test]
async fn upsert_get_list_and_clear_round_trip() {
    let store = InMemoryVectorStore::new();
    assert!(store.is_empty().await.unwrap());
    assert_eq!(store.len().await.unwrap(), 0);
    assert_eq!(store.name(), "default");

    let inserted = store
        .upsert(vec![
            record("a", vec![1.0, 0.0]),
            record("b", vec![0.0, 1.0]),
            record("c", vec![1.0, 1.0]),
        ])
        .await
        .unwrap();
    assert_eq!(inserted, 3);
    assert_eq!(store.len().await.unwrap(), 3);
    assert!(!store.is_empty().await.unwrap());

    let got = store.get("b").await.unwrap().expect("record b must exist");
    assert_eq!(got.text, "text-b");
    assert_eq!(got.vector, vec![0.0, 1.0]);
    assert!(store.get("missing").await.unwrap().is_none());

    // `list` follows insertion order and truncates without reordering.
    assert_eq!(record_ids(&store.list(Some(2)).await.unwrap()), ["a", "b"]);
    assert_eq!(
        record_ids(&store.list(None).await.unwrap()),
        ["a", "b", "c"]
    );

    store.clear().await.unwrap();
    assert_eq!(store.len().await.unwrap(), 0);
    assert!(store.is_empty().await.unwrap());
    assert_eq!(store.list(None).await.unwrap(), Vec::new());
}

#[tokio::test]
async fn delete_counts_existing_ids_and_removes_them() {
    let store = InMemoryVectorStore::new();
    store
        .upsert(vec![
            record("a", vec![1.0, 0.0]),
            record("b", vec![0.0, 1.0]),
            record("c", vec![1.0, 1.0]),
        ])
        .await
        .unwrap();

    // Only ids that existed are counted; unknown and duplicate ids are ignored.
    let removed = store
        .delete(&[
            "a".to_owned(),
            "missing".to_owned(),
            "c".to_owned(),
            "a".to_owned(),
        ])
        .await
        .unwrap();
    assert_eq!(removed, 2);
    assert_eq!(store.len().await.unwrap(), 1);
    assert!(store.get("a").await.unwrap().is_none());
    assert!(store.get("c").await.unwrap().is_none());
    assert!(store.get("b").await.unwrap().is_some());
    assert_eq!(store.delete(&[]).await.unwrap(), 0);
}

#[tokio::test]
async fn upsert_replaces_payload_without_duplicating_or_reordering() {
    let store = InMemoryVectorStore::new();
    store
        .upsert(vec![
            record("a", vec![1.0, 0.0]),
            record("b", vec![0.0, 1.0]),
            record("c", vec![1.0, 1.0]),
        ])
        .await
        .unwrap();

    let mut replacement = record("b", vec![0.0, 2.0]);
    replacement.text = "updated".to_owned();
    replacement.metadata = metadata(&[("kind", json!("b"))]);
    let inserted = store.upsert(vec![replacement.clone()]).await.unwrap();

    assert_eq!(inserted, 1);
    assert_eq!(store.len().await.unwrap(), 3);
    // `b` keeps its original position instead of moving to the end.
    assert_eq!(
        record_ids(&store.list(None).await.unwrap()),
        ["a", "b", "c"]
    );
    assert_eq!(store.get("b").await.unwrap(), Some(replacement));
}

#[tokio::test]
async fn search_orders_by_descending_similarity_and_truncates() {
    let store = quadrant_store().await;
    let hits = store
        .search(VectorQuery::new(vec![1.0, 0.0]).with_top_k(10))
        .await
        .unwrap();

    // Scores: a = (1+1)/2 = 1.0, b = (sqrt(1/2)+1)/2, c = 0.5, d = 0.0.
    assert_eq!(hit_ids(&hits), ["a", "b", "c", "d"]);
    assert!((hits[0].score - 1.0).abs() < TOLERANCE);
    assert!((hits[1].score - 0.853_553_4).abs() < TOLERANCE);
    assert!((hits[2].score - 0.5).abs() < TOLERANCE);
    assert!((hits[3].score - 0.0).abs() < TOLERANCE);
    for pair in hits.windows(2) {
        assert!(
            pair[0].score > pair[1].score,
            "scores must be strictly descending"
        );
    }

    // `top_k` keeps the best hits.
    let top = store
        .search(VectorQuery::new(vec![1.0, 0.0]).with_top_k(2))
        .await
        .unwrap();
    assert_eq!(hit_ids(&top), ["a", "b"]);
}

#[tokio::test]
async fn search_with_top_k_zero_returns_no_hits() {
    let store = quadrant_store().await;
    let hits = store
        .search(VectorQuery::new(vec![1.0, 0.0]).with_top_k(0))
        .await
        .unwrap();
    assert_eq!(hits, Vec::new());
    // The store itself is untouched.
    assert_eq!(store.len().await.unwrap(), 4);
}

#[tokio::test]
async fn search_score_threshold_is_inclusive() {
    let store = quadrant_store().await;
    let query = VectorQuery::new(vec![1.0, 0.0]).with_top_k(10);

    // c scores exactly 0.5, so `>= 0.5` keeps it.
    let inclusive = store
        .search(query.clone().with_score_threshold(0.5))
        .await
        .unwrap();
    assert_eq!(hit_ids(&inclusive), ["a", "b", "c"]);

    // Just above c's score drops it.
    let exclusive = store
        .search(query.clone().with_score_threshold(0.500_000_1))
        .await
        .unwrap();
    assert_eq!(hit_ids(&exclusive), ["a", "b"]);

    // An exact top score is still included.
    let top = store.search(query.with_score_threshold(1.0)).await.unwrap();
    assert_eq!(hit_ids(&top), ["a"]);
}

#[tokio::test]
async fn search_keeps_insertion_order_for_equal_scores() {
    let store = InMemoryVectorStore::new();
    store
        .upsert(vec![
            record("first", vec![1.0, 0.0]),
            record("second", vec![1.0, 0.0]),
            record("third", vec![1.0, 0.0]),
            record("other", vec![0.0, 1.0]),
        ])
        .await
        .unwrap();

    let tied = store
        .search(VectorQuery::new(vec![1.0, 0.0]).with_top_k(3))
        .await
        .unwrap();
    assert_eq!(hit_ids(&tied), ["first", "second", "third"]);
    for hit in &tied {
        assert!((hit.score - 1.0).abs() < TOLERANCE);
    }

    // The lower-scoring record sorts last without disturbing the tie order.
    let all = store
        .search(VectorQuery::new(vec![1.0, 0.0]).with_top_k(10))
        .await
        .unwrap();
    assert_eq!(hit_ids(&all), ["first", "second", "third", "other"]);
}

#[tokio::test]
async fn search_filters_by_metadata_equality() {
    let store = InMemoryVectorStore::new();
    let mut first = record("a", vec![1.0, 0.0]);
    first.metadata = metadata(&[("kind", json!("a")), ("rank", json!(1))]);
    let mut second = record("b", vec![1.0, 0.0]);
    second.metadata = metadata(&[("kind", json!("b")), ("rank", json!(2))]);
    store.upsert(vec![first, second]).await.unwrap();

    let by_constructor = store
        .search(VectorQuery::new(vec![1.0, 0.0]).with_filter(MetadataFilter::eq("kind", "a")))
        .await
        .unwrap();
    assert_eq!(hit_ids(&by_constructor), ["a"]);

    // The Python JSON shape deserializes to the same predicate.
    let from_json: MetadataFilter = serde_json::from_value(json!({"kind": "a"})).unwrap();
    assert_eq!(from_json, MetadataFilter::eq("kind", "a"));
    let by_json = store
        .search(VectorQuery::new(vec![1.0, 0.0]).with_filter(from_json))
        .await
        .unwrap();
    assert_eq!(hit_ids(&by_json), ["a"]);

    let no_match = store
        .search(VectorQuery::new(vec![1.0, 0.0]).with_filter(MetadataFilter::eq("kind", "zzz")))
        .await
        .unwrap();
    assert_eq!(no_match, Vec::new());
}

#[tokio::test]
async fn search_filters_by_numeric_comparison() {
    let store = InMemoryVectorStore::new();
    let mut records = Vec::new();
    for rank in 1..=3 {
        let mut entry = record(&format!("rank-{rank}"), vec![1.0, 0.0]);
        entry.metadata = metadata(&[("rank", json!(rank))]);
        records.push(entry);
    }
    store.upsert(records).await.unwrap();

    let filter: MetadataFilter = serde_json::from_value(json!({"rank": {"$gt": 1}})).unwrap();
    let hits = store
        .search(VectorQuery::new(vec![1.0, 0.0]).with_filter(filter))
        .await
        .unwrap();
    assert_eq!(hit_ids(&hits), ["rank-2", "rank-3"]);
}

#[tokio::test]
async fn search_filters_by_logical_and() {
    let store = InMemoryVectorStore::new();
    let mut first = record("a1", vec![1.0, 0.0]);
    first.metadata = metadata(&[("kind", json!("a")), ("rank", json!(1))]);
    let mut second = record("a2", vec![1.0, 0.0]);
    second.metadata = metadata(&[("kind", json!("a")), ("rank", json!(2))]);
    let mut third = record("b3", vec![1.0, 0.0]);
    third.metadata = metadata(&[("kind", json!("b")), ("rank", json!(3))]);
    store.upsert(vec![first, second, third]).await.unwrap();

    let filter: MetadataFilter =
        serde_json::from_value(json!({"$and": [{"kind": "a"}, {"rank": {"$gte": 2}}]})).unwrap();
    let hits = store
        .search(VectorQuery::new(vec![1.0, 0.0]).with_filter(filter))
        .await
        .unwrap();
    assert_eq!(hit_ids(&hits), ["a2"]);
}

#[tokio::test]
async fn search_filters_by_doc_id_pseudo_field() {
    let store = InMemoryVectorStore::new();
    // The metadata deliberately lacks `doc_id`: the store resolves it from the record.
    let mut first = record("c1", vec![1.0, 0.0]);
    first.doc_id = Some("doc-1".to_owned());
    first.metadata = metadata(&[("kind", json!("a"))]);
    let mut second = record("c2", vec![1.0, 0.0]);
    second.doc_id = Some("doc-2".to_owned());
    second.metadata = metadata(&[("kind", json!("a"))]);
    store.upsert(vec![first, second]).await.unwrap();

    assert!(!store
        .get("c1")
        .await
        .unwrap()
        .expect("record c1 must exist")
        .metadata
        .contains_key(DOC_ID_FIELD));

    let filter: MetadataFilter = serde_json::from_value(json!({"doc_id": "doc-2"})).unwrap();
    let hits = store
        .search(VectorQuery::new(vec![1.0, 0.0]).with_filter(filter))
        .await
        .unwrap();
    assert_eq!(hit_ids(&hits), ["c2"]);
    assert_eq!(hits[0].metadata[DOC_ID_FIELD], json!("doc-2"));

    let unknown = store
        .search(
            VectorQuery::new(vec![1.0, 0.0]).with_filter(MetadataFilter::eq(DOC_ID_FIELD, "doc-9")),
        )
        .await
        .unwrap();
    assert_eq!(unknown, Vec::new());
}

#[tokio::test]
async fn search_injects_doc_and_chunk_ids_into_result_metadata() {
    let store = InMemoryVectorStore::new();
    let mut annotated = record("c1", vec![1.0, 0.0]);
    annotated.doc_id = Some("d1".to_owned());
    annotated.chunk_id = Some("c1".to_owned());
    annotated.metadata = metadata(&[("kind", json!("a"))]);
    let plain = record("c2", vec![1.0, 0.0]);
    store.upsert(vec![annotated, plain]).await.unwrap();

    let hits = store
        .search(VectorQuery::new(vec![1.0, 0.0]).with_top_k(2))
        .await
        .unwrap();
    let annotated_hit = hits.iter().find(|hit| hit.id == "c1").expect("c1 must hit");
    assert_eq!(annotated_hit.metadata["doc_id"], json!("d1"));
    assert_eq!(annotated_hit.metadata["chunk_id"], json!("c1"));
    assert_eq!(annotated_hit.metadata["kind"], json!("a"));

    let plain_hit = hits.iter().find(|hit| hit.id == "c2").expect("c2 must hit");
    assert_eq!(plain_hit.metadata, Metadata::new());
}

#[tokio::test]
async fn search_skips_dimension_mismatch_and_handles_zero_query() {
    let store = InMemoryVectorStore::new();
    store
        .upsert(vec![
            record("two-d", vec![1.0, 0.0]),
            record("three-d", vec![1.0, 0.0, 0.0]),
        ])
        .await
        .unwrap();

    let hits = store
        .search(VectorQuery::new(vec![1.0, 0.0]).with_top_k(10))
        .await
        .unwrap();
    assert_eq!(hit_ids(&hits), ["two-d"]);

    // A zero query vector scores 0.5 after normalization instead of producing NaN.
    let zero = store
        .search(VectorQuery::new(vec![0.0, 0.0]).with_top_k(10))
        .await
        .unwrap();
    assert_eq!(hit_ids(&zero), ["two-d"]);
    assert!(zero[0].score.is_finite());
    assert!((zero[0].score - 0.5).abs() < TOLERANCE);
}

#[tokio::test]
async fn persistence_survives_reopen_and_reflects_deletes() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("vectors.json");

    // Opening a non-existent path yields an empty store bound to that path.
    let store = InMemoryVectorStore::open(&path).await.unwrap();
    assert_eq!(store.path(), Some(path.as_path()));
    assert!(store.is_empty().await.unwrap());

    let mut annotated = record("c1", vec![1.0, 0.0]);
    annotated.doc_id = Some("d1".to_owned());
    annotated.chunk_id = Some("c1".to_owned());
    annotated.metadata = metadata(&[("kind", json!("a")), ("rank", json!(1))]);
    store
        .upsert(vec![annotated.clone(), record("c2", vec![0.0, 1.0])])
        .await
        .unwrap();
    // Upsert auto-persists; an explicit `persist` is idempotent.
    store.persist().await.unwrap();
    assert!(tokio::fs::try_exists(&path).await.unwrap());

    let reopened = InMemoryVectorStore::open(&path).await.unwrap();
    assert_eq!(reopened.len().await.unwrap(), 2);
    assert_eq!(reopened.get("c1").await.unwrap(), Some(annotated));
    assert_eq!(
        record_ids(&reopened.list(None).await.unwrap()),
        ["c1", "c2"]
    );

    assert_eq!(reopened.delete(&["c1".to_owned()]).await.unwrap(), 1);

    let third = InMemoryVectorStore::open(&path).await.unwrap();
    assert_eq!(third.len().await.unwrap(), 1);
    assert!(third.get("c1").await.unwrap().is_none());
    assert_eq!(
        third
            .get("c2")
            .await
            .unwrap()
            .expect("c2 must survive")
            .vector,
        vec![0.0, 1.0]
    );
}

#[tokio::test]
async fn registry_builtin_providers_build_working_stores() {
    let registry = VectorStoreRegistry::with_builtin();
    assert!(registry.contains("memory"));
    assert!(registry.contains("in_memory"));
    assert!(!registry.contains("milvus"));

    // Every field is set explicitly because `VectorStoreConfig` has no `Default`.
    let memory_config = config("memory", "docs");
    let store = registry.create(&memory_config).unwrap();
    assert_eq!(store.name(), "docs");
    store
        .upsert(vec![record("a", vec![1.0, 0.0])])
        .await
        .unwrap();
    assert_eq!(store.len().await.unwrap(), 1);
    let hits = store
        .search(VectorQuery::new(vec![1.0, 0.0]))
        .await
        .unwrap();
    assert_eq!(hit_ids(&hits), ["a"]);

    // The alias builds the same store type; a blank collection falls back to the default.
    let alias = registry.create(&config("in_memory", "")).unwrap();
    assert_eq!(alias.name(), "default");
    assert!(alias.is_empty().await.unwrap());

    let milvus_config = VectorStoreConfig {
        store_provider: "milvus".to_owned(),
        database_name: "test-db".to_owned(),
        collection_name: "docs".to_owned(),
        distance_metric: "cosine".to_owned(),
    };
    let error = registry
        .create(&milvus_config)
        .err()
        .expect("milvus provider must be rejected");
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert!(error.to_string().contains("milvus"));
}

#[test]
fn cosine_helpers_match_documented_conventions() {
    let identical = cosine_similarity(&[0.5, 0.5], &[0.5, 0.5]);
    assert!((identical - 1.0).abs() < TOLERANCE);
    assert!((normalize_cosine(identical) - 1.0).abs() < TOLERANCE);

    let orthogonal = cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]);
    assert!(orthogonal.abs() < TOLERANCE);
    assert!((normalize_cosine(orthogonal) - 0.5).abs() < TOLERANCE);

    // A dimension mismatch degrades to 0.0 instead of panicking.
    assert!(cosine_similarity(&[1.0, 2.0], &[1.0]).abs() < TOLERANCE);
    assert!((normalize_cosine(0.0) - 0.5).abs() < TOLERANCE);
}

#[test]
fn from_config_reads_every_field_and_rejects_unhonoured_values() {
    let valid = InMemoryVectorStore::from_config(&config("memory", "docs")).unwrap();
    assert_eq!(valid.name(), "docs");
    // A blank provider and blank collection are accepted (defaults apply).
    let blank = InMemoryVectorStore::from_config(&config("", "")).unwrap();
    assert_eq!(blank.name(), "default");
    // Provider and metric names are case-insensitive.
    let mixed = VectorStoreConfig {
        store_provider: "In_Memory".to_owned(),
        distance_metric: "COSINE".to_owned(),
        ..config("", "docs")
    };
    assert_eq!(
        InMemoryVectorStore::from_config(&mixed).unwrap().name(),
        "docs"
    );

    let mut database = config("memory", "docs");
    database.database_name = "test-db".to_owned();
    let error = InMemoryVectorStore::from_config(&database).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Config);
    assert!(error.to_string().contains("no databases"), "{error}");

    for metric in ["euclidean", "dot", ""] {
        let mut bad = config("memory", "docs");
        bad.distance_metric = metric.to_owned();
        let error = InMemoryVectorStore::from_config(&bad).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Config, "metric {metric}");
        assert!(
            error.to_string().contains("only supports the `cosine`"),
            "{error}"
        );
    }

    let error = InMemoryVectorStore::from_config(&config("milvus", "docs")).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Config);
    assert!(
        error.to_string().contains("cannot serve provider `milvus`"),
        "{error}"
    );
}

#[tokio::test]
async fn registry_surfaces_an_invalid_memory_config_as_a_config_error() {
    let registry = VectorStoreRegistry::with_builtin();
    let mut bad = config("memory", "docs");
    bad.distance_metric = "euclidean".to_owned();
    let error = registry
        .create(&bad)
        .err()
        .expect("a non-cosine metric must be rejected");
    assert_eq!(error.kind(), ErrorKind::Config);
    // A database name is likewise reported rather than dropped.
    let mut database = config("in_memory", "docs");
    database.database_name = "test-db".to_owned();
    let error = registry
        .create(&database)
        .err()
        .expect("a database name must be rejected");
    assert_eq!(error.kind(), ErrorKind::Config);
}
