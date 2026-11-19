//! Unit tests for the in-memory vector store: search, filtering, and persistence.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use super::*;
use openjiuwen_common::error::ErrorKind;
use serde_json::json;

fn record(id: &str, vector: Vec<f32>) -> VectorRecord {
    VectorRecord {
        id: id.to_owned(),
        text: format!("text-{id}"),
        vector,
        ..VectorRecord::default()
    }
}

fn with_metadata(id: &str, vector: Vec<f32>, metadata: Metadata) -> VectorRecord {
    VectorRecord {
        metadata,
        ..record(id, vector)
    }
}

fn metadata(fields: &[(&str, Value)]) -> Metadata {
    fields
        .iter()
        .map(|(key, value)| ((*key).to_owned(), value.clone()))
        .collect()
}

async fn store_with(records: Vec<VectorRecord>) -> InMemoryVectorStore {
    let store = InMemoryVectorStore::new();
    store.upsert(records).await.unwrap();
    store
}

fn ids(hits: &[SearchResult]) -> Vec<&str> {
    hits.iter().map(|hit| hit.id.as_str()).collect()
}

#[tokio::test]
async fn upsert_get_len_is_empty_clear() {
    let store = InMemoryVectorStore::new();
    assert_eq!(store.name(), "default");
    assert!(store.is_empty().await.unwrap());
    assert_eq!(
        store
            .upsert(vec![record("a", vec![1.0, 0.0])])
            .await
            .unwrap(),
        1
    );
    assert_eq!(store.len().await.unwrap(), 1);
    assert!(!store.is_empty().await.unwrap());
    assert_eq!(store.get("a").await.unwrap().unwrap().text, "text-a");
    assert!(store.get("missing").await.unwrap().is_none());
    assert_eq!(store.list(Some(1)).await.unwrap().len(), 1);
    store.clear().await.unwrap();
    assert!(store.is_empty().await.unwrap());
    assert!(store.get("a").await.unwrap().is_none());
}

#[tokio::test]
async fn duplicate_upsert_replaces_without_reordering() {
    let store = store_with(vec![
        record("a", vec![1.0, 0.0]),
        record("b", vec![0.0, 1.0]),
    ])
    .await;
    let mut updated = record("a", vec![0.0, 2.0]);
    updated.text = "updated".to_owned();
    assert_eq!(store.upsert(vec![updated]).await.unwrap(), 1);
    let listed = store.list(None).await.unwrap();
    assert_eq!(
        listed.iter().map(|r| r.id.as_str()).collect::<Vec<_>>(),
        ["a", "b"]
    );
    assert_eq!(listed[0].text, "updated");
    assert_eq!(listed[0].vector, vec![0.0, 2.0]);
    assert_eq!(store.len().await.unwrap(), 2);
}

#[tokio::test]
async fn search_orders_by_descending_score() {
    let store = store_with(vec![
        record("orthogonal", vec![0.0, 1.0]),
        record("exact", vec![1.0, 0.0]),
        record("near", vec![0.8, 0.6]),
    ])
    .await;
    let hits = store
        .search(VectorQuery::new(vec![1.0, 0.0]))
        .await
        .unwrap();
    assert_eq!(ids(&hits), ["exact", "near", "orthogonal"]);
    assert!((hits[0].score - 1.0).abs() < 1e-6);
    assert!((hits[1].score - 0.9).abs() < 1e-6);
    assert!((hits[2].score - 0.5).abs() < 1e-6);
    assert_eq!(hits[0].text, "text-exact");
}

#[tokio::test]
async fn search_keeps_insertion_order_for_equal_scores() {
    let vector = vec![1.0, 0.0];
    let store = store_with(vec![
        record("first", vector.clone()),
        record("second", vector.clone()),
        record("third", vector),
    ])
    .await;
    let hits = store
        .search(VectorQuery::new(vec![1.0, 0.0]))
        .await
        .unwrap();
    assert_eq!(ids(&hits), ["first", "second", "third"]);
}

#[tokio::test]
async fn search_sorts_nan_scores_last_in_a_transitive_order() {
    // A stored vector holding a NaN produces a NaN score. `partial_cmp(…)
    // .unwrap_or(Equal)` treated every NaN pair as equal, so a NaN hit could
    // keep a position above a real score; the total order must push every NaN
    // below every finite score and still keep insertion order among the NaNs.
    let store = store_with(vec![
        record("nan-first", vec![f32::NAN, 1.0]),
        record("exact", vec![1.0, 0.0]),
        record("nan-second", vec![f32::NAN, 0.0]),
        record("near", vec![0.8, 0.6]),
    ])
    .await;
    let hits = store
        .search(VectorQuery::new(vec![1.0, 0.0]))
        .await
        .unwrap();
    assert_eq!(ids(&hits), ["exact", "near", "nan-first", "nan-second"]);
    assert!(hits[0].score.is_finite() && hits[1].score.is_finite());
    assert!(hits[2].score.is_nan() && hits[3].score.is_nan());
    // Truncation happens before the payload clone, so a NaN never survives a
    // `top_k` that the finite hits already fill.
    let top = store
        .search(VectorQuery::new(vec![1.0, 0.0]).with_top_k(1))
        .await
        .unwrap();
    assert_eq!(ids(&top), ["exact"]);
}

#[tokio::test]
async fn search_truncates_to_top_k_and_zero_returns_empty() {
    let store = store_with(vec![
        record("a", vec![1.0, 0.0]),
        record("b", vec![0.9, 0.1]),
        record("c", vec![0.5, 0.5]),
    ])
    .await;
    let hits = store
        .search(VectorQuery::new(vec![1.0, 0.0]).with_top_k(2))
        .await
        .unwrap();
    assert_eq!(ids(&hits), ["a", "b"]);
    let none = store
        .search(VectorQuery::new(vec![1.0, 0.0]).with_top_k(0))
        .await
        .unwrap();
    assert_eq!(none, Vec::<SearchResult>::new());
}

#[tokio::test]
async fn search_score_threshold_is_inclusive() {
    let store = store_with(vec![
        record("exact", vec![1.0, 0.0]),
        record("near", vec![0.8, 0.6]),
    ])
    .await;
    let baseline = store
        .search(VectorQuery::new(vec![1.0, 0.0]))
        .await
        .unwrap();
    let bound = baseline[1].score;
    let hits = store
        .search(VectorQuery::new(vec![1.0, 0.0]).with_score_threshold(bound))
        .await
        .unwrap();
    assert_eq!(ids(&hits), ["exact", "near"]);
    let strict = store
        .search(VectorQuery::new(vec![1.0, 0.0]).with_score_threshold(bound + 1e-3))
        .await
        .unwrap();
    assert_eq!(ids(&strict), ["exact"]);
}

#[tokio::test]
async fn search_applies_equality_and_numeric_filters() {
    let store = store_with(vec![
        with_metadata(
            "a",
            vec![1.0, 0.0],
            metadata(&[("year", json!(2023)), ("lang", json!("en"))]),
        ),
        with_metadata(
            "b",
            vec![1.0, 0.0],
            metadata(&[("year", json!(2024)), ("lang", json!("en"))]),
        ),
        with_metadata(
            "c",
            vec![1.0, 0.0],
            metadata(&[("year", json!(2025)), ("lang", json!("fr"))]),
        ),
    ])
    .await;
    let query = VectorQuery::new(vec![1.0, 0.0]).with_filter(MetadataFilter::eq("lang", "en"));
    assert_eq!(ids(&store.search(query).await.unwrap()), ["a", "b"]);
    let query = VectorQuery::new(vec![1.0, 0.0])
        .with_filter(MetadataFilter::from_value(&json!({"year": {"$gt": 2023}})).unwrap());
    assert_eq!(ids(&store.search(query).await.unwrap()), ["b", "c"]);
}

#[tokio::test]
async fn search_applies_and_filter() {
    let store = store_with(vec![
        with_metadata(
            "a",
            vec![1.0, 0.0],
            metadata(&[("year", json!(2024)), ("lang", json!("en"))]),
        ),
        with_metadata(
            "b",
            vec![1.0, 0.0],
            metadata(&[("year", json!(2024)), ("lang", json!("fr"))]),
        ),
    ])
    .await;
    let filter = MetadataFilter::and([
        MetadataFilter::eq("lang", "en"),
        MetadataFilter::from_value(&json!({"year": {"$gte": 2024}})).unwrap(),
    ]);
    let hits = store
        .search(VectorQuery::new(vec![1.0, 0.0]).with_filter(filter))
        .await
        .unwrap();
    assert_eq!(ids(&hits), ["a"]);
}

#[tokio::test]
async fn search_filter_resolves_doc_and_chunk_pseudo_fields() {
    let mut first = record("first", vec![1.0, 0.0]);
    first.doc_id = Some("doc-1".to_owned());
    first.chunk_id = Some("chunk-1".to_owned());
    let mut second = record("second", vec![1.0, 0.0]);
    second.doc_id = Some("doc-2".to_owned());
    let store = store_with(vec![first, second]).await;

    let by_doc = store
        .search(VectorQuery::new(vec![1.0, 0.0]).with_filter(MetadataFilter::eq("doc_id", "doc-1")))
        .await
        .unwrap();
    assert_eq!(ids(&by_doc), ["first"]);
    assert_eq!(by_doc[0].metadata["doc_id"], json!("doc-1"));
    assert_eq!(by_doc[0].metadata["chunk_id"], json!("chunk-1"));

    let by_chunk = store
        .search(
            VectorQuery::new(vec![1.0, 0.0]).with_filter(MetadataFilter::eq("chunk_id", "chunk-1")),
        )
        .await
        .unwrap();
    assert_eq!(ids(&by_chunk), ["first"]);
}

#[tokio::test]
async fn search_rejects_dimension_mismatch_and_scores_zero_queries() {
    // (a) A consistent store searched with a query of a different width is an error.
    let store = store_with(vec![
        record("two", vec![1.0, 0.0]),
        record("other", vec![0.0, 1.0]),
    ])
    .await;
    let error = store
        .search(VectorQuery::new(vec![1.0, 0.0, 0.0]))
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    let message = error.to_string();
    assert!(message.contains("vector dimension mismatch"), "{message}");
    assert!(message.contains("2-dimensional"), "{message}");
    assert!(message.contains("3 dimensions"), "{message}");

    // (b) A store mixing widths is an inconsistency error naming the first pair.
    let mixed = store_with(vec![
        record("two", vec![1.0, 0.0]),
        record("three", vec![1.0, 0.0, 0.0]),
    ])
    .await;
    let error = mixed
        .search(VectorQuery::new(vec![1.0, 0.0]))
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    let message = error.to_string();
    assert!(message.contains("internally inconsistent"), "{message}");
    assert!(
        message.contains("`two`") && message.contains("`three`"),
        "{message}"
    );

    // (c) An empty store has no known width, so any query width returns no hits.
    let empty = InMemoryVectorStore::new();
    for width in [0, 2, 3] {
        let hits = empty
            .search(VectorQuery::new(vec![0.5; width]))
            .await
            .unwrap();
        assert_eq!(hits, Vec::<SearchResult>::new());
    }

    // (d) A zero query vector scores 0.5 after normalization instead of producing NaN.
    let zero = store
        .search(VectorQuery::new(vec![0.0, 0.0]))
        .await
        .unwrap();
    assert_eq!(ids(&zero), ["two", "other"]);
    assert!(zero.iter().all(|hit| hit.score.is_finite()));
    assert!((zero[0].score - 0.5).abs() < 1e-6);
    assert!((zero[1].score - 0.5).abs() < 1e-6);
}

#[tokio::test]
async fn delete_returns_number_of_existing_ids() {
    let store = store_with(vec![record("a", vec![1.0]), record("b", vec![1.0])]).await;
    let removed = store
        .delete(&["a".to_owned(), "missing".to_owned()])
        .await
        .unwrap();
    assert_eq!(removed, 1);
    assert_eq!(store.list(None).await.unwrap().len(), 1);
    assert_eq!(store.delete(&[]).await.unwrap(), 0);
}

#[tokio::test]
async fn delete_by_filter_removes_matches() {
    let store = store_with(vec![
        with_metadata("keep", vec![1.0], metadata(&[("lang", json!("en"))])),
        with_metadata("drop", vec![1.0], metadata(&[("lang", json!("fr"))])),
    ])
    .await;
    let removed = store
        .delete_by_filter(&MetadataFilter::eq("lang", "fr"))
        .await
        .unwrap();
    assert_eq!(removed, 1);
    assert_eq!(
        ids(&store.search(VectorQuery::new(vec![1.0])).await.unwrap()),
        ["keep"]
    );
}

#[tokio::test]
async fn persistence_round_trips_across_instances() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("store.json");
    let store = InMemoryVectorStore::open_with_collection(&path, "docs")
        .await
        .unwrap();
    store
        .upsert(vec![with_metadata(
            "a",
            vec![1.0, 0.0],
            metadata(&[("lang", json!("en"))]),
        )])
        .await
        .unwrap();
    assert_eq!(store.path(), Some(path.as_path()));

    let reopened = InMemoryVectorStore::open(&path).await.unwrap();
    assert_eq!(reopened.name(), "docs");
    assert_eq!(reopened.len().await.unwrap(), 1);
    let hit = reopened
        .search(VectorQuery::new(vec![1.0, 0.0]))
        .await
        .unwrap();
    assert_eq!(hit[0].id, "a");
    assert_eq!(hit[0].metadata["lang"], json!("en"));
}

#[tokio::test]
async fn persistence_reflects_deletes_and_clear() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("store.json");
    let store = InMemoryVectorStore::open(&path).await.unwrap();
    store
        .upsert(vec![record("a", vec![1.0]), record("b", vec![1.0])])
        .await
        .unwrap();
    store.delete(&["a".to_owned()]).await.unwrap();
    let reopened = InMemoryVectorStore::open(&path).await.unwrap();
    assert_eq!(reopened.len().await.unwrap(), 1);
    assert!(reopened.get("a").await.unwrap().is_none());

    store.clear().await.unwrap();
    let cleared = InMemoryVectorStore::open(&path).await.unwrap();
    assert!(cleared.is_empty().await.unwrap());
}

#[tokio::test]
async fn open_missing_file_is_empty_and_persist_without_path_is_noop() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("absent.json");
    let store = InMemoryVectorStore::open(&path).await.unwrap();
    assert!(store.is_empty().await.unwrap());
    assert_eq!(store.name(), "default");

    let memory_only = InMemoryVectorStore::new();
    assert!(memory_only.path().is_none());
    memory_only.persist().await.unwrap();
}

#[tokio::test]
async fn open_rejects_malformed_and_unknown_version_snapshots() {
    let dir = tempfile::tempdir().unwrap();
    let malformed = dir.path().join("malformed.json");
    tokio::fs::write(&malformed, b"not json").await.unwrap();
    assert!(matches!(
        InMemoryVectorStore::open(&malformed).await,
        Err(Error::Retrieval(_))
    ));

    let future = dir.path().join("future.json");
    let snapshot = json!({"version": 2, "name": "docs", "records": []});
    tokio::fs::write(&future, serde_json::to_vec(&snapshot).unwrap())
        .await
        .unwrap();
    let error = InMemoryVectorStore::open(&future).await.unwrap_err();
    assert!(matches!(&error, Error::Retrieval(message) if message.contains("version 2")));
}
#[tokio::test]
async fn get_many_preserves_requested_order() {
    let store = store_with(vec![
        record("a", vec![1.0, 0.0]),
        record("b", vec![0.0, 1.0]),
        record("c", vec![1.0, 1.0]),
    ])
    .await;
    let requested = vec!["c".to_owned(), "a".to_owned(), "b".to_owned()];
    let records = store.get_many(&requested).await.unwrap();
    let got: Vec<&str> = records.iter().map(|record| record.id.as_str()).collect();
    assert_eq!(got, vec!["c", "a", "b"]);
    assert_eq!(records[0].text, "text-c");
    assert_eq!(records[1].text, "text-a");
}

#[tokio::test]
async fn get_many_skips_unknown_ids() {
    let store = store_with(vec![record("a", vec![1.0, 0.0])]).await;
    let requested = vec![
        "missing".to_owned(),
        "a".to_owned(),
        "also-missing".to_owned(),
    ];
    let records = store.get_many(&requested).await.unwrap();
    let got: Vec<&str> = records.iter().map(|record| record.id.as_str()).collect();
    assert_eq!(got, vec!["a"]);
    assert_eq!(store.get_many(&["nope".to_owned()]).await.unwrap(), vec![]);
}

#[tokio::test]
async fn get_many_with_empty_input_is_empty() {
    let store = store_with(vec![record("a", vec![1.0, 0.0])]).await;
    assert_eq!(store.get_many(&[]).await.unwrap(), vec![]);
    assert_eq!(
        InMemoryVectorStore::new().get_many(&[]).await.unwrap(),
        vec![]
    );
}
