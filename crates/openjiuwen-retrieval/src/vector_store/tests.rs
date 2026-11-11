//! Unit tests for the vector store registry, records, queries, and cosine helpers.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use super::*;
use serde_json::json;

fn record(id: &str, text: &str, vector: Vec<f32>) -> VectorRecord {
    VectorRecord {
        id: id.to_owned(),
        text: text.to_owned(),
        vector,
        ..VectorRecord::default()
    }
}

fn config(provider: &str, collection: &str) -> VectorStoreConfig {
    VectorStoreConfig {
        store_provider: provider.to_owned(),
        database_name: String::new(),
        collection_name: collection.to_owned(),
        distance_metric: crate::common::DEFAULT_DISTANCE_METRIC.to_owned(),
    }
}

fn create_error(registry: &VectorStoreRegistry, config: &VectorStoreConfig) -> Error {
    match registry.create(config) {
        Ok(_) => panic!(
            "expected the registry to reject '{}'",
            config.store_provider
        ),
        Err(error) => error,
    }
}

#[test]
fn vector_record_round_trips_with_python_field_names() {
    let original = VectorRecord {
        id: "c1".to_owned(),
        text: "hello".to_owned(),
        vector: vec![0.5, -0.5],
        metadata: Metadata::from_iter([("lang".to_owned(), json!("en"))]),
        doc_id: Some("d1".to_owned()),
        chunk_id: Some("c1".to_owned()),
    };
    let value = serde_json::to_value(&original).unwrap();
    assert_eq!(value["doc_id"], json!("d1"));
    assert_eq!(value["chunk_id"], json!("c1"));
    let restored: VectorRecord = serde_json::from_value(value).unwrap();
    assert_eq!(restored, original);

    // Missing optional payload fields fall back to their defaults.
    let sparse: VectorRecord = serde_json::from_value(json!({"id": "c2"})).unwrap();
    assert_eq!(sparse, record("c2", "", Vec::new()));
}

#[test]
fn vector_query_defaults_and_builders() {
    let query = VectorQuery::new(vec![1.0, 0.0]);
    assert_eq!(query.top_k, DEFAULT_TOP_K);
    assert!(query.filter.is_none());
    assert!(query.score_threshold.is_none());

    let built = query
        .clone()
        .with_top_k(3)
        .with_filter(MetadataFilter::eq("lang", "en"))
        .with_score_threshold(0.25);
    assert_eq!(built.top_k, 3);
    assert_eq!(built.filter, Some(MetadataFilter::eq("lang", "en")));
    assert_eq!(built.score_threshold, Some(0.25));

    let value = serde_json::to_value(&query).unwrap();
    assert_eq!(
        value,
        json!({"vector": [1.0, 0.0], "top_k": 5, "filter": null, "score_threshold": null})
    );
    let restored: VectorQuery = serde_json::from_value(json!({"vector": [1.0]})).unwrap();
    assert_eq!(restored.top_k, DEFAULT_TOP_K);
}

#[test]
fn cosine_similarity_handles_edge_cases() {
    assert!((cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
    assert!((cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]) - 0.0).abs() < 1e-6);
    assert!((cosine_similarity(&[1.0, 0.0], &[-1.0, 0.0]) + 1.0).abs() < 1e-6);
    // Zero vectors, dimension mismatch, and empty inputs score 0.0 instead of NaN.
    assert_eq!(cosine_similarity(&[0.0, 0.0], &[1.0, 1.0]), 0.0);
    assert_eq!(cosine_similarity(&[1.0], &[1.0, 2.0]), 0.0);
    assert_eq!(cosine_similarity(&[], &[]), 0.0);
    assert_eq!(normalize_cosine(0.0), 0.5);
    assert_eq!(normalize_cosine(1.0), 1.0);
    assert_eq!(normalize_cosine(-1.0), 0.0);
}

#[tokio::test]
async fn registry_builtin_lookup_builds_memory_stores() {
    let registry = VectorStoreRegistry::with_builtin();
    assert!(registry.contains("memory"));
    assert!(registry.contains("in_memory"));
    assert!(!registry.contains("milvus"));

    let store = registry.create(&config("memory", "docs")).unwrap();
    assert_eq!(store.name(), "docs");
    assert!(store.is_empty().await.unwrap());

    let alias = registry.create(&config("in_memory", "")).unwrap();
    assert_eq!(alias.name(), "default");
    assert_eq!(alias.list(None).await.unwrap(), Vec::<VectorRecord>::new());
}

#[test]
fn registry_rejects_duplicate_and_blank_providers() {
    let mut registry = VectorStoreRegistry::new();
    let factory: VectorStoreFactory = Arc::new(|_| Ok(Arc::new(InMemoryVectorStore::new())));
    registry.register("memory", Arc::clone(&factory)).unwrap();
    assert!(registry.contains("memory"));

    let duplicate = registry.register(" memory ", factory);
    assert!(
        matches!(&duplicate, Err(Error::Validation(message)) if message.contains("already registered")),
        "unexpected: {duplicate:?}"
    );

    let blank: VectorStoreFactory = Arc::new(|_| Ok(Arc::new(InMemoryVectorStore::new())));
    let error = registry.register("   ", blank).unwrap_err();
    assert!(matches!(error, Error::Validation(_)));
}

#[test]
fn registry_reports_unsupported_and_empty_providers() {
    let empty = VectorStoreRegistry::new();
    let error = create_error(&empty, &config("memory", "docs"));
    assert!(
        matches!(&error, Error::Unsupported(message) if message.contains("available providers are: none")),
        "unexpected: {error:?}"
    );

    let registry = VectorStoreRegistry::with_builtin();
    let error = create_error(&registry, &config("milvus", "docs"));
    assert!(
        matches!(&error, Error::Unsupported(message) if message.contains("milvus") && message.contains("memory")),
        "unexpected: {error:?}"
    );
}
