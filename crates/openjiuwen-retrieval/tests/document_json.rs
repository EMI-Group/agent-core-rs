#![allow(clippy::unwrap_used, clippy::expect_used)]
//! JSON wire-format tests for the `common` data model.
//!
//! Ports the serialization expectations of the Python models in
//! `openjiuwen/core/retrieval/common/document.py`,
//! `openjiuwen/core/retrieval/common/retrieval_result.py`, and
//! `openjiuwen/core/retrieval/common/config.py`.
//! Every assertion is offline and deterministic: generated ids are only checked
//! for shape and uniqueness, never for a concrete value.

use std::time::Duration;

use openjiuwen_retrieval::common::{
    Document, EmbeddingConfig, KnowledgeBaseConfig, Metadata, MetadataFilter,
    MultiKBRetrievalResult, RerankerConfig, RetrievalConfig, RetrievalResult, SearchResult,
    TextChunk, VectorStoreConfig,
};
use serde_json::{json, Value};

/// Returns the serialized object keys, sorted, so key sets can be compared exactly.
fn object_keys(value: &Value) -> Vec<String> {
    let mut keys: Vec<String> = value
        .as_object()
        .expect("serialized value must be a JSON object")
        .keys()
        .cloned()
        .collect();
    keys.sort_unstable();
    keys
}

/// Asserts the value looks like a dashed `uuid4` string (Python's `str(uuid.uuid4())`).
fn assert_dashed_uuid(value: &str) {
    assert_eq!(value.len(), 36, "uuid must have 36 characters: {value}");
    for (index, byte) in value.as_bytes().iter().enumerate() {
        if matches!(index, 8 | 13 | 18 | 23) {
            assert_eq!(*byte, b'-', "uuid must be dashed: {value}");
        } else {
            assert!(
                byte.is_ascii_hexdigit(),
                "uuid must be hexadecimal: {value}"
            );
        }
    }
}

/// A representative metadata map used by the filter tests.
fn sample_metadata() -> Metadata {
    Metadata::from_iter([
        ("author".to_owned(), json!("alice")),
        ("year".to_owned(), json!(2024)),
        ("tags".to_owned(), json!(["rust", "search"])),
    ])
}

fn parse_filter(value: Value) -> MetadataFilter {
    serde_json::from_value(value).expect("filter JSON must deserialize")
}

#[test]
fn document_serializes_exactly_the_python_keys() {
    // Python: `Document.model_dump()` -> {"id_", "text", "metadata"}.
    let doc = Document::with_id("doc-1", "hello")
        .metadata_entry("author", json!("alice"))
        .metadata_entry("year", json!(2024));
    let value = serde_json::to_value(&doc).unwrap();

    assert_eq!(object_keys(&value), vec!["id_", "metadata", "text"]);
    assert_eq!(value["id_"], json!("doc-1"));
    assert_eq!(value["text"], json!("hello"));
    assert_eq!(value["metadata"], json!({"author": "alice", "year": 2024}));
    // The Rust field name never leaks onto the wire.
    assert!(value.get("id").is_none());
}

#[test]
fn document_json_round_trips() {
    let doc = Document::with_id("doc-1", "hello world").metadata_entry("lang", json!("en"));
    let value = serde_json::to_value(&doc).unwrap();
    let restored: Document = serde_json::from_value(value).unwrap();

    assert_eq!(restored.id, "doc-1");
    assert_eq!(restored.text, "hello world");
    assert_eq!(restored.metadata, doc.metadata);
    assert_eq!(restored, doc);

    // Text is carried verbatim, including newlines and non-ASCII characters.
    let multiline = Document::with_id("doc-2", "line one\nline two — ünïcode");
    let text = serde_json::to_string(&multiline).unwrap();
    let restored: Document = serde_json::from_str(&text).unwrap();
    assert_eq!(restored, multiline);
}

#[test]
fn document_accepts_python_id_alias() {
    // Python callers occasionally pass `id`; it must be accepted on input and
    // still be emitted as `id_`.
    let doc: Document = serde_json::from_value(json!({"id": "alias-id", "text": "hi"})).unwrap();
    assert_eq!(doc.id, "alias-id");
    assert_eq!(doc.text, "hi");
    assert_eq!(
        serde_json::to_value(&doc).unwrap()["id_"],
        json!("alias-id")
    );

    // The canonical `id_` spelling is accepted as well.
    let doc: Document = serde_json::from_value(json!({"id_": "canonical", "text": "hi"})).unwrap();
    assert_eq!(doc.id, "canonical");
}

#[test]
fn document_defaults_ids_and_metadata() {
    // A missing `id_` generates a dashed uuid4 and missing metadata stays empty.
    let first: Document = serde_json::from_value(json!({"text": "hi"})).unwrap();
    let second: Document = serde_json::from_value(json!({"text": "hi"})).unwrap();
    assert_dashed_uuid(&first.id);
    assert_dashed_uuid(&second.id);
    assert_ne!(first.id, second.id);
    assert!(first.metadata.is_empty());
    assert!(second.metadata.is_empty());

    // `Document::new` behaves the same way.
    let a = Document::new("a");
    let b = Document::new("b");
    assert_dashed_uuid(&a.id);
    assert_dashed_uuid(&b.id);
    assert_ne!(a.id, b.id);
    assert!(a.metadata.is_empty());

    // Front matter without `metadata` in the payload: the field is not required.
    let without_metadata: Document =
        serde_json::from_value(json!({"id_": "doc-3", "text": "t"})).unwrap();
    assert_eq!(without_metadata.metadata, Metadata::new());
    assert_eq!(
        object_keys(&serde_json::to_value(&without_metadata).unwrap()),
        vec!["id_", "metadata", "text"]
    );
}

#[test]
fn document_tolerates_unknown_fields() {
    let value = json!({
        "id_": "doc-1",
        "text": "body",
        "metadata": {"k": 1},
        "permissions": ["read"],
        "future": {"nested": true}
    });
    let doc: Document = serde_json::from_value(value).unwrap();
    assert_eq!(doc.id, "doc-1");
    assert_eq!(doc.text, "body");
    assert_eq!(
        doc.metadata,
        Metadata::from_iter([("k".to_owned(), json!(1))])
    );

    // Unknown fields are dropped on the next serialization.
    let serialized = serde_json::to_value(&doc).unwrap();
    assert_eq!(object_keys(&serialized), vec!["id_", "metadata", "text"]);
    assert!(serialized.get("permissions").is_none());
}

#[test]
fn text_chunk_serializes_python_keys_with_null_embedding() {
    let chunk = TextChunk::from_document(
        &Document::with_id("doc-1", "src"),
        "piece",
        Some("chunk-1".to_owned()),
    );
    let value = serde_json::to_value(&chunk).unwrap();

    assert_eq!(
        object_keys(&value),
        vec!["doc_id", "embedding", "id_", "metadata", "text"]
    );
    assert_eq!(value["id_"], json!("chunk-1"));
    assert_eq!(value["text"], json!("piece"));
    assert_eq!(value["doc_id"], json!("doc-1"));
    assert_eq!(value["metadata"], json!({}));
    // Python: `embedding: Optional[List[float]] = None` -> JSON null.
    assert_eq!(value["embedding"], Value::Null);
}

#[test]
fn text_chunk_embedding_round_trips() {
    let chunk = TextChunk::from_document(
        &Document::with_id("doc-1", "src"),
        "piece",
        Some("chunk-1".to_owned()),
    )
    .with_embedding(vec![0.5, -1.25, 0.0]);
    let value = serde_json::to_value(&chunk).unwrap();
    assert_eq!(value["embedding"], json!([0.5, -1.25, 0.0]));

    let restored: TextChunk = serde_json::from_value(value).unwrap();
    assert_eq!(restored.embedding, Some(vec![0.5, -1.25, 0.0]));
    assert_eq!(restored, chunk);

    // A null embedding deserializes back to `None`.
    let restored: TextChunk = serde_json::from_value(json!({
        "id_": "chunk-1",
        "text": "piece",
        "doc_id": "doc-1",
        "embedding": null
    }))
    .unwrap();
    assert_eq!(restored.embedding, None);
    assert!(restored.metadata.is_empty());
}

#[test]
fn text_chunk_from_document_copies_doc_id_and_metadata() {
    let doc = Document::with_id("doc-7", "source").metadata_entry("source", json!("file"));
    let chunk = TextChunk::from_document(&doc, "piece", None);

    assert_eq!(chunk.doc_id, "doc-7");
    assert_eq!(chunk.text, "piece");
    assert_eq!(chunk.metadata, doc.metadata);
    assert!(chunk.embedding.is_none());
    // Python: `id_ if id_ else str(uuid.uuid4())`.
    assert_dashed_uuid(&chunk.id);

    // Two derived chunks get distinct generated ids.
    let other = TextChunk::from_document(&doc, "piece", None);
    assert_ne!(chunk.id, other.id);

    // An explicit id is preserved verbatim.
    let explicit = TextChunk::from_document(&doc, "piece", Some("chunk-9".to_owned()));
    assert_eq!(explicit.id, "chunk-9");
}

#[test]
fn text_chunk_with_embedding_sets_the_vector() {
    let chunk = TextChunk::from_document(
        &Document::with_id("doc-1", "src"),
        "piece",
        Some("chunk-1".to_owned()),
    );
    assert_eq!(chunk.embedding, None);

    let embedded = chunk.clone().with_embedding(vec![1.0, 0.0]);
    assert_eq!(embedded.embedding, Some(vec![1.0, 0.0]));
    assert_eq!(embedded.id, chunk.id);
    assert_eq!(embedded.doc_id, chunk.doc_id);
    // The original chunk is untouched.
    assert_eq!(chunk.embedding, None);
}

#[test]
fn search_result_keys_and_round_trip() {
    let result = SearchResult {
        id: "chunk-1".to_owned(),
        text: "hello".to_owned(),
        score: 0.75,
        metadata: sample_metadata(),
    };
    let value = serde_json::to_value(&result).unwrap();

    assert_eq!(object_keys(&value), vec!["id", "metadata", "score", "text"]);
    assert_eq!(value["id"], json!("chunk-1"));
    assert_eq!(value["score"], json!(0.75));
    // Results use Python's `id`, not the document `id_` spelling.
    assert!(value.get("id_").is_none());

    let restored: SearchResult = serde_json::from_value(value).unwrap();
    assert_eq!(restored, result);

    // `metadata` defaults to an empty object.
    let minimal: SearchResult =
        serde_json::from_value(json!({"id": "c", "text": "t", "score": 0.0})).unwrap();
    assert_eq!(minimal.metadata, Metadata::new());
}

#[test]
fn retrieval_result_ids_serialize_as_null_or_values() {
    let minimal = RetrievalResult::default();
    let value = serde_json::to_value(&minimal).unwrap();
    assert_eq!(
        object_keys(&value),
        vec!["chunk_id", "doc_id", "metadata", "score", "text"]
    );
    assert_eq!(value["doc_id"], Value::Null);
    assert_eq!(value["chunk_id"], Value::Null);

    let result = RetrievalResult {
        text: "hello".to_owned(),
        score: 0.5,
        metadata: sample_metadata(),
        doc_id: Some("doc-1".to_owned()),
        chunk_id: Some("chunk-1".to_owned()),
    };
    let value = serde_json::to_value(&result).unwrap();
    assert_eq!(value["doc_id"], json!("doc-1"));
    assert_eq!(value["chunk_id"], json!("chunk-1"));

    let restored: RetrievalResult = serde_json::from_value(value).unwrap();
    assert_eq!(restored, result);

    // Absent ids also parse as `None`, matching Python's optional fields.
    let restored: RetrievalResult =
        serde_json::from_value(json!({"text": "t", "score": 1.0})).unwrap();
    assert_eq!(restored.doc_id, None);
    assert_eq!(restored.chunk_id, None);
    assert_eq!(restored.metadata, Metadata::new());
}

#[test]
fn multi_kb_retrieval_result_keys_and_round_trip() {
    let result = MultiKBRetrievalResult {
        text: "hello".to_owned(),
        score: 0.5,
        raw_score: 1.5,
        raw_score_scaled: 0.281_25,
        kb_ids: vec!["kb-1".to_owned(), "kb-2".to_owned()],
        metadata: sample_metadata(),
    };
    let value = serde_json::to_value(&result).unwrap();

    assert_eq!(
        object_keys(&value),
        vec![
            "kb_ids",
            "metadata",
            "raw_score",
            "raw_score_scaled",
            "score",
            "text"
        ]
    );
    assert_eq!(value["kb_ids"], json!(["kb-1", "kb-2"]));
    assert_eq!(value["raw_score"], json!(1.5));
    assert_eq!(value["raw_score_scaled"], json!(0.28125));

    let restored: MultiKBRetrievalResult = serde_json::from_value(value).unwrap();
    assert_eq!(restored, result);

    // `kb_ids` and `metadata` default to empty collections.
    let minimal: MultiKBRetrievalResult = serde_json::from_value(json!({
        "text": "t",
        "score": 1.0,
        "raw_score": 1.0,
        "raw_score_scaled": 1.0
    }))
    .unwrap();
    assert_eq!(minimal.kb_ids, Vec::<String>::new());
    assert_eq!(minimal.metadata, Metadata::new());
}

#[test]
fn knowledge_base_config_defaults_match_python() {
    let config = KnowledgeBaseConfig::new("kb");
    assert_eq!(config.kb_id, "kb");
    assert_eq!(config.index_type, "hybrid");
    assert!(!config.use_graph);
    assert_eq!(config.chunk_size, 512);
    assert_eq!(config.chunk_overlap, 50);
    assert!(!config.use_caption_for_images);

    // Python's pydantic defaults apply when the fields are missing from JSON.
    let parsed: KnowledgeBaseConfig = serde_json::from_value(json!({"kb_id": "kb"})).unwrap();
    assert_eq!(parsed, config);
    let value = serde_json::to_value(&config).unwrap();
    assert_eq!(value["index_type"], json!("hybrid"));
    assert_eq!(value["chunk_size"], json!(512));
    assert_eq!(value["use_caption_for_images"], json!(false));
}

#[test]
fn retrieval_config_defaults() {
    let config = RetrievalConfig::default();
    assert_eq!(config.top_k, 5);
    assert!(!config.agentic);
    assert!(!config.graph_expansion);
    assert_eq!(config.score_threshold, None);
    assert_eq!(config.use_graph, None);
    assert_eq!(config.filters, None);

    let parsed: RetrievalConfig = serde_json::from_value(json!({})).unwrap();
    assert_eq!(parsed, config);

    // A configured query keeps its filter through serde.
    let configured = RetrievalConfig::default()
        .with_top_k(3)
        .with_score_threshold(0.25)
        .with_filter(MetadataFilter::eq("author", "alice"));
    let value = serde_json::to_value(&configured).unwrap();
    assert_eq!(value["top_k"], json!(3));
    assert_eq!(value["filters"], json!({"author": "alice"}));
    let restored: RetrievalConfig = serde_json::from_value(value).unwrap();
    assert_eq!(restored, configured);
}

#[test]
fn vector_store_config_defaults() {
    let config: VectorStoreConfig = serde_json::from_value(json!({})).unwrap();
    assert_eq!(config.distance_metric, "cosine");
    assert_eq!(config.store_provider, "");
    assert_eq!(config.database_name, "");
    assert_eq!(config.collection_name, "");

    let configured: VectorStoreConfig = serde_json::from_value(json!({
        "store_provider": "memory",
        "database_name": "db",
        "collection_name": "docs",
        "distance_metric": "dot"
    }))
    .unwrap();
    assert_eq!(configured.distance_metric, "dot");
    let value = serde_json::to_value(&configured).unwrap();
    assert_eq!(value["collection_name"], json!("docs"));
    assert_eq!(value["store_provider"], json!("memory"));
}

#[test]
fn reranker_config_accepts_model_alias() {
    // Python serializes the field as `model`; the Rust field is `model_name`.
    let parsed: RerankerConfig =
        serde_json::from_value(json!({"api_base": "https://rerank", "model": "bge-m3"})).unwrap();
    assert_eq!(parsed.model_name, "bge-m3");
    assert_eq!(parsed.api_base, "https://rerank");
    assert_eq!(parsed.timeout, Duration::from_secs(10));
    assert!(parsed.api_key.is_none());

    // Output uses the canonical `model_name` key only.
    let value = serde_json::to_value(&parsed).unwrap();
    assert_eq!(value["model_name"], json!("bge-m3"));
    assert!(value.get("model").is_none());

    let restored: RerankerConfig = serde_json::from_value(value).unwrap();
    assert_eq!(restored, parsed);
}

#[test]
fn embedding_config_timeout_round_trips_through_serde() {
    // Whole seconds serialize as an integer and come back as a `Duration`.
    let parsed: EmbeddingConfig = serde_json::from_value(json!({
        "model_name": "text-embedding-3-small",
        "base_url": "https://api.example/v1",
        "timeout": 30
    }))
    .unwrap();
    assert_eq!(parsed.timeout, Duration::from_secs(30));
    assert_eq!(serde_json::to_value(&parsed).unwrap()["timeout"], json!(30));

    // Fractional seconds survive the round trip as milliseconds.
    let fractional: EmbeddingConfig = serde_json::from_value(json!({
        "model_name": "m",
        "base_url": "u",
        "timeout": 2.5
    }))
    .unwrap();
    assert_eq!(fractional.timeout, Duration::from_millis(2500));
    let value = serde_json::to_value(&fractional).unwrap();
    assert_eq!(value["timeout"], json!(2.5));
    let restored: EmbeddingConfig = serde_json::from_value(value).unwrap();
    assert_eq!(restored, fractional);

    // The Python defaults apply when the timeout is absent.
    let defaults: EmbeddingConfig =
        serde_json::from_value(json!({"model_name": "m", "base_url": "u"})).unwrap();
    assert_eq!(defaults.timeout, Duration::from_secs(60));
}

#[test]
fn configs_tolerate_unknown_json_fields() {
    // Configurations are forward compatible: unknown keys are ignored.
    let kb: KnowledgeBaseConfig = serde_json::from_value(json!({
        "kb_id": "kb",
        "chunk_size": 128,
        "future_flag": true,
        "nested": {"x": 1}
    }))
    .unwrap();
    assert_eq!(kb.kb_id, "kb");
    assert_eq!(kb.chunk_size, 128);

    let store: VectorStoreConfig = serde_json::from_value(json!({
        "store_provider": "milvus",
        "collection_name": "docs",
        "extra": [1, 2, 3]
    }))
    .unwrap();
    assert_eq!(store.store_provider, "milvus");
    assert_eq!(store.distance_metric, "cosine");

    let reranker: RerankerConfig = serde_json::from_value(json!({
        "api_base": "https://rerank",
        "model_name": "bge-m3",
        "vendor_extension": "ignored"
    }))
    .unwrap();
    assert_eq!(reranker.model_name, "bge-m3");
}

#[test]
fn metadata_filter_equality_matches_and_round_trips() {
    // Python shape: a plain object is a conjunction of equalities.
    let filter = parse_filter(json!({"author": "alice"}));
    let metadata = sample_metadata();
    assert!(filter.matches(&metadata));
    assert!(!filter.matches(&Metadata::from_iter([("author".to_owned(), json!("bob"))])));
    assert!(!filter.matches(&Metadata::new()));

    // It round-trips to the same canonical object.
    let value = serde_json::to_value(&filter).unwrap();
    assert_eq!(value, json!({"author": "alice"}));
    let restored: MetadataFilter = serde_json::from_value(value).unwrap();
    assert_eq!(restored, filter);

    // Multiple fields require all of them to match.
    let both = parse_filter(json!({"author": "alice", "year": 2024}));
    assert!(both.matches(&metadata));
    assert!(!both.matches(&Metadata::from_iter([
        ("author".to_owned(), json!("alice")),
        ("year".to_owned(), json!(2023)),
    ])));
}

#[test]
fn metadata_filter_comparison_operator() {
    let filter = parse_filter(json!({"year": {"$gte": 2024}}));
    let metadata = sample_metadata();
    assert!(filter.matches(&metadata));
    assert!(filter.matches(&Metadata::from_iter([("year".to_owned(), json!(2025))])));
    assert!(!filter.matches(&Metadata::from_iter([("year".to_owned(), json!(2023))])));

    // Canonical JSON stays compact for a single-operator field.
    let value = serde_json::to_value(&filter).unwrap();
    assert_eq!(value, json!({"year": {"$gte": 2024}}));

    // Multiple operators on one field are AND-ed.
    let range = parse_filter(json!({"year": {"$gte": 2020, "$lt": 2030}}));
    assert!(range.matches(&metadata));
    assert_eq!(
        serde_json::to_value(&range).unwrap(),
        json!({"$and": [{"year": {"$gte": 2020}}, {"year": {"$lt": 2030}}]})
    );
}

#[test]
fn metadata_filter_or_matches_any_branch() {
    let filter = parse_filter(json!({"$or": [{"author": "bob"}, {"year": {"$gte": 2024}}]}));
    let metadata = sample_metadata();
    assert!(filter.matches(&metadata));
    assert!(filter.matches(&Metadata::from_iter([("author".to_owned(), json!("bob"))])));
    assert!(!filter.matches(&Metadata::from_iter([
        ("author".to_owned(), json!("carol")),
        ("year".to_owned(), json!(1999)),
    ])));

    let value = serde_json::to_value(&filter).unwrap();
    assert_eq!(
        value,
        json!({"$or": [{"author": "bob"}, {"year": {"$gte": 2024}}]})
    );
    let restored: MetadataFilter = serde_json::from_value(value).unwrap();
    assert_eq!(restored, filter);
}

#[test]
fn metadata_filter_not_inverts() {
    let filter = parse_filter(json!({"$not": {"author": "bob"}}));
    let metadata = sample_metadata();
    assert!(filter.matches(&metadata));
    assert!(!filter.matches(&Metadata::from_iter([("author".to_owned(), json!("bob"))])));

    let value = serde_json::to_value(&filter).unwrap();
    assert_eq!(value, json!({"$not": {"author": "bob"}}));
    let restored: MetadataFilter = serde_json::from_value(value).unwrap();
    assert_eq!(restored, filter);

    // A `$not` wrapping a comparison operator inverts it too.
    let negated = parse_filter(json!({"$not": {"year": {"$gte": 2024}}}));
    assert!(!negated.matches(&metadata));
    assert!(negated.matches(&Metadata::from_iter([("year".to_owned(), json!(2000))])));
}
