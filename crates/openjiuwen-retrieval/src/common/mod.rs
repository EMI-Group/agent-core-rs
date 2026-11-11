//! Shared data model, result types, configurations, and metadata filters.
//!
//! Port of the Python `openjiuwen/core/retrieval/common/` package plus the
//! foundation `Document` model. Everything here is a plain serializable value:
//! no I/O and no dependency on the other modules of this crate.

pub mod config;
pub mod document;
pub mod filter;
pub mod result;

pub use config::{
    EmbeddingConfig, IndexConfig, KnowledgeBaseConfig, RerankerConfig, RetrievalConfig,
    VectorStoreConfig, DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, DEFAULT_DISTANCE_METRIC,
    DEFAULT_EMBEDDING_BATCH_SIZE, DEFAULT_EMBEDDING_TIMEOUT, DEFAULT_TOP_K, INDEX_TYPE_BM25,
    INDEX_TYPE_HYBRID, INDEX_TYPE_VECTOR,
};
pub use document::{Document, Metadata, TextChunk};
pub use filter::MetadataFilter;
pub use result::{MultiKBRetrievalResult, RetrievalResult, SearchResult};

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn module_root_exposes_the_public_model() {
        let doc = Document::with_id("doc-1", "hello world")
            .with_metadata(Metadata::from_iter([("lang".to_owned(), json!("en"))]));
        let chunk = TextChunk::from_document(&doc, "hello", Some("chunk-1".to_owned()));
        let result = SearchResult {
            id: chunk.id.clone(),
            text: chunk.text.clone(),
            score: 0.9,
            metadata: chunk.metadata.clone(),
        };
        let filter = MetadataFilter::and([
            MetadataFilter::eq("lang", "en"),
            MetadataFilter::not(MetadataFilter::eq("lang", "fr")),
        ]);
        assert!(filter.matches(&result.metadata));

        let retrieval = RetrievalResult {
            text: result.text,
            score: result.score,
            metadata: result.metadata,
            doc_id: Some(doc.id),
            chunk_id: Some(result.id),
        };
        let json = serde_json::to_string(&retrieval).unwrap();
        let restored: RetrievalResult = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, retrieval);
    }

    #[test]
    fn defaults_and_constants_are_wired() {
        assert_eq!(DEFAULT_TOP_K, 5);
        assert_eq!(DEFAULT_CHUNK_SIZE, 512);
        assert_eq!(DEFAULT_CHUNK_OVERLAP, 50);
        assert_eq!(INDEX_TYPE_HYBRID, "hybrid");
        assert_eq!(DEFAULT_EMBEDDING_BATCH_SIZE, 8);
        assert_eq!(DEFAULT_DISTANCE_METRIC, "cosine");
        assert_eq!(RetrievalConfig::default().top_k, DEFAULT_TOP_K);
        assert_eq!(KnowledgeBaseConfig::new("kb").index_type, INDEX_TYPE_HYBRID);
        let results: MultiKBRetrievalResult = MultiKBRetrievalResult::default();
        assert_eq!(results.kb_ids, Vec::<String>::new());
    }
}
