//! Documents, chunking, embeddings, vector stores, rerankers, and knowledge base.
//!
//! Maps to the Python package `openjiuwen.core.retrieval` in the openJiuwen
//! reference implementation. The crate is split into independent layers:
//! [`common`] (shared data model, configs, metadata filters), [`chunking`],
//! [`embedding`], [`vector_store`], and [`reranker`].
#![forbid(unsafe_code)]
#![allow(
    clippy::module_name_repetitions,
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::cast_possible_wrap,
    clippy::cast_abs_to_unsigned
)]

pub mod chunking;
pub mod common;
pub mod embedding;
pub mod reranker;
pub mod vector_store;

pub use chunking::{
    validate_params, CharChunker, Chunker, ChunkerFactory, ChunkerRegistry, RecursiveCharChunker,
    TokenChunker, TokenCounter, WhitespaceTokenCounter,
};
pub use common::{
    Document, EmbeddingConfig, IndexConfig, KnowledgeBaseConfig, Metadata, MetadataFilter,
    MultiKBRetrievalResult, RetrievalConfig, RetrievalResult, RerankerConfig, SearchResult,
    TextChunk, VectorStoreConfig, DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE,
    DEFAULT_DISTANCE_METRIC, DEFAULT_EMBEDDING_BATCH_SIZE, DEFAULT_EMBEDDING_TIMEOUT,
    DEFAULT_TOP_K, INDEX_TYPE_BM25, INDEX_TYPE_HYBRID, INDEX_TYPE_VECTOR,
};
pub use embedding::{validate_query, validate_texts, Embedding, HashEmbedding, OpenAiEmbedding};
pub use reranker::{LlmReranker, LlmScorer, Reranker, ScoreReranker};
pub use vector_store::{
    cosine_similarity, normalize_cosine, InMemoryVectorStore, VectorQuery, VectorRecord,
    VectorStore, VectorStoreFactory, VectorStoreRegistry, CHUNK_ID_FIELD, DOC_ID_FIELD,
};

/// Returns the crate name.
#[must_use]
pub fn crate_name() -> &'static str {
    "openjiuwen-retrieval"
}
