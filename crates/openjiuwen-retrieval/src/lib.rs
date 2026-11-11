//! Documents, chunking, loaders, embeddings, vector stores, rerankers, and
//! knowledge bases.
//!
//! Maps to the Python package `openjiuwen.core.retrieval` in the openJiuwen
//! reference implementation. The crate is split into independent layers:
//! [`common`] (shared data model, configs, metadata filters), [`chunking`],
//! [`loaders`] (text/Markdown/JSON/CSV/HTML/URL sources plus auto dispatch),
//! [`embedding`], [`vector_store`], [`reranker`], and [`knowledge_base`], the
//! pipeline that composes them.
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
pub mod knowledge_base;
pub mod loaders;
pub mod reranker;
pub mod vector_store;

pub use chunking::{
    validate_params, CharChunker, Chunker, ChunkerFactory, ChunkerRegistry, RecursiveCharChunker,
    TokenChunker, TokenCounter, WhitespaceTokenCounter,
};
pub use common::{
    Document, EmbeddingConfig, IndexConfig, KnowledgeBaseConfig, Metadata, MetadataFilter,
    MultiKBRetrievalResult, RerankerConfig, RetrievalConfig, RetrievalResult, SearchResult,
    TextChunk, VectorStoreConfig, DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE,
    DEFAULT_DISTANCE_METRIC, DEFAULT_EMBEDDING_BATCH_SIZE, DEFAULT_EMBEDDING_TIMEOUT,
    DEFAULT_TOP_K, INDEX_TYPE_BM25, INDEX_TYPE_HYBRID, INDEX_TYPE_VECTOR,
};
pub use embedding::{validate_query, validate_texts, Embedding, HashEmbedding, OpenAiEmbedding};
pub use knowledge_base::{
    retrieve_multi_kb, retrieve_multi_kb_with_source, KnowledgeBaseStatistics, SimpleKnowledgeBase,
};
pub use loaders::{
    extract_title, html_to_text, AutoLoader, CsvLoader, DocumentLoader, HtmlLoader, JsonLoader,
    LoaderRegistry, MarkdownLoader, TextLoader, UrlLoader,
};
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
