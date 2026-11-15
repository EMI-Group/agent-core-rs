//! Documents, chunking, loaders, embeddings, vector stores, rerankers, and
//! knowledge bases.
//!
//! Maps to the Python package `openjiuwen.core.retrieval` in the openJiuwen
//! reference implementation. The crate is split into independent layers:
//! [`common`] (shared data model, configs, metadata filters), [`chunking`],
//! [`loaders`] (text/Markdown/JSON/CSV/HTML/URL sources, PDF/DOCX/XLSX/PPTX/ODT
//! readers, images, plus auto dispatch),
//! [`embedding`], [`vector_store`], [`reranker`], and [`knowledge_base`], the
//! pipeline that composes them.
//!
//! Every loader also accepts in-memory bytes: [`DocumentBytes`] pairs a display
//! file name with the bytes to parse, [`LoadOptions`] overrides the document id
//! and title, and [`DocumentLoader::load_bytes`] runs the same parse as the
//! path-based [`DocumentLoader::load`].
//! The `loaders::fixtures` module exposes the public document builders that turn
//! paragraphs, rows, or slides into exactly those bytes, so binary ingestion can
//! be exercised with no office suite and no temporary files.
//!
//! Retrieval channels: [`sparse`] adds the additive [`SparseVector`]
//! representation and a self-contained BM25 lexical index ([`Bm25Index`],
//! [`Bm25Config`], [`Bm25Query`]) so sparse search works without an external
//! engine, and [`retriever`] exposes the [`Retriever`] trait with dense,
//! sparse, hybrid, graph, and agentic implementations plus reciprocal rank
//! fusion ([`rrf_fusion`]); [`RankConfig`] selects the weighted or RRF ranker
//! those fused lists can be scored with ([`HybridRetriever::with_rank_config`]).
//! [`rewrite`] turns a conversation into a standalone
//! query, either heuristically ([`HeuristicQueryRewriter`]) or through the
//! model seam ([`ModelQueryRewriter`]).
//!
//! The model seam itself is [`model`]: [`CompletionModel`] is one prompt in,
//! one text reply out, with [`StaticCompletionModel`] as the deterministic
//! offline implementation and [`extract_json_object`] as the shared reply
//! parser; an adapter over `openjiuwen_foundation::Model` belongs in an upper
//! crate.
//!
//! Graph retrieval lives in [`graph`]: the [`Triple`]/[`Entity`] data models,
//! the [`GraphStore`] trait with the adjacency-indexed [`InMemoryGraphStore`],
//! and the deterministic, model-free [`RuleTripleExtractor`] behind the
//! [`TripleExtractor`] seam. A graph run reports what it did through
//! [`GraphExpansionReport`]: failed steps are recordable degradations that also
//! flag the hits with [`GRAPH_DEGRADED_FIELD`], while genuine emptiness is a
//! clean run.
//!
//! Multimodal ingestion lives in [`multimodal`]: [`MultimodalDocument`] with
//! typed [`ContentPart`]s and [`Modality`]s, [`ImageLoader`] with the offline
//! [`OfflineCaptioner`], and [`TableData`] with its data-URI helpers
//! ([`to_data_uri`], [`parse_data_uri`]).
//!
//! [`parsing`] is a pure-Rust document layer: a DEFLATE/ZIP reader
//! ([`inflate()`], [`zlib_decompress`], [`ZipArchive`]), a stored-entry ZIP
//! writer ([`ZipWriter`]) that builds archives without a compression crate, and
//! the DOCX, XLSX, PPTX, ODT, and PDF readers ([`DocxParser`], [`XlsxParser`],
//! [`PptxParser`], [`OdtParser`], [`PdfParser`]) plus the tolerant XML reader
//! they share ([`XmlDocument`]).
//! It reports typed
//! [`Error::Unsupported`](openjiuwen_common::error::Error::Unsupported) for
//! encrypted PDFs and compressed object streams instead of degrading silently;
//! image-only PDFs have no OCR, so they parse to empty text with
//! `has_text() == false`.
//!
//! Every capability here runs offline and deterministically: embeddings,
//! retrieval, rewriting, and extraction have rule-based or hash-based defaults,
//! network clients are opt-in, and the model seams are traits rather than
//! vendor SDK calls.
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
pub mod graph;
pub mod knowledge_base;
pub mod loaders;
pub mod model;
pub mod multimodal;
pub mod parsing;
pub mod reranker;
pub mod retriever;
pub mod rewrite;
pub mod sparse;
pub mod vector_store;

// Shared data model, configs, and constants.
pub use common::{
    parse_index_type, Document, EmbeddingConfig, IndexConfig, KnowledgeBaseConfig, Metadata,
    MetadataFilter, MultiKBRetrievalResult, RerankerConfig, RetrievalConfig, RetrievalResult,
    SearchResult, TextChunk, VectorStoreConfig, DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE,
    DEFAULT_DISTANCE_METRIC, DEFAULT_EMBEDDING_BATCH_SIZE, DEFAULT_EMBEDDING_TIMEOUT,
    DEFAULT_TOP_K, INDEX_TYPE_BM25, INDEX_TYPE_HYBRID, INDEX_TYPE_VECTOR,
};
// Result-ranking configs and the fixed channel layout they index.
pub use common::{
    RankConfig, RankMethod, RankerArgs, RrfRankConfig, WeightedRankConfig, CONTENT_DENSE,
    CONTENT_SPARSE, DEFAULT_RRF_RANK_K, NAME_DENSE, RANK_CHANNEL_COUNT,
};
// Chunking strategies, optional text preprocessing, and the chunker registry.
pub use chunking::{
    default_no_split, validate_params, CharChunker, ChunkUnit, Chunker, ChunkerFactory,
    ChunkerRegistry, HybridChunker, NoSplitPredicate, PreprocessingPipeline, RecursiveCharChunker,
    SpecialCharacterNormalizer, TextChunker, TextPreprocessor, TokenChunker, TokenCounter,
    URLEmailRemover, WhitespaceNormalizer, WhitespaceTokenCounter,
};
// Document loaders and the extension-based dispatch registry.
pub use loaders::{
    extract_title, html_to_text, AutoLoader, CsvLoader, DocumentBytes, DocumentLoader, HtmlLoader,
    JsonLoader, LoadOptions, LoaderRegistry, MarkdownLoader, OdtLoader, PptxLoader, TextLoader,
    UrlLoader,
};
// Embedding models: OpenAI-compatible client and the deterministic hash fake.
pub use embedding::{validate_query, validate_texts, Embedding, HashEmbedding, OpenAiEmbedding};
// Vector stores, queries, and the provider registry.
pub use vector_store::{
    cosine_similarity, normalize_cosine, InMemoryVectorStore, VectorQuery, VectorRecord,
    VectorStore, VectorStoreFactory, VectorStoreRegistry, CHUNK_ID_FIELD, DOC_ID_FIELD,
};
// Sparse representation and the BM25 lexical index.
pub use sparse::{tokenize, Bm25Config, Bm25Index, Bm25Query, Bm25Statistics, SparseVector};
// Retrievers (dense/sparse/hybrid/graph/agentic) and rank fusion.
pub use retriever::{
    deduplicate, rrf_fusion, rrf_fusion_channels, rrf_fusion_default, rrf_fusion_weighted,
    weighted_fusion, AgenticConfig, AgenticRetriever, AgenticStopReason, AgenticTrace,
    GraphRetriever, HybridRetriever, RetrieveMode, Retriever, SparseRetriever, VectorRetriever,
    DEFAULT_RRF_K,
};
// Graph-expansion diagnostics: what an expansion did, and which steps failed.
pub use retriever::graph::{
    GraphDegradation, GraphExpansionReport, GraphStage, GRAPH_DEGRADED_FIELD,
};
// Query rewriting: deterministic heuristics and the model-backed rewriter.
pub use rewrite::{
    HeuristicQueryRewriter, HeuristicRewriteConfig, ModelQueryRewriter, ModelRewriteConfig,
    QueryRewriter, RewriteOutput, RewriteRequest, TypoFix,
};
// The completion-model seam shared by model-backed retrieval components.
pub use model::{extract_json_object, CompletionModel, StaticCompletionModel};
// Triples, rule-based extraction, and the in-memory graph store.
pub use graph::{
    Entity, GraphStatistics, GraphStore, InMemoryGraphStore, RuleExtractionConfig,
    RuleTripleExtractor, Triple, TripleBeam, TripleExtractor, TripleMemory, TripleQuery,
};
// Multimodal documents, offline image captioning, and table data.
pub use multimodal::{
    parse_data_uri, to_data_uri, ContentPart, ImageCaptioner, ImageLoader, Modality,
    MultimodalDocument, OfflineCaptioner, TableData,
};
// Pure-Rust containers (DEFLATE/ZIP, XML) and the office/PDF readers.
pub use parsing::{
    adler32, crc32, inflate, parse_xml, zlib_decompress, DocxBlock, DocxDocument, DocxParser,
    OdtDocument, OdtParser, PdfDocument, PdfParser, PptxDocument, PptxParser, PptxSlide,
    XlsxParser, XlsxSheet, XlsxWorkbook, XmlDocument, XmlElement, ZipArchive, ZipEntry, ZipWriter,
    DEFAULT_MAX_OUTPUT_BYTES,
};
// Rerankers: score ordering, the LLM-scorer seam, and the deferred DashScope client.
pub use reranker::{DashScopeReranker, LlmReranker, LlmScorer, Reranker, ScoreReranker};
// The ingestion/retrieval pipeline and multi-KB retrieval.
pub use knowledge_base::{
    retrieve_multi_kb, retrieve_multi_kb_reporting, retrieve_multi_kb_strict,
    retrieve_multi_kb_with_source, retrieve_multi_kb_with_source_reporting,
    retrieve_multi_kb_with_source_strict, KnowledgeBaseStatistics, MultiKbFailure, MultiKbReport,
    MultiKbSourceReport, SimpleKnowledgeBase,
};

/// The most commonly used retrieval types, traits, and default implementations.
///
/// Curated rather than globbed: every item here is a stable entry point, so
/// `use openjiuwen_retrieval::prelude::*;` reads as the intended starting set
/// without pulling in the whole crate or creating name collisions.
pub mod prelude {
    pub use crate::knowledge_base::SimpleKnowledgeBase;
    pub use crate::{
        AutoLoader, CharChunker, Chunker, Document, DocumentLoader, Embedding, GraphStore,
        HashEmbedding, InMemoryGraphStore, InMemoryVectorStore, Metadata, MetadataFilter, Reranker,
        RetrievalResult, RetrieveMode, Retriever, ScoreReranker, SearchResult, TextChunk,
        TextChunker, Triple, VectorStore,
    };
}

/// Returns the crate name.
#[must_use]
pub fn crate_name() -> &'static str {
    "openjiuwen-retrieval"
}
