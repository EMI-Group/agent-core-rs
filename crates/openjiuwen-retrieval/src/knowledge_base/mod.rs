//! The knowledge base pipeline: load → chunk → embed → upsert → search → rerank.
//!
//! Port of the Python `openjiuwen/core/retrieval/simple_knowledge_base.py`
//! `SimpleKnowledgeBase`. Python wires this class to an `IndexManager`, a
//! `Retriever`, and a separate graph index; the Rust port keeps the same
//! composition story with one [`VectorStore`] as the canonical index and
//! performs the search inline, so every stage is visible on a single struct.
//!
//! Retrieval dispatches on [`KnowledgeBaseConfig::index_type`]:
//! `vector` uses the dense channel, `bm25` uses the BM25 channel once a sparse
//! index is enabled, and `hybrid` fuses both channels with reciprocal rank
//! fusion. Optional layers wrap the channel: a [`crate::retriever::GraphRetriever`]
//! when a [`crate::graph::GraphStore`] is attached, and a
//! [`crate::retriever::AgenticRetriever`] when `RetrievalConfig::agentic` is set.
//! See the private `retrieve` submodule for the dispatch itself.
//!
//! The knowledge base owns its collaborators as trait objects, which keeps the
//! pipeline testable offline (see [`crate::embedding::HashEmbedding`] and
//! [`crate::vector_store::InMemoryVectorStore`]).
mod multi;
mod retrieve;

use std::collections::BTreeSet;
use std::sync::Arc;

use indexmap::IndexSet;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::logging::warn;

use crate::chunking::{CharChunker, Chunker};
use crate::common::{
    Document, KnowledgeBaseConfig, MetadataFilter, RetrievalConfig, RetrievalResult, TextChunk,
    INDEX_TYPE_BM25,
};
use crate::embedding::Embedding;
use crate::graph::{GraphStore, RuleTripleExtractor};
use crate::loaders::{is_url, AutoLoader, DocumentLoader};
use crate::reranker::Reranker;
use crate::retriever::agentic::DEFAULT_MAX_ITER;
use crate::retriever::graph::DEFAULT_MAX_HOPS;
use crate::rewrite::QueryRewriter;
use crate::sparse::Bm25Index;
use crate::vector_store::{VectorRecord, VectorStore, DOC_ID_FIELD};

pub use multi::{retrieve_multi_kb, retrieve_multi_kb_with_source};

/// Default iteration budget of the agentic retrieval loop ([`DEFAULT_MAX_ITER`]).
pub const DEFAULT_AGENTIC_ITERATIONS: usize = DEFAULT_MAX_ITER;
/// Default number of graph hops explored per entity ([`DEFAULT_MAX_HOPS`]).
pub const DEFAULT_GRAPH_HOPS: usize = DEFAULT_MAX_HOPS;

/// Statistics describing a [`SimpleKnowledgeBase`].
// A statistics snapshot is a flat record of independent flags by design; the
// Python dictionary it ports has the same shape.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KnowledgeBaseStatistics {
    /// Knowledge base identifier (`KnowledgeBaseConfig::kb_id`).
    pub kb_id: String,
    /// Configured index type (`vector`, `bm25`, or `hybrid`).
    pub index_type: String,
    /// Number of chunks currently stored in the vector store.
    pub chunk_count: usize,
    /// Number of documents added through this knowledge base instance.
    pub document_count: usize,
    /// Whether a reranker is attached.
    pub has_reranker: bool,
    /// Whether a document loader is attached (always `true`; the default is [`AutoLoader`]).
    pub has_loader: bool,
    /// Number of chunks in the BM25 sparse index (`0` when the index is disabled).
    #[serde(default)]
    pub sparse_index_size: usize,
    /// Whether a graph store is attached for triple indexing and expansion.
    #[serde(default)]
    pub has_graph: bool,
    /// Whether a query rewriter was injected (the default heuristic one is not counted).
    #[serde(default)]
    pub has_rewriter: bool,
    /// Whether agentic retrieval is the default for this knowledge base.
    #[serde(default)]
    pub agentic_enabled: bool,
}

/// A single-collection knowledge base composed of a chunker, an embedder, a
/// vector store, an optional reranker, and a document loader.
///
/// `Debug` is not derived because the collaborators are trait objects that are
/// not necessarily `Debug`.
pub struct SimpleKnowledgeBase {
    /// Configuration supplied at construction time.
    config: KnowledgeBaseConfig,
    /// Index holding the embedded chunks.
    store: Arc<dyn VectorStore>,
    /// Encoder used for chunks and queries.
    embedder: Arc<dyn Embedding>,
    /// Splitter turning documents into chunks.
    chunker: Arc<dyn Chunker>,
    /// Source of documents for [`SimpleKnowledgeBase::parse_files`] and
    /// [`SimpleKnowledgeBase::parse_urls`].
    loader: Arc<dyn DocumentLoader>,
    /// Optional reranker applied to search hits.
    reranker: Option<Arc<dyn Reranker>>,
    /// Ids of the documents added through this instance, in insertion order.
    document_ids: RwLock<IndexSet<String>>,
    /// Optional BM25 lexical index, kept in step with the vector store.
    sparse: Option<Arc<RwLock<Bm25Index>>>,
    /// Optional injected rewriter for the agentic path.
    rewriter: Option<Arc<dyn QueryRewriter>>,
    /// Whether agentic retrieval runs by default for this knowledge base.
    agentic: bool,
    /// Iteration budget of the agentic loop.
    agentic_iterations: usize,
    /// Optional triple store for indexing and graph-expanded retrieval.
    graph_store: Option<Arc<dyn GraphStore>>,
    /// Hop count used when expanding graph entities.
    graph_hops: usize,
}

impl SimpleKnowledgeBase {
    /// Builds a knowledge base from its collaborators.
    ///
    /// The loader defaults to [`AutoLoader`] and no reranker is attached; use
    /// [`SimpleKnowledgeBase::with_loader`] and
    /// [`SimpleKnowledgeBase::with_reranker`] to override them. The BM25 sparse
    /// index starts enabled exactly when `config.index_type` is
    /// [`INDEX_TYPE_BM25`]; the other index types
    /// start dense-only and opt in through
    /// [`SimpleKnowledgeBase::with_sparse_index`].
    pub fn new(
        config: KnowledgeBaseConfig,
        store: Arc<dyn VectorStore>,
        embedder: Arc<dyn Embedding>,
        chunker: Arc<dyn Chunker>,
    ) -> Self {
        let sparse = initial_sparse_index(&config.index_type);
        Self {
            config,
            store,
            embedder,
            chunker,
            loader: Arc::new(AutoLoader::new()),
            reranker: None,
            document_ids: RwLock::new(IndexSet::new()),
            sparse,
            rewriter: None,
            agentic: false,
            agentic_iterations: DEFAULT_AGENTIC_ITERATIONS,
            graph_store: None,
            graph_hops: DEFAULT_GRAPH_HOPS,
        }
    }

    /// Builds a knowledge base with a [`CharChunker`] configured from `config`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when `config.chunk_size`/`chunk_overlap` do
    /// not describe a valid window, wrapped with the `"building chunker"` context.
    pub fn from_config(
        config: KnowledgeBaseConfig,
        store: Arc<dyn VectorStore>,
        embedder: Arc<dyn Embedding>,
    ) -> Result<Self> {
        let chunker = CharChunker::new(config.chunk_size, config.chunk_overlap)
            .map_err(|error| error.with_context("building chunker"))?;
        Ok(Self::new(config, store, embedder, Arc::new(chunker)))
    }

    /// Attaches a reranker applied to every search result set.
    #[must_use]
    pub fn with_reranker(mut self, reranker: Arc<dyn Reranker>) -> Self {
        self.reranker = Some(reranker);
        self
    }

    /// Replaces the document loader used by the parse helpers.
    #[must_use]
    pub fn with_loader(mut self, loader: Arc<dyn DocumentLoader>) -> Self {
        self.loader = loader;
        self
    }

    /// Enables or disables the BM25 sparse index.
    ///
    /// The index starts enabled for the `bm25` index type and disabled for
    /// `vector`/`hybrid`. Enabling it later only indexes chunks added afterwards:
    /// the builder is synchronous, so it cannot backfill the vector store.
    #[must_use]
    pub fn with_sparse_index(mut self, enabled: bool) -> Self {
        if enabled {
            self.sparse.get_or_insert_with(new_sparse_index);
        } else {
            self.sparse = None;
        }
        self
    }

    /// Whether a BM25 sparse index is attached (it may still be empty).
    #[must_use]
    pub fn has_sparse_index(&self) -> bool {
        self.sparse.is_some()
    }

    /// Number of chunks in the BM25 sparse index (`0` when it is disabled).
    #[must_use]
    pub fn sparse_index_len(&self) -> usize {
        self.sparse.as_ref().map_or(0, |index| index.read().len())
    }

    /// Injects the rewriter used by the agentic retrieval path.
    ///
    /// Without one, agentic retrieval builds a
    /// [`crate::rewrite::HeuristicQueryRewriter`] whose keyword ranking uses the
    /// document frequency of the knowledge base's own chunk vocabulary.
    #[must_use]
    pub fn with_rewriter(mut self, rewriter: Arc<dyn QueryRewriter>) -> Self {
        self.rewriter = Some(rewriter);
        self
    }

    /// Makes agentic retrieval the default for this knowledge base.
    ///
    /// A per-call [`RetrievalConfig::agentic`] still enables it for a single query.
    #[must_use]
    pub fn with_agentic(mut self, enabled: bool) -> Self {
        self.agentic = enabled;
        self
    }

    /// Sets the agentic iteration budget (default [`DEFAULT_AGENTIC_ITERATIONS`]).
    ///
    /// `0` is rejected at retrieval time with [`Error::Validation`], matching
    /// Python's `max_iter` validation.
    #[must_use]
    pub fn with_agentic_iterations(mut self, iterations: usize) -> Self {
        self.agentic_iterations = iterations;
        self
    }

    /// The agentic iteration budget in force.
    #[must_use]
    pub fn agentic_iterations(&self) -> usize {
        self.agentic_iterations
    }

    /// Attaches a triple store for graph indexing and graph-expanded retrieval.
    ///
    /// Triples are extracted from every added chunk while a store is attached, so
    /// a per-query [`RetrievalConfig::graph_expansion`] always finds them.
    #[must_use]
    pub fn with_graph_store(mut self, graph_store: Arc<dyn GraphStore>) -> Self {
        self.graph_store = Some(graph_store);
        self
    }

    /// Sets the graph hop count (default [`DEFAULT_GRAPH_HOPS`]).
    #[must_use]
    pub fn with_graph_hops(mut self, hops: usize) -> Self {
        self.graph_hops = hops;
        self
    }

    /// The attached triple store, when one is configured.
    #[must_use]
    pub fn graph_store(&self) -> Option<&Arc<dyn GraphStore>> {
        self.graph_store.as_ref()
    }

    /// Returns the configured knowledge base id.
    pub fn id(&self) -> &str {
        &self.config.kb_id
    }

    /// Returns the configuration this knowledge base was built with.
    pub fn config(&self) -> &KnowledgeBaseConfig {
        &self.config
    }

    /// Chunks, embeds, and stores `documents`, returning their ids in input order.
    ///
    /// Documents with a blank id receive a fresh `uuid4`. A document that
    /// produces no chunks is still recorded as known, so statistics stay
    /// accurate, but it contributes nothing to the store. Every stored chunk is
    /// also added to the BM25 sparse index, and mined for triples when a graph
    /// store is attached.
    ///
    /// # Errors
    ///
    /// Returns the chunker's error (context `"chunking documents"`), the
    /// embedder's error (context `"embedding chunks"`), the store's error
    /// (context `"upserting chunks"`), or [`Error::Retrieval`] when the embedder
    /// returns a different number of vectors than chunks.
    // The signature takes the documents by value so blank ids can be filled in
    // place before chunking, which keeps the caller's handles untouched.
    #[allow(clippy::needless_pass_by_value)]
    pub async fn add_documents(&self, documents: Vec<Document>) -> Result<Vec<String>> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }
        let documents = assign_missing_ids(documents);
        let document_ids: Vec<String> = documents.iter().map(|doc| doc.id.clone()).collect();

        let chunks = self
            .chunker
            .chunk_documents(&documents)
            .map_err(|error| error.with_context("chunking documents"))?;
        if chunks.is_empty() {
            self.remember_documents(&document_ids);
            return Ok(document_ids);
        }

        let texts: Vec<String> = chunks.iter().map(|chunk| chunk.text.clone()).collect();
        let vectors = self
            .embedder
            .embed_documents(&texts, None)
            .await
            .map_err(|error| error.with_context("embedding chunks"))?;
        if vectors.len() != chunks.len() {
            return Err(Error::retrieval(format!(
                "embedding model `{}` returned {} vectors for {} chunks",
                self.embedder.model_name(),
                vectors.len(),
                chunks.len()
            )));
        }

        let records: Vec<VectorRecord> = chunks
            .iter()
            .zip(vectors)
            .map(|(chunk, vector)| VectorRecord {
                id: chunk.id.clone(),
                text: chunk.text.clone(),
                vector,
                metadata: chunk.metadata.clone(),
                doc_id: Some(chunk.doc_id.clone()),
                chunk_id: Some(chunk.id.clone()),
            })
            .collect();
        self.store
            .upsert(records)
            .await
            .map_err(|error| error.with_context("upserting chunks"))?;

        self.index_chunks(&chunks).await?;
        self.remember_documents(&document_ids);
        Ok(document_ids)
    }

    /// Wraps every text in a [`Document`] and delegates to
    /// [`SimpleKnowledgeBase::add_documents`].
    ///
    /// # Errors
    ///
    /// See [`SimpleKnowledgeBase::add_documents`].
    // The signature takes the texts by value to mirror `add_documents`; the
    // texts are consumed when the documents are built.
    #[allow(clippy::needless_pass_by_value)]
    pub async fn add_texts(&self, texts: Vec<String>) -> Result<Vec<String>> {
        let documents = texts.into_iter().map(Document::new).collect();
        self.add_documents(documents).await
    }

    /// Deletes every chunk whose `doc_id` is in `doc_ids`.
    ///
    /// The BM25 sparse index drops the deleted chunks with them. Graph triples
    /// are **not** purged: triples have no document ownership beyond the chunk
    /// metadata they carry, and stale triples only widen a later expansion.
    ///
    /// # Errors
    ///
    /// Propagates vector-store failures.
    pub async fn delete_documents(&self, doc_ids: &[String]) -> Result<usize> {
        if doc_ids.is_empty() {
            return Ok(0);
        }
        // Resolve the chunk ids before the delete, while the store still has them.
        let chunk_ids = self.chunk_ids_of(doc_ids).await?;
        let filter = MetadataFilter::in_values(DOC_ID_FIELD, doc_ids.iter().cloned());
        let removed = self.store.delete_by_filter(&filter).await?;
        self.drop_indexed_chunks(&chunk_ids);
        self.forget_documents(doc_ids);
        Ok(removed)
    }

    /// Replaces the documents that already exist by deleting their chunks and
    /// then adding the new content.
    ///
    /// Delete failures of unknown ids are not errors: the delete only removes
    /// what is present. Documents with a blank id are simply added.
    ///
    /// # Errors
    ///
    /// See [`SimpleKnowledgeBase::add_documents`].
    // The signature takes the documents by value to mirror `add_documents`.
    #[allow(clippy::needless_pass_by_value)]
    pub async fn update_documents(&self, documents: Vec<Document>) -> Result<Vec<String>> {
        let existing: Vec<String> = documents
            .iter()
            .filter(|doc| !doc.id.trim().is_empty())
            .map(|doc| doc.id.clone())
            .collect();
        if !existing.is_empty() {
            // The removed count is intentionally ignored: an unknown id means
            // there was nothing to replace.
            self.delete_documents(&existing).await?;
        }
        self.add_documents(documents).await
    }

    /// Retrieves documents relevant to `query` with the default configuration.
    ///
    /// # Errors
    ///
    /// See [`SimpleKnowledgeBase::retrieve_with_config`].
    pub async fn retrieve(&self, query: &str) -> Result<Vec<RetrievalResult>> {
        self.retrieve_with_config(query, &RetrievalConfig::default())
            .await
    }

    /// Retrieves documents relevant to `query`, honoring `config`.
    ///
    /// The configured channel applies `filters` and the inclusive
    /// `score_threshold`, the optional graph layer expands entities, the optional
    /// agentic layer rewrites and re-retrieves, and the optional reranker then
    /// reorders the hits. Results are always truncated to `config.top_k`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for a blank `query` or `config.top_k == 0`,
    /// and propagates the retriever/store/embedder/reranker failures with stage
    /// contexts. See the private `retrieve` submodule for the dispatch details.
    pub async fn retrieve_with_config(
        &self,
        query: &str,
        config: &RetrievalConfig,
    ) -> Result<Vec<RetrievalResult>> {
        retrieve::retrieve_with_config(self, query, config).await
    }

    /// Loads documents from file paths, skipping sources that fail to load.
    ///
    /// Python's `parse_files` logs and continues on a bad path, so one missing
    /// file cannot abort a batch ingestion; the returned documents are exactly
    /// the ones that loaded.
    ///
    /// # Errors
    ///
    /// Only documented for symmetry: per-source failures are logged and skipped,
    /// so this method currently always returns `Ok`.
    pub async fn parse_files(&self, paths: &[String]) -> Result<Vec<Document>> {
        let mut documents = Vec::new();
        for source in paths {
            match self.loader.load(source).await {
                Ok(loaded) => documents.extend(loaded),
                Err(error) => warn!(
                    source = source.as_str(),
                    error = %error,
                    "skipping a source that failed to load"
                ),
            }
        }
        Ok(documents)
    }

    /// Loads documents from `HTTP(S)` URLs, skipping non-URL and failing sources.
    ///
    /// # Errors
    ///
    /// Only documented for symmetry: every source failure is logged and skipped,
    /// so this method currently always returns `Ok`.
    pub async fn parse_urls(&self, urls: &[String]) -> Result<Vec<Document>> {
        let mut documents = Vec::new();
        for source in urls {
            if !is_url(source) {
                warn!(
                    source = source.as_str(),
                    "skipping a source that is not an HTTP(S) URL"
                );
                continue;
            }
            match self.loader.load(source).await {
                Ok(loaded) => documents.extend(loaded),
                Err(error) => warn!(
                    source = source.as_str(),
                    error = %error,
                    "skipping a URL that failed to load"
                ),
            }
        }
        Ok(documents)
    }

    /// Returns the number of chunks currently stored.
    ///
    /// # Errors
    ///
    /// Propagates vector-store failures.
    pub async fn count(&self) -> Result<usize> {
        self.store.len().await
    }

    /// Returns a snapshot of the knowledge base statistics.
    ///
    /// # Errors
    ///
    /// Propagates vector-store failures from the chunk count.
    pub async fn get_statistics(&self) -> Result<KnowledgeBaseStatistics> {
        Ok(KnowledgeBaseStatistics {
            kb_id: self.config.kb_id.clone(),
            index_type: self.config.index_type.clone(),
            chunk_count: self.store.len().await?,
            document_count: self.document_ids.read().len(),
            has_reranker: self.reranker.is_some(),
            has_loader: true,
            sparse_index_size: self.sparse_index_len(),
            has_graph: self.graph_store.is_some(),
            has_rewriter: self.rewriter.is_some(),
            agentic_enabled: self.agentic,
        })
    }

    /// Indexes freshly stored chunks in the BM25 index and the triple store.
    ///
    /// Graph extraction is unconditional for a configured store (not gated on
    /// `KnowledgeBaseConfig::use_graph`) so a later per-query
    /// [`RetrievalConfig::graph_expansion`] always finds triples.
    async fn index_chunks(&self, chunks: &[TextChunk]) -> Result<()> {
        if let Some(index) = &self.sparse {
            index.write().add_chunks(chunks);
        }
        if let Some(graph_store) = &self.graph_store {
            let triples = RuleTripleExtractor::new().extract_chunks(chunks);
            if !triples.is_empty() {
                graph_store
                    .add_triples(triples)
                    .await
                    .map_err(|error| error.with_context("indexing graph triples"))?;
            }
        }
        Ok(())
    }

    /// The ids of the stored chunks belonging to `doc_ids`.
    ///
    /// Only needed when the BM25 index must mirror a delete, so a knowledge base
    /// without a sparse index never lists the store.
    async fn chunk_ids_of(&self, doc_ids: &[String]) -> Result<Vec<String>> {
        if self.sparse.is_none() {
            return Ok(Vec::new());
        }
        let requested: BTreeSet<&str> = doc_ids.iter().map(String::as_str).collect();
        let records = self.store.list(None).await?;
        Ok(records
            .into_iter()
            .filter(|record| record_doc_id(record).is_some_and(|doc_id| requested.contains(doc_id)))
            .map(|record| record.id)
            .collect())
    }

    /// Removes `chunk_ids` from the BM25 index; a no-op when it is disabled.
    fn drop_indexed_chunks(&self, chunk_ids: &[String]) {
        let Some(index) = &self.sparse else {
            return;
        };
        let mut index = index.write();
        for id in chunk_ids {
            index.remove(id);
        }
    }

    /// Records `ids` as known documents; the lock is never held across an await.
    fn remember_documents(&self, ids: &[String]) {
        let mut tracked = self.document_ids.write();
        for id in ids {
            tracked.insert(id.clone());
        }
    }

    /// Drops `ids` from the tracked document set.
    fn forget_documents(&self, ids: &[String]) {
        let mut tracked = self.document_ids.write();
        for id in ids {
            tracked.shift_remove(id);
        }
    }
}

/// The document a stored record belongs to, from its field or its metadata.
fn record_doc_id(record: &VectorRecord) -> Option<&str> {
    record.doc_id.as_deref().or_else(|| {
        record
            .metadata
            .get(DOC_ID_FIELD)
            .and_then(serde_json::Value::as_str)
    })
}

/// The initial sparse index: enabled exactly for the `bm25` index type.
fn initial_sparse_index(index_type: &str) -> Option<Arc<RwLock<Bm25Index>>> {
    (index_type == INDEX_TYPE_BM25).then(new_sparse_index)
}

/// A fresh, empty BM25 index with the default scoring parameters.
fn new_sparse_index() -> Arc<RwLock<Bm25Index>> {
    Arc::new(RwLock::new(Bm25Index::default()))
}

/// Fills blank document ids with fresh `uuid4` values, matching Python's
/// `if not doc.id_.strip(): doc.id_ = str(uuid.uuid4())`.
fn assign_missing_ids(mut documents: Vec<Document>) -> Vec<Document> {
    for document in &mut documents {
        if document.id.trim().is_empty() {
            document.id = Uuid::new_v4().to_string();
        }
    }
    documents
}
