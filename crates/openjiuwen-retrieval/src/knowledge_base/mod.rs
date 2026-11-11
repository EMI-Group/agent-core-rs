//! The knowledge base pipeline: load → chunk → embed → upsert → search → rerank.
//!
//! Port of the Python `openjiuwen/core/retrieval/simple_knowledge_base.py`
//! `SimpleKnowledgeBase`. Python wires this class to an `IndexManager`, a
//! `Retriever`, and a separate graph index; the Rust port keeps the same
//! composition story with one [`VectorStore`] as the only index and performs the
//! search inline, so every stage is visible on a single struct.
//!
//! The knowledge base owns its collaborators as trait objects, which keeps the
//! pipeline testable offline (see [`crate::embedding::HashEmbedding`] and
//! [`crate::vector_store::InMemoryVectorStore`]).
mod multi;

use std::sync::Arc;

use indexmap::IndexSet;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::logging::warn;

use crate::chunking::{CharChunker, Chunker};
use crate::common::{
    Document, KnowledgeBaseConfig, MetadataFilter, RetrievalConfig, RetrievalResult,
};
use crate::embedding::Embedding;
use crate::loaders::{is_url, AutoLoader, DocumentLoader};
use crate::reranker::Reranker;
use crate::vector_store::{VectorQuery, VectorRecord, VectorStore, DOC_ID_FIELD};

pub use multi::{retrieve_multi_kb, retrieve_multi_kb_with_source};

/// Statistics describing a [`SimpleKnowledgeBase`].
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
}

impl SimpleKnowledgeBase {
    /// Builds a knowledge base from its collaborators.
    ///
    /// The loader defaults to [`AutoLoader`] and no reranker is attached; use
    /// [`SimpleKnowledgeBase::with_loader`] and
    /// [`SimpleKnowledgeBase::with_reranker`] to override them.
    pub fn new(
        config: KnowledgeBaseConfig,
        store: Arc<dyn VectorStore>,
        embedder: Arc<dyn Embedding>,
        chunker: Arc<dyn Chunker>,
    ) -> Self {
        Self {
            config,
            store,
            embedder,
            chunker,
            loader: Arc::new(AutoLoader::new()),
            reranker: None,
            document_ids: RwLock::new(IndexSet::new()),
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
    /// accurate, but it contributes nothing to the store.
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
    /// # Errors
    ///
    /// Propagates vector-store failures.
    pub async fn delete_documents(&self, doc_ids: &[String]) -> Result<usize> {
        if doc_ids.is_empty() {
            return Ok(0);
        }
        let filter = MetadataFilter::in_values(DOC_ID_FIELD, doc_ids.iter().cloned());
        let removed = self.store.delete_by_filter(&filter).await?;
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
    /// The store applies `filters` and the inclusive `score_threshold`; the
    /// optional reranker then reorders (and usually truncates) the hits. Results
    /// are always truncated to `config.top_k`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for a blank `query` or `config.top_k == 0`,
    /// and the embedder/store/reranker errors with the stage contexts
    /// `"embedding query"`, `"searching vector store"`, and `"reranking results"`.
    pub async fn retrieve_with_config(
        &self,
        query: &str,
        config: &RetrievalConfig,
    ) -> Result<Vec<RetrievalResult>> {
        if query.trim().is_empty() {
            return Err(Error::validation("retrieval query must not be empty"));
        }
        if config.top_k == 0 {
            return Err(Error::validation("retrieval top_k must be greater than 0"));
        }

        let vector = self
            .embedder
            .embed_query(query)
            .await
            .map_err(|error| error.with_context("embedding query"))?;
        let vector_query = VectorQuery {
            vector,
            top_k: config.top_k,
            filter: config.filters.clone(),
            score_threshold: config.score_threshold,
        };
        let mut hits = self
            .store
            .search(vector_query)
            .await
            .map_err(|error| error.with_context("searching vector store"))?;

        if let Some(reranker) = &self.reranker {
            hits = reranker
                .rerank(query, hits, Some(config.top_k))
                .await
                .map_err(|error| error.with_context("reranking results"))?;
        }

        let mut results: Vec<RetrievalResult> = hits
            .into_iter()
            .map(|hit| RetrievalResult {
                doc_id: hit
                    .metadata
                    .get(DOC_ID_FIELD)
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_owned),
                text: hit.text,
                score: hit.score,
                metadata: hit.metadata,
                chunk_id: Some(hit.id),
            })
            .collect();
        results.truncate(config.top_k);
        Ok(results)
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
        })
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
