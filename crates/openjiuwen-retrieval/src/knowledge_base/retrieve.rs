//! Retrieval channel dispatch for [`SimpleKnowledgeBase`].
//!
//! Python's `SimpleKnowledgeBase.retrieve` picks a retriever from
//! `KnowledgeBaseConfig.index_type` (`vector` → `VectorRetriever`, `bm25` →
//! `SparseRetriever`, everything else → `HybridRetriever`), wraps it in an
//! `AgenticRetriever` when `RetrievalConfig.agentic` is set, and uses a graph
//! retriever when graph search is requested. This module ports that dispatch:
//!
//! 1. the **channel** comes from the index type — dense, BM25, or hybrid;
//! 2. a [`GraphRetriever`] wraps the channel when a graph store is attached and
//!    the query asks for graph retrieval;
//! 3. an [`AgenticRetriever`] wraps the result when agentic retrieval is on;
//! 4. the optional reranker reorders the hits and `top_k` truncates them.
//!
//! Unlike Python, the sparse index is opt-in for `hybrid` and `vector`: only the
//! `bm25` index type enables it by default, so dense and hybrid knowledge bases
//! added through [`SimpleKnowledgeBase::new`] keep returning exactly the dense
//! results they did before the sparse index existed. Enable it with
//! [`SimpleKnowledgeBase::with_sparse_index`] to get RRF-dense+BM25 fusion.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use async_trait::async_trait;
use parking_lot::RwLock;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::logging::warn;

use crate::common::{
    parse_index_type, MetadataFilter, RetrievalConfig, RetrievalResult, SearchResult,
    INDEX_TYPE_BM25, INDEX_TYPE_HYBRID,
};
use crate::retriever::{
    ensure_query, search_result_to_retrieval, AgenticConfig, AgenticRetriever, GraphRetriever,
    HybridRetriever, RetrieveMode, Retriever, VectorRetriever,
};
use crate::rewrite::{HeuristicQueryRewriter, HeuristicRewriteConfig, QueryRewriter};
use crate::sparse::{tokenize, Bm25Index, Bm25Query};

use super::SimpleKnowledgeBase;

/// Which retrieval channel a knowledge base serves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Channel {
    /// Dense vector similarity over the vector store.
    Dense,
    /// Lexical BM25 over the sparse index.
    Sparse,
    /// Dense + BM25 fused with reciprocal rank fusion.
    Hybrid,
}

impl Channel {
    /// The retriever mode that addresses this channel.
    fn mode(self) -> RetrieveMode {
        match self {
            Self::Dense => RetrieveMode::Vector,
            Self::Sparse => RetrieveMode::Sparse,
            Self::Hybrid => RetrieveMode::Hybrid,
        }
    }

    /// The wire name of the channel, for log messages.
    fn as_str(self) -> &'static str {
        match self {
            Self::Dense => "vector",
            Self::Sparse => "bm25",
            Self::Hybrid => "hybrid",
        }
    }

    /// The mode used to look up triple text during graph expansion.
    ///
    /// Triple text (`"subject predicate object"`) is lexically distinctive, so
    /// the sparse channel is the right lookup whenever one exists; the dense-only
    /// knowledge base falls back to vector similarity.
    fn triple_mode(self) -> RetrieveMode {
        match self {
            Self::Dense => RetrieveMode::Vector,
            Self::Sparse | Self::Hybrid => RetrieveMode::Sparse,
        }
    }
}

/// Runs `query` through the knowledge base's configured retrieval stack.
pub(super) async fn retrieve_with_config(
    kb: &SimpleKnowledgeBase,
    query: &str,
    config: &RetrievalConfig,
) -> Result<Vec<RetrievalResult>> {
    if query.trim().is_empty() {
        return Err(Error::validation("retrieval query must not be empty"));
    }
    if config.top_k == 0 {
        return Err(Error::validation("retrieval top_k must be greater than 0"));
    }

    let channel = channel_of(kb)?;
    let graph_requested = config.use_graph.unwrap_or(kb.config.use_graph) || config.graph_expansion;
    let graph_store = if graph_requested {
        kb.graph_store.clone()
    } else {
        None
    };
    if graph_requested && graph_store.is_none() {
        warn!(
            kb_id = kb.config.kb_id.as_str(),
            channel = channel.as_str(),
            "graph retrieval requested without a graph store; using the plain channel"
        );
    }

    let mut retriever = channel_retriever(kb, channel);
    if let Some(graph_store) = graph_store {
        retriever = Arc::new(
            GraphRetriever::new(retriever, graph_store)
                .with_max_hops(kb.graph_hops)
                .with_triple_mode(channel.triple_mode()),
        );
    }
    if config.agentic || kb.agentic {
        let rewriter = effective_rewriter(kb).await;
        let agentic_config = AgenticConfig::new().with_max_iter(kb.agentic_iterations);
        retriever = Arc::new(
            AgenticRetriever::new(retriever, rewriter)
                .with_config(agentic_config)
                .map_err(|error| error.with_context("configuring agentic retrieval"))?,
        );
    }

    let mut results = retriever
        .retrieve(
            query,
            config.top_k,
            config.score_threshold,
            channel.mode(),
            config.filters.as_ref(),
        )
        .await
        .map_err(|error| error.with_context("retrieving from the knowledge base"))?;

    if let Some(reranker) = &kb.reranker {
        let hits: Vec<SearchResult> = results
            .into_iter()
            .map(|result| SearchResult {
                id: result
                    .chunk_id
                    .clone()
                    .filter(|id| !id.trim().is_empty())
                    .unwrap_or_else(|| result.text.clone()),
                text: result.text,
                score: result.score,
                metadata: result.metadata,
            })
            .collect();
        let hits = reranker
            .rerank(query, hits, Some(config.top_k))
            .await
            .map_err(|error| error.with_context("reranking results"))?;
        results = hits.into_iter().map(search_result_to_retrieval).collect();
    }

    results.truncate(config.top_k);
    Ok(results)
}

/// Resolves the channel from the index type and sparse-index availability.
///
/// An unknown index type is a configuration error (see [`parse_index_type`]), so
/// a typo cannot silently turn a BM25 query into a dense one. `bm25`/`hybrid`
/// with the sparse index explicitly disabled do degrade to dense, with a
/// warning: [`SimpleKnowledgeBase::with_sparse_index`] is the only way to get
/// there, and the empty index would return nothing at all.
fn channel_of(kb: &SimpleKnowledgeBase) -> Result<Channel> {
    let requested = parse_index_type(&kb.config.index_type)?;
    let channel = match requested {
        INDEX_TYPE_BM25 => Channel::Sparse,
        INDEX_TYPE_HYBRID => Channel::Hybrid,
        _ => Channel::Dense,
    };
    if channel != Channel::Dense && !kb.has_sparse_index() {
        warn!(
            index_type = requested,
            "the BM25 sparse index is disabled; falling back to dense retrieval"
        );
        return Ok(Channel::Dense);
    }
    Ok(channel)
}

/// Builds the retriever serving `channel`.
fn channel_retriever(kb: &SimpleKnowledgeBase, channel: Channel) -> Arc<dyn Retriever> {
    match channel {
        Channel::Dense => dense_retriever(kb),
        // The sparse channel only exists when the index does; `channel_of`
        // guarantees it, and the dense fallback keeps this total.
        Channel::Sparse => sparse_retriever(kb).unwrap_or_else(|| dense_retriever(kb)),
        Channel::Hybrid => Arc::new(HybridRetriever::new(
            dense_retriever(kb),
            sparse_retriever(kb).unwrap_or_else(|| dense_retriever(kb)),
        )),
    }
}

/// The dense channel over the vector store and the embedder.
fn dense_retriever(kb: &SimpleKnowledgeBase) -> Arc<dyn Retriever> {
    Arc::new(VectorRetriever::new(
        Arc::clone(&kb.store),
        Arc::clone(&kb.embedder),
    ))
}

/// The BM25 channel over the live sparse index, when one is attached.
fn sparse_retriever(kb: &SimpleKnowledgeBase) -> Option<Arc<dyn Retriever>> {
    kb.sparse
        .as_ref()
        .map(|index| Arc::new(SparseIndexRetriever::new(Arc::clone(index))) as Arc<dyn Retriever>)
}

/// Picks the rewriter for the agentic path.
///
/// An injected rewriter wins; otherwise a [`HeuristicQueryRewriter`] ranks
/// keywords by the document frequency of the knowledge base's own chunks, so the
/// default works offline and without a model.
async fn effective_rewriter(kb: &SimpleKnowledgeBase) -> Arc<dyn QueryRewriter> {
    if let Some(rewriter) = &kb.rewriter {
        return Arc::clone(rewriter);
    }
    let config =
        HeuristicRewriteConfig::default().with_corpus_vocabulary(corpus_vocabulary(kb).await);
    Arc::new(HeuristicQueryRewriter::new(config))
}

/// Term → document frequency over every stored chunk text.
///
/// A store failure only costs the IDF ranking (a warning, empty vocabulary):
/// listing the corpus is an optimisation, so it must not fail a retrieval.
async fn corpus_vocabulary(kb: &SimpleKnowledgeBase) -> BTreeMap<String, usize> {
    let records = match kb.store.list(None).await {
        Ok(records) => records,
        Err(error) => {
            warn!(
                kb_id = kb.config.kb_id.as_str(),
                error = %error,
                "corpus vocabulary unavailable; ranking rewrite keywords by length"
            );
            return BTreeMap::new();
        }
    };
    let mut vocabulary: BTreeMap<String, usize> = BTreeMap::new();
    for record in &records {
        let mut seen: BTreeSet<String> = BTreeSet::new();
        for term in tokenize(&record.text) {
            if seen.insert(term.clone()) {
                *vocabulary.entry(term).or_insert(0) += 1;
            }
        }
    }
    vocabulary
}

/// A [`Retriever`] reading the knowledge base's BM25 index through its lock.
///
/// The public [`crate::retriever::SparseRetriever`] owns an immutable
/// `Arc<Bm25Index>`; the knowledge base keeps mutating its index as documents
/// arrive, so this adapter borrows the shared lock instead of snapshotting.
#[derive(Debug)]
struct SparseIndexRetriever {
    /// The shared lexical index.
    index: Arc<RwLock<Bm25Index>>,
}

impl SparseIndexRetriever {
    /// Wraps the shared index.
    fn new(index: Arc<RwLock<Bm25Index>>) -> Self {
        Self { index }
    }

    /// Rejects every mode but [`RetrieveMode::Sparse`].
    fn ensure_sparse_mode(mode: RetrieveMode) -> Result<()> {
        if mode == RetrieveMode::Sparse {
            return Ok(());
        }
        Err(Error::validation(format!(
            "the BM25 index only supports 'sparse' mode, got {}",
            mode.as_str()
        )))
    }

    /// Runs one BM25 search; the read lock is released before returning.
    fn search(
        &self,
        query: &str,
        top_k: usize,
        score_threshold: Option<f32>,
        filters: Option<&MetadataFilter>,
    ) -> Vec<SearchResult> {
        if top_k == 0 {
            return Vec::new();
        }
        self.index.read().search(&Bm25Query {
            text: query.to_owned(),
            top_k,
            filter: filters.cloned(),
            score_threshold,
        })
    }
}

#[async_trait]
impl Retriever for SparseIndexRetriever {
    async fn retrieve(
        &self,
        query: &str,
        top_k: usize,
        score_threshold: Option<f32>,
        mode: RetrieveMode,
        filters: Option<&MetadataFilter>,
    ) -> Result<Vec<RetrievalResult>> {
        Self::ensure_sparse_mode(mode)?;
        ensure_query(query)?;
        Ok(self
            .search(query, top_k, score_threshold, filters)
            .into_iter()
            .map(search_result_to_retrieval)
            .collect())
    }

    async fn retrieve_search_results(
        &self,
        query: &str,
        top_k: usize,
        mode: RetrieveMode,
        filters: Option<&MetadataFilter>,
    ) -> Result<Vec<SearchResult>> {
        Self::ensure_sparse_mode(mode)?;
        ensure_query(query)?;
        Ok(self.search(query, top_k, None, filters))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::float_cmp)]
mod tests {
    use super::*;
    use crate::chunking::CharChunker;
    use crate::common::{KnowledgeBaseConfig, INDEX_TYPE_VECTOR};
    use crate::embedding::HashEmbedding;
    use crate::vector_store::InMemoryVectorStore;
    use openjiuwen_common::error::ErrorKind;

    fn kb(index_type: &str) -> SimpleKnowledgeBase {
        let mut config = KnowledgeBaseConfig::new("kb");
        config.index_type = index_type.to_owned();
        SimpleKnowledgeBase::new(
            config,
            Arc::new(InMemoryVectorStore::new()),
            Arc::new(HashEmbedding::new(16)),
            Arc::new(CharChunker::default()),
        )
    }

    #[test]
    fn channel_follows_the_index_type_and_the_sparse_index() {
        assert_eq!(channel_of(&kb(INDEX_TYPE_BM25)).unwrap(), Channel::Sparse);
        assert_eq!(channel_of(&kb(INDEX_TYPE_HYBRID)).unwrap(), Channel::Dense);
        assert_eq!(channel_of(&kb(INDEX_TYPE_VECTOR)).unwrap(), Channel::Dense);
        // Case and surrounding whitespace are normalized.
        assert_eq!(channel_of(&kb(" BM25 ")).unwrap(), Channel::Sparse);
        // An unknown index type is a configuration error, not a dense fallback.
        let error = channel_of(&kb("made-up")).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Config);
        assert!(
            error
                .to_string()
                .contains("unsupported index type `made-up`"),
            "unexpected message: {error}"
        );
        assert_eq!(
            channel_of(&kb(INDEX_TYPE_HYBRID).with_sparse_index(true)).unwrap(),
            Channel::Hybrid
        );
        assert_eq!(
            channel_of(&kb(INDEX_TYPE_BM25).with_sparse_index(false)).unwrap(),
            Channel::Dense
        );
    }

    #[test]
    fn channel_modes_and_triple_modes_line_up() {
        assert_eq!(Channel::Dense.mode(), RetrieveMode::Vector);
        assert_eq!(Channel::Sparse.mode(), RetrieveMode::Sparse);
        assert_eq!(Channel::Hybrid.mode(), RetrieveMode::Hybrid);
        assert_eq!(Channel::Dense.triple_mode(), RetrieveMode::Vector);
        assert_eq!(Channel::Sparse.triple_mode(), RetrieveMode::Sparse);
        assert_eq!(Channel::Hybrid.triple_mode(), RetrieveMode::Sparse);
        assert_eq!(Channel::Dense.as_str(), "vector");
    }

    #[tokio::test]
    async fn sparse_index_retriever_reads_the_live_index() {
        let index = Arc::new(RwLock::new(Bm25Index::default()));
        let retriever = SparseIndexRetriever::new(Arc::clone(&index));
        assert!(index.write().is_empty());
        index
            .write()
            .add_document("chunk-1", "quokka", crate::common::Metadata::new());
        let hits = retriever
            .retrieve_search_results("quokka", 5, RetrieveMode::Sparse, None)
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, "chunk-1");
        let error = retriever
            .retrieve("quokka", 5, None, RetrieveMode::Vector, None)
            .await
            .unwrap_err();
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Validation
        );
        assert!(retriever
            .retrieve(" ", 5, None, RetrieveMode::Sparse, None)
            .await
            .is_err());
    }
}
