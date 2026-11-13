//! Unified retriever interface, the five retriever implementations, and rank
//! fusion.
//!
//! Port of the Python `openjiuwen/core/retrieval/retriever/` package and
//! `openjiuwen/core/retrieval/utils/fusion.py`:
//!
//! - [`Retriever`] is the abstract base: `retrieve`, `retrieve_search_results`,
//!   `batch_retrieve`, and `close`.
//! - [`VectorRetriever`] is the dense channel over a vector store and an
//!   embedding model.
//! - [`SparseRetriever`] is the BM25-backed sparse implementation. Python's
//!   version delegates to `VectorStore.sparse_search`; this crate owns the index,
//!   so the retriever wraps a [`std::sync::Arc`] of [`crate::sparse::Bm25Index`].
//! - [`HybridRetriever`] runs both channels and fuses them with reciprocal rank
//!   fusion.
//! - [`GraphRetriever`] expands chunk hits through a triple store.
//! - [`AgenticRetriever`] loops retrieve → read triples → rewrite and fuses every
//!   iteration, working offline with deterministic heuristics unless a model is
//!   supplied.
//! - [`rrf_fusion`] and [`deduplicate`] merge result lists the way Python's
//!   `rrf_fusion` does.

pub mod agentic;
pub mod fusion;
pub mod graph;
pub mod hybrid;
pub mod sparse;
pub mod vector;

pub use agentic::{AgenticConfig, AgenticRetriever, AgenticStopReason, AgenticTrace};
pub use fusion::{deduplicate, rrf_fusion, rrf_fusion_default, DEFAULT_RRF_K};
pub use graph::GraphRetriever;
pub use hybrid::HybridRetriever;
pub use sparse::SparseRetriever;
pub use vector::VectorRetriever;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use openjiuwen_common::error::{Error, Result};

use crate::common::{MetadataFilter, RetrievalResult, SearchResult};
use crate::vector_store::DOC_ID_FIELD;

/// Which retrieval channel a call should use.
///
/// Python spells these as the `Literal["vector", "sparse", "hybrid"]` mode
/// argument; the serialized form keeps the same lowercase names and the default
/// matches Python's `"hybrid"`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RetrieveMode {
    /// Dense vector similarity search.
    Vector,
    /// Lexical BM25 search.
    Sparse,
    /// Both channels, fused (the Python default).
    #[default]
    Hybrid,
}

impl RetrieveMode {
    /// The Python wire name of the mode (`"vector"`, `"sparse"`, or `"hybrid"`).
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Vector => "vector",
            Self::Sparse => "sparse",
            Self::Hybrid => "hybrid",
        }
    }
}

/// The retriever abstract base.
///
/// Implementations must be deterministic for identical inputs and must not
/// panic on empty queries, `top_k == 0`, or filters that match nothing.
#[async_trait]
pub trait Retriever: Send + Sync {
    /// Retrieves documents for `query` in `mode`.
    ///
    /// Returns at most `top_k` results ordered by descending relevance, dropping
    /// scores below `score_threshold` when it is set. `filters` is applied before
    /// scoring; Python's base accepts filters through `**kwargs` and drops them,
    /// while Rust plumbs them through to the underlying index.
    async fn retrieve(
        &self,
        query: &str,
        top_k: usize,
        score_threshold: Option<f32>,
        mode: RetrieveMode,
        filters: Option<&MetadataFilter>,
    ) -> Result<Vec<RetrievalResult>>;

    /// Retrieves raw [`SearchResult`]s for `query` in `mode`.
    ///
    /// This is the Python "optional" API that exposes store-level hits without
    /// the `RetrievalResult` projection; it has no score threshold because the
    /// raw scores are returned as produced.
    async fn retrieve_search_results(
        &self,
        query: &str,
        top_k: usize,
        mode: RetrieveMode,
        filters: Option<&MetadataFilter>,
    ) -> Result<Vec<SearchResult>>;

    /// Retrieves for several queries in order.
    ///
    /// The default implementation is sequential; results line up with `queries`
    /// (including duplicate queries) and the first failure aborts the batch.
    async fn batch_retrieve(
        &self,
        queries: &[String],
        top_k: usize,
        mode: RetrieveMode,
    ) -> Result<Vec<Vec<RetrievalResult>>> {
        let mut results = Vec::with_capacity(queries.len());
        for query in queries {
            results.push(self.retrieve(query, top_k, None, mode, None).await?);
        }
        Ok(results)
    }

    /// Releases retriever resources.
    ///
    /// The default is a no-op: the in-process sparse index owns no external
    /// connection. Implementations backed by a store should close it here.
    async fn close(&self) {}
}

/// Projects a store-level [`SearchResult`] onto the public [`RetrievalResult`].
///
/// `chunk_id` is the hit id and `doc_id` comes from the `doc_id` metadata field
/// the stores inject, matching Python's `SparseRetriever.retrieve`.
pub fn search_result_to_retrieval(result: SearchResult) -> RetrievalResult {
    let doc_id = result
        .metadata
        .get(DOC_ID_FIELD)
        .and_then(serde_json::Value::as_str)
        .map(ToOwned::to_owned);
    RetrievalResult {
        text: result.text,
        score: result.score,
        metadata: result.metadata,
        doc_id,
        chunk_id: Some(result.id),
    }
}

/// Rejects a blank retrieval query with [`Error::Validation`].
///
/// Shared by the concrete retrievers so an empty query fails deterministically
/// instead of reaching an index that would answer it with an empty result list.
pub(crate) fn ensure_query(query: &str) -> Result<()> {
    if query.trim().is_empty() {
        return Err(Error::validation("retrieval query must not be empty"));
    }
    Ok(())
}

/// Projects a [`RetrievalResult`] back onto a [`SearchResult`].
///
/// The chunk id is the identity when it is known, otherwise the text is used:
/// results produced by fusion keep Python's `text` identity, so this round trip
/// never loses a hit.
pub(crate) fn retrieval_to_search(result: RetrievalResult) -> SearchResult {
    let id = result
        .chunk_id
        .clone()
        .filter(|id| !id.trim().is_empty())
        .unwrap_or_else(|| result.text.clone());
    SearchResult {
        id,
        text: result.text,
        score: result.score,
        metadata: result.metadata,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::common::Metadata;
    use serde_json::json;

    #[test]
    fn mode_is_hybrid_by_default_and_serde_uses_snake_case() {
        assert_eq!(RetrieveMode::default(), RetrieveMode::Hybrid);
        assert_eq!(RetrieveMode::Vector.as_str(), "vector");
        assert_eq!(RetrieveMode::Sparse.as_str(), "sparse");
        assert_eq!(RetrieveMode::Hybrid.as_str(), "hybrid");
        assert_eq!(
            serde_json::to_value(RetrieveMode::Sparse).unwrap(),
            json!("sparse")
        );
        let parsed: RetrieveMode = serde_json::from_value(json!("vector")).unwrap();
        assert_eq!(parsed, RetrieveMode::Vector);
        assert!(serde_json::from_value::<RetrieveMode>(json!("graph")).is_err());
    }

    #[test]
    fn search_result_projection_uses_the_metadata_doc_id() {
        let mut metadata = Metadata::new();
        metadata.insert(DOC_ID_FIELD.to_owned(), json!("doc-9"));
        let result = search_result_to_retrieval(SearchResult {
            id: "chunk-1".to_owned(),
            text: "body".to_owned(),
            score: 1.5,
            metadata,
        });
        assert_eq!(result.chunk_id.as_deref(), Some("chunk-1"));
        assert_eq!(result.doc_id.as_deref(), Some("doc-9"));
        assert_eq!(result.text, "body");
        assert_eq!(result.score, 1.5);

        let anonymous = search_result_to_retrieval(SearchResult {
            id: "chunk-2".to_owned(),
            text: "body".to_owned(),
            score: 0.5,
            metadata: Metadata::new(),
        });
        assert_eq!(anonymous.doc_id, None);
        assert_eq!(anonymous.chunk_id.as_deref(), Some("chunk-2"));
    }
}
