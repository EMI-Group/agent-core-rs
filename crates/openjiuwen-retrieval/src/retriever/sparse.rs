//! BM25-backed sparse retriever.
//!
//! Port of Python's `SparseRetriever`, which forwards to
//! `VectorStore.sparse_search`. This crate owns the lexical index
//! ([`Bm25Index`]), so the retriever holds it behind an [`Arc`] that callers can
//! share with the indexing side.

use std::sync::Arc;

use async_trait::async_trait;

use openjiuwen_common::error::{Error, Result};

use super::{search_result_to_retrieval, RetrieveMode, Retriever};
use crate::common::{MetadataFilter, RetrievalResult, SearchResult};
use crate::sparse::{Bm25Index, Bm25Query};

/// A [`Retriever`] over a [`Bm25Index`] that only serves [`RetrieveMode::Sparse`].
#[derive(Debug, Clone)]
pub struct SparseRetriever {
    index: Arc<Bm25Index>,
}

impl SparseRetriever {
    /// Wraps `index`.
    pub fn new(index: Arc<Bm25Index>) -> Self {
        Self { index }
    }

    /// The shared index backing this retriever.
    pub fn index(&self) -> &Arc<Bm25Index> {
        &self.index
    }

    /// Rejects every mode but [`RetrieveMode::Sparse`].
    fn ensure_sparse_mode(mode: RetrieveMode) -> Result<()> {
        if mode == RetrieveMode::Sparse {
            return Ok(());
        }
        Err(Error::validation(format!(
            "SparseRetriever only supports 'sparse' mode, got {}",
            mode.as_str()
        )))
    }

    /// Runs the underlying BM25 search with the retriever arguments.
    fn search(
        &self,
        query: &str,
        top_k: usize,
        score_threshold: Option<f32>,
        filters: Option<&MetadataFilter>,
    ) -> Vec<SearchResult> {
        self.index.search(&Bm25Query {
            text: query.to_owned(),
            top_k,
            filter: filters.cloned(),
            score_threshold,
        })
    }
}

#[async_trait]
impl Retriever for SparseRetriever {
    async fn retrieve(
        &self,
        query: &str,
        top_k: usize,
        score_threshold: Option<f32>,
        mode: RetrieveMode,
        filters: Option<&MetadataFilter>,
    ) -> Result<Vec<RetrievalResult>> {
        Self::ensure_sparse_mode(mode)?;
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
        Ok(self.search(query, top_k, None, filters))
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::assert_is_empty
)]
mod tests {
    use super::*;
    use crate::common::{Document, Metadata};
    use crate::sparse::Bm25Config;

    fn retriever() -> SparseRetriever {
        let mut index = Bm25Index::new(Bm25Config::default()).unwrap();
        index.add_documents(&[
            Document::with_id("d1", "quick brown fox"),
            Document::with_id("d2", "quick quick fox"),
            Document::with_id("d3", "lazy dog")
                .with_metadata(Metadata::from_iter([("kind".to_owned(), "pet".into())])),
        ]);
        SparseRetriever::new(Arc::new(index))
    }

    #[tokio::test]
    async fn sparse_mode_returns_ranked_results() {
        let retriever = retriever();
        let results = retriever
            .retrieve("quick", 2, None, RetrieveMode::Sparse, None)
            .await
            .unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].chunk_id.as_deref(), Some("d2"));
        assert_eq!(results[1].chunk_id.as_deref(), Some("d1"));
        assert!(results[0].score > results[1].score);
        assert_eq!(results[0].doc_id, None);
        assert_eq!(retriever.index().len(), 3);
    }

    #[tokio::test]
    async fn other_modes_are_rejected_with_a_validation_error() {
        let retriever = retriever();
        for mode in [RetrieveMode::Vector, RetrieveMode::Hybrid] {
            let error = retriever
                .retrieve("quick", 5, None, mode, None)
                .await
                .unwrap_err();
            assert_eq!(
                error.kind(),
                openjiuwen_common::error::ErrorKind::Validation
            );
            assert!(error.to_string().contains("only supports 'sparse' mode"));
            assert!(error.to_string().contains(mode.as_str()));
            assert!(retriever
                .retrieve_search_results("quick", 5, mode, None)
                .await
                .is_err());
        }
    }

    #[tokio::test]
    async fn filters_and_thresholds_reach_the_index() {
        let retriever = retriever();
        let filter = MetadataFilter::eq("kind", "pet");
        let results = retriever
            .retrieve("lazy", 5, None, RetrieveMode::Sparse, Some(&filter))
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].chunk_id.as_deref(), Some("d3"));

        let hits = retriever
            .retrieve_search_results("quick", 5, RetrieveMode::Sparse, Some(&filter))
            .await
            .unwrap();
        assert!(hits.is_empty());

        let thresholded = retriever
            .retrieve("quick", 5, Some(100.0), RetrieveMode::Sparse, None)
            .await
            .unwrap();
        assert!(thresholded.is_empty());
    }

    #[tokio::test]
    async fn batch_retrieve_uses_the_default_and_keeps_query_order() {
        let retriever = retriever();
        let queries = vec!["quick".to_owned(), "dog".to_owned(), "quick".to_owned()];
        let batch = retriever
            .batch_retrieve(&queries, 5, RetrieveMode::Sparse)
            .await
            .unwrap();
        assert_eq!(batch.len(), 3);
        assert_eq!(batch[0][0].chunk_id.as_deref(), Some("d2"));
        assert_eq!(batch[1][0].chunk_id.as_deref(), Some("d3"));
        assert_eq!(batch[0], batch[2]);
        assert!(retriever
            .batch_retrieve(&queries, 5, RetrieveMode::Hybrid)
            .await
            .is_err());
        retriever.close().await;
    }
}
