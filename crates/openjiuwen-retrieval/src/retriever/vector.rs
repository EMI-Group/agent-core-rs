//! Dense vector retriever.
//!
//! Port of the Python `openjiuwen/core/retrieval/retriever/vector_retriever.py`.
//! Python forwards to `VectorStore.search` and falls back to BM25 when the dense
//! channel comes back empty; Rust keeps only the dense half, because the sparse
//! fallback belongs to [`crate::retriever::HybridRetriever`], which owns both
//! channels. Anything that contradicts that split is documented below.
//!
//! Only [`RetrieveMode::Vector`] is served: every other mode returns
//! [`Error::Validation`] naming the supported mode, mirroring Python's
//! `VectorRetriever only supports 'vector' mode` error and the existing
//! [`crate::retriever::SparseRetriever`] behaviour.

use std::sync::Arc;

use async_trait::async_trait;

use openjiuwen_common::error::{Error, Result};

use super::{ensure_query, search_result_to_retrieval, RetrieveMode, Retriever};
use crate::common::{MetadataFilter, RetrievalResult, SearchResult};
use crate::embedding::Embedding;
use crate::vector_store::{VectorQuery, VectorStore};

/// A [`Retriever`] over a [`VectorStore`] that only serves [`RetrieveMode::Vector`].
#[derive(Clone)]
pub struct VectorRetriever {
    store: Arc<dyn VectorStore>,
    embedder: Arc<dyn Embedding>,
}

impl std::fmt::Debug for VectorRetriever {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VectorRetriever")
            .field("store", &self.store.name())
            .field("embedder", &self.embedder.model_name())
            .finish()
    }
}

impl VectorRetriever {
    /// Builds a retriever over `store`, encoding queries with `embedder`.
    pub fn new(store: Arc<dyn VectorStore>, embedder: Arc<dyn Embedding>) -> Self {
        Self { store, embedder }
    }

    /// The shared vector store backing this retriever.
    pub fn store(&self) -> &Arc<dyn VectorStore> {
        &self.store
    }

    /// The shared embedding model used to encode queries.
    pub fn embedder(&self) -> &Arc<dyn Embedding> {
        &self.embedder
    }

    /// Rejects every mode but [`RetrieveMode::Vector`].
    fn ensure_vector_mode(mode: RetrieveMode) -> Result<()> {
        if mode == RetrieveMode::Vector {
            return Ok(());
        }
        Err(Error::validation(format!(
            "VectorRetriever only supports 'vector' mode, got {}",
            mode.as_str()
        )))
    }

    /// Encodes `query` and runs the store search.
    async fn search(
        &self,
        query: &str,
        top_k: usize,
        score_threshold: Option<f32>,
        filters: Option<&MetadataFilter>,
    ) -> Result<Vec<SearchResult>> {
        if top_k == 0 {
            return Ok(Vec::new());
        }
        let vector = self.embedder.embed_query(query).await?;
        let request = VectorQuery {
            vector,
            top_k,
            filter: filters.cloned(),
            score_threshold,
        };
        self.store.search(request).await
    }
}

#[async_trait]
impl Retriever for VectorRetriever {
    async fn retrieve(
        &self,
        query: &str,
        top_k: usize,
        score_threshold: Option<f32>,
        mode: RetrieveMode,
        filters: Option<&MetadataFilter>,
    ) -> Result<Vec<RetrievalResult>> {
        Self::ensure_vector_mode(mode)?;
        ensure_query(query)?;
        Ok(self
            .search(query, top_k, score_threshold, filters)
            .await?
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
        Self::ensure_vector_mode(mode)?;
        ensure_query(query)?;
        self.search(query, top_k, None, filters).await
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
    use async_trait::async_trait;

    use openjiuwen_common::error::ErrorKind;

    use crate::common::Metadata;
    use crate::embedding::validate_query;
    use crate::vector_store::{InMemoryVectorStore, VectorRecord};

    /// Embedder that always returns the same vector, so scores are hand-checkable.
    struct FixedEmbedding(Vec<f32>);

    #[async_trait]
    impl Embedding for FixedEmbedding {
        fn model_name(&self) -> &'static str {
            "fixed"
        }

        fn dimension_hint(&self) -> Option<usize> {
            Some(self.0.len())
        }

        async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
            validate_query(text)?;
            Ok(self.0.clone())
        }

        async fn embed_documents(
            &self,
            texts: &[String],
            _batch_size: Option<usize>,
        ) -> Result<Vec<Vec<f32>>> {
            Ok(texts.iter().map(|_| self.0.clone()).collect())
        }
    }

    /// Store with three vectors: `near` (identical), `mid`, and `far`.
    async fn retriever() -> VectorRetriever {
        let store = InMemoryVectorStore::new();
        store
            .upsert(vec![
                VectorRecord {
                    id: "far".to_owned(),
                    text: "far".to_owned(),
                    vector: vec![0.0, 1.0],
                    metadata: Metadata::from_iter([("doc_id".to_owned(), "doc-far".into())]),
                    doc_id: Some("doc-far".to_owned()),
                    chunk_id: Some("far".to_owned()),
                },
                VectorRecord {
                    id: "near".to_owned(),
                    text: "near".to_owned(),
                    vector: vec![1.0, 0.0],
                    metadata: Metadata::new(),
                    doc_id: Some("doc-near".to_owned()),
                    chunk_id: Some("near".to_owned()),
                },
            ])
            .await
            .unwrap();
        VectorRetriever::new(Arc::new(store), Arc::new(FixedEmbedding(vec![1.0, 0.0])))
    }

    #[tokio::test]
    async fn vector_mode_orders_hits_and_projects_ids() {
        let retriever = retriever().await;
        let results = retriever
            .retrieve("the query", 5, None, RetrieveMode::Vector, None)
            .await
            .unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].chunk_id.as_deref(), Some("near"));
        assert_eq!(results[0].doc_id.as_deref(), Some("doc-near"));
        assert_eq!(results[1].text, "far");
        assert!(results[0].score > results[1].score);

        let hits = retriever
            .retrieve_search_results("the query", 5, RetrieveMode::Vector, None)
            .await
            .unwrap();
        assert_eq!(hits[0].id, "near");
        assert_eq!(retriever.store().name(), "default");
        assert_eq!(retriever.embedder().model_name(), "fixed");
    }

    #[tokio::test]
    async fn other_modes_are_rejected_with_a_validation_error() {
        let retriever = retriever().await;
        for mode in [RetrieveMode::Sparse, RetrieveMode::Hybrid] {
            let error = retriever
                .retrieve("the query", 5, None, mode, None)
                .await
                .unwrap_err();
            assert_eq!(error.kind(), ErrorKind::Validation);
            assert!(error.to_string().contains("only supports 'vector' mode"));
            assert!(error.to_string().contains(mode.as_str()));
            assert!(retriever
                .retrieve_search_results("the query", 5, mode, None)
                .await
                .is_err());
        }
    }

    #[tokio::test]
    async fn blank_queries_are_validation_errors_and_zero_top_k_is_empty() {
        let retriever = retriever().await;
        let error = retriever
            .retrieve("   ", 5, None, RetrieveMode::Vector, None)
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("must not be empty"));

        assert!(retriever
            .retrieve("the query", 0, None, RetrieveMode::Vector, None)
            .await
            .unwrap()
            .is_empty());
        assert!(retriever
            .retrieve_search_results("the query", 0, RetrieveMode::Vector, None)
            .await
            .unwrap()
            .is_empty());
    }

    #[tokio::test]
    async fn thresholds_reach_the_store() {
        let retriever = retriever().await;
        let all = retriever
            .retrieve("the query", 5, None, RetrieveMode::Vector, None)
            .await
            .unwrap();
        let thresholded = retriever
            .retrieve("the query", 5, Some(0.9), RetrieveMode::Vector, None)
            .await
            .unwrap();
        assert_eq!(thresholded.len(), 1);
        assert!(all.len() > thresholded.len());
        assert!(retriever
            .retrieve("the query", 5, Some(1.1), RetrieveMode::Vector, None)
            .await
            .unwrap()
            .is_empty());
    }
}
