//! Hybrid retriever: dense + sparse channels fused with reciprocal rank fusion.
//!
//! Port of the Python `openjiuwen/core/retrieval/retriever/hybrid_retriever.py`
//! with one documented difference: Python delegates the fusion to
//! `VectorStore.hybrid_search(alpha=...)`, which this crate does not have, so the
//! Rust retriever over-fetches both channels and fuses them with
//! [`rrf_fusion`](crate::retriever::fusion::rrf_fusion) (`k = 60`, Python's
//! `utils/fusion.py` default).
//!
//! The fusion method is configurable through [`HybridRetriever::with_rank_config`]:
//! the default is reciprocal rank fusion with `k = 60`, and a
//! [`RankConfig::Weighted`] config selects the weighted ranker instead.
//! The two channels are supplied in the fixed ranking order — the name-embedding
//! channel (unused here) is empty, the dense channel is
//! [`CONTENT_DENSE`](crate::common::CONTENT_DENSE), and the sparse channel is
//! [`CONTENT_SPARSE`](crate::common::CONTENT_SPARSE).
//!
//! Mode semantics:
//! - [`RetrieveMode::Vector`] serves the dense channel and falls back to the
//!   sparse channel when the dense channel returns nothing (Python parity).
//! - [`RetrieveMode::Sparse`] serves the sparse channel only.
//! - [`RetrieveMode::Hybrid`] over-fetches `top_k * candidate_factor` candidates
//!   from both channels, applies `score_threshold` to the **dense** channel only
//!   (sparse BM25 scores live on a different scale), fuses, and truncates to
//!   `top_k`.
//!
//! `top_k == 0` yields an empty result vector instead of an error, so a caller
//! that shrinks `top_k` to zero does not have to special-case this retriever.

use std::sync::Arc;

use async_trait::async_trait;

use openjiuwen_common::error::Result;

use super::{
    ensure_query, search_result_to_retrieval, RetrieveMode, Retriever, SparseRetriever,
    VectorRetriever, DEFAULT_RRF_K,
};
use crate::common::{MetadataFilter, RankConfig, RetrievalResult, RrfRankConfig, SearchResult};
use crate::embedding::Embedding;
use crate::sparse::Bm25Index;
use crate::vector_store::VectorStore;

/// Default over-fetch factor for each channel in hybrid mode.
pub const DEFAULT_CANDIDATE_FACTOR: usize = 2;

/// A [`Retriever`] that fuses a dense and a sparse channel.
///
/// The default fusion is reciprocal rank fusion with `k = 60`; see
/// [`Self::with_rank_config`] for the weighted alternative.
pub struct HybridRetriever {
    dense: Arc<dyn Retriever>,
    sparse: Arc<dyn Retriever>,
    candidate_factor: usize,
    rank_config: RankConfig,
}

impl std::fmt::Debug for HybridRetriever {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HybridRetriever")
            .field("candidate_factor", &self.candidate_factor)
            .field("rank_config", &self.rank_config)
            .finish_non_exhaustive()
    }
}

impl HybridRetriever {
    /// Builds a retriever over an explicit dense and sparse channel.
    ///
    /// [`Self::candidate_factor`] defaults to [`DEFAULT_CANDIDATE_FACTOR`] and the
    /// rank config to [`RankConfig::Rrf`] with `k = `[`DEFAULT_RRF_K`].
    pub fn new(dense: Arc<dyn Retriever>, sparse: Arc<dyn Retriever>) -> Self {
        Self {
            dense,
            sparse,
            candidate_factor: DEFAULT_CANDIDATE_FACTOR,
            rank_config: RankConfig::Rrf(RrfRankConfig::new(DEFAULT_RRF_K)),
        }
    }

    /// Builds the dense [`VectorRetriever`] and sparse [`SparseRetriever`] pair
    /// from the shared store, embedder, and index.
    pub fn from_parts(
        store: Arc<dyn VectorStore>,
        embedder: Arc<dyn Embedding>,
        index: Arc<Bm25Index>,
    ) -> Self {
        Self::new(
            Arc::new(VectorRetriever::new(store, embedder)),
            Arc::new(SparseRetriever::new(index)),
        )
    }

    /// Sets how many candidates each channel fetches per requested result.
    ///
    /// A factor of `0` behaves like `1`, so the over-fetch can never make the
    /// retriever return less than `top_k` candidates per channel.
    #[must_use]
    pub fn with_candidate_factor(mut self, candidate_factor: usize) -> Self {
        self.candidate_factor = candidate_factor;
        self
    }

    /// Sets the RRF smoothing constant, selecting the RRF ranker.
    ///
    /// Kept for parity with the pre-config API; equivalent to
    /// `with_rank_config(RankConfig::Rrf(RrfRankConfig::new(rrf_k)))` but keeps
    /// any existing RRF channel flags.
    #[must_use]
    pub fn with_rrf_k(mut self, rrf_k: usize) -> Self {
        self.rank_config = self.rank_config.with_rrf_k(rrf_k);
        self
    }

    /// Selects the rank-fusion configuration used in [`RetrieveMode::Hybrid`].
    ///
    /// The default is [`RankConfig::Rrf`] with `k = 60`; pass
    /// [`RankConfig::Weighted`] to use the weighted ranker. The candidate
    /// over-fetch and the dense-only `score_threshold` apply to both methods.
    #[must_use]
    pub fn with_rank_config(mut self, rank_config: RankConfig) -> Self {
        self.rank_config = rank_config;
        self
    }

    /// The dense channel.
    pub fn dense(&self) -> &Arc<dyn Retriever> {
        &self.dense
    }

    /// The sparse channel.
    pub fn sparse(&self) -> &Arc<dyn Retriever> {
        &self.sparse
    }

    /// The effective over-fetch factor (never zero).
    pub fn candidate_factor(&self) -> usize {
        self.candidate_factor.max(1)
    }

    /// The active fusion configuration.
    pub fn rank_config(&self) -> &RankConfig {
        &self.rank_config
    }

    /// The active RRF smoothing constant.
    ///
    /// Returns [`DEFAULT_RRF_K`] when the active config is
    /// [`RankConfig::Weighted`], which has no RRF constant.
    pub fn rrf_k(&self) -> usize {
        self.rank_config.rrf_k().unwrap_or(DEFAULT_RRF_K)
    }

    /// Runs both channels and fuses them with the active rank config.
    async fn hybrid_search(
        &self,
        query: &str,
        top_k: usize,
        score_threshold: Option<f32>,
        filters: Option<&MetadataFilter>,
    ) -> Result<Vec<SearchResult>> {
        let candidates = top_k.saturating_mul(self.candidate_factor());
        let mut dense = self
            .dense
            .retrieve_search_results(query, candidates, RetrieveMode::Vector, filters)
            .await?;
        if let Some(threshold) = score_threshold {
            dense.retain(|result| result.score >= threshold);
        }
        let sparse = self
            .sparse
            .retrieve_search_results(query, candidates, RetrieveMode::Sparse, filters)
            .await?;
        // Fixed channel order: name-embedding (unused), content-dense, content-sparse.
        let mut fused = self.rank_config.apply(vec![Vec::new(), dense, sparse]);
        fused.truncate(top_k);
        Ok(fused)
    }
}

#[async_trait]
impl Retriever for HybridRetriever {
    async fn retrieve(
        &self,
        query: &str,
        top_k: usize,
        score_threshold: Option<f32>,
        mode: RetrieveMode,
        filters: Option<&MetadataFilter>,
    ) -> Result<Vec<RetrievalResult>> {
        ensure_query(query)?;
        if top_k == 0 {
            return Ok(Vec::new());
        }
        match mode {
            RetrieveMode::Vector => {
                let mut dense = self
                    .dense
                    .retrieve(query, top_k, None, RetrieveMode::Vector, filters)
                    .await?;
                if dense.is_empty() {
                    // Python parity: an empty dense channel falls back to BM25.
                    return self
                        .sparse
                        .retrieve(query, top_k, None, RetrieveMode::Sparse, filters)
                        .await;
                }
                if let Some(threshold) = score_threshold {
                    dense.retain(|result| result.score >= threshold);
                }
                dense.truncate(top_k);
                Ok(dense)
            }
            RetrieveMode::Sparse => {
                self.sparse
                    .retrieve(query, top_k, score_threshold, RetrieveMode::Sparse, filters)
                    .await
            }
            RetrieveMode::Hybrid => Ok(self
                .hybrid_search(query, top_k, score_threshold, filters)
                .await?
                .into_iter()
                .map(search_result_to_retrieval)
                .collect()),
        }
    }

    async fn retrieve_search_results(
        &self,
        query: &str,
        top_k: usize,
        mode: RetrieveMode,
        filters: Option<&MetadataFilter>,
    ) -> Result<Vec<SearchResult>> {
        ensure_query(query)?;
        if top_k == 0 {
            return Ok(Vec::new());
        }
        match mode {
            RetrieveMode::Vector => {
                let dense = self
                    .dense
                    .retrieve_search_results(query, top_k, RetrieveMode::Vector, filters)
                    .await?;
                if dense.is_empty() {
                    return self
                        .sparse
                        .retrieve_search_results(query, top_k, RetrieveMode::Sparse, filters)
                        .await;
                }
                Ok(dense)
            }
            RetrieveMode::Sparse => {
                self.sparse
                    .retrieve_search_results(query, top_k, RetrieveMode::Sparse, filters)
                    .await
            }
            RetrieveMode::Hybrid => self.hybrid_search(query, top_k, None, filters).await,
        }
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

    /// Channel returning a fixed list of hits, whatever the query.
    struct FixedChannel {
        hits: Vec<SearchResult>,
        calls: Arc<parking_lot::Mutex<Vec<(usize, RetrieveMode)>>>,
    }

    #[async_trait]
    impl Retriever for FixedChannel {
        async fn retrieve(
            &self,
            query: &str,
            top_k: usize,
            score_threshold: Option<f32>,
            mode: RetrieveMode,
            filters: Option<&MetadataFilter>,
        ) -> Result<Vec<RetrievalResult>> {
            let hits = self
                .retrieve_search_results(query, top_k, mode, filters)
                .await?;
            let mut results: Vec<RetrievalResult> =
                hits.into_iter().map(search_result_to_retrieval).collect();
            if let Some(threshold) = score_threshold {
                results.retain(|result| result.score >= threshold);
            }
            Ok(results)
        }

        async fn retrieve_search_results(
            &self,
            _query: &str,
            top_k: usize,
            mode: RetrieveMode,
            _filters: Option<&MetadataFilter>,
        ) -> Result<Vec<SearchResult>> {
            self.calls.lock().push((top_k, mode));
            Ok(self.hits.iter().take(top_k).cloned().collect())
        }
    }

    fn hit(id: &str, text: &str, score: f32) -> SearchResult {
        SearchResult {
            id: id.to_owned(),
            text: text.to_owned(),
            score,
            metadata: Metadata::new(),
        }
    }

    #[allow(clippy::type_complexity)]
    fn channels(
        dense: Vec<SearchResult>,
        sparse: Vec<SearchResult>,
    ) -> (
        HybridRetriever,
        Arc<parking_lot::Mutex<Vec<(usize, RetrieveMode)>>>,
        Arc<parking_lot::Mutex<Vec<(usize, RetrieveMode)>>>,
    ) {
        let dense_calls = Arc::new(parking_lot::Mutex::new(Vec::new()));
        let sparse_calls = Arc::new(parking_lot::Mutex::new(Vec::new()));
        let retriever = HybridRetriever::new(
            Arc::new(FixedChannel {
                hits: dense,
                calls: Arc::clone(&dense_calls),
            }),
            Arc::new(FixedChannel {
                hits: sparse,
                calls: Arc::clone(&sparse_calls),
            }),
        );
        (retriever, dense_calls, sparse_calls)
    }

    #[tokio::test]
    async fn hybrid_mode_orders_by_reciprocal_rank_fusion() {
        let (retriever, dense_calls, sparse_calls) = channels(
            vec![hit("d1", "alpha", 0.9), hit("d2", "beta", 0.8)],
            vec![hit("s1", "gamma", 12.0), hit("s2", "beta", 9.0)],
        );
        let results = retriever
            .retrieve("query", 3, None, RetrieveMode::Hybrid, None)
            .await
            .unwrap();
        let texts: Vec<&str> = results.iter().map(|result| result.text.as_str()).collect();
        // beta scores 1/62 + 1/61, alpha and gamma score 1/62 across one list each.
        assert_eq!(texts, vec!["beta", "alpha", "gamma"]);
        assert!(results[0].score > results[1].score);
        assert_eq!(results[0].chunk_id.as_deref(), Some("d2"));
        // Over-fetch: 3 requested results fetch 2 * 3 candidates per channel.
        assert_eq!(dense_calls.lock().as_slice(), &[(6, RetrieveMode::Vector)]);
        assert_eq!(sparse_calls.lock().as_slice(), &[(6, RetrieveMode::Sparse)]);
        assert_eq!(retriever.candidate_factor(), DEFAULT_CANDIDATE_FACTOR);
        assert_eq!(retriever.rrf_k(), DEFAULT_RRF_K);
    }

    #[tokio::test]
    async fn hybrid_threshold_applies_to_the_dense_channel_only() {
        let (retriever, _, _) = channels(
            vec![hit("d1", "alpha", 0.9), hit("d2", "beta", 0.2)],
            vec![hit("s1", "gamma", 1.0)],
        );
        let results = retriever
            .retrieve("query", 3, Some(0.5), RetrieveMode::Hybrid, None)
            .await
            .unwrap();
        let texts: Vec<&str> = results.iter().map(|result| result.text.as_str()).collect();
        // `beta` is dropped by the dense threshold, leaving alpha and gamma (BM25
        // scores are never thresholded).
        assert_eq!(texts, vec!["alpha", "gamma"]);
    }

    #[tokio::test]
    async fn vector_mode_falls_back_to_sparse_when_dense_is_empty() {
        let (retriever, _, sparse_calls) =
            channels(Vec::new(), vec![hit("s1", "only-sparse", 7.0)]);
        let results = retriever
            .retrieve("query", 2, None, RetrieveMode::Vector, None)
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].text, "only-sparse");
        assert_eq!(results[0].chunk_id.as_deref(), Some("s1"));
        assert_eq!(sparse_calls.lock().len(), 1);

        let (retriever, _, _) = channels(vec![hit("d1", "dense", 0.4)], Vec::new());
        let results = retriever
            .retrieve("query", 2, Some(0.5), RetrieveMode::Vector, None)
            .await
            .unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn sparse_mode_and_edge_cases_are_delegated() {
        let (retriever, dense_calls, _) = channels(
            vec![hit("d1", "alpha", 0.9)],
            vec![hit("s1", "sparse", 3.0)],
        );
        let results = retriever
            .retrieve("query", 2, None, RetrieveMode::Sparse, None)
            .await
            .unwrap();
        assert_eq!(results[0].text, "sparse");
        assert!(dense_calls.lock().is_empty());

        assert!(retriever
            .retrieve("query", 0, None, RetrieveMode::Hybrid, None)
            .await
            .unwrap()
            .is_empty());
        let error = retriever
            .retrieve(" ", 2, None, RetrieveMode::Hybrid, None)
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);

        let hits = retriever
            .retrieve_search_results("query", 2, RetrieveMode::Hybrid, None)
            .await
            .unwrap();
        assert_eq!(hits[0].text, "alpha");
        assert_eq!(hits.len(), 2);
    }
}
