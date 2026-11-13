//! Graph-expansion retriever.
//!
//! Port of Python's `openjiuwen/core/retrieval/retriever/graph_retriever.py`
//! `graph_expansion` flow, adapted to this crate's [`GraphStore`] seam: Python
//! stores triples as text in a vector index and walks them with beam search,
//! while Rust reads the triples out of a triple store and turns each triple back
//! into a chunk query.
//!
//! The flow is:
//! 1. retrieve chunks in the requested mode through the wrapped chunk retriever;
//! 2. collect entity candidates from the query and the top chunk texts with the
//!    graph module's rule-based extractor
//!    ([`RuleTripleExtractor::entities_in_text`]) — the public entity helper, so
//!    no duplicate entity logic lives here;
//! 3. fetch `neighbors(entity, max_hops)` triples, deduplicated by normalized key
//!    and bounded to [`MAX_TRIPLES`];
//! 4. expand by querying the chunk retriever with each triple's text, bounded to
//!    `expansion_limit` triples;
//! 5. fuse the expanded chunks with the original chunks through
//!    [`rrf_fusion`] and truncate to `top_k`.
//!
//! Python degrades to the plain chunk results whenever expansion is impossible or
//! fails; Rust does the same and logs the reason through `tracing` at debug level,
//! so a broken graph never fails a retrieval call or loses the chunk hits.

use std::collections::BTreeSet;
use std::sync::Arc;

use async_trait::async_trait;

use openjiuwen_common::error::Result;
use openjiuwen_common::logging::debug;

use super::fusion::{rrf_fusion, DEFAULT_RRF_K};
use super::{ensure_query, search_result_to_retrieval, RetrieveMode, Retriever};
use crate::common::{MetadataFilter, RetrievalResult, SearchResult};
use crate::graph::{GraphStore, RuleTripleExtractor, Triple};

/// Default number of expansion hops taken from each entity.
pub const DEFAULT_MAX_HOPS: usize = 2;

/// Default number of triples expanded into chunk queries.
pub const DEFAULT_EXPANSION_LIMIT: usize = 5;

/// Upper bound on the triples collected from the graph store.
pub const MAX_TRIPLES: usize = 32;

/// How many chunk hits contribute entity candidates.
pub const ENTITY_CHUNK_LIMIT: usize = 3;

/// A [`Retriever`] that expands chunk hits through a [`GraphStore`].
pub struct GraphRetriever {
    chunk_retriever: Arc<dyn Retriever>,
    graph_store: Arc<dyn GraphStore>,
    max_hops: usize,
    expansion_limit: usize,
    triple_mode: RetrieveMode,
}

impl std::fmt::Debug for GraphRetriever {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GraphRetriever")
            .field("max_hops", &self.max_hops)
            .field("expansion_limit", &self.expansion_limit)
            .field("triple_mode", &self.triple_mode)
            .finish_non_exhaustive()
    }
}

impl GraphRetriever {
    /// Builds a retriever over `chunk_retriever` and `graph_store`.
    ///
    /// Defaults: [`DEFAULT_MAX_HOPS`] hops, [`DEFAULT_EXPANSION_LIMIT`] expanded
    /// triples, and [`RetrieveMode::Sparse`] for the triple lookups (triple text
    /// is lexically distinctive, so BM25 is the cheap offline default).
    pub fn new(chunk_retriever: Arc<dyn Retriever>, graph_store: Arc<dyn GraphStore>) -> Self {
        Self {
            chunk_retriever,
            graph_store,
            max_hops: DEFAULT_MAX_HOPS,
            expansion_limit: DEFAULT_EXPANSION_LIMIT,
            triple_mode: RetrieveMode::Sparse,
        }
    }

    /// Sets how many hops each entity expands.
    #[must_use]
    pub fn with_max_hops(mut self, max_hops: usize) -> Self {
        self.max_hops = max_hops;
        self
    }

    /// Sets how many triples are turned into chunk queries.
    ///
    /// `0` disables expansion: the retriever then returns the original chunks.
    #[must_use]
    pub fn with_expansion_limit(mut self, expansion_limit: usize) -> Self {
        self.expansion_limit = expansion_limit;
        self
    }

    /// Sets the mode used to query the chunk retriever with triple text.
    #[must_use]
    pub fn with_triple_mode(mut self, triple_mode: RetrieveMode) -> Self {
        self.triple_mode = triple_mode;
        self
    }

    /// The wrapped chunk retriever.
    pub fn chunk_retriever(&self) -> &Arc<dyn Retriever> {
        &self.chunk_retriever
    }

    /// The triple store used for expansion.
    pub fn graph_store(&self) -> &Arc<dyn GraphStore> {
        &self.graph_store
    }

    /// The configured hop count.
    pub fn max_hops(&self) -> usize {
        self.max_hops
    }

    /// The configured expansion limit.
    pub fn expansion_limit(&self) -> usize {
        self.expansion_limit
    }

    /// The mode used for triple-text chunk lookups.
    pub fn triple_mode(&self) -> RetrieveMode {
        self.triple_mode
    }

    /// Collects entity candidates from the query, then from the top chunk texts.
    fn entity_candidates(query: &str, chunks: &[SearchResult]) -> Vec<String> {
        let extractor = RuleTripleExtractor::new();
        let texts = std::iter::once(query).chain(
            chunks
                .iter()
                .take(ENTITY_CHUNK_LIMIT)
                .map(|chunk| chunk.text.as_str()),
        );
        let mut seen = BTreeSet::new();
        let mut names = Vec::new();
        for text in texts {
            for entity in extractor.entities_in_text(text) {
                if !entity.name.trim().is_empty() && seen.insert(entity.normalized_name()) {
                    names.push(entity.name);
                }
            }
        }
        names
    }

    /// Walks the triples around every entity, deduplicated and bounded.
    async fn collect_triples(&self, query: &str, chunks: &[SearchResult]) -> Vec<Triple> {
        let entities = Self::entity_candidates(query, chunks);
        if entities.is_empty() {
            debug!("graph expansion skipped: no entity candidates in the query or chunks");
            return Vec::new();
        }
        let mut triples: Vec<Triple> = Vec::new();
        let mut seen: BTreeSet<String> = BTreeSet::new();
        'entities: for entity in &entities {
            let found = match self.graph_store.neighbors(entity, self.max_hops).await {
                Ok(found) => found,
                Err(error) => {
                    debug!(entity = %entity, error = %error, "skipping a graph entity");
                    continue;
                }
            };
            for triple in found {
                if triples.len() >= MAX_TRIPLES {
                    break 'entities;
                }
                if seen.insert(triple.normalized_key()) {
                    triples.push(triple);
                }
            }
        }
        triples
    }

    /// Runs the chunk retrieval plus the graph expansion, returning raw hits.
    async fn expand(
        &self,
        query: &str,
        top_k: usize,
        score_threshold: Option<f32>,
        mode: RetrieveMode,
        filters: Option<&MetadataFilter>,
    ) -> Result<Vec<SearchResult>> {
        let mut chunks = self
            .chunk_retriever
            .retrieve_search_results(query, top_k, mode, filters)
            .await?;
        if let Some(threshold) = score_threshold {
            chunks.retain(|chunk| chunk.score >= threshold);
        }
        chunks.truncate(top_k);
        if chunks.is_empty() || self.expansion_limit == 0 {
            return Ok(chunks);
        }
        let triples = self.collect_triples(query, &chunks).await;
        if triples.is_empty() {
            debug!("graph expansion skipped: no triples around the query entities");
            return Ok(chunks);
        }
        let mut expanded: Vec<SearchResult> = Vec::new();
        for triple in triples.iter().take(self.expansion_limit) {
            let text = triple.text();
            match self
                .chunk_retriever
                .retrieve_search_results(&text, 1, self.triple_mode, None)
                .await
            {
                Ok(hits) => {
                    if let Some(hit) = hits.into_iter().next() {
                        expanded.push(hit);
                    }
                }
                Err(error) => {
                    debug!(triple = %text, error = %error, "skipping a graph expansion query");
                }
            }
        }
        if expanded.is_empty() {
            debug!("graph expansion produced no new chunks");
            return Ok(chunks);
        }
        let mut fused = rrf_fusion(vec![expanded, chunks], DEFAULT_RRF_K);
        fused.truncate(top_k);
        Ok(fused)
    }
}

#[async_trait]
impl Retriever for GraphRetriever {
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
        Ok(self
            .expand(query, top_k, score_threshold, mode, filters)
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
        ensure_query(query)?;
        if top_k == 0 {
            return Ok(Vec::new());
        }
        self.expand(query, top_k, None, mode, filters).await
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

    use openjiuwen_common::error::{Error, ErrorKind};

    use crate::common::Metadata;
    use crate::graph::InMemoryGraphStore;

    /// Channel serving one hit per query keyword it recognises.
    struct ProbeChannel {
        hits: Vec<(&'static str, SearchResult)>,
    }

    #[async_trait]
    impl Retriever for ProbeChannel {
        async fn retrieve(
            &self,
            query: &str,
            top_k: usize,
            _score_threshold: Option<f32>,
            mode: RetrieveMode,
            filters: Option<&MetadataFilter>,
        ) -> Result<Vec<RetrievalResult>> {
            Ok(self
                .retrieve_search_results(query, top_k, mode, filters)
                .await?
                .into_iter()
                .map(search_result_to_retrieval)
                .collect())
        }

        async fn retrieve_search_results(
            &self,
            query: &str,
            top_k: usize,
            _mode: RetrieveMode,
            _filters: Option<&MetadataFilter>,
        ) -> Result<Vec<SearchResult>> {
            if query.trim() == "boom" {
                return Err(Error::retrieval("channel failure"));
            }
            let lowered = query.to_lowercase();
            Ok(self
                .hits
                .iter()
                .filter(|(keyword, _)| lowered.contains(keyword))
                .map(|(_, hit)| hit.clone())
                .take(top_k)
                .collect())
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

    fn channel() -> Arc<dyn Retriever> {
        Arc::new(ProbeChannel {
            hits: vec![
                ("ownership", hit("chunk-a", "Rust ownership model", 1.0)),
                (
                    "borrow checker",
                    hit("chunk-b", "the Borrow Checker guide", 0.5),
                ),
            ],
        })
    }

    fn store() -> Arc<dyn GraphStore> {
        let store = InMemoryGraphStore::with_triples(vec![
            Triple::new("Rust", "has", "Borrow Checker"),
            Triple::new("Rust", "owns", "Ownership"),
        ])
        .unwrap();
        Arc::new(store)
    }

    #[tokio::test]
    async fn expansion_finds_a_graph_connected_chunk() {
        let retriever = GraphRetriever::new(channel(), store());
        let results = retriever
            .retrieve("Rust ownership", 2, None, RetrieveMode::Sparse, None)
            .await
            .unwrap();
        let texts: Vec<&str> = results.iter().map(|result| result.text.as_str()).collect();
        assert_eq!(
            texts,
            vec!["Rust ownership model", "the Borrow Checker guide"]
        );
        assert_eq!(results[1].chunk_id.as_deref(), Some("chunk-b"));
        assert_eq!(retriever.max_hops(), DEFAULT_MAX_HOPS);
        assert_eq!(retriever.expansion_limit(), DEFAULT_EXPANSION_LIMIT);
        assert_eq!(retriever.triple_mode(), RetrieveMode::Sparse);
        assert_eq!(retriever.graph_store().len().await.unwrap(), 2);
        assert!(Arc::strong_count(retriever.chunk_retriever()) >= 1);
    }

    #[tokio::test]
    async fn an_empty_graph_degrades_to_the_original_chunks() {
        let retriever = GraphRetriever::new(channel(), Arc::new(InMemoryGraphStore::new()));
        let results = retriever
            .retrieve("Rust ownership", 2, None, RetrieveMode::Sparse, None)
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].text, "Rust ownership model");
        assert_eq!(results[0].chunk_id.as_deref(), Some("chunk-a"));
    }

    #[tokio::test]
    async fn edges_are_guarded_and_channel_failures_propagate() {
        let retriever = GraphRetriever::new(channel(), store());
        let error = retriever
            .retrieve("boom", 2, None, RetrieveMode::Sparse, None)
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Retrieval);

        let error = retriever
            .retrieve("  ", 2, None, RetrieveMode::Sparse, None)
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);

        assert!(retriever
            .retrieve("Rust ownership", 0, None, RetrieveMode::Sparse, None)
            .await
            .unwrap()
            .is_empty());

        let disabled = GraphRetriever::new(channel(), store()).with_expansion_limit(0);
        assert_eq!(
            disabled
                .retrieve_search_results("Rust ownership", 2, RetrieveMode::Sparse, None)
                .await
                .unwrap()
                .len(),
            1
        );

        let hops = GraphRetriever::new(channel(), store())
            .with_max_hops(0)
            .with_triple_mode(RetrieveMode::Hybrid);
        assert_eq!(hops.triple_mode(), RetrieveMode::Hybrid);
        assert_eq!(hops.max_hops(), 0);
        assert_eq!(hops.expansion_limit(), DEFAULT_EXPANSION_LIMIT);
    }
}
