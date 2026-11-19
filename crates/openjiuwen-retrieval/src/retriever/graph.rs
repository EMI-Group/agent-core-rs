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
//! 4. expand by querying the chunk retriever with each triple's text, in
//!    `triple_mode` and with the caller's metadata filter, bounded to
//!    `expansion_limit` triples;
//! 5. fuse the expanded chunks with the original chunks through
//!    [`rrf_fusion`] and truncate to `top_k`.
//!
//! Python degrades to the plain chunk results whenever expansion is impossible or
//! fails; Rust does the same, but no longer hides *why* it degraded. Every failed
//! graph step is recorded in a [`GraphExpansionReport`] and logged at `warn`
//! level, while genuine no-ops (no entity candidates, no triples around the
//! entities, no new chunks, expansion disabled) stay `debug`-only and are not
//! counted as degradations. A failed expansion still never fails a retrieval call
//! or loses the chunk hits: when the report holds degradations, every returned hit
//! carries `metadata["graph_degraded"] = true` so a caller can tell a clean run
//! from a broken graph store. The report is available through
//! [`GraphRetriever::retrieve_with_report`] and
//! [`GraphRetriever::retrieve_search_results_with_report`]; the [`Retriever`] trait
//! methods delegate to them and drop the report.

use std::collections::BTreeSet;
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use openjiuwen_common::error::Result;
use openjiuwen_common::logging::{debug, warn};

use super::fusion::{rrf_fusion, DEFAULT_RRF_K};
use super::{ensure_query, search_result_to_retrieval, RetrieveMode, Retriever};
use crate::common::{MetadataFilter, RetrievalResult, SearchResult};
use crate::graph::{GraphStore, RuleTripleExtractor, Triple};

/// Metadata key set to `true` on every hit of a call whose graph expansion
/// degraded.
///
/// The key is absent on a clean run, so a downstream consumer can distinguish
/// "the graph held nothing for this query" from "the graph call failed".
pub const GRAPH_DEGRADED_FIELD: &str = "graph_degraded";

/// The graph step a [`GraphDegradation`] was recorded for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GraphStage {
    /// Entity-candidate resolution from the query and the top chunk texts.
    ///
    /// The rule-based extractor is deterministic and infallible, so no current
    /// code path produces this stage; it exists so a future extractor seam (for
    /// example a model-backed one) can report a failed candidate lookup with the
    /// same mechanism.
    EntityLookup,
    /// A `neighbors(entity, max_hops)` lookup against the [`GraphStore`] failed.
    NeighborExpansion,
    /// A chunk query issued for a triple's text failed.
    ChunkQuery,
}

/// One failed step of a graph expansion.
///
/// Records which [`GraphStage`] failed, the entity (`NeighborExpansion`) or
/// triple subject (`ChunkQuery`) it was about — `None` when the stage is not tied
/// to a single subject — and the error message the store or channel produced.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphDegradation {
    /// The step that failed.
    pub stage: GraphStage,
    /// The entity or triple subject the failing step was about, when known.
    #[serde(default)]
    pub subject: Option<String>,
    /// The error message produced by the failing step.
    pub error: String,
}

/// What a graph expansion did, and what failed while doing it.
///
/// Returned alongside the hits by
/// [`GraphRetriever::retrieve_search_results_with_report`] and
/// [`GraphRetriever::retrieve_with_report`]. A report with no
/// [`degradations`](Self::degradations) describes a clean run — including the
/// non-degrading no-ops (an empty graph, no entity candidates, no expansion).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphExpansionReport {
    /// Entity candidates the retriever looked up, in candidate order.
    #[serde(default)]
    pub entities: Vec<String>,
    /// Distinct triples collected from the store, before the `expansion_limit`
    /// truncation.
    #[serde(default)]
    pub triple_count: usize,
    /// Triple-text chunk queries issued (attempted, including the failed ones),
    /// bounded by `expansion_limit`.
    #[serde(default)]
    pub expanded_queries: usize,
    /// The failed graph steps, in the order they failed.
    #[serde(default)]
    pub degradations: Vec<GraphDegradation>,
}

impl GraphExpansionReport {
    /// Whether at least one graph step failed.
    ///
    /// When this is `true`, the hits returned by the same call carry
    /// `metadata["graph_degraded"] = true` (see [`GRAPH_DEGRADED_FIELD`]).
    pub fn is_degraded(&self) -> bool {
        !self.degradations.is_empty()
    }
}

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

    /// Retrieves raw hits and reports what the graph expansion did.
    ///
    /// Behaves exactly like [`Retriever::retrieve_search_results`], but returns a
    /// [`GraphExpansionReport`] alongside the hits instead of only logging why
    /// the expansion fell back. A blank query is `Error::Validation`, and
    /// `top_k == 0` returns no hits with a default (clean) report.
    pub async fn retrieve_search_results_with_report(
        &self,
        query: &str,
        top_k: usize,
        mode: RetrieveMode,
        filters: Option<&MetadataFilter>,
    ) -> Result<(Vec<SearchResult>, GraphExpansionReport)> {
        ensure_query(query)?;
        if top_k == 0 {
            return Ok((Vec::new(), GraphExpansionReport::default()));
        }
        self.expand(query, top_k, None, mode, filters).await
    }

    /// Retrieves [`RetrievalResult`]s and reports what the graph expansion did.
    ///
    /// Behaves exactly like [`Retriever::retrieve`], but returns a
    /// [`GraphExpansionReport`] alongside the results. When the report
    /// [is degraded](GraphExpansionReport::is_degraded), every result carries
    /// `metadata["graph_degraded"] = true`.
    pub async fn retrieve_with_report(
        &self,
        query: &str,
        top_k: usize,
        score_threshold: Option<f32>,
        mode: RetrieveMode,
        filters: Option<&MetadataFilter>,
    ) -> Result<(Vec<RetrievalResult>, GraphExpansionReport)> {
        ensure_query(query)?;
        if top_k == 0 {
            return Ok((Vec::new(), GraphExpansionReport::default()));
        }
        let (hits, report) = self
            .expand(query, top_k, score_threshold, mode, filters)
            .await?;
        Ok((
            hits.into_iter().map(search_result_to_retrieval).collect(),
            report,
        ))
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
    ///
    /// Entity candidates are recorded in `report.entities`; a failed
    /// `neighbors` lookup is recorded as a
    /// [`GraphStage::NeighborExpansion`](GraphStage::NeighborExpansion)
    /// degradation and the walk continues with the other entities.
    async fn collect_triples(
        &self,
        query: &str,
        chunks: &[SearchResult],
        report: &mut GraphExpansionReport,
    ) -> Vec<Triple> {
        let entities = Self::entity_candidates(query, chunks);
        report.entities.clone_from(&entities);
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
                    warn!(
                        entity = %entity,
                        error = %error,
                        "graph entity lookup failed; continuing without its triples"
                    );
                    report.degradations.push(GraphDegradation {
                        stage: GraphStage::NeighborExpansion,
                        subject: Some(entity.clone()),
                        error: error.to_string(),
                    });
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

    /// Runs the chunk retrieval plus the graph expansion, returning raw hits and
    /// the expansion report.
    async fn expand(
        &self,
        query: &str,
        top_k: usize,
        score_threshold: Option<f32>,
        mode: RetrieveMode,
        filters: Option<&MetadataFilter>,
    ) -> Result<(Vec<SearchResult>, GraphExpansionReport)> {
        let mut report = GraphExpansionReport::default();
        let mut chunks = self
            .chunk_retriever
            .retrieve_search_results(query, top_k, mode, filters)
            .await?;
        if let Some(threshold) = score_threshold {
            chunks.retain(|chunk| chunk.score >= threshold);
        }
        chunks.truncate(top_k);
        if chunks.is_empty() || self.expansion_limit == 0 {
            return Ok((chunks, report));
        }
        let triples = self.collect_triples(query, &chunks, &mut report).await;
        report.triple_count = triples.len();
        if triples.is_empty() {
            debug!("graph expansion skipped: no triples around the query entities");
            return Ok(Self::finish(chunks, report));
        }
        let mut expanded: Vec<SearchResult> = Vec::new();
        for triple in triples.iter().take(self.expansion_limit) {
            let text = triple.text();
            report.expanded_queries += 1;
            // The caller's filter applies to every chunk query, not just the
            // first one: a chunk reachable only through expansion must satisfy
            // the same metadata predicate, or a filtered retrieve would answer
            // with documents the caller excluded.
            match self
                .chunk_retriever
                .retrieve_search_results(&text, 1, self.triple_mode, filters)
                .await
            {
                Ok(hits) => {
                    if let Some(hit) = hits.into_iter().next() {
                        expanded.push(hit);
                    }
                }
                Err(error) => {
                    warn!(
                        triple = %text,
                        error = %error,
                        "graph expansion query failed"
                    );
                    report.degradations.push(GraphDegradation {
                        stage: GraphStage::ChunkQuery,
                        subject: Some(triple.subject.clone()),
                        error: error.to_string(),
                    });
                }
            }
        }
        if expanded.is_empty() {
            debug!("graph expansion produced no new chunks");
            return Ok(Self::finish(chunks, report));
        }
        let mut fused = rrf_fusion(vec![expanded, chunks], DEFAULT_RRF_K);
        fused.truncate(top_k);
        Ok(Self::finish(fused, report))
    }

    /// Marks the final hits when the expansion degraded.
    ///
    /// The flag is applied after truncation, on the exact vector handed back to
    /// the caller, and only when a degradation was recorded, so a clean run never
    /// grows a `graph_degraded` key.
    fn finish(
        results: Vec<SearchResult>,
        report: GraphExpansionReport,
    ) -> (Vec<SearchResult>, GraphExpansionReport) {
        if !report.is_degraded() {
            return (results, report);
        }
        let marked = results
            .into_iter()
            .map(|mut result| {
                result
                    .metadata
                    .insert(GRAPH_DEGRADED_FIELD.to_owned(), Value::Bool(true));
                result
            })
            .collect();
        (marked, report)
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
        Ok(self
            .retrieve_with_report(query, top_k, score_threshold, mode, filters)
            .await?
            .0)
    }

    async fn retrieve_search_results(
        &self,
        query: &str,
        top_k: usize,
        mode: RetrieveMode,
        filters: Option<&MetadataFilter>,
    ) -> Result<Vec<SearchResult>> {
        Ok(self
            .retrieve_search_results_with_report(query, top_k, mode, filters)
            .await?
            .0)
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
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

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

    /// Returns `hit(id, text, 0.5)` carrying `source` in its metadata.
    fn sourced_hit(id: &str, text: &str, source: &str) -> SearchResult {
        let mut result = hit(id, text, 0.5);
        result
            .metadata
            .insert("source".to_owned(), Value::String(source.to_owned()));
        result
    }

    /// Channel that recognises query keywords *and* honours the metadata filter.
    ///
    /// Unlike [`ProbeChannel`] it applies the caller's filter to every hit and
    /// counts the filtered calls, so a test can prove the graph expansion pushed
    /// the filter down instead of relying on the returned texts alone.
    struct FilteringChannel {
        hits: Vec<(&'static str, SearchResult)>,
        filtered_calls: AtomicUsize,
    }

    #[async_trait]
    impl Retriever for FilteringChannel {
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
            filters: Option<&MetadataFilter>,
        ) -> Result<Vec<SearchResult>> {
            if filters.is_some() {
                self.filtered_calls.fetch_add(1, AtomicOrdering::SeqCst);
            }
            let lowered = query.to_lowercase();
            Ok(self
                .hits
                .iter()
                .filter(|(keyword, _)| lowered.contains(keyword))
                .filter(|(_, hit)| filters.is_none_or(|filter| filter.matches(&hit.metadata)))
                .map(|(_, hit)| hit.clone())
                .take(top_k)
                .collect())
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
    async fn an_expansion_query_honours_the_callers_filter() {
        // `chunk-b` is reachable only through the graph ("Borrow Checker") and
        // its metadata excludes it from the caller's filter; before the fix the
        // expansion query passed `None` and handed it back anyway.
        let channel = Arc::new(FilteringChannel {
            hits: vec![
                (
                    "ownership",
                    sourced_hit("chunk-a", "Rust ownership model", "book"),
                ),
                (
                    "borrow checker",
                    sourced_hit("chunk-b", "the Borrow Checker guide", "blog"),
                ),
            ],
            filtered_calls: AtomicUsize::new(0),
        });
        let retriever = GraphRetriever::new(channel.clone(), store());
        let filter: MetadataFilter =
            serde_json::from_value(serde_json::json!({"source": "book"})).unwrap();
        let (results, report) = retriever
            .retrieve_search_results_with_report(
                "Rust ownership",
                2,
                RetrieveMode::Sparse,
                Some(&filter),
            )
            .await
            .unwrap();
        let texts: Vec<&str> = results.iter().map(|result| result.text.as_str()).collect();
        assert_eq!(texts, vec!["Rust ownership model"]);
        // The initial retrieve and both triple-text expansion queries all carried
        // the filter, so the channel itself could never return a violating chunk.
        assert_eq!(report.expanded_queries, 2);
        assert_eq!(channel.filtered_calls.load(AtomicOrdering::SeqCst), 3);
    }

    #[tokio::test]
    async fn a_clean_expansion_reports_no_degradation() {
        let retriever = GraphRetriever::new(channel(), store());
        let (results, report) = retriever
            .retrieve_search_results_with_report("Rust ownership", 2, RetrieveMode::Sparse, None)
            .await
            .unwrap();
        assert_eq!(results.len(), 2);
        assert!(results
            .iter()
            .all(|result| !result.metadata.contains_key(GRAPH_DEGRADED_FIELD)));
        assert_eq!(report.entities, vec!["Rust".to_owned()]);
        assert_eq!(report.triple_count, 2);
        assert_eq!(report.expanded_queries, 2);
        assert!(report.degradations.is_empty());
        assert!(!report.is_degraded());
    }

    #[tokio::test]
    async fn zero_top_k_returns_a_default_report() {
        let retriever = GraphRetriever::new(channel(), store());
        let (results, report) = retriever
            .retrieve_with_report("Rust ownership", 0, None, RetrieveMode::Sparse, None)
            .await
            .unwrap();
        assert!(results.is_empty());
        assert_eq!(report, GraphExpansionReport::default());
        assert!(!report.is_degraded());

        let error = retriever
            .retrieve_search_results_with_report("   ", 2, RetrieveMode::Sparse, None)
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
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
