//! The experience collector: traces in, distilled experience items out.
//!
//! Python's `ExperienceBaseBuilder` runs cluster → LLM distill → write. The Rust port
//! keeps the buffering, the validity filter, the example cap, and the similar-pattern
//! merge, and replaces the LLM distillation with the centroid query that
//! [`cluster_traces_with_embedding`] computes:
//!
//! 1. Drop traces that are not usable evidence — not successful, no skills, or a blank
//!    query (Python's `_filter_valid_traces`/`add` apply the same three conditions).
//! 2. Cluster the survivors with [`cluster_traces_with_embedding`], producing one
//!    [`ExperienceItem`] per cluster: the centroid query as the pattern, at most
//!    [`QUERY_EXAMPLES_COUNT`] example queries, the shared skill set, and the cluster's
//!    success count.
//! 3. Merge each new item into an existing pending item when it has the same skill set
//!    and a normalized-pattern token similarity of at least
//!    [`PATTERN_MERGE_THRESHOLD`] (Python's `_merge_similar_patterns`, single-linkage
//!    greedy). A merge reuses the shared `merge_item` rule, so counters add up and
//!    examples union.
//! 4. Buffer the items and flush through the sink once
//!    [`PENDING_FLUSH_THRESHOLD`] items are pending, or when the caller calls
//!    [`ExperienceCollector::flush`].
//!
//! The clustering embedder defaults to
//! [`HashEmbedding`](openjiuwen_retrieval::HashEmbedding) — the crate's deterministic
//! offline embedder — and the k-means tunables to [`ClusterConfig::default`]; a caller with
//! a model installs it with [`ExperienceCollector::with_embedding`] and can narrow the
//! bounds with [`ExperienceCollector::with_cluster_config`]. A failing embedding call is
//! returned to the caller instead of dropping the batch: the collector buffers and counts
//! nothing in that case, so the evidence is still in the caller's hands.
//!
//! The sink is an [`ExperienceSink`], implemented for [`ExperienceBank`] and
//! [`ExperienceIndex`], so the collector feeds either catalog without duplicating the
//! write path. Everything is deterministic: no RNG, no clock, and items keep zeroed
//! timestamps.
use std::collections::BTreeSet;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use openjiuwen_retrieval::{Embedding, HashEmbedding};

use crate::error::Result;
use crate::experience::cluster::{
    cluster_traces_with_embedding, normalized_skills, ClusterConfig,
};
use crate::experience::ranking::query_terms;
use crate::experience::trace::TraceRecord;
use crate::experience::{merge_item, ExperienceBank, ExperienceIndex, ExperienceItem};

/// The number of buffered items that triggers an automatic flush (`20` in Python).
pub const PENDING_FLUSH_THRESHOLD: usize = 20;

/// The maximum number of example queries copied into one item (`5` in Python).
pub const QUERY_EXAMPLES_COUNT: usize = 5;

/// The normalized-pattern similarity at which two items merge (`0.9` in Python).
///
/// The similarity is the token Jaccard index over [`crate::search::tokenize`]. Python
/// compares embeddings here; this merge stays lexical on purpose, so buffering never
/// depends on a model call — the embedding stage is the clustering itself, which runs
/// through [`cluster_traces_with_embedding`].
pub const PATTERN_MERGE_THRESHOLD: f64 = 0.9;

/// A catalog the collector can write distilled items into.
///
/// Implemented for [`ExperienceBank`] and [`ExperienceIndex`]; a test or a custom catalog
/// can implement it too.
#[async_trait]
pub trait ExperienceSink: Send + Sync {
    /// Writes `items`, answering how many were accepted.
    ///
    /// # Errors
    ///
    /// Returns the catalog's error; the collector requeues the items so a later flush can
    /// retry them.
    async fn accept(&self, items: Vec<ExperienceItem>) -> Result<usize>;
}

#[async_trait]
impl ExperienceSink for ExperienceBank {
    async fn accept(&self, items: Vec<ExperienceItem>) -> Result<usize> {
        Ok(self.add_batch(items).await?.len())
    }
}

#[async_trait]
impl ExperienceSink for ExperienceIndex {
    async fn accept(&self, items: Vec<ExperienceItem>) -> Result<usize> {
        Ok(self.add_batch(items).await?.len())
    }
}

/// Counters describing what the collector did.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct CollectorStats {
    /// Traces accepted into the pipeline.
    pub accepted: usize,
    /// Traces dropped as unusable evidence.
    pub dropped: usize,
    /// Patterns merged into an already-buffered item.
    pub merged: usize,
    /// Items written through the sink by a flush.
    pub flushed: usize,
}

/// The mutable collector state behind one lock.
#[derive(Debug, Default)]
struct CollectorState {
    /// Items waiting for a flush, in insertion order.
    pending: Vec<ExperienceItem>,
    /// The counters.
    stats: CollectorStats,
}

/// Buffers distilled experience items and writes them to a catalog in batches.
pub struct ExperienceCollector {
    /// The catalog that receives a flush.
    sink: Arc<dyn ExperienceSink>,
    /// The embedder the clustering stage uses; `HashEmbedding` unless the caller sets one.
    embedding: Arc<dyn Embedding>,
    /// The k-means tunables the clustering stage uses.
    config: ClusterConfig,
    /// The pending buffer and the counters; never held across an `.await`.
    state: Mutex<CollectorState>,
}

impl std::fmt::Debug for ExperienceCollector {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ExperienceCollector")
            .field("embedding", &self.embedding.model_name())
            .field("config", &self.config)
            .field("pending", &self.pending())
            .field("stats", &self.stats())
            .finish_non_exhaustive()
    }
}

impl ExperienceCollector {
    /// Builds a collector that flushes into `sink`.
    ///
    /// Traces are clustered with
    /// [`HashEmbedding`](openjiuwen_retrieval::HashEmbedding) and
    /// [`ClusterConfig::default`]; use [`ExperienceCollector::with_embedding`] and
    /// [`ExperienceCollector::with_cluster_config`] to change that.
    #[must_use]
    pub fn new(sink: Arc<dyn ExperienceSink>) -> Self {
        Self {
            sink,
            embedding: Arc::new(HashEmbedding::default()),
            config: ClusterConfig::default(),
            state: Mutex::new(CollectorState::default()),
        }
    }

    /// Builds a collector that flushes into an [`ExperienceBank`].
    #[must_use]
    pub fn for_bank(bank: Arc<ExperienceBank>) -> Self {
        Self::new(bank)
    }

    /// Builds a collector that flushes into a persisted [`ExperienceIndex`].
    #[must_use]
    pub fn for_index(index: Arc<ExperienceIndex>) -> Self {
        Self::new(index)
    }

    /// Replaces the embedder the clustering stage uses.
    ///
    /// A caller with a model installs it here; without one the collector keeps the
    /// deterministic offline [`HashEmbedding`](openjiuwen_retrieval::HashEmbedding).
    #[must_use]
    pub fn with_embedding(mut self, embedding: Arc<dyn Embedding>) -> Self {
        self.embedding = embedding;
        self
    }

    /// Replaces the k-means tunables the clustering stage uses.
    #[must_use]
    pub fn with_cluster_config(mut self, config: ClusterConfig) -> Self {
        self.config = config;
        self
    }

    /// The sink this collector writes into.
    #[must_use]
    pub fn sink(&self) -> &Arc<dyn ExperienceSink> {
        &self.sink
    }

    /// The embedder this collector clusters with.
    #[must_use]
    pub fn embedding(&self) -> &Arc<dyn Embedding> {
        &self.embedding
    }

    /// The k-means tunables this collector clusters with.
    #[must_use]
    pub fn cluster_config(&self) -> &ClusterConfig {
        &self.config
    }

    /// Distils `records` into buffered items and flushes automatically at the threshold.
    ///
    /// Returns the number of items the batch produced (after merging); a merge into an
    /// already-buffered item still counts as one produced item. A flush triggered by the
    /// threshold may write and drain the buffer before this call returns.
    ///
    /// # Errors
    ///
    /// Returns the sink's error when an automatic flush fails; the items are put back into
    /// the buffer in that case, so no evidence is lost. Returns the clustering error when
    /// the embedder fails or [`ClusterConfig::validate`] rejects the configuration — the
    /// collector is left untouched then, so the caller can retry the same batch.
    pub async fn add(&self, records: &[TraceRecord]) -> Result<usize> {
        let candidates: Vec<TraceRecord> = records
            .iter()
            .filter(|record| record.is_experience_candidate())
            .cloned()
            .collect();
        let dropped = records.len() - candidates.len();
        let derived =
            items_from_traces(&candidates, self.embedding.as_ref(), &self.config).await?;
        let produced = derived.len();

        let mut snapshot = None;
        {
            let mut state = self.lock();
            state.stats.accepted += candidates.len();
            state.stats.dropped += dropped;
            for item in derived {
                match merge_target(&state.pending, &item) {
                    Some(index) => {
                        merge_item(&mut state.pending[index], item);
                        state.stats.merged += 1;
                    }
                    None => state.pending.push(item),
                }
            }
            if state.pending.len() >= PENDING_FLUSH_THRESHOLD {
                snapshot = Some(std::mem::take(&mut state.pending));
            }
        }

        if let Some(items) = snapshot {
            self.write(items).await?;
        }
        Ok(produced)
    }

    /// Writes every buffered item through the sink.
    ///
    /// Returns the number of items written; an empty buffer is `Ok(0)` and touches
    /// nothing, so a second `flush` is a no-op.
    ///
    /// # Errors
    ///
    /// Returns the sink's error; the items are requeued for the next flush.
    pub async fn flush(&self) -> Result<usize> {
        let items = {
            let mut state = self.lock();
            if state.pending.is_empty() {
                return Ok(0);
            }
            std::mem::take(&mut state.pending)
        };
        self.write(items).await
    }

    /// The number of items waiting for a flush.
    #[must_use]
    pub fn pending(&self) -> usize {
        self.lock().pending.len()
    }

    /// A copy of the counters.
    #[must_use]
    pub fn stats(&self) -> CollectorStats {
        self.lock().stats
    }

    /// Writes `items`, requeueing them ahead of newer pending items on failure.
    async fn write(&self, items: Vec<ExperienceItem>) -> Result<usize> {
        if items.is_empty() {
            return Ok(0);
        }
        let backup = items.clone();
        match self.sink.accept(items).await {
            Ok(written) => {
                self.lock().stats.flushed += written;
                Ok(written)
            }
            Err(error) => {
                let mut state = self.lock();
                state.pending.splice(0..0, backup);
                Err(error)
            }
        }
    }

    /// Locks the collector state, recovering from a poisoned lock.
    fn lock(&self) -> MutexGuard<'_, CollectorState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// Turns valid traces into one item per cluster.
async fn items_from_traces(
    records: &[TraceRecord],
    embedding: &dyn Embedding,
    config: &ClusterConfig,
) -> Result<Vec<ExperienceItem>> {
    Ok(
        cluster_traces_with_embedding(records, embedding, config)
            .await?
            .into_iter()
            .map(|cluster| {
                ExperienceItem::new(cluster.pattern)
                    .with_query_examples(
                        cluster
                            .query_examples
                            .iter()
                            .take(QUERY_EXAMPLES_COUNT)
                            .cloned(),
                    )
                    .with_skill_ids(cluster.skill_ids)
                    .with_success_count(cluster.success_count.max(1))
            })
            .collect(),
    )
}

/// Finds the first buffered item `incoming` may merge into.
///
/// The skill sets must be equal (order-free) and the normalized patterns must reach
/// [`PATTERN_MERGE_THRESHOLD`]; scanning in insertion order keeps the choice
/// deterministic.
fn merge_target(pending: &[ExperienceItem], incoming: &ExperienceItem) -> Option<usize> {
    let incoming_skills = normalized_skills(&incoming.skill_ids);
    let incoming_tokens = query_terms(&incoming.query_pattern);
    pending.iter().position(|existing| {
        normalized_skills(&existing.skill_ids) == incoming_skills
            && jaccard(&query_terms(&existing.query_pattern), &incoming_tokens)
                >= PATTERN_MERGE_THRESHOLD
    })
}

/// The Jaccard similarity of two token sets, in `[0, 1]`.
///
/// Two empty sets are `1.0` (the patterns carry no distinguishing terms), and an empty set
/// against a non-empty one is `0.0`.
#[allow(clippy::cast_precision_loss)]
fn jaccard(left: &BTreeSet<String>, right: &BTreeSet<String>) -> f64 {
    if left.is_empty() && right.is_empty() {
        return 1.0;
    }
    let union = left.union(right).count();
    if union == 0 {
        return 1.0;
    }
    left.intersection(right).count() as f64 / union as f64
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn record(trace_id: &str, query: &str, skills: &[&str]) -> TraceRecord {
        TraceRecord {
            trace_id: trace_id.to_string(),
            session_id: "s".to_string(),
            request_id: trace_id.to_string(),
            query: query.to_string(),
            skills: skills.iter().map(|skill| (*skill).to_string()).collect(),
            success: true,
            ..TraceRecord::default()
        }
    }

    #[tokio::test]
    async fn invalid_traces_are_dropped_before_clustering() {
        let mut failure = record("t3", "a query", &["a"]);
        failure.success = false;
        let records = [
            record("t1", "summarize the report", &["summarizer"]),
            record("t2", "no skills here", &[]),
            record("t4", "   ", &["summarizer"]),
            failure,
        ];
        let items = items_from_traces(
            &records
                .iter()
                .filter(|record| record.is_experience_candidate())
                .cloned()
                .collect::<Vec<_>>(),
            &HashEmbedding::default(),
            &ClusterConfig::default(),
        )
        .await
        .expect("distil");
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].skill_ids, vec!["summarizer"]);
    }

    #[test]
    fn jaccard_handles_empty_sets() {
        let empty = BTreeSet::new();
        let tokens: BTreeSet<String> = ["a".to_string()].into_iter().collect();
        assert_eq!(jaccard(&empty, &empty), 1.0);
        assert_eq!(jaccard(&empty, &tokens), 0.0);
        assert_eq!(jaccard(&tokens, &tokens), 1.0);
    }

    #[test]
    fn merge_prefers_the_first_pending_item_with_the_same_skills() {
        let first = ExperienceItem::new("alpha bravo charlie delta echo foxtrot golf hotel india")
            .with_skill_ids(["a"]);
        let candidate =
            ExperienceItem::new("alpha bravo charlie delta echo foxtrot golf hotel india juliet")
                .with_skill_ids(["a"]);
        let other_skills = ExperienceItem::new("alpha bravo charlie").with_skill_ids(["b"]);
        let pending = vec![first, other_skills];
        assert_eq!(merge_target(&pending, &candidate), Some(0));
    }
}
