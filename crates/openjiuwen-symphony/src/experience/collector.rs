//! The experience collector: traces in, distilled experience items out.
//!
//! Python's `ExperienceBaseBuilder` runs cluster → LLM distill → write. The Rust port
//! keeps the buffering, the validity filter, the example cap, and the similar-pattern
//! merge, and replaces the LLM distillation with the deterministic centroid query that
//! [`cluster_traces`](crate::experience::cluster_traces) already computes:
//!
//! 1. Drop traces that are not usable evidence — not successful, no skills, or a blank
//!    query (Python's `_filter_valid_traces`/`add` apply the same three conditions).
//! 2. Cluster the survivors with [`cluster_traces`], producing one
//!    [`ExperienceItem`](crate::experience::ExperienceItem) per cluster: the centroid
//!    query as the pattern, at most [`QUERY_EXAMPLES_COUNT`] example queries, the shared
//!    skill set, and the cluster's success count.
//! 3. Merge each new item into an existing pending item when it has the same skill set
//!    and a normalized-pattern token similarity of at least
//!    [`PATTERN_MERGE_THRESHOLD`] (Python's `_merge_similar_patterns`, single-linkage
//!    greedy). A merge reuses the shared
//!    [`merge_item`](crate::experience::ExperienceItem) rule, so counters add up and
//!    examples union.
//! 4. Buffer the items and flush through the sink once
//!    [`PENDING_FLUSH_THRESHOLD`] items are pending, or when the caller calls
//!    [`ExperienceCollector::flush`].
//!
//! The sink is an [`ExperienceSink`], implemented for
//! [`ExperienceBank`](crate::experience::ExperienceBank) and
//! [`ExperienceIndex`](crate::experience::ExperienceIndex), so the collector feeds either
//! catalog without duplicating the write path. Everything is deterministic: no RNG, no
//! clock, and items keep zeroed timestamps.
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::experience::cluster::{cluster_traces, jaccard, normalized_skills};
use crate::experience::ranking::query_terms;
use crate::experience::trace::TraceRecord;
use crate::experience::{merge_item, ExperienceBank, ExperienceIndex, ExperienceItem};

/// The number of buffered items that triggers an automatic flush (`20` in Python).
pub const PENDING_FLUSH_THRESHOLD: usize = 20;

/// The maximum number of example queries copied into one item (`5` in Python).
pub const QUERY_EXAMPLES_COUNT: usize = 5;

/// The normalized-pattern similarity at which two items merge (`0.9` in Python).
///
/// The similarity is the token Jaccard index over
/// [`crate::search::tokenize`](crate::search::tokenize), the same measure clustering uses;
/// Python compares embeddings, which needs the model stack this port deliberately does
/// not require.
pub const PATTERN_MERGE_THRESHOLD: f64 = 0.9;

/// A catalog the collector can write distilled items into.
///
/// Implemented for [`ExperienceBank`](crate::experience::ExperienceBank) and
/// [`ExperienceIndex`](crate::experience::ExperienceIndex); a test or a custom catalog
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
    /// The pending buffer and the counters; never held across an `.await`.
    state: Mutex<CollectorState>,
}

impl std::fmt::Debug for ExperienceCollector {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ExperienceCollector")
            .field("pending", &self.pending())
            .field("stats", &self.stats())
            .finish_non_exhaustive()
    }
}

impl ExperienceCollector {
    /// Builds a collector that flushes into `sink`.
    #[must_use]
    pub fn new(sink: Arc<dyn ExperienceSink>) -> Self {
        Self {
            sink,
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

    /// The sink this collector writes into.
    #[must_use]
    pub fn sink(&self) -> &Arc<dyn ExperienceSink> {
        &self.sink
    }

    /// Distils `records` into buffered items and flushes automatically at the threshold.
    ///
    /// Returns the number of items the batch produced (after merging); a merge into an
    /// already-buffered item still counts as one produced item. A flush triggered by the
    /// threshold may write and drain the buffer before this call returns.
    ///
    /// # Errors
    ///
    /// Returns the sink's error when an automatic flush fails. The items are put back
    /// into the buffer in that case, so no evidence is lost.
    pub async fn add(&self, records: &[TraceRecord]) -> Result<usize> {
        let candidates: Vec<TraceRecord> = records
            .iter()
            .filter(|record| record.is_experience_candidate())
            .cloned()
            .collect();
        let dropped = records.len() - candidates.len();
        let derived = items_from_traces(&candidates);
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
fn items_from_traces(records: &[TraceRecord]) -> Vec<ExperienceItem> {
    cluster_traces(records)
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
        .collect()
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

    #[test]
    fn invalid_traces_are_dropped_before_clustering() {
        let mut failure = record("t3", "a query", &["a"]);
        failure.success = false;
        let records = vec![
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
        );
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].skill_ids, vec!["summarizer"]);
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
