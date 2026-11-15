//! Multi-knowledge-base retrieval.
//!
//! Port of Python's free functions `retrieve_multi_kb` and
//! `retrieve_multi_kb_with_source` from `simple_knowledge_base.py`. Both fan out
//! to every knowledge base concurrently (`asyncio.gather` becomes
//! [`futures::future::join_all`]), tolerate a failing knowledge base by logging
//! and dropping its results, deduplicate by exact result text, and keep the
//! highest score seen for each text.
//!
//! The compatible functions never report which knowledge base failed. Callers
//! that must see per-KB failures can use the `_reporting` variants, which return
//! the same merged results plus a [`MultiKbFailure`] entry per failed knowledge
//! base, or the `_strict` variants, which turn "every knowledge base failed"
//! into an [`Error`].
use std::collections::HashMap;
use std::sync::Arc;

use futures::future::join_all;
use serde::{Deserialize, Serialize};

use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::logging::warn;

use crate::common::{Metadata, MultiKBRetrievalResult, RetrievalResult, DEFAULT_TOP_K};

use super::SimpleKnowledgeBase;

/// Metadata key holding the unmerged score.
const RAW_SCORE_FIELD: &str = "raw_score";
/// Metadata key holding the unmerged score scaled into `[0, 1]`.
const RAW_SCORE_SCALED_FIELD: &str = "raw_score_scaled";

/// One knowledge base that failed during a multi-KB retrieval.
///
/// `kb_id` is the knowledge base identifier (`SimpleKnowledgeBase::id()`) and
/// `error` is the failure's `Display` text, so a report is fully serializable
/// and independent of the error type.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MultiKbFailure {
    /// Identifier of the knowledge base whose retrieval failed.
    pub kb_id: String,
    /// The failure rendered as text (`error.to_string()`).
    pub error: String,
}

/// Merged results of a multi-KB retrieval plus every per-KB failure.
///
/// The failures are in the order the knowledge bases were supplied, so a caller
/// can correlate them with its own slice. A report whose `failures` is empty is
/// a fully successful retrieval; one whose `results` is empty with non-empty
/// `failures` means every knowledge base failed.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct MultiKbReport {
    /// Deduplicated, descending merged results (the same ones the compatible
    /// [`retrieve_multi_kb`] returns, with the same limit).
    pub results: Vec<RetrievalResult>,
    /// One entry per knowledge base that failed, in input order.
    pub failures: Vec<MultiKbFailure>,
}

/// Merged with-source results of a multi-KB retrieval plus every per-KB failure.
///
/// Identical to [`MultiKbReport`] except that the results carry their source
/// `kb_ids` ([`MultiKBRetrievalResult`]).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct MultiKbSourceReport {
    /// Deduplicated, descending merged results with their source knowledge base
    /// ids (the same ones the compatible [`retrieve_multi_kb_with_source`]
    /// returns, with the same limit).
    pub results: Vec<MultiKBRetrievalResult>,
    /// One entry per knowledge base that failed, in input order.
    pub failures: Vec<MultiKbFailure>,
}

/// Runs `query` against every knowledge base and returns the deduplicated,
/// descending results limited to `top_k` (default [`DEFAULT_TOP_K`]).
///
/// Duplicates are detected by exact `text`; the surviving result keeps the first
/// occurrence's metadata and the maximum score of all occurrences. A knowledge
/// base that fails contributes nothing and is logged at `warn`; use
/// [`retrieve_multi_kb_reporting`] or [`retrieve_multi_kb_strict`] to observe
/// those failures instead of only logging them.
pub async fn retrieve_multi_kb(
    kbs: &[Arc<SimpleKnowledgeBase>],
    query: &str,
    top_k: Option<usize>,
) -> Vec<RetrievalResult> {
    let (per_kb, _failures) = retrieve_all(kbs, query, "retrieve_multi_kb").await;
    merge_results(per_kb, limit(top_k))
}

/// Like [`retrieve_multi_kb`], but records which knowledge bases produced each
/// result and merges the `raw_score`/`raw_score_scaled` metadata fields.
///
/// `kb_ids` are sorted ascending and deduplicated; `metadata` comes from the
/// first occurrence of a text. A knowledge base that fails contributes nothing
/// and is logged at `warn`; use [`retrieve_multi_kb_with_source_reporting`] or
/// [`retrieve_multi_kb_with_source_strict`] to observe those failures.
pub async fn retrieve_multi_kb_with_source(
    kbs: &[Arc<SimpleKnowledgeBase>],
    query: &str,
    top_k: Option<usize>,
) -> Vec<MultiKBRetrievalResult> {
    let (per_kb, _failures) = retrieve_all(kbs, query, "retrieve_multi_kb_with_source").await;
    merge_results_with_source(per_kb, limit(top_k))
}

/// Like [`retrieve_multi_kb`], but also reports every knowledge base that
/// failed.
///
/// `results` uses exactly the merge, sort, and limit rules of
/// [`retrieve_multi_kb`]; `failures` lists one [`MultiKbFailure`] per failed
/// knowledge base in input order and is empty when every knowledge base
/// answered. An empty `kbs` slice yields an empty report.
pub async fn retrieve_multi_kb_reporting(
    kbs: &[Arc<SimpleKnowledgeBase>],
    query: &str,
    top_k: Option<usize>,
) -> MultiKbReport {
    let (per_kb, failures) = retrieve_all(kbs, query, "retrieve_multi_kb_reporting").await;
    MultiKbReport {
        results: merge_results(per_kb, limit(top_k)),
        failures,
    }
}

/// Like [`retrieve_multi_kb_with_source`], but also reports every knowledge base
/// that failed.
///
/// `results` uses exactly the merge, source-attribution, sort, and limit rules
/// of [`retrieve_multi_kb_with_source`]; `failures` lists one [`MultiKbFailure`]
/// per failed knowledge base in input order and is empty when every knowledge
/// base answered. An empty `kbs` slice yields an empty report.
pub async fn retrieve_multi_kb_with_source_reporting(
    kbs: &[Arc<SimpleKnowledgeBase>],
    query: &str,
    top_k: Option<usize>,
) -> MultiKbSourceReport {
    let (per_kb, failures) =
        retrieve_all(kbs, query, "retrieve_multi_kb_with_source_reporting").await;
    MultiKbSourceReport {
        results: merge_results_with_source(per_kb, limit(top_k)),
        failures,
    }
}

/// Like [`retrieve_multi_kb`], but fails when every knowledge base failed.
///
/// A partial failure still returns `Ok` with the surviving merged results, so
/// this is the strict variant only for the total-failure case. Rules:
///
/// - an empty `kbs` slice returns `Ok(vec![])` (no failure can occur);
/// - if **every** knowledge base failed, returns
///   [`ErrorKind::Retrieval`](openjiuwen_common::error::ErrorKind::Retrieval)
///   with the message `every knowledge base failed: {kb_id}: {error}; …`, one
///   entry per knowledge base in input order;
/// - otherwise returns `Ok` with the merged results of the knowledge bases that
///   answered (identical to [`retrieve_multi_kb`]).
///
/// Every failure is still logged at `warn`, exactly like the compatible variant.
pub async fn retrieve_multi_kb_strict(
    kbs: &[Arc<SimpleKnowledgeBase>],
    query: &str,
    top_k: Option<usize>,
) -> Result<Vec<RetrievalResult>> {
    let report = retrieve_multi_kb_reporting(kbs, query, top_k).await;
    ensure_not_all_failed(&report.failures, kbs.len())?;
    Ok(report.results)
}

/// Like [`retrieve_multi_kb_with_source`], but fails when every knowledge base
/// failed.
///
/// Follows the same rules as [`retrieve_multi_kb_strict`]: an empty `kbs` slice
/// returns `Ok(vec![])`, a total failure returns
/// [`ErrorKind::Retrieval`](openjiuwen_common::error::ErrorKind::Retrieval) with
/// the message `every knowledge base failed: {kb_id}: {error}; …`, and any
/// partial success returns `Ok` with the merged with-source results.
pub async fn retrieve_multi_kb_with_source_strict(
    kbs: &[Arc<SimpleKnowledgeBase>],
    query: &str,
    top_k: Option<usize>,
) -> Result<Vec<MultiKBRetrievalResult>> {
    let report = retrieve_multi_kb_with_source_reporting(kbs, query, top_k).await;
    ensure_not_all_failed(&report.failures, kbs.len())?;
    Ok(report.results)
}

/// Rejects a retrieval in which every supplied knowledge base failed.
///
/// `failure_count == kb_count` is only a total failure when at least one
/// knowledge base was supplied, so an empty slice stays `Ok`.
fn ensure_not_all_failed(failures: &[MultiKbFailure], kb_count: usize) -> Result<()> {
    if kb_count == 0 || failures.len() != kb_count {
        return Ok(());
    }
    let detail = failures
        .iter()
        .map(|failure| format!("{}: {}", failure.kb_id, failure.error))
        .collect::<Vec<_>>()
        .join("; ");
    Err(Error::retrieval(format!(
        "every knowledge base failed: {detail}"
    )))
}

/// Retrieves from every knowledge base concurrently, keeping the kb id with each
/// result set so the with-source merge can attribute hits.
///
/// Returns the per-KB result sets (in `kbs` order) together with the failures of
/// the knowledge bases that errored (also in `kbs` order), so both the
/// compatible and the reporting/strict callers share one fan-out. Each failure
/// is logged at `warn` for every caller.
async fn retrieve_all(
    kbs: &[Arc<SimpleKnowledgeBase>],
    query: &str,
    caller: &'static str,
) -> (Vec<(String, Vec<RetrievalResult>)>, Vec<MultiKbFailure>) {
    let futures = kbs.iter().map(|kb| {
        let kb_id = kb.id().to_owned();
        async move {
            match kb.retrieve(query).await {
                Ok(results) => Ok((kb_id, results)),
                Err(error) => {
                    warn!(
                        caller,
                        kb_id = kb_id.as_str(),
                        error = %error,
                        "knowledge base retrieval failed; skipping its results"
                    );
                    Err(MultiKbFailure {
                        kb_id,
                        error: error.to_string(),
                    })
                }
            }
        }
    });
    let mut per_kb = Vec::with_capacity(kbs.len());
    let mut failures = Vec::new();
    for outcome in join_all(futures).await {
        match outcome {
            Ok(entry) => per_kb.push(entry),
            Err(failure) => failures.push(failure),
        }
    }
    (per_kb, failures)
}

/// Resolves the result limit: the explicit `top_k`, or [`DEFAULT_TOP_K`].
fn limit(top_k: Option<usize>) -> usize {
    top_k.unwrap_or(DEFAULT_TOP_K)
}

/// Sorts by descending score (stable) and truncates to `limit`.
fn sort_and_truncate<T: SortableResult>(results: &mut Vec<T>, limit: usize) {
    results.sort_by(|left, right| right.result_score().total_cmp(&left.result_score()));
    results.truncate(limit);
}

/// Minimal shared view of the two result types this module ranks.
trait SortableResult {
    /// Score used for ordering.
    fn result_score(&self) -> f32;
}

impl SortableResult for RetrievalResult {
    fn result_score(&self) -> f32 {
        self.score
    }
}

impl SortableResult for MultiKBRetrievalResult {
    fn result_score(&self) -> f32 {
        self.score
    }
}

/// Deduplicates by text, keeping the first occurrence and the maximum score.
fn merge_results(
    per_kb: Vec<(String, Vec<RetrievalResult>)>,
    limit: usize,
) -> Vec<RetrievalResult> {
    let mut merged: Vec<RetrievalResult> = Vec::new();
    let mut positions: HashMap<String, usize> = HashMap::new();
    for (_, results) in per_kb {
        for result in results {
            if let Some(index) = positions.get(&result.text) {
                let existing = &mut merged[*index];
                if result.score > existing.score {
                    existing.score = result.score;
                }
            } else {
                positions.insert(result.text.clone(), merged.len());
                merged.push(result);
            }
        }
    }
    sort_and_truncate(&mut merged, limit);
    merged
}

/// Deduplicates by text while merging ids and the score metadata fields.
fn merge_results_with_source(
    per_kb: Vec<(String, Vec<RetrievalResult>)>,
    limit: usize,
) -> Vec<MultiKBRetrievalResult> {
    let mut merged: Vec<MultiKBRetrievalResult> = Vec::new();
    let mut positions: HashMap<String, usize> = HashMap::new();
    for (kb_id, results) in per_kb {
        for result in results {
            let raw = metadata_score(&result.metadata, RAW_SCORE_FIELD, result.score);
            let raw_scaled = metadata_score(&result.metadata, RAW_SCORE_SCALED_FIELD, result.score);
            if let Some(index) = positions.get(&result.text) {
                let existing = &mut merged[*index];
                existing.score = existing.score.max(result.score);
                existing.raw_score = existing.raw_score.max(raw);
                existing.raw_score_scaled = existing.raw_score_scaled.max(raw_scaled);
                push_unique(&mut existing.kb_ids, kb_id.clone());
            } else {
                positions.insert(result.text.clone(), merged.len());
                let mut kb_ids = Vec::new();
                push_unique(&mut kb_ids, kb_id.clone());
                merged.push(MultiKBRetrievalResult {
                    text: result.text,
                    score: result.score,
                    raw_score: raw,
                    raw_score_scaled: raw_scaled,
                    kb_ids,
                    metadata: result.metadata,
                });
            }
        }
    }
    // Python sorts the ids when building the final result.
    for result in &mut merged {
        result.kb_ids.sort();
    }
    sort_and_truncate(&mut merged, limit);
    merged
}

/// Reads a metadata score field, falling back to `default`.
fn metadata_score(metadata: &Metadata, field: &str, default: f32) -> f32 {
    metadata
        .get(field)
        .and_then(serde_json::Value::as_f64)
        .map_or(default, |value| value as f32)
}

/// Appends `kb_id` when it is not already recorded.
fn push_unique(kb_ids: &mut Vec<String>, kb_id: String) {
    if !kb_ids.contains(&kb_id) {
        kb_ids.push(kb_id);
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use serde_json::json;

    use super::*;

    fn result(text: &str, score: f32) -> RetrievalResult {
        RetrievalResult {
            text: text.to_owned(),
            score,
            ..RetrievalResult::default()
        }
    }

    #[test]
    fn merge_results_keeps_the_first_occurrence_and_the_max_score() {
        let per_kb = vec![
            ("kb-1".to_owned(), vec![result("a", 0.4), result("b", 0.8)]),
            ("kb-2".to_owned(), vec![result("a", 0.9)]),
        ];
        let merged = merge_results(per_kb, DEFAULT_TOP_K);
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].text, "a");
        assert!((merged[0].score - 0.9).abs() < 1e-6);
        assert_eq!(merged[1].text, "b");
    }

    #[test]
    fn merge_results_honours_the_limit() {
        let per_kb = vec![(
            "kb-1".to_owned(),
            vec![result("a", 0.1), result("b", 0.2), result("c", 0.3)],
        )];
        let merged = merge_results(per_kb, 2);
        let texts: Vec<&str> = merged.iter().map(|item| item.text.as_str()).collect();
        assert_eq!(texts, vec!["c", "b"]);
    }

    #[test]
    fn merge_with_source_collects_sorted_unique_kb_ids() {
        // kb-1 supplies an explicit `raw_score`; kb-2 supplies `raw_score_scaled`
        // and a higher score, so each merged field keeps the larger value.
        let mut first = result("shared", 0.1);
        first
            .metadata
            .insert(RAW_SCORE_FIELD.to_owned(), json!(1.5));
        let mut second = result("shared", 0.7);
        second
            .metadata
            .insert(RAW_SCORE_SCALED_FIELD.to_owned(), json!(0.25));
        let per_kb = vec![
            ("kb-2".to_owned(), vec![second]),
            ("kb-1".to_owned(), vec![first]),
        ];
        let merged = merge_results_with_source(per_kb, DEFAULT_TOP_K);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].kb_ids, vec!["kb-1".to_owned(), "kb-2".to_owned()]);
        assert!((merged[0].score - 0.7).abs() < 1e-6);
        assert!((merged[0].raw_score - 1.5).abs() < 1e-6);
        assert!((merged[0].raw_score_scaled - 0.25).abs() < 1e-6);
        // The first-seen metadata (kb-2's) survives.
        assert!(merged[0].metadata.contains_key(RAW_SCORE_SCALED_FIELD));
    }

    #[test]
    fn merge_with_source_defaults_raw_scores_to_the_score() {
        let per_kb = vec![("kb-1".to_owned(), vec![result("solo", 0.6)])];
        let merged = merge_results_with_source(per_kb, DEFAULT_TOP_K);
        assert!((merged[0].raw_score - 0.6).abs() < 1e-6);
        assert!((merged[0].raw_score_scaled - 0.6).abs() < 1e-6);
    }

    #[test]
    fn empty_inputs_merge_to_nothing() {
        assert_eq!(merge_results(Vec::new(), DEFAULT_TOP_K), Vec::new());
        assert_eq!(
            merge_results_with_source(Vec::new(), DEFAULT_TOP_K),
            Vec::new()
        );
        assert_eq!(limit(None), DEFAULT_TOP_K);
        assert_eq!(limit(Some(2)), 2);
    }
}
