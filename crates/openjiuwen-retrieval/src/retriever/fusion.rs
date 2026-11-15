//! Rank fusion helpers: reciprocal rank fusion, weighted fusion, and text
//! deduplication.
//!
//! Port of the Python `openjiuwen/core/retrieval/utils/fusion.py` plus the two
//! formulas Milvus executes server-side for the
//! [`WeightedRankConfig`](crate::common::WeightedRankConfig) and
//! [`RrfRankConfig`] ranking configs.
//! Results are keyed by their **text** (Python uses `result.text` as the
//! identity because chunk ids differ between the vector and lexical channels),
//! and the first-seen result object supplies the metadata of a fused hit.

use indexmap::IndexMap;

use openjiuwen_common::error::{Error, Result};

use crate::common::ranking::RrfRankConfig;
use crate::common::SearchResult;

/// Default RRF smoothing constant (`k = 60` in Python).
pub const DEFAULT_RRF_K: usize = 60;

/// Fuses several result lists with Reciprocal Rank Fusion.
///
/// Every result contributes `1 / (k + rank)` using its 1-based rank in its own
/// list; contributions are summed per distinct text. The fused score replaces the
/// original score, the first-seen result object is kept, and the output is sorted
/// by descending fused score with ties keeping first-seen order. `k = 0` is valid
/// and degenerates to `1 / rank`.
pub fn rrf_fusion(results_list: Vec<Vec<SearchResult>>, k: usize) -> Vec<SearchResult> {
    let weights = vec![1.0_f32; results_list.len()];
    rrf_fusion_weighted_impl(results_list, k, &weights)
}

/// [`rrf_fusion`] with the Python default constant [`DEFAULT_RRF_K`].
pub fn rrf_fusion_default(results_list: Vec<Vec<SearchResult>>) -> Vec<SearchResult> {
    rrf_fusion(results_list, DEFAULT_RRF_K)
}

/// Reciprocal rank fusion with a weight multiplying every channel's contribution.
///
/// Channel `i` contributes `weights[i] / (k + rank)` for each result at its
/// 1-based `rank`; channels with a non-positive weight are dropped. Results are
/// keyed by text, the first-seen object is kept, and the output is sorted by
/// descending fused score with stable ties.
///
/// # Errors
///
/// Returns [`Error::Validation`] when `weights.len() != results_list.len()`.
pub fn rrf_fusion_weighted(
    results_list: Vec<Vec<SearchResult>>,
    k: usize,
    weights: &[f32],
) -> Result<Vec<SearchResult>> {
    if weights.len() != results_list.len() {
        return Err(Error::validation(format!(
            "rrf fusion needs one weight per channel: got {} weights for {} lists",
            weights.len(),
            results_list.len()
        )));
    }
    Ok(rrf_fusion_weighted_impl(results_list, k, weights))
}

/// Reciprocal rank fusion driven by an [`RrfRankConfig`].
///
/// The config's `k` sets the smoothing constant and [`RrfRankConfig::is_active`]
/// enables or drops channels; inactive channels contribute nothing, which is how
/// Milvus turns the Python channel flags into a fusion mask.
pub fn rrf_fusion_channels(
    results_list: Vec<Vec<SearchResult>>,
    config: &RrfRankConfig,
) -> Vec<SearchResult> {
    let active = config.is_active();
    let weights: Vec<f32> = (0..results_list.len())
        .map(|index| {
            if active.get(index).copied().unwrap_or(false) {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    rrf_fusion_weighted_impl(results_list, config.k, &weights)
}

pub(crate) fn rrf_fusion_weighted_impl(
    results_list: Vec<Vec<SearchResult>>,
    k: usize,
    weights: &[f32],
) -> Vec<SearchResult> {
    let mut fused: IndexMap<String, (SearchResult, f32)> = IndexMap::new();
    for (results, weight) in results_list.into_iter().zip(weights.iter().copied()) {
        if weight <= 0.0 {
            continue;
        }
        for (position, result) in results.into_iter().enumerate() {
            let rank = position + 1;
            let contribution = weight / (k + rank) as f32;
            if let Some((_, score)) = fused.get_mut(&result.text) {
                *score += contribution;
            } else {
                let key = result.text.clone();
                fused.insert(key, (result, contribution));
            }
        }
    }
    let mut fused: Vec<SearchResult> = fused
        .into_values()
        .map(|(mut result, score)| {
            result.score = score;
            result
        })
        .collect();
    // Stable sort: equal fused scores keep first-seen order.
    fused.sort_by(|left, right| right.score.total_cmp(&left.score));
    fused
}

/// Fuses several result lists with a weighted sum of normalised scores.
///
/// Each channel's scores are min-max normalised into `[0, 1]` (a channel whose
/// scores are all equal normalises every hit to `1.0`) and multiplied by that
/// channel's weight; an empty channel or a non-positive weight contributes
/// nothing. Contributions are summed per distinct text, the first-seen result
/// object is kept, and the output is sorted by descending fused score with stable
/// ties.
///
/// This is the client-side implementation of Milvus' weighted ranker. Milvus
/// normalises the raw channel distances server-side according to each channel's
/// metric (where a smaller distance can be better); this crate's channel scores
/// are already similarities, so the normalisation is applied directly to the
/// scores and the fused list is sorted descending.
///
/// # Errors
///
/// Returns [`Error::Validation`] when `weights.len() != lists.len()`.
pub fn weighted_fusion(
    lists: Vec<Vec<SearchResult>>,
    weights: &[f32],
) -> Result<Vec<SearchResult>> {
    if weights.len() != lists.len() {
        return Err(Error::validation(format!(
            "weighted fusion needs one weight per channel: got {} weights for {} lists",
            weights.len(),
            lists.len()
        )));
    }
    Ok(weighted_fusion_impl(lists, weights))
}

pub(crate) fn weighted_fusion_impl(
    lists: Vec<Vec<SearchResult>>,
    weights: &[f32],
) -> Vec<SearchResult> {
    let mut fused: IndexMap<String, (SearchResult, f32)> = IndexMap::new();
    for (list, weight) in lists.into_iter().zip(weights.iter().copied()) {
        if weight <= 0.0 || list.is_empty() {
            continue;
        }
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        for result in &list {
            if result.score < min {
                min = result.score;
            }
            if result.score > max {
                max = result.score;
            }
        }
        let span = max - min;
        for result in list {
            let normalised = if span > 0.0 {
                (result.score - min) / span
            } else {
                1.0
            };
            let contribution = weight * normalised;
            if let Some((_, score)) = fused.get_mut(&result.text) {
                *score += contribution;
            } else {
                let key = result.text.clone();
                fused.insert(key, (result, contribution));
            }
        }
    }
    let mut fused: Vec<SearchResult> = fused
        .into_values()
        .map(|(mut result, score)| {
            result.score = score;
            result
        })
        .collect();
    fused.sort_by(|left, right| right.score.total_cmp(&left.score));
    fused
}

/// Removes duplicate texts, keeping the first-seen result and its highest score.
///
/// Input order is preserved: the surviving result stays where its first
/// occurrence appeared, with `score` raised to the maximum seen for that text.
pub fn deduplicate(results: Vec<SearchResult>) -> Vec<SearchResult> {
    let mut unique: IndexMap<String, SearchResult> = IndexMap::new();
    for result in results {
        match unique.get_mut(&result.text) {
            Some(existing) => {
                if result.score > existing.score {
                    existing.score = result.score;
                }
            }
            None => {
                unique.insert(result.text.clone(), result);
            }
        }
    }
    unique.into_values().collect()
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
    use crate::common::Metadata;
    use serde_json::json;

    const TOLERANCE: f32 = 1e-7;

    fn hit(id: &str, text: &str, score: f32) -> SearchResult {
        SearchResult {
            id: id.to_owned(),
            text: text.to_owned(),
            score,
            metadata: Metadata::new(),
        }
    }

    fn texts(results: &[SearchResult]) -> Vec<&str> {
        results.iter().map(|result| result.text.as_str()).collect()
    }

    #[test]
    fn rrf_uses_one_based_ranks_and_first_seen_objects() {
        let first = vec![
            SearchResult {
                metadata: Metadata::from_iter([("source".to_owned(), json!("vector"))]),
                ..hit("v1", "alpha", 0.9)
            },
            hit("v2", "beta", 0.8),
        ];
        let second = vec![hit("s1", "beta", 12.0), hit("s2", "gamma", 4.0)];
        let fused = rrf_fusion(vec![first, second], 60);

        assert_eq!(texts(&fused), vec!["beta", "alpha", "gamma"]);
        assert!((fused[0].score - (1.0 / 62.0 + 1.0 / 61.0)).abs() < TOLERANCE);
        assert!((fused[1].score - 1.0 / 61.0).abs() < TOLERANCE);
        assert!((fused[2].score - 1.0 / 62.0).abs() < TOLERANCE);
        // First-seen metadata and ids survive; only the score is rewritten.
        assert_eq!(fused[1].id, "v1");
        assert_eq!(fused[1].metadata.get("source"), Some(&json!("vector")));
        assert_eq!(fused[0].id, "v2");
    }

    #[test]
    fn rrf_ties_keep_first_seen_order_and_handle_k_zero() {
        let fused = rrf_fusion(
            vec![vec![hit("a", "alpha", 1.0)], vec![hit("b", "beta", 1.0)]],
            DEFAULT_RRF_K,
        );
        assert_eq!(texts(&fused), vec!["alpha", "beta"]);
        assert!((fused[0].score - fused[1].score).abs() < TOLERANCE);

        let degenerate = rrf_fusion(
            vec![vec![hit("a", "alpha", 1.0)], vec![hit("b", "beta", 1.0)]],
            0,
        );
        assert!((degenerate[0].score - 1.0).abs() < TOLERANCE);
        assert_eq!(DEFAULT_RRF_K, 60);
        assert!(rrf_fusion_default(vec![Vec::new()]).is_empty());
        assert!(rrf_fusion(Vec::new(), 60).is_empty());
    }

    #[test]
    fn rrf_duplicate_texts_inside_one_list_accumulate() {
        let fused = rrf_fusion(
            vec![vec![hit("a", "alpha", 1.0), hit("b", "alpha", 1.0)]],
            60,
        );
        assert_eq!(fused.len(), 1);
        assert!((fused[0].score - (1.0 / 61.0 + 1.0 / 62.0)).abs() < TOLERANCE);
    }

    #[test]
    fn deduplicate_keeps_first_seen_order_and_maximum_score() {
        let results = vec![
            hit("a", "alpha", 0.2),
            hit("b", "beta", 0.5),
            hit("c", "alpha", 0.7),
            hit("d", "beta", 0.1),
        ];
        let unique = deduplicate(results);
        assert_eq!(texts(&unique), vec!["alpha", "beta"]);
        assert_eq!(unique[0].id, "a");
        assert_eq!(unique[0].score, 0.7);
        assert_eq!(unique[1].id, "b");
        assert_eq!(unique[1].score, 0.5);
        assert!(deduplicate(Vec::new()).is_empty());
    }

    #[test]
    fn weighted_fusion_normalises_each_channel_and_sums() {
        let dense = vec![
            hit("a", "alpha", 0.9),
            hit("b", "beta", 0.5),
            hit("c", "gamma", 0.1),
        ];
        let sparse = vec![hit("d", "beta", 10.0), hit("e", "delta", 2.0)];
        let fused = weighted_fusion(vec![dense, sparse], &[0.6, 0.25]).unwrap();
        // alpha: 0.6 * 1.0; beta: 0.6 * 0.5 + 0.25 * 1.0; gamma: 0.6 * 0.0; delta: 0.25 * 0.0.
        assert_eq!(texts(&fused), vec!["alpha", "beta", "gamma", "delta"]);
        assert!((fused[0].score - 0.6).abs() < TOLERANCE);
        assert!((fused[1].score - 0.55).abs() < TOLERANCE);
        assert!(fused[2].score.abs() < TOLERANCE);
        assert!(fused[3].score.abs() < TOLERANCE);
    }

    #[test]
    fn weighted_fusion_constant_channel_normalises_to_one_and_drops_zero_weight() {
        let single = vec![hit("a", "alpha", 4.0), hit("b", "beta", 4.0)];
        let fused = weighted_fusion(vec![single], &[1.0]).unwrap();
        assert!((fused[0].score - 1.0).abs() < TOLERANCE);
        assert!((fused[1].score - 1.0).abs() < TOLERANCE);

        let dropped = weighted_fusion(vec![vec![hit("a", "alpha", 1.0)]], &[0.0]).unwrap();
        assert!(dropped.is_empty());
    }

    #[test]
    fn weighted_fusion_dedupes_on_text_keeping_first_seen_metadata() {
        let first = vec![SearchResult {
            metadata: Metadata::from_iter([("source".to_owned(), json!("vector"))]),
            ..hit("v1", "alpha", 0.9)
        }];
        let second = vec![SearchResult {
            metadata: Metadata::from_iter([("source".to_owned(), json!("sparse"))]),
            ..hit("s1", "alpha", 10.0)
        }];
        let fused = weighted_fusion(vec![first, second], &[0.5, 0.5]).unwrap();
        assert_eq!(fused.len(), 1);
        assert_eq!(fused[0].id, "v1");
        assert_eq!(fused[0].metadata.get("source"), Some(&json!("vector")));
        assert!((fused[0].score - 1.0).abs() < TOLERANCE);
    }

    #[test]
    fn weighted_fusion_rejects_mismatched_weights() {
        let error = weighted_fusion(vec![Vec::new()], &[0.5, 0.5]).unwrap_err();
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Validation
        );
    }

    #[test]
    fn rrf_weighted_multiplies_each_channel_contribution() {
        let first = vec![hit("a", "alpha", 1.0), hit("b", "beta", 1.0)];
        let second = vec![hit("c", "beta", 1.0)];
        let fused = rrf_fusion_weighted(vec![first, second], 40, &[1.0, 0.5]).unwrap();
        // beta: 1/42 + 0.5/41 > alpha: 1/41.
        assert_eq!(texts(&fused), vec!["beta", "alpha"]);
        assert!((fused[0].score - (1.0 / 42.0 + 0.5 / 41.0)).abs() < TOLERANCE);
        assert!((fused[1].score - 1.0 / 41.0).abs() < TOLERANCE);

        let error = rrf_fusion_weighted(vec![Vec::new()], 40, &[]).unwrap_err();
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Validation
        );
    }

    #[test]
    fn rrf_channels_honour_the_active_mask() {
        let config = RrfRankConfig {
            k: 40,
            name_dense: false,
            content_dense: true,
            content_sparse: false,
        };
        let channels = vec![
            vec![hit("n", "name", 1.0)],
            vec![hit("d", "dense", 3.0)],
            vec![hit("s", "sparse", 5.0)],
        ];
        let fused = rrf_fusion_channels(channels, &config);
        assert_eq!(texts(&fused), vec!["dense"]);
        assert!((fused[0].score - 1.0 / 41.0).abs() < TOLERANCE);

        // The default RRF config enables every channel and keeps k = 40.
        let all = rrf_fusion_channels(
            vec![
                Vec::new(),
                vec![hit("d", "dense", 3.0)],
                vec![hit("s", "sparse", 5.0)],
            ],
            &RrfRankConfig::default(),
        );
        assert_eq!(texts(&all), vec!["dense", "sparse"]);
        assert!((all[0].score - 1.0 / 41.0).abs() < TOLERANCE);
    }
}
