//! Rank fusion helpers: Reciprocal Rank Fusion and text deduplication.
//!
//! Port of the Python `openjiuwen/core/retrieval/utils/fusion.py`. Results are
//! keyed by their **text** (Python uses `result.text` as the identity because
//! chunk ids differ between the vector and lexical channels), and the first-seen
//! result object supplies the metadata of a fused hit.

use indexmap::IndexMap;

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
    let mut fused: IndexMap<String, (SearchResult, f32)> = IndexMap::new();
    for results in results_list {
        for (position, result) in results.into_iter().enumerate() {
            let rank = position + 1;
            let contribution = 1.0 / (k + rank) as f32;
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

/// [`rrf_fusion`] with the Python default constant [`DEFAULT_RRF_K`].
pub fn rrf_fusion_default(results_list: Vec<Vec<SearchResult>>) -> Vec<SearchResult> {
    rrf_fusion(results_list, DEFAULT_RRF_K)
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
}
