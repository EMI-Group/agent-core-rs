//! Ranking for the experience bank: hits, ranking configuration, and the two scorers.
//!
//! Two deterministic scorers share one filter/truncate/threshold pipeline:
//!
//! - Cosine similarity over embeddings (used whenever [`crate::experience::ExperienceBank`]
//!   has an [`openjiuwen_retrieval::Embedding`] configured). Python normalizes vectors
//!   to unit length and indexes them with inner product, which is exactly the cosine;
//!   the port computes the cosine directly so the score does not depend on whether the
//!   vectors were pre-normalized.
//! - A lexical term-coverage score over the normalized pattern, the example queries,
//!   and the skill ids. A distinct query term found in the pattern contributes `1.0`;
//!   one found only in the examples or skill ids contributes `0.5`; the sum is divided
//!   by the number of distinct query terms, so every score lands in `[0, 1]`.
//!
//! Both paths sort by score descending with ties broken by id ascending, so identical
//! catalogs always produce identical result orders — no hash-map iteration order or
//! wall-clock value reaches an output.

use std::collections::BTreeSet;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::experience::ExperienceItem;
use crate::search::tokenize;

/// The default `top_k` used by [`ExperienceRanking::default`] and [`SearchOptions::default`].
pub const DEFAULT_TOP_K: usize = 10;

/// The lexical weight of a query term matched by the item's pattern.
const PATTERN_WEIGHT: f64 = 1.0;

/// The lexical weight of a query term matched only by the examples or the skill ids.
const SUPPORT_WEIGHT: f64 = 0.5;

/// One ranked experience item and the score that placed it.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct ExperienceHit {
    /// The similarity score: cosine in `[-1, 1]` for embedding search, term coverage
    /// in `[0, 1]` for lexical search. Higher is more relevant.
    pub score: f64,
    /// The matched item.
    pub item: ExperienceItem,
}

impl ExperienceHit {
    /// The id of the matched item.
    #[must_use]
    pub fn id(&self) -> &str {
        &self.item.id
    }
}

/// The ranking half of a search: how many hits, how strict, and how skill filters combine.
///
/// This mirrors Python's `search_by_embedding(query, top_k=1, threshold=0.80)` knob set.
/// The Rust defaults are a `top_k` of [`DEFAULT_TOP_K`] and no score threshold, because
/// [`crate::experience::ExperienceBank::search`] takes `top_k` explicitly and a
/// threshold is only meaningful for the caller who knows the embedding model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct ExperienceRanking {
    /// The maximum number of hits; `0` means unlimited.
    pub top_k: usize,
    /// The minimum score a hit must reach; `None` keeps every hit.
    ///
    /// The threshold is applied after ranking and `top_k` truncation, matching
    /// Python's `search_by_embedding`.
    pub min_score: Option<f64>,
    /// Whether every entry of [`SearchOptions::required_skills`] must be present
    /// (`true`) or just one of them (`false`).
    pub require_all_skills: bool,
}

impl Default for ExperienceRanking {
    fn default() -> Self {
        Self {
            top_k: DEFAULT_TOP_K,
            min_score: None,
            require_all_skills: true,
        }
    }
}

impl ExperienceRanking {
    /// A ranking configuration with `top_k` and no threshold.
    #[must_use]
    pub fn new(top_k: usize) -> Self {
        Self {
            top_k,
            ..Self::default()
        }
    }

    /// Sets the minimum score.
    #[must_use]
    pub fn with_min_score(mut self, min_score: f64) -> Self {
        self.min_score = Some(min_score);
        self
    }

    /// Sets whether all required skills must be present.
    #[must_use]
    pub fn with_require_all_skills(mut self, require_all_skills: bool) -> Self {
        self.require_all_skills = require_all_skills;
        self
    }

    /// Converts this ranking into search options with no skill filter.
    #[must_use]
    pub fn options(&self) -> SearchOptions {
        SearchOptions {
            top_k: self.top_k,
            min_score: self.min_score,
            required_skills: Vec::new(),
            excluded_skills: Vec::new(),
            require_all_skills: self.require_all_skills,
        }
    }
}

/// The full search request: ranking, score threshold, and skill filters.
///
/// A hit is discarded before ranking when it lists any [`Self::excluded_skills`] id, or
/// when it does not satisfy [`Self::required_skills`] (every id when
/// [`Self::require_all_skills`] is `true`, at least one otherwise).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct SearchOptions {
    /// The maximum number of hits; `0` means unlimited.
    pub top_k: usize,
    /// The minimum score a hit must reach; `None` keeps every hit.
    pub min_score: Option<f64>,
    /// Skill ids the item must list (all of them by default; see
    /// [`Self::require_all_skills`]).
    pub required_skills: Vec<String>,
    /// Skill ids the item must not list.
    pub excluded_skills: Vec<String>,
    /// Whether every required skill must be present (`true`) or just one (`false`).
    pub require_all_skills: bool,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            top_k: DEFAULT_TOP_K,
            min_score: None,
            required_skills: Vec::new(),
            excluded_skills: Vec::new(),
            require_all_skills: true,
        }
    }
}

impl SearchOptions {
    /// Options that return at most `top_k` hits (`0` means unlimited) with no filters.
    #[must_use]
    pub fn new(top_k: usize) -> Self {
        Self {
            top_k,
            require_all_skills: true,
            ..Self::default()
        }
    }

    /// Options that return every hit (`top_k == 0`).
    #[must_use]
    pub fn unlimited() -> Self {
        Self::new(0)
    }

    /// Sets the minimum score.
    #[must_use]
    pub fn with_min_score(mut self, min_score: f64) -> Self {
        self.min_score = Some(min_score);
        self
    }

    /// Requires every listed skill id.
    #[must_use]
    pub fn with_required_skills(
        mut self,
        required_skills: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.required_skills = required_skills.into_iter().map(Into::into).collect();
        self
    }

    /// Excludes items listing any of the given skill ids.
    #[must_use]
    pub fn with_excluded_skills(
        mut self,
        excluded_skills: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.excluded_skills = excluded_skills.into_iter().map(Into::into).collect();
        self
    }

    /// Sets whether every required skill must be present, or just one.
    #[must_use]
    pub fn with_require_all_skills(mut self, require_all_skills: bool) -> Self {
        self.require_all_skills = require_all_skills;
        self
    }
}

/// Computes the cosine similarity of two vectors.
///
/// Returns `0.0` when either vector is empty, the two lengths differ (an embedding
/// from a different model is not comparable, and pretending otherwise would rank noise),
/// or either vector has zero norm. The result is clamped to `[-1, 1]` so float error
/// cannot produce an out-of-range score.
#[must_use]
pub fn cosine_similarity(left: &[f32], right: &[f32]) -> f64 {
    if left.is_empty() || left.len() != right.len() {
        return 0.0;
    }
    let mut dot = 0.0_f64;
    let mut left_norm = 0.0_f64;
    let mut right_norm = 0.0_f64;
    for (left_value, right_value) in left.iter().zip(right) {
        let left_value = f64::from(*left_value);
        let right_value = f64::from(*right_value);
        dot += left_value * right_value;
        left_norm += left_value * left_value;
        right_norm += right_value * right_value;
    }
    if left_norm == 0.0 || right_norm == 0.0 {
        return 0.0;
    }
    let similarity = dot / (left_norm.sqrt() * right_norm.sqrt());
    if similarity.is_finite() {
        similarity.clamp(-1.0, 1.0)
    } else {
        0.0
    }
}

/// The deterministic lexical score of `item` for the distinct `query_terms`.
///
/// Scores a query term matched by the normalized pattern at
/// [`PATTERN_WEIGHT`], one matched only by the examples or skill ids at
/// [`SUPPORT_WEIGHT`], and divides by the number of terms; a blank query scores `0.0`.
#[must_use]
pub(crate) fn lexical_score(query_terms: &BTreeSet<String>, item: &ExperienceItem) -> f64 {
    if query_terms.is_empty() {
        return 0.0;
    }
    let pattern_terms: BTreeSet<String> =
        tokenize(&item.normalized_pattern()).into_iter().collect();
    let support_terms: BTreeSet<String> = tokenize(&item.search_text()).into_iter().collect();
    let mut matched = 0.0_f64;
    for term in query_terms {
        if pattern_terms.contains(term) {
            matched += PATTERN_WEIGHT;
        } else if support_terms.contains(term) {
            matched += SUPPORT_WEIGHT;
        }
    }
    #[allow(clippy::cast_precision_loss)] // query term counts are far below 2^53
    let score = matched / query_terms.len() as f64;
    score.clamp(0.0, 1.0)
}

/// The distinct lowercased terms of `query`, sorted for deterministic iteration.
#[must_use]
pub(crate) fn query_terms(query: &str) -> BTreeSet<String> {
    tokenize(query).into_iter().collect()
}

/// Ranks `items` lexically and applies the filter/truncate/threshold pipeline.
pub(crate) fn rank_lexically(
    items: &[ExperienceItem],
    query: &str,
    options: &SearchOptions,
) -> Vec<ExperienceHit> {
    let terms = query_terms(query);
    let mut scored: Vec<ExperienceHit> = items
        .iter()
        .filter(|item| passes_filters(item, options))
        .map(|item| ExperienceHit {
            score: lexical_score(&terms, item),
            item: item.clone(),
        })
        .collect();
    finalize(&mut scored, options);
    scored
}

/// Ranks `items` by cosine similarity to `query_vector` and applies the same pipeline.
pub(crate) fn rank_by_embedding(
    items: &[ExperienceItem],
    query_vector: &[f32],
    options: &SearchOptions,
) -> Vec<ExperienceHit> {
    let mut scored: Vec<ExperienceHit> = items
        .iter()
        .filter(|item| passes_filters(item, options))
        .map(|item| ExperienceHit {
            score: cosine_similarity(query_vector, &item.embedding),
            item: item.clone(),
        })
        .collect();
    finalize(&mut scored, options);
    scored
}

/// Sorts by score descending (ties by id ascending), truncates to `top_k`, then applies
/// `min_score` — Python's `search_by_embedding` order.
fn finalize(scored: &mut Vec<ExperienceHit>, options: &SearchOptions) {
    for hit in scored.iter_mut() {
        if !hit.score.is_finite() {
            hit.score = 0.0;
        }
    }
    scored.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.item.id.cmp(&right.item.id))
    });
    if options.top_k > 0 {
        scored.truncate(options.top_k);
    }
    if let Some(minimum) = options.min_score {
        scored.retain(|hit| hit.score >= minimum);
    }
}

/// Reports whether `item` satisfies the skill filters of `options`.
fn passes_filters(item: &ExperienceItem, options: &SearchOptions) -> bool {
    if !options.excluded_skills.is_empty()
        && options
            .excluded_skills
            .iter()
            .any(|excluded| item.has_skill(excluded))
    {
        return false;
    }
    if options.required_skills.is_empty() {
        return true;
    }
    if options.require_all_skills {
        options
            .required_skills
            .iter()
            .all(|required| item.has_skill(required))
    } else {
        options
            .required_skills
            .iter()
            .any(|required| item.has_skill(required))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]
mod tests {
    use super::*;

    fn item(id: &str, pattern: &str, skills: &[&str]) -> ExperienceItem {
        ExperienceItem::new(pattern)
            .with_id(id)
            .with_skill_ids(skills.iter().copied())
    }

    #[test]
    fn cosine_is_one_for_parallel_and_zero_for_orthogonal() {
        let similarity = cosine_similarity(&[1.0, 0.0], &[2.0, 0.0]);
        assert!((similarity - 1.0).abs() < 1e-9);
        assert!(cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-9);
        assert!(cosine_similarity(&[1.0, 0.0], &[-1.0, 0.0]) < -0.99);
    }

    #[test]
    fn cosine_rejects_empty_and_mismatched_vectors() {
        assert!(cosine_similarity(&[], &[]).abs() < f64::EPSILON);
        assert!(cosine_similarity(&[1.0], &[1.0, 1.0]).abs() < f64::EPSILON);
        assert!(cosine_similarity(&[0.0, 0.0], &[1.0, 1.0]).abs() < f64::EPSILON);
    }

    #[test]
    fn lexical_score_rewards_pattern_matches_over_support_matches() {
        let pattern_hit = item("exp_0001", "translate text", &[]);
        let support_hit = item("exp_0002", "convert", &["translate"]);
        let terms = query_terms("translate");
        assert!((lexical_score(&terms, &pattern_hit) - 1.0).abs() < f64::EPSILON);
        assert!((lexical_score(&terms, &support_hit) - 0.5).abs() < f64::EPSILON);
        assert!(lexical_score(&BTreeSet::new(), &pattern_hit).abs() < f64::EPSILON);
    }

    #[test]
    fn ranking_is_bounded_and_ties_break_on_id() {
        let items = vec![
            item("exp_0002", "summarize", &["a"]),
            item("exp_0001", "summarize", &["b"]),
            item("exp_0003", "translate", &["c"]),
        ];
        let hits = rank_lexically(&items, "summarize", &SearchOptions::new(0));
        let ids: Vec<&str> = hits.iter().map(ExperienceHit::id).collect();
        assert_eq!(ids, vec!["exp_0001", "exp_0002", "exp_0003"]);

        let top = rank_lexically(&items, "summarize", &SearchOptions::new(1));
        assert_eq!(top.len(), 1);
        assert_eq!(top[0].id(), "exp_0001");
    }

    #[test]
    fn filters_require_and_exclude_skills() {
        let items = vec![
            item("exp_0001", "summarize", &["a", "b"]),
            item("exp_0002", "summarize", &["a"]),
        ];
        let all = SearchOptions::new(0).with_required_skills(["a", "b"]);
        let hits = rank_lexically(&items, "summarize", &all);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id(), "exp_0001");

        let any = SearchOptions::new(0)
            .with_required_skills(["a", "b"])
            .with_require_all_skills(false);
        assert_eq!(rank_lexically(&items, "summarize", &any).len(), 2);

        let excluded = SearchOptions::new(0).with_excluded_skills(["b"]);
        let hits = rank_lexically(&items, "summarize", &excluded);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id(), "exp_0002");
    }

    #[test]
    fn threshold_applies_after_truncation() {
        let items = vec![
            item("exp_0001", "summarize", &[]),
            item("exp_0002", "translate", &[]),
        ];
        // Both hits survive a two-slot request; only the strong one clears the threshold.
        let strict = SearchOptions::new(2).with_min_score(0.9);
        let hits = rank_lexically(&items, "translate", &strict);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id(), "exp_0002");

        // One slot: the strongest hit is below the threshold, so the result is empty
        // even though the next hit down the ranking would also have failed — the
        // threshold never widens the truncation, matching Python's order.
        let truncated = SearchOptions::new(1).with_min_score(0.9);
        assert!(rank_lexically(&items, "unrelated words", &truncated).is_empty());

        let dropped = SearchOptions::new(1)
            .with_min_score(0.9)
            .with_required_skills(["missing"]);
        assert!(rank_lexically(&items, "summarize", &dropped).is_empty());
    }
}
