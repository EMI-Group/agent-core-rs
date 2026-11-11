//! Cosine similarity, recency decay, the documented recall blend, and token budgeting.
//!
//! The scoring formula is pinned by the crate documentation and must stay identical
//! here, in [`blend`], and in `docs/PARITY.md`:
//!
//! ```text
//! semantic = (w_sim * similarity + w_kw * keyword) / (w_sim + w_kw)
//! final    = (w_sem * semantic + w_imp * importance + w_rec * recency)
//!            / (w_sem + w_imp + w_rec)
//! recency  = 0.5 ^ (age_secs / half_life_secs)
//! ```
//!
//! All inputs are clamped to `[0, 1]`, every division is guarded, and no `NaN`/infinite
//! value can escape: a zero-weight group is treated as a denominator of `1`, which
//! makes the remaining terms of that group dominate through the weighted average.
use crate::config::RankingConfig;
use crate::record::RecallHit;
use chrono::{DateTime, Utc};
use openjiuwen_foundation::prompt::tokens::TokenCounter;
/// Clamps a score component into `[0, 1]`, mapping non-finite values to `0.0`.
fn clamp_unit(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}
/// Returns the raw cosine similarity of two embedding vectors, in `[-1, 1]`.
///
/// The result is `0.0` when the vectors have different lengths, when either is empty,
/// when any component is non-finite, or when either vector has zero norm (the cosine is
/// undefined there). The dot product and norms are accumulated in `f64` so long
/// embeddings do not lose precision before the final `f32` conversion.
#[must_use]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    let mut dot = 0.0_f64;
    let mut norm_a = 0.0_f64;
    let mut norm_b = 0.0_f64;
    for (left, right) in a.iter().zip(b.iter()) {
        if !left.is_finite() || !right.is_finite() {
            return 0.0;
        }
        let left = f64::from(*left);
        let right = f64::from(*right);
        dot += left * right;
        norm_a += left * left;
        norm_b += right * right;
    }
    if norm_a <= 0.0 || norm_b <= 0.0 {
        return 0.0;
    }
    let cosine = dot / (norm_a.sqrt() * norm_b.sqrt());
    if cosine.is_finite() {
        cosine.clamp(-1.0, 1.0) as f32
    } else {
        0.0
    }
}
/// Returns the cosine similarity mapped from `[-1, 1]` into `[0, 1]`.
///
/// `(cosine + 1) / 2` keeps anti-correlated vectors distinguishable (`0.0` instead of
/// being clamped away). Like [`cosine_similarity`], the result is `0.0` for empty,
/// mismatched, non-finite, or zero-norm vectors — the "no semantic signal" value.
#[must_use]
pub fn normalized_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    if a.iter().chain(b.iter()).any(|value| !value.is_finite()) {
        return 0.0;
    }
    if a.iter().all(|value| *value == 0.0) || b.iter().all(|value| *value == 0.0) {
        return 0.0;
    }
    f32::midpoint(cosine_similarity(a, b), 1.0).clamp(0.0, 1.0)
}
/// Returns the exponential recency of `timestamp` relative to `now`.
///
/// `recency = 0.5 ^ (age_secs / half_life_secs)`, so the score is `1.0` at age `0`,
/// `0.5` at one half-life, and `0.25` at two half-lives. A timestamp in the future has
/// its age clamped to `0` (score `1.0`) rather than exceeding `1.0`. A `half_life_secs`
/// of `0` decays instantly: `1.0` for a zero age, `0.0` otherwise.
#[must_use]
pub fn recency_score(timestamp: DateTime<Utc>, now: DateTime<Utc>, half_life_secs: u64) -> f32 {
    let age_secs = (now - timestamp).num_seconds().max(0) as f64;
    if age_secs == 0.0 {
        return 1.0;
    }
    if half_life_secs == 0 {
        return 0.0;
    }
    let half_life = half_life_secs as f64;
    let decayed = 0.5_f64.powf(age_secs / half_life);
    clamp_unit(decayed as f32)
}
/// Returns the final blended recall score for one candidate.
///
/// Implements the documented two-stage weighted average:
///
/// ```text
/// semantic = (w_sim * similarity + w_kw * keyword) / (w_sim + w_kw)
/// final    = (w_sem * semantic + w_imp * importance + w_rec * recency)
///            / (w_sem + w_imp + w_rec)
/// ```
///
/// Every argument and weight is clamped to `[0, 1]`/`[0, ∞)`, and a zero denominator is
/// treated as `1`, so the result is always a finite value in `[0, 1]`. Because the
/// weight groups are renormalized per group, setting a weight to `0` disables that term.
#[must_use]
pub fn blend(
    similarity: f32,
    keyword: f32,
    importance: f32,
    recency: f32,
    config: &RankingConfig,
) -> f32 {
    let similarity_weight = clamp_weight(config.similarity_weight);
    let keyword_weight = clamp_weight(config.keyword_weight);
    let semantic_weight = clamp_weight(config.semantic_weight);
    let importance_weight = clamp_weight(config.importance_weight);
    let recency_weight = clamp_weight(config.recency_weight);
    let semantic_group = similarity_weight + keyword_weight;
    let semantic_group = if semantic_group > 0.0 {
        semantic_group
    } else {
        1.0
    };
    let semantic = (similarity_weight * clamp_unit(similarity)
        + keyword_weight * clamp_unit(keyword))
        / semantic_group;
    let final_group = semantic_weight + importance_weight + recency_weight;
    let final_group = if final_group > 0.0 { final_group } else { 1.0 };
    let final_score = (semantic_weight * semantic
        + importance_weight * clamp_unit(importance)
        + recency_weight * clamp_unit(recency))
        / final_group;
    clamp_unit(final_score)
}
/// Clamps a weight to a finite non-negative value, mapping invalid weights to `0.0`.
fn clamp_weight(weight: f32) -> f32 {
    if weight.is_finite() {
        weight.max(0.0)
    } else {
        0.0
    }
}
/// Keeps the highest-scoring hits that fit in a token budget.
///
/// `hits` must already be in score order; the returned prefix keeps that order and
/// stops at the first hit whose `counter.count_text(record.content)` would exceed the
/// remaining budget. A `budget` of `0` yields an empty vector, and empty content costs
/// zero tokens. Callers that have no budget pass the hits through untrimmed.
#[must_use]
pub fn trim_to_token_budget(
    hits: Vec<RecallHit>,
    budget: usize,
    counter: &dyn TokenCounter,
) -> Vec<RecallHit> {
    if budget == 0 {
        return Vec::new();
    }
    let mut kept = Vec::new();
    let mut used = 0usize;
    for hit in hits {
        let cost = counter.count_text(&hit.record.content);
        let Some(next) = used.checked_add(cost) else {
            break;
        };
        if next > budget {
            break;
        }
        used = next;
        kept.push(hit);
    }
    kept
}
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::record::{MemoryRecord, MemoryScope, MemoryType};
    use openjiuwen_foundation::prompt::tokens::ApproxTokenCounter;
    fn fixed_time() -> DateTime<Utc> {
        DateTime::from_timestamp(1_700_000_000, 0).unwrap()
    }
    fn hit(id: &str, content: &str, score: f32) -> RecallHit {
        RecallHit {
            record: MemoryRecord::new(
                MemoryScope::new("alice", "ws"),
                content,
                MemoryType::SemanticMemory,
            )
            .with_id(id),
            score,
            similarity: 0.5,
            keyword: 0.5,
            importance: 0.5,
            recency: 0.5,
        }
    }
    fn close(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() < 1e-5,
            "expected {expected}, got {actual}"
        );
    }
    #[test]
    fn cosine_matches_known_values() {
        close(cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]), 1.0);
        close(cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]), 0.0);
        close(cosine_similarity(&[1.0, 0.0], &[-1.0, 0.0]), -1.0);
        // 45 degree angle.
        close(
            cosine_similarity(&[1.0, 0.0], &[1.0, 1.0]),
            std::f32::consts::FRAC_1_SQRT_2,
        );
        close(cosine_similarity(&[2.0, 0.0], &[5.0, 0.0]), 1.0);
    }
    #[test]
    fn cosine_edge_cases_are_zero() {
        assert_eq!(cosine_similarity(&[], &[]), 0.0);
        assert_eq!(cosine_similarity(&[1.0], &[1.0, 2.0]), 0.0);
        assert_eq!(cosine_similarity(&[0.0, 0.0], &[1.0, 0.0]), 0.0);
        assert_eq!(cosine_similarity(&[f32::NAN, 1.0], &[1.0, 1.0]), 0.0);
        assert_eq!(cosine_similarity(&[f32::INFINITY], &[1.0]), 0.0);
        assert!(cosine_similarity(&[0.5; 512], &[0.25; 512]).is_finite());
    }
    #[test]
    fn normalized_similarity_maps_into_unit_range() {
        close(normalized_similarity(&[1.0, 0.0], &[1.0, 0.0]), 1.0);
        close(normalized_similarity(&[1.0, 0.0], &[-1.0, 0.0]), 0.0);
        close(normalized_similarity(&[1.0, 0.0], &[0.0, 1.0]), 0.5);
        assert_eq!(normalized_similarity(&[], &[]), 0.0);
        assert_eq!(normalized_similarity(&[1.0], &[1.0, 2.0]), 0.0);
        assert_eq!(normalized_similarity(&[0.0, 0.0], &[1.0, 0.0]), 0.0);
        assert_eq!(normalized_similarity(&[f32::NAN], &[1.0]), 0.0);
    }
    #[test]
    fn recency_decays_by_half_life() {
        let now = fixed_time();
        let half_life = 100_u64;
        close(recency_score(now, now, half_life), 1.0);
        close(
            recency_score(now - chrono::Duration::seconds(100), now, half_life),
            0.5,
        );
        close(
            recency_score(now - chrono::Duration::seconds(200), now, half_life),
            0.25,
        );
        let old = recency_score(now - chrono::Duration::days(30), now, 2_592_000);
        close(old, 0.5);
    }
    #[test]
    fn recency_clamps_future_timestamps_and_zero_half_life() {
        let now = fixed_time();
        close(
            recency_score(now + chrono::Duration::seconds(500), now, 100),
            1.0,
        );
        close(recency_score(now, now, 0), 1.0);
        close(
            recency_score(now - chrono::Duration::seconds(1), now, 0),
            0.0,
        );
        assert!((0.0..=1.0).contains(&recency_score(now - chrono::Duration::days(3650), now, 60)));
    }
    #[test]
    fn blend_matches_the_documented_formula() {
        let config = RankingConfig::default();
        // semantic = (0.70*0.8 + 0.30*0.4)/1.0 = 0.68
        // final = (0.85*0.68 + 0.10*0.5 + 0.05*0.25)/1.0 = 0.578 + 0.05 + 0.0125
        let score = blend(0.8, 0.4, 0.5, 0.25, &config);
        close(score, 0.6405);
    }
    #[test]
    fn blend_similarity_only_configuration_ignores_keyword() {
        let config = RankingConfig {
            similarity_weight: 1.0,
            keyword_weight: 0.0,
            semantic_weight: 1.0,
            importance_weight: 0.0,
            recency_weight: 0.0,
            ..Default::default()
        };
        close(blend(0.9, 0.0, 0.0, 0.0, &config), 0.9);
        close(blend(0.9, 1.0, 1.0, 1.0, &config), 0.9);
    }
    #[test]
    fn blend_keyword_only_configuration_ignores_similarity() {
        let config = RankingConfig {
            similarity_weight: 0.0,
            keyword_weight: 1.0,
            semantic_weight: 1.0,
            importance_weight: 0.0,
            recency_weight: 0.0,
            ..Default::default()
        };
        close(blend(0.0, 0.6, 0.0, 0.0, &config), 0.6);
        close(blend(1.0, 0.6, 1.0, 1.0, &config), 0.6);
    }
    #[test]
    fn blend_importance_and_recency_only_configuration() {
        let config = RankingConfig {
            similarity_weight: 1.0,
            keyword_weight: 0.0,
            semantic_weight: 0.0,
            importance_weight: 0.5,
            recency_weight: 0.5,
            ..Default::default()
        };
        close(blend(0.0, 0.0, 1.0, 0.0, &config), 0.5);
        close(blend(1.0, 1.0, 0.5, 0.5, &config), 0.5);
    }
    #[test]
    fn blend_clamps_inputs_and_never_returns_nan() {
        let config = RankingConfig::default();
        close(blend(2.0, 2.0, 2.0, 2.0, &config), 1.0);
        close(blend(-5.0, -5.0, -5.0, -5.0, &config), 0.0);
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let score = blend(value, value, value, value, &config);
            assert!(score.is_finite(), "score for {value} must be finite");
            close(score, 0.0);
        }
    }
    #[test]
    fn blend_treats_zero_denominators_as_one() {
        let config = RankingConfig {
            similarity_weight: 0.0,
            keyword_weight: 0.0,
            semantic_weight: 0.0,
            importance_weight: 0.0,
            recency_weight: 0.0,
            ..Default::default()
        };
        close(blend(1.0, 1.0, 1.0, 1.0, &config), 0.0);
        assert!(blend(1.0, 1.0, 1.0, 1.0, &config).is_finite());
        let keyword_only = RankingConfig {
            similarity_weight: 0.0,
            keyword_weight: 1.0,
            semantic_weight: 1.0,
            importance_weight: 0.0,
            recency_weight: 0.0,
            ..Default::default()
        };
        close(blend(0.0, 0.42, 0.0, 0.0, &keyword_only), 0.42);
    }
    #[test]
    fn trim_keeps_a_prefix_within_budget() {
        let counter = ApproxTokenCounter::new();
        let hits = vec![
            hit("a", "12345678", 0.9), // 2 tokens
            hit("b", "12345678", 0.8), // 2 tokens
            hit("c", "12345678", 0.7), // 2 tokens
        ];
        let trimmed = trim_to_token_budget(hits.clone(), 4, &counter);
        assert_eq!(trimmed.len(), 2);
        assert_eq!(trimmed[0].record.id, "a");
        assert_eq!(trimmed[1].record.id, "b");
        let all = trim_to_token_budget(hits.clone(), 6, &counter);
        assert_eq!(all.len(), 3);
        let none = trim_to_token_budget(hits.clone(), 1, &counter);
        assert_eq!(none, Vec::<RecallHit>::new());
        assert_eq!(
            trim_to_token_budget(hits.clone(), 0, &counter),
            Vec::<RecallHit>::new()
        );
        assert_eq!(trim_to_token_budget(hits, 100, &counter).len(), 3);
    }
    #[test]
    fn trim_stops_at_the_first_hit_that_does_not_fit() {
        let counter = ApproxTokenCounter::new();
        let hits = vec![
            hit("a", "", 0.9),         // 0 tokens
            hit("b", "1234", 0.8),     // 1 token
            hit("c", "12345678", 0.7), // 2 tokens
            hit("d", "1", 0.6),        // 1 token
        ];
        let trimmed = trim_to_token_budget(hits, 2, &counter);
        let ids: Vec<&str> = trimmed.iter().map(|hit| hit.record.id.as_str()).collect();
        assert_eq!(ids, vec!["a", "b"]);
    }
}
