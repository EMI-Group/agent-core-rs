//! Candidate ranking: turn scored candidates into sorted [`RecallHit`]s.
//!
//! The engine assembles candidates (vector search hits and/or keyword scope scans),
//! then calls [`rank`] once so every candidate is scored against the same corpus and
//! the same `now`. Ranking never applies a score threshold — the engine does that
//! afterwards — so the components of every hit stay inspectable for diagnostics.
use crate::config::RankingConfig;
use crate::recall::keyword::normalized_bm25_scores;
use crate::recall::scoring::{blend, recency_score};
use crate::record::{MemoryRecord, RecallHit};
use chrono::{DateTime, Utc};
/// One candidate entering the ranking stage.
///
/// `similarity` is the normalized semantic similarity when the candidate came from a
/// vector search; `None` marks a keyword-only candidate (scope scan or embedding-less
/// store) and contributes `similarity = 0.0` to the blend. `used_keyword_fallback`
/// records why the candidate exists so callers can log or filter the degraded path.
#[derive(Debug, Clone, PartialEq)]
pub struct ScoredCandidate {
    /// The candidate record.
    pub record: MemoryRecord,
    /// Normalized semantic similarity in `[0, 1]`, when a vector search produced it.
    pub similarity: Option<f32>,
    /// Whether this candidate came from the keyword fallback rather than vector search.
    pub used_keyword_fallback: bool,
}
impl ScoredCandidate {
    /// Builds a semantic candidate with an explicit similarity.
    #[must_use]
    pub fn new(record: MemoryRecord, similarity: Option<f32>) -> Self {
        let used_keyword_fallback = similarity.is_none();
        Self {
            record,
            similarity,
            used_keyword_fallback,
        }
    }
    /// Builds a keyword-only candidate (no semantic signal).
    #[must_use]
    pub fn keyword_only(record: MemoryRecord) -> Self {
        Self {
            record,
            similarity: None,
            used_keyword_fallback: true,
        }
    }
}
/// Clamps a component into `[0, 1]`, mapping non-finite values to `0.0`.
fn clamp_unit(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}
/// Scores and orders the candidates against `query`.
///
/// For every candidate the keyword component is the normalized `BM25` of its content
/// over the candidate corpus, the recency component comes from [`MemoryRecord::timestamp`]
/// relative to `now` using [`RankingConfig::recency_half_life_secs`], the importance
/// component is the record's own `importance`, and the similarity component is
/// `candidate.similarity` (or `0.0` for a keyword-only candidate). Those four values are
/// combined with [`blend`]:
///
/// ```text
/// semantic = (w_sim * similarity + w_kw * keyword) / (w_sim + w_kw)
/// final    = (w_sem * semantic + w_imp * importance + w_rec * recency)
///            / (w_sem + w_imp + w_rec)
/// ```
///
/// The result is sorted by `score` descending and then by record `id` ascending, so
/// identical inputs always produce the same order. Every `RecallHit` exposes the exact
/// component values that produced its score. No threshold filtering happens here.
#[must_use]
pub fn rank(
    candidates: Vec<ScoredCandidate>,
    query: &str,
    now: DateTime<Utc>,
    config: &RankingConfig,
) -> Vec<RecallHit> {
    if candidates.is_empty() {
        return Vec::new();
    }
    let documents: Vec<&str> = candidates
        .iter()
        .map(|candidate| candidate.record.content.as_str())
        .collect();
    let keyword_scores = normalized_bm25_scores(query, &documents);
    let mut hits: Vec<RecallHit> = candidates
        .into_iter()
        .zip(keyword_scores)
        .map(|(candidate, keyword)| {
            let similarity = candidate.similarity.map_or(0.0, clamp_unit);
            let importance = clamp_unit(candidate.record.importance);
            let recency = recency_score(
                candidate.record.timestamp,
                now,
                config.recency_half_life_secs,
            );
            let score = blend(similarity, keyword, importance, recency, config);
            RecallHit {
                record: candidate.record,
                score,
                similarity,
                keyword: clamp_unit(keyword),
                importance,
                recency,
            }
        })
        .collect();
    hits.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.record.id.cmp(&right.record.id))
    });
    hits
}
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::recall::scoring::normalized_similarity;
    use crate::record::{MemoryScope, MemoryType};
    fn fixed_time() -> DateTime<Utc> {
        DateTime::from_timestamp(1_700_000_000, 0).unwrap()
    }
    fn record(id: &str, content: &str) -> MemoryRecord {
        MemoryRecord::new(
            MemoryScope::new("alice", "ws"),
            content,
            MemoryType::SemanticMemory,
        )
        .with_id(id)
        .with_timestamp(fixed_time())
    }
    #[test]
    fn ranks_by_blended_score_with_exact_components() {
        let candidates = vec![
            ScoredCandidate::new(record("b", "alice prefers light mode"), Some(0.4)),
            ScoredCandidate::new(record("a", "alice prefers dark mode"), Some(0.9)),
        ];
        let hits = rank(
            candidates,
            "dark mode",
            fixed_time(),
            &RankingConfig::default(),
        );
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].record.id, "a");
        assert_eq!(hits[1].record.id, "b");
        assert!(hits[0].score >= hits[1].score);
        // Every component must equal the value it was computed from.
        for hit in &hits {
            let expected_recency = recency_score(hit.record.timestamp, fixed_time(), 2_592_000);
            assert!((hit.recency - expected_recency).abs() < f32::EPSILON);
            assert!((hit.importance - hit.record.importance).abs() < f32::EPSILON);
            let expected = blend(
                hit.similarity,
                hit.keyword,
                hit.importance,
                hit.recency,
                &RankingConfig::default(),
            );
            assert!((hit.score - expected).abs() < 1e-6);
        }
        // The best keyword match wins the keyword component.
        assert!((hits[0].keyword - 1.0).abs() < f32::EPSILON);
        assert!(hits[1].keyword < 1.0);
        assert!((hits[0].similarity - 0.9).abs() < f32::EPSILON);
    }
    #[test]
    fn sorts_ties_by_record_id() {
        let content = "alice prefers dark mode";
        let candidates = vec![
            ScoredCandidate::new(record("zzz", content), Some(0.5)),
            ScoredCandidate::new(record("aaa", content), Some(0.5)),
            ScoredCandidate::new(record("mmm", content), Some(0.5)),
        ];
        let hits = rank(
            candidates,
            "dark mode",
            fixed_time(),
            &RankingConfig::default(),
        );
        let ids: Vec<&str> = hits.iter().map(|hit| hit.record.id.as_str()).collect();
        assert_eq!(ids, vec!["aaa", "mmm", "zzz"]);
        assert!(hits.windows(2).all(|pair| pair[0].score >= pair[1].score));
    }
    #[test]
    fn keyword_only_candidates_rely_on_bm25() {
        let candidates = vec![
            ScoredCandidate::keyword_only(record("b", "the weather is nice")),
            ScoredCandidate::keyword_only(record("a", "dark mode preference")),
        ];
        let hits = rank(
            candidates,
            "dark mode",
            fixed_time(),
            &RankingConfig::default(),
        );
        assert_eq!(hits[0].record.id, "a");
        assert!((hits[0].similarity - 0.0).abs() < f32::EPSILON);
        assert!((hits[1].similarity - 0.0).abs() < f32::EPSILON);
        assert!((hits[0].keyword - 1.0).abs() < f32::EPSILON);
        assert!((hits[1].keyword - 0.0).abs() < f32::EPSILON);
        // Keyword-only scores are still positive because keyword is 30% of 85%.
        assert!(hits[0].score > 0.0);
    }
    #[test]
    fn empty_query_falls_back_to_importance_and_recency() {
        let now = fixed_time();
        let mut older = record("older", "anything at all");
        older.timestamp = now - chrono::Duration::days(30);
        older.importance = 0.2;
        let mut newer = record("newer", "something else entirely");
        newer.timestamp = now;
        newer.importance = 0.9;
        let hits = rank(
            vec![
                ScoredCandidate::new(older, Some(0.5)),
                ScoredCandidate::new(newer, Some(0.5)),
            ],
            "",
            now,
            &RankingConfig::default(),
        );
        assert_eq!(hits[0].record.id, "newer");
        assert!((hits[0].keyword - 0.0).abs() < f32::EPSILON);
        assert!((hits[1].keyword - 0.0).abs() < f32::EPSILON);
        assert!((hits[0].recency - 1.0).abs() < f32::EPSILON);
        assert!((hits[1].recency - 0.5).abs() < 1e-6);
        assert!(hits[0].score > hits[1].score);
    }
    #[test]
    fn component_edges_are_clamped_and_finite() {
        let mut candidate = ScoredCandidate::new(record("a", "dark mode"), Some(f32::NAN));
        candidate.similarity = Some(f32::INFINITY);
        let mut other = record("b", "dark mode again");
        other.importance = 5.0;
        let hits = rank(
            vec![candidate, ScoredCandidate::new(other, Some(-1.0))],
            "dark mode",
            fixed_time(),
            &RankingConfig::default(),
        );
        for hit in &hits {
            assert!(hit.score.is_finite());
            assert!((0.0..=1.0).contains(&hit.score));
            assert!((0.0..=1.0).contains(&hit.similarity));
            assert!((0.0..=1.0).contains(&hit.importance));
            assert!((0.0..=1.0).contains(&hit.recency));
            assert!((0.0..=1.0).contains(&hit.keyword));
        }
        let by_id = |id: &str| hits.iter().find(|hit| hit.record.id == id).unwrap();
        assert!((by_id("a").similarity - 0.0).abs() < f32::EPSILON);
        assert!((by_id("a").importance - 0.5).abs() < f32::EPSILON);
        assert!((by_id("b").similarity - 0.0).abs() < f32::EPSILON);
        assert!((by_id("b").importance - 1.0).abs() < f32::EPSILON);
    }
    #[test]
    fn no_threshold_filtering_happens() {
        let candidates = vec![ScoredCandidate::keyword_only(record(
            "a",
            "completely unrelated",
        ))];
        let hits = rank(
            candidates,
            "dark mode",
            fixed_time(),
            &RankingConfig::default(),
        );
        assert_eq!(hits.len(), 1);
        assert!(hits[0].score > 0.0);
        assert!(hits[0].score < 0.15);
    }
    #[test]
    fn empty_candidate_list_yields_empty_output() {
        assert_eq!(
            rank(Vec::new(), "dark", fixed_time(), &RankingConfig::default()),
            Vec::<RecallHit>::new()
        );
    }
    #[test]
    fn similarity_component_matches_the_normalized_cosine() {
        let expected = normalized_similarity(&[1.0, 0.0], &[0.5, 0.5]);
        let mut rec = record("a", "dark mode");
        rec.embedding = Some(vec![1.0, 0.0]);
        let candidate = ScoredCandidate::new(rec, Some(expected));
        let hits = rank(
            vec![candidate],
            "dark mode",
            fixed_time(),
            &RankingConfig::default(),
        );
        assert!((hits[0].similarity - expected).abs() < f32::EPSILON);
    }
    #[test]
    fn keyword_fallback_flag_is_preserved_by_constructors() {
        let semantic = ScoredCandidate::new(record("a", "x"), Some(0.5));
        assert!(semantic.similarity.is_some());
        assert!(!semantic.used_keyword_fallback);
        let fallback = ScoredCandidate::keyword_only(record("b", "x"));
        assert!(fallback.similarity.is_none());
        assert!(fallback.used_keyword_fallback);
    }
}
