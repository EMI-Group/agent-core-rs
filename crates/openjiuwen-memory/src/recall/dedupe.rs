//! Deterministic near-duplicate detection and record merging.
//!
//! Python's `MemUpdateChecker` asks an LLM to classify recalled memories as redundant,
//! conflicting, or unrelated. This module keeps the observable effect (merge instead of
//! duplicating) with fully deterministic rules so the write path is testable offline:
//! exact normalized-text equality, embedding cosine, and token Jaccard.
use crate::config::DedupeConfig;
use crate::recall::keyword::jaccard;
use crate::recall::scoring::normalized_similarity;
use crate::record::normalize_text;
use crate::record::{MemoryRecord, MemoryType};
use chrono::{DateTime, Utc};
/// A record recalled as a dedupe candidate, with the semantic similarity that produced
/// it.
///
/// The candidate comes either from a vector search (which sets `similarity`) or from a
/// keyword scope scan (`None`). Detection never mutates it.
#[derive(Debug, Clone, PartialEq)]
pub struct Candidate {
    /// The stored record under comparison.
    pub record: MemoryRecord,
    /// Normalized semantic similarity of the recall that produced this candidate.
    pub similarity: Option<f32>,
}
impl Candidate {
    /// Builds a candidate with an optional semantic similarity.
    #[must_use]
    pub fn new(record: MemoryRecord, similarity: Option<f32>) -> Self {
        Self { record, similarity }
    }
    /// Returns the candidate's similarity clamped into `[0, 1]`, or `0.0` when absent.
    fn similarity_score(&self) -> f32 {
        match self.similarity {
            Some(value) if value.is_finite() => value.clamp(0.0, 1.0),
            _ => 0.0,
        }
    }
}
/// How an incoming record matched an existing one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DuplicateKind {
    /// Both contents normalize to the same text.
    Exact,
    /// Cosine similarity or token Jaccard crossed the configured threshold.
    Similar,
    /// Both records carry the same id.
    SameId,
}
impl DuplicateKind {
    /// The lowercase name used in logs and JSON-ish diagnostics.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Exact => "exact",
            Self::Similar => "similar",
            Self::SameId => "same_id",
        }
    }
}
/// A duplicate found among the dedupe candidates.
#[derive(Debug, Clone, PartialEq)]
pub struct DuplicateMatch {
    /// Index into the candidate slice the caller supplied.
    pub index: usize,
    /// Which rule matched.
    pub kind: DuplicateKind,
    /// Similarity that justified the match: `1.0` for exact text, the semantic or
    /// Jaccard similarity for fuzzy matches, the candidate similarity for same-id
    /// matches.
    pub similarity: f32,
}
/// Finds the first duplicate of `incoming` among `candidates`.
///
/// Rule order for each candidate (highest-similarity candidates first, at most
/// `config.candidate_top_k` of them):
/// 1. same record id — [`DuplicateKind::SameId`];
/// 2. equal [`normalize_text`] output (non-empty) — [`DuplicateKind::Exact`];
/// 3. for contents of at least `config.min_content_len` characters, embedding cosine
///    `>= config.similarity_threshold` (falling back to the candidate's recall
///    similarity when an embedding is missing) or token Jaccard
///    `>= config.jaccard_threshold` — [`DuplicateKind::Similar`].
///
/// Candidates are ordered by `similarity` descending (ties by original index), so the
/// best match wins and equal-similarity candidates resolve deterministically. This
/// function is pure: it never consults `config.enabled`/`config.policy` — the engine
/// gates those.
#[must_use]
pub fn find_duplicate(
    incoming: &MemoryRecord,
    candidates: &[Candidate],
    config: &DedupeConfig,
) -> Option<DuplicateMatch> {
    if candidates.is_empty() {
        return None;
    }
    let mut order: Vec<usize> = (0..candidates.len()).collect();
    order.sort_by(|left, right| {
        let left_score = candidates[*left].similarity_score();
        let right_score = candidates[*right].similarity_score();
        right_score
            .partial_cmp(&left_score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.cmp(right))
    });
    let limit = if config.candidate_top_k == 0 {
        candidates.len()
    } else {
        config.candidate_top_k.min(candidates.len())
    };
    order.truncate(limit);
    let incoming_text = normalize_text(&incoming.content);
    let incoming_len = incoming.content.chars().count();
    for index in order {
        let candidate = &candidates[index];
        let stored = &candidate.record;
        if stored.id == incoming.id {
            return Some(DuplicateMatch {
                index,
                kind: DuplicateKind::SameId,
                similarity: candidate.similarity_score(),
            });
        }
        if !incoming_text.is_empty() && normalize_text(&stored.content) == incoming_text {
            return Some(DuplicateMatch {
                index,
                kind: DuplicateKind::Exact,
                similarity: 1.0,
            });
        }
        let stored_len = stored.content.chars().count();
        if incoming_len < config.min_content_len || stored_len < config.min_content_len {
            continue;
        }
        let similarity = match (incoming.embedding.as_deref(), stored.embedding.as_deref()) {
            (Some(left), Some(right)) => normalized_similarity(left, right),
            _ => candidate.similarity_score(),
        };
        let lexical = jaccard(&incoming.content, &stored.content);
        if similarity >= config.similarity_threshold || lexical >= config.jaccard_threshold {
            return Some(DuplicateMatch {
                index,
                kind: DuplicateKind::Similar,
                similarity: similarity.max(lexical).clamp(0.0, 1.0),
            });
        }
    }
    None
}
/// Merges `incoming` into `existing` under the documented field rules.
///
/// - `id`, `scope`, and `timestamp` are kept from `existing`;
/// - `content` becomes the longer text (`incoming` on a tie);
/// - `metadata` is the union with incoming values winning per key;
/// - `importance` is the maximum of the two (non-finite values are ignored);
/// - `access_count` is the sum of both sides;
/// - `source` prefers `incoming` when it has one;
/// - `kind` is `incoming.kind` unless that is [`MemoryType::Unknown`];
/// - `last_access` is the maximum of the two;
/// - `updated_at` is `now` and `embedding` is cleared, because the engine re-embeds the
///   merged content.
///
/// Neither input is modified.
#[must_use]
pub fn merge_records(
    existing: &MemoryRecord,
    incoming: &MemoryRecord,
    now: DateTime<Utc>,
) -> MemoryRecord {
    let content = if incoming.content.chars().count() > existing.content.chars().count() {
        incoming.content.clone()
    } else {
        existing.content.clone()
    };
    let mut metadata = existing.metadata.clone();
    for (key, value) in &incoming.metadata {
        metadata.insert(key.clone(), value.clone());
    }
    let kind = if incoming.kind == MemoryType::Unknown {
        existing.kind
    } else {
        incoming.kind
    };
    let last_access = match (existing.last_access, incoming.last_access) {
        (Some(left), Some(right)) => Some(left.max(right)),
        (Some(value), None) | (None, Some(value)) => Some(value),
        (None, None) => None,
    };
    MemoryRecord {
        id: existing.id.clone(),
        content,
        kind,
        importance: max_importance(existing.importance, incoming.importance),
        timestamp: existing.timestamp,
        updated_at: now,
        last_access,
        access_count: existing.access_count.saturating_add(incoming.access_count),
        metadata,
        embedding: None,
        source: incoming.source.clone().or_else(|| existing.source.clone()),
        scope: existing.scope.clone(),
    }
}
/// Returns the larger of two importances, ignoring non-finite values.
fn max_importance(left: f32, right: f32) -> f32 {
    match (left.is_finite(), right.is_finite()) {
        (true, true) => left.max(right),
        (true, false) => left,
        (false, true) => right,
        (false, false) => 0.0,
    }
}
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::record::{MemoryScope, MemoryType};
    use serde_json::json;
    // ports tests/unit_tests/core/memory/manage/test_mem_update_checker.py (deterministic subset)
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
    fn normalize_text_is_shared_with_the_record_module() {
        assert_eq!(
            normalize_text("  Alice   PREFERS dark mode. "),
            "alice prefers dark mode"
        );
        let direct: fn(&str) -> String = crate::record::normalize_text;
        assert_eq!(normalize_text("Hello!"), direct("Hello!"));
    }
    #[test]
    fn detects_same_id_before_text_rules() {
        let incoming = record("000000000000000000000001", "completely different text");
        let candidates = vec![Candidate::new(
            record("000000000000000000000001", "other"),
            Some(0.4),
        )];
        let matched = find_duplicate(&incoming, &candidates, &DedupeConfig::default()).unwrap();
        assert_eq!(matched.kind, DuplicateKind::SameId);
        assert_eq!(matched.index, 0);
        assert!((matched.similarity - 0.4).abs() < f32::EPSILON);
        assert_eq!(DuplicateKind::SameId.as_str(), "same_id");
    }
    #[test]
    fn detects_exact_duplicates_after_normalization() {
        let incoming = record("a", "Alice prefers DARK mode.");
        let candidates = vec![Candidate::new(
            record("b", "alice prefers dark  mode"),
            None,
        )];
        let matched = find_duplicate(&incoming, &candidates, &DedupeConfig::default()).unwrap();
        assert_eq!(matched.kind, DuplicateKind::Exact);
        assert_eq!(matched.index, 0);
        assert!((matched.similarity - 1.0).abs() < f32::EPSILON);
        assert_eq!(DuplicateKind::Exact.as_str(), "exact");
    }
    #[test]
    fn exact_dedupe_applies_to_short_text() {
        let incoming = record("a", "tea");
        let candidates = vec![Candidate::new(record("b", " Tea "), None)];
        let matched = find_duplicate(&incoming, &candidates, &DedupeConfig::default()).unwrap();
        assert_eq!(matched.kind, DuplicateKind::Exact);
    }
    #[test]
    fn detects_similar_records_by_jaccard() {
        let incoming = record("a", "alice prefers dark mode for the editor");
        let candidates = vec![Candidate::new(
            record("b", "alice prefers dark mode in the editor"),
            None,
        )];
        let matched = find_duplicate(&incoming, &candidates, &DedupeConfig::default()).unwrap();
        assert_eq!(matched.kind, DuplicateKind::Similar);
        assert!(matched.similarity >= 0.75);
        assert_eq!(DuplicateKind::Similar.as_str(), "similar");
    }
    #[test]
    fn detects_similar_records_by_embedding_cosine() {
        let incoming = record("a", "alice prefers dark themes").with_embedding(vec![1.0, 0.0, 0.0]);
        let candidates = vec![Candidate::new(
            record("b", "totally unrelated words here").with_embedding(vec![1.0, 0.0, 0.01]),
            None,
        )];
        let matched = find_duplicate(&incoming, &candidates, &DedupeConfig::default()).unwrap();
        assert_eq!(matched.kind, DuplicateKind::Similar);
        assert!(matched.similarity >= 0.92);
    }
    #[test]
    fn unrelated_records_are_not_duplicates() {
        let incoming = record("a", "alice prefers dark mode");
        let candidates = vec![
            Candidate::new(record("b", "the deployment failed yesterday"), Some(0.2)),
            Candidate::new(record("c", "sqlite stores the memory rows"), None),
        ];
        assert!(find_duplicate(&incoming, &candidates, &DedupeConfig::default()).is_none());
        assert!(find_duplicate(&incoming, &[], &DedupeConfig::default()).is_none());
    }
    #[test]
    fn short_contents_skip_fuzzy_matching() {
        let incoming = record("a", "mode dark");
        let candidates = vec![Candidate::new(record("b", "dark mode"), None)];
        let strict = DedupeConfig {
            min_content_len: 10,
            ..Default::default()
        };
        assert!(find_duplicate(&incoming, &candidates, &strict).is_none());
        let relaxed = DedupeConfig {
            min_content_len: 4,
            ..Default::default()
        };
        let matched = find_duplicate(&incoming, &candidates, &relaxed).unwrap();
        assert_eq!(matched.kind, DuplicateKind::Similar);
        assert!((matched.similarity - 1.0).abs() < f32::EPSILON);
    }
    #[test]
    fn respects_candidate_top_k_and_prefers_the_highest_similarity() {
        let incoming = record("a", "alice prefers dark mode in the editor window");
        let candidates = vec![
            // Highest similarity, but too short for a fuzzy comparison.
            Candidate::new(record("one", "ok"), Some(0.95)),
            Candidate::new(
                record("two", "alice prefers dark mode in the editor window"),
                Some(0.10),
            ),
            Candidate::new(record("three", "unrelated too"), Some(0.05)),
        ];
        let config = DedupeConfig {
            candidate_top_k: 1,
            ..Default::default()
        };
        // Only the highest-similarity candidate is inspected, so the exact duplicate at
        // index 1 is out of reach.
        assert!(find_duplicate(&incoming, &candidates, &config).is_none());
        let config = DedupeConfig {
            candidate_top_k: 2,
            ..Default::default()
        };
        let matched = find_duplicate(&incoming, &candidates, &config).unwrap();
        assert_eq!(matched.kind, DuplicateKind::Exact);
        assert_eq!(matched.index, 1);
    }
    #[test]
    fn ties_resolve_by_highest_similarity_then_index() {
        let incoming = record("a", "alice prefers dark mode");
        let candidates = vec![
            Candidate::new(record("one", "alice prefers dark mode"), Some(0.2)),
            Candidate::new(record("two", "alice prefers dark mode."), Some(0.9)),
        ];
        let matched = find_duplicate(&incoming, &candidates, &DedupeConfig::default()).unwrap();
        assert_eq!(matched.index, 1);
        assert_eq!(matched.kind, DuplicateKind::Exact);
        // Equal similarities keep the caller's order.
        let candidates = vec![
            Candidate::new(record("one", "alice prefers dark mode"), Some(0.5)),
            Candidate::new(record("two", "alice prefers dark mode."), Some(0.5)),
        ];
        let matched = find_duplicate(&incoming, &candidates, &DedupeConfig::default()).unwrap();
        assert_eq!(matched.index, 0);
    }
    #[test]
    fn merge_keeps_identity_and_combines_content() {
        let existing = record("keep", "short")
            .with_importance(0.4)
            .with_metadata("a", json!(1))
            .with_source("conversation");
        let incoming = record("drop", "a much longer incoming memory text")
            .with_importance(0.9)
            .with_metadata("b", json!(2));
        let now = fixed_time() + chrono::Duration::seconds(60);
        let merged = merge_records(&existing, &incoming, now);
        assert_eq!(merged.id, "keep");
        assert_eq!(merged.scope, existing.scope);
        assert_eq!(merged.timestamp, existing.timestamp);
        assert_eq!(merged.content, "a much longer incoming memory text");
        assert!((merged.importance - 0.9).abs() < f32::EPSILON);
        assert_eq!(merged.metadata.get("a"), Some(&json!(1)));
        assert_eq!(merged.metadata.get("b"), Some(&json!(2)));
        assert_eq!(merged.source.as_deref(), Some("conversation"));
        assert_eq!(merged.updated_at, now);
        assert!(merged.embedding.is_none());
        assert_eq!(merged.kind, MemoryType::SemanticMemory);
        // Inputs untouched.
        assert_eq!(existing.content, "short");
    }
    #[test]
    fn merge_keeps_existing_content_on_equal_or_longer_length() {
        let existing = record("keep", "12345678");
        let incoming = record("drop", "abcdefgh");
        let merged = merge_records(&existing, &incoming, fixed_time());
        assert_eq!(merged.content, "12345678");
        let longer = record("drop", "abcdefghij");
        assert_eq!(
            merge_records(&existing, &longer, fixed_time()).content,
            "abcdefghij"
        );
    }
    #[test]
    fn merge_unions_metadata_with_incoming_winning_and_sums_access() {
        let mut existing = record("keep", "alpha").with_metadata("key", json!("existing"));
        existing.access_count = 2;
        let mut incoming = record("drop", "beta")
            .with_metadata("key", json!("incoming"))
            .with_metadata("only", json!(true));
        incoming.access_count = 3;
        incoming.last_access = Some(fixed_time() + chrono::Duration::seconds(10));
        let merged = merge_records(&existing, &incoming, fixed_time());
        assert_eq!(merged.metadata.get("key"), Some(&json!("incoming")));
        assert_eq!(merged.metadata.get("only"), Some(&json!(true)));
        assert_eq!(merged.access_count, 5);
        assert_eq!(
            merged.last_access,
            Some(fixed_time() + chrono::Duration::seconds(10))
        );
    }
    #[test]
    fn merge_prefers_incoming_source_and_non_unknown_kind() {
        let existing = record("keep", "alpha").with_source("conversation");
        let mut incoming = record("drop", "alpha and more");
        incoming.kind = MemoryType::UserProfile;
        let merged = merge_records(&existing, &incoming, fixed_time());
        assert_eq!(merged.kind, MemoryType::UserProfile);
        assert_eq!(merged.source.as_deref(), Some("conversation"));

        let existing = record("keep", "alpha").with_source("conversation");
        let mut incoming = record("drop", "alpha and more").with_source("import");
        incoming.kind = MemoryType::Unknown;
        let merged = merge_records(&existing, &incoming, fixed_time());
        assert_eq!(merged.kind, MemoryType::SemanticMemory);
        assert_eq!(merged.source.as_deref(), Some("import"));
    }
    #[test]
    fn merge_ignores_non_finite_importance() {
        let existing = record("keep", "alpha");
        let mut incoming = record("drop", "beta");
        incoming.importance = f32::NAN;
        let merged = merge_records(&existing, &incoming, fixed_time());
        assert!(merged.importance.is_finite());
        assert!((merged.importance - existing.importance).abs() < f32::EPSILON);
    }
    #[test]
    fn candidates_carry_optional_similarity() {
        let candidate = Candidate::new(record("a", "x"), None);
        assert_eq!(candidate.similarity_score(), 0.0);
        let candidate = Candidate::new(record("a", "x"), Some(f32::NAN));
        assert_eq!(candidate.similarity_score(), 0.0);
        let candidate = Candidate::new(record("a", "x"), Some(1.5));
        assert!((candidate.similarity_score() - 1.0).abs() < f32::EPSILON);
    }
}
