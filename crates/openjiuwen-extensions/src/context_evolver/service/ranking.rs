//! Model-free ranking helpers shared by every `context_evolver` service.
//!
//! Ports the retrieval half of Python's
//! `openjiuwen/extensions/context_evolver/retrieve/task/` and
//! `core/vector_store/memory_vector_store.py` down to the part that does not
//! need a model: a lexical overlap score and the deterministic ordering every
//! response card depends on.
//!
//! Python's retrieve flows are two or three LLM round-trips (classify, rerank,
//! rewrite) scored by vector similarity. This module keeps a pure lexical
//! fallback so a service can rank memories offline, and the embedder-backed
//! service blends this signal with cosine similarity
//! (see [`super::deterministic`]). Nothing here is async, I/O-bound, or
//! model-dependent, so it is also what the LLM-driven service uses to order the
//! candidates it shows the model.
//!
//! # Ranking contract
//!
//! * [`tokenize`] is the one tokenizer: lower-cased `[a-z0-9]` runs, so
//!   punctuation, separators, and non-ASCII text are boundaries and a matched
//!   token counts once.
//! * [`lexical_score`] is the cosine of the binary token bags,
//!   `matched / sqrt(|query| * |document|)`, which lies in `[0.0, 1.0]` and is
//!   symmetric in its arguments. An empty bag scores `0.0`, never `NaN`.
//! * [`rank_candidates`] sorts by descending score with an ascending-id
//!   tie-break, so equal scores are still deterministic.
//! * ACE is special: Python's ACE retrieval loads the whole playbook unranked
//!   (`retrieve/task/ace/run.py`), so [`rank_candidates`] keeps id order and
//!   scores every candidate `1.0`.
//!
//! Python counterparts: `tests/unit_tests/extensions/context_evolver/`
//! (`test_retrieve_flow.py` covers algorithm normalization; the live ranking
//! flows are skipped without an API key).
use std::collections::BTreeSet;

use super::super::prompt::RetrievedHit;
use super::super::{Algo, MemoryRecord};

/// The score ACE assigns to every playbook entry.
const ACE_SCORE: f32 = 1.0;

/// Splits `text` into lower-cased alphanumeric tokens.
///
/// Tokens are maximal `[a-z0-9]` runs after lower-casing ASCII text; every other
/// character (whitespace, punctuation, and any non-ASCII character) is a
/// separator. The result keeps duplicates, so callers that want a bag of
/// distinct tokens collect into a set first.
#[must_use]
pub fn tokenize(text: &str) -> Vec<String> {
    text.split(|character: char| !character.is_ascii_alphanumeric())
        .filter(|token| !token.is_empty())
        .map(str::to_ascii_lowercase)
        .collect()
}

/// Scores the lexical overlap of `query` and `text` in `[0.0, 1.0]`.
///
/// The score is the cosine of the two binary token bags,
/// `matched / sqrt(|query| * |document|)`: `1.0` when both sides carry exactly
/// the same tokens, `0.0` when they share none. Either side being token-free
/// (so the denominator would vanish) scores `0.0`, and the result is never
/// `NaN`; the function is symmetric, since both arguments go through the same
/// normalisation.
#[must_use]
// The matched count is bounded by the token count, which is far below `2^24`.
#[allow(clippy::cast_precision_loss)]
pub fn lexical_score(query: &str, text: &str) -> f32 {
    let query_tokens = distinct_tokens(query);
    let text_tokens = distinct_tokens(text);
    if query_tokens.is_empty() || text_tokens.is_empty() {
        return 0.0;
    }
    let matched = query_tokens.intersection(&text_tokens).count() as f32;
    let denominator = (query_tokens.len() as f32 * text_tokens.len() as f32).sqrt();
    let score = matched / denominator;
    if score.is_finite() {
        score
    } else {
        0.0
    }
}

/// Collects the distinct tokens of `text`.
fn distinct_tokens(text: &str) -> BTreeSet<String> {
    tokenize(text).into_iter().collect()
}

/// The text [`lexical_score`] and an embedder see for `record`.
///
/// Python embeds a memory's `title`, `description`, and `content` separately and
/// concatenates the vectors; this port joins the present fields into one text —
/// `title`, `when_to_use`, `content`, then `description`, each on its own line —
/// so one embedding call covers a record. Blank and absent fields are skipped,
/// and a record with no text at all renders as `""`.
#[must_use]
pub fn scoring_text(record: &MemoryRecord) -> String {
    let mut parts: Vec<&str> = Vec::with_capacity(4);
    for field in [
        record.title.as_deref(),
        record.when_to_use.as_deref(),
        Some(record.content.as_str()),
        record.description.as_deref(),
    ] {
        if let Some(value) = field.map(str::trim).filter(|value| !value.is_empty()) {
            parts.push(value);
        }
    }
    parts.join("\n")
}

/// Ranks `records` against `query`, highest score first.
///
/// Every record becomes a [`RetrievedHit`] carrying its own id:
///
/// * [`Algo::Ace`] keeps ascending-id order and scores every hit `1.0`
///   (`ACE_SCORE`), because Python's ACE retrieve step returns the whole
///   playbook unranked;
/// * every other algorithm scores the record with [`lexical_score`] over
///   [`scoring_text`], ordered by descending score and then ascending id, so
///   ties are deterministic.
///
/// `limit` caps the returned hits (`0` returns an empty [`Vec`]); the scores are
/// the raw lexical scores, so `1.0` means "same tokens as the query".
#[must_use]
pub fn rank_candidates(
    algo: Algo,
    query: &str,
    records: &[MemoryRecord],
    limit: usize,
) -> Vec<RetrievedHit> {
    if limit == 0 || records.is_empty() {
        return Vec::new();
    }
    let ace = algo == Algo::Ace;
    let mut hits: Vec<RetrievedHit> = records
        .iter()
        .map(|record| {
            let score = if ace {
                ACE_SCORE
            } else {
                lexical_score(query, &scoring_text(record))
            };
            RetrievedHit::new(record.id.clone(), score)
        })
        .collect();
    if ace {
        hits.sort_by(|left, right| left.id.cmp(&right.id));
    } else {
        hits.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| left.id.cmp(&right.id))
        });
    }
    hits.truncate(limit);
    hits
}

#[cfg(test)]
mod tests {
    // Python counterparts: `extensions/context_evolver/retrieve/task/*/run.py`
    // (ranking order) and `retrieve/task/ace/run.py` (unranked playbook load).
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::float_cmp)]
    use super::*;

    fn record(id: &str, content: &str) -> MemoryRecord {
        MemoryRecord {
            id: id.to_string(),
            user_id: "user".into(),
            content: content.to_string(),
            ..MemoryRecord::default()
        }
    }

    fn ids(hits: &[RetrievedHit]) -> Vec<&str> {
        hits.iter().map(|hit| hit.id.as_str()).collect()
    }

    #[test]
    fn tokenize_splits_on_non_alphanumerics_and_lowercases() {
        assert_eq!(
            tokenize("Prefer `rg` over grep!"),
            ["prefer", "rg", "over", "grep"]
        );
        assert_eq!(tokenize("a_b-c.d"), ["a", "b", "c", "d"]);
        assert_eq!(tokenize("MiXeD 42"), ["mixed", "42"]);
        assert_eq!(tokenize("  \t\n"), Vec::<String>::new());
        assert_eq!(tokenize(""), Vec::<String>::new());
        // Non-ASCII characters are separators, so `café` splits and drops the tail.
        assert_eq!(tokenize("café crème"), ["caf", "cr", "me"]);
    }

    #[test]
    fn lexical_score_is_symmetric_and_bounded() {
        let score = lexical_score("alpha beta", "alpha beta gamma");
        assert!((score - 2.0 / 6.0_f32.sqrt()).abs() < 1e-6, "{score}");
        assert!(score > 0.0 && score < 1.0);
        assert_eq!(lexical_score("alpha beta", "alpha beta"), 1.0);
        assert_eq!(lexical_score("Alpha, BETA!", "beta alpha"), 1.0);
        assert_eq!(lexical_score("alpha", "beta"), 0.0);

        for (left, right) in [
            ("alpha beta", "alpha beta gamma"),
            ("alpha beta gamma", "alpha beta"),
            ("rg grep", "prefer rg over grep"),
            ("unrelated", "alpha beta"),
        ] {
            let forward = lexical_score(left, right);
            let backward = lexical_score(right, left);
            assert!((forward - backward).abs() < 1e-6, "{left}/{right}");
            assert!((0.0..=1.0).contains(&forward), "{forward}");
            assert!(!forward.is_nan());
        }
    }

    #[test]
    fn lexical_score_is_zero_for_empty_inputs() {
        for (query, text) in [("", "alpha"), ("alpha", ""), ("", ""), ("  ", "!!")] {
            assert_eq!(lexical_score(query, text), 0.0, "{query:?}/{text:?}");
        }
    }

    #[test]
    fn lexical_score_counts_a_repeated_token_once() {
        assert_eq!(lexical_score("alpha", "alpha alpha alpha"), 1.0);
        assert_eq!(
            lexical_score("alpha alpha", "alpha"),
            lexical_score("alpha", "alpha")
        );
    }

    #[test]
    fn scoring_text_joins_present_fields() {
        let full = MemoryRecord {
            title: Some("Title".into()),
            when_to_use: Some("When".into()),
            content: "Body".into(),
            description: Some("Description".into()),
            ..record("m", "Body")
        };
        assert_eq!(
            scoring_text(&full),
            "Title\nWhen\nBody\nDescription".to_string()
        );

        let sparse = MemoryRecord {
            title: Some("   ".into()),
            when_to_use: None,
            description: Some("Description".into()),
            ..record("m", "Body")
        };
        assert_eq!(scoring_text(&sparse), "Body\nDescription");

        assert_eq!(scoring_text(&record("m", "   ")), "");
    }

    #[test]
    fn rank_candidates_orders_by_score_then_id() {
        let records = [
            record("c", "alpha"),
            record("a", "alpha beta"),
            record("b", "alpha"),
            record("d", "gamma"),
        ];
        let hits = rank_candidates(Algo::ReasoningBank, "alpha beta", &records, 10);
        assert_eq!(ids(&hits), ["a", "b", "c", "d"]);
        assert_eq!(hits[0].score, 1.0);
        assert_eq!(hits[1].score, hits[2].score);
        assert_eq!(hits[3].score, 0.0);
        assert!(hits[1].reason.is_none());
    }

    #[test]
    fn rank_candidates_keeps_ace_in_id_order() {
        let records = [
            record("c", "zeta"),
            record("a", "alpha"),
            record("b", "beta"),
        ];
        let hits = rank_candidates(Algo::Ace, "alpha", &records, 10);
        assert_eq!(ids(&hits), ["a", "b", "c"]);
        assert!(hits.iter().all(|hit| hit.score == ACE_SCORE));
    }

    #[test]
    fn rank_candidates_respects_limit() {
        let records = [
            record("a", "alpha"),
            record("b", "alpha"),
            record("c", "alpha"),
        ];
        assert_eq!(
            ids(&rank_candidates(Algo::Reme, "alpha", &records, 2)),
            ["a", "b"]
        );
        assert_eq!(
            rank_candidates(Algo::Reme, "alpha", &records, 0),
            Vec::new()
        );
        assert_eq!(rank_candidates(Algo::Ace, "alpha", &records, 0), Vec::new());
        assert_eq!(rank_candidates(Algo::Reme, "alpha", &[], 5), Vec::new());
    }
}
