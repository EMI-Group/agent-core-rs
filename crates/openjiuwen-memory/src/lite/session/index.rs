//! The recall index over decoded session records: the chunk/hit models, the keyword
//! score, and deterministic ranking.
//!
//! Scoring is keyword-only and pure: the query is tokenized once, each token either occurs
//! in the record text or not, and the score is the covered fraction of query tokens. No
//! embeddings are consulted, so the index works offline and produces the same order on
//! every run.
use super::record::SessionRecord;
use super::TARGET;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::sync::LazyLock;
/// The maximum number of query tokens [`keyword_score`] considers.
const MAX_QUERY_TOKENS: usize = 10;
/// The query tokenizer: every Unicode word run is one token.
static TOKENS: LazyLock<Option<Regex>> = LazyLock::new(|| match Regex::new(r"\w+") {
    Ok(regex) => Some(regex),
    Err(error) => {
        tracing::warn!(
            target: TARGET,
            %error,
            "the session query tokenizer failed to compile; keyword scoring returns 0.0"
        );
        None
    }
});
/// One decoded record together with its source location.
///
/// `path` is relative to the store root with `/` separators (for example
/// `sessions/2026-11-14/abc.jsonl`), and `line` is the 1-based line number in that file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionChunk {
    /// Source file, relative to the store root and separated by `/`.
    pub path: String,
    /// 1-based line number of the record inside `path`.
    pub line: usize,
    /// The decoded record.
    pub record: SessionRecord,
}
/// One scored chunk of a session recall.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionHit {
    /// Source file, relative to the store root and separated by `/`.
    pub path: String,
    /// 1-based line number of the record inside `path`.
    pub line: usize,
    /// The ranking score in `0.0..=1.0`; keyword-only, so it equals [`SessionHit::keyword`].
    pub score: f32,
    /// The embedding similarity component; always `0.0` for the keyword-only session index.
    pub similarity: f32,
    /// The keyword component in `0.0..=1.0`.
    pub keyword: f32,
    /// The scored record.
    pub record: SessionRecord,
}
/// Scores `text` against `query` by keyword coverage, in `0.0..=1.0`.
///
/// The query is tokenized with `\w+` (lowercased, at most ten tokens, each kept as-is
/// including single characters), and a token counts when it occurs as a substring of the
/// lowercased `text`. The score is `matched / total`; a query without tokens scores `0.0`.
#[must_use]
pub fn keyword_score(query: &str, text: &str) -> f32 {
    let tokens = query_tokens(query);
    if tokens.is_empty() {
        return 0.0;
    }
    let haystack = text.to_lowercase();
    let matched = tokens
        .iter()
        .filter(|token| haystack.contains(token.as_str()))
        .count();
    matched as f32 / tokens.len() as f32
}
/// Ranks `chunks` against `query`, keyword-only.
///
/// Each chunk is scored on [`SessionRecord::text`] (role and content), so a query token may
/// match either. Hits keep a score of `>= threshold`, are ordered by score descending, then
/// path ascending, then line ascending, and the result is truncated to `limit.max(1)`.
#[must_use]
pub fn rank_chunks(
    chunks: &[SessionChunk],
    query: &str,
    threshold: f32,
    limit: usize,
) -> Vec<SessionHit> {
    let mut hits: Vec<SessionHit> = chunks
        .iter()
        .map(|chunk| {
            let score = keyword_score(query, &chunk.record.text());
            SessionHit {
                path: chunk.path.clone(),
                line: chunk.line,
                score,
                similarity: 0.0,
                keyword: score,
                record: chunk.record.clone(),
            }
        })
        .filter(|hit| hit.score >= threshold)
        .collect();
    hits.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.path.cmp(&right.path))
            .then_with(|| left.line.cmp(&right.line))
    });
    hits.truncate(limit.max(1));
    hits
}
/// Tokenizes `query` with `\w+`, lowercased and capped at `MAX_QUERY_TOKENS`.
fn query_tokens(query: &str) -> Vec<String> {
    let Some(regex) = TOKENS.as_ref() else {
        return Vec::new();
    };
    regex
        .find_iter(query)
        .map(|found| found.as_str().to_lowercase())
        .take(MAX_QUERY_TOKENS)
        .collect()
}
#[cfg(test)]
mod tests {
    use super::*;
    /// Builds a chunk at `path` line `line` with `content`.
    fn chunk(path: &str, line: usize, content: &str) -> SessionChunk {
        SessionChunk {
            path: path.to_owned(),
            line,
            record: SessionRecord::new(content),
        }
    }
    #[test]
    fn tokenizes_words_case_insensitively() {
        assert_eq!(query_tokens("Dark MODE"), vec!["dark", "mode"]);
        assert_eq!(query_tokens("a-b_c.1"), vec!["a", "b_c", "1"]);
        assert_eq!(query_tokens("!!! ---"), Vec::<String>::new());
        assert_eq!(query_tokens(""), Vec::<String>::new());
    }
    #[test]
    fn caps_the_query_tokens_at_ten() {
        let query = (1..=12)
            .map(|index| format!("t{index}"))
            .collect::<Vec<_>>()
            .join(" ");
        let tokens = query_tokens(&query);
        assert_eq!(tokens.len(), MAX_QUERY_TOKENS);
        assert_eq!(tokens.last().map(String::as_str), Some("t10"));
    }
    #[test]
    fn scores_keyword_coverage() {
        assert!((keyword_score("dark mode", "the Dark Mode theme") - 1.0).abs() < f32::EPSILON);
        assert!((keyword_score("dark light", "dark mode") - 0.5).abs() < f32::EPSILON);
        assert!((keyword_score("missing", "dark mode") - 0.0).abs() < f32::EPSILON);
        assert!((keyword_score("", "dark mode") - 0.0).abs() < f32::EPSILON);
        assert!((keyword_score("dark", "") - 0.0).abs() < f32::EPSILON);
    }
    #[test]
    fn matches_tokens_as_substrings() {
        assert!((keyword_score("migr", "we migrated") - 1.0).abs() < f32::EPSILON);
        assert!((keyword_score("mode", "modes") - 1.0).abs() < f32::EPSILON);
    }
    #[test]
    fn ranks_by_score_then_path_then_line() {
        let chunks = [
            chunk("sessions/b.jsonl", 1, "alpha beta"),
            chunk("sessions/a.jsonl", 2, "alpha beta"),
            chunk("sessions/a.jsonl", 1, "alpha beta"),
            chunk("sessions/c.jsonl", 1, "alpha"),
            chunk("sessions/d.jsonl", 1, "nothing"),
        ];
        let hits = rank_chunks(&chunks, "alpha beta", 0.1, 10);
        assert_eq!(hits.len(), 4);
        assert_eq!(
            hits.iter()
                .map(|hit| (hit.path.as_str(), hit.line))
                .collect::<Vec<_>>(),
            vec![
                ("sessions/a.jsonl", 1),
                ("sessions/a.jsonl", 2),
                ("sessions/b.jsonl", 1),
                ("sessions/c.jsonl", 1),
            ]
        );
        assert!((hits[0].score - 1.0).abs() < f32::EPSILON);
        assert!((hits[3].score - 0.5).abs() < f32::EPSILON);
        assert!((hits[3].keyword - 0.5).abs() < f32::EPSILON);
        assert!(hits.iter().all(|hit| hit.similarity.abs() < f32::EPSILON));
    }
    #[test]
    fn keeps_zero_scoring_chunks_at_a_zero_threshold() {
        let chunks = [
            chunk("sessions/a.jsonl", 1, "alpha"),
            chunk("sessions/b.jsonl", 1, "nothing"),
        ];
        let hits = rank_chunks(&chunks, "alpha", 0.0, 10);
        assert_eq!(
            hits.iter()
                .map(|hit| (hit.path.as_str(), hit.score))
                .collect::<Vec<_>>(),
            vec![("sessions/a.jsonl", 1.0), ("sessions/b.jsonl", 0.0)]
        );
    }
    #[test]
    fn filters_below_the_threshold() {
        let chunks = [
            chunk("sessions/a.jsonl", 1, "alpha"),
            chunk("sessions/b.jsonl", 1, "alpha beta"),
        ];
        let hits = rank_chunks(&chunks, "alpha beta", 0.6, 10);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].path, "sessions/b.jsonl");
        assert_eq!(rank_chunks(&chunks, "alpha beta", 1.0, 10).len(), 1);
        assert_eq!(
            rank_chunks(&chunks, "gamma", 0.1, 10),
            Vec::<SessionHit>::new()
        );
    }
    #[test]
    fn truncates_to_at_least_one_hit() {
        let chunks = [
            chunk("sessions/a.jsonl", 1, "alpha"),
            chunk("sessions/b.jsonl", 1, "alpha"),
        ];
        assert_eq!(rank_chunks(&chunks, "alpha", 0.0, 0).len(), 1);
        assert_eq!(rank_chunks(&chunks, "alpha", 0.0, 1).len(), 1);
        assert_eq!(rank_chunks(&chunks, "alpha", 0.0, 2).len(), 2);
        assert_eq!(rank_chunks(&chunks, "", 0.0, 2).len(), 2);
    }
    #[test]
    fn scores_the_record_text_including_the_role() {
        let chunks = [SessionChunk {
            path: "sessions/a.jsonl".to_owned(),
            line: 1,
            record: SessionRecord::new("deploy").with_role("assistant"),
        }];
        assert_eq!(rank_chunks(&chunks, "assistant", 1.0, 5).len(), 1);
        assert_eq!(
            rank_chunks(&chunks, "user", 0.1, 5),
            Vec::<SessionHit>::new()
        );
    }
    #[test]
    fn an_empty_chunk_set_ranks_to_nothing() {
        assert_eq!(
            rank_chunks(&[], "anything", 0.0, 10),
            Vec::<SessionHit>::new()
        );
    }
}
