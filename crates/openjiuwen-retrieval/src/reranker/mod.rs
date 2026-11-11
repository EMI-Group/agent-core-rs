//! Retrieval result reranking.
//!
//! Port of the Python `openjiuwen/core/retrieval/reranker` ABC, adapted to Rust
//! idioms: Python's `rerank` returns a `dict[str, float]` of scores, while
//! [`Reranker`] returns the reordered [`SearchResult`]s so callers never have to
//! rejoin ids with scores.
//!
//! Two implementations ship with the crate:
//! - [`ScoreReranker`] — deterministic default that reuses the scores produced
//!   by the vector store (Python has no equivalent no-op reranker).
//! - [`LlmReranker`] — adapter over the [`LlmScorer`] seam; a `Model`-backed
//!   scorer lives in an upper crate so this crate never depends upward.

mod llm;
mod score;

pub use llm::{LlmReranker, LlmScorer};
pub use score::ScoreReranker;

use async_trait::async_trait;

use crate::common::SearchResult;
use openjiuwen_common::error::Result;

/// Reorders retrieved results by relevance and optionally truncates them.
///
/// Implementations are stateless with respect to a single call: the per-call
/// `top_n` argument is the truncation limit for that call and takes precedence
/// over any limit configured on the implementation. `None` means "do not
/// truncate here"; the caller may still truncate afterwards.
#[async_trait]
pub trait Reranker: Send + Sync {
    /// Reranks `documents` for `query`, returning at most `top_n` results.
    ///
    /// The order of the returned results reflects their relevance to `query`;
    /// implementations must be deterministic for identical inputs.
    async fn rerank(
        &self,
        query: &str,
        documents: Vec<SearchResult>,
        top_n: Option<usize>,
    ) -> Result<Vec<SearchResult>>;
}

/// Sorts `results` by descending score (stable) and applies the optional limit.
///
/// The comparison is a total order (`f32::total_cmp`), so non-finite scores
/// sort deterministically instead of producing a partial-order result. Equal
/// scores keep their input order because the sort is stable.
fn sort_and_truncate(results: &mut Vec<SearchResult>, top_n: Option<usize>) {
    results.sort_by(|left, right| right.score.total_cmp(&left.score));
    if let Some(limit) = top_n {
        results.truncate(limit);
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]
mod tests {
    use super::*;

    fn hit(id: &str, score: f32) -> SearchResult {
        SearchResult {
            id: id.to_owned(),
            text: format!("text {id}"),
            score,
            metadata: crate::common::Metadata::new(),
        }
    }

    #[test]
    fn sort_and_truncate_orders_descending_and_keeps_ties_stable() {
        let mut results = vec![hit("a", 0.1), hit("b", 0.9), hit("c", 0.9), hit("d", 0.5)];
        sort_and_truncate(&mut results, None);
        let ids: Vec<&str> = results.iter().map(|result| result.id.as_str()).collect();
        assert_eq!(ids, vec!["b", "c", "d", "a"]);
    }

    #[test]
    fn sort_and_truncate_applies_limit() {
        let mut results = vec![hit("a", 0.1), hit("b", 0.9), hit("c", 0.5)];
        sort_and_truncate(&mut results, Some(2));
        let ids: Vec<&str> = results.iter().map(|result| result.id.as_str()).collect();
        assert_eq!(ids, vec!["b", "c"]);
    }

    #[test]
    fn sort_and_truncate_with_zero_limit_returns_empty() {
        let mut results = vec![hit("a", 0.1)];
        sort_and_truncate(&mut results, Some(0));
        assert!(results.is_empty());
    }
}
