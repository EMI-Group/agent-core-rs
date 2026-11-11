//! Deterministic score-based reranking.
//!
//! Python ships no default or no-op reranker; [`ScoreReranker`] is a Rust
//! addition that lets the knowledge-base pipeline rerank offline without an LLM
//! by reusing the relevance scores produced by the vector store.

use async_trait::async_trait;

use super::Reranker;
use crate::common::SearchResult;
use openjiuwen_common::error::Result;

/// Reranks results by their existing relevance score.
///
/// The sort is stable and descending, so results with equal scores keep the
/// order they arrived in. An optional `top_n` configured on the reranker acts as
/// the default limit; an explicit per-call `top_n` overrides it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ScoreReranker {
    /// Default number of results to keep; `None` keeps every result.
    top_n: Option<usize>,
}

impl ScoreReranker {
    /// Creates a reranker that keeps at most `top_n` results when a call does
    /// not supply its own limit.
    pub fn new(top_n: Option<usize>) -> Self {
        Self { top_n }
    }

    /// The default truncation limit, if any.
    pub fn top_n(&self) -> Option<usize> {
        self.top_n
    }
}

#[async_trait]
impl Reranker for ScoreReranker {
    async fn rerank(
        &self,
        _query: &str,
        mut documents: Vec<SearchResult>,
        top_n: Option<usize>,
    ) -> Result<Vec<SearchResult>> {
        if documents.is_empty() {
            return Ok(documents);
        }
        super::sort_and_truncate(&mut documents, top_n.or(self.top_n));
        Ok(documents)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]
mod tests {
    use super::*;
    use crate::common::Metadata;

    fn hit(id: &str, score: f32) -> SearchResult {
        SearchResult {
            id: id.to_owned(),
            text: format!("text {id}"),
            score,
            metadata: Metadata::new(),
        }
    }

    fn ids(results: &[SearchResult]) -> Vec<&str> {
        results.iter().map(|result| result.id.as_str()).collect()
    }

    #[tokio::test]
    async fn rerank_sorts_descending_by_score() {
        let reranker = ScoreReranker::default();
        let results = reranker
            .rerank(
                "query",
                vec![hit("low", 0.2), hit("high", 0.9), hit("mid", 0.5)],
                None,
            )
            .await
            .expect("rerank should succeed");
        assert_eq!(ids(&results), vec!["high", "mid", "low"]);
    }

    #[tokio::test]
    async fn rerank_keeps_input_order_for_equal_scores() {
        let reranker = ScoreReranker::default();
        let results = reranker
            .rerank(
                "query",
                vec![hit("first", 0.5), hit("second", 0.5), hit("third", 0.5)],
                None,
            )
            .await
            .expect("rerank should succeed");
        assert_eq!(ids(&results), vec!["first", "second", "third"]);
    }

    #[tokio::test]
    async fn rerank_truncates_to_call_limit() {
        let reranker = ScoreReranker::default();
        let results = reranker
            .rerank(
                "query",
                vec![hit("a", 0.1), hit("b", 0.9), hit("c", 0.5)],
                Some(2),
            )
            .await
            .expect("rerank should succeed");
        assert_eq!(ids(&results), vec!["b", "c"]);
    }

    #[tokio::test]
    async fn rerank_with_zero_limit_returns_empty() {
        let reranker = ScoreReranker::default();
        let results = reranker
            .rerank("query", vec![hit("a", 0.9)], Some(0))
            .await
            .expect("rerank should succeed");
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn rerank_without_limit_keeps_every_result() {
        let reranker = ScoreReranker::default();
        let results = reranker
            .rerank(
                "query",
                vec![hit("a", 0.4), hit("b", 0.3), hit("c", 0.2), hit("d", 0.1)],
                None,
            )
            .await
            .expect("rerank should succeed");
        assert_eq!(ids(&results), vec!["a", "b", "c", "d"]);
    }

    #[tokio::test]
    async fn rerank_returns_empty_input_unchanged() {
        let reranker = ScoreReranker::new(Some(3));
        let results = reranker
            .rerank("query", Vec::new(), Some(1))
            .await
            .expect("rerank should succeed");
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn configured_limit_applies_when_call_omits_one() {
        let reranker = ScoreReranker::new(Some(2));
        assert_eq!(reranker.top_n(), Some(2));
        let results = reranker
            .rerank(
                "query",
                vec![hit("a", 0.1), hit("b", 0.9), hit("c", 0.5)],
                None,
            )
            .await
            .expect("rerank should succeed");
        assert_eq!(ids(&results), vec!["b", "c"]);
    }

    #[tokio::test]
    async fn call_limit_overrides_configured_limit() {
        let reranker = ScoreReranker::new(Some(1));
        let results = reranker
            .rerank(
                "query",
                vec![hit("a", 0.1), hit("b", 0.9), hit("c", 0.5)],
                Some(2),
            )
            .await
            .expect("rerank should succeed");
        assert_eq!(ids(&results), vec!["b", "c"]);
    }

    #[tokio::test]
    async fn rerank_preserves_text_and_metadata() {
        let mut metadata = Metadata::new();
        metadata.insert("doc_id".to_owned(), serde_json::json!("d1"));
        let input = SearchResult {
            id: "c1".to_owned(),
            text: "hello".to_owned(),
            score: 0.25,
            metadata: metadata.clone(),
        };
        let reranker = ScoreReranker::default();
        let results = reranker
            .rerank("query", vec![input.clone()], None)
            .await
            .expect("rerank should succeed");
        assert_eq!(results, vec![input]);
        assert_eq!(results[0].metadata, metadata);
    }
}
