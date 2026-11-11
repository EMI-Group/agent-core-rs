//! LLM-backed reranking over the [`LlmScorer`] seam.
//!
//! Python's reranker classes call a model directly; the Rust seam keeps this
//! crate free of any LLM dependency. An adapter that scores documents with an
//! `openjiuwen_foundation::Model` lives in an upper crate.

use std::sync::Arc;

use async_trait::async_trait;

use super::Reranker;
use crate::common::SearchResult;
use openjiuwen_common::error::{Error, Result};

/// Scores documents against a query for [`LlmReranker`].
///
/// Implementations return one relevance score per input document, in the same
/// order as `documents`; higher means more relevant. The implementation is
/// responsible for any prompt construction, batching, and model calls.
#[async_trait]
pub trait LlmScorer: Send + Sync {
    /// Scores each entry of `documents` against `query`.
    ///
    /// The returned vector must have exactly `documents.len()` finite scores.
    async fn score(&self, query: &str, documents: &[String]) -> Result<Vec<f32>>;
}

/// Reranks results with scores produced by an [`LlmScorer`].
///
/// The scorer is called once per rerank with the query and the texts of the
/// incoming documents. The returned scores replace the existing `score` field,
/// then results are sorted descending (stably) and truncated to the limit.
pub struct LlmReranker {
    /// Scorer producing relevance scores.
    scorer: Arc<dyn LlmScorer>,
    /// Default number of results to keep; `None` keeps every result.
    top_n: Option<usize>,
}

impl LlmReranker {
    /// Creates a reranker backed by `scorer`, keeping at most `top_n` results
    /// when a call does not supply its own limit.
    pub fn new(scorer: Arc<dyn LlmScorer>, top_n: Option<usize>) -> Self {
        Self { scorer, top_n }
    }

    /// The scorer used for reranking.
    pub fn scorer(&self) -> &Arc<dyn LlmScorer> {
        &self.scorer
    }

    /// The default truncation limit, if any.
    pub fn top_n(&self) -> Option<usize> {
        self.top_n
    }
}

#[async_trait]
impl Reranker for LlmReranker {
    async fn rerank(
        &self,
        query: &str,
        mut documents: Vec<SearchResult>,
        top_n: Option<usize>,
    ) -> Result<Vec<SearchResult>> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }

        let texts: Vec<String> = documents
            .iter()
            .map(|document| document.text.clone())
            .collect();
        let scores = self.scorer.score(query, &texts).await?;

        let expected = documents.len();
        let found = scores.len();
        if found != expected {
            return Err(Error::retrieval(format!(
                "reranker expected {expected} scores, found {found}"
            )));
        }
        if let Some((index, score)) = scores
            .iter()
            .enumerate()
            .find(|(_, score)| !score.is_finite())
        {
            return Err(Error::retrieval(format!(
                "reranker received non-finite score {score} at index {index}"
            )));
        }

        for (document, score) in documents.iter_mut().zip(scores) {
            document.score = score;
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
    use openjiuwen_common::error::ErrorKind;
    use parking_lot::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn hit(id: &str, text: &str, score: f32) -> SearchResult {
        SearchResult {
            id: id.to_owned(),
            text: text.to_owned(),
            score,
            metadata: Metadata::new(),
        }
    }

    fn ids(results: &[SearchResult]) -> Vec<&str> {
        results.iter().map(|result| result.id.as_str()).collect()
    }

    /// Fake scorer returning a fixed script and recording how it was called.
    struct FakeScorer {
        scores: Vec<f32>,
        calls: AtomicUsize,
        seen: Mutex<Option<(String, Vec<String>)>>,
    }

    impl FakeScorer {
        fn new(scores: Vec<f32>) -> Arc<Self> {
            Arc::new(Self {
                scores,
                calls: AtomicUsize::new(0),
                seen: Mutex::new(None),
            })
        }

        fn calls(&self) -> usize {
            self.calls.load(Ordering::SeqCst)
        }

        fn seen(&self) -> (String, Vec<String>) {
            self.seen
                .lock()
                .clone()
                .expect("scorer should have been called")
        }
    }

    #[async_trait]
    impl LlmScorer for FakeScorer {
        async fn score(&self, query: &str, documents: &[String]) -> Result<Vec<f32>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            *self.seen.lock() = Some((query.to_owned(), documents.to_vec()));
            Ok(self.scores.clone())
        }
    }

    /// Scorer that always fails, to verify error propagation.
    struct FailingScorer;

    #[async_trait]
    impl LlmScorer for FailingScorer {
        async fn score(&self, _query: &str, _documents: &[String]) -> Result<Vec<f32>> {
            Err(Error::transport("scorer unavailable"))
        }
    }

    #[tokio::test]
    async fn scorer_receives_exact_query_and_document_texts() {
        let scorer = FakeScorer::new(vec![0.1, 0.2]);
        let reranker = LlmReranker::new(scorer.clone(), None);
        let results = reranker
            .rerank(
                "what is rust",
                vec![hit("a", "alpha", 0.0), hit("b", "beta", 0.0)],
                None,
            )
            .await
            .expect("rerank should succeed");

        assert_eq!(scorer.calls(), 1);
        let (query, documents) = scorer.seen();
        assert_eq!(query, "what is rust");
        assert_eq!(documents, vec!["alpha".to_owned(), "beta".to_owned()]);
        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn scorer_scores_replace_existing_scores() {
        let scorer = FakeScorer::new(vec![0.25, 0.75]);
        let reranker = LlmReranker::new(scorer, None);
        let results = reranker
            .rerank(
                "query",
                vec![hit("a", "alpha", 0.9), hit("b", "beta", 0.1)],
                None,
            )
            .await
            .expect("rerank should succeed");

        assert_eq!(results[0].id, "b");
        assert_eq!(results[0].score, 0.75);
        assert_eq!(results[1].id, "a");
        assert_eq!(results[1].score, 0.25);
    }

    #[tokio::test]
    async fn results_are_reordered_by_new_scores() {
        let scorer = FakeScorer::new(vec![0.2, 0.9, 0.5]);
        let reranker = LlmReranker::new(scorer, None);
        let results = reranker
            .rerank(
                "query",
                vec![
                    hit("a", "alpha", 0.9),
                    hit("b", "beta", 0.8),
                    hit("c", "gamma", 0.7),
                ],
                None,
            )
            .await
            .expect("rerank should succeed");

        assert_eq!(ids(&results), vec!["b", "c", "a"]);
    }

    #[tokio::test]
    async fn equal_scores_keep_input_order() {
        let scorer = FakeScorer::new(vec![0.5, 0.5, 0.5]);
        let reranker = LlmReranker::new(scorer, None);
        let results = reranker
            .rerank(
                "query",
                vec![
                    hit("first", "one", 0.1),
                    hit("second", "two", 0.2),
                    hit("third", "three", 0.3),
                ],
                None,
            )
            .await
            .expect("rerank should succeed");

        assert_eq!(ids(&results), vec!["first", "second", "third"]);
    }

    #[tokio::test]
    async fn score_count_mismatch_is_a_retrieval_error() {
        let scorer = FakeScorer::new(vec![0.5]);
        let reranker = LlmReranker::new(scorer, None);
        let error = reranker
            .rerank(
                "query",
                vec![hit("a", "alpha", 0.0), hit("b", "beta", 0.0)],
                None,
            )
            .await
            .expect_err("mismatched scores must fail");

        assert_eq!(error.kind(), ErrorKind::Retrieval);
        assert!(
            error.message().contains("expected 2 scores, found 1"),
            "unexpected message: {}",
            error.message()
        );
    }

    #[tokio::test]
    async fn non_finite_scores_are_rejected() {
        for bad_score in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let reranker = LlmReranker::new(FakeScorer::new(vec![0.5, bad_score]), None);
            let error = reranker
                .rerank(
                    "query",
                    vec![hit("a", "alpha", 0.0), hit("b", "beta", 0.0)],
                    None,
                )
                .await
                .expect_err("non-finite score must fail");

            assert_eq!(error.kind(), ErrorKind::Retrieval);
            assert!(
                error.message().contains("non-finite score"),
                "unexpected message for {bad_score}: {}",
                error.message()
            );
        }
    }

    #[tokio::test]
    async fn rerank_truncates_to_top_n() {
        let reranker = LlmReranker::new(FakeScorer::new(vec![0.1, 0.9, 0.5]), None);
        let results = reranker
            .rerank(
                "query",
                vec![
                    hit("a", "alpha", 0.0),
                    hit("b", "beta", 0.0),
                    hit("c", "gamma", 0.0),
                ],
                Some(2),
            )
            .await
            .expect("rerank should succeed");

        assert_eq!(ids(&results), vec!["b", "c"]);
    }

    #[tokio::test]
    async fn rerank_with_zero_top_n_returns_empty() {
        let reranker = LlmReranker::new(FakeScorer::new(vec![0.4]), None);
        let results = reranker
            .rerank("query", vec![hit("a", "alpha", 0.0)], Some(0))
            .await
            .expect("rerank should succeed");

        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn configured_limit_applies_when_call_omits_one() {
        let reranker = LlmReranker::new(FakeScorer::new(vec![0.1, 0.9, 0.5]), Some(2));
        assert_eq!(reranker.top_n(), Some(2));
        let results = reranker
            .rerank(
                "query",
                vec![
                    hit("a", "alpha", 0.0),
                    hit("b", "beta", 0.0),
                    hit("c", "gamma", 0.0),
                ],
                None,
            )
            .await
            .expect("rerank should succeed");

        assert_eq!(ids(&results), vec!["b", "c"]);
    }

    #[tokio::test]
    async fn empty_documents_never_call_the_scorer() {
        let scorer = FakeScorer::new(vec![]);
        let reranker = LlmReranker::new(scorer.clone(), Some(5));
        let results = reranker
            .rerank("query", Vec::new(), Some(3))
            .await
            .expect("rerank should succeed");

        assert!(results.is_empty());
        assert_eq!(scorer.calls(), 0);
        assert!(scorer.seen.lock().is_none());
    }

    #[tokio::test]
    async fn scorer_errors_propagate_unchanged() {
        let reranker = LlmReranker::new(Arc::new(FailingScorer), None);
        let error = reranker
            .rerank("query", vec![hit("a", "alpha", 0.0)], None)
            .await
            .expect_err("scorer failure must propagate");

        assert_eq!(error.kind(), ErrorKind::Transport);
    }
}
