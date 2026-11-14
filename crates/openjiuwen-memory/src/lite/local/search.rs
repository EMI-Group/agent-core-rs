//! Recall for the local store: the deterministic keyword scan, the optional FTS5 merge,
//! the optional chunked semantic signal, and the scoring/ranking helpers behind
//! [`LocalCodingMemory::search`].
//!
//! # Scoring
//!
//! Two signals exist per fact. The keyword score is the documented field-weighted token
//! match (see [`crate::lite::local`]), merged with the FTS5 hit when the `sqlite` feature
//! is enabled; the semantic score is the maximum cosine similarity over the fact's
//! chunks, normalized to `0.0..=1.0` by
//! [`similarity_from_cosine`](crate::lite::embed::similarity_from_cosine) and `0.0` for a
//! fact without an indexed vector.
//!
//! - [`SearchMode::Keyword`] scores every fact with the keyword signal alone. It never
//!   calls the embedding model, not even to embed the query.
//! - [`SearchMode::Semantic`] scores every fact with the semantic signal alone, and
//!   returns [`Error::Unsupported`] naming the fix when no embedding model is configured.
//! - [`SearchMode::Hybrid`] blends the two under the weights below. Without an embedding
//!   model it degrades to the keyword signal, which is the store's documented
//!   keyword-only behaviour.
//!
//! The blend is
//! [`blend`](crate::lite::embed::blend)`(similarity, keyword, `[`VECTOR_WEIGHT`]`,
//! `[`TEXT_WEIGHT`]`)`, i.e. `0.7 * semantic + 0.3 * keyword` for scores already in
//! `0.0..=1.0`.
//!
//! Every mode applies [`SearchOptions::threshold`](crate::record::SearchOptions::threshold)
//! to the final score, orders the hits by score descending and then by name ascending,
//! honours [`SearchOptions::top_k`](crate::record::SearchOptions::top_k), and fills the
//! [`RecallHit`] components: `similarity` is the semantic component, `keyword` is the
//! keyword component, and `score` is the mode's final value.
//!
//! Only `top_k`, `threshold`, `kinds`, `mode`, and the call's cancellation state are
//! honoured; `token_budget` is ignored because there is no context window here.
use std::collections::BTreeMap;
use std::sync::OnceLock;

use chrono::Utc;
use regex::Regex;

use openjiuwen_common::error::{Error, Result};

use crate::lite::embed::blend;
use crate::lite::fact::CodingFact;
use crate::lite::markdown::MarkdownMemory;
use crate::record::{MemoryType, RecallHit, SearchMode, SearchOptions};

use super::state::LocalCodingMemory;

/// Maximum number of query tokens the keyword scan considers.
const MAX_QUERY_TOKENS: usize = 10;

/// Half-life of the recency component, in seconds (30 days).
const RECENCY_HALF_LIFE_SECS: f32 = 2_592_000.0;

/// Weight of the semantic component of a [`SearchMode::Hybrid`] score (`0.7`).
const VECTOR_WEIGHT: f32 = 0.7;

/// Weight of the keyword component of a [`SearchMode::Hybrid`] score (`0.3`).
const TEXT_WEIGHT: f32 = 0.3;

/// The reason reported for [`SearchMode::Semantic`] without an embedding model.
const NO_EMBEDDING_REASON: &str = "the local coding-memory store has no embedding index; \
     configure one with `LocalCodingMemory::builder(..).embedding(..)` or use \
     keyword/hybrid search";

impl LocalCodingMemory {
    /// Recalls the facts matching `query`.
    ///
    /// Hits are ordered by descending score and then by name ascending; see the module
    /// docs for the scoring rules of each mode. `kinds` filters the fact types (an empty
    /// list means "every kind", and a list without [`MemoryType::Coding`] yields no hit).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for a blank query, [`Error::Unsupported`] for
    /// semantic mode on a store without an embedding model, the call's cancellation
    /// error, and the filesystem, index, or embedding error when the search fails.
    pub async fn search(&self, query: &str, options: &SearchOptions) -> Result<Vec<RecallHit>> {
        let cleaned = query.trim();
        if cleaned.is_empty() {
            return Err(Error::validation(
                "search query must not be empty (got \"\")",
            ));
        }
        let use_semantic = match options.mode {
            SearchMode::Keyword => false,
            SearchMode::Semantic => {
                if !self.has_embedding() {
                    return Err(Error::unsupported(NO_EMBEDDING_REASON));
                }
                true
            }
            SearchMode::Hybrid => self.has_embedding(),
        };
        if !options.kinds.is_empty() && !options.kinds.contains(&MemoryType::Coding) {
            return Ok(Vec::new());
        }
        options.call.check()?;
        let limit = options.top_k.max(1);
        let mut state = self.inner.state.lock().await;
        self.ensure_fresh(&mut state).await?;
        let facts = state.document.facts();
        drop(state);

        #[cfg_attr(not(feature = "sqlite"), allow(unused_mut))]
        let mut keyword = keyword_scores(&facts, cleaned);
        #[cfg(feature = "sqlite")]
        {
            let candidates = self.fts.search(cleaned, limit.saturating_mul(4)).await?;
            for candidate in candidates {
                let score = 1.0 - candidate.score;
                let entry = keyword
                    .iter_mut()
                    .find(|(name, _score)| *name == candidate.name);
                match entry {
                    Some((_name, score_slot)) => {
                        if score > *score_slot {
                            *score_slot = score;
                        }
                    }
                    None => keyword.push((candidate.name, score)),
                }
            }
        }
        let semantic = if use_semantic {
            self.semantic_scores(cleaned, limit.saturating_mul(4))
                .await?
        } else {
            BTreeMap::new()
        };
        let mut ranked: Vec<(String, RecallHit)> = facts
            .iter()
            .filter_map(|fact| {
                let keyword_score = keyword
                    .iter()
                    .find(|(name, _score)| *name == fact.name)
                    .map(|(_name, score)| score.clamp(0.0, 1.0));
                let similarity = semantic.get(&fact.name).copied().unwrap_or(0.0);
                let (score, keyword_component) = if !use_semantic {
                    let score = keyword_score?;
                    (score, score)
                } else if options.mode == SearchMode::Semantic {
                    (similarity, keyword_score.unwrap_or(0.0))
                } else {
                    let keyword_component = keyword_score.unwrap_or(0.0);
                    (
                        blend(similarity, keyword_component, VECTOR_WEIGHT, TEXT_WEIGHT),
                        keyword_component,
                    )
                };
                (score >= options.threshold).then(|| {
                    (
                        fact.name.clone(),
                        recall_hit(fact, score, similarity, keyword_component),
                    )
                })
            })
            .collect();
        ranked.sort_by(|(left_name, left), (right_name, right)| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left_name.cmp(right_name))
        });
        ranked.truncate(limit);
        Ok(ranked.into_iter().map(|(_name, hit)| hit).collect())
    }
}

/// Returns the facts ordered newest first and then by name.
pub(super) fn sorted_facts(document: &MarkdownMemory) -> Vec<CodingFact> {
    let mut facts = document.facts();
    facts.sort_by(|left, right| {
        right
            .updated_at
            .cmp(&left.updated_at)
            .then_with(|| left.name.cmp(&right.name))
    });
    facts
}

/// Splits a query into lowercased word tokens (at most [`MAX_QUERY_TOKENS`]).
fn query_tokens(query: &str) -> Vec<String> {
    static TOKENS: OnceLock<Option<Regex>> = OnceLock::new();
    let Some(regex) = TOKENS.get_or_init(|| Regex::new(r"\w+").ok()) else {
        return Vec::new();
    };
    let mut tokens: Vec<String> = Vec::new();
    for token in regex.find_iter(query) {
        let token = token.as_str().to_lowercase();
        if !tokens.contains(&token) {
            tokens.push(token);
        }
        if tokens.len() == MAX_QUERY_TOKENS {
            break;
        }
    }
    tokens
}

/// Scores every fact that matches at least one query token.
fn keyword_scores(facts: &[CodingFact], query: &str) -> Vec<(String, f32)> {
    let tokens = query_tokens(query);
    if tokens.is_empty() {
        return Vec::new();
    }
    let divisor = 3.0 * tokens.len() as f32;
    let mut scores = Vec::new();
    for fact in facts {
        let name = fact.name.to_lowercase();
        let description = fact.description.to_lowercase();
        let body = fact.content.to_lowercase();
        let tags: Vec<String> = fact.tags.iter().map(|tag| tag.to_lowercase()).collect();
        let mut matched = 0.0_f32;
        for token in &tokens {
            let mut weight = 0.0_f32;
            if name.contains(token.as_str()) {
                weight = 3.0;
            } else if description.contains(token.as_str())
                || tags.iter().any(|tag| tag.contains(token.as_str()))
            {
                weight = 2.0;
            } else if body.contains(token.as_str()) {
                weight = 1.0;
            }
            matched += weight;
        }
        let score = (matched / divisor).min(1.0);
        if score > 0.0 {
            scores.push((fact.name.clone(), score));
        }
    }
    scores
}

/// Builds a [`RecallHit`] for `fact` with the given score and components.
fn recall_hit(fact: &CodingFact, score: f32, similarity: f32, keyword: f32) -> RecallHit {
    let mut record = fact.to_record();
    record.importance = 0.5;
    let importance = record.importance;
    RecallHit {
        record,
        score,
        similarity,
        keyword,
        importance,
        recency: recency(fact),
    }
}

/// The recency component: `0.5^(age / 30 days)`, clamped to `0.0..=1.0`.
fn recency(fact: &CodingFact) -> f32 {
    let now = Utc::now();
    let age = match now.signed_duration_since(fact.updated_at).num_seconds() {
        age if age <= 0 => return 1.0,
        age => age as f32,
    };
    (0.5_f32).powf(age / RECENCY_HALF_LIFE_SECS).clamp(0.0, 1.0)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::float_cmp)]
mod tests {
    use std::sync::Arc;

    use tempfile::TempDir;

    use openjiuwen_common::error::Error as MemoryError;

    use super::super::embedding::test_support::{
        fact, hit_name, open_embedded, open_keyword, CountingEmbedding,
    };
    use super::*;

    /// The mode options used by the ranking assertions.
    fn options(mode: SearchMode) -> SearchOptions {
        SearchOptions::default().with_mode(mode).with_threshold(0.0)
    }

    #[tokio::test]
    async fn semantic_search_ranks_an_exact_match_first_and_filters_by_threshold() {
        let temp = TempDir::new().unwrap();
        let counter = Arc::new(CountingEmbedding::new());
        let store = open_embedded(&temp, &counter).await;
        store
            .write(fact("notes", "alpha beta gamma"))
            .await
            .unwrap();
        store.write(fact("other", "delta epsilon")).await.unwrap();

        let hits = store
            .search("alpha beta gamma", &options(SearchMode::Semantic))
            .await
            .unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hit_name(&hits[0]), "notes");
        assert!(
            hits[0].similarity > 0.99,
            "identical text must score 1.0, got {}",
            hits[0].similarity
        );
        assert!(hits[0].score > hits[1].score);

        let strict = options(SearchMode::Semantic).with_threshold(0.9);
        let hits = store.search("alpha beta gamma", &strict).await.unwrap();
        assert_eq!(hits.len(), 1, "the non-matching fact is filtered");
        assert_eq!(hit_name(&hits[0]), "notes");
    }

    #[tokio::test]
    async fn semantic_search_without_an_embedding_names_the_fix() {
        let temp = TempDir::new().unwrap();
        let store = open_keyword(&temp).await;
        let error = store
            .search("anything", &options(SearchMode::Semantic))
            .await
            .unwrap_err();
        assert!(matches!(error, MemoryError::Unsupported(_)), "{error:?}");
        let message = error.to_string();
        assert!(message.contains("embedding"), "{message}");
        assert!(message.contains("LocalCodingMemory::builder"), "{message}");
    }

    #[tokio::test]
    async fn hybrid_search_differs_from_keyword_when_the_signals_disagree() {
        let temp = TempDir::new().unwrap();
        let counter = Arc::new(CountingEmbedding::new());
        let store = open_embedded(&temp, &counter).await;
        // The keyword signal prefers `release-notes` (both query tokens match its name),
        // while the semantic signal prefers `shipping` (its body is the query verbatim).
        store
            .write(fact("release-notes", "unrelated words here"))
            .await
            .unwrap();
        store
            .write(fact("shipping", "release notes"))
            .await
            .unwrap();

        let keyword = store
            .search("release notes", &options(SearchMode::Keyword))
            .await
            .unwrap();
        assert_eq!(hit_name(&keyword[0]), "release-notes");
        assert_eq!(
            counter.query_calls(),
            0,
            "keyword mode must not embed the query"
        );
        assert_eq!(counter.document_calls(), 2, "only the writes embedded text");

        let hybrid = store
            .search("release notes", &options(SearchMode::Hybrid))
            .await
            .unwrap();
        assert_eq!(hybrid.len(), 2);
        assert_eq!(hit_name(&hybrid[0]), "shipping");
        assert_eq!(
            counter.query_calls(),
            1,
            "the hybrid query is embedded once"
        );
        assert!(
            hybrid[0].similarity > hybrid[1].similarity,
            "the semantic ranking disagrees with the keyword one"
        );
    }

    #[tokio::test]
    async fn hybrid_without_an_embedding_keeps_the_keyword_result() {
        let temp = TempDir::new().unwrap();
        let store = open_keyword(&temp).await;
        store.write(fact("build", "cargo build")).await.unwrap();
        let hybrid = store
            .search("cargo build", &options(SearchMode::Hybrid))
            .await
            .unwrap();
        let keyword = store
            .search("cargo build", &options(SearchMode::Keyword))
            .await
            .unwrap();
        assert_eq!(hybrid.len(), keyword.len());
        for (hybrid, keyword) in hybrid.iter().zip(&keyword) {
            assert_eq!(hit_name(hybrid), hit_name(keyword));
            assert_eq!(hybrid.score, keyword.score);
            assert_eq!(hybrid.keyword, keyword.keyword);
            assert_eq!(
                hybrid.similarity, 0.0,
                "no embedding means no semantic term"
            );
            assert_eq!(hybrid.score, hybrid.keyword);
        }
    }

    #[tokio::test]
    async fn recall_hits_report_the_documented_components() {
        let temp = TempDir::new().unwrap();
        let counter = Arc::new(CountingEmbedding::new());
        let store = open_embedded(&temp, &counter).await;
        store.write(fact("summary", "alpha beta")).await.unwrap();

        let hybrid = store
            .search("alpha beta", &options(SearchMode::Hybrid))
            .await
            .unwrap();
        let hit = &hybrid[0];
        assert!(hit.similarity > 0.99, "got similarity {}", hit.similarity);
        assert!(
            hit.keyword > 0.0 && hit.keyword < 1.0,
            "a body-only token match scores below one, got {}",
            hit.keyword
        );
        assert_eq!(
            hit.score,
            blend(hit.similarity, hit.keyword, VECTOR_WEIGHT, TEXT_WEIGHT),
            "score is the documented blend"
        );
        assert!(
            (hit.score - (0.7 * hit.similarity + 0.3 * hit.keyword)).abs() < 1e-5,
            "score uses the 0.7/0.3 weights, got {}",
            hit.score
        );

        let keyword = store
            .search("alpha beta", &options(SearchMode::Keyword))
            .await
            .unwrap();
        assert_eq!(
            keyword[0].keyword, hit.keyword,
            "the component is mode-neutral"
        );
        assert_eq!(keyword[0].score, hit.keyword);
        assert_eq!(keyword[0].similarity, 0.0);

        let semantic = store
            .search("alpha beta", &options(SearchMode::Semantic))
            .await
            .unwrap();
        assert_eq!(semantic[0].similarity, hit.similarity);
        assert_eq!(semantic[0].score, hit.similarity);
        assert_eq!(semantic[0].keyword, hit.keyword);
    }
}
