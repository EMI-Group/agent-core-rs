//! Recall for the local store: the deterministic keyword scan, the optional FTS5 merge,
//! and the scoring/ranking helpers behind [`LocalCodingMemory::search`].
//!
//! The scoring rules are the ones documented on
//! [`crate::lite::local`]; the constants here are their numeric form.
use std::sync::OnceLock;

use chrono::Utc;
use regex::Regex;

use openjiuwen_common::error::{Error, Result};

use crate::lite::fact::CodingFact;
use crate::lite::markdown::MarkdownMemory;
use crate::record::{MemoryType, RecallHit, SearchMode, SearchOptions};

use super::state::LocalCodingMemory;

/// Maximum number of query tokens the keyword scan considers.
const MAX_QUERY_TOKENS: usize = 10;

/// Half-life of the recency component, in seconds (30 days).
const RECENCY_HALF_LIFE_SECS: f32 = 2_592_000.0;

impl LocalCodingMemory {
    /// Recalls the facts matching `query`.
    ///
    /// Hits are ordered by descending score and then by name ascending; see the module
    /// docs for the scoring rules. `kinds` filters the fact types (an empty list means
    /// "every kind", and a list without [`MemoryType::Coding`] yields no hit).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for a blank query, [`Error::Unsupported`] for
    /// semantic mode (the local store has no embeddings), the call's cancellation
    /// error, and the filesystem/index error when the search fails.
    pub async fn search(&self, query: &str, options: &SearchOptions) -> Result<Vec<RecallHit>> {
        let cleaned = query.trim();
        if cleaned.is_empty() {
            return Err(Error::validation(
                "search query must not be empty (got \"\")",
            ));
        }
        if options.mode == SearchMode::Semantic {
            return Err(Error::unsupported(
                "the local coding-memory store has no embedding index; use keyword or hybrid search",
            ));
        }
        if !options.kinds.is_empty() && !options.kinds.contains(&MemoryType::Coding) {
            return Ok(Vec::new());
        }
        options.call.check()?;
        let mut state = self.inner.state.lock().await;
        self.ensure_fresh(&mut state).await?;
        let facts = state.document.facts();
        drop(state);

        let limit = options.top_k.max(1);
        #[cfg_attr(not(feature = "sqlite"), allow(unused_mut))]
        let mut scores = keyword_scores(&facts, cleaned);
        #[cfg(feature = "sqlite")]
        {
            let candidates = self.fts.search(cleaned, limit.saturating_mul(4)).await?;
            for candidate in candidates {
                let score = 1.0 - candidate.score;
                let entry = scores
                    .iter_mut()
                    .find(|(name, _score)| *name == candidate.name);
                match entry {
                    Some((_name, score_slot)) => {
                        if score > *score_slot {
                            *score_slot = score;
                        }
                    }
                    None => scores.push((candidate.name, score)),
                }
            }
        }
        let mut ranked: Vec<(String, RecallHit)> = facts
            .iter()
            .filter_map(|fact| {
                let score = scores
                    .iter()
                    .find(|(name, _score)| *name == fact.name)
                    .map(|(_name, score)| *score)?;
                let score = score.clamp(0.0, 1.0);
                (score >= options.threshold).then(|| (fact.name.clone(), recall_hit(fact, score)))
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

/// Builds a [`RecallHit`] for `fact` with `score` as its blended score.
fn recall_hit(fact: &CodingFact, score: f32) -> RecallHit {
    let mut record = fact.to_record();
    record.importance = 0.5;
    let importance = record.importance;
    RecallHit {
        record,
        score,
        similarity: 0.0,
        keyword: score,
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
