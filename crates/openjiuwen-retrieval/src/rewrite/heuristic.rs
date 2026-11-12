//! Deterministic, model-free query rewriting.
//!
//! Python ships only the LLM-backed rewriter, so this module is a Rust
//! addition required by the offline story: it resolves follow-up turns from
//! history, strips conversational filler, ranks keywords (optionally with IDF
//! weights from a corpus vocabulary), and expands synonyms — all without a
//! model, in time linear in the input, with a stable result for identical
//! inputs.
//!
//! It fills the same [`RewriteOutput`] shape the model rewriter parses from an
//! LLM, so callers can swap the two implementations behind
//! [`QueryRewriter`].

use std::collections::{BTreeMap, BTreeSet};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use openjiuwen_common::error::Result;

use super::{validate_query, QueryRewriter, RewriteOutput, RewriteRequest};

/// Default number of keywords returned by [`HeuristicQueryRewriter`].
pub const DEFAULT_MAX_KEYWORDS: usize = 8;
/// Default number of retrieval variants returned by expansion.
pub const DEFAULT_MAX_QUERIES: usize = 3;
/// Default number of history turns consulted for follow-up resolution.
pub const DEFAULT_HISTORY_WINDOW: usize = 4;

/// Longest `intention` string produced by the heuristic rewriter, in characters.
const MAX_INTENTION_CHARS: usize = 120;
/// Longest history topic copied into a standalone query, in tokens.
const MAX_HISTORY_TOPIC_TOKENS: usize = 16;
/// Queries at most this many tokens long are treated as follow-ups.
const SHORT_QUERY_TOKENS: usize = 3;

/// Conversational openers stripped from the start and end of a query.
const FILLER_WORDS: &[&str] = &[
    "please", "um", "uh", "er", "erm", "so", "ok", "okay", "well", "hey",
];

/// First tokens that make a query refer to an earlier turn.
const REFERENTIAL_TOKENS: &[&str] = &[
    "it", "its", "that", "this", "those", "these", "they", "them", "and", "but", "also", "then",
];

/// Opening phrases that make a query refer to an earlier turn.
const REFERENTIAL_PHRASES: &[&str] = &[
    "what about",
    "how about",
    "what if",
    "and what",
    "and how",
    "tell me more",
    "more about",
];

/// Built-in English stopwords dropped from keyword extraction.
const STOPWORDS: &[&str] = &[
    "a", "about", "after", "all", "also", "am", "an", "and", "any", "are", "as", "at", "be",
    "because", "been", "before", "but", "by", "can", "could", "did", "do", "does", "doing", "for",
    "from", "had", "has", "have", "he", "her", "here", "him", "his", "how", "i", "if", "in",
    "into", "is", "it", "its", "just", "me", "more", "most", "my", "no", "not", "now", "of", "on",
    "or", "our", "out", "over", "should", "so", "some", "such", "than", "that", "the", "their",
    "them", "then", "there", "these", "they", "this", "those", "to", "too", "up", "us", "very",
    "was", "we", "were", "what", "when", "where", "which", "who", "whom", "why", "will", "with",
    "would", "you", "your",
];

/// Configuration for [`HeuristicQueryRewriter`].
///
/// Every field is serializable so a rewrite policy can live in configuration
/// rather than code. Defaults mirror the constants above.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HeuristicRewriteConfig {
    /// Maximum number of keywords returned by extraction and recorded in
    /// [`RewriteOutput::missing`].
    #[serde(default = "default_max_keywords")]
    pub max_keywords: usize,
    /// Maximum number of retrieval variants returned by expansion (at least
    /// one: the original query).
    #[serde(default = "default_max_queries")]
    pub max_queries: usize,
    /// Whether the built-in and configured stopwords are dropped.
    #[serde(default = "default_true")]
    pub use_stopwords: bool,
    /// Extra stopwords, applied in addition to the built-in list when
    /// `use_stopwords` is set.
    #[serde(default)]
    pub stopwords: Vec<String>,
    /// Synonym expansions keyed by lowercase term; each entry becomes a
    /// `references` value in the rewrite output.
    #[serde(default)]
    pub synonyms: BTreeMap<String, Vec<String>>,
    /// Term → document frequency over the indexed corpus, used for IDF
    /// keyword ranking. `None` ranks by token length first.
    #[serde(default)]
    pub corpus_vocabulary: Option<BTreeMap<String, usize>>,
    /// Number of history turns consulted when resolving a follow-up.
    #[serde(default = "default_history_window")]
    pub history_window: usize,
}

/// Serde default for [`HeuristicRewriteConfig::max_keywords`].
fn default_max_keywords() -> usize {
    DEFAULT_MAX_KEYWORDS
}

/// Serde default for [`HeuristicRewriteConfig::max_queries`].
fn default_max_queries() -> usize {
    DEFAULT_MAX_QUERIES
}

/// Serde default for [`HeuristicRewriteConfig::history_window`].
fn default_history_window() -> usize {
    DEFAULT_HISTORY_WINDOW
}

/// Serde default for boolean policy switches.
fn default_true() -> bool {
    true
}

impl Default for HeuristicRewriteConfig {
    fn default() -> Self {
        Self {
            max_keywords: DEFAULT_MAX_KEYWORDS,
            max_queries: DEFAULT_MAX_QUERIES,
            use_stopwords: true,
            stopwords: Vec::new(),
            synonyms: BTreeMap::new(),
            corpus_vocabulary: None,
            history_window: DEFAULT_HISTORY_WINDOW,
        }
    }
}

impl HeuristicRewriteConfig {
    /// Builder: sets the keyword cap.
    #[must_use]
    pub fn with_max_keywords(mut self, max_keywords: usize) -> Self {
        self.max_keywords = max_keywords;
        self
    }

    /// Builder: sets the expansion cap.
    #[must_use]
    pub fn with_max_queries(mut self, max_queries: usize) -> Self {
        self.max_queries = max_queries;
        self
    }

    /// Builder: replaces the synonym table.
    #[must_use]
    pub fn with_synonyms(mut self, synonyms: BTreeMap<String, Vec<String>>) -> Self {
        self.synonyms = synonyms;
        self
    }

    /// Builder: attaches a corpus vocabulary for IDF ranking.
    #[must_use]
    pub fn with_corpus_vocabulary(mut self, vocabulary: BTreeMap<String, usize>) -> Self {
        self.corpus_vocabulary = Some(vocabulary);
        self
    }
}

/// A deterministic rewriter that needs no model.
///
/// The rewrite pipeline is: normalise whitespace and filler → detect whether
/// the query is a follow-up → resolve it against the last history turn →
/// extract keywords → record synonyms, missing terms, and the history turn
/// used. `typo` and `gibberish` are always empty: this rewriter performs no
/// spell checking and does not judge tokens meaningless.
#[derive(Debug, Clone)]
pub struct HeuristicQueryRewriter {
    /// Rewrite policy.
    config: HeuristicRewriteConfig,
    /// Effective stopword set (built-in plus configured, all lowercase).
    stopwords: BTreeSet<String>,
}

impl Default for HeuristicQueryRewriter {
    fn default() -> Self {
        Self::new(HeuristicRewriteConfig::default())
    }
}

impl HeuristicQueryRewriter {
    /// Builds a rewriter from `config`.
    #[must_use]
    pub fn new(config: HeuristicRewriteConfig) -> Self {
        let mut stopwords = BTreeSet::new();
        if config.use_stopwords {
            for stopword in STOPWORDS {
                stopwords.insert((*stopword).to_owned());
            }
            for stopword in &config.stopwords {
                let lowered = stopword.trim().to_lowercase();
                if !lowered.is_empty() {
                    stopwords.insert(lowered);
                }
            }
        }
        Self { config, stopwords }
    }

    /// The rewrite policy in use.
    #[must_use]
    pub fn config(&self) -> &HeuristicRewriteConfig {
        &self.config
    }

    /// Extracts keywords of `query`, most important first.
    ///
    /// Tokens are lowercased and split on non-alphanumeric characters; with
    /// `use_stopwords` the built-in and configured stopwords are dropped and
    /// duplicates are kept once (first occurrence). Ranking is by descending
    /// IDF when a corpus vocabulary is configured — unknown terms are treated
    /// as document frequency 1 and the corpus size is the vocabulary's largest
    /// document frequency — and otherwise by descending token length. Ties keep
    /// query order.
    #[must_use]
    pub fn extract_keywords(&self, query: &str) -> Vec<String> {
        let normalized = normalize_query(query);
        let mut seen = BTreeSet::new();
        let mut candidates: Vec<(usize, String)> = Vec::new();
        for (position, token) in tokenize(&normalized).into_iter().enumerate() {
            if self.stopwords.contains(&token) {
                continue;
            }
            if seen.insert(token.clone()) {
                candidates.push((position, token));
            }
        }

        if let Some(vocabulary) = self.config.corpus_vocabulary.as_ref() {
            let corpus_size = vocabulary.values().copied().max().unwrap_or(1);
            candidates.sort_by(|left, right| {
                idf(vocabulary, corpus_size, &right.1)
                    .total_cmp(&idf(vocabulary, corpus_size, &left.1))
                    .then_with(|| right.1.chars().count().cmp(&left.1.chars().count()))
                    .then_with(|| left.0.cmp(&right.0))
            });
        } else {
            candidates.sort_by(|left, right| {
                right
                    .1
                    .chars()
                    .count()
                    .cmp(&left.1.chars().count())
                    .then_with(|| left.0.cmp(&right.0))
            });
        }

        candidates
            .into_iter()
            .map(|(_, token)| token)
            .take(self.config.max_keywords)
            .collect()
    }

    /// Expands `query` into retrieval variants.
    ///
    /// The first variant is always the original query (whitespace collapsed).
    /// It is followed by the keyword-only query and then by a variant that
    /// appends each keyword's configured synonyms, space separated so a bag-of-
    /// words retriever treats them as alternatives. Variants are deduplicated
    /// and truncated to [`HeuristicRewriteConfig::max_queries`] (at least one).
    #[must_use]
    pub fn expand_query(&self, query: &str) -> Vec<String> {
        let keywords = self.extract_keywords(query);
        let mut variants = vec![collapse_whitespace(query)];
        if keywords.is_empty() {
            return truncate_variants(variants, self.config.max_queries);
        }

        variants.push(keywords.join(" "));
        let expanded = keywords
            .iter()
            .map(|keyword| {
                let mut terms = vec![keyword.clone()];
                if let Some(synonyms) = self.config.synonyms.get(keyword) {
                    terms.extend(
                        synonyms
                            .iter()
                            .map(|synonym| collapse_whitespace(synonym))
                            .filter(|synonym| !synonym.is_empty()),
                    );
                }
                dedupe(terms).join(" ")
            })
            .collect::<Vec<String>>()
            .join(" ");
        variants.push(expanded);

        truncate_variants(variants, self.config.max_queries)
    }

    /// Builds the rewrite output for `request` without touching a model.
    ///
    /// The query is normalised, so filler at either end disappears from the
    /// standalone query while `before` keeps the user's original wording. A
    /// follow-up (short query or referential opening) with non-empty history is
    /// completed as `"<last history topic> <query>"`, and that history turn is
    /// echoed in [`RewriteOutput::from_history`].
    #[must_use]
    pub fn build_output(&self, request: &RewriteRequest) -> RewriteOutput {
        let before = request.query.trim();
        let normalized = normalize_query(before);
        let recent = self.recent_history(&request.history);
        let keywords = self.extract_keywords(&normalized);

        let mut from_history = String::new();
        let mut standalone_query = normalized.clone();
        if Self::is_follow_up(&normalized) {
            if let Some(topic_turn) = recent.last() {
                let topic = history_topic(topic_turn);
                if !topic.is_empty() && !standalone_query.is_empty() {
                    standalone_query = format!("{topic} {standalone_query}");
                }
                from_history.clone_from(*topic_turn);
            }
        }

        let missing = if recent.is_empty() {
            Vec::new()
        } else {
            let history_tokens: BTreeSet<String> = recent
                .iter()
                .flat_map(|turn| tokenize(turn.as_str()))
                .collect();
            keywords
                .iter()
                .filter(|keyword| !history_tokens.contains(keyword.as_str()))
                .take(self.config.max_keywords)
                .cloned()
                .collect()
        };

        let mut references = serde_json::Map::new();
        for keyword in &keywords {
            let Some(synonyms) = self.config.synonyms.get(keyword) else {
                continue;
            };
            let expansions: Vec<Value> = dedupe(
                synonyms
                    .iter()
                    .map(|synonym| collapse_whitespace(synonym))
                    .filter(|synonym| !synonym.is_empty())
                    .collect(),
            )
            .into_iter()
            .map(Value::String)
            .collect();
            if !expansions.is_empty() {
                references.insert(keyword.clone(), Value::Array(expansions));
            }
        }

        RewriteOutput {
            before: before.to_owned(),
            intention: intent_of(before),
            standalone_query,
            references,
            missing,
            typo: Vec::new(),
            gibberish: Vec::new(),
            from_history,
        }
    }

    /// The non-blank history turns considered, oldest first, limited to the
    /// window.
    fn recent_history<'a>(&self, history: &'a [String]) -> Vec<&'a String> {
        let window = self.config.history_window.max(1);
        let turns: Vec<&'a String> = history
            .iter()
            .filter(|turn| !turn.trim().is_empty())
            .collect();
        let start = turns.len().saturating_sub(window);
        turns[start..].to_vec()
    }

    /// Whether `query` reads as a follow-up to an earlier turn.
    ///
    /// A query is a follow-up when it is short (at most
    /// [`SHORT_QUERY_TOKENS`] tokens), opens with a referential token such as
    /// `it` or `those`, or opens with a referential phrase such as
    /// `what about`.
    fn is_follow_up(query: &str) -> bool {
        let lowered = query.to_lowercase();
        if REFERENTIAL_PHRASES
            .iter()
            .any(|phrase| lowered.starts_with(phrase))
        {
            return true;
        }
        let tokens = tokenize(query);
        let Some(first) = tokens.first() else {
            return false;
        };
        if tokens.len() <= SHORT_QUERY_TOKENS {
            return true;
        }
        REFERENTIAL_TOKENS.contains(&first.as_str())
    }
}

#[async_trait]
impl QueryRewriter for HeuristicQueryRewriter {
    async fn rewrite(&self, request: &RewriteRequest) -> Result<RewriteOutput> {
        validate_query(&request.query)?;
        Ok(self.build_output(request))
    }

    async fn keywords(&self, query: &str) -> Result<Vec<String>> {
        validate_query(query)?;
        Ok(self.extract_keywords(query))
    }

    async fn expand(&self, query: &str) -> Result<Vec<String>> {
        validate_query(query)?;
        Ok(self.expand_query(query))
    }
}

/// Collapses whitespace and strips conversational filler from both ends.
///
/// Leading and trailing filler words (`"please"`, `"um"`, `"so"`, …) are
/// removed until a non-filler token is reached, at least one token is always
/// kept, and adjacent duplicate tokens are collapsed (`"very very good"` →
/// `"very good"`).
#[must_use]
pub fn normalize_query(text: &str) -> String {
    let tokens: Vec<&str> = text.split_whitespace().collect();
    if tokens.is_empty() {
        return String::new();
    }

    let mut start = 0;
    let mut end = tokens.len();
    while end > start + 1 && is_filler(tokens[start]) {
        start += 1;
    }
    while end > start + 1 && is_filler(tokens[end - 1]) {
        end -= 1;
    }

    let mut collapsed: Vec<&str> = Vec::with_capacity(end - start);
    for token in &tokens[start..end] {
        if collapsed
            .last()
            .is_some_and(|last| last.eq_ignore_ascii_case(token))
        {
            continue;
        }
        collapsed.push(token);
    }
    collapsed.join(" ")
}

/// Splits `text` into lowercase alphanumeric tokens.
///
/// Every non-alphanumeric character is a separator, so punctuation never
/// reaches keyword comparison. Scripts without spaces (Chinese, Japanese)
/// therefore stay one token per contiguous run; this is the documented
/// approximation of Python's `tiktoken`-based tokenisation.
#[must_use]
pub fn tokenize(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    for character in text.chars() {
        if character.is_alphanumeric() {
            current.extend(character.to_lowercase());
        } else if !current.is_empty() {
            tokens.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}

/// Whether `token` is a conversational filler word, ignoring punctuation.
fn is_filler(token: &str) -> bool {
    let trimmed = token.trim_matches(|character: char| !character.is_alphanumeric());
    if trimmed.is_empty() {
        return false;
    }
    FILLER_WORDS.contains(&trimmed.to_lowercase().as_str())
}

/// Joins whitespace-separated tokens with single spaces.
fn collapse_whitespace(text: &str) -> String {
    text.split_whitespace().collect::<Vec<&str>>().join(" ")
}

/// Inverse document frequency of `term` over a corpus of `corpus_size` terms.
fn idf(vocabulary: &BTreeMap<String, usize>, corpus_size: usize, term: &str) -> f64 {
    let frequency = vocabulary.get(term).copied().unwrap_or(1).max(1);
    let corpus_size = corpus_size as f64;
    let frequency = frequency as f64;
    ((corpus_size + 1.0) / (frequency + 1.0)).ln() + 1.0
}

/// The topic of a history turn: the first
/// [`MAX_HISTORY_TOPIC_TOKENS`] whitespace tokens of its normalised text.
fn history_topic(turn: &str) -> String {
    let normalized = normalize_query(turn);
    let mut topic = String::new();
    for (index, token) in normalized.split(' ').enumerate() {
        if index >= MAX_HISTORY_TOPIC_TOKENS {
            break;
        }
        if !topic.is_empty() {
            topic.push(' ');
        }
        topic.push_str(token);
    }
    topic
}

/// The lowercased first sentence of `query`, truncated to
/// [`MAX_INTENTION_CHARS`] characters.
fn intent_of(query: &str) -> String {
    let first_sentence = query
        .split(['.', '!', '?', '\n'])
        .map(str::trim)
        .find(|sentence| !sentence.is_empty())
        .unwrap_or_else(|| query.trim());
    first_sentence
        .to_lowercase()
        .chars()
        .take(MAX_INTENTION_CHARS)
        .collect()
}

/// Removes duplicate strings, keeping first-occurrence order.
fn dedupe(values: Vec<String>) -> Vec<String> {
    let mut seen = BTreeSet::new();
    values
        .into_iter()
        .filter(|value| seen.insert(value.clone()))
        .collect()
}

/// Deduplicates, drops blank variants, and applies the `max_queries` bound
/// (never below one).
fn truncate_variants(variants: Vec<String>, max_queries: usize) -> Vec<String> {
    dedupe(variants)
        .into_iter()
        .filter(|variant| !variant.is_empty())
        .take(max_queries.max(1))
        .collect()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]
mod tests {
    use super::*;

    fn synonyms() -> BTreeMap<String, Vec<String>> {
        let mut table = BTreeMap::new();
        table.insert("rust".to_owned(), vec!["rustlang".to_owned()]);
        table.insert("async".to_owned(), vec!["asynchronous".to_owned()]);
        table
    }

    #[test]
    fn normalize_collapses_whitespace_and_strips_fillers() {
        assert_eq!(
            normalize_query("  please   um   what is rust  "),
            "what is rust"
        );
        assert_eq!(normalize_query("what is rust, please"), "what is rust,");
        assert_eq!(normalize_query("so so what is rust"), "what is rust");
        assert_eq!(normalize_query("   "), "");
        assert_eq!(normalize_query("ok"), "ok");
    }

    #[test]
    fn tokenize_lowercases_and_splits_on_punctuation() {
        assert_eq!(
            tokenize("What's RUST, really? v1.2"),
            vec!["what", "s", "rust", "really", "v1", "2"]
        );
        assert!(tokenize("?!").is_empty());
    }

    #[test]
    fn keywords_drop_stopwords_and_rank_by_length() {
        let rewriter = HeuristicQueryRewriter::default();
        assert_eq!(
            rewriter.extract_keywords("What is the difference between rust and python"),
            vec!["difference", "between", "python", "rust"]
        );
    }

    #[test]
    fn keywords_dedupe_and_respect_the_cap() {
        let config = HeuristicRewriteConfig::default().with_max_keywords(2);
        let rewriter = HeuristicQueryRewriter::new(config);
        assert_eq!(
            rewriter.extract_keywords("tokio tokio runtime scheduler"),
            vec!["scheduler", "runtime"]
        );
        assert!(rewriter.extract_keywords("what is it").is_empty());
    }

    #[test]
    fn keywords_rank_by_idf_when_a_corpus_vocabulary_is_supplied() {
        let vocabulary: BTreeMap<String, usize> = [
            ("rust".to_owned(), 90),
            ("python".to_owned(), 80),
            ("tokio".to_owned(), 2),
        ]
        .into_iter()
        .collect();
        let config = HeuristicRewriteConfig::default().with_corpus_vocabulary(vocabulary);
        let rewriter = HeuristicQueryRewriter::new(config);
        assert_eq!(
            rewriter.extract_keywords("rust python tokio"),
            vec!["tokio", "python", "rust"]
        );
    }

    #[test]
    fn extra_stopwords_are_honoured() {
        let config = HeuristicRewriteConfig {
            stopwords: vec!["Rust".to_owned()],
            ..HeuristicRewriteConfig::default()
        };
        let rewriter = HeuristicQueryRewriter::new(config);
        assert_eq!(rewriter.extract_keywords("rust tokio"), vec!["tokio"]);
    }

    #[test]
    fn rewrite_keeps_the_original_wording_in_before() {
        let rewriter = HeuristicQueryRewriter::default();
        let output = rewriter.build_output(&RewriteRequest::new("  Please explain Tokio  "));
        assert_eq!(output.before, "Please explain Tokio");
        assert_eq!(output.standalone_query, "explain Tokio");
        assert_eq!(output.intention, "please explain tokio");
        assert!(output.typo.is_empty());
        assert!(output.gibberish.is_empty());
        assert!(output.from_history.is_empty());
    }

    #[test]
    fn follow_up_queries_are_completed_from_the_last_history_turn() {
        let rewriter = HeuristicQueryRewriter::default();
        let request = RewriteRequest::new("what about it")
            .with_history(["tell me about the rust book", "which chapter covers async"]);
        let output = rewriter.build_output(&request);
        assert_eq!(
            output.standalone_query,
            "which chapter covers async what about it"
        );
        assert_eq!(output.from_history, "which chapter covers async");
    }

    #[test]
    fn non_follow_up_queries_ignore_history() {
        let rewriter = HeuristicQueryRewriter::default();
        let request = RewriteRequest::new("explain the tokio scheduler in detail")
            .with_history(["tell me about the rust book"]);
        let output = rewriter.build_output(&request);
        assert_eq!(
            output.standalone_query,
            "explain the tokio scheduler in detail"
        );
        assert!(output.from_history.is_empty());
    }

    #[test]
    fn short_queries_without_history_are_not_completed() {
        let rewriter = HeuristicQueryRewriter::default();
        let output = rewriter.build_output(&RewriteRequest::new("tokio"));
        assert_eq!(output.standalone_query, "tokio");
        assert!(output.from_history.is_empty());
    }

    #[test]
    fn history_window_limits_which_turn_is_used() {
        let config = HeuristicRewriteConfig {
            history_window: 1,
            ..HeuristicRewriteConfig::default()
        };
        let rewriter = HeuristicQueryRewriter::new(config);
        let request =
            RewriteRequest::new("what about it").with_history(["first turn", "second turn"]);
        let output = rewriter.build_output(&request);
        assert_eq!(output.from_history, "second turn");
        assert_eq!(output.standalone_query, "second turn what about it");
    }

    #[test]
    fn missing_lists_query_keywords_absent_from_history() {
        let rewriter = HeuristicQueryRewriter::default();
        let request = RewriteRequest::new("tokio runtime scheduler")
            .with_history(["explain the tokio runtime"]);
        let output = rewriter.build_output(&request);
        assert_eq!(output.missing, vec!["scheduler"]);
    }

    #[test]
    fn missing_is_empty_without_history() {
        let rewriter = HeuristicQueryRewriter::default();
        let output = rewriter.build_output(&RewriteRequest::new("tokio runtime scheduler"));
        assert!(output.missing.is_empty());
    }

    #[test]
    fn references_carry_synonym_expansions() {
        let config = HeuristicRewriteConfig::default().with_synonyms(synonyms());
        let rewriter = HeuristicQueryRewriter::new(config);
        let output = rewriter.build_output(&RewriteRequest::new("rust async runtime"));
        assert_eq!(output.references["rust"], serde_json::json!(["rustlang"]));
        assert_eq!(
            output.references["async"],
            serde_json::json!(["asynchronous"])
        );
        assert!(output.references.get("runtime").is_none());
    }

    #[test]
    fn expansion_keeps_the_original_first_and_is_bounded() {
        let config = HeuristicRewriteConfig::default().with_synonyms(synonyms());
        let rewriter = HeuristicQueryRewriter::new(config);
        let variants = rewriter.expand_query("rust async runtime");
        assert_eq!(variants.len(), DEFAULT_MAX_QUERIES);
        assert_eq!(variants[0], "rust async runtime");
        assert_eq!(variants[1], "runtime async rust");
        assert_eq!(variants[2], "runtime async asynchronous rust rustlang");
    }

    #[test]
    fn expansion_deduplicates_and_respects_the_cap() {
        let config = HeuristicRewriteConfig {
            max_queries: 2,
            ..HeuristicRewriteConfig::default()
        };
        let rewriter = HeuristicQueryRewriter::new(config);
        let variants = rewriter.expand_query("   ");
        assert!(variants.is_empty());
        let variants = rewriter.expand_query("rust");
        assert_eq!(variants, vec!["rust"]);
    }

    #[test]
    fn expansion_always_keeps_at_least_the_original() {
        let config = HeuristicRewriteConfig {
            max_queries: 0,
            ..HeuristicRewriteConfig::default()
        };
        let rewriter = HeuristicQueryRewriter::new(config);
        assert_eq!(rewriter.expand_query("rust tokio"), vec!["rust tokio"]);
    }

    #[tokio::test]
    async fn trait_methods_validate_and_delegate() {
        let rewriter = HeuristicQueryRewriter::default();
        let output = rewriter
            .rewrite(&RewriteRequest::new("what is rust"))
            .await
            .unwrap();
        assert_eq!(output.before, "what is rust");
        assert_eq!(
            rewriter.keywords("what is rust").await.unwrap(),
            vec!["rust"]
        );
        assert_eq!(
            rewriter.expand("rust async").await.unwrap(),
            vec!["rust async", "async rust"]
        );
        assert!(rewriter.rewrite(&RewriteRequest::new("")).await.is_err());
        assert!(rewriter.keywords("").await.is_err());
        assert!(rewriter.expand("  ").await.is_err());
    }
}
