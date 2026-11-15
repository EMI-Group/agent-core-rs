//! Deterministic field-weighted lexical ranking over catalog records and tree branches.
//!
//! Ports Python `openjiuwen.symphony.retrieval.search.runtime.lexical` (v0.1.20) — the
//! scorer `symphony/discovery` uses for live-inventory search — and serves the same role in
//! this crate: the offline candidate ranking behind
//! [`LexicalIndex`](crate::retrieval::search::lexical::LexicalIndex), and the ranking a
//! model-free ([`SelectionMode::Lexical`](crate::retrieval::search::config::SelectionMode))
//! traversal can consume.
//!
//! # Scoring
//!
//! The corpus statistics keep, per document, one weighted term-frequency map over five
//! fields — key, name, aliases, description and body — with the weights
//! `[5, 5, 5, 4, 1]`. A document's length is the sum of those weighted frequencies and
//! `average_length` is their mean. The category field is scored by the phrase rule only, so
//! it never enters the frequency map (mirroring the reference).
//!
//! A live document is scored by three additive parts:
//!
//! ```text
//! score = bm25(terms) + field_score(terms) + phrase_score(raw_query)
//! ```
//!
//! - **bm25**: for every *distinct* query term present in the document,
//!   `idf² · tf·(k1+1) / (tf + k1·(1 - b + b·dl/avgdl))` with
//!   `idf = ln(1 + (N - df + 0.5)/(df + 0.5))`, `k1 = `[`BM25_K1`]` = 1.5` and
//!   `b = `[`BM25_B`]` = 0.72`; the strongest single `idf²` is then added once, which is what
//!   makes one rare matching term outrank a document that merely repeats a common one.
//! - **`field_score`**: `Σ_f w_f · |terms ∩ field_f| / |terms|` with `w = `[`FIELD_WEIGHTS`]
//!   over the six fields, plus `IDENTITY_BONUS · |terms ∩ (key ∪ name ∪ aliases)| /
//!   |key ∪ name ∪ aliases|`, so an identity hit counts far more than a body word.
//! - **`phrase_score`**: for each `|`-separated alternative of the raw query,
//!   [`PHRASE_FIELD_WEIGHTS`] `[12, 5, 0.5, 0.1]` for a substring hit in the identity
//!   fields, the description, the body and the category respectively (first hit wins, as in
//!   the reference).
//!
//! Ranking order: identity tier first (`0` key, `1` name, `2` alias, `3` prefix of one of
//! those, `4` anything else), then descending score (`f64::total_cmp`, so no `NaN`
//! ordering surprise), then key ascending.
//!
//! # Tokenization
//!
//! [`tokenize`] implements the reference's `_WORD_RE` character classes directly: an ASCII
//! alphanumeric start followed by `[A-Za-z0-9_+.#/-]`, or a run of CJK characters
//! (`U+3400..=U+9FFF`). A match is lowercased, `._-/` are trimmed from both ends, a CJK run
//! also contributes its overlapping bigrams, a symbol token (`c++`, `c#`) stays whole, and
//! anything else is split on `[_+.#/-]+`. ASCII tokens additionally contribute their
//! singular and verb-derivation forms, exactly like the reference's `_tokens`, so `skills`
//! and `skill` meet on one term.
//!
//! # Deviations from the reference
//!
//! - **The matcher is a subset.** [`compile_matcher`] supports literal substrings,
//!   `|`-separated alternatives, quoted phrases (quotes stripped) and `.`/`.*` as "any
//!   short run" wildcards. No regex engine ships in this workspace, so the reference's
//!   `compile_safe_pattern` constructs — character classes `[...]`, quantifiers `+ ? *`,
//!   bounded repetition `{m,n}`, anchors `^ $`, groups `(...)` and backreferences — are
//!   **not interpreted**: those characters match literally. Safety over parity: the matcher
//!   cannot backtrack catastrophically, and no user input can make it panic.
//! - **`search_terms` is not ported.** Python's plural/verb term-family coverage ranking
//!   (`LexicalIndex.search_terms`) has no consumer in this crate; every family here is the
//!   token itself, which [`missing_terms`](crate::retrieval::search::lexical::LexicalIndex::missing_terms)
//!   documents.
//! - **`casefold()` is approximated by `to_lowercase()`**, the closest `std` equivalent.
//!   Identity normalization, phrase matching and the matcher all use it.
//! - **The entry points fix the reference defaults**: [`LexicalIndex::search`] and
//!   [`LexicalIndex::search_in`] rank case-insensitively with pattern matching enabled,
//!   which is what Python's `LexicalIndex.search` defaults to.
//! - **`from_loaded_index` indexes tree branches too** (`node::<node_id>`), so a query can
//!   rank a branch and an item with the same scorer; Python's discovery-side index has no
//!   branch documents.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::error::{validation, Result};
use crate::retrieval::types::{LoadedIndex, RetrieverNode, ROOT_NODE_ID};

/// The BM25 term-frequency saturation (`k1`) used by the reference scorer.
pub const BM25_K1: f64 = 1.5;
/// The BM25 document-length normalization (`b`) used by the reference scorer.
pub const BM25_B: f64 = 0.72;

/// The per-field weights of the field-overlap score.
///
/// The order is key, name, aliases, description, body, category.
pub const FIELD_WEIGHTS: [f64; 6] = [5.0, 5.0, 5.0, 3.4, 0.35, 0.1];

/// The bonus for a query term that appears in the identity fields (key, name, aliases).
pub const IDENTITY_BONUS: f64 = 16.0;

/// The per-field weights of the phrase score.
///
/// The order is the identity fields (key/name/aliases), description, body, category.
pub const PHRASE_FIELD_WEIGHTS: [f64; 4] = [12.0, 5.0, 0.5, 0.1];

/// The per-field weights of the corpus frequency map.
///
/// The order is key, name, aliases, description, body. The category field is deliberately
/// absent: it is scored by the phrase rule only.
pub const FREQUENCY_FIELD_WEIGHTS: [usize; 5] = [5, 5, 5, 4, 1];

/// The key prefix of the branch documents [`LexicalIndex::from_loaded_index`] adds.
pub const BRANCH_DOCUMENT_PREFIX: &str = "node::";

/// The longest accepted query, mirroring the reference's `compile_safe_pattern` bound.
pub const MAX_QUERY_CHARS: usize = 512;

/// One searchable record.
///
/// Mirrors Python `LexicalDocument`: `key` is the identity (`cid` for a catalog item), and
/// `body` is the long text (the catalog record's `retrieval_text`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct LexicalDocument {
    /// The document's identity inside the index.
    pub key: String,
    /// The display name.
    pub name: String,
    /// The routing description.
    pub description: String,
    /// The long text; a leading YAML front-matter block is stripped before scoring.
    pub body: String,
    /// Alternative names the document answers to.
    pub aliases: Vec<String>,
    /// The taxonomy path (`"Skills > Data"`), scored by the phrase rule only.
    pub category: String,
}

impl LexicalDocument {
    /// Builds a document with `key` and `name`; every other field stays empty.
    #[must_use]
    pub fn new(key: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            name: name.into(),
            ..Self::default()
        }
    }

    /// Sets the description.
    #[must_use]
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    /// Sets the body.
    #[must_use]
    pub fn with_body(mut self, body: impl Into<String>) -> Self {
        self.body = body.into();
        self
    }

    /// Replaces the aliases.
    #[must_use]
    pub fn with_aliases(mut self, aliases: Vec<String>) -> Self {
        self.aliases = aliases;
        self
    }

    /// Sets the category.
    #[must_use]
    pub fn with_category(mut self, category: impl Into<String>) -> Self {
        self.category = category.into();
        self
    }
}

/// One ranked lexical match.
///
/// Mirrors Python `LexicalHit`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct LexicalHit {
    /// The document key.
    pub key: String,
    /// The score; higher is more relevant.
    pub score: f64,
}

/// A reusable deterministic index over one immutable document set.
///
/// Mirrors Python `LexicalIndex`: the corpus statistics are computed once at build time, so
/// a search never rebuilds them.
#[derive(Debug, Clone, Default)]
pub struct LexicalIndex {
    /// The documents, sorted by key.
    documents: BTreeMap<String, LexicalDocument>,
    /// Each document's six normalized fields (key, name, aliases, description, body,
    /// category; the body has its front matter stripped).
    fields: BTreeMap<String, [String; 6]>,
    /// The text the matcher scans: key, name, aliases, description, body, category.
    search_text: BTreeMap<String, String>,
    /// Each document's six token sets (deduplicated).
    field_tokens: BTreeMap<String, [BTreeSet<String>; 6]>,
    /// Each document's weighted term frequencies.
    frequencies: BTreeMap<String, BTreeMap<String, usize>>,
    /// Each document's weighted length (the sum of its frequencies).
    lengths: BTreeMap<String, usize>,
    /// The mean weighted length over all documents.
    average_length: f64,
    /// How many documents contain each term.
    postings: BTreeMap<String, usize>,
    /// How many documents the index holds.
    size: usize,
}

impl LexicalIndex {
    /// Builds an index over `documents`.
    ///
    /// Documents are stored by key, so a duplicated key keeps the **last** document, and
    /// the corpus is always read in key order regardless of input order.
    #[must_use]
    pub fn build(documents: impl IntoIterator<Item = LexicalDocument>) -> Self {
        let mut by_key: BTreeMap<String, LexicalDocument> = BTreeMap::new();
        for document in documents {
            by_key.insert(document.key.clone(), document);
        }
        let mut index = Self {
            documents: by_key,
            ..Self::default()
        };
        index.recompute_statistics();
        index
    }

    /// Builds an index over one loaded capability-tree index.
    ///
    /// One document is derived per catalog record — key = the record's `cid`, name,
    /// description, body = its `retrieval_text`, category = its `branch_path` joined with
    /// `" > "` — and one per tree branch (a node with children), keyed `node::<node_id>`
    /// with the label as name, the node description, and the ancestor node ids joined with
    /// `" > "` as category. The synthetic [`ROOT_NODE_ID`] is not part of a category path,
    /// matching [`CatalogRecord::branch_path`](crate::retrieval::types::CatalogRecord).
    #[must_use]
    pub fn from_loaded_index(index: &LoadedIndex) -> Self {
        let mut documents: Vec<LexicalDocument> = index
            .records
            .iter()
            .map(|record| LexicalDocument {
                key: record.cid.clone(),
                name: record.name.clone(),
                description: record.description.clone(),
                body: record.retrieval_text.clone(),
                aliases: Vec::new(),
                category: record.branch_path.join(" > "),
            })
            .collect();
        let mut ancestors: Vec<String> = Vec::new();
        collect_branch_documents(&index.root, &mut ancestors, &mut documents);
        Self::build(documents)
    }

    /// Ranks every document against `query`.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`](crate::error::Error) naming `query` when it is blank, or when
    /// it is not a usable search expression (see [`compile_matcher`]).
    pub fn search(&self, query: &str) -> Result<Vec<LexicalHit>> {
        self.search_in(query, &[])
    }

    /// Ranks the `keys` scope against `query`.
    ///
    /// Keys that the index does not hold are ignored, and a duplicated key ranks once. An
    /// **empty** `keys` slice ranks the whole corpus (the same as
    /// [`LexicalIndex::search`]), so a caller needs no separate "unscoped" entry point.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`](crate::error::Error) naming `query` when it is blank or not a
    /// usable search expression.
    pub fn search_in(&self, query: &str, keys: &[&str]) -> Result<Vec<LexicalHit>> {
        let text = query.trim();
        if text.is_empty() {
            return Err(validation("query", "must be non-empty"));
        }
        let matcher = compile_matcher(text, true, false)?;
        let query_tokens = tokenize(&ranking_query(text));
        let scope: Vec<&LexicalDocument> = if keys.is_empty() {
            self.documents.values().collect()
        } else {
            let mut seen: BTreeSet<&str> = BTreeSet::new();
            keys.iter()
                .filter_map(|key| self.documents.get(*key))
                .filter(|document| seen.insert(document.key.as_str()))
                .collect()
        };
        let mut scored: Vec<(usize, String, LexicalHit)> = Vec::new();
        for document in scope {
            let Some(search_text) = self.search_text.get(&document.key) else {
                continue;
            };
            if !matcher.matches(search_text) {
                continue;
            }
            let ranking = self.bm25_score(&document.key, &query_tokens)
                + self.field_score(&document.key, &query_tokens)
                + self.phrase_score(&document.key, text);
            scored.push((
                self.identity_tier(&document.key, text),
                document.key.to_lowercase(),
                LexicalHit {
                    key: document.key.clone(),
                    score: ranking,
                },
            ));
        }
        scored.sort_by(|left, right| {
            left.0
                .cmp(&right.0)
                .then_with(|| right.2.score.total_cmp(&left.2.score))
                .then_with(|| left.1.cmp(&right.1))
                .then_with(|| left.2.key.cmp(&right.2.key))
        });
        Ok(scored.into_iter().map(|(_, _, hit)| hit).collect())
    }

    /// The query terms no document holds.
    ///
    /// Each distinct surface term of `query` is checked against the corpus (its derived
    /// forms included); the terms that appear nowhere are returned in query order. Python's
    /// plural/verb *families* are not reconstructed — the family of a term is the term and
    /// its own derivations — because the reference's `search_terms` coverage ranking is not
    /// ported.
    #[must_use]
    pub fn missing_terms(&self, query: &str) -> Vec<String> {
        let mut missing: Vec<String> = Vec::new();
        for token in surface_tokens(query) {
            if missing.contains(&token) {
                continue;
            }
            let family = tokenize(&token);
            if !family.iter().any(|term| self.postings.contains_key(term)) {
                missing.push(token);
            }
        }
        missing
    }

    /// Returns evidence for `key` when the matching field is otherwise hidden.
    ///
    /// Mirrors Python `LexicalIndex.match_snippet`, in this order:
    ///
    /// 1. a query that equals (or contains, or is contained by) the document's name or one
    ///    of its aliases returns `"name: …"` / `"alias: …"` with a bounded excerpt;
    /// 2. a matched term hidden beyond the first `visible_description_chars` characters of
    ///    the description returns `"description: …"`;
    /// 3. a body-only match returns `"body: …"` (front matter stripped).
    ///
    /// The excerpt is at most `max_chars` characters, never splits a character, and an
    /// empty `queries` list or `max_chars == 0` yields `""`.
    #[must_use]
    pub fn match_snippet(
        &self,
        key: &str,
        queries: &[&str],
        visible_description_chars: usize,
        max_chars: usize,
    ) -> String {
        let Some(document) = self.documents.get(key) else {
            return String::new();
        };
        if max_chars == 0 {
            return String::new();
        }
        let Some(field_tokens) = self.field_tokens.get(key) else {
            return String::new();
        };
        let mut matched: BTreeSet<String> = BTreeSet::new();
        for query in queries {
            if let Some(snippet) = Self::identity_snippet(document, query, max_chars) {
                return snippet;
            }
            let present = dedupe(tokenize(query));
            matched.extend(self.informative_terms(&present));
        }
        if matched.is_empty() {
            return String::new();
        }
        let description = document
            .description
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");
        let visible: String = description.chars().take(visible_description_chars).collect();
        let visible_tokens: BTreeSet<String> = tokenize(&visible).into_iter().collect();
        let hidden_description: BTreeSet<String> = matched
            .intersection(&field_tokens[3])
            .filter(|term| !visible_tokens.contains(*term))
            .cloned()
            .collect();
        if !hidden_description.is_empty() {
            return format!(
                "description: {}",
                matched_excerpt(&description, &hidden_description, max_chars)
            );
        }
        let metadata_tokens: BTreeSet<String> = field_tokens[..4]
            .iter()
            .flat_map(|tokens| tokens.iter().cloned())
            .collect();
        let body_matches: BTreeSet<String> = matched
            .intersection(&field_tokens[4])
            .filter(|term| !metadata_tokens.contains(*term))
            .cloned()
            .collect();
        if body_matches.is_empty() {
            return String::new();
        }
        let body = without_front_matter(&document.body);
        format!(
            "body: {}",
            matched_excerpt(&body, &body_matches, max_chars)
        )
    }

    /// The document stored for `key`.
    #[must_use]
    pub fn document(&self, key: &str) -> Option<&LexicalDocument> {
        self.documents.get(key)
    }

    /// How many documents are indexed.
    #[must_use]
    pub fn len(&self) -> usize {
        self.documents.len()
    }

    /// Whether the index holds no document.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.documents.is_empty()
    }

    /// Derives and stores every corpus statistic.
    #[allow(clippy::cast_precision_loss)]
    fn recompute_statistics(&mut self) {
        self.fields.clear();
        self.search_text.clear();
        self.field_tokens.clear();
        self.frequencies.clear();
        self.lengths.clear();
        self.postings.clear();
        self.average_length = 0.0;
        self.size = self.documents.len();
        let mut total_length = 0_usize;
        let documents = std::mem::take(&mut self.documents);
        for (key, document) in &documents {
            let fields = document_fields(document);
            let tokens: Vec<Vec<String>> = fields.iter().map(|field| tokenize(field)).collect();
            let tokens: [Vec<String>; 6] =
                std::array::from_fn(|index| tokens.get(index).cloned().unwrap_or_default());
            let sets: [BTreeSet<String>; 6] =
                std::array::from_fn(|index| tokens[index].iter().cloned().collect());
            self.search_text.insert(key.clone(), search_text(document, &fields));
            self.fields.insert(key.clone(), fields);
            self.field_tokens.insert(key.clone(), sets);
            let frequencies = weighted_frequencies(&tokens);
            let length = frequencies.values().sum::<usize>();
            total_length += length;
            self.lengths.insert(key.clone(), length);
            for term in frequencies.keys() {
                *self.postings.entry(term.clone()).or_insert(0) += 1;
            }
            self.frequencies.insert(key.clone(), frequencies);
        }
        self.documents = documents;
        self.average_length = if self.size == 0 {
            0.0
        } else {
            total_length as f64 / self.size as f64
        };
    }

    /// The BM25 part of a document's score.
    fn bm25_score(&self, key: &str, query_tokens: &[String]) -> f64 {
        let Some(frequencies) = self.frequencies.get(key) else {
            return 0.0;
        };
        let length = self.lengths.get(key).copied().unwrap_or(0);
        let mut score = 0.0_f64;
        let mut strongest = 0.0_f64;
        let mut seen: BTreeSet<&str> = BTreeSet::new();
        for token in query_tokens {
            if !seen.insert(token.as_str()) {
                continue;
            }
            let frequency = frequencies.get(token).copied().unwrap_or(0);
            if frequency == 0 {
                continue;
            }
            let (contribution, information) = self.bm25_contribution(token, frequency, length);
            score += contribution;
            strongest = strongest.max(information);
        }
        score + strongest
    }

    /// One term's BM25 contribution and its squared inverse document frequency.
    #[allow(clippy::cast_precision_loss)]
    fn bm25_contribution(&self, token: &str, frequency: usize, length: usize) -> (f64, f64) {
        let document_frequency = self.postings.get(token).copied().unwrap_or(0) as f64;
        let population = self.size as f64;
        let inverse_frequency =
            (1.0 + (population - document_frequency + 0.5) / (document_frequency + 0.5)).ln();
        let length_ratio = length as f64 / self.average_length.max(1e-9);
        let denominator =
            frequency as f64 + BM25_K1 * (1.0 - BM25_B + BM25_B * length_ratio);
        let information = inverse_frequency * inverse_frequency;
        let numerator = information * (frequency as f64 * (BM25_K1 + 1.0));
        (numerator / denominator, information)
    }

    /// The field-overlap part of a document's score.
    #[allow(clippy::cast_precision_loss)]
    fn field_score(&self, key: &str, query_tokens: &[String]) -> f64 {
        let Some(field_tokens) = self.field_tokens.get(key) else {
            return 0.0;
        };
        let terms: BTreeSet<&String> = query_tokens.iter().collect();
        if terms.is_empty() {
            return 0.0;
        }
        let mut score = 0.0_f64;
        for (tokens, weight) in field_tokens.iter().zip(FIELD_WEIGHTS) {
            let overlap = terms.iter().filter(|term| tokens.contains(**term)).count();
            if overlap > 0 {
                score += weight * overlap as f64 / terms.len() as f64;
            }
        }
        let identity_terms: BTreeSet<&String> = field_tokens[..3]
            .iter()
            .flat_map(|tokens| tokens.iter())
            .collect();
        let identity_overlap = terms
            .iter()
            .filter(|term| identity_terms.contains(**term))
            .count();
        if identity_overlap > 0 {
            score += IDENTITY_BONUS * identity_overlap as f64
                / identity_terms.len().max(1) as f64;
        }
        score
    }

    /// The phrase part of a document's score (case-insensitive).
    fn phrase_score(&self, key: &str, query: &str) -> f64 {
        let alternatives: Vec<String> = query
            .split('|')
            .map(str::trim)
            .filter(|part| !part.is_empty())
            .map(str::to_lowercase)
            .collect();
        if alternatives.is_empty() {
            return 0.0;
        }
        let Some(fields) = self.fields.get(key) else {
            return 0.0;
        };
        let folded: [String; 6] = std::array::from_fn(|index| fields[index].to_lowercase());
        let mut score = 0.0_f64;
        for alternative in &alternatives {
            let matched = if folded[0].contains(alternative.as_str())
                || folded[1].contains(alternative.as_str())
                || folded[2].contains(alternative.as_str())
            {
                Some(PHRASE_FIELD_WEIGHTS[0])
            } else if folded[3].contains(alternative.as_str()) {
                Some(PHRASE_FIELD_WEIGHTS[1])
            } else if folded[4].contains(alternative.as_str()) {
                Some(PHRASE_FIELD_WEIGHTS[2])
            } else if folded[5].contains(alternative.as_str()) {
                Some(PHRASE_FIELD_WEIGHTS[3])
            } else {
                None
            };
            if let Some(weight) = matched {
                score += weight;
            }
        }
        score
    }

    /// The identity tier of `key` for `query` (`0` exact key … `4` no identity relation).
    fn identity_tier(&self, key: &str, query: &str) -> usize {
        let Some(document) = self.documents.get(key) else {
            return 4;
        };
        let terms: Vec<String> = query
            .split('|')
            .map(str::trim)
            .filter(|part| !part.is_empty() && identity_alternative_allowed(part))
            .map(str::to_lowercase)
            .collect();
        if terms.is_empty() {
            return 4;
        }
        let key_value = key.to_lowercase();
        let name = document.name.to_lowercase();
        let aliases: Vec<String> = document.aliases.iter().map(|alias| alias.to_lowercase()).collect();
        if terms.iter().any(|term| term == &key_value) {
            return 0;
        }
        if terms.iter().any(|term| term == &name) {
            return 1;
        }
        if aliases
            .iter()
            .any(|alias| terms.iter().any(|term| term == alias))
        {
            return 2;
        }
        let prefixes = terms.iter().any(|term| {
            key_value.starts_with(term.as_str())
                || name.starts_with(term.as_str())
                || aliases.iter().any(|alias| alias.starts_with(term.as_str()))
        });
        if prefixes {
            return 3;
        }
        4
    }

    /// The identity snippet of `document` for `query`, when the identity itself matched.
    fn identity_snippet(
        document: &LexicalDocument,
        query: &str,
        max_chars: usize,
    ) -> Option<String> {
        if !document.name.is_empty()
            && identity_value_matches_query(&document.name, &document.key, query)
        {
            let length = document.name.chars().count();
            return Some(format!(
                "name: {}",
                excerpt(&document.name, 0, length, max_chars)
            ));
        }
        for alias in &document.aliases {
            if !alias.is_empty() && identity_value_matches_query(alias, &document.key, query) {
                let length = alias.chars().count();
                return Some(format!("alias: {}", excerpt(alias, 0, length, max_chars)));
            }
        }
        None
    }

    /// The query terms worth quoting as evidence, best first.
    ///
    /// Mirrors Python's `_informative_terms`: terms held by at most 60% of the corpus, and
    /// when every term is that common, the rarest single term.
    fn informative_terms(&self, terms: &[String]) -> Vec<String> {
        if terms.is_empty() {
            return Vec::new();
        }
        let ordered = dedupe(terms.to_vec());
        let threshold = ((self.size * 6) / 10).max(1);
        let posting = |term: &str| self.postings.get(term).copied().unwrap_or(0);
        let mut candidates: Vec<String> = ordered
            .iter()
            .filter(|term| posting(term) <= threshold)
            .cloned()
            .collect();
        if candidates.is_empty() {
            if let Some(rarest) = ordered
                .iter()
                .min_by_key(|term| posting(term))
            {
                candidates.push(rarest.clone());
            }
        }
        candidates.sort_by_key(|term| {
            (
                posting(term),
                ordered.iter().position(|value| value == term).unwrap_or(0),
            )
        });
        candidates
    }
}

/// Collects the branch documents of a subtree, root first.
fn collect_branch_documents(
    node: &RetrieverNode,
    ancestors: &mut Vec<String>,
    out: &mut Vec<LexicalDocument>,
) {
    if node.children.is_empty() {
        return;
    }
    out.push(LexicalDocument {
        key: format!("{BRANCH_DOCUMENT_PREFIX}{}", node.node_id),
        name: node.label.clone(),
        description: node.description.clone(),
        category: ancestors.join(" > "),
        ..LexicalDocument::default()
    });
    let track = node.node_id != ROOT_NODE_ID;
    if track {
        ancestors.push(node.node_id.clone());
    }
    for child in node.children() {
        collect_branch_documents(child, ancestors, out);
    }
    if track {
        ancestors.pop();
    }
}

/// The six normalized fields of a document, in scoring order.
fn document_fields(document: &LexicalDocument) -> [String; 6] {
    [
        document.key.clone(),
        document.name.clone(),
        document.aliases.join("\n"),
        document.description.clone(),
        without_front_matter(&document.body),
        document.category.clone(),
    ]
}

/// The text the matcher scans: key, name, aliases, description, body, category.
fn search_text(document: &LexicalDocument, fields: &[String; 6]) -> String {
    let mut parts: Vec<&str> = vec![fields[0].as_str(), fields[1].as_str()];
    parts.extend(document.aliases.iter().map(String::as_str));
    parts.push(fields[3].as_str());
    parts.push(fields[4].as_str());
    parts.push(fields[5].as_str());
    parts.join("\n")
}

/// The weighted term frequencies over the key/name/aliases/description/body fields.
fn weighted_frequencies(tokens: &[Vec<String>; 6]) -> BTreeMap<String, usize> {
    let mut frequencies: BTreeMap<String, usize> = BTreeMap::new();
    for (field, weight) in tokens.iter().zip(FREQUENCY_FIELD_WEIGHTS) {
        let mut counts: BTreeMap<&String, usize> = BTreeMap::new();
        for token in field {
            *counts.entry(token).or_insert(0) += 1;
        }
        for (token, count) in counts {
            *frequencies.entry(token.clone()).or_insert(0) += count * weight;
        }
    }
    frequencies
}

/// Deduplicates `values`, keeping the first occurrence's order.
fn dedupe(values: Vec<String>) -> Vec<String> {
    let mut seen: BTreeSet<String> = BTreeSet::new();
    let mut deduped = Vec::with_capacity(values.len());
    for value in values {
        if seen.insert(value.clone()) {
            deduped.push(value);
        }
    }
    deduped
}

/// Replaces the query's pattern metacharacters with spaces before ranking tokens are taken.
fn ranking_query(query: &str) -> String {
    query
        .chars()
        .map(|character| {
            if matches!(
                character,
                '|' | '(' | ')' | '[' | ']' | '{' | '}' | '^' | '$' | '*' | '?' | '\\'
            ) {
                ' '
            } else {
                character
            }
        })
        .collect()
}

/// Whether a phrase alternative may be compared as an identity (the reference's
/// `[\w .+/#-]+` filter, `\w` being the alphanumeric-plus-underscore class).
fn identity_alternative_allowed(part: &str) -> bool {
    part.chars().all(|character| {
        character.is_alphanumeric()
            || matches!(character, '_' | ' ' | '.' | '+' | '/' | '#' | '-')
    })
}

/// Normalizes an identity value for tier comparison.
///
/// Lowercases, splits on whitespace and `_./-` runs, and rejoins with single spaces — the
/// reference's `_normalize_identity`, with `to_lowercase()` standing in for `casefold()`.
#[must_use]
pub fn normalize_identity(value: &str) -> String {
    value
        .to_lowercase()
        .split(|character: char| {
            character.is_whitespace() || matches!(character, '_' | '.' | '/' | '-')
        })
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Whether the identity `value` matches `query`, excluding the value's own canonical form.
fn identity_value_matches_query(value: &str, key: &str, query: &str) -> bool {
    let identity = normalize_identity(value);
    let canonical = normalize_identity(key);
    let requested = normalize_identity(query);
    if identity.is_empty() || identity == canonical || requested.is_empty() {
        return false;
    }
    if identity == requested
        || (identity.chars().count() >= 2 && requested.contains(identity.as_str()))
    {
        return true;
    }
    requested.chars().count() >= 2 && identity.contains(requested.as_str())
}

/// Strips a leading YAML front-matter block (`---` … `---`) from `body`.
///
/// Only `"\n"` separates lines (Python's `splitlines` also splits on `\r` and a few
/// Unicode separators); a body without a complete front-matter block is returned unchanged.
#[must_use]
pub fn without_front_matter(body: &str) -> String {
    let lines: Vec<&str> = body.split_inclusive('\n').collect();
    if lines.first().is_none_or(|line| line.trim() != "---") {
        return body.to_owned();
    }
    for (index, line) in lines.iter().enumerate().skip(1) {
        if line.trim() == "---" {
            return lines[index + 1..].concat();
        }
    }
    body.to_owned()
}

/// A bounded excerpt of `body` around `start..end`, measured in characters.
fn excerpt(body: &str, start: usize, end: usize, max_chars: usize) -> String {
    let characters: Vec<char> = body.chars().collect();
    let span = end.saturating_sub(start);
    let half = (max_chars.saturating_sub(span) / 2).max(1);
    let left = start.saturating_sub(half);
    let right = end.saturating_add(half).min(characters.len());
    if left >= right {
        return String::new();
    }
    let slice: String = characters[left..right].iter().collect();
    let mut text = slice.split_whitespace().collect::<Vec<_>>().join(" ");
    if left > 0 {
        text = format!("…{text}");
    }
    if right < characters.len() {
        text = format!("{text}…");
    }
    text.chars().take(max_chars).collect()
}

/// The excerpt that covers the most of `terms`, or a bounded head excerpt.
fn matched_excerpt(value: &str, terms: &BTreeSet<String>, max_chars: usize) -> String {
    let characters: Vec<char> = value.chars().collect();
    let mut best = String::new();
    let mut best_coverage: Option<usize> = None;
    for (start, end) in word_spans(value) {
        let occurrence: String = characters[start..end].iter().collect();
        let occurrence_tokens = tokenize(&occurrence);
        if !terms.iter().any(|term| occurrence_tokens.contains(term)) {
            continue;
        }
        let candidate = excerpt(value, start, end, max_chars);
        let candidate_tokens = tokenize(&candidate);
        let coverage = terms
            .iter()
            .filter(|term| candidate_tokens.contains(*term))
            .count();
        if best_coverage.is_none_or(|best_seen| coverage > best_seen) {
            best = candidate;
            best_coverage = Some(coverage);
        }
        if coverage == terms.len() {
            break;
        }
    }
    if best.is_empty() {
        let length = characters.len().min(max_chars);
        excerpt(value, 0, length, max_chars)
    } else {
        best
    }
}

/// The character spans of every `_WORD_RE` match in `value`.
fn word_spans(value: &str) -> Vec<(usize, usize)> {
    let characters: Vec<char> = value.chars().collect();
    let mut spans = Vec::new();
    let mut index = 0_usize;
    while index < characters.len() {
        let character = characters[index];
        if character.is_ascii_alphanumeric() {
            let start = index;
            index += 1;
            while index < characters.len() && is_word_continuation(characters[index]) {
                index += 1;
            }
            spans.push((start, index));
        } else if is_cjk(character) {
            let start = index;
            while index < characters.len() && is_cjk(characters[index]) {
                index += 1;
            }
            spans.push((start, index));
        } else {
            index += 1;
        }
    }
    spans
}

/// Whether `character` continues a `[A-Za-z0-9_+.#/-]` run.
fn is_word_continuation(character: char) -> bool {
    character.is_ascii_alphanumeric() || matches!(character, '_' | '+' | '.' | '#' | '/' | '-')
}

/// Whether `character` is in the CJK range the reference tokenizes (`U+3400..=U+9FFF`).
fn is_cjk(character: char) -> bool {
    ('\u{3400}'..='\u{9fff}').contains(&character)
}

/// Splits `value` into surface tokens, mirroring the reference's `_surface_tokens`.
fn surface_tokens(value: &str) -> Vec<String> {
    let characters: Vec<char> = value.chars().collect();
    let mut terms: Vec<String> = Vec::new();
    let mut index = 0_usize;
    while index < characters.len() {
        let character = characters[index];
        if character.is_ascii_alphanumeric() {
            let start = index;
            index += 1;
            while index < characters.len() && is_word_continuation(characters[index]) {
                index += 1;
            }
            push_surface(&characters[start..index], &mut terms);
        } else if is_cjk(character) {
            let start = index;
            while index < characters.len() && is_cjk(characters[index]) {
                index += 1;
            }
            push_surface(&characters[start..index], &mut terms);
        } else {
            index += 1;
        }
    }
    terms
}

/// Normalizes one raw `_WORD_RE` match into zero or more surface terms.
fn push_surface(raw: &[char], terms: &mut Vec<String>) {
    let raw: String = raw.iter().collect();
    let folded = raw.to_lowercase();
    let folded = folded
        .trim_matches(|character: char| matches!(character, '.' | '_' | '-' | '/'))
        .to_owned();
    if folded.is_empty() {
        return;
    }
    let characters: Vec<char> = folded.chars().collect();
    if characters.iter().all(|character| is_cjk(*character)) {
        terms.push(folded.clone());
        for index in 0..characters.len().saturating_sub(1) {
            terms.push(characters[index..index + 2].iter().collect());
        }
        return;
    }
    if is_symbol_token(&folded) {
        terms.push(folded);
        return;
    }
    for part in folded.split(['_', '+', '.', '#', '/', '-']) {
        if part.chars().count() > 1
            || (!part.is_empty() && part.chars().all(|character| character.is_ascii_digit()))
        {
            terms.push(part.to_owned());
        }
    }
}

/// Whether `value` is a whole symbol token (`[a-z0-9]+(?:\+\+|#)`).
fn is_symbol_token(value: &str) -> bool {
    let core = value
        .strip_suffix("++")
        .or_else(|| value.strip_suffix('#'));
    let Some(core) = core else {
        return false;
    };
    !core.is_empty()
        && core
            .chars()
            .all(|character| character.is_ascii_lowercase() || character.is_ascii_digit())
}

/// Splits `text` into index terms, mirroring the reference's `_tokens`.
///
/// Every surface token appears once, and an ASCII token also contributes its singular form
/// and its verb derivations — so `skills` indexes `skill`, and `running` indexes `runn` and
/// `runne`, exactly like the reference.
#[must_use]
pub fn tokenize(text: &str) -> Vec<String> {
    let mut tokens: Vec<String> = Vec::new();
    for part in surface_tokens(text) {
        tokens.push(part.clone());
        if !part.is_ascii() {
            continue;
        }
        let singular = singular(&part);
        if singular != part {
            tokens.push(singular.clone());
        }
        for form in verb_forms(&part) {
            if form != part && form != singular {
                tokens.push(form);
            }
        }
    }
    tokens
}

/// The singular form of an ASCII token, mirroring the reference's `_singular`.
fn singular(token: &str) -> String {
    let length = token.chars().count();
    if length <= 3
        || token.ends_with("ss")
        || token.ends_with("us")
        || token.ends_with("is")
        || token.ends_with("series")
        || token.ends_with("species")
    {
        return token.to_owned();
    }
    if let Some(stem) = token.strip_suffix("ies") {
        return format!("{stem}y");
    }
    if ["ches", "shes", "xes", "zes", "sses"]
        .iter()
        .any(|suffix| token.ends_with(suffix))
    {
        return take_chars(token, length - 2);
    }
    if let Some(stem) = token.strip_suffix('s') {
        return stem.to_owned();
    }
    token.to_owned()
}

/// The verb derivations of an ASCII token, mirroring the reference's `_verb_forms`.
fn verb_forms(token: &str) -> Vec<String> {
    if token.chars().count() <= 5 {
        return Vec::new();
    }
    let stem = if let Some(stem) = token.strip_suffix("ing") {
        stem.to_owned()
    } else if let Some(stem) = token.strip_suffix("ied") {
        return vec![format!("{stem}y")];
    } else if let Some(stem) = token.strip_suffix("ed") {
        stem.to_owned()
    } else {
        return Vec::new();
    };
    let stem_characters: Vec<char> = stem.chars().collect();
    let mut stems = vec![stem.clone()];
    let last = stem_characters.len();
    if last > 2
        && stem_characters[last - 1] == stem_characters[last - 2]
        && !stem.ends_with("ss")
    {
        stems.push(take_chars(&stem, last - 1));
    }
    let mut forms: Vec<String> = stems.clone();
    forms.extend(stems.iter().map(|value| format!("{value}e")));
    dedupe(forms)
}

/// The first `count` characters of `text`.
fn take_chars(text: &str, count: usize) -> String {
    text.chars().take(count).collect()
}

/// A bounded, regex-free matcher over one search expression.
///
/// Built by [`compile_matcher`]; see the module docs for the supported subset.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct QueryMatcher {
    /// The alternatives as written (trimmed, quotes stripped).
    alternatives: Vec<String>,
    /// The compiled alternative patterns.
    patterns: Vec<Vec<PatternSegment>>,
    /// Whether any alternative contains a wildcard.
    literal: bool,
    /// Whether matching folds case.
    case_insensitive: bool,
}

/// One segment of a compiled alternative.
#[derive(Debug, Clone, PartialEq, Eq)]
enum PatternSegment {
    /// A literal fragment that must appear, in order.
    Literal(String),
    /// An "any short run" wildcard (the `.`/`.*` wildcard of the reference).
    AnyRun,
}

impl QueryMatcher {
    /// Whether `text` matches any alternative.
    #[must_use]
    pub fn matches(&self, text: &str) -> bool {
        let haystack = if self.case_insensitive {
            text.to_lowercase()
        } else {
            text.to_owned()
        };
        self.patterns
            .iter()
            .any(|pattern| pattern_matches(&haystack, pattern))
    }

    /// Whether the expression has no wildcard, so every match is a plain substring test.
    #[must_use]
    pub fn is_literal(&self) -> bool {
        self.literal
    }

    /// The alternatives the expression was split into, in order.
    #[must_use]
    pub fn alternatives(&self) -> &[String] {
        &self.alternatives
    }
}

/// Whether one compiled alternative matches `haystack` (already lowercased when needed).
fn pattern_matches(haystack: &str, pattern: &[PatternSegment]) -> bool {
    let mut has_literal = false;
    let mut position = 0_usize;
    for segment in pattern {
        match segment {
            PatternSegment::AnyRun => {}
            PatternSegment::Literal(literal) => {
                has_literal = true;
                let Some(found) = haystack.get(position..).and_then(|rest| rest.find(literal.as_str()))
                else {
                    return false;
                };
                position += found + literal.len();
            }
        }
    }
    // A wildcard-only pattern needs at least one character, like the reference's `.`.
    has_literal || !haystack.is_empty()
}

/// Compiles a bounded matcher over `query`.
///
/// The supported subset:
///
/// - literal substrings (`fixed_strings` makes the whole query one literal, quotes,
///   wildcards and `|` included);
/// - `|`-separated alternatives, where any alternative matching wins (`|` inside quotes is
///   literal);
/// - double-quoted phrases, matched literally with the quotes stripped;
/// - `.` and `.*` as "any short run" wildcards between literal fragments.
///
/// Every other regex metacharacter (`+ ? * [ ] ( ) { } ^ $ \`) is matched **literally**, and
/// no character class, quantifier, anchor, group or backreference is interpreted — see the
/// module docs for why.
///
/// # Errors
///
/// [`Error::Validation`](crate::error::Error) naming `query` when it is blank or longer than
/// [`MAX_QUERY_CHARS`] characters (the reference's `compile_safe_pattern` bound).
pub fn compile_matcher(
    query: &str,
    case_insensitive: bool,
    fixed_strings: bool,
) -> Result<QueryMatcher> {
    let text = query.trim();
    let length = text.chars().count();
    if length == 0 || length > MAX_QUERY_CHARS {
        return Err(validation(
            "query",
            format!("search expression must contain 1-{MAX_QUERY_CHARS} characters"),
        ));
    }
    let alternatives = if fixed_strings {
        vec![text.to_owned()]
    } else {
        split_alternatives(text)
    };
    let patterns: Vec<Vec<PatternSegment>> = alternatives
        .iter()
        .map(|alternative| compile_pattern(alternative, fixed_strings))
        .collect();
    let literal = patterns.iter().all(|pattern| {
        !pattern
            .iter()
            .any(|segment| matches!(segment, PatternSegment::AnyRun))
    });
    Ok(QueryMatcher {
        alternatives,
        patterns,
        literal,
        case_insensitive,
    })
}

/// Splits an expression on top-level `|`, stripping double quotes.
fn split_alternatives(query: &str) -> Vec<String> {
    let mut parts: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    for character in query.chars() {
        match character {
            '"' => in_quotes = !in_quotes,
            '|' if !in_quotes => parts.push(std::mem::take(&mut current)),
            other => current.push(other),
        }
    }
    parts.push(current);
    parts
        .into_iter()
        .map(|part| part.trim().to_owned())
        .filter(|part| !part.is_empty())
        .collect()
}

/// Compiles one alternative into literal fragments separated by wildcards.
fn compile_pattern(alternative: &str, fixed_strings: bool) -> Vec<PatternSegment> {
    if fixed_strings {
        return vec![PatternSegment::Literal(alternative.to_owned())];
    }
    let mut segments: Vec<PatternSegment> = Vec::new();
    let mut literal = String::new();
    for character in alternative.chars() {
        if character == '.' || character == '*' {
            if !literal.is_empty() {
                segments.push(PatternSegment::Literal(std::mem::take(&mut literal)));
            }
            if !matches!(segments.last(), Some(PatternSegment::AnyRun)) {
                segments.push(PatternSegment::AnyRun);
            }
        } else {
            literal.push(character);
        }
    }
    if !literal.is_empty() {
        segments.push(PatternSegment::Literal(literal));
    }
    segments
}

impl fmt::Display for LexicalHit {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{} ({:.3})", self.key, self.score)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    fn documents() -> Vec<LexicalDocument> {
        vec![
            LexicalDocument::new("skills.alpha", "Alpha skill")
                .with_description("handles alpha tasks")
                .with_category("Skills > Data"),
            LexicalDocument::new("skills.beta", "Beta skill")
                .with_description("alpha alpha alpha mentions")
                .with_body("alpha body alpha")
                .with_category("Skills > Data"),
            LexicalDocument::new("skills.gamma", "Gamma skill")
                .with_description("unrelated")
                .with_category("Skills > Writing"),
        ]
    }

    fn keys(hits: &[LexicalHit]) -> Vec<&str> {
        hits.iter().map(|hit| hit.key.as_str()).collect()
    }

    #[test]
    fn tokenize_handles_ascii_symbols_punctuation_and_cjk() {
        assert_eq!(tokenize("Data-Flow"), vec!["data", "flow"]);
        assert_eq!(tokenize("a b"), Vec::<String>::new(), "single letters are dropped");
        assert_eq!(tokenize("c++ c#"), vec!["c++", "c#"]);
        assert_eq!(
            tokenize("skills.io_v2"),
            vec!["skills", "skill", "io", "v2"]
        );
        assert_eq!(tokenize("snake_case"), vec!["snake", "case"]);
        assert_eq!(tokenize("x1"), vec!["x1"]);
        assert_eq!(tokenize("+++---"), Vec::<String>::new());
        assert_eq!(tokenize("..."), Vec::<String>::new());
        // A CJK run stays whole and also yields its overlapping bigrams.
        assert_eq!(tokenize("能力树"), vec!["能力树", "能力", "力树"]);
        assert_eq!(tokenize("技能 skill"), vec!["技能", "技能", "skill"]);
        // Case folding and edge trimming.
        assert_eq!(tokenize(" .Alpha. "), vec!["alpha"]);
    }

    #[test]
    fn tokenize_adds_singular_and_verb_forms() {
        assert_eq!(tokenize("skills"), vec!["skills", "skill"]);
        assert_eq!(tokenize("queries"), vec!["queries", "query"]);
        assert_eq!(tokenize("watches"), vec!["watches", "watch"]);
        assert_eq!(
            tokenize("running"),
            vec!["running", "runn", "run", "runne", "rune"]
        );
        assert_eq!(tokenize("applied"), vec!["applied", "apply"]);
        assert_eq!(
            tokenize("stopped"),
            vec!["stopped", "stopp", "stop", "stoppe", "stope"]
        );
        // Short or protected tokens keep their form.
        assert_eq!(tokenize("bus"), vec!["bus"]);
        assert_eq!(tokenize("ss"), vec!["ss"]);
    }

    #[test]
    fn front_matter_is_stripped_only_when_complete() {
        assert_eq!(without_front_matter("---\nkey: v\n---\nbody\n"), "body\n");
        assert_eq!(without_front_matter("---\nkey: v\nbody\n"), "---\nkey: v\nbody\n");
        assert_eq!(without_front_matter("plain"), "plain");
        assert_eq!(without_front_matter(""), "");
        assert_eq!(without_front_matter("---\n---\n"), "");
    }

    #[test]
    fn bm25_saturates_and_rewards_rarer_terms() {
        let index = LexicalIndex::build(vec![
            LexicalDocument::new("a", "alpha"),
            LexicalDocument::new("b", "beta"),
            LexicalDocument::new("c", "gamma"),
        ]);
        let rare = index.search("gamma").unwrap();
        assert_eq!(keys(&rare), vec!["c"]);
        // A rare term beats a duplicated common one.
        let index = LexicalIndex::build(vec![
            LexicalDocument::new("common", "common").with_body("common common common rare"),
            LexicalDocument::new("rare", "rare"),
        ]);
        let hits = index.search("common|rare").unwrap();
        let common = hits.iter().find(|hit| hit.key == "common").unwrap().score;
        let rare_hit = hits.iter().find(|hit| hit.key == "rare").unwrap().score;

        // Term frequency contributes, but sub-linearly: the 3x repeated term does not
        // score 3x the single term.
        let single = LexicalIndex::build(vec![LexicalDocument::new("x", "common")]);
        let one = single.search("common").unwrap()[0].score;
        let repeated = LexicalIndex::build(vec![
            LexicalDocument::new("x", "common").with_body("common")
        ]);
        let two = repeated.search("common").unwrap()[0].score;
        assert!(two > one, "{two} should exceed {one}");
        assert!(two < one * 2.0, "{two} should saturate below 2x {one}");
        assert!(common > 0.0 && rare_hit > 0.0);
    }

    #[test]
    fn a_name_hit_outranks_a_description_only_hit() {
        let index = LexicalIndex::build(documents());
        let hits = index.search("alpha").unwrap();
        assert_eq!(keys(&hits), vec!["skills.alpha", "skills.beta"]);
        assert!(hits[0].score > hits[1].score);
    }

    #[test]
    fn identity_tiers_order_exact_keys_before_prefixes_and_bodies() {
        let index = LexicalIndex::build(vec![
            LexicalDocument::new("skills.alpha", "Alpha").with_body("alpha tools"),
            LexicalDocument::new("alpha", "Something else"),
            LexicalDocument::new("anything", "Alpha thing"),
            LexicalDocument::new("other", "Other").with_description("alpha"),
        ]);
        let hits = index.search("alpha").unwrap();
        assert_eq!(
            keys(&hits),
            vec!["alpha", "skills.alpha", "anything", "other"],
            "0 key, 1/2 name, 3 prefix, 4 otherwise"
        );
    }

    #[test]
    fn scores_tie_break_on_the_key() {
        let index = LexicalIndex::build(vec![
            LexicalDocument::new("skills.b", "Zeta"),
            LexicalDocument::new("skills.a", "Zeta"),
        ]);
        let hits = index.search("zeta").unwrap();
        assert_eq!(keys(&hits), vec!["skills.a", "skills.b"]);
        assert_eq!(hits[0].score, hits[1].score);
    }

    #[test]
    fn phrase_scores_follow_the_field_weights() {
        let index = LexicalIndex::build(vec![
            LexicalDocument::new("name", "Data Flow"),
            LexicalDocument::new("description", "Other").with_description("a data flow tool"),
            LexicalDocument::new("body", "Other").with_body("a data flow tool"),
            LexicalDocument::new("category", "Other").with_category("Data Flow"),
        ]);
        let hits = index.search("data flow").unwrap();
        assert_eq!(
            keys(&hits),
            vec!["name", "description", "body", "category"],
            "12 > 5 > 0.5 > 0.1"
        );
    }

    #[test]
    fn pipe_separated_alternatives_and_quoting_work_in_the_matcher() {
        let matcher = compile_matcher("alpha|beta", true, false).unwrap();
        assert_eq!(matcher.alternatives(), ["alpha", "beta"]);
        assert!(matcher.is_literal());
        assert!(matcher.matches("an ALPHA value"));
        assert!(matcher.matches("a beta value"));
        assert!(!matcher.matches("gamma"));
        assert!(!matcher.matches(""));

        let quoted = compile_matcher("\"data flow\"", true, false).unwrap();
        assert_eq!(quoted.alternatives(), ["data flow"]);
        assert!(quoted.matches("the data flow here"));
        assert!(!quoted.matches("data and flow"));

        let fixed = compile_matcher("a|b.*c", true, true).unwrap();
        assert!(fixed.is_literal());
        assert!(fixed.matches("a|b.*c"));
        assert!(!fixed.matches("axc"));
    }

    #[test]
    fn regex_metacharacters_are_literal_and_wildcards_are_linear() {
        let matcher = compile_matcher("a.c", true, false).unwrap();
        assert!(!matcher.is_literal());
        assert!(matcher.matches("abc"), "the dot is a one-run wildcard");
        assert!(matcher.matches("a-long-run-c"));
        assert!(matcher.matches("a.c"), "a literal dot still matches the wildcard");
        assert!(
            matcher.matches("ac"),
            "the wildcard may span an empty run"
        );

        // Classes, quantifiers, anchors and groups are literal characters.
        for query in ["[abc]+", "^start$", "(group)", "a{2,3}", "back\\1", "a+?"] {
            let matcher = compile_matcher(query, true, false).unwrap();
            assert!(
                matcher.matches(&format!("x {query} y")),
                "{query} must match itself"
            );
        }
        let star = compile_matcher("*", true, false).unwrap();
        assert!(star.matches("anything"));
        assert!(!star.matches(""), "a wildcard-only pattern needs a character");

        assert!(compile_matcher("   ", true, false).is_err());
        let long = "a".repeat(MAX_QUERY_CHARS + 1);
        assert!(compile_matcher(&long, true, false).is_err());
    }

    #[test]
    fn snippet_bounds_are_respected_and_never_split_a_character() {
        let body = format!("{}能力树索引构建{}", "x".repeat(60), "y".repeat(60));
        let index = LexicalIndex::build(vec![LexicalDocument::new("k", "K").with_body(body)]);
        let snippet = index.match_snippet("k", &["能力树"], 0, 20);
        assert!(snippet.starts_with("body: "), "{snippet}");
        let text = snippet.trim_start_matches("body: ");
        assert!(text.chars().count() <= 20, "{snippet}");
        assert!(snippet.contains('能'));

        let description = LexicalDocument::new("d", "D").with_description(format!(
            "{} alpha {}",
            "a".repeat(30),
            "b".repeat(200)
        ));
        let index = LexicalIndex::build(vec![description]);
        let snippet = index.match_snippet("d", &["alpha"], 0, 24);
        assert!(snippet.starts_with("description: "), "{snippet}");
        assert!(snippet.chars().count() <= 24 + "description: ".len());
        assert_eq!(index.match_snippet("d", &["absent"], 0, 24), "");
        assert_eq!(index.match_snippet("missing", &["alpha"], 0, 24), "");
        assert_eq!(index.match_snippet("d", &["alpha"], 0, 0), "");
        assert_eq!(index.match_snippet("d", &[], 0, 24), "");
    }

    #[test]
    fn a_name_or_alias_match_wins_the_snippet() {
        let index = LexicalIndex::build(vec![LexicalDocument::new("skills.flow", "Flow Builder")
            .with_aliases(vec!["pipe".to_owned()])
            .with_description("builds things")]);
        assert_eq!(
            index.match_snippet("skills.flow", &["flow builder"], 180, 40),
            "name: Flow Builder"
        );
        assert!(index
            .match_snippet("skills.flow", &["pipe"], 180, 40)
            .starts_with("alias: pipe"));
        assert!(index
            .match_snippet("skills.flow", &["flow"], 180, 40)
            .starts_with("name: "));
    }

    #[test]
    fn missing_terms_and_informative_terms_are_deterministic() {
        let index = LexicalIndex::build(documents());
        assert_eq!(index.missing_terms("alpha zeta"), vec!["zeta".to_owned()]);
        assert_eq!(index.missing_terms(""), Vec::<String>::new());
        assert_eq!(index.missing_terms("skill"), Vec::<String>::new());

        let terms = vec!["alpha".to_owned(), "skill".to_owned(), "alpha".to_owned()];
        // Neither term is rare enough, so the rarest single term is the evidence.
        assert_eq!(index.informative_terms(&terms), vec!["alpha".to_owned()]);
    }

    #[test]
    fn indexes_are_deterministic_and_order_independent() {
        let forward = LexicalIndex::build(documents());
        let mut reversed = documents();
        reversed.reverse();
        let backward = LexicalIndex::build(reversed);
        assert_eq!(
            forward.search("alpha data").unwrap(),
            backward.search("alpha data").unwrap()
        );
        assert_eq!(forward.len(), 3);
        assert!(!forward.is_empty());
        assert!(LexicalIndex::build(Vec::new()).is_empty());
        assert_eq!(LexicalIndex::build(Vec::new()).search("q").unwrap(), vec![]);
        // A duplicated key keeps the last document.
        let index = LexicalIndex::build(vec![
            LexicalDocument::new("k", "First"),
            LexicalDocument::new("k", "Second"),
        ]);
        assert_eq!(index.document("k").unwrap().name, "Second");
        assert_eq!(index.len(), 1);
    }

    #[test]
    fn search_in_scopes_the_corpus_and_rejects_a_blank_query() {
        let index = LexicalIndex::build(documents());
        let scoped = index.search_in("alpha", &["skills.beta", "unknown"]).unwrap();
        assert_eq!(keys(&scoped), vec!["skills.beta"]);
        assert_eq!(index.search_in("alpha", &["unknown"]).unwrap(), vec![]);
        assert_eq!(
            index.search_in("alpha", &[]).unwrap(),
            index.search("alpha").unwrap(),
            "an empty scope ranks the whole corpus"
        );
        let error = index.search("   ").unwrap_err();
        assert!(error.message().contains("query"), "{}", error.message());
        assert!(index.search_in("x", &[]).is_ok());
    }

    #[test]
    fn hits_render_with_their_score() {
        let hit = LexicalHit {
            key: "k".to_owned(),
            score: 1.5,
        };
        assert_eq!(hit.to_string(), "k (1.500)");
        let json = serde_json::to_value(&hit).unwrap();
        assert_eq!(json["key"], "k");
        let parsed: LexicalHit = serde_json::from_value(json).unwrap();
        assert_eq!(parsed, hit);
    }
}
