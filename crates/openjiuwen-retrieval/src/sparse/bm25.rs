//! A self-contained BM25 lexical index.
//!
//! Python's sparse retriever hands `sparse_search` to the vector database, so
//! this module has no Python counterpart: the scoring math is implemented here
//! over an in-process inverted index. Scoring follows the standard Okapi BM25
//! formula with term-frequency saturation (`k1`) and document-length
//! normalisation (`b`):
//!
//! ```text
//! idf(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
//! score(d, q) = Σ_{t ∈ q ∩ d} idf(t) · tf(t, d) · (k1 + 1)
//!                                   / (tf(t, d) + k1 · (1 - b + b · |d| / avgdl))
//! ```
//!
//! Scores are raw BM25 values: finite, non-negative, and only comparable within
//! one index.

use std::collections::{BTreeSet, HashMap, HashSet};

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use openjiuwen_common::error::{Error, Result};

use crate::common::{Document, Metadata, MetadataFilter, SearchResult, TextChunk, DEFAULT_TOP_K};
use crate::vector_store::{CHUNK_ID_FIELD, DOC_ID_FIELD};

/// Default BM25 term-frequency saturation (`k1`).
pub const DEFAULT_K1: f32 = 1.2;

/// Default BM25 document-length normalisation (`b`).
pub const DEFAULT_B: f32 = 0.75;

/// The stopword list applied when no other list is configured.
///
/// A small English list on purpose: it removes the most common function words
/// without changing ranking on domain vocabulary.
pub const DEFAULT_STOPWORDS: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is", "it", "of", "on",
    "or", "that", "the", "this", "to", "was", "were", "with",
];

fn default_k1() -> f32 {
    DEFAULT_K1
}

fn default_b() -> f32 {
    DEFAULT_B
}

fn default_top_k() -> usize {
    DEFAULT_TOP_K
}

/// BM25 scoring parameters.
///
/// `k1` must be finite and non-negative; `b` must be finite and inside
/// `[0, 1]`. `b = 0` disables document-length normalisation entirely.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Bm25Config {
    /// Term-frequency saturation; `0` makes every matched term contribute `idf`.
    #[serde(default = "default_k1")]
    pub k1: f32,
    /// Document-length normalisation strength in `[0, 1]`.
    #[serde(default = "default_b")]
    pub b: f32,
}

impl Default for Bm25Config {
    fn default() -> Self {
        Self {
            k1: DEFAULT_K1,
            b: DEFAULT_B,
        }
    }
}

impl Bm25Config {
    /// Creates a configuration from explicit parameters (see [`Self::validate`]).
    pub const fn new(k1: f32, b: f32) -> Self {
        Self { k1, b }
    }

    /// Checks that `k1 >= 0` and `b ∈ [0, 1]`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] naming the offending parameter and value.
    pub fn validate(&self) -> Result<()> {
        if !self.k1.is_finite() || self.k1 < 0.0 {
            return Err(Error::validation(format!(
                "BM25 k1 must be a finite value >= 0.0, got {}",
                self.k1
            )));
        }
        if !self.b.is_finite() || !(0.0..=1.0).contains(&self.b) {
            return Err(Error::validation(format!(
                "BM25 b must be a finite value in [0.0, 1.0], got {}",
                self.b
            )));
        }
        Ok(())
    }
}

/// A lexical (BM25) search request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Bm25Query {
    /// Query text; tokenized with [`tokenize`] and reduced by the stopword list.
    #[serde(default)]
    pub text: String,
    /// Maximum number of results to return.
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    /// Optional metadata predicate applied before scoring.
    #[serde(default)]
    pub filter: Option<MetadataFilter>,
    /// Optional inclusive minimum score (`score >= threshold`).
    #[serde(default)]
    pub score_threshold: Option<f32>,
}

impl Default for Bm25Query {
    fn default() -> Self {
        Self {
            text: String::new(),
            top_k: DEFAULT_TOP_K,
            filter: None,
            score_threshold: None,
        }
    }
}

impl Bm25Query {
    /// Creates a query for `text` with [`DEFAULT_TOP_K`] results and no filter.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            ..Self::default()
        }
    }

    /// Sets the maximum number of results.
    #[must_use]
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }

    /// Sets the metadata predicate.
    #[must_use]
    pub fn with_filter(mut self, filter: MetadataFilter) -> Self {
        self.filter = Some(filter);
        self
    }

    /// Sets the inclusive minimum score.
    #[must_use]
    pub fn with_score_threshold(mut self, threshold: f32) -> Self {
        self.score_threshold = Some(threshold);
        self
    }
}

/// Snapshot of an index's size and token accounting.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct Bm25Statistics {
    /// Number of indexed documents (chunks).
    pub document_count: usize,
    /// Vocabulary size: distinct terms surviving the stopword list.
    pub term_count: usize,
    /// Mean document length in tokens; `0.0` for an empty index.
    pub average_document_length: f32,
    /// Total number of indexed tokens across all documents.
    pub total_tokens: usize,
}

/// One indexed document: payload plus its pre-computed term statistics.
#[derive(Debug, Clone)]
struct IndexedDocument {
    text: String,
    metadata: Metadata,
    doc_id: Option<String>,
    term_frequencies: HashMap<String, usize>,
    length: usize,
}

/// An in-process BM25 inverted index.
///
/// Documents keep their insertion order, which is the tie-break for equal
/// scores. `upsert` replaces the payload of an existing id without moving it, so
/// re-indexing a document never changes the order of unrelated results.
#[derive(Debug, Clone)]
pub struct Bm25Index {
    config: Bm25Config,
    stopwords: HashSet<String>,
    documents: IndexMap<String, IndexedDocument>,
    document_frequency: HashMap<String, usize>,
    total_tokens: usize,
}

impl Default for Bm25Index {
    fn default() -> Self {
        Self {
            config: Bm25Config::default(),
            stopwords: default_stopwords(),
            documents: IndexMap::new(),
            document_frequency: HashMap::new(),
            total_tokens: 0,
        }
    }
}

fn default_stopwords() -> HashSet<String> {
    DEFAULT_STOPWORDS
        .iter()
        .map(|word| (*word).to_owned())
        .collect()
}

impl Bm25Index {
    /// Creates an empty index, validating `config`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the configuration is invalid; see
    /// [`Bm25Config::validate`].
    pub fn new(config: Bm25Config) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            config,
            ..Self::default()
        })
    }

    /// The scoring parameters in use.
    pub fn config(&self) -> &Bm25Config {
        &self.config
    }

    /// Replaces the stopword list.
    ///
    /// Entries are trimmed and lowercased, blank entries are ignored, and the
    /// default list is discarded. Only affects documents indexed afterwards.
    #[must_use]
    pub fn with_stopwords<I, S>(mut self, stopwords: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.stopwords = normalize_stopwords(stopwords);
        self
    }

    /// Disables stopword filtering entirely.
    ///
    /// Only affects documents indexed afterwards.
    #[must_use]
    pub fn without_stopwords(mut self) -> Self {
        self.stopwords.clear();
        self
    }

    /// Whether `id` is indexed.
    pub fn contains(&self, id: &str) -> bool {
        self.documents.contains_key(id)
    }

    /// Indexes a document under `id`, replacing any previous payload.
    pub fn add_document(
        &mut self,
        id: impl Into<String>,
        text: impl Into<String>,
        metadata: Metadata,
    ) {
        let _ = self.insert(id.into(), text.into(), metadata, None);
    }

    /// Indexes every document, keyed by [`Document::id`], and returns how many
    /// were supplied (`0` for an empty slice).
    pub fn add_documents(&mut self, documents: &[Document]) -> usize {
        for document in documents {
            let _ = self.insert(
                document.id.clone(),
                document.text.clone(),
                document.metadata.clone(),
                None,
            );
        }
        documents.len()
    }

    /// Indexes every chunk, keyed by [`TextChunk::id`], recording its parent
    /// `doc_id` so retrieval results carry the document link.
    ///
    /// Returns how many chunks were supplied (`0` for an empty slice).
    pub fn add_chunks(&mut self, chunks: &[TextChunk]) -> usize {
        for chunk in chunks {
            let _ = self.insert(
                chunk.id.clone(),
                chunk.text.clone(),
                chunk.metadata.clone(),
                Some(chunk.doc_id.clone()),
            );
        }
        chunks.len()
    }

    /// Indexes or replaces the document `id`, returning `true` when the id was
    /// new and `false` when an existing document was replaced in place.
    pub fn upsert(
        &mut self,
        id: impl Into<String>,
        text: impl Into<String>,
        metadata: Metadata,
    ) -> bool {
        self.insert(id.into(), text.into(), metadata, None)
    }

    /// Removes `id`, returning whether it existed.
    pub fn remove(&mut self, id: &str) -> bool {
        let Some(document) = self.documents.shift_remove(id) else {
            return false;
        };
        let terms: Vec<String> = document.term_frequencies.keys().cloned().collect();
        self.retract_terms(document.length, &terms);
        true
    }

    /// Number of indexed documents.
    pub fn len(&self) -> usize {
        self.documents.len()
    }

    /// Whether the index holds no documents.
    pub fn is_empty(&self) -> bool {
        self.documents.is_empty()
    }

    /// Removes every document and resets all token accounting.
    pub fn clear(&mut self) {
        self.documents.clear();
        self.document_frequency.clear();
        self.total_tokens = 0;
    }

    /// Current size and token statistics.
    pub fn stats(&self) -> Bm25Statistics {
        let document_count = self.documents.len();
        Bm25Statistics {
            document_count,
            term_count: self.document_frequency.len(),
            average_document_length: if document_count == 0 {
                0.0
            } else {
                self.total_tokens as f32 / document_count as f32
            },
            total_tokens: self.total_tokens,
        }
    }

    /// Inverse document frequency of `term` for this index.
    ///
    /// `term` must already be normalized the way [`tokenize`] normalizes text.
    /// Unseen terms (and every term of an empty index) report the `df = 0` value.
    pub fn idf(&self, term: &str) -> f32 {
        let document_count = self.documents.len();
        let document_frequency = self.document_frequency.get(term).copied().unwrap_or(0);
        inverse_document_frequency(document_count, document_frequency)
    }

    /// Runs a lexical search.
    ///
    /// Only documents matching at least one query term are returned; results are
    /// ordered by descending BM25 score with insertion order breaking ties, then
    /// truncated to [`Bm25Query::top_k`]. Result metadata is the indexed metadata
    /// plus the `doc_id`/`chunk_id` pseudo-fields, mirroring the vector stores.
    pub fn search(&self, query: &Bm25Query) -> Vec<SearchResult> {
        if query.top_k == 0 || self.documents.is_empty() {
            return Vec::new();
        }
        // A BTreeSet deduplicates repeated query terms and keeps summation order
        // stable, so identical queries always produce bit-identical scores.
        let terms: BTreeSet<String> = tokenize(&query.text)
            .into_iter()
            .filter(|term| !self.stopwords.contains(term))
            .collect();
        if terms.is_empty() {
            return Vec::new();
        }

        let document_count = self.documents.len();
        let average_length = if self.total_tokens == 0 {
            1.0
        } else {
            self.total_tokens as f32 / document_count as f32
        };
        let k1 = self.config.k1;
        let b = self.config.b;

        let mut hits: Vec<SearchResult> = Vec::new();
        for (id, document) in &self.documents {
            let metadata = effective_metadata(id, document);
            if query
                .filter
                .as_ref()
                .is_some_and(|filter| !filter.matches(&metadata))
            {
                continue;
            }
            let length = document.length as f32;
            let length_penalty = k1 * (1.0 - b + b * (length / average_length));
            let mut score = 0.0f32;
            let mut matched = false;
            for term in &terms {
                let Some(&frequency) = document.term_frequencies.get(term) else {
                    continue;
                };
                matched = true;
                let tf = frequency as f32;
                let document_frequency = self.document_frequency.get(term).copied().unwrap_or(0);
                let idf = inverse_document_frequency(document_count, document_frequency);
                score += idf * (tf * (k1 + 1.0)) / (tf + length_penalty);
            }
            if !matched {
                continue;
            }
            if query.score_threshold.is_some_and(|minimum| score < minimum) {
                continue;
            }
            hits.push(SearchResult {
                id: id.clone(),
                text: document.text.clone(),
                score,
                metadata,
            });
        }
        // Stable sort: equal scores keep the index's insertion order.
        hits.sort_by(|left, right| right.score.total_cmp(&left.score));
        hits.truncate(query.top_k);
        hits
    }

    /// Stores `id`, returning `true` when the id was new.
    fn insert(
        &mut self,
        id: String,
        text: String,
        metadata: Metadata,
        doc_id: Option<String>,
    ) -> bool {
        let (term_frequencies, length) = self.analyze(&text);
        let is_new = !self.documents.contains_key(&id);
        let previous = self.documents.get(&id).map(previous_statistics);
        if let Some((previous_length, previous_terms)) = previous {
            self.retract_terms(previous_length, &previous_terms);
        }
        self.total_tokens += length;
        for term in term_frequencies.keys() {
            *self.document_frequency.entry(term.clone()).or_insert(0) += 1;
        }
        // IndexMap::insert keeps an existing key in its original position.
        self.documents.insert(
            id,
            IndexedDocument {
                text,
                metadata,
                doc_id,
                term_frequencies,
                length,
            },
        );
        is_new
    }

    /// Tokenizes `text` and keeps the terms that survive the stopword list.
    fn analyze(&self, text: &str) -> (HashMap<String, usize>, usize) {
        let mut term_frequencies: HashMap<String, usize> = HashMap::new();
        let mut length = 0usize;
        for token in tokenize(text) {
            if self.stopwords.contains(&token) {
                continue;
            }
            *term_frequencies.entry(token).or_insert(0) += 1;
            length += 1;
        }
        (term_frequencies, length)
    }

    /// Removes the term statistics of a document that is no longer indexed.
    fn retract_terms(&mut self, length: usize, terms: &[String]) {
        self.total_tokens = self.total_tokens.saturating_sub(length);
        for term in terms {
            let exhausted = match self.document_frequency.get_mut(term) {
                Some(count) => {
                    *count = count.saturating_sub(1);
                    *count == 0
                }
                None => false,
            };
            if exhausted {
                let _ = self.document_frequency.remove(term);
            }
        }
    }
}

/// Length plus term list of an indexed document, captured before mutation.
fn previous_statistics(document: &IndexedDocument) -> (usize, Vec<String>) {
    (
        document.length,
        document.term_frequencies.keys().cloned().collect(),
    )
}

/// Metadata as seen by filters and results: indexed metadata plus ids.
fn effective_metadata(id: &str, document: &IndexedDocument) -> Metadata {
    let mut metadata = document.metadata.clone();
    if !metadata.contains_key(CHUNK_ID_FIELD) {
        metadata.insert(CHUNK_ID_FIELD.to_owned(), Value::String(id.to_owned()));
    }
    if let Some(doc_id) = &document.doc_id {
        if !metadata.contains_key(DOC_ID_FIELD) {
            metadata.insert(DOC_ID_FIELD.to_owned(), Value::String(doc_id.clone()));
        }
    }
    metadata
}

/// Standard BM25 idf; `0.0` for an empty index.
fn inverse_document_frequency(document_count: usize, document_frequency: usize) -> f32 {
    if document_count == 0 {
        return 0.0;
    }
    let n = document_count as f64;
    let df = document_frequency as f64;
    (1.0 + (n - df + 0.5) / (df + 0.5)).ln() as f32
}

/// Trims and lowercases stopwords, dropping blanks.
fn normalize_stopwords<I, S>(stopwords: I) -> HashSet<String>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    stopwords
        .into_iter()
        .map(Into::into)
        .map(|word: String| word.trim().to_lowercase())
        .filter(|word| !word.is_empty())
        .collect()
}

/// Tokenizes `text` into lowercase lexical terms.
///
/// Runs of Unicode alphanumerics form one token, case is folded, and every
/// separator (punctuation, whitespace, symbols) splits tokens. CJK ideographs and
/// kana become one token each because those scripts are not space-delimited,
/// matching the analyzer intent behind Python's `jieba` segmentation. Stopwords
/// are not removed here; [`Bm25Index`] applies its configured list.
pub fn tokenize(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    for character in text.chars() {
        if is_cjk(character) {
            flush_token(&mut current, &mut tokens);
            tokens.push(character.to_string());
        } else if character.is_alphanumeric() {
            for lower in character.to_lowercase() {
                current.push(lower);
            }
        } else {
            flush_token(&mut current, &mut tokens);
        }
    }
    flush_token(&mut current, &mut tokens);
    tokens
}

fn flush_token(current: &mut String, tokens: &mut Vec<String>) {
    if !current.is_empty() {
        tokens.push(std::mem::take(current));
    }
}

/// Whether `character` belongs to a script written without word separators.
fn is_cjk(character: char) -> bool {
    matches!(
        character,
        '\u{3040}'..='\u{30FF}'      // hiragana and katakana
            | '\u{3400}'..='\u{4DBF}' // CJK unified ideographs extension A
            | '\u{4E00}'..='\u{9FFF}' // CJK unified ideographs
            | '\u{F900}'..='\u{FAFF}' // CJK compatibility ideographs
            | '\u{FF66}'..='\u{FF9D}' // half-width katakana
            | '\u{20000}'..='\u{2FA1F}' // CJK extensions B-F and compatibility supplement
    )
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::assert_is_empty
)]
mod tests {
    use super::*;
    use serde_json::json;

    const TOLERANCE: f32 = 1e-5;

    fn index() -> Bm25Index {
        Bm25Index::default()
    }

    fn doc(id: &str, text: &str) -> Document {
        Document::with_id(id, text)
    }

    fn add(index: &mut Bm25Index, id: &str, text: &str) {
        index.add_document(id, text, Metadata::new());
    }

    fn ids(hits: &[SearchResult]) -> Vec<String> {
        hits.iter().map(|hit| hit.id.clone()).collect()
    }

    #[test]
    fn config_validation_rejects_invalid_parameters() {
        let error = Bm25Config::new(-1.0, 0.75).validate().unwrap_err();
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Validation
        );
        assert!(error.to_string().contains("k1"));

        assert!(Bm25Config::new(1.2, 1.5).validate().is_err());
        assert!(Bm25Config::new(1.2, -0.1).validate().is_err());
        assert!(Bm25Config::new(f32::NAN, 0.75).validate().is_err());
        assert!(Bm25Config::new(1.2, f32::NAN).validate().is_err());
        assert!(Bm25Config::default().validate().is_ok());
        assert!(Bm25Config::new(0.0, 0.0).validate().is_ok());

        let index = Bm25Index::new(Bm25Config::new(-2.0, 0.5));
        assert!(matches!(
            index,
            Err(error) if error.kind() == openjiuwen_common::error::ErrorKind::Validation
        ));
    }

    #[test]
    fn tokenizer_lowercases_splits_punctuation_and_splits_cjk() {
        assert_eq!(tokenize("Hello, world!"), vec!["hello", "world"]);
        assert_eq!(tokenize("Rust  RUST"), vec!["rust", "rust"]);
        assert_eq!(tokenize("a_b-c"), vec!["a", "b", "c"]);
        assert!(tokenize("   ").is_empty());
        assert_eq!(tokenize("你好世界"), vec!["你", "好", "世", "界"]);
        assert_eq!(tokenize("Rust语言"), vec!["rust", "语", "言"]);
        assert_eq!(tokenize("café"), vec!["café"]);
    }

    #[test]
    fn default_stopwords_are_filtered_from_search() {
        let mut index = index();
        add(&mut index, "d1", "the quick brown fox");
        let with_stopword = index.search(&Bm25Query::new("the quick"));
        let without_stopword = index.search(&Bm25Query::new("quick"));
        assert_eq!(ids(&with_stopword), ids(&without_stopword));
        assert_eq!(ids(&with_stopword), vec!["d1"]);
        assert_eq!(index.stats().total_tokens, 3);

        let no_stopwords = Bm25Index::default().without_stopwords();
        assert!(no_stopwords.stopwords.is_empty());
        let custom = Bm25Index::default().with_stopwords(["Quick", " "]);
        assert_eq!(custom.stopwords.len(), 1);
        assert!(custom.stopwords.contains("quick"));
    }

    #[test]
    fn statistics_track_documents_terms_and_tokens() {
        let mut index = index();
        assert_eq!(index.stats(), Bm25Statistics::default());
        assert!(index.is_empty());
        assert_eq!(index.idf("anything"), 0.0);

        add(&mut index, "d1", "alpha beta");
        add(&mut index, "d2", "the beta gamma");
        let stats = index.stats();
        assert_eq!(stats.document_count, 2);
        assert_eq!(stats.term_count, 3);
        assert_eq!(stats.total_tokens, 4);
        assert!((stats.average_document_length - 2.0).abs() < TOLERANCE);
        assert_eq!(index.len(), 2);
        assert!(index.contains("d1"));
    }

    #[test]
    fn replacement_keeps_position_and_updates_statistics() {
        let mut index = index();
        assert!(index.upsert("d1", "alpha", Metadata::new()));
        assert!(!index.upsert("d1", "beta gamma", Metadata::new()));
        assert_eq!(index.len(), 1);
        assert_eq!(index.stats().total_tokens, 2);
        assert!(index.search(&Bm25Query::new("alpha")).is_empty());
        assert_eq!(ids(&index.search(&Bm25Query::new("beta"))), vec!["d1"]);
    }

    #[test]
    fn remove_updates_statistics_and_reports_existence() {
        let mut index = index();
        add(&mut index, "d1", "alpha beta");
        add(&mut index, "d2", "alpha");
        assert!(index.remove("d1"));
        assert!(!index.remove("d1"));
        assert_eq!(index.len(), 1);
        assert_eq!(index.stats().total_tokens, 1);
        assert_eq!(index.stats().term_count, 1);
        index.clear();
        assert!(index.is_empty());
        assert_eq!(index.stats(), Bm25Statistics::default());
    }

    #[test]
    fn chunks_and_documents_carry_their_ids() {
        let mut index = index();
        let documents = vec![doc("doc-1", "beta gamma")];
        assert_eq!(index.add_documents(&documents), 1);
        assert_eq!(index.add_documents(&[]), 0);

        let chunk = TextChunk::from_document(&documents[0], "alpha", Some("chunk-1".into()));
        assert_eq!(index.add_chunks(std::slice::from_ref(&chunk)), 1);

        let hits = index.search(&Bm25Query::new("alpha"));
        assert_eq!(ids(&hits), vec!["chunk-1"]);
        assert_eq!(hits[0].metadata.get(DOC_ID_FIELD), Some(&json!("doc-1")));
        assert_eq!(
            hits[0].metadata.get(CHUNK_ID_FIELD),
            Some(&json!("chunk-1"))
        );
    }

    #[test]
    fn idf_matches_the_standard_formula() {
        let mut index = index();
        add(&mut index, "d1", "rare");
        add(&mut index, "d2", "common");
        add(&mut index, "d3", "common");
        add(&mut index, "d4", "common");
        // ln(1 + (4 - 1 + 0.5) / 1.5)
        assert!((index.idf("rare") - 1.203_972_7).abs() < TOLERANCE);
        // ln(1 + (4 - 3 + 0.5) / 3.5)
        assert!((index.idf("common") - 0.356_674_9).abs() < TOLERANCE);
        // df = 0: ln(1 + 4.5 / 0.5) = ln(10)
        assert!((index.idf("unseen") - std::f32::consts::LN_10).abs() < TOLERANCE);
    }
}
