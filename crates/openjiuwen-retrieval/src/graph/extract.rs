//! Deterministic, model-free triple extraction.
//!
//! Port of the extractor interface from the Python
//! `openjiuwen/core/retrieval/indexing/processor/extractor/` package. Python's only
//! implementation (`TripleExtractor`) is LLM-backed; `.py` parity is impossible
//! offline, so this module ships [`RuleTripleExtractor`], a pure-Rust,
//! deterministic OpenIE-style extractor built on a fixed pattern table.
//!
//! Extraction is chunk-major and sentence-major: each sentence is segmented on
//! `.`, `!`, `?`, `;`, and newlines, then matched against the pattern table. The
//! earliest usable pattern occurrence in a sentence wins (ties broken by table
//! order), which keeps the result stable and avoids "later keyword" noise such as
//! `"Acme created a tool such as Hammer"` being read as an `includes` triple.
//! Objects are the remainder of the sentence after the keyword, trimmed of
//! relative clauses (`which`, `that`, `who`, `whose`, `where`), articles, and
//! trailing punctuation.

use async_trait::async_trait;

use openjiuwen_common::error::{Error, Result};

use super::text::{
    capitalised_phrases, cjk_runs, find_keyword, has_entity_hint, quoted_spans, split_object_list,
    split_sentences, strip_leading, strip_prepositional_prefix, trim_object_tail, ARTICLES,
    LEADING_CONNECTORS, LEADING_PUNCTUATION, STOPWORDS, TRAILING_PUNCTUATION,
};
use super::triple::{normalize_component, Entity, Triple};
use crate::common::{Metadata, TextChunk};

/// Default upper bound on triples emitted per chunk.
pub const DEFAULT_MAX_TRIPLES_PER_CHUNK: usize = 16;
/// Default minimum entity length, in characters.
pub const DEFAULT_MIN_ENTITY_CHARS: usize = 2;
/// Default maximum entity length, in characters.
pub const DEFAULT_MAX_ENTITY_CHARS: usize = 80;

/// Longest subject accepted by the rule extractor, in whitespace-separated words.
///
/// Longer prefixes are rejected rather than truncated so a triple never carries a
/// half-clause as its subject.
const MAX_SUBJECT_WORDS: usize = 10;

/// Built-in predicate vocabulary (the default [`RuleExtractionConfig::predicates`]).
pub const BUILT_IN_PREDICATES: &[&str] = &[
    "acquired",
    "appears_in",
    "contains",
    "created",
    "describes",
    "founded_by",
    "has",
    "includes",
    "is_a",
    "owned_by",
    "uses",
    "with",
];

/// Configuration for [`RuleTripleExtractor`].
#[derive(Debug, Clone, PartialEq)]
pub struct RuleExtractionConfig {
    /// Maximum number of triples emitted per chunk (at least 1).
    pub max_triples_per_chunk: usize,
    /// Minimum entity length in characters (at least 1).
    pub min_entity_chars: usize,
    /// Maximum entity length in characters (at least `min_entity_chars`).
    pub max_entity_chars: usize,
    /// Predicates the extractor may emit, lowercased.
    pub predicates: Vec<String>,
    /// Whether entities without a matched pattern emit an `appears_in` triple.
    ///
    /// Defaults to `false`: an entity mention alone is not a relation. When
    /// enabled, a sentence with exactly one entity yields
    /// `(entity, "appears_in", <parent doc id>)`, the provenance anchor used when
    /// a graph should record bare mentions.
    pub include_single_entity: bool,
}

impl Default for RuleExtractionConfig {
    fn default() -> Self {
        Self {
            max_triples_per_chunk: DEFAULT_MAX_TRIPLES_PER_CHUNK,
            min_entity_chars: DEFAULT_MIN_ENTITY_CHARS,
            max_entity_chars: DEFAULT_MAX_ENTITY_CHARS,
            predicates: BUILT_IN_PREDICATES
                .iter()
                .map(|predicate| (*predicate).to_owned())
                .collect(),
            include_single_entity: false,
        }
    }
}

impl RuleExtractionConfig {
    /// Creates the default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the per-chunk triple cap.
    #[must_use]
    pub fn with_max_triples_per_chunk(mut self, max_triples_per_chunk: usize) -> Self {
        self.max_triples_per_chunk = max_triples_per_chunk;
        self
    }

    /// Replaces the allowed predicate vocabulary (entries are lowercased).
    #[must_use]
    pub fn with_predicates<I, S>(mut self, predicates: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.predicates = predicates
            .into_iter()
            .map(|predicate| normalize_component(&predicate.into()))
            .filter(|predicate| !predicate.is_empty())
            .collect();
        self
    }

    /// Enables or disables the `appears_in` fallback for bare entity mentions.
    #[must_use]
    pub fn with_include_single_entity(mut self, include_single_entity: bool) -> Self {
        self.include_single_entity = include_single_entity;
        self
    }

    /// Whether `predicate` may be emitted.
    pub fn allows(&self, predicate: &str) -> bool {
        self.predicates
            .iter()
            .any(|allowed| allowed == &normalize_component(predicate))
    }
}

/// Extracts triples from text chunks.
///
/// Mirrors Python's `Extractor.extract` signature (minus the `**kwargs`), so an
/// LLM-backed implementation can be dropped in later without changing callers.
#[async_trait]
pub trait TripleExtractor: Send + Sync {
    /// Extracts triples from `chunks`, in chunk order.
    ///
    /// An empty chunk list is not an error: it yields an empty vector.
    async fn extract(&self, chunks: &[TextChunk]) -> Result<Vec<Triple>>;
}

/// One row of the deterministic pattern table.
struct Pattern {
    /// Lowercase keyword matched case-insensitively on word boundaries.
    keyword: &'static str,
    /// Predicate emitted for a match.
    predicate: &'static str,
    /// Base confidence in `[0, 1]` before entity-evidence adjustment.
    confidence: f32,
    /// Whether the object is split on `,`, `and`, `or`, and `&`.
    split_list: bool,
}

impl Pattern {
    /// A single-object pattern.
    const fn single(keyword: &'static str, predicate: &'static str, confidence: f32) -> Self {
        Self {
            keyword,
            predicate,
            confidence,
            split_list: false,
        }
    }

    /// A pattern whose object is a list of exemplars.
    const fn list(keyword: &'static str, predicate: &'static str, confidence: f32) -> Self {
        Self {
            keyword,
            predicate,
            confidence,
            split_list: true,
        }
    }
}

/// Pattern table, ordered by specificity inside each family.
///
/// The order only matters for keyword occurrences that start at the same byte
/// offset (for example `is a` before `is`).
const PATTERNS: &[Pattern] = &[
    // --- organisation ------------------------------------------------------
    Pattern::single("is founded by", "founded_by", 0.85),
    Pattern::single("was founded by", "founded_by", 0.85),
    Pattern::single("founded by", "founded_by", 0.80),
    Pattern::single("is owned by", "owned_by", 0.80),
    Pattern::single("was owned by", "owned_by", 0.80),
    Pattern::single("owned by", "owned_by", 0.75),
    Pattern::single("acquires", "acquired", 0.80),
    Pattern::single("acquired", "acquired", 0.80),
    Pattern::single("owns", "owned_by", 0.75),
    // --- creation ----------------------------------------------------------
    Pattern::single("developed", "created", 0.75),
    Pattern::single("creates", "created", 0.75),
    Pattern::single("created", "created", 0.75),
    Pattern::single("invented", "created", 0.70),
    Pattern::single("built", "created", 0.70),
    // --- part-of / exemplar ------------------------------------------------
    Pattern::list("such as", "includes", 0.85),
    Pattern::list("e.g.", "includes", 0.85),
    Pattern::list("e.g", "includes", 0.85),
    Pattern::list("includes", "includes", 0.80),
    Pattern::list("include", "includes", 0.80),
    Pattern::list("including", "includes", 0.75),
    // --- possession --------------------------------------------------------
    Pattern::list("consists of", "contains", 0.75),
    Pattern::single("contains", "contains", 0.75),
    Pattern::single("contain", "contains", 0.75),
    Pattern::single("has", "has", 0.70),
    Pattern::single("have", "has", 0.70),
    // --- copula ------------------------------------------------------------
    Pattern::single("is an", "is_a", 0.85),
    Pattern::single("is a", "is_a", 0.85),
    Pattern::single("was an", "is_a", 0.80),
    Pattern::single("was a", "is_a", 0.80),
    Pattern::single("are", "is_a", 0.75),
    Pattern::single("were", "is_a", 0.75),
    Pattern::single("was", "is_a", 0.65),
    Pattern::single("is", "is_a", 0.60),
    // --- usage -------------------------------------------------------------
    Pattern::single("uses", "uses", 0.70),
    Pattern::single("used", "uses", 0.70),
    Pattern::single("use", "uses", 0.65),
    Pattern::single("with", "with", 0.50),
];

/// Labels that introduce prose rather than define a term.
const GLOSSARY_LABELS: &[&str] = &[
    "note",
    "example",
    "warning",
    "tip",
    "summary",
    "definition",
    "see",
    "e.g",
    "i.e",
];

/// Whether `(start, priority)` beats the current best candidate.
fn is_earlier(best: Option<&(usize, usize, Vec<Triple>)>, start: usize, priority: usize) -> bool {
    match best {
        None => true,
        Some((best_start, best_priority, _)) => {
            start < *best_start || (start == *best_start && priority < *best_priority)
        }
    }
}

/// Base confidence of the generic `X: Y` glossary form.
const GLOSSARY_CONFIDENCE: f32 = 0.55;
/// Confidence of the opt-in `appears_in` entity anchor.
const ENTITY_ANCHOR_CONFIDENCE: f32 = 0.30;

/// Rule-based, deterministic [`TripleExtractor`].
///
/// Pure Rust and model-free: no network, no randomness, no wall-clock input. The
/// same chunks always produce the same triples in the same order.
#[derive(Debug, Clone)]
pub struct RuleTripleExtractor {
    config: RuleExtractionConfig,
}

impl Default for RuleTripleExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl RuleTripleExtractor {
    /// Creates an extractor with [`RuleExtractionConfig::default`].
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: RuleExtractionConfig::default(),
        }
    }

    /// Creates an extractor with an explicit configuration.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when `max_triples_per_chunk` or
    /// `min_entity_chars` is zero, or when `max_entity_chars < min_entity_chars`.
    pub fn with_config(config: RuleExtractionConfig) -> Result<Self> {
        if config.max_triples_per_chunk == 0 {
            return Err(Error::validation(
                "max_triples_per_chunk must be greater than zero",
            ));
        }
        if config.min_entity_chars == 0 {
            return Err(Error::validation(
                "min_entity_chars must be greater than zero",
            ));
        }
        if config.max_entity_chars < config.min_entity_chars {
            return Err(Error::validation(format!(
                "max_entity_chars ({}) must be greater than or equal to min_entity_chars ({})",
                config.max_entity_chars, config.min_entity_chars
            )));
        }
        Ok(Self { config })
    }

    /// The extractor configuration.
    pub const fn config(&self) -> &RuleExtractionConfig {
        &self.config
    }

    /// Extracts triples synchronously (the async trait method delegates here).
    ///
    /// Output order is chunk-major, then sentence/text order, capped per chunk by
    /// [`RuleExtractionConfig::max_triples_per_chunk`] and deduplicated by
    /// [`Triple::normalized_key`].
    pub fn extract_chunks(&self, chunks: &[TextChunk]) -> Vec<Triple> {
        let mut triples = Vec::new();
        for chunk in chunks {
            triples.extend(self.chunk_triples(chunk));
        }
        triples
    }

    /// The entity candidates found in `text`, in text order.
    ///
    /// Candidates are quoted terms, CJK runs of two or more characters, and
    /// capitalised phrases (including multi-word ones joined by connectors such as
    /// `of` or `and`). The configuration's entity length bounds apply.
    pub fn entities_in_text(&self, text: &str) -> Vec<Entity> {
        let mut entities: Vec<Entity> = Vec::new();
        for sentence in split_sentences(text) {
            for entity in self.entity_candidates(&sentence) {
                let name = entity.normalized_name();
                if name.is_empty() || entities.iter().any(|seen| seen.normalized_name() == name) {
                    continue;
                }
                entities.push(entity);
            }
        }
        entities
    }

    /// Extracts the triples of one chunk.
    fn chunk_triples(&self, chunk: &TextChunk) -> Vec<Triple> {
        let mut triples: Vec<Triple> = Vec::new();
        let mut seen: Vec<String> = Vec::new();
        for sentence in split_sentences(&chunk.text) {
            if triples.len() >= self.config.max_triples_per_chunk {
                break;
            }
            for triple in self.sentence_triples(&sentence, chunk) {
                if triples.len() >= self.config.max_triples_per_chunk {
                    break;
                }
                let key = triple.normalized_key();
                if seen.contains(&key) {
                    continue;
                }
                seen.push(key);
                triples.push(triple);
            }
        }
        triples
    }

    /// Extracts the triples of one sentence.
    ///
    /// The earliest usable pattern occurrence wins (the glossary form competes on
    /// its colon offset and loses ties); the single-entity anchor is the last
    /// fallback and only runs when nothing else matched.
    fn sentence_triples(&self, sentence: &str, chunk: &TextChunk) -> Vec<Triple> {
        let mut best: Option<(usize, usize, Vec<Triple>)> = None;
        for (priority, pattern) in PATTERNS.iter().enumerate() {
            if !self.config.allows(pattern.predicate) {
                continue;
            }
            let Some((start, triples)) = self.pattern_triples(sentence, pattern, chunk) else {
                continue;
            };
            if is_earlier(best.as_ref(), start, priority) {
                best = Some((start, priority, triples));
            }
        }
        if let Some((start, triple)) = self.glossary_candidate(sentence, chunk) {
            if is_earlier(best.as_ref(), start, PATTERNS.len()) {
                best = Some((start, PATTERNS.len(), vec![triple]));
            }
        }
        if let Some((_, _, triples)) = best {
            return triples;
        }
        if !self.config.include_single_entity {
            return Vec::new();
        }
        self.single_entity_triple(sentence, chunk)
            .map_or_else(Vec::new, |triple| vec![triple])
    }

    /// Applies one pattern to a sentence, using its first usable occurrence.
    fn pattern_triples(
        &self,
        sentence: &str,
        pattern: &Pattern,
        chunk: &TextChunk,
    ) -> Option<(usize, Vec<Triple>)> {
        for (start, end) in find_keyword(sentence, pattern.keyword) {
            let Some(subject) = self.clean_subject(&sentence[..start]) else {
                continue;
            };
            let remainder = trim_object_tail(sentence[end..].trim());
            let candidates = if pattern.split_list {
                split_object_list(remainder)
            } else {
                vec![remainder.to_owned()]
            };
            let mut triples = Vec::new();
            for candidate in candidates {
                let Some(object) = self.clean_object(&candidate) else {
                    continue;
                };
                if let Some(triple) = build_triple(&subject, pattern, &object, chunk) {
                    triples.push(triple);
                }
            }
            if !triples.is_empty() {
                return Some((start, triples));
            }
        }
        None
    }

    /// The generic `X: Y` glossary form plus the byte offset of its colon.
    fn glossary_candidate(&self, sentence: &str, chunk: &TextChunk) -> Option<(usize, Triple)> {
        if !self.config.allows("describes") {
            return None;
        }
        let start = sentence.find(':')?;
        let (left, right) = sentence.split_at(start);
        let right = &right[1..];
        // `http://…` and other URI fragments are not glossary entries, and neither
        // are the labels people use to introduce a sentence ("Note: …").
        if right.starts_with("//") || left.trim().is_empty() {
            return None;
        }
        if left.split_whitespace().count() > MAX_SUBJECT_WORDS {
            return None;
        }
        if GLOSSARY_LABELS.contains(&normalize_component(left).as_str()) {
            return None;
        }
        let subject = self.clean_subject(left)?;
        let object = self.clean_object(right)?;
        let pattern = Pattern::single(":", "describes", GLOSSARY_CONFIDENCE);
        Some((start, build_triple(&subject, &pattern, &object, chunk)?))
    }

    /// The opt-in `(entity, "appears_in", <doc id>)` anchor.
    fn single_entity_triple(&self, sentence: &str, chunk: &TextChunk) -> Option<Triple> {
        if !self.config.allows("appears_in") {
            return None;
        }
        let doc_id = chunk.doc_id.trim();
        if doc_id.is_empty() {
            return None;
        }
        let entities = self.entity_candidates(sentence);
        if entities.len() != 1 {
            return None;
        }
        let pattern = Pattern::single("", "appears_in", ENTITY_ANCHOR_CONFIDENCE);
        build_triple(&entities[0].name, &pattern, doc_id, chunk)
    }

    /// Cleans a subject candidate: comma-last clause, no prepositional prefix.
    fn clean_subject(&self, raw: &str) -> Option<String> {
        let segments: Vec<&str> = raw
            .split(',')
            .map(str::trim)
            .filter(|segment| !segment.is_empty())
            .collect();
        let last = *segments.last()?;
        let candidate = if has_entity_hint(last) {
            last
        } else {
            segments
                .iter()
                .rev()
                .find(|segment| has_entity_hint(segment))
                .copied()
                .unwrap_or(last)
        };
        // A sentence label (`"Note: Acme uses Rust"`) is not part of the subject.
        let candidate = candidate.rsplit(':').next().unwrap_or(candidate);
        let candidate = strip_prepositional_prefix(candidate);
        self.normalize_entity(candidate, true)
    }

    /// Cleans an object candidate.
    fn clean_object(&self, raw: &str) -> Option<String> {
        self.normalize_entity(raw, false)
    }

    /// Trims, strips articles/connectors/punctuation, and validates the length.
    fn normalize_entity(&self, raw: &str, enforce_word_limit: bool) -> Option<String> {
        let mut text = raw.trim();
        for _ in 0..8 {
            let before = text;
            text = strip_leading(text, LEADING_CONNECTORS);
            text = strip_leading(text, ARTICLES);
            if text == before {
                break;
            }
        }
        let text = text
            .trim_start_matches(|character: char| {
                character.is_whitespace() || LEADING_PUNCTUATION.contains(character)
            })
            .trim_end_matches(|character: char| {
                character.is_whitespace() || TRAILING_PUNCTUATION.contains(character)
            });
        let cleaned = text.split_whitespace().collect::<Vec<_>>().join(" ");
        if cleaned.is_empty() {
            return None;
        }
        let length = cleaned.chars().count();
        if length < self.config.min_entity_chars || length > self.config.max_entity_chars {
            return None;
        }
        if !cleaned.chars().any(char::is_alphabetic) {
            return None;
        }
        let lowered = cleaned.to_lowercase();
        if STOPWORDS.contains(&lowered.as_str()) {
            return None;
        }
        if enforce_word_limit && cleaned.split_whitespace().count() > MAX_SUBJECT_WORDS {
            return None;
        }
        Some(cleaned)
    }

    /// Entity candidates of one sentence, in text order and deduplicated.
    fn entity_candidates(&self, sentence: &str) -> Vec<Entity> {
        let mut positioned: Vec<(usize, Entity)> = Vec::new();
        for (start, content) in quoted_spans(sentence) {
            if let Some(entity) = self.entity(&content, "quoted") {
                positioned.push((start, entity));
            }
        }
        for (start, run) in cjk_runs(sentence) {
            if let Some(entity) = self.entity(&run, "cjk") {
                positioned.push((start, entity));
            }
        }
        for (start, phrase) in capitalised_phrases(sentence) {
            if let Some(entity) = self.entity(&phrase, "capitalised") {
                positioned.push((start, entity));
            }
        }
        positioned.sort_by_key(|(start, _)| *start);
        let mut entities: Vec<Entity> = Vec::new();
        for (_, entity) in positioned {
            let name = entity.normalized_name();
            if name.is_empty() || entities.iter().any(|seen| seen.normalized_name() == name) {
                continue;
            }
            entities.push(entity);
        }
        entities
    }

    fn entity(&self, raw: &str, kind: &str) -> Option<Entity> {
        let name = self.normalize_entity(raw, false)?;
        Some(Entity::new(name).with_kind(kind))
    }
}

#[async_trait]
impl TripleExtractor for RuleTripleExtractor {
    async fn extract(&self, chunks: &[TextChunk]) -> Result<Vec<Triple>> {
        Ok(self.extract_chunks(chunks))
    }
}

/// Materialises a triple with provenance metadata and adjusted confidence.
fn build_triple(
    subject: &str,
    pattern: &Pattern,
    object: &str,
    chunk: &TextChunk,
) -> Option<Triple> {
    if normalize_component(subject) == normalize_component(object) {
        return None;
    }
    let confidence = adjust_confidence(pattern.confidence, subject, object);
    let mut metadata = Metadata::new();
    metadata.insert("doc_id".to_owned(), serde_json::json!(chunk.doc_id));
    metadata.insert("chunk_id".to_owned(), serde_json::json!(chunk.id));
    metadata.insert("confidence".to_owned(), serde_json::json!(confidence));
    Some(Triple::new(subject, pattern.predicate, object).with_metadata(metadata))
}

fn adjust_confidence(base: f32, subject: &str, object: &str) -> f64 {
    let mut value = f64::from(base);
    if !has_entity_hint(subject) {
        value -= 0.10;
    }
    if !has_entity_hint(object) {
        value -= 0.10;
    }
    if object.chars().count() <= 3 {
        value -= 0.05;
    }
    let percent = (value.clamp(0.05, 1.0) * 100.0).round();
    percent / 100.0
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::common::Document;

    fn chunk(text: &str) -> TextChunk {
        TextChunk::from_document(
            &Document::with_id("doc-1", text),
            text,
            Some("chunk-1".to_owned()),
        )
    }

    #[test]
    fn entities_are_only_emitted_when_enabled() {
        let extractor = RuleTripleExtractor::new();
        let chunks = [chunk("杭州")];
        assert_eq!(extractor.extract_chunks(&chunks), Vec::new());
        assert_eq!(extractor.entities_in_text("杭州")[0].name, "杭州");

        let enabled = RuleTripleExtractor::with_config(
            RuleExtractionConfig::new().with_include_single_entity(true),
        )
        .unwrap();
        let triples = enabled.extract_chunks(&chunks);
        assert_eq!(triples.len(), 1);
        assert_eq!(triples[0].predicate, "appears_in");
        assert_eq!(triples[0].object, "doc-1");
    }

    #[test]
    fn empty_inputs_yield_no_triples() {
        let extractor = RuleTripleExtractor::new();
        assert_eq!(extractor.extract_chunks(&[]), Vec::new());
        assert_eq!(extractor.extract_chunks(&[chunk("   \n\t ")]), Vec::new());
        assert_eq!(extractor.entities_in_text(""), Vec::new());
    }

    #[test]
    fn per_chunk_cap_and_dedupe_apply() {
        let extractor = RuleTripleExtractor::with_config(
            RuleExtractionConfig::new().with_max_triples_per_chunk(2),
        )
        .unwrap();
        let text = "Acme is a company. Acme is a company. Beta has Rust. Gamma uses Go.";
        let triples = extractor.extract_chunks(&[chunk(text)]);
        assert_eq!(triples.len(), 2);
        assert_eq!(triples[0].object, "company");
        assert_eq!(triples[1].object, "Rust");

        let repeated = extractor.extract_chunks(&[chunk("Acme is a company. acme is a Company.")]);
        assert_eq!(repeated.len(), 1);
    }

    #[test]
    fn configuration_is_validated() {
        assert!(RuleTripleExtractor::with_config(
            RuleExtractionConfig::new().with_max_triples_per_chunk(0)
        )
        .is_err());
        assert!(RuleTripleExtractor::with_config(
            RuleExtractionConfig::new().with_predicates(Vec::<String>::new())
        )
        .is_ok());
        let extractor =
            RuleTripleExtractor::with_config(RuleExtractionConfig::new().with_predicates(["is_a"]))
                .unwrap();
        assert!(extractor.config().allows("IS_A"));
        assert!(!extractor.config().allows("has"));
    }

    #[tokio::test]
    async fn trait_extract_delegates_to_the_sync_core() {
        let extractor = RuleTripleExtractor::new();
        let triples = extractor
            .extract(&[chunk("Acme is a company.")])
            .await
            .unwrap();
        assert_eq!(
            triples,
            extractor.extract_chunks(&[chunk("Acme is a company.")])
        );
    }
}
