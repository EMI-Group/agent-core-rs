//! Entity/relation extraction for graph memory.
//!
//! [`HeuristicExtractor`] is the deterministic, dependency-free default: it scans for
//! quoted/backticked names, `@mentions`, path-like tokens, and non-sentence-initial
//! capitalized runs, then derives simple verb-pattern relations. [`LlmExtractor`] asks
//! a [`Model`] for JSON and degrades to the heuristic
//! extractor when the model output stays unusable.
//!
//! The module is self-contained: it owns the extraction data types ([`Extraction`],
//! [`ExtractedEntity`], [`ExtractedRelation`]) and never touches graph persistence.

use std::collections::HashSet;
use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_common::error::{ErrorKind, Result};
use openjiuwen_foundation::llm::{Model, ModelRequestConfig};
use openjiuwen_foundation::prompt::BaseMessage;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::config::{ExtractionConfig, GraphMemoryConfig};
use crate::record::{CallOptions, MemoryScope};

/// System prompt for [`LlmExtractor`].
///
/// It pins the exact JSON shape the parser accepts and nothing else, so a compliant
/// model answer needs no repair pass.
pub const EXTRACTION_SYSTEM_PROMPT: &str = concat!(
    "Extract entities and relations from the user text. ",
    "Reply with JSON only, using exactly this shape and no other keys: ",
    "{\"entities\":[{\"name\":\"...\",\"type\":\"...\",\"importance\":0.0}],",
    "\"relations\":[{\"from\":\"...\",\"to\":\"...\",\"type\":\"...\"}]}. ",
    "Use a short `type` label such as Person, Project, Concept, or Organization ",
    "for entities and a short verb-like label such as works_at for relations. ",
    "`importance` must be a number between 0.0 and 1.0. ",
    "Do not add prose, Markdown fences, or explanations."
);

/// One entity found in a memory text.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case", default)]
pub struct ExtractedEntity {
    /// Entity name exactly as it should be persisted (`"Alice Smith"`, `"src/main.rs"`).
    pub name: String,
    /// Short kind label such as `Person`, `Project`, `Concept`, or `Organization`.
    pub entity_type: String,
    /// Optional salience in `0.0..=1.0`; heuristics leave it `None`.
    pub importance: Option<f32>,
}

/// One directed relation between two [`ExtractedEntity`] names.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case", default)]
pub struct ExtractedRelation {
    /// Name of the source entity.
    pub from: String,
    /// Name of the target entity.
    pub to: String,
    /// Short relation label such as `works_at` or `is_a`.
    pub relation_type: String,
}

/// The entities and relations extracted from one text.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case", default)]
pub struct Extraction {
    /// Entities in first-seen order.
    pub entities: Vec<ExtractedEntity>,
    /// Relations in first-seen order.
    pub relations: Vec<ExtractedRelation>,
}

/// Turns one text into an [`Extraction`].
///
/// Implementations must be usable from `GraphMemory` concurrently (`Send + Sync`),
/// must honour [`CallOptions`], and must never panic on arbitrary input.
#[async_trait]
pub trait EntityExtractor: Send + Sync {
    /// Stable implementation name (`"heuristic"`, `"llm"`), used in diagnostics.
    fn name(&self) -> &str;

    /// Extracts entities and relations from `text` within `scope`.
    ///
    /// # Errors
    ///
    /// Returns `Error::Cancelled` when `options` has been cancelled. Implementations
    /// that call a model may also surface the model's own error kind.
    async fn extract(
        &self,
        text: &str,
        scope: &MemoryScope,
        options: &CallOptions,
    ) -> Result<Extraction>;
}

/// Deterministic extractor: no model, no regex crate, no I/O.
///
/// Candidate sources are quoted substrings, backticked identifiers, `@mentions`,
/// path/filename-like tokens, and non-sentence-initial capitalized runs; relations are
/// fixed verb patterns between the surviving entities. The output depends only on the
/// input text and the extraction knobs.
pub struct HeuristicExtractor {
    /// Upper bound on returned entities (`0` means "no entities").
    max_entities: usize,
    /// Whether relations are derived from the surviving entity names.
    extract_relations: bool,
}

impl HeuristicExtractor {
    /// Builds an extractor from the graph limits.
    #[must_use]
    pub fn new(config: &GraphMemoryConfig) -> Self {
        Self {
            max_entities: config.max_entities,
            extract_relations: config.extract_relations,
        }
    }
}

#[async_trait]
impl EntityExtractor for HeuristicExtractor {
    #[allow(clippy::unnecessary_literal_bound)] // trait signature returns `&str` tied to `self`
    fn name(&self) -> &str {
        "heuristic"
    }

    async fn extract(
        &self,
        text: &str,
        _scope: &MemoryScope,
        options: &CallOptions,
    ) -> Result<Extraction> {
        options.check()?;
        let candidates = collect_candidates(text);
        let entities = self.surviving_entities(text, candidates);
        let relations = if self.extract_relations {
            relations_from(text, &entities)
        } else {
            Vec::new()
        };
        Ok(Extraction {
            entities,
            relations,
        })
    }
}

/// Model-backed extractor with heuristic fallback.
///
/// One [`Model::invoke`] call per attempt with [`EXTRACTION_SYSTEM_PROMPT`]; the answer
/// is parsed tolerantly by `parse_extraction`. Failed calls and unusable answers are
/// retried, and once the retry budget is spent the extractor falls back to
/// [`HeuristicExtractor`] so graph memory keeps working without a healthy model.
pub struct LlmExtractor {
    /// Model used for extraction.
    model: Arc<dyn Model>,
    /// Number of retries after the first attempt (`ExtractionConfig::model_retries`).
    retries: u32,
    /// Graph limits; the fallback heuristic and entity cap come from here.
    config: GraphMemoryConfig,
}

impl LlmExtractor {
    /// Builds an extractor around `model` using the graph limits.
    ///
    /// The retry budget is `ExtractionConfig::default().model_retries`, keeping it in
    /// step with the message-extraction path instead of a magic number.
    #[must_use]
    pub fn new(model: Arc<dyn Model>, config: &GraphMemoryConfig) -> Self {
        Self {
            model,
            retries: ExtractionConfig::default().model_retries,
            config: config.clone(),
        }
    }
}

#[async_trait]
impl EntityExtractor for LlmExtractor {
    #[allow(clippy::unnecessary_literal_bound)] // trait signature returns `&str` tied to `self`
    fn name(&self) -> &str {
        "llm"
    }

    async fn extract(
        &self,
        text: &str,
        scope: &MemoryScope,
        options: &CallOptions,
    ) -> Result<Extraction> {
        options.check()?;
        let messages = vec![
            BaseMessage::system(EXTRACTION_SYSTEM_PROMPT),
            BaseMessage::user(text),
        ];
        let request = ModelRequestConfig::new(self.model.model_name().to_owned());
        let attempts = self.retries.saturating_add(1);
        for attempt in 1..=attempts {
            let outcome = options.run(self.model.invoke(&messages, &request)).await;
            let response = match outcome {
                Ok(Ok(response)) => response,
                Ok(Err(error)) | Err(error) => {
                    if error.kind() == ErrorKind::Cancelled {
                        return Err(error);
                    }
                    tracing::warn!(
                        target: "openjiuwen_memory::graph::extract",
                        attempt,
                        %error,
                        "LLM entity extraction call failed; retrying"
                    );
                    continue;
                }
            };
            let raw = response
                .message()
                .map(|message| message.text().into_owned());
            if let Some(mut extraction) = raw.as_deref().and_then(parse_extraction) {
                if self.config.max_entities > 0 {
                    extraction.entities.truncate(self.config.max_entities);
                }
                return Ok(extraction);
            }
            tracing::warn!(
                target: "openjiuwen_memory::graph::extract",
                attempt,
                "LLM entity extraction returned unusable output; retrying"
            );
        }
        tracing::warn!(
            target: "openjiuwen_memory::graph::extract",
            attempts,
            "LLM entity extraction exhausted its retries; falling back to heuristics"
        );
        HeuristicExtractor::new(&self.config)
            .extract(text, scope, options)
            .await
    }
}

/// Which scan produced a [`Candidate`]; drives entity type inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CandidateCategory {
    /// Text between matching `"` or `'` quotes.
    Quoted,
    /// Text between matching backticks.
    Backtick,
    /// `@`-prefixed alphanumeric run, stored without the `@`.
    Mention,
    /// Whitespace-delimited path or filename-like token.
    Path,
    /// Run of capitalized words that does not start a sentence.
    Capitalized,
}

/// One extracted entity candidate plus its position in the source text.
#[derive(Debug, Clone)]
struct Candidate {
    /// Trimmed candidate text.
    text: String,
    /// Byte offset of the candidate's first character.
    start: usize,
    /// Byte offset just past the candidate in the source text.
    end: usize,
    /// Scan that produced this candidate.
    category: CandidateCategory,
}

/// Characters trimmed from both ends of a whitespace-delimited token.
const TRIM_PUNCTUATION: &[char] = &[
    '.', ',', ';', ':', '!', '?', '(', ')', '[', ']', '{', '}', '"', '\'',
];

/// Candidates dropped regardless of source: function words and bare pronouns.
const STOPWORDS: &[&str] = &[
    "the", "a", "an", "this", "that", "these", "those", "it", "its", "i", "we", "you", "he", "she",
    "they", "them", "there", "here", "me", "him", "her", "us", "my", "your", "our", "his", "hers",
    "their", "theirs", "mine", "yours", "ours", "in", "on", "at", "as", "if", "but", "and", "or",
    "for", "to", "of", "with", "from", "by", "when", "while", "what", "who", "why", "how", "is",
    "are", "was", "were", "be", "been", "being", "do", "does", "did", "can", "could", "will",
    "would", "should", "may", "might", "must", "not", "no", "yes", "also", "then", "so", "such",
    "more", "most", "some", "any", "all", "each", "every", "both", "either", "neither", "one",
    "two", "i'm", "it's", "that's", "there's",
];

/// Words after a quoted/capitalized candidate that mark it as a person.
const VERB_LIKE: &[&str] = &[
    "works",
    "works_at",
    "uses",
    "owns",
    "prefers",
    "likes",
    "depends",
    "leads",
    "manages",
    "builds",
    "wrote",
    "created",
    "maintains",
    "joined",
    "runs",
    "said",
    "wants",
    "needs",
    "reviews",
    "designs",
    "ships",
    "is",
    "was",
    "has",
    "had",
    "will",
    "can",
];

/// `(phrase, relation_type)` pairs checked in this fixed order for every entity pair.
const RELATION_PATTERNS: &[(&str, &str)] = &[
    ("is a", "is_a"),
    ("is the", "is_a"),
    ("works at", "works_at"),
    ("uses", "uses"),
    ("depends on", "depends_on"),
    ("owns", "owns"),
    ("is named", "named"),
];

/// Collects candidates from every scan and returns them in textual first-seen order.
fn collect_candidates(text: &str) -> Vec<Candidate> {
    let mut candidates = Vec::new();
    scan_pairs(text, '"', CandidateCategory::Quoted, &mut candidates);
    scan_pairs(text, '\'', CandidateCategory::Quoted, &mut candidates);
    scan_pairs(text, '`', CandidateCategory::Backtick, &mut candidates);
    scan_mentions(text, &mut candidates);
    scan_tokens(text, &mut candidates);
    scan_capitalized(text, &mut candidates);
    candidates.sort_by_key(|candidate| candidate.start);
    candidates
}

/// Records the text between matching `quote` characters.
///
/// The content is trimmed and dropped when empty or longer than 64 characters. An
/// unmatched quote does not abort the scan: the search continues after it.
fn scan_pairs(text: &str, quote: char, category: CandidateCategory, out: &mut Vec<Candidate>) {
    let mut offset = 0;
    while let Some(rest) = text.get(offset..) {
        let Some(open) = rest.find(quote) else {
            break;
        };
        let content_start = offset + open + quote.len_utf8();
        let Some(tail) = text.get(content_start..) else {
            break;
        };
        let Some(close) = tail.find(quote) else {
            offset = content_start;
            continue;
        };
        let content_end = content_start + close;
        if let Some(raw) = text.get(content_start..content_end) {
            let trimmed = raw.trim();
            if is_usable_length(trimmed) {
                out.push(Candidate {
                    text: trimmed.to_owned(),
                    start: content_start,
                    end: content_end,
                    category,
                });
            }
        }
        offset = content_end + quote.len_utf8();
    }
}

/// Records `@`-prefixed names (`@bob`, `@team.lead`) without the `@`.
///
/// A name needs at least two characters from `[alphanumeric]`, `_`, `-`, or `.`.
fn scan_mentions(text: &str, out: &mut Vec<Candidate>) {
    let mut offset = 0;
    while let Some(rest) = text.get(offset..) {
        let Some(at) = rest.find('@') else {
            break;
        };
        let name_start = offset + at + 1;
        let Some(tail) = text.get(name_start..) else {
            break;
        };
        let length = tail
            .chars()
            .take_while(|ch| ch.is_alphanumeric() || matches!(ch, '_' | '-' | '.'))
            .map(char::len_utf8)
            .sum::<usize>();
        if let Some(name) = text.get(name_start..name_start + length) {
            if name.chars().count() >= 2 {
                out.push(Candidate {
                    text: name.to_owned(),
                    start: name_start,
                    end: name_start + length,
                    category: CandidateCategory::Mention,
                });
            }
        }
        offset = name_start + length;
    }
}

/// Records whitespace-delimited tokens that look like paths or filenames.
fn scan_tokens(text: &str, out: &mut Vec<Candidate>) {
    let mut token_start = None;
    for (index, ch) in text.char_indices() {
        if ch.is_whitespace() {
            if let Some(start) = token_start.take() {
                consider_token(text, start, index, out);
            }
        } else if token_start.is_none() {
            token_start = Some(index);
        }
    }
    if let Some(start) = token_start {
        consider_token(text, start, text.len(), out);
    }
}

/// Keeps one trimmed token when it contains a path separator or looks like a filename.
fn consider_token(text: &str, raw_start: usize, raw_end: usize, out: &mut Vec<Candidate>) {
    let (start, end) = trim_bounds(text, raw_start, raw_end);
    let Some(token) = text.get(start..end) else {
        return;
    };
    let has_separator = token.contains('/') || token.contains('\\');
    let is_path = has_separator && token.chars().count() >= 2;
    if is_path || is_filename_like(token) {
        out.push(Candidate {
            text: token.to_owned(),
            start,
            end,
            category: CandidateCategory::Path,
        });
    }
}

/// Returns `true` for `stem.extension` tokens such as `main.rs` or `Cargo.toml`.
///
/// The stem must be alphanumeric and the extension must be 1-8 letters.
fn is_filename_like(token: &str) -> bool {
    let Some((stem, extension)) = token.rsplit_once('.') else {
        return false;
    };
    !stem.is_empty()
        && stem.chars().all(char::is_alphanumeric)
        && !extension.is_empty()
        && extension.chars().count() <= 8
        && extension.chars().all(char::is_alphabetic)
}

/// Scan state for one maximal sequence of capitalized words.
struct CapitalizedState {
    /// Run being kept: `(start, end, parts)`.
    run: Option<(usize, usize, Vec<String>)>,
    /// Whether the current sequence is a skipped sentence-initial run.
    skipping: bool,
    /// End of the previous capitalized word in the current sequence.
    previous_end: Option<usize>,
}

/// Records maximal runs of capitalized words that do not start a sentence.
///
/// A run is a sequence of whitespace-separated words whose first character is
/// uppercase, joined by single spaces (`"New York"`). It is skipped entirely when its
/// first word is sentence-initial, meaning it sits at the start of the text, is
/// preceded only by whitespace containing a newline, or follows one of `.`, `!`, `?`,
/// `:`, `;` — sentence-initial capitalization carries no signal. A run also restarts
/// when punctuation separates two capitalized words.
fn scan_capitalized(text: &str, out: &mut Vec<Candidate>) {
    let mut state = CapitalizedState {
        run: None,
        skipping: false,
        previous_end: None,
    };
    let mut word_start = None;
    for (index, ch) in text.char_indices() {
        if ch.is_whitespace() {
            if let Some(start) = word_start.take() {
                push_capitalized_word(text, start, index, &mut state, out);
            }
        } else if word_start.is_none() {
            word_start = Some(index);
        }
    }
    if let Some(start) = word_start {
        push_capitalized_word(text, start, text.len(), &mut state, out);
    }
    flush_capitalized(&mut state, out);
}

/// Handles one whitespace-delimited word in [`scan_capitalized`].
fn push_capitalized_word(
    text: &str,
    raw_start: usize,
    raw_end: usize,
    state: &mut CapitalizedState,
    out: &mut Vec<Candidate>,
) {
    let (start, end) = trim_bounds(text, raw_start, raw_end);
    let Some(word) = text.get(start..end) else {
        return;
    };
    let capitalized = word.chars().next().is_some_and(char::is_uppercase);
    if !capitalized {
        flush_capitalized(state, out);
        state.skipping = false;
        state.previous_end = None;
        return;
    }
    let contiguous = state.previous_end.is_some_and(|previous| {
        text.get(previous..start)
            .is_some_and(|gap| gap.chars().all(char::is_whitespace))
    });
    if contiguous {
        if let Some((_, run_end, parts)) = state.run.as_mut() {
            parts.push(word.to_owned());
            *run_end = end;
        }
    } else {
        flush_capitalized(state, out);
        state.skipping = false;
        if is_sentence_initial(text, start) {
            state.skipping = true;
        } else {
            state.run = Some((start, end, vec![word.to_owned()]));
        }
    }
    state.previous_end = Some(end);
}

/// Emits the run being kept, if any.
fn flush_capitalized(state: &mut CapitalizedState, out: &mut Vec<Candidate>) {
    if let Some((start, end, parts)) = state.run.take() {
        out.push(Candidate {
            text: parts.join(" "),
            start,
            end,
            category: CandidateCategory::Capitalized,
        });
    }
}

/// Returns `true` when the word starting at `start` opens a sentence.
fn is_sentence_initial(text: &str, start: usize) -> bool {
    let Some(prefix) = text.get(..start) else {
        return true;
    };
    for ch in prefix.chars().rev() {
        if matches!(ch, '\n' | '\r') {
            return true;
        }
        if ch.is_whitespace() {
            continue;
        }
        return matches!(ch, '.' | '!' | '?' | ':' | ';');
    }
    true
}

/// Trims whitespace and [`TRIM_PUNCTUATION`] from `raw_start..raw_end`.
fn trim_bounds(text: &str, raw_start: usize, raw_end: usize) -> (usize, usize) {
    let Some(raw) = text.get(raw_start..raw_end) else {
        return (raw_start, raw_start);
    };
    let trimmed_start =
        raw.trim_start_matches(|ch: char| ch.is_whitespace() || TRIM_PUNCTUATION.contains(&ch));
    let leading = raw.len() - trimmed_start.len();
    let trimmed = trimmed_start
        .trim_end_matches(|ch: char| ch.is_whitespace() || TRIM_PUNCTUATION.contains(&ch));
    (raw_start + leading, raw_start + leading + trimmed.len())
}

/// Whether a scanned candidate is long enough to keep (non-empty, at most 64 chars).
fn is_usable_length(text: &str) -> bool {
    let length = text.chars().count();
    (1..=64).contains(&length)
}

impl HeuristicExtractor {
    /// Filters, deduplicates, and types the candidates in first-seen order.
    fn surviving_entities(&self, text: &str, candidates: Vec<Candidate>) -> Vec<ExtractedEntity> {
        if self.max_entities == 0 {
            return Vec::new();
        }
        let mut seen: HashSet<String> = HashSet::new();
        let mut entities: Vec<ExtractedEntity> = Vec::new();
        for candidate in candidates {
            if entities.len() >= self.max_entities {
                break;
            }
            let name = candidate.text.trim();
            let length = name.chars().count();
            if !(2..=64).contains(&length) {
                continue;
            }
            if !name.chars().any(char::is_alphabetic) {
                continue;
            }
            let key = name.to_lowercase();
            if STOPWORDS.contains(&key.as_str()) || !seen.insert(key) {
                continue;
            }
            entities.push(ExtractedEntity {
                name: name.to_owned(),
                entity_type: infer_entity_type(text, &candidate),
                importance: None,
            });
        }
        entities
    }
}

/// Infers the entity kind from the producing scan and the following word.
fn infer_entity_type(text: &str, candidate: &Candidate) -> String {
    match candidate.category {
        CandidateCategory::Path => "Project".to_owned(),
        CandidateCategory::Capitalized | CandidateCategory::Quoted => {
            let person = next_word(text, candidate.end)
                .is_some_and(|word| VERB_LIKE.contains(&word.as_str()));
            if person {
                "Person".to_owned()
            } else {
                "Concept".to_owned()
            }
        }
        CandidateCategory::Backtick | CandidateCategory::Mention => "Concept".to_owned(),
    }
}

/// Returns the lowercase alphanumeric word after `from`, if any.
fn next_word(text: &str, from: usize) -> Option<String> {
    let tail = text.get(from..)?;
    let (start, _) = tail.char_indices().find(|(_, ch)| ch.is_alphanumeric())?;
    let word = tail.get(start..)?;
    let length = word
        .chars()
        .take_while(|ch| ch.is_alphanumeric())
        .map(char::len_utf8)
        .sum::<usize>();
    word.get(..length).map(str::to_lowercase)
}

/// Derives verb-pattern relations between the surviving entity names.
///
/// Every ordered pair of distinct entities is tested against [`RELATION_PATTERNS`] in
/// order, against the normalized (lowercase, whitespace-collapsed) text. Only the
/// first matching pattern per pair is emitted; results are deduplicated by
/// `(from, to, relation_type)` and keep the first occurrence.
fn relations_from(text: &str, entities: &[ExtractedEntity]) -> Vec<ExtractedRelation> {
    let mut relations: Vec<ExtractedRelation> = Vec::new();
    if entities.len() < 2 {
        return relations;
    }
    let haystack = normalize(text);
    for (from_index, from) in entities.iter().enumerate() {
        let from_lower = normalize(&from.name);
        for (to_index, to) in entities.iter().enumerate() {
            if from_index == to_index {
                continue;
            }
            let to_lower = normalize(&to.name);
            if from_lower == to_lower {
                continue;
            }
            for (phrase, relation_type) in RELATION_PATTERNS {
                let needle = format!("{from_lower} {phrase} {to_lower}");
                if !haystack.contains(&needle) {
                    continue;
                }
                let relation = ExtractedRelation {
                    from: from.name.clone(),
                    to: to.name.clone(),
                    relation_type: (*relation_type).to_owned(),
                };
                let duplicate = relations.iter().any(|existing| {
                    existing.from == relation.from
                        && existing.to == relation.to
                        && existing.relation_type == relation.relation_type
                });
                if !duplicate {
                    relations.push(relation);
                }
                break;
            }
        }
    }
    relations
}

/// Lowercases `text` and collapses every whitespace run to one space.
fn normalize(text: &str) -> String {
    let mut normalized = String::with_capacity(text.len());
    let mut pending_space = false;
    for ch in text.chars() {
        if ch.is_whitespace() {
            pending_space = !normalized.is_empty();
            continue;
        }
        if pending_space {
            normalized.push(' ');
            pending_space = false;
        }
        normalized.extend(ch.to_lowercase());
    }
    normalized
}

/// Parses a model answer into an [`Extraction`], returning `None` when unusable.
///
/// The parser accepts a fenced code block, an object or array under `entities`/
/// `relations`, alias field names, and surrounding prose around the JSON payload.
/// Blank names and incomplete or self-referential relations are skipped; an answer
/// with neither entities nor relations is rejected.
fn parse_extraction(raw: &str) -> Option<Extraction> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    let body = fenced_body(trimmed).unwrap_or(trimmed);
    let slice = json_slice(body)?;
    let value: Value = serde_json::from_str(slice).ok()?;
    let (entities, relations) = match &value {
        Value::Array(items) => (parse_entities(items), Vec::new()),
        Value::Object(map) => {
            let entities = parse_entities_field(map.get("entities"));
            let relations = parse_relations_field(map.get("relations"));
            (entities, relations)
        }
        _ => return None,
    };
    if entities.is_empty() && relations.is_empty() {
        None
    } else {
        Some(Extraction {
            entities,
            relations,
        })
    }
}

/// Returns the content of the first fenced code block, minus a `json` language tag.
fn fenced_body(raw: &str) -> Option<&str> {
    let open = raw.find("```")?;
    let after_open = raw.get(open + 3..)?;
    let close = after_open.find("```")?;
    let body = after_open.get(..close)?;
    Some(strip_language_tag(body))
}

/// Drops a leading `json`/`JSON` tag from a fenced block body.
fn strip_language_tag(body: &str) -> &str {
    let trimmed = body.trim_start();
    let Some(head) = trimmed.get(..4) else {
        return trimmed;
    };
    if !head.eq_ignore_ascii_case("json") {
        return trimmed;
    }
    let rest = trimmed.get(4..).unwrap_or("");
    let followed_by_whitespace = match rest.chars().next() {
        Some(ch) => ch.is_whitespace(),
        None => true,
    };
    if followed_by_whitespace {
        rest.trim_start()
    } else {
        trimmed
    }
}

/// Cuts the JSON payload out of surrounding prose.
fn json_slice(text: &str) -> Option<&str> {
    let object_start = text.find('{');
    let array_start = text.find('[');
    let (start, close) = match (object_start, array_start) {
        (Some(object), Some(array)) if array < object => (array, ']'),
        (Some(object), _) => (object, '}'),
        (None, Some(array)) => (array, ']'),
        (None, None) => return None,
    };
    let end = text.rfind(close)?;
    if end <= start {
        return None;
    }
    text.get(start..=end)
}

/// Expands an `entities`/`relations` value into items (array or single object).
fn value_items(value: Option<&Value>) -> Vec<&Value> {
    match value {
        Some(Value::Array(items)) => items.iter().collect(),
        Some(item) if item.is_object() => vec![item],
        _ => Vec::new(),
    }
}

/// Parses an already-collected entity array.
fn parse_entities(items: &[Value]) -> Vec<ExtractedEntity> {
    items.iter().filter_map(parse_entity).collect()
}

/// Parses the `entities` field.
fn parse_entities_field(value: Option<&Value>) -> Vec<ExtractedEntity> {
    value_items(value)
        .into_iter()
        .filter_map(parse_entity)
        .collect()
}

/// Parses the `relations` field.
fn parse_relations_field(value: Option<&Value>) -> Vec<ExtractedRelation> {
    value_items(value)
        .into_iter()
        .filter_map(parse_relation)
        .collect()
}

/// Parses one entity object, tolerating alias field names.
fn parse_entity(value: &Value) -> Option<ExtractedEntity> {
    let map = value.as_object()?;
    let name = string_field(map, &["name", "entity", "text"])?.trim();
    if name.is_empty() {
        return None;
    }
    let entity_type = string_field(map, &["type", "entity_type"])
        .map(str::trim)
        .filter(|label| !label.is_empty())
        .unwrap_or("Concept");
    Some(ExtractedEntity {
        name: name.to_owned(),
        entity_type: entity_type.to_owned(),
        importance: map.get("importance").and_then(clamped_importance),
    })
}

/// Parses one relation object, tolerating alias field names and skipping self-links.
fn parse_relation(value: &Value) -> Option<ExtractedRelation> {
    let map = value.as_object()?;
    let from = string_field(map, &["from", "source", "from_entity"])?.trim();
    let to = string_field(map, &["to", "target", "to_entity"])?.trim();
    let relation_type = string_field(map, &["type", "relation_type"])?.trim();
    if from.is_empty() || to.is_empty() || relation_type.is_empty() {
        return None;
    }
    if from.eq_ignore_ascii_case(to) {
        return None;
    }
    Some(ExtractedRelation {
        from: from.to_owned(),
        to: to.to_owned(),
        relation_type: relation_type.to_owned(),
    })
}

/// Returns the first present string field among `keys`.
fn string_field<'a>(map: &'a Map<String, Value>, keys: &[&str]) -> Option<&'a str> {
    keys.iter()
        .find_map(|key| map.get(*key).and_then(Value::as_str))
}

/// Reads an `importance` number, rejecting non-finite values and clamping to `[0, 1]`.
fn clamped_importance(value: &Value) -> Option<f32> {
    let number = value.as_f64()?;
    if !number.is_finite() {
        return None;
    }
    Some(number.clamp(0.0, 1.0) as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use openjiuwen_common::error::Error;
    use openjiuwen_test_support::ScriptedModel;
    use tokio_util::sync::CancellationToken;

    fn scope() -> MemoryScope {
        MemoryScope::new("user", "scope")
    }

    fn heuristic(config: &GraphMemoryConfig) -> HeuristicExtractor {
        HeuristicExtractor::new(config)
    }

    async fn names(extractor: &dyn EntityExtractor, text: &str) -> Vec<String> {
        extractor
            .extract(text, &scope(), &CallOptions::new())
            .await
            .unwrap()
            .entities
            .into_iter()
            .map(|entity| entity.name)
            .collect()
    }

    async fn extract_heuristic(text: &str) -> Extraction {
        heuristic(&GraphMemoryConfig::default())
            .extract(text, &scope(), &CallOptions::new())
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn quoted_and_backticked_candidates_become_entities() {
        let text = r#"He said "Alice Smith" then ran `cargo test`."#;
        let extraction = extract_heuristic(text).await;
        let names: Vec<_> = extraction
            .entities
            .iter()
            .map(|entity| entity.name.as_str())
            .collect();
        assert_eq!(names, vec!["Alice Smith", "cargo test"]);
    }

    #[tokio::test]
    async fn mentions_and_paths_become_typed_entities() {
        let text = "Ask @bob about src/main.rs and Cargo.toml";
        let extraction = extract_heuristic(text).await;
        let pairs: Vec<_> = extraction
            .entities
            .iter()
            .map(|entity| (entity.name.as_str(), entity.entity_type.as_str()))
            .collect();
        assert_eq!(
            pairs,
            vec![
                ("bob", "Concept"),
                ("src/main.rs", "Project"),
                ("Cargo.toml", "Project"),
            ]
        );
    }

    #[tokio::test]
    async fn capitalized_runs_are_joined_and_ordered() {
        let text = "I met Bob in New York yesterday";
        assert_eq!(
            names(&heuristic(&GraphMemoryConfig::default()), text).await,
            vec!["Bob", "New York"]
        );
    }

    #[tokio::test]
    async fn sentence_initial_capitalized_runs_are_skipped() {
        let text = "Alice works at Acme";
        let names = names(&heuristic(&GraphMemoryConfig::default()), text).await;
        assert_eq!(names, vec!["Acme"]);
        assert!(!names.iter().any(|name| name == "Alice"));
    }

    #[tokio::test]
    async fn stopwords_short_tokens_and_duplicates_are_dropped() {
        assert_eq!(
            names(&heuristic(&GraphMemoryConfig::default()), "The A I we").await,
            Vec::<String>::new()
        );
        let names = names(
            &heuristic(&GraphMemoryConfig::default()),
            "I saw Bob and BOB again",
        )
        .await;
        assert_eq!(names, vec!["Bob"]);
    }

    #[tokio::test]
    async fn max_entities_caps_the_result() {
        let config = GraphMemoryConfig {
            max_entities: 1,
            ..Default::default()
        };
        let text = r#"He met "Alice" and "Bob" and "Carol""#;
        assert_eq!(names(&heuristic(&config), text).await, vec!["Alice"]);
        let config = GraphMemoryConfig {
            max_entities: 0,
            ..Default::default()
        };
        assert_eq!(names(&heuristic(&config), text).await, Vec::<String>::new());
    }

    #[tokio::test]
    async fn entity_types_mark_people_projects_and_concepts() {
        let text = r#"We asked "Alice" builds things and src/main.rs"#;
        let extraction = extract_heuristic(text).await;
        let pairs: Vec<_> = extraction
            .entities
            .iter()
            .map(|entity| (entity.name.as_str(), entity.entity_type.as_str()))
            .collect();
        assert_eq!(pairs, vec![("Alice", "Person"), ("src/main.rs", "Project")]);
    }

    #[tokio::test]
    async fn is_a_and_is_the_patterns_map_to_is_a() {
        let extraction = extract_heuristic("I know Rust is a Language").await;
        assert_eq!(
            extraction.relations,
            vec![ExtractedRelation {
                from: "Rust".into(),
                to: "Language".into(),
                relation_type: "is_a".into(),
            }]
        );
        let extraction = extract_heuristic("I know Rust is the Language").await;
        assert_eq!(extraction.relations.len(), 1);
        assert_eq!(extraction.relations[0].relation_type, "is_a");
    }

    #[tokio::test]
    async fn verb_pattern_relations_are_detected() {
        let cases = [
            ("I heard Bob works at Acme", "works_at"),
            ("I heard Bob uses Rust", "uses"),
            ("I heard Rust depends on LLVM", "depends_on"),
            ("I heard Bob owns Acme", "owns"),
            ("I heard Bob is named Robert", "named"),
        ];
        for (text, expected) in cases {
            let extraction = extract_heuristic(text).await;
            assert!(
                extraction
                    .relations
                    .iter()
                    .any(|relation| relation.relation_type == expected),
                "{text:?} should yield a {expected} relation, got {:?}",
                extraction.relations
            );
        }
    }

    #[tokio::test]
    async fn relations_can_be_disabled() {
        let config = GraphMemoryConfig {
            extract_relations: false,
            ..Default::default()
        };
        let extraction = heuristic(&config)
            .extract("I heard Bob works at Acme", &scope(), &CallOptions::new())
            .await
            .unwrap();
        assert_eq!(extraction.entities.len(), 2);
        assert_eq!(extraction.relations, Vec::<ExtractedRelation>::new());
    }

    #[tokio::test]
    async fn relations_are_dropped_when_an_endpoint_is_filtered() {
        let extraction = extract_heuristic("I heard Bob works at The").await;
        assert_eq!(extraction.entities.len(), 1);
        assert_eq!(extraction.entities[0].name, "Bob");
        assert_eq!(extraction.relations, Vec::<ExtractedRelation>::new());
    }

    #[test]
    fn parse_extraction_reads_plain_json_objects() {
        let raw = r#"{
            "entities": [
                {"name": "Alice", "type": "Person", "importance": 0.9},
                {"name": "Acme", "type": "Organization"}
            ],
            "relations": [{"from": "Alice", "to": "Acme", "type": "works_at"}]
        }"#;
        let extraction = parse_extraction(raw).unwrap();
        assert_eq!(
            extraction.entities,
            vec![
                ExtractedEntity {
                    name: "Alice".into(),
                    entity_type: "Person".into(),
                    importance: Some(0.9),
                },
                ExtractedEntity {
                    name: "Acme".into(),
                    entity_type: "Organization".into(),
                    importance: None,
                },
            ]
        );
        assert_eq!(
            extraction.relations,
            vec![ExtractedRelation {
                from: "Alice".into(),
                to: "Acme".into(),
                relation_type: "works_at".into(),
            }]
        );
    }

    #[test]
    fn parse_extraction_reads_fenced_blocks() {
        let raw = "Here you go:\n```json\n{\"entities\": [{\"name\": \"Bob\"}]}\n```\n";
        let extraction = parse_extraction(raw).unwrap();
        assert_eq!(extraction.entities.len(), 1);
        assert_eq!(extraction.entities[0].name, "Bob");
        assert_eq!(extraction.entities[0].entity_type, "Concept");

        let raw = "```\n[{\"name\": \"Bob\"}]\n```";
        let extraction = parse_extraction(raw).unwrap();
        assert_eq!(extraction.entities[0].name, "Bob");
    }

    #[test]
    fn parse_extraction_reads_root_arrays_and_single_objects() {
        let extraction = parse_extraction(r#"[{"name": "Bob"}, {"entity": "Acme"}]"#).unwrap();
        assert_eq!(extraction.entities.len(), 2);
        assert_eq!(extraction.relations, Vec::<ExtractedRelation>::new());

        let raw = r#"{
            "entities": {"name": "Acme", "entity_type": "Organization", "importance": 2.0},
            "relations": {"source": "Bob", "target": "Acme", "relation_type": "works_at"}
        }"#;
        let extraction = parse_extraction(raw).unwrap();
        assert_eq!(extraction.entities[0].entity_type, "Organization");
        assert_eq!(extraction.entities[0].importance, Some(1.0));
        assert_eq!(extraction.relations[0].from, "Bob");
        assert_eq!(extraction.relations[0].relation_type, "works_at");
    }

    #[test]
    fn parse_extraction_rejects_unusable_output() {
        assert!(parse_extraction("").is_none());
        assert!(parse_extraction("no json here").is_none());
        assert!(parse_extraction("{}").is_none());
        assert!(parse_extraction(r#"{"entities": []}"#).is_none());
        assert!(parse_extraction("{not json}").is_none());
    }

    #[test]
    fn parse_extraction_skips_blank_names_and_clamps_importance() {
        let raw = r#"{"entities": [
            {"name": "   "},
            {"name": "Bob", "importance": -3.0},
            {"name": "Acme", "importance": 4.0}
        ]}"#;
        let extraction = parse_extraction(raw).unwrap();
        assert_eq!(extraction.entities.len(), 2);
        assert_eq!(extraction.entities[0].importance, Some(0.0));
        assert_eq!(extraction.entities[1].importance, Some(1.0));

        let raw = r#"{"entities": [{"name": "Bob", "importance": "high"}]}"#;
        let extraction = parse_extraction(raw).unwrap();
        assert_eq!(extraction.entities[0].importance, None);
    }

    #[test]
    fn parse_extraction_skips_self_and_incomplete_relations() {
        let raw = r#"{
            "entities": [{"name": "Bob"}],
            "relations": [
                {"from": "Bob", "to": "Bob", "type": "knows"},
                {"from": "Bob", "to": "", "type": "knows"},
                {"from": "Bob", "to": "Acme"},
                {"from": "Bob", "to": "Acme", "type": "works_at"}
            ]
        }"#;
        let extraction = parse_extraction(raw).unwrap();
        assert_eq!(
            extraction.relations,
            vec![ExtractedRelation {
                from: "Bob".into(),
                to: "Acme".into(),
                relation_type: "works_at".into(),
            }]
        );
    }

    #[tokio::test]
    async fn llm_extractor_parses_scripted_json() {
        let model = ScriptedModel::builder()
            .respond_text(
                r#"{"entities":[{"name":"Alice","type":"Person","importance":0.9}],
                    "relations":[{"from":"Alice","to":"Acme","type":"works_at"}]}"#,
            )
            .build();
        let handle = model.clone();
        let extractor = LlmExtractor::new(Arc::new(model), &GraphMemoryConfig::default());
        let extraction = extractor
            .extract("Alice works at Acme", &scope(), &CallOptions::new())
            .await
            .unwrap();
        assert_eq!(
            extraction.entities,
            vec![ExtractedEntity {
                name: "Alice".into(),
                entity_type: "Person".into(),
                importance: Some(0.9),
            }]
        );
        assert_eq!(extraction.relations.len(), 1);
        assert_eq!(handle.call_count(), 1);
        assert_eq!(handle.remaining(), 0);
    }

    #[tokio::test]
    async fn llm_extractor_accepts_fenced_output_and_caps_entities() {
        let model = ScriptedModel::builder()
            .respond_text("```json\n{\"entities\":[{\"name\":\"Alice\"},{\"name\":\"Bob\"}]}\n```")
            .build();
        let config = GraphMemoryConfig {
            max_entities: 1,
            ..Default::default()
        };
        let extractor = LlmExtractor::new(Arc::new(model), &config);
        let extraction = extractor
            .extract("Alice and Bob", &scope(), &CallOptions::new())
            .await
            .unwrap();
        assert_eq!(extraction.entities.len(), 1);
        assert_eq!(extraction.entities[0].name, "Alice");
    }

    #[tokio::test]
    async fn llm_extractor_falls_back_to_heuristics_after_retries() {
        let model = ScriptedModel::builder().build();
        let handle = model.clone();
        let config = GraphMemoryConfig::default();
        let extractor = LlmExtractor::new(Arc::new(model), &config);
        let text = "I heard Bob works at Acme";
        let extraction = extractor
            .extract(text, &scope(), &CallOptions::new())
            .await
            .unwrap();
        let expected = HeuristicExtractor::new(&config)
            .extract(text, &scope(), &CallOptions::new())
            .await
            .unwrap();
        assert_eq!(extraction, expected);
        assert_eq!(handle.call_count(), 4);
    }

    #[tokio::test]
    async fn llm_extractor_returns_cancelled_without_calling_the_model() {
        let model = ScriptedModel::text(r#"{"entities":[{"name":"Alice"}]}"#);
        let handle = model.clone();
        let extractor = LlmExtractor::new(Arc::new(model), &GraphMemoryConfig::default());
        let cancellation = CancellationToken::new();
        cancellation.cancel();
        let options = CallOptions::new().with_cancellation(cancellation);
        let error = extractor
            .extract("Alice", &scope(), &options)
            .await
            .unwrap_err();
        assert!(matches!(error, Error::Cancelled));
        assert_eq!(handle.call_count(), 0);
    }

    #[tokio::test]
    async fn llm_extractor_sends_the_system_prompt_and_user_text() {
        let model = ScriptedModel::builder()
            .respond_text(r#"{"entities":[{"name":"Alice"}]}"#)
            .build();
        let handle = model.clone();
        let extractor = LlmExtractor::new(Arc::new(model), &GraphMemoryConfig::default());
        let text = "Alice prefers dark mode";
        extractor
            .extract(text, &scope(), &CallOptions::new())
            .await
            .unwrap();
        let call = handle.last_call().expect("the model was called");
        let system = call.messages.first().expect("a system message");
        assert!(system.text().contains("entities"));
        assert_eq!(call.last_user_text().as_deref(), Some(text));
    }
}
