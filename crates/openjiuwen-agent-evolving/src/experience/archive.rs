//! The experience archive: deterministic dedupe, ranking and lifecycle storage.
//!
//! Python `experience/archive.py` and `experience/query.py` keep a JSON file per
//! skill and score candidates with a model. This crate keeps the same retrieval
//! shape as a pure in-memory archive:
//!
//! - [`ExperienceArchive::add`] inserts a record or merges it into a
//!   near-duplicate (content-token Jaccard at or above the archive's
//!   [`dedupe_threshold`](ExperienceArchive::dedupe_threshold)).
//! - [`ExperienceArchive::query`] ranks every record that passes the filters;
//!   ranking is a fixed linear blend of keyword overlap, text overlap and
//!   observed success rate, so it is reproducible from the stored values alone.
//! - [`tokens`] / [`jaccard`] / [`stopwords`] are the shared text primitives the
//!   extraction and sharing paths reuse, so dedupe behaves identically
//!   everywhere.
//!
//! No clock is read: recency is the archive's own monotonically increasing
//! sequence number ([`ExperienceArchive::seq`]), which makes archives
//! byte-identical across runs given the same insertion order.

use std::collections::{BTreeMap, BTreeSet};

use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};

use crate::experience::models::{
    ExperienceMatch, ExperienceQuery, ExperienceRecord, ExperienceSection,
};

/// The default number of records an archive keeps; `0` in
/// [`ExperienceArchive::with_capacity`] means unbounded.
pub const DEFAULT_CAPACITY: usize = 512;

/// The default content similarity at which two records are treated as
/// duplicates.
pub const DEFAULT_DEDUPE_THRESHOLD: f64 = 0.85;

/// The shortest ASCII token kept by [`tokens`]; one- and two-letter words are
/// mostly noise in English error text.
const MIN_ASCII_TOKEN_LEN: usize = 3;

/// The keyword weight used by [`ExperienceArchive::query`].
const KEYWORD_WEIGHT: f64 = 0.5;
/// The free-text overlap weight used by [`ExperienceArchive::query`].
const TEXT_WEIGHT: f64 = 0.3;
/// The observed-success-rate weight used by [`ExperienceArchive::query`].
const SUCCESS_WEIGHT: f64 = 0.2;

/// The English and Chinese stopwords shared by the experience and sharing paths.
///
/// The list is deliberately small and closed: it removes grammatical glue, not
/// domain vocabulary, so identifiers such as `read_file` or `timeout` survive
/// tokenisation. Callers that need their own list can build one and pass it to
/// [`crate::sharing::KeywordExtractor::with_stopwords`].
#[must_use]
pub fn stopwords() -> BTreeSet<String> {
    stopword_set().clone()
}

/// The cached stopword set, built once on first use.
///
/// The cache is immutable after initialisation, so it is a documented
/// memoisation of a constant table rather than mutable global state.
fn stopword_set() -> &'static BTreeSet<String> {
    static CACHE: std::sync::OnceLock<BTreeSet<String>> = std::sync::OnceLock::new();
    CACHE.get_or_init(build_stopwords)
}

/// Builds the stopword table.
fn build_stopwords() -> BTreeSet<String> {
    const WORDS: &[&str] = &[
        "a", "about", "after", "again", "all", "also", "am", "an", "and", "any", "are", "as", "at",
        "be", "because", "been", "before", "being", "but", "by", "can", "could", "did", "do",
        "does", "doing", "for", "from", "had", "has", "have", "having", "he", "her", "here", "him",
        "his", "how", "if", "in", "into", "is", "it", "its", "just", "me", "more", "most", "my",
        "no", "not", "now", "of", "on", "only", "or", "other", "our", "out", "over", "should",
        "so", "some", "such", "than", "that", "the", "their", "them", "then", "there", "these",
        "they", "this", "those", "to", "too", "up", "us", "very", "was", "we", "were", "what",
        "when", "where", "which", "while", "who", "why", "will", "with", "would", "you", "your",
    ];
    const CHINESE: &[&str] = &[
        "的", "了", "是", "在", "和", "我", "你", "他", "她", "它", "们", "也", "就", "都", "而",
        "及", "与", "或", "不", "没", "很", "更", "最", "这", "那", "有", "个", "上", "下", "中",
        "为", "以", "于", "被", "把", "到", "从", "对", "能", "会",
    ];
    WORDS
        .iter()
        .chain(CHINESE.iter())
        .map(|word| (*word).to_owned())
        .collect()
}

/// Whether `ch` is a CJK ideograph, kana, or hangul syllable.
///
/// Those scripts are not space-delimited, so [`tokens`] falls back to one token
/// per character for them instead of losing the text entirely.
fn is_cjk(ch: char) -> bool {
    matches!(ch,
        '\u{3040}'..='\u{30FF}'      // Hiragana and Katakana
        | '\u{3400}'..='\u{4DBF}'    // CJK extension A
        | '\u{4E00}'..='\u{9FFF}'    // CJK unified ideographs
        | '\u{AC00}'..='\u{D7AF}'    // Hangul syllables
        | '\u{F900}'..='\u{FAFF}'    // CJK compatibility ideographs
    )
}

/// Splits `text` into the lowercase tokens used for similarity.
///
/// ASCII alphanumeric runs become tokens when they are at least
/// [`MIN_ASCII_TOKEN_LEN`] characters long and not a stopword; underscores join
/// a run so tool identifiers such as `read_file` stay one token; CJK characters
/// become one token each (stopwords filtered). Every other character is a
/// separator. The result is sorted, so downstream dedupe is order-independent.
///
/// # Example
///
/// ```
/// use openjiuwen_agent_evolving::experience::tokens;
///
/// let tokens = tokens("Call read_file then the read_file again");
/// assert!(tokens.contains("read_file"));
/// assert!(!tokens.contains("the"));
/// ```
#[must_use]
pub fn tokens(text: &str) -> BTreeSet<String> {
    let stopwords = stopword_set();
    let mut tokens = BTreeSet::new();
    let mut ascii = String::new();
    for ch in text.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            ascii.push(ch.to_ascii_lowercase());
            continue;
        }
        flush_ascii(&mut ascii, stopwords, &mut tokens);
        if is_cjk(ch) {
            let symbol = ch.to_string();
            if !stopwords.contains(&symbol) {
                tokens.insert(symbol);
            }
        }
    }
    flush_ascii(&mut ascii, stopwords, &mut tokens);
    tokens
}

/// Pushes the pending ASCII run when it is long enough and not a stopword.
fn flush_ascii(ascii: &mut String, stopwords: &BTreeSet<String>, tokens: &mut BTreeSet<String>) {
    let has_letter = ascii.chars().any(|ch| ch.is_ascii_alphanumeric());
    if ascii.len() >= MIN_ASCII_TOKEN_LEN && has_letter && !stopwords.contains(ascii) {
        tokens.insert(std::mem::take(ascii));
    } else {
        ascii.clear();
    }
}

/// The Jaccard similarity of two token sets, in `0.0..=1.0`.
///
/// Two empty sets are dissimilar by definition (`0.0`), because they carry no
/// shared evidence at all.
///
/// # Example
///
/// ```
/// use openjiuwen_agent_evolving::experience::{jaccard, tokens};
///
/// let left = tokens("timeout while calling search");
/// let right = tokens("timeout while calling lookup");
/// assert!((jaccard(&left, &right) - 0.5).abs() < f64::EPSILON);
/// ```
#[must_use]
pub fn jaccard(left: &BTreeSet<String>, right: &BTreeSet<String>) -> f64 {
    let union = left.union(right).count();
    if union == 0 {
        return 0.0;
    }
    let intersection = left.intersection(right).count();
    intersection as f64 / union as f64
}

/// What [`ExperienceArchive::add`] did with the record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AddOutcome {
    /// The record was stored under this id.
    Inserted(String),
    /// The record was merged into an existing record.
    Merged {
        /// The id of the record that absorbed the new one.
        into: String,
        /// The content similarity that triggered the merge, in `0.0..=1.0`.
        similarity: f64,
    },
}

/// Counts returned by [`ExperienceArchive::merge_from`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct MergeSummary {
    /// Records that were new to the target archive.
    pub inserted: usize,
    /// Records that were merged into an existing record.
    pub merged: usize,
}

/// Aggregate numbers describing an archive, computed on demand.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ExperienceStats {
    /// How many records the archive holds.
    pub count: usize,
    /// How many of them are marked as shared.
    pub shared: usize,
    /// Mean [`ExperienceRecord::confidence`]; `0.0` for an empty archive.
    pub mean_confidence: f64,
    /// Mean [`ExperienceRecord::success_rate`]; `0.0` for an empty archive.
    pub mean_success_rate: f64,
    /// Record count per [`ExperienceSection::as_str`] label.
    pub sections: BTreeMap<String, usize>,
}

/// An in-memory, deduplicating store of experience records.
///
/// The archive is the unit the extractor fills and the lifecycle prunes. Its
/// fields are private because [`ExperienceArchive::add`] owns the sequence
/// counter and the capacity policy; read them through the accessors.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ExperienceArchive {
    records: Vec<ExperienceRecord>,
    capacity: usize,
    dedupe_threshold: f64,
    seq: u64,
}

impl Default for ExperienceArchive {
    fn default() -> Self {
        Self {
            records: Vec::new(),
            capacity: DEFAULT_CAPACITY,
            dedupe_threshold: DEFAULT_DEDUPE_THRESHOLD,
            seq: 0,
        }
    }
}

impl ExperienceArchive {
    /// Creates an empty archive with the default capacity and dedupe threshold.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns an empty archive keeping at most `capacity` records.
    ///
    /// `0` means unbounded. When a full archive receives a new record, the
    /// lowest-[`score`](ExperienceRecord::score) record is evicted (ties broken
    /// by the lexicographically larger `record_id`).
    #[must_use]
    pub fn with_capacity(mut self, capacity: usize) -> Self {
        self.capacity = capacity;
        self
    }

    /// Sets the content similarity at which records merge.
    ///
    /// The threshold is clamped into `0.0..=1.0`, so a non-finite value behaves
    /// like the nearest bound instead of poisoning every comparison.
    #[must_use]
    pub fn with_dedupe_threshold(mut self, threshold: f64) -> Self {
        self.dedupe_threshold = if threshold.is_nan() {
            DEFAULT_DEDUPE_THRESHOLD
        } else {
            threshold.clamp(0.0, 1.0)
        };
        self
    }

    /// The record capacity; `0` means unbounded.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// The content similarity at which records merge.
    #[must_use]
    pub const fn dedupe_threshold(&self) -> f64 {
        self.dedupe_threshold
    }

    /// The last sequence number the archive assigned.
    #[must_use]
    pub const fn seq(&self) -> u64 {
        self.seq
    }

    /// The stored records in insertion order.
    #[must_use]
    pub fn records(&self) -> &[ExperienceRecord] {
        &self.records
    }

    /// Iterates the stored records in insertion order.
    pub fn iter(&self) -> impl Iterator<Item = &ExperienceRecord> {
        self.records.iter()
    }

    /// Number of stored records.
    #[must_use]
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Whether the archive holds no records.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Looks a record up by id.
    #[must_use]
    pub fn get(&self, record_id: &str) -> Option<&ExperienceRecord> {
        self.records
            .iter()
            .find(|record| record.record_id == record_id)
    }

    /// Removes a record by id, returning it when it existed.
    pub fn remove(&mut self, record_id: &str) -> Option<ExperienceRecord> {
        let index = self
            .records
            .iter()
            .position(|record| record.record_id == record_id)?;
        Some(self.records.remove(index))
    }

    /// Applies `update` to the record with `record_id`.
    ///
    /// Returns `false` when no such record exists. The callback is the only way
    /// to mutate a stored record, so callers that change content or counters can
    /// keep the archive consistent; it does not re-stamp `updated_seq`.
    pub fn update(&mut self, record_id: &str, update: impl FnOnce(&mut ExperienceRecord)) -> bool {
        match self
            .records
            .iter_mut()
            .find(|record| record.record_id == record_id)
        {
            Some(record) => {
                update(record);
                true
            }
            None => false,
        }
    }

    /// Inserts `record`, or merges it into a near-duplicate.
    ///
    /// Similarity is content-token Jaccard, compared only between records with
    /// the same `skill_name` and `section`; the most similar candidate above
    /// [`dedupe_threshold`](Self::dedupe_threshold) wins. A merge keeps the
    /// archive's record id and content of the record with more successes (ties
    /// broken by higher confidence, then smaller id), unions keywords and
    /// metadata, sums the counters and adds one hit, takes the maximum weight
    /// and confidence, ORs `shared`, and re-stamps `updated_seq`.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] when the record fails
    /// [`ExperienceRecord::validate`].
    pub fn add(&mut self, record: ExperienceRecord) -> Result<AddOutcome> {
        record.validate()?;
        self.seq = self.seq.saturating_add(1);
        let seq = self.seq;
        let incoming_tokens = tokens(&record.content);
        let mut best: Option<(usize, f64)> = None;
        for (index, existing) in self.records.iter().enumerate() {
            if existing.skill_name != record.skill_name || existing.section != record.section {
                continue;
            }
            let similarity = jaccard(&tokens(&existing.content), &incoming_tokens);
            if similarity < self.dedupe_threshold {
                continue;
            }
            let better = match best {
                Some((_, current)) => similarity > current,
                None => true,
            };
            if better {
                best = Some((index, similarity));
            }
        }
        if let Some((index, similarity)) = best {
            let into = self.records[index].record_id.clone();
            self.records[index] = merge_records(&self.records[index], &record, seq);
            return Ok(AddOutcome::Merged { into, similarity });
        }
        let mut inserted = record;
        if inserted.created_seq == 0 {
            inserted.created_seq = seq;
        }
        inserted.updated_seq = seq;
        let record_id = inserted.record_id.clone();
        self.records.push(inserted);
        self.evict_overflow();
        Ok(AddOutcome::Inserted(record_id))
    }

    /// Evicts the lowest-scoring records until the archive fits its capacity.
    fn evict_overflow(&mut self) {
        if self.capacity == 0 {
            return;
        }
        while self.records.len() > self.capacity {
            let victim = self
                .records
                .iter()
                .enumerate()
                .min_by(|(_, left), (_, right)| {
                    left.score()
                        .total_cmp(&right.score())
                        .then_with(|| right.record_id.cmp(&left.record_id))
                })
                .map(|(index, _)| index);
            match victim {
                Some(index) => {
                    self.records.remove(index);
                }
                None => break,
            }
        }
    }

    /// Ranks every record that passes the query filters.
    ///
    /// The score is `0.5 * keyword_ratio + 0.3 * text_overlap + 0.2 *
    /// success_rate`, where `keyword_ratio` is the fraction of query keywords
    /// found on the record (case-insensitive), and `text_overlap` is the
    /// Jaccard similarity between the query text and the record content. Both
    /// terms are `0.0` when the corresponding query field is empty, so a
    /// filter-only query ranks by observed success rate. Results are sorted by
    /// score descending with `record_id` ascending as the deterministic
    /// tie-break, then truncated to [`ExperienceQuery::limit`] (`0` means
    /// unlimited).
    #[must_use]
    pub fn query(&self, query: &ExperienceQuery) -> Vec<ExperienceMatch> {
        let query_tokens = match query.text.as_deref() {
            Some(text) => tokens(text),
            None => BTreeSet::new(),
        };
        let query_keywords: BTreeSet<String> = query
            .keywords
            .iter()
            .map(|keyword| keyword.trim().to_lowercase())
            .filter(|keyword| !keyword.is_empty())
            .collect();
        let mut matches = Vec::new();
        for record in &self.records {
            if !query.include_shared && record.shared {
                continue;
            }
            if let Some(skill) = &query.skill_name {
                if record.skill_name != *skill {
                    continue;
                }
            }
            if let Some(section) = query.section {
                if record.section != section {
                    continue;
                }
            }
            let record_keywords: BTreeSet<String> = record
                .keywords
                .iter()
                .map(|keyword| keyword.trim().to_lowercase())
                .filter(|keyword| !keyword.is_empty())
                .collect();
            let matched_keywords: Vec<String> = query_keywords
                .intersection(&record_keywords)
                .cloned()
                .collect();
            let keyword_ratio = if query_keywords.is_empty() {
                0.0
            } else {
                matched_keywords.len() as f64 / query_keywords.len() as f64
            };
            let text_ratio = if query_tokens.is_empty() {
                0.0
            } else {
                jaccard(&query_tokens, &tokens(&record.content))
            };
            let score = KEYWORD_WEIGHT * keyword_ratio
                + TEXT_WEIGHT * text_ratio
                + SUCCESS_WEIGHT * record.success_rate();
            matches.push(ExperienceMatch::new(
                record.clone(),
                score,
                matched_keywords,
            ));
        }
        matches.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| left.record.record_id.cmp(&right.record.record_id))
        });
        if query.limit > 0 {
            matches.truncate(query.limit);
        }
        matches
    }

    /// Returns the `n` highest-scoring records, best first.
    ///
    /// Ties are broken by `record_id` ascending so the order never depends on
    /// insertion history.
    #[must_use]
    pub fn top(&self, n: usize) -> Vec<&ExperienceRecord> {
        let mut records: Vec<&ExperienceRecord> = self.records.iter().collect();
        records.sort_by(|left, right| {
            right
                .score()
                .total_cmp(&left.score())
                .then_with(|| left.record_id.cmp(&right.record_id))
        });
        records.truncate(n);
        records
    }

    /// Returns the records of one skill, in insertion order.
    #[must_use]
    pub fn by_skill(&self, skill_name: &str) -> Vec<&ExperienceRecord> {
        self.records
            .iter()
            .filter(|record| record.skill_name == skill_name)
            .collect()
    }

    /// Returns the records of one section, in insertion order.
    #[must_use]
    pub fn by_section(&self, section: ExperienceSection) -> Vec<&ExperienceRecord> {
        self.records
            .iter()
            .filter(|record| record.section == section)
            .collect()
    }

    /// Returns the records marked as shared, in insertion order.
    #[must_use]
    pub fn shared(&self) -> Vec<&ExperienceRecord> {
        self.records.iter().filter(|record| record.shared).collect()
    }

    /// Multiplies every record weight by `factor` and returns how many changed.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] when `factor` is not finite or outside
    /// `0.0..=1.0`; a larger factor would let decay inflate importance.
    pub fn decay(&mut self, factor: f64) -> Result<usize> {
        if !factor.is_finite() || !(0.0..=1.0).contains(&factor) {
            return Err(Error::validation(format!(
                "decay factor must be finite and between 0.0 and 1.0, got {factor}"
            )));
        }
        let mut changed = 0;
        for record in &mut self.records {
            let decayed = record.weight * factor;
            if (decayed - record.weight).abs() > f64::EPSILON {
                record.weight = decayed;
                changed += 1;
            }
        }
        Ok(changed)
    }

    /// Removes every record whose weight is below `weight`, returning the count.
    ///
    /// A non-finite threshold removes nothing, because no finite weight is
    /// below it.
    pub fn prune_below(&mut self, weight: f64) -> usize {
        let before = self.records.len();
        self.records.retain(|record| record.weight >= weight);
        before - self.records.len()
    }

    /// Adds every record of `other` through [`add`](Self::add), preserving
    /// dedupe across the two archives.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] from [`add`](Self::add) when a source record is
    /// invalid.
    pub fn merge_from(&mut self, other: &ExperienceArchive) -> Result<MergeSummary> {
        let mut summary = MergeSummary::default();
        for record in other.records() {
            match self.add(record.clone())? {
                AddOutcome::Inserted(_) => summary.inserted += 1,
                AddOutcome::Merged { .. } => summary.merged += 1,
            }
        }
        Ok(summary)
    }

    /// Computes aggregate statistics over the stored records.
    #[must_use]
    pub fn stats(&self) -> ExperienceStats {
        let count = self.records.len();
        let sections = self.records.iter().fold(
            BTreeMap::new(),
            |mut sections: BTreeMap<String, usize>, record| {
                *sections
                    .entry(record.section.as_str().to_owned())
                    .or_insert(0) += 1;
                sections
            },
        );
        if count == 0 {
            return ExperienceStats {
                count,
                shared: 0,
                mean_confidence: 0.0,
                mean_success_rate: 0.0,
                sections,
            };
        }
        let total_confidence: f64 = self.records.iter().map(|record| record.confidence).sum();
        let total_success: f64 = self
            .records
            .iter()
            .map(ExperienceRecord::success_rate)
            .sum();
        let shared = self.records.iter().filter(|record| record.shared).count();
        ExperienceStats {
            count,
            shared,
            mean_confidence: total_confidence / count as f64,
            mean_success_rate: total_success / count as f64,
            sections,
        }
    }
}

/// Whether the existing record should keep its content when merging.
fn prefer_existing(existing: &ExperienceRecord, incoming: &ExperienceRecord) -> bool {
    if existing.successes != incoming.successes {
        return existing.successes > incoming.successes;
    }
    if (existing.confidence - incoming.confidence).abs() > f64::EPSILON {
        return existing.confidence > incoming.confidence;
    }
    existing.record_id <= incoming.record_id
}

/// The earlier of two sequence stamps, ignoring unassigned (`0`) stamps.
fn earlier_seq(left: u64, right: u64) -> u64 {
    match (left, right) {
        (0, other) | (other, 0) => other,
        (left, right) => left.min(right),
    }
}

/// Merges `incoming` into `existing` under the archive's merge policy.
fn merge_records(
    existing: &ExperienceRecord,
    incoming: &ExperienceRecord,
    seq: u64,
) -> ExperienceRecord {
    let keep_existing = prefer_existing(existing, incoming);
    let winner = if keep_existing { existing } else { incoming };
    let loser = if keep_existing { incoming } else { existing };
    let mut keywords: BTreeSet<String> = winner.keywords.iter().cloned().collect();
    keywords.extend(loser.keywords.iter().cloned());
    let mut metadata = winner.metadata.clone();
    for (key, value) in &loser.metadata {
        metadata.entry(key.clone()).or_insert_with(|| value.clone());
    }
    ExperienceRecord {
        record_id: existing.record_id.clone(),
        skill_name: winner.skill_name.clone(),
        section: winner.section,
        content: winner.content.clone(),
        keywords: keywords.into_iter().collect(),
        source: winner.source.clone(),
        confidence: winner.confidence.max(loser.confidence),
        weight: winner.weight.max(loser.weight),
        hits: existing
            .hits
            .saturating_add(incoming.hits)
            .saturating_add(1),
        successes: existing.successes.saturating_add(incoming.successes),
        failures: existing.failures.saturating_add(incoming.failures),
        shared: existing.shared || incoming.shared,
        created_seq: earlier_seq(existing.created_seq, incoming.created_seq),
        updated_seq: seq,
        metadata,
    }
}
