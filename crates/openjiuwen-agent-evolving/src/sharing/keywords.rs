//! Deterministic keyword extraction and summarisation for the sharing path.
//!
//! Python's extractor calls a model for both the upload keywords and the
//! download query keywords; that pipeline is deferred. This module keeps the
//! offline half — tokenising, ranking and truncating text — so bundles carry
//! usable retrieval metadata without any model call.
//!
//! The stopword vocabulary comes from [`crate::experience::stopwords`], shared
//! with the experience archive, so both paths agree on what a content word is.

use std::collections::{BTreeMap, BTreeSet};

/// The default number of keywords one extraction keeps.
const DEFAULT_MAX_KEYWORDS: usize = 8;
/// The default minimum length of an ASCII keyword run.
const DEFAULT_MIN_LEN: usize = 3;
/// The CJK sentence terminators, alongside the ASCII ones.
const CJK_TERMINATORS: [char; 3] = ['。', '！', '？'];

/// Extracts ranked keywords and summaries from text.
///
/// Every method is deterministic and allocation-only: no clocks, randomness,
/// model calls or environment reads.
#[derive(Debug, Clone, PartialEq)]
pub struct KeywordExtractor {
    /// The maximum number of keywords kept (`0` means unlimited).
    pub max_keywords: usize,
    /// The minimum length of an ASCII token run.
    pub min_len: usize,
    /// Words dropped from the extraction, seeded from the shared vocabulary.
    pub stopwords: BTreeSet<String>,
}

impl Default for KeywordExtractor {
    fn default() -> Self {
        Self {
            max_keywords: DEFAULT_MAX_KEYWORDS,
            min_len: DEFAULT_MIN_LEN,
            stopwords: crate::experience::stopwords(),
        }
    }
}

impl KeywordExtractor {
    /// Creates an extractor with the default limits and the shared stopwords.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the maximum number of keywords; `0` means unlimited.
    #[must_use]
    pub fn with_max_keywords(mut self, max_keywords: usize) -> Self {
        self.max_keywords = max_keywords;
        self
    }

    /// Sets the minimum length of an ASCII token run.
    #[must_use]
    pub fn with_min_len(mut self, min_len: usize) -> Self {
        self.min_len = min_len;
        self
    }

    /// Replaces the stopword vocabulary.
    #[must_use]
    pub fn with_stopwords(mut self, stopwords: BTreeSet<String>) -> Self {
        self.stopwords = stopwords;
        self
    }

    /// Extracts keywords from `text`, ranked by frequency then alphabetically.
    ///
    /// ASCII alphanumeric runs (underscores join a run) shorter than
    /// [`min_len`](Self::min_len) are dropped; the rest are lowercased and
    /// filtered through the stopwords, so identifiers such as `read_file`
    /// survive as one keyword. CJK ideographs, kana and hangul characters become
    /// one keyword each (the minimum length does not apply to them) and are
    /// stopword-filtered as well. The result is deduplicated and truncated to
    /// [`max_keywords`](Self::max_keywords).
    #[must_use]
    pub fn extract(&self, text: &str) -> Vec<String> {
        let mut counts: BTreeMap<String, usize> = BTreeMap::new();
        let mut ascii = String::new();
        for ch in text.chars() {
            if ch.is_ascii_alphanumeric() || ch == '_' {
                ascii.push(ch.to_ascii_lowercase());
                continue;
            }
            self.flush(&mut ascii, &mut counts);
            if is_cjk(ch) {
                let symbol = ch.to_string();
                if !self.stopwords.contains(&symbol) {
                    *counts.entry(symbol).or_default() += 1;
                }
            }
        }
        self.flush(&mut ascii, &mut counts);
        self.rank(counts)
    }

    /// Summarises `text` into at most `max_chars` characters.
    ///
    /// The input is trimmed and returned unchanged when it already fits. Longer
    /// input is split into sentences on `.`, `!`, `?`, the matching CJK
    /// terminators and newlines; sentences are kept while the joined result
    /// (separated by one space) fits. When not even the first sentence fits, the
    /// input is hard-truncated to exactly `max_chars` characters, ending in
    /// `…`. `max_chars == 0` returns the empty string. Truncation counts
    /// characters, not bytes, so the result is always valid UTF-8.
    #[must_use]
    pub fn summarise(&self, text: &str, max_chars: usize) -> String {
        if max_chars == 0 {
            return String::new();
        }
        let trimmed = text.trim();
        if trimmed.chars().count() <= max_chars {
            return trimmed.to_owned();
        }
        let mut summary = String::new();
        let mut used = 0;
        for sentence in split_sentences(trimmed) {
            let length = sentence.chars().count();
            let separator = usize::from(!summary.is_empty());
            if used + separator + length > max_chars {
                break;
            }
            if !summary.is_empty() {
                summary.push(' ');
                used += 1;
            }
            summary.push_str(&sentence);
            used += length;
        }
        if summary.is_empty() {
            hard_truncate(trimmed, max_chars)
        } else {
            summary
        }
    }

    /// Aggregates keywords across `texts`.
    ///
    /// Each text contributes its extracted keywords once, so a word repeated
    /// inside one text does not dominate the aggregate. The result is ranked by
    /// how many texts mention the keyword, then alphabetically, and truncated to
    /// [`max_keywords`](Self::max_keywords).
    #[must_use]
    pub fn aggregate(&self, texts: &[String]) -> Vec<String> {
        let mut counts: BTreeMap<String, usize> = BTreeMap::new();
        for text in texts {
            for keyword in self.extract(text) {
                *counts.entry(keyword).or_default() += 1;
            }
        }
        self.rank(counts)
    }

    /// Pushes the pending ASCII run when it qualifies as a keyword.
    fn flush(&self, ascii: &mut String, counts: &mut BTreeMap<String, usize>) {
        let has_alphanumeric = ascii.chars().any(|ch| ch.is_ascii_alphanumeric());
        if has_alphanumeric
            && ascii.chars().count() >= self.min_len
            && !self.stopwords.contains(ascii)
        {
            *counts.entry(std::mem::take(ascii)).or_default() += 1;
        } else {
            ascii.clear();
        }
    }

    /// Orders counted keywords by frequency then alphabetically.
    fn rank(&self, counts: BTreeMap<String, usize>) -> Vec<String> {
        let mut ranked: Vec<(String, usize)> = counts.into_iter().collect();
        ranked.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));
        if self.max_keywords > 0 {
            ranked.truncate(self.max_keywords);
        }
        ranked.into_iter().map(|(keyword, _)| keyword).collect()
    }
}

/// Whether `ch` is a CJK ideograph, kana or hangul syllable.
///
/// Those scripts are not space-delimited, so one character per keyword is the
/// only lossless offline fallback.
fn is_cjk(ch: char) -> bool {
    matches!(ch,
        '\u{3040}'..='\u{30FF}'      // Hiragana and Katakana
        | '\u{3400}'..='\u{4DBF}'    // CJK extension A
        | '\u{4E00}'..='\u{9FFF}'    // CJK unified ideographs
        | '\u{AC00}'..='\u{D7AF}'    // Hangul syllables
        | '\u{F900}'..='\u{FAFF}'    // CJK compatibility ideographs
    )
}

/// Splits `text` into trimmed sentences, keeping terminators.
fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        if ch == '\n' || ch == '\r' {
            push_sentence(&mut sentences, &mut current);
            continue;
        }
        current.push(ch);
        if matches!(ch, '.' | '!' | '?') || CJK_TERMINATORS.contains(&ch) {
            push_sentence(&mut sentences, &mut current);
        }
    }
    push_sentence(&mut sentences, &mut current);
    sentences
}

/// Moves the non-empty trimmed `current` into `sentences` and clears it.
fn push_sentence(sentences: &mut Vec<String>, current: &mut String) {
    let trimmed = current.trim();
    if !trimmed.is_empty() {
        sentences.push(trimmed.to_owned());
    }
    current.clear();
}

/// Truncates `text` to exactly `max_chars` characters, ending in `…`.
fn hard_truncate(text: &str, max_chars: usize) -> String {
    if max_chars == 0 {
        return String::new();
    }
    let mut truncated: String = text.chars().take(max_chars - 1).collect();
    truncated.push('…');
    truncated
}
