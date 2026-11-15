//! The compact boundary code book and its prefix trie.
//!
//! Ports the pure-logic core of Python
//! `openjiuwen.symphony.retrieval.search.runtime.codebooks` (v0.1.20) together with the
//! validation Python keeps in `render/disclosure.py::_normalize_codebook`: a
//! [`CompactCodebook`](crate::retrieval::search::codebooks::CompactCodebook) rejects blank,
//! whitespace-bearing, duplicated and reserved-`"0"` entries, and a
//! [`CodeTrie`](crate::retrieval::search::codebooks::CodeTrie) answers the prefix questions
//! a constrained decoder asks (is this still a prefix of some code? is this a whole code?).
//!
//! # Deviation: the code book is generated, not tokenizer-tuned
//!
//! Python ships a fixed 512-entry permutation selected so every code is a single token for
//! its qwen tokenizer. No tokenizer exists in this workspace, so single-token-ness cannot
//! be verified; [`default_compact_boundary_codebook`] instead generates 512 distinct
//! two-uppercase-letter codes in the fixed `AA`, `AB`, … `ZZ` order (the first 512 of the
//! 676 available). Positional assignment, uniqueness, whitespace-freedom and "never `0`"
//! are preserved — those are the properties the decoder needs — and the order is
//! deterministic, which is what a prompt and a cache require.

use std::collections::BTreeSet;

use serde_json::Value;

use crate::error::{validation, Result};
use crate::retrieval::search::flow::ABSTAIN_TOKEN;

/// How many codes the default compact code book holds (Python's 512-entry tuple).
pub const DEFAULT_COMPACT_CODEBOOK_SIZE: usize = 512;

/// The first letter of the generated two-letter code alphabet.
const CODE_ALPHABET_START: u8 = b'A';

/// The generated code alphabet's size (the Latin uppercase letters).
const CODE_ALPHABET_SIZE: u8 = 26;

/// A validated list of compact candidate codes.
///
/// Codes are assigned positionally: the `n`-th exposed candidate gets `codes()[n]`. A
/// caller must therefore hold at least as many codes as it exposes — see
/// [`CompactCodebook::prefix`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CompactCodebook {
    /// The validated codes, trimmed and in assignment order.
    codes: Vec<String>,
}

impl CompactCodebook {
    /// Validates `codes` and stores them trimmed.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`](crate::error::Error) naming `compact_codebook` when an entry is
    /// blank, contains whitespace, repeats an earlier entry (after trimming), or is the
    /// reserved abstain token [`ABSTAIN_TOKEN`](crate::retrieval::search::flow::ABSTAIN_TOKEN).
    pub fn new(codes: impl IntoIterator<Item = String>) -> Result<Self> {
        let mut normalized: Vec<String> = Vec::new();
        let mut seen: BTreeSet<String> = BTreeSet::new();
        for raw in codes {
            let code = raw.trim();
            if code.is_empty() {
                return Err(validation(
                    "compact_codebook",
                    "entries must be non-empty strings",
                ));
            }
            if code == ABSTAIN_TOKEN {
                return Err(validation(
                    "compact_codebook",
                    "the reserved abstain code \"0\" cannot be a candidate code",
                ));
            }
            if code.chars().any(char::is_whitespace) {
                return Err(validation(
                    "compact_codebook",
                    format!("entry {code:?} cannot contain whitespace"),
                ));
            }
            if !seen.insert(code.to_owned()) {
                return Err(validation(
                    "compact_codebook",
                    format!("entry {code:?} is duplicated"),
                ));
            }
            normalized.push(code.to_owned());
        }
        Ok(Self { codes: normalized })
    }

    /// Reads a code book from a JSON value.
    ///
    /// A JSON array must hold strings; a JSON object is read through its `codes` (or
    /// `compact_boundary_codebook`) array. Anything else is rejected.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`](crate::error::Error) naming `compact_codebook` for a shape or an
    /// entry that is not a string, or for an entry [`CompactCodebook::new`] rejects.
    pub fn from_value(value: &Value) -> Result<Self> {
        match value {
            Value::Array(entries) => {
                let mut codes = Vec::with_capacity(entries.len());
                for (index, entry) in entries.iter().enumerate() {
                    let Some(code) = entry.as_str() else {
                        return Err(validation(
                            "compact_codebook",
                            format!("entry {index} must be a string"),
                        ));
                    };
                    codes.push(code.to_owned());
                }
                Self::new(codes)
            }
            Value::Object(map) => {
                for key in ["codes", "compact_boundary_codebook"] {
                    if let Some(nested) = map.get(key) {
                        return Self::from_value(nested);
                    }
                }
                Err(validation(
                    "compact_codebook",
                    "a code book object must hold a \"codes\" array",
                ))
            }
            _ => Err(validation(
                "compact_codebook",
                "must be a JSON array of strings or an object with a \"codes\" array",
            )),
        }
    }

    /// The validated codes, in assignment order.
    #[must_use]
    pub fn codes(&self) -> &[String] {
        &self.codes
    }

    /// How many codes the book holds.
    #[must_use]
    pub fn len(&self) -> usize {
        self.codes.len()
    }

    /// Whether the book holds no code.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.codes.is_empty()
    }

    /// The `index`-th code, or `None` when the book is shorter.
    #[must_use]
    pub fn code(&self, index: usize) -> Option<&str> {
        self.codes.get(index).map(String::as_str)
    }

    /// The first `count` codes, for an exposure of `count` selectable candidates.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`](crate::error::Error) naming `compact_codebook` when the book
    /// holds fewer than `count` codes — the same rejection Python's
    /// `build_exposed_fragment` raises instead of truncating the exposure.
    pub fn prefix(&self, count: usize) -> Result<Vec<String>> {
        if count > self.codes.len() {
            return Err(validation(
                "compact_codebook",
                format!(
                    "provides {} codes, but {count} selectable candidates were exposed",
                    self.codes.len()
                ),
            ));
        }
        Ok(self.codes[..count].to_vec())
    }

    /// A prefix trie over this book's codes.
    #[must_use]
    pub fn trie(&self) -> CodeTrie {
        CodeTrie::new(self.codes.iter().cloned())
    }
}

/// A prefix tree over a set of valid codes.
///
/// The trie answers the two questions a constrained decoder asks while reading a code:
/// "can this partial text still become a code?" and "is this text a code?".
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CodeTrie {
    /// Every valid code; the prefix test is a scan over this sorted set.
    ///
    /// A code book is a few hundred short strings, so a sorted set is both smaller and
    /// faster to build than a node-per-character tree, and its iteration order is the
    /// deterministic answer of [`CodeTrie::completions`].
    codes: BTreeSet<String>,
}

impl CodeTrie {
    /// Builds a trie over `codes`.
    #[must_use]
    pub fn new(codes: impl IntoIterator<Item = String>) -> Self {
        Self {
            codes: codes.into_iter().collect(),
        }
    }

    /// Whether some code starts with `prefix` (so the decoder may keep reading).
    ///
    /// An empty prefix is a valid prefix of an empty code book only when the book is
    /// non-empty; an empty book accepts nothing.
    #[must_use]
    pub fn is_valid_prefix(&self, prefix: &str) -> bool {
        self.codes.iter().any(|code| code.starts_with(prefix))
    }

    /// Whether `text` is exactly one of the codes.
    #[must_use]
    pub fn is_code(&self, text: &str) -> bool {
        self.codes.contains(text)
    }

    /// Every code starting with `prefix`, sorted.
    #[must_use]
    pub fn completions(&self, prefix: &str) -> Vec<String> {
        self.codes
            .iter()
            .filter(|code| code.starts_with(prefix))
            .cloned()
            .collect()
    }

    /// How many codes the trie holds.
    #[must_use]
    pub fn len(&self) -> usize {
        self.codes.len()
    }

    /// Whether the trie holds no code.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.codes.is_empty()
    }
}

/// The default compact code book: 512 distinct two-uppercase-letter codes.
///
/// The codes are the first [`DEFAULT_COMPACT_CODEBOOK_SIZE`] pairs of the fixed
/// `AA`, `AB`, … `AZ`, `BA`, … order. See the module docs for why the order is generated
/// instead of copied from Python's tokenizer-tuned tuple.
///
/// This cannot fail: the generator emits distinct, whitespace-free, non-`"0"` codes only.
#[must_use]
pub fn default_compact_boundary_codebook() -> CompactCodebook {
    let alphabet = CODE_ALPHABET_START..CODE_ALPHABET_START + CODE_ALPHABET_SIZE;
    let mut codes: Vec<String> = Vec::with_capacity(DEFAULT_COMPACT_CODEBOOK_SIZE);
    'outer: for first in alphabet.clone() {
        for second in alphabet.clone() {
            if codes.len() == DEFAULT_COMPACT_CODEBOOK_SIZE {
                break 'outer;
            }
            codes.push(format!("{}{}", char::from(first), char::from(second)));
        }
    }
    CompactCodebook { codes }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn codes(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_owned()).collect()
    }

    #[test]
    fn a_valid_codebook_trims_and_keeps_the_assignment_order() {
        let book = CompactCodebook::new(codes(&[" AA ", "BB", "C1"])).unwrap();
        assert_eq!(book.codes(), ["AA", "BB", "C1"]);
        assert_eq!(book.len(), 3);
        assert!(!book.is_empty());
        assert_eq!(book.code(1), Some("BB"));
        assert_eq!(book.code(3), None);
        assert_eq!(book.prefix(2).unwrap(), codes(&["AA", "BB"]));
        assert!(CompactCodebook::new(Vec::new()).unwrap().is_empty());
    }

    #[test]
    fn codebook_validation_rejects_blank_whitespace_duplicate_and_abstain_entries() {
        let error = CompactCodebook::new(codes(&["AA", "   "])).unwrap_err();
        assert!(error.message().contains("non-empty"), "{}", error.message());

        let error = CompactCodebook::new(codes(&["A A"])).unwrap_err();
        assert!(error.message().contains("whitespace"), "{}", error.message());

        let error = CompactCodebook::new(codes(&["AA", "AA"])).unwrap_err();
        assert!(error.message().contains("duplicated"), "{}", error.message());

        let error = CompactCodebook::new(codes(&["0"])).unwrap_err();
        assert!(error.message().contains("abstain"), "{}", error.message());

        // Trimming happens before the duplicate check, so these collide.
        assert!(CompactCodebook::new(codes(&["AA", " AA "])).is_err());
    }

    #[test]
    fn prefix_rejects_a_short_book() {
        let book = CompactCodebook::new(codes(&["AA"])).unwrap();
        assert_eq!(book.prefix(1).unwrap(), codes(&["AA"]));
        let error = book.prefix(2).unwrap_err();
        assert!(error.message().contains("provides 1 codes"), "{}", error.message());
    }

    #[test]
    fn codebook_reads_a_json_list_or_an_object_holding_one() {
        let book = CompactCodebook::from_value(&serde_json::json!(["AA", "BB"])).unwrap();
        assert_eq!(book.codes(), ["AA", "BB"]);
        let book = CompactCodebook::from_value(&serde_json::json!({"codes": ["AA"]})).unwrap();
        assert_eq!(book.codes(), ["AA"]);
        let book = CompactCodebook::from_value(&serde_json::json!({
            "compact_boundary_codebook": ["BB"]
        }))
        .unwrap();
        assert_eq!(book.codes(), ["BB"]);
        assert!(CompactCodebook::from_value(&serde_json::json!([1])).is_err());
        assert!(CompactCodebook::from_value(&serde_json::json!({"other": []})).is_err());
        assert!(CompactCodebook::from_value(&serde_json::json!("AA")).is_err());
        assert!(CompactCodebook::from_value(&serde_json::json!(["AA", "AA"])).is_err());
    }

    #[test]
    fn trie_answers_prefix_code_and_completion_questions() {
        let trie = CodeTrie::new(codes(&["AB", "AC", "BB"]));
        assert_eq!(trie.len(), 3);
        assert!(!trie.is_empty());
        assert!(trie.is_valid_prefix("A"));
        assert!(trie.is_valid_prefix("AB"));
        assert!(!trie.is_valid_prefix("AD"));
        assert!(!trie.is_valid_prefix("ABX"));
        assert!(trie.is_code("AB"));
        assert!(!trie.is_code("A"));
        assert_eq!(trie.completions("A"), codes(&["AB", "AC"]));
        assert_eq!(trie.completions(""), codes(&["AB", "AC", "BB"]));
        assert_eq!(trie.completions("Z"), Vec::<String>::new());

        let empty = CodeTrie::default();
        assert!(empty.is_empty());
        assert!(!empty.is_valid_prefix(""));
        assert_eq!(empty.completions("A"), Vec::<String>::new());
    }

    #[test]
    fn the_default_codebook_is_512_distinct_two_letter_codes() {
        let book = default_compact_boundary_codebook();
        assert_eq!(book.len(), DEFAULT_COMPACT_CODEBOOK_SIZE);
        let book_codes = book.codes();
        assert_eq!(book_codes[0], "AA");
        assert_eq!(book_codes[25], "AZ");
        assert_eq!(book_codes[26], "BA");
        assert_eq!(book_codes[511], "TR");
        let unique: BTreeSet<&String> = book_codes.iter().collect();
        assert_eq!(unique.len(), book_codes.len());
        assert!(book_codes
            .iter()
            .all(|code| code.len() == 2 && code.chars().all(|c| c.is_ascii_uppercase())));
        assert!(book_codes.iter().all(|code| code != ABSTAIN_TOKEN));

        let trie = book.trie();
        assert_eq!(trie.len(), DEFAULT_COMPACT_CODEBOOK_SIZE);
        assert!(trie.is_valid_prefix("T"));
        assert!(trie.is_code("TR"));
        assert_eq!(trie.completions("T").len(), 18);
        assert_eq!(trie.completions("TA"), codes(&["TA"]));
        assert_eq!(book.prefix(3).unwrap(), codes(&["AA", "AB", "AC"]));
    }
}
