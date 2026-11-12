//! Deterministic BM25 keyword search over capability cards.
//!
//! Python's symphony `retrieval` package selects capabilities with an LLM-guided
//! progressive tree walk. The Rust port keeps the same job — pick the capabilities
//! that describe a query — with an offline, reproducible ranking: Okapi BM25 over
//! [`CapabilityDescriptor::retrieval_text`].
//!
//! Scoring is the canonical BM25 formula with `k1 = 1.2` and `b = 0.75`:
//! `idf = ln(1 + (N - df + 0.5) / (df + 0.5))` and, per term,
//! `tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))`, summed over the distinct
//! query terms. Ties break on the qualified id ascending, so identical corpora always
//! produce identical orderings — no hash-map iteration order leaks into a result.

use std::cmp::Ordering;
use std::collections::BTreeMap;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::model::CapabilityDescriptor;

/// The BM25 term-frequency saturation parameter.
const BM25_K1: f32 = 1.2;
/// The BM25 document-length normalization parameter.
const BM25_B: f32 = 0.75;

/// Splits `text` into lowercase tokens.
///
/// The text is split on every character that is not alphanumeric
/// (`char::is_alphanumeric`, so unicode letters and digits are kept), each token is
/// lowercased, and empty tokens are dropped. Punctuation therefore separates terms
/// and `"Adds, Two\nnumbers!"` tokenizes to `["adds", "two", "numbers"]`.
///
/// ```
/// use openjiuwen_symphony::search::tokenize;
///
/// assert_eq!(tokenize("Adds, Two\nnumbers!"), ["adds", "two", "numbers"]);
/// assert!(tokenize("   ").is_empty());
/// ```
#[must_use]
pub fn tokenize(text: &str) -> Vec<String> {
    text.split(|character: char| !character.is_alphanumeric())
        .filter(|token| !token.is_empty())
        .map(str::to_lowercase)
        .collect()
}

/// One keyword hit: the capability, its BM25 score, and the query terms it matched.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct LexicalHit {
    /// The qualified id (`"{capability_type}:{capability_id}"`) of the capability.
    pub qualified_id: String,
    /// The BM25 score; higher is more relevant.
    pub score: f32,
    /// The distinct query terms the capability's retrieval text matched, sorted.
    pub matched_terms: Vec<String>,
}

/// One indexed capability: its identity, its length, and its term frequencies.
#[derive(Debug, Clone)]
struct IndexedDocument {
    /// The qualified id of the capability.
    qualified_id: String,
    /// The number of tokens in the retrieval text (the BM25 `dl`).
    length: usize,
    /// How often each token occurs in the retrieval text (the BM25 `tf`).
    term_frequencies: BTreeMap<String, usize>,
}

/// A BM25 index over the retrieval text of a capability catalog.
///
/// The index is a point-in-time snapshot: [`LexicalIndex::build`] copies the cards, so
/// callers rebuild it after a catalog change instead of invalidating an incremental
/// index (catalogs are small, and a rebuild cannot go stale).
#[derive(Debug, Clone, Default)]
pub struct LexicalIndex {
    /// The indexed capabilities, sorted by qualified id.
    documents: Vec<IndexedDocument>,
    /// The number of documents containing each token (the BM25 `df`).
    document_frequencies: BTreeMap<String, usize>,
    /// The mean document length (the BM25 `avgdl`); `0.0` when the index is empty.
    average_length: f32,
}

impl LexicalIndex {
    /// Builds an index over the retrieval text of `descriptors`.
    ///
    /// Duplicate qualified ids collapse onto the last descriptor seen, which keeps the
    /// index (and therefore every ranking) deterministic.
    #[must_use]
    pub fn build(descriptors: impl IntoIterator<Item = CapabilityDescriptor>) -> Self {
        let mut unique: BTreeMap<String, CapabilityDescriptor> = BTreeMap::new();
        for descriptor in descriptors {
            unique.insert(descriptor.qualified_id(), descriptor);
        }

        let mut documents = Vec::with_capacity(unique.len());
        let mut document_frequencies: BTreeMap<String, usize> = BTreeMap::new();
        let mut total_length = 0_usize;
        for (qualified_id, descriptor) in unique {
            let mut term_frequencies: BTreeMap<String, usize> = BTreeMap::new();
            let mut length = 0_usize;
            for term in tokenize(&descriptor.retrieval_text()) {
                *term_frequencies.entry(term).or_insert(0) += 1;
                length += 1;
            }
            for term in term_frequencies.keys() {
                *document_frequencies.entry(term.clone()).or_insert(0) += 1;
            }
            total_length += length;
            documents.push(IndexedDocument {
                qualified_id,
                length,
                term_frequencies,
            });
        }

        #[allow(clippy::cast_precision_loss)] // catalog sizes fit f32 exactly enough.
        let average_length = if documents.is_empty() {
            0.0
        } else {
            total_length as f32 / documents.len() as f32
        };
        Self {
            documents,
            document_frequencies,
            average_length,
        }
    }

    /// Ranks the indexed capabilities against `query`.
    ///
    /// Only capabilities whose retrieval text contains at least one query term are
    /// returned. `top_k == 0` means unlimited; otherwise the result is truncated to
    /// `top_k`. Hits are ordered by descending score, then ascending qualified id. A
    /// blank query (or an empty index) yields no hits.
    #[must_use]
    #[allow(clippy::cast_precision_loss)] // document/term counts fit f32 exactly enough.
    pub fn search(&self, query: &str, top_k: usize) -> Vec<LexicalHit> {
        let mut terms = tokenize(query);
        terms.sort_unstable();
        terms.dedup();
        if terms.is_empty() || self.documents.is_empty() {
            return Vec::new();
        }

        let document_count = self.documents.len() as f32;
        let mut hits: Vec<LexicalHit> = Vec::with_capacity(self.documents.len());
        for document in &self.documents {
            let length = document.length as f32;
            // An all-blank corpus has `avgdl == 0`; skipping the length ratio keeps the
            // score finite (such documents match no term, so they never reach a hit).
            let length_ratio = if self.average_length > 0.0 {
                length / self.average_length
            } else {
                0.0
            };
            let normalization = BM25_K1 * (1.0 - BM25_B + BM25_B * length_ratio);
            let mut score = 0.0_f32;
            let mut matched_terms = Vec::new();
            for term in &terms {
                let Some(&frequency) = document.term_frequencies.get(term) else {
                    continue;
                };
                let Some(&document_frequency) = self.document_frequencies.get(term) else {
                    continue;
                };
                let occurrences = document_frequency as f32;
                let idf = (1.0 + (document_count - occurrences + 0.5) / (occurrences + 0.5)).ln();
                let term_frequency = frequency as f32;
                score +=
                    idf * (term_frequency * (BM25_K1 + 1.0)) / (term_frequency + normalization);
                matched_terms.push(term.clone());
            }
            if matched_terms.is_empty() {
                continue;
            }
            hits.push(LexicalHit {
                qualified_id: document.qualified_id.clone(),
                score,
                matched_terms,
            });
        }

        sort_lexical_hits(&mut hits);
        if top_k > 0 {
            hits.truncate(top_k);
        }
        hits
    }

    /// The number of indexed capabilities.
    #[must_use]
    pub fn len(&self) -> usize {
        self.documents.len()
    }

    /// Whether the index holds no capabilities.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.documents.is_empty()
    }
}

/// Orders hits by descending score, then ascending qualified id.
fn sort_lexical_hits(hits: &mut [LexicalHit]) {
    hits.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.qualified_id.cmp(&right.qualified_id))
    });
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
    use crate::model::CapabilityIo;

    fn descriptor(
        capability_type: &str,
        id: &str,
        name: &str,
        description: &str,
    ) -> CapabilityDescriptor {
        CapabilityDescriptor::new(capability_type, id, name)
            .with_description(description)
            .with_inputs(vec![CapabilityIo::new("raw", "string")])
    }

    #[test]
    fn tokenize_folds_case_and_splits_on_punctuation() {
        assert_eq!(tokenize("Adds, Two\nnumbers!"), ["adds", "two", "numbers"]);
        assert_eq!(tokenize("  "), Vec::<String>::new());
        assert_eq!(tokenize(""), Vec::<String>::new());
        assert_eq!(tokenize("a-b_c"), ["a", "b", "c"]);
    }

    #[test]
    fn tokenize_keeps_unicode_alphanumerics() {
        assert_eq!(tokenize("Grüße 世界 42"), ["grüße", "世界", "42"]);
    }

    #[test]
    fn best_match_ranks_first_and_reports_terms() {
        let index = LexicalIndex::build([
            descriptor("tool", "sum", "Sum", "adds two numbers"),
            descriptor(
                "tool",
                "translate",
                "Translate",
                "translates text between languages",
            ),
        ]);
        assert_eq!(index.len(), 2);
        assert!(!index.is_empty());

        let hits = index.search("numbers sum", 10);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].qualified_id, "tool:sum");
        assert_eq!(hits[0].matched_terms, ["numbers", "sum"]);
        assert!(hits[0].score > 0.0);
    }

    #[test]
    fn rankings_are_deterministic_and_tie_broken_by_id() {
        let index = LexicalIndex::build([
            descriptor("tool", "b", "Alpha", "shared words"),
            descriptor("tool", "a", "Alpha", "shared words"),
        ]);
        let first = index.search("shared words", 0);
        let second = index.search("shared words", 0);
        assert_eq!(first, second);
        assert_eq!(
            first
                .iter()
                .map(|hit| hit.qualified_id.as_str())
                .collect::<Vec<_>>(),
            ["tool:a", "tool:b"]
        );
    }

    #[test]
    fn top_k_truncates_and_zero_is_unlimited() {
        let index = LexicalIndex::build([
            descriptor("tool", "a", "Alpha", "shared words"),
            descriptor("tool", "b", "Beta", "shared words"),
            descriptor("tool", "c", "Gamma", "shared words"),
        ]);
        assert_eq!(index.search("shared", 1).len(), 1);
        assert_eq!(index.search("shared", 2).len(), 2);
        assert_eq!(index.search("shared", 0).len(), 3);
    }

    #[test]
    fn blank_and_unknown_queries_yield_no_hits() {
        let index = LexicalIndex::build([descriptor("tool", "sum", "Sum", "adds numbers")]);
        assert!(index.search("", 10).is_empty());
        assert!(index.search("   ", 10).is_empty());
        assert!(index.search("unknown", 10).is_empty());
        assert!(LexicalIndex::default().is_empty());
    }
}
