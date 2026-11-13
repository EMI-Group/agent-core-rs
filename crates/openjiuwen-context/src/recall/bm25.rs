//! Small dependency-free BM25 index for compression recall.
//!
//! Mirrors Python's `context_engine.processor.forked.compressor.recall.bm25`:
//! the same positive-IDF scoring formula (`k1 = 1.5`, `b = 0.75`), the same
//! mixed ASCII/CJK tokenizer (ASCII runs plus their `_`-split parts, CJK runs
//! plus per-character bigrams), and no stemming or stopword removal.
//!
//! ```
//! use openjiuwen_context::recall::bm25::Bm25Index;
//!
//! let index = Bm25Index::build(&["rust ownership", "python packaging"]);
//! let scores = index.scores("rust");
//! assert!(scores[0] > 0.0);
//! assert_eq!(scores[1], 0.0);
//! ```
use std::collections::HashMap;

/// The BM25 term-frequency saturation constant.
const K1: f64 = 1.5;

/// The BM25 document-length normalization constant.
const B: f64 = 0.75;

/// Whether `character` is part of a CJK run indexed as whole-run + bigrams.
fn is_cjk(character: char) -> bool {
    matches!(character, '\u{3400}'..='\u{4dbf}' | '\u{4e00}'..='\u{9fff}')
}

/// Flushes an accumulated ASCII run and its `_`-split parts into `tokens`.
fn push_ascii_run(run: &mut String, tokens: &mut Vec<String>) {
    if run.is_empty() {
        return;
    }
    tokens.push(run.clone());
    if run.contains('_') {
        tokens.extend(
            run.split('_')
                .filter(|part| !part.is_empty())
                .map(str::to_owned),
        );
    }
    run.clear();
}

/// Flushes an accumulated CJK run, its bigrams, into `tokens`.
fn push_cjk_run(run: &mut String, tokens: &mut Vec<String>) {
    if run.is_empty() {
        return;
    }
    tokens.push(run.clone());
    let characters: Vec<char> = run.chars().collect();
    for pair in characters.windows(2) {
        tokens.push(pair.iter().collect());
    }
    run.clear();
}

/// Tokenizes mixed Chinese, English, paths, and code identifiers.
///
/// The text is lowercased first. Runs of `[a-z0-9_]+` are emitted whole and,
/// when they contain underscores, also as their sub-parts. CJK runs are emitted
/// whole plus as per-character bigrams (a one-character run contributes only
/// itself). Every other character separates tokens.
#[must_use]
pub fn tokenize(text: &str) -> Vec<String> {
    let lowered = text.to_lowercase();
    let mut tokens = Vec::new();
    let mut ascii_run = String::new();
    let mut cjk_run = String::new();
    for character in lowered.chars() {
        if character.is_ascii_alphanumeric() || character == '_' {
            push_cjk_run(&mut cjk_run, &mut tokens);
            ascii_run.push(character);
        } else if is_cjk(character) {
            push_ascii_run(&mut ascii_run, &mut tokens);
            cjk_run.push(character);
        } else {
            push_ascii_run(&mut ascii_run, &mut tokens);
            push_cjk_run(&mut cjk_run, &mut tokens);
        }
    }
    push_ascii_run(&mut ascii_run, &mut tokens);
    push_cjk_run(&mut cjk_run, &mut tokens);
    tokens
}

/// An immutable BM25 index over a fixed document collection.
#[derive(Debug, Clone)]
pub struct Bm25Index {
    /// Per-document term frequencies.
    term_frequencies: Vec<HashMap<String, usize>>,
    /// Per-document token counts.
    lengths: Vec<usize>,
    /// Number of documents containing each term.
    document_frequencies: HashMap<String, usize>,
    /// Mean document length in tokens.
    average_length: f64,
    /// Term-frequency saturation constant.
    k1: f64,
    /// Document-length normalization constant.
    b: f64,
}

impl Bm25Index {
    /// Builds an index over `documents` with the default `k1`/`b` parameters.
    #[must_use]
    pub fn build(documents: &[&str]) -> Self {
        Self::with_params(documents, K1, B)
    }

    /// Builds an index with explicit BM25 parameters.
    #[must_use]
    pub fn with_params(documents: &[&str], k1: f64, b: f64) -> Self {
        let tokenized: Vec<Vec<String>> = documents.iter().map(|text| tokenize(text)).collect();
        let mut document_frequencies: HashMap<String, usize> = HashMap::new();
        let mut term_frequencies = Vec::with_capacity(tokenized.len());
        let mut lengths = Vec::with_capacity(tokenized.len());
        for document in &tokenized {
            let mut frequencies: HashMap<String, usize> = HashMap::new();
            for token in document {
                *frequencies.entry(token.clone()).or_default() += 1;
            }
            for token in frequencies.keys() {
                *document_frequencies.entry(token.clone()).or_default() += 1;
            }
            lengths.push(document.len());
            term_frequencies.push(frequencies);
        }
        let total_tokens: usize = lengths.iter().sum();
        let average_length = if lengths.is_empty() {
            0.0
        } else {
            total_tokens as f64 / lengths.len() as f64
        };
        Self {
            term_frequencies,
            lengths,
            document_frequencies,
            average_length,
            k1,
            b,
        }
    }

    /// Number of indexed documents.
    #[must_use]
    pub fn document_count(&self) -> usize {
        self.lengths.len()
    }

    /// Returns one BM25 score per indexed document, in document order.
    ///
    /// Query term frequency is ignored: every distinct query token contributes
    /// once, exactly like the Python implementation.
    #[must_use]
    pub fn scores(&self, query: &str) -> Vec<f64> {
        let document_count = self.lengths.len();
        if document_count == 0 {
            return Vec::new();
        }
        let query_tokens = tokenize(query);
        if query_tokens.is_empty() || self.average_length <= 0.0 {
            return vec![0.0; document_count];
        }
        let mut unique_tokens: Vec<&str> = query_tokens.iter().map(String::as_str).collect();
        unique_tokens.sort_unstable();
        unique_tokens.dedup();
        let document_count = document_count as f64;
        self.term_frequencies
            .iter()
            .zip(&self.lengths)
            .map(|(frequencies, &document_length)| {
                let mut score = 0.0;
                for token in &unique_tokens {
                    let Some(&frequency) = frequencies.get(*token) else {
                        continue;
                    };
                    if frequency == 0 {
                        continue;
                    }
                    let document_frequency = self
                        .document_frequencies
                        .get(*token)
                        .copied()
                        .unwrap_or_default() as f64;
                    let inverse_document_frequency = (1.0
                        + (document_count - document_frequency + 0.5) / (document_frequency + 0.5))
                        .ln();
                    let normalization =
                        1.0 - self.b + self.b * document_length as f64 / self.average_length;
                    let denominator = frequency as f64 + self.k1 * normalization;
                    score += inverse_document_frequency * frequency as f64 * (self.k1 + 1.0)
                        / denominator;
                }
                score
            })
            .collect()
    }
}

/// Scores `documents` against `query` with the default BM25 parameters.
#[must_use]
pub fn rank_bm25(query: &str, documents: &[&str]) -> Vec<f64> {
    Bm25Index::build(documents).scores(query)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenizes_ascii_identifiers_and_parts() {
        assert_eq!(
            tokenize("Read_File foo-BAR 42"),
            vec!["read_file", "read", "file", "foo", "bar", "42"]
        );
    }

    #[test]
    fn tokenizes_cjk_runs_and_bigrams() {
        assert_eq!(
            tokenize("你好世界"),
            vec!["你好世界", "你好", "好世", "世界"]
        );
        assert_eq!(tokenize("中"), vec!["中"]);
    }

    #[test]
    fn ranks_rare_terms_above_common_terms() {
        let index = Bm25Index::build(&[
            "rare token here",
            "common word",
            "common word too",
            "common again",
        ]);
        let scores = index.scores("rare common");
        assert!(scores[0] > scores[1]);
        assert!(scores[1] > 0.0);
    }

    #[test]
    fn ignores_query_term_frequency_and_empty_queries() {
        let index = Bm25Index::build(&["alpha beta", "gamma"]);
        let single = index.scores("alpha");
        let repeated = index.scores("alpha alpha alpha");
        assert!((single[0] - repeated[0]).abs() < 1e-12);
        assert_eq!(index.scores(""), vec![0.0, 0.0]);
        assert_eq!(index.scores("missing"), vec![0.0, 0.0]);
        assert_eq!(Bm25Index::build(&[]).scores("any").len(), 0);
    }

    #[test]
    fn matches_cjk_queries_to_cjk_documents() {
        let index = Bm25Index::build(&["你好世界", "hello world"]);
        let scores = index.scores("你好");
        assert!(scores[0] > 0.0);
        assert_eq!(scores[1], 0.0);
    }

    #[test]
    fn ties_keep_document_order() {
        let index = Bm25Index::build(&["same text", "same text"]);
        let scores = index.scores("same");
        assert!((scores[0] - scores[1]).abs() < 1e-12);
    }
}
