//! Deterministic tokenization, `BM25` keyword scoring, and token-set similarity.
//!
//! Everything here is pure and synchronous: the same text always yields the same
//! tokens and the same scores, so recall ordering is reproducible in tests and across
//! processes. The corpus of a `BM25` run is the candidate set of a single query, which
//! is why [`bm25_scores`] takes the whole candidate list rather than precomputed
//! corpus statistics.
use std::collections::{HashMap, HashSet};
/// The default `BM25` term-frequency saturation parameter.
pub const DEFAULT_K1: f32 = 1.2;
/// The default `BM25` document-length normalization parameter.
pub const DEFAULT_B: f32 = 0.75;
/// Returns `true` for Han, Kana, and Hangul characters, which are tokenized as
/// character bigrams instead of whitespace/delimiter separated words.
fn is_cjk(c: char) -> bool {
    matches!(
        u32::from(c),
        0x3040..=0x30FF // Hiragana + Katakana
        | 0x3400..=0x4DBF // CJK Extension A
        | 0x4E00..=0x9FFF // CJK Unified Ideographs
        | 0xAC00..=0xD7AF // Hangul syllables
        | 0xF900..=0xFAFF // CJK Compatibility Ideographs
        | 0x2_0000..=0x2_FA1F // CJK Extension B and beyond
    )
}
/// Flushes the pending run of non-CJK alphanumeric characters as one token.
///
/// Runs of length `1` are dropped: a lone letter or digit carries no keyword signal
/// and would only add noise to `BM25` statistics.
fn flush_alnum(run: &mut String, tokens: &mut Vec<String>) {
    if run.is_empty() {
        return;
    }
    if run.chars().count() > 1 {
        tokens.push(std::mem::take(run));
    } else {
        run.clear();
    }
}
/// Flushes the pending run of CJK characters as unigrams/bigrams.
///
/// A run of `n` characters yields `n - 1` overlapping bigrams (the standard
/// segmentation-free approximation used for Chinese/Japanese text); a run of one
/// character stays a single unigram so short queries still match.
fn flush_cjk(run: &mut Vec<char>, tokens: &mut Vec<String>) {
    match run.len() {
        0 => {}
        1 => tokens.push(run[0].to_string()),
        _ => {
            for window in run.windows(2) {
                tokens.push(window.iter().collect());
            }
        }
    }
    run.clear();
}
/// Tokenizes text for keyword matching.
///
/// The rules are fixed and deterministic:
/// - the text is lowercased;
/// - every non-alphanumeric character is a separator;
/// - a CJK run becomes overlapping character bigrams, and a single CJK character
///   stays a unigram;
/// - tokens of length `1` are dropped unless they are CJK.
///
/// Repetitions are preserved (unlike a set) because `BM25` needs term frequencies.
#[must_use]
pub fn tokenize(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut alnum = String::new();
    let mut cjk = Vec::new();
    for c in text.chars() {
        if is_cjk(c) {
            flush_alnum(&mut alnum, &mut tokens);
            cjk.push(c);
        } else if c.is_alphanumeric() {
            flush_cjk(&mut cjk, &mut tokens);
            alnum.extend(c.to_lowercase());
        } else {
            flush_alnum(&mut alnum, &mut tokens);
            flush_cjk(&mut cjk, &mut tokens);
        }
    }
    flush_alnum(&mut alnum, &mut tokens);
    flush_cjk(&mut cjk, &mut tokens);
    tokens
}
/// Returns the distinct query tokens in first-occurrence order.
///
/// `BM25` sums one contribution per query term; repeating a term in the query would
/// scale the whole score instead of expressing coverage, so duplicates are dropped.
fn unique_terms(text: &str) -> Vec<String> {
    let mut seen = HashSet::new();
    tokenize(text)
        .into_iter()
        .filter(|token| seen.insert(token.clone()))
        .collect()
}
/// Returns `value` when it is a finite `BM25` parameter, else `fallback`.
fn sanitize_param(value: f32, fallback: f32) -> f64 {
    if value.is_finite() && value >= 0.0 {
        f64::from(value)
    } else {
        f64::from(fallback)
    }
}
/// Scores each document against the query with `BM25`.
///
/// For a query term `t` with document frequency `df` in a corpus of `N` documents,
/// the inverse document frequency is `ln(1 + (N - df + 0.5) / (df + 0.5))`, and the
/// term contribution to document `d` is
/// `idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))`
/// where `tf` is the term frequency in `d`, `dl` its token count, and `avgdl` the mean
/// token count of the corpus. Scores are raw and non-negative; they are `0.0` for an
/// empty query or an empty corpus, and the result always has the same length as
/// `documents`.
///
/// `k1` controls term-frequency saturation and `b` the document-length normalization;
/// non-finite or negative values fall back to [`DEFAULT_K1`]/[`DEFAULT_B`].
#[must_use]
pub fn bm25_scores(query: &str, documents: &[&str], k1: f32, b: f32) -> Vec<f32> {
    if documents.is_empty() {
        return Vec::new();
    }
    let query_terms = unique_terms(query);
    if query_terms.is_empty() {
        return vec![0.0; documents.len()];
    }
    let corpus: Vec<Vec<String>> = documents.iter().map(|text| tokenize(text)).collect();
    let corpus_size = corpus.len() as f64;
    let total_tokens: usize = corpus.iter().map(Vec::len).sum();
    let average_length = if total_tokens == 0 {
        0.0
    } else {
        total_tokens as f64 / corpus_size
    };
    let mut document_frequency: HashMap<&str, usize> = HashMap::new();
    for tokens in &corpus {
        let mut seen = HashSet::new();
        for token in tokens {
            if seen.insert(token.as_str()) {
                *document_frequency.entry(token.as_str()).or_insert(0) += 1;
            }
        }
    }
    let k1 = sanitize_param(k1, DEFAULT_K1);
    let b = if b.is_finite() {
        f64::from(b.clamp(0.0, 1.0))
    } else {
        f64::from(DEFAULT_B)
    };
    corpus
        .iter()
        .map(|tokens| {
            let length = tokens.len() as f64;
            let length_ratio = if average_length > 0.0 {
                length / average_length
            } else {
                1.0
            };
            let mut score = 0.0_f64;
            for term in &query_terms {
                let frequency = tokens.iter().filter(|token| *token == term).count() as f64;
                if frequency == 0.0 {
                    continue;
                }
                let frequency_in_corpus = document_frequency
                    .get(term.as_str())
                    .copied()
                    .unwrap_or_default() as f64;
                let idf = (1.0
                    + (corpus_size - frequency_in_corpus + 0.5) / (frequency_in_corpus + 0.5))
                    .ln();
                let denominator = frequency + k1 * (1.0 - b + b * length_ratio);
                score += idf * (frequency * (k1 + 1.0)) / denominator;
            }
            score as f32
        })
        .collect()
}
/// Scores each document with `BM25` (defaults `k1 = 1.2`, `b = 0.75`) and scales the
/// result so the best candidate is `1.0`.
///
/// The raw scores are divided by the maximum raw score; when that maximum is `0.0`
/// every element of the result is `0.0`. The vector is therefore directly usable as
/// the `keyword` component of the recall blend, whose inputs live in `[0, 1]`.
#[must_use]
pub fn normalized_bm25_scores(query: &str, documents: &[&str]) -> Vec<f32> {
    let raw = bm25_scores(query, documents, DEFAULT_K1, DEFAULT_B);
    let maximum = raw.iter().copied().fold(0.0_f32, f32::max);
    if maximum <= 0.0 || !maximum.is_finite() {
        return vec![0.0; raw.len()];
    }
    raw.into_iter()
        .map(|score| (score / maximum).clamp(0.0, 1.0))
        .collect()
}
/// Returns the fraction of distinct query terms that occur in `document`.
///
/// The result is in `[0, 1]` and is `0.0` for an empty query or a document that shares
/// no term. This is the semantic-free keyword signal used where no corpus statistics
/// exist (graph/entity recall): every query term counts equally.
#[must_use]
pub fn keyword_score(query: &str, document: &str) -> f32 {
    let query_terms = unique_terms(query);
    if query_terms.is_empty() {
        return 0.0;
    }
    let document_tokens: HashSet<String> = tokenize(document).into_iter().collect();
    let matched = query_terms
        .iter()
        .filter(|term| document_tokens.contains(*term))
        .count();
    matched as f32 / query_terms.len() as f32
}
/// Returns the token-set Jaccard similarity of two texts, in `[0, 1]`.
///
/// `|A ∩ B| / |A ∪ B|` over the token sets of both texts. Two texts with no tokens, or
/// no shared token, score `0.0`; identical token sets score `1.0`.
#[must_use]
pub fn jaccard(a: &str, b: &str) -> f32 {
    let first: HashSet<String> = tokenize(a).into_iter().collect();
    let second: HashSet<String> = tokenize(b).into_iter().collect();
    if first.is_empty() && second.is_empty() {
        return 0.0;
    }
    let intersection = first.intersection(&second).count();
    let union = first.union(&second).count();
    if union == 0 {
        return 0.0;
    }
    (intersection as f32 / union as f32).clamp(0.0, 1.0)
}
/// Returns the number of tokens [`tokenize`] produces for `text`.
///
/// This is the token count used by [`token_count`]-based budgeting heuristics, not a
/// provider tokenizer count.
#[must_use]
pub fn token_count(text: &str) -> usize {
    tokenize(text).len()
}
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    #[test]
    fn tokenizer_lowercases_and_splits_on_non_alphanumeric() {
        assert_eq!(
            tokenize("Alice PREFERS dark-mode; always!"),
            vec!["alice", "prefers", "dark", "mode", "always"]
        );
        assert_eq!(tokenize(""), Vec::<String>::new());
        assert_eq!(tokenize("a b c"), Vec::<String>::new());
        assert_eq!(tokenize("2024"), vec!["2024"]);
    }
    #[test]
    fn tokenizer_keeps_term_frequencies() {
        assert_eq!(tokenize("dark dark light"), vec!["dark", "dark", "light"]);
    }
    #[test]
    fn tokenizer_segments_cjk_runs_as_bigrams() {
        assert_eq!(tokenize("你好世界"), vec!["你好", "好世", "世界"]);
        assert_eq!(tokenize("我喜欢咖啡"), vec!["我喜", "喜欢", "欢咖", "咖啡"]);
        assert_eq!(tokenize("你"), vec!["你"]);
        assert_eq!(tokenize("我有 2 cats"), vec!["我有", "cats"]);
        assert_eq!(
            tokenize("使用 sqlite，数据库"),
            vec!["使用", "sqlite", "数据", "据库"]
        );
    }
    #[test]
    fn bm25_orders_by_query_overlap() {
        let documents = [
            "alice prefers dark mode",
            "bob likes light mode",
            "the weather is nice today",
        ];
        let scores = bm25_scores("dark mode preference", &documents, DEFAULT_K1, DEFAULT_B);
        assert_eq!(scores.len(), documents.len());
        assert!(scores[0] > scores[1]);
        assert!(scores[1] > scores[2]);
        assert_eq!(scores[2], 0.0);
        assert!(scores
            .iter()
            .all(|score| score.is_finite() && *score >= 0.0));
    }
    #[test]
    fn bm25_rewards_higher_term_frequency() {
        let documents = ["dark dark dark", "dark light", "unrelated words here"];
        let scores = bm25_scores("dark", &documents, DEFAULT_K1, DEFAULT_B);
        assert!(scores[0] > scores[1]);
        assert_eq!(scores[2], 0.0);
    }
    #[test]
    fn bm25_handles_empty_query_and_corpus() {
        assert_eq!(
            bm25_scores("dark", &[], DEFAULT_K1, DEFAULT_B),
            Vec::<f32>::new()
        );
        assert_eq!(
            bm25_scores("", &["dark mode"], DEFAULT_K1, DEFAULT_B),
            vec![0.0]
        );
        assert_eq!(
            bm25_scores("mode", &["", ""], DEFAULT_K1, DEFAULT_B),
            vec![0.0, 0.0]
        );
    }
    #[test]
    fn bm25_ignores_invalid_parameters() {
        let documents = ["dark mode", "light mode"];
        let invalid = bm25_scores("dark", &documents, f32::NAN, 5.0);
        let defaults = bm25_scores("dark", &documents, DEFAULT_K1, DEFAULT_B);
        assert_eq!(invalid, defaults);
    }
    #[test]
    fn normalized_bm25_scales_by_the_maximum() {
        let documents = ["dark", "dark mode and light", "nothing here"];
        let normalized = normalized_bm25_scores("dark", &documents);
        let maximum = normalized.iter().copied().fold(0.0_f32, f32::max);
        assert!((maximum - 1.0).abs() < 1e-6);
        assert!(normalized[0] > normalized[1]);
        assert_eq!(normalized[2], 0.0);
        assert!(normalized.iter().all(|score| (0.0..=1.0).contains(score)));
    }
    #[test]
    fn normalized_bm25_is_all_zero_without_matches() {
        assert_eq!(
            normalized_bm25_scores("absent", &["dark mode", "light mode"]),
            vec![0.0, 0.0]
        );
        assert_eq!(normalized_bm25_scores("dark", &["", ""]), vec![0.0, 0.0]);
        assert_eq!(normalized_bm25_scores("dark", &[]), Vec::<f32>::new());
    }
    #[test]
    fn keyword_score_measures_query_coverage() {
        let score = keyword_score("dark mode theme", "Alice prefers dark mode.");
        assert!((score - 2.0 / 3.0).abs() < 1e-6);
        assert!((keyword_score("dark", "dark mode") - 1.0).abs() < f32::EPSILON);
        assert_eq!(keyword_score("absent", "dark mode"), 0.0);
        assert_eq!(keyword_score("", "dark mode"), 0.0);
        assert_eq!(keyword_score("dark", ""), 0.0);
        // Repeated query terms count once.
        assert!((keyword_score("dark dark", "dark mode") - 1.0).abs() < f32::EPSILON);
    }
    #[test]
    fn jaccard_bounds() {
        assert!((jaccard("dark mode", "dark mode") - 1.0).abs() < f32::EPSILON);
        assert!((jaccard("dark mode", "mode dark") - 1.0).abs() < f32::EPSILON);
        // {dark, mode} vs {dark, light}: 1 shared of 3 union.
        assert!((jaccard("dark mode", "dark light") - 1.0 / 3.0).abs() < 1e-6);
        assert_eq!(jaccard("dark mode", "sunny weather"), 0.0);
        assert_eq!(jaccard("", ""), 0.0);
        assert_eq!(jaccard("", "dark"), 0.0);
        let score = jaccard("alice prefers dark mode", "alice prefers light mode");
        assert!((0.0..=1.0).contains(&score));
        assert!((score - 3.0 / 5.0).abs() < 1e-6);
    }
    #[test]
    fn token_count_matches_the_tokenizer() {
        assert_eq!(token_count("Alice prefers dark mode"), 4);
        assert_eq!(token_count("你好世界"), 3);
        assert_eq!(token_count(""), 0);
    }
    #[test]
    fn tokenizer_is_deterministic_for_cjk_and_ascii_mixes() {
        let text = "Use SQLite 数据库 for storage";
        assert_eq!(tokenize(text), tokenize(text));
        assert_eq!(
            tokenize(text),
            vec!["use", "sqlite", "数据", "据库", "for", "storage"]
        );
    }
}
