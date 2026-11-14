//! BM25F keyword ranking and snippet rendering for the `PersonalContext` search
//! index.
//!
//! This is the port of `_tokenize`, `_page_heading_fields`, `_search_page`,
//! `_average_field_lengths`, `_rank_bm25f` and `_snippet` from the Python
//! `openjiuwen.harness.personal_context.context_graph` module.
//!
//! Everything here is pure and offline: callers hand in [`SearchDocument`]
//! values, [`search`] returns ranked [`SearchHit`] values, and no function
//! touches the filesystem, the clock, the model or any global state. The
//! `PersonalContext` store owns ingestion and persistence; this module only
//! tokenizes, scores and renders excerpts.
//!
//! # Field weighting
//!
//! A page is indexed in four fields, mirroring Python's `_FIELD_WEIGHTS`:
//!
//! | Field | Weight | Content |
//! | --- | --- | --- |
//! | `title` | 4.0 | the file stem plus the document title |
//! | `path` | 3.0 | the workspace-relative path, `/` read as a separator |
//! | `headings` | 2.0 | the level-2 and level-3 subheadings |
//! | `body` | 1.0 | the whole page text |
//!
//! Field frequencies are combined with BM25F's per-field length
//! normalisation ([`BM25_K1`], [`BM25_B`]), so a term in a short title
//! outweighs the same term repeated in a long body.
//!
//! # Deviations from the Python original
//!
//! - Python folds with `str.casefold()`; this module folds with
//!   [`str::to_lowercase`]. They differ only for a handful of exotic
//!   ligatures (`ß`, `ẞ`, `İ`, `ﬁ`); for ASCII and CJK documents the token
//!   streams are identical.
//! - Python's `str.splitlines()` also breaks on `\v`, `\f`, `\x1c`–`\x1e`,
//!   `\x85`, `\u{2028}` and `\u{2029}`; [`page_heading_fields`] splits on
//!   `\n` and drops a trailing `\r`, which covers Markdown pages.
//! - [`search`] takes the result cap from the caller (clamped to
//!   [`MAX_SEARCH_RESULTS`]) and returns [`SearchHit`] values carrying their
//!   score, instead of Python's hard-coded ten-row JSON result list.
//! - The `title` field indexes the caller-supplied [`SearchDocument::title`]
//!   (ingestion fills it from the page's leading level-1 heading), while a
//!   hit's title is re-derived from the page text by
//!   [`page_heading_fields`], so a stale metadata title never hides the
//!   heading shown to the user.
//!
//! Nothing in this module panics on user input: page text, paths and queries
//! are arbitrary strings and are handled as data.
use std::collections::HashMap;
use std::sync::OnceLock;

use regex::Regex;
use serde::{Deserialize, Serialize};

/// The four weighted index fields, in the order Python iterates over them.
///
/// Index `0` is `title` (weight 4.0: the file stem plus the document title,
/// the strongest relevance signal), `1` is `path` (3.0: directory and file
/// name segments, a strong but weaker hint than an explicit title), `2` is
/// `headings` (2.0: the level-2/3 subheadings), and `3` is `body` (1.0: the
/// whole page text, the weakest per-occurrence signal).
pub const FIELD_WEIGHTS: [(&str, f64); 4] = [
    ("title", 4.0),
    ("path", 3.0),
    ("headings", 2.0),
    ("body", 1.0),
];

/// BM25 term-frequency saturation constant (`k1`).
pub const BM25_K1: f64 = 1.2;

/// BM25 field-length normalisation constant (`b`); `0.0` disables
/// normalisation, `1.0` normalises fully.
pub const BM25_B: f64 = 0.75;

/// Hard cap on returned hits, matching Python's `_MAX_SEARCH_RESULTS`.
pub const MAX_SEARCH_RESULTS: usize = 10;

/// Maximum number of characters in a rendered snippet.
pub const MAX_SNIPPET_CHARS: usize = 240;

/// Characters of context kept before the first match when a snippet is cut.
pub const SNIPPET_LEAD_CHARS: usize = 80;

/// Number of weighted fields; keeps the per-document field arrays in sync.
const FIELD_COUNT: usize = FIELD_WEIGHTS.len();

/// Python's `_SEARCH_TOKEN`: ASCII words plus CJK runs.
const TOKEN_PATTERN: &str = r"[a-z0-9_]+|[\u{3400}-\u{4dbf}\u{4e00}-\u{9fff}]+";

/// Python's `_HEADING`: `#`, `##` or `###` followed by whitespace and text.
const HEADING_PATTERN: &str = r"^(#{1,3})\s+(.+?)\s*$";

/// Compiled [`TOKEN_PATTERN`], shared process-wide.
static TOKEN_RE: OnceLock<Regex> = OnceLock::new();

/// Compiled [`HEADING_PATTERN`], shared process-wide.
static HEADING_RE: OnceLock<Regex> = OnceLock::new();

/// A page handed to the search index.
///
/// `source_id` links the page back to its ingested source, `path` is the
/// workspace-relative POSIX path, `title` is the document title (ingestion
/// derives it from the leading level-1 heading) and `text` is the full page
/// Markdown. Every field defaults to an empty string, so a partially
/// populated document is still searchable.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SearchDocument {
    /// Identifier of the ingested source the page belongs to.
    pub source_id: String,
    /// Workspace-relative POSIX path of the page.
    pub path: String,
    /// Document title, normally the page's leading level-1 heading.
    pub title: String,
    /// Full page text (Markdown).
    pub text: String,
}

impl SearchDocument {
    /// Builds a document from its parts.
    #[must_use]
    pub fn new(
        source_id: impl Into<String>,
        path: impl Into<String>,
        title: impl Into<String>,
        text: impl Into<String>,
    ) -> Self {
        Self {
            source_id: source_id.into(),
            path: path.into(),
            title: title.into(),
            text: text.into(),
        }
    }
}

/// One ranked search result.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SearchHit {
    /// Identifier of the ingested source the page belongs to.
    pub source_id: String,
    /// Page title: the first level-1 heading, or the file stem.
    pub title: String,
    /// Workspace-relative POSIX path of the page.
    pub path: String,
    /// Bounded excerpt around the first match, marked with `…` where cut.
    pub snippet: String,
    /// BM25F score; always strictly positive for a returned hit.
    pub score: f64,
}

impl SearchHit {
    /// Builds a hit.
    #[must_use]
    pub fn new(
        source_id: impl Into<String>,
        title: impl Into<String>,
        path: impl Into<String>,
        snippet: impl Into<String>,
        score: f64,
    ) -> Self {
        Self {
            source_id: source_id.into(),
            title: title.into(),
            path: path.into(),
            snippet: snippet.into(),
            score,
        }
    }
}

/// Splits `text` into lower-cased search tokens.
///
/// ASCII words (`[a-z0-9_]+`) are pushed whole; every CJK run
/// (`[\u{3400}-\u{4dbf}\u{4e00}-\u{9fff}]+`) is pushed as its single
/// characters followed by its adjacent character pairs, exactly like
/// Python's `_tokenize`. Punctuation and every other script are delimiters,
/// so an input with no tokens yields an empty vector.
///
/// # Examples
///
/// ```text
/// tokenize("Rust code") -> ["rust", "code"]
/// tokenize("中文")      -> ["中", "文", "中文"]
/// ```
#[must_use]
pub fn tokenize(text: &str) -> Vec<String> {
    let folded = text.to_lowercase();
    let mut tokens = Vec::new();
    for found in token_regex().find_iter(&folded) {
        let token = found.as_str();
        if token.is_ascii() {
            tokens.push(token.to_owned());
            continue;
        }
        let characters: Vec<char> = token.chars().collect();
        tokens.extend(characters.iter().map(char::to_string));
        for pair in characters.windows(2) {
            tokens.push(pair.iter().collect());
        }
    }
    tokens
}

/// Extracts the display title and the level-2/3 subheadings of a page.
///
/// The first ATX level-1 heading wins as the title; when a page has none the
/// file stem of `relative` is used instead. Level-2 and level-3 headings are
/// collected in document order and joined with a single space. Closing `#`
/// runs are stripped from every heading, and `\r\n` line endings are
/// tolerated.
///
/// # Examples
///
/// ```text
/// page_heading_fields("notes/daily.md", "# Monday\n## Plan\n### Details")
///     -> ("Monday", "Plan Details")
/// page_heading_fields("notes/daily.md", "no heading") -> ("daily", "")
/// ```
#[must_use]
pub fn page_heading_fields(relative: &str, text: &str) -> (String, String) {
    let mut title = String::new();
    let mut headings: Vec<String> = Vec::new();
    for line in text.split('\n') {
        let line = line.strip_suffix('\r').unwrap_or(line);
        let Some((level, heading)) = heading_parts(line) else {
            continue;
        };
        if level == 1 {
            if title.is_empty() {
                heading.clone_into(&mut title);
            }
        } else {
            headings.push(heading.to_owned());
        }
    }
    if title.is_empty() {
        path_stem(relative).clone_into(&mut title);
    }
    (title, headings.join(" "))
}

/// Ranks `documents` against `query` and returns the best [`SearchHit`]s.
///
/// A blank query, an empty document set, a `limit` of `0` and a query with no
/// usable tokens all return an empty vector. The effective cap is
/// `limit.min(MAX_SEARCH_RESULTS)`. Records are scored with BM25F over the
/// four fields of [`FIELD_WEIGHTS`] (distinct query terms only, in
/// first-seen order) and sorted by score descending; ties break on the page
/// path, ASCII-case-insensitively first and then by the raw path, so the
/// ordering is total and deterministic.
///
/// # Examples
///
/// ```text
/// let documents = [SearchDocument::new("src", "a.md", "Alpha", "alpha body")];
/// search(&documents, "alpha", 5).first().map(|hit| hit.path.as_str()) == Some("a.md")
/// ```
#[must_use]
pub fn search(documents: &[SearchDocument], query: &str, limit: usize) -> Vec<SearchHit> {
    let trimmed_query = query.trim();
    let cap = limit.min(MAX_SEARCH_RESULTS);
    if trimmed_query.is_empty() || documents.is_empty() || cap == 0 {
        return Vec::new();
    }
    let terms = distinct_terms(trimmed_query);
    if terms.is_empty() {
        return Vec::new();
    }
    let pages: Vec<DocumentStats> = documents.iter().map(document_stats).collect();
    let averages = average_lengths(&pages);
    let frequencies = document_frequencies(&pages, &terms);
    let mut scored: Vec<(f64, usize)> = Vec::new();
    for (index, page) in pages.iter().enumerate() {
        let score = bm25f_score(page, &terms, &frequencies, &averages, documents.len());
        if score > 0.0 {
            scored.push((score, index));
        }
    }
    scored.sort_by(|left, right| {
        right
            .0
            .total_cmp(&left.0)
            .then_with(|| {
                documents[left.1]
                    .path
                    .to_lowercase()
                    .cmp(&documents[right.1].path.to_lowercase())
            })
            .then_with(|| documents[left.1].path.cmp(&documents[right.1].path))
    });
    scored.truncate(cap);
    scored
        .into_iter()
        .map(|(score, index)| {
            let document = &documents[index];
            SearchHit {
                source_id: document.source_id.clone(),
                title: pages[index].title.clone(),
                path: document.path.clone(),
                snippet: snippet(&document.text, trimmed_query, &terms),
                score,
            }
        })
        .collect()
}

/// Renders a bounded, single-line excerpt of `text` around the query.
///
/// Runs of whitespace collapse to single spaces and the result is trimmed.
/// Text of at most [`MAX_SNIPPET_CHARS`] characters is returned unchanged.
/// Longer text is windowed around the first case-insensitive occurrence of
/// `query`, falling back to the earliest occurrence of any of `terms` and
/// then to the start of the text; the window keeps
/// [`SNIPPET_LEAD_CHARS`] characters of lead-in, spans at most
/// [`MAX_SNIPPET_CHARS`] characters, and gains a leading and/or trailing `…`
/// where it was cut. Indices are characters, never bytes, so multi-byte text
/// is never split.
///
/// # Examples
///
/// ```text
/// snippet("  short\n text ", "short", &[]) == "short text"
/// ```
#[must_use]
pub fn snippet(text: &str, query: &str, terms: &[String]) -> String {
    let collapsed = collapse_whitespace(text);
    let characters: Vec<char> = collapsed.chars().collect();
    if characters.len() <= MAX_SNIPPET_CHARS {
        return collapsed;
    }
    let folded = fold_chars(&collapsed);
    let index = first_match(&folded, &fold_chars(query.trim()))
        .or_else(|| earliest_term(&folded, terms))
        .unwrap_or(0)
        .min(characters.len());
    let lead = index.saturating_sub(SNIPPET_LEAD_CHARS);
    let end = lead.saturating_add(MAX_SNIPPET_CHARS).min(characters.len());
    let start = end.saturating_sub(MAX_SNIPPET_CHARS);
    let excerpt: String = characters[start..end].iter().collect();
    let mut rendered = String::with_capacity(excerpt.len() + 8);
    if start > 0 {
        rendered.push('…');
    }
    rendered.push_str(&excerpt);
    if end < characters.len() {
        rendered.push('…');
    }
    rendered
}

/// Per-field term frequencies and token count of one document.
#[derive(Debug, Default)]
struct FieldStat {
    /// Occurrences per token, in the field order of [`FIELD_WEIGHTS`].
    counts: HashMap<String, u32>,
    /// Total number of tokens in the field.
    length: u32,
}

/// The four field statistics of one document plus its display title.
#[derive(Debug, Default)]
struct DocumentStats {
    /// Field statistics, in the order of [`FIELD_WEIGHTS`].
    fields: [FieldStat; FIELD_COUNT],
    /// Display title derived from the page text.
    title: String,
}

/// Returns the compiled token pattern.
fn token_regex() -> &'static Regex {
    TOKEN_RE.get_or_init(|| {
        // The pattern is a constant covered by unit tests, so a compilation
        // failure can only mean the source itself is broken.
        Regex::new(TOKEN_PATTERN)
            .unwrap_or_else(|error| panic!("personal-context token pattern is invalid: {error}"))
    })
}

/// Returns the compiled heading pattern.
fn heading_regex() -> &'static Regex {
    HEADING_RE.get_or_init(|| {
        // Same invariant as `token_regex`: a constant, unit-tested pattern.
        Regex::new(HEADING_PATTERN)
            .unwrap_or_else(|error| panic!("personal-context heading pattern is invalid: {error}"))
    })
}

/// Splits one heading line into its hash level and cleaned heading text.
fn heading_parts(line: &str) -> Option<(usize, &str)> {
    let captures = heading_regex().captures(line)?;
    let level = captures.get(1)?.as_str().len();
    let heading = captures.get(2)?.as_str().trim_end_matches('#').trim();
    Some((level, heading))
}

/// Returns the file stem of a POSIX path, mirroring `PurePosixPath::stem`.
fn path_stem(relative: &str) -> &str {
    let name = relative.rsplit('/').next().unwrap_or(relative);
    match name.rfind('.') {
        Some(index) if index > 0 && index + 1 < name.len() => &name[..index],
        _ => name,
    }
}

/// Narrows a token count into the `u32` used by the BM25 arithmetic.
fn to_u32(value: usize) -> u32 {
    u32::try_from(value).unwrap_or(u32::MAX)
}

/// Counts the tokens of one field and returns its statistics.
fn field_stat(tokens: Vec<String>) -> FieldStat {
    let length = to_u32(tokens.len());
    let mut counts: HashMap<String, u32> = HashMap::with_capacity(tokens.len());
    for token in tokens {
        *counts.entry(token).or_insert(0) += 1;
    }
    FieldStat { counts, length }
}

/// Builds the four field statistics of one document.
fn document_stats(document: &SearchDocument) -> DocumentStats {
    let (title, headings) = page_heading_fields(&document.path, &document.text);
    let stem = path_stem(&document.path);
    let fields = [
        field_stat(tokenize(&format!("{stem} {}", document.title))),
        field_stat(tokenize(&document.path.replace('/', " "))),
        field_stat(tokenize(&headings)),
        field_stat(tokenize(&document.text)),
    ];
    DocumentStats { fields, title }
}

/// Returns the distinct query tokens in first-seen order.
fn distinct_terms(query: &str) -> Vec<String> {
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut terms = Vec::new();
    for token in tokenize(query) {
        if seen.insert(token.clone()) {
            terms.push(token);
        }
    }
    terms
}

/// Returns the average field length per field, floored at `1.0`.
fn average_lengths(pages: &[DocumentStats]) -> [f64; FIELD_COUNT] {
    let mut averages = [1.0; FIELD_COUNT];
    if pages.is_empty() {
        return averages;
    }
    let count = f64::from(to_u32(pages.len()));
    for (index, average) in averages.iter_mut().enumerate() {
        let total = pages.iter().fold(0_u32, |sum, page| {
            sum.saturating_add(page.fields[index].length)
        });
        *average = (f64::from(total) / count).max(1.0);
    }
    averages
}

/// Counts, per query term, the documents carrying it in at least one field.
fn document_frequencies(pages: &[DocumentStats], terms: &[String]) -> HashMap<String, u32> {
    let mut frequencies: HashMap<String, u32> = HashMap::with_capacity(terms.len());
    for term in terms {
        let count = pages
            .iter()
            .filter(|page| {
                page.fields
                    .iter()
                    .any(|field| field.counts.get(term).is_some_and(|count| *count > 0))
            })
            .count();
        frequencies.insert(term.clone(), to_u32(count));
    }
    frequencies
}

/// Scores one document against the query terms with BM25F.
fn bm25f_score(
    page: &DocumentStats,
    terms: &[String],
    frequencies: &HashMap<String, u32>,
    averages: &[f64; FIELD_COUNT],
    document_count: usize,
) -> f64 {
    let count = f64::from(to_u32(document_count));
    let mut score = 0.0;
    for term in terms {
        let frequency = frequencies.get(term).copied().unwrap_or(0);
        if frequency == 0 {
            continue;
        }
        let mut weighted = 0.0;
        for (index, (_name, weight)) in FIELD_WEIGHTS.iter().enumerate() {
            let field = &page.fields[index];
            let term_count = field.counts.get(term).copied().unwrap_or(0);
            if term_count == 0 {
                continue;
            }
            let length_ratio = f64::from(field.length) / averages[index];
            weighted += weight * f64::from(term_count) / (1.0 - BM25_B + BM25_B * length_ratio);
        }
        if weighted == 0.0 {
            continue;
        }
        let document_frequency = f64::from(frequency);
        let idf = (1.0 + (count - document_frequency + 0.5) / (document_frequency + 0.5)).ln();
        score += idf * (weighted * (BM25_K1 + 1.0) / (BM25_K1 + weighted));
    }
    score
}

/// Collapses whitespace runs to single spaces and trims the result.
fn collapse_whitespace(text: &str) -> String {
    let mut collapsed = String::with_capacity(text.len());
    let mut pending_space = false;
    for character in text.chars() {
        if character.is_whitespace() {
            pending_space = true;
            continue;
        }
        if pending_space && !collapsed.is_empty() {
            collapsed.push(' ');
        }
        pending_space = false;
        collapsed.push(character);
    }
    collapsed
}

/// Lower-cases `text` into its character sequence.
fn fold_chars(text: &str) -> Vec<char> {
    text.chars().flat_map(char::to_lowercase).collect()
}

/// Returns the first character index at which `needle` occurs in `haystack`.
fn first_match(haystack: &[char], needle: &[char]) -> Option<usize> {
    if needle.is_empty() || needle.len() > haystack.len() {
        return None;
    }
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

/// Returns the earliest position among the folded `terms`.
fn earliest_term(folded: &[char], terms: &[String]) -> Option<usize> {
    terms
        .iter()
        .filter_map(|term| first_match(folded, &fold_chars(term)))
        .min()
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    /// Builds a document owned by a single test source.
    fn document(path: &str, title: &str, text: &str) -> SearchDocument {
        SearchDocument::new("source", path, title, text)
    }

    /// Maps hits to their paths for ordering assertions.
    fn paths(hits: &[SearchHit]) -> Vec<&str> {
        hits.iter().map(|hit| hit.path.as_str()).collect()
    }

    #[test]
    fn tokenize_ascii_words() {
        assert_eq!(
            tokenize("Hello, World! snake_case 42"),
            ["hello", "world", "snake_case", "42"]
        );
    }

    #[test]
    fn tokenize_digits_and_underscores() {
        assert_eq!(tokenize("_42 a1_b2"), ["_42", "a1_b2"]);
    }

    #[test]
    fn tokenize_cjk_unigrams_then_bigrams() {
        assert_eq!(tokenize("中文"), ["中", "文", "中文"]);
        assert_eq!(tokenize("上下文"), ["上", "下", "文", "上下", "下文"]);
    }

    #[test]
    fn tokenize_mixed_ascii_and_cjk() {
        assert_eq!(tokenize("Rust语言2"), ["rust", "语", "言", "语言", "2"]);
    }

    #[test]
    fn tokenize_drops_punctuation_and_other_scripts() {
        assert_eq!(tokenize("a-b/c.d"), ["a", "b", "c", "d"]);
        assert_eq!(tokenize("— ¡! カタカナ"), Vec::<String>::new());
        assert!(tokenize("").is_empty());
        assert!(tokenize("!!!").is_empty());
    }

    #[test]
    fn page_heading_fields_extracts_title_and_subheadings() {
        let text = "# Weekly Plan\n\n## Goals\n### Details\nbody";
        let (title, headings) = page_heading_fields("notes/week.md", text);
        assert_eq!(title, "Weekly Plan");
        assert_eq!(headings, "Goals Details");
    }

    #[test]
    fn page_heading_fields_first_level_one_heading_wins() {
        let (title, headings) = page_heading_fields("notes/week.md", "## Sub\n# First\n# Second");
        assert_eq!(title, "First");
        assert_eq!(headings, "Sub");
    }

    #[test]
    fn page_heading_fields_falls_back_to_file_stem() {
        assert_eq!(
            page_heading_fields("notes/daily.log.md", "no heading here"),
            ("daily.log".to_owned(), String::new())
        );
        assert_eq!(
            page_heading_fields("notes/README", "plain"),
            ("README".to_owned(), String::new())
        );
    }

    #[test]
    fn page_heading_fields_strips_closing_hashes() {
        let (title, headings) = page_heading_fields("notes/x.md", "# Title ##\n### Sub ###");
        assert_eq!(title, "Title");
        assert_eq!(headings, "Sub");
    }

    #[test]
    fn page_heading_fields_handles_crlf() {
        let (title, headings) = page_heading_fields("notes/x.md", "# Title\r\n## Sub\r\nbody\r\n");
        assert_eq!(title, "Title");
        assert_eq!(headings, "Sub");
    }

    #[test]
    fn page_heading_fields_ignores_deeper_levels() {
        let (title, headings) = page_heading_fields("notes/x.md", "#### Deep\n# Title\nbody");
        assert_eq!(title, "Title");
        assert_eq!(headings, String::new());
    }

    #[test]
    fn search_title_match_outranks_body_only_match() {
        let documents = [
            document("alpha/doc.md", "zebra", "nothing at all"),
            document("beta/doc.md", "other", "zebra zebra nothing at all"),
        ];
        let hits = search(&documents, "zebra", 10);
        assert_eq!(paths(&hits), ["alpha/doc.md", "beta/doc.md"]);
        assert!(hits[0].score > hits[1].score);
    }

    #[test]
    fn search_field_weights_order_title_path_headings_body() {
        let documents = [
            document("alpha/doc.md", "zebra", "nothing here to see"),
            document("zebra/doc.md", "doc", "nothing here to see"),
            document(
                "beta/doc.md",
                "doc",
                "# Head\n## Zebra\nfiller filler filler filler",
            ),
            document("gamma/doc.md", "doc", "zebra nothing here to"),
        ];
        let hits = search(&documents, "zebra", 10);
        assert_eq!(
            paths(&hits),
            [
                "alpha/doc.md",
                "zebra/doc.md",
                "beta/doc.md",
                "gamma/doc.md"
            ]
        );
        assert!(hits.windows(2).all(|pair| pair[0].score > pair[1].score));
    }

    #[test]
    fn search_equal_scores_order_case_insensitively_by_path() {
        let documents = [
            document("B.md", "zebra", "x"),
            document("a.md", "zebra", "x"),
        ];
        let hits = search(&documents, "zebra", 10);
        assert_eq!(paths(&hits), ["a.md", "B.md"]);
        assert_eq!(hits[0].score, hits[1].score);
    }

    #[test]
    fn search_clamps_limit_to_maximum() {
        let documents: Vec<SearchDocument> = (0..12)
            .map(|index| document(&format!("page-{index}.md"), "doc", "zebra text"))
            .collect();
        let hits = search(&documents, "zebra", 100);
        assert_eq!(hits.len(), MAX_SEARCH_RESULTS);
        assert_eq!(search(&documents, "zebra", 2).len(), 2);
        assert_eq!(search(&documents, "zebra", 0), Vec::new());
    }

    #[test]
    fn search_returns_nothing_for_unknown_terms_and_blank_queries() {
        let documents = [document("a.md", "alpha", "hello world")];
        assert!(search(&documents, "zebra", 10).is_empty());
        assert!(search(&documents, "   ", 10).is_empty());
        assert!(search(&documents, "", 10).is_empty());
        assert!(search(&documents, "!!!", 10).is_empty());
        assert!(search(&[], "zebra", 10).is_empty());
    }

    #[test]
    fn search_hit_carries_heading_title_snippet_and_source() {
        let document = SearchDocument::new(
            "feed",
            "notes/raw.md",
            "stale metadata title",
            "# Real Title\n\nzebra facts and more zebra notes",
        );
        let hits = search(&[document], "zebra", 10);
        assert_eq!(hits.len(), 1);
        let hit = &hits[0];
        assert_eq!(hit.source_id, "feed");
        assert_eq!(hit.title, "Real Title");
        assert_eq!(hit.path, "notes/raw.md");
        assert!(hit.snippet.contains("zebra facts"));
        assert!(hit.score > 0.0);
    }

    #[test]
    fn search_matches_cjk_queries() {
        let documents = [document(
            "kb.md",
            "上下文",
            "# 上下文\n\n上下文引擎负责压缩长期会话",
        )];
        let hits = search(&documents, "上下文", 10);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].title, "上下文");
    }

    #[test]
    fn snippet_collapses_whitespace_and_returns_short_text() {
        assert_eq!(snippet("  short\n\t text ", "short", &[]), "short text");
        assert_eq!(snippet("", "short", &[]), "");
        assert_eq!(snippet(" \n ", "short", &[]), "");
    }

    #[test]
    fn snippet_centres_on_the_query_with_both_ellipses() {
        let text = format!("{}zebra{}", "a".repeat(300), "b".repeat(300));
        let expected = format!("…{}zebra{}…", "a".repeat(80), "b".repeat(155));
        assert_eq!(snippet(&text, "zebra", &[]), expected);
    }

    #[test]
    fn snippet_keeps_a_prefix_when_the_match_is_at_the_end() {
        let text = format!("{}zebra", "a".repeat(300));
        let expected = format!("…{}zebra", "a".repeat(235));
        assert_eq!(snippet(&text, "zebra", &[]), expected);
    }

    #[test]
    fn snippet_keeps_a_suffix_when_the_match_is_at_the_start() {
        let text = format!("zebra{}", "b".repeat(300));
        let expected = format!("zebra{}…", "b".repeat(235));
        assert_eq!(snippet(&text, "zebra", &[]), expected);
    }

    #[test]
    fn snippet_is_case_insensitive() {
        let text = format!("{}ZEBRA{}", "a".repeat(300), "b".repeat(300));
        let expected = format!("…{}ZEBRA{}…", "a".repeat(80), "b".repeat(155));
        assert_eq!(snippet(&text, "zebra", &[]), expected);
    }

    #[test]
    fn snippet_centres_cjk_queries_without_splitting_characters() {
        let text = format!("{}关键字{}", "中".repeat(300), "文".repeat(300));
        let expected = format!("…{}关键字{}…", "中".repeat(80), "文".repeat(157));
        assert_eq!(snippet(&text, "关键字", &[]), expected);
    }

    #[test]
    fn snippet_falls_back_to_the_earliest_term() {
        let text = format!(
            "{}alpha{}zebra{}",
            "x".repeat(300),
            "y".repeat(20),
            "z".repeat(300)
        );
        let terms = vec!["zebra".to_owned(), "alpha".to_owned()];
        let expected = format!(
            "…{}alpha{}zebra{}…",
            "x".repeat(80),
            "y".repeat(20),
            "z".repeat(130)
        );
        assert_eq!(snippet(&text, "missing", &terms), expected);
    }

    #[test]
    fn snippet_starts_at_the_text_when_nothing_matches() {
        let text = format!("{}", "q".repeat(400));
        let expected = format!("{}…", "q".repeat(240));
        assert_eq!(snippet(&text, "missing", &[]), expected);
    }
}
