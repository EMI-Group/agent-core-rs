//! The deterministic readability scoring ladder behind [`super::HTMLExtractor`].
//!
//! The extractor renders the document into markdown blocks; this module decides
//! which contiguous run of those blocks is readable content and which blocks are
//! page chrome (menus, sidebars, footers, cookie banners). It is a scoring ladder,
//! not a parser: no HTML crate is involved, every step is pure, and the same input
//! always yields the same selection.
//!
//! ## Measurements
//!
//! Each rendered block is measured first ([`BlockMetrics`]):
//!
//! - `visible` — characters that survive the markdown strip (`MARKDOWN_MARKERS`,
//!   the set the plain-text rendering removes) and the `](href)` targets of
//!   `[label](href)` spans,
//! - `links` — characters inside those `[label]` spans,
//! - `markup` — `block length - visible`,
//! - `digits` and `punctuation` — counts over the visible text.
//!
//! Five ratios follow:
//!
//! - `text_density = visible / (visible + markup)`,
//! - `link_density = links / visible`,
//! - `boilerplate_ratio = visible chars on a boilerplate line / visible`, where a
//!   boilerplate line is one repeated elsewhere in the document, or one of the
//!   marker phrases a footer/menu/cookie banner uses,
//! - `signal_density = 0.5 * min(digit_ratio / 0.05, 1) + 0.5 * min(punctuation_ratio / 0.10, 1)`,
//!   which is what makes a table or a code block rank above prose of equal length,
//! - `structural_bonus = 1` for a heading, a table or a fenced code block.
//!
//! ## The score
//!
//! ```text
//! score = 1.0 * ln(1 + visible) / 5     prose mass, log-scaled
//!       + 2.0 * text_density            rendered text rather than markup
//!       + 1.5 * (1 - 2 * link_density)  prose rather than a list of links; a
//!                                       link-only block pays 1.5 instead of 0
//!       + 1.0 * signal_density          tables and code carry data
//!       + 0.5 * structural_bonus        headings, tables, code fences
//!       - 5.0 * boilerplate_ratio       repeated headers, footers, menus
//!       - 1.0 * link_penalty            over 80 % links, or shorter than 120
//!                                       chars, over half links
//! ```
//!
//! The weights are ordered by how decisive each signal is, not tuned to a corpus:
//! plain prose of any length lands near 4 (2.0 density + 1.5 prose + ~0.5 mass),
//! a short link menu lands near 0.3 (0.7 density + 0 prose + 0.6 mass - 1.0
//! penalty), a related-links sidebar of under 120 characters loses the same
//! penalty even when its labels carry prose, and a footer line lands near 0.2
//! (5.0 of boilerplate outweighs its prose, digit and punctuation scores).
//! `KEEP_THRESHOLD = 1.0` therefore separates content from chrome without a
//! classifier.
//!
//! ## Selection
//!
//! 1. every block is scored ([`analyse_blocks`]),
//! 2. [`best_run`] takes the maximum-sum contiguous run of
//!    `score - KEEP_THRESHOLD`, preferring the longer run and then the earlier one
//!    on a tie, so a region has to earn its length instead of the whole page
//!    winning by default,
//! 3. inside that run, blocks scoring below `KEEP_THRESHOLD` are dropped,
//! 4. a selection that keeps fewer than [`MIN_REGION_VISIBLE_CHARS`] visible
//!    characters — or nothing at all — is replaced by the whole document and
//!    reported through [`LadderOutcome::fallback`], so text is never lost to the
//!    ladder.
use crate::processor::rule_compression::common::find_word;
use std::collections::{BTreeMap, BTreeSet};

/// Score at or above which a block counts as content.
pub(super) const KEEP_THRESHOLD: f64 = 1.0;
/// Visible characters a selected region must keep before it is accepted.
pub(super) const MIN_REGION_VISIBLE_CHARS: usize = 80;
/// Visible characters below which a block counts as short.
const SHORT_BLOCK_CHARS: usize = 120;
/// Above this link density a short block is treated as navigation.
const SHORT_LINK_DENSITY: f64 = 0.5;
/// Above this link density a block is navigation whatever its length.
const LINK_DOMINATED_DENSITY: f64 = 0.8;
// Why: the seven weights below are the ladder's whole model, so they are fixed
// constants rather than a tuned corpus — a deterministic score has to be
// explainable. Density outranks mass (2.0 against 1.0) because a block of
// markup is never content however long it is; the log in the mass term keeps a
// long article from drowning every short paragraph around it; the link term is
// signed so a link-only block pays for itself instead of scoring zero; the
// boilerplate weight is the largest because a repeated footer line is the
// single most reliable chrome signal; and the link penalty duplicates the veto
// for the two shapes that can still out-score prose — a short navigation list
// and a link list of any length.
/// Weight of the log-scaled prose mass.
const MASS_WEIGHT: f64 = 1.0;
/// Weight of the text density.
const DENSITY_WEIGHT: f64 = 2.0;
/// Weight of the prose-not-links ratio.
const LINK_WEIGHT: f64 = 1.5;
/// Weight of the digit/punctuation signal.
const SIGNAL_WEIGHT: f64 = 1.0;
/// Weight of the structural bonus.
const STRUCTURE_WEIGHT: f64 = 0.5;
/// Weight of the boilerplate ratio.
const BOILERPLATE_WEIGHT: f64 = 5.0;
/// Weight of the link penalty.
const LINK_PENALTY_WEIGHT: f64 = 1.0;
/// Digit ratio that saturates the digit half of the signal.
const DIGIT_SATURATION: f64 = 0.05;
/// Punctuation ratio that saturates the punctuation half of the signal.
const PUNCTUATION_SATURATION: f64 = 0.10;
/// Shortest line that can be boilerplate by repetition alone.
const REPEATED_LINE_MIN_CHARS: usize = 8;
/// Longest line a marker phrase can mark as boilerplate.
const MARKER_LINE_MAX_CHARS: usize = 120;
/// Words a weak chrome marker may appear in.
const WEAK_MARKER_MAX_WORDS: usize = 6;
/// Marker phrases that only appear in page chrome.
const STRONG_BOILERPLATE_MARKERS: &[&str] = &[
    "all rights reserved",
    "privacy policy",
    "terms of service",
    "terms and conditions",
    "cookie policy",
    "cookie settings",
    "accept cookies",
    "use of cookies",
    "skip to content",
    "skip to main content",
    "back to top",
    "powered by",
    "site map",
    "sitemap",
    "breadcrumb",
    "related articles",
    "related posts",
    "you may also like",
    "share this article",
    "advertisement",
    "subscribe to our newsletter",
    "sign up for our newsletter",
    "© ",
    "copyright 20",
];
/// Marker words that mark chrome only on a short line.
const WEAK_BOILERPLATE_MARKERS: &[&str] = &[
    "menu",
    "navigation",
    "comments",
    "log in",
    "sign in",
    "sign up",
    "share",
    "follow us",
    "read more",
    "toggle",
    "sidebar",
];
/// The markdown markers a rendered block carries but a reader does not see.
const MARKDOWN_MARKERS: &str = "#>*_`[]()|-";

/// What one rendered block contributes to its score.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct BlockMetrics {
    /// Characters a reader sees.
    pub visible: usize,
    /// Characters inside `[label](href)` labels.
    pub links: usize,
    /// Characters of markdown syntax and link targets.
    pub markup: usize,
    /// Digits in the visible text.
    pub digits: usize,
    /// Punctuation in the visible text.
    pub punctuation: usize,
}

/// The score of one rendered block, with the ratios it was built from.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct BlockScore {
    /// The ladder score.
    pub score: f64,
    /// `visible / (visible + markup)`.
    pub text_density: f64,
    /// `links / visible`.
    pub link_density: f64,
    /// `boilerplate chars / visible`.
    pub boilerplate_ratio: f64,
    /// Digit and punctuation saturation.
    pub signal_density: f64,
    /// `1` for a heading, table or code fence.
    pub structural_bonus: f64,
}

/// One analysed block.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct BlockAnalysis {
    /// What the block contributes.
    pub metrics: BlockMetrics,
    /// How it scored.
    pub score: BlockScore,
}

/// What the ladder kept, and why it kept everything when it did.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct LadderOutcome {
    /// The blocks to keep, in document order.
    pub blocks: Vec<String>,
    /// Blocks dropped as page chrome.
    pub dropped: usize,
    /// The score of the selected run.
    pub best_run_score: f64,
    /// Why the ladder fell back to the whole document, when it did.
    pub fallback: Option<&'static str>,
}

/// Selects the readable blocks of a rendered document.
pub(super) fn select_blocks(blocks: &[String]) -> LadderOutcome {
    if blocks.is_empty() {
        return LadderOutcome {
            blocks: Vec::new(),
            dropped: 0,
            best_run_score: 0.0,
            fallback: None,
        };
    }
    let analyses = analyse_blocks(blocks);
    let (start, end, best_run_score) = best_run(
        &analyses
            .iter()
            .map(|analysis| analysis.score.score - KEEP_THRESHOLD)
            .collect::<Vec<_>>(),
    );
    let kept: Vec<usize> = (start..end)
        .filter(|index| {
            analyses
                .get(*index)
                .is_some_and(|analysis| analysis.score.score >= KEEP_THRESHOLD)
        })
        .collect();
    let visible_kept: usize = kept
        .iter()
        .filter_map(|index| analyses.get(*index))
        .map(|analysis| analysis.metrics.visible)
        .sum();
    if kept.is_empty() {
        return LadderOutcome {
            blocks: blocks.to_vec(),
            dropped: 0,
            best_run_score,
            fallback: Some("empty_region"),
        };
    }
    if visible_kept < MIN_REGION_VISIBLE_CHARS {
        return LadderOutcome {
            blocks: blocks.to_vec(),
            dropped: 0,
            best_run_score,
            fallback: Some("region_too_small"),
        };
    }
    LadderOutcome {
        blocks: kept
            .iter()
            .filter_map(|index| blocks.get(*index).cloned())
            .collect(),
        dropped: blocks.len().saturating_sub(kept.len()),
        best_run_score,
        fallback: None,
    }
}

/// Scores every block, in order.
pub(super) fn analyse_blocks(blocks: &[String]) -> Vec<BlockAnalysis> {
    let boilerplate = boilerplate_lines(blocks);
    blocks
        .iter()
        .map(|block| {
            let metrics = measure(block);
            let score = score_block(block, &metrics, boilerplate_ratio(block, &boilerplate));
            BlockAnalysis { metrics, score }
        })
        .collect()
}

/// The maximum-sum contiguous run of `shifted`, as `start..end` plus its sum.
///
/// Ties go to the longer run, then to the earlier one, so the result is stable.
fn best_run(shifted: &[f64]) -> (usize, usize, f64) {
    let mut best = (0_usize, 0_usize, f64::NEG_INFINITY);
    for start in 0..shifted.len() {
        let mut sum = 0.0;
        for (end, value) in shifted.iter().enumerate().skip(start) {
            sum += *value;
            let length = end + 1 - start;
            let better = if sum > best.2 {
                true
            } else if sum < best.2 {
                false
            } else {
                length > best.1.saturating_sub(best.0)
            };
            if better {
                best = (start, end + 1, sum);
            }
        }
    }
    if best.1 == 0 {
        (0, 0, 0.0)
    } else {
        best
    }
}

/// Measures one rendered block.
pub(super) fn measure(block: &str) -> BlockMetrics {
    let chars: Vec<char> = block.chars().collect();
    let mut metrics = BlockMetrics::default();
    let mut index = 0;
    while index < chars.len() {
        if chars[index] == '[' {
            if let Some((label_chars, next)) = link_span(&chars, index) {
                metrics.visible += label_chars;
                metrics.links += label_chars;
                index = next;
                continue;
            }
        }
        let value = chars[index];
        if !is_markdown_marker(value) {
            metrics.visible += 1;
            if value.is_ascii_digit() {
                metrics.digits += 1;
            } else if value.is_ascii_punctuation() || is_cjk_punctuation(value) {
                metrics.punctuation += 1;
            }
        }
        index += 1;
    }
    metrics.markup = chars.len().saturating_sub(metrics.visible);
    metrics
}

/// The label length and the index after a `[label](href)` span.
fn link_span(chars: &[char], start: usize) -> Option<(usize, usize)> {
    let close = chars
        .get(start + 1..)?
        .iter()
        .position(|value| *value == ']')?
        + start
        + 1;
    if chars.get(close + 1) != Some(&'(') {
        return None;
    }
    let path = chars.get(close + 2..)?;
    let end = path.iter().position(|value| *value == ')')? + close + 2;
    let label = chars
        .get(start + 1..close)?
        .iter()
        .filter(|value| !is_markdown_marker(**value))
        .count();
    Some((label, end + 1))
}

/// Whether `value` is one of the markdown markers a rendering drops.
fn is_markdown_marker(value: char) -> bool {
    MARKDOWN_MARKERS.contains(value)
}

/// Whether `value` is full-width punctuation, which is not ASCII punctuation.
fn is_cjk_punctuation(value: char) -> bool {
    matches!(value, '\u{3000}'..='\u{303f}' | '\u{ff01}'..='\u{ff65}')
}

/// Scores one block from its measurements and boilerplate ratio.
fn score_block(block: &str, metrics: &BlockMetrics, boilerplate_ratio: f64) -> BlockScore {
    let visible = metrics.visible;
    let text_density = ratio(visible, visible + metrics.markup);
    let link_density = ratio(metrics.links, visible);
    let digit_ratio = ratio(metrics.digits, visible);
    let punctuation_ratio = ratio(metrics.punctuation, visible);
    let signal_density = 0.5 * (digit_ratio / DIGIT_SATURATION).min(1.0)
        + 0.5 * (punctuation_ratio / PUNCTUATION_SATURATION).min(1.0);
    let structural_bonus = if is_structural(block) { 1.0 } else { 0.0 };
    let link_penalty = if (visible < SHORT_BLOCK_CHARS && link_density > SHORT_LINK_DENSITY)
        || link_density > LINK_DOMINATED_DENSITY
    {
        LINK_PENALTY_WEIGHT
    } else {
        0.0
    };
    let score = MASS_WEIGHT * (1.0 + visible as f64).ln() / 5.0
        + DENSITY_WEIGHT * text_density
        + LINK_WEIGHT * (1.0 - 2.0 * link_density)
        + SIGNAL_WEIGHT * signal_density
        + STRUCTURE_WEIGHT * structural_bonus
        - BOILERPLATE_WEIGHT * boilerplate_ratio
        - link_penalty;
    BlockScore {
        score,
        text_density,
        link_density,
        boilerplate_ratio,
        signal_density,
        structural_bonus,
    }
}

/// `numerator / denominator`, or `0.0` when the denominator is zero.
fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

/// Whether a block is a heading, a table or a fenced code block.
fn is_structural(block: &str) -> bool {
    let trimmed = block.trim_start();
    if trimmed.starts_with('#') || trimmed.starts_with("```") {
        return true;
    }
    block
        .lines()
        .any(|line| line.trim_start().starts_with('|') && line.trim_end().ends_with('|'))
}

/// The normalised lines that count as boilerplate for the whole document.
fn boilerplate_lines(blocks: &[String]) -> BTreeSet<String> {
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for block in blocks {
        for line in block.lines() {
            let key = normalize_line(line);
            if !key.is_empty() {
                *counts.entry(key).or_insert(0) += 1;
            }
        }
    }
    counts
        .into_iter()
        .filter(|(key, count)| {
            (*count >= 2 && key.chars().count() >= REPEATED_LINE_MIN_CHARS) || is_marker_line(key)
        })
        .map(|(key, _)| key)
        .collect()
}

/// The share of a block's characters that sit on a boilerplate line.
fn boilerplate_ratio(block: &str, boilerplate: &BTreeSet<String>) -> f64 {
    let total = block.chars().filter(|value| !value.is_whitespace()).count();
    if total == 0 {
        return 0.0;
    }
    let flagged: usize = block
        .lines()
        .filter(|line| boilerplate.contains(&normalize_line(line)))
        .flat_map(str::chars)
        .filter(|value| !value.is_whitespace())
        .count();
    flagged as f64 / total as f64
}

/// Whether a normalised line names page chrome.
fn is_marker_line(line: &str) -> bool {
    if line.chars().count() > MARKER_LINE_MAX_CHARS {
        return false;
    }
    if STRONG_BOILERPLATE_MARKERS
        .iter()
        .any(|marker| line.contains(marker))
    {
        return true;
    }
    line.split_whitespace().count() <= WEAK_MARKER_MAX_WORDS
        && WEAK_BOILERPLATE_MARKERS
            .iter()
            .any(|marker| find_word(line, marker).is_some())
}

/// Collapses whitespace runs and lower-cases a line.
fn normalize_line(line: &str) -> String {
    line.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A page with an article, a link menu and a boilerplate footer.
    fn page() -> Vec<String> {
        vec![
            "[Home](/home) [Docs](/docs) [Blog](/blog) [About](/about)".to_owned(),
            "# Payment retries".to_owned(),
            "A failed capture is retried three times before the pipeline reports the \
             error to the caller, and every retry is logged with its request id."
                .to_owned(),
            "The retry budget is configurable per merchant, so a slow gateway does not \
             exhaust the default allowance."
                .to_owned(),
            "[Pricing](/pricing) [Careers](/careers)".to_owned(),
            "© 2026 Example Inc. All rights reserved. Privacy policy · Terms of service".to_owned(),
        ]
    }

    #[test]
    fn the_ladder_keeps_the_article_and_drops_the_chrome() {
        let blocks = page();
        let outcome = select_blocks(&blocks);
        assert_eq!(outcome.fallback, None);
        assert!(outcome.dropped >= 3, "dropped {}", outcome.dropped);
        let text = outcome.blocks.join("\n");
        assert!(text.contains("# Payment retries"));
        assert!(text.contains("A failed capture is retried three times"));
        assert!(text.contains("The retry budget is configurable"));
        assert!(!text.contains("[Home](/home)"));
        assert!(!text.contains("[Pricing](/pricing)"));
        assert!(!text.contains("All rights reserved"));
    }

    #[test]
    fn a_link_menu_scores_far_below_prose() {
        let analyses = analyse_blocks(&page());
        let menu = analyses[0].score.score;
        let article = analyses[2].score.score;
        assert!(menu < KEEP_THRESHOLD, "menu {menu}");
        assert!(article > 3.0, "article {article}");
        assert!(analyses[0].score.link_density > 0.8);
        assert!(analyses[2].score.link_density.abs() < 1e-9);
    }

    #[test]
    fn measurements_separate_text_from_markup() {
        let metrics = measure("[settled](https://example.test/pay) now");
        assert_eq!(metrics.links, 7);
        assert_eq!(metrics.visible, 11);
        assert!(metrics.markup > 0);

        let table = measure("| name | qty |\n| --- | --- |\n| a | 2 |");
        assert!(table.digits >= 1);
        // The pipes and dashes of a table are markup, not punctuation.
        assert!(table.markup >= 1);
        let prose = measure("latency: 12.5 ms, p95; ok.");
        assert!(prose.punctuation >= 4);
        assert!(is_structural("| a | b |"));
        assert!(is_structural("```\ncode\n```"));
        assert!(is_structural("# heading"));
        assert!(!is_structural("plain sentence"));
    }

    #[test]
    fn repeated_lines_are_boilerplate() {
        let blocks = vec![
            "Subscribe to the weekly newsletter".to_owned(),
            "the body of the first paragraph is long enough to matter".to_owned(),
            "Subscribe to the weekly newsletter".to_owned(),
        ];
        let boilerplate = boilerplate_lines(&blocks);
        assert!(boilerplate.contains("subscribe to the weekly newsletter"));
        assert!(!boilerplate.contains("the body of the first paragraph is long enough to matter"));
    }

    #[test]
    fn a_region_that_keeps_too_little_falls_back() {
        let blocks = vec!["tiny".to_owned(), "[a](/a) [b](/b)".to_owned()];
        let outcome = select_blocks(&blocks);
        assert_eq!(outcome.fallback, Some("region_too_small"));
        assert_eq!(outcome.blocks.len(), blocks.len());
        assert_eq!(outcome.dropped, 0);
    }

    #[test]
    fn the_best_run_prefers_the_longer_and_earlier_tie() {
        assert_eq!(best_run(&[1.0, 1.0]), (0, 2, 2.0));
        assert_eq!(best_run(&[-1.0, 2.0, -1.0]), (1, 2, 2.0));
        assert_eq!(best_run(&[0.0, 0.0]), (0, 2, 0.0));
    }
}
