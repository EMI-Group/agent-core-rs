//! Search-result compression: group `path:line:content` hits per file and rank.
//!
//! Ports `rule_compression/compressors/search_results_compressor.py`. Each line is
//! parsed by [`parse_search_match`] (the first `path:line:` — or `path-line-` —
//! marker wins, and a Windows drive prefix is skipped). Matches are grouped per
//! path in first-seen order and ranked by summed score:
//!
//! - `query_terms` hits in the content (0.25 each, capped at 0.60) and in the path
//!   (0.15 each, capped at 0.30),
//! - an error signal (0.50), else a warning (0.35), else an importance word
//!   (0.25).
//!
//! The file limit is `search_max_files`; the total match limit is adaptive
//! between `min(5, total)` and `search_max_total_matches` using content diversity
//! and high-signal counts; the per-file limit is `search_max_matches_per_file`,
//! keeping the first, the last and the top-scored matches. Non-match lines are
//! preserved but hoisted directly after the header, omission markers close each
//! omitted group, and `details` are reported for unchanged results too.
use super::{RuleCompressionResult, RuleCompressor};
use crate::processor::rule_compression::common::{
    contains_word, meets_savings_ratio, strip_display_line_prefixes,
};
use crate::processor::rule_compression::types::{ContentType, RuleContext};
use serde_json::json;
use std::collections::BTreeSet;
/// Words that raise a match's score to "error" level.
const ERROR_SIGNAL_WORDS: &[&str] = &[
    "error",
    "failed",
    "failure",
    "traceback",
    "exception",
    "fatal",
    "panic",
];
/// Words that raise a match's score to "warning" level.
const WARNING_SIGNAL_WORDS: &[&str] = &["warn", "warning", "deprecated"];
/// Words that raise a match's score to "important" level.
const IMPORTANCE_SIGNAL_WORDS: &[&str] = &["todo", "fixme", "important", "critical"];
/// Score contribution of one query-term hit in the matched content.
const QUERY_CONTENT_HIT_SCORE: f64 = 0.25;
/// Cap of the content query-term contribution.
const QUERY_CONTENT_SCORE_CAP: f64 = 0.60;
/// Score contribution of one query-term hit in the path.
const QUERY_PATH_HIT_SCORE: f64 = 0.15;
/// Cap of the path query-term contribution.
const QUERY_PATH_SCORE_CAP: f64 = 0.30;
/// Score added by an error signal.
const ERROR_SIGNAL_SCORE: f64 = 0.50;
/// Score added by a warning signal.
const WARNING_SIGNAL_SCORE: f64 = 0.35;
/// Score added by an importance signal.
const IMPORTANCE_SIGNAL_SCORE: f64 = 0.25;
/// Hard ceiling of one match's score.
const MAX_SEARCH_MATCH_SCORE: f64 = 1.0;
/// Score above which a match counts as high-signal.
const HIGH_SIGNAL_SCORE_THRESHOLD: f64 = 0.5;
/// One parsed `path:line:content` hit.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchMatch {
    /// The path before the marker.
    pub file_path: String,
    /// The line number after the marker.
    pub line_number: u64,
    /// The text after the marker.
    pub content: String,
    /// The whole normalised line, echoed into the output.
    pub original: String,
    /// Zero-based position in the input.
    pub position: usize,
    /// Relevance score computed by the compressor.
    pub score: f64,
}
/// Parses one search-output line into a [`SearchMatch`].
///
/// Tool display prefixes are stripped first, a Windows drive prefix (`C:\`, `C:/`)
/// is skipped, and the first `([:-])(\d+)([:-])` marker wins. A line without a
/// non-empty path before such a marker is not a match.
#[must_use]
pub fn parse_search_match(line: &str, position: usize) -> Option<SearchMatch> {
    let normalized = strip_display_line_prefixes(line);
    let drive_prefix_end = if is_drive_prefix(&normalized) { 2 } else { 0 };
    let bytes = normalized.as_bytes();
    let mut index = drive_prefix_end;
    while index < bytes.len() {
        let marker = bytes[index];
        if marker == b':' || marker == b'-' {
            let digits_start = index + 1;
            let mut digits_end = digits_start;
            while digits_end < bytes.len() && bytes[digits_end].is_ascii_digit() {
                digits_end += 1;
            }
            if digits_end > digits_start {
                let tail = bytes.get(digits_end).copied();
                if matches!(tail, Some(b':' | b'-')) {
                    let file_path = &normalized[..index];
                    if !file_path.is_empty() {
                        let line_number = normalized[digits_start..digits_end]
                            .parse::<u64>()
                            .unwrap_or(u64::MAX);
                        return Some(SearchMatch {
                            file_path: file_path.to_owned(),
                            line_number,
                            content: normalized[digits_end + 1..].to_owned(),
                            original: normalized.clone(),
                            position,
                            score: 0.0,
                        });
                    }
                }
                index = digits_end;
                continue;
            }
        }
        index += 1;
    }
    None
}
/// Whether `line` starts with a Windows drive prefix.
fn is_drive_prefix(line: &str) -> bool {
    let bytes = line.as_bytes();
    bytes.len() >= 3
        && bytes[0].is_ascii_alphabetic()
        && bytes[1] == b':'
        && matches!(bytes[2], b'\\' | b'/')
}
/// Compresses `path:line:content` search output.
#[derive(Debug, Clone, Copy, Default)]
pub struct SearchResultsCompressor;
impl RuleCompressor for SearchResultsCompressor {
    fn compress(&self, content: &str, ctx: &RuleContext) -> RuleCompressionResult {
        let (compressed, lossy, details) = run(content, ctx);
        if compressed == content {
            return RuleCompressionResult::declined(content, ContentType::SearchResults, details);
        }
        RuleCompressionResult::applied(compressed, ContentType::SearchResults, lossy, details)
    }
}
/// Runs the grouping, scoring and selection, returning the candidate.
fn run(content: &str, ctx: &RuleContext) -> (String, bool, Option<serde_json::Value>) {
    let normalized_content = strip_display_line_prefixes(content);
    let mut grouped: Vec<(String, Vec<SearchMatch>)> = Vec::new();
    let mut unparsed_lines: Vec<String> = Vec::new();
    for (position, line) in normalized_content.lines().enumerate() {
        match parse_search_match(line, position) {
            Some(found) => {
                if let Some((_, matches)) = grouped
                    .iter_mut()
                    .find(|(file_path, _)| *file_path == found.file_path)
                {
                    matches.push(found);
                } else {
                    grouped.push((found.file_path.clone(), vec![found]));
                }
            }
            None => {
                if !line.trim().is_empty() {
                    unparsed_lines.push(line.to_owned());
                }
            }
        }
    }
    if grouped.is_empty() {
        return (content.to_owned(), false, None);
    }

    let scored: Vec<(String, Vec<SearchMatch>)> = grouped
        .into_iter()
        .map(|(file_path, matches)| {
            let matches = matches
                .into_iter()
                .map(|matched| score_match(matched, ctx))
                .collect();
            (file_path, matches)
        })
        .collect();
    let original_match_count: usize = scored.iter().map(|(_, matches)| matches.len()).sum();
    let mut ranked_files: Vec<usize> = (0..scored.len()).collect();
    ranked_files.sort_by(|left, right| {
        let left_score: f64 = scored[*left].1.iter().map(|matched| matched.score).sum();
        let right_score: f64 = scored[*right].1.iter().map(|matched| matched.score).sum();
        right_score.total_cmp(&left_score).then(left.cmp(right))
    });
    let selected_files: Vec<usize> = ranked_files
        .into_iter()
        .take(ctx.search_max_files)
        .collect();
    let adaptive_limit = adaptive_limit(&scored, ctx.search_max_total_matches);

    let mut lines: Vec<String> = vec!["[SEARCH_RESULTS compressed]".to_owned()];
    lines.extend(unparsed_lines.iter().cloned());
    let mut selected_count = 0_usize;
    let mut processed_files = 0_usize;
    let mut omitted_any = selected_files.len() < scored.len();
    for file_index in &selected_files {
        let remaining = adaptive_limit.saturating_sub(selected_count);
        if remaining == 0 {
            omitted_any = true;
            break;
        }
        processed_files += 1;
        let (file_path, matches) = &scored[*file_index];
        let per_file_limit = ctx.search_max_matches_per_file.min(remaining);
        let kept = select_matches(matches, per_file_limit);
        lines.extend(kept.iter().map(|matched| matched.original.clone()));
        selected_count += kept.len();
        let omitted = matches.len().saturating_sub(kept.len());
        if omitted > 0 {
            omitted_any = true;
            lines.push(format!("[... and {omitted} more matches in {file_path}]"));
        }
    }
    let omitted_files = scored.len().saturating_sub(processed_files);
    if omitted_files > 0 {
        omitted_any = true;
        lines.push(format!("[... and {omitted_files} more files omitted]"));
    }
    let details = json!({
        "original_match_count": original_match_count,
        "retained_match_count": selected_count,
        "omitted_match_count": original_match_count.saturating_sub(selected_count),
        "files_affected": scored.len(),
        "files_retained": processed_files,
        "unparsed_line_count": unparsed_lines.len(),
        "adaptive_match_limit": adaptive_limit,
    });
    let candidate = lines.join("\n");
    if omitted_any && meets_savings_ratio(content, &candidate, ctx) {
        (candidate, true, Some(details))
    } else {
        (content.to_owned(), false, Some(details))
    }
}
/// The adaptive total match limit.
fn adaptive_limit(scored: &[(String, Vec<SearchMatch>)], hard_limit: usize) -> usize {
    let matches: Vec<&SearchMatch> = scored
        .iter()
        .flat_map(|(_, matches)| matches.iter())
        .collect();
    let total = matches.len();
    if total == 0 {
        return 0;
    }
    let normalized: BTreeSet<String> = matches
        .iter()
        .map(|matched| normalize_content(&matched.content))
        .collect();
    let diversity_ratio = normalized.len() as f64 / total as f64;
    let base_limit = total.min(5);
    let diversity_slots =
        round_half_even(total.min(hard_limit).saturating_sub(base_limit) as f64 * diversity_ratio);
    let high_signal_count = matches
        .iter()
        .filter(|matched| matched.score >= HIGH_SIGNAL_SCORE_THRESHOLD)
        .count();
    let signal_floor = (base_limit + high_signal_count).min(hard_limit).min(total);
    (base_limit + diversity_slots)
        .max(signal_floor)
        .min(hard_limit)
        .min(total)
}
/// Python's `round`: half-way values go to the even neighbour.
fn round_half_even(value: f64) -> usize {
    let floor = value.floor();
    let fraction = value - floor;
    let rounded = if (fraction - 0.5).abs() < f64::EPSILON {
        if (floor as i64) % 2 == 0 {
            floor
        } else {
            floor + 1.0
        }
    } else {
        value.round()
    };
    rounded.max(0.0) as usize
}
/// `\b\d+\b` replaced by `<n>`, lower-cased, whitespace-collapsed.
fn normalize_content(content: &str) -> String {
    let lowered = content.to_lowercase();
    let mut normalized = String::with_capacity(lowered.len());
    let mut chars = lowered.char_indices().peekable();
    while let Some((index, value)) = chars.next() {
        if !value.is_ascii_digit() {
            normalized.push(value);
            continue;
        }
        let before_ok = lowered[..index]
            .chars()
            .next_back()
            .is_none_or(|previous| !(previous.is_alphanumeric() || previous == '_'));
        let mut end = index + value.len_utf8();
        while let Some((next_index, next)) = chars.peek() {
            if next.is_ascii_digit() {
                end = *next_index + next.len_utf8();
                chars.next();
            } else {
                break;
            }
        }
        let after_ok = lowered[end..]
            .chars()
            .next()
            .is_none_or(|next| !(next.is_alphanumeric() || next == '_'));
        if before_ok && after_ok {
            normalized.push_str("<n>");
        } else {
            normalized.push_str(&lowered[index..end]);
        }
    }
    normalized.split_whitespace().collect::<Vec<_>>().join(" ")
}
/// Scores one match and returns the scored copy.
fn score_match(matched: SearchMatch, ctx: &RuleContext) -> SearchMatch {
    let lowered_content = matched.content.to_lowercase();
    let lowered_path = matched.file_path.to_lowercase();
    let content_hits = ctx
        .query_terms
        .iter()
        .filter(|term| lowered_content.contains(&term.to_lowercase()))
        .count();
    let path_hits = ctx
        .query_terms
        .iter()
        .filter(|term| lowered_path.contains(&term.to_lowercase()))
        .count();

    let mut score = (content_hits as f64 * QUERY_CONTENT_HIT_SCORE).min(QUERY_CONTENT_SCORE_CAP);
    score += (path_hits as f64 * QUERY_PATH_HIT_SCORE).min(QUERY_PATH_SCORE_CAP);
    if contains_word(&lowered_content, ERROR_SIGNAL_WORDS) {
        score += ERROR_SIGNAL_SCORE;
    } else if contains_word(&lowered_content, WARNING_SIGNAL_WORDS) {
        score += WARNING_SIGNAL_SCORE;
    } else if contains_word(&lowered_content, IMPORTANCE_SIGNAL_WORDS) {
        score += IMPORTANCE_SIGNAL_SCORE;
    }
    SearchMatch {
        score: score.min(MAX_SEARCH_MATCH_SCORE),
        ..matched
    }
}
/// The first, last and top-scored matches, limited to `limit`.
fn select_matches(matches: &[SearchMatch], limit: usize) -> Vec<SearchMatch> {
    if limit == 0 {
        return Vec::new();
    }
    if matches.len() <= limit {
        let mut all = matches.to_vec();
        all.sort_by_key(|matched| matched.position);
        return all;
    }
    let mut chosen: BTreeSet<usize> = BTreeSet::new();
    for matched in [matches.first(), matches.last()].into_iter().flatten() {
        if chosen.len() < limit {
            chosen.insert(matched.position);
        }
    }
    let mut ranked = matches.to_vec();
    ranked.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then(left.position.cmp(&right.position))
    });
    for matched in ranked {
        if chosen.len() >= limit {
            break;
        }
        chosen.insert(matched.position);
    }
    matches
        .iter()
        .filter(|matched| chosen.contains(&matched.position))
        .cloned()
        .collect()
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt::Write as _;
    #[test]
    fn markers_parse_path_line_and_content() {
        let matched = parse_search_match("src/main.rs:42:    let value = 1;", 0).expect("match");
        assert_eq!(matched.file_path, "src/main.rs");
        assert_eq!(matched.line_number, 42);
        assert_eq!(matched.content, "    let value = 1;");
        assert!(parse_search_match("no marker here", 0).is_none());
        assert!(parse_search_match(":12:no path", 0).is_none());
        let windows = parse_search_match("C:\\src\\a.rs:7:body", 0).expect("match");
        assert_eq!(windows.file_path, "C:\\src\\a.rs");
        assert_eq!(windows.line_number, 7);
    }
    #[test]
    fn grouped_matches_are_capped_and_labelled() {
        let mut content = String::new();
        for index in 0..40 {
            let _ = writeln!(
                content,
                "src/file{}.rs:{}: hit {} error",
                index % 4,
                index,
                index
            );
        }
        let ctx = RuleContext::new(10_000);
        let result = SearchResultsCompressor.compress(&content, &ctx);
        assert!(result.modified);
        assert!(result.content.starts_with("[SEARCH_RESULTS compressed]"));
        assert!(result.content.contains("more matches in src/file0.rs"));
        let details = result.details.expect("details");
        assert_eq!(details["original_match_count"], 40);
        assert_eq!(details["files_affected"], 4);
        assert_eq!(details["unparsed_line_count"], 0);
    }
    #[test]
    fn non_search_output_is_declined() {
        let content = "one\ntwo\nthree";
        let result = SearchResultsCompressor.compress(content, &RuleContext::new(100));
        assert!(!result.modified);
        assert!(result.details.is_none());
    }
}
