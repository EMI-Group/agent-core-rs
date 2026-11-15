//! Plain-text compression: deduplicate lines and collapse blank runs.
//!
//! Ports `rule_compression/compressors/plain_text_compressor.py`. Every line is
//! whitespace-normalised, an already-seen line is dropped, and runs of blank
//! lines collapse to one. Unlike Python, the accepted candidate reports
//! `lossy = true` whenever that dropped a duplicate line, collapsed a blank run,
//! or rewrote a line's internal spacing — the flag is honest here: `false` means
//! the output is a faithful rewrite of the input, so a reader can rely on it when
//! a compressed body is replayed much later.
//!
//! Leading and trailing blank lines are trimmed without changing the flag on
//! their own: trimming whitespace only is the one rewrite that keeps the input's
//! meaning intact.
use super::{RuleCompressionResult, RuleCompressor};
use crate::processor::rule_compression::common::meets_savings_ratio;
use crate::processor::rule_compression::types::{ContentType, RuleContext};
use std::collections::BTreeSet;
/// Deduplicates plain-text lines.
#[derive(Debug, Clone, Copy, Default)]
pub struct PlainTextCompressor;
impl RuleCompressor for PlainTextCompressor {
    fn compress(&self, content: &str, ctx: &RuleContext) -> RuleCompressionResult {
        let mut seen: BTreeSet<String> = BTreeSet::new();
        let mut lines: Vec<String> = Vec::new();
        let mut dropped = false;
        let mut respaced = false;
        for line in content.split('\n') {
            let normalized = line.split_whitespace().collect::<Vec<_>>().join(" ");
            if normalized.is_empty() {
                match lines.last() {
                    // A leading blank line is trimmed without losing anything.
                    None => {}
                    // A blank run collapses: only the first line survives.
                    Some(last) if last.is_empty() => dropped = true,
                    Some(_) => lines.push(String::new()),
                }
            } else if seen.insert(normalized.clone()) {
                respaced = respaced || normalized != line;
                lines.push(normalized);
            } else {
                dropped = true;
            }
        }
        let compressed = lines.join("\n").trim().to_owned();
        if compressed == content || !meets_savings_ratio(content, &compressed, ctx) {
            return RuleCompressionResult::unchanged(content, ContentType::PlainText);
        }
        RuleCompressionResult::applied(
            compressed,
            ContentType::PlainText,
            dropped || respaced,
            None,
        )
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn duplicate_and_blank_lines_collapse() {
        let content = "alpha\nbeta\nalpha\n\n\n\ngamma\n";
        let result = PlainTextCompressor.compress(content, &RuleContext::new(100));
        assert!(result.modified);
        assert_eq!(result.content, "alpha\nbeta\n\ngamma");
        assert!(result.lossy);
        assert!(result.details.is_none());
    }
    #[test]
    fn unique_content_is_declined() {
        let content = "alpha\nbeta\ngamma\ndelta";
        let result = PlainTextCompressor.compress(content, &RuleContext::new(100));
        assert!(!result.modified);
        assert_eq!(result.content, content);
        assert!(!result.lossy);
    }
    #[test]
    fn a_rewrite_that_only_trims_blank_edges_stays_faithful() {
        let content = "\nalpha\nbeta\n\ngamma\ndelta\n";
        let result = PlainTextCompressor.compress(content, &RuleContext::new(100));
        assert!(result.modified);
        assert_eq!(result.content, "alpha\nbeta\n\ngamma\ndelta");
        // Trimming the leading and trailing blank lines is not a loss.
        assert!(!result.lossy);
    }
    #[test]
    fn a_dropped_duplicate_marks_the_result_lossy() {
        let content = "alpha\nbeta\ngamma\ndelta\nalpha\n";
        let result = PlainTextCompressor.compress(content, &RuleContext::new(100));
        assert!(result.modified);
        assert!(result.lossy);
    }
    #[test]
    fn collapsed_internal_spacing_counts_as_lossy() {
        let content = "alpha   beta\ngamma\tdelta\n   epsilon\nzeta\n";
        let result = PlainTextCompressor.compress(content, &RuleContext::new(100));
        assert!(result.modified);
        assert!(result.lossy);
    }
}
