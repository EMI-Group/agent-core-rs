//! Plain-text compression: deduplicate lines and collapse blank runs.
//!
//! Ports `rule_compression/compressors/plain_text_compressor.py`. Every line is
//! whitespace-normalised, an already-seen line is dropped, and runs of blank
//! lines collapse to one. Like Python, the accepted candidate reports
//! `lossy = false` even though duplicate and blank lines were dropped; treat
//! plain-text output as lossy regardless of the flag.
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
        for line in content.split('\n') {
            let normalized = line.split_whitespace().collect::<Vec<_>>().join(" ");
            if normalized.is_empty() {
                if lines.last().is_some_and(|last| !last.is_empty()) {
                    lines.push(String::new());
                }
            } else if seen.insert(normalized.clone()) {
                lines.push(normalized);
            }
        }
        let compressed = lines.join("\n").trim().to_owned();
        if compressed == content || !meets_savings_ratio(content, &compressed, ctx) {
            return RuleCompressionResult::unchanged(content, ContentType::PlainText);
        }
        RuleCompressionResult::applied(compressed, ContentType::PlainText, false, None)
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
        assert!(!result.lossy);
        assert!(result.details.is_none());
    }
    #[test]
    fn unique_content_is_declined() {
        let content = "alpha\nbeta\ngamma\ndelta";
        let result = PlainTextCompressor.compress(content, &RuleContext::new(100));
        assert!(!result.modified);
        assert_eq!(result.content, content);
    }
}
