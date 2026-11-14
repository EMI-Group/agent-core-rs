//! HTML compression through a dependency-free readability extractor.
//!
//! Ports `rule_compression/compressors/html_compressor.py` without `BeautifulSoup`
//! or `trafilatura`: [`HTMLExtractor`] drops `<script>`, `<style>`, `<noscript>`,
//! `<template>`, `<svg>`, `<nav>`, `<footer>`, `<aside>` and `<form>` subtrees,
//! decodes the common entities, keeps the title, headings, links, code blocks,
//! lists, tables and paragraph text, and collapses whitespace into markdown (or
//! plain text).
//!
//! [`HtmlCompressor`] accepts a candidate only when its text is at least
//! `html_min_content_chars` long and [`meets_savings_ratio`] passes. Report-shaped
//! HTML (`pytest-html`, `results-table`, or two tables plus two report markers)
//! prunes the "passed" table rows once a failure row exists, exactly like the
//! Python `BeautifulSoup` fallback.
//!
//! [`meets_savings_ratio`]: crate::processor::rule_compression::common::meets_savings_ratio
mod extract;

#[cfg(test)]
mod tests;

use super::{RuleCompressionResult, RuleCompressor};
use crate::processor::rule_compression::common::meets_savings_ratio;
use crate::processor::rule_compression::types::{ContentType, RuleContext};
use serde_json::json;

use extract::looks_like_report_html;
pub use extract::{HTMLExtractionResult, HTMLExtractor, HTMLExtractorConfig, HtmlOutputFormat};

/// Compresses HTML through [`HTMLExtractor`].
#[derive(Debug, Clone, Copy, Default)]
pub struct HtmlCompressor {
    /// The extractor to use.
    pub extractor: HTMLExtractor,
}
impl HtmlCompressor {
    /// A compressor over `extractor`.
    #[must_use]
    pub fn new(extractor: HTMLExtractor) -> Self {
        Self { extractor }
    }
}
impl RuleCompressor for HtmlCompressor {
    fn compress(&self, content: &str, ctx: &RuleContext) -> RuleCompressionResult {
        let report = looks_like_report_html(content);
        let extracted = self.extractor.extract_with_report(content, report);
        let candidate = extracted.extracted.trim().to_owned();
        let extractor_label = if report { "builtin:report" } else { "builtin" };
        let details = json!({
            "extractor": extractor_label,
            "original_length": extracted.original_length,
            "extracted_length": extracted.extracted_length,
            "compression_ratio": extracted.compression_ratio,
            "reduction_percent": extracted.reduction_percent(),
            "title": extracted.title,
            "author": extracted.author,
            "date": extracted.date,
            "metadata": extracted.metadata,
            "removed_node_count": extracted.removed_node_count,
            "report_rows_removed": extracted.report_rows_removed,
            "duplicate_block_count": extracted.duplicate_block_count,
            "link_count": extracted.link_count,
            "table_count": extracted.table_count,
            "code_block_count": extracted.code_block_count,
            "output_block_count": extracted.output_block_count,
        });
        // `_plain_text` only replaces characters one-for-one, so its length is the
        // candidate's length; the check is the candidate size against the floor.
        if candidate.chars().count() < ctx.html_min_content_chars {
            return RuleCompressionResult::declined(content, ContentType::Html, Some(details));
        }
        if candidate != content && meets_savings_ratio(content, &candidate, ctx) {
            return RuleCompressionResult::applied(
                candidate,
                ContentType::Html,
                true,
                Some(details),
            );
        }
        RuleCompressionResult::declined(content, ContentType::Html, Some(details))
    }
}
