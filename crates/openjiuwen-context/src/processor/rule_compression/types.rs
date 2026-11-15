//! Types shared by the router, the compressors and the pipeline.
//!
//! Ports `rule_compression/types.py`: [`ContentType`] is the routing label,
//! [`RuleContext`] is the frozen bundle of budgets, routing hints and
//! per-compressor knobs, and [`RuleCompressionResult`] is what one compressor
//! reports. Every type is `serde`-serializable because the pipeline copies
//! `details` into message metadata (`rule_compression_details`), which is
//! persisted and replayed on resume.
use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeSet;
/// The repository-wide `len // 3` character-per-token estimator.
///
/// Mirrors Python's `CHARACTERS_PER_TOKEN` in `rule_compression/pipeline.py`;
/// rule-compression budgets are character-derived proxies, never real model
/// token budgets.
pub const CHARACTERS_PER_TOKEN: usize = 3;
/// Which tier of an offload pass may rewrite an oversized tool result.
///
/// Ports Python's `OffloadStrategy` (`message_offloader.py`), which the reference
/// keeps as a module constant; here it is a serde configuration so a caller can
/// choose the trade-off:
///
/// - [`RuleCompressionStrategy::RuleThenTruncate`] (the default, Python's current
///   behaviour) compresses deterministically first and, when the result still
///   exceeds the budget, falls through to the summary/offload tier,
/// - [`RuleCompressionStrategy::RuleOnly`] keeps the rule-compressed body even
///   when it still exceeds the budget and never summarises or offloads a
///   candidate the rule tier did not accept,
/// - [`RuleCompressionStrategy::TruncateOnly`] skips rule compression entirely and
///   goes straight to the summary/offload tier.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuleCompressionStrategy {
    /// Compress, then summarise/offload whatever is still too large.
    #[default]
    RuleThenTruncate,
    /// Compress and keep the result, never summarising or offloading.
    RuleOnly,
    /// Skip rule compression; summarise and offload directly.
    TruncateOnly,
}
impl RuleCompressionStrategy {
    /// The wire spelling (`rule_then_truncate`, `rule_only`, `truncate_only`).
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::RuleThenTruncate => "rule_then_truncate",
            Self::RuleOnly => "rule_only",
            Self::TruncateOnly => "truncate_only",
        }
    }
    /// Whether the rule tier runs under this strategy.
    #[must_use]
    pub fn runs_rule_compression(self) -> bool {
        !matches!(self, Self::TruncateOnly)
    }
    /// Whether the summary/offload tier runs under this strategy.
    #[must_use]
    pub fn runs_fallback_offload(self) -> bool {
        !matches!(self, Self::RuleOnly)
    }
    /// Whether a rule-compressed body that still exceeds the budget is summarised.
    #[must_use]
    pub fn truncates_rule_preview(self) -> bool {
        matches!(self, Self::RuleThenTruncate)
    }
    /// Parses a wire spelling, rejecting anything else with [`Error::Config`].
    ///
    /// Serde already refuses an unknown variant when a configuration is
    /// deserialized, so a misspelt strategy is a load error rather than a silent
    /// default; this is the typed-error path for callers holding a string.
    pub fn parse(value: &str) -> Result<Self> {
        match value.trim() {
            "rule_then_truncate" => Ok(Self::RuleThenTruncate),
            "rule_only" => Ok(Self::RuleOnly),
            "truncate_only" => Ok(Self::TruncateOnly),
            other => Err(Error::config(format!(
                "unknown rule-compression strategy {other:?}; expected \"rule_then_truncate\", \
                 \"rule_only\" or \"truncate_only\""
            ))),
        }
    }
}
/// What kind of content a rule compressor is looking at.
///
/// Wire values keep Python's upper-case spelling because the label lands in
/// message metadata (`rule_compression_type`) that a resumed Python session may
/// read back.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ContentType {
    /// A JSON array, rendered as CSV, JSONL or a salient-row digest.
    JsonArray,
    /// A unified diff, reduced to the interesting files, hunks and lines.
    GitDiff,
    /// An HTML document, reduced to its readable text.
    Html,
    /// `path:line:content` search output, grouped per file and ranked.
    SearchResults,
    /// Build or test log output, reduced to errors, warnings and traces.
    Log,
    /// Anything the router could not classify; deduplicated plain text.
    PlainText,
    /// A source, markup, config or prose file read by path.
    SourceFile,
}
impl ContentType {
    /// Python's `ContentType.BUILD_OUTPUT`, an alias of [`ContentType::Log`].
    pub const BUILD_OUTPUT: Self = Self::Log;
    /// The wire label, matching Python's enum *value* (`"JSON_ARRAY"`, ...).
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::JsonArray => "JSON_ARRAY",
            Self::GitDiff => "GIT_DIFF",
            Self::Html => "HTML",
            Self::SearchResults => "SEARCH_RESULTS",
            Self::Log => "LOG",
            Self::PlainText => "PLAIN_TEXT",
            Self::SourceFile => "SOURCE_FILE",
        }
    }
}
impl std::fmt::Display for ContentType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}
/// How [`RuleContext`] turns a string into a token count.
///
/// Python carries a `count_tokens` callable; the Rust port carries the policy
/// instead so [`RuleContext`] stays `serde`-serializable and frozen. Every
/// compressor goes through [`count_tokens`](super::common::count_tokens), never
/// through this policy directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TokenCountPolicy {
    /// `chars / n`, truncated, with a floor of one token.
    CharactersPerToken(usize),
}
impl Default for TokenCountPolicy {
    fn default() -> Self {
        Self::CharactersPerToken(CHARACTERS_PER_TOKEN)
    }
}
/// The frozen budget, routing hints and knobs one compressor call sees.
///
/// The defaults mirror the Python dataclass field for field; the pipeline
/// overrides the budget fields per call and fills the routing hints from the
/// tool call that produced the content.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct RuleContext {
    /// Token budget the compressed candidate should fit, as a character proxy.
    pub max_tokens: usize,
    /// Tokens of head content a head/tail fallback keeps.
    pub head_tokens: usize,
    /// Tokens of tail content a head/tail fallback keeps.
    pub tail_tokens: usize,
    /// How a string is measured.
    pub count_tokens: TokenCountPolicy,
    /// Minimum relative saving a candidate must achieve to be accepted.
    pub min_savings_ratio: f64,
    /// Object-array schema density above which rows are rendered as CSV.
    pub json_csv_min_density: f64,
    /// Lower-case relevance terms from the user request and the tool call.
    pub query_terms: BTreeSet<String>,
    /// Name of the tool that produced the content, when known.
    pub tool_name: Option<String>,
    /// Single file path named by the tool call arguments, when it names one.
    pub file_path: Option<String>,
    /// Matches kept per file by the search-results compressor.
    pub search_max_matches_per_file: usize,
    /// Matches kept in total by the search-results compressor.
    pub search_max_total_matches: usize,
    /// Files kept by the search-results compressor.
    pub search_max_files: usize,
    /// Line count below which a diff is left alone.
    pub diff_min_lines: usize,
    /// Context lines kept around a retained changed line.
    pub diff_max_context_lines: usize,
    /// Changed lines kept per hunk.
    pub diff_max_changed_lines_per_hunk: usize,
    /// Changed lines kept across the whole diff.
    pub diff_max_changed_lines_total: usize,
    /// Hunks kept per file.
    pub diff_max_hunks_per_file: usize,
    /// Files kept.
    pub diff_max_files: usize,
    /// Minimum readable text an HTML extraction must keep.
    pub html_min_content_chars: usize,
    /// Line count below which a log is left alone.
    pub log_min_lines: usize,
    /// Neighbour lines pulled in around a retained log line.
    pub log_error_context_lines: usize,
    /// Error lines kept.
    pub log_max_errors: usize,
    /// Warning lines kept.
    pub log_max_warnings: usize,
    /// Stack traces kept.
    pub log_max_stack_traces: usize,
    /// Lines kept per stack trace.
    pub log_stack_trace_max_lines: usize,
    /// Lines kept in total.
    pub log_max_total_lines: usize,
}
impl Default for RuleContext {
    fn default() -> Self {
        Self {
            max_tokens: 0,
            head_tokens: 2_000,
            tail_tokens: 2_000,
            count_tokens: TokenCountPolicy::default(),
            min_savings_ratio: 0.1,
            json_csv_min_density: 0.8,
            query_terms: BTreeSet::new(),
            tool_name: None,
            file_path: None,
            search_max_matches_per_file: 5,
            search_max_total_matches: 30,
            search_max_files: 15,
            diff_min_lines: 50,
            diff_max_context_lines: 2,
            diff_max_changed_lines_per_hunk: 8,
            diff_max_changed_lines_total: 200,
            diff_max_hunks_per_file: 10,
            diff_max_files: 20,
            html_min_content_chars: 100,
            log_min_lines: 50,
            log_error_context_lines: 3,
            log_max_errors: 10,
            log_max_warnings: 5,
            log_max_stack_traces: 3,
            log_stack_trace_max_lines: 20,
            log_max_total_lines: 100,
        }
    }
}
impl RuleContext {
    /// A context with `max_tokens` and every other field at its default.
    #[must_use]
    pub fn new(max_tokens: usize) -> Self {
        Self {
            max_tokens,
            ..Self::default()
        }
    }
    /// Measures `text` with this context's [`TokenCountPolicy`].
    #[must_use]
    pub fn token_count(&self, text: &str) -> usize {
        let divisor = match self.count_tokens {
            TokenCountPolicy::CharactersPerToken(divisor) => divisor.max(1),
        };
        (text.chars().count() / divisor).max(1)
    }
}
/// What one compressor decided.
///
/// `content` is the input verbatim whenever the compressor declined; callers key
/// off `modified`, never off equality. `lossy` describes an accepted candidate:
/// `true` means information was dropped, `false` means the rewrite preserves the
/// meaning (CSV/JSONL/compact JSON). The pipeline stamps the accepted value into
/// message metadata (`rule_compression_lossy`), so a downstream reader can tell a
/// digest from a faithful rewrite without re-running the compressor.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RuleCompressionResult {
    /// The compressed text, or the input when nothing was applied.
    pub content: String,
    /// The content type the router classified the input as.
    pub content_type: ContentType,
    /// Whether `content` differs from the input.
    pub modified: bool,
    /// Whether the accepted candidate dropped information.
    pub lossy: bool,
    /// JSON-serializable diagnostics; `None` when the compressor reports none.
    pub details: Option<Value>,
}
impl RuleCompressionResult {
    /// A result that leaves `content` untouched.
    #[must_use]
    pub fn unchanged(content: &str, content_type: ContentType) -> Self {
        Self {
            content: content.to_owned(),
            content_type,
            modified: false,
            lossy: false,
            details: None,
        }
    }
    /// A result that declined, carrying the `details` a caller may still read.
    #[must_use]
    pub fn declined(content: &str, content_type: ContentType, details: Option<Value>) -> Self {
        Self {
            content: content.to_owned(),
            content_type,
            modified: false,
            lossy: false,
            details,
        }
    }
    /// An accepted candidate.
    #[must_use]
    pub fn applied(
        compressed: String,
        content_type: ContentType,
        lossy: bool,
        details: Option<Value>,
    ) -> Self {
        Self {
            content: compressed,
            content_type,
            modified: true,
            lossy,
            details,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn strategies_parse_strictly_and_round_trip_through_serde() {
        for strategy in [
            RuleCompressionStrategy::RuleThenTruncate,
            RuleCompressionStrategy::RuleOnly,
            RuleCompressionStrategy::TruncateOnly,
        ] {
            assert_eq!(
                RuleCompressionStrategy::parse(strategy.as_str()).expect("parse"),
                strategy
            );
            let json = serde_json::to_string(&strategy).expect("serde");
            assert_eq!(json, format!("\"{}\"", strategy.as_str()));
            let parsed: RuleCompressionStrategy = serde_json::from_str(&json).expect("serde");
            assert_eq!(parsed, strategy);
        }
        assert_eq!(
            RuleCompressionStrategy::default(),
            RuleCompressionStrategy::RuleThenTruncate
        );
    }
    #[test]
    fn an_unknown_strategy_is_a_typed_config_error() {
        let error = RuleCompressionStrategy::parse("rule_and_truncate").expect_err("unknown");
        assert!(matches!(error, Error::Config(_)));
        assert!(error.to_string().contains("rule_and_truncate"));
        // Serde refuses the same value instead of falling back to the default.
        assert!(serde_json::from_str::<RuleCompressionStrategy>("\"rule_and_truncate\"").is_err());
    }
    #[test]
    fn strategy_gates_match_pythons_three_helpers() {
        assert!(RuleCompressionStrategy::RuleThenTruncate.runs_rule_compression());
        assert!(RuleCompressionStrategy::RuleThenTruncate.runs_fallback_offload());
        assert!(RuleCompressionStrategy::RuleThenTruncate.truncates_rule_preview());
        assert!(RuleCompressionStrategy::RuleOnly.runs_rule_compression());
        assert!(!RuleCompressionStrategy::RuleOnly.runs_fallback_offload());
        assert!(!RuleCompressionStrategy::RuleOnly.truncates_rule_preview());
        assert!(!RuleCompressionStrategy::TruncateOnly.runs_rule_compression());
        assert!(RuleCompressionStrategy::TruncateOnly.runs_fallback_offload());
        assert!(!RuleCompressionStrategy::TruncateOnly.truncates_rule_preview());
    }
}
