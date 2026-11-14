//! The deterministic, model-free compressors, one per [`ContentType`].
//!
//! Ports `rule_compression/compressors/`: every compressor is a pure function of
//! `(content, ctx)` — no I/O, no network, no clock, no model, no shared mutable
//! state — so a result is reproducible and cheap enough to run on the offloader
//! hot path before falling back to a head/tail preview.
//!
//! Each compressor must:
//!
//! - return the input verbatim with `modified = false` when it cannot improve on
//!   it (the acceptance gate is [`meets_savings_ratio`](super::common::meets_savings_ratio)
//!   and, where Python enforces it, [`fits_budget_and_saves`](super::common::fits_budget_and_saves));
//! - never panic on malformed input (parse failures degrade to "no change");
//! - keep `[N ... omitted]` markers in the candidate so the model knows the
//!   payload was trimmed.
//!
//! [`ContentType`]: super::types::ContentType
pub mod diff;
pub mod html;
pub mod json_array;
pub mod log;
pub mod plain_text;
pub mod search_results;
pub mod source_file;
use super::types::{RuleCompressionResult, RuleContext};
pub use diff::DiffCompressor;
pub use html::{
    HTMLExtractionResult, HTMLExtractor, HTMLExtractorConfig, HtmlCompressor, HtmlOutputFormat,
};
pub use json_array::JsonArrayCompressor;
pub use log::{LogCompressor, LogLevel, LogLine};
pub use plain_text::PlainTextCompressor;
pub use search_results::{parse_search_match, SearchMatch, SearchResultsCompressor};
pub use source_file::SourceFileCompressor;
/// One deterministic compressor.
pub trait RuleCompressor {
    /// Compresses `content` under `ctx`, never panicking.
    fn compress(&self, content: &str, ctx: &RuleContext) -> RuleCompressionResult;
}
