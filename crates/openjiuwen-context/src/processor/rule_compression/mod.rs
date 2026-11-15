//! Deterministic, model-free compression of large tool output.
//!
//! Ports `processor/forked/offloader/rule_compression/`: content-type routing,
//! the per-type compressors (HTML, JSON array, log, diff, search results,
//! source file, plain text), query-term extraction, the compression pipeline
//! and its optional debug dump.
//!
//! A pass is pure apart from the explicit dump: [`RuleCompressionPipeline`]
//! detects the content type of one tool message through [`RuleContentRouter`],
//! runs the matching compressor, and returns the message with the compressed
//! text plus the `rule_compressed_*` metadata keys. Nothing here calls a model,
//! touches the network, or depends on the wall clock (an injected clock drives
//! timestamps and dump names).
//!
//! Acceptance is uniform: a compressor returns its input with `modified = false`
//! unless the candidate saves at least `RuleContext::min_savings_ratio` (10 %)
//! of the measured tokens and, where Python enforces it, fits `max_tokens`.
//! Character counts use Python's `len(text) // 3` token estimate, so the gates
//! reproduce the reference thresholds without a tokenizer.
//!
//! Intentional differences from Python, all documented per module:
//!
//! - matching here is hand-rolled and Unicode-aware (`NFKC` folding, Arabic-Indic
//!   digits), and the crate's `regex` dependency is for the tokenizer nodes,
//!   never this node.
//!   Log lines fold with full Unicode lowercasing, and [`is_timestamp_line`]
//!   folds `unicode-normalization` `NFKC` plus the Arabic-Indic decimal blocks,
//!   so full-width and localized timestamp spellings are recognised instead of
//!   only ASCII ones;
//! - `json_repair` is replaced by the documented repair ladder in
//!   `router::repair` (fence, span, smart quotes, single quotes, Python
//!   literals, unquoted keys, trailing commas, close truncated, drop incomplete
//!   tail, wrap sequence/bare members) and a body it cannot repair is a typed
//!   [`JsonRepairFailure`], never a silent truncation;
//! - HTML readability is the deterministic scoring ladder in
//!   `compressors::html::ladder` over the built-in renderer instead of
//!   `BeautifulSoup` plus `trafilatura`, and `source_file` keeps Python's shape
//!   but adds the structure digest Python's no-op compressor never had;
//! - the dump directory has no workspace handle, so it falls back to the
//!   documented relative [`dump::DEFAULT_DUMP_DIR`] instead of
//!   `<workspace>/context/<session>_context/rule_compression_logs`.
pub mod common;
pub mod compressors;
pub mod dump;
pub mod pipeline;
pub mod query_terms;
pub mod router;
pub mod types;
pub use common::{
    count_tokens, extract_file_path_argument, file_name, fits_budget_and_saves, is_error_text,
    is_source_file_path, meets_savings_ratio, strip_display_line_prefixes,
    strip_display_line_prefixes_preserving_body_whitespace,
};
pub use compressors::{
    parse_search_match, DiffCompressor, HTMLExtractionResult, HTMLExtractor, HTMLExtractorConfig,
    HtmlCompressor, HtmlOutputFormat, JsonArrayCompressor, LogCompressor, LogLevel, LogLine,
    PlainTextCompressor, RuleCompressor, SearchMatch, SearchResultsCompressor,
    SourceFileCompressor,
};
pub use dump::{
    append_jsonl, dump_file_name, expand_dump_dir_template, jsonl_file_name, read_dump_dir,
    read_dump_file, read_dump_text, resolve_dump_dir, safe_filename_part, safe_filename_part_or,
    utc_date_stamp, write_dump, write_record, DumpRecord, DumpTemplateVars,
    RuleCompressionDumpFormat, RuleCompressionDumpRecord, DEFAULT_DUMP_DIR,
    RULE_COMPRESSION_DUMP_DIR_ENV, UNKNOWN_CONTEXT_ID, UNKNOWN_SESSION_ID,
};
pub use pipeline::{
    RuleClock, RuleCompressionPipeline, ADD_PASS, RULE_COMPRESSED_AT_KEY,
    RULE_COMPRESSION_DETAILS_KEY, RULE_COMPRESSION_DUMP_PATH_KEY, RULE_COMPRESSION_LOSSY_KEY,
    RULE_COMPRESSION_PASS_KEY, RULE_COMPRESSION_TYPE_KEY, TTL_PASS,
};
pub use query_terms::extract_query_terms;
pub use router::{
    is_timestamp_line, repair_json, repair_json_array, repair_json_array_checked, sniff_bytes,
    sniff_content, ContentRouter, JsonRepairFailure, RuleContentRouter, SniffKind, SniffedContent,
    BINARY_SNIFF_BYTES, MAX_CONTROL_RATIO_PERCENT, REPLACEMENT_CHAR_MIN,
    REPLACEMENT_CHAR_RATIO_PERCENT,
};
pub use types::{
    ContentType, RuleCompressionResult, RuleCompressionStrategy, RuleContext, TokenCountPolicy,
    CHARACTERS_PER_TOKEN,
};
