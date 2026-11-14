//! Session-trace ingestion: the input half of the experience pipeline.
//!
//! Python's `openjiuwen.symphony.experience.trace` turns agent-runtime session exports
//! into `TraceRecord` objects and caches the parsed records next to the sessions. This
//! module ports that front end so the rest of the Rust pipeline can be fed from real
//! evidence:
//!
//! - [`TraceRecord`] — one `request_id` segment: the query, the skills it exercised, the
//!   tool calls, the final text, and whether it failed.
//! - [`TraceStore`] — the incremental cache. [`TraceStore::parse_all_sessions`] parses
//!   only sessions missing from `trace_store/processed_index.json`, appends their records
//!   to `trace_store/records.jsonl` (deduping on `trace_id`), and is idempotent; a second
//!   run with no new sessions appends nothing.
//! - [`SessionPaths`] — the path layout, so a caller can address individual files.
//!
//! The parsed records feed [`cluster_traces`](crate::experience::cluster_traces) and then
//! [`ExperienceCollector`](crate::experience::ExperienceCollector), which writes the
//! distilled items into an [`ExperienceBank`](crate::experience::ExperienceBank) or an
//! [`ExperienceIndex`](crate::experience::ExperienceIndex).
//!
//! # Parsing and tolerance
//!
//! The parser folds the flat event stream (`chat.reasoning`, `chat.delta`, `chat.final`,
//! `chat.tool_call`, `chat.tool_result`, `chat.error`) into one record per `request_id`.
//! A session whose metadata is missing is skipped, a malformed history is treated as
//! empty, and an individual unreadable session is warned about and skipped — the scan
//! never aborts. Tool payloads may be JSON or a Python-style single-quoted literal; see
//! the parser's module documentation for the exact rules.
//!
//! `trace_id` is `"{session_id}/{request_id}"`, which fixes Python's known collision where
//! two sessions reusing a request id overwrite each other in the store.
#![allow(clippy::module_name_repetitions)]

mod models;
mod parser;
mod store;

pub use models::{TraceRecord, TraceToolCall};
pub use store::{
    SessionPaths, TraceStore, PROCESSED_INDEX_FILE, RECORDS_FILE, SESSIONS_DIR, TRACE_STORE_DIR,
};
