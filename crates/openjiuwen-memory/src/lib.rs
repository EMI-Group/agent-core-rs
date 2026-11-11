//! Long-term, entity/graph, and lite coding memory for the Rust rewrite of openJiuwen.
//!
//! Ports the core behavior of the Python package `openjiuwen.core.memory` v0.1.20
//! (`long_term_memory.py`, `config/`, `manage/`, `graph/`, `lite/`) to idiomatic
//! async Rust: scoped memory-record CRUD, semantic + keyword recall with a
//! documented ranking blend, deterministic dedupe/merge, basic entity/graph
//! memory, and lite coding facts.
//!
//! - [`record`] — memory kinds, scopes, records, inputs, filters, options, results.
//! - [`config`] — `MemoryEngineConfig` and the ranking/dedupe/graph/coding knobs.
//! - [`store`] — `MemoryStores` and record/vector/variable/message persistence.
//! - [`recall`] — tokenization, BM25, cosine/recency scoring, dedupe, ranking.
//! - [`engine`] — the async write and recall orchestration.
//! - [`facade`] — the [`facade::LongTermMemory`] public entry point.
//! - [`graph`] — entity/relation extraction, persistence, neighbor recall.
//! - [`lite`] — typed coding facts and a Markdown memory index.
//! - [`prompt`] — recall results as prompt text and [`openjiuwen_foundation::prompt::BaseMessage`]s.
//!
//! All I/O is async on tokio, public data types are `serde`-serializable with
//! Python-compatible JSON names, errors use [`openjiuwen_common::error::Error`],
//! and no `unsafe` is allowed.
#![forbid(unsafe_code)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]
#![allow(
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::module_name_repetitions,
    clippy::must_use_candidate,
    clippy::too_many_arguments,
    clippy::struct_excessive_bools,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::items_after_statements,
    clippy::unused_async,
    clippy::significant_drop_tightening,
    clippy::manual_let_else,
    clippy::needless_pass_by_value
)]
pub mod config;
pub mod engine;
pub mod facade;
pub mod graph;
pub mod lite;
pub mod prompt;
pub mod recall;
pub mod record;
pub mod store;
pub use config::*;
pub use engine::*;
pub use facade::*;
pub use graph::*;
pub use lite::*;
pub use prompt::*;
pub use recall::*;
pub use record::*;
pub use store::*;
/// The crate name, useful in diagnostics and documentation.
pub const CRATE_NAME: &str = env!("CARGO_PKG_NAME");
/// The crate version.
pub const CRATE_VERSION: &str = env!("CARGO_PKG_VERSION");
/// Returns the crate name.
pub fn crate_name() -> &'static str {
    CRATE_NAME
}
