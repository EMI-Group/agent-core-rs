//! Long-term, entity/graph, and lite coding memory for the Rust rewrite of openJiuwen.
//!
//! Ports the behavior of the Python package `openjiuwen.core.memory` v0.1.20
//! (`long_term_memory.py`, `config/`, `manage/`, `graph/`, `lite/`, `process/extract/`,
//! `external/`, `migration/`, `dreaming/`, `agent_teams.memory`) to idiomatic async Rust:
//! scoped memory-record CRUD, semantic + keyword recall with a documented ranking blend,
//! deterministic dedupe/merge, entity/graph memory, lite coding memory, the variable
//! namespace, the external provider SPI, migrations, dreaming, and team memory.
//!
//! - [`record`] — memory kinds, scopes, records, inputs, filters, options, results.
//! - [`config`] — `MemoryEngineConfig` and the ranking/dedupe/graph/coding knobs.
//! - [`store`] — `MemoryStores`, the key layout and prefix registry, record/vector/
//!   variable/scope-config persistence, keyset [`store::MessagePage`] history reads, the
//!   forward-only migration runner with its operation registry, backup, the scope↔user
//!   mapping, and the [`store::kv_lock`] distributed lock seam.
//! - [`recall`] — tokenization, BM25, cosine/recency scoring, dedupe, ranking.
//! - [`engine`] — the async write and recall orchestration, including the
//!   [`engine::VariableWrite`] variable-manager path that keeps a `MemoryType::Variable`
//!   record projection beside the KV entry.
//! - [`facade`] — the [`facade::LongTermMemory`] public entry point: CRUD, recall, the
//!   variable manager, message history and paging, per-scope
//!   [`facade::ScopeAnalyzerOverrideSpec`] overrides, `remember`/`consolidate`, and the
//!   subsystem constructors.
//! - [`graph`] — entity/relation extraction, persistence, neighbor recall.
//! - [`analyzer`] — rule/LLM fact extraction, conflict classification, consolidation
//!   plans, and the `StagedAnalyzer` pipeline (plan → extract → reasoning → proactive →
//!   conflict) with its per-scope overrides.
//! - [`external`] — memory-provider SPI, the engine provider, the registry, the REST/live
//!   vendor transports, and the vendor `Unsupported` handles.
//! - [`dreaming`] — the durable dream-job queue, the orchestrator with its caller-injected
//!   sweep lock, the LLM consolidation pipeline, leases, retention, and salience.
//! - [`team`] — team-scoped records, stores, hooks, toolkit, document rendering, the
//!   extractor, observability, and the unified-recall projection.
//! - [`lite`] — typed coding facts, a Markdown memory index, and the local Markdown +
//!   FTS5 + watcher + cross-process-lock coding-memory stack with its chunked embedding
//!   index, per-fact files, session indexing, sync triggers, and daily rollups.
//! - [`prompt`] — recall results as prompt text and [`openjiuwen_foundation::prompt::BaseMessage`]s.
//!
//! All I/O is async on tokio, public data types are `serde`-serializable with
//! Python-compatible JSON names, errors use [`openjiuwen_common::error::Error`],
//! and no `unsafe` is allowed.
//!
//! # Cross-cutting invariants
//!
//! - Variables are Python's own namespace: `set_variable` writes the KV entry and a
//!   `MemoryType::Variable` record, and record listing/statistics/`clear` see that
//!   projection only when a filter names the kind explicitly.
//! - Per-scope analyzer overrides are persisted beside the scope configuration
//!   (`memory_scope_overrides/{scope_id}`) and resolved on every `remember`/`consolidate`.
//! - Message history is read either as the legacy newest-N window
//!   ([`facade::LongTermMemory::get_recent_messages`]) or as keyset pages resumed from an
//!   opaque [`store::MessageCursor`].
//! - Nothing in this crate starts a background task, timer, watcher, or purge on its own
//!   except the explicitly constructed local-coding watcher and the caller-driven dream
//!   orchestrator loop.
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
pub mod analyzer;
pub mod config;
pub mod dreaming;
pub mod engine;
pub mod external;
pub mod facade;
pub mod graph;
pub mod lite;
pub mod prompt;
pub mod recall;
pub mod record;
pub mod store;
pub mod team;
pub use analyzer::*;
pub use config::*;
pub use dreaming::*;
pub use engine::*;
pub use external::*;
pub use facade::*;
pub use graph::*;
pub use lite::*;
pub use prompt::*;
pub use recall::*;
pub use record::*;
pub use store::*;
pub use team::*;
// `analyzer`, `graph` and `team::prompts` each define an `LlmExtractor`/`EXTRACTION_SYSTEM_PROMPT`
// pair, and `external` and `lite` each have a `local` module. The glob re-exports above
// would make those names ambiguous at the crate root, so the pre-existing root bindings
// are restated explicitly; the analyzer's extractor stays reachable as
// `analyzer::LlmExtractor`, `team::prompts` is not re-exported by `team`, and the
// external registry's adapters stay reachable as `external::local`.
pub use graph::{LlmExtractor, EXTRACTION_SYSTEM_PROMPT};
pub use lite::local;
// The cross-cutting surfaces every caller is expected to name directly. Each of these is
// also reachable through its glob above; the explicit binding keeps the root paths stable
// even if a module later stops glob-exporting it.
pub use engine::VariableWrite;
pub use facade::{ScopeAnalyzerOverrideSpec, ScopeOverrideHandles, VariableAssignmentReport};
pub use store::{MessageCursor, MessagePage};
// `store::kv_lock` and `lite::lock` are different seams that both deal with locks: the
// former is the distributed KV lock (`DistributedKvLock`, `KvLockGuard`, `LockOptions`),
// the latter the local coding-memory advisory file lock (`FileLock`). Neither is
// re-exported at the crate root, so the names cannot collide there.
pub use store::kv_lock::DistributedKvLock;
/// The crate name, useful in diagnostics and documentation.
pub const CRATE_NAME: &str = env!("CARGO_PKG_NAME");
/// The crate version.
pub const CRATE_VERSION: &str = env!("CARGO_PKG_VERSION");
/// Returns the crate name.
#[must_use]
pub fn crate_name() -> &'static str {
    CRATE_NAME
}
