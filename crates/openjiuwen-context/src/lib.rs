//! Context windows, token accounting, and compression/offload.
//!
//! Maps to the Python package `core.context_engine` in the openJiuwen reference
//! implementation. The crate owns the seam between the messages an agent
//! accumulates and the payload sent to a model:
//!
//! - [`ContextConfig`] — budgets, compression threshold, token counter, pipeline.
//! - [`context`] — `ModelContext`, the conversation history, `save_state`.
//! - [`engine`] — the per-session context pool, compression, and recovery.
//! - [`window`] — dialogue rounds, window assembly, tool-pair repair.
//! - [`tokens`] and [`stats`] — counting and per-role statistics.
//! - [`processor`] — truncation, tool-result trimming/offload, summarisation.
//! - [`offload`] — durable storage for content that leaves the window.
//! - [`prompts`] — the summarisation prompt templates.
//!
//! ```
//! use openjiuwen_context::{compute_stats, ContextConfig, WindowOptions, assemble_window};
//! use openjiuwen_foundation::prompt::{ApproxTokenCounter, BaseMessage};
//!
//! let config = ContextConfig::default();
//! let counter = ApproxTokenCounter::with_chars_per_token(4);
//! let history = vec![BaseMessage::user("hi"), BaseMessage::assistant("hello")];
//! let stats = compute_stats(&history, &counter);
//! assert_eq!(stats.total_dialogues, 1);
//!
//! let window = assemble_window(
//!     &history,
//!     &[BaseMessage::system("persona")],
//!     &WindowOptions::default(),
//!     &config,
//! );
//! assert_eq!(window.len(), 3);
//! ```
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

/// The crate name, as reported by `cargo`.
pub const CRATE_NAME: &str = env!("CARGO_PKG_NAME");

/// The crate version, as reported by `cargo`.
pub const CRATE_VERSION: &str = env!("CARGO_PKG_VERSION");

pub mod config;
pub mod context;
pub mod engine;
pub mod events;
pub mod offload;
pub mod processor;
pub mod prompts;
pub mod recall;
pub mod reload;
pub mod session_memory;
pub mod stats;
pub mod tokens;
pub mod window;

pub use config::{
    CompressionRecallConfig, ContextConfig, TokenCounterConfig, TokenCounterKind,
    DEFAULT_CONTEXT_TOKENS,
};
pub use context::{ContextState, ConversationContext, ModelContext};
pub use engine::{ContextEngine, ContextOptions, PrepareOptions, DEFAULT_SESSION_ID};
pub use events::{
    CompressReport, CompressionStatus, ContextEvent, ContextEventKind, ObserveOutcome,
    RecoveryOutcome,
};
pub use offload::{
    rehydrate, sanitize_key, InMemoryOffloadStore, KvOffloadStore, OffloadKind, OffloadMarker,
    OffloadRecord, OffloadScope, OffloadStore, SessionOffloadStore, OFFLOAD_MARKER_PREFIX,
};
pub use processor::{
    ContextPipeline, ContextProcessor, ProcessorConfig, ProcessorContext, SummarizationConfig,
    SummarizationProcessor, ToolResultConfig, ToolResultProcessor, TruncationConfig,
    TruncationProcessor, CONTEXT_SUMMARY_PREFIX, PERSISTED_OUTPUT_TAG, TRUNCATED_MARKER,
};
pub use prompts::{render_summary_prompt, SUMMARY_SYSTEM_PROMPT, SUMMARY_USER_TEMPLATE};
pub use recall::{
    archive_from_messages, group_turns, parse_recall_marker, recall_compressed_context,
    recall_marker, split_text, try_archive_from_messages, Bm25Index, InMemoryRecallArchiveStore,
    KvRecallArchiveStore, RecallArchive, RecallArchiveStore, RecallChunk, RecallChunkMatch,
    RecallOptions, RecallResult, RecallTurn, RecallTurnMatch, TurnRecord,
    COMPRESSION_RECALL_MARKER_PREFIX,
};
pub use reload::{
    has_offload_markers, inject_reload_hint, reload_hint, reload_system_message, ReloadLanguage,
    RELOAD_HINT_CN, RELOAD_HINT_EN,
};
pub use session_memory::{
    boundary_message, is_boundary_message, reinject_session_memory, stamp_message_id,
    summary_message, InMemorySessionMemoryStore, KvSessionMemoryStore, SessionMemoryNotes,
    SessionMemoryStore, CONTEXT_MESSAGE_ID_KEY, RECENT_MESSAGES_NOTICE, SESSION_MEMORY_BOUNDARY,
    SESSION_MEMORY_SUMMARY_INTRO,
};
pub use stats::{compute_stats, ContextStats, PreparedContext};
pub use tokens::{check_budget, count_message, count_messages, TokenBudget};
pub use window::{
    assemble_window, dialogue_round_ranges, drop_orphan_tool_results, find_last_n_rounds_start,
    strip_leading_tool_messages, truncate_keeping_rounds, validate_tool_pairs, WindowOptions,
};
