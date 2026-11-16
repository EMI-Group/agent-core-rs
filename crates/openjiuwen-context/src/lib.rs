//! Context windows, token accounting, compression/offload, and usage telemetry.
//!
//! Maps to the Python package `core.context_engine` in the openJiuwen reference
//! implementation. The crate owns the seam between the messages an agent
//! accumulates and the payload sent to a model:
//!
//! - [`ContextConfig`] — budgets, compression threshold, token counter, window policy, pipeline.
//! - [`context`] — `ModelContext`, the conversation history, `save_state`, window tracking.
//! - [`engine`] — the per-session context pool, compression, recovery, usage reports.
//! - [`window`] — dialogue rounds, window assembly, tool-pair repair.
//! - [`tokens`] and [`stats`] — counting and per-role statistics.
//! - [`processor`] — truncation, tool-result trimming/offload, summarisation, compressors,
//!   rule-based result compression, state reinjection.
//! - [`offload`] — durable storage for content that leaves the window.
//! - [`usage`] — `context.usage` snapshots, window token reports, KV-cache aggregation.
//! - [`telemetry`] — the compression-state recorder and its request-correlation scope.
//! - [`policy`] — context-window resolution (overrides, model table, `OpenRouter` cache) and
//!   prompt-cache window diffing.
//! - [`tokenizers`] — encoding selection plus offline artefact discovery, including native
//!   `tokenizer.json` and `tiktoken` `.model` loading.
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
pub mod engine_options;
pub mod events;
pub mod offload;
pub mod policy;
pub mod processor;
pub mod prompts;
pub mod recall;
pub mod reload;
pub mod session_memory;
pub mod session_memory_agent;
pub mod stats;
pub mod telemetry;
pub mod tokens;
pub mod usage;
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
pub use policy::{
    detect_context_window_change, diff_positions, global_openrouter_window_cache,
    lookup_model_default_window, tools_signature, ContextWindowChange, ContextWindowPolicy,
    ContextWindowSnapshot, ContextWindowSource, MessageFingerprint, OpenRouterWindowCache,
    PositionChange, PromptCacheImpact, ResolvedContextWindow, MODEL_DEFAULT_CONTEXT_WINDOW_TOKENS,
};
pub use processor::compressor::adaptive::{
    render_memory_block_from_plan, render_memory_block_from_plan_with_marker, AdaptiveConfig,
    AdaptiveContext, AdaptiveStrategy, AdaptiveStrategyConfig, CompressionStrategy,
    MemoryBlockPlan, StrategyDecision, StrategyInput, StrategyParse, StrategySource,
};
pub use processor::compressor::reinjection::{
    is_reinjected_state_message, strip_state_header, ContextSnapshotProvider, RenderedSection,
    SnapshotContext, SnapshotRegistry, SnapshotSection, DEFAULT_STATE_MARKER,
    DEFAULT_STATE_SNAPSHOT_MAX_CHARS, RECOVERED_CONTEXT_INSTRUCTION, RECOVERED_CONTEXT_TAG,
    REINJECTED_SECTIONS_METADATA_KEY, STATE_REINJECTION_MARKER,
};
pub use processor::rule_compression::{
    ContentRouter, ContentType, RuleCompressionPipeline, RuleCompressionResult, RuleCompressor,
    RuleContentRouter, RuleContext, TokenCountPolicy,
};
pub use processor::{
    ContextPipeline, ContextProcessor, CurrentRoundCompressor, CurrentRoundCompressorConfig,
    DialogueCompressor, DialogueCompressorConfig, FullCompactProcessor, FullCompactProcessorConfig,
    MemoryBlockKind, MessageOffloader, MessageOffloaderConfig, MessageSummaryOffloader,
    MessageSummaryOffloaderConfig, MicroCompactProcessor, MicroCompactProcessorConfig,
    PipelineDeps, ProcessorConfig, ProcessorContext, ProcessorState, RoundLevelCompressor,
    RoundLevelCompressorConfig, SummarizationConfig, SummarizationProcessor,
    ToolResultBudgetProcessor, ToolResultBudgetProcessorConfig, ToolResultConfig,
    ToolResultProcessor, ToolResultWindowProcessor, ToolResultWindowProcessorConfig,
    TruncationConfig, TruncationProcessor, CONTEXT_SUMMARY_PREFIX, CURRENT_ROUND_MEMORY_BLOCK,
    DIALOGUE_MEMORY_BLOCK, PERSISTED_OUTPUT_TAG, ROUND_LEVEL_MEMORY_BLOCK, TRUNCATED_MARKER,
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
pub use telemetry::{
    apply_operation, current_compression_operation, current_compression_operation_id,
    current_request_purpose, modified_messages, stamp_current_operation, with_operation,
    with_operation_scope, CompressionOperation, CompressionStatePhase, CompressionStateSink,
    CompressionStateStatus, ContextCompressionMetric, ContextCompressionModifiedMessage,
    ContextCompressionSaved, ContextCompressionState, ContextCompressionUsage,
    ContextProcessorStateInput, ContextProcessorStateRecorder, MeasuredTokens, RecordingSink,
    TokenProvenance, COMPACTION_PURPOSE, CONTEXT_COMPRESSION_STATE_TYPE, CONTEXT_OPERATION_ID_KEY,
    REQUEST_PURPOSE_KEY,
};
pub use tokens::{check_budget, count_message, count_messages, TokenBudget};
pub mod tokenizers;

pub use tokenizers::artifacts::{
    TokenizerArtifactManager, ARTIFACT_ERROR_CODES, LOCAL_ARTIFACT_MISSING_CODE,
    PROVIDER_OFFICIAL_ADAPTER_NOT_CONFIGURED_CODE, REMOTE_DOWNLOAD_DISABLED_CODE,
    REMOTE_URL_NOT_ALLOWED_CODE, TIKTOKEN_MODEL_MISSING_CODE, TOKENIZER_DOWNLOAD_FAILED_CODE,
    TOKENIZER_ID_MISSING_CODE, TOKENIZER_JSON_MISSING_CODE, TOKENIZER_NETWORK_UNAVAILABLE_CODE,
    TOKENIZER_NOT_IN_LOCAL_CACHE_CODE, TOKENIZER_REVISION_UNAVAILABLE_CODE,
    TOKENIZER_SHA256_MISMATCH_CODE,
};
pub use tokenizers::{
    load_native_tokenizer, CompatibleTokenizerSpec, ModelEncoding, ModelTokenCounter,
    SelectedTokenizer, TokenMeasurement, TokenizerEngine, TokenizerMatch, TokenizerMatchKind,
    TokenizerRegistry, TokenizerSelector, TokenizerSource, TokenizerSpec,
};
pub use usage::context_tokens::{
    count_context_window_tokens, count_messages_tokens, count_tools_tokens,
    count_usage_tokens_with_tail, ContextTokenCount,
};
pub use usage::models::CounterProvenance;
pub use usage::{
    is_fallback_source, request_usage_from_metadata, source_name, AnalysisInput, Attribution,
    CacheAggregationKey, ContextCategory, ContextPartUsage, ContextUsageAnalyzer,
    ContextUsageMeasurement, ContextUsageSnapshot, ContextWindowTokenReport, ContextWindowUsage,
    CounterIdentity, KvCacheUsage, MeasurementCacheInfo, PromptSection, RequestKVCacheUsage,
    SessionKVCacheAggregator, SessionKVCacheUsage, SnapshotOptions, WindowSignatures,
    CONTEXT_USAGE_EVENT_TYPE, CONTEXT_USAGE_SCHEMA_VERSION, SOURCE_PROVIDER_USAGE, STATUS_INVALID,
    STATUS_MISS, STATUS_PARTIAL_HIT,
};
pub use window::{
    assemble_window, dialogue_round_ranges, drop_orphan_tool_results, find_last_n_rounds_start,
    strip_leading_tool_messages, truncate_keeping_rounds, validate_tool_pairs, WindowOptions,
};
