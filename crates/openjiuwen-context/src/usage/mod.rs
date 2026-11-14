//! Prompt-token and KV-cache usage accounting.
//!
//! Ports `usage/`: the serialized `context.usage` contract, the category
//! analyzer with provider reconciliation, provider `usage_metadata`
//! normalization, session-scoped KV-cache aggregation, and the shared
//! `context_tokens` sizing helpers.
//!
//! The analyzer answers "where did this request's prompt tokens go?" and turns
//! a final window into a [`ContextUsageSnapshot`]; it never decides
//! compression, truncation or routing, because it only explains the window the
//! processors already produced.
//!
//! ```
//! use std::sync::Arc;
//!
//! use openjiuwen_context::usage::context_tokens::count_context_window_tokens;
//! use openjiuwen_context::usage::{
//!     AnalysisInput, ContextCategory, ContextUsageAnalyzer, CounterIdentity, PromptSection,
//! };
//! use openjiuwen_foundation::prompt::{ApproxTokenCounter, BaseMessage};
//!
//! let counter = Arc::new(ApproxTokenCounter::with_chars_per_token(1));
//! let analyzer = ContextUsageAnalyzer::new(counter.clone(), CounterIdentity::approx())
//!     .with_model("test-model")
//!     .with_context_window_limit(Some(1_000));
//!
//! let sections = vec![PromptSection::new(
//!     ContextCategory::SystemPrompt,
//!     "you are a helpful agent",
//! )];
//! let messages = vec![BaseMessage::user("hello")];
//! let mut input = AnalysisInput::new("req-1", "session-1");
//! input.system_prompt_sections = Some(&sections);
//! input.context_messages = &messages;
//! input.provider_input_tokens = Some(40);
//! let snapshot = analyzer.analyze(&input);
//! assert_eq!(snapshot.event_type, "context.usage");
//! assert_eq!(snapshot.context_window.tokens, 40);
//! assert!(snapshot.measurement.authoritative_total);
//! ```
//!
//! Everything here is pure CPU: no network, no filesystem, and no model call.

pub mod analyzer;
pub mod analyzer_options;
pub mod context_tokens;
pub mod models;
mod parts;
pub mod provider_usage;
pub mod session_aggregator;

pub use analyzer::{
    ContextUsageAnalyzer, MeasurementCache, ATTACHMENT_CARRIER_METADATA_KEY,
    ATTACHMENT_CATEGORY_METADATA_KEY, ATTACHMENT_FRAGMENTS_METADATA_KEY,
    DEFAULT_MEASUREMENT_CACHE_CAPACITY,
};
pub use analyzer_options::{AnalysisInput, PromptSection, SnapshotOptions, WindowSignatures};
pub use models::{
    is_fallback_source, source_name, Attribution, ContextCategory, ContextPartUsage,
    ContextUsageMeasurement, ContextUsageSnapshot, ContextWindowTokenReport, ContextWindowUsage,
    CounterIdentity, KvCacheUsage, MeasurementCacheHit, MeasurementCacheInfo, RequestKVCacheUsage,
    SessionKVCacheUsage, CONTEXT_USAGE_EVENT_TYPE, CONTEXT_USAGE_SCHEMA_VERSION,
    MEASUREMENT_VERSION, MIXED, NOT_REPORTED, PHASE_POST_CALL, PHASE_POST_COMPACT, PHASE_PRE_CALL,
    SOURCE_PROVIDER_USAGE, SOURCE_PROVIDER_USAGE_PROPORTIONAL, SOURCE_PROVIDER_USAGE_RESIDUAL,
    SOURCE_PROVIDER_USAGE_UNATTRIBUTED, STATUS_FULL_HIT, STATUS_INVALID, STATUS_MISS,
    STATUS_NOT_REPORTED, STATUS_PARTIAL_HIT,
};
pub use provider_usage::request_usage_from_metadata;
pub use session_aggregator::{CacheAggregationKey, SessionKVCacheAggregator};
