//! Processor pipeline: the ordered transforms that compress a history.
//!
//! A processor receives the mutable history and a [`ProcessorContext`] holding
//! the configuration, token counter, optional model, offload store, and
//! cancellation. It returns a [`ContextEvent`](crate::events::ContextEvent)
//! when it changed something.
//!
//! The engine builds its pipeline from [`ProcessorConfig`] once, at context
//! creation, and keeps that pipeline for the context's lifetime — Python's
//! attachment-order semantics. A caller that wants full control can construct a
//! [`ContextPipeline`] directly:
//!
//! ```
//! use std::sync::Arc;
//!
//! use openjiuwen_context::{ContextPipeline, TruncationConfig, TruncationProcessor};
//!
//! let pipeline = ContextPipeline::new(vec![Arc::new(TruncationProcessor {
//!     config: TruncationConfig::default(),
//! })]);
//! assert_eq!(pipeline.processors().len(), 1);
//! ```

pub mod compressor;
pub mod offloader;
pub mod rule_compression;
pub mod summarize;
pub mod summary_offloader;
pub mod tool_result;
pub mod tool_result_window;
pub mod truncate;

mod config;
mod pipeline;
#[cfg(test)]
mod tests;

pub use self::compressor::{
    find_last_completed_round_end, group_completed_rounds, is_memory_block,
    iter_summary_merge_ranges, summarize, summarize_offline, CurrentRoundCompressor,
    CurrentRoundCompressorConfig, DialogueCompressor, DialogueCompressorConfig,
    FullCompactProcessor, FullCompactProcessorConfig, MemoryBlockKind, MicroCompactProcessor,
    MicroCompactProcessorConfig, RoundLevelCompressor, RoundLevelCompressorConfig,
    CURRENT_ROUND_MEMORY_BLOCK, DIALOGUE_MEMORY_BLOCK, FULL_COMPACT_BOUNDARY,
    FULL_COMPACT_BOUNDARY_TEXT, FULL_COMPACT_SUMMARY_INTRO, MICRO_COMPACT_CLEARED_MARKER,
    ROUND_LEVEL_MEMORY_BLOCK, TRUNCATED_FULL_CONTEXT_SCOPE,
};
pub use self::config::{ContextProcessor, ProcessorConfig, ProcessorContext, ProcessorState};
pub use self::offloader::{
    wildcard_match, MessageOffloadRole, MessageOffloader, MessageOffloaderConfig,
    ToolResultBudgetProcessor, ToolResultBudgetProcessorConfig, OMIT_STRING, WINDOWED_OUTPUT_TAG,
    WINDOW_MANAGED_METADATA_KEY,
};
pub use self::pipeline::{ContextPipeline, PipelineDeps};
pub use self::summarize::{SummarizationConfig, SummarizationProcessor, CONTEXT_SUMMARY_PREFIX};
pub use self::summary_offloader::{
    summarize_text_offline, MessageSummaryOffloader, MessageSummaryOffloaderConfig,
    OFFLOADED_INFO_TAG, SUMMARY_COMPRESSION_SYSTEM_PROMPT,
};
pub use self::tool_result::{
    ToolResultConfig, ToolResultProcessor, PERSISTED_OUTPUT_TAG, TRUNCATED_MARKER,
};
pub use self::tool_result_window::{ToolResultWindowProcessor, ToolResultWindowProcessorConfig};
pub use self::truncate::{TruncationConfig, TruncationProcessor};

pub(crate) use self::pipeline::record_model_usage;

/// Implements the checkpoint hooks of a processor whose cross-run counters live
/// in its `state` field.
///
/// Both hooks are just the [`ProcessorState`] snapshot and restore, so every
/// stateful processor spells them the same way.
macro_rules! impl_processor_state {
    ($processor:ty) => {
        fn save_state(&self) -> serde_json::Value {
            self.state.snapshot()
        }

        fn load_state(&self, state: &serde_json::Value) {
            self.state.restore(state);
        }
    };
}
pub(crate) use impl_processor_state;
