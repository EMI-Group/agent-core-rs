//! The processor extension surface: the trait a transform implements, the
//! context a pass receives, the state bag it checkpoints, and the serializable
//! configuration the engine builds a pipeline from.
//!
//! This is what a processor author touches. [`ContextProcessor`] is the trait
//! itself; [`ProcessorContext`] carries the engine services a pass runs against
//! (token counter, model, offload store, recall archive, session-memory notes,
//! cancellation); [`ProcessorState`] is the JSON bag a processor keeps its
//! cross-run counters in; and [`ProcessorConfig`] is the serializable
//! description of one processor, which the engine turns into a pipeline once at
//! context creation. Execution itself lives in `super::pipeline`.

use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_common::error::Result;
use openjiuwen_foundation::llm::Model;
use openjiuwen_foundation::prompt::{BaseMessage, TokenCounter};
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

use super::compressor::{
    CurrentRoundCompressorConfig, DialogueCompressorConfig, FullCompactProcessorConfig,
    MicroCompactProcessorConfig, RoundLevelCompressorConfig,
};
use super::offloader::{MessageOffloaderConfig, ToolResultBudgetProcessorConfig};
use super::summarize::SummarizationConfig;
use super::summary_offloader::MessageSummaryOffloaderConfig;
use super::tool_result::ToolResultConfig;
use super::tool_result_window::ToolResultWindowProcessorConfig;
use super::truncate::TruncationConfig;
use crate::config::ContextConfig;
use crate::events::ContextEvent;
use crate::offload::{OffloadScope, OffloadStore};
use crate::recall::RecallArchiveStore;
use crate::session_memory::SessionMemoryStore;

/// A small, thread-safe JSON bag a processor keeps its cross-run counters in.
///
/// The bag is the persisted side of [`ContextProcessor::save_state`]: a
/// processor counts what it did while it ran, and
/// [`ContextPipeline::save_state`](crate::processor::ContextPipeline::save_state)
/// writes the snapshot into the session checkpoint through
/// [`ContextState`](crate::ContextState), so the counters survive a restart.
/// Counters are observability only — they never influence a transform, and a
/// processor that changed nothing does not advance them.
///
/// Values are read back through the typed accessors; a corrupt or foreign entry
/// restored with [`ProcessorState::restore`] is ignored rather than trusted.
///
/// ```
/// use openjiuwen_context::ProcessorState;
///
/// let state = ProcessorState::default();
/// assert_eq!(state.snapshot(), serde_json::Value::Null);
/// state.bump("compressions");
/// state.set_u64("blocks", 4);
/// assert_eq!(state.counter("compressions"), 1);
/// assert_eq!(state.counter("blocks"), 4);
/// assert_eq!(state.counter("missing"), 0);
/// ```
#[derive(Debug, Default)]
pub struct ProcessorState {
    /// The counters; `Value::Null` until the first write.
    inner: std::sync::Mutex<serde_json::Value>,
}

impl ProcessorState {
    /// Creates an empty state bag.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// A snapshot of the counters, [`serde_json::Value::Null`] when nothing was
    /// written yet.
    #[must_use]
    pub fn snapshot(&self) -> serde_json::Value {
        lock_state(&self.inner).clone()
    }

    /// Replaces the counters with a snapshot taken by [`ProcessorState::snapshot`].
    ///
    /// A `Null` or non-object value is ignored, so corrupt or foreign entries
    /// leave the current counters intact.
    pub fn restore(&self, state: &serde_json::Value) {
        if !state.is_object() {
            return;
        }
        *lock_state(&self.inner) = state.clone();
    }

    /// Increments the integer counter `key`, creating it when absent.
    pub fn bump(&self, key: &str) {
        self.add(key, 1);
    }

    /// The value of `key`, `0` when it is absent or not a non-negative integer.
    #[must_use]
    pub fn counter(&self, key: &str) -> u64 {
        lock_state(&self.inner)
            .get(key)
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0)
    }

    /// Sets the integer counter `key` to `value`.
    pub fn set_u64(&self, key: &str, value: u64) {
        with_object(&self.inner, |object| {
            object.insert(key.to_owned(), serde_json::Value::from(value));
        });
    }

    /// Records a token count under `key`.
    ///
    /// Counts are u64 counters too, so a size that does not fit saturates
    /// instead of wrapping.
    pub fn record_tokens(&self, key: &str, value: usize) {
        self.set_u64(key, u64::try_from(value).unwrap_or(u64::MAX));
    }

    /// Adds `delta` to the integer counter `key`, creating it when absent.
    pub(crate) fn add(&self, key: &str, delta: u64) {
        with_object(&self.inner, |object| {
            let current = object
                .get(key)
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
            object.insert(
                key.to_owned(),
                serde_json::Value::from(current.saturating_add(delta)),
            );
        });
    }
}

impl Clone for ProcessorState {
    /// Clones the snapshot, so the copy and the original advance independently.
    fn clone(&self) -> Self {
        Self {
            inner: std::sync::Mutex::new(self.snapshot()),
        }
    }
}

/// Locks the bag, recovering from a poisoned mutex.
fn lock_state(
    state: &std::sync::Mutex<serde_json::Value>,
) -> std::sync::MutexGuard<'_, serde_json::Value> {
    state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Applies `f` to the object of the bag, which is created when it is not an
/// object yet.
fn with_object(
    state: &std::sync::Mutex<serde_json::Value>,
    f: impl FnOnce(&mut serde_json::Map<String, serde_json::Value>),
) {
    let mut guard = lock_state(state);
    if !guard.is_object() {
        *guard = serde_json::Value::Object(serde_json::Map::new());
    }
    if let Some(object) = guard.as_object_mut() {
        f(object);
    }
}

/// One transform applied to a conversation history.
#[async_trait]
pub trait ContextProcessor: Send + Sync {
    /// Stable processor name, reported in events.
    fn name(&self) -> &'static str;

    /// Applies the transform in place and reports what changed.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Cancelled`](openjiuwen_common::error::Error::Cancelled)
    /// when the caller cancelled, and the processor's own failure (for example
    /// [`Error::Llm`](openjiuwen_common::error::Error::Llm) for a summarisation
    /// transport error) otherwise.
    async fn process(
        &self,
        messages: &mut Vec<BaseMessage>,
        cx: &ProcessorContext<'_>,
    ) -> Result<Option<ContextEvent>>;

    /// Serializes any state the processor keeps across runs.
    ///
    /// The default is [`serde_json::Value::Null`], which
    /// [`ContextPipeline::save_state`](crate::processor::ContextPipeline::save_state)
    /// omits from the checkpoint. A stateful processor returns its
    /// [`ProcessorState`] snapshot instead.
    fn save_state(&self) -> serde_json::Value {
        serde_json::Value::Null
    }

    /// Restores state previously returned by [`ContextProcessor::save_state`].
    ///
    /// The default ignores the state; a stateful processor hands it to
    /// [`ProcessorState::restore`].
    fn load_state(&self, _state: &serde_json::Value) {}
}

/// Everything a processor needs besides the messages themselves.
pub struct ProcessorContext<'a> {
    /// Engine configuration.
    pub config: &'a ContextConfig,
    /// Counter used for every token measurement.
    ///
    /// Carries the `Send + Sync` bounds so a processor future stays `Send`; the
    /// reference coerces to `&dyn TokenCounter` wherever the plain trait is
    /// expected.
    pub counter: &'a (dyn TokenCounter + Send + Sync),
    /// Model available to processors that call one (summarisation).
    pub model: Option<&'a Arc<dyn Model>>,
    /// Store for offloaded content.
    pub offload: &'a dyn OffloadStore,
    /// Scope under which offload records are written.
    pub scope: &'a OffloadScope,
    /// Maximum tokens the history may occupy.
    pub budget: usize,
    /// Token count at which compression should trigger.
    pub trigger_tokens: usize,
    /// Whether the caller asked for compression explicitly, bypassing thresholds.
    pub manual: bool,
    /// Cancellation signalled by the caller.
    pub cancellation: &'a CancellationToken,
    /// Archive store compressors use to keep the history they replaced.
    ///
    /// `None` when compression recall is disabled and no store was configured;
    /// a processor that archives history must then skip archiving.
    pub recall: Option<&'a dyn RecallArchiveStore>,
    /// Session-memory notes full compaction can reinject instead of summarising.
    ///
    /// `None` when no store was configured; [`crate::processor::FullCompactProcessor`]
    /// then falls back to the model or offline summary path.
    pub session_memory: Option<&'a dyn SessionMemoryStore>,
}

/// Serializable description of one processor.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ProcessorConfig {
    /// Drop the oldest complete dialogue rounds.
    Truncation(TruncationConfig),
    /// Shorten or offload old tool results.
    ToolResult(ToolResultConfig),
    /// Replace older rounds with an LLM summary.
    Summarization(SummarizationConfig),
    /// Replace older complete rounds with historical memory blocks.
    DialogueCompressor(DialogueCompressorConfig),
    /// Compress the earlier part of the current round into a memory block.
    CurrentRoundCompressor(CurrentRoundCompressorConfig),
    /// Compress a history down to a total token target with the fallback ladder.
    RoundLevel(RoundLevelCompressorConfig),
    /// Replace an exhausted conversation with one boundary and summary.
    FullCompact(FullCompactProcessorConfig),
    /// Clear the text of stale tool results, keeping the newest ones per tool.
    MicroCompact(MicroCompactProcessorConfig),
    /// Offload large messages once the history outgrows the thresholds.
    MessageOffloader(MessageOffloaderConfig),
    /// Enforce a per-round token budget on tool results.
    ToolResultBudget(ToolResultBudgetProcessorConfig),
    /// Keep only the newest results of the configured tools.
    ToolResultWindow(ToolResultWindowProcessorConfig),
    /// Replace an oversized message with a summary and offload the original.
    MessageSummaryOffloader(MessageSummaryOffloaderConfig),
}
