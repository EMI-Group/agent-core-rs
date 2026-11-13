//! Processor pipeline: the ordered transforms that compress a history.
//!
//! A processor receives the mutable history and a [`ProcessorContext`] holding
//! the configuration, token counter, optional model, offload store, and
//! cancellation. It returns a [`ContextEvent`] when it changed something.
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
use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::llm::Model;
use openjiuwen_foundation::prompt::{BaseMessage, TokenCounter};
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

use crate::config::ContextConfig;
use crate::events::ContextEvent;
use crate::offload::{OffloadScope, OffloadStore};
use crate::window::drop_orphan_tool_results;

pub mod compressor;
pub mod offloader;
pub mod summarize;
pub mod summary_offloader;
pub mod tool_result;
pub mod truncate;

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
pub use self::offloader::{
    wildcard_match, MessageOffloadRole, MessageOffloader, MessageOffloaderConfig,
    ToolResultBudgetProcessor, ToolResultBudgetProcessorConfig, ToolResultWindowProcessor,
    ToolResultWindowProcessorConfig, OMIT_STRING, WINDOWED_OUTPUT_TAG, WINDOW_MANAGED_METADATA_KEY,
};
pub use self::summarize::{SummarizationConfig, SummarizationProcessor, CONTEXT_SUMMARY_PREFIX};
pub use self::summary_offloader::{
    summarize_text_offline, MessageSummaryOffloader, MessageSummaryOffloaderConfig,
    OFFLOADED_INFO_TAG, SUMMARY_COMPRESSION_SYSTEM_PROMPT,
};
pub use self::tool_result::{
    ToolResultConfig, ToolResultProcessor, PERSISTED_OUTPUT_TAG, TRUNCATED_MARKER,
};
pub use self::truncate::{TruncationConfig, TruncationProcessor};

/// One transform applied to a conversation history.
#[async_trait]
pub trait ContextProcessor: Send + Sync {
    /// Stable processor name, reported in events.
    fn name(&self) -> &'static str;

    /// Applies the transform in place and reports what changed.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Cancelled`] when the caller cancelled, and the
    /// processor's own failure (for example [`Error::Llm`] for a summarisation
    /// transport error) otherwise.
    async fn process(
        &self,
        messages: &mut Vec<BaseMessage>,
        cx: &ProcessorContext<'_>,
    ) -> Result<Option<ContextEvent>>;

    /// Serializes any state the processor keeps across runs.
    fn save_state(&self) -> serde_json::Value {
        serde_json::Value::Null
    }

    /// Restores state previously returned by [`ContextProcessor::save_state`].
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
}

/// An ordered list of processors.
#[derive(Clone, Default)]
pub struct ContextPipeline {
    processors: Vec<Arc<dyn ContextProcessor>>,
}

impl std::fmt::Debug for ContextPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let names: Vec<&str> = self
            .processors
            .iter()
            .map(|processor| processor.name())
            .collect();
        f.debug_struct("ContextPipeline")
            .field("processors", &names)
            .finish()
    }
}

impl ContextPipeline {
    /// Creates a pipeline over `processors`, in execution order.
    #[must_use]
    pub fn new(processors: Vec<Arc<dyn ContextProcessor>>) -> Self {
        Self { processors }
    }

    /// Builds the pipeline described by `configs`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] when a summarisation processor is configured
    /// without a model, and when an offloader config fails its own
    /// [`validate`](MessageOffloaderConfig::validate) checks. The summary
    /// offloader never fails for a missing model: it falls back to the
    /// deterministic offline summary. Truncation and tool-result processors
    /// never fail.
    pub fn from_configs(
        configs: &[ProcessorConfig],
        model: Option<Arc<dyn Model>>,
        offload: Arc<dyn OffloadStore>,
    ) -> Result<Self> {
        let mut processors: Vec<Arc<dyn ContextProcessor>> = Vec::with_capacity(configs.len());
        for config in configs {
            match config {
                ProcessorConfig::Truncation(config) => {
                    processors.push(Arc::new(TruncationProcessor {
                        config: config.clone(),
                    }));
                }
                ProcessorConfig::ToolResult(config) => {
                    processors.push(Arc::new(ToolResultProcessor {
                        config: config.clone(),
                        store: Arc::clone(&offload),
                    }));
                }
                ProcessorConfig::Summarization(config) => {
                    let model = model.clone().ok_or_else(|| {
                        Error::Config(
                            "the summarization processor requires a model, but none is configured"
                                .to_owned(),
                        )
                    })?;
                    processors.push(Arc::new(SummarizationProcessor {
                        config: config.clone(),
                        model,
                    }));
                }
                ProcessorConfig::DialogueCompressor(config) => {
                    config.validate()?;
                    processors.push(Arc::new(DialogueCompressor::new(
                        config.clone(),
                        model.clone(),
                    )));
                }
                ProcessorConfig::CurrentRoundCompressor(config) => {
                    config.validate()?;
                    processors.push(Arc::new(CurrentRoundCompressor::new(
                        config.clone(),
                        model.clone(),
                    )));
                }
                ProcessorConfig::RoundLevel(config) => {
                    config.validate()?;
                    processors.push(Arc::new(RoundLevelCompressor::new(
                        config.clone(),
                        model.clone(),
                    )));
                }
                ProcessorConfig::FullCompact(config) => {
                    // The integration round wires the session-memory store; here
                    // the processor always has the model-summary path available.
                    processors.push(Arc::new(FullCompactProcessor::new(
                        config.clone(),
                        model.clone(),
                        None,
                    )));
                }
                ProcessorConfig::MicroCompact(config) => {
                    config.validate()?;
                    processors.push(Arc::new(MicroCompactProcessor::new(config.clone())));
                }
                ProcessorConfig::MessageOffloader(config) => {
                    config.validate()?;
                    processors.push(Arc::new(MessageOffloader::new(
                        config.clone(),
                        Arc::clone(&offload),
                    )));
                }
                ProcessorConfig::ToolResultBudget(config) => {
                    config.validate()?;
                    processors.push(Arc::new(ToolResultBudgetProcessor::new(
                        config.clone(),
                        Arc::clone(&offload),
                    )));
                }
                ProcessorConfig::ToolResultWindow(config) => {
                    config.validate()?;
                    processors.push(Arc::new(ToolResultWindowProcessor::new(
                        config.clone(),
                        Arc::clone(&offload),
                    )));
                }
                ProcessorConfig::MessageSummaryOffloader(config) => {
                    // A missing model is not an error: the processor falls back
                    // to the deterministic offline summary.
                    processors.push(Arc::new(MessageSummaryOffloader::new(
                        config.clone(),
                        model.clone(),
                        Arc::clone(&offload),
                    )));
                }
            }
        }
        Ok(Self::new(processors))
    }

    /// The processors, in execution order.
    #[must_use]
    pub fn processors(&self) -> &[Arc<dyn ContextProcessor>] {
        &self.processors
    }

    /// Runs every processor in order and returns the events they produced.
    ///
    /// Cancellation is checked before each processor. A processor error is
    /// propagated as-is; changes made by earlier processors stay applied, and
    /// every processor that mutates builds its replacement first so a failure
    /// never leaves a half-written history. Afterwards orphan tool results are
    /// dropped (a processor may have removed the assistant message that owned
    /// one) and the repair is logged.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Cancelled`] when the cancellation token is set, and the
    /// failing processor's error otherwise.
    pub async fn run(
        &self,
        messages: &mut Vec<BaseMessage>,
        cx: &ProcessorContext<'_>,
    ) -> Result<Vec<ContextEvent>> {
        let mut events = Vec::new();
        for processor in &self.processors {
            if cx.cancellation.is_cancelled() {
                return Err(Error::Cancelled);
            }
            if let Some(event) = processor.process(messages, cx).await? {
                events.push(event);
            }
        }
        let dropped = drop_orphan_tool_results(messages);
        if dropped > 0 {
            tracing::warn!(
                dropped,
                "dropped orphan tool results after the pipeline ran"
            );
        }
        Ok(events)
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::ContextEventKind;
    use crate::offload::{InMemoryOffloadStore, OffloadKind, OffloadRecord};
    use openjiuwen_foundation::prompt::{ApproxTokenCounter, Role, ToolCall};

    fn counter() -> ApproxTokenCounter {
        ApproxTokenCounter::with_chars_per_token(1)
    }

    /// A processor that records its name and appends nothing.
    struct Recorder {
        name: &'static str,
        seen: Arc<std::sync::Mutex<Vec<&'static str>>>,
    }

    #[async_trait]
    impl ContextProcessor for Recorder {
        fn name(&self) -> &'static str {
            self.name
        }

        async fn process(
            &self,
            _messages: &mut Vec<BaseMessage>,
            _cx: &ProcessorContext<'_>,
        ) -> Result<Option<ContextEvent>> {
            self.seen
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push(self.name);
            Ok(Some(ContextEvent {
                processor: self.name.to_owned(),
                kind: ContextEventKind::Truncated,
                ..ContextEvent::default()
            }))
        }
    }

    struct Failing;

    #[async_trait]
    impl ContextProcessor for Failing {
        fn name(&self) -> &'static str {
            "failing"
        }

        async fn process(
            &self,
            _messages: &mut Vec<BaseMessage>,
            _cx: &ProcessorContext<'_>,
        ) -> Result<Option<ContextEvent>> {
            Err(Error::Llm("boom".to_owned()))
        }
    }

    #[test]
    fn configs_deserialize_by_tag_and_round_trip() {
        let json = r#"[
            {"type": "tool_result", "max_result_tokens": 50},
            {"type": "truncation", "keep_recent_messages": 2},
            {"type": "summarization"}
        ]"#;
        let configs: Vec<ProcessorConfig> = serde_json::from_str(json).unwrap();
        assert!(matches!(configs[0], ProcessorConfig::ToolResult(_)));
        assert!(matches!(configs[1], ProcessorConfig::Truncation(_)));
        assert!(matches!(configs[2], ProcessorConfig::Summarization(_)));
        assert_eq!(
            serde_json::from_str::<Vec<ProcessorConfig>>(&serde_json::to_string(&configs).unwrap())
                .unwrap(),
            configs
        );
    }

    #[test]
    fn from_configs_requires_a_model_for_summarization() {
        let store: Arc<dyn OffloadStore> = Arc::new(InMemoryOffloadStore::new());
        let error = ContextPipeline::from_configs(
            &[ProcessorConfig::Summarization(
                SummarizationConfig::default(),
            )],
            None,
            store,
        )
        .unwrap_err();
        assert!(matches!(error, Error::Config(_)));

        let store: Arc<dyn OffloadStore> = Arc::new(InMemoryOffloadStore::new());
        let pipeline = ContextPipeline::from_configs(
            &[
                ProcessorConfig::ToolResult(ToolResultConfig::default()),
                ProcessorConfig::Truncation(TruncationConfig::default()),
            ],
            None,
            store,
        )
        .unwrap();
        assert_eq!(pipeline.processors().len(), 2);
        assert_eq!(pipeline.processors()[0].name(), "tool_result");
        assert_eq!(pipeline.processors()[1].name(), "truncation");
    }

    fn context<'a>(
        config: &'a ContextConfig,
        counter: &'a (dyn TokenCounter + Send + Sync),
        store: &'a dyn OffloadStore,
        scope: &'a OffloadScope,
        cancellation: &'a CancellationToken,
    ) -> ProcessorContext<'a> {
        ProcessorContext {
            config,
            counter,
            model: None,
            offload: store,
            scope,
            budget: 1000,
            trigger_tokens: 800,
            manual: false,
            cancellation,
        }
    }

    #[tokio::test]
    async fn run_is_ordered_and_collects_events() {
        let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
        let pipeline = ContextPipeline::new(vec![
            Arc::new(Recorder {
                name: "first",
                seen: Arc::clone(&seen),
            }),
            Arc::new(Recorder {
                name: "second",
                seen: Arc::clone(&seen),
            }),
        ]);
        let config = ContextConfig::default();
        let counter = counter();
        let store = InMemoryOffloadStore::new();
        let scope = OffloadScope::new("s", "main");
        let cancellation = CancellationToken::new();
        let cx = context(&config, &counter, &store, &scope, &cancellation);
        let mut messages = vec![BaseMessage::user("hi")];

        let events = pipeline.run(&mut messages, &cx).await.unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].processor, "first");
        assert_eq!(events[1].processor, "second");
        assert_eq!(
            *seen
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
            vec!["first", "second"]
        );
    }

    #[tokio::test]
    async fn run_propagates_errors_and_checks_cancellation() {
        let config = ContextConfig::default();
        let counter = counter();
        let store = InMemoryOffloadStore::new();
        let scope = OffloadScope::new("s", "main");
        let cancellation = CancellationToken::new();
        let cx = context(&config, &counter, &store, &scope, &cancellation);
        let mut messages = vec![BaseMessage::user("hi")];

        let pipeline = ContextPipeline::new(vec![Arc::new(Failing)]);
        let error = pipeline.run(&mut messages, &cx).await.unwrap_err();
        assert!(matches!(error, Error::Llm(_)));

        let cancelled = CancellationToken::new();
        cancelled.cancel();
        let pipeline = ContextPipeline::new(vec![Arc::new(Recorder {
            name: "never",
            seen: Arc::new(std::sync::Mutex::new(Vec::new())),
        })]);
        let cx = ProcessorContext {
            config: &config,
            counter: &counter,
            model: None,
            offload: &store,
            scope: &scope,
            budget: 1000,
            trigger_tokens: 800,
            manual: false,
            cancellation: &cancelled,
        };
        let error = pipeline.run(&mut messages, &cx).await.unwrap_err();
        assert!(matches!(error, Error::Cancelled));
    }

    #[tokio::test]
    async fn run_repairs_orphan_tool_results() {
        struct DropAssistant;
        #[async_trait]
        impl ContextProcessor for DropAssistant {
            fn name(&self) -> &'static str {
                "drop_assistant"
            }
            async fn process(
                &self,
                messages: &mut Vec<BaseMessage>,
                _cx: &ProcessorContext<'_>,
            ) -> Result<Option<ContextEvent>> {
                messages.retain(|message| message.role != Role::Assistant);
                Ok(None)
            }
        }

        let config = ContextConfig::default();
        let counter = counter();
        let store = InMemoryOffloadStore::new();
        let scope = OffloadScope::new("s", "main");
        let cancellation = CancellationToken::new();
        let cx = context(&config, &counter, &store, &scope, &cancellation);
        let mut messages = vec![
            BaseMessage::user("q"),
            BaseMessage::assistant("").with_tool_calls(vec![ToolCall::new("t").with_id("call_1")]),
            BaseMessage::tool("call_1", "result"),
        ];
        let pipeline = ContextPipeline::new(vec![Arc::new(DropAssistant)]);
        pipeline.run(&mut messages, &cx).await.unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, Role::User);
    }

    #[test]
    fn processors_are_dyn_compatible() {
        let processors: Vec<Arc<dyn ContextProcessor>> = vec![
            Arc::new(TruncationProcessor {
                config: TruncationConfig::default(),
            }),
            Arc::new(ToolResultProcessor {
                config: ToolResultConfig::default(),
                store: Arc::new(InMemoryOffloadStore::new()),
            }),
        ];
        assert_eq!(processors[0].name(), "truncation");
        assert_eq!(processors[1].name(), "tool_result");
        assert_eq!(processors[1].save_state(), serde_json::Value::Null);
        let _ = OffloadKind::InMemory;
        let _ = OffloadRecord::default();
    }
}
