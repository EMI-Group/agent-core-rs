//! Tool-result window processor: keep only the newest results of a tool.
//!
//! Split out of [`offloader`](crate::processor::offloader), whose three
//! processors it was written alongside; the re-export there keeps
//! `openjiuwen_context::processor::offloader::ToolResultWindowProcessor` valid.
use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::prompt::{BaseMessage, MessageContent, Role};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::events::ContextEvent;
use crate::offload::OffloadStore;
use crate::processor::impl_processor_state;
use crate::processor::offloader::{
    is_already_offloaded, offload_tool_result, offloaded_event, resolve_tool_name,
    WINDOWED_OUTPUT_TAG, WINDOW_MANAGED_METADATA_KEY,
};
use crate::processor::{ContextProcessor, ProcessorContext, ProcessorState};
use crate::tokens::count_messages;

/// Configuration of [`ToolResultWindowProcessor`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ToolResultWindowProcessorConfig {
    /// Tools whose results the window manages; empty disables the processor.
    pub tool_names: Vec<String>,
    /// Newest matching results kept in context with their full content.
    pub keep_last_k: usize,
    /// Characters kept as the preview of an offloaded result.
    pub trim_size: usize,
    /// Character length below which a result is compacted instead of offloaded.
    pub min_offload_chars: usize,
    /// Preview length used when a small result is compacted in place.
    pub small_result_trim_size: usize,
}

impl Default for ToolResultWindowProcessorConfig {
    fn default() -> Self {
        Self {
            tool_names: Vec::new(),
            keep_last_k: 3,
            trim_size: 3_000,
            min_offload_chars: 0,
            small_result_trim_size: 800,
        }
    }
}

impl ToolResultWindowProcessorConfig {
    /// Checks the window size.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] when
    /// [`ToolResultWindowProcessorConfig::keep_last_k`] is zero: an empty
    /// window would offload every result of the managed tools.
    pub fn validate(&self) -> Result<()> {
        if self.keep_last_k == 0 {
            return Err(Error::Config(
                "tool result window: keep_last_k must be greater than zero".to_owned(),
            ));
        }
        Ok(())
    }
}

/// Keeps only the newest `keep_last_k` results of the configured tools.
#[derive(Clone)]
pub struct ToolResultWindowProcessor {
    /// Processor configuration.
    pub config: ToolResultWindowProcessorConfig,
    /// Where offloaded originals are written.
    pub store: Arc<dyn OffloadStore>,
    /// Cross-run counters, persisted through the context checkpoint.
    pub state: ProcessorState,
}

impl std::fmt::Debug for ToolResultWindowProcessor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolResultWindowProcessor")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl ToolResultWindowProcessor {
    /// Creates a processor with `config` writing to `store`.
    #[must_use]
    pub fn new(config: ToolResultWindowProcessorConfig, store: Arc<dyn OffloadStore>) -> Self {
        Self {
            config,
            store,
            state: ProcessorState::default(),
        }
    }

    /// Whether the tool result at `index` belongs to a managed tool.
    fn is_target(&self, messages: &[BaseMessage], index: usize) -> bool {
        let message = &messages[index];
        if message.role != Role::Tool || message.content.as_text().is_none() {
            return false;
        }
        let Some(name) = resolve_tool_name(messages, index) else {
            return false;
        };
        self.config
            .tool_names
            .iter()
            .any(|suffix| name.ends_with(suffix.as_str()))
    }

    /// Whether the processor already handled the message at `index`.
    fn is_managed(messages: &[BaseMessage], index: usize) -> bool {
        let message = &messages[index];
        if message
            .metadata
            .get(WINDOW_MANAGED_METADATA_KEY)
            .and_then(Value::as_bool)
            == Some(true)
        {
            return true;
        }
        message.content.as_text().is_some_and(is_already_offloaded)
    }

    /// Shrinks the result in `message` without persisting it.
    fn compact(&self, message: &mut BaseMessage) {
        let Some(text) = message.content.as_text() else {
            return;
        };
        let characters = text.chars().count();
        let trim_size = self.config.small_result_trim_size;
        if characters > trim_size {
            let preview: String = text.chars().take(trim_size).collect();
            message.content = MessageContent::Text(format!(
                "<{WINDOWED_OUTPUT_TAG}>\n\
                 Older tool result compacted in context ({characters} chars); the full result was \
                 not persisted because it was below the offload threshold.\n\
                 Preview (first {trim_size} chars):\n\
                 {preview}\n\
                 ...\n\
                 </{WINDOWED_OUTPUT_TAG}>"
            ));
        }
        message
            .metadata
            .insert(WINDOW_MANAGED_METADATA_KEY.to_owned(), Value::Bool(true));
    }
}

#[async_trait]
impl ContextProcessor for ToolResultWindowProcessor {
    fn name(&self) -> &'static str {
        "tool_result_window"
    }

    impl_processor_state!(ToolResultWindowProcessor);

    async fn process(
        &self,
        messages: &mut Vec<BaseMessage>,
        cx: &ProcessorContext<'_>,
    ) -> Result<Option<ContextEvent>> {
        if cx.cancellation.is_cancelled() {
            return Err(Error::Cancelled);
        }
        if self.config.tool_names.is_empty() {
            return Ok(None);
        }
        let matched: Vec<usize> = (0..messages.len())
            .filter(|&index| self.is_target(messages, index))
            .collect();
        if matched.len() <= self.config.keep_last_k {
            return Ok(None);
        }

        let tokens_before = count_messages(cx.counter, messages);
        let outside = matched.len() - self.config.keep_last_k;
        let mut changed: Vec<usize> = Vec::new();
        let mut trimmed = 0u64;
        let mut last_handle: Option<String> = None;
        for &index in &matched[..outside] {
            if cx.cancellation.is_cancelled() {
                return Err(Error::Cancelled);
            }
            if Self::is_managed(messages, index) {
                continue;
            }
            let truncated = messages[index]
                .content
                .as_text()
                .map_or(0, |text| text.chars().count());
            if self.config.min_offload_chars > 0 && truncated < self.config.min_offload_chars {
                self.compact(&mut messages[index]);
                trimmed += 1;
            } else {
                last_handle = Some(
                    offload_tool_result(
                        &*self.store,
                        cx.scope,
                        self.name(),
                        &mut messages[index],
                        self.config.trim_size,
                        true,
                    )
                    .await?,
                );
            }
            changed.push(index);
        }
        if changed.is_empty() {
            return Ok(None);
        }
        let tokens_after = count_messages(cx.counter, messages);
        self.state
            .add("offloaded", (changed.len() as u64).saturating_sub(trimmed));
        if trimmed > 0 {
            self.state.add("trimmed", trimmed);
        }
        self.state
            .record_tokens("last_tokens_before", tokens_before);
        self.state.record_tokens("last_tokens_after", tokens_after);
        Ok(Some(offloaded_event(
            self.name(),
            &mut changed,
            last_handle,
            tokens_before,
            tokens_after,
        )))
    }
}
