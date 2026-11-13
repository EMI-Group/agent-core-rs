//! Summary offloader: replace an oversized message with a summary of itself.
//!
//! Ports the *contract* of Python's
//! `openjiuwen.core.context_engine.processor.offloader.message_summary_offloader`:
//! a message that is too large to stay in the window is summarised, the original
//! is written to the [`OffloadStore`], and the message keeps the summary plus an
//! [`OFFLOADED_INFO_TAG`] block telling the agent how to reload the original.
//!
//! The summary comes from the configured [`Model`] when one is available — the
//! processor's own model first, then the engine's — and from a deterministic
//! head/tail truncation ([`summarize_text_offline`]) otherwise. A failed model
//! call falls back to that truncation rather than failing the pass, so the
//! pipeline keeps working without credentials.
//!
//! Safety rules match [`offloader`](crate::processor::offloader): only plain text
//! bodies are rewritten, an assistant message that requests tool calls is never
//! rewritten, a message whose text already carries an offload marker is skipped,
//! and role, `name`, `metadata`, and `tool_call_id` are preserved.
use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::llm::Model;
use openjiuwen_foundation::prompt::{BaseMessage, MessageContent, Role};
use serde::{Deserialize, Serialize};

use crate::events::ContextEvent;
use crate::offload::OffloadStore;
use crate::processor::compressor::TRUNCATED_MARKER;
use crate::processor::impl_processor_state;
use crate::processor::offloader::{
    is_already_offloaded, is_protected_tool_message, offloaded_event, persist_offload,
    role_allowed, MessageOffloadRole,
};
use crate::processor::summarize;
use crate::processor::{ContextProcessor, ProcessorContext, ProcessorState};
use crate::tokens::{count_message, count_messages};

/// Tag that opens the block recording what was offloaded.
pub const OFFLOADED_INFO_TAG: &str = "[offloaded_info]";

/// Instruction sent to the model that writes the summary.
pub const SUMMARY_COMPRESSION_SYSTEM_PROMPT: &str = "Summarise this tool result so an agent can \
decide whether to reload the original. Keep identifiers, file paths, counts and error messages. \
Do not invent details.";

/// Configuration of [`MessageSummaryOffloader`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct MessageSummaryOffloaderConfig {
    /// Token size above which a message is summarised.
    pub large_message_threshold: usize,
    /// Roles eligible for summary offloading; other roles are always kept.
    pub offload_message_types: Vec<MessageOffloadRole>,
    /// Tools whose results are never summarised, `name` or `name:pattern`.
    pub protected_tool_names: Vec<String>,
    /// Token budget of the generated summary.
    pub summary_max_tokens: u32,
    /// Character budget of the offline summary fallback.
    pub content_max_chars_for_compression: usize,
}

impl Default for MessageSummaryOffloaderConfig {
    fn default() -> Self {
        Self {
            large_message_threshold: 1_000,
            offload_message_types: vec![MessageOffloadRole::Tool],
            protected_tool_names: vec!["read_file".to_owned()],
            summary_max_tokens: 900,
            content_max_chars_for_compression: 200_000,
        }
    }
}

/// Summarises oversized messages and offloads their originals.
#[derive(Clone)]
pub struct MessageSummaryOffloader {
    /// Processor configuration.
    pub config: MessageSummaryOffloaderConfig,
    /// Model used when the request does not supply one.
    pub model: Option<Arc<dyn Model>>,
    /// Where offloaded originals are written.
    pub store: Arc<dyn OffloadStore>,
    /// Cross-run counters, persisted through the context checkpoint.
    pub state: ProcessorState,
}

impl std::fmt::Debug for MessageSummaryOffloader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MessageSummaryOffloader")
            .field("config", &self.config)
            .field("has_model", &self.model.is_some())
            .finish_non_exhaustive()
    }
}

impl MessageSummaryOffloader {
    /// Creates a processor with `config`, an optional fallback `model`, and
    /// `store`.
    ///
    /// A missing model is not an error: the deterministic offline summary is
    /// used instead.
    #[must_use]
    pub fn new(
        config: MessageSummaryOffloaderConfig,
        model: Option<Arc<dyn Model>>,
        store: Arc<dyn OffloadStore>,
    ) -> Self {
        Self {
            config,
            model,
            store,
            state: ProcessorState::default(),
        }
    }

    /// Whether the message at `index` may be summarised.
    fn is_candidate(
        &self,
        messages: &[BaseMessage],
        index: usize,
        cx: &ProcessorContext<'_>,
    ) -> bool {
        let message = &messages[index];
        if !role_allowed(&self.config.offload_message_types, &message.role) {
            return false;
        }
        if message.role == Role::Assistant && !message.tool_calls.is_empty() {
            return false;
        }
        let Some(text) = message.content.as_text() else {
            return false;
        };
        if is_already_offloaded(text) {
            return false;
        }
        if is_protected_tool_message(messages, index, &self.config.protected_tool_names) {
            return false;
        }
        count_message(cx.counter, message) > self.config.large_message_threshold
    }

    /// Summarises `text`, falling back to the offline summary.
    async fn summarize(&self, text: &str, cx: &ProcessorContext<'_>) -> String {
        let model = self.model.as_ref().or(cx.model);
        summarize(
            model,
            SUMMARY_COMPRESSION_SYSTEM_PROMPT,
            text,
            self.config.summary_max_tokens,
            || summarize_text_offline(text, self.config.content_max_chars_for_compression),
        )
        .await
    }
}

/// Deterministic head/tail summary of `text`, used when no model can summarise.
///
/// `text` is returned unchanged when it is not longer than `max_chars`;
/// otherwise the head (~2/3 of the budget) and the tail (~1/3) are kept and
/// joined by [`TRUNCATED_MARKER`].
#[must_use]
pub fn summarize_text_offline(text: &str, max_chars: usize) -> String {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() <= max_chars {
        return text.to_owned();
    }
    let head_len = max_chars - max_chars / 3;
    let tail_len = max_chars - head_len;
    let head: String = chars[..head_len].iter().collect();
    let tail: String = chars[chars.len() - tail_len..].iter().collect();
    format!("{head}\n{TRUNCATED_MARKER}\n{tail}")
}

#[async_trait]
impl ContextProcessor for MessageSummaryOffloader {
    fn name(&self) -> &'static str {
        "message_summary_offloader"
    }

    impl_processor_state!(MessageSummaryOffloader);

    async fn process(
        &self,
        messages: &mut Vec<BaseMessage>,
        cx: &ProcessorContext<'_>,
    ) -> Result<Option<ContextEvent>> {
        if cx.cancellation.is_cancelled() {
            return Err(Error::Cancelled);
        }
        // The trigger *is* the candidate scan: a processor with nothing large
        // enough cannot invent work, so `manual` only matters for the absence of
        // any other threshold check.
        let candidates: Vec<usize> = (0..messages.len())
            .filter(|&index| self.is_candidate(messages, index, cx))
            .collect();
        if candidates.is_empty() {
            return Ok(None);
        }

        let tokens_before = count_messages(cx.counter, messages);
        let mut changed: Vec<usize> = Vec::new();
        let mut last_handle: Option<String> = None;
        for index in candidates {
            if cx.cancellation.is_cancelled() {
                return Err(Error::Cancelled);
            }
            let original = messages[index].clone();
            let text = original.content.as_text().unwrap_or_default();
            let characters = text.chars().count();
            let summary = self.summarize(text, cx).await;
            let (handle, marker) =
                persist_offload(&*self.store, cx.scope, self.name(), &original, characters).await?;
            messages[index].content = MessageContent::Text(format!(
                "{summary}\n\n{OFFLOADED_INFO_TAG}\noriginal_chars: {characters}\n{marker}"
            ));
            changed.push(index);
            last_handle = Some(handle);
        }
        let tokens_after = count_messages(cx.counter, messages);
        self.state.add("offloaded", changed.len() as u64);
        self.state.add("summaries", changed.len() as u64);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offline_summary_keeps_the_head_and_the_tail() {
        let text = "a".repeat(90);
        let summary = summarize_text_offline(&text, 30);
        assert!(summary.starts_with(&"a".repeat(20)));
        assert!(summary.contains(TRUNCATED_MARKER));
        assert_eq!(summarize_text_offline("short", 30), "short");
    }
}
