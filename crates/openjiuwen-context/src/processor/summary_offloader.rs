//! Summary offloader: replace an oversized message with a summary of itself.
//!
//! Ports the *contract* of Python's
//! `openjiuwen.core.context_engine.processor.offloader.message_summary_offloader`:
//! a message that is too large to stay in the window is rule-compressed first,
//! then (when it is still too large) summarised, the original is written to the
//! [`OffloadStore`], and the message keeps the summary plus an
//! [`OFFLOADED_INFO_TAG`] block telling the agent how to reload the original.
//!
//! Tier 1 is [`RuleCompressionPipeline`] — deterministic, model-free and offline.
//! It runs only on the same candidates as the summary path (role, size,
//! protection and marker gates unchanged), only when the message carries no
//! `rule_compressed_at` metadata, and only when its text exceeds
//! [`MessageSummaryOffloaderConfig::rule_compression_max_chars`]. Tier 2 then
//! handles whatever survives: a rule-compressed body that already fits the
//! character budget is kept as is, and a larger one is summarised and offloaded
//! exactly like before.
//!
//! [`RuleCompressionTierConfig`] selects between Python's three offload
//! strategies (see [`RuleCompressionStrategy`]) and the dump format, through
//! [`MessageSummaryOffloader::with_rule_compression_tier`]; the default,
//! `rule_then_truncate`, is the behaviour described above.
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
use serde_json::Value;

use crate::events::ContextEvent;
use crate::offload::OffloadStore;
use crate::processor::compressor::TRUNCATED_MARKER;
use crate::processor::impl_processor_state;
use crate::processor::offloader::{
    is_already_offloaded, is_protected_tool_message, offloaded_event, persist_offload,
    role_allowed, MessageOffloadRole,
};
use crate::processor::rule_compression::{
    RuleCompressionDumpFormat, RuleCompressionPipeline, RuleCompressionStrategy, ADD_PASS,
    RULE_COMPRESSED_AT_KEY,
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
    /// Whether oversized tool results are rule-compressed before summary offloading.
    pub enable_rule_compression: bool,
    /// Character budget of the rule-compression tier: the size above which it
    /// runs and the size it compresses towards.
    pub rule_compression_max_chars: usize,
    /// Whether every rule compression is dumped to disk for debugging.
    pub enable_rule_compression_dump: bool,
    /// Directory of those dumps; `None` falls back to
    /// `OPENJIUWEN_RULE_COMPRESSION_DUMP_DIR`.
    pub rule_compression_dump_dir: Option<String>,
}

impl Default for MessageSummaryOffloaderConfig {
    fn default() -> Self {
        Self {
            large_message_threshold: 1_000,
            offload_message_types: vec![MessageOffloadRole::Tool],
            protected_tool_names: vec!["read_file".to_owned()],
            summary_max_tokens: 900,
            content_max_chars_for_compression: 200_000,
            enable_rule_compression: true,
            rule_compression_max_chars: 20_000,
            enable_rule_compression_dump: false,
            rule_compression_dump_dir: None,
        }
    }
}

/// Whether, and in what shape, the rule-compression tier participates.
///
/// This is Python's `OffloadStrategy` promoted from a module constant to a serde
/// configuration, plus the dump format the tier writes:
///
/// - `rule_then_truncate` (the default) compresses deterministically and, when the
///   result still exceeds [`MessageSummaryOffloaderConfig::rule_compression_max_chars`],
///   falls through to the summary/offload tier,
/// - `rule_only` keeps the rule-compressed body even when it still exceeds that
///   budget and never summarises or offloads; a candidate the rule tier declined
///   is left untouched,
/// - `truncate_only` skips rule compression entirely and goes straight to the
///   summary/offload tier.
///
/// An unknown strategy spelling is refused: `serde` fails to deserialize it
/// instead of falling back to the default, and
/// [`RuleCompressionTierConfig::from_json_value`] maps that failure to
/// [`Error::Config`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct RuleCompressionTierConfig {
    /// Which tiers may rewrite a candidate.
    pub strategy: RuleCompressionStrategy,
    /// How a rule-compression application is dumped.
    pub dump_format: RuleCompressionDumpFormat,
}

impl RuleCompressionTierConfig {
    /// A tier that runs rule compression first and dumps pretty JSON.
    #[must_use]
    pub fn new(strategy: RuleCompressionStrategy) -> Self {
        Self {
            strategy,
            dump_format: RuleCompressionDumpFormat::default(),
        }
    }

    /// Parses the tier out of a JSON object, refusing unknown values.
    ///
    /// The typed twin of `serde_json::from_value`: an unknown `strategy` or
    /// `dump_format` becomes [`Error::Config`] naming the problem, never a silent
    /// default.
    pub fn from_json_value(value: &Value) -> Result<Self> {
        serde_json::from_value(value.clone()).map_err(|error| {
            Error::config(format!(
                "invalid rule-compression tier configuration {value}: {error}"
            ))
        })
    }

    /// Whether the rule tier runs.
    #[must_use]
    pub fn runs_rule_compression(&self) -> bool {
        self.strategy.runs_rule_compression()
    }

    /// Whether the summary/offload tier runs.
    #[must_use]
    pub fn runs_fallback_offload(&self) -> bool {
        self.strategy.runs_fallback_offload()
    }

    /// Whether a rule-compressed body that still exceeds the budget is summarised.
    #[must_use]
    pub fn truncates_rule_preview(&self) -> bool {
        self.strategy.truncates_rule_preview()
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
    /// Tier 1: the deterministic, model-free rule compressor.
    pub pipeline: RuleCompressionPipeline,
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
        let pipeline = RuleCompressionPipeline::new().with_dump(
            config.enable_rule_compression_dump,
            config.rule_compression_dump_dir.clone(),
        );
        Self {
            config,
            model,
            store,
            state: ProcessorState::default(),
            pipeline,
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

    /// Whether the message already carries a rule-compression stamp.
    fn is_rule_compressed(message: &BaseMessage) -> bool {
        message.metadata.contains_key(RULE_COMPRESSED_AT_KEY)
    }

    /// The tier configuration this processor runs with.
    #[must_use]
    pub fn rule_compression_tier(&self) -> RuleCompressionTierConfig {
        RuleCompressionTierConfig {
            strategy: self.pipeline.strategy(),
            dump_format: self.pipeline.dump_format(),
        }
    }

    /// Selects Python's offload strategy and the dump format of the rule tier.
    ///
    /// The two knobs live on the tier configuration rather than on
    /// [`MessageSummaryOffloaderConfig`] so the processor configuration stays
    /// exhaustively constructible.
    #[must_use]
    pub fn with_rule_compression_tier(mut self, tier: RuleCompressionTierConfig) -> Self {
        self.pipeline = self
            .pipeline
            .clone()
            .with_strategy(tier.strategy)
            .with_dump_format(tier.dump_format);
        self
    }

    /// Runs the tier-1 rule compressor on the message at `index`, under `cx`'s scope.
    ///
    /// Returns the indices entry replaced in `messages` and whether tier 2 may
    /// keep the result as is (it fits the character budget). The scope is what
    /// makes a dump carry the real session and context ids.
    fn rule_compress(
        &self,
        messages: &mut [BaseMessage],
        index: usize,
        cx: &ProcessorContext<'_>,
    ) -> (bool, bool) {
        if !self.config.enable_rule_compression
            || !self.pipeline.strategy().runs_rule_compression()
            || Self::is_rule_compressed(&messages[index])
        {
            return (false, false);
        }
        let max_chars = self.config.rule_compression_max_chars.max(1);
        let replaced = match self
            .pipeline
            .compress_in_messages_with_scope(messages, index, ADD_PASS, max_chars, false, cx.scope)
        {
            Ok(replaced) => replaced,
            Err(error) => {
                tracing::warn!("[RuleCompression] tier 1 failed for message {index}: {error}");
                return (false, false);
            }
        };
        if !Self::is_rule_compressed(&replaced) {
            return (false, false);
        }
        let fits = replaced
            .content
            .as_text()
            .is_none_or(|text| text.chars().count() <= max_chars);
        messages[index] = replaced;
        (true, fits)
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
        let tier = self.rule_compression_tier();
        let mut changed: Vec<usize> = Vec::new();
        let mut last_handle: Option<String> = None;
        let mut rule_compressed = 0_usize;
        let mut offloaded_count = 0_usize;
        for index in candidates {
            if cx.cancellation.is_cancelled() {
                return Err(Error::Cancelled);
            }
            let (tier1, fits) = self.rule_compress(messages, index, cx);
            if tier1 {
                rule_compressed += 1;
                if fits || !tier.truncates_rule_preview() {
                    // `rule_only` keeps the compressed body even when it still
                    // exceeds the budget; `rule_then_truncate` keeps it when it fits.
                    changed.push(index);
                    continue;
                }
            } else if !tier.runs_fallback_offload() {
                // `rule_only` never summarises or offloads: a candidate the rule
                // tier declined stays exactly as it was.
                continue;
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
            offloaded_count += 1;
        }
        let tokens_after = count_messages(cx.counter, messages);
        self.state.add("offloaded", offloaded_count as u64);
        self.state.add("summaries", offloaded_count as u64);
        self.state.add("rule_compressed", rule_compressed as u64);
        self.state
            .record_tokens("last_tokens_before", tokens_before);
        self.state.record_tokens("last_tokens_after", tokens_after);
        if changed.is_empty() {
            return Ok(None);
        }
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
