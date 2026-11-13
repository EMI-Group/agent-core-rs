//! Current-round compressor: memory blocks for the current round's own history.
//!
//! Long agent rounds accumulate the bulk of their context *inside* one round:
//! after the last user message come the tool calls, their results and the
//! assistant's progress messages. This compressor compresses that leading
//! portion while the newest messages and the user request stay raw.
//!
//! It runs two stages:
//!
//! 1. **Increment** — when the span between the last user message and the raw
//!    tail holds at least
//!    [`CurrentRoundCompressorConfig::min_selected_tokens_for_compression`]
//!    tokens, it is summarised into one
//!    [`CURRENT_ROUND_MEMORY_BLOCK`](crate::processor::compressor::CURRENT_ROUND_MEMORY_BLOCK)
//!    user message. The span is clamped back to the last completed API round, so
//!    a tool-call/tool-result exchange is never split, and the replacement must
//!    be smaller than the span.
//! 2. **Merge** — the first contiguous run of at least
//!    [`CurrentRoundCompressorConfig::summary_merge_min_blocks`] current-round
//!    blocks is collapsed into a single block, keeping the block bodies joined
//!    with `\n---\n`. This runs even when stage 1 was skipped, so repeated
//!    passes cannot leave a long tail of tiny blocks.
//!
//! Summaries come from the configured model when there is one and from the
//! deterministic offline transcript summary otherwise.
//!
//! [`CURRENT_ROUND_MEMORY_BLOCK`]: crate::processor::compressor::CURRENT_ROUND_MEMORY_BLOCK
use std::ops::Range;
use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::llm::Model;
use openjiuwen_foundation::prompt::{BaseMessage, Role};
use serde::{Deserialize, Serialize};

use crate::events::{ContextEvent, ContextEventKind};
use crate::processor::compressor::{
    append_recall_markers, archive_replaced_messages, find_last_completed_round_end,
    iter_summary_merge_ranges, rollback_recall_archive, split_recall_markers, summarize,
    summarize_offline, MemoryBlockKind, CURRENT_ROUND_MEMORY_BLOCK, SUMMARY_JOIN_SEPARATOR,
};
use crate::processor::{ContextProcessor, ProcessorContext};
use crate::tokens::{count_message, count_messages};

/// Characters per token used to size the offline summary fallback.
const CHARS_PER_TOKEN_ESTIMATE: usize = 4;

/// Instruction used when no custom compression prompt is configured.
const DEFAULT_COMPRESSION_PROMPT: &str = "Summarise the selected earlier part of the current \
task round into compact reference memory. Keep decisions, identifiers, file paths, open questions \
and the exact point where the work stopped. Do not invent details.";

/// Configuration of [`CurrentRoundCompressor`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct CurrentRoundCompressorConfig {
    /// Trigger when the history holds more than this many tokens.
    pub tokens_threshold: usize,
    /// Newest messages that are never compressed.
    pub messages_to_keep: usize,
    /// Minimum size of the compressible span.
    pub min_selected_tokens_for_compression: usize,
    /// Token budget each compressed block should stay within.
    pub compression_target_tokens: usize,
    /// Minimum number of blocks the merge stage collapses; at least `2`.
    pub summary_merge_min_blocks: usize,
    /// Reference window used by the model prompt.
    pub prior_context_window_size: usize,
    /// Instruction replacing the built-in summary prompt when set.
    pub custom_compression_prompt: Option<String>,
}

impl Default for CurrentRoundCompressorConfig {
    fn default() -> Self {
        Self {
            tokens_threshold: 100_000,
            messages_to_keep: 3,
            min_selected_tokens_for_compression: 20_000,
            compression_target_tokens: 4_000,
            summary_merge_min_blocks: 3,
            prior_context_window_size: 10,
            custom_compression_prompt: None,
        }
    }
}

impl CurrentRoundCompressorConfig {
    /// Checks the thresholds and the merge minimum.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] for a zero threshold or target, and when
    /// [`CurrentRoundCompressorConfig::summary_merge_min_blocks`] is smaller
    /// than `2` (a merge of fewer blocks cannot shrink the history).
    pub fn validate(&self) -> Result<()> {
        if self.tokens_threshold == 0 {
            return Err(Error::Config(
                "current-round compressor: tokens_threshold must be greater than zero".to_owned(),
            ));
        }
        if self.compression_target_tokens == 0 {
            return Err(Error::Config(
                "current-round compressor: compression_target_tokens must be greater than zero"
                    .to_owned(),
            ));
        }
        if self.summary_merge_min_blocks < 2 {
            return Err(Error::Config(format!(
                "current-round compressor: summary_merge_min_blocks ({}) must be at least 2",
                self.summary_merge_min_blocks
            )));
        }
        Ok(())
    }
}

/// Compresses the earlier part of the current round into a memory block.
#[derive(Clone)]
pub struct CurrentRoundCompressor {
    /// Processor configuration.
    pub config: CurrentRoundCompressorConfig,
    /// Model used when the request does not supply one.
    pub model: Option<Arc<dyn Model>>,
}

impl std::fmt::Debug for CurrentRoundCompressor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CurrentRoundCompressor")
            .field("config", &self.config)
            .field("has_model", &self.model.is_some())
            .finish()
    }
}

impl CurrentRoundCompressor {
    /// Creates a compressor with `config` and an optional fallback `model`.
    #[must_use]
    pub fn new(config: CurrentRoundCompressorConfig, model: Option<Arc<dyn Model>>) -> Self {
        Self { config, model }
    }

    /// Whether the thresholds ask for a pass.
    fn should_run(&self, messages: &[BaseMessage], cx: &ProcessorContext<'_>) -> bool {
        if cx.manual {
            return true;
        }
        messages.len() >= self.config.messages_to_keep
            && count_messages(cx.counter, messages) > self.config.tokens_threshold
    }

    /// The compressible span: everything between the last user message and the
    /// raw tail, clamped back to the last completed API round.
    ///
    /// Returns `None` when there is nothing safe to compress.
    fn span(&self, messages: &[BaseMessage], boundary: usize) -> Option<Range<usize>> {
        let len = messages.len();
        let keep = self.config.messages_to_keep;
        if boundary + 1 >= len || boundary + keep >= len {
            return None;
        }
        let start = boundary + 1;
        let end = len - keep;
        if start >= end {
            return None;
        }
        let end = find_last_completed_round_end(messages, start, end - 1)?;
        (end > start).then_some(start..end)
    }

    /// Maps an index of the post-stage-1 history back to the original history.
    ///
    /// Stage 1 replaces a contiguous span with one message, so every later
    /// position shifts left by the number of removed messages.
    fn map_to_original(index: usize, stage_one: Option<&Range<usize>>) -> usize {
        let Some(span) = stage_one else {
            return index;
        };
        if index <= span.start {
            index
        } else {
            index + (span.end - span.start - 1)
        }
    }

    /// Merges the first eligible run of memory blocks, returning the bodies of
    /// the blocks it collapsed.
    fn merge_blocks(
        &self,
        messages: &mut Vec<BaseMessage>,
        cx: &ProcessorContext<'_>,
        changed: &mut Vec<usize>,
        stage_one: Option<&Range<usize>>,
    ) -> Option<String> {
        let ranges = iter_summary_merge_ranges(
            messages,
            CURRENT_ROUND_MEMORY_BLOCK,
            self.config.summary_merge_min_blocks,
        );
        let run = ranges.into_iter().next()?;
        let blocks = messages[run.clone()].to_vec();
        let before = count_messages(cx.counter, &blocks);
        let gate = self
            .config
            .compression_target_tokens
            .saturating_mul(self.config.summary_merge_min_blocks);
        if before <= gate {
            return None;
        }
        // Bodies are merged; every recall marker the individual blocks carried is
        // kept, deduplicated and in original order, after the merged body.
        let mut markers: Vec<String> = Vec::new();
        let bodies: Vec<String> = blocks
            .iter()
            .map(|message| {
                let body = MemoryBlockKind::CurrentRound
                    .summary_body(&message.text())
                    .map_or_else(|| message.text().into_owned(), str::to_owned);
                let (body, found) = split_recall_markers(&body);
                for marker in found {
                    if !markers.contains(&marker) {
                        markers.push(marker);
                    }
                }
                body
            })
            .collect();
        let merged_body = bodies.join(SUMMARY_JOIN_SEPARATOR);
        let rendered =
            append_recall_markers(MemoryBlockKind::CurrentRound.render(&merged_body), &markers);
        let merged = BaseMessage::user(rendered);
        messages.splice(run.clone(), std::iter::once(merged));
        changed.extend(run.map(|index| Self::map_to_original(index, stage_one)));
        Some(merged_body)
    }
}

#[async_trait]
impl ContextProcessor for CurrentRoundCompressor {
    fn name(&self) -> &'static str {
        "current_round"
    }

    async fn process(
        &self,
        messages: &mut Vec<BaseMessage>,
        cx: &ProcessorContext<'_>,
    ) -> Result<Option<ContextEvent>> {
        if !self.should_run(messages, cx) {
            return Ok(None);
        }
        let Some(boundary) = messages
            .iter()
            .rposition(|message| message.role == Role::User)
        else {
            return Ok(None);
        };
        if boundary + 1 >= messages.len() {
            return Ok(None);
        }
        let tokens_before = count_messages(cx.counter, messages);
        let max_chars = self
            .config
            .compression_target_tokens
            .saturating_mul(CHARS_PER_TOKEN_ESTIMATE);
        let max_tokens = u32::try_from(self.config.compression_target_tokens).unwrap_or(u32::MAX);
        let system_prompt = self
            .config
            .custom_compression_prompt
            .as_deref()
            .unwrap_or(DEFAULT_COMPRESSION_PROMPT);
        let model = cx.model.or(self.model.as_ref());

        let mut changed: Vec<usize> = Vec::new();
        let mut summaries: Vec<String> = Vec::new();
        let mut stage_one: Option<Range<usize>> = None;

        if let Some(span) = self.span(messages, boundary) {
            let span_messages = messages[span.clone()].to_vec();
            let before = count_messages(cx.counter, &span_messages);
            if before >= self.config.min_selected_tokens_for_compression {
                let transcript = summarize_offline(&span_messages, max_chars);
                if cx.cancellation.is_cancelled() {
                    return Err(Error::Cancelled);
                }
                let summary = summarize(model, system_prompt, &transcript, max_tokens, || {
                    transcript.clone()
                })
                .await
                .trim()
                .to_owned();
                let marker = archive_replaced_messages(cx, &span_messages).await;
                let mut block = MemoryBlockKind::CurrentRound.render(&summary);
                if let Some(marker) = &marker {
                    block = marker.append(block);
                }
                let replacement = BaseMessage::user(block);
                let after = count_message(cx.counter, &replacement);
                if after < before {
                    messages.splice(span.clone(), std::iter::once(replacement));
                    changed.extend(span.clone());
                    summaries.push(summary);
                    stage_one = Some(span);
                } else {
                    tracing::debug!(
                        before,
                        after,
                        "current-round compressor left the span in place: no compression benefit"
                    );
                    if let Some(marker) = &marker {
                        rollback_recall_archive(cx, marker).await;
                    }
                }
            }
        }

        if let Some(merged) = self.merge_blocks(messages, cx, &mut changed, stage_one.as_ref()) {
            summaries.push(merged);
        }

        if changed.is_empty() {
            return Ok(None);
        }
        changed.sort_unstable();
        changed.dedup();
        let tokens_after = count_messages(cx.counter, messages);
        Ok(Some(ContextEvent {
            processor: self.name().to_owned(),
            kind: ContextEventKind::Summarized,
            messages_changed: changed,
            summary: Some(summaries.join("\n\n")),
            offload_handle: None,
            tokens_before,
            tokens_after,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ContextConfig;
    use crate::offload::{InMemoryOffloadStore, OffloadScope, OffloadStore};
    use openjiuwen_foundation::prompt::{ApproxTokenCounter, ToolCall};
    use tokio_util::sync::CancellationToken;

    fn history() -> Vec<BaseMessage> {
        vec![
            BaseMessage::user("q"),
            BaseMessage::assistant("").with_tool_calls(vec![ToolCall::new("t").with_id("c1")]),
            BaseMessage::tool("c1", "result"),
            BaseMessage::assistant("halfway"),
            BaseMessage::assistant("done"),
        ]
    }

    struct Fixture {
        config: ContextConfig,
        counter: ApproxTokenCounter,
        store: InMemoryOffloadStore,
        scope: OffloadScope,
        cancellation: CancellationToken,
    }

    impl Fixture {
        fn new() -> Self {
            Self {
                config: ContextConfig::default(),
                counter: ApproxTokenCounter::with_chars_per_token(1),
                store: InMemoryOffloadStore::new(),
                scope: OffloadScope::new("s", "main"),
                cancellation: CancellationToken::new(),
            }
        }

        fn context(&self, trigger: usize, manual: bool) -> ProcessorContext<'_> {
            ProcessorContext {
                config: &self.config,
                counter: &self.counter,
                model: None,
                offload: &self.store as &dyn OffloadStore,
                scope: &self.scope,
                budget: 100_000,
                trigger_tokens: trigger,
                manual,
                cancellation: &self.cancellation,
                recall: None,
                session_memory: None,
            }
        }
    }

    #[test]
    fn default_config_is_valid_and_round_trips() {
        let config = CurrentRoundCompressorConfig::default();
        assert!(config.validate().is_ok());
        let json = serde_json::to_string(&config).unwrap();
        assert_eq!(
            serde_json::from_str::<CurrentRoundCompressorConfig>(&json).unwrap(),
            config
        );
        assert_eq!(
            serde_json::from_str::<CurrentRoundCompressorConfig>("{}").unwrap(),
            config
        );
    }

    #[test]
    fn validation_rejects_a_single_block_merge() {
        let config = CurrentRoundCompressorConfig {
            summary_merge_min_blocks: 1,
            ..CurrentRoundCompressorConfig::default()
        };
        let error = config.validate().unwrap_err();
        assert!(matches!(error, Error::Config(_)));
        assert!(error.to_string().contains("summary_merge_min_blocks"));
    }

    #[test]
    fn span_stops_before_the_raw_tail_and_round_boundaries() {
        let compressor = CurrentRoundCompressor::new(
            CurrentRoundCompressorConfig {
                messages_to_keep: 1,
                ..CurrentRoundCompressorConfig::default()
            },
            None,
        );
        assert_eq!(compressor.span(&history(), 0), Some(1..4));

        // A span that ends inside the tool exchange has no completed round.
        let cut = &history()[..3];
        assert_eq!(compressor.span(cut, 0), None);

        // No room between the boundary and the tail.
        let last_user = vec![BaseMessage::user("q"), BaseMessage::assistant("a")];
        assert_eq!(compressor.span(&last_user, 0), None);
    }

    #[test]
    fn original_indices_are_restored_after_stage_one() {
        assert_eq!(CurrentRoundCompressor::map_to_original(0, None), 0);
        assert_eq!(CurrentRoundCompressor::map_to_original(2, Some(&(2..6))), 2);
        assert_eq!(CurrentRoundCompressor::map_to_original(3, Some(&(2..6))), 6);
    }

    #[tokio::test]
    async fn a_history_without_a_user_message_is_left_alone() {
        let fixture = Fixture::new();
        let compressor = CurrentRoundCompressor::new(CurrentRoundCompressorConfig::default(), None);
        let mut messages = vec![BaseMessage::assistant("a"), BaseMessage::assistant("b")];
        let cx = fixture.context(100_000, true);
        assert!(compressor
            .process(&mut messages, &cx)
            .await
            .unwrap()
            .is_none());
        assert_eq!(messages.len(), 2);
    }
}
