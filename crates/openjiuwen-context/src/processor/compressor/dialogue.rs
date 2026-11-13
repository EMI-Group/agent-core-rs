//! Dialogue compressor: replace old complete rounds with memory blocks.
//!
//! The compressor walks the completed rounds of a conversation — a user turn,
//! the assistant and tool messages that answer it, and the assistant turn that
//! closes it — and replaces the body of the older ones with a single
//! [`DIALOGUE_MEMORY_BLOCK`](crate::processor::compressor::DIALOGUE_MEMORY_BLOCK)
//! user message.
//!
//! Two rules shape what it touches:
//!
//! - The user message that anchors a round always survives raw, so the model
//!   keeps seeing the original request that the block summarises; only the
//!   answering exchange is replaced.
//! - Only whole completed rounds inside a boundary are eligible, and the
//!   replacement must be smaller than what it replaces. Tool-call/tool-result
//!   pairing and system messages are therefore preserved by construction.
//!
//! Summaries come from the configured model when there is one and from the
//! deterministic offline transcript summary otherwise; a failed model call falls
//! back to that summary rather than failing the pass.
//!
//! [`DIALOGUE_MEMORY_BLOCK`]: crate::processor::compressor::DIALOGUE_MEMORY_BLOCK
use std::ops::Range;
use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::llm::Model;
use openjiuwen_foundation::prompt::{BaseMessage, Role};
use serde::{Deserialize, Serialize};

use crate::events::{ContextEvent, ContextEventKind};
use crate::processor::compressor::{
    archive_replaced_messages, group_completed_rounds, rollback_recall_archive, summarize,
    summarize_offline, MemoryBlockKind,
};
use crate::processor::impl_processor_state;
use crate::processor::{ContextProcessor, ProcessorContext, ProcessorState};
use crate::tokens::{count_message, count_messages};

/// Characters per token used to size the offline summary fallback.
const CHARS_PER_TOKEN_ESTIMATE: usize = 4;

/// Instruction used when no custom compression prompt is configured.
const DEFAULT_COMPRESSION_PROMPT: &str = "Summarise this conversation round into a compact \
reference-memory block. Preserve decisions, identifiers, file paths and open questions. \
Do not invent details.";

/// Configuration of [`DialogueCompressor`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct DialogueCompressorConfig {
    /// Trigger when the history holds more than this many messages; `None`
    /// disables the message-count trigger.
    pub messages_threshold: Option<usize>,
    /// Trigger when the history holds more than this many tokens.
    pub tokens_threshold: usize,
    /// Newest messages that are never compressed.
    pub messages_to_keep: Option<usize>,
    /// Whether the newest completed round is protected.
    pub keep_last_round: bool,
    /// Token budget each compressed block should stay within.
    pub compression_target_tokens: usize,
    /// Instruction replacing the built-in summary prompt when set.
    pub custom_compression_prompt: Option<String>,
}

impl Default for DialogueCompressorConfig {
    fn default() -> Self {
        Self {
            messages_threshold: None,
            tokens_threshold: 10_000,
            messages_to_keep: None,
            keep_last_round: true,
            compression_target_tokens: 1_800,
            custom_compression_prompt: None,
        }
    }
}

impl DialogueCompressorConfig {
    /// Checks the thresholds and the keep-window against each other.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] for a zero threshold or target, and when
    /// [`DialogueCompressorConfig::messages_to_keep`] is not smaller than
    /// [`DialogueCompressorConfig::messages_threshold`] while both are set.
    pub fn validate(&self) -> Result<()> {
        if self.tokens_threshold == 0 {
            return Err(Error::Config(
                "dialogue compressor: tokens_threshold must be greater than zero".to_owned(),
            ));
        }
        if self.messages_threshold == Some(0) {
            return Err(Error::Config(
                "dialogue compressor: messages_threshold must be greater than zero".to_owned(),
            ));
        }
        if self.messages_to_keep == Some(0) {
            return Err(Error::Config(
                "dialogue compressor: messages_to_keep must be greater than zero".to_owned(),
            ));
        }
        if self.compression_target_tokens == 0 {
            return Err(Error::Config(
                "dialogue compressor: compression_target_tokens must be greater than zero"
                    .to_owned(),
            ));
        }
        if let (Some(keep), Some(threshold)) = (self.messages_to_keep, self.messages_threshold) {
            if keep >= threshold {
                return Err(Error::Config(format!(
                    "dialogue compressor: messages_to_keep ({keep}) must be smaller than \
                     messages_threshold ({threshold})"
                )));
            }
        }
        Ok(())
    }
}

/// Replaces older completed rounds with historical memory blocks.
#[derive(Clone)]
pub struct DialogueCompressor {
    /// Processor configuration.
    pub config: DialogueCompressorConfig,
    /// Model used when the request does not supply one.
    pub model: Option<Arc<dyn Model>>,
    /// Cross-run counters, persisted through the context checkpoint.
    pub state: ProcessorState,
}

impl std::fmt::Debug for DialogueCompressor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DialogueCompressor")
            .field("config", &self.config)
            .field("has_model", &self.model.is_some())
            .finish_non_exhaustive()
    }
}

impl DialogueCompressor {
    /// Creates a compressor with `config` and an optional fallback `model`.
    #[must_use]
    pub fn new(config: DialogueCompressorConfig, model: Option<Arc<dyn Model>>) -> Self {
        Self {
            config,
            model,
            state: ProcessorState::default(),
        }
    }

    /// Whether the thresholds ask for a pass.
    fn should_run(&self, messages: &[BaseMessage], cx: &ProcessorContext<'_>) -> bool {
        if cx.manual {
            return true;
        }
        if self
            .config
            .messages_threshold
            .is_some_and(|threshold| messages.len() > threshold)
        {
            return true;
        }
        if self
            .config
            .messages_to_keep
            .is_some_and(|keep| messages.len() < keep)
        {
            return false;
        }
        count_messages(cx.counter, messages) > self.config.tokens_threshold
    }

    /// The exclusive index beyond which no message may be compressed.
    ///
    /// With [`DialogueCompressorConfig::keep_last_round`] the region from the
    /// newest assistant message that closes a round onwards is protected, so the
    /// newest completed round is never rewritten. Returns `None` when the newest
    /// round should be protected but no assistant message closes a round yet, in
    /// which case nothing may be compressed.
    fn boundary(&self, messages: &[BaseMessage]) -> Option<usize> {
        let mut boundary = self
            .config
            .messages_to_keep
            .map_or(messages.len(), |keep| messages.len().saturating_sub(keep));
        if self.config.keep_last_round {
            let newest_close = messages.iter().rposition(|message| {
                message.role == Role::Assistant && message.tool_calls.is_empty()
            })?;
            boundary = boundary.min(newest_close);
        }
        Some(boundary)
    }

    /// Spans to replace, oldest first: each round's answering exchange without
    /// its anchor user message.
    fn targets(messages: &[BaseMessage], boundary: usize) -> Vec<Range<usize>> {
        group_completed_rounds(messages)
            .into_iter()
            .filter(|round| round.end <= boundary && round.len() > 2)
            .map(|round| round.start + 1..round.end)
            .collect()
    }
}

#[async_trait]
impl ContextProcessor for DialogueCompressor {
    fn name(&self) -> &'static str {
        "dialogue"
    }

    impl_processor_state!(DialogueCompressor);

    async fn process(
        &self,
        messages: &mut Vec<BaseMessage>,
        cx: &ProcessorContext<'_>,
    ) -> Result<Option<ContextEvent>> {
        if !self.should_run(messages, cx) {
            return Ok(None);
        }
        let Some(boundary) = self.boundary(messages) else {
            return Ok(None);
        };
        let targets = Self::targets(messages, boundary);
        if targets.is_empty() {
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

        // Newest first: replacing a span never shifts the indices of the older
        // spans still to be processed.
        let mut changed: Vec<usize> = Vec::new();
        let mut applied = 0u64;
        let mut last_block: Option<String> = None;
        for span in targets.into_iter().rev() {
            let span_messages = messages[span.clone()].to_vec();
            let transcript = summarize_offline(&span_messages, max_chars);
            if cx.cancellation.is_cancelled() {
                return Err(Error::Cancelled);
            }
            let summary = summarize(model, system_prompt, &transcript, max_tokens, || {
                transcript.clone()
            })
            .await;

            // The archive keeps the exact exchanged messages; its marker is part
            // of the block, so it is rendered before the benefit gate measures it.
            let marker = archive_replaced_messages(cx, &span_messages).await;
            let before = count_messages(cx.counter, &span_messages);
            let mut block = MemoryBlockKind::Dialogue.render(&summary);
            if let Some(marker) = &marker {
                block = marker.append(block);
            }
            let replacement = BaseMessage::user(block);
            let after = count_message(cx.counter, &replacement);
            if before == 0 || after >= before {
                tracing::debug!(
                    before,
                    after,
                    "dialogue compressor left a round in place: no compression benefit"
                );
                if let Some(marker) = &marker {
                    rollback_recall_archive(cx, marker).await;
                }
                continue;
            }
            messages.splice(span.clone(), std::iter::once(replacement.clone()));
            changed.extend(span);
            applied += 1;
            last_block = Some(replacement.text().into_owned());
        }

        let Some(last_block) = last_block else {
            return Ok(None);
        };
        changed.sort_unstable();
        changed.dedup();
        let tokens_after = count_messages(cx.counter, messages);
        self.state.add("compressions", applied);
        self.state.add("rounds_dropped", changed.len() as u64);
        self.state
            .record_tokens("last_tokens_before", tokens_before);
        self.state.record_tokens("last_tokens_after", tokens_after);
        Ok(Some(ContextEvent {
            processor: self.name().to_owned(),
            kind: ContextEventKind::Summarized,
            messages_changed: changed,
            summary: Some(last_block),
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
    use openjiuwen_foundation::prompt::ApproxTokenCounter;
    use tokio_util::sync::CancellationToken;

    fn history() -> Vec<BaseMessage> {
        vec![
            BaseMessage::system("persona"),
            BaseMessage::user("q1"),
            BaseMessage::assistant("a1"),
            BaseMessage::user("q2"),
            BaseMessage::assistant("a2"),
            BaseMessage::user("q3"),
            BaseMessage::assistant("a3"),
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
        let config = DialogueCompressorConfig::default();
        assert!(config.validate().is_ok());
        let json = serde_json::to_string(&config).unwrap();
        assert_eq!(
            serde_json::from_str::<DialogueCompressorConfig>(&json).unwrap(),
            config
        );
        assert_eq!(
            serde_json::from_str::<DialogueCompressorConfig>("{}").unwrap(),
            config
        );
    }

    #[test]
    fn validation_rejects_zero_and_inverted_thresholds() {
        let zero_tokens = DialogueCompressorConfig {
            tokens_threshold: 0,
            ..DialogueCompressorConfig::default()
        };
        assert!(zero_tokens.validate().is_err());

        let inverted = DialogueCompressorConfig {
            messages_threshold: Some(4),
            messages_to_keep: Some(4),
            ..DialogueCompressorConfig::default()
        };
        let error = inverted.validate().unwrap_err();
        assert!(matches!(error, Error::Config(_)));
    }

    #[test]
    fn boundary_protects_the_newest_completed_round() {
        let compressor = DialogueCompressor::new(DialogueCompressorConfig::default(), None);
        let open = DialogueCompressor::new(
            DialogueCompressorConfig {
                keep_last_round: false,
                ..DialogueCompressorConfig::default()
            },
            None,
        );
        // A trailing user turn starts a round that never completes; the newest
        // *completed* round is the protected one.
        let mut trailing = history();
        trailing.push(BaseMessage::user("q4"));
        assert_eq!(compressor.boundary(&trailing), Some(6));
        assert_eq!(open.boundary(&trailing), Some(8));

        let without_runs = vec![BaseMessage::user("q"), BaseMessage::assistant("a")];
        let keep = DialogueCompressor::new(
            DialogueCompressorConfig {
                messages_to_keep: Some(1),
                ..DialogueCompressorConfig::default()
            },
            None,
        );
        assert_eq!(keep.boundary(&without_runs), Some(1));
    }

    #[tokio::test]
    async fn short_rounds_are_not_compressed() {
        let fixture = Fixture::new();
        let compressor = DialogueCompressor::new(DialogueCompressorConfig::default(), None);
        let mut messages = vec![
            BaseMessage::user("q1"),
            BaseMessage::assistant("a1"),
            BaseMessage::user("q2"),
            BaseMessage::assistant("a2"),
        ];
        let cx = fixture.context(100_000, true);
        assert!(compressor
            .process(&mut messages, &cx)
            .await
            .unwrap()
            .is_none());
        assert_eq!(messages.len(), 4);
    }
}
