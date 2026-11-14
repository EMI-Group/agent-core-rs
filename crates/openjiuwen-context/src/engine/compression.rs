//! Compression passes and provider-overflow recovery.
//!
//! `compress` runs the pipeline over the whole history and persists the report
//! it produces; `recover_from_model_exception` calls it only for an
//! input-context overflow that emitted no visible output yet.
use std::sync::Arc;

use openjiuwen_common::error::Result;

use crate::context::{ConversationContext, ModelContext};
use crate::events::{CompressReport, CompressionStatus, RecoveryOutcome};
use crate::telemetry::CompressionStatePhase;
use crate::tokens::count_messages;

use super::ContextEngine;

impl ContextEngine {
    /// Runs the pipeline over the whole history and persists the result.
    ///
    /// The report says [`CompressionStatus::Busy`] when another pass already
    /// holds the context's compression lock, [`CompressionStatus::Noop`] when no
    /// processor changed anything, and [`CompressionStatus::Compressed`] when
    /// the rewritten history was stored.
    ///
    /// # Errors
    ///
    /// Returns the pipeline's error (for example
    /// [`Error::Llm`](openjiuwen_common::error::Error::Llm) when summarisation
    /// fails, or [`Error::Cancelled`](openjiuwen_common::error::Error::Cancelled)
    /// when the context was cancelled).
    pub async fn compress(&self, context: &Arc<ConversationContext>) -> Result<CompressReport> {
        let counter = context.token_counter();
        let Ok(lock) = context.compression_lock().try_lock() else {
            let messages = context.messages().await;
            let tokens = count_messages(counter.as_ref(), &messages);
            return Ok(CompressReport {
                status: CompressionStatus::Busy,
                events: Vec::new(),
                tokens_before: tokens,
                tokens_after: tokens,
                messages: messages.len(),
            });
        };

        let original = context.messages().await;
        let tokens_before = count_messages(counter.as_ref(), &original);
        let mut working = original.clone();
        let events = context
            .run_pipeline(&mut working, true, CompressionStatePhase::ActiveCompress)
            .await?;
        if events.is_empty() && working == original {
            drop(lock);
            return Ok(CompressReport {
                status: CompressionStatus::Noop,
                events,
                tokens_before,
                tokens_after: tokens_before,
                messages: original.len(),
            });
        }

        context.set_messages(working.clone()).await?;
        let tokens_after = count_messages(counter.as_ref(), &working);
        context
            .record_compression(tokens_before, tokens_after)
            .await;
        drop(lock);
        Ok(CompressReport {
            status: CompressionStatus::Compressed,
            events,
            tokens_before,
            tokens_after,
            messages: working.len(),
        })
    }

    /// Recovers from a provider error that a compression pass may fix.
    ///
    /// Recovery is attempted only when the error describes an input-context
    /// overflow (see [`ContextEngine::is_context_overflow_error`]) and no output
    /// was streamed yet: retrying after a visible prefix would repeat it.
    ///
    /// # Errors
    ///
    /// Returns every error of [`ContextEngine::compress`].
    pub async fn recover_from_model_exception(
        &self,
        context: &Arc<ConversationContext>,
        error: &str,
        streaming: bool,
        chunks_emitted: usize,
    ) -> Result<RecoveryOutcome> {
        if streaming && chunks_emitted > 0 {
            tracing::warn!(
                chunks_emitted,
                "skipping context recovery after streaming output was emitted"
            );
            return Ok(RecoveryOutcome::default());
        }
        if !Self::is_context_overflow_error(error) {
            return Ok(RecoveryOutcome::default());
        }
        let report = self.compress(context).await?;
        Ok(RecoveryOutcome {
            attempted: true,
            recovered: report.is_compressed(),
            report: Some(report),
        })
    }

    /// Whether `error` describes a model input that exceeded the context window.
    ///
    /// Matching is case-insensitive after `_` and `-` are folded to spaces, so
    /// provider codes such as `context_length_exceeded` match too; generic quota,
    /// authentication, and network errors stay unrecognized.
    #[must_use]
    pub fn is_context_overflow_error(error: &str) -> bool {
        /// Provider phrases that always mean an input-context overflow.
        const PHRASES: [&str; 9] = [
            "context length",
            "context window",
            "maximum context",
            "context limit",
            "prompt is too long",
            "prompt too long",
            "input is too long",
            "input too long",
            "range of input length",
        ];

        let message = error.to_lowercase().replace(['_', '-'], " ");
        if PHRASES.iter().any(|phrase| message.contains(phrase)) {
            return true;
        }
        let mentions_input = ["context", "prompt", "input"]
            .iter()
            .any(|term| message.contains(term));
        // An output/completion/response token limit is a generation budget, not
        // an input overflow, and must not trigger compression.
        let mentions_output = ["output", "completion", "response"]
            .iter()
            .any(|term| message.contains(term));
        if message.contains("token limit") && !mentions_output {
            return true;
        }
        if message.contains("too long") && mentions_input {
            return true;
        }
        (message.contains("exceed") || message.contains("over limit")) && mentions_input
    }
}
