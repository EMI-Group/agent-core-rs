//! Preparing the payload a model call sends: pipeline pass, window assembly, hint.
//!
//! `prepare` runs on a copy of the stored history, so it is read-only; the
//! optional deadline wraps the whole body and maps an expiry to
//! [`Error::Timeout`](openjiuwen_common::error::Error::Timeout).
use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::prompt::{BaseMessage, Role};

use crate::engine::PrepareOptions;
use crate::events::ContextEvent;
use crate::policy::ContextWindowSnapshot;
use crate::processor::ProcessorContext;
use crate::reload::{inject_reload_hint, ReloadLanguage};
use crate::stats::{compute_stats, PreparedContext};
use crate::telemetry::CompressionStatePhase;
use crate::tokens::count_messages;
use crate::window::{assemble_window, drop_orphan_tool_results, WindowOptions};

use super::{ConversationContext, ModelContext};

impl ConversationContext {
    /// Runs the processor pipeline over `messages` in `phase`.
    ///
    /// `manual` bypasses the threshold check, which is what an explicit
    /// compression asks for; `phase` is the context-engine entry point that
    /// reached the pipeline, so recorded passes carry `add_messages`,
    /// `get_context_window`, or `active_compress`.
    pub(crate) async fn run_pipeline(
        &self,
        messages: &mut Vec<BaseMessage>,
        manual: bool,
        phase: CompressionStatePhase,
    ) -> Result<Vec<ContextEvent>> {
        let cx = ProcessorContext {
            config: &self.config,
            counter: self.counter.as_ref(),
            model: self.model.as_ref(),
            offload: self.offload.as_ref(),
            scope: &self.scope,
            budget: self.config.effective_max_tokens(),
            trigger_tokens: self.config.compression_trigger_tokens(),
            manual,
            cancellation: &self.cancellation,
            recall: self.recall.as_deref(),
            session_memory: self.session_memory.as_deref(),
            // The prepare pass holds no agent: extraction belongs to the
            // compaction pass, which is the only one that replaces history.
            session_memory_agent: None,
        };
        self.pipeline.run_in_phase(messages, &cx, phase).await
    }

    /// Builds the payload to send to the model.
    ///
    /// The pipeline runs on a copy of the history, so the stored messages are
    /// never mutated: [`ContextEngine::prepare`] is a read-only operation. The
    /// window is assembled after the processors ran, and orphan tool results left
    /// behind by round selection are dropped last.
    ///
    /// When [`PrepareOptions::timeout`] is set, the whole body is wrapped in
    /// [`tokio::time::timeout`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::Cancelled`] when the caller's token or the context's own
    /// token is set, [`Error::Timeout`] when the deadline expires, and the
    /// pipeline's error (for example [`Error::Llm`]) when a processor fails.
    ///
    /// [`ContextEngine::prepare`]: crate::engine::ContextEngine::prepare
    pub async fn prepare(&self, options: &PrepareOptions) -> Result<PreparedContext> {
        match options.timeout {
            Some(timeout) => tokio::time::timeout(timeout, self.prepare_inner(options))
                .await
                .map_err(|_| {
                    Error::Timeout(format!("context preparation timed out after {timeout:?}"))
                })?,
            None => self.prepare_inner(options).await,
        }
    }

    /// The body of [`ConversationContext::prepare`], without the deadline.
    async fn prepare_inner(&self, options: &PrepareOptions) -> Result<PreparedContext> {
        if options.cancellation.is_cancelled() || self.cancellation.is_cancelled() {
            return Err(Error::Cancelled);
        }
        let mut working = self.messages().await;
        let events = self
            .run_pipeline(&mut working, false, CompressionStatePhase::GetContextWindow)
            .await?;
        let mut messages = assemble_window(
            &working,
            &options.system_messages,
            &WindowOptions {
                window_size: options.window_size,
                dialogue_rounds: options.dialogue_rounds,
            },
            &self.config,
        );
        drop_orphan_tool_results(&mut messages);
        if self.config.enable_reload {
            let injected = inject_reload_hint(&mut messages, ReloadLanguage::En);
            tracing::debug!(
                injected,
                "reload hint injection for a prepared context window"
            );
        }
        let stats = compute_stats(&messages, self.counter.as_ref());
        let token_count = count_messages(self.counter.as_ref(), &messages);
        // The LLM-bound window splits at the leading system messages; the split
        // feeds both the usage snapshot and the KV-cache diff.
        let system_count = messages
            .iter()
            .take_while(|message| message.role == Role::System)
            .count();
        let window = ContextWindowSnapshot::from_messages(
            &messages[..system_count],
            &messages[system_count..],
            &options.tools,
        );
        let usage = self.usage_snapshot_for(&window, &stats);
        self.track_window(window);
        Ok(PreparedContext {
            messages,
            tools: options.tools.clone(),
            stats,
            token_count,
            overflow: token_count > self.config.effective_max_tokens(),
            window_size: options.window_size.or(self.config.default_window_messages),
            events,
            usage,
        })
    }
}
