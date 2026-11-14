//! Appending messages: `observe` and `observe_tool_result`.
//!
//! `observe` appends one message, applies the rolling history limit, and — when
//! a recorder is configured — runs the passive processor chain in the
//! `add_messages` phase for telemetry only, discarding the transformed copy.
use std::sync::Arc;

use openjiuwen_common::error::Result;
use openjiuwen_foundation::prompt::{BaseMessage, Role};

use crate::context::{ConversationContext, ModelContext};
use crate::events::ObserveOutcome;
use crate::telemetry::CompressionStatePhase;
use crate::tokens::count_messages;

use super::ContextEngine;

impl ContextEngine {
    /// Appends one message and applies the rolling message limit.
    ///
    /// Eviction removes the oldest non-system messages, never the message just
    /// appended and never a system message, until the history fits
    /// [`ContextConfig::max_context_messages`](crate::config::ContextConfig::max_context_messages).
    ///
    /// When a recorder is configured, the appended message also runs through
    /// the passive processor chain in the `add_messages` phase purely so the
    /// record describes what the processors would do to the incoming batch; the
    /// pass's result is discarded, and a telemetry failure is logged rather
    /// than propagated. Without a recorder the append path is unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`](openjiuwen_common::error::Error::Validation)
    /// when [`ConversationContext::add_message`] rejects the message.
    pub async fn observe(
        &self,
        context: &Arc<ConversationContext>,
        message: BaseMessage,
    ) -> Result<ObserveOutcome> {
        let telemetry_batch = context.has_recorder().then(|| vec![message.clone()]);
        context.add_message(message).await?;
        let mut messages = context.messages().await;
        let mut evicted = 0;
        if let Some(limit) = context.config().max_context_messages {
            while messages.len() > limit {
                // The newest message (the one just appended) is always last and
                // is never evicted.
                let candidate = messages
                    .iter()
                    .enumerate()
                    .take(messages.len().saturating_sub(1))
                    .find(|(_, candidate)| candidate.role != Role::System)
                    .map(|(index, _)| index);
                let Some(index) = candidate else {
                    break;
                };
                messages.remove(index);
                evicted += 1;
            }
            if evicted > 0 {
                context.set_messages(messages.clone()).await?;
            }
        }
        if let Some(mut batch) = telemetry_batch {
            if let Err(error) = context
                .run_pipeline(&mut batch, false, CompressionStatePhase::AddMessages)
                .await
            {
                tracing::debug!(%error, "context observe telemetry pass failed");
            }
        }
        let counter = context.token_counter();
        Ok(ObserveOutcome {
            added: 1,
            evicted,
            total_messages: messages.len(),
            total_tokens: count_messages(counter.as_ref(), &messages),
        })
    }

    /// Appends the result of the tool call `tool_call_id`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`](openjiuwen_common::error::Error::Validation)
    /// when the tool result is empty.
    #[allow(clippy::similar_names)]
    pub async fn observe_tool_result(
        &self,
        context: &Arc<ConversationContext>,
        tool_call_id: &str,
        content: impl Into<String>,
    ) -> Result<ObserveOutcome> {
        self.observe(context, BaseMessage::tool(tool_call_id, content.into()))
            .await
    }
}
