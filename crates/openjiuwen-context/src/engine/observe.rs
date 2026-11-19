//! Appending messages: `observe` and `observe_tool_result`.
//!
//! `observe` appends one message, applies the rolling history limit, and — when
//! a recorder is configured — runs the passive processor chain in the
//! `add_messages` phase for telemetry only, discarding the transformed copy.
//!
//! Both steps happen on the context's own vector
//! ([`ConversationContext::append_message`]), so an append neither clones the
//! history nor re-tokenizes it: the token total rides the context's statistics
//! cache, which the append updates by the new message's count.
//!
//! [`ConversationContext::append_message`]: crate::context::ConversationContext
use std::sync::Arc;

use openjiuwen_common::error::Result;
use openjiuwen_foundation::prompt::BaseMessage;

use crate::context::ConversationContext;
use crate::events::ObserveOutcome;
use crate::telemetry::CompressionStatePhase;

use super::ContextEngine;

impl ContextEngine {
    /// Appends one message and applies the rolling message limit.
    ///
    /// Eviction removes the oldest non-system messages, never the message just
    /// appended and never a system message, until the history fits
    /// [`ContextConfig::max_context_messages`](crate::config::ContextConfig::max_context_messages).
    ///
    /// The append mutates the stored history in place and reuses the cached
    /// statistics for its token total, so observing `n` messages costs `O(n)`
    /// rather than a full clone and a full recount per message.
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
    /// when the message is empty for a role that requires a payload.
    pub async fn observe(
        &self,
        context: &Arc<ConversationContext>,
        message: BaseMessage,
    ) -> Result<ObserveOutcome> {
        let telemetry_batch = context.has_recorder().then(|| vec![message.clone()]);
        let appended = context
            .append_message(message, context.config().max_context_messages)
            .await?;
        if let Some(mut batch) = telemetry_batch {
            if let Err(error) = context
                .run_pipeline(&mut batch, false, CompressionStatePhase::AddMessages)
                .await
            {
                tracing::debug!(%error, "context observe telemetry pass failed");
            }
        }
        Ok(ObserveOutcome {
            added: 1,
            evicted: appended.evicted,
            total_messages: appended.total_messages,
            total_tokens: appended.total_tokens,
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
