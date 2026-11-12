//! One model call of the `ReAct` loop: request preparation, the `invoke`
//! versus streaming choice, and the reply the loop (and its rails) consume.
//!
//! Python parity: the model call inside
//! `openjiuwen/core/single_agent/agents/react_agent.py::_inner_invoke`.
//!
//! This round implements the non-streaming baseline: every call goes through
//! `Model::invoke` and is projected to a
//! [`ModelReply`]. Streaming (driving
//! `Model::stream` through a `StreamAccumulator` and emitting
//! `llm_reasoning`/`llm_output`/`llm_usage` chunks) arrives in a later round and
//! may replace the signature of `call_model` freely; the reply type is the
//! stable shape both paths must produce.
use openjiuwen_common::error::Result;
use openjiuwen_foundation::llm::{FinishReason, Model, ModelRequestConfig};
use openjiuwen_foundation::{BaseMessage, Usage};
use std::sync::Arc;
/// The normalized result of one model call.
///
/// It is the value rails observe on `AfterModelCall` (`RailContext::response`)
/// and the one the loop turns into an assistant message plus tool calls.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ModelReply {
    /// The assistant message, with tool calls when the model requested any.
    pub message: BaseMessage,
    /// Provider finish reason, when reported.
    pub finish_reason: Option<FinishReason>,
    /// Token accounting for the call.
    pub usage: Usage,
    /// Whether the model returned reasoning/thinking content.
    pub reasoning_present: bool,
}
/// Calls `model` once with `messages` and `request` and normalizes the reply.
///
/// An empty response (no choice at all) yields a default [`ModelReply`] rather
/// than an error, so the loop can apply its own empty-response policy.
///
/// TODO: stream when a session/publisher is attached — drive `Model::stream`
/// through a `StreamAccumulator`, emit `llm_reasoning`/`llm_output`/
/// `llm_usage` chunks while consuming the stream, and build the same
/// [`ModelReply`] from the accumulated turn.
///
/// # Errors
///
/// Returns the provider's error from `Model::invoke` unchanged.
#[allow(dead_code)] // the loop that calls this lands in a later round
pub(crate) async fn call_model(
    model: &Arc<dyn Model>,
    messages: &[BaseMessage],
    request: ModelRequestConfig,
) -> Result<ModelReply> {
    let response = model.invoke(messages, &request).await?;
    let message = response.message().cloned().unwrap_or_default();
    let finish_reason = response.finish_reason().cloned();
    let usage = response.usage.clone();
    let reasoning_present = message.reasoning_content.is_some();
    Ok(ModelReply {
        message,
        finish_reason,
        usage,
        reasoning_present,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use openjiuwen_test_support::{ScriptedError, ScriptedModel, ScriptedTurn};

    #[tokio::test]
    async fn text_reply_carries_content_and_finish_reason() {
        let model: Arc<dyn Model> = Arc::new(ScriptedModel::text("hello there"));
        let messages = vec![BaseMessage::user("hi")];
        let reply = call_model(&model, &messages, ModelRequestConfig::default())
            .await
            .unwrap();
        assert_eq!(reply.message.content.as_text(), Some("hello there"));
        assert_eq!(reply.finish_reason, Some(FinishReason::Stop));
        assert!(!reply.reasoning_present);
        assert_eq!(reply.usage, Usage::default());
        assert!(reply.message.reasoning_content.is_none());
    }

    #[tokio::test]
    async fn model_failure_propagates() {
        let model: Arc<dyn Model> = Arc::new(ScriptedModel::new([ScriptedTurn::fail_with(
            ScriptedError::llm("boom"),
        )]));
        let error = call_model(&model, &[], ModelRequestConfig::default())
            .await
            .unwrap_err();
        assert!(error.to_string().contains("boom"));
    }
}
