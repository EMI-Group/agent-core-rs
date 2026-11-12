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
use crate::output::{AgentChunk, LLM_OUTPUT, LLM_REASONING, LLM_USAGE};
use futures::StreamExt;
use openjiuwen_common::error::Result;
use openjiuwen_foundation::llm::{FinishReason, Model, ModelRequestConfig};
use openjiuwen_foundation::{BaseMessage, StreamAccumulator, StreamChunk, Usage};
use openjiuwen_session::Session;
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::mpsc::UnboundedSender;
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

/// Destination of the streaming frames a model call emits.
///
/// Frames go to the invocation's chunk publisher (when the caller is streaming)
/// and are mirrored into the session (when the agent runs session-scoped). The
/// sink owns the frame index sequence so every emitted frame carries a stable,
/// increasing position within the turn.
pub(crate) struct DeltaSink<'a> {
    publisher: Option<&'a UnboundedSender<AgentChunk>>,
    session: Option<&'a Arc<dyn Session>>,
    index: i64,
}

impl<'a> DeltaSink<'a> {
    /// Creates a sink that starts at index zero.
    #[allow(dead_code)] // the loop that builds a sink lands in a later round
    pub(crate) fn new(
        publisher: Option<&'a UnboundedSender<AgentChunk>>,
        session: Option<&'a Arc<dyn Session>>,
    ) -> Self {
        Self {
            publisher,
            session,
            index: 0,
        }
    }

    /// Returns the current frame index and advances past it.
    pub(crate) fn next_index(&mut self) -> i64 {
        let index = self.index;
        self.index += 1;
        index
    }

    /// Emits one frame of `kind` with `payload`.
    ///
    /// Delivery failures never abort the model call: a closed publisher channel
    /// is ignored and a session write error is logged only.
    pub(crate) async fn emit(&mut self, kind: &str, payload: Value) {
        let chunk = AgentChunk::new(kind, self.next_index(), payload);
        if let Some(publisher) = self.publisher {
            // The receiver may have gone away once the caller stopped reading;
            // that must not fail the model call.
            let _ = publisher.send(chunk.clone());
        }
        if let Some(session) = self.session {
            if let Err(error) = session.write_stream(chunk.to_stream_item()).await {
                tracing::warn!(
                    %error,
                    kind = %chunk.kind,
                    "failed to write stream frame to the session"
                );
            }
        }
    }
}

/// Calls `model` once in streaming mode and normalizes the accumulated reply.
///
/// Every content delta becomes an [`LLM_OUTPUT`] frame, every reasoning delta an
/// [`LLM_REASONING`] frame, and the accumulated usage exactly one [`LLM_USAGE`]
/// frame at the end of the stream (even when the provider reported no usage).
/// Provider errors propagate unchanged.
///
/// # Errors
///
/// Returns the provider's error from the stream unchanged.
#[allow(dead_code)] // the loop that calls this lands in a later round
pub(crate) async fn call_model_streaming(
    model: &Arc<dyn Model>,
    messages: &[BaseMessage],
    mut request: ModelRequestConfig,
    sink: &mut DeltaSink<'_>,
) -> Result<ModelReply> {
    request.stream = true;
    let mut stream = model.stream(messages.to_vec(), request);
    let mut accumulator = StreamAccumulator::new();
    while let Some(item) = stream.next().await {
        let chunk = item?;
        accumulator.push(&chunk);
        if let StreamChunk::Delta {
            content: Some(text),
            ..
        } = &chunk
        {
            sink.emit(
                LLM_OUTPUT,
                serde_json::json!({"content": text, "result_type": "answer"}),
            )
            .await;
        }
        if let StreamChunk::Delta {
            reasoning: Some(text),
            ..
        } = &chunk
        {
            sink.emit(
                LLM_REASONING,
                serde_json::json!({"content": text, "result_type": "answer"}),
            )
            .await;
        }
    }

    let turn = accumulator.finish();
    let usage_metadata = serde_json::to_value(&turn.usage).unwrap_or(Value::Null);
    sink.emit(
        LLM_USAGE,
        serde_json::json!({"usage_metadata": usage_metadata, "result_type": "answer"}),
    )
    .await;
    let reasoning_present = turn.message.reasoning_content.is_some();
    Ok(ModelReply {
        message: turn.message,
        finish_reason: turn.finish_reason,
        usage: turn.usage,
        reasoning_present,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use openjiuwen_test_support::{MockCompletion, ScriptedError, ScriptedModel, ScriptedTurn};

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

    fn drain(receiver: &mut tokio::sync::mpsc::UnboundedReceiver<AgentChunk>) -> Vec<AgentChunk> {
        let mut chunks = Vec::new();
        while let Ok(chunk) = receiver.try_recv() {
            chunks.push(chunk);
        }
        chunks
    }

    fn content_of(chunks: &[AgentChunk], kind: &str) -> String {
        chunks
            .iter()
            .filter(|chunk| chunk.kind == kind)
            .filter_map(|chunk| chunk.payload.get("content").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("")
    }

    #[tokio::test]
    async fn streaming_emits_content_reasoning_and_one_usage_frame() {
        let model: Arc<dyn Model> = Arc::new(
            ScriptedModel::builder()
                .stream_text("hello")
                .completion(MockCompletion::reasoning("think", "answer"))
                .build(),
        );
        let (publisher, mut receiver) = tokio::sync::mpsc::unbounded_channel();
        let mut sink = DeltaSink::new(Some(&publisher), None);
        let messages = vec![BaseMessage::user("hi")];

        let plain =
            call_model_streaming(&model, &messages, ModelRequestConfig::default(), &mut sink)
                .await
                .unwrap();
        assert_eq!(plain.message.content.as_text(), Some("hello"));
        assert_eq!(plain.finish_reason, Some(FinishReason::Stop));
        assert!(!plain.reasoning_present);
        let chunks = drain(&mut receiver);
        assert_eq!(content_of(&chunks, LLM_OUTPUT), "hello");
        assert_eq!(content_of(&chunks, LLM_REASONING), "");
        let usage_frames = chunks
            .iter()
            .filter(|chunk| chunk.kind == LLM_USAGE)
            .collect::<Vec<_>>();
        assert_eq!(usage_frames.len(), 1);
        assert_eq!(usage_frames[0].payload["result_type"], "answer");
        assert!(usage_frames[0].payload["usage_metadata"].is_object());
        assert_eq!(
            chunks.iter().map(|chunk| chunk.index).collect::<Vec<_>>(),
            vec![0, 1]
        );

        let reasoning =
            call_model_streaming(&model, &messages, ModelRequestConfig::default(), &mut sink)
                .await
                .unwrap();
        assert_eq!(reasoning.message.content.as_text(), Some("answer"));
        assert!(reasoning.reasoning_present);
        assert_eq!(
            reasoning.message.reasoning_content.as_deref(),
            Some("think")
        );
        let chunks = drain(&mut receiver);
        assert_eq!(content_of(&chunks, LLM_OUTPUT), "answer");
        assert_eq!(content_of(&chunks, LLM_REASONING), "think");
        assert_eq!(
            chunks
                .iter()
                .filter(|chunk| chunk.kind == LLM_USAGE)
                .count(),
            1
        );
        assert_eq!(
            chunks.iter().map(|chunk| chunk.index).collect::<Vec<_>>(),
            vec![2, 3, 4]
        );
    }

    #[tokio::test]
    async fn streaming_sets_the_request_stream_flag() {
        let model = ScriptedModel::builder().stream_text("hi").build();
        let shared: Arc<dyn Model> = Arc::new(model.clone());
        let mut sink = DeltaSink::new(None, None);
        call_model_streaming(&shared, &[], ModelRequestConfig::default(), &mut sink)
            .await
            .unwrap();
        let call = model.last_call().expect("stream call recorded");
        assert!(call.request.stream);
    }
}
