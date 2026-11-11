//! Provider-neutral model invocation: the [`Model`] trait and its data types.
//!
//! Ports `openjiuwen/core/foundation/llm/model.py` plus the response/stream
//! schema of `.../schema/`. Every provider client in this module tree implements
//! [`Model`]; callers above the foundation layer depend only on this trait, on
//! [`ChatResponse`] for non-streaming calls, and on [`StreamChunk`] for streaming
//! calls. [`StreamAccumulator`] turns a chunk stream into one assistant
//! [`BaseMessage`], and [`ModelExt::complete`] drives a stream to a complete
//! [`AssistantTurn`] so callers never have to re-implement the loop.
//!
//! ```
//! use openjiuwen_foundation::llm::model::{FinishReason, StreamAccumulator, StreamChunk};
//!
//! let mut accumulator = StreamAccumulator::new();
//! accumulator.push(&StreamChunk::Delta {
//!     index: 0,
//!     role: None,
//!     content: Some("hello".to_owned()),
//!     reasoning: None,
//!     tool_call: None,
//!     finish_reason: Some(FinishReason::Stop),
//! });
//! accumulator.push(&StreamChunk::Done);
//!
//! let turn = accumulator.finish();
//! assert_eq!(turn.message.text(), "hello");
//! assert_eq!(turn.finish_reason, Some(FinishReason::Stop));
//! ```

use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::BoxStream;
use futures::StreamExt;
use serde::de::{self, Deserializer};
use serde::{Deserialize, Serialize, Serializer};
use serde_json::Value;

use openjiuwen_common::error::Result;

use crate::prompt::{BaseMessage, MessageContent, Role, ToolCallAccumulator, ToolCallDelta, Usage};

use super::config::ModelRequestConfig;

/// Why the model finished generating.
///
/// Serializes as the wire string (`"stop"`, `"length"`, `"tool_calls"`,
/// `"content_filter"`) and deserializes any other string into
/// [`FinishReason::Other`], so vendor-specific reasons survive a round-trip.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FinishReason {
    /// The model reached a natural stopping point.
    Stop,
    /// The generated-token limit was reached.
    Length,
    /// The model requested one or more tool calls.
    ToolCalls,
    /// A provider content filter stopped generation.
    ContentFilter,
    /// Any reason this crate does not model; the payload is the wire string.
    Other(String),
}

impl FinishReason {
    /// The reason as a wire string ([`FinishReason::Other`] yields its payload).
    #[must_use]
    pub fn as_str(&self) -> &str {
        match self {
            Self::Stop => "stop",
            Self::Length => "length",
            Self::ToolCalls => "tool_calls",
            Self::ContentFilter => "content_filter",
            Self::Other(other) => other.as_str(),
        }
    }
}

impl From<&str> for FinishReason {
    fn from(value: &str) -> Self {
        match value {
            "stop" => Self::Stop,
            "length" => Self::Length,
            "tool_calls" => Self::ToolCalls,
            "content_filter" => Self::ContentFilter,
            other => Self::Other(other.to_owned()),
        }
    }
}

impl From<String> for FinishReason {
    fn from(value: String) -> Self {
        match value.as_str() {
            "stop" => Self::Stop,
            "length" => Self::Length,
            "tool_calls" => Self::ToolCalls,
            "content_filter" => Self::ContentFilter,
            _ => Self::Other(value),
        }
    }
}

impl Serialize for FinishReason {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for FinishReason {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        match value {
            Value::String(reason) => Ok(Self::from(reason)),
            other => Err(de::Error::custom(format!(
                "expected a finish reason string such as \"stop\", found {other}"
            ))),
        }
    }
}

/// One incremental piece of a streamed response.
///
/// Providers emit any number of [`StreamChunk::Delta`] items followed by
/// [`StreamChunk::Usage`] when the provider reports token counts, and the stream
/// always terminates with [`StreamChunk::Done`]. `finish_reason` is carried on
/// the delta that reports it, so a consumer does not need a second round-trip.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamChunk {
    /// A content, reasoning, or tool-call delta for one choice.
    Delta {
        /// Choice index the delta belongs to (always `0` for single-choice calls).
        index: u32,
        /// Role, usually reported only on the first delta.
        role: Option<Role>,
        /// Text appended to the message content.
        content: Option<String>,
        /// Reasoning/thinking text appended to the reasoning content.
        reasoning: Option<String>,
        /// Fragment of a tool call.
        tool_call: Option<ToolCallDelta>,
        /// Terminal reason, when the provider reports one.
        finish_reason: Option<FinishReason>,
    },
    /// Token accounting reported by the provider.
    Usage(Usage),
    /// The end of the stream.
    Done,
}

impl StreamChunk {
    /// The delta's content fragment, or `None` for non-delta chunks.
    #[must_use]
    pub fn content(&self) -> Option<&str> {
        match self {
            Self::Delta { content, .. } => content.as_deref(),
            Self::Usage(_) | Self::Done => None,
        }
    }
}

/// One choice of a non-streaming chat completion.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Choice {
    /// Position of the choice in the response.
    pub index: u32,
    /// The assistant message produced by the model.
    pub message: BaseMessage,
    /// Why generation stopped for this choice, when reported.
    pub finish_reason: Option<FinishReason>,
}

/// A complete (non-streaming) chat completion.
///
/// Mirrors the Python `ChatResponse`: the provider request id, the model that
/// answered, the choices, and the token usage.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ChatResponse {
    /// Provider request id, or an empty string when the provider sent none.
    pub id: String,
    /// Model name reported by the provider.
    pub model: String,
    /// One entry per returned choice.
    pub choices: Vec<Choice>,
    /// Token accounting for the call.
    pub usage: Usage,
}

impl ChatResponse {
    /// The first choice's message, or `None` when the response carries no choice.
    #[must_use]
    pub fn message(&self) -> Option<&BaseMessage> {
        self.choices.first().map(|choice| &choice.message)
    }

    /// The first choice's terminal reason, or `None` when absent.
    #[must_use]
    pub fn finish_reason(&self) -> Option<&FinishReason> {
        self.choices
            .first()
            .and_then(|choice| choice.finish_reason.as_ref())
    }
}

/// The accumulated result of one streamed model call.
///
/// [`ModelExt::complete`] returns this: the assistant message, the terminal
/// reason, and the usage reported by the provider.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct AssistantTurn {
    /// The assistant message built from the streamed fragments.
    pub message: BaseMessage,
    /// The last terminal reason the provider reported.
    pub finish_reason: Option<FinishReason>,
    /// The last non-empty token accounting the provider reported.
    pub usage: Usage,
}

/// Accumulates [`StreamChunk`] items into one [`AssistantTurn`].
///
/// Merge rules (Python `AssistantMessageChunk.__add__`):
///
/// - `content` and `reasoning` fragments concatenate in arrival order.
/// - Tool-call fragments merge through
///   [`ToolCallAccumulator::push_delta`](crate::prompt::ToolCallAccumulator::push_delta),
///   so fragments of the same call collapse into one call per index.
/// - The last non-empty [`Usage`] wins; an all-zero usage never overwrites an
///   earlier one.
/// - The last reported finish reason wins; [`StreamChunk::Done`] is a no-op.
/// - [`StreamAccumulator::finish`] infers [`FinishReason::ToolCalls`] when the
///   provider reported tool calls but no terminal reason.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct StreamAccumulator {
    content: String,
    reasoning: String,
    tool_calls: ToolCallAccumulator,
    finish_reason: Option<FinishReason>,
    usage: Option<Usage>,
}

impl StreamAccumulator {
    /// Creates an empty accumulator.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Merges one stream chunk.
    ///
    /// Returns nothing: malformed provider data is the parser's concern, so the
    /// accumulator only concatenates what it is given.
    pub fn push(&mut self, chunk: &StreamChunk) {
        match chunk {
            StreamChunk::Delta {
                content,
                reasoning,
                tool_call,
                finish_reason,
                ..
            } => {
                if let Some(text) = content {
                    self.content.push_str(text);
                }
                if let Some(text) = reasoning {
                    self.reasoning.push_str(text);
                }
                if let Some(fragment) = tool_call {
                    self.tool_calls.push_delta(fragment.clone());
                }
                if let Some(reason) = finish_reason {
                    self.finish_reason = Some(reason.clone());
                }
            }
            StreamChunk::Usage(usage) => {
                if !usage.is_empty() {
                    self.usage = Some(usage.clone());
                }
            }
            StreamChunk::Done => {}
        }
    }

    /// The content accumulated so far.
    #[must_use]
    pub fn content(&self) -> &str {
        &self.content
    }

    /// The reasoning text accumulated so far.
    #[must_use]
    pub fn reasoning(&self) -> &str {
        &self.reasoning
    }

    /// Consumes the accumulator and builds the assistant turn.
    #[must_use]
    pub fn finish(self) -> AssistantTurn {
        let tool_calls = self.tool_calls.into_calls();
        let inferred = if tool_calls.is_empty() {
            None
        } else {
            Some(FinishReason::ToolCalls)
        };
        let finish_reason = self.finish_reason.or(inferred);

        let mut message = BaseMessage::assistant(MessageContent::Text(self.content));
        message.tool_calls = tool_calls;
        if !self.reasoning.is_empty() {
            message.reasoning_content = Some(self.reasoning);
        }

        AssistantTurn {
            message,
            finish_reason,
            usage: self.usage.unwrap_or_default(),
        }
    }
}

/// A model client: one provider, one model, one way to call it.
///
/// `invoke` borrows its inputs and returns the whole response; `stream` takes
/// ownership so the returned [`BoxStream`] is `'static` and can outlive the
/// client call. Implementations echo the configuration they were built with
/// through [`provider`](Self::provider) and [`model_name`](Self::model_name).
#[async_trait]
pub trait Model: Send + Sync {
    /// Canonical provider name, for example `"openai"`.
    fn provider(&self) -> &str;

    /// Provider-facing model name, for example `"gpt-4o"`.
    fn model_name(&self) -> &str;

    /// Sends the conversation and returns the complete response.
    ///
    /// # Errors
    ///
    /// Returns the provider's own error: [`openjiuwen_common::error::Error::Llm`]
    /// for a rejected request, `Config` for authentication failures, and
    /// `Timeout` when a request deadline expires.
    async fn invoke(
        &self,
        messages: &[BaseMessage],
        request: &ModelRequestConfig,
    ) -> Result<ChatResponse>;

    /// Sends the conversation and returns the streamed response.
    ///
    /// The stream is `Send + 'static` and always ends with
    /// [`StreamChunk::Done`] on success; a provider failure surfaces as one final
    /// `Err` item.
    fn stream(
        &self,
        messages: Vec<BaseMessage>,
        request: ModelRequestConfig,
    ) -> BoxStream<'static, Result<StreamChunk>>;
}

/// Forwards every method to the inner model.
///
/// `Arc<dyn Model>` is the type every factory returns, and without this impl
/// callers would have to dereference it before using [`Model`] or [`ModelExt`].
#[async_trait]
impl Model for Arc<dyn Model> {
    fn provider(&self) -> &str {
        self.as_ref().provider()
    }

    fn model_name(&self) -> &str {
        self.as_ref().model_name()
    }

    async fn invoke(
        &self,
        messages: &[BaseMessage],
        request: &ModelRequestConfig,
    ) -> Result<ChatResponse> {
        self.as_ref().invoke(messages, request).await
    }

    fn stream(
        &self,
        messages: Vec<BaseMessage>,
        request: ModelRequestConfig,
    ) -> BoxStream<'static, Result<StreamChunk>> {
        self.as_ref().stream(messages, request)
    }
}

/// Convenience operations built on top of [`Model`].
///
/// A blanket impl covers every model, so `model.complete(...)` works on a client
/// and on an `Arc<dyn Model>` alike.
#[async_trait]
pub trait ModelExt: Model {
    /// Drives a streamed call to completion and returns the assistant turn.
    ///
    /// The request is cloned with `stream = true`, so a caller that filled in
    /// only `model_name` and sampling settings still gets streamed output. The
    /// first error item is returned immediately; a stream that ends without one
    /// produces the accumulated message.
    ///
    /// # Errors
    ///
    /// Propagates the first error the provider's stream yields.
    async fn complete(
        &self,
        messages: &[BaseMessage],
        request: &ModelRequestConfig,
    ) -> Result<AssistantTurn> {
        let request = request.clone().with_stream(true);
        let mut stream = self.stream(messages.to_vec(), request);
        let mut accumulator = StreamAccumulator::new();
        while let Some(chunk) = stream.next().await {
            accumulator.push(&chunk?);
        }
        Ok(accumulator.finish())
    }
}

impl<T: Model + ?Sized> ModelExt for T {}

#[cfg(test)]
mod tests {
    use super::*;
    use async_stream::stream;
    use async_trait::async_trait;
    use openjiuwen_common::error::{Error, ErrorKind};
    use serde_json::json;
    use std::sync::{Arc, Mutex};

    fn content_delta(content: &str) -> StreamChunk {
        StreamChunk::Delta {
            index: 0,
            role: None,
            content: Some(content.to_owned()),
            reasoning: None,
            tool_call: None,
            finish_reason: None,
        }
    }

    fn reasoning_delta(reasoning: &str) -> StreamChunk {
        StreamChunk::Delta {
            index: 0,
            role: None,
            content: None,
            reasoning: Some(reasoning.to_owned()),
            tool_call: None,
            finish_reason: None,
        }
    }

    fn finish_delta(reason: FinishReason) -> StreamChunk {
        StreamChunk::Delta {
            index: 0,
            role: None,
            content: None,
            reasoning: None,
            tool_call: None,
            finish_reason: Some(reason),
        }
    }

    fn tool_fragment(index: u32, id: Option<&str>, name: Option<&str>, args: &str) -> StreamChunk {
        StreamChunk::Delta {
            index: 0,
            role: None,
            content: None,
            reasoning: None,
            tool_call: Some(ToolCallDelta {
                index: Some(index),
                id: id.map(str::to_owned),
                name: name.map(str::to_owned),
                arguments: Some(args.to_owned()),
            }),
            finish_reason: None,
        }
    }

    #[test]
    fn accumulator_concatenates_content_and_reasoning() {
        let mut accumulator = StreamAccumulator::new();
        accumulator.push(&content_delta("hello "));
        accumulator.push(&reasoning_delta("think "));
        accumulator.push(&content_delta("world"));
        accumulator.push(&reasoning_delta("hard"));
        accumulator.push(&StreamChunk::Done);

        assert_eq!(accumulator.content(), "hello world");
        assert_eq!(accumulator.reasoning(), "think hard");

        let turn = accumulator.finish();
        assert_eq!(turn.message.role, Role::Assistant);
        assert_eq!(turn.message.text().as_ref(), "hello world");
        assert_eq!(
            turn.message.reasoning_content.as_deref(),
            Some("think hard")
        );
        assert_eq!(turn.message.tool_calls, Vec::new());
    }

    #[test]
    fn interleaved_tool_call_fragments_merge_into_one_call() {
        let mut accumulator = StreamAccumulator::new();
        accumulator.push(&content_delta("checking "));
        accumulator.push(&tool_fragment(
            0,
            Some("call_1"),
            Some("get_weather"),
            "{\"city\":",
        ));
        accumulator.push(&content_delta("weather"));
        accumulator.push(&tool_fragment(0, None, None, "\"Paris\"}"));
        accumulator.push(&StreamChunk::Done);

        let turn = accumulator.finish();
        assert_eq!(turn.message.text().as_ref(), "checking weather");
        assert_eq!(turn.message.tool_calls.len(), 1);
        let call = &turn.message.tool_calls[0];
        assert_eq!(call.id.as_deref(), Some("call_1"));
        assert_eq!(call.name, "get_weather");
        assert_eq!(call.arguments, "{\"city\":\"Paris\"}");
        assert_eq!(call.index, Some(0));
    }

    #[test]
    fn distinct_tool_call_indexes_stay_separate_calls() {
        let mut accumulator = StreamAccumulator::new();
        accumulator.push(&tool_fragment(0, Some("call_1"), Some("first"), "{}"));
        accumulator.push(&tool_fragment(1, Some("call_2"), Some("second"), "{}"));

        let calls = accumulator.finish().message.tool_calls;
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].index, Some(0));
        assert_eq!(calls[1].index, Some(1));
    }

    #[test]
    fn last_non_empty_usage_wins() {
        let mut accumulator = StreamAccumulator::new();
        accumulator.push(&StreamChunk::Usage(Usage::new(10, 2)));
        accumulator.push(&StreamChunk::Usage(Usage::new(11, 3)));

        assert_eq!(accumulator.finish().usage, Usage::new(11, 3));
    }

    #[test]
    fn zeroed_usage_does_not_clear_a_reported_one() {
        let mut accumulator = StreamAccumulator::new();
        accumulator.push(&StreamChunk::Usage(Usage::new(10, 2)));
        accumulator.push(&StreamChunk::Usage(Usage::default()));

        assert_eq!(accumulator.finish().usage, Usage::new(10, 2));
    }

    #[test]
    fn last_finish_reason_wins() {
        let mut accumulator = StreamAccumulator::new();
        accumulator.push(&finish_delta(FinishReason::Length));
        accumulator.push(&finish_delta(FinishReason::Stop));

        assert_eq!(accumulator.finish().finish_reason, Some(FinishReason::Stop));
    }

    #[test]
    fn tool_calls_infer_the_finish_reason() {
        let mut accumulator = StreamAccumulator::new();
        accumulator.push(&tool_fragment(0, Some("call_1"), Some("noop"), "{}"));

        let turn = accumulator.finish();
        assert_eq!(turn.finish_reason, Some(FinishReason::ToolCalls));
    }

    #[test]
    fn reported_finish_reason_beats_the_inference() {
        let mut accumulator = StreamAccumulator::new();
        accumulator.push(&tool_fragment(0, Some("call_1"), Some("noop"), "{}"));
        accumulator.push(&finish_delta(FinishReason::ContentFilter));

        assert_eq!(
            accumulator.finish().finish_reason,
            Some(FinishReason::ContentFilter)
        );
    }

    #[test]
    fn done_is_a_no_op() {
        let mut accumulator = StreamAccumulator::new();
        accumulator.push(&content_delta("text"));
        accumulator.push(&StreamChunk::Done);
        accumulator.push(&StreamChunk::Done);

        let turn = accumulator.finish();
        assert_eq!(turn.message.text().as_ref(), "text");
        assert_eq!(turn.finish_reason, None);
        assert_eq!(turn.usage, Usage::default());
    }

    #[test]
    fn finish_reason_maps_to_and_from_wire_strings() {
        let cases = [
            (FinishReason::Stop, "stop"),
            (FinishReason::Length, "length"),
            (FinishReason::ToolCalls, "tool_calls"),
            (FinishReason::ContentFilter, "content_filter"),
            (FinishReason::Other("eos".to_owned()), "eos"),
        ];
        for (reason, wire) in &cases {
            assert_eq!(reason.as_str(), *wire);
            assert_eq!(
                serde_json::to_value(reason).expect("serializes"),
                json!(wire)
            );
            let parsed: FinishReason = serde_json::from_value(json!(wire)).expect("deserializes");
            assert_eq!(&parsed, reason);
        }

        let unknown: FinishReason =
            serde_json::from_value(json!("malformed")).expect("unknown reasons are kept");
        assert_eq!(unknown, FinishReason::Other("malformed".to_owned()));

        let error = serde_json::from_value::<FinishReason>(json!(42))
            .expect_err("non-strings are rejected");
        assert!(error.to_string().contains("finish reason"));
    }

    #[test]
    fn stream_chunk_round_trips_through_json() {
        for chunk in [
            content_delta("hi"),
            finish_delta(FinishReason::ToolCalls),
            tool_fragment(0, Some("call_1"), Some("f"), "{}"),
            StreamChunk::Usage(Usage::new(1, 2)),
            StreamChunk::Done,
        ] {
            let encoded = serde_json::to_string(&chunk).expect("serializes");
            let restored: StreamChunk = serde_json::from_str(&encoded).expect("deserializes");
            assert_eq!(restored, chunk);
        }

        assert_eq!(content_delta("hi").content(), Some("hi"));
        assert_eq!(StreamChunk::Done.content(), None);
    }

    #[test]
    fn chat_response_round_trips_and_exposes_the_first_choice() {
        let response = ChatResponse {
            id: "req-1".to_owned(),
            model: "gpt-4o".to_owned(),
            choices: vec![Choice {
                index: 0,
                message: BaseMessage::assistant("answer"),
                finish_reason: Some(FinishReason::Stop),
            }],
            usage: Usage::new(12, 4),
        };

        assert_eq!(
            response.message().map(|message| &message.role),
            Some(&Role::Assistant)
        );
        assert_eq!(response.finish_reason(), Some(&FinishReason::Stop));

        let encoded = serde_json::to_string(&response).expect("serializes");
        let restored: ChatResponse = serde_json::from_str(&encoded).expect("deserializes");
        assert_eq!(restored, response);

        let empty = ChatResponse::default();
        assert!(empty.message().is_none());
        assert!(empty.finish_reason().is_none());
    }

    #[derive(Clone)]
    enum ScriptItem {
        Chunk(StreamChunk),
        Fail(&'static str),
    }

    /// A model that yields a fixed script and records the request it received.
    #[derive(Clone, Default)]
    struct ScriptedModel {
        script: Arc<Vec<ScriptItem>>,
        seen: Arc<Mutex<Option<ModelRequestConfig>>>,
    }

    fn scripted(script: Vec<ScriptItem>) -> ScriptedModel {
        ScriptedModel {
            script: Arc::new(script),
            seen: Arc::default(),
        }
    }

    #[async_trait]
    impl Model for ScriptedModel {
        fn provider(&self) -> &'static str {
            "scripted"
        }

        fn model_name(&self) -> &'static str {
            "scripted-model"
        }

        async fn invoke(
            &self,
            _messages: &[BaseMessage],
            _request: &ModelRequestConfig,
        ) -> Result<ChatResponse> {
            Err(Error::unsupported("invoke is not scripted"))
        }

        fn stream(
            &self,
            _messages: Vec<BaseMessage>,
            request: ModelRequestConfig,
        ) -> BoxStream<'static, Result<StreamChunk>> {
            *self.seen.lock().expect("request slot") = Some(request);
            let script = Arc::clone(&self.script);
            Box::pin(stream! {
                for item in script.iter() {
                    match item {
                        ScriptItem::Chunk(chunk) => yield Ok(chunk.clone()),
                        ScriptItem::Fail(message) => {
                            yield Err(Error::llm(*message));
                            break;
                        }
                    }
                }
            })
        }
    }

    #[tokio::test]
    async fn complete_forces_streaming_and_returns_the_turn() {
        let model = scripted(vec![
            ScriptItem::Chunk(content_delta("hello ")),
            ScriptItem::Chunk(content_delta("world")),
            ScriptItem::Chunk(StreamChunk::Usage(Usage::new(10, 4))),
            ScriptItem::Chunk(finish_delta(FinishReason::Stop)),
            ScriptItem::Chunk(StreamChunk::Done),
        ]);
        let request = ModelRequestConfig::new("gpt-4o");

        let turn = model
            .complete(&[BaseMessage::user("hi")], &request)
            .await
            .expect("completes");

        assert_eq!(turn.message.role, Role::Assistant);
        assert_eq!(turn.message.text().as_ref(), "hello world");
        assert_eq!(turn.finish_reason, Some(FinishReason::Stop));
        assert_eq!(turn.usage, Usage::new(10, 4));

        let seen = model
            .seen
            .lock()
            .expect("request slot")
            .clone()
            .expect("request recorded");
        assert!(seen.stream, "complete must force streaming");
        assert_eq!(seen.model_name, "gpt-4o");
        assert!(!request.stream, "the caller's request is not mutated");
    }

    #[tokio::test]
    async fn complete_propagates_the_first_stream_error() {
        let model = scripted(vec![
            ScriptItem::Chunk(content_delta("partial")),
            ScriptItem::Fail("provider exploded"),
            ScriptItem::Chunk(content_delta("never seen")),
        ]);

        let error = model
            .complete(&[], &ModelRequestConfig::new("gpt-4o"))
            .await
            .expect_err("a stream error must fail the call");

        assert_eq!(error.kind(), ErrorKind::Llm);
        assert!(error.to_string().contains("provider exploded"));
    }

    #[tokio::test]
    async fn arc_dyn_model_forwards_every_method() {
        let model: Arc<dyn Model> = Arc::new(scripted(vec![
            ScriptItem::Chunk(content_delta("ok")),
            ScriptItem::Chunk(StreamChunk::Done),
        ]));

        assert_eq!(model.provider(), "scripted");
        assert_eq!(model.model_name(), "scripted-model");

        let turn = model
            .complete(&[], &ModelRequestConfig::new("gpt-4o"))
            .await
            .expect("completes through the arc");
        assert_eq!(turn.message.text().as_ref(), "ok");

        let response = model.invoke(&[], &ModelRequestConfig::new("gpt-4o")).await;
        assert!(response.is_err(), "invoke forwards to the inner model");
    }
}
