//! Anthropic Messages event-stream conversion.
//!
//! The Messages API streams typed events rather than OpenAI-style deltas:
//! `message_start`, `content_block_start`, `content_block_delta` (text, thinking,
//! or tool-input fragments), `content_block_stop`, `message_delta`, and
//! `message_stop`. [`AnthropicStreamState`] folds that sequence into the
//! provider-neutral [`StreamChunk`] stream:
//!
//! - `message_start` yields the assistant role and the prompt usage;
//! - text and thinking deltas become `content` and `reasoning` fragments;
//! - tool-input fragments are buffered per content-block index and emitted as one
//!   complete [`ToolCallDelta`] on `content_block_stop`;
//! - `thinking`/`redacted_thinking` blocks are buffered per content-block index
//!   (text, `signature_delta` proof, redacted `data`) and kept on the state once
//!   the stream stops them, so [`AnthropicStreamState::apply_thinking_blocks`]
//!   can make the accumulated message replayable exactly like the non-streaming
//!   parser's message;
//! - `message_delta` yields the terminal reason and the merged usage (Anthropic
//!   reports the output count there while the prompt count arrives in
//!   `message_start`, so the state keeps the prompt side to produce one complete
//!   accounting);
//! - `message_stop` yields exactly one [`StreamChunk::Done`]; [`finish`] supplies
//!   it when the server closes the stream without one.
//!
//! Malformed JSON is an [`Error::Llm`]. Events whose payload is merely
//! informational (a missing block index, an unknown event name) are skipped with
//! a `tracing` diagnostic, because they cannot change the assembled message.
//!
//! [`finish`]: AnthropicStreamState::finish

mod thinking;

use std::collections::BTreeMap;

use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::llm::model::{FinishReason, StreamChunk};
use crate::llm::sse::SseEvent;
use crate::prompt::{BaseMessage, Role, ToolCallDelta, Usage};

use super::response::{finish_reason_from_stop_reason, parse_anthropic_usage};

use thinking::ThinkingBlocks;

/// Accumulator for one Anthropic Messages event stream.
///
/// Feed every [`SseEvent`] to [`handle_event`](Self::handle_event) and call
/// [`finish`](Self::finish) after the response body ends. The state is
/// `serde`-serializable so a caller can checkpoint a partially consumed stream.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct AnthropicStreamState {
    /// Provider message id reported by `message_start`.
    message_id: Option<String>,
    /// Model reported by `message_start`.
    model: Option<String>,
    /// Tool-use blocks being assembled, keyed by content-block index.
    blocks: BTreeMap<u32, ToolBlock>,
    /// Thinking blocks being assembled, and those already sealed for replay.
    thinking: ThinkingBlocks,
    /// Usage reported so far; `message_delta` merges its output count into it.
    usage: Option<Usage>,
    /// Whether a `Done` chunk has been emitted.
    done_emitted: bool,
}

/// One tool-use content block being accumulated from stream fragments.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
struct ToolBlock {
    /// Provider call id, known from `content_block_start`.
    id: Option<String>,
    /// Tool name, known from `content_block_start`.
    name: Option<String>,
    /// Concatenated `input_json_delta` fragments.
    arguments: String,
    /// Whether the block is a tool call (input fragments can arrive before start).
    is_tool_use: bool,
}

impl AnthropicStreamState {
    /// Creates an empty stream state.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The provider message id, once `message_start` has been seen.
    #[must_use]
    pub fn message_id(&self) -> Option<&str> {
        self.message_id.as_deref()
    }

    /// The model reported by `message_start`, when the provider sent one.
    #[must_use]
    pub fn model(&self) -> Option<&str> {
        self.model.as_deref()
    }

    /// The replayable thinking blocks this stream preserved, in block order.
    ///
    /// A block is sealed by its `content_block_stop`, or by
    /// [`finish`](Self::finish) when the stream ends without one, so read this
    /// after the stream has finished. Only blocks the provider vouched for are
    /// returned (a `signature` for `thinking`, opaque `data` for
    /// `redacted_thinking`); an unfinished block is dropped with a warning.
    #[must_use]
    pub fn thinking_blocks(&self) -> Vec<Value> {
        self.thinking.preserved()
    }

    /// Attaches the preserved thinking blocks to an accumulated assistant message.
    ///
    /// Pass the message a [`StreamAccumulator`](crate::llm::model::StreamAccumulator)
    /// built from this stream's chunks, after the stream ended: the thinking text
    /// already arrived there as `reasoning_content`, and this call adds the
    /// provider blocks a follow-up request has to echo back. A turn with no
    /// replayable block is left untouched.
    pub fn apply_thinking_blocks(&self, message: &mut BaseMessage) {
        self.thinking.apply(message);
    }

    /// Converts one Server-Sent-Events event into zero or more chunks.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Llm`] for malformed event JSON and for an `error` event
    /// (the payload's type and message are included, but never the raw body).
    pub fn handle_event(&mut self, event: &SseEvent) -> Result<Vec<StreamChunk>> {
        let data = parse_data(event)?;
        let name = event_name(event, &data);
        let mut chunks = Vec::new();

        match name {
            "message_start" => self.message_start(&data, &mut chunks),
            "content_block_start" => self.content_block_start(&data),
            "content_block_delta" => self.content_block_delta(&data, &mut chunks),
            "content_block_stop" => self.content_block_stop(&data, &mut chunks),
            "message_delta" => self.message_delta(&data, &mut chunks),
            "message_stop" => self.message_stop(&mut chunks),
            "ping" => {}
            "error" => return Err(stream_error(&data)),
            "" => tracing::warn!("ignoring a typed Anthropic event without a name"),
            other => tracing::debug!(event = other, "ignoring an unknown Anthropic event"),
        }

        Ok(chunks)
    }

    /// Emits the terminal [`StreamChunk::Done`] unless `message_stop` did.
    ///
    /// Calling it more than once is safe: only the first call can emit a chunk.
    /// It also seals any thinking block the server never stopped, so a block the
    /// provider fully signed is still replayable and an unsigned one is dropped.
    pub fn finish(&mut self) -> Vec<StreamChunk> {
        self.thinking.seal_pending();
        if self.done_emitted {
            return Vec::new();
        }
        self.done_emitted = true;
        vec![StreamChunk::Done]
    }

    /// Handles `message_start`: role delta, identity, and prompt usage.
    fn message_start(&mut self, data: &Value, chunks: &mut Vec<StreamChunk>) {
        let message = data.get("message").filter(|value| value.is_object());
        if let Some(message) = message {
            self.message_id = non_empty_string(message.get("id"));
            self.model = non_empty_string(message.get("model"));
        }

        chunks.push(StreamChunk::Delta {
            index: 0,
            role: Some(Role::Assistant),
            content: None,
            reasoning: None,
            tool_call: None,
            finish_reason: None,
        });

        if let Some(usage) = message
            .and_then(|message| message.get("usage"))
            .filter(|value| value.is_object())
        {
            let usage = parse_anthropic_usage(usage);
            self.usage = Some(usage.clone());
            chunks.push(StreamChunk::Usage(usage));
        }
    }

    /// Handles `content_block_start`: remembers a tool call's id and name, or
    /// opens a thinking block with what its start event already carries.
    fn content_block_start(&mut self, data: &Value) {
        let Some(index) = block_index(data) else {
            tracing::warn!("ignoring an Anthropic content_block_start without an index");
            return;
        };
        let Some(block) = data.get("content_block").filter(|value| value.is_object()) else {
            tracing::warn!("ignoring an Anthropic content_block_start without a block");
            return;
        };

        match block.get("type").and_then(Value::as_str) {
            Some("tool_use") => {
                self.blocks.insert(
                    index,
                    ToolBlock {
                        id: non_empty_string(block.get("id")),
                        name: non_empty_string(block.get("name")),
                        arguments: String::new(),
                        is_tool_use: true,
                    },
                );
            }
            Some("thinking") => self.thinking.open_thinking(
                index,
                non_empty_str(block.get("thinking")),
                non_empty_str(block.get("signature")),
            ),
            Some("redacted_thinking") => self
                .thinking
                .open_redacted(index, non_empty_str(block.get("data"))),
            _ => {}
        }
    }

    /// Handles `content_block_delta`: text, thinking, or a tool-input fragment.
    fn content_block_delta(&mut self, data: &Value, chunks: &mut Vec<StreamChunk>) {
        let Some(delta) = data.get("delta").filter(|value| value.is_object()) else {
            tracing::warn!("ignoring an Anthropic content_block_delta without a delta");
            return;
        };

        match delta.get("type").and_then(Value::as_str) {
            Some("text_delta") => {
                if let Some(text) = non_empty_str(delta.get("text")) {
                    chunks.push(delta_chunk(Some(text), None, None));
                }
            }
            Some("thinking_delta") => {
                if let Some(thinking) = non_empty_str(delta.get("thinking")) {
                    if let Some(index) = block_index(data) {
                        self.thinking.append_text(index, thinking);
                    }
                    chunks.push(delta_chunk(None, Some(thinking), None));
                }
            }
            Some("signature_delta") => {
                if let (Some(index), Some(signature)) =
                    (block_index(data), non_empty_str(delta.get("signature")))
                {
                    self.thinking.append_signature(index, signature);
                }
            }
            Some("data") => {
                if let (Some(index), Some(fragment)) =
                    (block_index(data), non_empty_str(delta.get("data")))
                {
                    self.thinking.append_data(index, fragment);
                }
            }
            Some("input_json_delta") => {
                let Some(index) = block_index(data) else {
                    tracing::warn!("ignoring a tool-input fragment without a block index");
                    return;
                };
                let partial = delta
                    .get("partial_json")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let block = self.blocks.entry(index).or_default();
                block.is_tool_use = true;
                block.arguments.push_str(partial);
            }
            // An unknown delta type cannot contribute to the assistant message.
            _ => {}
        }
    }

    /// Handles `content_block_stop`: seals a thinking block, emits one complete
    /// tool call for a tool block.
    fn content_block_stop(&mut self, data: &Value, chunks: &mut Vec<StreamChunk>) {
        let Some(index) = block_index(data) else {
            tracing::warn!("ignoring an Anthropic content_block_stop without an index");
            return;
        };
        self.thinking.seal(index);

        let Some(block) = self.blocks.get(&index) else {
            return;
        };
        if !block.is_tool_use {
            return;
        }

        let arguments = if block.arguments.is_empty() {
            "{}".to_owned()
        } else {
            block.arguments.clone()
        };
        chunks.push(delta_chunk(
            None,
            None,
            Some(ToolCallDelta {
                index: Some(index),
                id: block.id.clone(),
                name: block.name.clone(),
                arguments: Some(arguments),
            }),
        ));
    }

    /// Handles `message_delta`: terminal reason and completed usage.
    fn message_delta(&mut self, data: &Value, chunks: &mut Vec<StreamChunk>) {
        if let Some(stop_reason) = data
            .get("delta")
            .and_then(|delta| delta.get("stop_reason"))
            .and_then(Value::as_str)
            .filter(|reason| !reason.is_empty())
        {
            chunks.push(finish_chunk(finish_reason_from_stop_reason(stop_reason)));
        }

        if let Some(usage) = data.get("usage").filter(|value| value.is_object()) {
            let usage = merge_usage(self.usage.as_ref(), &parse_anthropic_usage(usage));
            self.usage = Some(usage.clone());
            chunks.push(StreamChunk::Usage(usage));
        }
    }

    /// Handles `message_stop`: emits the single terminal `Done`.
    fn message_stop(&mut self, chunks: &mut Vec<StreamChunk>) {
        self.thinking.seal_pending();
        if !self.done_emitted {
            self.done_emitted = true;
            chunks.push(StreamChunk::Done);
        }
    }
}

/// Merges a `message_delta` usage into the usage reported by `message_start`.
///
/// Anthropic reports the prompt count in `message_start` and the output count in
/// `message_delta`; a zero field in the update keeps the previously known value
/// so the terminal chunk carries the complete accounting.
fn merge_usage(previous: Option<&Usage>, update: &Usage) -> Usage {
    let Some(previous) = previous else {
        return update.clone();
    };

    Usage {
        input_tokens: if update.input_tokens == 0 {
            previous.input_tokens
        } else {
            update.input_tokens
        },
        output_tokens: if update.output_tokens == 0 {
            previous.output_tokens
        } else {
            update.output_tokens
        },
        cache_tokens: if update.cache_tokens == 0 {
            previous.cache_tokens
        } else {
            update.cache_tokens
        },
        reasoning_tokens: if update.reasoning_tokens == 0 {
            previous.reasoning_tokens
        } else {
            update.reasoning_tokens
        },
        ..Usage::default()
    }
    .normalized()
}

/// Builds a delta chunk for content, reasoning, or a tool-call fragment.
fn delta_chunk(
    content: Option<&str>,
    reasoning: Option<&str>,
    tool_call: Option<ToolCallDelta>,
) -> StreamChunk {
    StreamChunk::Delta {
        index: 0,
        role: None,
        content: content.map(str::to_owned),
        reasoning: reasoning.map(str::to_owned),
        tool_call,
        finish_reason: None,
    }
}

/// Builds a delta chunk that carries only a terminal reason.
fn finish_chunk(reason: FinishReason) -> StreamChunk {
    StreamChunk::Delta {
        index: 0,
        role: None,
        content: None,
        reasoning: None,
        tool_call: None,
        finish_reason: Some(reason),
    }
}

/// Parses an event's JSON payload; an empty payload becomes [`Value::Null`].
fn parse_data(event: &SseEvent) -> Result<Value> {
    let data = event.data.trim();
    if data.is_empty() {
        return Ok(Value::Null);
    }
    serde_json::from_str(data).map_err(|error| {
        Error::llm(format!(
            "malformed Anthropic stream event `{}`: {error}",
            event.event.as_deref().unwrap_or("data")
        ))
    })
}

/// Resolves the event name, preferring the `event:` field over the payload type.
fn event_name<'a>(event: &'a SseEvent, data: &'a Value) -> &'a str {
    event
        .event
        .as_deref()
        .filter(|name| !name.is_empty())
        .or_else(|| data.get("type").and_then(Value::as_str))
        .unwrap_or_default()
}

/// Reads a content-block index from an event payload.
fn block_index(data: &Value) -> Option<u32> {
    data.get("index")
        .and_then(Value::as_u64)
        .and_then(|raw| u32::try_from(raw).ok())
}

/// Reads a non-empty string field.
fn non_empty_str(value: Option<&Value>) -> Option<&str> {
    value
        .and_then(Value::as_str)
        .filter(|text| !text.is_empty())
}

/// Reads a non-empty string field into an owned value.
fn non_empty_string(value: Option<&Value>) -> Option<String> {
    non_empty_str(value).map(str::to_owned)
}

/// Builds the error reported by an Anthropic `error` event.
///
/// Only the structured type and message are used; the raw payload is never
/// echoed, so a provider that puts credentials in it cannot leak them here.
fn stream_error(data: &Value) -> Error {
    let error = data.get("error");
    let kind = error
        .and_then(|error| error.get("type"))
        .and_then(Value::as_str)
        .unwrap_or("error");
    let message = error
        .and_then(|error| error.get("message"))
        .and_then(Value::as_str)
        .unwrap_or("the provider reported an unspecified stream error");

    Error::llm(format!("Anthropic stream error ({kind}): {message}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use openjiuwen_common::error::ErrorKind;
    use serde_json::json;

    fn event(name: &str, data: Value) -> SseEvent {
        SseEvent {
            event: Some(name.to_owned()),
            data: data.to_string(),
        }
    }

    fn anonymous_event(data: Value) -> SseEvent {
        SseEvent {
            event: None,
            data: data.to_string(),
        }
    }

    fn message_start() -> SseEvent {
        event(
            "message_start",
            json!({
                "type": "message_start",
                "message": {
                    "id": "msg_1",
                    "model": "claude-sonnet-4",
                    "usage": {"input_tokens": 10, "cache_read_input_tokens": 2},
                },
            }),
        )
    }

    fn text_delta(text: &str) -> SseEvent {
        event(
            "content_block_delta",
            json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": text},
            }),
        )
    }

    fn tool_use_start(index: u32) -> SseEvent {
        event(
            "content_block_start",
            json!({
                "type": "content_block_start",
                "index": index,
                "content_block": {"type": "tool_use", "id": format!("call_{index}"), "name": "get_weather", "input": {}},
            }),
        )
    }

    fn input_json(index: u32, partial: &str) -> SseEvent {
        event(
            "content_block_delta",
            json!({
                "type": "content_block_delta",
                "index": index,
                "delta": {"type": "input_json_delta", "partial_json": partial},
            }),
        )
    }

    fn block_stop(index: u32) -> SseEvent {
        event(
            "content_block_stop",
            json!({"type": "content_block_stop", "index": index}),
        )
    }

    #[test]
    fn message_start_emits_the_role_identity_and_usage() {
        let mut state = AnthropicStreamState::new();
        let chunks = state.handle_event(&message_start()).expect("event");

        assert_eq!(state.message_id(), Some("msg_1"));
        assert_eq!(state.model(), Some("claude-sonnet-4"));
        assert_eq!(chunks.len(), 2);
        assert_eq!(
            chunks[0],
            StreamChunk::Delta {
                index: 0,
                role: Some(Role::Assistant),
                content: None,
                reasoning: None,
                tool_call: None,
                finish_reason: None,
            }
        );
        match &chunks[1] {
            StreamChunk::Usage(usage) => {
                assert_eq!(usage.input_tokens, 12);
                assert_eq!(usage.cache_tokens, 2);
                assert_eq!(usage.total_tokens, 12);
            }
            other => panic!("expected a usage chunk, got {other:?}"),
        }
    }

    #[test]
    fn text_and_thinking_deltas_map_to_content_and_reasoning() {
        let mut state = AnthropicStreamState::new();
        let chunks = state.handle_event(&text_delta("hello ")).expect("event");
        assert_eq!(chunks, vec![delta_chunk(Some("hello "), None, None)]);

        let chunks = state
            .handle_event(&event(
                "content_block_delta",
                json!({
                    "type": "content_block_delta",
                    "index": 1,
                    "delta": {"type": "thinking_delta", "thinking": "hmm"},
                }),
            ))
            .expect("event");
        assert_eq!(chunks, vec![delta_chunk(None, Some("hmm"), None)]);

        let empty = state.handle_event(&text_delta("")).expect("event");
        assert!(empty.is_empty(), "empty text fragments are not emitted");
    }

    #[test]
    fn tool_input_fragments_accumulate_until_the_block_stops() {
        let mut state = AnthropicStreamState::new();
        assert!(
            state
                .handle_event(&tool_use_start(0))
                .expect("event")
                .is_empty(),
            "no chunks expected"
        );
        assert!(
            state
                .handle_event(&input_json(0, "{\"city\":"))
                .expect("event")
                .is_empty(),
            "no chunks expected"
        );
        assert!(
            state
                .handle_event(&input_json(0, "\"Paris\"}"))
                .expect("event")
                .is_empty(),
            "no chunks expected"
        );

        let chunks = state.handle_event(&block_stop(0)).expect("event");
        assert_eq!(chunks.len(), 1);
        match &chunks[0] {
            StreamChunk::Delta {
                tool_call: Some(delta),
                ..
            } => {
                assert_eq!(
                    delta,
                    &ToolCallDelta {
                        index: Some(0),
                        id: Some("call_0".to_owned()),
                        name: Some("get_weather".to_owned()),
                        arguments: Some("{\"city\":\"Paris\"}".to_owned()),
                    }
                );
            }
            other => panic!("expected a tool-call delta, got {other:?}"),
        }
    }

    #[test]
    fn a_tool_call_without_input_fragments_gets_an_empty_object() {
        let mut state = AnthropicStreamState::new();
        state.handle_event(&tool_use_start(0)).expect("event");
        let chunks = state.handle_event(&block_stop(0)).expect("event");

        match &chunks[0] {
            StreamChunk::Delta {
                tool_call: Some(delta),
                ..
            } => assert_eq!(delta.arguments.as_deref(), Some("{}")),
            other => panic!("expected a tool-call delta, got {other:?}"),
        }
    }

    #[test]
    fn input_fragments_before_a_start_still_produce_a_call() {
        let mut state = AnthropicStreamState::new();
        assert!(
            state
                .handle_event(&input_json(3, "{}"))
                .expect("event")
                .is_empty(),
            "no chunks expected"
        );
        let chunks = state.handle_event(&block_stop(3)).expect("event");

        match &chunks[0] {
            StreamChunk::Delta {
                tool_call: Some(delta),
                ..
            } => {
                assert_eq!(delta.index, Some(3));
                assert_eq!(delta.id, None);
                assert_eq!(delta.arguments.as_deref(), Some("{}"));
            }
            other => panic!("expected a tool-call delta, got {other:?}"),
        }
    }

    #[test]
    fn text_block_stops_emit_nothing() {
        let mut state = AnthropicStreamState::new();
        state
            .handle_event(&event(
                "content_block_start",
                json!({
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                }),
            ))
            .expect("event");
        assert!(
            state
                .handle_event(&block_stop(0))
                .expect("event")
                .is_empty(),
            "a text block stop emits nothing"
        );
        assert!(
            state
                .handle_event(&block_stop(9))
                .expect("event")
                .is_empty(),
            "an unknown block stop emits nothing"
        );
    }

    #[test]
    fn message_delta_merges_the_prompt_and_output_usage() {
        let mut state = AnthropicStreamState::new();
        state.handle_event(&message_start()).expect("event");

        let chunks = state
            .handle_event(&event(
                "message_delta",
                json!({
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": null},
                    "usage": {"output_tokens": 4},
                }),
            ))
            .expect("event");

        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0], finish_chunk(FinishReason::Stop));
        match &chunks[1] {
            StreamChunk::Usage(usage) => {
                assert_eq!(usage.input_tokens, 12);
                assert_eq!(usage.output_tokens, 4);
                assert_eq!(usage.total_tokens, 16);
                assert_eq!(usage.cache_tokens, 2);
            }
            other => panic!("expected a usage chunk, got {other:?}"),
        }
    }

    #[test]
    fn message_delta_without_a_prior_start_uses_its_own_usage() {
        let mut state = AnthropicStreamState::new();
        let chunks = state
            .handle_event(&event(
                "message_delta",
                json!({
                    "type": "message_delta",
                    "delta": {"stop_reason": "max_tokens"},
                    "usage": {"input_tokens": 3, "output_tokens": 5},
                }),
            ))
            .expect("event");

        assert_eq!(chunks[0], finish_chunk(FinishReason::Length));
        match &chunks[1] {
            StreamChunk::Usage(usage) => assert_eq!(*usage, Usage::new(3, 5)),
            other => panic!("expected a usage chunk, got {other:?}"),
        }
    }

    #[test]
    fn message_stop_emits_one_done_and_finish_is_a_no_op() {
        let mut state = AnthropicStreamState::new();
        let chunks = state
            .handle_event(&event("message_stop", json!({"type": "message_stop"})))
            .expect("event");
        assert_eq!(chunks, vec![StreamChunk::Done]);
        assert!(
            state
                .handle_event(&event("message_stop", json!({"type": "message_stop"})))
                .expect("event")
                .is_empty(),
            "no chunks expected"
        );
        assert!(state.finish().is_empty(), "the stream already completed");
    }

    #[test]
    fn finish_supplies_done_when_message_stop_is_missing() {
        let mut state = AnthropicStreamState::new();
        assert_eq!(state.finish(), vec![StreamChunk::Done]);
        assert!(state.finish().is_empty(), "the stream already completed");
    }

    #[test]
    fn ping_and_unknown_events_are_ignored() {
        let mut state = AnthropicStreamState::new();
        assert!(
            state
                .handle_event(&event("ping", json!({"type": "ping"})))
                .expect("event")
                .is_empty(),
            "no chunks expected"
        );
        assert!(
            state
                .handle_event(&event("future_event", json!({"type": "future_event"})))
                .expect("event")
                .is_empty(),
            "no chunks expected"
        );
    }

    #[test]
    fn the_payload_type_is_used_when_the_event_field_is_missing() {
        let mut state = AnthropicStreamState::new();
        let chunks = state
            .handle_event(&anonymous_event(json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "hi"},
            })))
            .expect("event");
        assert_eq!(chunks, vec![delta_chunk(Some("hi"), None, None)]);
    }

    #[test]
    fn error_events_become_llm_errors_without_the_raw_body() {
        let mut state = AnthropicStreamState::new();
        let error = state
            .handle_event(&event(
                "error",
                json!({
                    "type": "error",
                    "error": {"type": "overloaded_error", "message": "try later"},
                    "secret": "sk-live-do-not-leak",
                }),
            ))
            .expect_err("the provider reported an error");

        assert_eq!(error.kind(), ErrorKind::Llm);
        let rendered = error.to_string();
        assert!(rendered.contains("overloaded_error"), "{rendered}");
        assert!(rendered.contains("try later"), "{rendered}");
        assert!(!rendered.contains("sk-live-do-not-leak"), "{rendered}");
    }

    #[test]
    fn malformed_event_json_is_an_llm_error() {
        let mut state = AnthropicStreamState::new();
        let error = state
            .handle_event(&SseEvent {
                event: Some("message_start".to_owned()),
                data: "{not json".to_owned(),
            })
            .expect_err("malformed payload");

        assert_eq!(error.kind(), ErrorKind::Llm);
        assert!(error.to_string().contains("message_start"));
    }

    #[test]
    fn missing_block_fields_are_skipped_without_panicking() {
        let mut state = AnthropicStreamState::new();
        for event in [
            event(
                "content_block_start",
                json!({"type": "content_block_start"}),
            ),
            event(
                "content_block_start",
                json!({"type": "content_block_start", "index": 0}),
            ),
            event(
                "content_block_delta",
                json!({"type": "content_block_delta", "index": 0}),
            ),
            event("content_block_stop", json!({"type": "content_block_stop"})),
            event("message_delta", json!({"type": "message_delta"})),
            SseEvent {
                event: Some("ping".to_owned()),
                data: String::new(),
            },
        ] {
            assert!(
                state.handle_event(&event).expect("event").is_empty(),
                "informational events emit nothing"
            );
        }
    }
}
