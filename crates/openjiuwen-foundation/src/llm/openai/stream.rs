//! OpenAI-compatible stream decoding: one SSE payload at a time.
//!
//! Ports `_parse_stream_chunk` of
//! `openjiuwen/core/foundation/llm/model_clients/openai_model_client.py`.
//! [`OpenAiStreamState`] owns the per-stream bookkeeping (the `[DONE]` sentinel
//! must produce exactly one [`StreamChunk::Done`]), while [`parse_stream_chunk`]
//! is a pure function over one decoded JSON frame.

use openjiuwen_common::error::{Error, Result};
use serde_json::{Map, Value};

use crate::llm::transport::redacted_body;
use crate::prompt::{Role, ToolCallDelta};

use super::super::model::StreamChunk;
use super::response::{json_kind, parse_finish_reason, parse_usage};

/// Payload that terminates an OpenAI-compatible event stream.
const DONE_SENTINEL: &str = "[DONE]";

/// Per-stream decoder state.
///
/// Feed every [`SseEvent::data`](crate::llm::sse::SseEvent::data) payload to
/// [`handle_data`](Self::handle_data), then call [`finish`](Self::finish) when the
/// HTTP body ends. A stream produces exactly one [`StreamChunk::Done`], whether
/// the provider sent `[DONE]` or simply closed the connection, and any payload
/// after `[DONE]` is ignored.
#[derive(Debug, Clone, Default)]
pub struct OpenAiStreamState {
    /// Whether the terminal `Done` chunk has already been produced.
    done: bool,
}

impl OpenAiStreamState {
    /// Creates an empty state.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Decodes one SSE `data` payload into zero or more chunks.
    ///
    /// `[DONE]` yields [`StreamChunk::Done`] once; an empty payload yields
    /// nothing. Payloads that are not valid JSON are logged with
    /// [`tracing::warn!`] and skipped, because a single undecodable keep-alive or
    /// vendor comment frame must not abort a whole generation.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Llm`] for a structurally invalid frame (not a JSON
    /// object, a bad `choices`/`tool_calls` shape) and for a provider `error`
    /// frame.
    pub fn handle_data(&mut self, data: &str) -> Result<Vec<StreamChunk>> {
        if self.done {
            return Ok(Vec::new());
        }
        let data = data.trim();
        if data.is_empty() {
            return Ok(Vec::new());
        }
        if data == DONE_SENTINEL {
            self.done = true;
            return Ok(vec![StreamChunk::Done]);
        }

        let value: Value = match serde_json::from_str(data) {
            Ok(value) => value,
            Err(error) => {
                tracing::warn!(
                    target: "openjiuwen_foundation::llm::openai",
                    %error,
                    "skipping undecodable stream frame"
                );
                return Ok(Vec::new());
            }
        };
        parse_stream_chunk(&value)
    }

    /// Ends the stream, emitting [`StreamChunk::Done`] when it was not seen yet.
    ///
    /// Calling this more than once emits nothing after the first call.
    pub fn finish(&mut self) -> Vec<StreamChunk> {
        if self.done {
            return Vec::new();
        }
        self.done = true;
        vec![StreamChunk::Done]
    }
}

/// Parses one already-decoded JSON stream frame.
///
/// A frame with at least one choice yields one delta per piece of information:
/// `delta.content` and `delta.reasoning_content` share one delta, every
/// `delta.tool_calls[]` entry becomes its own delta, the role rides on the first
/// delta and `finish_reason` on the last one. A frame with no choices (or an empty
/// `choices` array, which is how providers report usage with
/// `stream_options.include_usage`) yields a [`StreamChunk::Usage`] when it carries
/// a `usage` object.
///
/// A `message` object is accepted as a fallback for `delta`, matching providers
/// that repeat the full assistant message in the final frame.
///
/// # Errors
///
/// Returns [`Error::Llm`] when the frame is not a JSON object, when `choices` is
/// not an array, when a choice or a tool-call fragment has a bad shape, or when
/// the frame is a provider `error` payload.
pub fn parse_stream_chunk(value: &Value) -> Result<Vec<StreamChunk>> {
    let object = value.as_object().ok_or_else(|| {
        Error::llm(format!(
            "stream frame must be a JSON object, got {}",
            json_kind(value)
        ))
    })?;

    if let Some(error) = object.get("error").filter(|error| !error.is_null()) {
        return Err(Error::llm(format!(
            "provider reported a streaming error: {}",
            redacted_body(&error.to_string())
        )));
    }

    let choices = match object.get("choices") {
        None | Some(Value::Null) => None,
        Some(Value::Array(choices)) => Some(choices.as_slice()),
        Some(other) => {
            return Err(Error::llm(format!(
                "stream `choices` must be an array, got {}",
                json_kind(other)
            )))
        }
    };

    match choices {
        Some(choices) if !choices.is_empty() => parse_choices(choices),
        _ => Ok(usage_chunk(object)),
    }
}

/// Parses every choice of one frame, preserving provider order.
fn parse_choices(choices: &[Value]) -> Result<Vec<StreamChunk>> {
    let mut chunks = Vec::new();
    for (position, choice) in choices.iter().enumerate() {
        let object = choice.as_object().ok_or_else(|| {
            Error::llm(format!(
                "stream choice {position} must be a JSON object, got {}",
                json_kind(choice)
            ))
        })?;
        let index = object
            .get("index")
            .and_then(Value::as_u64)
            .and_then(|raw| u32::try_from(raw).ok())
            .unwrap_or(position as u32);

        let payload = object
            .get("delta")
            .and_then(Value::as_object)
            .or_else(|| object.get("message").and_then(Value::as_object));
        let role = payload
            .and_then(|payload| payload.get("role"))
            .and_then(Value::as_str)
            .filter(|role| !role.is_empty())
            .map(Role::from);
        let content = payload.and_then(|payload| text_field(payload, "content"));
        let reasoning = payload.and_then(|payload| text_field(payload, "reasoning_content"));
        let finish_reason = object
            .get("finish_reason")
            .and_then(Value::as_str)
            .and_then(parse_finish_reason);

        let mut frame = Vec::new();
        if content.is_some() || reasoning.is_some() {
            frame.push(delta(index, content, reasoning, None));
        }
        if let Some(raw_calls) = payload.and_then(|payload| payload.get("tool_calls")) {
            let calls = raw_calls.as_array().ok_or_else(|| {
                Error::llm(format!(
                    "stream choice {index} `tool_calls` must be an array, got {}",
                    json_kind(raw_calls)
                ))
            })?;
            for (call_position, raw_call) in calls.iter().enumerate() {
                let fragment = parse_tool_call_delta(raw_call, call_position)?;
                frame.push(delta(index, None, None, Some(fragment)));
            }
        }
        // A role/finish-reason-only frame still reports information the caller
        // needs, so it becomes one empty delta rather than disappearing.
        if frame.is_empty() && (role.is_some() || finish_reason.is_some()) {
            frame.push(delta(index, None, None, None));
        }

        if let Some(StreamChunk::Delta { role: slot, .. }) = frame.first_mut() {
            *slot = role;
        }
        if let Some(StreamChunk::Delta {
            finish_reason: slot,
            ..
        }) = frame.last_mut()
        {
            *slot = finish_reason;
        }

        chunks.extend(frame);
    }
    Ok(chunks)
}

/// Builds one delta chunk with every optional field spelled out.
fn delta(
    index: u32,
    content: Option<String>,
    reasoning: Option<String>,
    tool_call: Option<ToolCallDelta>,
) -> StreamChunk {
    StreamChunk::Delta {
        index,
        role: None,
        content,
        reasoning,
        tool_call,
        finish_reason: None,
    }
}

/// Parses one fragment of the `tool_calls` array.
fn parse_tool_call_delta(value: &Value, position: usize) -> Result<ToolCallDelta> {
    let object = value.as_object().ok_or_else(|| {
        Error::llm(format!(
            "stream tool call {position} must be a JSON object, got {}",
            json_kind(value)
        ))
    })?;
    let function = match object.get("function") {
        None | Some(Value::Null) => None,
        Some(Value::Object(function)) => Some(function),
        Some(other) => {
            return Err(Error::llm(format!(
                "stream tool call {position} `function` must be a JSON object, got {}",
                json_kind(other)
            )))
        }
    };

    Ok(ToolCallDelta {
        index: Some(
            object
                .get("index")
                .and_then(Value::as_u64)
                .and_then(|raw| u32::try_from(raw).ok())
                .unwrap_or(position as u32),
        ),
        id: object
            .get("id")
            .and_then(Value::as_str)
            .filter(|id| !id.is_empty())
            .map(str::to_owned),
        // The nested `function` object is what providers send; the flat shape is
        // accepted because a few gateways flatten the fragment.
        name: function
            .and_then(|function| text_field(function, "name"))
            .or_else(|| text_field(object, "name")),
        arguments: function
            .and_then(|function| text_field(function, "arguments"))
            .or_else(|| text_field(object, "arguments")),
    })
}

/// Reads a non-empty string field, so ``"content": ""`` and `null` both mean
/// "nothing in this frame".
fn text_field(object: &Map<String, Value>, key: &str) -> Option<String> {
    object
        .get(key)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
}

/// Builds a usage chunk for a frame that carries no choices.
fn usage_chunk(object: &Map<String, Value>) -> Vec<StreamChunk> {
    match object.get("usage") {
        Some(usage) if usage.is_object() => vec![StreamChunk::Usage(parse_usage(usage))],
        _ => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use openjiuwen_common::error::ErrorKind;
    use serde_json::json;

    use super::*;
    use crate::llm::model::{FinishReason, StreamAccumulator};
    use crate::prompt::Usage;

    fn frame(value: Value) -> Vec<StreamChunk> {
        parse_stream_chunk(&value).expect("the frame parses")
    }

    #[test]
    fn a_role_frame_emits_one_empty_delta() {
        let chunks = frame(json!({
            "id": "chatcmpl-1",
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]
        }));

        assert_eq!(
            chunks,
            vec![StreamChunk::Delta {
                index: 0,
                role: Some(Role::Assistant),
                content: None,
                reasoning: None,
                tool_call: None,
                finish_reason: None,
            }]
        );
    }

    #[test]
    fn content_and_reasoning_frames_emit_deltas() {
        assert_eq!(
            frame(json!({"choices": [{"index": 0, "delta": {"content": "Hel"}}]})),
            vec![StreamChunk::Delta {
                index: 0,
                role: None,
                content: Some("Hel".to_owned()),
                reasoning: None,
                tool_call: None,
                finish_reason: None,
            }]
        );
        assert_eq!(
            frame(json!({
                "choices": [{"index": 0, "delta": {"reasoning_content": "thinking"}}]
            })),
            vec![StreamChunk::Delta {
                index: 0,
                role: None,
                content: None,
                reasoning: Some("thinking".to_owned()),
                tool_call: None,
                finish_reason: None,
            }]
        );
    }

    #[test]
    fn content_and_finish_reason_share_one_delta() {
        let chunks = frame(json!({
            "choices": [{"index": 0, "delta": {"content": "done"}, "finish_reason": "stop"}]
        }));

        assert_eq!(
            chunks,
            vec![StreamChunk::Delta {
                index: 0,
                role: None,
                content: Some("done".to_owned()),
                reasoning: None,
                tool_call: None,
                finish_reason: Some(FinishReason::Stop),
            }]
        );
    }

    #[test]
    fn a_finish_reason_only_frame_emits_a_bare_delta() {
        let chunks = frame(json!({
            "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}]
        }));

        assert_eq!(
            chunks,
            vec![StreamChunk::Delta {
                index: 0,
                role: None,
                content: None,
                reasoning: None,
                tool_call: None,
                finish_reason: Some(FinishReason::Length),
            }]
        );
    }

    #[test]
    fn a_choice_index_is_preserved() {
        let chunks = frame(json!({
            "choices": [{"index": 1, "delta": {"content": "second"}}]
        }));

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].content(), Some("second"));
        match &chunks[0] {
            StreamChunk::Delta { index, .. } => assert_eq!(*index, 1),
            other => panic!("expected a delta, got {other:?}"),
        }
    }

    #[test]
    fn tool_call_fragments_carry_ids_names_arguments_and_indexes() {
        let chunks = frame(json!({
            "choices": [{
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": "{\"city\":"}
                        },
                        {
                            "index": 1,
                            "id": "call_2",
                            "type": "function",
                            "function": {"name": "get_time", "arguments": "{}"}
                        }
                    ]
                }
            }]
        }));

        assert_eq!(chunks.len(), 2);
        match (&chunks[0], &chunks[1]) {
            (
                StreamChunk::Delta {
                    tool_call: Some(first),
                    ..
                },
                StreamChunk::Delta {
                    tool_call: Some(second),
                    ..
                },
            ) => {
                assert_eq!(first.index, Some(0));
                assert_eq!(first.id.as_deref(), Some("call_1"));
                assert_eq!(first.name.as_deref(), Some("get_weather"));
                assert_eq!(first.arguments.as_deref(), Some("{\"city\":"));
                assert_eq!(second.index, Some(1));
                assert_eq!(second.name.as_deref(), Some("get_time"));
                assert_eq!(second.arguments.as_deref(), Some("{}"));
            }
            other => panic!("expected two tool-call deltas, got {other:?}"),
        }
    }

    #[test]
    fn interleaved_tool_call_fragments_merge_into_one_call() {
        let mut accumulator = StreamAccumulator::new();
        for value in [
            json!({"choices": [{"index": 0, "delta": {
                "role": "assistant",
                "tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                                "function": {"name": "get_weather", "arguments": ""}}]
            }}]}),
            json!({"choices": [{"index": 0, "delta": {
                "tool_calls": [{"index": 0, "function": {"arguments": "{\"city\":"}}]
            }}]}),
            json!({"choices": [{"index": 0, "delta": {
                "tool_calls": [{"index": 0, "function": {"arguments": "\"Paris\"}"}}]
            }}]}),
            json!({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}),
        ] {
            for chunk in frame(value) {
                accumulator.push(&chunk);
            }
        }

        let turn = accumulator.finish();
        assert_eq!(turn.message.tool_calls.len(), 1);
        assert_eq!(turn.message.tool_calls[0].id.as_deref(), Some("call_1"));
        assert_eq!(turn.message.tool_calls[0].name, "get_weather");
        assert_eq!(
            turn.message.tool_calls[0]
                .arguments_value()
                .expect("valid JSON"),
            json!({"city": "Paris"})
        );
        assert_eq!(turn.finish_reason, Some(FinishReason::ToolCalls));
    }

    #[test]
    fn a_usage_only_frame_yields_a_usage_chunk() {
        let chunks = frame(json!({
            "id": "chatcmpl-1",
            "choices": [],
            "usage": {
                "prompt_tokens": 9,
                "completion_tokens": 4,
                "total_tokens": 13,
                "completion_tokens_details": {"reasoning_tokens": 2}
            }
        }));

        assert_eq!(chunks.len(), 1);
        match &chunks[0] {
            StreamChunk::Usage(usage) => {
                assert_eq!(usage.input_tokens, 9);
                assert_eq!(usage.output_tokens, 4);
                assert_eq!(usage.total_tokens, 13);
                assert_eq!(usage.reasoning_tokens, 2);
            }
            other => panic!("expected usage, got {other:?}"),
        }
    }

    #[test]
    fn a_frame_without_choices_but_with_usage_yields_usage() {
        let chunks = frame(json!({"usage": {"prompt_tokens": 1, "total_tokens": 1}}));
        assert!(matches!(chunks.as_slice(), [StreamChunk::Usage(_)]));
    }

    #[test]
    fn an_empty_choices_array_without_usage_yields_nothing() {
        assert_eq!(frame(json!({"choices": []})), Vec::new());
        assert_eq!(frame(json!({"id": "keep-alive"})), Vec::new());
    }

    #[test]
    fn the_done_sentinel_emits_done_exactly_once() {
        let mut state = OpenAiStreamState::new();

        assert_eq!(
            state.handle_data("[DONE]").expect("ok"),
            vec![StreamChunk::Done]
        );
        assert_eq!(state.handle_data("[DONE]").expect("ok"), Vec::new());
        assert_eq!(
            state.finish(),
            Vec::new(),
            "finish after [DONE] emits nothing"
        );
    }

    #[test]
    fn trailing_whitespace_around_the_done_sentinel_is_tolerated() {
        let mut state = OpenAiStreamState::new();
        assert_eq!(
            state.handle_data("  [DONE] \n").expect("ok"),
            vec![StreamChunk::Done]
        );
    }

    #[test]
    fn finish_emits_done_for_a_stream_that_closed_without_it() {
        let mut state = OpenAiStreamState::new();
        let chunks = state.handle_data(r#"{"choices":[{"index":0,"delta":{"content":"hi"}}]}"#);
        assert_eq!(chunks.expect("ok").len(), 1);

        assert_eq!(state.finish(), vec![StreamChunk::Done]);
        assert_eq!(state.finish(), Vec::new());
    }

    #[test]
    fn payloads_after_done_are_ignored() {
        let mut state = OpenAiStreamState::new();
        state.handle_data("[DONE]").expect("ok");
        assert_eq!(
            state
                .handle_data(r#"{"choices":[{"index":0,"delta":{"content":"late"}}]}"#)
                .expect("ok"),
            Vec::new()
        );
    }

    #[test]
    fn an_empty_payload_yields_nothing() {
        let mut state = OpenAiStreamState::new();
        assert_eq!(state.handle_data("").expect("ok"), Vec::new());
        assert_eq!(state.handle_data("   ").expect("ok"), Vec::new());
    }

    #[test]
    fn malformed_json_is_skipped() {
        let mut state = OpenAiStreamState::new();
        assert_eq!(state.handle_data("{not json").expect("skipped"), Vec::new());
        let chunks = state
            .handle_data(r#"{"choices":[{"index":0,"delta":{"content":"still here"}}]}"#)
            .expect("ok");
        assert_eq!(chunks.len(), 1);
        assert_eq!(state.finish(), vec![StreamChunk::Done]);
    }

    #[test]
    fn a_non_object_frame_is_an_llm_error() {
        let error = parse_stream_chunk(&json!([1, 2, 3])).expect_err("must fail");
        assert_eq!(error.kind(), ErrorKind::Llm);
        assert!(error.to_string().contains("JSON object"), "{error}");
    }

    #[test]
    fn a_non_array_choices_value_is_an_llm_error() {
        let error = parse_stream_chunk(&json!({"choices": "nope"})).expect_err("must fail");
        assert_eq!(error.kind(), ErrorKind::Llm);
    }

    #[test]
    fn a_non_object_choice_is_an_llm_error() {
        let error = parse_stream_chunk(&json!({"choices": ["nope"]})).expect_err("must fail");
        assert_eq!(error.kind(), ErrorKind::Llm);
    }

    #[test]
    fn a_non_array_tool_calls_value_is_an_llm_error() {
        let error = parse_stream_chunk(&json!({
            "choices": [{"index": 0, "delta": {"tool_calls": "nope"}}]
        }))
        .expect_err("must fail");
        assert_eq!(error.kind(), ErrorKind::Llm);
    }

    #[test]
    fn a_provider_error_frame_is_an_llm_error() {
        let error = parse_stream_chunk(&json!({
            "error": {"message": "rate limited", "type": "rate_limit_error"}
        }))
        .expect_err("must fail");

        assert_eq!(error.kind(), ErrorKind::Llm);
        let message = error.to_string();
        assert!(message.contains("streaming error"), "{message}");
        assert!(
            !message.contains("rate limited"),
            "provider data is redacted: {message}"
        );
    }

    #[test]
    fn a_message_object_is_accepted_instead_of_delta() {
        let chunks = frame(json!({
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "final"},
                "finish_reason": "stop"
            }]
        }));

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].content(), Some("final"));
    }

    #[test]
    fn a_flat_tool_call_shape_is_accepted() {
        let chunks = frame(json!({
            "choices": [{"index": 0, "delta": {
                "tool_calls": [
                    {"index": 0, "id": "call_1", "name": "ping", "arguments": "{}"},
                    {"index": 1, "function": {"name": "pong", "arguments": ""}}
                ]
            }}]
        }));

        assert_eq!(chunks.len(), 2);
        match (&chunks[0], &chunks[1]) {
            (
                StreamChunk::Delta {
                    tool_call: Some(first),
                    ..
                },
                StreamChunk::Delta {
                    tool_call: Some(second),
                    ..
                },
            ) => {
                assert_eq!(first.id.as_deref(), Some("call_1"));
                assert_eq!(first.name.as_deref(), Some("ping"));
                assert_eq!(first.arguments.as_deref(), Some("{}"));
                assert_eq!(second.name.as_deref(), Some("pong"));
                assert_eq!(
                    second.arguments.as_deref(),
                    None,
                    "an empty argument fragment carries nothing"
                );
            }
            other => panic!("expected two tool calls, got {other:?}"),
        }
    }

    #[test]
    fn a_stream_assembles_into_the_expected_turn() {
        let mut state = OpenAiStreamState::new();
        let mut accumulator = StreamAccumulator::new();
        let frames = [
            r#"{"choices":[{"index":0,"delta":{"role":"assistant"}}]}"#,
            r#"{"choices":[{"index":0,"delta":{"content":"Hel"}}]}"#,
            r#"{"choices":[{"index":0,"delta":{"content":"lo"}}]}"#,
            r#"{"choices":[{"index":0,"delta":{"reasoning_content":"plan"}}]}"#,
            r#"{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#,
            r#"{"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}"#,
            "[DONE]",
        ];
        for payload in frames {
            for chunk in state.handle_data(payload).expect("every frame decodes") {
                accumulator.push(&chunk);
            }
        }
        for chunk in state.finish() {
            accumulator.push(&chunk);
        }

        let turn = accumulator.finish();
        assert_eq!(turn.message.text(), "Hello");
        assert_eq!(turn.message.reasoning_content.as_deref(), Some("plan"));
        assert_eq!(turn.finish_reason, Some(FinishReason::Stop));
        assert_eq!(turn.usage, Usage::new(3, 2));
    }
}
