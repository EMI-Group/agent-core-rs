//! `OpenAI` Responses stream decoding: one SSE payload at a time.
//!
//! Ports `ResponsesStreamParser` of
//! `openjiuwen/core/foundation/llm/utils/responses_utils.py`. Function-call
//! arguments stream as they are generated (`response.output_item.added` names the
//! call, each `response.function_call_arguments.delta` extends it), so a long tool
//! call shows progress instead of arriving all at once; `response.output_item.done`
//! repeats the full arguments, and the parser remembers what it already emitted per
//! item so the done event only contributes the never-streamed suffix.

use std::collections::BTreeMap;

use openjiuwen_common::error::{Error, Result};
use serde_json::Value;

use crate::llm::transport::redacted_body;
use crate::prompt::ToolCallDelta;

use super::super::model::{FinishReason, StreamChunk};
use super::response::parse_responses_usage;

/// Payload that terminates an OpenAI-compatible event stream.
const DONE_SENTINEL: &str = "[DONE]";

/// Message used when a failure event carries no detail.
const GENERIC_STREAM_ERROR: &str = "OpenAI account Responses stream failed.";

/// One function call whose arguments are being streamed.
#[derive(Debug, Clone, PartialEq, Eq)]
struct StreamedCall {
    /// Provider call id carried by the added item.
    call_id: String,
    /// Arguments text emitted to the caller so far.
    arguments: String,
}

/// Per-stream decoder state.
///
/// Feed every [`SseEvent::data`](crate::llm::sse::SseEvent::data) payload to
/// [`handle_data`](Self::handle_data), then call [`finish`](Self::finish) when the
/// HTTP body ends. A stream produces exactly one [`StreamChunk::Done`], whether the
/// provider sent `response.completed` or simply closed the connection, and any
/// event after the terminator is ignored.
///
/// The state also remembers the provider response id
/// ([`response_id`](Self::response_id)), which the transport stores per
/// conversation for a later `previous_response_id`.
#[derive(Debug, Clone, Default)]
pub struct ResponsesStreamState {
    /// Whether the terminal `Done` chunk has already been produced.
    done: bool,
    /// Arguments already emitted, keyed by provider response item id.
    streamed_calls: BTreeMap<String, StreamedCall>,
    /// Provider response id seen so far, if any event carried one.
    response_id: Option<String>,
}

impl ResponsesStreamState {
    /// Creates an empty state.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The provider response id the stream reported, when it reported one.
    ///
    /// The id travels on any event that carries a `response` object
    /// (`response.created`, `response.completed`, `response.incomplete`), and it is
    /// the value a follow-up call passes as `previous_response_id`.
    #[must_use]
    pub fn response_id(&self) -> Option<&str> {
        self.response_id.as_deref()
    }

    /// Decodes one SSE `data` payload into zero or more chunks.
    ///
    /// Payloads that are not valid JSON are logged with [`tracing::warn!`] and
    /// skipped, because a single undecodable keep-alive or vendor comment frame
    /// must not abort a whole generation. Events this crate does not model
    /// (`response.created`, content-part events, `ping`, ...) yield nothing.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Llm`] for a `response.failed` or `error` event, with the
    /// provider's message redacted.
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
                    target: "openjiuwen_foundation::llm::responses",
                    %error,
                    "skipping undecodable stream frame"
                );
                return Ok(Vec::new());
            }
        };
        let Some(object) = value.as_object() else {
            tracing::warn!(
                target: "openjiuwen_foundation::llm::responses",
                "skipping a stream frame that is not a JSON object"
            );
            return Ok(Vec::new());
        };

        let event_type = object
            .get("type")
            .and_then(Value::as_str)
            .or_else(|| object.get("event").and_then(Value::as_str))
            .unwrap_or_default();

        self.capture_response_id(object);

        match event_type {
            "response.output_text.delta" | "response.refusal.delta" => {
                Ok(text_delta(object, false))
            }
            "response.reasoning_text.delta" | "response.reasoning_summary_text.delta" => {
                Ok(text_delta(object, true))
            }
            "response.output_item.added" => Ok(self.added_function_call(object)),
            "response.function_call_arguments.delta" => Ok(self.arguments_delta(object)),
            "response.output_item.done" => Ok(self.done_output_item(object)),
            "response.completed" | "response.incomplete" => {
                Ok(self.terminal(object, event_type == "response.incomplete"))
            }
            "response.failed" | "error" => Err(stream_error(object)),
            _ => Ok(Vec::new()),
        }
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

    /// Records the provider response id an event carries, when it carries one.
    ///
    /// Both lifecycle events repeat the same id, so the last one simply wins; a
    /// blank id never overwrites a known one.
    fn capture_response_id(&mut self, event: &serde_json::Map<String, Value>) {
        let Some(id) = event
            .get("response")
            .and_then(Value::as_object)
            .and_then(|response| response.get("id"))
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|id| !id.is_empty())
        else {
            return;
        };
        self.response_id = Some(id.to_owned());
    }

    /// Handles `response.output_item.added` for a function call.
    ///
    /// An item without an id cannot be attributed to later deltas, so it yields
    /// nothing and the done event delivers the whole call instead.
    fn added_function_call(&mut self, event: &serde_json::Map<String, Value>) -> Vec<StreamChunk> {
        let Some(item) = event.get("item").and_then(Value::as_object) else {
            return Vec::new();
        };
        if item.get("type").and_then(Value::as_str) != Some("function_call") {
            return Vec::new();
        }
        let item_id = item
            .get("id")
            .and_then(Value::as_str)
            .map(str::trim)
            .unwrap_or_default();
        if item_id.is_empty() {
            return Vec::new();
        }

        let call_id = call_id(item);
        let arguments = argument_string(item.get("arguments"), "");
        self.streamed_calls.insert(
            item_id.to_owned(),
            StreamedCall {
                call_id: call_id.clone(),
                arguments: arguments.clone(),
            },
        );
        vec![tool_delta(
            Some(call_id),
            item.get("name").and_then(Value::as_str),
            Some(arguments),
            None,
        )]
    }

    /// Handles `response.function_call_arguments.delta`.
    fn arguments_delta(&mut self, event: &serde_json::Map<String, Value>) -> Vec<StreamChunk> {
        let item_id = event
            .get("item_id")
            .and_then(Value::as_str)
            .map(str::trim)
            .unwrap_or_default();
        let delta = event.get("delta").and_then(Value::as_str).unwrap_or("");
        if delta.is_empty() {
            return Vec::new();
        }
        let Some(streamed) = self.streamed_calls.get_mut(item_id) else {
            return Vec::new();
        };
        streamed.arguments.push_str(delta);
        vec![tool_delta(
            Some(streamed.call_id.clone()),
            None,
            Some(delta.to_owned()),
            None,
        )]
    }

    /// Handles `response.output_item.done` for a function call.
    ///
    /// The done event repeats the full arguments; only the suffix that was never
    /// streamed is emitted, and the call is marked finished with
    /// [`FinishReason::ToolCalls`].
    fn done_output_item(&mut self, event: &serde_json::Map<String, Value>) -> Vec<StreamChunk> {
        let Some(item) = event.get("item").and_then(Value::as_object) else {
            return Vec::new();
        };
        if item.get("type").and_then(Value::as_str) != Some("function_call") {
            return Vec::new();
        }

        let item_id = item
            .get("id")
            .and_then(Value::as_str)
            .map(str::trim)
            .unwrap_or_default();
        let name = item.get("name").and_then(Value::as_str);
        let full_arguments = argument_string(item.get("arguments"), "{}");

        match self.streamed_calls.remove(item_id) {
            None => vec![tool_delta(
                Some(call_id(item)),
                name,
                Some(full_arguments),
                Some(FinishReason::ToolCalls),
            )],
            Some(streamed) => {
                let remainder = full_arguments
                    .strip_prefix(streamed.arguments.as_str())
                    .unwrap_or_else(|| {
                        // The streamed fragments are already concatenated
                        // downstream and cannot be retracted; keep them rather
                        // than duplicate.
                        tracing::warn!(
                            target: "openjiuwen_foundation::llm::responses",
                            item = item_id,
                            "function call arguments diverged from the streamed deltas; \
                             keeping the streamed arguments"
                        );
                        ""
                    });
                vec![tool_delta(
                    Some(streamed.call_id),
                    None,
                    Some(remainder.to_owned()),
                    Some(FinishReason::ToolCalls),
                )]
            }
        }
    }

    /// Handles `response.completed` / `response.incomplete`.
    fn terminal(
        &mut self,
        event: &serde_json::Map<String, Value>,
        incomplete: bool,
    ) -> Vec<StreamChunk> {
        let response = event.get("response").and_then(Value::as_object);
        let usage = response
            .and_then(|response| response.get("usage"))
            .or_else(|| event.get("usage"));
        let finish_reason = if incomplete {
            FinishReason::Length
        } else {
            FinishReason::Stop
        };

        let mut chunks = vec![StreamChunk::Delta {
            index: 0,
            role: None,
            content: None,
            reasoning: None,
            tool_call: None,
            finish_reason: Some(finish_reason),
        }];
        if let Some(usage) = usage {
            chunks.push(StreamChunk::Usage(parse_responses_usage(usage)));
        }
        self.done = true;
        chunks.push(StreamChunk::Done);
        chunks
    }
}

/// Builds a content or reasoning delta from a `*.delta` event.
fn text_delta(event: &serde_json::Map<String, Value>, is_reasoning: bool) -> Vec<StreamChunk> {
    let delta = event.get("delta").and_then(Value::as_str).unwrap_or("");
    if delta.is_empty() {
        return Vec::new();
    }
    vec![StreamChunk::Delta {
        index: 0,
        role: None,
        content: (!is_reasoning).then(|| delta.to_owned()),
        reasoning: is_reasoning.then(|| delta.to_owned()),
        tool_call: None,
        finish_reason: None,
    }]
}

/// Builds a tool-call delta.
fn tool_delta(
    id: Option<String>,
    name: Option<&str>,
    arguments: Option<String>,
    finish_reason: Option<FinishReason>,
) -> StreamChunk {
    StreamChunk::Delta {
        index: 0,
        role: None,
        content: None,
        reasoning: None,
        tool_call: Some(ToolCallDelta {
            index: Some(0),
            id,
            name: name.map(str::to_owned),
            arguments,
        }),
        finish_reason,
    }
}

/// The call id of a function-call item: `call_id`, else the item id.
fn call_id(item: &serde_json::Map<String, Value>) -> String {
    item.get("call_id")
        .and_then(Value::as_str)
        .filter(|id| !id.trim().is_empty())
        .or_else(|| item.get("id").and_then(Value::as_str))
        .unwrap_or_default()
        .to_owned()
}

/// Renders a wire `arguments` value as JSON text.
fn argument_string(value: Option<&Value>, fallback: &str) -> String {
    match value {
        None | Some(Value::Null) => fallback.to_owned(),
        Some(Value::String(text)) => text.clone(),
        Some(other) => other.to_string(),
    }
}

/// Builds the [`Error::Llm`] for a failed stream event.
fn stream_error(event: &serde_json::Map<String, Value>) -> Error {
    let message = event
        .get("message")
        .and_then(Value::as_str)
        .filter(|message| !message.is_empty())
        .or_else(|| {
            event
                .get("error")
                .and_then(|error| error.get("message"))
                .and_then(Value::as_str)
                .filter(|message| !message.is_empty())
        })
        .unwrap_or(GENERIC_STREAM_ERROR);
    Error::llm(format!(
        "OpenAI Responses stream failed: {}",
        redacted_body(message)
    ))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn data(value: Value) -> String {
        value.to_string()
    }

    #[test]
    fn text_and_reasoning_deltas_become_chunks() {
        let mut state = ResponsesStreamState::new();
        let chunks = state
            .handle_data(&data(json!({
                "type": "response.output_text.delta",
                "delta": "Hel"
            })))
            .expect("text delta");
        assert_eq!(
            chunks,
            vec![StreamChunk::Delta {
                index: 0,
                role: None,
                content: Some("Hel".to_owned()),
                reasoning: None,
                tool_call: None,
                finish_reason: None,
            }]
        );

        let chunks = state
            .handle_data(&data(json!({
                "type": "response.reasoning_summary_text.delta",
                "delta": "why"
            })))
            .expect("reasoning delta");
        assert_eq!(chunks[0].content(), None);
        match &chunks[0] {
            StreamChunk::Delta { reasoning, .. } => {
                assert_eq!(reasoning.as_deref(), Some("why"));
            }
            other => panic!("expected a delta, got {other:?}"),
        }
    }

    #[test]
    fn empty_deltas_and_unknown_events_are_ignored() {
        let mut state = ResponsesStreamState::new();
        assert_eq!(
            state
                .handle_data(&data(
                    json!({"type": "response.output_text.delta", "delta": ""})
                ))
                .expect("empty delta"),
            Vec::<StreamChunk>::new()
        );
        for event_type in [
            "response.created",
            "response.in_progress",
            "response.content_part.added",
            "response.output_text.done",
            "ping",
        ] {
            assert_eq!(
                state
                    .handle_data(&data(json!({"type": event_type})))
                    .expect("ignored event"),
                Vec::<StreamChunk>::new(),
                "{event_type} must be ignored"
            );
        }
    }

    #[test]
    fn a_full_function_call_sequence_accumulates_one_call() {
        let mut state = ResponsesStreamState::new();
        let mut chunks = Vec::new();
        for event in [
            json!({"type": "response.output_item.added", "item": {
                "id": "fc_1", "type": "function_call", "call_id": "call_1",
                "name": "get_weather", "arguments": ""
            }}),
            json!({"type": "response.function_call_arguments.delta",
                   "item_id": "fc_1", "delta": "{\"city\":"}),
            json!({"type": "response.function_call_arguments.delta",
                   "item_id": "fc_1", "delta": "\"Paris\"}"}),
            json!({"type": "response.output_item.done", "item": {
                "id": "fc_1", "type": "function_call", "call_id": "call_1",
                "name": "get_weather", "arguments": "{\"city\":\"Paris\"}"
            }}),
            json!({"type": "response.completed", "response": {
                "id": "resp_1",
                "usage": {"input_tokens": 5, "output_tokens": 2}
            }}),
        ] {
            chunks.extend(state.handle_data(&data(event)).expect("event decodes"));
        }
        chunks.extend(state.finish());

        assert_eq!(
            chunks
                .iter()
                .filter(|chunk| matches!(chunk, StreamChunk::Done))
                .count(),
            1
        );
        let mut accumulator = crate::llm::model::StreamAccumulator::new();
        for chunk in &chunks {
            accumulator.push(chunk);
        }
        let turn = accumulator.finish();
        assert_eq!(turn.message.tool_calls.len(), 1);
        assert_eq!(turn.message.tool_calls[0].id.as_deref(), Some("call_1"));
        assert_eq!(turn.message.tool_calls[0].name, "get_weather");
        assert_eq!(turn.message.tool_calls[0].arguments, "{\"city\":\"Paris\"}");
        assert_eq!(turn.usage.input_tokens, 5);
        assert_eq!(turn.usage.output_tokens, 2);
        assert_eq!(turn.finish_reason, Some(FinishReason::Stop));
    }

    #[test]
    fn a_done_event_without_deltas_delivers_the_whole_call() {
        let mut state = ResponsesStreamState::new();
        let chunks = state
            .handle_data(&data(json!({"type": "response.output_item.done", "item": {
                "id": "fc_9", "type": "function_call", "call_id": "call_9",
                "name": "now", "arguments": "{}"
            }})))
            .expect("done event");
        match &chunks[0] {
            StreamChunk::Delta {
                tool_call: Some(call),
                finish_reason,
                ..
            } => {
                assert_eq!(call.id.as_deref(), Some("call_9"));
                assert_eq!(call.name.as_deref(), Some("now"));
                assert_eq!(call.arguments.as_deref(), Some("{}"));
                assert_eq!(finish_reason, &Some(FinishReason::ToolCalls));
            }
            other => panic!("expected a tool-call delta, got {other:?}"),
        }
    }

    #[test]
    fn a_diverging_done_event_keeps_the_streamed_arguments() {
        let mut state = ResponsesStreamState::new();
        let _ = state
            .handle_data(&data(json!({"type": "response.output_item.added", "item": {
                "id": "fc_1", "type": "function_call", "call_id": "call_1", "name": "f", "arguments": ""
            }})))
            .expect("added");
        let _ = state
            .handle_data(&data(
                json!({"type": "response.function_call_arguments.delta",
                "item_id": "fc_1", "delta": "{\"a\":"}),
            ))
            .expect("delta");
        let chunks = state
            .handle_data(&data(json!({"type": "response.output_item.done", "item": {
                "id": "fc_1", "type": "function_call", "call_id": "call_1",
                "name": "f", "arguments": "{\"b\":1}"
            }})))
            .expect("done");
        match &chunks[0] {
            StreamChunk::Delta {
                tool_call: Some(call),
                ..
            } => assert_eq!(call.arguments.as_deref(), Some("")),
            other => panic!("expected a tool-call delta, got {other:?}"),
        }
    }

    #[test]
    fn completed_reports_usage_then_done_and_incomplete_reports_length() {
        let mut state = ResponsesStreamState::new();
        let chunks = state
            .handle_data(&data(json!({"type": "response.completed", "response": {
                "usage": {"input_tokens": 3, "output_tokens": 1, "total_tokens": 4,
                          "input_tokens_details": {"cached_tokens": 2}}
            }})))
            .expect("completed");
        assert_eq!(chunks.len(), 3);
        assert_eq!(
            chunks
                .iter()
                .filter(|chunk| matches!(chunk, StreamChunk::Done))
                .count(),
            1
        );
        match &chunks[1] {
            StreamChunk::Usage(usage) => {
                assert_eq!(usage.input_tokens, 3);
                assert_eq!(usage.total_tokens, 4);
                assert_eq!(usage.cache_tokens, 2);
            }
            other => panic!("expected usage, got {other:?}"),
        }
        assert!(state.finish().is_empty(), "the terminator emits Done once");

        let mut state = ResponsesStreamState::new();
        let chunks = state
            .handle_data(&data(json!({"type": "response.incomplete"})))
            .expect("incomplete");
        match &chunks[0] {
            StreamChunk::Delta { finish_reason, .. } => {
                assert_eq!(finish_reason, &Some(FinishReason::Length));
            }
            other => panic!("expected a terminal delta, got {other:?}"),
        }
    }

    #[test]
    fn failed_and_error_events_are_llm_errors() {
        let mut state = ResponsesStreamState::new();
        for event in [
            json!({"type": "response.failed", "response": {"error": {"message": "boom"}}}),
            json!({"type": "error", "message": "also boom"}),
        ] {
            let error = state
                .handle_data(&data(event))
                .expect_err("failures surface as errors");
            assert_eq!(error.kind(), openjiuwen_common::error::ErrorKind::Llm);
            assert!(error.to_string().contains("OpenAI Responses stream failed"));
        }
    }

    #[test]
    fn malformed_frames_are_skipped() {
        let mut state = ResponsesStreamState::new();
        assert_eq!(
            state.handle_data("{not json").expect("skipped"),
            Vec::<StreamChunk>::new()
        );
        assert_eq!(
            state.handle_data("[]").expect("skipped"),
            Vec::<StreamChunk>::new()
        );
        assert_eq!(
            state.handle_data("  ").expect("skipped"),
            Vec::<StreamChunk>::new()
        );
        assert_eq!(state.finish(), vec![StreamChunk::Done]);
        assert_eq!(
            state.finish(),
            Vec::<StreamChunk>::new(),
            "Done is emitted exactly once"
        );

        let mut state = ResponsesStreamState::new();
        assert_eq!(
            state.handle_data("[DONE]").expect("done"),
            vec![StreamChunk::Done]
        );
        assert_eq!(state.finish(), Vec::<StreamChunk>::new());
    }
}
