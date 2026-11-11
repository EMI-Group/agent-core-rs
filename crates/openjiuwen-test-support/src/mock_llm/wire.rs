//! OpenAI-compatible wire projection: tolerant request parsing and the JSON/SSE
//! shapes the real `OpenAiCompatibleClient` parses back.
//!
//! Ports the response construction of
//! `openjiuwen/core/foundation/llm/model_clients/openai_model_client.py` into
//! its inverse: [`completion_body`] produces a non-streaming chat completion,
//! [`sse_frames`] produces the `data: <json>\n\n` frames of a stream, and
//! [`usage_json`] produces the `prompt_tokens`/`completion_tokens` counters the
//! client's `parse_usage` reads back.
//!
//! Everything is deterministic: `created` is always `0`, ids are counter-based,
//! and no clock or randomness is consulted.

use openjiuwen_foundation::llm::{FinishReason, StreamAccumulator, StreamChunk};
use openjiuwen_foundation::prompt::{ToolCall, ToolCallDelta, Usage};
use serde::Deserialize;
use serde_json::{json, Map, Value};

use crate::internal::FUNCTION_KIND;
use crate::responses::{MockCompletion, MockToolCall};

/// The assistant role name in an `OpenAI` message.
const ROLE_ASSISTANT: &str = "assistant";

/// The `OpenAI` error type for a malformed or unroutable request.
const INVALID_REQUEST_ERROR: &str = "invalid_request_error";

/// The final SSE payload of every mock stream.
const DONE_FRAME: &str = "data: [DONE]\n\n";

/// A tolerant view of the request body.
///
/// Every field defaults, unknown fields are ignored, and a wrongly typed field
/// degrades to its default through [`parse_request`]; the raw body remains the
/// authoritative record, so this struct only exists to answer "was a stream
/// requested, and for which model".
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
#[allow(dead_code)] // `messages`/`tools`/`tool_choice`/`stream_options` are parsed for tolerance only.
pub(crate) struct ChatCompletionRequest {
    /// Model the client asked for.
    pub model: Option<String>,
    /// Conversation sent by the client.
    pub messages: Vec<Value>,
    /// Whether the client asked for a streamed answer.
    pub stream: bool,
    /// Streaming options (`include_usage`); present but unused.
    pub stream_options: Option<Value>,
    /// Tool declarations sent by the client.
    pub tools: Option<Vec<Value>>,
    /// Tool-selection hint sent by the client.
    pub tool_choice: Option<Value>,
}

/// Parses a request body, or returns `None` when it is not a JSON object.
///
/// A wrong field type degrades to the default instead of failing, mirroring the
/// tolerance of a real provider: only a non-object payload is rejected.
pub(crate) fn parse_request(value: &Value) -> Option<ChatCompletionRequest> {
    if !value.is_object() {
        return None;
    }
    Some(serde_json::from_value(value.clone()).unwrap_or_default())
}

/// A scripted HTTP error response.
///
/// [`MockHttpError::new`] builds the `OpenAI` error envelope
/// (`{"error":{"message","type","param","code"}}`); [`MockHttpError::with_body`]
/// scripts an arbitrary body instead.
#[derive(Debug, Clone)]
pub struct MockHttpError {
    /// HTTP status code to answer with.
    pub status: u16,
    /// JSON body to answer with.
    pub body: Value,
}

impl MockHttpError {
    /// An error with the given status and `OpenAI` error envelope.
    #[must_use]
    pub fn new(status: u16, message: impl Into<String>) -> Self {
        Self {
            status,
            body: error_envelope(error_type_for_status(status), &message.into()),
        }
    }

    /// An error with an explicit body, for vendor-specific error shapes.
    #[must_use]
    pub fn with_body(status: u16, body: Value) -> Self {
        Self { status, body }
    }

    /// A `429 Too Many Requests` error.
    #[must_use]
    pub fn rate_limited() -> Self {
        Self::new(429, "rate limit exceeded")
    }

    /// A `401 Unauthorized` error.
    #[must_use]
    pub fn unauthorized() -> Self {
        Self::new(401, "invalid api key")
    }

    /// A `500 Internal Server Error` with the given message.
    #[must_use]
    pub fn server_error(message: impl Into<String>) -> Self {
        Self::new(500, message)
    }

    /// A `404 Not Found` error.
    #[must_use]
    pub fn not_found() -> Self {
        Self::new(404, "not found")
    }
}

/// The `OpenAI` error type that matches a status code.
fn error_type_for_status(status: u16) -> &'static str {
    match status {
        401 => "authentication_error",
        403 => "permission_error",
        429 => "rate_limit_error",
        500..=599 => "server_error",
        _ => INVALID_REQUEST_ERROR,
    }
}

/// Builds the `OpenAI` error envelope.
pub(crate) fn error_envelope(error_type: &str, message: &str) -> Value {
    json!({
        "error": {
            "message": message,
            "type": error_type,
            "param": Value::Null,
            "code": Value::Null,
        }
    })
}

/// The `GET /health` body.
pub(crate) fn health_body() -> Value {
    json!({"status": "ok"})
}

/// The body served for an unknown route or a wrong method.
pub(crate) fn not_found_body() -> Value {
    error_envelope(INVALID_REQUEST_ERROR, "not found")
}

/// The body served for a request that is not a JSON object.
pub(crate) fn bad_request_body(detail: &str) -> Value {
    error_envelope(
        INVALID_REQUEST_ERROR,
        &format!("invalid request body: {detail}"),
    )
}

/// Projects a scripted completion to a non-streaming chat completion.
///
/// `usage` is always present, `content` is `null` when the turn has no text, and
/// the tool calls use the `{"id","type":"function","function":{...}}` shape.
pub(crate) fn completion_body(id: &str, model: &str, completion: &MockCompletion) -> Value {
    let tool_calls = completion.tool_calls.iter().map(wire_tool_call).collect();
    let message = assistant_message(
        completion.content.as_deref(),
        completion.reasoning.as_deref(),
        tool_calls,
    );
    json!({
        "id": id,
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": completion.finish_reason.as_str(),
        }],
        "usage": usage_json(&completion.usage),
    })
}

/// Projects a scripted frame sequence to the same non-streaming body shape.
///
/// The frames are folded with [`StreamAccumulator`], so a request without
/// `"stream": true` still answers with the content, tool calls, and usage the
/// scripted frames carry; an absent finish reason defaults to `"stop"`.
pub(crate) fn chunks_body(id: &str, model: &str, chunks: &[StreamChunk]) -> Value {
    let mut accumulator = StreamAccumulator::new();
    for chunk in chunks {
        accumulator.push(chunk);
    }
    let turn = accumulator.finish();

    let text = turn.message.text().into_owned();
    let content = if text.is_empty() {
        None
    } else {
        Some(text.as_str())
    };
    let tool_calls = turn
        .message
        .tool_calls
        .iter()
        .map(ToolCall::to_openai)
        .collect();
    let message = assistant_message(
        content,
        turn.message.reasoning_content.as_deref(),
        tool_calls,
    );
    let finish_reason = turn.finish_reason.unwrap_or(FinishReason::Stop);

    json!({
        "id": id,
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason.as_str(),
        }],
        "usage": usage_json(&turn.usage),
    })
}

/// Builds the assistant message object shared by both non-streaming projections.
fn assistant_message(
    content: Option<&str>,
    reasoning: Option<&str>,
    tool_calls: Vec<Value>,
) -> Value {
    let mut message = Map::new();
    message.insert("role".to_owned(), Value::String(ROLE_ASSISTANT.to_owned()));
    message.insert(
        "content".to_owned(),
        content.map_or(Value::Null, |text| Value::String(text.to_owned())),
    );
    if let Some(reasoning) = reasoning {
        message.insert(
            "reasoning_content".to_owned(),
            Value::String(reasoning.to_owned()),
        );
    }
    if !tool_calls.is_empty() {
        message.insert("tool_calls".to_owned(), Value::Array(tool_calls));
    }
    Value::Object(message)
}

/// One `tool_calls[]` entry of a non-streaming assistant message.
fn wire_tool_call(call: &MockToolCall) -> Value {
    json!({
        "id": call.id.clone(),
        "type": FUNCTION_KIND,
        "function": {
            "name": call.name.clone(),
            "arguments": call.arguments.clone(),
        },
    })
}

/// Projects a frame sequence to SSE frames, terminated by `data: [DONE]\n\n`.
///
/// [`StreamChunk::Done`] never becomes its own JSON frame: it only marks the end
/// of the sequence, and exactly one `[DONE]` sentinel is appended after the last
/// frame.
pub(crate) fn sse_frames(id: &str, model: &str, chunks: &[StreamChunk]) -> Vec<String> {
    let mut frames = Vec::with_capacity(chunks.len().saturating_add(1));
    for chunk in chunks {
        if let Some(frame) = chunk_frame(id, model, chunk) {
            frames.push(format!("data: {frame}\n\n"));
        }
    }
    frames.push(DONE_FRAME.to_owned());
    frames
}

/// Projects one frame to its `data:` payload, or `None` for the sentinel chunk.
fn chunk_frame(id: &str, model: &str, chunk: &StreamChunk) -> Option<Value> {
    match chunk {
        StreamChunk::Delta {
            index,
            role,
            content,
            reasoning,
            tool_call,
            finish_reason,
        } => {
            let mut delta = Map::new();
            if let Some(role) = role {
                delta.insert("role".to_owned(), Value::String(role.as_str().to_owned()));
            }
            if let Some(content) = content {
                delta.insert("content".to_owned(), Value::String(content.clone()));
            }
            if let Some(reasoning) = reasoning {
                delta.insert(
                    "reasoning_content".to_owned(),
                    Value::String(reasoning.clone()),
                );
            }
            if let Some(fragment) = tool_call {
                delta.insert(
                    "tool_calls".to_owned(),
                    Value::Array(vec![tool_call_delta(fragment, *index)]),
                );
            }
            Some(json!({
                "id": id,
                "object": "chat.completion.chunk",
                "created": 0,
                "model": model,
                "choices": [{
                    "index": index,
                    "delta": Value::Object(delta),
                    "finish_reason": finish_reason.as_ref().map(FinishReason::as_str),
                }],
            }))
        }
        // `stream_options.include_usage` puts the accounting on a frame that has
        // no choices at all, and the client's parser reads it from there.
        StreamChunk::Usage(usage) => Some(json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": 0,
            "model": model,
            "choices": [],
            "usage": usage_json(usage),
        })),
        StreamChunk::Done => None,
    }
}

/// One `tool_calls[]` fragment of a delta.
///
/// Absent fragments are omitted so a test can script partial tool-call deltas the
/// way a provider streams them; `function` disappears when it would be empty.
fn tool_call_delta(fragment: &ToolCallDelta, choice_index: u32) -> Value {
    let mut entry = Map::new();
    entry.insert(
        "index".to_owned(),
        Value::from(fragment.index.unwrap_or(choice_index)),
    );
    if let Some(id) = &fragment.id {
        entry.insert("id".to_owned(), Value::String(id.clone()));
    }
    entry.insert("type".to_owned(), Value::String(FUNCTION_KIND.to_owned()));

    let mut function = Map::new();
    if let Some(name) = &fragment.name {
        function.insert("name".to_owned(), Value::String(name.clone()));
    }
    if let Some(arguments) = &fragment.arguments {
        function.insert("arguments".to_owned(), Value::String(arguments.clone()));
    }
    if !function.is_empty() {
        entry.insert("function".to_owned(), Value::Object(function));
    }
    Value::Object(entry)
}

/// Projects a [`Usage`] to the `OpenAI` counter object.
///
/// `prompt_tokens`/`completion_tokens`/`total_tokens` are always present; the
/// nested detail objects only appear when the corresponding counter is non-zero;
/// provider-specific counters from [`Usage::extra`] are merged at the top level.
pub(crate) fn usage_json(usage: &Usage) -> Value {
    let mut object = Map::new();
    for (key, value) in &usage.extra {
        object.insert(key.clone(), value.clone());
    }
    object.insert("prompt_tokens".to_owned(), Value::from(usage.input_tokens));
    object.insert(
        "completion_tokens".to_owned(),
        Value::from(usage.output_tokens),
    );
    object.insert("total_tokens".to_owned(), Value::from(usage.total_tokens));
    if usage.cache_tokens != 0 {
        object.insert(
            "prompt_tokens_details".to_owned(),
            json!({"cached_tokens": usage.cache_tokens}),
        );
    }
    if usage.reasoning_tokens != 0 {
        object.insert(
            "completion_tokens_details".to_owned(),
            json!({"reasoning_tokens": usage.reasoning_tokens}),
        );
    }
    Value::Object(object)
}
