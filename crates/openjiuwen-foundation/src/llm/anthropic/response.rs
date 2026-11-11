//! Anthropic Messages response parsing.
//!
//! Converts the JSON body of `POST {base}/v1/messages` into the provider-neutral
//! [`ChatResponse`]. Text blocks concatenate into the message content, `tool_use`
//! blocks become [`ToolCall`]s whose `arguments` are the JSON text of the block's
//! `input`, and Anthropic's `stop_reason` and cache-aware usage are normalized so
//! every layer above the foundation sees the same accounting as with the
//! OpenAI-compatible client.

use serde_json::{Map, Value};

use crate::llm::model::{ChatResponse, Choice, FinishReason};
use crate::prompt::{BaseMessage, MessageContent, ToolCall, Usage};
use openjiuwen_common::error::{Error, Result};

/// The only tool-call kind the built-in providers emit.
const FUNCTION_KIND: &str = "function";

/// Parses one Anthropic Messages response body.
///
/// Missing `id`, `model`, `content`, `usage`, or `stop_reason` are tolerated and
/// become empty values; a payload that is not a JSON object is an
/// [`Error::Llm`] because it cannot be a provider response.
///
/// # Errors
///
/// Returns [`Error::Llm`] when the payload is not a JSON object.
pub fn parse_messages_response(value: &Value) -> Result<ChatResponse> {
    let object = value.as_object().ok_or_else(|| {
        Error::llm(format!(
            "Anthropic response must be a JSON object, found {}",
            json_kind(value)
        ))
    })?;

    let id = object
        .get("id")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();
    let model = object
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();

    let mut text = String::new();
    let mut tool_calls = Vec::new();
    if let Some(blocks) = object.get("content").and_then(Value::as_array) {
        for block in blocks {
            let Some(block) = block.as_object() else {
                continue;
            };
            match block.get("type").and_then(Value::as_str) {
                Some("text") => {
                    if let Some(fragment) = block.get("text").and_then(Value::as_str) {
                        text.push_str(fragment);
                    }
                }
                Some("tool_use") => {
                    let index = u32::try_from(tool_calls.len()).unwrap_or(u32::MAX);
                    tool_calls.push(parse_tool_use(block, index));
                }
                // `thinking` and `redacted_thinking` blocks carry no displayable
                // assistant text and are not replayed by this client.
                _ => {}
            }
        }
    }

    let mut message = BaseMessage::assistant(MessageContent::Text(text));
    message.tool_calls = tool_calls;

    let finish_reason = object
        .get("stop_reason")
        .and_then(Value::as_str)
        .map(finish_reason_from_stop_reason);
    let usage = object
        .get("usage")
        .map_or_else(Usage::default, parse_anthropic_usage);

    Ok(ChatResponse {
        id,
        model,
        choices: vec![Choice {
            index: 0,
            message,
            finish_reason,
        }],
        usage,
    })
}

/// Parses an Anthropic `usage` object into the shared [`Usage`].
///
/// Anthropic reports the uncached prompt separately from cache reads and cache
/// writes, so `input_tokens` is their sum — the total prompt the model saw.
/// `cache_tokens` is the cache-read part (the cheap one), and `total_tokens` is
/// filled in by [`Usage::normalized`]. Unknown counters are ignored.
#[must_use]
pub fn parse_anthropic_usage(value: &Value) -> Usage {
    let Some(object) = value.as_object() else {
        return Usage::default();
    };

    let count = |key: &str| object.get(key).and_then(Value::as_u64).unwrap_or(0);
    let cache_read = count("cache_read_input_tokens");
    let cache_creation = count("cache_creation_input_tokens");
    let input_tokens = count("input_tokens")
        .saturating_add(cache_read)
        .saturating_add(cache_creation);
    let output_tokens = count("output_tokens");

    Usage {
        input_tokens,
        output_tokens,
        cache_tokens: cache_read,
        ..Usage::default()
    }
    .normalized()
}

/// Maps an Anthropic `stop_reason` onto the shared [`FinishReason`].
pub(crate) fn finish_reason_from_stop_reason(reason: &str) -> FinishReason {
    match reason {
        "tool_use" => FinishReason::ToolCalls,
        "end_turn" | "stop_sequence" => FinishReason::Stop,
        "max_tokens" => FinishReason::Length,
        other => FinishReason::Other(other.to_owned()),
    }
}

/// Parses one `tool_use` block into a [`ToolCall`].
fn parse_tool_use(block: &Map<String, Value>, index: u32) -> ToolCall {
    let id = block
        .get("id")
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty())
        .map(str::to_owned);
    let name = block
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();
    let input = block
        .get("input")
        .cloned()
        .unwrap_or_else(|| Value::Object(Map::new()));

    ToolCall {
        id,
        kind: FUNCTION_KIND.to_owned(),
        name,
        arguments: input.to_string(),
        index: Some(index),
    }
}

/// Short JSON kind name used in error messages.
fn json_kind(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "an array",
        Value::Object(_) => "an object",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use openjiuwen_common::error::ErrorKind;
    use serde_json::json;

    fn parse(value: Value) -> ChatResponse {
        parse_messages_response(&value).expect("valid response")
    }

    #[test]
    fn text_blocks_concatenate_in_order() {
        let response = parse(json!({
            "id": "msg_1",
            "model": "claude-sonnet-4",
            "content": [
                {"type": "text", "text": "hello "},
                {"type": "thinking", "thinking": "internal"},
                {"type": "text", "text": "world"},
            ],
            "stop_reason": "end_turn",
        }));

        assert_eq!(response.id, "msg_1");
        assert_eq!(response.model, "claude-sonnet-4");
        let message = response.message().expect("one choice");
        assert_eq!(message.role, crate::prompt::Role::Assistant);
        assert_eq!(message.text(), "hello world");
        assert!(message.tool_calls.is_empty(), "no tool calls expected");
        assert_eq!(response.finish_reason(), Some(&FinishReason::Stop));
    }

    #[test]
    fn tool_use_blocks_become_function_calls_with_json_arguments() {
        let response = parse(json!({
            "content": [
                {"type": "tool_use", "id": "call_1", "name": "get_weather", "input": {"city": "Paris", "days": 2}},
                {"type": "text", "text": "between"},
                {"type": "tool_use", "name": "get_time", "input": {}},
            ],
            "stop_reason": "tool_use",
        }));

        let message = response.message().expect("one choice");
        assert_eq!(message.text(), "between");
        assert_eq!(message.tool_calls.len(), 2);
        assert_eq!(message.tool_calls[0].id.as_deref(), Some("call_1"));
        assert_eq!(message.tool_calls[0].kind, "function");
        assert_eq!(message.tool_calls[0].name, "get_weather");
        assert_eq!(
            message.tool_calls[0].arguments,
            "{\"city\":\"Paris\",\"days\":2}"
        );
        assert_eq!(message.tool_calls[0].index, Some(0));
        assert_eq!(
            message.tool_calls[0].arguments_value().expect("json"),
            json!({"city": "Paris", "days": 2})
        );
        assert_eq!(message.tool_calls[1].id, None);
        assert_eq!(message.tool_calls[1].arguments, "{}");
        assert_eq!(message.tool_calls[1].index, Some(1));
        assert_eq!(response.finish_reason(), Some(&FinishReason::ToolCalls));
    }

    #[test]
    fn the_stop_reason_table_is_translated() {
        let cases = [
            ("tool_use", FinishReason::ToolCalls),
            ("end_turn", FinishReason::Stop),
            ("stop_sequence", FinishReason::Stop),
            ("max_tokens", FinishReason::Length),
            ("pause_turn", FinishReason::Other("pause_turn".to_owned())),
        ];

        for (reason, expected) in cases {
            let response = parse(json!({"content": [], "stop_reason": reason}));
            assert_eq!(
                response.finish_reason(),
                Some(&expected),
                "stop_reason: {reason}"
            );
        }

        let missing = parse(json!({"content": []}));
        assert_eq!(missing.finish_reason(), None);
    }

    #[test]
    fn usage_adds_the_cache_counters() {
        let response = parse(json!({
            "content": [],
            "usage": {
                "input_tokens": 10,
                "cache_read_input_tokens": 3,
                "cache_creation_input_tokens": 5,
                "output_tokens": 7,
            },
        }));

        assert_eq!(response.usage.input_tokens, 18);
        assert_eq!(response.usage.output_tokens, 7);
        assert_eq!(response.usage.total_tokens, 25);
        assert_eq!(response.usage.cache_tokens, 3);
        assert_eq!(response.usage.reasoning_tokens, 0);
        assert!(response.usage.extra.is_empty());
    }

    #[test]
    fn missing_and_malformed_usage_are_empty() {
        assert_eq!(
            parse(json!({"content": []})).usage,
            crate::prompt::Usage::default()
        );
        assert_eq!(
            parse(json!({"content": [], "usage": "nope"})).usage,
            crate::prompt::Usage::default()
        );
        assert_eq!(
            parse(json!({"content": [], "usage": {"input_tokens": "many"}})).usage,
            crate::prompt::Usage::default()
        );
    }

    #[test]
    fn missing_fields_are_tolerated() {
        let response = parse(json!({}));
        assert_eq!(response.id, "");
        assert_eq!(response.model, "");
        assert_eq!(response.choices.len(), 1);
        assert_eq!(response.message().expect("one choice").text(), "");
        assert_eq!(response.finish_reason(), None);
        assert_eq!(response.usage, crate::prompt::Usage::default());
    }

    #[test]
    fn non_object_blocks_and_payloads_are_tolerated() {
        let response = parse(json!({"content": ["text", 42, null]}));
        assert_eq!(response.message().expect("one choice").text(), "");

        let error = parse_messages_response(&json!(["not", "an", "object"]))
            .expect_err("the payload cannot be a provider response");
        assert_eq!(error.kind(), ErrorKind::Llm);
        assert!(error.to_string().contains("JSON object"));
    }

    #[test]
    fn usage_normalization_fills_the_total_from_the_parts() {
        let usage = parse_anthropic_usage(&json!({"input_tokens": 4, "output_tokens": 6}));
        assert_eq!(usage.input_tokens, 4);
        assert_eq!(usage.output_tokens, 6);
        assert_eq!(usage.total_tokens, 10);
    }
}
