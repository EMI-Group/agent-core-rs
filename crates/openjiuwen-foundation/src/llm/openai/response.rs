//! OpenAI-compatible response projection: messages, choices, finish reasons,
//! and token usage.
//!
//! Ports `_parse_response` and the usage helpers of
//! `openjiuwen/core/foundation/llm/model_clients/openai_model_client.py`.
//! Parsing is defensive: a structurally unusable response is an
//! [`Error::Llm`], never a panic, while a missing optional field degrades to its
//! default.

use std::collections::BTreeMap;

use openjiuwen_common::error::{Error, Result};
use serde_json::{Map, Value};

use crate::prompt::{BaseMessage, MessageContent, Role, ToolCall, Usage};

use super::super::model::{ChatResponse, Choice, FinishReason};

/// Parses one non-streaming chat completion.
///
/// `id` and `model` default to empty strings when the provider omits them, and
/// `usage` defaults to an all-zero [`Usage`]. A missing or non-object `choices`
/// entry, or a choice without a `message` object, is an [`Error::Llm`] because
/// the response cannot be interpreted.
///
/// # Errors
///
/// Returns [`Error::Llm`] when `value` is not an object, when `choices` is
/// missing or not an array, when a choice is not an object, when a choice has no
/// `message` object, or when a tool call cannot be decoded.
pub fn parse_chat_completion(value: &Value) -> Result<ChatResponse> {
    let object = value.as_object().ok_or_else(|| {
        Error::llm(format!(
            "chat completion response must be a JSON object, got {}",
            json_kind(value)
        ))
    })?;

    let id = string_field(object, "id").unwrap_or_default();
    let model = string_field(object, "model").unwrap_or_default();

    let choices_value = object
        .get("choices")
        .ok_or_else(|| Error::llm("chat completion response is missing the `choices` array"))?;
    let choices_array = choices_value.as_array().ok_or_else(|| {
        Error::llm(format!(
            "chat completion `choices` must be an array, got {}",
            json_kind(choices_value)
        ))
    })?;

    let mut choices = Vec::with_capacity(choices_array.len());
    for (position, raw) in choices_array.iter().enumerate() {
        choices.push(parse_choice(raw, position)?);
    }

    let usage = parse_usage(object.get("usage").unwrap_or(&Value::Null));

    Ok(ChatResponse {
        id,
        model,
        choices,
        usage,
    })
}

/// Parses one entry of the `choices` array.
fn parse_choice(value: &Value, position: usize) -> Result<Choice> {
    let object = value.as_object().ok_or_else(|| {
        Error::llm(format!(
            "chat completion choice {position} must be a JSON object, got {}",
            json_kind(value)
        ))
    })?;

    let index = object
        .get("index")
        .and_then(Value::as_u64)
        .and_then(|raw| u32::try_from(raw).ok())
        .unwrap_or(position as u32);

    let message_value = object.get("message").ok_or_else(|| {
        Error::llm(format!(
            "chat completion choice {index} is missing `message`"
        ))
    })?;
    let message_object = message_value.as_object().ok_or_else(|| {
        Error::llm(format!(
            "chat completion choice {index} `message` must be a JSON object, got {}",
            json_kind(message_value)
        ))
    })?;
    let message = parse_message(message_object)?;

    let reported = object
        .get("finish_reason")
        .and_then(Value::as_str)
        .and_then(parse_finish_reason);
    // A provider that returns tool calls always finished because of them, even
    // when it forgot to say so.
    let finish_reason = match reported {
        None if !message.tool_calls.is_empty() => Some(FinishReason::ToolCalls),
        reason => reason,
    };

    Ok(Choice {
        index,
        message,
        finish_reason,
    })
}

/// Parses the assistant message of one choice.
fn parse_message(object: &Map<String, Value>) -> Result<BaseMessage> {
    // Provider responses are assistant turns; a missing role must not turn the
    // message into a user turn.
    let role = object
        .get("role")
        .and_then(Value::as_str)
        .filter(|role| !role.is_empty())
        .map_or(Role::Assistant, Role::from);

    let content_value = object.get("content").cloned().unwrap_or(Value::Null);
    let content: MessageContent = serde_json::from_value(content_value).map_err(|error| {
        Error::llm(format!(
            "decoding the chat completion message content: {error}"
        ))
    })?;

    Ok(BaseMessage {
        role,
        content,
        name: string_field(object, "name"),
        tool_calls: parse_tool_calls(object.get("tool_calls"))?,
        tool_call_id: string_field(object, "tool_call_id"),
        reasoning_content: string_field(object, "reasoning_content"),
        metadata: BTreeMap::new(),
    })
}

/// Parses the `tool_calls` array, forcing each call's index to its position.
fn parse_tool_calls(value: Option<&Value>) -> Result<Vec<ToolCall>> {
    let value = match value {
        None | Some(Value::Null) => return Ok(Vec::new()),
        Some(value) => value,
    };
    let calls = value.as_array().ok_or_else(|| {
        Error::llm(format!(
            "chat completion `tool_calls` must be an array, got {}",
            json_kind(value)
        ))
    })?;

    calls
        .iter()
        .enumerate()
        .map(|(position, raw)| {
            ToolCall::from_openai_with_index(raw, position as u32).map_err(|error| {
                Error::llm(format!(
                    "decoding chat completion tool call {position}: {error}"
                ))
            })
        })
        .collect()
}

/// Maps a provider finish-reason string onto [`FinishReason`].
///
/// `""` and `"null"` mean "no reason reported" (the Python client uses the
/// string `"null"` as its sentinel); the four documented reasons map to their
/// variants; anything else is preserved as [`FinishReason::Other`].
///
/// ```
/// use openjiuwen_foundation::llm::model::FinishReason;
/// use openjiuwen_foundation::llm::openai::parse_finish_reason;
///
/// assert_eq!(parse_finish_reason("stop"), Some(FinishReason::Stop));
/// assert_eq!(parse_finish_reason("null"), None);
/// assert_eq!(
///     parse_finish_reason("eos"),
///     Some(FinishReason::Other("eos".to_owned()))
/// );
/// ```
#[must_use]
pub fn parse_finish_reason(value: &str) -> Option<FinishReason> {
    match value {
        "" | "null" => None,
        reason => Some(FinishReason::from(reason)),
    }
}

/// Parses one `usage` object.
///
/// Reads `prompt_tokens`/`completion_tokens`/`total_tokens`,
/// `prompt_tokens_details.cached_tokens` into [`Usage::cache_tokens`], and
/// `completion_tokens_details.reasoning_tokens` into
/// [`Usage::reasoning_tokens`], then fills in a missing total. A missing or
/// non-object value yields the default (all-zero) usage, so a provider that omits
/// accounting never fails a call.
#[must_use]
pub fn parse_usage(value: &Value) -> Usage {
    let Some(object) = value.as_object() else {
        return Usage::default();
    };
    let prompt_details = object
        .get("prompt_tokens_details")
        .and_then(Value::as_object);
    let completion_details = object
        .get("completion_tokens_details")
        .and_then(Value::as_object);

    Usage {
        input_tokens: counter(object.get("prompt_tokens")),
        output_tokens: counter(object.get("completion_tokens")),
        total_tokens: counter(object.get("total_tokens")),
        cache_tokens: prompt_details.map_or(0, |details| counter(details.get("cached_tokens"))),
        reasoning_tokens: completion_details
            .map_or(0, |details| counter(details.get("reasoning_tokens"))),
        extra: BTreeMap::new(),
    }
    .normalized()
}

/// Reads a non-negative token counter, accepting integer and float encodings.
fn counter(value: Option<&Value>) -> u64 {
    let Some(value) = value else {
        return 0;
    };
    value.as_u64().unwrap_or_else(|| {
        value
            .as_f64()
            .filter(|count| *count >= 0.0)
            .map_or(0, |count| count as u64)
    })
}

/// Reads a non-empty string field.
fn string_field(object: &Map<String, Value>, key: &str) -> Option<String> {
    object
        .get(key)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
}

/// Short JSON kind name used in error messages and shared with the stream parser.
pub(super) fn json_kind(value: &Value) -> &'static str {
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
    use openjiuwen_common::error::ErrorKind;
    use serde_json::json;

    use super::*;

    fn completion() -> Value {
        json!({
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "the weather is "},
                        {"type": "text", "text": "18 degrees"}
                    ],
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}
                        },
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {"name": "get_time", "arguments": "{}"}
                        }
                    ],
                    "reasoning_content": "checked the forecast"
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 10,
                "total_tokens": 30,
                "prompt_tokens_details": {"cached_tokens": 8},
                "completion_tokens_details": {"reasoning_tokens": 4}
            }
        })
    }

    #[test]
    fn parses_a_full_completion() {
        let response = parse_chat_completion(&completion()).expect("parses");

        assert_eq!(response.id, "chatcmpl-1");
        assert_eq!(response.model, "gpt-4o");
        assert_eq!(response.choices.len(), 1);

        let choice = &response.choices[0];
        assert_eq!(choice.index, 0);
        assert_eq!(choice.message.role, Role::Assistant);
        assert_eq!(choice.message.text(), "the weather is \n18 degrees");
        assert_eq!(
            choice.message.reasoning_content.as_deref(),
            Some("checked the forecast")
        );
        assert_eq!(choice.finish_reason, Some(FinishReason::ToolCalls));

        let calls = &choice.message.tool_calls;
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].id.as_deref(), Some("call_1"));
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].arguments, "{\"city\":\"Paris\"}");
        assert_eq!(calls[0].index, Some(0));
        assert_eq!(calls[1].name, "get_time");
        assert_eq!(calls[1].index, Some(1));

        assert_eq!(response.usage.input_tokens, 20);
        assert_eq!(response.usage.output_tokens, 10);
        assert_eq!(response.usage.total_tokens, 30);
        assert_eq!(response.usage.cache_tokens, 8);
        assert_eq!(response.usage.reasoning_tokens, 4);
    }

    #[test]
    fn a_null_content_becomes_empty_text() {
        let value = json!({
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": null},
                "finish_reason": "stop"
            }]
        });
        let response = parse_chat_completion(&value).expect("parses");

        assert_eq!(response.id, "");
        assert_eq!(response.model, "");
        assert_eq!(response.message().expect("one choice").text(), "");
        assert_eq!(response.finish_reason(), Some(&FinishReason::Stop));
    }

    #[test]
    fn a_string_content_is_preserved() {
        let value = json!({
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "hello"}
            }]
        });
        let response = parse_chat_completion(&value).expect("parses");

        assert_eq!(response.message().expect("one choice").text(), "hello");
    }

    #[test]
    fn a_missing_finish_reason_with_tool_calls_is_inferred() {
        let value = json!({
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "ping", "arguments": ""}
                    }]
                }
            }]
        });
        let response = parse_chat_completion(&value).expect("parses");

        assert_eq!(response.finish_reason(), Some(&FinishReason::ToolCalls));
    }

    #[test]
    fn a_missing_finish_reason_without_tool_calls_stays_none() {
        let value = json!({
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}}]
        });
        let response = parse_chat_completion(&value).expect("parses");

        assert_eq!(response.finish_reason(), None);
    }

    #[test]
    fn an_empty_choices_array_yields_no_message() {
        let value = json!({"id": "x", "choices": []});
        let response = parse_chat_completion(&value).expect("parses");

        assert_eq!(response.choices, Vec::new());
        assert!(response.message().is_none());
        assert_eq!(response.usage, Usage::default());
    }

    #[test]
    fn message_metadata_is_parsed() {
        let value = json!({
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "hi",
                    "name": "assistant-1",
                    "tool_call_id": "call_9"
                }
            }]
        });
        let message = parse_chat_completion(&value)
            .expect("parses")
            .message()
            .expect("one choice")
            .clone();

        assert_eq!(message.name.as_deref(), Some("assistant-1"));
        assert_eq!(message.tool_call_id.as_deref(), Some("call_9"));
    }

    #[test]
    fn a_missing_role_defaults_to_assistant() {
        let value = json!({"choices": [{"index": 0, "message": {"content": "hi"}}]});
        let message = parse_chat_completion(&value)
            .expect("parses")
            .message()
            .expect("one choice")
            .clone();

        assert_eq!(message.role, Role::Assistant);
    }

    #[test]
    fn a_missing_usage_is_all_zero() {
        let value = json!({"choices": [{"index": 0, "message": {"content": "hi"}}]});
        let response = parse_chat_completion(&value).expect("parses");

        assert_eq!(response.usage, Usage::default());
    }

    #[test]
    fn a_non_object_usage_is_all_zero() {
        assert_eq!(parse_usage(&json!("nope")), Usage::default());
        assert_eq!(parse_usage(&Value::Null), Usage::default());
    }

    #[test]
    fn a_missing_total_is_derived_from_the_parts() {
        let usage = parse_usage(&json!({"prompt_tokens": 7, "completion_tokens": 3}));
        assert_eq!(usage.total_tokens, 10);
    }

    #[test]
    fn float_counters_are_accepted() {
        let usage = parse_usage(&json!({"prompt_tokens": 7.0, "total_tokens": 7}));
        assert_eq!(usage.input_tokens, 7);
        assert_eq!(usage.total_tokens, 7);
    }

    #[test]
    fn a_non_object_response_is_an_llm_error() {
        for value in [json!("text"), json!([1, 2]), Value::Null] {
            let error = parse_chat_completion(&value).expect_err("must fail");
            assert_eq!(error.kind(), ErrorKind::Llm, "value: {value}");
        }
    }

    #[test]
    fn a_missing_choices_array_is_an_llm_error() {
        let error = parse_chat_completion(&json!({"id": "x"})).expect_err("must fail");
        assert_eq!(error.kind(), ErrorKind::Llm);
        assert!(error.to_string().contains("choices"), "{error}");
    }

    #[test]
    fn a_non_array_choices_value_is_an_llm_error() {
        let error = parse_chat_completion(&json!({"choices": "nope"})).expect_err("must fail");
        assert_eq!(error.kind(), ErrorKind::Llm);
    }

    #[test]
    fn a_choice_without_a_message_is_an_llm_error() {
        let error =
            parse_chat_completion(&json!({"choices": [{"index": 0}]})).expect_err("must fail");
        assert_eq!(error.kind(), ErrorKind::Llm);
        assert!(error.to_string().contains("message"), "{error}");
    }

    #[test]
    fn a_non_object_message_is_an_llm_error() {
        let error = parse_chat_completion(&json!({"choices": [{"index": 0, "message": "hi"}]}))
            .expect_err("must fail");
        assert_eq!(error.kind(), ErrorKind::Llm);
    }

    #[test]
    fn a_malformed_tool_call_is_an_llm_error() {
        let error = parse_chat_completion(&json!({
            "choices": [{
                "index": 0,
                "message": {"content": null, "tool_calls": [{"id": "call_1"}]}
            }]
        }))
        .expect_err("must fail");

        assert_eq!(error.kind(), ErrorKind::Llm);
        assert!(error.to_string().contains("tool call"), "{error}");
    }

    #[test]
    fn finish_reasons_follow_the_provider_table() {
        assert_eq!(parse_finish_reason("stop"), Some(FinishReason::Stop));
        assert_eq!(parse_finish_reason("length"), Some(FinishReason::Length));
        assert_eq!(
            parse_finish_reason("tool_calls"),
            Some(FinishReason::ToolCalls)
        );
        assert_eq!(
            parse_finish_reason("content_filter"),
            Some(FinishReason::ContentFilter)
        );
        assert_eq!(parse_finish_reason(""), None);
        assert_eq!(parse_finish_reason("null"), None);
        assert_eq!(
            parse_finish_reason("insufficient_system_resource"),
            Some(FinishReason::Other(
                "insufficient_system_resource".to_owned()
            ))
        );
    }
}
