//! `OpenAI` Responses response projection: output items, tool calls, finish
//! reasons, and token usage.
//!
//! Ports `parse_response`, `_read_output_items`, `_finish_reason`, and
//! `_usage_from_payload` of `openjiuwen/core/foundation/llm/utils/responses_utils.py`.
//! Parsing is defensive: a structurally unusable payload is an [`Error::Llm`],
//! never a panic, while a missing optional field degrades to its default.

use openjiuwen_common::error::{Error, Result};
use serde_json::Value;

use crate::prompt::{BaseMessage, MessageContent, ToolCall, Usage};

use super::super::model::{ChatResponse, Choice, FinishReason};

/// Parses one non-streaming Responses payload into a [`ChatResponse`].
///
/// `output` items of type `message` contribute their `output_text`/`text` parts,
/// `reasoning` items contribute their `summary[].text` and `text`, and
/// `function_call` items become [`ToolCall`]s. When no message item carried text,
/// a top-level `output_text` string is used instead (the convenience field some
/// gateways add). The finish reason is [`FinishReason::ToolCalls`] when the payload
/// carries tool calls, [`FinishReason::Length`] when `status` is `"incomplete"`,
/// and [`FinishReason::Stop`] otherwise.
///
/// # Errors
///
/// Returns [`Error::Llm`] when `value` is not a JSON object.
pub fn parse_responses_response(value: &Value) -> Result<ChatResponse> {
    let object = value.as_object().ok_or_else(|| {
        Error::llm(format!(
            "responses payload must be a JSON object, got {}",
            json_kind(value)
        ))
    })?;

    let mut content_parts: Vec<String> = Vec::new();
    let mut reasoning_parts: Vec<String> = Vec::new();
    let mut tool_calls: Vec<ToolCall> = Vec::new();

    if let Some(output) = object.get("output").and_then(Value::as_array) {
        for item in output {
            let Some(item) = item.as_object() else {
                continue;
            };
            match item.get("type").and_then(Value::as_str).unwrap_or_default() {
                "message" => content_parts.extend(message_content_text(item.get("content"))),
                "reasoning" => reasoning_parts.extend(reasoning_text(item)),
                "function_call" => {
                    let index = u32::try_from(tool_calls.len()).unwrap_or(u32::MAX);
                    tool_calls.push(tool_call_from_item(item, Some(index)));
                }
                _ => {}
            }
        }
    }
    if content_parts.is_empty() {
        if let Some(text) = object
            .get("output_text")
            .and_then(Value::as_str)
            .filter(|text| !text.is_empty())
        {
            content_parts.push(text.to_owned());
        }
    }

    let finish_reason = if tool_calls.is_empty() {
        if object.get("status").and_then(Value::as_str) == Some("incomplete") {
            FinishReason::Length
        } else {
            FinishReason::Stop
        }
    } else {
        FinishReason::ToolCalls
    };

    let mut message = BaseMessage::assistant(MessageContent::Text(content_parts.concat()));
    message.tool_calls = tool_calls;
    if !reasoning_parts.is_empty() {
        message.reasoning_content = Some(reasoning_parts.join("\n"));
    }

    Ok(ChatResponse {
        id: string_field(object, "id").unwrap_or_default(),
        model: string_field(object, "model").unwrap_or_default(),
        choices: vec![Choice {
            index: 0,
            message,
            finish_reason: Some(finish_reason),
        }],
        usage: parse_responses_usage(object.get("usage").unwrap_or(&Value::Null)),
    })
}

/// Parses one Responses `usage` object.
///
/// Accepts the Responses spellings (`input_tokens`/`output_tokens`) and the
/// chat-completion ones (`prompt_tokens`/`completion_tokens`), sums the two halves
/// when no `total_tokens` is reported, and reads
/// `input_tokens_details.cached_tokens` / `output_tokens_details.reasoning_tokens`
/// (with the `prompt_*`/`completion_*` fallbacks) into the matching counters. A
/// missing or non-object value yields the all-zero usage.
#[must_use]
pub fn parse_responses_usage(value: &Value) -> Usage {
    let Some(object) = value.as_object() else {
        return Usage::default();
    };

    let input_tokens = first_present(object, &["input_tokens", "prompt_tokens"]);
    let output_tokens = first_present(object, &["output_tokens", "completion_tokens"]);
    let total_tokens = object
        .get("total_tokens")
        .and_then(counter)
        .unwrap_or(input_tokens.saturating_add(output_tokens));

    let cache_tokens = first_object(object, &["input_tokens_details", "prompt_tokens_details"])
        .map_or(0, |details| {
            details.get("cached_tokens").and_then(counter).unwrap_or(0)
        });
    let reasoning_tokens = first_object(
        object,
        &["output_tokens_details", "completion_tokens_details"],
    )
    .map_or(0, |details| {
        details
            .get("reasoning_tokens")
            .and_then(counter)
            .unwrap_or(0)
    });

    Usage {
        input_tokens,
        output_tokens,
        total_tokens,
        cache_tokens,
        reasoning_tokens,
        extra: std::collections::BTreeMap::new(),
    }
    .normalized()
}

/// Reads the first present non-null counter from `keys`.
fn first_present(object: &serde_json::Map<String, Value>, keys: &[&str]) -> u64 {
    keys.iter()
        .find_map(|key| object.get(*key).filter(|value| !value.is_null()))
        .and_then(counter)
        .unwrap_or(0)
}

/// Reads the first present object from `keys`.
fn first_object<'a>(
    object: &'a serde_json::Map<String, Value>,
    keys: &[&str],
) -> Option<&'a serde_json::Map<String, Value>> {
    keys.iter()
        .find_map(|key| object.get(*key).and_then(Value::as_object))
}

/// Reads a non-negative token counter, accepting integer and float encodings.
fn counter(value: &Value) -> Option<u64> {
    if let Some(raw) = value.as_u64() {
        return Some(raw);
    }
    if let Some(raw) = value.as_i64() {
        return u64::try_from(raw).ok();
    }
    let raw = value.as_f64()?;
    if raw.is_finite() && raw >= 0.0 {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        return Some(raw as u64);
    }
    None
}

/// Collects the text of one `message` item's content parts.
fn message_content_text(content: Option<&Value>) -> Vec<String> {
    let Some(content) = content.and_then(Value::as_array) else {
        return Vec::new();
    };
    content
        .iter()
        .filter_map(|part| {
            let part = part.as_object()?;
            match part.get("type").and_then(Value::as_str) {
                Some("output_text" | "text") => part
                    .get("text")
                    .filter(|text| !text.is_null())
                    .map(ValueText::to_string_text),
                _ => None,
            }
        })
        .collect()
}

/// Collects the text of one `reasoning` item (`summary[]` plus `text`).
fn reasoning_text(item: &serde_json::Map<String, Value>) -> Vec<String> {
    let mut texts = Vec::new();
    if let Some(summary) = item.get("summary").and_then(Value::as_array) {
        for part in summary {
            match part {
                Value::String(text) => texts.push(text.clone()),
                Value::Object(object) => {
                    if let Some(text) = object.get("text").filter(|text| !text.is_null()) {
                        texts.push(text.to_string_text());
                    }
                }
                _ => {}
            }
        }
    }
    if let Some(text) = item.get("text").filter(|text| !text.is_null()) {
        texts.push(text.to_string_text());
    }
    texts
}

/// Builds a [`ToolCall`] from one `function_call` output item.
fn tool_call_from_item(item: &serde_json::Map<String, Value>, index: Option<u32>) -> ToolCall {
    let item_id = string_field(item, "id").filter(|id| !id.is_empty());
    let call_id = string_field(item, "call_id")
        .filter(|id| !id.is_empty())
        .or_else(|| item_id.clone());
    ToolCall {
        id: call_id,
        kind: "function".to_owned(),
        name: string_field(item, "name").unwrap_or_default(),
        arguments: argument_string(item.get("arguments"), "{}"),
        index,
    }
}

/// Renders a wire `arguments` value as JSON text.
///
/// A string is returned verbatim, a null/missing value becomes `fallback`, and any
/// other JSON value is serialized.
fn argument_string(value: Option<&Value>, fallback: &str) -> String {
    match value {
        None | Some(Value::Null) => fallback.to_owned(),
        Some(Value::String(text)) => text.clone(),
        Some(other) => other.to_string(),
    }
}

/// Reads a trimmed, non-empty string field.
fn string_field(object: &serde_json::Map<String, Value>, key: &str) -> Option<String> {
    object
        .get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
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

/// Converts a non-string JSON scalar to its text form.
trait ValueText {
    /// The string itself, or the compact JSON encoding of any other value.
    fn to_string_text(&self) -> String;
}

impl ValueText for Value {
    fn to_string_text(&self) -> String {
        match self {
            Value::String(text) => text.clone(),
            other => other.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn a_complete_payload_becomes_one_assistant_message() {
        let payload = json!({
            "id": "resp_1",
            "model": "gpt-5",
            "status": "completed",
            "output": [{
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "hello"}, {"type": "output_text", "text": " there"}]
            }],
            "usage": {"input_tokens": 5, "output_tokens": 2, "total_tokens": 7}
        });

        let response = parse_responses_response(&payload).expect("the payload parses");
        assert_eq!(response.id, "resp_1");
        assert_eq!(response.model, "gpt-5");
        assert_eq!(
            response.message().expect("one choice").text(),
            "hello there"
        );
        assert_eq!(response.finish_reason(), Some(&FinishReason::Stop));
        assert_eq!(response.usage, Usage::new(5, 2));
    }

    #[test]
    fn reasoning_items_fill_the_reasoning_content() {
        let payload = json!({
            "output": [
                {"type": "reasoning", "summary": [{"type": "summary_text", "text": "think"}]},
                {"type": "reasoning", "text": "harder"},
                {"type": "message", "content": [{"type": "text", "text": "answer"}]}
            ]
        });

        let response = parse_responses_response(&payload).expect("parses");
        let message = response.message().expect("one choice");
        assert_eq!(message.text(), "answer");
        assert_eq!(message.reasoning_content.as_deref(), Some("think\nharder"));
    }

    #[test]
    fn function_call_items_become_tool_calls_and_win_the_finish_reason() {
        let payload = json!({
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": "let me check"}]},
                {"type": "function_call", "id": "fc_1", "call_id": "call_1",
                 "name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}
            ]
        });

        let response = parse_responses_response(&payload).expect("parses");
        let message = response.message().expect("one choice");
        assert_eq!(message.tool_calls.len(), 1);
        assert_eq!(message.tool_calls[0].id.as_deref(), Some("call_1"));
        assert_eq!(message.tool_calls[0].name, "get_weather");
        assert_eq!(message.tool_calls[0].arguments, "{\"city\":\"Paris\"}");
        assert_eq!(message.tool_calls[0].index, Some(0));
        assert_eq!(response.finish_reason(), Some(&FinishReason::ToolCalls));
    }

    #[test]
    fn an_incomplete_status_maps_to_length() {
        let payload = json!({
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "output": [{"type": "message", "content": [{"type": "output_text", "text": "cut"}]}]
        });
        assert_eq!(
            parse_responses_response(&payload)
                .expect("parses")
                .finish_reason(),
            Some(&FinishReason::Length)
        );
    }

    #[test]
    fn a_top_level_output_text_is_the_fallback() {
        let payload = json!({"output_text": "fallback text"});
        let response = parse_responses_response(&payload).expect("parses");
        assert_eq!(
            response.message().expect("one choice").text(),
            "fallback text"
        );
    }

    #[test]
    fn arguments_accept_objects_and_missing_values() {
        let payload = json!({
            "output": [
                {"type": "function_call", "call_id": "a", "name": "one", "arguments": {"x": 1}},
                {"type": "function_call", "call_id": "b", "name": "two"}
            ]
        });
        let message = parse_responses_response(&payload)
            .expect("parses")
            .message()
            .cloned()
            .expect("one choice");
        assert_eq!(message.tool_calls[0].arguments, "{\"x\":1}");
        assert_eq!(message.tool_calls[1].arguments, "{}");
        assert_eq!(message.tool_calls[0].index, Some(0));
        assert_eq!(message.tool_calls[1].index, Some(1));
    }

    #[test]
    fn usage_accepts_both_spellings_and_details() {
        let usage = parse_responses_usage(&json!({
            "prompt_tokens": 10,
            "completion_tokens": 4,
            "prompt_tokens_details": {"cached_tokens": 6},
            "completion_tokens_details": {"reasoning_tokens": 3}
        }));
        assert_eq!(usage.input_tokens, 10);
        assert_eq!(usage.output_tokens, 4);
        assert_eq!(usage.total_tokens, 14, "the total is summed when absent");
        assert_eq!(usage.cache_tokens, 6);
        assert_eq!(usage.reasoning_tokens, 3);

        let usage = parse_responses_usage(
            &json!({"input_tokens": 1, "output_tokens": 2, "total_tokens": 3}),
        );
        assert_eq!(usage, Usage::new(1, 2));

        assert_eq!(parse_responses_usage(&Value::Null), Usage::default());
    }

    #[test]
    fn a_non_object_payload_is_an_llm_error() {
        let error = parse_responses_response(&json!([])).expect_err("an array is rejected");
        assert_eq!(error.kind(), openjiuwen_common::error::ErrorKind::Llm);
    }
}
