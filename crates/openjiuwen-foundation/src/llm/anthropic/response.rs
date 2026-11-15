//! Anthropic Messages response parsing.
//!
//! Converts the JSON body of `POST {base}/v1/messages` into the provider-neutral
//! [`ChatResponse`]. Text blocks concatenate into the message content, `tool_use`
//! blocks become [`ToolCall`]s whose `arguments` are the JSON text of the block's
//! `input`, and Anthropic's `stop_reason` and cache-aware usage are normalized so
//! every layer above the foundation sees the same accounting as with the
//! OpenAI-compatible client.
//!
//! Extended-thinking blocks are decoded rather than dropped: the `thinking` text
//! becomes the assistant message's
//! [`reasoning_content`](crate::prompt::BaseMessage::reasoning_content), and
//! every replayable `thinking`/`redacted_thinking` block is preserved verbatim
//! under [`THINKING_BLOCKS_METADATA_KEY`] in the message metadata. The Messages
//! API requires the original blocks, signature included, to be echoed back on a
//! follow-up `tool_use` turn, so [`crate::llm::anthropic::build_messages_body`]
//! re-emits them ahead of the text and `tool_use` blocks (see
//! [`preserved_thinking_blocks`]). A block without the provider's `signature`
//! (`thinking`) or opaque `data` (`redacted_thinking`) is never preserved,
//! because forwarding it would be a provider error, and this client never
//! invents one.

use serde_json::{Map, Value};

use crate::llm::model::{ChatResponse, Choice, FinishReason};
use crate::prompt::{BaseMessage, MessageContent, ToolCall, Usage};
use openjiuwen_common::error::{Error, Result};

/// The only tool-call kind the built-in providers emit.
const FUNCTION_KIND: &str = "function";

/// `BaseMessage::metadata` key holding the raw thinking blocks of an assistant turn.
///
/// The value is a JSON array of the provider's `thinking` and
/// `redacted_thinking` content blocks, in the order the response listed them
/// (the API requires thinking blocks first). It is what makes a follow-up
/// `tool_use` turn replayable: the request projection reads this key and emits
/// the blocks verbatim. Callers may copy it between messages, but should not
/// edit the blocks — a `thinking` block only replays with the provider's own
/// `signature`.
pub const THINKING_BLOCKS_METADATA_KEY: &str = "anthropic_thinking_blocks";

/// Parses one Anthropic Messages response body.
///
/// Missing `id`, `model`, `content`, `usage`, or `stop_reason` are tolerated and
/// become empty values; a payload that is not a JSON object is an
/// [`Error::Llm`] because it cannot be a provider response.
///
/// `thinking` blocks contribute their text to the message's
/// `reasoning_content`, and every replayable `thinking`/`redacted_thinking`
/// block is also stored under [`THINKING_BLOCKS_METADATA_KEY`] so the next
/// request can echo it back.
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
    let mut reasoning = String::new();
    let mut thinking_blocks: Vec<Value> = Vec::new();
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
                Some("thinking") => {
                    if let Some(fragment) = block.get("thinking").and_then(Value::as_str) {
                        reasoning.push_str(fragment);
                    }
                    preserve_thinking_block(&mut thinking_blocks, block);
                }
                Some("redacted_thinking") => preserve_thinking_block(&mut thinking_blocks, block),
                _ => {}
            }
        }
    }

    let mut message = BaseMessage::assistant(MessageContent::Text(text));
    message.tool_calls = tool_calls;
    if !reasoning.is_empty() {
        message.reasoning_content = Some(reasoning);
    }
    if !thinking_blocks.is_empty() {
        message.metadata.insert(
            THINKING_BLOCKS_METADATA_KEY.to_owned(),
            Value::Array(thinking_blocks),
        );
    }

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

/// Preserves one replayable thinking block, or logs why it is not replayed.
///
/// A `thinking` block is replayable only with the provider's own `signature`,
/// and a `redacted_thinking` block only with its opaque `data`; a block missing
/// either is dropped with a `tracing::warn!` instead of being sent back in a
/// shape the API rejects. The block is stored verbatim, so an unknown extra
/// field a future API version adds survives the round trip.
fn preserve_thinking_block(preserved: &mut Vec<Value>, block: &Map<String, Value>) {
    let kind = block
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if !is_replayable_block(block) {
        tracing::warn!(
            target: "openjiuwen_foundation::llm::anthropic",
            block_type = kind,
            "dropping a thinking block that cannot be replayed: \
             `thinking` needs a signature and `redacted_thinking` needs data"
        );
        return;
    }
    preserved.push(Value::Object(block.clone()));
}

/// Reports whether `block` is a thinking block the API will accept back.
///
/// The required field is the provider's own opaque proof (`signature` for
/// `thinking`, `data` for `redacted_thinking`) and must be non-empty; this
/// client never synthesizes one.
fn is_replayable_block(block: &Map<String, Value>) -> bool {
    let kind = block
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let proof = match kind {
        "thinking" => "signature",
        "redacted_thinking" => "data",
        _ => return false,
    };
    block
        .get(proof)
        .and_then(Value::as_str)
        .is_some_and(|value| !value.is_empty())
}

/// The replayable thinking blocks an assistant message carries, in order.
///
/// Reads [`THINKING_BLOCKS_METADATA_KEY`] from the message metadata and returns
/// only the entries that are still replayable (a known block type with its
/// `signature`/`data`), because the metadata map is caller-writable. An absent
/// key, a non-array value, and malformed entries all yield no block, so the
/// request projection can never emit a thinking block the API would reject.
pub(crate) fn preserved_thinking_blocks(message: &BaseMessage) -> Vec<Value> {
    let Some(Value::Array(blocks)) = message.metadata.get(THINKING_BLOCKS_METADATA_KEY) else {
        return Vec::new();
    };

    blocks
        .iter()
        .filter(|block| block.as_object().is_some_and(is_replayable_block))
        .cloned()
        .collect()
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
