//! Anthropic Messages request projection.
//!
//! Converts the provider-neutral [`BaseMessage`] list and [`ModelRequestConfig`]
//! into the JSON body of `POST {base}/v1/messages`. The transformations are the
//! ones the Messages API requires and that the OpenAI-compatible wire shape does
//! not express:
//!
//! - every `system` turn becomes one top-level `system` text block (the API
//!   takes `system` only as a request parameter), except where the target model
//!   accepts a mid-conversation `system` turn and the position is legal;
//! - `cache_control: {"type": "ephemeral"}` anchors are written on the last
//!   tool, the last system block, and the last cacheable block of the last
//!   stable message; `extra["disable_cache_control"] = true` turns them off and
//!   `extra["cache_control_ttl"] = "1h"` selects the long ephemeral lifetime;
//! - assistant `tool_calls` become `tool_use` blocks with a parsed JSON `input`;
//! - consecutive `tool` turns become one `user` message of `tool_result` blocks;
//! - `OpenAI` `image_url` parts become Anthropic `image` blocks (base64 or URL);
//! - `OpenAI` function schemas become `{name, description, input_schema}` tools.
//!
//! The converter is a pure function so tests assert the exact body without a
//! server. It never panics on caller input: an unrepresentable message is an
//! [`Error::Validation`].

use openjiuwen_common::error::{Error, Result};
use serde_json::{Map, Value};

use crate::llm::config::ModelRequestConfig;
use crate::prompt::BaseMessage;

use super::cache::{
    apply_messages_anchor, apply_system_anchor, apply_tool_anchor, is_control_key,
    last_input_is_transient, CacheControl,
};
use super::convert::{convert_messages, convert_tool_choice, convert_tools, ConvertedMessages};

/// `max_tokens` used when the request does not set one.
const DEFAULT_MAX_TOKENS: u32 = 32768;

/// Output cap shared by the `claude-3.5` families.
const CLAUDE_35_MAX_TOKENS: u32 = 8192;

/// Output cap shared by the original `claude-3` families.
const CLAUDE_3_MAX_TOKENS: u32 = 4096;

/// Strips the endpoint suffix from a configured Anthropic API base.
///
/// The client always appends `/v1/messages`, so a caller that pastes the full
/// endpoint (`https://api.anthropic.com/v1/messages`) must not end up with
/// `/v1/messages/v1/messages`. The function trims whitespace and trailing
/// slashes, removes a trailing `/v1/messages`, then removes a trailing `/v1`.
///
/// ```
/// use openjiuwen_foundation::llm::anthropic::normalize_api_base;
///
/// assert_eq!(
///     normalize_api_base(" https://api.anthropic.com/v1/messages/ "),
///     "https://api.anthropic.com"
/// );
/// assert_eq!(
///     normalize_api_base("https://gateway.test/anthropic/v1"),
///     "https://gateway.test/anthropic"
/// );
/// ```
#[must_use]
pub fn normalize_api_base(api_base: &str) -> String {
    let trimmed = api_base.trim().trim_end_matches('/');
    let trimmed = trimmed.strip_suffix("/v1/messages").unwrap_or(trimmed);
    let trimmed = trimmed.strip_suffix("/v1").unwrap_or(trimmed);
    trimmed.trim_end_matches('/').to_owned()
}

/// Builds the Anthropic Messages request body.
///
/// `stream` is written verbatim to the body's `stream` field; the caller decides
/// between the buffered and streaming endpoint behavior.
///
/// Prompt-caching anchors are always written. Two `request.extra` keys tune
/// them and are consumed here rather than forwarded: `disable_cache_control:
/// true` omits every anchor, and `cache_control_ttl: "1h"` writes the long
/// ephemeral lifetime on each marker (the default marker has no `ttl` key).
///
/// # Errors
///
/// Returns [`Error::Validation`] when the model name is empty, when a tool call
/// has no id, when a tool result has no `tool_call_id`, or when a content part
/// cannot be represented by the Messages API (for example a non-HTTP image URL).
pub fn build_messages_body(
    messages: &[BaseMessage],
    request: &ModelRequestConfig,
    stream: bool,
) -> Result<Value> {
    if request.model_name.trim().is_empty() {
        return Err(Error::validation(
            "model_name must not be empty for an Anthropic Messages request",
        ));
    }

    let cache = CacheControl::from_extra(&request.extra);
    // Read from the original messages: `metadata` does not survive projection.
    let transient_tail = last_input_is_transient(messages);
    let ConvertedMessages {
        mut system,
        messages: mut converted,
    } = convert_messages(messages, &request.model_name)?;
    apply_messages_anchor(&mut converted, transient_tail, &cache);

    let mut body = Map::new();
    body.insert(
        "model".to_owned(),
        Value::String(request.model_name.clone()),
    );
    body.insert("messages".to_owned(), Value::Array(converted));
    body.insert(
        "max_tokens".to_owned(),
        Value::from(max_tokens_for(&request.model_name, request.max_tokens)),
    );
    body.insert("stream".to_owned(), Value::Bool(stream));

    if let Some(temperature) = request.temperature {
        body.insert("temperature".to_owned(), Value::from(temperature));
    }
    if let Some(top_p) = request.top_p {
        body.insert("top_p".to_owned(), Value::from(top_p));
    }
    if !request.stop.is_empty() {
        body.insert(
            "stop_sequences".to_owned(),
            Value::Array(request.stop.iter().cloned().map(Value::String).collect()),
        );
    }
    if !system.is_empty() {
        apply_system_anchor(&mut system, &cache);
        body.insert("system".to_owned(), Value::Array(system));
    }
    if let Some(tools) = &request.tools {
        let mut converted_tools = convert_tools(tools);
        if !matches!(&converted_tools, Value::Array(items) if items.is_empty()) {
            apply_tool_anchor(&mut converted_tools, &cache);
            body.insert("tools".to_owned(), converted_tools);
        }
    }
    if let Some(tool_choice) = request.tool_choice.as_ref().and_then(convert_tool_choice) {
        body.insert("tool_choice".to_owned(), tool_choice);
    }

    // The caller's `extra` map wins over every derived field, as documented by
    // `ModelRequestConfig::extra`; the cache-control knobs are client-side and
    // are not forwarded to the API.
    if let Some(extra) = request.extra.as_object() {
        for (key, value) in extra {
            if is_control_key(key) {
                continue;
            }
            body.insert(key.clone(), value.clone());
        }
    }

    Ok(Value::Object(body))
}

/// Resolves `max_tokens`, using the documented default and per-family caps.
fn max_tokens_for(model: &str, requested: Option<u32>) -> u32 {
    let value = requested.unwrap_or(DEFAULT_MAX_TOKENS);
    let name = model.trim().to_ascii_lowercase();

    if name.starts_with("claude-3-5") || name.starts_with("claude-3.5") {
        value.min(CLAUDE_35_MAX_TOKENS)
    } else if name.starts_with("claude-3-opus")
        || name.starts_with("claude-3-sonnet")
        || name.starts_with("claude-3-haiku")
    {
        value.min(CLAUDE_3_MAX_TOKENS)
    } else {
        value
    }
}
#[cfg(test)]
mod cache_tests;
#[cfg(test)]
mod tests;
