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
//! - `OpenAI` function schemas become `{name, description, input_schema}` tools;
//! - a provider-neutral reasoning intent (`ModelRequestConfig::reasoning`, or
//!   the legacy `extra["reasoning"]`) is resolved for
//!   the Anthropic Messages protocol and merged as top-level body fields
//!   (`thinking`, `output_config`), unless the caller already set a legacy
//!   vendor control such as `extra["thinking"]`.
//!
//! The converter is a pure function so tests assert the exact body without a
//! server. It never panics on caller input: an unrepresentable message is an
//! [`Error::Validation`].

use openjiuwen_common::error::{Error, Result};
use serde_json::{Map, Value};

use crate::llm::config::ModelRequestConfig;
use crate::llm::reasoning::{
    apply_reasoning_plan, resolve_reasoning_plan_with_endpoint, ReasoningConfig, ReasoningProtocol,
};
use crate::prompt::BaseMessage;

use super::cache::{
    apply_messages_anchor, apply_system_anchor, apply_tool_anchor, is_control_key,
    last_input_is_transient, CacheControl,
};
use super::convert::{convert_messages, convert_tool_choice, convert_tools, ConvertedMessages};
use super::DEFAULT_PROVIDER;

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

/// Builds the Anthropic Messages request body for the `anthropic` provider.
///
/// Thin wrapper over [`build_messages_body_with_provider`] with no API-base host
/// hint; call sites that know their provider use that function directly.
///
/// # Errors
///
/// See [`build_messages_body_with_provider`].
pub fn build_messages_body(
    messages: &[BaseMessage],
    request: &ModelRequestConfig,
    stream: bool,
) -> Result<Value> {
    build_messages_body_with_provider(messages, request, stream, DEFAULT_PROVIDER, None)
}

/// Builds the Anthropic Messages request body, resolving provider hints.
///
/// `stream` is written verbatim to the body's `stream` field; the caller decides
/// between the buffered and streaming endpoint behavior.
///
/// Prompt-caching anchors are always written. Two `request.extra` keys tune
/// them and are consumed here rather than forwarded: `disable_cache_control:
/// true` omits every anchor, and `cache_control_ttl: "1h"` writes the long
/// ephemeral lifetime on each marker (the default marker has no `ttl` key).
///
/// `provider` and `api_base` are the client's resolved provider name and
/// endpoint; together they select the reasoning profile for a provider-neutral
/// reasoning intent in [`ModelRequestConfig::reasoning`] (or the legacy
/// `request.extra["reasoning"]` spelling; see [`crate::llm::reasoning`]).
/// A resolved plan is merged as top-level body fields — `sdk_params` first, then
/// `extra_body` — because the Messages API takes `thinking`, `output_config` and
/// friends at the top level. `request.extra` is merged after that, so a caller's
/// own keys still win, and the consumed control keys (`reasoning` plus the two
/// cache knobs) are never forwarded.
///
/// A caller's explicit legacy reasoning control — any of `reasoning_effort`,
/// `thinking`, `enable_thinking`, `thinking_budget`, `thinking_strategy`,
/// `chat_template_kwargs`, at the top level of `extra` or inside its
/// `extra_body` object — takes precedence over the plain intent: no plan is
/// resolved or applied, so the two encodings can never be mixed.
///
/// # Errors
///
/// Returns [`Error::Validation`] when the model name is empty, when a tool call
/// has no id, when a tool result has no `tool_call_id`, when a content part
/// cannot be represented by the Messages API (for example a non-HTTP image URL),
/// or when a `reasoning` value is present but is not an intent (`null`, a
/// boolean, a mode/effort string, or an object with `mode`, `effort` and/or
/// `budget_tokens`).
pub fn build_messages_body_with_provider(
    messages: &[BaseMessage],
    request: &ModelRequestConfig,
    stream: bool,
    provider: &str,
    api_base: Option<&str>,
) -> Result<Value> {
    build_messages_body_with_endpoint(messages, request, stream, provider, None, api_base)
}

/// Builds the Anthropic Messages request body, resolving an endpoint profile
/// first.
///
/// `endpoint_profile` names the wire dialect the configured endpoint speaks and
/// `api_base` is the resolved base URL; together with `provider` they select the
/// reasoning encoder for a provider-neutral reasoning intent
/// (see [`resolve_reasoning_plan_with_endpoint`]). A non-generic profile such as
/// `"vllm"` bypasses the profile table and writes the `enable_thinking` /
/// `thinking_budget` keys a self-hosted gateway honors, instead of the official
/// `thinking` object. Without an `endpoint_profile` the body is exactly the one
/// [`build_messages_body_with_provider`] produces.
///
/// # Errors
///
/// See [`build_messages_body_with_provider`].
pub fn build_messages_body_with_endpoint(
    messages: &[BaseMessage],
    request: &ModelRequestConfig,
    stream: bool,
    provider: &str,
    endpoint_profile: Option<&str>,
    api_base: Option<&str>,
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

    if let Some(intent) = reasoning_intent(request)? {
        if has_legacy_reasoning_controls(&request.extra) {
            tracing::debug!(
                target: "openjiuwen_foundation::llm::reasoning",
                model = request.model_name.as_str(),
                "legacy reasoning controls were supplied; \
                 the provider-neutral reasoning intent is ignored"
            );
        } else {
            let plan = resolve_reasoning_plan_with_endpoint(
                ReasoningProtocol::AnthropicMessages,
                provider,
                endpoint_profile,
                api_base,
                &request.model_name,
                &intent,
            );
            for warning in &plan.warnings {
                tracing::debug!(
                    target: "openjiuwen_foundation::llm::reasoning",
                    model = request.model_name.as_str(),
                    "reasoning config resolution: {warning}"
                );
            }
            if !plan.is_empty() {
                tracing::trace!(
                    target: "openjiuwen_foundation::llm::reasoning",
                    model = request.model_name.as_str(),
                    sdk_params = ?plan.sdk_param_names(),
                    extra_body = ?plan.extra_body_names(),
                    "applied the Anthropic reasoning plan"
                );
            }
            let mut resolved = Value::Object(std::mem::take(&mut body));
            apply_reasoning_plan(&mut resolved, &plan);
            if let Value::Object(object) = resolved {
                body = object;
            }
        }
    }

    // The caller's `extra` map wins over every derived field, as documented by
    // `ModelRequestConfig::extra`; the cache-control knobs and the reasoning
    // intent are client-side and are not forwarded to the API.
    if let Some(extra) = request.extra.as_object() {
        for (key, value) in extra {
            if is_control_key(key) || key == REASONING_KEY {
                continue;
            }
            body.insert(key.clone(), value.clone());
        }
    }

    Ok(Value::Object(body))
}

/// Key of the provider-neutral reasoning intent inside [`ModelRequestConfig::extra`].
const REASONING_KEY: &str = "reasoning";

/// Explicit vendor reasoning controls that replace the neutral intent.
const LEGACY_REASONING_KEYS: [&str; 6] = [
    "reasoning_effort",
    "thinking",
    "enable_thinking",
    "thinking_budget",
    "thinking_strategy",
    "chat_template_kwargs",
];

/// Reads the reasoning intent for a request.
///
/// [`ModelRequestConfig::reasoning`] wins over the legacy `extra` spelling. A
/// missing field and `null` both mean "no intent". Any other value must be an
/// intent (a boolean, a mode/effort string, or an object with `mode`, `effort`
/// and/or `budget_tokens`); anything else is [`Error::Validation`] rather than a
/// silently dropped request, because the Anthropic transport has no raw
/// `reasoning` body key to forward it to.
fn reasoning_intent(request: &ModelRequestConfig) -> Result<Option<ReasoningConfig>> {
    let value = request
        .reasoning
        .as_ref()
        .filter(|value| !value.is_null())
        .or_else(|| request.extra.get(REASONING_KEY));
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(value) => match ReasoningConfig::from_value(value)? {
            Some(intent) => Ok(Some(intent)),
            None => Err(Error::validation(format!(
                "invalid `reasoning` value: expected null, a boolean, a mode or effort string, \
                 or an object with `mode`, `effort` and/or `budget_tokens`; got `{value}`"
            ))),
        },
    }
}

/// Whether the caller supplied an explicit vendor reasoning control.
///
/// The keys are recognized at the top level of `extra` and inside a nested
/// `extra_body` object, matching how an OpenAI-SDK caller expresses vendor
/// extensions.
fn has_legacy_reasoning_controls(extra: &Value) -> bool {
    let Some(object) = extra.as_object() else {
        return false;
    };
    let top_level = LEGACY_REASONING_KEYS
        .iter()
        .any(|key| object.contains_key(*key));
    let nested = object
        .get("extra_body")
        .and_then(Value::as_object)
        .is_some_and(|nested| {
            LEGACY_REASONING_KEYS
                .iter()
                .any(|key| nested.contains_key(*key))
        });
    top_level || nested
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
mod reasoning_tests;
#[cfg(test)]
mod tests;
