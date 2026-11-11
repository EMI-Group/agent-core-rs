//! Anthropic Messages request projection.
//!
//! Converts the provider-neutral [`BaseMessage`] list and [`ModelRequestConfig`]
//! into the JSON body of `POST {base}/v1/messages`. The transformations are the
//! ones the Messages API requires and that the OpenAI-compatible wire shape does
//! not express:
//!
//! - every `system` turn is hoisted into the top-level `system` string (the API
//!   only accepts `system` as a request parameter);
//! - assistant `tool_calls` become `tool_use` blocks with a parsed JSON `input`;
//! - consecutive `tool` turns become one `user` message of `tool_result` blocks;
//! - `OpenAI` `image_url` parts become Anthropic `image` blocks (base64 or URL);
//! - `OpenAI` function schemas become `{name, description, input_schema}` tools.
//!
//! The converter is a pure function so tests assert the exact body without a
//! server. It never panics on caller input: an unrepresentable message is an
//! [`Error::Validation`].

use openjiuwen_common::error::{Error, Result};
use serde_json::{json, Map, Value};

use crate::llm::config::ModelRequestConfig;
use crate::prompt::{BaseMessage, ContentPart, ImageUrl, MessageContent, Role, ToolCall};

/// `max_tokens` used when the request does not set one.
const DEFAULT_MAX_TOKENS: u32 = 32768;

/// Output cap shared by the `claude-3.5` families.
const CLAUDE_35_MAX_TOKENS: u32 = 8192;

/// Output cap shared by the original `claude-3` families.
const CLAUDE_3_MAX_TOKENS: u32 = 4096;

/// Marker that separates a data URL's media type from its base64 payload.
const BASE64_MARKER: &str = ";base64,";

/// Number of characters of a rejected value kept in an error message.
const ERROR_PREVIEW_CHARS: usize = 120;

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

    let (system, converted) = convert_messages(messages)?;
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
    if let Some(system) = system {
        body.insert("system".to_owned(), Value::String(system));
    }
    if let Some(tools) = &request.tools {
        let converted_tools = convert_tools(tools);
        if !matches!(&converted_tools, Value::Array(items) if items.is_empty()) {
            body.insert("tools".to_owned(), converted_tools);
        }
    }
    if let Some(tool_choice) = request.tool_choice.as_ref().and_then(convert_tool_choice) {
        body.insert("tool_choice".to_owned(), tool_choice);
    }

    // The caller's `extra` map wins over every derived field, as documented by
    // `ModelRequestConfig::extra`.
    if let Some(extra) = request.extra.as_object() {
        for (key, value) in extra {
            body.insert(key.clone(), value.clone());
        }
    }

    Ok(Value::Object(body))
}

/// Splits a message list into the top-level system string and Anthropic messages.
fn convert_messages(messages: &[BaseMessage]) -> Result<(Option<String>, Vec<Value>)> {
    let mut system_parts: Vec<String> = Vec::new();
    let mut converted: Vec<Value> = Vec::new();
    let mut pending_tool_results: Vec<Value> = Vec::new();

    for message in messages {
        match &message.role {
            Role::System => {
                flush_tool_results(&mut pending_tool_results, &mut converted);
                collect_system_text(message, &mut system_parts);
            }
            Role::Tool => pending_tool_results.push(tool_result_block(message)?),
            Role::Assistant => {
                flush_tool_results(&mut pending_tool_results, &mut converted);
                converted.push(assistant_message(message)?);
            }
            // The Messages API only knows `user` and `assistant`; an unknown role
            // is projected onto `user`, matching the Python client's fallback.
            Role::User | Role::Other(_) => {
                flush_tool_results(&mut pending_tool_results, &mut converted);
                converted.push(json!({
                    "role": "user",
                    "content": content_value(&message.content)?,
                }));
            }
        }
    }
    flush_tool_results(&mut pending_tool_results, &mut converted);

    let system = if system_parts.is_empty() {
        None
    } else {
        Some(system_parts.join("\n\n"))
    };
    Ok((system, converted))
}

/// Emits one `user` message for the tool results accumulated so far.
fn flush_tool_results(pending: &mut Vec<Value>, converted: &mut Vec<Value>) {
    if pending.is_empty() {
        return;
    }
    let blocks = std::mem::take(pending);
    converted.push(json!({"role": "user", "content": blocks}));
}

/// Collects the text blocks of a system turn; non-text parts have no destination.
fn collect_system_text(message: &BaseMessage, parts: &mut Vec<String>) {
    match &message.content {
        MessageContent::Text(text) => {
            if !text.is_empty() {
                parts.push(text.clone());
            }
        }
        MessageContent::Parts(items) => {
            for item in items {
                let text = item.text().unwrap_or_default();
                if !text.is_empty() {
                    parts.push(text.to_owned());
                }
            }
        }
    }
}

/// Converts a tool turn into one Anthropic `tool_result` block.
fn tool_result_block(message: &BaseMessage) -> Result<Value> {
    let tool_call_id = message
        .tool_call_id
        .as_deref()
        .filter(|id| !id.is_empty())
        .ok_or_else(|| {
            Error::validation(
                "a tool message needs `tool_call_id` to become an Anthropic `tool_result` block",
            )
        })?;

    Ok(json!({
        "type": "tool_result",
        "tool_use_id": tool_call_id,
        "content": message.content.to_plain_text(),
    }))
}

/// Converts an assistant turn, including its `tool_use` blocks.
fn assistant_message(message: &BaseMessage) -> Result<Value> {
    if message.tool_calls.is_empty() {
        return Ok(json!({
            "role": "assistant",
            "content": content_value(&message.content)?,
        }));
    }

    let mut blocks = message_blocks(&message.content)?;
    for call in &message.tool_calls {
        blocks.push(tool_use_block(call)?);
    }
    Ok(json!({"role": "assistant", "content": blocks}))
}

/// Renders user/assistant content: a string for text, a block array for parts.
fn content_value(content: &MessageContent) -> Result<Value> {
    match content {
        MessageContent::Text(text) => Ok(Value::String(text.clone())),
        MessageContent::Parts(parts) => Ok(Value::Array(blocks_from_parts(parts)?)),
    }
}

/// Renders content as an Anthropic block array, dropping an empty text block.
fn message_blocks(content: &MessageContent) -> Result<Vec<Value>> {
    match content {
        MessageContent::Text(text) => {
            if text.is_empty() {
                Ok(Vec::new())
            } else {
                Ok(vec![json!({"type": "text", "text": text})])
            }
        }
        MessageContent::Parts(parts) => blocks_from_parts(parts),
    }
}

/// Converts every part of a multimodal message into an Anthropic block.
fn blocks_from_parts(parts: &[ContentPart]) -> Result<Vec<Value>> {
    parts.iter().map(convert_part).collect()
}

/// Converts one provider-neutral content part into an Anthropic block.
///
/// Text and images are re-shaped; audio, file, and vendor blocks are passed
/// through verbatim, so a caller can hand a provider-shaped block to the client.
fn convert_part(part: &ContentPart) -> Result<Value> {
    match part {
        ContentPart::Text { text } => Ok(json!({"type": "text", "text": text})),
        ContentPart::ImageUrl { image_url } => image_block(image_url),
        ContentPart::Audio(_) | ContentPart::File(_) | ContentPart::Other(_) => {
            serde_json::to_value(part).map_err(|error| {
                Error::validation(format!("unsupported Anthropic content part: {error}"))
            })
        }
    }
}

/// Converts an OpenAI-style image reference into an Anthropic `image` block.
fn image_block(image: &ImageUrl) -> Result<Value> {
    let url = image.url.as_str();

    if let Some(rest) = url.strip_prefix("data:") {
        return match rest.split_once(BASE64_MARKER) {
            Some((media_type, data)) if !media_type.is_empty() && !data.is_empty() => Ok(json!({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": data,
                },
            })),
            _ => Err(Error::validation(format!(
                "Anthropic image data URLs must be `data:<media-type>;base64,<payload>`, got `{}`",
                preview(url)
            ))),
        };
    }

    if url.starts_with("http://") || url.starts_with("https://") {
        return Ok(json!({
            "type": "image",
            "source": {"type": "url", "url": url},
        }));
    }

    Err(Error::validation(format!(
        "Anthropic image inputs support `http(s)` URLs or base64 `data:` URLs, got `{}`",
        preview(url)
    )))
}

/// Converts one tool call into an Anthropic `tool_use` block.
fn tool_use_block(call: &ToolCall) -> Result<Value> {
    let id = call
        .id
        .as_deref()
        .filter(|id| !id.is_empty())
        .ok_or_else(|| {
            Error::validation(format!(
                "assistant tool call `{}` has no `id`, which the Anthropic Messages API requires",
                call.name
            ))
        })?;

    Ok(json!({
        "type": "tool_use",
        "id": id,
        "name": call.name,
        "input": tool_input(&call.arguments),
    }))
}

/// Parses a tool call's JSON arguments for the `input` field.
///
/// Empty arguments mean a call without parameters and become `{}`. Text that is
/// not JSON is preserved as `{"_raw_arguments": <raw>}` so the provider still
/// receives the model's own output instead of a silently invented object.
fn tool_input(arguments: &str) -> Value {
    if arguments.trim().is_empty() {
        return json!({});
    }
    match serde_json::from_str::<Value>(arguments) {
        Ok(value) => value,
        Err(_) => json!({"_raw_arguments": arguments}),
    }
}

/// Converts the request's tool definitions into the Anthropic schema shape.
fn convert_tools(tools: &Value) -> Value {
    match tools {
        Value::Array(items) => Value::Array(items.iter().map(convert_tool).collect()),
        other => other.clone(),
    }
}

/// Converts one `OpenAI` function definition; other shapes pass through verbatim.
fn convert_tool(tool: &Value) -> Value {
    let Some(function) = tool.get("function").and_then(Value::as_object) else {
        return tool.clone();
    };

    let name = function
        .get("name")
        .and_then(Value::as_str)
        .or_else(|| tool.get("name").and_then(Value::as_str))
        .unwrap_or_default();
    let description = function
        .get("description")
        .and_then(Value::as_str)
        .or_else(|| tool.get("description").and_then(Value::as_str))
        .unwrap_or_default();
    let input_schema = function
        .get("input_schema")
        .or_else(|| function.get("parameters"))
        .or_else(|| tool.get("input_schema"))
        .cloned()
        .unwrap_or_else(|| json!({"type": "object", "properties": {}}));

    json!({
        "name": name,
        "description": description,
        "input_schema": input_schema,
    })
}

/// Maps an `OpenAI` tool-choice value onto the Anthropic shape.
///
/// `auto` needs no field (it is the API default); unrecognized values are
/// omitted rather than forwarded in a shape the API rejects.
fn convert_tool_choice(choice: &Value) -> Option<Value> {
    match choice {
        Value::String(name) => match name.as_str() {
            "none" => Some(json!({"type": "none"})),
            "required" | "any" => Some(json!({"type": "any"})),
            _ => None,
        },
        Value::Object(object) => {
            if let Some(function) = object.get("function").and_then(Value::as_object) {
                let name = function
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                return Some(json!({"type": "tool", "name": name}));
            }
            match object.get("type").and_then(Value::as_str) {
                Some("none") => Some(json!({"type": "none"})),
                Some("any") => Some(json!({"type": "any"})),
                Some("tool") => Some(choice.clone()),
                _ => None,
            }
        }
        _ => None,
    }
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

/// Returns a bounded, single-line preview of a rejected value.
fn preview(value: &str) -> String {
    let mut preview: String = value.chars().take(ERROR_PREVIEW_CHARS).collect();
    if preview.chars().count() < value.chars().count() {
        preview.push('…');
    }
    preview
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prompt::{ImageUrl, ToolCall};

    fn system_text(body: &Value) -> Option<&str> {
        body.get("system").and_then(Value::as_str)
    }

    fn messages(body: &Value) -> &[Value] {
        body.get("messages")
            .and_then(Value::as_array)
            .map(Vec::as_slice)
            .unwrap_or_default()
    }

    fn request(model: &str) -> ModelRequestConfig {
        ModelRequestConfig::new(model)
    }

    #[test]
    fn normalize_api_base_strips_the_endpoint_and_version() {
        assert_eq!(
            normalize_api_base(" https://api.anthropic.com/v1/messages/ "),
            "https://api.anthropic.com"
        );
        assert_eq!(
            normalize_api_base("https://api.anthropic.com/v1/"),
            "https://api.anthropic.com"
        );
        assert_eq!(
            normalize_api_base("https://api.anthropic.com/"),
            "https://api.anthropic.com"
        );
        assert_eq!(
            normalize_api_base("https://api.anthropic.com"),
            "https://api.anthropic.com"
        );
        assert_eq!(
            normalize_api_base("https://gateway.test/anthropic/v1"),
            "https://gateway.test/anthropic"
        );
        assert_eq!(
            normalize_api_base("https://gateway.test/anthropic/v1/messages"),
            "https://gateway.test/anthropic"
        );
        assert_eq!(normalize_api_base("   "), "");
    }

    #[test]
    fn an_empty_model_name_is_a_validation_error() {
        let error = build_messages_body(&[], &ModelRequestConfig::default(), false)
            .expect_err("the model name is required");
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Validation
        );
        assert!(error.to_string().contains("model_name"));
    }

    #[test]
    fn the_body_carries_the_model_messages_max_tokens_and_stream_flag() {
        let body = build_messages_body(
            &[BaseMessage::user("hello")],
            &request("claude-sonnet-4"),
            false,
        )
        .expect("body");

        assert_eq!(
            body,
            json!({
                "model": "claude-sonnet-4",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 32768,
                "stream": false,
            })
        );
    }

    #[test]
    fn the_stream_flag_is_written_verbatim() {
        let body = build_messages_body(
            &[BaseMessage::user("hi")],
            &request("claude-sonnet-4"),
            true,
        )
        .expect("body");
        assert_eq!(body.get("stream"), Some(&Value::Bool(true)));
    }

    #[test]
    fn sampling_top_p_and_stop_are_mapped() {
        let request = request("claude-sonnet-4")
            .with_temperature(0.2)
            .with_top_p(0.9)
            .with_stop(["END".to_owned(), "STOP".to_owned()]);
        let body = build_messages_body(&[BaseMessage::user("hi")], &request, false).expect("body");

        assert_eq!(body.get("temperature"), Some(&json!(0.2_f32)));
        assert_eq!(body.get("top_p"), Some(&json!(0.9_f32)));
        assert_eq!(body.get("stop_sequences"), Some(&json!(["END", "STOP"])));
        assert!(
            body.get("stop").is_none(),
            "the OpenAI key must not be sent"
        );
    }

    #[test]
    fn system_turns_are_joined_and_removed_from_messages() {
        let body = build_messages_body(
            &[
                BaseMessage::system("be brief"),
                BaseMessage::system(vec![
                    ContentPart::Text {
                        text: "and kind".to_owned(),
                    },
                    ContentPart::Audio(crate::prompt::AudioPart {
                        data: "ignored".to_owned(),
                        format: None,
                    }),
                ]),
                BaseMessage::user("hello"),
            ],
            &request("claude-sonnet-4"),
            false,
        )
        .expect("body");

        assert_eq!(system_text(&body), Some("be brief\n\nand kind"));
        assert_eq!(
            messages(&body),
            [json!({"role": "user", "content": "hello"})]
        );
    }

    #[test]
    fn an_empty_system_turn_omits_the_field() {
        let body = build_messages_body(
            &[
                BaseMessage::system(""),
                BaseMessage::system(vec![ContentPart::Other(json!({"type": "image"}))]),
                BaseMessage::user("hi"),
            ],
            &request("claude-sonnet-4"),
            false,
        )
        .expect("body");

        assert!(system_text(&body).is_none());
        assert_eq!(messages(&body).len(), 1);
    }

    #[test]
    fn multimodal_parts_become_anthropic_image_blocks() {
        let body = build_messages_body(
            &[BaseMessage::user(vec![
                ContentPart::Text {
                    text: "what is this?".to_owned(),
                },
                ContentPart::ImageUrl {
                    image_url: ImageUrl {
                        url: "data:image/png;base64,AAAA".to_owned(),
                        detail: None,
                    },
                },
                ContentPart::ImageUrl {
                    image_url: ImageUrl {
                        url: "https://example.test/cat.jpg".to_owned(),
                        detail: Some("high".to_owned()),
                    },
                },
            ])],
            &request("claude-sonnet-4"),
            false,
        )
        .expect("body");

        assert_eq!(
            messages(&body),
            [json!({
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this?"},
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"},
                    },
                    {
                        "type": "image",
                        "source": {"type": "url", "url": "https://example.test/cat.jpg"},
                    },
                ],
            })]
        );
    }

    #[test]
    fn unsupported_image_urls_are_validation_errors() {
        for url in [
            "ftp://example.test/cat.png",
            "data:image/png;base64,",
            "data:image/png,not-base64",
            "relative/cat.png",
        ] {
            let error = build_messages_body(
                &[BaseMessage::user(vec![ContentPart::ImageUrl {
                    image_url: ImageUrl {
                        url: url.to_owned(),
                        detail: None,
                    },
                }])],
                &request("claude-sonnet-4"),
                false,
            )
            .expect_err("the image cannot be represented");

            assert_eq!(
                error.kind(),
                openjiuwen_common::error::ErrorKind::Validation,
                "url: {url}"
            );
            let rendered = error.to_string();
            assert!(rendered.contains("Anthropic image"), "error: {rendered}");
        }
    }

    #[test]
    fn audio_file_and_other_parts_pass_through_verbatim() {
        let vendor = json!({"type": "document", "source": {"type": "text", "media_type": "text/plain", "data": "x"}});
        let body = build_messages_body(
            &[BaseMessage::user(vec![
                ContentPart::Audio(crate::prompt::AudioPart {
                    data: "AAAA".to_owned(),
                    format: Some("wav".to_owned()),
                }),
                ContentPart::File(crate::prompt::FilePart {
                    url: Some("https://example.test/a.pdf".to_owned()),
                    data: None,
                    mime_type: Some("application/pdf".to_owned()),
                    filename: None,
                }),
                ContentPart::Other(vendor.clone()),
            ])],
            &request("claude-sonnet-4"),
            false,
        )
        .expect("body");

        assert_eq!(
            messages(&body)[0]["content"],
            json!([
                {"type": "audio", "audio": {"data": "AAAA", "format": "wav"}},
                {
                    "type": "file",
                    "file": {
                        "url": "https://example.test/a.pdf",
                        "mime_type": "application/pdf",
                    },
                },
                vendor,
            ])
        );
    }

    #[test]
    fn assistant_tool_calls_become_tool_use_blocks() {
        let body = build_messages_body(
            &[BaseMessage::assistant("let me check").with_tool_calls(vec![
                ToolCall::new("get_weather")
                    .with_id("call_1")
                    .with_arguments(json!({"city": "Paris"})),
                ToolCall::new("get_time").with_id("call_2"),
            ])],
            &request("claude-sonnet-4"),
            false,
        )
        .expect("body");

        assert_eq!(
            messages(&body),
            [json!({
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "let me check"},
                    {
                        "type": "tool_use",
                        "id": "call_1",
                        "name": "get_weather",
                        "input": {"city": "Paris"},
                    },
                    {
                        "type": "tool_use",
                        "id": "call_2",
                        "name": "get_time",
                        "input": {},
                    },
                ],
            })]
        );
    }

    #[test]
    fn unparseable_tool_arguments_are_preserved() {
        let mut call = ToolCall::new("broken").with_id("call_1");
        call.arguments = "{not json".to_owned();
        let body = build_messages_body(
            &[BaseMessage::assistant("").with_tool_calls(vec![call])],
            &request("claude-sonnet-4"),
            false,
        )
        .expect("body");

        assert_eq!(
            messages(&body)[0]["content"],
            json!([{
                "type": "tool_use",
                "id": "call_1",
                "name": "broken",
                "input": {"_raw_arguments": "{not json"},
            }])
        );
    }

    #[test]
    fn a_tool_call_without_an_id_is_a_validation_error() {
        let error = build_messages_body(
            &[BaseMessage::assistant("").with_tool_calls(vec![ToolCall::new("nameless")])],
            &request("claude-sonnet-4"),
            false,
        )
        .expect_err("the API requires an id");

        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Validation
        );
        assert!(error.to_string().contains("id"));
    }

    #[test]
    fn consecutive_tool_messages_merge_into_one_user_message() {
        let body = build_messages_body(
            &[
                BaseMessage::user("run both"),
                BaseMessage::assistant("").with_tool_calls(vec![
                    ToolCall::new("a").with_id("call_1"),
                    ToolCall::new("b").with_id("call_2"),
                ]),
                BaseMessage::tool("call_1", "first"),
                BaseMessage::tool(
                    "call_2",
                    vec![ContentPart::Text {
                        text: "second".to_owned(),
                    }],
                ),
            ],
            &request("claude-sonnet-4"),
            false,
        )
        .expect("body");

        let messages = messages(&body);
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0], json!({"role": "user", "content": "run both"}));
        assert_eq!(messages[1]["role"], "assistant");
        assert_eq!(
            messages[2],
            json!({
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "call_1", "content": "first"},
                    {"type": "tool_result", "tool_use_id": "call_2", "content": "second"},
                ],
            })
        );
    }

    #[test]
    fn a_tool_message_without_a_call_id_is_a_validation_error() {
        let mut message = BaseMessage::tool("call_1", "result");
        message.tool_call_id = None;
        let error = build_messages_body(&[message], &request("claude-sonnet-4"), false)
            .expect_err("tool_call_id is required");

        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Validation
        );
        assert!(error.to_string().contains("tool_call_id"));
    }

    #[test]
    fn tools_are_converted_from_the_openai_shape() {
        let body = build_messages_body(
            &[BaseMessage::user("hi")],
            &request("claude-sonnet-4").with_tools(vec![
                json!({
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Weather lookup",
                        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                    },
                }),
                json!({
                    "type": "function",
                    "function": {
                        "name": "get_time",
                        "input_schema": {"type": "object", "properties": {}},
                    },
                }),
                json!({"name": "already_anthropic", "input_schema": {"type": "object"}}),
            ]),
            false,
        )
        .expect("body");

        assert_eq!(
            body.get("tools"),
            Some(&json!([
                {
                    "name": "get_weather",
                    "description": "Weather lookup",
                    "input_schema": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                },
                {
                    "name": "get_time",
                    "description": "",
                    "input_schema": {"type": "object", "properties": {}},
                },
                {
                    "name": "already_anthropic",
                    "input_schema": {"type": "object"},
                },
            ]))
        );
    }

    #[test]
    fn an_empty_tool_list_is_omitted() {
        let body = build_messages_body(
            &[BaseMessage::user("hi")],
            &request("claude-sonnet-4").with_tools(Vec::new()),
            false,
        )
        .expect("body");
        assert!(body.get("tools").is_none());
    }

    #[test]
    fn tool_choice_values_are_mapped() {
        let cases = [
            (json!("auto"), None),
            (json!("none"), Some(json!({"type": "none"}))),
            (json!("required"), Some(json!({"type": "any"}))),
            (json!("any"), Some(json!({"type": "any"}))),
            (
                json!({"type": "function", "function": {"name": "get_weather"}}),
                Some(json!({"type": "tool", "name": "get_weather"})),
            ),
            (
                json!({"type": "tool", "name": "get_time"}),
                Some(json!({"type": "tool", "name": "get_time"})),
            ),
            (json!({"type": "unknown"}), None),
            (json!(42), None),
        ];

        for (choice, expected) in cases {
            let body = build_messages_body(
                &[BaseMessage::user("hi")],
                &request("claude-sonnet-4").with_tool_choice(choice.clone()),
                false,
            )
            .expect("body");
            assert_eq!(
                body.get("tool_choice"),
                expected.as_ref(),
                "tool_choice: {choice}"
            );
        }
    }

    #[test]
    fn max_tokens_uses_the_default_and_the_family_caps() {
        let cases: [(&str, Option<u32>, u32); 8] = [
            ("claude-sonnet-4", None, 32768),
            ("claude-3-5-sonnet-latest", None, 8192),
            ("claude-3.5-sonnet", None, 8192),
            ("Claude-3-Opus-20240229", None, 4096),
            ("claude-3-sonnet-20240229", Some(16_384), 4096),
            ("claude-3-haiku-20240307", Some(1024), 1024),
            ("claude-3-5-sonnet", Some(32_000), 8192),
            ("claude-sonnet-4", Some(64_000), 64_000),
        ];

        for (model, requested, expected) in cases {
            let request = ModelRequestConfig {
                max_tokens: requested,
                ..ModelRequestConfig::new(model)
            };
            let body =
                build_messages_body(&[BaseMessage::user("hi")], &request, false).expect("body");
            assert_eq!(
                body.get("max_tokens"),
                Some(&Value::from(expected)),
                "model: {model}, requested: {requested:?}"
            );
        }
    }

    #[test]
    fn extra_is_merged_last_and_wins() {
        let request = request("claude-sonnet-4")
            .with_temperature(0.2)
            .with_extra(json!({"temperature": 0.7, "custom_flag": true, "model": "override"}));
        let body = build_messages_body(&[BaseMessage::user("hi")], &request, false).expect("body");

        assert_eq!(body.get("temperature"), Some(&json!(0.7)));
        assert_eq!(body.get("custom_flag"), Some(&json!(true)));
        assert_eq!(body.get("model"), Some(&json!("override")));
    }

    #[test]
    fn non_object_extra_is_ignored() {
        let request = request("claude-sonnet-4").with_extra(json!([1, 2, 3]));
        let body = build_messages_body(&[BaseMessage::user("hi")], &request, false).expect("body");
        assert_eq!(body.get("model"), Some(&json!("claude-sonnet-4")));
    }
}
