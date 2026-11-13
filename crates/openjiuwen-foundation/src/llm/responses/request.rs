//! `OpenAI` Responses request projection: base URL, input items, tools, and body.
//!
//! Ports `build_request_body`, `convert_messages`, `convert_tools`,
//! `expand_nested_extra_body`, `_move_response_format_to_text`, and the message
//! helpers of `openjiuwen/core/foundation/llm/utils/responses_utils.py`.
//! Everything here is pure, so the exact JSON a provider would receive is
//! asserted in tests without a server.

use openjiuwen_common::error::{Error, Result};
use serde_json::{json, Map, Value};

use crate::prompt::content::{ContentPart, MessageContent};
use crate::prompt::{BaseMessage, ToolCall};

use super::super::config::ModelRequestConfig;
use super::super::reasoning::{
    apply_reasoning_plan, resolve_reasoning_plan_for, ReasoningConfig, ReasoningProtocol,
};

/// Suffix a caller may have left on `api_base` by pasting the full endpoint.
const RESPONSES_SUFFIX: &str = "/responses";

/// Key of the provider-neutral reasoning intent inside [`ModelRequestConfig::extra`].
const REASONING_KEY: &str = "reasoning";

/// Key of a nested extra body that must be flattened into the body root.
const EXTRA_BODY_KEY: &str = "extra_body";

/// Key that asks the provider to include encrypted reasoning content.
const INCLUDE_REASONING_KEY: &str = "include_reasoning_encrypted_content";

/// Values the Responses API accepts as `include`.
const REASONING_INCLUDE: &str = "reasoning.encrypted_content";

/// Default tool-selection policy when tools are present (Python `tool_choice`).
const DEFAULT_TOOL_CHOICE: &str = "auto";

/// Provider key used to resolve a reasoning intent; the Responses protocol does
/// not consult it, but the resolver requires one.
const REASONING_PROVIDER: &str = "openai";

/// Removes artifacts from a configured `api_base` so the `/responses` path can be
/// appended to it.
///
/// Whitespace and trailing slashes are trimmed, and a value that already ends in
/// `/responses` loses that suffix, so a caller who pasted the full endpoint does
/// not produce `.../responses/responses`. No `/v1` is ever invented: the provider
/// table owns the default base URL.
///
/// ```
/// use openjiuwen_foundation::llm::responses::normalize_api_base;
///
/// assert_eq!(
///     normalize_api_base(" https://chatgpt.com/backend-api/codex/ "),
///     "https://chatgpt.com/backend-api/codex"
/// );
/// assert_eq!(
///     normalize_api_base("https://api.openai.com/v1/responses"),
///     "https://api.openai.com/v1"
/// );
/// ```
#[must_use]
pub fn normalize_api_base(api_base: &str) -> String {
    let base = api_base.trim().trim_end_matches('/');
    match base.strip_suffix(RESPONSES_SUFFIX) {
        Some(stripped) => stripped.trim_end_matches('/').to_owned(),
        None => base.to_owned(),
    }
}

/// Converts a conversation into Responses `instructions` and `input` items.
///
/// System and developer turns become the `instructions` text, joined with a blank
/// line; tool turns become `function_call_output` items (dropped when they carry no
/// call id); assistant turns become one `output_text` message plus one
/// `function_call` item per requested tool call; every other role becomes a `user`
/// message. Parts that are neither text nor images are dropped, matching the
/// Python projection.
///
/// # Errors
///
/// Returns [`Error::Validation`] when `messages` is empty, because the Responses
/// API rejects a request with no input.
pub fn convert_messages(messages: &[BaseMessage]) -> Result<(String, Vec<Value>)> {
    if messages.is_empty() {
        return Err(Error::validation(
            "OpenAI Responses requests require at least one message",
        ));
    }

    let mut instructions: Vec<String> = Vec::new();
    let mut inputs: Vec<Value> = Vec::new();
    let mut fallback_index = 0_usize;

    for message in messages {
        let role = message.role.as_str();
        let content_text = content_as_text(&message.content);

        if role == "system" || role == "developer" {
            if !content_text.is_empty() {
                instructions.push(content_text);
            }
            continue;
        }

        if role == "tool" {
            if let Some(item) = tool_result_item(message, &content_text) {
                inputs.push(item);
            }
            continue;
        }

        if role == "assistant" {
            if !content_text.is_empty() {
                inputs.push(json!({
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": content_text}],
                }));
            }
            for call in &message.tool_calls {
                inputs.push(function_call_item(call, fallback_index));
                fallback_index += 1;
            }
            continue;
        }

        inputs.push(json!({
            "role": "user",
            "content": user_content_parts(&message.content, &content_text),
        }));
    }

    Ok((instructions.join("\n\n"), inputs))
}

/// Converts tool declarations into the Responses `tools` shape.
///
/// OpenAI-shaped entries (`{"type":"function","function":{...}}`) are flattened
/// into `{"type":"function","name","description","parameters"}` without `strict`;
/// an entry that is already flat but named is kept; anything else passes through
/// unchanged so a vendor extension is never lost.
#[must_use]
pub fn convert_tools(tools: &[Value]) -> Vec<Value> {
    tools.iter().map(tool_definition).collect()
}

/// Builds the `POST {base}/responses` request body for one call.
///
/// The body always contains `model`, `instructions`, `input`, and `store`;
/// sampling parameters, `max_output_tokens`, and `stop` are present only when the
/// request sets them, and `tools`/`parallel_tool_calls`/`tool_choice` only when at
/// least one tool is configured. `include` asks for encrypted reasoning content
/// when `extra.include_reasoning_encrypted_content` is `true`.
///
/// A provider-neutral reasoning intent ([`ModelRequestConfig::reasoning`] first,
/// legacy `extra.reasoning` second) is resolved through
/// [`ReasoningProtocol::Responses`](super::super::reasoning::ReasoningProtocol) and
/// applied before the caller's own `extra` keys, which are merged last and win.
/// A one-level nested `extra.extra_body` object is flattened into the body root
/// (outer keys win), and a chat-style `response_format` moves to `text.format`.
///
/// `stream` adds `"stream": true`; the non-streaming shape omits the field because
/// the client always asks the endpoint to stream.
///
/// # Errors
///
/// Returns [`Error::Config`] when [`ModelRequestConfig::model_name`] is empty and
/// [`Error::Validation`] when the conversation is empty or a reasoning intent is
/// malformed.
pub fn build_responses_body(
    messages: &[BaseMessage],
    request: &ModelRequestConfig,
    stream: bool,
) -> Result<Value> {
    let model_name = request.model_name.trim();
    if model_name.is_empty() {
        return Err(Error::config(
            "model_name must not be empty: set ModelRequestConfig::model_name \
             (or the MODEL_NAME environment variable)",
        ));
    }

    let (instructions, input) = convert_messages(messages)?;

    let mut body = Map::new();
    body.insert("model".to_owned(), Value::String(model_name.to_owned()));
    body.insert("instructions".to_owned(), Value::String(instructions));
    body.insert("input".to_owned(), Value::Array(input));
    body.insert("store".to_owned(), Value::Bool(false));

    if let Some(temperature) = request.temperature.filter(|value| value.is_finite()) {
        body.insert("temperature".to_owned(), sampling_value(temperature));
    }
    if let Some(top_p) = request.top_p.filter(|value| value.is_finite()) {
        body.insert("top_p".to_owned(), sampling_value(top_p));
    }
    if let Some(max_tokens) = request.max_tokens {
        body.insert("max_output_tokens".to_owned(), Value::from(max_tokens));
    }
    if !request.stop.is_empty() {
        body.insert("stop".to_owned(), Value::from(request.stop.clone()));
    }

    let tools = request
        .tools
        .as_ref()
        .and_then(Value::as_array)
        .filter(|tools| !tools.is_empty());
    if let Some(tools) = tools {
        body.insert("tools".to_owned(), Value::Array(convert_tools(tools)));
        body.insert("parallel_tool_calls".to_owned(), Value::Bool(true));
        let choice = request
            .tool_choice
            .clone()
            .unwrap_or_else(|| Value::String(DEFAULT_TOOL_CHOICE.to_owned()));
        body.insert("tool_choice".to_owned(), choice);
    }

    if stream {
        body.insert("stream".to_owned(), Value::Bool(true));
    }

    if include_reasoning_encrypted_content(request) {
        body.insert(
            "include".to_owned(),
            Value::Array(vec![Value::String(REASONING_INCLUDE.to_owned())]),
        );
    }

    let (intent, raw_reasoning, consumes_reasoning) = reasoning_intent(request)?;
    if let Some(intent) = intent {
        let plan = resolve_reasoning_plan_for(
            ReasoningProtocol::Responses,
            REASONING_PROVIDER,
            None,
            model_name,
            &intent,
        );
        for warning in &plan.warnings {
            tracing::debug!(
                target: "openjiuwen_foundation::llm::responses",
                model = model_name,
                "reasoning config resolution: {warning}"
            );
        }
        let mut resolved = Value::Object(std::mem::take(&mut body));
        apply_reasoning_plan(&mut resolved, &plan);
        if let Value::Object(object) = resolved {
            body = object;
        }
    }
    if let Some(raw) = raw_reasoning {
        body.insert(REASONING_KEY.to_owned(), raw);
    }

    if let Some(extra) = request.extra.as_object() {
        for (key, value) in expand_extra_body(extra, consumes_reasoning) {
            body.insert(key, value);
        }
    }

    move_response_format_to_text(&mut body);

    Ok(Value::Object(body))
}

/// Reads the reasoning intent for a request.
///
/// [`ModelRequestConfig::reasoning`] wins over the legacy `extra` spelling.
/// Returns the parsed intent, a legacy raw payload the caller still wants
/// forwarded verbatim, and whether the `extra` `reasoning` key must stay out of
/// the body. A typed value is always consumed by this layer (it is a field, not
/// a body key); a `null` or intent-shaped `extra` value is consumed too, while
/// anything else is a legacy raw payload that keeps its `extra` wiring.
fn reasoning_intent(
    request: &ModelRequestConfig,
) -> Result<(Option<ReasoningConfig>, Option<Value>, bool)> {
    if let Some(value) = request.reasoning.as_ref().filter(|value| !value.is_null()) {
        let intent = ReasoningConfig::from_value(value)?;
        let raw = intent.is_none().then(|| value.clone());
        return Ok((intent, raw, true));
    }
    match request.extra.get(REASONING_KEY) {
        None => Ok((None, None, false)),
        Some(value) => {
            let intent = ReasoningConfig::from_value(value)?;
            let consumed = intent.is_some();
            Ok((intent, None, consumed))
        }
    }
}

/// Whether the caller asked for encrypted reasoning content.
fn include_reasoning_encrypted_content(request: &ModelRequestConfig) -> bool {
    request
        .extra
        .get(INCLUDE_REASONING_KEY)
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

/// Flattens `extra` into the top-level body keys, removing the nested
/// `extra_body` key.
///
/// Nested fields are laid down first and the outer keys win a collision, matching
/// the Python `expand_nested_extra_body`; the consumed keys
/// (`extra_body`, the reasoning intent, and the include toggle) are not forwarded.
fn expand_extra_body(extra: &Map<String, Value>, consumes_reasoning: bool) -> Map<String, Value> {
    let mut expanded = Map::new();
    if let Some(Value::Object(nested)) = extra.get(EXTRA_BODY_KEY) {
        for (key, value) in nested {
            expanded.insert(key.clone(), value.clone());
        }
    }
    for (key, value) in extra {
        if key == EXTRA_BODY_KEY
            || key == INCLUDE_REASONING_KEY
            || (consumes_reasoning && key == REASONING_KEY)
        {
            continue;
        }
        expanded.insert(key.clone(), value.clone());
    }
    expanded
}

/// Moves a chat-style `response_format` to the Responses `text.format` field.
///
/// `json_schema` formats are flattened (`name`, `schema`, `strict`, `description`
/// move next to `type`); other formats are kept as they are. A `text.format` the
/// caller already set takes precedence, and a `response_format` without a `type`
/// stays where it was.
fn move_response_format_to_text(body: &mut Map<String, Value>) {
    let Some(response_format) = body.remove("response_format") else {
        return;
    };
    if response_format.is_null() {
        return;
    }
    let Some(format) = text_format_from_response_format(&response_format) else {
        body.insert("response_format".to_owned(), response_format);
        return;
    };

    let mut text = body
        .get("text")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    if text.contains_key("format") {
        tracing::warn!(
            target: "openjiuwen_foundation::llm::responses",
            "Responses request sets both response_format and text.format; keeping text.format"
        );
    } else {
        text.insert("format".to_owned(), format);
    }
    body.insert("text".to_owned(), Value::Object(text));
}

/// Maps a chat-style `response_format` to a Responses `text.format` value.
fn text_format_from_response_format(response_format: &Value) -> Option<Value> {
    let object = response_format.as_object()?;
    let format_type = object.get("type").and_then(Value::as_str)?;
    if format_type.is_empty() {
        return None;
    }
    if format_type == "json_schema" {
        if let Some(json_schema) = object.get("json_schema").and_then(Value::as_object) {
            let mut flattened = Map::new();
            flattened.insert("type".to_owned(), Value::String(format_type.to_owned()));
            for (key, value) in json_schema {
                flattened.insert(key.clone(), value.clone());
            }
            return Some(Value::Object(flattened));
        }
    }
    Some(response_format.clone())
}

/// Flattens a message's content to text.
///
/// Python's `_content_as_text`: a text payload is returned verbatim, a part list
/// concatenates the text of every text-bearing part, and a null/other payload
/// becomes an empty string.
fn content_as_text(content: &MessageContent) -> String {
    match content {
        MessageContent::Text(text) => text.clone(),
        MessageContent::Parts(parts) => parts.iter().map(part_text).collect(),
    }
}

/// The text of one content part; non-text parts contribute nothing.
fn part_text(part: &ContentPart) -> String {
    match part {
        ContentPart::Text { text } => text.clone(),
        ContentPart::Other(value) => {
            for key in ["text", "content"] {
                if let Some(value) = value.get(key).filter(|value| !value.is_null()) {
                    return match value {
                        Value::String(text) => text.clone(),
                        other => other.to_string(),
                    };
                }
            }
            String::new()
        }
        ContentPart::ImageUrl { .. } | ContentPart::Audio(_) | ContentPart::File(_) => {
            String::new()
        }
    }
}

/// Maps user content to Responses `input_text`/`input_image` parts, in order.
///
/// Adjacent text parts merge into one `input_text`. Parts that are neither text
/// nor images are dropped; an empty result still carries one empty `input_text`,
/// matching the Python projection.
fn user_content_parts(content: &MessageContent, content_text: &str) -> Vec<Value> {
    let MessageContent::Parts(parts) = content else {
        return vec![json!({"type": "input_text", "text": content_text})];
    };

    let mut result: Vec<Value> = Vec::new();
    let mut pending = String::new();
    for part in parts {
        if let Some(image) = input_image_part(part) {
            if !pending.is_empty() {
                result.push(json!({"type": "input_text", "text": std::mem::take(&mut pending)}));
            }
            result.push(image);
            continue;
        }
        pending.push_str(&part_text(part));
    }
    if !pending.is_empty() {
        result.push(json!({"type": "input_text", "text": pending}));
    }

    if result.is_empty() {
        result.push(json!({"type": "input_text", "text": ""}));
    }
    result
}

/// Converts a chat-style `image_url` part to a Responses `input_image` part.
fn input_image_part(part: &ContentPart) -> Option<Value> {
    match part {
        ContentPart::ImageUrl { image_url } => {
            let mut image = Map::new();
            image.insert("type".to_owned(), Value::String("input_image".to_owned()));
            image.insert("image_url".to_owned(), Value::String(image_url.url.clone()));
            if let Some(detail) = image_url.detail.as_ref().filter(|value| !value.is_empty()) {
                image.insert("detail".to_owned(), Value::String(detail.clone()));
            }
            Some(Value::Object(image))
        }
        ContentPart::Other(value)
            if value.get("type").and_then(Value::as_str) == Some("image_url") =>
        {
            let raw = value.get("image_url")?;
            let (url, detail) = match raw {
                Value::String(url) => (url.as_str(), None),
                Value::Object(object) => (
                    object.get("url").and_then(Value::as_str)?,
                    object
                        .get("detail")
                        .and_then(Value::as_str)
                        .filter(|detail| !detail.is_empty()),
                ),
                _ => return None,
            };
            if url.is_empty() {
                return None;
            }
            let mut image = Map::new();
            image.insert("type".to_owned(), Value::String("input_image".to_owned()));
            image.insert("image_url".to_owned(), Value::String(url.to_owned()));
            if let Some(detail) = detail {
                image.insert("detail".to_owned(), Value::String(detail.to_owned()));
            }
            Some(Value::Object(image))
        }
        _ => None,
    }
}

/// Builds a `function_call_output` item from a tool message.
///
/// A tool message with no call id cannot be answered, so it is dropped (Python
/// parity).
fn tool_result_item(message: &BaseMessage, output: &str) -> Option<Value> {
    let call_id = message
        .tool_call_id
        .as_deref()
        .map(str::trim)
        .filter(|id| !id.is_empty())?;
    Some(json!({
        "type": "function_call_output",
        "call_id": call_id,
        "output": output,
    }))
}

/// Builds a `function_call` input item from a replayed assistant tool call.
///
/// A call without an id gets the deterministic `call_{index}` fallback, so a
/// replayed history round-trips without inventing random ids.
fn function_call_item(call: &ToolCall, fallback_index: usize) -> Value {
    let call_id = call
        .id
        .as_deref()
        .filter(|id| !id.is_empty())
        .map_or_else(|| format!("call_{fallback_index}"), str::to_owned);
    json!({
        "type": "function_call",
        "id": call_id,
        "call_id": call_id,
        "name": call.name,
        "arguments": call.arguments,
    })
}

/// Flattens one tool declaration into the Responses shape.
fn tool_definition(tool: &Value) -> Value {
    let Some(object) = tool.as_object() else {
        return tool.clone();
    };
    let function = object.get("function").and_then(Value::as_object);
    let (name, description, parameters) = match function {
        Some(function) => (
            function.get("name").and_then(Value::as_str),
            function.get("description").and_then(Value::as_str),
            function.get("parameters"),
        ),
        None if object.get("type").and_then(Value::as_str) == Some("function")
            && object.get("name").is_some() =>
        {
            (
                object.get("name").and_then(Value::as_str),
                object.get("description").and_then(Value::as_str),
                object.get("parameters"),
            )
        }
        None => return tool.clone(),
    };

    json!({
        "type": "function",
        "name": name.unwrap_or_default(),
        "description": description.unwrap_or_default(),
        "parameters": parameters.cloned().unwrap_or_else(|| json!({})),
    })
}

/// Renders an `f32` sampling parameter as the decimal the caller wrote.
///
/// Rounded to six decimal places so `0.7f32` reaches the wire as `0.7` instead of
/// its `f64` expansion; non-finite values are filtered out by the caller.
fn sampling_value(value: f32) -> Value {
    let rounded = (f64::from(value) * 1_000_000.0).round() / 1_000_000.0;
    serde_json::Number::from_f64(rounded).map_or(Value::Null, Value::Number)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::prompt::content::{AudioPart, FilePart, ImageUrl};
    use crate::prompt::MessageContent;

    fn request() -> ModelRequestConfig {
        ModelRequestConfig::new("gpt-5")
    }

    fn plain_request() -> ModelRequestConfig {
        ModelRequestConfig::new("gpt-5")
    }

    fn build_body(messages: &[BaseMessage], request: &ModelRequestConfig) -> Value {
        build_responses_body(messages, request, false).expect("the body builds")
    }

    #[test]
    fn a_simple_user_message_becomes_one_input_text_item() {
        let body = build_body(&[BaseMessage::user("hello")], &request());

        assert_eq!(
            body,
            json!({
                "model": "gpt-5",
                "instructions": "",
                "input": [{"role": "user", "content": [{"type": "input_text", "text": "hello"}]}],
                "store": false,
            })
        );
    }

    #[test]
    fn system_and_developer_turns_become_instructions() {
        let messages = [
            BaseMessage::system("be brief"),
            BaseMessage {
                role: crate::prompt::Role::Other("developer".to_owned()),
                content: MessageContent::Text("and kind".to_owned()),
                ..BaseMessage::default()
            },
            BaseMessage::user("hello"),
        ];
        let body = build_body(&messages, &request());

        assert_eq!(body["instructions"], json!("be brief\n\nand kind"));
        assert_eq!(body["input"].as_array().expect("input").len(), 1);
    }

    #[test]
    fn assistant_tool_calls_and_tool_results_round_trip() {
        let messages = [
            BaseMessage::user("weather?"),
            BaseMessage::assistant("").with_tool_calls(vec![ToolCall {
                id: Some("call_1".to_owned()),
                name: "get_weather".to_owned(),
                arguments: "{\"city\":\"Paris\"}".to_owned(),
                ..ToolCall::default()
            }]),
            BaseMessage::tool("call_1", "sunny"),
        ];
        let body = build_body(&messages, &request());
        let input = body["input"].as_array().expect("input");

        assert_eq!(
            input[1],
            json!({
                "type": "function_call",
                "id": "call_1",
                "call_id": "call_1",
                "name": "get_weather",
                "arguments": "{\"city\":\"Paris\"}",
            })
        );
        assert_eq!(
            input[2],
            json!({
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "sunny",
            })
        );
    }

    #[test]
    fn tool_messages_without_a_call_id_are_dropped() {
        let body = build_body(
            &[BaseMessage::user("hi"), BaseMessage::tool("", "orphan")],
            &request(),
        );
        assert_eq!(body["input"].as_array().expect("input").len(), 1);
    }

    #[test]
    fn assistant_text_becomes_output_text() {
        let body = build_body(
            &[BaseMessage::user("hi"), BaseMessage::assistant("hello")],
            &request(),
        );
        assert_eq!(
            body["input"][1],
            json!({"role": "assistant", "content": [{"type": "output_text", "text": "hello"}]})
        );
    }

    #[test]
    fn images_become_input_image_parts_and_other_parts_are_dropped() {
        let content = MessageContent::Parts(vec![
            ContentPart::Text {
                text: "look: ".to_owned(),
            },
            ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: "https://example.test/a.png".to_owned(),
                    detail: Some("high".to_owned()),
                },
            },
            ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: "https://example.test/b.png".to_owned(),
                    detail: None,
                },
            },
            ContentPart::Audio(AudioPart {
                data: "AAAA".to_owned(),
                format: Some("wav".to_owned()),
            }),
            ContentPart::File(FilePart {
                url: Some("https://example.test/f.pdf".to_owned()),
                data: None,
                mime_type: None,
                filename: None,
            }),
            ContentPart::Text {
                text: " done".to_owned(),
            },
        ]);
        let body = build_body(&[BaseMessage::user(content)], &request());

        assert_eq!(
            body["input"][0]["content"],
            json!([
                {"type": "input_text", "text": "look: "},
                {"type": "input_image", "image_url": "https://example.test/a.png", "detail": "high"},
                {"type": "input_image", "image_url": "https://example.test/b.png"},
                {"type": "input_text", "text": " done"},
            ])
        );
    }

    #[test]
    fn an_image_only_user_turn_keeps_only_the_image() {
        let content = MessageContent::Parts(vec![ContentPart::ImageUrl {
            image_url: ImageUrl {
                url: "https://example.test/a.png".to_owned(),
                detail: None,
            },
        }]);
        let body = build_body(&[BaseMessage::user(content)], &request());
        assert_eq!(
            body["input"][0]["content"],
            json!([{"type": "input_image", "image_url": "https://example.test/a.png"}])
        );
    }

    #[test]
    fn tools_are_flattened_and_tool_choice_is_forwarded() {
        let request = request()
            .with_tools(vec![json!({
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "weather",
                    "parameters": {"type": "object", "properties": {}},
                    "strict": true
                }
            })])
            .with_tool_choice(json!({"type": "function", "name": "get_weather"}));

        let body = build_body(&[BaseMessage::user("hi")], &request);

        assert_eq!(
            body["tools"],
            json!([{
                "type": "function",
                "name": "get_weather",
                "description": "weather",
                "parameters": {"type": "object", "properties": {}},
            }])
        );
        assert_eq!(body["parallel_tool_calls"], json!(true));
        assert_eq!(
            body["tool_choice"],
            json!({"type": "function", "name": "get_weather"})
        );
    }

    #[test]
    fn tool_choice_defaults_to_auto_and_unknown_shapes_pass_through() {
        let request = request().with_tools(vec![json!({"type": "custom", "name": "x"})]);
        let body = build_body(&[BaseMessage::user("hi")], &request);
        assert_eq!(body["tool_choice"], json!("auto"));
        assert_eq!(body["tools"], json!([{"type": "custom", "name": "x"}]));

        let flat = plain_request().with_tools(vec![json!({
            "type": "function",
            "name": "flat",
            "parameters": {}
        })]);
        let flat_body = build_body(&[BaseMessage::user("hi")], &flat);
        assert_eq!(
            flat_body["tools"],
            json!([{"type": "function", "name": "flat", "description": "", "parameters": {}}])
        );
    }

    #[test]
    fn optional_request_fields_are_added_only_when_set() {
        let request = request()
            .with_temperature(0.7)
            .with_top_p(0.9)
            .with_max_tokens(64)
            .with_stop(vec!["STOP".to_owned()]);
        let body = build_body(&[BaseMessage::user("hi")], &request);

        assert_eq!(body["temperature"], json!(0.7));
        assert_eq!(body["top_p"], json!(0.9));
        assert_eq!(body["max_output_tokens"], json!(64));
        assert_eq!(body["stop"], json!(["STOP"]));
        assert!(body.get("stream").is_none());

        let plain = build_body(&[BaseMessage::user("hi")], &plain_request());
        for key in ["temperature", "top_p", "max_output_tokens", "stop"] {
            assert!(plain.get(key).is_none(), "{key} must stay absent");
        }
        let streaming = build_responses_body(&[BaseMessage::user("hi")], &plain_request(), true)
            .expect("streaming body");
        assert_eq!(streaming["stream"], json!(true));
    }

    #[test]
    fn include_and_reasoning_come_from_extra() {
        let request = request().with_extra(json!({
            "include_reasoning_encrypted_content": true,
            "reasoning": {"mode": "enabled", "effort": "high"},
        }));
        let body = build_body(&[BaseMessage::user("hi")], &request);

        assert_eq!(body["include"], json!(["reasoning.encrypted_content"]));
        assert_eq!(body["reasoning"], json!({"effort": "high"}));
        assert!(
            body.get("include_reasoning_encrypted_content").is_none(),
            "the toggle is consumed by the body builder"
        );
    }

    #[test]
    fn typed_reasoning_matches_the_extra_spelling() {
        // Python's `ModelRequestConfig.reasoning` field emits the same body as
        // the legacy `extra.reasoning` spelling.
        let typed = plain_request().with_reasoning(json!({"mode": "enabled", "effort": "high"}));
        let legacy = plain_request().with_extra(json!({
            "reasoning": {"mode": "enabled", "effort": "high"},
        }));
        let typed_body = build_body(&[BaseMessage::user("hi")], &typed);
        let legacy_body = build_body(&[BaseMessage::user("hi")], &legacy);

        assert_eq!(typed_body, legacy_body);
        assert_eq!(typed_body["reasoning"], json!({"effort": "high"}));
    }

    #[test]
    fn the_typed_reasoning_field_wins_over_the_extra_spelling() {
        let request = plain_request()
            .with_reasoning(json!({"mode": "enabled", "effort": "low"}))
            .with_extra(json!({"reasoning": {"mode": "enabled", "effort": "high"}}));
        let body = build_body(&[BaseMessage::user("hi")], &request);

        assert_eq!(body["reasoning"], json!({"effort": "low"}));
    }

    #[test]
    fn a_typed_legacy_reasoning_payload_is_forwarded_verbatim() {
        let legacy = json!({"enabled": true});
        let request = plain_request().with_reasoning(legacy.clone());
        let body = build_body(&[BaseMessage::user("hi")], &request);

        assert_eq!(body["reasoning"], legacy);
    }

    #[test]
    fn context_window_never_reaches_the_responses_body() {
        let request = plain_request().with_context_window(200_000);
        let body = build_body(&[BaseMessage::user("hi")], &request);
        assert!(body.get("context_window").is_none());
    }

    #[test]
    fn extra_body_is_flattened_and_outer_keys_win() {
        let request = request().with_extra(json!({
            "extra_body": {"thinking": {"type": "enabled"}, "store": true},
            "store": false,
            "seed": 7,
        }));
        let body = build_body(&[BaseMessage::user("hi")], &request);

        assert_eq!(body["thinking"], json!({"type": "enabled"}));
        assert_eq!(body["store"], json!(false), "outer keys win");
        assert_eq!(body["seed"], json!(7));
        assert!(body.get("extra_body").is_none());
    }

    #[test]
    fn response_format_moves_to_text_format() {
        let request = request().with_extra(json!({
            "response_format": {"type": "json_object"},
        }));
        let body = build_body(&[BaseMessage::user("hi")], &request);
        assert_eq!(body["text"], json!({"format": {"type": "json_object"}}));
        assert!(body.get("response_format").is_none());

        let schema_request = plain_request().with_extra(json!({
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "answer", "schema": {"type": "object"}}
            },
        }));
        let schema_body = build_body(&[BaseMessage::user("hi")], &schema_request);
        assert_eq!(
            schema_body["text"],
            json!({"format": {
                "type": "json_schema",
                "name": "answer",
                "schema": {"type": "object"}
            }})
        );
    }

    #[test]
    fn an_existing_text_format_wins_over_response_format() {
        let request = request().with_extra(json!({
            "text": {"format": {"type": "text"}},
            "response_format": {"type": "json_object"},
        }));
        let body = build_body(&[BaseMessage::user("hi")], &request);
        assert_eq!(body["text"], json!({"format": {"type": "text"}}));
    }

    #[test]
    fn an_empty_model_name_is_a_config_error() {
        let request = ModelRequestConfig::new("   ");
        let error = build_responses_body(&[BaseMessage::user("hi")], &request, false)
            .expect_err("a blank model is rejected");
        assert_eq!(error.kind(), openjiuwen_common::error::ErrorKind::Config);
    }

    #[test]
    fn an_empty_conversation_is_a_validation_error() {
        let error = build_responses_body(&[], &request(), false)
            .expect_err("the endpoint needs at least one message");
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Validation
        );
    }

    #[test]
    fn normalize_api_base_strips_a_pasted_endpoint() {
        assert_eq!(
            normalize_api_base("https://chatgpt.com/backend-api/codex/responses"),
            "https://chatgpt.com/backend-api/codex"
        );
        assert_eq!(
            normalize_api_base("  https://api.openai.com/v1/  "),
            "https://api.openai.com/v1"
        );
        assert_eq!(normalize_api_base(""), "");
    }
}
