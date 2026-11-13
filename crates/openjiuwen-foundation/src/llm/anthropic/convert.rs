//! OpenAI-shape → Anthropic Messages projection.
//!
//! Splits a provider-neutral [`BaseMessage`] list into the top-level `system`
//! block array and the `user`/`assistant` messages the Messages API accepts:
//!
//! - each `system`/developer turn becomes its own `text` block; the API takes
//!   `system` only as a request parameter, so the turn is hoisted out of
//!   `messages` unless the target model accepts mid-conversation system turns
//!   *and* the position is one the API allows (see
//!   [`convert_messages`]);
//! - assistant `tool_calls` become `tool_use` blocks with a parsed JSON `input`;
//! - consecutive `tool` turns become one `user` message of `tool_result` blocks;
//! - `OpenAI` `image_url` parts become Anthropic `image` blocks (base64 or URL);
//! - `OpenAI` function schemas become `{name, description, input_schema}` tools.
//!
//! The converter is pure, so tests assert the exact body without a server. It
//! never panics on caller input: an unrepresentable message is an
//! [`Error::Validation`].

use openjiuwen_common::error::{Error, Result};
use serde_json::{json, Value};

use crate::prompt::{BaseMessage, ContentPart, ImageUrl, MessageContent, Role, ToolCall};

/// Model families that accept `role: "system"` inside `messages`, GA and with
/// no beta header. Every other model answers 400 ("role 'system' is not
/// supported on this model"), so the turn is folded into top-level `system`.
const MID_CONVERSATION_SYSTEM_FAMILIES: [&str; 4] = [
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-fable-5",
    "claude-mythos-5",
];

/// Marker that separates a data URL's media type from its base64 payload.
const BASE64_MARKER: &str = ";base64,";

/// Number of characters of a rejected value kept in an error message.
const ERROR_PREVIEW_CHARS: usize = 120;

/// The projected request: top-level system blocks and the messages to send.
///
/// `messages` carries a `system` role only when the target model accepts
/// mid-conversation system turns and the position was legal.
pub(crate) struct ConvertedMessages {
    /// One `{"type": "text", "text": …}` block per hoisted system turn.
    pub(crate) system: Vec<Value>,
    /// `user`/`assistant` messages, in order.
    pub(crate) messages: Vec<Value>,
}

/// Splits a message list into top-level system blocks and Anthropic messages.
///
/// System turns are laid out in place first and partitioned afterwards,
/// because whether one may stay where it was written depends on the neighbours
/// it ends up with. A turn that stays keeps the cached conversation prefix
/// intact; a turn that is hoisted is emitted as its own top-level block (never
/// joined with another turn, never `<system>`-prefixed, never dropped). Empty
/// system turns have nothing to send and are skipped.
pub(crate) fn convert_messages(messages: &[BaseMessage], model: &str) -> Result<ConvertedMessages> {
    let allow_inline_system = supports_mid_conversation_system(model);
    let mut in_place: Vec<Value> = Vec::new();
    let mut pending_tool_results: Vec<Value> = Vec::new();

    for message in messages {
        match &message.role {
            Role::System => {
                flush_tool_results(&mut pending_tool_results, &mut in_place);
                let text = system_turn_text(message);
                if !text.is_empty() {
                    in_place.push(json!({"role": "system", "content": text}));
                }
            }
            Role::Tool => pending_tool_results.push(tool_result_block(message)?),
            Role::Assistant => {
                flush_tool_results(&mut pending_tool_results, &mut in_place);
                in_place.push(assistant_message(message)?);
            }
            // The Messages API only knows `user` and `assistant`; an unknown role
            // is projected onto `user`, matching the Python client's fallback.
            Role::User | Role::Other(_) => {
                flush_tool_results(&mut pending_tool_results, &mut in_place);
                in_place.push(json!({
                    "role": "user",
                    "content": content_value(&message.content)?,
                }));
            }
        }
    }
    flush_tool_results(&mut pending_tool_results, &mut in_place);

    Ok(partition_system_turns(in_place, allow_inline_system))
}

/// Moves the system turns the API cannot keep in place into the top-level list.
fn partition_system_turns(in_place: Vec<Value>, allow_inline_system: bool) -> ConvertedMessages {
    let mut system: Vec<Value> = Vec::new();
    let mut messages: Vec<Value> = Vec::new();

    for (index, message) in in_place.iter().enumerate() {
        if message_role(message) != Some("system") {
            messages.push(message.clone());
            continue;
        }
        if allow_inline_system && inline_system_is_placeable(&in_place, index) {
            messages.push(message.clone());
            continue;
        }
        let text = message
            .get("content")
            .and_then(Value::as_str)
            .unwrap_or_default();
        if !text.is_empty() {
            system.push(json!({"type": "text", "text": text}));
        }
    }

    ConvertedMessages { system, messages }
}

/// Reports whether the model accepts `role: "system"` inside `messages`.
///
/// Gateways prefix and suffix the id (`anthropic/`, `us.anthropic.`, `-v1:0`),
/// so the family is matched as a substring; an unrecognized id keeps the
/// always-valid behaviour of folding the turn into top-level `system`.
fn supports_mid_conversation_system(model: &str) -> bool {
    let name = model.trim().to_ascii_lowercase();
    !name.is_empty()
        && MID_CONVERSATION_SYSTEM_FAMILIES
            .iter()
            .any(|family| name.contains(family))
}

/// Reports whether an in-place system turn may stay where it sits.
///
/// The API accepts one only after a `user` turn, and only when it is the last
/// message or is followed by an `assistant` turn. It can never be
/// `messages[0]` — that is the leading system prompt, which keeps going to the
/// top-level parameter it belongs in. Neighbours are read from the pre-partition
/// list on purpose: two adjacent system turns are judged against each other and
/// both fall back instead of each assuming the other will move away.
fn inline_system_is_placeable(converted: &[Value], index: usize) -> bool {
    if index == 0 {
        return false;
    }
    if message_role(&converted[index - 1]) != Some("user") {
        return false;
    }
    if index == converted.len() - 1 {
        return true;
    }
    message_role(&converted[index + 1]) == Some("assistant")
}

/// Emits one `user` message for the tool results accumulated so far.
fn flush_tool_results(pending: &mut Vec<Value>, converted: &mut Vec<Value>) {
    if pending.is_empty() {
        return;
    }
    let blocks = std::mem::take(pending);
    converted.push(json!({"role": "user", "content": blocks}));
}

/// Flattens a system turn to the plain text a system block allows.
///
/// A system turn is text-only, so image and tool parts have nowhere to go and
/// are dropped rather than sent in a shape the API rejects; text parts are
/// joined with a newline.
fn system_turn_text(message: &BaseMessage) -> String {
    match &message.content {
        MessageContent::Text(text) => text.clone(),
        MessageContent::Parts(parts) => parts
            .iter()
            .filter_map(ContentPart::text)
            .filter(|text| !text.is_empty())
            .collect::<Vec<_>>()
            .join("\n"),
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
pub(crate) fn convert_tools(tools: &Value) -> Value {
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
pub(crate) fn convert_tool_choice(choice: &Value) -> Option<Value> {
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

/// The `role` of a converted message, when it has one.
fn message_role(message: &Value) -> Option<&str> {
    message.get("role").and_then(Value::as_str)
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

    #[test]
    fn mid_conversation_system_support_is_a_family_match() {
        for model in [
            "claude-opus-4-8",
            "anthropic/claude-opus-5",
            "us.anthropic.claude-fable-5-v1:0",
            "CLAUDE-MYTHOS-5",
        ] {
            assert!(
                supports_mid_conversation_system(model),
                "model: {model} should accept inline system turns"
            );
        }
        for model in ["", "claude-sonnet-4", "claude-opus-4-7", "claude-opus-8"] {
            assert!(
                !supports_mid_conversation_system(model),
                "model: {model} must fold system turns into the top-level parameter"
            );
        }
    }

    #[test]
    fn a_system_turn_flattens_to_its_text_parts() {
        let parts = MessageContent::Parts(vec![
            ContentPart::Text {
                text: String::new(),
            },
            ContentPart::Text {
                text: "first".to_owned(),
            },
            ContentPart::Audio(crate::prompt::AudioPart {
                data: "ignored".to_owned(),
                format: None,
            }),
            ContentPart::Text {
                text: "second".to_owned(),
            },
        ]);
        assert_eq!(
            system_turn_text(&BaseMessage::system(parts)),
            "first\nsecond"
        );
        assert_eq!(system_turn_text(&BaseMessage::system("plain")), "plain");
        assert_eq!(system_turn_text(&BaseMessage::system("")), "");
    }
}
