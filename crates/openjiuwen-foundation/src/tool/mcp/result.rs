//! Normalization of MCP tool-call results.
//!
//! Ports `extract_mcp_tool_result_content` and `preserve_mcp_tool_result_status`
//! from `openjiuwen/core/foundation/tool/mcp/base.py`: an MCP `content` block
//! list becomes a [`McpToolResult`] whose `data` map carries the model-visible
//! payload (`content` for text, `multimodal` for inline images, `resources` for
//! embedded resources) and whose `success`/`error` mirror the server's
//! `isError` flag.
//!
//! The Python function returns a bare `str`/`dict` for successful calls and a
//! `McpToolResult` only for errors (the caller then wraps bare values as
//! `{"result": ...}`). Rust has one uniform shape instead: `data` is always a
//! map, and a single text block lands there under `content`. The rendering
//! [`crate::tool::mcp::tool::McpTool`] performs is identical to Python's.

use rmcp::model::{CallToolResult, Content, RawContent};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// Error text used when an MCP server reports `isError` without any text block.
///
/// Python parity: `"MCP tool reported an error."`.
pub const DEFAULT_MCP_ERROR_TEXT: &str = "MCP tool reported an error.";

/// The normalized result of one MCP tool call.
///
/// Python parity: `McpToolResult` in
/// `openjiuwen/core/foundation/tool/mcp/base.py`. `success` defaults to `true`
/// so a payload that omits it reads as a success, matching the Python default.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct McpToolResult {
    /// Whether the call succeeded.
    pub success: bool,
    /// Model-visible payload; `content`, `multimodal`, and `resources` keys.
    pub data: Map<String, Value>,
    /// Failure message, present when the server reported an error.
    pub error: Option<String>,
}

impl Default for McpToolResult {
    fn default() -> Self {
        Self {
            success: true,
            data: Map::new(),
            error: None,
        }
    }
}

impl McpToolResult {
    /// Creates a successful result carrying `data`.
    #[must_use]
    pub fn success(data: Map<String, Value>) -> Self {
        Self {
            success: true,
            data,
            error: None,
        }
    }

    /// Creates a failed result carrying `error` and an empty payload.
    #[must_use]
    pub fn failure(error: impl Into<String>) -> Self {
        Self {
            success: false,
            data: Map::new(),
            error: Some(error.into()),
        }
    }

    /// Returns whether the call succeeded.
    #[must_use]
    pub fn is_success(&self) -> bool {
        self.success
    }

    /// Returns the payload map.
    #[must_use]
    pub fn data_map(&self) -> &Map<String, Value> {
        &self.data
    }

    /// Returns the failure message, if any.
    #[must_use]
    pub fn error_msg(&self) -> Option<&str> {
        self.error.as_deref()
    }
}

/// Normalizes a raw MCP `CallToolResult` into a [`McpToolResult`].
///
/// Python parity: `preserve_mcp_tool_result_status` wrapping
/// `extract_mcp_tool_result_content`. Content is normalized first; when the
/// server sets `isError: true` the result is marked failed and its `error` is
/// the joined text blocks (never the normalized payload, which may instead hold
/// images or resources).
#[must_use]
pub fn normalize_call_tool_result(
    result: &CallToolResult,
    include_image_content: bool,
) -> McpToolResult {
    let mut normalized = extract_content(&result.content, include_image_content);
    if result.is_error != Some(true) {
        return normalized;
    }
    let error = result
        .content
        .iter()
        .filter_map(|content| match &content.raw {
            RawContent::Text(text) if !text.text.is_empty() => Some(text.text.clone()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n\n");
    normalized.success = false;
    normalized.error = Some(if error.trim().is_empty() {
        DEFAULT_MCP_ERROR_TEXT.to_owned()
    } else {
        error
    });
    normalized
}

/// Normalizes an MCP content block list into a successful [`McpToolResult`].
///
/// Rules (Python `extract_mcp_tool_result_content`):
/// - text blocks keep their order and are joined with `"\n\n"` under `content`;
/// - images become `[image content: <mime>, <n> base64 chars]` placeholders, or,
///   when `include_image_content` is set, `multimodal` entries carrying a data
///   URL plus an `"<n> image(s) attached as multimodal input."` content note;
/// - audio blocks become `[audio content: <mime>, <n> base64 chars]`;
/// - embedded resources and resource links are serialized: a single one and no
///   text lands directly under `content`, otherwise under a `resources` array;
/// - an empty block list yields an empty `data` map.
#[must_use]
pub fn extract_content(contents: &[Content], include_image_content: bool) -> McpToolResult {
    let mut text_parts: Vec<String> = Vec::new();
    let mut images: Vec<(String, String)> = Vec::new();
    let mut resources: Vec<Value> = Vec::new();

    for content in contents {
        match &content.raw {
            RawContent::Text(text) => text_parts.push(text.text.clone()),
            RawContent::Image(image) => {
                if include_image_content {
                    images.push((image.mime_type.clone(), image.data.clone()));
                } else {
                    text_parts.push(base64_placeholder("image", &image.mime_type, &image.data));
                }
            }
            RawContent::Audio(audio) => {
                text_parts.push(base64_placeholder("audio", &audio.mime_type, &audio.data));
            }
            RawContent::Resource(resource) => resources.push(json_value(&resource.resource)),
            RawContent::ResourceLink(link) => resources.push(json_value(link)),
        }
    }

    let mut data = Map::new();
    let mut text = text_parts.join("\n\n");
    if !images.is_empty() {
        let note = format!("{} image(s) attached as multimodal input.", images.len());
        text = if text.is_empty() {
            note
        } else {
            format!("{text}\n\n{note}")
        };
        let multimodal = images
            .iter()
            .map(|(mime_type, payload)| {
                let mut entry = Map::new();
                entry.insert("type".to_owned(), Value::String("image".to_owned()));
                entry.insert("source".to_owned(), Value::String("mcp".to_owned()));
                entry.insert(
                    "source_path".to_owned(),
                    Value::String("mcp_tool".to_owned()),
                );
                entry.insert("mime_type".to_owned(), Value::String(mime_type.clone()));
                entry.insert(
                    "data_url".to_owned(),
                    Value::String(format!("data:{mime_type};base64,{payload}")),
                );
                Value::Object(entry)
            })
            .collect();
        data.insert("multimodal".to_owned(), Value::Array(multimodal));
    }
    if !text.is_empty() {
        data.insert("content".to_owned(), Value::String(text));
    }
    if !resources.is_empty() {
        if resources.len() == 1 && !data.contains_key("content") {
            if let Some(single) = resources.pop() {
                data.insert("content".to_owned(), single);
            }
        } else {
            data.insert("resources".to_owned(), Value::Array(resources));
        }
    }

    McpToolResult::success(data)
}

/// Builds the `[<kind> content: <mime>, <n> base64 chars]` placeholder.
///
/// Python counts `len(str(data))`, i.e. characters, so multi-byte payloads count
/// as one per code point rather than per byte.
fn base64_placeholder(kind: &str, mime_type: &str, payload: &str) -> String {
    format!(
        "[{kind} content: {mime_type}, {} base64 chars]",
        payload.chars().count()
    )
}

/// Serializes an MCP model value, falling back to JSON null.
fn json_value(value: &impl Serialize) -> Value {
    serde_json::to_value(value).unwrap_or(Value::Null)
}
