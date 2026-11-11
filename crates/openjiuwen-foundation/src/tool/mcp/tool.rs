//! [`McpTool`]: a remote MCP tool presented as a local [`Tool`].
//!
//! Ports `MCPTool` from `openjiuwen/core/foundation/tool/mcp/base.py`: the card
//! is the model-facing projection, the client is the transport, and `invoke`
//! validates inputs, calls the server, and returns the normalized
//! [`McpToolResult`] as JSON.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::tool::card::ToolCard;
use crate::tool::context::ToolContext;
use crate::tool::schema::{apply_defaults, ensure_object_schema, validate_inputs};
use crate::tool::traits::{render_payload_text, Tool, EMPTY_SUCCESS_TEXT};

use super::client::{McpClient, McpRemoteTool};
use super::result::{McpToolResult, DEFAULT_MCP_ERROR_TEXT};

/// Per-tool behavior for an [`McpTool`].
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct McpToolOptions {
    /// Forward image blocks as `multimodal` data instead of text placeholders.
    pub include_image_content: bool,
    /// Default per-call deadline, used when the [`ToolContext`] has none.
    pub timeout: Option<Duration>,
}

impl McpToolOptions {
    /// Returns these options with the image-forwarding flag set.
    #[must_use]
    pub fn with_include_image_content(mut self, include: bool) -> Self {
        self.include_image_content = include;
        self
    }

    /// Returns these options with a default per-call deadline.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }
}

/// A tool served by an MCP server.
///
/// The card carries `metadata` entries `mcp: true`, `server_name`, and
/// `server_id`; `invoke` returns the JSON form of the normalized
/// [`McpToolResult`] and reports server-side failures as [`Error::Mcp`].
pub struct McpTool {
    card: ToolCard,
    client: Arc<dyn McpClient>,
    tool_name: String,
    options: McpToolOptions,
}

impl std::fmt::Debug for McpTool {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("McpTool")
            .field("card", &self.card)
            .field("tool_name", &self.tool_name)
            .field("server_name", &self.client.server_name())
            .field("options", &self.options)
            .finish()
    }
}

impl McpTool {
    /// Creates a tool for `tool_name` on `client`.
    #[must_use]
    pub fn new(card: ToolCard, client: Arc<dyn McpClient>, tool_name: impl Into<String>) -> Self {
        Self {
            card,
            client,
            tool_name: tool_name.into(),
            options: McpToolOptions::default(),
        }
    }

    /// Returns this tool with `options` applied.
    #[must_use]
    pub fn with_options(mut self, options: McpToolOptions) -> Self {
        self.options = options;
        self
    }

    /// Returns the client this tool calls through.
    #[must_use]
    pub fn client(&self) -> &Arc<dyn McpClient> {
        &self.client
    }

    /// Returns the remote tool name.
    #[must_use]
    pub fn tool_name(&self) -> &str {
        &self.tool_name
    }

    /// Returns the effective options.
    #[must_use]
    pub fn options(&self) -> &McpToolOptions {
        &self.options
    }

    /// Returns this tool as a shared trait object.
    #[must_use]
    pub fn boxed(self) -> Arc<dyn Tool> {
        Arc::new(self)
    }
}

#[async_trait]
impl Tool for McpTool {
    fn card(&self) -> &ToolCard {
        &self.card
    }

    async fn invoke(&self, inputs: Value, ctx: &ToolContext) -> Result<Value> {
        let schema = ensure_object_schema(self.card.input_params.clone());
        let mut arguments = if inputs.is_object() {
            inputs
        } else {
            json!({})
        };
        apply_defaults(&schema, &mut arguments);
        validate_inputs(&schema, &arguments)?;
        ctx.check_cancelled()?;

        let timeout = ctx.timeout.or(self.options.timeout);
        let result = self
            .client
            .call_tool_with(
                &self.tool_name,
                arguments,
                timeout,
                self.options.include_image_content,
            )
            .await
            .map_err(|error| match error {
                Error::Cancelled | Error::Timeout(_) => error,
                other => Error::mcp(format!("MCP tool `{}` failed: {other}", self.card.name)),
            })?;
        ctx.check_cancelled()?;

        if !result.success {
            return Err(Error::mcp(
                result
                    .error
                    .unwrap_or_else(|| DEFAULT_MCP_ERROR_TEXT.to_owned()),
            ));
        }
        serde_json::to_value(&result)
            .map_err(|error| Error::mcp(format!("failed to serialize MCP result: {error}")))
    }

    /// Renders the normalized result for the model.
    ///
    /// Python parity (`MCPTool.render_for_llm`): the payload (`data`) is rendered
    /// and `error` is ignored, so a failing result still shows its payload text.
    /// A plain object carrying a `result` key (the Python shape for values that
    /// are not an `McpToolResult`) is unwrapped first.
    fn render_for_llm(&self, output: &Value) -> String {
        if let Some(map) = output.as_object() {
            if map.contains_key("success") || map.contains_key("data") {
                if let Ok(result) = serde_json::from_value::<McpToolResult>(output.clone()) {
                    return success_text(render_payload_text(&Value::Object(result.data)));
                }
            }
            if let Some(inner) = map.get("result") {
                return success_text(render_payload_text(inner));
            }
        }
        success_text(render_payload_text(output))
    }
}

/// Returns `text`, or [`EMPTY_SUCCESS_TEXT`] when nothing rendered.
fn success_text(text: String) -> String {
    if text.is_empty() {
        EMPTY_SUCCESS_TEXT.to_owned()
    } else {
        text
    }
}

/// Loads every tool `client` advertises as an [`McpTool`].
///
/// Python parity: the `ToolMgr` projection — each card gets the id
/// `<server_id>.<server_name>.<tool>`, the remote name/description/schema, and
/// `metadata` entries `mcp: true`, `server_name`, `server_id`.
///
/// # Errors
///
/// Returns the [`McpClient::list_tools`] error when the server cannot be listed.
pub async fn load_mcp_tools(
    client: Arc<dyn McpClient>,
    options: McpToolOptions,
) -> Result<Vec<McpTool>> {
    let server_id = client.server_id().to_owned();
    let server_name = client.server_name().to_owned();
    let remote_tools = client.list_tools().await?;
    Ok(remote_tools
        .into_iter()
        .map(|remote| {
            let card = project_card(&server_id, &server_name, &remote);
            McpTool::new(card, client.clone(), remote.name).with_options(options.clone())
        })
        .collect())
}

/// Projects one remote tool into a [`ToolCard`].
fn project_card(server_id: &str, server_name: &str, remote: &McpRemoteTool) -> ToolCard {
    ToolCard::new(
        remote.name.clone(),
        remote.description.clone(),
        remote.input_schema.clone(),
    )
    .with_id(format!("{server_id}.{server_name}.{}", remote.name))
    .with_metadata_entry("mcp", json!(true))
    .with_metadata_entry("server_name", json!(server_name))
    .with_metadata_entry("server_id", json!(server_id))
}
