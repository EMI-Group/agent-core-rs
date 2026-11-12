//! `rmcp` adapter: exposes [`TeamMcpServer`] as a low-level MCP server.
//!
//! Python parity source: `openjiuwen/agent_teams/mcp/server.py`, built on the low-level MCP server
//! (not `FastMCP`) so member tools advertise their own card schema and every result is the rendered
//! text verbatim. The adapter is deliberately thin: tool listing and dispatch live in
//! [`super::TeamMcpServer`], and no `rmcp` type leaks out of this module.
use super::{TeamMcpServer, SERVER_NAME};
use rmcp::model::{
    CallToolRequestParam, CallToolResult, Content, Implementation, JsonObject, ListToolsResult,
    PaginatedRequestParam, ProtocolVersion, ServerCapabilities, ServerInfo, Tool, ToolsCapability,
};
use rmcp::service::RequestContext;
use rmcp::{ErrorData as McpError, RoleServer, ServerHandler};
use serde_json::Value;
use std::borrow::Cow;
use std::sync::Arc;
/// The MCP server handler over an [`TeamMcpServer`] facade.
pub struct TeamServerHandler {
    server: TeamMcpServer,
}
impl TeamServerHandler {
    /// Wraps the facade.
    #[must_use]
    pub fn new(server: TeamMcpServer) -> Self {
        Self { server }
    }
}
impl ServerHandler for TeamServerHandler {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            protocol_version: ProtocolVersion::default(),
            capabilities: ServerCapabilities {
                // Only the `tools` capability is advertised (Python parity).
                tools: Some(ToolsCapability::default()),
                ..ServerCapabilities::default()
            },
            server_info: Implementation {
                name: SERVER_NAME.to_string(),
                title: None,
                version: env!("CARGO_PKG_VERSION").to_string(),
                icons: None,
                website_url: None,
            },
            instructions: self.server.instructions().map(str::to_string),
        }
    }
    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParam>,
        _context: RequestContext<RoleServer>,
    ) -> std::result::Result<ListToolsResult, McpError> {
        let tools = self
            .server
            .list_tools()
            .await
            .into_iter()
            .map(|definition| Tool {
                name: Cow::Owned(definition.name),
                title: None,
                description: Some(Cow::Owned(definition.description)),
                input_schema: Arc::new(json_object(definition.input_schema)),
                output_schema: None,
                annotations: None,
                icons: None,
            })
            .collect();
        Ok(ListToolsResult {
            tools,
            next_cursor: None,
        })
    }
    async fn call_tool(
        &self,
        request: CallToolRequestParam,
        _context: RequestContext<RoleServer>,
    ) -> std::result::Result<CallToolResult, McpError> {
        let name = request.name;
        let arguments = request.arguments.map_or(Value::Null, Value::Object);
        let text = self.server.call_tool(&name, arguments).await;
        // Always a successful result carrying one text block: tool failures are text, never MCP
        // protocol errors (Python parity).
        Ok(CallToolResult::success(vec![Content::text(text)]))
    }
}
/// Converts a JSON value into the schema object `rmcp` expects; non-objects become an empty schema.
fn json_object(value: Value) -> JsonObject {
    match value {
        Value::Object(map) => map,
        _ => JsonObject::new(),
    }
}
