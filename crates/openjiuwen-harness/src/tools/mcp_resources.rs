//! MCP resource tools: `list_mcp_resources` and `read_mcp_resource`.
//!
//! Ports `openjiuwen/harness/tools/mcp_tools.py`. The tools work through a narrow
//! [`McpResourceProvider`] seam so they stay independent of the MCP client transport; with no
//! provider wired they return a typed `success=false` payload instead of degrading silently.
//!
//! # Required public API (registry contract)
//!
//! - `McpResourceProvider` (async trait, `Send + Sync`): `list_resources(&self) ->
//!   Result<Vec<McpResource>>` and `read_resource(&self, uri: &str) ->
//!   Result<McpResourceContent>`; `McpResource { uri, name, mime_type, description }`.
//! - `ListMcpResourcesTool::new(provider: Option<Arc<dyn McpResourceProvider>>, language:
//!   impl Into<String>) -> Self` and the same for `ReadMcpResourceTool`, each with
//!   `card_for(language: &str) -> ToolCard`; model-facing names `list_mcp_resources` and
//!   `read_mcp_resource`.
