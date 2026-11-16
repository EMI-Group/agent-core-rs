# mcp/ — harness `ToolGateway` bridge over kernel MCP tools (`mcp` feature)
## Intent
The crate's `mcp` feature (`openjiuwen-foundation/mcp`) gates one surface: this adapter between kernel tools and the harness `ToolGateway` seam.
MCP client/server tooling itself stays in `openjiuwen-foundation::tool::mcp` (stdio, SSE, streamable-HTTP, OpenAPI, JSON-RPC, `McpTool`), whose `mcp` feature is a foundation default the umbrella cannot turn off.
This module adds no transport and no protocol: it projects `ToolCard`s onto `ToolDefinition`s and runs `Tool::invoke` on behalf of an external harness provider.
Python parity: `openjiuwen.extensions` (v0.1.20) ships no MCP module, so the projection and outcome tables in `mod.rs` are the contract, not a port.
## API Surface
```rust
pub const MCP_SERVED_METADATA_KEY: &str = "mcp";          // card metadata marker set by load_mcp_tools
pub const MCP_SERVER_NAME_METADATA_KEY: &str = "server_name";
pub const MCP_SERVER_ID_METADATA_KEY: &str = "server_id";
pub const HARNESS_CALL_ID_METADATA_KEY: &str = "harness_call_id"; // ToolContext metadata key
pub type PermissionFilter = Arc<dyn Fn(&str) -> bool + Send + Sync>;
pub struct McpToolEntry { pub definition: ToolDefinition, pub card_id: String, pub parallel_safe: bool,
                          pub metadata: BTreeMap<String, JsonValue> }
impl McpToolEntry {
    pub fn name(&self) -> &str;
    pub fn is_mcp_served(&self) -> bool;
    pub fn server_name(&self) -> Option<&str>;
    pub fn server_id(&self) -> Option<&str>;
}
pub struct McpToolGateway { /* sorted served tools, descriptive servers, optional filter */ }
impl McpToolGateway {
    /// Empty or duplicate/blank tool names -> HarnessError::Validation.
    pub fn new(tools: impl IntoIterator<Item = Arc<dyn Tool>>) -> HarnessResult<Self>;
    /// Validates each config, rejects duplicate names, sorts by name.
    pub fn with_servers(self, servers: impl IntoIterator<Item = McpServerConfig>) -> HarnessResult<Self>;
    pub fn with_permission(self, filter: PermissionFilter) -> Self;
    pub fn entries(&self) -> impl Iterator<Item = &McpToolEntry>;  // unfiltered, sorted by name
    pub fn servers(&self) -> &[McpServerConfig];
    pub fn is_empty(&self) -> bool;
}
#[async_trait] impl ToolGateway for McpToolGateway { definitions, invoke }
```
`ToolDefinition` carries no metadata, so the MCP provenance (`mcp`, `server_name`, `server_id`), the card id and `parallel_safe` live on `McpToolEntry`.
`definitions` is deterministic (sorted by name) and applies the permission filter; `entries` is unfiltered.
## Constraints
Build the gateway from `load_mcp_tools(client, McpToolOptions::default())` results (`McpTool::boxed`) or any other kernel tools.
Invocation outcomes: caller mistakes (unknown name, blank `call_id`/`name`, kernel `Validation`) are typed `HarnessError::Validation`; kernel `Unsupported` -> `UnsupportedCapability`; kernel `Config` -> `Protocol`; kernel `Cancelled` -> `State`; every other kernel error and a `ToolOutput` with `success: false` travel in the result channel as `ToolResult { is_error: true }`, so the harness can show the failure and continue.
Success content is `{"text": <kernel render>, "structured": <payload>}`; failure content is `{"error": "MCP tool `<name>` failed: <truncated detail>"}`.
Error text is the kernel's own message truncated to 512 characters; invocation arguments are never echoed and no credentials are added.
Each invocation gets a fresh `ToolContext` (no session, no deadline) with the harness call id in metadata under `HARNESS_CALL_ID_METADATA_KEY`.
`McpServerConfig`s attached with `with_servers` are descriptive host metadata: the harness invokes through this gateway, it does not connect to the servers itself.
Policy stays with the harness rails; `with_permission` is the modelled seam (rejected tools are hidden from `definitions` and rejected with `HarnessError::State`).
## Tests
`tests.rs` builds two real `McpTool`s over a scripted in-process `McpClient` (`load_mcp_tools` included) plus scripted `FixtureTool`s for kernel outcomes an `McpTool` cannot return as a value.
Covered: sorted definitions and schemas, MCP provenance on entries, successful round-trip with recorded arguments, unknown tool, blank invocation, server failure as an error result, invalid arguments (validation error, no server call), duplicate/blank names, server validation and sorting, permission filter, `ToolOutput` failure, empty-payload placeholder, plain-value projection, `Unsupported`/`Config` mapping, error-result truncation.
`../../tests/mcp_gateway.rs` repeats the public-path contract from outside the crate with plain kernel `Tool` impls.
