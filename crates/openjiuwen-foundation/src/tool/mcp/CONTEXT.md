# mcp — Model Context Protocol client (feature `mcp`)

## Intent

Expose MCP servers as tools: a provider-neutral client trait, `rmcp`-backed transports, and `McpTool` wrapping a remote tool as a `Tool`.
Ports `openjiuwen/core/foundation/tool/mcp/` — transports limited to stdio and streamable-HTTP/SSE.

## API Surface

- `client`: `McpClientType` (`Stdio | Sse | StreamableHttp`), `McpServerConfig` (see parent), `McpClient` trait (`server_name`, `connect`, `disconnect`, `is_connected`, `list_tools`, `call_tool`, `list_resources`, `read_resource`), `RmcpClient` (the `rmcp`-backed implementation), `connect_mcp_server(&McpServerConfig) -> Result<Arc<dyn McpClient>>`, `FakeMcpClient` (test double).
- `result`: `McpToolResult { success: bool, data: Map<String, Value>, error: Option<String> }`, `extract_content(&CallToolResult, include_image_content: bool) -> Map<String, Value>`.
- `tool`: `McpTool { card: ToolCard, client: Arc<dyn McpClient>, tool_name }` implementing `crate::tool::traits::Tool`, `load_mcp_tools(client, options) -> Result<Vec<McpTool>>`, `McpToolOptions { include_image_content }`.

## Constraints

- Feature-gated: `#[cfg(feature = "mcp")]`; the rest of the crate compiles without `rmcp`.
- No test spawns a process or opens a socket: transport tests use `FakeMcpClient`; the real client is covered by pure conversion tests and a `#[ignore]`d live test.
- Reconnect honors `OPENJIUWEN_MCP_RECONNECT_ATTEMPTS` via `openjiuwen_common::config::env::mcp_reconnect_attempts`.
- No secrets in logs: auth header values are redacted.

## Design Decisions

- **`McpClient` is the seam**: tools depend on the trait, not on `rmcp`, so transports can be swapped and tests need no process.
- **Transports**: `Stdio` spawns `server_path` (command line) with `params.args`/`params.env`; `Sse` and `StreamableHttp` connect to `server_path` as a URL with `auth_headers`/`auth_query_params`. Only `StreamableHttp` is the default remote transport (Python also supports `sse`).
- **Reconnect**: `connect`/`list_tools`/`call_tool` retry transport failures up to the configured attempts with capped exponential backoff; `call_tool` timeouts (`Error::Timeout`) are never retried; a failed call after reconnect attempts returns `Error::Mcp`.
- **Result normalization** (Python `extract_mcp_tool_result_content`): one text block → returned raw under `content`; multiple text blocks → joined with `\n\n`; images → placeholder text or data URLs when `include_image_content`; `isError` → `success: false` with the text as `error`.
- **`McpTool::invoke`** validates inputs against the remote `inputSchema`, calls the client, and returns the `McpToolResult` as JSON; errors become `Error::Mcp`.

## Routing Table

- `./client.rs` → `McpClient`, `McpClientType`, `connect_mcp_server`, `RmcpClient`, `FakeMcpClient`
- `./result.rs` → `McpToolResult`, content extraction
- `./tool.rs` → `McpTool`, `load_mcp_tools`, `McpToolOptions`
