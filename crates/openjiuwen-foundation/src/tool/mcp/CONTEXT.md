# mcp — Model Context Protocol client (feature `mcp`)

## Intent

Expose MCP servers as tools: a provider-neutral client trait, `rmcp`-backed stdio / streamable-HTTP / SSE transports with reconnect, and `McpTool` wrapping a remote tool as a `Tool`.
Ports `openjiuwen/core/foundation/tool/mcp/` (`base.py`, `client/mcp_client.py`, `client/reconnect.py`, `client/{stdio,sse,streamable_http}_client.py`); the Python `openapi`/`playwright` transports are out of scope.
The whole module is compiled only with the `mcp` feature (declared on `tool::mcp` in `src/tool/mod.rs`); nothing here may be referenced from outside that gate.

## API Surface

`client`:
- `McpClientType { Stdio, Sse, StreamableHttp /* default */ }` — `Serialize`/`Deserialize` via `parse`, accepting `stdio`, `sse`, `streamable_http`, `streamable-http`, `http` in any ASCII case; `as_str()`, `McpClientType::parse(&str) -> Result<Self>` (`Error::Validation` on unknown names).
- `McpServerConfig { server_id: String /* uuid_hex default */, server_name: String, server_path: String, client_type: McpClientType, params: Value, auth_headers: BTreeMap<String, String>, auth_query_params: BTreeMap<String, String>, include_image_content: bool }` with `Default`, `stdio(name, command)`, `http(name, url)`, `with_server_id`, `with_auth_header`, `with_auth_query_param`, `with_include_image_content`, and the tolerant `params` readers `command()` (`params.command`, else `server_path`), `args()`, `env()`, `cwd()`, plus `redacted_headers()`.
- `McpRemoteTool { name, description, input_schema: Value }` with `new(name, description, input_schema)`.
- `McpClient` (`#[async_trait]`, object-safe): `server_name() -> &str`, `server_id() -> &str` (default `""`), `include_image_content() -> bool` (default `false`), `connect()`, `disconnect()`, `is_connected()`, `list_tools() -> Result<Vec<McpRemoteTool>>`, `call_tool(name, arguments: Value, timeout: Option<Duration>) -> Result<McpToolResult>` (provided), `call_tool_with(name, arguments, timeout, include_image_content) -> Result<McpToolResult>` (required), `list_resources() -> Result<Vec<Value>>` (default `Error::Unsupported`), `read_resource(uri) -> Result<Value>` (default `Error::Unsupported`).
- `connect_mcp_server(&McpServerConfig) -> Result<Arc<dyn McpClient>>` — validates the transport and returns a lazily connecting `RmcpClient` (`Error::Validation` for a stdio config without a command or a non-http(s)/relative endpoint).
- `RmcpClient` — `new(&McpServerConfig)`, `config()`, implements `McpClient`; one session per client, held behind a `tokio::sync::Mutex`, created on first use, dropped on disconnect or after a transport failure.
- Reconnect policy: `reconnect_attempt_budget() -> u32` (total attempts, `OPENJIUWEN_MCP_RECONNECT_ATTEMPTS` else 3, never < 1) and `reconnect_delay(attempt) -> Duration` (`min(120, 5 * 2^(attempt-1))` seconds, saturating at 120).
- `FakeMcpClient` (`#[cfg(test)]`) — `new(server_id, server_name)`, `set_tools`, `set_result`, `set_failure`, `calls() -> Vec<FakeCall>`, `connect_count()`; `FakeCall { tool_name, arguments, timeout, include_image_content }`.

`result`:
- `McpToolResult { success: bool /* default true */, data: Map<String, Value>, error: Option<String> }` with `success(data)`, `failure(error)`, `is_success()`, `data_map()`, `error_msg()`; `#[serde(default)]`, snake_case.
- `extract_content(&[rmcp::model::Content], include_image_content: bool) -> McpToolResult` — pure normalization of the content block list.
- `normalize_call_tool_result(&CallToolResult, include_image_content: bool) -> McpToolResult` — extraction plus the `isError` mapping.
- `DEFAULT_MCP_ERROR_TEXT = "MCP tool reported an error."`.

`tool`:
- `McpToolOptions { include_image_content: bool, timeout: Option<Duration> }` with `Default`, `with_include_image_content`, `with_timeout`.
- `McpTool::new(card: ToolCard, client: Arc<dyn McpClient>, tool_name: impl Into<String>)`, `with_options`, `client()`, `tool_name()`, `options()`, `boxed() -> Arc<dyn Tool>`; implements `Tool` (`card`, `invoke`, `render_for_llm`; `stream` keeps the default single-item stream); manual `Debug`.
- `load_mcp_tools(client: Arc<dyn McpClient>, options: McpToolOptions) -> Result<Vec<McpTool>>` — one card per remote tool: `ToolCard::new(name, description, inputSchema)` with id `<server_id>.<server_name>.<tool>` and metadata `mcp: true`, `server_name`, `server_id`.

`http` (private module):
- `StreamableHttpTransport` / `SseTransport` implementing `rmcp::transport::Transport<RoleClient>`, `validate_endpoint(&str) -> Result<()>`, and the pure `SseParser`/`SseFrame` used by both and by the tests.

## Constraints

- Feature-gated by `tool::mcp` (`#[cfg(feature = "mcp")]`); `cargo test -p openjiuwen-foundation --no-default-features` must keep passing.
- No test spawns a process or opens a socket: transport tests cover the pure `SseParser` and endpoint validation; `McpTool` behavior is tested through `FakeMcpClient`. Any future live-transport test must be `#[ignore]`d.
- No `unsafe`; no `unwrap`/`expect` outside tests; every public item documented (`missing_docs` + clippy `-D warnings`); files stay well under 1000 lines.
- Auth header/query values are never logged: only `redacted_headers()` (via `openjiuwen_common::security::redact::sanitize_headers`) reaches `tracing`.
- Reconnect retries only `Error::Transport`; `Error::Timeout`, `Error::Cancelled`, `Error::Validation`, `Error::Unsupported`, and `Error::Mcp` (a server-level rejection) are final.
- `call_tool` timeouts bound one attempt; the timeout applies to the whole JSON-RPC round trip, not to reconnect backoff.
- `Tool` file/`ToolCard` immutability applies: `McpTool` never mutates its card.

## Design Decisions

- **HTTP transports are hand-written on `reqwest`** (deviation from "`rmcp`-backed transports"): the workspace enables `rmcp` with `client`/`server`/`transport-child-process`/`transport-sse-client`/`transport-streamable-http-client`, but not `transport-*-reqwest`, so `rmcp`'s HTTP clients are not compiled in and this crate cannot add features. The hand-written transports implement `rmcp::transport::Transport<RoleClient>` directly, so the initialize handshake, request ids, ping handling, and cancellation still come from `rmcp::serve_client`.
- **Streamable HTTP** POSTs each message with `Accept: application/json, text/event-stream`; a JSON body is one message, an SSE body is drained in a spawned task, and `202`/`204` mean "no content" (notifications). `Mcp-Session-Id` is captured from any response and replayed; `close` sends `DELETE` with that header and ignores the result. There is no separate server-initiated `GET` stream: server requests arrive only on the SSE stream of a POST (documented limitation).
- **SSE (legacy HTTP+SSE)** opens one `GET` stream, learns the POST URL from the `endpoint` event (resolved against the server URL with `url::Url::join`), POSTs messages there, and receives replies on the stream. The stream restarts on the next send after it ends; waiting for `endpoint` is bounded at 30 s.
- **Auth** lives on the transport: `auth_headers` are applied to every HTTP request and `auth_query_params` are appended to the endpoint URL, so both auth styles work without `rmcp` hooks.
- **Stdio** uses `rmcp::transport::TokioChildProcess` with `params.command` (else `server_path`), `params.args`, `params.env`, `params.cwd`, and `kill_on_drop(true)`.
- **Reconnect** is a per-operation loop in `RmcpClient::with_reconnect`: on a retryable failure it drops the session, sleeps `reconnect_delay(attempt)` (`5, 10, 20, 40, 80, 120, 120 …` seconds, deterministic — the Python ±20 % jitter is dropped so the policy is testable), and retries, up to `reconnect_attempt_budget()` total attempts.
- **Uniform result shape** (deviation from Python's bare `str`/`dict` + `{"result": ...}` wrapper): `McpToolResult.data` is always a map. One text block lands under `content`; multiple text blocks are joined with `\n\n`; audio blocks become `[audio content: <mime>, <n> base64 chars]` (Python returns the raw base64 for a single audio block); a single embedded resource with no text lands under `content` as its serialized JSON, otherwise resources land in a `resources` array (Python flattens `model_dump` into the text join); images become placeholders or, with `include_image_content`, `multimodal` entries (`type`, `source: "mcp"`, `source_path: "mcp_tool"`, `mime_type`, `data_url`) plus an `<n> image(s) attached as multimodal input.` note. `CallToolResult.structured_content` is ignored.
- **`isError` mapping** is Python's: `success: false`, `error` rebuilt from the raw text blocks (`DEFAULT_MCP_ERROR_TEXT` when there are none), payload kept as extracted.
- **Per-tool image flag** (deviation from Python, which reads `include_image_content` only from the server config): `McpToolOptions.include_image_content` reaches the client through `call_tool_with`, while `McpServerConfig.include_image_content` remains the client default used by `call_tool`.
- **`McpTool::invoke`** normalizes the schema with `ensure_object_schema`, sends non-object inputs as `{}` (Python's `isinstance(inputs, dict)` fallback), fills defaults, validates (`Error::Validation` before any call), checks cancellation, calls with `ctx.timeout.or(options.timeout)`, and serializes the result as JSON. A `success: false` result becomes `Error::Mcp` with the result's `error` message (deviation: Python returns the `McpToolResult` object unchanged); `Error::Cancelled`/`Error::Timeout` from the client pass through unchanged.
- **`render_for_llm`** keeps Python's MCP behavior: it renders `data` (ignoring `error`), unwraps a plain `{"result": ...}` payload, and falls back to `render_payload_text`, with `EMPTY_SUCCESS_TEXT` for an empty render.
- **Tool naming** follows the objective: the card keeps the raw remote name and gets the registry id `<server_id>.<server_name>.<tool>`. The Python model-facing rename (`mcp_<server>_<tool>`) belongs to the ability manager, not to this layer.

## Test Strategy

- `tests.rs` (44 tests, `#[cfg(test)] mod tests;`) covers: the result-normalization matrix (text join, image placeholder vs. multimodal, audio, one vs. many resources, empty content, both `isError` paths); config serde and `params` extraction (`command`/`args`/`env`/`cwd`, transport aliases); `connect_mcp_server` validation; reconnect budget/delays; `SseParser` (chunk splitting, multi-frame chunks, multi-line data, CRLF, comments, `endpoint` events); `McpTool` through `FakeMcpClient` (success envelope, schema rejection and cancellation with zero calls, failure mapping, timeout precedence, `render_for_llm` cases); `load_mcp_tools` card projection; `FakeMcpClient` connection counting.
- `cargo test -p openjiuwen-foundation mcp` runs 44 tests; `cargo test -p openjiuwen-foundation --no-default-features` builds and passes without the feature.
- `FakeMcpClient` lives in `client.rs` under `#[cfg(test)]`; it is not part of the public API.

## Routing Table

- `./client.rs` → `McpClientType`, `McpServerConfig`, `McpRemoteTool`, `McpClient`, `connect_mcp_server`, `RmcpClient`, reconnect helpers, `FakeMcpClient`
- `./http.rs` → `StreamableHttpTransport`, `SseTransport`, `HttpTransportError`, `validate_endpoint`, `SseParser`/`SseFrame`
- `./result.rs` → `McpToolResult`, `extract_content`, `normalize_call_tool_result`
- `./tool.rs` → `McpTool`, `McpToolOptions`, `load_mcp_tools`
- `./tests.rs` → the module's unit-test suite
