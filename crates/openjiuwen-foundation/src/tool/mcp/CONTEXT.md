# mcp — Model Context Protocol client (feature `mcp`)

## Intent

Expose remote capabilities as tools: a provider-neutral client trait, `rmcp`-backed stdio / streamable-HTTP / SSE transports with reconnect and a server-initiated event stream, an OpenAPI→tools transport, and `McpTool` wrapping a remote tool as a `Tool`.
Ports `openjiuwen/core/foundation/tool/mcp/` (`base.py`, `client/mcp_client.py`, `client/reconnect.py`, `client/{stdio,sse,streamable_http,openapi}_client.py`); the Python `playwright` transport is a documented deferral because it needs a browser runtime.
The streamable-HTTP transport also opens the MCP server-initiated `GET` event stream, so server notifications and server-initiated requests reach the client.
Resource subscriptions (`resources/subscribe`, `resources/unsubscribe`) are optional in MCP: they are defaulted on `McpClient` and the `rmcp` client implements them behind the capability the server negotiated in its `initialize` result.
The legacy SSE transport reopens its event stream with `Last-Event-ID`, so a server replays the events the client has not seen.
The whole module is compiled only with the `mcp` feature (declared on `tool::mcp` in `src/tool/mod.rs`); nothing here may be referenced from outside that gate.

## API Surface

`client`:
- `McpClientType { Stdio, Sse, StreamableHttp /* default */, OpenApi, Playwright }` — `Serialize`/`Deserialize` via `parse`, accepting `stdio`, `sse`, `streamable_http`, `streamable-http`, `http`, `openapi`, `open_api`, `playwright` in any ASCII case; `as_str()` (`"stdio"`, `"sse"`, `"streamable_http"`, `"openapi"`, `"playwright"`), `McpClientType::parse(&str) -> Result<Self>` (`Error::Validation` on unknown names).
- `McpServerConfig { server_id: String /* uuid_hex default */, server_name: String, server_path: String, client_type: McpClientType, params: Value, auth_headers: BTreeMap<String, String>, auth_query_params: BTreeMap<String, String>, include_image_content: bool }` with `Default`, `stdio(name, command)`, `http(name, url)`, `with_server_id`, `with_auth_header`, `with_auth_query_param`, `with_include_image_content`, and the tolerant `params` readers `command()` (`params.command`, else `server_path`), `args()`, `env()`, `cwd()`, plus `redacted_headers()`.
- `McpRemoteTool { name, description, input_schema: Value }` with `new(name, description, input_schema)`.
- `McpNotification { method: String, params: Value }` — one server-pushed notification; `#[serde(default)]`, snake_case.
- `McpClient` (`#[async_trait]`, object-safe): `server_name() -> &str`, `server_id() -> &str` (default `""`), `include_image_content() -> bool` (default `false`), `connect()`, `disconnect()`, `is_connected()`, `list_tools() -> Result<Vec<McpRemoteTool>>`, `call_tool(name, arguments: Value, timeout: Option<Duration>) -> Result<McpToolResult>` (provided), `call_tool_with(name, arguments, timeout, include_image_content) -> Result<McpToolResult>` (required), `list_resources() -> Result<Vec<Value>>` (default `Error::Unsupported`), `read_resource(uri) -> Result<Value>` (default `Error::Unsupported`), `take_notifications() -> Result<Vec<McpNotification>>` (drains; default empty), `supports_server_stream() -> bool` (default `false`).
- `connect_mcp_server(&McpServerConfig) -> Result<Arc<dyn McpClient>>` — dispatches `OpenApi` to `OpenApiClient`, rejects `Playwright` with `Error::Unsupported("playwright MCP transport requires a browser runtime")`, and otherwise validates the transport and returns a lazily connecting `RmcpClient` (`Error::Validation` for a stdio config without a command or a non-http(s)/relative endpoint).
- `RmcpClient` — `new(&McpServerConfig)`, `config()`, implements `McpClient`; one session per client, held behind a `tokio::sync::Mutex`, created on first use, dropped on disconnect or after a transport failure. Holds the session's `ServerStreamState` in a `parking_lot::Mutex<Option<ServerStreamState>>` (set on connect for `streamable_http`, cleared and `reset()` on drop), which backs `take_notifications` (drains, never connects) and `supports_server_stream` (supported and not refused). `new` rejects the `OpenApi`/`Playwright` variants with `Error::Unsupported`.
- Reconnect policy: `reconnect_attempt_budget() -> u32` (total attempts, `OPENJIUWEN_MCP_RECONNECT_ATTEMPTS` else 3, never < 1) and `reconnect_delay(attempt) -> Duration` (`min(120, 5 * 2^(attempt-1))` seconds, saturating at 120).
- `FakeMcpClient` (`#[cfg(test)]`) — `new(server_id, server_name)`, `set_tools`, `set_result`, `set_failure`, `calls() -> Vec<FakeCall>`, `connect_count()`; `FakeCall { tool_name, arguments, timeout, include_image_content }`.

`openapi`:
- `OpenApiSpec` — the parsed document plus its flattened operations: `load(source)` / `load_with(&HttpClient, source)` (async), `from_str(text)`, `from_value(Value)`, `base_url() -> &str`, `operations() -> &[OpenApiOperation]`, `document() -> &Value`, `source_url() -> Option<&str>`.
- Spec source: a filesystem path (`.json` → JSON, `.yaml`/`.yml` → YAML, any other extension tried as JSON then YAML; missing path/non-file/symlink is `Error::Validation`) or an `http(s)` URL fetched once through `openjiuwen_common::http::HttpRequest`.
- `OpenApiOperation { name, description, method /* uppercase */, path, parameters: Vec<OpenApiParameter>, body_schema: Option<Value>, body_required: bool, input_schema: Value }` and `OpenApiParameter { name, location /* "path" | "query" | "header" */, required, schema: Value, description: Option<String> }`; both `Debug + Clone + PartialEq + Serialize + Deserialize`.
- `OpenApiClient` — `new(&McpServerConfig)`, `with_http_client(&McpServerConfig, HttpClient)` (injection for tests), `config()`, `spec() -> Option<Arc<OpenApiSpec>>`, and `impl McpClient`: `connect` loads the cached spec, `disconnect` drops it (no network), `is_connected` reports a loaded spec, `list_tools` maps operations to `McpRemoteTool`, `call_tool_with` executes the matching operation (unknown name is `Error::Mcp`); resource/notification/stream methods keep their trait defaults.
- Naming: `operationId.split("__")[0]` → `summary` → `"{METHOD}_{path}"`, truncated to 64 characters and de-duplicated per spec with `_2`, `_3`, ….
- Flattened input schema: `{"type":"object","properties":{…},"required":[…]}` from path/query/header parameters (parameter description merged when the schema has none) plus the `application/json` `requestBody` properties.
- Call mapping: `{name}` path templating with RFC 3986 percent-encoding (missing path argument is `Error::Validation`), declared query/header arguments, `auth_query_params`/`auth_headers` from the config, JSON body from the requestBody properties, `Content-Type: application/json` + `Accept: application/json` when a body is sent.
- Response mapping: 2xx JSON object → `data` as-is, other JSON → `data = {"result": …}`, non-JSON → `data = {"content": …}`; non-2xx → `success: false` with `data = {status, body}` and an `error` naming the status, method and URL (body truncated at `MAX_ERROR_BODY_CHARS = 512`).

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
- `SseFrame { event, data, id, retry }` — `id`/`retry` (`retry` in milliseconds, unparsable values ignored) are captured into the next completed frame; frames are still delimited by `event`/`data` alone.
- `ServerStreamState` (Clone, Default, `Arc`-shared) with `supports_server_stream`, `mark_supported`, `is_refused`, `mark_refused(status)`, `mark_started`, `is_started`, `last_event_id`, `set_last_event_id`, `push_notification`, `take_notifications`, `reset`; the notification buffer is bounded by `NOTIFICATION_BUFFER_BOUND = 64` (oldest dropped with one `tracing::warn!`).
- `server_stream_headers(session_id, last_event_id) -> Vec<(&'static str, String)>` and `stream_reconnect_delay(attempt, hint) -> Duration` (`hint` defaults to `DEFAULT_STREAM_RETRY` = 1 s, capped by `reconnect_delay(attempt)` and 120 s) — both pure and unit-tested.

## Constraints

- Feature-gated by `tool::mcp` (`#[cfg(feature = "mcp")]`); `cargo test -p openjiuwen-foundation --no-default-features` must keep passing.
- No test spawns a process or reaches the network: `SseParser`/`SseFrame`, endpoint validation, and the stream header/delay helpers are pure, `McpTool` behavior is tested through `FakeMcpClient`, and the stream/OpenAPI tests bind a loopback `wiremock` server and `tempfile` specs only. Any future live-transport test must be `#[ignore]`d.
- No new dependencies: OpenAPI parsing uses `serde_json` + `serde_yaml_ng`, and every HTTP call goes through `openjiuwen_common::http::{HttpClient, HttpRequest}` (the `rmcp` transports are the sole exception and stay on `reqwest` streaming).
- No `unsafe`; no `unwrap`/`expect` outside tests; every public item documented (`missing_docs` + clippy `-D warnings`); files stay under ~1000 lines (`spec.rs` 740, `client.rs` 986 including its test module).
- Auth header/query values are never logged: only `redacted_headers()` (via `openjiuwen_common::security::redact::sanitize_headers`) reaches `tracing`.
- Reconnect retries only `Error::Transport`; `Error::Timeout`, `Error::Cancelled`, `Error::Validation`, `Error::Unsupported`, and `Error::Mcp` (a server-level rejection) are final.
- `call_tool` timeouts bound one attempt; the timeout applies to the whole JSON-RPC round trip (or one OpenAPI HTTP request), not to reconnect backoff.
- `Tool`/`ToolCard` immutability applies: `McpTool` never mutates its card.

## Design Decisions

- **HTTP transports are hand-written on `reqwest`** (deviation from "`rmcp`-backed transports"): the workspace enables `rmcp` with `client`/`server`/`transport-child-process`/`transport-sse-client`/`transport-streamable-http-client`, but not `transport-*-reqwest`, so `rmcp`'s HTTP clients are not compiled in and this crate cannot add features. The hand-written transports implement `rmcp::transport::Transport<RoleClient>` directly, so the initialize handshake, request ids, ping handling, and cancellation still come from `rmcp::serve_client`.
- **Streamable HTTP** POSTs each message with `Accept: application/json, text/event-stream`; a JSON body is one message, an SSE body is drained in a spawned task, and `202`/`204` mean "no content" (notifications). `Mcp-Session-Id` is captured from any response and replayed; `close` sends `DELETE` with that header and ignores the result (no session id means no `DELETE`).
- **Server-initiated stream**: when the initialize response first carries `Mcp-Session-Id`, the transport spawns one `GET` loop (`Accept: text/event-stream`, `Mcp-Session-Id`, `MCP-Protocol-Version` = `rmcp::model::ProtocolVersion::LATEST`, `Cache-Control: no-store`, plus `Last-Event-ID` once an event id is known) guarded by `Arc<AtomicBool>` start-once and recorded in a cloned `ServerStreamState`. `405`/`404`/`501` mark the channel refused (never retried); other failures and a normal body end are retried up to `reconnect_attempt_budget()` with `stream_reconnect_delay(attempt, hint)`, where the server `retry` hint wins but never exceeds `reconnect_delay(attempt)` nor 120 s (`DEFAULT_STREAM_RETRY` = 1 s). Frames are forwarded to `rmcp` unchanged (responses correlate by id, server requests are handled by `rmcp`); parsed notifications are additionally buffered as `McpNotification { method, params }` in a 64-entry `VecDeque` (oldest dropped with one `tracing::warn!`). `method`/`params` are read from the raw frame JSON because `rmcp` models several notifications without parameters.
- **SSE (legacy HTTP+SSE)** opens one `GET` stream, learns the POST URL from the `endpoint` event (resolved against the server URL with `url::Url::join`), POSTs messages there, and receives replies on the stream. The stream restarts on the next send after it ends; waiting for `endpoint` is bounded at 30 s.
- **Auth** lives on the transport: `auth_headers` are applied to every HTTP request and `auth_query_params` are appended to the endpoint URL (and to every OpenAPI call), so both auth styles work without `rmcp` hooks.
- **Stdio** uses `rmcp::transport::TokioChildProcess` with `params.command` (else `server_path`), `params.args`, `params.env`, `params.cwd`, and `kill_on_drop(true)`.
- **Reconnect** is a per-operation loop in `RmcpClient::with_reconnect`: on a retryable failure it drops the session, sleeps `reconnect_delay(attempt)` (`5, 10, 20, 40, 80, 120, 120 …` seconds, deterministic — the Python ±20 % jitter is dropped so the policy is testable), and retries, up to `reconnect_attempt_budget()` total attempts.
- **OpenAPI is a client, not an MCP session**: there is no JSON-RPC and no session; `connect` only loads and caches the document, `list_tools` projects the operations, and `call_tool` performs one plain HTTP request. The Python implementation delegates to `fastmcp` (`OpenAPITool`, `RequestDirector`); this port reimplements the required subset — route collection, name generation, schema flattening, and direct request dispatch.
- **OpenAPI paths**: string arguments are percent-encoded with the RFC 3986 unreserved set (hand-rolled, because `url`'s encoders are query-oriented); arrays become comma-joined query values and objects compact JSON; JSON `null` query/header arguments are omitted while `null` body properties are forwarded.
- **OpenAPI `$ref`** resolution is shallow and local: a top-level `$ref` on a schema/parameter/request body is inlined with sibling keys winning, JSON pointers (`#/…`, `~0`/`~1`) are supported with a depth guard of 32, and remote/file references are rejected with `Error::Validation`. Nested refs inside `properties` remain pointers in the emitted schema.
- **Playwright is deferred, not stubbed**: `McpClientType` parses it for configuration compatibility, but `connect_mcp_server` returns `Error::Unsupported("playwright MCP transport requires a browser runtime")` because the workspace ships no browser runtime.
- **Uniform result shape** (deviation from Python's bare `str`/`dict` + `{"result": ...}` wrapper): `McpToolResult.data` is always a map. One text block lands under `content`; multiple text blocks are joined with `\n\n`; audio blocks become `[audio content: <mime>, <n> base64 chars]` (Python returns the raw base64 for a single audio block); a single embedded resource with no text lands under `content` as its serialized JSON, otherwise resources land in a `resources` array (Python flattens `model_dump` into the text join); images become placeholders or, with `include_image_content`, `multimodal` entries (`type`, `source: "mcp"`, `source_path: "mcp_tool"`, `mime_type`, `data_url`) plus an `<n> image(s) attached as multimodal input.` note. `CallToolResult.structured_content` is ignored.
- **`isError` mapping** is Python's: `success: false`, `error` rebuilt from the raw text blocks (`DEFAULT_MCP_ERROR_TEXT` when there are none), payload kept as extracted.
- **Per-tool image flag** (deviation from Python, which reads `include_image_content` only from the server config): `McpToolOptions.include_image_content` reaches the client through `call_tool_with`, while `McpServerConfig.include_image_content` remains the client default used by `call_tool`.
- **`McpTool::invoke`** normalizes the schema with `ensure_object_schema`, sends non-object inputs as `{}` (Python's `isinstance(inputs, dict)` fallback), fills defaults, validates (`Error::Validation` before any call), checks cancellation, calls with `ctx.timeout.or(options.timeout)`, and serializes the result as JSON. A `success: false` result becomes `Error::Mcp` with the result's `error` message (deviation: Python returns the `McpToolResult` object unchanged); `Error::Cancelled`/`Error::Timeout` from the client pass through unchanged.
- **`render_for_llm`** keeps Python's MCP behavior: it renders `data` (ignoring `error`), unwraps a plain `{"result": ...}` payload, and falls back to `render_payload_text`, with `EMPTY_SUCCESS_TEXT` for an empty render.
- **Tool naming** follows the objective: the card keeps the raw remote name and gets the registry id `<server_id>.<server_name>.<tool>`. The Python model-facing rename (`mcp_<server>_<tool>`) belongs to the ability manager, not to this layer.

## Test Strategy

- `tests.rs` (44 tests, `#[cfg(test)] mod tests;`) covers: the result-normalization matrix (text join, image placeholder vs. multimodal, audio, one vs. many resources, empty content, both `isError` paths); config serde and `params` extraction (`command`/`args`/`env`/`cwd`, transport aliases); `connect_mcp_server` validation; reconnect budget/delays; `SseParser` (chunk splitting, multi-frame chunks, multi-line data, CRLF, comments, `endpoint` events); `McpTool` through `FakeMcpClient` (success envelope, schema rejection and cancellation with zero calls, failure mapping, timeout precedence, `render_for_llm` cases); `load_mcp_tools` card projection; `FakeMcpClient` connection counting.
- `stream_tests.rs` (13 tests, loopback `wiremock` only, `BOUND` = 5 s) covers the server-initiated `GET` stream end to end: a mocked handshake whose `InitializeResponder` echoes the request id and returns `Mcp-Session-Id: sess-1`, then either a `200 text/event-stream` body carrying one `notifications/tools/list_changed` frame (notification buffered with its `params`, `supports_server_stream()` true) or a `405` (capability reported false, no notification, exactly one `GET` request — never retried); plus the pure `server_stream_headers` (names, `Last-Event-ID` absent/present), `stream_reconnect_delay` (1 s default, 250 ms hint, 120 s cap, policy cap), `SseParser` `id`/`retry` capture (including an unparsable `retry`) with unchanged `event`/`data`, and the notification buffer bound.
- `openapi_tests.rs` (14 tests, inline specs + `tempfile` + loopback `wiremock`) covers naming (`operationId`/`summary`/`METHOD_path`, 64-char truncation, `_2` de-duplication), descriptions with appended parameter/body text, flattened schemas (path/query/header + merged JSON body + `$ref`), shallow `$ref` resolution/cycles/missing targets, YAML↔JSON parity and strict `.json` parsing, a `GET` call with path + query + `auth_query_params`, a `POST` call with `body_json` and `Content-Type`, a non-2xx failure (`success: false`, status in `error`, `data.status`), an unknown tool (`Error::Mcp`), a spec fetched from a URL through `connect_mcp_server` → `list_tools`, the Playwright `Unsupported` dispatch, client-type parse/serde round-trips, an empty `server_path`, and `build_url` encoding errors.
- `cargo test -p openjiuwen-foundation mcp` runs 71 tests; `cargo test -p openjiuwen-foundation --no-default-features` builds and passes without the feature.
- `FakeMcpClient` lives in `client.rs` under `#[cfg(test)]`; it is not part of the public API.

## Known Issues and Gaps

- Playwright is the only deferred transport: it parses as a client type but `connect_mcp_server` rejects it with `Error::Unsupported`.
- OpenAPI `$ref` resolution is shallow and local only (remote/file references rejected; nested `properties` refs stay pointers).
- Only `application/json` request bodies are supported; `multipart/form-data`, `form-urlencoded`, and other media types are ignored (`body_schema = None`), and `in: cookie` parameters are skipped.
- OpenAPI clients ignore `include_image_content` (responses are plain JSON) and support no resources, notifications, or server stream (trait defaults).
- Parameters sharing a name across locations collapse into one flat property (last wins), and OAuth/security-scheme requirements from the spec are not enforced — auth comes from `McpServerConfig`.
- The legacy `SseTransport` opens its `GET` stream on first send and has no `Last-Event-ID` replay, unlike the streamable-HTTP server stream.
- Notification buffering reads `method`/`params` from the raw frame because `rmcp` models several notifications without parameters; a notification `rmcp` cannot parse is only logged at debug and is not buffered.

## Routing Table

- `./client.rs` → `McpClientType`, `McpServerConfig`, `McpRemoteTool`, `McpNotification`, `McpClient`, `connect_mcp_server` dispatch, `RmcpClient`, reconnect helpers, `FakeMcpClient`
- `./http.rs` → `StreamableHttpTransport` (with the server-initiated `GET` loop), `SseTransport`, `HttpTransportError`, `validate_endpoint`, `ServerStreamState`, `server_stream_headers`, `stream_reconnect_delay`, `SseParser`/`SseFrame`
- `./openapi/` → the OpenAPI→tools transport (`OpenApiClient`, `OpenApiSpec`, operations)
- `./openapi/mod.rs` → `OpenApiClient` (lazy spec load, `list_tools`, `call_tool_with`)
- `./openapi/spec.rs` → spec loading (path/URL, JSON/YAML), `$ref` resolution, operation collection, naming, schema flattening
- `./openapi/call.rs` → URL/query/header/body builders, `execute`, response mapping
- `./result.rs` → `McpToolResult`, `extract_content`, `normalize_call_tool_result`
- `./tool.rs` → `McpTool`, `McpToolOptions`, `load_mcp_tools`
- `./tests.rs` → the module's unit-test suite
- `./stream_tests.rs` → server-initiated `GET` stream tests (loopback wiremock) plus the stream header/delay/parser helpers
- `./openapi_tests.rs` → OpenAPI naming/schema/call/dispatch tests (inline specs, `tempfile`, loopback wiremock)
