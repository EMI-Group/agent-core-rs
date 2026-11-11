# tool — cards, Tool contract, function/REST/MCP tools

## Intent

Tools as data plus behavior: a serializable, LLM-facing `ToolCard` and a `Tool` trait whose implementation may hold transport details that never serialize.
Ports `openjiuwen/core/foundation/tool/` (`base.py`, `function/function.py`, `service_api/*`, `mcp/*`, the `@tool` decorator).

## API Surface

- `card`: `ToolCard { id, name, description, input_params: Value /* JSON Schema */, exposure: ToolExposure (Direct | Deferred), execution: ToolExecution (Server | Client), parallel_safe: bool /* true */, stateless: bool, idempotent: bool, version: Option<String>, metadata: BTreeMap<String,Value> }`
  with `new(name, description, input_params)`, `tool_info() -> ToolInfo`, `openai_tool() -> Value`; implements `openjiuwen_common::card::Card`.
  `ToolInfo { kind: "function", name, description, parameters }` serializes to `{"type":"function","function":{...}}` (the OpenAI tools shape).
- `context`: `ToolContext { session_id: Option<String>, invocation_id: String, cancellation: CancellationToken, metadata: BTreeMap<String,Value>, timeout: Option<Duration> }` with `new()`, `with_session`, `with_metadata`, `with_timeout`, `is_cancelled()`, `check_cancelled()`.
- `traits`: `Tool` trait (`card`, `invoke`, `stream`, `render_for_llm`, `is_parallel_safe`), `ToolOutput { success, data, error }`, `render_payload_text`, `render_tool_output`, `EMPTY_SUCCESS_TEXT`, `EMPTY_FAILURE_TEXT`.
- `function`: `FunctionTool`, `FunctionToolBuilder` (`name`, `description`, `input_schema(Value)`, `input_schema_for::<T>()`, `handler`, `typed_handler`, `render`, `build`), `ToolHandler` alias.
- `schema`: `schema_for::<T: JsonSchema>() -> Value`, `ensure_object_schema(Value) -> Value`, `apply_defaults(&Value, &mut Value)`, `validate_inputs(&Value, &Value) -> Result<()>`, `openai_tools(cards) -> Value`.
- `restful`: `RestfulApiConfig { url, method, headers, queries, paths, timeout /* 60s */, max_response_bytes /* 10 MiB */ }`, `RestfulApiTool`, `ApiParamMapper`, `ParamLocation` (`Query | Path | Body | Header`).
- `mcp` (feature `mcp`): `McpServerConfig { server_id, server_name, server_path, client_type: McpClientType (Stdio | Sse | StreamableHttp), params: Value, auth_headers, auth_query_params }`, `McpClient` trait, `connect_mcp_server`, `load_mcp_tools`, `McpTool`, `McpToolResult { success, data, error }`.

## Constraints

- `ToolCard` is immutable after construction (`&self` only); registration-side mutation from Python does not exist here — callers clone and adjust.
- Inputs are validated against `input_params` before any handler runs; validation failures are `Error::Validation`, handler failures `Error::Tool`.
- No tool implementation may panic on user input; REST/MCP failures map to the shared `Error` enum.
- `stream` must yield `Result<Value>` items and end after `StreamChunk`-style completion; the default implementation yields the single `invoke` result.
- `mcp` code is `#[cfg(feature = "mcp")]`; the crate must build without it.

## Design Decisions

- **ToolCard is flat and implements `Card`** (deviation from `docs/ARCHITECTURE.md` §2.3's nested `CardBase`): consumers use `card.name`/`card.id` directly, and the card stays registry-safe.
- **Card/Config split**: the card is the LLM-facing projection; REST url/method/timeouts live in `RestfulApiConfig`, MCP server details in `McpServerConfig`. Python subclasses `ToolCard` instead; the Rust split keeps `ToolCard` payload-stable for every transport.
- **`ToolInfo` is the projection for `ModelRequestConfig.tools`** — `tool_info()` produces exactly the OpenAI function-tool object Python builds in `BaseModelClient._convert_tools_to_dict`.
- **Render contract** (Python `render_payload_text`/`render_tool_output`): `None` → `""`; an object with a `"content"` key → that value rendered; a string → itself; anything else → compact JSON.
  `ToolOutput` success → `render_payload_text(data)` or `EMPTY_SUCCESS_TEXT`; failure → `error`, else payload text, else `EMPTY_FAILURE_TEXT`.
- **`FunctionTool` semantics** (Python `LocalFunction`): the handler receives the validated, default-filled input object as a single `Value` plus `&ToolContext`; the return value is passed through verbatim (string, object, array — all legal).
  `typed_handler::<I, O>` deserializes `I` and serializes `O`; deserialization failure is `Error::Validation`. Python's blocking sync handlers are replaced by async-only handlers (call `spawn_blocking` inside the handler when needed).
- **`#[tool]` macro contract** (`openjiuwen-macros`, later): the macro expands to
  `FunctionTool::builder().name(<fn name>).description(<doc comment>).input_schema(schema_for::<Args>()).typed_handler::<Args, _>(<fn>).build()`
  where `Args` derives `serde::Deserialize` + `schemars::JsonSchema`. `schema_for::<T>()` is therefore the single schema entry point the macro targets.
- **Schema generation** uses `schemars` (typed handlers) or a literal JSON Schema (closures/remote tools). `schema_for` removes `$schema` and keeps `$defs` (schemars 1 emits inline-or-`$defs`; `jsonschema` validates both).
  An empty/absent `properties` schema **rejects all arguments** in Python (`additionalProperties: false`); the Rust default is permissive (`additionalProperties` untouched) except where the schema says otherwise — documented deviation, because closures with `Value` input must accept arbitrary objects.
- **REST invocation** (Python `RestfulApi`): parameters are routed by the `"location"` keyword in each property (`query`/`path`/`body`/`header`, default `body`); `GET`/`HEAD`/`OPTIONS`/`DELETE` send the body bucket as query parameters, other methods send JSON; `{name}` placeholders in the url are substituted from `path` parameters.
  The result is the Python envelope `{code, message, data, url, headers}`; the parser picks JSON for `*/json` and `*+json` content types and text otherwise.
  Non-2xx → `Error::Tool` (Python raises), request timeout → `Error::Timeout`, response over `max_response_bytes` → `Error::Tool`.
- **MCP**: only stdio and streamable-HTTP/SSE transports (Python `openapi`/`playwright` transports are out of scope). Reconnect attempts come from `OPENJIUWEN_MCP_RECONNECT_ATTEMPTS` (default 3) with capped exponential backoff; call timeouts are never retried.
  Tool results normalize multi-block text by joining with `\n\n`, images become `"[image content: <mime>, <n> base64 chars]"` (or data URLs when `include_image_content`), and `isError` maps to `McpToolResult { success: false }`.

## Routing Table

- `./card.rs` → `ToolCard`, `ToolInfo`, `ToolExposure`, `ToolExecution`
- `./context.rs` → `ToolContext`
- `./traits.rs` → `Tool` trait, `ToolOutput`, render helpers
- `./function.rs` → `FunctionTool`, `FunctionToolBuilder`, `ToolHandler`
- `./schema.rs` → schema derivation, default filling, validation, OpenAI tool arrays
- `./restful.rs` → `RestfulApiTool`, `RestfulApiConfig`, `ApiParamMapper`, response envelope
- `./mcp/` → `McpServerConfig`, `McpClient` transports, `McpTool`, `McpToolResult` (`mcp/CONTEXT.md`)
