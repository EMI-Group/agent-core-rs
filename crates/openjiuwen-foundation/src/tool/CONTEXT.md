# tool — cards, Tool contract, function/REST/MCP tools

## Intent

Tools as data plus behavior: a serializable, LLM-facing `ToolCard` and a `Tool` trait whose implementation may hold transport details that never serialize.
Ports `openjiuwen/core/foundation/tool/` (`base.py`, `function/function.py`, `service_api/*`, `mcp/*`, the `@tool` decorator).
Every crate above `openjiuwen-foundation` invokes tools only through `Tool`, so its surface is a stable contract.
The `card`, `context`, `traits`, `schema`, `function`, `restful`, and `mcp` subtrees are implemented.

## API Surface

`src/lib.rs` additionally re-exports every tool item at the crate root, so `openjiuwen_foundation::{FunctionTool, FunctionToolBuilder, ToolHandler, Tool, ToolCard, ToolContext, ToolOutput, ToolExposure, ToolExecution, ToolInfo, schema_for, ensure_object_schema, apply_defaults, validate_inputs, openai_tools, render_payload_text, render_tool_output, EMPTY_SUCCESS_TEXT, EMPTY_FAILURE_TEXT, RestfulApiTool, RestfulApiConfig, HttpMethod, ApiParamMapper, ParamLocation, ParamMap}` all resolve without the `tool::` prefix (MCP items re-exported only under `#[cfg(feature = "mcp")]`).
Root re-exports from `crate::tool` (for call sites that should not care about module paths):
`Tool`, `ToolCard`, `ToolContext`, `ToolOutput`, `ToolExposure`, `ToolExecution`, `ToolInfo`, `FunctionTool`, `FunctionToolBuilder`, `ToolHandler`, `RestfulApiTool`, `RestfulApiConfig`, `HttpMethod`, `ApiParamMapper`, `ParamMap`, `ParamLocation`, `schema_for`, `ensure_object_schema`, `apply_defaults`, `validate_inputs`, `openai_tools`, `render_payload_text`, `render_tool_output`, `EMPTY_SUCCESS_TEXT`, `EMPTY_FAILURE_TEXT`.
`ToolFunctionInfo`, `DEFAULT_TIMEOUT` and `DEFAULT_MAX_RESPONSE_BYTES` are **not** crate-root re-exports: reach them as `openjiuwen_foundation::tool::ToolFunctionInfo` and `openjiuwen_foundation::tool::{DEFAULT_TIMEOUT, DEFAULT_MAX_RESPONSE_BYTES}`.

`card`:
- `ToolCard { id: String, name: String, description: String, input_params: Value /* JSON Schema */, properties: BTreeMap<String, Value> /* {}, always serialized */, exposure: ToolExposure (Direct | Deferred), execution: ToolExecution (Server | Client), parallel_safe: bool /* true */, stateless: bool, idempotent: bool, version: Option<String> /* omitted when None */, metadata: BTreeMap<String, Value> /* omitted when empty */ }`.
- Serialization rule: a default card serializes to exactly the Python key set `id` (uuid hex), `name`, `description`, `input_params: {}`, `properties: {}`, `exposure: "direct"`, `execution: "server"`, `parallel_safe: true`, `stateless: false`, `idempotent: false`; the Rust-only `version` (when `None`) and `metadata` (when `{}`) are omitted, while `properties` always serializes (Python declares it `default_factory=dict`).
- `ToolCard::new(name, description, input_params)` plus chainable `with_id`, `with_name`, `with_description`, `with_input_params(Value)`, `with_properties(BTreeMap<String, Value>)`, `with_exposure`, `with_execution`, `with_parallel_safe`, `with_stateless`, `with_idempotent`, `with_version`, `with_metadata`, `with_metadata_entry`; `tool_info() -> ToolInfo`, `openai_tool() -> Value` (nested wire shape), `flat_tool_info() -> Value` (Python program shape); implements `openjiuwen_common::card::Card` (`description()` is `None` when the field is empty).
- `ToolInfo { #[serde(rename = "type")] kind: String /* "function" */, function: ToolFunctionInfo }` and `ToolFunctionInfo { name, description, parameters }`, both with `new(...)`; serializes to exactly `{"type":"function","function":{"name":...,"description":...,"parameters":...}}` (the Python-flat `ToolInfo` nested into the OpenAI tools shape — see Design Decisions).
- `ToolInfo` wire conversions: `flat(&self) -> Value` (Python-flat `{type,name,description,parameters}`, non-object `parameters` normalized to `{}`), `from_flat(Value) -> Result<ToolInfo>`, `openai(&self) -> Value` (canonical nested shape, byte-identical to the serde form), `from_openai(Value) -> Result<ToolInfo>`, `from_wire(Value) -> Result<ToolInfo>` (auto-detects the shape); batch helpers `card::flat_tools(&[ToolInfo]) -> Value` and `card::from_wire_tools(Value) -> Result<Vec<ToolInfo>>`, re-exported as `tool::{flat_tools, from_wire_tools}`.
- `properties` and `metadata` are program-only: they never enter `tool_info()`/`openai_tool()`, and `ToolCard.input_params` reaches `parameters` verbatim (a pydantic-model-dumped schema included) after `ensure_object_schema`.

`context`:
- `ToolContext { session_id: Option<String>, invocation_id: String /* uuid_hex */, cancellation: CancellationToken, metadata: BTreeMap<String, Value>, timeout: Option<Duration> }` with `new()`/`Default`, `with_session`, `with_metadata`, `with_metadata_entry`, `with_timeout`, `is_cancelled()`, `check_cancelled() -> Result<()>` (`Error::Cancelled`); `Debug + Clone`, deliberately not `serde`.

`traits`:
- `ToolOutput { success: bool, data: Option<Value>, error: Option<String> }` with `ToolOutput::success(data)` / `ToolOutput::failure(error)`.
- `render_payload_text(&Value) -> String`, `render_tool_output(&ToolOutput) -> String`, `EMPTY_SUCCESS_TEXT = "Tool succeeded with empty output."`, `EMPTY_FAILURE_TEXT = "Tool failed without an error message."`.
- The object-safe trait (usable as `Arc<dyn Tool>`), implemented with `#[async_trait]`:
  `fn card(&self) -> &ToolCard;`
  `async fn invoke(&self, inputs: Value, ctx: &ToolContext) -> Result<Value>;`
  `fn stream(self: Arc<Self>, inputs: Value, ctx: ToolContext) -> BoxStream<'static, Result<Value>>;` (default = invoke once, yield one item)
  `fn render_for_llm(&self, output: &Value) -> String;` (default: `render_tool_output` for a `ToolOutput`, else `render_payload_text`)
  `fn is_parallel_safe(&self) -> bool;` (default: `card().parallel_safe`)

`function`:
- `ToolHandler = Arc<dyn Fn(Value, ToolContext) -> BoxFuture<'static, Result<Value>> + Send + Sync>`.
- `FunctionTool::new(card, handler)`, `FunctionTool::builder()`, `FunctionTool::boxed() -> Arc<dyn Tool>`, `impl From<FunctionTool> for Arc<dyn Tool>`, manual `Debug`.
- `FunctionToolBuilder` (`Default`): `name`, `description`, `input_schema(Value)`, `input_schema_for::<T: JsonSchema>()`, `handler<F, Fut>(F)`, `typed_handler<I, O, F, Fut>(F)`, `render(F)`, `build() -> Result<FunctionTool>`.

`schema`:
- `schema_for::<T: JsonSchema>() -> Value`, `ensure_object_schema(Value) -> Value`, `apply_defaults(&Value, &mut Value)`, `validate_inputs(&Value, &Value) -> Result<()>`, `openai_tools(&[ToolCard]) -> Value` (array of `openai_tool()` in card order).

`restful`:
- `HttpMethod` (`Get | Post /* default */ | Put | Patch | Delete | Head | Options`; serializes uppercase, deserializes case-insensitively, `as_str`, `is_bodyless`, `From` conversion to `openjiuwen_common::http::Method`), `DEFAULT_TIMEOUT` (60 s), `DEFAULT_MAX_RESPONSE_BYTES` (10 MiB).
- `RestfulApiConfig { url: String, method: HttpMethod, headers: BTreeMap<String, Value>, queries: BTreeMap<String, Value>, paths: BTreeMap<String, Value>, timeout: Duration, max_response_bytes: u64 }` with `new(url)`, `with_method`, `with_header`, `with_query`, `with_path`, `with_timeout`, `with_max_response_bytes`; `timeout` serializes as seconds (int or float).
- `ParamLocation` (`Query | Path | Body /* default */ | Header`, serde lowercase), `ParamMap = BTreeMap<ParamLocation, BTreeMap<String, Value>>`.
- `ApiParamMapper::{new(schema), with_defaults(schema, queries, headers, paths), schema(), defaults_for(location), map(&Value) -> Result<ParamMap>, render_url(&str, &ParamMap) -> Result<String>}`.
- `RestfulApiTool::{new(card: ToolCard, config: RestfulApiConfig) -> Result<Self>, config()}` plus `impl Tool` (validating `invoke`; `stream` yields a single `Error::Unsupported`).

`mcp` (feature `mcp`):
- `McpServerConfig { server_id, server_name, server_path, client_type: McpClientType (Stdio | Sse | StreamableHttp), params, auth_headers, auth_query_params, include_image_content }`, `McpRemoteTool { name, description, input_schema }`.
- `McpClient` trait (`server_name`, `server_id`, `connect`, `disconnect`, `is_connected`, `list_tools`, `call_tool`, `call_tool_with`, `list_resources`, `read_resource`), `RmcpClient`, `connect_mcp_server`, `reconnect_attempt_budget`, `reconnect_delay`.
- `McpTool::new(card, client, tool_name)` + `with_options`, `load_mcp_tools(client, &McpToolOptions)`, `McpToolResult { success, data, error }`.

## Constraints

- `ToolCard` is immutable after construction (`&self` only); registration-side mutation from Python does not exist here — callers clone and adjust.
- `ToolCard.input_params` may hold `{}`; every consumer normalizes through `schema::ensure_object_schema` before validating or projecting.
- Inputs are validated against `input_params` before any handler runs; validation failures are `Error::Validation`, handler failures are `Error::Tool` (a handler's `Err` is propagated unchanged), cancellation is `Error::Cancelled`.
- No tool implementation may panic on user input; REST/MCP failures map to the shared `Error` enum; no `unwrap`/`expect` outside tests.
- `stream` yields `Result<Value>` items and ends after the last item; the default implementation yields exactly the single `invoke` result.
- `mcp` code is `#[cfg(feature = "mcp")]`; the crate must build and test without it (`cargo test -p openjiuwen-foundation --no-default-features tool`).
- Files stay under the ~1000-line guideline; `restful` is split into `config`/`mapper`/`tool` for this reason.
- Format files explicitly with `rustfmt --edition 2021 <files>`; running `cargo fmt` across the crate rewrites other agents' empty stub files.

## Design Decisions

- **ToolCard is flat and implements `Card`** (deviation from `docs/ARCHITECTURE.md` §2.3's nested `CardBase`): consumers use `card.name`/`card.id` directly, and the card stays registry-safe.
- **`ToolCard` field parity with Python**: after the `id`/`name`/`description` trio the fields are `input_params`, `properties`, `exposure`, `execution`, `parallel_safe`, `stateless`, `idempotent`; `properties` is Python's program-only `Dict[str, Any]` bookkeeping map, always serialized (default `{}`) and never projected into `ToolInfo`.
  `version`/`metadata` remain documented Rust-only extensions, omitted from JSON while unset/empty, so an unset card serializes to exactly the Python key set instead of emitting an empty `metadata`.
  Unknown JSON keys are tolerated and dropped on deserialization; a card round-trips stably.
- **`ToolInfo` serde stays nested** (`{type, function}`), and flat accessors bridge to Python's `{type, name, description, parameters}` instead of changing the representation: `ModelRequestConfig::tools` carries the nested OpenAI tools shape (the OpenAI chat client sends it verbatim, the Responses builder flattens it itself to `{type, name, description, parameters}`, Anthropic maps it to `{name, description, input_schema}`), while `openjiuwen-dev-tools`, `openjiuwen-agent-evolving` and the examples all round-trip the nested shape.
  `ToolInfo::openai()` is the canonical nested projection (byte-identical to `serde_json::to_value`) and `ToolInfo::from_openai` parses it; `ToolInfo::flat()` renders the Python program/tool-card shape and `ToolInfo::from_flat` parses it; `ToolInfo::from_wire` auto-detects either shape (a `function` key means nested, otherwise a top-level `name` means flat).
  `flat()` always emits an object `parameters` (`null`/non-object normalized to `{}`, Python's `default_factory=dict`); `from_flat` does the same normalization but rejects a non-object `parameters` with `Error::Validation`, and `from_flat` defaults a missing `type` to `"function"` and missing name/description to `""`.
  Batch helpers mirror `schema::openai_tools`: `card::flat_tools(&[ToolInfo]) -> Value` and `card::from_wire_tools(Value) -> Result<Vec<ToolInfo>>` (array of either shape, `Error::Validation` otherwise); they are re-exported as `tool::{flat_tools, from_wire_tools}` and reached through `ToolCard::flat_tool_info()` on the card side.
- **Card/Config split**: the card is the LLM-facing projection; REST url/method/timeouts live in `RestfulApiConfig`, MCP server details in `McpServerConfig`. Python subclasses `ToolCard` instead; the Rust split keeps `ToolCard` payload-stable for every transport.
- **Render contract** (Python `render_payload_text`/`render_tool_output`): `Null` → `""`; an object with a `"content"` key → that inner value rendered (null → `""`, string → itself, else compact JSON); a string → itself; anything else → compact JSON.
  `ToolOutput` success → `render_payload_text(data)` or `EMPTY_SUCCESS_TEXT`; failure → non-empty `error`, else payload text, else `EMPTY_FAILURE_TEXT`.
  `Tool::render_for_llm` renders non-`ToolOutput` values as compact JSON instead of Python's `str(output)` (Rust has no stable `str(dict)` equivalent).
- **`stream` takes `self: Arc<Self>`** (deviation from a borrowed receiver): a `BoxStream<'static, _>` cannot borrow the tool, so the default single-item stream owns an `Arc`. Callers stream through `Arc<dyn Tool>`; `FunctionTool` inherits the default, `RestfulApiTool` overrides it with `Error::Unsupported`.
- **`ToolContext` is runtime-only** (not `serde`): `CancellationToken` has no wire form and Python's context object is never serialized.
- **`FunctionTool` semantics** (Python `LocalFunction`): the handler receives the validated, default-filled input object as a single `Value` plus an owned `ToolContext`; the return value is passed through verbatim (string, object, array — all legal).
  Handlers are async-only; Python's blocking sync handlers become `tokio::task::spawn_blocking` inside the handler.
  `typed_handler<I, O, F, Fut>` deserializes `I` (`Error::Validation` on failure) and serializes `O` (`Error::Tool` on failure); it requires `I: JsonSchema` and, when no schema was set explicitly, `build` derives it with `schema_for::<I>()` so the two builder paths never diverge.
  `build()` returns `Result<FunctionTool>` (`Error::Validation` when no handler was set).
  The builder's `render` override wins over the trait default; the private `default_render` helper mirrors the trait default body and must be kept in sync with it.
- **`#[tool]` macro contract** (`openjiuwen-macros`, implemented): the macro expands an `async fn Args -> Result<O>` into a wrapper struct plus a `{fn}_tool() -> Arc<dyn Tool>` constructor around `FunctionTool::new(ToolCard::new(name, description, input_params), ToolHandler)`, where `input_params` defaults to `schema_for::<Args>()` and `Args` needs `serde::Deserialize` + `schemars::JsonSchema`; `schema_for::<T>()` is the single schema entry point it targets.
  `FunctionTool` itself is a plain struct (not macro-generated); `openjiuwen-foundation` does not re-export the `openjiuwen_macros::tool` attribute, so a consumer crate must depend on `openjiuwen-macros` directly to use it, while those that only need `FunctionTool::builder` do not.
  The generated `Tool` impl carries `#[::async_trait::async_trait]`, so a consumer crate compiling macro-generated tools needs `async-trait` in its own dependencies; hand-written impls can use the crate-root re-export `#[openjiuwen_foundation::async_trait::async_trait]` instead (`pub use async_trait;` in `src/lib.rs`).
- **Schema generation** uses `schemars` (typed handlers) or a literal JSON Schema (closures/remote tools). `schema_for` removes `$schema`, keeps `$defs`, and normalizes non-object schemas to `{"type":"object"}`.
  `additionalProperties` is left untouched (permissive), a documented deviation from Python's reject-all-empty-schema behavior, because closures with `Value` input must accept arbitrary objects.
  `validate_inputs` reports the failing instance path (root rendered as `/`) in the `Error::Validation` message.
- **REST invocation** (Python `RestfulApi`): parameters are routed by the `"location"` keyword in each property (`query`/`path`/`body`/`header`, case-insensitive, default `body`); `GET`/`HEAD`/`OPTIONS`/`DELETE` send the body bucket as query parameters, `POST`/`PUT`/`PATCH` send it as a JSON body; `{name}` placeholders in the url are substituted from `path` parameters.
  Config `queries`/`paths`/`headers` are defaults: an input value overrides them unless it is JSON null or the empty string.
  A schema without `properties` forwards all inputs to the body; an explicit `"location": "form"` is `Error::Validation` (Python's multipart handler is out of scope).
  The url must be an absolute http/https URL, and every url placeholder must be declared as a `path` property (else `Error::Validation`); extra path properties only log a warning.
  The result is the Python envelope `{code, message, data, url, headers}`; `data` is parsed as JSON for `application/json`, `text/json`, `text/x-json`, `application/javascript`, and `*+json` content types (invalid JSON there is `Error::Tool`), text otherwise; `message` is `"success"` for 2xx and the canonical reason phrase otherwise.
  Non-2xx → `Error::Tool`, request timeout → `Error::Timeout`, cancellation → `Error::Cancelled`, other transport failures → `Error::Tool`, response over `max_response_bytes` → `Error::Tool` (enforced after buffering, not through `HttpClientConfig::max_response_bytes`, so the error variant is stable).
- **MCP**: stdio, SSE and streamable-HTTP transports, plus the OpenAPI transport (one tool per OpenAPI operation, executed over plain HTTP); the Playwright transport is a typed `Error::Unsupported` because no browser runtime ships here. Reconnect attempts come from `OPENJIUWEN_MCP_RECONNECT_ATTEMPTS` (default 3) with capped exponential backoff; call timeouts are never retried.
  Tool results normalize multi-block text by joining with `\n\n`, images become `"[image content: <mime>, <n> base64 chars]"` (or data URLs when `include_image_content`), and `isError` maps to `McpToolResult { success: false }`.

## Notes for Agents

- **MCP**: `McpTool` validates inputs through `schema::ensure_object_schema` / `apply_defaults` / `validate_inputs`, maps a `success: false` result to `Error::Mcp`, and keeps the default `render_for_llm`; HTTP transports are hand-written on `reqwest` because the workspace `rmcp` features do not include the reqwest transports.
- `ToolCard.input_params` is stored as given (possibly `{}`); only the projection/validation paths normalize it, so tests that assert raw JSON must account for that.
- `FunctionTool::default_render` duplicates the `Tool::render_for_llm` default body (Rust cannot call a trait default from an override); change both together.
- `Tool` is object-safe with an `Arc<Self>` `stream`; adding a `&self` streaming method instead would break `'static` streams.

## Known Issues

- The MCP HTTP client is built with `reqwest::Client::builder().build()` (`mcp/http.rs:144`): no `connect_timeout` and no per-request timeout, so a server that accepts the connection but never answers stalls the POST (`mcp/http.rs:377`), the JSON body read (`mcp/http.rs:414`), the `GET` stream connect and the `close` `DELETE`; because `rmcp`'s serve loop awaits `transport.close()`, `RmcpClient::disconnect` hangs with it.
- `SseParser::push` decodes every body chunk with `String::from_utf8_lossy` (`mcp/http.rs:719`), so a chunk boundary inside a multi-byte UTF-8 sequence injects U+FFFD and silently corrupts non-ASCII JSON-RPC payloads instead of erroring.
- The SSE parser's line buffer and frame `data` grow without a size cap (`mcp/http.rs:719`, `mcp/http.rs:762`) and completed frames go into an unbounded channel (`mcp/http.rs:322`, `mcp/http/sse.rs:54`, sends at `mcp/http.rs:494`/`681`), so a server that never emits a newline or outruns `rmcp`'s consumer grows memory without bound; only `ServerStreamState` notifications are bounded (64).
- `RestfulApiTool` checks `max_response_bytes` only after `HttpClient` buffered the whole body (`restful/tool.rs:220`) because its client is built without `with_max_response_bytes` (`restful/tool.rs:91`), so the documented cap does not bound memory; enabling it still surfaces as `Error::Tool` through the execute-error mapping in `invoke`.
- `collect_operations` never resolves a path-item `$ref` (`mcp/openapi/spec.rs:416`), so an OpenAPI document that factors path items into `components/pathItems` (or any `$ref` path item) yields no tools for those paths, silently.
- OpenAPI tool-name de-duplication is not collision-free (`mcp/openapi/spec.rs:595`): operations `foo`, `foo_2`, `foo` produce two `foo_2` entries, and `call_tool_with`'s `find` (`mcp/openapi/mod.rs:196`) can reach only the first; the `_N` suffix can also push a name past the documented 64-character bound (`generate_name`, `mcp/openapi/spec.rs:591`).
- `SseTransport::send` posts the configured `auth_headers` to the `endpoint` URL the server announces without a same-origin check (`mcp/http/sse.rs:116`), so a malicious or compromised MCP server can redirect those credentials to another host.
- Every `Tool::invoke` deep-clones `input_params` and compiles a fresh JSON Schema validator (`schema.rs:130`, from `function.rs:164`, `restful/tool.rs:187`, `mcp/tool.rs:125`), so each tool call pays a schema compile on the ReAct hot path.
- `wait_post_endpoint` tests its state and only then awaits `endpoint_ready.notified()` (`mcp/http/sse.rs:186`) while producers call `notify_waiters` (which stores no permit), so a notification landing between the check and the registration is lost and the POST waits out the 30 s deadline.
- The `send` that reopens a stopped SSE stream returns the previous `SseStop` before the fresh loop clears it (`mcp/http/sse.rs:101` versus the clear at `mcp/http/sse.rs:255`), so one POST fails per stream restart although a new stream was just spawned.

## Test Strategy

- Unit tests live inline as `#[cfg(test)] mod tests` in each file, except `card`'s wire-conversion suite (`card/flat_tests.rs`, declared as `mod flat_tests;` in `card.rs` to keep the module file under the size baseline); there is no `tests/` subdirectory under this module.
- `card` (13), `context` (3), `schema` (8), `traits` (7) cover serde shape/defaults, `Card` identity, cancellation, schema derivation/defaults/validation paths, render contracts, and the default single-item stream.
- `card/flat_tests` (13) covers the flat↔nested conversions: the nested→flat→nested identity, `from_flat` on the exact Python literal and its field defaults, `from_wire` on both shapes plus rejection of non-objects and shape-less objects, `parameters` normalization (`null`/non-object → `{}`, non-object on `from_flat` → `Error::Validation` naming the field), `from_openai`'s missing-`function` error and `type` normalization, `flat_tools`/`from_wire_tools` batching (mixed shapes, unknown-field tolerance, non-array and per-element errors), and `ToolCard::flat_tool_info()`/`openai_tool()` agreement with `tool_info()`.
- `card`'s serialization tests pin the exact wire shape: `default_card_serializes_exactly_the_python_key_set` (exact key set, 32-char hex id, `properties: {}`, no `version`/`metadata`), `card_round_trips_python_shaped_json` (all fields in, unknown key dropped, re-serialization stable), `tool_info_projection_is_the_exact_function_shape` and `tool_info_keeps_a_pydantic_model_shaped_input_params` (`parameters` mirrors `input_params` verbatim, `properties` never leaks).
- `ToolOutput` serde is covered by `traits`: all-fields serialization, round-trip, and tolerant deserialization of Python's extra `extracted_content`/`long_term_memory` keys.
- `function` (9) covers closure/typed handlers, verbatim and array/object returns, validation and cancellation short-circuits (handler not called, counted with `AtomicUsize`), error propagation, render override, builder errors, and the default stream.
- `restful` (19) covers `wiremock`-backed routing (path/query/header/body for POST and GET), JSON and text envelopes, `+json` parsing, non-2xx, size cap, placeholder validation, cancellation, defaults merging, method/location serde, and URL rendering; tests bind `127.0.0.1` only.
- `cargo test -p openjiuwen-foundation tool` runs the tool suites (130 lib tests today, 71 of them behind the `mcp` feature) and must pass both with default features and with `--no-default-features`.
- Behavior-port tests name the Python test they mirror in a comment where applicable.

## Routing Table

- `./card.rs` → `ToolCard`, `ToolInfo`, `ToolFunctionInfo`, `ToolExposure`, `ToolExecution`, flat/OpenAI wire conversions
- `./card/flat_tests.rs` → flat↔nested `ToolInfo` conversion tests (`mod flat_tests;` under `card.rs`)
- `./context.rs` → `ToolContext`
- `./traits.rs` → `Tool` trait, `ToolOutput`, render helpers, empty-text constants
- `./function.rs` → `FunctionTool`, `FunctionToolBuilder`, `ToolHandler`
- `./schema.rs` → schema derivation, default filling, validation, OpenAI tool arrays
- `./restful/` → `RestfulApiTool`, `RestfulApiConfig`, `ApiParamMapper`, response envelope
- `./restful/config.rs` → `HttpMethod`, `RestfulApiConfig`, duration/byte defaults
- `./restful/mapper.rs` → `ParamLocation`, `ParamMap`, `ApiParamMapper`, url templating
- `./restful/tool.rs` → `RestfulApiTool` construction, request building, envelope, wiremock tests
- `./mcp/` → `McpServerConfig`, `McpClient` transports, `McpTool`, `McpToolResult` (`mcp/CONTEXT.md`)
