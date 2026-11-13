# openjiuwen-foundation

## Intent

Foundation layer of the kernel: messages and prompts, LLM clients and the model factory, tools (local function, REST, MCP, OpenAPI), and storage backends (in-process, local files, SQLite, and network adapters for Redis/Chroma/Milvus/Elasticsearch/S3).
Ports the Python package `openjiuwen/core/foundation/` (`prompt/`, `llm/`, `tool/`, `store/`) from openJiuwen v0.1.20.
This crate is the lowest provider seam of the SDK: every crate above it (agents, workflow, context, memory, retrieval, harness) invokes models and tools only through the traits defined here, so its public surface is a stable contract.

## API Surface

Messages and prompts (`prompt`):
- `Role` (`System`/`User`/`Assistant`/`Tool` + tolerant `Other(String)`), `MessageContent` (`Text` | `Parts`), `ContentPart` (`Text`, `ImageUrl`, `Audio`, `File`, `Other(Value)`), `ImageUrl`, `AudioPart`, `FilePart`.
- `BaseMessage { role, content, name, tool_calls, tool_call_id, reasoning_content, metadata }` with `system`/`user`/`assistant`/`tool` constructors.
- `ToolCall { id, kind, name, arguments, index }`, `ToolCallDelta`, `ToolCallAccumulator`, `Usage`.
- `PromptTemplate { name, content: TemplateContent, placeholder_prefix, placeholder_suffix }` with `format`, `to_messages`, `placeholders`, `input_keys`, `from_yaml_str`, `load_yaml`.
- `TokenCounter`, `ApproxTokenCounter`, `TiktokenCounter` (feature `tiktoken`), `count_message_tokens`, `truncate_messages`.

Models (`llm`):
- `Model` trait: `provider()`, `model_name()`, `invoke(&[BaseMessage], &ModelRequestConfig) -> Result<ChatResponse>`, `stream(Vec<BaseMessage>, ModelRequestConfig) -> BoxStream<'static, Result<StreamChunk>>`.
- `ModelClientConfig` (including `api_mode: Option<ApiMode>`, `auth_mode: AuthMode`, `endpoint_profile`, and the Python `client_provider`/`custom_headers` decode aliases), `ModelRequestConfig` (including `reasoning`, `context_window`, string-or-array `stop`) with builders and `from_env()`; `ChatResponse`, `Choice`, `FinishReason`, `StreamChunk`, `AssistantTurn`, `StreamAccumulator`, `ModelExt`.
- A Python `model_dump(mode="json")` decodes directly: durations accept seconds numbers (or the legacy `{secs,nanos}` form) and serialize as seconds, `custom_headers` values accept strings/numbers/booleans, and a top-level `reasoning` intent is consumed by all three protocol builders.
- `init_model(ModelClientConfig, ModelRequestConfig)`, `init_model_with(provider, model_name)`, `ModelProvider`, `ModelProviderRegistry` (`global()` plus local instances), `register_model_provider`, `provider_profile`, `normalize_provider_name`.
- Transports: `OpenAiCompatibleClient` (`llm::openai`), `AnthropicClient` (`llm::anthropic`), and `OpenAiResponsesClient` (`llm::responses`, `POST {base}/responses` with `input_text`/`input_image`/`function_call` items and `response.*` SSE events); the Anthropic projection writes `system` as a block array, the three automatic `cache_control` anchors, inline-or-hoisted mid-conversation system turns, and a reasoning plan resolved from `extra["reasoning"]`.
- `llm::reasoning`: `ReasoningConfig`, `ReasoningCapability`, `ReasoningProfile`, `ReasoningProtocol`, `ReasoningPlan`, `find_reasoning_profile`, `provider_key_for`/`provider_key_for_endpoint`, `reasoning_capability(_for_endpoint)`, `resolve_reasoning_plan(_for)`/`resolve_reasoning_plan_with_endpoint`, `apply_reasoning_plan(_with)`, the `REASONING_PROFILES`/`MODEL_REASONING_FALLBACKS` tables, and the wire encoders (including the `endpoint_profile: "vllm"` override) applied to the OpenAI-compatible and Anthropic bodies when a reasoning intent is present.
- `llm::routing`: `ModelSelection`, `ResolvedModel`/`ResolvedRoute`/`ResolvedModelGroup`, `ResolvedRoutingConfig`, `RoutingStrategy`, `compile_model_selection`, `compile_route`, `resolve_model_group_routes`, `RoutingModel` (ordered-failover/tag-filtered, optional additive `priority`).
- `llm::oauth`: `OpenAiOAuthClient` (device code, poll, exchange, refresh with a 120 s skew and the `expires_at` → `expires_in` → previous expiry → JWT `exp` chain; no resolvable expiry means never refresh), `OAuthToken`, `DeviceCode`, `TokenStore` with `InMemoryTokenStore`/`FileTokenStore` and the `$OPENJIUWEN_AUTH_FILE` → `$OPENJIUWEN_HOME/auth.json` → `~/.openjiuwen/auth.json` resolution.
- `llm::dashscope`: `DashScopeMediaClient` + `DashScopeConfig` for image, speech, and video generation (async video task submission/polling); `ImageRequest`/`SpeechRequest`/`VideoRequest` and their response types.
- Flat crate-root re-exports (`src/lib.rs`) for the long-standing call sites: `Model`, `ModelExt`, `ChatResponse`, `Choice`, `StreamChunk`, `FinishReason`, `AssistantTurn`, `StreamAccumulator`, `ModelClientConfig`, `ModelRequestConfig`, `BaseMessage`, `MessageContent`, `Role`, `Usage`, `ToolCall`, `ToolCallDelta`, `ToolCallAccumulator`; the module paths (`llm::*`, `prompt::*`) are the canonical form and the only place the new llm modules are surfaced.

Tools (`tool`):
- `ToolCard` (flat, implements `openjiuwen_common::card::Card`; includes Python's `properties` map and serializes the exact Python key set by default), `ToolInfo`, `ToolExposure`, `ToolExecution`.
- `Tool` trait: `card()`, `invoke(Value, &ToolContext)`, `stream(Value, ToolContext)` (default = single invoke), `render_for_llm(&Value)`.
- `ToolContext { session_id, invocation_id, cancellation, metadata, timeout }`, `ToolOutput`, `render_payload_text`, `render_tool_output`.
- `FunctionTool` + `FunctionToolBuilder` (`handler`, `typed_handler`, `input_schema`, `input_schema_for::<T>`), `schema_for::<T>`, `validate_inputs`, `apply_defaults`.
- Crate-root re-export `pub use async_trait;` so downstream hand-written `Tool`/`Model` impls can write `#[openjiuwen_foundation::async_trait::async_trait]` without a direct `async-trait` dependency; macro-generated tools still reference `::async_trait::` and therefore need `async-trait` in the consumer crate.
- `RestfulApiTool` + `RestfulApiConfig` + `ApiParamMapper`/`ParamLocation`.
- `mcp` (feature `mcp`): `McpServerConfig`, `McpClientType` (`Stdio`/`Sse`/`StreamableHttp`/`OpenApi`/`Playwright`), `McpClient` trait (incl. defaulted `take_notifications`/`supports_server_stream`), `connect_mcp_server`, `load_mcp_tools`, `McpTool`, `McpToolResult`, `McpNotification`, `RmcpClient`, and `mcp::openapi::{OpenApiClient, OpenApiSpec, OpenApiOperation, OpenApiParameter}`.

Storage (`store`):
- Traits `KvStore`, `DocumentStore`, `VectorStore`, `GraphStore`, `ObjectStore` and their data types (`KvValue`, `Document`, `Query`, `CollectionSchema`, `FieldSchema`, `VectorQuery`, `VectorSearchResult`, `GraphNode`, `GraphEdge`, `ObjectMeta`).
- In-process and local backends: `InMemory*`, `LocalFile*`/`Local*`, and `SqliteKvStore`/`SqliteDocumentStore` behind feature `sqlite`.
- Network adapters (`store::external`, always compiled): `ChromaVectorStore`, `MilvusVectorStore`, `ElasticsearchVectorStore`, `ElasticsearchDocumentStore`, `RedisKvStore`, `S3ObjectStore`, each with its `*Config`.
- Name-driven registry (`store::external::registry`): `BackendKind`, `StoreSpec`, `StoreRegistry`/`global_registry`, `create_{kv,document,vector,object,graph}_store`, `register_*_backend`, `available_backend_names`.
- There is no `KnowledgeBase`, embedding, chunking, or reranker type in this crate; those live in `openjiuwen-retrieval`, which consumes the `store::VectorStore` contract here.

## Constraints

- Error type is `openjiuwen_common::error::{Error, Result}`; no panics on user input, no ad-hoc error enums across the public surface.
- Async-first on tokio; public streams are `Send + 'static`; cancellation via `tokio_util::sync::CancellationToken`; no blocking I/O on async tasks.
- Every public data type is `serde`-serializable, snake_case, tolerant of unknown fields; `#[serde(default)]` on structs; secrets use `openjiuwen_common::security::SecretString` and are never logged or `Debug`-printed.
- Model clients, storage adapters, the OpenAPI client, DashScope, and OAuth perform all HTTP through `openjiuwen_common::http::HttpClient` (retry/backoff, TLS, auth headers).
- Two exception sites use `reqwest` directly: `src/tool/mcp/http.rs` and the MCP `openapi`/DashScope-adjacent transports build their own `reqwest::Client`/`rmcp` transport; the Redis adapter uses a raw `tokio::net::TcpStream` because RESP is not HTTP. No other vendor transport is hand-rolled.
- `tracing` only; no `println!`/`eprintln!`; logging targets are `openjiuwen_foundation::<module>`.
- No `unsafe`; workspace lints are `deny`, and `cargo clippy --all-targets -- -D warnings` must pass, so every public item needs a doc comment.
- Files stay around ≤1000 lines; shared helpers live at the lowest common module (`prompt` types for the clients, `store::filter` for all backends). `store::external::{chroma,milvus,elasticsearch}` exceed the baseline on purpose: each is one complete wire mapping plus client-side validation and its tests are split into sibling `tests.rs` trees.
- Default features (`mcp`, `tiktoken`) compile and test offline; no test may hit the network or need credentials. The network storage adapters need no feature gate because their dependencies are non-optional.

## Design Decisions

- **Messages live in `prompt`**, not `llm`: the Python `llm/schema` message types are the crate's most widely used data types, and `llm` depends on `prompt`, never the reverse. `llm` re-exports `BaseMessage`, `ToolCall`, `Usage` for Python-shaped call sites.
- **One `BaseMessage` struct** instead of Python's four role subclasses: role-specific construction lives in constructors; serde rejects nothing and providers never need downcasting. This is a deliberate deviation from the Python class hierarchy.
- **Flat `ToolCard`** instead of nesting `CardBase` (deviation from `docs/ARCHITECTURE.md` §2.3): `id`, `name`, `description`, `input_params`, policy flags and `metadata` are direct fields and `ToolCard` implements the `Card` trait. Transport details (REST url/method, MCP server) live in the tool struct, matching the Python Card/Config split.
- **`StreamChunk::Delta` additionally carries `finish_reason`** (deviation from §2.2): consumers need the terminal reason without a second round-trip, and providers emit it mid-stream. `StreamChunk::Done` still marks the end of the stream.
- **Provider registry**: built-in OpenAI-compatible providers are data (`ProviderProfile`), not code; `openai`, `pangu`, `qwen`, `deepseek` and the other OpenAI-compatible aliases all resolve to one client. `anthropic` uses the Messages transport; `openai_account` uses the Responses transport, and `openai` switches to it when `api_mode: responses`. Extension crates register `ModelProvider` impls on the global registry; tests build local registries.
- **Retries** are the shared `HttpClient`'s job (`max_retries`, `RetryPolicy`, backoff). Provider clients retry only up to the connect boundary; a mid-stream failure surfaces as an error.
- **Storage**: in-memory implementations remain the behavioral reference and the local-filesystem variants reuse the same in-memory engine with JSON persistence, so behavior cannot drift.
  The network adapters are thin REST/RESP mappings over `HttpClient` (S3 signs in-crate with SigV4 over `sha2`; Redis speaks RESP2/3 over `tokio::net::TcpStream` with AUTH/SELECT and a documented single reconnect+retry) and each normalizes vendor scores into the `Metric` `[0,1]` higher-is-better convention.
  Backend selection is data-driven (`StoreSpec`/`StoreRegistry`); unknown names are `Error::Validation` listing the known ones, genuinely impossible transports (`pgvector`, `gauss`, external graph stores) are typed `Error::Unsupported`. The global plugin registry is the one documented mutable global in this subtree; tests use local registries.
  Redis values are stored verbatim as UTF-8 for `KvValue::Text` and with a `\0ojb:`+base64 binary marker for `KvValue::Bytes`, so both variants round-trip losslessly; the optional `key_prefix` is applied by the store, not the caller.
- **Anthropic prompt caching** is always on with no beta header: the last tool, the last `system` block, and the last cacheable block of the last stable message carry `cache_control: {"type":"ephemeral"}`; `extra.disable_cache_control` and `extra.cache_control_ttl` are the only opt-outs, and control keys are never forwarded to the provider.
  Mid-conversation `system` turns are hoisted as individual blocks by default and stay inline only for the model families that accept them (`claude-opus-5`, `claude-opus-4-8`, `claude-fable-5`, `claude-mythos-5`) when the neighbouring roles allow it.
- **Reasoning profiles** are data (`ReasoningProfile` table + `"*"` fallbacks) resolved by provider/model glob; the plan is applied to the OpenAI-compatible body and to the Anthropic body (protocol `AnthropicMessages`), always subordinate to explicit legacy controls and caller `extra`.
- **Routing** ports IntelliRouter's selection/compilation shape: enabled routes in list order, merge precedence `route.request_defaults` < `group.request_config` < `route.request_overrides`, `num_retries = max(enabled-1, 0)`; a route `priority` field is an additive rule layer (absent = list order) and `RoutingModel` performs the failover.
- **MCP** HTTP transports are implemented on `reqwest` and `rmcp` supplies JSON-RPC framing; the server-initiated `GET` stream is opened once the session id is known (bounded reconnect, buffered notifications), and the OpenAPI transport turns each operation into a tool executed over plain HTTP. Playwright is a typed `Unsupported`.
- **Storage replaces Python's SQLAlchemy-engine `BaseDbStore`** with a `DocumentStore` contract; `ElasticsearchDocumentStore` is the only network document backend and `pgvector`/Gauss/SQL driver backends are not portable offline.

## Routing Table

- `./src/prompt/` → messages, content parts, tool calls, `PromptTemplate`, token helpers (`src/prompt/CONTEXT.md`)
- `./src/llm/` → `Model` trait, configs, providers, factory, all three transports, reasoning/routing/oauth/dashscope (`src/llm/CONTEXT.md`)
- `./src/llm/openai/` → OpenAI-compatible client and its pure request/response/stream conversion + wiremock tests
- `./src/llm/responses/` → OpenAI Responses client, `input` item projection, `response.*` event stream, non-stream parser (`src/llm/responses/CONTEXT.md`)
- `./src/llm/anthropic/` → Anthropic Messages client, projection, caching anchors, reasoning plan, event stream (`src/llm/anthropic/CONTEXT.md`)
- `./src/llm/reasoning/` → reasoning profiles, capabilities, plan resolution and wire encoders (`src/llm/reasoning/CONTEXT.md`)
- `./src/llm/routing/` → IntelliRouter selection/route compiler and the failover `RoutingModel` (`src/llm/routing/CONTEXT.md`)
- `./src/llm/oauth/` → OpenAI account device authorization, token store and refresh (`src/llm/oauth/CONTEXT.md`)
- `./src/llm/dashscope/` → DashScope image/speech/video client and pure body builders (`src/llm/dashscope/CONTEXT.md`)
- `./src/tool/` → `ToolCard`, `Tool` trait, `FunctionTool`, REST tool, MCP (`src/tool/CONTEXT.md`)
- `./src/tool/mcp/` → MCP server config, `McpClient` + rmcp transports, server `GET` stream, `McpTool`, OpenAPI transport (`src/tool/mcp/CONTEXT.md`)
- `./src/tool/mcp/openapi/` → OpenAPI spec parsing, operation→tool mapping, HTTP call execution
- `./src/tool/restful/` → REST tool config, parameter mapper, request/envelope building
- `./src/store/` → kv/document/vector/graph/object traits and in-process/file/sqlite backends (`src/store/CONTEXT.md`)
- `./src/store/external/` → Redis/Chroma/Milvus/Elasticsearch/S3 adapters and the name-driven registry (`src/store/external/CONTEXT.md`)
- `./src/store/external/elasticsearch/` → ES vector + document backends sharing one REST client and filter translator
- `./src/store/{kv,document,vector,graph,object}/` → one directory per storage contract (memory / local-file / sqlite backends)
- `./src/store/tests/` → cross-backend contract suites compiled under `#[cfg(test)]`
- `./tests/` → crate-level integration tests (`messages.rs`, `prompt_template.rs`); every other test lives in module-level `#[cfg(test)]` trees next to the code it exercises

## Test Strategy

- Tests live beside the code they exercise, mostly as module-level `#[cfg(test)]` trees; only `tests/messages.rs` and `tests/prompt_template.rs` are standalone integration crates.
- `wiremock` mocks every HTTP call (all three clients including Responses, the REST tool, Chroma, Milvus, Elasticsearch, S3, the OpenAPI client, OAuth, DashScope, and the failover routing tests); the Redis adapter is driven by a scripted `tokio::net::TcpListener` fake server; MCP uses `FakeMcpClient` and a mocked `GET` SSE channel. No test needs the network, an API key, or a spawned MCP server.
- Pure conversion suites assert exact JSON for the Responses `input` items, `response.*` events, and the vLLM reasoning encoding, so wire-shape regressions fail without a server.
- SigV4 signing is proven by known-answer vectors (AWS `get-vanilla`, the S3 `GET /test.txt` example) plus RFC 4231 HMAC vectors and a fixed-clock self-consistency case.
- Storage behavior is covered by one contract suite per trait (`assert_kv_contract`, `assert_document_contract`, `assert_vector_contract`, `assert_graph_contract`, `assert_object_contract`) for every backend that can run in-process; network adapters assert exact wire requests and parsed responses instead.
- Behavior-port tests name the Python test they mirror in a comment (e.g. `// ports tests/unit_tests/core/foundation/prompt/test_template_assemble.py`).
- Integration test crates start with `#![allow(clippy::unwrap_used, clippy::expect_used)]`; the lib-level test cfg allows them too.
- Gates: `cargo test -p openjiuwen-foundation`, `--no-default-features`, and `--features sqlite` all pass offline; `cargo clippy -p openjiuwen-foundation --all-targets -- -D warnings` is clean in every feature combination.

## Python Mapping

| Python | Rust |
|---|---|
| `foundation/prompt/template.py` `PromptTemplate` | `prompt::PromptTemplate` |
| `foundation/prompt/assemble/*` (`PromptAssembler`, `TextableVariable`, `DictableVariable`) | `prompt::template` + private `prompt::assembler` |
| `foundation/llm/schema/message.py` (`BaseMessage` + subclasses) | `prompt::BaseMessage` |
| `foundation/llm/schema/tool_call.py` `ToolCall`, `message_chunk.py` `AssistantMessageChunk.__add__` | `prompt::ToolCall`, `prompt::ToolCallDelta`, `prompt::ToolCallAccumulator` |
| `foundation/llm/schema/config.py` `ModelClientConfig`/`ModelRequestConfig` | `llm::ModelClientConfig`/`llm::ModelRequestConfig` |
| `foundation/llm/model.py` `Model`, `init_model` | `llm::Model`, `llm::init_model`/`init_model_with` |
| `foundation/llm/model_clients/{openai,anthropic}_model_client.py` | `llm::openai`, `llm::anthropic` |
| `foundation/llm/utils/responses_utils.py` + `responses_transport.py` + `openai_account_model_client.py` | `llm::responses` |
| `foundation/llm/reasoning_profiles.py` + `reasoning.py` | `llm::reasoning` |
| `foundation/llm/routing/*` + `model_clients/intelli_router_model_client.py` | `llm::routing` (`compile_model_selection`, `RoutingModel`) |
| `extensions/external_provider/openai_auth/*` and the account OAuth transport | `llm::oauth` (+ token store seam) |
| `openai_model_client.py` `generate_image/speech/video` (DashScope) | `llm::dashscope` (HTTP instead of the vendor SDK) |
| `foundation/tool/base.py` `Tool`/`ToolCard` | `tool::traits::Tool`, `tool::card::ToolCard` |
| `foundation/tool/function/function.py` `LocalFunction`, `@tool` decorator | `tool::FunctionTool` (macro target) |
| `foundation/tool/service_api/restful_api.py` | `tool::restful` |
| `foundation/tool/mcp/*` | `tool::mcp` (+ `mcp::openapi` for the Python `openapi_client`) |
| `foundation/store/base_kv_store.py`, `kv/in_memory_kv_store.py` | `store::kv` |
| `foundation/store/base_vector_store.py` + `vector/{chroma,milvus}_vector_store.py` | `store::vector`, `store::external::{chroma,milvus}` |
| `extensions/store/vector/es_vector_store.py` | `store::external::elasticsearch` (vector + document) |
| `extensions/store/kv/redis_store.py` | `store::external::redis` |
| `foundation/store/object/aioboto_storage_client.py` | `store::external::s3` (SigV4 in-crate) |
| `foundation/store/{graph,object}/*` (in-process parts) | `store::graph`, `store::object` |

## Deviations from `docs/ARCHITECTURE.md`

- §2.2 message block: `BaseMessage` is one struct (no role subclasses) and lives in `prompt`, not `llm`; `prompt` is the crate's lowest module.
- §2.2 `StreamChunk::Delta` additionally carries `finish_reason` so consumers learn the terminal reason mid-stream.
- §2.3 `ToolCard` is flat (`id`/`name`/`description`/`input_params`/`properties` + policy fields) and implements `openjiuwen_common::card::Card` instead of nesting `CardBase`; transport details live in `RestfulApiConfig`/`McpServerConfig`.
- §2.3 `Tool` adds `stream(self: Arc<Self>) -> BoxStream<'static, Result<Value>>`, `render_for_llm`, and `is_parallel_safe`, all with defaults so external impls stay cheap.
- `ModelExt`/`AssistantTurn`/`StreamAccumulator` are additions (Python accumulates chunks in `AssistantMessageChunk.__add__`).
- `init_model` and `init_model_with` are `async` without awaiting, per §2.2; the Python positional `init_model(provider, model_name, api_key, api_base, ...)` maps to `ModelClientConfig` builders or `init_model_with`.
- MCP HTTP transports are implemented directly on `reqwest` (POST/SSE framing, `Mcp-Session-Id` replay, server-initiated GET stream) because the workspace `rmcp` features do not include its reqwest transports; the `McpClient` seam is unaffected.
- Storage replaces Python's SQLAlchemy-engine `BaseDbStore` with a `DocumentStore` contract; the Python `create_vector_store` entry-point discovery is replaced by the typed `StoreSpec`/`StoreRegistry` (unknown names are errors, not `None`), and vendor SDKs are replaced by REST/RESP adapters over the shared HTTP client.
- DashScope media generation is an HTTP client here (`llm::dashscope`) instead of the Python `dashscope` SDK, and OAuth token storage is a `TokenStore` trait instead of the filelock-guarded `auth.json` reader.

## Known Issues and Gaps

- `pgvector`/`postgres`/`gauss`/`gaussvector` vector backends and every external graph backend select through the registry but return typed `Error::Unsupported` (no PostgreSQL driver and no graph network contract offline).
- Redis: no TLS (`rediss://`), no cluster, and commands serialize behind one connection mutex; the reconnect-retry is a single best-effort retry, not a backoff loop.
- Milvus delegates duplicate-primary-key detection to the service; Chroma supports one embedding per record and scalar-only metadata; Elasticsearch document `query`/`delete` fall back to a bounded local scan (10 000 docs) for filters that do not translate to ES DSL.
- MCP `Playwright` is a typed `Unsupported` (browser runtime absent); the OpenAPI transport resolves only local `$ref`, sends `application/json` bodies, and skips `in: cookie` parameters; the legacy SSE transport has no `Last-Event-ID` replay.
- Anthropic: thinking blocks are still ignored in non-streaming responses; only the three automatic cache anchors exist (no per-message caller breakpoints or caching modes).
- Reasoning: an OpenAI-compatible raw `reasoning` payload is forwarded verbatim while Anthropic rejects it with `Validation`; the Responses parser drops `reasoning.encrypted_content` items instead of replaying them on the next turn.
- Routing: TPM/RPM and group `timeout_seconds` are recorded but not enforced; no circuit breaker or cost/latency scoring; `Custom` strategies are treated as ordered failover.
- OAuth: no `/models` catalog, no login/status/logout CLI, no cross-process file lock (atomic writes, last writer wins), and the Responses client takes an access token as `api_key` rather than reading the auth store itself.
- `ToolInfo` is nested (`{"type":"function","function":{...}}`) rather than Python's flat `{type,name,description,parameters}`; the nested shape is load-bearing in `openjiuwen-dev-tools` and `openjiuwen-agent-evolving`, so flattening needs a cross-crate change.
- Responses: `previous_response_id`/conversation state, `background` mode, hosted tools, and structured `text.format` validation are not ported.
- Model-config decoding: unknown Python keys are tolerated (ignored) rather than forwarded into `extra`; `intelli_router`, `ssl_cert`, `use_shared_llm_http_client`, and `legacy_client_provider` are not typed fields.
- DashScope: no streaming media, no result-URL byte download, no workspace scoping, no voice/language allow-lists.
- There is no `ToolCollection`/tool-registry type: callers hold `Arc<dyn Tool>` values themselves (`FunctionTool::boxed` is the only built-in conversion).
- `Model` has no abort/cancel or custom-invoke method; a stream is cancelled by dropping its `BoxStream`.
- There is no `format_messages` helper; the provider-side projection is `llm::openai::{convert_message, convert_messages}` (public), and `BaseMessage` has no `to_openai`.
- `FunctionTool` handlers are async-only (`spawn_blocking` inside a handler for blocking work); Python's sync handlers and generator-based tool streams are not reproduced.
- `Cargo.toml` declares `openjiuwen-macros.workspace = true`, but no code in the crate references `openjiuwen_macros` and nothing re-exports `#[tool]`, so the dependency is unused (callers add `openjiuwen-macros` themselves).
