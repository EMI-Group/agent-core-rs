# openjiuwen-foundation

## Intent

Foundation layer of the kernel: messages and prompts, LLM clients and the model factory, tools (local function, REST, MCP), and storage backends.
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
- `ModelClientConfig`, `ModelRequestConfig` with builders and `from_env()`; `ChatResponse`, `Choice`, `FinishReason`, `StreamChunk`, `AssistantTurn`, `StreamAccumulator`, `ModelExt`.
- `init_model(ModelClientConfig, ModelRequestConfig)`, `init_model_with(provider, model_name)`, `ModelProvider`, `ModelProviderRegistry` (`global()` plus local instances), `register_model_provider`, `provider_profile`, `normalize_provider_name`.
- Flat crate-root re-exports (`src/lib.rs`) for call sites that skip the module path: `openjiuwen_foundation::{Model, ModelExt, ChatResponse, Choice, StreamChunk, FinishReason, AssistantTurn, StreamAccumulator, ModelClientConfig, ModelRequestConfig, BaseMessage, MessageContent, Role, Usage, ToolCall, ToolCallDelta, ToolCallAccumulator}`; the module paths (`openjiuwen_foundation::llm::*`, `::prompt::*`) remain the canonical form.

Tools (`tool`):
- `ToolCard` (flat, implements `openjiuwen_common::card::Card`), `ToolInfo`, `ToolExposure`, `ToolExecution`.
- `Tool` trait: `card()`, `invoke(Value, &ToolContext)`, `stream(Value, ToolContext)` (default = single invoke), `render_for_llm(&Value)`.
- `ToolContext { session_id, invocation_id, cancellation, metadata, timeout }`, `ToolOutput`, `render_payload_text`, `render_tool_output`.
- `FunctionTool` + `FunctionToolBuilder` (`handler`, `typed_handler`, `input_schema`, `input_schema_for::<T>`), `schema_for::<T>`, `validate_inputs`, `apply_defaults`.
- Crate-root re-export `pub use async_trait;` so downstream hand-written `Tool`/`Model` impls can write `#[openjiuwen_foundation::async_trait::async_trait]` without a direct `async-trait` dependency; macro-generated tools still reference `::async_trait::` and therefore need `async-trait` in the consumer crate.
- `RestfulApiTool` + `RestfulApiConfig` + `ApiParamMapper`/`ParamLocation`.
- `mcp` (feature `mcp`): `McpServerConfig`, `McpClientType`, `McpClient` trait, `connect_mcp_server`, `load_mcp_tools`, `McpTool`, `McpToolResult`.

Storage (`store`):
- Traits `KvStore`, `DocumentStore`, `VectorStore`, `GraphStore`, `ObjectStore` and their data types (`KvValue`, `Document`, `Query`, `CollectionSchema`, `FieldSchema`, `VectorQuery`, `VectorSearchResult`, `GraphNode`, `GraphEdge`, `ObjectMeta`).
- Implementations: `InMemory*` and `LocalFile*`/`Local*` for every contract; `SqliteKvStore`/`SqliteDocumentStore` behind feature `sqlite`.
- There is no `KnowledgeBase`, embedding, chunking, or reranker type in this crate; those live in `openjiuwen-retrieval`, which consumes the `store::VectorStore` contract here.

## Constraints

- Error type is `openjiuwen_common::error::{Error, Result}`; no panics on user input, no ad-hoc error enums across the public surface.
- Async-first on tokio; public streams are `Send + 'static`; cancellation via `tokio_util::sync::CancellationToken`; no blocking I/O on async tasks.
- Every public data type is `serde`-serializable, snake_case, tolerant of unknown fields; `#[serde(default)]` on structs; secrets use `openjiuwen_common::security::SecretString` and are never logged or `Debug`-printed.
- HTTP goes through `openjiuwen_common::http::HttpClient` (retry/backoff, TLS, auth headers) — never raw `reqwest` calls in this crate.
- `tracing` only; no `println!`/`eprintln!`; logging targets are `openjiuwen_foundation::<module>`.
- No `unsafe`; workspace lints are `deny`, and `cargo clippy --all-targets -- -D warnings` must pass, so every public item needs a doc comment.
- Files stay around ≤1000 lines; shared helpers live at the lowest common module (`prompt` types for the clients, `store::filter` for all backends).
- Default features (`mcp`, `tiktoken`) compile and test offline; no test may hit the network or need credentials.

## Design Decisions

- **Messages live in `prompt`**, not `llm`: the Python `llm/schema` message types are the crate's most widely used data types, and `llm` depends on `prompt`, never the reverse. `llm` re-exports `BaseMessage`, `ToolCall`, `Usage` for Python-shaped call sites.
- **One `BaseMessage` struct** instead of Python's four role subclasses: role-specific construction lives in constructors; serde rejects nothing and providers never need downcasting. This is a deliberate deviation from the Python class hierarchy.
- **Flat `ToolCard`** instead of nesting `CardBase` (deviation from `docs/ARCHITECTURE.md` §2.3): `id`, `name`, `description`, `input_params`, policy flags and `metadata` are direct fields and `ToolCard` implements the `Card` trait. Transport details (REST url/method, MCP server) live in the tool struct, matching the Python Card/Config split.
- **`StreamChunk::Delta` additionally carries `finish_reason`** (deviation from §2.2): consumers need the terminal reason without a second round-trip, and providers emit it mid-stream. `StreamChunk::Done` still marks the end of the stream.
- **Provider registry**: built-in OpenAI-compatible providers are data (`ProviderProfile`), not code; `openai`, `pangu`, `qwen`, `deepseek` and the other OpenAI-compatible aliases all resolve to one client. `anthropic` is the only distinct transport. Extension crates register `ModelProvider` impls on the global registry; tests build local registries.
- **Retries** are the shared `HttpClient`'s job (`max_retries`, `RetryPolicy`, backoff). Provider clients retry only up to the connect boundary; a mid-stream failure surfaces as an error.
- **Storage**: vendor backends are out of scope. In-memory implementations are the behavioral reference; the local-filesystem variants reuse the same in-memory engine with JSON persistence so behavior cannot drift.

## Routing Table

- `./src/prompt/` → messages, content parts, tool calls, `PromptTemplate`, token helpers (`src/prompt/CONTEXT.md`)
- `./src/llm/` → `Model` trait, configs, providers, OpenAI-compatible and Anthropic clients, `init_model` (`src/llm/CONTEXT.md`)
- `./src/tool/` → `ToolCard`, `Tool` trait, `FunctionTool`, REST tool, MCP (`src/tool/CONTEXT.md`)
- `./src/store/` → kv/document/vector/graph/object traits and backends (`src/store/CONTEXT.md`)
- `./src/llm/openai/` → OpenAI-compatible client and its pure request/response/stream conversion + wiremock tests
- `./src/llm/anthropic/` → Anthropic Messages client, message projection, event stream
- `./src/tool/restful/` → REST tool config, parameter mapper, request/envelope building
- `./src/tool/mcp/` → MCP server config, `McpClient` + rmcp transports, `McpTool` (`src/tool/mcp/CONTEXT.md`)
- `./src/store/{kv,document,vector,graph,object}/` → one directory per storage contract (memory / local-file / sqlite backends)
- `./src/store/tests/` → cross-backend contract suites compiled under `#[cfg(test)]`
- `./tests/` → crate-level integration tests (`messages.rs`, `prompt_template.rs`); every other test lives in module-level `#[cfg(test)]` trees next to the code it exercises

## Test Strategy

- Tests live beside the code they exercise, mostly as module-level `#[cfg(test)]` trees (`src/llm/openai/tests.rs`, `src/llm/anthropic/tests.rs`, `src/tool/tests/…`, `src/tool/mcp/tests.rs`, `src/store/tests/…`); only `tests/messages.rs` and `tests/prompt_template.rs` are standalone integration crates.
- `wiremock` mocks every HTTP call (both clients and the REST tool); no test needs the network, an API key, or a spawned MCP server — MCP uses `FakeMcpClient`.
- Storage behavior is covered by one contract suite per trait (`assert_kv_contract`, `assert_document_contract`, `assert_vector_contract`, `assert_graph_contract`, `assert_object_contract`) instantiated for every backend, so backends cannot drift.
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
| `foundation/tool/base.py` `Tool`/`ToolCard` | `tool::traits::Tool`, `tool::card::ToolCard` |
| `foundation/tool/function/function.py` `LocalFunction`, `@tool` decorator | `tool::FunctionTool` (macro target) |
| `foundation/tool/service_api/restful_api.py` | `tool::restful` |
| `foundation/tool/mcp/*` | `tool::mcp` |
| `foundation/store/base_kv_store.py`, `kv/in_memory_kv_store.py` | `store::kv` |
| `foundation/store/base_vector_store.py` | `store::vector` |
| `foundation/store/{graph,object}/*` | `store::graph`, `store::object` |

## Deviations from `docs/ARCHITECTURE.md`

- §2.2 message block: `BaseMessage` is one struct (no role subclasses) and lives in `prompt`, not `llm`; `prompt` is the crate's lowest module.
- §2.2 `StreamChunk::Delta` additionally carries `finish_reason` so consumers learn the terminal reason mid-stream.
- §2.3 `ToolCard` is flat (`id`/`name`/`description`/`input_params` + policy fields) and implements `openjiuwen_common::card::Card` instead of nesting `CardBase`; transport details live in `RestfulApiConfig`/`McpServerConfig`.
- §2.3 `Tool` adds `stream(self: Arc<Self>) -> BoxStream<'static, Result<Value>>`, `render_for_llm`, and `is_parallel_safe`, all with defaults so external impls stay cheap.
- `ModelExt`/`AssistantTurn`/`StreamAccumulator` are additions (Python accumulates chunks in `AssistantMessageChunk.__add__`).
- `init_model` and `init_model_with` are `async` without awaiting, per §2.2; the Python positional `init_model(provider, model_name, api_key, api_base, ...)` maps to `ModelClientConfig` builders or `init_model_with`.
- MCP HTTP transports are implemented directly on `reqwest` (POST/SSE framing, `Mcp-Session-Id` replay) because the workspace `rmcp` features do not include its reqwest transports; the `McpClient` seam is unaffected.
- Storage replaces Python's SQLAlchemy-engine `BaseDbStore` with a `DocumentStore` contract; there is no `KvStore`/`VectorStore` factory registry (Python's `create_vector_store` plugin discovery is not ported).

## Known Issues and Gaps

- MCP streamable HTTP opens no separate server-initiated `GET` stream; server requests arrive only on a POST's SSE stream.
- There is no `ToolCollection`/tool-registry type: callers hold `Arc<dyn Tool>` values themselves (`FunctionTool::boxed` is the only built-in conversion).
- `Model` has no abort/cancel or custom-invoke method; a stream is cancelled by dropping its `BoxStream`.
- There is no `format_messages` helper; the provider-side projection is `llm::openai::{convert_message, convert_messages}` (public), and `BaseMessage` has no `to_openai`.
- Not ported (tracked in `docs/PARITY.md`): reasoning-profile resolution, IntelliRouter routing, OpenAI account/OAuth transport, DashScope media generation, `client.py` client-side tools, Anthropic `cache_control` breakpoints and mid-conversation `system` turns, Chroma/Milvus/Gauss/ES/pgvector/S3/Redis backends, OpenAPI and Playwright MCP transports.
- `FunctionTool` handlers are async-only (`spawn_blocking` inside a handler for blocking work); Python's sync handlers and generator-based tool streams are not reproduced.
- `Cargo.toml` declares `openjiuwen-macros.workspace = true`, but no code in the crate references `openjiuwen_macros` and nothing re-exports `#[tool]`, so the dependency is unused (callers add `openjiuwen-macros` themselves).
