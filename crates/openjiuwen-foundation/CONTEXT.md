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

Tools (`tool`):
- `ToolCard` (flat, implements `openjiuwen_common::card::Card`), `ToolInfo`, `ToolExposure`, `ToolExecution`.
- `Tool` trait: `card()`, `invoke(Value, &ToolContext)`, `stream(Value, ToolContext)` (default = single invoke), `render_for_llm(&Value)`.
- `ToolContext { session_id, invocation_id, cancellation, metadata, timeout }`, `ToolOutput`, `render_payload_text`, `render_tool_output`.
- `FunctionTool` + `FunctionToolBuilder` (`handler`, `typed_handler`, `input_schema`, `input_schema_for::<T>`), `schema_for::<T>`, `validate_inputs`, `apply_defaults`.
- `RestfulApiTool` + `RestfulApiConfig` + `ApiParamMapper`/`ParamLocation`.
- `mcp` (feature `mcp`): `McpServerConfig`, `McpClientType`, `McpClient` trait, `connect_mcp_server`, `load_mcp_tools`, `McpTool`, `McpToolResult`.

Storage (`store`):
- Traits `KvStore`, `DocumentStore`, `VectorStore`, `GraphStore`, `ObjectStore` and their data types (`KvValue`, `Document`, `Query`, `CollectionSchema`, `FieldSchema`, `VectorQuery`, `VectorSearchResult`, `GraphNode`, `GraphEdge`, `ObjectMeta`).
- Implementations: `InMemory*` and `LocalFile*`/`Local*` for every contract; `SqliteKvStore`/`SqliteDocumentStore` behind feature `sqlite`.

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
- `./tests/` → wiremock-backed integration tests for clients and tools, storage behavior tests (see Test Strategy)

## Test Strategy

- Unit tests live beside the module they test (`#[cfg(test)] mod tests`), one behavior per test, no network.
- Integration tests live in `tests/`: `wiremock` mocks every HTTP call (`llm_openai.rs`, `llm_openai_stream.rs`, `llm_anthropic.rs`, `tool_restful.rs`); pure conversion/rendering tests need no server (`messages.rs`, `prompt_template.rs`, `tool_function.rs`, `store_*.rs`).
- Behavior-port tests name the Python test they mirror in a comment (e.g. `// ports tests/unit_tests/core/foundation/prompt/test_template_assemble.py`).
- Integration test crates start with `#![allow(clippy::unwrap_used, clippy::expect_used)]`.
- `cargo test -p openjiuwen-foundation` and `cargo test -p openjiuwen-foundation --no-default-features` must both pass offline; `--features sqlite` is exercised separately.
- MCP tests are compile-gated: trait-level tests use a fake `McpClient`; no test spawns a real server or connects to a URL.

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

Not ported (documented in `docs/PARITY.md`): IntelliRouter routing, OpenAI account/OAuth transport, DashScope media generation, reasoning-profile resolution, `client.py` client-side tools, Chroma/Milvus/Gauss/ES/pgvector/S3/Redis backends, OpenAPI and Playwright MCP transports.
