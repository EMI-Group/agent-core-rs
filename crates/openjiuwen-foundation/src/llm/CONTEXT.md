# llm — models, configs, providers, clients

## Intent

Provider-neutral model invocation: the `Model` trait, the two transports (OpenAI-compatible chat completions, Anthropic Messages), the serializable configs, and the provider registry/factory.
Ports `openjiuwen/core/foundation/llm/` (`model.py`, `schema/config.py`, `model_clients/*`).

## API Surface

`config`:
- `ModelClientConfig { client_id, provider, api_key: Option<SecretString>, api_base: Option<String>, verify_ssl: bool /* true */, timeout: Option<Duration> /* 360s */, stream_first_chunk_timeout: Option<Duration> /* 300s */, stream_idle_timeout: Option<Duration> /* 60s */, max_retries: u32 /* 0 */, headers: BTreeMap<String,String>, extra: Value }`
  Builders `new(provider)`, `with_api_key`, `with_api_base`, `with_timeout`, `with_max_retries`, `with_header`, `with_verify_ssl`, `from_env()`; `resolve_for(provider)` applies env fallbacks and provider defaults; `sanitized_headers()` drops protected names (`host`, `content-length`, `transfer-encoding`, `connection`, `authorization`) and returns the rest.
  serde aliases: `client_provider` → `provider`.
- `ModelRequestConfig { model_name, temperature, top_p, max_tokens, stop: Vec<String>, tools: Option<Value>, tool_choice: Option<Value>, stream: bool, extra: Value }` with builders (`with_temperature`, `with_max_tokens`, `with_tools(Vec<Value>)`, `with_tool_choice`, `with_stop`, `with_stream`, `with_extra`) and `from_env()`; serde alias `model` → `model_name`.

`model`:
- `Model` trait: `provider()`, `model_name()`, `invoke(&[BaseMessage], &ModelRequestConfig)`, `stream(Vec<BaseMessage>, ModelRequestConfig) -> BoxStream<'static, Result<StreamChunk>>`.
- `ChatResponse { id, model, choices: Vec<Choice>, usage: Usage }`, `Choice { index, message: BaseMessage, finish_reason: Option<FinishReason> }`.
- `FinishReason { Stop, Length, ToolCalls, ContentFilter, Other(String) }` with `as_str()` (`"stop"`, `"length"`, `"tool_calls"`, `"content_filter"`).
- `StreamChunk::Delta { index, role, content, reasoning, tool_call, finish_reason }`, `StreamChunk::Usage(Usage)`, `StreamChunk::Done`.
- `AssistantTurn { message, finish_reason, usage }`, `StreamAccumulator` (`push(&StreamChunk)`, `finish() -> AssistantTurn`), `ModelExt::complete` (drive a stream to an `AssistantTurn`).

`provider` / `registry` / `factory`:
- `ProviderProfile { name, flavor: ProviderFlavor (OpenAiCompatible | Anthropic), default_api_base: Option<&'static str>, chat_path: &'static str, extra_headers: &'static [(&'static str, &'static str)] }`, `provider_profile(name)`, `normalize_provider_name(name)`, `ANTHROPIC_VERSION`.
- `ModelProvider` trait: `name()`, `aliases()`, `default_api_base()`, `create(&ModelClientConfig, &ModelRequestConfig) -> Result<Arc<dyn Model>>`.
- `ModelProviderRegistry` (`new`, `register`, `get`, `resolve`, `names`, `global`), `register_model_provider`, `register_builtin_providers`.
- `init_model(ModelClientConfig, ModelRequestConfig) -> Result<Arc<dyn Model>>`, `init_model_with(provider, model_name) -> Result<Arc<dyn Model>>`.

## Constraints

- Only `openjiuwen_common::http::HttpClient` performs I/O; no direct `reqwest` calls.
- `api_key` is a `SecretString`: never logged, never formatted; provider errors redact the body (`security::redact_str`) and truncate it to ~1000 chars.
- All bodies are built by pure functions (`build_chat_completion_body`, `build_messages_body`) and parsed by pure functions, so tests can assert exact JSON without a server.
- `provider` strings are case- and separator-insensitive: `normalize_provider_name` lowercases, strips `-`/`_`/spaces and maps vendor aliases.
- Streaming never panics on malformed events: undecodable SSE data is logged and skipped; a terminal `error` event becomes `Err(Error::Llm)`.

## Design Decisions

- **Config defaults** follow `ModelClientConfig` in Python (timeout 360 s, `verify_ssl` true, `max_retries` 0); `init_model_with` (the Python `init_model` analogue) uses the more permissive factory defaults (timeout 60 s, `verify_ssl` false, `max_retries` 3) — documented so callers are not surprised.
- **Environment fallbacks** (Python has none in this package, `docs/ARCHITECTURE.md` §4 requires them): `MODEL_PROVIDER` (default `openai`), `API_KEY` → `OPENAI_API_KEY` → `OPENJIUWEN_API_KEY`, `API_BASE` → `OPENAI_BASE_URL` → provider default, `MODEL_NAME`. Explicit config always wins.
- **Provider table is data, not code**: `openai` `https://api.openai.com/v1`, `qwen`/`dashscope` `https://dashscope.aliyuncs.com/compatible-mode/v1`, `deepseek` `https://api.deepseek.com/v1`, `moonshot` `https://api.moonshot.cn/v1`, `openrouter` `https://openrouter.ai/api/v1`, `siliconflow` `https://api.siliconflow.cn/v1`, `zhipu` `https://open.bigmodel.cn/api/paas/v4`, `minimax` `https://api.minimax.chat/v1`, `ollama` `http://localhost:11434/v1`, `lmstudio` `http://localhost:1234/v1`, `anthropic` `https://api.anthropic.com`.
  `pangu` is OpenAI-compatible but has **no** built-in base URL (air-gapped deployments), so it requires `api_base` and fails with `Error::Config` otherwise.
  No default model names are invented; `MODEL_NAME` or an explicit `model_name` is required.
- **`Model::stream` takes ownership** of messages/request so the returned stream is `'static`; `invoke` borrows. `BoxStream<'static, Result<StreamChunk>>` is built with `async_stream::try_stream!`.
- **Retries** are delegated to `HttpClient` (`RetryPolicy`, exponential backoff, retryable statuses); clients therefore set the retry budget from `max_retries` and do not implement their own loop. Streams retry only up to the connect boundary.
- **Timeouts**: per-request `timeout` on the HTTP request; streaming applies `stream_first_chunk_timeout` and `stream_idle_timeout` with `tokio::time::timeout`, yielding `Error::Timeout` (Python uses the same two budgets).
- **Error mapping**: 401/403 → `Error::Config` (auth), 408/504 → `Error::Timeout`, other non-2xx → `Error::Llm`, transport failures and JSON decode errors propagate from `HttpClient`/serde with `Error::transport`/`Error::Llm` context.
- **`finish_reason` semantics**: `None` until the provider reports one; when the response carries tool calls but no reason, `FinishReason::ToolCalls` is inferred. The Python `"null"` string sentinel is not reproduced.

## OpenAI-compatible wire contract

- Request: `{model, messages, stream}` plus `temperature`/`top_p`/`max_tokens`/`stop` when set, `tools`/`tool_choice` only when tools are present (`tool_choice` defaults to `"auto"`), and `extra` merged last (caller wins). `stream: true` adds `stream_options: {"include_usage": true}`.
- Messages: `{"role", "content"}`; assistant turns add `tool_calls: [{id, type:"function", function:{name, arguments}}]` (arguments is the JSON string) — and replay `reasoning_content` when the provider profile is DeepSeek-like; tool turns add `tool_call_id`. Parts pass through verbatim (provider-shaped), so no normalization happens here.
- Response: `choices[0].message` → `BaseMessage` (`content` string or parts; `tool_calls` flattened with `index`); `usage.prompt_tokens`/`completion_tokens`/`total_tokens`, `prompt_tokens_details.cached_tokens` → `cache_tokens`, `completion_tokens_details.reasoning_tokens` → `reasoning_tokens`.
- Stream frames: `data: {...}` lines terminated by `data: [DONE]`; `choices[].delta.content` → `content`, `.reasoning_content` → `reasoning`, `.tool_calls[]` (with `index`, `id`, `function.name`, `function.arguments`) → `ToolCallDelta`, `.finish_reason` → `finish_reason`, `role` on the first frame; choice-less frames carrying `usage` → `StreamChunk::Usage`; every stream terminates with `StreamChunk::Done`.

## Anthropic wire contract

- Request: `POST {base}/v1/messages` with `x-api-key` and `anthropic-version: 2023-06-01`; body `{model, messages, max_tokens, stream}`.
  `max_tokens` is required: `request.max_tokens` else 32768, capped at 8192 for `claude-3-5*`/`claude-3.5*` and 4096 for `claude-3-opus|sonnet|haiku`.
  All `system` turns are hoisted into a top-level `system` string (text blocks joined) instead of staying in `messages`.
  Assistant `tool_calls` → `{"type":"tool_use","id","name","input":<parsed JSON>}` (unparseable arguments → `{"_raw_arguments": "..."}`); consecutive `tool` messages merge into one `user` message of `{"type":"tool_result","tool_use_id","content"}` blocks.
  Images: `data:image/png;base64,...` → base64 `source`; `http(s)` urls → url `source`.
  Tools: `[{name, description, input_schema}]`; `tool_choice` mapping `auto`→omitted, `required`→`{"type":"any"}`, `none`→`{"type":"none"}`, `{"function":{"name":N}}`→`{"type":"tool","name":N}`; `stop` → `stop_sequences`.
- Response: `content` blocks → text concatenated into `message.content`, `tool_use` → `ToolCall` with `arguments = json.dumps(input)`; `stop_reason` mapping `tool_use` → `ToolCalls`, `end_turn`/`stop_sequence`/`max_tokens` → `Stop`/`Length`, `max_tokens` also → `Length`; usage `input_tokens` includes `cache_read_input_tokens` + `cache_creation_input_tokens`, `cache_tokens` = cache read, `reasoning_tokens` = thinking blocks ignored for now.
- Stream: `message_start` (role + input usage), `content_block_start` (text or `tool_use` with id/name), `content_block_delta` (`text_delta`, `thinking_delta`, `input_json_delta` buffered per index), `content_block_stop` (emit the accumulated `ToolCall` for `tool_use`), `message_delta` (`stop_reason`, usage), `message_stop`, `ping` (ignored).

## Rust Seams (binding signatures)

Every item below is public unless marked `pub(crate)`; parallel work in this module targets these exact names and shapes.

`config.rs`:
- `pub struct ModelClientConfig { pub client_id: String, pub provider: String, pub api_key: Option<SecretString>, pub api_base: Option<String>, pub verify_ssl: bool, pub timeout: Option<Duration>, pub stream_first_chunk_timeout: Option<Duration>, pub stream_idle_timeout: Option<Duration>, pub max_retries: u32, pub headers: BTreeMap<String, String>, pub extra: Value }`.
  `Default` generates a UUID v4 `client_id` (as Python's `default_factory`), sets `verify_ssl: true`, `timeout`/first-chunk/idle to 360/300/60 s and `max_retries: 0`.
- Builders: `new(impl Into<String>)`, `with_client_id`, `with_api_key(impl Into<String>)`, `with_secret(SecretString)`, `with_api_base`, `with_timeout(Duration)`, `with_stream_timeouts(Option<Duration>, Option<Duration>)`, `with_max_retries(u32)`, `with_header`, `with_headers`, `with_verify_ssl(bool)`, `with_extra(Value)`.
- `from_env() -> Self`, `resolve_for(&self, provider: &str) -> Result<Self>`, `sanitized_headers(&self) -> BTreeMap<String, String>`, `http_client_config(&self) -> HttpClientConfig`.
- `pub struct ModelRequestConfig { pub model_name: String, pub temperature: Option<f32>, pub top_p: Option<f32>, pub max_tokens: Option<u32>, pub stop: Vec<String>, pub tools: Option<Value>, pub tool_choice: Option<Value>, pub stream: bool, pub extra: Value }` plus the builders listed in API Surface and `from_env() -> Self`.

`model.rs`:
- `#[async_trait] pub trait Model: Send + Sync { fn provider(&self) -> &str; fn model_name(&self) -> &str; async fn invoke(&self, messages: &[BaseMessage], request: &ModelRequestConfig) -> Result<ChatResponse>; fn stream(&self, messages: Vec<BaseMessage>, request: ModelRequestConfig) -> BoxStream<'static, Result<StreamChunk>>; }`.
- `pub struct ChatResponse`, `pub struct Choice`, `pub enum FinishReason`, `pub enum StreamChunk`, `pub struct AssistantTurn` exactly as in API Surface.
- `pub struct StreamAccumulator` with `new()`, `push(&mut self, chunk: &StreamChunk)`, `finish(self) -> AssistantTurn`.
- `impl Model for Arc<dyn Model>` forwards to the inner model, so `ModelExt` applies to `Arc<dyn Model>`.
- `pub trait ModelExt: Model` with `async fn complete(&self, messages: &[BaseMessage], request: &ModelRequestConfig) -> Result<AssistantTurn>` (default body: force `stream = true` on a clone of the request, drive `Model::stream`, feed a `StreamAccumulator`) and a blanket `impl<T: Model + ?Sized> ModelExt for T`.

`provider.rs`:
- `pub enum ProviderFlavor { OpenAiCompatible, Anthropic }` (serde snake_case).
- `pub struct ProviderProfile { pub name: &'static str, pub flavor: ProviderFlavor, pub default_api_base: Option<&'static str>, pub chat_path: &'static str, pub extra_headers: &'static [(&'static str, &'static str)] }`.
- `pub fn normalize_provider_name(name: &str) -> String`, `pub fn provider_profile(name: &str) -> Option<ProviderProfile>`, `pub const ANTHROPIC_VERSION: &str = "2023-06-01";`.

`registry.rs`:
- `pub trait ModelProvider: Send + Sync { fn name(&self) -> &str; fn aliases(&self) -> &[&str]; fn default_api_base(&self) -> Option<&str>; fn create(&self, client: &ModelClientConfig, request: &ModelRequestConfig) -> Result<Arc<dyn Model>>; }` (the last three have default bodies).
- `pub struct ModelProviderRegistry` with `new()`, `register(Arc<dyn ModelProvider>) -> Option<Arc<dyn ModelProvider>>` (replaces a same-name entry and returns the old one), `get(&str)`, `resolve(&str)`, `names() -> Vec<String>`, `global() -> &'static Self`.
- `pub fn register_model_provider(provider: Arc<dyn ModelProvider>)`, `pub fn register_builtin_providers()`.

`sse.rs`:
- `pub struct SseEvent { pub event: Option<String>, pub data: String }`; `pub struct SseDecoder` with `new()`, `push(&mut self, chunk: &[u8]) -> Vec<SseEvent>`, `finish(&mut self) -> Option<SseEvent>`.

`transport.rs` (`pub(crate)`, shared by both clients):
- `pub(crate) fn join_url(base: &str, path: &str) -> String`, `pub(crate) fn error_for_status(status: u16, url: &str, body: &str) -> Error`, `pub(crate) fn redacted_body(body: &str) -> String` (redact + truncate to 1000 chars).

`openai/mod.rs`:
- `pub struct OpenAiCompatibleClient` with `new(ModelClientConfig, ModelRequestConfig) -> Result<Self>`, `with_http_client(HttpClient) -> Self`, `client_config() -> &ModelClientConfig`, `request_config() -> &ModelRequestConfig`, `base_url() -> &str`, and `impl Model`.
- Re-exports: `pub use request::{build_chat_completion_body, normalize_api_base, convert_message, convert_messages, convert_tool_choice, convert_tools};`, `pub use response::{parse_chat_completion, parse_finish_reason, parse_usage};`, `pub use stream::{parse_stream_chunk, OpenAiStreamState};`.

`anthropic/mod.rs`:
- `pub struct AnthropicClient` with the same inherent surface as `OpenAiCompatibleClient` plus `impl Model`.
- Re-exports: `pub use request::{build_messages_body, normalize_api_base};`, `pub use response::{parse_anthropic_usage, parse_messages_response};`, `pub use stream::AnthropicStreamState;`.

`factory.rs`:
- `pub async fn init_model(client: ModelClientConfig, request: ModelRequestConfig) -> Result<Arc<dyn Model>>`, `pub async fn init_model_with(provider: impl Into<String>, model_name: impl Into<String>) -> Result<Arc<dyn Model>>`.

`mod.rs` re-exports (in addition to the module list): `AssistantTurn`, `ChatResponse`, `Choice`, `FinishReason`, `Model`, `ModelExt`, `ModelClientConfig`, `ModelRequestConfig`, `StreamAccumulator`, `StreamChunk`, `ProviderFlavor`, `ProviderProfile`, `ANTHROPIC_VERSION`, `normalize_provider_name`, `provider_profile`, `ModelProvider`, `ModelProviderRegistry`, `register_builtin_providers`, `register_model_provider`, `SseDecoder`, `SseEvent`, `init_model`, `init_model_with`, `OpenAiCompatibleClient`, `AnthropicClient`.

## Routing Table

- `./transport.rs` → shared URL joining, status→error mapping, redacted/truncated provider error bodies
- `./config.rs` → `ModelClientConfig`, `ModelRequestConfig`, env resolution, header sanitization
- `./model.rs` → `Model`, `ChatResponse`, `Choice`, `FinishReason`, `StreamChunk`, `AssistantTurn`, `StreamAccumulator`, `ModelExt`
- `./provider.rs` → provider table, profile lookup, name normalization
- `./registry.rs` → `ModelProvider`, `ModelProviderRegistry`, built-ins
- `./factory.rs` → `init_model`, `init_model_with`
- `./sse.rs` → incremental SSE decoder shared by both clients
- `./openai/` → OpenAI-compatible client (`request.rs`, `response.rs`, `stream.rs`)
- `./anthropic/` → Anthropic Messages client (`request.rs`, `response.rs`, `stream.rs`)
