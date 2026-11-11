# openjiuwen-test-support

## Intent

Dev-only shared test infrastructure for the Rust rewrite: a real loopback OpenAI-compatible HTTP server, an in-process scripted `Model`, temp-directory and environment fixtures, JSON assertions, and async eventual-consistency helpers.
It replaces two Python patterns from `tests/unit_tests/fixtures/mock_llm.py`: the scripted `MockLLMModel` (ordered response queue, no prompt matching) and `httpx.MockTransport` request fakes.
The Rust version upgrades the HTTP fake to a real loopback server so the actual `OpenAiCompatibleClient` transport, SSE decoder, and error mapping are exercised instead of stubbed.
Never depended on by a library crate; only `[dev-dependencies]` and `openjiuwen-e2e` may use it.

## API Surface

Crate root re-exports every public item listed below, so downstream tests use `openjiuwen_test_support::MockLlmServer` without module paths.
`pub use wiremock;` is re-exported (with its `MockServer`, `ResponseTemplate`, `matchers`) so test crates can mock non-LLM endpoints without adding their own dependency; the mock server here is NOT built on wiremock and does not duplicate it.

### Scripted completions — `src/responses.rs`

Shared by the server and the scripted model: one neutral description of an assistant turn that converts to/from `ChatResponse` and to/from `StreamChunk` sequences.

```rust
pub struct MockToolCall { pub id: String, pub name: String, pub arguments: String }
impl MockToolCall {
    pub fn new(name: impl Into<String>, arguments: Value) -> Self;      // compact JSON arguments, id "call_<name>"
    pub fn with_id(self, id: impl Into<String>) -> Self;
    pub fn with_raw_arguments(self, arguments: impl Into<String>) -> Self;
}

pub struct MockCompletion {
    pub id: Option<String>, pub model: Option<String>,
    pub content: Option<String>, pub reasoning: Option<String>,
    pub tool_calls: Vec<MockToolCall>, pub finish_reason: FinishReason, pub usage: Usage,
}
impl MockCompletion {
    pub fn text(content: impl Into<String>) -> Self;                    // finish_reason Stop
    pub fn json(content: impl Into<String>) -> Self;                    // alias of text, documents intent
    pub fn reasoning(reasoning: impl Into<String>, content: impl Into<String>) -> Self;
    pub fn tool_call(name: impl Into<String>, arguments: Value) -> Self; // finish_reason ToolCalls
    pub fn tool_calls(calls: Vec<MockToolCall>) -> Self;
    pub fn with_id(self, id: impl Into<String>) -> Self;
    pub fn with_model(self, model: impl Into<String>) -> Self;
    pub fn with_usage(self, usage: Usage) -> Self;
    pub fn with_finish_reason(self, reason: FinishReason) -> Self;
    pub fn with_reasoning(self, reasoning: impl Into<String>) -> Self;
    pub fn push_tool_call(&mut self, call: MockToolCall);
    pub fn has_tool_calls(&self) -> bool;
    pub fn to_chat_response(&self, default_model: &str) -> ChatResponse;
    pub fn to_chunks(&self) -> Vec<StreamChunk>;                        // role delta, content/reasoning delta, tool deltas, usage, finish, Done
    pub fn from_chunks(chunks: &[StreamChunk]) -> ChatResponse;         // folds with StreamAccumulator; model/id empty
}
impl From<ChatResponse> for MockCompletion;
pub fn text_chunks(content: impl Into<String>) -> Vec<StreamChunk>;     // MockCompletion::text(..).to_chunks()
pub fn tool_call_chunks(calls: Vec<MockToolCall>) -> Vec<StreamChunk>;
```

### Mock LLM server — `src/mock_llm/`

```rust
pub struct MockLlmServer;
impl MockLlmServer {
    pub fn builder() -> MockLlmServerBuilder;
    pub async fn start() -> Self;                                       // empty script + default fallback; panics only on bind failure
    pub async fn try_start() -> Result<Self>;
    pub async fn with_script(items: impl IntoIterator<Item = ScriptedResponse>) -> Self;
    pub fn base_url(&self) -> &str;                                     // "http://127.0.0.1:<port>/v1"
    pub fn endpoint(&self) -> &str;                                     // base_url + "/chat/completions"
    pub fn addr(&self) -> SocketAddr;
    pub fn client_config(&self) -> ModelClientConfig;                   // provider "openai", key "test-api-key", api_base base_url
    pub fn request_config(&self, model: &str) -> ModelRequestConfig;
    pub fn openai_client(&self, model: &str) -> Result<OpenAiCompatibleClient>;
    pub fn push(&self, response: ScriptedResponse);
    pub fn push_many(&self, items: impl IntoIterator<Item = ScriptedResponse>);
    pub fn clear_script(&self);
    pub fn set_fallback(&self, response: ScriptedResponse);
    pub fn queued(&self) -> usize;
    pub fn request_count(&self) -> usize;
    pub fn requests(&self) -> Vec<RecordedRequest>;
    pub fn last_request(&self) -> Option<RecordedRequest>;
    pub async fn wait_for_requests(&self, count: usize);                // waits indefinitely
    pub async fn wait_for_requests_timeout(&self, count: usize, timeout: Duration) -> Result<()>;
    pub async fn shutdown(self);                                        // graceful; aborts the task on Drop otherwise
}

pub struct MockLlmServerBuilder;                                        // Default; every method consumes and returns Self
impl MockLlmServerBuilder {
    pub fn script(self, items: impl IntoIterator<Item = ScriptedResponse>) -> Self;
    pub fn item(self, response: ScriptedResponse) -> Self;
    pub fn fallback(self, response: ScriptedResponse) -> Self;          // default MockCompletion::text("Default mock response")
    pub fn id_prefix(self, prefix: impl Into<String>) -> Self;          // default "mock-"
    pub fn default_model(self, model: impl Into<String>) -> Self;       // default "mock-model"
    pub fn chunk_delay(self, delay: Duration) -> Self;                  // per SSE frame delay, default zero
    pub async fn start(self) -> MockLlmServer;
    pub async fn try_start(self) -> Result<MockLlmServer>;
}

pub enum ScriptedResponse { Completion(MockCompletion), Chunks(Vec<StreamChunk>), Error(MockHttpError) }
impl ScriptedResponse {
    pub fn text(content: impl Into<String>) -> Self;
    pub fn completion(completion: MockCompletion) -> Self;
    pub fn chunks(chunks: Vec<StreamChunk>) -> Self;
    pub fn error(status: u16, message: impl Into<String>) -> Self;
}
impl From<MockCompletion> for ScriptedResponse;

pub struct MockHttpError { pub status: u16, pub body: Value }
impl MockHttpError {
    pub fn new(status: u16, message: impl Into<String>) -> Self;        // OpenAI error envelope
    pub fn with_body(status: u16, body: Value) -> Self;
    pub fn rate_limited() -> Self;                                      // 429
    pub fn unauthorized() -> Self;                                      // 401
    pub fn server_error(message: impl Into<String>) -> Self;            // 500
    pub fn not_found() -> Self;                                         // 404
}

pub struct RecordedRequest {
    pub method: String, pub path: String, pub query: Option<String>,
    pub headers: BTreeMap<String, String>,                              // names lowercased
    pub body: Value,                                                    // parsed JSON, Value::Null when unparsable
    pub raw_body: String, pub stream: bool, pub model: Option<String>,
    pub response_id: Option<String>,
}
impl RecordedRequest {
    pub fn header(&self, name: &str) -> Option<&str>;                   // case-insensitive
    pub fn messages(&self) -> Vec<Value>;                               // body.messages entries
    pub fn texts_of_role(&self, role: &str) -> Vec<String>;
    pub fn user_texts(&self) -> Vec<String>;
}
```

HTTP surface: `POST /v1/chat/completions` and `POST /chat/completions` (alias), `GET /health` (`{"status":"ok"}`), JSON 404 for anything else, JSON 400 for a malformed body.
Loopback only (`127.0.0.1:0`, OS-assigned port).
Non-streaming requests receive an OpenAI chat-completion object (`id`, `object`, `created`, `model`, `choices[].message`, `usage` with `prompt_tokens`/`completion_tokens`/`total_tokens`, `reasoning_content`, `tool_calls[].function`).
Streaming requests receive `text/event-stream` frames `data: <json>\n\n` ending with `data: [DONE]\n\n`; frames carry role deltas, `content`, `reasoning_content`, `tool_calls` deltas, a usage-only frame (`choices: []`) when usage is non-empty, and a finish-reason frame, matching `openjiuwen-foundation`'s `parse_stream_chunk` expectations.
Each request pops exactly one scripted response (`Completion` is serialized for non-streaming and expanded to frames for streaming; `Chunks` is emitted as SSE and folded for non-streaming; `Error` short-circuits with its status/body).
When the queue is empty the fallback response is served without consuming anything (Python parity: `"Default mock response"`).
Response ids are deterministic: `chatcmpl-{id_prefix}{seq:04}`, `seq` counting popped responses from 1; a `MockCompletion::id` overrides it, and the id is also stamped on the request recording.

### Scripted model — `src/scripted_model.rs`

```rust
pub struct ScriptedModel;                                               // Clone; shares one script/recording state
impl ScriptedModel {
    pub fn builder() -> ScriptedModelBuilder;
    pub fn new(turns: impl IntoIterator<Item = ScriptedTurn>) -> Self;  // provider "scripted", model "scripted-model"
    pub fn text(content: impl Into<String>) -> Self;                    // one-turn text model
    pub fn push(&self, turn: ScriptedTurn);
    pub fn push_many(&self, turns: impl IntoIterator<Item = ScriptedTurn>);
    pub fn set_fallback(&self, turn: ScriptedTurn);
    pub fn remaining(&self) -> usize;
    pub fn call_count(&self) -> usize;
    pub fn calls(&self) -> Vec<ModelCall>;
    pub fn last_call(&self) -> Option<ModelCall>;
    pub async fn wait_for_calls(&self, count: usize);
    pub async fn wait_for_calls_timeout(&self, count: usize, timeout: Duration) -> Result<()>;
}
impl Model for ScriptedModel;                                           // provider/model_name echo the builder values
impl Default for ScriptedModel;                                         // empty script, fallback text

pub struct ScriptedModelBuilder;                                        // Default
impl ScriptedModelBuilder {
    pub fn provider(self, provider: impl Into<String>) -> Self;         // default "scripted"
    pub fn model_name(self, model_name: impl Into<String>) -> Self;     // default "scripted-model"
    pub fn respond(self, response: ChatResponse) -> Self;
    pub fn respond_text(self, content: impl Into<String>) -> Self;
    pub fn completion(self, completion: MockCompletion) -> Self;
    pub fn chunks(self, chunks: Vec<StreamChunk>) -> Self;
    pub fn stream_text(self, content: impl Into<String>) -> Self;
    pub fn fail(self, error: ScriptedError) -> Self;
    pub fn fallback(self, turn: ScriptedTurn) -> Self;
    pub fn build(self) -> ScriptedModel;
}

pub enum ScriptedTurn { Response(ChatResponse), Chunks(Vec<StreamChunk>), Fail(ScriptedError) }  // Clone
impl ScriptedTurn {
    pub fn text(content: impl Into<String>) -> Self;
    pub fn chunks(chunks: Vec<StreamChunk>) -> Self;
    pub fn fail(message: impl Into<String>) -> Self;                    // Error::Llm
    pub fn fail_with(error: ScriptedError) -> Self;
}

pub struct ScriptedError { pub kind: ErrorKind, pub message: String }   // Clone
impl ScriptedError {
    pub fn llm(message: impl Into<String>) -> Self;
    pub fn config(message: impl Into<String>) -> Self;
    pub fn validation(message: impl Into<String>) -> Self;
    pub fn transport(message: impl Into<String>) -> Self;
    pub fn timeout(message: impl Into<String>) -> Self;
    pub fn with_kind(kind: ErrorKind, message: impl Into<String>) -> Self;
    pub fn into_error(self) -> Error;                                   // Cancelled ignores the message
}

pub enum CallMode { Invoke, Stream }
pub struct ModelCall { pub mode: CallMode, pub messages: Vec<BaseMessage>, pub request: ModelRequestConfig }
impl ModelCall { pub fn last_user_text(&self) -> Option<String>; }
```

Behavior: `invoke` pops one turn (a `Chunks` turn is folded with `StreamAccumulator` into a `ChatResponse` whose id is `scripted-{n}` and model the configured name); `stream` pops one turn (a `Response` turn becomes `to_chunks()`) and yields `Err` as the first item for a `Fail` turn.
Calls are recorded at call time (before the stream is polled), matching Python's `call_history`/`call_count`.
Empty id/model on a popped `ChatResponse` are filled from the configured model name and a `scripted-{n}` id; the fallback turn is reused without consuming the queue (default `"Default mock response"`).

### Fixtures — `src/fixtures/`

```rust
pub struct TempWorkspace;                                               // plain temp root, no env mutation
impl TempWorkspace {
    pub fn new() -> io::Result<Self>;                                   // root + `home/`, `workspace/`, `logs/`
    pub fn root(&self) -> &Path;
    pub fn path(&self, rel: impl AsRef<Path>) -> PathBuf;               // never creates
    pub fn home(&self) -> PathBuf;                                      // creates on call
    pub fn workspace(&self) -> PathBuf;
    pub fn logs(&self) -> PathBuf;
    pub fn create_dir(&self, rel: impl AsRef<Path>) -> io::Result<PathBuf>;
    pub fn write(&self, rel: impl AsRef<Path>, contents: impl AsRef<[u8]>) -> io::Result<PathBuf>;
    pub fn read(&self, rel: impl AsRef<Path>) -> io::Result<String>;
    pub fn write_json(&self, rel: impl AsRef<Path>, value: &Value) -> Result<PathBuf>;   // Error::sys_op / Error::validation
    pub fn read_json(&self, rel: impl AsRef<Path>) -> Result<Value>;
    pub fn exists(&self, rel: impl AsRef<Path>) -> bool;
    pub fn files(&self) -> Vec<PathBuf>;                                // relative, sorted, excludes directories
}

pub struct EnvGuard;                                                    // process-global lock + save/clear/restore
impl EnvGuard {
    pub fn new(names: &[&'static str]) -> Self;                         // clears every name, restores on Drop
    pub fn hold(names: &'static [&'static str]) -> Self;                // same, explicit static slice (convenience)
    pub fn set(&self, name: &str, value: impl AsRef<OsStr>);
    pub fn remove(&self, name: &str);
    pub fn get(&self, name: &str) -> Option<String>;
}
pub const ISOLATED_ENV_VARS: &[&str];                                   // OPENJIUWEN_HOME, OPENJIUWEN_API_KEY, API_KEY, API_BASE, MODEL_PROVIDER, MODEL_NAME, OPENAI_API_KEY, OPENAI_BASE_URL, SSRF_PROTECT_ENABLED, OPENJIUWEN_LOG_LEVEL, OPENJIUWEN_LOG_FORMAT

pub struct TempHomeGuard;                                               // temp root + OPENJIUWEN_HOME isolation
impl TempHomeGuard {
    pub fn new() -> io::Result<Self>;                                   // creates `<tmp>/home`, points OPENJIUWEN_HOME at it
    pub fn isolated() -> io::Result<Self>;                              // also clears ISOLATED_ENV_VARS (model env) first
    pub fn root(&self) -> &Path;
    pub fn home(&self) -> &Path;
    pub fn set_var(&self, name: &str, value: impl AsRef<OsStr>);
}
```

### JSON assertions — `src/assert_json.rs`

Test-oriented: panics with a readable diff (allowed by the crate constraint for assertion helpers).

```rust
pub fn assert_json_eq(actual: &Value, expected: &Value);
pub fn assert_json_subset(actual: &Value, expected: &Value);            // objects: expected keys must match; arrays: same length, element-wise subset
pub fn assert_json_pointer_eq(actual: &Value, pointer: &str, expected: &Value);
pub fn assert_json_roundtrip<T: Serialize + DeserializeOwned + PartialEq + Debug>(value: &T);
pub fn json_at<'a>(value: &'a Value, pointer: &str) -> Option<&'a Value>;
pub fn json_pretty(value: &Value) -> String;
pub fn first_difference(actual: &Value, expected: &Value) -> Option<String>;  // JSON-pointer-ish path of the first mismatch
```

### Async helpers — `src/eventually.rs`

```rust
pub const DEFAULT_EVENTUALLY_TIMEOUT: Duration = Duration::from_secs(2);
pub const DEFAULT_POLL_INTERVAL: Duration = Duration::from_millis(10);
pub async fn eventually<F, Fut>(timeout: Duration, predicate: F) -> Result<()>;
pub async fn eventually_with_interval<F, Fut>(interval: Duration, timeout: Duration, predicate: F) -> Result<()>;
pub async fn eventually_some<F, Fut, T>(timeout: Duration, predicate: F) -> Result<T>;
pub async fn eventually_with_default_timeout<F, Fut>(predicate: F) -> Result<()>;
```

Each polls the (possibly async) predicate at least once, sleeps `interval` between attempts, and returns `Error::timeout("eventually: predicate did not become true within {timeout:?}")` at the deadline.

## Usage

```rust
use openjiuwen_test_support::{MockCompletion, MockLlmServer, ScriptedResponse};

let server = MockLlmServer::with_script([ScriptedResponse::text("hello")]).await;
let client = server.openai_client("gpt-4o").expect("client");
let response = client
    .invoke(&[BaseMessage::user("hi")], &server.request_config("gpt-4o"))
    .await
    .expect("mock server answers");
assert_eq!(response.message().expect("choice").text(), "hello");
server.wait_for_requests(1).await;
assert_eq!(server.requests()[0].path, "/v1/chat/completions");
server.shutdown().await;
```

```rust
use openjiuwen_test_support::{ScriptedModel, TempHomeGuard};

let home = TempHomeGuard::isolated().expect("temp home");
let model = ScriptedModel::builder()
    .respond_text("first")
    .chunks(openjiuwen_test_support::text_chunks("second"))
    .fail(openjiuwen_test_support::ScriptedError::transport("boom"))
    .build();
```

## Constraints

- Dev-only (`publish = false`); a library crate must never depend on this crate.
- No `unsafe` (workspace lint denies it); the server binds `127.0.0.1:0` only and the port is OS-assigned.
- No hidden global mutable state: the only process-global lock is the environment lock held by `EnvGuard`/`TempHomeGuard`.
- `unwrap`/`expect`/`panic` are forbidden in library code except in assertion helpers (`assert_json_*`) and documented test-oriented entry points (`MockLlmServer::start`, `ScriptedModel` infallible constructors).
- Errors use `openjiuwen_common::error::{Error, Result}`; `io::Result` only for raw filesystem operations on `TempWorkspace`.
- Files stay well under 1000 lines; each module owns one concern and re-exports through `lib.rs`.
- Do not hold a `std::sync::MutexGuard` across `.await`; recordings and script queues are mutated in short synchronous sections.
- No `println!`/`eprintln!`; no `tracing` requirement (test infrastructure stays silent by default).
- Every helper must be deterministic: fixed timestamps (`created: 0`), counter-based ids, no randomness, no wall-clock reads, no network beyond loopback.

## Design Decisions

- One loopback HTTP server instead of `httpx.MockTransport`: the Rust client owns retries, SSE decode, and error mapping, so only a real socket proves the wire format end to end.
- The script queue is mode-agnostic (one entry per request, regardless of `stream`): scripted turns are stated once as a neutral `MockCompletion` and projected to JSON or SSE, which avoids keeping two scripts in sync.
- Python parity kept where observable: default fallback text `"Default mock response"`, ordered queue with no prompt matching, `call_count`/`call_history` recorded at call time, and one assistant chunk per streamed turn unless the test scripts explicit chunks.
- `ScriptedModel` and `MockLlmServer` cover different layers: `ScriptedModel` fakes the `Model` trait (agent/workflow tests), `MockLlmServer` fakes the provider (client/transport/e2e tests).
- `MockLlmServer::builder()` exposes only what tests need; the server has no auth enforcement (any `Authorization` header is accepted and recorded).
- `wiremock` is re-exported rather than wrapped: wrapping it would duplicate a mature mock server and couple us to its version.
- Session factory helpers are deliberately omitted: `openjiuwen-session` is still a stub with no `Session` implementation, so there is nothing to construct.
  When it lands, add `src/fixtures/session.rs` with `workflow_session()` / `agent_session()` helpers mirroring Python `create_agent_session`, and re-export them from `fixtures` and `lib.rs`.
- `EnvGuard` is reentrant on one thread (nesting allowed) but serializes across threads/parallel tests through one process-global lock; guards are `!Send` so they stay on the test's runtime thread.

## Routing Table

- `./src/mock_llm/` → loopback OpenAI-compatible server: routing, request recording, script queue, wire (JSON/SSE) projection (`src/mock_llm/CONTEXT.md`)
- `./src/fixtures/` → temp workspace, `EnvironmentGuard`, `TempHomeGuard` (`src/fixtures/CONTEXT.md`)
- `./src/responses.rs` → `MockCompletion`/`MockToolCall` and `ChatResponse` ↔ `StreamChunk` conversions shared by both fakes
- `./src/scripted_model.rs` → `ScriptedModel`, script turns, call recording, `Model` implementation
- `./src/assert_json.rs` → JSON equality/subset/round-trip assertions and diff paths
- `./src/eventually.rs` → async eventual-consistency polling helpers
- `./tests/` → crate-level integration tests driving the fakes through `OpenAiCompatibleClient` and `Model`
- `./Cargo.toml` → dev-only manifest (`publish = false`), no external services, no extra features

## Test Strategy

- Unit tests beside each module: conversion round-trips (`MockCompletion` ↔ `ChatResponse` ↔ chunks), wire frames (JSON body, SSE frame sequence, tool-call deltas, usage frame, `[DONE]`), queue/fallback behavior, deterministic ids, recordings, scripted-model invoke/stream/failure, env guard save/restore, workspace paths, JSON assertions (matching and failing cases), `eventually` success and timeout (`tokio::time::pause` where useful).
- `tests/mock_server.rs`: end-to-end through `OpenAiCompatibleClient` (non-streaming, streaming, tool calls, reasoning, error status, queue exhaustion, request recording, `wait_for_requests`, shutdown).
- `tests/scripted_model.rs`: trait-level behavior through `Arc<dyn Model>` and `ModelExt::complete`.
- `tests/fixtures.rs`: `TempWorkspace` + `TempHomeGuard` isolation and cleanup.
- All tests are offline, deterministic, and bind only loopback; timeouts use short fixed durations (≤2 s) so a regression fails fast instead of hanging.
- Ported tests cite the Python source they mirror in a comment (for example `tests/unit_tests/fixtures/mock_llm.py`).
