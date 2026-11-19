# a2a/ — Agent2Agent client and server (`feature = "a2a"`)

## Intent

Port of `openjiuwen/extensions/a2a/` (v0.1.20): talk to remote A2A agents and expose this runtime as an A2A agent.
The Python package is a thin adapter over the `a2a-sdk`; this crate owns the wire layer itself (JSON-RPC 2.0 over HTTP, SSE for streaming), so the method names, card JSON, status values, and roles must match what `a2a-sdk` 1.0 emits.
Wire parity matters more than Rust ergonomics here: these types cross process boundaries and are also spoken to non-Rust agents.

## API Surface

```rust
// types.rs — wire models (serde camelCase, unknown fields tolerated, None omitted)
pub struct AgentCard { name, description, version, supported_interfaces: Vec<AgentInterface>,
                       capabilities: AgentCapabilities, default_input_modes: Vec<String>,
                       default_output_modes: Vec<String>, skills: Vec<AgentSkill>,
                       provider: Option<AgentProvider>,
                       security_schemes: serde_json::Map<String, Value>,
                       security: Vec<BTreeMap<String, Vec<String>>> }
pub struct AgentInterface { url, protocol_binding, protocol_version, tenant: Option<String> }
pub struct AgentCapabilities { streaming: bool, push_notifications: bool }
pub struct AgentSkill { id, name, description, tags, examples, input_modes, output_modes }
pub struct AgentProvider { organization, url }
pub struct Message { message_id, context_id: Option<String>, task_id: Option<String>,
                     role: Role, parts: Vec<Part>, metadata: Map<String, Value>,
                     extensions, reference_task_ids }
pub enum Role { Unspecified, User, Agent }             // wire ROLE_USER / ROLE_AGENT / ROLE_UNSPECIFIED
pub struct Part { text: Option<String>, raw: Option<String>, url: Option<String>,
                  data: Option<Value>, filename, media_type, metadata }   // Part::text(..)
pub struct Task { id, context_id: String, status: TaskStatus, artifacts: Vec<Artifact>, history: Vec<Message>, metadata }
pub struct TaskStatus { state: TaskState, message: Option<Message>, timestamp: Option<String> }
pub enum TaskState { Unspecified, Submitted, Working, Completed, Failed, Canceled, InputRequired, Rejected, AuthRequired }
pub struct Artifact { artifact_id: String, name: Option<String>, description: Option<String>, parts, metadata, extensions }
pub struct TaskStatusUpdateEvent { task_id, context_id, status, metadata }
pub struct TaskArtifactUpdateEvent { task_id, context_id, artifact, append: bool, last_chunk: bool, metadata }
pub struct SendMessageRequest { tenant: Option<String>, message: Message, configuration: Option<Value>, metadata }
pub struct CancelTaskRequest { tenant: Option<String>, id: String, metadata }

// jsonrpc.rs
pub struct JsonRpcRequest { jsonrpc, id: Value, method: String, params: Option<Value> }
pub struct JsonRpcResponse { jsonrpc, id: Value, result: Option<Value>, error: Option<JsonRpcError> }
pub struct JsonRpcError { code: i32, message: String, data: Option<Value> }   // JsonRpcError::into_error()
pub mod error_code { PARSE=-32700; INVALID_REQUEST=-32600; METHOD_NOT_FOUND=-32601; INVALID_PARAMS=-32602;
                     INTERNAL=-32603; TASK_NOT_FOUND=-32001; TASK_NOT_CANCELABLE=-32002; PUSH_NOT_SUPPORTED=-32003;
                     UNSUPPORTED_OPERATION=-32004; CONTENT_TYPE_NOT_SUPPORTED=-32005; INVALID_AGENT_RESPONSE=-32006;
                     EXTENDED_CARD_NOT_CONFIGURED=-32007; EXTENSION_SUPPORT_REQUIRED=-32008; VERSION_NOT_SUPPORTED=-32009 }
pub mod method { SEND_MESSAGE="SendMessage"; SEND_STREAMING_MESSAGE="SendStreamingMessage"; GET_TASK="GetTask";
                 CANCEL_TASK="CancelTask"; SUBSCRIBE_TO_TASK="SubscribeToTask"; V0_3_ALIASES;
                 pub fn canonical(&str) -> Option<&'static str>; pub fn is_streaming(&str) -> bool }
pub fn parse_request(&[u8]) -> std::result::Result<JsonRpcRequest, JsonRpcError>;
pub fn serialize(&JsonRpcResponse) -> String;

// transform.rs — openjiuwen-side result model + request/response mapping
pub struct AgentResult { pub task_id: Option<String>, pub session_id: Option<String>,
                         pub status: AgentStatus, pub artifacts: Vec<AgentArtifact>,
                         pub metadata: Map<String, Value> }        // JSON: task_id + sessionId
pub enum AgentStatus { Submitted, Working, Paused, InputRequired, Completed, Canceled, Failed, Waiting, Unknown }
pub struct AgentArtifact { artifact_id: Option<String>, name, description, parts: Vec<AgentPart>, metadata }
pub struct AgentPart { text, raw, url, data: Option<Value>, filename, media_type, metadata }
pub struct A2aTransformer;
impl A2aTransformer {
    pub fn to_request(inputs: &Value) -> Result<SendMessageRequest>;
    pub fn from_request(params: &Value) -> Result<Value>;
    pub fn from_response(result: &Value) -> AgentResult;          // infallible
}
pub fn map_task_state(state: TaskState) -> AgentStatus;
pub fn merge_results(base: AgentResult, next: AgentResult) -> AgentResult;   // client streaming fold
pub fn resolve_session_id(inputs: &Value) -> Option<String>;                // conversation_id then sessionId
pub fn with_session_id(result: AgentResult, session_id: Option<&str>) -> AgentResult;
pub fn to_a2a_part(part: &AgentPart) -> Part;
pub fn from_a2a_part(part: &Part) -> AgentPart;
pub fn from_a2a_artifact(artifact: &Artifact) -> AgentArtifact;
pub fn to_a2a_artifact(artifact: &AgentArtifact, artifact_id: Option<&str>) -> Artifact;

// card.rs
pub struct AgentCardAdapter;
impl AgentCardAdapter {
    pub const DEFAULT_INPUT_MODES: [&str; 2] = ["text/plain", "application/json"];
    pub const DEFAULT_OUTPUT_MODES: [&str; 2] = ["text/plain", "application/json"];
    pub fn to_a2a(card: &openjiuwen_core::single_agent::AgentCard, interface_url: Option<&str>,
                  protocol_binding: &str, protocol_version: &str) -> AgentCard;
    pub fn from_a2a(card: &AgentCard) -> openjiuwen_core::single_agent::AgentCard;
}
pub fn build_description(base_description: &str, input_params: Option<&Value>, output_params: Option<&Value>) -> String;
pub fn serialize_param_payload(value: &Value) -> String;    // json.dumps sort_keys=True, ensure_ascii=False

// client.rs
pub struct A2aClientConfig { pub rpc_url: String, pub timeout: Option<Duration>, pub headers: BTreeMap<String, String> }
// A2aClientConfig: Default (empty rpc_url, timeout None = 60 s, no headers) + builders
//   new(rpc_url: impl Into<String>) -> Self, with_timeout(Duration) -> Self, with_header(name, value) -> Self
// rpc_url is the FULL JSON-RPC endpoint, POSTed verbatim (no base-URL joining by the client)
pub struct A2aClient { /* reqwest::Client + request-id counter */ }
impl A2aClient {
    pub fn new(config: A2aClientConfig) -> Result<Self>;       // empty rpc_url -> Error::Config
    pub fn config(&self) -> &A2aClientConfig;
    pub async fn invoke(&self, inputs: Value) -> Result<AgentResult>;
    pub fn stream(&self, inputs: Value) -> BoxStream<'static, Result<AgentResult>>;
    pub async fn cancel_task(&self, task_id: &str, tenant: Option<&str>) -> Result<AgentResult>;
    pub async fn stop(&self);                                   // documented no-op (Python parity)
}
pub struct RemoteClientConfig { pub id, version, name, description, protocol: String,
                                pub card: Option<AgentCard>, pub url: String,
                                pub timeout: Option<Duration>, pub kwargs: Map<String, Value> }
// RemoteClientConfig: Default sets protocol "A2A"; new(id, url) sets both and defaults the rest
pub struct A2aRemoteClient { .. }
impl A2aRemoteClient {
    pub fn new(config: RemoteClientConfig) -> Result<Self>;     // protocol "A2A" + card required -> Validation
    pub fn config(&self) -> &RemoteClientConfig; pub fn card(&self) -> &AgentCard; pub fn rpc_url(&self) -> &str;
    pub async fn start(&self) -> Result<()>;                    // lazy, idempotent
    pub fn is_started(&self) -> bool;
    pub async fn stop(&self);
    pub async fn invoke(&self, inputs: Value, timeout: Option<Duration>) -> Result<AgentResult>;
    pub fn stream(&self, inputs: Value, timeout: Option<Duration>) -> BoxStream<'static, Result<AgentResult>>;
    pub async fn cancel_task(&self, task_id: &str, tenant: Option<&str>) -> Result<AgentResult>;
}

// server/ (everything re-exported at crate::a2a::server::)
pub enum A2aTransport { JsonRpc, Rest, Both }   // serde "JSONRPC" / "HTTP+JSON" / "BOTH" + aliases
impl A2aTransport { pub const fn mounts_json_rpc(self) -> bool; pub const fn mounts_rest(self) -> bool; }
pub struct A2aServerConfig { pub agent_card: AgentCard, pub interface_url: Option<String>,
                             pub rpc_path /* "/a2a/jsonrpc/" */, pub card_path /* "/.well-known/agent-card.json" */,
                             pub transport /* JsonRpc */, pub rest_path /* "/a2a/rest" */,
                             pub rest_prefix /* "/v1" */, pub rest_interface_url: Option<String>,
                             pub protocol_version /* "1.0" */, pub host /* "127.0.0.1" */, pub port /* 8000 */,
                             pub max_tasks /* 10_000; 0 = unbounded */, pub terminal_task_ttl_ms /* 1 h; 0 = forever */ }
// A2aServerConfig: Serialize/Deserialize with #[serde(default)] ({} -> defaults, unknown fields tolerated),
// Default + new(card), with_interface_url, with_bind, with_rpc_path, with_transport, with_rest_path,
// with_rest_prefix, with_rest_interface_url, with_max_tasks, with_terminal_task_ttl_ms,
// rest_base_path() -> String, retention_policy() -> RetentionPolicy
pub const DEFAULT_RPC_PATH/DEFAULT_CARD_PATH/DEFAULT_REST_PATH/DEFAULT_REST_PREFIX/DEFAULT_PROTOCOL_VERSION/DEFAULT_HOST/DEFAULT_PORT/DEFAULT_MAX_TASKS/DEFAULT_TERMINAL_TASK_TTL_MS;
#[async_trait] pub trait A2aExecutor: Send + Sync {
    async fn execute(&self, ctx: A2aExecutionContext, sink: &EventSink) -> Result<()>;
    async fn cancel(&self, ctx: A2aExecutionContext) -> Result<()>;   // default: Ok(()), server publishes CANCELED
}
pub struct A2aExecutionContext { pub task_id: String, pub context_id: String,
                                 pub message: Message, pub metadata: Map<String, Value> }
pub struct EventSink { .. }   // task(), working(), artifact(Artifact, last_chunk), complete(), fail(msg),
                              // require_input(msg?), canceled(), task_id(), context_id(), is_terminal(),
                              // subscriber_overflowed()
pub trait TaskStore: Send + Sync { fn get(&self, id: &str) -> Option<Task>; fn put(&self, task: Task); fn all(&self) -> Vec<Task>; }
pub struct RetentionPolicy { pub max_tasks: usize, pub terminal_ttl: Option<Duration> }
pub struct InMemoryTaskStore { .. }   // new() (default policy), with_policy(RetentionPolicy), policy()
// retention: expired terminal tasks dropped; over the cap, terminal tasks evicted oldest-first;
// running/awaited (INPUT_REQUIRED) tasks are never evicted, so the cap is a soft bound for in-flight work
// streaming: EVENT_BUFFER_CAPACITY = 256 (bounded, try_send, slow subscriber dropped with a terminal error frame)
pub struct A2aServer { .. }   // Clone
impl A2aServer {
    pub fn new(config: A2aServerConfig, executor: Arc<dyn A2aExecutor>) -> Result<Self>;
    pub fn with_store(config: A2aServerConfig, executor: Arc<dyn A2aExecutor>, store: Arc<dyn TaskStore>) -> Result<Self>;
    pub fn config(&self) -> &A2aServerConfig; pub fn store(&self) -> Arc<dyn TaskStore>;
    pub fn agent_card(&self) -> &AgentCard;                      // derived for the mounted transport
    pub fn router(&self) -> axum::Router;                       // card route + every mounted binding
    pub async fn serve(self, shutdown: CancellationToken) -> Result<()>;
    pub async fn serve_in_background(self, shutdown: CancellationToken) -> Result<ServerHandle>; // binds 127.0.0.1:0 when port == 0
}
pub struct ServerHandle { .. }  // local_addr(), shutdown() (non-consuming), join().await
pub struct A2aServerAdapter { .. }  // new(server), async start()/stop(), active(), local_addr()

// harness_bridge.rs — A2A ⇄ openjiuwen-harness-protocol (no Python counterpart)
pub fn message_to_harness_input(message: &Message) -> HarnessInput;
pub fn harness_input_to_message(input: &HarnessInput, message_id: &str) -> Message;
pub fn turn_result_to_agent_result(result: &TurnResult) -> AgentResult;
pub fn agent_result_to_task(result: &AgentResult, task_id: Option<&str>, context_id: &str) -> Task;
pub fn part_to_content_block(part: &Part, block_id: &str) -> ContentBlock;
pub fn content_block_to_part(block: &ContentBlock) -> Part;
pub fn turn_message_to_artifact(message: &TurnMessage) -> AgentArtifact;
pub fn agent_status_to_task_state(status: AgentStatus) -> TaskState;
pub fn message_role_name(role: MessageRole) -> &'static str;
```

## Ported Behavior (Python → Rust)

- `to_request` (`A2ATransformer.to_a2a_request`): fresh 32-hex `message_id`; `role = ROLE_USER`; `context_id` from `conversation_id` then `sessionId`; `query` becomes a single `Part{text}`; every other input key with a non-null value is copied into `message.metadata` (`query`, `sessionId`, `conversation_id` excluded).
- `from_request` (`from_a2a_request`): accepts a `SendMessageRequest`-shaped params object, a bare `Message`, or a plain payload dict; produced payload has `query` (first non-empty text part), `task_id` (when set), `sessionId` (from `context_id` when set); metadata merges only for keys the payload does not already have, and `None` values are skipped.
- `from_response` (`from_a2a_response`) precedence on the JSON-RPC `result`: artifact-update → `WORKING` with that artifact; status-update → mapped state, no artifacts; `Message` → `COMPLETED` with one artifact `message` holding every part; `Task` → its state and artifacts; anything else → `COMPLETED` with no artifacts. Wrapper keys `artifact_update`/`status_update` (proto names) are unwrapped too. `TASK_STATE_REJECTED → FAILED`; `TASK_STATE_AUTH_REQUIRED → INPUT_REQUIRED`; unknown/unspecified → `UNKNOWN`.
- Client methods use the PascalCase v1 names; the server accepts both PascalCase and the v0.3 lowercase aliases (`method::canonical`, trailing slash and surrounding whitespace tolerated).
- Streaming: one SSE `data:` frame per JSON-RPC response; `A2aClient::stream` yields one `AgentResult` per frame; `invoke` folds frames with `merge_results` (concatenate artifacts, override status when the newer one is set/known, fall back on `task_id`/`session_id`, dict-merge metadata). Frames are parsed incrementally over `bytes_stream()` by a pure `SseFramer` (comments/`event:`/`id:`/`retry:` ignored, multi-line `data:` joined, CR/LF delimiters and split CRLF handled).
- Server event order: the server emits the Python executor prelude itself — `Task{state=SUBMITTED, history=[message]}` first, then `WORKING` with "Processing your request..." — before calling `A2aExecutor::execute`; the sink then publishes artifacts (each with `last_chunk` on terminal results), closes open artifacts with `append=true,last_chunk=true` (ascending id order), and emits the terminal state (`FAILED`/`CANCELED`/`INPUT_REQUIRED`, otherwise `COMPLETED`). Every emission also updates the `TaskStore`, so `GetTask`/`CancelTask` see coherent state. A non-terminating executor is completed by the server; a failing one emits `FAILED` and is returned as a JSON-RPC `-32603` error for non-streaming calls.
- JSON-RPC dispatch: `SendMessage`/`SendStreamingMessage` (v0.3 aliases accepted), `GetTask` (`params.id`, also `taskId`), `CancelTask` (unknown → `-32001`, already terminal → `-32002`, best-effort `A2aExecutor::cancel` then `CANCELED`), `SubscribeToTask` (answers with the stored task snapshot plus its current status, then ends — not a live subscription), anything else → `-32601`. All non-streaming responses are HTTP 200 envelopes; streaming responses are `text/event-stream` with one JSON-RPC response per `data:` frame.
- REST (`HTTP+JSON`, `A2aTransport::Rest`/`Both`): mounted at `rest_base_path()` (default `/a2a/rest/v1`), serving `GET {base}/card`, `POST {base}/message:send` and `{base}/messages:send`, `POST {base}/message:stream`, `GET {base}/tasks/{id}`, `POST {base}/tasks/{id}[/:]cancel`, and `GET|POST {base}/tasks/{id}[/:]subscribe`; routing is exclusive with JSON-RPC, as in Python's `a2a_server.py`.
- REST requests carry the same params object as the JSON-RPC binding (both call `plan_send`); non-streaming replies are the stored `Task` JSON, streams carry bare `data: <json>` frames (no JSON-RPC envelope), and errors use `{"error": {"code", "status", "message"}}` with the HTTP status mapped from the A2A code (400 `INVALID_ARGUMENT`, 405 `METHOD_NOT_ALLOWED`, 404 `NOT_FOUND`, 409 `FAILED_PRECONDITION`, 501 `UNIMPLEMENTED`, 415 `UNSUPPORTED_MEDIA_TYPE`, 500 `INTERNAL`).
- Served card derivation: `A2aTransport::JsonRpc` serves the configured card unchanged; `Rest` advertises exactly one `HTTP+JSON` interface first (configured transport interfaces removed); `Both` advertises exactly one `HTTP+JSON` and one `JSONRPC` interface, followed by every non-transport interface; derived URLs use `rest_interface_url`/`interface_url`, else `http://{host}:{port}{path}`.
- Agent card: `description` = base description, then `[input_params] {json}` and `[output_params] {json}` sections (`json.dumps(..., sort_keys=True, ensure_ascii=False)` — keys sorted recursively, `": "`/`", "` separators, non-ASCII preserved); default input/output modes are `text/plain` and `application/json`; capabilities `streaming=true`, `push_notifications=false`; `version` deliberately left empty and no skills/provider, exactly like the Python adapter; `from_a2a` copies only `name` and `description`.

## Documented Deviations from the Python SDK

- `Part.data` crosses the boundary as a structured JSON `Value`; Python's `_a2a_part_to_part` stringifies it with `str()`, so rich payloads do not round-trip there.
- `AgentResult.status` defaults to `AgentStatus::Unknown` (Python's field defaults to `None`); `merge_results` treats `Unknown` as "unset", matching Python's `(None, UNKNOWN)` check.
- `from_a2a` and the card adapter are lossy by design (Python parity): no interfaces/skills on the kernel card, empty `version` on the A2A card.
- `A2aServer::serve_in_background` is `async` (the socket must be bound before `local_addr()` is known); `A2aServerAdapter::start/stop` are `async` instead of Python's thread launcher.
- `A2aExecutor::cancel`'s default body returns `Ok(())`; the server publishes `CANCELED` itself (same observable behavior as Python's "default: emit Canceled").
- `A2aClient::stop` is a documented no-op (reqwest owns no persistent connection to close); it exists for Python call-sequence parity.
- `harness_bridge` has no Python counterpart; its mapping tables (part kinds `text`/`data`/`file`/`binary`/`empty`, part conversion also normalizing empty strings via `transform::from_a2a_part`, `TurnStatus::Interrupted → InputRequired`, `AgentStatus → TaskState`) are defined in the module docs.
- `security` on `AgentCard` is `Vec<BTreeMap<String, Vec<String>>>` because `serde_json::Map<String, Vec<String>>` has no serde impls; the JSON is identical.
- REST deviations (documented in `server/rest.rs`): Starlette answers a known path with the wrong method with a plain-text `405`, this server emits the JSON error envelope; the `a2a-sdk` maps `UnsupportedOperation` to HTTP 400, this server uses 501 `UNIMPLEMENTED`; SSE frames end with `\n\n` (`sse_starlette` uses `\r\n\r\n`); `POST {base}/messages:send` (plural) is accepted as an extra alias for `message:send`.
- The colon action spellings (`/tasks/{id}:cancel`) are served by one dispatcher route because `matchit` 0.8 cannot express a dynamic segment followed by a literal suffix; the dispatcher splits the segment on its last `:` (task ids are 32-hex UUIDs, so no ambiguity).

## Constraints

- No `unsafe`; no blocking I/O; `reqwest` with rustls; SSE parsing is incremental over `bytes_stream` (no whole-body buffering).
- Timeouts: `A2aClientConfig.timeout` (default 60 s) bounds a unary call as a whole and bounds connecting on both paths (`reqwest::ClientBuilder::connect_timeout`); on the streaming path it is a per-chunk *idle* deadline instead, so a healthy long stream outlives the budget while a silent peer still fails — every expiry and every body error reported as a timeout maps onto `Error::Timeout`. `A2aRemoteClient::invoke/stream` additionally wrap the call in the caller timeout and stop the client on timeout/error exactly like Python (`stop()` before re-raising; `Cancelled` propagates without calling `stop`); a remote `stream` deadline is applied per chunk against a single total budget.
- gRPC is **not** implemented: a gRPC `protocol_binding` yields `Error::Unsupported("gRPC transport is not supported")`; `JSONRPC` and `HTTP+JSON` are both served, selected by `A2aServerConfig::transport`.
- Endpoint derivation is private (`resolve_rpc_url`, `derive_interface_url`): no public base-URL→JSON-RPC-URL helper is exported, so a plain `A2aClient` caller must pass the full endpoint in `rpc_url`; only `A2aRemoteClient::new` derives it (card JSON-RPC interface, else `trim_end_matches('/')` + `/a2a/jsonrpc/`).
- The JSON-RPC route is mounted at the literal `rpc_path` (default trailing slash), matching Python's normalization; the well-known card route stays mounted in every mode and serves `A2aServer::agent_card()`.
- Construction validation: `card_path`, `rpc_path`, and `rest_path` must start with `/`; a REST base must be non-empty and absolute and must not collide with `card_path`/`rpc_path` (violations are `Error::Config`, never a router panic).
- Errors map to `Error::Transport` (HTTP/SSE), `Error::Timeout`, `Error::Validation` (card/inputs/JSON-RPC params), `Error::NotFound` (JSON-RPC `TASK_NOT_FOUND`), `Error::Unsupported` (`METHOD_NOT_FOUND`/gRPC), `Error::Config` (client construction and invalid mount paths).
- Serde: wire structs are camelCase with unknown fields tolerated and unset `Option`s omitted; enums serialize proto strings (`ROLE_USER`, `TASK_STATE_WORKING`) and decode tolerantly (bare suffix, snake_case, camelCase, `cancelled`, unknown → `Unspecified`). Empty maps serialize as `{}`.
- Logging: `tracing` only (targets `openjiuwen_extensions::*`); payload contents never logged above `debug`, except a `message` member that does not decode: the server reports it at `warn` because the request is then answered from the synthesized message and the caller's parts are lost.
- Files ≤ ~1000 lines; the server lives in `./server/` with one file per concern.

## Test Strategy

- Unit tests live beside the code they own: `./types.rs`, `./jsonrpc.rs`, `./card.rs`, `./client/tests.rs`, `./transform/tests.rs`, `./harness_bridge/tests.rs`, `./server/tests.rs` (JSON-RPC routing/dispatch, event order, store, lifecycle, plus a compile-time check that every public path resolves at `crate::a2a::server::`), and `./server/rest_tests.rs` (config defaults/serde/`rest_base_path`, transport serde/builders, card derivation in all three modes, construction errors, and every REST route through `router()` + `ServiceExt::oneshot`).
- `./e2e_tests.rs` holds the in-crate integration suites as two in-module submodules (`client_suite` with `wiremock`, `server_suite` with `Router::oneshot` via `tower::ServiceExt` + `http_body_util::BodyExt` and one real ephemeral socket).
- `../tests/a2a_client.rs` and `../tests/a2a_server.rs` are the crate-boundary mirrors of those suites; they use only `openjiuwen_extensions::a2a::` public paths and are gated with `#![cfg(feature = "a2a")]`.
- The streaming budgets are proven against a real paced `axum` stream (`serve_paced_stream`), which a `wiremock` body cannot express: `a_slow_stream_outlives_the_request_budget` (three frames paced 200 ms apart under a 500 ms budget still arrive) and `a_stalled_stream_times_out_with_the_typed_error` (one frame, then silence → `ErrorKind::Timeout`).
- Coverage: send folds to `AgentResult`; SSE stream of three frames yields three results in order; `CancelTask` fields and `CANCELED`; JSON-RPC error → typed `Error`; non-2xx → `Transport`; remote-client protocol/card validation and card-interface invocation; card route; `message/send` + v0.3 alias; `tasks/get` unknown → `-32001`; `tasks/cancel` → `CANCELED` then `-32002`; `message/stream` frame order; executor error → `FAILED` + `-32603`; real-socket card/send round trip.
- Test gap: no suite pairs the two halves — `A2aClient` is only driven against `wiremock`, and the single real-socket test (`e2e_tests.rs`, `server_suite::server_answers_over_a_real_socket`) posts with raw `reqwest`; the only loopback wiring recipe is `A2aServerConfig::with_bind("127.0.0.1", 0)` + `serve_in_background` + `A2aClientConfig::new(format!("http://{}{}", handle.local_addr(), rpc_path))`.
- Python counterparts named in comments: `tests/unit_tests/extensions/a2a/test_a2a_transformer.py`, `test_a2a_agentcard_adapter.py`, `test_a2a_client.py`, `test_a2a_server.py`.
- Definition of done: `cargo fmt -p openjiuwen-extensions`, `cargo test -p openjiuwen-extensions --features a2a --lib`, and `cargo clippy -p openjiuwen-extensions --all-targets --features a2a -- -D warnings` all clean for this subtree.

## Routing Table

- `./types.rs` → wire models and their serde mapping
- `./jsonrpc.rs` → envelopes, error codes, method names/aliases, request parsing
- `./transform.rs` → openjiuwen `AgentResult` model, request/response mapping, result folding, part/artifact conversion
- `./transform/tests.rs` → transformer and mapping-table unit tests
- `./card.rs` → agent-card adaptation both ways, Python-style description embedding
- `./client.rs` → `A2aClient`, `A2aRemoteClient`, JSON-RPC POST, incremental SSE framing
- `./client/tests.rs` → client-config, endpoint-derivation, SSE-framer, and wiremock client tests
- `./server/` → axum router for both bindings, executor SPI, event sink, task store, server handle, adapter (`server/CONTEXT.md`)
- `./server/ops.rs` → binding-neutral send/stream/snapshot/cancel operations shared by JSON-RPC and REST
- `./server/rest.rs` → REST (`HTTP+JSON`) routes, colon-action dispatcher, error envelope and status mapping
- `./server/sse.rs` → SSE framing for JSON-RPC envelopes and bare REST values
- `./harness_bridge.rs` → conversions to/from `openjiuwen_harness_protocol`
- `./harness_bridge/tests.rs` → bridge mapping-table and round-trip tests
- `./e2e_tests.rs` → end-to-end client/server suites (in-crate, public paths only; mirrored by `../tests/a2a_client.rs` and `../tests/a2a_server.rs`)
