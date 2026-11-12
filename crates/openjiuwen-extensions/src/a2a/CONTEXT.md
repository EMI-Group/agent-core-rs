# a2a/ — Agent2Agent client and server (`feature = "a2a"`)

## Intent

Port of `openjiuwen/extensions/a2a/` (v0.1.20): talk to remote A2A agents and expose this runtime as an A2A agent.
The Python package is a thin adapter over the `a2a-sdk`; this crate owns the wire layer itself (JSON-RPC 2.0 over HTTP, SSE for streaming), so the method names, card JSON, status values, and roles must match what `a2a-sdk` 1.0 emits.
Wire parity matters more than Rust ergonomics here: these types cross process boundaries and are also spoken to non-Rust agents.

## API Surface

```rust
// types.rs — wire models (serde camelCase, unknown fields tolerated)
pub struct AgentCard { name, description, version, supported_interfaces: Vec<AgentInterface>,
                       capabilities: AgentCapabilities, default_input_modes: Vec<String>,
                       default_output_modes: Vec<String>, skills: Vec<AgentSkill>,
                       provider: Option<AgentProvider>, security_schemes, security }
pub struct AgentInterface { url, protocol_binding, protocol_version, tenant: Option<String> }
pub struct AgentCapabilities { streaming: bool, push_notifications: bool }
pub struct AgentSkill { id, name, description, tags, examples, input_modes, output_modes }
pub struct Message { message_id, context_id: Option<String>, task_id: Option<String>,
                     role: Role, parts: Vec<Part>, metadata: Map<String, Value>,
                     extensions, reference_task_ids }
pub enum Role { Unspecified, User, Agent }             // wire ROLE_USER / ROLE_AGENT / ROLE_UNSPECIFIED
pub struct Part { text: Option<String>, raw: Option<String>, url: Option<String>,
                  data: Option<Value>, filename, media_type, metadata }
pub struct Task { id, context_id, status: TaskStatus, artifacts: Vec<Artifact>, history: Vec<Message>, metadata }
pub struct TaskStatus { state: TaskState, message: Option<Message>, timestamp: Option<String> }
pub enum TaskState { Unspecified, Submitted, Working, Completed, Failed, Canceled, InputRequired, Rejected, AuthRequired }
pub struct Artifact { artifact_id, name, description, parts, metadata, extensions }
pub struct TaskStatusUpdateEvent { task_id, context_id, status, metadata }
pub struct TaskArtifactUpdateEvent { task_id, context_id, artifact, append: bool, last_chunk: bool, metadata }
pub struct SendMessageRequest { tenant: Option<String>, message: Message, configuration: Option<Value>, metadata }
pub struct CancelTaskRequest { tenant: Option<String>, id: String, metadata }

// jsonrpc.rs
pub struct JsonRpcRequest { jsonrpc, id: Value, method: String, params: Option<Value> }
pub struct JsonRpcResponse { jsonrpc, id: Value, result: Option<Value>, error: Option<JsonRpcError> }
pub struct JsonRpcError { code: i32, message: String, data: Option<Value> }
pub mod error_code { pub const PARSE: i32 = -32700; INVALID_REQUEST=-32600; METHOD_NOT_FOUND=-32601;
                     INVALID_PARAMS=-32602; INTERNAL=-32603; TASK_NOT_FOUND=-32001; TASK_NOT_CANCELABLE=-32002;
                     PUSH_NOT_SUPPORTED=-32003; UNSUPPORTED_OPERATION=-32004; CONTENT_TYPE_NOT_SUPPORTED=-32005;
                     INVALID_AGENT_RESPONSE=-32006; EXTENDED_CARD_NOT_CONFIGURED=-32007; EXTENSION_SUPPORT_REQUIRED=-32008;
                     VERSION_NOT_SUPPORTED=-32009 }
pub mod method { // v1 PascalCase is canonical on the wire; v0.3 lowercase aliases are accepted on receive
  pub const SEND_MESSAGE: &str = "SendMessage";            // alias "message/send"
  pub const SEND_STREAMING_MESSAGE: &str = "SendStreamingMessage"; // alias "message/stream"
  pub const GET_TASK: &str = "GetTask";                    // alias "tasks/get"
  pub const CANCEL_TASK: &str = "CancelTask";              // alias "tasks/cancel"
  pub const SUBSCRIBE_TO_TASK: &str = "SubscribeToTask";   // alias "tasks/resubscribe"
  pub fn canonical(method: &str) -> Option<&'static str>;
}

// transform.rs — openjiuwen-side result model + request/response mapping
pub struct AgentResult { pub task_id: Option<String>, pub session_id: Option<String>,
                         pub status: AgentStatus, pub artifacts: Vec<AgentArtifact>,
                         pub metadata: Map<String, Value> }
pub enum AgentStatus { Submitted, Working, Paused, InputRequired, Completed, Canceled, Failed, Waiting, Unknown }
pub struct AgentArtifact { artifact_id, name, description, parts: Vec<AgentPart>, metadata }
pub struct AgentPart { text, raw, url, data, filename, media_type, metadata }
pub struct A2aTransformer;
impl A2aTransformer {
    pub fn to_request(inputs: &Value) -> Result<SendMessageRequest>;
    pub fn from_request(params: &Value) -> Result<Value>;
    pub fn from_response(result: &Value) -> AgentResult;
}
pub fn map_task_state(state: TaskState) -> AgentStatus;
pub fn merge_results(base: AgentResult, next: AgentResult) -> AgentResult;  // client streaming fold

// card.rs
pub struct AgentCardAdapter;
impl AgentCardAdapter {
    pub const DEFAULT_INPUT_MODES: [&'static str; 2] = ["text/plain", "application/json"];
    pub fn to_a2a(card: &openjiuwen_core::single_agent::AgentCard, interface_url: Option<&str>,
                  protocol_binding: &str, protocol_version: &str) -> AgentCard;
    pub fn from_a2a(card: &AgentCard) -> openjiuwen_core::single_agent::AgentCard;
}

// client.rs
pub struct A2aClientConfig { pub rpc_url: String, pub timeout: Option<Duration>, pub headers: BTreeMap<String, String> }
pub struct A2aClient { /* reqwest::Client */ }
impl A2aClient {
    pub fn new(config: A2aClientConfig) -> Result<Self>;
    pub async fn invoke(&self, inputs: Value) -> Result<AgentResult>;
    pub fn stream(&self, inputs: Value) -> BoxStream<'static, Result<AgentResult>>;
    pub async fn cancel_task(&self, task_id: &str, tenant: Option<&str>) -> Result<AgentResult>;
    pub async fn stop(&self);
}
pub struct RemoteClientConfig { pub id, version, name, description: String, pub protocol: String,
                                pub card: Option<AgentCard>, pub url: String,
                                pub timeout: Option<Duration>, pub kwargs: Map<String, Value> }
pub struct A2aRemoteClient { .. }
impl A2aRemoteClient {
    pub fn new(config: RemoteClientConfig) -> Result<Self>;   // Error::Validation when protocol != "A2A" or card missing
    pub async fn start(&self) -> Result<()>;
    pub fn is_started(&self) -> bool;
    pub async fn stop(&self);
    pub async fn invoke(&self, inputs: Value, timeout: Option<Duration>) -> Result<AgentResult>;
    pub fn stream(&self, inputs: Value, timeout: Option<Duration>) -> BoxStream<'static, Result<AgentResult>>;
    pub async fn cancel_task(&self, task_id: &str, tenant: Option<&str>) -> Result<AgentResult>;
}

// server.rs
pub struct A2aServerConfig { pub agent_card: AgentCard, pub interface_url: Option<String>,
                             pub rpc_path: String /* "/a2a/jsonrpc/" */, pub card_path: String /* "/.well-known/agent-card.json" */,
                             pub protocol_version: String /* "1.0" */, pub host: String, pub port: u16 }
#[async_trait] pub trait A2aExecutor: Send + Sync {
    async fn execute(&self, ctx: A2aExecutionContext, sink: &EventSink) -> Result<()>;
    async fn cancel(&self, ctx: A2aExecutionContext) -> Result<()>;   // default: emit Canceled
}
pub struct A2aExecutionContext { pub task_id: String, pub context_id: String,
                                 pub message: Message, pub metadata: Map<String, Value> }
pub struct EventSink { .. }   // task(), working(), artifact(Artifact, last_chunk), complete(), fail(message), require_input(), canceled()
pub trait TaskStore: Send + Sync { fn get(&self, id: &str) -> Option<Task>; fn put(&self, task: Task); fn all(&self) -> Vec<Task>; }
pub struct InMemoryTaskStore;
pub struct A2aServer { .. }
impl A2aServer {
    pub fn new(config: A2aServerConfig, executor: Arc<dyn A2aExecutor>) -> Result<Self>;
    pub fn router(&self) -> axum::Router;                       // card route + JSON-RPC route
    pub async fn serve(self, shutdown: CancellationToken) -> Result<()>;
    pub fn serve_in_background(self, shutdown: CancellationToken) -> Result<ServerHandle>; // binds 127.0.0.1:0 when port == 0
}
pub struct ServerHandle { .. }  // local_addr(), shutdown(), join().await
pub struct A2aServerAdapter { .. }  // Python-compatible start()/stop()/active() wrapper over serve_in_background

// harness_bridge.rs — A2A ⇄ openjiuwen-harness-protocol
pub fn message_to_harness_input(message: &Message) -> HarnessInput;
pub fn harness_input_to_message(input: &HarnessInput, message_id: &str) -> Message;
pub fn turn_result_to_agent_result(result: &TurnResult) -> AgentResult;
pub fn agent_result_to_task(result: &AgentResult, task_id: Option<&str>, context_id: &str) -> Task;
pub fn part_to_content_block(part: &Part, block_id: &str) -> ContentBlock;
pub fn content_block_to_part(block: &ContentBlock) -> Part;
```

## Ported Behavior (Python → Rust)

- `to_request` (`A2ATransformer.to_a2a_request`): fresh `message_id = uuid4().hex`; `role = ROLE_USER`; `context_id` from `conversation_id` then `sessionId`; `query` becomes a single `Part{text}`; every other input key with a non-null value is copied into `message.metadata` (`query`, `sessionId`, `conversation_id` excluded).
- `from_request` (`from_a2a_request`): produced payload has `query` (first text part), `task_id` (when set), `sessionId` (from `context_id` when set); metadata merges only for keys the payload does not already have, and `None` values are skipped.
- `from_response` (`from_a2a_response`) precedence: artifact-update → `WORKING` with that artifact; status-update → mapped state, no artifacts; `Message` → `COMPLETED` with one artifact `message` holding every part; `Task` → its state and artifacts; anything else → `COMPLETED` with no artifacts. `TASK_STATE_REJECTED → FAILED`; `TASK_STATE_AUTH_REQUIRED → INPUT_REQUIRED`; unknown/unspecified → `UNKNOWN`.
- Client methods use the PascalCase v1 names; the server accepts both PascalCase and the v0.3 lowercase aliases (`method::canonical`).
- Streaming: one SSE `data:` frame per JSON-RPC response; `A2aClient::stream` yields one `AgentResult` per frame; `invoke` folds frames with `merge_results` (concatenate artifacts, override status when the newer one is set/known, fall back on `task_id`/`session_id`, dict-merge metadata).
- Server event order (Python `A2AAgentExecutor`): `Task{state=SUBMITTED, history=[message]}` first, then `WORKING`, then artifacts (each with `last_chunk` on terminal results), then close open artifacts with `append=true,last_chunk=true`, then the terminal state (`FAILED`/`CANCELED`/`INPUT_REQUIRED`, otherwise `COMPLETED`). Handler errors emit `FAILED` and are also returned as JSON-RPC errors for non-streaming calls.
- Agent card: `description` = base description, then `[input_params] {json}` and `[output_params] {json}` sections (`json.dumps(..., sort_keys=True, ensure_ascii=False)`); default input/output modes are `text/plain` and `application/json`; capabilities `streaming=true`, `push_notifications=false`; `from_a2a` copies only `name` and `description`.

## Constraints

- No `unsafe`; no blocking I/O; `reqwest` with rustls; SSE parsing is incremental over `bytes_stream` (no whole-body buffering).
- Timeouts: per-request via `A2aClientConfig.timeout` (default 60 s); `A2aRemoteClient::invoke/stream` wrap the call in the caller timeout and stop the client on timeout/error exactly like Python (`stop()` before re-raising; `Cancelled` propagates without calling `stop`).
- gRPC and REST (`HTTP+JSON`) transports are **not** implemented; `protocol_binding` other than `JSONRPC` yields `Error::Unsupported("gRPC transport is not supported")` / the REST route is not mounted (documented).
- Errors map to `Error::Transport` (HTTP/SSE), `Error::Timeout`, `Error::Validation` (card/inputs/JSON-RPC params), `Error::NotFound` (task id), and `Error::External`-free typed JSON-RPC codes surfaced in the message/`data`.
- Files ≤ ~1000 lines: `types.rs` (wire models), `jsonrpc.rs`, `transform.rs`, `card.rs`, `client.rs`, `server.rs`, `harness_bridge.rs`.

## Test Strategy

- `./tests/a2a_client.rs` (`#![cfg(feature = "a2a")]`): `wiremock` mocks — send returns a `Task` and folds to `AgentResult`; SSE stream of three frames yields three results (status/artifact terminal `last_chunk`); `CancelTask` fields and `CANCELED`; JSON-RPC error → `Error`; missing card in `RemoteClientConfig` → `Validation`.
- `./tests/a2a_server.rs` (`#![cfg(feature = "a2a")]`): `Router::oneshot` — card JSON at `/.well-known/agent-card.json`; `message/send` and `SendMessage` both dispatch; `tasks/get` for unknown id → `-32001`; `tasks/cancel` → `CANCELED`; `message/stream` emits `text/event-stream` with the expected frame order; executor error → `TASK_STATE_FAILED`.
- Unit tests beside `transform.rs`/`card.rs`/`jsonrpc.rs` pin the mapping tables, the metadata merge rule, and the method aliases.
- Python counterparts named in comments: `tests/unit_tests/extensions/a2a/test_a2a_transformer.py`, `test_a2a_agentcard_adapter.py`, `test_a2a_client.py`, `test_a2a_server.py`.

## Routing Table

- `./types.rs` → wire models and their serde mapping
- `./jsonrpc.rs` → envelopes, error codes, method names/aliases
- `./transform.rs` → openjiuwen `AgentResult` model, request/response mapping, result folding
- `./card.rs` → agent-card adaptation both ways
- `./client.rs` → `A2aClient`, `A2aRemoteClient`, SSE streaming
- `./server.rs` → axum router, JSON-RPC dispatch, executor/event sink/task store, server handle
- `./harness_bridge.rs` → conversions to/from `openjiuwen_harness_protocol`
