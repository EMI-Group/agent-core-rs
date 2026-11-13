# src/attach — cross-process team attach over Unix domain sockets

## Intent

Let a process drive a team hosted by another process over a Unix domain socket: the `openjiuwen` host binds the socket, and the `team-member` CLI or the `openjiuwen-team-mcp` server connects to it.
Python parity source: `openjiuwen/agent_teams/external/` (descriptor/client) and `openjiuwen/agent_teams/organization/transport_api.py`; Python reaches a team through shared storage and pyzmq, so the socket protocol is the Rust equivalent of that cross-process seam.
The layer is deliberately split: [`protocol`] owns the platform-independent wire contract, [`client`] drives a hosted team, [`server`] hosts one.
This module never imports `crate::skill` or `crate::mcp`; those layers adapt their own `TeamClient` onto [`AttachHandler`], which keeps the dependency direction one-way and lets the wiring change without touching the protocol.
Everything is in-process-tokio and offline: no broker, no network, no new dependencies.

## API Surface

Flat re-exports from `mod.rs`: `AttachError`, `AttachHandler`, `AttachOp`, `AttachResult`, `AttachClient`, `AttachServer`, `RequestFrame`, `ResponseFrame`, `MAX_FRAME_BYTES`, `PROTOCOL_VERSION`.
The frame IO helpers (`read_frame_line`, `write_all_line`) and `to_data` stay in `attach::protocol`.

```rust
// protocol.rs — platform independent
pub const PROTOCOL_VERSION: u32 = 1;
pub const MAX_FRAME_BYTES: usize = 1024 * 1024;

pub struct AttachError { pub kind: ErrorKind, pub message: String }
impl AttachError {
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self;
    pub fn protocol(message: impl Into<String>) -> Self;    // Validation
    pub fn transport(message: impl Into<String>) -> Self;   // Transport
    pub fn timeout(message: impl Into<String>) -> Self;     // Timeout
    pub fn remote(kind: ErrorKind, message: impl Into<String>) -> Self;
    pub fn into_error(self) -> Error;
}
impl From<Error> for AttachError;      // preserves kind()/message()
impl From<AttachError> for Error;      // kind-preserving; External -> Error::external(io::Error::other(message))
pub type AttachResult<T> = std::result::Result<T, AttachError>;

#[serde(tag = "op", rename_all = "snake_case")]
pub enum AttachOp { ListMembers, MemberStatus { member_name }, ReadInbox, SendMember { to, content },
    Broadcast { content }, ListTasks { status: Option<TaskStatus> }, ClaimableTasks,
    TaskDetail { task_id }, CreateTask { title, content, task_id, dependencies },
    UpdateTask { task_id, title, content }, ClaimTask { task_id }, CompleteTask { task_id },
    InvokeTool { name, arguments }, Snapshot, StopTeam }
impl AttachOp {
    pub fn name(&self) -> &'static str;                       // the wire `op`
    pub fn args(&self) -> AttachResult<Value>;                // object without the `op` key
    pub fn from_parts(op: &str, args: Value) -> AttachResult<Self>;
    pub fn validate(&self) -> AttachResult<()>;
}

pub struct RequestFrame { pub v: u32, pub id: u64, pub op: String, pub args: Value }
impl RequestFrame { pub fn new(id: u64, op: &AttachOp) -> AttachResult<Self>; pub fn decode(line: &[u8]) -> AttachResult<Self>;
                    pub fn op(&self) -> AttachResult<AttachOp>; pub fn encode(&self) -> AttachResult<Vec<u8>>; }

pub struct ResponseFrame { pub v: u32, pub id: u64, pub ok: bool, pub data: Option<Value>, pub error: Option<AttachError> }
impl ResponseFrame { pub fn ok(id: u64, data: Value) -> Self; pub fn err(id: u64, error: AttachError) -> Self;
                     pub fn decode(line: &[u8]) -> AttachResult<Self>; pub fn into_result(self) -> AttachResult<Value>;
                     pub fn encode(&self) -> AttachResult<Vec<u8>>; }

pub async fn read_frame_line<R: AsyncBufRead + Unpin>(reader: &mut R) -> AttachResult<Option<Vec<u8>>>;
pub async fn write_all_line<W: AsyncWrite + Unpin>(writer: &mut W, bytes: &[u8]) -> AttachResult<()>;

#[async_trait]
pub trait AttachHandler: Send + Sync { async fn handle(&self, op: AttachOp) -> AttachResult<Value>; }
pub fn to_data<T: Serialize>(value: &T) -> AttachResult<Value>;
```

```rust
// client.rs — same surface on every target
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(10);   // per call (write + read)
pub const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);   // per connect

pub struct AttachClient;
impl AttachClient {
    pub async fn connect(path: impl AsRef<Path>) -> AttachResult<Self>;
    pub async fn connect_with_timeout(path: impl AsRef<Path>, timeout: Duration) -> AttachResult<Self>;
    pub fn path(&self) -> &Path;
    pub fn timeout(&self) -> Duration;
    pub async fn call(&mut self, op: AttachOp) -> AttachResult<Value>;
    pub async fn is_available(path: impl AsRef<Path>) -> bool;   // 1 s probe, connects then drops
    pub async fn list_members(&mut self) -> AttachResult<Vec<MemberSnapshot>>;
    pub async fn member_status(&mut self, member_name: &str) -> AttachResult<MemberSnapshot>;
    pub async fn read_inbox(&mut self) -> AttachResult<String>;
    pub async fn send_member(&mut self, to: &str, content: &str) -> AttachResult<Value>;
    pub async fn broadcast(&mut self, content: &str) -> AttachResult<Value>;
    pub async fn list_tasks(&mut self, status: Option<TaskStatus>) -> AttachResult<Vec<TaskSummary>>;
    pub async fn claimable_tasks(&mut self) -> AttachResult<Vec<TaskSummary>>;
    pub async fn task_detail(&mut self, task_id: &str) -> AttachResult<Option<TaskDetail>>;
    pub async fn create_task(&mut self, title: &str, content: &str, task_id: Option<&str>, dependencies: &[String]) -> AttachResult<Value>;
    pub async fn update_task(&mut self, task_id: &str, title: Option<&str>, content: Option<&str>) -> AttachResult<Value>;
    pub async fn claim_task(&mut self, task_id: &str) -> AttachResult<Value>;
    pub async fn complete_task(&mut self, task_id: &str) -> AttachResult<Value>;
    pub async fn invoke_tool(&mut self, name: &str, arguments: Value) -> AttachResult<Value>;
    pub async fn snapshot(&mut self) -> AttachResult<TeamSnapshot>;
    pub async fn stop_team(&mut self) -> AttachResult<Value>;
}
```

```rust
// server.rs — same surface on every target
pub struct AttachServer;
impl AttachServer {
    pub async fn bind(path: impl AsRef<Path>) -> openjiuwen_common::error::Result<Self>;
    #[must_use] pub fn path(&self) -> &Path;
    pub fn shutdown(&self);
    #[must_use] pub fn cancellation_token(&self) -> CancellationToken;
    pub async fn serve(&self, handler: Arc<dyn AttachHandler>) -> openjiuwen_common::error::Result<()>;
}
impl Drop for AttachServer;   // Unix: removes the path only while it is still a socket
```

## Wire Contract

One UTF-8 JSON object per line, terminated by `\n`; decoders accept a line without the trailing newline and tolerate one trailing `\r`.
Request: `{"v":1,"id":7,"op":"list_members","args":{}}`.
Response: `{"v":1,"id":7,"ok":true,"data":...}` or `{"v":1,"id":7,"ok":false,"error":{"kind":"not_found","message":"..."}}`.
`data` and `error` are omitted when absent; an `ok` response without `data` is JSON `null`; an error response without `error` is a protocol error.
Every request is answered by exactly one response echoing its `id`; the client rejects a mismatched id and never retries.
`v` must equal `PROTOCOL_VERSION`; unknown versions are `Validation` errors naming the version.
The format is additive: unknown JSON fields are ignored everywhere, and an unknown error `kind` decodes as `ErrorKind::External`.
Frame-level faults (bad UTF-8/JSON, wrong `v`, oversized line, unknown `op`, invalid op fields) get one typed error response and close the connection.
Op-level faults (handler `AttachError`) travel in-band with `ok:false`, keep their `ErrorKind`, and leave the connection open.
The server answers an unparseable frame with `id: 0`; a frame whose envelope parses but whose `op` does not keeps the request id.

## Op Table

| `op` | args | `data` |
| --- | --- | --- |
| `list_members` | — | array of `MemberSnapshot` |
| `member_status` | `member_name` | one `MemberSnapshot`; `not_found` when the member is unknown |
| `read_inbox` | — | a string |
| `send_member` | `to`, `content` | `{"ok": bool, "message_id": string|null, "reason": string|null}` |
| `broadcast` | `content` | same as `send_member` (`to = "*"`) |
| `list_tasks` | `status?` | array of `TaskSummary` |
| `claimable_tasks` | — | array of `TaskSummary` |
| `task_detail` | `task_id` | a `TaskDetail` or `null` |
| `create_task` | `title`, `content`, `task_id?`, `dependencies` | `{"ok": bool, "task_id": string|null, "reason": string|null}` |
| `update_task` | `task_id`, `title?`, `content?` | `{"ok": bool, "reason": string|null}` |
| `claim_task` | `task_id` | `{"ok": bool, "reason": string|null}` |
| `complete_task` | `task_id` | `{"ok": bool, "reason": string|null}` |
| `invoke_tool` | `name`, `arguments` | the tool's own result JSON |
| `snapshot` | — | a `TeamSnapshot` |
| `stop_team` | — | `{"stopped": true}` |

Op coverage is at parity with `crate::skill::TeamClient` ops plus the runtime's member-status, snapshot and stop operations.
`TeamClient::tools()` has no op: the tool catalogue is local metadata the adapter already holds, so sending it over the socket would add nothing.
Identity fields are validated before any I/O: `member_name`, `task_id`, `to`, `name` and a `create_task` `title` must be non-empty, and the error names the op and field.

## Security & Limits

The protocol carries no authentication and no encryption; trust is delegated to the filesystem.
`AttachServer::bind` creates a `0600` socket and expects it inside a directory only the owner can enter (`0700`), so only same-user processes can connect.
Handlers must still treat every payload as untrusted: validate names, ids and tool arguments before acting, because the socket constrains *who* may connect, not *what* they may ask for.
`MAX_FRAME_BYTES` (1 MiB) bounds one line; `read_frame_line` drains the rest of an oversized line, and the server treats it as a frame-level fault, answers once and closes.
Bind refuses to clobber a live socket (`ErrorKind::Config`, "attach socket already in use"), reclaims a stale socket file left by a dead process, refuses to unlink a regular file at the path, and removes the file on shutdown and on drop.
A socket path is limited by the OS (`sun_path`, ~108 bytes on Linux); tests and callers keep socket paths short (`<tmpdir>/s`).
The server has no connection cap and no per-request deadline: each connection is a task and an in-flight handler call may run until the handler returns (shutdown waits a 2 s grace, then aborts).
The client bounds everything it does: `CONNECT_TIMEOUT` per connect, `DEFAULT_TIMEOUT` (or the per-client timeout) per call covering write plus read, and a 1 s `is_available` probe.
One client owns one connection with at most one in-flight request (`&mut self`); request ids start at 1 and wrap back to 1 (never 0) after `u64::MAX`.
`is_available` proves the socket accepts connections, not that the hosted team is healthy.

## Platform Matrix

| Target | `protocol` | `AttachClient` | `AttachServer` |
| --- | --- | --- | --- |
| Unix | full | real Unix socket client | real Unix socket server, unlinks on shutdown/drop |
| non-Unix | full (platform independent) | `connect`/`call`/typed wrappers return `AttachError::remote(Unsupported, "cross-process attach requires Unix domain sockets")`; `is_available` is `false` | `bind`/`serve` return `Error::Unsupported` with the same text; `shutdown` is a no-op |

The cfg split is per method block, not per file: the public API is identical on both targets so callers compile unchanged.

## Constraints

- Edits stay inside `crates/openjiuwen-agent-teams/src/attach/`; the skill, MCP, umbrella and binary layers are separate nodes.
- Depends only on `openjiuwen-common`, `serde`, `serde_json`, `tokio`, `async-trait`, `tokio-util` and `crate::schema`; never on `crate::skill` or `crate::mcp`.
- Errors are `AttachError` inside the module and `openjiuwen_common::error::{Error, Result}` at the socket boundary; no `anyhow`, no panic on input.
- Async-first tokio; locks are never held across `.await`; `tracing` only (no `println!`/`eprintln!` in library code); no `unsafe`.
- Every public item is documented (workspace `missing_docs`); files stay around ≤1000 lines, with the test files as the largest files (legitimately long: one contract, one scenario per test).

## Routing Table

- `./protocol.rs` → envelope types, `AttachOp`, serde, validation, frame IO, `AttachError`, `AttachHandler`, `to_data`
- `./client.rs` → `AttachClient`: connect, one-in-flight `call`, typed wrappers, `is_available`, non-Unix stubs
- `./server.rs` → `AttachServer`: `bind` (parents, stale reclaim, `0600`), concurrent `serve`, cancellation, socket cleanup
- `./protocol_tests.rs` → contract unit tests (`#[cfg(test)]`, no sockets)
- `./client_tests.rs` → client tests against a raw in-test Unix listener (`#![cfg(unix)]`)
- `./server_tests.rs` → server tests driven through `AttachClient` plus raw streams (`#![cfg(unix)]`)
- `../skill/` → `TeamClient`/`TeamJoinDescriptor` the adapter will implement over `AttachHandler` (sibling node, read-only here)
- `../mcp/` → MCP server that consumes a `TeamClient` (sibling node, read-only here)

## Test Strategy

84 tests, all offline and deterministic with bounded (≤10 s) waits and no `#[ignore]`.
`protocol_tests.rs` (34): exact request/response/error wire bytes, every `AttachOp` variant round-tripping `name`/`args`/`from_parts`/frame, unit ops producing `{}`, unknown-field tolerance, `args` missing/null/array, bad JSON, missing/unsupported `v`, unknown op, empty-field validation per field, error serde incl. unknown kind → `External`, `From<Error>`/`into_error` kind table, `ok:false` without error, `ok:true` without data, frame-line EOF/oversize/drain over `tokio::io::duplex`, `to_data`, and `Arc<dyn AttachHandler>` object safety.
`client_tests.rs` (27): real socket in a `tempfile` dir; connect (live/missing), sequential clients, exact request framing and id sequence 1→2→3, all 15 ops through `call`, typed decoders (incl. `task_detail` `Some`/`null`), remote `not_found`/`validation` kind preservation, malformed response, id and version mismatch, oversized response, 80 ms timeout against a silent server, EOF, unexpected payload, and the `is_available` matrix (missing/listening/removed, and that the probe sends no frame).
`server_tests.rs` (23): parent-dir creation, `0600` mode, live-socket refusal, stale reclamation, all ops round-tripped through `AttachClient` against a recording handler with per-op payloads, typed wrappers, `NotFound` surviving while the connection keeps serving, bad JSON/unknown op/invalid fields/bad version/empty line/oversized line each answered once with the server surviving, the `is_available` connect-and-close probe, two concurrent clients, `shutdown()` returning `Ok(())` and removing the file, cloned-token cancellation, idle-connection close on shutdown, and drop safety (socket removed; a replacing regular file left alone).
Definition of done: `cargo fmt -p openjiuwen-agent-teams`, `cargo test -p openjiuwen-agent-teams --offline`, `cargo test -p openjiuwen-agent-teams --offline --no-default-features`, `cargo clippy -p openjiuwen-agent-teams --all-targets --offline -- -D warnings`, and `cargo doc -p openjiuwen-agent-teams --no-deps --offline` warning-free.

## Deferred

- Authentication and TLS beyond filesystem permissions; there is no token, peer-credential check (`SO_PEERCRED`) or encryption.
- Windows named pipes (and any non-Unix transport); both socket types keep API shape but return `Error::Unsupported`.
- Multi-writer fan-out, subscriptions and streaming events: one request/response per line, no server-push channel, so the CLI's `--watch` and event streaming stay in-process.
- Connection caps, per-request deadlines and backpressure on the server side (the client side is bounded).
- Cross-process wiring itself: `crate::skill::connect_live_client` and the MCP resolver still resolve only teams hosted in the current process, so the socket client/server exist but nothing constructs them yet.

## Notes for Agents

- The consumer contract is a single trait impl: `impl AttachHandler for TeamClientHandler` holding an `Arc<dyn TeamClient>`, matching `AttachOp` to the corresponding `TeamClient`/runtime call, returning `protocol::to_data(&value)` (or `Ok(Value::Null)` for a `task_detail` miss) and mapping failures through `AttachError::from(error)` so the kind survives the wire.
- Host it with `AttachServer::bind(path)` → `tokio::spawn(server.serve(Arc::new(handler)))` → stop with `server.shutdown()` or a clone of `server.cancellation_token()`; `serve` shares one `Arc<dyn AttachHandler>` across all connections, so the adapter must be `Send + Sync` and must not hold a lock across an `await`.
- The socket path has to come from the join descriptor (`transport_config`/`workspace_config` are the natural carriers) and must stay short; the CLI/MCP resolver should try the socket first and fall back to the in-process registry with the existing "no live team '<name>' hosted in this process" error.
- Use `Error::validation` for user-supplied path/op problems, `Error::config` for an occupied socket, `Error::transport` for socket IO, and `Error::sys_op` when the bound socket's mode cannot be set.
- `RequestFrame::decode` deliberately does not interpret `op`; unknown ops surface at `RequestFrame::op()`/`AttachOp::from_parts`, which is where the server turns them into a frame-level fault.
- `AttachError` is the wire form of `openjiuwen_common::error::Error`: `From<Error>` keeps `kind()`/`message()`, and `into_error()` maps every kind back (`External` becomes `Error::external(io::Error::other(message))`).
- The client's id wrap keeps ids non-zero; `ResponseFrame::into_result` is the single place that turns `ok:false` into an `AttachError`, so remote kinds reach callers unchanged.
