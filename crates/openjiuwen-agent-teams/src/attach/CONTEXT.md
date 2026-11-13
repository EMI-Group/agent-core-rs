# attach

## Intent

Cross-process team attach: let another process drive a hosted team over a Unix domain socket.
This directory owns the platform-independent wire contract; the socket server and client are built on top of it.
The protocol does not import `skill` or `mcp` — those layers implement `AttachHandler` over their own client seams.

## API Surface

Module `protocol` (re-exported flat from `mod.rs`):

```rust
pub const PROTOCOL_VERSION: u32 = 1;
pub const MAX_FRAME_BYTES: usize = 1024 * 1024;

pub struct AttachError { pub kind: ErrorKind, pub message: String }
impl AttachError {
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self;
    pub fn protocol(message: impl Into<String>) -> Self;   // Validation
    pub fn transport(message: impl Into<String>) -> Self;  // Transport
    pub fn timeout(message: impl Into<String>) -> Self;    // Timeout
    pub fn remote(kind: ErrorKind, message: impl Into<String>) -> Self;
    pub fn into_error(self) -> Error;
}
// Serde: {"kind":"...","message":"..."}; unknown kind -> External; unknown fields ignored.
pub type AttachResult<T> = std::result::Result<T, AttachError>;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum AttachOp { ListMembers, MemberStatus { member_name }, ReadInbox, SendMember { to, content },
    Broadcast { content }, ListTasks { status: Option<TaskStatus> }, ClaimableTasks,
    TaskDetail { task_id }, CreateTask { title, content, task_id, dependencies },
    UpdateTask { task_id, title, content }, ClaimTask { task_id }, CompleteTask { task_id },
    InvokeTool { name, arguments }, Snapshot, StopTeam }
impl AttachOp {
    pub fn name(&self) -> &'static str;
    pub fn args(&self) -> AttachResult<Value>;                     // object without "op"
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

## Wire Contract

One UTF-8 JSON object per line, terminated by `\n`; decoders also accept a line without it.
Request `{"v":1,"id":7,"op":"list_members","args":{}}`; response `{"v":1,"id":7,"ok":true,"data":...}` or `{"v":1,"id":7,"ok":false,"error":{"kind":"...","message":"..."}}`.
`data`/`error` are omitted when absent; `ok` without `data` is JSON `null`; `ok: false` without `error` is a protocol error.
Version other than `PROTOCOL_VERSION` and frames longer than `MAX_FRAME_BYTES` are rejected as `Validation`.
A frame-level error closes the connection; an op-level `AttachError` keeps its kind across the wire and keeps the connection open.
Per-op `data` shapes are documented in the `protocol` module header (schema types for roster/task/snapshot ops, adapter-local `{ok, message_id, reason}` objects for mutations).

## Constraints

- Unix domain sockets only: the socket server/client constructors (added by later stages) return `Error::Unsupported` with "cross-process attach requires Unix domain sockets" on non-Unix targets; this protocol layer compiles everywhere.
- No authentication in the protocol: trust comes from the socket living in a `0700` directory owned by the user, so handlers still validate every name, id and tool argument.
- Depends only on `openjiuwen-common`, `serde`, `serde_json`, `tokio`, `async-trait` and `crate::schema`; never on `crate::skill` or `crate::mcp`.
- `tokio` IO traits are generic in the frame helpers so server and client share them; helpers do not bind sockets.
- Every public item is documented; no `unwrap`/`expect` in non-test code; errors are `AttachError` (never `anyhow`, never a panic on input).

## Routing Table

- `./protocol.rs` → envelope types, serde, validation, frame IO, `AttachHandler`
- `./protocol_tests.rs` → unit tests for the contract (included as `#[cfg(test)] mod protocol_tests`), no sockets
- `server.rs` / `client.rs` → socket server and client built on this contract (to be added by later stages)

## Notes

- `RequestFrame::decode` does not interpret `op`; unknown ops surface at `RequestFrame::op()` / `AttachOp::from_parts`.
- `AttachError` is the wire form of `openjiuwen_common::error::Error`: `From<Error>` keeps `kind()`/`message()`, and `into_error()` maps every kind back (`External` becomes `Error::external(std::io::Error::other(message))`).
- `read_frame_line` drains the remainder of an oversized line so a tolerant caller can resume; EOF before a new line returns `Ok(None)`, a final unterminated line is returned.
