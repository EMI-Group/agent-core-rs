//! Wire contract for cross-process team attach: one JSON object per line over a Unix socket.
//!
//! This is a Rust-side protocol for driving an already-running team from another process
//! (the `team-member` CLI and the `openjiuwen-team-mcp` server are the first consumers).
//! It is deliberately independent of the `skill` and `mcp` layers: both of them adapt their
//! existing client seams onto [`AttachHandler`], and this module never imports either of them.
//!
//! # Framing
//!
//! Every frame is a single UTF-8 JSON object terminated by one `\n` byte.
//! A JSON object is always one line, because `serde_json` escapes any newline inside strings.
//! The trailing newline is required on the wire; the decoders accept a frame with or without it,
//! and tolerate one trailing `\r` for hand-written test fixtures.
//!
//! A request carries a monotonically increasing `id` chosen by the client and a `v` version:
//!
//! ```json
//! {"v":1,"id":7,"op":"list_members","args":{}}
//! ```
//!
//! The server echoes the same `id` in exactly one response, which is either a success:
//!
//! ```json
//! {"v":1,"id":7,"ok":true,"data":[]}
//! ```
//!
//! or a typed failure:
//!
//! ```json
//! {"v":1,"id":7,"ok":false,"error":{"kind":"not_found","message":"task missing"}}
//! ```
//!
//! `data` and `error` are omitted from the wire when absent; an `ok` response without `data`
//! decodes to JSON `null`, and an error response without `error` is rejected as a protocol error.
//!
//! # Versioning
//!
//! [`PROTOCOL_VERSION`] is `1`.
//! Both frame decoders reject any other `v` with a [`ErrorKind::Validation`] error whose message
//! starts with `unsupported protocol version`; a frame without `v` is malformed and also rejected.
//! The wire format is additive: unknown JSON fields are ignored in requests, responses and errors,
//! and unknown error kinds decode as [`ErrorKind::External`] so an older peer can still surface a
//! newer peer's failure.
//!
//! # Limits
//!
//! [`MAX_FRAME_BYTES`] bounds one line, including its trailing newline.
//! [`read_frame_line`] rejects longer lines with a `Validation` error and drains the rest of the
//! offending line so that the connection can, in principle, continue; callers that treat framing
//! errors as fatal simply close the socket.
//!
//! # Error semantics
//!
//! There are two failure levels.
//! A *frame-level* failure (bad UTF-8, malformed JSON, wrong version, oversized line) means the
//! peer cannot be trusted to stay in sync, so the connection is closed.
//! An *op-level* failure travels in a response with `ok: false` and keeps the connection open:
//! the handler's [`AttachError`] preserves its [`ErrorKind`] across the wire, so the remote caller
//! can react to `not_found`, `validation`, `unsupported` and friends exactly as it would locally.
//!
//! # Data payloads
//!
//! The protocol module only owns the envelope; each op's `data` is the adapter's own JSON.
//! These are the shapes the skill/MCP adapters produce (all are `serde` types from
//! [`crate::schema`], serialized with their existing wire names):
//!
//! | `op` | `data` |
//! | --- | --- |
//! | `list_members` | array of `MemberSnapshot` |
//! | `member_status` | one `MemberSnapshot` |
//! | `read_inbox` | a string |
//! | `send_member`, `broadcast` | `{"ok": bool, "message_id": string|null, "reason": string|null}` |
//! | `list_tasks`, `claimable_tasks` | array of `TaskSummary` |
//! | `task_detail` | a `TaskDetail` or `null` |
//! | `create_task` | `{"ok": bool, "task_id": string|null, "reason": string|null}` |
//! | `update_task`, `claim_task`, `complete_task` | `{"ok": bool, "reason": string|null}` |
//! | `invoke_tool` | the tool's own result JSON |
//! | `snapshot` | a `TeamSnapshot` |
//! | `stop_team` | `{"stopping": true}` |
//!
//! # Security
//!
//! The transport carries no authentication or encryption of its own.
//! Trust is delegated to the filesystem: the server binds a Unix domain socket inside a
//! directory that is only accessible to the owning user (`0700`), so only processes running as
//! that user can connect.
//! Handlers must therefore still treat every payload as untrusted input and validate names,
//! ids and tool arguments before acting on them — the socket protects *who* may connect, not
//! *what* they may ask for.
use std::fmt;

use async_trait::async_trait;
use openjiuwen_common::error::{Error, ErrorKind};
use serde::de::Deserializer;
use serde::ser::Serializer;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::io::{AsyncBufRead, AsyncBufReadExt, AsyncWrite, AsyncWriteExt};

use crate::schema::TaskStatus;

/// Version of the attach wire protocol implemented by this module.
///
/// Peers must send exactly this value in a frame's `v` field.
pub const PROTOCOL_VERSION: u32 = 1;

/// Maximum size of one frame line in bytes, including its trailing newline.
///
/// [`read_frame_line`] rejects longer lines instead of buffering them without bound.
pub const MAX_FRAME_BYTES: usize = 1024 * 1024;

/// A typed failure that can cross the attach wire.
///
/// The pair of [`ErrorKind`] and message is the wire representation of an
/// [`Error`] as `{"kind": "transport", "message": "..."}`.
/// Unknown kinds decode as [`ErrorKind::External`] so that a newer peer's error still surfaces.
#[derive(Debug, Clone, PartialEq)]
pub struct AttachError {
    /// Category of the failure, preserved across the wire.
    pub kind: ErrorKind,
    /// Human-readable detail; may be empty.
    pub message: String,
}

impl AttachError {
    /// Builds an error from a kind and a message.
    #[must_use]
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }

    /// A protocol violation: malformed frames, unknown ops, failed validation.
    #[must_use]
    pub fn protocol(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Validation, message)
    }

    /// A transport-level failure: read or write errors on the socket.
    #[must_use]
    pub fn transport(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Transport, message)
    }

    /// A deadline was exceeded while waiting for the peer.
    #[must_use]
    pub fn timeout(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Timeout, message)
    }

    /// A failure reported by the remote handler, keeping its kind.
    #[must_use]
    pub fn remote(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self::new(kind, message)
    }

    /// Converts into the shared [`Error`], preserving the kind.
    ///
    /// [`ErrorKind::External`] becomes [`Error::external`] wrapping the message,
    /// because [`Error::External`] keeps a source error rather than a string.
    pub fn into_error(self) -> Error {
        let Self { kind, message } = self;
        match kind {
            ErrorKind::Config => Error::config(message),
            ErrorKind::Validation => Error::validation(message),
            ErrorKind::Llm => Error::llm(message),
            ErrorKind::Tool => Error::tool(message),
            ErrorKind::Workflow => Error::workflow(message),
            ErrorKind::Graph => Error::graph(message),
            ErrorKind::Session => Error::session(message),
            ErrorKind::Memory => Error::memory(message),
            ErrorKind::Retrieval => Error::retrieval(message),
            ErrorKind::SysOp => Error::sys_op(message),
            ErrorKind::Security => Error::security(message),
            ErrorKind::Mcp => Error::mcp(message),
            ErrorKind::Transport => Error::transport(message),
            ErrorKind::Timeout => Error::timeout(message),
            ErrorKind::NotFound => Error::not_found(message),
            ErrorKind::Unsupported => Error::unsupported(message),
            ErrorKind::Cancelled => Error::cancelled(),
            ErrorKind::External => Error::external(std::io::Error::other(message)),
        }
    }
}

impl From<Error> for AttachError {
    fn from(error: Error) -> Self {
        Self::new(error.kind(), error.message().into_owned())
    }
}

impl From<AttachError> for Error {
    fn from(error: AttachError) -> Self {
        error.into_error()
    }
}

impl fmt::Display for AttachError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.message.is_empty() {
            write!(formatter, "{}", self.kind)
        } else {
            write!(formatter, "{}: {}", self.kind, self.message)
        }
    }
}

impl std::error::Error for AttachError {}

impl Serialize for AttachError {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;

        let mut state = serializer.serialize_struct("AttachError", 2)?;
        state.serialize_field("kind", self.kind.as_str())?;
        state.serialize_field("message", &self.message)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for AttachError {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Default, Deserialize)]
        #[serde(default)]
        struct RawAttachError {
            kind: Option<String>,
            message: Option<String>,
        }

        let raw = RawAttachError::deserialize(deserializer)?;
        Ok(Self {
            kind: raw
                .kind
                .as_deref()
                .map_or(ErrorKind::External, kind_from_wire),
            message: raw.message.unwrap_or_default(),
        })
    }
}

/// Maps a wire kind string back to [`ErrorKind`]; anything unknown is [`ErrorKind::External`].
fn kind_from_wire(value: &str) -> ErrorKind {
    match value {
        "config" => ErrorKind::Config,
        "validation" => ErrorKind::Validation,
        "llm" => ErrorKind::Llm,
        "tool" => ErrorKind::Tool,
        "workflow" => ErrorKind::Workflow,
        "graph" => ErrorKind::Graph,
        "session" => ErrorKind::Session,
        "memory" => ErrorKind::Memory,
        "retrieval" => ErrorKind::Retrieval,
        "sys_op" => ErrorKind::SysOp,
        "security" => ErrorKind::Security,
        "mcp" => ErrorKind::Mcp,
        "transport" => ErrorKind::Transport,
        "timeout" => ErrorKind::Timeout,
        "not_found" => ErrorKind::NotFound,
        "unsupported" => ErrorKind::Unsupported,
        "cancelled" => ErrorKind::Cancelled,
        _ => ErrorKind::External,
    }
}

/// Result of an attach protocol operation.
pub type AttachResult<T> = std::result::Result<T, AttachError>;

/// An operation a remote client asks the hosting team to perform.
///
/// On the wire the variant name is the `op` field (`snake_case`), and struct fields are inlined
/// into the same JSON object, for example:
/// `{"op":"member_status","member_name":"dev-1"}`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum AttachOp {
    /// List the current roster.
    ListMembers,
    /// Read one member's lifecycle and execution status.
    MemberStatus {
        /// Name of the member to inspect.
        member_name: String,
    },
    /// Read the caller's inbox as rendered text.
    ReadInbox,
    /// Send a direct message to one member.
    SendMember {
        /// Recipient member name; role placeholders are resolved by the adapter.
        to: String,
        /// Message body.
        content: String,
    },
    /// Send a message to every member.
    Broadcast {
        /// Message body.
        content: String,
    },
    /// List tasks, optionally filtered by status.
    ListTasks {
        /// Only return tasks in this status when set.
        status: Option<TaskStatus>,
    },
    /// List tasks that the caller may claim right now.
    ClaimableTasks,
    /// Read one task in detail.
    TaskDetail {
        /// Identity of the task to read.
        task_id: String,
    },
    /// Create a task on the board.
    CreateTask {
        /// Short title shown on the board.
        title: String,
        /// Long-form description and acceptance criteria.
        content: String,
        /// Explicit id to use instead of a generated one.
        task_id: Option<String>,
        /// Task ids that must complete before this task unblocks.
        dependencies: Vec<String>,
    },
    /// Update a task's title and/or content.
    UpdateTask {
        /// Identity of the task to update.
        task_id: String,
        /// New title when set.
        title: Option<String>,
        /// New content when set.
        content: Option<String>,
    },
    /// Claim a pending task for the caller.
    ClaimTask {
        /// Identity of the task to claim.
        task_id: String,
    },
    /// Mark a claimed task complete.
    CompleteTask {
        /// Identity of the task to complete.
        task_id: String,
    },
    /// Invoke a tool exposed by the hosting team.
    InvokeTool {
        /// Tool name from the advertised catalogue.
        name: String,
        /// Tool-specific arguments as a JSON value.
        arguments: Value,
    },
    /// Read a full team snapshot (roster plus task board).
    Snapshot,
    /// Ask the hosting process to stop the team.
    StopTeam,
}

impl AttachOp {
    /// The `op` string used on the wire.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::ListMembers => "list_members",
            Self::MemberStatus { .. } => "member_status",
            Self::ReadInbox => "read_inbox",
            Self::SendMember { .. } => "send_member",
            Self::Broadcast { .. } => "broadcast",
            Self::ListTasks { .. } => "list_tasks",
            Self::ClaimableTasks => "claimable_tasks",
            Self::TaskDetail { .. } => "task_detail",
            Self::CreateTask { .. } => "create_task",
            Self::UpdateTask { .. } => "update_task",
            Self::ClaimTask { .. } => "claim_task",
            Self::CompleteTask { .. } => "complete_task",
            Self::InvokeTool { .. } => "invoke_tool",
            Self::Snapshot => "snapshot",
            Self::StopTeam => "stop_team",
        }
    }

    /// The op's arguments as a JSON object, without the `op` discriminator.
    ///
    /// # Errors
    ///
    /// Returns a `Validation` error if the op cannot be serialized, which only happens for
    /// non-string map keys inside [`AttachOp::InvokeTool`] arguments.
    pub fn args(&self) -> AttachResult<Value> {
        let value = serde_json::to_value(self).map_err(|error| {
            AttachError::protocol(format!("encoding op `{}` failed: {error}", self.name()))
        })?;
        let Value::Object(mut map) = value else {
            return Err(AttachError::protocol(format!(
                "op `{}` did not serialize to a JSON object",
                self.name()
            )));
        };
        map.remove("op");
        Ok(Value::Object(map))
    }

    /// Rebuilds an op from a wire `op` string and its `args` object.
    ///
    /// # Errors
    ///
    /// Returns a `Validation` error when `args` is not a JSON object, when `op` is unknown, when a
    /// field has the wrong type, or when [`AttachOp::validate`] rejects the resulting value.
    pub fn from_parts(op: &str, args: Value) -> AttachResult<Self> {
        let Value::Object(mut map) = args else {
            return Err(AttachError::protocol(format!(
                "op `{op}` args must be a JSON object"
            )));
        };
        map.insert("op".to_owned(), Value::String(op.to_owned()));
        let parsed: Self = serde_json::from_value(Value::Object(map)).map_err(|error| {
            AttachError::protocol(format!("unknown or malformed op `{op}`: {error}"))
        })?;
        parsed.validate()?;
        Ok(parsed)
    }

    /// Checks the op's required string fields are non-empty.
    ///
    /// # Errors
    ///
    /// Returns a `Validation` error naming both the op and the offending field.
    pub fn validate(&self) -> AttachResult<()> {
        match self {
            Self::MemberStatus { member_name } => {
                require_field(self.name(), "member_name", member_name)
            }
            Self::SendMember { to, .. } => require_field(self.name(), "to", to),
            Self::TaskDetail { task_id }
            | Self::ClaimTask { task_id }
            | Self::CompleteTask { task_id }
            | Self::UpdateTask { task_id, .. } => require_field(self.name(), "task_id", task_id),
            Self::CreateTask { title, .. } => require_field(self.name(), "title", title),
            Self::InvokeTool { name, .. } => require_field(self.name(), "name", name),
            Self::ListMembers
            | Self::ReadInbox
            | Self::Broadcast { .. }
            | Self::ListTasks { .. }
            | Self::ClaimableTasks
            | Self::Snapshot
            | Self::StopTeam => Ok(()),
        }
    }
}

/// Rejects an empty (or whitespace-only) required string field.
fn require_field(op: &str, field: &str, value: &str) -> AttachResult<()> {
    if value.trim().is_empty() {
        return Err(AttachError::protocol(format!(
            "op `{op}`: field `{field}` must not be empty"
        )));
    }
    Ok(())
}

/// One client request line: `{"v":1,"id":<u64>,"op":"<op>","args":{...}}`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RequestFrame {
    /// Protocol version; must equal [`PROTOCOL_VERSION`].
    pub v: u32,
    /// Client-chosen request id, echoed by the matching response.
    pub id: u64,
    /// The op name, i.e. [`AttachOp::name`].
    pub op: String,
    /// The op arguments; always a JSON object after decoding.
    #[serde(default)]
    pub args: Value,
}

impl RequestFrame {
    /// Builds a request frame for `op`, validating the op first.
    ///
    /// # Errors
    ///
    /// Propagates [`AttachOp::validate`] and [`AttachOp::args`] failures.
    pub fn new(id: u64, op: &AttachOp) -> AttachResult<Self> {
        op.validate()?;
        Ok(Self {
            v: PROTOCOL_VERSION,
            id,
            op: op.name().to_owned(),
            args: op.args()?,
        })
    }

    /// Decodes one request line, with or without its trailing newline.
    ///
    /// # Errors
    ///
    /// Returns a `Validation` error when the line is not UTF-8, not a JSON object, missing
    /// required fields, or carries a version other than [`PROTOCOL_VERSION`]; `args` must be a
    /// JSON object or `null`.
    pub fn decode(line: &[u8]) -> AttachResult<Self> {
        let text = std::str::from_utf8(line).map_err(|error| {
            AttachError::protocol(format!("request frame is not valid UTF-8: {error}"))
        })?;
        let text = text.trim_end_matches(['\n', '\r']);
        let value: Value = serde_json::from_str(text)
            .map_err(|error| AttachError::protocol(format!("malformed request frame: {error}")))?;
        let mut frame: Self = serde_json::from_value(value)
            .map_err(|error| AttachError::protocol(format!("malformed request frame: {error}")))?;
        if frame.v != PROTOCOL_VERSION {
            return Err(AttachError::protocol(format!(
                "unsupported protocol version {}",
                frame.v
            )));
        }
        let args = std::mem::replace(&mut frame.args, Value::Object(serde_json::Map::new()));
        frame.args = match args {
            Value::Object(map) => Value::Object(map),
            Value::Null => Value::Object(serde_json::Map::new()),
            _ => {
                return Err(AttachError::protocol(
                    "request `args` must be a JSON object",
                ))
            }
        };
        Ok(frame)
    }

    /// Rebuilds the typed op carried by this frame.
    ///
    /// # Errors
    ///
    /// Propagates [`AttachOp::from_parts`] failures (unknown op, bad fields, failed validation).
    pub fn op(&self) -> AttachResult<AttachOp> {
        AttachOp::from_parts(&self.op, self.args.clone())
    }

    /// Encodes this frame as compact JSON plus one trailing `\n`.
    ///
    /// # Errors
    ///
    /// Returns a `Validation` error if serialization fails.
    pub fn encode(&self) -> AttachResult<Vec<u8>> {
        let mut bytes = serde_json::to_vec(self).map_err(|error| {
            AttachError::protocol(format!("encoding request frame failed: {error}"))
        })?;
        bytes.push(b'\n');
        Ok(bytes)
    }
}

/// One server response line, echoing the request `id`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResponseFrame {
    /// Protocol version; always [`PROTOCOL_VERSION`] on the wire.
    pub v: u32,
    /// Id of the request this response answers.
    pub id: u64,
    /// `true` for a success response, `false` for an op-level failure.
    pub ok: bool,
    /// Result payload; omitted from the wire when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
    /// Failure payload; omitted from the wire when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<AttachError>,
}

impl ResponseFrame {
    /// A success response carrying `data`.
    #[must_use]
    pub fn ok(id: u64, data: Value) -> Self {
        Self {
            v: PROTOCOL_VERSION,
            id,
            ok: true,
            data: Some(data),
            error: None,
        }
    }

    /// An op-level failure response carrying `error`.
    #[must_use]
    pub fn err(id: u64, error: AttachError) -> Self {
        Self {
            v: PROTOCOL_VERSION,
            id,
            ok: false,
            data: None,
            error: Some(error),
        }
    }

    /// Decodes one response line, with or without its trailing newline.
    ///
    /// # Errors
    ///
    /// Returns a `Validation` error when the line is not UTF-8, not a JSON object, missing
    /// required fields, carrying a version other than [`PROTOCOL_VERSION`], or claiming
    /// `ok: false` without an error payload.
    pub fn decode(line: &[u8]) -> AttachResult<Self> {
        let text = std::str::from_utf8(line).map_err(|error| {
            AttachError::protocol(format!("response frame is not valid UTF-8: {error}"))
        })?;
        let text = text.trim_end_matches(['\n', '\r']);
        let value: Value = serde_json::from_str(text)
            .map_err(|error| AttachError::protocol(format!("malformed response frame: {error}")))?;
        let frame: Self = serde_json::from_value(value)
            .map_err(|error| AttachError::protocol(format!("malformed response frame: {error}")))?;
        if frame.v != PROTOCOL_VERSION {
            return Err(AttachError::protocol(format!(
                "unsupported protocol version {}",
                frame.v
            )));
        }
        if !frame.ok && frame.error.is_none() {
            return Err(AttachError::protocol("error response carried no error"));
        }
        Ok(frame)
    }

    /// Collapses the response into a result: `data` (or JSON `null`) on success, the error otherwise.
    ///
    /// # Errors
    ///
    /// Returns the carried [`AttachError`] when `ok` is `false`; a manually built frame that is
    /// neither a valid success nor a valid error yields a `Validation` error.
    pub fn into_result(self) -> AttachResult<Value> {
        if self.ok {
            Ok(self.data.unwrap_or(Value::Null))
        } else {
            match self.error {
                Some(error) => Err(error),
                None => Err(AttachError::protocol("error response carried no error")),
            }
        }
    }

    /// Encodes this frame as compact JSON plus one trailing `\n`.
    ///
    /// `data` and `error` are omitted when they are `None`.
    ///
    /// # Errors
    ///
    /// Returns a `Validation` error if serialization fails.
    pub fn encode(&self) -> AttachResult<Vec<u8>> {
        let mut bytes = serde_json::to_vec(self).map_err(|error| {
            AttachError::protocol(format!("encoding response frame failed: {error}"))
        })?;
        bytes.push(b'\n');
        Ok(bytes)
    }
}

/// Reads one newline-terminated frame line.
///
/// Returns `Ok(None)` at EOF before any byte of a new line.
/// A frame at EOF without its newline is returned as-is, so a cleanly closed peer does not lose
/// its last frame.
///
/// # Errors
///
/// Returns a `Transport` error on read failure, and a `Validation` error when the line exceeds
/// [`MAX_FRAME_BYTES`]; the rest of an oversized line is drained so a tolerant caller can resume.
pub async fn read_frame_line<R: AsyncBufRead + Unpin>(
    reader: &mut R,
) -> AttachResult<Option<Vec<u8>>> {
    let mut line = Vec::new();
    loop {
        let available = reader
            .fill_buf()
            .await
            .map_err(|error| AttachError::transport(format!("reading frame failed: {error}")))?;
        if available.is_empty() {
            return if line.is_empty() {
                Ok(None)
            } else {
                Ok(Some(line))
            };
        }
        let newline = available.iter().position(|&byte| byte == b'\n');
        let take = newline.map_or(available.len(), |position| position + 1);
        line.extend_from_slice(&available[..take]);
        reader.consume(take);
        if line.len() > MAX_FRAME_BYTES {
            if newline.is_none() {
                discard_to_newline(reader).await?;
            }
            return Err(AttachError::protocol(format!(
                "frame exceeds {MAX_FRAME_BYTES} bytes"
            )));
        }
        if newline.is_some() {
            return Ok(Some(line));
        }
    }
}

/// Drains buffered input up to and including the next newline (or EOF).
async fn discard_to_newline<R: AsyncBufRead + Unpin>(reader: &mut R) -> AttachResult<()> {
    loop {
        let (take, finished) = {
            let available = reader.fill_buf().await.map_err(|error| {
                AttachError::transport(format!("discarding oversized frame failed: {error}"))
            })?;
            if available.is_empty() {
                return Ok(());
            }
            match available.iter().position(|&byte| byte == b'\n') {
                Some(position) => (position + 1, true),
                None => (available.len(), false),
            }
        };
        reader.consume(take);
        if finished {
            return Ok(());
        }
    }
}

/// Writes `bytes` and flushes the writer.
///
/// # Errors
///
/// Returns a `Transport` error when the write or flush fails.
pub async fn write_all_line<W: AsyncWrite + Unpin>(
    writer: &mut W,
    bytes: &[u8],
) -> AttachResult<()> {
    writer
        .write_all(bytes)
        .await
        .map_err(|error| AttachError::transport(format!("writing frame failed: {error}")))?;
    writer
        .flush()
        .await
        .map_err(|error| AttachError::transport(format!("flushing frame failed: {error}")))?;
    Ok(())
}

/// Object-safe async handler the socket server dispatches every op through.
///
/// The `skill` and `mcp` layers implement this trait over their existing client seams, so the
/// server never needs to know which of them is hosting the team.
#[async_trait]
pub trait AttachHandler: Send + Sync {
    /// Executes one op and returns its `data` payload.
    ///
    /// # Errors
    ///
    /// Return an [`AttachError`] to send an op-level failure response; the connection stays open.
    async fn handle(&self, op: AttachOp) -> AttachResult<Value>;
}

/// Serializes a handler's typed outcome into a response `data` payload.
///
/// # Errors
///
/// Returns a `Validation` error when the value cannot be represented as JSON.
pub fn to_data<T: Serialize>(value: &T) -> AttachResult<Value> {
    serde_json::to_value(value)
        .map_err(|error| AttachError::protocol(format!("encoding result failed: {error}")))
}
