//! Client half of the attach bridge: drive a hosted team over its Unix socket.
//!
//! [`AttachClient`] speaks the one-JSON-object-per-line framing of
//! [`protocol`](crate::attach::protocol) against the socket a hosting process listens on.
//! It owns exactly one connection: [`AttachClient::connect`] establishes it, every call writes one
//! request frame and reads the one response that answers it, and dropping the client closes the
//! socket.
//!
//! # Platform matrix
//!
//! Cross-process attach uses Unix domain sockets, so the transport half of this module is
//! Unix-only.
//! On Unix the client really connects.
//! On other targets the same public surface exists, but [`AttachClient::connect`] and
//! [`AttachClient::connect_with_timeout`] fail with `ErrorKind::Unsupported` (message
//! "cross-process attach requires Unix domain sockets"), every call returns that same typed error
//! and [`AttachClient::is_available`] is always `false`.
//! The typed wrappers below (`list_members` … `stop_team`) are platform independent: they only
//! delegate to [`AttachClient::call`].
//!
//! # Connect and close semantics
//!
//! Connecting is the whole availability contract: there is no handshake frame, no version
//! negotiation and no keep-alive.
//! A successful connect therefore proves only that *some* process is listening on the path — it
//! does not prove the team is healthy, nor that the peer speaks the same protocol version.
//! That is checked per exchange: a response carrying another [`PROTOCOL_VERSION`] is a protocol
//! error, and a peer that never answers trips the per-call deadline.
//!
//! Closing is implicit: dropping the client drops its socket halves, which the server observes as
//! EOF.
//! The client never attempts to resynchronise the stream.
//! A transport or framing failure leaves the connection unusable, so a caller that wants to retry
//! should drop the client and [`AttachClient::connect`] again.
//!
//! # Concurrency
//!
//! Every method takes `&mut self`, so one client carries at most one in-flight request.
//! The protocol correlates responses by request id; the client still verifies the correlation
//! (a mismatched id is a protocol error) instead of trusting the peer to answer in order.
//! Concurrent callers should use one client each rather than sharing one behind a lock.
//!
//! [`PROTOCOL_VERSION`]: crate::attach::protocol::PROTOCOL_VERSION

use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::de::DeserializeOwned;
use serde_json::Value;

#[cfg(unix)]
use tokio::net::unix::{OwnedReadHalf, OwnedWriteHalf};

#[cfg(not(unix))]
use openjiuwen_common::error::ErrorKind;

#[cfg(unix)]
use super::protocol::{read_frame_line, write_all_line, RequestFrame, ResponseFrame};
use super::protocol::{AttachError, AttachOp, AttachResult};
use crate::schema::{MemberSnapshot, TaskDetail, TaskStatus, TaskSummary, TeamSnapshot};

/// Default deadline for one request/response exchange (and for the connect attempt).
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(10);

/// Deadline applied by [`AttachClient::connect`] when establishing the socket.
pub const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// Upper bound [`AttachClient::is_available`] spends probing a socket, so a probe never stalls.
#[cfg(unix)]
const PROBE_TIMEOUT: Duration = Duration::from_secs(1);

/// A connected attach client: one socket, one in-flight request at a time.
///
/// Build it with [`AttachClient::connect`] (or [`AttachClient::connect_with_timeout`]) and drive
/// the hosted team through the typed wrappers, or through [`AttachClient::call`] for ops whose
/// payload the caller wants to interpret itself.
/// See the module documentation for the platform matrix and the connect/close contract.
#[cfg(unix)]
#[derive(Debug)]
pub struct AttachClient {
    /// Socket path this client is connected to.
    path: PathBuf,
    /// Deadline applied to one request/response exchange.
    timeout: Duration,
    /// Last request id handed out; `0` before the first call, so the first id is `1`.
    next_id: u64,
    /// Buffered read half of the socket.
    reader: tokio::io::BufReader<OwnedReadHalf>,
    /// Buffered write half of the socket.
    writer: OwnedWriteHalf,
}

/// A connected attach client: one socket, one in-flight request at a time.
///
/// Unix domain sockets are the only transport, so on this target every constructor fails with
/// `ErrorKind::Unsupported` and every call returns the same typed error.
/// The type keeps the shared API shape (path, timeout and the typed wrappers) so callers do not
/// need platform-specific code paths to compile.
#[cfg(not(unix))]
#[derive(Debug)]
pub struct AttachClient {
    /// Socket path the client would use.
    path: PathBuf,
    /// Deadline a call would apply.
    timeout: Duration,
}

#[cfg(unix)]
impl AttachClient {
    /// Connects to the attach socket at `path`, using [`CONNECT_TIMEOUT`].
    ///
    /// No handshake is exchanged: a successful connect is all the protocol requires.
    ///
    /// # Errors
    ///
    /// Returns a `Transport` error when the socket cannot be reached (missing path, refused
    /// connection, not a socket), and a `Timeout` error when [`CONNECT_TIMEOUT`] elapses first.
    pub async fn connect(path: impl AsRef<Path>) -> AttachResult<Self> {
        Self::connect_with_timeout(path, CONNECT_TIMEOUT).await
    }

    /// Connects to the attach socket at `path`, bounding the attempt with `timeout`.
    ///
    /// # Errors
    ///
    /// Returns a `Transport` error when the socket cannot be reached, and a `Timeout` error when
    /// `timeout` elapses before the connection is established.
    pub async fn connect_with_timeout(
        path: impl AsRef<Path>,
        timeout: Duration,
    ) -> AttachResult<Self> {
        let path = path.as_ref().to_path_buf();
        let stream =
            match tokio::time::timeout(timeout, tokio::net::UnixStream::connect(&path)).await {
                Ok(Ok(stream)) => stream,
                Ok(Err(error)) => {
                    return Err(AttachError::transport(format!(
                        "attach connect to {} failed: {error}",
                        path.display()
                    )))
                }
                Err(_) => {
                    return Err(AttachError::timeout(format!(
                        "attach connect to {} timed out after {}ms",
                        path.display(),
                        timeout.as_millis()
                    )))
                }
            };
        let (read_half, write_half) = stream.into_split();
        tracing::debug!(path = %path.display(), "attach client connected");
        Ok(Self {
            path,
            timeout,
            next_id: 0,
            reader: tokio::io::BufReader::new(read_half),
            writer: write_half,
        })
    }

    /// The socket path this client is connected to.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The per-call deadline in force for this client.
    #[must_use]
    pub fn timeout(&self) -> Duration {
        self.timeout
    }

    /// Sends one op and returns its `data` payload.
    ///
    /// Exactly one request frame is written and exactly one response line is read; the response id
    /// must echo the request id.
    /// The op is validated before anything is written, so an invalid op never reaches the wire.
    ///
    /// # Errors
    ///
    /// - `Validation` when [`AttachOp::validate`] rejects the op, when the response cannot be
    ///   decoded (bad JSON, wrong protocol version, oversized line) or when its id does not match.
    /// - `Transport` when the write fails or the peer closes the connection before answering.
    /// - `Timeout` when the round trip exceeds [`AttachClient::timeout`].
    /// - Whatever the remote handler reported, with its `ErrorKind` intact (for example
    ///   `NotFound` for an unknown member or task).
    pub async fn call(&mut self, op: AttachOp) -> AttachResult<Value> {
        let op_name = op.name();
        op.validate()?;
        let id = self.next_request_id();
        let bytes = RequestFrame::new(id, &op)?.encode()?;
        let timeout = self.timeout;
        let path_display = self.path.display().to_string();
        let exchange = async {
            write_all_line(&mut self.writer, &bytes).await?;
            let line = read_frame_line(&mut self.reader).await?.ok_or_else(|| {
                AttachError::transport("attach connection closed before a response arrived")
            })?;
            let response = ResponseFrame::decode(&line)?;
            if response.id != id {
                return Err(AttachError::protocol(format!(
                    "response id {} did not match request id {id}",
                    response.id
                )));
            }
            response.into_result()
        };
        match tokio::time::timeout(timeout, exchange).await {
            Ok(result) => result,
            Err(_) => Err(AttachError::timeout(format!(
                "attach request '{op_name}' to {path_display} timed out after {}ms",
                timeout.as_millis()
            ))),
        }
    }

    /// Probes whether a process is listening on the attach socket at `path`.
    ///
    /// The probe is bounded by `PROBE_TIMEOUT` (one second), connects and immediately drops the
    /// connection (no frame is sent) and never fails: a missing path, a refused connection or a
    /// closed socket all report `false`.
    /// `true` means the socket accepts connections, not that the hosted team is healthy — use
    /// [`AttachClient::snapshot`] for that.
    #[must_use]
    pub async fn is_available(path: impl AsRef<Path>) -> bool {
        let path = path.as_ref();
        tokio::time::timeout(PROBE_TIMEOUT, tokio::net::UnixStream::connect(path))
            .await
            .is_ok_and(|result| result.is_ok())
    }

    /// Hands out the next request id.
    ///
    /// Ids start at `1` and increase by one per call.
    /// The counter is a `u64`, so it wraps back to `1` (never `0`) only after about 1.8e19
    /// exchanges on one connection — unreachable in practice, but defined rather than panicking.
    fn next_request_id(&mut self) -> u64 {
        self.next_id = self.next_id.wrapping_add(1);
        if self.next_id == 0 {
            self.next_id = 1;
        }
        self.next_id
    }
}

#[cfg(not(unix))]
#[allow(clippy::unused_async)]
impl AttachClient {
    /// Fails on this target: cross-process attach requires Unix domain sockets.
    ///
    /// # Errors
    ///
    /// Always returns an `Unsupported` error with the message
    /// "cross-process attach requires Unix domain sockets".
    pub async fn connect(path: impl AsRef<Path>) -> AttachResult<Self> {
        Err(unix_only())
    }

    /// Fails on this target: cross-process attach requires Unix domain sockets.
    ///
    /// # Errors
    ///
    /// Always returns an `Unsupported` error with the message
    /// "cross-process attach requires Unix domain sockets".
    pub async fn connect_with_timeout(
        _path: impl AsRef<Path>,
        _timeout: Duration,
    ) -> AttachResult<Self> {
        Err(unix_only())
    }

    /// The socket path this client would use.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The per-call deadline in force for this client.
    #[must_use]
    pub fn timeout(&self) -> Duration {
        self.timeout
    }

    /// Fails on this target: cross-process attach requires Unix domain sockets.
    ///
    /// # Errors
    ///
    /// Always returns an `Unsupported` error with the message
    /// "cross-process attach requires Unix domain sockets".
    pub async fn call(&mut self, _op: AttachOp) -> AttachResult<Value> {
        Err(unix_only())
    }

    /// Always `false` on this target: there is no attach transport to probe.
    #[must_use]
    pub async fn is_available(_path: impl AsRef<Path>) -> bool {
        false
    }
}

/// The typed-error report of an unavailable attach transport on non-Unix targets.
#[cfg(not(unix))]
fn unix_only() -> AttachError {
    AttachError::remote(
        ErrorKind::Unsupported,
        "cross-process attach requires Unix domain sockets",
    )
}

/// Typed wrappers over [`AttachClient::call`], decoded against the documented payload shapes.
///
/// Each wrapper sends exactly one request, so a wrapper failure follows the same rules as
/// [`AttachClient::call`]; an extra `Validation` error reports a payload that does not match the
/// shape documented in [`protocol`](crate::attach::protocol).
impl AttachClient {
    /// Lists the roster: one [`MemberSnapshot`] per member, in the server's order.
    ///
    /// # Errors
    ///
    /// Propagates [`AttachClient::call`] failures and reports an unexpected payload as
    /// `Validation`.
    pub async fn list_members(&mut self) -> AttachResult<Vec<MemberSnapshot>> {
        decode("list_members", self.call(AttachOp::ListMembers).await?)
    }

    /// Reads one member's roster row.
    ///
    /// # Errors
    ///
    /// Propagates [`AttachClient::call`] failures (including `NotFound` for an unknown member) and
    /// reports an unexpected payload as `Validation`.
    pub async fn member_status(&mut self, member_name: &str) -> AttachResult<MemberSnapshot> {
        let op = AttachOp::MemberStatus {
            member_name: member_name.to_owned(),
        };
        decode("member_status", self.call(op).await?)
    }

    /// Reads the caller's inbox as rendered text.
    ///
    /// # Errors
    ///
    /// Propagates [`AttachClient::call`] failures and reports an unexpected payload as
    /// `Validation`.
    pub async fn read_inbox(&mut self) -> AttachResult<String> {
        decode("read_inbox", self.call(AttachOp::ReadInbox).await?)
    }

    /// Sends a direct message to one member.
    ///
    /// The payload is the adapter's own `{"ok", "message_id", "reason"}` object, returned as raw
    /// JSON.
    ///
    /// # Errors
    ///
    /// Propagates [`AttachClient::call`] failures.
    pub async fn send_member(&mut self, to: &str, content: &str) -> AttachResult<Value> {
        let op = AttachOp::SendMember {
            to: to.to_owned(),
            content: content.to_owned(),
        };
        self.call(op).await
    }

    /// Sends a message to every member.
    ///
    /// The payload is the adapter's own `{"ok", "message_id", "reason"}` object, returned as raw
    /// JSON.
    ///
    /// # Errors
    ///
    /// Propagates [`AttachClient::call`] failures.
    pub async fn broadcast(&mut self, content: &str) -> AttachResult<Value> {
        let op = AttachOp::Broadcast {
            content: content.to_owned(),
        };
        self.call(op).await
    }

    /// Lists tasks, optionally restricted to one status.
    ///
    /// # Errors
    ///
    /// Propagates [`AttachClient::call`] failures and reports an unexpected payload as
    /// `Validation`.
    pub async fn list_tasks(
        &mut self,
        status: Option<TaskStatus>,
    ) -> AttachResult<Vec<TaskSummary>> {
        decode(
            "list_tasks",
            self.call(AttachOp::ListTasks { status }).await?,
        )
    }

    /// Lists the tasks the caller may claim right now.
    ///
    /// # Errors
    ///
    /// Propagates [`AttachClient::call`] failures and reports an unexpected payload as
    /// `Validation`.
    pub async fn claimable_tasks(&mut self) -> AttachResult<Vec<TaskSummary>> {
        decode(
            "claimable_tasks",
            self.call(AttachOp::ClaimableTasks).await?,
        )
    }

    /// Reads one task in full, or `None` when the server has no such task.
    ///
    /// # Errors
    ///
    /// Propagates [`AttachClient::call`] failures and reports an unexpected payload as
    /// `Validation`.
    pub async fn task_detail(&mut self, task_id: &str) -> AttachResult<Option<TaskDetail>> {
        let op = AttachOp::TaskDetail {
            task_id: task_id.to_owned(),
        };
        decode("task_detail", self.call(op).await?)
    }

    /// Creates a task, optionally with an explicit id and dependencies.
    ///
    /// The payload is the adapter's own `{"ok", "task_id", "reason"}` object, returned as raw JSON.
    ///
    /// # Errors
    ///
    /// Propagates [`AttachClient::call`] failures.
    pub async fn create_task(
        &mut self,
        title: &str,
        content: &str,
        task_id: Option<&str>,
        dependencies: &[String],
    ) -> AttachResult<Value> {
        let op = AttachOp::CreateTask {
            title: title.to_owned(),
            content: content.to_owned(),
            task_id: task_id.map(str::to_owned),
            dependencies: dependencies.to_vec(),
        };
        self.call(op).await
    }

    /// Updates a task's title and/or content; `None` leaves that field unchanged.
    ///
    /// The payload is the adapter's own `{"ok", "reason"}` object, returned as raw JSON.
    ///
    /// # Errors
    ///
    /// Propagates [`AttachClient::call`] failures.
    pub async fn update_task(
        &mut self,
        task_id: &str,
        title: Option<&str>,
        content: Option<&str>,
    ) -> AttachResult<Value> {
        let op = AttachOp::UpdateTask {
            task_id: task_id.to_owned(),
            title: title.map(str::to_owned),
            content: content.map(str::to_owned),
        };
        self.call(op).await
    }

    /// Claims a pending task for the caller.
    ///
    /// The payload is the adapter's own `{"ok", "reason"}` object, returned as raw JSON.
    ///
    /// # Errors
    ///
    /// Propagates [`AttachClient::call`] failures.
    pub async fn claim_task(&mut self, task_id: &str) -> AttachResult<Value> {
        let op = AttachOp::ClaimTask {
            task_id: task_id.to_owned(),
        };
        self.call(op).await
    }

    /// Marks a claimed task complete.
    ///
    /// The payload is the adapter's own `{"ok", "reason"}` object, returned as raw JSON.
    ///
    /// # Errors
    ///
    /// Propagates [`AttachClient::call`] failures.
    pub async fn complete_task(&mut self, task_id: &str) -> AttachResult<Value> {
        let op = AttachOp::CompleteTask {
            task_id: task_id.to_owned(),
        };
        self.call(op).await
    }

    /// Invokes a tool exposed by the hosting team and returns the tool's own result JSON.
    ///
    /// # Errors
    ///
    /// Propagates [`AttachClient::call`] failures, including the remote tool error kind when the
    /// tool itself failed.
    pub async fn invoke_tool(&mut self, name: &str, arguments: Value) -> AttachResult<Value> {
        let op = AttachOp::InvokeTool {
            name: name.to_owned(),
            arguments,
        };
        self.call(op).await
    }

    /// Reads a full team snapshot: roster plus task board.
    ///
    /// # Errors
    ///
    /// Propagates [`AttachClient::call`] failures and reports an unexpected payload as
    /// `Validation`.
    pub async fn snapshot(&mut self) -> AttachResult<TeamSnapshot> {
        decode("snapshot", self.call(AttachOp::Snapshot).await?)
    }

    /// Asks the hosting process to stop the team.
    ///
    /// The payload is the adapter's own `{"stopping": true}` object, returned as raw JSON: the
    /// handler cancels the team and acknowledges the request, so the object says the stop is under
    /// way rather than that it has finished.
    ///
    /// # Errors
    ///
    /// Propagates [`AttachClient::call`] failures.
    pub async fn stop_team(&mut self) -> AttachResult<Value> {
        self.call(AttachOp::StopTeam).await
    }
}

/// Decodes an op's `data` payload into its documented type.
fn decode<T: DeserializeOwned>(op: &str, data: Value) -> AttachResult<T> {
    serde_json::from_value(data).map_err(|error| {
        AttachError::protocol(format!("op '{op}' returned an unexpected payload: {error}"))
    })
}
