//! Server half of the attach bridge: bind a Unix socket and answer remote clients.
//!
//! [`AttachServer`] is the hosting side of the contract in [`protocol`](crate::attach::protocol).
//! [`AttachServer::bind`](crate::attach::server::AttachServer::bind) owns the socket file;
//! [`AttachServer::serve`](crate::attach::server::AttachServer::serve) accepts connections and
//! dispatches every request frame through an
//! [`AttachHandler`];
//! [`AttachServer::shutdown`](crate::attach::server::AttachServer::shutdown) stops the accept loop
//! and dropping the server removes the socket file.
//!
//! It is the counterpart of [`AttachClient`](crate::attach::AttachClient): the client opens one
//! connection and sends one request at a time, the server here answers exactly one response per
//! request and serves every connection concurrently.
//!
//! # Binding
//!
//! `bind` runs these steps in order:
//!
//! 1. The path must be non-empty and must not be a directory, else `ErrorKind::Validation`.
//! 2. Missing parent directories are created.
//! 3. An existing path is probed with a one-second `connect`.
//!    A successful probe means another process is listening, so binding fails with
//!    `ErrorKind::Config` ("attach socket already in use: `/some/team.sock`").
//!    A refused (or timed-out) probe means the file was left behind by a dead process, so the stale
//!    socket file is removed and reclaimed.
//!    A path that exists but is not a socket at all (for example a regular file that replaced the
//!    socket) is never removed; it is refused with `ErrorKind::Config` instead.
//!    Any other probe error is reported as `ErrorKind::Config` naming the OS error.
//! 4. The listener binds the path.
//! 5. The socket file's mode is set to `0600`, so only the owning user can connect.
//!    If that fails the freshly bound file is removed again and `ErrorKind::SysOp` is returned,
//!    because the socket would otherwise be usable by other users.
//!
//! # Serving
//!
//! `serve` accepts connections until the server's [`CancellationToken`] is cancelled; each accepted
//! stream is served by its own task, so a slow handler blocks only its own connection.
//! A failed `accept` is logged and does not end the loop.
//!
//! Per connection the server reads one request line at a time and writes exactly one response
//! echoing the request `id`:
//!
//! - A handler failure is an *op-level* error: the response carries it with its `ErrorKind` intact
//!   and the connection stays open for the next request.
//! - A frame-level fault ends the connection after one typed error response: a read failure or an
//!   oversized line (answered with id `0`), a malformed frame or an unsupported protocol version
//!   (answered with the `id` the line still carries, `0` when it cannot be recovered), and an
//!   unknown or invalid op (answered with the frame's `id`).
//!   Treating an unknown op as fatal is deliberate: a peer that sends one is not speaking this
//!   protocol version, so the stream can no longer be trusted to stay in sync.
//! - A connection that closes without sending anything is a clean EOF: it is logged at `debug`
//!   level and never answered, because probing for availability works exactly that way.
//!
//! On shutdown the accept loop stops, connections observe the same token and close after finishing
//! the frame they are on, and the socket file is removed.
//! In-flight handler calls are allowed to complete; the wait is capped by a short grace period
//! after which lingering connection tasks are aborted.
//!
//! # Dropping
//!
//! Dropping a server removes the socket file, which also covers a server that was bound but never
//! served.
//! The removal is deliberately conservative: it only deletes a path that still exists *and* is
//! still a socket, so a regular file or directory that replaced the socket is left untouched.
//!
//! # Platform matrix
//!
//! Cross-process attach uses Unix domain sockets, so the socket half of this module is Unix-only.
//! On other targets the same public surface exists, but
//! [`AttachServer::bind`](crate::attach::server::AttachServer::bind) and
//! [`AttachServer::serve`](crate::attach::server::AttachServer::serve) return `ErrorKind::Unsupported`
//! with the message "cross-process attach requires Unix domain sockets", `shutdown` is a no-op and
//! `cancellation_token` hands out a fresh token.
//! No path is ever bound there, so there is no drop cleanup and the path accessor only reports the
//! path the caller asked for.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use openjiuwen_common::error::{Error, Result};
use tokio_util::sync::CancellationToken;

use super::protocol::AttachHandler;

#[cfg(unix)]
use std::io::ErrorKind as IoErrorKind;
#[cfg(unix)]
use std::os::unix::fs::{FileTypeExt, PermissionsExt};
#[cfg(unix)]
use std::time::Duration;

#[cfg(unix)]
use tokio::io::BufReader;
#[cfg(unix)]
use tokio::net::unix::OwnedWriteHalf;
#[cfg(unix)]
use tokio::net::{UnixListener, UnixStream};

#[cfg(unix)]
use super::protocol::{read_frame_line, write_all_line, AttachResult, RequestFrame, ResponseFrame};

/// Deadline for probing an existing socket file during [`AttachServer::bind`].
#[cfg(unix)]
const PROBE_TIMEOUT: Duration = Duration::from_secs(1);

/// How long [`AttachServer::serve`] waits for connection tasks after shutdown.
#[cfg(unix)]
const SHUTDOWN_GRACE: Duration = Duration::from_secs(2);

/// Mode the socket file is created with: owner read and write only.
#[cfg(unix)]
const SOCKET_MODE: u32 = 0o600;

/// Hosting end of the attach bridge: a bound Unix socket serving one team.
///
/// Binding creates the socket file, serving answers requests through an [`AttachHandler`], and
/// dropping the value removes the socket file.
/// One server owns one socket path; connections are accepted and served concurrently, each with at
/// most one in-flight request.
#[cfg(unix)]
#[derive(Debug)]
pub struct AttachServer {
    /// Socket path this server bound.
    path: PathBuf,
    /// Listener feeding the accept loop.
    listener: UnixListener,
    /// Token that stops the accept loop and closes idle connections.
    cancel: CancellationToken,
}

#[cfg(unix)]
impl AttachServer {
    /// Binds the attach socket at `path`.
    ///
    /// Creates missing parent directories, refuses to clobber a live socket, reclaims a stale
    /// socket file left behind by a dead process, and sets the socket mode to `0600`.
    /// The bind steps are described in the module documentation.
    ///
    /// # Errors
    ///
    /// - `Validation` when `path` is empty or names an existing directory.
    /// - `Config` when another process is listening on `path`, when the path is an existing file
    ///   that is not a socket, or when the liveness probe fails for another reason; the message
    ///   names the OS error or the path.
    /// - `SysOp` when the parent directory cannot be created, a stale socket cannot be removed, or
    ///   the socket mode cannot be set.
    /// - `Transport` when the listener itself cannot be bound.
    pub async fn bind(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        validate_path(&path).await?;
        ensure_parent(&path).await?;
        reclaim_stale(&path).await?;
        let listener = UnixListener::bind(&path)
            .map_err(|error| Error::transport(format!("attach socket error: {error}")))?;
        if let Err(error) = set_socket_mode(&path) {
            // A world-readable socket is worse than no socket, so undo the bind.
            let _ = std::fs::remove_file(&path);
            return Err(error);
        }
        tracing::info!(path = %path.display(), "attach server listening");
        Ok(Self {
            path,
            listener,
            cancel: CancellationToken::new(),
        })
    }

    /// The socket path this server is bound to.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Stops the accept loop.
    ///
    /// The call returns immediately; a running [`AttachServer::serve`] observes the cancellation,
    /// lets in-flight connections finish the frame they are on, and then returns.
    /// Calling it more than once is harmless.
    pub fn shutdown(&self) {
        self.cancel.cancel();
    }

    /// The token that stops this server.
    ///
    /// Cancelling a clone has the same effect as [`AttachServer::shutdown`], which makes it usable
    /// as a shutdown signal for a supervising task.
    #[must_use]
    pub fn cancellation_token(&self) -> CancellationToken {
        self.cancel.clone()
    }

    /// Serves the attach socket until the server is shut down.
    ///
    /// Every accepted connection is served by its own task, so connections do not block each
    /// other; the `handler` is shared between them and must therefore be `Send + Sync`.
    /// An `accept` failure is logged and the loop continues.
    /// When the token is cancelled the loop stops, connections close (finishing the frame they are
    /// on), and the socket file is removed.
    ///
    /// # Errors
    ///
    /// Returns `Ok(())` for a clean shutdown.
    /// No error is currently returned: failed accept and cleanup operations are logged instead, so
    /// a supervisor only has to distinguish "stopped" from "panicked".
    pub async fn serve(&self, handler: Arc<dyn AttachHandler>) -> Result<()> {
        let mut tasks: tokio::task::JoinSet<()> = tokio::task::JoinSet::new();
        loop {
            tokio::select! {
                () = self.cancel.cancelled() => {
                    tracing::debug!(path = %self.path.display(), "attach server stopping");
                    break;
                }
                accepted = self.listener.accept() => {
                    match accepted {
                        Ok((stream, _peer)) => {
                            let handler = Arc::clone(&handler);
                            let cancel = self.cancel.clone();
                            tasks.spawn(async move {
                                serve_connection(stream, handler, cancel).await;
                            });
                            while tasks.try_join_next().is_some() {}
                        }
                        Err(error) => {
                            if self.cancel.is_cancelled() {
                                break;
                            }
                            tracing::warn!(error = %error, "attach accept failed");
                        }
                    }
                }
            }
        }
        drain(&mut tasks).await;
        remove_bound_socket(&self.path).await;
        tracing::debug!(path = %self.path.display(), "attach server stopped");
        Ok(())
    }
}

#[cfg(unix)]
impl Drop for AttachServer {
    fn drop(&mut self) {
        self.cancel.cancel();
        drop_bound_socket(&self.path);
    }
}

/// Rejects a path that cannot name a socket file.
#[cfg(unix)]
async fn validate_path(path: &Path) -> Result<()> {
    if path.as_os_str().is_empty() {
        return Err(Error::validation("attach socket path must not be empty"));
    }
    match tokio::fs::metadata(path).await {
        Ok(metadata) if metadata.is_dir() => Err(Error::validation(format!(
            "attach socket path {} is a directory",
            path.display()
        ))),
        Ok(_) => Ok(()),
        Err(error) if error.kind() == IoErrorKind::NotFound => Ok(()),
        Err(error) => Err(Error::sys_op(format!(
            "inspecting attach socket path {} failed: {error}",
            path.display()
        ))),
    }
}

/// Creates the socket's parent directories when the path has any.
#[cfg(unix)]
async fn ensure_parent(path: &Path) -> Result<()> {
    let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    else {
        return Ok(());
    };
    tokio::fs::create_dir_all(parent).await.map_err(|error| {
        Error::sys_op(format!(
            "creating attach socket directory {} failed: {error}",
            parent.display()
        ))
    })
}

/// Reclaims a stale socket file, refusing to touch a live socket or a foreign file.
#[cfg(unix)]
async fn reclaim_stale(path: &Path) -> Result<()> {
    match tokio::fs::symlink_metadata(path).await {
        Ok(metadata) => reclaim_existing(path, &metadata).await,
        Err(error) if error.kind() == IoErrorKind::NotFound => Ok(()),
        Err(error) => Err(Error::sys_op(format!(
            "inspecting attach socket {} failed: {error}",
            path.display()
        ))),
    }
}

/// Decides whether an existing path is a stale socket that may be removed.
#[cfg(unix)]
async fn reclaim_existing(path: &Path, metadata: &std::fs::Metadata) -> Result<()> {
    if !metadata.file_type().is_socket() {
        return Err(Error::config(format!(
            "attach socket path {} exists and is not a socket",
            path.display()
        )));
    }
    match tokio::time::timeout(PROBE_TIMEOUT, UnixStream::connect(path)).await {
        Ok(Ok(stream)) => {
            drop(stream);
            Err(Error::config(format!(
                "attach socket already in use: {}",
                path.display()
            )))
        }
        Ok(Err(error)) if is_stale_probe(&error) => remove_stale(path).await,
        Ok(Err(error)) => Err(Error::config(format!(
            "probing attach socket {} failed: {error}",
            path.display()
        ))),
        Err(_elapsed) => {
            tracing::debug!(
                path = %path.display(),
                "attach socket probe timed out; reclaiming the path"
            );
            remove_stale(path).await
        }
    }
}

/// Reports whether a failed probe means "the listening process is gone".
#[cfg(unix)]
fn is_stale_probe(error: &std::io::Error) -> bool {
    matches!(
        error.kind(),
        IoErrorKind::ConnectionRefused | IoErrorKind::NotFound
    )
}

/// Removes a stale socket file, tolerating one that vanished in the meantime.
#[cfg(unix)]
async fn remove_stale(path: &Path) -> Result<()> {
    match tokio::fs::remove_file(path).await {
        Ok(()) => {
            tracing::debug!(path = %path.display(), "reclaimed stale attach socket");
            Ok(())
        }
        Err(error) if error.kind() == IoErrorKind::NotFound => Ok(()),
        Err(error) => Err(Error::sys_op(format!(
            "removing stale attach socket {} failed: {error}",
            path.display()
        ))),
    }
}

/// Sets the socket file to `0600`.
#[cfg(unix)]
fn set_socket_mode(path: &Path) -> Result<()> {
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(SOCKET_MODE)).map_err(|error| {
        Error::sys_op(format!(
            "setting mode {SOCKET_MODE:o} on attach socket {} failed: {error}",
            path.display()
        ))
    })
}

/// Waits for connection tasks after shutdown, aborting the stragglers.
#[cfg(unix)]
async fn drain(tasks: &mut tokio::task::JoinSet<()>) {
    let drained = tokio::time::timeout(SHUTDOWN_GRACE, async {
        while tasks.join_next().await.is_some() {}
    })
    .await;
    if drained.is_ok() {
        return;
    }
    tasks.abort_all();
    while tasks.join_next().await.is_some() {}
    tracing::debug!("attach server aborted lingering connections");
}

/// Serves one connection until EOF, a frame-level fault, or shutdown.
#[cfg(unix)]
async fn serve_connection(
    stream: UnixStream,
    handler: Arc<dyn AttachHandler>,
    cancel: CancellationToken,
) {
    let (read_half, mut write_half) = stream.into_split();
    let mut reader = BufReader::new(read_half);
    loop {
        let read = tokio::select! {
            () = cancel.cancelled() => {
                tracing::debug!("attach connection closed on shutdown");
                return;
            }
            read = read_frame_line(&mut reader) => read,
        };
        let line = match read {
            Ok(Some(line)) => line,
            Ok(None) => {
                tracing::debug!("attach connection closed by the client");
                return;
            }
            Err(error) => {
                tracing::debug!(kind = %error.kind, "attach frame read failed");
                let _ = send_frame(&mut write_half, ResponseFrame::err(0, error)).await;
                return;
            }
        };
        let frame = match RequestFrame::decode(&line) {
            Ok(frame) => frame,
            Err(error) => {
                let id = frame_id(&line).unwrap_or(0);
                tracing::debug!(id, kind = %error.kind, "attach request frame rejected");
                let _ = send_frame(&mut write_half, ResponseFrame::err(id, error)).await;
                return;
            }
        };
        let op = match frame.op() {
            Ok(op) => op,
            Err(error) => {
                tracing::debug!(
                    id = frame.id,
                    op = %frame.op,
                    kind = %error.kind,
                    "attach request op rejected"
                );
                let _ = send_frame(&mut write_half, ResponseFrame::err(frame.id, error)).await;
                return;
            }
        };
        let response = match handler.handle(op).await {
            Ok(data) => ResponseFrame::ok(frame.id, data),
            Err(error) => {
                tracing::debug!(id = frame.id, kind = %error.kind, "attach handler failed");
                ResponseFrame::err(frame.id, error)
            }
        };
        if let Err(error) = send_frame(&mut write_half, response).await {
            tracing::debug!(kind = %error.kind, "attach response write failed");
            return;
        }
    }
}

/// Encodes and writes one response frame.
#[cfg(unix)]
async fn send_frame(writer: &mut OwnedWriteHalf, frame: ResponseFrame) -> AttachResult<()> {
    let bytes = frame.encode()?;
    write_all_line(writer, &bytes).await
}

/// Recovers the `id` of a request line that did not decode, so the client can still match it.
#[cfg(unix)]
fn frame_id(line: &[u8]) -> Option<u64> {
    let text = std::str::from_utf8(line).ok()?;
    let value: serde_json::Value =
        serde_json::from_str(text.trim_end_matches(['\n', '\r'])).ok()?;
    value.get("id").and_then(serde_json::Value::as_u64)
}

/// Removes the socket file after a clean shutdown, ignoring a path that is already gone.
#[cfg(unix)]
async fn remove_bound_socket(path: &Path) {
    match tokio::fs::remove_file(path).await {
        Ok(()) => tracing::debug!(path = %path.display(), "attach socket removed"),
        Err(error) if error.kind() == IoErrorKind::NotFound => {}
        Err(error) => {
            tracing::warn!(path = %path.display(), error = %error, "removing attach socket failed");
        }
    }
}

/// Removes the socket file on drop, but never a path that is no longer a socket.
#[cfg(unix)]
fn drop_bound_socket(path: &Path) {
    let Ok(metadata) = std::fs::symlink_metadata(path) else {
        return;
    };
    if !metadata.file_type().is_socket() {
        tracing::debug!(
            path = %path.display(),
            "attach socket path was replaced; leaving it in place"
        );
        return;
    }
    match std::fs::remove_file(path) {
        Ok(()) => tracing::debug!(path = %path.display(), "attach socket removed on drop"),
        Err(error) => {
            tracing::debug!(path = %path.display(), error = %error, "removing attach socket failed");
        }
    }
}

/// Hosting end of the attach bridge, unavailable on this target.
///
/// Unix domain sockets are the only attach transport, so
/// [`AttachServer::bind`](crate::attach::server::AttachServer::bind) and
/// [`AttachServer::serve`](crate::attach::server::AttachServer::serve) always fail with
/// `ErrorKind::Unsupported`.
/// The type keeps the shared API shape so callers do not need platform-specific code paths to
/// compile, and no path is ever bound, so there is nothing to clean up on drop.
#[cfg(not(unix))]
#[derive(Debug)]
pub struct AttachServer {
    /// Socket path a bound server would own.
    path: PathBuf,
}

#[cfg(not(unix))]
#[allow(clippy::unused_async)]
impl AttachServer {
    /// Fails on this target: cross-process attach requires Unix domain sockets.
    ///
    /// # Errors
    ///
    /// Always returns an `Unsupported` error with the message
    /// "cross-process attach requires Unix domain sockets".
    pub async fn bind(_path: impl AsRef<Path>) -> Result<Self> {
        Err(unix_only())
    }

    /// The socket path a bound server would own.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// No-op on this target: there is never an accept loop to stop.
    pub fn shutdown(&self) {}

    /// Hands out a fresh token on this target: there is no accept loop to cancel.
    #[must_use]
    pub fn cancellation_token(&self) -> CancellationToken {
        CancellationToken::new()
    }

    /// Fails on this target: cross-process attach requires Unix domain sockets.
    ///
    /// # Errors
    ///
    /// Always returns an `Unsupported` error with the message
    /// "cross-process attach requires Unix domain sockets".
    pub async fn serve(&self, _handler: Arc<dyn AttachHandler>) -> Result<()> {
        Err(unix_only())
    }
}

/// The typed-error report of an unavailable attach transport on non-Unix targets.
#[cfg(not(unix))]
fn unix_only() -> Error {
    Error::unsupported("cross-process attach requires Unix domain sockets")
}
