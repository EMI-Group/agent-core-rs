//! The transport seam and its process and in-memory implementations.
//!
//! Ports the stdin/stdout JSON-RPC channel of `openjiuwen/harness/lsp/core/client.py`.
//! [`LspTransport`] is the only place the client touches IO, so tests inject
//! [`ChannelTransport`] (an in-memory [`tokio::io::duplex`] pair) and never spawn a language
//! server. [`StdioTransport`] launches a configured process and drains its stderr into
//! `tracing` so a chatty server cannot block the pipe.
use std::process::Stdio;

use async_trait::async_trait;
use serde_json::Value;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, DuplexStream};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};

use openjiuwen_common::error::{Error, Result};

use super::config::LspServerConfig;
use super::protocol::{write_message, MessageReader};

/// A bidirectional JSON-RPC message channel.
///
/// Implementations own their read/write state; the client drives them from one I/O task, so
/// `send` and `recv` are never called concurrently on the same value.
#[async_trait]
pub trait LspTransport: Send {
    /// Sends one already-encoded JSON-RPC message.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Transport`] when the channel is closed or the write fails.
    async fn send(&mut self, message: Value) -> Result<()>;

    /// Receives the next JSON-RPC message, or `None` on a clean end of stream.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Transport`] on read failure and [`Error::Validation`] on a malformed
    /// frame.
    async fn recv(&mut self) -> Result<Option<Value>>;

    /// Closes the channel, releasing any process or socket it owns.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Transport`] when the shutdown itself fails; callers treat shutdown
    /// failures as best effort.
    async fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

/// A language server launched as a child process.
///
/// The child is killed when this value is dropped, matching Python's `terminate()` on the
/// manager's teardown path, so a leaked client cannot leak a server.
pub struct StdioTransport {
    child: Child,
    stdin: Option<ChildStdin>,
    reader: MessageReader<ChildStdout>,
}

impl StdioTransport {
    /// Launches the server described by `config` and wires its stdin/stdout.
    ///
    /// The child gets piped stdin/stdout/stderr and `kill_on_drop`; stderr is drained on a
    /// background task so verbose servers cannot fill the pipe.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] when `command` is empty or the process cannot be spawned,
    /// and [`Error::Transport`] when the child's pipes are unavailable.
    pub fn spawn(config: &LspServerConfig) -> Result<Self> {
        if config.command.trim().is_empty() {
            return Err(Error::config("LSP server command is empty"));
        }
        let mut command = Command::new(&config.command);
        command
            .args(&config.args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        if let Some(root) = &config.root {
            command.current_dir(root);
        }
        for (key, value) in &config.env {
            command.env(key, value);
        }
        let mut child = command.spawn().map_err(|error| {
            Error::config(format!(
                "failed to start LSP server '{}': {error}",
                config.command
            ))
        })?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| Error::transport("LSP server stdin is unavailable"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| Error::transport("LSP server stdout is unavailable"))?;
        if let Some(stderr) = child.stderr.take() {
            tokio::spawn(async move {
                let mut lines = BufReader::new(stderr).lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    tracing::debug!(target: "openjiuwen_harness::lsp", "lsp server stderr: {line}");
                }
            });
        }
        Ok(Self {
            child,
            stdin: Some(stdin),
            reader: MessageReader::new(stdout),
        })
    }

    /// Returns the operating-system process id of the server, when it is still running.
    #[must_use]
    pub fn process_id(&self) -> Option<u32> {
        self.child.id()
    }
}

#[async_trait]
impl LspTransport for StdioTransport {
    async fn send(&mut self, message: Value) -> Result<()> {
        let stdin = self
            .stdin
            .as_mut()
            .ok_or_else(|| Error::transport("LSP server stdin is closed"))?;
        write_message(stdin, &message).await
    }

    async fn recv(&mut self) -> Result<Option<Value>> {
        self.reader.next().await
    }

    async fn close(&mut self) -> Result<()> {
        self.stdin.take();
        let _ = self.child.start_kill();
        self.child
            .wait()
            .await
            .map_err(|error| Error::transport(format!("failed to reap LSP server: {error}")))?;
        Ok(())
    }
}

impl Drop for StdioTransport {
    fn drop(&mut self) {
        let _ = self.child.start_kill();
    }
}

/// An in-memory transport over one half of a [`tokio::io::duplex`] pair.
///
/// [`ChannelTransport::new`] returns the client transport plus the peer stream: tests spawn a
/// fake server task over the peer. This keeps every client and tool test offline and able to
/// script slow responses, malformed frames, and end-of-stream.
pub struct ChannelTransport {
    writer: tokio::io::WriteHalf<DuplexStream>,
    reader: MessageReader<tokio::io::ReadHalf<DuplexStream>>,
}

impl ChannelTransport {
    /// Creates a connected transport pair with an internal buffer of `buffer` bytes.
    ///
    /// The returned [`DuplexStream`] is the server side of the channel.
    #[must_use]
    pub fn new(buffer: usize) -> (Self, DuplexStream) {
        let (client, server) = tokio::io::duplex(buffer.max(1));
        let (reader, writer) = tokio::io::split(client);
        (
            Self {
                writer,
                reader: MessageReader::new(reader),
            },
            server,
        )
    }
}

#[async_trait]
impl LspTransport for ChannelTransport {
    async fn send(&mut self, message: Value) -> Result<()> {
        write_message(&mut self.writer, &message).await
    }

    async fn recv(&mut self) -> Result<Option<Value>> {
        self.reader.next().await
    }

    async fn close(&mut self) -> Result<()> {
        self.writer
            .shutdown()
            .await
            .map_err(|error| Error::transport(format!("failed to close LSP channel: {error}")))
    }
}
