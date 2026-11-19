//! One wire connection to a `Redis` node: transport, handshake, and reply reads.
//!
//! The transport is either plain TCP or TLS ([`super::tls`]); both share the same
//! connection logic through [`Transport`], which implements [`tokio::io::AsyncRead`]
//! and [`tokio::io::AsyncWrite`]. A connection reconnects lazily: any transport
//! failure or timeout drops the socket and the next command re-runs the handshake.
//!
//! # Cancellation and desynchronisation
//!
//! `RESP` has no request ids, so a reply is only attributable to a command while
//! the command is in flight. [`Connection::exchange`] therefore closes the
//! connection whenever its future is dropped before the replies have been read —
//! a caller-side `timeout`, a `select!` arm, or a `JoinHandle` abort — so the
//! next command reconnects instead of reading the aborted command's reply. Bytes
//! left in the buffer by anything else (a peer that answered more than it was
//! asked) are treated as desynchronised too: the connection is closed and the
//! command fails with a typed `Transport` error rather than reusing the wire.
use super::tls;
use super::ConnectionOptions;
use crate::redis::codec::{encode_command, parse_reply, Reply};
use openjiuwen_common::error::{Error, Result};
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Duration;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt, ReadBuf};
use tokio::net::TcpStream;
use tokio_rustls::client::TlsStream;

/// Maximum bytes buffered for one in-flight reply before the connection is dropped.
const MAX_REPLY_BYTES: usize = 512 * 1024 * 1024;

/// The socket backing a connection: plain TCP or a TLS session over TCP.
pub(super) enum Transport {
    /// `redis://`
    Plain(TcpStream),
    /// `rediss://`
    Tls(Box<TlsStream<TcpStream>>),
}

impl Transport {
    /// The transport kind, used in log and error messages.
    pub(super) const fn kind(&self) -> &'static str {
        match self {
            Self::Plain(_) => "tcp",
            Self::Tls(_) => "tls",
        }
    }
}

impl AsyncRead for Transport {
    fn poll_read(
        self: Pin<&mut Self>,
        context: &mut Context<'_>,
        buffer: &mut ReadBuf<'_>,
    ) -> Poll<std::io::Result<()>> {
        match self.get_mut() {
            Self::Plain(stream) => Pin::new(stream).poll_read(context, buffer),
            Self::Tls(stream) => Pin::new(stream.as_mut()).poll_read(context, buffer),
        }
    }
}

impl AsyncWrite for Transport {
    fn poll_write(
        self: Pin<&mut Self>,
        context: &mut Context<'_>,
        buffer: &[u8],
    ) -> Poll<std::io::Result<usize>> {
        match self.get_mut() {
            Self::Plain(stream) => Pin::new(stream).poll_write(context, buffer),
            Self::Tls(stream) => Pin::new(stream.as_mut()).poll_write(context, buffer),
        }
    }

    fn poll_flush(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        match self.get_mut() {
            Self::Plain(stream) => Pin::new(stream).poll_flush(context),
            Self::Tls(stream) => Pin::new(stream.as_mut()).poll_flush(context),
        }
    }

    fn poll_shutdown(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        match self.get_mut() {
            Self::Plain(stream) => Pin::new(stream).poll_shutdown(context),
            Self::Tls(stream) => Pin::new(stream.as_mut()).poll_shutdown(context),
        }
    }
}

/// A lazily opened connection to one `Redis` node.
#[derive(Default)]
pub(super) struct Connection {
    stream: Option<Transport>,
    buffer: Vec<u8>,
}

impl Connection {
    /// Connects and runs the handshake when the connection is not open.
    ///
    /// The socket is published to `self` only once `AUTH`/`SELECT` have been
    /// answered, so a cancellation in the middle of the handshake leaves the
    /// connection closed instead of marking an unauthenticated socket open.
    pub(super) async fn ensure_open(&mut self, options: &ConnectionOptions) -> Result<()> {
        if self.stream.is_some() {
            return Ok(());
        }
        // A closed connection never carries bytes forward: anything still here
        // belongs to a command whose reply was abandoned, so it cannot be
        // attributed to the command the caller is about to send.
        self.buffer.clear();
        let stream = connect_transport(options).await?;
        tracing::debug!(
            host = %options.host,
            port = options.port,
            transport = stream.kind(),
            "redis transport open"
        );
        let mut staging = Self {
            stream: Some(stream),
            buffer: Vec::new(),
        };
        staging.handshake(options).await?;
        self.stream = staging.stream.take();
        self.buffer = std::mem::take(&mut staging.buffer);
        Ok(())
    }

    /// Runs `AUTH` and `SELECT` for a freshly opened transport.
    async fn handshake(&mut self, options: &ConnectionOptions) -> Result<()> {
        if let Some(password) = options.password.clone() {
            let result = if let Some(username) = options.username.clone() {
                let args: [&[u8]; 3] = [b"AUTH", username.as_bytes(), password.as_bytes()];
                self.perform(&encode_command(&args), options.timeout, "AUTH")
                    .await
            } else {
                let args: [&[u8]; 2] = [b"AUTH", password.as_bytes()];
                self.perform(&encode_command(&args), options.timeout, "AUTH")
                    .await
            };
            match result {
                Ok(reply) => expect_ok(reply, "AUTH")?,
                Err(error) => return Err(error),
            }
        }
        if let Some(db) = options.db {
            let db = db.to_string();
            let args: [&[u8]; 2] = [b"SELECT", db.as_bytes()];
            match self
                .perform(&encode_command(&args), options.timeout, "SELECT")
                .await
            {
                Ok(reply) => expect_ok(reply, "SELECT")?,
                Err(error) => return Err(error),
            }
        }
        Ok(())
    }

    /// Reads exactly one reply, closing the connection on failure.
    pub(super) async fn perform(
        &mut self,
        payload: &[u8],
        timeout: Duration,
        name: &str,
    ) -> Result<Reply> {
        let mut replies = self.exchange(payload, 1, timeout, name).await?;
        replies
            .pop()
            .ok_or_else(|| Error::transport("redis connection closed before a reply was read"))
    }

    /// Writes `payload` once, then reads `expected` replies in order.
    ///
    /// Any transport failure or timeout closes the connection so the next command
    /// reconnects instead of reading a stale reply, and so does a caller that
    /// drops this future mid-flight (see [`InFlight`]).
    pub(super) async fn exchange(
        &mut self,
        payload: &[u8],
        expected: usize,
        timeout: Duration,
        name: &str,
    ) -> Result<Vec<Reply>> {
        if expected == 0 {
            return Ok(Vec::new());
        }
        if self.stream.is_none() {
            return Err(Error::transport("redis connection is closed"));
        }
        if !self.buffer.is_empty() {
            // The peer sent bytes nobody asked for (or a previous command was
            // abandoned without closing the connection): the next reply cannot be
            // attributed to this command, so the wire is unusable.
            let buffered = self.buffer.len();
            self.close();
            tracing::warn!(
                command = name,
                buffered,
                "redis connection was desynchronised; closing it"
            );
            return Err(Error::transport(format!(
                "redis connection was desynchronised: {buffered} byte(s) of an unattributable \
                 reply were buffered when command '{name}' started; the connection was closed"
            )));
        }
        let mut in_flight = InFlight {
            connection: self,
            finished: false,
        };
        let outcome = in_flight.run(payload, expected, timeout).await;
        // Every non-`Ok` outcome has already closed the connection through the
        // guard; the assignment only tells the guard this conversion is complete.
        in_flight.finished = outcome.is_ok();
        drop(in_flight);
        match outcome {
            Ok(replies) => Ok(replies),
            Err(Interrupted::Failed(error)) => Err(error),
            Err(Interrupted::TimedOut) => Err(Error::timeout(format!(
                "redis command '{name}' timed out after {timeout:?}"
            ))),
        }
    }

    /// Drops the socket so the next command reconnects.
    fn close(&mut self) {
        self.stream.take();
        self.buffer.clear();
    }
}

/// Why an in-flight exchange produced no replies.
enum Interrupted {
    /// A transport, protocol, or limit failure; the connection was closed.
    Failed(Error),
    /// The round trip exceeded its budget; the connection was closed.
    TimedOut,
}

/// Closes the connection when a command future is dropped before its replies are
/// read.
///
/// `RESP` replies are positional, so a partially exchanged command leaves the
/// wire in an unknown state. Dropping the future — a caller-side `timeout`, a
/// `select!` arm, an abort — runs this guard's `Drop` and closes the socket, so
/// the next command reconnects instead of reading the abandoned reply.
struct InFlight<'a> {
    /// The connection whose wire state the guard owns.
    connection: &'a mut Connection,
    /// Set once the caller has its replies; before that, a drop is a cancellation.
    finished: bool,
}

impl InFlight<'_> {
    /// Writes `payload` and reads `expected` replies within `timeout`.
    async fn run(
        &mut self,
        payload: &[u8],
        expected: usize,
        timeout: Duration,
    ) -> std::result::Result<Vec<Reply>, Interrupted> {
        let connection = &mut *self.connection;
        let Some(stream) = connection.stream.as_mut() else {
            return Err(Interrupted::Failed(Error::transport(
                "redis connection is closed",
            )));
        };
        let buffer = &mut connection.buffer;
        let result = tokio::time::timeout(timeout, async {
            stream
                .write_all(payload)
                .await
                .map_err(|error| Error::transport(format!("redis write failed: {error}")))?;
            stream
                .flush()
                .await
                .map_err(|error| Error::transport(format!("redis write failed: {error}")))?;
            let mut replies = Vec::with_capacity(expected);
            for _ in 0..expected {
                replies.push(read_protocol_reply(stream, buffer).await?);
            }
            Ok::<Vec<Reply>, Error>(replies)
        })
        .await;
        match result {
            Ok(Ok(replies)) => Ok(replies),
            Ok(Err(error)) => Err(Interrupted::Failed(error)),
            Err(_) => Err(Interrupted::TimedOut),
        }
    }
}

impl Drop for InFlight<'_> {
    fn drop(&mut self) {
        if !self.finished && self.connection.stream.is_some() {
            tracing::debug!(
                buffered = self.connection.buffer.len(),
                "a redis command future was dropped in flight; closing the connection"
            );
            self.connection.close();
        }
    }
}

/// Opens the TCP or TLS transport described by `options`.
async fn connect_transport(options: &ConnectionOptions) -> Result<Transport> {
    let stream = connect_tcp(options).await?;
    let _ = stream.set_nodelay(true);
    if !options.tls {
        return Ok(Transport::Plain(stream));
    }
    tls::handshake(stream, &options.host, options.port, options.timeout)
        .await
        .map(|stream| Transport::Tls(Box::new(stream)))
}

/// Opens a TCP connection to the configured host and port.
async fn connect_tcp(options: &ConnectionOptions) -> Result<TcpStream> {
    tokio::time::timeout(
        options.timeout,
        TcpStream::connect((options.host.as_str(), options.port)),
    )
    .await
    .map_err(|_| {
        Error::timeout(format!(
            "redis connection to {}:{} timed out after {:?}",
            options.host, options.port, options.timeout
        ))
    })?
    .map_err(|error| {
        Error::transport(format!(
            "failed to connect to redis at {}:{}: {error}",
            options.host, options.port
        ))
    })
}

/// Reads one reply from `stream`, buffering until [`parse_reply`] reports completion.
async fn read_protocol_reply<S>(stream: &mut S, buffer: &mut Vec<u8>) -> Result<Reply>
where
    S: AsyncRead + Unpin,
{
    let mut chunk = [0u8; 8192];
    loop {
        if let Some((reply, consumed)) = parse_reply(buffer)? {
            buffer.drain(..consumed);
            return Ok(reply);
        }
        if buffer.len() > MAX_REPLY_BYTES {
            return Err(Error::transport(format!(
                "redis reply exceeds the {MAX_REPLY_BYTES}-byte limit"
            )));
        }
        let read = stream
            .read(&mut chunk)
            .await
            .map_err(|error| Error::transport(format!("redis read failed: {error}")))?;
        if read == 0 {
            return Err(Error::transport(
                "redis connection closed by the server before a reply was received",
            ));
        }
        buffer.extend_from_slice(&chunk[..read]);
    }
}

/// Validates a handshake reply (`AUTH`/`SELECT` must answer `+OK`).
fn expect_ok(reply: Reply, step: &str) -> Result<()> {
    match reply {
        Reply::Simple(text) if text == "OK" => Ok(()),
        Reply::Error(message) => Err(Error::transport(format!("redis {step} failed: {message}"))),
        other => Err(Error::transport(format!(
            "redis {step} returned an unexpected {} reply",
            other.kind_name()
        ))),
    }
}
