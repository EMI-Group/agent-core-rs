//! A minimal RESP2 client over plain TCP or TLS.
//!
//! [`RedisConnection`] owns one socket, performs the `AUTH`/`SELECT` handshake on
//! connect, and turns a server error frame into an [`Error`]. There is no
//! connection pool: callers serialize access to one connection (the key-value
//! backend holds it behind a [`tokio::sync::Mutex`]).
//!
//! # Transport
//!
//! [`RedisConnection::connect`] and [`RedisConnection::connect_with_timeout`]
//! speak cleartext RESP over [`tokio::net::TcpStream`] (`redis://`).
//! [`RedisConnection::connect_tls`] and [`RedisConnection::connect_tls_with_timeout`]
//! first connect the same way, then run a `tokio-rustls` client handshake over it
//! (`rediss://`).
//! Both are hidden behind the internal `RedisStream` transport, so every method
//! above the transport is shared.
//!
//! The TLS client trusts the certificate authorities bundled in
//! [`webpki_roots::TLS_SERVER_ROOTS`] only: there is no custom-CA bundle, no
//! certificate pinning and no `insecure-skip-verify` switch. The server name is
//! the host of the configured `addr` (a literal IP address is used as an
//! `IpAddress` server name, so no SNI extension is sent for it). The
//! configuration's timeout covers the TCP connect and the TLS handshake together.
//!
//! # Reconnect and retry rule
//!
//! A transport failure triggers at most one transparent reconnect-and-retry:
//!
//! - A failure that happened **before or while writing** the request is retried
//!   for every command. A partial write means the server never received a
//!   complete command, so it cannot have executed it.
//! - A failure while **reading the reply** is retried only for the read-only
//!   commands in `READ_ONLY_COMMANDS` (`PING`, `GET`, `MGET`, `EXISTS`, `SCAN`).
//!   Re-sending a mutating command after its bytes reached the socket could apply
//!   it twice (a lost `SET key value NX` reply, for example, must not become a
//!   second write), so those errors surface to the caller instead.
//!
//! Every reconnect re-runs the handshake, so `AUTH` and `SELECT` are replayed.

use std::fmt;
use std::io;
use std::net::IpAddr;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Duration;

use bytes::Bytes;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt, BufReader, ReadBuf};
use tokio::net::TcpStream;
use tokio_rustls::client::TlsStream;
use tokio_rustls::rustls::pki_types::ServerName;
use tokio_rustls::rustls::{ClientConfig, RootCertStore};
use tokio_rustls::TlsConnector;

use openjiuwen_common::error::{Error, Result};

use super::codec::{encode_command, protocol_error, Decoder, RespValue};

/// Default timeout of a connection attempt, a write, and a reply read.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(5);

/// Number of keys requested per `SCAN` iteration.
const SCAN_COUNT: &str = "100";

/// Upper bound on `SCAN` iterations, so a misbehaving cursor cannot spin forever.
const MAX_SCAN_PAGES: usize = 100_000;

/// Commands that are safe to re-send after their request bytes were written.
pub const READ_ONLY_COMMANDS: [&[u8]; 5] = [b"PING", b"GET", b"MGET", b"EXISTS", b"SCAN"];

/// An error frame returned by the server, for example `-ERR unknown command`.
///
/// [`RedisConnection::command`] surfaces it through
/// [`Error::External`](openjiuwen_common::error::Error), whose `Display` keeps the
/// server text intact.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RedisServerError(pub String);

impl fmt::Display for RedisServerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "redis server error: {}", self.0)
    }
}

impl std::error::Error for RedisServerError {}

/// The username/password pair sent by `AUTH` on connect.
pub type RedisAuth = Option<(Option<String>, String)>;

/// One RESP2 connection to a Redis server.
///
/// The connection is established eagerly by [`RedisConnection::connect`], so a
/// bad address fails there rather than on the first command.
pub struct RedisConnection {
    /// `host:port` of the server.
    addr: String,
    /// Credentials replayed by every handshake.
    auth: RedisAuth,
    /// Database selected by every handshake.
    db: Option<u32>,
    /// Whether the socket is wrapped in TLS.
    tls: bool,
    /// Per-operation timeout.
    timeout: Duration,
    /// The live socket, or `None` after a connection-level failure.
    stream: Option<BufReader<RedisStream>>,
    /// Frames read from the socket but not yet consumed.
    decoder: Decoder,
}

impl fmt::Debug for RedisConnection {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RedisConnection")
            .field("addr", &self.addr)
            .field("tls", &self.tls)
            .field("authenticated", &self.auth.is_some())
            .field("db", &self.db)
            .field("timeout", &self.timeout)
            .field("connected", &self.stream.is_some())
            .finish_non_exhaustive()
    }
}

impl RedisConnection {
    /// Connects, authenticates, and selects `db` over cleartext TCP, using
    /// [`DEFAULT_TIMEOUT`].
    pub async fn connect(addr: &str, auth: RedisAuth, db: Option<u32>) -> Result<Self> {
        Self::connect_with_timeout(addr, auth, db, DEFAULT_TIMEOUT).await
    }

    /// Connects, authenticates, and selects `db` over cleartext TCP with an
    /// explicit timeout.
    pub async fn connect_with_timeout(
        addr: &str,
        auth: RedisAuth,
        db: Option<u32>,
        timeout: Duration,
    ) -> Result<Self> {
        Self::with_options(addr, auth, db, false, timeout).await
    }

    /// Connects, authenticates, and selects `db` over TLS (`rediss://`), using
    /// [`DEFAULT_TIMEOUT`].
    pub async fn connect_tls(addr: &str, auth: RedisAuth, db: Option<u32>) -> Result<Self> {
        Self::connect_tls_with_timeout(addr, auth, db, DEFAULT_TIMEOUT).await
    }

    /// Connects, authenticates, and selects `db` over TLS (`rediss://`) with an
    /// explicit timeout.
    ///
    /// The timeout covers the TCP connect and the TLS handshake together. The TLS
    /// client trusts only the bundled [`webpki_roots::TLS_SERVER_ROOTS`]; a host
    /// of `addr` that is neither a DNS name nor an IP address is an
    /// [`Error::Config`](openjiuwen_common::error::Error).
    pub async fn connect_tls_with_timeout(
        addr: &str,
        auth: RedisAuth,
        db: Option<u32>,
        timeout: Duration,
    ) -> Result<Self> {
        Self::with_options(addr, auth, db, true, timeout).await
    }

    /// Builds a connection and opens it with the requested transport.
    async fn with_options(
        addr: &str,
        auth: RedisAuth,
        db: Option<u32>,
        tls: bool,
        timeout: Duration,
    ) -> Result<Self> {
        let mut connection = Self {
            addr: addr.to_string(),
            auth,
            db,
            tls,
            timeout,
            stream: None,
            decoder: Decoder::new(),
        };
        connection.open().await?;
        Ok(connection)
    }

    /// The `host:port` this connection targets.
    pub fn addr(&self) -> &str {
        &self.addr
    }

    /// Whether this connection speaks TLS (`rediss://`).
    pub fn is_tls(&self) -> bool {
        self.tls
    }

    /// Whether the socket is currently open.
    pub fn is_connected(&self) -> bool {
        self.stream.is_some()
    }

    /// Sends one command and returns its reply frame.
    ///
    /// A server error frame (`-ERR ...`) is returned as
    /// [`Error::External`](openjiuwen_common::error::Error) wrapping
    /// [`RedisServerError`], never as a value. Transport failures follow the
    /// reconnect rule described at the module level.
    pub async fn command(&mut self, args: &[&[u8]]) -> Result<RespValue> {
        let read_only = is_read_only(args);
        match self.round_trip(args).await {
            Ok(value) => into_result(value),
            Err(attempt) => {
                if !attempt.wrote || read_only {
                    self.reconnect().await?;
                    match self.round_trip(args).await {
                        Ok(value) => into_result(value),
                        Err(second) => Err(second.error),
                    }
                } else {
                    Err(attempt.error)
                }
            }
        }
    }

    /// Sends `PING` and checks that the server answers `PONG`.
    pub async fn ping(&mut self) -> Result<()> {
        match self.command(&[b"PING"]).await? {
            RespValue::SimpleString(text) if text.eq_ignore_ascii_case("PONG") => Ok(()),
            RespValue::BulkString(Some(bytes)) if bytes.eq_ignore_ascii_case(b"PONG") => Ok(()),
            other => Err(protocol_error(format!("unexpected PING reply: {other:?}"))),
        }
    }

    /// Collects every key matching `<prefix>*` with a `SCAN` cursor loop.
    ///
    /// Keys are deduplicated (a `SCAN` may return the same key more than once) and
    /// returned in sorted order. The loop ends when the server reports cursor `0`.
    pub async fn scan_pattern(&mut self, prefix: &str) -> Result<Vec<String>> {
        let pattern = format!("{prefix}*");
        let mut cursor = String::from("0");
        let mut keys = std::collections::BTreeSet::new();
        for _ in 0..MAX_SCAN_PAGES {
            let reply = self
                .command(&[
                    b"SCAN",
                    cursor.as_bytes(),
                    b"MATCH",
                    pattern.as_bytes(),
                    b"COUNT",
                    SCAN_COUNT.as_bytes(),
                ])
                .await?;
            let RespValue::Array(Some(items)) = reply else {
                return Err(protocol_error(format!("unexpected SCAN reply: {reply:?}")));
            };
            let [cursor_frame, key_frame] = items.as_slice() else {
                return Err(protocol_error(format!(
                    "SCAN reply did not carry a cursor and a key page: {items:?}"
                )));
            };
            cursor = cursor_frame
                .as_string()
                .ok_or_else(|| protocol_error("SCAN cursor was not a string"))?;
            let RespValue::Array(Some(page)) = key_frame else {
                return Err(protocol_error(format!(
                    "SCAN key page was not an array: {key_frame:?}"
                )));
            };
            for key in page {
                if let Some(key) = key.as_string() {
                    keys.insert(key);
                }
            }
            if cursor == "0" {
                return Ok(keys.into_iter().collect());
            }
        }
        Err(protocol_error(format!(
            "SCAN did not finish within {MAX_SCAN_PAGES} pages"
        )))
    }

    /// Reads several keys with one `MGET`, preserving input order.
    ///
    /// A missing key keeps its slot as `None`.
    pub async fn mget(&mut self, keys: &[String]) -> Result<Vec<Option<Bytes>>> {
        if keys.is_empty() {
            return Ok(Vec::new());
        }
        let mut args: Vec<&[u8]> = Vec::with_capacity(keys.len() + 1);
        args.push(b"MGET");
        args.extend(keys.iter().map(String::as_bytes));
        let reply = self.command(&args).await?;
        let RespValue::Array(Some(items)) = reply else {
            return Err(protocol_error(format!("unexpected MGET reply: {reply:?}")));
        };
        if items.len() != keys.len() {
            return Err(protocol_error(format!(
                "MGET returned {} values for {} keys",
                items.len(),
                keys.len()
            )));
        }
        let mut values = Vec::with_capacity(items.len());
        for item in items {
            match item {
                RespValue::BulkString(Some(bytes)) => values.push(Some(bytes)),
                RespValue::BulkString(None) | RespValue::Null => values.push(None),
                other => {
                    return Err(protocol_error(format!(
                        "unexpected MGET element: {other:?}"
                    )))
                }
            }
        }
        Ok(values)
    }

    /// Opens a fresh socket and replays the handshake.
    async fn reconnect(&mut self) -> Result<()> {
        self.stream = None;
        self.decoder = Decoder::new();
        self.open().await
    }

    /// Connects the socket (running the TLS handshake when configured) and then
    /// replays the handshake.
    async fn open(&mut self) -> Result<()> {
        let addr = self.addr.clone();
        let timeout = self.timeout;
        let stream = match tokio::time::timeout(timeout, Self::dial(&addr, self.tls)).await {
            Err(_) => {
                return Err(Error::timeout(format!(
                    "redis connect to {addr} timed out after {timeout:?}"
                )))
            }
            Ok(Err(error)) => return Err(error),
            Ok(Ok(stream)) => stream,
        };
        self.stream = Some(BufReader::new(stream));
        self.decoder = Decoder::new();
        self.handshake().await
    }

    /// Opens the transport: a plain TCP socket, or a TLS session over it.
    ///
    /// The TLS server name and client configuration are resolved before the TCP
    /// connect, so an unusable host is an [`Error::Config`] rather than a DNS
    /// failure. The caller wraps this in the connection timeout, which therefore
    /// covers the TCP connect and the TLS handshake together.
    async fn dial(addr: &str, tls: bool) -> Result<RedisStream> {
        // Resolve the TLS pieces first: an unusable host never opens a socket.
        let tls = if tls {
            Some((server_name(host_of(addr))?, tls_connector()?))
        } else {
            None
        };
        let tcp = TcpStream::connect(addr).await.map_err(|error| {
            Error::external(io::Error::new(
                error.kind(),
                format!("redis connect to {addr} failed: {error}"),
            ))
        })?;
        let Some((server_name, connector)) = tls else {
            return Ok(RedisStream::Plain(tcp));
        };
        let stream = connector.connect(server_name, tcp).await.map_err(|error| {
            Error::external(io::Error::other(format!(
                "redis TLS handshake with {addr} failed: {error}"
            )))
        })?;
        Ok(RedisStream::Tls(Box::pin(stream)))
    }

    /// Replays `AUTH` and `SELECT` on a freshly opened socket.
    async fn handshake(&mut self) -> Result<()> {
        if let Some((username, password)) = self.auth.clone() {
            let reply = match &username {
                Some(user) => {
                    self.exchange(&[b"AUTH", user.as_bytes(), password.as_bytes()])
                        .await?
                }
                None => self.exchange(&[b"AUTH", password.as_bytes()]).await?,
            };
            match reply {
                RespValue::SimpleString(_) => {}
                RespValue::Error(message) => {
                    return Err(Error::config(format!(
                        "redis authentication failed: {message}"
                    )))
                }
                other => return Err(protocol_error(format!("unexpected AUTH reply: {other:?}"))),
            }
        }
        if let Some(db) = self.db {
            let db = db.to_string();
            match self.exchange(&[b"SELECT", db.as_bytes()]).await? {
                RespValue::SimpleString(_) => {}
                RespValue::Error(message) => {
                    return Err(Error::config(format!(
                        "redis SELECT {db} failed: {message}"
                    )))
                }
                other => {
                    return Err(protocol_error(format!(
                        "unexpected SELECT reply: {other:?}"
                    )))
                }
            }
        }
        Ok(())
    }

    /// Writes one encoded request and reads exactly one reply.
    async fn round_trip(&mut self, args: &[&[u8]]) -> std::result::Result<RespValue, AttemptError> {
        let request = encode_command(args);
        self.write_request(&request)
            .await
            .map_err(|error| AttemptError {
                error,
                wrote: false,
            })?;
        self.read_reply()
            .await
            .map_err(|error| AttemptError { error, wrote: true })
    }

    /// Writes one request and reads one reply on the already-open socket.
    ///
    /// This is the handshake's exchange; it never opens a connection, which keeps
    /// `open` out of a recursive `async fn` cycle.
    async fn exchange(&mut self, args: &[&[u8]]) -> Result<RespValue> {
        let request = encode_command(args);
        self.write_all(&request).await?;
        self.read_reply().await
    }

    /// Writes the encoded request, opening the socket first when necessary.
    async fn write_request(&mut self, request: &[u8]) -> Result<()> {
        if self.stream.is_none() {
            self.open().await?;
        }
        self.write_all(request).await
    }

    /// Writes the encoded request to the open socket.
    async fn write_all(&mut self, request: &[u8]) -> Result<()> {
        let timeout = self.timeout;
        let addr = self.addr.clone();
        let outcome = {
            let Some(stream) = self.stream.as_mut() else {
                return Err(Error::transport("redis connection is not open"));
            };
            tokio::time::timeout(timeout, async {
                stream.write_all(request).await?;
                stream.flush().await
            })
            .await
        };
        match outcome {
            Err(_) => {
                self.stream = None;
                Err(Error::timeout(format!(
                    "redis write to {addr} timed out after {timeout:?}"
                )))
            }
            Ok(Err(error)) => {
                self.stream = None;
                Err(io_error("write to", &addr, &error))
            }
            Ok(Ok(())) => Ok(()),
        }
    }

    /// Reads frames until one is a reply, skipping out-of-band push messages.
    async fn read_reply(&mut self) -> Result<RespValue> {
        loop {
            match self.decoder.try_next() {
                Ok(Some(RespValue::Push(_))) => continue,
                Ok(Some(value)) => return Ok(value),
                Ok(None) => {}
                Err(error) => {
                    self.stream = None;
                    return Err(error);
                }
            }
            let timeout = self.timeout;
            let addr = self.addr.clone();
            let mut chunk = [0_u8; 4096];
            let outcome = {
                let Some(stream) = self.stream.as_mut() else {
                    return Err(Error::transport("redis connection is not open"));
                };
                tokio::time::timeout(timeout, stream.read(&mut chunk)).await
            };
            match outcome {
                Err(_) => {
                    self.stream = None;
                    return Err(Error::timeout(format!(
                        "redis read from {addr} timed out after {timeout:?}"
                    )));
                }
                Ok(Err(error)) => {
                    self.stream = None;
                    return Err(io_error("read from", &addr, &error));
                }
                Ok(Ok(0)) => {
                    self.stream = None;
                    return Err(Error::external(std::io::Error::new(
                        std::io::ErrorKind::ConnectionAborted,
                        format!("redis server {addr} closed the connection"),
                    )));
                }
                Ok(Ok(read)) => self.decoder.push(&chunk[..read]),
            }
        }
    }
}

/// A failed request plus whether any request byte may have reached the server.
struct AttemptError {
    /// The underlying failure.
    error: Error,
    /// `true` once the whole request was written to the socket.
    wrote: bool,
}

/// The transport under a [`RedisConnection`].
enum RedisStream {
    /// A cleartext `redis://` socket.
    Plain(TcpStream),
    /// A `rediss://` session: a TLS client over the TCP socket.
    Tls(Pin<Box<TlsStream<TcpStream>>>),
}

impl AsyncRead for RedisStream {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        match self.as_mut().get_mut() {
            Self::Plain(stream) => Pin::new(stream).poll_read(cx, buf),
            Self::Tls(stream) => stream.as_mut().poll_read(cx, buf),
        }
    }
}

impl AsyncWrite for RedisStream {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        match self.as_mut().get_mut() {
            Self::Plain(stream) => Pin::new(stream).poll_write(cx, buf),
            Self::Tls(stream) => stream.as_mut().poll_write(cx, buf),
        }
    }

    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        match self.as_mut().get_mut() {
            Self::Plain(stream) => Pin::new(stream).poll_flush(cx),
            Self::Tls(stream) => stream.as_mut().poll_flush(cx),
        }
    }

    fn poll_shutdown(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        match self.as_mut().get_mut() {
            Self::Plain(stream) => Pin::new(stream).poll_shutdown(cx),
            Self::Tls(stream) => stream.as_mut().poll_shutdown(cx),
        }
    }
}

/// Returns the host part of a `host:port` endpoint, stripping IPv6 brackets.
fn host_of(addr: &str) -> &str {
    if let Some(rest) = addr.strip_prefix('[') {
        if let Some((host, _)) = rest.split_once(']') {
            return host;
        }
    }
    // A bare IP literal (notably an unbracketed IPv6 address) has no port to cut.
    if addr.parse::<IpAddr>().is_ok() {
        return addr;
    }
    match addr.rsplit_once(':') {
        Some((host, _port)) => host,
        None => addr,
    }
}

/// Resolves the TLS server name (the SNI value) of `host`.
///
/// A literal IP address becomes [`ServerName::IpAddress`], which sends no SNI
/// extension; everything else must be a valid DNS name.
fn server_name(host: &str) -> Result<ServerName<'static>> {
    if let Ok(ip) = host.parse::<IpAddr>() {
        return Ok(ServerName::IpAddress(ip.into()));
    }
    ServerName::try_from(host.to_owned()).map_err(|_| {
        Error::config(format!(
            "redis TLS host {host:?} is not a valid DNS name or IP address"
        ))
    })
}

/// Builds the TLS client of a `rediss://` connection.
///
/// The roots are the certificate authorities bundled in
/// [`webpki_roots::TLS_SERVER_ROOTS`]; a custom-CA bundle, certificate pinning and
/// `insecure-skip-verify` are deliberately unavailable.
fn tls_connector() -> Result<TlsConnector> {
    let mut roots = RootCertStore::empty();
    roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
    let provider = Arc::new(tokio_rustls::rustls::crypto::ring::default_provider());
    let config = ClientConfig::builder_with_provider(provider)
        .with_safe_default_protocol_versions()
        .map_err(|error| Error::config(format!("redis TLS configuration failed: {error}")))?
        .with_root_certificates(roots)
        .with_no_client_auth();
    Ok(TlsConnector::from(Arc::new(config)))
}

/// Whether `args` names a command that is safe to re-send after a lost reply.
fn is_read_only(args: &[&[u8]]) -> bool {
    let Some(name) = args.first() else {
        return false;
    };
    READ_ONLY_COMMANDS
        .iter()
        .any(|candidate| name.eq_ignore_ascii_case(candidate))
}

/// Converts an error frame into `Err`, leaving every other frame as a value.
fn into_result(value: RespValue) -> Result<RespValue> {
    match value {
        RespValue::Error(message) => Err(Error::external(RedisServerError(message))),
        other => Ok(other),
    }
}

/// Wraps an I/O failure with the operation and peer it happened on.
fn io_error(action: &str, addr: &str, error: &std::io::Error) -> Error {
    Error::external(std::io::Error::new(
        error.kind(),
        format!("redis {action} {addr} failed: {error}"),
    ))
}
