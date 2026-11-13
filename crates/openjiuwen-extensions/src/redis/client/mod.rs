//! Minimal single-node `Redis` client over [`tokio::net::TcpStream`].
//!
//! The workspace has no `Redis` driver crate and builds offline, so this module
//! speaks `RESP2` directly with [`crate::redis::codec`]. A client owns exactly one
//! mutex-guarded connection (commands are serialized), supports pipelining, and
//! re-opens the connection lazily after a transport failure or timeout.
//!
//! Supported URL schemes are `redis://` and `redis+cluster://`; the latter only
//! records [`ConnectionOptions::cluster_mode`] and then connects as a single node
//! because slot routing is not implemented. `rediss://` and `rediss+cluster://`
//! return [`openjiuwen_common::error::ErrorKind::Unsupported`] naming TLS as the
//! missing piece.
//!
//! Server error replies are mapped here: `MOVED`/`ASK` redirections become
//! `Unsupported`, `WRONGTYPE` becomes `Validation`, and every other error reply
//! becomes `Transport`, keeping the server message. Credentials never appear in
//! an error or log message: [`ConnectionOptions`] has a redacting `Debug`, and
//! timeout messages carry only the command name.
use super::codec::{encode_command, parse_reply, Reply};
use openjiuwen_common::error::{Error, Result};
use serde_json::Value;
use std::collections::BTreeMap;
use std::fmt;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::sync::Mutex;
use url::Url;
/// Default per-round-trip timeout.
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(10);
/// Default `Redis` port used when a URL does not specify one.
const DEFAULT_PORT: u16 = 6379;
/// Maximum bytes buffered for one in-flight reply before the connection is dropped.
const MAX_REPLY_BYTES: usize = 512 * 1024 * 1024;
/// How to reach a single `Redis` node, including authentication and database.
///
/// Build one with [`ConnectionOptions::default`] or parse a URL with
/// [`ConnectionOptions::from_url`]; `connection_args` maps onto the same fields
/// through [`ConnectionOptions::with_connection_args`].
#[derive(Clone, PartialEq, Eq)]
pub struct ConnectionOptions {
    /// Server host name or IP literal (without brackets for IPv6).
    pub host: String,
    /// Server port.
    pub port: u16,
    /// `ACL` user name for `AUTH user password`; `None` sends a password-only `AUTH`.
    pub username: Option<String>,
    /// Password; never logged or echoed in an error message.
    pub password: Option<String>,
    /// Database index selected with `SELECT` after connecting.
    pub db: Option<i64>,
    /// Per-round-trip timeout (connect, each command, and each pipeline batch).
    pub timeout: Duration,
    /// Whether the URL asked for cluster mode; the client still connects to one node.
    pub cluster_mode: bool,
}
impl Default for ConnectionOptions {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_owned(),
            port: DEFAULT_PORT,
            username: None,
            password: None,
            db: None,
            timeout: DEFAULT_TIMEOUT,
            cluster_mode: false,
        }
    }
}
impl fmt::Debug for ConnectionOptions {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ConnectionOptions")
            .field("host", &self.host)
            .field("port", &self.port)
            .field("username", &self.username)
            .field("password", &self.password.as_ref().map(|_| "<redacted>"))
            .field("db", &self.db)
            .field("timeout", &self.timeout)
            .field("cluster_mode", &self.cluster_mode)
            .finish()
    }
}
impl ConnectionOptions {
    /// Parses `redis://[user[:password]@]host[:port][/db]`.
    ///
    /// `redis+cluster://` is accepted but connects as a single node. Percent-encoded
    /// user info (`%40`) is decoded.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::ErrorKind::Validation`] for a malformed
    /// URL, a missing host, or a non-numeric database path, and
    /// [`openjiuwen_common::error::ErrorKind::Unsupported`] for the TLS schemes
    /// `rediss://` and `rediss+cluster://`.
    pub fn from_url(url: &str) -> Result<Self> {
        let parsed = Url::parse(url)
            .map_err(|error| Error::validation(format!("invalid redis URL: {error}")))?;
        let cluster_mode = match parsed.scheme() {
            "redis" => false,
            "redis+cluster" => true,
            "rediss" | "rediss+cluster" => {
                return Err(Error::unsupported(
                    "redis TLS (rediss://) is not supported: the workspace does not vendor a \
                     TLS stack; use redis:// to a trusted network or a TLS terminator",
                ));
            }
            scheme => {
                return Err(Error::validation(format!(
                    "unsupported redis URL scheme '{scheme}'; accepted schemes are redis:// and \
                     redis+cluster://"
                )));
            }
        };
        let host = parsed
            .host_str()
            .filter(|host| !host.is_empty())
            .map(|host| {
                host.trim_start_matches('[')
                    .trim_end_matches(']')
                    .to_owned()
            })
            .ok_or_else(|| Error::validation("redis URL is missing a host".to_owned()))?;
        let username = match percent_decode(parsed.username()) {
            username if username.is_empty() => None,
            username => Some(username),
        };
        let password = parsed.password().map(percent_decode);
        Ok(Self {
            host,
            port: parsed.port().unwrap_or(DEFAULT_PORT),
            username,
            password,
            db: database_from_path(parsed.path())?,
            timeout: DEFAULT_TIMEOUT,
            cluster_mode,
        })
    }
    /// Applies `username`/`password`/`db`/`timeout` from Python-style connection
    /// arguments; unknown keys are ignored and `null` clears a value.
    ///
    /// `timeout` is in seconds and may be an integer, a float, or a numeric string.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::ErrorKind::Validation`] when a recognised
    /// key has the wrong type, when `db` is negative, or when `timeout` is not a
    /// positive finite number.
    pub fn with_connection_args(
        mut self,
        args: &BTreeMap<String, serde_json::Value>,
    ) -> Result<Self> {
        for (key, value) in args {
            match key.as_str() {
                "username" => self.username = optional_string(key, value)?,
                "password" => self.password = optional_string(key, value)?,
                "db" => {
                    if let Some(db) = optional_i64(key, value)? {
                        if db < 0 {
                            return Err(Error::validation(format!(
                                "redis connection argument '{key}' must be non-negative, got {db}"
                            )));
                        }
                        self.db = Some(db);
                    } else {
                        self.db = None;
                    }
                }
                "timeout" => {
                    if let Some(seconds) = optional_f64(key, value)? {
                        self.timeout = duration_from_seconds(seconds)?;
                    }
                }
                _ => {}
            }
        }
        Ok(self)
    }
    /// Records the requested cluster mode (the connection stays single-node).
    #[must_use]
    pub fn with_cluster_mode(mut self, cluster_mode: bool) -> Self {
        self.cluster_mode = cluster_mode;
        self
    }
}
/// A serialized connection to a single `Redis` node.
///
/// The client is `Send + Sync`: one [`tokio::sync::Mutex`]-guarded connection is
/// shared by all callers, so commands and pipelines never interleave on the wire.
pub struct Client {
    options: ConnectionOptions,
    connection: Mutex<Connection>,
}
impl fmt::Debug for Client {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Client")
            .field("options", &self.options)
            .finish_non_exhaustive()
    }
}
impl Client {
    /// Opens a connection and performs `AUTH` (when a password is configured) and
    /// `SELECT` (when a database is configured).
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::ErrorKind::Validation`] for an empty host
    /// or port `0`, `Transport` for a connection or handshake failure, and `Timeout`
    /// when the connect or handshake exceeds [`ConnectionOptions::timeout`].
    pub async fn connect(options: ConnectionOptions) -> Result<Self> {
        if options.host.trim().is_empty() {
            return Err(Error::validation("redis host must not be empty"));
        }
        if options.port == 0 {
            return Err(Error::validation("redis port must not be 0"));
        }
        if options.cluster_mode {
            tracing::warn!(
                host = %options.host,
                port = options.port,
                "redis cluster mode requested; connecting to one node because slot routing is not implemented"
            );
        }
        tracing::debug!(
            host = %options.host,
            port = options.port,
            db = ?options.db,
            "connecting to redis"
        );
        let mut connection = Connection::default();
        connection.ensure_open(&options).await?;
        Ok(Self {
            options,
            connection: Mutex::new(connection),
        })
    }
    /// Opens a client from a `redis://[user[:password]@]host[:port][/db]` URL.
    ///
    /// # Errors
    ///
    /// Same as [`ConnectionOptions::from_url`] and [`Client::connect`].
    pub async fn from_url(url: &str) -> Result<Self> {
        Self::connect(ConnectionOptions::from_url(url)?).await
    }
    /// The options this client was built with (including credentials).
    #[must_use]
    pub fn options(&self) -> &ConnectionOptions {
        &self.options
    }
    /// Sends one command and returns its reply.
    ///
    /// # Errors
    ///
    /// `Validation` for an empty command or a `WRONGTYPE` reply, `Unsupported` for a
    /// `MOVED`/`ASK` redirection, `Timeout` when the round trip exceeds the
    /// configured timeout, and `Transport` for any other server error or I/O failure.
    pub async fn command(&self, args: &[&[u8]]) -> Result<Reply> {
        if args.is_empty() {
            return Err(Error::validation(
                "redis command requires at least one argument",
            ));
        }
        let name = command_name(args);
        let payload = encode_command(args);
        let reply = self
            .send(&payload, 1, &name)
            .await?
            .into_iter()
            .next()
            .ok_or_else(|| Error::transport("redis connection closed before a reply was read"))?;
        map_reply(reply)
    }
    /// Convenience wrapper over [`Client::command`] for text arguments.
    ///
    /// # Errors
    ///
    /// Same as [`Client::command`].
    pub async fn command_str(&self, args: &[&str]) -> Result<Reply> {
        let args: Vec<&[u8]> = args.iter().map(|arg| arg.as_bytes()).collect();
        self.command(&args).await
    }
    /// Sends every command in one write and returns the replies in order.
    ///
    /// An error reply among the responses fails the whole call, exactly like
    /// [`Client::command`].
    ///
    /// # Errors
    ///
    /// `Validation` for an empty command inside the batch or a `WRONGTYPE` reply,
    /// `Unsupported` for a `MOVED`/`ASK` redirection, `Timeout` for the batch
    /// exceeding the configured timeout, and `Transport` for other failures.
    pub async fn pipeline(&self, commands: &[Vec<Vec<u8>>]) -> Result<Vec<Reply>> {
        if commands.is_empty() {
            return Ok(Vec::new());
        }
        let mut payload = Vec::new();
        for command in commands {
            if command.is_empty() {
                return Err(Error::validation(
                    "each pipelined redis command requires at least one argument",
                ));
            }
            let args: Vec<&[u8]> = command.iter().map(Vec::as_slice).collect();
            payload.extend_from_slice(&encode_command(&args));
        }
        let replies = self.send(&payload, commands.len(), "pipeline").await?;
        replies.into_iter().map(map_reply).collect()
    }
    /// Sends `PING` and requires the `PONG` reply.
    ///
    /// # Errors
    ///
    /// Same as [`Client::command`], plus `Transport` when the reply is not `PONG`.
    pub async fn ping(&self) -> Result<()> {
        let args: [&[u8]; 1] = [b"PING"];
        let reply = self.command(&args).await?;
        match &reply {
            Reply::Simple(text) if text == "PONG" => Ok(()),
            Reply::Bulk(bytes) if bytes == b"PONG" => Ok(()),
            other => Err(Error::transport(format!(
                "redis PING returned an unexpected {} reply",
                other.kind_name()
            ))),
        }
    }
    /// Acquires the connection lock and exchanges a payload under a single timeout.
    async fn send(&self, payload: &[u8], expected: usize, name: &str) -> Result<Vec<Reply>> {
        let timeout = self.options.timeout;
        let mut connection = tokio::time::timeout(timeout, self.connection.lock())
            .await
            .map_err(|_| {
                Error::timeout(format!(
                    "timed out waiting for the redis connection to run '{name}'"
                ))
            })?;
        connection.ensure_open(&self.options).await?;
        connection.exchange(payload, expected, timeout, name).await
    }
}
/// The single connection shared by a [`Client`].
#[derive(Default)]
struct Connection {
    stream: Option<TcpStream>,
    buffer: Vec<u8>,
}
impl Connection {
    /// Connects and runs the handshake when the connection is not open.
    async fn ensure_open(&mut self, options: &ConnectionOptions) -> Result<()> {
        if self.stream.is_some() {
            return Ok(());
        }
        let stream = connect_stream(options).await?;
        let _ = stream.set_nodelay(true);
        self.stream = Some(stream);
        self.buffer.clear();
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
                Ok(reply) => {
                    if let Err(error) = expect_ok(reply, "AUTH") {
                        self.close();
                        return Err(error);
                    }
                }
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
                Ok(reply) => {
                    if let Err(error) = expect_ok(reply, "SELECT") {
                        self.close();
                        return Err(error);
                    }
                }
                Err(error) => return Err(error),
            }
        }
        Ok(())
    }
    /// Reads exactly one reply, closing the connection on failure.
    async fn perform(&mut self, payload: &[u8], timeout: Duration, name: &str) -> Result<Reply> {
        let mut replies = self.exchange(payload, 1, timeout, name).await?;
        replies
            .pop()
            .ok_or_else(|| Error::transport("redis connection closed before a reply was read"))
    }
    /// Writes `payload` once, then reads `expected` replies in order.
    ///
    /// Any transport failure or timeout closes the connection so the next command
    /// reconnects instead of reading a stale reply.
    async fn exchange(
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
        let result = tokio::time::timeout(timeout, async {
            let Some(stream) = self.stream.as_mut() else {
                return Err(Error::transport("redis connection is closed"));
            };
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
                replies.push(read_protocol_reply(stream, &mut self.buffer).await?);
            }
            Ok::<Vec<Reply>, Error>(replies)
        })
        .await;
        match result {
            Ok(Ok(replies)) => Ok(replies),
            Ok(Err(error)) => {
                self.close();
                Err(error)
            }
            Err(_) => {
                self.close();
                Err(Error::timeout(format!(
                    "redis command '{name}' timed out after {timeout:?}"
                )))
            }
        }
    }
    /// Drops the socket so the next command reconnects.
    fn close(&mut self) {
        self.stream.take();
        self.buffer.clear();
    }
}
/// Opens a TCP connection to the configured host and port.
async fn connect_stream(options: &ConnectionOptions) -> Result<TcpStream> {
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
async fn read_protocol_reply(stream: &mut TcpStream, buffer: &mut Vec<u8>) -> Result<Reply> {
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
/// Maps a server error reply to the client's typed error.
fn map_reply(reply: Reply) -> Result<Reply> {
    match reply {
        Reply::Error(message) => Err(map_server_error(&message)),
        other => Ok(other),
    }
}
/// Maps a `-` error reply to a typed error, preserving the server message.
fn map_server_error(message: &str) -> Error {
    if message.starts_with("MOVED ") || message.starts_with("ASK ") {
        Error::unsupported(
            "redis cluster slot redirection (MOVED/ASK) is not implemented; connect to a cluster \
             entry point or use standalone Redis",
        )
    } else if message.starts_with("WRONGTYPE") {
        Error::validation(format!("redis server error: {message}"))
    } else {
        Error::transport(format!("redis server error: {message}"))
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
/// The command name used in timeout messages; argument values never leak.
fn command_name(args: &[&[u8]]) -> String {
    args.first().map_or_else(
        || "<empty>".to_owned(),
        |name| {
            let name = String::from_utf8_lossy(name).to_ascii_uppercase();
            if name.is_empty() {
                "<empty>".to_owned()
            } else {
                name
            }
        },
    )
}
/// Parses the `/db` path component of a Redis URL.
fn database_from_path(path: &str) -> Result<Option<i64>> {
    let trimmed = path.trim_start_matches('/').trim_end_matches('/');
    if trimmed.is_empty() {
        return Ok(None);
    }
    let db = trimmed.parse::<i64>().map_err(|error| {
        Error::validation(format!(
            "invalid redis database index '{trimmed}' in URL path: {error}"
        ))
    })?;
    if db < 0 {
        return Err(Error::validation(format!(
            "redis database index must be non-negative, got {db}"
        )));
    }
    Ok(Some(db))
}
/// Decodes `%XX` escapes in URL user info (credentials may contain `@`, `:`, ...).
#[allow(clippy::cast_possible_truncation)]
fn percent_decode(input: &str) -> String {
    let bytes = input.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'%' && index + 2 < bytes.len() {
            let high = char::from(bytes[index + 1]).to_digit(16);
            let low = char::from(bytes[index + 2]).to_digit(16);
            if let (Some(high), Some(low)) = (high, low) {
                // `high`/`low` are hexadecimal digits, so the sum fits in a `u8`.
                decoded.push((high * 16 + low) as u8);
                index += 3;
                continue;
            }
        }
        decoded.push(bytes[index]);
        index += 1;
    }
    String::from_utf8_lossy(&decoded).into_owned()
}
/// Reads an optional string connection argument.
fn optional_string(key: &str, value: &Value) -> Result<Option<String>> {
    match value {
        Value::Null => Ok(None),
        Value::String(text) => Ok(Some(text.clone())),
        _ => Err(Error::validation(format!(
            "redis connection argument '{key}' must be a string"
        ))),
    }
}
/// Reads an optional integer connection argument (number or numeric string).
fn optional_i64(key: &str, value: &Value) -> Result<Option<i64>> {
    match value {
        Value::Null => Ok(None),
        Value::Number(number) => number.as_i64().map(Some).ok_or_else(|| {
            Error::validation(format!(
                "redis connection argument '{key}' must be an integer"
            ))
        }),
        Value::String(text) => text.parse::<i64>().map(Some).map_err(|error| {
            Error::validation(format!(
                "redis connection argument '{key}' must be an integer: {error}"
            ))
        }),
        _ => Err(Error::validation(format!(
            "redis connection argument '{key}' must be an integer"
        ))),
    }
}
/// Reads an optional numeric connection argument (number or numeric string).
fn optional_f64(key: &str, value: &Value) -> Result<Option<f64>> {
    match value {
        Value::Null => Ok(None),
        Value::Number(number) => number.as_f64().map(Some).ok_or_else(|| {
            Error::validation(format!(
                "redis connection argument '{key}' must be a number"
            ))
        }),
        Value::String(text) => text.parse::<f64>().map(Some).map_err(|error| {
            Error::validation(format!(
                "redis connection argument '{key}' must be a number: {error}"
            ))
        }),
        _ => Err(Error::validation(format!(
            "redis connection argument '{key}' must be a number"
        ))),
    }
}
/// Converts a positive number of seconds into a [`Duration`].
fn duration_from_seconds(seconds: f64) -> Result<Duration> {
    if !seconds.is_finite() || seconds <= 0.0 {
        return Err(Error::validation(
            "redis timeout must be a positive number of seconds",
        ));
    }
    Duration::try_from_secs_f64(seconds)
        .map_err(|error| Error::validation(format!("invalid redis timeout {seconds}: {error}")))
}
#[cfg(test)]
mod tests;
