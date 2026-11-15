//! Minimal `Redis` client over TCP or TLS, with cluster redirection handling.
//!
//! The workspace has no `Redis` driver crate and builds offline, so this module
//! speaks `RESP2`/`RESP3` directly with [`crate::redis::codec`].
//! Commands are serialized per connection, pipelining is supported, and a
//! connection re-opens lazily after a transport failure or timeout.
//!
//! Supported URL schemes are `redis://`, `rediss://`, `redis+cluster://`, and
//! `rediss+cluster://`: `rediss` selects TLS ([`ConnectionOptions::tls`], verified
//! against the web PKI through `rustls` and `webpki-roots`) and `+cluster` selects
//! [`ConnectionOptions::cluster_mode`], which is verified at connect time instead of
//! being assumed.
//!
//! Cluster redirections are followed: a `-MOVED <slot> host:port` reply re-issues the
//! command once on the reported node and caches the slot, an `-ASK <slot> host:port`
//! reply sends `ASKING` and then the command once, and a second consecutive
//! redirection for the same command is a typed transport error rather than a loop.
//!
//! Server error replies are mapped here: `WRONGTYPE` becomes
//! [`openjiuwen_common::error::ErrorKind::Validation`] and every other error reply
//! (including an unparseable redirection) becomes `Transport`, keeping the server
//! message. Credentials never appear in an error or log message:
//! [`ConnectionOptions`] has a redacting `Debug`, and timeout messages carry only the
//! command name.
//!
//! Reply framing is unchanged: an out-of-band `RESP3` push frame is returned to the
//! caller as [`Reply::Push`] like any other value, because these adapters subscribe to
//! no keyspace channel and therefore expect none.
mod cluster;
mod connection;
mod tls;
use crate::redis::codec::{encode_command, Reply};
use cluster::{Node, Nodes, RedirectKind};
use openjiuwen_common::error::{Error, Result};
use serde_json::Value;
use std::collections::BTreeMap;
use std::fmt;
use std::time::Duration;
use url::Url;
/// Default per-round-trip timeout.
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(10);
/// Default `Redis` port used when a URL does not specify one.
const DEFAULT_PORT: u16 = 6379;
/// How to reach a `Redis` node, including authentication, database, TLS, and mode.
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
    /// Per-round-trip timeout (connect, TLS handshake, each command, each batch).
    pub timeout: Duration,
    /// Whether the URL asked for cluster mode; the seed node is verified as a cluster
    /// node at connect time and redirections are followed.
    pub cluster_mode: bool,
    /// Whether to wrap the transport in TLS (`rediss://`), verified against the web
    /// PKI. Redirect targets inherit this flag.
    pub tls: bool,
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
            tls: false,
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
            .field("tls", &self.tls)
            .finish()
    }
}
impl ConnectionOptions {
    /// Parses `redis[s][+cluster]://[user[:password]@]host[:port][/db]`.
    ///
    /// `rediss` selects TLS and `+cluster` selects cluster mode, so
    /// `rediss+cluster://` sets both. Percent-encoded user info (`%40`) is decoded.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::ErrorKind::Validation`] for a malformed
    /// URL, an unsupported scheme, a missing host, or a non-numeric database path.
    pub fn from_url(url: &str) -> Result<Self> {
        let parsed = Url::parse(url)
            .map_err(|error| Error::validation(format!("invalid redis URL: {error}")))?;
        let (cluster_mode, tls) = match parsed.scheme() {
            "redis" => (false, false),
            "rediss" => (false, true),
            "redis+cluster" => (true, false),
            "rediss+cluster" => (true, true),
            scheme => {
                return Err(Error::validation(format!(
                    "unsupported redis URL scheme '{scheme}'; accepted schemes are redis://, \
                     rediss://, redis+cluster://, and rediss+cluster://"
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
            tls,
        })
    }
    /// Applies `username`/`password`/`db`/`timeout` from Python-style connection
    /// arguments; unknown keys are ignored and `null` clears a value.
    ///
    /// `timeout` is in seconds and may be an integer, a float, or a numeric string.
    /// The TLS and cluster flags are owned by the URL scheme and are left unchanged.
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
    /// Records the requested cluster mode; the seed node is verified at connect time.
    #[must_use]
    pub fn with_cluster_mode(mut self, cluster_mode: bool) -> Self {
        self.cluster_mode = cluster_mode;
        self
    }
    /// Records whether the transport uses TLS (`rediss://`).
    ///
    /// Callers normally get this from the URL scheme; it is exposed so a
    /// configuration that builds options field by field can still ask for TLS.
    #[must_use]
    pub fn with_tls(mut self, tls: bool) -> Self {
        self.tls = tls;
        self
    }
}
/// How a reply at the top of a command is handled.
enum Outcome {
    /// A value for the caller.
    Value(Reply),
    /// A `MOVED`/`ASK` redirection to follow once.
    Redirect(cluster::Redirect),
    /// A typed error (including `WRONGTYPE`).
    Failure(Error),
}
/// A `Redis` client over one seed node plus any cluster redirection targets.
///
/// The client is `Send + Sync`: every node connection is guarded by its own
/// [`tokio::sync::Mutex`], so commands and pipelines never interleave on one wire.
pub struct Client {
    options: ConnectionOptions,
    nodes: Nodes,
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
    /// Opens a connection to the seed node and performs `AUTH` (when a password is
    /// configured) and `SELECT` (when a database is configured).
    ///
    /// When [`ConnectionOptions::cluster_mode`] is set, the seed is verified as a
    /// cluster node (`CLUSTER INFO` must report `cluster_enabled:1`, or
    /// `CLUSTER SLOTS` must return slot ranges) and the slot map is pre-seeded from
    /// `CLUSTER SLOTS` on a best-effort basis.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::ErrorKind::Validation`] for an empty host,
    /// port `0`, or cluster mode against a non-cluster node, `Transport` for a
    /// connection, TLS, or handshake failure, and `Timeout` when the connect or
    /// handshake exceeds [`ConnectionOptions::timeout`].
    pub async fn connect(options: ConnectionOptions) -> Result<Self> {
        if options.host.trim().is_empty() {
            return Err(Error::validation("redis host must not be empty"));
        }
        if options.port == 0 {
            return Err(Error::validation("redis port must not be 0"));
        }
        tracing::debug!(
            host = %options.host,
            port = options.port,
            db = ?options.db,
            tls = options.tls,
            cluster_mode = options.cluster_mode,
            "connecting to redis"
        );
        let nodes = Nodes::new(options.clone());
        let seed = nodes.seed();
        seed.ensure_open().await?;
        if options.cluster_mode {
            cluster::verify_cluster(&seed).await?;
            cluster::preseed_slots(&nodes, &seed).await;
        }
        Ok(Self { options, nodes })
    }
    /// Opens a client from a `redis[s][+cluster]://[user[:password]@]host[/db]` URL.
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
    /// A `MOVED`/`ASK` redirection is followed once (on the reported node, with
    /// `ASKING` for `ASK`); a second consecutive redirection is a `Transport` error.
    ///
    /// # Errors
    ///
    /// `Validation` for an empty command or a `WRONGTYPE` reply, `Timeout` when the
    /// round trip exceeds the configured timeout, and `Transport` for any other server
    /// error, a doubled redirection, or an I/O or TLS failure.
    pub async fn command(&self, args: &[&[u8]]) -> Result<Reply> {
        if args.is_empty() {
            return Err(Error::validation(
                "redis command requires at least one argument",
            ));
        }
        let name = command_name(args);
        let payload = encode_command(args);
        let node = self.route(args);
        let reply = node.reply(&payload, &name).await?;
        self.settle(reply, &payload, &name).await
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
    /// The batch goes to the seed node. A `MOVED`/`ASK` reply inside the batch makes
    /// the client re-issue **that** command once on the reported node and keep the
    /// positional result, so the returned vector always has one reply per command; a
    /// second consecutive redirection for one command fails the batch with a typed
    /// error. Any other error reply fails the whole call, exactly like
    /// [`Client::command`].
    ///
    /// # Errors
    ///
    /// `Validation` for an empty command inside the batch or a `WRONGTYPE` reply,
    /// `Timeout` for the batch exceeding the configured timeout, and `Transport` for
    /// other failures.
    pub async fn pipeline(&self, commands: &[Vec<Vec<u8>>]) -> Result<Vec<Reply>> {
        if commands.is_empty() {
            return Ok(Vec::new());
        }
        let mut payload = Vec::new();
        let mut prepared: Vec<(Vec<u8>, String)> = Vec::with_capacity(commands.len());
        for command in commands {
            if command.is_empty() {
                return Err(Error::validation(
                    "each pipelined redis command requires at least one argument",
                ));
            }
            let args: Vec<&[u8]> = command.iter().map(Vec::as_slice).collect();
            let name = command_name(&args);
            let encoded = encode_command(&args);
            payload.extend_from_slice(&encoded);
            prepared.push((encoded, name));
        }
        let replies = self
            .nodes
            .seed()
            .exchange(&payload, commands.len(), "pipeline")
            .await?;
        let mut results = Vec::with_capacity(replies.len());
        for (reply, (command_payload, name)) in replies.into_iter().zip(prepared) {
            results.push(self.settle(reply, &command_payload, &name).await?);
        }
        Ok(results)
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
    /// The node a command starts on: the cached slot owner, or else the seed.
    fn route(&self, args: &[&[u8]]) -> std::sync::Arc<Node> {
        cluster::command_slot(args)
            .and_then(|slot| self.nodes.slot_node(slot))
            .unwrap_or_else(|| self.nodes.seed())
    }
    /// Maps one reply, following a single `MOVED`/`ASK` redirection.
    async fn settle(&self, reply: Reply, payload: &[u8], name: &str) -> Result<Reply> {
        match classify(reply) {
            Outcome::Value(reply) => Ok(reply),
            Outcome::Failure(error) => Err(error),
            Outcome::Redirect(redirect) => {
                let node = self.nodes.node(&redirect.addr);
                let reply = match redirect.kind {
                    RedirectKind::Moved => node.reply(payload, name).await?,
                    RedirectKind::Ask => node.asking_reply(payload, name).await?,
                };
                match classify(reply) {
                    Outcome::Value(reply) => {
                        if redirect.kind == RedirectKind::Moved {
                            self.nodes.remember_slot(redirect.slot, redirect.addr);
                        }
                        Ok(reply)
                    }
                    Outcome::Failure(error) => Err(error),
                    Outcome::Redirect(second) => Err(Error::transport(format!(
                        "redis cluster replied '{}' for '{name}' after following the redirection to \
                         {}; a second consecutive redirection is not followed",
                        second.raw, redirect.addr
                    ))),
                }
            }
        }
    }
}
/// Classifies a raw reply into a value, a redirection, or a typed error.
fn classify(reply: Reply) -> Outcome {
    match reply {
        Reply::Error(message) => match cluster::parse_redirect(&message) {
            Some(redirect) => Outcome::Redirect(redirect),
            None => Outcome::Failure(map_server_error(&message)),
        },
        other => Outcome::Value(other),
    }
}
/// Maps a non-redirection `-` error reply to a typed error, keeping the message.
///
/// `MOVED`/`ASK` never reach this function: [`classify`] turns them into
/// [`Outcome::Redirect`] and the client follows them itself.
fn map_server_error(message: &str) -> Error {
    if message.starts_with("WRONGTYPE") {
        Error::validation(format!("redis server error: {message}"))
    } else {
        Error::transport(format!("redis server error: {message}"))
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
