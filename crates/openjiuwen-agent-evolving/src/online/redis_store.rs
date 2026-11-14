//! The Redis-backed [`OnlineStore`], behind the crate feature `redis`.
//!
//! The state machines ([`TaskRegistry`](crate::online::TaskRegistry),
//! [`SampleLedger`](crate::online::SampleLedger),
//! [`TrainingRunner`](crate::online::TrainingRunner)) are written against the
//! domain-free [`OnlineStore`] seam, so this module only maps primitives onto
//! Redis commands; no rule of the task, sample or training machine lives here.
//!
//! The transport is [`openjiuwen_extensions::redis`]'s in-workspace client (one
//! serialized connection per store, `redis://` only). TLS (`rediss://`) and
//! cluster slot routing are not vendored in this workspace and are rejected with
//! [`Error::Unsupported`] by
//! [`RedisOnlineStoreOptions::validate`].
//!
//! # Command mapping
//!
//! | `OnlineStore` | Redis | Result mapping |
//! |---|---|---|
//! | [`get`](OnlineStore::get) | `GET` | null → `None` |
//! | [`set`](OnlineStore::set) | `SET` | requires `+OK` |
//! | [`set_if_absent`](OnlineStore::set_if_absent) | `SET k v NX` | null → `false`, `+OK` → `true` |
//! | [`del`](OnlineStore::del) | `DEL` | affected count > 0 |
//! | [`set_add`](OnlineStore::set_add) | `SADD` | added count > 0 |
//! | [`set_remove`](OnlineStore::set_remove) | `SREM` | removed count > 0 |
//! | [`set_members`](OnlineStore::set_members) | `SMEMBERS` | sorted ascending locally |
//! | [`set_contains`](OnlineStore::set_contains) | `SISMEMBER` | `1` → `true` |
//!
//! Every key is written under the store's prefix as `{prefix}:{key}`, so the
//! durable layout is `openjiuwen:rl:rl:v1:…` by default; the `rl:v1` namespace
//! (documented once in the [`crate::online`] module docs) stays intact and two
//! stores with different prefixes share one database without colliding.
//!
//! # Errors are never swallowed
//!
//! A server error reply (`WRONGTYPE` when a key is read as the wrong type, a
//! protocol error, `MOVED`/`ASK` from a cluster reply) surfaces as an
//! [`openjiuwen_common::error::Error`] — `Validation` for `WRONGTYPE`,
//! `Unsupported` for a redirection, `Transport`/`Timeout` otherwise — and the
//! state machines turn it into [`OnlineError::Store`](crate::online::OnlineError::Store).
//! A reply of an unexpected type is a `Validation` error too. Nothing degrades to
//! an empty read or a silent no-op.
//!
//! # Durability and concurrency
//!
//! The state machines compose several primitives per operation (a capture commit
//! writes a record, stages a sample and rewrites two index sets) and this backend
//! executes them as ordered single commands, not as a Lua script or a
//! `WATCH`/`MULTI` transaction (Python used the latter). On the one serialized
//! connection every caller sees its own writes, so a single service process is
//! correct; concurrent *processes* pointing at the same database can interleave
//! between the commands of one sequence, leaving an index briefly ahead of or
//! behind the record it describes.
//!
//! The two single-flight paths keep their guarantee because each is a single
//! command:
//!
//! - the training-run claim uses `SET rl:v1:training_run:active … NX`, so exactly
//!   one claimant wins the marker and the loser deletes its orphan record and
//!   requeues its batch;
//! - the task-start adoption path uses `SET rl:v1:task:{id} … NX`, so exactly one
//!   creator wins and the loser re-reads the winner's record.
//!
//! Everything else follows Python's documented operational rule: one writable
//! service process per database (see the `online` module's known issues).

use std::fmt;
use std::time::Duration;

use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_extensions::redis::{Client, ConnectionOptions, Reply};
use serde::{Deserialize, Serialize};

use super::store::OnlineStore;

/// Default prefix every key of the durable state machine gets.
pub const DEFAULT_PREFIX: &str = "openjiuwen:rl";

/// Default per-command timeout of the Redis store.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(10);

/// Configuration of a [`RedisOnlineStore`].
///
/// `url` is a `redis://[user[:password]@]host[:port][/db]` URL; `prefix` is the
/// namespace prepended to every key; `database` overrides the database index the
/// URL selects; `timeout` bounds one round trip.
///
/// `Debug` never prints credentials: the URL is parsed into the client's
/// [`ConnectionOptions`], whose `Debug` redacts the password, and an unparseable
/// URL is printed as a placeholder.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct RedisOnlineStoreOptions {
    /// `redis://` URL of the server, including credentials and database.
    pub url: String,
    /// Namespace prepended to every key; blank means [`DEFAULT_PREFIX`].
    pub prefix: String,
    /// Database index, overriding the one in `url` when set.
    pub database: Option<i64>,
    /// Timeout of one command round trip.
    pub timeout: Duration,
}

impl Default for RedisOnlineStoreOptions {
    fn default() -> Self {
        Self {
            url: "redis://127.0.0.1:6379".to_owned(),
            prefix: DEFAULT_PREFIX.to_owned(),
            database: None,
            timeout: DEFAULT_TIMEOUT,
        }
    }
}

impl fmt::Debug for RedisOnlineStoreOptions {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut debug = formatter.debug_struct("RedisOnlineStoreOptions");
        match ConnectionOptions::from_url(self.url.trim()) {
            // The client's options have a redacting `Debug`, so a password in the
            // URL never reaches a log line.
            Ok(connection) => {
                debug.field("connection", &connection);
            }
            Err(_) => {
                debug.field("connection", &"<unparseable url>");
            }
        }
        debug
            .field("prefix", &self.prefix)
            .field("database", &self.database)
            .field("timeout", &self.timeout)
            .finish()
    }
}

impl RedisOnlineStoreOptions {
    /// Creates options for `url` with every other field at its default.
    #[must_use]
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            ..Self::default()
        }
    }

    /// Rejects a blank URL, a non-`redis://` scheme, TLS and cluster mode.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] for a blank or
    /// malformed URL and a negative database index;
    /// [`Error::Unsupported`] for `rediss://` /
    /// `rediss+cluster://` (no TLS stack is vendored) and for a cluster scheme
    /// (slot routing is not implemented).
    pub fn validate(&self) -> Result<()> {
        let url = self.url.trim();
        if url.is_empty() {
            return Err(Error::validation(
                "redis online store url must not be empty",
            ));
        }
        match scheme(url) {
            Some("redis") => {}
            Some("redis+cluster" | "rediss+cluster") => {
                return Err(Error::unsupported(
                    "redis cluster mode is not supported by the online RL store: slot routing \
                     is not implemented; point the url at one node with redis://",
                ));
            }
            Some("rediss") => {
                return Err(Error::unsupported(
                    "redis TLS (rediss://) is not supported by the online RL store: the \
                     workspace does not vendor a TLS stack; use redis:// to a trusted network or \
                     a TLS terminator",
                ));
            }
            Some(other) => {
                return Err(Error::validation(format!(
                    "unsupported redis online store url scheme '{other}'; accepted scheme is \
                     redis://"
                )));
            }
            None => {
                return Err(Error::validation(format!(
                    "invalid redis online store url '{url}': expected redis://host[:port][/db]"
                )));
            }
        }
        if let Some(database) = self.database {
            if database < 0 {
                return Err(Error::validation(format!(
                    "redis online store database must be non-negative, got {database}"
                )));
            }
        }
        // Confirms the host, port and database path the client will parse.
        self.connection_options().map(|_| ())
    }

    /// Builds the client connection options these settings describe.
    fn connection_options(&self) -> Result<ConnectionOptions> {
        let mut connection = ConnectionOptions::from_url(self.url.trim())?;
        if let Some(database) = self.database {
            connection.db = Some(database);
        }
        connection.timeout = self.timeout;
        Ok(connection)
    }
}

/// Reads the scheme of a URL without parsing it, so a rejected scheme gets a
/// typed error instead of a generic parse failure.
fn scheme(url: &str) -> Option<&str> {
    let (scheme, _) = url.split_once("://")?;
    let scheme = scheme.trim();
    if scheme.is_empty() {
        return None;
    }
    Some(scheme)
}

/// Normalizes a configured prefix: surrounding whitespace and trailing `:`
/// separators are dropped, and a blank prefix means [`DEFAULT_PREFIX`].
fn normalize_prefix(prefix: &str) -> String {
    let trimmed = prefix.trim().trim_end_matches(':');
    if trimmed.trim().is_empty() {
        DEFAULT_PREFIX.to_owned()
    } else {
        trimmed.to_owned()
    }
}

/// An [`OnlineStore`] backed by one Redis connection.
///
/// The store owns a single serialized connection, so commands issued by one
/// caller are ordered as written. It is `Send + Sync` but not `Clone`; wrap it in
/// an [`Arc`](std::sync::Arc) to share it between components (every `Arc<S>` is
/// an `OnlineStore`).
#[derive(Debug)]
pub struct RedisOnlineStore {
    client: Client,
    prefix: String,
}

impl RedisOnlineStore {
    /// Connects to `url` with the default prefix, database and timeout.
    ///
    /// The URL is validated against [`RedisOnlineStoreOptions::validate`] first,
    /// so TLS and cluster URLs are rejected with
    /// [`Error::Unsupported`].
    pub async fn connect(url: &str) -> Result<Self> {
        Self::connect_with_options(&RedisOnlineStoreOptions::new(url)).await
    }

    /// Connects with explicit options.
    pub async fn connect_with_options(options: &RedisOnlineStoreOptions) -> Result<Self> {
        options.validate()?;
        let client = Client::connect(options.connection_options()?).await?;
        Ok(Self {
            client,
            prefix: normalize_prefix(&options.prefix),
        })
    }

    /// Wraps an already-connected client under `prefix`.
    ///
    /// A blank prefix (or one made only of `:`) means [`DEFAULT_PREFIX`].
    pub fn from_client(client: Client, prefix: impl Into<String>) -> Self {
        Self {
            client,
            prefix: normalize_prefix(&prefix.into()),
        }
    }

    /// The key prefix every write uses.
    #[must_use]
    pub fn prefix(&self) -> &str {
        &self.prefix
    }

    /// The underlying client, for diagnostics (`INFO`, `DBSIZE`, …).
    #[must_use]
    pub fn client(&self) -> &Client {
        &self.client
    }

    /// Sends `PING`, proving the connection is reachable.
    pub async fn ping(&self) -> Result<()> {
        self.client.ping().await
    }

    /// The durable key of `key`: `{prefix}:{key}`.
    fn key(&self, key: &str) -> String {
        format!("{}:{}", self.prefix, key)
    }
}

/// Requires the `+OK` reply of a mutating command.
fn expect_ok(reply: &Reply, command: &str) -> Result<()> {
    match reply {
        Reply::Simple(text) if text == "OK" => Ok(()),
        Reply::Bulk(bytes) if bytes == b"OK" => Ok(()),
        other => Err(Error::transport(format!(
            "redis {command} returned an unexpected {} reply",
            other.kind_name()
        ))),
    }
}

/// Interprets an integer reply as "something changed" (`> 0`).
fn changed(reply: &Reply, command: &str) -> Result<bool> {
    Ok(reply.as_integer().map_err(|error| {
        Error::validation(format!("redis {command} returned no integer: {error}"))
    })? > 0)
}

#[async_trait]
impl OnlineStore for RedisOnlineStore {
    async fn get(&self, key: &str) -> Result<Option<String>> {
        let full = self.key(key);
        let reply = self.client.command_str(&["GET", full.as_str()]).await?;
        reply.as_opt_string_owned()
    }

    async fn set(&self, key: &str, value: &str) -> Result<()> {
        let full = self.key(key);
        let reply = self
            .client
            .command_str(&["SET", full.as_str(), value])
            .await?;
        expect_ok(&reply, "SET")
    }

    async fn set_if_absent(&self, key: &str, value: &str) -> Result<bool> {
        let full = self.key(key);
        let reply = self
            .client
            .command_str(&["SET", full.as_str(), value, "NX"])
            .await?;
        if reply.is_null() {
            return Ok(false);
        }
        expect_ok(&reply, "SET NX").map(|()| true)
    }

    async fn del(&self, key: &str) -> Result<bool> {
        let full = self.key(key);
        let reply = self.client.command_str(&["DEL", full.as_str()]).await?;
        changed(&reply, "DEL")
    }

    async fn set_add(&self, key: &str, member: &str) -> Result<bool> {
        let full = self.key(key);
        let reply = self
            .client
            .command_str(&["SADD", full.as_str(), member])
            .await?;
        changed(&reply, "SADD")
    }

    async fn set_remove(&self, key: &str, member: &str) -> Result<bool> {
        let full = self.key(key);
        let reply = self
            .client
            .command_str(&["SREM", full.as_str(), member])
            .await?;
        changed(&reply, "SREM")
    }

    async fn set_members(&self, key: &str) -> Result<Vec<String>> {
        let full = self.key(key);
        let reply = self
            .client
            .command_str(&["SMEMBERS", full.as_str()])
            .await?;
        let mut members = Vec::new();
        for item in reply.as_array()? {
            members.push(item.as_string_owned()?);
        }
        // Redis returns set members in hash order; sorting keeps scans stable
        // across servers and restarts, matching the in-memory store.
        members.sort();
        Ok(members)
    }

    async fn set_contains(&self, key: &str, member: &str) -> Result<bool> {
        let full = self.key(key);
        let reply = self
            .client
            .command_str(&["SISMEMBER", full.as_str(), member])
            .await?;
        changed(&reply, "SISMEMBER")
    }
}

#[cfg(test)]
mod tests {
    //! Offline unit tests of the option handling; the store itself is exercised
    //! over the fake server in `tests/redis.rs`.

    use std::time::Duration;

    use openjiuwen_common::error::{Error, ErrorKind};

    use super::{normalize_prefix, RedisOnlineStoreOptions, DEFAULT_PREFIX};

    #[test]
    fn default_options_target_a_local_server() {
        let options = RedisOnlineStoreOptions::default();
        assert_eq!(options.url, "redis://127.0.0.1:6379");
        assert_eq!(options.prefix, DEFAULT_PREFIX);
        assert_eq!(options.database, None);
        assert_eq!(options.timeout, Duration::from_secs(10));
        options.validate().expect("default options are valid");
    }

    #[test]
    fn options_round_trip_through_json() {
        let options = RedisOnlineStoreOptions {
            url: "redis://cache:6380/3".to_owned(),
            prefix: "team-a".to_owned(),
            database: Some(4),
            timeout: Duration::from_millis(250),
        };
        let encoded = serde_json::to_string(&options).expect("encode");
        let decoded: RedisOnlineStoreOptions = serde_json::from_str(&encoded).expect("decode");
        assert_eq!(decoded, options);
    }

    #[test]
    fn blank_and_malformed_urls_are_validation_errors() {
        for url in ["", "   ", "cache:6379", "http://cache:6379"] {
            let error = RedisOnlineStoreOptions::new(url)
                .validate()
                .expect_err("invalid url");
            assert_eq!(error.kind(), ErrorKind::Validation, "url {url:?}");
        }
    }

    #[test]
    fn a_negative_database_is_a_validation_error() {
        let error = RedisOnlineStoreOptions {
            database: Some(-1),
            ..RedisOnlineStoreOptions::default()
        }
        .validate()
        .expect_err("negative db");
        assert_eq!(error.kind(), ErrorKind::Validation);
    }

    #[test]
    fn tls_and_cluster_urls_are_unsupported() {
        let tls = RedisOnlineStoreOptions::new("rediss://cache:6379/0")
            .validate()
            .expect_err("tls");
        assert!(matches!(tls, Error::Unsupported(_)));
        assert!(tls.to_string().contains("TLS"), "{tls}");

        let cluster = RedisOnlineStoreOptions::new("redis+cluster://cache:6379/0")
            .validate()
            .expect_err("cluster");
        assert!(matches!(cluster, Error::Unsupported(_)));
        assert!(cluster.to_string().contains("cluster"), "{cluster}");

        let tls_cluster = RedisOnlineStoreOptions::new("rediss+cluster://cache:6379/0")
            .validate()
            .expect_err("tls cluster");
        assert!(matches!(tls_cluster, Error::Unsupported(_)));
    }

    #[test]
    fn debug_redacts_the_password() {
        let options = RedisOnlineStoreOptions::new("redis://user:secret@cache:6379/0");
        let rendered = format!("{options:?}");
        assert!(!rendered.contains("secret"), "{rendered}");
        assert!(rendered.contains("cache"), "{rendered}");

        let broken = RedisOnlineStoreOptions::new("not-a-url");
        let rendered = format!("{broken:?}");
        assert!(rendered.contains("unparseable"), "{rendered}");
    }

    #[test]
    fn prefixes_are_normalized() {
        assert_eq!(normalize_prefix("team-a"), "team-a");
        assert_eq!(normalize_prefix("team-a:"), "team-a");
        assert_eq!(normalize_prefix(" team-a:: "), "team-a");
        assert_eq!(normalize_prefix(""), DEFAULT_PREFIX);
        assert_eq!(normalize_prefix(" : "), DEFAULT_PREFIX);
    }
}
