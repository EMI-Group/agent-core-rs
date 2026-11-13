//! Live `Redis` store backends over the crate's RESP client.
//!
//! Three backends share one connection layer and one configuration type:
//!
//! - [`RedisStore`] — the key-value store of
//!   `openjiuwen/extensions/store/kv/redis_store.py`, which is the only `Redis`
//!   store the Python tree ships.
//! - [`RedisDocumentStore`] — JSON documents with the kernel [`DocumentStore`]
//!   filter/order/paging contract.
//! - [`RedisVectorStore`] — embeddings with the kernel [`VectorStore`] contract.
//!
//! The document and vector backends have **no Python counterpart** (the Python
//! `extensions/store` package holds only the Redis key-value store, GaussDB and
//! the Elasticsearch vector store), so their key layout and semantics are
//! documented here as a Rust-side extension of that family.
//!
//! The workspace vendors no `Redis` driver, so every backend speaks RESP
//! through [`crate::redis`]: [`RedisStoreConfig`] keeps the Python
//! configuration surface, each `new` validates it synchronously, and a lazily
//! opened [`tokio::sync::OnceCell`] client runs the commands. Every `connect`
//! opens the connection eagerly and verifies it with `PING`, so a bad URL or an
//! unreachable host fails before the caller stores anything.
//!
//! # Key layout
//!
//! Namespaces keep the three families apart, and every key component is
//! percent-escaped (`%` → `%25`, `:` → `%3A`) so a collection name or document
//! id can never blur a key boundary:
//!
//! | Key | Type | Backend |
//! | --- | --- | --- |
//! | `openjiuwen:doc:collections` | set | every document collection name |
//! | `openjiuwen:doc:ids:{collection}` | set | the ids stored in `collection` |
//! | `openjiuwen:doc:doc:{collection}:{id}` | string | one JSON `Document` |
//! | `openjiuwen:vec:collections` | set | every vector collection name |
//! | `openjiuwen:vec:schema:{collection}` | string | the JSON `CollectionSchema` |
//! | `openjiuwen:vec:ids:{collection}` | set | the primary keys of `collection` |
//! | `openjiuwen:vec:doc:{collection}:{id}` | string | one JSON document |
//!
//! Documents are addressed through their id set, never through `SCAN`, so ids
//! containing glob metacharacters (`*`, `?`, `[`, `]`, `\`) are safe. There is
//! no TTL anywhere: neither kernel contract has an expiry knob, so no expiry is
//! invented.
//!
//! # Shared semantics and delegations
//!
//! Both new backends keep the kernel's behavioral reference (`InMemoryDocumentStore`,
//! `InMemoryVectorStore`) authoritative: they hydrate a throwaway in-memory store
//! from `Redis`, delegate the operation to it, and persist the effect. Filtering,
//! ordering, paging, scoring, tie-breaks, schema validation and error messages
//! therefore cannot drift from the reference (the reference's `run_query` and
//! document preparation are private to `openjiuwen-foundation`, so delegation is
//! the only way to reuse them exactly).
//!
//! The price is an O(n) read of the collection per operation and no
//! cross-process locking: a single writer per collection is assumed. A partial
//! write (an interrupted batch) can leave the key-value view temporarily
//! inconsistent, because multi-key transactions are not used.
//!
//! # Deferred
//!
//! `rediss://` (TLS) and real cluster slot routing stay deferred and are
//! reported by [`crate::redis`] as [`Error::Unsupported`]. `RediSearch` module
//! commands (`FT.*`) are not available offline, so vector ranking is brute force
//! on the client (the same O(n) scan the in-memory reference performs).
#![allow(clippy::module_name_repetitions)] // `Redis*` names are fixed by the contract.
use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::fmt;
use tokio::sync::OnceCell;

use crate::redis::{Client, ConnectionOptions, Reply};

pub mod document;
pub mod kv;
pub mod vector;

pub use document::RedisDocumentStore;
pub use kv::{RedisPipeline, RedisStore};
pub use vector::RedisVectorStore;

/// The accepted connection-scheme prefixes.
///
/// `rediss://` is listed separately because it does not start with `redis://`.
pub(crate) const ACCEPTED_URL_SCHEMES: [&str; 4] = [
    "redis://",
    "rediss://",
    "redis+cluster://",
    "rediss+cluster://",
];
/// Number of keys or members per `MGET`/`DEL`/`SADD` chunk (Python's `batch_size` default).
pub(crate) const CHUNK_SIZE: usize = 500;
/// Key namespace of the document backend.
const DOCUMENT_NAMESPACE: &str = "openjiuwen:doc";
/// Key namespace of the vector backend.
const VECTOR_NAMESPACE: &str = "openjiuwen:vec";

/// Configuration for a live `Redis` backend.
///
/// Mirrors the Python constructor's inputs: a `url` or explicit
/// `connection_args`, plus an optional forced `cluster_mode`. When
/// `cluster_mode` is unset, the store constructors infer it from a `+cluster`
/// URL scheme. The same type configures [`RedisStore`], [`RedisDocumentStore`]
/// and [`RedisVectorStore`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "snake_case")]
pub struct RedisStoreConfig {
    /// Connection URL; one of `redis://`, `rediss://`, `redis+cluster://`, or
    /// `rediss+cluster://`.
    pub url: Option<String>,
    /// Whether to use a cluster client; inferred from the URL when `None`.
    pub cluster_mode: Option<bool>,
    /// Extra connection parameters, used when no `url` is supplied.
    pub connection_args: BTreeMap<String, Value>,
}

/// The shared connection layer of every live `Redis` backend.
///
/// [`RedisConnection::new`] normalizes the configuration once, so the three
/// stores validate URLs, infer `cluster_mode`, and resolve connection arguments
/// identically, and one lazily opened [`Client`] serves every operation. The
/// connection is opened on first use or eagerly by a store's `connect`.
pub(crate) struct RedisConnection {
    /// The normalized configuration.
    config: RedisStoreConfig,
    /// The lazily connected client.
    client: OnceCell<Client>,
}

impl fmt::Debug for RedisConnection {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        // `connection_args` can carry a password, so only its key names are
        // rendered; the URL user-info is redacted for the same reason.
        let connection_args: Vec<&str> = self
            .config
            .connection_args
            .keys()
            .map(String::as_str)
            .collect();
        formatter
            .debug_struct("RedisConnection")
            .field("url", &self.config.url.as_deref().map(redact_url))
            .field("cluster_mode", &self.config.cluster_mode)
            .field("connection_args", &connection_args)
            .field("connected", &self.is_connected())
            .finish()
    }
}

impl RedisConnection {
    /// Validates and normalizes `config`.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::Validation`](openjiuwen_common::error::ErrorKind::Validation)
    /// when neither a `url` nor a non-empty `connection_args` is present, or
    /// when the URL does not use an accepted scheme.
    pub(crate) fn new(mut config: RedisStoreConfig) -> Result<Self> {
        normalize(&mut config)?;
        Ok(Self {
            config,
            client: OnceCell::new(),
        })
    }

    /// The normalized configuration this connection was built from.
    pub(crate) fn config(&self) -> &RedisStoreConfig {
        &self.config
    }

    /// Whether the client has already been opened.
    pub(crate) fn is_connected(&self) -> bool {
        self.client.get().is_some()
    }

    /// The lazily connected client, built from the configuration.
    ///
    /// # Errors
    ///
    /// Surfaces the client's connection error
    /// ([`ErrorKind::Transport`](openjiuwen_common::error::ErrorKind::Transport),
    /// [`ErrorKind::Timeout`](openjiuwen_common::error::ErrorKind::Timeout), or
    /// [`ErrorKind::Unsupported`](openjiuwen_common::error::ErrorKind::Unsupported)
    /// for TLS and cluster redirections).
    pub(crate) async fn client(&self) -> Result<&Client> {
        self.client
            .get_or_try_init(|| async {
                let options = connection_options(&self.config)?;
                Client::connect(options).await
            })
            .await
    }

    /// Opens the connection (if needed) and verifies it with `PING`.
    ///
    /// # Errors
    ///
    /// Same as [`RedisConnection::client`], plus
    /// [`ErrorKind::Transport`](openjiuwen_common::error::ErrorKind::Transport)
    /// when the server does not answer `PING` with `PONG`.
    pub(crate) async fn ping(&self) -> Result<()> {
        self.client().await?.ping().await
    }

    /// Sends one command through the shared client.
    ///
    /// # Errors
    ///
    /// Same as [`Client::command`].
    pub(crate) async fn run(&self, args: &[&[u8]]) -> Result<Reply> {
        self.client().await?.command(args).await
    }

    /// Sends every command in one batch and returns the replies in order.
    ///
    /// # Errors
    ///
    /// Same as [`Client::pipeline`].
    pub(crate) async fn pipeline(&self, commands: &[Vec<Vec<u8>>]) -> Result<Vec<Reply>> {
        self.client().await?.pipeline(commands).await
    }

    /// Reads one string key, or `None` when it is missing.
    ///
    /// # Errors
    ///
    /// Same as [`Client::command`], plus `Transport` for a reply that is not a
    /// string.
    pub(crate) async fn get_bytes(&self, key: &str) -> Result<Option<Vec<u8>>> {
        let reply = self.run(&[b"GET", key.as_bytes()]).await?;
        Ok(reply.as_opt_bytes()?.map(<[u8]>::to_vec))
    }

    /// Writes one string key.
    ///
    /// # Errors
    ///
    /// Same as [`Client::command`], plus `Transport` when the server does not
    /// answer `+OK`.
    pub(crate) async fn set_bytes(&self, key: &str, payload: &[u8]) -> Result<()> {
        let reply = self.run(&[b"SET", key.as_bytes(), payload]).await?;
        expect_ok(&reply, "SET")
    }

    /// Reads `keys` with chunked `MGET`, keeping the input order.
    ///
    /// # Errors
    ///
    /// Same as [`Client::command`], plus `Transport` for a reply of the wrong
    /// shape.
    pub(crate) async fn mget_bytes(&self, keys: &[String]) -> Result<Vec<Option<Vec<u8>>>> {
        let mut values = Vec::with_capacity(keys.len());
        for chunk in keys.chunks(CHUNK_SIZE) {
            let mut args: Vec<&[u8]> = Vec::with_capacity(chunk.len() + 1);
            args.push(b"MGET");
            args.extend(chunk.iter().map(String::as_bytes));
            let reply = self.run(&args).await?;
            let replies = reply.as_array()?;
            if replies.len() != chunk.len() {
                return Err(Error::transport(format!(
                    "redis MGET returned {} values for {} keys",
                    replies.len(),
                    chunk.len()
                )));
            }
            for reply in replies {
                values.push(reply.as_opt_bytes()?.map(<[u8]>::to_vec));
            }
        }
        Ok(values)
    }

    /// The members of the set at `key`, in unspecified order.
    ///
    /// # Errors
    ///
    /// Same as [`Client::command`], plus `Transport` for a reply that is not an
    /// array of strings.
    pub(crate) async fn smembers(&self, key: &str) -> Result<Vec<String>> {
        let reply = self.run(&[b"SMEMBERS", key.as_bytes()]).await?;
        reply
            .as_array()?
            .iter()
            .map(Reply::as_string_owned)
            .collect()
    }

    /// Adds `members` to the set at `key`, in [`CHUNK_SIZE`] batches.
    ///
    /// # Errors
    ///
    /// Same as [`Client::command`], plus `Transport` when the server does not
    /// answer with an integer count.
    pub(crate) async fn sadd(&self, key: &str, members: &[String]) -> Result<()> {
        self.set_mutation(b"SADD", key, members).await
    }

    /// Removes `members` from the set at `key`, in [`CHUNK_SIZE`] batches.
    ///
    /// # Errors
    ///
    /// Same as [`Client::command`], plus `Transport` when the server does not
    /// answer with an integer count.
    pub(crate) async fn srem(&self, key: &str, members: &[String]) -> Result<()> {
        self.set_mutation(b"SREM", key, members).await
    }

    /// Deletes `keys` with chunked `DEL`, returning the number of live keys
    /// removed.
    ///
    /// # Errors
    ///
    /// Same as [`Client::command`], plus `Transport` when the server does not
    /// answer with a non-negative integer.
    pub(crate) async fn delete_keys(&self, keys: &[String]) -> Result<usize> {
        let mut deleted = 0_usize;
        for chunk in keys.chunks(CHUNK_SIZE) {
            let mut args: Vec<&[u8]> = Vec::with_capacity(chunk.len() + 1);
            args.push(b"DEL");
            args.extend(chunk.iter().map(String::as_bytes));
            let reply = self.run(&args).await?;
            deleted = deleted.saturating_add(reply_count(&reply)?);
        }
        Ok(deleted)
    }

    /// Runs one `SADD`/`SREM`-shaped mutation over `members`.
    async fn set_mutation(&self, command: &[u8], key: &str, members: &[String]) -> Result<()> {
        for chunk in members.chunks(CHUNK_SIZE) {
            let mut args: Vec<&[u8]> = Vec::with_capacity(chunk.len() + 2);
            args.push(command);
            args.push(key.as_bytes());
            args.extend(chunk.iter().map(String::as_bytes));
            let reply = self.run(&args).await?;
            reply.as_integer()?;
        }
        Ok(())
    }
}

/// The set of every collection name registered by `namespace`.
fn collections_key(namespace: &str) -> String {
    format!("{namespace}:collections")
}

/// The set of ids stored in `collection`.
fn ids_key(namespace: &str, collection: &str) -> String {
    format!("{namespace}:ids:{}", escape_component(collection))
}

/// The JSON key of document `id` inside `collection`.
fn document_key(namespace: &str, collection: &str, id: &str) -> String {
    format!(
        "{namespace}:doc:{}:{}",
        escape_component(collection),
        escape_component(id)
    )
}

/// The JSON key holding the schema of a vector collection.
fn schema_key(collection: &str) -> String {
    format!("{VECTOR_NAMESPACE}:schema:{}", escape_component(collection))
}

/// Escapes `:` and `%` so a key component cannot blur a key boundary.
///
/// Collection names and document ids may contain any character, including the
/// separator used between them; percent-escaping keeps `doc:a:b:c` unambiguous.
fn escape_component(component: &str) -> String {
    let mut escaped = String::with_capacity(component.len());
    for character in component.chars() {
        match character {
            '%' => escaped.push_str("%25"),
            ':' => escaped.push_str("%3A"),
            other => escaped.push(other),
        }
    }
    escaped
}

/// Rejects collection names that could escape or corrupt the key namespace.
///
/// The kernel's own `validate_collection` is `pub(crate)` to
/// `openjiuwen-foundation`, so the rule is replicated here once and used by both
/// store families: a name must be non-empty, must not contain `/`, `\`, or NUL,
/// and must not be `.` or `..`. The error names the collection.
///
/// # Errors
///
/// Returns [`ErrorKind::Validation`](openjiuwen_common::error::ErrorKind::Validation)
/// for an escaped or corrupt name.
pub(crate) fn validate_collection(collection: &str) -> Result<()> {
    let invalid = if collection.is_empty() {
        Some("must not be empty")
    } else if collection.contains('/') || collection.contains('\\') {
        Some("must not contain a path separator")
    } else if collection.contains('\0') {
        Some("must not contain a NUL byte")
    } else if collection == "." || collection == ".." {
        Some("must not be a dot segment")
    } else {
        None
    };
    match invalid {
        Some(reason) => Err(Error::validation(format!(
            "invalid collection name `{collection}`: {reason}"
        ))),
        None => Ok(()),
    }
}

/// Normalizes `config` in place, inferring `cluster_mode` and validating the URL.
fn normalize(config: &mut RedisStoreConfig) -> Result<()> {
    let url = config
        .url
        .take()
        .map(|url| url.trim().to_string())
        .filter(|url| !url.is_empty());
    if let Some(url) = url.as_deref() {
        validate_url(url)?;
    }
    if url.is_none() && config.connection_args.is_empty() {
        return Err(Error::validation(
            "redis store config requires a `url` or non-empty `connection_args`",
        ));
    }
    if config.cluster_mode.is_none() {
        config.cluster_mode = Some(url.as_deref().is_some_and(|url| url.contains("+cluster")));
    }
    config.url = url;
    Ok(())
}

/// Rejects a URL whose scheme is not a Redis scheme.
fn validate_url(url: &str) -> Result<()> {
    if ACCEPTED_URL_SCHEMES
        .iter()
        .any(|scheme| url.starts_with(scheme))
    {
        Ok(())
    } else {
        Err(Error::validation(format!(
            "redis url `{url}` must start with one of: redis://, rediss://, redis+cluster://, rediss+cluster://"
        )))
    }
}

/// Builds the client options from a normalized configuration.
fn connection_options(config: &RedisStoreConfig) -> Result<ConnectionOptions> {
    let mut options = match config.url.as_deref() {
        Some(url) => ConnectionOptions::from_url(url)?,
        None => ConnectionOptions::default(),
    };
    if let Some(host) = config.connection_args.get("host") {
        options.host = host_from_arg(host)?;
    }
    if let Some(port) = config.connection_args.get("port") {
        options.port = port_from_arg(port)?;
    }
    Ok(options
        .with_connection_args(&config.connection_args)?
        .with_cluster_mode(config.cluster_mode.unwrap_or(false)))
}

/// Reads the `host` connection argument.
fn host_from_arg(value: &Value) -> Result<String> {
    match value {
        Value::String(host) if !host.trim().is_empty() => Ok(host.trim().to_string()),
        Value::String(_) => Err(Error::validation(
            "redis connection argument 'host' must not be empty",
        )),
        _ => Err(Error::validation(
            "redis connection argument 'host' must be a string",
        )),
    }
}

/// Reads the `port` connection argument (an integer or a numeric string).
fn port_from_arg(value: &Value) -> Result<u16> {
    let port = match value {
        Value::Number(number) => number.as_u64().and_then(|port| u16::try_from(port).ok()),
        Value::String(text) => text.trim().parse::<u16>().ok(),
        _ => None,
    };
    port.filter(|port| *port != 0).ok_or_else(|| {
        Error::validation("redis connection argument 'port' must be an integer between 1 and 65535")
    })
}

/// Redacts the user-info of a URL so `Debug` output never carries credentials.
fn redact_url(url: &str) -> String {
    let (scheme, rest) = url.split_once("://").unwrap_or(("", url));
    let rest = rest.rsplit_once('@').map_or(rest, |(_, host)| host);
    if scheme.is_empty() {
        rest.to_string()
    } else {
        format!("{scheme}://{rest}")
    }
}

/// Whether a status reply is `+OK`.
fn is_ok_status(reply: &Reply) -> bool {
    match reply {
        Reply::Simple(text) => text.eq_ignore_ascii_case("OK"),
        Reply::Bulk(bytes) => bytes.eq_ignore_ascii_case(b"OK"),
        _ => false,
    }
}

/// Requires `reply` to be the `+OK` status of `command`.
fn expect_ok(reply: &Reply, command: &str) -> Result<()> {
    if is_ok_status(reply) {
        Ok(())
    } else {
        Err(unexpected(command, reply))
    }
}

/// The transport error for an unexpected reply to `command`.
fn unexpected(command: &str, reply: &Reply) -> Error {
    Error::transport(format!(
        "redis {command} returned an unexpected {} reply",
        reply.kind_name()
    ))
}

/// Interprets a delete reply as the number of live keys removed.
fn reply_count(reply: &Reply) -> Result<usize> {
    let count = reply.as_integer()?;
    usize::try_from(count)
        .map_err(|_| Error::transport(format!("redis returned a negative delete count: {count}")))
}

#[cfg(test)]
mod tests;
