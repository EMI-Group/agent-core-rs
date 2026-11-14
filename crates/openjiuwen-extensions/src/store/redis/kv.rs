//! Live `Redis` key-value backend over the crate's RESP client.
//!
//! Ports `openjiuwen/extensions/store/kv/redis_store.py` (`RedisStore`), which
//! wraps `redis.asyncio`. The workspace vendors no `Redis` driver, so this
//! backend speaks RESP through [`crate::redis`] instead: [`RedisStoreConfig`]
//! keeps the Python configuration surface, [`RedisStore::new`] validates it
//! synchronously, and the shared `RedisConnection` opens one client on first
//! use. [`RedisStore::connect`] opens the connection eagerly and verifies it
//! with `PING`, so a bad URL or an unreachable host fails before the caller
//! stores anything.
//!
//! # Key and value encoding
//!
//! Keys are used verbatim; Python's `index_prefix`-style namespacing stays the
//! caller's responsibility. (The document and vector backends next to this
//! module do namespace their keys, because they own their whole key space.)
//!
//! RESP has no text/bytes distinction, but the kernel contract requires a
//! [`KvValue`] to keep its variant across a round trip. Every value is therefore
//! stored as a one-byte kind tag (`0x00` text, `0x01` bytes) followed by the raw
//! payload, and a payload with a missing or unknown tag is a validation error.
//! **This is an intentional deviation from Python**, whose `RedisStore` writes
//! the bare payload (a `str` encoded as UTF-8 or the `bytes` unchanged): values
//! written by the Python backend are not directly readable here (and vice
//! versa) and must be re-encoded.
//!
//! `set` issues a plain `SET`, which drops any TTL the key had, exactly as the
//! kernel contract requires. `exclusive_set` issues `SET key value NX [PX ms]`
//! and rounds a sub-millisecond expiry up to one millisecond so a near-zero TTL
//! never degrades into "no expiry".
//!
//! # Prefix and batch operations
//!
//! `get_by_prefix` and `delete_by_prefix` walk `SCAN` with
//! `MATCH <escaped-prefix>*` and `COUNT 500`, then apply `MGET` or `UNLINK` in
//! 500-key chunks. The glob metacharacters `*`, `?`, `[`, `]`, and `\` in the
//! prefix are backslash-escaped, so a key that literally contains them is not
//! over-matched. `batch_delete` chunks the same way, and both delete paths sum
//! the server's integer replies so only live keys count. Python's optional
//! `batch_size` argument has no counterpart: the kernel [`KvStore`] trait takes
//! the whole prefix or key set, so 500 is a fixed chunk size.
//!
//! `pipeline` queues `set`/`get`/`exists` and sends every command in one
//! write-and-read batch ([`Client::pipeline`]) instead of reusing the kernel's
//! per-operation `StorePipeline`, which would issue one round trip per op.
//! Redis executes a batch in order, so a queued `get` observes an earlier
//! queued `set`; the queue is cleared once a batch has been executed.
//!
//! # Deferred
//!
//! `rediss://` (TLS) and real cluster slot routing stay deferred and are
//! reported by [`crate::redis`] as [`Error::Unsupported`]. This module is a
//! key-value store only; documents and vectors live in [`super::document`] and
//! [`super::vector`].
//!
//! Python counterpart tests:
//! `tests/unit_tests/extensions/checkpointer/test_redis_checkpointer.py`
//! exercises `Redis` through the checkpointer; the Python tree has no dedicated
//! `RedisStore` unit test.
#![allow(clippy::module_name_repetitions)] // `Redis*` names are fixed by the contract.
use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_core::foundation::store::{KvPipeline, KvStore, KvValue};
use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use super::{
    expect_ok, is_ok_status, reply_count, unexpected, RedisConnection, RedisStoreConfig, CHUNK_SIZE,
};
use crate::redis::{Client, Reply};

/// Kind tag stored in front of a [`KvValue::Text`] payload.
pub(crate) const TEXT_TAG: u8 = 0x00;
/// Kind tag stored in front of a [`KvValue::Bytes`] payload.
pub(crate) const BYTES_TAG: u8 = 0x01;
/// Upper bound on `SCAN` rounds, so a misbehaving server cannot loop forever.
const MAX_SCAN_ROUNDS: usize = 10_000;

/// A live `Redis` key-value store over the in-crate RESP client.
///
/// [`RedisStore::new`] only validates the configuration; the client is opened on
/// first use (or eagerly by [`RedisStore::connect`]). The store is `Send +
/// Sync`, so it can be shared behind an `Arc` like every kernel [`KvStore`].
/// It is not `Clone` (a connection is shared, not duplicated); use [`RedisStore::connect`]
/// to obtain the shared `Arc<dyn KvStore>`.
pub struct RedisStore {
    connection: RedisConnection,
}

impl fmt::Debug for RedisStore {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        // `connection_args` can carry a password, so only its key names are
        // rendered; the URL user-info is redacted for the same reason.
        formatter
            .debug_struct("RedisStore")
            .field("connection", &self.connection)
            .finish()
    }
}

impl RedisStore {
    /// Validates `config` and wraps it in a store.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::Validation`](openjiuwen_common::error::ErrorKind::Validation)
    /// when neither a `url` nor a non-empty `connection_args` is present, or
    /// when the URL does not use an accepted scheme.
    pub fn new(config: RedisStoreConfig) -> Result<Self> {
        Ok(Self {
            connection: RedisConnection::new(config)?,
        })
    }

    /// Validates `config`, connects eagerly, and returns the shared store.
    ///
    /// The connection is verified with `PING`, so a bad URL, an unreachable
    /// host, or a server that only answers with cluster redirections fails here
    /// instead of on the caller's first operation.
    ///
    /// # Errors
    ///
    /// Returns the validation errors of [`RedisStore::new`], followed by the
    /// client's connection error ([`ErrorKind::Transport`](openjiuwen_common::error::ErrorKind::Transport),
    /// [`ErrorKind::Timeout`](openjiuwen_common::error::ErrorKind::Timeout), or
    /// [`ErrorKind::Unsupported`](openjiuwen_common::error::ErrorKind::Unsupported)
    /// for TLS and cluster redirections).
    pub async fn connect(config: RedisStoreConfig) -> Result<Arc<dyn KvStore>> {
        let store = Self::new(config)?;
        store.ping().await?;
        Ok(Arc::new(store))
    }

    /// The normalized configuration this store was built from.
    #[must_use]
    pub fn config(&self) -> &RedisStoreConfig {
        self.connection.config()
    }

    /// Opens the connection (if needed) and verifies it with `PING`.
    ///
    /// # Errors
    ///
    /// Same as [`RedisStore::connect`] after validation.
    pub async fn ping(&self) -> Result<()> {
        self.connection.ping().await
    }

    /// The lazily connected client, built from [`RedisStoreConfig`].
    async fn client(&self) -> Result<&Client> {
        self.connection.client().await
    }

    /// Sends one command through the shared client.
    async fn run(&self, args: &[&[u8]]) -> Result<Reply> {
        self.connection.run(args).await
    }

    /// Deletes `keys` with `UNLINK`, falling back to `DEL` when the server does
    /// not know `UNLINK`.
    async fn delete_chunk(&self, keys: &[String]) -> Result<usize> {
        let client = self.client().await?;
        match client.command(&delete_args(b"UNLINK", keys)).await {
            Ok(reply) => reply_count(&reply),
            Err(error) if is_unknown_command(&error, "UNLINK") => {
                let reply = client.command(&delete_args(b"DEL", keys)).await?;
                reply_count(&reply)
            }
            Err(error) => Err(error),
        }
    }

    /// Walks `SCAN` for every key starting with `prefix`.
    async fn scan_keys(&self, prefix: &str) -> Result<Vec<String>> {
        let pattern = format!("{}*", escape_glob(prefix));
        let client = self.client().await?;
        let mut cursor = "0".to_string();
        let mut keys = Vec::new();
        for _ in 0..MAX_SCAN_ROUNDS {
            let reply = client
                .command_str(&["SCAN", &cursor, "MATCH", &pattern, "COUNT", "500"])
                .await?;
            let [next, found] = reply.as_array()? else {
                return Err(Error::transport(
                    "redis SCAN returned an unexpected reply shape",
                ));
            };
            cursor = next.as_string_owned()?;
            for key in found.as_array()? {
                keys.push(key.as_string_owned()?);
            }
            if cursor == "0" {
                return Ok(keys);
            }
        }
        Err(Error::transport(format!(
            "redis SCAN did not finish after {MAX_SCAN_ROUNDS} rounds"
        )))
    }
}

#[async_trait]
impl KvStore for RedisStore {
    async fn set(&self, key: &str, value: KvValue) -> Result<()> {
        let payload = encode_value(&value);
        let reply = self
            .run(&[b"SET", key.as_bytes(), payload.as_slice()])
            .await?;
        expect_ok(&reply, "SET")
    }

    async fn get(&self, key: &str) -> Result<Option<KvValue>> {
        let reply = self.run(&[b"GET", key.as_bytes()]).await?;
        reply
            .as_opt_bytes()?
            .map_or(Ok(None), |payload| decode_value(key, payload).map(Some))
    }

    async fn exists(&self, key: &str) -> Result<bool> {
        let reply = self.run(&[b"EXISTS", key.as_bytes()]).await?;
        Ok(reply.as_integer()? > 0)
    }

    async fn delete(&self, key: &str) -> Result<()> {
        let reply = self.run(&[b"DEL", key.as_bytes()]).await?;
        let _count = reply.as_integer()?;
        Ok(())
    }

    async fn exclusive_set(
        &self,
        key: &str,
        value: KvValue,
        expiry: Option<Duration>,
    ) -> Result<bool> {
        let payload = encode_value(&value);
        let mut args: Vec<&[u8]> = vec![b"SET", key.as_bytes(), payload.as_slice(), b"NX"];
        let ttl = expiry.map(expiry_millis).map(|millis| millis.to_string());
        if let Some(ttl) = ttl.as_deref() {
            args.push(b"PX");
            args.push(ttl.as_bytes());
        }
        let reply = self.run(&args).await?;
        match &reply {
            Reply::Null => Ok(false),
            other if is_ok_status(other) => Ok(true),
            other => Err(unexpected("SET NX", other)),
        }
    }

    async fn get_by_prefix(&self, prefix: &str) -> Result<BTreeMap<String, KvValue>> {
        let keys = self.scan_keys(prefix).await?;
        let mut entries = BTreeMap::new();
        for chunk in keys.chunks(CHUNK_SIZE) {
            // Keys that vanished between SCAN and MGET read as `None` and are
            // skipped, matching the kernel's lazy-expiry semantics.
            for (key, value) in chunk.iter().zip(self.mget(chunk).await?) {
                if let Some(value) = value {
                    entries.insert(key.clone(), value);
                }
            }
        }
        Ok(entries)
    }

    async fn delete_by_prefix(&self, prefix: &str) -> Result<usize> {
        let keys = self.scan_keys(prefix).await?;
        let mut deleted = 0_usize;
        for chunk in keys.chunks(CHUNK_SIZE) {
            deleted = deleted.saturating_add(self.delete_chunk(chunk).await?);
        }
        Ok(deleted)
    }

    async fn mget(&self, keys: &[String]) -> Result<Vec<Option<KvValue>>> {
        if keys.is_empty() {
            return Ok(Vec::new());
        }
        let mut args: Vec<&[u8]> = Vec::with_capacity(keys.len() + 1);
        args.push(b"MGET");
        args.extend(keys.iter().map(String::as_bytes));
        let reply = self.client().await?.command(&args).await?;
        let values = reply.as_array()?;
        if values.len() != keys.len() {
            return Err(Error::transport(format!(
                "redis MGET returned {} values for {} keys",
                values.len(),
                keys.len()
            )));
        }
        values
            .iter()
            .zip(keys)
            .map(|(reply, key)| {
                reply
                    .as_opt_bytes()?
                    .map_or(Ok(None), |payload| decode_value(key, payload).map(Some))
            })
            .collect()
    }

    async fn batch_delete(&self, keys: &[String]) -> Result<usize> {
        let mut deleted = 0_usize;
        for chunk in keys.chunks(CHUNK_SIZE) {
            deleted = deleted.saturating_add(self.delete_chunk(chunk).await?);
        }
        Ok(deleted)
    }

    fn pipeline(&self) -> Box<dyn KvPipeline + '_> {
        Box::new(RedisPipeline::new(self))
    }
}

/// One queued pipeline operation.
enum QueuedOp {
    /// A write, with the already encoded payload.
    Set { key: String, payload: Vec<u8> },
    /// A read.
    Get(String),
    /// A key probe.
    Exists(String),
}

/// A real batched [`KvPipeline`] over `Redis`.
///
/// Obtained from [`KvStore::pipeline`]; it is public so a caller can name the
/// type it gets back without importing a private item.
///
/// The kernel's `StorePipeline` issues one round trip per operation; this
/// pipeline instead queues operations and sends them as a single
/// [`Client::pipeline`] batch, which is the point of `Redis` pipelining.
pub struct RedisPipeline<'a> {
    store: &'a RedisStore,
    ops: Vec<QueuedOp>,
}

impl<'a> RedisPipeline<'a> {
    /// Binds an empty pipeline to `store`.
    fn new(store: &'a RedisStore) -> Self {
        Self {
            store,
            ops: Vec::new(),
        }
    }
}

#[async_trait]
impl KvPipeline for RedisPipeline<'_> {
    fn set(&mut self, key: &str, value: KvValue) {
        self.ops.push(QueuedOp::Set {
            key: key.to_string(),
            payload: encode_value(&value),
        });
    }

    fn get(&mut self, key: &str) {
        self.ops.push(QueuedOp::Get(key.to_string()));
    }

    fn exists(&mut self, key: &str) {
        self.ops.push(QueuedOp::Exists(key.to_string()));
    }

    async fn execute(&mut self) -> Result<Vec<Option<KvValue>>> {
        // Taking the queue clears it, so reusing the pipeline never replays it.
        let ops = std::mem::take(&mut self.ops);
        if ops.is_empty() {
            return Ok(Vec::new());
        }
        let commands: Vec<Vec<Vec<u8>>> = ops.iter().map(command_for).collect();
        let replies = self.store.client().await?.pipeline(&commands).await?;
        if replies.len() != ops.len() {
            return Err(Error::transport(format!(
                "redis pipeline returned {} replies for {} commands",
                replies.len(),
                ops.len()
            )));
        }
        let mut results = Vec::with_capacity(ops.len());
        for (op, reply) in ops.iter().zip(replies) {
            match op {
                QueuedOp::Set { .. } => {
                    expect_ok(&reply, "SET")?;
                    results.push(None);
                }
                QueuedOp::Get(key) => results.push(
                    reply
                        .as_opt_bytes()?
                        .map_or(Ok(None), |payload| decode_value(key, payload).map(Some))?,
                ),
                QueuedOp::Exists(_) => {
                    results.push(Some(KvValue::text(if reply.as_integer()? > 0 {
                        "true"
                    } else {
                        "false"
                    })));
                }
            }
        }
        Ok(results)
    }
}

/// Encodes a value as its kind tag followed by the raw payload.
fn encode_value(value: &KvValue) -> Vec<u8> {
    let payload = value.as_bytes();
    let mut encoded = Vec::with_capacity(payload.len() + 1);
    encoded.push(if value.is_text() { TEXT_TAG } else { BYTES_TAG });
    encoded.extend_from_slice(payload);
    encoded
}

/// Decodes a stored payload back into the [`KvValue`] variant it was written as.
pub(crate) fn decode_value(key: &str, payload: &[u8]) -> Result<KvValue> {
    let (tag, rest) = payload.split_first().ok_or_else(|| {
        Error::validation(format!(
            "redis key `{key}` holds an empty value; expected a kind tag followed by the payload"
        ))
    })?;
    match *tag {
        TEXT_TAG => std::str::from_utf8(rest)
            .map(|text| KvValue::text(text.to_string()))
            .map_err(|error| {
                Error::validation(format!(
                    "redis key `{key}` holds a text value that is not UTF-8: {error}"
                ))
            }),
        BYTES_TAG => Ok(KvValue::bytes(rest.to_vec())),
        other => Err(Error::validation(format!(
            "redis key `{key}` holds an unknown value kind tag {other:#04x}"
        ))),
    }
}

/// The `PX` argument for an expiry, rounding sub-millisecond values up to 1 ms.
pub(crate) fn expiry_millis(expiry: Duration) -> u64 {
    let millis = u64::try_from(expiry.as_millis()).unwrap_or(u64::MAX);
    millis.max(1)
}

/// Escapes the glob metacharacters of a key prefix.
///
/// `SCAN MATCH` treats `*` and `?` as wildcards and `[...]` as a character
/// class, with `\` as the escape character; a literal occurrence of any of them
/// in the prefix must be escaped to avoid over-matching.
pub(crate) fn escape_glob(prefix: &str) -> String {
    let mut escaped = String::with_capacity(prefix.len());
    for character in prefix.chars() {
        if matches!(character, '*' | '?' | '[' | ']' | '\\') {
            escaped.push('\\');
        }
        escaped.push(character);
    }
    escaped
}

/// Builds the argument vector of a multi-key `UNLINK`/`DEL`.
fn delete_args<'a>(command: &'a [u8], keys: &'a [String]) -> Vec<&'a [u8]> {
    let mut args = Vec::with_capacity(keys.len() + 1);
    args.push(command);
    args.extend(keys.iter().map(String::as_bytes));
    args
}

/// The wire command for a queued pipeline operation.
fn command_for(op: &QueuedOp) -> Vec<Vec<u8>> {
    match op {
        QueuedOp::Set { key, payload } => {
            vec![b"SET".to_vec(), key.as_bytes().to_vec(), payload.clone()]
        }
        QueuedOp::Get(key) => vec![b"GET".to_vec(), key.as_bytes().to_vec()],
        QueuedOp::Exists(key) => vec![b"EXISTS".to_vec(), key.as_bytes().to_vec()],
    }
}

/// Whether `error` is the server rejecting `command` as unknown.
pub(crate) fn is_unknown_command(error: &Error, command: &str) -> bool {
    let message = error.message().to_ascii_lowercase();
    message.contains("unknown command") && message.contains(&command.to_ascii_lowercase())
}
