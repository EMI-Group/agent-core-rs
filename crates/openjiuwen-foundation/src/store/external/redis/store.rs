//! [`KvStore`] over a RESP connection.
//!
//! [`RedisKvStore`] serializes access to one [`RedisConnection`] behind a
//! [`tokio::sync::Mutex`], because the trait hands out `&self`. Every method maps
//! one logical operation onto one or a few Redis commands; the [`KvStore`]
//! documentation in [`crate::store::kv`] remains the behavioral contract, and the
//! in-memory backend there is the reference implementation.

use std::collections::BTreeMap;
use std::fmt;
use std::time::Duration;

use async_trait::async_trait;
use base64::Engine as _;
use tokio::sync::Mutex;

use openjiuwen_common::error::Result;

use super::client::RedisConnection;
use super::codec::{protocol_error, RespValue};
use super::RedisConfig;
use crate::store::kv::{KvPipeline, KvStore, KvValue, StorePipeline};

/// Marker that introduces a base64-encoded byte value.
///
/// A `NUL` byte cannot appear in a text payload this store writes on purpose, so
/// the marker distinguishes the two [`KvValue`] kinds unambiguously.
const BYTES_MARKER: &[u8] = b"\0ojb:";

/// Number of keys sent in one `MGET`/`DEL`/`SCAN` batch.
const BATCH: usize = 100;

/// A [`KvStore`] backed by a Redis server.
///
/// Cloning is not supported on purpose: one store owns one connection. Build one
/// per task with [`RedisConfig::connect`](super::RedisConfig::connect) or
/// [`RedisKvStore::connect`].
pub struct RedisKvStore {
    /// The single connection, serialized behind an async mutex.
    connection: Mutex<RedisConnection>,
    /// The configuration this store was built from (including `key_prefix`).
    config: RedisConfig,
}

impl fmt::Debug for RedisKvStore {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RedisKvStore")
            .field("connection", &self.connection)
            .field("key_prefix", &self.config.key_prefix)
            .finish_non_exhaustive()
    }
}

impl RedisKvStore {
    /// Connects a store from `config`.
    pub async fn connect(config: &RedisConfig) -> Result<Self> {
        config.connect().await
    }

    /// Wraps an already-connected connection and the configuration it uses.
    ///
    /// The configuration's `key_prefix` and timeout apply to every operation, so
    /// the connection is expected to have been opened with the same parameters.
    pub fn from_connection(connection: RedisConnection, config: RedisConfig) -> Self {
        Self {
            connection: Mutex::new(connection),
            config,
        }
    }

    /// The configuration this store was built from.
    pub fn config(&self) -> &RedisConfig {
        &self.config
    }

    /// Applies `key_prefix` to a caller key.
    fn prefixed(&self, key: &str) -> Vec<u8> {
        if self.config.key_prefix.is_empty() {
            key.as_bytes().to_vec()
        } else {
            format!("{}{key}", self.config.key_prefix).into_bytes()
        }
    }

    /// Sends one command over the shared connection.
    async fn execute(&self, args: &[&[u8]]) -> Result<RespValue> {
        self.connection.lock().await.command(args).await
    }

    /// Runs a `SCAN` for `prefix` over the shared connection.
    async fn scan(&self, prefix: &str) -> Result<Vec<String>> {
        self.connection.lock().await.scan_pattern(prefix).await
    }

    /// Encodes a value for storage.
    fn encode_value(value: &KvValue) -> Vec<u8> {
        match value {
            KvValue::Text(text) => text.as_bytes().to_vec(),
            KvValue::Bytes(bytes) => {
                let encoded = base64::engine::general_purpose::STANDARD.encode(bytes);
                let mut payload = BYTES_MARKER.to_vec();
                payload.extend_from_slice(encoded.as_bytes());
                payload
            }
        }
    }

    /// Decodes a stored payload.
    ///
    /// Nil is handled by the caller. A payload carrying [`BYTES_MARKER`] decodes
    /// from base64; valid UTF-8 becomes text; anything else stays opaque bytes, so
    /// a value written by another client survives the round trip unchanged.
    fn decode_value(payload: &[u8]) -> KvValue {
        if let Some(encoded) = payload.strip_prefix(BYTES_MARKER) {
            if let Ok(decoded) = base64::engine::general_purpose::STANDARD.decode(encoded) {
                return KvValue::Bytes(decoded);
            }
            return KvValue::Bytes(payload.to_vec());
        }
        match std::str::from_utf8(payload) {
            Ok(text) => KvValue::Text(text.to_string()),
            Err(_) => KvValue::Bytes(payload.to_vec()),
        }
    }
}

/// Accepts a simple-string acknowledgement as success.
fn expect_ok(reply: &RespValue, command: &str) -> Result<()> {
    match reply {
        RespValue::SimpleString(_) => Ok(()),
        other => Err(protocol_error(format!(
            "unexpected {command} reply: {other:?}"
        ))),
    }
}

/// Reads an integer reply, naming the command when the frame is not one.
fn expect_integer(reply: &RespValue, command: &str) -> Result<i64> {
    match reply {
        RespValue::Integer(count) => Ok(*count),
        other => Err(protocol_error(format!(
            "unexpected {command} reply: {other:?}"
        ))),
    }
}

/// Builds the `DEL key [key ...]` argument list.
fn del_args(keys: &[Vec<u8>]) -> Vec<&[u8]> {
    let mut args: Vec<&[u8]> = Vec::with_capacity(keys.len() + 1);
    args.push(b"DEL");
    args.extend(keys.iter().map(Vec::as_slice));
    args
}

/// Decodes a `GET`-style reply: a bulk string, or nil.
fn expect_value(reply: RespValue, command: &str) -> Result<Option<KvValue>> {
    match reply {
        RespValue::BulkString(Some(bytes)) => Ok(Some(RedisKvStore::decode_value(&bytes))),
        RespValue::BulkString(None) | RespValue::Null => Ok(None),
        other => Err(protocol_error(format!(
            "unexpected {command} reply: {other:?}"
        ))),
    }
}

#[async_trait]
impl KvStore for RedisKvStore {
    /// Stores the value with a plain `SET`.
    ///
    /// A plain `SET` drops any TTL the key carried, matching the trait contract
    /// ("stores or overwrites a key, clearing any TTL it had").
    async fn set(&self, key: &str, value: KvValue) -> Result<()> {
        let key = self.prefixed(key);
        let value = Self::encode_value(&value);
        let reply = self
            .execute(&[b"SET".as_slice(), key.as_slice(), value.as_slice()])
            .await?;
        expect_ok(&reply, "SET")
    }

    async fn get(&self, key: &str) -> Result<Option<KvValue>> {
        let key = self.prefixed(key);
        let reply = self.execute(&[b"GET".as_slice(), key.as_slice()]).await?;
        expect_value(reply, "GET")
    }

    async fn exists(&self, key: &str) -> Result<bool> {
        let key = self.prefixed(key);
        let reply = self
            .execute(&[b"EXISTS".as_slice(), key.as_slice()])
            .await?;
        Ok(expect_integer(&reply, "EXISTS")? > 0)
    }

    /// Removes the key with `DEL`; deleting a missing key is a no-op.
    async fn delete(&self, key: &str) -> Result<()> {
        let key = self.prefixed(key);
        let reply = self.execute(&[b"DEL".as_slice(), key.as_slice()]).await?;
        expect_integer(&reply, "DEL").map(|_| ())
    }

    /// Stores the value with `SET key value NX [EX seconds | PX milliseconds]`.
    ///
    /// TTL handling:
    ///
    /// - `None` stores without an expiry.
    /// - `Some(ttl)` with `ttl >= 1s` adds `EX <seconds>`.
    /// - `Some(ttl)` below one second adds `PX <milliseconds>` (at least `1`), so a
    ///   sub-second lock still expires.
    /// - `Some(Duration::ZERO)` is already expired at write time: nothing is sent
    ///   and `false` is returned, mirroring how an expired entry reads as absent.
    async fn exclusive_set(
        &self,
        key: &str,
        value: KvValue,
        expiry: Option<Duration>,
    ) -> Result<bool> {
        if expiry.is_some_and(|ttl| ttl.is_zero()) {
            return Ok(false);
        }
        let key = self.prefixed(key);
        let value = Self::encode_value(&value);
        let mut args: Vec<Vec<u8>> = vec![b"SET".to_vec(), key, value, b"NX".to_vec()];
        if let Some(ttl) = expiry {
            if ttl >= Duration::from_secs(1) {
                args.push(b"EX".to_vec());
                args.push(ttl.as_secs().max(1).to_string().into_bytes());
            } else {
                args.push(b"PX".to_vec());
                args.push(ttl.as_millis().max(1).to_string().into_bytes());
            }
        }
        let refs: Vec<&[u8]> = args.iter().map(Vec::as_slice).collect();
        let reply = self.execute(&refs).await?;
        match reply {
            RespValue::SimpleString(_) => Ok(true),
            RespValue::BulkString(None) | RespValue::Null => Ok(false),
            other => Err(protocol_error(format!(
                "unexpected SET NX reply: {other:?}"
            ))),
        }
    }

    /// Scans the keyspace, then reads each page with `MGET`.
    ///
    /// The returned map is ordered by key and carries keys without the
    /// configuration's `key_prefix`; the caller's `prefix` is retained, exactly
    /// like the reference backend. Keys that expired between the scan and the
    /// `MGET` are skipped.
    async fn get_by_prefix(&self, prefix: &str) -> Result<BTreeMap<String, KvValue>> {
        let scan_prefix = format!("{}{prefix}", self.config.key_prefix);
        let keys = self.scan(&scan_prefix).await?;
        let mut entries = BTreeMap::new();
        for chunk in keys.chunks(BATCH) {
            let values = self.connection.lock().await.mget(chunk).await?;
            for (key, value) in chunk.iter().zip(values) {
                let Some(value) = value else { continue };
                let key = key
                    .strip_prefix(&self.config.key_prefix)
                    .unwrap_or(key.as_str());
                entries.insert(key.to_string(), Self::decode_value(&value));
            }
        }
        Ok(entries)
    }

    /// Scans the keyspace, then removes each page with `DEL`.
    ///
    /// `scan` already returns fully prefixed keys (the store's `key_prefix` plus
    /// the caller's `prefix`), so they are passed to `DEL` verbatim: re-applying
    /// `key_prefix` would turn `app:app:key` into a miss and silently leak the
    /// scanned keys. The count is the number of live keys the server reported as
    /// deleted, so a key that expired after the scan does not inflate it.
    async fn delete_by_prefix(&self, prefix: &str) -> Result<usize> {
        let scan_prefix = format!("{}{prefix}", self.config.key_prefix);
        let keys = self.scan(&scan_prefix).await?;
        let mut deleted = 0;
        for chunk in keys.chunks(BATCH) {
            let scanned: Vec<Vec<u8>> = chunk.iter().map(|key| key.as_bytes().to_vec()).collect();
            let reply = self.execute(&del_args(&scanned)).await?;
            deleted += expect_integer(&reply, "DEL")?.max(0) as usize;
        }
        Ok(deleted)
    }

    /// Reads several keys with one `MGET`, preserving input order.
    async fn mget(&self, keys: &[String]) -> Result<Vec<Option<KvValue>>> {
        if keys.is_empty() {
            return Ok(Vec::new());
        }
        let prefixed: Vec<String> = keys.iter().map(|key| self.prefix_key(key)).collect();
        let values = self.connection.lock().await.mget(&prefixed).await?;
        Ok(values
            .into_iter()
            .map(|value| value.map(|bytes| Self::decode_value(&bytes)))
            .collect())
    }

    /// Removes several keys with chunked `DEL` calls, returning the live count.
    async fn batch_delete(&self, keys: &[String]) -> Result<usize> {
        if keys.is_empty() {
            return Ok(0);
        }
        let mut deleted = 0;
        for chunk in keys.chunks(BATCH) {
            let prefixed: Vec<Vec<u8>> = chunk.iter().map(|key| self.prefixed(key)).collect();
            let reply = self.execute(&del_args(&prefixed)).await?;
            deleted += expect_integer(&reply, "DEL")?.max(0) as usize;
        }
        Ok(deleted)
    }

    fn pipeline(&self) -> Box<dyn KvPipeline + '_> {
        Box::new(StorePipeline::new(self))
    }
}

impl RedisKvStore {
    /// Applies `key_prefix` to a caller key as a `String`.
    fn prefix_key(&self, key: &str) -> String {
        if self.config.key_prefix.is_empty() {
            key.to_string()
        } else {
            format!("{}{key}", self.config.key_prefix)
        }
    }
}
