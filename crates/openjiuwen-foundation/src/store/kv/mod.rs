//! Key-value storage.
//!
//! The trait mirrors the Python `BaseKVStore` contract: string keys, text-or-bytes
//! values, prefix scans, `mget` in input order, batch delete counts, a
//! create-only `exclusive_set` with optional TTL, and a lightweight pipeline.
//!
//! Four backends implement [`KvStore`]:
//!
//! - [`InMemoryKvStore`] — the behavioral reference (a `BTreeMap` behind a lock).
//! - [`LocalFileKvStore`] — one JSON envelope per key below a caller-provided root.
//! - `SqliteKvStore` — `SQLite` behind the `sqlite` feature.
//! - `RedisKvStore` — a Redis server over RESP2, in
//!   [`external::redis`](super::external::redis).
//!
//! All four share the same [`StorePipeline`], so pipeline ordering and results
//! cannot drift between backends.

use std::collections::BTreeMap;
use std::time::Duration;

use async_trait::async_trait;
use base64::Engine as _;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use openjiuwen_common::error::{Error, Result};

pub mod file;
pub mod memory;
#[cfg(feature = "sqlite")]
pub mod sqlite;

pub use file::LocalFileKvStore;
pub use memory::InMemoryKvStore;
#[cfg(feature = "sqlite")]
pub use sqlite::SqliteKvStore;

/// The object key used by the JSON representation of byte values.
const BYTES_KEY: &str = "$bytes";

/// A stored value: UTF-8 text or opaque bytes.
///
/// This is Rust's replacement for Python's `str | bytes` union. Both variants are
/// preserved through [`KvValue::to_json`] and [`KvValue::from_json`], so a store
/// round-trip never changes a value's kind.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KvValue {
    /// A UTF-8 string value (Python `str`).
    Text(String),
    /// An opaque byte value (Python `bytes`).
    Bytes(Vec<u8>),
}

impl KvValue {
    /// Builds a text value.
    pub fn text(value: impl Into<String>) -> Self {
        Self::Text(value.into())
    }

    /// Builds a byte value.
    pub fn bytes(value: impl Into<Vec<u8>>) -> Self {
        Self::Bytes(value.into())
    }

    /// Decodes the JSON representation of a value.
    ///
    /// A JSON string becomes [`KvValue::Text`]; an object carrying a string
    /// `"$bytes"` member becomes [`KvValue::Bytes`] (its payload is base64); any
    /// other JSON value is rendered as text with its JSON form (`null`, `42`,
    /// `{"a":1}`), which keeps the converter total for tolerant input.
    pub fn from_json(value: &Value) -> Result<Self> {
        match value {
            Value::String(text) => Ok(Self::Text(text.clone())),
            Value::Object(map) => match map.get(BYTES_KEY) {
                Some(Value::String(encoded)) => {
                    let decoded = base64::engine::general_purpose::STANDARD
                        .decode(encoded)
                        .map_err(Error::external)?;
                    Ok(Self::Bytes(decoded))
                }
                _ => Ok(Self::Text(value.to_string())),
            },
            other => Ok(Self::Text(other.to_string())),
        }
    }

    /// Encodes the value as JSON: text as a string, bytes as
    /// `{"$bytes": "<base64>"}`.
    pub fn to_json(&self) -> Result<Value> {
        Ok(match self {
            Self::Text(text) => Value::String(text.clone()),
            Self::Bytes(bytes) => {
                let encoded = base64::engine::general_purpose::STANDARD.encode(bytes);
                let mut map = serde_json::Map::new();
                map.insert(BYTES_KEY.to_string(), Value::String(encoded));
                Value::Object(map)
            }
        })
    }

    /// The text payload, or `None` for a byte value.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::Text(text) => Some(text),
            Self::Bytes(_) => None,
        }
    }

    /// The value bytes: the UTF-8 encoding for text, the payload for bytes.
    pub fn as_bytes(&self) -> &[u8] {
        match self {
            Self::Text(text) => text.as_bytes(),
            Self::Bytes(bytes) => bytes,
        }
    }

    /// Whether the value is text (Python `isinstance(value, str)`).
    pub const fn is_text(&self) -> bool {
        matches!(self, Self::Text(_))
    }
}

impl From<String> for KvValue {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

impl From<&str> for KvValue {
    fn from(value: &str) -> Self {
        Self::Text(value.to_string())
    }
}

impl From<Vec<u8>> for KvValue {
    fn from(value: Vec<u8>) -> Self {
        Self::Bytes(value)
    }
}

impl From<&[u8]> for KvValue {
    fn from(value: &[u8]) -> Self {
        Self::Bytes(value.to_vec())
    }
}

impl Serialize for KvValue {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        self.to_json()
            .map_err(serde::ser::Error::custom)?
            .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for KvValue {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        let value = Value::deserialize(deserializer)?;
        Self::from_json(&value).map_err(serde::de::Error::custom)
    }
}

/// The contract every key-value backend implements.
///
/// Semantics (Python parity, see `src/store/CONTEXT.md`):
///
/// - A missing key reads as `None`; deleting a missing key is a no-op.
/// - `mget` preserves input order and keeps a `None` placeholder per miss.
/// - `get_by_prefix("")` returns every live entry, ordered by key.
/// - `exclusive_set` writes only when the key is absent or expired and accepts an
///   absolute TTL (`Duration` from now); it returns `true` when it wrote.
/// - TTLs are honored lazily on every read path: an expired entry reads as absent
///   and the backends purge it as they notice it.
#[async_trait]
pub trait KvStore: Send + Sync {
    /// Stores or overwrites a key, clearing any TTL it had.
    async fn set(&self, key: &str, value: KvValue) -> Result<()>;

    /// Reads a key, or `None` when it is absent or expired.
    async fn get(&self, key: &str) -> Result<Option<KvValue>>;

    /// Whether a live (present and unexpired) key exists.
    async fn exists(&self, key: &str) -> Result<bool>;

    /// Removes a key; removing a missing key is a no-op.
    async fn delete(&self, key: &str) -> Result<()>;

    /// Stores a key only when it is absent or expired.
    ///
    /// `expiry` is a duration from now; `None` stores the key without a TTL.
    /// Returns `true` when the value was written.
    async fn exclusive_set(
        &self,
        key: &str,
        value: KvValue,
        expiry: Option<Duration>,
    ) -> Result<bool>;

    /// Reads every live entry whose key starts with `prefix`, ordered by key.
    async fn get_by_prefix(&self, prefix: &str) -> Result<BTreeMap<String, KvValue>>;

    /// Removes every entry whose key starts with `prefix`, returning how many live
    /// entries were deleted.
    async fn delete_by_prefix(&self, prefix: &str) -> Result<usize>;

    /// Reads several keys at once, preserving input order and leaving `None` for
    /// misses.
    async fn mget(&self, keys: &[String]) -> Result<Vec<Option<KvValue>>>;

    /// Removes several keys, returning how many live entries were deleted.
    async fn batch_delete(&self, keys: &[String]) -> Result<usize>;

    /// Creates a pipeline that queues `set`/`get`/`exists` operations.
    fn pipeline(&self) -> Box<dyn KvPipeline + '_>;
}

/// Batched operations against a [`KvStore`].
///
/// Operations are queued synchronously and applied in insertion order by
/// [`KvPipeline::execute`], so a `get` observes an earlier queued `set`.
#[async_trait]
pub trait KvPipeline: Send {
    /// Queues a `set` operation.
    fn set(&mut self, key: &str, value: KvValue);

    /// Queues a `get` operation.
    fn get(&mut self, key: &str);

    /// Queues an `exists` operation.
    fn exists(&mut self, key: &str);

    /// Applies the queued operations in order and clears the queue.
    ///
    /// A `set` yields `None`; a `get` yields the stored option; an `exists` yields
    /// [`KvValue::Text`] holding `"true"` or `"false"`.
    async fn execute(&mut self) -> Result<Vec<Option<KvValue>>>;
}

/// One queued pipeline operation.
#[derive(Debug, Clone, PartialEq, Eq)]
enum QueuedOp {
    /// Write a value.
    Set { key: String, value: KvValue },
    /// Read a value.
    Get(String),
    /// Probe a key.
    Exists(String),
}

/// The shared [`KvPipeline`] implementation used by every backend.
///
/// It holds a borrowed store and a queue of operations; each backend's
/// `pipeline()` is exactly `Box::new(StorePipeline::new(self))`.
pub struct StorePipeline<'a> {
    store: &'a dyn KvStore,
    ops: Vec<QueuedOp>,
}

impl<'a> StorePipeline<'a> {
    /// Creates an empty pipeline bound to `store`.
    pub fn new(store: &'a dyn KvStore) -> Self {
        Self {
            store,
            ops: Vec::new(),
        }
    }

    /// Number of operations currently queued.
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    /// Whether no operation is queued.
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }
}

#[async_trait]
impl KvPipeline for StorePipeline<'_> {
    fn set(&mut self, key: &str, value: KvValue) {
        self.ops.push(QueuedOp::Set {
            key: key.to_string(),
            value,
        });
    }

    fn get(&mut self, key: &str) {
        self.ops.push(QueuedOp::Get(key.to_string()));
    }

    fn exists(&mut self, key: &str) {
        self.ops.push(QueuedOp::Exists(key.to_string()));
    }

    async fn execute(&mut self) -> Result<Vec<Option<KvValue>>> {
        let ops = std::mem::take(&mut self.ops);
        let mut results = Vec::with_capacity(ops.len());
        for op in ops {
            match op {
                QueuedOp::Set { key, value } => {
                    self.store.set(&key, value).await?;
                    results.push(None);
                }
                QueuedOp::Get(key) => results.push(self.store.get(&key).await?),
                QueuedOp::Exists(key) => {
                    let exists = self.store.exists(&key).await?;
                    results.push(Some(KvValue::text(if exists { "true" } else { "false" })));
                }
            }
        }
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn text_value_json_round_trip() {
        let value = KvValue::text("hello");
        let json = value.to_json().expect("encode");
        assert_eq!(json, json!("hello"));
        assert_eq!(KvValue::from_json(&json).expect("decode"), value);
    }

    #[test]
    fn bytes_value_json_uses_base64_envelope() {
        let value = KvValue::bytes(vec![0_u8, 1, 2, 255]);
        let json = value.to_json().expect("encode");
        assert_eq!(json, json!({"$bytes": "AAEC/w=="}));
        assert_eq!(KvValue::from_json(&json).expect("decode"), value);
    }

    #[test]
    fn from_json_is_total_for_non_string_values() {
        assert_eq!(
            KvValue::from_json(&json!(42)).expect("decode"),
            KvValue::Text("42".to_string())
        );
        assert_eq!(
            KvValue::from_json(&json!(null)).expect("decode"),
            KvValue::Text("null".to_string())
        );
        assert_eq!(
            KvValue::from_json(&json!({"a": 1})).expect("decode"),
            KvValue::Text("{\"a\":1}".to_string())
        );
    }

    #[test]
    fn from_json_rejects_malformed_base64() {
        let error = KvValue::from_json(&json!({"$bytes": "***"})).expect_err("invalid base64");
        assert_eq!(error.kind(), openjiuwen_common::error::ErrorKind::External);
    }

    #[test]
    fn accessors_distinguish_text_and_bytes() {
        let text = KvValue::text("abc");
        assert!(text.is_text());
        assert_eq!(text.as_str(), Some("abc"));
        assert_eq!(text.as_bytes(), b"abc");

        let bytes = KvValue::bytes(vec![1_u8, 2]);
        assert!(!bytes.is_text());
        assert_eq!(bytes.as_str(), None);
        assert_eq!(bytes.as_bytes(), &[1_u8, 2]);
    }

    #[test]
    fn serde_round_trips_both_kinds() {
        let text = KvValue::text("x");
        let encoded = serde_json::to_string(&text).expect("serialize");
        assert_eq!(encoded, "\"x\"");
        assert_eq!(
            serde_json::from_str::<KvValue>(&encoded).expect("deserialize"),
            text
        );

        let bytes = KvValue::bytes(vec![7_u8, 8]);
        let encoded = serde_json::to_string(&bytes).expect("serialize");
        assert_eq!(
            serde_json::from_str::<KvValue>(&encoded).expect("deserialize"),
            bytes
        );
    }
}
