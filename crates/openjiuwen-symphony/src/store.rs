//! Capability-asset persistence: the [`AssetStore`](crate::store::AssetStore) seam and
//! its KV backend.
//!
//! Python's symphony engine receives capability inventories from the host and keeps no
//! durable catalog of its own. The Rust port needs CRUD and search across restarts, so
//! an [`AssetStore`](crate::store::AssetStore) persists descriptors, and
//! [`KvAssetStore`](crate::store::KvAssetStore) implements it on top of
//! `openjiuwen-foundation`'s [`KvStore`](openjiuwen_foundation::store::KvStore)
//! (in-memory, one JSON file per entry, or `SQLite`).
//!
//! Storage layout: the key is `<prefix><qualified capability id>` and the value is the
//! descriptor's JSON, so a backend can be inspected (and repaired) with plain JSON
//! tooling. Reads are tolerant: an entry that is not text or not a valid descriptor is
//! logged with `tracing::warn!` and skipped rather than failing the whole scan.

use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_foundation::store::kv::{InMemoryKvStore, KvStore, KvValue, LocalFileKvStore};
use tracing::warn;

use crate::error::{Error, Result};
use crate::model::{CapabilityDescriptor, CapabilityKey};

/// The default key prefix used by [`KvAssetStore::new`].
pub const DEFAULT_PREFIX: &str = "capability/";

/// Persistence for capability descriptors, keyed by [`CapabilityKey`].
///
/// Reads return `Option`/empty collections instead of [`openjiuwen_common::error::Error::NotFound`]
/// because a missing asset is a normal catalog state, not a failure.
#[async_trait]
pub trait AssetStore: Send + Sync {
    /// Returns the stored descriptor for `key`, or `None` when it is absent.
    ///
    /// # Errors
    ///
    /// Returns the backend's error when the underlying store fails; malformed entries
    /// are reported as `None`.
    async fn get(&self, key: &CapabilityKey) -> Result<Option<CapabilityDescriptor>>;

    /// Inserts or replaces `descriptor`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::External`] when the descriptor cannot be serialized, or the
    /// backend's error when the write fails.
    async fn put(&self, descriptor: &CapabilityDescriptor) -> Result<()>;

    /// Removes `key`, returning whether it existed.
    ///
    /// # Errors
    ///
    /// Returns the backend's error when the read or delete fails.
    async fn remove(&self, key: &CapabilityKey) -> Result<bool>;

    /// Lists every stored descriptor, sorted by qualified id.
    ///
    /// Malformed entries are skipped (with a warning) instead of failing the scan.
    ///
    /// # Errors
    ///
    /// Returns the backend's error when the prefix scan fails.
    async fn list(&self) -> Result<Vec<CapabilityDescriptor>>;

    /// Removes every descriptor in this store's namespace.
    ///
    /// # Errors
    ///
    /// Returns the backend's error when the prefix delete fails.
    async fn clear(&self) -> Result<()>;
}

/// An [`AssetStore`] backed by a foundation [`KvStore`] and a key prefix.
///
/// The KV store is shared (`Arc<dyn KvStore>`) so a caller can reuse the same backend
/// for other namespaces by choosing a different prefix.
pub struct KvAssetStore {
    /// The backing key-value store.
    kv: Arc<dyn KvStore>,
    /// Prefix applied to every key written by this store.
    prefix: String,
}

impl KvAssetStore {
    /// Wraps `kv`, writing under the [`DEFAULT_PREFIX`] namespace.
    #[must_use]
    pub fn new(kv: Arc<dyn KvStore>) -> Self {
        Self::with_prefix(kv, DEFAULT_PREFIX)
    }

    /// Wraps `kv`, writing under `prefix`.
    #[must_use]
    pub fn with_prefix(kv: Arc<dyn KvStore>, prefix: impl Into<String>) -> Self {
        Self {
            kv,
            prefix: prefix.into(),
        }
    }

    /// Builds a store in the process's memory; useful for tests and ephemeral runs.
    #[must_use]
    pub fn in_memory() -> Self {
        Self::new(Arc::new(InMemoryKvStore::new()))
    }

    /// Builds a store holding one JSON file per descriptor below `root`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] when `root` cannot be used as a directory, or
    /// [`Error::External`] for the underlying I/O failure.
    pub async fn json_file(root: impl AsRef<Path>) -> Result<Self> {
        let kv = LocalFileKvStore::new(root).await?;
        Ok(Self::new(Arc::new(kv)))
    }

    /// Builds a store backed by the `SQLite` database at `url`.
    ///
    /// # Errors
    ///
    /// Returns the backend's error when the database cannot be opened or migrated.
    #[cfg(feature = "sqlite")]
    pub async fn sqlite(url: &str) -> Result<Self> {
        let kv = openjiuwen_foundation::store::kv::SqliteKvStore::connect(url).await?;
        Ok(Self::new(Arc::new(kv)))
    }

    /// The configured key prefix.
    #[must_use]
    pub fn prefix(&self) -> &str {
        &self.prefix
    }
}

#[async_trait]
impl AssetStore for KvAssetStore {
    async fn get(&self, key: &CapabilityKey) -> Result<Option<CapabilityDescriptor>> {
        let stored_key = key_for(&self.prefix, key);
        let Some(value) = self.kv.get(&stored_key).await? else {
            return Ok(None);
        };
        Ok(parse_entry(&stored_key, &value))
    }

    async fn put(&self, descriptor: &CapabilityDescriptor) -> Result<()> {
        let stored_key = key_for(&self.prefix, &descriptor.key());
        let json = serde_json::to_string(descriptor).map_err(Error::external)?;
        self.kv.set(&stored_key, KvValue::text(json)).await
    }

    async fn remove(&self, key: &CapabilityKey) -> Result<bool> {
        let stored_key = key_for(&self.prefix, key);
        let existed = self.kv.get(&stored_key).await?.is_some();
        if existed {
            self.kv.delete(&stored_key).await?;
        }
        Ok(existed)
    }

    async fn list(&self) -> Result<Vec<CapabilityDescriptor>> {
        let entries = self.kv.get_by_prefix(&self.prefix).await?;
        let mut descriptors: Vec<CapabilityDescriptor> = entries
            .iter()
            .filter_map(|(stored_key, value)| parse_entry(stored_key, value))
            .collect();
        descriptors.sort_by_key(CapabilityDescriptor::qualified_id);
        Ok(descriptors)
    }

    async fn clear(&self) -> Result<()> {
        self.kv.delete_by_prefix(&self.prefix).await?;
        Ok(())
    }
}

/// The KV key of `key` under `prefix`.
fn key_for(prefix: &str, key: &CapabilityKey) -> String {
    format!("{prefix}{}", key.qualified())
}

/// Parses one KV entry, warning and returning `None` when it is malformed.
fn parse_entry(stored_key: &str, value: &KvValue) -> Option<CapabilityDescriptor> {
    let Some(text) = value.as_str() else {
        warn!(
            key = stored_key,
            "capability asset entry is not text; skipping it"
        );
        return None;
    };
    match serde_json::from_str::<CapabilityDescriptor>(text) {
        Ok(descriptor) => Some(descriptor),
        Err(error) => {
            warn!(
                key = stored_key,
                %error,
                "capability asset entry is not a valid descriptor; skipping it"
            );
            None
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn sample(id: &str) -> CapabilityDescriptor {
        CapabilityDescriptor::new("tool", id, format!("Tool {id}"))
            .with_description("does something")
            .with_available(true)
    }

    #[tokio::test]
    async fn memory_store_round_trips_and_lists_in_order() {
        let store = KvAssetStore::in_memory();
        let second = sample("b");
        let first = sample("a");
        store.put(&second).await.expect("put b");
        store.put(&first).await.expect("put a");

        assert_eq!(
            store.get(&first.key()).await.expect("get"),
            Some(first.clone())
        );
        assert_eq!(store.get(&second.key()).await.expect("get"), Some(second));
        let listed = store.list().await.expect("list");
        assert_eq!(
            listed
                .iter()
                .map(CapabilityDescriptor::qualified_id)
                .collect::<Vec<_>>(),
            vec!["tool:a".to_string(), "tool:b".to_string()]
        );

        assert!(store.remove(&first.key()).await.expect("remove"));
        assert!(!store.remove(&first.key()).await.expect("remove again"));
        assert!(store.get(&first.key()).await.expect("get").is_none());

        store.clear().await.expect("clear");
        assert_eq!(store.list().await.expect("list").len(), 0);
    }

    #[tokio::test]
    async fn keys_use_the_configured_prefix() {
        let kv = Arc::new(InMemoryKvStore::new());
        let store = KvAssetStore::with_prefix(kv.clone(), "assets/");
        let descriptor = sample("sum");
        store.put(&descriptor).await.expect("put");
        assert_eq!(store.prefix(), "assets/");
        let stored = kv.get("assets/tool:sum").await.expect("raw get");
        assert!(stored.is_some(), "entry must live under the prefix");
        assert!(kv
            .get("capability/tool:sum")
            .await
            .expect("raw get")
            .is_none());
    }

    #[tokio::test]
    async fn malformed_entries_are_skipped_not_fatal() {
        let kv = Arc::new(InMemoryKvStore::new());
        let store = KvAssetStore::with_prefix(kv.clone(), "capability/");
        store.put(&sample("good")).await.expect("put");
        kv.set("capability/tool:broken", KvValue::text("{not json"))
            .await
            .expect("write malformed");
        kv.set("capability/tool:bytes", KvValue::bytes(vec![1, 2, 3]))
            .await
            .expect("write bytes");

        let listed = store.list().await.expect("list");
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].qualified_id(), "tool:good");

        let broken = CapabilityKey::new("tool", "broken");
        assert!(store.get(&broken).await.expect("get malformed").is_none());
        let binary = CapabilityKey::new("tool", "bytes");
        assert!(store.get(&binary).await.expect("get bytes").is_none());
    }

    #[tokio::test]
    async fn file_store_survives_reopen() {
        let dir = tempfile::tempdir().expect("tempdir");
        let descriptor = sample("persisted");
        {
            let store = KvAssetStore::json_file(dir.path()).await.expect("open");
            store.put(&descriptor).await.expect("put");
        }
        let reopened = KvAssetStore::json_file(dir.path()).await.expect("reopen");
        let listed = reopened.list().await.expect("list");
        assert_eq!(listed, vec![descriptor]);
    }

    #[tokio::test]
    async fn stored_values_are_json_text() {
        let kv = Arc::new(InMemoryKvStore::new());
        let store = KvAssetStore::with_prefix(kv.clone(), DEFAULT_PREFIX);
        store.put(&sample("sum")).await.expect("put");
        let raw = kv
            .get("capability/tool:sum")
            .await
            .expect("raw get")
            .expect("present");
        let text = raw.as_str().expect("text value");
        let parsed: Value = serde_json::from_str(text).expect("valid json");
        assert_eq!(parsed["capability_id"], "sum");
        assert_eq!(parsed["available"], Value::Bool(true));
    }
}
