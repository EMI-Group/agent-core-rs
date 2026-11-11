//! In-memory key-value store.
//!
//! The behavioral reference for the [`KvStore`] contract: a `BTreeMap` behind a
//! `parking_lot::Mutex`, with lazy TTL expiry on every read path.
//! Ports `openjiuwen/core/foundation/store/kv/in_memory_kv_store.py`.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use parking_lot::Mutex;

use openjiuwen_common::error::Result;

use super::{KvPipeline, KvStore, KvValue, StorePipeline};

/// One stored entry: its value plus an optional absolute expiry instant.
#[derive(Debug, Clone)]
struct Entry {
    /// The stored payload.
    value: KvValue,
    /// When the entry stops being readable; `None` means it never expires.
    expiry: Option<Instant>,
}

impl Entry {
    /// Whether the entry is still readable at `now`.
    fn is_live(&self, now: Instant) -> bool {
        self.expiry.is_none_or(|expiry| now < expiry)
    }
}

/// A process-local key-value store.
///
/// All state lives behind an `Arc`, so clones share one map (handy when a store is
/// handed to several agents). No I/O happens, so every operation is cheap and
/// cancellation-safe.
#[derive(Debug, Clone, Default)]
pub struct InMemoryKvStore {
    /// Key → entry. `BTreeMap` gives prefix scans a sorted iteration order.
    state: Arc<Mutex<BTreeMap<String, Entry>>>,
}

impl InMemoryKvStore {
    /// Creates an empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of live entries (expired entries are purged first).
    pub fn len(&self) -> usize {
        let now = Instant::now();
        let mut state = self.state.lock();
        state.retain(|_, entry| entry.is_live(now));
        state.len()
    }

    /// Whether the store holds no live entry.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[async_trait]
impl KvStore for InMemoryKvStore {
    async fn set(&self, key: &str, value: KvValue) -> Result<()> {
        self.state.lock().insert(
            key.to_string(),
            Entry {
                value,
                expiry: None,
            },
        );
        Ok(())
    }

    async fn get(&self, key: &str) -> Result<Option<KvValue>> {
        let now = Instant::now();
        let mut state = self.state.lock();
        let Some(entry) = state.get(key) else {
            return Ok(None);
        };
        if entry.is_live(now) {
            return Ok(Some(entry.value.clone()));
        }
        // Expired: drop it lazily so a later read cannot see it again.
        state.remove(key);
        Ok(None)
    }

    async fn exists(&self, key: &str) -> Result<bool> {
        Ok(self.get(key).await?.is_some())
    }

    async fn delete(&self, key: &str) -> Result<()> {
        self.state.lock().remove(key);
        Ok(())
    }

    async fn exclusive_set(
        &self,
        key: &str,
        value: KvValue,
        expiry: Option<Duration>,
    ) -> Result<bool> {
        let now = Instant::now();
        let expiry_at = expiry.map(|ttl| now + ttl);
        let mut state = self.state.lock();
        if let Some(entry) = state.get(key) {
            if entry.is_live(now) {
                return Ok(false);
            }
        }
        // Absent, or present but expired: the write is allowed.
        state.insert(
            key.to_string(),
            Entry {
                value,
                expiry: expiry_at,
            },
        );
        Ok(true)
    }

    async fn get_by_prefix(&self, prefix: &str) -> Result<BTreeMap<String, KvValue>> {
        let now = Instant::now();
        let mut state = self.state.lock();
        state.retain(|_, entry| entry.is_live(now));
        let entries = state
            .range(prefix.to_string()..)
            .take_while(|(key, _)| key.starts_with(prefix))
            .map(|(key, entry)| (key.clone(), entry.value.clone()))
            .collect();
        Ok(entries)
    }

    async fn delete_by_prefix(&self, prefix: &str) -> Result<usize> {
        let now = Instant::now();
        let mut state = self.state.lock();
        // Expired entries are not counted: they are already logically absent.
        state.retain(|_, entry| entry.is_live(now));
        let mut deleted = 0;
        state.retain(|key, _| {
            let matches = key.starts_with(prefix);
            if matches {
                deleted += 1;
            }
            !matches
        });
        Ok(deleted)
    }

    async fn mget(&self, keys: &[String]) -> Result<Vec<Option<KvValue>>> {
        let now = Instant::now();
        let mut state = self.state.lock();
        state.retain(|_, entry| entry.is_live(now));
        Ok(keys
            .iter()
            .map(|key| state.get(key).map(|entry| entry.value.clone()))
            .collect())
    }

    async fn batch_delete(&self, keys: &[String]) -> Result<usize> {
        let now = Instant::now();
        let mut state = self.state.lock();
        state.retain(|_, entry| entry.is_live(now));
        let mut deleted = 0;
        for key in keys {
            if state.remove(key).is_some() {
                deleted += 1;
            }
        }
        Ok(deleted)
    }

    fn pipeline(&self) -> Box<dyn KvPipeline + '_> {
        Box::new(StorePipeline::new(self))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn text_and_bytes_round_trip() {
        let store = InMemoryKvStore::new();
        store.set("text", KvValue::text("hello")).await.unwrap();
        store
            .set("bytes", KvValue::bytes(vec![0_u8, 255]))
            .await
            .unwrap();

        assert_eq!(
            store.get("text").await.unwrap(),
            Some(KvValue::text("hello"))
        );
        assert_eq!(
            store.get("bytes").await.unwrap(),
            Some(KvValue::bytes(vec![0_u8, 255]))
        );
        assert_eq!(store.get("missing").await.unwrap(), None);
        assert!(store.exists("text").await.unwrap());
        assert!(!store.exists("missing").await.unwrap());
    }

    #[tokio::test]
    async fn delete_removes_entry_and_is_idempotent() {
        let store = InMemoryKvStore::new();
        store.set("k", KvValue::text("v")).await.unwrap();
        store.delete("k").await.unwrap();
        assert_eq!(store.get("k").await.unwrap(), None);
        // A second delete of the same (missing) key is a no-op.
        store.delete("k").await.unwrap();
        assert!(store.is_empty());
    }

    #[tokio::test]
    async fn exclusive_set_writes_only_when_absent() {
        let store = InMemoryKvStore::new();
        assert!(store
            .exclusive_set("k", KvValue::text("first"), None)
            .await
            .unwrap());
        assert!(!store
            .exclusive_set("k", KvValue::text("second"), None)
            .await
            .unwrap());
        assert_eq!(store.get("k").await.unwrap(), Some(KvValue::text("first")));
    }

    #[tokio::test]
    async fn ttl_expires_on_read_and_allows_reacquire() {
        let store = InMemoryKvStore::new();
        assert!(store
            .exclusive_set("lock", KvValue::text("a"), Some(Duration::from_millis(20)))
            .await
            .unwrap());
        assert!(store.exists("lock").await.unwrap());

        tokio::time::sleep(Duration::from_millis(80)).await;
        assert_eq!(store.get("lock").await.unwrap(), None);
        assert!(!store.exists("lock").await.unwrap());
        assert!(store
            .exclusive_set("lock", KvValue::text("b"), None)
            .await
            .unwrap());
        assert_eq!(store.get("lock").await.unwrap(), Some(KvValue::text("b")));
    }

    #[tokio::test]
    async fn set_clears_previous_ttl() {
        let store = InMemoryKvStore::new();
        store
            .exclusive_set("k", KvValue::text("a"), Some(Duration::from_millis(20)))
            .await
            .unwrap();
        store.set("k", KvValue::text("b")).await.unwrap();
        tokio::time::sleep(Duration::from_millis(80)).await;
        assert_eq!(store.get("k").await.unwrap(), Some(KvValue::text("b")));
    }

    #[tokio::test]
    async fn get_by_prefix_is_sorted_and_supports_empty_prefix() {
        let store = InMemoryKvStore::new();
        store.set("b/2", KvValue::text("2")).await.unwrap();
        store.set("a/1", KvValue::text("1")).await.unwrap();
        store.set("other", KvValue::text("x")).await.unwrap();

        let all = store.get_by_prefix("").await.unwrap();
        assert_eq!(
            all.keys().cloned().collect::<Vec<_>>(),
            vec!["a/1".to_string(), "b/2".to_string(), "other".to_string()]
        );

        let prefixed = store.get_by_prefix("a/").await.unwrap();
        assert_eq!(prefixed.len(), 1);
        assert_eq!(prefixed.get("a/1"), Some(&KvValue::text("1")));

        assert!(store.get_by_prefix("zzz").await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn get_by_prefix_skips_expired_entries() {
        let store = InMemoryKvStore::new();
        store
            .exclusive_set("p/live", KvValue::text("v"), None)
            .await
            .unwrap();
        store
            .exclusive_set(
                "p/dead",
                KvValue::text("v"),
                Some(Duration::from_millis(20)),
            )
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(80)).await;

        let entries = store.get_by_prefix("p/").await.unwrap();
        assert_eq!(entries.len(), 1);
        assert!(entries.contains_key("p/live"));
        assert_eq!(store.len(), 1);
    }

    #[tokio::test]
    async fn delete_by_prefix_returns_count() {
        let store = InMemoryKvStore::new();
        for key in ["p/1", "p/2", "p/3", "q/1"] {
            store.set(key, KvValue::text(key)).await.unwrap();
        }
        assert_eq!(store.delete_by_prefix("p/").await.unwrap(), 3);
        assert_eq!(store.delete_by_prefix("p/").await.unwrap(), 0);
        assert_eq!(store.get_by_prefix("q/").await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn mget_preserves_order_and_misses() {
        let store = InMemoryKvStore::new();
        store.set("a", KvValue::text("1")).await.unwrap();
        store.set("c", KvValue::text("3")).await.unwrap();

        let keys = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let values = store.mget(&keys).await.unwrap();
        assert_eq!(
            values,
            vec![Some(KvValue::text("1")), None, Some(KvValue::text("3")),]
        );
        assert_eq!(store.mget(&[]).await.unwrap().len(), 0);
    }

    #[tokio::test]
    async fn batch_delete_counts_present_keys_only() {
        let store = InMemoryKvStore::new();
        store.set("a", KvValue::text("1")).await.unwrap();
        store.set("b", KvValue::text("2")).await.unwrap();

        let keys = vec![
            "a".to_string(),
            "b".to_string(),
            "a".to_string(),
            "missing".to_string(),
        ];
        // Duplicate keys are only removed (and counted) once.
        assert_eq!(store.batch_delete(&keys).await.unwrap(), 2);
        assert_eq!(store.batch_delete(&[]).await.unwrap(), 0);
        assert!(store.is_empty());
    }

    #[tokio::test]
    async fn pipeline_applies_operations_in_order() {
        let store = InMemoryKvStore::new();
        store.set("existing", KvValue::text("old")).await.unwrap();

        let mut pipeline = StorePipeline::new(&store);
        pipeline.set("queued", KvValue::text("new"));
        pipeline.get("queued");
        pipeline.exists("queued");
        pipeline.exists("missing");
        pipeline.get("existing");
        assert_eq!(pipeline.len(), 5);
        assert!(!pipeline.is_empty());

        let results = pipeline.execute().await.unwrap();
        assert_eq!(
            results,
            vec![
                None,
                Some(KvValue::text("new")),
                Some(KvValue::text("true")),
                Some(KvValue::text("false")),
                Some(KvValue::text("old")),
            ]
        );
        // The queue is cleared after execution.
        assert!(pipeline.is_empty());
        assert_eq!(
            store.get("queued").await.unwrap(),
            Some(KvValue::text("new"))
        );

        // The trait method hands back the same shared pipeline over `&dyn KvStore`.
        let mut boxed = store.pipeline();
        boxed.get("queued");
        assert_eq!(
            boxed.execute().await.unwrap(),
            vec![Some(KvValue::text("new"))]
        );
    }
}
