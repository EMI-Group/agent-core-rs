//! In-memory object store.
//!
//! The reference [`ObjectStore`] implementation: buckets live in a
//! [`BTreeMap`], which gives the sorted bucket and object listings the trait
//! promises for free. Payloads are kept as [`Bytes`] handle clones, so `get` is
//! a pointer-free copy of the handle, not a payload copy.

use std::collections::BTreeMap;

use async_trait::async_trait;
use bytes::Bytes;
use chrono::{DateTime, Utc};
use parking_lot::Mutex;

use openjiuwen_common::error::Result;

use super::{bucket_not_found, ObjectMeta, ObjectStore};

/// Buckets and objects held in process memory.
///
/// ```
/// use bytes::Bytes;
/// use openjiuwen_foundation::store::object::{InMemoryObjectStore, ObjectStore};
///
/// # #[tokio::main(flavor = "current_thread")]
/// # async fn main() -> openjiuwen_common::error::Result<()> {
/// let store = InMemoryObjectStore::new();
/// store.create_bucket("uploads").await?;
/// store.put("uploads", "a.txt", Bytes::from_static(b"hi")).await?;
/// assert_eq!(store.list_buckets().await?, vec!["uploads".to_string()]);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Default)]
pub struct InMemoryObjectStore {
    buckets: Mutex<BucketMap>,
}

/// Buckets of objects, each holding its payload and write instant.
type BucketMap = BTreeMap<String, BTreeMap<String, (Bytes, DateTime<Utc>)>>;

impl InMemoryObjectStore {
    /// Creates an empty store.
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl ObjectStore for InMemoryObjectStore {
    async fn create_bucket(&self, bucket: &str) -> Result<()> {
        self.buckets.lock().entry(bucket.to_string()).or_default();
        Ok(())
    }

    async fn delete_bucket(&self, bucket: &str) -> Result<()> {
        self.buckets.lock().remove(bucket);
        Ok(())
    }

    async fn list_buckets(&self) -> Result<Vec<String>> {
        Ok(self.buckets.lock().keys().cloned().collect())
    }

    async fn put(&self, bucket: &str, key: &str, data: Bytes) -> Result<ObjectMeta> {
        let updated_at = Utc::now();
        let size = data.len() as u64;
        let mut buckets = self.buckets.lock();
        let objects = buckets
            .get_mut(bucket)
            .ok_or_else(|| bucket_not_found(bucket))?;
        objects.insert(key.to_string(), (data, updated_at));
        Ok(ObjectMeta {
            key: key.to_string(),
            size,
            updated_at,
        })
    }

    async fn get(&self, bucket: &str, key: &str) -> Result<Option<Bytes>> {
        let buckets = self.buckets.lock();
        let objects = buckets
            .get(bucket)
            .ok_or_else(|| bucket_not_found(bucket))?;
        Ok(objects.get(key).map(|(data, _)| data.clone()))
    }

    async fn delete(&self, bucket: &str, key: &str) -> Result<()> {
        let mut buckets = self.buckets.lock();
        let objects = buckets
            .get_mut(bucket)
            .ok_or_else(|| bucket_not_found(bucket))?;
        objects.remove(key);
        Ok(())
    }

    async fn exists(&self, bucket: &str, key: &str) -> Result<bool> {
        let buckets = self.buckets.lock();
        let objects = buckets
            .get(bucket)
            .ok_or_else(|| bucket_not_found(bucket))?;
        Ok(objects.contains_key(key))
    }

    async fn list(
        &self,
        bucket: &str,
        prefix: &str,
        limit: Option<usize>,
    ) -> Result<Vec<ObjectMeta>> {
        let buckets = self.buckets.lock();
        let objects = buckets
            .get(bucket)
            .ok_or_else(|| bucket_not_found(bucket))?;
        let entries = objects
            .iter()
            .filter(|(key, _)| key.starts_with(prefix))
            .map(|(key, (data, updated_at))| ObjectMeta {
                key: key.clone(),
                size: data.len() as u64,
                updated_at: *updated_at,
            });
        Ok(entries.take(limit.unwrap_or(usize::MAX)).collect())
    }
}

#[cfg(test)]
mod tests {
    use openjiuwen_common::error::ErrorKind;

    use super::*;

    #[tokio::test]
    async fn put_get_exists_delete_round_trip() {
        let store = InMemoryObjectStore::new();
        store.create_bucket("b").await.expect("create bucket");
        let before = Utc::now();
        let meta = store
            .put("b", "a.txt", Bytes::from_static(b"hello"))
            .await
            .expect("put");
        assert_eq!(meta.key, "a.txt");
        assert_eq!(meta.size, 5);
        assert!(meta.updated_at >= before);

        assert!(store.exists("b", "a.txt").await.expect("exists"));
        assert_eq!(
            store.get("b", "a.txt").await.expect("get"),
            Some(Bytes::from_static(b"hello"))
        );

        store.delete("b", "a.txt").await.expect("delete");
        assert!(!store.exists("b", "a.txt").await.expect("exists"));
        assert_eq!(store.get("b", "a.txt").await.expect("get"), None);
        // Deleting twice stays a no-op.
        store.delete("b", "a.txt").await.expect("delete again");
    }

    #[tokio::test]
    async fn put_replaces_existing_payload() {
        let store = InMemoryObjectStore::new();
        store.create_bucket("b").await.expect("create bucket");
        store
            .put("b", "k", Bytes::from_static(b"one"))
            .await
            .expect("first put");
        let meta = store
            .put("b", "k", Bytes::from_static(b"second"))
            .await
            .expect("second put");
        assert_eq!(meta.size, 6);
        assert_eq!(
            store.get("b", "k").await.expect("get"),
            Some(Bytes::from_static(b"second"))
        );
    }

    #[tokio::test]
    async fn bytes_round_trip_is_lossless_for_non_utf8() {
        let store = InMemoryObjectStore::new();
        store.create_bucket("b").await.expect("create bucket");
        let payload = Bytes::from_static(&[0xff, 0x00, 0x80, 0xfe]);
        store.put("b", "raw", payload.clone()).await.expect("put");
        assert_eq!(store.get("b", "raw").await.expect("get"), Some(payload));
    }

    #[tokio::test]
    async fn list_filters_by_prefix_sorted_and_limited() {
        let store = InMemoryObjectStore::new();
        store.create_bucket("b").await.expect("create bucket");
        for key in ["docs/b.txt", "docs/a.txt", "docs/c.txt", "other.txt"] {
            store
                .put("b", key, Bytes::from_static(b"x"))
                .await
                .expect("put");
        }

        let keys: Vec<String> = store
            .list("b", "", None)
            .await
            .expect("list all")
            .into_iter()
            .map(|meta| meta.key)
            .collect();
        assert_eq!(
            keys,
            vec!["docs/a.txt", "docs/b.txt", "docs/c.txt", "other.txt"]
        );

        let docs: Vec<String> = store
            .list("b", "docs/", None)
            .await
            .expect("list prefix")
            .into_iter()
            .map(|meta| meta.key)
            .collect();
        assert_eq!(docs, vec!["docs/a.txt", "docs/b.txt", "docs/c.txt"]);

        let limited = store.list("b", "docs/", Some(2)).await.expect("list limit");
        let limited_keys: Vec<&str> = limited.iter().map(|meta| meta.key.as_str()).collect();
        assert_eq!(limited_keys, vec!["docs/a.txt", "docs/b.txt"]);
    }

    #[tokio::test]
    async fn list_reports_sizes_and_timestamps() {
        let store = InMemoryObjectStore::new();
        store.create_bucket("b").await.expect("create bucket");
        let meta = store
            .put("b", "k", Bytes::from_static(b"abc"))
            .await
            .expect("put");
        let listed = store.list("b", "k", None).await.expect("list");
        assert_eq!(listed, vec![meta]);
    }

    #[tokio::test]
    async fn missing_bucket_is_not_found() {
        let store = InMemoryObjectStore::new();
        assert_eq!(
            store
                .put("nope", "k", Bytes::from_static(b"x"))
                .await
                .unwrap_err()
                .kind(),
            ErrorKind::NotFound
        );
        assert_eq!(
            store.get("nope", "k").await.unwrap_err().kind(),
            ErrorKind::NotFound
        );
        assert_eq!(
            store.delete("nope", "k").await.unwrap_err().kind(),
            ErrorKind::NotFound
        );
        assert_eq!(
            store.exists("nope", "k").await.unwrap_err().kind(),
            ErrorKind::NotFound
        );
        assert_eq!(
            store.list("nope", "", None).await.unwrap_err().kind(),
            ErrorKind::NotFound
        );
    }

    #[tokio::test]
    async fn delete_bucket_removes_contents_and_ignores_missing() {
        let store = InMemoryObjectStore::new();
        store.create_bucket("b").await.expect("create bucket");
        store
            .put("b", "a/b.txt", Bytes::from_static(b"x"))
            .await
            .expect("put");
        store.delete_bucket("b").await.expect("delete bucket");
        assert_eq!(
            store.get("b", "a/b.txt").await.unwrap_err().kind(),
            ErrorKind::NotFound
        );
        assert_eq!(
            store.list_buckets().await.expect("buckets"),
            Vec::<String>::new()
        );
        // Deleting a missing bucket is a no-op.
        store
            .delete_bucket("b")
            .await
            .expect("delete missing bucket");
    }

    #[tokio::test]
    async fn list_buckets_is_sorted_and_create_is_idempotent() {
        let store = InMemoryObjectStore::new();
        for bucket in ["zeta", "alpha", "beta"] {
            store.create_bucket(bucket).await.expect("create bucket");
        }
        store.create_bucket("alpha").await.expect("create again");
        assert_eq!(
            store.list_buckets().await.expect("buckets"),
            vec!["alpha".to_string(), "beta".to_string(), "zeta".to_string()]
        );
    }
}
