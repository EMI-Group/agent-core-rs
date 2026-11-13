//! Object storage.
//!
//! Buckets of opaque byte objects with prefix listing and metadata. The local
//! implementation maps buckets to directories under a root path.
//!
//! The contract mirrors the abstract half of the Python
//! `openjiuwen/core/foundation/store/object/base_storage_client.py`
//! (`BaseObjectStorageClient`): bucket lifecycle plus put/get/delete/exists/list
//! of opaque objects. The Python S3/aioboto3 client is replaced by
//! [`S3ObjectStore`](super::external::s3::S3ObjectStore), an in-crate
//! S3-compatible adapter with its own `SigV4` signer, so only the `aioboto3`
//! dependency is gone; [`InMemoryObjectStore`] and [`LocalObjectStore`] stay the
//! reference implementations.
//!
//! All operations on a bucket that does not exist fail with
//! [`openjiuwen_common::error::ErrorKind::NotFound`], except `delete_bucket`
//! (a no-op for a missing bucket, matching the tolerated-missing convention of
//! the other stores). Reading a missing object is `Ok(None)` and deleting one is
//! a no-op.

use async_trait::async_trait;
use bytes::Bytes;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use openjiuwen_common::error::{Error, Result};

pub mod local;
pub mod memory;

pub use local::LocalObjectStore;
pub use memory::InMemoryObjectStore;

/// Metadata for one stored object.
///
/// `size` is the object payload length in bytes and `updated_at` is the instant
/// the current payload was written.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ObjectMeta {
    /// Object key (path-like, `/`-separated inside a bucket).
    pub key: String,
    /// Payload length in bytes.
    pub size: u64,
    /// Instant the current payload was written.
    pub updated_at: DateTime<Utc>,
}

/// Buckets of opaque byte objects.
///
/// Implementations are expected to be cheap to share: every method takes
/// `&self`, so a store can be used through an `Arc<dyn ObjectStore>`.
///
/// # Errors
///
/// `put`, `get`, `delete`, `exists` and `list` return
/// [`openjiuwen_common::error::ErrorKind::NotFound`] when the bucket does not
/// exist. Implementations may also return validation errors for malformed
/// bucket names or object keys.
#[async_trait]
pub trait ObjectStore: Send + Sync {
    /// Creates a bucket. Creating an existing bucket is a successful no-op.
    async fn create_bucket(&self, bucket: &str) -> Result<()>;

    /// Deletes a bucket and its contents. A missing bucket is a no-op.
    async fn delete_bucket(&self, bucket: &str) -> Result<()>;

    /// Lists every bucket name in ascending order.
    async fn list_buckets(&self) -> Result<Vec<String>>;

    /// Writes `data` under `key` in `bucket`, replacing any previous payload.
    ///
    /// Returns the resulting [`ObjectMeta`] with `size` set to the payload
    /// length and `updated_at` set to the current instant.
    async fn put(&self, bucket: &str, key: &str, data: Bytes) -> Result<ObjectMeta>;

    /// Reads the payload stored under `key`, or `None` when the object is absent.
    async fn get(&self, bucket: &str, key: &str) -> Result<Option<Bytes>>;

    /// Deletes the object stored under `key`. A missing object is a no-op.
    async fn delete(&self, bucket: &str, key: &str) -> Result<()>;

    /// Reports whether an object is stored under `key`.
    async fn exists(&self, bucket: &str, key: &str) -> Result<bool>;

    /// Lists object metadata in `bucket`, ascending by key.
    ///
    /// Only keys starting with `prefix` are returned (an empty prefix matches
    /// everything); `limit` caps the number of returned entries after sorting,
    /// so `None` means "no limit".
    async fn list(
        &self,
        bucket: &str,
        prefix: &str,
        limit: Option<usize>,
    ) -> Result<Vec<ObjectMeta>>;
}

/// The shared not-found error for a missing bucket.
pub(crate) fn bucket_not_found(bucket: &str) -> Error {
    Error::not_found(format!("object bucket {bucket:?} not found"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn object_meta_serializes_snake_case() {
        let meta = ObjectMeta {
            key: "a/b.txt".to_string(),
            size: 3,
            updated_at: DateTime::from_timestamp(1_700_000_000, 0).expect("valid timestamp"),
        };
        let json = serde_json::to_value(&meta).expect("ObjectMeta serializes");
        assert_eq!(json["key"], "a/b.txt");
        assert_eq!(json["size"], 3);
        assert!(json.get("updated_at").is_some());

        let back: ObjectMeta = serde_json::from_value(json).expect("ObjectMeta round-trips");
        assert_eq!(back, meta);

        let tolerant: ObjectMeta = serde_json::from_str(
            r#"{"key":"k","size":0,"updated_at":"2023-11-14T22:13:20Z","extra":true}"#,
        )
        .expect("unknown fields are tolerated");
        assert_eq!(tolerant.key, "k");
    }

    #[tokio::test]
    async fn trait_is_object_safe() {
        let store = InMemoryObjectStore::new();
        let store: &dyn ObjectStore = &store;
        store.create_bucket("b").await.expect("create bucket");
        let meta = store
            .put("b", "k", Bytes::from_static(b"hi"))
            .await
            .expect("put object");
        assert_eq!(meta.size, 2);
        assert!(store.exists("b", "k").await.expect("exists"));
    }
}
