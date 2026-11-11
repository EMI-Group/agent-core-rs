//! Local-filesystem object store.
//!
//! Buckets map to directories below a caller-provided root and object keys map
//! to files below the bucket directory, so `bucket/key` becomes
//! `<root>/<bucket>/<key>`. Nested keys (`a/b.txt`) create their parent
//! directories on `put` and are listed recursively.
//!
//! Only `tokio::fs` is used: no blocking filesystem call runs on the async path.
//! Every bucket name and key is validated before it is turned into a path, so a
//! store can never read or write outside its root; keys that resolve to `.` or
//! `..` segments, absolute keys, and Windows-style separators are rejected with
//! a validation error.

use std::path::{Component, Path, PathBuf};

use async_trait::async_trait;
use bytes::Bytes;
use chrono::{DateTime, Utc};
use tokio::fs;

use openjiuwen_common::error::{Error, Result};

use super::{bucket_not_found, ObjectMeta, ObjectStore};

/// An [`ObjectStore`] backed by directories and files under a root path.
///
/// The store keeps no in-memory index: buckets are discovered by scanning the
/// root, which makes reopening a directory tree and concurrent use by several
/// stores on the same root safe.
#[derive(Debug, Clone)]
pub struct LocalObjectStore {
    root: PathBuf,
}

impl LocalObjectStore {
    /// Opens the store rooted at `root`, creating the directory (and its
    /// parents) when it does not exist yet.
    pub async fn new(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        fs::create_dir_all(&root)
            .await
            .map_err(|error| io_error("creating object store root", &root, error))?;
        Ok(Self { root })
    }

    /// The root directory this store reads and writes below.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Resolves and verifies an existing bucket directory.
    async fn bucket_dir(&self, bucket: &str) -> Result<PathBuf> {
        validate_bucket(bucket)?;
        let path = self.root.join(bucket);
        match fs::metadata(&path).await {
            Ok(metadata) if metadata.is_dir() => Ok(path),
            Ok(_) => Err(bucket_not_found(bucket)),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                Err(bucket_not_found(bucket))
            }
            Err(error) => Err(io_error("reading bucket directory", &path, error)),
        }
    }
}

#[async_trait]
impl ObjectStore for LocalObjectStore {
    async fn create_bucket(&self, bucket: &str) -> Result<()> {
        validate_bucket(bucket)?;
        let path = self.root.join(bucket);
        fs::create_dir_all(&path)
            .await
            .map_err(|error| io_error("creating bucket directory", &path, error))
    }

    async fn delete_bucket(&self, bucket: &str) -> Result<()> {
        validate_bucket(bucket)?;
        let path = self.root.join(bucket);
        match fs::remove_dir_all(&path).await {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(io_error("deleting bucket directory", &path, error)),
        }
    }

    async fn list_buckets(&self) -> Result<Vec<String>> {
        let mut entries = fs::read_dir(&self.root)
            .await
            .map_err(|error| io_error("reading object store root", &self.root, error))?;
        let mut buckets = Vec::new();
        while let Some(entry) = entries
            .next_entry()
            .await
            .map_err(|error| io_error("reading object store root", &self.root, error))?
        {
            let file_type = entry
                .file_type()
                .await
                .map_err(|error| io_error("reading object store entry", &entry.path(), error))?;
            if file_type.is_dir() {
                if let Some(name) = entry.file_name().to_str() {
                    buckets.push(name.to_string());
                }
            }
        }
        buckets.sort();
        Ok(buckets)
    }

    async fn put(&self, bucket: &str, key: &str, data: Bytes) -> Result<ObjectMeta> {
        let directory = self.bucket_dir(bucket).await?;
        let size = data.len() as u64;
        let path = object_path(&directory, key)?;
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .await
                .map_err(|error| io_error("creating object directory", parent, error))?;
        }
        fs::write(&path, &data)
            .await
            .map_err(|error| io_error("writing object", &path, error))?;
        Ok(ObjectMeta {
            key: key.to_string(),
            size,
            updated_at: Utc::now(),
        })
    }

    async fn get(&self, bucket: &str, key: &str) -> Result<Option<Bytes>> {
        let directory = self.bucket_dir(bucket).await?;
        let path = object_path(&directory, key)?;
        match fs::read(&path).await {
            Ok(data) => Ok(Some(Bytes::from(data))),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(io_error("reading object", &path, error)),
        }
    }

    async fn delete(&self, bucket: &str, key: &str) -> Result<()> {
        let directory = self.bucket_dir(bucket).await?;
        let path = object_path(&directory, key)?;
        match fs::remove_file(&path).await {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(io_error("deleting object", &path, error)),
        }
    }

    async fn exists(&self, bucket: &str, key: &str) -> Result<bool> {
        let directory = self.bucket_dir(bucket).await?;
        let path = object_path(&directory, key)?;
        match fs::metadata(&path).await {
            Ok(metadata) => Ok(metadata.is_file()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
            Err(error) => Err(io_error("reading object metadata", &path, error)),
        }
    }

    async fn list(
        &self,
        bucket: &str,
        prefix: &str,
        limit: Option<usize>,
    ) -> Result<Vec<ObjectMeta>> {
        let directory = self.bucket_dir(bucket).await?;
        let mut stack = vec![directory.clone()];
        let mut objects = Vec::new();
        while let Some(current) = stack.pop() {
            let mut entries = fs::read_dir(&current)
                .await
                .map_err(|error| io_error("reading object directory", &current, error))?;
            while let Some(entry) = entries
                .next_entry()
                .await
                .map_err(|error| io_error("reading object directory", &current, error))?
            {
                let path = entry.path();
                let file_type = entry
                    .file_type()
                    .await
                    .map_err(|error| io_error("reading object entry", &path, error))?;
                // Skip symlinks: following them could leave the bucket.
                if file_type.is_dir() {
                    stack.push(path);
                    continue;
                }
                if !file_type.is_file() {
                    continue;
                }
                let Some(key) = relative_key(&directory, &path) else {
                    continue;
                };
                if !key.starts_with(prefix) {
                    continue;
                }
                let metadata = entry
                    .metadata()
                    .await
                    .map_err(|error| io_error("reading object metadata", &path, error))?;
                let updated_at = metadata
                    .modified()
                    .map_or_else(|_| Utc::now(), DateTime::<Utc>::from);
                objects.push(ObjectMeta {
                    key,
                    size: metadata.len(),
                    updated_at,
                });
            }
        }
        objects.sort_by(|left, right| left.key.cmp(&right.key));
        if let Some(limit) = limit {
            objects.truncate(limit);
        }
        Ok(objects)
    }
}

/// Validates a bucket name: a single, non-empty path segment without separators.
fn validate_bucket(bucket: &str) -> Result<()> {
    if bucket.is_empty() {
        return Err(Error::validation("object bucket name must not be empty"));
    }
    if bucket == "." || bucket == ".." {
        return Err(Error::validation(format!(
            "invalid object bucket name {bucket:?}: must not be a relative path segment"
        )));
    }
    if bucket.contains('/') || bucket.contains('\\') || bucket.contains('\0') {
        return Err(Error::validation(format!(
            "invalid object bucket name {bucket:?}: must be a single path segment"
        )));
    }
    Ok(())
}

/// Validates an object key and appends it to `directory`.
///
/// Keys are `/`-separated relative paths: empty keys, absolute keys,
/// backslashes, and any `.`/`..`/empty segment are rejected so the resulting
/// path can never escape the bucket directory.
fn object_path(directory: &Path, key: &str) -> Result<PathBuf> {
    let mut path = directory.to_path_buf();
    for segment in key_segments(key)? {
        path.push(segment);
    }
    Ok(path)
}

/// Splits and validates the `/`-separated segments of an object key.
fn key_segments(key: &str) -> Result<Vec<&str>> {
    if key.is_empty() {
        return Err(Error::validation("object key must not be empty"));
    }
    if key.starts_with('/') {
        return Err(Error::validation(format!(
            "invalid object key {key:?}: must be relative"
        )));
    }
    if key.contains('\\') {
        return Err(Error::validation(format!(
            "invalid object key {key:?}: must not contain backslashes"
        )));
    }
    let mut segments = Vec::new();
    for segment in key.split('/') {
        if segment.is_empty() || segment == "." || segment == ".." {
            return Err(Error::validation(format!(
                "invalid object key {key:?}: must not contain empty, '.' or '..' segments"
            )));
        }
        segments.push(segment);
    }
    Ok(segments)
}

/// Builds the `/`-joined key of `path` relative to the bucket `directory`.
///
/// Returns `None` for paths that are not losslessly representable as a UTF-8
/// key, which makes `list` skip them instead of failing.
fn relative_key(directory: &Path, path: &Path) -> Option<String> {
    let relative = path.strip_prefix(directory).ok()?;
    let mut segments = Vec::new();
    for component in relative.components() {
        match component {
            Component::Normal(segment) => segments.push(segment.to_str()?),
            Component::CurDir
            | Component::ParentDir
            | Component::RootDir
            | Component::Prefix(_) => return None,
        }
    }
    if segments.is_empty() {
        return None;
    }
    Some(segments.join("/"))
}

/// Wraps an I/O error with the operation and path that produced it.
fn io_error(action: &str, path: &Path, error: std::io::Error) -> Error {
    Error::external(error).with_context(format!("{action} {}", path.display()))
}

#[cfg(test)]
mod tests {
    use openjiuwen_common::error::ErrorKind;
    use tempfile::TempDir;

    use super::*;

    async fn store_in(root: &Path) -> LocalObjectStore {
        LocalObjectStore::new(root)
            .await
            .expect("open object store")
    }

    #[tokio::test]
    async fn new_creates_root_and_is_idempotent() {
        let temp = TempDir::new().expect("temp dir");
        let root = temp.path().join("nested/objects");
        let store = store_in(&root).await;
        assert_eq!(store.root(), root.as_path());
        assert!(root.is_dir());
        // Opening the same root again succeeds and keeps existing objects.
        store.create_bucket("b").await.expect("create bucket");
        store
            .put("b", "k", Bytes::from_static(b"x"))
            .await
            .expect("put");
        let reopened = store_in(&root).await;
        assert!(reopened.exists("b", "k").await.expect("exists"));
    }

    #[tokio::test]
    async fn put_get_exists_delete_round_trip() {
        let temp = TempDir::new().expect("temp dir");
        let store = store_in(temp.path()).await;
        store.create_bucket("b").await.expect("create bucket");

        let before = Utc::now();
        let meta = store
            .put("b", "dir/sub/a.txt", Bytes::from_static(b"hello"))
            .await
            .expect("put");
        assert_eq!(meta.key, "dir/sub/a.txt");
        assert_eq!(meta.size, 5);
        assert!(meta.updated_at >= before);
        assert!(temp.path().join("b/dir/sub/a.txt").is_file());

        assert!(store.exists("b", "dir/sub/a.txt").await.expect("exists"));
        assert_eq!(
            store.get("b", "dir/sub/a.txt").await.expect("get"),
            Some(Bytes::from_static(b"hello"))
        );

        store.delete("b", "dir/sub/a.txt").await.expect("delete");
        assert!(!store.exists("b", "dir/sub/a.txt").await.expect("exists"));
        assert_eq!(store.get("b", "dir/sub/a.txt").await.expect("get"), None);
        // Deleting a missing object is a no-op.
        store
            .delete("b", "dir/sub/a.txt")
            .await
            .expect("delete again");
    }

    #[tokio::test]
    async fn put_replaces_existing_payload() {
        let temp = TempDir::new().expect("temp dir");
        let store = store_in(temp.path()).await;
        store.create_bucket("b").await.expect("create bucket");
        store
            .put("b", "k", Bytes::from_static(b"one"))
            .await
            .expect("put");
        let meta = store
            .put("b", "k", Bytes::from_static(b"second"))
            .await
            .expect("put again");
        assert_eq!(meta.size, 6);
        assert_eq!(
            store.get("b", "k").await.expect("get"),
            Some(Bytes::from_static(b"second"))
        );
    }

    #[tokio::test]
    async fn bytes_round_trip_is_lossless_for_non_utf8() {
        let temp = TempDir::new().expect("temp dir");
        let store = store_in(temp.path()).await;
        store.create_bucket("b").await.expect("create bucket");
        let payload = Bytes::from_static(&[0xff, 0x00, 0x80, 0xfe]);
        store
            .put("b", "raw.bin", payload.clone())
            .await
            .expect("put");
        assert_eq!(store.get("b", "raw.bin").await.expect("get"), Some(payload));
    }

    #[tokio::test]
    async fn list_filters_by_prefix_sorted_and_limited() {
        let temp = TempDir::new().expect("temp dir");
        let store = store_in(temp.path()).await;
        store.create_bucket("b").await.expect("create bucket");
        for key in ["docs/b.txt", "docs/a.txt", "docs/nested/c.txt", "other.txt"] {
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
            vec!["docs/a.txt", "docs/b.txt", "docs/nested/c.txt", "other.txt"]
        );

        let docs: Vec<String> = store
            .list("b", "docs/", None)
            .await
            .expect("list prefix")
            .into_iter()
            .map(|meta| meta.key)
            .collect();
        assert_eq!(docs, vec!["docs/a.txt", "docs/b.txt", "docs/nested/c.txt"]);

        let limited = store.list("b", "docs/", Some(2)).await.expect("list limit");
        let limited_keys: Vec<&str> = limited.iter().map(|meta| meta.key.as_str()).collect();
        assert_eq!(limited_keys, vec!["docs/a.txt", "docs/b.txt"]);

        let listed = store.list("b", "other", None).await.expect("list one");
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].size, 1);

        assert_eq!(
            store.list("b", "missing/", None).await.expect("list none"),
            Vec::<ObjectMeta>::new()
        );
    }

    #[tokio::test]
    async fn missing_bucket_is_not_found() {
        let temp = TempDir::new().expect("temp dir");
        let store = store_in(temp.path()).await;
        for kind in [
            store
                .put("nope", "k", Bytes::from_static(b"x"))
                .await
                .unwrap_err()
                .kind(),
            store.get("nope", "k").await.unwrap_err().kind(),
            store.delete("nope", "k").await.unwrap_err().kind(),
            store.exists("nope", "k").await.unwrap_err().kind(),
            store.list("nope", "", None).await.unwrap_err().kind(),
        ] {
            assert_eq!(kind, ErrorKind::NotFound);
        }
        // A missing bucket on delete is a no-op.
        store
            .delete_bucket("nope")
            .await
            .expect("delete missing bucket");
    }

    #[tokio::test]
    async fn path_traversal_keys_are_rejected() {
        let temp = TempDir::new().expect("temp dir");
        let store = store_in(temp.path()).await;
        store.create_bucket("b").await.expect("create bucket");

        for key in [
            "",
            "/absolute.txt",
            "../escape.txt",
            "..",
            ".",
            "a/../../escape.txt",
            "a/./b.txt",
            "a//b.txt",
            "dir/",
            "back\\slash.txt",
        ] {
            let error = store
                .put("b", key, Bytes::from_static(b"x"))
                .await
                .unwrap_err();
            assert_eq!(error.kind(), ErrorKind::Validation, "key {key:?}");
            assert!(
                store.get("b", key).await.is_err(),
                "get rejects key {key:?}"
            );
            assert!(
                store.delete("b", key).await.is_err(),
                "delete rejects key {key:?}"
            );
            assert!(
                store.exists("b", key).await.is_err(),
                "exists rejects key {key:?}"
            );
        }

        // Nothing escaped the bucket.
        assert!(!temp.path().join("escape.txt").exists());
        assert!(temp.path().join("b").is_dir());
    }

    #[tokio::test]
    async fn invalid_bucket_names_are_rejected() {
        let temp = TempDir::new().expect("temp dir");
        let store = store_in(temp.path()).await;
        for bucket in ["", ".", "..", "a/b", "../b", "back\\slash", "a\0b"] {
            let error = store.create_bucket(bucket).await.unwrap_err();
            assert_eq!(error.kind(), ErrorKind::Validation, "bucket {bucket:?}");
            assert!(store.delete_bucket(bucket).await.is_err());
            assert!(store.list(bucket, "", None).await.is_err());
        }
        // No stray directories were created.
        assert_eq!(
            store.list_buckets().await.expect("list buckets"),
            Vec::<String>::new()
        );
    }

    #[tokio::test]
    async fn delete_bucket_removes_contents() {
        let temp = TempDir::new().expect("temp dir");
        let store = store_in(temp.path()).await;
        store.create_bucket("b").await.expect("create bucket");
        store
            .put("b", "dir/a.txt", Bytes::from_static(b"x"))
            .await
            .expect("put");
        store
            .put("b", "a.txt", Bytes::from_static(b"y"))
            .await
            .expect("put");

        store.delete_bucket("b").await.expect("delete bucket");
        assert_eq!(
            store.list_buckets().await.expect("list buckets"),
            Vec::<String>::new()
        );
        assert_eq!(
            store.get("b", "a.txt").await.unwrap_err().kind(),
            ErrorKind::NotFound
        );
        assert!(!temp.path().join("b").exists());
    }

    #[tokio::test]
    async fn persistence_across_reopen() {
        let temp = TempDir::new().expect("temp dir");
        {
            let store = store_in(temp.path()).await;
            store.create_bucket("first").await.expect("create first");
            store.create_bucket("second").await.expect("create second");
            store
                .put(
                    "first",
                    "dir/data.bin",
                    Bytes::from_static(&[1, 2, 3, 0xff]),
                )
                .await
                .expect("put");
        }

        let reopened = store_in(temp.path()).await;
        assert_eq!(
            reopened.list_buckets().await.expect("list buckets"),
            vec!["first".to_string(), "second".to_string()]
        );
        assert_eq!(
            reopened.get("first", "dir/data.bin").await.expect("get"),
            Some(Bytes::from_static(&[1, 2, 3, 0xff]))
        );
        let listed = reopened.list("first", "", None).await.expect("list");
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].key, "dir/data.bin");
        assert_eq!(listed[0].size, 4);
    }

    #[tokio::test]
    async fn list_buckets_sorted_and_skips_files() {
        let temp = TempDir::new().expect("temp dir");
        let store = store_in(temp.path()).await;
        for bucket in ["zeta", "alpha", "beta"] {
            store.create_bucket(bucket).await.expect("create bucket");
        }
        // A bucket name is created twice without error.
        store.create_bucket("alpha").await.expect("create again");
        tokio::fs::write(temp.path().join("stray.txt"), b"x")
            .await
            .expect("write stray file");
        assert_eq!(
            store.list_buckets().await.expect("list buckets"),
            vec!["alpha".to_string(), "beta".to_string(), "zeta".to_string()]
        );
    }
}
