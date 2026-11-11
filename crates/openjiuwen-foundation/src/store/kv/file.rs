//! Local-filesystem key-value store.
//!
//! One JSON envelope per key below `<root>/entries/`, named after the SHA-256 of
//! the key. The store keeps an in-memory index of key → TTL so prefix scans and
//! TTL checks work after a reopen without reading every file.
//!
//! This replaces Python's `shelve_store.py` / `db_based_kv_store.py`: the format is
//! plain JSON (inspectable and portable) and writes are atomic (write-then-rename).

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use base64::Engine as _;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use openjiuwen_common::error::{Error, Result};

use super::{KvPipeline, KvStore, KvValue, StorePipeline};

/// Directory holding the entry files, relative to the store root.
const ENTRIES_DIR: &str = "entries";
/// File extension of a committed entry file.
const ENTRY_EXT: &str = "json";
/// Envelope tag for a text value.
const KIND_TEXT: &str = "text";
/// Envelope tag for a byte value.
const KIND_BYTES: &str = "bytes";

/// The on-disk representation of one key-value pair.
///
/// `data` holds the raw text for a text value and the base64 payload for a byte
/// value; `expiry_ms` is an absolute epoch-millisecond deadline, absent when the
/// entry never expires.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Envelope {
    /// The logical key (the filename is its hash).
    key: String,
    /// `"text"` or `"bytes"`.
    kind: String,
    /// Raw text, or base64 for bytes.
    data: String,
    /// Absolute expiry in epoch milliseconds.
    #[serde(default)]
    expiry_ms: Option<i64>,
}

impl Envelope {
    /// Encodes a value into an envelope.
    fn new(key: &str, value: &KvValue, expiry: Option<Instant>) -> Self {
        let (kind, data) = match value {
            KvValue::Text(text) => (KIND_TEXT, text.clone()),
            KvValue::Bytes(bytes) => (
                KIND_BYTES,
                base64::engine::general_purpose::STANDARD.encode(bytes),
            ),
        };
        Self {
            key: key.to_string(),
            kind: kind.to_string(),
            data,
            expiry_ms: expiry.map(epoch_ms_from),
        }
    }

    /// Decodes the stored value.
    fn into_value(self) -> Result<KvValue> {
        match self.kind.as_str() {
            KIND_TEXT => Ok(KvValue::Text(self.data)),
            KIND_BYTES => {
                let bytes = base64::engine::general_purpose::STANDARD
                    .decode(self.data.as_bytes())
                    .map_err(Error::external)?;
                Ok(KvValue::Bytes(bytes))
            }
            other => Err(Error::validation(format!(
                "unknown kv envelope kind `{other}` for key `{}`",
                self.key
            ))),
        }
    }

    /// The TTL deadline carried by the envelope, as a monotonic instant.
    fn expiry_instant(&self) -> Option<Instant> {
        self.expiry_ms.map(instant_from_epoch_ms)
    }
}

/// A key-value store persisted as one JSON file per key below a root directory.
///
/// Clones share the same index (and therefore the same files); concurrent writers
/// in one process are serialized by the index lock.
#[derive(Debug, Clone)]
pub struct LocalFileKvStore {
    /// The caller-provided root; nothing outside it is ever touched.
    root: PathBuf,
    /// `<root>/entries`, where the envelope files live.
    entries_dir: PathBuf,
    /// Key → optional TTL, rebuilt by scanning `entries/` on open.
    index: Arc<Mutex<BTreeMap<String, Option<Instant>>>>,
}

impl LocalFileKvStore {
    /// Opens (or creates) a store rooted at `root`.
    ///
    /// The directory tree is created when missing, then the existing envelopes are
    /// scanned to rebuild the key index, so entries written by an earlier process
    /// remain visible with their TTLs intact.
    pub async fn new(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        let entries_dir = root.join(ENTRIES_DIR);
        tokio::fs::create_dir_all(&entries_dir)
            .await
            .map_err(|err| file_error(err, &entries_dir))?;
        let store = Self {
            root,
            entries_dir,
            index: Arc::new(Mutex::new(BTreeMap::new())),
        };
        store.load_index().await?;
        Ok(store)
    }

    /// The store root.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Number of live entries (expired entries are purged from the index first).
    pub fn len(&self) -> usize {
        let now = Instant::now();
        let mut index = self.index.lock();
        index.retain(|_, expiry| expiry.is_none_or(|expiry| now < expiry));
        index.len()
    }

    /// Whether the store holds no live entry.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The file backing `key`.
    fn path_for(&self, key: &str) -> PathBuf {
        self.entries_dir
            .join(format!("{}.{ENTRY_EXT}", key_hash(key)))
    }

    /// Scans `entries/` and rebuilds the key index.
    async fn load_index(&self) -> Result<()> {
        let mut entries = tokio::fs::read_dir(&self.entries_dir)
            .await
            .map_err(|err| file_error(err, &self.entries_dir))?;
        let mut index = BTreeMap::new();
        while let Some(entry) = entries
            .next_entry()
            .await
            .map_err(|err| file_error(err, &self.entries_dir))?
        {
            let path = entry.path();
            // Only committed envelopes count; in-flight `*.tmp` files are ignored.
            if path.extension().and_then(|ext| ext.to_str()) != Some(ENTRY_EXT) {
                continue;
            }
            let raw = tokio::fs::read(&path)
                .await
                .map_err(|err| file_error(err, &path))?;
            let envelope: Envelope = serde_json::from_slice(&raw).map_err(|err| {
                Error::external(err).with_context(format!("reading {}", path.display()))
            })?;
            index.insert(envelope.key.clone(), envelope.expiry_instant());
        }
        *self.index.lock() = index;
        Ok(())
    }

    /// Writes `value` to the file for `key` atomically.
    async fn write_entry(&self, key: &str, value: &KvValue, expiry: Option<Instant>) -> Result<()> {
        let envelope = Envelope::new(key, value, expiry);
        let bytes = serde_json::to_vec(&envelope)
            .map_err(|err| Error::external(err).with_context(format!("encoding key `{key}`")))?;
        let target = self.path_for(key);
        let tmp = self
            .entries_dir
            .join(format!("{}.tmp", uuid::Uuid::new_v4()));
        tokio::fs::write(&tmp, &bytes)
            .await
            .map_err(|err| file_error(err, &tmp))?;
        if let Err(err) = tokio::fs::rename(&tmp, &target).await {
            if let Err(cleanup) = tokio::fs::remove_file(&tmp).await {
                tracing::debug!(
                    path = %tmp.display(),
                    error = %cleanup,
                    "failed to remove temporary kv entry file"
                );
            }
            return Err(file_error(err, &target));
        }
        Ok(())
    }

    /// Reads the file for `key`; `None` when it does not exist.
    async fn read_entry(&self, key: &str) -> Result<Option<KvValue>> {
        let path = self.path_for(key);
        let raw = match tokio::fs::read(&path).await {
            Ok(raw) => raw,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(err) => return Err(file_error(err, &path)),
        };
        let envelope: Envelope = serde_json::from_slice(&raw).map_err(|err| {
            Error::external(err).with_context(format!("reading {}", path.display()))
        })?;
        Ok(Some(envelope.into_value()?))
    }

    /// Removes the file for `key`; a missing file is not an error.
    async fn remove_file(&self, key: &str) -> Result<()> {
        let path = self.path_for(key);
        match tokio::fs::remove_file(&path).await {
            Ok(()) => Ok(()),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(err) => Err(file_error(err, &path)),
        }
    }

    /// The live keys starting with `prefix`, purging expired index entries.
    fn live_keys_with_prefix(&self, prefix: &str) -> Vec<String> {
        let now = Instant::now();
        let mut index = self.index.lock();
        index.retain(|_, expiry| expiry.is_none_or(|expiry| now < expiry));
        index
            .range(prefix.to_string()..)
            .take_while(|(key, _)| key.starts_with(prefix))
            .map(|(key, _)| key.clone())
            .collect()
    }
}

#[async_trait]
impl KvStore for LocalFileKvStore {
    async fn set(&self, key: &str, value: KvValue) -> Result<()> {
        // Write first: a failed write must not make the index claim a key exists.
        self.write_entry(key, &value, None).await?;
        self.index.lock().insert(key.to_string(), None);
        Ok(())
    }

    async fn get(&self, key: &str) -> Result<Option<KvValue>> {
        let now = Instant::now();
        let live = {
            let mut index = self.index.lock();
            match index.get(key) {
                None => return Ok(None),
                Some(expiry) if expiry.is_none_or(|expiry| now < expiry) => true,
                Some(_) => {
                    // Expired: drop it from the index and delete the file below.
                    index.remove(key);
                    false
                }
            }
        };
        if !live {
            self.remove_file(key).await?;
            return Ok(None);
        }
        let Some(value) = self.read_entry(key).await? else {
            // The file vanished behind our back; keep the index honest.
            self.index.lock().remove(key);
            return Ok(None);
        };
        Ok(Some(value))
    }

    async fn exists(&self, key: &str) -> Result<bool> {
        Ok(self.get(key).await?.is_some())
    }

    async fn delete(&self, key: &str) -> Result<()> {
        self.index.lock().remove(key);
        self.remove_file(key).await
    }

    async fn exclusive_set(
        &self,
        key: &str,
        value: KvValue,
        expiry: Option<Duration>,
    ) -> Result<bool> {
        let now = Instant::now();
        let expiry_at = expiry.map(|ttl| now + ttl);
        {
            let mut index = self.index.lock();
            if let Some(expiry) = index.get(key) {
                if expiry.is_none_or(|expiry| now < expiry) {
                    return Ok(false);
                }
            }
            // Reserve the key before awaiting the write so a concurrent
            // `exclusive_set` on the same key cannot also succeed.
            index.insert(key.to_string(), expiry_at);
        }
        if let Err(err) = self.write_entry(key, &value, expiry_at).await {
            // Roll back the reservation: nobody else could have taken it.
            self.index.lock().remove(key);
            return Err(err);
        }
        Ok(true)
    }

    async fn get_by_prefix(&self, prefix: &str) -> Result<BTreeMap<String, KvValue>> {
        let keys = self.live_keys_with_prefix(prefix);
        let mut entries = BTreeMap::new();
        for key in keys {
            if let Some(value) = self.read_entry(&key).await? {
                entries.insert(key, value);
            }
        }
        Ok(entries)
    }

    async fn delete_by_prefix(&self, prefix: &str) -> Result<usize> {
        let keys = self.live_keys_with_prefix(prefix);
        {
            let mut index = self.index.lock();
            for key in &keys {
                index.remove(key);
            }
        }
        for key in &keys {
            self.remove_file(key).await?;
        }
        Ok(keys.len())
    }

    async fn mget(&self, keys: &[String]) -> Result<Vec<Option<KvValue>>> {
        let mut values = Vec::with_capacity(keys.len());
        for key in keys {
            values.push(self.get(key).await?);
        }
        Ok(values)
    }

    async fn batch_delete(&self, keys: &[String]) -> Result<usize> {
        let mut deleted = 0;
        for key in keys {
            if self.get(key).await?.is_some() {
                self.delete(key).await?;
                deleted += 1;
            } else {
                // Absent or expired: make sure no stale file can resurrect the key
                // when the store is reopened.
                self.remove_file(key).await?;
            }
        }
        Ok(deleted)
    }

    fn pipeline(&self) -> Box<dyn KvPipeline + '_> {
        Box::new(StorePipeline::new(self))
    }
}

/// The hex SHA-256 of a key, used as the entry filename.
fn key_hash(key: &str) -> String {
    let digest = Sha256::digest(key.as_bytes());
    let mut hex = String::with_capacity(digest.len() * 2);
    for byte in digest {
        // Writing into a `String` cannot fail.
        let _ = write!(hex, "{byte:02x}");
    }
    hex
}

/// Milliseconds since the Unix epoch.
fn now_epoch_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| {
            i64::try_from(elapsed.as_millis()).unwrap_or(i64::MAX)
        })
}

/// The epoch-millisecond representation of an absolute instant.
fn epoch_ms_from(instant: Instant) -> i64 {
    let remaining = instant.saturating_duration_since(Instant::now());
    let millis = i64::try_from(remaining.as_millis()).unwrap_or(i64::MAX);
    now_epoch_ms().saturating_add(millis)
}

/// Converts an epoch-millisecond deadline into an instant.
///
/// Deadlines already in the past yield an instant that is not in the future, so
/// the entry reads as expired immediately.
fn instant_from_epoch_ms(ms: i64) -> Instant {
    let remaining = ms.saturating_sub(now_epoch_ms());
    let millis = u64::try_from(remaining).unwrap_or(0);
    Instant::now() + Duration::from_millis(millis)
}

/// Wraps an I/O error with the path that produced it.
fn file_error(err: std::io::Error, path: &Path) -> Error {
    Error::external(err).with_context(format!("kv store path {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store_dir(dir: &tempfile::TempDir) -> PathBuf {
        dir.path().join("kv")
    }

    #[tokio::test]
    async fn text_and_bytes_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let store = LocalFileKvStore::new(store_dir(&dir)).await.unwrap();

        store.set("text", KvValue::text("hello")).await.unwrap();
        store
            .set("bytes", KvValue::bytes(vec![0_u8, 255, 7]))
            .await
            .unwrap();

        assert_eq!(
            store.get("text").await.unwrap(),
            Some(KvValue::text("hello"))
        );
        assert_eq!(
            store.get("bytes").await.unwrap(),
            Some(KvValue::bytes(vec![0_u8, 255, 7]))
        );
        assert_eq!(store.get("missing").await.unwrap(), None);
        assert!(store.exists("text").await.unwrap());
        assert!(!store.exists("missing").await.unwrap());
        assert_eq!(store.len(), 2);
    }

    #[tokio::test]
    async fn envelope_shape_and_filename_follow_the_contract() {
        let dir = tempfile::tempdir().unwrap();
        let store = LocalFileKvStore::new(store_dir(&dir)).await.unwrap();
        store.set("greeting", KvValue::text("hi")).await.unwrap();

        let path = store_dir(&dir)
            .join(ENTRIES_DIR)
            .join(format!("{}.json", key_hash("greeting")));
        assert!(path.exists());
        let raw: serde_json::Value =
            serde_json::from_slice(&tokio::fs::read(&path).await.unwrap()).unwrap();
        assert_eq!(raw["key"], serde_json::json!("greeting"));
        assert_eq!(raw["kind"], serde_json::json!("text"));
        assert_eq!(raw["data"], serde_json::json!("hi"));
        assert_eq!(raw["expiry_ms"], serde_json::Value::Null);

        store
            .set("blob", KvValue::bytes(vec![1_u8, 2]))
            .await
            .unwrap();
        let path = store_dir(&dir)
            .join(ENTRIES_DIR)
            .join(format!("{}.json", key_hash("blob")));
        let raw: serde_json::Value =
            serde_json::from_slice(&tokio::fs::read(&path).await.unwrap()).unwrap();
        assert_eq!(raw["kind"], serde_json::json!("bytes"));
        assert_eq!(raw["data"], serde_json::json!("AQI="));
    }

    #[tokio::test]
    async fn entries_survive_reopen() {
        let dir = tempfile::tempdir().unwrap();
        {
            let store = LocalFileKvStore::new(store_dir(&dir)).await.unwrap();
            store.set("a/1", KvValue::text("1")).await.unwrap();
            store.set("a/2", KvValue::text("2")).await.unwrap();
            store.set("b", KvValue::bytes(vec![9_u8])).await.unwrap();
        }

        let reopened = LocalFileKvStore::new(store_dir(&dir)).await.unwrap();
        assert_eq!(reopened.len(), 3);
        assert_eq!(reopened.get("a/1").await.unwrap(), Some(KvValue::text("1")));
        assert_eq!(
            reopened.get("b").await.unwrap(),
            Some(KvValue::bytes(vec![9_u8]))
        );
        let prefixed = reopened.get_by_prefix("a/").await.unwrap();
        assert_eq!(
            prefixed.keys().cloned().collect::<Vec<_>>(),
            vec!["a/1".to_string(), "a/2".to_string()]
        );
    }

    #[tokio::test]
    async fn ttl_survives_reopen_and_allows_reacquire() {
        let dir = tempfile::tempdir().unwrap();
        {
            let store = LocalFileKvStore::new(store_dir(&dir)).await.unwrap();
            assert!(store
                .exclusive_set("lock", KvValue::text("a"), Some(Duration::from_millis(20)))
                .await
                .unwrap());
            assert!(!store
                .exclusive_set("lock", KvValue::text("b"), None)
                .await
                .unwrap());
        }

        tokio::time::sleep(Duration::from_millis(80)).await;

        let reopened = LocalFileKvStore::new(store_dir(&dir)).await.unwrap();
        assert_eq!(reopened.get("lock").await.unwrap(), None);
        assert!(!reopened.exists("lock").await.unwrap());
        assert!(reopened
            .exclusive_set("lock", KvValue::text("c"), None)
            .await
            .unwrap());
        assert_eq!(
            reopened.get("lock").await.unwrap(),
            Some(KvValue::text("c"))
        );
    }

    #[tokio::test]
    async fn get_by_prefix_supports_empty_prefix_and_is_sorted() {
        let dir = tempfile::tempdir().unwrap();
        let store = LocalFileKvStore::new(store_dir(&dir)).await.unwrap();
        store.set("b", KvValue::text("2")).await.unwrap();
        store.set("a", KvValue::text("1")).await.unwrap();

        let all = store.get_by_prefix("").await.unwrap();
        assert_eq!(
            all.keys().cloned().collect::<Vec<_>>(),
            vec!["a".to_string(), "b".to_string()]
        );
        assert!(store.get_by_prefix("zzz").await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn delete_and_prefix_deletes_remove_files() {
        let dir = tempfile::tempdir().unwrap();
        let store = LocalFileKvStore::new(store_dir(&dir)).await.unwrap();
        for key in ["p/1", "p/2", "q/1"] {
            store.set(key, KvValue::text(key)).await.unwrap();
        }

        assert_eq!(store.delete_by_prefix("p/").await.unwrap(), 2);
        assert_eq!(store.delete_by_prefix("p/").await.unwrap(), 0);
        assert_eq!(store.len(), 1);

        store.delete("q/1").await.unwrap();
        // Deleting a missing key is a no-op.
        store.delete("q/1").await.unwrap();
        assert!(store.get_by_prefix("").await.unwrap().is_empty());
        assert!(!store_dir(&dir)
            .join(ENTRIES_DIR)
            .join(format!("{}.json", key_hash("q/1")))
            .exists());
    }

    #[tokio::test]
    async fn mget_preserves_order_and_misses() {
        let dir = tempfile::tempdir().unwrap();
        let store = LocalFileKvStore::new(store_dir(&dir)).await.unwrap();
        store.set("a", KvValue::text("1")).await.unwrap();
        store.set("c", KvValue::text("3")).await.unwrap();

        let keys = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        assert_eq!(
            store.mget(&keys).await.unwrap(),
            vec![Some(KvValue::text("1")), None, Some(KvValue::text("3")),]
        );
        assert_eq!(store.mget(&[]).await.unwrap().len(), 0);
    }

    #[tokio::test]
    async fn batch_delete_counts_present_keys_only() {
        let dir = tempfile::tempdir().unwrap();
        let store = LocalFileKvStore::new(store_dir(&dir)).await.unwrap();
        store.set("a", KvValue::text("1")).await.unwrap();
        store.set("b", KvValue::text("2")).await.unwrap();

        let keys = vec![
            "a".to_string(),
            "b".to_string(),
            "a".to_string(),
            "missing".to_string(),
        ];
        assert_eq!(store.batch_delete(&keys).await.unwrap(), 2);
        assert_eq!(store.batch_delete(&[]).await.unwrap(), 0);
        assert!(store.is_empty());
    }

    #[tokio::test]
    async fn pipeline_applies_operations_in_order() {
        let dir = tempfile::tempdir().unwrap();
        let store = LocalFileKvStore::new(store_dir(&dir)).await.unwrap();
        store.set("existing", KvValue::text("old")).await.unwrap();

        let mut pipeline = StorePipeline::new(&store);
        pipeline.set("queued", KvValue::text("new"));
        pipeline.get("queued");
        pipeline.exists("queued");
        pipeline.exists("missing");
        pipeline.get("existing");
        assert_eq!(pipeline.len(), 5);

        assert_eq!(
            pipeline.execute().await.unwrap(),
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
    }

    #[tokio::test]
    async fn store_stays_inside_its_root() {
        let dir = tempfile::tempdir().unwrap();
        let root = store_dir(&dir);
        let store = LocalFileKvStore::new(&root).await.unwrap();
        assert_eq!(store.root(), root.as_path());
        store
            .set("../../../escape", KvValue::text("x"))
            .await
            .unwrap();

        let mut entries = tokio::fs::read_dir(root.join(ENTRIES_DIR)).await.unwrap();
        let mut count = 0;
        while entries.next_entry().await.unwrap().is_some() {
            count += 1;
        }
        assert_eq!(count, 1);
        assert!(!dir.path().join("escape.json").exists());
    }
}
