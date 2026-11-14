//! Persistence for the experience bank: the [`ExperienceStore`] seam and its backends.
//!
//! Python keeps the bank in an `index_dir` directory (`meta.json`, a JSONL scalar
//! file, and FAISS vector files) and repairs it from a backup snapshot when the
//! manifest disagrees with the data. This seam stores only the durable items: the
//! embeddings are recomputed by the configured
//! [`openjiuwen_retrieval::Embedding`] at search time. The catalog that also persists the
//! vectors and their integrity manifest is
//! [`ExperienceIndex`](crate::experience::ExperienceIndex).
//!
//! Three backends implement the seam:
//!
//! - [`MemoryExperienceStore`](crate::experience::MemoryExperienceStore) — the default
//!   for [`crate::experience::ExperienceBank::new`]; useful for tests and ephemeral runs.
//! - [`JsonFileExperienceStore`](crate::experience::JsonFileExperienceStore) — one JSON
//!   document file (JSONL, or a JSON array), with the whole file rewritten atomically
//!   through `<file>.tmp` plus `rename` so a reader never observes a half-written bank.
//!   Malformed lines are logged and skipped.
//! - `SqliteExperienceStore` (feature `sqlite`) — the item JSON stored under
//!   `experience/<id>` in an `openjiuwen-foundation` `SqliteKvStore`.
//!
//! Every write is a full-item replacement: an item is small, and a store never merges
//! on its own — the dedupe/merge rules live in [`crate::experience::ExperienceBank`].
#![allow(clippy::unused_async_trait_impl)] // the `ExperienceStore` contract is async for the I/O backends

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
#[cfg(feature = "sqlite")]
use std::sync::Arc;
use std::sync::{PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard};

use async_trait::async_trait;
#[cfg(feature = "sqlite")]
use openjiuwen_foundation::store::kv::{KvStore, KvValue};
use tracing::warn;

use crate::error::{Error, Result};
use crate::experience::ExperienceItem;

/// The KV key prefix used by `SqliteExperienceStore` values (`experience/<id>`).
pub const DEFAULT_KEY_PREFIX: &str = "experience/";

/// Persistence for experience items, keyed by their id.
///
/// Reads return `Option`/empty collections rather than
/// [`Error::NotFound`](crate::error::Error::NotFound) because a missing item is a normal
/// catalog state, not a failure.
#[async_trait]
pub trait ExperienceStore: Send + Sync {
    /// Lists every stored item, sorted by id.
    ///
    /// Malformed entries are skipped (with a warning) instead of failing the scan.
    ///
    /// # Errors
    ///
    /// Returns the backend's error when the underlying read fails.
    async fn list(&self) -> Result<Vec<ExperienceItem>>;

    /// Returns the item stored under `id`, or `None` when it is absent.
    ///
    /// # Errors
    ///
    /// Returns the backend's error when the underlying read fails; a malformed value
    /// is reported as `None` with a warning.
    async fn get(&self, id: &str) -> Result<Option<ExperienceItem>>;

    /// Inserts or replaces `item`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::External`] when the item cannot be serialized, or the backend's
    /// error when the write fails.
    async fn put(&self, item: &ExperienceItem) -> Result<()>;

    /// Removes `id`, returning whether it existed.
    ///
    /// # Errors
    ///
    /// Returns the backend's error when the read or delete fails.
    async fn remove(&self, id: &str) -> Result<bool>;

    /// Removes every stored item.
    ///
    /// # Errors
    ///
    /// Returns the backend's error when the delete fails.
    async fn clear(&self) -> Result<()>;
}

/// An [`ExperienceStore`] that keeps items in process memory.
///
/// The map lives behind a [`std::sync::RwLock`] that is only held for map operations,
/// never across an `.await`.
#[derive(Debug, Default)]
pub struct MemoryExperienceStore {
    /// The items, keyed and ordered by id.
    items: RwLock<BTreeMap<String, ExperienceItem>>,
}

impl MemoryExperienceStore {
    /// Creates an empty in-memory store.
    #[must_use]
    pub fn new() -> Self {
        Self {
            items: RwLock::new(BTreeMap::new()),
        }
    }

    /// The number of stored items.
    #[must_use]
    pub fn len(&self) -> usize {
        self.read().len()
    }

    /// Reports whether the store holds no items.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.read().is_empty()
    }

    /// Locks the map for reading, recovering from a poisoned lock.
    fn read(&self) -> RwLockReadGuard<'_, BTreeMap<String, ExperienceItem>> {
        self.items.read().unwrap_or_else(PoisonError::into_inner)
    }

    /// Locks the map for writing, recovering from a poisoned lock.
    fn write(&self) -> RwLockWriteGuard<'_, BTreeMap<String, ExperienceItem>> {
        self.items.write().unwrap_or_else(PoisonError::into_inner)
    }
}

#[async_trait]
impl ExperienceStore for MemoryExperienceStore {
    async fn list(&self) -> Result<Vec<ExperienceItem>> {
        Ok(self.read().values().cloned().collect())
    }

    async fn get(&self, id: &str) -> Result<Option<ExperienceItem>> {
        Ok(self.read().get(id).cloned())
    }

    async fn put(&self, item: &ExperienceItem) -> Result<()> {
        self.write().insert(item.id.clone(), item.clone());
        Ok(())
    }

    async fn remove(&self, id: &str) -> Result<bool> {
        Ok(self.write().remove(id).is_some())
    }

    async fn clear(&self) -> Result<()> {
        self.write().clear();
        Ok(())
    }
}

/// An [`ExperienceStore`] backed by one JSON document file.
///
/// The file is written as JSONL (one item JSON per line) and accepted as either JSONL
/// or a JSON array, so a hand-edited or externally produced bank still loads. Every
/// mutation rewrites the whole file through `<file>.tmp` plus a rename, which makes the
/// swap atomic on every supported platform. Blank lines and malformed lines are logged
/// with `tracing::warn!` and skipped; an item without an id cannot be addressed and is
/// skipped too.
#[derive(Debug)]
pub struct JsonFileExperienceStore {
    /// The document path.
    path: PathBuf,
    /// The loaded catalog, keyed and ordered by id.
    items: RwLock<BTreeMap<String, ExperienceItem>>,
}

impl JsonFileExperienceStore {
    /// Opens (or creates) the bank document at `path` and loads its items.
    ///
    /// A missing file is an empty bank, not an error; the file is created by the first
    /// write.
    ///
    /// # Errors
    ///
    /// Returns [`Error::External`] when the file exists but cannot be read.
    pub async fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let items = match tokio::fs::read_to_string(&path).await {
            Ok(text) => parse_document(&text, &path),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => BTreeMap::new(),
            Err(error) => return Err(Error::external(error)),
        };
        Ok(Self {
            path,
            items: RwLock::new(items),
        })
    }

    /// The path of the backing document.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Reloads the catalog from disk and returns the number of items loaded.
    ///
    /// # Errors
    ///
    /// Returns [`Error::External`] when the file exists but cannot be read.
    pub async fn reload(&self) -> Result<usize> {
        let items = match tokio::fs::read_to_string(&self.path).await {
            Ok(text) => parse_document(&text, &self.path),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => BTreeMap::new(),
            Err(error) => return Err(Error::external(error)),
        };
        let count = items.len();
        *self.write() = items;
        Ok(count)
    }

    /// Locks the map for reading, recovering from a poisoned lock.
    fn read(&self) -> RwLockReadGuard<'_, BTreeMap<String, ExperienceItem>> {
        self.items.read().unwrap_or_else(PoisonError::into_inner)
    }

    /// Locks the map for writing, recovering from a poisoned lock.
    fn write(&self) -> RwLockWriteGuard<'_, BTreeMap<String, ExperienceItem>> {
        self.items.write().unwrap_or_else(PoisonError::into_inner)
    }

    /// Serializes the whole catalog and atomically replaces the document.
    async fn persist(&self) -> Result<()> {
        let document = {
            let items = self.read();
            let mut document = String::new();
            for item in items.values() {
                let line = serde_json::to_string(item).map_err(Error::external)?;
                document.push_str(&line);
                document.push('\n');
            }
            document
        };
        write_atomic(&self.path, document.as_bytes()).await
    }
}

/// Atomically replaces `path` with `bytes` through a `<path>.tmp` sibling and a rename.
///
/// Missing parent directories are created first, so a backend can publish into a fresh
/// directory. A reader therefore observes either the previous document or the complete new
/// one, never a partial write.
///
/// # Errors
///
/// Returns [`Error::External`] when a directory cannot be created, the temporary file
/// cannot be written, or the rename fails.
pub(crate) async fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(Error::external)?;
        }
    }
    let temporary = temporary_path(path);
    tokio::fs::write(&temporary, bytes)
        .await
        .map_err(Error::external)?;
    tokio::fs::rename(&temporary, path)
        .await
        .map_err(Error::external)
}

#[async_trait]
impl ExperienceStore for JsonFileExperienceStore {
    async fn list(&self) -> Result<Vec<ExperienceItem>> {
        Ok(self.read().values().cloned().collect())
    }

    async fn get(&self, id: &str) -> Result<Option<ExperienceItem>> {
        Ok(self.read().get(id).cloned())
    }

    async fn put(&self, item: &ExperienceItem) -> Result<()> {
        self.write().insert(item.id.clone(), item.clone());
        self.persist().await
    }

    async fn remove(&self, id: &str) -> Result<bool> {
        let removed = self.write().remove(id).is_some();
        if removed {
            self.persist().await?;
        }
        Ok(removed)
    }

    async fn clear(&self) -> Result<()> {
        self.write().clear();
        self.persist().await
    }
}

/// An [`ExperienceStore`] backed by an `SQLite` key-value database.
///
/// Each item is stored as JSON text under `experience/<id>`, the same layout
/// [`crate::store::KvAssetStore`] uses for capability descriptors.
#[cfg(feature = "sqlite")]
pub struct SqliteExperienceStore {
    /// The backing key-value store.
    kv: Arc<dyn KvStore>,
    /// Prefix applied to every key written by this store.
    prefix: String,
}

#[cfg(feature = "sqlite")]
impl std::fmt::Debug for SqliteExperienceStore {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SqliteExperienceStore")
            .field("prefix", &self.prefix)
            .finish_non_exhaustive()
    }
}

#[cfg(feature = "sqlite")]
impl SqliteExperienceStore {
    /// Connects to the `SQLite` database at `url` (for example `sqlite:experience.db`)
    /// and prepares the schema.
    ///
    /// # Errors
    ///
    /// Returns the foundation backend's error when the database cannot be opened or
    /// migrated.
    pub async fn connect(url: &str) -> Result<Self> {
        let kv = openjiuwen_foundation::store::kv::SqliteKvStore::connect(url).await?;
        Ok(Self::with_kv(Arc::new(kv)))
    }

    /// Wraps an existing key-value store, writing under [`DEFAULT_KEY_PREFIX`].
    #[must_use]
    pub fn with_kv(kv: Arc<dyn KvStore>) -> Self {
        Self {
            kv,
            prefix: DEFAULT_KEY_PREFIX.to_string(),
        }
    }

    /// The configured key prefix.
    #[must_use]
    pub fn prefix(&self) -> &str {
        &self.prefix
    }

    /// The KV key of `id`.
    fn key_for(&self, id: &str) -> String {
        format!("{}{id}", self.prefix)
    }
}

#[cfg(feature = "sqlite")]
#[async_trait]
impl ExperienceStore for SqliteExperienceStore {
    async fn list(&self) -> Result<Vec<ExperienceItem>> {
        let entries = self.kv.get_by_prefix(&self.prefix).await?;
        let mut items: Vec<ExperienceItem> = entries
            .iter()
            .filter_map(|(key, value)| parse_kv_entry(key, value))
            .collect();
        items.sort_by(|left, right| left.id.cmp(&right.id));
        Ok(items)
    }

    async fn get(&self, id: &str) -> Result<Option<ExperienceItem>> {
        let key = self.key_for(id);
        let Some(value) = self.kv.get(&key).await? else {
            return Ok(None);
        };
        Ok(parse_kv_entry(&key, &value))
    }

    async fn put(&self, item: &ExperienceItem) -> Result<()> {
        let key = self.key_for(&item.id);
        let json = serde_json::to_string(item).map_err(Error::external)?;
        self.kv.set(&key, KvValue::text(json)).await
    }

    async fn remove(&self, id: &str) -> Result<bool> {
        let key = self.key_for(id);
        let existed = self.kv.get(&key).await?.is_some();
        if existed {
            self.kv.delete(&key).await?;
        }
        Ok(existed)
    }

    async fn clear(&self) -> Result<()> {
        self.kv.delete_by_prefix(&self.prefix).await?;
        Ok(())
    }
}

/// The `<file>.tmp` sibling used for the atomic document swap.
pub(crate) fn temporary_path(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_os_string();
    name.push(".tmp");
    PathBuf::from(name)
}

/// Parses a bank document: a JSON array, or JSONL when it does not start with `[`.
///
/// Every entry is decoded on its own, so one corrupt line does not discard the rest.
fn parse_document(text: &str, path: &Path) -> BTreeMap<String, ExperienceItem> {
    let mut items = BTreeMap::new();
    let trimmed = text.trim_start();
    if trimmed.is_empty() {
        return items;
    }
    let display_path = path.display().to_string();
    if trimmed.starts_with('[') {
        match serde_json::from_str::<Vec<serde_json::Value>>(trimmed) {
            Ok(entries) => {
                for (index, entry) in entries.into_iter().enumerate() {
                    match serde_json::from_value::<ExperienceItem>(entry) {
                        Ok(item) => insert_item(&mut items, item, &display_path),
                        Err(error) => warn!(
                            path = %display_path,
                            index,
                            %error,
                            "experience item is malformed; skipping it"
                        ),
                    }
                }
            }
            Err(error) => warn!(
                path = %display_path,
                %error,
                "experience file is not a JSON array; starting empty"
            ),
        }
        return items;
    }
    for (index, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        match serde_json::from_str::<ExperienceItem>(line) {
            Ok(item) => insert_item(&mut items, item, &display_path),
            Err(error) => warn!(
                path = %display_path,
                index,
                %error,
                "experience line is malformed; skipping it"
            ),
        }
    }
    items
}

/// Inserts a parsed item, warning when it cannot be addressed because it has no id.
fn insert_item(items: &mut BTreeMap<String, ExperienceItem>, item: ExperienceItem, path: &str) {
    if item.id.trim().is_empty() {
        warn!(path, "experience item has no id; skipping it");
        return;
    }
    items.insert(item.id.clone(), item);
}

/// Parses one KV entry, warning and returning `None` when it is malformed.
#[cfg(feature = "sqlite")]
fn parse_kv_entry(key: &str, value: &KvValue) -> Option<ExperienceItem> {
    let Some(text) = value.as_str() else {
        warn!(key, "experience entry is not text; skipping it");
        return None;
    };
    match serde_json::from_str::<ExperienceItem>(text) {
        Ok(item) => Some(item),
        Err(error) => {
            warn!(
                key,
                %error,
                "experience entry is not a valid item; skipping it"
            );
            None
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]
mod tests {
    #[cfg(feature = "sqlite")]
    use openjiuwen_foundation::store::kv::{InMemoryKvStore, KvStore};

    use super::*;

    fn sample(id: &str) -> ExperienceItem {
        ExperienceItem::new("Summarize the report")
            .with_id(id)
            .with_query_examples(["summarize this report"])
            .with_skill_ids(["summarizer"])
    }

    #[tokio::test]
    async fn memory_store_round_trips_and_lists_in_order() {
        let store = MemoryExperienceStore::new();
        assert!(store.is_empty());
        store.put(&sample("exp_0002")).await.expect("put");
        store.put(&sample("exp_0001")).await.expect("put");

        assert_eq!(store.len(), 2);
        let listed = store.list().await.expect("list");
        assert_eq!(
            listed
                .iter()
                .map(|item| item.id.as_str())
                .collect::<Vec<_>>(),
            vec!["exp_0001", "exp_0002"]
        );
        assert!(store.get("exp_0001").await.expect("get").is_some());
        assert!(store.get("absent").await.expect("get").is_none());

        assert!(store.remove("exp_0001").await.expect("remove"));
        assert!(!store.remove("exp_0001").await.expect("remove again"));
        store.clear().await.expect("clear");
        assert!(store.is_empty());
    }

    #[tokio::test]
    async fn json_file_store_reopens_and_persists_json_lines() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("experience.jsonl");
        {
            let store = JsonFileExperienceStore::open(&path).await.expect("open");
            assert_eq!(store.path(), path.as_path());
            store.put(&sample("exp_0001")).await.expect("put");
            store.put(&sample("exp_0002")).await.expect("put");
        }
        let text = tokio::fs::read_to_string(&path).await.expect("read");
        assert_eq!(text.lines().count(), 2);
        assert!(text.contains(r#""id":"exp_0001""#));
        assert!(!tokio::fs::try_exists(temporary_path(&path))
            .await
            .expect("stat"));

        let reopened = JsonFileExperienceStore::open(&path).await.expect("reopen");
        assert_eq!(reopened.list().await.expect("list").len(), 2);
        assert_eq!(reopened.reload().await.expect("reload"), 2);
    }

    #[tokio::test]
    async fn json_file_store_accepts_a_json_array() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("array.json");
        let document = serde_json::json!([sample("exp_0001"), sample("exp_0002")]);
        tokio::fs::write(&path, serde_json::to_vec(&document).expect("encode"))
            .await
            .expect("write");
        let store = JsonFileExperienceStore::open(&path).await.expect("open");
        assert_eq!(store.list().await.expect("list").len(), 2);
    }

    #[tokio::test]
    async fn malformed_lines_are_skipped_and_removal_rewrites_the_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("experience.jsonl");
        let valid = serde_json::to_string(&sample("exp_0001")).expect("encode");
        let document = format!("{valid}\n{{not json\n\n");
        tokio::fs::write(&path, document).await.expect("write");

        let store = JsonFileExperienceStore::open(&path).await.expect("open");
        let listed = store.list().await.expect("list");
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].id, "exp_0001");

        assert!(store.remove("exp_0001").await.expect("remove"));
        let text = tokio::fs::read_to_string(&path).await.expect("read");
        assert!(text.is_empty());
        assert!(!store.remove("exp_0001").await.expect("remove again"));
    }

    #[test]
    fn document_parsing_skips_items_without_an_id() {
        let items = parse_document(
            "{\"query_pattern\":\"orphan\"}\n{\"id\":\"exp_0001\"}\n",
            Path::new("bank.jsonl"),
        );
        assert_eq!(items.len(), 1);
        assert!(items.contains_key("exp_0001"));
    }

    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn sqlite_store_round_trips_and_clear_empties_the_namespace() {
        let dir = tempfile::tempdir().expect("tempdir");
        let url = format!("sqlite:{}", dir.path().join("experience.db").display());
        let store = SqliteExperienceStore::connect(&url).await.expect("connect");
        assert_eq!(store.prefix(), DEFAULT_KEY_PREFIX);
        store.put(&sample("exp_0001")).await.expect("put");
        store.put(&sample("exp_0002")).await.expect("put");
        assert_eq!(store.list().await.expect("list").len(), 2);
        assert!(store.get("exp_0001").await.expect("get").is_some());
        assert!(store.remove("exp_0001").await.expect("remove"));
        assert!(!store.remove("exp_0001").await.expect("remove again"));
        store.clear().await.expect("clear");
        assert!(store.list().await.expect("list").is_empty());
    }

    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn kv_store_skips_malformed_values() {
        let kv: Arc<dyn KvStore> = Arc::new(InMemoryKvStore::new());
        kv.set("experience/exp_0009", KvValue::text("{not an item"))
            .await
            .expect("write malformed");
        let store = SqliteExperienceStore::with_kv(kv);
        assert!(store.list().await.expect("list").is_empty());
        assert!(store.get("exp_0009").await.expect("get").is_none());
    }
}
