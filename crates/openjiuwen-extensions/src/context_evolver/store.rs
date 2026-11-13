//! Per-user persistence seam for normalized memory records.
//!
//! Ports the storage half of Python's
//! `openjiuwen/extensions/context_evolver/core/persistence.py`
//! (`MemoryPersistenceHelper`) and `core/file_connector/JSONFileConnector.py`:
//! a small, explicitly typed seam that every summarize/retrieve pipeline uses
//! instead of touching a backend directly.
//!
//! Two implementations ship:
//!
//! * [`InMemoryMemoryStore`] — the test/dev backend, an in-process map keyed by
//!   user and then record id.
//! * [`JsonFileMemoryStore`] — one JSON file per user inside a directory,
//!   holding `{"memories": [...]}` (the Python persistence shape).
//!
//! Milvus persistence, Python's `persist_type="auto"` probe, and the
//! `{algo_name}` segment of Python's `persist_path` template are not ported:
//! there is no Milvus transport in this crate, and the algorithm is already
//! encoded in the generated record id, so one file per user is enough.
//!
//! # Deviations from Python
//!
//! * Python writes `{node_id: node_dict}` mappings merged into an existing
//!   file; this port treats the file as an ordered list of
//!   [`MemoryRecord`]s, so feedback counters and record identity survive a
//!   round trip.
//! * The on-disk file name is `{sanitised_user}-{hash8}.json` rather than
//!   `{user_id}.json`, so user ids with separators or non-ASCII characters
//!   cannot escape the directory, and two users whose sanitised names collide
//!   still get distinct files.
//! * `hashlib.md5` is replaced by the first 8 hex characters of SHA-256.
//! * Writes are atomic (temporary sibling + rename) and serialized per store,
//!   where Python truncates the target file in place.
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex, RwLock};

use super::record::{sanitize_user, sha256_hex_prefix, MemoryRecord};

/// The persistence seam every `context_evolver` pipeline stores memories
/// through.
///
/// Implementations are `Send + Sync` and cheap to clone (clones share their
/// backend state), so they can be handed to concurrent pipelines.
///
/// Contract:
///
/// * [`MemoryStore::load`] on an unknown user returns an empty [`Vec`], never
///   an error, and always returns records ordered by ascending
///   [`MemoryRecord::id`] so callers are deterministic.
/// * [`MemoryStore::replace`] replaces only that user's records; every other
///   user is untouched, and the stored copies carry the requested `user_id`.
/// * [`MemoryStore::upsert`] inserts or replaces one record, keyed by
///   [`MemoryRecord::id`], files it under its own
///   [`MemoryRecord::user_id`], and rejects a blank id with
///   [`Error::Validation`].
/// * [`MemoryStore::clear`] removes every record for that user and leaves
///   other users untouched.
#[async_trait]
pub trait MemoryStore: Send + Sync + std::fmt::Debug {
    /// Loads every stored record for `user_id`, ordered by ascending
    /// [`MemoryRecord::id`].
    ///
    /// # Errors
    ///
    /// Implementations report persistence or deserialization failures; an
    /// unknown user is not an error.
    async fn load(&self, user_id: &str) -> Result<Vec<MemoryRecord>>;

    /// Replaces the entire record set for `user_id`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when a record id is blank, and the
    /// implementation's persistence error otherwise.
    async fn replace(&self, user_id: &str, records: Vec<MemoryRecord>) -> Result<()>;

    /// Inserts or replaces one record, keyed by [`MemoryRecord::id`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when [`MemoryRecord::id`] or
    /// [`MemoryRecord::user_id`] is blank, and the implementation's
    /// persistence error otherwise.
    async fn upsert(&self, record: MemoryRecord) -> Result<()>;

    /// Removes every record for `user_id`.
    ///
    /// # Errors
    ///
    /// Implementations report persistence failures.
    async fn clear(&self, user_id: &str) -> Result<()>;
}

/// An in-process [`MemoryStore`] backed by a user → id → record map.
///
/// Cloning a store shares the same state (it sits behind an [`Arc`]); use
/// [`InMemoryMemoryStore::new`] for an independent store, exactly like
/// `MemoryVectorStore`.
#[derive(Debug, Clone, Default)]
pub struct InMemoryMemoryStore {
    users: Arc<RwLock<BTreeMap<String, BTreeMap<String, MemoryRecord>>>>,
}

impl InMemoryMemoryStore {
    /// Creates an empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl MemoryStore for InMemoryMemoryStore {
    async fn load(&self, user_id: &str) -> Result<Vec<MemoryRecord>> {
        let users = self.users.read().await;
        Ok(users
            .get(user_id)
            .map(|records| records.values().cloned().collect())
            .unwrap_or_default())
    }

    async fn replace(&self, user_id: &str, records: Vec<MemoryRecord>) -> Result<()> {
        let mut keyed = BTreeMap::new();
        for mut record in records {
            require_id(&record)?;
            record.user_id = user_id.to_string();
            keyed.insert(record.id.clone(), record);
        }
        self.users.write().await.insert(user_id.to_string(), keyed);
        Ok(())
    }

    async fn upsert(&self, record: MemoryRecord) -> Result<()> {
        require_id(&record)?;
        require_user(&record)?;
        let mut users = self.users.write().await;
        users
            .entry(record.user_id.clone())
            .or_default()
            .insert(record.id.clone(), record);
        Ok(())
    }

    async fn clear(&self, user_id: &str) -> Result<()> {
        self.users.write().await.remove(user_id);
        Ok(())
    }
}

/// A [`MemoryStore`] holding one JSON file per user inside a directory.
///
/// The file for a user is [`JsonFileMemoryStore::path_for`], which is
/// `{dir}/{sanitised_user}-{hash8}.json`. Its shape is the Python persistence
/// shape, pretty-printed with two spaces:
///
/// ```json
/// {"memories": [{"id": "ace_user-1_0123456789abcdef", "user_id": "user-1", "content": "…"}]}
/// ```
///
/// Each record carries `id`, `user_id`, `helpful`, `harmful`, and `source_type`
/// alongside the `AddMemoryRequest` fields (`content`, `when_to_use`, `title`,
/// `description`, `section`, `label`, `attributes`). A missing file, a missing
/// `memories` key, and unknown extra keys all load as an empty list; malformed
/// JSON is [`Error::Validation`].
///
/// Cloning a store shares the same directory and write lock.
#[derive(Debug, Clone)]
pub struct JsonFileMemoryStore {
    dir: Arc<PathBuf>,
    write_lock: Arc<Mutex<()>>,
}

impl JsonFileMemoryStore {
    /// Creates a store rooted at `dir`, without creating the directory yet.
    ///
    /// The directory is created on the first [`MemoryStore::upsert`] or
    /// [`MemoryStore::replace`].
    #[must_use]
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self {
            dir: Arc::new(dir.into()),
            write_lock: Arc::new(Mutex::new(())),
        }
    }

    /// The file that backs `user_id`: `{dir}/{sanitised_user}-{hash8}.json`.
    ///
    /// The user segment keeps ASCII alphanumerics plus `.`, `_`, and `-` and
    /// replaces everything else with `_` (empty becomes `default`); `hash8` is
    /// the first 8 hex characters of SHA-256 over the raw `user_id`, so
    /// sanitisation collisions cannot merge two users' files.
    #[must_use]
    pub fn path_for(&self, user_id: &str) -> PathBuf {
        self.dir.join(format!(
            "{}-{}.json",
            sanitize_user(user_id),
            sha256_hex_prefix(user_id, 8)
        ))
    }

    /// Reads the records for `user_id`; a missing file is an empty list.
    async fn read_records(&self, user_id: &str) -> Result<Vec<MemoryRecord>> {
        let path = self.path_for(user_id);
        let bytes = match tokio::fs::read(&path).await {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => return Err(read_write_error("read", &path, &error)),
        };
        let mut file: MemoryFile = serde_json::from_slice(&bytes).map_err(|error| {
            Error::Validation(format!(
                "context_evolver memory file {} is not valid JSON: {error}",
                path.display()
            ))
        })?;
        file.memories.sort_by(|left, right| left.id.cmp(&right.id));
        Ok(file.memories)
    }

    /// Writes `records` for `user_id` through a temporary sibling and a rename.
    async fn write_records(&self, user_id: &str, records: &[MemoryRecord]) -> Result<()> {
        let path = self.path_for(user_id);
        tokio::fs::create_dir_all(self.dir.as_ref())
            .await
            .map_err(|error| read_write_error("create", self.dir.as_ref(), &error))?;
        let file = MemoryFile {
            memories: records.to_vec(),
        };
        let bytes = serde_json::to_vec_pretty(&file).map_err(|error| {
            Error::Memory(format!(
                "context_evolver memory store failed to serialize memories for '{user_id}': {error}"
            ))
        })?;
        let temp = Self::temp_path(&path);
        tokio::fs::write(&temp, &bytes)
            .await
            .map_err(|error| read_write_error("write", &temp, &error))?;
        tokio::fs::rename(&temp, &path)
            .await
            .map_err(|error| read_write_error("rename", &temp, &error))
    }

    /// A unique `*.tmp` sibling of `path`; unique across stores and processes
    /// so a stale temporary file is never reused.
    fn temp_path(path: &Path) -> PathBuf {
        static SEQUENCE: AtomicU64 = AtomicU64::new(0);
        let sequence = SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let name = path.file_name().map_or_else(
            || "memories".to_string(),
            |name| name.to_string_lossy().into_owned(),
        );
        path.with_file_name(format!("{name}.{}.{sequence}.tmp", std::process::id()))
    }
}

#[async_trait]
impl MemoryStore for JsonFileMemoryStore {
    async fn load(&self, user_id: &str) -> Result<Vec<MemoryRecord>> {
        let _guard = self.write_lock.lock().await;
        self.read_records(user_id).await
    }

    async fn replace(&self, user_id: &str, records: Vec<MemoryRecord>) -> Result<()> {
        let mut stamped = Vec::with_capacity(records.len());
        for mut record in records {
            require_id(&record)?;
            record.user_id = user_id.to_string();
            stamped.push(record);
        }
        stamped.sort_by(|left, right| left.id.cmp(&right.id));
        let _guard = self.write_lock.lock().await;
        self.write_records(user_id, &stamped).await
    }

    async fn upsert(&self, record: MemoryRecord) -> Result<()> {
        require_id(&record)?;
        require_user(&record)?;
        let user_id = record.user_id.clone();
        let _guard = self.write_lock.lock().await;
        let mut records: BTreeMap<String, MemoryRecord> = self
            .read_records(&user_id)
            .await?
            .into_iter()
            .map(|record| (record.id.clone(), record))
            .collect();
        records.insert(record.id.clone(), record);
        let ordered: Vec<MemoryRecord> = records.into_values().collect();
        self.write_records(&user_id, &ordered).await
    }

    async fn clear(&self, user_id: &str) -> Result<()> {
        let path = self.path_for(user_id);
        let _guard = self.write_lock.lock().await;
        match tokio::fs::remove_file(&path).await {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(read_write_error("remove", &path, &error)),
        }
    }
}

/// The on-disk document: the Python `{"memories": [...]}` persistence shape.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct MemoryFile {
    /// Stored records; a missing key means an empty list.
    #[serde(default)]
    memories: Vec<MemoryRecord>,
}

/// Rejects a record whose id cannot key the store.
fn require_id(record: &MemoryRecord) -> Result<()> {
    if record.id.trim().is_empty() {
        return Err(Error::Validation(format!(
            "context_evolver memory store requires a non-empty record id (user '{}')",
            record.user_id
        )));
    }
    Ok(())
}

/// Rejects a record that cannot be filed under a user.
fn require_user(record: &MemoryRecord) -> Result<()> {
    if record.user_id.trim().is_empty() {
        return Err(Error::Validation(format!(
            "context_evolver memory store requires a non-empty user_id (record '{}')",
            record.id
        )));
    }
    Ok(())
}

/// Maps a filesystem failure onto the memory subsystem error with the path.
fn read_write_error(action: &str, path: &Path, error: &std::io::Error) -> Error {
    Error::Memory(format!(
        "context_evolver memory store failed to {action} {}: {error}",
        path.display()
    ))
}

#[cfg(test)]
mod tests {
    // Python counterparts: `tests/unit_tests/extensions/context_evolver/`
    // (`TestMemoryPersistence` / JSON connector round trips) plus the
    // persistence behavior every `PersistMemoryOp` relies on.
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use crate::context_evolver::record::MemoryDraft;
    use openjiuwen_common::error::ErrorKind;
    use serde_json::json;

    fn record(id: &str, user_id: &str, content: &str) -> MemoryRecord {
        MemoryRecord::new(user_id, MemoryDraft::new(content)).with_id(id)
    }

    fn ids(records: &[MemoryRecord]) -> Vec<&str> {
        records.iter().map(|record| record.id.as_str()).collect()
    }

    #[tokio::test]
    async fn in_memory_upsert_loads_in_ascending_id_order() {
        let store = InMemoryMemoryStore::new();
        for (id, content) in [("c", "third"), ("a", "first"), ("b", "second")] {
            store.upsert(record(id, "user-1", content)).await.unwrap();
        }
        let loaded = store.load("user-1").await.unwrap();
        assert_eq!(ids(&loaded), ["a", "b", "c"]);
        assert_eq!(loaded[0].content, "first");
        assert_eq!(store.load("unknown").await.unwrap(), Vec::new());
    }

    #[tokio::test]
    async fn in_memory_upsert_replaces_by_id() {
        let store = InMemoryMemoryStore::new();
        store.upsert(record("a", "user-1", "first")).await.unwrap();
        store
            .upsert(record("a", "user-1", "updated"))
            .await
            .unwrap();
        let loaded = store.load("user-1").await.unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].content, "updated");
    }

    #[tokio::test]
    async fn in_memory_replace_touches_only_the_target_user() {
        let store = InMemoryMemoryStore::new();
        store
            .upsert(record("a", "user-1", "keep me"))
            .await
            .unwrap();
        store.upsert(record("z", "user-2", "other")).await.unwrap();

        let incoming = record("b", "ignored", "second");
        store
            .replace("user-1", vec![record("d", "user-9", "d"), incoming])
            .await
            .unwrap();

        let loaded = store.load("user-1").await.unwrap();
        assert_eq!(ids(&loaded), ["b", "d"]);
        assert!(loaded.iter().all(|record| record.user_id == "user-1"));
        assert_eq!(ids(&store.load("user-2").await.unwrap()), ["z"]);

        store.replace("user-1", Vec::new()).await.unwrap();
        assert_eq!(store.load("user-1").await.unwrap(), Vec::new());
    }

    #[tokio::test]
    async fn in_memory_clear_removes_one_user() {
        let store = InMemoryMemoryStore::new();
        store.upsert(record("a", "user-1", "x")).await.unwrap();
        store.upsert(record("b", "user-2", "y")).await.unwrap();
        store.clear("user-1").await.unwrap();
        assert_eq!(store.load("user-1").await.unwrap(), Vec::new());
        assert_eq!(ids(&store.load("user-2").await.unwrap()), ["b"]);
        store.clear("unknown").await.unwrap();
    }

    #[tokio::test]
    async fn in_memory_rejects_blank_ids_and_users() {
        let store = InMemoryMemoryStore::new();
        let error = store.upsert(record("  ", "user-1", "x")).await.unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.message().contains("record id"), "{}", error.message());

        let error = store.upsert(record("a", "", "x")).await.unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.message().contains("user_id"), "{}", error.message());

        let error = store
            .replace("user-1", vec![record("", "user-1", "x")])
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);

        assert_eq!(store.load("user-1").await.unwrap(), Vec::new());
    }

    #[tokio::test]
    async fn in_memory_clones_share_state() {
        let store = InMemoryMemoryStore::new();
        let clone = store.clone();
        clone.upsert(record("a", "user-1", "x")).await.unwrap();
        assert_eq!(ids(&store.load("user-1").await.unwrap()), ["a"]);
        assert_eq!(
            InMemoryMemoryStore::default().load("user-1").await.unwrap(),
            Vec::new()
        );
    }

    #[tokio::test]
    async fn json_store_round_trips_and_writes_the_python_shape() {
        let dir = tempfile::tempdir().unwrap();
        let store = JsonFileMemoryStore::new(dir.path());
        let mut memory = record("ace_user-1_0123456789abcdef", "user-1", "Prefer rg");
        memory.helpful = 2;
        memory.harmful = 1;
        memory.source_type = Some("success".into());
        memory
            .attributes
            .insert("target".into(), json!("workspace"));

        store.upsert(memory.clone()).await.unwrap();
        assert_eq!(store.load("user-1").await.unwrap(), vec![memory.clone()]);
        assert_eq!(store.load("unknown").await.unwrap(), Vec::new());

        let path = store.path_for("user-1");
        assert!(path.exists());
        let raw = std::fs::read_to_string(&path).unwrap();
        assert!(raw.starts_with("{\n  \"memories\": ["), "{raw}");

        let on_disk: serde_json::Value = serde_json::from_str(&raw).unwrap();
        let memories = on_disk["memories"].as_array().unwrap();
        assert_eq!(memories.len(), 1);
        assert_eq!(memories[0]["id"], json!("ace_user-1_0123456789abcdef"));
        assert_eq!(memories[0]["user_id"], json!("user-1"));
        assert_eq!(memories[0]["content"], json!("Prefer rg"));
        assert_eq!(memories[0]["section"], json!("general"));
        assert_eq!(memories[0]["helpful"], json!(2));
        assert_eq!(memories[0]["harmful"], json!(1));
        assert_eq!(memories[0]["source_type"], json!("success"));
        assert_eq!(memories[0]["attributes"]["target"], json!("workspace"));

        assert_eq!(store.load("unknown").await.unwrap(), Vec::new());
    }

    #[tokio::test]
    async fn json_store_creates_the_directory_on_first_write() {
        let dir = tempfile::tempdir().unwrap();
        let nested = dir.path().join("nested").join("memories");
        let store = JsonFileMemoryStore::new(&nested);
        assert!(!nested.exists());
        store.upsert(record("a", "user-1", "x")).await.unwrap();
        assert!(nested.is_dir());
        assert_eq!(ids(&store.load("user-1").await.unwrap()), ["a"]);
    }

    #[tokio::test]
    async fn json_store_replace_and_clear() {
        let dir = tempfile::tempdir().unwrap();
        let store = JsonFileMemoryStore::new(dir.path());
        store
            .replace(
                "user-1",
                vec![record("b", "other", "b"), record("a", "other", "a")],
            )
            .await
            .unwrap();
        store.upsert(record("z", "user-2", "z")).await.unwrap();

        let loaded = store.load("user-1").await.unwrap();
        assert_eq!(ids(&loaded), ["a", "b"]);
        assert!(loaded.iter().all(|record| record.user_id == "user-1"));

        store.clear("user-1").await.unwrap();
        assert_eq!(store.load("user-1").await.unwrap(), Vec::new());
        assert!(!store.path_for("user-1").exists());
        assert_eq!(ids(&store.load("user-2").await.unwrap()), ["z"]);
        store.clear("unknown").await.unwrap();
    }

    #[test]
    fn path_for_is_deterministic_and_sanitised() {
        let dir = tempfile::tempdir().unwrap();
        let store = JsonFileMemoryStore::new(dir.path());
        assert_eq!(store.path_for("user-1"), store.path_for("user-1"));

        let path = store.path_for("a b/c");
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        assert!(name.starts_with("a_b_c-"), "{name}");
        assert!(
            Path::new(&name)
                .extension()
                .is_some_and(|extension| extension.eq_ignore_ascii_case("json")),
            "{name}"
        );
        let hash = &name["a_b_c-".len()..name.len() - ".json".len()];
        assert_eq!(hash.len(), 8);
        assert!(hash.chars().all(|character| character.is_ascii_hexdigit()));
        assert_eq!(
            hash,
            super::super::record::sha256_hex_prefix("a b/c", 8).as_str()
        );
        assert_eq!(path.parent(), Some(dir.path()));

        let empty = store.path_for("");
        assert!(empty
            .file_name()
            .unwrap()
            .to_string_lossy()
            .starts_with("default-"));

        assert_ne!(store.path_for("a b"), store.path_for("a_b"));
    }

    #[tokio::test]
    async fn json_store_tolerates_missing_files_keys_and_extras() {
        let dir = tempfile::tempdir().unwrap();
        let store = JsonFileMemoryStore::new(dir.path());
        assert_eq!(store.load("user-1").await.unwrap(), Vec::new());

        let path = store.path_for("user-1");
        std::fs::write(
            &path,
            r#"{"other": 1, "memories": [{"id": "a", "content": "x"}]}"#,
        )
        .unwrap();
        assert_eq!(ids(&store.load("user-1").await.unwrap()), ["a"]);

        std::fs::write(&path, r#"{"other": 1}"#).unwrap();
        assert_eq!(store.load("user-1").await.unwrap(), Vec::new());

        std::fs::write(&path, "{}").unwrap();
        assert_eq!(store.load("user-1").await.unwrap(), Vec::new());
    }

    #[tokio::test]
    async fn json_store_reports_malformed_files_as_validation() {
        let dir = tempfile::tempdir().unwrap();
        let store = JsonFileMemoryStore::new(dir.path());
        let path = store.path_for("user-1");
        std::fs::write(&path, "{ not json").unwrap();

        let error = store.load("user-1").await.unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(
            error.message().contains(&path.display().to_string()),
            "{}",
            error.message()
        );
        assert!(error.message().contains("not valid JSON"));

        let error = store.upsert(record("a", "user-1", "x")).await.unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
    }

    #[tokio::test]
    async fn json_store_rejects_blank_ids_without_writing() {
        let dir = tempfile::tempdir().unwrap();
        let store = JsonFileMemoryStore::new(dir.path());
        let error = store.upsert(record("", "user-1", "x")).await.unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(!store.path_for("user-1").exists());

        let error = store
            .replace(
                "user-1",
                vec![record("a", "user-1", "x"), record(" ", "user-1", "y")],
            )
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert_eq!(store.load("user-1").await.unwrap(), Vec::new());
    }

    #[tokio::test]
    async fn json_store_concurrent_upserts_keep_every_record() {
        let dir = tempfile::tempdir().unwrap();
        let store = JsonFileMemoryStore::new(dir.path());
        let (first, second, third, fourth) = tokio::join!(
            store.upsert(record("d", "user-1", "d")),
            store.upsert(record("b", "user-1", "b")),
            store.upsert(record("a", "user-1", "a")),
            store.upsert(record("c", "user-1", "c")),
        );
        for result in [first, second, third, fourth] {
            result.unwrap();
        }
        assert_eq!(
            ids(&store.load("user-1").await.unwrap()),
            ["a", "b", "c", "d"]
        );
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
    }
}
