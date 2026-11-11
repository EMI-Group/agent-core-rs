//! Local-filesystem document store.
//!
//! One JSON snapshot per collection below the store root, named
//! `<collection>.json` and shaped
//!
//! ```json
//! {"collection": "docs", "documents": [{"id": "a", "fields": {"n": 1}}]}
//! ```
//!
//! The snapshots are read once when the store opens and then held in memory, so
//! reads never touch the disk and a reopen always sees the last committed state.
//! Every mutation rewrites the whole snapshot (write-then-rename, so a reader
//! never observes a half-written file). This replaces Python's `SQLAlchemy`
//! `BaseDbStore` for deployments that need no database server: the format is
//! plain JSON (inspectable and portable) and nothing outside the root is ever
//! touched.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use parking_lot::Mutex;

use openjiuwen_common::error::{Error, Result};

use super::{matching, run_query, validate_collection, Document, DocumentStore, Query};

/// File extension of a committed snapshot.
const SNAPSHOT_EXT: &str = "json";

/// The on-disk form of one collection.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct Snapshot {
    /// The collection name; must match the file stem.
    collection: String,
    /// Every document of the collection.
    #[serde(default)]
    documents: Vec<Document>,
}

/// A [`DocumentStore`] persisted as one JSON file per collection.
///
/// Clones share the same in-memory view and the same write lock, so concurrent
/// writes through clones cannot lose documents.
#[derive(Debug, Clone)]
pub struct LocalFileDocumentStore {
    /// The caller-provided root; nothing outside it is ever touched.
    root: PathBuf,
    /// The loaded snapshots: collection name → document id → document.
    collections: Arc<Mutex<BTreeMap<String, BTreeMap<String, Document>>>>,
    /// Serializes read-modify-write-snapshot cycles.
    write_lock: Arc<tokio::sync::Mutex<()>>,
}

impl LocalFileDocumentStore {
    /// Opens (or creates) a store rooted at `root`.
    ///
    /// The directory is created when missing, then every `*.json` snapshot is
    /// loaded. Non-JSON files (including in-flight `*.tmp` writes) are ignored;
    /// a malformed snapshot fails the open so corrupt data is never served.
    pub async fn new(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        tokio::fs::create_dir_all(&root)
            .await
            .map_err(|err| file_error(err, &root))?;
        let store = Self {
            root,
            collections: Arc::new(Mutex::new(BTreeMap::new())),
            write_lock: Arc::new(tokio::sync::Mutex::new(())),
        };
        store.load().await?;
        Ok(store)
    }

    /// The store root.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The snapshot file of `collection`.
    fn path_for(&self, collection: &str) -> PathBuf {
        self.root.join(format!("{collection}.{SNAPSHOT_EXT}"))
    }

    /// Reads every snapshot below the root into memory.
    async fn load(&self) -> Result<()> {
        let mut entries = tokio::fs::read_dir(&self.root)
            .await
            .map_err(|err| file_error(err, &self.root))?;
        let mut collections = BTreeMap::new();
        while let Some(entry) = entries
            .next_entry()
            .await
            .map_err(|err| file_error(err, &self.root))?
        {
            let path = entry.path();
            if path.extension().and_then(|ext| ext.to_str()) != Some(SNAPSHOT_EXT) {
                continue;
            }
            let raw = tokio::fs::read(&path)
                .await
                .map_err(|err| file_error(err, &path))?;
            let snapshot: Snapshot = serde_json::from_slice(&raw).map_err(|err| {
                Error::external(err).with_context(format!("reading {}", path.display()))
            })?;
            validate_collection(&snapshot.collection)?;
            let stem = path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .unwrap_or_default();
            if stem != snapshot.collection {
                return Err(Error::validation(format!(
                    "document snapshot {} declares collection `{}`",
                    path.display(),
                    snapshot.collection
                )));
            }
            let documents = snapshot
                .documents
                .into_iter()
                .map(|document| (document.id.clone(), document))
                .collect();
            collections.insert(snapshot.collection, documents);
        }
        *self.collections.lock() = collections;
        Ok(())
    }

    /// A copy of the currently loaded documents of `collection`.
    fn snapshot_of(&self, collection: &str) -> BTreeMap<String, Document> {
        self.collections
            .lock()
            .get(collection)
            .cloned()
            .unwrap_or_default()
    }

    /// Applies `mutate` to a copy of the collection, persists it, then commits it.
    ///
    /// Writing the snapshot before the in-memory commit keeps memory and disk
    /// consistent: a failed write leaves the previous state in both.
    async fn mutate<M>(&self, collection: &str, mutate: M) -> Result<()>
    where
        M: FnOnce(&mut BTreeMap<String, Document>),
    {
        let _guard = self.write_lock.lock().await;
        let mut updated = self.snapshot_of(collection);
        mutate(&mut updated);
        self.write_snapshot(collection, &updated).await?;
        self.collections
            .lock()
            .insert(collection.to_string(), updated);
        Ok(())
    }

    /// Rewrites `<root>/<collection>.json` atomically.
    async fn write_snapshot(
        &self,
        collection: &str,
        documents: &BTreeMap<String, Document>,
    ) -> Result<()> {
        let snapshot = Snapshot {
            collection: collection.to_string(),
            documents: documents.values().cloned().collect(),
        };
        let bytes = serde_json::to_vec(&snapshot).map_err(|err| {
            Error::external(err).with_context(format!("encoding collection `{collection}`"))
        })?;
        let target = self.path_for(collection);
        let tmp = self.root.join(format!(
            "{collection}.{SNAPSHOT_EXT}.{}.tmp",
            uuid::Uuid::new_v4()
        ));
        tokio::fs::write(&tmp, &bytes)
            .await
            .map_err(|err| file_error(err, &tmp))?;
        if let Err(err) = tokio::fs::rename(&tmp, &target).await {
            if let Err(cleanup) = tokio::fs::remove_file(&tmp).await {
                tracing::debug!(
                    path = %tmp.display(),
                    error = %cleanup,
                    "failed to remove temporary document snapshot"
                );
            }
            return Err(file_error(err, &target));
        }
        Ok(())
    }
}

#[async_trait]
impl DocumentStore for LocalFileDocumentStore {
    async fn create_collection(&self, collection: &str) -> Result<()> {
        validate_collection(collection)?;
        if self.collections.lock().contains_key(collection) {
            return Ok(());
        }
        // Persist an empty snapshot so the collection survives a reopen.
        self.mutate(collection, |_| {}).await
    }

    async fn list_collections(&self) -> Result<Vec<String>> {
        Ok(self.collections.lock().keys().cloned().collect())
    }

    async fn drop_collection(&self, collection: &str) -> Result<()> {
        validate_collection(collection)?;
        let _guard = self.write_lock.lock().await;
        let path = self.path_for(collection);
        match tokio::fs::remove_file(&path).await {
            Ok(()) => {}
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
            Err(err) => return Err(file_error(err, &path)),
        }
        self.collections.lock().remove(collection);
        Ok(())
    }

    async fn upsert(&self, collection: &str, documents: Vec<Document>) -> Result<usize> {
        validate_collection(collection)?;
        let written = documents.len();
        self.mutate(collection, move |stored| {
            for document in documents {
                stored.insert(document.id.clone(), document);
            }
        })
        .await?;
        Ok(written)
    }

    async fn get(&self, collection: &str, id: &str) -> Result<Option<Document>> {
        validate_collection(collection)?;
        Ok(self
            .collections
            .lock()
            .get(collection)
            .and_then(|documents| documents.get(id).cloned()))
    }

    async fn query(&self, collection: &str, query: &Query) -> Result<Vec<Document>> {
        validate_collection(collection)?;
        let collections = self.collections.lock();
        let Some(documents) = collections.get(collection) else {
            return Ok(Vec::new());
        };
        Ok(run_query(documents.values(), query)
            .into_iter()
            .cloned()
            .collect())
    }

    async fn delete(&self, collection: &str, query: &Query) -> Result<usize> {
        validate_collection(collection)?;
        let doomed: Vec<String> = {
            let collections = self.collections.lock();
            let Some(documents) = collections.get(collection) else {
                return Ok(0);
            };
            matching(documents.values(), &query.filter)
                .into_iter()
                .map(|document| document.id.clone())
                .collect()
        };
        if doomed.is_empty() {
            return Ok(0);
        }
        let removed = doomed.len();
        self.mutate(collection, move |stored| {
            for id in &doomed {
                stored.remove(id);
            }
        })
        .await?;
        Ok(removed)
    }

    async fn count(&self, collection: &str, query: &Query) -> Result<usize> {
        validate_collection(collection)?;
        let collections = self.collections.lock();
        let Some(documents) = collections.get(collection) else {
            return Ok(0);
        };
        Ok(matching(documents.values(), &query.filter).len())
    }
}

/// Wraps an I/O error with the path that produced it.
fn file_error(err: std::io::Error, path: &Path) -> Error {
    Error::external(err).with_context(format!("document store path {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::super::contract;
    use super::*;
    use serde_json::json;

    fn store_root(dir: &tempfile::TempDir) -> PathBuf {
        dir.path().join("documents")
    }

    async fn store(dir: &tempfile::TempDir) -> LocalFileDocumentStore {
        LocalFileDocumentStore::new(store_root(dir)).await.unwrap()
    }

    #[tokio::test]
    async fn satisfies_the_shared_contract() {
        let dir = tempfile::tempdir().unwrap();
        contract::run(&store(&dir).await).await;
    }

    #[tokio::test]
    async fn documents_survive_a_reopen() {
        let dir = tempfile::tempdir().unwrap();
        {
            let store = store(&dir).await;
            store
                .upsert(
                    "docs",
                    vec![
                        Document::new("a")
                            .with_field("name", "Ada")
                            .with_field("meta", json!({"owner": "ada"})),
                        Document::new("b").with_field("tags", json!(["rust"])),
                    ],
                )
                .await
                .unwrap();
            // A replacement must also survive.
            store
                .upsert("docs", vec![Document::new("b").with_field("name", "Bob")])
                .await
                .unwrap();
        }

        let reopened = store(&dir).await;
        assert_eq!(reopened.root(), store_root(&dir).as_path());
        assert_eq!(reopened.list_collections().await.unwrap(), vec!["docs"]);

        let ada = reopened.get("docs", "a").await.unwrap().expect("a");
        assert_eq!(ada.field("name"), Some(&json!("Ada")));
        assert_eq!(ada.field("meta"), Some(&json!({"owner": "ada"})));

        let bob = reopened.get("docs", "b").await.unwrap().expect("b");
        assert_eq!(bob.field("name"), Some(&json!("Bob")));
        assert_eq!(bob.field("tags"), None);
        assert_eq!(
            reopened
                .query("docs", &Query::default())
                .await
                .unwrap()
                .len(),
            2
        );
    }

    #[tokio::test]
    async fn deletions_survive_a_reopen() {
        let dir = tempfile::tempdir().unwrap();
        {
            let store = store(&dir).await;
            store
                .upsert(
                    "docs",
                    vec![Document::new("a"), Document::new("b"), Document::new("c")],
                )
                .await
                .unwrap();
            let query = Query {
                filter: json!({"id": {"$in": ["a", "c"]}}),
                ..Query::default()
            };
            assert_eq!(store.delete("docs", &query).await.unwrap(), 2);
            store.drop_collection("dropped").await.unwrap();
        }

        let reopened = store(&dir).await;
        assert!(reopened.get("docs", "a").await.unwrap().is_none());
        assert!(reopened.get("docs", "c").await.unwrap().is_none());
        assert!(reopened.get("docs", "b").await.unwrap().is_some());
    }

    #[tokio::test]
    async fn empty_collections_persist() {
        let dir = tempfile::tempdir().unwrap();
        {
            let store = store(&dir).await;
            store.create_collection("empty").await.unwrap();
        }
        let reopened = store(&dir).await;
        assert_eq!(reopened.list_collections().await.unwrap(), vec!["empty"]);
        assert_eq!(
            reopened.query("empty", &Query::default()).await.unwrap(),
            Vec::<Document>::new()
        );
        assert!(store_root(&dir).join("empty.json").exists());
    }

    #[tokio::test]
    async fn drop_removes_the_snapshot_file() {
        let dir = tempfile::tempdir().unwrap();
        let store = store(&dir).await;
        store
            .upsert("docs", vec![Document::new("a")])
            .await
            .unwrap();
        let path = store_root(&dir).join("docs.json");
        assert!(path.exists());

        store.drop_collection("docs").await.unwrap();
        assert!(!path.exists());
        assert_eq!(
            store.list_collections().await.unwrap(),
            Vec::<String>::new()
        );
        // Dropping a collection that never existed is a no-op.
        store.drop_collection("never").await.unwrap();
    }

    #[tokio::test]
    async fn snapshot_has_the_documented_envelope() {
        let dir = tempfile::tempdir().unwrap();
        let store = store(&dir).await;
        store
            .upsert("docs", vec![Document::new("a").with_field("n", 1)])
            .await
            .unwrap();

        let raw = std::fs::read(store_root(&dir).join("docs.json")).unwrap();
        let value: serde_json::Value = serde_json::from_slice(&raw).unwrap();
        assert_eq!(
            value,
            json!({
                "collection": "docs",
                "documents": [{"id": "a", "fields": {"n": 1}}],
            })
        );
    }

    #[tokio::test]
    async fn in_flight_temp_files_are_ignored() {
        let dir = tempfile::tempdir().unwrap();
        let root = store_root(&dir);
        tokio::fs::create_dir_all(&root).await.unwrap();
        tokio::fs::write(root.join("docs.json.1234.tmp"), b"not json")
            .await
            .unwrap();
        tokio::fs::write(root.join("notes.txt"), b"not a snapshot")
            .await
            .unwrap();

        let store = LocalFileDocumentStore::new(&root).await.unwrap();
        assert_eq!(
            store.list_collections().await.unwrap(),
            Vec::<String>::new()
        );
    }

    #[tokio::test]
    async fn corrupt_snapshots_fail_the_open() {
        let dir = tempfile::tempdir().unwrap();
        let root = store_root(&dir);
        tokio::fs::create_dir_all(&root).await.unwrap();
        tokio::fs::write(root.join("broken.json"), b"{not json")
            .await
            .unwrap();
        let error = LocalFileDocumentStore::new(&root)
            .await
            .expect_err("malformed json");
        assert_eq!(error.kind(), openjiuwen_common::error::ErrorKind::External);

        // A readable envelope whose name disagrees with its file name is rejected.
        tokio::fs::remove_file(root.join("broken.json"))
            .await
            .unwrap();
        tokio::fs::write(
            root.join("declared.json"),
            br#"{"collection": "other", "documents": []}"#,
        )
        .await
        .unwrap();
        let error = LocalFileDocumentStore::new(&root)
            .await
            .expect_err("name mismatch");
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Validation
        );
    }

    #[tokio::test]
    async fn writes_stay_inside_the_root() {
        let dir = tempfile::tempdir().unwrap();
        let root = store_root(&dir);
        let store = LocalFileDocumentStore::new(&root).await.unwrap();
        store
            .upsert("docs", vec![Document::new("a")])
            .await
            .unwrap();
        store.create_collection("more").await.unwrap();

        let mut names = Vec::new();
        let mut entries = tokio::fs::read_dir(&root).await.unwrap();
        while let Some(entry) = entries.next_entry().await.unwrap() {
            names.push(entry.file_name().to_string_lossy().into_owned());
        }
        names.sort();
        assert_eq!(names, vec!["docs.json", "more.json"]);

        // The parent directory only holds the store root.
        let mut parents = Vec::new();
        let mut entries = tokio::fs::read_dir(dir.path()).await.unwrap();
        while let Some(entry) = entries.next_entry().await.unwrap() {
            parents.push(entry.file_name().to_string_lossy().into_owned());
        }
        assert_eq!(parents, vec!["documents"]);
    }
}
