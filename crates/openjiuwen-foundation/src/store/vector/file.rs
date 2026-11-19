//! Local-filesystem vector store.
//!
//! One JSON snapshot per collection lives below `<root>/vector/`, named
//! `<collection>.json`. Every mutation rewrites the affected collection's
//! snapshot (write to a temporary file, then rename), so a store reopened on the
//! same root sees exactly the committed state.
//!
//! Only `tokio::fs` is used: no blocking filesystem call runs on the async path.
//! Collection names are validated before they become file names, so a store can
//! never read or write outside its root.
//!
//! Unlike the object store, the vector store loads its snapshots once in
//! [`LocalFileVectorStore::new`] and keeps the index in memory; two stores on the
//! same root therefore do not observe each other's writes until one is reopened.
//! The store rebuilds no indexes on load — the persisted `f32` embeddings are
//! used as-is.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use async_trait::async_trait;
use parking_lot::Mutex;
use serde_json::{Map, Value};
use tokio::fs;
use tokio::io::AsyncWriteExt;
use tokio::sync::Mutex as AsyncMutex;

use openjiuwen_common::error::{Error, Result};

use super::memory::{CollectionData, VectorIndex};
use super::{CollectionSchema, Metric, VectorQuery, VectorSearchResult, VectorStore};

/// Name of the directory holding the collection snapshots, below the root.
const SNAPSHOT_DIR: &str = "vector";
/// Extension of every collection snapshot.
const SNAPSHOT_EXTENSION: &str = "json";
/// Disambiguates the temporary files of concurrent writers within one process.
static TEMPORARY_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// A [`VectorStore`] that persists one JSON snapshot per collection.
///
/// The store keeps the shared `VectorIndex` in memory and rewrites the
/// snapshot of the collection a mutation touched, which makes behavior identical
/// to [`super::InMemoryVectorStore`] while surviving a reopen.
#[derive(Debug)]
pub struct LocalFileVectorStore {
    root: PathBuf,
    index: Mutex<VectorIndex>,
    /// Serializes writers so a mutation and the snapshot that reports it as
    /// committed are one unit: an insert that returned must be on disk.
    writes: AsyncMutex<()>,
}

impl LocalFileVectorStore {
    /// Opens the store rooted at `root`, creating `<root>/vector` when missing
    /// and loading every collection snapshot found there.
    pub async fn new(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        let directory = root.join(SNAPSHOT_DIR);
        fs::create_dir_all(&directory)
            .await
            .map_err(|error| io_error("creating vector snapshot directory", &directory, error))?;

        let mut index = VectorIndex::new();
        let mut entries = fs::read_dir(&directory)
            .await
            .map_err(|error| io_error("reading vector snapshot directory", &directory, error))?;
        while let Some(entry) = entries
            .next_entry()
            .await
            .map_err(|error| io_error("reading vector snapshot entry", &directory, error))?
        {
            let file_type = entry
                .file_type()
                .await
                .map_err(|error| io_error("reading vector snapshot entry", &entry.path(), error))?;
            let path = entry.path();
            if !file_type.is_file()
                || path.extension().and_then(|ext| ext.to_str()) != Some(SNAPSHOT_EXTENSION)
            {
                continue;
            }
            let Some(name) = path.file_stem().and_then(|stem| stem.to_str()) else {
                return Err(Error::validation(format!(
                    "vector snapshot file name is not valid UTF-8: {}",
                    path.display()
                )));
            };
            validate_collection(name)?;
            let text = fs::read_to_string(&path)
                .await
                .map_err(|error| io_error("reading vector snapshot", &path, error))?;
            let data: CollectionData = serde_json::from_str(&text).map_err(|error| {
                Error::external(error)
                    .with_context(format!("parsing vector snapshot {}", path.display()))
            })?;
            index.restore_collection(name.to_string(), data)?;
        }

        Ok(Self {
            root,
            index: Mutex::new(index),
            writes: AsyncMutex::new(()),
        })
    }

    /// The root directory this store reads and writes below.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Path of a collection's snapshot file.
    fn snapshot_path(&self, collection: &str) -> PathBuf {
        self.root
            .join(SNAPSHOT_DIR)
            .join(format!("{collection}.{SNAPSHOT_EXTENSION}"))
    }

    /// Path of a unique temporary snapshot file for `collection`.
    ///
    /// The name carries the process id and a sequence number, so two writers
    /// (in this process or another) never share a temporary file. The `.tmp`
    /// extension keeps [`LocalFileVectorStore::new`] from loading it.
    fn temporary_path(&self, collection: &str) -> PathBuf {
        let sequence = TEMPORARY_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        self.root.join(SNAPSHOT_DIR).join(format!(
            ".{collection}.{}.{sequence}.{SNAPSHOT_EXTENSION}.tmp",
            std::process::id()
        ))
    }

    /// Writes the snapshot of `collection`, replacing the previous file atomically.
    ///
    /// The bytes are written to a unique temporary file, flushed to the device
    /// (`fsync`), and only then renamed over the target. A concurrent
    /// [`LocalFileVectorStore`] therefore sees either the old or the new
    /// snapshot, never a partial file; the directory entry is synced on a
    /// best-effort basis so the rename itself survives a crash.
    async fn write_snapshot(&self, collection: &str, snapshot: &Value) -> Result<()> {
        let target = self.snapshot_path(collection);
        let temporary = self.temporary_path(collection);
        let text = serde_json::to_vec_pretty(snapshot).map_err(Error::external)?;
        if let Err(error) = Self::write_and_sync(&temporary, &text).await {
            let _ = fs::remove_file(&temporary).await;
            return Err(error);
        }
        if let Err(error) = fs::rename(&temporary, &target).await {
            let _ = fs::remove_file(&temporary).await;
            return Err(io_error("replacing vector snapshot", &target, error));
        }
        if let Some(directory) = target.parent() {
            if let Ok(handle) = fs::File::open(directory).await {
                let _ = handle.sync_all().await;
            }
        }
        Ok(())
    }

    /// Writes `text` to `path` and flushes it to the device.
    async fn write_and_sync(path: &Path, text: &[u8]) -> Result<()> {
        let mut file = fs::File::create(path)
            .await
            .map_err(|error| io_error("writing vector snapshot", path, error))?;
        file.write_all(text)
            .await
            .map_err(|error| io_error("writing vector snapshot", path, error))?;
        file.sync_all()
            .await
            .map_err(|error| io_error("flushing vector snapshot", path, error))
    }

    /// Persists the current state of `collection`; a missing collection is a no-op.
    async fn persist(&self, collection: &str) -> Result<()> {
        let snapshot = {
            let index = self.index.lock();
            index.collection_value(collection)?
        };
        match snapshot {
            Some(snapshot) => self.write_snapshot(collection, &snapshot).await,
            None => Ok(()),
        }
    }

    /// Searches a collection with an explicit similarity metric.
    ///
    /// This is the metric-aware variant of [`VectorStore::search`], which always
    /// uses [`Metric::Cosine`]. It is synchronous because the index lives in
    /// memory; callers never await I/O.
    pub fn search_with_metric(
        &self,
        collection: &str,
        query: &VectorQuery,
        metric: Metric,
    ) -> Result<Vec<VectorSearchResult>> {
        validate_collection(collection)?;
        self.index.lock().search(collection, query, metric)
    }
}

#[async_trait]
impl VectorStore for LocalFileVectorStore {
    async fn create_collection(&self, collection: &str, schema: CollectionSchema) -> Result<()> {
        validate_collection(collection)?;
        let _guard = self.writes.lock().await;
        self.index.lock().create_collection(collection, schema)?;
        self.persist(collection).await
    }

    async fn delete_collection(&self, collection: &str) -> Result<()> {
        validate_collection(collection)?;
        let _guard = self.writes.lock().await;
        self.index.lock().delete_collection(collection);
        let path = self.snapshot_path(collection);
        match fs::remove_file(&path).await {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(io_error("deleting vector snapshot", &path, error)),
        }
    }

    async fn collection_exists(&self, collection: &str) -> Result<bool> {
        validate_collection(collection)?;
        Ok(self.index.lock().collection_exists(collection))
    }

    async fn get_schema(&self, collection: &str) -> Result<Option<CollectionSchema>> {
        validate_collection(collection)?;
        Ok(self.index.lock().schema(collection).cloned())
    }

    async fn add_docs(&self, collection: &str, docs: Vec<Map<String, Value>>) -> Result<usize> {
        validate_collection(collection)?;
        let _guard = self.writes.lock().await;
        let inserted = self.index.lock().add_docs(collection, docs)?;
        if inserted > 0 {
            self.persist(collection).await?;
        }
        Ok(inserted)
    }

    async fn search(
        &self,
        collection: &str,
        query: &VectorQuery,
    ) -> Result<Vec<VectorSearchResult>> {
        self.search_with_metric(collection, query, Metric::Cosine)
    }

    async fn delete_docs_by_ids(&self, collection: &str, ids: &[String]) -> Result<usize> {
        validate_collection(collection)?;
        let _guard = self.writes.lock().await;
        let deleted = self.index.lock().delete_docs_by_ids(collection, ids)?;
        if deleted > 0 {
            self.persist(collection).await?;
        }
        Ok(deleted)
    }

    async fn delete_docs_by_filters(&self, collection: &str, filters: &Value) -> Result<usize> {
        validate_collection(collection)?;
        let _guard = self.writes.lock().await;
        let deleted = self
            .index
            .lock()
            .delete_docs_by_filters(collection, filters)?;
        if deleted > 0 {
            self.persist(collection).await?;
        }
        Ok(deleted)
    }

    async fn list_collection_names(&self) -> Result<Vec<String>> {
        Ok(self.index.lock().collection_names())
    }
}

/// Validates a collection name: a single, non-empty, file-name-safe path segment.
fn validate_collection(collection: &str) -> Result<()> {
    if collection.is_empty() {
        return Err(Error::validation(
            "vector collection name must not be empty",
        ));
    }
    if collection == "." || collection == ".." {
        return Err(Error::validation(format!(
            "invalid vector collection name {collection:?}: must not be a relative path segment"
        )));
    }
    if !collection
        .chars()
        .all(|value| value.is_ascii_alphanumeric() || matches!(value, '-' | '_' | '.'))
    {
        return Err(Error::validation(format!(
            "invalid vector collection name {collection:?}: only ASCII letters, digits, '-', '_' and '.' are allowed"
        )));
    }
    Ok(())
}

/// Wraps an I/O error with the operation and path that produced it.
fn io_error(action: &str, path: &Path, error: std::io::Error) -> Error {
    Error::external(error).with_context(format!("{action} {}", path.display()))
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use tempfile::TempDir;

    use openjiuwen_common::error::ErrorKind;

    use super::*;
    use crate::store::vector::{FieldSchema, InMemoryVectorStore, VectorDataType};

    fn field(name: &str, dtype: VectorDataType) -> FieldSchema {
        FieldSchema {
            name: name.to_string(),
            dtype,
            ..FieldSchema::default()
        }
    }

    fn schema() -> CollectionSchema {
        CollectionSchema::new(vec![
            FieldSchema {
                is_primary: true,
                ..field("id", VectorDataType::Varchar)
            },
            FieldSchema {
                dim: Some(2),
                ..field("embedding", VectorDataType::FloatVector)
            },
            field("kind", VectorDataType::Varchar),
        ])
    }

    /// The JSON object of `value`, for the document helpers below.
    fn obj(value: Value) -> Map<String, Value> {
        value.as_object().cloned().unwrap_or_default()
    }

    fn doc(id: &str, embedding: [f32; 2], kind: &str) -> Map<String, Value> {
        obj(json!({"id": id, "embedding": [embedding[0], embedding[1]], "kind": kind}))
    }

    fn ids(hits: &[VectorSearchResult]) -> Vec<String> {
        hits.iter()
            .map(|hit| hit.fields["id"].as_str().unwrap_or_default().to_string())
            .collect()
    }

    /// Asserts that `result` failed with `expected`.
    fn assert_kind<T: std::fmt::Debug>(result: Result<T>, expected: ErrorKind) {
        assert_eq!(result.unwrap_err().kind(), expected);
    }

    async fn store_in(root: &Path) -> LocalFileVectorStore {
        LocalFileVectorStore::new(root)
            .await
            .expect("open vector store")
    }

    /// The behavior every [`VectorStore`] backend must share.
    async fn check_contract(store: &dyn VectorStore) {
        check_collection_lifecycle(store).await;
        check_insert_only(store).await;
        check_search_and_deletes(store).await;
    }

    /// Collection creation, lookup, listing, and idempotent deletion.
    async fn check_collection_lifecycle(store: &dyn VectorStore) {
        store
            .create_collection("docs", schema())
            .await
            .expect("create");
        assert!(store.collection_exists("docs").await.expect("exists"));
        assert_eq!(
            store.get_schema("docs").await.expect("schema"),
            Some(schema())
        );
        assert_eq!(
            store.list_collection_names().await.expect("list"),
            ["docs".to_string()]
        );
        assert_kind(
            store.create_collection("docs", schema()).await,
            ErrorKind::Validation,
        );

        // Missing collections are reported, not invented.
        assert!(!store.collection_exists("missing").await.expect("exists"));
        assert_eq!(store.get_schema("missing").await.expect("schema"), None);

        store.delete_collection("docs").await.expect("delete");
        store.delete_collection("docs").await.expect("delete again");
        assert_eq!(
            store.list_collection_names().await.expect("list"),
            Vec::<String>::new()
        );
    }

    /// Missing collections and insert-only adds.
    async fn check_insert_only(store: &dyn VectorStore) {
        store
            .create_collection("docs", schema())
            .await
            .expect("create");

        // Missing collections fail every document operation.
        assert_kind(
            store.add_docs("missing", Vec::new()).await,
            ErrorKind::NotFound,
        );
        assert_kind(
            store
                .search("missing", &VectorQuery::new(vec![1.0, 0.0]))
                .await,
            ErrorKind::NotFound,
        );
        assert_kind(
            store.delete_docs_by_ids("missing", &[]).await,
            ErrorKind::NotFound,
        );
        assert_kind(
            store.delete_docs_by_filters("missing", &json!({})).await,
            ErrorKind::NotFound,
        );

        // Insert-only adds, including a duplicate inside one batch.
        assert_eq!(
            store
                .add_docs(
                    "docs",
                    vec![
                        doc("a", [1.0, 0.0], "x"),
                        doc("b", [0.0, 1.0], "y"),
                        doc("c", [0.8, 0.2], "x"),
                    ],
                )
                .await
                .expect("add"),
            3
        );
        assert_kind(
            store
                .add_docs("docs", vec![doc("a", [1.0, 0.0], "x")])
                .await,
            ErrorKind::Validation,
        );
        assert_kind(
            store
                .add_docs(
                    "docs",
                    vec![doc("dup", [1.0, 0.0], "x"), doc("dup", [0.0, 1.0], "y")],
                )
                .await,
            ErrorKind::Validation,
        );
        // The rejected batch inserted nothing.
        assert_eq!(
            ids(&store
                .search("docs", &VectorQuery::new(vec![1.0, 0.0]).with_top_k(10))
                .await
                .expect("search")),
            ["a", "c", "b"]
        );
    }

    /// Search ordering, scores, `top_k`, filters, and deletes.
    async fn check_search_and_deletes(store: &dyn VectorStore) {
        let hits = store
            .search("docs", &VectorQuery::new(vec![1.0, 0.0]).with_top_k(10))
            .await
            .expect("search");
        assert_eq!(ids(&hits), ["a", "c", "b"]);
        // Ordering, scores, `top_k`, and filters.
        assert!(hits.windows(2).all(|pair| pair[0].score >= pair[1].score));
        assert!(hits.iter().all(|hit| hit.score.is_finite()));
        assert!(hits.iter().all(|hit| (0.0..=1.0).contains(&hit.score)));
        assert!(
            (hits[0].score - 1.0).abs() < 1e-6,
            "identical vector is 1.0"
        );
        assert_eq!(hits[0].fields["embedding"], json!([1.0, 0.0]));
        let top = store
            .search("docs", &VectorQuery::new(vec![1.0, 0.0]).with_top_k(1))
            .await
            .expect("search");
        assert_eq!(ids(&top), ["a"]);
        let empty = store
            .search("docs", &VectorQuery::new(vec![1.0, 0.0]).with_top_k(0))
            .await
            .expect("search");
        assert_eq!(empty, Vec::<VectorSearchResult>::new());
        let filtered = store
            .search(
                "docs",
                &VectorQuery::new(vec![1.0, 0.0]).with_filters(json!({"kind": "x"})),
            )
            .await
            .expect("search");
        assert_eq!(ids(&filtered), ["a", "c"]);
        let none = store
            .search(
                "docs",
                &VectorQuery::new(vec![1.0, 0.0]).with_filters(json!({"kind": "none"})),
            )
            .await
            .expect("search");
        assert_eq!(none, Vec::<VectorSearchResult>::new());

        // Deletes.
        assert_eq!(
            store
                .delete_docs_by_ids("docs", &["a".to_string(), "missing".to_string()])
                .await
                .expect("delete ids"),
            1
        );
        assert_eq!(
            store
                .delete_docs_by_filters("docs", &json!({"kind": "x"}))
                .await
                .expect("delete filters"),
            1
        );
        assert_eq!(
            ids(&store
                .search("docs", &VectorQuery::new(vec![1.0, 0.0]).with_top_k(10))
                .await
                .expect("search")),
            ["b"]
        );
    }

    #[tokio::test]
    async fn in_memory_store_satisfies_contract() {
        check_contract(&InMemoryVectorStore::new()).await;
    }

    #[tokio::test]
    async fn local_file_store_satisfies_contract() {
        let temp = TempDir::new().expect("temp dir");
        check_contract(&store_in(temp.path()).await).await;
    }

    #[tokio::test]
    async fn new_creates_snapshot_directory_and_is_idempotent() {
        let temp = TempDir::new().expect("temp dir");
        let root = temp.path().join("nested/vectors");
        let store = store_in(&root).await;
        assert_eq!(store.root(), root.as_path());
        assert!(root.join(SNAPSHOT_DIR).is_dir());
        assert_eq!(
            store.list_collection_names().await.expect("list"),
            Vec::<String>::new()
        );

        store
            .create_collection("docs", schema())
            .await
            .expect("create");
        assert!(root.join("vector/docs.json").is_file());
        let reopened = store_in(&root).await;
        assert_eq!(
            reopened.list_collection_names().await.expect("list"),
            ["docs".to_string()]
        );
    }

    #[tokio::test]
    async fn snapshots_persist_across_reopen() {
        let temp = TempDir::new().expect("temp dir");
        {
            let store = store_in(temp.path()).await;
            store
                .create_collection("first", schema())
                .await
                .expect("create first");
            store
                .create_collection("second", schema())
                .await
                .expect("create second");
            store
                .add_docs(
                    "first",
                    vec![doc("a", [1.0, 0.0], "x"), doc("b", [0.5, 0.5], "y")],
                )
                .await
                .expect("add first");
            store
                .add_docs("second", vec![doc("z", [0.0, 1.0], "z")])
                .await
                .expect("add second");
            assert_eq!(
                store
                    .delete_docs_by_ids("second", &["z".to_string()])
                    .await
                    .expect("delete"),
                1
            );
            store
                .add_docs("second", vec![doc("y", [0.25, 0.75], "y")])
                .await
                .expect("add second again");
        }

        let reopened = store_in(temp.path()).await;
        assert_eq!(
            reopened.list_collection_names().await.expect("list"),
            ["first".to_string(), "second".to_string()]
        );
        assert_eq!(
            reopened.get_schema("first").await.expect("schema"),
            Some(schema())
        );
        let hits = reopened
            .search("first", &VectorQuery::new(vec![1.0, 0.0]).with_top_k(10))
            .await
            .expect("search");
        assert_eq!(ids(&hits), ["a", "b"]);
        assert_eq!(hits[0].fields["embedding"], json!([1.0, 0.0]));
        assert_eq!(hits[0].fields["kind"], "x");
        assert_eq!(
            ids(&reopened
                .search("second", &VectorQuery::new(vec![0.0, 1.0]).with_top_k(10))
                .await
                .expect("search second")),
            ["y"],
            "the reopened store sees the latest snapshot"
        );

        // Deletions are durable as well.
        reopened
            .delete_docs_by_ids("first", &["a".to_string()])
            .await
            .expect("delete");
        drop(reopened);
        let reopened = store_in(temp.path()).await;
        assert_eq!(
            ids(&reopened
                .search("first", &VectorQuery::new(vec![1.0, 0.0]).with_top_k(10))
                .await
                .expect("search")),
            ["b"]
        );
    }

    #[tokio::test]
    async fn concurrent_inserts_all_survive_a_reopen() {
        let temp = TempDir::new().expect("temp dir");
        let store = std::sync::Arc::new(store_in(temp.path()).await);
        store
            .create_collection("docs", schema())
            .await
            .expect("create");

        let mut tasks = Vec::new();
        for index in 0..16 {
            let store = std::sync::Arc::clone(&store);
            tasks.push(tokio::spawn(async move {
                store
                    .add_docs(
                        "docs",
                        vec![doc(&format!("id-{index:02}"), [1.0, 0.0], "x")],
                    )
                    .await
            }));
        }
        for task in tasks {
            assert_eq!(task.await.expect("join").expect("add"), 1);
        }
        drop(store);

        // Every insert that returned a success is on disk.
        let reopened = store_in(temp.path()).await;
        let hits = reopened
            .search("docs", &VectorQuery::new(vec![1.0, 0.0]).with_top_k(100))
            .await
            .expect("search");
        assert_eq!(hits.len(), 16, "an acknowledged insert vanished on reopen");
        let mut found = ids(&hits);
        found.sort();
        let mut expected: Vec<String> = (0..16).map(|index| format!("id-{index:02}")).collect();
        expected.sort();
        assert_eq!(found, expected);

        // The atomic-replace temporary files were cleaned up.
        let strays: Vec<String> = std::fs::read_dir(temp.path().join(SNAPSHOT_DIR))
            .expect("read snapshot dir")
            .filter_map(Result::ok)
            .filter_map(|entry| entry.file_name().into_string().ok())
            .filter(|name| {
                std::path::Path::new(name)
                    .extension()
                    .is_some_and(|extension| extension.eq_ignore_ascii_case("tmp"))
            })
            .collect();
        assert!(strays.is_empty(), "unexpected temporary files: {strays:?}");
    }

    #[tokio::test]
    async fn delete_collection_removes_snapshot_and_is_idempotent() {
        let temp = TempDir::new().expect("temp dir");
        let store = store_in(temp.path()).await;
        store
            .create_collection("docs", schema())
            .await
            .expect("create");
        let path = temp.path().join("vector/docs.json");
        assert!(path.is_file());
        store.delete_collection("docs").await.expect("delete");
        assert!(!path.exists());
        assert!(!store.collection_exists("docs").await.expect("exists"));
        store.delete_collection("docs").await.expect("delete again");
        drop(store);
        assert_eq!(
            store_in(temp.path())
                .await
                .list_collection_names()
                .await
                .expect("list"),
            Vec::<String>::new()
        );
    }

    #[tokio::test]
    async fn failed_calls_are_not_persisted() {
        let temp = TempDir::new().expect("temp dir");
        let store = store_in(temp.path()).await;
        assert_kind(
            store
                .create_collection("bad", CollectionSchema::default())
                .await,
            ErrorKind::Validation,
        );
        store
            .create_collection("docs", schema())
            .await
            .expect("create");
        store
            .add_docs("docs", vec![doc("a", [1.0, 0.0], "x")])
            .await
            .expect("add");
        assert_kind(
            store
                .add_docs("docs", vec![doc("a", [0.0, 1.0], "y")])
                .await,
            ErrorKind::Validation,
        );
        assert_kind(
            store
                .add_docs(
                    "docs",
                    vec![obj(json!({"id": "z", "embedding": [1.0], "kind": "x"}))],
                )
                .await,
            ErrorKind::Validation,
        );

        let reopened = store_in(temp.path()).await;
        assert_eq!(
            reopened.list_collection_names().await.expect("list"),
            ["docs".to_string()]
        );
        assert_eq!(
            ids(&reopened
                .search("docs", &VectorQuery::new(vec![1.0, 0.0]).with_top_k(10))
                .await
                .expect("search")),
            ["a"]
        );
    }

    #[tokio::test]
    async fn invalid_collection_names_are_rejected() {
        let temp = TempDir::new().expect("temp dir");
        let store = store_in(temp.path()).await;
        for name in [
            "",
            ".",
            "..",
            "a/b",
            "../b",
            "back\\slash",
            "a\0b",
            "sp ace",
        ] {
            assert_kind(
                store.create_collection(name, schema()).await,
                ErrorKind::Validation,
            );
            assert!(store.delete_collection(name).await.is_err());
            assert!(store.collection_exists(name).await.is_err());
            assert!(store.get_schema(name).await.is_err());
            assert!(store
                .add_docs(name, vec![doc("a", [1.0, 0.0], "x")])
                .await
                .is_err());
            assert!(store
                .search(name, &VectorQuery::new(vec![1.0, 0.0]))
                .await
                .is_err());
            assert!(store
                .delete_docs_by_ids(name, &["a".to_string()])
                .await
                .is_err());
            assert!(store
                .delete_docs_by_filters(name, &json!({}))
                .await
                .is_err());
        }
        // Nothing escaped the snapshot directory.
        assert_eq!(
            store.list_collection_names().await.expect("list"),
            Vec::<String>::new()
        );
        let stray = std::fs::read_dir(temp.path().join(SNAPSHOT_DIR))
            .expect("read snapshot dir")
            .filter_map(Result::ok)
            .filter_map(|entry| entry.file_name().into_string().ok())
            .filter(|name| name != "docs.json")
            .collect::<Vec<_>>();
        assert!(stray.is_empty(), "unexpected files: {stray:?}");
    }

    #[tokio::test]
    async fn search_with_metric_supports_every_metric() {
        let temp = TempDir::new().expect("temp dir");
        let store = store_in(temp.path()).await;
        store
            .create_collection("docs", schema())
            .await
            .expect("create");
        store
            .add_docs(
                "docs",
                vec![doc("near", [1.0, 0.0], "x"), doc("far", [-1.0, 0.0], "y")],
            )
            .await
            .expect("add");
        let query = VectorQuery::new(vec![1.0, 0.0]);
        for metric in [Metric::Cosine, Metric::Dot, Metric::Euclidean] {
            let hits = store
                .search_with_metric("docs", &query, metric)
                .expect("search");
            assert_eq!(ids(&hits), ["near", "far"], "{metric:?}");
            assert!(hits[0].score > hits[1].score, "{metric:?}");
        }
    }

    #[tokio::test]
    async fn corrupt_snapshot_fails_to_open() {
        let temp = TempDir::new().expect("temp dir");
        let directory = temp.path().join(SNAPSHOT_DIR);
        std::fs::create_dir_all(&directory).expect("create snapshot dir");
        std::fs::write(directory.join("docs.json"), b"{not json").expect("write snapshot");
        let error = LocalFileVectorStore::new(temp.path())
            .await
            .expect_err("corrupt snapshot");
        assert_eq!(error.kind(), ErrorKind::External);
        assert!(error.to_string().contains("docs.json"), "{error}");

        let invalid = json!({"schema": {"fields": []}, "docs": {}});
        std::fs::write(
            directory.join("docs.json"),
            serde_json::to_vec(&invalid).expect("serialize"),
        )
        .expect("write invalid snapshot");
        assert_eq!(
            LocalFileVectorStore::new(temp.path())
                .await
                .expect_err("invalid schema")
                .kind(),
            ErrorKind::Validation
        );
    }
}
