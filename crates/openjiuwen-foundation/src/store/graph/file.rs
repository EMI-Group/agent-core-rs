//! Local-filesystem graph store.
//!
//! The whole graph lives in one JSON snapshot at `<root>/graph.json`, shaped as
//! `{"nodes": [...], "edges": [...]}` with both arrays in ascending id order. The
//! snapshot replaces Python's Milvus-backed graph store: it is inspectable,
//! portable, and needs no server.
//!
//! Every mutation writes the complete snapshot, so a store is meant to be held by
//! one live instance per root (open it once, share it); two independent instances
//! on the same root would overwrite each other's snapshots.
//!
//! Writes are atomic (temp file then rename) and serialized by an async writer
//! lock, so concurrent mutations cannot persist a stale snapshot. All I/O goes
//! through `tokio::fs`; readers take only the short in-memory index lock and never
//! wait for the filesystem. A mutation is persisted before it becomes visible, so
//! a failed write leaves both the store and the file unchanged.

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use parking_lot::Mutex;
use serde_json::Value;
use tokio::fs;
use tokio::sync::Mutex as AsyncMutex;

use openjiuwen_common::error::{Error, Result};

use super::memory::GraphIndex;
use super::{Direction, GraphEdge, GraphNode, GraphQuery, GraphStore};

/// The snapshot filename inside the store root.
const SNAPSHOT_FILE: &str = "graph.json";

/// A [`GraphStore`] persisted as one JSON snapshot below a root directory.
///
/// Opening the store loads the snapshot (or starts empty when the file does not
/// exist yet); every mutation rewrites it atomically.
#[derive(Debug)]
pub struct LocalFileGraphStore {
    /// The caller-provided root; nothing outside it is ever touched.
    root: PathBuf,
    /// `<root>/graph.json`, the single snapshot file.
    snapshot_path: PathBuf,
    /// The live graph.
    index: Mutex<GraphIndex>,
    /// Serializes mutations (and their snapshot writes).
    writer: AsyncMutex<()>,
}

impl LocalFileGraphStore {
    /// Opens the store rooted at `root`, creating the directory (and its parents)
    /// when missing and loading `<root>/graph.json` when it exists.
    ///
    /// A snapshot that is not valid JSON, or that carries an empty id, fails with
    /// the corresponding error instead of silently starting empty.
    pub async fn new(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        fs::create_dir_all(&root)
            .await
            .map_err(|error| file_error("creating graph store root", &root, error))?;
        let snapshot_path = root.join(SNAPSHOT_FILE);
        let index = load_snapshot(&snapshot_path).await?;
        Ok(Self {
            root,
            snapshot_path,
            index: Mutex::new(index),
            writer: AsyncMutex::new(()),
        })
    }

    /// The root directory this store reads and writes below.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The snapshot file holding the graph.
    pub fn snapshot_path(&self) -> &Path {
        &self.snapshot_path
    }

    /// Applies `mutate` to a copy of the graph and persists the result.
    ///
    /// The copy is written to disk before it replaces the live index, so memory
    /// never claims a mutation the disk rejected; the writer lock keeps two
    /// concurrent mutations from overwriting each other's snapshot.
    async fn mutate<T>(&self, mutate: impl FnOnce(&mut GraphIndex) -> Result<T>) -> Result<T> {
        let _writer = self.writer.lock().await;
        let mut next = self.index.lock().clone();
        let outcome = mutate(&mut next)?;
        let snapshot = next.to_value()?;
        self.persist(&snapshot).await?;
        *self.index.lock() = next;
        Ok(outcome)
    }

    /// Writes `snapshot` to [`Self::snapshot_path`], atomically.
    async fn persist(&self, snapshot: &Value) -> Result<()> {
        let bytes = serde_json::to_vec_pretty(snapshot)
            .map_err(|error| Error::external(error).with_context("encoding graph snapshot"))?;
        let temp = self
            .root
            .join(format!("{SNAPSHOT_FILE}.{}.tmp", uuid::Uuid::new_v4()));
        fs::write(&temp, &bytes)
            .await
            .map_err(|error| file_error("writing graph snapshot", &temp, error))?;
        if let Err(error) = fs::rename(&temp, &self.snapshot_path).await {
            if let Err(cleanup) = fs::remove_file(&temp).await {
                tracing::debug!(
                    path = %temp.display(),
                    error = %cleanup,
                    "failed to remove temporary graph snapshot"
                );
            }
            return Err(file_error(
                "committing graph snapshot",
                &self.snapshot_path,
                error,
            ));
        }
        Ok(())
    }
}

#[async_trait]
impl GraphStore for LocalFileGraphStore {
    async fn upsert_nodes(&self, nodes: Vec<GraphNode>) -> Result<usize> {
        self.mutate(move |index| index.upsert_nodes(nodes)).await
    }

    async fn upsert_edges(&self, edges: Vec<GraphEdge>) -> Result<usize> {
        self.mutate(move |index| index.upsert_edges(edges)).await
    }

    async fn get_node(&self, id: &str) -> Result<Option<GraphNode>> {
        Ok(self.index.lock().get_node(id))
    }

    async fn get_edge(&self, id: &str) -> Result<Option<GraphEdge>> {
        Ok(self.index.lock().get_edge(id))
    }

    async fn neighbors(
        &self,
        id: &str,
        direction: Direction,
        labels: Option<&[String]>,
    ) -> Result<Vec<GraphNode>> {
        Ok(self.index.lock().neighbors(id, direction, labels))
    }

    async fn query_nodes(&self, query: &GraphQuery) -> Result<Vec<GraphNode>> {
        Ok(self.index.lock().query_nodes(query))
    }

    async fn delete_nodes(&self, ids: &[String]) -> Result<usize> {
        self.mutate(|index| Ok(index.delete_nodes(ids))).await
    }

    async fn delete_edges(&self, ids: &[String]) -> Result<usize> {
        self.mutate(|index| Ok(index.delete_edges(ids))).await
    }

    async fn len(&self) -> Result<(usize, usize)> {
        Ok(self.index.lock().len())
    }
}

/// Loads the snapshot at `path`, or an empty graph when the file is absent.
async fn load_snapshot(path: &Path) -> Result<GraphIndex> {
    let raw = match fs::read(path).await {
        Ok(raw) => raw,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(GraphIndex::default())
        }
        Err(error) => return Err(file_error("reading graph snapshot", path, error)),
    };
    let value: Value = serde_json::from_slice(&raw).map_err(|error| {
        Error::external(error).with_context(format!("decoding {}", path.display()))
    })?;
    GraphIndex::from_value(value)
        .map_err(|error| error.with_context(format!("loading {}", path.display())))
}

/// Wraps an I/O error with the operation and path that produced it.
fn file_error(action: &str, path: &Path, error: std::io::Error) -> Error {
    Error::external(error).with_context(format!("{action} at {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::graph::contract;

    /// Opens a fresh store below `dir`.
    async fn open(dir: &tempfile::TempDir) -> LocalFileGraphStore {
        LocalFileGraphStore::new(dir.path().join("graph"))
            .await
            .expect("open graph store")
    }

    #[tokio::test]
    async fn upserts_replace_by_id() {
        let dir = tempfile::tempdir().expect("temp dir");
        contract::upsert_and_replace(&open(&dir).await).await;
    }

    #[tokio::test]
    async fn edges_upsert_and_replace_by_id() {
        let dir = tempfile::tempdir().expect("temp dir");
        contract::upsert_edges_and_replace(&open(&dir).await).await;
    }

    #[tokio::test]
    async fn queries_match_any_label() {
        let dir = tempfile::tempdir().expect("temp dir");
        contract::query_labels_any(&open(&dir).await).await;
    }

    #[tokio::test]
    async fn queries_apply_filter_operators() {
        let dir = tempfile::tempdir().expect("temp dir");
        contract::query_filter_operators(&open(&dir).await).await;
    }

    #[tokio::test]
    async fn queries_sort_by_id_and_limit() {
        let dir = tempfile::tempdir().expect("temp dir");
        contract::query_limit_and_order(&open(&dir).await).await;
    }

    #[tokio::test]
    async fn neighbors_follow_directions_and_deduplicate() {
        let dir = tempfile::tempdir().expect("temp dir");
        contract::neighbors_directions(&open(&dir).await).await;
    }

    #[tokio::test]
    async fn neighbors_filter_on_edge_label() {
        let dir = tempfile::tempdir().expect("temp dir");
        contract::neighbors_label_filter(&open(&dir).await).await;
    }

    #[tokio::test]
    async fn neighbors_of_missing_node_are_empty() {
        let dir = tempfile::tempdir().expect("temp dir");
        contract::neighbors_missing_node_and_dangling(&open(&dir).await).await;
    }

    #[tokio::test]
    async fn deleting_a_node_cascades_to_incident_edges() {
        let dir = tempfile::tempdir().expect("temp dir");
        contract::delete_nodes_cascade(&open(&dir).await).await;
    }

    #[tokio::test]
    async fn deletes_report_how_many_entries_existed() {
        let dir = tempfile::tempdir().expect("temp dir");
        contract::delete_counts(&open(&dir).await).await;
    }

    #[tokio::test]
    async fn len_counts_nodes_and_edges() {
        let dir = tempfile::tempdir().expect("temp dir");
        contract::len_counts(&open(&dir).await).await;
    }

    #[tokio::test]
    async fn empty_ids_are_rejected() {
        let dir = tempfile::tempdir().expect("temp dir");
        contract::invalid_ids_rejected(&open(&dir).await).await;
    }

    #[tokio::test]
    async fn graph_survives_reopen() {
        let dir = tempfile::tempdir().expect("temp dir");
        {
            let store = open(&dir).await;
            store
                .upsert_nodes(vec![
                    GraphNode {
                        id: "a".to_string(),
                        labels: vec!["Person".to_string()],
                        properties: serde_json::json!({"age": 30})
                            .as_object()
                            .cloned()
                            .expect("object"),
                    },
                    GraphNode {
                        id: "b".to_string(),
                        labels: vec!["Person".to_string()],
                        ..GraphNode::default()
                    },
                ])
                .await
                .expect("upsert nodes");
            store
                .upsert_edges(vec![GraphEdge {
                    id: "e1".to_string(),
                    from: "a".to_string(),
                    to: "b".to_string(),
                    label: "KNOWS".to_string(),
                    properties: serde_json::Map::new(),
                }])
                .await
                .expect("upsert edge");
            assert_eq!(store.len().await.expect("len"), (2, 1));
        }

        let reopened = open(&dir).await;
        assert_eq!(reopened.len().await.expect("len"), (2, 1));
        let node = reopened
            .get_node("a")
            .await
            .expect("get node")
            .expect("node exists");
        assert_eq!(node.labels, vec!["Person".to_string()]);
        assert_eq!(node.properties.get("age"), Some(&serde_json::json!(30)));
        let edge = reopened
            .get_edge("e1")
            .await
            .expect("get edge")
            .expect("edge exists");
        assert_eq!((edge.from.as_str(), edge.to.as_str()), ("a", "b"));
        let neighbors = reopened
            .neighbors("a", Direction::Outgoing, None)
            .await
            .expect("neighbors");
        assert_eq!(
            neighbors
                .iter()
                .map(|node| node.id.as_str())
                .collect::<Vec<_>>(),
            vec!["b"]
        );

        // Deletions persist too.
        reopened
            .delete_nodes(&["a".to_string()])
            .await
            .expect("delete node");
        drop(reopened);
        let again = open(&dir).await;
        assert_eq!(again.len().await.expect("len"), (1, 0));
    }

    #[tokio::test]
    async fn snapshot_is_a_nodes_and_edges_document() {
        let dir = tempfile::tempdir().expect("temp dir");
        let store = open(&dir).await;
        store
            .upsert_nodes(vec![
                GraphNode {
                    id: "b".to_string(),
                    ..GraphNode::default()
                },
                GraphNode {
                    id: "a".to_string(),
                    ..GraphNode::default()
                },
            ])
            .await
            .expect("upsert nodes");

        assert_eq!(store.snapshot_path(), store.root().join(SNAPSHOT_FILE));
        let raw = fs::read(store.snapshot_path())
            .await
            .expect("read snapshot");
        let value: Value = serde_json::from_slice(&raw).expect("snapshot is JSON");
        assert_eq!(value["nodes"][0]["id"], "a");
        assert_eq!(value["nodes"][1]["id"], "b");
        assert_eq!(value["edges"], serde_json::json!([]));

        // No temporary files are left behind.
        let mut entries = fs::read_dir(store.root()).await.expect("read root");
        let mut names = Vec::new();
        while let Some(entry) = entries.next_entry().await.expect("entry") {
            names.push(entry.file_name().to_string_lossy().into_owned());
        }
        names.sort();
        assert_eq!(names, vec![SNAPSHOT_FILE.to_string()]);
    }

    #[tokio::test]
    async fn failed_mutations_leave_the_snapshot_unchanged() {
        let dir = tempfile::tempdir().expect("temp dir");
        {
            let store = open(&dir).await;
            store
                .upsert_nodes(vec![GraphNode {
                    id: "kept".to_string(),
                    ..GraphNode::default()
                }])
                .await
                .expect("upsert node");
            let error = store
                .upsert_nodes(vec![
                    GraphNode {
                        id: "staged".to_string(),
                        ..GraphNode::default()
                    },
                    GraphNode::default(),
                ])
                .await
                .expect_err("empty id is rejected");
            assert_eq!(
                error.kind(),
                openjiuwen_common::error::ErrorKind::Validation
            );
            assert_eq!(store.len().await.expect("len"), (1, 0));
        }

        let reopened = open(&dir).await;
        assert_eq!(reopened.len().await.expect("len"), (1, 0));
        assert!(reopened.get_node("staged").await.expect("get").is_none());
        assert!(reopened.get_node("kept").await.expect("get").is_some());
    }

    #[tokio::test]
    async fn reopen_starts_empty_and_creates_the_root() {
        let dir = tempfile::tempdir().expect("temp dir");
        let root = dir.path().join("nested").join("graph");
        let store = LocalFileGraphStore::new(&root).await.expect("open store");
        assert_eq!(store.root(), root.as_path());
        assert_eq!(store.len().await.expect("len"), (0, 0));
        assert!(!store.snapshot_path().exists());

        let reopened = LocalFileGraphStore::new(&root).await.expect("reopen");
        assert_eq!(reopened.len().await.expect("len"), (0, 0));
    }

    #[tokio::test]
    async fn a_corrupt_snapshot_fails_to_open() {
        let dir = tempfile::tempdir().expect("temp dir");
        let root = dir.path().join("graph");
        fs::create_dir_all(&root).await.expect("create root");
        fs::write(root.join(SNAPSHOT_FILE), b"{not json")
            .await
            .expect("write corrupt snapshot");

        let error = LocalFileGraphStore::new(&root)
            .await
            .expect_err("corrupt snapshot is rejected");
        assert_eq!(error.kind(), openjiuwen_common::error::ErrorKind::External);
    }
}
