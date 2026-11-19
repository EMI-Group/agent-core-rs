//! Graph snapshot model and pluggable snapshot stores.
//!
//! A snapshot captures everything needed to resume a run at a superstep
//! boundary: the graph state, the channel fired sets, and the pending message
//! buffer. [`InMemoryGraphStore`] keeps snapshots in-process;
//! [`CheckpointerStore`] maps them onto the session crate's checkpoint seam.
//!
//! Snapshots are JSON documents, not pickles, so every backend shares one
//! format.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_session::{Checkpoint, CheckpointNamespace, Checkpointer};

use crate::state::GraphState;

/// The persisted channel state of one node.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ChannelSnapshot {
    /// The node's CNF groups (each group is a set of source ids).
    pub groups: Vec<Vec<String>>,
    /// For every group, the sources that already fired.
    pub fired: Vec<Vec<String>>,
    /// For a trigger channel (no groups), the sources that fired.
    pub triggers: Vec<String>,
}

/// A routing message waiting to be flushed into its target channel.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ChannelMessage {
    /// The node to activate.
    pub target: String,
    /// The node that routed to it.
    pub source: String,
}

/// A resumable capture of a run at a superstep boundary.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct GraphSnapshot {
    /// The graph this snapshot belongs to.
    pub graph_id: String,
    /// The snapshot namespace (the graph id unless overridden at compile time).
    pub namespace: String,
    /// The number of supersteps completed when the snapshot was taken.
    pub superstep: u32,
    /// The graph state at that boundary.
    pub state: GraphState,
    /// Each node's channel state.
    pub channels: BTreeMap<String, ChannelSnapshot>,
    /// Routing messages accepted but not yet flushed (normally empty).
    pub buffer: Vec<ChannelMessage>,
}

impl GraphSnapshot {
    /// Reports whether this snapshot can be resumed: at least one superstep must
    /// have completed.
    #[must_use]
    pub fn is_resumable(&self) -> bool {
        self.superstep > 0
    }
}

/// The persistence seam for resumable graph runs.
///
/// Implementors are async and internally synchronized.
#[async_trait]
pub trait GraphStore: Send + Sync + std::fmt::Debug {
    /// Loads the snapshot of `session_id` in `namespace`, or `None` when none
    /// was saved.
    ///
    /// # Errors
    ///
    /// Returns an error when the backend cannot read or decode the snapshot.
    async fn get(&self, session_id: &str, namespace: &str) -> Result<Option<GraphSnapshot>>;

    /// Saves `snapshot` for `session_id` in `namespace`, replacing any previous
    /// snapshot.
    ///
    /// The snapshot is borrowed: the executor keeps the boundary it just saved
    /// as the resume point of a later failure, so a backend that can serialize
    /// or copy only what it stores never forces a second deep copy.
    ///
    /// # Errors
    ///
    /// Returns an error when the backend cannot write the snapshot.
    async fn save(&self, session_id: &str, namespace: &str, snapshot: &GraphSnapshot)
        -> Result<()>;

    /// Deletes the snapshot of `session_id` in `namespace`, or every snapshot of
    /// the session when `namespace` is `None`.
    ///
    /// # Errors
    ///
    /// Returns an error when the backend cannot remove the snapshot.
    async fn delete(&self, session_id: &str, namespace: Option<&str>) -> Result<()>;
}

/// An in-process, lock-guarded snapshot store.
///
/// Clones share the same storage. Intended for tests and single-process runs;
/// snapshots do not survive a restart.
#[derive(Debug, Clone, Default)]
pub struct InMemoryGraphStore {
    snapshots: Arc<Mutex<BTreeMap<(String, String), GraphSnapshot>>>,
}

impl InMemoryGraphStore {
    /// Creates an empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the number of stored snapshots.
    #[must_use]
    pub fn len(&self) -> usize {
        self.lock().len()
    }

    /// Reports whether the store holds no snapshots.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.lock().is_empty()
    }

    /// Removes every snapshot.
    pub fn clear(&self) {
        self.lock().clear();
    }

    /// Locks the map, recovering from a poisoned mutex (the map itself stays
    /// consistent because every critical section is a single map operation).
    fn lock(&self) -> std::sync::MutexGuard<'_, BTreeMap<(String, String), GraphSnapshot>> {
        self.snapshots
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

#[async_trait]
impl GraphStore for InMemoryGraphStore {
    async fn get(&self, session_id: &str, namespace: &str) -> Result<Option<GraphSnapshot>> {
        Ok(self
            .lock()
            .get(&(session_id.to_string(), namespace.to_string()))
            .cloned())
    }

    async fn save(
        &self,
        session_id: &str,
        namespace: &str,
        snapshot: &GraphSnapshot,
    ) -> Result<()> {
        self.lock().insert(
            (session_id.to_string(), namespace.to_string()),
            snapshot.clone(),
        );
        Ok(())
    }

    async fn delete(&self, session_id: &str, namespace: Option<&str>) -> Result<()> {
        let mut snapshots = self.lock();
        match namespace {
            Some(namespace) => {
                snapshots.remove(&(session_id.to_string(), namespace.to_string()));
            }
            None => {
                snapshots.retain(|(session, _), _| session != session_id);
            }
        }
        Ok(())
    }
}

/// A [`GraphStore`] backed by the session crate's [`Checkpointer`].
///
/// Each snapshot is stored as one [`Checkpoint`] in
/// [`CheckpointNamespace::WorkflowGraph`], with the namespace doubling as the
/// checkpoint id and the serialized snapshot in `global_state`. This makes graph
/// resume share the session's persisted store.
#[derive(Debug, Clone)]
pub struct CheckpointerStore {
    checkpointer: Arc<dyn Checkpointer>,
    entity_id: String,
}

impl CheckpointerStore {
    /// Wraps `checkpointer`; `entity_id` is recorded on every checkpoint.
    #[must_use]
    pub fn new(checkpointer: Arc<dyn Checkpointer>, entity_id: impl Into<String>) -> Self {
        Self {
            checkpointer,
            entity_id: entity_id.into(),
        }
    }

    /// Returns the wrapped checkpointer.
    #[must_use]
    pub fn checkpointer(&self) -> &Arc<dyn Checkpointer> {
        &self.checkpointer
    }

    /// Returns the entity id recorded on checkpoints.
    #[must_use]
    pub fn entity_id(&self) -> &str {
        &self.entity_id
    }
}

#[async_trait]
impl GraphStore for CheckpointerStore {
    async fn get(&self, session_id: &str, namespace: &str) -> Result<Option<GraphSnapshot>> {
        let checkpoint = self.checkpointer.load(session_id, namespace).await?;
        match checkpoint {
            None => Ok(None),
            Some(checkpoint) => {
                let snapshot = snapshot_from_value(checkpoint.global_state).map_err(|error| {
                    error.with_context(format!(
                        "checkpoint `{namespace}` for session `{session_id}` is not a graph snapshot"
                    ))
                })?;
                Ok(Some(snapshot))
            }
        }
    }

    async fn save(
        &self,
        session_id: &str,
        namespace: &str,
        snapshot: &GraphSnapshot,
    ) -> Result<()> {
        let global_state = serde_json::to_value(snapshot).map_err(|error| {
            Error::graph(format!("graph snapshot is not serializable: {error}"))
        })?;
        let checkpoint = Checkpoint::new(
            session_id,
            CheckpointNamespace::WorkflowGraph,
            self.entity_id.clone(),
        )
        .with_checkpoint_id(namespace)
        .with_global_state(global_state);
        self.checkpointer.save(checkpoint).await?;
        Ok(())
    }

    async fn delete(&self, session_id: &str, namespace: Option<&str>) -> Result<()> {
        match namespace {
            Some(namespace) => self.checkpointer.delete(session_id, namespace).await,
            None => self.checkpointer.release(session_id).await,
        }
    }
}

/// Deserializes a snapshot produced by a backend that stores JSON.
pub(crate) fn snapshot_from_value(value: Value) -> Result<GraphSnapshot> {
    serde_json::from_value(value)
        .map_err(|error| Error::graph(format!("invalid graph snapshot: {error}")))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]
mod tests {
    use super::*;
    use openjiuwen_session::MemoryCheckpointer;
    use serde_json::json;

    fn snapshot() -> GraphSnapshot {
        let mut channels = BTreeMap::new();
        channels.insert(
            "b".to_string(),
            ChannelSnapshot {
                groups: vec![vec!["a".to_string()]],
                fired: vec![vec!["a".to_string()]],
                triggers: Vec::new(),
            },
        );
        GraphSnapshot {
            graph_id: "g".to_string(),
            namespace: "ns".to_string(),
            superstep: 2,
            state: GraphState::from_value(json!({"count": 2})).unwrap(),
            channels,
            buffer: vec![ChannelMessage {
                target: "c".to_string(),
                source: "b".to_string(),
            }],
        }
    }

    #[test]
    fn snapshot_resumability_and_serde() {
        let snapshot = snapshot();
        assert!(snapshot.is_resumable());
        let text = serde_json::to_string(&snapshot).unwrap();
        let decoded: GraphSnapshot = serde_json::from_str(&text).unwrap();
        assert_eq!(decoded, snapshot);

        let mut fresh = GraphSnapshot::default();
        assert!(!fresh.is_resumable());
        fresh.superstep = 1;
        assert!(fresh.is_resumable());
    }

    #[tokio::test]
    async fn memory_store_round_trips_and_deletes() {
        let store = InMemoryGraphStore::new();
        assert!(store.is_empty());
        assert!(store.get("s1", "ns").await.unwrap().is_none());

        store.save("s1", "ns", &snapshot()).await.unwrap();
        store.save("s1", "other", &snapshot()).await.unwrap();
        store.save("s2", "ns", &snapshot()).await.unwrap();
        assert_eq!(store.len(), 3);

        let loaded = store.get("s1", "ns").await.unwrap().unwrap();
        assert_eq!(loaded.state.get("count"), Some(&json!(2)));

        store.delete("s1", Some("ns")).await.unwrap();
        assert!(store.get("s1", "ns").await.unwrap().is_none());
        assert!(store.get("s1", "other").await.unwrap().is_some());

        store.delete("s1", None).await.unwrap();
        assert!(store.get("s1", "other").await.unwrap().is_none());
        assert!(store.get("s2", "ns").await.unwrap().is_some());

        store.clear();
        assert!(store.is_empty());
    }

    #[tokio::test]
    async fn checkpointer_store_round_trips() {
        let checkpointer = Arc::new(MemoryCheckpointer::new());
        let store = CheckpointerStore::new(checkpointer.clone(), "workflow-1");
        assert_eq!(store.entity_id(), "workflow-1");
        assert!(store.get("s1", "ns").await.unwrap().is_none());

        store.save("s1", "ns", &snapshot()).await.unwrap();
        let checkpoint = checkpointer.load("s1", "ns").await.unwrap().unwrap();
        assert_eq!(checkpoint.namespace, CheckpointNamespace::WorkflowGraph);
        assert_eq!(checkpoint.entity_id, "workflow-1");

        let loaded = store.get("s1", "ns").await.unwrap().unwrap();
        assert_eq!(loaded, snapshot());

        store.delete("s1", Some("ns")).await.unwrap();
        assert!(store.get("s1", "ns").await.unwrap().is_none());

        store.save("s1", "ns", &snapshot()).await.unwrap();
        store.delete("s1", None).await.unwrap();
        assert!(store.get("s1", "ns").await.unwrap().is_none());
        assert!(!checkpointer.session_exists("s1").await.unwrap());
    }

    #[tokio::test]
    async fn checkpointer_store_rejects_foreign_checkpoints() {
        let checkpointer = Arc::new(MemoryCheckpointer::new());
        let store = CheckpointerStore::new(checkpointer.clone(), "workflow-1");
        let checkpoint = Checkpoint::new("s1", CheckpointNamespace::WorkflowGraph, "workflow-1")
            .with_checkpoint_id("ns")
            .with_global_state(json!("not a snapshot"));
        checkpointer.save(checkpoint).await.unwrap();
        assert!(store.get("s1", "ns").await.is_err());
    }

    #[test]
    fn helper_functions() {
        assert!(snapshot_from_value(json!("nope")).is_err());
    }
}
