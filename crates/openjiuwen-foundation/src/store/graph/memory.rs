//! In-memory graph store and the graph engine shared by both backends.
//!
//! [`GraphIndex`] holds every behavior of the [`GraphStore`] contract: id-keyed
//! replacement, traversal, filtering, and cascading deletion. It is
//! `pub(crate)` so [`LocalFileGraphStore`](super::LocalFileGraphStore) can reuse
//! it verbatim and the two backends cannot drift apart. [`InMemoryGraphStore`]
//! is the reference implementation; the file backend adds persistence only.

use std::collections::{BTreeMap, BTreeSet};

use async_trait::async_trait;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use openjiuwen_common::error::{Error, Result};

use super::{Direction, GraphEdge, GraphNode, GraphQuery, GraphStore};
use crate::store::filter::matches;

/// The persisted shape of a graph: node and edge arrays.
///
/// Kept separate from [`GraphIndex`] so the on-disk JSON is a plain
/// `{"nodes": [...], "edges": [...]}` document rather than a map keyed by id.
/// Both fields default to empty, so a partial or empty document loads as an empty
/// graph.
#[derive(Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
struct Snapshot {
    /// Every node, in id order.
    nodes: Vec<GraphNode>,
    /// Every edge, in id order.
    edges: Vec<GraphEdge>,
}

/// The graph engine: id-keyed nodes and edges with the full contract behavior.
///
/// Both maps are [`BTreeMap`]s, which is what makes every listing (queries,
/// neighbors, snapshots) sorted by id for free.
#[derive(Debug, Clone, Default)]
pub(crate) struct GraphIndex {
    /// Nodes keyed by `GraphNode::id`.
    nodes: BTreeMap<String, GraphNode>,
    /// Edges keyed by `GraphEdge::id`.
    edges: BTreeMap<String, GraphEdge>,
}

impl GraphIndex {
    /// Inserts or replaces every node, validating the whole batch first.
    ///
    /// Returns the number of input entries written; a repeated id keeps its last
    /// value but is still counted. Nothing is written when any entry is invalid.
    pub(crate) fn upsert_nodes(&mut self, nodes: Vec<GraphNode>) -> Result<usize> {
        for node in &nodes {
            validate_node(node)?;
        }
        let written = nodes.len();
        for node in nodes {
            self.nodes.insert(node.id.clone(), node);
        }
        Ok(written)
    }

    /// Inserts or replaces every edge, validating the whole batch first.
    ///
    /// Endpoints need not exist. Returns the number of input entries written,
    /// counting repeated ids like [`Self::upsert_nodes`].
    pub(crate) fn upsert_edges(&mut self, edges: Vec<GraphEdge>) -> Result<usize> {
        for edge in &edges {
            validate_edge(edge)?;
        }
        let written = edges.len();
        for edge in edges {
            self.edges.insert(edge.id.clone(), edge);
        }
        Ok(written)
    }

    /// The node stored under `id`, if any.
    pub(crate) fn get_node(&self, id: &str) -> Option<GraphNode> {
        self.nodes.get(id).cloned()
    }

    /// The edge stored under `id`, if any.
    pub(crate) fn get_edge(&self, id: &str) -> Option<GraphEdge> {
        self.edges.get(id).cloned()
    }

    /// The adjacent nodes of `id`, deduplicated and sorted by id.
    ///
    /// A missing node yields an empty vector; dangling endpoints are skipped.
    /// `labels` filters on the **edge label**, and an empty slice matches nothing.
    pub(crate) fn neighbors(
        &self,
        id: &str,
        direction: Direction,
        labels: Option<&[String]>,
    ) -> Vec<GraphNode> {
        if !self.nodes.contains_key(id) {
            return Vec::new();
        }
        let keep = |edge: &GraphEdge| match labels {
            Some(labels) => labels.iter().any(|label| label == &edge.label),
            None => true,
        };
        // A `BTreeSet` deduplicates (self-loops insert the same id twice) and
        // yields ascending ids, which is also the promised result order.
        let mut adjacent: BTreeSet<&str> = BTreeSet::new();
        for edge in self.edges.values() {
            if !keep(edge) {
                continue;
            }
            let outgoing = direction != Direction::Incoming && edge.from == id;
            let incoming = direction != Direction::Outgoing && edge.to == id;
            if outgoing {
                adjacent.insert(edge.to.as_str());
            }
            if incoming {
                adjacent.insert(edge.from.as_str());
            }
        }
        adjacent
            .into_iter()
            .filter_map(|adjacent_id| self.nodes.get(adjacent_id).cloned())
            .collect()
    }

    /// The nodes matching `query`, sorted by id with the limit applied last.
    pub(crate) fn query_nodes(&self, query: &GraphQuery) -> Vec<GraphNode> {
        let mut nodes: Vec<GraphNode> = self
            .nodes
            .values()
            .filter(|node| {
                query.labels.is_empty()
                    || node.labels.iter().any(|label| query.labels.contains(label))
            })
            .filter(|node| matches(&query.filter, &node_view(node)))
            .cloned()
            .collect();
        // `BTreeMap` order is already ascending; sorting keeps the contract
        // explicit if the storage ever changes.
        nodes.sort_by(|left, right| left.id.cmp(&right.id));
        if let Some(limit) = query.limit {
            nodes.truncate(limit);
        }
        nodes
    }

    /// Deletes the named nodes and every edge incident to a requested id.
    ///
    /// Returns how many nodes existed. Edges are removed even for a requested id
    /// that has no node, so dangling edges can be cleaned up.
    pub(crate) fn delete_nodes(&mut self, ids: &[String]) -> usize {
        let mut deleted = 0;
        for id in ids {
            if self.nodes.remove(id).is_some() {
                deleted += 1;
            }
        }
        if !ids.is_empty() {
            let removed: BTreeSet<&str> = ids.iter().map(String::as_str).collect();
            self.edges.retain(|_, edge| {
                !removed.contains(edge.from.as_str()) && !removed.contains(edge.to.as_str())
            });
        }
        deleted
    }

    /// Deletes the named edges and returns how many existed.
    pub(crate) fn delete_edges(&mut self, ids: &[String]) -> usize {
        ids.iter()
            .filter(|id| self.edges.remove(id.as_str()).is_some())
            .count()
    }

    /// The `(node_count, edge_count)` pair.
    pub(crate) fn len(&self) -> (usize, usize) {
        (self.nodes.len(), self.edges.len())
    }

    /// Encodes the graph as a JSON snapshot document.
    pub(crate) fn to_value(&self) -> Result<Value> {
        let snapshot = Snapshot {
            nodes: self.nodes.values().cloned().collect(),
            edges: self.edges.values().cloned().collect(),
        };
        serde_json::to_value(&snapshot)
            .map_err(|error| Error::external(error).with_context("encoding graph snapshot"))
    }

    /// Decodes a graph snapshot document, validating every id.
    ///
    /// Unknown fields are ignored and missing `nodes`/`edges` default to empty.
    pub(crate) fn from_value(value: Value) -> Result<Self> {
        let snapshot: Snapshot = serde_json::from_value(value)
            .map_err(|error| Error::external(error).with_context("decoding graph snapshot"))?;
        let mut index = Self::default();
        for node in snapshot.nodes {
            validate_node(&node)?;
            index.nodes.insert(node.id.clone(), node);
        }
        for edge in snapshot.edges {
            validate_edge(&edge)?;
            index.edges.insert(edge.id.clone(), edge);
        }
        Ok(index)
    }
}

/// The JSON view of a node that filters run against.
///
/// `id` and `labels` are inserted first and the properties are laid over them, so
/// a property named `id` or `labels` wins. That keeps the view faithful to the
/// caller's data; [`GraphNode`] docs advise against those property names.
fn node_view(node: &GraphNode) -> Value {
    let mut view = Map::new();
    view.insert("id".to_string(), Value::String(node.id.clone()));
    view.insert(
        "labels".to_string(),
        Value::Array(
            node.labels
                .iter()
                .cloned()
                .map(Value::String)
                .collect::<Vec<_>>(),
        ),
    );
    for (key, value) in &node.properties {
        view.insert(key.clone(), value.clone());
    }
    Value::Object(view)
}

/// Rejects a node without an id.
fn validate_node(node: &GraphNode) -> Result<()> {
    if node.id.is_empty() {
        return Err(Error::validation("graph node id must not be empty"));
    }
    Ok(())
}

/// Rejects an edge without an id or with an empty endpoint.
fn validate_edge(edge: &GraphEdge) -> Result<()> {
    if edge.id.is_empty() {
        return Err(Error::validation("graph edge id must not be empty"));
    }
    if edge.from.is_empty() {
        return Err(Error::validation(format!(
            "graph edge `{}` has an empty `from` endpoint",
            edge.id
        )));
    }
    if edge.to.is_empty() {
        return Err(Error::validation(format!(
            "graph edge `{}` has an empty `to` endpoint",
            edge.id
        )));
    }
    Ok(())
}

/// A graph held in process memory.
///
/// This is the reference [`GraphStore`] implementation and the behavior every
/// other backend must match.
///
/// ```
/// use openjiuwen_foundation::store::graph::{GraphNode, GraphStore, InMemoryGraphStore};
///
/// # #[tokio::main(flavor = "current_thread")]
/// # async fn main() -> openjiuwen_common::error::Result<()> {
/// let store = InMemoryGraphStore::new();
/// store
///     .upsert_nodes(vec![
///         GraphNode {
///             id: "a".to_string(),
///             labels: vec!["Person".to_string()],
///             ..GraphNode::default()
///         },
///         GraphNode {
///             id: "b".to_string(),
///             labels: vec!["Person".to_string()],
///             ..GraphNode::default()
///         },
///     ])
///     .await?;
/// assert_eq!(store.len().await?, (2, 0));
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Default)]
pub struct InMemoryGraphStore {
    /// The engine, behind a lock so the store is `Sync` and cheap to share.
    index: Mutex<GraphIndex>,
}

impl InMemoryGraphStore {
    /// Creates an empty graph store.
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl GraphStore for InMemoryGraphStore {
    async fn upsert_nodes(&self, nodes: Vec<GraphNode>) -> Result<usize> {
        self.index.lock().upsert_nodes(nodes)
    }

    async fn upsert_edges(&self, edges: Vec<GraphEdge>) -> Result<usize> {
        self.index.lock().upsert_edges(edges)
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
        Ok(self.index.lock().delete_nodes(ids))
    }

    async fn delete_edges(&self, ids: &[String]) -> Result<usize> {
        Ok(self.index.lock().delete_edges(ids))
    }

    async fn len(&self) -> Result<(usize, usize)> {
        Ok(self.index.lock().len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::graph::contract;

    #[tokio::test]
    async fn upserts_replace_by_id() {
        contract::upsert_and_replace(&InMemoryGraphStore::new()).await;
    }

    #[tokio::test]
    async fn edges_upsert_and_replace_by_id() {
        contract::upsert_edges_and_replace(&InMemoryGraphStore::new()).await;
    }

    #[tokio::test]
    async fn queries_match_any_label() {
        contract::query_labels_any(&InMemoryGraphStore::new()).await;
    }

    #[tokio::test]
    async fn queries_apply_filter_operators() {
        contract::query_filter_operators(&InMemoryGraphStore::new()).await;
    }

    #[tokio::test]
    async fn queries_sort_by_id_and_limit() {
        contract::query_limit_and_order(&InMemoryGraphStore::new()).await;
    }

    #[tokio::test]
    async fn neighbors_follow_directions_and_deduplicate() {
        contract::neighbors_directions(&InMemoryGraphStore::new()).await;
    }

    #[tokio::test]
    async fn neighbors_filter_on_edge_label() {
        contract::neighbors_label_filter(&InMemoryGraphStore::new()).await;
    }

    #[tokio::test]
    async fn neighbors_of_missing_node_are_empty() {
        contract::neighbors_missing_node_and_dangling(&InMemoryGraphStore::new()).await;
    }

    #[tokio::test]
    async fn deleting_a_node_cascades_to_incident_edges() {
        contract::delete_nodes_cascade(&InMemoryGraphStore::new()).await;
    }

    #[tokio::test]
    async fn deletes_report_how_many_entries_existed() {
        contract::delete_counts(&InMemoryGraphStore::new()).await;
    }

    #[tokio::test]
    async fn len_counts_nodes_and_edges() {
        contract::len_counts(&InMemoryGraphStore::new()).await;
    }

    #[tokio::test]
    async fn empty_ids_are_rejected() {
        contract::invalid_ids_rejected(&InMemoryGraphStore::new()).await;
    }

    #[test]
    fn direction_serializes_snake_case() {
        for (direction, name) in [
            (Direction::Outgoing, "outgoing"),
            (Direction::Incoming, "incoming"),
            (Direction::Both, "both"),
        ] {
            assert_eq!(
                serde_json::to_value(direction).expect("serializes"),
                serde_json::json!(name)
            );
        }
        assert_eq!(Direction::default(), Direction::Outgoing);
    }

    #[test]
    fn node_and_edge_round_trip_and_tolerate_unknown_fields() {
        let node: GraphNode = serde_json::from_str(
            r#"{"id":"a","labels":["Person"],"properties":{"age":30},"extra":true}"#,
        )
        .expect("node tolerates unknown fields");
        assert_eq!(node.id, "a");
        assert_eq!(node.labels, vec!["Person".to_string()]);
        assert_eq!(node.properties.get("age"), Some(&serde_json::json!(30)));

        let json = serde_json::to_value(&node).expect("node serializes");
        assert_eq!(json["id"], "a");
        assert_eq!(json["labels"][0], "Person");
        assert_eq!(json["properties"]["age"], 30);
        assert_eq!(
            serde_json::from_value::<GraphNode>(json).expect("node round-trips"),
            node
        );

        let edge: GraphEdge = serde_json::from_str(
            r#"{"id":"e1","from":"a","to":"b","label":"KNOWS","properties":{},"extra":1}"#,
        )
        .expect("edge tolerates unknown fields");
        assert_eq!(edge.from, "a");
        assert_eq!(edge.to, "b");
        assert_eq!(
            serde_json::from_value::<GraphEdge>(
                serde_json::to_value(&edge).expect("edge serializes")
            )
            .expect("edge round-trips"),
            edge
        );

        // Every field defaults, so a bare object is a valid (empty) node.
        let empty: GraphNode = serde_json::from_str("{}").expect("defaults apply");
        assert_eq!(empty, GraphNode::default());
    }

    #[test]
    fn query_round_trips_and_defaults() {
        let query: GraphQuery =
            serde_json::from_str(r#"{"labels":["N"]}"#).expect("query tolerates missing fields");
        assert_eq!(query.labels, vec!["N".to_string()]);
        assert_eq!(query.filter, serde_json::Value::Null);
        assert_eq!(query.limit, None);

        let full = GraphQuery {
            labels: vec!["N".to_string()],
            filter: serde_json::json!({"age": {"$gt": 1}}),
            limit: Some(5),
        };
        assert_eq!(
            serde_json::from_value::<GraphQuery>(
                serde_json::to_value(&full).expect("query serializes")
            )
            .expect("query round-trips"),
            full
        );
    }

    #[test]
    fn index_snapshot_round_trips() {
        let mut index = GraphIndex::default();
        index
            .upsert_nodes(vec![GraphNode {
                id: "a".to_string(),
                labels: vec!["Person".to_string()],
                properties: serde_json::json!({"age": 30})
                    .as_object()
                    .cloned()
                    .expect("object"),
            }])
            .expect("upsert node");
        index
            .upsert_edges(vec![GraphEdge {
                id: "e1".to_string(),
                from: "a".to_string(),
                to: "b".to_string(),
                label: "KNOWS".to_string(),
                properties: serde_json::Map::new(),
            }])
            .expect("upsert edge");

        let value = index.to_value().expect("encode");
        assert_eq!(value["nodes"][0]["id"], "a");
        assert_eq!(value["edges"][0]["id"], "e1");

        let restored = GraphIndex::from_value(value).expect("decode");
        assert_eq!(restored.len(), (1, 1));
        assert_eq!(restored.get_node("a").expect("node").labels, ["Person"]);
        assert_eq!(restored.get_edge("e1").expect("edge").from, "a");
    }

    #[test]
    fn index_snapshot_tolerates_missing_sections_and_rejects_bad_ids() {
        let index = GraphIndex::from_value(serde_json::json!({})).expect("empty snapshot");
        assert_eq!(index.len(), (0, 0));

        let error = GraphIndex::from_value(serde_json::json!({
            "nodes": [{"id": "", "labels": []}],
        }))
        .expect_err("empty node id in a snapshot is rejected");
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Validation
        );

        let error = GraphIndex::from_value(serde_json::json!({
            "edges": [{"id": "e1", "from": "a", "to": "", "label": "X"}],
        }))
        .expect_err("empty endpoint in a snapshot is rejected");
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Validation
        );
    }

    #[test]
    fn node_view_lets_properties_override_id_and_labels() {
        let node = GraphNode {
            id: "a".to_string(),
            labels: vec!["Person".to_string()],
            properties: serde_json::json!({"id": "shadow", "labels": ["Other"], "age": 1})
                .as_object()
                .cloned()
                .expect("object"),
        };
        let view = super::node_view(&node);
        assert_eq!(view["id"], "shadow");
        assert_eq!(view["labels"][0], "Other");
        assert_eq!(view["age"], 1);
    }
}
