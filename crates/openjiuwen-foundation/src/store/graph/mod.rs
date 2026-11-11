//! Graph storage.
//!
//! Nodes and edges as plain data (Python's `Entity`/`Relation`/`Episode` collapse
//! into [`GraphNode`]/[`GraphEdge`] with typed properties), with upsert, lookup,
//! neighbor traversal, and property filters.
//!
//! This replaces the Python `GraphStore` protocol
//! (`openjiuwen/core/foundation/store/graph/base_graph_store.py`), whose only
//! backend is Milvus and whose surface is vector-search shaped
//! (`add_entity`/`add_relation`/`query`). Vendor backends are out of scope, so
//! the contract is graph-shaped and the reference behavior lives in
//! [`InMemoryGraphStore`] and [`LocalFileGraphStore`]. Both delegate to one
//! shared engine, so their semantics cannot drift.
//!
//! # Semantics
//!
//! - **The id is the identity.** [`GraphStore::upsert_nodes`] and
//!   [`GraphStore::upsert_edges`] replace the value stored under the same id and
//!   return the number of input entries written; a repeated id in one batch
//!   collapses to its last value but still counts. A batch is validated before
//!   anything is written, so an invalid entry leaves the store unchanged.
//! - **Empty ids are rejected.** A node id, an edge id, and an edge's `from` /
//!   `to` must be non-empty; violations fail with
//!   [`openjiuwen_common::error::ErrorKind::Validation`] before any mutation.
//!   Deleting an id that does not exist is a no-op, not an error.
//! - **Edges may dangle.** Upserting an edge does not require its endpoints to
//!   exist, and [`GraphStore::neighbors`] silently skips a neighbor id that has
//!   no node (a dangling endpoint). Deleting a node removes its incident edges,
//!   which is the supported way to clean dangling edges up.
//! - **Traversal** ([`GraphStore::neighbors`]) returns the nodes adjacent to the
//!   given node, deduplicated by id and sorted by id ascending. The direction is
//!   read from the perspective of that node — `Outgoing` follows edges whose
//!   `from` equals the id, `Incoming` edges whose `to` equals it, and `Both`
//!   the union (`(from, to)` and `(to, from)`; a self-loop yields the node
//!   once). `labels`, when `Some`, keeps only edges whose **edge label** is in
//!   the list (an empty list therefore matches nothing). Neighbors of a missing
//!   node are the empty vector.
//! - **Queries** ([`GraphStore::query_nodes`]) keep nodes whose `labels` contain
//!   any of [`GraphQuery::labels`] (an empty list means "any node"), then apply
//!   [`filter::matches`](super::filter::matches) to a JSON view of the node —
//!   `{"id": ..., "labels": [...], ...properties}` — then sort by id and apply
//!   [`GraphQuery::limit`]. Properties are laid over `id`/`labels`, so a
//!   property named `id` or `labels` wins in filters; keep those names out of
//!   `properties`.
//! - **Deletion** removes the named entries and reports how many existed.
//!   [`GraphStore::delete_nodes`] also deletes every edge incident to a deleted
//!   id (including edges dangling off an id that had no node).
//! - **Size** ([`GraphStore::len`]) returns `(node_count, edge_count)`.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use openjiuwen_common::error::Result;

pub mod file;
pub mod memory;

pub use file::LocalFileGraphStore;
pub use memory::InMemoryGraphStore;

/// A graph node: an id, free-form labels, and JSON properties.
///
/// `labels` are used both as the node's classification (`Person`, `Document`,
/// ...) and by [`GraphStore::query_nodes`] for membership filtering. `properties`
/// is an insertion-ordered JSON object so a round-trip through
/// [`LocalFileGraphStore`] preserves the field order the caller wrote.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case", default)]
pub struct GraphNode {
    /// Stable identity of the node; replaces any node with the same id on upsert.
    pub id: String,
    /// Classification labels, matched with "any of" semantics by queries.
    pub labels: Vec<String>,
    /// Arbitrary JSON properties of the node.
    pub properties: Map<String, Value>,
}

/// A directed, labeled graph edge between two node ids.
///
/// Edges are keyed by `id` alone; endpoints are plain strings and are not
/// required to exist. `label` is the relationship type (`KNOWS`, `MENTIONS`,
/// ...) and is what [`GraphStore::neighbors`] filters on.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case", default)]
pub struct GraphEdge {
    /// Stable identity of the edge; replaces any edge with the same id on upsert.
    pub id: String,
    /// Id of the source node (the "out" end).
    pub from: String,
    /// Id of the target node (the "in" end).
    pub to: String,
    /// Relationship type, e.g. `KNOWS`.
    pub label: String,
    /// Arbitrary JSON properties of the edge.
    pub properties: Map<String, Value>,
}

/// Which edges of a node [`GraphStore::neighbors`] follows.
///
/// The direction is relative to the node being traversed: `Outgoing` follows
/// edges leaving it, `Incoming` edges entering it, and `Both` the union.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Direction {
    /// Follow edges whose `from` is the traversed node.
    #[default]
    Outgoing,
    /// Follow edges whose `to` is the traversed node.
    Incoming,
    /// Follow both outgoing and incoming edges.
    Both,
}

/// A node query: label membership, a property filter, and a result limit.
///
/// The filter uses the shared [filter language](super::filter): `null` or `{}`
/// matches every node, and field paths resolve inside the node's JSON view
/// (`{"id": ..., "labels": [...], ...properties}`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case", default)]
pub struct GraphQuery {
    /// Keep nodes carrying any of these labels; empty means "every label".
    pub labels: Vec<String>,
    /// Property filter applied to the node view; `null` matches everything.
    pub filter: Value,
    /// Maximum number of nodes returned after sorting by id; `None` means all.
    pub limit: Option<usize>,
}

/// A graph of nodes and directed, labeled edges.
///
/// Implementations are expected to be cheap to share: every method takes
/// `&self`, so a store can be used through an `Arc<dyn GraphStore>`. See the
/// [module docs](self) for the traversal, query, and deletion semantics every
/// implementation shares.
///
/// # Errors
///
/// Mutations reject empty node/edge ids and empty edge endpoints with
/// [`openjiuwen_common::error::ErrorKind::Validation`]; storage failures surface
/// as the underlying error. Reading or deleting an unknown id is not an error.
#[async_trait]
// `len` reports two counts, so the usual `is_empty` counterpart does not apply.
#[allow(clippy::len_without_is_empty)]
pub trait GraphStore: Send + Sync {
    /// Inserts or replaces `nodes` by id and returns how many entries were
    /// written.
    ///
    /// The whole batch is validated first, so an invalid entry leaves the store
    /// untouched.
    async fn upsert_nodes(&self, nodes: Vec<GraphNode>) -> Result<usize>;

    /// Inserts or replaces `edges` by id and returns how many entries were
    /// written.
    ///
    /// Endpoints need not exist; the batch is validated before any mutation.
    async fn upsert_edges(&self, edges: Vec<GraphEdge>) -> Result<usize>;

    /// Returns the node stored under `id`, or `None` when no node has that id.
    async fn get_node(&self, id: &str) -> Result<Option<GraphNode>>;

    /// Returns the edge stored under `id`, or `None` when no edge has that id.
    async fn get_edge(&self, id: &str) -> Result<Option<GraphEdge>>;

    /// Returns the nodes adjacent to `id`, deduplicated and sorted by id.
    ///
    /// `direction` selects outgoing/incoming/both edges. When `labels` is
    /// `Some`, only edges whose edge label is in the slice are followed (an
    /// empty slice matches no edge). A missing node yields an empty vector, as
    /// does every dangling endpoint of a followed edge.
    async fn neighbors(
        &self,
        id: &str,
        direction: Direction,
        labels: Option<&[String]>,
    ) -> Result<Vec<GraphNode>>;

    /// Returns the nodes matching `query`, sorted by id ascending.
    ///
    /// Label membership is "any of" (empty matches all), the filter runs against
    /// the node view, and `limit` is applied last.
    async fn query_nodes(&self, query: &GraphQuery) -> Result<Vec<GraphNode>>;

    /// Deletes the nodes named by `ids` and every incident edge.
    ///
    /// Returns how many nodes actually existed. Edges are removed for every
    /// requested id, even one whose node is missing (so dangling edges can be
    /// cleaned up).
    async fn delete_nodes(&self, ids: &[String]) -> Result<usize>;

    /// Deletes the edges named by `ids` and returns how many actually existed.
    async fn delete_edges(&self, ids: &[String]) -> Result<usize>;

    /// Returns `(node_count, edge_count)`.
    async fn len(&self) -> Result<(usize, usize)>;
}

// Shared behavior suite: both graph backends run it, so the in-memory and
// persistent implementations are held to exactly the same contract.
#[cfg(test)]
pub(crate) mod contract {
    use serde_json::{json, Map, Value};

    use super::{Direction, GraphEdge, GraphNode, GraphQuery, GraphStore};
    use openjiuwen_common::error::ErrorKind;

    /// A node built from JSON properties.
    fn node(id: &str, labels: &[&str], properties: Value) -> GraphNode {
        GraphNode {
            id: id.to_string(),
            labels: labels.iter().map(|label| (*label).to_string()).collect(),
            properties: object(properties),
        }
    }

    /// An edge built from JSON properties.
    fn edge(id: &str, from: &str, to: &str, label: &str, properties: Value) -> GraphEdge {
        GraphEdge {
            id: id.to_string(),
            from: from.to_string(),
            to: to.to_string(),
            label: label.to_string(),
            properties: object(properties),
        }
    }

    /// The object form of a JSON properties value.
    fn object(value: Value) -> Map<String, Value> {
        value
            .as_object()
            .cloned()
            .expect("properties are an object")
    }

    /// Ids of a node list, for order assertions.
    fn ids(nodes: &[GraphNode]) -> Vec<String> {
        nodes.iter().map(|node| node.id.clone()).collect()
    }

    /// Upserts replace by id, wholesale.
    pub(crate) async fn upsert_and_replace(store: &dyn GraphStore) {
        assert_eq!(
            store
                .upsert_nodes(vec![node("a", &["Person"], json!({"age": 30}))])
                .await
                .expect("upsert node"),
            1
        );
        let stored = store
            .get_node("a")
            .await
            .expect("get node")
            .expect("node exists");
        assert_eq!(stored.labels, vec!["Person".to_string()]);
        assert_eq!(stored.properties.get("age"), Some(&json!(30)));

        // Same id: labels and properties are replaced, not merged.
        assert_eq!(
            store
                .upsert_nodes(vec![node("a", &["Person", "Employee"], json!({"age": 31}))])
                .await
                .expect("replace node"),
            1
        );
        let replaced = store
            .get_node("a")
            .await
            .expect("get node")
            .expect("node exists");
        assert_eq!(replaced.labels.len(), 2);
        assert_eq!(replaced.properties.get("age"), Some(&json!(31)));
        assert_eq!(store.len().await.expect("len"), (1, 0));
        assert!(store.get_node("missing").await.expect("get node").is_none());
    }

    /// Edge upserts replace by id and report the written count.
    pub(crate) async fn upsert_edges_and_replace(store: &dyn GraphStore) {
        store
            .upsert_nodes(vec![
                node("a", &["Person"], json!({})),
                node("b", &["Person"], json!({})),
            ])
            .await
            .expect("upsert nodes");
        assert_eq!(
            store
                .upsert_edges(vec![edge("e1", "a", "b", "KNOWS", json!({"since": 2020}))])
                .await
                .expect("upsert edge"),
            1
        );
        let stored = store
            .get_edge("e1")
            .await
            .expect("get edge")
            .expect("edge exists");
        assert_eq!(stored.from, "a");
        assert_eq!(stored.to, "b");
        assert_eq!(stored.label, "KNOWS");

        assert_eq!(
            store
                .upsert_edges(vec![edge("e1", "b", "a", "LIKES", json!({"since": 2021}))])
                .await
                .expect("replace edge"),
            1
        );
        let replaced = store
            .get_edge("e1")
            .await
            .expect("get edge")
            .expect("edge exists");
        assert_eq!((replaced.from.as_str(), replaced.to.as_str()), ("b", "a"));
        assert_eq!(replaced.label, "LIKES");
        assert_eq!(store.len().await.expect("len"), (2, 1));
        assert!(store.get_edge("missing").await.expect("get edge").is_none());
    }

    /// Label membership is "any of"; an empty label list matches every node.
    pub(crate) async fn query_labels_any(store: &dyn GraphStore) {
        store
            .upsert_nodes(vec![
                node("a", &["Person"], json!({})),
                node("b", &["Employee", "Person"], json!({})),
                node("c", &["Document"], json!({})),
            ])
            .await
            .expect("upsert nodes");

        let people = store
            .query_nodes(&GraphQuery {
                labels: vec!["Person".to_string()],
                ..GraphQuery::default()
            })
            .await
            .expect("query");
        assert_eq!(ids(&people), vec!["a".to_string(), "b".to_string()]);

        let either = store
            .query_nodes(&GraphQuery {
                labels: vec!["Employee".to_string(), "Document".to_string()],
                ..GraphQuery::default()
            })
            .await
            .expect("query");
        assert_eq!(ids(&either), vec!["b".to_string(), "c".to_string()]);

        let all = store
            .query_nodes(&GraphQuery::default())
            .await
            .expect("query");
        assert_eq!(
            ids(&all),
            vec!["a".to_string(), "b".to_string(), "c".to_string()]
        );

        let none = store
            .query_nodes(&GraphQuery {
                labels: vec!["Unknown".to_string()],
                ..GraphQuery::default()
            })
            .await
            .expect("query");
        assert_eq!(ids(&none), Vec::<String>::new());
    }

    /// Filters run against `{id, labels, ...properties}` with the shared operators.
    pub(crate) async fn query_filter_operators(store: &dyn GraphStore) {
        store
            .upsert_nodes(vec![
                node(
                    "a",
                    &["Person"],
                    json!({"age": 30, "city": "Berlin", "meta": {"score": 7}}),
                ),
                node(
                    "b",
                    &["Person"],
                    json!({"age": 45, "city": "Paris", "meta": {"score": 3}}),
                ),
                node(
                    "c",
                    &["Person"],
                    json!({"age": 60, "city": "Berlin", "meta": {"score": 9}}),
                ),
            ])
            .await
            .expect("upsert nodes");

        let older = store
            .query_nodes(&GraphQuery {
                filter: json!({"age": {"$gt": 40}}),
                ..GraphQuery::default()
            })
            .await
            .expect("query");
        assert_eq!(ids(&older), vec!["b".to_string(), "c".to_string()]);

        let in_cities = store
            .query_nodes(&GraphQuery {
                filter: json!({"city": {"$in": ["Berlin"]}, "age": {"$lte": 30}}),
                ..GraphQuery::default()
            })
            .await
            .expect("query");
        assert_eq!(ids(&in_cities), vec!["a".to_string()]);

        let dotted = store
            .query_nodes(&GraphQuery {
                filter: json!({"meta.score": {"$gte": 7}}),
                ..GraphQuery::default()
            })
            .await
            .expect("query");
        assert_eq!(ids(&dotted), vec!["a".to_string(), "c".to_string()]);

        let by_id = store
            .query_nodes(&GraphQuery {
                filter: json!({"id": {"$in": ["b"]}}),
                ..GraphQuery::default()
            })
            .await
            .expect("query");
        assert_eq!(ids(&by_id), vec!["b".to_string()]);

        let by_label = store
            .query_nodes(&GraphQuery {
                filter: json!({"labels": {"$contains": ["Person"]}}),
                ..GraphQuery::default()
            })
            .await
            .expect("query");
        assert_eq!(ids(&by_label).len(), 3);
    }

    /// Results are sorted by id and the limit applies after sorting.
    pub(crate) async fn query_limit_and_order(store: &dyn GraphStore) {
        store
            .upsert_nodes(vec![
                node("c", &["N"], json!({})),
                node("a", &["N"], json!({})),
                node("b", &["N"], json!({})),
            ])
            .await
            .expect("upsert nodes");

        let sorted = store
            .query_nodes(&GraphQuery::default())
            .await
            .expect("query");
        assert_eq!(
            ids(&sorted),
            vec!["a".to_string(), "b".to_string(), "c".to_string()]
        );

        let limited = store
            .query_nodes(&GraphQuery {
                limit: Some(2),
                ..GraphQuery::default()
            })
            .await
            .expect("query");
        assert_eq!(ids(&limited), vec!["a".to_string(), "b".to_string()]);

        let zero = store
            .query_nodes(&GraphQuery {
                limit: Some(0),
                ..GraphQuery::default()
            })
            .await
            .expect("query");
        assert_eq!(ids(&zero), Vec::<String>::new());
    }

    /// Outgoing, incoming, and both-direction traversal, deduplicated.
    pub(crate) async fn neighbors_directions(store: &dyn GraphStore) {
        store
            .upsert_nodes(vec![
                node("a", &["N"], json!({})),
                node("b", &["N"], json!({})),
                node("c", &["N"], json!({})),
                node("d", &["N"], json!({})),
            ])
            .await
            .expect("upsert nodes");
        store
            .upsert_edges(vec![
                edge("e1", "a", "b", "KNOWS", json!({})),
                edge("e2", "c", "a", "MENTIONS", json!({})),
                edge("e3", "a", "d", "KNOWS", json!({})),
                // A self-loop must yield the node once, not twice.
                edge("e4", "a", "a", "SELF", json!({})),
            ])
            .await
            .expect("upsert edges");

        let outgoing = store
            .neighbors("a", Direction::Outgoing, None)
            .await
            .expect("outgoing");
        assert_eq!(
            ids(&outgoing),
            vec!["a".to_string(), "b".to_string(), "d".to_string()]
        );

        let incoming = store
            .neighbors("a", Direction::Incoming, None)
            .await
            .expect("incoming");
        assert_eq!(ids(&incoming), vec!["a".to_string(), "c".to_string()]);

        let both = store
            .neighbors("a", Direction::Both, None)
            .await
            .expect("both");
        assert_eq!(
            ids(&both),
            vec![
                "a".to_string(),
                "b".to_string(),
                "c".to_string(),
                "d".to_string()
            ]
        );
        // Traversal is not symmetric for directed edges.
        let b_incoming = store
            .neighbors("b", Direction::Incoming, None)
            .await
            .expect("incoming of b");
        assert_eq!(ids(&b_incoming), vec!["a".to_string()]);
        assert_eq!(
            ids(&store
                .neighbors("b", Direction::Outgoing, None)
                .await
                .expect("outgoing of b")),
            Vec::<String>::new()
        );
    }

    /// The edge-label filter keeps only matching relationship types.
    pub(crate) async fn neighbors_label_filter(store: &dyn GraphStore) {
        store
            .upsert_nodes(vec![
                node("a", &["N"], json!({})),
                node("b", &["N"], json!({})),
                node("c", &["N"], json!({})),
            ])
            .await
            .expect("upsert nodes");
        store
            .upsert_edges(vec![
                edge("e1", "a", "b", "KNOWS", json!({})),
                edge("e2", "a", "c", "MENTIONS", json!({})),
            ])
            .await
            .expect("upsert edges");

        let knows = vec!["KNOWS".to_string()];
        let only_knows = store
            .neighbors("a", Direction::Outgoing, Some(&knows))
            .await
            .expect("filtered");
        assert_eq!(ids(&only_knows), vec!["b".to_string()]);

        let both_labels = vec!["KNOWS".to_string(), "MENTIONS".to_string()];
        let both = store
            .neighbors("a", Direction::Outgoing, Some(&both_labels))
            .await
            .expect("filtered");
        assert_eq!(ids(&both), vec!["b".to_string(), "c".to_string()]);

        // An empty label slice matches no edge, unlike `None` (which matches all).
        assert_eq!(
            ids(&store
                .neighbors("a", Direction::Outgoing, Some(&[]))
                .await
                .expect("filtered")),
            Vec::<String>::new()
        );
        let unfiltered = store
            .neighbors("a", Direction::Outgoing, None)
            .await
            .expect("unfiltered");
        assert_eq!(ids(&unfiltered), vec!["b".to_string(), "c".to_string()]);
    }

    /// A missing node has no neighbors, and dangling endpoints are skipped.
    pub(crate) async fn neighbors_missing_node_and_dangling(store: &dyn GraphStore) {
        assert_eq!(
            ids(&store
                .neighbors("missing", Direction::Outgoing, None)
                .await
                .expect("missing node")),
            Vec::<String>::new()
        );

        store
            .upsert_nodes(vec![node("a", &["N"], json!({}))])
            .await
            .expect("upsert node");
        store
            .upsert_edges(vec![
                edge("e1", "a", "ghost", "KNOWS", json!({})),
                edge("e2", "ghost", "a", "KNOWS", json!({})),
            ])
            .await
            .expect("dangling edges are accepted");

        assert_eq!(
            ids(&store
                .neighbors("a", Direction::Both, None)
                .await
                .expect("dangling")),
            Vec::<String>::new()
        );
    }

    /// Deleting a node removes every incident edge in both directions.
    pub(crate) async fn delete_nodes_cascade(store: &dyn GraphStore) {
        store
            .upsert_nodes(vec![
                node("a", &["N"], json!({})),
                node("b", &["N"], json!({})),
                node("c", &["N"], json!({})),
            ])
            .await
            .expect("upsert nodes");
        store
            .upsert_edges(vec![
                edge("e1", "a", "b", "KNOWS", json!({})),
                edge("e2", "c", "a", "KNOWS", json!({})),
                edge("e3", "b", "c", "KNOWS", json!({})),
                // Incident to `a` but dangling; it must be cleaned up too.
                edge("e4", "ghost", "a", "KNOWS", json!({})),
            ])
            .await
            .expect("upsert edges");

        assert_eq!(
            store
                .delete_nodes(&["a".to_string()])
                .await
                .expect("delete"),
            1
        );
        assert!(store.get_node("a").await.expect("get node").is_none());
        assert!(store.get_edge("e1").await.expect("get edge").is_none());
        assert!(store.get_edge("e2").await.expect("get edge").is_none());
        assert!(store.get_edge("e4").await.expect("get edge").is_none());
        assert!(store.get_edge("e3").await.expect("get edge").is_some());
        assert_eq!(store.len().await.expect("len"), (2, 1));
    }

    /// Delete calls report how many entries actually existed.
    pub(crate) async fn delete_counts(store: &dyn GraphStore) {
        store
            .upsert_nodes(vec![
                node("a", &["N"], json!({})),
                node("b", &["N"], json!({})),
            ])
            .await
            .expect("upsert nodes");
        store
            .upsert_edges(vec![
                edge("e1", "a", "b", "KNOWS", json!({})),
                edge("e2", "b", "a", "KNOWS", json!({})),
            ])
            .await
            .expect("upsert edges");

        assert_eq!(
            store
                .delete_edges(&["e1".to_string(), "missing".to_string(), "e1".to_string()])
                .await
                .expect("delete edges"),
            1
        );
        assert_eq!(store.delete_edges(&[]).await.expect("delete edges"), 0);
        // A repeated id is deleted once.
        assert_eq!(
            store
                .delete_nodes(&["a".to_string(), "a".to_string()])
                .await
                .expect("delete nodes"),
            1
        );
        assert_eq!(
            store
                .delete_nodes(&["a".to_string()])
                .await
                .expect("delete nodes"),
            0
        );
        // Deleting `a` removed the remaining incident edge `e2`.
        assert_eq!(store.len().await.expect("len"), (1, 0));
    }

    /// `len` counts nodes and edges independently.
    pub(crate) async fn len_counts(store: &dyn GraphStore) {
        assert_eq!(store.len().await.expect("len"), (0, 0));
        store
            .upsert_nodes(vec![
                node("a", &["N"], json!({})),
                node("b", &["N"], json!({})),
            ])
            .await
            .expect("upsert nodes");
        assert_eq!(store.len().await.expect("len"), (2, 0));
        store
            .upsert_edges(vec![edge("e1", "a", "b", "KNOWS", json!({}))])
            .await
            .expect("upsert edge");
        assert_eq!(store.len().await.expect("len"), (2, 1));
        assert_eq!(
            store
                .upsert_nodes(vec![node("a", &["Other"], json!({"x": 1}))])
                .await
                .expect("replace node"),
            1
        );
        assert_eq!(store.len().await.expect("len"), (2, 1));
    }

    /// Empty ids are validation errors and reject the whole batch.
    pub(crate) async fn invalid_ids_rejected(store: &dyn GraphStore) {
        let error = store
            .upsert_nodes(vec![node("", &["N"], json!({}))])
            .await
            .expect_err("empty node id is rejected");
        assert_eq!(error.kind(), ErrorKind::Validation);

        let error = store
            .upsert_nodes(vec![
                node("valid", &["N"], json!({})),
                node("", &["N"], json!({})),
            ])
            .await
            .expect_err("batch with an empty id is rejected");
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert_eq!(store.len().await.expect("len"), (0, 0));

        for invalid in [
            edge("", "a", "b", "KNOWS", json!({})),
            edge("e1", "", "b", "KNOWS", json!({})),
            edge("e1", "a", "", "KNOWS", json!({})),
        ] {
            let error = store
                .upsert_edges(vec![invalid])
                .await
                .expect_err("empty edge field is rejected");
            assert_eq!(error.kind(), ErrorKind::Validation);
        }
        assert_eq!(store.len().await.expect("len"), (0, 0));
    }
}
