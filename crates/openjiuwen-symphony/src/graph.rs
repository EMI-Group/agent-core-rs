//! The `can_feed` capability relationship graph and its JGF v2 document form.
//!
//! Ports the deterministic half of Python `openjiuwen.symphony.orchestration.graph`:
//! a [`CapabilityGraph`] holds one [`CapabilityNode`] per capability and
//! [`CapabilityEdge`]s describing directed relations between them. The relation the
//! planner consumes is [`CAN_FEED`], derived from the data-type vocabulary.
//!
//! # Determinism
//!
//! Nodes live in a [`BTreeMap`], so iteration is always sorted by node id. Edges keep
//! their insertion order in `edges` (an upsert replaces in place) but every read —
//! [`CapabilityGraph::outgoing`], [`CapabilityGraph::incoming`], and the JSON produced
//! by [`CapabilityGraph::to_value`] — sorts by `(source, target, relation)`. No output
//! depends on hash order or on the clock.
//!
//! # Errors
//!
//! Malformed input is rejected with [`crate::error::validation`], naming the offending
//! field: [`CapabilityGraph::add_edge`] names `edge.source`, `edge.target`,
//! `edge.relation`, or `edge.weight`, and [`CapabilityGraph::from_value`] names the
//! document path (for example `graph.nodes[2]`).

use std::collections::{BTreeMap, BTreeSet};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::error::{validation, Result};
use crate::model::CapabilityDescriptor;

/// The relation name of a data-flow edge, i.e. "the source output can feed the target
/// input".
pub const CAN_FEED: &str = "can_feed";

/// The `graph_type` of a capability relationship graph.
const GRAPH_TYPE: &str = "capability_graph";

/// The weight of a fully confident [`CAN_FEED`] edge built by
/// [`CapabilityEdge::can_feed`].
const CAN_FEED_WEIGHT: f64 = 1.0;

/// One capability inside a [`CapabilityGraph`].
///
/// `id` is the capability's qualified id (`"{capability_type}:{capability_id}"`), and
/// `node_type` is its capability type, so a JGF document is self-describing. Unknown
/// JSON fields are ignored and every field defaults on deserialization.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct CapabilityNode {
    /// The node id, equal to the capability's qualified id.
    pub id: String,
    /// The human-readable label, equal to the capability name.
    pub label: String,
    /// The node kind, equal to the capability type (`tool`, `workflow`, ...).
    pub node_type: String,
    /// Free-form node metadata (`capability_id`, `capability_type`, ...).
    pub metadata: BTreeMap<String, Value>,
}

impl CapabilityNode {
    /// Builds a node with an id, label, and node type; metadata starts empty.
    #[must_use]
    pub fn new(
        id: impl Into<String>,
        label: impl Into<String>,
        node_type: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            node_type: node_type.into(),
            metadata: BTreeMap::new(),
        }
    }

    /// Inserts one metadata entry.
    #[must_use]
    pub fn with_metadata(mut self, key: impl Into<String>, value: Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }
}

/// One directed, weighted relation between two [`CapabilityNode`]s.
///
/// The identity of an edge is the triple `(source, target, relation)`: adding an edge
/// with the same triple replaces the existing one in place. Unknown JSON fields are
/// ignored and every field defaults on deserialization.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct CapabilityEdge {
    /// The source node id.
    pub source: String,
    /// The target node id.
    pub target: String,
    /// The relation name; see [`CAN_FEED`].
    pub relation: String,
    /// The relation weight in `[0, 1]`.
    pub weight: f64,
    /// Free-form edge metadata (matching strategy, evidence, ...).
    pub metadata: BTreeMap<String, Value>,
}

impl CapabilityEdge {
    /// Builds an edge with a source, target, relation, and weight.
    ///
    /// The weight is not validated here; [`CapabilityGraph::add_edge`] validates it
    /// when the edge is inserted.
    #[must_use]
    pub fn new(
        source: impl Into<String>,
        target: impl Into<String>,
        relation: impl Into<String>,
        weight: f64,
    ) -> Self {
        Self {
            source: source.into(),
            target: target.into(),
            relation: relation.into(),
            weight,
            metadata: BTreeMap::new(),
        }
    }

    /// Builds a full-confidence [`CAN_FEED`] edge from `source` to `target`.
    #[must_use]
    pub fn can_feed(source: impl Into<String>, target: impl Into<String>) -> Self {
        Self::new(source, target, CAN_FEED, CAN_FEED_WEIGHT)
    }

    /// Inserts one metadata entry.
    #[must_use]
    pub fn with_metadata(mut self, key: impl Into<String>, value: Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }
}

/// The directed capability relationship graph.
///
/// Nodes are keyed by id in a [`BTreeMap`]; edges are kept in a `Vec` so an upsert can
/// preserve insertion order. See the module docs for the determinism guarantees.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct CapabilityGraph {
    /// The graph identifier.
    pub graph_id: String,
    /// The graph kind; always `"capability_graph"` for graphs built here.
    pub graph_type: String,
    /// Whether edges are directed; always `true` for graphs built here.
    pub directed: bool,
    /// The nodes, keyed by node id (iteration is sorted by id).
    pub nodes: BTreeMap<String, CapabilityNode>,
    /// The edges, in insertion order (upserts replace in place).
    pub edges: Vec<CapabilityEdge>,
    /// Free-form graph metadata.
    pub metadata: BTreeMap<String, Value>,
}

impl Default for CapabilityGraph {
    /// Returns an empty graph with the canonical type and directed edges.
    fn default() -> Self {
        Self {
            graph_id: String::new(),
            graph_type: GRAPH_TYPE.to_string(),
            directed: true,
            nodes: BTreeMap::new(),
            edges: Vec::new(),
            metadata: BTreeMap::new(),
        }
    }
}

impl CapabilityGraph {
    /// Builds an empty directed graph with the canonical graph type.
    #[must_use]
    pub fn new(graph_id: impl Into<String>) -> Self {
        Self {
            graph_id: graph_id.into(),
            ..Self::default()
        }
    }

    /// Builds a graph with one node per descriptor and no edges.
    ///
    /// Each node id is [`CapabilityDescriptor::qualified_id`], its label is the
    /// capability name, its type is the capability type, and its metadata carries
    /// `capability_id` and `capability_type`. Duplicate qualified ids collapse (the last
    /// descriptor wins).
    #[must_use]
    pub fn from_descriptors(
        graph_id: impl Into<String>,
        capabilities: &[CapabilityDescriptor],
    ) -> Self {
        let mut graph = Self::new(graph_id);
        for descriptor in capabilities {
            let node = CapabilityNode {
                id: descriptor.qualified_id(),
                label: descriptor.name.clone(),
                node_type: descriptor.capability_type.clone(),
                metadata: BTreeMap::from([
                    (
                        "capability_id".to_string(),
                        Value::String(descriptor.capability_id.clone()),
                    ),
                    (
                        "capability_type".to_string(),
                        Value::String(descriptor.capability_type.clone()),
                    ),
                ]),
            };
            graph.upsert_node(node);
        }
        graph
    }

    /// Returns the node with `id`, if present.
    #[must_use]
    pub fn node(&self, id: &str) -> Option<&CapabilityNode> {
        self.nodes.get(id)
    }

    /// Reports whether the graph contains a node with `id`.
    #[must_use]
    pub fn has_node(&self, id: &str) -> bool {
        self.nodes.contains_key(id)
    }

    /// Inserts `node`, replacing any node with the same id.
    ///
    /// Edges pointing at the replaced node are kept: replacement is a metadata update,
    /// not a removal.
    pub fn upsert_node(&mut self, node: CapabilityNode) {
        self.nodes.insert(node.id.clone(), node);
    }

    /// Removes node `id` together with every incident edge.
    ///
    /// Returns whether the node itself existed. Dangling edges that mention `id` are
    /// removed even when the node does not exist.
    pub fn remove_node(&mut self, id: &str) -> bool {
        let removed = self.nodes.remove(id).is_some();
        self.remove_incident_edges(id);
        removed
    }

    /// Adds `edge`, validating it and upserting by `(source, target, relation)`.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::Error::Validation`] when the source or target node is
    /// missing, the edge is a self-loop, the relation is blank, or the weight is
    /// non-finite or outside `[0, 1]`. The message names the offending field.
    pub fn add_edge(&mut self, edge: CapabilityEdge) -> Result<()> {
        if edge.source.trim().is_empty() {
            return Err(validation("edge.source", "must not be blank"));
        }
        if !self.has_node(&edge.source) {
            return Err(validation(
                "edge.source",
                format!("node `{}` is not in the graph", edge.source),
            ));
        }
        if edge.target.trim().is_empty() {
            return Err(validation("edge.target", "must not be blank"));
        }
        if !self.has_node(&edge.target) {
            return Err(validation(
                "edge.target",
                format!("node `{}` is not in the graph", edge.target),
            ));
        }
        if edge.source == edge.target {
            return Err(validation(
                "edge.target",
                format!(
                    "must differ from the source `{}` (self-loops are not allowed)",
                    edge.source
                ),
            ));
        }
        if edge.relation.trim().is_empty() {
            return Err(validation("edge.relation", "must not be blank"));
        }
        if !edge.weight.is_finite() {
            return Err(validation("edge.weight", "must be finite"));
        }
        if !(0.0..=1.0).contains(&edge.weight) {
            return Err(validation(
                "edge.weight",
                format!("must be within [0, 1], got {}", edge.weight),
            ));
        }
        match self.edge_position(&edge.source, &edge.target, &edge.relation) {
            Some(index) => self.edges[index] = edge,
            None => self.edges.push(edge),
        }
        Ok(())
    }

    /// Removes every edge between `source` and `target`, whatever its relation.
    ///
    /// Returns whether at least one edge was removed.
    pub fn remove_edge(&mut self, source: &str, target: &str) -> bool {
        let before = self.edges.len();
        self.edges
            .retain(|edge| !(edge.source == source && edge.target == target));
        self.edges.len() != before
    }

    /// Removes every edge incident to `id` and returns how many were removed.
    pub fn remove_incident_edges(&mut self, id: &str) -> usize {
        let before = self.edges.len();
        self.edges
            .retain(|edge| edge.source != id && edge.target != id);
        before - self.edges.len()
    }

    /// The outgoing edges of `id`, sorted by `(source, target, relation)`.
    #[must_use]
    pub fn outgoing(&self, id: &str) -> Vec<&CapabilityEdge> {
        self.sorted_edges(|edge| edge.source == id)
    }

    /// The incoming edges of `id`, sorted by `(source, target, relation)`.
    #[must_use]
    pub fn incoming(&self, id: &str) -> Vec<&CapabilityEdge> {
        self.sorted_edges(|edge| edge.target == id)
    }

    /// The ids of nodes with no incoming edge, sorted.
    #[must_use]
    pub fn roots(&self) -> Vec<String> {
        let targets: BTreeSet<&str> = self.edges.iter().map(|edge| edge.target.as_str()).collect();
        self.nodes
            .keys()
            .filter(|id| !targets.contains(id.as_str()))
            .cloned()
            .collect()
    }

    /// The ids of nodes with no outgoing edge, sorted.
    #[must_use]
    pub fn leaves(&self) -> Vec<String> {
        let sources: BTreeSet<&str> = self.edges.iter().map(|edge| edge.source.as_str()).collect();
        self.nodes
            .keys()
            .filter(|id| !sources.contains(id.as_str()))
            .cloned()
            .collect()
    }

    /// The number of nodes.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// The number of edges.
    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Renders the graph as a JGF v2 document: `{"graph": {...}}`.
    ///
    /// Nodes are sorted by id and edges by `(source, target, relation)`, so the output
    /// never depends on insertion or hash order.
    #[must_use]
    pub fn to_value(&self) -> Value {
        let nodes: Vec<Value> = self.nodes.values().map(node_to_value).collect();
        let mut edges: Vec<&CapabilityEdge> = self.edges.iter().collect();
        edges.sort_by(|left, right| edge_sort_key(left).cmp(&edge_sort_key(right)));
        let edges: Vec<Value> = edges.iter().map(|edge| edge_to_value(edge)).collect();

        let mut graph = Map::new();
        graph.insert("id".to_string(), Value::String(self.graph_id.clone()));
        graph.insert("type".to_string(), Value::String(self.graph_type.clone()));
        graph.insert("directed".to_string(), Value::Bool(self.directed));
        graph.insert("nodes".to_string(), Value::Array(nodes));
        graph.insert("edges".to_string(), Value::Array(edges));
        graph.insert(
            "metadata".to_string(),
            Value::Object(self.metadata.clone().into_iter().collect()),
        );

        let mut root = Map::new();
        root.insert("graph".to_string(), Value::Object(graph));
        Value::Object(root)
    }

    /// Parses a JGF v2 document produced by [`Self::to_value`].
    ///
    /// Unknown members are ignored and every node/edge field is default-tolerant; a
    /// malformed document (a missing or non-object `graph` member, or a non-object
    /// node/edge or wrongly typed field) is a validation error naming the path.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::Error::Validation`] for a structurally invalid document.
    pub fn from_value(value: &Value) -> Result<Self> {
        let root = value.as_object().ok_or_else(|| {
            validation(
                "graph",
                format!("expected an object, got {}", json_kind(value)),
            )
        })?;
        let graph = root
            .get("graph")
            .ok_or_else(|| validation("graph", "missing the `graph` member"))?;
        let graph = graph.as_object().ok_or_else(|| {
            validation(
                "graph",
                format!(
                    "expected the `graph` member to be an object, got {}",
                    json_kind(graph)
                ),
            )
        })?;

        let graph_id = string_member(graph, "id", "graph")?;
        let parsed_type = string_member(graph, "type", "graph")?;
        let graph_type = if parsed_type.is_empty() {
            GRAPH_TYPE.to_string()
        } else {
            parsed_type
        };
        let directed = bool_member(graph, "directed", "graph")?.unwrap_or(true);
        let metadata = metadata_member(graph, "metadata", "graph")?;
        let nodes = parse_nodes(graph)?;
        let edges = parse_edges(graph)?;

        Ok(Self {
            graph_id,
            graph_type,
            directed,
            nodes,
            edges,
            metadata,
        })
    }

    /// Returns the index of the edge with this identity, if any.
    fn edge_position(&self, source: &str, target: &str, relation: &str) -> Option<usize> {
        self.edges.iter().position(|edge| {
            edge.source == source && edge.target == target && edge.relation == relation
        })
    }

    /// Collects edges matching `predicate`, sorted by `(source, target, relation)`.
    fn sorted_edges(&self, predicate: impl Fn(&CapabilityEdge) -> bool) -> Vec<&CapabilityEdge> {
        let mut edges: Vec<&CapabilityEdge> =
            self.edges.iter().filter(|edge| predicate(edge)).collect();
        edges.sort_by(|left, right| edge_sort_key(left).cmp(&edge_sort_key(right)));
        edges
    }
}

/// The deterministic sort key of an edge.
fn edge_sort_key(edge: &CapabilityEdge) -> (&str, &str, &str) {
    (
        edge.source.as_str(),
        edge.target.as_str(),
        edge.relation.as_str(),
    )
}

/// Renders one node as its JGF member object.
fn node_to_value(node: &CapabilityNode) -> Value {
    let mut map = Map::new();
    map.insert("id".to_string(), Value::String(node.id.clone()));
    map.insert("label".to_string(), Value::String(node.label.clone()));
    map.insert(
        "node_type".to_string(),
        Value::String(node.node_type.clone()),
    );
    map.insert(
        "metadata".to_string(),
        Value::Object(node.metadata.clone().into_iter().collect()),
    );
    Value::Object(map)
}

/// Renders one edge as its JGF member object.
fn edge_to_value(edge: &CapabilityEdge) -> Value {
    let mut map = Map::new();
    map.insert("source".to_string(), Value::String(edge.source.clone()));
    map.insert("target".to_string(), Value::String(edge.target.clone()));
    map.insert("relation".to_string(), Value::String(edge.relation.clone()));
    map.insert(
        "weight".to_string(),
        Value::Number(
            serde_json::Number::from_f64(edge.weight).unwrap_or_else(|| {
                // Non-finite weights are rejected by `add_edge`; a graph assembled by hand
                // (or deserialized from JSON, where non-finite numbers cannot occur) can
                // still carry one, so fall back to zero instead of panicking.
                serde_json::Number::from(0)
            }),
        ),
    );
    map.insert(
        "metadata".to_string(),
        Value::Object(edge.metadata.clone().into_iter().collect()),
    );
    Value::Object(map)
}

/// Parses the `nodes` array of a `graph` object.
fn parse_nodes(graph: &Map<String, Value>) -> Result<BTreeMap<String, CapabilityNode>> {
    let mut nodes = BTreeMap::new();
    let Some(raw) = graph.get("nodes") else {
        return Ok(nodes);
    };
    if raw.is_null() {
        return Ok(nodes);
    }
    let list = raw.as_array().ok_or_else(|| {
        validation(
            "graph.nodes",
            format!("expected an array, got {}", json_kind(raw)),
        )
    })?;
    for (index, item) in list.iter().enumerate() {
        let path = format!("graph.nodes[{index}]");
        let map = item.as_object().ok_or_else(|| {
            validation(
                &path,
                format!("expected an object, got {}", json_kind(item)),
            )
        })?;
        let id = string_member(map, "id", &path)?;
        let label = string_member(map, "label", &path)?;
        let node_type = string_member(map, "node_type", &path)?;
        let metadata = metadata_member(map, "metadata", &path)?;
        nodes.insert(
            id.clone(),
            CapabilityNode {
                id,
                label,
                node_type,
                metadata,
            },
        );
    }
    Ok(nodes)
}

/// Parses the `edges` array of a `graph` object, preserving array order.
fn parse_edges(graph: &Map<String, Value>) -> Result<Vec<CapabilityEdge>> {
    let mut edges = Vec::new();
    let Some(raw) = graph.get("edges") else {
        return Ok(edges);
    };
    if raw.is_null() {
        return Ok(edges);
    }
    let list = raw.as_array().ok_or_else(|| {
        validation(
            "graph.edges",
            format!("expected an array, got {}", json_kind(raw)),
        )
    })?;
    for (index, item) in list.iter().enumerate() {
        let path = format!("graph.edges[{index}]");
        let map = item.as_object().ok_or_else(|| {
            validation(
                &path,
                format!("expected an object, got {}", json_kind(item)),
            )
        })?;
        let source = string_member(map, "source", &path)?;
        let target = string_member(map, "target", &path)?;
        let relation = string_member(map, "relation", &path)?;
        let weight = number_member(map, "weight", &path)?.unwrap_or(0.0);
        let metadata = metadata_member(map, "metadata", &path)?;
        edges.push(CapabilityEdge {
            source,
            target,
            relation,
            weight,
            metadata,
        });
    }
    Ok(edges)
}

/// Reads an optional string member, treating `null` as absent.
fn string_member(map: &Map<String, Value>, field: &str, path: &str) -> Result<String> {
    match map.get(field) {
        None | Some(Value::Null) => Ok(String::new()),
        Some(Value::String(value)) => Ok(value.clone()),
        Some(other) => Err(validation(
            &format!("{path}.{field}"),
            format!("expected a string, got {}", json_kind(other)),
        )),
    }
}

/// Reads an optional boolean member, treating `null` as absent.
fn bool_member(map: &Map<String, Value>, field: &str, path: &str) -> Result<Option<bool>> {
    match map.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Bool(value)) => Ok(Some(*value)),
        Some(other) => Err(validation(
            &format!("{path}.{field}"),
            format!("expected a boolean, got {}", json_kind(other)),
        )),
    }
}

/// Reads an optional number member, treating `null` as absent.
fn number_member(map: &Map<String, Value>, field: &str, path: &str) -> Result<Option<f64>> {
    match map.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => value.as_f64().map(Some).ok_or_else(|| {
            validation(
                &format!("{path}.{field}"),
                format!("expected a number, got {}", json_kind(value)),
            )
        }),
    }
}

/// Reads an optional object member, treating `null` as absent (an empty map).
fn metadata_member(
    map: &Map<String, Value>,
    field: &str,
    path: &str,
) -> Result<BTreeMap<String, Value>> {
    match map.get(field) {
        None | Some(Value::Null) => Ok(BTreeMap::new()),
        Some(Value::Object(entries)) => Ok(entries
            .iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect()),
        Some(other) => Err(validation(
            &format!("{path}.{field}"),
            format!("expected an object, got {}", json_kind(other)),
        )),
    }
}

/// A short human-readable name for a JSON value's kind, used in error messages.
fn json_kind(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "an array",
        Value::Object(_) => "an object",
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn edge_sort_keys_order_by_source_target_relation() {
        let edge = CapabilityEdge::new("b", "a", CAN_FEED, 0.5);
        assert_eq!(edge_sort_key(&edge), ("b", "a", CAN_FEED));
    }

    #[test]
    fn json_kind_names_every_json_kind() {
        assert_eq!(json_kind(&Value::Null), "null");
        assert_eq!(json_kind(&Value::Bool(true)), "a boolean");
        assert_eq!(json_kind(&Value::from(1)), "a number");
        assert_eq!(json_kind(&Value::from("x")), "a string");
        assert_eq!(json_kind(&Value::Array(vec![])), "an array");
        assert_eq!(json_kind(&Value::Object(Map::new())), "an object");
    }

    #[test]
    fn membership_helpers_default_on_absent_and_null() {
        let mut map = Map::new();
        map.insert("present".to_string(), Value::String("x".to_string()));
        map.insert("null".to_string(), Value::Null);
        assert_eq!(string_member(&map, "present", "p").unwrap(), "x");
        assert_eq!(string_member(&map, "absent", "p").unwrap(), "");
        assert_eq!(string_member(&map, "null", "p").unwrap(), "");
        assert_eq!(bool_member(&map, "absent", "p").unwrap(), None);
        assert_eq!(number_member(&map, "absent", "p").unwrap(), None);
        assert!(metadata_member(&map, "absent", "p").unwrap().is_empty());
    }
}
