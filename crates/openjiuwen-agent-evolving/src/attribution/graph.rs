//! The attribution dependency graph: nodes, edges and traversal algorithms.
//!
//! Ports the graph half of Python `openjiuwen.agent_evolving.attribution.nodes`
//! (`closure.py` supplies the completion rules that the deferred run loop uses)
//! without the trajectory projection: nodes are caller-supplied step records and
//! edges say "the `from` step produced the input the `to` step consumed".
//!
//! Everything here is deterministic: node order is insertion order, edge order is
//! insertion order, and the topological sort always emits the earliest-inserted
//! ready node. Cycles are legal in the stored graph (they happen in real runs);
//! only [`AttributionGraph::topological_order`] reports them.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// The default [`AttributionEdge::weight`]: a full dependency.
fn default_edge_weight() -> f64 {
    1.0
}

/// One step in the attributed run.
///
/// Python: a projected `TrajectoryNode`, reduced to what attribution needs. `kind`
/// is a free-form label (`"tool"`, `"message"`, `"subagent"`, ...).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct AttributionNode {
    /// The node identifier, unique within a graph.
    #[serde(default)]
    pub node_id: String,
    /// The display name (the tool name, the subagent display name, ...).
    #[serde(default)]
    pub name: String,
    /// The node kind label.
    #[serde(default)]
    pub kind: String,
    /// Extra host-specific payload.
    #[serde(default)]
    pub metadata: BTreeMap<String, Value>,
}

impl AttributionNode {
    /// Creates a node with no metadata.
    #[must_use]
    pub fn new(
        node_id: impl Into<String>,
        name: impl Into<String>,
        kind: impl Into<String>,
    ) -> Self {
        Self {
            node_id: node_id.into(),
            name: name.into(),
            kind: kind.into(),
            metadata: BTreeMap::new(),
        }
    }

    /// Replaces the metadata payload.
    #[must_use]
    pub fn with_metadata(mut self, metadata: BTreeMap<String, Value>) -> Self {
        self.metadata = metadata;
        self
    }
}

/// One dependency edge: `from` produced something `to` consumed.
///
/// The weight scales how much credit flows from `to` back to `from`; `0.0`
/// blocks propagation along the edge entirely.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct AttributionEdge {
    /// The producing node id.
    #[serde(default)]
    pub from: String,
    /// The consuming node id.
    #[serde(default)]
    pub to: String,
    /// The propagation weight; must be finite and non-negative.
    #[serde(default = "default_edge_weight")]
    pub weight: f64,
}

impl AttributionEdge {
    /// Creates an edge with weight `1.0`.
    #[must_use]
    pub fn new(from: impl Into<String>, to: impl Into<String>) -> Self {
        Self {
            from: from.into(),
            to: to.into(),
            weight: default_edge_weight(),
        }
    }

    /// Sets the propagation weight.
    #[must_use]
    pub fn with_weight(mut self, weight: f64) -> Self {
        self.weight = weight;
        self
    }

    /// Checks that the weight is finite and non-negative.
    pub fn validate(&self) -> Result<()> {
        if !self.weight.is_finite() || self.weight < 0.0 {
            return Err(Error::validation(format!(
                "attribution edge {} -> {} must carry a finite, non-negative weight, got {}",
                self.from, self.to, self.weight
            )));
        }
        Ok(())
    }
}

/// A directed dependency graph over [`AttributionNode`]s.
///
/// The graph is built incrementally ([`AttributionGraph::new`],
/// [`AttributionGraph::add_node`], [`AttributionGraph::add_edge`]) and every
/// mutation validates: node ids are unique, edge endpoints exist, edge weights are
/// finite and non-negative, and an ordered pair appears at most once.
///
/// # Example
///
/// ```
/// use openjiuwen_agent_evolving::attribution::{
///     AttributionEdge, AttributionGraph, AttributionNode,
/// };
///
/// let graph = AttributionGraph::new(
///     vec![
///         AttributionNode::new("root", "plan", "message"),
///         AttributionNode::new("leaf", "search", "tool"),
///     ],
///     vec![AttributionEdge::new("root", "leaf")],
/// )
/// .expect("valid graph");
/// assert_eq!(graph.roots(), ["root"]);
/// assert_eq!(graph.leaves(), ["leaf"]);
/// assert_eq!(graph.topological_order().expect("acyclic"), ["root", "leaf"]);
/// ```
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct AttributionGraph {
    /// The nodes, in insertion order.
    #[serde(default)]
    pub nodes: Vec<AttributionNode>,
    /// The edges, in insertion order.
    #[serde(default)]
    pub edges: Vec<AttributionEdge>,
}

impl AttributionGraph {
    /// Builds a graph from nodes and edges, validating every invariant.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] on a duplicate node id, an edge whose endpoints are
    /// not both nodes of the graph, an invalid edge weight, or a duplicate
    /// ordered pair.
    pub fn new(nodes: Vec<AttributionNode>, edges: Vec<AttributionEdge>) -> Result<Self> {
        let mut graph = Self::default();
        for node in nodes {
            graph.add_node(node)?;
        }
        for edge in edges {
            graph.add_edge(edge)?;
        }
        Ok(graph)
    }

    /// Appends a node.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] when the node id is already present.
    pub fn add_node(&mut self, node: AttributionNode) -> Result<()> {
        if self.contains(&node.node_id) {
            return Err(Error::validation(format!(
                "attribution graph already contains node `{}`",
                node.node_id
            )));
        }
        self.nodes.push(node);
        Ok(())
    }

    /// Appends an edge.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] when an endpoint is unknown, the weight is invalid,
    /// or the ordered pair is already present.
    pub fn add_edge(&mut self, edge: AttributionEdge) -> Result<()> {
        if !self.contains(&edge.from) {
            return Err(Error::validation(format!(
                "attribution edge references unknown node `{}`",
                edge.from
            )));
        }
        if !self.contains(&edge.to) {
            return Err(Error::validation(format!(
                "attribution edge references unknown node `{}`",
                edge.to
            )));
        }
        edge.validate()?;
        if self
            .edges
            .iter()
            .any(|existing| existing.from == edge.from && existing.to == edge.to)
        {
            return Err(Error::validation(format!(
                "attribution graph already contains edge {} -> {}",
                edge.from, edge.to
            )));
        }
        self.edges.push(edge);
        Ok(())
    }

    /// The index of a node id, when present.
    fn index_of(&self, node_id: &str) -> Option<usize> {
        self.nodes.iter().position(|node| node.node_id == node_id)
    }

    /// Looks up a node by id.
    #[must_use]
    pub fn get(&self, node_id: &str) -> Option<&AttributionNode> {
        self.nodes.iter().find(|node| node.node_id == node_id)
    }

    /// Whether the graph contains a node id.
    #[must_use]
    pub fn contains(&self, node_id: &str) -> bool {
        self.index_of(node_id).is_some()
    }

    /// The number of nodes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the graph has no nodes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// The node ids in insertion order.
    pub fn node_ids(&self) -> impl Iterator<Item = &str> {
        self.nodes.iter().map(|node| node.node_id.as_str())
    }

    /// The direct producers feeding `node_id`, in edge insertion order.
    #[must_use]
    pub fn parents(&self, node_id: &str) -> Vec<&str> {
        self.edges
            .iter()
            .filter(|edge| edge.to == node_id)
            .map(|edge| edge.from.as_str())
            .collect()
    }

    /// The direct consumers of `node_id`, in edge insertion order.
    #[must_use]
    pub fn children(&self, node_id: &str) -> Vec<&str> {
        self.edges
            .iter()
            .filter(|edge| edge.from == node_id)
            .map(|edge| edge.to.as_str())
            .collect()
    }

    /// The nodes without incoming edges, in node insertion order.
    #[must_use]
    pub fn roots(&self) -> Vec<&str> {
        self.nodes
            .iter()
            .filter(|node| self.parents(&node.node_id).is_empty())
            .map(|node| node.node_id.as_str())
            .collect()
    }

    /// The nodes without outgoing edges, in node insertion order.
    #[must_use]
    pub fn leaves(&self) -> Vec<&str> {
        self.nodes
            .iter()
            .filter(|node| self.children(&node.node_id).is_empty())
            .map(|node| node.node_id.as_str())
            .collect()
    }

    /// A deterministic topological order (Kahn's algorithm).
    ///
    /// Ready nodes are emitted earliest-inserted first, so the order depends only
    /// on insertion order.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] naming the cycle members (sorted) when the graph has
    /// a cycle.
    pub fn topological_order(&self) -> Result<Vec<&str>> {
        let count = self.nodes.len();
        let mut out: Vec<Vec<usize>> = vec![Vec::new(); count];
        let mut indegree = vec![0usize; count];
        for edge in &self.edges {
            if let (Some(from), Some(to)) = (self.index_of(&edge.from), self.index_of(&edge.to)) {
                out[from].push(to);
                indegree[to] += 1;
            }
        }
        let mut ready: BTreeSet<usize> = (0..count).filter(|index| indegree[*index] == 0).collect();
        let mut order = Vec::with_capacity(count);
        while let Some(index) = ready.pop_first() {
            order.push(self.nodes[index].node_id.as_str());
            for &to in &out[index] {
                indegree[to] -= 1;
                if indegree[to] == 0 {
                    ready.insert(to);
                }
            }
        }
        if order.len() == count {
            return Ok(order);
        }
        let remaining: Vec<usize> = (0..count).filter(|index| indegree[*index] > 0).collect();
        let members = self.cycle_members(&remaining);
        Err(Error::validation(format!(
            "attribution graph contains a cycle through nodes: {}",
            members.join(", ")
        )))
    }

    /// The sorted ids of the nodes that lie on a cycle inside `remaining`.
    fn cycle_members(&self, remaining: &[usize]) -> Vec<String> {
        let allowed: BTreeSet<usize> = remaining.iter().copied().collect();
        let mut members: BTreeSet<String> = remaining
            .iter()
            .copied()
            .filter(|start| self.reaches_itself(*start, &allowed))
            .map(|index| self.nodes[index].node_id.clone())
            .collect();
        if members.is_empty() {
            // Defensive: Kahn cannot stall without a cycle, but never report a
            // cycle without naming the nodes involved.
            members = remaining
                .iter()
                .map(|index| self.nodes[*index].node_id.clone())
                .collect();
        }
        members.into_iter().collect()
    }

    /// Whether `start` reaches itself in one or more edges inside `allowed`.
    fn reaches_itself(&self, start: usize, allowed: &BTreeSet<usize>) -> bool {
        let mut stack = self.successors(start, allowed);
        let mut seen = BTreeSet::new();
        while let Some(index) = stack.pop() {
            if index == start {
                return true;
            }
            if !seen.insert(index) {
                continue;
            }
            stack.extend(self.successors(index, allowed));
        }
        false
    }

    /// The in-`allowed` successor indices of one node.
    fn successors(&self, index: usize, allowed: &BTreeSet<usize>) -> Vec<usize> {
        let from = self.nodes[index].node_id.as_str();
        self.edges
            .iter()
            .filter(|edge| edge.from == from)
            .filter_map(|edge| self.index_of(&edge.to).filter(|to| allowed.contains(to)))
            .collect()
    }

    /// The seeds that exist plus every transitive descendant (breadth-first).
    ///
    /// Unknown seeds are ignored. A visited set makes the walk terminate on
    /// cycles.
    #[must_use]
    pub fn closure(&self, seeds: &[&str]) -> BTreeSet<String> {
        self.walk(seeds, false)
    }

    /// The seeds that exist plus every transitive ancestor (breadth-first).
    ///
    /// Unknown seeds are ignored. A visited set makes the walk terminate on
    /// cycles.
    #[must_use]
    pub fn ancestors(&self, seeds: &[&str]) -> BTreeSet<String> {
        self.walk(seeds, true)
    }

    /// The shared breadth-first walk behind [`Self::closure`] and
    /// [`Self::ancestors`].
    fn walk(&self, seeds: &[&str], upward: bool) -> BTreeSet<String> {
        let mut visited: BTreeSet<String> = BTreeSet::new();
        let mut queue: VecDeque<&str> = VecDeque::new();
        for seed in seeds {
            if self.contains(seed) && visited.insert((*seed).to_owned()) {
                queue.push_back(*seed);
            }
        }
        while let Some(current) = queue.pop_front() {
            let next = if upward {
                self.parents(current)
            } else {
                self.children(current)
            };
            for node_id in next {
                if visited.insert(node_id.to_owned()) {
                    queue.push_back(node_id);
                }
            }
        }
        visited
    }

    /// The induced subgraph over `node_ids`.
    ///
    /// Node and edge order follow this graph; an edge survives only when both
    /// endpoints are selected. Unknown ids are ignored.
    #[must_use]
    pub fn subgraph(&self, node_ids: &[&str]) -> Self {
        let selected: BTreeSet<&str> = node_ids
            .iter()
            .copied()
            .filter(|node_id| self.contains(node_id))
            .collect();
        let nodes = self
            .nodes
            .iter()
            .filter(|node| selected.contains(node.node_id.as_str()))
            .cloned()
            .collect();
        let edges = self
            .edges
            .iter()
            .filter(|edge| {
                selected.contains(edge.from.as_str()) && selected.contains(edge.to.as_str())
            })
            .cloned()
            .collect();
        Self { nodes, edges }
    }
}
