//! Serializable graph description shared by the builder, the compiled graph,
//! and consumers that need to inspect or persist a graph's shape.
//!
//! [`GraphSpec`] is data only: node runnables live in the compiled graph. Field
//! names are the JSON contract (`snake_case`), unknown fields are tolerated, and
//! missing collections default to empty.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use openjiuwen_common::error::{Error, Result};

use crate::node::NodeOptions;

/// The reserved edge target of a conditional edge whose targets are only known
/// at runtime.
pub const DYNAMIC_TARGET: &str = "*";

/// How an edge activates its target.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EdgeKind {
    /// A plain one-to-one edge.
    #[default]
    Static,
    /// A barrier edge: the target is ready only when every source fired.
    Barrier,
    /// A conditional edge driven by a [`crate::Router`].
    Conditional,
}

/// One outgoing edge of the graph spec.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct EdgeSpec {
    /// The source node ids; a barrier edge has several, the others one.
    pub sources: Vec<String>,
    /// The target node id, or [`DYNAMIC_TARGET`] for a router-only edge.
    pub target: String,
    /// How the target is activated.
    pub kind: EdgeKind,
}

/// One node of the graph spec.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct NodeSpec {
    /// The node id.
    pub id: String,
    /// A free-form label (`node`, `agent`, `llm`, …) used by visualization.
    pub kind: String,
    /// The node's execution options.
    pub options: NodeOptions,
}

impl Default for NodeSpec {
    fn default() -> Self {
        Self {
            id: String::new(),
            kind: "node".to_string(),
            options: NodeOptions::default(),
        }
    }
}

/// The serializable shape of a graph.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct GraphSpec {
    /// The graph id.
    pub graph_id: String,
    /// Explicit entry node ids, in declaration order.
    pub entry: Vec<String>,
    /// Explicit exit node ids, in declaration order.
    pub exit: Vec<String>,
    /// Every node, in declaration order.
    pub nodes: Vec<NodeSpec>,
    /// Every edge, in declaration order.
    pub edges: Vec<EdgeSpec>,
    /// Opaque graph metadata copied into every [`crate::GraphContext`].
    pub metadata: BTreeMap<String, Value>,
}

impl GraphSpec {
    /// Looks up a node by id.
    #[must_use]
    pub fn node(&self, id: &str) -> Option<&NodeSpec> {
        self.nodes.iter().find(|node| node.id == id)
    }

    /// Returns every edge whose sources contain `id`, in declaration order.
    #[must_use]
    pub fn outgoing(&self, id: &str) -> Vec<&EdgeSpec> {
        self.edges
            .iter()
            .filter(|edge| edge.sources.iter().any(|source| source == id))
            .collect()
    }

    /// Validates the spec as one [`Error::graph`] whose message lists every
    /// problem separated by `"; "`.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::ErrorKind::Graph`] when the graph is
    /// empty, has duplicate or empty node ids, has an edge with an empty or
    /// unknown endpoint, or has a missing/unknown entry.
    pub fn validate(&self) -> Result<()> {
        let problems = self.problems();
        if problems.is_empty() {
            Ok(())
        } else {
            Err(Error::graph(problems.join("; ")))
        }
    }

    /// Collects every validation problem; the builder folds these into its own
    /// compile error.
    pub(crate) fn problems(&self) -> Vec<String> {
        let mut problems = Vec::new();
        if self.nodes.is_empty() {
            problems.push(format!("graph `{}` has no nodes", self.graph_id));
        }
        let mut seen = std::collections::BTreeSet::new();
        for node in &self.nodes {
            if node.id.is_empty() {
                problems.push("graph has a node with an empty id".to_string());
            } else if !seen.insert(node.id.as_str()) {
                problems.push(format!("duplicate node id `{}`", node.id));
            }
        }
        if self.nodes.is_empty() {
            // Edge and entry checks would only restate the empty-graph problem.
            return problems;
        }
        for edge in &self.edges {
            if edge.sources.is_empty() {
                problems.push(format!("edge to `{}` has no sources", display_target(edge)));
                continue;
            }
            if edge.sources.iter().any(String::is_empty) {
                problems.push(format!(
                    "edge to `{}` has an empty source",
                    display_target(edge)
                ));
            }
            if edge.target.is_empty() {
                problems.push(format!(
                    "edge from `{}` has no target",
                    edge.sources.join(", ")
                ));
            }
            for source in &edge.sources {
                if !source.is_empty() && self.node(source).is_none() {
                    problems.push(format!("edge source `{source}` is not a node"));
                }
            }
            if !edge.target.is_empty()
                && edge.target != DYNAMIC_TARGET
                && self.node(&edge.target).is_none()
            {
                problems.push(format!("edge target `{}` is not a node", edge.target));
            }
        }
        if self.entry.is_empty() {
            problems.push("graph has no entry nodes".to_string());
        }
        for entry in &self.entry {
            if self.node(entry).is_none() {
                problems.push(format!("entry `{entry}` is not a node"));
            }
        }
        problems
    }

    /// Renders a short human-readable summary (used in error messages).
    #[must_use]
    pub fn summary(&self) -> String {
        let mut summary = String::new();
        let _ = write!(
            summary,
            "graph `{}` ({} nodes, {} edges)",
            self.graph_id,
            self.nodes.len(),
            self.edges.len()
        );
        summary
    }
}

/// The target as displayed in validation messages.
fn display_target(edge: &EdgeSpec) -> &str {
    if edge.target.is_empty() {
        "?"
    } else {
        &edge.target
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]
mod tests {
    use super::*;

    fn linear_spec() -> GraphSpec {
        GraphSpec {
            graph_id: "g".to_string(),
            entry: vec!["a".to_string()],
            exit: vec!["b".to_string()],
            nodes: vec![
                NodeSpec {
                    id: "a".to_string(),
                    ..NodeSpec::default()
                },
                NodeSpec {
                    id: "b".to_string(),
                    ..NodeSpec::default()
                },
            ],
            edges: vec![EdgeSpec {
                sources: vec!["a".to_string()],
                target: "b".to_string(),
                kind: EdgeKind::Static,
            }],
            metadata: BTreeMap::new(),
        }
    }

    #[test]
    fn node_spec_defaults_to_kind_node() {
        assert_eq!(NodeSpec::default().kind, "node");
        assert_eq!(NodeSpec::default().options, NodeOptions::default());
    }

    #[test]
    fn valid_spec_passes() {
        linear_spec().validate().unwrap();
    }

    #[test]
    fn rejects_empty_graph() {
        let error = GraphSpec::default().validate().unwrap_err();
        assert_eq!(error.kind(), openjiuwen_common::error::ErrorKind::Graph);
        assert!(error.message().contains("has no nodes"));
    }

    #[test]
    fn rejects_duplicate_nodes() {
        let mut spec = linear_spec();
        spec.nodes.push(NodeSpec {
            id: "a".to_string(),
            ..NodeSpec::default()
        });
        let error = spec.validate().unwrap_err();
        assert!(error.message().contains("duplicate node id `a`"));
    }

    #[test]
    fn rejects_unknown_edge_endpoints_and_empty_target() {
        let mut spec = linear_spec();
        spec.edges.push(EdgeSpec {
            sources: vec!["zzz".to_string()],
            target: "b".to_string(),
            kind: EdgeKind::Static,
        });
        spec.edges.push(EdgeSpec {
            sources: vec!["a".to_string()],
            target: "yyy".to_string(),
            kind: EdgeKind::Static,
        });
        spec.edges.push(EdgeSpec {
            sources: vec!["a".to_string()],
            target: String::new(),
            kind: EdgeKind::Conditional,
        });
        let error = spec.validate().unwrap_err();
        let message = error.message();
        assert!(message.contains("edge source `zzz` is not a node"));
        assert!(message.contains("edge target `yyy` is not a node"));
        assert!(message.contains("has no target"));
        assert!(message.contains("; "));
    }

    #[test]
    fn rejects_missing_and_unknown_entry() {
        let mut spec = linear_spec();
        spec.entry.clear();
        assert!(spec.validate().unwrap_err().message().contains("no entry"));
        spec.entry = vec!["zzz".to_string()];
        assert!(spec
            .validate()
            .unwrap_err()
            .message()
            .contains("entry `zzz` is not a node"));
    }

    #[test]
    fn dynamic_conditional_target_is_allowed() {
        let mut spec = linear_spec();
        spec.edges.push(EdgeSpec {
            sources: vec!["a".to_string()],
            target: DYNAMIC_TARGET.to_string(),
            kind: EdgeKind::Conditional,
        });
        spec.validate().unwrap();
    }

    #[test]
    fn outgoing_filters_by_source() {
        let spec = linear_spec();
        let edges = spec.outgoing("a");
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].target, "b");
        assert!(spec.outgoing("b").is_empty());
        assert_eq!(spec.summary(), "graph `g` (2 nodes, 1 edges)");
    }

    #[test]
    fn spec_round_trips_and_tolerates_unknown_fields() {
        let spec = linear_spec();
        let value = serde_json::to_value(&spec).unwrap();
        assert_eq!(value["nodes"][0]["kind"], "node");
        let decoded: GraphSpec = serde_json::from_value(value).unwrap();
        assert_eq!(decoded, spec);

        let partial: GraphSpec = serde_json::from_value(serde_json::json!({
            "graph_id": "g",
            "unknown_field": true
        }))
        .unwrap();
        assert!(partial.nodes.is_empty());
        assert!(partial.edges.is_empty());
        assert_eq!(partial.entry, Vec::<String>::new());
    }
}
