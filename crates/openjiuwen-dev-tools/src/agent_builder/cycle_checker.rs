//! Deterministic cycle detection over the workflow DSL graph.
//!
//! The Python builder asks a model whether a generated workflow loops and, if
//! so, for a description of the loop. The Rust port answers the same question
//! with a graph algorithm over [`WorkflowDsl`]: a three-colour depth-first
//! search that reports the cycle itself as a node-id path, so callers can feed
//! the path into a refinement prompt or a validation report without any model
//! round trip.
//!
//! Traversal order is the document order of nodes and edges, so repeated calls
//! on the same document always return the same path. Edges whose endpoints are
//! not existing nodes are ignored — structural problems are
//! [`WorkflowDsl::validate`]'s responsibility — which keeps the check total and
//! panic-free on malformed documents.

use std::collections::HashMap;

use crate::agent_builder::workflow_dsl::WorkflowDsl;

/// The depth-first-search colour of one node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Color {
    /// Not visited yet.
    White,
    /// On the current search path (its subtree is still being explored).
    Gray,
    /// Fully explored, so it cannot be part of a cycle found later.
    Black,
}

/// Returns one directed cycle as a node-id path.
///
/// The returned path starts and ends with the repeated node: `a -> b -> c -> a`
/// yields `["a", "b", "c", "a"]`, and a self-loop `a -> a` yields
/// `["a", "a"]`. When the graph has several cycles, the one whose root node
/// appears first in document order wins; neighbour edges are followed in edge
/// document order, so the result is deterministic.
///
/// Nodes and edges are examined in document order, every unvisited node is used
/// as a root, and edges naming an unknown node are ignored.
#[must_use]
pub fn find_cycle(dsl: &WorkflowDsl) -> Option<Vec<String>> {
    let adjacency = adjacency(dsl);
    let mut colors: HashMap<&str, Color> = dsl
        .nodes
        .iter()
        .map(|node| (node.id.as_str(), Color::White))
        .collect();

    for root in &dsl.nodes {
        let root_id = root.id.as_str();
        if colors.get(root_id).copied() != Some(Color::White) {
            continue;
        }

        // The explicit path stack holds exactly the gray nodes, root first;
        // `stack` pairs each of them with the position of its next unseen
        // neighbour so the search can resume after descending.
        let mut path: Vec<&str> = vec![root_id];
        let mut stack: Vec<(&str, usize)> = vec![(root_id, 0)];
        colors.insert(root_id, Color::Gray);

        while !stack.is_empty() {
            let top = stack.len() - 1;
            let (node, cursor) = stack[top];
            let neighbors = adjacency.get(node).map_or(&[][..], Vec::as_slice);

            if cursor >= neighbors.len() {
                colors.insert(node, Color::Black);
                stack.pop();
                path.pop();
                continue;
            }

            let next = neighbors[cursor];
            if let Some(entry) = stack.get_mut(top) {
                entry.1 += 1;
            }

            match colors.get(next).copied() {
                // A gray neighbour closes a loop: slice the current path from
                // that node and repeat it, giving `first ..= last, first`.
                Some(Color::Gray) => {
                    let start = path.iter().position(|id| *id == next).unwrap_or(0);
                    let mut cycle: Vec<String> =
                        path[start..].iter().map(|id| (*id).to_string()).collect();
                    cycle.push(next.to_string());
                    return Some(cycle);
                }
                // Fully explored: any cycle through it was already reported or
                // does not exist, so it can never close a new one.
                Some(Color::Black) => {}
                // Unknown endpoints were filtered out; defaulting to white also
                // keeps this arm total for out-of-contract input.
                Some(Color::White) | None => {
                    colors.insert(next, Color::Gray);
                    path.push(next);
                    stack.push((next, 0));
                }
            }
        }
    }

    None
}

/// Whether the graph contains at least one directed cycle, including a
/// self-loop.
///
/// Equivalent to `find_cycle(dsl).is_some()`.
#[must_use]
pub fn has_cycle(dsl: &WorkflowDsl) -> bool {
    find_cycle(dsl).is_some()
}

/// Builds the outgoing-edge map, dropping edges with an unknown endpoint.
///
/// Neighbours are pushed in edge document order, and nodes without out-edges
/// have no entry at all.
fn adjacency(dsl: &WorkflowDsl) -> HashMap<&str, Vec<&str>> {
    let mut adjacency: HashMap<&str, Vec<&str>> = HashMap::new();
    for edge in &dsl.edges {
        if dsl.node(&edge.source).is_none() || dsl.node(&edge.target).is_none() {
            continue;
        }
        adjacency
            .entry(edge.source.as_str())
            .or_default()
            .push(edge.target.as_str());
    }
    adjacency
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent_builder::workflow_dsl::{DslEdge, DslNode};

    /// A node with only the fields cycle detection looks at.
    fn node(id: &str) -> DslNode {
        DslNode {
            id: id.to_string(),
            node_type: "llm".to_string(),
            ..Default::default()
        }
    }

    /// An unconditional edge.
    fn edge(source: &str, target: &str) -> DslEdge {
        DslEdge {
            source: source.to_string(),
            target: target.to_string(),
            condition: None,
        }
    }

    /// A document with the given node ids (in document order) and edges.
    fn graph(nodes: &[&str], edges: &[(&str, &str)]) -> WorkflowDsl {
        WorkflowDsl {
            name: "graph".to_string(),
            description: String::new(),
            nodes: nodes.iter().map(|id| node(id)).collect(),
            edges: edges
                .iter()
                .map(|(source, target)| edge(source, target))
                .collect(),
        }
    }

    #[test]
    fn acyclic_chain_has_no_cycle() {
        let dsl = graph(
            &["start", "llm", "end"],
            &[("start", "llm"), ("llm", "end")],
        );
        dsl.validate().expect("the chain is structurally valid");
        assert_eq!(find_cycle(&dsl), None);
        assert!(!has_cycle(&dsl));
    }

    #[test]
    fn diamond_has_no_cycle() {
        let dsl = graph(
            &["a", "b", "c", "d"],
            &[("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")],
        );
        assert_eq!(find_cycle(&dsl), None);
        assert!(!has_cycle(&dsl));
    }

    #[test]
    fn self_loop_is_a_cycle() {
        let dsl = graph(&["a"], &[("a", "a")]);
        assert_eq!(
            find_cycle(&dsl),
            Some(vec!["a".to_string(), "a".to_string()])
        );
        assert!(has_cycle(&dsl));
    }

    #[test]
    fn three_node_cycle_returns_full_path() {
        let dsl = graph(&["a", "b", "c"], &[("a", "b"), ("b", "c"), ("c", "a")]);
        assert_eq!(
            find_cycle(&dsl),
            Some(vec![
                "a".to_string(),
                "b".to_string(),
                "c".to_string(),
                "a".to_string()
            ])
        );
        assert!(has_cycle(&dsl));
    }

    #[test]
    fn cycle_behind_an_isolated_first_node_is_found() {
        let dsl = graph(&["first", "x", "y"], &[("x", "y"), ("y", "x")]);
        assert_eq!(
            find_cycle(&dsl),
            Some(vec!["x".to_string(), "y".to_string(), "x".to_string()])
        );
        assert!(has_cycle(&dsl));
    }

    #[test]
    fn edges_to_unknown_nodes_are_ignored() {
        let dsl = graph(&["a", "b"], &[("a", "ghost"), ("ghost", "a"), ("a", "b")]);
        assert!(
            dsl.validate().is_err(),
            "dangling edges are invalid documents"
        );
        assert_eq!(find_cycle(&dsl), None);
        assert!(!has_cycle(&dsl));
    }

    #[test]
    fn unknown_edge_cannot_fake_a_cycle_through_a_known_node() {
        // `a -> ghost -> a` would be a cycle if the unknown hop were followed.
        let dsl = graph(&["a"], &[("a", "ghost"), ("ghost", "a")]);
        assert_eq!(find_cycle(&dsl), None);
    }

    #[test]
    fn empty_document_has_no_cycle() {
        let dsl = WorkflowDsl::default();
        assert_eq!(find_cycle(&dsl), None);
        assert!(!has_cycle(&dsl));
    }

    #[test]
    fn graph_without_edges_has_no_cycle() {
        let dsl = graph(&["a", "b", "c"], &[]);
        assert_eq!(find_cycle(&dsl), None);
    }

    #[test]
    fn repeated_calls_are_deterministic() {
        let dsl = graph(
            &["a", "b", "c", "d"],
            &[("a", "b"), ("b", "c"), ("c", "a"), ("c", "d"), ("d", "b")],
        );
        let first = find_cycle(&dsl).expect("the graph loops");
        let second = find_cycle(&dsl).expect("the graph loops");
        assert_eq!(first, second);
        assert_eq!(
            first,
            vec![
                "a".to_string(),
                "b".to_string(),
                "c".to_string(),
                "a".to_string()
            ]
        );
    }

    #[test]
    fn has_cycle_agrees_with_find_cycle() {
        let acyclic = graph(&["a", "b"], &[("a", "b")]);
        let cyclic = graph(&["a", "b"], &[("a", "b"), ("b", "a")]);
        for dsl in [acyclic, cyclic, WorkflowDsl::default()] {
            assert_eq!(has_cycle(&dsl), find_cycle(&dsl).is_some());
        }
    }
}
