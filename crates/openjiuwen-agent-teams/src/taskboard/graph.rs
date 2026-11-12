//! Pure dependency-graph helpers: cycle detection and subtask depth.
//!
//! The task board owns the graph *state* (edges live in [`crate::storage::TeamStore`]); this module
//! owns the two decisions that need no I/O — "would this edge set contain a cycle?" and "how deep
//! does this subtask nest?" — so both are testable without a store and identical for every caller.
//!
//! Python parity source: `openjiuwen/agent_teams/tools/database/graph.py`
//! (`mutate_dependency_graph`, which rejects a mutation whose post-state has a cycle) and
//! `TeamTaskManager.task_depth` (F_128 subtask nesting).

use std::collections::{BTreeMap, BTreeSet};

/// Depth-first mark of one node in [`detect_cycle`].
#[derive(Clone, Copy, PartialEq, Eq)]
enum Mark {
    /// On the current traversal path.
    Active,
    /// Fully explored; cannot reach a cycle on its own.
    Done,
}

/// Returns one dependency cycle in `edges`, or `None` when the graph is acyclic.
///
/// `edges` maps a task id to the ids it depends on, so a cycle means "A needs B, B needs A" — a
/// deadlock no completion can ever resolve. The returned path starts and ends at the same id and
/// runs in dependency order: `["a", "b", "a"]` reads "a depends on b, which depends on a".
///
/// The result is deterministic: nodes are traversed in sorted order and the reported path is
/// rotated so it starts at its smallest id, which keeps the rendered
/// `Circular dependency: a -> b -> a` message stable across runs and store backends. Ids that
/// appear only as a dependency (no outgoing edge of their own) cannot participate in a cycle but
/// are traversed anyway so a partially materialized graph is judged on its full node set.
#[must_use]
pub fn detect_cycle(edges: &BTreeMap<String, BTreeSet<String>>) -> Option<Vec<String>> {
    let mut nodes: BTreeSet<&str> = BTreeSet::new();
    for (node, dependencies) in edges {
        nodes.insert(node.as_str());
        for dependency in dependencies {
            nodes.insert(dependency.as_str());
        }
    }
    let mut marks: BTreeMap<&str, Mark> = BTreeMap::new();
    let mut path: Vec<&str> = Vec::new();
    for node in nodes {
        if let Some(cycle) = walk(node, edges, &mut marks, &mut path) {
            return Some(close_cycle(cycle));
        }
    }
    None
}

/// Depth-first walk from `node`; returns the raw stack slice of the first cycle it finds.
fn walk<'a>(
    node: &'a str,
    edges: &'a BTreeMap<String, BTreeSet<String>>,
    marks: &mut BTreeMap<&'a str, Mark>,
    path: &mut Vec<&'a str>,
) -> Option<Vec<String>> {
    match marks.get(node) {
        Some(Mark::Done) => return None,
        Some(Mark::Active) => {
            // `node` is on the current path, so everything above it closes one cycle.
            let start = path.iter().position(|seen| *seen == node)?;
            return Some(path[start..].iter().map(|id| (*id).to_owned()).collect());
        }
        None => {}
    }
    marks.insert(node, Mark::Active);
    path.push(node);
    if let Some(dependencies) = edges.get(node) {
        for dependency in dependencies {
            if let Some(cycle) = walk(dependency, edges, marks, path) {
                return Some(cycle);
            }
        }
    }
    path.pop();
    marks.insert(node, Mark::Done);
    None
}

/// Rotates `path` to start at its smallest id and closes the loop with that same id.
fn close_cycle(mut path: Vec<String>) -> Vec<String> {
    let smallest = path
        .iter()
        .enumerate()
        .min_by(|(_, left), (_, right)| left.cmp(right))
        .map(|(index, _)| index);
    if let Some(index) = smallest {
        path.rotate_left(index);
    }
    if let Some(first) = path.first().cloned() {
        path.push(first);
    }
    path
}

/// Returns the nesting depth of a task from its ancestor chain.
///
/// The chain is the task's `parent_task_id` links: the chain of the *child* includes its parent, so
/// a top-level task (empty chain) has depth `0` and the direct subtask of a top-level task has depth
/// `1`. Order does not matter — only the number of links, which is why the board passes the chain it
/// walked from `parent_task_id` upward.
#[must_use]
pub fn depth_of(parent_chain: &[String]) -> usize {
    parent_chain.len()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn edges(rows: &[(&str, &[&str])]) -> BTreeMap<String, BTreeSet<String>> {
        rows.iter()
            .map(|(node, dependencies)| {
                (
                    (*node).to_owned(),
                    dependencies.iter().map(|dep| (*dep).to_owned()).collect(),
                )
            })
            .collect()
    }

    #[test]
    fn an_acyclic_graph_has_no_cycle() {
        assert_eq!(detect_cycle(&edges(&[])), None);
        assert_eq!(
            detect_cycle(&edges(&[("a", &[]), ("b", &["a"]), ("c", &["a", "b"])])),
            None
        );
    }

    #[test]
    fn a_two_node_cycle_is_reported_in_dependency_order() {
        let cycle = detect_cycle(&edges(&[("a", &["b"]), ("b", &["a"])])).unwrap();
        assert_eq!(cycle, vec!["a".to_owned(), "b".to_owned(), "a".to_owned()]);
    }

    #[test]
    fn a_self_dependency_is_a_one_node_cycle() {
        let cycle = detect_cycle(&edges(&[("a", &["a"])])).unwrap();
        assert_eq!(cycle, vec!["a".to_owned(), "a".to_owned()]);
    }

    #[test]
    fn the_reported_path_starts_at_the_smallest_node_of_the_cycle() {
        // `z -> m -> b -> z`: the traversal starts at `b` only if it is visited first; the rotated
        // path must always begin at `b`.
        let cycle = detect_cycle(&edges(&[("m", &["b"]), ("z", &["m"]), ("b", &["z"])])).unwrap();
        assert_eq!(
            cycle,
            vec![
                "b".to_owned(),
                "z".to_owned(),
                "m".to_owned(),
                "b".to_owned()
            ]
        );
    }

    #[test]
    fn an_unrelated_cycle_next_to_a_diamond_is_still_found() {
        let cycle = detect_cycle(&edges(&[
            ("root", &["left", "right"]),
            ("left", &["leaf"]),
            ("right", &["leaf"]),
            ("leaf", &[]),
            ("x", &["y"]),
            ("y", &["x"]),
        ]))
        .unwrap();
        assert_eq!(cycle, vec!["x".to_owned(), "y".to_owned(), "x".to_owned()]);
    }

    #[test]
    fn a_dependency_only_node_is_traversed_safely() {
        // `dangling` has no outgoing edge; it must neither panic nor fabricate a cycle.
        assert_eq!(detect_cycle(&edges(&[("a", &["dangling"])])), None);
    }

    #[test]
    fn depth_counts_the_ancestor_chain() {
        assert_eq!(depth_of(&[]), 0);
        assert_eq!(depth_of(&["parent".to_owned()]), 1);
        assert_eq!(
            depth_of(&["parent".to_owned(), "grandparent".to_owned()]),
            2
        );
    }
}
