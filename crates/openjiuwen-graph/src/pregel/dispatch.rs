//! Compilation and evaluation of a node's outgoing routing.
//!
//! Static (and barrier) edges are compiled into one static target list per
//! source; conditional routers are appended after it. A node's
//! [`NodeOutcome::Route`] replaces the routers when it carries targets.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use openjiuwen_common::error::{Error, Result};

use crate::node::{GraphContext, NodeOutcome};
use crate::router::{Router, RouterContext};
use crate::spec::{EdgeKind, GraphSpec, DYNAMIC_TARGET};

/// One source node's compiled dispatch.
#[derive(Default)]
struct DispatchEntry {
    static_targets: Vec<String>,
    routers: Vec<Arc<dyn Router>>,
}

/// The compiled dispatch table of a graph.
#[derive(Default)]
pub(crate) struct DispatchTable {
    entries: BTreeMap<String, DispatchEntry>,
    nodes: BTreeSet<String>,
}

impl DispatchTable {
    /// Compiles the spec's edges and the builder's routers.
    ///
    /// # Errors
    ///
    /// Returns [`Error::graph`] when a router is attached to a node that does
    /// not exist.
    pub(crate) fn compile(spec: &GraphSpec, routers: &[(String, Arc<dyn Router>)]) -> Result<Self> {
        let mut entries: BTreeMap<String, DispatchEntry> = BTreeMap::new();
        for node in &spec.nodes {
            entries.insert(node.id.clone(), DispatchEntry::default());
        }
        for edge in &spec.edges {
            if edge.kind == EdgeKind::Conditional || edge.target == DYNAMIC_TARGET {
                continue;
            }
            for source in &edge.sources {
                let entry = entries.entry(source.clone()).or_default();
                if !entry.static_targets.contains(&edge.target) {
                    entry.static_targets.push(edge.target.clone());
                }
            }
        }
        for (source, router) in routers {
            let entry = entries
                .get_mut(source)
                .ok_or_else(|| Error::graph(format!("router source `{source}` is not a node")))?;
            entry.routers.push(Arc::clone(router));
        }
        Ok(Self {
            entries,
            nodes: spec.nodes.iter().map(|node| node.id.clone()).collect(),
        })
    }

    /// Resolves the nodes to activate after `node_id` produced `outcome`.
    ///
    /// A non-empty [`NodeOutcome::Route`] replaces the routers; otherwise the
    /// static targets come first, followed by every router's result, deduped in
    /// order. An empty [`NodeOutcome::Route`] falls back to normal routing.
    ///
    /// # Errors
    ///
    /// Returns [`Error::graph`] when a target is not a node of the graph, or
    /// when a router fails.
    pub(crate) async fn targets_for(
        &self,
        node_id: &str,
        outcome: &NodeOutcome,
        state: &crate::state::GraphState,
        ctx: &GraphContext,
    ) -> Result<Vec<String>> {
        if let NodeOutcome::Route { targets } = outcome {
            if !targets.is_empty() {
                return self.validate(node_id, targets.iter().cloned());
            }
        }
        let Some(entry) = self.entries.get(node_id) else {
            return Ok(Vec::new());
        };
        let mut targets: Vec<String> = entry.static_targets.clone();
        for router in &entry.routers {
            let router_ctx = RouterContext::new(
                node_id.to_string(),
                state.clone(),
                ctx.superstep(),
                Arc::clone(ctx.session()),
                ctx.metadata_arc(),
            );
            let result = router.route(&router_ctx).await.map_err(|error| {
                error.with_context(format!(
                    "router `{}` for node `{node_id}` failed",
                    router.name()
                ))
            })?;
            for target in result {
                if !targets.contains(&target) {
                    targets.push(target);
                }
            }
        }
        self.validate(node_id, targets)
    }

    /// Checks that every target is a known node, preserving declaration order.
    fn validate(
        &self,
        node_id: &str,
        targets: impl IntoIterator<Item = String>,
    ) -> Result<Vec<String>> {
        let mut known = Vec::new();
        for target in targets {
            if !self.nodes.contains(&target) {
                return Err(Error::graph(format!(
                    "node `{node_id}` routed to unknown node `{target}`"
                )));
            }
            if !known.contains(&target) {
                known.push(target);
            }
        }
        Ok(known)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]
mod tests {
    use super::*;
    use crate::node::GraphContext;
    use crate::router::FnRouter;
    use crate::spec::{EdgeSpec, NodeSpec};
    use crate::state::GraphState;
    use openjiuwen_session::{Session, WorkflowSession};
    use serde_json::json;
    use std::collections::BTreeMap;
    use tokio_util::sync::CancellationToken;

    fn spec(edges: Vec<EdgeSpec>) -> GraphSpec {
        GraphSpec {
            graph_id: "g".to_string(),
            entry: vec!["a".to_string()],
            exit: Vec::new(),
            nodes: ["a", "b", "c"]
                .into_iter()
                .map(|id| NodeSpec {
                    id: id.to_string(),
                    ..NodeSpec::default()
                })
                .collect(),
            edges,
            metadata: BTreeMap::new(),
        }
    }

    fn edge(sources: &[&str], target: &str, kind: EdgeKind) -> EdgeSpec {
        EdgeSpec {
            sources: sources.iter().map(|source| (*source).to_string()).collect(),
            target: target.to_string(),
            kind,
        }
    }

    fn context() -> GraphContext {
        GraphContext::new(
            Arc::new(WorkflowSession::new("s1")) as Arc<dyn Session>,
            "g",
            "a",
            1,
            CancellationToken::new(),
            Arc::new(BTreeMap::new()),
        )
    }

    fn state() -> GraphState {
        GraphState::from_value(json!({"go": true})).unwrap()
    }

    #[tokio::test]
    async fn static_targets_are_followed() {
        let table =
            DispatchTable::compile(&spec(vec![edge(&["a"], "b", EdgeKind::Static)]), &[]).unwrap();
        let targets = table
            .targets_for("a", &NodeOutcome::Continue, &state(), &context())
            .await
            .unwrap();
        assert_eq!(targets, vec!["b".to_string()]);
    }

    #[tokio::test]
    async fn route_outcome_replaces_routers() {
        let router = Arc::new(FnRouter::new("never", |_ctx: &RouterContext| {
            Box::pin(async { Ok(vec!["c".to_string()]) })
        }));
        let table = DispatchTable::compile(
            &spec(vec![edge(&["a"], "b", EdgeKind::Static)]),
            &[("a".to_string(), router)],
        )
        .unwrap();
        let targets = table
            .targets_for("a", &NodeOutcome::route(["b"]), &state(), &context())
            .await
            .unwrap();
        assert_eq!(targets, vec!["b".to_string()]);
    }

    #[tokio::test]
    async fn empty_route_falls_back_to_routers() {
        let table = DispatchTable::compile(
            &spec(vec![edge(&["a"], "b", EdgeKind::Static)]),
            &[(
                "a".to_string(),
                Arc::new(FnRouter::new("to_c", |_ctx: &RouterContext| {
                    Box::pin(async { Ok(vec!["c".to_string()]) })
                })),
            )],
        )
        .unwrap();
        let targets = table
            .targets_for(
                "a",
                &NodeOutcome::Route {
                    targets: Vec::new(),
                },
                &state(),
                &context(),
            )
            .await
            .unwrap();
        assert_eq!(targets, vec!["b".to_string(), "c".to_string()]);
    }

    #[tokio::test]
    async fn routers_run_after_static_targets_and_dedupe() {
        let table = DispatchTable::compile(
            &spec(vec![edge(&["a"], "b", EdgeKind::Static)]),
            &[(
                "a".to_string(),
                Arc::new(FnRouter::new("both", |_ctx: &RouterContext| {
                    Box::pin(async { Ok(vec!["b".to_string(), "c".to_string(), "c".to_string()]) })
                })),
            )],
        )
        .unwrap();
        let targets = table
            .targets_for("a", &NodeOutcome::Continue, &state(), &context())
            .await
            .unwrap();
        assert_eq!(targets, vec!["b".to_string(), "c".to_string()]);
    }

    #[tokio::test]
    async fn unknown_targets_fail() {
        let table =
            DispatchTable::compile(&spec(vec![edge(&["a"], "b", EdgeKind::Static)]), &[]).unwrap();
        let error = table
            .targets_for("a", &NodeOutcome::route(["zzz"]), &state(), &context())
            .await
            .unwrap_err();
        assert!(error.message().contains("unknown node `zzz`"));
    }

    #[test]
    fn conditional_edges_do_not_become_static_targets() {
        let table =
            DispatchTable::compile(&spec(vec![edge(&["a"], "b", EdgeKind::Conditional)]), &[])
                .unwrap();
        let entry = table.entries.get("a").unwrap();
        assert!(entry.static_targets.is_empty());
    }

    #[test]
    fn router_for_unknown_node_fails() {
        let error = DispatchTable::compile(
            &spec(Vec::new()),
            &[(
                "zzz".to_string(),
                Arc::new(FnRouter::new("bad", |_ctx: &RouterContext| {
                    Box::pin(async { Ok(Vec::new()) })
                })),
            )],
        )
        .err()
        .unwrap();
        assert!(error
            .message()
            .contains("router source `zzz` is not a node"));
    }

    #[tokio::test]
    async fn router_errors_carry_context() {
        let table = DispatchTable::compile(
            &spec(Vec::new()),
            &[(
                "a".to_string(),
                Arc::new(FnRouter::new("broken", |_ctx: &RouterContext| {
                    Box::pin(async { Err(openjiuwen_common::error::Error::validation("nope")) })
                })),
            )],
        )
        .unwrap();
        let error = table
            .targets_for("a", &NodeOutcome::Continue, &state(), &context())
            .await
            .unwrap_err();
        let message = error.message();
        assert!(message.contains("router `broken`"));
        assert!(message.contains("nope"));
    }
}
