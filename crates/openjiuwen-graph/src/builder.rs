//! Graph builder and compile-time validation.
//!
//! Builder methods never fail: they record the graph description (and any
//! problems they can already see) so [`GraphBuilder::compile`] can report every
//! problem as one [`openjiuwen_common::error::Error::graph`].
//!
//! ```
//! use openjiuwen_graph::{GraphBuilder, NodeOutcome, SyncNode};
//! # use openjiuwen_common::error::Result;
//! # use openjiuwen_graph::{GraphContext, GraphState};
//!
//! # fn main() -> Result<()> {
//! let graph = GraphBuilder::new("ping")
//!     .node(
//!         "start",
//!         SyncNode::new("start", |state: &mut GraphState, _ctx: &GraphContext| {
//!             state.set("ready", serde_json::json!(true));
//!             Ok(NodeOutcome::Continue)
//!         }),
//!     )
//!     .node("end", SyncNode::new("end", |_state: &mut GraphState, _ctx: &GraphContext| {
//!         Ok(NodeOutcome::Continue)
//!     }))
//!     .edge("start", "end")
//!     .exit("end")
//!     .compile()?;
//! assert_eq!(graph.id(), "ping");
//! # Ok(())
//! # }
//! ```

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use serde_json::Value;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_session::Checkpointer;

use crate::compiled::{CompiledGraph, CompiledInner};
use crate::node::{Node, NodeOptions};
use crate::pregel::channel::{ChannelManager, NodeChannel};
use crate::pregel::dispatch::DispatchTable;
use crate::router::Router;
use crate::spec::{EdgeKind, EdgeSpec, GraphSpec, NodeSpec, DYNAMIC_TARGET};
use crate::store::{CheckpointerStore, GraphStore};

/// Options that shape how a builder compiles into a [`CompiledGraph`].
#[derive(Debug, Clone)]
pub struct CompileOptions {
    /// Snapshot store used for checkpointing and resume. Takes precedence over
    /// [`CompileOptions::checkpointer`].
    pub store: Option<Arc<dyn GraphStore>>,
    /// Checkpointer wrapped into a [`CheckpointerStore`] when no explicit store
    /// is given.
    pub checkpointer: Option<Arc<dyn Checkpointer>>,
    /// Snapshot namespace; defaults to the graph id.
    pub namespace: Option<String>,
    /// Default superstep budget for runs of this graph.
    pub max_supersteps: Option<usize>,
    /// Default per-attempt node timeout in milliseconds.
    pub node_timeout_ms: Option<u64>,
    /// Save a snapshot after every completed superstep (a no-op without a
    /// store).
    pub checkpoint_every_superstep: bool,
}

impl Default for CompileOptions {
    fn default() -> Self {
        Self {
            store: None,
            checkpointer: None,
            namespace: None,
            max_supersteps: None,
            node_timeout_ms: None,
            checkpoint_every_superstep: true,
        }
    }
}

/// Assembles a graph description and compiles it into a runnable
/// [`CompiledGraph`].
///
/// Every method consumes and returns `Self`, so a graph is built in one
/// expression.
pub struct GraphBuilder {
    spec: GraphSpec,
    nodes: BTreeMap<String, Arc<dyn Node>>,
    routers: Vec<(String, Arc<dyn Router>)>,
    problems: Vec<String>,
}

impl GraphBuilder {
    /// Starts a graph with `graph_id`.
    #[must_use]
    pub fn new(graph_id: impl Into<String>) -> Self {
        Self {
            spec: GraphSpec {
                graph_id: graph_id.into(),
                ..GraphSpec::default()
            },
            nodes: BTreeMap::new(),
            routers: Vec::new(),
            problems: Vec::new(),
        }
    }

    /// Adds a node with the default [`NodeOptions`].
    #[must_use]
    pub fn node(self, id: impl Into<String>, node: impl Node + 'static) -> Self {
        self.add_node(id.into(), Arc::new(node), NodeOptions::default())
    }

    /// Adds a node with explicit [`NodeOptions`].
    #[must_use]
    pub fn node_with(
        self,
        id: impl Into<String>,
        node: impl Node + 'static,
        options: NodeOptions,
    ) -> Self {
        self.add_node(id.into(), Arc::new(node), options)
    }

    /// Adds a plain one-to-one edge.
    #[must_use]
    pub fn edge(mut self, from: impl Into<String>, to: impl Into<String>) -> Self {
        self.spec.edges.push(EdgeSpec {
            sources: vec![from.into()],
            target: to.into(),
            kind: EdgeKind::Static,
        });
        self
    }

    /// Adds one edge per source, so `to` is ready only when every source fired.
    #[must_use]
    pub fn barrier_edge(
        mut self,
        from: impl IntoIterator<Item = impl Into<String>>,
        to: impl Into<String>,
    ) -> Self {
        let target = to.into();
        let sources: Vec<String> = from.into_iter().map(Into::into).collect();
        if sources.is_empty() {
            self.problems
                .push(format!("barrier edge to `{target}` has no sources"));
            return self;
        }
        self.spec.edges.push(EdgeSpec {
            sources,
            target,
            kind: EdgeKind::Barrier,
        });
        self
    }

    /// Attaches a conditional router to `from`.
    ///
    /// The router's targets are only known at runtime, so the spec records a
    /// dynamic conditional edge.
    #[must_use]
    pub fn conditional_edges(
        mut self,
        from: impl Into<String>,
        router: impl Router + 'static,
    ) -> Self {
        let source = from.into();
        self.routers.push((source.clone(), Arc::new(router)));
        self.spec.edges.push(EdgeSpec {
            sources: vec![source],
            target: DYNAMIC_TARGET.to_string(),
            kind: EdgeKind::Conditional,
        });
        self
    }

    /// Attaches a conditional router and declares its possible targets.
    ///
    /// The declared targets are what visualization and spec consumers see; the
    /// router still decides the actual routing at run time.
    #[must_use]
    pub fn conditional_edges_to(
        mut self,
        from: impl Into<String>,
        targets: impl IntoIterator<Item = impl Into<String>>,
        router: impl Router + 'static,
    ) -> Self {
        let source = from.into();
        self.routers.push((source.clone(), Arc::new(router)));
        for target in targets {
            self.spec.edges.push(EdgeSpec {
                sources: vec![source.clone()],
                target: target.into(),
                kind: EdgeKind::Conditional,
            });
        }
        self
    }

    /// Marks `id` as an entry node.
    #[must_use]
    pub fn entry(mut self, id: impl Into<String>) -> Self {
        self.spec.entry.push(id.into());
        self
    }

    /// Marks `id` as an exit node.
    #[must_use]
    pub fn exit(mut self, id: impl Into<String>) -> Self {
        self.spec.exit.push(id.into());
        self
    }

    /// Records one metadata entry, copied into every [`crate::GraphContext`].
    #[must_use]
    pub fn metadata(mut self, key: impl Into<String>, value: Value) -> Self {
        self.spec.metadata.insert(key.into(), value);
        self
    }

    /// Returns the partial spec built so far.
    #[must_use]
    pub fn spec(&self) -> &GraphSpec {
        &self.spec
    }

    /// Compiles with [`CompileOptions::default`].
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::ErrorKind::Graph`] listing every
    /// validation problem.
    pub fn compile(self) -> Result<CompiledGraph> {
        self.compile_with(CompileOptions::default())
    }

    /// Compiles the graph, reporting every recorded problem as one error.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::ErrorKind::Graph`] when the builder
    /// or the spec has a validation problem.
    pub fn compile_with(mut self, options: CompileOptions) -> Result<CompiledGraph> {
        let mut problems = std::mem::take(&mut self.problems);
        if self.spec.entry.is_empty() {
            self.spec.entry = implicit_entries(&self.spec);
        }
        problems.extend(self.spec.problems());
        if !problems.is_empty() {
            return Err(Error::graph(problems.join("; ")));
        }

        let channels = compile_channels(&self.spec);
        let dispatch = DispatchTable::compile(&self.spec, &self.routers)?;
        let graph_id = self.spec.graph_id.clone();
        let store = match (options.store, options.checkpointer) {
            (Some(store), _) => Some(store),
            (None, Some(checkpointer)) => Some(Arc::new(CheckpointerStore::new(
                checkpointer,
                graph_id.clone(),
            )) as Arc<dyn GraphStore>),
            (None, None) => None,
        };
        let node_options = self
            .spec
            .nodes
            .iter()
            .map(|node| (node.id.clone(), node.options.clone()))
            .collect();
        let order: Vec<String> = self.spec.nodes.iter().map(|node| node.id.clone()).collect();
        let metadata = Arc::new(self.spec.metadata.clone());
        let inner = CompiledInner {
            graph_id: graph_id.clone(),
            spec: self.spec,
            nodes: self.nodes,
            options: node_options,
            order,
            channels,
            dispatch,
            store,
            namespace: options.namespace.unwrap_or(graph_id),
            metadata,
            max_supersteps: options.max_supersteps,
            node_timeout_ms: options.node_timeout_ms,
            checkpoint_every_superstep: options.checkpoint_every_superstep,
        };
        Ok(CompiledGraph {
            inner: Arc::new(inner),
        })
    }

    /// Records a node, or a duplicate problem when the id is already taken.
    fn add_node(mut self, id: String, node: Arc<dyn Node>, options: NodeOptions) -> Self {
        if self.nodes.contains_key(&id) {
            self.problems.push(format!("duplicate node id `{id}`"));
        } else {
            self.nodes.insert(id.clone(), node);
        }
        self.spec.nodes.push(NodeSpec {
            id,
            kind: "node".to_string(),
            options,
        });
        self
    }
}

impl std::fmt::Debug for GraphBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GraphBuilder")
            .field("spec", &self.spec)
            .field("routers", &self.routers.len())
            .field("problems", &self.problems)
            .finish_non_exhaustive()
    }
}

/// Entry nodes when the builder declares none: nodes no edge points at.
fn implicit_entries(spec: &GraphSpec) -> Vec<String> {
    let incoming: BTreeSet<&str> = spec
        .edges
        .iter()
        .filter(|edge| edge.target != DYNAMIC_TARGET)
        .map(|edge| edge.target.as_str())
        .collect();
    spec.nodes
        .iter()
        .filter(|node| !incoming.contains(node.id.as_str()))
        .map(|node| node.id.clone())
        .collect()
}

/// Compiles each node's CNF channel groups from the spec's edges.
fn compile_channels(spec: &GraphSpec) -> ChannelManager {
    let mut channels = BTreeMap::new();
    for node in &spec.nodes {
        let mut groups: Vec<BTreeSet<String>> = Vec::new();
        for edge in &spec.edges {
            if edge.target != node.id || edge.target == DYNAMIC_TARGET {
                continue;
            }
            match edge.kind {
                EdgeKind::Barrier => {
                    for source in &edge.sources {
                        let group = BTreeSet::from([source.clone()]);
                        if !groups.contains(&group) {
                            groups.push(group);
                        }
                    }
                }
                EdgeKind::Static | EdgeKind::Conditional => {
                    let group: BTreeSet<String> = edge.sources.iter().cloned().collect();
                    if !group.is_empty() && !groups.contains(&group) {
                        groups.push(group);
                    }
                }
            }
        }
        channels.insert(node.id.clone(), NodeChannel::new(groups));
    }
    ChannelManager::new(channels)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]
mod tests {
    use super::*;
    use crate::node::{GraphContext, NodeOutcome, SyncNode};
    use crate::router::FnRouter;
    use crate::state::GraphState;
    use crate::store::InMemoryGraphStore;
    use openjiuwen_session::MemoryCheckpointer;
    use serde_json::json;

    fn noop(id: &str) -> SyncNode<impl Fn(&mut GraphState, &GraphContext) -> Result<NodeOutcome>> {
        let name = id.to_string();
        SyncNode::new(id, move |_state: &mut GraphState, _ctx: &GraphContext| {
            let _ = &name;
            Ok(NodeOutcome::Continue)
        })
    }

    fn linear() -> GraphBuilder {
        GraphBuilder::new("g")
            .node("a", noop("a"))
            .node("b", noop("b"))
            .edge("a", "b")
            .exit("b")
    }

    #[test]
    fn compiles_a_linear_graph() {
        let graph = linear().compile().unwrap();
        assert_eq!(graph.id(), "g");
        assert_eq!(graph.spec().entry, vec!["a".to_string()]);
        assert_eq!(graph.spec().exit, vec!["b".to_string()]);
        assert_eq!(graph.node_ids().collect::<Vec<_>>(), vec!["a", "b"]);
        assert_eq!(graph.entry_nodes(), ["a".to_string()]);
        assert_eq!(graph.exit_nodes(), vec!["b"]);
    }

    #[test]
    fn implicit_entries_are_nodes_without_incoming_edges() {
        let graph = GraphBuilder::new("g")
            .node("a", noop("a"))
            .node("b", noop("b"))
            .node("c", noop("c"))
            .edge("a", "c")
            .edge("b", "c")
            .compile()
            .unwrap();
        assert_eq!(graph.spec().entry, vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn explicit_entries_win() {
        let graph = GraphBuilder::new("g")
            .node("a", noop("a"))
            .node("b", noop("b"))
            .edge("a", "b")
            .entry("b")
            .compile()
            .unwrap();
        assert_eq!(graph.entry_nodes(), ["b".to_string()]);
    }

    #[test]
    fn reports_every_problem_in_one_error() {
        let error = GraphBuilder::new("g")
            .node("a", noop("a"))
            .node("a", noop("a"))
            .edge("a", "missing")
            .compile()
            .unwrap_err();
        let message = error.message();
        assert_eq!(error.kind(), openjiuwen_common::error::ErrorKind::Graph);
        assert!(message.contains("duplicate node id `a`"));
        assert!(message.contains("edge target `missing` is not a node"));
        assert!(message.contains("; "));
    }

    #[test]
    fn empty_graph_fails_to_compile() {
        let error = GraphBuilder::new("g").compile().unwrap_err();
        assert!(error.message().contains("has no nodes"));
    }

    #[test]
    fn missing_entry_fails_when_every_node_has_an_incoming_edge() {
        let error = GraphBuilder::new("g")
            .node("a", noop("a"))
            .node("b", noop("b"))
            .edge("a", "b")
            .edge("b", "a")
            .compile()
            .unwrap_err();
        assert!(error.message().contains("no entry nodes"));
    }

    #[test]
    fn barrier_edge_with_no_sources_is_reported() {
        let error = GraphBuilder::new("g")
            .node("a", noop("a"))
            .barrier_edge(Vec::<String>::new(), "a")
            .compile()
            .unwrap_err();
        assert!(error
            .message()
            .contains("barrier edge to `a` has no sources"));
    }

    #[test]
    fn conditional_edges_record_the_router_source() {
        let router = FnRouter::new("pick", |_ctx: &crate::RouterContext| {
            Box::pin(async { Ok(vec!["b".to_string()]) })
        });
        let graph = GraphBuilder::new("g")
            .node("a", noop("a"))
            .node("b", noop("b"))
            .conditional_edges_to("a", ["b"], router)
            .compile()
            .unwrap();
        let spec = graph.spec();
        assert_eq!(spec.outgoing("a").len(), 1);
        assert_eq!(spec.outgoing("a")[0].target, "b");
        assert_eq!(spec.outgoing("a")[0].kind, EdgeKind::Conditional);
        assert_eq!(spec.entry, vec!["a".to_string()]);
    }

    #[test]
    fn dynamic_conditional_edge_is_recorded() {
        let router = FnRouter::new("pick", |_ctx: &crate::RouterContext| {
            Box::pin(async { Ok(Vec::new()) })
        });
        let builder = GraphBuilder::new("g")
            .node("a", noop("a"))
            .node("b", noop("b"))
            .edge("a", "b")
            .conditional_edges("a", router);
        let dynamic = builder
            .spec()
            .edges
            .iter()
            .find(|edge| edge.target == DYNAMIC_TARGET)
            .unwrap();
        assert_eq!(dynamic.kind, EdgeKind::Conditional);
        builder.compile().unwrap();
    }

    #[tokio::test]
    async fn explicit_store_is_preferred_over_a_checkpointer() {
        let store = Arc::new(InMemoryGraphStore::new());
        let graph = linear()
            .compile_with(CompileOptions {
                store: Some(store.clone()),
                checkpointer: Some(Arc::new(MemoryCheckpointer::new())),
                ..CompileOptions::default()
            })
            .unwrap();
        let stored: Arc<dyn GraphStore> = graph.inner.store.clone().unwrap();
        stored
            .save("s1", "ns", crate::store::GraphSnapshot::default())
            .await
            .unwrap();
        assert_eq!(store.len(), 1);
    }

    #[tokio::test]
    async fn checkpointer_is_wrapped_and_namespaced() {
        let checkpointer = Arc::new(MemoryCheckpointer::new());
        let graph = linear()
            .compile_with(CompileOptions {
                checkpointer: Some(checkpointer.clone()),
                ..CompileOptions::default()
            })
            .unwrap();
        assert_eq!(graph.inner.namespace, "g");
        let stored = graph.inner.store.clone().unwrap();
        stored
            .save("s1", "g", crate::store::GraphSnapshot::default())
            .await
            .unwrap();
        assert_eq!(checkpointer.len("s1"), 1);
    }

    #[test]
    fn compile_options_defaults_checkpoint_every_superstep() {
        let options = CompileOptions::default();
        assert!(options.checkpoint_every_superstep);
        assert!(options.store.is_none());
        assert!(options.max_supersteps.is_none());
    }

    #[test]
    fn namespace_defaults_to_graph_id_and_can_be_overridden() {
        let default = linear().compile().unwrap();
        assert_eq!(default.inner.namespace, "g");
        let custom = linear()
            .compile_with(CompileOptions {
                namespace: Some("custom".to_string()),
                ..CompileOptions::default()
            })
            .unwrap();
        assert_eq!(custom.inner.namespace, "custom");
    }

    #[test]
    fn metadata_reaches_the_compiled_graph() {
        let graph = linear().metadata("owner", json!("test")).compile().unwrap();
        assert_eq!(graph.spec().metadata.get("owner"), Some(&json!("test")));
        assert_eq!(graph.inner.metadata.get("owner"), Some(&json!("test")));
    }

    #[test]
    fn node_options_are_carried_into_the_spec() {
        let graph = GraphBuilder::new("g")
            .node_with(
                "a",
                noop("a"),
                NodeOptions {
                    timeout_ms: Some(10),
                    max_retries: 3,
                },
            )
            .compile()
            .unwrap();
        let node = graph.spec().node("a").unwrap();
        assert_eq!(node.options.max_retries, 3);
        assert_eq!(node.options.timeout_ms, Some(10));
        assert_eq!(node.kind, "node");
    }

    #[test]
    fn barrier_edges_produce_one_group_per_source() {
        let graph = GraphBuilder::new("g")
            .node("a", noop("a"))
            .node("b", noop("b"))
            .node("c", noop("c"))
            .barrier_edge(["a", "b"], "c")
            .compile()
            .unwrap();
        let snapshot = graph.inner.channels.snapshot();
        assert_eq!(
            snapshot["c"].groups,
            vec![vec!["a".to_string()], vec!["b".to_string()]]
        );
    }

    #[test]
    fn builder_debug_names_its_parts() {
        let builder = linear();
        let rendered = format!("{builder:?}");
        assert!(rendered.contains("GraphBuilder"));
        assert!(rendered.contains("\"g\""));
    }
}
