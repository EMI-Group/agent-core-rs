//! The workflow engine: the recorded definition, its validation, the graph adapter, and
//! the compiled workflow.
//!
//! [`WorkflowDefinition`] is what the builder records: components with their node
//! configuration, edges, conditional fan-out, the start/end ids, and any problems the
//! builder noticed. [`validate::validate`] turns a definition into one coded compile
//! error listing every problem. [`CompiledWorkflow`] validates, maps the definition onto
//! an `openjiuwen-graph` graph, and runs it.
//!
//! Everything about supersteps, channels, retries, checkpoints, and Mermaid rendering
//! belongs to `openjiuwen-graph`; this module compiles a graph per run (node instances
//! carry per-run interaction state and frame writers) and projects the result into a
//! [`WorkflowOutput`](crate::output::WorkflowOutput) or a frame stream.
//!
//! # Fan-in
//!
//! A regular, stream, or barrier edge compiles to a CNF channel, so a node with several
//! inbound edges waits for every predecessor — Python's `wait_for_all = True`.
//!
//! A component registered with `wait_for_all = false` that has at least two distinct
//! regular/stream/barrier predecessors instead uses Python's *first-arrival* fan-in: its
//! static edges are compiled as dynamic router edges, so the node keeps no CNF group and
//! `openjiuwen-graph` treats its channel as a trigger channel. The node runs as soon as
//! any predecessor arrives and runs again when a later one does; a template key whose
//! producer has not run yet resolves to `null`. A source that also has declared
//! conditional edges into the node keeps its routing decision: it reaches the node only
//! when its router selects it.
//!
//! Nodes left on CNF barriers: a single-source node (byte-identical to a plain edge, so
//! `wait_for_all = false` changes nothing), a node whose only fan-in is a declared
//! conditional edge, and `Collect`/`Transform` nodes, which validation rejects with
//! `[100011]` because buffered stream composition needs every producer.
//!
//! ```no_run
//! use std::sync::Arc;
//!
//! use openjiuwen_session::{Session, WorkflowSession};
//! use openjiuwen_workflow::engine::{CompiledWorkflow, WorkflowRunOptions};
//! use openjiuwen_workflow::{NodeConfig, StartComponent, WorkflowCard, WorkflowDefinition};
//! use serde_json::json;
//!
//! # async fn example() -> openjiuwen_common::error::Result<()> {
//! let mut definition = WorkflowDefinition::new(WorkflowCard::new("wf"));
//! definition.insert_component("start", Arc::new(StartComponent::new()), NodeConfig::default());
//! let workflow = CompiledWorkflow::build(definition)?;
//! let session: Arc<dyn Session> = Arc::new(WorkflowSession::new("s1"));
//! let output = workflow
//!     .invoke(json!({"query": "hi"}), session, WorkflowRunOptions::default())
//!     .await?;
//! assert_eq!(output.result, json!({"query": "hi"}));
//! # Ok(())
//! # }
//! ```

pub mod definition;
mod mutation;
pub mod node;
mod run;
pub mod validate;

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use openjiuwen_graph::{
    CompileOptions, CompiledGraph, GraphBuilder, GraphStore, NodeOptions, Router, RouterContext,
};
use openjiuwen_session::stream::StreamMode;
use openjiuwen_session::InteractiveInput;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio_util::sync::CancellationToken;

use openjiuwen_common::error::{Error, Result};

use crate::ability::ComponentAbility;
use crate::card::{WorkflowCard, WorkflowConfig, WorkflowSpec};
use crate::engine::definition::EdgeKind as DefinitionEdgeKind;
use crate::engine::node::{ComponentNode, NodeRuntime, NodeSchemas, RunScope};
use crate::error;

pub use definition::{ConditionalSource, Edge, EdgeKind, Entry, NodeConfig, WorkflowDefinition};

/// How one run delivers frames from a streaming producer to its consumers.
///
/// Python's engine feeds producer frames to consumers live through
/// `openjiuwen.core.graph.stream_actor`; this rewrite composes them instead. The mode is an
/// explicit run option so a caller asking for Python's live behaviour learns that it is not
/// available instead of silently receiving composed frames.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamDelivery {
    /// Buffered composition: a `Stream`/`Transform` producer's frames are stored under its
    /// node id, a `Collect`/`Transform` consumer resolves them afterwards through its
    /// stream input schema, and the end node replays its template frames in template order.
    ///
    /// This is this crate's documented delivery model. The observable text and its order
    /// match Python for finite streams; only live interleaving of two concurrently
    /// streaming producers is lost. It is the default.
    #[default]
    Buffered,
    /// Python's live `stream_actor` interleaving, where a producer's frames reach a
    /// consumer while both are running.
    ///
    /// [`openjiuwen_graph`] does not port `stream_actor`, so this engine cannot provide it:
    /// every run entry point rejects this mode with a typed
    /// [`Error::Unsupported`] naming the gap instead of degrading to
    /// [`StreamDelivery::Buffered`].
    Interleaved,
}

impl StreamDelivery {
    /// The lowercase wire name (`"buffered"`, `"interleaved"`).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Buffered => "buffered",
            Self::Interleaved => "interleaved",
        }
    }

    /// Every delivery mode, in declaration order.
    #[must_use]
    pub const fn all() -> [Self; 2] {
        [Self::Buffered, Self::Interleaved]
    }

    /// Whether this mode can be served by the engine in this crate.
    #[must_use]
    pub const fn is_supported(self) -> bool {
        matches!(self, Self::Buffered)
    }
}

impl std::fmt::Display for StreamDelivery {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Tunables for one workflow run.
///
/// Every field has a permissive default, so [`WorkflowRunOptions::default`] runs to
/// completion with no cancellation, no explicit timeout, every stream mode, and
/// [`StreamDelivery::Buffered`] composition.
#[derive(Clone, Debug)]
pub struct WorkflowRunOptions {
    /// Cancels the run when triggered.
    pub cancellation: CancellationToken,
    /// Whole-run timeout; `None` falls back to `SessionConfig::execute_timeout_ms`.
    pub timeout: Option<Duration>,
    /// Answers seeded into the run's interaction state (used by `resume`).
    pub interactive: Option<InteractiveInput>,
    /// Answers replayed before `interactive`, per interaction id (a nested run forwards
    /// the answers its child already consumed so a replayed child sees them in order).
    pub answers: BTreeMap<String, Vec<Value>>,
    /// Continue from the graph snapshot stored for this session.
    pub resume: bool,
    /// Which frame modes `stream` yields; `None` yields every mode.
    pub stream_modes: Option<Vec<StreamMode>>,
    /// Maximum number of supersteps; `None` uses the graph default.
    pub max_supersteps: Option<usize>,
    /// Nesting level of this run: `0` for a top-level run, `1` for a sub-workflow, and
    /// so on. A [`crate::components::sub_workflow::SubWorkflowComponent`] sets it on the
    /// child run; every other caller leaves it at the default.
    pub depth: usize,
    /// How streaming producer frames reach their consumers; see [`StreamDelivery`].
    ///
    /// [`StreamDelivery::Interleaved`] is rejected by every run entry point with a typed
    /// `Error::Unsupported`, because Python's live `stream_actor` interleaving is not
    /// available here.
    pub stream_delivery: StreamDelivery,
}

impl Default for WorkflowRunOptions {
    fn default() -> Self {
        Self {
            cancellation: CancellationToken::new(),
            timeout: None,
            interactive: None,
            answers: BTreeMap::new(),
            resume: false,
            stream_modes: None,
            max_supersteps: None,
            depth: 0,
            stream_delivery: StreamDelivery::Buffered,
        }
    }
}

impl WorkflowRunOptions {
    /// Options with every default (never cancelled, no timeout, all stream modes).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Use `cancellation` as the run's cancellation token.
    #[must_use]
    pub fn with_cancellation(mut self, cancellation: CancellationToken) -> Self {
        self.cancellation = cancellation;
        self
    }

    /// Limit the whole run to `timeout`.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Seed `interactive` answers into the run.
    #[must_use]
    pub fn with_interactive(mut self, interactive: InteractiveInput) -> Self {
        self.interactive = Some(interactive);
        self
    }

    /// Replay `answers` before the run's `interactive` answers.
    ///
    /// Each id's values are queued in order, ahead of an answer supplied through
    /// [`WorkflowRunOptions::interactive`]; a nested run uses this to hand a replayed child
    /// the answers it already consumed.
    #[must_use]
    pub fn with_answers(mut self, answers: BTreeMap<String, Vec<Value>>) -> Self {
        self.answers = answers;
        self
    }

    /// Continue from a stored snapshot instead of starting a fresh run.
    #[must_use]
    pub fn with_resume(mut self, resume: bool) -> Self {
        self.resume = resume;
        self
    }

    /// Yield only frames whose mode is in `modes`.
    #[must_use]
    pub fn with_stream_modes(mut self, modes: Vec<StreamMode>) -> Self {
        self.stream_modes = Some(modes);
        self
    }

    /// Cap the run at `max_supersteps` supersteps.
    #[must_use]
    pub fn with_max_supersteps(mut self, max_supersteps: usize) -> Self {
        self.max_supersteps = Some(max_supersteps);
        self
    }

    /// Declare this run as the `depth`-th nested level (`0` is a top-level run).
    #[must_use]
    pub fn with_depth(mut self, depth: usize) -> Self {
        self.depth = depth;
        self
    }

    /// Request `delivery` for this run's streaming frames.
    ///
    /// [`StreamDelivery::Buffered`] is the default and always runs;
    /// [`StreamDelivery::Interleaved`] is refused by `invoke`, `stream` and `resume` with a
    /// typed `Error::Unsupported` instead of being silently buffered.
    #[must_use]
    pub fn with_stream_delivery(mut self, delivery: StreamDelivery) -> Self {
        self.stream_delivery = delivery;
        self
    }
}

/// The immutable compiled form of a workflow.
///
/// Building one validates the definition and records the Python-compatible spec; the
/// graph itself is compiled per run, so one `CompiledWorkflow` can run concurrently.
#[derive(Clone)]
pub struct CompiledWorkflow {
    inner: Arc<Inner>,
}

/// Everything shared by the clones of a [`CompiledWorkflow`].
struct Inner {
    definition: WorkflowDefinition,
    card: WorkflowCard,
    config: WorkflowConfig,
    spec: WorkflowSpec,
    abilities: BTreeMap<String, ComponentAbility>,
    is_streaming: bool,
    start_id: String,
    end_id: String,
}

impl std::fmt::Debug for CompiledWorkflow {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompiledWorkflow")
            .field("card", &self.inner.card)
            .field("start", &self.inner.start_id)
            .field("end", &self.inner.end_id)
            .field("components", &self.inner.abilities)
            .field("streaming", &self.inner.is_streaming)
            .finish_non_exhaustive()
    }
}

impl CompiledWorkflow {
    /// Validates `definition` and compiles it into a runnable workflow.
    ///
    /// # Errors
    ///
    /// Returns the coded workflow error listing every validation problem, and the
    /// compile error for a definition without a start or end component.
    pub fn build(definition: WorkflowDefinition) -> Result<Self> {
        validate::validate(&definition)?;
        let card = definition.card.clone();
        let end_id = definition
            .end
            .clone()
            .ok_or_else(|| error::compile_error("workflow has no end component", &card.id))?;
        let start_id = definition
            .start
            .clone()
            .ok_or_else(|| error::compile_error("workflow has no start component", &card.id))?;

        let spec = definition.spec();
        let abilities: BTreeMap<String, ComponentAbility> = definition
            .component_ids()
            .into_iter()
            .map(|id| (id.to_owned(), definition.selected_ability(id)))
            .collect();
        let is_streaming = abilities
            .get(&end_id)
            .is_some_and(|ability| ability.is_streaming());

        let mut config = definition.config.clone();
        config.card = card.clone();
        config.spec = spec.clone();

        Ok(Self {
            inner: Arc::new(Inner {
                definition,
                card,
                config,
                spec,
                abilities,
                is_streaming,
                start_id,
                end_id,
            }),
        })
    }

    /// The workflow's card.
    #[must_use]
    pub fn card(&self) -> &WorkflowCard {
        &self.inner.card
    }

    /// The workflow's configuration, with `card` and `spec` refreshed from the definition.
    #[must_use]
    pub fn config(&self) -> &WorkflowConfig {
        &self.inner.config
    }

    /// The Python-compatible workflow spec.
    #[must_use]
    pub fn spec(&self) -> &WorkflowSpec {
        &self.inner.spec
    }

    /// The definition this workflow was compiled from.
    #[must_use]
    pub fn definition(&self) -> &WorkflowDefinition {
        &self.inner.definition
    }

    /// The start component id.
    #[must_use]
    pub fn start_comp_id(&self) -> &str {
        &self.inner.start_id
    }

    /// The end component id.
    #[must_use]
    pub fn end_comp_id(&self) -> &str {
        &self.inner.end_id
    }

    /// Whether the end component streams its output.
    #[must_use]
    pub fn is_streaming(&self) -> bool {
        self.inner.is_streaming
    }

    /// The ability a component is driven with, or `None` for an unknown id.
    #[must_use]
    pub fn ability(&self, id: &str) -> Option<ComponentAbility> {
        self.inner.abilities.get(id).copied()
    }

    /// Renders the workflow as a Mermaid flowchart.
    ///
    /// Compilation for rendering has no run scope and no snapshot store. Because that
    /// compile is render-only, the diagram shows the recorded topology: a
    /// `wait_for_all = false` fan-in is drawn with an edge from every predecessor instead
    /// of the dynamic router edges the run path compiles.
    ///
    /// # Errors
    ///
    /// Returns the compile error when the recorded definition cannot be compiled. No
    /// placeholder Mermaid text is produced; a caller that wants an error document must
    /// render one itself (see [`crate::render::error_svg`]).
    pub fn to_mermaid(&self) -> Result<String> {
        let graph = self.build_graph(None, None, None)?;
        Ok(graph.to_mermaid())
    }

    /// Renders the workflow as an inline SVG document.
    ///
    /// The end component is passed explicitly so the renderer shapes it as an exit node,
    /// and the card name becomes the document title when it is not empty.
    ///
    /// # Errors
    ///
    /// Returns the compile error when the recorded definition cannot be compiled. No
    /// placeholder SVG document is produced; a caller that wants one must build it through
    /// [`crate::render::error_svg`].
    pub fn to_svg(&self) -> Result<String> {
        self.build_graph(None, None, None)?;
        Ok(crate::render::render_svg(
            self.spec(),
            self.card(),
            &crate::render::SvgOptions {
                title: Some(self.inner.card.name.clone()).filter(|name| !name.trim().is_empty()),
                end_nodes: vec![self.inner.end_id.clone()],
                ..crate::render::SvgOptions::default()
            },
        ))
    }

    /// The ability recorded for `id`, defaulting to `Invoke`.
    fn ability_or_default(&self, id: &str) -> ComponentAbility {
        self.inner.abilities.get(id).copied().unwrap_or_default()
    }

    /// Compiles a fresh graph for one run (or for rendering, when `scope` is `None`).
    ///
    /// Node instances carry the run's interaction state and frame writer, so a graph is
    /// never reused across runs. A scope-less compile is render-only: it exists solely so
    /// [`CompiledWorkflow::to_mermaid`] and [`CompiledWorkflow::to_svg`] can shape the
    /// diagram, and the resulting graph is never executed.
    fn build_graph(
        &self,
        scope: Option<&Arc<RunScope>>,
        store: Option<Arc<dyn GraphStore>>,
        max_supersteps: Option<usize>,
    ) -> Result<CompiledGraph> {
        let inner = &self.inner;
        let render_only = scope.is_none();
        let card_id = inner.card.id.clone();
        let mut builder = GraphBuilder::new(card_id.clone());

        for id in inner.definition.component_ids() {
            let entry = inner.definition.component(id).ok_or_else(|| {
                error::compile_error(
                    format!("component '{id}' is not registered").as_str(),
                    &card_id,
                )
            })?;
            let runtime = NodeRuntime {
                id: id.to_owned(),
                component: Arc::clone(&entry.component),
                schemas: NodeSchemas::compile(&entry.node),
                node: entry.node.clone(),
                ability: self.ability_or_default(id),
                is_end: id == inner.end_id.as_str(),
                workflow_id: card_id.clone(),
                scope: scope.cloned(),
            };
            builder = builder.node_with(
                id.to_owned(),
                ComponentNode::new(Arc::new(runtime)),
                NodeOptions {
                    timeout_ms: entry.node.timeout_ms,
                    max_retries: entry.node.max_retries,
                },
            );
        }

        wire_edges(builder, &inner.definition, render_only)
            .entry(inner.start_id.clone())
            .exit(inner.end_id.clone())
            .metadata("workflow_id", json!(card_id))
            .metadata("end_comp_id", json!(inner.end_id))
            .compile_with(CompileOptions {
                store,
                namespace: Some(card_id),
                max_supersteps,
                checkpoint_every_superstep: true,
                ..CompileOptions::default()
            })
    }
}

/// Wires every recorded edge onto `builder`.
///
/// A static or barrier edge into a first-arrival node is skipped: those predecessors
/// activate the node through the dynamic router edges registered by
/// [`wire_conditional`] and by the unconditional-arrival loop below, which leaves its
/// channel without a CNF group.
///
/// `render_only` compiles the *recorded* topology for a diagram instead of a run, so the
/// first-arrival rewiring is skipped and every regular/stream/barrier edge is wired as
/// recorded (`edge`/`barrier_edge`). A dynamic edge targets `"*"`, which the graph
/// renderer skips (`openjiuwen-graph`'s visualize), so the CNF shape is the faithful
/// picture of the builder's edges; the graph is never executed on this path.
fn wire_edges(
    builder: GraphBuilder,
    definition: &WorkflowDefinition,
    render_only: bool,
) -> GraphBuilder {
    let first_arrival = if render_only {
        BTreeSet::new()
    } else {
        first_arrival_nodes(definition)
    };
    let arrivals = first_arrival_arrivals(definition, &first_arrival);
    let mut builder = builder;

    for edge in &definition.edges {
        if first_arrival.contains(&edge.target) {
            continue;
        }
        builder = match edge.kind {
            DefinitionEdgeKind::Barrier => {
                builder.barrier_edge(edge.sources.clone(), edge.target.clone())
            }
            DefinitionEdgeKind::Regular | DefinitionEdgeKind::Stream => {
                edge.sources.iter().fold(builder, |builder, source| {
                    builder.edge(source.clone(), edge.target.clone())
                })
            }
        };
    }

    let mut builder = wire_conditional(builder, definition, &first_arrival, &arrivals);

    // A source that reaches a first-arrival node only through regular/stream/barrier edges
    // is otherwise a source with no selection logic: it always activates every such node.
    for (source, targets) in &arrivals {
        if definition.conditional.contains_key(source) {
            continue;
        }
        builder = builder.conditional_edges(
            source.clone(),
            CombinedRouter {
                from: source.clone(),
                allowed: targets.clone(),
                always: targets.clone(),
                routers: Vec::new(),
            },
        );
    }

    builder
}

/// Wires every source's declared conditional edges.
///
/// A source that also reaches a first-arrival node merges both decisions into one router:
/// the dynamic edge it registers carries the declared routers' results *and* the
/// unconditional first-arrival targets, so no static edge is left on those targets. Its
/// remaining declared targets keep their static conditional edges (and therefore their CNF
/// channels) through `DeclaredEdgesOnly`.
fn wire_conditional(
    builder: GraphBuilder,
    definition: &WorkflowDefinition,
    first_arrival: &BTreeSet<String>,
    arrivals: &BTreeMap<String, Vec<String>>,
) -> GraphBuilder {
    let mut builder = builder;
    for (from, conditional) in &definition.conditional {
        let unconditional = arrivals.get(from).map_or_else(Vec::new, Clone::clone);
        let routes_to_first_arrival = conditional
            .targets
            .iter()
            .any(|target| first_arrival.contains(target));
        if unconditional.is_empty() && !routes_to_first_arrival {
            builder = builder.conditional_edges_to(
                from.clone(),
                conditional.targets.clone(),
                CombinedRouter {
                    from: from.clone(),
                    allowed: conditional.targets.clone(),
                    always: Vec::new(),
                    routers: conditional.routers.clone(),
                },
            );
            continue;
        }

        let mut allowed = conditional.targets.clone();
        for target in &unconditional {
            if !allowed.contains(target) {
                allowed.push(target.clone());
            }
        }
        builder = builder.conditional_edges(
            from.clone(),
            CombinedRouter {
                from: from.clone(),
                allowed,
                always: unconditional,
                routers: conditional.routers.clone(),
            },
        );

        let declared: Vec<String> = conditional
            .targets
            .iter()
            .filter(|target| !first_arrival.contains(*target))
            .cloned()
            .collect();
        if !declared.is_empty() {
            builder = builder.conditional_edges_to(from.clone(), declared, DeclaredEdgesOnly);
        }
    }
    builder
}

/// The node ids compiled with Python's first-arrival fan-in.
///
/// A node qualifies when it recorded `wait_for_all = false` *and* at least two distinct
/// regular/stream/barrier edges feed it. A single inbound source is deliberately excluded:
/// its channel then stays a plain one-source CNF group, byte-identical to a node that
/// never asked for first arrival.
fn first_arrival_nodes(definition: &WorkflowDefinition) -> BTreeSet<String> {
    let mut inbound: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
    for edge in &definition.edges {
        let sources = inbound.entry(edge.target.as_str()).or_default();
        for source in &edge.sources {
            sources.insert(source.as_str());
        }
    }
    definition
        .component_ids()
        .into_iter()
        .filter(|id| {
            definition
                .component(id)
                .is_some_and(|entry| entry.node.wait_for_all == Some(false))
                && inbound.get(id).is_some_and(|sources| sources.len() >= 2)
        })
        .map(str::to_owned)
        .collect()
}

/// The unconditional first-arrival targets of every source, in edge declaration order.
///
/// Only regular/stream/barrier edges contribute: a declared conditional edge into a
/// first-arrival node stays conditional, so its source still reaches the node only when
/// its router selects it.
fn first_arrival_arrivals(
    definition: &WorkflowDefinition,
    first_arrival: &BTreeSet<String>,
) -> BTreeMap<String, Vec<String>> {
    let mut arrivals: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for edge in &definition.edges {
        if !first_arrival.contains(&edge.target) {
            continue;
        }
        for source in &edge.sources {
            let targets = arrivals.entry(source.clone()).or_default();
            if !targets.contains(&edge.target) {
                targets.push(edge.target.clone());
            }
        }
    }
    arrivals
}

/// The single router attached to one conditional source.
///
/// Every router the builder registered for that source runs in registration order; the
/// results are unioned with duplicates removed, so two routers agreeing on a target
/// activate it once. A target the builder did not declare for this source is an error,
/// mirroring the graph crate's "router routed to an unknown node" policy.
///
/// `always` holds the first-arrival targets reached through regular/stream/barrier edges:
/// those activate on every arrival, before any router consults the run state.
struct CombinedRouter {
    /// The source component id, used in error messages.
    from: String,
    /// The targets a declared router may return (declared targets plus `always`).
    allowed: Vec<String>,
    /// The first-arrival targets this source always activates.
    always: Vec<String>,
    /// The routers to run, in registration order.
    routers: Vec<Arc<dyn Router>>,
}

impl std::fmt::Debug for CombinedRouter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CombinedRouter")
            .field("from", &self.from)
            .field("allowed", &self.allowed)
            .field("always", &self.always)
            .field(
                "routers",
                &self
                    .routers
                    .iter()
                    .map(|router| router.name())
                    .collect::<Vec<_>>(),
            )
            .finish()
    }
}

#[async_trait]
impl Router for CombinedRouter {
    fn name(&self) -> &'static str {
        "workflow_combined"
    }

    async fn route(&self, ctx: &RouterContext) -> Result<Vec<String>> {
        let mut targets: Vec<String> = Vec::new();
        for target in &self.always {
            if !targets.contains(target) {
                targets.push(target.clone());
            }
        }
        for router in &self.routers {
            for target in router.route(ctx).await? {
                if !self.allowed.iter().any(|allowed| allowed == &target) {
                    return Err(Error::graph(format!(
                        "router on '{}' returned undeclared target '{}'",
                        self.from, target
                    )));
                }
                if !targets.contains(&target) {
                    targets.push(target);
                }
            }
        }
        Ok(targets)
    }
}

/// A router that activates nothing.
///
/// It exists so [`GraphBuilder::conditional_edges_to`] records a source's declared
/// conditional edges after its real routing decision moved into a merged
/// [`CombinedRouter`]: those edges are what give the declared targets their CNF channels.
/// Returning an empty target list keeps this registration from taking a second routing
/// decision.
struct DeclaredEdgesOnly;

impl std::fmt::Debug for DeclaredEdgesOnly {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("DeclaredEdgesOnly")
    }
}

#[async_trait]
impl Router for DeclaredEdgesOnly {
    fn name(&self) -> &'static str {
        "workflow_declared_edges"
    }

    async fn route(&self, _ctx: &RouterContext) -> Result<Vec<String>> {
        Ok(Vec::new())
    }
}

/// Fakes shared by the engine's unit tests.
#[cfg(test)]
pub(crate) mod test_support {
    use std::sync::Arc;

    use async_trait::async_trait;
    use futures::stream::BoxStream;
    use futures::StreamExt;
    use openjiuwen_common::error::Result;
    use openjiuwen_graph::Router;
    use serde_json::{json, Value};

    use crate::ability::ComponentAbility;
    use crate::card::WorkflowCard;
    use crate::component::{Component, ComponentCard};
    use crate::components::BranchRouterAdapter;
    use crate::condition::BranchRouter;
    use crate::context::ComponentContext;

    /// A component whose type and abilities are fixed by the test.
    #[derive(Clone)]
    struct FakeComponent {
        ty: &'static str,
        abilities: Vec<ComponentAbility>,
    }

    #[async_trait]
    impl Component for FakeComponent {
        fn card(&self) -> ComponentCard {
            ComponentCard::new(self.ty, self.ty).with_abilities(self.abilities.clone())
        }

        fn abilities(&self) -> Vec<ComponentAbility> {
            self.abilities.clone()
        }

        fn component_type(&self) -> &'static str {
            self.ty
        }
    }

    /// A component of type `ty` implementing exactly `abilities`.
    #[must_use]
    pub(crate) fn fake(ty: &'static str, abilities: Vec<ComponentAbility>) -> Arc<dyn Component> {
        Arc::new(FakeComponent { ty, abilities })
    }

    /// A graph router with no branches, enough to register a conditional edge.
    #[must_use]
    pub(crate) fn router() -> Arc<dyn Router> {
        Arc::new(BranchRouterAdapter::new(Arc::new(BranchRouter::new())))
    }

    /// The workflow card every unit test uses.
    #[must_use]
    pub(crate) fn card() -> WorkflowCard {
        WorkflowCard::new("wf")
    }

    /// A test component: echoes its inputs, can stream, collect, or ask a question.
    #[derive(Clone, Default)]
    pub(crate) struct TestComponent {
        /// Frames a `stream`/`transform` dispatches, in order.
        pub(crate) frames: Vec<Value>,
        /// Abilities the component declares.
        pub(crate) abilities: Vec<ComponentAbility>,
        /// A question `invoke` asks through the context (interrupting without an answer).
        pub(crate) question: Option<Value>,
        /// The targets `route` returns when set.
        pub(crate) targets: Option<Vec<String>>,
        /// How long `invoke` waits before answering.
        pub(crate) delay: Option<std::time::Duration>,
        /// A custom frame `invoke` publishes through the context.
        pub(crate) custom: Option<Value>,
        /// The reason `invoke` fails with, when set.
        pub(crate) failure: Option<String>,
    }

    impl TestComponent {
        /// A component that implements `abilities` and echoes its inputs.
        #[must_use]
        pub(crate) fn new(abilities: Vec<ComponentAbility>) -> Self {
            Self {
                abilities,
                ..Self::default()
            }
        }

        /// Yield `frames` from `stream`/`transform`.
        #[must_use]
        pub(crate) fn with_frames(mut self, frames: Vec<Value>) -> Self {
            self.frames = frames;
            self
        }

        /// Ask `question` from `invoke`.
        #[must_use]
        pub(crate) fn with_question(mut self, question: Value) -> Self {
            self.question = Some(question);
            self
        }

        /// Let `route` return `targets`.
        #[must_use]
        pub(crate) fn with_targets(mut self, targets: Vec<String>) -> Self {
            self.targets = Some(targets);
            self
        }

        /// Wait `delay` in `invoke` before answering.
        #[must_use]
        pub(crate) fn with_delay(mut self, delay: std::time::Duration) -> Self {
            self.delay = Some(delay);
            self
        }

        /// Publish `payload` as a custom frame from `invoke`.
        #[must_use]
        pub(crate) fn with_custom(mut self, payload: Value) -> Self {
            self.custom = Some(payload);
            self
        }

        /// Fail `invoke` with `reason`.
        #[must_use]
        pub(crate) fn with_failure(mut self, reason: &str) -> Self {
            self.failure = Some(reason.to_owned());
            self
        }

        /// Wraps the component in its `Arc`.
        #[must_use]
        pub(crate) fn shared(self) -> Arc<dyn Component> {
            Arc::new(self)
        }
    }

    #[async_trait]
    impl Component for TestComponent {
        fn card(&self) -> ComponentCard {
            ComponentCard::new("test", "test").with_abilities(self.abilities.clone())
        }

        fn abilities(&self) -> Vec<ComponentAbility> {
            self.abilities.clone()
        }

        fn component_type(&self) -> &'static str {
            "test"
        }

        async fn invoke(&self, inputs: Value, ctx: &ComponentContext) -> Result<Value> {
            if let Some(delay) = self.delay {
                tokio::time::sleep(delay).await;
            }
            if let Some(reason) = self.failure.as_deref() {
                return Err(openjiuwen_common::error::Error::workflow(reason));
            }
            if let Some(payload) = self.custom.clone() {
                ctx.write_custom_stream(payload).await?;
            }
            if let Some(question) = self.question.clone() {
                let answer = ctx.interact(question).await?;
                return Ok(json!({"answer": answer}));
            }
            Ok(json!({"echo": inputs}))
        }

        async fn collect(&self, inputs: Value, _ctx: &ComponentContext) -> Result<Value> {
            Ok(json!({"collected": inputs}))
        }

        fn stream(
            self: Arc<Self>,
            _inputs: Value,
            _ctx: Arc<ComponentContext>,
        ) -> BoxStream<'static, Result<Value>> {
            futures::stream::iter(self.frames.clone().into_iter().map(Ok)).boxed()
        }

        fn transform(
            self: Arc<Self>,
            _inputs: Value,
            _ctx: Arc<ComponentContext>,
        ) -> BoxStream<'static, Result<Value>> {
            futures::stream::iter(self.frames.clone().into_iter().map(Ok)).boxed()
        }

        fn route(&self, _source: &Value) -> Option<Result<Vec<String>>> {
            self.targets.clone().map(Ok)
        }
    }

    /// Counts its invocations in a shared counter and fails the first `failures` of them.
    ///
    /// Resume tests need a component that behaves differently on a second run: the counter
    /// proves which nodes the resumed run re-executed.
    #[derive(Clone)]
    pub(crate) struct CountingComponent {
        /// Incremented once per `invoke`.
        calls: Arc<std::sync::atomic::AtomicUsize>,
        /// Failures before the component starts succeeding.
        failures: usize,
    }

    impl CountingComponent {
        /// A component failing its first `failures` calls and recording them in `calls`.
        #[must_use]
        pub(crate) fn new(calls: Arc<std::sync::atomic::AtomicUsize>, failures: usize) -> Self {
            Self { calls, failures }
        }

        /// Wraps the component in its `Arc`.
        #[must_use]
        pub(crate) fn shared(self) -> Arc<dyn Component> {
            Arc::new(self)
        }
    }

    #[async_trait]
    impl Component for CountingComponent {
        fn card(&self) -> ComponentCard {
            ComponentCard::new("counting", "counting")
                .with_abilities(vec![ComponentAbility::Invoke])
        }

        fn abilities(&self) -> Vec<ComponentAbility> {
            vec![ComponentAbility::Invoke]
        }

        fn component_type(&self) -> &'static str {
            "counting"
        }

        async fn invoke(&self, inputs: Value, _ctx: &ComponentContext) -> Result<Value> {
            let call = self
                .calls
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
                .saturating_add(1);
            if call <= self.failures {
                return Err(openjiuwen_common::error::Error::workflow(format!(
                    "attempt {call} failed"
                )));
            }
            Ok(json!({"call": call, "echo": inputs}))
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::assert_is_empty)]
mod tests;
