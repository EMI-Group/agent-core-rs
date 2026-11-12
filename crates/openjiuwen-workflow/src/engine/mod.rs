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
pub mod node;
mod run;
pub mod validate;

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use openjiuwen_graph::{
    CompileOptions, CompiledGraph, GraphBuilder, GraphStore, NodeOptions, Router, RouterContext,
};
use openjiuwen_session::stream::StreamMode;
use openjiuwen_session::InteractiveInput;
use serde_json::json;
use tokio_util::sync::CancellationToken;

use openjiuwen_common::error::{Error, Result};

use crate::ability::ComponentAbility;
use crate::card::{WorkflowCard, WorkflowConfig, WorkflowSpec};
use crate::engine::definition::EdgeKind as DefinitionEdgeKind;
use crate::engine::node::{ComponentNode, NodeRuntime, RunScope};
use crate::error;

pub use definition::{ConditionalSource, Edge, EdgeKind, Entry, NodeConfig, WorkflowDefinition};

/// Tunables for one workflow run.
///
/// Every field has a permissive default, so [`WorkflowRunOptions::default`] runs to
/// completion with no cancellation, no explicit timeout, and every stream mode.
#[derive(Clone, Debug)]
pub struct WorkflowRunOptions {
    /// Cancels the run when triggered.
    pub cancellation: CancellationToken,
    /// Whole-run timeout; `None` falls back to `SessionConfig::execute_timeout_ms`.
    pub timeout: Option<Duration>,
    /// Answers seeded into the run's interaction state (used by `resume`).
    pub interactive: Option<InteractiveInput>,
    /// Continue from the graph snapshot stored for this session.
    pub resume: bool,
    /// Which frame modes `stream` yields; `None` yields every mode.
    pub stream_modes: Option<Vec<StreamMode>>,
    /// Maximum number of supersteps; `None` uses the graph default.
    pub max_supersteps: Option<usize>,
}

impl Default for WorkflowRunOptions {
    fn default() -> Self {
        Self {
            cancellation: CancellationToken::new(),
            timeout: None,
            interactive: None,
            resume: false,
            stream_modes: None,
            max_supersteps: None,
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
    /// Compilation for rendering has no run scope and no snapshot store; a definition
    /// that cannot be compiled yields a Mermaid comment naming the problem instead of
    /// panicking.
    #[must_use]
    pub fn to_mermaid(&self) -> String {
        match self.build_graph(None, None, None) {
            Ok(graph) => graph.to_mermaid(),
            Err(failure) => {
                tracing::warn!(
                    workflow = self.inner.card.id.as_str(),
                    error = %failure,
                    "failed to compile the workflow graph for rendering"
                );
                format!(
                    "%% workflow '{}' could not be compiled: {failure}",
                    self.inner.card.id
                )
            }
        }
    }

    /// The ability recorded for `id`, defaulting to `Invoke`.
    fn ability_or_default(&self, id: &str) -> ComponentAbility {
        self.inner.abilities.get(id).copied().unwrap_or_default()
    }

    /// Compiles a fresh graph for one run (or for rendering, when `scope` is `None`).
    ///
    /// Node instances carry the run's interaction state and frame writer, so a graph is
    /// never reused across runs.
    fn build_graph(
        &self,
        scope: Option<&Arc<RunScope>>,
        store: Option<Arc<dyn GraphStore>>,
        max_supersteps: Option<usize>,
    ) -> Result<CompiledGraph> {
        let inner = &self.inner;
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

        for edge in &inner.definition.edges {
            match edge.kind {
                DefinitionEdgeKind::Barrier => {
                    builder = builder.barrier_edge(edge.sources.clone(), edge.target.clone());
                }
                DefinitionEdgeKind::Regular | DefinitionEdgeKind::Stream => {
                    for source in &edge.sources {
                        builder = builder.edge(source.clone(), edge.target.clone());
                    }
                }
            }
        }

        for (from, conditional) in &inner.definition.conditional {
            builder = builder.conditional_edges_to(
                from.clone(),
                conditional.targets.clone(),
                CombinedRouter {
                    from: from.clone(),
                    allowed: conditional.targets.clone(),
                    routers: conditional.routers.clone(),
                },
            );
        }

        builder
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

/// The single router attached to one conditional source.
///
/// Every router the builder registered for that source runs in registration order; the
/// results are unioned with duplicates removed, so two routers agreeing on a target
/// activate it once. A target the builder did not declare for this source is an error,
/// mirroring the graph crate's "router routed to an unknown node" policy.
struct CombinedRouter {
    /// The source component id, used in error messages.
    from: String,
    /// The targets the builder declared for this source.
    allowed: Vec<String>,
    /// The routers to run, in registration order.
    routers: Vec<Arc<dyn Router>>,
}

impl std::fmt::Debug for CombinedRouter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CombinedRouter")
            .field("from", &self.from)
            .field("allowed", &self.allowed)
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
