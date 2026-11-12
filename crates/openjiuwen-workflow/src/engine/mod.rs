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
//! [`WorkflowOutput`] or a frame stream.
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
        /// Ask `question` from `invoke`.
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
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::assert_is_empty)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use futures::StreamExt;
    use openjiuwen_common::error::ErrorKind;
    use openjiuwen_session::stream::StreamMode;
    use openjiuwen_session::{InteractiveInput, Session, WorkflowSession};
    use serde_json::json;

    use super::test_support::TestComponent;
    use super::*;
    use crate::components::{BranchRouterAdapter, EndComponent, StartComponent};
    use crate::condition::{Branch, BranchRouter, Condition};
    use crate::output::{WorkflowExecutionState, END_NODE_STREAM, WORKFLOW_FINAL};

    fn session(name: &str) -> Arc<dyn Session> {
        Arc::new(WorkflowSession::new(name))
    }

    fn echo() -> TestComponent {
        TestComponent::new(vec![ComponentAbility::Invoke])
    }

    /// Records the conventional `start`/`end` component ids.
    fn with_endpoints(mut definition: WorkflowDefinition) -> WorkflowDefinition {
        definition.start = Some("start".to_owned());
        definition.end = Some("end".to_owned());
        definition
    }

    /// `start -> mid -> end`, where `mid` is the component under test.
    fn linear(mid: TestComponent, end: NodeConfig) -> WorkflowDefinition {
        let mut definition = WorkflowDefinition::new(test_support::card());
        definition
            .insert_component(
                "start",
                Arc::new(StartComponent::new()),
                NodeConfig::default(),
            )
            .insert_component("mid", mid.shared(), NodeConfig::default())
            .insert_component("end", Arc::new(EndComponent::default()), end)
            .add_edge(
                vec!["start".to_owned()],
                "mid".to_owned(),
                DefinitionEdgeKind::Regular,
            )
            .add_edge(
                vec!["mid".to_owned()],
                "end".to_owned(),
                DefinitionEdgeKind::Regular,
            );
        with_endpoints(definition)
    }

    /// An end node whose declared ability is `Stream`.
    fn streaming_end() -> NodeConfig {
        NodeConfig {
            abilities: Some(vec![ComponentAbility::Stream]),
            ..NodeConfig::default()
        }
    }

    #[test]
    fn build_reports_validation_problems() {
        let mut definition = WorkflowDefinition::new(test_support::card());
        definition.insert_component(
            "start",
            Arc::new(StartComponent::new()),
            NodeConfig::default(),
        );
        let error = CompiledWorkflow::build(definition).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Workflow);
        assert!(!error.message().is_empty());
    }

    #[test]
    fn build_exposes_the_card_the_spec_and_the_abilities() {
        let workflow = CompiledWorkflow::build(linear(echo(), NodeConfig::default())).unwrap();

        assert_eq!(workflow.card().id, "wf");
        assert_eq!(workflow.start_comp_id(), "start");
        assert_eq!(workflow.end_comp_id(), "end");
        assert_eq!(workflow.spec().start_nodes, vec!["start".to_owned()]);
        assert!(workflow.spec().comp_configs.contains_key("mid"));
        assert!(!workflow.is_streaming());
        assert_eq!(workflow.ability("mid"), Some(ComponentAbility::Invoke));
        assert_eq!(workflow.ability("missing"), None);
        assert_eq!(workflow.config().card.id, "wf");
        assert_eq!(workflow.config().spec.start_nodes, vec!["start".to_owned()]);
        assert_eq!(
            workflow.definition().component_ids(),
            vec!["start", "mid", "end"]
        );
        assert!(format!("{workflow:?}").contains("CompiledWorkflow"));
    }

    #[test]
    fn a_streaming_end_makes_the_workflow_streaming() {
        let workflow = CompiledWorkflow::build(linear(echo(), streaming_end())).unwrap();
        assert!(workflow.is_streaming());
        assert_eq!(workflow.ability("end"), Some(ComponentAbility::Stream));
    }

    #[tokio::test]
    async fn invoke_runs_a_linear_workflow() {
        let session = session("linear");
        let end = NodeConfig {
            inputs_schema: Some(json!({"answer": "${mid.echo.query}"})),
            ..NodeConfig::default()
        };
        let workflow = CompiledWorkflow::build(linear(echo(), end)).unwrap();

        let output = workflow
            .invoke(
                json!({"query": "hi"}),
                Arc::clone(&session),
                WorkflowRunOptions::default(),
            )
            .await
            .unwrap();

        assert_eq!(output.state, WorkflowExecutionState::Completed);
        assert_eq!(output.result, json!({"output": {"answer": "hi"}}));
        // Python's `commit_user_inputs`: the inputs landed in the global lane.
        assert_eq!(session.get_state("query").await.unwrap(), Some(json!("hi")));
    }

    #[tokio::test]
    async fn root_and_node_references_both_resolve() {
        let session = session("references");
        let mid = TestComponent {
            abilities: vec![ComponentAbility::Invoke],
            ..TestComponent::new(vec![ComponentAbility::Invoke])
        };
        let mut definition = linear(mid, NodeConfig::default());
        definition.insert_component(
            "mid",
            echo().shared(),
            NodeConfig {
                inputs_schema: Some(json!({
                    "root": "${query}",
                    "node": "${start.query}",
                    "missing": "${nope.x}",
                })),
                ..NodeConfig::default()
            },
        );
        let workflow = CompiledWorkflow::build(definition).unwrap();

        let output = workflow
            .invoke(
                json!({"query": "hi"}),
                Arc::clone(&session),
                WorkflowRunOptions::default(),
            )
            .await
            .unwrap();

        let node = output
            .result
            .get("output")
            .and_then(|state| state.get("mid"))
            .unwrap();
        assert_eq!(
            node,
            &json!({"echo": {"root": "hi", "node": "hi", "missing": null}})
        );
    }

    #[tokio::test]
    async fn non_object_inputs_are_not_merged_into_the_session() {
        let session = session("scalars");
        let workflow = CompiledWorkflow::build(linear(echo(), NodeConfig::default())).unwrap();
        let output = workflow
            .invoke(
                json!("plain"),
                Arc::clone(&session),
                WorkflowRunOptions::default(),
            )
            .await
            .unwrap();
        assert_eq!(output.state, WorkflowExecutionState::Completed);
        assert_eq!(session.get_state("query").await.unwrap(), None);
        assert_eq!(
            output.result.get("output").unwrap().get("inputs"),
            Some(&json!("plain"))
        );
    }

    #[tokio::test]
    async fn an_interrupt_reports_input_required_with_the_wires() {
        let session = session("interrupt");
        let mid = echo().with_question(json!({"id": "ask", "question": "go?"}));
        let workflow = CompiledWorkflow::build(linear(mid, NodeConfig::default())).unwrap();

        let output = workflow
            .invoke(
                json!({"query": "hi"}),
                Arc::clone(&session),
                WorkflowRunOptions::default(),
            )
            .await
            .unwrap();

        assert_eq!(output.state, WorkflowExecutionState::InputRequired);
        assert_eq!(
            output.result,
            json!([{
                "type": "__interaction__",
                "index": 0,
                "payload": {"id": "ask", "index": 0, "question": {"id": "ask", "question": "go?"}}
            }])
        );
    }

    #[tokio::test]
    async fn invoke_reports_cancellation_and_timeouts() {
        let session = session("timeouts");
        let workflow = CompiledWorkflow::build(linear(
            echo().with_delay(Duration::from_secs(30)),
            NodeConfig::default(),
        ))
        .unwrap();

        let token = CancellationToken::new();
        token.cancel();
        let cancelled = workflow
            .invoke(
                json!({"query": "hi"}),
                Arc::clone(&session),
                WorkflowRunOptions::default().with_cancellation(token),
            )
            .await
            .unwrap_err();
        assert_eq!(cancelled.kind(), ErrorKind::Cancelled);

        let timed_out = workflow
            .invoke(
                json!({"query": "hi"}),
                Arc::clone(&session),
                WorkflowRunOptions::default().with_timeout(Duration::from_millis(10)),
            )
            .await
            .unwrap_err();
        assert_eq!(timed_out.kind(), ErrorKind::Timeout);
        assert!(timed_out.to_string().contains("execution timed out"));
    }

    #[tokio::test]
    async fn resume_requires_a_checkpointer() {
        let workflow = CompiledWorkflow::build(linear(echo(), NodeConfig::default())).unwrap();
        let error = workflow
            .resume(
                InteractiveInput::new(),
                session("no-checkpointer"),
                WorkflowRunOptions::default(),
            )
            .await
            .unwrap_err();
        assert_eq!(
            error.message(),
            "interactive resume requested without a checkpointer"
        );
    }

    #[tokio::test]
    async fn stream_yields_end_frames_then_the_final_output() {
        let workflow = CompiledWorkflow::build(linear(echo(), streaming_end())).unwrap();
        let mut stream = workflow.stream(
            json!({"query": "hi"}),
            session("stream"),
            WorkflowRunOptions::default(),
        );

        let mut chunks = Vec::new();
        while let Some(chunk) = stream.next().await {
            chunks.push(chunk.unwrap());
        }
        assert!(!chunks.is_empty());

        let final_chunk = chunks.pop().unwrap();
        assert_eq!(final_chunk.item_type, WORKFLOW_FINAL);
        assert_eq!(final_chunk.mode, StreamMode::Output);
        assert_eq!(final_chunk.index, Some(0));

        assert!(!chunks.is_empty());
        for (index, chunk) in chunks.iter().enumerate() {
            assert_eq!(chunk.item_type, END_NODE_STREAM);
            assert_eq!(chunk.mode, StreamMode::Output);
            assert_eq!(chunk.index, Some(i64::try_from(index).unwrap()));
        }
        // The final payload is the collected `end node stream` wire list.
        assert_eq!(final_chunk.payload.as_array().unwrap().len(), chunks.len());
        assert_eq!(final_chunk.payload[0]["type"], json!(END_NODE_STREAM));
        assert_eq!(final_chunk.payload[0]["index"], json!(0));
    }

    #[tokio::test]
    async fn stream_forwards_custom_frames_and_filters_modes() {
        let workflow = CompiledWorkflow::build(linear(
            echo().with_custom(json!({"progress": 1})),
            NodeConfig::default(),
        ))
        .unwrap();

        let mut stream = workflow.stream(
            json!({"query": "hi"}),
            session("custom"),
            WorkflowRunOptions::default().with_stream_modes(vec![StreamMode::Custom]),
        );
        let mut chunks = Vec::new();
        while let Some(chunk) = stream.next().await {
            chunks.push(chunk.unwrap());
        }

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].mode, StreamMode::Custom);
        assert_eq!(chunks[0].payload, json!({"progress": 1}));
    }

    #[tokio::test]
    async fn stream_yields_exactly_one_error() {
        let workflow =
            CompiledWorkflow::build(linear(echo().with_failure("kaboom"), NodeConfig::default()))
                .unwrap();
        let mut stream = workflow.stream(
            json!({"query": "hi"}),
            session("failure"),
            WorkflowRunOptions::default(),
        );

        let mut chunks = Vec::new();
        while let Some(chunk) = stream.next().await {
            chunks.push(chunk);
        }
        assert_eq!(chunks.len(), 1);
        let error = chunks.pop().unwrap().unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Workflow);
        assert!(error.to_string().contains("kaboom"), "{error}");
    }

    #[tokio::test]
    async fn stream_ends_without_a_final_frame_when_the_run_interrupts() {
        let workflow = CompiledWorkflow::build(linear(
            echo().with_question(json!({"id": "ask", "question": "go?"})),
            NodeConfig::default(),
        ))
        .unwrap();
        let mut stream = workflow.stream(
            json!({"query": "hi"}),
            session("stream-interrupt"),
            WorkflowRunOptions::default(),
        );

        let mut chunks = Vec::new();
        while let Some(chunk) = stream.next().await {
            chunks.push(chunk.unwrap());
        }
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].item_type, "__interaction__");
        assert_eq!(chunks[0].payload["id"], json!("ask"));
    }

    #[tokio::test]
    async fn conditional_routers_select_targets_and_union_their_results() {
        let mut definition = WorkflowDefinition::new(test_support::card());
        definition
            .insert_component(
                "start",
                Arc::new(StartComponent::new()),
                NodeConfig::default(),
            )
            .insert_component("mid", echo().shared(), NodeConfig::default())
            .insert_component("a", echo().shared(), NodeConfig::default())
            .insert_component("b", echo().shared(), NodeConfig::default())
            .insert_component(
                "end",
                Arc::new(EndComponent::default()),
                NodeConfig::default(),
            )
            .add_edge(
                vec!["start".to_owned()],
                "mid".to_owned(),
                DefinitionEdgeKind::Regular,
            )
            .add_edge(
                vec!["a".to_owned()],
                "end".to_owned(),
                DefinitionEdgeKind::Regular,
            );

        let mut router = BranchRouter::new();
        let _ = router.add_branch(Branch::new(
            Condition::new("${query} == 'hi'").unwrap(),
            vec!["a".to_owned()],
        ));
        definition.add_conditional(
            "mid",
            "a",
            Arc::new(BranchRouterAdapter::new(Arc::new(router))),
        );
        let workflow = CompiledWorkflow::build(with_endpoints(definition)).unwrap();

        let output = workflow
            .invoke(
                json!({"query": "hi"}),
                session("routing"),
                WorkflowRunOptions::default(),
            )
            .await
            .unwrap();
        let state = output.result.get("output").unwrap();
        assert!(state.get("a").is_some(), "{state}");
        assert!(state.get("b").is_none(), "{state}");
    }

    #[tokio::test]
    async fn an_undeclared_router_target_is_a_run_error() {
        let mut definition = WorkflowDefinition::new(test_support::card());
        definition
            .insert_component(
                "start",
                Arc::new(StartComponent::new()),
                NodeConfig::default(),
            )
            .insert_component("mid", echo().shared(), NodeConfig::default())
            .insert_component("a", echo().shared(), NodeConfig::default())
            .insert_component(
                "end",
                Arc::new(EndComponent::default()),
                NodeConfig::default(),
            )
            .add_edge(
                vec!["start".to_owned()],
                "mid".to_owned(),
                DefinitionEdgeKind::Regular,
            )
            .add_edge(
                vec!["a".to_owned()],
                "end".to_owned(),
                DefinitionEdgeKind::Regular,
            );

        let mut router = BranchRouter::new();
        let _ = router.add_branch(Branch::new(
            Condition::new("${query} == 'hi'").unwrap(),
            vec!["b".to_owned()],
        ));
        definition.add_conditional(
            "mid",
            "a",
            Arc::new(BranchRouterAdapter::new(Arc::new(router))),
        );
        let workflow = CompiledWorkflow::build(with_endpoints(definition)).unwrap();

        let error = workflow
            .invoke(
                json!({"query": "hi"}),
                session("undeclared"),
                WorkflowRunOptions::default(),
            )
            .await
            .unwrap_err();
        assert!(
            error.to_string().contains("undeclared target 'b'"),
            "{error}"
        );
    }

    #[tokio::test]
    async fn component_routing_short_circuits_the_routers() {
        let mut definition = WorkflowDefinition::new(test_support::card());
        definition
            .insert_component(
                "start",
                Arc::new(StartComponent::new()),
                NodeConfig::default(),
            )
            .insert_component(
                "mid",
                echo().with_targets(vec!["b".to_owned()]).shared(),
                NodeConfig::default(),
            )
            .insert_component("a", echo().shared(), NodeConfig::default())
            .insert_component("b", echo().shared(), NodeConfig::default())
            .insert_component(
                "end",
                Arc::new(EndComponent::default()),
                NodeConfig::default(),
            )
            .add_edge(
                vec!["start".to_owned()],
                "mid".to_owned(),
                DefinitionEdgeKind::Regular,
            )
            .add_edge(
                vec!["b".to_owned()],
                "end".to_owned(),
                DefinitionEdgeKind::Regular,
            );
        // The router below would pick `a`; the component's own `route` wins and picks
        // `b`, which is the only branch that reaches `end`.
        let mut router = BranchRouter::new();
        let _ = router.add_branch(Branch::new(
            Condition::new("${query} == 'hi'").unwrap(),
            vec!["a".to_owned()],
        ));
        definition
            .add_conditional(
                "mid",
                "a",
                Arc::new(BranchRouterAdapter::new(Arc::new(router))),
            )
            .add_conditional("mid", "b", test_support::router());
        let workflow = CompiledWorkflow::build(with_endpoints(definition)).unwrap();

        let output = workflow
            .invoke(
                json!({"query": "hi"}),
                session("component-route"),
                WorkflowRunOptions::default(),
            )
            .await
            .unwrap();
        let state = output.result.get("output").unwrap();
        assert!(state.get("b").is_some(), "{state}");
        assert!(state.get("a").is_none(), "{state}");
    }

    #[test]
    fn to_mermaid_renders_nodes_and_edges() {
        let workflow = CompiledWorkflow::build(linear(echo(), NodeConfig::default())).unwrap();
        let rendered = workflow.to_mermaid();
        assert!(rendered.contains("start"), "{rendered}");
        assert!(rendered.contains("mid"), "{rendered}");
        assert!(rendered.contains("end"), "{rendered}");
    }

    #[test]
    fn run_options_are_chainable_and_permissive() {
        let options = WorkflowRunOptions::new();
        assert!(!options.resume);
        assert!(options.timeout.is_none());
        assert!(options.interactive.is_none());
        assert!(options.stream_modes.is_none());
        assert!(options.max_supersteps.is_none());
        assert!(!options.cancellation.is_cancelled());

        let token = CancellationToken::new();
        token.cancel();
        let mut interactive = InteractiveInput::new();
        interactive.answer("ask", json!("yes"));
        let configured = WorkflowRunOptions::default()
            .with_cancellation(token.clone())
            .with_timeout(Duration::from_secs(1))
            .with_interactive(interactive)
            .with_resume(true)
            .with_stream_modes(vec![StreamMode::Trace])
            .with_max_supersteps(3);
        assert!(configured.cancellation.is_cancelled());
        assert_eq!(configured.timeout, Some(Duration::from_secs(1)));
        assert!(configured.interactive.is_some());
        assert!(configured.resume);
        assert_eq!(configured.stream_modes, Some(vec![StreamMode::Trace]));
        assert_eq!(configured.max_supersteps, Some(3));
    }
}
