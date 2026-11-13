//! The `Workflow` builder.
//!
//! [`Workflow`] is the mutable front end Python exposes as `core.workflow.Workflow`: it
//! records components (with their [`ComponentOptions`]), edges, and conditional fan-out
//! into a [`WorkflowDefinition`], and compiles that definition on demand.
//!
//! Builder methods never return `Err`. Problems the builder can already see — a duplicate
//! component id, a second start/end, a barrier edge without sources — are recorded on the
//! definition and reported the next time the workflow is compiled (`build`, `invoke`,
//! `stream`, `to_mermaid`). Everything else (id syntax, edge endpoints, capability
//! mismatches) is [`validate`](crate::engine::validate::validate)'s job.
//!
//! ```
//! use openjiuwen_workflow::{
//!     ComponentOptions, EndComponent, StartComponent, Workflow, WorkflowCard,
//! };
//!
//! let mut workflow = Workflow::new(WorkflowCard::new("demo"));
//! workflow
//!     .set_start_comp("start", StartComponent::new(), ComponentOptions::new())
//!     .set_end_comp("end", EndComponent::new(), ComponentOptions::new())
//!     .add_connection("start", "end");
//! let compiled = workflow.build().unwrap();
//! assert_eq!(compiled.start_comp_id(), "start");
//! ```

use std::sync::Arc;

use futures::stream::{self, BoxStream, StreamExt};
use openjiuwen_common::error::Result;
use openjiuwen_graph::Router;
use openjiuwen_session::{InteractiveInput, Session};
use serde_json::Value;

use crate::ability::ComponentAbility;
use crate::card::{ExceptionConfig, WorkflowCard, WorkflowConfig, WorkflowSpec};
use crate::component::Component;
use crate::engine::definition::{EdgeKind, NodeConfig, WorkflowDefinition};
use crate::engine::{CompiledWorkflow, WorkflowRunOptions};
use crate::output::{WorkflowChunk, WorkflowOutput};
use crate::render::DrawFormat;

/// The per-component configuration [`Workflow`] records for one registration.
///
/// Mirrors the keyword arguments of Python's `Workflow.add_workflow_comp`: the batch and
/// stream schema pairs, an explicit ability override, the fan-in flag, the retry and
/// timeout limits, and a display name. Every setter consumes `self` so registrations
/// chain:
///
/// ```
/// use openjiuwen_workflow::ComponentOptions;
/// use serde_json::json;
///
/// let options = ComponentOptions::new()
///     .with_inputs_schema(json!({"query": "${start.query}"}))
///     .with_max_retries(2)
///     .with_name("Summarise");
/// assert_eq!(options.max_retries, 2);
/// ```
#[derive(Debug, Clone, Default)]
pub struct ComponentOptions {
    /// The JSON template resolved against workflow state for batch inputs.
    pub inputs_schema: Option<Value>,
    /// The JSON template describing batch outputs.
    pub outputs_schema: Option<Value>,
    /// The JSON template resolved against workflow state for streamed inputs.
    pub stream_inputs_schema: Option<Value>,
    /// The JSON template describing streamed outputs.
    pub stream_outputs_schema: Option<Value>,
    /// An explicit ability override; `None` infers the ability from the edge topology.
    pub abilities: Option<Vec<ComponentAbility>>,
    /// Whether to wait for every predecessor; the graph's barriers always do.
    pub wait_for_all: Option<bool>,
    /// How many times a failing invocation is retried (Python's default is 0).
    pub max_retries: u32,
    /// The per-node timeout in milliseconds; `None` means no timeout.
    pub timeout_ms: Option<u64>,
    /// How a node's failure is handled after every retry (Python's `exception_config`).
    pub exception_config: Option<ExceptionConfig>,
    /// An optional display name used by tracing and the spec.
    pub name: Option<String>,
}

impl ComponentOptions {
    /// Options with every default: no schemas, no ability override, no retries, no
    /// timeout, and no name.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the batch input template.
    #[must_use]
    pub fn with_inputs_schema(mut self, inputs_schema: Value) -> Self {
        self.inputs_schema = Some(inputs_schema);
        self
    }

    /// Sets the batch output template.
    #[must_use]
    pub fn with_outputs_schema(mut self, outputs_schema: Value) -> Self {
        self.outputs_schema = Some(outputs_schema);
        self
    }

    /// Sets the stream input template.
    #[must_use]
    pub fn with_stream_inputs_schema(mut self, stream_inputs_schema: Value) -> Self {
        self.stream_inputs_schema = Some(stream_inputs_schema);
        self
    }

    /// Sets the stream output template.
    #[must_use]
    pub fn with_stream_outputs_schema(mut self, stream_outputs_schema: Value) -> Self {
        self.stream_outputs_schema = Some(stream_outputs_schema);
        self
    }

    /// Overrides the abilities the engine may drive this component with.
    #[must_use]
    pub fn with_abilities(mut self, abilities: Vec<ComponentAbility>) -> Self {
        self.abilities = Some(abilities);
        self
    }

    /// Sets the fan-in flag (accepted for compatibility; see the crate docs).
    #[must_use]
    pub fn with_wait_for_all(mut self, wait_for_all: bool) -> Self {
        self.wait_for_all = Some(wait_for_all);
        self
    }

    /// Sets how often a failing invocation is retried.
    #[must_use]
    pub fn with_max_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = max_retries;
        self
    }

    /// Sets the per-node timeout in milliseconds.
    #[must_use]
    pub fn with_timeout_ms(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = Some(timeout_ms);
        self
    }

    /// Sets the display name.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Sets the failure policy applied after every retry is exhausted.
    #[must_use]
    pub fn with_exception_config(mut self, exception_config: Option<ExceptionConfig>) -> Self {
        self.exception_config = exception_config;
        self
    }

    /// Marks a streaming registration (Python's `response_mode="streaming"`).
    ///
    /// The abilities become `[Invoke, Stream]`, plus `Transform` when a stream input
    /// schema is set (the node then consumes a stream too). A derived set containing
    /// `Transform` implies `wait_for_all = true`; otherwise the flag keeps whatever the
    /// caller set (by default `None`, i.e. Python's first-arrival behaviour is not
    /// requested).
    #[must_use]
    pub fn streaming(mut self) -> Self {
        let mut abilities = vec![ComponentAbility::Invoke, ComponentAbility::Stream];
        if self.stream_inputs_schema.is_some() {
            abilities.push(ComponentAbility::Transform);
            self.wait_for_all = Some(true);
        }
        self.abilities = Some(abilities);
        self
    }

    /// Converts these options into the node configuration the engine stores.
    pub(crate) fn into_node_config(self) -> NodeConfig {
        NodeConfig {
            inputs_schema: self.inputs_schema,
            outputs_schema: self.outputs_schema,
            stream_inputs_schema: self.stream_inputs_schema,
            stream_outputs_schema: self.stream_outputs_schema,
            abilities: self.abilities,
            wait_for_all: self.wait_for_all,
            max_retries: self.max_retries,
            timeout_ms: self.timeout_ms,
            exception_config: self.exception_config,
            name: self.name,
        }
    }
}

/// A workflow under construction: components, edges, and the start/end choice.
///
/// The builder records everything in a [`WorkflowDefinition`] and never fails; compilation
/// happens in [`Workflow::build`] (or implicitly in `invoke` / `stream`). A `Workflow` is
/// not mutated by a run, so one instance may be invoked concurrently.
///
/// ```
/// use openjiuwen_workflow::{
///     ComponentOptions, EndComponent, StartComponent, Workflow, WorkflowCard,
/// };
///
/// let mut workflow = Workflow::new(WorkflowCard::new("demo"));
/// workflow.set_start_comp("start", StartComponent::new(), ComponentOptions::new());
/// assert_eq!(workflow.component_ids(), vec!["start"]);
/// ```
#[derive(Clone, Debug)]
pub struct Workflow {
    definition: WorkflowDefinition,
}

impl Workflow {
    /// Creates an empty workflow for `card` with Python's default configuration.
    #[must_use]
    pub fn new(card: WorkflowCard) -> Self {
        Self {
            definition: WorkflowDefinition::new(card),
        }
    }

    /// Creates an empty workflow from a full configuration.
    ///
    /// The card comes from `config.card` and the nesting depth (and any other user
    /// settings) are kept, but the stored spec is recomputed from the builder's state, so
    /// a spec carried in `config` does not survive.
    #[must_use]
    pub fn with_config(config: WorkflowConfig) -> Self {
        let mut definition = WorkflowDefinition::with_config(config);
        let spec = definition.spec();
        definition.config.spec = spec;
        Self { definition }
    }

    /// The workflow's card.
    #[must_use]
    pub fn card(&self) -> &WorkflowCard {
        &self.definition.card
    }

    /// The workflow's configuration, with the spec kept in step with the builder.
    #[must_use]
    pub fn config(&self) -> &WorkflowConfig {
        &self.definition.config
    }

    /// Registers the start component.
    ///
    /// A second call keeps the first start and records a problem that the next compile
    /// reports.
    pub fn set_start_comp(
        &mut self,
        id: &str,
        component: impl Component + 'static,
        options: ComponentOptions,
    ) -> &mut Self {
        self.register(id, component, options);
        if let Some(existing) = self.definition.start.clone() {
            self.definition
                .record_problem(format!("start component is already set to '{existing}'"));
        } else {
            self.definition.start = Some(id.to_string());
        }
        self.refresh_spec();
        self
    }

    /// Registers an intermediate component.
    pub fn add_workflow_comp(
        &mut self,
        id: &str,
        component: impl Component + 'static,
        options: ComponentOptions,
    ) -> &mut Self {
        self.register(id, component, options);
        self.refresh_spec();
        self
    }

    /// Registers the end component.
    ///
    /// A second call keeps the first end and records a problem that the next compile
    /// reports.
    pub fn set_end_comp(
        &mut self,
        id: &str,
        component: impl Component + 'static,
        options: ComponentOptions,
    ) -> &mut Self {
        self.register(id, component, options);
        if let Some(existing) = self.definition.end.clone() {
            self.definition
                .record_problem(format!("end component is already set to '{existing}'"));
        } else {
            self.definition.end = Some(id.to_string());
        }
        self.refresh_spec();
        self
    }

    /// Adds a regular edge from `from` to `to`.
    pub fn add_connection(&mut self, from: &str, to: &str) -> &mut Self {
        self.definition
            .add_edge(vec![from.to_string()], to.to_string(), EdgeKind::Regular);
        self.refresh_spec();
        self
    }

    /// Adds one barrier edge from every source in `from` to `to`.
    ///
    /// An empty source list records a problem instead of an edge; the graph's channels
    /// are barriers, so the target then never runs.
    pub fn add_connections(&mut self, from: &[&str], to: &str) -> &mut Self {
        if from.is_empty() {
            self.definition
                .record_problem(format!("barrier edge to '{to}' has no sources"));
        } else {
            self.definition.add_edge(
                from.iter().map(|source| (*source).to_string()).collect(),
                to.to_string(),
                EdgeKind::Barrier,
            );
        }
        self.refresh_spec();
        self
    }

    /// Adds a stream edge from `from` to `to`.
    pub fn add_stream_connection(&mut self, from: &str, to: &str) -> &mut Self {
        self.definition
            .add_edge(vec![from.to_string()], to.to_string(), EdgeKind::Stream);
        self.refresh_spec();
        self
    }

    /// Adds a conditional edge from `from` to `to`, backed by `router`.
    ///
    /// Several routers may be registered for one source; the engine combines them in
    /// registration order. `to` is declared once per source.
    pub fn add_conditional_connection(
        &mut self,
        from: &str,
        to: &str,
        router: Arc<dyn Router>,
    ) -> &mut Self {
        self.definition.add_conditional(from, to, router);
        self.refresh_spec();
        self
    }

    /// The start component id, when one has been set.
    #[must_use]
    pub fn start_comp_id(&self) -> Option<&str> {
        self.definition.start.as_deref()
    }

    /// The end component id, when one has been set.
    #[must_use]
    pub fn end_comp_id(&self) -> Option<&str> {
        self.definition.end.as_deref()
    }

    /// The registered component ids in declaration order.
    #[must_use]
    pub fn component_ids(&self) -> Vec<&str> {
        self.definition.component_ids()
    }

    /// The Python-compatible spec computed from the current builder state.
    #[must_use]
    pub fn spec(&self) -> WorkflowSpec {
        self.definition.spec()
    }

    /// Validates and compiles the workflow.
    ///
    /// # Errors
    ///
    /// Returns one coded workflow error listing every recorded problem and every
    /// validation failure.
    pub fn build(&self) -> Result<CompiledWorkflow> {
        CompiledWorkflow::build(self.definition.clone())
    }

    /// Renders the workflow as a Mermaid flowchart.
    ///
    /// # Errors
    ///
    /// Returns the same compile error as [`Workflow::build`]; rendering itself cannot fail
    /// (an uncompilable graph renders as a Mermaid comment).
    pub fn to_mermaid(&self) -> Result<String> {
        self.build().map(|compiled| compiled.to_mermaid())
    }

    /// Renders the workflow as a standalone SVG document.
    ///
    /// # Errors
    ///
    /// Returns the same compile error as [`Workflow::build`]; once a workflow compiles,
    /// [`CompiledWorkflow::to_svg`] always returns a document.
    pub fn to_svg(&self) -> Result<String> {
        self.build().map(|compiled| compiled.to_svg())
    }

    /// Renders the workflow in `format`.
    ///
    /// This is the Rust stand-in for Python's `Workflow.draw(..., output_format=...)`:
    /// [`DrawFormat`] replaces the format string. PNG is intentionally not provided —
    /// Python renders both PNG and SVG through the mermaid.ink network service, whereas
    /// [`DrawFormat::Svg`] is rendered locally and deterministically.
    ///
    /// # Errors
    ///
    /// Returns the same compile error as [`Workflow::build`]; rendering itself cannot fail.
    pub fn draw(&self, format: DrawFormat) -> Result<String> {
        match format {
            DrawFormat::Mermaid => self.to_mermaid(),
            DrawFormat::Svg => self.to_svg(),
        }
    }

    /// Compiles the workflow and runs it to completion.
    ///
    /// # Errors
    ///
    /// Returns the compile error, then the same failures as
    /// [`CompiledWorkflow::invoke`](crate::engine::CompiledWorkflow::invoke).
    pub async fn invoke(&self, inputs: Value, session: Arc<dyn Session>) -> Result<WorkflowOutput> {
        self.invoke_with(inputs, session, WorkflowRunOptions::default())
            .await
    }

    /// Compiles the workflow and runs it with explicit [`WorkflowRunOptions`].
    ///
    /// # Errors
    ///
    /// Returns the compile error, then the same failures as
    /// [`CompiledWorkflow::invoke`](crate::engine::CompiledWorkflow::invoke).
    pub async fn invoke_with(
        &self,
        inputs: Value,
        session: Arc<dyn Session>,
        options: WorkflowRunOptions,
    ) -> Result<WorkflowOutput> {
        self.build()?.invoke(inputs, session, options).await
    }

    /// Compiles the workflow and resumes an interrupted run with `interactive` answers.
    ///
    /// # Errors
    ///
    /// Returns the compile error, then the same failures as
    /// [`CompiledWorkflow::resume`](crate::engine::CompiledWorkflow::resume) (including a
    /// missing checkpointer).
    pub async fn resume(
        &self,
        interactive: InteractiveInput,
        session: Arc<dyn Session>,
    ) -> Result<WorkflowOutput> {
        self.build()?
            .resume(interactive, session, WorkflowRunOptions::default())
            .await
    }

    /// Compiles the workflow and streams its frames.
    ///
    /// # Errors
    ///
    /// A compile error is delivered as the stream's single `Err` item rather than
    /// returned here; run failures behave like
    /// [`CompiledWorkflow::stream`](crate::engine::CompiledWorkflow::stream).
    pub fn stream(
        &self,
        inputs: Value,
        session: Arc<dyn Session>,
    ) -> BoxStream<'static, Result<WorkflowChunk>> {
        self.stream_with(inputs, session, WorkflowRunOptions::default())
    }

    /// Compiles the workflow and streams its frames with explicit [`WorkflowRunOptions`].
    ///
    /// # Errors
    ///
    /// A compile error is delivered as the stream's single `Err` item; run failures behave
    /// like [`CompiledWorkflow::stream`](crate::engine::CompiledWorkflow::stream).
    pub fn stream_with(
        &self,
        inputs: Value,
        session: Arc<dyn Session>,
        options: WorkflowRunOptions,
    ) -> BoxStream<'static, Result<WorkflowChunk>> {
        match self.build() {
            Ok(compiled) => compiled.stream(inputs, session, options),
            Err(failure) => stream::once(async move { Err(failure) }).boxed(),
        }
    }

    /// Registers `component` unless `id` is already taken.
    fn register(
        &mut self,
        id: &str,
        component: impl Component + 'static,
        options: ComponentOptions,
    ) {
        if self.definition.component(id).is_some() {
            self.definition
                .record_problem(format!("component id '{id}' is already registered"));
            return;
        }
        self.definition.insert_component(
            id,
            Arc::new(component) as Arc<dyn Component>,
            options.into_node_config(),
        );
    }

    /// Recomputes the stored spec so `config()` and `spec()` always agree.
    fn refresh_spec(&mut self) {
        let spec = self.definition.spec();
        self.definition.config.spec = spec;
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::assert_is_empty,
    clippy::needless_pass_by_value
)]
mod tests {
    use super::*;
    use crate::components::{EndComponent, StartComponent};
    use openjiuwen_session::WorkflowSession;
    use serde_json::json;

    fn session() -> Arc<dyn Session> {
        Arc::new(WorkflowSession::new("session-1"))
    }

    /// A `defaultOutputs` policy carrying one recovery value.
    fn recovery_config() -> ExceptionConfig {
        ExceptionConfig {
            handle_type: "defaultOutputs".to_string(),
            default_outputs: Some(json!({"result": "recovered"})),
            metadata: std::collections::BTreeMap::new(),
        }
    }

    /// start → end, both built-in components.
    fn workflow() -> Workflow {
        let mut workflow = Workflow::new(WorkflowCard::new("wf"));
        workflow
            .set_start_comp(
                "start",
                StartComponent::new(),
                ComponentOptions::new().with_inputs_schema(json!({"query": "${inputs.query}"})),
            )
            .set_end_comp(
                "end",
                EndComponent::new(),
                ComponentOptions::new().with_inputs_schema(json!({"answer": "${start.query}"})),
            )
            .add_connection("start", "end");
        workflow
    }

    fn build_error(workflow: &Workflow) -> String {
        workflow.build().unwrap_err().to_string()
    }

    #[test]
    fn options_default_to_no_limits() {
        let options = ComponentOptions::new();
        assert_eq!(options.inputs_schema, None);
        assert_eq!(options.stream_inputs_schema, None);
        assert_eq!(options.abilities, None);
        assert_eq!(options.wait_for_all, None);
        assert_eq!(options.max_retries, 0);
        assert_eq!(options.timeout_ms, None);
        assert_eq!(options.exception_config, None);
        assert_eq!(options.name, None);
        assert_eq!(ComponentOptions::default().max_retries, 0);
    }

    #[test]
    fn options_chain_every_setter() {
        let options = ComponentOptions::new()
            .with_inputs_schema(json!({"a": 1}))
            .with_outputs_schema(json!({"b": 2}))
            .with_stream_inputs_schema(json!({"c": 3}))
            .with_stream_outputs_schema(json!({"d": 4}))
            .with_abilities(vec![ComponentAbility::Collect])
            .with_wait_for_all(true)
            .with_max_retries(3)
            .with_timeout_ms(1500)
            .with_exception_config(Some(recovery_config()))
            .with_name("Node");

        assert_eq!(options.inputs_schema, Some(json!({"a": 1})));
        assert_eq!(options.outputs_schema, Some(json!({"b": 2})));
        assert_eq!(options.stream_inputs_schema, Some(json!({"c": 3})));
        assert_eq!(options.stream_outputs_schema, Some(json!({"d": 4})));
        assert_eq!(options.abilities, Some(vec![ComponentAbility::Collect]));
        assert_eq!(options.wait_for_all, Some(true));
        assert_eq!(options.max_retries, 3);
        assert_eq!(options.timeout_ms, Some(1500));
        assert_eq!(options.exception_config, Some(recovery_config()));
        assert_eq!(options.name.as_deref(), Some("Node"));
    }

    #[test]
    fn into_node_config_copies_every_field() {
        let node = ComponentOptions::new()
            .with_inputs_schema(json!({"a": 1}))
            .with_outputs_schema(json!({"b": 2}))
            .with_stream_inputs_schema(json!({"c": 3}))
            .with_stream_outputs_schema(json!({"d": 4}))
            .with_abilities(vec![ComponentAbility::Stream])
            .with_wait_for_all(true)
            .with_max_retries(1)
            .with_timeout_ms(250)
            .with_exception_config(Some(recovery_config()))
            .with_name("Node")
            .into_node_config();

        assert_eq!(node.inputs_schema, Some(json!({"a": 1})));
        assert_eq!(node.outputs_schema, Some(json!({"b": 2})));
        assert_eq!(node.stream_inputs_schema, Some(json!({"c": 3})));
        assert_eq!(node.stream_outputs_schema, Some(json!({"d": 4})));
        assert_eq!(node.abilities, Some(vec![ComponentAbility::Stream]));
        assert_eq!(node.wait_for_all, Some(true));
        assert_eq!(node.max_retries, 1);
        assert_eq!(node.timeout_ms, Some(250));
        assert_eq!(node.exception_config, Some(recovery_config()));
        assert_eq!(node.name.as_deref(), Some("Node"));
    }

    #[test]
    fn streaming_without_stream_inputs_is_invoke_and_stream() {
        let options = ComponentOptions::new().streaming();
        assert_eq!(
            options.abilities,
            Some(vec![ComponentAbility::Invoke, ComponentAbility::Stream])
        );
        assert_eq!(options.wait_for_all, None);
    }

    #[test]
    fn streaming_with_stream_inputs_adds_transform_and_waits() {
        let options = ComponentOptions::new()
            .with_stream_inputs_schema(json!({"query": "${start.query}"}))
            .streaming();

        assert_eq!(
            options.abilities,
            Some(vec![
                ComponentAbility::Invoke,
                ComponentAbility::Stream,
                ComponentAbility::Transform,
            ])
        );
        assert_eq!(options.wait_for_all, Some(true));
    }

    #[test]
    fn streaming_keeps_a_caller_set_wait_for_all() {
        let options = ComponentOptions::new().with_wait_for_all(false).streaming();
        assert_eq!(options.wait_for_all, Some(false));
    }

    #[test]
    fn new_workflow_starts_empty() {
        let workflow = Workflow::new(WorkflowCard::new("wf"));
        assert_eq!(workflow.card(), &WorkflowCard::new("wf"));
        assert_eq!(workflow.start_comp_id(), None);
        assert_eq!(workflow.end_comp_id(), None);
        assert!(workflow.component_ids().is_empty());
        assert_eq!(workflow.config().workflow_max_nesting_depth, 5);
        assert_eq!(workflow.spec(), WorkflowSpec::default());
    }

    #[test]
    fn with_config_keeps_the_nesting_depth_and_recomputes_the_spec() {
        let config = WorkflowConfig {
            card: WorkflowCard::new("configured"),
            workflow_max_nesting_depth: 3,
            spec: WorkflowSpec {
                start_nodes: vec!["stale".to_string()],
                ..WorkflowSpec::default()
            },
        };
        let workflow = Workflow::with_config(config);

        assert_eq!(workflow.card(), &WorkflowCard::new("configured"));
        assert_eq!(workflow.config().workflow_max_nesting_depth, 3);
        assert_eq!(workflow.spec(), WorkflowSpec::default());
        assert_eq!(workflow.config().spec, WorkflowSpec::default());
    }

    #[test]
    fn accessors_track_the_builder() {
        let workflow = workflow();

        assert_eq!(workflow.start_comp_id(), Some("start"));
        assert_eq!(workflow.end_comp_id(), Some("end"));
        assert_eq!(workflow.component_ids(), vec!["start", "end"]);
        assert_eq!(workflow.card().id, "wf");
    }

    #[test]
    fn spec_is_refreshed_by_every_mutator() {
        let mut workflow = Workflow::new(WorkflowCard::new("wf"));
        workflow.set_start_comp("start", StartComponent::new(), ComponentOptions::new());
        assert_eq!(workflow.config().spec.start_nodes, vec!["start"]);
        assert_eq!(workflow.config().spec, workflow.spec());

        workflow
            .add_workflow_comp("end", EndComponent::new(), ComponentOptions::new())
            .add_connection("start", "end");
        let spec = workflow.spec();
        assert_eq!(workflow.config().spec, spec);
        assert_eq!(
            spec.edges.get("start"),
            Some(&vec!["end".to_string()]),
            "regular edges populate `edges`"
        );
        assert_eq!(spec.comp_configs.len(), 2);
        assert_eq!(spec.start_nodes, vec!["start"]);
    }

    #[test]
    fn stream_edges_are_recorded_separately() {
        let mut workflow = workflow();
        workflow
            .add_stream_connection("start", "end")
            .add_connections(&["start", "end"], "end");

        let spec = workflow.spec();
        assert_eq!(
            spec.stream_edges.get("start"),
            Some(&vec!["end".to_string()])
        );
        assert_eq!(
            spec.stream_source_groups.get("end"),
            Some(&vec![vec!["start".to_string()]])
        );
        assert_eq!(
            spec.edges.get("end"),
            Some(&vec!["end".to_string()]),
            "the barrier edge contributes one entry per source"
        );
    }

    #[test]
    fn duplicate_component_is_kept_once_and_reported_by_build() {
        let mut workflow = workflow();
        // The first registration wins, so the second component (a start here) is dropped.
        workflow.add_workflow_comp("end", StartComponent::new(), ComponentOptions::new());

        assert_eq!(workflow.component_ids(), vec!["start", "end"]);
        assert_eq!(
            workflow.definition.problems(),
            ["component id 'end' is already registered"]
        );
        let message = build_error(&workflow);
        assert!(
            message.contains("component id 'end' is already registered"),
            "unexpected message: {message}"
        );
    }

    #[test]
    fn duplicate_start_keeps_the_first_and_is_reported() {
        let mut workflow = workflow();
        workflow.set_start_comp("start2", StartComponent::new(), ComponentOptions::new());

        assert_eq!(workflow.start_comp_id(), Some("start"));
        assert_eq!(
            workflow.definition.problems(),
            ["start component is already set to 'start'"]
        );
        assert!(build_error(&workflow).contains("start component is already set to 'start'"));
    }

    #[test]
    fn duplicate_end_keeps_the_first_and_is_reported() {
        let mut workflow = Workflow::new(WorkflowCard::new("wf"));
        workflow
            .set_start_comp("start", StartComponent::new(), ComponentOptions::new())
            .set_end_comp("end", EndComponent::new(), ComponentOptions::new())
            .set_end_comp("end2", EndComponent::new(), ComponentOptions::new());

        assert_eq!(workflow.end_comp_id(), Some("end"));
        assert_eq!(
            workflow.definition.problems(),
            ["end component is already set to 'end'"]
        );
        assert!(build_error(&workflow).contains("end component is already set to 'end'"));
    }

    #[test]
    fn barrier_edge_without_sources_is_reported() {
        let mut workflow = workflow();
        workflow.add_connections(&[], "end");

        assert_eq!(
            workflow.definition.problems(),
            ["barrier edge to 'end' has no sources"]
        );
        assert!(
            build_error(&workflow).contains("barrier edge to 'end' has no sources"),
            "the recorded problem must surface from build"
        );
    }

    #[test]
    fn bad_component_id_is_reported_by_build() {
        let mut workflow = workflow();
        workflow.add_workflow_comp("bad id", StartComponent::new(), ComponentOptions::new());

        let message = build_error(&workflow);
        assert!(
            message.contains("[100010]"),
            "expected the id validation code: {message}"
        );
        assert!(message.contains("bad id"), "unexpected message: {message}");
    }

    #[test]
    fn missing_start_and_end_are_build_errors() {
        let workflow = Workflow::new(WorkflowCard::new("wf"));
        let message = build_error(&workflow);
        assert!(message.contains("no start component"), "{message}");
        assert!(message.contains("no end component"), "{message}");
    }

    #[test]
    fn build_compiles_a_valid_workflow() {
        let compiled = workflow().build().unwrap();

        assert_eq!(compiled.card().id, "wf");
        assert_eq!(compiled.start_comp_id(), "start");
        assert_eq!(compiled.end_comp_id(), "end");
        assert!(!compiled.is_streaming());
        assert_eq!(compiled.spec(), &workflow().spec());
    }

    #[tokio::test]
    async fn invoke_compiles_on_the_fly() {
        let output = workflow()
            .invoke(json!({"query": "hi"}), session())
            .await
            .unwrap();

        assert_eq!(output.result, json!({"output": {"answer": "hi"}}));
    }

    #[tokio::test]
    async fn stream_on_an_invalid_workflow_yields_one_error() {
        let invalid = Workflow::new(WorkflowCard::new("wf"));
        let frames: Vec<_> = invalid.stream(json!({}), session()).collect().await;

        assert_eq!(frames.len(), 1);
        let failure = frames.into_iter().next().unwrap().unwrap_err();
        assert!(failure.to_string().contains("no start component"));
    }

    #[tokio::test]
    async fn stream_with_on_a_valid_workflow_yields_frames() {
        let frames: Vec<_> = workflow()
            .stream_with(json!({"query": "hi"}), session(), WorkflowRunOptions::new())
            .collect()
            .await;

        assert!(
            frames.iter().all(Result::is_ok),
            "unexpected failure: {frames:?}"
        );
        assert!(
            frames
                .iter()
                .flatten()
                .any(|frame| frame.mode == openjiuwen_session::stream::StreamMode::Output),
            "expected at least one output frame"
        );
    }

    #[test]
    fn to_mermaid_renders_a_flowchart() {
        let mermaid = workflow().to_mermaid().unwrap();
        assert!(mermaid.contains("flowchart"), "{mermaid}");
    }

    #[test]
    fn to_mermaid_reports_a_compile_error() {
        let error = Workflow::new(WorkflowCard::new("wf"))
            .to_mermaid()
            .unwrap_err();
        assert!(error.to_string().contains("no start component"));
    }

    #[test]
    fn workflow_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Workflow>();
        assert_send_sync::<ComponentOptions>();
    }
}
