//! The graph adapter: one [`Component`] driven as an [`openjiuwen_graph::Node`].
//!
//! `ComponentNode` is the only place where a workflow component meets the graph
//! executor. It resolves the node's input templates against workflow state, hands the
//! component a [`ComponentContext`], dispatches exactly one ability, projects the result
//! through `outputs_schema`, and writes it back under the node's id. Everything about
//! supersteps, channels, retries, and node timeouts stays in `openjiuwen-graph`.
//!
//! Two behaviours deserve a note:
//!
//! - **The root seed carries the run inputs.** `super::run` seeds the graph with the
//!   workflow inputs plus a reserved `"inputs"` key holding the original value, so both
//!   `${query}` (root reference) and `${start.query}` (node reference) resolve.
//! - **Streaming is buffered.** A `Stream`/`Transform` node drains its component stream
//!   into a JSON array stored under its id instead of feeding live generators
//!   downstream. When the node is the workflow end, every frame is also published as an
//!   `end node stream` output frame and the stored array holds the
//!   `{"type", "index", "payload"}` wires, which is exactly what `invoke` returns.
//!
//! Interrupts are data: a component that asks a question without a seeded answer returns
//! an interrupt error, and the adapter propagates it (keeping the tracing `interrupted`
//! marker) instead of swallowing it. The run fails at that node, so the graph engine
//! stores the *pre-superstep* boundary and `resume` re-runs the interrupted superstep
//! with the seeded answers.
//!
//! **Error recovery** ports Python's `_run_error_recovery` / `_post_invoke`
//! (`openjiuwen/core/graph/vertex.py`): once the graph's retries are exhausted, a failed
//! `Invoke`/`Collect` node whose `ExceptionConfig` carries `default_outputs` and a
//! non-`interrupt` `handle_type` is treated as recovered. The defaults go through the
//! node's `outputs_schema` and are written to state under the node id, so downstream
//! `${node.field}` references resolve, exactly like Python. Deviation: Python only
//! recovers through a registered `component_error_recovery` callback and re-raises
//! `Stream`/`Transform` failures; this port applies the declared policy inline and,
//! matching Python, never recovers a streaming failure.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::BoxStream;
use futures::StreamExt;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::utils::json as json_utils;
use openjiuwen_graph::{GraphContext, GraphState, Node, NodeOutcome};
use openjiuwen_session::tracer::SpanHandle;
use parking_lot::Mutex;
use serde_json::Value;

use crate::ability::ComponentAbility;
use crate::component::Component;
use crate::context::ComponentContext;
use crate::engine::definition::NodeConfig;
use crate::error;
use crate::interrupt::{find_interrupt, InteractionState};
use crate::output::{FrameWriter, OutputFrame, END_NODE_STREAM};
use crate::reference::CompiledSchema;

// The interaction-wire builder sits with the invoke path that reports it in the private
// `engine::run` module; re-export it so crate-level callers (such as `crate::single`)
// can reuse it without a second copy.
pub(crate) use crate::engine::run::interaction_wires;

/// Trace reason recorded when a component halts the run for human input.
const INTERRUPTED_REASON: &str = "workflow interrupted waiting for input";

/// The per-run state every node of one execution shares.
///
/// A [`NodeRuntime`] only has a scope while it is executed; workflows compiled for
/// [`super::CompiledWorkflow::to_mermaid`] carry `None` and are never run.
pub(crate) struct RunScope {
    /// Seeded answers and pending questions of this run.
    pub(crate) interactions: Arc<InteractionState>,
    /// The frame writer every component of this run publishes through.
    pub(crate) writer: Arc<FrameWriter>,
    /// Nesting level of this run (`0` for a top-level run).
    pub(crate) depth: usize,
    /// Consecutive failed attempts per node id, so error recovery waits for the
    /// graph's retries to be exhausted.
    failures: Mutex<HashMap<String, u32>>,
}

impl RunScope {
    /// Builds a scope from a run's interaction state, frame writer, and nesting level.
    #[must_use]
    pub(crate) fn new(
        interactions: Arc<InteractionState>,
        writer: Arc<FrameWriter>,
        depth: usize,
    ) -> Self {
        Self {
            interactions,
            writer,
            depth,
            failures: Mutex::new(HashMap::new()),
        }
    }

    /// Records one failed attempt of `id` and returns its consecutive failure count.
    fn record_failure(&self, id: &str) -> u32 {
        let mut failures = self.failures.lock();
        let attempts = failures.entry(id.to_owned()).or_insert(0);
        *attempts = attempts.saturating_add(1);
        *attempts
    }

    /// Forgets `id`'s failed attempts, called once it produces a value.
    fn clear_failures(&self, id: &str) {
        self.failures.lock().remove(id);
    }
}

/// Everything one graph node needs to run its component.
///
/// The runtime is built once per compiled graph (that is, once per run) so the node
/// instances stay cheap to clone and free of mutable state.
pub(crate) struct NodeRuntime {
    /// The component id, which is also the graph node id and the state key.
    pub(crate) id: String,
    /// The component runnable.
    pub(crate) component: Arc<dyn Component>,
    /// The node's configuration (schemas, retries, timeout, display name).
    pub(crate) node: NodeConfig,
    /// The node's templates, compiled once at graph build.
    pub(crate) schemas: NodeSchemas,
    /// The single ability this node is driven with.
    pub(crate) ability: ComponentAbility,
    /// Whether this node is the workflow's end component.
    pub(crate) is_end: bool,
    /// The enclosing workflow's card id.
    pub(crate) workflow_id: String,
    /// Per-run state; `None` only for graphs compiled for visualization.
    pub(crate) scope: Option<Arc<RunScope>>,
}

impl std::fmt::Debug for NodeRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NodeRuntime")
            .field("id", &self.id)
            .field("component_type", &self.component.component_type())
            .field("node", &self.node)
            .field("schemas", &self.schemas)
            .field("ability", &self.ability)
            .field("is_end", &self.is_end)
            .field("workflow_id", &self.workflow_id)
            .field("scope", &self.scope.as_ref().map(|_| "RunScope"))
            .finish()
    }
}

/// One workflow component exposed as a graph node.
pub(crate) struct ComponentNode {
    rt: Arc<NodeRuntime>,
}

impl ComponentNode {
    /// Wraps `rt` as a graph node.
    #[must_use]
    pub(crate) fn new(rt: Arc<NodeRuntime>) -> Self {
        Self { rt }
    }

    /// Starts the component's trace span, when the session has a tracer.
    ///
    /// The span gets the node's display name (its configured `name`, else the
    /// component card's name) and records the resolved inputs immediately. Nothing is
    /// cloned when the session has no tracer, which is the common case.
    fn start_span(&self, ctx: &GraphContext, inputs: &Value) -> Option<SpanHandle> {
        let tracer = ctx.session().tracer()?;
        let name = self
            .rt
            .node
            .name
            .clone()
            .unwrap_or_else(|| self.rt.component.card().name);
        let span = tracer.start_workflow_span(self.rt.id.clone(), name);
        span.set_inputs(inputs.clone());
        Some(span)
    }

    /// Runs the component's selected ability.
    ///
    /// Routing wins over every ability: a component that returns
    /// `Some(Ok(targets))` from [`Component::route`] routes control flow instead of
    /// producing a value.
    async fn dispatch(
        &self,
        source: &Value,
        inputs: Value,
        ctx: &ComponentContext,
        scope: &RunScope,
    ) -> Result<Dispatch> {
        if let Some(routed) = self.rt.component.route(source) {
            return routed.map(Dispatch::Route);
        }
        match self.rt.ability {
            ComponentAbility::Invoke => Ok(Dispatch::Output(
                self.rt.component.invoke(inputs, ctx).await?,
            )),
            ComponentAbility::Collect => Ok(Dispatch::Output(
                self.rt.component.collect(inputs, ctx).await?,
            )),
            ComponentAbility::Stream => {
                let frames = Arc::clone(&self.rt.component).stream(inputs, Arc::new(ctx.clone()));
                self.drain(frames, scope).await.map(Dispatch::Output)
            }
            ComponentAbility::Transform => {
                let frames =
                    Arc::clone(&self.rt.component).transform(inputs, Arc::new(ctx.clone()));
                self.drain(frames, scope).await.map(Dispatch::Output)
            }
        }
    }

    /// Drains a component stream into the value stored under the node id.
    ///
    /// End nodes publish every frame as an `end node stream` output frame and store the
    /// Python-compatible `{"type", "index", "payload"}` wires; other nodes store the raw
    /// payloads.
    async fn drain(
        &self,
        mut frames: BoxStream<'static, Result<Value>>,
        scope: &RunScope,
    ) -> Result<Value> {
        let mut collected: Vec<Value> = Vec::new();
        let mut index: u32 = 0;
        while let Some(frame) = frames.next().await {
            let payload = frame?;
            if !self.rt.is_end {
                collected.push(payload);
                continue;
            }
            scope
                .writer
                .write_output(END_NODE_STREAM, index, payload.clone())
                .await?;
            collected.push(
                OutputFrame {
                    item_type: END_NODE_STREAM.to_owned(),
                    index,
                    payload,
                }
                .to_wire(),
            );
            index = index.saturating_add(1);
        }
        Ok(Value::Array(collected))
    }

    /// The value substituted when a failed invocation is recoverable, if it is.
    ///
    /// Mirrors Python's `_run_error_recovery`: only `Invoke` and `Collect` failures are
    /// recoverable (a stream's frames cannot be replaced by a fixed value), and only when
    /// the node's `ExceptionConfig` declares `default_outputs` under a policy other than
    /// `"interrupt"`. An `"interrupt"` policy — Python's default — always propagates.
    fn recovered_output(&self) -> Option<&Value> {
        if !matches!(
            self.rt.ability,
            ComponentAbility::Invoke | ComponentAbility::Collect
        ) {
            return None;
        }
        let config = self.rt.node.exception_config.as_ref()?;
        if config.handle_type == "interrupt" {
            return None;
        }
        config.default_outputs.as_ref()
    }
}

impl std::fmt::Debug for ComponentNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ComponentNode")
            .field("id", &self.rt.id)
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl Node for ComponentNode {
    fn id(&self) -> &str {
        &self.rt.id
    }

    async fn run(&self, state: &mut GraphState, ctx: &GraphContext) -> Result<NodeOutcome> {
        let rt = &*self.rt;
        let scope = rt.scope.as_ref().ok_or_else(|| {
            Error::workflow(format!(
                "component '{}' executed without a run scope",
                rt.id
            ))
        })?;
        let context = ComponentContext::new(
            Arc::clone(ctx.session()),
            rt.id.clone(),
            rt.workflow_id.clone(),
        )
        .with_interactions(Arc::clone(&scope.interactions))
        .with_writer(Arc::clone(&scope.writer))
        .with_cancellation(ctx.cancellation().clone())
        .with_depth(scope.depth);

        // Hold the state's map instead of deep-cloning it into a `Value`; the view hands it
        // back when it drops, so the state survives every outcome below.
        let mut view = StateView::new(state);
        let inputs = rt.schemas.resolve_inputs(rt.ability, view.source());
        // Nothing is cloned for the span unless the session really has a tracer.
        let span = self.start_span(ctx, view.source());

        match self.dispatch(view.source(), inputs, &context, scope).await {
            Ok(Dispatch::Route(targets)) => {
                scope.clear_failures(&rt.id);
                if let Some(span) = &span {
                    // A branch produces no output; the span still closes.
                    span.finish(Value::Null);
                }
                Ok(NodeOutcome::Route { targets })
            }
            Ok(Dispatch::Output(raw)) => {
                scope.clear_failures(&rt.id);
                let output = rt.schemas.project_output(view.source(), &rt.id, raw);
                if let Some(span) = &span {
                    span.finish(output.clone());
                }
                view.set(rt.id.clone(), output);
                Ok(NodeOutcome::Continue)
            }
            Err(failure) => {
                if find_interrupt(&failure).is_some() {
                    if let Some(span) = &span {
                        span.interrupted(INTERRUPTED_REASON);
                    }
                    // The `__interaction__` frames were already written by
                    // `ComponentContext::interact`. Propagating the interrupt (instead of
                    // halting) makes the graph save the *pre-superstep* boundary, so
                    // `resume` re-runs this superstep with the seeded answers.
                    return Err(failure);
                }
                // Python recovers only once every retry is exhausted, so the failed
                // attempts are counted here; the graph owns the retry loop itself.
                let attempts = scope.record_failure(&rt.id);
                if attempts > rt.node.max_retries {
                    if let Some(defaults) = self.recovered_output() {
                        // Python's `_run_error_recovery` calls `_post_invoke` with the
                        // handler's value, so the defaults go through `outputs_schema` too.
                        let output =
                            rt.schemas
                                .project_output(view.source(), &rt.id, defaults.clone());
                        if let Some(span) = &span {
                            span.set_metadata(
                                "error_recovery",
                                serde_json::json!({
                                    "handle_type": rt
                                        .node
                                        .exception_config
                                        .as_ref()
                                        .map(|config| config.handle_type.as_str())
                                        .unwrap_or_default(),
                                    "error": failure.to_string(),
                                }),
                            );
                            span.finish(output.clone());
                        }
                        tracing::warn!(
                            component = %rt.id,
                            error = %failure,
                            "recovered a failed component with its configured default outputs"
                        );
                        scope.clear_failures(&rt.id);
                        view.set(rt.id.clone(), output);
                        return Ok(NodeOutcome::Continue);
                    }
                }
                let failure = error::component_execution_error(
                    rt.component.component_type(),
                    rt.ability.as_str(),
                    failure.to_string().as_str(),
                    &rt.workflow_id,
                );
                if let Some(span) = &span {
                    span.fail(failure.to_string());
                }
                Err(failure)
            }
        }
    }
}

/// What one node produced: a value for workflow state, or routing targets.
enum Dispatch {
    /// The component returned a value that is stored under the node id.
    Output(Value),
    /// The component routed control flow instead of producing a value.
    Route(Vec<String>),
}

/// The node's templates, compiled once when the graph is built.
///
/// Every `${node.field}` reference in an input, stream-input or output template is split
/// into [`PathSegment`](crate::reference::PathSegment)s here, at graph build, instead of
/// on every node execution ([`crate::reference::resolve_schema`] re-parses them and stays
/// the entry point for a one-off template).
///
/// `None` means the node has no template for that slot; an input node without a template
/// hands its component the whole workflow state.
#[derive(Debug)]
pub(crate) struct NodeSchemas {
    inputs: Option<CompiledSchema>,
    stream_inputs: Option<CompiledSchema>,
    outputs: Option<CompiledSchema>,
}

impl NodeSchemas {
    /// Compiles `node`'s input, stream-input and output templates.
    pub(crate) fn compile(node: &NodeConfig) -> Self {
        Self {
            inputs: node.inputs_schema.as_ref().map(CompiledSchema::compile),
            stream_inputs: node
                .stream_inputs_schema
                .as_ref()
                .map(CompiledSchema::compile),
            outputs: node.outputs_schema.as_ref().map(CompiledSchema::compile),
        }
    }

    /// Resolves the node's input template (plus its stream template for
    /// `collect`/`transform`).
    ///
    /// Without an input template the whole state object is handed to the component, which
    /// is what Python's components do when no explicit input mapping exists.
    fn resolve_inputs(&self, ability: ComponentAbility, source: &Value) -> Value {
        let inputs = match &self.inputs {
            Some(schema) => schema.resolve(source),
            None => source.clone(),
        };
        if !matches!(
            ability,
            ComponentAbility::Collect | ComponentAbility::Transform
        ) {
            return inputs;
        }
        let Some(schema) = self.stream_inputs.as_ref() else {
            return inputs;
        };
        merge_stream_inputs(inputs, schema.resolve(source))
    }

    /// Projects a component's raw output through the node's output template.
    ///
    /// The template resolves against the workflow state plus `{"<id>": raw, "output":
    /// raw}`, so it can reference its own component id or the bare `output` key. The
    /// extra keys are root bindings rather than a copy of the state, so projecting an
    /// output costs nothing when the node has no output template.
    fn project_output(&self, source: &Value, id: &str, raw: Value) -> Value {
        project_output(self.outputs.as_ref(), source, id, raw)
    }
}

/// Projects a component's raw output through `schema` (see [`NodeSchemas::project_output`]).
///
/// This is the engine's state convention; `crate::single` reuses it against a
/// single-component call's resolved inputs so both paths project outputs identically.
pub(crate) fn project_output(
    schema: Option<&CompiledSchema>,
    source: &Value,
    id: &str,
    raw: Value,
) -> Value {
    let Some(schema) = schema else {
        return raw;
    };
    schema.resolve_with(source, &[(id, &raw), ("output", &raw)])
}

/// One node's view of the graph state as the JSON root its templates resolve against.
///
/// [`GraphState`] stores its data as a `Map` and `Value::Object` needs ownership, so
/// reading the state as a `Value` used to mean a deep clone of the whole state on every
/// node execution (and a second one for the component's inputs). The view *moves* the map
/// out of the state — an O(1) `take` — hands out a `&Value`, and puts the map back (with
/// the node's output, if it produced one) when it is dropped, including when the node
/// fails or its future is dropped for a timeout/cancellation. `openjiuwen-graph` hands each
/// node its own cloned state (`run_node` clones per attempt) and reads it only after the
/// node returns, so nothing observes the map while the view holds it.
struct StateView<'a> {
    state: &'a mut GraphState,
    value: Value,
}

impl<'a> StateView<'a> {
    /// Takes the state's map into a JSON object value.
    fn new(state: &'a mut GraphState) -> Self {
        #[cfg(test)]
        crate::engine::node::view_metrics::record();
        Self {
            value: Value::Object(std::mem::take(&mut state.0)),
            state,
        }
    }

    /// The state as the JSON object templates resolve against.
    fn source(&self) -> &Value {
        &self.value
    }

    /// Writes `value` under `key`, as `GraphState::set` would.
    fn set(&mut self, key: impl Into<String>, value: Value) {
        if let Value::Object(map) = &mut self.value {
            map.insert(key.into(), value);
        }
    }
}

impl Drop for StateView<'_> {
    fn drop(&mut self) {
        if let Value::Object(map) = std::mem::take(&mut self.value) {
            self.state.0 = map;
        }
    }
}

/// Test-only observation of how many times a node took the state view on this thread.
///
/// A regression that went back to materializing the state with `GraphState::as_value()` (a
/// deep clone per node) would take no views at all, so the engine's regression suite asserts
/// one view per executed node. The counter is thread-local so tests running in parallel do
/// not disturb it, and it only exists in test builds.
#[cfg(test)]
pub(crate) mod view_metrics {
    use std::cell::Cell;

    thread_local! {
        static TAKEN: Cell<usize> = const { Cell::new(0) };
    }

    /// Records one [`super::StateView::new`] on this thread.
    pub(crate) fn record() {
        TAKEN.with(|taken| taken.set(taken.get().saturating_add(1)));
    }

    /// How many state views were taken on this thread so far.
    pub(crate) fn views() -> usize {
        TAKEN.with(Cell::get)
    }
}

/// Deep-merges the streamed inputs over the batch inputs, stream keys winning.
///
/// A non-object streamed value replaces the batch inputs, matching the deep-merge rule
/// of [`openjiuwen_common::utils::json::merge`].
fn merge_stream_inputs(batch: Value, streamed: Value) -> Value {
    let mut merged = batch;
    json_utils::merge(&mut merged, streamed);
    merged
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::assert_is_empty)]
mod tests {
    use serde_json::json;

    use super::*;

    /// Compiles `node` and resolves its inputs, exactly as one node execution does.
    fn inputs(node: &NodeConfig, ability: ComponentAbility, source: &Value) -> Value {
        NodeSchemas::compile(node).resolve_inputs(ability, source)
    }

    /// Projects `raw` through `schema`, exactly as one node execution does.
    fn outputs(schema: Option<&Value>, source: &Value, id: &str, raw: Value) -> Value {
        let schema = schema.map(CompiledSchema::compile);
        project_output(schema.as_ref(), source, id, raw)
    }

    #[test]
    fn input_schema_falls_back_to_the_whole_state() {
        let source = json!({"query": "hello"});
        assert_eq!(
            inputs(&NodeConfig::default(), ComponentAbility::Invoke, &source),
            source
        );
    }

    #[test]
    fn input_schema_resolves_references() {
        let node = NodeConfig {
            inputs_schema: Some(json!({"q": "${start.query}"})),
            ..NodeConfig::default()
        };
        let source = json!({"start": {"query": "hello"}});
        assert_eq!(
            inputs(&node, ComponentAbility::Invoke, &source),
            json!({"q": "hello"})
        );
    }

    #[test]
    fn stream_inputs_win_over_batch_inputs() {
        let node = NodeConfig {
            inputs_schema: Some(json!({"q": "${start.query}", "keep": 1})),
            stream_inputs_schema: Some(json!({"q": "streamed"})),
            ..NodeConfig::default()
        };
        let source = json!({"start": {"query": "hello"}});
        assert_eq!(
            inputs(&node, ComponentAbility::Collect, &source),
            json!({"q": "streamed", "keep": 1})
        );
        // Batch abilities ignore the stream template.
        assert_eq!(
            inputs(&node, ComponentAbility::Invoke, &source),
            json!({"q": "hello", "keep": 1})
        );
    }

    #[test]
    fn non_object_stream_inputs_replace_the_batch_inputs() {
        let node = NodeConfig {
            inputs_schema: Some(json!({"q": "${start.query}"})),
            stream_inputs_schema: Some(json!(["a", "b"])),
            ..NodeConfig::default()
        };
        let source = json!({"start": {"query": "hello"}});
        assert_eq!(
            inputs(&node, ComponentAbility::Transform, &source),
            json!(["a", "b"])
        );
    }

    #[test]
    fn output_schema_sees_the_raw_value_under_its_own_id() {
        let source = json!({"start": {"query": "hello"}});
        let raw = json!({"text": "hi"});
        assert_eq!(
            outputs(
                Some(&json!({"answer": "${fake.text}"})),
                &source,
                "fake",
                raw.clone()
            ),
            json!({"answer": "hi"})
        );
        assert_eq!(
            outputs(
                Some(&json!({"answer": "${output.text}"})),
                &source,
                "fake",
                raw.clone()
            ),
            json!({"answer": "hi"})
        );
        // A missing path resolves to null and never fails.
        assert_eq!(
            outputs(
                Some(&json!({"answer": "${nope.x}"})),
                &source,
                "fake",
                raw.clone()
            ),
            json!({"answer": null})
        );
        // Without a schema the raw value is stored verbatim.
        assert_eq!(outputs(None, &source, "fake", raw.clone()), raw);
    }

    #[test]
    fn output_schema_projects_against_a_non_object_state() {
        let raw = json!(7);
        assert_eq!(
            outputs(
                Some(&json!({"value": "${output}"})),
                &json!("scalar"),
                "c",
                raw
            ),
            json!({"value": 7})
        );
    }

    #[test]
    fn stream_merge_keeps_batch_keys_and_deep_merges_objects() {
        assert_eq!(
            merge_stream_inputs(json!({"a": {"x": 1, "y": 2}}), json!({"a": {"y": 3}})),
            json!({"a": {"x": 1, "y": 3}})
        );
        assert_eq!(
            merge_stream_inputs(json!({"a": 1}), json!({"b": 2})),
            json!({"a": 1, "b": 2})
        );
    }

    #[test]
    fn the_state_view_moves_the_map_instead_of_cloning_it() {
        let mut state = GraphState::from_map(
            json!({"payload": {"big": [1, 2, 3]}, "other": "kept"})
                .as_object()
                .unwrap()
                .clone(),
        );
        let payload = std::ptr::from_ref(state.get("payload").unwrap());
        {
            let mut view = StateView::new(&mut state);
            // Same allocation: the view took the map instead of cloning it (`Value::clone`
            // would have produced a fresh `payload` allocation).
            assert_eq!(
                std::ptr::from_ref(view.source().pointer("/payload").unwrap()),
                payload
            );
            view.set("out", json!(true));
        }
        assert_eq!(state.get("payload").unwrap(), &json!({"big": [1, 2, 3]}));
        assert_eq!(state.get("other").unwrap(), &json!("kept"));
        assert_eq!(state.get("out").unwrap(), &json!(true));
        // A dropped view restores the state even without a write.
        {
            let _view = StateView::new(&mut state);
        }
        assert_eq!(state.get("other").unwrap(), &json!("kept"));
    }
}
