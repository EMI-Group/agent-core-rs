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
use serde_json::{Map, Value};

use crate::ability::ComponentAbility;
use crate::component::Component;
use crate::context::ComponentContext;
use crate::engine::definition::NodeConfig;
use crate::error;
use crate::interrupt::{find_interrupt, InteractionState};
use crate::output::{FrameWriter, OutputFrame, END_NODE_STREAM};
use crate::reference;

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
    /// component card's name) and records the resolved inputs immediately.
    fn start_span(&self, ctx: &GraphContext, inputs: Value) -> Option<SpanHandle> {
        let tracer = ctx.session().tracer()?;
        let name = self
            .rt
            .node
            .name
            .clone()
            .unwrap_or_else(|| self.rt.component.card().name);
        let span = tracer.start_workflow_span(self.rt.id.clone(), name);
        span.set_inputs(inputs);
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
        let source = state.as_value();
        let inputs = resolve_inputs(&self.rt.node, self.rt.ability, &source);
        let scope = self.rt.scope.as_ref().ok_or_else(|| {
            Error::workflow(format!(
                "component '{}' executed without a run scope",
                self.rt.id
            ))
        })?;
        let context = ComponentContext::new(
            Arc::clone(ctx.session()),
            self.rt.id.clone(),
            self.rt.workflow_id.clone(),
        )
        .with_interactions(Arc::clone(&scope.interactions))
        .with_writer(Arc::clone(&scope.writer))
        .with_cancellation(ctx.cancellation().clone())
        .with_depth(scope.depth);

        let span = self.start_span(ctx, inputs.clone());

        match self.dispatch(&source, inputs, &context, scope).await {
            Ok(Dispatch::Route(targets)) => {
                scope.clear_failures(&self.rt.id);
                if let Some(span) = &span {
                    // A branch produces no output; the span still closes.
                    span.finish(Value::Null);
                }
                Ok(NodeOutcome::Route { targets })
            }
            Ok(Dispatch::Output(raw)) => {
                scope.clear_failures(&self.rt.id);
                let output = project_output(
                    self.rt.node.outputs_schema.as_ref(),
                    &source,
                    &self.rt.id,
                    &raw,
                );
                if let Some(span) = &span {
                    span.finish(output.clone());
                }
                state.set(self.rt.id.clone(), output);
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
                let attempts = scope.record_failure(&self.rt.id);
                if attempts > self.rt.node.max_retries {
                    if let Some(defaults) = self.recovered_output() {
                        // Python's `_run_error_recovery` calls `_post_invoke` with the
                        // handler's value, so the defaults go through `outputs_schema` too.
                        let output = project_output(
                            self.rt.node.outputs_schema.as_ref(),
                            &source,
                            &self.rt.id,
                            defaults,
                        );
                        if let Some(span) = &span {
                            span.set_metadata(
                                "error_recovery",
                                serde_json::json!({
                                    "handle_type": self
                                        .rt
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
                            component = %self.rt.id,
                            error = %failure,
                            "recovered a failed component with its configured default outputs"
                        );
                        scope.clear_failures(&self.rt.id);
                        state.set(self.rt.id.clone(), output);
                        return Ok(NodeOutcome::Continue);
                    }
                }
                let failure = error::component_execution_error(
                    self.rt.component.component_type(),
                    self.rt.ability.as_str(),
                    failure.to_string().as_str(),
                    &self.rt.workflow_id,
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

/// Resolves a node's input template (plus its stream template for `collect`/`transform`).
///
/// Without an `inputs_schema` the whole state object is handed to the component, which
/// is what Python's components do when no explicit input mapping exists.
fn resolve_inputs(node: &NodeConfig, ability: ComponentAbility, source: &Value) -> Value {
    let inputs = node.inputs_schema.as_ref().map_or_else(
        || source.clone(),
        |schema| reference::resolve_schema(schema, source),
    );
    if !matches!(
        ability,
        ComponentAbility::Collect | ComponentAbility::Transform
    ) {
        return inputs;
    }
    let Some(schema) = node.stream_inputs_schema.as_ref() else {
        return inputs;
    };
    merge_stream_inputs(inputs, reference::resolve_schema(schema, source))
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

/// Projects a component's raw output through the node's `outputs_schema`.
///
/// The schema resolves against the full workflow state plus `{"<id>": raw, "output":
/// raw}`, so a template can reference its own component id or the bare `output` key.
fn project_output(schema: Option<&Value>, source: &Value, id: &str, raw: &Value) -> Value {
    let Some(schema) = schema else {
        return raw.clone();
    };
    let mut scope = if source.is_object() {
        source.clone()
    } else {
        Value::Object(Map::new())
    };
    if let Value::Object(map) = &mut scope {
        map.insert(id.to_owned(), raw.clone());
        map.insert("output".to_owned(), raw.clone());
    }
    reference::resolve_schema(schema, &scope)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::assert_is_empty)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn input_schema_falls_back_to_the_whole_state() {
        let source = json!({"query": "hello"});
        assert_eq!(
            resolve_inputs(&NodeConfig::default(), ComponentAbility::Invoke, &source),
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
            resolve_inputs(&node, ComponentAbility::Invoke, &source),
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
            resolve_inputs(&node, ComponentAbility::Collect, &source),
            json!({"q": "streamed", "keep": 1})
        );
        // Batch abilities ignore the stream template.
        assert_eq!(
            resolve_inputs(&node, ComponentAbility::Invoke, &source),
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
            resolve_inputs(&node, ComponentAbility::Transform, &source),
            json!(["a", "b"])
        );
    }

    #[test]
    fn output_schema_sees_the_raw_value_under_its_own_id() {
        let source = json!({"start": {"query": "hello"}});
        let raw = json!({"text": "hi"});
        assert_eq!(
            project_output(
                Some(&json!({"answer": "${fake.text}"})),
                &source,
                "fake",
                &raw
            ),
            json!({"answer": "hi"})
        );
        assert_eq!(
            project_output(
                Some(&json!({"answer": "${output.text}"})),
                &source,
                "fake",
                &raw
            ),
            json!({"answer": "hi"})
        );
        // A missing path resolves to null and never fails.
        assert_eq!(
            project_output(Some(&json!({"answer": "${nope.x}"})), &source, "fake", &raw),
            json!({"answer": null})
        );
        // Without a schema the raw value is stored verbatim.
        assert_eq!(project_output(None, &source, "fake", &raw), raw);
    }

    #[test]
    fn output_schema_projects_against_a_non_object_state() {
        let raw = json!(7);
        assert_eq!(
            project_output(
                Some(&json!({"value": "${output}"})),
                &json!("scalar"),
                "c",
                &raw
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
}
