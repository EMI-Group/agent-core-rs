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

use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::BoxStream;
use futures::StreamExt;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::utils::json as json_utils;
use openjiuwen_graph::{GraphContext, GraphState, Node, NodeOutcome};
use openjiuwen_session::tracer::SpanHandle;
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
        }
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
                if let Some(span) = &span {
                    // A branch produces no output; the span still closes.
                    span.finish(Value::Null);
                }
                Ok(NodeOutcome::Route { targets })
            }
            Ok(Dispatch::Output(raw)) => {
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
