//! Built-in `sub_workflow` component: a compiled workflow driven as one component.
//!
//! Ports Python's `openjiuwen.core.workflow.components.flow.workflow_comp`
//! (`SubWorkflowComponent`, `SubWorkflowStreamState`, `SUB_WORKFLOW_COMPONENT`) together
//! with the two hooks it calls on the child workflow: `Workflow._sub_invoke`,
//! `Workflow._sub_stream`, and `Workflow._create_workflow_session(is_sub=True)` in
//! `openjiuwen.core.workflow.workflow`. The nesting guard mirrors Python's
//! `Workflow.compile` check (`openjiuwen/core/workflow/_workflow.py:297-309`).
//!
//! A [`SubWorkflowComponent`] holds a [`CompiledWorkflow`] and runs it against a *fresh
//! child session* whose parent is the calling component's session: config, stream hub,
//! tracer, and checkpointer are shared, but the child owns its own workflow state store,
//! so a child node id never collides with a parent node id of the same name.
//!
//! Invocation shapes:
//!
//! - **invoke** returns the child's end output unchanged (`{"output": …}` when the child
//!   ends with the stock `end` component), so downstream references read
//!   `${<sub node>.output.<key>}` exactly like Python.
//! - **stream** forwards the child's frames as their `{"type", "index", "payload"}`
//!   wires, dropping the child's own terminal `workflow_final` frame (the parent run
//!   publishes its own terminal frame).
//! - **Interrupts** are relayed: the child's `__interaction__` frames are re-published on
//!   the parent context and the component fails with the first interaction's interrupt
//!   error, so the parent run reports `INPUT_REQUIRED` with the same questions.
//!
//! Deviations from Python:
//!
//! - **Cache merge.** Python's `SubWorkflowStreamState.build_final_result` merges with
//!   `existing + value`, which concatenates strings, concatenates lists, adds numbers,
//!   and raises `TypeError` for mismatched types. Rust concatenates strings, extends
//!   arrays, and lets the newest value win for every other type (numbers, objects,
//!   mismatched pairs) — no run fails because two chunks disagreed on a type.
//! - **Empty cache.** Python's `get_stream_output` returns `{}` when nothing was
//!   accumulated and `invoke` then returns `{"output": {}}` without running the child.
//!   Here an empty cache falls through to a real child run, so a `cache_stream`
//!   component can still be invoked before it ever streamed.
//! - **Nested resume.** A parent `resume` forwards its seeded answers into the child run,
//!   keyed by interaction id (including Python's dotted `"<sub node id>.<child id>"`
//!   addressing, which is rewritten to the bare child id the child run reports), and
//!   records them in the parent session's component lane under
//!   [`NESTED_ANSWERS_KEY`]; a later resume replays that history before the newly
//!   supplied answers, so a replayed child consumes the answers it already saw in order.
//!   The child still starts from a *fresh* session and snapshot: state the child itself
//!   kept (questioner rounds, component lanes) does not survive into the next resume.
//! - **Buffered parent streams.** The engine buffers a non-end producer's frames (see
//!   `engine::node`), so a parent `stream` run surfaces this component's frames only when
//!   the sub-workflow node *is* the parent's end node; the frames are still buffered under
//!   the node id for downstream consumers, and a parent `invoke` relays the child's
//!   questions through `INPUT_REQUIRED` either way.
//! - Python streams the child through an actor network that live-interleaves producers;
//!   Rust relies on the engine's buffered composition, so the frames are the same but not
//!   interleaved in real time.

use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::BoxStream;
use futures::StreamExt;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_session::interaction::INTERACTION_FRAME_TYPE;
use openjiuwen_session::{Session, WorkflowSession, WorkflowSessionOptions};
use parking_lot::Mutex;
use serde_json::{json, Map, Value};

use crate::ability::ComponentAbility;
use crate::component::{Component, ComponentCard};
use crate::context::ComponentContext;
use crate::engine::{CompiledWorkflow, WorkflowRunOptions};
use crate::error;
use crate::interrupt::{interrupt_error, WorkflowInteraction};
use crate::output::{WorkflowExecutionState, WorkflowOutput, WORKFLOW_FINAL};
use crate::workflow::Workflow;

/// The component type discriminator of a nested workflow (Python
/// `SUB_WORKFLOW_COMPONENT`).
pub const SUB_WORKFLOW_COMPONENT: &str = "sub_workflow";

/// The parent session's component-state key holding the answers already forwarded to the
/// child, per interaction id.
///
/// The value is a JSON object `{"<interaction id>": [<answer>, …]}`; a later resume replays
/// it into the child run so the replayed child consumes the same answers in order.
pub const NESTED_ANSWERS_KEY: &str = "sub_workflow_answers";

/// A compiled workflow exposed as a single workflow component.
///
/// The component is immutable during a run, so one instance may serve concurrent runs;
/// only the optional stream cache is shared (and therefore per component instance, not
/// per run, matching Python's `SubWorkflowStreamState`).
pub struct SubWorkflowComponent {
    /// The child workflow, already validated and compiled.
    compiled: CompiledWorkflow,
    /// Whether stream frame payloads are accumulated for later `invoke` calls.
    cache_stream: bool,
    /// The maximum nesting level a child run may reach.
    max_nesting_depth: u32,
    /// Accumulated stream frame payloads (Python `SubWorkflowStreamState`).
    cache: Mutex<Vec<Value>>,
}

impl std::fmt::Debug for SubWorkflowComponent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SubWorkflowComponent")
            .field("workflow_id", &self.compiled.card().id)
            .field("cache_stream", &self.cache_stream)
            .field("max_nesting_depth", &self.max_nesting_depth)
            .finish_non_exhaustive()
    }
}

impl SubWorkflowComponent {
    /// Compiles `workflow` and wraps it.
    ///
    /// # Errors
    ///
    /// Propagates [`Workflow::build`] failures (the child's builder problems and
    /// validation errors).
    pub fn new(workflow: &Workflow) -> Result<Self> {
        Ok(Self::from_compiled(workflow.build()?))
    }

    /// Wraps an already compiled child workflow, taking the nesting limit from its
    /// configuration (`workflow_max_nesting_depth`, Python's default is `5`).
    #[must_use]
    pub fn from_compiled(compiled: CompiledWorkflow) -> Self {
        let max_nesting_depth = compiled.config().workflow_max_nesting_depth;
        Self {
            compiled,
            cache_stream: false,
            max_nesting_depth,
            cache: Mutex::new(Vec::new()),
        }
    }

    /// Enable or disable accumulating stream frame payloads for later `invoke` calls.
    #[must_use]
    pub fn with_cache_stream(mut self, cache_stream: bool) -> Self {
        self.cache_stream = cache_stream;
        self
    }

    /// Override the maximum nesting level of child runs.
    #[must_use]
    pub fn with_max_nesting_depth(mut self, max_nesting_depth: u32) -> Self {
        self.max_nesting_depth = max_nesting_depth;
        self
    }

    /// Whether stream frame payloads are cached.
    #[must_use]
    pub fn cache_stream(&self) -> bool {
        self.cache_stream
    }

    /// The maximum nesting level of child runs.
    #[must_use]
    pub fn max_nesting_depth(&self) -> u32 {
        self.max_nesting_depth
    }

    /// The compiled child workflow.
    #[must_use]
    pub fn compiled(&self) -> &CompiledWorkflow {
        &self.compiled
    }

    /// The cached stream output as a merged object (Python `get_stream_output`).
    ///
    /// `None` unless [`SubWorkflowComponent::cache_stream`] is enabled; an empty cache
    /// merges to `{}`.
    #[must_use]
    pub fn stream_output(&self) -> Option<Value> {
        if !self.cache_stream {
            return None;
        }
        let accumulated = self.cache.lock().clone();
        Some(build_final_result(&accumulated))
    }

    /// Records one forwarded frame payload when caching is enabled.
    fn accumulate(&self, payload: Value) {
        if self.cache_stream {
            self.cache.lock().push(payload);
        }
    }

    /// Enforces the nesting limit (`_workflow.py:297-309`).
    ///
    /// Python raises when `workflow_nesting_depth() > workflow_max_nesting_depth`, where
    /// the child's depth is this run's depth plus one; `depth >= max` is the same test.
    fn check_depth(&self, ctx: &ComponentContext) -> Result<()> {
        if ctx.depth() >= self.max_nesting_depth as usize {
            return Err(error::compile_error(
                &format!(
                    "workflow nesting hierarchy is too big, must <= {}",
                    self.max_nesting_depth
                ),
                ctx.workflow_id(),
            ));
        }
        Ok(())
    }

    /// The child session: a fresh state store whose parent is the calling session
    /// (Python's `_create_workflow_session(is_sub=True)`).
    fn child_session(&self, ctx: &ComponentContext) -> Arc<dyn Session> {
        Arc::new(WorkflowSession::with_options(WorkflowSessionOptions {
            workflow_id: Some(self.compiled.card().id.clone()),
            parent: Some(Arc::clone(ctx.session())),
            ..WorkflowSessionOptions::default()
        }))
    }

    /// The child run options: the parent's cancellation, one nesting level, and the
    /// forwarded resume answers.
    ///
    /// The answers already forwarded in earlier resumes are read back from the parent
    /// session's component lane ([`NESTED_ANSWERS_KEY`]) and merged with the ones queued in
    /// this run ([`ComponentContext::queued_answers`]), per id in FIFO order (recorded
    /// history first). The merged map is written back before the child runs, so the next
    /// resume replays the whole history to the replayed child.
    ///
    /// # Errors
    ///
    /// Propagates the component-state lane read and write failures.
    async fn child_options(&self, ctx: &ComponentContext) -> Result<WorkflowRunOptions> {
        let recorded = ctx.get_component_state(NESTED_ANSWERS_KEY).await?;
        let mut answers = match recorded {
            Some(Value::Object(recorded)) => recorded_answers(&recorded),
            Some(Value::Null) | None => BTreeMap::new(),
            Some(other) => {
                tracing::warn!(
                    workflow = ctx.workflow_id(),
                    component = %ctx.component_id(),
                    value = %other,
                    "the recorded nested answers are not an object; starting without them"
                );
                BTreeMap::new()
            }
        };
        for (id, values) in ctx.queued_answers() {
            let bare = bare_answer_id(&id, ctx.component_id());
            if bare != id {
                // Python addresses a nested interaction by its dotted node path
                // (`"sub.questioner"`); the child run reports the bare id, so the prefixed
                // key is folded into the suffix and dropped instead of being consumed twice.
                answers.remove(&id);
            }
            answers.entry(bare).or_default().extend(values);
        }
        let recorded = serde_json::to_value(&answers).map_err(|error| {
            Error::workflow(format!(
                "failed to serialize the nested workflow answers: {error}"
            ))
        })?;
        ctx.update_component_state(json!({ NESTED_ANSWERS_KEY: recorded }))
            .await?;
        Ok(WorkflowRunOptions::default()
            .with_cancellation(ctx.cancellation().clone())
            .with_depth(ctx.depth().saturating_add(1))
            .with_answers(answers))
    }
    /// The result the child `End` component produced (Python `_sub_invoke`).
    ///
    /// A `Completed` child already answers `{"output": …}`, so the value is returned
    /// unchanged. `InputRequired` relays every interaction wire through `ctx` and fails
    /// with the first one; any other state is an error rather than a silent success.
    async fn project(&self, output: WorkflowOutput, ctx: &ComponentContext) -> Result<Value> {
        match output.state {
            WorkflowExecutionState::Completed => Ok(output.result),
            WorkflowExecutionState::InputRequired => {
                let mut interactions = Vec::new();
                for wire in output.result.as_array().into_iter().flatten() {
                    let Some(interaction) = parse_interaction_wire(wire) else {
                        continue;
                    };
                    ctx.relay_interaction(interaction.clone()).await?;
                    interactions.push(interaction);
                }
                let Some(first) = interactions.into_iter().next() else {
                    return Err(Error::workflow(format!(
                        "sub workflow '{}' halted for input without reporting an interaction",
                        self.compiled.card().id
                    )));
                };
                Err(interrupt_error(first))
            }
            WorkflowExecutionState::Error => Err(Error::workflow(format!(
                "sub workflow '{}' finished in the '{}' state",
                self.compiled.card().id,
                WorkflowExecutionState::Error
            ))),
        }
    }
}

#[async_trait]
impl Component for SubWorkflowComponent {
    fn card(&self) -> ComponentCard {
        let card = self.compiled.card();
        let input_params = card
            .input_params
            .clone()
            .unwrap_or_else(|| json!({"type": "object"}));
        ComponentCard::new(card.name.clone(), SUB_WORKFLOW_COMPONENT)
            .with_id(card.id.clone())
            .with_description(format!(
                "runs workflow '{}' as one component",
                if card.name.is_empty() {
                    card.id.as_str()
                } else {
                    card.name.as_str()
                }
            ))
            .with_input_params(input_params)
            .with_output_params(json!({
                "output": {"description": format!("result of workflow '{}'", card.id)},
            }))
            .with_abilities(self.abilities())
    }

    fn abilities(&self) -> Vec<ComponentAbility> {
        vec![ComponentAbility::Invoke, ComponentAbility::Stream]
    }

    fn component_type(&self) -> &str {
        SUB_WORKFLOW_COMPONENT
    }

    async fn invoke(&self, inputs: Value, ctx: &ComponentContext) -> Result<Value> {
        // Python returns the cached stream output before running the child.
        if let Some(cached) = self.stream_output() {
            if cached.as_object().is_some_and(|merged| !merged.is_empty()) {
                return Ok(json!({"output": cached}));
            }
        }
        self.check_depth(ctx)?;
        let child_session = self.child_session(ctx);
        let child_options = self.child_options(ctx).await?;
        let output = self
            .compiled
            .invoke(inputs, child_session, child_options)
            .await?;
        self.project(output, ctx).await
    }

    fn stream(
        self: Arc<Self>,
        inputs: Value,
        ctx: Arc<ComponentContext>,
    ) -> BoxStream<'static, Result<Value>> {
        let stream = async_stream::try_stream! {
            self.check_depth(&ctx)?;
            let child_session = self.child_session(&ctx);
            let child_options = self.child_options(&ctx).await?;
            let mut frames = self
                .compiled
                .stream(inputs, child_session, child_options);
            let mut pending: Vec<WorkflowInteraction> = Vec::new();
            while let Some(frame) = frames.next().await {
                let chunk = frame?;
                // The child's terminal frame belongs to the child's own run; the parent
                // stream publishes its own `workflow_final`.
                if chunk.item_type == WORKFLOW_FINAL {
                    continue;
                }
                let wire = chunk.to_wire();
                if chunk.item_type == INTERACTION_FRAME_TYPE {
                    // The frame is forwarded verbatim, so recording is enough; the
                    // interrupt error below makes the parent run halt as INPUT_REQUIRED.
                    if let Some(interaction) = parse_interaction_wire(&wire) {
                        ctx.record_interaction(interaction.clone());
                        pending.push(interaction);
                    }
                } else {
                    self.accumulate(chunk.payload.clone());
                }
                yield wire;
            }
            if let Some(first) = pending.into_iter().next() {
                Err(interrupt_error(first))?;
            }
        };
        Box::pin(stream)
    }
}

/// Parses the recorded nested answers: every array value is that id's FIFO queue.
///
/// A lone non-array value is read as a one-element queue (tolerating a hand-written lane),
/// and a non-array carries no answers at all.
fn recorded_answers(recorded: &Map<String, Value>) -> BTreeMap<String, Vec<Value>> {
    let mut answers: BTreeMap<String, Vec<Value>> = BTreeMap::new();
    for (id, value) in recorded {
        match value {
            Value::Array(values) => {
                answers
                    .entry(id.clone())
                    .or_default()
                    .extend(values.iter().cloned());
            }
            Value::Null => {}
            other => {
                answers.entry(id.clone()).or_default().push(other.clone());
            }
        }
    }
    answers
}

/// Re-keys a dotted, node-addressed interaction id (`"<sub node id>.<child id>"`) under the
/// bare child id a child run reports.
///
/// Python addresses a nested interaction by its dotted node path while this crate's
/// components raise the bare child id, so both spellings must reach the same queue.
fn bare_answer_id(id: &str, component_id: &str) -> String {
    if component_id.is_empty() {
        return id.to_owned();
    }
    match id.strip_prefix(component_id) {
        Some(rest) if rest.len() > 1 && rest.starts_with('.') => rest[1..].to_owned(),
        _ => id.to_owned(),
    }
}

/// Parses one `__interaction__` frame wire into a [`WorkflowInteraction`].
///
/// The wire is `{"type", "index", "payload": {"id", "index", "question"}}`; `id` is
/// required, while a missing index defaults to `0` and a missing question to `null`.
fn parse_interaction_wire(wire: &Value) -> Option<WorkflowInteraction> {
    let payload = wire.get("payload")?;
    let id = payload.get("id")?.as_str()?.to_owned();
    let index = payload
        .get("index")
        .and_then(Value::as_u64)
        .map_or(0, |index| u32::try_from(index).unwrap_or(u32::MAX));
    let question = payload.get("question").cloned().unwrap_or(Value::Null);
    Some(WorkflowInteraction::new(id, index, question))
}

/// Merges accumulated stream frame payloads (Python
/// `SubWorkflowStreamState.build_final_result`).
///
/// A payload that is an object contributes its `output` sub-object when present (Python's
/// `{'output': {'out': 'xxx'}}` nesting) and its own keys otherwise. Strings concatenate
/// and arrays extend; every other pair keeps the newest value (see the module-level
/// deviation note on Python's `existing + value`).
fn build_final_result(outputs: &[Value]) -> Value {
    let mut merged = Map::new();
    for output in outputs {
        let Value::Object(map) = output else {
            continue;
        };
        let inner = map.get("output").and_then(Value::as_object).unwrap_or(map);
        for (key, value) in inner {
            match (merged.get(key), value) {
                (Some(Value::String(existing)), Value::String(new)) => {
                    merged.insert(key.clone(), Value::String(format!("{existing}{new}")));
                }
                (Some(Value::Array(existing)), Value::Array(new)) => {
                    let mut combined = existing.clone();
                    combined.extend(new.iter().cloned());
                    merged.insert(key.clone(), Value::Array(combined));
                }
                _ => {
                    merged.insert(key.clone(), value.clone());
                }
            }
        }
    }
    Value::Object(merged)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::assert_is_empty)]
mod tests {
    use super::*;
    use crate::card::{WorkflowCard, WorkflowConfig};
    use crate::components::{EndComponent, StartComponent};
    use crate::workflow::ComponentOptions;

    /// `start → end`, the smallest compilable child workflow.
    fn child() -> Workflow {
        let mut workflow = Workflow::new(WorkflowCard::new("child"));
        workflow
            .set_start_comp("start", StartComponent::new(), ComponentOptions::new())
            .set_end_comp("end", EndComponent::new(), ComponentOptions::new())
            .add_connection("start", "end");
        workflow
    }

    #[test]
    fn identity_and_card_describe_the_child_workflow() {
        let component = SubWorkflowComponent::new(&child()).unwrap();

        assert_eq!(component.component_type(), SUB_WORKFLOW_COMPONENT);
        assert_eq!(
            component.abilities(),
            vec![ComponentAbility::Invoke, ComponentAbility::Stream]
        );
        assert_eq!(component.compiled().card().id, "child");
        assert!(!component.cache_stream());
        assert_eq!(component.stream_output(), None);

        let card = component.card();
        assert_eq!(card.id, "child");
        assert_eq!(card.name, "child");
        assert_eq!(card.component_type, SUB_WORKFLOW_COMPONENT);
        assert!(card.description.contains("child"), "{card:?}");
        assert_eq!(card.input_params, json!({"type": "object"}));
        assert_eq!(
            card.output_params["output"]["description"],
            json!("result of workflow 'child'")
        );
        assert_eq!(
            card.abilities,
            vec![ComponentAbility::Invoke, ComponentAbility::Stream]
        );
    }

    #[test]
    fn the_nesting_limit_defaults_to_the_child_config() {
        let config = WorkflowConfig {
            card: WorkflowCard::new("child"),
            workflow_max_nesting_depth: 2,
            ..WorkflowConfig::default()
        };
        let mut workflow = Workflow::with_config(config);
        workflow
            .set_start_comp("start", StartComponent::new(), ComponentOptions::new())
            .set_end_comp("end", EndComponent::new(), ComponentOptions::new())
            .add_connection("start", "end");

        let component = SubWorkflowComponent::new(&workflow).unwrap();
        assert_eq!(component.max_nesting_depth(), 2);
        assert_eq!(
            SubWorkflowComponent::new(&child())
                .unwrap()
                .max_nesting_depth(),
            5,
            "Python's default nesting depth"
        );
        assert_eq!(
            component.with_max_nesting_depth(0).max_nesting_depth(),
            0,
            "the override wins"
        );
    }

    #[test]
    fn the_stream_cache_is_opt_in() {
        let component = SubWorkflowComponent::new(&child()).unwrap();
        assert_eq!(component.stream_output(), None);

        let caching = SubWorkflowComponent::new(&child())
            .unwrap()
            .with_cache_stream(true);
        assert!(caching.cache_stream());
        assert_eq!(caching.stream_output(), Some(json!({})));
    }

    // port of tests/unit_tests/core/component/test_sub_workflow_comp.py::test_build_final_result
    #[test]
    fn merge_concatenates_strings_and_extends_arrays() {
        assert_eq!(
            build_final_result(&[json!({"a": "x", "l": [1]}), json!({"a": "y", "l": [2, 3]})]),
            json!({"a": "xy", "l": [1, 2, 3]})
        );
    }

    #[test]
    fn merge_keeps_the_newest_number_and_object() {
        assert_eq!(
            build_final_result(&[
                json!({"n": 1, "o": {"k": 1}}),
                json!({"n": 2, "o": {"k": 2}})
            ]),
            json!({"n": 2, "o": {"k": 2}})
        );
        assert_eq!(
            build_final_result(&[json!({"n": "text"}), json!({"n": 7})]),
            json!({"n": 7}),
            "a type mismatch keeps the newest value instead of raising"
        );
    }

    #[test]
    fn merge_unwraps_the_nested_output_key() {
        assert_eq!(
            build_final_result(&[
                json!({"output": {"response": "a"}}),
                json!({"output": {"response": "b"}}),
            ]),
            json!({"response": "ab"})
        );
        // A payload without `output` contributes its own keys.
        assert_eq!(
            build_final_result(&[json!({"output": {"r": "a"}}), json!({"r": "b"})]),
            json!({"r": "ab"})
        );
    }

    #[test]
    fn merge_ignores_non_object_payloads_and_empty_input() {
        assert_eq!(build_final_result(&[]), json!({}));
        assert_eq!(
            build_final_result(&[json!("plain"), json!(7), json!(null)]),
            json!({})
        );
    }

    #[test]
    fn interaction_wires_parse_with_defaults() {
        let wire = json!({
            "type": INTERACTION_FRAME_TYPE,
            "index": 2,
            "payload": {"id": "ask", "index": 2, "question": {"text": "continue?"}}
        });
        let interaction = parse_interaction_wire(&wire).unwrap();
        assert_eq!(interaction.id, "ask");
        assert_eq!(interaction.index, 2);
        assert_eq!(interaction.value, json!({"text": "continue?"}));

        let minimal = parse_interaction_wire(&json!({"payload": {"id": "q"}})).unwrap();
        assert_eq!((minimal.id.as_str(), minimal.index), ("q", 0));
        assert_eq!(minimal.value, Value::Null);

        assert_eq!(parse_interaction_wire(&json!({"payload": {}})), None);
        assert_eq!(parse_interaction_wire(&json!({})), None);
    }
}
