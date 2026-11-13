//! Running one component directly, without a workflow graph.
//!
//! Ports Python's `execute_single_component`
//! (`openjiuwen/core/workflow/_workflow.py`), which Python exports from the
//! `openjiuwen.core.workflow` package. It builds a session context for one component,
//! resolves an optional input template, invokes the component once, projects the result
//! through an optional output template, and reports the question when the component
//! interrupts for human input.
//!
//! ```
//! use std::sync::Arc;
//!
//! use openjiuwen_session::{Session, WorkflowSession};
//! use openjiuwen_workflow::components::StartComponent;
//! use openjiuwen_workflow::single::{execute_single_component, SingleComponentOptions};
//! use serde_json::json;
//!
//! # async fn example() -> openjiuwen_common::error::Result<()> {
//! let session: Arc<dyn Session> = Arc::new(WorkflowSession::new("s1"));
//! let output = execute_single_component(
//!     "start",
//!     Arc::new(StartComponent::new()),
//!     json!({"query": "hi"}),
//!     session.clone(),
//! )
//! .await?;
//! assert_eq!(output.result, json!({"query": "hi"}));
//!
//! let projected = execute_single_component(
//!     "start",
//!     Arc::new(StartComponent::new()),
//!     json!({"source": {"query": "hi"}}),
//!     session,
//! )
//! .await?;
//! assert_eq!(projected.result, json!({"source": {"query": "hi"}}));
//! # Ok(())
//! # }
//! ```
//!
//! # What is supported
//!
//! - The component's [`crate::Component::invoke`] ability, driven
//!   directly instead of through an `openjiuwen-graph` node.
//! - `inputs_schema`, resolved with the same `${node.field}` template semantics the
//!   engine uses ([`crate::reference::resolve_schema`]); without a schema the component
//!   receives the inputs unchanged.
//! - `outputs_schema`, resolved by the engine's own output-projection helper against the
//!   resolved inputs plus `{"<component_id>": raw, "output": raw}`, so `${output.field}`
//!   and `${<component_id>.field}` both work.
//! - Human-in-the-loop: an unseeded [`ComponentContext::interact`](crate::ComponentContext::interact)
//!   returns [`crate::WorkflowExecutionState::InputRequired`]
//!   with the same `{"type", "index", "payload"}` interaction wires an interrupted run
//!   reports (built by the engine's wire builder); seeding
//!   [`SingleComponentOptions::interactive`] answers the question instead, exactly as
//!   Python's `InteractiveInput` fast path does.
//!
//! Both mappings reuse the engine helpers instead of reimplementing them, so a
//! single-component call and a workflow run agree on output projection and on the shape
//! of an interrupted result.
//!
//! # Deviations from Python
//!
//! - **No graph, no snapshot.** Python marks the session with
//!   `SINGLE_COMPONENT_EXECUTION_KEY` and runs the component through a one-vertex Pregel
//!   graph so the checkpointer's `pre/post_workflow_execute` hooks fire. Rust drives the
//!   component directly: no graph node, no superstep, no barrier participation, and no
//!   checkpointer snapshot (a resumed single-component call therefore has nothing to
//!   restore).
//! - **Outputs are returned, not committed to session state.** The engine writes a node's
//!   result under its id in a graph run; here the projected value is only returned.
//! - **No `ModelContext` argument.** Python's optional `context` parameter (conversation
//!   history) has no counterpart: [`ComponentContext::history`](crate::ComponentContext::history)
//!   already reads and writes the session-backed `__messages__` lane.
//! - **Only `invoke` is dispatched.** Python also routes a single component's streaming
//!   abilities through the vertex; this port invokes the batch ability only.
//! - **No error wrapping.** A non-interrupt failure is returned verbatim instead of being
//!   wrapped in the engine's `[100054]` component-execution error.

use std::sync::Arc;

use openjiuwen_common::error::Result;
use openjiuwen_session::{InteractiveInput, Session};
use serde_json::Value;

use crate::component::Component;
use crate::context::ComponentContext;
use crate::engine::node::{interaction_wires, project_output};
use crate::interrupt::{find_interrupt, InteractionState};
use crate::output::{FrameWriter, WorkflowExecutionState, WorkflowOutput};
use crate::reference;

/// Per-call data mapping and seeded answers for [`execute_single_component_with`].
///
/// Every field is optional: the defaults invoke the component with the inputs unchanged,
/// return its output as-is, and start with no seeded human-in-the-loop answers.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SingleComponentOptions {
    /// The input template resolved against the call's inputs before the component runs.
    pub inputs_schema: Option<Value>,
    /// The output template resolved against the call's inputs and the component's output.
    pub outputs_schema: Option<Value>,
    /// Seeded human-in-the-loop answers, mirroring
    /// [`crate::engine::WorkflowRunOptions::interactive`].
    pub interactive: Option<InteractiveInput>,
}

impl SingleComponentOptions {
    /// Options that pass the inputs through unchanged and seed no answers.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Use `inputs_schema` as the component's input template.
    #[must_use]
    pub fn with_inputs_schema(mut self, inputs_schema: Value) -> Self {
        self.inputs_schema = Some(inputs_schema);
        self
    }

    /// Use `outputs_schema` as the component's output template.
    #[must_use]
    pub fn with_outputs_schema(mut self, outputs_schema: Value) -> Self {
        self.outputs_schema = Some(outputs_schema);
        self
    }

    /// Seed `interactive` as the call's human-in-the-loop answers.
    #[must_use]
    pub fn with_interactive(mut self, interactive: InteractiveInput) -> Self {
        self.interactive = Some(interactive);
        self
    }
}

/// Runs `component` once, with no input/output template and no seeded answers.
///
/// Shorthand for [`execute_single_component_with`] with
/// [`SingleComponentOptions::default`].
///
/// # Errors
///
/// Propagates the component's own failure. An interrupt is not an error: it is reported
/// as [`crate::WorkflowExecutionState::InputRequired`].
pub async fn execute_single_component(
    component_id: &str,
    component: Arc<dyn Component>,
    inputs: Value,
    session: Arc<dyn Session>,
) -> Result<WorkflowOutput> {
    execute_single_component_with(
        component_id,
        component,
        inputs,
        session,
        SingleComponentOptions::default(),
    )
    .await
}

/// Runs `component` once against `session`, applying `options`.
///
/// The component is registered as `component_id` (the Python call's `component_id`,
/// which also names the workflow session the context reports). An interrupt without a
/// seeded answer returns [`WorkflowExecutionState::InputRequired`] carrying one
/// `{"type": "__interaction__", "index", "payload": {"id", "index", "question"}}` wire per
/// pending question; otherwise the result is the projected output of a completed run.
///
/// # Errors
///
/// Propagates failures from the component, and from state/frame access on the session.
pub async fn execute_single_component_with(
    component_id: &str,
    component: Arc<dyn Component>,
    inputs: Value,
    session: Arc<dyn Session>,
    options: SingleComponentOptions,
) -> Result<WorkflowOutput> {
    let interactions = Arc::new(InteractionState::new());
    if let Some(interactive) = options.interactive.as_ref() {
        interactions.seed(interactive);
    }
    let writer = Arc::new(FrameWriter::new(None, Arc::clone(&session)));
    // Python builds a `WorkflowSession(workflow_id=component_id)` for the call, so the
    // component's workflow id is its own id here too.
    let context = ComponentContext::new(Arc::clone(&session), component_id, component_id)
        .with_interactions(Arc::clone(&interactions))
        .with_writer(writer)
        .with_depth(0);

    // Python: `SimpleNodeConfig(inputs_schema=inputs_schema)` — without a template the
    // component receives the committed inputs verbatim.
    let resolved = match options.inputs_schema.as_ref() {
        Some(schema) => reference::resolve_schema(schema, &inputs),
        None => inputs,
    };

    match component.invoke(resolved.clone(), &context).await {
        Ok(raw) => Ok(WorkflowOutput {
            result: project_output(
                options.outputs_schema.as_ref(),
                &resolved,
                component_id,
                &raw,
            ),
            state: WorkflowExecutionState::Completed,
        }),
        Err(failure) => {
            if find_interrupt(&failure).is_some() {
                // The `__interaction__` frames were already written by
                // `ComponentContext::interact`; the wires are what an interrupted run
                // reports as its result.
                return Ok(WorkflowOutput {
                    result: Value::Array(interaction_wires(&interactions)),
                    state: WorkflowExecutionState::InputRequired,
                });
            }
            Err(failure)
        }
    }
}
