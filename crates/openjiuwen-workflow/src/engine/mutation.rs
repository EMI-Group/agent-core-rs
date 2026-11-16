//! Per-run topology mutation: retune or rewire a compiled workflow for one run.
//!
//! Python mutates a workflow's graph between runs; a [`CompiledWorkflow`] in this crate is
//! immutable, so mutation is expressed as *a new compiled workflow for one run*:
//! [`CompiledWorkflow::mutated`] clones the recorded [`WorkflowDefinition`], applies the
//! caller's closure, re-validates it, and compiles the result. The receiver is never
//! touched, so the original workflow keeps running its original topology and one instance
//! still serves concurrent runs.
//!
//! This is deliberately **not** mid-execution mutation: the closure runs before the run
//! starts, so a topology change never races a superstep. Mutation also decides the run's
//! ability selection and spec again, because the mutated definition is validated and
//! compiled from scratch.
//!
//! The definition helpers below are the minimum a mutation closure needs: insert a
//! component (already available as [`WorkflowDefinition::insert_component`]), retune an
//! existing node, drop a node, and retarget or drop edges. Everything else is plain field
//! access (`start`, `end`, `card`).
//!
//! ```no_run
//! use std::sync::Arc;
//!
//! use openjiuwen_session::{Session, WorkflowSession};
//! use openjiuwen_workflow::engine::{CompiledWorkflow, NodeConfig, WorkflowRunOptions};
//! use openjiuwen_workflow::{EndComponent, EndConfig, WorkflowCard, WorkflowDefinition};
//! use serde_json::json;
//!
//! # async fn example(workflow: CompiledWorkflow) -> openjiuwen_common::error::Result<()> {
//! let session: Arc<dyn Session> = Arc::new(WorkflowSession::new("s1"));
//! let output = workflow
//!     .invoke_mutated(
//!         json!({"query": "hi"}),
//!         session,
//!         WorkflowRunOptions::default(),
//!         |definition| {
//!             // Retarget the end node onto a template end component for this run only.
//!             definition.insert_component(
//!                 "end",
//!                 Arc::new(EndComponent::with_config(EndConfig::new("{{start.query}}!"))),
//!                 NodeConfig::default(),
//!             );
//!         },
//!     )
//!     .await?;
//! assert_eq!(output.result, json!({"response": "hi!"}));
//! # Ok(())
//! # }
//! ```
use std::sync::Arc;

use futures::StreamExt;
use serde_json::Value;

use openjiuwen_common::error::Result;
use openjiuwen_session::Session;

use crate::engine::definition::{Entry, WorkflowDefinition};
use crate::engine::{CompiledWorkflow, WorkflowRunOptions};
use crate::output::{WorkflowChunk, WorkflowOutput};

impl WorkflowDefinition {
    /// The registered entry for `id`, mutable, so a per-run mutation can retune its node
    /// configuration (schemas, retries, timeout, declared abilities) in place.
    pub fn component_mut(&mut self, id: &str) -> Option<&mut Entry> {
        self.components.get_mut(id)
    }

    /// Removes the component `id` together with every edge and conditional reference to it.
    ///
    /// An edge loses `id` from its sources and is dropped when no source is left; an edge
    /// that targeted `id` is dropped whole. A conditional source keyed by `id` is dropped,
    /// and `id` is removed from every other source's conditional targets (dropping that
    /// source's entry when it has no target left). A start or end choice naming `id` is
    /// cleared, so a later mutation (or [`WorkflowDefinition`] field assignment) can pick
    /// another one; validation reports the gap otherwise.
    pub fn remove_component(&mut self, id: &str) -> &mut Self {
        self.components.remove(id);
        self.order.retain(|existing| existing != id);

        self.edges.retain_mut(|edge| {
            if edge.target == id {
                return false;
            }
            edge.sources.retain(|source| source != id);
            !edge.sources.is_empty()
        });

        self.conditional.remove(id);
        self.conditional.retain(|_, conditional| {
            conditional.targets.retain(|target| target != id);
            !conditional.targets.is_empty()
        });

        if self.start.as_deref() == Some(id) {
            self.start = None;
        }
        if self.end.as_deref() == Some(id) {
            self.end = None;
        }
        self
    }

    /// Drops every edge whose target is `target`.
    ///
    /// Used to rewire a join for one run without disturbing the rest of the topology; the
    /// edges leaving `target` are untouched.
    pub fn remove_edges_to(&mut self, target: &str) -> &mut Self {
        self.edges.retain(|edge| edge.target != target);
        self
    }

    /// Retargets the edge that connects `source` to `target` so it points at `new_target`.
    ///
    /// Every matching edge is retargeted, so a `sources → target` fan-in keeps its sources
    /// and simply feeds `new_target`. Returns whether at least one edge matched; a miss is
    /// reported rather than silently ignored. A caller that also needs a conditional
    /// retarget must update [`WorkflowDefinition::conditional`] itself, because a router's
    /// decision is not part of an ordinary edge.
    pub fn retarget_edge(
        &mut self,
        source: &str,
        target: &str,
        new_target: impl Into<String>,
    ) -> bool {
        let new_target = new_target.into();
        let mut matched = false;
        for edge in &mut self.edges {
            if edge.target == target && edge.sources.iter().any(|from| from == source) {
                edge.target.clone_from(&new_target);
                matched = true;
            }
        }
        matched
    }
}

impl CompiledWorkflow {
    /// Compiles a mutated copy of this workflow, leaving `self` untouched.
    ///
    /// `mutate` receives a clone of the recorded definition and may rewire it freely; the
    /// result is validated and compiled exactly like [`CompiledWorkflow::build`], so the
    /// returned workflow reports its own [`spec`](CompiledWorkflow::spec),
    /// [`card`](CompiledWorkflow::card), and [`ability`](CompiledWorkflow::ability) values.
    ///
    /// # Errors
    ///
    /// Returns the same coded compile error [`CompiledWorkflow::build`] reports, so an
    /// invalid mutation (an unregistered start, a dropped end, a dangling edge) fails here
    /// instead of running.
    pub fn mutated(&self, mutate: impl FnOnce(&mut WorkflowDefinition)) -> Result<Self> {
        let mut definition = self.definition().clone();
        mutate(&mut definition);
        Self::build(definition)
    }

    /// Runs this workflow once with `mutate` applied to a copy of its definition.
    ///
    /// The receiver is not modified: the mutated copy is compiled and run, and the original
    /// keeps its own topology for later runs. `mutate` runs before the run starts, so there
    /// is no mid-execution mutation.
    ///
    /// # Errors
    ///
    /// Returns the compile error when the mutation is invalid, then the same failures as
    /// [`CompiledWorkflow::invoke`] (including `Error::Unsupported` for
    /// [`StreamDelivery::Interleaved`](crate::engine::StreamDelivery::Interleaved)).
    pub async fn invoke_mutated(
        &self,
        inputs: Value,
        session: Arc<dyn Session>,
        options: WorkflowRunOptions,
        mutate: impl FnOnce(&mut WorkflowDefinition),
    ) -> Result<WorkflowOutput> {
        self.mutated(mutate)?.invoke(inputs, session, options).await
    }

    /// Streams this workflow once with `mutate` applied to a copy of its definition.
    ///
    /// # Errors
    ///
    /// An invalid mutation is delivered as the stream's single `Err` item (the same policy
    /// [`crate::Workflow::stream_with`] uses for a compile error); run failures behave like
    /// [`CompiledWorkflow::stream`].
    pub fn stream_mutated(
        &self,
        inputs: Value,
        session: Arc<dyn Session>,
        options: WorkflowRunOptions,
        mutate: impl FnOnce(&mut WorkflowDefinition),
    ) -> futures::stream::BoxStream<'static, Result<WorkflowChunk>> {
        match self.mutated(mutate) {
            Ok(workflow) => workflow.stream(inputs, session, options),
            Err(failure) => futures::stream::once(async move { Err(failure) }).boxed(),
        }
    }
}
