//! Plan execution: compile an [`OrchestrationPlan`] onto [`openjiuwen_graph`] and run it.
//!
//! Python's symphony engine hands its plan to a host-driven `graph_engine` facade that
//! has no executor of its own. The Rust port closes that gap: [`PlanExecutor::compile`]
//! turns a plan into a real [`CompiledGraph`] (one node per plan step, one edge per plan
//! edge), and [`PlanExecutor::execute`] runs it through a [`CapabilityRunner`] while
//! recording one outcome record per step.
//!
//! # Compilation mapping
//!
//! - Every plan step becomes one graph node whose id is the step's qualified id
//!   ([`PlanStep::step_id`]).
//! - Every plan edge becomes one plain `source -> target` graph edge (duplicate edges
//!   collapse, and the set is sorted so the compiled spec is canonical).
//! - Steps with no incoming edge are the graph entries; steps with no outgoing edge are
//!   the exits.
//! - Nodes carry [`NodeOptions`] with [`ExecutorConfig::node_timeout_ms`] and
//!   `max_retries: 0`, so a timeout is reported instead of silently retried.
//!
//! # Input overlay
//!
//! A step's invocation payload starts from the top-level `state["inputs"]` object (a
//! non-object seed is ignored) and is then overlaid by its predecessors' outputs, which
//! are read from the top-level state key each predecessor writes:
//!
//! 1. A declared predecessor output whose normalized name ([`normalize_name_key`])
//!    matches one of the step's input names fills that input. When the predecessor's
//!    returned JSON is an object, the matching member is used (exact key first, then a
//!    normalized key match); otherwise the whole value is the output.
//! 2. A predecessor that declares exactly one output and matches none of the step's
//!    input names fills the step's single remaining unmatched input with that output,
//!    so a `tool:a -> tool:b` link works even when the names differ.
//!
//! A missing record, a record without an `output` member, and a `null` output are all
//! treated as "nothing to overlay" — never as a panic.
//!
//! # Outcome records
//!
//! Each node writes exactly one top-level state key, the step id, holding
//! `{"success": bool, "output": Value, "error": Option<String>, "duration_ms": u64}`.
//! The duration comes from a monotonic [`std::time::Instant`]. A failed runner call
//! still writes its record before the failure propagates.
//!
//! # Failure semantics
//!
//! With [`ExecutorConfig::fail_fast`] (the default) the first runner error aborts the
//! run: the node returns an [`Error::Graph`] whose message is exactly
//! ``step `<id>` failed: <runner message>``, so [`PlanExecutor::execute`] returns that
//! error. The graph engine wraps a node failure as
//! ``node `<id>` failed at superstep <n>: <node error>``; the executor trims that
//! context back off (the module-private failure normaliser) so callers see exactly the documented
//! message. Two structured kinds survive the abort unchanged, because callers are
//! expected to act on them: [`ErrorKind::Cancelled`] stays a cancellation and
//! [`ErrorKind::Unsupported`] (a deferred capability) keeps `ErrorKind::Unsupported`
//! while keeping the same message shape.
//!
//! With `fail_fast = false` every step runs, a failing step records
//! `success: false` plus its error, and the run returns `Ok(PlanExecution)`; callers use
//! [`PlanExecution::failed_steps`] to inspect the failures.
//!
//! # Seeding caller inputs
//!
//! [`PlanExecutor::execute`] has no caller-input parameter, so it runs the graph with an
//! empty state: the overlay then starts from each predecessor's output only. A caller
//! that must supply request inputs seeds the graph itself by invoking the compiled
//! graph directly with `{"inputs": {...}}` (see [`PlanExecutor::compile`]); the executor
//! documents that shape because [`PlanExecutor::execute`] never invents it.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use openjiuwen_common::utils::id::uuid_hex;
use openjiuwen_graph::{
    CompiledGraph, ExecutionOptions, GraphBuilder, GraphContext, GraphState, Node, NodeOptions,
    NodeOutcome,
};
use openjiuwen_session::{Session, WorkflowSession};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tokio_util::sync::CancellationToken;

use crate::error::{validation, Error, ErrorKind, Result};
use crate::model::{normalize_name_key, CapabilityDescriptor};
use crate::orchestration::plan::{OrchestrationPlan, PlanStep};

/// The default cap on a plan's step count.
const DEFAULT_MAX_STEPS: usize = 32;

/// The top-level state key carrying caller-supplied step inputs.
const SEED_INPUTS_KEY: &str = "inputs";

/// The state-key member holding a step record's success flag.
const RECORD_SUCCESS: &str = "success";
/// The state-key member holding a step record's output value.
const RECORD_OUTPUT: &str = "output";
/// The state-key member holding a step record's error message.
const RECORD_ERROR: &str = "error";
/// The state-key member holding a step record's monotonic duration.
const RECORD_DURATION_MS: &str = "duration_ms";

/// The message recorded for a plan step that never produced a state record.
const MISSING_RECORD_ERROR: &str = "step did not run";

/// The context handed to every [`CapabilityRunner::invoke`] call.
///
/// It names the step being executed, the session the graph runs in, and the run's
/// cancellation token so a long capability can stop cooperatively.
#[derive(Debug, Clone)]
pub struct RunnerContext {
    /// The plan step's qualified capability id (also the graph node id).
    pub step_id: String,
    /// The id of the session the graph run executes in.
    pub session_id: Option<String>,
    /// The run's cancellation token; cancelling it aborts the graph run.
    pub cancellation: CancellationToken,
}

/// Runs one capability of a compiled plan.
///
/// The executor never knows how a capability is invoked, so a host supplies the
/// strategy: a tool registry, an MCP client, a sub-agent, or an in-process fake in tests.
/// Implementations must be `Send + Sync` because steps of one superstep run
/// concurrently.
#[async_trait]
pub trait CapabilityRunner: Send + Sync {
    /// Invokes `capability` with the overlaid `inputs`.
    ///
    /// `context` carries the step id, the session id, and the run's cancellation token.
    ///
    /// # Errors
    ///
    /// Returns an [`Error`] when the capability fails. With
    /// [`ExecutorConfig::fail_fast`] the first error aborts the run; otherwise it is
    /// recorded and execution continues.
    async fn invoke(
        &self,
        capability: &CapabilityDescriptor,
        inputs: Value,
        context: &RunnerContext,
    ) -> Result<Value>;
}

/// A [`CapabilityRunner`] that refuses every invocation.
///
/// Used when no host runner is configured: compiling and planning still work, and
/// execution reports the documented deferred-scope error instead of panicking.
#[derive(Debug, Clone, Copy, Default)]
pub struct UnsupportedRunner;

impl UnsupportedRunner {
    /// Builds the runner.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

#[async_trait]
impl CapabilityRunner for UnsupportedRunner {
    async fn invoke(
        &self,
        _capability: &CapabilityDescriptor,
        _inputs: Value,
        _context: &RunnerContext,
    ) -> Result<Value> {
        Err(crate::error::unsupported("capability execution"))
    }
}

/// Execution limits and failure handling.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ExecutorConfig {
    /// The maximum number of plan steps that may be executed.
    pub max_steps: usize,
    /// Per-step timeout in milliseconds; `None` leaves the graph default (no timeout).
    pub node_timeout_ms: Option<u64>,
    /// Whether the first step failure aborts the run.
    pub fail_fast: bool,
}

impl Default for ExecutorConfig {
    /// Returns `max_steps = 32`, no per-step timeout, and `fail_fast = true`.
    fn default() -> Self {
        Self {
            max_steps: DEFAULT_MAX_STEPS,
            node_timeout_ms: None,
            fail_fast: true,
        }
    }
}

/// What one plan step did.
///
/// `step_id` is the qualified capability id, and `duration_ms` measures the runner call
/// with a monotonic clock.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct StepOutcome {
    /// The plan step's qualified capability id.
    pub step_id: String,
    /// The step's capability type.
    pub capability_type: String,
    /// Whether the runner returned an output.
    pub success: bool,
    /// The runner's output; [`Value::Null`] when the step failed.
    pub output: Value,
    /// The failure message; `None` when the step succeeded.
    pub error: Option<String>,
    /// The runner call's monotonic duration in milliseconds.
    pub duration_ms: u64,
}

/// The result of executing one plan.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct PlanExecution {
    /// The executed plan's id.
    pub plan_id: String,
    /// The compiled graph's id, which equals the plan id.
    pub graph_id: String,
    /// One outcome per plan step, keyed and ordered by step id.
    pub steps: BTreeMap<String, StepOutcome>,
    /// The output of every successful step, keyed by step id.
    pub outputs: BTreeMap<String, Value>,
    /// The final graph state as a JSON object.
    pub state: Value,
}

impl PlanExecution {
    /// Returns the failed steps, ordered by step id.
    #[must_use]
    pub fn failed_steps(&self) -> Vec<&StepOutcome> {
        self.steps.values().filter(|step| !step.success).collect()
    }

    /// Reports whether every recorded step succeeded (true for a plan with no steps).
    #[must_use]
    pub fn succeeded(&self) -> bool {
        self.steps.values().all(|step| step.success)
    }
}

/// Compiles plans onto [`openjiuwen_graph`] and executes them through a
/// [`CapabilityRunner`].
#[derive(Clone)]
pub struct PlanExecutor {
    config: ExecutorConfig,
    runner: Arc<dyn CapabilityRunner>,
}

impl PlanExecutor {
    /// Builds an executor with [`ExecutorConfig::default`].
    #[must_use]
    pub fn new(runner: Arc<dyn CapabilityRunner>) -> Self {
        Self {
            config: ExecutorConfig::default(),
            runner,
        }
    }

    /// Builds an executor with an explicit configuration.
    #[must_use]
    pub fn with_config(config: ExecutorConfig, runner: Arc<dyn CapabilityRunner>) -> Self {
        Self { config, runner }
    }

    /// Compiles `plan` into a runnable graph.
    ///
    /// Nodes are created for the plan's steps in plan order, edges are deduplicated and
    /// sorted, and entries/exits are derived from the edge set. The returned graph can
    /// be invoked directly when the caller must seed `{"inputs": {...}}`.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::Validation`] when the plan exceeds
    /// [`ExecutorConfig::max_steps`], repeats a step id, has an edge whose endpoint is
    /// not a step, contains a cycle (via [`OrchestrationPlan::stages`]), or names a step
    /// whose capability is absent from `capabilities`. Graph construction failures are
    /// returned as [`ErrorKind::Graph`].
    pub fn compile(
        &self,
        plan: &OrchestrationPlan,
        capabilities: &[CapabilityDescriptor],
    ) -> Result<CompiledGraph> {
        self.validate_plan(plan, capabilities)?;

        let resolved: BTreeMap<String, &CapabilityDescriptor> = capabilities
            .iter()
            .map(|capability| (capability.qualified_id(), capability))
            .collect();
        let edges = plan_edges(plan);
        let predecessors = predecessors(plan, &edges);
        let options = NodeOptions {
            timeout_ms: self.config.node_timeout_ms,
            max_retries: 0,
        };

        let mut builder = GraphBuilder::new(plan.plan_id.clone());
        for step in &plan.steps {
            let Some(capability) = resolved.get(&step.step_id) else {
                return Err(validation(
                    "step_id",
                    format!("no capability `{}` in the capability set", step.step_id),
                ));
            };
            let node = StepNode {
                step_id: step.step_id.clone(),
                capability: (*capability).clone(),
                runner: Arc::clone(&self.runner),
                input_names: step.inputs.iter().map(|io| io.name.clone()).collect(),
                predecessors: predecessors
                    .get(&step.step_id)
                    .map(|steps| {
                        steps
                            .iter()
                            .map(|step| Predecessor {
                                step_id: step.step_id.clone(),
                                output_names: step
                                    .outputs
                                    .iter()
                                    .map(|io| io.name.clone())
                                    .collect(),
                            })
                            .collect()
                    })
                    .unwrap_or_default(),
                fail_fast: self.config.fail_fast,
            };
            builder = builder.node_with(step.step_id.clone(), node, options.clone());
        }

        for (source, target) in &edges {
            builder = builder.edge(source.clone(), target.clone());
        }
        for entry in endpoints(plan, &edges, Endpoint::Entry) {
            builder = builder.entry(entry);
        }
        for exit in endpoints(plan, &edges, Endpoint::Exit) {
            builder = builder.exit(exit);
        }
        builder.compile()
    }

    /// Executes `plan`, returning one outcome per step.
    ///
    /// `session` defaults to a fresh [`WorkflowSession`] with a generated id. The graph
    /// is seeded with an empty state; caller-supplied inputs reach the graph only when a
    /// caller invokes the compiled graph directly with `{"inputs": {...}}`.
    ///
    /// A plan with no steps has nothing to run and returns an empty [`PlanExecution`].
    ///
    /// # Errors
    ///
    /// Returns the [`PlanExecutor::compile`] validation errors, and with
    /// [`ExecutorConfig::fail_fast`] the first runner failure as an [`ErrorKind::Graph`]
    /// error (see the module docs for the preserved kinds). Engine-level failures such as
    /// a node timeout or a cancellation are returned unchanged.
    pub async fn execute(
        &self,
        plan: &OrchestrationPlan,
        capabilities: &[CapabilityDescriptor],
        session: Option<Arc<dyn Session>>,
    ) -> Result<PlanExecution> {
        self.validate_plan(plan, capabilities)?;
        if plan.steps.is_empty() {
            return Ok(PlanExecution {
                plan_id: plan.plan_id.clone(),
                graph_id: plan.plan_id.clone(),
                state: Value::Object(Map::new()),
                ..PlanExecution::default()
            });
        }

        let graph = self.compile(plan, capabilities)?;
        let session = session.unwrap_or_else(default_session);
        let run = graph
            .run(
                Value::Object(Map::new()),
                session,
                ExecutionOptions::default(),
            )
            .await
            .map_err(|error| trim_engine_context(plan, error))?;

        let mut steps = BTreeMap::new();
        let mut outputs = BTreeMap::new();
        let mut records = plan.steps.clone();
        records.sort_by(|left, right| left.step_id.cmp(&right.step_id));
        for step in &records {
            let outcome = match run.state.get(&step.step_id) {
                Some(record) => {
                    let mut outcome: StepOutcome =
                        serde_json::from_value(record.clone()).map_err(|error| {
                            validation("step_outcome", format!("step `{}`: {error}", step.step_id))
                        })?;
                    outcome.step_id.clone_from(&step.step_id);
                    outcome.capability_type.clone_from(&step.capability_type);
                    outcome
                }
                None => StepOutcome {
                    step_id: step.step_id.clone(),
                    capability_type: step.capability_type.clone(),
                    success: false,
                    output: Value::Null,
                    error: Some(MISSING_RECORD_ERROR.to_string()),
                    duration_ms: 0,
                },
            };
            if outcome.success {
                outputs.insert(step.step_id.clone(), outcome.output.clone());
            }
            steps.insert(step.step_id.clone(), outcome);
        }

        Ok(PlanExecution {
            plan_id: plan.plan_id.clone(),
            graph_id: graph.id().to_string(),
            steps,
            outputs,
            state: run.state.into_value(),
        })
    }

    /// Checks everything the plan must satisfy before a graph is built.
    ///
    /// The checks run in a fixed order so the reported error is deterministic: step
    /// count, step ids, the plan's own edge layering ([`OrchestrationPlan::stages`],
    /// which rejects unknown endpoints and cycles), then capability resolution.
    fn validate_plan(
        &self,
        plan: &OrchestrationPlan,
        capabilities: &[CapabilityDescriptor],
    ) -> Result<()> {
        if plan.steps.len() > self.config.max_steps {
            return Err(validation(
                "max_steps",
                format!(
                    "plan `{}` has {} steps but the limit is {}",
                    plan.plan_id,
                    plan.steps.len(),
                    self.config.max_steps
                ),
            ));
        }

        let mut seen: BTreeSet<&str> = BTreeSet::new();
        for step in &plan.steps {
            if step.step_id.trim().is_empty() {
                return Err(validation("step_id", "must not be blank"));
            }
            if !seen.insert(step.step_id.as_str()) {
                return Err(validation(
                    "step_id",
                    format!("duplicate step id `{}`", step.step_id),
                ));
            }
        }

        plan.stages()?;

        let available: BTreeSet<String> = capabilities
            .iter()
            .map(CapabilityDescriptor::qualified_id)
            .collect();
        for step in &plan.steps {
            if !available.contains(&step.step_id) {
                return Err(validation(
                    "step_id",
                    format!("no capability `{}` in the capability set", step.step_id),
                ));
            }
        }
        Ok(())
    }
}

impl std::fmt::Debug for PlanExecutor {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PlanExecutor")
            .field("config", &self.config)
            .field("runner", &"dyn CapabilityRunner")
            .finish()
    }
}

/// Builds the default execution session.
fn default_session() -> Arc<dyn Session> {
    Arc::new(WorkflowSession::new(uuid_hex())) as Arc<dyn Session>
}

/// Strips the graph engine's wrapper off a step failure.
///
/// The engine reports a node failure as
/// ``node `<id>` failed at superstep <n>: <node error>``. The executor owns the node
/// error, so it trims that engine context back off using the failing step's own
/// ``step `<id>` failed: `` marker: the caller then sees exactly the documented message
/// and the error keeps its kind. A failure without a marker (a timeout, a cancellation,
/// a channel error) is returned unchanged.
fn trim_engine_context(plan: &OrchestrationPlan, error: Error) -> Error {
    let message = error.message().into_owned();
    for step in &plan.steps {
        let marker = format!("step `{}` failed: ", step.step_id);
        if let Some(index) = message.find(&marker) {
            let trimmed = message[index..].to_string();
            return match error.kind() {
                ErrorKind::Unsupported => Error::unsupported(trimmed),
                _ => Error::graph(trimmed),
            };
        }
    }
    error
}

/// One declared predecessor of a step, as the overlay needs it.
struct Predecessor {
    /// The predecessor's step id, which is also its state key.
    step_id: String,
    /// The predecessor's declared output names, in plan order.
    output_names: Vec<String>,
}

/// One plan step's graph node.
struct StepNode {
    /// The step id, used as the node id and the state key.
    step_id: String,
    /// The capability handed to the runner.
    capability: CapabilityDescriptor,
    /// The host's invocation strategy.
    runner: Arc<dyn CapabilityRunner>,
    /// The step's declared input names, in plan order.
    input_names: Vec<String>,
    /// The step's predecessors, sorted by step id.
    predecessors: Vec<Predecessor>,
    /// Whether a runner error aborts the run.
    fail_fast: bool,
}

impl StepNode {
    /// Builds the invocation payload: the caller seed overlaid with predecessor outputs.
    fn collect_inputs(&self, state: &GraphState) -> Value {
        let mut inputs: Map<String, Value> = match state.get(SEED_INPUTS_KEY) {
            Some(Value::Object(map)) => map.clone(),
            _ => Map::new(),
        };

        let input_keys: Vec<(&str, String)> = self
            .input_names
            .iter()
            .map(|name| (name.as_str(), normalize_name_key(name)))
            .collect();
        let mut filled: BTreeSet<String> = input_keys
            .iter()
            .filter(|(name, _)| inputs.contains_key(*name))
            .map(|(_, key)| key.clone())
            .collect();

        for predecessor in &self.predecessors {
            let Some(record) = state.get(&predecessor.step_id) else {
                continue;
            };
            let Some(output) = record.get(RECORD_OUTPUT) else {
                continue;
            };
            if output.is_null() {
                continue;
            }
            let resolved = resolve_outputs(output, &predecessor.output_names);
            let mut matched = false;
            for (output_key, value) in &resolved {
                for (name, key) in &input_keys {
                    if key == output_key {
                        inputs.insert((*name).to_string(), value.clone());
                        filled.insert(key.clone());
                        matched = true;
                    }
                }
            }
            if matched || resolved.len() != 1 {
                continue;
            }
            let remaining: Vec<&str> = input_keys
                .iter()
                .filter(|(_, key)| !filled.contains(key))
                .map(|(name, _)| *name)
                .collect();
            if let [only] = remaining.as_slice() {
                if let Some((_, value)) = resolved.first() {
                    inputs.insert((*only).to_string(), value.clone());
                    filled.insert(normalize_name_key(only));
                }
            }
        }

        Value::Object(inputs)
    }

    /// Builds the state record this step writes.
    fn record(success: bool, output: Value, error: Option<String>, duration_ms: u64) -> Value {
        let mut record = Map::new();
        record.insert(RECORD_SUCCESS.to_string(), Value::Bool(success));
        record.insert(RECORD_OUTPUT.to_string(), output);
        record.insert(
            RECORD_ERROR.to_string(),
            error.map_or(Value::Null, Value::String),
        );
        record.insert(RECORD_DURATION_MS.to_string(), Value::from(duration_ms));
        Value::Object(record)
    }
}

#[async_trait]
impl Node for StepNode {
    fn id(&self) -> &str {
        &self.step_id
    }

    async fn run(&self, state: &mut GraphState, context: &GraphContext) -> Result<NodeOutcome> {
        let started = Instant::now();
        let inputs = self.collect_inputs(state);
        let runner_context = RunnerContext {
            step_id: self.step_id.clone(),
            session_id: Some(context.session().id().to_string()),
            cancellation: context.cancellation().clone(),
        };
        let result = self
            .runner
            .invoke(&self.capability, inputs, &runner_context)
            .await;
        let duration_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);

        match result {
            Ok(output) => {
                state.set(
                    self.step_id.clone(),
                    Self::record(true, output, None, duration_ms),
                );
                Ok(NodeOutcome::Continue)
            }
            Err(error) => {
                let message = error.message().into_owned();
                state.set(
                    self.step_id.clone(),
                    Self::record(false, Value::Null, Some(message.clone()), duration_ms),
                );
                if !self.fail_fast {
                    return Ok(NodeOutcome::Continue);
                }
                Err(match error.kind() {
                    ErrorKind::Cancelled => Error::cancelled(),
                    ErrorKind::Unsupported => {
                        Error::unsupported(format!("step `{}` failed: {message}", self.step_id))
                    }
                    _ => Error::graph(format!("step `{}` failed: {message}", self.step_id)),
                })
            }
        }
    }
}

/// Resolves a predecessor output value to declared output names.
///
/// An object member matching a declared name wins (exact key, then a normalized key
/// match); a predecessor with a single declared output falls back to the whole value.
fn resolve_outputs(output: &Value, declared: &[String]) -> Vec<(String, Value)> {
    let mut resolved = Vec::new();
    for name in declared {
        let key = normalize_name_key(name);
        if let Value::Object(map) = output {
            if let Some(value) = map.get(name) {
                resolved.push((key, value.clone()));
                continue;
            }
            let matched = map
                .iter()
                .find(|(candidate, _)| normalize_name_key(candidate) == key)
                .map(|(_, value)| value.clone());
            if let Some(value) = matched {
                resolved.push((key, value));
                continue;
            }
        }
        if declared.len() == 1 {
            resolved.push((key, output.clone()));
        }
    }
    resolved
}

/// Deduplicates and sorts the plan's edges by `(source, target)`.
fn plan_edges(plan: &OrchestrationPlan) -> BTreeSet<(String, String)> {
    plan.edges
        .iter()
        .map(|edge| (edge.source.clone(), edge.target.clone()))
        .collect()
}

/// Maps each step id to its predecessors, sorted by predecessor step id.
fn predecessors<'a>(
    plan: &'a OrchestrationPlan,
    edges: &BTreeSet<(String, String)>,
) -> BTreeMap<String, Vec<&'a PlanStep>> {
    let by_id: BTreeMap<&str, &PlanStep> = plan
        .steps
        .iter()
        .map(|step| (step.step_id.as_str(), step))
        .collect();
    let mut map: BTreeMap<String, Vec<&PlanStep>> = BTreeMap::new();
    for (source, target) in edges {
        if let Some(step) = by_id.get(source.as_str()) {
            map.entry(target.clone()).or_default().push(step);
        }
    }
    map
}

/// Which side of the edge set to derive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Endpoint {
    /// Steps with no incoming edge.
    Entry,
    /// Steps with no outgoing edge.
    Exit,
}

/// Derives the entry or exit step ids, sorted.
fn endpoints(
    plan: &OrchestrationPlan,
    edges: &BTreeSet<(String, String)>,
    side: Endpoint,
) -> Vec<String> {
    let connected: BTreeSet<&str> = match side {
        Endpoint::Entry => edges.iter().map(|(_, target)| target.as_str()).collect(),
        Endpoint::Exit => edges.iter().map(|(source, _)| source.as_str()).collect(),
    };
    let mut ids: Vec<String> = plan
        .steps
        .iter()
        .filter(|step| !connected.contains(step.step_id.as_str()))
        .map(|step| step.step_id.clone())
        .collect();
    ids.sort();
    ids
}
