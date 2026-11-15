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
//! `{"success": bool, "output": Value, "error": Option<String>, "error_kind":
//! Option<String>, "deferred": bool, "duration_ms": u64}`. The duration comes from a
//! monotonic [`std::time::Instant`]. A failed runner call still writes its record before
//! the failure propagates.
//!
//! `error_kind` is the runner failure's [`ErrorKind`] as its stable `snake_case` name
//! ([`ErrorKind::as_str`]) and stays `null` on success, so a caller tells a genuine
//! runtime failure (`"tool"`, `"timeout"`, `"graph"`, …) from a typed capability gap
//! (`"unsupported"`) without parsing the message. `deferred` is `true` exactly when a
//! failed step's `error_kind` is `"unsupported"`: the step names a capability the port
//! does not implement (see the crate's deferred scope), which is a typed gap rather than
//! a runtime error. Both fields deserialize with defaults, so a payload written before
//! them (or by a hand-built record) reads back as `error_kind: None, deferred: false`.
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
//! With `fail_fast = false` every step runs and the run returns `Ok(PlanExecution)`; a
//! failing step records `success: false` plus its message and its `error_kind`, so the
//! returned value still names every degradation. Callers use
//! [`PlanExecution::failed_steps`] for all failures and
//! [`PlanExecution::deferred_steps`] / [`PlanExecution::has_deferrals`] to isolate the
//! typed capability gaps; a deferred step is a failure too, never a silent success.
//!
//! # Strict deferral mode
//!
//! [`ExecutorConfig::fail_on_deferral`] (default `false`) promotes the first deferred
//! step to a run-level [`ErrorKind::Unsupported`]: the node returns
//! ``step `<id>` deferred: <runner message>``, so a caller that opts in never receives a
//! partial [`PlanExecution`] in which some capabilities silently did not run. It applies
//! whether or not [`ExecutorConfig::fail_fast`] is set, because it is about the typed gap
//! rather than generic failure handling; other runner failures keep the `fail_fast`
//! behaviour.
//!
//! # Recorded degradations
//!
//! No degraded invocation is absorbed silently: a step whose required inputs remain
//! unfilled after the overlay, a non-object caller seed, a declared predecessor with no
//! state record, and a recorded-but-failed predecessor each emit a `tracing::warn!`
//! naming the step and the inputs. A step that never produced a state record is
//! reported as `success: false` with `error_kind: "graph"` (an engine-level
//! inconsistency, not a runner failure) and logs its own warning.
//!
//! # Seeding caller inputs
//!
//! [`PlanExecutor::execute`] has no caller-input parameter, so it runs the graph with an
//! empty state: the overlay then starts from each predecessor's output only.
//! [`PlanExecutor::execute_with_inputs`] is the same run with a caller-supplied seed: a
//! non-`null` value is wrapped as the top-level `{"inputs": {...}}` object the overlay
//! reads, while [`Value::Null`] keeps the empty state. A caller that needs full control
//! can also invoke the compiled graph directly with that shape (see
//! [`PlanExecutor::compile`]).

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
use crate::model::{normalize_name_key, CapabilityDescriptor, CapabilityIo};
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
/// The state-key member holding a step record's failure kind (`ErrorKind::as_str`).
const RECORD_ERROR_KIND: &str = "error_kind";
/// The state-key member holding a step record's deferred flag.
const RECORD_DEFERRED: &str = "deferred";
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
    /// Whether the first deferred step aborts the run with a typed
    /// [`ErrorKind::Unsupported`].
    ///
    /// Defaults to `false`, which records deferrals in the returned
    /// [`PlanExecution`] and keeps running (subject to [`Self::fail_fast`] for other
    /// failures). Set it to `true` to never receive a partial plan silently: the run
    /// then fails with ``step `<id>` deferred: <runner message>``.
    pub fail_on_deferral: bool,
}

impl Default for ExecutorConfig {
    /// Returns `max_steps = 32`, no per-step timeout, `fail_fast = true`, and
    /// `fail_on_deferral = false`.
    fn default() -> Self {
        Self {
            max_steps: DEFAULT_MAX_STEPS,
            node_timeout_ms: None,
            fail_fast: true,
            fail_on_deferral: false,
        }
    }
}

/// What one plan step did.
///
/// `step_id` is the qualified capability id, and `duration_ms` measures the runner call
/// with a monotonic clock. `error_kind` and `deferred` classify a failure so a caller
/// can tell a typed capability gap from a runtime error without parsing `error`.
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
    /// The failure's [`ErrorKind`] as its stable `snake_case` name
    /// ([`ErrorKind::as_str`]); `None` when the step succeeded.
    ///
    /// A runner failure records its own kind (`"tool"`, `"timeout"`, …), a deferred
    /// capability records `"unsupported"`, and a step the graph never recorded records
    /// `"graph"`.
    pub error_kind: Option<String>,
    /// Whether the step failed because its capability is a typed capability gap.
    ///
    /// `true` exactly when the step failed with [`ErrorKind::Unsupported`]; such a step
    /// is still a failure for [`PlanExecution::failed_steps`] but is reported separately
    /// by [`PlanExecution::deferred_steps`].
    pub deferred: bool,
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
    ///
    /// Deferred steps are failures too, so they appear here; use
    /// [`Self::deferred_steps`] to isolate the typed capability gaps.
    #[must_use]
    pub fn failed_steps(&self) -> Vec<&StepOutcome> {
        self.steps.values().filter(|step| !step.success).collect()
    }

    /// Returns the deferred steps, ordered by step id.
    ///
    /// A deferred step names a capability the port does not implement (a typed
    /// capability gap, [`ErrorKind::Unsupported`]), not a runtime error. The list is
    /// empty when every step either succeeded or failed for a runtime reason.
    #[must_use]
    pub fn deferred_steps(&self) -> Vec<&StepOutcome> {
        self.steps.values().filter(|step| step.deferred).collect()
    }

    /// Reports whether any step failed with a typed capability gap.
    #[must_use]
    pub fn has_deferrals(&self) -> bool {
        self.steps.values().any(|step| step.deferred)
    }

    /// Reports whether every recorded step succeeded (true for a plan with no steps).
    ///
    /// A deferred step does not count as a success.
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
                inputs: step.inputs.clone(),
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
                fail_on_deferral: self.config.fail_on_deferral,
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
    /// is seeded with an empty state, exactly like
    /// [`Self::execute_with_inputs`] with [`Value::Null`] inputs; a caller that must
    /// supply request inputs uses that method instead.
    ///
    /// A plan with no steps has nothing to run and returns an empty [`PlanExecution`].
    ///
    /// # Errors
    ///
    /// Returns the [`PlanExecutor::compile`] validation errors, and with
    /// [`ExecutorConfig::fail_fast`] the first runner failure as an [`ErrorKind::Graph`]
    /// error (see the module docs for the preserved kinds). With
    /// [`ExecutorConfig::fail_on_deferral`] the first deferred step is returned as an
    /// [`ErrorKind::Unsupported`] error naming the step and the runner's reason. Engine-level
    /// failures such as a node timeout or a cancellation are returned unchanged.
    pub async fn execute(
        &self,
        plan: &OrchestrationPlan,
        capabilities: &[CapabilityDescriptor],
        session: Option<Arc<dyn Session>>,
    ) -> Result<PlanExecution> {
        self.execute_with_inputs(plan, capabilities, Value::Null, session)
            .await
    }

    /// Executes `plan` with caller-supplied `inputs` seeded into the graph state.
    ///
    /// A non-`null` `inputs` value is wrapped as the top-level `{"inputs": <inputs>}`
    /// object the node overlay reads, so a root step's declared inputs are filled by the
    /// caller and the overlay precedence of the module docs applies unchanged.
    /// [`Value::Null`] seeds an empty state, which is exactly what [`Self::execute`]
    /// does. Everything else — step records, failure handling, the default session —
    /// matches [`Self::execute`].
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Self::execute`].
    pub async fn execute_with_inputs(
        &self,
        plan: &OrchestrationPlan,
        capabilities: &[CapabilityDescriptor],
        inputs: Value,
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
            .run(seed_state(inputs), session, ExecutionOptions::default())
            .await
            .map_err(|error| trim_engine_context(plan, error))?;

        let mut steps = BTreeMap::new();
        let mut outputs = BTreeMap::new();
        let mut records = plan.steps.clone();
        records.sort_by(|left, right| left.step_id.cmp(&right.step_id));
        for step in &records {
            let outcome = if let Some(record) = run.state.get(&step.step_id) {
                let mut outcome: StepOutcome =
                    serde_json::from_value(record.clone()).map_err(|error| {
                        validation("step_outcome", format!("step `{}`: {error}", step.step_id))
                    })?;
                outcome.step_id.clone_from(&step.step_id);
                outcome.capability_type.clone_from(&step.capability_type);
                // Keep the two failure-class fields consistent even for a hand-built
                // or older record: `deferred` mirrors `error_kind` on a failure.
                outcome.deferred = !outcome.success
                    && outcome.error_kind.as_deref() == Some(ErrorKind::Unsupported.as_str());
                outcome
            } else {
                tracing::warn!(
                    step = %step.step_id,
                    "plan step produced no state record; recording it as a failed step"
                );
                StepOutcome {
                    step_id: step.step_id.clone(),
                    capability_type: step.capability_type.clone(),
                    success: false,
                    output: Value::Null,
                    error: Some(MISSING_RECORD_ERROR.to_string()),
                    error_kind: Some(ErrorKind::Graph.as_str().to_string()),
                    deferred: false,
                    duration_ms: 0,
                }
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

/// Wraps caller-supplied inputs into the top-level state the node overlay reads.
///
/// A [`Value::Null`] seed produces the empty state `execute` always used; any other value
/// becomes `{"inputs": <value>}`, which the overlay treats as the starting payload (a
/// non-object seed is ignored there, so wrapping a scalar is harmless but pointless).
fn seed_state(inputs: Value) -> Value {
    match inputs {
        Value::Null => Value::Object(Map::new()),
        inputs => Value::Object(Map::from_iter([(SEED_INPUTS_KEY.to_string(), inputs)])),
    }
}

/// Strips the graph engine's wrapper off a step failure.
///
/// The engine reports a node failure as
/// ``node `<id>` failed at superstep <n>: <node error>``. The executor owns the node
/// error, so it trims that engine context back off using the failing step's own
/// ``step `<id>` failed: `` or ``step `<id>` deferred: `` marker: the caller then sees
/// exactly the documented message and the error keeps its kind. A failure without a
/// marker (a timeout, a cancellation, a channel error) is returned unchanged.
fn trim_engine_context(plan: &OrchestrationPlan, error: Error) -> Error {
    let message = error.message().into_owned();
    for step in &plan.steps {
        for marker in [
            format!("step `{}` failed: ", step.step_id),
            format!("step `{}` deferred: ", step.step_id),
        ] {
            if let Some(index) = message.find(&marker) {
                let trimmed = message[index..].to_string();
                return match error.kind() {
                    ErrorKind::Unsupported => Error::unsupported(trimmed),
                    _ => Error::graph(trimmed),
                };
            }
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
    /// The step's declared inputs, in plan order.
    inputs: Vec<CapabilityIo>,
    /// The step's predecessors, sorted by step id.
    predecessors: Vec<Predecessor>,
    /// Whether a non-deferral runner error aborts the run.
    fail_fast: bool,
    /// Whether a deferred (unsupported) runner error aborts the run.
    fail_on_deferral: bool,
}

impl StepNode {
    /// Builds the invocation payload: the caller seed overlaid with predecessor outputs.
    ///
    /// Every degradation is logged: a non-object seed, a declared predecessor that never
    /// recorded an outcome, a failed predecessor whose output is `null`, and required
    /// declared inputs left unfilled by the overlay.
    fn collect_inputs(&self, state: &GraphState) -> Value {
        let mut inputs: Map<String, Value> = match state.get(SEED_INPUTS_KEY) {
            Some(Value::Object(map)) => map.clone(),
            Some(other) => {
                tracing::warn!(
                    step = %self.step_id,
                    seed = %other,
                    "ignoring a non-object step-input seed"
                );
                Map::new()
            }
            None => Map::new(),
        };

        let input_keys: Vec<(&str, String)> = self
            .inputs
            .iter()
            .map(|io| (io.name.as_str(), normalize_name_key(&io.name)))
            .collect();
        let mut filled: BTreeSet<String> = input_keys
            .iter()
            .filter(|(name, _)| inputs.contains_key(*name))
            .map(|(_, key)| key.clone())
            .collect();

        for predecessor in &self.predecessors {
            let Some(record) = state.get(&predecessor.step_id) else {
                tracing::warn!(
                    step = %self.step_id,
                    predecessor = %predecessor.step_id,
                    "declared predecessor recorded no outcome; nothing to overlay"
                );
                continue;
            };
            let Some(output) = record.get(RECORD_OUTPUT) else {
                tracing::warn!(
                    step = %self.step_id,
                    predecessor = %predecessor.step_id,
                    "declared predecessor recorded no output; nothing to overlay"
                );
                continue;
            };
            if output.is_null() {
                if record
                    .get(RECORD_SUCCESS)
                    .and_then(Value::as_bool)
                    .is_some_and(|success| !success)
                {
                    tracing::warn!(
                        step = %self.step_id,
                        predecessor = %predecessor.step_id,
                        "predecessor failed; its output contributes nothing to the overlay"
                    );
                }
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

        let missing: Vec<&str> = self
            .inputs
            .iter()
            .filter(|io| io.required != Some(false) && !inputs.contains_key(io.name.as_str()))
            .map(|io| io.name.as_str())
            .collect();
        if !missing.is_empty() {
            tracing::warn!(
                step = %self.step_id,
                missing = ?missing,
                "required step inputs are unfilled; invoking the runner with a partial payload"
            );
        }

        Value::Object(inputs)
    }

    /// Builds the state record this step writes.
    ///
    /// A failure records its message plus its [`ErrorKind`] name, and flags a deferred
    /// (`unsupported`) failure so [`PlanExecution::deferred_steps`] can report it.
    fn record(success: bool, output: Value, error: Option<&Error>, duration_ms: u64) -> Value {
        let mut record = Map::new();
        record.insert(RECORD_SUCCESS.to_string(), Value::Bool(success));
        record.insert(RECORD_OUTPUT.to_string(), output);
        record.insert(
            RECORD_ERROR.to_string(),
            error.map_or(Value::Null, |error| {
                Value::String(error.message().into_owned())
            }),
        );
        record.insert(
            RECORD_ERROR_KIND.to_string(),
            error.map_or(Value::Null, |error| {
                Value::String(error.kind().as_str().to_string())
            }),
        );
        record.insert(
            RECORD_DEFERRED.to_string(),
            Value::Bool(error.is_some_and(|error| error.kind() == ErrorKind::Unsupported)),
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
                let kind = error.kind();
                let message = error.message().into_owned();
                state.set(
                    self.step_id.clone(),
                    Self::record(false, Value::Null, Some(&error), duration_ms),
                );
                if kind == ErrorKind::Unsupported && self.fail_on_deferral {
                    return Err(Error::unsupported(format!(
                        "step `{}` deferred: {message}",
                        self.step_id
                    )));
                }
                if !self.fail_fast {
                    return Ok(NodeOutcome::Continue);
                }
                Err(match kind {
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

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]
mod tests {
    use super::*;
    use crate::orchestration::plan::{PlanEdge, PlanStatus};
    use async_trait::async_trait;
    use serde_json::json;

    /// Builds a capability card with no inputs or outputs.
    fn capability(id: &str) -> CapabilityDescriptor {
        CapabilityDescriptor::new("tool", id, format!("cap tool:{id}"))
    }

    /// Builds a plan step whose capability type is taken from its qualified id.
    fn step(step_id: &str) -> PlanStep {
        PlanStep {
            step_id: step_id.to_string(),
            capability_type: "tool".to_string(),
            name: format!("cap {step_id}"),
            ..PlanStep::default()
        }
    }

    /// Builds a ready plan over `steps`, linked by `edges`.
    fn plan(steps: Vec<PlanStep>, edges: Vec<(&str, &str)>) -> OrchestrationPlan {
        OrchestrationPlan::new("plan-test", "q", "en")
            .with_status(PlanStatus::Ready)
            .with_steps(steps)
            .with_edges(
                edges
                    .into_iter()
                    .map(|(source, target)| PlanEdge {
                        source: source.to_string(),
                        target: target.to_string(),
                    })
                    .collect(),
            )
    }

    /// What a scripted fake runner does for one capability.
    enum Behavior {
        /// Returns this value.
        Output(Value),
        /// Returns an [`Error::tool`] with this message.
        Fail(&'static str),
        /// Returns the crate's typed deferred-scope error.
        Deferred,
    }

    /// Replays one scripted behavior per qualified capability id.
    struct FakeRunner {
        behaviors: BTreeMap<String, Behavior>,
    }

    #[async_trait]
    impl CapabilityRunner for FakeRunner {
        async fn invoke(
            &self,
            capability: &CapabilityDescriptor,
            _inputs: Value,
            _context: &RunnerContext,
        ) -> Result<Value> {
            match self.behaviors.get(&capability.qualified_id()) {
                Some(Behavior::Output(value)) => Ok(value.clone()),
                Some(Behavior::Fail(message)) => Err(Error::tool(*message)),
                Some(Behavior::Deferred) => Err(crate::error::unsupported("test capability")),
                None => Ok(Value::Null),
            }
        }
    }

    #[test]
    fn config_defaults_never_fail_on_a_deferral() {
        let config = ExecutorConfig::default();
        assert!(!config.fail_on_deferral);
        assert!(config.fail_fast);
        // A payload written before the field existed reads back with the default.
        let legacy: ExecutorConfig = serde_json::from_value(
            json!({"max_steps": 2, "node_timeout_ms": null, "fail_fast": false}),
        )
        .expect("legacy config");
        assert_eq!(legacy.max_steps, 2);
        assert_eq!(legacy.node_timeout_ms, None);
        assert!(!legacy.fail_fast);
        assert!(!legacy.fail_on_deferral);
    }

    #[tokio::test]
    async fn all_deferred_steps_are_classified_not_silently_failed() {
        let capabilities = vec![capability("a"), capability("b")];
        let plan = plan(
            vec![step("tool:a"), step("tool:b")],
            vec![("tool:a", "tool:b")],
        );
        let executor = PlanExecutor::with_config(
            ExecutorConfig {
                fail_fast: false,
                ..ExecutorConfig::default()
            },
            Arc::new(UnsupportedRunner::new()),
        );

        let execution = executor
            .execute(&plan, &capabilities, None)
            .await
            .expect("the lenient run returns a PlanExecution");

        assert!(!execution.succeeded());
        let failed: Vec<&str> = execution
            .failed_steps()
            .iter()
            .map(|step| step.step_id.as_str())
            .collect();
        assert_eq!(failed, ["tool:a", "tool:b"]);
        let deferred: Vec<&str> = execution
            .deferred_steps()
            .iter()
            .map(|step| step.step_id.as_str())
            .collect();
        assert_eq!(deferred, ["tool:a", "tool:b"]);
        assert!(execution.has_deferrals());
        for step_id in ["tool:a", "tool:b"] {
            let outcome = &execution.steps[step_id];
            assert!(!outcome.success);
            assert_eq!(outcome.error_kind.as_deref(), Some("unsupported"));
            assert!(outcome.deferred);
            assert_eq!(execution.state[step_id]["error_kind"], json!("unsupported"));
            assert_eq!(execution.state[step_id]["deferred"], json!(true));
        }
    }

    #[tokio::test]
    async fn strict_mode_turns_the_first_deferral_into_a_typed_error() {
        let capabilities = vec![capability("a"), capability("b")];
        let plan = plan(
            vec![step("tool:a"), step("tool:b")],
            vec![("tool:a", "tool:b")],
        );

        for fail_fast in [false, true] {
            let executor = PlanExecutor::with_config(
                ExecutorConfig {
                    fail_fast,
                    fail_on_deferral: true,
                    ..ExecutorConfig::default()
                },
                Arc::new(UnsupportedRunner::new()),
            );
            let error = executor
                .execute(&plan, &capabilities, None)
                .await
                .expect_err("strict mode must abort on the first deferral");
            assert_eq!(
                error.kind(),
                ErrorKind::Unsupported,
                "fail_fast = {fail_fast}"
            );
            assert!(
                error
                    .message()
                    .starts_with("step `tool:a` deferred: capability execution"),
                "unexpected message: {error}"
            );
        }
    }

    #[tokio::test]
    async fn genuine_tool_failures_keep_the_graph_mapping() {
        let capabilities = vec![capability("b")];
        let plan = plan(vec![step("tool:b")], Vec::new());
        let runner = FakeRunner {
            behaviors: BTreeMap::from([("tool:b".to_string(), Behavior::Fail("boom"))]),
        };

        let strict = PlanExecutor::with_config(
            ExecutorConfig {
                fail_on_deferral: true,
                ..ExecutorConfig::default()
            },
            Arc::new(runner),
        );
        let error = strict
            .execute(&plan, &capabilities, None)
            .await
            .expect_err("a tool failure aborts under fail_fast");
        assert_eq!(error.kind(), ErrorKind::Graph);
        assert_eq!(error.message(), "step `tool:b` failed: boom");

        let lenient = PlanExecutor::with_config(
            ExecutorConfig {
                fail_fast: false,
                ..ExecutorConfig::default()
            },
            Arc::new(FakeRunner {
                behaviors: BTreeMap::from([("tool:b".to_string(), Behavior::Fail("boom"))]),
            }),
        );
        let execution = lenient
            .execute(&plan, &capabilities, None)
            .await
            .expect("lenient run");
        let outcome = &execution.steps["tool:b"];
        assert_eq!(outcome.error_kind.as_deref(), Some("tool"));
        assert!(!outcome.deferred);
        assert!(!execution.has_deferrals());
        assert!(execution.deferred_steps().is_empty());
    }

    #[tokio::test]
    async fn mixed_plan_keeps_success_and_deferral_distinct() {
        let capabilities = vec![capability("ok"), capability("deferred")];
        let plan = plan(vec![step("tool:ok"), step("tool:deferred")], Vec::new());
        let runner = FakeRunner {
            behaviors: BTreeMap::from([
                (
                    "tool:ok".to_string(),
                    Behavior::Output(json!({"done": true})),
                ),
                ("tool:deferred".to_string(), Behavior::Deferred),
            ]),
        };
        let executor = PlanExecutor::with_config(
            ExecutorConfig {
                fail_fast: false,
                ..ExecutorConfig::default()
            },
            Arc::new(runner),
        );

        let execution = executor
            .execute(&plan, &capabilities, None)
            .await
            .expect("run");
        let ok = &execution.steps["tool:ok"];
        assert!(ok.success);
        assert_eq!(ok.error, None);
        assert_eq!(ok.error_kind, None);
        assert!(!ok.deferred);
        assert_eq!(execution.outputs["tool:ok"], json!({"done": true}));

        let deferred = &execution.steps["tool:deferred"];
        assert!(!deferred.success);
        assert_eq!(deferred.error_kind.as_deref(), Some("unsupported"));
        assert!(deferred.deferred);
        assert!(!execution.outputs.contains_key("tool:deferred"));
        assert_eq!(
            execution
                .failed_steps()
                .iter()
                .map(|step| step.step_id.as_str())
                .collect::<Vec<_>>(),
            ["tool:deferred"]
        );
    }

    #[test]
    fn legacy_payloads_default_the_new_fields() {
        let legacy_outcome: StepOutcome = serde_json::from_value(json!({
            "step_id": "tool:a",
            "capability_type": "tool",
            "success": false,
            "output": null,
            "error": "boom",
            "duration_ms": 3,
        }))
        .expect("legacy outcome");
        assert_eq!(legacy_outcome.error.as_deref(), Some("boom"));
        assert_eq!(legacy_outcome.error_kind, None);
        assert!(!legacy_outcome.deferred);

        let execution: PlanExecution = serde_json::from_value(json!({
            "plan_id": "plan-test",
            "graph_id": "plan-test",
            "steps": {"tool:a": legacy_outcome},
            "outputs": {},
            "state": {},
        }))
        .expect("legacy execution");
        assert_eq!(execution.failed_steps().len(), 1);
        assert!(execution.deferred_steps().is_empty());
        assert!(!execution.has_deferrals());

        let deferred: StepOutcome = serde_json::from_value(json!({
            "step_id": "tool:d",
            "capability_type": "tool",
            "success": false,
            "output": null,
            "error": "deferred",
            "error_kind": "unsupported",
            "deferred": true,
            "duration_ms": 0,
        }))
        .expect("deferred outcome");
        let encoded = serde_json::to_value(&deferred).expect("serialize");
        assert_eq!(encoded["error_kind"], json!("unsupported"));
        assert_eq!(encoded["deferred"], json!(true));
        assert_eq!(
            serde_json::from_value::<StepOutcome>(encoded).expect("round trip"),
            deferred
        );
    }
}
