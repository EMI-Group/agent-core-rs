//! Behaviour tests for [`PlanExecutor`]: plan compilation onto `openjiuwen-graph`,
//! per-step state records, the input overlay, both failure modes, and the explicit
//! deferral accounting (`error_kind`/`deferred`, `deferred_steps`, `has_deferrals` and
//! `ExecutorConfig::fail_on_deferral`).
//!
//! Everything here is offline and deterministic: a fake [`CapabilityRunner`] records the
//! payloads it receives, no network or filesystem is touched, and the only clock use is
//! a monotonic duration plus one deliberately short timeout.
#![allow(clippy::unwrap_used, clippy::expect_used)]
#![allow(clippy::float_cmp)]
#![allow(clippy::assert_is_empty)]

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use openjiuwen_session::{Session, WorkflowSession};
use openjiuwen_symphony::error::{is_unsupported, unsupported, Error, ErrorKind, Result};
use openjiuwen_symphony::model::{CapabilityDescriptor, CapabilityIo};
use openjiuwen_symphony::orchestration::executor::{
    CapabilityRunner, ExecutorConfig, PlanExecution, PlanExecutor, RunnerContext, StepOutcome,
    UnsupportedRunner,
};
use openjiuwen_symphony::orchestration::{OrchestrationPlan, PlanEdge, PlanStatus, PlanStep};
use serde_json::{json, Value};

/// One string-typed IO, which is all the overlay and the tests need.
fn io(name: &str) -> CapabilityIo {
    CapabilityIo::new(name, "string")
}

/// Builds a capability card with the given input and output names.
fn capability(
    capability_type: &str,
    id: &str,
    inputs: &[&str],
    outputs: &[&str],
) -> CapabilityDescriptor {
    CapabilityDescriptor::new(capability_type, id, format!("cap {capability_type}:{id}"))
        .with_inputs(inputs.iter().map(|name| io(name)).collect())
        .with_outputs(outputs.iter().map(|name| io(name)).collect())
}

/// Builds a plan step whose capability type is taken from its qualified id.
fn step(step_id: &str, inputs: &[&str], outputs: &[&str]) -> PlanStep {
    let capability_type = step_id.split_once(':').map_or(step_id, |(kind, _)| kind);
    PlanStep {
        step_id: step_id.to_string(),
        capability_type: capability_type.to_string(),
        name: format!("cap {step_id}"),
        inputs: inputs.iter().map(|name| io(name)).collect(),
        outputs: outputs.iter().map(|name| io(name)).collect(),
        ..PlanStep::default()
    }
}

/// Builds a plan edge.
fn edge(source: &str, target: &str) -> PlanEdge {
    PlanEdge {
        source: source.to_string(),
        target: target.to_string(),
    }
}

/// Builds a ready plan with the given steps and edges.
fn plan(steps: Vec<PlanStep>, edges: Vec<PlanEdge>) -> OrchestrationPlan {
    OrchestrationPlan::new("plan-1", "build a report", "en")
        .with_status(PlanStatus::Ready)
        .with_steps(steps)
        .with_edges(edges)
}

/// What the fake runner does for one capability.
#[derive(Debug, Clone, PartialEq)]
enum Behavior {
    /// Returns this value.
    Output(Value),
    /// Returns an [`Error::tool`] with this message.
    Fail(String),
    /// Returns the crate's typed deferred-scope error (a capability gap).
    Deferred,
    /// Sleeps, then returns this value (used by the timeout test).
    Sleep(Duration, Value),
}

/// The feature name the fake runner reports when it defers a capability.
const DEFERRED_FEATURE: &str = "deferred test capability";

/// The crate's typed deferred-scope error, exactly as [`Behavior::Deferred`] returns it.
///
/// The strict-mode message embeds this text, so the deferral tests derive their expected
/// message from the same constructor instead of hard-coding it.
fn deferred_error() -> Error {
    unsupported(DEFERRED_FEATURE)
}

/// One recorded invocation.
#[derive(Debug, Clone, PartialEq)]
struct Call {
    /// The qualified id of the capability the executor resolved for the step.
    capability_id: String,
    /// The overlay payload the step was invoked with.
    inputs: Value,
    /// The session id the executor passed along.
    session_id: Option<String>,
}

/// A [`CapabilityRunner`] that records every call and replays scripted behaviors.
#[derive(Clone, Default)]
struct RecordingRunner {
    behaviors: BTreeMap<String, Behavior>,
    calls: Arc<Mutex<Vec<Call>>>,
}

impl RecordingRunner {
    /// Builds a runner from `(capability qualified id, behavior)` pairs.
    fn new(behaviors: impl IntoIterator<Item = (&'static str, Behavior)>) -> Self {
        Self {
            behaviors: behaviors
                .into_iter()
                .map(|(id, behavior)| (id.to_string(), behavior))
                .collect(),
            calls: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Returns every recorded call in invocation order.
    fn calls(&self) -> Vec<Call> {
        self.calls.lock().expect("calls mutex").clone()
    }

    /// Returns the payload recorded for `step_id`.
    fn inputs_for(&self, step_id: &str) -> Value {
        self.calls()
            .into_iter()
            .find(|call| call.capability_id == step_id)
            .unwrap_or_else(|| panic!("no call recorded for `{step_id}`"))
            .inputs
    }
}

#[async_trait]
impl CapabilityRunner for RecordingRunner {
    async fn invoke(
        &self,
        capability: &CapabilityDescriptor,
        inputs: Value,
        context: &RunnerContext,
    ) -> Result<Value> {
        let capability_id = capability.qualified_id();
        self.calls.lock().expect("calls mutex").push(Call {
            capability_id: capability_id.clone(),
            inputs: inputs.clone(),
            session_id: context.session_id.clone(),
        });
        match self.behaviors.get(&capability_id) {
            Some(Behavior::Output(value)) => Ok(value.clone()),
            Some(Behavior::Fail(message)) => Err(Error::tool(message.clone())),
            Some(Behavior::Deferred) => Err(deferred_error()),
            Some(Behavior::Sleep(duration, value)) => {
                tokio::time::sleep(*duration).await;
                Ok(value.clone())
            }
            // Echoing the payload keeps unscripted capabilities observable.
            None => Ok(inputs),
        }
    }
}

/// The three capabilities of the standard chain: at most one input each.
fn chain_capabilities() -> Vec<CapabilityDescriptor> {
    vec![
        capability("tool", "a", &[], &["text"]),
        capability("tool", "b", &["text"], &["result"]),
        capability("tool", "c", &["prompt"], &[]),
    ]
}

/// `tool:a -> tool:b -> tool:c`: one output-slot name match, one single-output fill.
fn chain_plan() -> OrchestrationPlan {
    plan(
        vec![
            step("tool:a", &[], &["text"]),
            step("tool:b", &["text"], &["result"]),
            step("tool:c", &["prompt"], &[]),
        ],
        vec![edge("tool:a", "tool:b"), edge("tool:b", "tool:c")],
    )
}

/// The scripted behaviors of the standard chain.
fn chain_runner() -> RecordingRunner {
    RecordingRunner::new([
        ("tool:a", Behavior::Output(json!({"text": "alpha"}))),
        ("tool:b", Behavior::Output(json!("alpha-beta"))),
        ("tool:c", Behavior::Output(json!({"done": true}))),
    ])
}

/// The standard chain with a genuine runtime failure at `tool:b`.
fn failing_chain_runner() -> RecordingRunner {
    RecordingRunner::new([
        ("tool:a", Behavior::Output(json!({"text": "alpha"}))),
        ("tool:b", Behavior::Fail("boom".to_string())),
        ("tool:c", Behavior::Output(json!({"done": true}))),
    ])
}

/// `tool:x -> tool:y`: a two-step chain whose capabilities both name a typed gap.
fn deferred_chain_plan() -> OrchestrationPlan {
    plan(
        vec![
            step("tool:x", &[], &["text"]),
            step("tool:y", &["text"], &[]),
        ],
        vec![edge("tool:x", "tool:y")],
    )
}

/// The two capabilities of [`deferred_chain_plan`].
fn deferred_chain_capabilities() -> Vec<CapabilityDescriptor> {
    vec![
        capability("tool", "x", &[], &["text"]),
        capability("tool", "y", &["text"], &[]),
    ]
}

/// A runner that defers both steps of [`deferred_chain_plan`].
fn deferred_chain_runner() -> RecordingRunner {
    RecordingRunner::new([
        ("tool:x", Behavior::Deferred),
        ("tool:y", Behavior::Deferred),
    ])
}

/// Collects the step ids of an outcome list for order-sensitive assertions.
fn step_ids<'a>(outcomes: &[&'a StepOutcome]) -> Vec<&'a str> {
    outcomes
        .iter()
        .map(|outcome| outcome.step_id.as_str())
        .collect()
}

#[tokio::test]
async fn three_step_chain_records_each_step_and_overlays_inputs() {
    let runner = Arc::new(chain_runner());
    let executor = PlanExecutor::new(Arc::clone(&runner) as Arc<dyn CapabilityRunner>);
    let execution = executor
        .execute(&chain_plan(), &chain_capabilities(), None)
        .await
        .unwrap();

    assert_eq!(execution.plan_id, "plan-1");
    assert_eq!(execution.graph_id, "plan-1");
    assert!(execution.succeeded(), "every step must succeed");
    assert!(execution.failed_steps().is_empty());

    let step_ids: Vec<&str> = execution.steps.keys().map(String::as_str).collect();
    assert_eq!(step_ids, ["tool:a", "tool:b", "tool:c"]);
    for (step_id, outcome) in &execution.steps {
        assert!(
            outcome.success,
            "step `{step_id}` must be recorded as successful"
        );
        assert_eq!(outcome.step_id, *step_id);
        assert_eq!(outcome.capability_type, "tool");
        assert_eq!(outcome.error, None);
    }

    // Each node writes exactly one top-level key with the documented record shape.
    for step_id in ["tool:a", "tool:b", "tool:c"] {
        let record = &execution.state[step_id];
        assert_eq!(record["success"], json!(true), "record of `{step_id}`");
        assert_eq!(record["error"], Value::Null, "record of `{step_id}`");
        assert!(
            record["duration_ms"].is_u64(),
            "record of `{step_id}` must carry a monotonic duration: {record}"
        );
        assert!(
            record.get("output").is_some(),
            "record of `{step_id}` must carry an output member"
        );
    }
    assert_eq!(
        execution.state["tool:a"]["output"],
        json!({"text": "alpha"})
    );
    assert_eq!(execution.state["tool:b"]["output"], json!("alpha-beta"));

    let outputs: Vec<(&String, &Value)> = execution.outputs.iter().collect();
    assert_eq!(outputs.len(), 3);
    assert_eq!(execution.outputs["tool:a"], json!({"text": "alpha"}));
    assert_eq!(execution.outputs["tool:b"], json!("alpha-beta"));
    assert_eq!(execution.outputs["tool:c"], json!({"done": true}));

    // Overlay: `a` has no predecessor, so it sees only the (empty) seed.
    assert_eq!(runner.inputs_for("tool:a"), json!({}));
    // `a` returns `{"text": "alpha"}` and declares output `text`, which matches `b`'s
    // single input `text`, so the matching member is used.
    assert_eq!(runner.inputs_for("tool:b"), json!({"text": "alpha"}));
    // `b` returns a bare string with one declared output `result`; the name does not
    // match `c`'s input `prompt`, so the single-output fill rule supplies it.
    assert_eq!(runner.inputs_for("tool:c"), json!({"prompt": "alpha-beta"}));

    let calls = runner.calls();
    assert_eq!(calls.len(), 3);
    assert!(
        calls.iter().all(|call| call.session_id.is_some()),
        "a default WorkflowSession must be used when none is passed"
    );
}

#[tokio::test]
async fn compile_maps_steps_edges_entries_and_exits() {
    let diamond = plan(
        vec![
            step("tool:a", &[], &["text"]),
            step("tool:b", &["text"], &["left"]),
            step("tool:c", &["text"], &["right"]),
            step("tool:d", &["left", "right"], &[]),
        ],
        vec![
            edge("tool:a", "tool:b"),
            edge("tool:a", "tool:c"),
            edge("tool:b", "tool:d"),
            edge("tool:c", "tool:d"),
            // A duplicate edge must not create a second channel group.
            edge("tool:a", "tool:b"),
        ],
    );
    let capabilities = vec![
        capability("tool", "a", &[], &["text"]),
        capability("tool", "b", &["text"], &["left"]),
        capability("tool", "c", &["text"], &["right"]),
        capability("tool", "d", &["left", "right"], &[]),
    ];
    let executor = PlanExecutor::new(Arc::new(RecordingRunner::default()));
    let graph = executor.compile(&diamond, &capabilities).unwrap();

    assert_eq!(graph.id(), "plan-1");
    assert_eq!(
        graph.node_ids().collect::<Vec<_>>(),
        ["tool:a", "tool:b", "tool:c", "tool:d"]
    );
    assert_eq!(graph.entry_nodes(), ["tool:a"]);
    assert_eq!(graph.exit_nodes(), ["tool:d"]);
    assert_eq!(graph.spec().edges.len(), 4, "duplicate edges collapse");
    let targets: Vec<&str> = graph
        .spec()
        .edges
        .iter()
        .map(|edge| edge.target.as_str())
        .collect();
    assert_eq!(targets, ["tool:b", "tool:c", "tool:d", "tool:d"]);
}

#[tokio::test]
async fn execute_seeds_nothing_but_the_compiled_graph_accepts_caller_inputs() {
    let capabilities = vec![
        capability("tool", "a", &["text"], &["text"]),
        capability("tool", "b", &["text"], &[]),
    ];
    let plan = plan(
        vec![
            step("tool:a", &["text"], &["text"]),
            step("tool:b", &["text"], &[]),
        ],
        vec![edge("tool:a", "tool:b")],
    );
    let runner = Arc::new(RecordingRunner::new([
        ("tool:a", Behavior::Output(json!({"text": "from-a"}))),
        ("tool:b", Behavior::Output(json!({"done": true}))),
    ]));
    let executor = PlanExecutor::new(Arc::clone(&runner) as Arc<dyn CapabilityRunner>);

    // `execute` has no caller-input parameter: it seeds an empty graph state.
    executor.execute(&plan, &capabilities, None).await.unwrap();
    let calls = runner.calls();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].capability_id, "tool:a");
    assert_eq!(calls[0].inputs, json!({}), "no seed reaches the first step");
    assert!(
        calls[0].session_id.is_some(),
        "the executor must pass the default session id"
    );
    assert_eq!(calls[1].capability_id, "tool:b");
    assert_eq!(calls[1].inputs, json!({"text": "from-a"}));

    // A caller that needs request inputs invokes the compiled graph with `{"inputs": …}`.
    let graph = executor.compile(&plan, &capabilities).unwrap();
    let session: Arc<dyn Session> = Arc::new(WorkflowSession::new("seed-test"));
    let state = graph
        .invoke(json!({"inputs": {"text": "seed"}}), Arc::clone(&session))
        .await
        .unwrap();
    assert_eq!(state["tool:a"]["success"], json!(true));
    let calls = runner.calls();
    assert_eq!(calls.len(), 4);
    assert_eq!(calls[2].inputs, json!({"text": "seed"}));
    assert_eq!(calls[2].session_id.as_deref(), Some("seed-test"));
    assert_eq!(
        calls[3].inputs,
        json!({"text": "from-a"}),
        "the predecessor output overlays the seed"
    );

    // A non-object seed is ignored rather than rejected or panicking.
    let state = graph
        .invoke(json!({"inputs": "not an object"}), session)
        .await
        .unwrap();
    assert_eq!(state["tool:a"]["success"], json!(true));
    let calls = runner.calls();
    assert_eq!(calls.len(), 6);
    assert_eq!(calls[4].inputs, json!({}));
}

#[tokio::test]
async fn fail_fast_true_aborts_with_the_documented_graph_error() {
    let runner = Arc::new(RecordingRunner::new([
        ("tool:a", Behavior::Output(json!({"text": "alpha"}))),
        ("tool:b", Behavior::Fail("boom".to_string())),
        ("tool:c", Behavior::Output(json!({"done": true}))),
    ]));
    let executor = PlanExecutor::new(Arc::clone(&runner) as Arc<dyn CapabilityRunner>);
    let error = executor
        .execute(&chain_plan(), &chain_capabilities(), None)
        .await
        .unwrap_err();

    assert_eq!(error.kind(), ErrorKind::Graph);
    assert_eq!(error.message(), "step `tool:b` failed: boom");
    assert_eq!(
        runner.calls().len(),
        2,
        "`tool:c` must not run after the abort"
    );
}

#[tokio::test]
async fn fail_fast_false_records_the_failure_and_keeps_running() {
    let runner = Arc::new(RecordingRunner::new([
        ("tool:a", Behavior::Output(json!({"text": "alpha"}))),
        ("tool:b", Behavior::Fail("boom".to_string())),
        ("tool:c", Behavior::Output(json!({"done": true}))),
    ]));
    let executor = PlanExecutor::with_config(
        ExecutorConfig {
            fail_fast: false,
            ..ExecutorConfig::default()
        },
        Arc::clone(&runner) as Arc<dyn CapabilityRunner>,
    );
    let execution = executor
        .execute(&chain_plan(), &chain_capabilities(), None)
        .await
        .unwrap();

    assert!(!execution.succeeded());
    let failed: Vec<&str> = execution
        .failed_steps()
        .iter()
        .map(|step| step.step_id.as_str())
        .collect();
    assert_eq!(failed, ["tool:b"]);

    let failed_step = &execution.steps["tool:b"];
    assert!(!failed_step.success);
    assert_eq!(failed_step.error.as_deref(), Some("boom"));
    assert_eq!(failed_step.output, Value::Null);
    // A genuine runner failure carries its own kind and is never a deferral.
    assert_eq!(failed_step.error_kind.as_deref(), Some("tool"));
    assert!(!failed_step.deferred);
    assert_eq!(execution.state["tool:b"]["success"], json!(false));
    assert_eq!(execution.state["tool:b"]["error"], json!("boom"));
    assert_eq!(execution.state["tool:b"]["error_kind"], json!("tool"));
    assert_eq!(execution.state["tool:b"]["deferred"], json!(false));
    assert!(!execution.outputs.contains_key("tool:b"));

    // `failed_steps` reports the runtime failure and `deferred_steps` stays empty.
    assert_eq!(execution.failed_steps().len(), 1);
    assert!(!execution.has_deferrals());
    assert!(execution.deferred_steps().is_empty());

    // Downstream steps still run and are recorded, with the predecessor's `null` output
    // contributing nothing to the overlay.
    assert_eq!(runner.calls().len(), 3);
    assert!(execution.steps["tool:c"].success);
    assert_eq!(runner.inputs_for("tool:c"), json!({}));
}

#[tokio::test]
async fn max_steps_is_rejected_before_compilation() {
    let capabilities = vec![
        capability("tool", "a", &[], &["text"]),
        capability("tool", "b", &["text"], &[]),
    ];
    let two_steps = plan(
        vec![
            step("tool:a", &[], &["text"]),
            step("tool:b", &["text"], &[]),
        ],
        vec![edge("tool:a", "tool:b")],
    );
    let executor = PlanExecutor::with_config(
        ExecutorConfig {
            max_steps: 1,
            ..ExecutorConfig::default()
        },
        Arc::new(RecordingRunner::default()),
    );

    let compile_error = executor.compile(&two_steps, &capabilities).unwrap_err();
    assert_eq!(compile_error.kind(), ErrorKind::Validation);
    assert!(
        compile_error.message().starts_with("max_steps:"),
        "message `{}` must name the limit",
        compile_error.message()
    );

    let execute_error = executor
        .execute(&two_steps, &capabilities, None)
        .await
        .unwrap_err();
    assert_eq!(execute_error.kind(), ErrorKind::Validation);
    assert!(execute_error.message().starts_with("max_steps:"));
}

#[tokio::test]
async fn unknown_capability_step_is_rejected() {
    let executor = PlanExecutor::new(Arc::new(RecordingRunner::default()));
    let orphan = plan(vec![step("tool:ghost", &[], &[])], Vec::new());
    let error = executor
        .execute(&orphan, &chain_capabilities(), None)
        .await
        .unwrap_err();

    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().starts_with("step_id:"));
    assert!(error.message().contains("tool:ghost"));
}

#[tokio::test]
async fn duplicate_step_ids_are_rejected() {
    let executor = PlanExecutor::new(Arc::new(RecordingRunner::default()));
    let duplicated = plan(
        vec![
            step("tool:a", &[], &["text"]),
            step("tool:a", &[], &["text"]),
        ],
        Vec::new(),
    );
    let error = executor
        .compile(&duplicated, &chain_capabilities())
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().starts_with("step_id:"));
    assert!(error.message().contains("duplicate"));
}

#[tokio::test]
async fn edge_with_unknown_endpoint_is_rejected() {
    let executor = PlanExecutor::new(Arc::new(RecordingRunner::default()));
    let dangling = plan(
        vec![step("tool:a", &[], &["text"])],
        vec![edge("tool:a", "tool:ghost")],
    );
    let error = executor
        .compile(&dangling, &chain_capabilities())
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().starts_with("plan.edges:"));
    assert!(error.message().contains("tool:ghost"));
}

#[tokio::test]
async fn edge_cycle_is_rejected_by_stages() {
    let executor = PlanExecutor::new(Arc::new(RecordingRunner::default()));
    let cyclic = plan(
        vec![
            step("tool:a", &["x"], &["x"]),
            step("tool:b", &["x"], &["x"]),
        ],
        vec![edge("tool:a", "tool:b"), edge("tool:b", "tool:a")],
    );
    let error = executor
        .compile(&cyclic, &chain_capabilities())
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains("cycle"));
}

#[tokio::test]
async fn node_timeout_fails_the_step() {
    let capabilities = vec![capability("tool", "slow", &[], &[])];
    let plan = plan(vec![step("tool:slow", &[], &[])], Vec::new());
    let runner = Arc::new(RecordingRunner::new([(
        "tool:slow",
        Behavior::Sleep(Duration::from_millis(500), Value::Null),
    )]));
    let executor = PlanExecutor::with_config(
        ExecutorConfig {
            node_timeout_ms: Some(10),
            ..ExecutorConfig::default()
        },
        Arc::clone(&runner) as Arc<dyn CapabilityRunner>,
    );

    let error = executor
        .execute(&plan, &capabilities, None)
        .await
        .unwrap_err();
    assert!(
        matches!(error.kind(), ErrorKind::Timeout | ErrorKind::Graph),
        "unexpected kind for `{error}`"
    );
    assert!(
        error.message().contains("timed out"),
        "message `{}` must report the timeout",
        error.message()
    );
    assert!(
        error.message().contains("tool:slow"),
        "message `{}` must name the step",
        error.message()
    );
}

#[tokio::test]
async fn unsupported_runner_surfaces_the_unsupported_kind() {
    let capabilities = vec![capability("tool", "a", &[], &["text"])];
    let single = plan(vec![step("tool:a", &[], &["text"])], Vec::new());

    let executor = PlanExecutor::new(Arc::new(UnsupportedRunner::new()));
    let error = executor
        .execute(&single, &capabilities, None)
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert!(error.message().starts_with("step `tool:a` failed: "));
    assert!(error.message().contains("capability execution"));
    // Without `fail_on_deferral` a deferral keeps the `failed:` wording, not `deferred:`.
    assert_eq!(
        error.message(),
        format!(
            "step `tool:a` failed: {}",
            unsupported("capability execution").message()
        )
    );
    assert!(is_unsupported(&error));

    let lenient = PlanExecutor::with_config(
        ExecutorConfig {
            fail_fast: false,
            ..ExecutorConfig::default()
        },
        Arc::new(UnsupportedRunner::new()),
    );
    let execution = lenient.execute(&single, &capabilities, None).await.unwrap();
    assert!(!execution.succeeded());
    assert_eq!(execution.failed_steps().len(), 1);
    let recorded = execution.steps["tool:a"]
        .error
        .as_deref()
        .expect("the failure message is recorded");
    assert!(recorded.contains("not implemented"));

    // The recorded outcome classifies the gap instead of only reporting a failure.
    let outcome = &execution.steps["tool:a"];
    assert_eq!(outcome.error_kind.as_deref(), Some("unsupported"));
    assert!(outcome.deferred);
    assert_eq!(step_ids(&execution.deferred_steps()), ["tool:a"]);
    assert!(execution.has_deferrals());
    assert_eq!(
        execution.state["tool:a"]["error_kind"],
        json!("unsupported")
    );
    assert_eq!(execution.state["tool:a"]["deferred"], json!(true));
}

#[tokio::test]
async fn an_empty_plan_executes_nothing() {
    let executor = PlanExecutor::new(Arc::new(RecordingRunner::default()));
    let empty = OrchestrationPlan::new("plan-empty", "q", "en");
    let execution = executor.execute(&empty, &[], None).await.unwrap();

    assert_eq!(execution.plan_id, "plan-empty");
    assert_eq!(execution.graph_id, "plan-empty");
    assert!(execution.steps.is_empty());
    assert!(execution.outputs.is_empty());
    assert!(execution.succeeded(), "an empty plan has no failures");
    assert_eq!(execution.state, json!({}));
}

#[tokio::test]
async fn all_deferred_steps_are_accounted_as_gaps_not_hidden_failures() {
    let runner = Arc::new(deferred_chain_runner());
    let executor = PlanExecutor::with_config(
        ExecutorConfig {
            fail_fast: false,
            ..ExecutorConfig::default()
        },
        Arc::clone(&runner) as Arc<dyn CapabilityRunner>,
    );

    let execution = executor
        .execute(&deferred_chain_plan(), &deferred_chain_capabilities(), None)
        .await
        .expect("a capability gap is recorded, not fatal, while fail_on_deferral is off");

    // Both steps ran: a deferral does not stop a lenient run.
    assert_eq!(runner.calls().len(), 2);

    assert!(!execution.succeeded());
    assert_eq!(execution.steps.len(), 2);
    let failed = execution.failed_steps();
    assert_eq!(failed.len(), 2);
    assert_eq!(step_ids(&failed), ["tool:x", "tool:y"]);
    assert!(
        failed.iter().all(|outcome| outcome.deferred),
        "every failure here is a typed capability gap"
    );
    let deferred = execution.deferred_steps();
    assert_eq!(deferred.len(), 2);
    assert_eq!(step_ids(&deferred), ["tool:x", "tool:y"]);
    assert!(execution.has_deferrals());

    for step_id in ["tool:x", "tool:y"] {
        let outcome = &execution.steps[step_id];
        assert!(
            !outcome.success,
            "a deferred step is a failure, never a silent success"
        );
        assert_eq!(outcome.output, Value::Null, "step `{step_id}`");
        assert_eq!(
            outcome.error_kind.as_deref(),
            Some("unsupported"),
            "step `{step_id}`"
        );
        assert!(outcome.deferred, "step `{step_id}`");
        let error = outcome
            .error
            .as_deref()
            .expect("the deferral message is recorded");
        assert!(
            error.contains(DEFERRED_FEATURE),
            "message `{error}` must name the deferred feature"
        );
        assert!(
            error.contains("not implemented"),
            "message `{error}` must say the feature is unimplemented"
        );

        // The state record exposes the same two classifications.
        let record = &execution.state[step_id];
        assert_eq!(record["success"], json!(false), "record of `{step_id}`");
        assert_eq!(record["output"], Value::Null, "record of `{step_id}`");
        assert_eq!(
            record["error_kind"],
            json!("unsupported"),
            "record of `{step_id}`"
        );
        assert_eq!(record["deferred"], json!(true), "record of `{step_id}`");
        assert!(
            record["duration_ms"].is_u64(),
            "record of `{step_id}` must carry a monotonic duration: {record}"
        );
    }

    assert!(
        execution.outputs.is_empty(),
        "a deferred step contributes no output"
    );
}

#[tokio::test]
async fn strict_deferral_mode_aborts_on_the_first_gap_for_both_fail_fast_values() {
    let expected_message = format!("step `tool:x` deferred: {}", deferred_error().message());

    for fail_fast in [true, false] {
        let runner = Arc::new(deferred_chain_runner());
        let executor = PlanExecutor::with_config(
            ExecutorConfig {
                fail_fast,
                fail_on_deferral: true,
                ..ExecutorConfig::default()
            },
            Arc::clone(&runner) as Arc<dyn CapabilityRunner>,
        );

        let error = executor
            .execute(&deferred_chain_plan(), &deferred_chain_capabilities(), None)
            .await
            .expect_err("fail_on_deferral must abort on the first deferral");

        assert_eq!(
            error.kind(),
            ErrorKind::Unsupported,
            "fail_fast = {fail_fast}"
        );
        assert_ne!(error.kind(), ErrorKind::Graph);
        assert_eq!(error.message(), expected_message, "fail_fast = {fail_fast}");
        assert!(is_unsupported(&error));

        // The abort is the deferral itself, so the second step never runs — even when a
        // runner error would otherwise be tolerated.
        let calls = runner.calls();
        assert_eq!(
            calls.len(),
            1,
            "`tool:y` must not run; fail_fast = {fail_fast}"
        );
        assert_eq!(calls[0].capability_id, "tool:x");
    }
}

#[tokio::test]
async fn mixed_plan_reports_success_and_deferral_separately() {
    let capabilities = vec![
        capability("tool", "ok", &[], &["done"]),
        capability("tool", "gap", &[], &[]),
    ];
    let mixed = plan(
        vec![step("tool:ok", &[], &["done"]), step("tool:gap", &[], &[])],
        Vec::new(),
    );
    let runner = Arc::new(RecordingRunner::new([
        ("tool:ok", Behavior::Output(json!({"done": true}))),
        ("tool:gap", Behavior::Deferred),
    ]));
    let config = ExecutorConfig {
        fail_fast: false,
        ..ExecutorConfig::default()
    };
    assert!(
        !config.fail_on_deferral,
        "the default keeps deferrals non-fatal"
    );
    let executor =
        PlanExecutor::with_config(config, Arc::clone(&runner) as Arc<dyn CapabilityRunner>);

    let execution = executor
        .execute(&mixed, &capabilities, None)
        .await
        .expect("a mixed plan stays Ok under the default deferral policy");
    assert_eq!(runner.calls().len(), 2);

    let success = &execution.steps["tool:ok"];
    assert!(success.success);
    assert_eq!(success.error, None);
    assert_eq!(success.error_kind, None);
    assert!(!success.deferred);
    assert_eq!(execution.outputs["tool:ok"], json!({"done": true}));
    assert_eq!(execution.state["tool:ok"]["error_kind"], Value::Null);
    assert_eq!(execution.state["tool:ok"]["deferred"], json!(false));

    let gap = &execution.steps["tool:gap"];
    assert!(!gap.success);
    assert_eq!(gap.output, Value::Null);
    assert_eq!(gap.error_kind.as_deref(), Some("unsupported"));
    assert!(gap.deferred);
    assert_eq!(
        execution.state["tool:gap"]["error_kind"],
        json!("unsupported")
    );
    assert_eq!(execution.state["tool:gap"]["deferred"], json!(true));
    assert!(!execution.outputs.contains_key("tool:gap"));
    assert_eq!(execution.outputs.len(), 1);

    // Deferred steps are a subset of the failed steps, and the success is in neither.
    assert_eq!(step_ids(&execution.failed_steps()), ["tool:gap"]);
    assert_eq!(step_ids(&execution.deferred_steps()), ["tool:gap"]);
    assert!(
        !step_ids(&execution.failed_steps()).contains(&"tool:ok"),
        "a successful step must never be reported as failed or deferred"
    );
    assert!(!step_ids(&execution.deferred_steps()).contains(&"tool:ok"));
    assert!(execution.has_deferrals());
    assert!(!execution.succeeded());
}

#[tokio::test]
async fn strict_deferral_mode_does_not_reclassify_a_genuine_failure() {
    let capabilities = chain_capabilities();
    let chain = chain_plan();

    let strict = PlanExecutor::with_config(
        ExecutorConfig {
            fail_on_deferral: true,
            ..ExecutorConfig::default()
        },
        Arc::new(failing_chain_runner()),
    );
    let error = strict
        .execute(&chain, &capabilities, None)
        .await
        .expect_err("a tool failure aborts under fail_fast");
    assert_eq!(
        error.kind(),
        ErrorKind::Graph,
        "a runtime failure keeps the graph mapping under fail_on_deferral"
    );
    assert_eq!(error.message(), "step `tool:b` failed: boom");
    assert!(!is_unsupported(&error));
    assert!(
        !error.message().contains("deferred"),
        "a genuine failure must not be reported as a deferral: {error}"
    );

    // A lenient run records the same failure as a non-deferred failure.
    let lenient = PlanExecutor::with_config(
        ExecutorConfig {
            fail_fast: false,
            fail_on_deferral: true,
            ..ExecutorConfig::default()
        },
        Arc::new(failing_chain_runner()),
    );
    let execution = lenient
        .execute(&chain, &capabilities, None)
        .await
        .expect("a runtime failure is recorded, not fatal, when fail_fast is off");
    let failing = &execution.steps["tool:b"];
    assert!(!failing.success);
    assert_eq!(failing.error_kind.as_deref(), Some("tool"));
    assert!(!failing.deferred);
    assert_eq!(execution.state["tool:b"]["error_kind"], json!("tool"));
    assert_eq!(execution.state["tool:b"]["deferred"], json!(false));
    assert_eq!(step_ids(&execution.failed_steps()), ["tool:b"]);
    assert!(!execution.has_deferrals());
    assert!(execution.deferred_steps().is_empty());
}

#[tokio::test]
async fn executed_mixed_plan_round_trips_through_json() {
    let capabilities = vec![
        capability("tool", "ok", &[], &["done"]),
        capability("tool", "gap", &[], &[]),
    ];
    let mixed = plan(
        vec![step("tool:ok", &[], &["done"]), step("tool:gap", &[], &[])],
        Vec::new(),
    );
    let executor = PlanExecutor::with_config(
        ExecutorConfig {
            fail_fast: false,
            ..ExecutorConfig::default()
        },
        Arc::new(RecordingRunner::new([
            ("tool:ok", Behavior::Output(json!({"done": true}))),
            ("tool:gap", Behavior::Deferred),
        ])),
    );
    let execution = executor
        .execute(&mixed, &capabilities, None)
        .await
        .expect("mixed run");

    let encoded = serde_json::to_value(&execution).expect("serialize the execution");
    assert_eq!(encoded["plan_id"], json!("plan-1"));
    assert_eq!(encoded["steps"]["tool:ok"]["success"], json!(true));
    assert_eq!(encoded["steps"]["tool:ok"]["error_kind"], Value::Null);
    assert_eq!(encoded["steps"]["tool:ok"]["deferred"], json!(false));
    assert_eq!(encoded["steps"]["tool:gap"]["success"], json!(false));
    assert_eq!(
        encoded["steps"]["tool:gap"]["error_kind"],
        json!("unsupported")
    );
    assert_eq!(encoded["steps"]["tool:gap"]["deferred"], json!(true));

    let decoded: PlanExecution =
        serde_json::from_value(encoded.clone()).expect("deserialize the execution");
    assert_eq!(decoded, execution);
    assert_eq!(step_ids(&decoded.deferred_steps()), ["tool:gap"]);
    assert!(decoded.has_deferrals());
    assert_eq!(step_ids(&decoded.failed_steps()), ["tool:gap"]);
    assert_eq!(decoded.outputs["tool:ok"], json!({"done": true}));

    let outcome = decoded.steps["tool:gap"].clone();
    let encoded_outcome = serde_json::to_value(&outcome).expect("serialize the outcome");
    assert_eq!(
        serde_json::from_value::<StepOutcome>(encoded_outcome).expect("round trip"),
        outcome
    );
}

#[test]
fn legacy_payloads_without_the_deferral_fields_use_defaults() {
    // A failure written before `error_kind`/`deferred` existed.
    let legacy_failure: StepOutcome = serde_json::from_value(json!({
        "step_id": "tool:b",
        "capability_type": "tool",
        "success": false,
        "output": null,
        "error": "boom",
        "duration_ms": 7,
    }))
    .expect("legacy failed outcome");
    assert_eq!(legacy_failure.error.as_deref(), Some("boom"));
    assert_eq!(legacy_failure.error_kind, None, "the documented default");
    assert!(!legacy_failure.deferred, "the documented default");

    let legacy_success: StepOutcome = serde_json::from_value(json!({
        "step_id": "tool:a",
        "capability_type": "tool",
        "success": true,
        "output": {"text": "alpha"},
        "duration_ms": 2,
    }))
    .expect("legacy successful outcome");
    assert_eq!(legacy_success.error, None, "the documented default");
    assert_eq!(legacy_success.error_kind, None);
    assert!(!legacy_success.deferred);

    // A whole execution document from before the fields, with no `outputs`/`state`.
    let legacy_execution: PlanExecution = serde_json::from_value(json!({
        "plan_id": "plan-legacy",
        "graph_id": "plan-legacy",
        "steps": {
            "tool:a": legacy_success,
            "tool:b": legacy_failure,
        },
    }))
    .expect("legacy execution");
    assert_eq!(legacy_execution.steps.len(), 2);
    assert_eq!(step_ids(&legacy_execution.failed_steps()), ["tool:b"]);
    assert!(legacy_execution.deferred_steps().is_empty());
    assert!(!legacy_execution.has_deferrals());
    assert!(legacy_execution.outputs.is_empty());
    assert_eq!(legacy_execution.state, Value::Null);
}
