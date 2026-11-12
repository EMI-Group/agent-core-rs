#![allow(clippy::unwrap_used, clippy::expect_used, clippy::too_many_lines)]
//! Symphony scenario: the capability-asset engine driven as a whole.
//!
//! This binary is the cross-crate seam above `openjiuwen-symphony`: a
//! [`SymphonyRuntime`] derives a capability graph from registered cards, the bounded
//! planner turns a request into an `OrchestrationPlan`, `PlanExecutor` compiles that plan
//! onto `openjiuwen::core::graph` and runs it through a `CapabilityRunner`, and the
//! artifact store persists a revision that a fresh runtime reads back from disk.
//!
//! Everything is offline and deterministic: capabilities are cards, relevance scores come
//! from a hand-set `SemanticScorer` instead of tokenizer details, the runner is a local
//! recorder, and file-backed revisions live in a `tempfile` directory. Evidence is
//! recorded state — registry contents, plan steps and edges, execution outcomes, revision
//! metadata — never generated ids (`plan_id` is a uuid), `created_at`, or `duration_ms`.
//!
//! [`SymphonyRuntime`]: openjiuwen::symphony::runtime::SymphonyRuntime

mod support;

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use openjiuwen::core::graph::CompiledGraph;
use openjiuwen::symphony::model::{CapabilityDescriptor, CapabilityIo};
use openjiuwen::symphony::orchestration::{
    CapabilityRunner, PlanConfig, PlanEdge, PlanExecutor, PlanRequest, PlanStatus, Planner,
    RunnerContext, SemanticScorer,
};
use openjiuwen::symphony::runtime::{RuntimeState, SymphonyConfig, SymphonyRuntime};
use openjiuwen::symphony::Result;
use serde_json::{json, Value};
use support::within;

/// The entry capability of the chain: it consumes nothing and produces `data`.
const FETCH: &str = "tool:fetch";
/// The middle capability: it consumes `data` and produces `result`.
const TRANSFORM: &str = "tool:transform";
/// The tail capability: it consumes `result` and produces nothing.
const SUMMARIZE: &str = "tool:summarize";

/// One string-typed IO, which is all the chain contract needs.
fn io(name: &str) -> CapabilityIo {
    CapabilityIo::new(name, "string")
}

/// The three-capability chain `fetch -> transform -> summarize`.
///
/// The IO names match exactly across each link (`data`, then `result`), so the candidate
/// rules derive two `can_feed` edges at confidence `1.0`.
/// Only the entry capability is offered as a plan start; the two downstream cards stay
/// reachable through the derived edges, which is the catalog shape the planner's own
/// chain fixture uses to steer a search along a single path.
fn chain_catalog() -> Vec<CapabilityDescriptor> {
    vec![
        CapabilityDescriptor::new("tool", "fetch", "Fetch Data")
            .with_description("fetches the raw data for a request")
            .with_outputs(vec![io("data")]),
        CapabilityDescriptor::new("tool", "transform", "Transform Data")
            .with_description("transforms data into a result")
            .with_inputs(vec![io("data")])
            .with_outputs(vec![io("result")])
            .with_available(false),
        CapabilityDescriptor::new("tool", "summarize", "Summarize Result")
            .with_description("summarizes a result into a summary")
            .with_inputs(vec![io("result")])
            .with_available(false),
    ]
}

/// A scorer with hand-set relevance, so plan selection is arithmetic rather than a
/// tokenizer detail.
///
/// The entry capability is deliberately weak and both downstream ones strong: that is what
/// makes the full three-step chain outrank its one- and two-step prefixes once the
/// planner's depth penalty is applied.
struct FixedScorer {
    /// Qualified capability id to relevance score.
    scores: BTreeMap<String, f64>,
}

impl FixedScorer {
    /// The chain's scores: `0.2` for the entry, `1.0` for both downstream capabilities.
    fn chain() -> Self {
        Self {
            scores: BTreeMap::from([
                (FETCH.to_string(), 0.2),
                (TRANSFORM.to_string(), 1.0),
                (SUMMARIZE.to_string(), 1.0),
            ]),
        }
    }
}

#[async_trait]
impl SemanticScorer for FixedScorer {
    async fn score(
        &self,
        _query: &str,
        capabilities: &[CapabilityDescriptor],
    ) -> Result<BTreeMap<String, f64>> {
        Ok(capabilities
            .iter()
            .map(|capability| {
                let id = capability.qualified_id();
                let score = self.scores.get(&id).copied().unwrap_or(0.0);
                (id, score)
            })
            .collect())
    }
}

/// The runtime's planner: the fixed scorer over the default bounded search.
fn chain_planner() -> Planner {
    Planner::with_scorer(
        PlanConfig::default(),
        Arc::new(FixedScorer::chain()) as Arc<dyn SemanticScorer>,
    )
}

/// One recorded [`CapabilityRunner`] invocation.
#[derive(Debug, Clone, PartialEq)]
struct Call {
    /// The qualified id of the capability the executor resolved for the step.
    capability_id: String,
    /// The overlay payload the step was invoked with.
    inputs: Value,
    /// The session id the executor passed along.
    session_id: Option<String>,
}

/// A [`CapabilityRunner`] that replays scripted outputs and records every call.
#[derive(Clone)]
struct RecordingRunner {
    /// Qualified capability id to the output to return.
    outputs: BTreeMap<String, Value>,
    /// The calls, in invocation order.
    calls: Arc<Mutex<Vec<Call>>>,
}

impl RecordingRunner {
    /// Builds a runner from `(qualified capability id, output)` pairs.
    fn new(outputs: impl IntoIterator<Item = (&'static str, Value)>) -> Self {
        Self {
            outputs: outputs
                .into_iter()
                .map(|(id, output)| (id.to_string(), output))
                .collect(),
            calls: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Returns every recorded call in invocation order.
    fn calls(&self) -> Vec<Call> {
        self.calls.lock().expect("calls mutex").clone()
    }

    /// Returns the qualified ids invoked so far, in invocation order.
    fn invoked(&self) -> Vec<String> {
        self.calls()
            .into_iter()
            .map(|call| call.capability_id)
            .collect()
    }

    /// Returns the payload recorded for `capability_id`.
    fn inputs_for(&self, capability_id: &str) -> Value {
        self.calls()
            .into_iter()
            .find(|call| call.capability_id == capability_id)
            .unwrap_or_else(|| panic!("no call recorded for `{capability_id}`"))
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
        let output = self
            .outputs
            .get(&capability_id)
            .unwrap_or_else(|| panic!("no scripted output for `{capability_id}`"))
            .clone();
        self.calls.lock().expect("calls mutex").push(Call {
            capability_id,
            inputs,
            session_id: context.session_id.clone(),
        });
        Ok(output)
    }
}

#[tokio::test]
async fn register_plan_and_execute_against_a_compiled_graph() {
    let workspace = tempfile::tempdir().expect("a scenario-owned temp dir");
    let config = SymphonyConfig {
        graph_artifact_root: Some(workspace.path().join("artifacts")),
        ..SymphonyConfig::default()
    };
    let runtime = within(
        "the symphony runtime opens",
        SymphonyRuntime::builder(config)
            .with_planner(chain_planner())
            .open(),
    )
    .await
    .expect("open the runtime");
    assert_eq!(runtime.state(), RuntimeState::Created);
    for capability in chain_catalog() {
        runtime
            .registry()
            .register(capability)
            .await
            .expect("register a chain capability");
    }
    within("the runtime starts", runtime.start())
        .await
        .expect("start");
    assert_eq!(runtime.state(), RuntimeState::Running);

    // Starting derives the graph from the catalog: one node per card and one `can_feed`
    // edge per exact-IO match (`data`, then `result`).
    let status = runtime.graph_status().await.expect("graph status");
    assert_eq!(status.current_version.as_deref(), Some("v1"));
    assert_eq!(status.node_count, 3);
    assert_eq!(status.edge_count, 2);

    let mut request = PlanRequest::new("fetch the data, transform it, then summarize the result");
    request.language = "en".to_string();
    let plan = within("the request plans", runtime.plan(&request))
        .await
        .expect("plan");
    assert_eq!(
        plan.status,
        PlanStatus::Ready,
        "the entry capability needs no caller input: {:?}",
        plan.message
    );
    let step_ids: Vec<&str> = plan
        .steps
        .iter()
        .map(|step| step.step_id.as_str())
        .collect();
    assert_eq!(step_ids, vec![FETCH, TRANSFORM, SUMMARIZE]);
    assert_eq!(
        plan.stages().expect("the plan is acyclic"),
        vec![
            vec![FETCH.to_string()],
            vec![TRANSFORM.to_string()],
            vec![SUMMARIZE.to_string()],
        ],
        "one capability per stage, in chain order"
    );
    assert_eq!(
        plan.edges,
        vec![
            PlanEdge {
                source: FETCH.to_string(),
                target: TRANSFORM.to_string(),
            },
            PlanEdge {
                source: TRANSFORM.to_string(),
                target: SUMMARIZE.to_string(),
            },
        ],
        "the derived can_feed edges carry over into the plan"
    );
    for step in &plan.steps {
        assert!(
            step.missing_inputs.is_empty(),
            "step `{}` must have every input filled: {:?}",
            step.step_id,
            step.missing_inputs
        );
        assert_eq!(step.capability_type, "tool");
    }

    let capabilities = runtime.registry().list().await;
    assert_eq!(capabilities.len(), 3);
    let runner = Arc::new(RecordingRunner::new([
        (FETCH, json!({"data": "raw data"})),
        (TRANSFORM, json!({"result": "transformed result"})),
        (SUMMARIZE, json!({"summary": "final summary"})),
    ]));
    let executor = PlanExecutor::new(Arc::clone(&runner) as Arc<dyn CapabilityRunner>);
    let compiled: CompiledGraph = executor
        .compile(&plan, &capabilities)
        .expect("the plan compiles onto the graph engine");
    assert_eq!(compiled.id(), plan.plan_id);
    assert_eq!(
        compiled.node_ids().collect::<Vec<_>>(),
        vec![FETCH, TRANSFORM, SUMMARIZE]
    );
    assert_eq!(compiled.entry_nodes().to_vec(), vec![FETCH.to_string()]);
    assert_eq!(compiled.exit_nodes(), vec![SUMMARIZE]);

    let execution = within(
        "the compiled plan executes",
        executor.execute_with_inputs(&plan, &capabilities, json!({}), None),
    )
    .await
    .expect("execute");
    assert_eq!(execution.plan_id, plan.plan_id);
    assert_eq!(execution.graph_id, plan.plan_id);
    assert!(
        execution.succeeded(),
        "failed steps: {:?}",
        execution.failed_steps()
    );
    // `PlanExecution::steps` is a `BTreeMap`, so its keys read back sorted rather than in
    // plan order; every planned step must still have exactly one recorded outcome.
    let mut executed: Vec<&str> = execution.steps.keys().map(String::as_str).collect();
    let mut planned: Vec<&str> = plan
        .steps
        .iter()
        .map(|step| step.step_id.as_str())
        .collect();
    executed.sort_unstable();
    planned.sort_unstable();
    assert_eq!(executed, planned);
    for (step_id, outcome) in &execution.steps {
        assert!(outcome.success, "step `{step_id}` must be successful");
        assert_eq!(outcome.step_id, *step_id);
        assert_eq!(outcome.capability_type, "tool");
        assert_eq!(outcome.error, None);
        assert_eq!(execution.state[step_id]["success"], json!(true));
        assert_eq!(execution.state[step_id]["error"], Value::Null);
    }
    let expected_outputs: BTreeMap<String, Value> = BTreeMap::from([
        (FETCH.to_string(), json!({"data": "raw data"})),
        (
            TRANSFORM.to_string(),
            json!({"result": "transformed result"}),
        ),
        (SUMMARIZE.to_string(), json!({"summary": "final summary"})),
    ]);
    assert_eq!(execution.outputs, expected_outputs);

    // Recorded evidence: one invocation per planned step, all inside one default session,
    // and each payload overlaid from its predecessor's output.
    let calls = runner.calls();
    assert_eq!(calls.len(), plan.steps.len());
    let mut invoked = runner.invoked();
    invoked.sort();
    assert_eq!(
        invoked,
        vec![
            FETCH.to_string(),
            SUMMARIZE.to_string(),
            TRANSFORM.to_string()
        ]
    );
    assert!(
        calls.iter().all(|call| call.session_id.is_some()),
        "the executor seeds a default workflow session"
    );
    assert_eq!(runner.inputs_for(FETCH), json!({}));
    assert_eq!(runner.inputs_for(TRANSFORM), json!({"data": "raw data"}));
    assert_eq!(
        runner.inputs_for(SUMMARIZE),
        json!({"result": "transformed result"})
    );

    within("the runtime closes", runtime.close())
        .await
        .expect("close");
    assert_eq!(runtime.state(), RuntimeState::Stopped);
}

#[tokio::test]
async fn plan_is_deterministic_and_reports_when_inputs_are_missing() {
    let runtime = within(
        "the runtime opens",
        SymphonyRuntime::open(SymphonyConfig::default()),
    )
    .await
    .expect("open");
    runtime
        .registry()
        .register(
            CapabilityDescriptor::new("tool", "search", "Search Documentation")
                .with_description("searches the documentation index")
                .with_inputs(vec![
                    CapabilityIo::new("prompt", "string").with_required(true)
                ]),
        )
        .await
        .expect("register the search capability");

    let mut request = PlanRequest::new("search the documentation");
    request.language = "en".to_string();
    let first = within("the first plan", runtime.plan(&request))
        .await
        .expect("plan");
    assert_eq!(first.status, PlanStatus::NeedsInput);
    assert_eq!(first.language, "en");
    assert_eq!(first.steps.len(), 1);
    assert_eq!(first.steps[0].step_id, "tool:search");
    assert_eq!(first.steps[0].missing_inputs, vec!["prompt".to_string()]);
    assert!(
        first.steps[0].filled_inputs.is_empty(),
        "the caller supplied no inputs, so none are filled"
    );
    assert_eq!(first.message.as_deref(), Some("missing inputs: prompt"));

    // The same request planned twice yields the same plan; `plan_id` is a fresh uuid per
    // call and is deliberately not part of the comparison.
    let second = within("the repeated plan", runtime.plan(&request))
        .await
        .expect("plan again");
    assert_eq!(second.status, first.status);
    assert_eq!(second.title, first.title);
    assert_eq!(second.message, first.message);
    assert_eq!(second.steps, first.steps);
    assert_eq!(second.edges, first.edges);
    assert!(
        (second.score - first.score).abs() < 1e-9,
        "scores {} and {} must agree",
        second.score,
        first.score
    );

    // A request no card matches lexically has no start capability at all.
    let unmatched = PlanRequest {
        query: "quarterly payroll audit".to_string(),
        language: "en".to_string(),
        ..PlanRequest::default()
    };
    let none = within("the unmatched plan", runtime.plan(&unmatched))
        .await
        .expect("plan");
    assert_eq!(none.status, PlanStatus::NoPlan);
    assert!(none.steps.is_empty(), "an unmatched request plans no steps");
    assert!(
        none.edges.is_empty(),
        "an unmatched request derives no edges"
    );
    assert!(none.message.is_none());

    // Supplying the input the caller promised turns the unfed request into a ready plan.
    let mut supplied = request.clone();
    supplied.provided_inputs = vec!["prompt".to_string()];
    let ready = within("the supplied plan", runtime.plan(&supplied))
        .await
        .expect("plan");
    assert_eq!(ready.status, PlanStatus::Ready);
    assert!(
        ready.steps[0].missing_inputs.is_empty(),
        "supplying `prompt` fills the only input"
    );
    assert_eq!(ready.steps[0].filled_inputs, vec!["prompt".to_string()]);
    assert!(ready.message.is_none());
}

#[tokio::test]
async fn graph_snapshot_reloads_into_a_fresh_runtime() {
    let workspace = tempfile::tempdir().expect("a scenario-owned temp dir");
    let config = SymphonyConfig {
        graph_artifact_root: Some(workspace.path().join("artifacts")),
        ..SymphonyConfig::default()
    };

    let first = within(
        "the first runtime opens",
        SymphonyRuntime::open(config.clone()),
    )
    .await
    .expect("open the first runtime");
    within("the first runtime starts", first.start())
        .await
        .expect("start");
    let added = within(
        "the capabilities are added",
        first.add_capabilities(chain_catalog(), "req-snapshot"),
    )
    .await
    .expect("add capabilities");
    assert_eq!(
        added.revision, "v2",
        "starting publishes v1, the add publishes v2"
    );
    assert_eq!(added.changed.len(), 3);

    let published = within("the published revision reads back", first.read_graph(None))
        .await
        .expect("read graph")
        .expect("a revision is published");
    assert_eq!(published.node_count(), 3);
    assert_eq!(published.edge_count(), 2);
    for node in [FETCH, TRANSFORM, SUMMARIZE] {
        assert!(published.has_node(node), "`{node}` is in the snapshot");
    }
    let published_status = first.graph_status().await.expect("status");
    assert!(published_status.is_ready());
    assert_eq!(published_status.current_version.as_deref(), Some("v2"));
    assert_eq!(published_status.node_count, published.node_count());
    assert_eq!(published_status.edge_count, published.edge_count());

    within("the first runtime closes", first.close())
        .await
        .expect("close");
    drop(first);

    // A fresh runtime on the same artifact root starts empty and reads the snapshot back.
    let second = within(
        "a fresh runtime opens on the same root",
        SymphonyRuntime::open(config),
    )
    .await
    .expect("reopen the runtime");
    assert_eq!(second.state(), RuntimeState::Created);
    assert_eq!(second.registry().len(), 0);
    let status = within("the reloaded status", second.graph_status())
        .await
        .expect("status");
    assert_eq!(status.status, published_status.status);
    assert_eq!(status.current_version, published_status.current_version);
    assert_eq!(status.node_count, published_status.node_count);
    assert_eq!(status.edge_count, published_status.edge_count);
    let reloaded = within("the reloaded revision reads back", second.read_graph(None))
        .await
        .expect("read graph")
        .expect("the snapshot survives the restart");
    assert_eq!(reloaded, published);
}
