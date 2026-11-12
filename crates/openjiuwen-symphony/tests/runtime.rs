//! Behaviour tests for [`SymphonyRuntime`]: lifecycle, graph build and publication,
//! planning, invocation, idempotent mutations, evolution feedback, semantic search, and
//! file-backed revision persistence.
//!
//! Everything here is offline and deterministic: the capability runner is a local fake,
//! the semantic index embeds with `HashEmbedding`, file-backed runs use `tempfile`, and no
//! assertion depends on the wall clock or on hash-map iteration order.
#![allow(clippy::unwrap_used, clippy::expect_used)]
#![allow(clippy::float_cmp)]
#![allow(clippy::assert_is_empty)]

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use openjiuwen_retrieval::{Embedding, HashEmbedding};
use openjiuwen_symphony::artifact::{GraphArtifactStore, MemoryGraphArtifactStore};
use openjiuwen_symphony::error::{ErrorKind, Result};
use openjiuwen_symphony::graph::{CapabilityEdge, CapabilityGraph, CAN_FEED};
use openjiuwen_symphony::model::{CapabilityDescriptor, CapabilityIo, CapabilityKey};
use openjiuwen_symphony::orchestration::PlanExecution;
use openjiuwen_symphony::orchestration::{
    CandidateConfig, CapabilityRunner, ExecutorConfig, PlanConfig, PlanRequest, PlanStatus,
    RunnerContext, StepOutcome,
};
use openjiuwen_symphony::runtime::{
    InvokeRequest, MutationOperation, RuntimeState, SymphonyConfig, SymphonyRuntime,
};
use openjiuwen_symphony::search::{SearchRequest, SemanticIndex};
use serde_json::{json, Value};

/// One string-typed IO, which is all the catalog needs.
fn io(name: &str) -> CapabilityIo {
    CapabilityIo::new(name, "string")
}

/// `tool:fetch` — produces `data`.
fn fetch() -> CapabilityDescriptor {
    CapabilityDescriptor::new("tool", "fetch", "Fetch Data")
        .with_description("fetches raw data from the store")
        .with_outputs(vec![io("data")])
}

/// `tool:transform` — consumes `data` and produces `result`.
fn transform() -> CapabilityDescriptor {
    CapabilityDescriptor::new("tool", "transform", "Transform Data")
        .with_description("transforms data into a result")
        .with_inputs(vec![io("data")])
        .with_outputs(vec![io("result")])
}

/// A two-capability catalog whose `can_feed` edge is `fetch -> transform`.
fn catalog() -> Vec<CapabilityDescriptor> {
    vec![fetch(), transform()]
}

/// A capability with a natural-language identity, for the search tests.
fn sum() -> CapabilityDescriptor {
    CapabilityDescriptor::new("tool", "sum", "Sum Numbers")
        .with_description("adds two numbers together")
        .with_inputs(vec![io("left"), io("right")])
        .with_outputs(vec![io("total")])
}

/// Opens an in-memory runtime holding [`catalog`] in its registry, without starting it.
async fn runtime_with_catalog() -> SymphonyRuntime {
    let runtime = SymphonyRuntime::open(SymphonyConfig::default())
        .await
        .expect("open runtime");
    for descriptor in catalog() {
        runtime
            .registry()
            .register(descriptor)
            .await
            .expect("register");
    }
    runtime
}

/// A [`CapabilityRunner`] that returns scripted outputs and records every call.
#[derive(Clone, Default)]
struct FakeRunner {
    /// The output keyed by qualified capability id.
    outputs: BTreeMap<String, Value>,
    /// The `(qualified id, payload)` pairs recorded in invocation order.
    calls: Arc<Mutex<Vec<(String, Value)>>>,
}

impl FakeRunner {
    /// Builds a runner from `(qualified id, output)` pairs.
    fn new(outputs: impl IntoIterator<Item = (&'static str, Value)>) -> Self {
        Self {
            outputs: outputs
                .into_iter()
                .map(|(id, output)| (id.to_string(), output))
                .collect(),
            calls: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// The qualified ids invoked so far, in order.
    fn invoked(&self) -> Vec<String> {
        self.calls
            .lock()
            .expect("calls mutex")
            .iter()
            .map(|(id, _)| id.clone())
            .collect()
    }

    /// The payload the runner received for `step_id`.
    fn inputs_for(&self, step_id: &str) -> Value {
        self.calls
            .lock()
            .expect("calls mutex")
            .iter()
            .find(|(id, _)| id == step_id)
            .map_or(Value::Null, |(_, inputs)| inputs.clone())
    }
}

#[async_trait]
impl CapabilityRunner for FakeRunner {
    async fn invoke(
        &self,
        capability: &CapabilityDescriptor,
        inputs: Value,
        _context: &RunnerContext,
    ) -> Result<Value> {
        let qualified_id = capability.qualified_id();
        self.calls
            .lock()
            .expect("calls mutex")
            .push((qualified_id.clone(), inputs));
        Ok(self
            .outputs
            .get(&qualified_id)
            .cloned()
            .unwrap_or(Value::Null))
    }
}

#[test]
fn config_defaults_are_the_documented_values() {
    let config = SymphonyConfig::default();
    assert!(config.graph_artifact_root.is_none());
    assert_eq!(config.graph_scope_id, "default");
    assert_eq!(config.plan, PlanConfig::default());
    assert_eq!(config.executor, ExecutorConfig::default());
    assert_eq!(config.candidates, CandidateConfig::default());

    let encoded = serde_json::to_value(&config).expect("serialize config");
    assert_eq!(encoded["graph_scope_id"], json!("default"));
    assert_eq!(encoded["graph_artifact_root"], Value::Null);
    let decoded: SymphonyConfig = serde_json::from_value(encoded).expect("deserialize config");
    assert_eq!(decoded, config);
}

#[tokio::test]
async fn open_starts_created_and_lifecycle_transitions_are_idempotent() {
    let runtime = SymphonyRuntime::open(SymphonyConfig::default())
        .await
        .expect("open runtime");
    assert_eq!(runtime.state(), RuntimeState::Created);

    runtime.start().await.expect("start");
    assert_eq!(runtime.state(), RuntimeState::Running);
    // An empty catalog still produces one (empty) revision, and a second start is a no-op.
    assert_eq!(
        runtime
            .graph_status()
            .await
            .expect("status")
            .current_version
            .as_deref(),
        Some("v1")
    );
    runtime.start().await.expect("start again");
    assert_eq!(runtime.state(), RuntimeState::Running);

    runtime.stop().await.expect("stop");
    assert_eq!(runtime.state(), RuntimeState::Stopped);
    runtime.stop().await.expect("stop again");
    assert_eq!(runtime.state(), RuntimeState::Stopped);

    runtime.close().await.expect("close");
    assert_eq!(runtime.state(), RuntimeState::Stopped);
    runtime.close().await.expect("close again");

    // A stopped runtime can be started again without publishing a second revision.
    runtime.start().await.expect("restart");
    assert_eq!(runtime.state(), RuntimeState::Running);
    assert_eq!(
        runtime
            .graph_status()
            .await
            .expect("status")
            .current_version
            .as_deref(),
        Some("v1")
    );
}

#[tokio::test]
async fn start_builds_a_revision_when_the_catalog_has_capabilities() {
    let runtime = runtime_with_catalog().await;
    assert!(!runtime.graph_status().await.expect("status").is_ready());

    runtime.start().await.expect("start");
    let status = runtime.graph_status().await.expect("status");
    assert!(status.is_ready());
    assert_eq!(status.current_version.as_deref(), Some("v1"));
    assert_eq!(status.graph_id.as_deref(), Some("default"));
    assert_eq!(status.node_count, 2);
    assert_eq!(status.edge_count, 1);
    assert_eq!(status.snapshot["current_version"], json!("v1"));
}

#[tokio::test]
async fn build_versions_are_incrementing_and_older_versions_stay_readable() {
    let runtime = runtime_with_catalog().await;
    let first = runtime.build().await.expect("first build");
    assert_eq!(first.version, "v1");
    assert_eq!(first.node_count, 2);
    assert_eq!(first.edge_count, 1);

    let second = runtime.build().await.expect("second build");
    assert_eq!(second.version, "v2");

    let v1 = runtime
        .read_graph(Some("v1"))
        .await
        .expect("read v1")
        .expect("v1 exists");
    assert!(v1.has_node("tool:fetch"));
    assert!(v1.has_node("tool:transform"));
    assert_eq!(v1.node_count(), 2);
    assert_eq!(v1.edges.len(), 1);
    assert_eq!(v1.edges[0].relation, CAN_FEED);

    let current = runtime
        .read_graph(None)
        .await
        .expect("read current")
        .expect("current exists");
    assert_eq!(current.graph_id, "default");
    assert!(runtime
        .read_graph(Some("v99"))
        .await
        .expect("read absent")
        .is_none());
}

#[tokio::test]
async fn graph_status_and_snapshot_report_nulls_before_the_first_build() {
    let runtime = SymphonyRuntime::open(SymphonyConfig::default())
        .await
        .expect("open runtime");

    let status = runtime.graph_status().await.expect("status");
    assert!(!status.is_ready());
    assert_eq!(status.current_version, None);
    let snapshot = runtime.capture_graph_snapshot();
    assert_eq!(snapshot["current_version"], Value::Null);
    assert_eq!(snapshot["graph_id"], Value::Null);
    assert_eq!(snapshot["graph_scope_id"], json!("default"));

    let revision = runtime.build().await.expect("build");
    assert_eq!(revision.version, "v1");
    let snapshot = runtime.capture_graph_snapshot();
    assert_eq!(snapshot["current_version"], json!("v1"));
    assert_eq!(snapshot["graph_id"], json!("default"));
    assert_eq!(snapshot["graph_scope_id"], json!("default"));
}

#[tokio::test]
async fn graph_falls_back_to_catalog_nodes_without_a_revision() {
    let runtime = runtime_with_catalog().await;
    let graph = runtime.graph().await;
    assert_eq!(graph.graph_id, "default");
    assert_eq!(graph.node_count(), 2);
    assert!(graph.edges.is_empty());
    assert!(graph.has_node("tool:fetch"));
}

#[tokio::test]
async fn invoke_plans_without_executing_by_default() {
    let runner = Arc::new(FakeRunner::new([]));
    let runtime = SymphonyRuntime::builder(SymphonyConfig::default())
        .with_runner(Arc::clone(&runner) as Arc<dyn CapabilityRunner>)
        .open()
        .await
        .expect("open runtime");
    for descriptor in catalog() {
        runtime
            .registry()
            .register(descriptor)
            .await
            .expect("register");
    }
    runtime.start().await.expect("start");

    let response = runtime
        .invoke(InvokeRequest::new("transform data"))
        .await
        .expect("invoke");
    assert_eq!(response.plan.status, PlanStatus::Ready);
    assert_eq!(response.plan.steps.len(), 2);
    assert_eq!(response.plan.steps[0].step_id, "tool:fetch");
    assert_eq!(response.plan.steps[1].step_id, "tool:transform");
    assert!(response.execution.is_none());
    assert!(response.graph_status.is_ready());
    assert!(runner.invoked().is_empty());
}

#[tokio::test]
async fn invoke_executes_the_plan_with_the_configured_runner() {
    let runner = FakeRunner::new([
        ("tool:fetch", json!({"data": "raw"})),
        ("tool:transform", json!({"result": "done"})),
    ]);
    let runtime = SymphonyRuntime::builder(SymphonyConfig::default())
        .with_runner(Arc::new(runner.clone()) as Arc<dyn CapabilityRunner>)
        .open()
        .await
        .expect("open runtime");
    for descriptor in catalog() {
        runtime
            .registry()
            .register(descriptor)
            .await
            .expect("register");
    }
    runtime.start().await.expect("start");

    let response = runtime
        .invoke(
            InvokeRequest::new("transform data")
                .execute()
                .with_inputs(json!({"request": "raw"})),
        )
        .await
        .expect("invoke");
    let execution = response.execution.expect("execution");
    assert!(execution.succeeded());
    assert_eq!(execution.steps.len(), 2);
    assert_eq!(execution.outputs["tool:fetch"], json!({"data": "raw"}));
    assert_eq!(
        execution.outputs["tool:transform"],
        json!({"result": "done"})
    );
    assert_eq!(
        runner.invoked(),
        vec!["tool:fetch".to_string(), "tool:transform".to_string()]
    );
    // The predecessor output is overlaid onto the successor's payload.
    assert_eq!(runner.inputs_for("tool:transform")["data"], json!("raw"));
}

#[tokio::test]
async fn invoke_without_a_runner_reports_unsupported() {
    let runtime = runtime_with_catalog().await;
    runtime.start().await.expect("start");
    let error = runtime
        .invoke(InvokeRequest::new("transform data").execute())
        .await
        .expect_err("execution without a runner must fail");
    assert_eq!(error.kind(), ErrorKind::Unsupported);
}

#[tokio::test]
async fn invoke_requires_the_running_state() {
    let runtime = runtime_with_catalog().await;
    let error = runtime
        .invoke(InvokeRequest::new("transform data"))
        .await
        .expect_err("a created runtime must not invoke");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains("runtime.state"));
}

#[tokio::test]
async fn mutations_publish_a_revision_and_report_the_changed_ids() {
    let runtime = SymphonyRuntime::open(SymphonyConfig::default())
        .await
        .expect("open runtime");
    runtime.start().await.expect("start");

    let added = runtime
        .add_capabilities(catalog(), "req-add")
        .await
        .expect("add");
    assert_eq!(added.operation, MutationOperation::Add);
    assert_eq!(added.request_id, "req-add");
    assert_eq!(added.changed, vec!["tool:fetch", "tool:transform"]);
    assert_eq!(added.added_edges, 1);
    assert_eq!(added.removed_edges, 0);
    assert_eq!(added.revision, "v2");
    assert!(!added.replayed);
    assert_eq!(runtime.registry().len(), 2);

    let updated = runtime
        .update_capabilities(
            vec![fetch().with_description("fetches anything")],
            "req-update",
        )
        .await
        .expect("update");
    assert_eq!(updated.operation, MutationOperation::Update);
    assert_eq!(updated.changed, vec!["tool:fetch"]);
    assert_eq!(updated.revision, "v3");
    assert_eq!(
        runtime
            .registry()
            .get(&CapabilityKey::new("tool", "fetch"))
            .await
            .expect("fetch")
            .description,
        "fetches anything"
    );

    let removed = runtime
        .remove_capabilities(&[CapabilityKey::new("tool", "transform")], "req-remove")
        .await
        .expect("remove");
    assert_eq!(removed.operation, MutationOperation::Delete);
    assert_eq!(removed.changed, vec!["tool:transform"]);
    assert_eq!(removed.added_edges, 0);
    assert_eq!(removed.removed_edges, 1);
    assert_eq!(removed.revision, "v4");
    assert_eq!(runtime.registry().len(), 1);

    let graph = runtime
        .read_graph(Some("v4"))
        .await
        .expect("read v4")
        .expect("v4 exists");
    assert!(!graph.has_node("tool:transform"));
    assert!(graph.edges.is_empty());
}

#[tokio::test]
async fn mutation_replays_return_the_recorded_result_without_side_effects() {
    let runtime = SymphonyRuntime::open(SymphonyConfig::default())
        .await
        .expect("open runtime");
    runtime.start().await.expect("start");

    let first = runtime
        .add_capabilities(catalog(), "req-replay")
        .await
        .expect("add");
    let second = runtime
        .add_capabilities(catalog(), "req-replay")
        .await
        .expect("replay");

    assert!(!first.replayed);
    assert!(second.replayed);
    assert_eq!(first.operation, second.operation);
    assert_eq!(first.request_id, second.request_id);
    assert_eq!(first.changed, second.changed);
    assert_eq!(first.added_edges, second.added_edges);
    assert_eq!(first.removed_edges, second.removed_edges);
    assert_eq!(first.revision, second.revision);
    assert_eq!(
        runtime
            .graph_status()
            .await
            .expect("status")
            .current_version
            .as_deref(),
        Some("v2"),
        "a replay must not publish another revision"
    );
}

#[tokio::test]
async fn mutation_request_ids_must_not_be_blank() {
    let runtime = runtime_with_catalog().await;
    runtime.start().await.expect("start");
    let error = runtime
        .add_capabilities(vec![sum()], "  ")
        .await
        .expect_err("a blank request id must be rejected");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains("request_id"));
}

#[tokio::test]
async fn update_of_an_unknown_capability_is_not_found() {
    let runtime = runtime_with_catalog().await;
    runtime.start().await.expect("start");
    let before = runtime
        .graph_status()
        .await
        .expect("status")
        .current_version;

    let error = runtime
        .update_capabilities(
            vec![CapabilityDescriptor::new("tool", "missing", "Missing")],
            "req-missing",
        )
        .await
        .expect_err("an unknown capability must not update");
    assert_eq!(error.kind(), ErrorKind::NotFound);
    assert!(error.message().contains("tool:missing"));
    assert_eq!(
        runtime
            .graph_status()
            .await
            .expect("status")
            .current_version,
        before,
        "a rejected update must not publish a revision"
    );
    assert_eq!(runtime.registry().len(), 2);
}

#[tokio::test]
async fn remove_of_an_absent_key_is_a_no_op() {
    let runtime = runtime_with_catalog().await;
    runtime.start().await.expect("start");
    let result = runtime
        .remove_capabilities(&[CapabilityKey::new("tool", "ghost")], "req-ghost")
        .await
        .expect("remove absent");
    assert!(result.changed.is_empty());
    assert_eq!(result.added_edges, 0);
    assert_eq!(result.removed_edges, 0);
    assert_eq!(result.revision, "v2");
    assert_eq!(runtime.registry().len(), 2);
}

#[tokio::test]
async fn evolution_updates_matching_edge_weights_with_an_ema() {
    let store = Arc::new(MemoryGraphArtifactStore::new());
    let runtime = SymphonyRuntime::builder(SymphonyConfig::default())
        .with_artifacts(Arc::clone(&store) as Arc<dyn GraphArtifactStore>)
        .open()
        .await
        .expect("open runtime");
    for descriptor in catalog() {
        runtime
            .registry()
            .register(descriptor)
            .await
            .expect("register");
    }

    let mut graph = CapabilityGraph::from_descriptors("default", &catalog());
    graph
        .add_edge(CapabilityEdge::new(
            "tool:fetch",
            "tool:transform",
            CAN_FEED,
            0.5,
        ))
        .expect("edge");
    let published = store.publish(&graph, None).await.expect("publish");
    assert_eq!(published.version, "v1");

    let execution = execution_of(&["tool:fetch", "tool:transform"]);
    let result = runtime.submit_evolution(&execution).await.expect("evolve");
    assert_eq!(result.accepted_edges, 1);
    assert_eq!(result.rejected_edges, 0);
    assert_eq!(result.revision, "v2");

    let evolved = runtime
        .read_graph(Some("v2"))
        .await
        .expect("read v2")
        .expect("v2 exists");
    assert_eq!(evolved.edges[0].weight, 0.75);

    // A transition with no matching edge is rejected and publishes nothing.
    let unmatched = execution_of(&["tool:fetch", "tool:missing"]);
    let result = runtime.submit_evolution(&unmatched).await.expect("evolve");
    assert_eq!(result.accepted_edges, 0);
    assert_eq!(result.rejected_edges, 1);
    assert_eq!(result.revision, "v2");

    // A single successful step has no transition at all.
    let result = runtime
        .submit_evolution(&execution_of(&["tool:fetch"]))
        .await
        .expect("evolve");
    assert_eq!(result.accepted_edges, 0);
    assert_eq!(result.rejected_edges, 0);
    assert_eq!(result.revision, "v2");
}

#[tokio::test]
async fn evolution_without_a_revision_rejects_every_transition() {
    let runtime = runtime_with_catalog().await;
    let result = runtime
        .submit_evolution(&execution_of(&["tool:fetch", "tool:transform"]))
        .await
        .expect("evolve");
    assert_eq!(result.accepted_edges, 0);
    assert_eq!(result.rejected_edges, 1);
    assert_eq!(result.revision, "");
}

#[tokio::test]
async fn semantic_search_indexes_the_catalog_snapshot() {
    let embedder: Arc<dyn Embedding> = Arc::new(HashEmbedding::new(64));
    let runtime = SymphonyRuntime::builder(SymphonyConfig::default())
        .with_semantic_index(Arc::new(SemanticIndex::new(embedder)))
        .open()
        .await
        .expect("open runtime");
    runtime.registry().register(sum()).await.expect("register");

    let hits = runtime
        .semantic_search("adds two numbers", 10)
        .await
        .expect("semantic search");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].descriptor.qualified_id(), "tool:sum");

    // Re-indexing on every search means removals are reflected too.
    runtime
        .registry()
        .remove(&CapabilityKey::new("tool", "sum"))
        .await
        .expect("remove");
    assert!(runtime
        .semantic_search("adds two numbers", 10)
        .await
        .expect("semantic search")
        .is_empty());
}

#[tokio::test]
async fn semantic_search_without_an_index_is_unsupported() {
    let runtime = runtime_with_catalog().await;
    let error = runtime
        .semantic_search("adds two numbers", 10)
        .await
        .expect_err("no semantic index is configured");
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert!(error.message().contains("semantic index"));
}

#[tokio::test]
async fn stopped_runtime_rejects_writes_but_serves_reads() {
    let runtime = runtime_with_catalog().await;
    runtime.start().await.expect("start");
    runtime.stop().await.expect("stop");
    assert_eq!(runtime.state(), RuntimeState::Stopped);

    let invoke = runtime
        .invoke(InvokeRequest::new("transform data"))
        .await
        .expect_err("a stopped runtime must not invoke");
    assert_eq!(invoke.kind(), ErrorKind::Validation);
    let add = runtime
        .add_capabilities(vec![sum()], "req-stopped")
        .await
        .expect_err("a stopped runtime must not mutate");
    assert_eq!(add.kind(), ErrorKind::Validation);
    let remove = runtime
        .remove_capabilities(&[CapabilityKey::new("tool", "fetch")], "req-stopped-2")
        .await
        .expect_err("a stopped runtime must not mutate");
    assert_eq!(remove.kind(), ErrorKind::Validation);

    assert!(runtime.graph_status().await.expect("status").is_ready());
    assert_eq!(runtime.graph().await.node_count(), 2);
    let plan = runtime
        .plan(&PlanRequest::new("transform data"))
        .await
        .expect("plan");
    assert_eq!(plan.steps.len(), 2);
    let hits = runtime
        .search(&SearchRequest::for_text("fetches raw data"))
        .await
        .expect("search");
    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].descriptor.qualified_id(), "tool:fetch");
    assert_eq!(
        runtime.capture_graph_snapshot()["current_version"],
        json!("v1")
    );
    assert_eq!(runtime.build().await.expect("build").version, "v2");
    assert_eq!(runtime.registry().len(), 2);
}

#[tokio::test]
async fn file_backed_artifacts_persist_across_runtimes() {
    let directory = tempfile::tempdir().expect("tempdir");
    let config = SymphonyConfig {
        graph_artifact_root: Some(directory.path().to_path_buf()),
        ..SymphonyConfig::default()
    };

    let first = SymphonyRuntime::open(config.clone())
        .await
        .expect("open first runtime");
    for descriptor in catalog() {
        first
            .registry()
            .register(descriptor)
            .await
            .expect("register");
    }
    first.start().await.expect("start");
    assert_eq!(
        first.graph_status().await.expect("status").current_version,
        Some("v1".to_string())
    );
    first.close().await.expect("close");
    first.close().await.expect("close again");
    drop(first);

    let second = SymphonyRuntime::open(config).await.expect("reopen runtime");
    assert_eq!(second.state(), RuntimeState::Created);
    let status = second.graph_status().await.expect("status");
    assert!(status.is_ready());
    assert_eq!(status.current_version.as_deref(), Some("v1"));
    assert_eq!(status.node_count, 2);
    assert_eq!(status.edge_count, 1);
    assert_eq!(
        second.capture_graph_snapshot()["current_version"],
        json!("v1")
    );

    let graph = second
        .read_graph(None)
        .await
        .expect("read current")
        .expect("current exists");
    assert!(graph.has_node("tool:fetch"));
    assert_eq!(graph.graph_id, "default");
}

/// Builds a plan execution with every listed step successful, in the given order.
fn execution_of(step_ids: &[&str]) -> PlanExecution {
    let steps = step_ids
        .iter()
        .map(|step_id| {
            (
                (*step_id).to_string(),
                StepOutcome {
                    step_id: (*step_id).to_string(),
                    success: true,
                    ..StepOutcome::default()
                },
            )
        })
        .collect();
    PlanExecution {
        plan_id: "plan-1".to_string(),
        graph_id: "plan-1".to_string(),
        steps,
        ..PlanExecution::default()
    }
}
