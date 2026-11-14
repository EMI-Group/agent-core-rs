//! Behaviour tests for [`SymphonyRuntime`]: lifecycle, graph build and publication,
//! planning, invocation, idempotent mutations (digests, the optional journal, and
//! source-revision compare-and-swap), evolution feedback, semantic search, and file-backed
//! revision persistence.
//!
//! Everything here is offline and deterministic: the capability runner is a local fake,
//! the semantic index embeds with `HashEmbedding`, file-backed runs use `tempfile`, and no
//! assertion depends on the wall clock or on hash-map iteration order.
#![allow(clippy::unwrap_used, clippy::expect_used)]
#![allow(clippy::float_cmp)]
#![allow(clippy::assert_is_empty)]

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use openjiuwen_retrieval::{Embedding, HashEmbedding};
use openjiuwen_symphony::artifact::{GraphArtifactStore, MemoryGraphArtifactStore};
use openjiuwen_symphony::error::{ErrorKind, Result};
use openjiuwen_symphony::graph::{CapabilityEdge, CapabilityGraph, CAN_FEED};
use openjiuwen_symphony::model::{CapabilityDescriptor, CapabilityIo, CapabilityKey};
use openjiuwen_symphony::orchestration::mutation::{MutationJournal, MutationRequest};
use openjiuwen_symphony::orchestration::PlanExecution;
use openjiuwen_symphony::orchestration::{
    CandidateConfig, CapabilityRunner, ExecutorConfig, PlanConfig, PlanRequest, PlanStatus,
    Planner, RunnerContext, SemanticScorer, StepOutcome,
};
use openjiuwen_symphony::registry::CapabilityRegistry;
use openjiuwen_symphony::runtime::{
    GraphMutationResult, InvokeRequest, MutationOperation, MutationStatus, RuntimeState,
    SymphonyConfig, SymphonyRuntime,
};
use openjiuwen_symphony::search::{SearchHit, SearchRequest, SemanticIndex};
use openjiuwen_symphony::store::{AssetStore, KvAssetStore};
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
async fn mutation_results_report_status_digest_and_serialized_shape() {
    let runtime = SymphonyRuntime::open(SymphonyConfig::default())
        .await
        .expect("open runtime");
    runtime.start().await.expect("start");

    let added = runtime
        .add_capabilities(catalog(), "req-status")
        .await
        .expect("add");
    assert_eq!(added.status, MutationStatus::Published);
    assert!(!added.replayed);
    assert_eq!(added.digest.len(), 64);
    assert!(added.source_revision.is_none());
    assert!(added.diagnostics.is_empty());

    let json = serde_json::to_value(&added).expect("serialize");
    assert_eq!(json["status"], json!("published"));
    assert_eq!(json["digest"], json!(added.digest));
    assert_eq!(json["source_revision"], Value::Null);
    assert_eq!(json["diagnostics"], json!([]));
    assert_eq!(json["operation"], json!("add"));

    let replay = runtime
        .add_capabilities(catalog(), "req-status")
        .await
        .expect("replay");
    assert!(replay.replayed);
    assert_eq!(replay.status, MutationStatus::Replayed);
    assert_eq!(replay.digest, added.digest);
    assert_eq!(replay.revision, added.revision);
    assert_eq!(replay.changed, added.changed);

    let noop = runtime
        .remove_capabilities(&[CapabilityKey::new("tool", "ghost")], "req-noop")
        .await
        .expect("remove an absent key");
    assert_eq!(noop.status, MutationStatus::Noop);
    assert!(!noop.replayed);
    assert!(noop.changed.is_empty());

    // The new fields are additive: a record written before them still deserializes.
    let legacy = json!({
        "operation": "add",
        "request_id": "req-status",
        "changed": ["tool:fetch"],
        "revision": "v2",
    });
    let restored: GraphMutationResult =
        serde_json::from_value(legacy).expect("a legacy record stays readable");
    assert_eq!(restored.status, MutationStatus::Published);
    assert_eq!(restored.request_id, "req-status");
    assert!(restored.digest.is_empty());
    assert!(!restored.replayed);
}

#[tokio::test]
async fn mutation_replay_with_a_different_batch_is_rejected() {
    let runtime = runtime_with_catalog().await;
    runtime.start().await.expect("start");
    let first = runtime
        .add_capabilities(vec![sum()], "req-conflict")
        .await
        .expect("add");
    assert_eq!(first.changed, vec!["tool:sum"]);

    // The digest covers the declared ids, so a *different id set* is another request ...
    let error = runtime
        .add_capabilities(
            vec![CapabilityDescriptor::new("tool", "other", "Other")],
            "req-conflict",
        )
        .await
        .expect_err("a reused request id with another batch must be rejected");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(
        error.message().contains("req-conflict"),
        "the error must name the request id: {}",
        error.message()
    );
    assert_eq!(
        runtime
            .registry()
            .get(&CapabilityKey::new("tool", "sum"))
            .await
            .expect("sum")
            .description,
        "adds two numbers together",
        "a rejected replay must not apply the conflicting batch"
    );
    assert!(
        !runtime
            .registry()
            .contains(&CapabilityKey::new("tool", "other"))
            .await,
        "a rejected replay must not register the conflicting descriptor"
    );
    assert_eq!(
        runtime
            .graph_status()
            .await
            .expect("status")
            .current_version
            .as_deref(),
        Some("v2"),
        "a rejected replay must not publish a revision"
    );
}

#[tokio::test]
async fn mutation_digest_covers_the_declared_ids_not_the_payload() {
    let runtime = runtime_with_catalog().await;
    runtime.start().await.expect("start");
    runtime
        .add_capabilities(vec![sum()], "req-ids-only")
        .await
        .expect("add");
    // Python's digest covers `changed_skill_ids`, so re-sending the same ids with different
    // descriptor content is a replay, not a conflict.
    let replay = runtime
        .add_capabilities(vec![sum().with_description("different")], "req-ids-only")
        .await
        .expect("the same id batch replays");
    assert!(replay.replayed);
    assert_eq!(
        runtime
            .registry()
            .get(&CapabilityKey::new("tool", "sum"))
            .await
            .expect("sum")
            .description,
        "adds two numbers together",
        "a replay must not apply the new payload"
    );
}

#[tokio::test]
async fn mutation_requests_must_match_their_payload() {
    let runtime = runtime_with_catalog().await;
    runtime.start().await.expect("start");

    let request = MutationRequest::prepare(
        MutationOperation::Add,
        &["tool:sum".to_string()],
        "req-shape",
        None,
    )
    .expect("prepare");
    let error = runtime
        .mutate(request, vec![fetch()])
        .await
        .expect_err("a request that disagrees with its payload must be rejected");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains("changed_skill_ids"));
    assert!(error.message().contains("tool:sum"));
    assert_eq!(runtime.registry().len(), 2);

    let empty = runtime
        .add_capabilities(Vec::new(), "req-empty")
        .await
        .expect_err("an empty batch must be rejected");
    assert_eq!(empty.kind(), ErrorKind::Validation);
    assert!(empty.message().contains("changed_skill_ids"));

    let blank = runtime
        .add_capabilities(vec![sum()], "   ")
        .await
        .expect_err("a blank request id must be rejected");
    assert_eq!(blank.kind(), ErrorKind::Validation);
    assert!(blank.message().contains("request_id"));
    assert_eq!(
        runtime
            .graph_status()
            .await
            .expect("status")
            .current_version
            .as_deref(),
        Some("v1"),
        "a rejected batch must not publish a revision"
    );
}

#[tokio::test]
async fn source_revision_compare_and_swap_rejects_stale_batches() {
    let runtime = runtime_with_catalog().await;
    runtime.start().await.expect("start");
    let current = runtime
        .graph_status()
        .await
        .expect("status")
        .current_version
        .expect("a revision exists");

    let request = MutationRequest::prepare(
        MutationOperation::Add,
        &["tool:sum".to_string()],
        "req-cas-ok",
        Some(&current),
    )
    .expect("prepare");
    let published = runtime
        .mutate(request, vec![sum()])
        .await
        .expect("a batch against the current revision applies");
    assert_eq!(published.status, MutationStatus::Published);
    assert_eq!(published.source_revision.as_deref(), Some(current.as_str()));
    assert_ne!(published.revision, current);

    let stale = MutationRequest::prepare(
        MutationOperation::Update,
        &["tool:sum".to_string()],
        "req-cas-stale",
        Some(&current),
    )
    .expect("prepare");
    let error = runtime
        .mutate(stale, vec![sum().with_description("stale write")])
        .await
        .expect_err("a batch against a replaced revision must be rejected");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(
        error.message().contains("source_revision"),
        "the error must name source_revision: {}",
        error.message()
    );
    assert_eq!(
        runtime
            .registry()
            .get(&CapabilityKey::new("tool", "sum"))
            .await
            .expect("sum")
            .description,
        "adds two numbers together",
        "a stale batch must not touch the catalog"
    );

    let fresh = MutationRequest::prepare(
        MutationOperation::Update,
        &["tool:sum".to_string()],
        "req-cas-fresh",
        Some(&published.revision),
    )
    .expect("prepare");
    let updated = runtime
        .mutate(fresh, vec![sum().with_description("fresh write")])
        .await
        .expect("a batch against the current revision applies");
    assert_eq!(updated.status, MutationStatus::Published);
    assert_eq!(
        runtime
            .registry()
            .get(&CapabilityKey::new("tool", "sum"))
            .await
            .expect("sum")
            .description,
        "fresh write"
    );
}

#[tokio::test]
async fn builder_with_mutation_journal_writes_the_recorded_result() {
    let directory = tempfile::tempdir().expect("tempdir");
    let path = directory.path().join("mutations.json");
    let journal = MutationJournal::open(&path).await.expect("open journal");
    let runtime = SymphonyRuntime::builder(SymphonyConfig::default())
        .with_mutation_journal(journal)
        .open()
        .await
        .expect("open runtime");
    runtime.start().await.expect("start");

    let first = runtime
        .add_capabilities(catalog(), "req-journal")
        .await
        .expect("add");
    assert_eq!(first.status, MutationStatus::Published);

    let written = MutationJournal::open(&path).await.expect("reopen journal");
    let record = written
        .lookup("req-journal")
        .await
        .expect("the request is journaled");
    assert_eq!(record.digest, first.digest);
    assert_eq!(record.revision, first.revision);
    assert_eq!(record.operation, MutationOperation::Add);
    assert!(!record.inserted_at.is_empty());
    let payload = record.result.expect("the recorded result is kept");
    let restored: GraphMutationResult = serde_json::from_value(payload).expect("decode result");
    assert_eq!(restored.request_id, "req-journal");
    assert_eq!(restored.changed, first.changed);
    assert_eq!(restored.status, MutationStatus::Published);
}

#[tokio::test]
async fn journal_backed_idempotency_survives_a_runtime_rebuild() {
    let directory = tempfile::tempdir().expect("tempdir");
    let path = directory.path().join("mutations.json");
    let config = SymphonyConfig {
        mutation_journal_path: Some(path.clone()),
        ..SymphonyConfig::default()
    };
    let registry = Arc::new(CapabilityRegistry::new());
    let artifacts = Arc::new(MemoryGraphArtifactStore::new());

    let first = SymphonyRuntime::builder(config.clone())
        .with_registry(Arc::clone(&registry))
        .with_artifacts(Arc::clone(&artifacts) as Arc<dyn GraphArtifactStore>)
        .open()
        .await
        .expect("open runtime");
    first.start().await.expect("start");
    let added = first
        .add_capabilities(catalog(), "req-durable")
        .await
        .expect("add");
    assert_eq!(added.status, MutationStatus::Published);
    assert_eq!(added.revision, "v2");
    drop(first);

    let second = SymphonyRuntime::builder(config)
        .with_registry(Arc::clone(&registry))
        .with_artifacts(Arc::clone(&artifacts) as Arc<dyn GraphArtifactStore>)
        .open()
        .await
        .expect("rebuild runtime");
    second.start().await.expect("start");
    let replay = second
        .add_capabilities(catalog(), "req-durable")
        .await
        .expect("the journal answers the replay");
    assert!(replay.replayed);
    assert_eq!(replay.status, MutationStatus::Replayed);
    assert_eq!(replay.revision, added.revision);
    assert_eq!(replay.changed, added.changed);
    assert_eq!(replay.added_edges, added.added_edges);
    assert_eq!(replay.digest, added.digest);
    assert_eq!(
        second.graph_status().await.expect("status").current_version,
        Some("v2".to_string()),
        "a journaled replay must not publish another revision"
    );

    let conflict = second
        .update_capabilities(vec![sum()], "req-durable")
        .await
        .expect_err("the recorded digest must reject another batch");
    assert_eq!(conflict.kind(), ErrorKind::Validation);
    assert!(conflict.message().contains("req-durable"));

    // A brand-new request id still applies normally.
    let fresh = second
        .add_capabilities(vec![sum()], "req-after-rebuild")
        .await
        .expect("add");
    assert_eq!(fresh.status, MutationStatus::Published);
    assert_eq!(fresh.revision, "v3");
    assert_eq!(registry.len(), 3);
    assert!(MutationJournal::open(&path)
        .await
        .expect("reopen")
        .contains("req-after-rebuild"));
}

#[tokio::test]
async fn default_config_keeps_the_mutation_ledger_in_memory() {
    let config = SymphonyConfig::default();
    assert!(config.mutation_journal_path.is_none());
    let json = serde_json::to_value(&config).expect("serialize config");
    assert_eq!(json["mutation_journal_path"], Value::Null);
    let restored: SymphonyConfig = serde_json::from_value(json).expect("deserialize config");
    assert_eq!(restored, config);

    let runtime = runtime_with_catalog().await;
    runtime.start().await.expect("start");
    let first = runtime
        .add_capabilities(vec![sum()], "req-memory")
        .await
        .expect("add");
    let replay = runtime
        .add_capabilities(vec![sum()], "req-memory")
        .await
        .expect("replay");
    assert!(!first.replayed);
    assert!(replay.replayed);
    assert_eq!(replay.revision, first.revision);
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

/// A [`HashEmbedding`] that counts document-embedding invocations, so a test can observe
/// whether the semantic index was re-embedded or only the query was.
struct CountingEmbedding {
    /// The deterministic embedder doing the actual work.
    inner: HashEmbedding,
    /// How many times `embed_documents` was called.
    document_calls: AtomicUsize,
}

impl CountingEmbedding {
    /// Creates a counting embedder over a [`HashEmbedding`] of `dimension` components.
    fn new(dimension: usize) -> Self {
        Self {
            inner: HashEmbedding::new(dimension),
            document_calls: AtomicUsize::new(0),
        }
    }

    /// How many document-embedding batches were requested so far.
    fn document_calls(&self) -> usize {
        self.document_calls.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl Embedding for CountingEmbedding {
    fn model_name(&self) -> &str {
        self.inner.model_name()
    }

    fn dimension_hint(&self) -> Option<usize> {
        self.inner.dimension_hint()
    }

    async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        self.inner.embed_query(text).await
    }

    async fn embed_documents(
        &self,
        texts: &[String],
        batch_size: Option<usize>,
    ) -> Result<Vec<Vec<f32>>> {
        self.document_calls.fetch_add(1, Ordering::SeqCst);
        self.inner.embed_documents(texts, batch_size).await
    }
}

/// The qualified ids of `hits`, in hit order.
fn hit_ids(hits: &[SearchHit]) -> Vec<String> {
    hits.iter()
        .map(|hit| hit.descriptor.qualified_id())
        .collect()
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
async fn semantic_search_reembeds_only_when_the_catalog_changed() {
    let embedder = Arc::new(CountingEmbedding::new(64));
    let runtime = SymphonyRuntime::builder(SymphonyConfig::default())
        .with_semantic_index(Arc::new(SemanticIndex::new(embedder.clone())))
        .open()
        .await
        .expect("open runtime");
    runtime.registry().register(sum()).await.expect("register");

    let first = runtime
        .semantic_search("adds two numbers", 10)
        .await
        .expect("first semantic search");
    let second = runtime
        .semantic_search("adds two numbers", 10)
        .await
        .expect("second semantic search");

    assert_eq!(hit_ids(&first), vec!["tool:sum"]);
    assert_eq!(hit_ids(&second), vec!["tool:sum"]);
    // The catalog did not change, so only the query was embedded once.
    assert_eq!(embedder.document_calls(), 1);
}

#[tokio::test]
async fn semantic_search_reflects_added_capabilities() {
    let embedder = Arc::new(CountingEmbedding::new(64));
    let runtime = SymphonyRuntime::builder(SymphonyConfig::default())
        .with_semantic_index(Arc::new(SemanticIndex::new(embedder.clone())))
        .open()
        .await
        .expect("open runtime");
    runtime.registry().register(sum()).await.expect("register");
    runtime.start().await.expect("start");

    let before = runtime
        .semantic_search("fetches raw data", 0)
        .await
        .expect("semantic search before the add");
    assert_eq!(hit_ids(&before), vec!["tool:sum"]);

    runtime
        .add_capabilities(vec![fetch()], "req-semantic-add")
        .await
        .expect("add capability");

    let after = runtime
        .semantic_search("fetches raw data", 0)
        .await
        .expect("semantic search after the add");
    assert!(hit_ids(&after).contains(&"tool:fetch".to_string()));
    assert_eq!(embedder.document_calls(), 2);
}

#[tokio::test]
async fn semantic_search_reflects_removed_capabilities() {
    let embedder = Arc::new(CountingEmbedding::new(64));
    let runtime = SymphonyRuntime::builder(SymphonyConfig::default())
        .with_semantic_index(Arc::new(SemanticIndex::new(embedder.clone())))
        .open()
        .await
        .expect("open runtime");
    runtime.registry().register(sum()).await.expect("register");
    runtime
        .registry()
        .register(fetch())
        .await
        .expect("register");
    runtime.start().await.expect("start");
    let before = runtime
        .semantic_search("fetches raw data", 0)
        .await
        .expect("semantic search");
    assert!(hit_ids(&before).contains(&"tool:sum".to_string()));

    runtime
        .remove_capabilities(&[CapabilityKey::new("tool", "sum")], "req-semantic-remove")
        .await
        .expect("remove capability");

    let after = runtime
        .semantic_search("adds two numbers", 0)
        .await
        .expect("semantic search after the removal");
    assert_eq!(hit_ids(&after), vec!["tool:fetch"]);
    // The removal invalidated the index, so the remaining catalog was re-embedded once.
    assert_eq!(embedder.document_calls(), 2);
}

#[tokio::test]
async fn reindex_semantics_reembeds_the_whole_catalog_repeatedly() {
    let embedder = Arc::new(CountingEmbedding::new(64));
    let runtime = SymphonyRuntime::builder(SymphonyConfig::default())
        .with_semantic_index(Arc::new(SemanticIndex::new(embedder.clone())))
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

    assert_eq!(runtime.reindex_semantics().await.expect("reindex"), 2);
    assert_eq!(runtime.reindex_semantics().await.expect("reindex again"), 2);
    assert_eq!(embedder.document_calls(), 2);
    let hits = runtime
        .semantic_search("fetches raw data", 0)
        .await
        .expect("semantic search after reindex");
    assert_eq!(hits.len(), 2);
}

#[tokio::test]
async fn semantic_reindex_without_an_index_is_unsupported() {
    let runtime = runtime_with_catalog().await;
    let error = runtime
        .reindex_semantics()
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

/// A [`SemanticScorer`] that scores one capability `1.0` and every other `0.0`.
struct FixedScorer {
    /// The qualified id the scorer prefers.
    preferred: &'static str,
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
                let qualified_id = capability.qualified_id();
                let score = if qualified_id == self.preferred {
                    1.0
                } else {
                    0.0
                };
                (qualified_id, score)
            })
            .collect())
    }
}

#[tokio::test]
async fn builder_with_registry_adopts_the_supplied_catalog() {
    let registry = Arc::new(CapabilityRegistry::new());
    for descriptor in catalog() {
        registry.register(descriptor).await.expect("register");
    }

    let runtime = SymphonyRuntime::builder(SymphonyConfig::default())
        .with_registry(Arc::clone(&registry))
        .open()
        .await
        .expect("open runtime");

    assert!(
        Arc::ptr_eq(runtime.registry(), &registry),
        "the runtime must use the caller's registry, not a copy"
    );
    assert_eq!(runtime.registry().len(), 2);

    let revision = runtime.build().await.expect("build");
    assert_eq!(revision.node_count, 2);
    assert_eq!(revision.edge_count, 1);
    let graph = runtime
        .read_graph(None)
        .await
        .expect("read current")
        .expect("current exists");
    assert!(graph.has_node("tool:fetch"));
    assert!(graph.has_node("tool:transform"));
}

/// Opens a runtime over the `fetch`/`transform`/`sum` catalog with `planner` when given.
async fn runtime_with_planner(planner: Option<Planner>) -> SymphonyRuntime {
    let builder = SymphonyRuntime::builder(SymphonyConfig::default());
    let builder = match planner {
        Some(planner) => builder.with_planner(planner),
        None => builder,
    };
    let runtime = builder.open().await.expect("open runtime");
    for descriptor in catalog().into_iter().chain(std::iter::once(sum())) {
        runtime
            .registry()
            .register(descriptor)
            .await
            .expect("register");
    }
    runtime
}

#[tokio::test]
async fn builder_with_planner_honours_the_supplied_scorer() {
    // The default lexical scorer matches `fetch`/`transform` on this query.
    let default_runtime = runtime_with_planner(None).await;
    let default_plan = default_runtime
        .plan(&PlanRequest::new("fetches raw data"))
        .await
        .expect("plan");
    assert_eq!(default_plan.steps[0].step_id, "tool:fetch");

    // The injected planner swaps in a scorer that prefers `tool:sum` only.
    let scorer: Arc<dyn SemanticScorer> = Arc::new(FixedScorer {
        preferred: "tool:sum",
    });
    let runtime =
        runtime_with_planner(Some(Planner::with_scorer(PlanConfig::default(), scorer))).await;
    let plan = runtime
        .plan(&PlanRequest::new("fetches raw data"))
        .await
        .expect("plan");
    assert_eq!(plan.steps.len(), 1);
    assert_eq!(plan.steps[0].step_id, "tool:sum");
    assert_eq!(plan.status, PlanStatus::NeedsInput);
}

#[tokio::test]
async fn builder_with_asset_store_hydrates_and_writes_through() {
    let store: Arc<dyn AssetStore> = Arc::new(KvAssetStore::in_memory());
    store.put(&fetch()).await.expect("seed store");

    let runtime = SymphonyRuntime::builder(SymphonyConfig::default())
        .with_asset_store(Arc::clone(&store))
        .open()
        .await
        .expect("open runtime");
    assert_eq!(runtime.registry().len(), 1);
    assert!(
        runtime
            .registry()
            .contains(&CapabilityKey::new("tool", "fetch"))
            .await
    );

    let plan = runtime
        .plan(&PlanRequest::new("fetch data"))
        .await
        .expect("plan");
    assert_eq!(plan.steps.len(), 1);
    assert_eq!(plan.steps[0].step_id, "tool:fetch");

    runtime.start().await.expect("start");
    runtime
        .add_capabilities(vec![sum()], "req-persist")
        .await
        .expect("add");
    assert!(store
        .get(&CapabilityKey::new("tool", "sum"))
        .await
        .expect("read through the store")
        .is_some());
    drop(runtime);

    let reopened = SymphonyRuntime::builder(SymphonyConfig::default())
        .with_asset_store(store)
        .open()
        .await
        .expect("reopen runtime");
    assert_eq!(reopened.registry().len(), 2);
    assert!(
        reopened
            .registry()
            .contains(&CapabilityKey::new("tool", "sum"))
            .await
    );
    let plan = reopened
        .plan(&PlanRequest::new("adds two numbers"))
        .await
        .expect("plan");
    assert_eq!(plan.steps[0].step_id, "tool:sum");
}

#[tokio::test]
async fn file_artifacts_and_asset_store_persist_together() {
    let directory = tempfile::tempdir().expect("tempdir");
    let config = SymphonyConfig {
        graph_artifact_root: Some(directory.path().to_path_buf()),
        ..SymphonyConfig::default()
    };
    let store: Arc<dyn AssetStore> = Arc::new(KvAssetStore::in_memory());

    let first = SymphonyRuntime::builder(config.clone())
        .with_asset_store(Arc::clone(&store))
        .open()
        .await
        .expect("open first runtime");
    first.start().await.expect("start");
    let added = first
        .add_capabilities(catalog(), "req-artifacts")
        .await
        .expect("add");
    assert_eq!(added.revision, "v2");
    first.close().await.expect("close");
    drop(first);

    let second = SymphonyRuntime::builder(config)
        .with_asset_store(store)
        .open()
        .await
        .expect("reopen runtime");
    assert_eq!(second.registry().len(), 2);
    let status = second.graph_status().await.expect("status");
    assert!(status.is_ready());
    assert_eq!(status.current_version.as_deref(), Some("v2"));
    assert_eq!(status.node_count, 2);
    assert_eq!(status.edge_count, 1);
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
