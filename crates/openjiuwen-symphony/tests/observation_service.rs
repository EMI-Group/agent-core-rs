//! Integration tests for `GraphObservationService`: qualification gating, duplicate
//! receipts, the runtime-only threshold, bounded weights, the high-water cursor, the safe
//! fallback, rebinding across static revisions, merged-revision pinning, and the sanitised
//! on-disk artifacts.
//!
//! Everything goes through the public `observation` API and is offline and deterministic:
//! `tempfile` roots, explicit `observed_at` strings, in-memory static artifacts, no sleeps,
//! and no assertion that depends on the wall clock (`generated_at` is metadata only).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;
use std::ops::RangeInclusive;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use openjiuwen_symphony::artifact::{GraphArtifactStore, MemoryGraphArtifactStore};
use openjiuwen_symphony::error::ErrorKind;
use openjiuwen_symphony::graph::{CapabilityEdge, CapabilityGraph, CapabilityNode, CAN_FEED};
use openjiuwen_symphony::observation::contracts::{
    EvidenceStrength, EvolutionEdgeMetadata, EvolutionGraph, EvolutionGraphEdge,
    EvolutionGraphNode, FailureDomain, GraphEvolutionInput, GraphSnapshot, GraphSnapshotRef,
    ObservationReceipt, ObservationStatus, TaskEvidence, TaskOutcome, TaskOutcomeLabel,
    TraceEvidence, CAPABILITY_IDENTITY_MISMATCH, MERGED_OBSERVATION_REVISION_MISMATCH,
    MERGED_STATIC_REVISION_MISMATCH, NO_QUALIFIED_EDGE_OUTCOME, OUTCOME_NOT_STRONG, SAFE_FALLBACK,
    TRACE_QUALITY_FLAGS, TRUNCATED_TRACE, UNKNOWN_MERGED_REVISION, UNKNOWN_STATIC_REVISION,
};
use openjiuwen_symphony::observation::identity::{
    normalize_capability_id, private_hash, EdgeIdentity, StaticGraphIndex,
};
use openjiuwen_symphony::observation::service::{
    GraphObservationService, MIN_RUNTIME_ONLY_SUCCESS_SESSIONS,
};
use openjiuwen_symphony::observation::store::{EvidenceStore, RevisionStore};
use openjiuwen_symphony::redaction::REDACTED;
use serde_json::{json, Map, Value};
use tempfile::TempDir;

/// A service over a fresh artifact root whose static graph is published as `v1`.
struct Harness {
    /// Keeps the artifact root alive for the duration of a test.
    root: TempDir,
    /// The static revision source, republished by the rebinding scenario.
    artifacts: Arc<dyn GraphArtifactStore>,
    /// The service under test.
    service: GraphObservationService,
}

impl Harness {
    /// The evidence log rebuilt over the harness's artifact root.
    fn evidence(&self) -> EvidenceStore {
        EvidenceStore::new(self.root.path())
    }

    /// The scope directory `graph_scope_id` maps to below the artifact root.
    fn scope_root(&self, graph_scope_id: &str) -> PathBuf {
        RevisionStore::new(self.root.path()).scope_root(graph_scope_id)
    }

    /// A second service over the same artifact root and static store.
    fn reopen(&self) -> GraphObservationService {
        GraphObservationService::new(Arc::clone(&self.artifacts), self.root.path())
    }
}

/// Builds a harness with the two-node static graph published as `v1`.
async fn harness() -> Harness {
    let root = tempfile::tempdir().expect("a temporary artifact root");
    let artifacts: Arc<dyn GraphArtifactStore> = Arc::new(MemoryGraphArtifactStore::new());
    artifacts
        .publish(&static_graph(), Some("v1"))
        .await
        .expect("publishing the static `v1` graph");
    let service = GraphObservationService::new(Arc::clone(&artifacts), root.path());
    Harness {
        root,
        artifacts,
        service,
    }
}

/// A two-capability static graph with one `can_feed` edge from `skill:a` to `skill:b`.
fn static_graph() -> CapabilityGraph {
    let mut graph = CapabilityGraph::new("observation-service-test");
    graph.upsert_node(node("skill:a", "hash-skill:a"));
    graph.upsert_node(node("skill:b", "hash-skill:b"));
    graph
        .add_edge(CapabilityEdge::new("skill:a", "skill:b", CAN_FEED, 1.0))
        .expect("a `can_feed` edge between the two nodes");
    graph
}

/// The same two nodes and edge, with `skill:b`'s content hash moved.
fn static_graph_with_rehashed_target() -> CapabilityGraph {
    let mut graph = CapabilityGraph::new("observation-service-test");
    graph.upsert_node(node("skill:a", "hash-skill:a"));
    graph.upsert_node(node("skill:b", "hash-skill:b-v2"));
    graph
        .add_edge(CapabilityEdge::new("skill:a", "skill:b", CAN_FEED, 1.0))
        .expect("a `can_feed` edge between the two nodes");
    graph
}

/// The same two nodes without the `can_feed` edge.
fn static_graph_without_edges() -> CapabilityGraph {
    let mut graph = CapabilityGraph::new("observation-service-test");
    graph.upsert_node(node("skill:a", "hash-skill:a"));
    graph.upsert_node(node("skill:b", "hash-skill:b"));
    graph
}

/// A single-node static graph, i.e. the observed edge's target disappears.
fn static_graph_without_target() -> CapabilityGraph {
    let mut graph = CapabilityGraph::new("observation-service-test");
    graph.upsert_node(node("skill:a", "hash-skill:a"));
    graph
}

/// One static node carrying a capability type and an explicit content hash.
fn node(id: &str, content_hash: &str) -> CapabilityNode {
    let mut node = CapabilityNode::new(id, id, "skill");
    node.metadata
        .insert("capability_type".to_string(), json!("skill"));
    node.metadata
        .insert("content_hash".to_string(), json!(content_hash));
    node
}

/// One execution node carrying the metadata the contract allow-lists.
fn execution_node() -> EvolutionGraphNode {
    let mut metadata = BTreeMap::new();
    metadata.insert("capability_type".to_string(), json!("skill"));
    metadata.insert("version".to_string(), json!("1.0.0"));
    EvolutionGraphNode {
        label: "skill".to_string(),
        metadata,
    }
}

/// An execution graph over `node_ids` with `edges`.
fn execution_graph_with(node_ids: &[&str], edges: Vec<EvolutionGraphEdge>) -> EvolutionGraph {
    let nodes = node_ids
        .iter()
        .map(|id| ((*id).to_string(), execution_node()))
        .collect();
    EvolutionGraph {
        graph_type: "execution_graph".to_string(),
        nodes,
        edges,
        ..EvolutionGraph::default()
    }
}

/// One observed transition with its optional edge-level outcome.
fn execution_edge(
    source: &str,
    target: &str,
    success: Option<bool>,
    failure_domain: Option<FailureDomain>,
) -> EvolutionGraphEdge {
    EvolutionGraphEdge {
        source_id: source.to_string(),
        target_id: target.to_string(),
        relation_type: CAN_FEED.to_string(),
        metadata: EvolutionEdgeMetadata {
            success,
            failure_domain,
        },
    }
}

/// The successful transition along the static edge.
fn success_edges() -> Vec<EvolutionGraphEdge> {
    vec![execution_edge("skill:a", "skill:b", Some(true), None)]
}

/// A valid input that credits the static edge as a strong success.
fn success_input(evidence_id: &str, scope: &str, session_id: &str) -> GraphEvolutionInput {
    GraphEvolutionInput {
        evidence_id: evidence_id.to_string(),
        graph_scope_id: scope.to_string(),
        observed_at: "2026-03-01T00:00:00Z".to_string(),
        graph_snapshot: GraphSnapshotRef {
            static_revision: "v1".to_string(),
            observation_revision: "observation-seed".to_string(),
            merged_revision: None,
        },
        trace: TraceEvidence {
            trajectory_id: format!("trajectory-{evidence_id}"),
            session_id: session_id.to_string(),
            capture_mode: "agent".to_string(),
            ..TraceEvidence::default()
        },
        task: TaskEvidence {
            outcome: TaskOutcome {
                label: TaskOutcomeLabel::VerifiedSuccess,
                evidence_strength: EvidenceStrength::Strong,
                failure_domain: None,
            },
            ..TaskEvidence::default()
        },
        execution_graph: execution_graph_with(&["skill:a", "skill:b"], success_edges()),
        ..GraphEvolutionInput::default()
    }
}

/// A valid input whose edge and task both carry a verified failure in `domain`.
fn failure_input(
    evidence_id: &str,
    scope: &str,
    session_id: &str,
    source: &str,
    target: &str,
    domain: FailureDomain,
) -> GraphEvolutionInput {
    let mut input = success_input(evidence_id, scope, session_id);
    input.task.outcome = TaskOutcome {
        label: TaskOutcomeLabel::VerifiedFailure,
        evidence_strength: EvidenceStrength::Strong,
        failure_domain: Some(domain),
    };
    input.execution_graph = execution_graph_with(
        &["skill:a", "skill:b"],
        vec![execution_edge(source, target, Some(false), Some(domain))],
    );
    input
}

/// An input crediting the reverse transition, which the static graph does not contain.
fn runtime_only_input(evidence_id: &str, scope: &str, session_id: &str) -> GraphEvolutionInput {
    let mut input = success_input(evidence_id, scope, session_id);
    input.execution_graph = execution_graph_with(
        &["skill:a", "skill:b"],
        vec![execution_edge("skill:b", "skill:a", Some(true), None)],
    );
    input
}

/// The overlay edge records of a snapshot.
fn overlay_edges(snapshot: &GraphSnapshot) -> &Map<String, Value> {
    snapshot.overlay["edges"]
        .as_object()
        .expect("the overlay carries an `edges` object")
}

/// The single overlay edge record of a snapshot.
fn only_edge(snapshot: &GraphSnapshot) -> &Map<String, Value> {
    let edges = overlay_edges(snapshot);
    assert_eq!(
        edges.len(),
        1,
        "expected exactly one overlay edge: {edges:?}"
    );
    edges
        .values()
        .next()
        .expect("one edge record")
        .as_object()
        .expect("an edge record object")
}

/// The point-edge identity the service derives for `source -> target` on static `v1`.
fn edge_identity(source: &str, target: &str) -> String {
    let index = StaticGraphIndex::from_graph("v1", &static_graph());
    let source_id = normalize_capability_id(source);
    let target_id = normalize_capability_id(target);
    let source_hash = index
        .content_hash_by_id
        .get(&source_id)
        .expect("a source content hash")
        .clone();
    let target_hash = index
        .content_hash_by_id
        .get(&target_id)
        .expect("a target content hash")
        .clone();
    EdgeIdentity::new(source_id, target_id, CAN_FEED, source_hash, target_hash).identity_hash()
}

/// The `runtime_weight` of `identity`, defaulting to the neutral `1.0` when unobserved.
fn weight_of(snapshot: &GraphSnapshot, identity: &str) -> f64 {
    overlay_edges(snapshot)
        .get(identity)
        .and_then(|record| record["runtime_weight"].as_f64())
        .unwrap_or(1.0)
}

/// Asserts a weight to within floating-point noise.
fn assert_weight(record: &Map<String, Value>, expected: f64) {
    let weight = record["runtime_weight"].as_f64().expect("a runtime weight");
    assert!(
        (weight - expected).abs() < 1e-9,
        "expected weight {expected}, got {weight}"
    );
}

/// Asserts `actual` equals `expected` to within floating-point noise.
fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-9,
        "expected {expected}, got {actual}"
    );
}

/// Submits `input` and asserts the receipt is `AuditOnly` with exactly `reason`.
async fn assert_audit_only(
    service: &GraphObservationService,
    input: &GraphEvolutionInput,
    reason: &str,
) -> ObservationReceipt {
    let receipt = service.submit(input).await.expect("an audited submission");
    assert_eq!(
        receipt.status,
        ObservationStatus::AuditOnly,
        "expected an audit-only receipt, got {:?} with reason `{}`",
        receipt.status,
        receipt.reason
    );
    assert_eq!(receipt.reason, reason);
    receipt
}

/// Asserts the current snapshot carries no observation for the static edge.
async fn assert_no_contribution(service: &GraphObservationService) {
    let snapshot = service
        .get_snapshot("default", None)
        .await
        .expect("the current snapshot");
    assert!(
        overlay_edges(&snapshot).is_empty(),
        "an audit-only row must publish no overlay edges: {:?}",
        overlay_edges(&snapshot)
    );
    assert_close(
        weight_of(&snapshot, &edge_identity("skill:a", "skill:b")),
        1.0,
    );
}

/// Submits one successful observation per index in `range`, each in its own session.
async fn submit_successes(harness: &Harness, range: RangeInclusive<usize>) {
    for index in range {
        let input = success_input(
            &format!("success-{index}"),
            "default",
            &format!("session-success-{index}"),
        );
        harness
            .service
            .submit(&input)
            .await
            .expect("a successful observation");
    }
}

/// Submits one orchestration failure per index in `range`, each in its own session.
async fn submit_failures(harness: &Harness, range: RangeInclusive<usize>) {
    for index in range {
        let input = failure_input(
            &format!("failure-{index}"),
            "default",
            &format!("session-failure-{index}"),
            "skill:a",
            "skill:b",
            FailureDomain::Orchestration,
        );
        harness
            .service
            .submit(&input)
            .await
            .expect("a failing observation");
    }
}

/// Reads a JSON document from disk.
fn read_json(path: &Path) -> Value {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("reading {}: {error}", path.display()));
    serde_json::from_str(&text)
        .unwrap_or_else(|error| panic!("decoding {}: {error}", path.display()))
}

/// Asserts `text` is the 64-character lowercase hex digest of `raw`.
fn assert_private_hash(text: &str, raw: &str) {
    assert_eq!(
        text.len(),
        64,
        "expected a 64-character digest, got `{text}`"
    );
    assert!(
        text.chars()
            .all(|character| character.is_ascii_hexdigit() && !character.is_ascii_uppercase()),
        "expected lowercase hex, got `{text}`"
    );
    assert_eq!(text, private_hash(raw));
}

#[tokio::test]
async fn truncated_trace_is_audit_only() {
    let harness = harness().await;
    let mut input = success_input("e1", "default", "session-1");
    input.trace.truncated = true;

    let receipt = assert_audit_only(&harness.service, &input, TRUNCATED_TRACE).await;
    assert_eq!(receipt.sequence, 1);
    assert_no_contribution(&harness.service).await;
}

#[tokio::test]
async fn trace_quality_flags_are_audit_only() {
    let harness = harness().await;
    let mut input = success_input("e1", "default", "session-1");
    input.trace.quality_flags = vec!["missing_spans".to_string()];

    assert_audit_only(&harness.service, &input, TRACE_QUALITY_FLAGS).await;
    assert_no_contribution(&harness.service).await;
}

#[tokio::test]
async fn weak_outcome_evidence_is_audit_only() {
    let harness = harness().await;
    let mut input = success_input("e1", "default", "session-1");
    input.task.outcome.evidence_strength = EvidenceStrength::Weak;

    assert_audit_only(&harness.service, &input, OUTCOME_NOT_STRONG).await;
    assert_no_contribution(&harness.service).await;
}

#[tokio::test]
async fn unknown_capability_ids_are_audit_only() {
    let harness = harness().await;
    let mut input = success_input("e1", "default", "session-1");
    input.execution_graph =
        execution_graph_with(&["skill:a", "skill:b", "skill:c"], success_edges());

    assert_audit_only(&harness.service, &input, CAPABILITY_IDENTITY_MISMATCH).await;
    assert_no_contribution(&harness.service).await;
}

#[tokio::test]
async fn unknown_static_revision_is_audit_only() {
    let harness = harness().await;
    let mut input = success_input("e1", "default", "session-1");
    input.graph_snapshot.static_revision = "v9".to_string();

    let receipt = assert_audit_only(&harness.service, &input, UNKNOWN_STATIC_REVISION).await;
    assert_eq!(receipt.sequence, 1);
    assert_no_contribution(&harness.service).await;
}

#[tokio::test]
async fn unknown_merged_revision_is_audit_only() {
    let harness = harness().await;
    let mut input = success_input("e1", "default", "session-1");
    input.graph_snapshot.merged_revision = Some("merged-missing".to_string());

    assert_audit_only(&harness.service, &input, UNKNOWN_MERGED_REVISION).await;
    assert_no_contribution(&harness.service).await;
}

#[tokio::test]
async fn pinned_merged_revision_from_another_static_revision_is_audit_only() {
    let harness = harness().await;
    let seed = harness
        .service
        .reproject("default")
        .await
        .expect("an empty seed projection");
    assert_eq!(seed.static_revision, "v1");
    harness
        .artifacts
        .publish(&static_graph(), Some("v2"))
        .await
        .expect("publishing `v2`");

    let mut input = success_input("e1", "default", "session-1");
    input.graph_snapshot.static_revision = "v2".to_string();
    input.graph_snapshot.merged_revision = Some(seed.merged_revision.clone());

    assert_audit_only(&harness.service, &input, MERGED_STATIC_REVISION_MISMATCH).await;
    assert_no_contribution(&harness.service).await;
}

#[tokio::test]
async fn pinned_merged_revision_from_another_observation_is_audit_only() {
    let harness = harness().await;
    let seed = harness
        .service
        .reproject("default")
        .await
        .expect("an empty seed projection");
    assert_ne!(seed.observation_revision, "observation-other");

    let mut input = success_input("e1", "default", "session-1");
    input.graph_snapshot.merged_revision = Some(seed.merged_revision.clone());
    input.graph_snapshot.observation_revision = "observation-other".to_string();

    assert_audit_only(
        &harness.service,
        &input,
        MERGED_OBSERVATION_REVISION_MISMATCH,
    )
    .await;
    assert_no_contribution(&harness.service).await;
}

#[tokio::test]
async fn edge_without_an_outcome_is_audit_only() {
    let harness = harness().await;
    let mut input = success_input("e1", "default", "session-1");
    input.execution_graph = execution_graph_with(
        &["skill:a", "skill:b"],
        vec![execution_edge("skill:a", "skill:b", None, None)],
    );

    assert_audit_only(&harness.service, &input, NO_QUALIFIED_EDGE_OUTCOME).await;
    assert_no_contribution(&harness.service).await;
}

#[tokio::test]
async fn duplicate_submissions_are_idempotent() {
    let harness = harness().await;
    let first = harness
        .service
        .submit(&success_input("e1", "default", "session-1"))
        .await
        .expect("the first submission");
    assert_eq!(first.status, ObservationStatus::Accepted);
    let before = harness
        .service
        .get_snapshot("default", None)
        .await
        .expect("the first snapshot");

    let duplicate = harness
        .service
        .submit(&success_input("e1", "default", "session-1"))
        .await
        .expect("the replayed submission");
    assert_eq!(duplicate.status, ObservationStatus::Duplicate);
    assert_eq!(duplicate.sequence, first.sequence);
    assert_eq!(duplicate.reason, first.reason);

    let after = harness
        .service
        .get_snapshot("default", None)
        .await
        .expect("the snapshot after the replay");
    assert_eq!(after.observation_revision, before.observation_revision);
    assert_eq!(after.merged_revision, before.merged_revision);
    assert_eq!(
        harness
            .evidence()
            .rows()
            .await
            .expect("the evidence log")
            .len(),
        1
    );

    let mut changed = success_input("e1", "default", "session-2");
    changed.observed_at = "2026-03-02T00:00:00Z".to_string();
    let error = harness
        .service
        .submit(&changed)
        .await
        .expect_err("a reused evidence id with different content is rejected");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains("evidence_id"));
    assert_eq!(
        harness
            .evidence()
            .rows()
            .await
            .expect("the evidence log")
            .len(),
        1
    );
}

#[tokio::test]
async fn runtime_only_edges_need_two_failure_free_success_sessions() {
    let harness = harness().await;
    let identity = edge_identity("skill:b", "skill:a");

    harness
        .service
        .submit(&runtime_only_input("e1", "default", "session-1"))
        .await
        .expect("the first runtime-only success");
    let first = harness
        .service
        .get_snapshot("default", None)
        .await
        .expect("the first snapshot");
    let edge = only_edge(&first);
    assert_eq!(edge["binding_status"], json!("insufficient_evidence"));
    assert_eq!(edge["runtime_only"], json!(false));
    assert_eq!(edge["success_count"], json!(1));
    assert_close(weight_of(&first, &identity), 1.05);

    let threshold = MIN_RUNTIME_ONLY_SUCCESS_SESSIONS;
    assert!(threshold >= 2, "the threshold must need distinct sessions");
    for index in 2..=threshold {
        harness
            .service
            .submit(&runtime_only_input(
                &format!("e{index}"),
                "default",
                &format!("session-{index}"),
            ))
            .await
            .expect("another runtime-only success");
    }
    let promoted = harness
        .service
        .get_snapshot("default", None)
        .await
        .expect("the promoted snapshot");
    let edge = only_edge(&promoted);
    assert_eq!(edge["binding_status"], json!("active_runtime_only"));
    assert_eq!(edge["runtime_only"], json!(true));
    assert_eq!(edge["success_count"], json!(threshold));
    assert_eq!(edge["attempt_count"], json!(threshold));
    let expected_weight = 1.0 + 0.05 * f64::from(u32::try_from(threshold).expect("a small bound"));
    assert_close(weight_of(&promoted, &identity), expected_weight);

    // A single failing session withdraws the runtime-only evidence.
    harness
        .service
        .submit(&failure_input(
            "failure-1",
            "default",
            "session-failure-1",
            "skill:b",
            "skill:a",
            FailureDomain::Orchestration,
        ))
        .await
        .expect("a runtime-only failure");
    let demoted = harness
        .service
        .get_snapshot("default", None)
        .await
        .expect("the demoted snapshot");
    let edge = only_edge(&demoted);
    assert_eq!(edge["binding_status"], json!("insufficient_evidence"));
    assert_eq!(edge["runtime_only"], json!(false));
    assert_eq!(edge["failure_count"], json!(1));
    assert_close(weight_of(&demoted, &identity), 1.05);

    // Later successes cannot outvote the recorded failure.
    harness
        .service
        .submit(&runtime_only_input("e-extra", "default", "session-extra"))
        .await
        .expect("one more runtime-only success");
    let still = harness
        .service
        .get_snapshot("default", None)
        .await
        .expect("the final snapshot");
    let edge = only_edge(&still);
    assert_eq!(edge["binding_status"], json!("insufficient_evidence"));
    assert_eq!(edge["runtime_only"], json!(false));
    assert_eq!(edge["success_count"], json!(threshold + 1));
}

#[tokio::test]
async fn weights_are_symmetric_bounded_and_session_deduplicated() {
    // Two evidences from one session count once.
    let dedupe = harness().await;
    dedupe
        .service
        .submit(&success_input("d1", "default", "session-shared"))
        .await
        .expect("the first shared-session evidence");
    dedupe
        .service
        .submit(&success_input("d2", "default", "session-shared"))
        .await
        .expect("the second shared-session evidence");
    let snapshot = dedupe
        .service
        .get_snapshot("default", None)
        .await
        .expect("the deduplicated snapshot");
    let edge = only_edge(&snapshot);
    assert_eq!(edge["success_count"], json!(1));
    assert_eq!(edge["failure_count"], json!(0));
    assert_eq!(edge["attempt_count"], json!(1));
    assert_weight(edge, 1.05);

    let harness = harness().await;
    submit_successes(&harness, 1..=1).await;
    let snapshot = harness
        .service
        .get_snapshot("default", None)
        .await
        .expect("one success");
    assert_weight(only_edge(&snapshot), 1.05);

    submit_failures(&harness, 1..=1).await;
    let snapshot = harness
        .service
        .get_snapshot("default", None)
        .await
        .expect("one success and one failure");
    assert_weight(only_edge(&snapshot), 1.0);

    submit_successes(&harness, 2..=20).await;
    let snapshot = harness
        .service
        .get_snapshot("default", None)
        .await
        .expect("twenty successes and one failure");
    assert_eq!(only_edge(&snapshot)["success_count"], json!(20));
    assert_weight(only_edge(&snapshot), 1.95);

    submit_failures(&harness, 2..=20).await;
    let snapshot = harness
        .service
        .get_snapshot("default", None)
        .await
        .expect("twenty successes and twenty failures");
    assert_eq!(only_edge(&snapshot)["failure_count"], json!(20));
    assert_weight(only_edge(&snapshot), 1.0);

    submit_successes(&harness, 21..=40).await;
    let snapshot = harness
        .service
        .get_snapshot("default", None)
        .await
        .expect("the upper bound");
    assert_eq!(only_edge(&snapshot)["success_count"], json!(40));
    assert_weight(only_edge(&snapshot), 2.0);

    submit_successes(&harness, 41..=41).await;
    let snapshot = harness
        .service
        .get_snapshot("default", None)
        .await
        .expect("the clamped upper bound");
    assert_weight(only_edge(&snapshot), 2.0);

    submit_failures(&harness, 21..=61).await;
    let snapshot = harness
        .service
        .get_snapshot("default", None)
        .await
        .expect("the lower bound");
    assert_eq!(only_edge(&snapshot)["failure_count"], json!(61));
    assert_weight(only_edge(&snapshot), 0.0);

    submit_failures(&harness, 62..=62).await;
    let snapshot = harness
        .service
        .get_snapshot("default", None)
        .await
        .expect("the clamped lower bound");
    let edge = only_edge(&snapshot);
    assert_eq!(edge["success_count"], json!(41));
    assert_eq!(edge["failure_count"], json!(62));
    assert_eq!(edge["attempt_count"], json!(103));
    assert_weight(edge, 0.0);
}

#[tokio::test]
async fn high_water_cursor_is_monotonic_and_never_replays() {
    let harness = harness().await;
    let first = harness
        .service
        .submit(&success_input("e1", "default", "session-1"))
        .await
        .expect("the first submission");
    let second = harness
        .service
        .submit(&success_input("e2", "default", "session-2"))
        .await
        .expect("the second submission");
    assert!(first.sequence < second.sequence);

    let snapshot = harness
        .service
        .get_snapshot("default", None)
        .await
        .expect("the projected snapshot");
    assert_eq!(snapshot.high_water_sequence, second.sequence);
    assert_eq!(snapshot.high_water_sequence, 2);

    // Nothing new was appended, so the replay publishes the identical revisions.
    let replayed = harness
        .service
        .reproject("default")
        .await
        .expect("the idempotent reprojection");
    assert_eq!(replayed.observation_revision, snapshot.observation_revision);
    assert_eq!(replayed.merged_revision, snapshot.merged_revision);
    assert_eq!(replayed.high_water_sequence, snapshot.high_water_sequence);
    assert_eq!(replayed.overlay, snapshot.overlay);

    // An audit-only row advances the cursor and is never re-qualified.
    let mut truncated = success_input("e3", "default", "session-3");
    truncated.trace.truncated = true;
    let audited = harness
        .service
        .submit(&truncated)
        .await
        .expect("an audited submission");
    assert_eq!(audited.status, ObservationStatus::AuditOnly);
    assert_eq!(audited.reason, TRUNCATED_TRACE);
    assert_eq!(audited.sequence, 3);

    let after = harness
        .service
        .reproject("default")
        .await
        .expect("the snapshot after the audit-only row");
    assert_eq!(after.high_water_sequence, 3);
    assert_eq!(only_edge(&after)["success_count"], json!(2));

    let again = harness
        .service
        .reproject("default")
        .await
        .expect("a second reprojection");
    assert_eq!(again.high_water_sequence, 3);
    assert_eq!(again.observation_revision, after.observation_revision);
    assert_eq!(only_edge(&again)["success_count"], json!(2));
}

#[tokio::test]
async fn safe_fallback_is_retryable_and_replays_the_log() {
    let harness = harness().await;
    harness
        .service
        .submit(&success_input("e1", "default", "session-1"))
        .await
        .expect("a successful observation");

    let fallback = harness
        .service
        .publish_safe_fallback("default", "projection_failed")
        .await
        .expect("a safe fallback");
    assert_eq!(fallback.overlay["projection_status"], json!(SAFE_FALLBACK));
    assert!(overlay_edges(&fallback).is_empty());
    assert_eq!(
        fallback.overlay["diagnostics"],
        json!(["projection_failed"])
    );
    assert_eq!(fallback.high_water_sequence, 0);

    // A reprojection replays the whole scope log because the cursor was reset to zero.
    let restored = harness
        .service
        .reproject("default")
        .await
        .expect("the replayed projection");
    assert_eq!(restored.overlay["projection_status"], json!("ready"));
    assert_eq!(restored.high_water_sequence, 1);
    assert_weight(only_edge(&restored), 1.05);

    // The same repair happens on a read of a fresh fallback.
    harness
        .service
        .publish_safe_fallback("default", "graph")
        .await
        .expect("a second safe fallback");
    let replayed = harness
        .service
        .get_snapshot("default", None)
        .await
        .expect("the repaired read");
    assert_eq!(replayed.overlay["projection_status"], json!("ready"));
    assert_eq!(replayed.high_water_sequence, 1);
    assert_weight(only_edge(&replayed), 1.05);
}

#[tokio::test]
async fn republication_rebinds_observed_edges() {
    let harness = harness().await;
    harness
        .service
        .submit(&success_input("e1", "default", "session-1"))
        .await
        .expect("a successful observation");
    let active = harness
        .service
        .get_snapshot("default", None)
        .await
        .expect("the `v1` snapshot");
    assert_eq!(only_edge(&active)["binding_status"], json!("active_static"));
    assert_weight(only_edge(&active), 1.05);

    // `v2` keeps the edge but moves an endpoint's content hash.
    harness
        .artifacts
        .publish(&static_graph_with_rehashed_target(), Some("v2"))
        .await
        .expect("publishing `v2`");
    let invalidated = harness
        .service
        .get_snapshot("default", None)
        .await
        .expect("the `v2` snapshot");
    assert_eq!(invalidated.static_revision, "v2");
    assert_eq!(
        only_edge(&invalidated)["binding_status"],
        json!("invalidated")
    );
    assert_weight(only_edge(&invalidated), 1.05);
    assert_ne!(
        invalidated.observation_revision,
        active.observation_revision
    );

    // `v3` drops the edge but keeps both endpoints.
    harness
        .artifacts
        .publish(&static_graph_without_edges(), Some("v3"))
        .await
        .expect("publishing `v3`");
    let quarantined = harness
        .service
        .get_snapshot("default", None)
        .await
        .expect("the `v3` snapshot");
    assert_eq!(quarantined.static_revision, "v3");
    let edge = only_edge(&quarantined);
    assert_eq!(edge["binding_status"], json!("quarantined"));
    assert_eq!(edge["last_evidence_static_revision"], json!("v1"));
    assert_eq!(edge["runtime_only"], json!(false));
    assert_weight(edge, 1.05);
    assert_ne!(
        quarantined.observation_revision,
        invalidated.observation_revision
    );

    // `v4` drops the target endpoint entirely.
    harness
        .artifacts
        .publish(&static_graph_without_target(), Some("v4"))
        .await
        .expect("publishing `v4`");
    let orphaned = harness
        .service
        .get_snapshot("default", None)
        .await
        .expect("the `v4` snapshot");
    assert_eq!(orphaned.static_revision, "v4");
    assert_eq!(only_edge(&orphaned)["binding_status"], json!("orphaned"));
    assert_weight(only_edge(&orphaned), 1.05);
    assert_ne!(
        orphaned.observation_revision,
        quarantined.observation_revision
    );
}

#[tokio::test]
async fn merged_revisions_are_pinnable_and_missing_revisions_are_not_found() {
    let harness = harness().await;
    harness
        .service
        .submit(&success_input("e1", "default", "session-1"))
        .await
        .expect("the first observation");
    let pinned = harness
        .service
        .get_snapshot("default", None)
        .await
        .expect("the pinned snapshot");

    // Newer evidence moves the current snapshot, not the pinned one.
    harness
        .service
        .submit(&success_input("e2", "default", "session-2"))
        .await
        .expect("the second observation");
    let current = harness
        .service
        .get_snapshot("default", None)
        .await
        .expect("the current snapshot");
    assert_ne!(current.merged_revision, pinned.merged_revision);

    let reread = harness
        .service
        .get_snapshot("default", Some(&pinned.merged_revision))
        .await
        .expect("the pinned snapshot");
    assert_eq!(reread.merged_revision, pinned.merged_revision);
    assert_eq!(reread.observation_revision, pinned.observation_revision);
    assert_eq!(reread.high_water_sequence, pinned.high_water_sequence);
    assert_eq!(reread.overlay, pinned.overlay);

    // A newer static publication does not move the pinned snapshot either.
    harness
        .artifacts
        .publish(&static_graph(), Some("v2"))
        .await
        .expect("publishing `v2`");
    let pinned_after_publication = harness
        .service
        .get_snapshot("default", Some(&pinned.merged_revision))
        .await
        .expect("the still-pinned snapshot");
    assert_eq!(
        pinned_after_publication.merged_revision,
        pinned.merged_revision
    );
    assert_eq!(pinned_after_publication.static_revision, "v1");
    assert_eq!(pinned_after_publication.overlay, pinned.overlay);

    let error = harness
        .service
        .get_snapshot("default", Some("merged-missing"))
        .await
        .expect_err("an unknown pinned revision is not found");
    assert_eq!(error.kind(), ErrorKind::NotFound);
    assert!(error.message().contains("merged_revision"));
    assert!(error.message().contains("merged-missing"));

    // An unknown *static* revision is audited, not rejected.
    let mut unknown = success_input("e3", "default", "session-3");
    unknown.graph_snapshot.static_revision = "v9".to_string();
    let receipt = harness
        .service
        .submit(&unknown)
        .await
        .expect("an audited unknown static revision");
    assert_eq!(receipt.status, ObservationStatus::AuditOnly);
    assert_eq!(receipt.reason, UNKNOWN_STATIC_REVISION);

    // A store that never published a static graph has nothing to bind to.
    let empty_root = tempfile::tempdir().expect("a second artifact root");
    let unpubublished: Arc<dyn GraphArtifactStore> = Arc::new(MemoryGraphArtifactStore::new());
    let service = GraphObservationService::new(unpubublished, empty_root.path());
    let error = service
        .get_snapshot("default", None)
        .await
        .expect_err("a missing static revision is not found");
    assert_eq!(error.kind(), ErrorKind::NotFound);
    assert!(error.message().contains("static_revision"));
}

#[tokio::test]
async fn evidence_and_revisions_persist_sanitised() {
    let harness = harness().await;
    let mut input = success_input("e1", "default", "session-raw");
    input.trace.trajectory_id = "trajectory-raw".to_string();
    input.trace.request_id = Some("request-raw".to_string());
    input.task.query = Some("api_key: sk-live-0123456789abcdef".to_string());
    let receipt = harness
        .service
        .submit(&input)
        .await
        .expect("a successful observation");
    assert_eq!(receipt.status, ObservationStatus::Accepted);
    let snapshot = harness
        .service
        .get_snapshot("default", None)
        .await
        .expect("the current snapshot");

    let log_path = harness.root.path().join("evolution").join("evidence.jsonl");
    assert!(log_path.is_file(), "missing {}", log_path.display());
    let text = std::fs::read_to_string(&log_path).expect("reading the evidence log");
    let lines: Vec<&str> = text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .collect();
    assert_eq!(lines.len(), 1, "one submission appends exactly one row");
    let row: Value = serde_json::from_str(lines[0]).expect("decoding the evidence row");
    assert_eq!(row["sequence"], json!(receipt.sequence));
    assert_eq!(row["graph_scope_id"], json!("default"));
    assert_eq!(row["evidence_id"], json!("e1"));
    assert_eq!(row["eligible"], json!(true));
    assert_eq!(row["reason"], json!(""));

    let payload: Value = serde_json::from_str(row["payload_json"].as_str().expect("payload text"))
        .expect("decoding the persisted payload");
    assert_private_hash(
        payload["trajectory_id_hash"]
            .as_str()
            .expect("a trajectory hash"),
        "trajectory-raw",
    );
    assert_private_hash(
        payload["session_id_hash"].as_str().expect("a session hash"),
        "session-raw",
    );
    assert_private_hash(
        payload["request_id_hash"].as_str().expect("a request hash"),
        "request-raw",
    );
    for raw in [
        "trajectory-raw",
        "session-raw",
        "request-raw",
        "sk-live-0123456789abcdef",
    ] {
        assert!(!text.contains(raw), "the log leaked `{raw}`");
    }
    let query = payload["query"].as_str().expect("the redacted query");
    assert_eq!(query, format!("api_key={REDACTED}"));

    let scope_root = harness.scope_root("default");
    let overlay_path = scope_root
        .join("observations")
        .join(&snapshot.observation_revision)
        .join("overlay.json");
    let manifest_path = scope_root
        .join("merged")
        .join(&snapshot.merged_revision)
        .join("manifest.json");
    for path in [
        scope_root.join("current.json"),
        overlay_path.clone(),
        manifest_path,
    ] {
        assert!(path.is_file(), "missing {}", path.display());
    }
    let overlay = read_json(&overlay_path);
    assert_eq!(
        overlay["observation_revision"],
        json!(snapshot.observation_revision)
    );
    assert_eq!(overlay["static_revision"], json!("v1"));
    assert_eq!(overlay["projection_status"], json!("ready"));
    assert_eq!(overlay["high_water_sequence"], json!(1));
    assert_eq!(
        overlay["edges"].as_object().expect("overlay edges").len(),
        1
    );

    // `close` is a documented no-op, and a fresh service rehydrates the same snapshot.
    harness.service.close().await.expect("a successful close");
    let reopened = harness.reopen();
    let rehydrated = reopened
        .get_snapshot("default", None)
        .await
        .expect("the rehydrated snapshot");
    assert_eq!(rehydrated, snapshot);
}

#[tokio::test]
async fn observed_at_is_persisted_as_utc() {
    let harness = harness().await;
    let mut input = success_input("e1", "default", "session-1");
    input.observed_at = "2026-03-01T00:00:00-02:00".to_string();
    harness
        .service
        .submit(&input)
        .await
        .expect("a successful observation");

    let rows = harness.evidence().rows().await.expect("the evidence log");
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].observed_at, "2026-03-01T02:00:00Z");
    let payload = rows[0].payload().expect("the persisted payload");
    assert_eq!(payload["observed_at"], json!("2026-03-01T02:00:00Z"));

    let snapshot = harness
        .service
        .get_snapshot("default", None)
        .await
        .expect("the current snapshot");
    let edge = only_edge(&snapshot);
    assert_eq!(edge["first_observed_at"], json!("2026-03-01T02:00:00Z"));
    assert_eq!(edge["last_observed_at"], json!("2026-03-01T02:00:00Z"));
}
