use std::collections::BTreeMap;

use serde_json::json;
use tempfile::TempDir;

use super::*;
use crate::artifact::MemoryGraphArtifactStore;
use crate::error::ErrorKind;
use crate::graph::{CapabilityEdge, CapabilityGraph, CapabilityNode, CAN_FEED};
use crate::observation::contracts::{
    EvolutionGraph, EvolutionGraphEdge, EvolutionGraphNode, GraphSnapshotRef, TaskEvidence,
    TaskOutcome, TraceEvidence,
};

/// A service over a fresh artifact root whose static graph is published as `v1`.
struct Harness {
    /// Keeps the artifact root alive for as long as the service lives.
    _root: TempDir,
    /// The static revision source, kept typed so a test can republish.
    artifacts: Arc<MemoryGraphArtifactStore>,
    /// The service under test.
    service: GraphObservationService,
}

/// Builds a harness with `v1` published.
async fn harness() -> Harness {
    let root = tempfile::tempdir().unwrap();
    let artifacts = Arc::new(MemoryGraphArtifactStore::new());
    artifacts
        .publish(&static_graph(), Some("v1"))
        .await
        .unwrap();
    let service = GraphObservationService::new(
        Arc::clone(&artifacts) as Arc<dyn GraphArtifactStore>,
        root.path(),
    );
    Harness {
        _root: root,
        artifacts,
        service,
    }
}

/// A two-capability static graph with one `can_feed` edge.
fn static_graph() -> CapabilityGraph {
    let mut graph = CapabilityGraph::new("observation-test");
    graph.upsert_node(node("skill:a"));
    graph.upsert_node(node("skill:b"));
    graph
        .add_edge(CapabilityEdge::new("skill:a", "skill:b", CAN_FEED, 1.0))
        .unwrap();
    graph
}

/// The same nodes without the edge, for the quarantine scenario.
fn static_graph_without_edges() -> CapabilityGraph {
    let mut graph = CapabilityGraph::new("observation-test");
    graph.upsert_node(node("skill:a"));
    graph.upsert_node(node("skill:b"));
    graph
}

/// One static node carrying a capability type and a content hash.
fn node(id: &str) -> CapabilityNode {
    let mut node = CapabilityNode::new(id, id, "skill");
    node.metadata
        .insert("capability_type".to_string(), json!("skill"));
    node.metadata
        .insert("content_hash".to_string(), json!(format!("hash-{id}")));
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

/// One observed transition.
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

/// An execution graph over both capabilities with `edges`.
fn execution_graph_with(edges: Vec<EvolutionGraphEdge>) -> EvolutionGraph {
    let nodes = ["skill:a", "skill:b"]
        .into_iter()
        .map(|id| (id.to_string(), execution_node()))
        .collect();
    EvolutionGraph {
        graph_type: "execution_graph".to_string(),
        nodes,
        edges,
        ..EvolutionGraph::default()
    }
}

/// A successful transition along the static edge.
fn execution_graph() -> EvolutionGraph {
    execution_graph_with(vec![execution_edge("skill:a", "skill:b", Some(true), None)])
}

/// A valid input that credits the static edge as a strong success.
fn input(evidence_id: &str, scope: &str, session_id: &str) -> GraphEvolutionInput {
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
        execution_graph: execution_graph(),
        ..GraphEvolutionInput::default()
    }
}

/// A valid input whose edge and task both carry a verified failure in `domain`.
fn failure_input(
    evidence_id: &str,
    scope: &str,
    session_id: &str,
    domain: FailureDomain,
) -> GraphEvolutionInput {
    let mut input = input(evidence_id, scope, session_id);
    input.task.outcome = TaskOutcome {
        label: TaskOutcomeLabel::VerifiedFailure,
        evidence_strength: EvidenceStrength::Strong,
        failure_domain: Some(domain),
    };
    input.execution_graph = execution_graph_with(vec![execution_edge(
        "skill:a",
        "skill:b",
        Some(false),
        Some(domain),
    )]);
    input
}

/// An input crediting the reverse transition, which the static graph does not contain.
fn runtime_only_input(evidence_id: &str, scope: &str, session_id: &str) -> GraphEvolutionInput {
    let mut input = input(evidence_id, scope, session_id);
    input.execution_graph =
        execution_graph_with(vec![execution_edge("skill:b", "skill:a", Some(true), None)]);
    input
}

/// The overlay edges of a snapshot.
fn edges_of(snapshot: &GraphSnapshot) -> &Map<String, Value> {
    snapshot.overlay[EDGES_FIELD].as_object().unwrap()
}

/// The single overlay edge record of a snapshot.
fn only_edge(snapshot: &GraphSnapshot) -> &Map<String, Value> {
    let edges = edges_of(snapshot);
    assert_eq!(edges.len(), 1, "expected exactly one overlay edge");
    edges.values().next().unwrap().as_object().unwrap()
}

/// Asserts a weight to within floating-point noise.
fn assert_weight(record: &Map<String, Value>, expected: f64) {
    let weight = record[RUNTIME_WEIGHT_FIELD].as_f64().unwrap();
    assert!(
        (weight - expected).abs() < 1e-9,
        "expected weight {expected}, got {weight}"
    );
}

#[tokio::test]
async fn strong_success_on_a_static_edge_is_accepted() {
    let harness = harness().await;
    let receipt = harness
        .service
        .submit(&input("e1", "default", "session-1"))
        .await
        .unwrap();
    assert_eq!(receipt.status, ObservationStatus::Accepted);
    assert_eq!(receipt.reason, "");
    assert_eq!(receipt.sequence, 1);

    let snapshot = harness.service.get_snapshot("default", None).await.unwrap();
    assert_eq!(snapshot.high_water_sequence, 1);
    assert_eq!(snapshot.static_revision, "v1");
    let edge = only_edge(&snapshot);
    assert_eq!(edge[BINDING_STATUS_FIELD], json!("active_static"));
    assert_eq!(edge[SUCCESS_COUNT_FIELD], json!(1));
    assert_eq!(edge[ATTEMPT_COUNT_FIELD], json!(1));
    assert_eq!(edge[RUNTIME_ONLY_FIELD], json!(false));
    assert_eq!(edge[EVER_STATIC_FIELD], json!(true));
    assert_weight(edge, 1.05);
}

#[tokio::test]
async fn identical_resubmission_is_a_duplicate() {
    let harness = harness().await;
    let first = harness
        .service
        .submit(&input("e1", "default", "session-1"))
        .await
        .unwrap();
    let second = harness
        .service
        .submit(&input("e1", "default", "session-1"))
        .await
        .unwrap();
    assert_eq!(second.status, ObservationStatus::Duplicate);
    assert_eq!(second.sequence, first.sequence);
    assert_eq!(harness.service.evidence.len().await.unwrap(), 1);
    assert_eq!(
        edges_of(&harness.service.get_snapshot("default", None).await.unwrap()).len(),
        1
    );
}

#[tokio::test]
async fn reused_evidence_id_with_different_content_is_rejected() {
    let harness = harness().await;
    harness
        .service
        .submit(&input("e1", "default", "session-1"))
        .await
        .unwrap();
    let mut changed = input("e1", "default", "session-2");
    changed.observed_at = "2026-03-02T00:00:00Z".to_string();
    let error = harness.service.submit(&changed).await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains("evidence_id"));
}

#[tokio::test]
async fn truncated_trace_is_audit_only_and_advances_the_cursor() {
    let harness = harness().await;
    let mut truncated = input("e1", "default", "session-1");
    truncated.trace.truncated = true;
    let receipt = harness.service.submit(&truncated).await.unwrap();
    assert_eq!(receipt.status, ObservationStatus::AuditOnly);
    assert_eq!(receipt.reason, TRUNCATED_TRACE);

    let snapshot = harness.service.get_snapshot("default", None).await.unwrap();
    assert_eq!(snapshot.high_water_sequence, 1);
    assert!(edges_of(&snapshot).is_empty());

    // The cursor has already passed the row, so a resubmission can never become
    // eligible later.
    let again = harness.service.submit(&truncated).await.unwrap();
    assert_eq!(again.status, ObservationStatus::Duplicate);
    assert_eq!(again.reason, TRUNCATED_TRACE);
    let snapshot = harness.service.get_snapshot("default", None).await.unwrap();
    assert_eq!(snapshot.high_water_sequence, 1);
    assert!(edges_of(&snapshot).is_empty());
}

#[tokio::test]
async fn unknown_static_revision_is_audited_against_the_current_index() {
    let harness = harness().await;
    let mut unknown = input("e1", "default", "session-1");
    unknown.graph_snapshot.static_revision = "v9".to_string();
    let receipt = harness.service.submit(&unknown).await.unwrap();
    assert_eq!(receipt.status, ObservationStatus::AuditOnly);
    assert!(receipt.reason.starts_with(UNKNOWN_STATIC_REVISION));

    let rows = harness.service.evidence.rows().await.unwrap();
    assert_eq!(rows.len(), 1);
    let payload = rows[0].payload().unwrap();
    assert_eq!(payload[STATIC_REVISION_FIELD], json!("v9"));
    assert_eq!(payload[EDGES_FIELD].as_array().unwrap().len(), 1);
    assert!(edges_of(&harness.service.get_snapshot("default", None).await.unwrap()).is_empty());
}

#[tokio::test]
async fn orchestration_failure_decrements_the_weight() {
    let harness = harness().await;
    let receipt = harness
        .service
        .submit(&failure_input(
            "e1",
            "default",
            "session-1",
            FailureDomain::Orchestration,
        ))
        .await
        .unwrap();
    assert_eq!(receipt.status, ObservationStatus::Accepted);
    let snapshot = harness.service.get_snapshot("default", None).await.unwrap();
    let edge = only_edge(&snapshot);
    assert_eq!(edge[FAILURE_COUNT_FIELD], json!(1));
    assert_weight(edge, 0.95);
}

#[tokio::test]
async fn network_failure_counts_nothing() {
    let harness = harness().await;
    harness
        .service
        .submit(&input("e1", "default", "session-1"))
        .await
        .unwrap();
    harness
        .service
        .submit(&failure_input(
            "e2",
            "default",
            "session-2",
            FailureDomain::Orchestration,
        ))
        .await
        .unwrap();
    // One success and one orchestration failure cancel out exactly.
    assert_weight(
        only_edge(&harness.service.get_snapshot("default", None).await.unwrap()),
        1.0,
    );

    let receipt = harness
        .service
        .submit(&failure_input(
            "e3",
            "default",
            "session-3",
            FailureDomain::Network,
        ))
        .await
        .unwrap();
    assert_eq!(receipt.status, ObservationStatus::AuditOnly);
    assert_eq!(receipt.reason, NO_QUALIFIED_EDGE_OUTCOME);
    let snapshot = harness.service.get_snapshot("default", None).await.unwrap();
    let edge = only_edge(&snapshot);
    assert_eq!(edge[FAILURE_COUNT_FIELD], json!(1));
    assert_weight(edge, 1.0);
}

#[tokio::test]
async fn runtime_only_edges_need_two_successful_sessions() {
    let harness = harness().await;
    harness
        .service
        .submit(&runtime_only_input("e1", "default", "session-1"))
        .await
        .unwrap();
    let snapshot = harness.service.get_snapshot("default", None).await.unwrap();
    let edge = only_edge(&snapshot);
    assert_eq!(
        edge[BINDING_STATUS_FIELD],
        json!(BINDING_INSUFFICIENT_EVIDENCE)
    );
    assert_eq!(edge[RUNTIME_ONLY_FIELD], json!(false));
    assert_weight(edge, 1.05);

    harness
        .service
        .submit(&runtime_only_input("e2", "default", "session-2"))
        .await
        .unwrap();
    let snapshot = harness.service.get_snapshot("default", None).await.unwrap();
    let edge = only_edge(&snapshot);
    assert_eq!(
        edge[BINDING_STATUS_FIELD],
        json!(BINDING_ACTIVE_RUNTIME_ONLY)
    );
    assert_eq!(edge[RUNTIME_ONLY_FIELD], json!(true));
    assert_eq!(edge[SUCCESS_COUNT_FIELD], json!(2));
    assert_eq!(edge[ATTEMPT_COUNT_FIELD], json!(2));
    assert_weight(edge, 1.1);
}

#[tokio::test]
async fn a_dropped_static_edge_is_quarantined() {
    let harness = harness().await;
    harness
        .service
        .submit(&input("e1", "default", "session-1"))
        .await
        .unwrap();
    harness
        .artifacts
        .publish(&static_graph_without_edges(), Some("v2"))
        .await
        .unwrap();
    let snapshot = harness.service.reproject("default").await.unwrap();
    assert_eq!(snapshot.static_revision, "v2");
    let edge = only_edge(&snapshot);
    assert_eq!(edge[BINDING_STATUS_FIELD], json!(BINDING_QUARANTINED));
    assert_eq!(edge[RUNTIME_ONLY_FIELD], json!(false));
    assert_eq!(edge[LAST_EVIDENCE_STATIC_REVISION_FIELD], json!("v1"));
}

#[tokio::test]
async fn pinned_snapshots_are_returned_or_not_found() {
    let harness = harness().await;
    harness
        .service
        .submit(&input("e1", "default", "session-1"))
        .await
        .unwrap();
    let snapshot = harness.service.get_snapshot("default", None).await.unwrap();
    let pinned = harness
        .service
        .get_snapshot("default", Some(&snapshot.merged_revision))
        .await
        .unwrap();
    assert_eq!(pinned.merged_revision, snapshot.merged_revision);
    assert_eq!(pinned.observation_revision, snapshot.observation_revision);
    assert_eq!(pinned.high_water_sequence, snapshot.high_water_sequence);
    assert_eq!(pinned.overlay, snapshot.overlay);

    let error = harness
        .service
        .get_snapshot("default", Some("merged-missing"))
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::NotFound);
    assert!(error.message().contains("merged_revision"));
    assert!(error.message().contains("merged-missing"));
}

#[tokio::test]
async fn safe_fallback_replays_the_whole_log_on_the_next_read() {
    let harness = harness().await;
    harness
        .service
        .submit(&input("e1", "default", "session-1"))
        .await
        .unwrap();
    let fallback = harness
        .service
        .publish_safe_fallback("default", "graph")
        .await
        .unwrap();
    assert_eq!(fallback.high_water_sequence, 0);
    assert!(edges_of(&fallback).is_empty());
    assert_eq!(
        fallback.overlay[PROJECTION_STATUS_FIELD],
        json!(SAFE_FALLBACK)
    );
    assert_eq!(fallback.overlay["diagnostics"], json!(["graph"]));

    let replayed = harness.service.get_snapshot("default", None).await.unwrap();
    assert_eq!(replayed.high_water_sequence, 1);
    assert_eq!(
        replayed.overlay[PROJECTION_STATUS_FIELD],
        json!(PROJECTION_READY)
    );
    assert_eq!(only_edge(&replayed)[SUCCESS_COUNT_FIELD], json!(1));
}

#[tokio::test]
async fn reproject_all_covers_every_scope_and_flush_reports_nothing() {
    let harness = harness().await;
    harness
        .service
        .submit(&input("e1", "alpha", "session-1"))
        .await
        .unwrap();
    harness
        .service
        .submit(&input("e2", "beta", "session-2"))
        .await
        .unwrap();
    harness.service.flush().await.unwrap();
    let snapshots = harness.service.reproject_all().await.unwrap();
    let scopes: Vec<&str> = snapshots
        .iter()
        .map(|snapshot| snapshot.graph_scope_id.as_str())
        .collect();
    assert_eq!(scopes, vec!["alpha", "beta"]);
    assert!(snapshots
        .iter()
        .all(|snapshot| snapshot.high_water_sequence >= 1));
    assert_eq!(
        harness.service.reproject_all_safely().await.unwrap().len(),
        2
    );
    harness.service.close().await.unwrap();
}

#[tokio::test]
async fn submit_without_a_static_revision_is_not_found() {
    let root = tempfile::tempdir().unwrap();
    let artifacts = Arc::new(MemoryGraphArtifactStore::new());
    let service =
        GraphObservationService::new(artifacts as Arc<dyn GraphArtifactStore>, root.path());
    let error = service
        .submit(&input("e1", "default", "session-1"))
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::NotFound);
    assert!(error.message().contains("static_revision"));
    let error = service.get_snapshot("default", None).await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::NotFound);
}
