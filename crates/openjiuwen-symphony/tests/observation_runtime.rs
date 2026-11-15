//! Integration tests for the runtime's graph-observation wiring: the observation service is
//! derived from `graph_artifact_root`, submitted evidence round-trips into a merged snapshot
//! with the observed `runtime_weight`, an in-memory runtime reports `Unsupported`, and the
//! append-only evidence log survives `close` plus a rebuilt runtime.
//!
//! Everything is offline and deterministic: `tempfile` roots, explicit `observed_at` strings,
//! no sleeps, and no assertion that depends on the wall clock.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;

use openjiuwen_symphony::error::is_unsupported;
use openjiuwen_symphony::model::{CapabilityDescriptor, CapabilityIo};
use openjiuwen_symphony::observation::contracts::{
    EvidenceStrength, EvolutionEdgeMetadata, EvolutionGraph, EvolutionGraphEdge,
    EvolutionGraphNode, GraphEvolutionInput, GraphSnapshot, GraphSnapshotRef, ObservationStatus,
    TaskEvidence, TaskOutcome, TaskOutcomeLabel, TraceEvidence,
};
use openjiuwen_symphony::runtime::{SymphonyConfig, SymphonyRuntime};
use serde_json::{json, Value};
use tempfile::TempDir;

/// `skill:a` — produces `text`.
fn producer() -> CapabilityDescriptor {
    CapabilityDescriptor::new("skill", "a", "A")
        .with_outputs(vec![CapabilityIo::new("text", "string")])
}

/// `skill:b` — consumes `text`.
fn consumer() -> CapabilityDescriptor {
    CapabilityDescriptor::new("skill", "b", "B")
        .with_inputs(vec![CapabilityIo::new("text", "string")])
}

/// A runtime configuration whose graph revisions (and so its evidence log) live under `root`.
fn filesystem_config(root: &TempDir) -> SymphonyConfig {
    SymphonyConfig {
        graph_artifact_root: Some(root.path().to_path_buf()),
        ..SymphonyConfig::default()
    }
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

/// An execution graph observing the static `skill:a -> skill:b` transition as a success.
fn success_execution_graph() -> EvolutionGraph {
    let nodes = ["skill:a", "skill:b"]
        .into_iter()
        .map(|id| (id.to_string(), execution_node()))
        .collect();
    EvolutionGraph {
        graph_type: "execution_graph".to_string(),
        nodes,
        edges: vec![EvolutionGraphEdge {
            source_id: "skill:a".to_string(),
            target_id: "skill:b".to_string(),
            relation_type: "can_feed".to_string(),
            metadata: EvolutionEdgeMetadata {
                success: Some(true),
                failure_domain: None,
            },
        }],
        ..EvolutionGraph::default()
    }
}

/// A valid input crediting the static edge as one strong, verified success.
fn evidence_input(graph_scope_id: &str, static_revision: &str) -> GraphEvolutionInput {
    GraphEvolutionInput {
        evidence_id: "evidence-1".to_string(),
        graph_scope_id: graph_scope_id.to_string(),
        observed_at: "2026-03-01T00:00:00Z".to_string(),
        graph_snapshot: GraphSnapshotRef {
            static_revision: static_revision.to_string(),
            observation_revision: "observation-seed".to_string(),
            merged_revision: None,
        },
        trace: TraceEvidence {
            trajectory_id: "trajectory-1".to_string(),
            session_id: "session-1".to_string(),
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
        execution_graph: success_execution_graph(),
        ..GraphEvolutionInput::default()
    }
}

/// The single overlay edge record of `snapshot`.
fn only_edge(snapshot: &GraphSnapshot) -> &serde_json::Map<String, Value> {
    let edges = snapshot.overlay["edges"]
        .as_object()
        .expect("the overlay carries an `edges` object");
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

/// Opens a file-backed runtime holding [`producer`] and [`consumer`], publishes the graph, and
/// returns it together with the config and the published static revision.
async fn runtime_with_static_edge(root: &TempDir) -> (SymphonyRuntime, SymphonyConfig, String) {
    let config = filesystem_config(root);
    let runtime = SymphonyRuntime::open(config.clone())
        .await
        .expect("open the file-backed runtime");
    for descriptor in [producer(), consumer()] {
        runtime
            .registry()
            .register(descriptor)
            .await
            .expect("register the descriptor");
    }
    let revision = runtime.build().await.expect("publish the static graph");
    assert_eq!(
        revision.edge_count, 1,
        "the catalog derives exactly `skill:a -> skill:b`"
    );
    let static_revision = runtime
        .graph_status()
        .await
        .expect("read the artifact status")
        .current_version
        .expect("a published static revision");
    assert_eq!(static_revision, revision.version);
    (runtime, config, static_revision)
}

#[tokio::test]
async fn observation_round_trips_through_the_runtime() {
    let root = TempDir::new().expect("a temporary artifact root");
    let (runtime, config, static_revision) = runtime_with_static_edge(&root).await;

    let input = evidence_input(&config.graph_scope_id, &static_revision);
    let receipt = runtime
        .submit_observation(&input)
        .await
        .expect("submit the evidence");
    assert_eq!(receipt.evidence_id, "evidence-1");
    assert_eq!(receipt.graph_scope_id, config.graph_scope_id);
    assert_eq!(
        receipt.sequence, 1,
        "the first row of the log is sequence 1"
    );
    assert_eq!(receipt.status, ObservationStatus::Accepted);
    assert!(
        receipt.reason.is_empty(),
        "an accepted row has no refusal reason: {}",
        receipt.reason
    );

    let snapshot = runtime
        .observation_snapshot(&config.graph_scope_id, None)
        .await
        .expect("read the observation snapshot");
    assert_eq!(snapshot.graph_scope_id, config.graph_scope_id);
    assert_eq!(snapshot.static_revision, static_revision);
    assert_eq!(snapshot.high_water_sequence, 1);

    let edge = only_edge(&snapshot);
    // Overlay ids are the normalized capability ids, without the `skill:` qualifier.
    assert_eq!(edge["source_id"], json!("a"));
    assert_eq!(edge["target_id"], json!("b"));
    assert_eq!(edge["relation_type"], json!("can_feed"));
    assert_eq!(
        edge["binding_status"],
        json!("active_static"),
        "the observed transition is a current static edge: {edge:?}"
    );
    assert_eq!(edge["runtime_only"], json!(false));
    assert_eq!(edge["success_count"], json!(1));
    assert_eq!(edge["attempt_count"], json!(1));
    let weight = edge["runtime_weight"].as_f64().expect("a runtime weight");
    assert!(
        (weight - 1.05).abs() < 1e-9,
        "one net success moves the weight by 0.05, got {weight}"
    );

    runtime
        .flush_observations()
        .await
        .expect("flush the evidence log");
}

#[tokio::test]
async fn observation_requires_a_filesystem_root() {
    let runtime = SymphonyRuntime::open(SymphonyConfig::default())
        .await
        .expect("open the in-memory runtime");
    let input = evidence_input(&runtime.config().graph_scope_id, "v1");

    let submit = runtime.submit_observation(&input).await;
    let submit = submit.expect_err("an in-memory runtime has no evidence log");
    assert!(
        is_unsupported(&submit),
        "expected `Unsupported`, got {:?}",
        submit.kind()
    );

    let snapshot = runtime.observation_snapshot("default", None).await;
    let snapshot = snapshot.expect_err("an in-memory runtime has no merged snapshot");
    assert!(
        is_unsupported(&snapshot),
        "expected `Unsupported`, got {:?}",
        snapshot.kind()
    );

    let flush = runtime.flush_observations().await;
    let flush = flush.expect_err("an in-memory runtime has nothing to flush");
    assert!(
        is_unsupported(&flush),
        "expected `Unsupported`, got {:?}",
        flush.kind()
    );

    // `close` never needs an observation service, so an in-memory runtime still closes.
    runtime.close().await.expect("close the in-memory runtime");
}

#[tokio::test]
async fn close_flushes_observations() {
    let root = TempDir::new().expect("a temporary artifact root");
    let (runtime, config, static_revision) = runtime_with_static_edge(&root).await;
    runtime
        .submit_observation(&evidence_input(&config.graph_scope_id, &static_revision))
        .await
        .expect("submit the evidence");
    runtime
        .close()
        .await
        .expect("close flushes the evidence log");
    drop(runtime);

    let reopened = SymphonyRuntime::open(config.clone())
        .await
        .expect("reopen the runtime over the same root");
    let snapshot = reopened
        .observation_snapshot(&config.graph_scope_id, None)
        .await
        .expect("the evidence log survived the close");
    assert_eq!(snapshot.static_revision, static_revision);
    assert_eq!(snapshot.high_water_sequence, 1);
    let edge = only_edge(&snapshot);
    assert_eq!(edge["binding_status"], json!("active_static"));
    let weight = edge["runtime_weight"].as_f64().expect("a runtime weight");
    assert!(
        (weight - 1.05).abs() < 1e-9,
        "the rehydrated overlay keeps the observed weight, got {weight}"
    );
    assert_eq!(edge["success_count"], json!(1));
}
