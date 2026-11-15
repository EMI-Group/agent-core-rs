//! Behaviour tests for the graph-observation contract, identity layer and stores.
//!
//! Covers `GraphEvolutionInput` validation, `observed_at` timezone normalisation, the
//! trace-record bridges, point-edge identity and the static graph index, and the exact
//! on-disk layout of the append-only evidence log and the immutable observation/merged
//! revisions.
//!
//! Everything here is offline and deterministic: `tempfile` roots, literal timestamps, and
//! no assertion that depends on the wall clock. `generated_at` is metadata only, so it is
//! never compared with a real time — only with strings this test itself supplied.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use tempfile::TempDir;

use openjiuwen_symphony::error::ErrorKind;
use openjiuwen_symphony::experience::TraceRecord;
use openjiuwen_symphony::graph::{CapabilityEdge, CapabilityGraph, CapabilityNode};
use openjiuwen_symphony::llm::canonical_json;
use openjiuwen_symphony::model::CapabilityDescriptor;
use openjiuwen_symphony::observation::contracts::{
    EvidenceStrength, EvolutionEdgeMetadata, EvolutionGraph, EvolutionGraphEdge,
    EvolutionGraphNode, FailureDomain, GraphEvolutionInput, GraphSnapshot, GraphSnapshotRef,
    TaskEvidence, TaskOutcome, TaskOutcomeLabel, TraceEvidence, GRAPH_EVOLUTION_INPUT_SCHEMA,
};
use openjiuwen_symphony::observation::identity::{
    edge_identity_from_observation, normalize_capability_id, stable_hash, EdgeIdentity,
    StaticGraphIndex, POINT_EDGE_IDENTITY_SCHEMA,
};
use openjiuwen_symphony::observation::store::{
    CurrentPointer, EvidenceAppendInput, EvidenceRow, EvidenceStore, OverlayInput, RevisionStore,
    MERGED_SCHEMA, OBSERVATION_SCHEMA,
};

/// The graph scope every store assertion writes into.
const SCOPE: &str = "scope-a";

/// The static `can_feed` revision the revision-store assertions publish against.
const STATIC_REVISION: &str = "v1";

/// How many hash characters a revision id suffix keeps.
const REVISION_HASH_CHARS: usize = 20;

/// A compliant execution node carrying the four allow-listed metadata keys.
fn execution_node() -> EvolutionGraphNode {
    let metadata = BTreeMap::from([
        ("capability_type".to_string(), json!("skill")),
        ("version".to_string(), json!("1.0.0")),
        ("description".to_string(), json!("summarizes text")),
        (
            "inputs".to_string(),
            json!([{"name": "text", "type": "string", "required": true}]),
        ),
        (
            "outputs".to_string(),
            json!([{"name": "summary", "type": "string"}]),
        ),
    ]);
    EvolutionGraphNode {
        label: "skill".to_string(),
        metadata,
    }
}

/// A compliant two-node execution graph with one qualified edge.
fn execution_graph() -> EvolutionGraph {
    EvolutionGraph {
        id: Some("exec-1".to_string()),
        graph_type: "execution_graph".to_string(),
        directed: true,
        nodes: BTreeMap::from([
            ("summarizer".to_string(), execution_node()),
            ("rewriter".to_string(), execution_node()),
        ]),
        edges: vec![EvolutionGraphEdge {
            source_id: "summarizer".to_string(),
            target_id: "rewriter".to_string(),
            relation_type: "can_feed".to_string(),
            metadata: EvolutionEdgeMetadata {
                success: Some(true),
                failure_domain: None,
            },
        }],
    }
}

/// A fully formed, valid input.
fn valid_input() -> GraphEvolutionInput {
    GraphEvolutionInput {
        schema_version: GRAPH_EVOLUTION_INPUT_SCHEMA.to_string(),
        evidence_id: "ev-1".to_string(),
        graph_scope_id: "scope".to_string(),
        observed_at: "2024-03-01T08:30:00+08:00".to_string(),
        graph_snapshot: GraphSnapshotRef {
            static_revision: "v1".to_string(),
            observation_revision: "observation-abc".to_string(),
            merged_revision: Some("merged-def".to_string()),
        },
        trace: TraceEvidence {
            trajectory_id: "s/1".to_string(),
            session_id: "s".to_string(),
            capture_mode: "agent".to_string(),
            request_id: Some("1".to_string()),
            truncated: false,
            quality_flags: Vec::new(),
            span_refs: Vec::new(),
            member_trajectory_refs: Vec::new(),
        },
        task: TaskEvidence {
            query: Some("summarize this".to_string()),
            task_cluster_id: None,
            outcome: TaskOutcome {
                label: TaskOutcomeLabel::VerifiedSuccess,
                evidence_strength: EvidenceStrength::Strong,
                failure_domain: None,
            },
        },
        planned_graph: None,
        execution_graph: execution_graph(),
    }
}

/// The valid input with `observed_at` replaced by `raw`.
fn with_observed_at(raw: &str) -> GraphEvolutionInput {
    GraphEvolutionInput {
        observed_at: raw.to_string(),
        ..valid_input()
    }
}

/// The `summarizer` execution node, for mutation.
fn node_mut(input: &mut GraphEvolutionInput) -> &mut EvolutionGraphNode {
    input
        .execution_graph
        .nodes
        .get_mut("summarizer")
        .expect("summarizer node")
}

/// Asserts that validation fails with a `Validation` error naming `field`.
fn assert_rejected(input: &GraphEvolutionInput, field: &str) {
    let error = input.validate().expect_err("input must be rejected");
    assert_eq!(error.kind(), ErrorKind::Validation, "{}", error.message());
    assert!(
        error.message().starts_with(field),
        "`{}` does not name `{field}`",
        error.message()
    );
}

/// Returns the lowercase hex SHA-256 of `bytes`.
fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

/// Returns the first 20 hex characters of the SHA-256 of `value`'s canonical JSON.
fn short_digest(value: &Value) -> String {
    sha256_hex(canonical_json(value).as_bytes())
        .chars()
        .take(REVISION_HASH_CHARS)
        .collect()
}

/// The sorted JSON key list of `value`, which must be an object.
fn keys_of(value: &Value) -> Vec<String> {
    let mut keys: Vec<String> = value
        .as_object()
        .expect("a JSON object")
        .keys()
        .cloned()
        .collect();
    keys.sort_unstable();
    keys
}

/// Every regular file below `root`, recursively, sorted by path.
fn files_under(root: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    let mut pending = vec![root.to_path_buf()];
    while let Some(directory) = pending.pop() {
        for entry in std::fs::read_dir(&directory).expect("read_dir") {
            let entry = entry.expect("directory entry");
            let path = entry.path();
            if entry.file_type().expect("file type").is_dir() {
                pending.push(path);
            } else {
                files.push(path);
            }
        }
    }
    files.sort();
    files
}

/// Reads one JSON document from disk.
fn read_document(path: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(path).expect("read document"))
        .expect("decode document")
}

/// A two-node, one-edge static capability graph with one published content hash.
fn capability_graph() -> CapabilityGraph {
    let mut graph = CapabilityGraph::new("graph-1");
    graph.upsert_node(
        CapabilityNode::new("skill:a", "Skill A", "skill")
            .with_metadata("content_hash", json!("artifact-hash-a")),
    );
    graph.upsert_node(CapabilityNode::new("skill:b", "Skill B", "skill"));
    graph
        .add_edge(CapabilityEdge::can_feed("skill:a", "skill:b"))
        .unwrap();
    graph
}

/// One append request for `payload`.
fn evidence_input<'a>(
    scope: &'a str,
    evidence_id: &'a str,
    payload: &'a Value,
    eligible: bool,
    reason: &'a str,
) -> EvidenceAppendInput<'a> {
    EvidenceAppendInput {
        graph_scope_id: scope,
        evidence_id,
        observed_at: "2024-03-01T00:30:00Z",
        eligible,
        reason,
        payload,
    }
}

/// The sequences of `rows`, in order.
fn sequences(rows: &[EvidenceRow]) -> Vec<u64> {
    rows.iter().map(|row| row.sequence).collect()
}

/// Two id-sorted overlay edges with the planner-facing keys.
fn overlay_edges() -> BTreeMap<String, Value> {
    BTreeMap::from([
        (
            "skill:a->skill:b".to_string(),
            json!({"binding_status": "active_static", "runtime_weight": 1.1}),
        ),
        (
            "skill:b->skill:c".to_string(),
            json!({"binding_status": "active_static", "runtime_weight": 0.9}),
        ),
    ])
}

/// The canonical overlay document an observation revision id is hashed over.
fn overlay_document(scope: &str, edges: &BTreeMap<String, Value>) -> Value {
    json!({
        "schema_version": OBSERVATION_SCHEMA,
        "graph_scope_id": scope,
        "static_revision": STATIC_REVISION,
        "high_water_sequence": 7,
        "edges": edges,
        "projection_status": "ready",
        "diagnostics": [],
    })
}

/// The canonical merged manifest a merged revision id is hashed over.
fn merged_document(scope: &str, observation_revision: &str) -> Value {
    json!({
        "schema_version": MERGED_SCHEMA,
        "graph_scope_id": scope,
        "static_revision": STATIC_REVISION,
        "observation_revision": observation_revision,
    })
}

/// Publishes one overlay revision for `scope` and returns the snapshot.
async fn publish(
    store: &RevisionStore,
    scope: &str,
    edges: &BTreeMap<String, Value>,
    generated_at: &str,
) -> GraphSnapshot {
    let diagnostics: Vec<String> = Vec::new();
    let input = OverlayInput {
        graph_scope_id: scope,
        static_revision: STATIC_REVISION,
        high_water_sequence: 7,
        edges,
        generated_at,
        projection_status: "ready",
        diagnostics: &diagnostics,
    };
    store.publish(&input).await.expect("publish")
}

#[test]
fn validate_accepts_a_fully_formed_input() {
    let input = valid_input();
    input.validate().unwrap();

    let normalized = input.clone().normalized().unwrap();
    assert_eq!(normalized.evidence_id, "ev-1");
    assert_eq!(normalized.observed_at, "2024-03-01T00:30:00Z");

    // A planned graph is only accepted with its own graph type.
    let mut with_plan = valid_input();
    with_plan.planned_graph = Some(EvolutionGraph {
        graph_type: "planned_graph".to_string(),
        ..EvolutionGraph::default()
    });
    with_plan.validate().unwrap();
}

#[test]
fn validate_rejects_a_wrong_graph_type() {
    let mut input = valid_input();
    input.execution_graph.graph_type = "planned_graph".to_string();
    assert_rejected(&input, "execution_graph.type");

    let mut input = valid_input();
    input.execution_graph.graph_type = String::new();
    assert_rejected(&input, "execution_graph.type");

    let mut input = valid_input();
    input.planned_graph = Some(EvolutionGraph {
        graph_type: "execution_graph".to_string(),
        ..EvolutionGraph::default()
    });
    assert_rejected(&input, "planned_graph.type");
}

#[test]
fn validate_rejects_a_non_skill_execution_node() {
    let mut input = valid_input();
    node_mut(&mut input).label = "tool".to_string();
    let error = input.validate().expect_err("a tool node is rejected");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().starts_with("execution_graph.nodes"));
    assert!(error.message().contains("summarizer"));
    assert!(error.message().contains("label `skill`"));
}

#[test]
fn validate_rejects_unsupported_execution_node_metadata() {
    let mut input = valid_input();
    node_mut(&mut input)
        .metadata
        .insert("owner".to_string(), json!("ops"));
    assert_rejected(&input, "execution_graph.nodes.summarizer.metadata");
    let error = input.validate().unwrap_err();
    assert!(error
        .message()
        .contains("unsupported metadata field `owner`"));

    // `capability_type` must repeat the node label.
    let mut input = valid_input();
    node_mut(&mut input)
        .metadata
        .insert("capability_type".to_string(), json!("tool"));
    assert_rejected(&input, "execution_graph.nodes.summarizer.metadata");

    // `version` is mandatory and must be contract text.
    let mut input = valid_input();
    node_mut(&mut input).metadata.remove("version");
    assert_rejected(&input, "execution_graph.nodes.summarizer.metadata");

    let mut input = valid_input();
    node_mut(&mut input)
        .metadata
        .insert("version".to_string(), json!("  "));
    assert_rejected(&input, "execution_graph.nodes.summarizer.metadata");
}

#[test]
fn validate_rejects_malformed_capability_ports() {
    let cases = [
        (json!([{"type": "string"}]), "missing name"),
        (json!([{"name": "  ", "type": "string"}]), "blank name"),
        (json!([{"name": "text"}]), "missing type"),
        (json!([{"name": "text", "type": ""}]), "blank type"),
        (
            json!([{"name": "text", "type": "string", "required": "yes"}]),
            "non-boolean required",
        ),
        (
            json!([{"name": "text", "type": "string", "format": "uuid"}]),
            "unknown port key",
        ),
        (json!(["text"]), "non-object port"),
        (json!("text"), "non-array ports"),
    ];
    for (ports, problem) in cases {
        let mut input = valid_input();
        node_mut(&mut input)
            .metadata
            .insert("outputs".to_string(), ports);
        let error = input.validate().expect_err(problem);
        assert_eq!(error.kind(), ErrorKind::Validation, "{problem}");
        assert!(
            error
                .message()
                .starts_with("execution_graph.nodes.summarizer.metadata.outputs"),
            "{problem}: {}",
            error.message()
        );
    }

    // A well-formed port with an explicit description is accepted.
    let mut input = valid_input();
    node_mut(&mut input).metadata.insert(
        "inputs".to_string(),
        json!([{"name": "text", "type": "string", "description": "the text"}]),
    );
    input.validate().unwrap();
}

#[test]
fn validate_rejects_an_edge_with_an_unknown_endpoint() {
    let mut input = valid_input();
    input.execution_graph.edges.push(EvolutionGraphEdge {
        source_id: "summarizer".to_string(),
        target_id: "missing".to_string(),
        relation_type: "can_feed".to_string(),
        metadata: EvolutionEdgeMetadata::default(),
    });
    assert_rejected(&input, "execution_graph.edges");
    let error = input.validate().unwrap_err();
    assert!(error.message().contains("missing node `missing`"));

    let mut input = valid_input();
    input
        .execution_graph
        .edges
        .push(EvolutionGraphEdge::default());
    assert_rejected(&input, "execution_graph.edges");
}

#[test]
fn validate_requires_a_timezone_on_observed_at() {
    let input = with_observed_at("2024-03-01T08:30:00");
    assert_rejected(&input, "observed_at");
    let error = input.validate().unwrap_err();
    assert!(error.message().contains("must include a timezone"));
    assert!(input.normalized().is_err());
}

#[test]
fn validate_requires_a_failure_domain_for_a_verified_failure() {
    let mut input = valid_input();
    input.task.outcome = TaskOutcome {
        label: TaskOutcomeLabel::VerifiedFailure,
        evidence_strength: EvidenceStrength::Strong,
        failure_domain: None,
    };
    assert_rejected(&input, "task.outcome.failure_domain");

    input.task.outcome.failure_domain = Some(FailureDomain::Orchestration);
    input.validate().unwrap();
}

#[test]
fn edges_accept_the_source_target_and_relation_aliases() {
    let mut payload = serde_json::to_value(valid_input()).unwrap();
    payload["future_field"] = json!({"nested": true});
    payload["trace"]["future_trace_field"] = json!(1);
    payload["execution_graph"]["edges"] = json!([{
        "source": "summarizer",
        "target": "rewriter",
        "relation": "can_feed",
        "metadata": {"success": true},
    }]);

    let parsed: GraphEvolutionInput = serde_json::from_value(payload).unwrap();
    let edge = &parsed.execution_graph.edges[0];
    assert_eq!(edge.source_id, "summarizer");
    assert_eq!(edge.target_id, "rewriter");
    assert_eq!(edge.relation_type, "can_feed");
    assert_eq!(edge.metadata.success, Some(true));
    assert_eq!(edge.metadata.failure_domain, None);
    parsed.validate().unwrap();

    // The canonical wire form uses the `_id`/`_type` spellings, not the aliases.
    let rendered = serde_json::to_value(&parsed).unwrap();
    let rendered_edge = &rendered["execution_graph"]["edges"][0];
    assert_eq!(rendered_edge["source_id"], json!("summarizer"));
    assert!(rendered_edge.get("source").is_none());
    assert!(rendered_edge.get("relation").is_none());

    // Absent aliases fall back to the documented defaults.
    let edge: EvolutionGraphEdge =
        serde_json::from_value(json!({"source": "a", "target": "b"})).expect("aliases only");
    assert_eq!(edge.relation_type, "can_feed");
    assert_eq!(edge.metadata, EvolutionEdgeMetadata::default());
}

#[test]
fn normalized_shifts_a_negative_offset_across_midnight() {
    let normalized = with_observed_at("2024-02-29T22:30:00-02:00")
        .normalized()
        .expect("normalized");
    assert_eq!(normalized.observed_at, "2024-03-01T00:30:00Z");
}

#[test]
fn normalized_accepts_a_colonless_offset_and_a_lowercase_z() {
    let compact = with_observed_at("2024-03-01T08:30:00+0830")
        .normalized()
        .expect("normalized");
    assert_eq!(compact.observed_at, "2024-03-01T00:00:00Z");

    let lowercase = with_observed_at("2024-03-01T08:30:00z")
        .normalized()
        .expect("normalized");
    assert_eq!(lowercase.observed_at, "2024-03-01T08:30:00Z");
}

#[test]
fn normalized_preserves_fractional_seconds() {
    let normalized = with_observed_at("2024-03-01T08:30:00.123456+08:00")
        .normalized()
        .expect("normalized");
    assert_eq!(normalized.observed_at, "2024-03-01T00:30:00.123456Z");

    let leap_day = with_observed_at("2024-02-29T00:00:00Z")
        .normalized()
        .expect("normalized");
    assert_eq!(leap_day.observed_at, "2024-02-29T00:00:00Z");
}

#[test]
fn observed_at_rejects_trailing_garbage_and_an_impossible_date() {
    for raw in [
        "2024-03-01T08:30:00Zx",
        "2023-02-29T00:00:00Z",
        "2024-13-01T00:00:00Z",
    ] {
        let input = with_observed_at(raw);
        let error = input.validate().expect_err("rejected timestamp");
        assert_eq!(error.kind(), ErrorKind::Validation, "{raw}");
        assert!(error.message().starts_with("observed_at"), "{raw}");
        assert!(input.normalized().is_err(), "{raw}");
    }
}

#[test]
fn trace_bridges_map_a_successful_record() {
    let record = TraceRecord {
        trace_id: "sess-1/req-1".to_string(),
        session_id: "sess-1".to_string(),
        request_id: "req-1".to_string(),
        query: "summarize this".to_string(),
        success: true,
        skills: vec!["summarizer".to_string()],
        ..TraceRecord::default()
    };

    let trace = TraceEvidence::from_trace_record(&record);
    assert_eq!(trace.trajectory_id, "sess-1/req-1");
    assert_eq!(trace.session_id, "sess-1");
    assert_eq!(trace.capture_mode, "agent");
    assert_eq!(trace.request_id.as_deref(), Some("req-1"));
    assert!(!trace.truncated);
    assert!(trace.quality_flags.is_empty());
    assert!(trace.span_refs.is_empty());
    assert!(trace.member_trajectory_refs.is_empty());
    assert_eq!(
        TraceEvidence::from_trace_record_with_mode(&record, "team").capture_mode,
        "team"
    );

    let outcome = TaskOutcome::from_trace_record(&record);
    assert_eq!(outcome.label, TaskOutcomeLabel::VerifiedSuccess);
    assert_eq!(outcome.evidence_strength, EvidenceStrength::Strong);
    assert_eq!(outcome.failure_domain, None);

    let task = TaskEvidence::from_trace_record(&record);
    assert_eq!(task.query.as_deref(), Some("summarize this"));
    assert_eq!(task.task_cluster_id, None);
    assert_eq!(task.outcome, outcome);

    // The bridges produce a contract-valid payload with no extra rework.
    let input = GraphEvolutionInput {
        trace,
        task,
        ..valid_input()
    };
    input.validate().unwrap();
}

#[test]
fn trace_bridges_map_a_failed_record() {
    let record = TraceRecord {
        trace_id: "sess-2/req-9".to_string(),
        session_id: "sess-2".to_string(),
        request_id: "   ".to_string(),
        query: String::new(),
        success: false,
        ..TraceRecord::default()
    };

    let trace = TraceEvidence::from_trace_record(&record);
    assert_eq!(trace.trajectory_id, "sess-2/req-9");
    assert_eq!(trace.request_id, None, "a blank request id is not carried");

    let outcome = TaskOutcome::from_trace_record(&record);
    assert_eq!(outcome.label, TaskOutcomeLabel::VerifiedFailure);
    assert_eq!(outcome.evidence_strength, EvidenceStrength::Strong);
    assert_eq!(
        outcome.failure_domain,
        Some(FailureDomain::Unknown),
        "only a caller-supplied verdict can attribute a domain"
    );

    let task = TaskEvidence::from_trace_record(&record);
    assert_eq!(task.query, None);
    assert_eq!(task.task_cluster_id, None);

    // `FailureDomain::Unknown` satisfies the verified-failure rule.
    let input = GraphEvolutionInput {
        trace,
        task,
        ..valid_input()
    };
    input.validate().unwrap();
}

#[test]
fn stable_hash_ignores_key_order_and_tracks_content() {
    let first = json!({"b": 1, "a": {"y": 2, "x": [3, {"d": 4, "c": 5}]}});
    let reordered = json!({"a": {"x": [3, {"c": 5, "d": 4}], "y": 2}, "b": 1});
    let changed = json!({"b": 1, "a": {"y": 2, "x": [3, {"d": 4, "c": 6}]}});

    let digest = stable_hash(&first);
    assert_eq!(digest, stable_hash(&reordered));
    assert_eq!(digest, sha256_hex(canonical_json(&first).as_bytes()));
    assert_ne!(digest, stable_hash(&changed));
    assert_eq!(digest.len(), 64);
    assert!(digest
        .chars()
        .all(|character| character.is_ascii_hexdigit() && !character.is_ascii_uppercase()));
}

#[test]
fn normalize_capability_id_strips_one_prefix_of_each_kind() {
    assert_eq!(
        normalize_capability_id("  skill:web.search  "),
        "web.search"
    );
    assert_eq!(normalize_capability_id("capability:x"), "x");
    assert_eq!(normalize_capability_id("skill:capability:x"), "x");
    assert_eq!(normalize_capability_id("plain.id"), "plain.id");
    assert_eq!(normalize_capability_id("   "), "");
}

#[test]
fn edge_identity_hash_is_the_sha256_of_its_document() {
    let identity = EdgeIdentity::new("a", "b", "can_feed", "hash-a", "hash-b");
    let document = identity.to_value();
    assert_eq!(
        document["identity_schema"],
        json!(POINT_EDGE_IDENTITY_SCHEMA)
    );

    // The canonical rendering sorts the six keys.
    let canonical = format!(
        "{{\"identity_schema\":\"{POINT_EDGE_IDENTITY_SCHEMA}\",\"relation_type\":\"can_feed\",\
         \"source_content_hash\":\"hash-a\",\"source_id\":\"a\",\
         \"target_content_hash\":\"hash-b\",\"target_id\":\"b\"}}"
    );
    assert_eq!(canonical_json(&document), canonical);
    assert_eq!(identity.identity_hash(), sha256_hex(canonical.as_bytes()));
    assert_eq!(EdgeIdentity::from_value(&document), Some(identity.clone()));

    let other = EdgeIdentity::new("a", "b", "can_feed", "hash-a", "hash-b2");
    assert_ne!(identity.identity_hash(), other.identity_hash());
}

#[test]
fn static_index_reads_published_and_overridden_content_hashes() {
    let graph = capability_graph();
    let index = StaticGraphIndex::from_graph("v1", &graph);

    assert_eq!(index.revision, "v1");
    assert_eq!(
        index.capability_ids.iter().cloned().collect::<Vec<_>>(),
        vec!["a".to_string(), "b".to_string()]
    );
    assert_eq!(index.content_hash_by_id["a"], "artifact-hash-a");
    // No published hash: the node identity hash is the fallback.
    assert_eq!(index.content_hash_by_id["b"], index.graph_hash_by_id["b"]);
    assert_ne!(index.graph_hash_by_id["a"], "artifact-hash-a");

    let descriptor = CapabilityDescriptor::new("skill", "a", "Skill A");
    let overridden = StaticGraphIndex::from_graph_with_descriptors(
        "v1",
        &graph,
        std::slice::from_ref(&descriptor),
    );
    assert_eq!(
        overridden.content_hash_by_id["a"],
        descriptor.compute_content_hash()
    );
    assert_ne!(overridden.content_hash_by_id["a"], "artifact-hash-a");
    assert_eq!(
        overridden.content_hash_by_id["b"],
        index.content_hash_by_id["b"]
    );
    assert_eq!(overridden.graph_hash_by_id, index.graph_hash_by_id);
    assert_eq!(overridden.capability_ids, index.capability_ids);
}

#[test]
fn static_index_validates_nodes_and_binds_only_known_endpoints() {
    let graph = capability_graph();
    let index = StaticGraphIndex::from_graph("v1", &graph);

    assert!(index.validates_nodes(["skill:a", "b", " capability:a "]));
    assert!(index.validates_nodes(Vec::<&str>::new()));
    assert!(!index.validates_nodes(["a", "missing"]));

    let observed = EvolutionGraphEdge {
        source_id: "skill:a".to_string(),
        target_id: " b ".to_string(),
        relation_type: "can_feed".to_string(),
        metadata: EvolutionEdgeMetadata::default(),
    };
    let identity = edge_identity_from_observation(&observed, &index).expect("a bound edge");
    assert_eq!(identity.source_id, "a");
    assert_eq!(identity.target_id, "b");
    assert_eq!(identity.source_content_hash, "artifact-hash-a");
    assert_eq!(identity.target_content_hash, index.graph_hash_by_id["b"]);

    // The static edge is indexed under its own point-edge identity hash.
    let identity_hash = identity.identity_hash();
    assert_eq!(index.edge_identities.len(), 1);
    assert_eq!(index.edge_identities[&identity_hash], identity);
    assert_eq!(
        index.edges_by_identity[&identity_hash]["source"],
        json!("skill:a")
    );

    let unknown = EvolutionGraphEdge {
        source_id: "a".to_string(),
        target_id: "missing".to_string(),
        relation_type: "can_feed".to_string(),
        metadata: EvolutionEdgeMetadata::default(),
    };
    assert_eq!(edge_identity_from_observation(&unknown, &index), None);
}

#[tokio::test]
async fn evidence_log_writes_the_documented_jsonl_rows() {
    let temp = TempDir::new().unwrap();
    let store = EvidenceStore::new(temp.path());
    let payload = json!({"edge": {"source_id": "a", "target_id": "b"}, "success": true});

    let first = store
        .append(evidence_input(SCOPE, "e1", &payload, true, ""))
        .await
        .unwrap();
    let second = store
        .append(evidence_input(
            SCOPE,
            "e2",
            &payload,
            false,
            "truncated_trace",
        ))
        .await
        .unwrap();

    assert_eq!(
        store.path(),
        temp.path().join("evolution").join("evidence.jsonl")
    );
    assert_eq!(
        (first.sequence, first.duplicate, first.eligible),
        (1, false, true)
    );
    assert_eq!(first.reason, "");
    assert_eq!((second.sequence, second.eligible), (2, false));

    let text = std::fs::read_to_string(store.path()).unwrap();
    assert!(text.ends_with('\n'), "every row is newline terminated");
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 2);

    let row: Value = serde_json::from_str(lines[0]).unwrap();
    assert_eq!(
        keys_of(&row),
        vec![
            "eligible",
            "evidence_id",
            "graph_scope_id",
            "observed_at",
            "payload_digest",
            "payload_json",
            "reason",
            "sequence",
        ]
    );
    assert_eq!(row["sequence"], json!(1));
    assert_eq!(row["graph_scope_id"], json!(SCOPE));
    assert_eq!(row["evidence_id"], json!("e1"));
    assert_eq!(row["observed_at"], json!("2024-03-01T00:30:00Z"));
    assert_eq!(row["eligible"], json!(true));
    assert_eq!(row["reason"], json!(""));
    assert_eq!(row["payload_digest"], json!(stable_hash(&payload)));
    let payload_json = row["payload_json"].as_str().unwrap();
    assert_eq!(payload_json, canonical_json(&payload));
    assert!(!payload_json.contains(' '), "payload text is compact");
    assert_eq!(
        serde_json::from_str::<Value>(payload_json).unwrap(),
        payload
    );
    assert_eq!(
        row["payload_digest"].as_str().unwrap(),
        sha256_hex(canonical_json(&payload).as_bytes())
    );

    let audit: Value = serde_json::from_str(lines[1]).unwrap();
    assert_eq!(audit["sequence"], json!(2));
    assert_eq!(audit["eligible"], json!(false));
    assert_eq!(audit["reason"], json!("truncated_trace"));
    assert_eq!(audit["payload_digest"], row["payload_digest"]);

    let rows = store.rows().await.unwrap();
    assert_eq!(sequences(&rows), vec![1, 2]);
    assert_eq!(rows[0].payload().unwrap(), payload);
}

#[tokio::test]
async fn evidence_duplicates_and_reused_ids_are_rejected() {
    let temp = TempDir::new().unwrap();
    let store = EvidenceStore::new(temp.path());
    let payload = json!({"success": true});

    let first = store
        .append(evidence_input(SCOPE, "e1", &payload, true, ""))
        .await
        .unwrap();
    let replay = store
        .append(evidence_input(SCOPE, "e1", &payload, true, ""))
        .await
        .unwrap();

    assert!(replay.duplicate);
    assert_eq!(replay.sequence, first.sequence);
    assert_eq!(store.len().await.unwrap(), 1);
    assert_eq!(
        std::fs::read_to_string(store.path())
            .unwrap()
            .lines()
            .count(),
        1,
        "a duplicate must not append a line"
    );

    let error = store
        .append(evidence_input(
            SCOPE,
            "e1",
            &json!({"success": false}),
            true,
            "",
        ))
        .await
        .expect_err("a used id may not change its payload");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(
        error.message().starts_with("evidence_id"),
        "{}",
        error.message()
    );
    assert_eq!(store.len().await.unwrap(), 1);
    // The same id in another scope is independent evidence.
    store
        .append(evidence_input("scope-b", "e1", &payload, true, ""))
        .await
        .unwrap();
    assert_eq!(store.len().await.unwrap(), 2);
}

#[tokio::test]
async fn evidence_sequences_continue_after_reopening_the_store() {
    let temp = TempDir::new().unwrap();
    let payload = json!({"success": true});
    let before = {
        let store = EvidenceStore::new(temp.path());
        store
            .append(evidence_input(SCOPE, "e1", &payload, true, ""))
            .await
            .unwrap();
        store
            .append(evidence_input(SCOPE, "e2", &payload, true, ""))
            .await
            .unwrap();
        store.rows().await.unwrap()
    };

    let reopened = EvidenceStore::new(temp.path());
    assert_eq!(reopened.rows().await.unwrap(), before);
    let third = reopened
        .append(evidence_input(SCOPE, "e3", &payload, true, ""))
        .await
        .unwrap();
    assert_eq!(third.sequence, 3, "numbering resumes from the log");

    // The ids read from disk still deduplicate.
    let replay = reopened
        .append(evidence_input(SCOPE, "e1", &payload, true, ""))
        .await
        .unwrap();
    assert!(replay.duplicate);
    assert_eq!(replay.sequence, 1);
}

#[tokio::test]
async fn evidence_iter_after_is_cursor_exclusive_and_scope_isolated() {
    let temp = TempDir::new().unwrap();
    let store = EvidenceStore::new(temp.path());
    let payload = json!({"success": true});
    for id in ["e1", "e2", "e3"] {
        store
            .append(evidence_input("scope-a", id, &payload, true, ""))
            .await
            .unwrap();
        store
            .append(evidence_input("scope-b", id, &payload, true, ""))
            .await
            .unwrap();
    }

    let scope_a = store.iter_after("scope-a", 0).await.unwrap();
    assert_eq!(sequences(&scope_a), vec![1, 3, 5]);
    assert_eq!(
        scope_a
            .iter()
            .map(|row| row.evidence_id.as_str())
            .collect::<Vec<_>>(),
        vec!["e1", "e2", "e3"]
    );
    assert!(scope_a.iter().all(|row| row.graph_scope_id == "scope-a"));
    assert_eq!(
        sequences(&store.iter_after("scope-b", 0).await.unwrap()),
        vec![2, 4, 6]
    );

    assert_eq!(
        sequences(&store.iter_after("scope-a", 3).await.unwrap()),
        vec![5]
    );
    assert!(store.iter_after("scope-a", 5).await.unwrap().is_empty());
    assert_eq!(
        store.scope_ids().await.unwrap(),
        vec!["scope-a".to_string(), "scope-b".to_string()]
    );
}

#[tokio::test]
async fn revision_store_writes_the_documented_scope_layout() {
    let temp = TempDir::new().unwrap();
    let store = RevisionStore::new(temp.path());
    let edges = overlay_edges();
    let snapshot = publish(&store, SCOPE, &edges, "t1").await;

    let scope_root = store.scope_root(SCOPE);
    let scope_digest: String = stable_hash(&json!({"graph_scope_id": SCOPE}))
        .chars()
        .take(REVISION_HASH_CHARS)
        .collect();
    assert_eq!(
        scope_root,
        temp.path()
            .join("evolution")
            .join("scopes")
            .join(scope_digest)
    );
    assert!(snapshot.observation_revision.starts_with("observation-"));
    assert!(snapshot.merged_revision.starts_with("merged-"));
    assert_eq!(snapshot.graph_scope_id, SCOPE);
    assert_eq!(snapshot.static_revision, STATIC_REVISION);
    assert_eq!(snapshot.high_water_sequence, 7);

    let overlay_path = scope_root
        .join("observations")
        .join(&snapshot.observation_revision)
        .join("overlay.json");
    let manifest_path = scope_root
        .join("merged")
        .join(&snapshot.merged_revision)
        .join("manifest.json");
    let current_path = scope_root.join("current.json");
    let mut expected = vec![
        current_path.clone(),
        manifest_path.clone(),
        overlay_path.clone(),
    ];
    expected.sort();
    assert_eq!(files_under(&scope_root), expected);

    let current = read_document(&current_path);
    assert_eq!(
        current["observation_revision"],
        json!(snapshot.observation_revision)
    );
    assert_eq!(current["merged_revision"], json!(snapshot.merged_revision));

    let overlay = read_document(&overlay_path);
    assert_eq!(
        keys_of(&overlay),
        vec![
            "diagnostics",
            "edges",
            "generated_at",
            "graph_scope_id",
            "high_water_sequence",
            "observation_revision",
            "projection_status",
            "schema_version",
            "static_revision",
        ]
    );
    assert_eq!(overlay["schema_version"], json!(OBSERVATION_SCHEMA));
    assert_eq!(overlay["edges"], json!(edges));
    assert_eq!(overlay["projection_status"], json!("ready"));

    let manifest = read_document(&manifest_path);
    let pointer_keys = vec![
        "generated_at",
        "graph_scope_id",
        "high_water_sequence",
        "merged_revision",
        "observation_revision",
        "schema_version",
        "static_revision",
    ];
    assert_eq!(keys_of(&manifest), pointer_keys);
    assert_eq!(manifest["schema_version"], json!(MERGED_SCHEMA));
    assert_eq!(keys_of(&current), pointer_keys);
}

#[tokio::test]
async fn revision_ids_are_content_derived_and_ignore_generated_at() {
    let temp = TempDir::new().unwrap();
    let edges = overlay_edges();
    let first_store = RevisionStore::new(temp.path());
    let republished_store = RevisionStore::new(temp.path());

    let first = publish(&first_store, SCOPE, &edges, "t1").await;
    assert_eq!(
        first.observation_revision,
        format!(
            "observation-{}",
            short_digest(&overlay_document(SCOPE, &edges))
        )
    );
    assert_eq!(
        first.merged_revision,
        format!(
            "merged-{}",
            short_digest(&merged_document(SCOPE, &first.observation_revision))
        )
    );

    // Identical state from a second store with another timestamp: same ids.
    let again = publish(&republished_store, SCOPE, &edges, "t2").await;
    assert_eq!(again.observation_revision, first.observation_revision);
    assert_eq!(again.merged_revision, first.merged_revision);
    let pointer = first_store.current_pointer(SCOPE).await.unwrap().unwrap();
    assert_eq!(pointer.observation_revision, first.observation_revision);
    assert_eq!(pointer.merged_revision, first.merged_revision);
    assert_eq!(
        pointer.generated_at, "t2",
        "the pointer is rewritten; the revisions are not"
    );

    // Different aggregated content moves both ids.
    let mut changed = edges.clone();
    changed.insert(
        "skill:a->skill:c".to_string(),
        json!({"binding_status": "active_static", "runtime_weight": 2.0}),
    );
    let moved = publish(&first_store, SCOPE, &changed, "t1").await;
    assert_ne!(moved.observation_revision, first.observation_revision);
    assert_ne!(moved.merged_revision, first.merged_revision);
    assert_eq!(
        moved.merged_revision,
        format!(
            "merged-{}",
            short_digest(&merged_document(SCOPE, &moved.observation_revision))
        )
    );
}

#[tokio::test]
async fn republishing_a_changed_revision_document_is_rejected() {
    let temp = TempDir::new().unwrap();
    let store = RevisionStore::new(temp.path());
    let edges = overlay_edges();
    let diagnostics: Vec<String> = Vec::new();
    let input = OverlayInput {
        graph_scope_id: SCOPE,
        static_revision: STATIC_REVISION,
        high_water_sequence: 7,
        edges: &edges,
        generated_at: "t1",
        projection_status: "ready",
        diagnostics: &diagnostics,
    };
    let revision = store.publish_overlay(&input).await.unwrap();

    let path = store
        .scope_root(SCOPE)
        .join("observations")
        .join(&revision)
        .join("overlay.json");
    let mut document = read_document(&path);
    document["projection_status"] = json!("safe_fallback");
    std::fs::write(&path, serde_json::to_vec_pretty(&document).unwrap()).unwrap();

    let error = store
        .publish_overlay(&input)
        .await
        .expect_err("a published revision is immutable");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(
        error.message().contains("immutable Symphony revision"),
        "{}",
        error.message()
    );
    assert!(error.message().contains(&revision), "{}", error.message());
}

#[tokio::test]
async fn atomic_writes_leave_no_temporary_siblings() {
    let temp = TempDir::new().unwrap();
    let store = RevisionStore::new(temp.path());
    let edges = overlay_edges();
    let snapshot = publish(&store, SCOPE, &edges, "t1").await;
    store
        .write_current(&CurrentPointer::from_snapshot(&snapshot, "t9"))
        .await
        .unwrap();
    let evidence = EvidenceStore::new(temp.path());
    evidence
        .append(evidence_input(
            SCOPE,
            "e1",
            &json!({"success": true}),
            true,
            "",
        ))
        .await
        .unwrap();

    let files = files_under(temp.path());
    assert_eq!(files.len(), 4, "{files:?}");
    for path in &files {
        assert!(
            !path.to_string_lossy().ends_with(".tmp"),
            "a temporary sibling survived: {}",
            path.display()
        );
        let text = std::fs::read_to_string(path).unwrap();
        if path.to_string_lossy().ends_with(".jsonl") {
            for line in text.lines().filter(|line| !line.trim().is_empty()) {
                let row: Value = serde_json::from_str(line).expect("a JSONL row");
                assert!(row.is_object());
            }
        } else {
            let document: Value = serde_json::from_str(&text).expect("a JSON document");
            assert!(document.is_object());
        }
    }
}
