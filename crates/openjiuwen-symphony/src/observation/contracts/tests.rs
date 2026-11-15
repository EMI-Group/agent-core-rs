use super::*;
use crate::error::ErrorKind;

/// A compliant execution node carrying the four allow-listed metadata keys.
fn execution_node() -> EvolutionGraphNode {
    let mut metadata = BTreeMap::new();
    metadata.insert(
        "capability_type".to_string(),
        Value::String(SKILL_LABEL.to_string()),
    );
    metadata.insert("version".to_string(), Value::String("1.0.0".to_string()));
    metadata.insert(
        "description".to_string(),
        Value::String("summarizes text".to_string()),
    );
    metadata.insert(
        "inputs".to_string(),
        serde_json::json!([{ "name": "text", "type": "string", "required": true }]),
    );
    metadata.insert(
        "outputs".to_string(),
        serde_json::json!([{ "name": "summary", "type": "string" }]),
    );
    EvolutionGraphNode {
        label: SKILL_LABEL.to_string(),
        metadata,
    }
}

/// A compliant two-node execution graph with one qualified edge.
fn execution_graph() -> EvolutionGraph {
    let mut nodes = BTreeMap::new();
    nodes.insert("summarizer".to_string(), execution_node());
    nodes.insert("rewriter".to_string(), execution_node());
    EvolutionGraph {
        id: Some("exec-1".to_string()),
        graph_type: EXECUTION_GRAPH_TYPE.to_string(),
        directed: true,
        nodes,
        edges: vec![EvolutionGraphEdge {
            source_id: "summarizer".to_string(),
            target_id: "rewriter".to_string(),
            relation_type: RELATION_CAN_FEED.to_string(),
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

/// Asserts that validation fails with a `Validation` error naming `field`.
fn assert_rejected(input: &GraphEvolutionInput, field: &str) {
    let error = input.validate().unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(
        error.message().starts_with(field),
        "`{}` does not name `{field}`",
        error.message()
    );
}

/// Returns the first execution node (the map is keyed by id, so this is stable).
fn first_node_mut(input: &mut GraphEvolutionInput) -> &mut EvolutionGraphNode {
    input
        .execution_graph
        .nodes
        .get_mut("summarizer")
        .expect("summarizer node")
}

#[test]
fn enums_render_their_wire_spellings() {
    assert_eq!(
        TaskOutcomeLabel::VerifiedSuccess.as_str(),
        "verified_success"
    );
    assert_eq!(
        TaskOutcomeLabel::VerifiedFailure.as_str(),
        "verified_failure"
    );
    assert_eq!(EvidenceStrength::None.as_str(), "none");
    assert_eq!(FailureDomain::ExternalService.as_str(), "external_service");
    assert_eq!(FailureDomain::SkillInput.as_str(), "skill_input");
    assert_eq!(ObservationStatus::AuditOnly.as_str(), "audit_only");
    assert_eq!(ObservationStatus::Duplicate.as_str(), "duplicate");
    assert_eq!(
        serde_json::to_value(TaskOutcomeLabel::Partial).unwrap(),
        Value::String("partial".to_string())
    );
}

#[test]
fn defaults_match_the_python_model_defaults() {
    let graph: EvolutionGraph = serde_json::from_str(r#"{"type":"execution_graph"}"#).unwrap();
    assert!(graph.directed);
    assert!(graph.edges.is_empty());
    assert!(graph.nodes.is_empty());
    let edge: EvolutionGraphEdge = serde_json::from_str(r#"{"source":"a","target":"b"}"#).unwrap();
    assert_eq!(edge.relation_type, RELATION_CAN_FEED);
    assert_eq!(edge.metadata, EvolutionEdgeMetadata::default());
    let node: EvolutionGraphNode = serde_json::from_str("{}").unwrap();
    assert_eq!(node.label, SKILL_LABEL);
    let input: GraphEvolutionInput = serde_json::from_str(r#"{"evidence_id":"ev-1"}"#).unwrap();
    assert_eq!(input.schema_version, GRAPH_EVOLUTION_INPUT_SCHEMA);
    assert_eq!(input.graph_scope_id, "default");
    assert!(EvolutionGraph::default().directed);
    assert_eq!(EvolutionGraphNode::default().label, SKILL_LABEL);
    assert_eq!(
        EvolutionGraphEdge::default().relation_type,
        RELATION_CAN_FEED
    );
    assert_eq!(ObservationStatus::default(), ObservationStatus::Accepted);
    assert_eq!(TaskOutcomeLabel::default(), TaskOutcomeLabel::Unknown);
}

#[test]
fn graph_round_trips_through_serde_with_aliases() {
    let raw = serde_json::json!({
        "id": "exec-1",
        "type": "execution_graph",
        "nodes": {
            "summarizer": { "label": "skill", "metadata": { "capability_type": "skill" } },
            "rewriter": {}
        },
        "edges": [{ "source": "summarizer", "target": "rewriter", "relation": "can_feed" }],
    });
    let graph: EvolutionGraph = serde_json::from_value(raw).unwrap();
    assert_eq!(graph.edges[0].source_id, "summarizer");
    assert_eq!(graph.edges[0].target_id, "rewriter");
    assert_eq!(graph.edges[0].relation_type, RELATION_CAN_FEED);
    let rendered = serde_json::to_value(&graph).unwrap();
    assert_eq!(
        rendered["type"],
        Value::String(EXECUTION_GRAPH_TYPE.to_string())
    );
    assert!(rendered["edges"][0].get("source_id").is_some());
    assert!(rendered["edges"][0].get("source").is_none());
    assert!(rendered["edges"][0].get("relation").is_none());
    assert_eq!(
        serde_json::from_value::<EvolutionGraph>(rendered).unwrap(),
        graph
    );
}

#[test]
fn serde_tolerates_unknown_fields() {
    let mut payload = serde_json::to_value(valid_input()).unwrap();
    payload["future_field"] = serde_json::json!({ "nested": true });
    payload["trace"]["future_trace_field"] = serde_json::json!(1);
    let parsed: GraphEvolutionInput = serde_json::from_value(payload).unwrap();
    assert_eq!(parsed.evidence_id, "ev-1");
    parsed.validate().unwrap();
}

#[test]
fn validate_accepts_a_fully_formed_input() {
    valid_input().validate().unwrap();
}

#[test]
fn rejects_a_wrong_schema_version() {
    let mut input = valid_input();
    input.schema_version = "symphony.graph_evolution_input.v1".to_string();
    assert_rejected(&input, "schema_version");
}

#[test]
fn rejects_blank_contract_text() {
    let mut input = valid_input();
    input.evidence_id = String::new();
    assert_rejected(&input, "evidence_id");

    let mut input = valid_input();
    input.graph_scope_id = " padded ".to_string();
    assert_rejected(&input, "graph_scope_id");

    let mut input = valid_input();
    input.graph_snapshot.static_revision = String::new();
    assert_rejected(&input, "graph_snapshot.static_revision");

    let mut input = valid_input();
    input.graph_snapshot.merged_revision = Some(String::new());
    assert_rejected(&input, "graph_snapshot.merged_revision");

    let mut input = valid_input();
    input.trace.session_id = String::new();
    assert_rejected(&input, "trace.session_id");

    let mut input = valid_input();
    input.trace.capture_mode = String::new();
    assert_rejected(&input, "trace.capture_mode");

    let mut input = valid_input();
    input.trace.quality_flags = vec!["ok".to_string(), "bad\u{200B}flag".to_string()];
    assert_rejected(&input, "trace.quality_flags");

    let mut input = valid_input();
    input.trace.span_refs = vec![String::new()];
    assert_rejected(&input, "trace.span_refs");

    let mut input = valid_input();
    input.trace.member_trajectory_refs = vec!["s/2".to_string(), "s/3 ".to_string()];
    assert_rejected(&input, "trace.member_trajectory_refs");

    let mut input = valid_input();
    input.trace.trajectory_id = "line\nbreak".to_string();
    assert_rejected(&input, "trace.trajectory_id");
}

#[test]
fn rejects_a_missing_timezone() {
    let mut input = valid_input();
    input.observed_at = "2024-03-01T08:30:00".to_string();
    assert_rejected(&input, "observed_at");
    let error = input.validate().unwrap_err();
    assert!(error.message().contains("must include a timezone"));
}

#[test]
fn rejects_a_wrong_graph_type() {
    let mut input = valid_input();
    input.execution_graph.graph_type = PLANNED_GRAPH_TYPE.to_string();
    assert_rejected(&input, "execution_graph.type");

    let mut input = valid_input();
    input.planned_graph = Some(EvolutionGraph {
        graph_type: EXECUTION_GRAPH_TYPE.to_string(),
        ..EvolutionGraph::default()
    });
    assert_rejected(&input, "planned_graph.type");

    let mut input = valid_input();
    input.planned_graph = Some(EvolutionGraph {
        graph_type: PLANNED_GRAPH_TYPE.to_string(),
        ..EvolutionGraph::default()
    });
    input.validate().unwrap();
}

#[test]
fn rejects_a_non_skill_node_label() {
    let mut input = valid_input();
    first_node_mut(&mut input).label = "tool".to_string();
    assert_rejected(&input, "execution_graph.nodes");
}

#[test]
fn rejects_invalid_node_metadata() {
    let mut input = valid_input();
    first_node_mut(&mut input)
        .metadata
        .insert("owner".to_string(), Value::String("ops".to_string()));
    assert_rejected(&input, "execution_graph.nodes.summarizer.metadata");

    let mut input = valid_input();
    first_node_mut(&mut input).metadata.insert(
        "capability_type".to_string(),
        Value::String("tool".to_string()),
    );
    assert_rejected(&input, "execution_graph.nodes.summarizer.metadata");

    let mut input = valid_input();
    first_node_mut(&mut input)
        .metadata
        .insert("version".to_string(), Value::String("  ".to_string()));
    assert_rejected(&input, "execution_graph.nodes.summarizer.metadata");

    let mut input = valid_input();
    first_node_mut(&mut input).metadata.remove("version");
    assert_rejected(&input, "execution_graph.nodes.summarizer.metadata");

    let mut input = valid_input();
    first_node_mut(&mut input)
        .metadata
        .insert("description".to_string(), serde_json::json!(7));
    assert_rejected(
        &input,
        "execution_graph.nodes.summarizer.metadata.description",
    );

    let mut input = valid_input();
    first_node_mut(&mut input)
        .metadata
        .insert("inputs".to_string(), Value::String("text".to_string()));
    assert_rejected(&input, "execution_graph.nodes.summarizer.metadata.inputs");
}

#[test]
fn rejects_bad_capability_ports() {
    let port_cases = [
        serde_json::json!([{ "name": "text", "type": "string", "required": "yes" }]),
        serde_json::json!([{ "name": "text", "type": "string", "format": "uuid" }]),
        serde_json::json!([{ "name": "  ", "type": "string" }]),
        serde_json::json!([{ "name": "text", "type": "" }]),
        serde_json::json!([{ "name": "text" }]),
        serde_json::json!([{ "name": "text", "type": "string", "description": 7 }]),
        serde_json::json!(["text"]),
        serde_json::json!("text"),
    ];
    for ports in port_cases {
        let mut input = valid_input();
        first_node_mut(&mut input)
            .metadata
            .insert("outputs".to_string(), ports);
        assert_rejected(&input, "execution_graph.nodes.summarizer.metadata.outputs");
    }
    let mut input = valid_input();
    first_node_mut(&mut input).metadata.insert(
        "inputs".to_string(),
        serde_json::json!([{ "name": "text", "type": "string", "description": "the text" }]),
    );
    input.validate().unwrap();
}

#[test]
fn rejects_an_edge_with_a_missing_endpoint() {
    let mut input = valid_input();
    input.execution_graph.edges.push(EvolutionGraphEdge {
        source_id: "summarizer".to_string(),
        target_id: "missing".to_string(),
        relation_type: RELATION_CAN_FEED.to_string(),
        metadata: EvolutionEdgeMetadata::default(),
    });
    assert_rejected(&input, "execution_graph.edges");

    let mut input = valid_input();
    input
        .execution_graph
        .edges
        .push(EvolutionGraphEdge::default());
    assert_rejected(&input, "execution_graph.edges");
}

#[test]
fn rejects_verified_failure_without_a_failure_domain() {
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
fn normalized_rewrites_observed_at_and_ports() {
    let normalized = valid_input().normalized().unwrap();
    assert_eq!(normalized.observed_at, "2024-03-01T00:30:00Z");
    let node = &normalized.execution_graph.nodes["summarizer"];
    assert_eq!(
        node.metadata["inputs"],
        serde_json::json!([{
            "name": "text",
            "type": "string",
            "required": true,
            "description": "",
        }])
    );
    assert_eq!(
        node.metadata["outputs"],
        serde_json::json!([{
            "name": "summary",
            "type": "string",
            "required": false,
            "description": "",
        }])
    );
    assert_eq!(node.metadata["version"], Value::String("1.0.0".to_string()));
    assert!(normalized.planned_graph.is_none());
    assert_eq!(normalized.evidence_id, "ev-1");
}

#[test]
fn normalized_keeps_absent_ports_absent_and_rejects_invalid_input() {
    let mut input = valid_input();
    first_node_mut(&mut input).metadata.remove("inputs");
    let normalized = input.normalized().unwrap();
    assert!(!normalized.execution_graph.nodes["summarizer"]
        .metadata
        .contains_key("inputs"));

    let mut broken = valid_input();
    broken.observed_at = String::new();
    assert!(broken.normalized().is_err());
}

#[test]
fn trace_bridges_cover_success_and_failure_records() {
    let success = TraceRecord {
        trace_id: "s/1".to_string(),
        session_id: "s".to_string(),
        request_id: "1".to_string(),
        query: "summarize this".to_string(),
        success: true,
        ..TraceRecord::default()
    };
    let evidence = TraceEvidence::from_trace_record(&success);
    assert_eq!(evidence.trajectory_id, "s/1");
    assert_eq!(evidence.session_id, "s");
    assert_eq!(evidence.capture_mode, "agent");
    assert_eq!(evidence.request_id.as_deref(), Some("1"));
    assert!(!evidence.truncated);
    assert!(evidence.quality_flags.is_empty());
    assert!(evidence.span_refs.is_empty());
    assert!(evidence.member_trajectory_refs.is_empty());
    assert_eq!(
        TraceEvidence::from_trace_record_with_mode(&success, "team").capture_mode,
        "team"
    );

    let outcome = TaskOutcome::from_trace_record(&success);
    assert_eq!(outcome.label, TaskOutcomeLabel::VerifiedSuccess);
    assert_eq!(outcome.evidence_strength, EvidenceStrength::Strong);
    assert_eq!(outcome.failure_domain, None);
    let task = TaskEvidence::from_trace_record(&success);
    assert_eq!(task.query.as_deref(), Some("summarize this"));
    assert_eq!(task.task_cluster_id, None);
    assert_eq!(task.outcome, outcome);

    let failure = TraceRecord {
        trace_id: "s/2".to_string(),
        session_id: "s".to_string(),
        request_id: "   ".to_string(),
        query: String::new(),
        success: false,
        ..TraceRecord::default()
    };
    let failed = TaskOutcome::from_trace_record(&failure);
    assert_eq!(failed.label, TaskOutcomeLabel::VerifiedFailure);
    assert_eq!(failed.evidence_strength, EvidenceStrength::Strong);
    assert_eq!(failed.failure_domain, Some(FailureDomain::Unknown));
    assert_eq!(TaskEvidence::from_trace_record(&failure).query, None);
    assert_eq!(TraceEvidence::from_trace_record(&failure).request_id, None);

    let mut input = valid_input();
    input.trace = TraceEvidence::from_trace_record(&success);
    input.task = TaskEvidence::from_trace_record(&success);
    input.validate().unwrap();
    let mut input = valid_input();
    input.trace = TraceEvidence::from_trace_record(&failure);
    input.task = TaskEvidence::from_trace_record(&failure);
    input.validate().unwrap();
}

#[test]
fn receipts_and_snapshots_round_trip() {
    let receipt = ObservationReceipt {
        evidence_id: "ev-1".to_string(),
        graph_scope_id: "scope".to_string(),
        sequence: 3,
        status: ObservationStatus::AuditOnly,
        reason: TRUNCATED_TRACE.to_string(),
    };
    let rendered = serde_json::to_value(&receipt).unwrap();
    assert_eq!(rendered["status"], Value::String("audit_only".to_string()));
    assert_eq!(
        serde_json::from_value::<ObservationReceipt>(rendered).unwrap(),
        receipt
    );

    let mut overlay = BTreeMap::new();
    overlay.insert("runtime_weight".to_string(), serde_json::json!(1.5));
    let snapshot = GraphSnapshot {
        graph_scope_id: "scope".to_string(),
        static_revision: "v1".to_string(),
        observation_revision: "observation-abc".to_string(),
        merged_revision: "merged-def".to_string(),
        high_water_sequence: 3,
        overlay,
    };
    let rendered = serde_json::to_value(&snapshot).unwrap();
    assert_eq!(rendered["high_water_sequence"], serde_json::json!(3));
    assert_eq!(
        serde_json::from_value::<GraphSnapshot>(rendered).unwrap(),
        snapshot
    );
    assert_eq!(GraphSnapshot::default().high_water_sequence, 0);
}
