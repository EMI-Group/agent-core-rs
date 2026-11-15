//! The sanitised evidence payload appended to the observation log.
//!
//! Ports Python `openjiuwen.symphony.observation.service._qualify`'s `normalized` document
//! (v0.1.20) into a pure builder: the qualification rules live in
//! [`service`](crate::observation::service), and the normalized edges they accepted are
//! rendered here.
//!
//! # Persisted payloads carry hashes only
//!
//! Trajectory, session and request ids never reach the log verbatim: each one is stored as
//! its SHA-256 hex digest ([`private_hash`]), so a persisted payload can be compared and
//! aggregated without retaining an identity. The only free text in a payload is the
//! task query, and it is passed through [`redact_sensitive_text`] before it is stored —
//! a Rust-only addition, because the Python package drops the query entirely.
//!
//! # Document shape
//!
//! [`evidence_payload`] writes exactly these keys, in this order:
//!
//! | Key | Value |
//! | --- | --- |
//! | `schema_version` | [`GraphEvolutionInput::schema_version`] |
//! | `evidence_id` | the caller's evidence id |
//! | `graph_scope_id` | the caller's graph scope |
//! | `observed_at` | the canonical UTC text of `normalized()` |
//! | `static_revision` | the static revision the evidence was observed against |
//! | `trajectory_id_hash` | `private_hash(trajectory_id)` |
//! | `session_id_hash` | `private_hash(session_id)` |
//! | `request_id_hash` | `private_hash(request_id)`, or `null` |
//! | `capture_mode` | `agent` or `team` |
//! | `trace_refs` | the sorted, deduped union of `span_refs` and `member_trajectory_refs` |
//! | `task_outcome` | `{label, evidence_strength, failure_domain}` |
//! | `task_cluster_id` | the caller's cluster id, or `null` |
//! | `query` | the redacted query, present only when it is non-blank |
//! | `edges` | one object per [`QualifiedEdge`] |
//!
//! Each edge object is the point-edge identity ([`EdgeIdentity::to_value`], which carries
//! `identity_schema`) plus `edge_identity` (the identity hash the overlay is keyed by),
//! `outcome` (`success` or `failure`) and `observed_as_static`.

use std::collections::BTreeSet;

use serde_json::{Map, Value};

use crate::observation::contracts::GraphEvolutionInput;
use crate::observation::identity::{private_hash, EdgeIdentity};
use crate::redaction::redact_sensitive_text;

/// The `schema_version` key of a payload (also the overlay's first key).
pub(crate) const SCHEMA_VERSION_FIELD: &str = "schema_version";
/// The `evidence_id` key of a payload.
pub(crate) const EVIDENCE_ID_FIELD: &str = "evidence_id";
/// The `graph_scope_id` key of a payload (also the overlay's scope key).
pub(crate) const GRAPH_SCOPE_ID_FIELD: &str = "graph_scope_id";
/// The `observed_at` key of a payload.
pub(crate) const OBSERVED_AT_FIELD: &str = "observed_at";
/// The `static_revision` key of a payload.
pub(crate) const STATIC_REVISION_FIELD: &str = "static_revision";
/// The `trajectory_id_hash` key of a payload.
pub(crate) const TRAJECTORY_ID_HASH_FIELD: &str = "trajectory_id_hash";
/// The `session_id_hash` key of a payload.
pub(crate) const SESSION_ID_HASH_FIELD: &str = "session_id_hash";
/// The `request_id_hash` key of a payload.
pub(crate) const REQUEST_ID_HASH_FIELD: &str = "request_id_hash";
/// The `capture_mode` key of a payload.
pub(crate) const CAPTURE_MODE_FIELD: &str = "capture_mode";
/// The `trace_refs` key of a payload.
pub(crate) const TRACE_REFS_FIELD: &str = "trace_refs";
/// The `task_outcome` key of a payload.
pub(crate) const TASK_OUTCOME_FIELD: &str = "task_outcome";
/// The `task_cluster_id` key of a payload.
pub(crate) const TASK_CLUSTER_ID_FIELD: &str = "task_cluster_id";
/// The `query` key of a payload.
pub(crate) const QUERY_FIELD: &str = "query";
/// The `edges` key of a payload and of an overlay.
pub(crate) const EDGES_FIELD: &str = "edges";
/// The `identity_schema` key of a serialized point-edge identity.
pub(crate) const IDENTITY_SCHEMA_FIELD: &str = "identity_schema";
/// The `edge_identity` key of an edge object or overlay record.
pub(crate) const EDGE_IDENTITY_FIELD: &str = "edge_identity";
/// The `outcome` key of an edge object.
pub(crate) const OUTCOME_FIELD: &str = "outcome";
/// The `observed_as_static` key of an edge object.
pub(crate) const OBSERVED_AS_STATIC_FIELD: &str = "observed_as_static";
/// The `source_id` key of an edge object or overlay record.
pub(crate) const SOURCE_ID_FIELD: &str = "source_id";
/// The `target_id` key of an edge object or overlay record.
pub(crate) const TARGET_ID_FIELD: &str = "target_id";
/// The `relation_type` key of an edge object or overlay record.
pub(crate) const RELATION_TYPE_FIELD: &str = "relation_type";
/// The `source_content_hash` key of an edge object or overlay record.
pub(crate) const SOURCE_CONTENT_HASH_FIELD: &str = "source_content_hash";
/// The `target_content_hash` key of an edge object or overlay record.
pub(crate) const TARGET_CONTENT_HASH_FIELD: &str = "target_content_hash";

/// `value` as spelled in an edge object's `outcome` key.
pub(crate) const OUTCOME_SUCCESS: &str = "success";
/// `value` as spelled in an edge object's `outcome` key.
pub(crate) const OUTCOME_FAILURE: &str = "failure";

/// The qualified outcome of one observed transition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EdgeOutcome {
    /// The transition carried a success in a verified-success task.
    Success,
    /// The transition carried an orchestration-attributed failure.
    Failure,
}

impl EdgeOutcome {
    /// The wire spelling of this outcome.
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Success => OUTCOME_SUCCESS,
            Self::Failure => OUTCOME_FAILURE,
        }
    }
}

/// One observed transition that survived qualification.
///
/// `identity` is the point-edge identity derived against the static index the payload was
/// qualified with, and `observed_as_static` records whether that identity belonged to the
/// static graph at qualification time. Both are stored so aggregation never has to re-derive
/// an identity from a trace.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct QualifiedEdge {
    /// The point-edge identity of the transition.
    pub identity: EdgeIdentity,
    /// The qualified outcome of the transition.
    pub outcome: EdgeOutcome,
    /// Whether the identity matched a static `can_feed` edge when it was qualified.
    pub observed_as_static: bool,
}

/// Builds the normalized, sanitised payload persisted for one observation.
///
/// The document is deterministic: every collection is sorted (or `BTreeMap`-ordered), and
/// every id is hashed, so two submissions of the same evidence render byte-identical JSON.
/// Only the task query is free text, and only in redacted form.
pub(crate) fn evidence_payload(input: &GraphEvolutionInput, edges: &[QualifiedEdge]) -> Value {
    let mut payload = Map::new();
    payload.insert(
        SCHEMA_VERSION_FIELD.to_string(),
        Value::from(input.schema_version.clone()),
    );
    payload.insert(
        EVIDENCE_ID_FIELD.to_string(),
        Value::from(input.evidence_id.clone()),
    );
    payload.insert(
        GRAPH_SCOPE_ID_FIELD.to_string(),
        Value::from(input.graph_scope_id.clone()),
    );
    payload.insert(
        OBSERVED_AT_FIELD.to_string(),
        Value::from(input.observed_at.clone()),
    );
    payload.insert(
        STATIC_REVISION_FIELD.to_string(),
        Value::from(input.graph_snapshot.static_revision.clone()),
    );
    payload.insert(
        TRAJECTORY_ID_HASH_FIELD.to_string(),
        Value::from(private_hash(&input.trace.trajectory_id)),
    );
    payload.insert(
        SESSION_ID_HASH_FIELD.to_string(),
        Value::from(private_hash(&input.trace.session_id)),
    );
    payload.insert(
        REQUEST_ID_HASH_FIELD.to_string(),
        input
            .trace
            .request_id
            .as_deref()
            .map_or(Value::Null, |request_id| {
                Value::from(private_hash(request_id))
            }),
    );
    payload.insert(
        CAPTURE_MODE_FIELD.to_string(),
        Value::from(input.trace.capture_mode.clone()),
    );

    let refs: BTreeSet<&str> = input
        .trace
        .span_refs
        .iter()
        .chain(input.trace.member_trajectory_refs.iter())
        .map(String::as_str)
        .collect();
    payload.insert(
        TRACE_REFS_FIELD.to_string(),
        Value::Array(refs.into_iter().map(Value::from).collect()),
    );

    let mut outcome = Map::new();
    outcome.insert(
        "label".to_string(),
        Value::from(input.task.outcome.label.as_str()),
    );
    outcome.insert(
        "evidence_strength".to_string(),
        Value::from(input.task.outcome.evidence_strength.as_str()),
    );
    outcome.insert(
        "failure_domain".to_string(),
        input
            .task
            .outcome
            .failure_domain
            .map_or(Value::Null, |domain| Value::from(domain.as_str())),
    );
    payload.insert(TASK_OUTCOME_FIELD.to_string(), Value::Object(outcome));

    payload.insert(
        TASK_CLUSTER_ID_FIELD.to_string(),
        input
            .task
            .task_cluster_id
            .clone()
            .map_or(Value::Null, Value::from),
    );

    if let Some(query) = input
        .task
        .query
        .as_deref()
        .filter(|query| !query.trim().is_empty())
    {
        payload.insert(
            QUERY_FIELD.to_string(),
            Value::from(redact_sensitive_text(query)),
        );
    }

    payload.insert(
        EDGES_FIELD.to_string(),
        Value::Array(edges.iter().map(edge_object).collect()),
    );
    Value::Object(payload)
}

/// Renders one qualified edge: its identity document plus the outcome bookkeeping.
fn edge_object(edge: &QualifiedEdge) -> Value {
    let mut object = match edge.identity.to_value() {
        Value::Object(document) => document,
        // `EdgeIdentity::to_value` always renders an object; an unexpected shape degrades to
        // the bookkeeping keys alone rather than dropping the edge.
        _ => Map::new(),
    };
    object.insert(
        EDGE_IDENTITY_FIELD.to_string(),
        Value::from(edge.identity.identity_hash()),
    );
    object.insert(
        OUTCOME_FIELD.to_string(),
        Value::from(edge.outcome.as_str()),
    );
    object.insert(
        OBSERVED_AS_STATIC_FIELD.to_string(),
        Value::Bool(edge.observed_as_static),
    );
    Value::Object(object)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::observation::contracts::{
        EvidenceStrength, EvolutionEdgeMetadata, EvolutionGraph, EvolutionGraphEdge,
        EvolutionGraphNode, GraphSnapshotRef, TaskEvidence, TaskOutcome, TaskOutcomeLabel,
        TraceEvidence,
    };
    use crate::observation::identity::EdgeIdentity;

    /// Builds an input whose ids are recognizable, so a test can prove they are hashed.
    fn input() -> GraphEvolutionInput {
        GraphEvolutionInput {
            evidence_id: "evidence-1".to_string(),
            graph_scope_id: "default".to_string(),
            observed_at: "2026-03-01T00:00:00Z".to_string(),
            graph_snapshot: GraphSnapshotRef {
                static_revision: "v1".to_string(),
                observation_revision: "observation-seed".to_string(),
                merged_revision: None,
            },
            trace: TraceEvidence {
                trajectory_id: "trajectory-secret".to_string(),
                session_id: "session-secret".to_string(),
                capture_mode: "agent".to_string(),
                request_id: Some("request-secret".to_string()),
                truncated: false,
                quality_flags: Vec::new(),
                span_refs: vec!["span-b".to_string(), "span-a".to_string()],
                member_trajectory_refs: vec!["span-a".to_string(), "member-1".to_string()],
            },
            task: TaskEvidence {
                query: Some("password = hunter2".to_string()),
                task_cluster_id: Some("cluster-1".to_string()),
                outcome: TaskOutcome {
                    label: TaskOutcomeLabel::VerifiedSuccess,
                    evidence_strength: EvidenceStrength::Strong,
                    failure_domain: None,
                },
            },
            planned_graph: None,
            execution_graph: EvolutionGraph {
                graph_type: "execution_graph".to_string(),
                nodes: [
                    ("skill:a".to_string(), EvolutionGraphNode::default()),
                    ("skill:b".to_string(), EvolutionGraphNode::default()),
                ]
                .into_iter()
                .collect(),
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
            },
            ..GraphEvolutionInput::default()
        }
    }

    fn edge() -> QualifiedEdge {
        QualifiedEdge {
            identity: EdgeIdentity::new("a", "b", "can_feed", "hash-a", "hash-b"),
            outcome: EdgeOutcome::Success,
            observed_as_static: true,
        }
    }

    #[test]
    fn identifiers_are_hashed_and_free_text_is_redacted() {
        let payload = evidence_payload(&input(), &[edge()]);
        let rendered = payload.to_string();
        for raw in [
            "trajectory-secret",
            "session-secret",
            "request-secret",
            "hunter2",
        ] {
            assert!(
                !rendered.contains(raw),
                "payload leaked `{raw}`: {rendered}"
            );
        }
        assert_eq!(
            payload[SESSION_ID_HASH_FIELD],
            json!(private_hash("session-secret"))
        );
        assert_eq!(
            payload[REQUEST_ID_HASH_FIELD],
            json!(private_hash("request-secret"))
        );
        assert_eq!(payload[QUERY_FIELD], json!("password=<redacted>"));
    }

    #[test]
    fn document_keys_and_reference_ordering_are_pinned() {
        let payload = evidence_payload(&input(), &[edge()]);
        let object = payload.as_object().unwrap();
        let keys: Vec<&str> = object.keys().map(String::as_str).collect();
        assert_eq!(
            keys,
            vec![
                SCHEMA_VERSION_FIELD,
                EVIDENCE_ID_FIELD,
                GRAPH_SCOPE_ID_FIELD,
                OBSERVED_AT_FIELD,
                STATIC_REVISION_FIELD,
                TRAJECTORY_ID_HASH_FIELD,
                SESSION_ID_HASH_FIELD,
                REQUEST_ID_HASH_FIELD,
                CAPTURE_MODE_FIELD,
                TRACE_REFS_FIELD,
                TASK_OUTCOME_FIELD,
                TASK_CLUSTER_ID_FIELD,
                QUERY_FIELD,
                EDGES_FIELD,
            ]
        );
        assert_eq!(
            payload[TRACE_REFS_FIELD],
            json!(["member-1", "span-a", "span-b"])
        );
        assert_eq!(
            payload[TASK_OUTCOME_FIELD],
            json!({
                "label": "verified_success",
                "evidence_strength": "strong",
                "failure_domain": null,
            })
        );
        assert_eq!(payload[TASK_CLUSTER_ID_FIELD], json!("cluster-1"));
        assert_eq!(
            payload[EDGES_FIELD],
            json!([{
                "identity_schema": "symphony.point-edge.v1",
                "source_id": "a",
                "target_id": "b",
                "relation_type": "can_feed",
                "source_content_hash": "hash-a",
                "target_content_hash": "hash-b",
                "edge_identity": edge().identity.identity_hash(),
                "outcome": "success",
                "observed_as_static": true,
            }])
        );
    }

    #[test]
    fn absent_optional_fields_render_as_null_or_disappear() {
        let mut input = input();
        input.trace.request_id = None;
        input.task.query = Some("   ".to_string());
        input.task.task_cluster_id = None;
        let payload = evidence_payload(&input, &[]);
        assert!(payload[REQUEST_ID_HASH_FIELD].is_null());
        assert_eq!(payload[TASK_CLUSTER_ID_FIELD], Value::Null);
        assert!(payload.get(QUERY_FIELD).is_none());
        assert_eq!(payload[EDGES_FIELD], json!([]));
    }
}
