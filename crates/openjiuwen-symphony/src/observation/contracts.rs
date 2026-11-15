//! The canonical evidence contract for Symphony graph observations.
//!
//! Ports Python `openjiuwen.symphony.observation.contracts` (v0.1.20): the
//! [`GraphEvolutionInput`] payload an evolution rail or evaluator hands to the observation
//! layer, plus its evidence models (`GraphSnapshotRef`, [`TraceEvidence`],
//! [`TaskEvidence`], [`TaskOutcome`]), the JGF-like [`EvolutionGraph`] it carries, and the
//! result models ([`ObservationReceipt`], [`GraphSnapshot`]).
//!
//! Evidence is always *caller supplied*: nothing here infers a task outcome, a failure
//! domain, or an edge result from a trace. The rail resolves those and the contract only
//! checks that the payload is internally consistent.
//!
//! # Contract rules
//!
//! [`GraphEvolutionInput::validate`] mirrors Python's `model_validator`s field by field and
//! never panics:
//!
//! - `schema_version` must equal [`GRAPH_EVOLUTION_INPUT_SCHEMA`].
//! - `observed_at` must be an RFC 3339 timestamp that carries a timezone; Python's
//!   tz-aware `datetime` is text here, and [`GraphEvolutionInput::normalized`] rewrites it
//!   to the canonical UTC `...Z` form.
//! - `execution_graph.graph_type` must be `execution_graph` and a `planned_graph`, when
//!   present, must be typed `planned_graph`.
//! - every execution node is a `skill` whose metadata is allow-listed to
//!   `{capability_type, version, description, inputs, outputs}`, with ports allow-listed to
//!   `{name, type, required, description}`.
//! - every edge endpoint must exist in `execution_graph.nodes`.
//! - `verified_failure` requires a `failure_domain`.
//! - ids, modes and every reference list entry must be *contract text*: non-empty, already
//!   trimmed, and free of control (`Cc`) or format (`Cf`) characters.
//!
//! # Deliberate differences from Python
//!
//! - **Unknown fields are tolerated.** Python's models are strict (`extra="forbid"`); this
//!   crate keeps every data type tolerant of unknown fields so a newer rail can talk to an
//!   older observation layer. [`GraphEvolutionInput::validate`] therefore reports only the
//!   rules Python can express against *known* fields.
//! - **`observed_at` is text, not a `datetime`.** No clock type is involved, and the
//!   fractional digits survive a round trip verbatim.
//! - **Failure domains are never derived.** [`TaskOutcome::from_trace_record`] can only say
//!   `unknown`, because a real domain is an upstream verdict.
//! - **Empty collections serialise as `[]`/`{}`.** Python's pydantic models use tuples and
//!   dicts with the same JSON shape; nothing here is order dependent because node maps and
//!   metadata maps are `BTreeMap`s.

use std::collections::BTreeMap;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{validation, Result};
use crate::experience::trace::TraceRecord;
use crate::observation::timestamp;

/// The schema version this contract accepts (`GraphEvolutionInput.schema_version`).
pub const GRAPH_EVOLUTION_INPUT_SCHEMA: &str = "symphony.graph_evolution_input.v2";

/// Refusal reason: the trace was truncated, so its edge outcomes are not trustworthy.
pub const TRUNCATED_TRACE: &str = "truncated_trace";

/// Refusal reason: the trace carried quality flags that disqualify it as evidence.
pub const TRACE_QUALITY_FLAGS: &str = "trace_quality_flags";

/// Refusal reason: the task outcome is not strong enough to move an edge weight.
pub const OUTCOME_NOT_STRONG: &str = "outcome_not_strong";

/// Refusal reason: a node's identity metadata does not match the static capability graph.
pub const CAPABILITY_IDENTITY_MISMATCH: &str = "capability_identity_mismatch";

/// Refusal reason: evidence named a static revision the observation layer has never seen.
pub const UNKNOWN_STATIC_REVISION: &str = "unknown_static_revision";

/// Refusal reason: evidence pinned a merged revision that does not exist.
pub const UNKNOWN_MERGED_REVISION: &str = "unknown_merged_revision";

/// Refusal reason: the pinned merged revision was built on a different static revision.
pub const MERGED_STATIC_REVISION_MISMATCH: &str = "merged_static_revision_mismatch";

/// Refusal reason: the pinned merged revision was built on a different observation revision.
pub const MERGED_OBSERVATION_REVISION_MISMATCH: &str = "merged_observation_revision_mismatch";

/// Refusal reason: no edge carried a qualified outcome, so nothing can be aggregated.
pub const NO_QUALIFIED_EDGE_OUTCOME: &str = "no_qualified_edge_outcome";

/// The `projection_status` of a neutral overlay published after a projection failure.
pub const SAFE_FALLBACK: &str = "safe_fallback";

/// The only node label an execution graph may use.
const SKILL_LABEL: &str = "skill";

/// The default relation type of an observed edge.
const RELATION_CAN_FEED: &str = "can_feed";

/// The only graph type an execution graph may use.
const EXECUTION_GRAPH_TYPE: &str = "execution_graph";

/// The only graph type a planned graph may use.
const PLANNED_GRAPH_TYPE: &str = "planned_graph";

/// The metadata keys an execution node may carry.
const EXECUTION_NODE_METADATA_FIELDS: [&str; 5] = [
    "capability_type",
    "version",
    "description",
    "inputs",
    "outputs",
];

/// The keys a capability port object may carry.
const CAPABILITY_PORT_FIELDS: [&str; 4] = ["name", "type", "required", "description"];

/// The default capture mode of [`TraceEvidence::from_trace_record`].
const DEFAULT_CAPTURE_MODE: &str = "agent";

/// How strongly an upstream component has verified an outcome.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceStrength {
    /// A rail or evaluator verified the outcome directly.
    Strong,
    /// The outcome is plausible but was inferred rather than verified.
    Weak,
    /// No evidence about the outcome was produced.
    #[default]
    None,
}

impl EvidenceStrength {
    /// The wire spelling of this variant.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Strong => "strong",
            Self::Weak => "weak",
            Self::None => "none",
        }
    }
}

/// A canonical task outcome produced by an evolution rail or evaluator.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum TaskOutcomeLabel {
    /// The task completed and was verified as a success.
    VerifiedSuccess,
    /// The task failed and the failure was verified.
    VerifiedFailure,
    /// The task partially succeeded, so its edges are neither credited nor blamed.
    Partial,
    /// The task was cancelled; cancellation is not evidence of a wrong edge.
    Cancelled,
    /// No verdict was produced.
    #[default]
    Unknown,
}

impl TaskOutcomeLabel {
    /// The wire spelling of this variant.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::VerifiedSuccess => "verified_success",
            Self::VerifiedFailure => "verified_failure",
            Self::Partial => "partial",
            Self::Cancelled => "cancelled",
            Self::Unknown => "unknown",
        }
    }
}

/// The failure domain that decides whether a failure may update graph weights.
///
/// Only [`FailureDomain::Orchestration`] failures qualify an edge: input, permission,
/// network, external-service, and timeout failures blame something other than the edge.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum FailureDomain {
    /// The edge itself was mis-ordered or mis-supplied.
    Orchestration,
    /// The upstream step produced an input the downstream skill could not accept.
    SkillInput,
    /// The skill ran but its own logic failed.
    SkillLogic,
    /// A permission or authorization boundary was hit.
    Permission,
    /// The network transport failed.
    Network,
    /// A third-party service failed.
    ExternalService,
    /// A deadline expired.
    Timeout,
    /// The run was cancelled.
    Cancelled,
    /// The failure was not classified.
    #[default]
    Unknown,
}

impl FailureDomain {
    /// The wire spelling of this variant.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Orchestration => "orchestration",
            Self::SkillInput => "skill_input",
            Self::SkillLogic => "skill_logic",
            Self::Permission => "permission",
            Self::Network => "network",
            Self::ExternalService => "external_service",
            Self::Timeout => "timeout",
            Self::Cancelled => "cancelled",
            Self::Unknown => "unknown",
        }
    }
}

/// The disposition of one appended evidence row.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ObservationStatus {
    /// The row is eligible and will be aggregated into the next revision.
    #[default]
    Accepted,
    /// The row was recorded for audit only and never influences weights.
    AuditOnly,
    /// The row repeated an already recorded `(graph_scope_id, evidence_id)`.
    Duplicate,
}

impl ObservationStatus {
    /// The wire spelling of this variant.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Accepted => "accepted",
            Self::AuditOnly => "audit_only",
            Self::Duplicate => "duplicate",
        }
    }
}

/// The static and observation revisions a task was observed against.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct GraphSnapshotRef {
    /// The static `can_feed` graph revision the evidence was qualified against.
    pub static_revision: String,
    /// The observation revision that was current when the task ran.
    pub observation_revision: String,
    /// The merged revision the planner pinned, when the caller pinned one.
    pub merged_revision: Option<String>,
}

/// One JGF node in a planned or observed execution graph.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct EvolutionGraphNode {
    /// The node kind; an execution node is always `skill`.
    #[serde(default = "default_skill_label")]
    pub label: String,
    /// The allow-listed capability snapshot the rail attached to the node.
    pub metadata: BTreeMap<String, Value>,
}

impl Default for EvolutionGraphNode {
    fn default() -> Self {
        Self {
            label: SKILL_LABEL.to_string(),
            metadata: BTreeMap::new(),
        }
    }
}

/// The minimal edge outcome produced by the rail.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct EvolutionEdgeMetadata {
    /// Whether the transition succeeded, when the rail could tell.
    pub success: Option<bool>,
    /// The failure domain of a failed transition, when the rail attributed one.
    pub failure_domain: Option<FailureDomain>,
}

/// An observed transition plus its optional edge-level outcome evidence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct EvolutionGraphEdge {
    /// The id of the node the transition left (JSON alias `source`).
    #[serde(alias = "source")]
    pub source_id: String,
    /// The id of the node the transition entered (JSON alias `target`).
    #[serde(alias = "target")]
    pub target_id: String,
    /// The relation kind; `can_feed` unless the rail said otherwise (JSON alias `relation`).
    #[serde(default = "default_relation_type", alias = "relation")]
    pub relation_type: String,
    /// The outcome the rail observed for this transition.
    pub metadata: EvolutionEdgeMetadata,
}

impl Default for EvolutionGraphEdge {
    fn default() -> Self {
        Self {
            source_id: String::new(),
            target_id: String::new(),
            relation_type: RELATION_CAN_FEED.to_string(),
            metadata: EvolutionEdgeMetadata::default(),
        }
    }
}

/// A JGF-compatible graph exchanged between a rail, an evaluator, and the graph engine.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct EvolutionGraph {
    /// The graph id, when the producer assigned one.
    pub id: Option<String>,
    /// The graph kind: `planned_graph` or `execution_graph` as validated.
    #[serde(rename = "type")]
    pub graph_type: String,
    /// Whether the edges are directed; observation graphs always are.
    #[serde(default = "default_directed")]
    pub directed: bool,
    /// The nodes, keyed by capability id.
    pub nodes: BTreeMap<String, EvolutionGraphNode>,
    /// The observed transitions, in the order the rail reported them.
    pub edges: Vec<EvolutionGraphEdge>,
}

impl Default for EvolutionGraph {
    fn default() -> Self {
        Self {
            id: None,
            graph_type: String::new(),
            directed: true,
            nodes: BTreeMap::new(),
            edges: Vec::new(),
        }
    }
}

/// Trace identity and quality metadata, without coupling to raw session JSON.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct TraceEvidence {
    /// The trajectory the evidence was distilled from.
    pub trajectory_id: String,
    /// The session that trajectory belongs to.
    pub session_id: String,
    /// The capture mode, `agent` or `team` in Python.
    pub capture_mode: String,
    /// The `request_id` segment inside the session, when known.
    pub request_id: Option<String>,
    /// Whether the trace was truncated, which disqualifies its outcomes.
    pub truncated: bool,
    /// Quality flags raised while capturing the trace.
    pub quality_flags: Vec<String>,
    /// References to the spans the evidence came from.
    pub span_refs: Vec<String>,
    /// References to member trajectories, for team captures.
    pub member_trajectory_refs: Vec<String>,
}

impl TraceEvidence {
    /// Builds evidence from an ingested trace record using the `agent` capture mode.
    ///
    /// Python accepts `agent` and `team`; `agent` is the default here because a single
    /// parsed session record is one agent run. The record's `request_id` is only carried
    /// when it is non-blank, and nothing is marked truncated: a parsed record is complete
    /// by construction.
    #[must_use]
    pub fn from_trace_record(record: &TraceRecord) -> Self {
        Self::from_trace_record_with_mode(record, DEFAULT_CAPTURE_MODE)
    }

    /// Builds evidence from an ingested trace record with an explicit `capture_mode`.
    #[must_use]
    pub fn from_trace_record_with_mode(
        record: &TraceRecord,
        capture_mode: impl Into<String>,
    ) -> Self {
        Self {
            trajectory_id: record.trace_id.clone(),
            session_id: record.session_id.clone(),
            capture_mode: capture_mode.into(),
            request_id: non_blank(&record.request_id),
            truncated: false,
            quality_flags: Vec::new(),
            span_refs: Vec::new(),
            member_trajectory_refs: Vec::new(),
        }
    }
}

/// The task-level verdict resolved outside the graph engine.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct TaskOutcome {
    /// The verified outcome label.
    pub label: TaskOutcomeLabel,
    /// How strongly the outcome was verified.
    pub evidence_strength: EvidenceStrength,
    /// The failure domain; required when the label is `verified_failure`.
    pub failure_domain: Option<FailureDomain>,
}

impl TaskOutcome {
    /// Derives the conservative outcome of an ingested trace record.
    ///
    /// A successful record is a strong `verified_success`; anything else is a strong
    /// `verified_failure` with [`FailureDomain::Unknown`], because a real failure domain
    /// needs a caller-supplied verdict ([`TaskOutcome::from_trace_record`] refuses to guess
    /// one from an error string).
    #[must_use]
    pub fn from_trace_record(record: &TraceRecord) -> Self {
        if record.success {
            Self {
                label: TaskOutcomeLabel::VerifiedSuccess,
                evidence_strength: EvidenceStrength::Strong,
                failure_domain: None,
            }
        } else {
            Self {
                label: TaskOutcomeLabel::VerifiedFailure,
                evidence_strength: EvidenceStrength::Strong,
                failure_domain: Some(FailureDomain::Unknown),
            }
        }
    }
}

/// The task-level verdict plus the query it belongs to.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct TaskEvidence {
    /// The user query the task answered, when the trace carried one.
    pub query: Option<String>,
    /// The task cluster the run belongs to, when the caller clustered it.
    pub task_cluster_id: Option<String>,
    /// The resolved outcome.
    pub outcome: TaskOutcome,
}

impl TaskEvidence {
    /// Builds task evidence from an ingested trace record.
    ///
    /// The query is only carried when the record's query is non-blank; no cluster id is
    /// invented, and the outcome comes from [`TaskOutcome::from_trace_record`].
    #[must_use]
    pub fn from_trace_record(record: &TraceRecord) -> Self {
        Self {
            query: non_blank(&record.query),
            task_cluster_id: None,
            outcome: TaskOutcome::from_trace_record(record),
        }
    }
}

/// The canonical input accepted by the observation layer.
///
/// Serialise this type and hand it to the observation service; the service qualifies the
/// payload and appends it to the evidence log. See the module docs for the validation
/// rules and the deliberate differences from Python.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct GraphEvolutionInput {
    /// The contract version; must equal [`GRAPH_EVOLUTION_INPUT_SCHEMA`].
    #[serde(default = "default_schema_version")]
    pub schema_version: String,
    /// The caller-assigned evidence id, unique per graph scope.
    pub evidence_id: String,
    /// The graph scope this evidence belongs to.
    #[serde(default = "default_graph_scope_id")]
    pub graph_scope_id: String,
    /// When the task was observed, as RFC 3339 text with a timezone.
    pub observed_at: String,
    /// The revisions the task was observed against.
    pub graph_snapshot: GraphSnapshotRef,
    /// The trace identity and quality metadata.
    pub trace: TraceEvidence,
    /// The task-level verdict.
    pub task: TaskEvidence,
    /// The plan the task executed, when the caller captured one.
    pub planned_graph: Option<EvolutionGraph>,
    /// The transitions the task actually executed.
    pub execution_graph: EvolutionGraph,
}

impl Default for GraphEvolutionInput {
    fn default() -> Self {
        Self {
            schema_version: GRAPH_EVOLUTION_INPUT_SCHEMA.to_string(),
            evidence_id: String::new(),
            graph_scope_id: default_graph_scope_id(),
            observed_at: String::new(),
            graph_snapshot: GraphSnapshotRef::default(),
            trace: TraceEvidence::default(),
            task: TaskEvidence::default(),
            planned_graph: None,
            execution_graph: EvolutionGraph::default(),
        }
    }
}

impl GraphEvolutionInput {
    /// Validates the payload against the contract.
    ///
    /// Every rule from Python's field and model validators is applied, in document order,
    /// so the first violation wins.
    ///
    /// # Errors
    ///
    /// Returns an `Error::Validation` naming the offending
    /// field: `schema_version`, `evidence_id`, `graph_scope_id`, `observed_at`,
    /// `graph_snapshot.*`, `planned_graph.type`, `execution_graph.type`,
    /// `execution_graph.nodes`, `execution_graph.nodes.{id}.metadata`,
    /// `execution_graph.edges`, `task.outcome.failure_domain`, `trace.*`.
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != GRAPH_EVOLUTION_INPUT_SCHEMA {
            return Err(validation(
                "schema_version",
                format!("must be `{GRAPH_EVOLUTION_INPUT_SCHEMA}`"),
            ));
        }
        require_contract_text("evidence_id", &self.evidence_id)?;
        require_contract_text("graph_scope_id", &self.graph_scope_id)?;
        timestamp::normalize_observed_at(&self.observed_at)?;
        require_contract_text(
            "graph_snapshot.static_revision",
            &self.graph_snapshot.static_revision,
        )?;
        require_contract_text(
            "graph_snapshot.observation_revision",
            &self.graph_snapshot.observation_revision,
        )?;
        if let Some(merged_revision) = &self.graph_snapshot.merged_revision {
            require_contract_text("graph_snapshot.merged_revision", merged_revision)?;
        }
        if let Some(planned_graph) = &self.planned_graph {
            require_graph_type(planned_graph, PLANNED_GRAPH_TYPE, "planned_graph.type")?;
        }
        require_graph_type(
            &self.execution_graph,
            EXECUTION_GRAPH_TYPE,
            "execution_graph.type",
        )?;
        for (node_id, node) in &self.execution_graph.nodes {
            require_contract_text("execution_graph.nodes", node_id)?;
            if node.label != SKILL_LABEL {
                return Err(validation(
                    "execution_graph.nodes",
                    format!("node `{node_id}` must have label `{SKILL_LABEL}`"),
                ));
            }
            validate_node_metadata(node_id, node)?;
        }
        for edge in &self.execution_graph.edges {
            require_contract_text("execution_graph.edges", &edge.relation_type)?;
            for endpoint in [&edge.source_id, &edge.target_id] {
                if !self.execution_graph.nodes.contains_key(endpoint) {
                    return Err(validation(
                        "execution_graph.edges",
                        format!("references missing node `{endpoint}`"),
                    ));
                }
            }
        }
        if self.task.outcome.label == TaskOutcomeLabel::VerifiedFailure
            && self.task.outcome.failure_domain.is_none()
        {
            return Err(validation(
                "task.outcome.failure_domain",
                "verified_failure requires a failure domain",
            ));
        }
        require_contract_text("trace.trajectory_id", &self.trace.trajectory_id)?;
        require_contract_text("trace.session_id", &self.trace.session_id)?;
        require_contract_text("trace.capture_mode", &self.trace.capture_mode)?;
        for (field, values) in [
            ("trace.quality_flags", &self.trace.quality_flags),
            ("trace.span_refs", &self.trace.span_refs),
            (
                "trace.member_trajectory_refs",
                &self.trace.member_trajectory_refs,
            ),
        ] {
            for value in values {
                require_contract_text(field, value)?;
            }
        }
        Ok(())
    }

    /// Validates the payload and rewrites its presentation-only fields into canonical form.
    ///
    /// Two rewrites happen, mirroring Python's validators: `observed_at` becomes the
    /// canonical UTC `...Z` text, and every execution node port becomes an object with
    /// exactly the four keys `{name, type, required, description}`, defaulting `required`
    /// to `false` and `description` to `""`. Node metadata keys are otherwise untouched.
    ///
    /// # Errors
    ///
    /// Returns whatever [`GraphEvolutionInput::validate`] returns; normalization itself
    /// cannot fail once validation succeeded.
    pub fn normalized(mut self) -> Result<Self> {
        self.validate()?;
        self.observed_at = timestamp::normalize_observed_at(&self.observed_at)?;
        for node in self.execution_graph.nodes.values_mut() {
            for field in ["inputs", "outputs"] {
                let canonical = match node.metadata.get(field) {
                    Some(Value::Array(ports)) => {
                        Some(Value::Array(ports.iter().map(canonical_port).collect()))
                    }
                    _ => None,
                };
                if let Some(value) = canonical {
                    node.metadata.insert(field.to_string(), value);
                }
            }
        }
        Ok(self)
    }
}

/// The receipt appended for one evidence row, before aggregation completes.
///
/// `sequence` is assigned by the evidence log and is `>= 1`; the log (not this type) owns
/// that invariant, so a deserialised receipt with `sequence == 0` is simply not something
/// the service ever produces.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct ObservationReceipt {
    /// The caller-assigned evidence id the row was recorded under.
    pub evidence_id: String,
    /// The graph scope the row belongs to.
    pub graph_scope_id: String,
    /// The append sequence, `>= 1`.
    pub sequence: u64,
    /// Whether the row will be aggregated, is audit-only, or was a duplicate.
    pub status: ObservationStatus,
    /// The refusal reason, empty for an accepted row.
    pub reason: String,
}

/// One immutable static-plus-observation snapshot as a planner reads it.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct GraphSnapshot {
    /// The graph scope this snapshot belongs to.
    pub graph_scope_id: String,
    /// The static `can_feed` graph revision the overlay was built on.
    pub static_revision: String,
    /// The observation revision the overlay was derived from.
    pub observation_revision: String,
    /// The merged revision id (a content hash of the scope and both inputs).
    pub merged_revision: String,
    /// The highest evidence sequence the snapshot has consumed, `>= 0`.
    pub high_water_sequence: u64,
    /// The per-edge overlay (`runtime_weight`, `binding_status`, counters).
    pub overlay: BTreeMap<String, Value>,
}

/// The serde default for [`EvolutionGraphNode::label`].
fn default_skill_label() -> String {
    SKILL_LABEL.to_string()
}

/// The serde default for [`EvolutionGraphEdge::relation_type`].
fn default_relation_type() -> String {
    RELATION_CAN_FEED.to_string()
}

/// The serde default for [`EvolutionGraph::directed`].
const fn default_directed() -> bool {
    true
}

/// The serde default for [`GraphEvolutionInput::schema_version`].
fn default_schema_version() -> String {
    GRAPH_EVOLUTION_INPUT_SCHEMA.to_string()
}

/// The serde default for [`GraphEvolutionInput::graph_scope_id`].
fn default_graph_scope_id() -> String {
    "default".to_string()
}

/// Returns `Some(text)` when `text` is non-blank.
fn non_blank(text: &str) -> Option<String> {
    (!text.trim().is_empty()).then_some(text.to_string())
}

/// Rejects blank, untrimmed, or control/format-character-bearing contract text.
fn require_contract_text(field: &str, value: &str) -> Result<()> {
    if valid_contract_text(value) {
        Ok(())
    } else {
        Err(validation(field, "must be non-blank contract text"))
    }
}

/// Requires `graph` to be typed `expected`, reporting `field` otherwise.
fn require_graph_type(graph: &EvolutionGraph, expected: &str, field: &str) -> Result<()> {
    if graph.graph_type == expected {
        Ok(())
    } else {
        Err(validation(field, format!("must be `{expected}`")))
    }
}

/// Validates one execution node's allow-listed capability snapshot.
fn validate_node_metadata(node_id: &str, node: &EvolutionGraphNode) -> Result<()> {
    let path = format!("execution_graph.nodes.{node_id}.metadata");
    if let Some(key) = node
        .metadata
        .keys()
        .find(|key| !EXECUTION_NODE_METADATA_FIELDS.contains(&key.as_str()))
    {
        return Err(validation(
            &path,
            format!("unsupported metadata field `{key}`"),
        ));
    }
    match node.metadata.get("capability_type").and_then(Value::as_str) {
        Some(capability_type) if capability_type == node.label => {}
        _ => {
            return Err(validation(
                &path,
                "`capability_type` must be a string equal to the node label",
            ));
        }
    }
    match node.metadata.get("version").and_then(Value::as_str) {
        Some(version) if valid_contract_text(version) => {}
        _ => {
            return Err(validation(
                &path,
                "`version` must be non-blank contract text",
            ));
        }
    }
    if let Some(description) = node.metadata.get("description") {
        require_optional_description(&format!("{path}.description"), description)?;
    }
    for field in ["inputs", "outputs"] {
        if let Some(ports) = node.metadata.get(field) {
            validate_ports(&format!("{path}.{field}"), ports)?;
        }
    }
    Ok(())
}

/// Validates an optional description: absent, `""`, or contract text.
fn require_optional_description(path: &str, value: &Value) -> Result<()> {
    match value.as_str() {
        Some(text) if text.is_empty() || valid_contract_text(text) => Ok(()),
        _ => Err(validation(path, "must be `\"\"` or contract text")),
    }
}

/// Validates an optional capability-port array.
fn validate_ports(path: &str, value: &Value) -> Result<()> {
    let Value::Array(ports) = value else {
        return Err(validation(path, "must be an array of capability ports"));
    };
    for port in ports {
        validate_port(path, port)?;
    }
    Ok(())
}

/// Validates one capability port object.
fn validate_port(path: &str, value: &Value) -> Result<()> {
    let Value::Object(port) = value else {
        return Err(validation(path, "capability port must be an object"));
    };
    if let Some(key) = port
        .keys()
        .find(|key| !CAPABILITY_PORT_FIELDS.contains(&key.as_str()))
    {
        return Err(validation(
            path,
            format!("unsupported capability port field `{key}`"),
        ));
    }
    for field in ["name", "type"] {
        match port.get(field).and_then(Value::as_str) {
            Some(text) if valid_contract_text(text) => {}
            _ => {
                return Err(validation(
                    path,
                    format!("capability port `{field}` must be non-blank contract text"),
                ));
            }
        }
    }
    if let Some(required) = port.get("required") {
        if !required.is_boolean() {
            return Err(validation(
                path,
                "capability port `required` must be a boolean",
            ));
        }
    }
    if let Some(description) = port.get("description") {
        require_optional_description(path, description)?;
    }
    Ok(())
}

/// Rewrites one validated capability port into the canonical four-key object.
///
/// [`GraphEvolutionInput::validate`] has already established that the port is an object
/// with non-blank `name` and `type`, so the fallbacks below are unreachable; they keep the
/// function total instead of hiding a validation gap.
fn canonical_port(port: &Value) -> Value {
    let Some(object) = port.as_object() else {
        return port.clone();
    };
    let name = object
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let port_type = object
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let required = object
        .get("required")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let description = object
        .get("description")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let mut canonical = serde_json::Map::new();
    canonical.insert("name".to_string(), Value::String(name.to_string()));
    canonical.insert("type".to_string(), Value::String(port_type.to_string()));
    canonical.insert("required".to_string(), Value::Bool(required));
    canonical.insert(
        "description".to_string(),
        Value::String(description.to_string()),
    );
    Value::Object(canonical)
}

/// Reports whether `value` is *contract text*: non-empty, already trimmed, and free of
/// control (`Cc`) and format (`Cf`) characters.
///
/// This mirrors Python's `_valid_contract_text`. Python's extra UTF-8 encodability check is
/// vacuous here, because a Rust `str` is always valid UTF-8.
fn valid_contract_text(value: &str) -> bool {
    !value.is_empty()
        && value == value.trim()
        && !value
            .chars()
            .any(|character| character.is_control() || is_format_char(character))
}

/// Reports whether `character` is in Unicode general category `Cf` (a format character).
///
/// Rust's standard library has no Unicode general-category table, so this hand-rolls the
/// main `Cf` ranges; it *approximates* Python's `unicodedata.category(character) == "Cf"`
/// for the characters a caller might realistically embed in an id (soft hyphen, Arabic
/// number signs, zero-width and bidi controls, the word joiner ranges, the interlinear
/// annotation controls, and the tag characters).
fn is_format_char(character: char) -> bool {
    matches!(
        character,
        '\u{00AD}'
            | '\u{0600}'..='\u{0605}'
            | '\u{061C}'
            | '\u{06DD}'
            | '\u{070F}'
            | '\u{180E}'
            | '\u{200B}'..='\u{200F}'
            | '\u{202A}'..='\u{202E}'
            | '\u{2060}'..='\u{2064}'
            | '\u{2066}'..='\u{206F}'
            | '\u{FEFF}'
            | '\u{FFF9}'..='\u{FFFB}'
            | '\u{110BD}'
            | '\u{1D173}'..='\u{1D17A}'
            | '\u{E0001}'
            | '\u{E0020}'..='\u{E007F}'
    )
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]
mod tests {
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
        let edge: EvolutionGraphEdge =
            serde_json::from_str(r#"{"source":"a","target":"b"}"#).unwrap();
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
}
