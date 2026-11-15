//! The response half of the matcher protocol: expanding and validating a reply.
//!
//! Ports Python `openjiuwen.symphony.orchestration.graph.matcher.protocol`'s
//! `parse_match_response`. The model answers one compact document and everything after it
//! is decided here — the model never chooses a relation, an endpoint or a port:
//!
//! ```json
//! {"matches":[{"id":"c3","direction":"forward","confidence":0.9,"reason":"…"}]}
//! ```
//!
//! The pipeline runs in Python's order:
//!
//! 1. **explicit rejections** — an item whose `accepted` flag is falsy (`false`, `0`,
//!    `"0"`, `"false"`, `"no"`, `""`) has its confidence forced to `0.0` before anything
//!    else. Only a real `true`, `"1"`, `"true"` or `"yes"` counts as accepted, so a
//!    string flag never becomes truthy by coercion.
//! 2. **expansion** — a compact item becomes a full match: `candidate_id` from the
//!    candidate key, `source_id`/`target_id` swapped for `direction: "reverse"`,
//!    `relation_type` always [`CAN_FEED`],
//!    `method` always `llm_ontology_match`, and `supporting_fields` derived from that
//!    direction's evidence. A non-object item, an unknown id, a repeated id, a direction
//!    other than `forward`/`reverse`, or a direction without evidence is dropped with a
//!    diagnostic. An item that already carries `candidate_id`/`source_id`/`target_id` is
//!    passed through to validation instead of being dropped (a Rust-only leniency, so a
//!    caller may hand in an already-normalized item).
//! 3. **validation** — endpoints must belong to a candidate, the relation is always
//!    [`CAN_FEED`], the confidence must parse as a number
//!    (unparseable → `0` plus an error) and is clamped into `[0, 1]` (out of range → error,
//!    clamped value kept), the item is matched to its candidate by `candidate_id` and then
//!    by the `(source, target)` pair in either orientation, and `can_feed` supporting
//!    fields must be consistent with the candidate evidence: requested `port_mappings`
//!    must be a subset of the evidence pairs, otherwise the named outputs/inputs must
//!    intersect the evidence, otherwise the evidence must hold some output/input pair.
//! 4. **thresholding** — `accepted` is "no errors *and* confidence at or above the
//!    relation's threshold" ([`DEFAULT_CAN_FEED_THRESHOLD`]).
//!
//! Errors are collected, never early-returned, and every rejected match still appears in
//! [`MatchOutcome::matches`] with its [`ValidatedMatch::errors`] so a caller can report it.
//! Diagnostics come in a fixed order: protocol diagnostics (expansion) first, then
//! validation diagnostics, each in payload order.
//!
//! # Evidence for validation
//!
//! [`parse_match_response`] validates against the candidate's own matched output/input
//! pair, which certifies the forward direction only. Pass
//! [`MatchRequest::evidence`](crate::orchestration::matcher::MatchRequest::evidence) to
//! [`parse_match_response_with_evidence`] to validate against every pair the model was
//! actually shown — which is what makes a reverse-direction verdict judgeable.

use std::collections::{BTreeMap, BTreeSet};

use schemars::JsonSchema;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{json, Map, Value};

use crate::graph::CAN_FEED;
use crate::llm::ontology::{OntologyMatch, METHOD_LLM};
use crate::orchestration::candidates::RelationCandidate;
use crate::orchestration::matcher::protocol::{
    candidate_key, CandidateEvidence, DirectionEvidence, DIRECTION_FORWARD, DIRECTION_REVERSE,
    MAX_REASON_LENGTH,
};

/// The default inclusive confidence a `can_feed` verdict must reach to be accepted.
pub const DEFAULT_CAN_FEED_THRESHOLD: f64 = 0.7;
/// The threshold applied to a relation without a configured entry.
const FALLBACK_THRESHOLD: f64 = 1.0;
/// A compact match item that is not an object.
pub const DIAGNOSTIC_INVALID_MATCH_ITEM: &str = "invalid_match_item";
/// A compact match naming a candidate id that the request never assigned.
pub const DIAGNOSTIC_UNKNOWN_CANDIDATE_ID: &str = "unknown_candidate_id";
/// A compact match repeating a candidate id; the first answer wins.
pub const DIAGNOSTIC_DUPLICATE_CANDIDATE_ID: &str = "duplicate_candidate_id";
/// A compact match asking for a direction other than `forward` or `reverse`.
pub const DIAGNOSTIC_INVALID_CANDIDATE_DIRECTION: &str = "invalid_candidate_direction";
/// A compact match asking for a direction the candidate has no evidence for.
pub const DIAGNOSTIC_UNSUPPORTED_CANDIDATE_DIRECTION: &str = "unsupported_candidate_direction";
/// A payload whose `matches` field is not a list.
pub const DIAGNOSTIC_INVALID_MATCHES_PAYLOAD: &str = "invalid_matches_payload";
/// A match that failed validation; the match is kept with its errors.
pub const DIAGNOSTIC_REJECTED_LLM_MATCH: &str = "rejected_llm_match";
/// A match that validated but stayed below the relation threshold.
pub const DIAGNOSTIC_LOW_CONFIDENCE_LLM_MATCH: &str = "low_confidence_llm_match";

/// One machine-readable record about the protocol pipeline.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct GraphDiagnostic {
    /// The stable diagnostic code ([`DIAGNOSTIC_UNKNOWN_CANDIDATE_ID`], …).
    pub code: String,
    /// The human-readable explanation.
    pub message: String,
    /// The machine-readable context; carries `candidate_id` when one is known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<Value>,
}

impl GraphDiagnostic {
    /// Builds a diagnostic without context.
    #[must_use]
    pub fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
            detail: None,
        }
    }

    /// Attaches the machine-readable context.
    #[must_use]
    pub fn with_detail(mut self, detail: Value) -> Self {
        self.detail = Some(detail);
        self
    }

    /// The candidate id this diagnostic is attributable to, when one is known.
    #[must_use]
    pub fn candidate_id(&self) -> Option<&str> {
        self.detail
            .as_ref()
            .and_then(|detail| detail.get("candidate_id"))
            .and_then(Value::as_str)
            .filter(|id| !id.is_empty())
    }
}

/// One requested output→input wiring, as the model reported it.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct PortMappingPair {
    /// The producing output name.
    pub source_output: String,
    /// The consuming input name.
    pub target_input: String,
}

/// The supporting fields a `can_feed` verdict rests on.
///
/// Deserialization is tolerant and mirrors Python's `_validation_field_names`: a field may
/// be a plain string (`"text"`, `"text: string"`, `"text(string)"`) or an object with a
/// `name`. Every value is trimmed, and empty collections are skipped when serializing, so
/// an empty instance renders as `{}`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct SupportingFields {
    /// The requested output→input wiring, deduplicated in report order.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub port_mappings: Vec<PortMappingPair>,
    /// The producing output names the verdict relied on.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub source_outputs: Vec<String>,
    /// The consuming input names the verdict relied on.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub target_inputs: Vec<String>,
}

impl SupportingFields {
    /// Reads the fields from a JSON object, normalizing names and port pairs.
    ///
    /// A non-object value yields empty fields.
    #[must_use]
    pub fn from_value(value: &Value) -> Self {
        let Some(object) = value.as_object() else {
            return Self::default();
        };
        Self {
            port_mappings: object
                .get("port_mappings")
                .map(port_pairs)
                .unwrap_or_default(),
            source_outputs: object
                .get("source_outputs")
                .map(field_names)
                .unwrap_or_default(),
            target_inputs: object
                .get("target_inputs")
                .map(field_names)
                .unwrap_or_default(),
        }
    }

    /// Derives the fields from one direction's evidence.
    ///
    /// Every port pair becomes a wiring entry; a direction with no wiring falls back to the
    /// evidence output/input names.
    #[must_use]
    pub fn from_evidence(evidence: &DirectionEvidence) -> Self {
        let mut pairs: Vec<PortMappingPair> = Vec::new();
        for (output, input) in evidence.port_pairs() {
            let pair = PortMappingPair {
                source_output: output.to_string(),
                target_input: input.to_string(),
            };
            if !pairs.contains(&pair) {
                pairs.push(pair);
            }
        }
        if !pairs.is_empty() {
            return Self {
                source_outputs: sorted_names(pairs.iter().map(|pair| pair.source_output.as_str())),
                target_inputs: sorted_names(pairs.iter().map(|pair| pair.target_input.as_str())),
                port_mappings: pairs,
            };
        }
        Self {
            port_mappings: Vec::new(),
            source_outputs: sorted_names(evidence.output_names()),
            target_inputs: sorted_names(evidence.input_names()),
        }
    }

    /// The requested wiring as a pair set.
    #[must_use]
    pub fn pairs(&self) -> BTreeSet<(String, String)> {
        self.port_mappings
            .iter()
            .map(|pair| (pair.source_output.clone(), pair.target_input.clone()))
            .collect()
    }

    /// Reports whether nothing is recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.port_mappings.is_empty()
            && self.source_outputs.is_empty()
            && self.target_inputs.is_empty()
    }

    /// Completes the output/input names from the requested wiring when they are absent.
    #[must_use]
    pub fn completed_from_pairs(mut self, pairs: &BTreeSet<(String, String)>) -> Self {
        if self.source_outputs.is_empty() {
            self.source_outputs = sorted_names(pairs.iter().map(|(output, _)| output.as_str()));
        }
        if self.target_inputs.is_empty() {
            self.target_inputs = sorted_names(pairs.iter().map(|(_, input)| input.as_str()));
        }
        self
    }

    /// Renders the fields as a pruned JSON object.
    #[must_use]
    pub fn to_value(&self) -> Value {
        let mut object = Map::new();
        if !self.port_mappings.is_empty() {
            let pairs: Vec<Value> = self
                .port_mappings
                .iter()
                .map(|pair| json!({"source_output": pair.source_output, "target_input": pair.target_input}))
                .collect();
            object.insert("port_mappings".to_string(), Value::Array(pairs));
        }
        if !self.source_outputs.is_empty() {
            object.insert(
                "source_outputs".to_string(),
                Value::from(self.source_outputs.clone()),
            );
        }
        if !self.target_inputs.is_empty() {
            object.insert(
                "target_inputs".to_string(),
                Value::from(self.target_inputs.clone()),
            );
        }
        Value::Object(object)
    }
}

impl<'de> Deserialize<'de> for SupportingFields {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        Ok(Self::from_value(&value))
    }
}

/// One validated relation verdict, richer than [`OntologyMatch`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct ValidatedMatch {
    /// The candidate the verdict answers, when it resolved to one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub candidate_id: Option<String>,
    /// The qualified id of the producing capability.
    pub source_id: String,
    /// The qualified id of the consuming capability.
    pub target_id: String,
    /// The relation kind; always [`CAN_FEED`].
    pub relation: String,
    /// The clamped confidence in `[0.0, 1.0]`.
    pub confidence: f64,
    /// How the verdict was produced (`llm_ontology_match`).
    pub method: String,
    /// The reasons the model gave, trimmed and empty ones dropped.
    pub reasons: Vec<String>,
    /// The supporting fields the verdict rests on.
    pub supporting_fields: SupportingFields,
    /// Whether the verdict passed validation *and* the threshold.
    pub accepted: bool,
    /// Every validation error, in detection order; empty when the verdict validated.
    pub errors: Vec<String>,
}

impl ValidatedMatch {
    /// Projects the verdict onto the crate's slimmer [`OntologyMatch`].
    #[must_use]
    pub fn to_ontology_match(&self) -> OntologyMatch {
        OntologyMatch {
            source_id: self.source_id.clone(),
            target_id: self.target_id.clone(),
            relation: self.relation.clone(),
            confidence: self.confidence,
            method: self.method.clone(),
            reasons: self.reasons.clone(),
            accepted: self.accepted,
        }
    }
}

/// The expansion and validation outcome of one matcher reply.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct MatchOutcome {
    /// Every match that survived expansion, accepted or rejected, in payload order.
    pub matches: Vec<ValidatedMatch>,
    /// The protocol and validation diagnostics, protocol first.
    pub diagnostics: Vec<GraphDiagnostic>,
}

impl MatchOutcome {
    /// The accepted verdicts.
    #[must_use]
    pub fn accepted_matches(&self) -> Vec<&ValidatedMatch> {
        self.matches.iter().filter(|item| item.accepted).collect()
    }

    /// Every verdict projected onto [`OntologyMatch`], accepted or rejected.
    #[must_use]
    pub fn ontology_matches(&self) -> Vec<OntologyMatch> {
        self.matches
            .iter()
            .map(ValidatedMatch::to_ontology_match)
            .collect()
    }
}

/// The inclusive confidence thresholds per relation kind.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct MatchThresholds {
    /// The `can_feed` threshold ([`DEFAULT_CAN_FEED_THRESHOLD`]).
    pub can_feed: f64,
}

impl Default for MatchThresholds {
    fn default() -> Self {
        Self {
            can_feed: DEFAULT_CAN_FEED_THRESHOLD,
        }
    }
}

impl MatchThresholds {
    /// Builds thresholds with the given `can_feed` gate.
    #[must_use]
    pub fn new(can_feed: f64) -> Self {
        Self { can_feed }
    }

    /// The threshold of one relation; an unknown relation needs certainty (`1.0`).
    #[must_use]
    pub fn threshold(&self, relation: &str) -> f64 {
        if relation == CAN_FEED {
            self.can_feed
        } else {
            FALLBACK_THRESHOLD
        }
    }
}

/// Expands and validates one matcher reply with the default thresholds.
#[must_use]
pub fn parse_match_response(payload: &Value, candidates: &[RelationCandidate]) -> MatchOutcome {
    parse_match_response_with_evidence(
        payload,
        candidates,
        &CandidateEvidence::new(),
        &MatchThresholds::default(),
    )
}

/// Expands and validates one matcher reply with explicit thresholds.
#[must_use]
pub fn parse_match_response_with(
    payload: &Value,
    candidates: &[RelationCandidate],
    thresholds: &MatchThresholds,
) -> MatchOutcome {
    parse_match_response_with_evidence(payload, candidates, &CandidateEvidence::new(), thresholds)
}

/// Expands and validates one matcher reply against the evidence it was asked with.
///
/// `evidence` is
/// [`MatchRequest::evidence`](crate::orchestration::matcher::MatchRequest::evidence); a
/// direction it does not cover falls back to the candidate's own matched pair.
#[must_use]
pub fn parse_match_response_with_evidence(
    payload: &Value,
    candidates: &[RelationCandidate],
    evidence: &CandidateEvidence,
    thresholds: &MatchThresholds,
) -> MatchOutcome {
    let mut working = payload.clone();
    apply_explicit_rejections(&mut working);
    let (expanded, mut diagnostics) = expand_compact_response(&working, candidates, evidence);
    let Some(items) = expanded else {
        diagnostics.push(GraphDiagnostic::new(
            DIAGNOSTIC_INVALID_MATCHES_PAYLOAD,
            "LLM payload field 'matches' must be a list.",
        ));
        return MatchOutcome {
            matches: Vec::new(),
            diagnostics,
        };
    };
    let (matches, validation_diagnostics) =
        validate_matches(&items, candidates, evidence, *thresholds);
    diagnostics.extend(validation_diagnostics);
    MatchOutcome {
        matches,
        diagnostics,
    }
}

/// Forces the confidence of an explicitly rejected item to `0.0`.
fn apply_explicit_rejections(payload: &mut Value) {
    let Some(items) = payload.get_mut("matches").and_then(Value::as_array_mut) else {
        return;
    };
    for item in items {
        let Some(object) = item.as_object_mut() else {
            continue;
        };
        let Some(flag) = object.get("accepted") else {
            continue;
        };
        if !explicitly_accepted(flag) {
            object.insert("confidence".to_string(), Value::from(0.0));
        }
    }
}

/// Reports whether an `accepted` flag is a real, explicit acceptance.
///
/// Only `true`, `"1"`, `"true"` and `"yes"` count; every other value (including `0` and
/// `"0"`) is a rejection, so nothing becomes truthy by coercion.
fn explicitly_accepted(flag: &Value) -> bool {
    match flag {
        Value::Bool(accepted) => *accepted,
        Value::String(text) => {
            let text = text.trim().to_lowercase();
            matches!(text.as_str(), "1" | "true" | "yes")
        }
        _ => false,
    }
}

/// Expands compact items into full matches.
///
/// `None` means the payload's `matches` field is not a list; the caller reports
/// [`DIAGNOSTIC_INVALID_MATCHES_PAYLOAD`]. A missing `matches` field expands to no items.
fn expand_compact_response(
    payload: &Value,
    candidates: &[RelationCandidate],
    evidence: &CandidateEvidence,
) -> (Option<Vec<Value>>, Vec<GraphDiagnostic>) {
    let Some(field) = payload.get("matches") else {
        return (Some(Vec::new()), Vec::new());
    };
    let Some(items) = field.as_array() else {
        return (None, Vec::new());
    };
    let candidates_by_id: BTreeMap<String, &RelationCandidate> = candidates
        .iter()
        .enumerate()
        .map(|(index, candidate)| (format!("c{}", index + 1), candidate))
        .collect();
    let mut diagnostics = Vec::new();
    let mut seen: BTreeSet<String> = BTreeSet::new();
    let mut expanded = Vec::with_capacity(items.len());
    for item in items {
        let Some(object) = item.as_object() else {
            diagnostics.push(GraphDiagnostic::new(
                DIAGNOSTIC_INVALID_MATCH_ITEM,
                "Compact LLM match item must be an object.",
            ));
            continue;
        };
        if is_full_match(object) {
            expanded.push(item.clone());
            continue;
        }
        if let Some(value) = expand_item(
            object,
            &candidates_by_id,
            evidence,
            &mut seen,
            &mut diagnostics,
        ) {
            expanded.push(value);
        }
    }
    (Some(expanded), diagnostics)
}

/// Reports whether an item is already a full match rather than a compact answer.
fn is_full_match(object: &Map<String, Value>) -> bool {
    !object.contains_key("id")
        && (object.contains_key("candidate_id")
            || object.contains_key("source_id")
            || object.contains_key("target_id"))
}

/// Expands one compact item, or records why it cannot be expanded.
fn expand_item(
    object: &Map<String, Value>,
    candidates_by_id: &BTreeMap<String, &RelationCandidate>,
    evidence: &CandidateEvidence,
    seen: &mut BTreeSet<String>,
    diagnostics: &mut Vec<GraphDiagnostic>,
) -> Option<Value> {
    let id = scalar_text(object.get("id")).trim().to_string();
    let Some(candidate) = candidates_by_id.get(&id) else {
        diagnostics.push(
            GraphDiagnostic::new(
                DIAGNOSTIC_UNKNOWN_CANDIDATE_ID,
                format!("Compact LLM match returned unknown candidate id: '{id}'."),
            )
            .with_detail(json!({"candidate_id": id})),
        );
        return None;
    };
    if !seen.insert(id.clone()) {
        diagnostics.push(
            GraphDiagnostic::new(
                DIAGNOSTIC_DUPLICATE_CANDIDATE_ID,
                format!("Compact LLM match repeated candidate id: '{id}'."),
            )
            .with_detail(json!({"candidate_id": id})),
        );
        return None;
    }
    let direction = expand_direction(object, &id, diagnostics)?;
    let (source_id, target_id) = directed_ids(candidate, &direction);
    let directional = directional_evidence(evidence, candidate, &direction, &source_id, &target_id);
    if directional.is_empty() {
        diagnostics.push(
            GraphDiagnostic::new(
                DIAGNOSTIC_UNSUPPORTED_CANDIDATE_DIRECTION,
                format!("Candidate '{id}' has no evidence for {direction}."),
            )
            .with_detail(json!({"candidate_id": id, "direction": direction})),
        );
        return None;
    }
    let reason = object
        .get("reason")
        .map_or_else(String::new, |value| scalar_text(Some(value)));
    let reason: String = reason.trim().chars().take(MAX_REASON_LENGTH).collect();
    let mut expanded = Map::new();
    expanded.insert(
        "candidate_id".to_string(),
        Value::String(candidate_key(candidate)),
    );
    expanded.insert("source_id".to_string(), Value::String(source_id));
    expanded.insert("target_id".to_string(), Value::String(target_id));
    expanded.insert(
        "relation_type".to_string(),
        Value::String(CAN_FEED.to_string()),
    );
    expanded.insert(
        "confidence".to_string(),
        object
            .get("confidence")
            .cloned()
            .unwrap_or_else(|| Value::from(0.0)),
    );
    expanded.insert("method".to_string(), Value::String(METHOD_LLM.to_string()));
    expanded.insert(
        "reasons".to_string(),
        Value::from(if reason.is_empty() {
            Vec::new()
        } else {
            vec![Value::String(reason)]
        }),
    );
    expanded.insert(
        "supporting_fields".to_string(),
        SupportingFields::from_evidence(&directional).to_value(),
    );
    Some(Value::Object(expanded))
}

/// Reads the direction of one compact item, or records why it is unusable.
fn expand_direction(
    object: &Map<String, Value>,
    id: &str,
    diagnostics: &mut Vec<GraphDiagnostic>,
) -> Option<String> {
    let direction = object
        .get("direction")
        .map_or_else(String::new, |value| scalar_text(Some(value)));
    let direction = direction.trim().to_lowercase();
    let direction = if direction.is_empty() {
        DIRECTION_FORWARD.to_string()
    } else {
        direction
    };
    if direction == DIRECTION_FORWARD || direction == DIRECTION_REVERSE {
        return Some(direction);
    }
    diagnostics.push(
        GraphDiagnostic::new(
            DIAGNOSTIC_INVALID_CANDIDATE_DIRECTION,
            format!("Compact LLM match returned invalid direction: '{direction}'."),
        )
        .with_detail(json!({"candidate_id": id, "direction": direction})),
    );
    None
}

/// The match orientation of one direction.
fn directed_ids(candidate: &RelationCandidate, direction: &str) -> (String, String) {
    if direction == DIRECTION_REVERSE {
        (candidate.target.clone(), candidate.source.clone())
    } else {
        (candidate.source.clone(), candidate.target.clone())
    }
}

/// The evidence of one direction: the recorded table first, then the matched pair.
fn directional_evidence(
    evidence: &CandidateEvidence,
    candidate: &RelationCandidate,
    direction: &str,
    source_id: &str,
    target_id: &str,
) -> DirectionEvidence {
    if let Some(found) = evidence.direction(&candidate_key(candidate), direction) {
        return found.clone();
    }
    if source_id == candidate.source && target_id == candidate.target {
        return DirectionEvidence::from_matched(
            candidate.matched_output.as_deref(),
            candidate.matched_input.as_deref(),
        );
    }
    DirectionEvidence::default()
}

/// Validates every expanded (or already full) item.
fn validate_matches(
    items: &[Value],
    candidates: &[RelationCandidate],
    evidence: &CandidateEvidence,
    thresholds: MatchThresholds,
) -> (Vec<ValidatedMatch>, Vec<GraphDiagnostic>) {
    let endpoints: BTreeSet<&str> = candidates
        .iter()
        .flat_map(|candidate| [candidate.source.as_str(), candidate.target.as_str()])
        .collect();
    let mut by_key: BTreeMap<String, &RelationCandidate> = BTreeMap::new();
    let mut by_pair: BTreeMap<(&str, &str), &RelationCandidate> = BTreeMap::new();
    for candidate in candidates {
        by_key.insert(candidate_key(candidate), candidate);
        by_pair.insert(
            (candidate.source.as_str(), candidate.target.as_str()),
            candidate,
        );
        by_pair.insert(
            (candidate.target.as_str(), candidate.source.as_str()),
            candidate,
        );
    }
    let mut matches = Vec::with_capacity(items.len());
    let mut diagnostics = Vec::new();
    for (index, raw) in items.iter().enumerate() {
        let Some(object) = raw.as_object() else {
            diagnostics.push(
                GraphDiagnostic::new(
                    DIAGNOSTIC_INVALID_MATCH_ITEM,
                    "LLM match item is not an object.",
                )
                .with_detail(json!({"index": index, "item": raw})),
            );
            continue;
        };
        let (validated, item_diagnostics) =
            normalize_match(object, &endpoints, &by_key, &by_pair, evidence, thresholds);
        diagnostics.extend(item_diagnostics);
        matches.push(validated);
    }
    (matches, diagnostics)
}

/// One validated item plus the diagnostics it produced.
fn normalize_match(
    object: &Map<String, Value>,
    endpoints: &BTreeSet<&str>,
    by_key: &BTreeMap<String, &RelationCandidate>,
    by_pair: &BTreeMap<(&str, &str), &RelationCandidate>,
    evidence: &CandidateEvidence,
    thresholds: MatchThresholds,
) -> (ValidatedMatch, Vec<GraphDiagnostic>) {
    let source_id = scalar_text(object.get("source_id"));
    let target_id = scalar_text(object.get("target_id"));
    let reported_relation = scalar_text(object.get("relation_type"));
    let reported_candidate_id = object
        .get("candidate_id")
        .map(|value| scalar_text(Some(value)))
        .filter(|id| !id.is_empty());
    let mut errors: Vec<String> = Vec::new();
    if !endpoints.contains(source_id.as_str()) {
        errors.push("source_id does not exist".to_string());
    }
    if !endpoints.contains(target_id.as_str()) {
        errors.push("target_id does not exist".to_string());
    }
    if !reported_relation.is_empty() && reported_relation != CAN_FEED {
        errors.push("relation_type is not allowed".to_string());
    }
    let candidate = reported_candidate_id
        .as_deref()
        .and_then(|id| by_key.get(id))
        .copied()
        .or_else(|| {
            by_pair
                .get(&(source_id.as_str(), target_id.as_str()))
                .copied()
        });
    if candidate.is_none() {
        errors.push("match does not correspond to an input candidate".to_string());
    }
    let confidence = if let Some(confidence) = confidence_of(object.get("confidence")) {
        confidence
    } else {
        errors.push("confidence is not numeric".to_string());
        0.0
    };
    let confidence = if (0.0..=1.0).contains(&confidence) {
        confidence
    } else {
        errors.push("confidence must be between 0 and 1".to_string());
        confidence.clamp(0.0, 1.0)
    };
    let reasons = reasons_of(object.get("reasons"));
    let mut supporting_fields =
        SupportingFields::from_value(object.get("supporting_fields").unwrap_or(&Value::Null));
    if let Some(candidate) = candidate {
        supporting_fields = check_supporting_fields(
            supporting_fields,
            candidate,
            &source_id,
            &target_id,
            evidence,
            &mut errors,
        );
    }
    let relation = CAN_FEED.to_string();
    let resolved_candidate_id = candidate.map(candidate_key).or(reported_candidate_id);
    let method = {
        let reported = object
            .get("method")
            .map_or_else(String::new, |value| scalar_text(Some(value)));
        let reported = reported.trim();
        if reported.is_empty() {
            METHOD_LLM.to_string()
        } else {
            reported.to_string()
        }
    };
    let accepted = errors.is_empty() && confidence >= thresholds.threshold(&relation);
    let diagnostics = diagnostics_for(
        object,
        resolved_candidate_id.as_deref(),
        &source_id,
        &errors,
        accepted,
        confidence,
        thresholds.threshold(&relation),
    );
    let validated = ValidatedMatch {
        candidate_id: resolved_candidate_id,
        source_id,
        target_id,
        relation,
        confidence,
        method,
        reasons,
        supporting_fields,
        accepted,
        errors,
    };
    (validated, diagnostics)
}

/// Checks `can_feed` supporting fields against the candidate evidence, completing names.
fn check_supporting_fields(
    supporting_fields: SupportingFields,
    candidate: &RelationCandidate,
    source_id: &str,
    target_id: &str,
    evidence: &CandidateEvidence,
    errors: &mut Vec<String>,
) -> SupportingFields {
    let direction = if source_id == candidate.source && target_id == candidate.target {
        DIRECTION_FORWARD
    } else {
        DIRECTION_REVERSE
    };
    let directional = directional_evidence(evidence, candidate, direction, source_id, target_id);
    let requested_pairs = supporting_fields.pairs();
    let evidence_pairs: BTreeSet<(String, String)> = directional
        .port_pairs()
        .into_iter()
        .map(|(output, input)| (output.to_string(), input.to_string()))
        .collect();
    let requested_outputs: BTreeSet<&str> = supporting_fields
        .source_outputs
        .iter()
        .map(String::as_str)
        .collect();
    let requested_inputs: BTreeSet<&str> = supporting_fields
        .target_inputs
        .iter()
        .map(String::as_str)
        .collect();
    if !requested_pairs.is_empty() {
        if !requested_pairs.is_subset(&evidence_pairs) {
            errors.push("port_mappings do not match candidate evidence".to_string());
        }
        return supporting_fields.completed_from_pairs(&requested_pairs);
    }
    let evidence_outputs: BTreeSet<&str> = directional.output_names().into_iter().collect();
    let evidence_inputs: BTreeSet<&str> = directional.input_names().into_iter().collect();
    if !requested_outputs.is_empty() && !requested_inputs.is_empty() {
        if requested_outputs.is_disjoint(&evidence_outputs)
            || requested_inputs.is_disjoint(&evidence_inputs)
        {
            errors.push("supporting_fields do not match candidate evidence".to_string());
        }
    } else if evidence_outputs.is_empty() || evidence_inputs.is_empty() {
        errors.push("can_feed has no supported output/input pair".to_string());
    }
    supporting_fields
}

/// The diagnostics of one validated match.
fn diagnostics_for(
    object: &Map<String, Value>,
    candidate_id: Option<&str>,
    source_id: &str,
    errors: &[String],
    accepted: bool,
    confidence: f64,
    threshold: f64,
) -> Vec<GraphDiagnostic> {
    if !errors.is_empty() {
        return vec![GraphDiagnostic::new(
            DIAGNOSTIC_REJECTED_LLM_MATCH,
            "LLM match failed validation.",
        )
        .with_detail(json!({
            "candidate_id": candidate_id,
            "source_id": source_id,
            "errors": errors,
            "match": Value::Object(object.clone()),
        }))];
    }
    if accepted {
        return Vec::new();
    }
    vec![GraphDiagnostic::new(
        DIAGNOSTIC_LOW_CONFIDENCE_LLM_MATCH,
        "LLM match is below the relation threshold.",
    )
    .with_detail(json!({
        "candidate_id": candidate_id,
        "threshold": threshold,
        "confidence": confidence,
    }))]
}

/// Reads a confidence: a number, a numeric string, or a boolean (never a null or object).
fn confidence_of(value: Option<&Value>) -> Option<f64> {
    let confidence = match value? {
        Value::Number(number) => number.as_f64()?,
        Value::Bool(flag) => {
            if *flag {
                1.0
            } else {
                0.0
            }
        }
        Value::String(text) => text.trim().parse::<f64>().ok()?,
        _ => return None,
    };
    if confidence.is_finite() {
        Some(confidence)
    } else {
        None
    }
}

/// Reads the reported reasons, dropping blank ones.
fn reasons_of(value: Option<&Value>) -> Vec<String> {
    let Some(items) = value.and_then(Value::as_array) else {
        return Vec::new();
    };
    items
        .iter()
        .filter_map(|item| {
            let reason = scalar_text(Some(item));
            (!reason.trim().is_empty()).then_some(reason)
        })
        .collect()
}

/// Renders a scalar JSON value as text; containers and null render as empty.
fn scalar_text(value: Option<&Value>) -> String {
    match value {
        Some(Value::String(text)) => text.clone(),
        Some(Value::Number(number)) => number.to_string(),
        Some(Value::Bool(flag)) => flag.to_string(),
        _ => String::new(),
    }
}

/// Reads the output→input wiring of a supporting-fields object.
fn port_pairs(value: &Value) -> Vec<PortMappingPair> {
    let Some(items) = value.as_array() else {
        return Vec::new();
    };
    let mut pairs: Vec<PortMappingPair> = Vec::new();
    for item in items {
        let Some(object) = item.as_object() else {
            continue;
        };
        let source_output = scalar_text(object.get("source_output")).trim().to_string();
        let target_input = scalar_text(object.get("target_input")).trim().to_string();
        if source_output.is_empty() || target_input.is_empty() {
            continue;
        }
        let pair = PortMappingPair {
            source_output,
            target_input,
        };
        if !pairs.contains(&pair) {
            pairs.push(pair);
        }
    }
    pairs
}

/// Reads IO names: plain strings (`"text"`, `"text: string"`) or objects with a `name`.
fn field_names(value: &Value) -> Vec<String> {
    let Some(items) = value.as_array() else {
        return Vec::new();
    };
    let mut names: Vec<String> = Vec::new();
    for item in items {
        let name = match item {
            Value::String(text) => field_name_from_string(text),
            Value::Object(object) => scalar_text(object.get("name")).trim().to_string(),
            _ => String::new(),
        };
        if !name.is_empty() && !names.contains(&name) {
            names.push(name);
        }
    }
    names
}

/// Reduces `"text: string"` or `"text(string)"` to `"text"`.
fn field_name_from_string(value: &str) -> String {
    let text = value.trim();
    if text.is_empty() {
        return String::new();
    }
    let head = text.split(':').next().unwrap_or(text).trim();
    head.split('(').next().unwrap_or(head).trim().to_string()
}

/// Collects a name stream into a sorted, deduplicated list.
fn sorted_names<'a>(names: impl IntoIterator<Item = &'a str>) -> Vec<String> {
    let set: BTreeSet<String> = names.into_iter().map(str::to_string).collect();
    set.into_iter().collect()
}

/// Groups matches by candidate id, falling back to `"{source_id}->{target_id}"`.
///
/// A match without a usable `candidate_id` is grouped under its endpoint pair, so every
/// match lands in exactly one bucket and iteration order is deterministic.
#[must_use]
pub fn matches_by_candidate(matches: &[ValidatedMatch]) -> BTreeMap<String, Vec<&ValidatedMatch>> {
    let mut grouped: BTreeMap<String, Vec<&ValidatedMatch>> = BTreeMap::new();
    for item in matches {
        let key = match &item.candidate_id {
            Some(id) if !id.is_empty() => id.clone(),
            _ => format!("{}->{}", item.source_id, item.target_id),
        };
        grouped.entry(key).or_default().push(item);
    }
    grouped
}

/// Groups diagnostics by the candidate id recorded in `detail["candidate_id"]`.
///
/// A diagnostic with no attributable candidate (a malformed item, a non-list `matches`
/// field) is omitted here; the flat [`MatchOutcome::diagnostics`] list keeps it.
#[must_use]
pub fn diagnostics_by_candidate(
    diagnostics: &[GraphDiagnostic],
) -> BTreeMap<String, Vec<&GraphDiagnostic>> {
    let mut grouped: BTreeMap<String, Vec<&GraphDiagnostic>> = BTreeMap::new();
    for diagnostic in diagnostics {
        if let Some(id) = diagnostic.candidate_id() {
            grouped.entry(id.to_string()).or_default().push(diagnostic);
        }
    }
    grouped
}
