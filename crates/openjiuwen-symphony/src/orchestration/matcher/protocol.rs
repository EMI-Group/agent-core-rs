//! The request half of the matcher protocol: what a matcher model is shown.
//!
//! Ports Python `openjiuwen.symphony.orchestration.graph.matcher.protocol`'s
//! `build_match_request` and its evidence compaction. A request is one compact JSON
//! document, one entry per [`RelationCandidate`](crate::orchestration::RelationCandidate):
//!
//! ```json
//! {"candidates":[{"id":"c1",
//!   "source":{"name":"Extract","description":"…"},
//!   "target":{"name":"Summarise"},
//!   "directions":{"forward":{"outputs":[{"name":"text","type":"string"}],
//!                            "inputs":[{"name":"text","type":"string"}],
//!                            "ports":[{"output":"text","output_type":"string",
//!                                      "input":"text","input_type":"string"}]}}}]}
//! ```
//!
//! Positional ids are assigned in input order (`c1`, `c2`, …);
//! [`build_match_request`]'s `reverse_skill_order` reverses that assignment, so the *last*
//! input candidate becomes `c1` (Python's `indexed.reverse()`). Parsing a reversed request
//! needs the candidates in the matching order — pass `candidates.iter().rev()` to
//! [`parse_match_response`](crate::orchestration::matcher::parse_match_response).
//!
//! Descriptions are truncated on **character** boundaries (CJK-safe), never byte
//! boundaries: [`MAX_SKILL_DESCRIPTION_LENGTH`] for a skill and
//! [`MAX_PORT_DESCRIPTION_LENGTH`] for one port. Empty values are pruned from the
//! document (a blank description, an absent `required` flag, an empty direction).
//!
//! # Evidence derivation (a Rust-only substitution)
//!
//! Python's candidate generator attaches an `evidence` dict with the exact
//! `source_outputs`, `target_inputs` and `port_mappings` it matched.
//! [`RelationCandidate`](crate::orchestration::RelationCandidate) carries only the matched
//! IO names plus the strategy, so the directional evidence is derived from the two
//! [`CapabilityDescriptor`]s instead:
//!
//! - the forward direction (`source → target`) uses the candidate's matched output/input
//!   pair when both resolve in the descriptors, and otherwise falls back to the same
//!   scan the reverse direction uses;
//! - the reverse direction (`target → source`) proposes pairs by scanning the target's
//!   outputs against the source's inputs for a
//!   [`DataTypeVocabulary::can_feed`] type relation, preferring an exact
//!   [`normalize_name_key`] name match, then a shared name token;
//! - a scan keeps at most [`MAX_DERIVED_PORTS`] port pairs, sorted by that preference and
//!   then by `(output name, input name)`, so the derivation is deterministic;
//! - a direction with no pair is omitted entirely, which is exactly what the model needs
//!   to know it cannot judge that orientation.
//!
//! A candidate naming a capability that is absent from the catalog is skipped with a
//! `tracing` warning instead of failing the whole request.

use std::collections::BTreeMap;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::model::{normalize_name_key, CapabilityDescriptor, CapabilityIo, DataTypeVocabulary};
use crate::orchestration::candidates::RelationCandidate;
use crate::search::tokenize;

/// The maximum number of characters of a skill description shown to the model.
pub const MAX_SKILL_DESCRIPTION_LENGTH: usize = 240;
/// The maximum number of characters of one port description shown to the model.
pub const MAX_PORT_DESCRIPTION_LENGTH: usize = 160;
/// The maximum number of characters of a model-supplied reason kept on a match.
pub const MAX_REASON_LENGTH: usize = 160;
/// The direction asking whether the candidate's source can feed its target.
pub const DIRECTION_FORWARD: &str = "forward";
/// The direction asking whether the candidate's target can feed its source.
pub const DIRECTION_REVERSE: &str = "reverse";
/// The maximum number of derived port pairs kept per direction.
pub const MAX_DERIVED_PORTS: usize = 8;

/// One capability as shown to the model: its name and a truncated description.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct SkillBrief {
    /// The capability name; omitted from the document when blank.
    #[serde(skip_serializing_if = "String::is_empty")]
    pub name: String,
    /// The capability description, truncated to [`MAX_SKILL_DESCRIPTION_LENGTH`].
    #[serde(skip_serializing_if = "String::is_empty")]
    pub description: String,
}

impl SkillBrief {
    /// Builds a brief, truncating the description to [`MAX_SKILL_DESCRIPTION_LENGTH`].
    #[must_use]
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: truncate_chars(&description.into(), MAX_SKILL_DESCRIPTION_LENGTH),
        }
    }

    /// Builds the brief of one capability card.
    #[must_use]
    pub fn from_descriptor(descriptor: &CapabilityDescriptor) -> Self {
        Self::new(descriptor.name.clone(), descriptor.description.clone())
    }

    /// Renders the brief as a pruned JSON object.
    #[must_use]
    pub fn to_value(&self) -> Value {
        let mut object = Map::new();
        insert_text(&mut object, "name", &self.name);
        insert_text(&mut object, "description", &self.description);
        Value::Object(object)
    }
}

/// One input or output as shown to the model.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct IoBrief {
    /// The IO name; omitted from the document when blank.
    #[serde(skip_serializing_if = "String::is_empty")]
    pub name: String,
    /// The data type; the JSON key is `"type"`, matching the card wire format.
    #[serde(rename = "type", skip_serializing_if = "String::is_empty")]
    pub io_type: String,
    /// Whether a plan must supply the value; `None` is omitted.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required: Option<bool>,
    /// The IO description, truncated to [`MAX_PORT_DESCRIPTION_LENGTH`].
    #[serde(skip_serializing_if = "String::is_empty")]
    pub description: String,
}

impl IoBrief {
    /// Builds a brief carrying only a name.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Self::default()
        }
    }

    /// The data type (the JSON key is `"type"`, so the accessor mirrors `CapabilityIo`).
    #[must_use]
    pub fn io_type(&self) -> &str {
        &self.io_type
    }

    /// Builds the brief of one card input or output.
    #[must_use]
    pub fn from_io(io: &CapabilityIo) -> Self {
        Self {
            name: io.name.clone(),
            io_type: io.io_type().to_string(),
            required: io.required,
            description: truncate_chars(&io.description, MAX_PORT_DESCRIPTION_LENGTH),
        }
    }

    /// Renders the brief as a pruned JSON object.
    #[must_use]
    pub fn to_value(&self) -> Value {
        let mut object = Map::new();
        insert_text(&mut object, "name", &self.name);
        insert_text(&mut object, "type", &self.io_type);
        if let Some(required) = self.required {
            object.insert("required".to_string(), Value::Bool(required));
        }
        insert_text(&mut object, "description", &self.description);
        Value::Object(object)
    }
}

/// One output→input wiring proposal inside a direction.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct PortBrief {
    /// The producing output name.
    #[serde(skip_serializing_if = "String::is_empty")]
    pub output: String,
    /// The producing output type.
    #[serde(skip_serializing_if = "String::is_empty")]
    pub output_type: String,
    /// The consuming input name.
    #[serde(skip_serializing_if = "String::is_empty")]
    pub input: String,
    /// The consuming input type.
    #[serde(skip_serializing_if = "String::is_empty")]
    pub input_type: String,
}

impl PortBrief {
    /// Builds the port proposal of one output/input pair.
    #[must_use]
    pub fn new(output: &CapabilityIo, input: &CapabilityIo) -> Self {
        Self {
            output: output.name.clone(),
            output_type: output.io_type().to_string(),
            input: input.name.clone(),
            input_type: input.io_type().to_string(),
        }
    }

    /// Renders the port proposal as a pruned JSON object.
    #[must_use]
    pub fn to_value(&self) -> Value {
        let mut object = Map::new();
        insert_text(&mut object, "output", &self.output);
        insert_text(&mut object, "output_type", &self.output_type);
        insert_text(&mut object, "input", &self.input);
        insert_text(&mut object, "input_type", &self.input_type);
        Value::Object(object)
    }
}

/// The compact evidence for one direction of one candidate.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct DirectionEvidence {
    /// The producing outputs, sorted by name.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub outputs: Vec<IoBrief>,
    /// The consuming inputs, sorted by name.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub inputs: Vec<IoBrief>,
    /// The proposed output/input wiring, in derivation order.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub ports: Vec<PortBrief>,
}

impl DirectionEvidence {
    /// Builds the evidence of one capability pair's matched output/input.
    ///
    /// Names that are blank are dropped, so a candidate that matched no named IO yields
    /// empty evidence (and therefore no judgeable direction).
    #[must_use]
    pub fn from_matched(output: Option<&str>, input: Option<&str>) -> Self {
        let output = output.map(str::trim).filter(|name| !name.is_empty());
        let input = input.map(str::trim).filter(|name| !name.is_empty());
        match (output, input) {
            (Some(output), Some(input)) => Self {
                outputs: vec![IoBrief::new(output)],
                inputs: vec![IoBrief::new(input)],
                ports: vec![PortBrief {
                    output: output.to_string(),
                    output_type: String::new(),
                    input: input.to_string(),
                    input_type: String::new(),
                }],
            },
            _ => Self::default(),
        }
    }

    /// Reports whether no output, input or port is known.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.outputs.is_empty() && self.inputs.is_empty() && self.ports.is_empty()
    }

    /// The producing output names, in document order.
    #[must_use]
    pub fn output_names(&self) -> Vec<&str> {
        self.outputs.iter().map(|io| io.name.as_str()).collect()
    }

    /// The consuming input names, in document order.
    #[must_use]
    pub fn input_names(&self) -> Vec<&str> {
        self.inputs.iter().map(|io| io.name.as_str()).collect()
    }

    /// The proposed port pairs, in document order.
    #[must_use]
    pub fn port_pairs(&self) -> Vec<(&str, &str)> {
        self.ports
            .iter()
            .map(|port| (port.output.as_str(), port.input.as_str()))
            .collect()
    }

    /// Renders the evidence as a pruned JSON object.
    #[must_use]
    pub fn to_value(&self) -> Value {
        let mut object = Map::new();
        insert_array(&mut object, "outputs", &self.outputs);
        insert_array(&mut object, "inputs", &self.inputs);
        insert_array(&mut object, "ports", &self.ports);
        Value::Object(object)
    }
}

/// One numbered candidate and the directions the model may judge.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct CandidateContext {
    /// The positional id (`c1`, `c2`, …) the model answers with.
    #[serde(skip_serializing_if = "String::is_empty")]
    pub id: String,
    /// The candidate's own `"{source}->{target}"` key, which positional ids do not carry.
    #[serde(skip)]
    #[schemars(skip)]
    pub candidate_key: String,
    /// The producing capability.
    pub source: SkillBrief,
    /// The consuming capability.
    pub target: SkillBrief,
    /// The judgeable directions, keyed by [`DIRECTION_FORWARD`] or [`DIRECTION_REVERSE`];
    /// a direction without evidence is absent.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub directions: BTreeMap<String, DirectionEvidence>,
}

impl CandidateContext {
    /// Renders the candidate as the protocol object the model sees.
    #[must_use]
    pub fn to_value(&self) -> Value {
        let mut object = Map::new();
        insert_text(&mut object, "id", &self.id);
        object.insert("source".to_string(), self.source.to_value());
        object.insert("target".to_string(), self.target.to_value());
        if !self.directions.is_empty() {
            let mut directions = Map::new();
            for (name, evidence) in &self.directions {
                directions.insert(name.clone(), evidence.to_value());
            }
            object.insert("directions".to_string(), Value::Object(directions));
        }
        Value::Object(object)
    }
}

/// The `{"candidates": […]}` document a matcher model is asked to judge.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct MatchRequest {
    /// The numbered candidates, in the order their ids were assigned.
    pub candidates: Vec<CandidateContext>,
}

impl MatchRequest {
    /// Wraps already-built candidate contexts.
    #[must_use]
    pub fn new(candidates: Vec<CandidateContext>) -> Self {
        Self { candidates }
    }

    /// The candidate contexts, in the order their ids were assigned.
    #[must_use]
    pub fn candidates(&self) -> &[CandidateContext] {
        &self.candidates
    }

    /// Renders the request as the exact `{"candidates": […]}` JSON document.
    ///
    /// The rendering is total (no fallible serialization step), so callers can log or
    /// send it without handling an error.
    #[must_use]
    pub fn to_value(&self) -> Value {
        let candidates: Vec<Value> = self
            .candidates
            .iter()
            .map(CandidateContext::to_value)
            .collect();
        let mut object = Map::new();
        object.insert("candidates".to_string(), Value::Array(candidates));
        Value::Object(object)
    }

    /// The directional evidence of every emitted candidate, keyed by candidate key.
    ///
    /// Feed it to
    /// [`parse_match_response_with_evidence`](crate::orchestration::matcher::parse_match_response_with_evidence)
    /// so a reply is validated against exactly the evidence the model was shown.
    #[must_use]
    pub fn evidence(&self) -> CandidateEvidence {
        let mut evidence = CandidateEvidence::new();
        for context in &self.candidates {
            for (direction, directional) in &context.directions {
                evidence.insert(
                    context.candidate_key.clone(),
                    direction.clone(),
                    directional.clone(),
                );
            }
        }
        evidence
    }
}

/// Directional evidence keyed by `(candidate key, direction)`.
///
/// A reply is validated against the evidence the request showed the model; without this
/// table the parser falls back to the candidate's own matched output/input pair, which
/// certifies the forward direction only.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CandidateEvidence {
    entries: BTreeMap<(String, String), DirectionEvidence>,
}

impl CandidateEvidence {
    /// Builds an empty evidence table.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Records the evidence of one direction of one candidate.
    pub fn insert(
        &mut self,
        candidate_key: impl Into<String>,
        direction: impl Into<String>,
        evidence: DirectionEvidence,
    ) {
        self.entries
            .insert((candidate_key.into(), direction.into()), evidence);
    }

    /// The recorded evidence of one direction of one candidate.
    #[must_use]
    pub fn direction(&self, candidate_key: &str, direction: &str) -> Option<&DirectionEvidence> {
        self.entries
            .get(&(candidate_key.to_string(), direction.to_string()))
    }

    /// Reports whether no evidence is recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The number of recorded directions.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }
}

/// The candidate's `"{source}->{target}"` key.
#[must_use]
pub fn candidate_key(candidate: &RelationCandidate) -> String {
    format!("{}->{}", candidate.source, candidate.target)
}

/// Builds the compact match request for `candidates`.
///
/// Ids are assigned `c1..cN` in input order; `reverse_skill_order` reverses that
/// assignment, so the *last* input candidate becomes `c1` (Python's `indexed.reverse()`,
/// read as a reversed position). A candidate naming a capability that is absent from
/// `capabilities` is skipped with a `tracing` warning, and a direction without any derived
/// port pair is omitted.
#[must_use]
pub fn build_match_request(
    capabilities: &[CapabilityDescriptor],
    candidates: &[RelationCandidate],
    reverse_skill_order: bool,
) -> MatchRequest {
    let mut ordered: Vec<&RelationCandidate> = candidates.iter().collect();
    if reverse_skill_order {
        ordered.reverse();
    }
    let catalog: BTreeMap<String, &CapabilityDescriptor> = capabilities
        .iter()
        .map(|capability| (capability.qualified_id(), capability))
        .collect();
    let mut contexts = Vec::with_capacity(ordered.len());
    for (index, candidate) in ordered.into_iter().enumerate() {
        let id = format!("c{}", index + 1);
        let (Some(source), Some(target)) = (
            catalog.get(&candidate.source),
            catalog.get(&candidate.target),
        ) else {
            tracing::warn!(
                source = candidate.source.as_str(),
                target = candidate.target.as_str(),
                "match request: skipping a candidate whose capability is not in the catalog"
            );
            continue;
        };
        contexts.push(candidate_context(id, candidate, source, target));
    }
    MatchRequest::new(contexts)
}

/// Derives the directional evidence of one candidate from its two capability cards.
#[must_use]
pub fn directional_evidence(
    source: &CapabilityDescriptor,
    target: &CapabilityDescriptor,
    candidate: &RelationCandidate,
    direction: &str,
) -> DirectionEvidence {
    if direction == DIRECTION_REVERSE {
        return scan_pairs(&target.outputs, &source.inputs);
    }
    matched_pair(source, target, candidate)
        .unwrap_or_else(|| scan_pairs(&source.outputs, &target.inputs))
}

/// Builds one candidate context, including only the directions that have evidence.
fn candidate_context(
    id: String,
    candidate: &RelationCandidate,
    source: &CapabilityDescriptor,
    target: &CapabilityDescriptor,
) -> CandidateContext {
    let mut directions = BTreeMap::new();
    for direction in [DIRECTION_FORWARD, DIRECTION_REVERSE] {
        let evidence = directional_evidence(source, target, candidate, direction);
        if !evidence.is_empty() {
            directions.insert(direction.to_string(), evidence);
        }
    }
    CandidateContext {
        id,
        candidate_key: candidate_key(candidate),
        source: SkillBrief::from_descriptor(source),
        target: SkillBrief::from_descriptor(target),
        directions,
    }
}

/// The evidence of the candidate's own matched output/input pair, when both resolve.
fn matched_pair(
    source: &CapabilityDescriptor,
    target: &CapabilityDescriptor,
    candidate: &RelationCandidate,
) -> Option<DirectionEvidence> {
    let output_name = candidate.matched_output.as_deref()?;
    let input_name = candidate.matched_input.as_deref()?;
    let output = source.outputs.iter().find(|io| io.name == output_name)?;
    let input = target.inputs.iter().find(|io| io.name == input_name)?;
    Some(DirectionEvidence {
        outputs: vec![IoBrief::from_io(output)],
        inputs: vec![IoBrief::from_io(input)],
        ports: vec![PortBrief::new(output, input)],
    })
}

/// Proposes the most plausible output/input pairs between two IO lists.
///
/// Only pairs the data-type vocabulary accepts are proposed. They rank exact
/// [`normalize_name_key`] name matches first, then a shared name token, then every other
/// accepted pair; [`MAX_DERIVED_PORTS`] pairs are kept and the output/input lists are
/// sorted by name.
fn scan_pairs(outputs: &[CapabilityIo], inputs: &[CapabilityIo]) -> DirectionEvidence {
    let vocabulary = DataTypeVocabulary::standard();
    let mut ranked: Vec<(u8, String, String, usize, usize)> = Vec::new();
    for (output_index, output) in outputs.iter().enumerate() {
        for (input_index, input) in inputs.iter().enumerate() {
            if !vocabulary.can_feed(output.io_type(), input.io_type()) {
                continue;
            }
            ranked.push((
                pair_rank(output, input),
                output.name.clone(),
                input.name.clone(),
                output_index,
                input_index,
            ));
        }
    }
    ranked.sort();
    let mut evidence = DirectionEvidence::default();
    for (_, _, _, output_index, input_index) in ranked.into_iter().take(MAX_DERIVED_PORTS) {
        let output = &outputs[output_index];
        let input = &inputs[input_index];
        let port = PortBrief::new(output, input);
        if evidence.ports.contains(&port) {
            continue;
        }
        if !evidence.outputs.iter().any(|io| io.name == port.output) {
            evidence.outputs.push(IoBrief::from_io(output));
        }
        if !evidence.inputs.iter().any(|io| io.name == port.input) {
            evidence.inputs.push(IoBrief::from_io(input));
        }
        evidence.ports.push(port);
    }
    evidence
        .outputs
        .sort_by(|left, right| left.name.cmp(&right.name));
    evidence
        .inputs
        .sort_by(|left, right| left.name.cmp(&right.name));
    evidence
}

/// Ranks one accepted pair: exact normalized name, then a shared name token, then other.
fn pair_rank(output: &CapabilityIo, input: &CapabilityIo) -> u8 {
    let output_key = normalize_name_key(&output.name);
    let input_key = normalize_name_key(&input.name);
    if !output_key.is_empty() && output_key == input_key {
        return 0;
    }
    let output_tokens = tokenize(&output.name);
    let input_tokens = tokenize(&input.name);
    if output_tokens
        .iter()
        .any(|token| input_tokens.iter().any(|candidate| candidate == token))
    {
        return 1;
    }
    2
}

/// Truncates `text` to at most `limit` characters, never splitting a code point.
#[must_use]
pub fn truncate_chars(text: &str, limit: usize) -> String {
    if text.chars().count() <= limit {
        return text.to_string();
    }
    text.chars().take(limit).collect()
}

/// Inserts `text` under `key`, skipping a blank value (Python's `_prune_empty`).
fn insert_text(object: &mut Map<String, Value>, key: &str, text: &str) {
    if !text.is_empty() {
        object.insert(key.to_string(), Value::String(text.to_string()));
    }
}

/// Inserts a rendered list under `key`, skipping an empty one.
fn insert_array<T>(object: &mut Map<String, Value>, key: &str, values: &[T])
where
    T: RenderValue,
{
    if !values.is_empty() {
        let rendered = values.iter().map(RenderValue::render).collect();
        object.insert(key.to_string(), Value::Array(rendered));
    }
}

/// One renderable protocol value.
trait RenderValue {
    /// Renders the value as JSON.
    fn render(&self) -> Value;
}

impl RenderValue for IoBrief {
    fn render(&self) -> Value {
        self.to_value()
    }
}

impl RenderValue for PortBrief {
    fn render(&self) -> Value {
        self.to_value()
    }
}
