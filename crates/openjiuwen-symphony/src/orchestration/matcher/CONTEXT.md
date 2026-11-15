# openjiuwen-symphony / orchestration / matcher

## Intent

The request/response protocol between a matcher model and the deterministic relation
candidates, mapping to Python `openjiuwen.symphony.orchestration.graph.matcher.protocol`.
`protocol.rs` renders the compact document a matcher model answers; `validate.rs` expands
that answer into full matches and validates every field against the candidate evidence;
`support.rs` holds the payload types both sides read and report.
The module never calls a model: the model-calling side is
[`crate::llm::ontology::OntologyMatcher`](../../llm/ontology.rs), and this is the contract
between the two.
Everything is offline and deterministic — no clock, no RNG, no hash-map iteration order.

## API Surface

`protocol.rs` — request building.

```rust
pub const MAX_SKILL_DESCRIPTION_LENGTH: usize = 240;   pub const MAX_PORT_DESCRIPTION_LENGTH: usize = 160;
pub const MAX_REASON_LENGTH: usize = 160;              pub const MAX_DERIVED_PORTS: usize = 8;
pub const DIRECTION_FORWARD: &str = "forward";         pub const DIRECTION_REVERSE: &str = "reverse";

pub struct SkillBrief { pub name: String, pub description: String }
pub struct IoBrief { pub name: String, pub io_type: String /* JSON "type" */, pub required: Option<bool>, pub description: String }
pub struct PortBrief { pub output: String, pub output_type: String, pub input: String, pub input_type: String }
pub struct DirectionEvidence { pub outputs: Vec<IoBrief>, pub inputs: Vec<IoBrief>, pub ports: Vec<PortBrief> }
pub struct CandidateContext { pub id: String, pub candidate_key: String /* serde skip */,
                              pub source: SkillBrief, pub target: SkillBrief,
                              pub directions: BTreeMap<String, DirectionEvidence> }
pub struct MatchRequest { pub candidates: Vec<CandidateContext> }
pub struct CandidateEvidence;   // (candidate key, direction) → DirectionEvidence

pub fn build_match_request(capabilities: &[CapabilityDescriptor], candidates: &[RelationCandidate], reverse_skill_order: bool) -> MatchRequest;
pub fn directional_evidence(source: &CapabilityDescriptor, target: &CapabilityDescriptor, candidate: &RelationCandidate, direction: &str) -> DirectionEvidence;
pub fn candidate_key(candidate: &RelationCandidate) -> String;   // "{source}->{target}"
pub fn truncate_chars(text: &str, limit: usize) -> String;        // char-boundary safe
```

`validate.rs` — response pipeline.

```rust
pub const DEFAULT_CAN_FEED_THRESHOLD: f64 = 0.7;
pub const DIAGNOSTIC_INVALID_MATCH_ITEM: &str = "invalid_match_item";
pub const DIAGNOSTIC_UNKNOWN_CANDIDATE_ID: &str = "unknown_candidate_id";
pub const DIAGNOSTIC_DUPLICATE_CANDIDATE_ID: &str = "duplicate_candidate_id";
pub const DIAGNOSTIC_INVALID_CANDIDATE_DIRECTION: &str = "invalid_candidate_direction";
pub const DIAGNOSTIC_UNSUPPORTED_CANDIDATE_DIRECTION: &str = "unsupported_candidate_direction";
pub const DIAGNOSTIC_INVALID_MATCHES_PAYLOAD: &str = "invalid_matches_payload";
pub const DIAGNOSTIC_REJECTED_LLM_MATCH: &str = "rejected_llm_match";
pub const DIAGNOSTIC_LOW_CONFIDENCE_LLM_MATCH: &str = "low_confidence_llm_match";

pub struct ValidatedMatch { pub candidate_id: Option<String>, pub source_id: String, pub target_id: String,
                            pub relation: String, pub confidence: f64, pub method: String,
                            pub reasons: Vec<String>, pub supporting_fields: SupportingFields,
                            pub accepted: bool, pub errors: Vec<String> }
pub struct MatchOutcome { pub matches: Vec<ValidatedMatch>, pub diagnostics: Vec<GraphDiagnostic> }
pub struct MatchThresholds { pub can_feed: f64 /*0.7*/ }

pub fn parse_match_response(payload: &Value, candidates: &[RelationCandidate]) -> MatchOutcome;
pub fn parse_match_response_with(payload: &Value, candidates: &[RelationCandidate], thresholds: &MatchThresholds) -> MatchOutcome;
pub fn parse_match_response_with_evidence(payload: &Value, candidates: &[RelationCandidate], evidence: &CandidateEvidence, thresholds: &MatchThresholds) -> MatchOutcome;
pub fn matches_by_candidate(matches: &[ValidatedMatch]) -> BTreeMap<String, Vec<&ValidatedMatch>>;
pub fn diagnostics_by_candidate(diagnostics: &[GraphDiagnostic]) -> BTreeMap<String, Vec<&GraphDiagnostic>>;
```

`support.rs` — response payload types.

```rust
pub struct GraphDiagnostic { pub code: String, pub message: String, pub detail: Option<Value> }
pub struct PortMappingPair { pub source_output: String, pub target_input: String }
pub struct SupportingFields { pub port_mappings: Vec<PortMappingPair>, pub source_outputs: Vec<String>, pub target_inputs: Vec<String> }

impl GraphDiagnostic { pub fn new(code, message) -> Self; pub fn with_detail(self, Value) -> Self;
                       pub fn candidate_id(&self) -> Option<&str> }
impl SupportingFields { pub fn from_value(&Value) -> Self; pub fn from_evidence(&DirectionEvidence) -> Self;
                        pub fn pairs(&self) -> BTreeSet<(String, String)>; pub fn is_empty(&self) -> bool;
                        pub fn completed_from_pairs(self, &BTreeSet<(String, String)>) -> Self; pub fn to_value(&self) -> Value }
```

`SupportingFields`' `Deserialize` is hand-written on top of `from_value`, so the same tolerant
reading applies to a payload and to a serde caller.

`MatchRequest::to_value()` is total (no fallible serialization) and stays in sync with the derived
`Serialize` impl; `MatchOutcome::{accepted_matches, ontology_matches}` and
`ValidatedMatch::to_ontology_match` bridge to `crate::llm::ontology::OntologyMatch`.

## Rules

Positional ids are `c1..cN` in input order; `reverse_skill_order` reverses the assignment (the last
input candidate becomes `c1`), and such a request is parsed against `candidates.iter().rev()`.
Skill descriptions truncate to 240 characters and port descriptions to 160, always on character
boundaries; blank values, an absent `required`, and empty collections are pruned from the document.
The forward direction uses the candidate's matched output/input pair when both resolve; the reverse
direction is derived by scanning the target's outputs against the source's inputs for
`DataTypeVocabulary::can_feed` compatibility, ranking an exact `normalize_name_key` match first, then
a shared name token, keeping at most `MAX_DERIVED_PORTS` pairs, and sorting outputs/inputs by name.
With no matched pair to resolve, the forward direction falls back to the same scan; a direction with
no accepted pair is omitted, and a candidate naming a capability outside the catalog is skipped with
a `tracing` warning instead of failing the request.

`parse_match_response*` runs Python's pipeline in order.
Explicit rejections come first: an `accepted` flag that is `false`, `"0"`, `"false"`, `"no"` or
blank forces the confidence to `0.0`; only `true`, `"1"`, `"true"` and `"yes"` accept.
Expansion then maps a compact item to its candidate: an unknown id, a repeated id (first wins), a
non-object item, a direction other than `forward`/`reverse`, or a direction without evidence is
dropped with a diagnostic; a surviving item gets `candidate_id`, the ids in the replied direction,
`relation_type = "can_feed"`, `method = "llm_ontology_match"`, the trimmed reason truncated to 160
characters, and supporting fields derived from that direction's evidence.
An item already carrying `candidate_id`/`source_id`/`target_id` skips expansion and is validated
directly (a Rust-only leniency; Python drops those).
Validation checks the endpoints against the candidate catalog, keeps the relation `can_feed` (a
full-form `relation_type` other than `can_feed` is an error), parses the confidence (a missing value
is `0`, an unparseable or non-finite one is `0` plus an error), clamps out-of-range confidences into
`[0, 1]` with an error, resolves the candidate by `candidate_id` then by the endpoint pair in either
orientation, and checks `can_feed` supporting fields: requested `port_mappings` must be a subset of
the evidence pairs, otherwise named outputs/inputs must intersect the evidence, otherwise the
evidence must hold some output/input pair.
Missing `source_outputs`/`target_inputs` are completed from the requested pairs, sorted and deduped.
`accepted` is "no errors and confidence at or above the relation threshold" (`1.0` for a relation
without an entry); a rejected match is kept with its `errors`.

Evidence for validation comes from the passed `CandidateEvidence` when it covers that direction, and
otherwise from the candidate's own matched pair, which certifies the forward direction only.
`parse_match_response` without an evidence table therefore reports `unsupported_candidate_direction`
for a reverse claim; passing `MatchRequest::evidence()` makes reverse verdicts judgeable.

Diagnostics carry `detail["candidate_id"]` whenever a candidate is known, so
`diagnostics_by_candidate` can attribute them; protocol diagnostics come first, then validation
diagnostics, each in payload order. `matches_by_candidate` keys on `candidate_id` and falls back to
`"{source_id}->{target_id}"`; both helpers return a `BTreeMap`, so iteration order is deterministic.

## Constraints

- No model call, no network, no clock and no RNG: the module is pure data transformation.
- Only `crate::error` types and `tracing` — no `println!`, no `unwrap`/`expect` outside tests.
- Public data types are `serde`-serializable with `snake_case` names, defaults, and unknown-field
  tolerance; the wire shapes keep Python's `"type"`, `output`/`output_type`/`input`/`input_type`
  port keys and `{"candidates": […]}`, `{"matches": […]}` envelopes.
- Files stay ≲1000 lines: `protocol.rs` builds the request, `validate.rs` runs the reply
  pipeline, `support.rs` holds the response payload types, `mod.rs` documents and re-exports.

## Routing Table

- `./mod.rs` → module docs, `protocol`/`support`/`validate` declarations and the public re-exports
- `./protocol.rs` → request building: briefs, `MatchRequest`, `CandidateEvidence`, `build_match_request`
- `./validate.rs` → reply pipeline: diagnostics codes, `MatchThresholds`, `ValidatedMatch`, `MatchOutcome`, `parse_match_response*`, grouping helpers
- `./support.rs` → response payload types: `GraphDiagnostic`, `PortMappingPair`, `SupportingFields` and their field readers
