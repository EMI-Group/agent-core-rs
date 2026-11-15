# openjiuwen-symphony / orchestration

## Intent

Deterministic relation candidates, bounded planning, plan execution, and the graph-mutation
request contract, mapping to Python `openjiuwen.symphony.orchestration`.
Python's orchestration package is host-driven and split across planning, matching, artifacts, and a
mutation service; this crate ports the pure, offline-deterministic half of each area.
Everything here is offline by default: relation matching, planning, and mutation validation use
fixed rules, and the LLM-guided stages live behind the `CompletionModel` seam in `crate::llm`.
`mutation.rs` ports the pure half of Python's `mutations.py`: batch normalization, the request
digest, the change-set check, and the durable idempotency journal.
The mutation *service* (`mutation_service.py`, `artifacts.py`) is composed into
`crate::runtime::SymphonyRuntime` instead of being a separate orchestrator type.
`matcher/` ports `graph/matcher/protocol.py`: the compact request a matcher model is shown and the
expansion/validation of its reply, without any model call.

## API Surface

### Candidates

`candidates.rs` derives `can_feed` relation candidates with three deterministic strategies
(`ExactIo`, `SemanticOverlap`, `TextualCoercion`), each carrying a confidence and the matched IO
names; `CandidateGenerator::edges` projects them into `CapabilityEdge`s for the graph.
Candidates sort by strategy rank, confidence, source, then target, and each source keeps at most
`max_candidates_per_source`.

Rules (deterministic, sorted by strategy rank, then confidence desc, then source, then target):
`ExactIo` — an output and an input share the normalized name *and* normalized type: confidence `1.0`.
`SemanticOverlap` — token Jaccard of output text vs input text `>= min_token_overlap`: confidence `0.70 + 0.25 * overlap`.
`TextualCoercion` — `DataTypeVocabulary::can_feed(output.type, input.type)` with at least one shared name token: confidence `0.75`.
Candidates below `min_confidence` are dropped; each source keeps at most `max_candidates_per_source` (highest confidence first).
`CandidateConfig` defaults: `min_confidence 0.7`, `min_token_overlap 0.6`, `max_candidates_per_source 32`.

### Match request/response protocol

`matcher/` is the protocol between a matcher model and the deterministic candidates: it builds the
compact document the model answers and it expands and validates the reply.
It never calls a model — `crate::llm::OntologyMatcher` is the model-calling side.
See [`matcher/CONTEXT.md`](matcher/CONTEXT.md) for the per-function rules.

```rust
pub const MAX_SKILL_DESCRIPTION_LENGTH: usize = 240;   pub const MAX_PORT_DESCRIPTION_LENGTH: usize = 160;
pub const MAX_REASON_LENGTH: usize = 160;              pub const MAX_DERIVED_PORTS: usize = 8;
pub const DEFAULT_CAN_FEED_THRESHOLD: f64 = 0.7;
pub const DIRECTION_FORWARD: &str = "forward";         pub const DIRECTION_REVERSE: &str = "reverse";

pub struct SkillBrief { pub name: String, pub description: String }
pub struct IoBrief { pub name: String, pub io_type: String /* JSON "type" */, pub required: Option<bool>, pub description: String }
pub struct PortBrief { pub output: String, pub output_type: String, pub input: String, pub input_type: String }
pub struct DirectionEvidence { pub outputs: Vec<IoBrief>, pub inputs: Vec<IoBrief>, pub ports: Vec<PortBrief> }
pub struct CandidateContext { pub id: String, pub candidate_key: String /* off the wire */,
                              pub source: SkillBrief, pub target: SkillBrief,
                              pub directions: BTreeMap<String, DirectionEvidence> }
pub struct MatchRequest { pub candidates: Vec<CandidateContext> }
impl MatchRequest {
    pub fn new(candidates: Vec<CandidateContext>) -> Self;
    pub fn candidates(&self) -> &[CandidateContext];
    pub fn to_value(&self) -> Value;                 // {"candidates": […]} — total, pruned
    pub fn evidence(&self) -> CandidateEvidence;     // (candidate key, direction) → evidence
}
pub struct CandidateEvidence;                        // new / insert / direction / is_empty / len
pub fn candidate_key(candidate: &RelationCandidate) -> String;    // "{source}->{target}"
pub fn build_match_request(capabilities: &[CapabilityDescriptor], candidates: &[RelationCandidate], reverse_skill_order: bool) -> MatchRequest;
pub fn directional_evidence(source: &CapabilityDescriptor, target: &CapabilityDescriptor, candidate: &RelationCandidate, direction: &str) -> DirectionEvidence;
pub fn truncate_chars(text: &str, limit: usize) -> String;        // char-boundary safe

pub struct GraphDiagnostic { pub code: String, pub message: String, pub detail: Option<Value> }   // candidate_id() = detail["candidate_id"]
pub struct PortMappingPair { pub source_output: String, pub target_input: String }
pub struct SupportingFields { pub port_mappings: Vec<PortMappingPair>, pub source_outputs: Vec<String>, pub target_inputs: Vec<String> }
pub struct ValidatedMatch { pub candidate_id: Option<String>, pub source_id: String, pub target_id: String,
                            pub relation: String, pub confidence: f64, pub method: String, pub reasons: Vec<String>,
                            pub supporting_fields: SupportingFields, pub accepted: bool, pub errors: Vec<String> }
pub struct MatchOutcome { pub matches: Vec<ValidatedMatch>, pub diagnostics: Vec<GraphDiagnostic> }
pub struct MatchThresholds { pub can_feed: f64 /*0.7*/ }

pub fn parse_match_response(payload: &Value, candidates: &[RelationCandidate]) -> MatchOutcome;
pub fn parse_match_response_with(payload: &Value, candidates: &[RelationCandidate], thresholds: &MatchThresholds) -> MatchOutcome;
pub fn parse_match_response_with_evidence(payload: &Value, candidates: &[RelationCandidate], evidence: &CandidateEvidence, thresholds: &MatchThresholds) -> MatchOutcome;
pub fn matches_by_candidate(matches: &[ValidatedMatch]) -> BTreeMap<String, Vec<&ValidatedMatch>>;
pub fn diagnostics_by_candidate(diagnostics: &[GraphDiagnostic]) -> BTreeMap<String, Vec<&GraphDiagnostic>>;
```

Ids are `c1..cN` in input order; `reverse_skill_order` reverses the assignment, so the last input
candidate becomes `c1`, and such a request is parsed against the reversed candidate order.
Descriptions truncate on character boundaries (240 for a skill, 160 for a port), and empty fields
are pruned from the document.
The forward direction uses the candidate's matched output/input pair; the reverse direction is
derived by scanning target outputs against source inputs for `can_feed` type compatibility,
preferring an exact `normalize_name_key` match, then a shared name token, capped at
`MAX_DERIVED_PORTS` — a Rust-only substitution for Python's generator-provided `evidence` dict.
A direction without a pair is omitted, and a candidate naming an unknown capability is skipped with
a `tracing` warning.

`parse_match_response*` applies explicit rejections (`false`/`"0"`/`"no"` force the confidence to
`0`), expands compact items (`direction: "reverse"` swaps the ids, the relation is always
`can_feed`, the method is always `llm_ontology_match`, the reason is trimmed and truncated to 160),
then validates endpoints, relation, confidence (unparseable → `0` plus an error, out of range →
clamped plus an error) and `can_feed` supporting fields against the candidate evidence.
`accepted` is "no errors *and* confidence ≥ the relation threshold"; rejected matches stay in
`matches` with their `errors`.
Diagnostics use the codes `invalid_match_item`, `unknown_candidate_id`, `duplicate_candidate_id`,
`invalid_candidate_direction`, `unsupported_candidate_direction`, `invalid_matches_payload`,
`rejected_llm_match` and `low_confidence_llm_match` — protocol diagnostics first, then validation
diagnostics, each in payload order.
`ValidatedMatch::to_ontology_match` projects a verdict onto `crate::llm::OntologyMatch`.

### Plans and planning

```rust
#[serde(rename_all = "snake_case")] pub enum PlanStatus { Ready, NeedsInput, NoPlan }
pub struct PlanEdge { pub source: String, pub target: String }
pub struct PlanStep {
    pub step_id: String,             // qualified capability id
    pub capability_type: String, pub name: String,
    pub inputs: Vec<CapabilityIo>, pub outputs: Vec<CapabilityIo>,
    pub missing_inputs: Vec<String>, pub filled_inputs: Vec<String>, pub score: f64,
}
pub struct OrchestrationPlan {
    pub plan_id: String, pub query: String, pub language: String,   // "cn" | "en"
    pub status: PlanStatus, pub title: String, pub message: Option<String>,
    pub steps: Vec<PlanStep>, pub edges: Vec<PlanEdge>, pub score: f64,
}
impl OrchestrationPlan {
    pub fn planned_graph(&self) -> Value;            // {"graph": {id, type:"planned_graph", directed:true, nodes, edges}}
    pub fn to_value(&self) -> Value;                 // {"plan_id",...,"steps",...,"edges",...,"planned_graph"}
    pub fn from_value(value: &Value) -> Result<Self>;
    pub fn stages(&self) -> Result<Vec<Vec<String>>>; // Kahn layering; cycle -> Error::Validation
    pub fn is_empty(&self) -> bool;
}

#[serde(rename_all = "snake_case")] pub enum PlanMode { Fast, Beam }   // default Fast
pub struct PlanConfig {
    pub mode: PlanMode, pub top_k: usize /*3*/, pub max_depth: usize /*4*/,
    pub min_edge_confidence: f64 /*0.7*/, pub max_expansions: usize /*128*/,
    pub max_plans: usize /*256*/, pub dynamic_graph_enabled: bool /*false*/,
}
pub struct PlanRequest {
    pub query: String, pub candidate_ids: Option<Vec<String>>, pub disabled_ids: Vec<String>,
    pub provided_inputs: Vec<String>, pub language: String /*"cn"*/,
    pub top_k: Option<usize>, pub max_depth: Option<usize>,
}
#[async_trait] pub trait SemanticScorer: Send + Sync {
    async fn score(&self, query: &str, capabilities: &[CapabilityDescriptor]) -> Result<BTreeMap<String, f64>>;
}
pub struct LexicalSemanticScorer;    // deterministic default: query-term coverage in [0,1]
impl Planner {
    pub fn new(config: PlanConfig) -> Self;
    pub fn with_scorer(config: PlanConfig, scorer: Arc<dyn SemanticScorer>) -> Self;
    pub async fn plan(&self, request: &PlanRequest, graph: &CapabilityGraph, capabilities: &[CapabilityDescriptor]) -> Result<OrchestrationPlan>;
}
pub fn normalize_language(raw: &str) -> String;   // "zh"|"cn"|"zh_cn" -> "cn"; "en" -> "en"; other -> "cn"
pub fn plan_title(query: &str, language: &str) -> String;   // "计划：{query}" / "Plan: {query}"
```

`plan.rs` holds `PlanStatus` (`ready` / `needs_input` / `no_plan`), `PlanStep`, `PlanEdge`, and
`OrchestrationPlan` with its JGF projection (`planned_graph`) and Kahn `stages()`.
`planner.rs` holds `PlanConfig` (`Fast` / `Beam`, `top_k`, `max_depth`, `min_edge_confidence`,
`max_expansions`, `max_plans`), `PlanRequest`, the `SemanticScorer` seam (default
`LexicalSemanticScorer`), `Planner::plan`, and the language helpers.

Scoring (Python's formula, deterministic substitute scorer):
`score = 0.45*mean(semantic) + 0.25*mean(edge_confidence) + 0.20*input_coverage - 0.03*(steps - 1)`.
`input_coverage` = distinct step inputs produced by an earlier step or listed in `provided_inputs`, divided by all distinct step inputs (`1.0` when there are none).
`mean(semantic)` is the mean scorer value over the path's steps, falling back to Python's `SEED_BASELINE_SCORE` (`0.5`) when no step is scored; `mean(edge_confidence)` is the mean edge weight along the path, `1.0` when the path has no edges.
A step input is *missing* when it is required (`required != Some(false)`) and neither an earlier step output (normalized name plus compatible type) nor a `provided_inputs` name supplies it; `missing_inputs`/`filled_inputs` hold sorted, deduped names.
`candidate_ids`/`disabled_ids` prune expansion targets as well as the start set; `available` gates the start set only, so a withdrawn capability can still be a downstream step.
`plan_id` is a fresh `uuid_hex()`, so determinism holds for every field except `plan_id` and the embedded graph id.
Rank key: `(-score, missing_inputs, step_count, step_ids, edge_weights)`; non-finite scores are sanitized to `0.0`.
Start set: capabilities with a positive scorer value (all when the query has no tokens), minus `disabled_ids`, intersected with `candidate_ids` when given, `available` only.
Expansion follows graph edges with `weight >= min_edge_confidence`, never revisiting a step, bounded by `max_depth`, `max_expansions` and `max_plans`; `Fast` keeps one greedy best partial path per start, `Beam` keeps the best `top_k` partial paths per depth.
`status` is `Ready` when the best plan's root steps have no missing inputs, `NeedsInput` when a plan exists but a root input is unfilled (listed in `message`), and `NoPlan` when no candidate plan exists.

### Execution

```rust
pub struct RunnerContext { pub step_id: String, pub session_id: Option<String>, pub cancellation: CancellationToken }
#[async_trait] pub trait CapabilityRunner: Send + Sync {
    async fn invoke(&self, capability: &CapabilityDescriptor, inputs: Value, context: &RunnerContext) -> Result<Value>;
}
pub struct UnsupportedRunner;          // invoke -> Error::Unsupported
pub struct ExecutorConfig { pub max_steps: usize /*32*/, pub node_timeout_ms: Option<u64> /*None*/, pub fail_fast: bool /*true*/, pub fail_on_deferral: bool /*false*/ }
pub struct StepOutcome { pub step_id: String, pub capability_type: String, pub success: bool, pub output: Value, pub error: Option<String>, pub error_kind: Option<String>, pub deferred: bool, pub duration_ms: u64 }
pub struct PlanExecution { pub plan_id: String, pub graph_id: String, pub steps: BTreeMap<String, StepOutcome>, pub outputs: BTreeMap<String, Value>, pub state: Value }
impl PlanExecution { pub fn failed_steps(&self) -> Vec<&StepOutcome>; pub fn deferred_steps(&self) -> Vec<&StepOutcome>; pub fn has_deferrals(&self) -> bool; pub fn succeeded(&self) -> bool; }
impl PlanExecutor {
    pub fn new(runner: Arc<dyn CapabilityRunner>) -> Self;
    pub fn with_config(config: ExecutorConfig, runner: Arc<dyn CapabilityRunner>) -> Self;
    pub fn compile(&self, plan: &OrchestrationPlan, capabilities: &[CapabilityDescriptor]) -> Result<CompiledGraph>;
    pub async fn execute(&self, plan: &OrchestrationPlan, capabilities: &[CapabilityDescriptor], session: Option<Arc<dyn Session>>) -> Result<PlanExecution>;
    pub async fn execute_with_inputs(&self, plan: &OrchestrationPlan, capabilities: &[CapabilityDescriptor], inputs: Value, session: Option<Arc<dyn Session>>) -> Result<PlanExecution>;   // seeds state["inputs"]; `execute` delegates with Value::Null
}
```

`executor.rs` compiles a plan into an `openjiuwen_graph::CompiledGraph` and runs it through a
`CapabilityRunner`, recording per-step `StepOutcome`s; `ExecutorConfig` bounds steps, timeouts,
fail-fast and strict-deferral behaviour, and `UnsupportedRunner` is the no-runner default.

Execution mapping: each plan step becomes an `openjiuwen_graph` node whose id is the step id; plan edges become graph edges; entries are steps with no incoming edge and exits steps with no outgoing edge.
A step's inputs start from `state["inputs"]` and are overlaid by predecessor step outputs whose normalized name matches an input name (a predecessor with exactly one output also fills a single unmatched input).
Each node writes the top-level state key equal to its step id: `{"success", "output", "error", "error_kind", "deferred", "duration_ms"}`.
`error_kind` is the failure's `ErrorKind` snake_case name (`"tool"`, `"timeout"`, `"graph"`, …) and stays `null` on success; `deferred` is `true` exactly when a failed step's kind is `"unsupported"`, so a typed capability gap is distinguishable from a runtime error without parsing the message, and both fields serde-default to `None`/`false` for a payload written before them.
With `fail_fast` the first non-deferred runner error aborts with `Error::Graph("step `<id>` failed: ...")`; otherwise every step runs and the returned `PlanExecution` records every failure.
`PlanExecution::failed_steps` lists every failed step (deferred ones included), `deferred_steps` isolates the typed capability gaps, `has_deferrals` reports whether any exist, and `succeeded` is `false` for a deferred step.
`ExecutorConfig::fail_on_deferral` (default `false`) instead aborts the run on the first deferred step with `Error::Unsupported("step `<id>` deferred: ...")`, regardless of `fail_fast`, so an opted-in caller never receives a partial plan silently.
`max_steps` is checked before compilation; the default session is `WorkflowSession::new(<uuid>)`; an empty plan executes to an empty `PlanExecution`.
A `Cancelled` or `Unsupported` runner error keeps its own `ErrorKind` instead of becoming `Error::Graph`.
No degradation is silent: an unfilled required input, a non-object seed, a declared predecessor without a record and a failed predecessor each emit a `tracing::warn!`, and a step the graph never recorded is reported as `success: false` with `error_kind: "graph"`.

### Mutation requests

```rust
pub const REQUEST_SCHEMA_VERSION: &str = "Symphony-skill-graph-mutation-request-v2";
pub const JOURNAL_SCHEMA_VERSION: &str = "Symphony-skill-graph-mutation-requests-v1";
pub const MAX_REQUEST_ID_CHARS: usize = 256;
pub const MAX_SOURCE_REVISION_CHARS: usize = 512;
pub const MAX_CHANGED_IDS: usize = 1024;

pub enum MutationOperation { Add, Update, Delete }   // add | update | delete; `as_str()`

pub struct MutationRequest {
    pub request_id: String, pub operation: MutationOperation,
    pub source_revision: Option<String>, pub changed_skill_ids: Vec<String>,
}
impl MutationRequest {
    pub fn prepare(operation, changed_skill_ids: &[String], request_id: &str, source_revision: Option<&str>) -> Result<Self>;
    pub fn digest(&self) -> String;                    // request_digest(self)
    pub fn declared_ids(&self) -> BTreeSet<&str>;      // trimmed, non-blank ids
}

pub fn request_digest(request: &MutationRequest) -> String;
pub fn validate_mutation_against_inventory(request: &MutationRequest, changed: &[String], inventory_ids: &[String]) -> Result<()>;

pub struct JournalRecord {
    pub digest: String, pub operation: MutationOperation, pub revision: String,
    pub inserted_at: String, pub result: Option<Value>,
}
impl JournalRecord {
    pub fn new(operation, revision, digest) -> Self;   // stamps inserted_at
    pub fn with_inserted_at(self, inserted_at) -> Self;
    pub fn with_result(self, result: Value) -> Self;
}

pub struct MutationJournal { /* path + BTreeMap<String, JournalRecord> */ }
impl MutationJournal {
    pub fn new(path: impl Into<PathBuf>) -> Self;
    pub async fn open(path: impl Into<PathBuf>) -> Result<Self>;
    pub fn path(&self) -> &Path;
    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
    pub fn contains(&self, request_id: &str) -> bool;
    pub fn request_ids(&self) -> impl Iterator<Item = &str>;
    pub async fn lookup(&self, request_id: &str) -> Option<JournalRecord>;
    pub async fn record(&mut self, request_id: impl Into<String>, record: JournalRecord) -> Result<()>;
    pub async fn clear(&mut self) -> Result<()>;
}
```

`prepare` trims and sorts the ids and rejects a blank or oversized `request_id`
(1..=`MAX_REQUEST_ID_CHARS`), a blank or oversized `source_revision` (1..=`MAX_SOURCE_REVISION_CHARS`),
an empty batch (including a `delete` without ids), a batch over `MAX_CHANGED_IDS`, a blank id, and
duplicate ids after trimming; every rejection is `Error::Validation` naming the field.
`changed_skill_ids` are qualified capability ids (`type:id`), so they match
`GraphMutationResult::changed`.

`request_digest` is the SHA-256 hex of the canonical (sorted-key, compact) JSON of
`schema_version`, `request_id`, `operation`, `source_revision` (JSON `null` when undeclared), and
the sorted `changed_skill_ids`. It is the replay identity, covers every field, and is insensitive to
id order — but it covers the *declared ids*, not the descriptor payloads, exactly like Python's.
`validate_mutation_against_inventory` requires the declared batch to equal the applied change set
exactly (naming the missing and extra ids) and then checks membership: `add`/`update` ids must be
present in the inventory, `delete` ids absent from it.

The journal document is `{"schema_version": "...mutation-requests-v1", "records": {request_id:
{digest, operation, revision, inserted_at, result?}}}`. Records are held in a `BTreeMap` (sorted by
request id, byte-stable), and every write goes through `crate::flow::store::atomic_write` (a
temporary sibling plus a rename). `inserted_at` is metadata only. `result` is an additive Rust
extension carrying the recorded public result, so a reopened runtime answers a replay with the exact
recorded `GraphMutationResult`; Python stores that result inside the graph artifact instead.
An absent or empty file opens empty; malformed JSON, a non-object envelope, and a missing or
different `schema_version` are `Error::Validation` naming `mutation_journal_path`; a malformed
individual record is warned about and skipped.

## Constraints

- Every ordering is explicit: candidates sort on the strategy/confidence/source/target tuple, plans
  on `(-score, missing_inputs, step_count, step_ids, edge_weights)`, graph ids are qualified
  capability ids, and a digest never depends on hash-map order.
- Determinism forbids clocks in identities: `inserted_at` never enters a digest or a lookup, and the
  journal's ordering comes from the sorted request ids.
- The planner and executor never mutate the catalog; only the runtime's mutation path does, and it
  does so under the runtime's mutation lock.
- Validation failures are typed errors, never silent no-ops; `delete` skipping an unregistered key is
  the documented exception at the runtime level, not here.
- Files stay ≲1000 lines and split by responsibility (`mutation.rs` is the request contract only).

## Notes for Agents

- Python's `MutationJournal` is backed by the graph artifact store; here it owns one JSON file, so
  the runtime composes it with the artifact backend instead of nesting it inside.
- Python's `validate_mutation_against_inventory` compares against a provider change set; the port
  takes the applied change set and the resulting inventory as arguments, which is what an in-process
  registry can observe.
- Python's `graph_delta`, `mutation_version`, the artifact staging/`activate_if_current` CAS, and the
  matcher/build-protocol identity checks live in the runtime's `publish_graph`/`mutate` path (an
  immutable revision chain plus `source_revision` compare-and-swap) rather than here.

## Routing Table

- `./candidates.rs` → `CandidateConfig`, `CandidateGenerator`, `MatchStrategy`, `RelationCandidate`
- `./plan.rs` → `OrchestrationPlan`, `PlanStep`, `PlanEdge`, `PlanStatus`, JGF projection, `stages`
- `./planner.rs` → `Planner`, `PlanConfig`, `PlanMode`, `PlanRequest`, `SemanticScorer`, `LexicalSemanticScorer`, language helpers
- `./executor.rs` → `PlanExecutor`, `ExecutorConfig`, `CapabilityRunner`, `RunnerContext`, `StepOutcome`, `PlanExecution`, `UnsupportedRunner`
- `./mutation.rs` → `MutationOperation`, `MutationRequest`, `request_digest`, `validate_mutation_against_inventory`, `JournalRecord`, `MutationJournal`
- `./matcher/` → the matcher request/response protocol: request building (`protocol.rs`) and reply expansion/validation (`validate.rs`); see [`matcher/CONTEXT.md`](matcher/CONTEXT.md)
