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
pub struct ExecutorConfig { pub max_steps: usize /*32*/, pub node_timeout_ms: Option<u64> /*None*/, pub fail_fast: bool /*true*/ }
pub struct StepOutcome { pub step_id: String, pub capability_type: String, pub success: bool, pub output: Value, pub error: Option<String>, pub duration_ms: u64 }
pub struct PlanExecution { pub plan_id: String, pub graph_id: String, pub steps: BTreeMap<String, StepOutcome>, pub outputs: BTreeMap<String, Value>, pub state: Value }
impl PlanExecution { pub fn failed_steps(&self) -> Vec<&StepOutcome>; pub fn succeeded(&self) -> bool; }
impl PlanExecutor {
    pub fn new(runner: Arc<dyn CapabilityRunner>) -> Self;
    pub fn with_config(config: ExecutorConfig, runner: Arc<dyn CapabilityRunner>) -> Self;
    pub fn compile(&self, plan: &OrchestrationPlan, capabilities: &[CapabilityDescriptor]) -> Result<CompiledGraph>;
    pub async fn execute(&self, plan: &OrchestrationPlan, capabilities: &[CapabilityDescriptor], session: Option<Arc<dyn Session>>) -> Result<PlanExecution>;
    pub async fn execute_with_inputs(&self, plan: &OrchestrationPlan, capabilities: &[CapabilityDescriptor], inputs: Value, session: Option<Arc<dyn Session>>) -> Result<PlanExecution>;   // seeds state["inputs"]; `execute` delegates with Value::Null
}
```

`executor.rs` compiles a plan into an `openjiuwen_graph::CompiledGraph` and runs it through a
`CapabilityRunner`, recording per-step `StepOutcome`s; `ExecutorConfig` bounds steps, timeouts, and
fail-fast behaviour, and `UnsupportedRunner` is the no-runner default.

Execution mapping: each plan step becomes an `openjiuwen_graph` node whose id is the step id; plan edges become graph edges; entries are steps with no incoming edge and exits steps with no outgoing edge.
A step's inputs start from `state["inputs"]` and are overlaid by predecessor step outputs whose normalized name matches an input name (a predecessor with exactly one output also fills a single unmatched input).
Each node writes the top-level state key equal to its step id: `{"success", "output", "error", "duration_ms"}`.
With `fail_fast` the first runner error aborts with `Error::Graph("step `<id>` failed: ...")`; otherwise every step runs and `failed_steps()` reports failures.
`max_steps` is checked before compilation; the default session is `WorkflowSession::new(<uuid>)`; an empty plan executes to an empty `PlanExecution`.
A `Cancelled` or `Unsupported` runner error keeps its own `ErrorKind` instead of becoming `Error::Graph`.

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
