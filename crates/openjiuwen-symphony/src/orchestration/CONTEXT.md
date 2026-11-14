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

### Plans and planning

`plan.rs` holds `PlanStatus` (`ready` / `needs_input` / `no_plan`), `PlanStep`, `PlanEdge`, and
`OrchestrationPlan` with its JGF projection (`planned_graph`) and Kahn `stages()`.
`planner.rs` holds `PlanConfig` (`Fast` / `Beam`, `top_k`, `max_depth`, `min_edge_confidence`,
`max_expansions`, `max_plans`), `PlanRequest`, the `SemanticScorer` seam (default
`LexicalSemanticScorer`), `Planner::plan`, and the language helpers.

### Execution

`executor.rs` compiles a plan into an `openjiuwen_graph::CompiledGraph` and runs it through a
`CapabilityRunner`, recording per-step `StepOutcome`s; `ExecutorConfig` bounds steps, timeouts, and
fail-fast behaviour, and `UnsupportedRunner` is the no-runner default.

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
