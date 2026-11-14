# openjiuwen-symphony / runtime

## Intent

The composition root: lifecycle, catalog/graph publication, planning and invocation, lazy semantic-index synchronization, evolution feedback, and the capability-mutation path with its journal-backed replay and source-revision compare-and-swap.
`mod.rs` owns the runtime state machine and the read/plan/invoke surface; `mutations.rs` owns the mutation order, `GraphMutationResult` and the journal wiring.
Everything is offline and deterministic: mutation identity comes from request digests, artifact revisions from the catalog, and no clock value decides behaviour.

## API Surface

```rust
#[serde(rename_all = "snake_case")] pub enum RuntimeState { Created, Running, Stopped }

pub struct SymphonyConfig {
    pub graph_artifact_root: Option<PathBuf>,   // None = in-memory artifacts
    pub graph_scope_id: String,                 // default "default"
    pub mutation_journal_path: Option<PathBuf>, // None = in-memory request ledger
    pub plan: PlanConfig,
    pub executor: ExecutorConfig,
    pub candidates: CandidateConfig,
}
impl Default for SymphonyConfig;

pub struct SymphonyRuntimeBuilder;
impl SymphonyRuntimeBuilder {
    pub fn new(config: SymphonyConfig) -> Self;
    pub fn with_registry(self, registry: Arc<CapabilityRegistry>) -> Self;
    pub fn with_asset_store(self, store: Arc<dyn AssetStore>) -> Self;
    pub fn with_artifacts(self, artifacts: Arc<dyn GraphArtifactStore>) -> Self;
    pub fn with_runner(self, runner: Arc<dyn CapabilityRunner>) -> Self;
    pub fn with_planner(self, planner: Planner) -> Self;
    pub fn with_semantic_index(self, index: Arc<SemanticIndex>) -> Self;
    pub fn with_mutation_journal(self, journal: MutationJournal) -> Self;   // takes precedence over the config path
    pub async fn open(self) -> Result<SymphonyRuntime>;
}

pub struct SymphonyRuntime;
impl SymphonyRuntime {
    pub async fn open(config: SymphonyConfig) -> Result<Self>;
    pub fn builder(config: SymphonyConfig) -> SymphonyRuntimeBuilder;
    pub fn config(&self) -> &SymphonyConfig;
    pub fn state(&self) -> RuntimeState;
    pub fn registry(&self) -> &Arc<CapabilityRegistry>;
    pub async fn start(&self) -> Result<()>;
    pub async fn stop(&self) -> Result<()>;
    pub async fn close(&self) -> Result<()>;
    pub async fn build(&self) -> Result<GraphRevision>;
    pub async fn graph(&self) -> CapabilityGraph;
    pub async fn graph_status(&self) -> Result<GraphArtifactStatus>;
    pub async fn read_graph(&self, version: Option<&str>) -> Result<Option<CapabilityGraph>>;
    pub async fn plan(&self, request: &PlanRequest) -> Result<OrchestrationPlan>;
    pub async fn invoke(&self, request: InvokeRequest) -> Result<InvokeResponse>;
    pub async fn search(&self, request: &SearchRequest) -> Result<Vec<SearchHit>>;
    pub async fn semantic_search(&self, query: &str, top_k: usize) -> Result<Vec<SearchHit>>;   // lazily synchronized; `top_k == 0` = unlimited
    pub async fn reindex_semantics(&self) -> Result<usize>;          // force a full re-embed; returns the indexed count
    pub async fn mutate(&self, request: MutationRequest, descriptors: Vec<CapabilityDescriptor>) -> Result<GraphMutationResult>;
    pub async fn add_capabilities(&self, descriptors: Vec<CapabilityDescriptor>, request_id: &str) -> Result<GraphMutationResult>;   // delegates to `mutate`
    pub async fn update_capabilities(&self, descriptors: Vec<CapabilityDescriptor>, request_id: &str) -> Result<GraphMutationResult>;
    pub async fn remove_capabilities(&self, keys: &[CapabilityKey], request_id: &str) -> Result<GraphMutationResult>;
    pub async fn submit_evolution(&self, execution: &PlanExecution) -> Result<EvolutionResult>;
    pub fn capture_graph_snapshot(&self) -> BTreeMap<String, Value>;
}

pub struct InvokeRequest {
    pub query: String, pub candidate_ids: Option<Vec<String>>, pub disabled_ids: Vec<String>,
    pub execute: bool, pub inputs: Value, pub session: Option<Arc<dyn Session>>,
}
impl InvokeRequest { pub fn new(query: impl Into<String>) -> Self; pub fn execute(self) -> Self; pub fn with_inputs(self, inputs: Value) -> Self; pub fn with_session(self, session: Arc<dyn Session>) -> Self; }
pub struct InvokeResponse { pub plan: OrchestrationPlan, pub execution: Option<PlanExecution>, pub graph_status: GraphArtifactStatus }

#[serde(rename_all = "snake_case")] pub enum MutationOperation { Add, Update, Delete }
#[serde(rename_all = "snake_case")] pub enum MutationStatus { Published, Noop, Replayed }   // default Published
pub struct GraphMutationResult {
    pub operation: MutationOperation, pub request_id: String, pub changed: Vec<String>,
    pub added_edges: usize, pub removed_edges: usize, pub revision: String, pub replayed: bool,
    // additive, serde-defaulted bookkeeping
    pub status: MutationStatus, pub digest: String,
    pub source_revision: Option<String>, pub diagnostics: Vec<String>,
}
pub struct EvolutionResult { pub accepted_edges: usize, pub rejected_edges: usize, pub revision: String }
```

Lifecycle: `Created` → `start` → `Running` → `stop`/`close` → `Stopped`; both `start` and `stop` are idempotent and `close` is `stop` plus artifact flush, and `start` also restarts a `Stopped` runtime.
`start` builds and publishes the graph when no revision exists, flipping to `Running` only after the build succeeds; `build` re-derives the graph from the registry plus candidate edges and publishes a new revision.
`add/update/remove_capabilities` delegate to `mutate` with a prepared request; the mutations mutate the registry, re-derive and publish the graph, and are idempotent per `request_id` (a replay returns the recorded result with `replayed = true` and `status = replayed`).
The mutation path is ordered: request preparation → replay lookup (in-memory ledger, then the journal) → digest-conflict check → `Running` gate → active-state snapshot → `source_revision` compare-and-swap → apply (add/update validate the whole batch first) → change-set check → publish → record.
`request_digest` is SHA-256 over canonical JSON of `{schema_version, request_id, operation, source_revision, changed_skill_ids}`, so it covers the declared id batch and not the descriptor payloads: the same batch under the same id replays, changed content under a new id does not.
A replay whose digest differs from the recorded one is `Error::Validation` naming `request_id`; a declared `source_revision` that is not the current artifact revision is `Error::Validation` naming `source_revision`.
The ledger is in memory by default; `SymphonyConfig::mutation_journal_path` or `SymphonyRuntimeBuilder::with_mutation_journal` makes it durable (see [`orchestration::mutation`](src/orchestration/mutation.rs)), so a runtime rebuilt over the same journal still answers replays.
`invoke` and the three mutations require `Running` (`Error::Validation` naming `runtime.state` otherwise); planning, graph reads, search, build, evolution and the accessors work in every state.
`invoke` plans, optionally executes when `execute` is set, and returns both; a `NoPlan` result with `execute` set returns `execution: None` rather than an error.
A blank `request_id` is `Error::Validation`; `update` requires every key to exist (`Error::NotFound`, whole batch rejected) while `remove` skips absent keys.
`graph()` is infallible: when the artifact read fails it logs a warning and answers with the catalog-derived, node-only graph.
`submit_evolution` applies successful step transitions as an EMA on edge weights (`w' = (w + 1.0) / 2`), publishes only when a weight changed, and reports accepted/rejected edges.
`capture_graph_snapshot` returns `{"current_version", "graph_id", "graph_scope_id"}` with `null` values when no revision exists.
`SymphonyRuntimeBuilder::with_asset_store` hydrates the registry from the store, so a preconfigured asset store survives restarts.
`semantic_search` clears and re-embeds the catalog only when `CapabilityRegistry::revision()` differs from the last synchronization; `reindex_semantics` force-re-embeds and returns the count; both serialize through one `tokio::sync::Mutex`.
Without a semantic index both entry points return `Error::Unsupported`.

## Constraints

- Every mutation holds no `std::sync` guard across an `.await`; the mutation lock is one `tokio::sync::Mutex` covering lookup, apply, publish and record, so two batches can never interleave.
- `invoke` and the three mutations require `RuntimeState::Running`; planning, graph reads, search, build, evolution and the accessors work in every state.
- Determinism: a replay answers the recorded result, the digest never covers payloads, and `inserted_at` is metadata only.
- `graph()` is infallible (a failed artifact read logs a warning and answers the catalog-derived node-only graph); `build`/`mutate` propagate the artifact backend's errors.

## Routing Table

- `./mod.rs` → `SymphonyConfig`, `SymphonyRuntimeBuilder`, `SymphonyRuntime`, lifecycle, `plan`/`invoke`/`search`, `build`, evolution, snapshots
- `./mutations.rs` → `MutationStatus`, `GraphMutationResult`, the mutation path (`mutate`), journal-backed replay, digest conflict, `source_revision` CAS
