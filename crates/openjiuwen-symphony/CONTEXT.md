# openjiuwen-symphony

## Intent

The capability-asset engine of openJiuwen, mapping to the Python package `openjiuwen.symphony` (v0.1.20, ~52k lines, 201 files).
Python's engine is host-driven: it fingerprints capability assets, indexes them, builds a capability relationship graph, plans asset compositions with an LLM, and distils observed flows into skills.
This crate ports that engine end to end: the asset cards, a real registry with CRUD and search, keyword and embedding-backed retrieval, SKILL.md discovery with a lexical skill catalog, a versioned `can_feed` capability graph with atomic artifact publication in-memory, on disk and in an object store, a bounded deterministic planner, a plan executor over `openjiuwen-graph`, flow distillation into versioned recipes and reviewed `SkillPack` artifacts, a deduplicating experience bank, and a persisted embedding-backed experience index.
Every LLM-guided stage (progressive tree retrieval, ontology matching, semantic plan scoring, fingerprint extraction, LLM judging, flow narratives) sits behind the `CompletionModel` seam and carries a documented deterministic fallback, so the crate is fully usable offline with no model configured.
Everything here is offline, deterministic, and free of external services by default.
Graph artifacts publish to memory, disk, and three object-storage backends (S3/OBS, Google Cloud Storage, Azure Blob).
Evaluation covers static card metrics, trace metrics and case windows; analysis runs over ingested session traces; credentials are redacted before text reaches an artifact.
The typed `Error::Unsupported` gaps that remain are the GPU/ML ones (faiss IVF index training in `ExperienceIndex`, `transformers`/`vLLM` logit selection in [`deferred`]) and two vendor credential flows that need a signing/issuer stack (`GcsObjectStore::from_service_account_file`, `AzureBlobStore::from_client_credentials`).
A few pure-logic Python areas are still unported; they are listed under `## Deferred Scope` and are absent rather than stubbed.

## API Surface

All paths below are crate-relative. Every public item is documented; `cargo clippy -D warnings` includes `missing_docs`.

### `error` (`src/error.rs`)

```rust
pub use openjiuwen_common::error::{Error, ErrorKind, Result};

pub fn unsupported(feature: &str) -> Error;                       // Error::Unsupported with a pointer to Deferred Scope
pub fn validation(field: &str, reason: impl fmt::Display) -> Error; // Error::Validation "<field>: <reason>"
pub fn is_unsupported(err: &Error) -> bool;
```

### `model` (`src/model/`)

```rust
pub struct CapabilityKey { pub capability_type: String, pub capability_id: String }
impl CapabilityKey {
    pub fn new(capability_type: impl Into<String>, capability_id: impl Into<String>) -> Self;
    pub fn qualified(&self) -> String;          // "{capability_type}:{capability_id}"
}
impl fmt::Display for CapabilityKey;            // qualified()
impl std::str::FromStr for CapabilityKey;       // splits on the first ':' else Error::Validation

pub struct CapabilityIo {
    pub name: String,
    #[serde(rename = "type")] pub io_type: String,   // JSON key is "type"; accessor `io_type()`
    pub description: String,
    pub required: Option<bool>,
    pub default: Option<Value>,
    pub metadata: BTreeMap<String, Value>,
}
impl CapabilityIo {
    pub fn new(name: impl Into<String>, io_type: impl Into<String>) -> Self;
    // with_description / with_required / with_default / with_metadata
}

pub struct CapabilityDescriptor {
    pub capability_id: String, pub capability_type: String, pub name: String,
    pub description: String, pub source: String, pub available: bool,
    pub inputs: Vec<CapabilityIo>, pub outputs: Vec<CapabilityIo>,
    pub classification: String, pub tags: Vec<String>, pub content_hash: String,
    #[serde(default, skip_serializing)] pub semantic_content: String, // Python excludes it from dumps
    pub metadata: BTreeMap<String, Value>,
}
impl CapabilityDescriptor {
    pub fn new(capability_type: impl Into<String>, capability_id: impl Into<String>, name: impl Into<String>) -> Self;
    // with_description / with_source / with_available / with_classification / with_tags /
    // with_inputs / with_outputs / with_semantic_content / with_metadata
    pub fn key(&self) -> CapabilityKey;
    pub fn qualified_id(&self) -> String;
    pub fn validate(&self) -> Result<()>;          // non-blank capability_id/capability_type/name
    pub fn retrieval_text(&self) -> String;        // name + description + classification + tags + semantic_content
    pub fn input_schema(&self) -> Value;           // JSON Schema object derived from `inputs`
    pub fn output_schema(&self) -> Value;          // JSON Schema object derived from `outputs`
    pub fn compute_content_hash(&self) -> String;  // sha256 hex over the canonical JSON of the identity fields
    pub fn fingerprint(&self) -> CapabilityFingerprint;
}
impl openjiuwen_common::card::Card for CapabilityDescriptor; // id=capability_id, name, description
// derives: Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema, serde(rename_all="snake_case", default)

pub struct SemanticProfile {
    pub summary: String, pub capabilities: Vec<String>, pub use_cases: Vec<String>,
    pub limitations: Vec<String>, pub keywords: Vec<String>,
}
pub struct CapabilityFingerprint {
    #[serde(flatten)] pub descriptor: CapabilityDescriptor,
    pub version: String,                            // "1.0.0"
    pub static_data: BTreeMap<String, Value>,
    pub semantic_profile: SemanticProfile,
    pub failures: Vec<String>, pub evidence: Vec<Value>, pub suggestions: Vec<String>,
}
pub struct FingerprintArtifact {
    pub schema_version: String,                     // "1.0"
    pub generated_at: String,                       // RFC 3339
    pub source_snapshot: BTreeMap<String, Value>,
    pub fingerprints: Vec<CapabilityFingerprint>,
}

pub struct DataTypeVocabulary { /* canonical names + aliases */ }
impl DataTypeVocabulary {
    pub fn standard() -> Self;
    pub fn normalize(&self, raw: &str) -> String;   // blank -> "any"; aliases -> canonical
    pub fn canonical_types(&self) -> Vec<&'static str>;
    pub fn can_feed(&self, source: &str, target: &str) -> bool;
    pub fn is_subtype(&self, candidate: &str, parent: &str) -> bool;
}
pub fn normalize_type(raw: &str) -> String;         // standard vocabulary shorthand

pub fn normalize_name_key(name: &str) -> String;    // lowercase, non-alphanumerics -> '_', collapsed, trimmed
pub fn fuzzy_name_distance(a: &str, b: &str) -> usize;   // Levenshtein over normalized keys
pub fn fuzzy_name_similarity(a: &str, b: &str) -> f64;   // 1 - distance/max_len (1.0 for two empty keys)
```

Canonical data types: `any`, `string`, `text`, `number`, `integer`, `boolean`, `object`, `array`, `null`, `file`, `image`, `audio`.
Aliases: `str`→`string`, `int`→`integer`, `float`/`double`→`number`, `bool`→`boolean`, `dict`/`map`/`record`/`json`→`object`, `list`/`tuple`→`array`, `bytes`→`bytes` (canonical), `text`→`text`.
`can_feed` is true when the normalized types are equal, the target is `any`, or the source is a subtype of the target (`integer`→`number`, `text`→`string`); it is false when the source is `any` and the target is not.

### `store` (`src/store.rs`)

```rust
#[async_trait]
pub trait AssetStore: Send + Sync {
    async fn get(&self, key: &CapabilityKey) -> Result<Option<CapabilityDescriptor>>;
    async fn put(&self, descriptor: &CapabilityDescriptor) -> Result<()>;
    async fn remove(&self, key: &CapabilityKey) -> Result<bool>;
    async fn list(&self) -> Result<Vec<CapabilityDescriptor>>;   // sorted by qualified id
    async fn clear(&self) -> Result<()>;
}

pub struct KvAssetStore { /* Arc<dyn KvStore> + key prefix */ }
impl KvAssetStore {
    pub fn new(kv: Arc<dyn KvStore>) -> Self;                          // prefix "capability/"
    pub fn with_prefix(kv: Arc<dyn KvStore>, prefix: impl Into<String>) -> Self;
    pub fn in_memory() -> Self;                                        // foundation InMemoryKvStore
    pub async fn json_file(root: impl AsRef<Path>) -> Result<Self>;    // foundation LocalFileKvStore
    #[cfg(feature = "sqlite")]
    pub async fn sqlite(url: &str) -> Result<Self>;                    // foundation SqliteKvStore::connect
}
```

KV keys are `<prefix><qualified_id>`; values are the descriptor's JSON. `get`/`list` skip (and `tracing::warn!`) malformed entries instead of failing the scan.

### `registry` (`src/registry.rs`)

```rust
pub struct CapabilityRegistry { /* std::sync::RwLock<BTreeMap<CapabilityKey, CapabilityDescriptor>>, Option<Arc<dyn AssetStore>>, std::sync::atomic::AtomicU64 */ }
impl CapabilityRegistry {
    pub fn new() -> Self;
    pub fn with_store(store: Arc<dyn AssetStore>) -> Self;          // write-through; call `reload` to hydrate
    pub async fn load(store: Arc<dyn AssetStore>) -> Result<Self>;   // hydrate from the store; revision starts at 0
    pub async fn reload(&self) -> Result<usize>;                     // replace the in-memory map from the store
    pub fn store(&self) -> Option<&Arc<dyn AssetStore>>;
    pub fn revision(&self) -> u64;                                  // monotonic catalog revision, SeqCst, starts at 0
    pub fn len(&self) -> usize;   pub fn is_empty(&self) -> bool;
    pub async fn register(&self, descriptor: CapabilityDescriptor) -> Result<()>;  // validate + upsert + persist
    pub async fn register_all(&self, descriptors: Vec<CapabilityDescriptor>) -> Result<usize>;
    pub async fn get(&self, key: &CapabilityKey) -> Option<CapabilityDescriptor>;
    pub async fn contains(&self, key: &CapabilityKey) -> bool;
    pub async fn remove(&self, key: &CapabilityKey) -> Result<bool>;
    pub async fn clear(&self) -> Result<()>;
    pub async fn list(&self) -> Vec<CapabilityDescriptor>;           // sorted by qualified id
    pub async fn search(&self, request: &SearchRequest) -> Result<Vec<SearchHit>>;
}
impl Default for CapabilityRegistry;
```

The registry is authoritative in memory; writes are write-through to the optional store.
The `std::sync::RwLock` is only held for map mutations and snapshot clones, never across an `.await`.
`revision()` is an `AtomicU64` read/written outside that lock with `SeqCst`; it advances on `register` (+1), a non-empty `register_all` batch (+1 per batch), a `remove` that actually dropped a key, a `clear` of a non-empty catalog, and a store-backed `reload` — never on a no-op mutation or a failed one.
Freshness rule: an observer (the runtime's semantic index) re-derives its view only when `revision()` differs from the value it recorded, so an unchanged catalog costs no re-embedding.

### `search` (`src/search/`)

```rust
pub struct SearchRequest {
    pub text: Option<String>, pub capability_types: Vec<String>, pub tags: Vec<String>,
    pub exclude_tags: Vec<String>, pub available: Option<bool>, pub metadata_filter: Option<Value>,
    pub top_k: usize,               // default 10; 0 means unlimited
    pub min_score: Option<f32>,
}
impl Default for SearchRequest; impl SearchRequest { pub fn for_text(text: impl Into<String>) -> Self; }

pub struct SearchHit { pub descriptor: CapabilityDescriptor, pub score: f32, pub matched_terms: Vec<String> }

pub struct LexicalIndex { /* BM25 over retrieval_text */ }
impl LexicalIndex {
    pub fn build(descriptors: impl IntoIterator<Item = CapabilityDescriptor>) -> Self;
    pub fn search(&self, query: &str, top_k: usize) -> Vec<LexicalHit>;   // 0 = unlimited
    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
}
pub struct LexicalHit { pub qualified_id: String, pub score: f32, pub matched_terms: Vec<String> }
pub fn tokenize(text: &str) -> Vec<String>;

pub fn search_descriptors(descriptors: &[CapabilityDescriptor], request: &SearchRequest) -> Result<Vec<SearchHit>>; // filter -> rank -> min_score -> top_k; `CapabilityRegistry::search` delegates here after dropping its lock
pub const DEFAULT_TOP_K: usize;                     // 10
pub const CAPABILITY_TYPE_FIELD: &str;              // "capability_type" (semantic metadata key)
pub const CAPABILITY_ID_FIELD: &str;                // "capability_id"
pub const DESCRIPTOR_FIELD: &str;                   // "descriptor" (full descriptor JSON)

pub struct SemanticIndex { /* Arc<dyn Embedding> + InMemoryVectorStore */ }
impl SemanticIndex {
    pub fn new(embedding: Arc<dyn Embedding>) -> Self;
    pub async fn index(&self, descriptors: &[CapabilityDescriptor]) -> Result<usize>;
    pub async fn upsert(&self, descriptor: &CapabilityDescriptor) -> Result<()>;
    pub async fn remove(&self, descriptor: &CapabilityDescriptor) -> Result<bool>;
    pub async fn clear(&self) -> Result<()>;
    pub async fn search(&self, query: &str, top_k: usize) -> Result<Vec<SearchHit>>;
}
```

`LexicalIndex` is rebuilt from a snapshot for each registry search (catalog-sized data; no index-invalidations bugs), BM25 with `k1 = 1.2`, `b = 0.75`, and a deterministic tie-break by qualified id.
Filter-only requests (no text) return score `0.0` sorted by qualified id.
`metadata_filter` is a Mongo-style JSON predicate evaluated by `openjiuwen_foundation::store::filter::matches`.
`SemanticIndex` stores one record per descriptor (`id = qualified id`, `text = retrieval_text()`, metadata carries `capability_type`, `capability_id`, and the descriptor JSON) and returns hits with the reconstructed descriptor.

### `graph` (`src/graph.rs`)

```rust
pub const CAN_FEED: &str = "can_feed";

pub struct CapabilityNode { pub id: String, pub label: String, pub node_type: String, pub metadata: BTreeMap<String, Value> }
pub struct CapabilityEdge { pub source: String, pub target: String, pub relation: String, pub weight: f64, pub metadata: BTreeMap<String, Value> }
pub struct CapabilityGraph {
    pub graph_id: String, pub graph_type: String /* "capability_graph" */, pub directed: bool,
    pub nodes: BTreeMap<String, CapabilityNode>, pub edges: Vec<CapabilityEdge>,
    pub metadata: BTreeMap<String, Value>,
}
impl CapabilityGraph {
    pub fn new(graph_id: impl Into<String>) -> Self;
    pub fn from_descriptors(graph_id: impl Into<String>, capabilities: &[CapabilityDescriptor]) -> Self;
    pub fn node(&self, id: &str) -> Option<&CapabilityNode>;
    pub fn has_node(&self, id: &str) -> bool;
    pub fn upsert_node(&mut self, node: CapabilityNode);
    pub fn remove_node(&mut self, id: &str) -> bool;                 // also drops incident edges
    pub fn add_edge(&mut self, edge: CapabilityEdge) -> Result<()>;  // validates + upserts by (source,target,relation)
    pub fn remove_edge(&mut self, source: &str, target: &str) -> bool;
    pub fn remove_incident_edges(&mut self, id: &str) -> usize;
    pub fn outgoing(&self, id: &str) -> Vec<&CapabilityEdge>;
    pub fn incoming(&self, id: &str) -> Vec<&CapabilityEdge>;
    pub fn roots(&self) -> Vec<String>;
    pub fn leaves(&self) -> Vec<String>;
    pub fn node_count(&self) -> usize;
    pub fn edge_count(&self) -> usize;
    pub fn to_value(&self) -> Value;                                 // {"graph": {...}} JGF v2
    pub fn from_value(value: &Value) -> Result<Self>;
}
```

`add_edge` rejects a missing endpoint, a self-loop, a blank relation, a non-finite weight, or a weight outside `[0, 1]` with `Error::Validation`.
Edge ordering is deterministic: `(source, target, relation)` sorted on read, insertion order preserved in `edges` with upsert-in-place.

### `artifact` (`src/artifact.rs`)

```rust
pub struct GraphRevision { pub version: String, pub graph_id: String, pub created_at: String, pub node_count: usize, pub edge_count: usize }
pub struct GraphArtifactStatus {
    pub status: String,                 // "missing" | "ready"
    pub current_version: Option<String>, pub graph_id: Option<String>,
    pub node_count: usize, pub edge_count: usize, pub updated_at: Option<String>,
    pub snapshot: BTreeMap<String, Value>,   // {"graph_id", "current_version"} when ready
}
#[async_trait]
pub trait GraphArtifactStore: Send + Sync {
    async fn publish(&self, graph: &CapabilityGraph, version: Option<&str>) -> Result<GraphRevision>;
    async fn read(&self, version: Option<&str>) -> Result<Option<CapabilityGraph>>;   // None = current
    async fn status(&self) -> Result<GraphArtifactStatus>;
    async fn versions(&self) -> Result<Vec<GraphRevision>>;
}
pub struct MemoryGraphArtifactStore;    // no persistent state; ::new(), Default
pub struct FileGraphArtifactStore;      // ::new(root) loads existing versions
impl FileGraphArtifactStore { pub async fn new(root: impl AsRef<Path>) -> Result<Self>; pub fn root(&self) -> &Path; }

pub const STATUS_MISSING: &str = "missing";   // GraphArtifactStatus::status values
pub const STATUS_READY: &str = "ready";
impl GraphArtifactStatus {
    pub fn missing() -> Self;   // Default: status "missing", empty snapshot
    pub fn ready(revision: &GraphRevision) -> Self;
    pub fn is_ready(&self) -> bool;
}
```

Revisions are immutable: publishing an existing version fails with `Error::Validation`.
A `None` version publishes `v{n+1}` where `n` is the number of existing revisions.
Explicit versions must match `[A-Za-z0-9._-]+` and must not be `.`/`..`.
The file backend writes `root/versions/<version>/graph.json` and `root/current.json`, each via a `<name>.tmp` write followed by `rename`, so readers never observe a partial file.

### `orchestration` (`src/orchestration/`)

Relation candidates, bounded planning, plan execution over `openjiuwen-graph`, and the graph-mutation request contract, all deterministic and offline.

```rust
#[serde(rename_all = "snake_case")] pub enum PlanStatus { Ready, NeedsInput, NoPlan }
#[serde(rename_all = "snake_case")] pub enum MatchStrategy { ExactIo, SemanticOverlap, TextualCoercion }
#[serde(rename_all = "snake_case")] pub enum PlanMode { Fast, Beam }          // default Fast

pub struct RelationCandidate { pub source: String, pub target: String, pub relation: String /* can_feed */,
                               pub confidence: f64, pub strategy: MatchStrategy,
                               pub matched_input: Option<String>, pub matched_output: Option<String> }
pub struct CandidateConfig { pub min_confidence: f64 /*0.7*/, pub min_token_overlap: f64 /*0.6*/,
                             pub max_candidates_per_source: usize /*32*/ }
pub struct CandidateGenerator;
impl CandidateGenerator {
    pub fn new(config: CandidateConfig) -> Self;
    pub fn generate(&self, capabilities: &[CapabilityDescriptor]) -> Vec<RelationCandidate>;
    pub fn edges(&self, candidates: &[RelationCandidate]) -> Vec<CapabilityEdge>;
}

pub struct PlanStep { pub step_id: String /* qualified id */, pub capability_type: String, pub name: String,
                      pub inputs: Vec<CapabilityIo>, pub outputs: Vec<CapabilityIo>,
                      pub missing_inputs: Vec<String>, pub filled_inputs: Vec<String>, pub score: f64 }
pub struct OrchestrationPlan { pub plan_id: String, pub query: String, pub language: String /* "cn" | "en" */,
                               pub status: PlanStatus, pub title: String, pub message: Option<String>,
                               pub steps: Vec<PlanStep>, pub edges: Vec<PlanEdge>, pub score: f64 }
impl OrchestrationPlan { pub fn planned_graph(&self) -> Value; pub fn to_value(&self) -> Value;
                         pub fn from_value(value: &Value) -> Result<Self>;
                         pub fn stages(&self) -> Result<Vec<Vec<String>>>; pub fn is_empty(&self) -> bool; }

pub struct PlanConfig { pub mode: PlanMode, pub top_k: usize /*3*/, pub max_depth: usize /*4*/,
                        pub min_edge_confidence: f64 /*0.7*/, pub max_expansions: usize /*128*/,
                        pub max_plans: usize /*256*/, pub dynamic_graph_enabled: bool /*false*/ }
pub struct PlanRequest { pub query: String, pub candidate_ids: Option<Vec<String>>, pub disabled_ids: Vec<String>,
                         pub provided_inputs: Vec<String>, pub language: String /*"cn"*/,
                         pub top_k: Option<usize>, pub max_depth: Option<usize> }
#[async_trait] pub trait SemanticScorer: Send + Sync {
    async fn score(&self, query: &str, capabilities: &[CapabilityDescriptor]) -> Result<BTreeMap<String, f64>>;
}
pub struct LexicalSemanticScorer;                                             // deterministic default
impl Planner { pub fn new(config: PlanConfig) -> Self;
               pub fn with_scorer(config: PlanConfig, scorer: Arc<dyn SemanticScorer>) -> Self;
               pub async fn plan(&self, request: &PlanRequest, graph: &CapabilityGraph,
                                 capabilities: &[CapabilityDescriptor]) -> Result<OrchestrationPlan>; }
pub fn normalize_language(raw: &str) -> String;                              // zh|cn|zh_cn -> "cn"; en -> "en"
pub fn plan_title(query: &str, language: &str) -> String;                    // "计划：{query}" / "Plan: {query}"

pub struct RunnerContext { pub step_id: String, pub session_id: Option<String>, pub cancellation: CancellationToken }
#[async_trait] pub trait CapabilityRunner: Send + Sync {
    async fn invoke(&self, capability: &CapabilityDescriptor, inputs: Value, context: &RunnerContext) -> Result<Value>;
}
pub struct UnsupportedRunner;                                                // invoke -> Error::Unsupported
pub struct ExecutorConfig { pub max_steps: usize /*32*/, pub node_timeout_ms: Option<u64>, pub fail_fast: bool /*true*/ }
pub struct StepOutcome { pub step_id: String, pub capability_type: String, pub success: bool,
                         pub output: Value, pub error: Option<String>, pub duration_ms: u64 }
pub struct PlanExecution { pub plan_id: String, pub graph_id: String, pub steps: BTreeMap<String, StepOutcome>,
                           pub outputs: BTreeMap<String, Value>, pub state: Value }
impl PlanExecutor {
    pub fn new(runner: Arc<dyn CapabilityRunner>) -> Self;
    pub fn compile(&self, plan: &OrchestrationPlan, capabilities: &[CapabilityDescriptor]) -> Result<CompiledGraph>;
    pub async fn execute(&self, plan: &OrchestrationPlan, capabilities: &[CapabilityDescriptor], session: Option<Arc<dyn Session>>) -> Result<PlanExecution>;
    pub async fn execute_with_inputs(&self, plan: &OrchestrationPlan, capabilities: &[CapabilityDescriptor], inputs: Value, session: Option<Arc<dyn Session>>) -> Result<PlanExecution>;
}

pub struct MutationRequest { pub request_id: String, pub operation: MutationOperation,
                             pub source_revision: Option<String>, pub changed_skill_ids: Vec<String> }
pub fn request_digest(request: &MutationRequest) -> String;
pub fn validate_mutation_against_inventory(request: &MutationRequest, changed: &[String], inventory_ids: &[String]) -> Result<()>;
pub struct JournalRecord { pub digest: String, pub operation: MutationOperation, pub revision: String,
                           pub inserted_at: String, pub result: Option<Value> }
pub struct MutationJournal;   // new/open/path/len/is_empty/contains/request_ids/lookup/record/clear
```

Candidate rules: `ExactIo` (same normalized name and type, confidence `1.0`), `SemanticOverlap` (output/input token Jaccard `>= min_token_overlap`, confidence `0.70 + 0.25 * overlap`), `TextualCoercion` (`can_feed` plus a shared name token, confidence `0.75`); sorted by strategy rank, confidence, source, target; below `min_confidence` dropped; per-source cap.
Plan score: `0.45*mean(semantic) + 0.25*mean(edge_confidence) + 0.20*input_coverage - 0.03*(steps - 1)`, ranked by `(-score, missing_inputs, step_count, step_ids, edge_weights)`; expansion is bounded by `max_depth`/`max_expansions`/`max_plans` (`Fast` greedy, `Beam` `top_k`), and `status` is `Ready`/`NeedsInput`/`NoPlan` from the best plan's root inputs.
Execution maps each step to an `openjiuwen_graph` node writing `state[step_id] = {"success", "output", "error", "duration_ms"}`; `fail_fast` aborts with `Error::Graph`, while `Cancelled`/`Unsupported` keep their own `ErrorKind`.
Mutations: `MutationRequest::prepare` trims/sorts/rejects, the digest is SHA-256 over canonical JSON of `{schema_version, request_id, operation, source_revision, changed_skill_ids}` (declared ids, not payloads), the inventory check requires the applied change set to equal the declared batch, and `MutationJournal` is a `BTreeMap`-ordered JSON document written through atomically (`inserted_at` is metadata; the malformed-envelope case names the file).
See [`orchestration`](src/orchestration/CONTEXT.md) for the full rules, formulas, bounds and journal semantics.

### `evaluation` (`src/evaluation/`)

Fail-soft evaluation over capability cards and recorded traces: static card metrics, trace metrics, event-time case windows, status-aware aggregation, and the LLM judge.

```rust
pub struct EvaluationInput { pub capability: CapabilityDescriptor, pub target: Option<String>,
                             pub trace: Option<EvaluationTrace>, pub window: Option<EvaluationWindow> }
pub struct MetricScore { pub metric: String, pub score: Option<f64>, pub passed: bool,
                         pub detail: Option<String>, pub status: MetricStatus }
#[serde(rename_all = "snake_case")] pub enum MetricStatus { Pass, Fail, NotApplicable, Error, Observed }
#[async_trait] pub trait Metric: Send + Sync {
    fn name(&self) -> &str; fn scope(&self) -> MetricScope; fn cache_key(&self) -> Option<String>;
    async fn evaluate(&self, input: &EvaluationInput) -> Result<MetricScore>;
}
// static: CompletenessMetric, SchemaValidityMetric, StructureConformanceMetric,
//         DescriptionQualityMetric, ClassificationConsistencyMetric, LlmJudgedMetric
// trace:  SuccessRateMetric, LatencyMetric, AccuracyMetric, CapabilitySelectionMetric,
//         CompositionEffectivenessMetric
pub struct QualityResult { pub confidence: QualityConfidence, pub score: Option<f64>, pub aggregation: String,
                           pub per_metric: BTreeMap<String, MetricScore> }
pub fn percentile(values: &[f64], fraction: f64) -> f64;   pub fn latency_statistics(samples_ms: &[f64]) -> LatencyStatistics;
pub fn select_cases(cases: &[EvaluationCase], window: Option<&EvaluationWindow>) -> Vec<&EvaluationCase>;
pub fn aggregate_metric(scores: &[MetricScore]) -> MetricScore;

impl EvaluationSuite {
    pub fn default_suite() -> Self;                                  // completeness + schema validity
    pub fn with_llm_judge(self) -> Self;   pub fn with_llm_judge_model(self, model: Arc<dyn CompletionModel>) -> Self;
    pub async fn evaluate(&self, input: &EvaluationInput) -> EvaluationReport;
    pub async fn evaluate_static(&self, input: &EvaluationInput) -> EvaluationReport;
    pub async fn evaluate_case(&self, input: &EvaluationInput) -> EvaluationReport;
    pub async fn evaluate_quality(&self, input: &EvaluationInput) -> QualityResult;
    pub fn cache_signature(&self) -> Option<String>;   pub fn aggregate(&self, reports: &[EvaluationReport]) -> EvaluationReport;
}
pub const SYMPHONY_EVALUATION_PROTOCOL: &str = "symphony-evaluation-suite-v5";
```

A metric either scores (`score: Some`) or reports a status without a score: latency is always `Observed`, a missing trace or case is `NotApplicable`, a metric failure is `Error` (also recorded in `EvaluationReport::errors`).
`evaluation::LlmJudgedMetric` is a thin delegating wrapper over [`llm::LlmJudgeMetric`](crate::llm::LlmJudgeMetric), so one judge implementation serves both and `with_llm_judge()` scores offline.
See [`evaluation`](src/evaluation/CONTEXT.md) for the per-metric rubrics, check lists and thresholds.

### `runtime` (`src/runtime/`)

The composition root: state machine, graph publication, planning/invocation, lazy semantic sync, evolution, and the mutation path.

```rust
#[serde(rename_all = "snake_case")] pub enum RuntimeState { Created, Running, Stopped }
pub struct SymphonyConfig { pub graph_artifact_root: Option<PathBuf>, pub graph_scope_id: String,
                            pub mutation_journal_path: Option<PathBuf>, pub plan: PlanConfig,
                            pub executor: ExecutorConfig, pub candidates: CandidateConfig }
pub struct SymphonyRuntimeBuilder;   // new/with_registry/with_asset_store/with_artifacts/with_runner/
                                     // with_planner/with_semantic_index/with_mutation_journal/open
pub struct SymphonyRuntime;
impl SymphonyRuntime {
    pub async fn open(config: SymphonyConfig) -> Result<Self>;
    pub fn builder(config: SymphonyConfig) -> SymphonyRuntimeBuilder;
    pub fn state(&self) -> RuntimeState;   pub fn registry(&self) -> &Arc<CapabilityRegistry>;
    pub async fn start(&self) -> Result<()>;   pub async fn stop(&self) -> Result<()>;   pub async fn close(&self) -> Result<()>;
    pub async fn build(&self) -> Result<GraphRevision>;
    pub async fn graph(&self) -> CapabilityGraph;
    pub async fn graph_status(&self) -> Result<GraphArtifactStatus>;
    pub async fn read_graph(&self, version: Option<&str>) -> Result<Option<CapabilityGraph>>;
    pub async fn plan(&self, request: &PlanRequest) -> Result<OrchestrationPlan>;
    pub async fn invoke(&self, request: InvokeRequest) -> Result<InvokeResponse>;
    pub async fn search(&self, request: &SearchRequest) -> Result<Vec<SearchHit>>;
    pub async fn semantic_search(&self, query: &str, top_k: usize) -> Result<Vec<SearchHit>>;
    pub async fn reindex_semantics(&self) -> Result<usize>;
    pub async fn mutate(&self, request: MutationRequest, descriptors: Vec<CapabilityDescriptor>) -> Result<GraphMutationResult>;
    pub async fn add_capabilities(&self, descriptors: Vec<CapabilityDescriptor>, request_id: &str) -> Result<GraphMutationResult>;
    pub async fn update_capabilities(&self, descriptors: Vec<CapabilityDescriptor>, request_id: &str) -> Result<GraphMutationResult>;
    pub async fn remove_capabilities(&self, keys: &[CapabilityKey], request_id: &str) -> Result<GraphMutationResult>;
    pub async fn submit_evolution(&self, execution: &PlanExecution) -> Result<EvolutionResult>;
    pub fn capture_graph_snapshot(&self) -> BTreeMap<String, Value>;
}
pub struct InvokeRequest;   pub struct InvokeResponse { pub plan, pub execution: Option<PlanExecution>, pub graph_status }
#[serde(rename_all = "snake_case")] pub enum MutationOperation { Add, Update, Delete }
#[serde(rename_all = "snake_case")] pub enum MutationStatus { Published, Noop, Replayed }
pub struct GraphMutationResult { operation, request_id, changed, added_edges, removed_edges, revision, replayed,
                                 status, digest, source_revision, diagnostics }   // last four serde-defaulted
pub struct EvolutionResult { pub accepted_edges: usize, pub rejected_edges: usize, pub revision: String }
```

Lifecycle: `Created` → `start` → `Running` → `stop`/`close` → `Stopped`; `start`/`stop` are idempotent, `close` is `stop` plus artifact flush, and `start` restarts a stopped runtime.
`start` publishes a graph when none exists; `build` re-derives it from the registry plus candidate edges.
The mutation order is replay lookup → digest conflict → `Running` gate → change-set/CAS → apply → publish → record; a replay with a different digest or a stale `source_revision` is `Error::Validation` naming that field, and the ledger is in-memory unless a journal is configured.
`submit_evolution` applies successful transitions as an EMA on edge weights (`w' = (w + 1.0) / 2`); `semantic_search` re-embeds only when `CapabilityRegistry::revision()` changed, and without a semantic index it is `Error::Unsupported`.
See [`runtime`](src/runtime/CONTEXT.md) for the full surface and rules.

### `discovery` (`src/discovery/`)

SKILL.md discovery plus the lexical skill catalog: both Python scan modes (immediate-child and bounded recursive), per-skill diagnostics instead of scan failure, descriptor derivation with SHA-256 content hashes, an order-independent inventory fingerprint, BM25 search with excerpts, and the model-facing prompt surface.
`SkillFs::{visible_skill_names, prompt_snapshot(&PromptSnapshotOptions), notice_state}` render the prompt snapshot (`PromptSnapshot { rendered, included, omitted, truncated, fingerprint }`) and produce a persistable `SkillNoticeState` whose `consume(&SkillInventory)` reports added/removed/changed skills plus an incremental reminder.
See [`discovery`](src/discovery/CONTEXT.md) for the full surface, rendering rules and notice semantics.

### `experience` (`src/experience/`)

The experience layer: query patterns bound to skill sets, deterministic dedupe/merge, embedding-optional ranking (cosine or lexical term coverage), and memory/JSONL/SQLite stores with atomic writes.
`ExperienceIndex` is the persisted embedding catalog: one vector per item, Python's `index_dir` layout (`meta.json`, `scalar/metadata.jsonl`, and a JSON vector matrix instead of faiss/numpy files), digest-verified loads, and exact flat cosine search; `ivf*` algorithm names stay `Error::Unsupported` (faiss training).
The input half is implemented too: `TraceStore` ingests `sessions/<id>/{metadata.json,history.json}` into `TraceRecord`s with an idempotent `trace_store/{records.jsonl,processed_index.json}` cache, `cluster_traces` groups them (skill set, then token Jaccard), and `ExperienceCollector` feeds `ExperienceBank` or `ExperienceIndex` (5-example cap, 0.9 similarity merge, 20-item auto-flush).
See [`experience`](src/experience/CONTEXT.md) for the full surface and rules.

### `flow` (`src/flow/`)

Trace ingestion and normalization, structural distillation into versioned `ExperienceRecipe`s, content-addressed `SkillPack` packaging with a ten-check review (`structure`, `dependencies`, `code_safety`, `secrets`, `artifact_paths`, `integrity`, `meta_naming`, `materials`, `permissions`, `license`), `SwarmFlow` script generation, package rendering, and an eager atomic flow store.
Every model-written narrative and trace-derived free-text field is redacted before it enters a recipe or package.

```rust
pub const SCRIPT_FILENAME: &str = "swarmflow/run.py";   pub const SCRIPT_MATERIAL_KEY: &str = "swarmflow_script";
pub struct SwarmflowScript { pub identifier: String, pub source: String, pub meta: Value, pub task: String, pub steps: Vec<String> }
pub fn swarmflow_identifier(materials: &Value, recipe_id: &str) -> String;   // "combo-<12 hex>"
pub fn generate_swarmflow_script(recipe: &ExperienceRecipe) -> Result<SwarmflowScript>;
pub fn python_string_literal(value: &str) -> String;   pub fn validate_generated_script(source: &str) -> Result<()>;

pub struct RenderedArtifact { pub target: String, pub path: String, pub content: String }
pub trait SkillAdapter: Send + Sync { fn target(&self) -> &str; fn render(&self, pack: &SkillPack) -> Result<RenderedArtifact>; }
pub struct SkillPackAdapter;   // new, validate_members
pub fn render_package(pack: &SkillPack, target_kind: &str) -> Result<RenderedArtifact>;   // "" = "skill"; else Unsupported
```

A recipe-derived package carries `swarmflow_script` and `trigger_conditions` in its materials and renders them as `swarmflow/run.py`, so both participate in the package id; the generated script must pass the review's `code_safety` check.
`validate_generated_script` is a documented line-oriented approximation of Python's AST whitelist; `SkillPackAdapter::validate_members` checks member presence, uniqueness and a linear `can_feed` chain (the `active` + `verified` half is enforced at pack time).
See [`flow`](src/flow/CONTEXT.md) for the script shape, rendering rules and review checks.

### `redaction` (`src/redaction/`)

Credential/PII redaction for model-facing and packaged text, mirroring Python's `flow/privacy.py` and `models/_redaction.py`.

```rust
pub const REDACTED: &str = "<redacted>";
pub const REDACTED_CREDENTIAL: &str = "<redacted-credential>";
pub const REDACTED_REQUEST: &str = "<redacted-request>";
pub const DISTILLED_TEXT_LIMIT: usize = 2_000;

pub fn redact_sensitive_text(text: &str) -> String;
pub fn redact_common_credentials(text: &str) -> String;
pub fn sanitize_distilled_text(text: &str, verbatim_query: Option<&str>) -> String;
pub fn normalize_sensitive_name(name: &str) -> String;
pub fn is_sensitive_name(name: &str) -> bool;
pub fn redact_sensitive_json(value: &Value) -> Value;
pub fn redact_private_json(value: &Value, private_keys: &[&str]) -> Value;
```

The scanner is hand rolled (Python's patterns use lookbehind, which `regex` cannot express) and covers PEM private-key blocks, `Bearer` tokens, JWTs, `sk-…` keys, AWS access key ids, and `key=value` / `"key": "value"` assignments whose key `is_sensitive_name` recognizes.
Every pass is idempotent, operates on char boundaries, and replaces only the secret (surrounding text and line structure survive).
`redact_sensitive_json`/`redact_private_json` recurse with a depth cap of 64 and preserve object key order.
`llm::identity::sanitize_metadata` reuses `is_sensitive_name` for the match-cache metadata.

### `llm` (`src/llm/`)

The single `CompletionModel` seam and its five stages (tree retrieval, ontology matching, semantic scoring, fingerprint extraction, LLM judging), each with a documented deterministic fallback under `FallbackMode::Deterministic`.
`identity.rs` adds model/metadata identity digests (`ModelIdentity`, `stable_metadata_sha256`, `endpoint_sha256`, `graph_identity_sha256`, `sanitize_metadata`), and `cache.rs` the persistent relation-match cache (`MatchCache`, `CachedMatch`, `cache_key`) that `OntologyMatcher` uses opt-in via `with_cache` + `MatchOptions { force }`, keyed by candidate, both fingerprint digests and `matcher_signature()`.
See [`llm`](src/llm/CONTEXT.md) for the full surface, the fallback table, and the stage rules.

### `object_artifact` (`src/object_artifact.rs`)

`ObjectGraphArtifactStore` implements `GraphArtifactStore` over an `ObjectStore`, mirroring the filesystem key layout (`<prefix>/versions/<version>/graph.json`, `<prefix>/current.json`) and the same `{"revision", "graph"}` envelope.
Revisions are written before the `current` pointer, publications serialize through an async mutex, and generated versions skip pinned ids.
`ArtifactTransport` exposes `ObjectStore`, `S3(S3Config)`, `Gcs(GcsConfig)` and `AzureBlob(AzureBlobConfig)`, resolved by `ObjectGraphArtifactStore::{s3, gcs, azure_blob, open}`; a blank bucket is `Error::Validation`.
`GcsObjectStore` speaks the GCS JSON API (`storage/v1` metadata/list, `upload/storage/v1` media upload, `alt=media` download) with an OAuth bearer token or anonymously; `AzureBlobStore` speaks Blob REST (`x-ms-version 2021-12-02`) with SharedKey HMAC-SHA256 signing, a SAS token, or anonymously.
`GcsObjectStore::from_service_account_file` and `AzureBlobStore::from_client_credentials` are typed `Error::Unsupported` (RS256 JWT signing / Azure AD token exchange are out of scope).
See the module docs for the constructors, signing rules and environment variables.

### `deferred` (`src/deferred.rs`)

One API-shaped stub remains, and every entry point returns [`Error::Unsupported`] via [`error::unsupported`]: the `transformers`/vLLM logit-selection backend (Python `retrieval`), whose constructor and `select` both report `vLLM logit selection`.
The stub documents the Python class it mirrors, and its data types stay ordinary `serde` types so a caller can build a request without depending on the deferred machinery.
The other typed deferrals live with their implementations: `ExperienceIndex`'s `ivf*` algorithms, `GcsObjectStore::from_service_account_file`, and `AzureBlobStore::from_client_credentials`.

## Notes for Agents

- Intra-doc links in a module-level (`//!`) doc block do not resolve names defined or re-exported in that same file module; write them as labeled links to absolute paths, e.g. `[`AssetStore`](crate::store::AssetStore)`.
- Items in sibling modules and dependency types are linkable the same way (`crate::model::CapabilityDescriptor`, `openjiuwen_foundation::store::KvStore`); std generics used in prose and private items stay plain code spans.
- `cargo doc -p openjiuwen-symphony --no-deps` is the definition of done for doc links: it must print no warnings.

## Constraints

- Cargo: `edition 2021`, `rust-version 1.85`, `publish = false`, `#![forbid(unsafe_code)]`; `default = []` and the sole feature `sqlite` only enables the SQLite backends (`store::KvAssetStore::sqlite` and `experience::SqliteExperienceStore`) via `openjiuwen-foundation/sqlite`.
- Cargo dependencies additionally include `serde_yaml_ng` (front-matter parsing and rendered YAML manifests), `bytes` (object-store artifact payloads) and `base64` (Azure SharedKey signatures); all three are workspace dependencies, and `wiremock` is the dev-dependency behind the GCS/Azure emulator suites.
- Cloud transports use `openjiuwen_common::http` (never raw `reqwest`); no `hmac` and no XML crate exists offline, so SharedKey HMAC-SHA256 and the Azure list XML scanner are hand rolled and unit-tested (RFC 4231 vectors).
- The umbrella crate `openjiuwen` re-exports this crate as `openjiuwen::symphony`; `crate_name()` returns `"openjiuwen-symphony"`.
- Only files under `crates/openjiuwen-symphony/` may change; the root manifest, sibling crates, and `docs/` are read-only from this node.
- Depend downward only: `openjiuwen-common`, `openjiuwen-foundation`, `openjiuwen-graph`, `openjiuwen-retrieval`, `openjiuwen-session`, and the `openjiuwen-core` umbrella.
- No `unsafe`; no `println!`/`eprintln!` (use `tracing`); no panics on user input (no `unwrap`/`expect` outside tests).
- Every public item has a doc comment; files stay ≲1000 lines and split by responsibility (`src/runtime/mod.rs` is the documented composition-root exception; `src/flow/pack.rs` sits just over the soft cap with the script plumbing kept in `codegen.rs`).
- All public data types are `serde`-serializable with `snake_case` field names, tolerant unknown fields, and empty collections as defaults.
- Async-first: tokio; `std::sync::RwLock` is never held across `.await` (snapshot, drop, then await); blocking filesystem work uses tokio's `fs`.
- Errors use `openjiuwen_common::error::Error` only; validation errors name the offending field.
- Determinism: no wall-clock ordering, no hash-map iteration order in outputs; ties break on qualified id (or `worker_id`/record id in the catalog and experience layers); timestamps are metadata only.

## Design Decisions

- **A real registry exists in Rust.** Python inverts discovery to the host (`CapabilityProvider` protocol) and has no registry type; `CapabilityRegistry` is the host-side provider plus a write-through [`AssetStore`].
- **`CapabilityDescriptor::available` defaults to `true`**, and the planner's start set filters on it, so a capability is plannable unless explicitly withdrawn.
- **`top_k == 0` means unlimited** in search requests and `SemanticIndex::search`, a Rust-only sentinel; Python clamps every retrieval `top_k` to `max(1, ...)`.
- **Two retrieval paths over one catalog.** BM25 keyword search stays the deterministic default; `llm::TreeRetriever` implements Python's LLM-guided progressive tree walk and falls back to the same lexical ranking.
- **The LLM stages share one seam and one fallback switch.** `openjiuwen_retrieval::CompletionModel` keeps the stages model-agnostic; `FallbackMode::Deterministic` guarantees a documented offline answer and `FallbackMode::Error` surfaces model failures.
- **SKILL.md discovery is a scan, not a plugin protocol.** Per-skill problems become diagnostics, and discovered skills become ordinary descriptors that flow into the registry, graph, and planner.
- **The skill catalog reuses the crate's own retrieval.** `SkillFs` builds a `LexicalIndex` over discovered descriptors with the markdown body as `semantic_content`.
- **An inventory fingerprint is order-independent.** The SHA-256 is taken over canonical JSON of records sorted by `(worker_id, skill_file)`.
- **Experience ranking is embedding-optional.** The bank persists items, not vectors; an `Embedding` re-derives them at search time, and without one a deterministic lexical score keeps the bank usable offline.
- **Experience dedupe is a Rust-only improvement.** Python keys on `id`; here the normalized pattern plus sorted skill set is the merge key, so repeated observations reinforce one record.
- **Flow distillation is structural, not narrative.** Evidence groups on its cycle-broken successful-edge signature, versions are content-addressed and immutable, and a model can only write prose.
- **A SkillPack is a frozen, reviewable artifact.** Only an `active` + `verified` linear chain can be packed; the package id hashes canonical materials; a static ten-check review decides the verdict without a model or network.
- **Redaction happens where text enters an artifact**, not at render time — one sanitize call per path, so no double redaction and no leak through a copy; the review's `secrets` check re-runs the redactor as the last line of defence.
- **Evaluation separates scored from scoreless verdicts.** `MetricScore::score` is optional and `status` is explicit, so latency stays an `Observed` measurement and a missing case a `NotApplicable` rather than a zero.
- **Object storage is the third artifact backend, not a new abstraction**: `ObjectGraphArtifactStore` reuses the revision envelope and key layout of `FileGraphArtifactStore`, so memory, local, S3, GCS and Azure Blob stores are interchangeable.
- **GCS and Azure Blob are Rust-only transports.** Python's `shared/storage.py` is a boto3 helper for `s3`/`obs` only; the crate adds the GCS JSON API and Blob REST over `openjiuwen_common::http`, hand-rolling HMAC-SHA256 and the XML scanner because neither crate exists offline.
- **File and artifact writes are atomic everywhere**: flow stores, experience documents and indexes, and graph artifacts use a temp sibling plus rename; an object store writes the revision before the pointer.
- **Timestamps never decide behavior.** `reviewed_at`, `created_at`, and `last_hit_at` are metadata; ordering always comes from sorted ids or content hashes.
- **Semantic search uses embeddings** via `openjiuwen_retrieval::{Embedding, InMemoryVectorStore}` (with `HashEmbedding` for deterministic tests), additively over the shared `SearchHit` type.
- **The semantic index is catalog-revision-driven.** `CapabilityRegistry::revision()` lets `semantic_search` re-embed only after a mutation; `reindex_semantics` covers changes the counter cannot see.
- **`SemanticScorer` has two implementations.** `LexicalSemanticScorer` is the deterministic default, and `llm::LlmSemanticScorer` adds model-judged scores with the lexical scorer as its fallback.
- **Planning is a bounded graph search**, using the same score formula and rank tuple as Python with explicit `max_expansions`/`max_plans` bounds.
- **The graph engine facade maps onto `openjiuwen-graph` for execution.** `CapabilityGraph`/`GraphArtifactStore` keep the JGF document semantics; `PlanExecutor` compiles a plan into a real `openjiuwen_graph::CompiledGraph`.
- **Revisions are immutable and atomically published**, and object-store publications serialize through an async mutex; one store is the writer of its root or prefix.
- **Evolution feedback is an EMA on observed edge weight**, reducing Python's append-only evidence and merged revisions to one observable weight update.
- **The mutation digest covers the declared batch, not the payloads** (Python's field set exactly), so an id batch replays even if descriptor content changed; a conflicting digest for a used id is rejected, not answered.
- **The mutation journal is opt-in and separate from the artifact store.** In-memory stays the default; a configured journal survives rebuilds and writes one id-sorted record per request through atomically, where Python embeds the journal in the graph artifact store.
- **SwarmFlow generation is content-addressed and validator-gated.** The identifier is `combo-<sha256(materials)[:12]>`, the script enters the pack's materials so the package id moves with it, and the existing `code_safety` review check re-validates the generated source before a verdict.
- **`FlowEngine::close` is a no-op**: every ingestion, recipe version, and package is written eagerly and atomically.
- **A generated version bumps past pinned ids**, so an explicit `v2` never makes a later generated publish fail; an illegal version is a validation error on read as well as write.
- **One judge implementation.** `evaluation::LlmJudgedMetric` is a thin delegating wrapper over `llm::LlmJudgeMetric`, so the prompt, the single repair retry and the deterministic rubric live in one place; `with_llm_judge()` therefore scores offline instead of recording a fail-soft error.
- **Trace ingestion is deterministic, not statistical.** Python uses faiss k-means plus model distillation; the port groups by skill set and token Jaccard, picks a centroid query deterministically, and merges at a documented threshold — no RNG, no clock.
- **The matcher cache is opt-in and signature-keyed.** The key covers the candidate, both fingerprint digests and `matcher_signature()` (prompt/version constants, model identity, threshold); cache failures degrade to a model call, never to stale matches.
- **Static evaluation remains the offline default.** The suite ships deterministic card and trace metrics, and every LLM-backed metric carries a static rubric as its fallback.

## Deferred Scope

Two kinds of gaps remain: typed `Error::Unsupported` entry points (GPU/ML stacks and vendor credential flows) and pure-logic Python areas that are not ported yet (absent, not stubbed).

| Typed deferral | Why deferred |
|---|---|
| `experience/` — faiss IVF index training | Needs faiss' k-means/inverted lists (a native ML stack); `ExperienceIndex` implements the exact flat cosine index instead, so `ivf`/`ivfpq` names return `Error::Unsupported` while every item stays searchable |
| `retrieval/` — `transformers`/`vLLM` logit-based branch selection | GPU inference stack; the LLM tree walk over `CompletionModel` and its lexical fallback are implemented |
| GCS service-account keys (`GcsObjectStore::from_service_account_file`) | Signing an RS256 JWT needs an RSA signer the offline workspace does not ship; OAuth bearer tokens, anonymous access and emulators work |
| Azure AD client credentials (`AzureBlobStore::from_client_credentials`) | An identity-endpoint token exchange plus a token cache is out of scope; SharedKey, SAS and anonymous cover the artifact path |

Pure-logic Python areas not ported:

- `observation/` — graph-evolution input contract, append-only evidence log, qualification/rebind and merged overlays (`runtime::submit_evolution` takes a `PlanExecution` and applies only an EMA).
- `discovery/{directory_toolkit,toolkit}.py` — the model-facing `skill_index` operations (directory view, list/search/read, request validation, pagination cursors); `SkillFs` provides `visible_skill_names`, `prompt_snapshot` and the persistable `SkillNoticeState` cursor.
- `orchestration/graph/artifacts.py`'s artifact-embedded journal and any diffing beyond the request contract — the request digest, durable journal, source-revision CAS and change-set validation are implemented (see [`orchestration::mutation`](src/orchestration/mutation.rs) and `runtime::mutations`); the journal is a separate JSON document, its digests are not chained over the file, and `source_revision` is optional.
- `orchestration/graph/matcher/{protocol,model}.py` — the `matches_by_candidate`/`diagnostics_by_candidate` helpers and the cache's file lock/pending-merge batching; the persistent match cache, identity digests, endpoint hashing and `sanitize_metadata` are implemented.
- `shared/fingerprint/normalization.py` — IO-name vocabulary, alias merging and normalization audit records.
- `retrieval/search/runtime/{lexical,selector,disclosure}.py` and `retrieval/build/io/*` — field-weighted lexical scoring, prebuilt-index loading and prompt disclosure rendering.
- `retrieval/build/**` and `agent/retrieval_toolkit.py` — LLM-driven index construction over the progressive retriever (~9k lines, overlapping `llm::TreeRetriever`).
- `experience/cluster.py`'s faiss k-means and LLM pattern distillation — replaced by deterministic token-Jaccard clustering and the collector's similarity merge.

Implemented with deterministic offline substitutes: S3/OBS, GCS and Azure Blob artifacts; SKILL.md scanning, the `SkillFs` catalog, prompt snapshot and notice cursor; flow distillation, redaction, ten-check review, `SkillPack` packaging, `SwarmFlow` script generation and package rendering; the mutation request contract, digest, durable idempotency journal and source-revision CAS; the experience bank, persisted index, trace ingestion, clustering and collector; evaluation static/trace metrics and windows; LLM matching (with cache), scoring, retrieval, fingerprinting and judging.
Typed deferrals are proven, not just documented: `tests/deferred.rs` asserts the stub shape, and `tests/{artifact_gcs,artifact_azure,experience_index}.rs` assert the remaining GPU/vendor boundaries.

## Error Strategy

- `Error::Validation` — malformed user input: blank ids/names/patterns/trace ids, illegal graph edge, bad version or meta name, draft/unverified/non-chain recipe, blank bucket, unknown artifact transport detail, runtime in the wrong state, a corrupted experience-index file, a malformed match cache.
- `Error::NotFound` — explicit lookups that must exist (packing an unknown recipe); registry, store, and catalog reads return `Option`.
- `Error::Graph` — plan execution failures, wrapping the failing step id and the runner error message.
- `Error::Unsupported` — the remaining GPU/ML and vendor-credential boundaries: `vLLM` logit selection, faiss IVF training, GCS service-account signing, Azure AD client credentials, and execution without a configured runner.
- `Error::Llm` — model call or unusable-reply failures surfaced when `FallbackMode::Error` is configured; the deterministic mode converts them into the stage's documented fallback instead.
- `Error::Validation`/`Error::Retrieval` — argument and backend failures propagated from `openjiuwen-retrieval`; they are never swallowed.
- Artifact/store/flow I/O failures map to `Error::Config` (invalid root or endpoint) or `Error::External` (I/O); malformed individual records are warned and skipped rather than aborting a scan.
- A match-cache read/write failure is warned and degrades to a model call; it never yields stale matches and never hides the model result.

## Test Strategy

- Unit tests live beside modules (`#[cfg(test)] mod tests`); cross-module behavior lives in `crates/openjiuwen-symphony/tests/`.
- `tests/models.rs` — serde round-trips, `Card` impl, schemas, content hash, datatype `can_feed`, naming helpers.
- `tests/registry.rs` — CRUD, validation, ordering, persistence, `load`/`reload`, revision counter.
- `tests/search.rs` — BM25, filters, `top_k`, filter-only ordering, `HashEmbedding` semantic index.
- `tests/graph.rs` — graph from descriptors, edge validation/upsert, incident-edge removal, JGF round-trip.
- `tests/artifact.rs` — memory/file artifact stores, immutability, `current` pointer, atomicity, illegal versions, reopen hydration.
- `tests/planner.rs` — candidate strategies/ordering, plan selection, filters, depth, scoring/ties, statuses, serialization, `stages`.
- `tests/executor.rs` — compile + execute with a fake runner, per-step state keys, input overlay, `max_steps`, `fail_fast=false`, `UnsupportedRunner`.
- `tests/runtime.rs` — lifecycle idempotency, invoke, build revisions, mutation idempotency, `MutationStatus`/`digest`/`source_revision` shape, digest conflict, payload-vs-ids replay, request/payload mismatch, stale vs current `source_revision`, builder journal persistence across a rebuild, default in-memory behaviour, evolution EMA, lazy semantic re-embedding, stopped-runtime errors, snapshot.
- `tests/mutation_journal.rs` — request preparation normalization and rejections, pinned digests (order independence, per-field sensitivity), change-set validation, journal round-trip/reopen, atomic layout, clear, and malformed documents.
- `tests/discovery.rs` — immediate/recursive scans, front-matter diagnostics, content-hash stability, inventory fingerprint, catalog search/excerpts, disabled skills, symlinks, settings.
- `tests/skill_prompt.rs` — visible-name ordering/filtering, pinned snapshot text (plain/category/empty), multi-byte `max_chars` truncation with `included`/`omitted`/`truncated`/fingerprint bookkeeping, and added/removed/changed notices with exact reminder wording, idempotence and serde persistence.
- `tests/experience.rs` — generated/explicit ids, dedupe merging, lexical and cosine ranking, skill filters, JSONL/SQLite persistence, id-counter continuity.
- `tests/experience_index.rs` — add/search/dedupe, `top_k`/threshold ordering, skill-id flattening, `exist`, digest/count/dimension corruption, persist round-trip, reopen id continuity, IVF/unknown-algorithm errors.
- `tests/experience_trace.rs` — section fixtures → exact `TraceRecord`s, idempotent re-parse, malformed-session tolerance, clustering/centroid determinism, collector drop/auto-flush/merge/stats, end-to-end bank search.
- `tests/flow.rs` — trace normalization, ingest rejection/idempotency, grouping/quality, versioning, rehydration, `SkillPack` render/save/load, review verdicts, install preparation, the rendered `swarmflow/run.py` artifact, narrative fallback, and the ten-check review including `code_safety`/`secrets`/`artifact_paths`.
- `tests/flow_codegen.rs` — the pinned two-step `SwarmFlow` script, the empty combination, identifier stability and content sensitivity, quote/newline/CJK escaping, validator accept plus first-bad-line rejects, member-validation failures, the unsupported render target, script participation in the package identity, and a full pack → review → install path carrying `run.py` (with a broken script failing `code_safety`).
- `tests/redaction.rs` — every redaction entry point, idempotence, non-ASCII input, the JSON depth cap and key order, and a flow run whose model emits a key and a PEM block (recipe and pack materials must contain `<redacted>` and no secret).
- `tests/evaluation.rs` — static card metrics, structure conformance, mixed success rates, `Observed` latency, hand-computed percentile interpolation, window boundaries, aggregation precedence and confidence tiers, `cache_signature` stability, the offline/model-backed judge, and serde round-trips.
- `tests/llm.rs` — every stage with a scripted model and with no model: tree walk/fallback, ontology consensus, scoring, fingerprint, judge repair, round-trips, error mode.
- `tests/matcher_cache.rs` — identity digests, endpoint normalization, `sanitize_metadata`, cache round-trip and byte-stable layout, replay with zero model calls, `force` bypass, signature invalidation, malformed cache, degradation.
- `tests/object_artifact.rs` — numbered revisions, `current` pointer/status, reopen persistence, object keys, pinned-version skipping, blank bucket, malformed `current.json`.
- `tests/artifact_gcs.rs` / `tests/artifact_azure.rs` — full publish → read → status → versions round-trips against a stateful wiremock cloud emulator, plus auth-header assertions (bearer / `SharedKey` / SAS / anonymous), pagination, prefix sorting, 404 semantics and endpoint failure mapping.
- `tests/deferred.rs` — the remaining `vLLM` stub and the `ivf*` algorithm deferral return `Error::Unsupported`, and `error::is_unsupported` recognizes them.
- Tests are offline and deterministic: `tempfile` for filesystem stores, `HashEmbedding` for semantic search, `wiremock` for the cloud transports, no live network, no clock-dependent assertions.
- `cargo fmt -p openjiuwen-symphony`, `cargo test -p openjiuwen-symphony`, and `cargo clippy -p openjiuwen-symphony --all-targets --all-features -- -D warnings` plus `--features sqlite` compile checks must pass.

## Routing Table

- `./src/lib.rs` → crate docs, module declarations, `crate_name`
- `./src/error.rs` → error helpers (`unsupported`, `validation`, `is_unsupported`)
- `./src/model/` → capability cards, fingerprints, data-type vocabulary, naming
  - `./src/model/descriptor.rs` → `CapabilityIo`, `CapabilityDescriptor`, schema derivation, content hash
  - `./src/model/fingerprint.rs` → `CapabilityFingerprint`, `SemanticProfile`, `FingerprintArtifact`
  - `./src/model/datatype.rs` → `DataTypeVocabulary`, `can_feed`, `is_subtype`
  - `./src/model/naming.rs` → `normalize_name_key`, fuzzy name distance/similarity
- `./src/store.rs` → `AssetStore`, `KvAssetStore` (memory/JSON/SQLite)
- `./src/registry.rs` → `CapabilityRegistry` CRUD + search
- `./src/search/` → retrieval
  - `./src/search/mod.rs` → `SearchRequest`, `SearchHit`, filtering, BM25 orchestration
  - `./src/search/lexical.rs` → `tokenize`, `LexicalIndex`, BM25 scoring
  - `./src/search/semantic.rs` → `SemanticIndex` over `openjiuwen-retrieval`
- `./src/graph.rs` → `CapabilityGraph`, nodes/edges, JGF v2 conversion
- `./src/artifact.rs` → `GraphArtifactStore`, immutable revisions, `current` pointer
- `./src/object_artifact.rs` → `ObjectGraphArtifactStore` over `ObjectStore` (in-memory/local/S3/GCS/Azure), `ArtifactTransport`, `open`
  - `./src/object_artifact/gcs.rs` → `GcsConfig`, `GcsObjectStore` (GCS JSON API)
  - `./src/object_artifact/azure/` → `AzureBlobConfig`, `AzureBlobAuth`, `AzureBlobStore` (Blob REST + SharedKey signing, SAS, paths, XML)
  - `./src/object_artifact/http.rs` → shared non-2xx error mapping and the hand-rolled HMAC-SHA256
- `./src/discovery/` → SKILL.md discovery, the lexical skill catalog, prompt snapshot, notice cursor; see [`src/discovery/CONTEXT.md`](src/discovery/CONTEXT.md)
- `./src/experience/` → the experience bank, the persisted `ExperienceIndex`, trace ingestion, clustering and collector; see [`src/experience/CONTEXT.md`](src/experience/CONTEXT.md)
- `./src/flow/` → trace distillation, recipes, SkillPacks, review, `SwarmFlow` codegen, package rendering; see [`src/flow/CONTEXT.md`](src/flow/CONTEXT.md)
  - `./src/flow/codegen.rs` → `generate_swarmflow_script`, `swarmflow_identifier`, `python_string_literal`, `validate_generated_script`
  - `./src/flow/render.rs` → `SkillPackAdapter`, `SkillAdapter`, `RenderedArtifact`, `render_package`
- `./src/redaction/` → credential/PII redaction: scanner, sensitive-name matching, JSON redaction
- `./src/llm/` → LLM-guided stages with deterministic fallbacks, model/metadata identity, match cache; see [`src/llm/CONTEXT.md`](src/llm/CONTEXT.md)
- `./src/orchestration/` → planning and execution
  - `./src/orchestration/plan.rs` → `OrchestrationPlan`, `PlanStep`, `PlanStatus`, JGF projection, stages
  - `./src/orchestration/candidates.rs` → `CandidateGenerator`, relation candidates
  - `./src/orchestration/planner.rs` → `Planner`, `PlanConfig`, `SemanticScorer`, language helpers
  - `./src/orchestration/executor.rs` → `PlanExecutor`, `CapabilityRunner`, plan → `openjiuwen-graph`
  - `./src/orchestration/mutation.rs` → `MutationRequest`, `request_digest`, `MutationJournal`, `validate_mutation_against_inventory`; see [`src/orchestration/CONTEXT.md`](src/orchestration/CONTEXT.md)
- `./src/evaluation/` → fail-soft evaluation: shared contract, static and trace metrics, suite aggregation, windows; see [`src/evaluation/CONTEXT.md`](src/evaluation/CONTEXT.md)
- `./src/runtime/` → `SymphonyRuntime`, config, lifecycle, evolution
  - `./src/runtime/mod.rs` → lifecycle, planning/invocation, lazy semantic sync, evolution
  - `./src/runtime/mutations.rs` → the mutation path, `GraphMutationResult`, `MutationStatus`, journal-backed replay and source-revision CAS
- `./src/deferred.rs` → the remaining GPU-only `Error::Unsupported` stub (`vLLM` logit selection)
- `./tests/` → behavior suites: `models`, `registry`, `search`, `graph`, `artifact`, `object_artifact`, `artifact_gcs`, `artifact_azure`, `planner`, `executor`, `runtime`, `mutation_journal`, `discovery`, `skill_prompt`, `experience`, `experience_index`, `experience_trace`, `flow`, `flow_codegen`, `redaction`, `evaluation`, `llm`, `matcher_cache`, `deferred`
