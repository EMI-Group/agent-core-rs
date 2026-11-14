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
    pub fn with_description(self, description: impl Into<String>) -> Self;
    pub fn with_required(self, required: bool) -> Self;
    pub fn with_default(self, value: Value) -> Self;
    pub fn with_metadata(self, key: impl Into<String>, value: Value) -> Self;
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
    pub fn with_description(self, description: impl Into<String>) -> Self;
    pub fn with_source(self, source: impl Into<String>) -> Self;
    pub fn with_available(self, available: bool) -> Self;
    pub fn with_classification(self, classification: impl Into<String>) -> Self;
    pub fn with_tags(self, tags: impl IntoIterator<Item = impl Into<String>>) -> Self;
    pub fn with_inputs(self, inputs: Vec<CapabilityIo>) -> Self;
    pub fn with_outputs(self, outputs: Vec<CapabilityIo>) -> Self;
    pub fn with_semantic_content(self, text: impl Into<String>) -> Self;
    pub fn with_metadata(self, key: impl Into<String>, value: Value) -> Self;
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
    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
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

```rust
#[derive(Serialize, Deserialize)] #[serde(rename_all = "snake_case")]
pub enum PlanStatus { Ready, NeedsInput, NoPlan }
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

#[serde(rename_all = "snake_case")] pub enum MatchStrategy { ExactIo, SemanticOverlap, TextualCoercion }
impl MatchStrategy { pub fn rank(self) -> u8; pub fn as_str(self) -> &'static str; }   // "exact_io" | "semantic_overlap" | "textual_coercion"
pub struct RelationCandidate {
    pub source: String, pub target: String, pub relation: String /* CAN_FEED */,
    pub confidence: f64, pub strategy: MatchStrategy,
    pub matched_input: Option<String>, pub matched_output: Option<String>,
}
pub struct CandidateConfig { pub min_confidence: f64 /*0.7*/, pub min_token_overlap: f64 /*0.6*/, pub max_candidates_per_source: usize /*32*/ }
pub struct CandidateGenerator;
impl CandidateGenerator {
    pub fn new(config: CandidateConfig) -> Self;
    pub fn config(&self) -> &CandidateConfig;
    pub fn generate(&self, capabilities: &[CapabilityDescriptor]) -> Vec<RelationCandidate>;
    pub fn edges(&self, candidates: &[RelationCandidate]) -> Vec<CapabilityEdge>;
}
impl Default for CandidateConfig; impl Default for CandidateGenerator;

#[serde(rename_all = "snake_case")] pub enum PlanMode { Fast, Beam }   // default Fast
pub struct PlanConfig {
    pub mode: PlanMode, pub top_k: usize /*3*/, pub max_depth: usize /*4*/,
    pub min_edge_confidence: f64 /*0.7*/, pub max_expansions: usize /*128*/,
    pub max_plans: usize /*256*/, pub dynamic_graph_enabled: bool /*false*/,
}
impl Default for PlanConfig;
pub struct PlanRequest {
    pub query: String, pub candidate_ids: Option<Vec<String>>, pub disabled_ids: Vec<String>,
    pub provided_inputs: Vec<String>, pub language: String /*"cn"*/,
    pub top_k: Option<usize>, pub max_depth: Option<usize>,
}
impl Default for PlanRequest; impl PlanRequest { pub fn new(query: impl Into<String>) -> Self; }

#[async_trait]
pub trait SemanticScorer: Send + Sync {
    async fn score(&self, query: &str, capabilities: &[CapabilityDescriptor]) -> Result<BTreeMap<String, f64>>;
}
pub struct LexicalSemanticScorer;    // deterministic default: query-term coverage in [0,1]
pub struct Planner;
impl Planner {
    pub fn new(config: PlanConfig) -> Self;
    pub fn with_scorer(config: PlanConfig, scorer: Arc<dyn SemanticScorer>) -> Self;
    pub fn config(&self) -> &PlanConfig;
    pub async fn plan(&self, request: &PlanRequest, graph: &CapabilityGraph, capabilities: &[CapabilityDescriptor]) -> Result<OrchestrationPlan>;
}
pub fn normalize_language(raw: &str) -> String;   // "zh"|"cn"|"zh_cn" -> "cn"; "en" -> "en"; other -> "cn"
pub fn plan_title(query: &str, language: &str) -> String;   // "计划：{query}" / "Plan: {query}"

pub struct RunnerContext { pub step_id: String, pub session_id: Option<String>, pub cancellation: CancellationToken }
#[async_trait]
pub trait CapabilityRunner: Send + Sync {
    async fn invoke(&self, capability: &CapabilityDescriptor, inputs: Value, context: &RunnerContext) -> Result<Value>;
}
pub struct UnsupportedRunner;          // invoke -> Error::Unsupported
pub struct ExecutorConfig { pub max_steps: usize /*32*/, pub node_timeout_ms: Option<u64> /*None*/, pub fail_fast: bool /*true*/ }
pub struct StepOutcome { pub step_id: String, pub capability_type: String, pub success: bool, pub output: Value, pub error: Option<String>, pub duration_ms: u64 }
pub struct PlanExecution { pub plan_id: String, pub graph_id: String, pub steps: BTreeMap<String, StepOutcome>, pub outputs: BTreeMap<String, Value>, pub state: Value }
impl PlanExecution { pub fn failed_steps(&self) -> Vec<&StepOutcome>; pub fn succeeded(&self) -> bool; }
pub struct PlanExecutor;
impl PlanExecutor {
    pub fn new(runner: Arc<dyn CapabilityRunner>) -> Self;
    pub fn with_config(config: ExecutorConfig, runner: Arc<dyn CapabilityRunner>) -> Self;
    pub fn compile(&self, plan: &OrchestrationPlan, capabilities: &[CapabilityDescriptor]) -> Result<CompiledGraph>;
    pub async fn execute(&self, plan: &OrchestrationPlan, capabilities: &[CapabilityDescriptor], session: Option<Arc<dyn Session>>) -> Result<PlanExecution>;
    pub async fn execute_with_inputs(&self, plan: &OrchestrationPlan, capabilities: &[CapabilityDescriptor], inputs: Value, session: Option<Arc<dyn Session>>) -> Result<PlanExecution>;   // seeds state["inputs"]; `execute` delegates with Value::Null
}
impl Default for ExecutorConfig;
```

**Candidate rules** (deterministic, sorted by strategy rank, then confidence desc, then source, then target):
`ExactIo` — an output and an input share the normalized name *and* normalized type: confidence `1.0`.
`SemanticOverlap` — token Jaccard of output text vs input text `>= min_token_overlap`: confidence `0.70 + 0.25 * overlap`.
`TextualCoercion` — `DataTypeVocabulary::can_feed(output.type, input.type)` with at least one shared name token: confidence `0.75`.
Candidates below `min_confidence` are dropped; each source keeps at most `max_candidates_per_source` (highest confidence first).

**Plan scoring** (Python formula, deterministic substitute scorer):
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

**Execution mapping:** each plan step becomes an `openjiuwen_graph` node whose id is the step id; plan edges become graph edges; entries are steps with no incoming edge and exits steps with no outgoing edge.
A step's inputs start from `state["inputs"]` and are overlaid by predecessor step outputs whose normalized name matches an input name (a predecessor with exactly one output also fills a single unmatched input).
Each node writes the top-level state key equal to its step id: `{"success", "output", "error", "duration_ms"}`.
With `fail_fast` the first runner error aborts with `Error::Graph("step `<id>` failed: ...")`; otherwise every step runs and `failed_steps()` reports failures.
`max_steps` is checked before compilation; the default session is `WorkflowSession::new(<uuid>)`; an empty plan executes to an empty `PlanExecution`.
A `Cancelled` or `Unsupported` runner error keeps its own [`ErrorKind`] instead of becoming `Error::Graph`.

### `evaluation` (`src/evaluation/`)

Fail-soft evaluation over capability cards and recorded traces: static card metrics, trace metrics, event-time case windows, status-aware aggregation, and the LLM judge.

```rust
pub struct EvaluationInput { pub capability: CapabilityDescriptor, pub target: Option<String>,
                             pub trace: Option<EvaluationTrace>, pub window: Option<EvaluationWindow> }
pub struct MetricScore { pub metric: String, pub score: Option<f64>, pub passed: bool,
                         pub detail: Option<String>, pub status: MetricStatus }
#[serde(rename_all = "snake_case")] pub enum MetricStatus { Pass, Fail, NotApplicable, Error, Observed }
#[async_trait]
pub trait Metric: Send + Sync {
    fn name(&self) -> &str;
    fn scope(&self) -> MetricScope;                                   // static | trace
    fn cache_key(&self) -> Option<String>;
    async fn evaluate(&self, input: &EvaluationInput) -> Result<MetricScore>;
}
pub struct CompletenessMetric;          pub struct SchemaValidityMetric;
pub struct StructureConformanceMetric;  pub struct DescriptionQualityMetric;
pub struct ClassificationConsistencyMetric;  pub struct LlmJudgedMetric;   // delegates to llm::LlmJudgeMetric
pub struct SuccessRateMetric;           pub struct LatencyMetric;
pub struct AccuracyMetric;              pub struct CapabilitySelectionMetric;
pub struct CompositionEffectivenessMetric;

pub struct CapabilityCall { name, success, duration_ms, event_time, arguments: Value }
pub struct EvaluationCase { case_id, query, event_time: Option<String>, calls,
                            expected_answer: Option<String>, actual_answer: Option<String>, metadata }
pub struct EvaluationTrace { cases: Vec<EvaluationCase> }
pub struct EvaluationWindow { start: Option<String>, end: Option<String> }   // start inclusive, end exclusive
pub struct LatencyStatistics { avg_ms, p50_ms, p95_ms, max_ms, count, samples_ms }
pub struct QualityResult { confidence: QualityConfidence, score: Option<f64>, aggregation, per_metric: BTreeMap<String, MetricScore> }

pub fn percentile(values: &[f64], fraction: f64) -> f64;        // pos = (n-1) * f
pub fn latency_statistics(samples_ms: &[f64]) -> LatencyStatistics;
pub fn select_cases(cases: &[EvaluationCase], window: Option<&EvaluationWindow>) -> Vec<&EvaluationCase>;
pub fn aggregate_metric(scores: &[MetricScore]) -> MetricScore;

impl EvaluationSuite {
    pub fn new(metrics: Vec<Arc<dyn Metric>>) -> Self;
    pub fn default_suite() -> Self;                                  // completeness + schema validity
    pub fn with_llm_judge(self) -> Self;                             // model-free judge, deterministic rubric
    pub fn with_llm_judge_model(self, model: Arc<dyn CompletionModel>) -> Self;
    pub fn metrics(&self) -> &[Arc<dyn Metric>];
    pub async fn evaluate(&self, input: &EvaluationInput) -> EvaluationReport;
    pub async fn evaluate_static(&self, input: &EvaluationInput) -> EvaluationReport;
    pub async fn evaluate_case(&self, input: &EvaluationInput) -> EvaluationReport;
    pub async fn evaluate_quality(&self, input: &EvaluationInput) -> QualityResult;
    pub fn cache_signature(&self) -> Option<String>;                 // SHA-256, SYMPHONY_EVALUATION_PROTOCOL
    pub fn aggregate(&self, reports: &[EvaluationReport]) -> EvaluationReport;
}
pub const SYMPHONY_EVALUATION_PROTOCOL: &str = "symphony-evaluation-suite-v5";
```

A metric either scores (`score: Some`) or reports a status without a score: latency is always `Observed`, a missing trace or case is `NotApplicable`, a metric failure is `Error` (also recorded in `EvaluationReport::errors`).
Aggregation follows Python's precedence (`Error` > `Observed` > `NotApplicable`, and a mean `Pass` only when every scored sample passed); `QualityConfidence` maps sample counts 0 / 1 / 2–9 / ≥10 to none / low / normal / high, and `QualityResult.score` stays `None` with `aggregation = "per_metric"`.
`evaluation::LlmJudgedMetric` is a thin delegating wrapper over [`llm::LlmJudgeMetric`](crate::llm::LlmJudgeMetric): it owns only the `"llm_judged"` name and the default rubric and forwards `with_model`/`with_rubric`/`with_fallback` plus `Metric::evaluate`, so there is one judge implementation and `with_llm_judge()` scores a card offline with the deterministic rubric instead of recording an error.
See [`evaluation`](src/evaluation/CONTEXT.md) for the per-metric rubrics, check lists and thresholds.

### `runtime` (`src/runtime.rs`)

```rust
#[serde(rename_all = "snake_case")] pub enum RuntimeState { Created, Running, Stopped }

pub struct SymphonyConfig {
    pub graph_artifact_root: Option<PathBuf>,   // None = in-memory artifacts
    pub graph_scope_id: String,                 // default "default"
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
    pub async fn add_capabilities(&self, descriptors: Vec<CapabilityDescriptor>, request_id: &str) -> Result<GraphMutationResult>;
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
pub struct GraphMutationResult {
    pub operation: MutationOperation, pub request_id: String, pub changed: Vec<String>,
    pub added_edges: usize, pub removed_edges: usize, pub revision: String, pub replayed: bool,
}
pub struct EvolutionResult { pub accepted_edges: usize, pub rejected_edges: usize, pub revision: String }
```

Lifecycle: `Created` → `start` → `Running` → `stop`/`close` → `Stopped`; both `start` and `stop` are idempotent and `close` is `stop` plus artifact flush, and `start` also restarts a `Stopped` runtime.
`start` builds and publishes the graph when no revision exists, flipping to `Running` only after the build succeeds; `build` re-derives the graph from the registry plus candidate edges and publishes a new revision.
`add/update/remove_capabilities` mutate the registry, re-derive and publish the graph, and are idempotent per `request_id` (a replay returns the recorded result with `replayed = true`).
`invoke` and the three mutations require `Running` (`Error::Validation` naming `runtime.state` otherwise); planning, graph reads, search, build, evolution and the accessors work in every state.
`invoke` plans, optionally executes when `execute` is set, and returns both; a `NoPlan` result with `execute` set returns `execution: None` rather than an error.
A blank `request_id` is `Error::Validation`; `update` requires every key to exist (`Error::NotFound`, whole batch rejected) while `remove` skips absent keys.
`graph()` is infallible: when the artifact read fails it logs a warning and answers with the catalog-derived, node-only graph.
`submit_evolution` applies successful step transitions as an EMA on edge weights (`w' = (w + 1.0) / 2`), publishes only when a weight changed, and reports accepted/rejected edges.
`capture_graph_snapshot` returns `{"current_version", "graph_id", "graph_scope_id"}` with `null` values when no revision exists.
`SymphonyRuntimeBuilder::with_asset_store` hydrates the registry from the store, so a preconfigured asset store survives restarts.
`semantic_search` clears and re-embeds the catalog only when `CapabilityRegistry::revision()` differs from the last synchronization; `reindex_semantics` force-re-embeds and returns the count; both serialize through one `tokio::sync::Mutex`.
Without a semantic index both entry points return `Error::Unsupported`.

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

Trace ingestion and normalization, structural distillation into versioned `ExperienceRecipe`s, content-addressed `SkillPack` packaging with a ten-check review (`structure`, `dependencies`, `code_safety`, `secrets`, `artifact_paths`, `integrity`, `meta_naming`, `materials`, `permissions`, `license`), and an eager atomic flow store.
Every model-written narrative and trace-derived free-text field is redacted before it enters a recipe or package.
See [`flow`](src/flow/CONTEXT.md) for the full surface and rules.

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
- Cargo dependencies additionally include `serde_yaml_ng` (front-matter parsing and rendered YAML manifests) and `bytes` (object-store artifact payloads); both are workspace dependencies.
- The umbrella crate `openjiuwen` re-exports this crate as `openjiuwen::symphony`; `crate_name()` returns `"openjiuwen-symphony"`.
- Only files under `crates/openjiuwen-symphony/` may change; the root manifest, sibling crates, and `docs/` are read-only from this node.
- Depend downward only: `openjiuwen-common`, `openjiuwen-foundation`, `openjiuwen-graph`, `openjiuwen-retrieval`, `openjiuwen-session`, and the `openjiuwen-core` umbrella.
- No `unsafe`; no `println!`/`eprintln!` (use `tracing`); no panics on user input (no `unwrap`/`expect` outside tests).
- Every public item has a doc comment; files stay ≲1000 lines and split by responsibility (`src/runtime.rs` is the documented composition-root exception).
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
- **A SkillPack is a frozen, reviewable artifact.** Only an `active` + `verified` linear chain can be packed; the package id hashes canonical materials; a static seven-check review decides the verdict without a model or network.
- **Object storage is the third artifact backend, not a new abstraction.** `ObjectGraphArtifactStore` reuses the revision envelope and key layout of `FileGraphArtifactStore`, so in-memory, local, and S3-compatible stores are interchangeable.
- **File and artifact writes are atomic everywhere.** Flow stores, experience JSON documents, experience-index artifacts, and graph artifacts write a temp sibling and rename; an object store writes the revision before the pointer.
- **Timestamps never decide behavior.** `reviewed_at`, `created_at`, and `last_hit_at` are metadata; ordering always comes from sorted ids or content hashes.
- **Semantic search uses embeddings** via `openjiuwen_retrieval::{Embedding, InMemoryVectorStore}` (with `HashEmbedding` for deterministic tests), additively over the shared `SearchHit` type.
- **The semantic index is catalog-revision-driven.** `CapabilityRegistry::revision()` lets `semantic_search` re-embed only after a mutation; `reindex_semantics` covers changes the counter cannot see.
- **`SemanticScorer` has two implementations.** `LexicalSemanticScorer` is the deterministic default, and `llm::LlmSemanticScorer` adds model-judged scores with the lexical scorer as its fallback.
- **Planning is a bounded graph search**, using the same score formula and rank tuple as Python with explicit `max_expansions`/`max_plans` bounds.
- **The graph engine facade maps onto `openjiuwen-graph` for execution.** `CapabilityGraph`/`GraphArtifactStore` keep the JGF document semantics; `PlanExecutor` compiles a plan into a real `openjiuwen_graph::CompiledGraph`.
- **Revisions are immutable and atomically published**, and object-store publications serialize through an async mutex; one store is the writer of its root or prefix.
- **Evolution feedback is an EMA on observed edge weight**, reducing Python's append-only evidence and merged revisions to one observable weight update.
- **`FlowEngine::close` is a no-op**: every ingestion, recipe version, and package is written eagerly and atomically.
- **A generated version bumps past pinned ids**, so an explicit `v2` never makes a later generated publish fail; an illegal version is a validation error on read as well as write.
- **Static evaluation remains the offline default.** The suite ships deterministic completeness/schema metrics, and `llm::LlmJudgeMetric` implements Python's LLM-judged metric with a static rubric as its fallback.

## Deferred Scope

GPU/ML-dependent and vendor-transport areas are typed deferrals; each keeps an `Error::Unsupported` entry point.
The following pure-logic Python areas are not ported yet — the capability is absent rather than stubbed:

- `observation/` — graph-evolution input contract, append-only evidence log, qualification/rebind and merged overlays (`runtime::submit_evolution` takes a `PlanExecution` and applies only an EMA).
- `evaluation/{base,evaluators,suite}.py` — `StructureConformance`, `SuccessRate`, `Latency` (p50/p95), `Accuracy`, `Trace`, `CapabilitySelection`, `CompositionEffectiveness` and LLM-judged description/classification metrics, plus case/window selection and metric aggregation.
- `discovery/{directory_toolkit,toolkit}.py` — the model-facing `skill_index` operations (directory view, list/search/read, request validation, pagination cursors, prompt snapshot, incremental notice state).
- `flow/{privacy,codegen,render}.py` — credential/PII redaction, Swarmflow `run.py` generation (packs carry only `SKILL.md` and `dependencies.yaml`), single-skill `SkillAdapter` rendering, and the `code_safety`/`secrets`/`artifact_paths` review checks.
- `orchestration/{mutations,mutation_service,artifacts}.py` — request digests, a persistent mutation journal, source-revision CAS and provider change-set validation (the runtime keeps an in-memory `request_id` ledger instead).
- `experience/{trace,cluster,collector}.py` — session-trace parsing, query clustering and the collector/merge pipeline over the bank.
- `orchestration/graph/matcher/cache.py` — persistent consensus-match cache; `shared/identity.sanitize_metadata` and endpoint hashing.
- `shared/fingerprint/normalization.py` — IO-name vocabulary, alias merging and normalization audit records.
- `retrieval/search/runtime/{lexical,selector,disclosure}.py` and `retrieval/build/io/*` — field-weighted lexical scoring, prebuilt-index loading and prompt disclosure rendering.

| Python area | Why deferred |
|---|---|
| `experience/` — faiss IVF index training | Needs faiss' k-means/inverted lists (a GPU/ML stack); `ExperienceIndex` implements the exact flat cosine index instead, so `ivf`/`ivfpq` names return `Error::Unsupported` while every item remains searchable |
| `retrieval/` — `transformers`/`vLLM` logit-based branch selection | GPU inference stack; the LLM tree walk over `CompletionModel` and its lexical fallback are implemented |
| `shared/storage.py` — GCS and Azure Blob artifact transports | No in-crate transport; typed `Error::Unsupported` variants of `object_artifact::ArtifactTransport`, while S3-compatible and local object stores are implemented |

S3/OBS-compatible graph artifacts, SKILL.md scanning and the `SkillFs` catalog, flow distillation and `SkillPack` packaging/review, the experience bank, and LLM matching/scoring/retrieval/fingerprint/judging are all implemented with deterministic offline fallbacks.
Deferred areas are proven, not just documented: `tests/deferred.rs` asserts every remaining stub returns `Error::Unsupported`, and `tests/object_artifact.rs` asserts the GCS/Azure transports are typed unsupported errors.

## Error Strategy

- `Error::Validation` — malformed user input: blank ids/names/patterns/trace ids, illegal graph edge, bad version or meta name, draft/unverified/non-chain recipe, blank bucket, unknown artifact transport detail, runtime in the wrong state.
- `Error::NotFound` — explicit lookups that must exist (packing an unknown recipe); registry, store, and catalog reads return `Option`.
- `Error::Graph` — plan execution failures, wrapping the failing step id and the runner error message.
- `Error::Unsupported` — the remaining deferred ML/GPU stubs, execution without a configured runner, and the GCS/Azure artifact transports.
- `Error::Llm` — model call or unusable-reply failures surfaced when `FallbackMode::Error` is configured; the deterministic mode converts them into the stage's documented fallback instead.
- `Error::Validation`/`Error::Retrieval` — argument and backend failures propagated from `openjiuwen-retrieval`; they are never swallowed.
- Artifact/store/flow I/O failures map to `Error::Config` (invalid root or endpoint) or `Error::External` (I/O); malformed individual records are warned and skipped rather than aborting a scan.

## Test Strategy

- Unit tests live beside modules (`#[cfg(test)] mod tests`); cross-module behavior lives in `crates/openjiuwen-symphony/tests/`.
- `tests/models.rs` — serde round-trips, `Card` impl, schemas, content hash, datatype `can_feed`, naming helpers.
- `tests/registry.rs` — CRUD, validation, ordering, persistence, `load`/`reload`, revision counter.
- `tests/search.rs` — BM25, filters, `top_k`, filter-only ordering, `HashEmbedding` semantic index.
- `tests/graph.rs` — graph from descriptors, edge validation/upsert, incident-edge removal, JGF round-trip.
- `tests/artifact.rs` — memory/file artifact stores, immutability, `current` pointer, atomicity, illegal versions, reopen hydration.
- `tests/planner.rs` — candidate strategies/ordering, plan selection, filters, depth, scoring/ties, statuses, serialization, `stages`.
- `tests/executor.rs` — compile + execute with a fake runner, per-step state keys, input overlay, `max_steps`, `fail_fast=false`, `UnsupportedRunner`.
- `tests/runtime.rs` — lifecycle idempotency, invoke, build revisions, mutation idempotency, evolution EMA, lazy semantic re-embedding, stopped-runtime errors, snapshot.
- `tests/discovery.rs` — immediate/recursive scans, front-matter diagnostics, content-hash stability, inventory fingerprint, catalog search/excerpts, disabled skills, symlinks, settings.
- `tests/experience.rs` — generated/explicit ids, dedupe merging, lexical and cosine ranking, skill filters, JSONL/SQLite persistence, id-counter continuity.
- `tests/experience_index.rs` — `ExperienceIndex` add/search/dedupe, `top_k`/threshold ordering, skill-id flattening, `exist`, digest/count/dimension corruption, empty-directory open, persist round-trip and idempotence, reopen id continuity, and the IVF/unknown-algorithm errors.
- `tests/flow.rs` — trace normalization, ingest rejection/idempotency, grouping/quality, versioning, rehydration, `SkillPack` render/save/load, review verdicts, install preparation, narrative fallback.
- `tests/llm.rs` — every stage with a scripted model and with no model: tree walk/fallback, ontology consensus, scoring, fingerprint, judge repair, round-trips, error mode.
- `tests/object_artifact.rs` — numbered revisions, `current` pointer/status, reopen persistence, object keys, pinned-version skipping, unsupported transports, blank bucket, malformed `current.json`.
- `tests/deferred.rs` — every remaining stub returns `Error::Unsupported` and `error::is_unsupported` recognizes it.
- Tests are offline and deterministic: `tempfile` for filesystem stores, `HashEmbedding` for semantic search, no network, no clock-dependent assertions.
- `cargo fmt -p openjiuwen-symphony`, `cargo test -p openjiuwen-symphony`, and `cargo clippy -p openjiuwen-symphony --all-targets -- -D warnings` plus `--features sqlite` compile checks must pass.

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
- `./src/object_artifact.rs` → `ObjectGraphArtifactStore` over `ObjectStore` (in-memory/local/S3), `ArtifactTransport`
- `./src/discovery/` → SKILL.md discovery and the lexical skill catalog; see [`src/discovery/CONTEXT.md`](src/discovery/CONTEXT.md)
- `./src/experience/` → the experience bank and the persisted `ExperienceIndex`; see [`src/experience/CONTEXT.md`](src/experience/CONTEXT.md)
- `./src/flow/` → trace distillation, recipes, SkillPacks, review; see [`src/flow/CONTEXT.md`](src/flow/CONTEXT.md)
- `./src/llm/` → LLM-guided stages with deterministic fallbacks; see [`src/llm/CONTEXT.md`](src/llm/CONTEXT.md)
- `./src/orchestration/` → planning and execution
  - `./src/orchestration/plan.rs` → `OrchestrationPlan`, `PlanStep`, `PlanStatus`, JGF projection, stages
  - `./src/orchestration/candidates.rs` → `CandidateGenerator`, relation candidates
  - `./src/orchestration/planner.rs` → `Planner`, `PlanConfig`, `SemanticScorer`, language helpers
  - `./src/orchestration/executor.rs` → `PlanExecutor`, `CapabilityRunner`, plan → `openjiuwen-graph`
- `./src/evaluation.rs` → fail-soft evaluation suite and static metrics
- `./src/runtime.rs` → `SymphonyRuntime`, config, lifecycle, mutations, evolution
- `./src/deferred.rs` → the remaining `Error::Unsupported` ML/GPU stubs
- `./tests/` → behavior suites: `models`, `registry`, `search`, `graph`, `artifact`, `planner`, `executor`, `runtime`, `discovery`, `experience`, `experience_index`, `flow`, `llm`, `object_artifact`, `deferred`
