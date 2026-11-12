# openjiuwen-symphony

## Intent

The capability-asset engine of openJiuwen, mapping to the Python package `openjiuwen.symphony` (v0.1.20, ~52k lines, 201 files).
Python's engine is host-driven: it fingerprints capability assets, indexes them, builds a capability relationship graph, plans asset compositions with an LLM, and distils observed flows into skills.
This crate ports the **offline, deterministic core**: the asset cards, a real registry with CRUD and search, a keyword/metadata + embedding-backed semantic retrieval layer, a versioned `can_feed` capability graph with atomic artifact publication, a bounded deterministic planner, and an executor that maps a plan onto `openjiuwen-graph`.
LLM-guided retrieval, LLM ontology matching, LLM-judged evaluation, and the ML-backed experience bank are deferred behind documented `Error::Unsupported` stubs in [`deferred`].
Everything here works offline, without external services, and deterministically.

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
pub struct CapabilityRegistry { /* std::sync::RwLock<BTreeMap<CapabilityKey, CapabilityDescriptor>>, Option<Arc<dyn AssetStore>> */ }
impl CapabilityRegistry {
    pub fn new() -> Self;
    pub fn with_store(store: Arc<dyn AssetStore>) -> Self;          // write-through; call `reload` to hydrate
    pub async fn load(store: Arc<dyn AssetStore>) -> Result<Self>;   // hydrate from the store
    pub async fn reload(&self) -> Result<usize>;                     // replace the in-memory map from the store
    pub fn store(&self) -> Option<&Arc<dyn AssetStore>>;
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
`mean(semantic)` is the mean scorer value over the path's steps, falling back to Python's `SEED_BASELINE_SCORE` (`0.5`) when no step is scored; `mean(edge_confidence)` is the mean edge weight along the path, `1.0` when the path has no edges (Python `_state_edge_confidence`).
A step input is *missing* when it is required (`required != Some(false)`; `None` means required) and neither an earlier step output (normalized name plus compatible type: equal, either side `any`/unknown, or a `DataTypeVocabulary` subtype) nor a `provided_inputs` name supplies it; `missing_inputs` and `filled_inputs` hold sorted, deduped normalized input names.
`candidate_ids`/`disabled_ids` prune expansion targets as well as the start set; `available` gates the start set only, so a withdrawn capability can still be a downstream step.
`plan_id` is a fresh `uuid_hex()`, so determinism holds for every field except `plan_id` and the `planned_graph.graph.id` that embeds it.
Rank key: `(-score, missing_inputs, step_count, step_ids, edge_indices)` — the first element wins, later elements only break exact ties.
The Rust planner has no graph edge indices (paths are linear), so the final slot is the path's edge weights; float comparisons use `total_cmp` and non-finite scores are sanitized to `0.0`.
Start set: capabilities with a positive scorer value (all when the query has no tokens), minus `disabled_ids`, intersected with `candidate_ids` when given, `available` only.
Expansion follows graph edges with `weight >= min_edge_confidence`, never revisiting a step, bounded by `max_depth`, `max_expansions` and `max_plans`.
`Fast` keeps one greedy best partial path per start plus its single-node plan; `Beam` keeps the best `top_k` partial paths per depth before ranking.
`status` is `Ready` when the best plan's root steps have no missing inputs, `NeedsInput` when a plan exists but a root input is unfilled (listed in `message`), and `NoPlan` when no candidate plan exists.

**Execution mapping:** each plan step becomes an `openjiuwen_graph` node whose id is the step id; plan edges become graph edges; entries are steps with no incoming edge and exits steps with no outgoing edge.

A step's inputs start from `state["inputs"]` (an object) and are overlaid by predecessor step outputs whose normalized name matches one of the step's input names (a predecessor with exactly one output also fills a step's single unmatched input).
Each node writes the top-level state key equal to its step id: `{"success": bool, "output": Value, "error": Option<String>, "duration_ms": u64}`.
With `fail_fast` the first runner error aborts the run with `Error::Graph("step `<id>` failed: ...")`; otherwise every step runs and `PlanExecution::failed_steps()` reports the failures.
`max_steps` is checked before compilation; exceeding it is `Error::Validation`.
The default session is `WorkflowSession::new(<generated uuid>)` when the caller passes `None`.
`execute` seeds no inputs (`execute_with_inputs` is the seeding entry point); an empty plan executes to an empty `PlanExecution` with `state = {}` rather than failing to compile.
The engine prefixes node failures with `node `<id>` failed at superstep <n>: `; `execute` trims that wrapper so callers see exactly the documented `step `<id>` failed: <message>` text.
A `Cancelled` or `Unsupported` runner error keeps its own [`ErrorKind`] instead of becoming `Error::Graph`, so a missing runner stays recognisable through [`error::is_unsupported`].

### `evaluation` (`src/evaluation.rs`)

```rust
pub struct EvaluationInput { pub capability: CapabilityDescriptor, pub target: Option<String> }
pub struct MetricScore { pub metric: String, pub score: f64, pub passed: bool, pub detail: Option<String> }
pub struct EvaluationReport { pub capability_id: String, pub scores: Vec<MetricScore>, pub overall: f64, pub errors: Vec<String> }
#[async_trait]
pub trait Metric: Send + Sync { fn name(&self) -> &str; async fn evaluate(&self, input: &EvaluationInput) -> Result<MetricScore>; }
pub struct CompletenessMetric;      // description, tags, and described IO
pub struct SchemaValidityMetric;    // non-blank unique IO names, required flags coherent
pub struct LlmJudgedMetric;         // evaluate -> Error::Unsupported
pub struct EvaluationSuite;
impl EvaluationSuite {
    pub fn new(metrics: Vec<Arc<dyn Metric>>) -> Self;
    pub fn default_suite() -> Self;                 // CompletenessMetric + SchemaValidityMetric
    pub fn with_llm_judge(self) -> Self;            // adds LlmJudgedMetric (records a fail-soft error)
    pub fn metrics(&self) -> &[Arc<dyn Metric>];
    pub async fn evaluate(&self, input: &EvaluationInput) -> EvaluationReport;  // fail-soft: metric errors -> report.errors
}
```

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
    pub async fn semantic_search(&self, query: &str, top_k: usize) -> Result<Vec<SearchHit>>;
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

Lifecycle: `Created` → `start` → `Running` → `stop`/`close` → `Stopped`; both `start` and `stop` are idempotent and `close` is `stop` plus artifact flush.
`start` builds and publishes the graph when no revision exists.
`build` re-derives the graph from the registry plus candidate edges and publishes a new revision.
`add/update/remove_capabilities` mutate the registry, re-derive and publish the graph, and are idempotent per `request_id` (a replay returns the recorded result with `replayed = true`).
`plan` uses the current revision, building one when none exists.
`invoke` requires `Running`; it plans, optionally executes when `execute` is set (with the configured runner), and returns both.
`submit_evolution` applies successful step transitions as an exponential moving average on the corresponding graph edge weights (`w' = (w + 1.0) / 2`), publishes the new revision, and reports accepted/rejected edge counts.
`capture_graph_snapshot` returns `{"current_version", "graph_id", "graph_scope_id"}` with `null` values when no revision exists.
`invoke` and the three mutations require `Running` (`Error::Validation` naming `runtime.state` otherwise); planning, graph reads, search, build, evolution and the plain accessors work in every state.
`start` publishes the first revision before flipping the state to `Running`, so a failed build leaves the runtime untouched, and it also restarts a `Stopped` runtime.
A blank `request_id` is `Error::Validation` naming `request_id`; a replayed `request_id` publishes nothing.
`update` requires every key to already exist (otherwise `Error::NotFound` naming the qualified id, with the whole batch rejected), while `remove` skips absent keys.
`graph()` is infallible: when the artifact read fails it logs a warning and answers with the catalog-derived, node-only graph.
`capture_graph_snapshot` is synchronous, so it reads a materialized pointer refreshed on open, publish, and `graph_status`.
`submit_evolution` publishes only when at least one edge weight changed; otherwise it reports the current version.
`invoke` with `execute` set on a `NoPlan` result returns `execution: None` rather than an error.
`SymphonyRuntimeBuilder::with_asset_store` hydrates the registry from the store (`CapabilityRegistry::load`), so a preconfigured asset store survives restarts.

### `deferred` (`src/deferred.rs`)

API-shaped stubs whose constructors or entry points return [`Error::Unsupported`] via [`error::unsupported`]:

```rust
pub struct LlmTreeRetriever;         // Python `retrieval.Retriever`: LLM progressive tree disclosure
pub struct LlmTreeRetrieverConfig { pub top_k: usize, pub model: Option<String> }
pub struct TreeSearchRequest { pub top_k: Option<usize> }
pub struct TreeSearchResult { pub payloads: Vec<String>, pub method: String, pub elapsed_ms: u64 }
pub struct FlowEngine;               // Python `flow.SymphonyFlowEngine`: distill/pack
pub struct FlowReceipt { pub revision: String }
pub struct ExperienceBank;           // Python `experience`: faiss + sentence-transformers
pub struct ExperienceRecord { pub text: String, pub score: f64 }
pub struct SkillFolderScanner;       // Python `shared.fingerprint.SkillFolderScanner`: SKILL.md scanning
```

Each stub documents the Python class it mirrors and the reason it is deferred.

## Constraints

- Only files under `crates/openjiuwen-symphony/` may change; the root manifest, sibling crates, and `docs/` are read-only from this node.
- Depend downward only: `openjiuwen-common`, `openjiuwen-foundation`, `openjiuwen-graph`, `openjiuwen-retrieval`, `openjiuwen-session`, and the `openjiuwen-core` umbrella.
- No `unsafe`; no `println!`/`eprintln!` (use `tracing`); no panics on user input (no `unwrap`/`expect` outside tests).
- Every public item has a doc comment; files stay around ≤1000 lines and split by responsibility.
- All public data types are `serde`-serializable with `snake_case` field names, tolerant unknown fields, and empty collections as defaults.
- Async-first: tokio; `std::sync::RwLock` is never held across `.await` (snapshot, drop, then await); blocking filesystem work uses tokio's `fs`.
- Errors use `openjiuwen_common::error::Error` only; validation errors name the offending field.
- Determinism: no wall-clock ordering, no hash-map iteration order in outputs; ties break on qualified id; timestamps are metadata only.

## Design Decisions

- **A real registry exists in Rust.** Python inverts discovery to the host (`CapabilityProvider` protocol) and has no registry type; the objective requires CRUD/search, so `CapabilityRegistry` is the host-side provider plus a write-through [`AssetStore`]. It also satisfies `CapabilityProvider`-shaped access via `list`/`get`.
- **`CapabilityDescriptor::available` defaults to `true`** (Python parity: `available: bool = True` in `models/capability.py`), and the planner's start set filters on it, so a capability is plannable unless it is explicitly withdrawn with `with_available(false)`.
- **`top_k == 0` means unlimited** in search requests, a Rust-only sentinel; Python clamps every retrieval `top_k` to `max(1, ...)` and has no unlimited mode. The sentinel is explicit in [`SearchRequest`] and `SemanticIndex::search`.
- **Keyword search instead of LLM tree traversal for the default path.** Python's `retrieval.Retriever` is LLM-guided progressive tree disclosure (no vectors). BM25 is deterministic, offline, and covers the same selection job for a capability catalog; the LLM retriever keeps its API shape in [`deferred`].
- **Semantic search uses embeddings**, via `openjiuwen_retrieval::{Embedding, InMemoryVectorStore}` (with `HashEmbedding` for deterministic tests). This is additive: Python symphony has no embedding path, and the search result type is shared with keyword search.
- **Deterministic scoring replaces LLM semantic scoring.** `SemanticScorer` is the seam; `LexicalSemanticScorer` (query-term coverage) is the default, so planning is reproducible offline. An LLM-backed scorer is a drop-in implementation of the same trait and is deferred.
- **Planning is a bounded graph search**, not an LLM beam search: same score formula and rank tuple as Python, with explicit `max_expansions`/`max_plans` bounds for termination.
- **The graph engine facade maps onto `openjiuwen-graph` for execution.** Python's `graph_engine` is a JGF-document facade with no executor; here `CapabilityGraph`/`GraphArtifactStore` keep the document semantics and `PlanExecutor` compiles a plan into a real `openjiuwen_graph::CompiledGraph`.
- **Registering capabilities is not the same as planning them.** Mutations rebuild the graph and publish a revision; planning never mutates the registry.
- **Revisions are immutable and atomically published** (`tmp` + `rename`, `current.json` pointer), matching Python's artifact store; no file locks are needed because only one writer (the runtime) exists per root.
- **Evolution feedback is an EMA on observed edge weight**; Python's append-only evidence store, merged revisions, and revision pinning are reduced to a single observable weight update.
- **Python's `close()` raising when a flow engine is configured does not apply**: this crate has no flow engine, so `close` is always idempotent.
- **The file artifact store persists a `{"revision", "graph"}` envelope** (`current.json` points at the current revision's envelope) so a reopened store reproduces revisions byte-for-byte, including `created_at`, without clock-based ordering.
- **A generated version bumps past pinned ids**: `None` publishes `v{n+1}` with `n` = revision count, skipping forward when a caller already pinned that id, so an explicit `v2` never makes a later generated publish fail.
- **An illegal version is a validation error on read as well as write** (`read(Some("../evil"))`), because it would otherwise escape the store root; a legal-but-unknown version returns `Ok(None)`.
- **The planner follows the Python scoring constants exactly** (`SEED_BASELINE_SCORE = 0.5` for an unscored path, `1.0` edge confidence for an edgeless path) while using this contract's distinct-input coverage definition, which counts every declared input rather than Python's required-flag count.
- **`src/runtime.rs` is ~1050 lines by design**: it is the single composition root, and splitting it would contradict this routing table; every other module stays well under the ~1000-line guideline.
- **Static evaluation is a documented rubric** (four equal completeness checks, three equal schema-validity checks) because Python's metrics are LLM-judged; the LLM metric stays available as a fail-soft no-op.

## Deferred Scope

These Python areas are not implemented; each keeps an API-shaped stub in [`deferred`] (or `evaluation::LlmJudgedMetric`) that returns `Error::Unsupported` with a reason.

| Python area | Why deferred |
|---|---|
| `retrieval/` — `Retriever` LLM progressive tree disclosure, offline tree building, `transformers`/`vLLM` logit-selection backends | LLM/GPU stack; the deterministic BM25 + embedding paths are implemented instead |
| `orchestration/graph/matcher/ontology.py` — LLM `OntologyMatcher` / consensus matching | LLM judgement; `CandidateGenerator` covers the deterministic candidate rules |
| `orchestration/planning/` fast + bidirectional beam planners' semantic scoring | Semantic scores come from the LLM; the deterministic planner is implemented |
| `flow/` — `SymphonyFlowEngine` distill/review/SkillPack packaging | LLM narrative/review gate and a large artifact pipeline |
| `experience/` — trace → experience KB | `faiss`, `sentence-transformers`, OpenAI embeddings (and it is orphaned in Python) |
| `shared/fingerprint/` — SKILL.md scanning, LLM fingerprint extraction | Filesystem scanning + LLM extraction pipeline; static `CapabilityDescriptor::fingerprint` is implemented |
| `evaluation/` LLM-judged metrics | LLM judgement; static metrics are implemented |
| S3/OBS artifact storage (`shared/storage.py`) | External object storage; local filesystem only |
| `discovery/` `SkillFS` + BM25 content search | Host-side installed-skill catalog; deferred with the scanner |

## Error Strategy

- `Error::Validation` — malformed user input: blank ids/names, illegal graph edge, unknown version, blank query where required, runtime in the wrong state.
- `Error::NotFound` — reserved for explicit lookups that must exist (currently `AssetStore`/registry reads return `Option`).
- `Error::Graph` — plan execution failures, wrapping the failing step id and the runner error message.
- `Error::Unsupported` — deferred areas and execution without a configured runner.
- `Error::Validation`/`Error::Retrieval` — argument and backend failures propagated from `openjiuwen-retrieval`; they are never swallowed.
- Artifact/store I/O failures map to `Error::Config` (invalid root) or `Error::External` (I/O) — never ignored.

## Test Strategy

- Unit tests live beside modules (`#[cfg(test)] mod tests`); cross-module behavior lives in `crates/openjiuwen-symphony/tests/`.
- `tests/models.rs` — serde round-trips (`type` key, unknown-field tolerance, skipped `semantic_content`), `Card` impl, derived JSON schemas, content hash stability, datatype `can_feed` matrix, `normalize_name_key`/`fuzzy_name_distance`.
- `tests/registry.rs` — CRUD, validation errors, deterministic ordering, JSON-file/KV persistence and `load`/`reload`, remove/clear.
- `tests/search.rs` — BM25 ranking and matching, metadata/tag/type/available filters, `top_k` semantics, filter-only ordering, `HashEmbedding` semantic index add/remove/search.
- `tests/graph.rs` — graph from descriptors, edge validation/upsert, incident-edge removal, roots/leaves, JGF round-trip, malformed-document rejections.
- `tests/artifact.rs` — memory + file artifact stores (versions, immutability, `current` pointer, missing status, `tmp`+`rename` atomicity, illegal versions, reopen hydration).
- `tests/planner.rs` — candidate strategies/ordering, plan selection, disabled/candidate filters, `max_depth`, scoring/rank tie-breaks, `Ready`/`NeedsInput`/`NoPlan`, language normalization, plan serialization and `stages`.
- `tests/executor.rs` — compile + execute with a fake runner, per-step state keys, input overlay, `max_steps` rejection, `fail_fast=false` failure reporting, `UnsupportedRunner`.
- `tests/runtime.rs` — open/start/stop/close idempotency, invoke (plan-only and execute), build revisions, mutation idempotency by `request_id`, evolution EMA, semantic search, stopped-runtime errors, snapshot.
- `tests/deferred.rs` — every stub returns `Error::Unsupported` and `error::is_unsupported` recognizes it.
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
- `./src/orchestration/` → planning and execution
  - `./src/orchestration/plan.rs` → `OrchestrationPlan`, `PlanStep`, `PlanStatus`, JGF projection, stages
  - `./src/orchestration/candidates.rs` → `CandidateGenerator`, relation candidates
  - `./src/orchestration/planner.rs` → `Planner`, `PlanConfig`, `SemanticScorer`, language helpers
  - `./src/orchestration/executor.rs` → `PlanExecutor`, `CapabilityRunner`, plan → `openjiuwen-graph`
- `./src/evaluation.rs` → fail-soft evaluation suite and static metrics
- `./src/runtime.rs` → `SymphonyRuntime`, config, lifecycle, mutations, evolution
- `./src/deferred.rs` → documented `Error::Unsupported` stubs
- `./tests/` → behavior tests listed in Test Strategy
