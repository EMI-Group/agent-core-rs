# openjiuwen-symphony

## Intent

The capability-asset engine of openJiuwen, mapping to the Python package `openjiuwen.symphony` (v0.1.20, ~52k lines, 201 files).
Python's engine is host-driven: it fingerprints capability assets, indexes them, builds a capability relationship graph, plans asset compositions with an LLM, and distils observed flows into skills.
This crate ports that engine end to end: the asset cards, a real registry with CRUD and search, keyword and embedding-backed retrieval, SKILL.md discovery with a lexical skill catalog, a versioned `can_feed` capability graph with atomic artifact publication in-memory, on disk and in an object store, a bounded deterministic planner, a plan executor over `openjiuwen-graph`, flow distillation into versioned recipes and reviewed `SkillPack` artifacts, and a deduplicating experience bank.
Every LLM-guided stage (progressive tree retrieval, ontology matching, semantic plan scoring, fingerprint extraction, LLM judging, flow narratives) sits behind the `CompletionModel` seam and carries a documented deterministic fallback, so the crate is fully usable offline with no model configured.
Everything here is offline, deterministic, and free of external services by default.
The only remaining gaps are the Python areas that need a GPU/ML stack (a faiss + sentence-transformers experience index, `transformers`/`vLLM` logit selection) and two object-storage transports; they keep typed `Error::Unsupported` stubs in [`deferred`] and [`object_artifact`].

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

`evaluation::LlmJudgedMetric` remains the API-shaped in-module stub (`Error::Unsupported`, recorded as a fail-soft report error by `with_llm_judge`); the implemented LLM judge is [`llm::LlmJudgeMetric`](crate::llm::LlmJudgeMetric), which can be inserted into the suite as an ordinary [`Metric`].

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
`semantic_search` synchronizes lazily: the catalog is cleared and re-embedded only when `CapabilityRegistry::revision()` differs from the revision recorded by the last synchronization (or none is recorded yet), so an unchanged catalog embeds only the query; `start` does not seed the index.
`reindex_semantics` force-re-embeds the whole catalog, records the current revision, and returns the indexed count; it is the escape hatch after an embedding-model change and serializes with `semantic_search` through the same `tokio::sync::Mutex` (held across the catalog snapshot, the clear, and the embedding calls, released before the vector search).
Without a semantic index both entry points return `Error::Unsupported`.

### `discovery` (`src/discovery/`)

```rust
pub struct SkillFolderScanner;
impl SkillFolderScanner {
    pub fn new(root: impl Into<PathBuf>) -> Self;
    pub fn with_config(root: impl Into<PathBuf>, config: SkillScanConfig) -> Self;
    pub fn root(&self) -> &Path;
    pub fn config(&self) -> &SkillScanConfig;
    pub async fn scan(&self) -> Result<Vec<CapabilityDescriptor>>;
    pub async fn scan_skills(&self) -> Result<Vec<ScannedSkill>>;
    pub async fn scan_result(&self) -> Result<SkillScanResult>;
    pub async fn discover(&self) -> Result<SkillDiscovery>;
}
pub struct SkillScanConfig { pub max_depth: Option<usize>, pub source: String, pub disabled_skills: BTreeSet<String>, pub recursive: bool }
impl SkillScanConfig {
    pub fn recursive(max_depth: Option<usize>) -> Self;
    pub fn with_source(self, source: impl Into<String>) -> Self;
    pub fn with_max_depth(self, max_depth: usize) -> Self;
    pub fn with_disabled_skills(self, skills: impl IntoIterator<Item = impl Into<String>>) -> Self;
    pub fn effective_max_depth(&self) -> usize;              // max_depth or DEFAULT_MAX_DEPTH (4)
    pub fn is_disabled(&self, folder_name: &str, descriptor: &CapabilityDescriptor) -> bool;
}
pub enum DiagnosticSeverity { Warning, Error }               // Warning default
pub struct SkillDiagnostic { pub code: String, pub severity: DiagnosticSeverity, pub path: String, pub message: String }
pub struct ScannedSkill { pub descriptor: CapabilityDescriptor, pub body: String }
pub struct SkillDiscovery { pub skills: Vec<ScannedSkill>, pub diagnostics: Vec<SkillDiagnostic>, pub source: String }
pub struct SkillScanResult { pub capabilities: Vec<CapabilityDescriptor>, pub diagnostics: Vec<SkillDiagnostic>, pub source: String }
impl SkillScanResult { pub fn error_codes(&self) -> Vec<&str>; pub fn capability_ids(&self) -> Vec<&str>; }

pub enum FrontMatterStatus { Ok, Missing, Unterminated }     // Missing default
pub struct FrontMatterOutcome { pub status: FrontMatterStatus, pub metadata: Option<Value>, pub body: String, pub error: Option<String> }
impl FrontMatterOutcome { pub fn fences_ok(&self) -> bool; pub fn mapping(&self) -> Option<&Map<String, Value>>; pub fn get(&self, key: &str) -> Option<&Value>; pub fn get_str(&self, key: &str) -> Option<&str>; pub fn first_str(&self, keys: &[&str]) -> Option<&str>; pub fn get_bool(&self, key: &str) -> Option<bool>; }
pub fn parse_front_matter(raw: &str) -> FrontMatterOutcome;
pub fn boolean_value(value: &Value) -> Option<bool>;
pub fn normalize_capability_type(raw: &str) -> String;
pub fn tag_values(value: &Value) -> Vec<String>;
pub fn io_values(value: &Value) -> Vec<CapabilityIo>;

pub struct SkillRecord { pub worker_id: String, pub name: String, pub description: String, pub skill_file: String, pub source: String, pub version: String, pub author: String, pub content_hash: String, pub source_root: String, pub aliases: Vec<String> }
pub struct SkillInventory { pub items: Vec<SkillRecord>, pub fingerprint: String, pub count: usize }
impl SkillInventory { pub fn new(items: Vec<SkillRecord>) -> Self; pub fn is_empty(&self) -> bool; pub fn get(&self, worker_id: &str) -> Option<&SkillRecord>; pub fn worker_ids(&self) -> Vec<&str>; }
pub fn sort_records(records: &mut [SkillRecord]);
pub fn inventory_fingerprint(records: &[SkillRecord]) -> String;

pub struct SkillFs;
impl SkillFs {
    pub async fn open(root: impl AsRef<Path>) -> Result<Self>;
    pub async fn open_with(config: SkillScanConfig, root: impl AsRef<Path>) -> Result<Self>;
    pub fn root(&self) -> &Path;
    pub fn config(&self) -> &SkillScanConfig;
    pub fn inventory(&self) -> &SkillInventory;
    pub fn records(&self) -> &[SkillRecord];
    pub fn diagnostics(&self) -> &[SkillDiagnostic];
    pub fn read_body(&self, worker_id: &str) -> Option<&str>;
    pub fn selection_cards(&self) -> Vec<SelectionCard>;
    pub fn search(&self, query: &str, top_k: usize) -> Result<Vec<SkillHit>>;   // 0 = unlimited
}
pub struct SkillHit { pub worker_id: String, pub name: String, pub description: String, pub score: f32, pub matched_terms: Vec<String>, pub excerpt: String }
pub struct SelectionCard { pub worker_id: String, pub name: String, pub description: String, pub tags: Vec<String>, pub version: String }
pub const EXCERPT_WIDTH: usize = 160;
pub const EXCERPT_LEAD: usize = 60;

pub struct DiscoverySettings { pub max_results: usize, pub max_output_chars: usize, pub candidate_budget_ratio: f64, pub context_window_tokens: usize, pub use_existing_index: bool }
impl DiscoverySettings {
    pub fn candidate_budget_tokens(&self) -> usize;          // context_window_tokens * clamp(ratio, 0, 1)
    pub fn from_value(value: Value) -> Self;                 // lenient: every unusable field keeps its default
}
```
The module also exports the scan constants (`SKILL_CAPABILITY_TYPE` = `skill`, `SKILL_FILE_NAMES`, `RECURSIVE_SKILL_FILE`, `DEFAULT_MAX_DEPTH` = 4, `MAX_DESCRIPTION_CHARS` = 500), the `DIAGNOSTIC_*` codes, `DEFAULT_SOURCE` = `local`, and the `DEFAULT_*` settings constants.

The default scan mode ports `scan_skill_directories`: every immediate child directory is probed for `SKILL.md`/`skill.md`/`Skill.md` in that priority order, with no recursion below a skill folder, and names starting with `.` or `_` are skipped.
Recursive mode walks to `effective_max_depth()` for an exact `SKILL.md` match and does not descend into a folder that holds one; symlinks are skipped with a diagnostic.
A scan never fails for one bad skill: every problem becomes a `SkillDiagnostic` and the scan continues.
Warnings keep the skill and explain the gap: missing front matter (descriptor derived from the body), malformed YAML (metadata dropped), a skipped symlink, and a duplicate capability id (first skill kept).
Errors skip the affected skill: unreadable file or directory, non-UTF-8 content, unterminated front matter, and a parsed skill without a usable identity.
Only an unusable root is a scan error: `Error::Validation` naming `root`.
A descriptor's capability id is the front-matter `capability_id` or `id`, otherwise a hyphenated `normalize_name_key` slug of the folder name; its name is `name`, else the folder name, else the id; `capability_type` is normalized (`skills` → `skill`, `subagent`/`sub-agent` → `agent`).
A missing description falls back to the first body paragraph, whitespace-collapsed and capped at `MAX_DESCRIPTION_CHARS` (500) with `…`.
The descriptor metadata carries `entrypoint` (the `SKILL.md` path relative to the root with `/` separators), `version`, `author`, `source_root`, and optional `aliases`; its `content_hash` is the SHA-256 hex digest of the raw `SKILL.md` bytes.
Unknown front-matter keys never reach the descriptor or its retrieval text.
`SkillRecord`s reuse those fields, and the inventory fingerprint is the SHA-256 hex digest of the canonical JSON array of records sorted by `(worker_id, skill_file)` with object keys sorted, so it is independent of scan order and vector order.
`SkillFs` indexes every descriptor with `semantic_content = body` through [`LexicalIndex`], ranks hits by BM25 descending and ties on `worker_id`, and treats `top_k == 0` as unlimited.
An empty or all-punctuation query is a filter-only request returning every record in `worker_id` order with score `0.0` and an empty excerpt.
An excerpt is the whitespace-collapsed window of `EXCERPT_WIDTH` characters (`EXCERPT_LEAD` kept before the term) around the earliest matched term of whichever of the description and body matches more terms, with `…` on a truncated side.
Front matter is an optional YAML mapping between a first line holding exactly `---` (CRLF tolerated) and a closing `---` or `...`, parsed with `serde_yaml_ng`; a null, non-mapping, or malformed document yields `metadata = None` plus an `error` note instead of failing or panicking.

### `experience` (`src/experience/`)

```rust
pub struct ExperienceItem {
    pub id: String,                                   // generated `exp_NNNN` when blank
    pub query_pattern: String, pub query_examples: Vec<String>, pub skill_ids: Vec<String>,
    pub success_count: u32,                           // default 1
    #[serde(default, skip_serializing)] pub embedding: Vec<f32>,
    pub created_at: f64, pub last_hit_at: f64,       // metadata only, never used for ordering
}
impl ExperienceItem {
    pub fn new(query_pattern: impl Into<String>) -> Self;
    pub fn with_id(self, id: impl Into<String>) -> Self;
    pub fn with_query_examples(self, examples: impl IntoIterator<Item = impl Into<String>>) -> Self;
    pub fn with_skill_ids(self, skill_ids: impl IntoIterator<Item = impl Into<String>>) -> Self;
    pub fn with_success_count(self, success_count: u32) -> Self;
    pub fn with_embedding(self, embedding: Vec<f32>) -> Self;
    pub fn with_created_at(self, created_at: f64) -> Self;
    pub fn with_last_hit_at(self, last_hit_at: f64) -> Self;
    pub fn normalized_pattern(&self) -> String;       // lowercased, whitespace-collapsed
    pub fn dedupe_key(&self) -> String;               // sha256(normalized pattern + sorted deduped skill ids)
    pub fn search_text(&self) -> String;              // pattern + examples + skill ids, one per line
    pub fn skill_set(&self) -> BTreeSet<&str>;
    pub fn has_skill(&self, skill_id: &str) -> bool;
}

pub struct ExperienceBank;
impl ExperienceBank {
    pub fn new() -> Self;                                                // in-memory store
    pub fn with_store(store: Arc<dyn ExperienceStore>) -> Self;          // starts empty; use `load` to hydrate
    pub async fn load(store: Arc<dyn ExperienceStore>) -> Result<Self>;
    pub async fn open_json_file(path: impl AsRef<Path>) -> Result<Self>;
    #[cfg(feature = "sqlite")] pub async fn open_sqlite(url: &str) -> Result<Self>;
    pub fn with_embedding(self, embedding: Arc<dyn Embedding>) -> Self;
    pub fn store(&self) -> &Arc<dyn ExperienceStore>;
    pub fn embedding(&self) -> Option<&Arc<dyn Embedding>>;
    pub fn len(&self) -> usize; pub fn is_empty(&self) -> bool;
    pub fn get(&self, id: &str) -> Option<ExperienceItem>;
    pub fn list(&self) -> Vec<ExperienceItem>;
    pub fn exists_for_skills(&self, skill_ids: &[String]) -> bool;
    pub fn generate_id(&self) -> String;                                 // `exp_0001`, zero-padded to ID_DIGITS (4)
    pub async fn add(&self, item: ExperienceItem) -> Result<AddOutcome>;
    pub async fn add_batch(&self, items: Vec<ExperienceItem>) -> Result<Vec<AddOutcome>>;
    pub async fn remove(&self, id: &str) -> Result<bool>;
    pub async fn clear(&self) -> Result<()>;
    pub async fn search(&self, query: &str, top_k: usize) -> Result<Vec<ExperienceHit>>;
    pub async fn search_with_options(&self, query: &str, options: &SearchOptions) -> Result<Vec<ExperienceHit>>;
    pub async fn search_with_ranking(&self, query: &str, ranking: &ExperienceRanking) -> Result<Vec<ExperienceHit>>;
    pub async fn search_with_skill_ids(&self, query: &str, top_k: usize) -> Result<Vec<String>>;
}
pub struct AddOutcome { pub id: String, pub merged: bool }

pub struct ExperienceHit { pub score: f64, pub item: ExperienceItem }
pub struct ExperienceRanking { pub top_k: usize, pub min_score: Option<f64>, pub require_all_skills: bool }
pub struct SearchOptions { pub top_k: usize, pub min_score: Option<f64>, pub required_skills: Vec<String>, pub excluded_skills: Vec<String>, pub require_all_skills: bool }
pub fn cosine_similarity(left: &[f32], right: &[f32]) -> f64;   // 0.0 on empty/mismatched/zero-norm, clamped to [-1, 1]

#[async_trait]
pub trait ExperienceStore: Send + Sync {
    async fn list(&self) -> Result<Vec<ExperienceItem>>;    // sorted by id; malformed entries warned and skipped
    async fn get(&self, id: &str) -> Result<Option<ExperienceItem>>;
    async fn put(&self, item: &ExperienceItem) -> Result<()>;
    async fn remove(&self, id: &str) -> Result<bool>;
    async fn clear(&self) -> Result<()>;
}
pub struct MemoryExperienceStore;
pub struct JsonFileExperienceStore;                         // ::open(path), ::path(), ::reload()
#[cfg(feature = "sqlite")] pub struct SqliteExperienceStore;  // ::connect(url), ::with_kv(kv), ::prefix()
```

`add` is deterministic: a blank normalized `query_pattern` is `Error::Validation` naming `query_pattern`; a blank id becomes the next free `exp_NNNN`; an explicit id is kept trimmed; and the item is persisted before it becomes visible in memory.
A duplicate (same normalized pattern and same skill set, regardless of skill order) is merged instead of inserted: success counters add up, example lists union in first-seen order, the later `last_hit_at` wins, an existing embedding survives an empty one, the stored id is preserved, and `AddOutcome.merged` is `true`.
The generated-id counter continues past the highest `exp_NNNN` loaded from a store and is never reset by `clear`, so ids are not reused.
Ranking uses cosine similarity when an embedder is configured and the query is non-blank, and a deterministic lexical term-coverage score otherwise: a distinct query term found in the normalized pattern scores `1.0`, one found only in the examples or skill ids scores `0.5`, and the sum is divided by the number of distinct query terms so scores land in `[0, 1]`.
Both paths filter `excluded_skills` and `required_skills` (all of them by default, any when `require_all_skills` is false), sort by score descending with ties on id ascending, truncate to `top_k` (`0` = unlimited), and only then apply `min_score`, matching Python's `search_by_embedding` order.
`JsonFileExperienceStore` rewrites one JSONL document through `<file>.tmp` plus `rename`, accepts JSONL or a JSON array on load, and warns-and-skips blank, malformed, or id-less entries; a missing file is an empty bank.
`SqliteExperienceStore` (feature `sqlite`) stores each item's JSON under `experience/<id>` in a foundation `SqliteKvStore`, the same key prefix layout as capability descriptors.

### `flow` (`src/flow/`)

```rust
pub enum TraceOutcome { Success, Failed, Partial, Unknown }   // aliases: succeeded/ok/done, failure/error, partially
pub struct TraceNode { pub capability_id: String, pub capability_type: String, pub skill_type: String, pub version: String, pub description: String, pub inputs: Vec<CapabilityIo>, pub outputs: Vec<CapabilityIo> }
pub struct TraceEdge { pub source: String, pub target: String, pub relation: String /*can_feed*/, pub success: bool, pub reason: String }
pub struct ExecutionGraph { pub id: String, #[serde(rename = "type")] pub graph_type: String, pub label: String, pub directed: bool, pub nodes: BTreeMap<String, TraceNode>, pub edges: Vec<TraceEdge> }
pub struct FlowTrace { pub trace_id: String, pub query: String, pub outcome: TraceOutcome, pub graph: ExecutionGraph }
impl FlowTrace {
    pub fn from_value(value: Value) -> Result<Self>;    // strips skill:/capability: prefixes; keeps success edges with existing endpoints; sorts edges
    pub fn successful_edges(&self) -> Vec<(String, String, String)>;
}

pub struct RecipeQuality { pub execution_count: u64, pub success_count: u64, pub failure_count: u64, pub pack_success_rate: f64, pub qualified_edge_count: usize }
pub struct RecipeEdge { pub source: String, pub target: String, pub relation: String }
pub struct CombinationStructure { #[serde(rename = "type")] pub structure_type: String /*skill_pack*/, pub direction: String /*directed*/, pub nodes: BTreeMap<String, Value>, pub edges: Vec<RecipeEdge>, pub loop_guards: Vec<Value> }
pub struct ExperienceRecipe {
    pub schema_version: String /*symphony.experience_recipe.v1*/, pub recipe_id: String, pub name: String,
    pub version: u64 /*starts at 1*/, pub status: String /*draft|active*/, pub grade: String /*candidate|verified*/,
    pub applicability: BTreeMap<String, Value>, pub combination_structure: CombinationStructure,
    pub execution_narrative: String, pub quality: RecipeQuality, pub provenance: BTreeMap<String, Value>,
}
impl ExperienceRecipe { pub fn capability_ids(&self) -> Vec<String>; pub fn applicability_text(&self, key: &str) -> &str; pub fn content_identity(&self) -> String; pub fn is_active_verified(&self) -> bool; pub fn candidate(&self) -> CombinationCandidate; }
pub struct CombinationCandidate;    // display projection of an active, verified recipe
pub fn topological_order(structure: &CombinationStructure) -> Vec<String>;
pub fn sha256_hex(text: &str) -> String; pub fn sha256_short(text: &str, length: usize) -> String;
pub fn canonical_json(value: &Value) -> String; pub fn content_hash(value: &Value) -> String;
pub fn normalize_capability_id(raw: &str) -> String; pub fn normalize_meta_name(raw: &str, fallback: &str) -> String; pub fn is_valid_meta_name(name: &str) -> bool;

pub struct FlowEngine;
impl FlowEngine {
    pub fn new(root: impl Into<PathBuf>) -> Self;            // no disk reads or writes
    pub async fn open(root: impl AsRef<Path>) -> Result<Self>;
    pub fn with_model(self, model: Arc<dyn CompletionModel>) -> Self;
    pub fn root(&self) -> &Path;
    pub async fn ingest(&self, trace: FlowTrace) -> Result<bool>;
    pub async fn ingest_value(&self, value: Value) -> Result<bool>;
    pub async fn evidence(&self) -> Vec<FlowTrace>;
    pub async fn distill(&self) -> Result<DistillReport>;
    pub async fn recipes(&self) -> Vec<ExperienceRecipe>;
    pub async fn recipe(&self, recipe_id: &str) -> Option<ExperienceRecipe>;
    pub async fn candidates(&self) -> Vec<CombinationCandidate>;
    pub async fn pack(&self, recipe_id: &str) -> Result<SkillPack>;
    pub async fn package(&self, package_id: &str) -> Option<SkillPack>;
    pub async fn review_and_prepare_install(&self, recipe_id: &str, target_kind: &str, dest_dir: impl AsRef<Path>) -> Result<InstallPreparation>;
    pub async fn prepare_install(&self, recipe_id: &str, recipe_version: u64, target_kind: &str, dest_dir: impl AsRef<Path>) -> Result<InstallPreparation>;
    pub async fn close(&self) -> Result<()>;                 // every write is already durable
}
pub struct DistillReport { pub evidence_total: usize, pub groups_total: usize, pub groups_without_structure: Vec<String>, pub recipes_saved: Vec<String>, pub recipes_unchanged: Vec<String>, pub narrative_fallbacks: Vec<String> }

pub struct SkillPackStep { pub id: String, pub name: String, pub description: String, pub capability_type: String, pub inputs: Vec<CapabilityIo>, pub outputs: Vec<CapabilityIo>, pub depends_on: Vec<String> }
pub struct SkillPack {
    pub schema_version: String /*symphony.capability_package.v1*/, pub package_id: String /*cap-<sha256(materials)[:12]*/,
    pub recipe_id: String, pub recipe_version: u64, pub target_kind: String /*skill*/, pub name: String,
    pub description: String /*[技能包] ...*/, pub steps: Vec<SkillPackStep>,
    pub artifacts: BTreeMap<String, String>, pub permissions: Vec<String>, pub license: String,
}
impl SkillPack {
    pub fn materials(&self) -> Value;  pub fn materials_hash(&self) -> String;  pub fn compute_package_id(&self) -> String;
    pub fn recompute_identity(&mut self); pub fn step_ids(&self) -> Vec<&str>; pub fn task_description(&self) -> &str;
    pub fn render(&self) -> BTreeMap<String, String>;
    pub async fn save(&self, dir: impl AsRef<Path>) -> Result<()>;
    pub async fn load(dir: impl AsRef<Path>) -> Result<Self>;
    pub fn to_json(&self) -> Result<String>; pub fn from_value(value: Value) -> Result<Self>;
}

pub struct Reviewer;
impl Reviewer { pub fn new() -> Self; pub fn review(&self, pack: &SkillPack) -> ReviewResult; }
pub struct ReviewCheck { pub check: String, pub result: String /*pass|fail*/, pub reasons: Vec<String> }
pub struct ReviewResult { pub review_id: String, pub package_id: String, pub target_kind: String, pub verdict: String, pub checks: Vec<ReviewCheck>, pub materials_hash: String, pub reviewed_at: String, pub schema_version: String }
impl ReviewResult { pub fn failed_checks(&self) -> Vec<&ReviewCheck>; pub fn failure_reasons(&self) -> Vec<String>; }
pub struct InstallPreparation { pub recipe_id: String, pub target_kind: String, pub verdict: String, pub package: Option<SkillPack>, pub artifact_dir: Option<PathBuf>, pub review: Option<ReviewResult>, pub reasons: Vec<String> }
```

`FlowEngine::ingest` accepts only successful traces with at least one successful edge and is idempotent per `trace_id` (a replay answers `false`); a blank id, a non-success outcome, or an edgeless graph is `Error::Validation`.
Distillation groups successful traces by the cycle-broken, sorted successful-edge signature `source>target:relation` joined with `|`; the recipe id is `recipe_<sha256(signature)[:12]>`.
The first distillation of a group writes version `1`, and a changed content identity (name, applicability, structure, or narrative) writes `version + 1`; replaying unchanged evidence writes nothing and appears in `recipes_unchanged`.
Quality counts are captured when a version is written: `pack_success_rate` is `success_count / execution_count` rounded to four decimals.
A recipe is `active` + `verified` when `success_count >= MIN_SUCCESSES` (1) and `pack_success_rate >= MIN_PACK_SUCCESS_RATE` (0.8); otherwise it stays `draft` + `candidate`.
A configured `CompletionModel` only supplies name, applicability, and narrative prose; every structural decision is deterministic, and a failed or unusable reply falls back to the template and is reported in `narrative_fallbacks`.
The flow store writes eagerly and atomically: `evidence.jsonl` (one trace per line, ordered by `trace_id`), `recipes/<recipe_id>/v<N>.json`, `recipes/<recipe_id>/current.json`, and `packages/<package_id>.json`, each through a unique `.<name>.<uuid>.tmp` sibling plus a rename; malformed files are warned and skipped on load.
`pack` freezes only an `active`, `verified` recipe whose structure is a simple linear chain (otherwise `Error::Validation`), and an unknown recipe is `Error::NotFound`.
`SkillPack` materials are `recipe_id`, `target_kind`, `name`, `description`, and the ordered steps, and the content-addressed id is `cap-<sha256(canonical materials)[:12]>`; artifacts, permissions, and license are excluded.
The rendered `SKILL.md` has YAML front matter (`name`, `kind: skillpack`, `description`, `skills` in step order) and the body sections `# name`, `When to use`, `Required inputs`, `Included Skills`, `Execution Process`, `Failure handling`, `Final output`, and a fenced `Workflow Graph` block.
The workflow-graph block is `{"graph":{"id":"<name>-workflow","type":"skillpack_workflow","label":<description>,"directed":true,"nodes":...,"edges":...}}`; `dependencies.yaml` carries the ordered capabilities plus a package header, and `load` validates the front-matter skill order and the graph block against it.
The `Reviewer` runs seven checks in order — `structure`, `dependencies`, `integrity`, `meta_naming`, `materials`, `permissions`, `license` — and returns `approved` when all pass, `needs_human_review` when `meta_naming` is the only failure, and `rejected` otherwise.
A pack may not request `credential_access`, `filesystem_write`, `process`, or `shell`, and may only declare `Apache-2.0`, `BSD-3-Clause`, `MIT`, or `Proprietary`; `review_id` is `review_<12 hex>` over the identity fields, and `reviewed_at` is metadata only.
`review_and_prepare_install`/`prepare_install` return structural rejections (unsupported target kind, unknown recipe, stale requested version, failed review) as `InstallPreparation::verdict` with reasons, not errors, and only I/O failures are `Err`.

### `llm` (`src/llm/`)

```rust
pub use openjiuwen_retrieval::{extract_json_object, CompletionModel, StaticCompletionModel};
pub struct FoundationCompletionModel;   // adapter over Arc<dyn openjiuwen_foundation::llm::Model>
impl FoundationCompletionModel { pub fn new(model: Arc<dyn Model>) -> Self; pub fn with_system(self, system: impl Into<String>) -> Self; pub fn with_temperature(self, temperature: f32) -> Self; pub fn with_max_tokens(self, max_tokens: usize) -> Self; pub fn model(&self) -> &Arc<dyn Model>; }

pub enum FallbackMode { Deterministic, Error }   // Deterministic default
pub struct ModelConfig { pub model: Option<Arc<dyn CompletionModel>>, pub temperature: Option<f32>, pub max_tokens: Option<usize>, pub fallback: FallbackMode }
impl ModelConfig { pub fn new() -> Self; pub fn with_model(self, model: Arc<dyn CompletionModel>) -> Self; pub fn with_temperature(self, temperature: f32) -> Self; pub fn with_max_tokens(self, max_tokens: usize) -> Self; pub fn with_fallback(self, fallback: FallbackMode) -> Self; pub fn model(&self) -> Option<&dyn CompletionModel>; pub fn is_configured(&self) -> bool; }
pub fn render_prompt(system: &str, user: &str) -> String;
pub async fn complete_json(model: Option<&dyn CompletionModel>, system: &str, user: &str) -> Result<Option<Value>>;   // no model => Ok(None)

pub struct RetrieverNode { pub node_id: String, pub label: String, pub description: String, pub children: Vec<RetrieverNode>, pub items: Vec<RetrieverItem> }
pub struct RetrieverItem { pub item_id: String, pub payload: String, pub label: String, pub description: String }
pub struct TreeRetrieverConfig { pub top_k: usize /*10*/, pub model: Option<Arc<dyn CompletionModel>>, pub max_depth: usize /*16*/ }
pub struct TreeSearchRequest { pub top_k: Option<usize> }        // 0 = unlimited
pub struct TreeSearchResult { pub payloads: Vec<String>, pub method: String, pub elapsed_ms: u64 }
pub type LlmTreeRetriever = TreeRetriever;
impl TreeRetriever { pub fn new(config: TreeRetrieverConfig) -> Self; pub fn with_root(self, root: RetrieverNode) -> Self; pub fn config(&self) -> &TreeRetrieverConfig; pub fn root(&self) -> &RetrieverNode; pub async fn search(&self, query: &str, request: &TreeSearchRequest) -> Result<TreeSearchResult>; }
pub const METHOD_LLM: &str = "llm"; pub const METHOD_LEXICAL_FALLBACK: &str = "lexical_fallback";

pub struct OntologyConfig { pub threshold: f64 /*0.7*/, pub require_consensus: bool /*true*/, pub max_candidates: usize /*32, 0 = unlimited*/, pub model: Option<Arc<dyn CompletionModel>>, pub fallback: FallbackMode }
pub struct OntologyMatcher;
impl OntologyMatcher { pub fn new(capabilities: impl IntoIterator<Item = CapabilityDescriptor>) -> Self; pub fn with_config(self, config: OntologyConfig) -> Self; pub fn with_model(self, model: Arc<dyn CompletionModel>) -> Self; pub fn config(&self) -> &OntologyConfig; pub fn capabilities(&self) -> &[CapabilityDescriptor]; pub async fn match_candidates(&self, candidates: &[RelationCandidate]) -> Result<Vec<OntologyMatch>>; }
pub struct OntologyMatch { pub source_id: String, pub target_id: String, pub relation: String /*can_feed*/, pub confidence: f64, pub method: String, pub reasons: Vec<String>, pub accepted: bool }
pub const METHOD_CONSENSUS: &str = "llm_consensus_match"; pub const METHOD_DETERMINISTIC: &str = "deterministic_fallback";

pub struct LlmSemanticScorer;   // implements orchestration::SemanticScorer
impl LlmSemanticScorer { pub fn new(config: ModelConfig) -> Self; pub fn with_model(model: Arc<dyn CompletionModel>) -> Self; pub fn config(&self) -> &ModelConfig; }
pub use crate::orchestration::LexicalSemanticScorer as DeterministicScorer;

pub struct FingerprintConfig { pub fallback: FallbackMode }
pub struct FingerprintExtractor;
impl FingerprintExtractor { pub fn new() -> Self; pub fn with_model(self, model: Arc<dyn CompletionModel>) -> Self; pub fn with_config(self, config: FingerprintConfig) -> Self; pub fn config(&self) -> &FingerprintConfig; pub fn has_model(&self) -> bool; pub async fn extract(&self, descriptor: &CapabilityDescriptor) -> Result<SemanticProfile>; }

pub struct LlmJudgeMetric;   // implements evaluation::Metric
impl LlmJudgeMetric { pub fn new(name: impl Into<String>, rubric: impl Into<String>) -> Self; pub fn with_model(self, model: Arc<dyn CompletionModel>) -> Self; pub fn with_rubric(self, rubric: impl Into<String>) -> Self; pub fn with_fallback(self, fallback: FallbackMode) -> Self; pub fn rubric(&self) -> &str; pub fn has_model(&self) -> bool; }
pub const DEFAULT_JUDGE_NAME: &str = "llm_judge";
```

The seam is one prompt in, one reply out: each stage renders its own system/user prompt, calls `complete_json`, and parses the reply; a reply with no extractable JSON object is a soft `None` with a `tracing::warn!`, while a model invocation failure is returned so the stage can apply its `FallbackMode`.
`TreeRetriever` follows a model-chosen branch down to `max_depth`, drops unknown ids, deduplicates payloads, and reports `method = "llm"`; with no model, an empty reply, or a failed call it ranks the reachable tree lexically by query-term coverage and reports `"lexical_fallback"`.
`OntologyMatcher` validates `RelationCandidate`s: candidates with unknown capabilities are skipped, the batch is sorted by `(source, target)` and capped by `max_candidates`, the model picks `forward`/`reverse` and a confidence that is clamped to `[0, 1]`, and the orientation is always resolved to `source_id -> target_id` with `relation = can_feed`.
With `require_consensus` the batch is asked twice (forward and reversed) and only the intersection survives, keeping the lower confidence; `accepted` is `confidence >= threshold`, and matches sort by `(source_id, target_id)`.
Without a model, or with an unusable reply under `FallbackMode::Deterministic`, the matcher reports each candidate's own rule confidence with `method = "deterministic_fallback"`; `FallbackMode::Error` surfaces the failure as `Error::Llm`.
`LlmSemanticScorer` reads `{"judgements":[{"candidate_id","score","reason"}]}`, drops unknown ids and duplicate verdicts, treats a missing or non-numeric score as `0.0`, and clamps to `[0, 1]`; with no model or a failed/unusable reply it delegates to `LexicalSemanticScorer`.
`FingerprintExtractor` reads only the reply's `semantic_profile` (the descriptor stays authoritative for classification and tags), deduplicates lists first-wins, drops blanks, and appends the model's keywords to the card's tags; the deterministic fallback is `summary` = description or name, `capabilities` = qualified id, `use_cases`/`keywords` = tags, `limitations` = empty.
`LlmJudgeMetric` accepts only `{"score":0|1,"reason":"<non-empty>"}`, makes exactly one repair attempt that re-sends the request with the validation error appended, and then applies the fallback mode; the deterministic rubric gives three equal checks (described description, described input, described output) so all three score `1.0`, a description alone `0.5`, and a card without a description `0.0`, passing at `>= 0.5` with `detail` prefixed `deterministic_fallback:`.

### `object_artifact` (`src/object_artifact.rs`)

```rust
pub enum ArtifactTransport { ObjectStore, S3(S3Config), Gcs, AzureBlob }   // snake_case
impl ArtifactTransport { pub fn as_str(&self) -> &'static str; }
pub struct ObjectGraphArtifactStore;
impl ObjectGraphArtifactStore {
    pub fn new(store: Arc<dyn ObjectStore>, bucket: impl Into<String>, prefix: impl Into<String>) -> Self;
    pub fn in_memory(prefix: impl Into<String>) -> Self;                     // InMemoryObjectStore + DEFAULT_BUCKET
    pub async fn local(root: impl AsRef<Path>, prefix: impl Into<String>) -> Result<Self>;   // LocalObjectStore
    pub fn s3(config: S3Config, bucket: impl Into<String>, prefix: impl Into<String>) -> Result<Self>;
    pub fn open(transport: ArtifactTransport, bucket: impl Into<String>, prefix: impl Into<String>) -> Result<Self>;
    pub fn bucket(&self) -> &str;
    pub fn prefix(&self) -> &str;
}
impl GraphArtifactStore for ObjectGraphArtifactStore;   // publish/read/status/versions
pub const DEFAULT_BUCKET: &str = "openjiuwen-artifacts";
```

Object keys mirror the filesystem backend below the normalized (`/`-trimmed) prefix: `<prefix>/versions/<version>/graph.json` and `<prefix>/current.json`, each holding the same `{"revision", "graph"}` envelope, with an empty prefix placing documents at the bucket root.
A revision object is always written before the `current` pointer, so a reader sees either the previous pointer with its immutable revision or the new one, never a pointer to a half-written graph.
Publications are serialized by an async mutex so two tasks cannot claim the same generated version; generated versions skip pinned identifiers exactly like `FileGraphArtifactStore`.
The store reads the bucket on every call instead of caching, so it never serves a revision another writer published after construction; missing buckets are tolerated on the read path and created on first publish.
A blank bucket is `Error::Validation` on every operation; a malformed `current.json` is an error rather than a silent "missing", while malformed listing entries are skipped with a warning.
`ArtifactTransport::Gcs` and `ArtifactTransport::AzureBlob` return typed `Error::Unsupported`, and `ArtifactTransport::ObjectStore` is `Error::Validation` pointing at `new`.

### `deferred` (`src/deferred.rs`)

Two API-shaped stubs remain, and each entry point returns [`Error::Unsupported`] via [`error::unsupported`] with a reason that points at this section:

```rust
pub struct ExperienceIndex;   // Python `experience`: faiss + sentence-transformers vector index
pub struct LogitSelector;     // Python `retrieval`: transformers/vLLM logit-based branch selection
```

Every other Python area these stubs used to cover now has a real implementation: LLM tree retrieval lives in [`llm::TreeRetriever`], flow distillation and packaging in [`flow`], the experience bank in [`experience`], and SKILL.md scanning in [`discovery`].

## Notes for Agents

- Intra-doc links in a module-level (`//!`) doc block do not resolve names that are defined or re-exported in that same file module (rustdoc answers `no item named X in scope`, with no file span); write them as labeled links to absolute paths, e.g. `[`AssetStore`](crate::store::AssetStore)`.
- Items in sibling modules and dependency types are linkable the same way (`crate::model::CapabilityDescriptor`, `openjiuwen_foundation::store::KvStore`); std generics used in prose and private items stay plain code spans.
- `cargo doc -p openjiuwen-symphony --no-deps` is the definition of done for doc links: it must print no warnings.

## Constraints

- Cargo: `edition 2021`, `rust-version 1.85`, `publish = false`, `#![forbid(unsafe_code)]`; `default = []` and the sole feature `sqlite` only enables the SQLite backends (`store::KvAssetStore::sqlite` and `experience::SqliteExperienceStore`) via `openjiuwen-foundation/sqlite`, so a default build and `--no-default-features` are identical.
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

- **A real registry exists in Rust.** Python inverts discovery to the host (`CapabilityProvider` protocol) and has no registry type; the objective requires CRUD/search, so `CapabilityRegistry` is the host-side provider plus a write-through [`AssetStore`]. It also satisfies `CapabilityProvider`-shaped access via `list`/`get`.
- **`CapabilityDescriptor::available` defaults to `true`** (Python parity: `available: bool = True` in `models/capability.py`), and the planner's start set filters on it, so a capability is plannable unless it is explicitly withdrawn with `with_available(false)`.
- **`top_k == 0` means unlimited** in search requests, a Rust-only sentinel; Python clamps every retrieval `top_k` to `max(1, ...)` and has no unlimited mode. The sentinel is explicit in [`SearchRequest`] and `SemanticIndex::search`.
- **Two retrieval paths over one catalog.** BM25 keyword search stays the deterministic default; `llm::TreeRetriever` implements Python's LLM-guided progressive tree walk and falls back to the same lexical ranking, so a model is an optimisation rather than a requirement.
- **The LLM stages share one seam and one fallback switch.** `openjiuwen_retrieval::CompletionModel` keeps the stages model-agnostic, and `FallbackMode::Deterministic` (the default) guarantees every stage has a documented offline answer while `FallbackMode::Error` opts into surfacing model failures.
- **SKILL.md discovery is a scan, not a plugin protocol.** `SkillFolderScanner` keeps both Python modes (immediate-child and bounded recursive), reports per-skill problems as diagnostics instead of failing the scan, and derives ordinary capability descriptors so discovered skills flow straight into the registry, graph, and planner.
- **The skill catalog reuses the crate's own retrieval.** `SkillFs` builds a `LexicalIndex` over the discovered descriptors with the markdown body as `semantic_content` and maps hits back to `SkillRecord`s, so skill search and capability search share BM25 semantics instead of duplicating them.
- **An inventory fingerprint is order-independent.** The SHA-256 is taken over canonical JSON of records sorted by `(worker_id, skill_file)`, so directory iteration order and vector construction order cannot change it.
- **Experience ranking is embedding-optional.** The bank persists items, not vectors: a configured `Embedding` re-derives them at search time (cosine), and without one a deterministic lexical term-coverage score keeps the bank usable offline.
- **Experience dedupe is a Rust-only improvement.** Python appends blindly and keys on `id`; here the normalized pattern plus the sorted skill set is the merge key, so repeated observations reinforce one record instead of fragmenting the catalog.
- **Flow distillation is structural, not narrative.** Evidence groups on its cycle-broken successful-edge signature, versions are content-addressed and immutable, and a model can only write prose; every status, grade, and id is deterministic.
- **A SkillPack is a frozen, reviewable artifact.** Only an `active` + `verified` linear chain can be packed, the package id hashes canonical materials, and a static seven-check review decides `approved`/`needs_human_review`/`rejected` without a model or network.
- **Object storage is the third artifact backend, not a new abstraction.** `ObjectGraphArtifactStore` reuses the exact revision envelope and key layout of `FileGraphArtifactStore`, so local, in-memory, and S3-compatible stores are interchangeable; only the object store implementation changes.
- **Deferred surface is small and typed.** The remaining gaps (faiss/sentence-transformers experience index, `transformers`/`vLLM` logit selection, GCS/Azure transports) fail with `Error::Unsupported`, never with a silent degradation.
- **File and artifact writes are atomic everywhere.** Flow stores, experience JSON documents, and graph artifacts all write a temp sibling and rename, so a reader never observes a partial document.
- **Timestamps never decide behavior.** `reviewed_at`, `distilled_at`, `created_at`, and `last_hit_at` are metadata; ordering always comes from sorted ids or content hashes.
- **Semantic search uses embeddings**, via `openjiuwen_retrieval::{Embedding, InMemoryVectorStore}` (with `HashEmbedding` for deterministic tests). This is additive: Python symphony has no embedding path, and the search result type is shared with keyword search.
- **The semantic index is catalog-revision-driven, not query-driven.** A `CapabilityRegistry::revision()` counter lets `semantic_search` embed the catalog only after a mutation (one re-embed per change instead of per query), which is what keeps a real embedding provider affordable; `reindex_semantics` covers the changes the counter cannot see (a new embedding model).
- **`SemanticScorer` has two implementations.** `LexicalSemanticScorer` (query-term coverage) is the deterministic default so planning is reproducible offline, and `llm::LlmSemanticScorer` adds model-judged scores with the lexical scorer as its fallback.
- **Planning is a bounded graph search**, not an LLM beam search: same score formula and rank tuple as Python, with explicit `max_expansions`/`max_plans` bounds for termination.
- **The graph engine facade maps onto `openjiuwen-graph` for execution.** Python's `graph_engine` is a JGF-document facade with no executor; here `CapabilityGraph`/`GraphArtifactStore` keep the document semantics and `PlanExecutor` compiles a plan into a real `openjiuwen_graph::CompiledGraph`.
- **Registering capabilities is not the same as planning them.** Mutations rebuild the graph and publish a revision; planning never mutates the registry.
- **Revisions are immutable and atomically published** (`tmp` + `rename` on disk, revision-before-pointer in an object store), matching Python's artifact store; one store is the writer of its root or prefix, and object-store publications serialize through an async mutex.
- **Evolution feedback is an EMA on observed edge weight**; Python's append-only evidence store, merged revisions, and revision pinning are reduced to a single observable weight update.
- **`FlowEngine::close` is a no-op**: every ingestion, recipe version, and package is written eagerly and atomically, so there is nothing left to flush and close never fails.
- **The file artifact store persists a `{"revision", "graph"}` envelope** (`current.json` points at the current revision's envelope) so a reopened store reproduces revisions byte-for-byte, including `created_at`, without clock-based ordering.
- **A generated version bumps past pinned ids**: `None` publishes `v{n+1}` with `n` = revision count, skipping forward when a caller already pinned that id, so an explicit `v2` never makes a later generated publish fail.
- **An illegal version is a validation error on read as well as write** (`read(Some("../evil"))`), because it would otherwise escape the store root; a legal-but-unknown version returns `Ok(None)`.
- **The planner follows the Python scoring constants exactly** (`SEED_BASELINE_SCORE = 0.5` for an unscored path, `1.0` edge confidence for an edgeless path) while using this contract's distinct-input coverage definition, which counts every declared input rather than Python's required-flag count.
- **`src/runtime.rs` is ~1050 lines by design**: it is the single composition root, and splitting it would contradict this routing table; every other module stays well under the ~1000-line guideline.
- **Static evaluation remains the offline default.** The evaluation suite ships deterministic completeness/schema metrics, and `llm::LlmJudgeMetric` implements Python's LLM-judged metric with a static description/input/output rubric as its fallback.

## Deferred Scope

Only GPU/ML-dependent and vendor-transport areas remain; each keeps a typed `Error::Unsupported` entry point.

| Python area | Why deferred |
|---|---|
| `experience/` — faiss + sentence-transformers vector index | A GPU/ML stack (`faiss`, `sentence-transformers`, OpenAI embeddings) is out of scope; the deterministic lexical ranking and the optional `Embedding`-backed cosine ranking are implemented instead |
| `retrieval/` — `transformers`/`vLLM` logit-based branch selection | GPU inference stack; the LLM tree walk over `CompletionModel` and its lexical fallback are implemented |
| `shared/storage.py` — GCS and Azure Blob artifact transports | No in-crate transport; exposed as typed `Error::Unsupported` variants of `object_artifact::ArtifactTransport`, while S3-compatible and local object stores are implemented |

S3/OBS-compatible graph artifacts, SKILL.md scanning and the `SkillFs` catalog, flow distillation and `SkillPack` packaging/review, the experience bank, and LLM matching/scoring/retrieval/fingerprint/judging are all implemented with deterministic offline fallbacks.
Deferred areas are proven, not just documented: `tests/deferred.rs` asserts every remaining stub returns `Error::Unsupported`, and `tests/object_artifact.rs` asserts the GCS/Azure transports are typed unsupported errors.

## Error Strategy

- `Error::Validation` — malformed user input: blank ids/names/patterns/trace ids, illegal graph edge, bad version or meta name, draft/unverified/non-chain recipe, blank bucket, unknown or unsupported artifact transport detail, runtime in the wrong state.
- `Error::NotFound` — explicit lookups that must exist (packing an unknown recipe); registry, store, and catalog reads return `Option`.
- `Error::Graph` — plan execution failures, wrapping the failing step id and the runner error message.
- `Error::Unsupported` — the remaining deferred ML/GPU stubs, execution without a configured runner, and the GCS/Azure artifact transports.
- `Error::Llm` — model call or unusable-reply failures surfaced when `FallbackMode::Error` is configured; the deterministic mode converts them into the stage's documented fallback instead.
- `Error::Validation`/`Error::Retrieval` — argument and backend failures propagated from `openjiuwen-retrieval`; they are never swallowed.
- Artifact/store/flow I/O failures map to `Error::Config` (invalid root or endpoint) or `Error::External` (I/O) — never ignored; malformed individual records are warned and skipped rather than aborting a scan.

## Test Strategy

- Unit tests live beside modules (`#[cfg(test)] mod tests`); cross-module behavior lives in `crates/openjiuwen-symphony/tests/`.
- `tests/models.rs` — serde round-trips (`type` key, unknown-field tolerance, skipped `semantic_content`), `Card` impl, derived JSON schemas, content hash stability, datatype `can_feed` matrix, `normalize_name_key`/`fuzzy_name_distance`.
- `tests/registry.rs` — CRUD, validation errors, deterministic ordering, JSON-file/KV persistence and `load`/`reload`, remove/clear, and the revision counter (advances per mutation, never on a no-op or rejected write).
- `tests/search.rs` — BM25 ranking and matching, metadata/tag/type/available filters, `top_k` semantics, filter-only ordering, `HashEmbedding` semantic index add/remove/search.
- `tests/graph.rs` — graph from descriptors, edge validation/upsert, incident-edge removal, roots/leaves, JGF round-trip, malformed-document rejections.
- `tests/artifact.rs` — memory + file artifact stores (versions, immutability, `current` pointer, missing status, `tmp`+`rename` atomicity, illegal versions, reopen hydration).
- `tests/planner.rs` — candidate strategies/ordering, plan selection, disabled/candidate filters, `max_depth`, scoring/rank tie-breaks, `Ready`/`NeedsInput`/`NoPlan`, language normalization, plan serialization and `stages`.
- `tests/executor.rs` — compile + execute with a fake runner, per-step state keys, input overlay, `max_steps` rejection, `fail_fast=false` failure reporting, `UnsupportedRunner`.
- `tests/runtime.rs` — open/start/stop/close idempotency, invoke (plan-only and execute), build revisions, mutation idempotency by `request_id`, evolution EMA, semantic search (including lazy re-embedding proven by a counting `Embedding`, add/remove invalidation, and `reindex_semantics`), stopped-runtime errors, snapshot.
- `tests/discovery.rs` — immediate-child and recursive scans, front-matter field parsing, missing/malformed/unterminated diagnostics, content-hash stability, inventory fingerprint order-independence, catalog records/bodies/selection cards, BM25 search with excerpts/`top_k`/ties, disabled skills, symlink skipping, source stamping, and settings defaults.
- `tests/experience.rs` — generated and explicit ids, blank-pattern rejection, dedupe merging versus distinct skill sets, lexical and `HashEmbedding` cosine ranking with thresholds and skill filters, `search_with_skill_ids`, JSON-file and SQLite persistence, corrupt-line tolerance, and id-counter continuity.
- `tests/flow.rs` — trace normalization, ingest rejection/idempotency, edge-signature grouping and quality counts, versioning on content change versus unchanged replay, persistence rehydration, `SkillPack` rendering/save/load/JSON round-trip, non-chain rejection, the seven review checks and verdicts, install preparation with stale-version rejection, and model-narrative fallback.
- `tests/llm.rs` — every stage twice (scripted `StaticCompletionModel` and no model): tree walk/ranking/fallback/limits, ontology confidence/direction/consensus/fallback, semantic scoring clamping and fallback, fingerprint extraction and dedupe, judge repair/fallback, JSON round-trips, and error-mode propagation.
- `tests/object_artifact.rs` — numbered immutable revisions, `current` pointer and status, reopen persistence, the documented object keys (including a prefix-free layout), pinned-version skipping, typed unsupported transports, blank-bucket validation, malformed `current.json`, and unrelated-object tolerance.
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
- `./src/discovery/` → SKILL.md discovery and the lexical skill catalog
  - `./src/discovery/frontmatter.rs` → `parse_front_matter`, statuses, tag/IO/boolean coercion
  - `./src/discovery/scanner.rs` → `SkillFolderScanner`, `SkillScanConfig`, descriptors, `SkillDiagnostic`s
  - `./src/discovery/models.rs` → `SkillRecord`, `SkillInventory`, inventory fingerprint
  - `./src/discovery/catalog.rs` → `SkillFs`, `SkillHit`, excerpts, `SelectionCard`
  - `./src/discovery/settings.rs` → `DiscoverySettings`, token budget
- `./src/experience/` → the experience bank
  - `./src/experience/mod.rs` → `ExperienceItem`, dedupe/normalization helpers
  - `./src/experience/bank.rs` → `ExperienceBank`, `AddOutcome`, generated ids, merge rules
  - `./src/experience/ranking.rs` → `ExperienceHit`, `ExperienceRanking`, `SearchOptions`, lexical/cosine scorers
  - `./src/experience/store.rs` → `ExperienceStore`, memory/JSON-file/SQLite backends
- `./src/flow/` → trace distillation, recipes, SkillPacks, review
  - `./src/flow/model/` → `FlowTrace`, `ExecutionGraph`, `TraceNode`/`TraceEdge`, recipes, canonical JSON, naming
  - `./src/flow/store.rs` → atomic flow-root persistence (`evidence.jsonl`, recipe versions, packages)
  - `./src/flow/engine.rs` → `FlowEngine`, `DistillReport`, grouping, versioning, quality/status
  - `./src/flow/pack.rs` → `SkillPack`, content-addressed identity, `SKILL.md`/`dependencies.yaml` render, save/load
  - `./src/flow/review.rs` → `Reviewer`, the seven checks, verdicts, `InstallPreparation`
- `./src/llm/` → LLM-guided stages with deterministic fallbacks
  - `./src/llm/mod.rs` → `CompletionModel` seam, `ModelConfig`, `FallbackMode`, `complete_json`
  - `./src/llm/tree.rs` → `TreeRetriever`/`LlmTreeRetriever`, lexical fallback
  - `./src/llm/ontology.rs` → `OntologyMatcher`, thresholds, consensus, fallback
  - `./src/llm/scoring.rs` → `LlmSemanticScorer`, `DeterministicScorer`
  - `./src/llm/fingerprint.rs` → `FingerprintExtractor`, `SemanticProfile` fallback
  - `./src/llm/judge.rs` → `LlmJudgeMetric`, repair attempt, static rubric
  - `./src/llm/foundation.rs` → `FoundationCompletionModel` adapter
- `./src/orchestration/` → planning and execution
  - `./src/orchestration/plan.rs` → `OrchestrationPlan`, `PlanStep`, `PlanStatus`, JGF projection, stages
  - `./src/orchestration/candidates.rs` → `CandidateGenerator`, relation candidates
  - `./src/orchestration/planner.rs` → `Planner`, `PlanConfig`, `SemanticScorer`, language helpers
  - `./src/orchestration/executor.rs` → `PlanExecutor`, `CapabilityRunner`, plan → `openjiuwen-graph`
- `./src/evaluation.rs` → fail-soft evaluation suite and static metrics
- `./src/runtime.rs` → `SymphonyRuntime`, config, lifecycle, mutations, evolution
- `./src/deferred.rs` → the remaining `Error::Unsupported` ML/GPU stubs
- `./tests/` → behavior tests listed in Test Strategy (`discovery`, `experience`, `flow`, `llm`, `object_artifact` included)
