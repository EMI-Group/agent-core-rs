# openjiuwen-symphony / llm

## Intent

LLM-guided stages over one small completion seam, mapping to Python's LLM-backed `symphony` stages.
The seam is one prompt in, one reply out: each stage renders its own system/user prompt, calls `complete_json`, and parses the reply.
Every stage carries a documented deterministic fallback under `FallbackMode::Deterministic` (the default), while `FallbackMode::Error` opts into surfacing model failures.
The stages are progressive tree retrieval, `can_feed` ontology matching, semantic plan scoring, fingerprint extraction, and LLM judging.
This keeps the crate fully usable offline with no model configured, and a model is an optimisation rather than a requirement.
Two supporting layers make model-backed matching economical and reproducible: `identity` derives stable model/metadata fingerprints, and `cache` memoizes ontology-match answers in a JSON file keyed by those fingerprints.

### Fallback table

| Stage | `FallbackMode::Deterministic` behavior | `FallbackMode::Error` |
|---|---|---|
| `TreeRetriever` | Rank the reachable tree lexically by query-term coverage, `method = "lexical_fallback"` | model failure returned |
| `OntologyMatcher` | Report each candidate's own rule confidence, `method = "deterministic_fallback"` | `Error::Llm` |
| `LlmSemanticScorer` | Delegate to `orchestration::LexicalSemanticScorer` (query-term coverage) | `Error::Llm` |
| `FingerprintExtractor` | Build a `SemanticProfile` from the descriptor (description/name, qualified id, tags) | `Error::Llm` |
| `LlmJudgeMetric` | Apply the static description/input/output rubric, `detail` prefixed `deterministic_fallback:` | `Error::Llm` |

A reply with no extractable JSON object is a soft `None` with a `tracing::warn!`; a model invocation failure is returned so the stage can apply its `FallbackMode`.

## API Surface

### Seam

```rust
pub use openjiuwen_retrieval::{extract_json_object, CompletionModel, StaticCompletionModel};

pub enum FallbackMode { Deterministic, Error }   // Deterministic default
pub struct ModelConfig { pub model: Option<Arc<dyn CompletionModel>>, pub temperature: Option<f32>, pub max_tokens: Option<usize>, pub fallback: FallbackMode }
impl ModelConfig {
    pub fn new() -> Self;
    pub fn with_model(self, model: Arc<dyn CompletionModel>) -> Self;
    pub fn with_temperature(self, temperature: f32) -> Self;
    pub fn with_max_tokens(self, max_tokens: usize) -> Self;
    pub fn with_fallback(self, fallback: FallbackMode) -> Self;
    pub fn model(&self) -> Option<&dyn CompletionModel>;
    pub fn is_configured(&self) -> bool;
}
pub fn render_prompt(system: &str, user: &str) -> String;
pub async fn complete_json(model: Option<&dyn CompletionModel>, system: &str, user: &str) -> Result<Option<Value>>;   // no model => Ok(None)

pub struct FoundationCompletionModel;   // adapter over Arc<dyn openjiuwen_foundation::llm::Model>
impl FoundationCompletionModel {
    pub fn new(model: Arc<dyn Model>) -> Self;
    pub fn with_system(self, system: impl Into<String>) -> Self;
    pub fn with_temperature(self, temperature: f32) -> Self;
    pub fn with_max_tokens(self, max_tokens: usize) -> Self;
    pub fn model(&self) -> &Arc<dyn Model>;
}
```

### Identity fingerprints
```rust
pub const IDENTITY_VERSION: &str = "Symphony-model-identity-v1";
pub const MAX_METADATA_DEPTH: usize = 64;

pub struct ModelIdentity {
    pub model: String, pub backend: String,
    pub temperature: Option<f64>, pub top_p: Option<f64>, pub max_tokens: Option<u64>,
    pub stop: Vec<String>,
    pub request_config_sha256: String, pub client_config_sha256: String,
}   // Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema
impl ModelIdentity {
    pub fn of(config: &ModelConfig, backend: &str) -> Self;
    pub fn with_top_p(self, top_p: f64) -> Self;   // refreshes the request digest
    pub fn with_stop(self, stop: impl IntoIterator<Item = impl Into<String>>) -> Self;
}

pub fn canonical_json(value: &Value) -> String;             // sorted keys, compact, depth-capped
pub fn stable_metadata_sha256(value: &Value) -> String;     // sha256 hex of canonical_json
pub fn endpoint_sha256(endpoint: &str) -> String;           // normalized endpoint digest
pub fn graph_identity_sha256(descriptor: &CapabilityDescriptor) -> String;
pub fn sanitize_metadata(value: &Value) -> Value;           // Python's recursive sanitiser
```
`ModelIdentity::of` reads the model name from `CompletionModel::name()`, defaults a blank/absent backend to `"none"`, and computes both digests as SHA-256 over canonical JSON: `request_config_sha256` covers `IDENTITY_VERSION`, `temperature`, `top_p`, `max_tokens`, `stop`; `client_config_sha256` covers `IDENTITY_VERSION`, `backend`, `model`. `ModelConfig` has no `top_p`/`stop`, so those start unset and only `with_top_p`/`with_stop` fill them in; a non-finite sampling value is treated as absent.
`canonical_json` sorts object keys recursively, keeps array order, emits no whitespace, renders a non-finite number (unreachable through `serde_json`) as `null`, and replaces containers deeper than `MAX_METADATA_DEPTH` with `null` so the result is always well-formed JSON and recursion never overflows.
`endpoint_sha256` trims, lowercases the scheme and host, drops userinfo, drops a default port (`https:443`, `http:80`), strips path trailing slashes, ignores the fragment — and **excludes the query string** (a documented difference from Python's sorted-query normalization, so a signed URL and its bare endpoint agree). Text without `scheme://`, or with an unparsable authority (unbracketed IPv6, out-of-range port), falls back to the trimmed, slash-stripped text.
`sanitize_metadata` drops keys `crate::redaction::is_sensitive_name` recognizes and keys whose normalized form ends in `_key`, `secret`, `token`, or `password`; replaces the five endpoint key names with `<key>_sha256`; keeps the five safe key fields (`cache_key`, `key_type`, `partition_key`, `public_key`, `routing_key`); and recurses through objects and arrays preserving key order, with `null` past the depth cap.
`graph_identity_sha256` hashes `capability_id`, `capability_type`, `name`, `description`, `inputs`, and `outputs` (Python's `graph_identity_dict` minus `version`, which the Rust descriptor lacks), so availability and metadata never invalidate a cache entry while a changed IO list, name, or description does.

### Match cache
```rust
pub const CACHE_SCHEMA_VERSION: &str = "Symphony-relation-match-cache-v1";

pub struct CachedMatch {
    pub candidate_id: String, pub matches: Value, pub diagnostics: Value, pub updated_at: String,
}   // Default: empty arrays, empty strings
impl CachedMatch {
    pub fn new(candidate_id: impl Into<String>, matches: Value) -> Self;   // stamps now_rfc3339()
    pub fn with_diagnostics(self, diagnostics: Value) -> Self;
    pub fn with_updated_at(self, updated_at: impl Into<String>) -> Self;   // caller-supplied metadata
}

pub struct MatchCache { /* PathBuf + BTreeMap<String, CachedMatch> */ }
impl MatchCache {
    pub fn new(path: impl Into<PathBuf>) -> Self;                  // empty, no I/O
    pub async fn open(path: impl Into<PathBuf>) -> Result<Self>;   // absent/empty file = empty cache
    pub fn path(&self) -> &Path;
    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
    pub fn contains(&self, key: &str) -> bool;
    pub fn keys(&self) -> impl Iterator<Item = &str>;              // file (sorted) order
    pub async fn lookup(&self, key: &str) -> Option<CachedMatch>;  // in-memory clone
    pub async fn record(&mut self, key: String, entry: CachedMatch) -> Result<()>;   // writes through
    pub async fn clear(&mut self) -> Result<()>;                   // empties the file, keeps it
}

pub fn cache_key(candidate: &Value, source_fingerprint_hash: &str, target_fingerprint_hash: &str, signature: &str) -> String;
pub fn cache_key_document(candidate: &Value, source_fingerprint_hash: &str, target_fingerprint_hash: &str, signature: &str) -> String;
```
The file shape is `{"schema_version": CACHE_SCHEMA_VERSION, "records": {key: {"candidate_id", "matches", "diagnostics", "updated_at"}}}`.
Records live in a `BTreeMap`, so two files written from the same records are byte-identical; `record` and `clear` rewrite the whole file through a temporary sibling and a rename, so a reader never sees a partial document.
There is no TTL, eviction, or size bound: the cache is a memo, not a policy, and `clear` is the only way to forget entries.
`updated_at` is metadata only — it never participates in a key or a lookup, and `MatchCache::lookup` cannot fail.
An absent or empty file opens empty; malformed JSON, a non-object envelope, or a foreign `schema_version` is `Error::Validation` naming the file (never a silent reset); a single malformed record is warned about and skipped.
`cache_key` is the SHA-256 of the canonical JSON of `{"candidate", "source_fingerprint_hash", "target_fingerprint_hash", "matcher_signature"}`, so key order never matters while any component change produces a new key.

### Tree retrieval

```rust
pub const METHOD_LLM: &str = "llm";
pub const METHOD_LEXICAL_FALLBACK: &str = "lexical_fallback";

pub struct RetrieverNode { pub node_id: String, pub label: String, pub description: String, pub children: Vec<RetrieverNode>, pub items: Vec<RetrieverItem> }
pub struct RetrieverItem { pub item_id: String, pub payload: String, pub label: String, pub description: String }
pub struct TreeRetrieverConfig { pub top_k: usize /*10*/, pub model: Option<Arc<dyn CompletionModel>>, pub max_depth: usize /*16*/ }
pub struct TreeSearchRequest { pub top_k: Option<usize> }        // 0 = unlimited
pub struct TreeSearchResult { pub payloads: Vec<String>, pub method: String, pub elapsed_ms: u64 }
pub type LlmTreeRetriever = TreeRetriever;
impl TreeRetriever {
    pub fn new(config: TreeRetrieverConfig) -> Self;
    pub fn with_root(self, root: RetrieverNode) -> Self;
    pub fn config(&self) -> &TreeRetrieverConfig;
    pub fn root(&self) -> &RetrieverNode;
    pub async fn search(&self, query: &str, request: &TreeSearchRequest) -> Result<TreeSearchResult>;
}
```

`TreeRetriever` follows a model-chosen branch down to `max_depth`, drops unknown ids, deduplicates payloads, and reports `method = "llm"`; with no model, an empty reply, or a failed call it ranks the reachable tree lexically by query-term coverage and reports `"lexical_fallback"`.

### Ontology matching

```rust
pub const METHOD_LLM: &str = "llm_ontology_match";
pub const METHOD_CONSENSUS: &str = "llm_consensus_match";
pub const METHOD_DETERMINISTIC: &str = "deterministic_fallback";

pub struct OntologyConfig { pub threshold: f64 /*0.7*/, pub require_consensus: bool /*true*/, pub max_candidates: usize /*32, 0 = unlimited*/, pub model: Option<Arc<dyn CompletionModel>>, pub fallback: FallbackMode }
pub struct OntologyMatcher;
impl OntologyMatcher {
    pub fn new(capabilities: impl IntoIterator<Item = CapabilityDescriptor>) -> Self;
    pub fn with_config(self, config: OntologyConfig) -> Self;
    pub fn with_model(self, model: Arc<dyn CompletionModel>) -> Self;
    pub fn config(&self) -> &OntologyConfig;
    pub fn capabilities(&self) -> &[CapabilityDescriptor];
    pub async fn match_candidates(&self, candidates: &[RelationCandidate]) -> Result<Vec<OntologyMatch>>;
    pub async fn match_candidates_with(&self, candidates: &[RelationCandidate], options: &MatchOptions) -> Result<Vec<OntologyMatch>>;
    pub fn with_cache(self, cache: Arc<tokio::sync::Mutex<MatchCache>>) -> Self;
    pub fn cache(&self) -> Option<&Arc<tokio::sync::Mutex<MatchCache>>>;
    pub fn matcher_signature(&self) -> String;
}

pub const MATCHER_VERSION: &str = "Symphony-ontology-matcher-v2";
pub const PROMPT_VERSION: &str = "Orchestration-graph-match-v2";
pub const MATCH_SCHEMA_VERSION: &str = "Symphony-ontology-match-schema-v1";

pub struct MatchOptions { pub force: bool }   // Default: false
impl MatchOptions {
    pub fn new() -> Self;
    pub fn with_force(self, force: bool) -> Self;
}
pub struct OntologyMatch { pub source_id: String, pub target_id: String, pub relation: String /*can_feed*/, pub confidence: f64, pub method: String, pub reasons: Vec<String>, pub accepted: bool }
```

`OntologyMatcher` validates `RelationCandidate`s: candidates with unknown capabilities are skipped, the batch is sorted by `(source, target)` and capped by `max_candidates`, the model picks `forward`/`reverse` and a confidence that is clamped to `[0, 1]`, and the orientation is always resolved to `source_id -> target_id` with `relation = can_feed`.

With `require_consensus` the batch is asked twice (forward and reversed) and only the intersection survives, keeping the lower confidence; `accepted` is `confidence >= threshold`, and matches sort by `(source_id, target_id)`.

Without a model, or with an unusable reply under `FallbackMode::Deterministic`, the matcher reports each candidate's own rule confidence with `method = "deterministic_fallback"`; `FallbackMode::Error` surfaces the failure as `Error::Llm`.
With `with_cache` installed (and a model configured), the matcher first replays every candidate whose `cache_key` is present, then asks the model about the remaining ones in one batch (two with consensus) — a fully cached call performs no model calls at all.
Fresh model answers are written back per candidate under the same key; a deterministic fallback answer is never cached, so a later run with a working model still asks.
`MatchOptions::with_force(true)` ignores every hit and refreshes the entries.
`matcher_signature` is canonical JSON over `MATCHER_VERSION`, `PROMPT_VERSION`, `MATCH_SCHEMA_VERSION`, `CACHE_SCHEMA_VERSION`, `IDENTITY_VERSION`, a digest of the system prompt, the `ModelIdentity` (whose backend is `CompletionModel::name()`), the threshold, the consensus switch, the candidate cap, and the relation/direction/method/description-limit strategy constants; a changed threshold, consensus flag, prompt, or model identity therefore misses the cache instead of returning a stale answer.

### Semantic scoring

```rust
pub struct LlmSemanticScorer;   // implements orchestration::SemanticScorer
impl LlmSemanticScorer {
    pub fn new(config: ModelConfig) -> Self;
    pub fn with_model(model: Arc<dyn CompletionModel>) -> Self;
    pub fn config(&self) -> &ModelConfig;
}
pub use crate::orchestration::LexicalSemanticScorer as DeterministicScorer;
```

`LlmSemanticScorer` reads `{"judgements":[{"candidate_id","score","reason"}]}`, drops unknown ids and duplicate verdicts, treats a missing or non-numeric score as `0.0`, and clamps to `[0, 1]`; with no model or a failed/unusable reply it delegates to `LexicalSemanticScorer`.

### Fingerprint extraction

```rust
pub struct FingerprintConfig { pub fallback: FallbackMode }
pub struct FingerprintExtractor;
impl FingerprintExtractor {
    pub fn new() -> Self;
    pub fn with_model(self, model: Arc<dyn CompletionModel>) -> Self;
    pub fn with_config(self, config: FingerprintConfig) -> Self;
    pub fn config(&self) -> &FingerprintConfig;
    pub fn has_model(&self) -> bool;
    pub async fn extract(&self, descriptor: &CapabilityDescriptor) -> Result<SemanticProfile>;
}
```

`FingerprintExtractor` reads only the reply's `semantic_profile` (the descriptor stays authoritative for classification and tags), deduplicates lists first-wins, drops blanks, and appends the model's keywords to the card's tags; the deterministic fallback is `summary` = description or name, `capabilities` = qualified id, `use_cases`/`keywords` = tags, `limitations` = empty.

### LLM judge

```rust
pub const DEFAULT_JUDGE_NAME: &str = "llm_judge";
pub const DEFAULT_RUBRIC: &str = "The capability card must be self-describing: a non-empty ...";

pub struct LlmJudgeMetric;   // implements evaluation::Metric
impl LlmJudgeMetric {
    pub fn new(name: impl Into<String>, rubric: impl Into<String>) -> Self;
    pub fn with_model(self, model: Arc<dyn CompletionModel>) -> Self;
    pub fn with_rubric(self, rubric: impl Into<String>) -> Self;
    pub fn with_fallback(self, fallback: FallbackMode) -> Self;
    pub fn rubric(&self) -> &str;
    pub fn has_model(&self) -> bool;
}
```

`LlmJudgeMetric` accepts only `{"score":0|1,"reason":"<non-empty>"}`, makes exactly one repair attempt that re-sends the request with the validation error appended, and then applies the fallback mode.

The deterministic rubric gives three equal checks (described description, described input, described output) so all three score `1.0`, a description alone `0.5`, and a card without a description `0.0`, passing at `>= 0.5` with `detail` prefixed `deterministic_fallback:`.
`evaluation::LlmJudgedMetric` is a thin delegating wrapper over `LlmJudgeMetric` (name `llm_judged`, default rubric, plus `with_model`/`with_rubric`/`with_fallback`/`rubric`/`has_model` and a delegating `Metric` impl), so the crate has exactly one judge implementation and an offline suite scores the card instead of erroring.

## Constraints

- One seam: stages depend on `CompletionModel`, never on a vendor SDK; `StaticCompletionModel` covers tests and `FoundationCompletionModel` adapts a foundation `Model`.
- `FallbackMode::Deterministic` must always produce a usable answer offline; only `FallbackMode::Error` may return `Error::Llm`.
- No network in default builds or tests; model calls happen only through an injected `CompletionModel`.
- Determinism: deduplication is first-wins, ordering is by id or `(source_id, target_id)`, clamping is explicit.
- No `unsafe`, no `println!`, no panics on user input; every public item is documented.
- Do not duplicate retrieval or orchestration types; re-export `CompletionModel`, `StaticCompletionModel`, and `DeterministicScorer`.
- Files stay around or under 1000 lines; `tree.rs` and `ontology.rs` are the largest and stay focused on their stage.
- The cache never changes a matching decision: without a cache the matcher is byte-identical, and every cache failure (read, decode, write) is `tracing::warn!`-logged and degrades to a normal model call.
- Identity helpers are total and deterministic: sorted-key canonical JSON, no wall-clock input to any digest, no panic on malformed or over-deep input.
- `MatchCache` is not internally synchronized; callers share it through a `tokio::sync::Mutex`, which is only ever held across in-memory work, never across a model call.

## Routing Table

- `./mod.rs` → `CompletionModel` seam re-exports, `ModelConfig`, `FallbackMode`, `render_prompt`, `complete_json`
- `./tree.rs` → `TreeRetriever`/`LlmTreeRetriever`, `RetrieverNode`/`RetrieverItem`, lexical fallback, method constants
- `./ontology.rs` → `OntologyMatcher`, `OntologyConfig`, thresholds, consensus, candidate validation, fallback
- `./scoring.rs` → `LlmSemanticScorer`, `DeterministicScorer`
- `./fingerprint.rs` → `FingerprintExtractor`, `FingerprintConfig`, `SemanticProfile` fallback
- `./judge.rs` → `LlmJudgeMetric`, `DEFAULT_JUDGE_NAME`, `DEFAULT_RUBRIC`, repair attempt, static rubric
- `./foundation.rs` → `FoundationCompletionModel` adapter over `openjiuwen_foundation::llm::Model`
- `./identity.rs` → `ModelIdentity`, canonical JSON, metadata/endpoint/graph-identity digests, `sanitize_metadata`
- `./cache.rs` → `MatchCache`, `CachedMatch`, `CACHE_SCHEMA_VERSION`, `cache_key`, and the cached-payload helpers (`candidate_identity`, `covers`, `decode_cached_matches`)
