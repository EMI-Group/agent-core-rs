# openjiuwen-symphony / llm

## Intent

LLM-guided stages over one small completion seam, mapping to Python's LLM-backed `symphony` stages.
The seam is one prompt in, one reply out: each stage renders its own system/user prompt, calls `complete_json`, and parses the reply.
Every stage carries a documented deterministic fallback under `FallbackMode::Deterministic` (the default), while `FallbackMode::Error` opts into surfacing model failures.
The stages are progressive tree retrieval, `can_feed` ontology matching, semantic plan scoring, fingerprint extraction, and LLM judging.
This keeps the crate fully usable offline with no model configured, and a model is an optimisation rather than a requirement.

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
}
pub struct OntologyMatch { pub source_id: String, pub target_id: String, pub relation: String /*can_feed*/, pub confidence: f64, pub method: String, pub reasons: Vec<String>, pub accepted: bool }
```

`OntologyMatcher` validates `RelationCandidate`s: candidates with unknown capabilities are skipped, the batch is sorted by `(source, target)` and capped by `max_candidates`, the model picks `forward`/`reverse` and a confidence that is clamped to `[0, 1]`, and the orientation is always resolved to `source_id -> target_id` with `relation = can_feed`.

With `require_consensus` the batch is asked twice (forward and reversed) and only the intersection survives, keeping the lower confidence; `accepted` is `confidence >= threshold`, and matches sort by `(source_id, target_id)`.

Without a model, or with an unusable reply under `FallbackMode::Deterministic`, the matcher reports each candidate's own rule confidence with `method = "deterministic_fallback"`; `FallbackMode::Error` surfaces the failure as `Error::Llm`.

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

## Constraints

- One seam: stages depend on `CompletionModel`, never on a vendor SDK; `StaticCompletionModel` covers tests and `FoundationCompletionModel` adapts a foundation `Model`.
- `FallbackMode::Deterministic` must always produce a usable answer offline; only `FallbackMode::Error` may return `Error::Llm`.
- No network in default builds or tests; model calls happen only through an injected `CompletionModel`.
- Determinism: deduplication is first-wins, ordering is by id or `(source_id, target_id)`, clamping is explicit.
- No `unsafe`, no `println!`, no panics on user input; every public item is documented.
- Do not duplicate retrieval or orchestration types; re-export `CompletionModel`, `StaticCompletionModel`, and `DeterministicScorer`.
- Files stay around or under 1000 lines; `tree.rs` and `ontology.rs` are the largest and stay focused on their stage.

## Routing Table

- `./mod.rs` → `CompletionModel` seam re-exports, `ModelConfig`, `FallbackMode`, `render_prompt`, `complete_json`
- `./tree.rs` → `TreeRetriever`/`LlmTreeRetriever`, `RetrieverNode`/`RetrieverItem`, lexical fallback, method constants
- `./ontology.rs` → `OntologyMatcher`, `OntologyConfig`, thresholds, consensus, candidate validation, fallback
- `./scoring.rs` → `LlmSemanticScorer`, `DeterministicScorer`
- `./fingerprint.rs` → `FingerprintExtractor`, `FingerprintConfig`, `SemanticProfile` fallback
- `./judge.rs` → `LlmJudgeMetric`, `DEFAULT_JUDGE_NAME`, `DEFAULT_RUBRIC`, repair attempt, static rubric
- `./foundation.rs` → `FoundationCompletionModel` adapter over `openjiuwen_foundation::llm::Model`
