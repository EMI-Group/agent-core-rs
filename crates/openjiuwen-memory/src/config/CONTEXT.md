# config

## Intent

Serializable configuration for the memory engine and its optional sub-systems.
Mirrors the Python `core/memory/config/config.py` triples (`MemoryEngineConfig`, `MemoryScopeConfig`, `AgentMemoryConfig`) and adds the Rust-only ranking, dedupe, extraction, graph, coding, and storage knobs the port needs.
All types are `serde`-serializable, `snake_case`, tolerant of unknown/missing fields, and validate their invariants with `Error::Config`.

## API Surface

```rust
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct MemoryEngineConfig {
    pub default_top_k: usize,                 // 5
    pub default_threshold: f32,               // 0.3
    pub token_budget: Option<usize>,          // None
    pub searchable_kinds: Vec<MemoryType>,    // [UserProfile, SemanticMemory, EpisodicMemory, Summary, Coding]
    pub ranking: RankingConfig,
    pub dedupe: DedupeConfig,
    pub extraction: ExtractionConfig,
    pub graph: GraphMemoryConfig,
    pub coding: CodingMemoryConfig,
    pub forbidden_variables: Vec<String>,     // Python's comma-split string, as a list
    pub input_msg_max_len: usize,             // 8192
    pub single_turn_history_summary_max_token: usize, // 128
    pub message_history_limit: usize,         // 10 (Python get_recent_messages default)
    pub importance_default: f32,              // 0.5
}
impl Default for MemoryEngineConfig;
impl MemoryEngineConfig { pub fn validate(&self) -> Result<()>; }

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct RankingConfig {
    pub similarity_weight: f32,        // 0.70
    pub keyword_weight: f32,           // 0.30
    pub semantic_weight: f32,          // 0.85
    pub importance_weight: f32,        // 0.10
    pub recency_weight: f32,           // 0.05
    pub recency_half_life_secs: u64,   // 2_592_000 (30 days)
    pub keyword_min_score: f32,        // 0.10
    pub candidate_multiplier: f32,     // 2.0
    pub max_candidates: usize,         // 200
}
impl Default for RankingConfig;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum DedupePolicy { #[default] Merge, Skip, Replace, Add }

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct DedupeConfig {
    pub enabled: bool,                 // true
    pub policy: DedupePolicy,          // Merge
    pub similarity_threshold: f32,     // 0.92 (normalized cosine)
    pub jaccard_threshold: f32,        // 0.75
    pub candidate_top_k: usize,        // 5
    pub min_content_len: usize,        // 4 (text shorter than this is only exact-deduped)
}
impl Default for DedupeConfig;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ExtractionConfig {
    pub enabled: bool,                 // true (add_messages extraction)
    pub max_facts: usize,              // 8
    pub model_retries: u32,            // 3
    pub use_history_messages: usize,   // 2
    pub min_message_len: usize,        // 8
}
impl Default for ExtractionConfig;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct GraphMemoryConfig {
    pub enabled: bool,                 // false
    pub max_depth: usize,              // 1 (0 = seeds only)
    pub max_neighbors: usize,          // 8
    pub min_score: f32,                // 0.3
    pub extract_relations: bool,       // true
    pub max_entities: usize,           // 16
}
impl Default for GraphMemoryConfig;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct CodingMemoryConfig {
    pub enabled: bool,                 // true
    pub max_index_lines: usize,        // 200 (Python MAX_INDEX_LINES)
    pub fact_types: Vec<String>,       // ["user","feedback","project","reference"]
    pub default_fact_type: String,     // "project"
}
impl Default for CodingMemoryConfig;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct AgentMemoryConfig {
    pub enable_long_term_mem: bool,    // true
    pub enable_user_profile: bool,     // true
    pub enable_semantic_memory: bool,  // true
    pub enable_episodic_memory: bool,  // true
    pub enable_summary_memory: bool,   // true
    pub mem_variables: Vec<String>,    // []
}
impl Default for AgentMemoryConfig;
impl AgentMemoryConfig { pub fn kind_enabled(&self, kind: MemoryType) -> bool; }

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct MemoryScopeConfig {
    pub user_profile_definition: Option<String>,
    pub semantic_memory_definition: Option<String>,
    pub episodic_memory_definition: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum StorageBackend {
    InMemory,
    LocalFile { root: PathBuf },
    Sqlite { path: PathBuf },
}
impl Default for StorageBackend { /* InMemory */ }
```

## Validation Rules (`MemoryEngineConfig::validate`)

- `default_top_k > 0`; `0.0 <= default_threshold <= 1.0`; `0.0 <= importance_default <= 1.0`.
- `searchable_kinds` non-empty and contains no `Variable`; duplicates rejected.
- `ranking`: every weight finite and `>= 0`; `similarity_weight + keyword_weight > 0`; `semantic_weight + importance_weight + recency_weight > 0`; `recency_half_life_secs > 0`; `max_candidates > 0`; `keyword_min_score`/`threshold` in `0..=1`.
- `dedupe`: thresholds in `0..=1`, `candidate_top_k > 0`.
- `extraction`: `max_facts > 0`, `min_message_len > 0`.
- `graph`: `max_depth <= 4`, `max_neighbors > 0`, `min_score` in `0..=1`.
- `coding`: `max_index_lines > 0`, `default_fact_type` non-empty.
- `input_msg_max_len > 0`, `single_turn_history_summary_max_token > 0`, `message_history_limit > 0`.
- Errors are `Error::Config` with the field name and offending value.

Defaults mirror Python where the field exists (`top_k=5`, `threshold=0.3`, `input_msg_max_len=8192`, `summary_max_token=128`, `get_recent_messages(num=10)`, lite `max_index_lines=200`).
Rust-only defaults are the ranking/dedupe/graph/coding knobs documented above; changing a default requires updating the formula in `recall/CONTEXT.md`.

## Constraints

- No runtime state: no `Arc`, no `CancellationToken`, no clients. Embeddings/models/stores are injected at engine construction, not held in config.
- Every type has `Default`, except `StorageBackend` which defaults explicitly.
- JSON round-trip must preserve all fields; unknown fields are ignored.
- `AgentMemoryConfig::kind_enabled` returns false for `Variable`/`Unknown` and maps each `MemoryType` to its flag.

## Routing Table

- `./mod.rs` → all config types, re-exports
- `./engine.rs` → `MemoryEngineConfig`, defaults, validation
- `./ranking.rs` → `RankingConfig`
- `./dedupe.rs` → `DedupeConfig`, `DedupePolicy`
- `./extraction.rs` → `ExtractionConfig`, `AgentMemoryConfig`, `MemoryScopeConfig`
- `./graph.rs` → `GraphMemoryConfig`
- `./coding.rs` → `CodingMemoryConfig`
- `./storage.rs` → `StorageBackend`

## Test Strategy

- Unit tests assert every default value, JSON round-trip, unknown-field tolerance, and each validation failure (one test per rule).
- A test pins the documented weight sums (`0.70 + 0.30`, `0.85 + 0.10 + 0.05`) so a default drift breaks visibly.
