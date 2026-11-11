# facade

## Intent

`LongTermMemory`: the public entry point that assembles the engine, storage, optional graph memory, and coding memory, and preserves the Python call-site vocabulary (`add_messages`, `search_user_mem`, `update_mem_by_id`, variables, scope configs).
Everything here is wiring and argument adaptation; ranking, dedupe, and persistence live in `engine`/`store`/`recall`.

## API Surface

```rust
pub struct LongTermMemory { /* engine: Arc<MemoryEngine>, graph: Option<Arc<GraphMemory>>, coding: Option<CodingMemory> */ }

pub struct LongTermMemoryBuilder { /* config, stores?, embedding?, model? */ }
impl LongTermMemoryBuilder {
    pub fn stores(self, MemoryStores) -> Self;
    pub fn embedding(self, Arc<dyn Embedding>) -> Self;
    pub fn model(self, Arc<dyn Model>) -> Self;
    pub fn build(self) -> Result<Arc<LongTermMemory>>;
}

impl LongTermMemory {
    pub fn builder(config: MemoryEngineConfig) -> LongTermMemoryBuilder;
    pub async fn in_memory(config: MemoryEngineConfig) -> Result<Arc<Self>>;
    pub async fn local_file(config: MemoryEngineConfig, root: impl AsRef<Path>) -> Result<Arc<Self>>;
    #[cfg(feature = "sqlite")]
    pub async fn sqlite(config: MemoryEngineConfig, path: impl AsRef<Path>) -> Result<Arc<Self>>;
    pub async fn with_stores(config: MemoryEngineConfig, stores: MemoryStores) -> Result<Arc<Self>>;

    pub fn engine(&self) -> &Arc<MemoryEngine>;
    pub fn config(&self) -> &MemoryEngineConfig;
    pub fn graph(&self) -> Option<&Arc<GraphMemory>>;
    pub fn coding(&self) -> Option<&CodingMemory>;

    // --- record CRUD ---
    pub async fn add(&self, scope: &MemoryScope, memory: NewMemory, options: &AddOptions) -> Result<AddOutcome>;
    pub async fn add_many(&self, scope: &MemoryScope, memories: Vec<NewMemory>, options: &AddOptions) -> Result<AddOutcome>;
    pub async fn get(&self, scope: &MemoryScope, id: &str) -> Result<Option<MemoryRecord>>;
    pub async fn update(&self, scope: &MemoryScope, id: &str, patch: MemoryUpdate, options: &CallOptions) -> Result<MemoryRecord>;
    pub async fn delete(&self, scope: &MemoryScope, id: &str) -> Result<bool>;
    pub async fn clear(&self, filter: &MemoryFilter, options: &CallOptions) -> Result<usize>;
    pub async fn list(&self, scope: &MemoryScope, kinds: &[MemoryType]) -> Result<Vec<MemoryRecord>>;
    pub async fn stats(&self, scope: &MemoryScope, kinds: &[MemoryType]) -> Result<MemoryStats>;

    // --- recall ---
    pub async fn search(&self, scope: &MemoryScope, query: &str, options: &SearchOptions) -> Result<Vec<RecallHit>>;
    pub async fn recall(&self, scope: &MemoryScope, query: &str, options: &SearchOptions) -> Result<Vec<RecallHit>>; // alias
    pub async fn search_all(&self, scope: &MemoryScope, query: &str, options: &SearchOptions) -> Result<CombinedRecall>;

    // --- messages (Python `add_messages` / `get_recent_messages`) ---
    pub async fn add_messages(&self, scope: &MemoryScope, messages: &[BaseMessage],
                              agent_config: &AgentMemoryConfig, options: &AddOptions) -> Result<AddMemResult>;
    pub async fn get_recent_messages(&self, scope: &MemoryScope, num: usize) -> Result<Vec<BaseMessage>>;

    // --- variables ---
    pub async fn set_variable(&self, scope: &MemoryScope, name: &str, value: &str) -> Result<()>;
    pub async fn get_variables(&self, scope: &MemoryScope, names: Option<&[String]>) -> Result<BTreeMap<String, String>>;
    pub async fn delete_variables(&self, scope: &MemoryScope, names: &[String]) -> Result<usize>;

    // --- scope config ---
    pub async fn set_scope_config(&self, scope_id: &str, config: MemoryScopeConfig) -> Result<bool>;
    pub async fn get_scope_config(&self, scope_id: &str) -> Result<Option<MemoryScopeConfig>>;
    pub async fn delete_scope_config(&self, scope_id: &str) -> Result<bool>;

    // --- Python parity aliases ---
    pub async fn search_user_mem(&self, query: &str, num: usize, user_id: &str, scope_id: &str,
                                 threshold: f32) -> Result<Vec<RecallHit>>;
    pub async fn search_user_history_summary(&self, query: &str, num: usize, user_id: &str,
                                             scope_id: &str, threshold: f32) -> Result<Vec<RecallHit>>;
    pub async fn update_mem_by_id(&self, id: &str, content: &str, user_id: &str, scope_id: &str) -> Result<()>;
    pub async fn delete_mem_by_id(&self, id: &str, user_id: &str, scope_id: &str) -> Result<bool>;
    pub async fn delete_mem_by_user_id(&self, user_id: &str, scope_id: &str) -> Result<usize>;
    pub async fn delete_mem_by_scope(&self, scope_id: &str) -> Result<usize>;
    pub async fn user_mem_total_num(&self, user_id: &str, scope_id: &str) -> Result<usize>;
    pub async fn get_user_mem_by_page(&self, user_id: &str, scope_id: &str, page_size: usize,
                                      page_idx: usize, kind: MemoryType) -> Result<Vec<MemoryRecord>>;
}
```

## Wiring Rules

- `in_memory` / `local_file` / `sqlite` build `MemoryStores`, then call `build`.
- `builder().build()` defaults to `MemoryStores::in_memory()` when no stores are supplied; `config.validate()` runs first.
- `graph` is `Some(GraphMemory)` only when `config.graph.enabled` and the stores carry a `GraphStore`; the graph shares the engine's `Embedding` and `Model`.
- `coding` is `Some(CodingMemory)` when `config.coding.enabled`; it wraps the same `Arc<MemoryEngine>`.
- `search_all` = `engine.search(...)` plus, when graph is enabled, `graph.search(scope, query, GraphSearchOptions { top_k: options.top_k, threshold: options.threshold, max_depth: config.graph.max_depth, max_neighbors: config.graph.max_neighbors, ..Default::default() })`. Failure of graph recall does **not** fail memory recall; log a warning and return memories only (rail parity: memory degrades, it never breaks the turn).
- `search_user_mem`/`search_user_history_summary` build a `MemoryScope` from `user_id`/`scope_id` and call `search` with `SearchOptions { top_k: num, threshold, .. }` (summary restricts `kinds` to `[Summary]`).
- `update_mem_by_id` maps to `update(..., MemoryUpdate::content(content), ...)` and discards the record (Python returns `None`).
- `get_user_mem_by_page`: `page_size`/`page_idx` are 1-based; returns the slice `[(page_idx-1)*page_size .. page_idx*page_size]` of the scoped listing, filtered by `kind` unless `kind == Unknown` (then all kinds); an out-of-range page returns an empty vector. `page_size == 0` is `Error::Validation`.
- `user_mem_total_num` counts scoped records for `user_id`/`scope_id` (all kinds; variables excluded).
- `delete_mem_by_user_id` clears records for the user in that scope; `delete_mem_by_scope` clears all users in that scope (returns the record count). Both also delete embeddings via the engine. Messages and variables are **not** cleared by these aliases (documented deviation; use `clear` with a `MemoryFilter` and the variable/message methods).
- `add_messages`:
  1. Validate/truncate messages (`input_msg_max_len`), append them to message history (`store::MessageStore`) with now.
  2. Build candidates from message text. With a `Model`, run one extraction prompt requesting `{"facts":[{"content":"...","type":"user_profile|semantic_memory|episodic_memory|summary","importance":0.0}],"variables":{"name":"value"}}`, parse tolerantly, retry `model_retries`; on failure fall back to heuristics with a warning.
  3. Heuristic fallback: every user/assistant message with `trim().len() >= min_message_len` becomes an `EpisodicMemory`; a concatenated last-N summary becomes a `Summary`; variables are not inferred.
  4. Filter by `AgentMemoryConfig::kind_enabled`, `forbidden_variables`, and `max_facts`; add records via the engine (dedupe applies); set inferred variables via `set_variable` for `mem_variables` only.
  5. Return `AddMemResult` grouped by kind (records actually added/merged, `coding` normally empty).
  A model failure never fails `add_messages`; it degrades to the heuristic path. Cancellation returns `Error::Cancelled` and leaves already-added records.
- `get_recent_messages` uses the message store (scope's `session_id`, else all sessions) with `num` clamped to `message_history_limit` and returns chronological `BaseMessage`s; `Role` maps from the stored role name (unknown roles default to `User` with a warning).
- Variables: names are validated (non-empty, no `/`, no NUL). `get_variables(None)` lists all user-scope variables (not session variables); session variables are addressed by `scope.session_id`.
- Scope config: `scope_id` validated like `MemoryScope`; JSON stored under `memory_scope_config/{scope_id}`.

## Constraints

- The facade is thin: no ranking or storage logic beyond mapping calls.
- `Arc<LongTermMemory>` is cloneable and `Send + Sync`; all methods take `&self`.
- Every method validates its scope before I/O.
- Errors follow the crate strategy; no panics.
- Python parity methods keep their positional order and defaults; Rust-idiomatic methods keep `&CallOptions`/`&AddOptions` explicit.

## Routing Table

- `./mod.rs` → `LongTermMemory`, builder, constructors, accessors, re-exports
- `./records.rs` → CRUD/listing/stats methods and Python parity aliases
- `./recall.rs` → `search`, `recall`, `search_all`
- `./messages.rs` → `add_messages`, `get_recent_messages`, extraction/fallback
- `./variables.rs` → variable and scope-config methods

## Test Strategy

- Thin facade tests cover constructor defaults, builder injection, Python aliases, page math, and `search_all` degradation when graph recall fails.
- The heavy behavior suites live in `tests/` (`crud.rs`, `scoping.rs`, `recall.rs`, `dedupe.rs`, `graph.rs`, `lite.rs`, `sqlite.rs`).
