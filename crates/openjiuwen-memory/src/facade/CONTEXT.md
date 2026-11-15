# facade

## Intent

`LongTermMemory`: the public entry point that assembles the engine, storage, optional graph memory, and coding memory, and preserves the Python call-site vocabulary (`add_messages`, `search_user_mem`, `update_mem_by_id`, variables, scope configs).
Everything here is wiring and argument adaptation; ranking, dedupe, and persistence live in `engine`/`store`/`recall`.

## API Surface

```rust
pub struct LongTermMemory { /* engine: Arc<MemoryEngine>, graph: Option<Arc<GraphMemory>>, coding: Option<CodingMemory>, analyzer: Arc<MemoryAnalyzer> */ }

pub struct LongTermMemoryBuilder { /* config, stores?, embedding?, model?, analyzer?, named_models, named_embeddings */ }
impl LongTermMemoryBuilder {
    pub fn stores(self, MemoryStores) -> Self;
    pub fn embedding(self, Arc<dyn Embedding>) -> Self;
    pub fn model(self, Arc<dyn Model>) -> Self;
    pub fn model_named(self, name: impl Into<String>, Arc<dyn Model>) -> Self;
    pub fn embedding_named(self, name: impl Into<String>, Arc<dyn Embedding>) -> Self;
    pub fn analyzer(self, Arc<MemoryAnalyzer>) -> Self;
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
    pub fn scope_override_handles(&self) -> ScopeOverrideHandles;

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
    pub async fn clear_messages(&self, filter: &MemoryFilter, options: &CallOptions) -> Result<usize>;
    pub async fn get_message_page(&self, scope: &MemoryScope, page_size: usize, cursor: Option<&str>,
                                  filter: Option<&MemoryFilter>) -> Result<MessagePage>;
    pub async fn page_messages(&self, scope: &MemoryScope, page_size: usize,
                               cursor: Option<&str>) -> Result<MessagePage>;

    // --- variables ---
    pub async fn set_variable(&self, scope: &MemoryScope, name: &str, value: &str) -> Result<()>;
    pub async fn get_variables(&self, scope: &MemoryScope, names: Option<&[String]>) -> Result<BTreeMap<String, String>>;
    pub async fn delete_variables(&self, scope: &MemoryScope, names: &[String]) -> Result<usize>;
    pub async fn assign_variable(&self, scope: &MemoryScope, name: &str, value: &str,
                                 options: &AddOptions) -> Result<VariableWrite>;
    pub async fn variable_records(&self, scope: &MemoryScope) -> Result<Vec<MemoryRecord>>;
    pub async fn remember_variables(&self, scope: &MemoryScope, facts: &[ExtractedFact],
                                    options: &AddOptions) -> Result<VariableAssignmentReport>;
    pub async fn remember_with_assignments(&self, scope: &MemoryScope, input: &ExtractionInput,
                                           options: &AddOptions) -> Result<VariableAssignmentReport>;

    // --- scope config ---
    pub async fn set_scope_config(&self, scope_id: &str, config: MemoryScopeConfig) -> Result<bool>;
    pub async fn get_scope_config(&self, scope_id: &str) -> Result<Option<MemoryScopeConfig>>;
    pub async fn delete_scope_config(&self, scope_id: &str) -> Result<bool>;

    // --- per-scope analyzer overrides (`overrides.rs`) ---
    pub async fn set_scope_overrides(&self, scope_id: &str, spec: &ScopeAnalyzerOverrideSpec) -> Result<()>;
    pub async fn get_scope_overrides(&self, scope_id: &str) -> Result<Option<ScopeAnalyzerOverrideSpec>>;
    pub async fn delete_scope_overrides(&self, scope_id: &str) -> Result<bool>;
    pub async fn resolve_scope_overrides(&self, scope: &MemoryScope) -> Result<ScopeAnalyzerOverrides>;

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

    // --- subsystem wiring (`integrations.rs`) ---
    pub fn analyzer(&self) -> &Arc<MemoryAnalyzer>;
    pub async fn remember(&self, scope: &MemoryScope, input: &ExtractionInput, options: &AddOptions) -> Result<RememberReport>;
    pub async fn consolidate(&self, scope: &MemoryScope, incoming: &NewMemory, options: &CallOptions) -> Result<ConsolidationPlan>;
    pub fn provider(&self) -> Arc<dyn MemoryProvider>;
    pub fn team_memory(&self) -> TeamMemory;
    #[cfg(feature = "sqlite")] pub async fn team_memory_sqlite(&self, path: impl AsRef<Path>) -> Result<TeamMemory>;
    pub fn dreaming_in_memory(&self) -> Result<DreamPlanner>;
    #[cfg(feature = "sqlite")] pub async fn dreaming_sqlite(&self, path: impl AsRef<Path>) -> Result<DreamPlanner>;
    #[cfg(feature = "sqlite")] pub async fn sqlite_migrated(path: impl AsRef<Path>, config: MemoryEngineConfig) -> Result<Arc<Self>>;
    pub async fn open_local_coding(&self, root: impl AsRef<Path>) -> Result<LocalCodingMemory>;
}
```

Per-scope analyzer overrides (`overrides`, `handles`):
- `ScopeAnalyzerOverrideSpec` — the serializable document (`extractor`, `classifier`, `min_importance`, `max_facts`, `default_importance`, `stages`, `model`, `embedding`), `snake_case`, `#[serde(default)]`, tolerant of unknown fields; `new`, the `with_*` builders, `is_empty`, `validate`, and `resolve(&ScopeOverrideHandles) -> Result<ScopeAnalyzerOverrides>`.
- `ScopeOverrideHandles` (`handles.rs`) — the named registry a spec resolves against: the engine's own model/embedding under their `model_name()`, plus every handle registered through `LongTermMemoryBuilder::model_named`/`embedding_named`; `Debug` reports names only, never credentials.
- `ScopeAnalyzerOverrideSpec::resolve` returns the runtime `ScopeAnalyzerOverrides` (handles included) or a typed `Unsupported` that lists every registered name, so a swap is never a silent fallback to the engine handle.
- `RememberReport` gains `gated: Option<String>`: why the proactive stage withheld the write under a scope override (it stays `None` in the unconfigured path).

## Wiring Rules

- `in_memory` / `local_file` / `sqlite` build `MemoryStores`, then call `build`.
- `builder().build()` defaults to `MemoryStores::in_memory()` when no stores are supplied; `config.validate()` runs first.
- `graph` is `Some(GraphMemory)` only when `config.graph.enabled` and the stores carry a `GraphStore`; the graph shares the engine's `Embedding` and `Model`.
- `coding` is `Some(CodingMemory)` when `config.coding.enabled`; it wraps the same `Arc<MemoryEngine>`.
- `search_all` = `engine.search_with_embedding(...)` with the scope's resolved override embedder plus, when graph is enabled, `graph.search(scope, query, GraphSearchOptions { top_k: options.top_k, threshold: options.threshold, max_depth: config.graph.max_depth, max_neighbors: config.graph.max_neighbors, ..Default::default() })`. Failure of graph recall does **not** fail memory recall: the failure is logged and reported on `CombinedRecall::graph_error` (`is_partial()`) with `entities` empty (rail parity: memory degrades, it never breaks the turn), while a cancelled graph recall propagates `Error::Cancelled`.
- `search_user_mem`/`search_user_history_summary` build a `MemoryScope` from `user_id`/`scope_id` and call `search` with `SearchOptions { top_k: num, threshold, .. }` (summary restricts `kinds` to `[Summary]`).
- `update_mem_by_id` maps to `update(..., MemoryUpdate::content(content), ...)` and discards the record (Python returns `None`).
- `get_user_mem_by_page`: `page_size`/`page_idx` are 1-based; returns the slice `[(page_idx-1)*page_size .. page_idx*page_size]` of the scoped listing, filtered by `kind` unless `kind == Unknown` (then all kinds); an out-of-range page returns an empty vector. `page_size == 0` is `Error::Validation`.
- `user_mem_total_num` counts scoped records for `user_id`/`scope_id` (all kinds; variables excluded).
- `delete_mem_by_user_id` clears records for the user in that scope; `delete_mem_by_scope` clears all users in that scope (returns the record count). Both also delete embeddings via the engine. Messages and variables are **not** cleared by these aliases (documented deviation; use `clear` with a `MemoryFilter` and the variable/message methods).
- `add_messages`:
  1. Validate/truncate messages (`input_msg_max_len`), append them to message history (`store::MessageStore`) with now.
  2. Build candidates from message text. With a `Model`, run one extraction prompt requesting `{"facts":[{"content":"...","type":"user_profile|semantic_memory|episodic_memory|summary","importance":0.0}],"variables":{"name":"value"}}`, parse tolerantly, retry `model_retries`; on failure fall back to heuristics with a warning and an `AddMemResult::degraded` reason naming the failure.
  3. A payload with no usable content is a failure, not a success: no fact and no variable counts as a parse failure so the heuristics still run, and a fact whose `type` is unknown (or `variable`) is stored as a `SemanticMemory` with a warning instead of being dropped by the kind filter.
  4. Heuristic fallback: every user/assistant message with `trim().len() >= min_message_len` becomes an `EpisodicMemory`; a concatenated last-N summary becomes a `Summary`; variables are not inferred.
  5. Filter by `AgentMemoryConfig::kind_enabled`, `forbidden_variables`, and `max_facts`; add records via the engine (dedupe applies); set inferred variables via `assign_variable` for `mem_variables` only, so each one gets the KV entry *and* its `Variable` record projection; a name the variable write path still rejects (a case/whitespace variant of a forbidden name) is skipped with a warning, never failing the turn.
  6. Return `AddMemResult` grouped by kind (records actually added/merged, `coding` normally empty) with the engine's `skipped`/`deleted` counts forwarded, so an all-duplicate batch is distinguishable from an empty extraction; a caller-supplied `DedupePolicy::Replace` still reports its replacement in `merged` and the removed row in `deleted`, matching the engine's contract.
  A model failure never fails `add_messages`; it degrades to the heuristic path. Cancellation returns `Error::Cancelled` and leaves already-added records.
- `get_recent_messages` uses the message store (scope's `session_id`, else all sessions) with `num` clamped to `message_history_limit` and returns chronological `BaseMessage`s; `Role` maps from the stored role name (unknown roles default to `User` with a warning).
- `get_message_page(scope, page_size, cursor, filter)` is the forward keyset walk over the same history: `page_size == 0` returns an empty `MessagePage`, an oversized page returns everything (no `message_history_limit` clamp — that limit belongs to the newest-N window), and `filter` is `AND`ed with the scope clauses by the store, so it can never widen the page past its scope. A garbage or tampered cursor is `Error::Validation`. `page_messages` is the same call without a filter.
- `clear_messages` deletes conversation history rows matching the filter (records and variables are untouched); cancellation is checked before the delete.
- Variables: names are validated (non-empty, no `/`, no NUL). `get_variables(None)` reads the KV entries (all user-scope variables, not session variables; session variables are addressed by `scope.session_id`).
- `set_variable` keeps its signature and delegating shape: it is `assign_variable` with `AddOptions::default()`, so the write is the KV entry **plus** the `MemoryType::Variable` record projection built by `MemoryEngine::write_variable`; a name in `forbidden_variables` is `Error::Validation` naming the variable before any write. `assign_variable` is the options-aware (cancellable, timeout, `skip_embedding`) form returning `VariableWrite { record_id, created }`.
- `delete_variables` validates every name first, then removes both representations per name through `MemoryEngine::delete_variable_everywhere` (KV entry plus projection row and vector) and counts the names that existed. `variable_records` lists the projections ascending by `(timestamp, id)`; the same `(user, scope, session, name)` always maps to one deterministic id, so a rewrite updates in place and never enters the text-similarity dedupe scan.
- `remember_variables(scope, facts, options)` is the opt-in variable path for analyzer facts: a fact tagged `variable` whose metadata carries `variable` (name) and `value` writes a `Variable` projection (`VariableAssignmentReport::variables`), every other fact is stored as a record exactly like `remember` (`records_added`/`records_merged`/`records_skipped`). `remember` itself still stores such a fact as a `SemanticMemory` record, which keeps `tests/analyzer_wiring.rs` behaviour unchanged.
- `remember_with_assignments(scope, input, options)` is the end-to-end variant: `analyze_for_write` (exactly `remember`'s staged pipeline and override gate) followed by the same write split as `remember_variables`, reporting `extracted`/`variables`/`records_*` plus the pipeline's `degraded` and `gated` reasons on `VariableAssignmentReport`; `write_assignment_facts` folds the first `AddOutcome::degraded` of its record writes into `VariableAssignmentReport::degraded` through the same `note_write_degraded` helper.
- `analyze_for_write(scope, input, options)` is the shared `remember` analysis pass: it returns the facts to write plus a `RememberReport` carrying only `extracted`/`degraded`/`gated`, so `remember` and `remember_with_assignments` cannot drift; a gated batch returns no facts.
- `record_input(fact)` in `mod.rs` is the facade's single `ExtractedFact` → `NewMemory` conversion (tags under `TAGS_KEY = "tags"`, an existing key wins); `remember`, `remember_variables` and `remember_with_assignments` all use it.
- `write_assignment_facts` is the shared write loop of both variable paths: cancellation between facts, one engine call per fact, a rejected variable name failing the call with the fact index and name.
- Scope config: `scope_id` validated like `MemoryScope`; JSON stored under `memory_scope_config/{scope_id}`.
- Scope analyzer overrides (`overrides.rs`, `handles.rs`): the serializable `ScopeAnalyzerOverrideSpec` (`extractor`, `classifier`, `min_importance`, `max_facts`, `default_importance`, `stages`, `model`, `embedding`) is stored as JSON under `memory_scope_overrides/{scope_id}`; `set` validates the document's shape only, `get` rejects a mis-shaped payload with `Error::Validation` naming the scope, and `resolve_scope_overrides` resolves it against the builder's `ScopeOverrideHandles`. The registry holds the engine model/embedding under their `model_name()` plus every `model_named`/`embedding_named` handle, so a scope can swap its model *and* its embedder; an unregistered name is `Error::Unsupported` listing the registered names, never a fallback to the engine handle, and `build` rejects a blank registered name with `Error::Config`. `delete_scope_overrides` is separate: `delete_scope_config` keeps its signature and clears `memory_scope_config/{scope_id}` only.
- A resolved override handle is applied per call: the model reaches the analyzer/pipeline stages, and the embedding reaches `engine.add`/`add_many`/`update`/`search`/`write_variable` through their `*_with_embedding` variants (writes, dedupe scan, updates, variable projections, recall), with the engine caching dimensions per `model_name` so two embedders coexist. `engine()`, `coding()`, `provider()` and `graph()` bypass the override namespace and keep using the engine's own handles.
- `analyzer` defaults to `MemoryAnalyzer::from_engine(config, model)`: rules unless `config.extraction.extractor` asks for the LLM, and an LLM request without an attached model fails `build` with the analyzer's typed `Unsupported`. `builder().analyzer(...)` replaces it (tests use this to inject a scripted extractor).
- `remember(scope, input, options)` resolves the scope overrides and runs `StagedAnalyzer::from_engine(config, engine.model())` with `.with_overrides(resolved)` and `.with_dedupe(config.dedupe)`, so extraction, reasoning, and the proactive decision all see the effective (base + scope) configuration. It writes each fact through `engine.add` (tags are preserved under the crate's `tags` metadata key because `to_new_memory` drops them), checks cancellation before each fact, counts `added`/`merged`/`skipped` from each `AddOutcome`, and surfaces the staged `degraded` reason on `RememberReport::degraded`, folding the first `AddOutcome::degraded` a fact write reports into the same field through `note_write_degraded` (an analysis reason keeps its place and the write reason is appended, so both halves are named). A rejected write fails the call with the fact index/content and the engine message — never a silent drop.
- `remember` keeps two documented stability rules: (a) a builder-injected analyzer keeps owning extraction while the scope carries no overrides (the staged pipeline cannot reach a custom `FactExtractor`), and (b) the proactive decision is advisory (every fact is written) unless overrides are in effect — then `should_write()` gates the batch, `writable_facts()` selects it, and `RememberReport::gated` names the decision's reason with nothing written.
- `consolidate(scope, incoming, options)` validates the content, resolves the scope overrides, recalls candidates at `dedupe.candidate_top_k` with threshold `0.0` (a keyword-only blended score never clears a positive floor), converts `incoming` into one `ExtractedFact`, and delegates to `StagedAnalyzer::consolidate` (same engine config, scope overrides, `config.dedupe`). It returns a plan only; nothing is applied, and the plan is dedupe-policy work — the classifier is never consulted.
- `provider()` returns an `EngineProvider` over the same engine — a stable SPI view, capabilities reflecting the attached embedding model.
- `team_memory` / `team_memory_sqlite` build a team facade over its own store; `dreaming_in_memory` / `dreaming_sqlite` build a `DreamPlanner` with `NoopConsolidator` and `DreamConfig::default()` (they return `Result` because construction validates the config).
- Nothing here starts a background task, timer, watcher, or purge: the caller owns `run_due`/`run_once`/`purge`, and no cross-process lease exists.
- `sqlite_migrated(path, config)` runs `MemoryStores::sqlite_migrated` (migration first, then the stores) and fails with the path plus the migration reason; never open a SQLite facade through `sqlite` when the migration gate must hold.
- `open_local_coding(root)` bridges to `LocalCodingMemory` with the engine's `coding` config; the local stack keeps Markdown as its own source of truth (FTS5 index, lock, watcher) and is not mirrored into the engine record path — pick one path per workspace.

## Constraints

- The facade is thin: no ranking or storage logic beyond mapping calls.
- `Arc<LongTermMemory>` is cloneable and `Send + Sync`; all methods take `&self`.
- Every method validates its scope before I/O.
- Errors follow the crate strategy; no panics.
- Python parity methods keep their positional order and defaults; Rust-idiomatic methods keep `&CallOptions`/`&AddOptions` explicit.

## Routing Table

- `./mod.rs` → `LongTermMemory`, builder, constructors, accessors, re-exports, and the shared `record_input`/`TAGS_KEY` conversion
- `./records.rs` → CRUD/listing/stats methods and Python parity aliases
- `./recall.rs` → `search`, `recall`, `search_all`
- `./messages.rs` → `add_messages`, `get_recent_messages`, `get_message_page`/`page_messages`, `clear_messages`, extraction/fallback
- `./variables.rs` → variable methods (`set`/`assign`/`get`/`delete`, `variable_records`, `remember_variables`, `remember_with_assignments`), `VariableAssignmentReport` (`extracted`, `variables`, `records_added`, `records_merged`, `records_skipped`, `degraded`, `gated`), and scope-config methods
- `./overrides.rs` → `ScopeAnalyzerOverrideSpec`/`ScopeOverrideHandles`, persisted per-scope overrides, resolution, and the value the staged pipeline consumes
- `./handles.rs` → `ScopeOverrideHandles`: the named model/embedding registry (engine handles under `model_name()` plus `model_named`/`embedding_named`), name-only `Debug`
- `./integrations.rs` → analyzer/remember/consolidate (staged pipeline + scope overrides + the shared `analyze_for_write`), provider SPI, team memory, dreaming, SQLite migration constructor, local coding bridge

## Test Strategy

- Thin facade tests cover constructor defaults, builder injection, Python aliases, page math, and `search_all` degradation when graph recall fails (memory hits kept, `graph_error` set, `is_partial()` true).
- `handles.rs` unit tests pin the registry: engine handles registered under their `model_name()` and becoming the default, caller-chosen names resolving without displacing the default, a duplicate name replacing the earlier handle, a blank name registering nothing, and `Debug` printing names only; `mod.rs` covers `build` rejecting a blank registered name.
- `variables.rs` adds KV-plus-projection, in-place rewrite, forbidden-name, delete-both, cancellation and `remember_variables` coverage, plus `remember_with_assignments` end-to-end coverage (rule assignment, in-place rewrite, forbidden name keeping earlier facts, LLM-fallback `degraded`, pre-cancelled and mid-loop cancellation via a cancelling `Embedding`) and a direct `record_input` tag assertion; `messages.rs` adds the page walk, tampered cursor, empty/oversized page, filter-composition and projected-variable coverage.
- `overrides.rs` unit tests pin the spec's JSON round trip (unknown/absent fields), empty detection, scalar resolution, the typed `Unsupported` errors for unresolvable names, and persistence across in-memory, local-file, and SQLite facades (including a malformed stored payload).
- `integrations.rs` unit tests pin the override→pipeline wiring: `max_facts` changing what `remember` writes, an empty spec matching the unconfigured run, the proactive gate reporting on `gated`, a degraded model fallback under an `llm` override, cancellation writing nothing, and `consolidate` resolving a scope classifier while still returning a plan only.
- The heavy behavior suites live in `tests/` (`crud.rs`, `scoping.rs`, `recall.rs`, `dedupe.rs`, `graph.rs`, `lite.rs`, `lite_local.rs`, `sqlite.rs`, `analyzer_wiring.rs`, `scope_overrides.rs`, `extraction_resilience.rs`, `external_provider.rs`, `team_memory.rs`, `dreaming.rs`, `migration.rs`).
