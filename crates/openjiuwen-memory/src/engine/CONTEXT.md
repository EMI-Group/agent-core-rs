# engine

## Intent

The write/read orchestration at the heart of long-term memory: validation, embedding, dedupe/merge, persistence, candidate generation, ranking, and statistics.
It also owns the variable write path, which stores a named variable as both the existing KV entry and a `MemoryType::Variable` record projection.
This is the Rust replacement for Python's `LongTermMemory` internals (`WriteManager` + `SearchManager` + `Generator`) plus the write half of `manage/index/variable_manager.py`; the public facade in `../facade/` only wires it to message/variable/graph concerns.

## API Surface

```rust
pub struct MemoryEngine { /* config, storage, embedding, model, per-scope write locks, dimension cache */ }
pub struct MemoryEngineBuilder { /* config, stores, embedding, model, token_counter */ }

impl MemoryEngineBuilder {
    pub fn new(config: MemoryEngineConfig) -> Self;
    pub fn stores(self, stores: MemoryStores) -> Self;                  // default: MemoryStores::in_memory()
    pub fn embedding(self, embedding: Arc<dyn Embedding>) -> Self;      // optional
    pub fn model(self, model: Arc<dyn Model>) -> Self;                  // optional (add_messages extraction)
    pub fn token_counter(self, Arc<dyn TokenCounter + Send + Sync>) -> Self; // default ApproxTokenCounter
    pub fn build(self) -> Result<Arc<MemoryEngine>>;                    // validates config; creates storage
}

impl MemoryEngine {
    pub fn builder(config: MemoryEngineConfig) -> MemoryEngineBuilder;
    pub fn new(config: MemoryEngineConfig, stores: MemoryStores) -> Result<Arc<Self>>;
    pub fn config(&self) -> &MemoryEngineConfig;
    pub fn stores(&self) -> &MemoryStores;
    pub fn embedding(&self) -> Option<&Arc<dyn Embedding>>;
    pub fn model(&self) -> Option<&Arc<dyn Model>>;
    pub fn has_embedding(&self) -> bool;

    pub async fn add(&self, scope: &MemoryScope, memory: NewMemory, options: &AddOptions) -> Result<AddOutcome>;
    pub async fn add_with_embedding(&self, scope: &MemoryScope, memory: NewMemory, options: &AddOptions,
                                    embedding: Option<&Arc<dyn Embedding>>) -> Result<AddOutcome>;
    pub async fn add_many(&self, scope: &MemoryScope, memories: Vec<NewMemory>, options: &AddOptions) -> Result<AddOutcome>;
    pub async fn add_many_with_embedding(&self, scope: &MemoryScope, memories: Vec<NewMemory>, options: &AddOptions,
                                         embedding: Option<&Arc<dyn Embedding>>) -> Result<AddOutcome>;
    pub async fn get(&self, scope: &MemoryScope, id: &str) -> Result<Option<MemoryRecord>>;
    pub async fn update(&self, scope: &MemoryScope, id: &str, patch: MemoryUpdate, options: &CallOptions) -> Result<MemoryRecord>;
    pub async fn update_with_embedding(&self, scope: &MemoryScope, id: &str, patch: MemoryUpdate, options: &CallOptions,
                                       embedding: Option<&Arc<dyn Embedding>>) -> Result<MemoryRecord>;
    pub async fn delete(&self, scope: &MemoryScope, id: &str) -> Result<bool>;
    pub async fn clear(&self, filter: &MemoryFilter, options: &CallOptions) -> Result<usize>;
    pub async fn list(&self, filter: &MemoryFilter) -> Result<Vec<MemoryRecord>>;
    pub async fn search(&self, scope: &MemoryScope, query: &str, options: &SearchOptions) -> Result<Vec<RecallHit>>;
    pub async fn search_with_embedding(&self, scope: &MemoryScope, query: &str, options: &SearchOptions,
                                       embedding: Option<&Arc<dyn Embedding>>) -> Result<Vec<RecallHit>>;
    pub async fn stats(&self, filter: &MemoryFilter) -> Result<MemoryStats>;

    pub async fn write_variable(&self, scope: &MemoryScope, name: &str, value: &str, options: &AddOptions) -> Result<VariableWrite>;
    pub async fn write_variable_with_embedding(&self, scope: &MemoryScope, name: &str, value: &str,
                                               options: &AddOptions,
                                               embedding: Option<&Arc<dyn Embedding>>) -> Result<VariableWrite>;
    pub async fn delete_variable_everywhere(&self, scope: &MemoryScope, name: &str) -> Result<bool>;
    pub async fn variable_records(&self, scope: &MemoryScope) -> Result<Vec<MemoryRecord>>;
}

pub struct VariableWrite { pub record_id: String, pub created: bool }  // serde
```

The `*_with_embedding` variants carry the handle a per-scope override resolved to: `Some` uses it instead of the engine's embedder, `None` means the engine's own model, so a direct engine call keeps its documented behaviour while the facade can serve one scope with another embedder.

## Write Path

1. `scope.validate()`; `memory.validate()`; reject `MemoryType::Variable` (`Error::Validation`).
2. Acquire the per-scope async write lock keyed by `(user_id, scope_id)`; release after the operation. `parking_lot` only guards the lock map and is dropped before `.await`.
3. Build a `MemoryRecord` (new id via `record::new_memory_id`, timestamps now, `importance = memory.importance.unwrap_or(config.importance_default)`, scope copied, metadata/source/timestamp from the input).
4. Dedupe candidates (unless `options.dedupe == Some(DedupePolicy::Add)` or `config.dedupe.enabled == false`): embed the content (if embedding available and not `skip_embedding`) and run `search` over the same scope with `top_k = config.dedupe.candidate_top_k`, `mode = Hybrid`, threshold `0`. Compare with `recall::dedupe::find_duplicate`.
   - A scan that fails for any reason other than cancellation/timeout still writes the record and sets `AddOutcome::degraded` to the reason, so "the scope had no duplicates" is distinguishable from "the scope could not be scanned"; `Error::Cancelled`/`Error::Timeout` abort instead.
   - `Merge` => persist `merge_records(existing, incoming, now)` (same key, new embedding), outcome `merged`.
   - `Skip` => no write, outcome `skipped` (id of the incoming record).
   - `Replace` => delete the existing record+embedding, write the incoming record under the existing id, outcome `merged` for the replacement plus the removed id in `deleted`, so a caller sees the removal even though the id survives.
   - `Add` / no duplicate => write.
5. Persist the record (KV) first, then index the embedding; if indexing fails, delete the KV record and return the error (no half-indexed records).
6. Cancellation is checked before embedding and before each batch item; `CallOptions::run` wraps the embedding/model future. A cancelled batch returns `Error::Cancelled`; already written records stay.
7. `add_many` processes inputs sequentially under one scope lock and dedupes against both stored and earlier batch records; stop/return on the first error (previous writes remain committed). `AddOutcome` accumulates all outcomes.

## Variables

- `write_variable` writes one variable twice: the KV entry (`user_var/{user}/{scope}/{name}` or `session_var/{user}/{scope}/{session}/{name}`, chosen by `scope.session_id`) and a `MemoryType::Variable` record projection.
- Validation runs before any I/O: `scope.validate()`, a non-blank name, no `/`, no NUL, and no name listed in `config.forbidden_variables` (trimmed, ASCII case-insensitive). `options.call.check()` runs before every I/O step.
- The projection id is deterministic per `(user_id, scope_id, session_id, name)`: `var-` plus 16 lowercase hex digits of the same 64-bit FNV-1a hash family `record::new_memory_id` uses. `agent_id` is not hashed because it is not part of the KV layout either; the projection mirrors one storage key.
- Rewriting a name updates the projection in place: creation `timestamp`, importance, access counters, and extra metadata are preserved, while `content` (`"{name}={value}"`), `scope`, `updated_at`, and the `name`/`value` metadata fields are refreshed. `VariableWrite::created` is `true` only for the first write.
- The projection never enters the ordinary dedupe/merge scan and is never absorbed by it: `variable` is rejected in `searchable_kinds`, so `add` candidate scans never include it.
- The projection is embedded through `embed_text` when the engine has a model and `options.skip_embedding` is unset, and persisted through the shared `persist` helper (KV row first, then the vector; a failed index rolls the row back). No embedding means keyword-only, as everywhere else.
- A row at the deterministic id held by another kind is never overwritten silently: it is logged and treated as "no projection" for the write. An unreadable row at that id is logged and rewritten by the next write.
- `delete_variable_everywhere` removes the KV entry and the projection (row plus vector) and returns whether either existed; it validates the scope and name first.
- `variable_records` lists the scope's `Variable` records ascending by `(timestamp, id)` through the same prefix listing `list` uses; malformed rows are skipped with a warning by the storage scan.

## Read Path

1. Validate scope and options (`top_k > 0`; threshold in `0..=1`; `mode` compatible).
2. `kinds = options.kinds` or `config.searchable_kinds`; filter to `is_searchable`, with one exception: `MemoryType::Variable` is kept when a caller names it explicitly (the configuration rejects it in `searchable_kinds`, so no default or configured recall selects it).
3. Build candidates:
   - `Semantic`/`Hybrid` with embedding: for each kind call `storage.search_embeddings(scope, kind, query_vec, candidate_k)` where `candidate_k = min(max(top_k * candidate_multiplier as usize, top_k), max_candidates)`. Fetch records by `(id)` via `storage.get_record` (the vector store has no payload). Missing records (stale index) are skipped with a warning.
   - `Keyword` or no embedding: `storage.list_records` for the scope+kinds, limited to `max_candidates` by timestamp desc; `similarity = None`. The scan scores only the `ranking.max_candidates` newest records and a `tracing::warn!` on target `openjiuwen_memory::engine` names the cap, the pool size and how many records went unscored, so a truncated scope is never mistaken for a small one. If `mode == Semantic` and no embedding => `Error::Unsupported("semantic search requires an embedding model")`.
4. Rank with `recall::pipeline::rank(candidates, query, now, &config.ranking)`.
5. Filter `score >= options.threshold`; take `top_k`; apply `options.token_budget.or(config.token_budget)` via `trim_to_token_budget`.
6. Update `last_access`/`access_count` for the returned records **best-effort** (write failures are logged at `warn`, never fail a search). Do not mutate `updated_at` on access.
7. `search` returns `Variable` records only when the caller names the kind; `Unknown` is never returned.

An empty query is `Error::Validation`.

## Update / Delete / Clear

- `update`: load the record (`NotFound` if absent), validate the patch, apply content/kind/importance/metadata merge/source, re-embed when content changed, `updated_at = now`, persist. `kind == Variable` or `kind == Unknown` is rejected.
- A kind change moves the vector to the new kind's collection, and a failed re-index restores the previous state instead of leaving a half-migrated record: when the previous kind's vector was already deleted, the previous record row and its re-computed vector are put back (row first, so a failing re-embedding still leaves the previous JSON), and the returned error keeps its kind with context naming the record, the attempted and previous kinds, and whether the restore succeeded.
- `delete`: load, delete KV + embedding, return `true` when the record existed.
- `clear(filter)`: list matching records (prefix scan), delete each record+embedding; returns the count. `filter.kinds` empty means every memory kind except `Variable` projections; variables/messages are otherwise cleared by the facade/store methods, not `clear`.
- `list(filter)`: storage listing (timestamp desc), no scoring.
- `stats(filter)`: count by kind (`BTreeMap<String, usize>` keyed by `MemoryType::as_str`), oldest/newest timestamp, and `total_tokens` from the engine token counter over record contents.

## Variable Visibility

- `MemoryType::Variable` is Python's separate variable namespace, not ordinary memory: `list`, `stats` (including `by_kind`), and `clear` exclude the variable projections whenever `filter.kinds` does not name the kind.
- A `MemoryFilter` whose `kinds` explicitly contains `MemoryType::Variable` includes (and, for `clear`, deletes) them; an explicit kind list is the only way those three paths see a projection.
- `search` keeps its own rule — `Variable` records are returned only when `SearchOptions::kinds` names the kind, because `searchable_kinds` validation rejects it as a configured kind.
- `MemoryEngine::variable_records` always lists the scope's projections, and `write_variable`/`delete_variable_everywhere` keep owning both representations.
- The narrowing lives in one place: `includes_record(filter, record)` (with `includes_variables(filter)`) in `mod.rs`, applied after the unchanged prefix-scan listing by `list`/`clear`/`stats`.
- The rule is a behaviour rule, not an API change: no signature, key layout, or storage-layer behaviour changes, and `[Unknown]` keeps its existing meaning (no special casing in a filter; `get_user_mem_by_page` maps a `Unknown` kind argument to "every kind").

## Concurrency and Inputs

- One `tokio::sync::Mutex` per `(user_id, scope_id)`; the map is guarded by `parking_lot::Mutex<HashMap<(String,String), Arc<tokio::sync::Mutex<()>>>>` and cloned out before awaiting.
- Reads are lock-free (storage operations are internally synchronized). Two concurrent writes to one scope serialize; different scopes run in parallel.
- Embedding dimension is cached per embedding model name in a `parking_lot::Mutex<HashMap<String, usize>>` filled from `Embedding::dimension()`; vector collection creation and the post-embedding length check use the cache entry of the model that produced the vector, so several embedders of different dimensions coexist in one engine.
- `MemoryEngine` is `Send + Sync + 'static` and used as `Arc<MemoryEngine>`.

## Constraints

- Errors: validation `Error::Validation`, missing update `Error::NotFound`, semantic-without-embedding `Error::Unsupported`, cancellation `Error::Cancelled`; embedding/storage errors propagate.
- Never hold a std/parking_lot lock across `.await`.
- No ranking formula duplicated here: call `recall` functions.
- A failed embedding never leaves a record committed; a failed dedupe search degrades to "no candidates" with a `warn` and an `AddOutcome::degraded` reason (dedupe is best-effort, Python parity), while cancellation/timeout abort the write.
- `add` of an empty/whitespace content is rejected; content is not trimmed (preserve caller text) but blank detection uses `trim()`.
- A variable write is not atomic across its two representations: the KV entry is written after embedding and before the projection, so a failed projection write leaves the KV value (the Python-authoritative entry) in place and no half-indexed record.

## Routing Table

- `./mod.rs` → `MemoryEngine`, `MemoryEngineBuilder`, re-exports (`VariableWrite`), the `includes_record`/`includes_variables` visibility helpers
- `./write.rs` → add/add_many/update/delete/clear, dedupe integration, the shared `persist` helper
- `./variables.rs` → `write_variable`, `delete_variable_everywhere`, `variable_records`, deterministic variable ids
- `./search.rs` → candidate generation, search, access-touch
- `./stats.rs` → statistics aggregation
- `./locks.rs` → per-scope write-lock registry
- `./embedding.rs` → embedding helper (dimension cache, cancellable embed calls)

## Test Strategy

- Module tests: duplicate policy matrix, failed-index rollback, blank content, a failed candidate scan reported on `AddOutcome::degraded`, a kind-changing update restoring the previous record and vector, the keyword-scan cap warning, same-scope lock serialization is covered indirectly by concurrent add tests in `tests/crud.rs`.
- `variables.rs` tests: KV entry plus one projection, session namespace, in-place rewrite (creation time preserved, `created == false`), forbidden names, invalid names, deleting both representations plus the vector, ascending/scope-isolated listing, unfiltered `list`/`stats`/`clear` exclusion with explicit-kind inclusion, `clear` counts and KV survival, explicit-kind recall, no merge with ordinary records, cancellation, unreadable-row repair, id determinism, `VariableWrite` JSON, and SQLite persistence across a reopen (feature `sqlite`).
- `mod.rs` tests: the `includes_record`/`includes_variables` predicate matrix (an unfiltered filter hides `Variable` and keeps every other kind, an explicit `Variable` kind includes it).
- Integration tests: `tests/crud.rs` (write/read/update/delete/clear/stats), `tests/recall.rs` (pipeline, thresholds, budgets, keyword fallback), `tests/dedupe.rs`, `tests/scoping.rs`.
- Deterministic: `HashEmbedding`, fixed `DateTime` values in assertions, `ScriptedModel` for extraction.
