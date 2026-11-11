# engine

## Intent

The write/read orchestration at the heart of long-term memory: validation, embedding, dedupe/merge, persistence, candidate generation, ranking, and statistics.
This is the Rust replacement for Python's `LongTermMemory` internals (`WriteManager` + `SearchManager` + `Generator`); the public facade in `../facade/` only wires it to message/variable/graph concerns.

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
    pub async fn add_many(&self, scope: &MemoryScope, memories: Vec<NewMemory>, options: &AddOptions) -> Result<AddOutcome>;
    pub async fn get(&self, scope: &MemoryScope, id: &str) -> Result<Option<MemoryRecord>>;
    pub async fn update(&self, scope: &MemoryScope, id: &str, patch: MemoryUpdate, options: &CallOptions) -> Result<MemoryRecord>;
    pub async fn delete(&self, scope: &MemoryScope, id: &str) -> Result<bool>;
    pub async fn clear(&self, filter: &MemoryFilter, options: &CallOptions) -> Result<usize>;
    pub async fn list(&self, filter: &MemoryFilter) -> Result<Vec<MemoryRecord>>;
    pub async fn search(&self, scope: &MemoryScope, query: &str, options: &SearchOptions) -> Result<Vec<RecallHit>>;
    pub async fn stats(&self, filter: &MemoryFilter) -> Result<MemoryStats>;
}
```

## Write Path

1. `scope.validate()`; `memory.validate()`; reject `MemoryType::Variable` (`Error::Validation`).
2. Acquire the per-scope async write lock keyed by `(user_id, scope_id)`; release after the operation. `parking_lot` only guards the lock map and is dropped before `.await`.
3. Build a `MemoryRecord` (new id via `record::new_memory_id`, timestamps now, `importance = memory.importance.unwrap_or(config.importance_default)`, scope copied, metadata/source/timestamp from the input).
4. Dedupe candidates (unless `options.dedupe == Some(DedupePolicy::Add)` or `config.dedupe.enabled == false`): embed the content (if embedding available and not `skip_embedding`) and run `search` over the same scope with `top_k = config.dedupe.candidate_top_k`, `mode = Hybrid`, threshold `0`. Compare with `recall::dedupe::find_duplicate`.
   - `Merge` => persist `merge_records(existing, incoming, now)` (same key, new embedding), outcome `merged`.
   - `Skip` => no write, outcome `skipped` (id of the incoming record).
   - `Replace` => delete the existing record+embedding, write the incoming record under the existing id, outcome `merged`.
   - `Add` / no duplicate => write.
5. Persist the record (KV) first, then index the embedding; if indexing fails, delete the KV record and return the error (no half-indexed records).
6. Cancellation is checked before embedding and before each batch item; `CallOptions::run` wraps the embedding/model future. A cancelled batch returns `Error::Cancelled`; already written records stay.
7. `add_many` processes inputs sequentially under one scope lock and dedupes against both stored and earlier batch records; stop/return on the first error (previous writes remain committed). `AddOutcome` accumulates all outcomes.

## Read Path

1. Validate scope and options (`top_k > 0`; threshold in `0..=1`; `mode` compatible).
2. `kinds = options.kinds` or `config.searchable_kinds`; filter to `is_searchable`.
3. Build candidates:
   - `Semantic`/`Hybrid` with embedding: for each kind call `storage.search_embeddings(scope, kind, query_vec, candidate_k)` where `candidate_k = min(max(top_k * candidate_multiplier as usize, top_k), max_candidates)`. Fetch records by `(id)` via `storage.get_record` (the vector store has no payload). Missing records (stale index) are skipped with a warning.
   - `Keyword` or no embedding: `storage.list_records` for the scope+kinds, limited to `max_candidates` by timestamp desc; `similarity = None`. If `mode == Semantic` and no embedding => `Error::Unsupported("semantic search requires an embedding model")`.
4. Rank with `recall::pipeline::rank(candidates, query, now, &config.ranking)`.
5. Filter `score >= options.threshold`; take `top_k`; apply `options.token_budget.or(config.token_budget)` via `trim_to_token_budget`.
6. Update `last_access`/`access_count` for the returned records **best-effort** (write failures are logged at `warn`, never fail a search). Do not mutate `updated_at` on access.
7. `search` never includes `Variable`/`Unknown` records.

An empty query is `Error::Validation`.

## Update / Delete / Clear

- `update`: load the record (`NotFound` if absent), validate the patch, apply content/kind/importance/metadata merge/source, re-embed when content changed, `updated_at = now`, persist. `kind == Variable` is rejected.
- `delete`: load, delete KV + embedding, return `true` when the record existed.
- `clear(filter)`: list matching records (prefix scan), delete each record+embedding; returns the count. `filter.kinds` empty means every memory kind; variables/messages are cleared by the facade/store methods, not `clear`.
- `list(filter)`: storage listing (timestamp desc), no scoring.
- `stats(filter)`: count by kind (`BTreeMap<String, usize>` keyed by `MemoryType::as_str`), oldest/newest timestamp, and `total_tokens` from the engine token counter over record contents.

## Concurrency and Inputs

- One `tokio::sync::Mutex` per `(user_id, scope_id)`; the map is guarded by `parking_lot::Mutex<HashMap<(String,String), Arc<tokio::sync::Mutex<()>>>>` and cloned out before awaiting.
- Reads are lock-free (storage operations are internally synchronized). Two concurrent writes to one scope serialize; different scopes run in parallel.
- Embedding dimension is cached in a `tokio::sync::OnceCell<usize>` initialized from `Embedding::dimension()`; vector collection creation uses it.
- `MemoryEngine` is `Send + Sync + 'static` and used as `Arc<MemoryEngine>`.

## Constraints

- Errors: validation `Error::Validation`, missing update `Error::NotFound`, semantic-without-embedding `Error::Unsupported`, cancellation `Error::Cancelled`; embedding/storage errors propagate.
- Never hold a std/parking_lot lock across `.await`.
- No ranking formula duplicated here: call `recall` functions.
- A failed embedding never leaves a record committed; a failed dedupe search degrades to "no candidates" with a `warn` (dedupe is best-effort, Python parity).
- `add` of an empty/whitespace content is rejected; content is not trimmed (preserve caller text) but blank detection uses `trim()`.

## Routing Table

- `./mod.rs` → `MemoryEngine`, `MemoryEngineBuilder`, re-exports
- `./write.rs` → add/add_many/update/delete/clear and dedupe integration
- `./search.rs` → candidate generation, search, access-touch
- `./stats.rs` → statistics aggregation
- `./locks.rs` → per-scope write-lock registry
- `./embedding.rs` → embedding helper (dimension cache, cancellable embed calls)

## Test Strategy

- Module tests: duplicate policy matrix, failed-index rollback, blank content, same-scope lock serialization is covered indirectly by concurrent add tests in `tests/crud.rs`.
- Integration tests: `tests/crud.rs` (write/read/update/delete/clear/stats), `tests/recall.rs` (pipeline, thresholds, budgets, keyword fallback), `tests/dedupe.rs`, `tests/scoping.rs`.
- Deterministic: `HashEmbedding`, fixed `DateTime` values in assertions, `ScriptedModel` for extraction.
