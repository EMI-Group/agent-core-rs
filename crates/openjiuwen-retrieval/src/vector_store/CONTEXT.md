# vector_store — similarity search backends

## Intent

Port the Python `VectorStore` ABC shape and add the offline in-memory implementation (Python has no in-memory store) with metadata filtering, top-k ordering, batch upsert/delete/get, and optional JSON persistence.
Milvus/Chroma/PGVector/Elasticsearch stay behind a provider registry seam and return `Error::Unsupported`.

## API Surface

### `mod.rs`
- `VectorRecord { id, text, vector, metadata, doc_id, chunk_id }` (`doc_id`/`chunk_id` optional; snake_case JSON).
- `VectorQuery { vector, top_k (default 5), filter, score_threshold }` with builder methods (`new`, `with_top_k`, `with_filter`, `with_score_threshold`).
- `#[async_trait] pub trait VectorStore: Send + Sync` with `name`, `upsert`, `get`, `list`, `delete`, `delete_by_filter`, `search`, `len`, `is_empty`, `clear` (see crate CONTEXT for exact signatures).
- `pub type VectorStoreFactory = Arc<dyn Fn(&VectorStoreConfig) -> Result<Arc<dyn VectorStore>> + Send + Sync>`.
- `VectorStoreRegistry`: `with_builtin()` registers `"memory"` and `"in_memory"`; `register(provider, factory)`; `create(&VectorStoreConfig)`; unregistered provider → `Error::Unsupported` naming the provider (documents the external-store seam).
- Helpers used by search: `cosine_similarity(a, b) -> f32` (0.0 for zero vectors/dimension mismatch) and `normalize_cosine(cos) = (cos + 1.0) / 2`.

### `memory.rs`
- `InMemoryVectorStore { name, records: parking_lot::RwLock<IndexMap<String, VectorRecord>>, path: Option<PathBuf> }`.
- `new()`, `with_collection(name)`, `open(path)`/`open_with_collection(path, name)` (loads JSON when present and enables auto-persist), `persist()`; `Default` = `new()`.
- `upsert` replaces records with the same id, preserves first-insertion order for ordering/tie-breaks, returns the number of records supplied; `delete` returns how many existed; `delete_by_filter` evaluates `MetadataFilter` per record; `list(limit)` returns insertion order.
- `search`: iterate records, skip records whose vector length differs from the query, evaluate filter (with `doc_id`/`chunk_id` pseudo-fields when absent from metadata), compute `(cosine+1)/2`, apply `score_threshold` (`score >= threshold`), sort by score descending with stable insertion-order ties, truncate to `top_k`, and emit `SearchResult { id, text, score, metadata }` with `doc_id`/`chunk_id` injected into metadata when present.
- Persistence format `{"version":1,"name":"...","records":[...]}`; write a temp file then rename; snapshots are cloned before the async write so no lock is held across `.await`.

## Constraints

- All methods async and `Result`-returning; no panics for unknown ids or empty inputs.
- `top_k == 0` is valid and returns an empty vector (validation happens in higher layers where Python requires it).
- Locks are never held across `.await`; persistence failures propagate.
- Files ≤400 lines (`mod.rs`, `memory.rs`, and their test modules).

## Test Strategy

Upsert/get/len/clear; duplicate-id replacement; descending ordering and stable ties; top-k truncation; score threshold inclusivity; filters (equality, numeric operators, `$and`, pseudo `doc_id`); delete by id and filter; dimension mismatch skipped; zero vectors; persistence: write, reopen in a second instance, records intact, delete persists; registry lookup and unsupported provider.

## Routing Table

- `./mod.rs` → `VectorRecord`, `VectorQuery`, `VectorStore` trait, scoring helpers, provider registry
- `./memory.rs` → `InMemoryVectorStore`, cosine search, metadata filtering, JSON persistence
- `./tests.rs` → registry, record/query serialization, and scoring-helper tests
- `./memory/tests.rs` → in-memory search, filtering, deletion, and persistence tests
