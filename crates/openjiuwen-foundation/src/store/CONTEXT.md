# store — kv, document, vector, graph, object storage

## Intent

The storage contracts every subsystem stores through, plus reference backends.
Ports the abstract half of `openjiuwen/core/foundation/store/` (`base_kv_store.py`, `base_vector_store.py`, `base_db_store.py`, `graph/`, `object/`) with the Python `InMemoryKVStore` behavior as the reference.
Vendor backends (Redis, Milvus, Chroma, Gauss, Elasticsearch, pgvector, S3) are deliberately out of scope.
The five traits and their core types are re-exported at the module root (`store::KvStore`, `store::DocumentStore`, `store::VectorStore`, `store::GraphStore`, `store::ObjectStore`, ...), so consumers never import backend paths.

## API Surface

Key-value (`kv`):
- `KvValue` — `Text(String) | Bytes(Vec<u8>)`, with `text`/`bytes` constructors, `from_json`/`to_json`, `as_str`, `as_bytes`, `is_text`; JSON uses `"text"` / `{"$bytes": "<base64>"}`.
- `KvStore` async trait: `set(&str, KvValue)`, `get(&str) -> Option<KvValue>`, `exists`, `delete`, `exclusive_set(&str, KvValue, Option<Duration>) -> bool`, `get_by_prefix(&str) -> BTreeMap<String, KvValue>`, `delete_by_prefix(&str) -> usize`, `mget(&[String]) -> Vec<Option<KvValue>>` (input order), `batch_delete(&[String]) -> usize`, `pipeline() -> Box<dyn KvPipeline + '_>`.
- `KvPipeline` (`set`, `get`, `exists`, async `execute -> Vec<Option<KvValue>>`) and the shared `StorePipeline`; results follow op order: `set` → `None`, `get` → stored option, `exists` → `Some(KvValue::Text("true"|"false"))`.
- Backends: `InMemoryKvStore::new()/Default`, `LocalFileKvStore::new(root)` (async), `SqliteKvStore::connect(url)`/`from_pool(pool)` behind feature `sqlite`.

Document (`document`):
- `Document { id, fields: Map<String, Value> }` with `new`/`with_field`/`field`/`to_value`/`from_value`; the filter/order JSON view is `{"id": <id>, ..fields}` and `ID_FIELD = "id"`.
- `SortOrder` (`Asc | Desc`), `Query { filter (default null), limit: Option<usize>, offset: usize, order_by: Vec<(String, SortOrder)> }`.
- `DocumentStore` async trait: `create_collection` (idempotent), `list_collections`, `drop_collection`, `upsert(collection, Vec<Document>) -> usize` (auto-creates), `get(collection, id)`, `query(collection, &Query)`, `delete(collection, &Query) -> usize`, `count(collection, &Query) -> usize`.
- Backends: `InMemoryDocumentStore`, `LocalFileDocumentStore::new(root)`, `SqliteDocumentStore::connect(url)`/`from_pool(pool)` behind feature `sqlite`.

Vector (`vector`):
- `VectorDataType` — snake_case `varchar`, `float_vector`, `int64`, `int32`, `float`, `double`, `bool`, `json`, `array`.
- `FieldSchema { name, dtype, is_primary, auto_id, max_length, dim, description, default_value }`.
- `CollectionSchema { fields, description, enable_dynamic_field }` with `new`, `primary_key`, `vector_field`, `default_vector_field`, `vector_fields`, `validate`.
- `VectorQuery { query_vector, vector_field: Option<String>, top_k (default 5), filters }`, `VectorSearchResult { score, fields }`, `DEFAULT_TOP_K`.
- `Metric` (`Cosine` default, `Dot`, `Euclidean`) with normalized `score` in `[0,1]`, higher is better: cosine `(cos+1)/2`, dot `(tanh(dot)+1)/2`, euclidean `1/(1+distance)`; mismatched/empty vectors score `0.0` (empty vs empty `1.0`), a zero vector scores `0.5` against a non-zero one (orthogonal by convention), never `NaN`.
- `VectorStore` async trait: `create_collection(collection, CollectionSchema)`, `delete_collection`, `collection_exists`, `get_schema`, `add_docs(collection, Vec<Map<String, Value>>) -> usize` (insert-only), `search(collection, &VectorQuery)`, `delete_docs_by_ids`, `delete_docs_by_filters`, `list_collection_names`.
- Backends: `InMemoryVectorStore` (+ synchronous `search_with_metric`), `LocalFileVectorStore::new(root)` (+ `search_with_metric`); both share `vector::memory::VectorIndex`.

Graph (`graph`):
- `GraphNode { id, labels, properties }`, `GraphEdge { id, from, to, label, properties }`, `Direction` (`Outgoing | Incoming | Both`), `GraphQuery { labels, filter, limit }`.
- `GraphStore` async trait: `upsert_nodes(Vec<GraphNode>) -> usize`, `upsert_edges(Vec<GraphEdge>) -> usize`, `get_node`, `get_edge`, `neighbors(id, Direction, Option<&[String]>) -> Vec<GraphNode>`, `query_nodes(&GraphQuery)`, `delete_nodes(&[String]) -> usize`, `delete_edges(&[String]) -> usize`, `len() -> (usize, usize)`.
- Backends: `InMemoryGraphStore`, `LocalFileGraphStore::new(root)`; both share `graph::memory::GraphIndex`.

Object (`object`):
- `ObjectMeta { key, size, updated_at }` (`chrono::DateTime<Utc>`).
- `ObjectStore` async trait: `create_bucket` (idempotent), `delete_bucket`, `list_buckets`, `put(bucket, key, Bytes) -> ObjectMeta`, `get -> Option<Bytes>`, `delete` (missing = no-op), `exists`, `list(bucket, prefix, Option<usize>) -> Vec<ObjectMeta>`.
- Backends: `InMemoryObjectStore`, `LocalObjectStore::new(root)` (directories under `<root>/<bucket>/<key>`).

Filter (`filter`):
- `matches(filter: &Value, document: &Value) -> bool`, `matches_all(&[Value], &Value) -> bool`.
- Null/`{}` matches everything; non-object filters match nothing; implicit equality on a plain object of field→value pairs; dotted paths and `[n]` indices; `$and`/`$or`/`$not`; `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$contains`; numeric-aware equality (`1 == 1.0`); missing fields fail every operator except `$ne`; malformed operators match nothing and never panic.

## Constraints

- Every backend implements the same trait; behavior is verified by one shared contract suite (`src/store/tests/`) instantiated per backend.
- Values are JSON (`serde_json::Value`) or opaque bytes; no backend-specific types leak into the traits.
- Local backends live under a caller-provided root path and never touch paths outside it; collection/bucket/key names are validated as safe single segments where they become paths.
- `sqlite` code is `#[cfg(feature = "sqlite")]`; default builds and `--no-default-features` must not require `sqlx`.
- No blocking std I/O on async paths: backends use `tokio::fs`; in-memory state is guarded by `parking_lot::Mutex`, file backends by a `tokio::sync::Mutex` writer lock.
- The `KvStore` seam carries a TTL only on `exclusive_set` (`set` stores without expiry and clears any existing one) and has no `flush`, `close`, or `len`; a remote backend owns its connection object and maps transport failures to `Error::external`.
- Files stay ≤1000 lines; every public item is documented; no `unsafe`; no `unwrap`/`expect` outside tests.

## Design Decisions

- **Document store replaces `BaseDbStore`**: Python returns a SQLAlchemy `AsyncEngine`; not portable. A document-shaped contract (collections, JSON documents, filter queries, ordering, pagination) covers every Python consumer (message store, checkpoints, memory) without leaking a driver.
- **KV semantics** (Python parity): missing key → `None` (also inside `mget`, preserving input order); `get_by_prefix` → map in BTreeMap order (empty when nothing matches); `delete` of a missing key is a no-op; `batch_delete`/`delete_by_prefix` return the number deleted; `exclusive_set` writes only when absent or expired and accepts a relative expiry (`Duration`), returning `true` on write.
  TTL is honored lazily on every read path (and counts as absent for delete counts); the Python shelve/db backends never honored TTL — that bug is not reproduced.
- **Values are text or bytes**: JSON callers use `KvValue::from_json`; the file backend stores one envelope `{key, kind, data, expiry_ms}` per key under `<root>/entries/<sha256>.json`, so prefix scans and TTL work after reload.
- **File backends are snapshots**: document = `<root>/<collection>.json` (`{collection, documents}`), vector = `<root>/vector/<collection>.json`, graph = `<root>/graph.json`; each mutation rewrites the snapshot atomically (temp + rename) and a new instance reloads it on open.
  One live instance per root is assumed; separate instances do not observe each other's later writes until reopened.
- **Document semantics**: `upsert` replaces by id and auto-creates the collection; `get`/`query`/`count`/`delete` on a missing collection return `None`/empty/`0`; `filter` runs against `{"id", ..fields}` (a field named `id` is shadowed); ordering is stable, criteria primary-first, numbers numerically, strings lexicographically, `false < true`, null/missing before values for `Asc` and after for `Desc`, unlike types by type rank then serialized JSON; `offset` then `limit`; `delete`/`count` ignore paging and order.
- **Vector semantics**: `create_collection` validates the schema — exactly one primary key (not a vector field), at least one `FLOAT_VECTOR` with `dim > 0`, unique field names — and rejects duplicate collection names (`Error::Validation`).
  `add_docs` validates the whole batch first, inserts only (duplicate primary key within the batch or stored → `Error::Validation`), generates UUIDs for `auto_id`, applies `default_value`, type-checks populated scalar fields, and stores unknown fields as-is.
  `search` resolves `vector_field` (default first vector field), validates the query dimension, filters through `store::filter::matches`, sorts `score desc, primary key asc`, truncates to `top_k`; the trait scores with cosine, `search_with_metric` exposes dot/L2.
  In-memory and file backends share `VectorIndex`, so rankings cannot drift.
- **Graph semantics**: Python's `Entity`/`Relation`/`Episode` triple becomes `GraphNode`/`GraphEdge` carrying labels; upserts are id-keyed wholesale replacements (not merges) and validate empty ids before mutating.
  `neighbors` follows `from`/`to` per `Direction`, filters on the **edge** label when given, deduplicates adjacent nodes and sorts by id; a missing node returns an empty vec.
  `query_nodes` matches nodes whose labels contain ANY requested label, then applies `filter::matches` to `{"id", "labels", ..properties}` (properties override id/labels), sorts by id and applies `limit`.
  `delete_nodes` cascades every incident edge and counts only nodes that existed; `len` returns `(nodes, edges)`.
- **Object semantics**: missing bucket → `Error::NotFound` for all object operations; `create_bucket` is idempotent and `delete_bucket` removes contents (missing = no-op); listings are sorted by key and honor `prefix`/`limit`.
  `LocalObjectStore` rejects empty, absolute, `\\`-containing, `.`/`..`-segment keys and buckets with `Error::Validation` and skips symlinks when listing.
- **SQLite** (feature `sqlite`) implements `KvStore` and `DocumentStore` over `sqlx`: `kv_store(k VARCHAR(255) PRIMARY KEY, value TEXT, kind TEXT, expiry_ms INTEGER)` with upsert and `substr`-based prefix matching; documents use one `documents(collection, id, doc_json)` table for all collections plus `document_collections(name)`, with filtering/ordering/paging done by the shared Rust query engine.
  `sqlite::memory:` URLs are pinned to `max_connections(1)` so the database survives across pool checkouts.

## Routing Table

- `./kv/` → `KvStore`, `KvValue`, `KvPipeline`, in-memory / file / sqlite backends
- `./kv/mod.rs` → `KvValue`, `KvStore`, `KvPipeline`, shared `StorePipeline`, re-exports
- `./kv/memory.rs` → `InMemoryKvStore` (lazy TTL, BTreeMap)
- `./kv/file.rs` → `LocalFileKvStore` (JSON envelopes under `entries/`, index rebuilt on open)
- `./kv/sqlite.rs` → `SqliteKvStore` (feature `sqlite`)
- `./document/` → `DocumentStore`, `Document`, `Query`, `SortOrder`, in-memory / file / sqlite backends
- `./document/mod.rs` → data types, trait, collection-name validation, shared query engine, contract suite
- `./document/memory.rs` → `InMemoryDocumentStore`
- `./document/file.rs` → `LocalFileDocumentStore` (one JSON snapshot per collection)
- `./document/sqlite.rs` → `SqliteDocumentStore` (feature `sqlite`, single documents table)
- `./vector/` → `VectorStore`, schema types, `Metric`, shared index engine, in-memory / file backends
- `./vector/mod.rs` → data types, schema validation, `Metric`, trait
- `./vector/memory.rs` → `VectorIndex` engine + `InMemoryVectorStore`
- `./vector/file.rs` → `LocalFileVectorStore` (per-collection snapshots)
- `./graph/` → `GraphStore`, node/edge models, `GraphIndex`, in-memory / file backends
- `./object/` → `ObjectStore`, `ObjectMeta`, in-memory / local-directory backends
- `./filter.rs` → shared filter matcher
- `./tests/` → cross-backend contract suite (`kv`, `document`, `vector`, `graph_object`, `sqlite`)

## Test Strategy

- Unit tests live beside each implementation (`#[cfg(test)] mod tests`), one behavior per test, no network.
- The contract suite in `src/store/tests/` runs the same assertions against every backend: `assert_kv_contract`, `assert_document_contract`, `assert_vector_contract`, `assert_graph_contract`, `assert_object_contract`; sqlite runs the KV and document contracts with `sqlite::memory:`.
- File backends are exercised with `tempfile::TempDir` and reopened to verify persistence; TTL tests use short durations with ≥40ms margins.
- Behavior-port tests name the mirrored Python suite in a file comment (`// ports tests/unit_tests/core/foundation/store/...`).
- `cargo test -p openjiuwen-foundation store`, `--no-default-features store`, and `--features sqlite store` must all pass offline; `cargo clippy -p openjiuwen-foundation --all-targets` must be clean for `src/store`.
