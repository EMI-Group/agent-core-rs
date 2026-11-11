# store — kv, document, vector, graph, object storage

## Intent

The storage contracts every subsystem stores through, plus reference backends.
Ports the abstract half of `openjiuwen/core/foundation/store/` (`base_kv_store.py`, `base_vector_store.py`, `base_db_store.py`, `graph/`, `object/`) with the Python `InMemoryKVStore`/`ShelveStore` behavior as the reference.
Vendor backends (Redis, Milvus, Chroma, Gauss, Elasticsearch, pgvector, S3) are deliberately out of scope.

## API Surface

- `kv`: `KvValue` (`Text(String) | Bytes(Vec<u8>)`, with `from_json`/`to_json` helpers), `KvStore` trait (`set`, `get`, `exists`, `delete`, `exclusive_set(key, value, expiry)`, `get_by_prefix`, `delete_by_prefix`, `mget`, `batch_delete`, `pipeline`), `KvPipeline` (`set`, `get`, `exists`, `execute`), `InMemoryKvStore`, `LocalFileKvStore`, `SqliteKvStore` (feature `sqlite`).
- `document`: `Document { id, fields: Map<String, Value> }`, `Query { filter: Value, limit: Option<usize>, offset: usize, order_by: Vec<(String, SortOrder)> }`, `DocumentStore` trait (`create_collection`, `list_collections`, `drop_collection`, `upsert`, `get`, `query`, `delete`, `count`), `InMemoryDocumentStore`, `LocalFileDocumentStore`, `SqliteDocumentStore` (feature `sqlite`).
- `vector`: `VectorDataType`, `FieldSchema { name, dtype, is_primary, auto_id, max_length, dim, description, default_value }`, `CollectionSchema { fields, description, enable_dynamic_field }`, `VectorQuery { query_vector, vector_field, top_k /* 5 */, filters }`, `VectorSearchResult { score, fields }`, `VectorStore` trait (`create_collection`, `delete_collection`, `collection_exists`, `get_schema`, `add_docs`, `search`, `delete_docs_by_ids`, `delete_docs_by_filters`, `list_collection_names`), `InMemoryVectorStore`, `LocalFileVectorStore`.
- `graph`: `GraphNode { id, labels, properties }`, `GraphEdge { id, from, to, label, properties }`, `Direction` (`Outgoing | Incoming | Both`), `GraphQuery { labels, filter, limit }`, `GraphStore` trait (`upsert_nodes`, `upsert_edges`, `get_node`, `get_edge`, `neighbors`, `query_nodes`, `delete_nodes`, `delete_edges`, `len`), `InMemoryGraphStore`, `LocalFileGraphStore`.
- `object`: `ObjectMeta { key, size, updated_at }`, `ObjectStore` trait (`create_bucket`, `delete_bucket`, `list_buckets`, `put`, `get`, `delete`, `exists`, `list`), `InMemoryObjectStore`, `LocalObjectStore`.
- `filter`: `matches(filter: &Value, document: &Value) -> bool` plus `FilterOp` parsing — the single filter language shared by every backend.

## Constraints

- Every backend implements the same trait; behavior verified by one shared contract test suite instantiated per backend (`store::contract` test helpers or per-file copies kept identical).
- Values are JSON (`serde_json::Value`) or opaque bytes; no backend-specific types leak into the traits.
- Local backends live under a caller-provided root path and never touch paths outside it.
- `sqlite` code is `#[cfg(feature = "sqlite")]`; default builds and `--no-default-features` must not require `sqlx`.

## Design Decisions

- **Document store replaces `BaseDbStore`**: Python returns a SQLAlchemy `AsyncEngine`; not portable. A document-shaped contract (collections, JSON documents, filter queries, ordering, pagination) covers every Python consumer of the db store (message store, checkpoints, memory) without leaking a driver.
- **KV semantics** (Python parity): missing key → `None` (also inside `mget`, preserving input order); `get_by_prefix` → map (empty when nothing matches); `delete` of a missing key is a no-op; `batch_delete` returns the number deleted; `exclusive_set` writes only when absent and accepts an absolute expiry (`Duration` from now), returning `true` on write.
  TTL is honored lazily on read for the in-memory and file backends (Python's shelve/db backends never honored it — that bug is not reproduced).
- **Values are text or bytes**: JSON callers use `KvValue::from_json`; the file backend stores a small JSON envelope `{key, kind, data, expiry_ms}` so prefix scans and TTL work after reload.
- **Vector semantics**: one primary key per collection, at least one `FLOAT_VECTOR` field with `dim > 0` (validated at `create_collection` → `Error::Validation`); `add_docs` is insert-only (duplicate id → `Error::Validation`); `search` returns scores normalized to "higher is better" in `[0, 1]` (cosine similarity, or `1 - distance` clamped), never `NaN`; `top_k` defaults to 5; filters use `store::filter`.
  In-memory and file backends share one index engine (`vector::memory::VectorIndex`) so rankings cannot drift; the file backend persists a JSON snapshot after each mutation.
- **Graph** replaces Python's `Entity`/`Relation`/`Episode` triple with `GraphNode`/`GraphEdge` carrying a `label`; traversal is bidirectional by `Direction`. Node/edge upserts are id-keyed and replace the previous value.
- **Object store** exposes bucket lifecycle plus put/get/delete/list with `limit`; the local backend maps `bucket/key` to `root/bucket/key` directories and files.
- **SQLite** (feature `sqlite`) implements `KvStore` and `DocumentStore` over `sqlx` (`sqlite::memory:` supported for tests), with `key TEXT PRIMARY KEY` and one JSON column per document collection.

## Routing Table

- `./kv/` → `KvStore`, `KvValue`, pipeline, in-memory / file / sqlite backends
- `./document/` → `DocumentStore`, `Document`, `Query`, in-memory / file / sqlite backends
- `./vector/` → `VectorStore`, schema types, shared index engine, in-memory / file backends
- `./graph/` → `GraphStore`, node/edge models, in-memory / file backends
- `./object/` → `ObjectStore`, metadata, in-memory / local-directory backends
- `./filter.rs` → shared filter matcher (`$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$contains`, implicit equality, dotted paths)
