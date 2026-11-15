# store — kv, document, vector, graph, object storage

## Intent

The storage contracts every subsystem stores through, plus reference backends and network-backed adapters.
Ports the abstract half of `openjiuwen/core/foundation/store/` (`base_kv_store.py`, `base_vector_store.py`, `base_db_store.py`, `graph/`, `object/`) with the Python `InMemoryKVStore` behavior as the reference.
The five contracts also have external adapters under `external/` — Chroma, Milvus and Elasticsearch vectors, Elasticsearch documents, Redis key-value, S3-compatible objects — built from JSON configuration by `external::registry`.
In-memory, local-filesystem and SQLite backends stay the behavioral reference: an adapter matches their semantics wherever the service can express them and documents the difference where it cannot.
The five traits and their core types are re-exported at the module root (`store::KvStore`, `store::DocumentStore`, `store::VectorStore`, `store::GraphStore`, `store::ObjectStore`, ...), so consumers never import backend paths.

## API Surface

Key-value (`kv`):
- `KvValue` — `Text(String) | Bytes(Vec<u8>)`, with `text`/`bytes` constructors, `from_json`/`to_json`, `as_str`, `as_bytes`, `is_text`; JSON uses `"text"` / `{"$bytes": "<base64>"}`.
- `KvStore` async trait: `set(&str, KvValue)`, `get(&str) -> Option<KvValue>`, `exists`, `delete`, `exclusive_set(&str, KvValue, Option<Duration>) -> bool`, `get_by_prefix(&str) -> BTreeMap<String, KvValue>`, `delete_by_prefix(&str) -> usize`, `mget(&[String]) -> Vec<Option<KvValue>>` (input order), `batch_delete(&[String]) -> usize`, `pipeline() -> Box<dyn KvPipeline + '_>`.
- `KvPipeline` (`set`, `get`, `exists`, async `execute -> Vec<Option<KvValue>>`) and the shared `StorePipeline`; results follow op order: `set` → `None`, `get` → stored option, `exists` → `Some(KvValue::Text("true"|"false"))`.
- Backends: `InMemoryKvStore::new()/Default`, `LocalFileKvStore::new(root)` (async), `SqliteKvStore::connect(url)`/`from_pool(pool)` behind feature `sqlite`, `RedisKvStore::connect(&RedisConfig)`.

Document (`document`):
- `Document { id, fields: Map<String, Value> }` with `new`/`with_field`/`field`/`to_value`/`from_value`; the filter/order JSON view is `{"id": <id>, ..fields}` and `ID_FIELD = "id"`.
- `SortOrder` (`Asc | Desc`), `Query { filter (default null), limit: Option<usize>, offset: usize, order_by: Vec<(String, SortOrder)> }`.
- `DocumentStore` async trait: `create_collection` (idempotent), `list_collections`, `drop_collection`, `upsert(collection, Vec<Document>) -> usize` (auto-creates), `get(collection, id)`, `query(collection, &Query)`, `delete(collection, &Query) -> usize`, `count(collection, &Query) -> usize`.
- Backends: `InMemoryDocumentStore`, `LocalFileDocumentStore::new(root)`, `SqliteDocumentStore::connect(url)`/`from_pool(pool)` behind feature `sqlite`, `ElasticsearchDocumentStore::new(ElasticsearchConfig)`.

Vector (`vector`):
- `VectorDataType` — snake_case `varchar`, `float_vector`, `int64`, `int32`, `float`, `double`, `bool`, `json`, `array`.
- `FieldSchema { name, dtype, is_primary, auto_id, max_length, dim, description, default_value }`.
- `CollectionSchema { fields, description, enable_dynamic_field }` with `new`, `primary_key`, `vector_field`, `default_vector_field`, `vector_fields`, `validate`.
- `VectorQuery { query_vector, vector_field: Option<String>, top_k (default 5), filters }`, `VectorSearchResult { score, fields }`, `DEFAULT_TOP_K`.
- `Metric` (`Cosine` default, `Dot`, `Euclidean`) with normalized `score` in `[0,1]`, higher is better: cosine `(cos+1)/2`, dot `(tanh(dot)+1)/2`, euclidean `1/(1+distance)`; mismatched/empty vectors score `0.0` (empty vs empty `1.0`), never `NaN`.
- `VectorStore` async trait: `create_collection(collection, CollectionSchema)`, `delete_collection`, `collection_exists`, `get_schema`, `add_docs(collection, Vec<Map<String, Value>>) -> usize` (insert-only), `search(collection, &VectorQuery)`, `delete_docs_by_ids`, `delete_docs_by_filters`, `list_collection_names`.
- Backends: `InMemoryVectorStore` (+ synchronous `search_with_metric`), `LocalFileVectorStore::new(root)` (+ `search_with_metric`); both share `vector::memory::VectorIndex`.
  External: `ChromaVectorStore::new(ChromaConfig)`, `MilvusVectorStore::new(MilvusConfig)`, `ElasticsearchVectorStore::new(ElasticsearchConfig)`.

Graph (`graph`):
- `GraphNode { id, labels, properties }`, `GraphEdge { id, from, to, label, properties }`, `Direction` (`Outgoing | Incoming | Both`), `GraphQuery { labels, filter, limit }`.
- `GraphStore` async trait: `upsert_nodes(Vec<GraphNode>) -> usize`, `upsert_edges(Vec<GraphEdge>) -> usize`, `get_node`, `get_edge`, `neighbors(id, Direction, Option<&[String]>) -> Vec<GraphNode>`, `query_nodes(&GraphQuery)`, `delete_nodes(&[String]) -> usize`, `delete_edges(&[String]) -> usize`, `len() -> (usize, usize)`.
- Backends: `InMemoryGraphStore`, `LocalFileGraphStore::new(root)`; both share `graph::memory::GraphIndex`.
  There is no external graph backend: the registry's `Graph` family resolves `memory` and `file`/`local` only.

Object (`object`):
- `ObjectMeta { key, size, updated_at }` (`chrono::DateTime<Utc>`).
- `ObjectStore` async trait: `create_bucket` (idempotent), `delete_bucket`, `list_buckets`, `put(bucket, key, Bytes) -> ObjectMeta`, `get -> Option<Bytes>`, `delete` (missing = no-op), `exists`, `list(bucket, prefix, Option<usize>) -> Vec<ObjectMeta>`.
- Backends: `InMemoryObjectStore`, `LocalObjectStore::new(root)` (directories under `<root>/<bucket>/<key>`), `S3ObjectStore::new(S3Config)`.

Filter (`filter`):
- `matches(filter: &Value, document: &Value) -> bool`, `matches_all(&[Value], &Value) -> bool`.
- Null/`{}` matches everything; non-object filters match nothing; implicit equality on a plain object of field→value pairs; dotted paths and `[n]` indices; `$and`/`$or`/`$not`; `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$contains`; numeric-aware equality (`1 == 1.0`); missing fields fail every operator except `$ne`; malformed operators match nothing and never panic.

External (`external`):
- `external::chroma` — `ChromaConfig`, `ChromaVectorStore` (`VectorStore`).
- `external::milvus` — `MilvusConfig`, `MilvusVectorStore` (`VectorStore`).
- `external::elasticsearch` — `ElasticsearchConfig`, `ElasticsearchVectorStore` (`VectorStore`), `ElasticsearchDocumentStore` (`DocumentStore`), and `query::translate` (filter → query-DSL, `None` = not expressible).
- `external::redis` — `RedisConfig`, `RedisKvStore` (`KvStore`), plus `redis::codec` (`RespValue`, `encode_command`, `encode`, `Decoder`) and `redis::client` (`RedisConnection`, `RedisServerError`, `READ_ONLY_COMMANDS`).
- `external::s3` — `S3Config`, `S3ObjectStore` (`ObjectStore`), plus `s3::sigv4` (`Credentials`, `SigningInput`, `SigningResult`, `sign_request`, `hmac_sha256`, `sha256_hex`).
- `external::registry` — `StoreFamily`, `BackendKind`, `StoreSpec`, `StoreRegistry`, `global_registry`, `available_backend_names`, the free `create_{kv,document,vector,object,graph}_store`, the free `register_*_backend`, and the `KvStoreFactory`/`DocumentStoreFactory`/`VectorStoreFactory`/`ObjectStoreFactory`/`GraphStoreFactory` aliases.
  Every adapter and registry item is re-exported at `store::` as well (`store::ChromaVectorStore`, `store::create_vector_store`, ...).

## Constraints

- Every backend implements the same trait; behavior is verified by one shared contract suite (`src/store/tests/`) instantiated per backend.
- Values are JSON (`serde_json::Value`) or opaque bytes; no backend-specific types leak into the traits.
- Local backends live under a caller-provided root path and never touch paths outside it; collection/bucket/key names are validated as safe single segments where they become paths.
- `sqlite` code is `#[cfg(feature = "sqlite")]`; default builds and `--no-default-features` must not require `sqlx`.
- No blocking std I/O on async paths: backends use `tokio::fs`; in-memory state is guarded by `parking_lot::Mutex`, file backends by a `tokio::sync::Mutex` writer lock.
- External adapters send every HTTP request through `openjiuwen_common::http::HttpClient` and speak RESP2 over a raw `tokio::net::TcpStream` for Redis (wrapped in `tokio-rustls` for `rediss://`): no vendor SDK and no vendor HTTP client.
- External adapters are always compiled — their dependencies are plain, non-optional crate dependencies — so only the `sqlite` branches are feature-gated.
- External tests are offline like the rest: `wiremock` servers for HTTP, a scripted `TcpListener` for Redis, known-answer vectors for the S3 signer.
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
- **Shared config convention for external backends**: each adapter has one `*Config` that is `serde`-deserializable with `#[serde(default)]`, has a `Default` impl, and exposes `new(...)` plus `with_*` builder setters.
  Secrets are `openjiuwen_common::security::SecretString`, each config carries `timeout` (30 s HTTP, 5 s Redis), and the vector configs carry `metric: Metric` defaulting to `Cosine` because `CollectionSchema` has no metric member.
  The registry decodes any backend's config from the JSON object it was given, so no backend needs a separate construction path for configuration files.
- **Score normalization per adapter**: every `search` returns scores in `[0, 1]`, higher = more similar, matching `Metric::score`.
  Chroma maps cosine/IP distance `d` to `1 - d/2` and squared-L2 distance `d` to `1 / (1 + d)`.
  Milvus maps COSINE/IP similarity `s` to `(s + 1) / 2` and L2 distance `d` to `max(0, (4 - d) / 4)`.
  Elasticsearch `_score` is used as-is and clamped into `[0, 1]`; every adapter clamps.
- **Redis value encoding**: `KvValue::Text` is stored as raw UTF-8 and `KvValue::Bytes` as the ASCII marker `\0ojb:` plus standard base64, so text can never collide with bytes; a non-UTF-8 payload without the marker decodes as `Bytes` (lossless fallback, never a lossy replacement).
- **Redis reconnect rule**: at most one transparent reconnect per command — always for a failure before or while writing, and only for the read-only `PING`/`GET`/`MGET`/`EXISTS`/`SCAN` when the reply was lost, because replaying a mutating command could apply it twice.
- **Redis TLS follows the URL scheme**: `rediss://` (or `with_tls(true)`) wraps the socket in a `tokio-rustls` client that trusts the webpki roots only; an explicit `addr` never clears TLS, so a `rediss://` URL is never silently downgraded, while custom CA bundles, pinning and skip-verify are deliberately absent (`external/redis/CONTEXT.md`).
- **S3 Signature V4 in-crate**: `s3::sigv4` derives SigV4 from `sha2` with hand-written hex/base64 helpers (no `hmac`/`hex` crate) and takes an explicit `now`, so signatures are reproducible in tests.
  Requests are path-style by default and virtual-host when `path_style = false`; a 404 on `GET`/`HEAD`/`DELETE` is tolerated as "absent" because S3 cannot tell a missing bucket from a missing key.
- **Registry globals**: the plugin maps behind `register_*_backend` are the one documented mutable global of this subtree; `global_registry()` exposes them and `StoreRegistry::new()` builds an isolated instance over per-instance maps for tests.
  A registered name wins over a built-in of the same spelling, a backend of the wrong family is a `Validation` error listing the valid names, and `pgvector`/`postgres`/`gauss`/`gaussvector` parse but return `Unsupported`.

## Known Issues and Gaps

- PostgreSQL/pgvector and Gauss vector backends are deferred: `pgvector`, `postgres` and `gauss`/`gaussvector` are recognized registry names whose construction returns `Error::Unsupported` with the missing-driver reason.
- No external graph store exists: the `Graph` family registers only `memory` and `file`/`local`, and a spec naming any other backend fails rather than degrading.
- Redis has no cluster/Sentinel or connection-pool support (commands serialize behind one connection mutex), but `rediss://` TLS works through `tokio-rustls` with the webpki roots only — no custom CA, pinning or skip-verify.
- Elasticsearch document `query`/`count`/`delete` page with `search_after` (`SEARCH_PAGE_SIZE` 500) and fail with `Unsupported` once a read would exceed `MAX_SCAN_DOCUMENTS` (10 000); they never return a truncated list, and an untranslatable filter warns before taking the local scan.
- Milvus `add_docs` detects duplicate primary keys with a pre-insert `entities/query` probe and returns `Validation` naming them; `MilvusConfig::reject_duplicate_primary_keys = false` restores the service's append semantics.
- Chroma keeps one embedding per record and scalar-only metadata, so multi-vector schemas and JSON/array fields are rejected, and a schema is read from the Python `schema`/`fields` metadata keys or the legacy `schema_metadata_key` (a collection carrying none of them cannot be read).
- S3 cannot distinguish a missing bucket from a missing key on `GET`/`HEAD`/`DELETE`, so both read as "absent".

## Routing Table

- `./kv/` → `KvStore`, `KvValue`, `KvPipeline`, in-memory / file / sqlite / Redis backends
- `./kv/mod.rs` → `KvValue`, `KvStore`, `KvPipeline`, shared `StorePipeline`, re-exports
- `./kv/memory.rs` → `InMemoryKvStore` (lazy TTL, BTreeMap)
- `./kv/file.rs` → `LocalFileKvStore` (JSON envelopes under `entries/`, index rebuilt on open)
- `./kv/sqlite.rs` → `SqliteKvStore` (feature `sqlite`)
- `./document/` → `DocumentStore`, `Document`, `Query`, `SortOrder`, in-memory / file / sqlite / Elasticsearch backends
- `./document/mod.rs` → data types, trait, collection-name validation, shared query engine, contract suite
- `./document/memory.rs` → `InMemoryDocumentStore`
- `./document/file.rs` → `LocalFileDocumentStore` (one JSON snapshot per collection)
- `./document/sqlite.rs` → `SqliteDocumentStore` (feature `sqlite`, single documents table)
- `./vector/` → `VectorStore`, schema types, `Metric`, shared index engine, in-memory / file backends (Chroma, Milvus and Elasticsearch live in `external/`)
- `./vector/mod.rs` → data types, schema validation, `Metric`, trait
- `./vector/memory.rs` → `VectorIndex` engine + `InMemoryVectorStore`
- `./vector/file.rs` → `LocalFileVectorStore` (per-collection snapshots)
- `./graph/` → `GraphStore`, node/edge models, `GraphIndex`, in-memory / file backends
- `./object/` → `ObjectStore`, `ObjectMeta`, in-memory / local-directory backends (S3 lives in `external/s3/`)
- `./filter.rs` → shared filter matcher
- `./external/` → network-backed adapters (Chroma, Milvus, Elasticsearch, Redis, S3) and the JSON-config registry (`external/CONTEXT.md`)
- `./external/elasticsearch/` → Elasticsearch config/client, filter→query-DSL translation, vector and document stores
- `./tests/` → cross-backend contract suite (`kv`, `document`, `vector`, `graph_object`, `sqlite`)

## Test Strategy

- Unit tests live beside each implementation (`#[cfg(test)] mod tests`), one behavior per test, no network.
- The contract suite in `src/store/tests/` runs the same assertions against every backend: `assert_kv_contract`, `assert_document_contract`, `assert_vector_contract`, `assert_graph_contract`, `assert_object_contract`; sqlite runs the KV and document contracts with `sqlite::memory:`.
- The external adapters are covered by their own offline suites: `wiremock` loopback servers for Chroma, Milvus and Elasticsearch (`chroma/tests.rs`, `milvus/tests.rs`, `elasticsearch/{document,vector}/tests.rs`) and for S3 (`s3/tests.rs`), a scripted `tokio::net::TcpListener` speaking RESP2 for Redis (`redis/tests.rs`), and AWS SigV4 known-answer vectors (`s3/tests/sigv4_vectors.rs`).
- `external/registry/tests.rs` exercises every family, the aliases, the deferred names and plugin precedence on isolated `StoreRegistry` instances (plus the global registry).
- File backends are exercised with `tempfile::TempDir` and reopened to verify persistence; TTL tests use short durations with ≥40ms margins.
- Behavior-port tests name the mirrored Python suite in a file comment (`// ports tests/unit_tests/core/foundation/store/...`).
- `cargo test -p openjiuwen-foundation store`, `--no-default-features store`, and `--features sqlite store` must all pass offline; `cargo clippy -p openjiuwen-foundation --all-targets` must be clean for `src/store`.
