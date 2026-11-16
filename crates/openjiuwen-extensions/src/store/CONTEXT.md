# store/ — backend selection, live SQLite (kv/document/brute-force vector), Redis (kv/document/vector) and Elasticsearch (vector/document) adapters

## Intent

Python's `extensions/store/` contributes optional-driver backends (`RedisStore`, `ElasticsearchVectorStore`, `GaussDbStore`) on top of the kernel's storage contracts; memory/file/sqlite live in `openjiuwen_foundation::store`.
Rust mirrors that: a config-driven factory over the kernel backends, a live single-node Redis family (key-value, document, vector) over the crate's in-house RESP client (`src/redis/`), a live Elasticsearch family (vector, document) over REST, and a live SQLite vector backend for the one vector surface the kernel does not ship.
Pick this node for "which KV/document/vector store do I build from config", for the SQLite vector blob layout and its brute-force limits, for the Redis key layout and its reference-delegation contract, for the Elasticsearch credential handling, and for the remaining vendor-backend gaps (Gauss/PostgreSQL).

## API Surface

```rust
pub use openjiuwen_core::foundation::store::{
    KvStore, KvValue, DocumentStore, Document, Query, SortOrder, VectorStore, VectorQuery,
    VectorSearchResult, CollectionSchema, FieldSchema, VectorDataType, Metric, InMemoryKvStore,
    LocalFileKvStore, InMemoryDocumentStore, LocalFileDocumentStore, InMemoryVectorStore,
    LocalFileVectorStore, ElasticsearchConfig, ElasticsearchDocumentStore,
};
#[cfg(feature = "sqlite")]
pub use openjiuwen_core::foundation::store::{SqliteKvStore, SqliteDocumentStore};
#[cfg(feature = "sqlite")]
pub use sqlite_vector::SqliteVectorStore;   // brute-force VectorStore over `sqlx`

pub enum StoreBackend {
    Memory,
    File { root: PathBuf },
    Sqlite { url: String },
    Redis { url: Option<String> },
    Elasticsearch { url: String, index_prefix: String,
                    username: Option<String>, password: Option<String>, api_key: Option<String> },
}
impl StoreBackend {
    pub fn kind(&self) -> StoreKind;                 // Memory | File | Sqlite | Redis | Elasticsearch
    pub fn from_json(value: &Value) -> Result<Self>; // tolerant snake_case decoder; unknown kind = Error::Validation
}

pub async fn build_kv_store(backend: &StoreBackend) -> Result<Arc<dyn KvStore>>;
pub async fn build_document_store(backend: &StoreBackend) -> Result<Arc<dyn DocumentStore>>;
pub async fn build_vector_store(backend: &StoreBackend) -> Result<Arc<dyn VectorStore>>;

/// `redis` feature only: three stores over one shared connection layer.
pub mod redis {
    pub struct RedisStoreConfig { pub url: Option<String>, pub cluster_mode: Option<bool>,
                                  pub connection_args: BTreeMap<String, Value> }
    pub struct RedisStore { .. }          // key-value
    pub struct RedisDocumentStore { .. }  // DocumentStore
    pub struct RedisVectorStore { .. }    // VectorStore, `with_metric` (default Metric::Cosine)
    impl RedisStore         { pub fn new(config) -> Result<Self>; pub async fn connect(config) -> Result<Arc<dyn KvStore>>;
                              pub fn config(&self) -> &RedisStoreConfig; pub async fn ping(&self) -> Result<()>; }
    impl RedisDocumentStore { pub fn new(config) -> Result<Self>; pub async fn connect(config) -> Result<Arc<dyn DocumentStore>>;
                              pub fn config(&self) -> &RedisStoreConfig; pub async fn ping(&self) -> Result<()>; }
    impl RedisVectorStore   { pub fn new(config) -> Result<Self>; pub fn with_metric(self, m: Metric) -> Self;
                              pub async fn connect(config) -> Result<Arc<dyn VectorStore>>;
                              pub fn config(&self) -> &RedisStoreConfig; pub fn metric(&self) -> Metric;
                              pub async fn ping(&self) -> Result<()>; }
}

/// Elasticsearch vector store (no feature; live REST over `reqwest`).
pub struct ElasticsearchVectorStoreConfig { pub url: String, pub index_prefix: String, pub dimension: usize }
pub struct ElasticsearchVectorStore { .. }
impl ElasticsearchVectorStore {
    pub fn new(config: ElasticsearchVectorStoreConfig) -> Result<Self>;       // URL/dimension validation, no I/O
    pub async fn connect(config: ElasticsearchVectorStoreConfig) -> Result<Arc<dyn VectorStore>>;
    pub fn index_name(&self, collection: &str) -> String;  // "{index_prefix}__{collection}"
}
#[async_trait] impl VectorStore for ElasticsearchVectorStore { /* full contract, REST-backed */ }
```

Factory matrix: `memory`/`file` build KV + document + vector; `sqlite` builds KV + document, plus a brute-force vector store under the `sqlite` feature (feature off: `Unsupported` naming the feature); `redis` builds all three under the `redis` feature (feature off: all three `Unsupported` naming the feature); `elasticsearch` builds document + vector (KV `Unsupported`, it is not a key-value store).
A backend without a surface for the requested factory returns `Error::Unsupported` naming the backend and the supported alternatives; no method silently succeeds.

## Ported Behavior

- Backend selection is config-driven exactly like the Python `type` discriminators (`memory`, `file`, `sqlite`, `redis`, `elasticsearch`).
- `RedisStore::new` (and the two sibling `new`s) only validate and normalize; the `OnceCell` client opens on first use, and every `connect` opens eagerly and verifies the connection with `PING`, so a bad URL, a closed port, or a redirect-only server fails before the first write.
- Redis keys are used verbatim by the key-value store: Python's `index_prefix`-style namespacing stays the caller's responsibility, and the factory passes `StoreBackend::Redis::url` straight into `RedisStoreConfig`.
- Redis values are stored as a one-byte kind tag (`0x00` text, `0x01` bytes) followed by the raw payload, so a `KvValue` keeps its `Text`/`Bytes` variant across a round trip; this is an intentional deviation from Python's bare-payload `RedisStore`, so Python-written values are not directly readable (and vice versa).
- A Redis payload with a missing or unknown kind tag is `Error::Validation` naming the key; keys and values are never logged.
- `set` issues a plain `SET` (which clears any TTL), `get` issues `GET` (`Null` → `None`), `exists` reads `EXISTS > 0`, and `delete` issues `DEL`.
- `exclusive_set` issues `SET key value NX [PX <ms>]`, returns `false` on `Null`, and rounds a sub-millisecond expiry up to 1 ms so a near-zero TTL never means "no expiry".
- `mget` sends one `MGET`, preserving input order with `None` per miss; an empty input returns an empty vector without a round trip.
- `get_by_prefix` and `delete_by_prefix` walk `SCAN` with `MATCH <escaped-prefix>*` and `COUNT 500`, escaping `*`, `?`, `[`, `]`, and `\` in the prefix, then apply `MGET`/`UNLINK` in 500-key chunks; keys that vanish between `SCAN` and `MGET` are skipped.
- `delete_by_prefix` and `batch_delete` sum the integer replies, so only live keys count (duplicates and dead keys count once), and delete with `UNLINK` falling back to `DEL` when the server answers an unknown-command error.
- `pipeline()` returns a real batched `KvPipeline`: commands are queued synchronously, sent through one `Client::pipeline` write-and-read, results are mapped in order (`set` → `None`, `get` → decoded option, `exists` → `"true"`/`"false"`), and the queue is cleared so reuse cannot replay old operations.
- The kernel `StorePipeline` is deliberately not used for Redis: it issues one round trip per operation, while batching is the point of the Redis pipeline.
- `RedisStoreConfig` accepts `url` **or** `connection_args`; the Rust surface validates the URL scheme (`redis://`, `rediss://`, `redis+cluster://`, `rediss+cluster://`), infers `cluster_mode` from `+cluster`, and builds the client options from the URL plus `host`/`port` overrides and `username`/`password`/`db`/`timeout`.
- Elasticsearch collections are indices named `"{index_prefix}__{collection}"`, the full `CollectionSchema` is stored in the mapping's `_meta.openjiuwen_schema` (replacing Python's `_collection_metadata__` document), dense vectors are `dense_vector` fields with cosine similarity, and search is a `knn` query.
- Elasticsearch `create_collection` on an existing index is a `Validation` error; `auto_id`/`default_value` filling, duplicate-key detection, and type/dimension validation happen before the single `_bulk` request; `delete_docs_by_filters` with `null`/`{}` deletes everything.
- Elasticsearch vector defaults: index = `"{index_prefix}__{collection}"`, vector field default `768` dimensions, cosine similarity; search hits are clamped to `[0,1]`, sorted score-desc then primary-key-asc, and truncated to `top_k`.
- Elasticsearch vector deviations: single node only (no sniffing/cloud id), no scroll/PIT/`search_after`, no per-collection metric registry (`CollectionSchema` has no metric, so cosine always), one `_bulk` request per `add_docs` call (no batching knob), no index template/ILM, array equality translated as `terms` membership, and `$contains`, object equality, `null` equality, and non-scalar `terms` candidates rejected with `Error::Validation` pointing at the in-memory backend (see the `elasticsearch/client.rs` module docs for the exact filter subset).
- Elasticsearch error reporting never masks the status: a non-2xx answer maps to the typed error built from the status and the body text, and a body that cannot be read contributes `<the response body could not be read: …>` plus a `debug` line rather than an empty detail; a stored `_meta.openjiuwen_schema` that does not deserialize is reported at `warn` and the schema is rebuilt from the mapping's properties.
- Elasticsearch document operations reuse the kernel `ElasticsearchDocumentStore` unchanged: same index naming, same probe-then-create (`HEAD` then `PUT`) on first upsert, same filter translation with a bounded 10 000-document local fallback for inexpressible filters, and the same error mapping.
- Gauss/PostgreSQL document store is not surfaced: the kernel's document contract replaces the SQLAlchemy engine (`docs/PARITY.md` deviation), so there is no `GaussDbStore` analogue.
- Redis TLS and cluster routing are implemented by the in-crate client: `rediss://` verifies the server against the web PKI (`tokio-rustls` + `webpki-roots`, both locked offline), `redis+cluster://` verifies the seed with `CLUSTER INFO`/`CLUSTER SLOTS` and follows one `MOVED`/`ASK` redirection per command; a non-cluster node in cluster mode is a `Validation` error, and a second consecutive redirection is a `Transport` error.
- The `sqlite` vector backend is the reverse delegation: SQLite owns collection schemas, metrics, documents, and embeddings, while a throwaway `InMemoryVectorStore` owns document preparation (validation, `default_value` filling, `auto_id` minting), so a persisted document is exactly the reference-prepared one.

## Constraints

- Delegation only for kernel backends: memory/file/sqlite are constructed through the kernel (`InMemoryKvStore::new`, `LocalFileKvStore::new(root).await`, `SqliteKvStore::connect(url).await`, and the document/vector equivalents); Elasticsearch document/vector construction goes through the kernel's `ElasticsearchDocumentStore` and this crate's `ElasticsearchVectorStore`.
- The `sqlite` feature forwards to `openjiuwen-foundation/sqlite`; without it a `Sqlite` backend returns the kernel's `Error::Unsupported` for key-value/documents and this crate's typed `Unsupported` for vectors (the one SQLite surface that only exists with the feature).
- The `redis` feature adds no dependency: the backends use the in-crate RESP client (`crate::redis`) over `tokio::net::TcpStream`, and the `redis` module is absent when the feature is off.
- Elasticsearch adds no client dependency: it speaks REST through `reqwest` (already a crate dependency) in `elasticsearch/`, and documents reuse the kernel's `openjiuwen_common::http` client.
- Errors: `Error::Validation` (config, kind, value encoding, collection-name/dimension/credential problems), `Error::Unsupported` (deferred capability or missing feature), `Error::Transport`/`Error::Timeout` (server I/O), `Error::NotFound` (unknown vector collection/index).
- Credentials never reach `Debug` output or logs: `RedisStore`'s `Debug` redacts `connection_args` values and URL user-info, `StoreBackend::Elasticsearch` keeps credentials in plain `Option<String>` but is only rendered through derived `Debug` of configuration values a caller supplied, and the kernel `ElasticsearchConfig` stores them as `SecretString`.
- Files ≤ ~1000 lines: `mod.rs` (factories + config), `redis/` split into `mod.rs` (config, connection, key layout, helpers), `kv.rs`, `document.rs`, `vector.rs`, `tests/`, `elasticsearch/` split into `mod.rs`, `client.rs`, `tests.rs`, `sqlite_vector/` split into `mod.rs` (tables, `VectorStore` impl, helpers) and `tests.rs`.

## Redis Backend Notes

- The client is `crate::redis::{Client, ConnectionOptions, Reply}`; it speaks RESP2/RESP3 directly, supports `command`, `command_str`, `pipeline`, and `ping`, follows one `MOVED`/`ASK` redirection per command, maps `WRONGTYPE` to `Validation` and every other server error to `Transport`, and never logs credentials.
- `ConnectionOptions::from_url` parses `redis[s][+cluster]://[user:pass@]host:port[/db]` (percent-decoding user-info), so `rediss://` selects TLS and `+cluster` selects cluster mode; `with_connection_args` applies `username`/`password`/`db`/`timeout` and ignores unknown keys.
- All three stores share `pub(crate) RedisConnection` (config + `OnceCell<Client>`): `client`, `ping`, `run`, `pipeline`, plus `get_bytes`/`set_bytes`/`mget_bytes`/`smembers`/`sadd`/`srem`/`delete_keys`.
- Key layout (every component percent-escaped, `%` → `%25`, `:` → `%3A`, so names can never blur a key boundary):
  - `openjiuwen:doc:collections` (set), `openjiuwen:doc:ids:{collection}` (set), `openjiuwen:doc:doc:{collection}:{id}` (JSON `Document`).
  - `openjiuwen:vec:collections` (set), `openjiuwen:vec:schema:{collection}` (JSON `CollectionSchema`), `openjiuwen:vec:ids:{collection}` (set), `openjiuwen:vec:doc:{collection}:{id}` (JSON document).
- Documents are addressed through their id set, never through `SCAN`, so ids holding glob metacharacters are safe. There is no TTL anywhere: neither kernel contract has an expiry knob, so none is invented.
- Reference delegation (the reason semantics cannot drift): `RedisDocumentStore` hydrates a throwaway `InMemoryDocumentStore` and delegates `query`/`count`/`delete`, so filter, ordering, and offset-then-limit behavior are the kernel's (`run_query` is private). `RedisVectorStore` hydrates a throwaway `InMemoryVectorStore` and delegates `add_docs`, `search` (via `search_with_metric` with the store metric), and both deletes; an add reads the reference-prepared documents back with a zero query vector and `top_k = existing + added`, so minted auto ids and filled defaults persist, and a failed add persists nothing.
- Consequences to know: O(n) read of a collection per operation (no RediSearch module offline, so ranking is brute force client-side exactly like the in-memory index); single-writer per collection (no cross-process locking, no multi-key transaction, so an interrupted batch can leave a partially written view); the metric is store-level because `CollectionSchema` carries none.
- Collection names are validated with a replica of the kernel's `pub(crate) validate_collection` rule (non-empty, no `/`, `\`, NUL, not `.`/`..`) shared by both non-KV stores; a missing collection reads as `None`/empty/`0`, duplicate collection creation is `Validation`, unknown vector collections are `NotFound`.
- `collection_exists`/`get_schema` read the schema key rather than `SISMEMBER`, which the crate-boundary fake does not implement. The stores use only the commands both fakes implement: `PING`, `GET`, `SET`, `EXISTS`, `DEL`/`UNLINK`, `MGET`, `TTL`/`PTTL`, `EXPIRE`, `SCAN`, `KEYS`, `SADD`/`SREM`/`SMEMBERS`, `ZCARD`/`ZADD`/`ZRANGE`/`ZREM`, `TYPE`, `FLUSHDB`, plus the handshake.

## Elasticsearch Backend Notes

- Vector REST surface (`elasticsearch/client.rs`): `PUT/HEAD/DELETE /{index}`, `GET /{index}/_mapping`, `POST /_bulk?refresh=false`, `POST /{index}/_search`, `POST /{index}/_delete_by_query?refresh=true`, `GET /_cat/indices/{prefix}__*?format=json&h=index`; 404 → `NotFound`, 400/409 → `Validation`, 401/403 → `Validation` without echoing credentials, other non-2xx → `Transport` with the Elasticsearch `error.reason`, and `knn.filter` is only sent for a non-empty filter.
- The kernel `validate_collection` helper is `pub(crate)` in `openjiuwen-foundation`, so the vector backend re-implements collection-name validation.
- Elasticsearch vector parity traps: `create_collection` on an existing name must fail with `Validation` (unlike document collections), `auto_id`/`default_value` filling is the store's job, `delete_docs_by_filters` with `null`/`{}` deletes everything, and `CollectionSchema` does not round-trip through an index mapping so it needs `_meta` metadata.
- Document backend wiring lives in `mod.rs::elasticsearch_document_config`: the URL is trimmed and must be `http://`/`https://` with a host; userinfo becomes HTTP Basic auth, an explicit `username`/`password` replaces it field by field, and a password without a user name is not sent (the kernel only applies Basic auth when both are present); `api_key` is forwarded as `ApiKey <key>` and the kernel ranks it below Basic auth; userinfo, query, fragment, and trailing `/` are stripped from the endpoint; `index_prefix` is forwarded unchanged.
- Document wire contract (asserted by the crate-boundary suite, index `"{prefix}__{collection}"`): `HEAD {index}` probe → `PUT {index}` with `{"mappings":{"properties":{"id":{"type":"keyword"}}}}` on 404 → `POST {index}/_bulk` NDJSON; `GET {index}/_doc/{id}`; `POST {index}/_search` with `query`/`from`/`size`/`track_total_hits`/`sort`; `POST {index}/_count`; `POST {index}/_delete_by_query`; `GET {prefix}__*` for `list_collections`; `DELETE {index}`.
- Document adapter kernel-contract details: `upsert` returns `documents.len()` (not the bulk item count), `list_collections` uses the object-keyed `GET /{prefix}__*` listing (not `_cat/indices`), and only `term`/`terms`/`bool`/`range` filters are translated (dotted paths fall back to the bounded local scan). A hostless URL never reaches the explicit host check because `url::Url` already rejects it.

## SQLite Vector Backend Notes

- Scope: `sqlite_vector::SqliteVectorStore` implements the whole kernel `VectorStore` contract (`create_collection`, `delete_collection`, `collection_exists`, `get_schema`, `add_docs`, `search`, `delete_docs_by_ids`, `delete_docs_by_filters`, `list_collection_names`) plus `count`, `metric`, `with_metric`, `pool`, `connect`, and `from_pool`.
- Row layout: three fixed tables created by every constructor (`openjiuwen_vector_collections(name, schema_json, metric)`, `openjiuwen_vector_documents(collection, id, doc_json)`, `openjiuwen_vector_embeddings(collection, id, field, dim, embedding)`); no per-collection DDL, so a collection name is a value and is never quoted into SQL.
- Blob layout: `embedding` holds exactly `dim` little-endian `f32` components written with `f32::to_le_bytes` (`4 * dim` bytes), one row per vector field of a document, so a schema with several `FLOAT_VECTOR` fields stores one blob each and no embedding ever round-trips through text.
- Table names carry an `openjiuwen_` prefix because SQLite reserves the `sqlite_` prefix for its own internal objects.
- Brute-force limitation: there is no ANN index, so `search` reads every embedding row of the target field, joins it with its document row, scores it in Rust, sorts (score descending, primary key ascending), and truncates to `top_k`; a query is linear in the collection size, with no quantization, clustering, or approximate recall.
- Reference parity: scores and ranking come from the kernel `Metric` implementation and the kernel filter language, and documents are prepared by a throwaway `InMemoryVectorStore`, so the backend cannot drift from the in-memory reference on scoring, filter matching, ordering, `top_k`, defaults, or auto ids.
- Metric policy: the metric is recorded per collection at `create_collection` from `SqliteVectorStore::metric` (cosine unless `with_metric` changed it); a store opened with a different metric refuses both reads and writes with `Error::Validation` instead of ranking the collection differently.
- Error taxonomy: `Error::NotFound` for every operation on an unknown collection (`delete_collection` is a no-op and `get_schema`/`collection_exists` report the absence); `Error::Validation` for an invalid schema or dimension, an unusable collection name, a duplicate primary key, a mistyped or missing field, a query vector whose length disagrees with the collection's `dim`, a mismatched or undecodable metric, and a filter that is neither `null` nor an object or that names an unknown operator; `Error::External` for a `sqlx` failure or a stored row that no longer matches the layout (invalid JSON, an unparseable schema, a blob whose length is not `4 * dim`).
- The filter validator is stricter than the kernel matcher on purpose: a malformed filter is a `Validation` error here instead of silently matching nothing, while every well-formed filter (implicit equality, `$eq`/`$ne`/`$gt`/`$gte`/`$lt`/`$lte`/`$in`/`$contains`, and `$and`/`$or`/`$not`) behaves exactly as in the reference.
- Single-writer per collection is assumed, as with the Redis vector store, but each mutation runs in one SQLite transaction, so an interrupted batch never leaves a half-written document behind.

## Test Strategy

- `tests.rs` unit suite (child module of `mod.rs`, `use super::*`): `StoreBackend::from_json` for every kind, every credential field, and unknown/non-string input; serde round-trips (including the minimal ES object); memory KV/document/vector factory round trips; file KV under a `tempfile::TempDir`; sqlite branch under `--features sqlite`.
- ES factory unit tests in `tests.rs`: URL trimming/host/prefix forwarding, credential precedence (userinfo-only, explicit-wins, empty password, orphan password), URL validation, credential forwarding into the built store's `config()`, and the KV arm staying `Unsupported`.
- Redis factory unit tests in `tests.rs`: all three factories report a missing `url` as `Validation` naming redis; with the feature on, a fake-server round trip covers documents (create/upsert/list/filtered query) and vectors (schema round trip, add, search ordering); with the feature off, all three report `Unsupported` naming the `redis` feature.
- `redis/tests/{kv,document,vector}.rs` (unit, `crate::redis::testsupport::FakeRedis`): KV config validation, encoding, prefix/scan, pipeline, and connection failures; document lifecycle, upsert/get/replace, `query` matching the reference window, `count` ignoring paging, delete, empty-collection reads, name validation, key cross-talk, dangling id handling; vector schema round trip, duplicate/invalid collection, list/delete/exists, defaults + auto ids, duplicate keys persisting nothing, score/tie-break/`top_k`, reference-result parity, filters, metric change, deletes, `NotFound`, connection failures.
- `../../tests/redis.rs` (crate boundary, `tests/redis/fake.rs` peer): codec, KV store, checkpointer, TTL, factory/deferral, plus the document and vector suites (`document_store_round_trips_collections_and_documents`, `document_store_query_windows_match_the_reference_store`, `document_store_rejects_invalid_names_and_unreachable_peers`, `vector_store_round_trips_schema_documents_and_searches`, `vector_store_metric_defaults_to_cosine_and_can_be_changed`, `vector_store_rejects_invalid_collections_and_unreachable_peers`). That binary is intentionally one file per adapter family and runs past 1000 lines because the peer fake and every surface share one process; splitting it would need a separate `tests/redis/` module file.
- `elasticsearch/tests.rs` covers configuration, index naming, the mapping builder and its schema round-trip, the filter-translation truth table, score clamping, and result ordering.
- `../../tests/elasticsearch.rs` drives every REST operation against a `wiremock` server: the vector suite (create/exists/schema/delete/list, 409, NDJSON bulk, k-NN request and hit mapping, 404/503/400 mapping, dimensions, duplicate keys, delete-by-query) and the document suite (`document_upsert_creates_the_index_and_bulks_ndjson`, `document_upsert_reuses_an_existing_index`, `document_upsert_surfaces_item_level_errors`, `document_get_reads_a_hit_and_a_miss`, `document_query_sends_the_filter_order_and_paging`, `document_count_sends_the_translated_filter`, `document_delete_uses_delete_by_query`, `document_list_collections_strips_the_prefix`, `document_drop_collection_ignores_a_missing_index`, `document_missing_index_reads_as_absent`, `document_server_failure_is_a_transport_error`, `document_store_factory_round_trips_a_document`, and the three credential-header tests).
- `sqlite_vector/tests.rs` unit suite (`#![cfg(feature = "sqlite")]` through the module gate, `use super::*`): create/add/search over a `tempfile` database file (including reopening it), score/ranking/field parity against `InMemoryVectorStore` for cosine, dot, and euclidean on filtered and truncated queries, `top_k` truncation plus primary-key tie-breaking, malformed-filter rejections, delete/count/list/drop/null-filter deletion, dimension and unknown-vector-field validation, `NotFound` for every operation on a missing collection, the metric-mismatch and unknown-metric errors, unusable collection names, duplicate primary keys writing nothing, reference-filled defaults and auto ids, multi-vector-field blobs, corrupt-blob/corrupt-document `External` errors, and the little-endian encode/decode round trip.
- `../../tests/adapters.rs` adds `sqlite_vector_store_factory_round_trips_a_search` under `#[cfg(feature = "sqlite")]` (factory-built store over a `TempDir` database: create, add, search, list), and `tests.rs` keeps `sqlite_vector_store_requires_the_feature` for the default build's typed `Unsupported`.
- Python counterparts named in comments: `tests/unit_tests/extensions/checkpointer/test_redis_checkpointer.py`, `tests/unit_tests/extensions/store/test_es_vector_store.py`, `test_gauss_db.py` (behavior not ported — noted in the test comment); the Redis document/vector, SQLite vector, and Elasticsearch document backends have no Python counterpart and say so.

## Routing Table

- `./mod.rs` → `StoreBackend`/`StoreKind`, the three factories, sqlite/redis/elasticsearch factory helpers
- `./tests.rs` → the `mod.rs` factory unit tests (child `tests` module declared in `mod.rs`)
- `./redis/` → Redis family: `mod.rs` (config, URL normalisation, shared `RedisConnection`, key layout, collection validation, re-exports), `kv.rs` (key-value + batched pipeline), `document.rs` (`RedisDocumentStore`), `vector.rs` (`RedisVectorStore`), `tests/` (per-store unit suites)
- `./sqlite_vector/` → brute-force SQLite vector store (`sqlite` feature): `mod.rs` (three-table layout, `SqliteVectorStore` and its `VectorStore` impl, filter validation, reference preparation), `tests.rs` (unit suite)
- `./elasticsearch/` → live REST vector store: `mod.rs` (config, `VectorStore` impl, document validation), `client.rs` (REST client, mapping builder, filter translation), `tests.rs` (unit tests)
