# store/ — backend selection, the live Redis key-value store, and the Elasticsearch vector backend

## Intent

Python's `extensions/store/` contributes only optional-driver backends (`RedisStore`, `ElasticsearchVectorStore`, `GaussDbStore`) on top of the kernel's storage contracts; memory/file/sqlite live in `openjiuwen_foundation::store`.
Rust mirrors that: a config-driven factory over the kernel backends, a live single-node Redis key-value backend over the crate's in-house RESP client (`src/redis/`), and an Elasticsearch vector backend.
Pick this node for "which KV/document/vector store do I build from config", for the Redis key/value encoding contract, and for the remaining vendor-backend deferrals.

## API Surface

```rust
pub use openjiuwen_core::foundation::store::{
    KvStore, KvValue, DocumentStore, Document, Query, VectorStore, VectorQuery, VectorSearchResult,
    CollectionSchema, InMemoryKvStore, LocalFileKvStore, InMemoryDocumentStore, LocalFileDocumentStore,
    InMemoryVectorStore, LocalFileVectorStore,
};
#[cfg(feature = "sqlite")]
pub use openjiuwen_core::foundation::store::{SqliteKvStore, SqliteDocumentStore};

pub enum StoreBackend {
    Memory,
    File { root: PathBuf },
    Sqlite { url: String },
    Redis { url: Option<String> },
    Elasticsearch { url: String, index_prefix: String },
}
impl StoreBackend {
    pub fn kind(&self) -> StoreKind;                 // Memory | File | Sqlite | Redis | Elasticsearch
    pub fn from_json(value: &Value) -> Result<Self>; // tolerant snake_case decoder; unknown kind = Error::Validation
}

pub async fn build_kv_store(backend: &StoreBackend) -> Result<Arc<dyn KvStore>>;
pub async fn build_document_store(backend: &StoreBackend) -> Result<Arc<dyn DocumentStore>>;
pub async fn build_vector_store(backend: &StoreBackend) -> Result<Arc<dyn VectorStore>>;

/// `redis` feature only.
pub mod redis {
    pub struct RedisStoreConfig { pub url: Option<String>, pub cluster_mode: Option<bool>,
                                  pub connection_args: BTreeMap<String, Value> }
    pub struct RedisStore { config: RedisStoreConfig, client: tokio::sync::OnceCell<Client> }
    impl RedisStore {
        pub fn new(config: RedisStoreConfig) -> Result<Self>;                        // validation only
        pub async fn connect(config: RedisStoreConfig) -> Result<Arc<dyn KvStore>>;  // connects + PINGs
        pub fn config(&self) -> &RedisStoreConfig;
        pub async fn ping(&self) -> Result<()>;
    }
    // `KvStore`: set, get, exists, delete, exclusive_set, get_by_prefix, delete_by_prefix,
    // mget, batch_delete, pipeline (a real batched `Client::pipeline`, not `StorePipeline`).
}

/// Elasticsearch vector store surface (no feature).
pub struct ElasticsearchVectorStoreConfig { pub url: String, pub index_prefix: String, pub dimension: usize }
pub struct ElasticsearchVectorStore { .. }
impl ElasticsearchVectorStore {
    pub fn new(config: ElasticsearchVectorStoreConfig) -> Result<Self>;
    pub async fn connect(config: ElasticsearchVectorStoreConfig) -> Result<Arc<dyn VectorStore>>;
    pub fn index_name(&self, collection: &str) -> String;  // "{index_prefix}__{collection}"
}
```

## Ported Behavior

- Backend selection is config-driven exactly like the Python `type` discriminators (`memory`, `file`, `sqlite`, `redis`, `elasticsearch`).
- `RedisStore::new` only validates and normalizes; the `OnceCell` client opens on first use, and `connect` opens eagerly and verifies the connection with `PING`, so a bad URL, a closed port, or a redirect-only server fails before the first write.
- Keys are used verbatim: Python's `index_prefix`-style namespacing stays the caller's responsibility, and the factory passes `StoreBackend::Redis::url` straight into `RedisStoreConfig`.
- Values are stored as a one-byte kind tag (`0x00` text, `0x01` bytes) followed by the raw payload, so a `KvValue` keeps its `Text`/`Bytes` variant across a round trip; this is an intentional deviation from Python's bare-payload `RedisStore`, so Python-written values are not directly readable (and vice versa).
- A payload with a missing or unknown kind tag is `Error::Validation` naming the key; keys and values are never logged.
- `set` issues a plain `SET` (which clears any TTL), `get` issues `GET` (`Null` → `None`), `exists` reads `EXISTS > 0`, and `delete` issues `DEL`.
- `exclusive_set` issues `SET key value NX [PX <ms>]`, returns `false` on `Null`, and rounds a sub-millisecond expiry up to 1 ms so a near-zero TTL never means "no expiry".
- `mget` sends one `MGET`, preserving input order with `None` per miss; an empty input returns an empty vector without a round trip.
- `get_by_prefix` and `delete_by_prefix` walk `SCAN` with `MATCH <escaped-prefix>*` and `COUNT 500`, escaping `*`, `?`, `[`, `]`, and `\` in the prefix, then apply `MGET`/`UNLINK` in 500-key chunks; keys that vanish between `SCAN` and `MGET` are skipped.
- `delete_by_prefix` and `batch_delete` sum the integer replies, so only live keys count (duplicates and dead keys count once), and delete with `UNLINK` falling back to `DEL` when the server answers an unknown-command error.
- `pipeline()` returns a real batched `KvPipeline`: commands are queued synchronously, sent through one `Client::pipeline` write-and-read, results are mapped in order (`set` → `None`, `get` → decoded option, `exists` → `"true"`/`"false"`), and the queue is cleared so reuse cannot replay old operations.
- The kernel `StorePipeline` is deliberately not used: it issues one round trip per operation, while batching is the point of the Redis pipeline.
- `RedisStoreConfig` accepts `url` **or** `connection_args`; the Rust surface validates the URL scheme (`redis://`, `rediss://`, `redis+cluster://`, `rediss+cluster://`), infers `cluster_mode` from `+cluster`, and builds the client options from the URL plus `host`/`port` overrides and `username`/`password`/`db`/`timeout`.
- Elasticsearch defaults: index = `"{index_prefix}__{collection}"`, vector field default `768` dimensions, cosine similarity.
- Gauss/PostgreSQL document store is not surfaced: the kernel's document contract replaces the SQLAlchemy engine (`docs/PARITY.md` deviation), so there is no `GaussDbStore` analogue.
- Remaining Redis deferrals: `rediss://` TLS and real cluster slot routing (both `Error::Unsupported` from the client), plus no document or vector surface.
- A backend without a surface for the requested factory returns `Error::Unsupported` naming the backend and the supported alternatives; no method silently succeeds.

## Constraints

- Delegation only for kernel backends: memory/file/sqlite are constructed through the kernel (`InMemoryKvStore::new`, `LocalFileKvStore::new(root).await`, `SqliteKvStore::connect(url).await`, and the document/vector equivalents).
- The `sqlite` feature forwards to `openjiuwen-foundation/sqlite`; without it a `Sqlite` backend returns the kernel's `Error::Unsupported`.
- The `redis` feature adds no dependency: the backend uses the in-crate RESP client (`crate::redis`) over `tokio::net::TcpStream`, and the `redis` module is absent when the feature is off.
- Elasticsearch adds no client dependency today.
- Errors: `Error::Validation` (config, kind, value encoding), `Error::Unsupported` (deferred capability or missing feature), and `Error::Transport`/`Error::Timeout` (server I/O).
- `RedisStore` is `Send + Sync` but not `Clone` (the `OnceCell` connection is shared through `Arc<dyn KvStore>`); its `Debug` redacts `connection_args` values and URL user-info.
- Files ≤ ~1000 lines; one file per backend family, with the Redis unit tests in `./redis/tests.rs`.

## Redis Backend Notes

- The client is `crate::redis::{Client, ConnectionOptions, Reply}`; it speaks RESP2/RESP3 directly, supports `command`, `command_str`, `pipeline`, and `ping`, maps `MOVED`/`ASK` to `Unsupported` and `WRONGTYPE` to `Validation`, and never logs credentials.
- `ConnectionOptions::from_url` parses `redis://[user:pass@]host:port[/db]` (percent-decoding user-info) and rejects `rediss://` with an `Unsupported` TLS error; `with_connection_args` applies `username`/`password`/`db`/`timeout` and ignores unknown keys.
- The kernel contract puts TTL only on `exclusive_set` (`Option<Duration>`, relative) and requires `set` to clear any TTL, so `SET` plus `SET … NX PX` covers it without Lua.
- `StoreBackend::Redis` carries only `url`, so the factory constructs `RedisStoreConfig { url, ..Default::default() }`; `cluster_mode` and `connection_args` remain available when building a `RedisStore` directly.
- The documents and vectors factories answer `Unsupported` for `Redis`, and `RedisStore` has no document or vector surface at all.
- `tests/adapters.rs` has no Redis store assertions, so the live backend does not change its expectations; the deferral contract is covered by the inline factory tests instead.
- The kernel `validate_collection` helper is `pub(crate)` in `openjiuwen-foundation`, so a live Elasticsearch backend must re-implement collection-name validation.
- Elasticsearch parity traps: `create_collection` on an existing name must fail with `Validation` (unlike document collections), `auto_id`/`default_value` filling is the store's job, `delete_docs_by_filters` with `null`/`{}` deletes everything, and `CollectionSchema` does not round-trip through an index mapping so it needs a metadata document.

## Test Strategy

- Unit tests beside `mod.rs`: `StoreBackend::from_json` for every kind and an unknown kind; memory KV set/get round-trip through the factory; file KV under a `tempfile::TempDir`; sqlite branch under `--features sqlite`; the Elasticsearch backend's `Unsupported` answers.
- Redis factory tests in `mod.rs`: a fake-server round trip (`set`/`get`/`exists`/`get_by_prefix`/`mget`/`pipeline`/`delete_by_prefix`/`delete`), a missing `url` as `ErrorKind::Validation`, document/vector factories as `Unsupported` naming redis, and the feature-off KV factory as `Unsupported` naming the `redis` feature.
- `./redis/tests.rs`: the Python config-validation assertions (every accepted scheme, invalid scheme, url-or-connection_args, `cluster_mode` inference, serde round trip) plus `host`/`port` overrides and invalid types, `Debug` redaction, glob escaping, expiry rounding, and fake-server behavior.
- Fake-server behavior: text and binary round trips (kind preserved, including UTF-8 written as bytes), missing key, `delete`, `exclusive_set` NX/TTL with `set` clearing the TTL, `mget` order, prefix read/delete counting live keys, `batch_delete` counting, pipeline ordering and queue clearing, and connect failures (closed port `Transport`, `rediss://` `Unsupported` naming TLS, MOVED cluster `Unsupported`).
- Python counterparts named in comments: `tests/unit_tests/extensions/checkpointer/test_redis_checkpointer.py`, `tests/unit_tests/extensions/store/test_es_vector_store.py`, `test_gauss_db.py` (behavior not ported — noted in the test comment).

## Routing Table

- `./mod.rs` → `StoreBackend`, `StoreKind`, three factories (memory/file/sqlite/redis/elasticsearch), redis factory helpers, tests
- `./redis.rs` → live Redis key-value backend: config normalization, client wiring, `KvStore` methods, batched pipeline
- `./redis/tests.rs` → `FakeRedis`-backed unit tests for the Redis backend
- `./elasticsearch.rs` → Elasticsearch vector-store surface (`ElasticsearchVectorStoreConfig`, `ElasticsearchVectorStore`)
