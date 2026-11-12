# store/ — storage backend selection and deferred Redis/Elasticsearch backends

## Intent

Python's `extensions/store/` contributes only optional-driver backends (`RedisStore`, `ElasticsearchVectorStore`, `GaussDbStore`) on top of the kernel's storage contracts; memory/file/sqlite live in `openjiuwen_foundation::store`.
Rust mirrors that: a config-driven factory over the kernel backends, plus a Redis key-value surface and an Elasticsearch vector surface, both deferred (`Error::Unsupported`) with the Python constructors preserved.
Pick this node for "which KV/document/vector store do I build from config" and for the deferred vendor backends.

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
    pub struct RedisStore { .. }
    impl RedisStore {
        pub fn new(config: RedisStoreConfig) -> Result<Self>;          // scheme validation
        pub async fn connect(config: RedisStoreConfig) -> Result<Arc<dyn KvStore>>;  // Error::Unsupported
    }
    #[async_trait] impl KvStore for RedisStore { /* every method -> Error::Unsupported */ }
}

/// Elasticsearch vector store (no feature; always deferred).
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
- `RedisStore` keeps Python's method list and constants: `index_prefix`-style key prefixing is the caller's, `exclusive_set(key, value, expiry)` maps to the kernel's `KvValue`/`Duration` signature, and Python's 500-key chunking for `delete_by_prefix`/`batch_delete` is documented in `redis.rs` because the kernel `KvStore` trait takes the whole prefix or key set instead of a `batch_size` parameter.
- `RedisStoreConfig` accepts `url` **or** `connection_args` in Python; the Rust surface validates the URL scheme (`redis://`, `rediss://`, `redis+cluster://`, `rediss+cluster://`) and records `cluster_mode` (inferred from `+cluster`).
- Elasticsearch defaults: index = `"{index_prefix}__{collection}"`, vector field default `768` dimensions, cosine similarity — documented, not implemented.
- Gauss/PostgreSQL document store is not surfaced: the kernel's document contract replaces the SQLAlchemy engine (`docs/PARITY.md` deviation), so there is no `GaussDbStore` analogue.
- Every deferred backend returns `Error::Unsupported` naming the backend and the supported alternatives; no method silently succeeds.

## Constraints

- Delegation only: memory/file/sqlite are constructed through the kernel (`InMemoryKvStore::new`, `LocalFileKvStore::new(root).await`, `SqliteKvStore::connect(url).await`, and the document/vector equivalents); this crate adds no storage behavior.
- The `sqlite` feature forwards to `openjiuwen-foundation/sqlite`; without it a `Sqlite` backend returns the kernel's `Error::Unsupported`.
- Redis/Elasticsearch add no client dependency; the `redis` feature is API-only.
- Errors: `Error::Validation` (config/kind), `Error::Unsupported` (deferred backend); path/URL validation before any I/O.
- Files ≤ ~1000 lines; one file per backend family.

## Test Strategy

- Unit tests beside `mod.rs`: `StoreBackend::from_json` for every kind and an unknown kind; memory KV set/get round-trip through the factory; file KV under a `tempfile::TempDir`; sqlite branch under `--features sqlite`; `StoreBackend::Redis`/`Elasticsearch` → `ErrorKind::Unsupported` with the message naming the backend.
- `redis` tests: scheme validation, `cluster_mode` inference, and every `KvStore` method returning `ErrorKind::Unsupported`.
- Python counterparts named in comments: `tests/unit_tests/extensions/store/test_es_vector_store.py`, `test_gauss_db.py` (behavior not ported — noted in the test comment).

## Routing Table

- `./mod.rs` → `StoreBackend`, `StoreKind`, three factories, tests
- `./redis.rs` → `redis` feature key-value surface (`RedisStoreConfig`, `RedisStore`)
- `./elasticsearch.rs` → deferred vector-store surface (`ElasticsearchVectorStoreConfig`, `ElasticsearchVectorStore`)
