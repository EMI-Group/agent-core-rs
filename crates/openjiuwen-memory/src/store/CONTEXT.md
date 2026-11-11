# store

## Intent

Persistence for memory records, embeddings, variables, scope configs, and message history on top of the `openjiuwen-foundation` storage traits.
Python stores records in KV + vectors in per-type collections + messages in SQL tables; this module keeps the same logical layout on `KvStore`, `VectorStore`, `DocumentStore`, and `GraphStore`.
No ranking, no extraction, no engine policy live here: only key derivation, serialization, and store calls.

## API Surface

### `keys.rs`
```rust
pub const RECORD_PREFIX: &str = "UMD";                     // Python KV namespace
pub const VARIABLE_PREFIX: &str = "user_var";
pub const SESSION_VARIABLE_PREFIX: &str = "session_var";
pub const SCOPE_CONFIG_PREFIX: &str = "memory_scope_config";
pub const MESSAGES_COLLECTION: &str = "messages";

pub fn record_key(scope: &MemoryScope, id: &str) -> String;     // "UMD/{user}/{scope}/{id}"
pub fn record_prefix(scope: &MemoryScope) -> String;            // "UMD/{user}/{scope}/"
pub fn record_prefix_for(user_id: &str) -> String;              // "UMD/{user}/"
pub fn record_prefix_for_scope(scope_id: &str) -> String;       // "UMD/" (scope-wide scan; filter afterwards)
pub fn variable_key(scope: &MemoryScope, name: &str) -> String; // "user_var/{user}/{scope}/{name}"
pub fn variable_prefix(scope: &MemoryScope) -> String;
pub fn session_variable_key(scope: &MemoryScope, session_id: &str, name: &str) -> String;
pub fn session_variable_prefix(scope: &MemoryScope, session_id: &str) -> String;
pub fn scope_config_key(scope_id: &str) -> String;
pub fn vector_collection(scope: &MemoryScope, kind: MemoryType) -> String;
pub fn sanitize_segment(value: &str) -> String;                 // [A-Za-z0-9._-] only + short FNV hash suffix when changed
```
`vector_collection` follows Python `uid_{user}_gid_{scope}_mtype_{kind}` with each segment sanitized; an unchanged safe segment keeps its readable form, otherwise a 8-hex FNV-1a suffix is appended to avoid collisions. Collection names must satisfy `LocalFileVectorStore` (`[A-Za-z0-9._-]`, no `.`/`..`).

### `mod.rs`
```rust
#[derive(Clone)]
pub struct MemoryStores {
    pub kv: Arc<dyn KvStore>,
    pub vector: Arc<dyn VectorStore>,
    pub document: Arc<dyn DocumentStore>,
    pub graph: Option<Arc<dyn GraphStore>>,
}
impl MemoryStores {
    pub fn in_memory() -> Self;                                      // includes InMemoryGraphStore
    pub async fn local_file(root: impl AsRef<Path>) -> Result<Self>;// kv/, vector/, documents/, graph/ subdirs
    #[cfg(feature = "sqlite")] pub async fn sqlite(path: impl AsRef<Path>) -> Result<Self>; // SqliteKvStore + SqliteDocumentStore + LocalFileVectorStore under "<path>.vectors/" + LocalFileGraphStore under "<path>.graph/"
    pub fn from_parts(kv: Arc<dyn KvStore>, vector: Arc<dyn VectorStore>, document: Arc<dyn DocumentStore>) -> Self; // graph = None
    pub fn with_graph(self, graph: Arc<dyn GraphStore>) -> Self;
}
```
`local_file` creates `<root>/kv`, `<root>/documents`, `<root>/graph` via the foundation constructors and `<root>/vector` for vectors. `sqlite` uses `SqliteKvStore::connect(&format!("sqlite:{}", path.display()))`, `SqliteDocumentStore::connect(...)`; the vector/graph snapshots use `LocalFileVectorStore::new("<path>.vectors")` and `LocalFileGraphStore::new("<path>.graph")`. A `MemoryStores::sqlite` test must reopen the same path and observe committed records.

### `storage.rs` (crate-internal)
```rust
pub(crate) struct MemoryStorage { /* kv, vector, document, providers */ }
impl MemoryStorage {
    pub fn new(stores: MemoryStores) -> Self;
    pub fn stores(&self) -> &MemoryStores;

    pub async fn put_record(&self, record: &MemoryRecord) -> Result<()>;
    pub async fn get_record(&self, scope: &MemoryScope, id: &str) -> Result<Option<MemoryRecord>>;
    pub async fn list_records(&self, filter: &MemoryFilter) -> Result<Vec<MemoryRecord>>;
    pub async fn delete_record(&self, record: &MemoryRecord) -> Result<bool>; // record + embedding

    pub async fn index_embedding(&self, record: &MemoryRecord, embedding: &[f32]) -> Result<()>;
    pub async fn delete_embedding(&self, record: &MemoryRecord) -> Result<()>;
    pub async fn delete_embeddings_for(&self, filter: &MemoryFilter) -> Result<usize>;
    pub async fn search_embeddings(&self, scope: &MemoryScope, kind: MemoryType,
                                   vector: &[f32], top_k: usize) -> Result<Vec<(String, f32)>>;

    pub async fn set_variable(&self, scope: &MemoryScope, session_id: Option<&str>, name: &str, value: &str) -> Result<()>;
    pub async fn get_variable(&self, scope: &MemoryScope, session_id: Option<&str>, name: &str) -> Result<Option<String>>;
    pub async fn list_variables(&self, scope: &MemoryScope, session_id: Option<&str>) -> Result<BTreeMap<String, String>>;
    pub async fn delete_variable(&self, scope: &MemoryScope, session_id: Option<&str>, name: &str) -> Result<bool>;

    pub async fn set_scope_config(&self, scope_id: &str, config: &MemoryScopeConfig) -> Result<bool>;
    pub async fn get_scope_config(&self, scope_id: &str) -> Result<Option<MemoryScopeConfig>>;
    pub async fn delete_scope_config(&self, scope_id: &str) -> Result<bool>;
}
```
Semantics:
- `put_record` serializes `record` with `embedding: None` (the embedding column is authoritative) to the record key. The scope is already flattened in the JSON.
- `list_records` chooses the narrowest prefix from `filter.user_id`/`scope_id` (user+scope when both, user when only user, whole `UMD/` when neither), calls `kv.get_by_prefix`, parses each value, ignores malformed entries with `tracing::warn!` (a corrupt record never fails a listing), then applies `MemoryFilter::matches` and sorts by `(timestamp desc, id asc)`.
- `delete_record` deletes the KV key and the vector id; returns whether the KV key existed.
- Embeddings: the vector collection schema is `id VARCHAR(256) PRIMARY KEY` + `embedding FLOAT_VECTOR(dim)`; `index_embedding` ensures the collection exists (create on first use), deletes the id if present, then `add_docs` (foundation is insert-only). `delete_embedding` tolerates a missing collection.
- `search_embeddings` checks `collection_exists`; a missing collection yields an empty vector. Scores are foundation-normalized `[0,1]`.
- Variables are stored as JSON strings? No: store the raw string as `KvValue::Text` for Python-compatible inspection; `get_variable` returns text only.
- `set_scope_config` serializes the config JSON under `memory_scope_config/{scope_id}`; malformed stored JSON returns `Error::Validation` (never silently dropped).
- Graph access is exposed through `stores().graph`; this module does not implement graph semantics.

### `messages.rs`
```rust
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StoredMessage {
    pub id: String, pub user_id: String, pub scope_id: String,
    pub session_id: Option<String>, pub role: String, pub content: String,
    pub timestamp: DateTime<Utc>,
}
pub(crate) struct MessageStore { /* document */ }
impl MessageStore {
    pub fn new(document: Arc<dyn DocumentStore>) -> Self;
    pub async fn append(&self, scope: &MemoryScope, messages: &[BaseMessage], timestamp: DateTime<Utc>) -> Result<Vec<StoredMessage>>;
    pub async fn recent(&self, scope: &MemoryScope, limit: usize) -> Result<Vec<StoredMessage>>; // newest first, then reversed to chronological
    pub async fn delete_for(&self, filter: &MemoryFilter) -> Result<usize>;
    pub async fn count(&self, scope: &MemoryScope) -> Result<usize>;
}
```
`id` is deterministic: `msg_{fnv64(content + role + millis):016x}_{millis}` (Python's `msg_{sha256(...)}_{ms}` shape, no crypto dependency). Collection `messages`; documents are the flat `StoredMessage` JSON. `recent` queries `user_id`/`scope_id`/optional `session_id`, orders by `timestamp` desc, limit, then reverses. `content` is the plain-text view of `BaseMessage::content`; non-text parts are JSON-rendered into content. `role` uses the `Role` serde name (`system`/`user`/`assistant`/`tool`).

## Constraints

- Only foundation storage calls; no direct filesystem/database access.
- Never hold a `parking_lot` lock across `.await`; the foundation stores already do their own synchronization.
- Prefix scans are the listing mechanism; there are no id-list keys (documented deviation from Python's 24-char id lists).
- A missing collection/key is not an error: `get` returns `None`, `delete` returns `0`/`false`, search returns `[]`.
- Store errors propagate unchanged; add context only with `Error::with_context` where the operation is non-obvious.
- `MemoryStores` is `Clone` (all fields are `Arc`s) so engine/facade/graph can share it.

## Routing Table

- `./mod.rs` → `MemoryStores` and backend constructors
- `./keys.rs` → KV key and vector collection naming, `sanitize_segment`
- `./storage.rs` → record/vector/variable/scope-config persistence
- `./messages.rs` → `StoredMessage`, `MessageStore`
- `./json.rs` → internal record (de)serialization helpers and `plain_text` for `BaseMessage`

## Test Strategy

- Unit tests beside each file: key shapes, sanitization/collision resistance, record JSON round-trip, malformed-record tolerance, vector collection naming.
- Integration tests `tests/crud.rs`, `tests/scoping.rs`, `tests/sqlite.rs` exercise the public facade against these stores; `tests/recall.rs` covers vector candidate retrieval.
- Local-file tests use `tempfile::tempdir`; SQLite tests use a temp path and reopen the store to prove persistence.
