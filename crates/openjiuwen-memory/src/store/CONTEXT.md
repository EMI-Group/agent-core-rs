# store

## Intent

Persistence for memory records, embeddings, variables, scope configs, and message history on top of the `openjiuwen-foundation` storage traits.
Python stores records in KV + vectors in per-type collections + messages in SQL tables; this module keeps the same logical layout on `KvStore`, `VectorStore`, `DocumentStore`, and `GraphStore`.
No ranking, no extraction, no engine policy live here: only key derivation, serialization, store calls, the versioned SQLite schema runner, and legacy key-layout read compatibility.
`migration` owns the forward-only schema-version table and the builtin SQLite baseline; `legacy` owns the read-only fallback to the pre-registry key prefixes; `storage` is a thin caller over both.

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

// Legacy (read-only) layout, one prefix per family — see "Legacy Prefix Compatibility" below.
pub const LEGACY_RECORD_PREFIX: &str = "user_mem";
pub const LEGACY_VARIABLE_PREFIX: &str = "user_variable";
pub const LEGACY_SESSION_VARIABLE_PREFIX: &str = "session_variable";
pub const LEGACY_SCOPE_CONFIG_PREFIX: &str = "scope_config";
pub fn legacy_record_key / legacy_record_prefix / legacy_record_prefix_for / legacy_record_prefix_for_scope(...);
pub fn legacy_variable_key / legacy_variable_prefix(...);
pub fn legacy_session_variable_key / legacy_session_variable_prefix(...);
pub fn legacy_scope_config_key(scope_id: &str) -> String;
pub fn legacy_alternatives(key: &str) -> Vec<String>;   // legacy candidates of a current key or prefix (first-segment rewrite)
pub fn current_equivalent(key: &str) -> Option<String>; // inverse, for diagnostics
pub fn legacy_prefix(current_prefix: &str) -> Option<&'static str>;
pub fn current_prefix(legacy_prefix: &str) -> Option<&'static str>;
```
`vector_collection` follows Python `uid_{user}_gid_{scope}_mtype_{kind}` with each segment sanitized; an unchanged safe segment keeps its readable form, otherwise a 8-hex FNV-1a suffix is appended to avoid collisions. Collection names must satisfy `LocalFileVectorStore` (`[A-Za-z0-9._-]`, no `.`/`..`).
The legacy builders keep the current field order and `/` separator and differ only in the first segment, so `legacy_alternatives` is a single first-segment rewrite that works for both point keys and prefixes; one private `(current, legacy)` table backs all four mapping helpers so the directions cannot drift.

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
pub(crate) struct MemoryStorage { /* stores: MemoryStores */ }
impl MemoryStorage {
    pub(crate) fn new(stores: MemoryStores) -> Self;

    pub(crate) async fn put_record(&self, record: &MemoryRecord) -> Result<()>;
    pub(crate) async fn get_record(&self, scope: &MemoryScope, id: &str) -> Result<Option<MemoryRecord>>;
    pub(crate) async fn list_records(&self, filter: &MemoryFilter) -> Result<Vec<MemoryRecord>>;
    pub(crate) async fn delete_record(&self, record: &MemoryRecord) -> Result<bool>; // record + embedding

    pub(crate) async fn index_embedding(&self, record: &MemoryRecord, embedding: &[f32]) -> Result<()>;
    pub(crate) async fn delete_embedding(&self, record: &MemoryRecord) -> Result<()>;
    pub(crate) async fn search_embeddings(&self, scope: &MemoryScope, kind: MemoryType,
                                          vector: &[f32], top_k: usize) -> Result<Vec<(String, f32)>>;

    pub(crate) async fn set_variable(&self, scope: &MemoryScope, session_id: Option<&str>, name: &str, value: &str) -> Result<()>;
    pub(crate) async fn get_variable(&self, scope: &MemoryScope, session_id: Option<&str>, name: &str) -> Result<Option<String>>;
    pub(crate) async fn list_variables(&self, scope: &MemoryScope, session_id: Option<&str>) -> Result<BTreeMap<String, String>>;
    pub(crate) async fn delete_variable(&self, scope: &MemoryScope, session_id: Option<&str>, name: &str) -> Result<bool>;

    pub(crate) async fn set_scope_config(&self, scope_id: &str, config: &MemoryScopeConfig) -> Result<bool>;
    pub(crate) async fn get_scope_config(&self, scope_id: &str) -> Result<Option<MemoryScopeConfig>>;
    pub(crate) async fn delete_scope_config(&self, scope_id: &str) -> Result<bool>;
}
```
Semantics:
- `put_record` serializes `record` with `embedding: None` (the embedding column is authoritative) to the record key. The scope is already flattened in the JSON.
- `get_record` reads the current key first and falls back to the legacy layout only when it is absent (`legacy::read_text`); a missing key is `None`, a stored non-text/malformed value for the key actually read is `Error::Validation`.
- `list_records` chooses the narrowest prefix from `filter.user_id`/`scope_id` (user+scope when both, user when only user, whole `UMD/` when neither), then delegates to `legacy::list_records`: it scans the current prefix and then each legacy prefix, parses each value, ignores malformed entries with `tracing::warn!` (a corrupt record never fails a listing), lets the current layout win on a duplicate record key, applies `MemoryFilter::matches`, and sorts by `(timestamp desc, id asc)`.
- `delete_record` deletes the KV key and every legacy alternative plus the vector id; returns whether either KV key existed.
- Embeddings: the vector collection schema is `id VARCHAR(256) PRIMARY KEY` + `embedding FLOAT_VECTOR(dim)`; `index_embedding` ensures the collection exists (create on first use), deletes the id if present, then `add_docs` (foundation is insert-only). `delete_embedding` tolerates a missing collection.
- `search_embeddings` checks `collection_exists`; a missing collection yields an empty vector. Scores are foundation-normalized `[0,1]`.
- Variables are stored as JSON strings? No: store the raw string as `KvValue::Text` for Python-compatible inspection; `get_variable` returns text only (legacy fallback applies, non-text stays `None`), `list_variables` merges the current namespace with the legacy one and lets a current value win for the same name.
- `set_scope_config` serializes the config JSON under `memory_scope_config/{scope_id}`; malformed stored JSON returns `Error::Validation` (never silently dropped); reads fall back to `scope_config/{scope_id}` when the current key is absent.
- Graph storage is carried by `MemoryStores::graph` and used by the `graph` module; this module does not implement graph semantics.

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
    pub(crate) fn new(document: Arc<dyn DocumentStore>) -> Self;
    pub(crate) async fn append(&self, scope: &MemoryScope, messages: &[BaseMessage], timestamp: DateTime<Utc>) -> Result<Vec<StoredMessage>>;
    pub(crate) async fn recent(&self, scope: &MemoryScope, limit: usize) -> Result<Vec<StoredMessage>>; // newest first, then reversed to chronological
    pub(crate) async fn delete_for(&self, filter: &MemoryFilter) -> Result<usize>;
}
```
`id` is deterministic: `msg_{fnv64(content + role + millis):016x}_{millis}` (Python's `msg_{sha256(...)}_{ms}` shape, no crypto dependency). Collection `messages`; documents are the flat `StoredMessage` JSON. `recent` queries `user_id`/`scope_id`/optional `session_id`, orders by `timestamp` desc, limit, then reverses. `content` is the plain-text view of `BaseMessage::content`; non-text parts are JSON-rendered into content. `role` uses the `Role` serde name (`system`/`user`/`assistant`/`tool`).

### `migration.rs`
```rust
pub struct Migration { pub version: u32, pub name: &'static str, pub description: &'static str, pub up_sql: &'static [&'static str] }
pub struct MigrationRegistry { /* Vec<Migration> */ }
impl MigrationRegistry {
    pub fn new(migrations: Vec<Migration>) -> Result<Self>; // validates: v1 start, strictly ascending, contiguous, unique non-empty names, non-empty description/SQL
    pub fn register(&mut self, migration: Migration) -> Result<()>;
    pub fn with(mut self, migration: Migration) -> Result<Self>;
    pub fn migrations(&self) -> &[Migration];
    pub fn len(&self) -> usize; pub fn is_empty(&self) -> bool;
    pub fn latest_version(&self) -> Option<u32>;
    pub fn get(&self, version: u32) -> Option<&Migration>;
}
pub fn builtin_migrations() -> Vec<Migration>; // v1 kv_store, v2 documents
pub fn default_registry() -> MigrationRegistry;

#[cfg(feature = "sqlite")]
pub const SCHEMA_VERSION_TABLE: &str = "memory_schema_version";
pub struct AppliedMigration { pub version: u32, pub name: String, pub applied_at: String }
pub struct MigrationReport { pub from_version: Option<u32>, pub to_version: Option<u32>, pub applied: Vec<AppliedMigration> }
impl MigrationReport { pub fn applied_count(&self) -> usize; pub fn is_noop(&self) -> bool; pub fn applied_versions(&self) -> impl Iterator<Item = u32> + '_; }
pub struct MigrationStatus { pub current: Option<u32>, pub pending: Vec<u32>, pub applied: usize }
pub struct MigrationRunner { /* registry */ }
impl MigrationRunner {
    pub fn new(registry: MigrationRegistry) -> Self;
    pub fn registry(&self) -> &MigrationRegistry;
    pub async fn run(&self, pool: &sqlx::SqlitePool) -> Result<MigrationReport>;
    pub async fn status(&self, pool: &sqlx::SqlitePool) -> Result<MigrationStatus>;
    pub async fn is_up_to_date(&self, pool: &sqlx::SqlitePool) -> Result<bool>;
}
pub async fn migrate_sqlite(path: impl AsRef<Path>) -> Result<MigrationReport>;
```
The registry half compiles without the `sqlite` feature; the runner and report types are feature-gated. `status`/`run` bootstrap `CREATE TABLE IF NOT EXISTS memory_schema_version (version INTEGER PRIMARY KEY, name TEXT NOT NULL, applied_at TEXT NOT NULL)`. `run` applies only versions above the recorded maximum, wraps each migration's DDL **and** its version row in one SQLite transaction, and returns the storage error with `applying memory migration v{n} (\`{name}\`)` context on failure (nothing recorded, DDL rolled back). `migrate_sqlite(path)` opens `sqlite:{path}` (`create_if_missing`) and runs `default_registry()`; it takes a path because `MemoryStores` holds `Arc<dyn KvStore>`/`Arc<dyn DocumentStore>` with no downcast seam, so a `run_on_stores(&MemoryStores)` variant needs a foundation downcast hook (deferred).

### `legacy.rs` (crate-private)
```rust
pub(crate) async fn read_text(kv: &dyn KvStore, key: String) -> Result<Option<(String, String)>>;
pub(crate) async fn list_records(kv: &dyn KvStore, prefix: &str, filter: &MemoryFilter) -> Result<Vec<MemoryRecord>>;
pub(crate) async fn list_variables(kv: &dyn KvStore, prefix: &str) -> Result<BTreeMap<String, String>>;
pub(crate) async fn delete_with_legacy(kv: &dyn KvStore, key: &str) -> Result<bool>;
```
All legacy fallback mechanics live here so `storage.rs` stays a thin caller: current key/prefix first, then `keys::legacy_alternatives`, current layout wins on collisions, malformed listing entries warn-and-skip, and every legacy hit logs `tracing::debug!` with target `openjiuwen_memory::store`.

## Migration Model

- Forward-only: there are no down migrations; an older binary must not open a newer database and a recorded version above the registry is reported (with a warning) as up to date rather than downgraded.
- One `memory_schema_version` row per applied migration (`version` PK, `name`, RFC 3339 `applied_at`); `current` is the maximum recorded version, `pending` is every registry version above it.
- Every migration is atomic: its statements and its version row commit in one transaction, so a failure leaves both the schema and the version table untouched. The one documented exception is `PRAGMA`, which SQLite cannot roll back — connection-level pragmas belong in pool setup, never in `up_sql`.
- Validation rejects duplicate versions, gaps, a version other than 1 first, backwards steps, duplicate/empty names, and empty descriptions/statement lists.
- Builtin baseline mirrors the foundation DDL exactly and is idempotent on an already-populated database: v1 `kv_store` (+ `kv_store_expiry_idx`), v2 `documents` (`document_collections`, `documents`, `documents_collection_idx`).
- Version reservation: the builtin registry ships only real steps (v1, v2). The next free version numbers are reserved for the modules being added elsewhere in the crate — v3 for `src/dreaming/` tables, v4 for `src/team/` tables — appended later via `register`/`with` with their real DDL. No placeholder or no-op migration is ever shipped.
- Python mapping: KV's `MEMORY_MIGRATION_KV_SCHEMA_VERSION` and SQL's per-table `memory_meta` rows collapse into this single global version table.
- `migration.rs` is legitimately ~1000 lines: registry + validation + runner + report types + their rollback/idempotency suite are one cohesive unit; split it only when a second migration backend arrives.

## Legacy Prefix Compatibility

Python v0.1.20 registers **no** legacy prefix (`UserMemStore.LEGACY_PREFIXES` and `VariableManager.LEGACY_PREFIXES` are both empty), so this is the crate's explicitly documented compatibility seam rather than an inherited keyspace. The legacy layout keeps the current field order and separator and differs only in the first segment:

| Data family | Current prefix | Legacy prefix |
| --- | --- | --- |
| records | `UMD` | `user_mem` |
| user variables | `user_var` | `user_variable` |
| session variables | `session_var` | `session_variable` |
| scope config | `memory_scope_config` | `scope_config` |

- Precedence: **the current key always wins**; the legacy key is consulted only when the current key is absent.
- Writes (`put_record`, `set_variable`, `set_scope_config`, message append) use the current layout only; legacy keys are read-only compatibility.
- Point reads (`get_record`, `get_variable`, `get_scope_config`) and namespace reads (`list_records`, `list_variables`) fall back; duplicate ids/names resolve to the current layout.
- Deletes remove the current key and every legacy alternative, reporting whether any existed.
- Provenance is log-only (`tracing::debug!`); no public type gains a legacy field.

## Deferred (migration and legacy compatibility)

- Python's per-backend migrators and their version markers: `SQLMigrator` (Alembic add/rename/retype column operations), `VectorMigrator` (collection-metadata versions, per-collection sequences), `MessageMigrator` (message-store schema version), `IndexVersionMigrator` (MemoryDoc field rename/add/remove/transform), and the `OperationRegistry` DTO layer.
- Migration backup/restore: Python snapshots every registered KV prefix into a `MEMORY_MIGRATION_KV_SCHEMA_VERSION_BACKUP_{ms}` blob and restores it on failure; here each migration is transactional instead, so no backup keyspace exists.
- `KvPrefixRegistry` as a runtime registry: the Rust equivalent is the compile-time `keys` mapping table; there is no process-global prefix registration or dynamic discovery.
- Distributed KV lock, AES at-rest storage codec (`codec/aes_storage_codec.py`), and `scope_user_mapping` (Python's scope→user index) are not ported.
- Data migration of legacy message/document rows: legacy fallback covers the KV record/variable/scope-config families only; existing message documents under a legacy collection name are not read or rewritten.

## Constraints

- Memory data goes only through foundation storage calls; the `migration` module is the single exception — it issues schema DDL/version rows directly through `sqlx` under feature `sqlite`, and still has no filesystem access of its own beyond `migrate_sqlite`'s parent-directory creation.
- Migrations are forward-only, contiguous from v1, one transaction per migration, and never a no-op placeholder; bad registration is `Error::Validation`, a failed step propagates the storage error with context.
- Legacy keys are read-only: writes always use the current layout, current wins over legacy, and deletes remove both.
- Never hold a `parking_lot` lock across `.await`; the foundation stores already do their own synchronization.
- Prefix scans are the listing mechanism; there are no id-list keys (documented deviation from Python's 24-char id lists).
- A missing collection/key is not an error: `get` returns `None`, `delete` returns `0`/`false`, search returns `[]`.
- Store errors propagate unchanged; add context only with `Error::with_context` where the operation is non-obvious.
- `MemoryStores` is `Clone` (all fields are `Arc`s) so engine/facade/graph can share it.

## Routing Table

- `./mod.rs` → `MemoryStores` and backend constructors
- `./keys.rs` → KV key and vector collection naming, current↔legacy prefix mapping (`legacy_alternatives`, `current_equivalent`)
- `./storage.rs` → record/vector/variable/scope-config persistence (thin caller over `legacy` for fallback)
- `./legacy.rs` → crate-private legacy key-layout read/scan/delete fallback helpers
- `./migration.rs` → forward-only SQLite schema migration registry, runner, status, builtin baseline
- `./messages.rs` → `StoredMessage`, `MessageStore`
- `./json.rs` → internal record (de)serialization helpers and `plain_text` for `BaseMessage`

## Test Strategy

- Unit tests beside each file: key shapes and legacy↔current mapping round-trips, sanitization/collision resistance, record JSON round-trip, malformed-record tolerance, vector collection naming.
- `legacy.rs` tests seed a legacy key directly in a public `MemoryStores` bundle and read/delete through `MemoryStorage`: legacy record read, current-wins, listing dedupe, user vs session variable namespaces, scope config, delete-removes-both.
- `migration.rs` tests (feature `sqlite`, `tempfile`): fresh DB status/run/status, idempotent second run, a failing migration rolls back both its DDL and its version row, a pre-populated foundation database migrates cleanly, `migrate_sqlite` idempotency, and an out-of-range recorded version is left alone; the non-gated tests cover every registry validation rule.
- Integration tests `tests/crud.rs`, `tests/scoping.rs`, `tests/sqlite.rs` exercise the public facade against these stores; `tests/recall.rs` covers vector candidate retrieval.
- Local-file tests use `tempfile::tempdir`; SQLite tests use a temp path and reopen the store to prove persistence.
