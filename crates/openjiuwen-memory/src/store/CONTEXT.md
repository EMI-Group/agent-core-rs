# store

## Intent

Persistence for memory records, embeddings, variables, scope configs, and message history on top of the `openjiuwen-foundation` storage traits.
Python stores records in KV + vectors in per-type collections + messages in SQL tables; this module keeps that logical layout on `KvStore`, `VectorStore`, `DocumentStore`, and `GraphStore`.
No ranking, no extraction, no engine policy live here: only key derivation, serialization, store calls, the versioned SQLite schema runner, the data-migration suite, and legacy key-layout read compatibility.
`prefix` owns the runtime `KvPrefixRegistry` (current prefix + historical aliases per family); `legacy` owns the read-only fallback to the alias prefixes; `storage` is a thin caller over both.
`migration` owns the forward-only SQLite schema-version table, the builtin baseline, and the four data-migrator families planned through a resumable operation journal.
`backup` owns the backup/restore keyspace, `mapping` the persisted `scope_user_mapping/{scope}/{user}` rows, and `kv_lock` the distributed-lock seam (in-process, cross-process file lock, typed `Unsupported` broker).

## API Surface

### `keys.rs`
```rust
pub const RECORD_PREFIX: &str = "UMD";                     // Python KV namespace
pub const VARIABLE_PREFIX: &str = "user_var";
pub const SESSION_VARIABLE_PREFIX: &str = "session_var";
pub const SCOPE_CONFIG_PREFIX: &str = "memory_scope_config";
pub const SCOPE_USER_MAPPING_PREFIX: &str = "scope_user_mapping";
pub const MESSAGES_COLLECTION: &str = "messages";

pub fn record_key(scope, id) -> String;                    // "UMD/{user}/{scope}/{id}"
pub fn record_prefix(scope) -> String;                     // "UMD/{user}/{scope}/"
pub fn record_prefix_for(user_id) -> String;               // "UMD/{user}/"
pub fn record_prefix_for_scope(scope_id) -> String;        // "UMD/" (scope-wide scan; filter afterwards)
pub fn variable_key(scope, name) -> String;                // "user_var/{user}/{scope}/{name}"
pub fn variable_prefix(scope) -> String;
pub fn session_variable_key(scope, session_id, name) -> String;
pub fn session_variable_prefix(scope, session_id) -> String;
pub fn scope_config_key(scope_id) -> String;
pub fn vector_collection(scope, kind: MemoryType) -> String;
pub fn sanitize_segment(value: &str) -> String;            // [A-Za-z0-9._-] only + short FNV hash suffix when changed

// Legacy (read-only) layout, one prefix per family — see "Legacy Prefix Compatibility" below.
pub const LEGACY_RECORD_PREFIX: &str = "user_mem";
pub const LEGACY_VARIABLE_PREFIX: &str = "user_variable";
pub const LEGACY_SESSION_VARIABLE_PREFIX: &str = "session_variable";
pub const LEGACY_SCOPE_CONFIG_PREFIX: &str = "scope_config";
pub fn legacy_record_key / legacy_record_prefix / legacy_record_prefix_for / legacy_record_prefix_for_scope(...);
pub fn legacy_variable_key / legacy_variable_prefix / legacy_session_variable_key / legacy_session_variable_prefix(...);
pub fn legacy_scope_config_key(scope_id: &str) -> String;
pub fn legacy_alternatives(key: &str) -> Vec<String>;   // legacy candidates of a current key or prefix
pub fn current_equivalent(key: &str) -> Option<String>; // inverse, for diagnostics
pub fn legacy_prefix(current_prefix: &str) -> Option<&'static str>;
pub fn current_prefix(legacy_prefix: &str) -> Option<&'static str>;

// Registry-aware builders: the same shapes, head taken from the caller's KvPrefixRegistry.
pub fn record_key_in / record_prefix_in / record_prefix_for_in / record_prefix_for_scope_in(prefixes: &KvPrefixRegistry, …) -> String;
pub fn variable_key_in / variable_prefix_in(prefixes: &KvPrefixRegistry, scope: &MemoryScope, …) -> String;
pub fn session_variable_key_in / session_variable_prefix_in(prefixes: &KvPrefixRegistry, scope: &MemoryScope, session_id: &str, …) -> String;
pub fn scope_config_key_in(prefixes, scope_id) -> String;
pub fn scope_config_prefix_in(prefixes) -> String;                                   // "<scope_config current>/"
pub fn scope_user_mapping_key_in(prefixes, scope_id, user_id) -> String;             // "<current>/{scope_id}/{user_id}"
pub fn scope_user_mapping_prefix_in(prefixes, scope_id) -> String;                   // "<current>/{scope_id}/"
pub fn messages_collection_in(prefixes) -> &str;
pub fn legacy_alternatives_in(prefixes, key) -> Vec<String>;
pub fn current_equivalent_in(prefixes, key) -> Option<String>;
```
`vector_collection` follows Python `uid_{user}_gid_{scope}_mtype_{kind}` with each segment sanitized: an unchanged safe segment stays readable, otherwise an 8-hex FNV-1a suffix avoids collisions; names must satisfy `LocalFileVectorStore` (`[A-Za-z0-9._-]`, no `.`/`..`).
The legacy builders keep the current field order and `/` and differ only in the first segment, so `legacy_alternatives` is one first-segment rewrite that works for point keys and prefixes; one private `(current, legacy)` table backs all four mapping helpers so the directions cannot drift.
The `*_in` builders are the same shapes with the head from a `KvPrefixRegistry`; the plain builders are the default-registry form, and a test pins both to the same output.
`legacy_alternatives_in` emits one candidate per alias of the key's family (registration order, never the segment of `key` itself); `current_equivalent_in` maps an alias head back to the current prefix and returns `None` when the head is current, unknown, or absent.
The plain `legacy_alternatives`/`current_equivalent` and the `(current, legacy)` table are the default-registry instances of those rules; `keys::vector_collection` stays prefix-independent because vector collections are not a KV family.
`scope_user_mapping_*_in` are that family's only builders (no plain form: Python has no legacy prefix for it).

### `prefix.rs`
```rust
pub enum StorageArea { Kv, Document }
pub enum KvFamily { Records, UserVariables, SessionVariables, ScopeConfig, ScopeUserMapping, Messages }
impl KvFamily {
    pub const ALL: [KvFamily; 6];
    pub const fn label(self) -> &'static str;  // "records", "user_variables", "session_variables", "scope_config", "scope_user_mapping", "messages"
    pub const fn area(self) -> StorageArea;    // Document only for Messages
}
pub struct FamilyPrefixes { pub current: String, pub aliases: Vec<String> }
pub struct KvPrefixRegistry { /* BTreeMap<KvFamily, FamilyPrefixes> */ }
impl Default for KvPrefixRegistry;  // new() == default
impl KvPrefixRegistry {
    pub fn new() -> Self;
    pub fn register_current(&mut self, family: KvFamily, prefix: impl Into<String>) -> Result<()>;
    pub fn register_legacy(&mut self, family: KvFamily, prefix: impl Into<String>) -> Result<()>;
    pub fn with_current(self, family: KvFamily, prefix: impl Into<String>) -> Result<Self>;
    pub fn with_legacy(self, family: KvFamily, prefix: impl Into<String>) -> Result<Self>;
    pub fn current(&self, family) -> &str;    pub fn aliases(&self, family) -> &[String];
    pub fn family_prefixes(&self, family) -> &FamilyPrefixes;
    pub fn prefixes(&self, family) -> impl Iterator<Item = &str> + '_; // current, then aliases in order
    pub fn all_prefixes(&self) -> Vec<String>;                        // ALL order, current then aliases, de-duplicated
    pub fn families(&self) -> impl Iterator<Item = KvFamily> + '_;
    pub fn family_of(&self, key: &str) -> Option<KvFamily>;           // first `/`-segment match, current or alias
    pub fn contains_prefix(&self, family, prefix: &str) -> bool;
    pub fn validate(&self) -> Result<()>;
    pub fn to_json(&self) -> Result<String>;
    pub fn from_json(text: &str) -> Result<Self>;                     // decode, then validate()
}
```
The registry is a value passed to the code that needs it, never Python's process-global `kv_prefix_registry` singleton.
`Default`/`new` reproduce today's layout from the `keys` constants and the one `(current, legacy)` pair table: `records` `UMD`/`user_mem`, `user_variables` `user_var`/`user_variable`, `session_variables` `session_var`/`session_variable`, `scope_config` `memory_scope_config`/`scope_config`, `scope_user_mapping` (no alias), `messages` (no alias, a document collection).
A prefix is one key/collection segment: `validate_prefix` rejects empty/whitespace-only, over 128 characters, and any `/`, `\` or NUL with `Error::Validation` naming the family and the value.
`register_current` demotes the previous current prefix to the front of `aliases` (de-duplicated, and removed when it becomes current again), so data written under it stays readable; `register_legacy` appends an alias unless it is the current prefix or already present; both are `Ok(())` no-ops when nothing changes.
`validate` requires every `KvFamily` to be present and rejects an alias equal to the current prefix or a duplicated alias; `from_json` decodes and validates, so a decoded registry is always usable for key derivation.
`family_of` matches only the first `/`-separated segment, so `user_variable/...` is never mistaken for the `user_var` namespace, and unknown or empty keys give `None`.
`current`/`family_prefixes` answer with the documented default (or an empty entry) for a registry decoded from JSON without validation; only `validate` reports the missing family.

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
    #[cfg(feature = "sqlite")] pub async fn sqlite(path: impl AsRef<Path>) -> Result<Self>;        // SqliteKvStore + SqliteDocumentStore, LocalFileVectorStore under "<path>.vectors/", LocalFileGraphStore under "<path>.graph/"
    #[cfg(feature = "sqlite")] pub async fn sqlite_migrated(path: impl AsRef<Path>) -> Result<Self>; // migrate_sqlite(path), then sqlite(path)
    pub fn from_parts(kv: Arc<dyn KvStore>, vector: Arc<dyn VectorStore>, document: Arc<dyn DocumentStore>) -> Self; // graph = None
    pub fn with_graph(self, graph: Arc<dyn GraphStore>) -> Self;
}
```
`local_file` creates `<root>/kv`, `<root>/documents`, `<root>/graph` via the foundation constructors and `<root>/vector` for vectors. `sqlite` uses `SqliteKvStore::connect(&format!("sqlite:{}", path.display()))`, `SqliteDocumentStore::connect(...)`; the vector/graph snapshots use `LocalFileVectorStore::new("<path>.vectors")` and `LocalFileGraphStore::new("<path>.graph")`. A `MemoryStores::sqlite` test must reopen the same path and observe committed records.

### `storage.rs` (crate-internal)
```rust
pub(crate) struct MemoryStorage { /* stores: MemoryStores, prefixes: KvPrefixRegistry */ }
impl MemoryStorage {
    pub(crate) fn new(stores: MemoryStores) -> Self;                                       // default prefix layout
    pub(crate) fn with_prefixes(stores: MemoryStores, prefixes: KvPrefixRegistry) -> Self; // explicit layout
    pub(crate) fn prefixes(&self) -> &KvPrefixRegistry;
    // Records: put_record / get_record(scope, id) / list_records(filter) / delete_record(record) -> Result<bool>
    // Embeddings: index_embedding(record, embedding) / delete_embedding(record) /
    //             search_embeddings(scope, kind, vector, top_k) -> Vec<(String, f32)>
    // Variables: set_variable / get_variable / list_variables / delete_variable(scope, session_id, name)
    // Scope config: set_scope_config / get_scope_config / delete_scope_config(scope_id)
    // Every method is `pub(crate) async fn … -> Result<…>`; see the semantics below.
}
```
Semantics:
- `put_record` serializes `record` with `embedding: None` (the embedding column is authoritative) to the record key. The scope is already flattened in the JSON.
Every key derivation goes through the `keys::*_in` builders and the `legacy::*_in` helpers with `self.prefixes`; vector collection names stay prefix-independent.
- `get_record` reads the current key first and falls back to the alias layout only when it is absent (`legacy::read_text_in`); a missing key is `None`, a stored non-text/malformed value for the key actually read is `Error::Validation`.
- `list_records` chooses the narrowest prefix from `filter.user_id`/`scope_id` (user+scope when both, user when only user, whole `UMD/` when neither), then delegates to `legacy::list_records_in`: it scans the current prefix and then each legacy prefix, parses each value, ignores malformed entries with `tracing::warn!` (a corrupt record never fails a listing), lets the current layout win on a duplicate record key, applies `MemoryFilter::matches`, and sorts by `(timestamp desc, id asc)`.
- `delete_record` deletes the KV key and every legacy alternative plus the vector id; returns whether either KV key existed.
- Embeddings: the vector collection schema is `id VARCHAR(256) PRIMARY KEY` + `embedding FLOAT_VECTOR(dim)`; `index_embedding` ensures the collection exists (create on first use), deletes the id if present, then `add_docs` (foundation is insert-only); `delete_embedding` tolerates a missing collection and `search_embeddings` checks `collection_exists` (a missing collection yields an empty vector, scores are foundation-normalized `[0,1]`).
- Variables store the raw string as `KvValue::Text` for Python-compatible inspection; `get_variable` returns text only (legacy fallback applies, non-text stays `None`), `list_variables` merges the current namespace with the legacy one and lets a current value win for the same name.
- `set_scope_config` serializes the config JSON under `memory_scope_config/{scope_id}`; malformed stored JSON returns `Error::Validation` (never silently dropped); reads fall back to `scope_config/{scope_id}` when the current key is absent.
- Graph storage is carried by `MemoryStores::graph` and used by the `graph` module; this module does not implement graph semantics.

### `messages.rs`
```rust
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct StoredMessage {
    pub id: String, pub user_id: String, pub scope_id: String,
    pub session_id: Option<String>, pub role: String, pub content: String,
    pub timestamp: DateTime<Utc>,
}
pub(crate) struct MessageStore { /* document: Arc<dyn DocumentStore>, prefixes: KvPrefixRegistry */ }
impl MessageStore {
    pub(crate) fn new(document: Arc<dyn DocumentStore>) -> Self;                                        // default collection name
    pub(crate) fn with_prefixes(document: Arc<dyn DocumentStore>, prefixes: KvPrefixRegistry) -> Self;
    pub(crate) async fn append(&self, scope: &MemoryScope, messages: &[BaseMessage], timestamp) -> Result<Vec<StoredMessage>>;
    pub(crate) async fn recent(&self, scope: &MemoryScope, limit: usize) -> Result<Vec<StoredMessage>>; // newest first, then reversed
    pub(crate) async fn delete_for(&self, filter: &MemoryFilter) -> Result<usize>;
}
pub(crate) fn derived_message_id(content: &str, role: &str, timestamp: DateTime<Utc>) -> String;
pub(crate) fn stored_document(message: &StoredMessage) -> Document;
```
`derived_message_id` is crate-visible and is the single source of the deterministic id `msg_{fnv64(content + role + millis):016x}_{millis}` (Python's `msg_{sha256(...)}_{ms}` shape, no crypto dependency); `append` calls it and the legacy-row import reuses it, so re-importing the same row replaces the same document.
`stored_document` is the flat `StoredMessage` document shape (`user_id`, `scope_id`, optional `session_id`, `role`, `content`, RFC 3339 `timestamp`); `append` and the message migrator both write through it, so the import cannot drift from the append path.
The collection name comes from `keys::messages_collection_in(&self.prefixes)` (`messages` by default), so a renamed document family is written and read consistently. `recent` queries `user_id`/`scope_id`/optional `session_id`, orders by `timestamp` desc, limit, then reverses. `content` is the plain-text view of `BaseMessage::content`; non-text parts are JSON-rendered into content. `role` uses the `Role` serde name (`system`/`user`/`assistant`/`tool`).

### `backup.rs`
```rust
pub const BACKUP_FORMAT: &str = "openjiuwen.memory.backup";
pub const BACKUP_FORMAT_VERSION: u32 = 1;

pub struct BackupCounts { pub records: usize, pub user_variables: usize,
                          pub session_variables: usize, pub scope_configs: usize, pub total: usize }
impl BackupCounts {
    pub fn from_entries(entries: &BTreeMap<String, KvValue>, prefixes: &KvPrefixRegistry) -> Self;
    pub fn merge(&mut self, other: &Self);
}
pub struct MemoryBackup {
    pub format: String, pub version: u32, pub created_at: DateTime<Utc>,
    pub prefixes: BTreeMap<String, String>,
    pub entries: BTreeMap<String, KvValue>,
    pub counts: BackupCounts,
}
impl MemoryBackup {
    pub fn new(prefixes: BTreeMap<String, String>, entries: BTreeMap<String, KvValue>, created_at: DateTime<Utc>) -> Self;
    pub fn len(&self) -> usize; pub fn is_empty(&self) -> bool;
    pub fn counts(&self) -> BackupCounts;
    pub fn recompute_counts(&self, prefixes: &KvPrefixRegistry) -> BackupCounts;
    pub fn verify(&self) -> Result<()>;
    pub fn to_json(&self) -> Result<String>;
    pub fn from_json(text: &str) -> Result<Self>;   // decode, then verify()
}
pub struct RestoreReport { pub dry_run: bool, pub written: usize, pub skipped: usize,
                           pub total: usize, pub counts: BackupCounts }
impl RestoreReport { pub fn processed(&self) -> usize; } // written + skipped

pub async fn dump(stores: &MemoryStores) -> Result<MemoryBackup>;
pub async fn dump_with(stores: &MemoryStores, prefixes: &KvPrefixRegistry) -> Result<MemoryBackup>;
pub async fn restore(stores: &MemoryStores, backup: &MemoryBackup) -> Result<RestoreReport>;
pub async fn restore_with(stores: &MemoryStores, backup: &MemoryBackup,
                          prefixes: &KvPrefixRegistry, dry_run: bool) -> Result<RestoreReport>;
```
A backup covers exactly the four KV families `Records`, `UserVariables`, `SessionVariables`, `ScopeConfig`, every `KvValue` verbatim.
`dump_with` scans each family's current prefix and every alias, rewrites an alias key to its current-layout key (`keys::current_equivalent_in`), visits the current prefix first, and never overwrites a held entry, so the current value wins and a round trip survives a family rename.
`restore_with` maps each key through the target registry (a target-renamed family keeps the target's current prefix; an unexplained key is written verbatim), is additive (equal values count `skipped`, target keys absent from the backup are never deleted), and in a dry run performs every read and no write.
`verify` rejects a foreign `format` (`Validation`), a `version` above `BACKUP_FORMAT_VERSION` (`Unsupported`), and a `BackupCounts` that disagrees with the entries; counts are checked against the backup's own `prefixes`, never the restoring registry.
Messages live in the document store and `scope_user_mapping` is rebuilt by the migrator suite, so neither is dumped, restored, or counted.

### `mapping.rs`
```rust
pub struct ScopeUserMapping { pub user_id: String, pub scope_id: String }
pub struct ScopeUserMappingStore { /* kv: Arc<dyn KvStore>, prefixes: KvPrefixRegistry */ }
impl ScopeUserMappingStore {
    pub fn new(kv: Arc<dyn KvStore>) -> Self;
    pub fn from_stores(stores: &MemoryStores) -> Self;
    pub fn with_prefixes(mut self, prefixes: KvPrefixRegistry) -> Self;
    pub fn prefixes(&self) -> &KvPrefixRegistry;

    pub async fn add(&self, user_id: &str, scope_id: &str) -> Result<bool>;   // true when inserted
    pub async fn contains(&self, user_id: &str, scope_id: &str) -> Result<bool>;
    pub async fn get(&self, user_id: &str, scope_id: &str) -> Result<Option<ScopeUserMapping>>;
    pub async fn users_for_scope(&self, scope_id: &str) -> Result<Vec<String>>;
    pub async fn scopes_for_user(&self, user_id: &str) -> Result<Vec<String>>;
    pub async fn list(&self) -> Result<Vec<ScopeUserMapping>>;               // sorted by (scope_id, user_id)
    pub async fn len(&self) -> Result<usize>;
    pub async fn is_empty(&self) -> Result<bool>;
    pub async fn delete(&self, user_id: &str, scope_id: &str) -> Result<bool>;
    pub async fn delete_scope(&self, scope_id: &str) -> Result<usize>;
    pub async fn delete_user(&self, user_id: &str) -> Result<usize>;
}
```
Every row is the JSON `{"user_id": …, "scope_id": …}` stored as UTF-8 text under `<scope_user_mapping current>/{scope_id}/{user_id}`, derived through `keys::scope_user_mapping_key_in`/`prefix_in` with the store's registry.
Both ids are validated before any store call: non-empty, at most 128 characters, no `/`, `\` or NUL, no leading/trailing whitespace; a bad id is `Error::Validation` naming field and value.
`add` is idempotent through one conditional write (`exclusive_set`), so two concurrent adds cannot both report `true`; there is no alias fallback (Python ships no legacy prefix for this family).
Missing rows are not errors (`get` `Ok(None)`, `delete` `Ok(false)`, `delete_scope`/`delete_user` `Ok(0)`, lists empty); decoding is tolerant and warns (`tracing::warn!`, target `openjiuwen_memory::store`) instead of failing a listing.
`delete_scope` removes the whole `<prefix>/{scope}/` namespace (the trailing `/` keeps `ws2` from matching `ws`); `delete_user` scans the family prefix for keys whose final segment is the user id, since the layout is scope-ordered with no reverse index.

### `kv_lock.rs`
```rust
pub const DEFAULT_LOCK_TTL: Duration = Duration::from_secs(30);
pub const DEFAULT_POLL_INTERVAL: Duration = Duration::from_millis(10);

pub struct LockOptions { pub owner: String, pub ttl: Option<Duration>,
                         pub timeout: Option<Duration>, pub poll_interval: Duration }
impl LockOptions {
    pub fn new(owner: impl Into<String>) -> Self;   // ttl None, timeout None (wait forever), default poll
    pub fn with_ttl(self, ttl: Duration) -> Self;
    pub fn with_timeout(self, timeout: Duration) -> Self;
    pub fn with_poll_interval(self, interval: Duration) -> Self;
    pub fn ttl(&self) -> Duration;                  // ttl or DEFAULT_LOCK_TTL
    pub fn validate(&self) -> Result<()>;
}

#[async_trait]
pub trait DistributedKvLock: Send + Sync {
    fn backend(&self) -> &'static str;                     // "in-process", "file", vendor label
    async fn acquire(&self, name: &str, options: &LockOptions) -> Result<KvLockGuard>;
}

pub struct KvLockGuard { /* name, owner, backend, expires_at, opaque held handle */ }
impl KvLockGuard {
    pub fn name(&self) -> &str;
    pub fn owner(&self) -> &str;
    pub fn backend(&self) -> &'static str;
    pub fn is_held(&self) -> bool;
    pub fn expires_at(&self) -> Option<DateTime<Utc>>;
    pub fn release(self);
}
impl Drop for KvLockGuard;                                  // drop == release

pub struct InProcessKvLock { /* one tokio::sync::Mutex per name */ }
impl InProcessKvLock { pub fn new() -> Self; pub fn held_names(&self) -> Vec<String>; }

pub struct FileKvLock { /* openjiuwen_sysop FileLockManager + root */ }
impl FileKvLock {
    pub fn new(root: impl Into<PathBuf>) -> Result<Self>;   // creates the root directory
    pub fn with_timeout(self, timeout: Option<Duration>) -> Self;
    pub fn root(&self) -> &Path;
    pub fn lock_path(&self, name: &str) -> PathBuf;         // "<root>/<name>.lock" (logical target)
    pub fn sidecar_path(&self, name: &str) -> PathBuf;      // "<root>/<sha256(target)>.lock" (real flock file)
    pub fn is_enabled(&self) -> bool;                       // false where the platform has no flock
}

pub struct UnsupportedKvLock { /* broker label */ }
impl UnsupportedKvLock { pub fn new(backend: &'static str) -> Self; pub fn redis() -> Self; }
```
The lock name is the unit of exclusion (Python's `_lock/{name}` key): a migration run takes `migrate`, the dream worker `dream`; callers may pick others such as `team_memory`.
`LockOptions::validate` (called by every backend) rejects an empty/blank/NUL/over-128-character owner and a zero timeout or poll interval with `Error::Validation`; `acquire` also rejects a name containing `/` or `\`.
A `timeout` bounds the wait and expires with `Error::timeout`; `None` waits forever. The `ttl` is only recorded on `expires_at` (advisory): no local backend evicts a holder on expiry.
Release is drop-based and synchronous — dropping the guard frees the in-process name or closes the `flock` descriptor — so there is no daemon, cleanup task, or background expiry.
`FileKvLock` derives one logical sidecar per name (`<root>/<name>.lock`) but the real advisory lock comes from `openjiuwen_sysop::fs::lock::FileLockManager`, which hashes its target into `<root>/<sha256(target)>.lock` (`sidecar_path`); `flock` is advisory, so only processes sharing one root take part.
`InProcessKvLock` clones share one table (a mutex per name, never pruned); two separately constructed instances do not exclude each other, so an application builds one per store root.
`UnsupportedKvLock` constructs without I/O and returns `Error::Unsupported` naming the broker plus the local alternatives from every `acquire`.
The module is named `kv_lock`, not `lock`, because `lite::lock` already owns the crate-root `lock` name.

### `migration/mod.rs`, `migration/registry.rs`, `migration/runner.rs`
```rust
pub struct Migration { pub version: u32, pub name: &'static str, pub description: &'static str, pub up_sql: &'static [&'static str] }
pub struct MigrationRegistry { /* Vec<Migration> */ }
impl MigrationRegistry {
    pub fn new(migrations: Vec<Migration>) -> Result<Self>; // v1 start, strictly ascending, contiguous, unique non-empty names, non-empty description/SQL
    pub fn register(&mut self, m: Migration) -> Result<()>;   pub fn with(self, m: Migration) -> Result<Self>;
    pub fn migrations(&self) -> &[Migration];   pub fn len(&self) -> usize; pub fn is_empty(&self) -> bool;
    pub fn latest_version(&self) -> Option<u32>;   pub fn get(&self, version: u32) -> Option<&Migration>;
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
    pub fn new(registry: MigrationRegistry) -> Self;   pub fn registry(&self) -> &MigrationRegistry;
    pub async fn run(&self, pool: &sqlx::SqlitePool) -> Result<MigrationReport>;
    pub async fn status(&self, pool: &sqlx::SqlitePool) -> Result<MigrationStatus>;
    pub async fn is_up_to_date(&self, pool: &sqlx::SqlitePool) -> Result<bool>;
}
pub async fn migrate_sqlite(path: impl AsRef<Path>) -> Result<MigrationReport>;
```
The registry half compiles without the `sqlite` feature; the runner and report types are feature-gated. `status`/`run` bootstrap `CREATE TABLE IF NOT EXISTS memory_schema_version (version INTEGER PRIMARY KEY, name TEXT NOT NULL, applied_at TEXT NOT NULL)`. `run` applies only versions above the recorded maximum, wraps each migration's DDL **and** its version row in one SQLite transaction, and returns the storage error with `applying memory migration v{n} (\`{name}\`)` context on failure (nothing recorded, DDL rolled back). `migrate_sqlite(path)` opens `sqlite:{path}` (`create_if_missing`) and runs `default_registry()`; it takes a path because `MemoryStores` holds `Arc<dyn KvStore>`/`Arc<dyn DocumentStore>` with no downcast seam, so a `run_on_stores(&MemoryStores)` variant needs a foundation downcast hook (deferred).

### `migration/plan.rs`
```rust
pub const PLAN_FORMAT_VERSION: u32 = 1;

pub struct MigrationContext { pub stores: MemoryStores, pub prefixes: KvPrefixRegistry,
                              pub dry_run: bool, pub now: DateTime<Utc>, pub max_issues: usize }
impl MigrationContext {
    pub fn new(stores: MemoryStores) -> Self;   // default layout, dry_run false, max_issues 100, now = Utc::now()
    pub fn with_prefixes(self, prefixes) -> Self;   pub fn with_dry_run(self, dry_run: bool) -> Self;
    pub fn is_dry_run(&self) -> bool;
    pub fn kv() / document() / vector() -> &Arc<dyn …Store>;
}

pub struct MigrationIssue { pub key: String, pub reason: String }
pub struct MigrationCounts { pub scanned: usize, pub affected: usize, pub skipped: usize, pub failed: usize }
impl MigrationCounts { pub fn merge(&mut self, other: &Self); pub fn merged(self, other: &Self) -> Self; pub fn total(&self) -> usize; }

pub struct MigrationStep { pub operation_id, migrator, version, description,
    pub source_prefix: Option<String>, pub target_prefix: Option<String>,
    pub params: Value, pub counts: MigrationCounts, pub issues: Vec<MigrationIssue> }
impl MigrationStep {
    pub fn new(operation_id, migrator, version, description) -> Self;
    pub fn with_prefixes / with_params / with_counts(self, …) -> Self;
    pub fn push_issue(&mut self, key, reason, max_issues: usize);   // bounded, never errors
}

pub struct MigrationPlan { pub format_version: u32, pub dry_run: bool, pub steps: Vec<MigrationStep> }
impl MigrationPlan {
    pub fn new(steps: Vec<MigrationStep>) -> Self;                  // real run, PLAN_FORMAT_VERSION
    pub fn with_dry_run(self, dry_run: bool) -> Self;   pub fn push(&mut self, step: MigrationStep);
    pub fn len(&self) -> usize; pub fn is_empty(&self) -> bool; pub fn operation_ids(&self) -> Vec<&str>;
    pub fn counts(&self) -> MigrationCounts;                        // merged over steps
    pub fn issues(&self) -> impl Iterator<Item = &MigrationIssue> + '_;
    pub fn steps_for(&self, migrator: &str) -> MigrationPlan;       // keeps format_version and dry_run
    pub fn to_json(&self) -> Result<String>;
    pub fn from_json(text: &str) -> Result<Self>;                   // tolerant decode, Validation on malformed JSON
}

#[async_trait]
pub trait Migrator: Send + Sync {
    fn id(&self) -> &'static str;   fn version(&self) -> u32;   fn description(&self) -> &'static str;
    async fn is_applicable(&self, context: &MigrationContext) -> Result<bool>;
    async fn plan(&self, context: &MigrationContext) -> Result<Vec<MigrationStep>>;
    fn operations(&self) -> Vec<Arc<dyn MigrationOperationHandler>>;
    async fn apply(&self, context: &MigrationContext, plan: &MigrationPlan,
                   journal: &mut MigrationJournal) -> Result<MigrationRun>; // default: register self.operations() and execute
}

pub struct MigrationPlanner { /* Vec<Arc<dyn Migrator>> in registration order */ }
impl MigrationPlanner {
    pub fn new(migrators: Vec<Arc<dyn Migrator>>) -> Self;
    pub fn migrators(&self) -> &[Arc<dyn Migrator>];   pub fn add(&mut self, migrator: Arc<dyn Migrator>);
    pub fn is_empty(&self) -> bool;
    pub fn registry(&self) -> Result<OperationRegistry>;            // duplicate/invalid id across migrators is Validation
    pub async fn plan(&self, context: &MigrationContext) -> Result<MigrationPlan>;
    pub async fn run(&self, context: &MigrationContext, journal: &mut MigrationJournal) -> Result<MigrationRun>;
}

// crate-visible read-only helpers, consumed by the `vector` and `index_version` migrators:
pub(crate) fn vector_field(schema: &CollectionSchema) -> Option<(&str, usize)>;
pub(crate) const SCAN_TOP_K: usize = usize::MAX;
pub(crate) async fn scan_collection(store: &dyn VectorStore, collection: &str) -> Result<Vec<Map<String, Value>>>;
```
`plan` skips a migrator whose `is_applicable` answers `false` (`tracing::debug!`, target `openjiuwen_memory::store`) and mirrors `context.dry_run` onto the plan; `run` plans once, then applies each sub-plan in registration order against the shared journal, stopping at the first error after journalling it.
Planning may scan stores but never mutates them, so a plan is safe to persist or show an operator; each step carries its counters and a `max_issues`-bounded issue list, making the plan both work order and report.
`scan_collection` is a zero-vector similarity search with no filters and `SCAN_TOP_K`, because the vector store has no listing API; a missing collection scans empty and one declaring no `FLOAT_VECTOR` field is `Error::Unsupported` naming it.
`MigrationContext` is `Clone` and deliberately not `Debug` (trait objects); tests assert its fields.

### `migration/operation.rs`
```rust
pub enum OperationOutcome { Succeeded, Failed, Skipped }   // snake_case

pub struct OperationRecord { pub operation_id: String, pub attempts: u32, pub outcome: OperationOutcome,
                             pub error: Option<String>, pub affected: usize, pub skipped: usize, pub failed: usize }
impl OperationRecord { pub fn succeeded(&self) -> bool;   pub fn retryable(&self) -> bool; }

pub struct MigrationJournal { /* BTreeMap<String, OperationRecord> */ }
impl MigrationJournal {
    pub fn new() -> Self;   pub fn get(&self, id: &str) -> Option<&OperationRecord>;
    pub fn records(&self) -> impl Iterator<Item = &OperationRecord> + '_;
    pub fn len(&self) -> usize;   pub fn is_empty(&self) -> bool;
    pub fn completed(&self) -> Vec<String>;   pub fn failed(&self) -> Vec<&OperationRecord>;
    pub fn succeeded(&self, id: &str) -> bool;   pub fn record(&mut self, record: OperationRecord);
    pub fn clear(&mut self);   pub fn to_json(&self) -> Result<String>;   pub fn from_json(text: &str) -> Result<Self>;
}

pub struct OperationReport { pub operation_id: String, pub outcome: OperationOutcome,
                             pub affected: usize, pub skipped: usize, pub failed: usize, pub note: Option<String> }
impl OperationReport {
    pub fn succeeded(operation_id, affected, skipped, failed) -> Self;
    pub fn skipped(operation_id, note) -> Self;   pub fn failed(operation_id, note) -> Self;
}

pub struct MigrationRun { pub plan: MigrationPlan, pub journal: MigrationJournal, pub reports: Vec<OperationReport> }
impl MigrationRun {
    pub fn is_success(&self) -> bool;   // no journal record Failed
    pub fn executed(&self) -> usize;    // journal-wide succeeded records
    pub fn skipped(&self) -> usize;     // journal-wide skipped records
    pub fn counts(&self) -> MigrationCounts;
    pub fn to_json(&self) -> Result<String>;   pub fn from_json(text: &str) -> Result<Self>;
}

#[async_trait]
pub trait MigrationOperationHandler: Send + Sync {
    fn operation_id(&self) -> &'static str;
    async fn execute(&self, context: &MigrationContext, step: &MigrationStep) -> Result<OperationReport>;
}

pub struct OperationRegistry { /* BTreeMap<String, Arc<dyn MigrationOperationHandler>> */ }
impl OperationRegistry {
    pub fn new() -> Self;
    pub fn register(&mut self, handler: Arc<dyn MigrationOperationHandler>) -> Result<()>;
    pub fn register_all(&mut self, handlers: Vec<Arc<dyn MigrationOperationHandler>>) -> Result<()>; // all or nothing
    pub fn with_handler(self, handler: Arc<dyn MigrationOperationHandler>) -> Result<Self>;
    pub fn operation_ids(&self) -> Vec<&str>;  // sorted   pub fn contains(&self, id: &str) -> bool;
    pub fn len(&self) -> usize;   pub fn is_empty(&self) -> bool;
    pub async fn execute(&self, context: &MigrationContext, plan: &MigrationPlan,
                         journal: &mut MigrationJournal) -> Result<MigrationRun>;
}
```
An operation id is `namespace.name`: non-empty, containing `.`, no whitespace or NUL, at most 128 characters; a duplicate or malformed id is `Error::Validation`, and `register_all` stages its insertions so a rejected batch leaves the registry unchanged.
`execute` validates the plan first (a repeated or unregistered id fails the run before any handler executes), then per step: a `Succeeded` journal record is reported `Skipped` (`already completed`) untouched; a plan or context dry run journals `Skipped` (`dry run`) and invokes no handler; otherwise the handler runs, and a `Failed`/`Skipped` record is retried with an incremented attempt counter.
A handler error journals `Failed` with the error message *before* the error returns annotated with `running memory migration operation \`{id}\``; a handler-returned `OperationReport::failed` records the failure and lets the plan continue.
`MigrationRun::executed`/`skipped`/`counts`/`is_success` derive from the run's journal (a resumed run reports the whole migration), while `reports` holds one entry per step this run considered.

### `migration/sql.rs`
```rust
pub const MIGRATOR_ID: &str = "sql";
pub const MIGRATOR_VERSION: u32 = 1;
pub const COPY_USER_VARIABLES: &str = "sql.copy_user_variables";
pub const COPY_SESSION_VARIABLES: &str = "sql.copy_session_variables";
pub const BACKFILL_SCOPE_USER_MAPPING: &str = "sql.backfill_scope_user_mapping";

pub struct SqlMigrator;   // Debug + Clone + Default
impl SqlMigrator { pub fn new() -> Self; }
```
Python's `SQLMigrator` owns Alembic-style column operations; the foundation schema has no evolving relational columns (schema change is `migration/runner.rs`'s job), so this family is the relational work actually left: legacy variable rows into the current prefixes and the `scope_user_mapping` backfill.
`plan` emits at most three steps in the fixed order user variables, session variables, mapping backfill; a copy step is planned only when its family has an alias holding a key, the backfill only when it would insert a pair or report an unparseable record key, so `is_applicable` is "the plan is not empty" (a second read-only scan).
Counters: `scanned` = legacy keys (copy) or record keys (backfill), `affected` = rows to copy / pairs to insert, `skipped` = rows the current layout holds or pairs already mapped, `failed` = unwritable keys or rejected ids (each a `MigrationIssue` bounded by `max_issues`, never a plan failure).
Handlers recompute their work from `context.prefixes` and the live store and treat `params` (`{"sources": [...], "target": "..."}`, `{"records_prefixes": [...], "mapping_prefix": "..."}`) as informational, so a persisted plan cannot move data into a wrong key.
Copy steps delete every legacy key they converted and the backfill sees every mapped pair, so a rerun plans nothing; each handler honours `context.is_dry_run()` itself.

### `migration/message.rs`
```rust
pub const LEGACY_MESSAGE_PREFIXES: &[&str] = &["user_message"];
pub const MIGRATOR_ID: &str = "message";
pub const MIGRATOR_VERSION: u32 = 1;
pub const IMPORT_LEGACY_ROWS: &str = "message.import_legacy_rows";

pub struct LegacyMessageRow {
    pub message_id: String, pub user_id: String, pub scope_id: String,
    pub session_id: Option<String>, pub role: String, pub content: String,
    pub timestamp: DateTime<Utc>,
}
impl LegacyMessageRow {
    pub fn parse(text: &str) -> Result<Self>;
    pub fn parse_value(value: &Value) -> Result<Self>;
}

pub struct MessageMigrator;   // Debug + Clone + Default
impl MessageMigrator { pub fn new() -> Self; }
```
Python keeps history in the `user_message` SQL table; this migrator is the one-way import into the `messages` document collection, decoding rows tolerantly from the flat row and the nested backup shape (nested `message`/`metadata` win over top-level ones).
`user_id`/`scope_id` are mandatory non-empty strings, `role` may be empty and is preserved verbatim, `content` may be a string or a parts array (rendered as plain text), and `timestamp` may be epoch millis (number or numeric string) or RFC 3339 — a missing or unparseable timestamp is a validation failure naming the row, since a fabricated one would silently reorder history.
A row without an id gets `derived_message_id`, documents are written through `stored_document` (upsert), and only then are the migrated legacy keys deleted; an undecodable row becomes a bounded `failed` issue and its key stays in place, visible and repairable.
Identically stored rows count `skipped` and their redundant legacy keys are removed, so a rerun after a successful import scans nothing; dry runs only read and report the counts a real run would produce.

### `migration/vector.rs`
```rust
pub const MIGRATOR_ID: &str = "vector";
pub const MIGRATOR_VERSION: u32 = 1;
pub const RECOLLECT: &str = "vector.recollection";

pub struct RecollectionPair {
    pub source: String, pub target: String, pub user_id: String,
    pub scope_id: String, pub kind: String, pub documents: usize,
}

pub struct VectorMigrator;   // Debug + Clone + Default
impl VectorMigrator { pub fn new() -> Self; }
```
A memory embedding collection is named `uid_{user}_gid_{scope}_mtype_{kind}`, so a historic or short spelling (`mtype_semantic`, `mtype_SemanticMemory`, an unsanitized segment) names a collection current code never looks up; renaming without copying would lose vectors, so the migrator **recollects** into `keys::vector_collection` of the parsed scope.
Detection lists the store's collections and keeps `uid_` names that parse around the `_gid_`/`_mtype_` separators and decode to a known `MemoryType`; an already-canonical name is left alone, a `uid_` name that does not parse / has an empty segment / names no known kind becomes a bounded issue and is **never** touched, and a non-`uid_` collection belongs to another subsystem and is ignored.
`plan` produces at most one `RECOLLECT` step whose params carry the planned pairs (informational; the handler re-detects) with `scanned` = source documents, `affected` = documents the target lacks, `skipped` = documents the target holds, `failed` = unreadable collections.
Applying a pair copies the source schema to a missing target, inserts only the missing documents in batches of 100 (the store rejects a batch holding an existing primary key), verifies every source document is present, and only then deletes the source; an unverifiable pair keeps its source and reports `failed` so the next run copies the remainder.
Idempotency follows from that deletion plus canonical targets, and a dry run creates, inserts, and deletes nothing while reporting the counts a real run would produce.

### `migration/index_version.rs`
```rust
pub const INDEX_VERSION_PREFIX: &str = "memory_index_version";
pub const CURRENT_INDEX_VERSION: u32 = 1;
pub const MIGRATOR_ID: &str = "index_version";
pub const MIGRATOR_VERSION: u32 = 1;
pub const STAMP_COLLECTIONS: &str = "index_version.stamp_collections";
pub const PURGE_STALE_EMBEDDINGS: &str = "index_version.purge_stale_embeddings";

pub struct IndexStamp {
    pub collection: String, pub version: u32, pub dimensions: Option<usize>,
    pub needs_rebuild: bool, pub updated_at: Option<DateTime<Utc>>, pub reason: Option<String>,
}
impl IndexStamp {
    pub fn new(collection: impl Into<String>, version: u32, dimensions: Option<usize>) -> Self;
    pub fn key(collection: &str) -> String;                 // "memory_index_version/{collection}"
    pub fn is_stale(&self, target: u32) -> bool;            // version < target
    pub async fn load(kv: &dyn KvStore, collection: &str) -> Result<Option<Self>>;  // malformed == absent
    pub async fn store(&self, kv: &dyn KvStore) -> Result<()>;
    pub async fn needs_rebuild(kv: &dyn KvStore, collection: &str) -> Result<bool>;
}

pub struct IndexVersionOptions { pub target_version: u32, pub purge_stale: bool }
impl Default for IndexVersionOptions;   // CURRENT_INDEX_VERSION, purge_stale false

pub struct IndexVersionMigrator { /* options */ }
impl IndexVersionMigrator {
    pub fn new() -> Self;
    pub fn with_options(options: IndexVersionOptions) -> Self;
    pub fn options(&self) -> &IndexVersionOptions;
}
```
The crate has no index process to rebuild, so the migrator ports the bookkeeping: one stamp per memory embedding collection recording dimensions, the current version, `needs_rebuild`, the reference timestamp, and a reason; a collection is stale when it has no stamp or a stamp below the target.
`STAMP_COLLECTIONS` writes a rebuild-requesting stamp per stale collection (`affected`) and counts current ones `skipped`; `PURGE_STALE_EMBEDDINGS` is the opt-in second step (`IndexVersionOptions::purge_stale`) that deletes collections whose stamp still asks for a rebuild, counts the documents dropped, and keeps the stamp so the rebuild stays visible.
Only `uid_*` collections are considered; a malformed stamp reads as absent (a warning, never an error), since an absent stamp is exactly the rebuild request a corrupt row deserves.
Stamps are KV metadata under `memory_index_version/`, not a `KvFamily`, so a family rename never moves them; stamping and purging are idempotent, and a dry run writes and deletes nothing.

### `migration/standard.rs`
```rust
pub fn standard_migrators() -> Vec<Arc<dyn Migrator>>;   // SqlMigrator, MessageMigrator, VectorMigrator, IndexVersionMigrator
pub fn standard_planner() -> MigrationPlanner;
pub fn standard_registry() -> Result<OperationRegistry>;  // seven operation ids
pub async fn plan_migrations(context: &MigrationContext) -> Result<MigrationPlan>;
pub async fn run_migrations(context: &MigrationContext) -> Result<MigrationRun>;
pub async fn run_migrations_with(context: &MigrationContext, journal: &mut MigrationJournal) -> Result<MigrationRun>;
```
The order is dependency-driven: `sql` rewrites the variable/record keyspaces first (the mapping backfill derives from record keys), `message` imports rows into the document collection named by that layout, `vector` recollects embeddings once the layout is final, and `index_version` stamps the resulting collection set last (stamping earlier would stamp a collection about to be renamed away and miss its successor).
`standard_registry` holds exactly `sql.copy_user_variables`, `sql.copy_session_variables`, `sql.backfill_scope_user_mapping`, `message.import_legacy_rows`, `vector.recollection`, `index_version.stamp_collections`, `index_version.purge_stale_embeddings` (the purge handler is always registered; planning it depends on `IndexVersionOptions`).
The set is a plain value with no process-global registry: a caller may append its own migrator to `standard_planner()`, run a subset, share a `MigrationJournal` across processes, or call `plan_migrations` for review before `run_migrations`.
Deferred capabilities of a standard run surface as `Error::Unsupported` (for example a remote lock broker coordinating two nodes) rather than a silent no-op, so an unsupported run never looks completed.

### `legacy.rs` (crate-private)
```rust
pub(crate) async fn read_text(kv: &dyn KvStore, key: String) -> Result<Option<(String, String)>>;                                  // default registry
pub(crate) async fn list_records(kv: &dyn KvStore, prefix: &str, filter: &MemoryFilter) -> Result<Vec<MemoryRecord>>;             // default registry
pub(crate) async fn list_variables(kv: &dyn KvStore, prefix: &str) -> Result<BTreeMap<String, String>>;                           // default registry
pub(crate) async fn delete_with_legacy(kv: &dyn KvStore, key: &str) -> Result<bool>;                                             // default registry
// Registry-aware forms; `storage` always passes its own layout.
pub(crate) async fn read_text_in(kv: &dyn KvStore, key: String, prefixes: &KvPrefixRegistry) -> Result<Option<(String, String)>>;
pub(crate) async fn list_records_in(kv: &dyn KvStore, prefix: &str, filter: &MemoryFilter, prefixes: &KvPrefixRegistry) -> Result<Vec<MemoryRecord>>;
pub(crate) async fn list_variables_in(kv: &dyn KvStore, prefix: &str, prefixes: &KvPrefixRegistry) -> Result<BTreeMap<String, String>>;
pub(crate) async fn delete_with_legacy_in(kv: &dyn KvStore, key: &str, prefixes: &KvPrefixRegistry) -> Result<bool>;
```
The plain functions delegate to the `_in` forms with `KvPrefixRegistry::default()`, so there is exactly one implementation of the fallback rules.
All legacy fallback mechanics live here so `storage.rs` stays a thin caller: current key/prefix first, then every `keys::legacy_alternatives_in` candidate, current layout wins on collisions (dedupe keys use `keys::record_key_in`), malformed listing entries warn-and-skip, and every alias hit logs `tracing::debug!` with target `openjiuwen_memory::store`.

## Migration Model

### SQLite schema versions (`migration/registry.rs`, `migration/runner.rs`)

- Forward-only: no down migrations; an older binary must not open a newer database, and a recorded version above the registry is reported (with a warning) as up to date rather than downgraded.
- One `memory_schema_version` row per applied migration (`version` PK, `name`, RFC 3339 `applied_at`); `current` is the maximum recorded version, `pending` is every registry version above it.
- Every migration is atomic: statements and version row commit in one transaction, so a failure leaves schema and version table untouched. The one exception is `PRAGMA`, which SQLite cannot roll back — connection-level pragmas belong in pool setup, never in `up_sql`.
- Validation rejects duplicate versions, gaps, a first version other than 1, backwards steps, duplicate/empty names, and empty descriptions/statement lists.
- The builtin baseline mirrors the foundation DDL and is idempotent on a populated database: v1 `kv_store` (+ `kv_store_expiry_idx`), v2 `documents` (`document_collections`, `documents`, `documents_collection_idx`).
- Version reservation: only real steps ship (v1, v2); v3 is reserved for `src/dreaming/` tables and v4 for `src/team/` tables, appended later via `register`/`with` with their real DDL. No placeholder or no-op migration is ever shipped.
- Python mapping: KV's `MEMORY_MIGRATION_KV_SCHEMA_VERSION` and SQL's per-table `memory_meta` rows collapse into this single global version table.

### Data migrations (`migration/plan.rs`, `migration/operation.rs`, the four families, `migration/standard.rs`)

- Plan → validate → execute: `Migrator::plan` reads stores and returns `MigrationStep`s, `OperationRegistry::execute` validates the plan (unique, registered ids), then runs each step's handler in plan order.
- One step per operation, with its `source_prefix`/`target_prefix` and the collection/prefix detail in `params`; handlers re-derive the concrete work from the context, so `params` is documentation and never an instruction.
- Journal-keyed resume: a `MigrationJournal` maps operation id → attempts/outcome; a `Succeeded` record is skipped (`already completed`) instead of re-run, a `Failed`/`Skipped` record is retried with a growing attempt counter, so passing the same journal continues an interrupted migration.
- Dry-run contract: `MigrationContext::with_dry_run(true)` (or `MigrationPlan::with_dry_run`) makes every step report `Skipped`/`dry run` without invoking a handler, and every handler also honours `context.is_dry_run()` itself so a direct call writes nothing; a dry run reports the work a real run would do.
- Issue reporting is bounded by `MigrationContext::max_issues` (default 100) per step; counters (`scanned`/`affected`/`skipped`/`failed`) always account for every item, so a bounded issue list hides nothing.
- A failed operation is recorded (`Failed`, with the error text) and then propagated with the operation id in its context; a handler-returned `OperationReport::failed` records the failure and lets the run continue.
- Both layers are independent: the schema runner creates the tables a backend opens, the migrator suite converts rows inside them.

## Legacy Prefix Compatibility

Python v0.1.20 registers **no** legacy prefix (`UserMemStore.LEGACY_PREFIXES` and `VariableManager.LEGACY_PREFIXES` are both empty), so this is the crate's explicitly documented compatibility seam rather than an inherited keyspace. The legacy layout keeps the current field order and separator and differs only in the first segment:

| Data family | Current prefix | Legacy prefix |
| --- | --- | --- |
| records | `UMD` | `user_mem` |
| user variables | `user_var` | `user_variable` |
| session variables | `session_var` | `session_variable` |
| scope config | `memory_scope_config` | `scope_config` |

- Precedence: **the current key always wins**; a legacy key is consulted only when the current key is absent.
- Writes (`put_record`, `set_variable`, `set_scope_config`, message append) use the current layout only; legacy keys are read-only compatibility.
- Point reads (`get_record`, `get_variable`, `get_scope_config`) and namespace reads (`list_records`, `list_variables`) fall back; duplicate ids/names resolve to the current layout, and deletes remove the current key plus every legacy alternative.
- Provenance is log-only (`tracing::debug!`); no public type gains a legacy field.
- `scope_user_mapping` and `messages` have no legacy alias; the migrator suite retires legacy prefixes by *moving* rows, not just reading them.

## Deferred

### SQLite schema runner
- Python's per-backend version markers stay collapsed into `memory_schema_version`: no per-table marker, no `KV` marker, no `MigrationPlan`-less registry.
- No `run_on_stores(&MemoryStores)` entry point: `migrate_sqlite` takes a path because `MemoryStores` holds trait objects without a downcast seam (needs a foundation hook).
- No backup keyspace: Python snapshots every registered prefix into `MEMORY_MIGRATION_KV_SCHEMA_VERSION_BACKUP_{ms}` and restores it on failure, while each step here is transactional instead.

### Data migrators
- Python's `SQLMigrator` Alembic column operations (`AddColumn`/`RenameColumn`/`UpdateColumnType`) and dialect handling are not ported: the foundation schema has no evolving relational columns, schema change is `migration/runner.rs`'s job, and this family only moves legacy variable rows plus the mapping backfill.
- Python's per-backend `MessageMigrator` backup-restore-in-memory semantics are replaced by an idempotent `stored_document` upsert plus deletion of the migrated legacy keys; no snapshot, rollback, or per-store schema version exists.
- `VectorMigrator` cannot migrate a collection that declares no vector field or whose primary key differs (typed `Unsupported`/issue), and it has no per-collection metadata version or sequence — it recollects data.
- The index-version migrator stamps and optionally purges but runs no background rebuild, chunked re-embedding, or MemoryDoc field transform; the caller owns the rebuild.
- `scope_user_mapping` is KV-only: no SQL table backend, no alias/legacy prefix fallback (Python ships none), no reverse index beyond a family-prefix scan.
- Distributed locks are local-only: `UnsupportedKvLock` returns `Error::Unsupported` for multi-node brokers (Redis/ZooKeeper/etcd/database rows), and `ttl` is advisory because no local backend evicts an expired holder.
- The KV backup excludes message documents and the vector/graph stores; there is no incremental or per-family backup, and restore never deletes target keys.
- Process-global prefix registration and dynamic discovery are absent: the `KvPrefixRegistry` is injected (no singleton, no `sys.modules`-style discovery) and `keys` stays the compile-time default table.
- The AES at-rest storage codec (`codec/aes_storage_codec.py`) is not ported.
- Legacy fallback covers the KV record/variable/scope-config families only; existing message documents under a legacy collection name are not read by `legacy`, only imported by the message migrator from the `user_message` KV prefixes.

## Constraints

- A prefix is one segment (non-empty, <=128 characters, no `/`, `\` or NUL); `KvPrefixRegistry::validate` and every registration reject anything else with `Error::Validation`.
- The prefix registry is a value injected into `MemoryStorage`, `MessageStore`, `ScopeUserMappingStore`, `MigrationContext`, and the backup functions; there is no process-global registry and no dynamic discovery.
- Memory data goes only through foundation storage calls: `migration/runner.rs` issues schema DDL/version rows through `sqlx` under `sqlite`, `kv_lock`'s file backend goes through `openjiuwen_sysop::fs::lock`, and `migrate_sqlite`'s parent-directory creation is the only filesystem call in the migration tree.
- Schema migrations are forward-only, contiguous from v1, one transaction each, never a no-op placeholder; bad registration is `Error::Validation`, a failed step propagates the storage error with context.
- Data migrations are plan-then-execute, journal-keyed, and dry-run-safe; a handler re-derives its work from the context instead of trusting `params`, and a failed operation is journalled before it is propagated.
- Legacy keys are read-only: writes use the current layout, current wins over legacy, and deletes remove both.
- Restore never deletes target keys, is additive, and maps entry keys through the target registry; a dry run writes nothing.
- Messages are documents, not KV rows: outside `dump`/`restore`, outside `legacy`, written only through `MessageStore`/`stored_document`.
- Index stamps are KV metadata under `memory_index_version/`, not a `KvFamily`, so family renames and backups never touch them.
- The module is `store::kv_lock`, never `store::lock`: `lite::lock` already owns the crate-root `lock` name.
- No new dependency: backup/mapping/prefix/keys use `serde`, `serde_json`, `chrono`; the file lock uses the always-compiled `openjiuwen-sysop`; the migrators use `openjiuwen-foundation` stores plus `async_trait`.
- Never hold a `parking_lot` lock across `.await`; `InProcessKvLock` drops its table lock before awaiting the per-name mutex.
- Prefix scans are the listing mechanism; there are no id-list keys (documented deviation from Python's 24-char id lists).
- A missing collection/key is not an error: `get` returns `None`, `delete` returns `0`/`false`, search returns `[]`, a missing stamp makes a collection stale.
- Store errors propagate unchanged; add context only with `Error::with_context` where the operation is non-obvious.
- `MemoryStores` is `Clone` (all fields are `Arc`s) so engine/facade/graph/migrations can share it.
- File size: `migration/plan.rs` and `migration/operation.rs` are ~1200 lines each and the four migrator modules ~900–1000 because half of each file is its in-file unit-test module and the production half is one cohesive unit; they are not to be split — do not re-investigate that.

## Routing Table

- `./mod.rs` → `MemoryStores` and backend constructors
- `./keys.rs` → KV key and vector collection naming, plain and registry-aware (`*_in`) builders, current↔alias prefix mapping (`legacy_alternatives[_in]`, `current_equivalent[_in]`)
- `./prefix.rs` → `KvFamily`, `StorageArea`, `FamilyPrefixes`, `KvPrefixRegistry` (defaults, registration/demotion, lookup, validation, JSON round trip)
- `./backup.rs` → `BACKUP_FORMAT`, `BackupCounts`, `MemoryBackup` (dump document, normalisation, verify), `RestoreReport`, `dump`/`dump_with`/`restore`/`restore_with`
- `./mapping.rs` → `ScopeUserMapping`, `ScopeUserMappingStore` (add/get/list/delete CRUD over `scope_user_mapping/…`, id validation, tolerant decoding)
- `./kv_lock.rs` → `DistributedKvLock`, `LockOptions`, `KvLockGuard`, `InProcessKvLock`, `FileKvLock`, `UnsupportedKvLock`
- `./storage.rs` → record/vector/variable/scope-config persistence (thin caller over `legacy` for fallback)
- `./legacy.rs` → crate-private legacy key-layout read/scan/delete fallback helpers
- `./migration/mod.rs` → module docs (schema-runner + data-migrator layers, forward-only/atomic rules) and the explicit re-exports of every submodule's public items
- `./migration/registry.rs` → `Migration`, `MigrationRegistry`, `builtin_migrations`, `default_registry`, the DDL constants, `validate`
- `./migration/runner.rs` → feature `sqlite`: `SCHEMA_VERSION_TABLE`, `MigrationRunner`, `MigrationReport`, `MigrationStatus`, `AppliedMigration`, `migrate_sqlite`, and the bootstrap/fetch/apply helpers
- `./migration/plan.rs` → `MigrationContext`, `MigrationIssue`, `MigrationCounts`, `MigrationStep`, `MigrationPlan`, `Migrator`, `MigrationPlanner`, and the crate-visible `vector_field`/`scan_collection` snapshot helpers
- `./migration/operation.rs` → `OperationOutcome`, `OperationRecord`, `MigrationJournal`, `OperationReport`, `MigrationRun`, `MigrationOperationHandler`, `OperationRegistry`
- `./migration/sql.rs` → `SqlMigrator` and the variable-copy plus scope-user-mapping-backfill operations
- `./migration/message.rs` → `MessageMigrator`, `LegacyMessageRow`, `LEGACY_MESSAGE_PREFIXES`, the legacy-row import operation
- `./migration/vector.rs` → `VectorMigrator`, `RecollectionPair`, the embedding recollection operation
- `./migration/index_version.rs` → `IndexStamp`, `IndexVersionOptions`, `IndexVersionMigrator`, `INDEX_VERSION_PREFIX`, `CURRENT_INDEX_VERSION`, the stamp/purge operations
- `./migration/standard.rs` → the standard migrator set, its planner/registry, and the plan/run entry points
- `./messages.rs` → `StoredMessage`, `MessageStore`, `derived_message_id`, `stored_document`
- `./json.rs` → internal record (de)serialization helpers and `plain_text` for `BaseMessage`

## Test Strategy

- Every suite is offline: in-memory stores, `tempfile`, scripted fixtures, no wall-clock ordering.
- Unit tests live beside each file and assert behavior, not implementation.
- `keys.rs`/`prefix.rs`: key shapes, legacy↔current round trips, registry defaults/demotion/dedup/`family_of`/validation/JSON, vector-collection sanitization, and `*_in` builders matching the plain ones.
- `legacy.rs`/`storage.rs`/`messages.rs`: legacy reads with current-wins, listing dedupe, deletes removing both layouts, malformed-value tolerance, record/embedding/variable/scope-config round trips, SQLite persistence across reopen, `with_prefixes` renamed families, deterministic message ids, a renamed collection.
- `backup.rs`: empty and populated round trips (in-memory and local-file), legacy-key normalisation, current-wins, dry-run restore, target-registry remapping, `verify` rejecting a foreign format, a newer version, and tampered counts.
- `mapping.rs`: CRUD round trip, key/JSON shape, invalid ids writing nothing, missing rows not erroring, malformed rows skipped, a renamed prefix, local-file reopen.
- `kv_lock.rs`: `Send` guards and option defaults, invalid names/options, in-process acquire/release/serialisation/contention+timeout, independent names, a `#[cfg(unix)]` file lock excluding another handle, typed `Unsupported` brokers, owner/TTL/release/`Debug`.
- `migration/plan.rs`: context builders, plan/step JSON round trips and tolerant decode, count merging, `steps_for`, bounded issues, planner order/applicability/duplicate ids/default `apply`, journal-before-failure resume, `scan_collection`, `vector_field`.
- `migration/operation.rs`: journal tracking/JSON, run counters from the journal, id validation and atomic `register_all`, failure journalling + resume, validation failures running nothing, dry runs invoking no handler, completed steps skipped, retries, a reported failure continuing the run, a narrowed plan executing nothing.
- `migration/registry.rs`/`runner.rs`: every validation rule, the builtin plan, appends; fresh-DB run/status, idempotent rerun, DDL+version-row rollback, a pre-populated database, `migrate_sqlite`, a newer version left alone, report JSON.
- Each migrator: identity, empty-store plan, plan/apply/idempotency/dry-run, plus its own cases (`sql` malformed record keys and renamed aliases, `message` flat/nested/malformed rows, `vector` unusable names and partially populated targets, `index_version` malformed stamps and purge).
- `migration/standard.rs`: the ordered ids `["sql", "message", "vector", "index_version"]`, the seven registered operations, an end-to-end run over every seeded family plus a fresh-journal idempotent rerun, a dry run writing nothing, a shared-journal resume reporting `already completed`, a local-file run after a reopen, and a renamed `user_variables` prefix.
- Integration tests `tests/crud.rs`, `tests/scoping.rs`, `tests/sqlite.rs` exercise the public facade; `tests/recall.rs` covers vector candidate retrieval.
- Gates (all offline): `cargo test -p openjiuwen-memory`, `--features sqlite`, `--all-features`; `cargo fmt -p openjiuwen-memory`; `cargo clippy -p openjiuwen-memory --all-targets --all-features -- -D warnings`; `cargo doc -p openjiuwen-memory --no-deps`.
