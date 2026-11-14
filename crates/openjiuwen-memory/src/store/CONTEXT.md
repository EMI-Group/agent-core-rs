# store

## Intent

Persistence for memory records, embeddings, variables, scope configs, and message history on top of the `openjiuwen-foundation` storage traits.
Python stores records in KV, vectors in per-type collections, and messages in SQL tables; this module keeps that logical layout on `KvStore`, `VectorStore`, `DocumentStore`, and `GraphStore`.
No ranking, no extraction, and no engine policy live here — only key derivation, serialization, store calls, the versioned SQLite schema runner, the data-migration suite, and legacy key-layout read compatibility.

Per-file responsibilities are in the Routing Table.

## API Surface

### `keys.rs` — key layout

Prefix constants: `RECORD_PREFIX = "UMD"` (the Python KV namespace), `VARIABLE_PREFIX = "user_var"`, `SESSION_VARIABLE_PREFIX = "session_var"`, `SCOPE_CONFIG_PREFIX = "memory_scope_config"`, `SCOPE_OVERRIDES_PREFIX = "memory_scope_overrides"`, `SCOPE_USER_MAPPING_PREFIX = "scope_user_mapping"`, `MESSAGES_COLLECTION = "messages".

| Data | Builder | Shape |
| --- | --- | --- |
| record | `record_key(scope, id)` | `UMD/{user}/{scope}/{id}` |
| record namespaces | `record_prefix(scope)`, `record_prefix_for(user_id)`, `record_prefix_for_scope(scope_id)` | `UMD/{user}/{scope}/`, `UMD/{user}/`, `UMD/` (then filter) |
| user variable | `variable_key(scope, name)` | `user_var/{user}/{scope}/{name}` |
| session variable | `session_variable_key(scope, session_id, name)` | `session_var/{user}/{scope}/{session}/{name}` |
| scope config | `scope_config_key(scope_id)` | `memory_scope_config/{scope_id}` |
| scope analyzer overrides | `scope_overrides_key(scope_id)` | `memory_scope_overrides/{scope_id}` |
| scope-user mapping | `scope_user_mapping_key_in(prefixes, scope_id, user_id)` | `<current>/{scope_id}/{user_id}` |
| message history | `messages_collection_in(prefixes)` | document collection `messages` |
| embedding | `vector_collection(scope, kind: MemoryType)` | `uid_{user}_gid_{scope}_mtype_{kind}` |

`vector_collection` sanitizes each segment through `sanitize_segment`: an unchanged safe segment stays readable, otherwise an 8-hex FNV-1a suffix avoids collisions; names must satisfy `LocalFileVectorStore` (`[A-Za-z0-9._-]`, no `.`/`..`).
The legacy builders keep the current field order and `/` and differ only in the first segment, so `legacy_alternatives` is one first-segment rewrite that works for point keys and prefixes.
They are the same builders under a `legacy_` prefix (`legacy_record_key`, `legacy_record_prefix`, `legacy_record_prefix_for`, `legacy_record_prefix_for_scope`, `legacy_variable_key`, `legacy_variable_prefix`, `legacy_session_variable_key`, `legacy_session_variable_prefix`, `legacy_scope_config_key`) plus the matching `LEGACY_*` constants, and `legacy_prefix`/`current_prefix` are the two directions over the single `(current, legacy)` table that backs the other mapping helpers, so the directions cannot drift.
The `*_in` variants are the same shapes with the head taken from the caller's `KvPrefixRegistry`; the plain builders are the default-registry form and a test pins both to the same output.
`legacy_alternatives_in` emits one candidate per alias of the key's family (registration order, never the segment of `key` itself), and `current_equivalent_in` maps an alias head back to the current prefix, returning `None` when the head is current, unknown, or absent.
Vector collections are not a KV family, so `keys::vector_collection` stays prefix-independent, and `scope_user_mapping_*_in` are that family's only builders (Python ships no legacy prefix for it).
`memory_scope_overrides` is not a KV family either: `scope_overrides_key` is the only builder, so the registry never renames it, `backup` never dumps it, and it has no legacy alias.

### `prefix.rs`

Types: `StorageArea { Kv, Document }`; `KvFamily { Records, UserVariables, SessionVariables, ScopeConfig, ScopeUserMapping, Messages }` with `ALL: [KvFamily; 6]`, `label()` (the snake_case family name) and `area()` (Document only for `Messages`); `FamilyPrefixes { current: String, aliases: Vec<String> }`; `KvPrefixRegistry`, a `BTreeMap<KvFamily, FamilyPrefixes>` whose `new()` equals `Default`.
Methods: `register_current`, `register_legacy`, `with_current`, `with_legacy`, `current`, `aliases`, `family_prefixes`, `prefixes` (current, then aliases in order), `all_prefixes` (ALL order, current then aliases, de-duplicated), `families`, `family_of`, `contains_prefix`, `validate`, `to_json`, `from_json`.

The registry is a value passed to the code that needs it, never Python's process-global `kv_prefix_registry` singleton.
`Default`/`new` reproduce today's layout from the `keys` constants and the one `(current, legacy)` pair table: `records` `UMD`/`user_mem`, `user_variables` `user_var`/`user_variable`, `session_variables` `session_var`/`session_variable`, `scope_config` `memory_scope_config`/`scope_config`, `scope_user_mapping` (no alias), `messages` (no alias, a document collection).
A prefix is one key/collection segment: `validate_prefix` rejects empty/whitespace-only, over 128 characters, and any `/`, `\` or NUL with `Error::Validation` naming the family and the value.
`register_current` demotes the previous current prefix to the front of `aliases` (de-duplicated, and removed when it becomes current again), so data written under it stays readable; `register_legacy` appends an alias unless it is the current prefix or already present; both are `Ok(())` no-ops when nothing changes.
`validate` requires every `KvFamily` to be present and rejects an alias equal to the current prefix or a duplicated alias; `from_json` decodes then validates, so a decoded registry is always usable for key derivation.
`family_of` matches only the first `/`-separated segment, so `user_variable/…` is never mistaken for the `user_var` namespace, and unknown or empty keys give `None`.
`current`/`family_prefixes` answer with the documented default for a registry decoded from JSON without validation; only `validate` reports a missing family.

### `mod.rs`

`MemoryStores` is `Clone` and holds `kv: Arc<dyn KvStore>`, `vector: Arc<dyn VectorStore>`, `document: Arc<dyn DocumentStore>`, and `graph: Option<Arc<dyn GraphStore>>`.
Constructors: `in_memory()` (includes an `InMemoryGraphStore`), `local_file(root)` (creating `kv/`, `vector/`, `documents/`, `graph/` subdirectories through the foundation constructors), `sqlite(path)` and `sqlite_migrated(path)` under feature `sqlite`, `from_parts(kv, vector, document)` (leaves `graph = None`), and `with_graph(graph)`.
`sqlite` connects `SqliteKvStore`/`SqliteDocumentStore` to `"sqlite:{path}"` and puts `LocalFileVectorStore` (`<path>.vectors`) and `LocalFileGraphStore` (`<path>.graph`) beside it; `sqlite_migrated` runs `migrate_sqlite(path)` first, and reopening the same path must observe committed records.

### `storage.rs` (crate-internal)

`MemoryStorage { stores: MemoryStores, prefixes: KvPrefixRegistry }` with `new(stores)` (default layout), `with_prefixes(stores, prefixes)`, and `prefixes()`.
Records: `put_record`, `get_record(scope, id)`, `list_records(filter)`, `delete_record(record) -> Result<bool>`; embeddings: `index_embedding`, `delete_embedding`, `search_embeddings(scope, kind, vector, top_k) -> Vec<(String, f32)>`; variables: `set_variable`, `get_variable`, `list_variables`, `delete_variable(scope, session_id, name)`; scope config: `set_scope_config`, `get_scope_config`, `delete_scope_config(scope_id)`; scope analyzer overrides: `set_scope_overrides(scope_id, value: serde_json::Value)`, `get_scope_overrides(scope_id) -> Option<serde_json::Value>`, `delete_scope_overrides(scope_id) -> bool`.
Every method is `pub(crate) async fn … -> Result<…>` and derives keys through the `keys::*_in` builders and the `legacy::*_in` helpers with `self.prefixes`; vector collection names stay prefix-independent.

- `put_record` serializes the record with `embedding: None` (the embedding column is authoritative) to the record key; the scope is already flattened in the JSON.
- `get_record` reads the current key first and falls back to the alias layout only when it is absent (`legacy::read_text_in`); a missing key is `None`, while a stored non-text/malformed value for the key actually read is `Error::Validation`.
- `list_records` chooses the narrowest prefix from `filter.user_id`/`scope_id` and delegates to `legacy::list_records_in`, which scans the current prefix then each legacy prefix, warns-and-skips malformed entries (a corrupt record never fails a listing), lets the current layout win on a duplicate record key, applies `MemoryFilter::matches`, and sorts by `(timestamp desc, id asc)`.
- `delete_record` deletes the KV key, every legacy alternative, and the vector id; it returns whether either KV key existed.
- The vector collection schema is `id VARCHAR(256) PRIMARY KEY` plus `embedding FLOAT_VECTOR(dim)`; `index_embedding` creates the collection on first use, deletes the id if present, then calls `add_docs` (foundation is insert-only), and `search_embeddings` checks `collection_exists`, so a missing collection yields an empty vector and scores are foundation-normalized `[0,1]`.
- Variables keep the raw string as `KvValue::Text`; `get_variable` returns text only (legacy fallback applies, non-text stays `None`) and `list_variables` lets a current value win for the same name.
- `set_scope_config` serializes the config JSON under `memory_scope_config/{scope_id}`; malformed stored JSON is `Error::Validation`, and reads fall back to `scope_config/{scope_id}`.
- `set_scope_overrides`/`get_scope_overrides`/`delete_scope_overrides` store the analyzer-override payload verbatim as JSON text under `memory_scope_overrides/{scope_id}`: the store never interprets it, a missing key is `None`/`false`, and a stored non-text or non-JSON value is `Error::Validation`.
- `validate_scope_id` (the `MemoryScope` rule for `scope_id`: non-empty, no `/`, `\`, NUL, at most 128 characters) runs before the override keys are derived, so an invalid scope id writes and reads nothing; the scope-config methods keep their crate-internal callers' validation.
- Graph storage is carried by `MemoryStores::graph` and used by the `graph` module; this module implements no graph semantics.

### `messages/mod.rs` and `messages/cursor.rs`

`StoredMessage { id, user_id, scope_id, session_id, role, content, timestamp }` derives `Debug + Clone + Default + PartialEq + Serialize + Deserialize`.
`MessageStore { document: Arc<dyn DocumentStore>, prefixes: KvPrefixRegistry }` has `new(document)` (default collection name), `with_prefixes(document, prefixes)`, `append(scope, messages, timestamp)`, `recent(scope, limit)`, `page(scope, page_size, cursor, filter) -> MessagePage`, and `delete_for(filter) -> Result<usize>`; the crate-visible helpers are `derived_message_id(content, role, timestamp)`, `stored_document(message) -> Document`, and `decode(documents)` (warns and skips malformed rows).
`derived_message_id` is the single source of the deterministic id `msg_{fnv64(content + role + millis):016x}_{millis}` (Python's `msg_{sha256(…)}` shape, no crypto dependency), and `append` and the legacy-row import both call it, so re-importing a row replaces the same document.
`stored_document` is the flat document shape (`user_id`, `scope_id`, optional `session_id`, `role`, `content`, RFC 3339 `timestamp`) that `append` and the message migrator both write through, so the import cannot drift from the append path.
The collection name comes from `keys::messages_collection_in(&self.prefixes)` (`messages` by default); `recent` queries `user_id`/`scope_id`/optional `session_id`, orders by `timestamp` desc, applies the limit, then reverses so the newest message is last, and `content` is the plain-text view of `BaseMessage::content` (non-text parts are JSON-rendered).
`page` walks history oldest-first by `(timestamp asc, id asc)` (`Query.order_by` is a priority list, so identical timestamps tie-break on the document id), reads one look-ahead row to set `has_more`, and returns `next_cursor` only when `has_more`.
`page` is `#[cfg_attr(not(test), allow(dead_code))]` until an in-crate caller wires history pagination; its `page_size == 0` path returns an empty page without touching the store.
`MessageCursor { timestamp, id }` is the `(timestamp, id)` keyset position of one row: `encode()` yields the opaque `mc1.{millis}.{checksum}.{id}` token and `decode(token)` returns `Error::Validation` naming the token for garbage, truncated, or tampered input (`mc1` is the token version, the checksum is the 64-bit FNV-1a of `{millis}.{id}`, and a millisecond stamp out of range is rejected), so no offset or row number ever reaches the caller.
`MessagePage { items, next_cursor, has_more }` is `serde` snake_case and tolerant of unknown fields; `items` are oldest-first and `next_cursor` is the encoded position of the last item (absent when `has_more` is false).
`page_filter(scope, filter, cursor)` is the single page filter: the scope clauses AND the caller `MemoryFilter` clauses AND an inclusive `timestamp >= cursor.timestamp` bound, merged per field (equal conditions collapse, contradicting ones become `$in` over an empty list, range operators keep the stricter bound), because the filter language has no `$or`.
The cursor tie-break happens in memory: `page` reads a growing window of `timestamp >= cursor.timestamp` rows ordered by `(timestamp, id)`, drops rows at or before the cursor, and stops once it holds a full page plus one look-ahead row or the history is exhausted.
`messages/cursor.rs` also owns `scope_filter`/`message_filter` (shared with `recent`/`delete_for`) and `CURSOR_VERSION`.

### `backup.rs`

`BACKUP_FORMAT = "openjiuwen.memory.backup"` and `BACKUP_FORMAT_VERSION = 1`.
`BackupCounts { records, user_variables, session_variables, scope_configs, total }` has `from_entries(entries, prefixes)` and `merge`.
`MemoryBackup { format, version, created_at, prefixes: BTreeMap<String, String>, entries: BTreeMap<String, KvValue>, counts }` has `new`, `len`/`is_empty`, `counts`, `recompute_counts(prefixes)`, `verify`, `to_json`, and `from_json` (decode, then `verify`).
`RestoreReport { dry_run, written, skipped, total, counts }` has `processed()` (written + skipped), and the entry points are `dump(stores)`, `dump_with(stores, prefixes)`, `restore(stores, backup)`, and `restore_with(stores, backup, prefixes, dry_run)`.

A backup covers exactly the four KV families `Records`, `UserVariables`, `SessionVariables`, `ScopeConfig`, every `KvValue` verbatim.
`dump_with` scans each family's current prefix and every alias, rewrites an alias key to its current-layout key (`keys::current_equivalent_in`), visits the current prefix first, and never overwrites a held entry, so the current value wins and a round trip survives a family rename.
`restore_with` maps each key through the target registry (a target-renamed family keeps the target's current prefix; an unexplained key is written verbatim), is additive (equal values count `skipped`, target keys absent from the backup are never deleted), and in a dry run performs every read and no write.
`verify` rejects a foreign `format` (`Validation`), a `version` above `BACKUP_FORMAT_VERSION` (`Unsupported`), and a `BackupCounts` that disagrees with the entries, checked against the backup's own `prefixes` rather than the restoring registry.
Messages live in the document store and `scope_user_mapping` is rebuilt by the migrator suite, so neither is dumped, restored, or counted.

### `mapping.rs`

`ScopeUserMapping { user_id, scope_id }` and `ScopeUserMappingStore` (holding `kv: Arc<dyn KvStore>` and `prefixes: KvPrefixRegistry`) with `new(kv)`, `from_stores(stores)`, `with_prefixes(prefixes)`, `prefixes()`, `add` (true when inserted), `contains`, `get`, `users_for_scope`, `scopes_for_user`, `list` (sorted by `(scope_id, user_id)`), `len`, `is_empty`, `delete`, `delete_scope`, and `delete_user`.
Every row is the JSON `{"user_id": …, "scope_id": …}` stored as UTF-8 text under `<scope_user_mapping current>/{scope_id}/{user_id}`, derived through `keys::scope_user_mapping_key_in`/`prefix_in` with the store's registry.
Both ids are validated before any store call: non-empty, at most 128 characters, no `/`, `\` or NUL, no leading/trailing whitespace; a bad id is `Error::Validation` naming field and value.
`add` is idempotent through one conditional write (`exclusive_set`), so two concurrent adds cannot both report `true`; there is no alias fallback (Python ships no legacy prefix for this family).
Missing rows are not errors (`get` `Ok(None)`, `delete` `Ok(false)`, `delete_scope`/`delete_user` `Ok(0)`, lists empty); decoding is tolerant and warns (`tracing::warn!`, target `openjiuwen_memory::store`) instead of failing a listing.
`delete_scope` removes the whole `<prefix>/{scope}/` namespace (the trailing `/` keeps `ws2` from matching `ws`), and `delete_user` scans the family prefix for keys whose final segment is the user id, since the layout is scope-ordered with no reverse index.

### `kv_lock.rs`

`DEFAULT_LOCK_TTL` is 30 s and `DEFAULT_POLL_INTERVAL` 10 ms.
`LockOptions { owner, ttl: Option<Duration>, timeout: Option<Duration>, poll_interval }` has `new(owner)` (ttl `None`, timeout `None` meaning wait forever, default poll), `with_ttl`, `with_timeout`, `with_poll_interval`, `ttl()` (ttl or `DEFAULT_LOCK_TTL`), and `validate()`.
`DistributedKvLock` (`async_trait`, `Send + Sync`) has `backend()` (`"in-process"`, `"file"`, or a vendor label) and `acquire(name, options) -> Result<KvLockGuard>`; `KvLockGuard` exposes `name`, `owner`, `backend`, `is_held`, `expires_at`, and `release(self)`, and `Drop` releases too.
Backends: `InProcessKvLock` (one `tokio::sync::Mutex` per name; `new`, `held_names`), `FileKvLock` (`new(root)` creates the root, `with_timeout`, `root`, `lock_path`, `sidecar_path`, `is_enabled`), and `UnsupportedKvLock` (`new(backend)`, `redis()`).

The lock name is the unit of exclusion (Python's `_lock/{name}` key); a caller picks one per store root, for example `migrate` for a migration run or `dream` for a dream-worker sweep.
Nothing in the crate takes a lock implicitly and no background task is started, so `run_migrations`/`run_due` callers own the acquisition.
`LockOptions::validate` (called by every backend) rejects an empty/blank/NUL/over-128-character owner and a zero timeout or poll interval with `Error::Validation`; `acquire` also rejects a name containing `/` or `\`.
A `timeout` bounds the wait and expires with `Error::timeout`; the `ttl` is only recorded on `expires_at` (advisory), because no local backend evicts a holder on expiry.
Release is drop-based and synchronous — dropping the guard frees the in-process name or closes the `flock` descriptor — so there is no daemon, cleanup task, or background expiry.
`FileKvLock` derives one logical sidecar per name (`<root>/<name>.lock`) but the real advisory lock comes from `openjiuwen_sysop::fs::lock::FileLockManager`, which hashes its target into `<root>/<sha256(target)>.lock` (`sidecar_path`); `flock` is advisory, so only processes sharing one root take part.
`InProcessKvLock` clones share one table (a mutex per name, never pruned), and two separately constructed instances do not exclude each other, so an application builds one per store root.
`UnsupportedKvLock` constructs without I/O and returns `Error::Unsupported` naming the broker plus the local alternatives from every `acquire`.

### `migration/mod.rs`, `migration/registry.rs`, `migration/runner.rs`

`Migration { version, name, description, up_sql: &'static [&'static str] }` and `MigrationRegistry` with `new` (validates a v1 start, strictly ascending and contiguous versions, unique non-empty names, non-empty description/SQL), `register`, `with`, `migrations`, `len`, `is_empty`, `latest_version`, and `get`.
`builtin_migrations()` is v1 `kv_store` (plus `kv_store_expiry_idx`) and v2 `documents` (`document_collections`, `documents`, `documents_collection_idx`); `default_registry()` wraps them.
Under feature `sqlite`: `SCHEMA_VERSION_TABLE = "memory_schema_version"`, `AppliedMigration { version, name, applied_at }`, `MigrationReport { from_version, to_version, applied }` (`applied_count`, `is_noop`, `applied_versions`), `MigrationStatus { current, pending, applied }`, `MigrationRunner` (`new`, `registry`, `run(pool)`, `status(pool)`, `is_up_to_date(pool)`), and `migrate_sqlite(path)`; the registry half compiles without `sqlite`, and a failing `run` returns the storage error with `applying memory migration v{n} (\`{name}\`)` context.
`migrate_sqlite(path)` opens `sqlite:{path}` (`create_if_missing`) and runs `default_registry()`; it takes a path because `MemoryStores` holds `Arc<dyn KvStore>`/`Arc<dyn DocumentStore>` with no downcast seam, so a `run_on_stores(&MemoryStores)` variant needs a foundation downcast hook (deferred).

### `migration/plan.rs`

`PLAN_FORMAT_VERSION = 1`.
`MigrationContext { stores, prefixes, dry_run, now, max_issues }` has `new(stores)` (default layout, dry_run false, max_issues 100, `now = Utc::now()`), `with_prefixes`, `with_dry_run`, `is_dry_run`, and `kv`/`document`/`vector` accessors; it is `Clone` and deliberately not `Debug` (trait objects).
`MigrationIssue { key, reason }` and `MigrationCounts { scanned, affected, skipped, failed }` (`merge`, `merged`, `total`).
`MigrationStep { operation_id, migrator, version, description, source_prefix, target_prefix, params, counts, issues }` has `new`, `with_prefixes`/`with_params`/`with_counts`, and `push_issue(key, reason, max_issues)` (bounded, never errors).
`MigrationPlan { format_version, dry_run, steps }` has `new` (real run, `PLAN_FORMAT_VERSION`), `with_dry_run`, `push`, `len`/`is_empty`, `operation_ids`, `counts()` (merged over steps), `issues()`, `steps_for(migrator)` (keeping format version and dry_run), `to_json`, and `from_json` (tolerant decode, `Validation` on malformed JSON).
`Migrator` (`async_trait`, `Send + Sync`) declares `id`, `version`, `description`, `is_applicable`, `plan`, `operations`, and a default `apply(context, plan, journal)` that registers `self.operations()` and executes.
`MigrationPlanner` holds a `Vec<Arc<dyn Migrator>>` in registration order and has `new`, `migrators`, `add`, `is_empty`, `registry()` (a duplicate or invalid id across migrators is `Validation`), `plan`, and `run`.
Crate-visible read-only helpers, consumed by the `vector` and `index_version` migrators: `vector_field(schema) -> Option<(&str, usize)>`, `SCAN_TOP_K = usize::MAX`, and `scan_collection(store, collection)`.
`plan` skips a migrator whose `is_applicable` answers `false` (`tracing::debug!`, target `openjiuwen_memory::store`) and mirrors `context.dry_run` onto the plan; `run` plans once, then applies each sub-plan in registration order against the shared journal, stopping at the first error after journalling it.
Planning may scan stores but never mutates them, so a plan is safe to persist or show an operator, and each step carries its counters plus a bounded issue list, making the plan both work order and report.
`scan_collection` is a zero-vector similarity search with no filters and `SCAN_TOP_K` because the vector store has no listing API; a missing collection scans empty and one declaring no `FLOAT_VECTOR` field is `Error::Unsupported` naming it.

### `migration/operation.rs`

`OperationOutcome { Succeeded, Failed, Skipped }` (snake_case).
`OperationRecord { operation_id, attempts, outcome, error, affected, skipped, failed }` has `succeeded()` and `retryable()`.
`MigrationJournal`, a `BTreeMap<String, OperationRecord>`, has `new`, `get`, `records`, `len`/`is_empty`, `completed`, `failed`, `succeeded(id)`, `record`, `clear`, `to_json`, and `from_json`.
`OperationReport { operation_id, outcome, affected, skipped, failed, note }` has `succeeded(operation_id, affected, skipped, failed)`, `skipped(operation_id, note)`, and `failed(operation_id, note)`.
`MigrationRun { plan, journal, reports }` has `is_success` (no journal record `Failed`), `executed` and `skipped` (journal-wide succeeded/skipped records), `counts`, `to_json`, and `from_json`.
`MigrationOperationHandler` (`async_trait`, `Send + Sync`) declares `operation_id` and `execute(context, step) -> Result<OperationReport>`.
`OperationRegistry`, a `BTreeMap<String, Arc<dyn MigrationOperationHandler>>`, has `new`, `register`, `register_all` (all or nothing), `with_handler`, `operation_ids` (sorted), `contains`, `len`/`is_empty`, and `execute(context, plan, journal)`.
An operation id is `namespace.name`: non-empty, containing `.`, no whitespace or NUL, at most 128 characters; a duplicate or malformed id is `Error::Validation`, and `register_all` stages its insertions so a rejected batch leaves the registry unchanged.
`execute` validates the plan first (a repeated or unregistered id fails the run before any handler executes), then per step: a `Succeeded` journal record is reported `Skipped` (`already completed`) untouched; a plan or context dry run journals `Skipped` (`dry run`) and invokes no handler; otherwise the handler runs, and a `Failed`/`Skipped` record is retried with an incremented attempt counter.
A handler error journals `Failed` with the error message *before* the error returns annotated with `running memory migration operation \`{id}\``; a handler-returned `OperationReport::failed` records the failure and lets the plan continue.
`executed`/`skipped`/`counts`/`is_success` derive from the journal (a resumed run reports the whole migration), while `reports` holds one entry per step this run considered.

### `migration/sql.rs`

`MIGRATOR_ID = "sql"`, `MIGRATOR_VERSION = 1`, and the operation ids `COPY_USER_VARIABLES = "sql.copy_user_variables"`, `COPY_SESSION_VARIABLES = "sql.copy_session_variables"`, `BACKFILL_SCOPE_USER_MAPPING = "sql.backfill_scope_user_mapping"`; `SqlMigrator` is `Debug + Clone + Default` with `new()`.
This family is the relational work left once the foundation schema owns column management: legacy variable rows into the current prefixes and the `scope_user_mapping` backfill.
`plan` emits at most three steps in the fixed order user variables, session variables, mapping backfill; a copy step is planned only when its family has an alias holding a key, the backfill only when it would insert a pair or report an unparseable record key, so `is_applicable` is "the plan is not empty" (a second read-only scan).
Counters: `scanned` = legacy keys (copy) or record keys (backfill), `affected` = rows to copy or pairs to insert, `skipped` = rows the current layout holds or pairs already mapped, and `failed` = unwritable keys or rejected ids, each a `MigrationIssue` bounded by `max_issues` and never a plan failure.
Handlers recompute their work from `context.prefixes` and the live store and treat `params` (`{"sources": [...], "target": "..."}`, `{"records_prefixes": [...], "mapping_prefix": "..."}`) as informational, so a persisted plan cannot move data into a wrong key.
Copy steps delete every legacy key they converted and the backfill sees every mapped pair, so a rerun plans nothing; each handler honours `context.is_dry_run()` itself.

### `migration/message.rs`

`LEGACY_MESSAGE_PREFIXES = &["user_message"]`, `MIGRATOR_ID = "message"`, `MIGRATOR_VERSION = 1`, `IMPORT_LEGACY_ROWS = "message.import_legacy_rows"`.
`LegacyMessageRow { message_id, user_id, scope_id, session_id, role, content, timestamp }` has `parse(text)` and `parse_value(value)`; `MessageMigrator` is `Debug + Clone + Default` with `new()`.
This is the one-way import of Python's `user_message` SQL rows into the `messages` document collection, decoding tolerantly from the flat row and the nested backup shape (nested `message`/`metadata` win over top-level ones).
`user_id`/`scope_id` are mandatory non-empty strings, `role` may be empty and is preserved verbatim, `content` may be a string or a parts array (rendered as plain text), and `timestamp` may be epoch millis (number or numeric string) or RFC 3339 — a missing or unparseable timestamp is a validation failure naming the row, since a fabricated one would silently reorder history.
A row without an id gets `derived_message_id`, documents are written through `stored_document` (upsert), and only then are the migrated legacy keys deleted; an undecodable row becomes a bounded `failed` issue and its key stays in place, visible and repairable.
Identically stored rows count `skipped` and their redundant legacy keys are removed, so a rerun after a successful import scans nothing; dry runs only read and report the counts a real run would produce.

### `migration/vector.rs`

`MIGRATOR_ID = "vector"`, `MIGRATOR_VERSION = 1`, `RECOLLECT = "vector.recollection"`; `RecollectionPair { source, target, user_id, scope_id, kind, documents }`; `VectorMigrator` is `Debug + Clone + Default` with `new()`.
A memory embedding collection is named `uid_{user}_gid_{scope}_mtype_{kind}`, so a historic or short spelling (`mtype_semantic`, `mtype_SemanticMemory`, an unsanitized segment) names a collection current code never looks up; renaming without copying would lose vectors, so the migrator **recollects** into `keys::vector_collection` of the parsed scope.
Detection lists the store's collections and keeps `uid_` names that parse around the `_gid_`/`_mtype_` separators and decode to a known `MemoryType`; an already-canonical name is left alone, a `uid_` name that does not parse, has an empty segment, or names no known kind becomes a bounded issue and is **never** touched, and a non-`uid_` collection belongs to another subsystem and is ignored.
`plan` produces at most one `RECOLLECT` step whose params carry the planned pairs (informational; the handler re-detects), with `scanned` = source documents, `affected` = documents the target lacks, `skipped` = documents the target holds, and `failed` = unreadable collections.
Applying a pair copies the source schema to a missing target, inserts only the missing documents in batches of 100 (the store rejects a batch holding an existing primary key), verifies every source document is present, and only then deletes the source; an unverifiable pair keeps its source and reports `failed` so the next run copies the remainder, which makes the migrator idempotent.
A dry run creates, inserts, and deletes nothing while reporting the counts a real run would produce.

### `migration/index_version.rs`

`INDEX_VERSION_PREFIX = "memory_index_version"`, `CURRENT_INDEX_VERSION = 1`, `MIGRATOR_ID = "index_version"`, `MIGRATOR_VERSION = 1`, `STAMP_COLLECTIONS = "index_version.stamp_collections"`, `PURGE_STALE_EMBEDDINGS = "index_version.purge_stale_embeddings"`.
`IndexStamp { collection, version, dimensions, needs_rebuild, updated_at, reason }` has `new`, `key(collection)` (`"memory_index_version/{collection}"`), `is_stale(target)` (version below target), `load(kv, collection)` (malformed == absent), `store(kv)`, and `needs_rebuild(kv, collection)`.
`IndexVersionOptions { target_version, purge_stale }` defaults to `CURRENT_INDEX_VERSION` with `purge_stale` false; `IndexVersionMigrator` has `new`, `with_options`, and `options`.
The crate has no index process to rebuild, so the migrator ports the bookkeeping: one stamp per memory embedding collection, and a collection is stale when it has no stamp or a stamp below the target.
`STAMP_COLLECTIONS` writes a rebuild-requesting stamp per stale collection (`affected`) and counts current ones `skipped`; `PURGE_STALE_EMBEDDINGS` is the opt-in second step that deletes collections whose stamp still asks for a rebuild, counts the documents dropped, and keeps the stamp so the rebuild stays visible.
Only `uid_*` collections are considered, and a malformed stamp reads as absent (a warning, never an error), since an absent stamp is exactly the rebuild request a corrupt row deserves.
Stamps are KV metadata under `memory_index_version/`, not a `KvFamily`, so a family rename never moves them; stamping and purging are idempotent, and a dry run writes and deletes nothing.

### `migration/standard.rs`

`standard_migrators()` (SqlMigrator, MessageMigrator, VectorMigrator, IndexVersionMigrator), `standard_planner()`, `standard_registry() -> Result<OperationRegistry>` (seven operation ids), `plan_migrations(context)`, `run_migrations(context)`, and `run_migrations_with(context, journal)`.
The order is dependency-driven: `sql` rewrites the variable/record keyspaces first (the mapping backfill derives from record keys), `message` imports rows into the document collection named by that layout, `vector` recollects embeddings once the layout is final, and `index_version` stamps the resulting collection set last (stamping earlier would stamp a collection about to be renamed away and miss its successor).
`standard_registry` holds exactly `sql.copy_user_variables`, `sql.copy_session_variables`, `sql.backfill_scope_user_mapping`, `message.import_legacy_rows`, `vector.recollection`, `index_version.stamp_collections`, and `index_version.purge_stale_embeddings` (the purge handler is always registered; planning it depends on `IndexVersionOptions`).
The set is a plain value with no process-global registry: a caller may append its own migrator to `standard_planner()`, run a subset, share a `MigrationJournal`, or call `plan_migrations` for review before `run_migrations`.
Deferred capabilities of a standard run surface as `Error::Unsupported` (for example a remote lock broker coordinating two nodes) rather than a silent no-op, so an unsupported run never looks completed.

### `legacy.rs` (crate-private)

`read_text`, `list_records`, `list_variables`, and `delete_with_legacy` are the default-registry forms; `read_text_in`, `list_records_in`, `list_variables_in`, and `delete_with_legacy_in` take the caller's `KvPrefixRegistry`, and `storage` always passes its own layout.
The plain functions delegate to the `_in` forms with `KvPrefixRegistry::default()`, so there is exactly one implementation of the fallback rules.
All legacy fallback mechanics live here so `storage.rs` stays a thin caller: current key/prefix first, then every `keys::legacy_alternatives_in` candidate, current layout wins on collisions (dedupe keys use `keys::record_key_in`), malformed listing entries warn-and-skip, and every alias hit logs `tracing::debug!` with target `openjiuwen_memory::store`.

## Legacy Prefix Compatibility

Python v0.1.20 registers **no** legacy prefix (`UserMemStore.LEGACY_PREFIXES` and `VariableManager.LEGACY_PREFIXES` are both empty), so this is the crate's explicitly documented compatibility seam rather than an inherited keyspace.
The legacy layout keeps the current field order and separator and differs only in the first segment:

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

## Migration Model

### SQLite schema versions

- Forward-only: no down migrations; an older binary must not open a newer database, and a recorded version above the registry is reported (with a warning) as up to date rather than downgraded.
- One `memory_schema_version` row per applied migration (`version` PK, `name`, RFC 3339 `applied_at`); `current` is the maximum recorded version, `pending` is every registry version above it.
- Every migration is atomic: statements and version row commit in one transaction, so a failure leaves schema and version table untouched. The one exception is `PRAGMA`, which SQLite cannot roll back, so connection-level pragmas belong in pool setup, never in `up_sql`.
- Validation rejects duplicate versions, gaps, a first version other than 1, backwards steps, duplicate/empty names, and empty descriptions/statement lists.
- The builtin baseline mirrors the foundation DDL and is idempotent on a populated database: v1 `kv_store` (+ `kv_store_expiry_idx`), v2 `documents` (`document_collections`, `documents`, `documents_collection_idx`).
- Version reservation: only real steps ship (v1, v2); v3 is reserved for `src/dreaming/` tables and v4 for `src/team/` tables, appended later via `register`/`with` with their real DDL. No placeholder or no-op migration is ever shipped.
- Python mapping: KV's `MEMORY_MIGRATION_KV_SCHEMA_VERSION` and SQL's per-table `memory_meta` rows collapse into this single global version table.

### Data migrations

- Plan → validate → execute: `Migrator::plan` reads stores and returns `MigrationStep`s; `OperationRegistry::execute` validates the plan (unique, registered ids) and runs each step's handler in plan order.
- One step per operation, carrying `source_prefix`/`target_prefix` and the detail in `params`; handlers re-derive the work from the context, so `params` is documentation and never an instruction.
- Journal-keyed resume: a `MigrationJournal` maps operation id → attempts/outcome; a `Succeeded` record is skipped, a `Failed`/`Skipped` record is retried with a growing attempt counter, so passing the same journal continues an interrupted migration.
- Dry runs (`MigrationContext::with_dry_run` or `MigrationPlan::with_dry_run`) report `Skipped`/`dry run` without invoking a handler, and every handler also honours `context.is_dry_run()` itself, so a direct call writes nothing while still reporting the work a real run would do.
- Issues are bounded by `max_issues` (default 100) per step, while the `scanned`/`affected`/`skipped`/`failed` counters always account for every item.
- A failed operation is recorded with its error text and then propagated with the operation id in its context; a handler-returned `OperationReport::failed` records the failure and lets the run continue.
- The two layers are independent: the schema runner creates the tables a backend opens, the migrator suite converts the rows inside them.

## Deferred

Gaps relative to the Python package, each typed rather than silent:

- Schema runner: per-backend version markers stay collapsed into `memory_schema_version` (no per-table or `KV` marker, no `MigrationPlan`-less registry); there is no `run_on_stores(&MemoryStores)` entry point because `migrate_sqlite` can only take a path until a foundation downcast hook lands; and there is no backup keyspace (Python snapshots every registered prefix into `MEMORY_MIGRATION_KV_SCHEMA_VERSION_BACKUP_{ms}`, while each step here is transactional instead).
- `sql`: Python's Alembic column operations (`AddColumn`/`RenameColumn`/`UpdateColumnType`) and dialect handling are not ported — the foundation schema has no evolving relational columns, so this family only moves legacy variable rows plus the mapping backfill.
- `message`: Python's backup-restore-in-memory semantics are replaced by an idempotent `stored_document` upsert plus deletion of the migrated legacy keys; no snapshot, rollback, or per-store schema version exists.
- `vector`: a collection with no vector field or a different primary key cannot be migrated (typed `Unsupported` or a bounded issue), and there is no per-collection metadata version or sequence — it recollects data.
- `index_version`: stamps and optionally purges but runs no background rebuild, chunked re-embedding, or MemoryDoc field transform; the caller owns the rebuild.
- `scope_user_mapping`: KV-only — no SQL table backend, no alias/legacy prefix fallback (Python ships none), and no reverse index beyond a family-prefix scan.
- `kv_lock`: local-only — `UnsupportedKvLock` returns `Error::Unsupported` for multi-node brokers (Redis/ZooKeeper/etcd/database rows), and `ttl` is advisory because no local backend evicts an expired holder.
- `backup`: excludes message documents and the vector/graph stores, and there is no incremental or per-family backup.
- `prefix`: process-global registration and dynamic discovery are absent — the registry is injected (no singleton, no `sys.modules`-style discovery) and `keys` stays the compile-time default table.
- `legacy`: fallback covers the KV record/variable/scope-config families only; message documents under a legacy collection name are not read by `legacy`, only imported by the message migrator from the `user_message` KV prefixes.
- Storage codec: the AES at-rest codec (`codec/aes_storage_codec.py`) is not ported.

## Constraints

- A prefix is one segment (non-empty, <=128 characters, no `/`, `\` or NUL); `KvPrefixRegistry::validate` and every registration reject anything else with `Error::Validation`.
- The registry is injected into `MemoryStorage`, `MessageStore`, `ScopeUserMappingStore`, `MigrationContext`, and the backup functions; there is no process-global registry and no dynamic discovery.
- Memory data goes only through foundation storage calls: `migration/runner.rs` issues schema DDL/version rows through `sqlx` under `sqlite`, `kv_lock`'s file backend goes through `openjiuwen_sysop::fs::lock`, and `migrate_sqlite`'s parent-directory creation is the only filesystem call in the migration tree.
- Schema migrations are forward-only, contiguous from v1, one transaction each, never a no-op placeholder; bad registration is `Error::Validation` and a failed step propagates the storage error with context.
- Data migrations are plan-then-execute, journal-keyed, and dry-run-safe; a handler re-derives its work from the context instead of trusting `params`, and a failed operation is journalled before it is propagated.
- Legacy keys are read-only (writes use the current layout, current wins, deletes remove both), restore never deletes target keys and maps entry keys through the target registry, and a dry run writes nothing.
- Messages are documents, not KV rows: outside `dump`/`restore` and outside `legacy`, they are written only through `MessageStore`/`stored_document`.
- The module is `store::kv_lock`, never `store::lock`, because `lite::lock` already owns the crate-root `lock` name.
- No new dependency: backup/mapping/prefix/keys use `serde`, `serde_json`, `chrono`; the file lock uses the always-compiled `openjiuwen-sysop`; the migrators use `openjiuwen-foundation` stores plus `async_trait`.
- Never hold a `parking_lot` lock across `.await`; `InProcessKvLock` drops its table lock before awaiting the per-name mutex.
- Prefix scans are the listing mechanism; there are no id-list keys (documented deviation from Python's 24-char id lists).
- A missing collection/key is not an error: `get` is `None`, `delete` is `0`/`false`, search is `[]`, and a missing stamp makes a collection stale.
- Store errors propagate unchanged (add context only with `Error::with_context` where the operation is non-obvious), and `MemoryStores` is `Clone` so engine/facade/graph/migrations can share it.
- File size: `migration/plan.rs` and `migration/operation.rs` are ~1200 lines each and the four migrator modules ~900–1000 because half of each file is its in-file unit-test module and the production half is one cohesive unit; they are not to be split — do not re-investigate that.
- Default builds and every suite stay offline (in-memory stores, `tempfile`, loopback only); offline builds must keep working and no new dependency may be added.

## Routing Table

- `./mod.rs` → `MemoryStores` and backend constructors
- `./keys.rs` → KV key and vector collection naming, plain and `*_in` builders, current↔alias mapping (`legacy_alternatives[_in]`, `current_equivalent[_in]`)
- `./prefix.rs` → `KvFamily`, `StorageArea`, `FamilyPrefixes`, `KvPrefixRegistry`
- `./backup.rs` → `BACKUP_FORMAT`, `BackupCounts`, `MemoryBackup`, `RestoreReport`, `dump`/`dump_with`/`restore`/`restore_with`
- `./mapping.rs` → `ScopeUserMapping`, `ScopeUserMappingStore`
- `./kv_lock.rs` → `DistributedKvLock`, `LockOptions`, `KvLockGuard`, `InProcessKvLock`, `FileKvLock`, `UnsupportedKvLock`
- `./storage.rs` → record/vector/variable/scope-config persistence (thin caller over `legacy`)
- `./legacy.rs` → crate-private legacy read/scan/delete fallback helpers
- `./migration/mod.rs` → module docs and the submodule re-exports
- `./migration/registry.rs` → `Migration`, `MigrationRegistry`, `builtin_migrations`, `default_registry`
- `./migration/runner.rs` → feature `sqlite`: `SCHEMA_VERSION_TABLE`, `MigrationRunner`, `MigrationReport`, `MigrationStatus`, `AppliedMigration`, `migrate_sqlite`
- `./migration/plan.rs` → `MigrationContext`, `MigrationIssue`, `MigrationCounts`, `MigrationStep`, `MigrationPlan`, `Migrator`, `MigrationPlanner`, `vector_field`/`scan_collection`
- `./migration/operation.rs` → `OperationOutcome`, `OperationRecord`, `MigrationJournal`, `OperationReport`, `MigrationRun`, `MigrationOperationHandler`, `OperationRegistry`
- `./migration/sql.rs` → `SqlMigrator` and its variable-copy/backfill operations
- `./migration/message.rs` → `MessageMigrator`, `LegacyMessageRow`, `LEGACY_MESSAGE_PREFIXES`, the legacy-row import operation
- `./migration/vector.rs` → `VectorMigrator`, `RecollectionPair`, the embedding recollection operation
- `./migration/index_version.rs` → `IndexStamp`, `IndexVersionOptions`, `IndexVersionMigrator`, `INDEX_VERSION_PREFIX`, `CURRENT_INDEX_VERSION`, the stamp/purge operations
- `./migration/standard.rs` → the standard migrator set, planner/registry, and plan/run entry points
- `./messages/mod.rs` → `StoredMessage`, `MessageStore` (`append`, `recent`, `page`, `delete_for`), `derived_message_id`, `stored_document`, `decode`
- `./messages/cursor.rs` → `MessageCursor`, `MessagePage`, `CURSOR_VERSION`, `scope_filter`/`message_filter`/`page_filter` and the per-field merge of filter clauses
- `./json.rs` → internal record (de)serialization helpers and `plain_text` for `BaseMessage`

## Test Strategy

Every suite is offline (in-memory stores, `tempfile`, scripted fixtures, no wall-clock ordering), and unit tests live beside each file.

- `keys.rs`/`prefix.rs`: key shapes, legacy↔current round trips, registry defaults/demotion/dedup/`family_of`/validation/JSON, collection sanitization, `*_in` parity.
- `legacy.rs`/`storage.rs`/`messages/mod.rs`: current-wins reads, listing dedupe, deletes removing both layouts, malformed tolerance, store round trips, SQLite reopen, renamed families/collections, deterministic message ids, scope-override round trips/overwrite/delete/malformed/invalid-id/local-file+SQLite reopen.
- `messages/cursor.rs`/`messages/mod.rs` pagination: three pages over five messages, identical timestamps paging by id with no repeats or gaps, tampered/truncated cursors, `page_size == 0` and empty history, filter composition (`ids`, `created_after`, a conflicting user clause), cursor survival across a local-file reopen, and the exact merged filter shape.
- `backup.rs`: in-memory and local-file round trips, normalisation, current-wins, dry-run restore, target-registry remapping, `verify` rejections.
- `mapping.rs`: CRUD, key/JSON shape, invalid ids writing nothing, missing/malformed rows, renamed prefix, reopen.
- `kv_lock.rs`: option defaults, invalid names/options, acquire/release/serialisation/contention+timeout, `#[cfg(unix)]` file lock across handles, typed `Unsupported` brokers.
- `migration/plan.rs`/`operation.rs`: JSON round trips and tolerant decode, count merging, `steps_for`, bounded issues, planner order/applicability, journal tracking, resume after failure, atomic `register_all`, dry runs invoking no handler, retries.
- `migration/registry.rs`/`runner.rs`: every validation rule, the builtin plan, fresh-DB run/status, idempotent rerun, DDL+version-row rollback, `migrate_sqlite`, a newer version left alone.
- Each migrator: identity, empty-store plan, plan/apply/idempotency/dry-run, plus its malformed-input cases (`sql` unparseable record keys, `message` flat/nested rows, `vector` unusable collection names, `index_version` malformed stamps).
- `migration/standard.rs`: the ordered ids `["sql", "message", "vector", "index_version"]`, the seven registered operations, an end-to-end run plus an idempotent rerun, a shared-journal resume (`already completed`), a run after a reopen.
- `tests/crud.rs`, `tests/scoping.rs` and `tests/sqlite.rs` exercise the public facade; `tests/recall.rs` covers vector candidate retrieval.
- Gates (all offline): `cargo test -p openjiuwen-memory` (also `--features sqlite` and `--all-features`), `cargo fmt -p openjiuwen-memory`, `cargo clippy -p openjiuwen-memory --all-targets --all-features -- -D warnings`, `cargo doc -p openjiuwen-memory --no-deps`.
