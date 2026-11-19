# kv — key-value store contract and backends

## Intent

The `KvStore` contract every subsystem stores through, plus three in-tree backends and the shared pipeline.
Ports `openjiuwen/core/foundation/store/kv/in_memory_kv_store.py` (the behavioral reference) and replaces Python's `shelve_store.py` / `db_based_kv_store.py` with a plain-JSON-per-key file store and a `SQLite` store.
The fourth backend, `RedisKvStore`, lives in `../external/redis/` and reuses the same trait and pipeline.

## API Surface

- `KvValue` — `Text(String) | Bytes(Vec<u8>)` with `text`/`bytes`, `from_json`/`to_json` (`"text"` / `{"$bytes": "<base64>"}`), `as_str`, `as_bytes`, `is_text`; `Serialize`/`Deserialize` go through `to_json`/`from_json`.
- `KvStore` async trait: `set`, `get`, `exists`, `delete`, `exclusive_set(key, value, Option<Duration>) -> bool`, `get_by_prefix -> BTreeMap`, `delete_by_prefix -> usize`, `mget -> Vec<Option<KvValue>>` (input order), `batch_delete -> usize`, `pipeline()`.
- `KvPipeline` (`set`/`get`/`exists` + `execute -> Vec<Option<KvValue>>`) and the single shared `StorePipeline` implementation (`set` → `None`, `get` → stored option, `exists` → `Text("true"|"false")`).
- `InMemoryKvStore` (`BTreeMap<String, Entry>` behind `parking_lot::Mutex`, `Arc`-shared by clones), `LocalFileKvStore` (one JSON envelope per key at `<root>/entries/<sha256(key)>.json`), `SqliteKvStore` (feature `sqlite`, table `kv_store`, re-exported from `kv::`).

## Semantics (the contract backends must match)

- A missing key reads as `None`; `delete` of a missing key is a no-op; `mget` keeps a `None` placeholder per miss in input order.
- `get_by_prefix("")` returns every live entry ordered by key; `delete_by_prefix`/`batch_delete` return the number of *live* entries removed (expired ones do not count).
- `exclusive_set` writes only when the key is absent or expired and returns `true` on write; `set` clears any TTL.
- TTLs are honored lazily on every read path (`get`, `exists`, `get_by_prefix`, `mget`, `batch_delete`, `delete_by_prefix`, `len`).

## Constraints

- All file I/O goes through `tokio::fs`; there is no `std::fs`/`std::io` call on any non-test async path (`std::io::ErrorKind` matching and `Error` wrapping are the only `std::io` uses).
- `parking_lot` locks are never held across an `await`: every `state.lock()`/`index.lock()` guard is scoped to a block or a single statement, and the awaits follow it.
- Entry filenames are the hex SHA-256 of the key, so a key containing `/`, `..` or an absolute prefix can never escape `<root>/entries` (`store_stays_inside_its_root` pins this).
- Expiry math saturates (`saturating_duration_since`/`saturating_add`/`try_from(...).unwrap_or`), so a hostile or corrupt `expiry_ms` degrades to "expired" or "far future" rather than panicking.
- No `unwrap`/`expect`/`as` casts and no slicing on UTF-8 outside `#[cfg(test)]`; every backend returns `openjiuwen_common::error::{Error, Result}`.
- A corrupt envelope is a hard error (`Error::external` with the path), never an empty value: `new()` and `get()` fail loudly.
- `sqlite` code is behind `#[cfg(feature = "sqlite")]`; default and `--no-default-features` builds must not pull `sqlx`.

## Known Issues and Gaps

- `file.rs` `load_index` reads and fully parses every entry file (including the whole `data` payload, base64-inflated for bytes) at every `LocalFileKvStore::new`, so open cost is O(total bytes on disk) with one sequential read per entry.
- Both `memory.rs` and `file.rs` call `retain` over the *entire* map/index on nearly every operation (`get_by_prefix`, `mget`, `batch_delete`, `delete_by_prefix`, `len`, `is_empty`), making each of those O(total keys) even when only one key is touched.
- `exists` is not a cheap probe: the in-memory backend deep-clones the value (`get(..).is_some()`) and the file backend reads, parses and base64-decodes the whole file; only `sqlite` answers with `SELECT 1`.
- `file.rs`: expired entries are dropped from the index by the prefix paths but their files are never removed, so TTL locks leaked disk until someone calls `get(key)`/`delete(key)` for that exact key; `delete_by_prefix` cannot reclaim them either.
- `file.rs` `delete`/`delete_by_prefix` remove the index entries *before* the files and abort on the first removal error, leaving files on disk that reappear as live keys when the store is reopened.
- `file.rs` `exclusive_set` rolls back its index reservation with an unconditional `index.remove(key)`, which can evict a key a concurrent `set` wrote during the awaited `write_entry`; a `*.tmp` file left by a crash mid-write is never garbage-collected.
- `mod.rs` `StorePipeline::execute` is not a batch: it takes the queue with `std::mem::take` and awaits one store call per op, so an error mid-run leaves earlier ops committed with no transaction and drops the remaining queue (and the collected results) with no way to retry.
- `sqlite.rs` `mget`/`batch_delete` build one `IN (?1..?n)` statement with no chunking, so a batch above `SQLITE_MAX_VARIABLE_NUMBER` (~32 766) fails where the memory/file backends accept any size; every call prepares a fresh statement.
- `sqlite.rs` `connect` detects in-memory databases with `url.contains("memory")`, so a file URL whose path merely contains "memory" is pinned to `max_connections(1)` and every operation serializes.
- `sqlite.rs` never deletes expired rows (`get`/`prefix` paths only filter them), and `delete_by_prefix` skips them too, so only a per-key `delete` reclaims that space.
- The file backend assumes one live instance per root: a second instance opened earlier never observes later writes because `get` answers `None` from its stale index without probing the disk.

## Test Strategy

- Each backend carries its own module-level `#[cfg(test)]` tree with TTL margins (sleeps start after the write so no assertion races the deadline) and a `tempfile::TempDir` per file-store test.
- The cross-backend contract suite lives in `../tests/kv.rs` (`assert_kv_contract`) and `../tests/sqlite.rs` runs the KV contract over `sqlite::memory:`; `pipeline_applies_operations_in_order` exists per backend.
- The file store's TTL test backdates `expiry_ms` in the envelope instead of sleeping out a real deadline.
- `cargo test -p openjiuwen-foundation store`, `--no-default-features store` and `--features sqlite store` must pass offline; no test needs the network.

## Routing Table

- `./mod.rs` → `KvValue`, `KvStore`, `KvPipeline`, `StorePipeline` (shared batching), backend re-exports
- `./memory.rs` → `InMemoryKvStore` (BTreeMap + lazy TTL, the behavioral reference)
- `./file.rs` → `LocalFileKvStore` (sha256-named JSON envelopes under `entries/`, in-memory key→TTL index, `Envelope`, epoch-ms expiry helpers)
- `./sqlite.rs` → `SqliteKvStore` (feature `sqlite`, one `kv_store` table, SQL-side TTL filtering, `placeholders`/`live_at` helpers)
