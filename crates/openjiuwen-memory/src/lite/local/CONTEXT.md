# local

## Intent

The file-backed coding-memory store rooted at one directory: a Markdown representation that
a human can edit, derived indexes (FTS5 and an optional chunked vector index), a
cross-process advisory lock, and a debounced watcher.
Ports the behaviour of Python's `MemoryIndexManager` for one coding-memory root and adds the
Rust-side surfaces the objective asked for: per-fact files with the `MEMORY.md`/`USER.md`
split, JSONL session recall, caller-driven sync triggers, `daily_memory` rollups, and write
modes with a bounded conflict-retry checker.
The authoritative representation is the Markdown document by default and the per-fact files
when `LocalCodingMemoryBuilder::fact_files(true)` is set; every derived index may be rebuilt.
The module-level prose (layout, consistency model, search contract, module split) lives in
`mod.rs`; this file is the routing map for the directory.

## API Surface

`LocalCodingMemory` (re-exported as `crate::lite::local::LocalCodingMemory`) groups its API by
file; all of it is additive to the original store API, which is unchanged.

- Construction: `open`, `open_in_scope`, `builder` (`LocalCodingMemoryBuilder::{new, scope,
  embedding, chunking, fact_files, open}`).
- Facts: `write`, `read`, `edit`, `delete`, `list`, `search`, `render_index`, `reindex`,
  `indexed_count` (feature `sqlite`), `markdown`, `reload_count`.
- Embeddings: `embedding`, `has_embedding`, `chunking`, `embedding_chunks`,
  `refresh_embeddings`, `embedding_cache_len`.
- Per-fact files: `fact_files_enabled`, `facts_dir`, `user_path`, `export_index`,
  `import_index`.
- Sync: `sync_now`, `on_trigger`, `start_interval_sync`, `stop_interval_sync`,
  `is_interval_sync_running`.
- Sessions: `sessions_dir`, `append_session`, `session_records`, `search_sessions`.
- Daily rollups: `daily_dir`, `rollup_day`, `write_daily_rollup`.
- Write modes: `write_mode`.
- Watching: `start_watching`, `stop_watching`, `is_watching`.
- Accessors: `root`, `memory_path`, `lock_path`, `config`, `scope`.
- Public constants: `MEMORY_FILE`, `STATE_DIR`, `INDEX_FILE`, `LOCK_FILE`, `DEFAULT_DEBOUNCE`.

## Constraints

- Lock order is file lock then state guard, everywhere; no state guard is held across an
  embedding call, a model call, or an embedding-cache await.
- Per-fact and index-document writes are atomic (temporary file plus rename).
- The default layout is byte-for-byte the pre-existing behaviour; `tests/lite_local.rs` and
  `local/tests.rs` must keep passing unchanged.
- No new dependencies, no `unsafe`, `tracing` only, no `println!`, no `unwrap`/`expect` outside
  tests, every public item documented.
- Filesystem failures are `Error::External` carrying the path; nothing is swallowed.

## Routing Table

- `./mod.rs` → module docs (layout, consistency model, keyword/semantic search contract,
  sessions/rollups/write-mode summaries), module declarations, public re-exports, `TARGET`
- `./layout.rs` → on-disk constants, `FileStamp`, atomic document load/write helpers
- `./state.rs` → `LocalCodingMemory` handle, `Inner`, `LocalState`, `ActiveWatch`,
  `IntervalTask`, constructors and cheap accessors
- `./builder.rs` → `LocalCodingMemoryBuilder`: the open sequence, scope, embedding, chunking,
  per-fact flag, and the fresh-open reconciliation
- `./store.rs` → mutation/read API, locking, staleness checks, index rebuilds, fact-type
  resolution
- `./search.rs` → keyword/FTS scoring, semantic/hybrid blend, ranking, recency
- `./embedding.rs` → chunked vector index lifecycle, `refresh_embeddings`, the `CountingEmbedding`
  test support
- `./facts_mode.rs` → opt-in per-fact layout: export/import, index-document reconciliation,
  fact-file refresh, stamp recording
- `./sync_mode.rs` → `sync_now`, `on_trigger`, the interval scheduler slot and the opt-in
  cancellation-aware interval task
- `./session_mode.rs` → JSONL session append/list/recall, keyword+semantic session scoring
- `./daily_mode.rs` → `rollup_day` and the idempotent `write_daily_rollup`
- `./write_mode.rs` → `write_mode` over `MemUpdateChecker` (create/append/merge/replace/skip,
  bounded retries, typed exhaustion error)
- `./watcher.rs` → `start_watching`, `stop_watching`, `is_watching`
- `./tests.rs` → the original 13 store unit tests (moved verbatim from the former `local.rs`)

## Test Strategy

- 47 unit tests in this directory: `tests.rs` (13, unchanged behaviour) plus the new suites in
  `embedding.rs`, `search.rs`, `facts_mode.rs`, `sync_mode.rs`, `session_mode.rs`,
  `daily_mode.rs`, and `write_mode.rs`.
- Deterministic and offline: `tempfile` roots, `HashEmbedding` and counting wrappers, explicit
  `Instant` values for the scheduler, no network, bounded awaits in the watcher tests.
- The sqlite-only assertions (persisted embedding cache, `indexed_count`) are gated with
  `#[cfg(all(test, feature = "sqlite"))]`.
- Gates: `cargo fmt -p openjiuwen-memory`, `cargo test -p openjiuwen-memory --offline --lib
  lite::local` with and without `--features sqlite`, `cargo clippy -p openjiuwen-memory
  --all-targets -- -D warnings` with and without the feature, and `cargo doc -p
  openjiuwen-memory --no-deps` warning-free.
