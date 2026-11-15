# local

## Intent

The file-backed coding-memory store rooted at one directory: a Markdown representation that a human can edit, derived indexes (FTS5 and an optional chunked vector index), a cross-process advisory lock, and a debounced watcher.
Ports the behaviour of Python's `MemoryIndexManager` for one coding-memory root and adds per-fact files with the `MEMORY.md`/`USER.md` split, JSONL session recall, caller-driven sync triggers, `daily_memory` rollups, and write modes with a bounded conflict-retry checker.
The authoritative representation is the Markdown document by default and the per-fact files when `LocalCodingMemoryBuilder::fact_files(true)` is set; every derived index may be rebuilt.
The module-level prose (layout, consistency model, search contract, module split) lives in `mod.rs`; this file is the routing map for the directory.

## API Surface

`LocalCodingMemory` (re-exported as `crate::lite::local::LocalCodingMemory`) groups its API by file.

- Construction: `open`, `open_in_scope`, `builder` (`LocalCodingMemoryBuilder::{new, scope, embedding, chunking, fact_files, open}`).
- Facts: `write`, `read`, `edit`, `delete`, `list`, `search`, `render_index`, `reindex`, `indexed_count` (feature `sqlite`), `markdown`, `reload_count`.
- Embeddings: `embedding`, `has_embedding`, `chunking`, `embedding_chunks`, `refresh_embeddings`, `embedding_cache_len`.
- Per-fact files: `fact_files_enabled`, `facts_dir`, `user_path`, `export_index`, `import_index`.
- Sync: `sync_now`, `on_trigger`, `start_interval_sync`, `stop_interval_sync`, `is_interval_sync_running`.
- Sessions: `sessions_dir`, `append_session`, `session_records`, `search_sessions`.
- Daily rollups: `daily_dir`, `rollup_day`, `write_daily_rollup`.
- Write modes: `write_mode`.
- Watching: `start_watching`, `stop_watching`, `is_watching`.
- Accessors: `root`, `memory_path`, `lock_path`, `config`, `scope`.
- Public constants: `MEMORY_FILE`, `STATE_DIR`, `INDEX_FILE`, `LOCK_FILE`, `DEFAULT_DEBOUNCE`.

## Constraints

- Lock order is file lock then state guard, everywhere; no state guard is held across an embedding call, a model call, or an embedding-cache await.
- The cached document carries a `version` bumped on every change, and the embedding index records the version it was built from, so a semantic recall after an external edit rebuilds the index instead of scoring stale vectors.
- A failed embedding rebuild is never installed: the new index is built off to the side and swapped in only on success, so the previous index (or none, after a failed first build) stays and the next recall retries.
- Per-fact and index-document writes are atomic (temporary file plus rename).
- The default layout stays the single-document layout (`MEMORY.md` plus `.memory/`); `tests/lite_local.rs` and `local/tests.rs` pin it.
- No additional dependencies, no `unsafe`, `tracing` only, no `println!`, no `unwrap`/`expect` outside tests, every public item documented.
- Filesystem failures are `Error::External` carrying the path; the documented exceptions are a missing file (read as absent) and a failed modification-time read in `FactFiles::stamp` (`newest_ms = 0`).

## Routing Table

- `./mod.rs` → module docs (layout, consistency model, keyword/semantic search contract, sessions/rollups/write-mode summaries), module declarations, public re-exports, `TARGET`
- `./layout.rs` → on-disk constants, `FileStamp`, atomic document load/write helpers
- `./state.rs` → the `LocalCodingMemory` handle, `Inner`, `LocalState`, `ActiveWatch`, the interval-task slot, constructors and cheap accessors
- `./builder.rs` → `LocalCodingMemoryBuilder`: the open sequence, scope, embedding, chunking, per-fact flag, and the open-time reconciliation
- `./store.rs` → mutation/read API, locking, staleness checks, index rebuilds, fact-type resolution
- `./search.rs` → keyword/FTS scoring, semantic/hybrid blend, ranking, recency
- `./embedding.rs` → chunked vector index lifecycle, `refresh_embeddings`, the `CountingEmbedding` test support
- `./facts_mode.rs` → opt-in per-fact layout: export/import, index-document reconciliation, fact-file refresh, stamp recording
- `./sync_mode.rs` → `sync_now`, `on_trigger`, the interval scheduler slot (`IntervalTask`) and the opt-in cancellation-aware interval task
- `./session_mode.rs` → JSONL session append/list/recall, keyword+semantic session scoring
- `./daily_mode.rs` → `rollup_day` and the idempotent `write_daily_rollup`
- `./write_mode.rs` → `write_mode` over `MemUpdateChecker` (create/append/merge/replace/skip, bounded retries, typed exhaustion error)
- `./watcher.rs` → `start_watching`, `stop_watching`, `is_watching`
- `./tests.rs` → the 13 store unit tests covering open/layout, read/write/edit/delete, search, the summary index, reloads/`reindex`, locking, the watcher, and the handle's `Send + Sync` bounds

## Known Issues

- A cached chunk vector whose length differs from the query vector silently scores `0.0` (`lite::embed::EmbeddingIndex::search`), so a model whose dimension changed under the same `model_name` turns semantic recall into an empty, error-free result (`lite::embed` reports the same defect).
- When the `facts/` directory is gone or every file in it fails to decode, `refresh_facts` adopts an empty fact set, so the next export re-renders both index documents empty while the undecodable files stay on disk; a `tracing::warn!` names the skipped files but nothing keeps the previous document.
- "No information" is decided per index document by absence or whitespace-only content: a document that survives with its front matter or title but no `###` entry is authoritative-empty, so the fact files of that level are deleted by the next export.
- `start_watching`'s reload task logs a failed `ensure_fresh` as a warning and continues, so `search`/`list` can stay stale with no observable signal, and `is_watching` reports `true` even if that reload task died (`ActiveWatch.task` is never polled).

## Test Strategy

- 51 unit tests with default features and 52 with `--features sqlite`: `tests.rs` 13, `facts_mode.rs` 11, `session_mode.rs` 6, `write_mode.rs` 6, `search.rs` 5, `embedding.rs` 5 (one sqlite-gated), `sync_mode.rs` 3, `daily_mode.rs` 3.
- `facts_mode.rs` pins the reconciliation rules: an export that removes only entries deleted from a non-empty document, an empty or missing index document keeping that level's fact files, an undecodable fact file surviving an export and a delete, the `USER.md` precedence rule, and the byte-for-byte hand-edit re-import.
- Deterministic and offline: `tempfile` roots, `HashEmbedding` and counting wrappers, explicit `Instant` values for the scheduler, no network, bounded awaits in the watcher tests.
- The sqlite-only assertions (persisted embedding cache, `indexed_count`) are gated with `#[cfg(feature = "sqlite")]`.
- Gates: `cargo fmt -p openjiuwen-memory`, `cargo test -p openjiuwen-memory --offline --lib lite::local` with and without `--features sqlite`, `cargo clippy -p openjiuwen-memory --all-targets -- -D warnings` with and without the feature, and `cargo doc -p openjiuwen-memory --no-deps` warning-free.
