# tests

## Intent

Cross-module integration suites for `openjiuwen-memory`.
Unit tests live beside the code; these files drive the public facade and the public store/graph seams end to end with deterministic fakes and both storage backends.

## Files

- `crud.rs` (10 tests) → `add`/`add_many`/`get`/`update`/`delete`/`clear`/`list`/`stats`, input and scope validation, the Python-compatible `MemoryRecord` JSON shape (`text`/`type`/`fields`), and cancellation without side effects.
- `scoping.rs` (13 tests) → user/scope/session/agent isolation, user and session variables, scope configs, message history (session filter and configured limit), and the Python-parity aliases (`search_user_mem`, `search_user_history_summary`, `update_mem_by_id`, `delete_mem_by_*`, `user_mem_total_num`, `get_user_mem_by_page`).
- `recall.rs` (11 tests) → semantic ordering with `HashEmbedding`, keyword-only fallback (no embedding), `Semantic` without an embedding -> `Unsupported`, threshold/top-k, hybrid blend against `recall::blend`, importance/recency effects, token budget, `render_snippets` order, `search_all` with and without graph, and empty-query validation.
- `dedupe.rs` (10 tests) → exact normalized duplicate, near-duplicate cosine/Jaccard merge, `Skip`/`Replace`/`Add` policies, metadata union/importance max/id stability, dedupe inside one `add_many` batch, and texts below the similarity thresholds staying separate.
- `graph.rs` (9 tests) → graph memory through the `LongTermMemory` facade: `ltm.graph()`, facade-level extraction/persistence/recall, `search_all` merging and degradation, scope isolation, delete/clear/stats, `ScriptedModel` extraction with malformed-output fallback.
- `graph_memory.rs` (12 tests) → the `GraphMemory` facade directly: extraction wiring, seed scoring, BFS neighbor expansion, keyword-only and `HashEmbedding` search, scope isolation, entity deletion with relation cleanup, `clear` per scope, validation/cancellation, and embedding computation only when configured.
- `lite.rs` (10 tests) → `CodingMemory` write/read/edit/delete/list/search, upsert by name with a stable record id, index grouping and truncation (including the title), citations, and scope/user isolation; keyword BM25 only, no embedding model attached.
- `sqlite.rs` (5 tests with `--features sqlite`, 0 without) → `MemoryStores::sqlite` persistence of records, variables, scope configs, messages, embeddings, and graph entities across drop/reopen, deletes/clears remaining visible, and independent databases not sharing state.
- `store_backends.rs` (2 tests) → the public `MemoryStores` surface only: a record written through `stores.kv` and a message through `stores.document` via `keys::record_key`/`MESSAGES_COLLECTION`, read back in memory and after closing and reopening the same local-file directory.
- `facade_smoke.rs` (5 tests) → end-to-end facade smoke: constructors, CRUD + stats, keyword and `HashEmbedding` recall, variables and scope config, and the `add_messages` history/extraction pipeline with both the heuristic path and a `ScriptedModel`.

## Conventions

- Every integration file opens with the crate-level `#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]` (first line, or immediately after the `//!` module docs; `store_backends.rs` allows only `unwrap_used` and `expect_used`).
- Async tests use `#[tokio::test]`.
- Use `LongTermMemory::in_memory(MemoryEngineConfig::default())` unless the test targets a backend; `store_backends.rs` constructs `MemoryStores` directly, and `lite.rs` builds a `MemoryEngine` behind `CodingMemory`.
- Use `.embedding(Arc::new(HashEmbedding::new(16)))` for semantic tests; leave the embedding unset for keyword/BM25 fallback tests.
- Use `ScriptedModel` for LLM extraction paths and hand-built `Extraction`/`ExtractedEntity` values where a heuristic quirk would obscure the assertion.
- Use fixed `DateTime::from_timestamp(1_700_000_000 + offset, 0)` values via `NewMemory::with_timestamp`, pass `now` only where the API exposes it, and never assert wall-clock values.
- Local-file and SQLite roots come from `tempfile::tempdir()`, one directory per test; never reuse a global temp path.
- Add a behavior-port comment naming the Python test where one exists (e.g. `// ports tests/unit_tests/core/memory/manage/test_manage.py`).
- Tests must pass offline and stay deterministic under `--test-threads` > 1.

## Gotchas

- A whole-crate `#![cfg(feature = "...")]` gate strips the crate docs, so `sqlite.rs` places `#![allow(missing_docs)]` before the `cfg` attribute; an `allow` after it would be stripped too and the workspace `missing_docs` deny would fire on the empty crate root under `-D warnings`.
- Session and agent dimensions isolate listings and statistics but not point reads: record keys are `UMD/{user_id}/{scope_id}/{id}`, so `get` is only user/scope-scoped, and a session-free scope's `list`/`stats` includes all of its session/agent records (as `scoping.rs` asserts).
- `dedupe.rs` uses a local constant-vector `Embedding` impl for the cosine-merge cases because `HashEmbedding`'s mean-pooled signed hash vectors for unrelated texts are nearly orthogonal (normalized similarity around 0.5), below the `0.92` dedupe threshold; do not "simplify" those tests to `HashEmbedding`.

## Gates

- `cargo fmt -p openjiuwen-memory -- --check`
- `cargo test -p openjiuwen-memory`
- `cargo test -p openjiuwen-memory --features sqlite`
- `cargo clippy -p openjiuwen-memory --all-targets -- -D warnings`, with and without `--features sqlite`
