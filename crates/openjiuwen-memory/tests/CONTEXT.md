# tests

## Intent

Cross-module integration suites for `openjiuwen-memory`.
Unit tests live beside the code; these files exercise the public facade end to end with deterministic fakes and both storage backends.

## Files

- `crud.rs` → `add`/`add_many`/`get`/`update`/`delete`/`clear`/`list`/`stats`, JSON round-trip (`text`/`type`/`fields`), record id shape, blank-content rejection, cancellation, duplicate policy outcomes.
- `scoping.rs` → user/scope/session/agent isolation, variables (user and session), scope configs, Python-parity aliases (`search_user_mem`, `update_mem_by_id`, `delete_mem_by_*`, `user_mem_total_num`, `get_user_mem_by_page`), message history.
- `recall.rs` → semantic ordering with `HashEmbedding`, keyword-only fallback (no embedding), hybrid blend, `Semantic` without embedding -> `Unsupported`, threshold/top-k, token budget, importance/recency effects, `render_snippets` order, empty-query validation.
- `dedupe.rs` → exact normalized duplicate, near-duplicate cosine/Jaccard merge, `Skip`/`Replace`/`Add` policies, metadata union/importance max/id stability, dedupe inside one `add_many` batch.
- `graph.rs` → heuristic extraction, `ScriptedModel` LLM extraction + malformed-response fallback, relations, neighbor expansion, scope isolation, delete/clear/stats, `search_all` merging/dagradation.
- `lite.rs` → coding fact write/read/edit/delete/list/search, upsert by name, index rendering/truncation, citations, scope isolation.
- `sqlite.rs` (`#![cfg(feature = "sqlite")]`) → write records/variables/messages, drop the instance, reopen the same path, observe the committed state.

## Conventions

- Every integration crate starts with `#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]`.
- Async tests use `#[tokio::test]`.
- Use `LongTermMemory::in_memory(MemoryEngineConfig::default())` unless the test targets a backend; use `.embedding(Arc::new(HashEmbedding::new(16)))` for semantic tests, otherwise no embedding for keyword fallback.
- Use fixed `DateTime::from_timestamp(...)` values and explicit `with_timestamp`, and pass `now` only where the API exposes it; never assert wall-clock values.
- Local-file/SQLite roots come from `tempfile::tempdir()`; do not reuse a global temp path.
- Behavior-port comments name the Python test where one exists (e.g. `// ports tests/unit_tests/core/memory/manage/test_manage.py`).
- Tests must pass offline and be deterministic under `--test-threads` > 1.

## Gates

`cargo test -p openjiuwen-memory` and `cargo test -p openjiuwen-memory --features sqlite`, plus `cargo clippy -p openjiuwen-memory --all-targets -- -D warnings` (with and without `--features sqlite`).
