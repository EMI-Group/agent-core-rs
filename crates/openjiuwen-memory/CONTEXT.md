# openjiuwen-memory

## Intent

Long-term, entity/graph, and lite coding memory for the Rust rewrite of openJiuwen.
Ports the core behavior of the Python package `openjiuwen.core.memory` v0.1.20 (`long_term_memory.py`, `config/`, `manage/`, `graph/`, `lite/`) to idiomatic async Rust.
The Python package is broad (104 files, ~18k lines: migrations, dreaming, external providers, scope-level model config, SQL message store); this crate ports the coherent subset promised by `docs/PARITY.md`: scoped memory-record CRUD, semantic + keyword recall with a documented ranking blend, deterministic dedupe/merge, basic entity/graph memory, and lite coding facts.
Memory is a kernel service for `openjiuwen-agent` rails, `openjiuwen-workflow` memory components, and `openjiuwen-harness`; it depends only downward on `openjiuwen-foundation` (storage, messages, model) and `openjiuwen-retrieval` (`Embedding`).

## API Surface

Crate root `openjiuwen_memory` re-exports every public item; `CRATE_NAME`/`CRATE_VERSION` are exported too.

Execution seam (one instance per application; the Python singleton becomes an explicit `Arc`):

```rust
let ltm = LongTermMemory::in_memory(MemoryEngineConfig::default()).await?;
let scope = MemoryScope::new("alice", "workspace");
ltm.add(&scope, NewMemory::semantic("Alice prefers dark mode"), &AddOptions::default()).await?;
let hits = ltm.search(&scope, "theme preference", &SearchOptions::default()).await?;
let text = openjiuwen_memory::prompt::render_snippets(&hits, 512);
```

Record model (`record`):
- `MemoryType` (`user_profile`, `semantic_memory`, `episodic_memory`, `variable`, `summary`, `coding`, `unknown`).
- `MemoryScope { user_id, scope_id, session_id, agent_id }`; both required ids default to `DEFAULT_ID = "__default__"`; `validate()` enforces non-empty ids, no NUL, `scope_id` without `/` and at most 128 chars.
- `MemoryRecord { id, content, kind, importance, timestamp, updated_at, last_access, access_count, metadata, embedding, source, scope }`.
  JSON is Python-compatible: `content` serializes as `"text"` (accepts `"content"`/`"mem"`), `kind` as `"type"` (accepts `"kind"`/`"mem_type"`), `metadata` as `"fields"` (accepts `"metadata"`), scope fields flattened at top level, timestamps RFC 3339.
- `NewMemory`, `MemoryUpdate`, `MemoryFilter`, `CallOptions`, `AddOptions`, `AddOutcome`, `SearchOptions`, `SearchMode`, `RecallHit`, `MemoryStats`, `AddMemResult`, `CombinedRecall`.

Configuration (`config`):
- `MemoryEngineConfig` with `default_top_k = 5`, `default_threshold = 0.3`, `token_budget`, `searchable_kinds`, `ranking`, `dedupe`, `extraction`, `graph`, `coding`, `forbidden_variables`, `input_msg_max_len = 8192`, `single_turn_history_summary_max_token = 128`, `message_history_limit = 10`, `importance_default = 0.5`.
- `RankingConfig`, `DedupeConfig`, `DedupePolicy`, `ExtractionConfig`, `GraphMemoryConfig`, `CodingMemoryConfig`, `AgentMemoryConfig`, `MemoryScopeConfig`, `StorageBackend`.

Storage (`store`):
- `MemoryStores { kv, vector, document, graph }` with async constructors `in_memory()`, `local_file(root)`, `sqlite(path)` (feature `sqlite`), and `from_parts`.
- `in_memory`/`local_file`/`sqlite` bundle a graph backend; `from_parts(kv, vector, document)` leaves `graph = None`, so graph memory stays absent until `.with_graph(store)`.
- Python-compatible key layout: records `UMD/{user_id}/{scope_id}/{id}`, variables `user_var/{user}/{scope}/{name}`, session variables `session_var/{user}/{scope}/{session}/{name}`, scope config `memory_scope_config/{scope_id}`, vector collections `uid_{user}_gid_{scope}_mtype_{kind}`, message history in document collection `messages`.

Facade (`facade`):
- `LongTermMemory` — `in_memory`, `local_file`, `sqlite`, `with_stores`, `builder`, `engine`, `config`, `graph`, `coding`.
- CRUD: `add`, `add_many`, `get`, `update`, `delete`, `clear`, `list`, `stats`.
- Recall: `search`, `recall` (alias), `search_all` (memories + graph entities).
- Messages: `add_messages`, `get_recent_messages`, `clear_messages` (deletes history rows matching a `MemoryFilter`; records and variables are untouched).
- Variables: `set_variable`, `get_variables`, `delete_variables`.
- Scope config: `set_scope_config`, `get_scope_config`, `delete_scope_config`.
- Python-parity aliases: `search_user_mem`, `search_user_history_summary`, `update_mem_by_id`, `delete_mem_by_id`, `delete_mem_by_user_id`, `delete_mem_by_scope`, `user_mem_total_num`, `get_user_mem_by_page`.

Graph memory (`graph`):
- `GraphMemory`, `Entity`, `Relation`, `EntityExtractor` trait, `HeuristicExtractor`, `LlmExtractor`, `Extraction`, `ExtractedEntity`, `ExtractedRelation`, `GraphRecallHit`, `GraphAddOptions`, `GraphSearchOptions`, `GraphMemoryStats`.

Lite coding memory (`lite`):
- `CodingMemory`, `CodingFact`, `CodingFactType`; facts are `MemoryType::Coding` records with `fact_type`/`name`/`description`/`tags`/`path` metadata plus a rendered Markdown index.

Prompt integration (`prompt`):
- `render_snippets`, `render_records`, `hits_to_system_message`, `hits_to_messages`, `to_memory_json`, `MEMORY_SECTION_TITLE`, `MEMORY_SECTION_TAG`.
- Depends on `openjiuwen_foundation::prompt::BaseMessage` only; there is **no** dependency on `openjiuwen-context` (no cycle, no window manipulation).

## Constraints

- Errors are `openjiuwen_common::error::{Error, Result}`: validation failures `Error::Validation`, missing records `Error::NotFound`, invalid/absent capability `Error::Unsupported`, cancellation `Error::Cancelled`; storage/embedding errors propagate with their original kind.
- Async-first on tokio; every public I/O API is `async`; cancellation via `tokio_util::sync::CancellationToken` carried in `CallOptions` and checked at await points; optional `timeout` uses `tokio::time::timeout`.
- No blocking I/O on async tasks; a `parking_lot` lock is never held across `.await`; per-scope write serialization uses `tokio::sync::Mutex`.
- Every public data type is `serde`-serializable, `snake_case`, tolerant of unknown/missing fields; configs round-trip through JSON.
- No `unsafe`; `tracing` only (targets `openjiuwen_memory::<module>`); no `println!`/`eprintln!`.
- Files stay around ≤1000 lines; split by responsibility inside the owning module directory.
- Default features compile and test offline; no test needs the network, an API key, or wall-clock-dependent ordering.
- `openjiuwen-test-support` is a dev-dependency only; library code never depends on test crates.

## Design Decisions

- **Explicit engine instance, not a process-global singleton.** Python's `LongTermMemory` is a singleton with `register_store` + `set_config`; Rust callers build an `Arc<LongTermMemory>` once and inject it. This keeps tests and multi-tenant processes isolated. `Runner`/harness wiring owns the instance.
- **Richer recall than the Python LTM path.** Python `SearchManager` returns raw vector similarity (`score >= threshold`, top-k) and has no keyword fallback, recency, or importance term. This crate keeps vector similarity as the semantic signal but adds a deterministic hybrid blend so offline keyword-only operation and recency/importance re-ranking are possible. The formula is fixed:
  `semantic = (w_sim * similarity + w_kw * keyword) / (w_sim + w_kw)`;
  `final = (w_sem * semantic + w_imp * importance + w_rec * recency) / (w_sem + w_imp + w_rec)`;
  `similarity = (cosine + 1) / 2` for embedding hits (0 when the record/query has no embedding);
  `keyword = BM25(query, candidate corpus)` normalized by the maximum candidate score to `[0, 1]`;
  `recency = 0.5^(age_secs / half_life_secs)` with a 30-day half-life.
  Defaults: `w_sim = 0.70`, `w_kw = 0.30`, `w_sem = 0.85`, `w_imp = 0.10`, `w_rec = 0.05`. Weights are renormalized per group, so disabling a term is safe. The raw components are returned on `RecallHit` (`similarity`, `keyword`, `importance`, `recency`).
- **Deterministic dedupe/merge instead of the Python LLM conflict checker.** Python recalls 5 candidates at similarity `> 0.75` and asks an LLM to classify redundant/conflicting/none. The Rust subset detects duplicates deterministically: exact normalized text, cosine `>= 0.92`, or token Jaccard `>= 0.75`; the policy is `Merge` (default), `Skip`, `Replace`, or `Add`. Merge keeps the existing id/creation time, takes the longer content, unions metadata (incoming wins), and takes the maximum importance/access count. An LLM-assisted conflict classifier is deferred.
- **Graph memory is a separate facade, as in Python.** `GraphMemory` is not embedded in `LongTermMemory`'s record path; `LongTermMemory::search_all` merges `search` hits with graph hits when graph memory is enabled. Graph nodes are `openjiuwen_foundation::store::GraphNode` (labels `["Entity", <type>]`, properties carry name/scope/embedding) and relations are `GraphEdge`; extraction is heuristic by default and LLM-assisted when a `Model` is attached.
- **Lite coding memory is a thin typed layer on the same engine.** Coding facts are ordinary `MemoryType::Coding` records so they inherit scoping, dedupe, recall, and persistence. Python's Markdown-file source of truth, SQLite FTS5/trigram/`sqlite_vec` index, watchers, and cross-process rebuild lock are deferred; `render_index` reproduces the `MEMORY.md` summary shape.
- **Storage seams.** `MemoryStores` injects the foundation traits, so in-memory, local-file, SQLite, and future vendor backends share one engine. Records live in the KV store (JSON, no embedding), embeddings live in one vector collection per `(user, scope, kind)`, message history lives in a document collection, and variables/scope config live in KV. Listing uses prefix scans instead of Python's 24-char concatenated id lists (no stale-index class of bugs).
- **Embedding is optional.** With no `Embedding`, records are still stored and searched with keyword BM25; `SearchMode::Hybrid` degrades to keyword, `SearchMode::Semantic` returns `Error::Unsupported`. This mirrors the Python lite "keyword-only" degradation and keeps every default test offline.
- **Context seam.** Prompt injection is expressed as `render_snippets` (string for a template placeholder) and `hits_to_system_message` (`BaseMessage`). Agents may interpolate `{{sys_long_term_memory}}` (Python parity) or append the system message; the crate never manipulates a context window.
- **Time is injectable for tests.** Scoring takes `now`; record creation uses `Utc::now()`. Tests assert orderings and formulas, never absolute timestamps.

## Error Strategy

- `MemoryScope::validate` and record/option validation reject bad input with `Error::Validation` naming the offending field and value.
- `update`/`get` of an unknown id: `update` returns `NotFound`, `get`/`delete` return `None`/`false` (no panic).
- Vector collection absence is treated as "no embedding hits"; a dimension mismatch raises the foundation validation error.
- Embedding/model failures keep the provider error kind; batch writes are atomic per record (a failed embedding never stores a half-indexed record).
- Cancellation is checked before expensive work and around embedding/model calls; a cancelled operation returns `Error::Cancelled` and leaves previously committed records intact.
- No silent error swallowing: degraded paths (keyword fallback, heuristic extraction) are explicit `tracing::warn!` plus documented behavior, never an invisible `unwrap_or_default`.

## Routing Table

- `./src/record/` → `MemoryType`, `MemoryScope`, `MemoryRecord`, `NewMemory`, `MemoryUpdate`, `MemoryFilter`, options, results (`record/CONTEXT.md`)
- `./src/config/` → `MemoryEngineConfig` and all nested configs, defaults, validation (`config/CONTEXT.md`)
- `./src/store/` → `MemoryStores`, key layout, record/vector/variable/message/scope-config persistence (`store/CONTEXT.md`)
- `./src/recall/` → keyword tokenization/BM25, scoring formula, dedupe/merge, candidate ranking (`recall/CONTEXT.md`)
- `./src/engine/` → `MemoryEngine`, write path, recall pipeline, stats, per-scope write locks (`engine/CONTEXT.md`)
- `./src/facade/` → `LongTermMemory` facade, messages/variables/scope-config wiring, Python-parity aliases (`facade/CONTEXT.md`)
- `./src/graph/` → entity/relation model, heuristic + LLM extraction, graph store persistence, neighbor expansion (`graph/CONTEXT.md`)
- `./src/lite/` → `CodingMemory` facts, Markdown index rendering, citation helper (`lite/CONTEXT.md`)
- `./src/prompt/` → memory snippets and `BaseMessage` construction for prompt injection (`prompt/CONTEXT.md`)
- `./src/lib.rs` → crate docs, module declaration, public re-exports, lint policy
- `./tests/` → cross-module integration suites (`tests/CONTEXT.md`)

## Test Strategy

- Unit tests live beside each module (`#[cfg(test)] mod tests`): record serde/JSON compatibility, config defaults/validation, key derivation, scoring formula, BM25 ordering, dedupe/merge, heuristic entity extraction, prompt rendering, citation rendering.
- Integration tests under `tests/` (see `tests/CONTEXT.md` for per-file scope): `crud.rs`, `scoping.rs`, `recall.rs`, `dedupe.rs`, `graph.rs` (through the facade), `graph_memory.rs` (`GraphMemory` directly), `lite.rs`, `sqlite.rs` (feature `sqlite`: persistence across reopen; reports 0 tests without the feature), `store_backends.rs` (public `MemoryStores` and local-file reopen), `facade_smoke.rs` (constructors, CRUD, recall, variables, `add_messages` with `ScriptedModel`).
- Deterministic offline fakes: `openjiuwen_retrieval::HashEmbedding`, `openjiuwen_test_support::ScriptedModel`, `tempfile::tempdir` for local-file and SQLite roots.
- Gates (all must be green): `cargo fmt -p openjiuwen-memory -- --check`, `cargo test -p openjiuwen-memory`, `cargo test -p openjiuwen-memory --features sqlite`, `cargo clippy -p openjiuwen-memory --all-targets -- -D warnings` (with and without `--features sqlite`), and `cargo doc -p openjiuwen-memory --no-deps` with no warnings.

## Deferred (Python `core.memory` features not ported)

- External memory-provider SPI and vendor adapters (`external/`: AgentArts, Jiuwen, LakeBase, Mem0, OpenViking, OpenViking) and the `MemoryProvider` trait.
- Migration framework (`migration/`), legacy key prefixes, `KvPrefixRegistry`, AES at-rest codec (`codec/`), distributed KV lock (`DistributedLock`), and `scope_user_mapping`.
- LLM memory analyzer/extractor prompt suite (`process/extract/`), proactive memory, LLM conflict classification (`MemUpdateChecker`), and per-scope model/embedding override resolution.
- Full Python graph stack: Milvus-backed `GraphMemory`, LLM entity dedupe/merge and relation rewiring, time-validity (`valid_since`/`valid_until`), episode/relation retrieval strategies, reranker integration, BFS at the vector-store layer.
- Full lite stack: Markdown/`USER.md` source of truth, SQLite FTS5 trigram + `sqlite_vec`, embedding cache and provider-fingerprint rebuild, filesystem watchers, cross-process lock, session `.jsonl` indexing, coding-memory conflict retry / `WriteMode`.
- `dreaming/` background orchestrator, team memory (`agent_teams` variant), SQL message tables and schema migrations, pagination cursor parity beyond page index/size.

## Python Mapping

| Python | Rust |
|---|---|
| `long_term_memory.py::LongTermMemory` | `facade::LongTermMemory` |
| `config/config.py::{MemoryEngineConfig, MemoryScopeConfig, AgentMemoryConfig}` | `config::{MemoryEngineConfig, MemoryScopeConfig, AgentMemoryConfig}` |
| `manage/mem_model/memory_unit.py::{MemoryType, FragmentMemoryUnit, VariableUnit, SummaryUnit}` | `record::{MemoryType, MemoryRecord, NewMemory}` |
| `core/foundation/store/base_memory_index.py::MemoryDoc` | `record::MemoryRecord` (JSON `text`/`type`/`fields`) |
| `manage/index/write_manager.py::WriteManager` | `engine::MemoryEngine` write path |
| `manage/search/search_manager.py::SearchManager` | `engine::MemoryEngine` recall path + `recall` scoring |
| `manage/update/mem_update_checker.py::MemUpdateChecker` | `recall::dedupe` deterministic policies |
| `manage/index/variable_manager.py` | `store` variable keys + `facade` variable methods |
| `manage/mem_model/message_manager.py` | `store::messages` + `facade::add_messages`/`get_recent_messages` |
| `graph/graph_memory/base.py::GraphMemory` | `graph::GraphMemory` |
| `graph/extraction/*` | `graph::extract::{HeuristicExtractor, LlmExtractor}` |
| `lite/manager.py::MemoryIndexManager` | `lite::CodingMemory` |
| `lite/{frontmatter,internal}.py` fact model / index | `lite::{CodingFact, CodingFactType}` + `render_index` |

## Deviations from `docs/ARCHITECTURE.md`

- `LongTermMemory` is instance-scoped (`Arc`), not a process-global singleton; the Python `register_store`/`set_config` pair becomes `MemoryStores` injection plus `MemoryEngineConfig`.
- The crate does not depend on `openjiuwen-context` or `openjiuwen-session`; prompt injection is expressed as strings/messages from `openjiuwen-foundation`.
- Ranking adds keyword/importance/recency terms beyond the Python LTM vector-only score (documented formula above); this is an additive parity extension, not a wire-format change.
- `MemoryType` adds `coding` (an extension used by `lite`); Python's `unknown` remains the default.
