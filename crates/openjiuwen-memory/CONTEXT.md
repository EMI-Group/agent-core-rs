# openjiuwen-memory

## Intent

Long-term, entity/graph, lite coding, analyzer, external-provider, migration, dreaming, and team memory for the Rust rewrite of openJiuwen.
Ports the behavior of the Python package `openjiuwen.core.memory` v0.1.20 (`long_term_memory.py`, `config/`, `manage/`, `graph/`, `lite/`, `process/extract/`, `external/`, `migration/`, `dreaming/`) plus the `agent_teams.memory` data layer to idiomatic async Rust.
Scope: scoped memory-record CRUD, semantic + keyword recall with a documented ranking blend, deterministic dedupe/merge, entity/graph memory, engine-backed coding facts plus a local Markdown + FTS5 + watcher + lock coding stack, rule-first fact extraction and conflict classification with model fallback, an external memory-provider SPI whose vendor adapters are typed `Unsupported`, a forward-only SQLite migration runner with legacy-prefix read compatibility, durable dream-job data/persistence/planner, and team-scoped memory data/persistence.
Every subsystem is a documented subset: each module `CONTEXT.md` lists what it defers, and the crate-level list is under Deferred below.
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
- `MemoryStores { kv, vector, document, graph }` with async constructors `in_memory()`, `local_file(root)`, `sqlite(path)` (feature `sqlite`), `sqlite_migrated(path)` (feature `sqlite`: migrate the schema first, then open), and `from_parts`.
- `in_memory`/`local_file`/`sqlite`/`sqlite_migrated` bundle a graph backend; `from_parts(kv, vector, document)` leaves `graph = None`, so graph memory stays absent until `.with_graph(store)`.
- Key layout: records `UMD/{user_id}/{scope_id}/{id}`, variables `user_var/{user}/{scope}/{name}`, session variables `session_var/{user}/{scope}/{session}/{name}`, scope config `memory_scope_config/{scope_id}`, vector collections `uid_{user}_gid_{scope}_mtype_{kind}`, message history in document collection `messages`.
- `store::keys` also carries the current↔legacy prefix mapping: `legacy_alternatives`, `current_equivalent`, `legacy_prefix`, `current_prefix`, and the legacy key builders.

Migrations (`store`):
- `store::migration`: `Migration`, `MigrationRegistry`, `builtin_migrations`, `default_registry`, and (feature `sqlite`) `MigrationRunner`, `MigrationReport`, `MigrationStatus`, `AppliedMigration`, `SCHEMA_VERSION_TABLE = "memory_schema_version"`, `migrate_sqlite(path)`.
- `store::legacy` is crate-private: read-only fallback to the legacy prefixes (`user_mem`, `user_variable`, `session_variable`, `scope_config`) with the current layout winning.
- Layout, migration model, and legacy rules are detailed in `src/store/CONTEXT.md`.

Facade (`facade`):
- `LongTermMemory` — `in_memory`, `local_file`, `sqlite` (sqlite), `sqlite_migrated` (sqlite), `with_stores`, `builder`, `engine`, `config`, `graph`, `coding`, `analyzer`.
- CRUD: `add`, `add_many`, `get`, `update`, `delete`, `clear`, `list`, `stats`.
- Recall: `search`, `recall` (alias), `search_all` (memories + graph entities).
- Messages: `add_messages`, `get_recent_messages`, `clear_messages` (deletes history rows matching a `MemoryFilter`; records and variables are untouched).
- Variables: `set_variable`, `get_variables`, `delete_variables`.
- Scope config: `set_scope_config`, `get_scope_config`, `delete_scope_config`.
- Analyzer and provider wiring: `remember` (analyze + write, returns `RememberReport`), `consolidate` (returns a `ConsolidationPlan`, applies nothing), `provider` (an `EngineProvider` over the same engine).
- Subsystem constructors: `team_memory`, `team_memory_sqlite` (sqlite), `dreaming_in_memory`, `dreaming_sqlite` (sqlite), `open_local_coding(root)` for the local lite stack.
- `LongTermMemoryBuilder::analyzer` injects a custom `MemoryAnalyzer`; `builder().model(..)` supplies the model for the default rule-first analyzer.
- Python-parity aliases: `search_user_mem`, `search_user_history_summary`, `update_mem_by_id`, `delete_mem_by_id`, `delete_mem_by_user_id`, `delete_mem_by_scope`, `user_mem_total_num`, `get_user_mem_by_page`.

Graph memory (`graph`):
- `GraphMemory`, `Entity`, `Relation`, `EntityExtractor` trait, `HeuristicExtractor`, `LlmExtractor`, `Extraction`, `ExtractedEntity`, `ExtractedRelation`, `GraphRecallHit`, `GraphAddOptions`, `GraphSearchOptions`, `GraphMemoryStats`.

Lite coding memory (`lite`):
- Two paths share the fact model: `CodingMemory` (facts as engine records) and `LocalCodingMemory` (Markdown source of truth + derived FTS5 index + advisory lock + watcher).
- `CodingFact`, `CodingFactType`, `render_index`, `citation`; `MarkdownMemory`/`FrontMatter` own the `MEMORY.md` document format.
- `FtsIndex`/`FtsHit` (feature `sqlite`) own the derived index, `FileLock` the cross-process `flock`, `MemoryWatcher`/`WatchEvent`/`WatchKind` the debounced file watch.
- `local::{MEMORY_FILE, STATE_DIR, INDEX_FILE, LOCK_FILE, DEFAULT_DEBOUNCE}`; layout, format, and scoring are in `src/lite/CONTEXT.md`.

Analyzer (`analyzer`):
- Two seams: `FactExtractor` (`RuleExtractor` default, `LlmExtractor` with rule fallback) and `ConflictClassifier` (`RuleClassifier` default, `LlmClassifier` with rule fallback).
- `MemoryAnalyzer` orchestrates both and returns `ExtractionResult` / `ConsolidationPlan`; supporting types are `AnalyzerConfig`, `ExtractorKind`, `ClassifierKind`, `ConversationTurn`, `ExtractionInput`, `ExtractedFact`, `ConflictKind`, `ConflictVerdict`, `ConsolidationAction`.
- `parse_fact_response`, `parse_conflict_response`, and the extraction/conflict prompts and JSON schemas are public.
- `analyzer::LlmExtractor` is a distinct type from `graph::LlmExtractor`: the crate root re-exports the graph one (with `EXTRACTION_SYSTEM_PROMPT`), so the analyzer's is reached as `analyzer::LlmExtractor`.

External providers (`external`):
- `MemoryProvider` SPI plus `ProviderCapabilities` and `ProviderHealth`.
- Registry types: `ProviderRegistry`, `ProviderFactory`, `ProviderConfig`, `ProviderKind`, and `EngineProvider`/`LocalProviderFactory` (`EngineProvider` adapts the local engine).
- Vendor handles: `AgentArtsMemoryProvider`, `JiuwenMemoryProvider`, `LakeBaseMemoryProvider`, `Mem0MemoryProvider`, `OpenVikingMemoryProvider`, plus `vendor_factories()`; each handle constructs without I/O and returns typed `Unsupported` from every operation (`src/external/CONTEXT.md`).

Dreaming (`dreaming`):
- Job model: `DreamJob`, `DreamState`, `DreamKind`, `DreamTrigger`, `DreamMetrics`, `DreamOutcome`, `DreamRunSummary`, `DreamConfig`, `DreamFilter`.
- Persistence seam: `DreamRepository`, `InMemoryDreamRepository`, `SqliteDreamRepository` (feature `sqlite`, table `dream_jobs`).
- Planner: `DreamPlanner` (alias `DreamingEngine`), the `DreamConsolidator` seam, and `NoopConsolidator` (`src/dreaming/CONTEXT.md`).

Team memory (`team`):
- Records: `TeamMemoryRecord`, `TeamMemoryScope`, `TeamMemoryKind`, `TeamVisibility`, `NewTeamMemory`, `TeamMemoryUpdate`, `TeamMemoryFilter`, `TeamMemoryStats`, `TeamSearchHit`.
- Store seam: `TeamMemoryStore`, `TeamSearchOptions`, `InMemoryTeamMemoryStore`, `SqliteTeamMemoryStore` (feature `sqlite`, table `team_memories`).
- Facade and projection: `TeamMemory` (`in_memory`, `sqlite`, `remember`, `search`, `forget`, `stats`, `to_jsonl`/`from_jsonl`), `TeamMemoryRecord::to_memory_record`/`from_memory_record`, `TEAM_MEMORY_METADATA_KEY` (`src/team/CONTEXT.md`).

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
- `notify` (file watching), `nix` (unix advisory `flock` for `FileLock`) and `regex` (rule extractor) are always-compiled workspace dependencies; on non-unix targets `FileLock` returns `Error::Unsupported` rather than degrading.
- All SQL lives behind the `sqlite` feature — the FTS5 table `coding_facts_fts`, `memory_schema_version`, `dream_jobs`, `team_memories`; every non-`sqlite` path must compile and be tested without the feature.
- Cross-process coordination is advisory file locking only: no daemon, no broker, no background process.
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
- **Deterministic dedupe stays the write path.** Duplicates are detected deterministically: exact normalized text, cosine `>= 0.92`, or token Jaccard `>= 0.75`; the policy is `Merge` (default), `Skip`, `Replace`, or `Add`. Merge keeps the existing id/creation time, takes the longer content, unions metadata (incoming wins), and takes the maximum importance/access count. Redundancy and conflict classification is a separate, explicit step: `analyzer::ConflictClassifier` (rules by default, LLM with rule fallback) produces a `ConsolidationPlan`, and `LongTermMemory::consolidate` is its only caller — the write path never applies a plan on its own.
- **Graph memory is a separate facade, as in Python.** `GraphMemory` is not embedded in `LongTermMemory`'s record path; `LongTermMemory::search_all` merges `search` hits with graph hits when graph memory is enabled. Graph nodes are `openjiuwen_foundation::store::GraphNode` (labels `["Entity", <type>]`, properties carry name/scope/embedding) and relations are `GraphEdge`; extraction is heuristic by default and LLM-assisted when a `Model` is attached.
- **Two lite paths, deliberately not mirrored.** `CodingMemory` stores coding facts as ordinary `MemoryType::Coding` records, so they inherit scoping, dedupe, recall, and every storage backend. `LocalCodingMemory` is the local stack: a human-editable `MEMORY.md` is the source of truth, an FTS5 index under `.memory/` is derived and may always be rebuilt, an advisory file lock serializes cross-process writes, and a debounced watcher reloads external edits. The two paths are not synchronized in either direction; `LongTermMemory::open_local_coding` is the documented bridge and copies no records.
- **Rule-first analyzer with model fallback.** `ExtractorKind::Rule`/`ClassifierKind::Rule` are the defaults and work offline; an LLM extractor/classifier retries `model_retries + 1` attempts, counts unparseable output as a failed attempt, records the first failure reason as a `degraded` flag on the result/verdicts, and checks cancellation before the first model call so a cancelled call makes zero model invocations.
- **External provider SPI is an injected seam.** `MemoryProvider` plus `ProviderRegistry` are passed in, never global. `EngineProvider` adapts the local engine, so the SPI is exercised by real code; the five vendor adapters register factories whose every operation returns `Unsupported` naming the missing SDK/transport (`agentarts-sdk`, `jiuwen_memory`, `httpx.AsyncClient`, `mem0ai`, the OpenViking client), and `ProviderConfig` redacts credentials in `Debug`.
- **Migrations are forward-only.** Each step's DDL and its `memory_schema_version` row commit in one transaction, the registry must start at v1 and stay contiguous, and a recorded version above the registry is reported as up to date rather than downgraded. Legacy key prefixes are read-only compatibility: the current key wins, writes use the current layout only, and deletes remove both.
- **Dreaming and team memory are data/persistence/planner layers with injected seams.** `DreamConsolidator` owns the memory mutation and `TeamMemoryStore` owns the rows; the crate starts no background loop, timer, or purge, and no cross-process lease exists — the caller owns `run_due`/`run_once`/`purge`.
- **Storage seams.** `MemoryStores` injects the foundation traits, so in-memory, local-file, SQLite, and future vendor backends share one engine. Records live in the KV store (JSON, no embedding), embeddings live in one vector collection per `(user, scope, kind)`, message history lives in a document collection, and variables/scope config live in KV. Listing uses prefix scans instead of Python's 24-char concatenated id lists (no stale-index class of bugs).
- **Embedding is optional.** With no `Embedding`, records are still stored and searched with keyword BM25; `SearchMode::Hybrid` degrades to keyword, `SearchMode::Semantic` returns `Error::Unsupported`. This mirrors the Python lite "keyword-only" degradation and keeps every default test offline.
- **Retrieval integration is the `Embedding` trait only.** No public API accepts `openjiuwen_retrieval` documents, chunks, or search hits; the hybrid blend operates on embedding similarity plus BM25 (`RankingConfig`), and `CombinedRecall` merges memory hits with graph entities, not retrieval results.
- **`LocalCodingMemory` works without the `sqlite` feature.** The Markdown document and keyword scan are always available; `sqlite` only adds the derived `FtsIndex` FTS5 merge, so the same lite suites pass in both builds and stay fully offline.
- **Context seam.** Prompt injection is expressed as `render_snippets` (string for a template placeholder) and `hits_to_system_message` (`BaseMessage`). Agents may interpolate `{{sys_long_term_memory}}` (Python parity) or append the system message; the crate never manipulates a context window.
- **Time is injectable for tests.** Scoring, dream-job claiming, team filters, and expiry take `now`; record creation uses `Utc::now()`. Tests assert orderings and formulas, never absolute timestamps.

## Error Strategy

- `MemoryScope::validate` and record/option validation reject bad input with `Error::Validation` naming the offending field and value.
- `update`/`get` of an unknown id: `update` returns `NotFound`, `get`/`delete` return `None`/`false` (no panic).
- Vector collection absence is treated as "no embedding hits"; a dimension mismatch raises the foundation validation error.
- Embedding/model failures keep the provider error kind; batch writes are atomic per record (a failed embedding never stores a half-indexed record).
- Cancellation is checked before expensive work and around embedding/model calls; a cancelled operation returns `Error::Cancelled` and leaves previously committed records intact.
- No silent error swallowing: degraded paths (keyword fallback, heuristic extraction, LLM-to-rule fallback, tolerant record decoding, non-unix locking) are explicit `tracing::warn!`, a recorded `degraded` reason, or a typed `Error::Unsupported`, never an invisible `unwrap_or_default`.

## Routing Table

- `./src/record/` → `MemoryType`, `MemoryScope`, `MemoryRecord`, `NewMemory`, `MemoryUpdate`, `MemoryFilter`, options, results (`record/CONTEXT.md`)
- `./src/config/` → `MemoryEngineConfig` and all nested configs, defaults, validation (`config/CONTEXT.md`)
- `./src/store/` → `MemoryStores`, key layout, legacy prefixes, migrations, record/vector/variable/message/scope-config persistence (`store/CONTEXT.md`)
- `./src/recall/` → keyword tokenization/BM25, scoring formula, dedupe/merge, candidate ranking (`recall/CONTEXT.md`)
- `./src/engine/` → `MemoryEngine`, write path, recall pipeline, stats, per-scope write locks (`engine/CONTEXT.md`)
- `./src/facade/` → `LongTermMemory` facade, messages/variables/scope-config wiring, subsystem constructors, Python-parity aliases (`facade/CONTEXT.md`)
- `./src/graph/` → entity/relation model, heuristic + LLM extraction, graph store persistence, neighbor expansion (`graph/CONTEXT.md`)
- `./src/analyzer/` → `FactExtractor`/`ConflictClassifier` seams, rule + LLM implementations, `MemoryAnalyzer` and consolidation plans (`analyzer/CONTEXT.md`)
- `./src/external/` → `MemoryProvider` SPI, `EngineProvider`, registry/config, vendor `Unsupported` adapters (`external/CONTEXT.md`)
- `./src/dreaming/` → dream job model, repositories (in-memory/SQLite), planner and consolidator seam (`dreaming/CONTEXT.md`)
- `./src/team/` → team memory records, store contract, in-memory/SQLite backends, `MemoryRecord` projection (`team/CONTEXT.md`)
- `./src/lite/` → `CodingMemory` and `LocalCodingMemory`, `MEMORY.md` format, FTS5 index, file lock, watcher, index rendering (`lite/CONTEXT.md`)
- `./src/prompt/` → memory snippets and `BaseMessage` construction for prompt injection (`prompt/CONTEXT.md`)
- `./src/lib.rs` → crate docs, module declaration, public re-exports, lint policy
- `./tests/` → cross-module integration suites (`tests/CONTEXT.md`)

## Test Strategy

- Unit tests live beside each module (`#[cfg(test)] mod tests`): record serde/JSON compatibility, config defaults/validation, key derivation and legacy mapping, scoring formula, BM25 ordering, dedupe/merge, heuristic entity extraction, rule/LLM extraction and fallback, conflict verdicts, prompt rendering, citation rendering, Markdown round trips, FTS5 and lock/watcher behavior, migration validation and rollback, dream claim/retry transitions, team visibility and ranking.
- Integration tests under `tests/` (see `tests/CONTEXT.md` for per-file scope): `crud.rs`, `scoping.rs`, `recall.rs`, `dedupe.rs`, `graph.rs` (through the facade), `graph_memory.rs` (`GraphMemory` directly), `lite.rs` (engine-backed `CodingMemory`), `lite_local.rs` (Markdown round trip, FTS5 bookkeeping, reindex, two handles on one root, unix watcher reload), `sqlite.rs` (feature `sqlite`: persistence across reopen; reports 0 tests without the feature), `store_backends.rs` (public `MemoryStores` and local-file reopen), `facade_smoke.rs` (constructors, CRUD, recall, variables, `add_messages` with `ScriptedModel`).
- Subsystem integration binaries: `analyzer_wiring.rs` (rule extraction through `remember`, LLM fallback surfaced as `degraded`, cancellation writing nothing, `consolidate` plans per `DedupePolicy`), `external_provider.rs` (`provider()` as a view of the same engine, capability/health tracking the embedding model, vendor `Unsupported`, registry semantics, redaction), `migration.rs` (feature `sqlite`: builtin versions, no-op rerun, `sqlite_migrated` reopen, custom registry through `MigrationRunner`), `team_memory.rs` (CRUD/filters/ranking/projection plus SQLite persistence across a reopen), `dreaming.rs` (claim/retry/cancel/noop sweeps plus SQLite job durability).
- Deterministic offline fakes: `openjiuwen_retrieval::HashEmbedding`, `openjiuwen_test_support::ScriptedModel`, `TeamMemory::in_memory`, `InMemoryDreamRepository`, `tempfile::tempdir` for local-file and SQLite roots.
- Gates (all must be green): `cargo fmt -p openjiuwen-memory -- --check`, `cargo test -p openjiuwen-memory`, `cargo test -p openjiuwen-memory --features sqlite`, `cargo clippy -p openjiuwen-memory --all-targets -- -D warnings` (with and without `--features sqlite`), and `cargo doc -p openjiuwen-memory --no-deps` with no warnings.

## Deferred (Python `core.memory` features not ported)

- External providers: real vendor transports and SDKs (AgentArts, Jiuwen, LakeBase, mem0, OpenViking), payload shaping, retries, failover, and the Python tool-rail surface (`get_tool_schemas`, `handle_tool_call`, `prefetch`, `sync_turn`, `system_prompt_block`, `on_session_end`) — the Rust SPI is record-only.
- Migrations: Python's per-backend migrators (`SQLMigrator`, `VectorMigrator`, `MessageMigrator`, `IndexVersionMigrator`, `OperationRegistry`), the backup/restore keyspace, a runtime `KvPrefixRegistry`, the AES at-rest codec, the distributed KV lock, `scope_user_mapping`, and data migration of legacy message/document rows.
- Analyzer: provider-specific extraction tuning, Python's proactive-memory stage and the `process/extract` plan/reasoning stages, per-scope model/embedding override resolution, and Python's variable-manager write path (`MemoryType::Variable` is never produced).
- Full Python graph stack: Milvus-backed `GraphMemory`, LLM entity dedupe/merge and relation rewiring, time validity (`valid_since`/`valid_until`), episode/relation retrieval strategies, reranker integration, BFS at the vector-store layer.
- Lite: `sqlite_vec`/embeddings (local search is keyword + FTS5, so `SearchMode::Semantic` is `Unsupported`), per-fact `.md` files and the `USER.md` split, chunked embedding index and `embedding_cache`, `.jsonl` session indexing, interval/session-start sync triggers, `daily_memory` rollups, and `WriteMode`/`MemUpdateChecker` retries.
- Dreaming: the background orchestrator with its idle/backoff loop, the LLM-driven consolidation pipeline, cross-process workers/leases, timer-driven retention, and ML-based salience scoring.
- Team memory: sync/bus lifecycle hooks and `MemberMemoryToolkit` tools, the `TEAM_MEMORY.md` file format, the LLM extractor, non-SQLite durable backends, OTel observability, and the org/tenant layer above `team_id`.
- SQL message tables and pagination cursor parity beyond page index/size.

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
| `manage/update/mem_update_checker.py` (conflict classification) | `analyzer::{ConflictClassifier, RuleClassifier, LlmClassifier}`, applied by `LongTermMemory::consolidate` |
| `process/extract/*` | `analyzer::{RuleExtractor, LlmExtractor, MemoryAnalyzer}` |
| `external/*` | `external::{MemoryProvider, ProviderRegistry, ProviderConfig, EngineProvider}` plus five vendor adapters, each an `Unsupported` stub |
| `migration/*` | `store::migration` (registry/runner) + `store::legacy` (read-only prefix fallback) |
| `dreaming/*` | `dreaming::{DreamJob, DreamRepository, DreamPlanner}` |
| `agent_teams/memory` | `team::{TeamMemoryRecord, TeamMemoryStore, TeamMemory}` |
| `manage/index/variable_manager.py` | `store` variable keys + `facade` variable methods |
| `manage/mem_model/message_manager.py` | `store::messages` + `facade::add_messages`/`get_recent_messages` |
| `graph/graph_memory/base.py::GraphMemory` | `graph::GraphMemory` |
| `graph/extraction/*` | `graph::extract::{HeuristicExtractor, LlmExtractor}` |
| `lite/manager.py::MemoryIndexManager` | `lite::LocalCodingMemory` (Markdown/FTS5/lock/watcher); `lite::CodingMemory` is the engine-backed variant |
| `lite/{frontmatter,internal}.py` fact model / index | `lite::{CodingFact, CodingFactType}` + `render_index` |

## Deviations from `docs/ARCHITECTURE.md`

- `LongTermMemory` is instance-scoped (`Arc`), not a process-global singleton; the Python `register_store`/`set_config` pair becomes `MemoryStores` injection plus `MemoryEngineConfig`.
- The crate does not depend on `openjiuwen-context` or `openjiuwen-session`; prompt injection is expressed as strings/messages from `openjiuwen-foundation`.
- Ranking adds keyword/importance/recency terms beyond the Python LTM vector-only score (documented formula above); this is an additive parity extension, not a wire-format change.
- `MemoryType` adds `coding` (an extension used by `lite`); Python's `unknown` remains the default.
- The external SPI is record-oriented (`add`/`search`/`get`/`update`/`delete`/`list`/`clear`/`health`) where Python's is tool-rail oriented.
- Dreaming is a durable job queue (state machine, attempts, metrics, repositories) rather than Python's idle-scheduler only.
- Team memory is a typed record store plus a `MemoryRecord` projection rather than a `TEAM_MEMORY.md` file.
