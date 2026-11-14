# openjiuwen-memory

## Intent

Long-term, entity/graph, lite coding, analyzer, external-provider, migration, dreaming and team memory for the Rust rewrite of openJiuwen.
Ports Python `openjiuwen.core.memory` v0.1.20 (`long_term_memory.py`, `config/`, `manage/`, `graph/`, `lite/`, `process/extract/`, `external/`, `migration/`, `dreaming/`) plus the `agent_teams.memory` data layer to async Rust.
Scope: scoped record CRUD; hybrid recall with a documented blend; deterministic dedupe/merge; a variable namespace with a record projection; per-scope analyzer overrides; graph memory; engine-backed and local (Markdown + FTS5 + watcher + flock) coding memory; a rule-first analyzer plus a composable staged pipeline; a provider SPI carrying Python's tool rail, one vendor REST transport and five vendor handles; a forward-only SQLite schema runner plus a plan/journal/dry-run migration suite, backup, a prefix registry, a scope↔user mapping and a KV lock seam; a durable dream queue with a caller-driven orchestrator; and team records with a `TEAM_MEMORY.md` document, extractor, hooks, toolkit and observer.
Depends only downward on `openjiuwen-foundation`, `openjiuwen-retrieval` (`Embedding`), `openjiuwen-sysop` (advisory file locks) and `openjiuwen-common`.
Each module `CONTEXT.md` owns its detail; the crate-level deferred roll-up is below.
Memory is a kernel service for `openjiuwen-agent` rails, `openjiuwen-workflow` memory components and `openjiuwen-harness`.

## API Surface

Crate root re-exports the module globs `openjiuwen_memory::{analyzer, config, dreaming, engine, external, facade, graph, lite, prompt, recall, record, store, team}`, plus `CRATE_NAME`, `CRATE_VERSION`, `crate_name()`.
Root-name disambiguation: `graph::{LlmExtractor, EXTRACTION_SYSTEM_PROMPT}`, `lite::local`, `engine::VariableWrite`, `facade::{ScopeAnalyzerOverrideSpec, ScopeOverrideHandles, VariableAssignmentReport}`, `store::{MessageCursor, MessagePage}`, `store::kv_lock::DistributedKvLock`; `store::kv_lock` versus `lite::lock` stays a module-path choice.

```rust
let ltm = LongTermMemory::in_memory(MemoryEngineConfig::default()).await?;
let scope = MemoryScope::new("alice", "workspace");
ltm.add(&scope, NewMemory::semantic("Alice prefers dark mode"), &AddOptions::default()).await?;
let hits = ltm.search(&scope, "theme preference", &SearchOptions::default()).await?;
```

`record`/`config`: `MemoryScope` (`DEFAULT_ID = "__default__"`; `validate` rejects empty ids, NUL, `scope_id` with `/`/`\` or over 128 chars), `MemoryType` (`user_profile`, `semantic_memory`, `episodic_memory`, `variable`, `summary`, `coding`, `unknown`), `MemoryRecord`, `NewMemory`, `MemoryUpdate`, `MemoryFilter`, the option and outcome types; record JSON is Python-compatible (`"text"`/`"type"`/`"fields"`, flattened scope, RFC 3339) and tolerant on decode. `MemoryEngineConfig` holds `default_top_k = 5`, `default_threshold = 0.3`, `token_budget`, `searchable_kinds`, `ranking`, `dedupe`, `extraction`, `graph`, `coding`, `forbidden_variables`, message-history limits and `importance_default = 0.5`, with ranking/dedupe/extraction/graph/coding/agent/scope nested configs.

`store`: `MemoryStores { kv, vector, document, graph }` (`in_memory`, `local_file`, `sqlite`/`sqlite_migrated` under `sqlite`, `from_parts`, `with_graph`); key layout `UMD/{user}/{scope}/{id}`, `user_var/…`, `session_var/…`, `memory_scope_config/{scope_id}`, `memory_scope_overrides/{scope_id}`, `scope_user_mapping/{scope_id}/{user_id}`, vector collections `uid_{user}_gid_{scope}_mtype_{kind}`, messages in `messages`; `user_mem`/`user_variable`/`session_variable`/`scope_config` are read-only legacy fallbacks.
`prefix::KvPrefixRegistry` (+ `KvFamily`, `FamilyPrefixes`) is the injectable registry over five KV families and `messages`; `messages::{StoredMessage, MessageStore, MessageCursor, MessagePage}` provide `append`, `recent`, keyset `page` and `delete_for`.
`migration`: schema runner (`Migration`, `MigrationRegistry`, `builtin_migrations`, `default_registry`; under `sqlite` `MigrationRunner`, `MigrationReport`, `MigrationStatus`, `migrate_sqlite`) and data suite (`Migrator`, `MigrationPlanner`, `MigrationPlan`, `MigrationContext`, `OperationRegistry`, `MigrationJournal`, `run_migrations`, `standard_planner`, plus the `sql`/`message`/`vector`/`index_version` migrators).
`backup::{dump, dump_with, restore, restore_with, MemoryBackup, RestoreReport}`; `mapping::ScopeUserMappingStore`; `kv_lock::{DistributedKvLock, KvLockGuard, LockOptions, InProcessKvLock, FileKvLock, UnsupportedKvLock}`.

`facade::LongTermMemory` (+ `LongTermMemoryBuilder`): `in_memory`, `local_file`, `sqlite`/`sqlite_migrated`, `with_stores`, `builder`, with `engine`/`config`/`graph`/`coding`/`analyzer` accessors.
Records `add`, `add_many`, `get`, `update`, `delete`, `clear`, `list`, `stats`; recall `search`, `recall`, `search_all`; messages `add_messages`, `get_recent_messages`, `clear_messages`, `get_message_page`, `page_messages`.
Variables `set_variable`, `assign_variable`, `get_variables`, `delete_variables`, `variable_records`, `remember_variables`, `remember_with_assignments`; scope config and overrides `set_scope_config`, `get_scope_config`, `delete_scope_config`, `set_scope_overrides`, `get_scope_overrides`, `delete_scope_overrides`, `resolve_scope_overrides` with `ScopeAnalyzerOverrideSpec`, `ScopeOverrideHandles`, `VariableAssignmentReport`.
Integration `remember`, `consolidate`, `provider`, `team_memory`, `team_memory_sqlite`, `dreaming_in_memory`, `dreaming_sqlite`, `open_local_coding`; Python aliases `search_user_mem`, `search_user_history_summary`, `update_mem_by_id`, `delete_mem_by_id`, `delete_mem_by_user_id`, `delete_mem_by_scope`, `user_mem_total_num`, `get_user_mem_by_page`.

`engine`: `MemoryEngine`, `MemoryEngineBuilder`, write/recall orchestration, per-scope write locks, and variables `write_variable -> VariableWrite { record_id, created }`, `delete_variable_everywhere`, `variable_records`. `graph`: `GraphMemory`, `Entity`, `Relation`, `EntityExtractor`, `HeuristicExtractor`, `LlmExtractor`, `GraphRecallHit`.

`lite`: `CodingMemory` (facts as `Coding` records), `CodingFact`, `CodingFactType`, `render_index`, `citation`; `LocalCodingMemory` (Markdown source of truth, derived FTS5, advisory lock, debounced watcher), `MarkdownMemory`/`FrontMatter`, `FtsIndex` (sqlite), `FileLock`, `MemoryWatcher`, `local` constants.
`lite::embed` (chunking, `EmbeddingCache`/`CacheKey`, `EmbeddingIndex`, `content_hash`, `blend`), `lite::facts` (`FactDocument`, `FactFiles`, `IndexLevel { Memory, User }`), `lite::session` (`SessionStore`, `SessionRecord`, `SessionHit`, `sanitize_session_id`), `lite::sync` (`SyncTrigger`, `SyncReport`, `IntervalScheduler`), `lite::daily` (`DailyRollup`, `write_rollup`), `lite::write` (`WriteMode`, `WritePolicy`, `WritePlan`, `WriteOutcome`, `MemUpdateChecker`).

`analyzer`: `FactExtractor` (`RuleExtractor`, `LlmExtractor`), `ConflictClassifier` (`RuleClassifier`, `LlmClassifier`), `MemoryAnalyzer`, `ExtractionResult`, `ConsolidationPlan`, `AnalyzerConfig`, `ExtractorKind`, `ClassifierKind`, `ExtractionInput`, `ConflictVerdict`.
`analyzer::stage`: `StagedAnalyzer`, `ExtractionStage`, `StageInput`, `StageOutput`, `StageKind`, `StageConfig`, `StagedResult` (executed names in the public field `stages`), `RulePlanStage`/`LlmPlanStage`, `ExtractStage`, `RuleReasoningStage`/`LlmReasoningStage`, `RuleProactiveStage`/`LlmProactiveStage`, `ConflictStage`, `ExtractionPlan`, `PlanIntent`, `PlanOperation`, `FactReasoning`, `ReasoningBasis`, `ProactiveConfig`, `ProactiveDecision`, `ScopeAnalyzerOverrides`.

`external`: `MemoryProvider` (`name`, `capabilities`, `initialize`, `add`, `search`, `get`, `update`, `delete`, `list`, `clear`, `health`) with provided rail methods `get_tool_schemas`, `handle_tool_call`, `prefetch`, `sync_turn`, `system_prompt_block`, `on_session_end`, `is_available`, `is_initialized`, `default_scope`; `ProviderCapabilities`, `ProviderHealth`, `external::rail` helpers, `ProviderRegistry`, `ProviderFactory`, `ProviderConfig` (redacting `Debug`), `ProviderKind`, `EngineProvider`, `LocalProviderFactory`.
Transport `rest::{RestProvider, WireSpec, Op, SpecContext, RestCall, AuthScheme, resolve_auth}` and `rest_json`; handles `Mem0MemoryProvider`, `JiuwenMemoryProvider`, `LakeBaseMemoryProvider`, `OpenVikingMemoryProvider`, `AgentArtsMemoryProvider` (+ `vendor_factories()`) — four over HTTP, `agentarts` SDK-only and typed `Unsupported`.

`dreaming`: `DreamJob`, `DreamState`, `DreamKind`, `DreamTrigger`, `DreamMetrics`, `DreamOutcome`, `DreamRunSummary`, `DreamConfig`, `DreamFilter`; `DreamRepository`, `InMemoryDreamRepository`, `SqliteDreamRepository` (sqlite, table `dream_jobs`); `DreamPlanner` (alias `DreamingEngine`), `DreamConsolidator`, `NoopConsolidator`.
`orchestrator::{DreamOrchestrator, DreamOrchestratorConfig, DreamOrchestratorStatus}` (`new`, `tick`, `tick_with`, `run_until_cancelled`, `with_sweep_lock`, `sweep_lock`), `sweep_lock::{DREAM_SWEEP_LOCK_NAME, SweepLockOutcome, SweepLockSkip, acquire_sweep_lock}`, `pipeline::{LlmConsolidator, LlmConsolidatorConfig, DreamCandidateSource, ConsolidationSink, RecordOnlySink, DreamConsolidationPlan}`, `lease::{DreamLease, DreamLeaseOptions, DreamLeaseGuard, LeasedJob, LeaseStatus}`, `retention::{DreamRetention, RetentionReport, purge_expired}`, `salience::{SalienceBackend, SalienceConfig, SalienceScore, salience_score, rank_by_salience}`.

`team`: `TeamMemoryRecord`, `TeamMemoryScope`, `TeamMemoryKind`, `TeamVisibility`, `NewTeamMemory`, `TeamMemoryUpdate`, `TeamMemoryFilter`, `TeamMemoryStats`, `TeamSearchHit`; `TeamMemoryStore`, `TeamSearchOptions`, `InMemoryTeamMemoryStore`, `SqliteTeamMemoryStore` (sqlite, table `team_memories`); `TeamMemory`; projection `to_memory_record`/`from_memory_record`, `TEAM_MEMORY_METADATA_KEY`.
Adapters `document::{TeamMemoryDocument, TEAM_MEMORY_FILE, truncate_for_prompt}`, `extractor::{TeamMemoryExtractor, RuleTeamExtractor, LlmTeamExtractor, TeamTranscript, TeamExtraction}`, `hooks::{TeamMemoryHooks, TeamMemoryHookConfig, TeamMemoryFlush}`, `toolkit::{MemberMemoryToolkit, MemberMemoryContext}` (schemas carry `input_schema`), `observe::{TeamMemoryObserver, NoopObserver, TracingObserver}`.

`prompt`: `render_snippets`, `render_records`, `hits_to_system_message`, `hits_to_messages`, `to_memory_json`, `MEMORY_SECTION_TITLE`, `MEMORY_SECTION_TAG`; depends only on `openjiuwen_foundation::prompt::BaseMessage`, never on `openjiuwen-context`.

## Constraints

- Errors are `openjiuwen_common::error::{Error, Result}`: `Validation`, `NotFound`, `Unsupported`, `Cancelled`; storage/embedding errors keep their kind.
- Async-first on tokio; every public I/O API is `async`; cancellation via `tokio_util::sync::CancellationToken` in `CallOptions`, checked before expensive work and around embedding/model/HTTP calls.
- No blocking I/O on async tasks; a `parking_lot` lock is never held across `.await`; per-scope writes use `tokio::sync::Mutex`.
- Public data types are `serde`-serializable, `snake_case`, tolerant of unknown fields; configs, plans and journals round-trip through JSON.
- No `unsafe` (`#![forbid(unsafe_code)]`); `tracing` only (target `openjiuwen_memory::<module>`); no `println!`/`eprintln!`; public items documented (`missing_docs` via clippy `-D warnings`).
- Files stay around ≤1000 lines and split by responsibility inside the owning module directory.
- Dependencies: `openjiuwen-{common,foundation,retrieval,sysop}`, serde/serde_json, async-trait, tokio/tokio-util, futures, tracing, uuid, chrono, parking_lot, notify, nix, regex; `sqlx` only under `sqlite`; HTTP through `openjiuwen_common::http` (no direct `reqwest`); dev: `openjiuwen-test-support`, `tempfile`, `wiremock`.
- Features: `default = []`; `sqlite` enables `sqlx` and `openjiuwen-foundation/sqlite`. All SQL is behind it (`memory_schema_version`, `dream_jobs`, `team_memories`, fact-keyed `coding_facts_fts`); every non-`sqlite` path must compile and be tested without it.
- Cross-process coordination is advisory file locking only (`#[cfg(unix)]` `flock` via `openjiuwen_sysop::fs::lock`; elsewhere `FileLock`/`FileKvLock` are typed `Unsupported`): no daemon, broker or background process.
- Default features compile and test offline: no network, API key or wall-clock ordering; vendor tests use `wiremock`; `openjiuwen-test-support` is a dev-dependency only.

## Design Decisions

- **Explicit engine instance.** Python's singleton becomes an injected `Arc<LongTermMemory>` owned by `Runner`/harness wiring, so tests and tenants stay isolated.
- **Hybrid recall beyond Python's vector score.** `similarity = (cosine + 1) / 2`; `keyword` is BM25 normalized against the best candidate; `recency = 0.5^(age / 30 days)`; `semantic = 0.70·sim + 0.30·kw`; `final = 0.85·semantic + 0.10·importance + 0.05·recency`. Weights renormalize per group and raw components ride on `RecallHit`.
- **Variables are a namespace with a hidden record projection.** `set_variable`/`assign_variable` write the KV entry and one deterministic `MemoryType::Variable` record (`var-` + 16 hex of an FNV-1a hash of the storage key). Listing, `stats` and `clear` hide projections unless a filter names the kind, recall returns them only on an explicit kind request, and dedupe never sees them. `remember` still stores a rule-extracted assignment as a `SemanticMemory` fact; `remember_with_assignments` is the opt-in route to real writes.
- **Per-scope analyzer overrides are data, resolved per call.** They persist as JSON under `memory_scope_overrides/{scope_id}` and the store never interprets them; an unresolvable named model/embedding (none attached, wrong name, any embedding swap) is a typed `Unsupported` naming the handle.
- **Two message read shapes, one row set.** The newest-N window and keyset pages read the same documents; the cursor is an opaque checksummed `mc1.{millis}.{checksum}.{id}` token over `(timestamp, id)` ascending, and a tampered token is `Validation`.
- **Deterministic dedupe stays the write path.** Duplicates are exact normalized text, cosine `>= 0.92`, or token Jaccard `>= 0.75`, under `Merge` (default), `Skip`, `Replace` or `Add`; a classifier's `ConsolidationPlan` is returned only by `consolidate`.
- **One staged analyzer, composable stages.** `StagedAnalyzer` runs a `Vec<Arc<dyn ExtractionStage>>` over one mutable `StageInput`; the default is plan → extract → reasoning → proactive → conflict, `with_stages` uses the caller's list verbatim (a plan-less sequence keeps the default non-actionable plan), and `StageOutput` merges field-wise.
- **Graph memory is a separate facade.** `search_all` merges record hits with graph hits; nodes are `GraphNode` (`["Entity", <type>]`) and extraction is heuristic by default.
- **Two lite paths, not mirrored.** `CodingMemory` stores facts as ordinary `Coding` records; `LocalCodingMemory` keeps a human-editable `MEMORY.md` as source of truth over a derived, rebuildable FTS5 index, an advisory lock and a debounced watcher, with per-fact files, a chunked embedding index cached by `(model, content hash)`, JSONL sessions, sync triggers, rollups and `WriteMode` as additive caller-driven layers. `open_local_coding` bridges them and copies nothing.
- **Rule-first with model fallback.** All `Rule` kinds work offline; an LLM extractor/classifier/stage retries `model_retries + 1` times, counts unparseable output as a failure, records the first failure as `degraded`, and checks cancellation before its first call. `StageKind::Llm` without a model builds the rule stage and warns.
- **Vendor transports are real HTTP over configurable endpoint and auth.** One `WireSpec` per vendor plus one `RestProvider` own validation, auth, cancellation, retries (idempotent verbs only), endpoint failover, status mapping and error typing. `endpoint` is the switch: configured speaks HTTP, endpoint-less is inert with a typed `Unsupported`; `agentarts` has no documented REST contract, so it stays `Unsupported`.
- **Migrations are a plan, a journal and a dry run; the schema runner stays forward-only.** Each SQLite step commits its DDL with its `memory_schema_version` row in one transaction and treats a newer recorded version as up to date. The data suite plans read-only, journals `Succeeded`/`Failed`/`Skipped` per `namespace.name` operation, resumes from a shared journal, retries with an attempt counter, and invokes no handler in a dry run; `standard_migrators()` plans six operations with default options.
- **Backup normalizes and is additive.** `dump` rewrites alias keys to the current layout so the current value wins; `restore` maps through the target registry, counts equal values `skipped`, deletes nothing, and writes nothing in a dry run. `KvPrefixRegistry` is a passed-in value whose `*_in(prefixes)` builders keep renamed families readable.
- **Dreaming and team memory are caller-driven.** `DreamOrchestrator` ticks on demand or loops in `run_until_cancelled`; a caller-injected `DistributedKvLock` is the only cross-process serialization and an unavailable lock is a reported skip. Leases, retention and purge are explicit seams, team rows live behind `TeamMemoryStore` with Python's runtime pieces as adapters, and neither side starts a timer or daemon.
- **The SQLite dream queue needs an existing parent directory.** `dreaming::SqliteDreamRepository::connect` creates the database file (`create_if_missing`) but not its parent, so a caller passes `sqlite:{dir}/x.db` inside an existing directory (e.g. a `tempfile::TempDir`); `LongTermMemory::dreaming_sqlite` is the variant that runs `create_dir_all` first and returns a `DreamPlanner`, not an orchestrator.
- **Provider SPI and storage seams are injected.** `MemoryProvider`/`ProviderRegistry` are passed in, never global, and `EngineProvider` adapts the local engine so the SPI is exercised by real code. `MemoryStores` injects the foundation traits; listing uses prefix scans, not Python's concatenated id lists. Without an `Embedding`, records are stored and searched by BM25 (`Hybrid` degrades to keyword, `Semantic` is `Unsupported`), and retrieval integration is the `Embedding` trait only.
- **Prompt injection is strings and messages.** `render_snippets` fills a template placeholder (agents may interpolate `{{sys_long_term_memory}}`) and `hits_to_system_message` returns a `BaseMessage`; the crate never manipulates a context window.
- **Time is injectable.** Scoring, dream claiming, retention, team filters and expiry take `now`; tests assert orderings and formulas, never absolute timestamps.

## Error Strategy

- Scope, record, option, name, lock-option, operation-id, prefix and cursor validation reject bad input with `Error::Validation` naming the field and value; `update` of an unknown id is `NotFound`, while `get`/`delete` return `None`/`false`.
- Vector collection absence means "no embedding hits"; a dimension mismatch raises the foundation validation error; a failed embedding rolls back the KV row so no half-indexed record survives.
- Cancellation returns `Error::Cancelled`, leaves committed work intact, makes zero model invocations and is never reported as degraded.
- Typed `Unsupported` marks every absent capability: unconfigured vendor handles, vendor verbs with no documented endpoint (`jiuwen` get/update/delete/list/clear, `lakebase` update, `openviking` delete/update/clear), `agentarts`, `Semantic` or lite semantic search without an embedding index, per-fact APIs with `fact_files(false)`, unresolvable override handles, `SalienceBackend::Ml`, read-only toolkit writes, `UnsupportedKvLock` brokers, non-unix file locks, unreadable vector collections.
- Degraded paths are always explicit — a `tracing::warn!`, a recorded `degraded` reason, or a typed error: keyword fallback, heuristic extraction, LLM-to-rule fallback, unparseable rule regex, malformed rows skipped during listing, lock-unavailable sweep skips.

## Routing Table

- `./src/CONTEXT.md` → source-tree layout, cross-module invariants, dependency order, root re-export rules
- `./src/lib.rs` → crate docs, module declarations, glob re-exports, root disambiguation, lint policy
- `./src/record/` → kinds, scopes, records, inputs, filters, options, results; `./src/config/` → engine config and knobs
- `./src/store/` → store bundles, key layout, persistence, messages, migrations, backup, mapping, KV lock
- `./src/store/migration/` → schema registry/runner, planner/journal/`OperationRegistry`, four migrators, standard entry points
- `./src/store/messages/` → stored messages, `append`/`recent`/`page`, keyset cursors, filter merging
- `./src/recall/` → tokenization, BM25, scoring blend, dedupe/merge, candidate ranking
- `./src/engine/` → write/recall orchestration, per-scope locks, stats, `variables.rs`
- `./src/facade/` → `LongTermMemory` (records, recall, messages and paging, `variables.rs`, `overrides.rs`, integrations)
- `./src/graph/` → entity/relation model, extraction, persistence, neighbor recall
- `./src/analyzer/` → extraction/classification seams, `MemoryAnalyzer`, consolidation plans; `./src/analyzer/stage/` → staged pipeline and overrides
- `./src/external/` → provider SPI, tool rail, engine provider, registry; `./src/external/rest/` → shared vendor transport; `./src/external/vendor/` → five adapter handles
- `./src/dreaming/` → job model, repositories, planner, orchestrator, leases, retention, salience; `./src/dreaming/pipeline/` → `LlmConsolidator`, sources, sinks
- `./src/team/` → team records, stores, facade, projection, and the `document`/`extractor`/`prompts`/`hooks`/`toolkit`/`observe` adapters
- `./src/lite/` → `CodingMemory`, `LocalCodingMemory`, `MEMORY.md` format, FTS5, lock, watcher; `./src/lite/embed/`, `./src/lite/facts/`, `./src/lite/session/` → embedding index, per-fact files, JSONL sessions
- `./src/prompt/` → memory snippets and `BaseMessage` construction
- `./tests/` → the 22 cross-module integration binaries (`tests/CONTEXT.md`)

## Test Strategy

- Unit tests live beside each module (`#[cfg(test)] mod tests`, SQLite cases additionally gated on `feature = "sqlite"`): JSON compatibility, config validation, key/prefix mapping, scoring and BM25, dedupe/merge, rule and LLM fallback, staged-pipeline parity, Markdown/FTS5/lock/watcher, embedding-cache hits, session/rollup/sync/write modes, migration planning/journal/rollback, backup, lock contention, dream claim/lease/retention/salience, team visibility/hooks/toolkit/observer, provider rail and per-vendor wire shape.
- Module docs carry the per-suite counts (for example `external` 160 unit tests, `team` 185 without `sqlite` and 225 with); the crate-wide totals they record disagree with each other and are not restated here.
Integration binaries in `tests/`, one line each:

| Binary | Scope |
|---|---|
| `crud.rs`, `scoping.rs` | facade CRUD/stats/validation/JSON shape; user/scope/session/agent isolation, variables, scope configs, parity aliases |
| `recall.rs`, `dedupe.rs` | semantic/keyword/hybrid, thresholds, budgets, `Unsupported`; every dedupe policy, metadata union, batch |
| `graph.rs`, `graph_memory.rs` | graph memory via the facade / `GraphMemory` directly |
| `lite.rs`, `lite_local.rs`, `lite_embedding.rs` | engine-backed `CodingMemory`; Markdown + FTS5 + lock + watcher; semantic ranking, cache hits, per-fact files, sessions, rollups, write modes |
| `sqlite.rs`, `store_backends.rs`, `facade_smoke.rs` | SQLite persistence across reopen; public `MemoryStores` round trip; constructors, CRUD, recall, variables, `add_messages` |
| `analyzer_wiring.rs`, `analyzer_stages.rs` | `remember`/`consolidate` and degradation / stage order, gating, overrides |
| `external_provider.rs`, `external_rest.rs` | provider view, redaction, registry / wiremock wire shape, retry, failover |
| `migration.rs`, `migration_advanced.rs` | schema versions, `sqlite_migrated` / plan/run, journal resume, dry run, backup |
| `team_memory.rs`, `team_toolkit.rs` | team CRUD/ranking/reopen / hooks, toolkit, `TEAM_MEMORY.md`, observer |
| `dreaming.rs`, `dreaming_orchestrator.rs` | queue transitions and durability / tick, leases, retention, salience |

- Deterministic offline fakes: `HashEmbedding`, `ScriptedModel`, `wiremock::MockServer`, `TeamMemory::in_memory`, `InMemoryDreamRepository`, `tempfile::tempdir`.
- Gates: `cargo fmt -p openjiuwen-memory -- --check`, `cargo test -p openjiuwen-memory` (also `--features sqlite` and `--all-features`), `cargo clippy -p openjiuwen-memory --all-targets --all-features -- -D warnings`, `cargo doc -p openjiuwen-memory --no-deps` warning-free.

## Deferred (Python `core.memory` features not ported)

- External: `agentarts`' real SDK transport, session-mapping KV store and `MemorySearchFilter`; mem0's `rerank`, `queue_prefetch` and 5-failure/120 s circuit breaker; jiuwen's in-process SDK backend (`mode="sdk"`) and circuit breaker; lakebase's branch/version endpoints, `digest`, `traits`, `switch_base`. No workspace code drives `prefetch`/`sync_turn`/`on_session_end` yet; the SPI is record-only; provider pooling and live-API tests are absent (`wiremock` only).
- Migrations: Python's Alembic column operations and dialect handling; `run_on_stores(&MemoryStores)` (needs a foundation downcast hook); per-collection version stamps with background rebuild; vector/graph/document and incremental backup; an AES at-rest codec; multi-node brokers (Redis/etcd/ZooKeeper behind `UnsupportedKvLock`); `FileKvLock` has no TTL eviction or fencing token.
- Analyzer: provider-specific extraction tuning and few-shot exemplars; `resolve_references`/`normalize_time` applied to fact text (they are plan metadata); a second prompt pass that re-runs extraction; an LLM redundancy checker.
- Graph: the Milvus-backed `GraphMemory`, LLM entity dedupe/merge and relation rewiring, time validity, episode/relation retrieval strategies, reranker integration, BFS at the vector-store layer.
- Lite: `sqlite_vec`/native ANN and Python's `files`/`chunks`/`chunks_vec`/`meta` schema (recall is brute-force cosine); a provider fingerprint in the cache key (`Embedding` exposes only `model_name`, so the key is `(model, content hash)`); the LLM redundancy checker (`MemUpdateChecker` is deterministic); chunk-level trigram FTS and FTS over session/rollup text; harness-written transcripts, watcher-driven session sync, session recency boosts, durable session vectors; an automatic `OnSessionStart` hook and `sync.intervalMinutes`; prompt-rail `daily_memory` injection; non-unix lock parity.
- Dreaming: an ML salience backend (`SalienceBackend::Ml` is `Unsupported`: no inference runtime or training data); a `LongTermMemory`-backed plan sink (`RecordOnlySink` mutates nothing); background lease heartbeats, fencing tokens and distributed lease brokers; timer-driven retention (every purge is caller-driven); Python's idle-layer extras and a graph-extraction consolidator.
- Team: Python's `register_tools`/`strip_rails_by_type`/`load_and_inject` and the per-member lite `MemoryIndexManager` (the `openjiuwen-agent-teams` runtime owns tool registration and prompt injection); the DeepAgent extractor that rewrites `TEAM_MEMORY.md` across turns; the `MEMORY_ENABLED` switch and coding-memory tools; cross-process/distributed stores, a cross-process bus, retention daemon, monitor/UI; the OTel exporter (in `openjiuwen-extensions`), an org/tenant layer above `team_id`, and embedding-based team recall (no embedding column).
- Variables and history: Python's variable-manager extras beyond the KV entry plus projection, and SQL message-table/pagination parity beyond page index/size and keyset cursors.

## Python Mapping

| Python | Rust |
|---|---|
| `long_term_memory.py::LongTermMemory` | `facade::LongTermMemory` |
| `manage/mem_model/memory_unit.py::{MemoryType, FragmentMemoryUnit, VariableUnit, SummaryUnit}`, `core/foundation/store/base_memory_index.py::MemoryDoc` | `record::{MemoryType, MemoryRecord, NewMemory}` (JSON `text`/`type`/`fields`) |
| `manage/{index/write_manager,search/search_manager}.py` | `engine::MemoryEngine` write and recall paths + `recall` scoring |
| `manage/update/mem_update_checker.py::MemUpdateChecker` | `recall::dedupe` policies; conflict plans via `analyzer::ConflictClassifier` + `facade::consolidate` |
| `manage/index/variable_manager.py` | `engine::{write_variable, delete_variable_everywhere, variable_records}` + `store` variable keys + `facade` variable methods |
| `manage/mem_model/message_manager.py` | `store::messages` (`append`/`recent`/`page`/`delete_for`) + `facade::{add_messages, get_recent_messages, get_message_page}` |
| `process/extract/*` | `analyzer::{RuleExtractor, LlmExtractor, MemoryAnalyzer}` + `analyzer::stage` |
| `external/*` | `external::{MemoryProvider, rail, ProviderRegistry, EngineProvider}`, `external::rest`, five vendor handles |
| `migration/*` | `store::migration` + `store::{prefix, backup, mapping, legacy}` |
| `dreaming/*` | `dreaming::{DreamJob, DreamRepository, DreamPlanner, DreamOrchestrator, LlmConsolidator, lease, retention, salience}` |
| `agent_teams/memory` | `team` (records, stores, `TeamMemory`, projection, `document`, `extractor`, `hooks`, `toolkit`, `observe`) |
| `graph/{graph_memory/base,extraction}/*` | `graph::GraphMemory` + `graph::extract::{HeuristicExtractor, LlmExtractor}` |
| `lite/manager.py::MemoryIndexManager` and `lite/{frontmatter,internal}.py` | `lite::LocalCodingMemory` (`embed`, `facts`, `session`, `sync`, `daily`, `write`); `lite::CodingMemory` is the engine-backed variant over `lite::{CodingFact, CodingFactType}` + `render_index` |

## Deviations from `docs/ARCHITECTURE.md`

- `LongTermMemory` is instance-scoped (`Arc`) with `MemoryStores` injection instead of Python's singleton, and the KV prefix registry is a value rather than a process global.
- The crate does not depend on `openjiuwen-context` or `openjiuwen-session`; prompt injection is strings and `BaseMessage`s.
- Ranking adds keyword/importance/recency terms beyond Python's vector-only LTM score (additive, no wire-format change), and `MemoryType` adds `coding` for `lite`.
- Variables extend Python's variable manager: the KV string keeps its key layout and one deterministic `MemoryType::Variable` record projection is added so a variable can be recalled and filtered.
- Message paging uses checksummed keyset tokens over `(timestamp, id)` where Python uses offset/page-index cursors, so concurrent appends cannot skip or repeat a row.
- The provider SPI is record-oriented with Python's tool rail as provided methods, and one shared HTTP transport replaces per-vendor clients; `agentarts` stays SDK-only and typed `Unsupported`.
- Migrations are a plan/journal/dry-run suite over a forward-only SQLite schema runner instead of Python's per-backend migrators; legacy keys are read-only compatibility, never rewritten.
- Dreaming is a durable job queue with caller-driven orchestration, lease and retention seams rather than Python's idle scheduler; ML salience is typed `Unsupported`.
- Team memory is a typed record store with a `MemoryRecord` projection and a lossless `TEAM_MEMORY.md` document view rather than Python's summary-file-only model.
- `store::kv_lock` carries the lock seam because `lite::lock` already owns `lock` inside `lite`; both stay module paths, never root re-exports.
