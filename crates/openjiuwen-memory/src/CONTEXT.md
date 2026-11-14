# openjiuwen-memory/src — crate source layout

## Intent

The source tree of `openjiuwen-memory`, the crate that ports Python `openjiuwen.core.memory` (v0.1.20) plus the `agent_teams.memory` data layer.
Every module directory owns its own `CONTEXT.md`; this file is the routing table for the tree and the home of the cross-module wiring rules.
`src/lib.rs` is the crate root: module declarations, glob re-exports, root-name disambiguation, crate docs, and the lint policy.

## API Surface

- `openjiuwen_memory::{analyzer, config, dreaming, engine, external, facade, graph, lite, prompt, recall, record, store, team}` — one public module per subsystem.
- Crate root re-exports every module glob, plus explicit bindings for names two modules would otherwise both bind: `graph::{LlmExtractor, EXTRACTION_SYSTEM_PROMPT}` (analyzer has its own `LlmExtractor`; team's prompt constant is reachable as `team::prompts::EXTRACTION_SYSTEM_PROMPT`), `lite::local` (external has a `local` module too), `engine::VariableWrite`, `facade::{ScopeAnalyzerOverrideSpec, ScopeOverrideHandles, VariableAssignmentReport}`, `store::{MessageCursor, MessagePage}`, and `store::kv_lock::DistributedKvLock`.
- `CRATE_NAME`, `CRATE_VERSION`, `crate_name()`.
- New names must not be added to `src/lib.rs` without checking they do not ambiguate an existing root binding (`store::kv_lock` versus `lite::lock` is the documented near-collision; both stay module paths).

## Cross-module invariants

- Variables are their own namespace: `facade::LongTermMemory::set_variable` writes the KV entry and delegates to `engine::MemoryEngine::write_variable`, which maintains one deterministic `MemoryType::Variable` record per `(user, scope, session, name)`.
- Record listing, statistics and `clear` exclude `Variable` projections unless a `MemoryFilter` names the kind explicitly; recall returns them only for an explicit kind request.
- `facade::LongTermMemory::remember` keeps storing a rule-extracted assignment as a `SemanticMemory` record; `remember_with_assignments` (and the model payload's `variables` map in `add_messages`) is the opt-in route that writes real `MemoryType::Variable` projections.
- Per-scope analyzer overrides are persisted as `memory_scope_overrides/{scope_id}` (`store::keys::scope_overrides_key`), resolved by `facade::overrides`, and consumed by `remember`/`consolidate` through `analyzer::stage::StagedAnalyzer`.
- Message history is read two ways over the same document rows: the legacy newest-N window and keyset pages resumed from an opaque `store::MessageCursor` (`(timestamp, id)` ascending, tampered token → `Validation`).
- Dreaming stays caller-driven: `dreaming::DreamOrchestrator` sweeps on `tick`, and a caller-injected `store::kv_lock::DistributedKvLock` (`DreamOrchestrator::with_sweep_lock`) is the only cross-process serialization.
- Storage seams are injected (`store::MemoryStores`); no module reads process-global state, and the crate starts no background task except the explicitly constructed local-coding watcher and the caller-driven dream loop (`run_until_cancelled`).
- SQL lives only behind the `sqlite` feature; every non-`sqlite` path must compile and be tested without it.

## Constraints

- Modules depend downward: `record`/`config`/`store`/`recall` at the bottom, `engine` above them, `facade` above `engine`, and `analyzer`/`graph`/`lite`/`dreaming`/`team`/`external` as subsystem trees used by the facade.
- `openjiuwen-context` and `openjiuwen-session` are deliberately not dependencies; prompt injection is expressed as strings and `openjiuwen_foundation::prompt::BaseMessage`s.
- Files stay around ≤1000 lines and split by responsibility inside the owning module directory.
- Errors are `openjiuwen_common::error::{Error, Result}`; every public I/O API is `async`; `tracing` only, no `println!`; no `unsafe`; no blocking I/O on async tasks.
- Unit tests live beside each module under `#[cfg(test)] mod tests`; SQLite-only tests are additionally gated on `feature = "sqlite"`.

## Routing Table

- `./analyzer/` → fact extraction, conflict classification, consolidation plans, and the staged pipeline (`analyzer/stage/`)
- `./config/` → `MemoryEngineConfig` and the ranking/dedupe/extraction/graph/coding/storage knobs
- `./dreaming/` → dream-job model, repositories, planner, orchestrator (with the sweep lock), LLM pipeline, leases, retention, salience
- `./engine/` → write/recall orchestration, per-scope locks, statistics, and the variable-manager projection
- `./external/` → memory-provider SPI, engine provider, registry, REST transports, vendor handles, tool rail
- `./facade/` → `LongTermMemory` entry point: records, recall, messages/paging, variables, scope config and analyzer overrides, integrations
- `./graph/` → entity/relation model, extraction, graph persistence, neighbor recall
- `./lite/` → coding facts, Markdown index, FTS5, file lock, watcher, embedding index, per-fact files, session/day views, sync, write modes
- `./prompt/` → recall rendering and `BaseMessage` construction for prompt injection
- `./recall/` → tokenization, BM25, scoring blend, dedupe, candidate ranking
- `./record/` → kinds, scopes, records, inputs, filters, options, results
- `./store/` → store bundles, key layout, prefix registry, persistence, message history, migrations, backup, mapping, distributed KV lock
- `./team/` → team-scoped records and stores, hooks, toolkit, document rendering, extractor, observability, projection
- `./lib.rs` → crate docs, module declarations, glob re-exports, root disambiguation, lint policy
