# openjiuwen-core

## Intent

Kernel umbrella crate that mirrors the Python `openjiuwen.core` package tree with Rust module paths.
It is a pure re-export facade: it implements no behavior and owns no data types.
The concrete kernel crates remain the canonical home for every item's implementation, docs, and tests.
Consumers use this crate when they want Python-shaped paths; workspace-internal code should depend on the concrete crate instead so the dependency graph stays acyclic.

## Module Mapping

| Python path | Rust path | Source crate |
| --- | --- | --- |
| `openjiuwen.core.common` | `openjiuwen_core::common` | `openjiuwen-common` |
| `openjiuwen.core.foundation` | `openjiuwen_core::foundation` | `openjiuwen-foundation` |
| `openjiuwen.core.session` | `openjiuwen_core::session` | `openjiuwen-session` |
| `openjiuwen.core.graph` | `openjiuwen_core::graph` | `openjiuwen-graph` |
| `openjiuwen.core.workflow` | `openjiuwen_core::workflow` | `openjiuwen-workflow` |
| `openjiuwen.core.context_engine` | `openjiuwen_core::context_engine` | `openjiuwen-context` |
| `openjiuwen.core.memory` | `openjiuwen_core::memory` | `openjiuwen-memory` |
| `openjiuwen.core.retrieval` | `openjiuwen_core::retrieval` | `openjiuwen-retrieval` |
| `openjiuwen.core.sys_operation` | `openjiuwen_core::sys_operation` | `openjiuwen-sysop` |
| `openjiuwen.core.security` | `openjiuwen_core::security` | `openjiuwen-security` |
| `openjiuwen.core.single_agent` | `openjiuwen_core::single_agent` | `openjiuwen-agent` (glob re-export) |
| `openjiuwen.core.application` | `openjiuwen_core::application` | `openjiuwen-agent` (application-agent surface only) |
| `openjiuwen.core.controller` | `openjiuwen_core::controller` | `openjiuwen-agent` (`controller` module glob) |
| `openjiuwen.core.kv_cache` | `openjiuwen_core::kv_cache` | `openjiuwen-agent` (`kv_cache` module glob) |
| `openjiuwen.core.operator` | `openjiuwen_core::operator` | `openjiuwen-agent` (`operator` module glob) |
| `openjiuwen.core.multi_agent` | `openjiuwen_core::multi_agent` | `openjiuwen-multiagent` |
| `openjiuwen.core.runner` | `openjiuwen_core::runner` | `openjiuwen-runner` |

The `application` module re-exports `LLMAgent` (from `LlmAgent`), `LLMAgentConfig`, `WorkflowAgent`,
`WorkflowAgentConfig`, `DefaultResponse`, `ConstrainConfig`, `ModelConfig`, `ModelInfo`, and `create_llm_agent_config`;
it deliberately does not glob the whole agent crate.
`single_agent` is a blanket glob of `openjiuwen-agent`, while `controller`, `kv_cache`, and `operator` are narrower
per-module globs mirroring Python's top-level packages; nested paths such as `controller::schema`, `kv_cache::runtime`,
and `operator::base` stay reachable through them, and `single_agent` also exposes the same items as an alternative spelling.
Because the other aliases are whole-crate re-exports, every nested public module of those kernel crates is reachable under its Python-shaped path without being re-listed: notable trees are
`single_agent::{memory_provider, usage, prompt_pack, abilities, policy, rails}`,
`context_engine::{usage, telemetry, policy, tokenizers, processor::rule_compression, recall, reload, offload, window}`,
`memory::{lite::{facts, embed, session, local, markdown, fts}, dreaming, team, external, store::migration, graph, analyzer}`,
`retrieval::{loaders, parsing, chunking, embedding, vector_store, retriever, reranker, rewrite}`,
`workflow::{single, components, engine, render, registry, interrupt}`, and `runner::{pool, resources, callback, hitl, monitor, spawn}`.

## Features

`default = []`; every other feature is additive and opt-in, and each one forwards to the concrete crate that owns the gated surface.

| Feature | Forwards to | Effect |
| --- | --- | --- |
| `otel` | `openjiuwen-session/otel` | `session::install_trace_sink` really installs the sink instead of being the documented no-op |
| `sqlite` | `openjiuwen-foundation/sqlite`, `openjiuwen-session/sqlite`, `openjiuwen-memory/sqlite` | SQLite KV/document stores (`foundation::store::{SqliteKvStore, SqliteDocumentStore}`), SQLite checkpointer (`session::SqliteCheckpointer`), FTS5 coding index (`memory::lite::fts`) and the migration runner (`memory::store::migration::runner`) |
| `sandbox` | `openjiuwen-sysop/sandbox` | `sys_operation::sandbox::http` compiles; its remote `invoke` still returns `Error::Unsupported` |
| `live-tests` | `openjiuwen-foundation/live-tests` | gates the concrete crate's opt-in network tests |

Those four forwarders cover the entire gated kernel surface, so no further forwarder is missing.

- `openjiuwen-foundation` declares `default = ["mcp", "tiktoken"]`, `mcp`, `tiktoken`, `sqlite`, `live-tests`.
- `openjiuwen-session` declares `default = []`, `sqlite`, `otel`; `openjiuwen-memory` declares `default = []`, `sqlite` (which itself enables `openjiuwen-foundation/sqlite`); `openjiuwen-sysop` declares `default = []`, `sandbox`.
- `openjiuwen-common`, `openjiuwen-graph`, `openjiuwen-workflow`, `openjiuwen-context`, `openjiuwen-retrieval`, `openjiuwen-security`, `openjiuwen-agent`, `openjiuwen-multiagent` and `openjiuwen-runner` declare `default = []` and no other feature, and have no optional dependency — notably `openjiuwen-runner`, which contributes nothing to forward.

Deliberately not forwarded:

- `mcp` / `tiktoken` — `openjiuwen-foundation`'s defaults, and the dependency is taken with default features; Cargo features are additive, so an umbrella feature could never disable them.
- `redis` — no kernel crate declares such a feature: `session::RedisCheckpointer` and foundation's `store::external::redis` always compile, and live Redis wiring is `openjiuwen-extensions`'s `redis` feature (a different layer).
- `fts5` — memory's FTS5 index is gated by memory's `sqlite` feature, so it is covered by the umbrella `sqlite` feature.
- `distributed` / `vendor` / `notify` / `browser` / `cua` / `mobile` — teams-, harness-, or extension-layer features, not kernel ones.

## Constraints

- Re-exports only: no logic, no new public data types, and no `unsafe` (`#![forbid(unsafe_code)]`).
- `publish = false` until the SDK API stabilizes.
- Concrete crates must not depend on this umbrella, except the product/harness layers and above; the crate stays acyclic.
- Item identity is guaranteed: each alias is the same type as the concrete crate's item, verified by `TypeId` in tests.
- Naming follows the Python tree even where it diverges from the crate name (`context_engine` → `openjiuwen-context`, `sys_operation` → `openjiuwen-sysop`, `multi_agent` → `openjiuwen-multiagent`).
- Every dependency is taken with default features only: the crate declares `default = []` plus the opt-in forwarders `otel`, `sqlite`, `sandbox`, and `live-tests` (see Features), never `default-features = false`, so foundation's `mcp`/`tiktoken` defaults always apply.

## Routing Table

- `./src/lib.rs` → module aliases, the `single_agent`, `application`, `controller`, `kv_cache`, and `operator` shims, the Python→Rust path table and the feature-forwarding section in the crate docs.
- `./tests/module_paths.rs` → resolve/identity checks for one representative item per module, plus per-area extended tests for the kernel's wider surface.
- `./tests/feature_forwarding.rs` → per-feature reachability and behaviour of the `otel` / `sqlite` / `sandbox` / `live-tests` forwarders.

## Test Strategy

- `tests/module_paths.rs` proves every module path resolves and aliases the same type as the source crate via `TypeId`;
  `single_agent`, `application`, and `runner` (its `Runner`, `RunOptions`, `ResourceMgr`, `CallbackEvent`, and `InMemoryMessageQueue` exports, plus a `MessageQueue` trait-object coercion) are covered explicitly.
- Per-area identity tests cover the extended kernel surface: `controller_*`, `kv_cache_*`, `operator_*`, `single_agent_extended_*`, `session_extended_*`, `runner_extended_*` (including nested module paths), `foundation_extended_*`, `memory_extended_*`, and `retrieval_extended_*`.
- `single_agent_modules_paths_resolve`, `context_engine_modules_paths_resolve`, `memory_modules_paths_resolve`, `retrieval_modules_paths_resolve`, and `workflow_single_component_paths_resolve` cover the deep trees the concrete crates expose: agent `memory_provider` / `usage` / `prompt_pack` (plus `prompt_pack::{directory, pack}`), context `usage` / `telemetry` / `policy` / `tokenizers` / `processor::rule_compression`, memory `lite::{facts, embed, session, local}` / `dreaming` / `team` / `external` / `store::migration`, retrieval `loaders::{pptx, odt, fixtures}` / `parsing::{zip, pptx, odt}`, and workflow `single` plus the questioner and HTTP-request configuration.
  Items gated on a feature are excluded there: the memory `store::migration::runner` re-export is `sqlite`-gated and is only reachable under `--all-features`.
  `workflow_components_paths_resolve` covers the twelve components plus the `components::sub_workflow` / `questioner` / `http_request` nested paths.
  `memory_extended_paths_resolve` covers `lite::{CodingFact, CodingMemory}` plus `lite::markdown` and `lite::local::LocalCodingMemory`;
  `retrieval_extended_paths_resolve` covers `sparse::bm25` on top of `sparse`.
  Traits (`TaskExecutor`, `QueryRewriter`, `CompressionStateSink`, `TeamMemoryStore`, `external::MemoryProvider`) are proved by generic bounds on the concrete trait object; function items (`abilities::parts::extract_parts`, `install_trace_sink`, `diff_snapshots`, `provider_tools`, `default_usage_analyzer`, `lookup_model_default_window`, `extract_query_terms`, `load_native_tokenizer`, `slug`, `cosine_similarity`, `sanitize_session_id`, `standard_registry`, the `loaders::fixtures` builders) by fn-pointer identity, and `load_mcp_abilities` / `execute_single_component` (async) by reference only.
  `load_native_tokenizer` takes `impl AsRef<Path>`, so its identity is pinned at the `PathBuf` instantiation; `ContentRouter` is a type alias for `RuleContentRouter`, not a trait.
- `tests/feature_forwarding.rs` proves each forwarder: `sqlite` reaches the SQLite stores/checkpointer/FTS5 index, `sandbox` reaches `HttpSandboxGateway` and keeps its `Error::Unsupported` remote contract, and `otel` really installs a `MemoryTraceSink` on a `WorkflowSession` (its `#[cfg(not(feature = "otel"))]` counterpart proves the documented no-op).
  Its always-on tests plus the `#[cfg(not(feature = "sqlite"))]` / `#[cfg(not(feature = "sandbox"))]` tests prove the reduced configuration still builds the always-compiled card surfaces (`StoreRegistry`, `SandboxGatewayConfig`).
- Gate commands: `cargo test -p openjiuwen-core --offline` and `--no-default-features` exercise the no-forwarder configuration; `--all-features` exercises `otel` + `sqlite` + `sandbox` + `live-tests`.
- The crate-level doc example compiles as a doctest (`cargo test -p openjiuwen-core`), guarding the public paths.
- No unit tests here: the umbrella has no behavior; behavior is tested in the concrete crates.
- `cargo clippy -p openjiuwen-core --all-targets --all-features -- -D warnings` must stay clean.
