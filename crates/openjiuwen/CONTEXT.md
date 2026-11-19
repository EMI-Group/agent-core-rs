# openjiuwen

## Intent

Top-level umbrella crate: the Rust counterpart of the Python package root `openjiuwen`.
It is a pure facade — module aliases onto concrete crates plus one curated `prelude` — and owns no logic or data types of its own.
Consumers use it for Python-shaped paths (`openjiuwen::harness::DeepAgent`); workspace-internal code depends on the concrete crate instead, which keeps the dependency graph acyclic.
The concrete crates stay canonical for documentation, semver, behavior and tests; this crate only re-exports.

## API Surface

### Module aliases

| Python | Rust | Source crate |
| --- | --- | --- |
| `openjiuwen.core` | `core` | `openjiuwen-core` (kernel umbrella, including its `single_agent` / `application` shims) |
| `openjiuwen.harness` | `harness` | `openjiuwen-harness` |
| `openjiuwen.harness_protocol` | `harness_protocol` | `openjiuwen-harness-protocol` |
| `openjiuwen.harness_providers` | `harness_providers` | `openjiuwen-harness-providers` |
| `openjiuwen.agent_teams` | `agent_teams` | `openjiuwen-agent-teams` |
| `openjiuwen.extensions` | `extensions` | `openjiuwen-extensions` |
| `openjiuwen.dev_tools` | `dev_tools` | `openjiuwen-dev-tools` |
| `openjiuwen.symphony` | `symphony` | `openjiuwen-symphony` |
| `openjiuwen.rsi` | `rsi` | `openjiuwen-rsi` |
| `openjiuwen.agent_evolving` | `agent_evolving` | `openjiuwen-agent-evolving` |
| `openjiuwen.auto_harness` | `auto_harness` | `openjiuwen-auto-harness` (shim over `rsi::auto_harness`) |

Every alias names the same item as the concrete crate; `tests/module_paths.rs` proves identity via `TypeId` and function-pointer equality.
`core` is a whole-crate alias, so the kernel umbrella's `controller`, `kv_cache` and `operator` packages are reachable as `openjiuwen::core::<module>` alongside every other kernel module.
`crate_name()` returns `"openjiuwen"` and is `#[must_use] pub const fn`.
`VERSION: &str = env!("CARGO_PKG_VERSION")` is the mirror of Python's `openjiuwen.__version__`; Python's top-level `__init__.py` exports nothing else, so `VERSION` plus the aliases is the complete top-level surface.

### prelude

`openjiuwen::prelude` re-exports the most-used items for `use openjiuwen::prelude::*;`.
It is convenience only: no new types, no behavior, and no stability promise beyond the items it forwards.
Beyond the `a2a` and `otel` items it forwards, the prelude is feature-free: Python's top-level `__init__.py` exports only `__version__` (mirrored by `VERSION`), so feature-gated trees are reached by module path.

| Item | Canonical path |
| --- | --- |
| `Error`, `Result` | `core::common::error` |
| `Agent`, `AgentCard` | `core::single_agent` |
| `ReActAgent`, `ReActAgentConfig` | `core::single_agent` (`openjiuwen_agent::react`) |
| `AgentInput`, `AgentOutput`, `AgentChunk`, `ResultType` | `core::single_agent` (`openjiuwen_agent::{input, output}`) |
| `Observation` | `core::single_agent` (`openjiuwen_agent::output`) |
| `LlmAgent`, `WorkflowAgent` | `core::single_agent` (`openjiuwen_agent::{llm_agent, workflow_agent}`) |
| `ControllerAgent` | `core::controller` (`openjiuwen_agent::controller`) |
| `Session`, `AgentSession`, `SessionController` | `core::session` (`SessionController` from `openjiuwen_session::session_controller`) |
| `Model`, `BaseMessage` | `core::foundation` |
| `Tool`, `FunctionTool` | `core::foundation` |
| `CompiledGraph` | `core::graph` |
| `Workflow` | `core::workflow` |
| `ContextEngine` | `core::context_engine` |
| `LongTermMemory` | `core::memory` |
| `Runner`, `TeamPool`, `RunMonitor` | `core::runner` |
| `DeepAgent`, `DeepAgentConfig`, `DeepAgentBuilder`, `create_deep_agent` | `harness` |
| `HarnessCard`, `HarnessProtocol` | `harness_protocol` |
| `create_harness`, `HarnessIOAdapter` | `harness_providers` |
| `TeamAgent`, `TeamAgentSpec` | `agent_teams` |
| `A2aClient` (feature `a2a`) | `extensions::a2a::client` |
| `OtelTracerConfig` (feature `otel`) | `extensions::otel` |

`AgentSession` is the in-memory session implementation (`openjiuwen-session`); there is no separate `InMemorySession` type in the Rust API.
The prelude is capped at ~40 items so `use openjiuwen::prelude::*;` stays unpolluted: agent-construction types and entry points only, no module globs, and the only config structs it forwards are `ReActAgentConfig`, `DeepAgentConfig` and `TeamAgentSpec`.

### Feature matrix

Dependencies are non-optional and taken with their own default features, so every forwarded feature is additive: `openjiuwen-harness`'s `native` and `openjiuwen-harness-providers`'s `native` always apply, and `openjiuwen-foundation`'s `mcp`/`tiktoken` defaults cannot be switched off from here.
Every feature that is off by default keeps the default build free of network access and external services.

Harness and providers:

| Feature | Default | Forwards to | Effect |
| --- | --- | --- | --- |
| `native` | yes | `openjiuwen-harness/native`, `openjiuwen-harness-providers/native` | the in-process `native` provider (the half that really gates source); `openjiuwen-harness/native` is an empty marker — no source reads it and the native `DeepAgent` harness is always compiled |
| `native-v2` | no | `openjiuwen-harness-providers/native-v2` | the agent-teams-hosted `native_v2` provider at `harness_providers::providers::native_v2` |
| `vendor` | no | `openjiuwen-harness-providers/vendor` | registers the vendor providers `claudecode`, `codex`, `dsh`, each still answering `Unsupported` when driven |
| `browser` | no | `openjiuwen-harness/browser` | the `browser` subagent preset, which reports its missing implementation as `Unsupported` |
| `cua` | no | `openjiuwen-harness/cua` | the `cua` subagent preset, which reports its missing implementation as `Unsupported` |
| `mobile` | no | `openjiuwen-harness/mobile` | the `mobile` subagent preset, which reports its missing implementation as `Unsupported` |
| `notify` | no | `openjiuwen-harness/notify` | compiles the harness's `notify`-backed watcher backend, so `harness::watcher::WorkspaceWatcher::start()` really arms an OS watcher; without the feature the watcher API still resolves and `start()` returns `Unsupported` (the `openjiuwen` binary enables the harness feature on its own, so its `watch` and `chat --watch` commands never depend on this forward) |

Storage:

| Feature | Default | Forwards to | Effect |
| --- | --- | --- | --- |
| `sqlite` | no | `openjiuwen-foundation/sqlite`, `openjiuwen-session/sqlite`, `openjiuwen-memory/sqlite`, `openjiuwen-extensions/sqlite`, `openjiuwen-symphony/sqlite`, `openjiuwen-agent-teams/sqlite` | SQLite checkpointer, key-value/document stores, memory stores, capability-asset store, and team store |

Transports and remote backends:

| Feature | Default | Forwards to | Effect |
| --- | --- | --- | --- |
| `a2a` | no | `openjiuwen-extensions/a2a` | Agent-to-Agent client/server modules |
| `mcp` | no | `openjiuwen-extensions/mcp` | compiles that crate's `mcp` module — `extensions::mcp::{McpToolGateway, McpToolEntry, PermissionFilter}`, the harness `ToolGateway` bridge over kernel MCP tools — and its `openjiuwen-foundation/mcp` forward; agent-teams' MCP server is compiled regardless because that crate's `mcp` default applies |
| `redis` | no | `openjiuwen-extensions/redis`, `openjiuwen-agent-evolving/redis` | the extensions' Redis KV/document/vector stores and `RedisCheckpointer` over its in-crate RESP client (`redis://` connects lazily; `rediss://` TLS and cluster slot routing stay `Unsupported`), and the RL online service's Redis-backed `OnlineStore` (`agent_evolving::online::{RedisOnlineStore, RedisOnlineStoreOptions}`, likewise `Unsupported` for TLS and cluster) |
| `pulsar` | no | `openjiuwen-extensions/pulsar` | the extensions' Pulsar message-queue module, whose construction always returns `Unsupported` (no broker client is vendored) |
| `distributed` | no | `openjiuwen-agent-teams/distributed` | the distributed `PyZmqMessager` seam, which rejects at construction |

Observability, sandbox, test gates:

| Feature | Default | Forwards to | Effect |
| --- | --- | --- | --- |
| `otel` | no | `openjiuwen-extensions/otel`, `openjiuwen-core/otel`, `openjiuwen-session/otel`, `openjiuwen-agent-teams/otel` | live OpenTelemetry exporters and sink, the kernel trace-sink hook (`core::session::install_trace_sink` really installs the sink), and the team OTel span sink at `agent_teams::observability::OtelSpanSink` |
| `otlp` | no | `openjiuwen-agent-teams/otlp` | the team-memory OTLP bridge (`agent_teams::observability::{OtlpMemoryObserver, otlp_memory_observer}`), which forwards memory events into `openjiuwen-extensions`' `OtelTraceSink`; it also turns on that crate's `otel` feature transitively, while `OTLP_EXPORT_AVAILABLE` stays `false` |
| `sandbox` | no | `openjiuwen-sysop/sandbox` | compiles the `reqwest`-backed HTTP sandbox transport behind `core::sys_operation::sandbox` and registers it in `core::sys_operation::sandbox::provider::SandboxProviderRegistry`; without it the sandbox configuration and registry still compile, only the HTTP provider is absent |
| `live-tests` | no | `openjiuwen-foundation/live-tests`, `openjiuwen-harness/live-tests`, `openjiuwen-harness-providers/live-tests`, `openjiuwen-agent-teams/live-tests`, `openjiuwen-extensions/live-tests`, `openjiuwen-core/live-tests` | empty opt-in gates — no source path and no test in this workspace reads them; the forward only pins one switch for a downstream consumer (`openjiuwen-core/live-tests` re-forwards the foundation gate) |

Deliberately not forwarded: `openjiuwen-foundation`'s `mcp` and `tiktoken` are default-on and every dependency is taken with default features, so they cannot be switched off through the umbrella.

## Constraints

- Re-exports only: no logic, no new public data types, no `unsafe` (`#![forbid(unsafe_code)]`); the `thiserror`/`serde` stack is not a dependency of this crate.
- `publish = false` until the SDK API stabilizes; sibling crates and `docs/` are never edited from here.
- Every feature in this crate forwards to an existing sibling feature and mirrors it under the same name where the sibling uses one; the only rename is `native-v2` → `openjiuwen-harness-providers/native-v2`; `otlp` is a second observability forward kept separate from `otel` because `openjiuwen-agent-teams` splits its in-memory/tracing span seam from the OTLP memory bridge (enabling `otlp` alone still compiles `extensions::otel` through that edge), and `redis` covers two crates, `openjiuwen-extensions` (the store surface) and `openjiuwen-agent-evolving` (the RL online service's Redis-backed `OnlineStore`).
- `openjiuwen-harness-providers` is a non-optional dependency (`Cargo.toml:28`; workspace entry `../Cargo.toml:64` sets no `default-features = false`), so that crate's own `default = ["native"]` always applies and the umbrella's `native` feature cannot actually disable the providers' native provider (same for `openjiuwen-harness`, workspace `../Cargo.toml:63`).
- The `vendor` feature enables `openjiuwen-harness-providers/vendor`, so `claudecode`, `codex` and `dsh` register through the umbrella; driving them still returns `Unsupported`.
- `notify` is the one forward that gates an optional OS dependency: `openjiuwen-harness/notify` activates the optional `notify` dependency *and* compiles `watcher::backend::start`, so without it `harness::watcher::WorkspaceWatcher::start()` is a typed `Unsupported`.
- `mcp` compiles `openjiuwen-extensions`' `mcp` module (the `ToolGateway` MCP bridge) and its `openjiuwen-foundation/mcp` forward.
- `openjiuwen-foundation`'s `mcp` and `tiktoken` defaults are always on: the umbrella has no feature that could disable them.
- Python package spelling (`harness_protocol`, `agent_teams`, `sys_operation`) is mirrored even where it repeats the crate suffix, so `clippy::module_name_repetitions` is allowed crate-wide.
- The crate-level doc example is the "getting started" block: it must compile as a doctest and touch no network.
- Concrete crates are canonical: documentation lives there; this file documents only the facade.

## Notes for Agents

- `harness_providers` (`src/lib.rs`) and `harness` (`src/lib.rs`) are unconditional `pub use` aliases with no `#[cfg]`; among the prelude items only `a2a` and `otel` carry a feature gate.
- The provider registry is reached as `openjiuwen::harness_providers::{create_harness, resolve_provider, ...}`; `create_harness` is `openjiuwen_harness_providers::factory::create_harness` (proved by `tests/module_paths.rs::harness_providers_alias_points_at_openjiuwen_harness_providers`) and the prelude forwards it ungated (`src/prelude.rs:57`).
- `TeamAgent` is defined in `openjiuwen_agent_teams::runtime::team_agent` and re-exported from both `agent_teams::runtime` and the crate root, so the prelude forwards the crate-root spelling.
- Every prelude item has a verified canonical path; `Role` and `JsonObject` are intentionally not in the prelude and are addressed by their full paths in docs and tests.
- `ReActAgent`/`ReActAgentConfig` are forwarded from `core::single_agent` (`openjiuwen_agent::react`) because examples and construction code need them without spelling the kernel path; `tests/module_paths.rs::prelude_resolves_the_agent_construction_items` proves identity for the agent-construction items and the `ContextEngine`/`LongTermMemory`/`Runner` lifecycle items.
- `LlmAgent` is forwarded with the crate-root spelling (`openjiuwen_agent::LlmAgent`); `core::application` exposes the same type as `LLMAgent`, so the prelude deliberately carries only one of the two spellings.
- `openjiuwen::core::session` is the whole `openjiuwen-session` crate root (`openjiuwen-core/src/lib.rs:67` re-exports the crate as `session`), so `AgentSession`, `AgentSessionOptions`, `create_agent_session`, `Session`, `SessionController`, `DiskSessionStore`, `Version`, `SessionType`, `SessionConfig`, `Checkpointer`, `MemoryCheckpointer`, `FileCheckpointer`, `Checkpoint`, `CheckpointMeta`, `CheckpointerConfig`, `create_checkpointer`, `StreamItem` and `StreamEvent` are all reachable directly at `openjiuwen::core::session::<Name>`; the nested `session_controller::*`, `vcs::*`, `checkpoint::*` and inner `session::*` modules are alternative spellings of the same items.
- The `sqlite` feature is wired to `openjiuwen-foundation`, `openjiuwen-session`, `openjiuwen-memory`, `openjiuwen-extensions`, `openjiuwen-symphony` and `openjiuwen-agent-teams`; `cargo check -p openjiuwen --features sqlite` succeeds.
- Feature gates per dependency: `native-v2` and `vendor` reach `harness_providers::providers::{native_v2}` and `providers::{vendor, claudecode, codex, dsh}`; `browser`/`cua`/`mobile` reach `harness::subagents::SubAgentPreset`; `sqlite` reaches `core::session::checkpoint::SqliteCheckpointer`, `core::foundation::store::SqliteKvStore`, `extensions::store::SqliteKvStore`, `agent_teams::storage::SqliteTeamStore` and the `sqlite` constructors of `core::memory::store::MemoryStores`/`symphony::store::KvAssetStore`; `redis`/`pulsar` reach `extensions::store::redis` and `extensions::message_queue::{PulsarConfig, PulsarMessageQueue}`; `distributed` reaches `agent_teams::messager::PyZmqMessager`; `sandbox` reaches `core::sys_operation::sandbox::provider::SandboxProviderRegistry`; `otel` reaches `agent_teams::observability::{OtelSpanSink, OtelHooks, SpanRecord}` in addition to the kernel trace-sink hook.
- `otlp` reaches `agent_teams::observability::{OtlpMemoryObserver, otlp_memory_observer}` and transitively enables `openjiuwen-extensions/otel`, so `openjiuwen::extensions::otel::OtelTracerConfig` resolves under `otlp` alone and `OTLP_EXPORT_AVAILABLE` remains `false`.
- `redis` reaches both `extensions::store::redis::{RedisStore, RedisStoreConfig}` and `agent_evolving::online::{RedisOnlineStore, RedisOnlineStoreOptions}`; both reject TLS (`rediss://`) and cluster configurations with `Unsupported`, and neither test connects.
- These direct dependencies declare no features at all, so there is nothing to forward: `openjiuwen-common`, `openjiuwen-graph`, `openjiuwen-workflow`, `openjiuwen-context`, `openjiuwen-retrieval`, `openjiuwen-security`, `openjiuwen-agent`, `openjiuwen-multiagent`, `openjiuwen-runner` (no optional dependencies at all), `openjiuwen-harness-protocol`, `openjiuwen-dev-tools`, `openjiuwen-rsi` and `openjiuwen-auto-harness`.
- The prelude stays feature-free apart from `a2a`/`otel`: Python's top-level `openjiuwen/__init__.py` exports only `__version__`, so feature-gated trees are reached by module path rather than through `prelude::*`.
- `core` is `pub use openjiuwen_core as core` (`src/lib.rs:131`) and that crate aliases the whole foundation crate (`pub use openjiuwen_foundation as foundation`, `../openjiuwen-core/src/lib.rs:62`), so `openjiuwen::core::foundation::…` is the full `openjiuwen_foundation` module tree, not a hand-written shim.
- Because `core` is a whole-crate alias, every kernel module tree the kernel umbrella grows is reachable without a change here: `core::{controller, kv_cache, operator}` (globbed from `openjiuwen-agent`), `core::session::{session_controller, vcs}`, `core::runner::{pool, resources, callback, hitl, monitor, spawn, dreaming}`, `core::memory::{lite, analyzer, external, team, dreaming}` and the rest of the `§ Notable module trees` table in `src/lib.rs`.
- The newest kernel trees are reachable only through those whole-crate aliases: `openjiuwen::core::single_agent::{memory_provider, usage, policy, prompt_pack}` (a blanket glob of `openjiuwen-agent`), `core::context_engine::{usage, telemetry, policy, tokenizers}`, `core::foundation::llm::{external_provider, responses}`, `core::runner::dreaming` and `agent_teams::memory`.
- Tools resolve at both depths: `openjiuwen::core::foundation::{Tool, ToolCard, ToolContext, ToolOutput, FunctionTool, FunctionToolBuilder, ToolHandler, schema_for, RestfulApiTool, openai_tools}` (flat, `../openjiuwen-foundation/src/lib.rs:102`) and `openjiuwen::core::foundation::tool::{…same items…}` (`../openjiuwen-foundation/src/tool/mod.rs:31`); `Tool`/`FunctionTool` are also in the prelude (`src/prelude.rs:44`).
- Only three crates consume this facade: `openjiuwen-cli` (the attach types, `openjiuwen::agent_teams::{…}` at `../openjiuwen-cli/src/team/mod.rs:58-61`), `openjiuwen-e2e` and `openjiuwen-examples` (both dev crates, which must use the umbrella by workspace rule); every library crate depends on the concrete crate it needs.
- The `#[tool]` proc macro is reachable through no alias of this crate: `openjiuwen` does not depend on `openjiuwen-macros` (`Cargo.toml:11-35`) and `openjiuwen-foundation` neither uses nor re-exports it, so a consumer adds `openjiuwen-macros` as its own dependency and writes `#[openjiuwen_macros::tool(...)]` (generated code also needs `async-trait` in the consumer).

## Routing Table

- `./src/lib.rs` → crate docs (Python→Rust table, notable module trees, per-block feature matrix, getting-started doctest), module aliases, `crate_name()`, `VERSION`
- `./src/prelude.rs` → curated convenience re-exports, the "prelude is convenience only" note, and the "why the prelude stays feature-free" paragraph
- `./Cargo.toml` → dependency list and the feature forwarding table
- `./tests/module_paths.rs` → identity checks for one representative item per module alias, the `core` `controller`/`kv_cache`/`operator` packages, one item per grown module tree (including the newest `single_agent`/`context_engine`/`foundation::llm`/`runner::dreaming`/`agent_teams::memory` paths), prelude compile/identity checks (common, agent-construction, extended items), the crate `VERSION`, plus one feature-gated test per umbrella feature (`native-v2`, `vendor`, `sqlite`, `redis`, the agent-evolving `redis` half, `pulsar`, `mcp`, `distributed`, `sandbox`, `otel`, `otlp`, `browser`, `cua`, `mobile`), and the ungated `notify` test that covers both watcher builds

## Test Strategy

- `tests/module_paths.rs` asserts `TypeId` identity (and `std::ptr::eq` for functions) between each alias and the concrete crate, so a mis-pointed alias fails loudly; `a2a` and every dependency-forwarded item sit behind `#[cfg(feature = ...)]`.
- `core_alias_exposes_the_new_kernel_modules` and `every_public_module_tree_is_reachable` walk the kernel and product-framework subtrees (`core::session::{session_controller, vcs}`, `core::runner::{pool, resources, hitl, monitor, spawn}`, `core::memory::{lite, analyzer, external, team, dreaming}`, `core::retrieval::{sparse, rewrite, parsing}`, `harness::{lsp, spec, worktree, subagents, tools}`, `agent_teams::{attach, org, kv_cache, models, monitor, observability}`, `symphony::{discovery, experience, flow, llm, object_artifact}`, `dev_tools::{judge, tune, skill_creator, agent_builder}`, `extensions::context_evolver`), so a module tree that stops being re-exported fails there; the `same_paths!` helper keeps each check to one line.
- `rsi_alias_points_at_openjiuwen_rsi` also proves `rsi::auto_harness::{AutoHarnessConfig, AutoHarnessOrchestrator}` is `openjiuwen_rsi::harness_rsi::auto_harness::{…}`.
- `core_alias_exposes_the_newest_agent_context_and_llm_trees` asserts the `single_agent`/`context_engine`/`foundation::llm`/`runner::dreaming`/`agent_teams::memory` paths against the concrete crates.
- `prelude_resolves_the_common_items`, `prelude_resolves_the_agent_construction_items` and `prelude_resolves_the_extended_items` import `openjiuwen::prelude::*` and assert both usability and `TypeId` identity against the concrete crates, so a dropped or mis-pointed prelude export fails loudly.
- `crate_version_is_exported` asserts `VERSION` equals `CARGO_PKG_VERSION` and is non-empty; `crate_name_is_the_python_package_root` asserts `crate_name() == "openjiuwen"`.
- `tests/module_paths.rs` holds 37 test functions in one binary; `native` and `live-tests` have no test of their own (they gate no umbrella-visible source), `notify` has one ungated test that branches on `cfg!(feature = "notify")` to cover both watcher builds, and `auto_harness_alias_points_at_openjiuwen_auto_harness` compares `crate_name` function pointers because that shim exposes no types.
- `otel_feature_reaches_the_kernel_trace_sink_hook` is `#[cfg(feature = "otel")]` and proves the umbrella forwards `openjiuwen-core/otel`, so `core::session::install_trace_sink` is the session crate's installed-sink function; `otel_feature_reaches_the_agent_teams_span_sink` proves the `openjiuwen-agent-teams/otel` half.
- `native_v2_feature_reaches_the_agent_teams_hosted_provider` and `vendor_feature_reaches_the_vendor_providers` assert type identity *and* that `harness_providers::providers::builtin_provider` returns `Some(Ok(_))` for `native_v2`, `claudecode`, `codex` and `dsh`.
- `sqlite_feature_reaches_every_backend` covers the checkpointer, the key-value store (both spellings), the team store, and binds the `MemoryStores::sqlite`/`KvAssetStore::sqlite` function items without opening a database.
- `redis_feature_reaches_the_store_surface` and `pulsar_feature_reaches_the_message_queue_surface` assert type identity only; the extensions' Redis store connects lazily over `redis://` (only `rediss://` TLS and cluster slot routing are `Unsupported`) and the Pulsar queue rejects at construction — nothing connects.
- `notify_feature_gates_the_workspace_watcher_backend` drives one poll of `harness::watcher::WorkspaceWatcher::new().start()` (no runtime, no filesystem) and asserts the empty-plan error is `Config` with `notify` and `Unsupported` naming the feature without it, so the forward is proven rather than assumed.
- `redis_feature_also_reaches_the_agent_evolving_store` is `#[cfg(feature = "redis")]` and asserts type identity for the RL online store only — nothing connects.
- `otlp_feature_reaches_the_team_memory_bridge` is `#[cfg(feature = "otlp")]` and asserts type plus function-pointer identity for `OtlpMemoryObserver`/`otlp_memory_observer`, that `OTLP_EXPORT_AVAILABLE == false`, and builds one in-memory observer.
- `mcp_feature_is_wired_and_the_team_mcp_surface_resolves` keeps the umbrella's `mcp` forward exercised and asserts the always-on `agent_teams::mcp` server surface matches the concrete crate.
- `distributed_feature_reaches_the_messager_seam` and `sandbox_feature_reaches_the_provider_registry` assert the `PyZmqMessager` and `SandboxProviderRegistry` types.
- `browser_feature_reaches_the_reporting_preset`, `cua_feature_reaches_the_reporting_preset` and `mobile_feature_reaches_the_reporting_preset` call `harness::subagents::build_preset` and assert the `Unsupported` message names the preset (`browser_agent`/`cua_agent`/`mobile_gui_agent`) and its cargo feature; they touch no network or filesystem.
- The crate-level getting-started example is the doctest that guards the primary public paths (`cargo test -p openjiuwen`).
- No unit tests: the facade has no behavior; behavior is tested in the concrete crates.
- Definition of done: `cargo fmt -p openjiuwen`, `cargo test -p openjiuwen` (default and `--all-features`), `cargo clippy -p openjiuwen --all-targets -- -D warnings` (default and `--all-features`), `cargo doc -p openjiuwen --no-deps` (default and `--all-features`), and a `cargo check -p openjiuwen --no-default-features --features <one>` sweep over every feature.
