//! Top-level umbrella crate: the Rust counterpart of the Python package root `openjiuwen`.
//!
//! This crate is a pure facade. Every item it exposes is a re-export of a concrete
//! workspace crate, so code can keep the Python-shaped dotted paths it uses in the
//! reference SDK (`openjiuwen::harness::DeepAgent`,
//! `openjiuwen::core::single_agent::AgentCard`, …) without depending on each crate
//! individually. The concrete crates remain canonical: they own the documentation,
//! behavior, semver and tests.
//!
//! # Python → Rust module aliases
//!
//! | Python | Rust | Concrete crate |
//! | --- | --- | --- |
//! | `openjiuwen.core` | [`core`] | `openjiuwen-core` (kernel umbrella) |
//! | `openjiuwen.harness` | [`harness`] | `openjiuwen-harness` |
//! | `openjiuwen.harness_protocol` | [`harness_protocol`] | `openjiuwen-harness-protocol` |
//! | `openjiuwen.harness_providers` | [`harness_providers`] | `openjiuwen-harness-providers` |
//! | `openjiuwen.agent_teams` | [`agent_teams`] | `openjiuwen-agent-teams` |
//! | `openjiuwen.extensions` | [`extensions`] | `openjiuwen-extensions` |
//! | `openjiuwen.dev_tools` | [`dev_tools`] | `openjiuwen-dev-tools` |
//! | `openjiuwen.symphony` | [`symphony`] | `openjiuwen-symphony` |
//! | `openjiuwen.rsi` | [`rsi`] | `openjiuwen-rsi` |
//! | `openjiuwen.agent_evolving` | [`agent_evolving`] | `openjiuwen-agent-evolving` |
//! | `openjiuwen.auto_harness` | [`auto_harness`] | `openjiuwen-auto-harness` (shim over `rsi::auto_harness`) |
//!
//! Python's top-level `openjiuwen/__init__.py` exports exactly one name, `__version__`;
//! here that is [`VERSION`], alongside the module aliases above.
//!
//! # Notable module trees
//!
//! Every alias below is a re-export of the whole concrete crate, so each crate's
//! public module tree is reachable through the umbrella unchanged. The subtrees worth
//! knowing about:
//!
//! | Path | Contents |
//! | --- | --- |
//! | `core::{controller, kv_cache, operator}` | controller stack, KV-cache runtime, tunable operators (hosted by `openjiuwen-agent`) |
//! | `core::session::{session_controller, vcs, checkpoint}` | disk-backed product sessions, local session version history, checkpointers (the `sqlite` member is behind the `sqlite` feature) |
//! | `core::runner::{pool, resources, callback, hitl, monitor, spawn, dreaming}` | team pool, resource registry, callbacks, HITL, run monitor, subprocess spawn, the dreaming service |
//! | `core::foundation::store::external` | external-store registry and provider seams |
//! | `core::memory::{lite, analyzer, external, team, dreaming, store}` | coding memory, extraction/consolidation, external providers, team memory, dreaming, pluggable stores (the `sqlite` constructor is behind the `sqlite` feature) |
//! | `core::retrieval::{sparse, rewrite, graph, multimodal, parsing}` | BM25, query rewriting, graph/multimodal/parsed retrieval |
//! | `core::sys_operation::sandbox::provider` | sandbox provider registry (the `sandbox` transport is behind the `sandbox` feature) |
//! | `harness::{lsp, spec, worktree, subagents, tools, watcher}` | LSP client, spec registry, git worktrees, subagents, built-in tool catalog, debounced workspace watcher (the `browser`/`cua`/`mobile` presets are behind the matching features and report `Unsupported`; `watcher::WorkspaceWatcher::start` needs the `notify` feature and reports `Unsupported` without it) |
//! | `harness_providers::providers::{native, native_v2, vendor, claudecode, codex, dsh}` | the built-in provider table: `native` is always compiled in, `native_v2` is behind the `native-v2` feature, the vendor tree (`vendor`, `claudecode`, `codex`, `dsh`) is behind the `vendor` feature |
//! | `agent_teams::{attach, org, monitor, models, kv_cache, memory, observability, storage}` | cross-process attach (`attach` is feature-free; its transport seams are not), org namespacing, member cache affinity, model routing, monitoring, team memory configuration, tools and wiring, observability (the `OTel` sink is behind the `otel` feature, the OTLP memory bridge behind `otlp`), team stores (the SQLite store is behind the `sqlite` feature) |
//! | `core::single_agent::{memory_provider, usage, policy, prompt_pack}` | provider-memory rail and tools, usage telemetry, client-tool/exposure policy, prompt packs (`core::{controller, kv_cache, operator}` above are the same crate's other top-level packages) |
//! | `core::context_engine::{usage, telemetry, policy, tokenizers}` | usage accounting, compression telemetry and recorder, window/change policy, tokenizer selection (native artifacts are pure Rust; remote downloads are typed `Unsupported`) |
//! | `core::foundation::llm::{external_provider, responses}` | external-provider auth/catalog registry and factories; the `OpenAI` `Responses` transport helpers |
//! | `symphony::{discovery, experience, flow, llm, object_artifact, store}` | skill discovery, experience bank, distillation flow, LLM judges, artifact persistence, capability-asset store (the `sqlite` constructor is behind the `sqlite` feature) |
//! | `dev_tools::{judge, tune, skill_creator, agent_builder}` | LLM-as-judge, prompt tuning, skill creation, agent/workflow builder |
//! | `extensions::{context_evolver, store, message_queue}` | context-evolution pipeline, store adapters (Redis behind `redis`, SQLite behind `sqlite`), message queues (Pulsar behind `pulsar`) |
//! | `rsi::auto_harness` | the self-optimisation orchestrator (`AutoHarnessOrchestrator`, `AutoHarnessConfig`) re-exported from `openjiuwen-rsi` |
//!
//! # Feature matrix
//!
//! Dependencies are non-optional and taken with their own default features, so a
//! forwarded feature can only *add* capability: `openjiuwen-harness`'s `native` and
//! `openjiuwen-harness-providers`'s `native` always apply, and
//! `openjiuwen-foundation`'s `mcp`/`tiktoken` defaults cannot be switched off from here.
//! Every feature that is off by default keeps the default build free of network access
//! and external services.
//!
//! `otlp` and `otel` are deliberately separate forwards: `openjiuwen-agent-teams` keeps
//! its in-memory/tracing span seam and its OTLP memory bridge behind two features, and
//! enabling `otlp` alone still compiles `extensions::otel`, because the agent-teams edge
//! pulls `openjiuwen-extensions` with that feature.
//!
//! ## Harness and providers
//!
//! | Feature | Default | Forwards to | Effect |
//! | --- | --- | --- | --- |
//! | `native` | yes | `openjiuwen-harness/native`, `openjiuwen-harness-providers/native` | the in-process `native` provider (that is the half which really gates source); `openjiuwen-harness/native` is an empty marker — no source reads it and the native `DeepAgent` harness is always compiled |
//! | `native-v2` | no | `openjiuwen-harness-providers/native-v2` | the agent-teams-hosted `native_v2` provider (`harness_providers::providers::native_v2`) |
//! | `vendor` | no | `openjiuwen-harness-providers/vendor` | registers the vendor providers `claudecode`, `codex`, `dsh` (each still answers `Unsupported` when driven) |
//! | `browser` | no | `openjiuwen-harness/browser` | the `browser` subagent preset, which reports its missing implementation as `Unsupported` |
//! | `cua` | no | `openjiuwen-harness/cua` | the `cua` subagent preset, which reports its missing implementation as `Unsupported` |
//! | `mobile` | no | `openjiuwen-harness/mobile` | the `mobile` subagent preset, which reports its missing implementation as `Unsupported` |
//! | `notify` | no | `openjiuwen-harness/notify` | compiles the harness's `notify`-backed watcher backend, so `harness::watcher::WorkspaceWatcher::start` really arms an OS watcher; without the feature the watcher API still resolves and `start` answers `Unsupported` (the `openjiuwen` binary enables the harness feature on its own, so its `watch` and `chat --watch` commands never depend on this forward) |
//!
//! ## Storage backends
//!
//! | Feature | Default | Forwards to | Effect |
//! | --- | --- | --- | --- |
//! | `sqlite` | no | `openjiuwen-foundation/sqlite`, `openjiuwen-session/sqlite`, `openjiuwen-memory/sqlite`, `openjiuwen-extensions/sqlite`, `openjiuwen-symphony/sqlite`, `openjiuwen-agent-teams/sqlite` | SQLite checkpointer, key-value/document stores, memory stores, capability-asset store, and team store |
//!
//! ## Transports and remote backends
//!
//! | Feature | Default | Forwards to | Effect |
//! | --- | --- | --- | --- |
//! | `a2a` | no | `openjiuwen-extensions/a2a` | Agent-to-Agent client/server modules |
//! | `mcp` | no | `openjiuwen-extensions/mcp` | activates that crate's optional `rmcp`/`tokio-stream` dependencies only: it gates no source there and exposes no `extensions::mcp` module, while agent-teams' MCP server is compiled regardless because that crate's `mcp` default applies |
//! | `redis` | no | `openjiuwen-extensions/redis`, `openjiuwen-agent-evolving/redis` | the extensions' Redis KV/document/vector stores and `RedisCheckpointer` over its in-crate RESP client (`redis://` connects lazily; `rediss://` TLS and cluster slot routing stay `Unsupported`), plus the RL online service's Redis-backed `OnlineStore` (`agent_evolving::online::{RedisOnlineStore, RedisOnlineStoreOptions}`, likewise `Unsupported` for TLS and cluster) |
//! | `pulsar` | no | `openjiuwen-extensions/pulsar` | the extensions' Pulsar message-queue module, whose construction always returns `Unsupported` (no broker client is vendored) |
//! | `distributed` | no | `openjiuwen-agent-teams/distributed` | the distributed `PyZmqMessager` seam, which rejects at construction |
//!
//! ## Observability, sandbox, test gates
//!
//! | Feature | Default | Forwards to | Effect |
//! | --- | --- | --- | --- |
//! | `otel` | no | `openjiuwen-extensions/otel`, `openjiuwen-core/otel`, `openjiuwen-session/otel`, `openjiuwen-agent-teams/otel` | live OpenTelemetry exporters and sink, the kernel trace-sink hook (`core::session::install_trace_sink` really installs the sink), and the team `OTel` span sink (`agent_teams::observability::OtelSpanSink`) |
//! | `otlp` | no | `openjiuwen-agent-teams/otlp` | the team-memory OTLP bridge (`agent_teams::observability::{OtlpMemoryObserver, otlp_memory_observer}`), which forwards memory events into `openjiuwen-extensions`' `OtelTraceSink`; it also turns on that crate's `otel` feature transitively, while `OTLP_EXPORT_AVAILABLE` stays `false` |
//! | `sandbox` | no | `openjiuwen-sysop/sandbox` | compiles the `reqwest`-backed HTTP sandbox transport behind `core::sys_operation::sandbox` and registers it in the provider registry; without it the sandbox configuration and registry still compile, only the HTTP provider is absent |
//! | `live-tests` | no | `openjiuwen-foundation/live-tests`, `openjiuwen-harness/live-tests`, `openjiuwen-harness-providers/live-tests`, `openjiuwen-agent-teams/live-tests`, `openjiuwen-extensions/live-tests`, `openjiuwen-core/live-tests` | empty opt-in gates — no source path and no test in this workspace reads them; the forward only pins one switch for a downstream consumer (`openjiuwen-core/live-tests` re-forwards the foundation gate) |
//!
//! # Getting started
//!
//! ```
//! use openjiuwen::prelude::*;
//!
//! let card = AgentCard::new("my-agent");
//! assert_eq!(card.name, "my-agent");
//!
//! let message = BaseMessage::user("hello");
//! assert_eq!(message.role, openjiuwen::core::foundation::Role::User);
//!
//! let session = AgentSession::new("s-1");
//! assert_eq!(session.id(), "s-1");
//!
//! assert!(!openjiuwen::VERSION.is_empty());
//! ```
#![forbid(unsafe_code)]
// The aliases below keep the Python package spelling, which repeats the crate
// suffix (`harness_protocol`, `agent_teams`, `sys_operation`, …).
#![allow(clippy::module_name_repetitions)]

// --- module aliases (alphabetical) ------------------------------------------
pub use openjiuwen_agent_evolving as agent_evolving;
pub use openjiuwen_agent_teams as agent_teams;
pub use openjiuwen_auto_harness as auto_harness;
pub use openjiuwen_core as core;
pub use openjiuwen_dev_tools as dev_tools;
pub use openjiuwen_extensions as extensions;
pub use openjiuwen_harness as harness;
pub use openjiuwen_harness_protocol as harness_protocol;
pub use openjiuwen_harness_providers as harness_providers;
pub use openjiuwen_rsi as rsi;
pub use openjiuwen_symphony as symphony;

pub mod prelude;

/// Returns the crate name.
#[must_use]
pub const fn crate_name() -> &'static str {
    "openjiuwen"
}

/// Version of this facade crate — the Rust counterpart of Python's `openjiuwen.__version__`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
