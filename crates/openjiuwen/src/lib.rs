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
//! | `core::runner::{pool, resources, callback, hitl, monitor, spawn}` | team pool, resource registry, callbacks, HITL, run monitor, subprocess spawn |
//! | `core::foundation::store::external` | external-store registry and provider seams |
//! | `core::memory::{lite, analyzer, external, team, dreaming, store}` | coding memory, extraction/consolidation, external providers, team memory, dreaming, pluggable stores (the `sqlite` constructor is behind the `sqlite` feature) |
//! | `core::retrieval::{sparse, rewrite, graph, multimodal, parsing}` | BM25, query rewriting, graph/multimodal/parsed retrieval |
//! | `core::sys_operation::sandbox::provider` | sandbox provider registry (the `sandbox` transport is behind the `sandbox` feature) |
//! | `harness::{lsp, spec, worktree, subagents, tools}` | LSP client, spec registry, git worktrees, subagents, built-in tool catalog (the `browser`/`cua`/`mobile` presets are behind the matching features and report `Unsupported`) |
//! | `harness_providers::providers::{native, native_v2, vendor, claudecode, codex, dsh}` | the built-in provider table: `native` is always compiled in, `native_v2` is behind the `native-v2` feature, the vendor tree (`vendor`, `claudecode`, `codex`, `dsh`) is behind the `vendor` feature |
//! | `agent_teams::{attach, org, monitor, models, kv_cache, observability, storage}` | cross-process attach (`attach` is feature-free; its transport seams are not), org namespacing, member cache affinity, model routing, monitoring, observability (the `OTel` sink is behind the `otel` feature), team stores (the SQLite store is behind the `sqlite` feature) |
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
//! ## Harness and providers
//!
//! | Feature | Default | Forwards to | Effect |
//! | --- | --- | --- | --- |
//! | `native` | yes | `openjiuwen-harness/native`, `openjiuwen-harness-providers/native` | in-process native `DeepAgent` harness and provider |
//! | `native-v2` | no | `openjiuwen-harness-providers/native-v2` | the agent-teams-hosted `native_v2` provider (`harness_providers::providers::native_v2`) |
//! | `vendor` | no | `openjiuwen-harness-providers/vendor` | registers the vendor providers `claudecode`, `codex`, `dsh` (each still answers `Unsupported` when driven) |
//! | `browser` | no | `openjiuwen-harness/browser` | the `browser` subagent preset, which reports its missing implementation as `Unsupported` |
//! | `cua` | no | `openjiuwen-harness/cua` | the `cua` subagent preset, which reports its missing implementation as `Unsupported` |
//! | `mobile` | no | `openjiuwen-harness/mobile` | the `mobile` subagent preset, which reports its missing implementation as `Unsupported` |
//! | `notify` | no | `openjiuwen-harness/notify` | activates the workspace file-watcher dependency (no umbrella-visible type) |
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
//! | `mcp` | no | `openjiuwen-extensions/mcp` | the extensions' MCP adapter surface (it currently gates no source; agent-teams' MCP server is always built) |
//! | `redis` | no | `openjiuwen-extensions/redis` | the Redis store surface, whose construction returns `Unsupported` |
//! | `pulsar` | no | `openjiuwen-extensions/pulsar` | the Pulsar message-queue surface, whose construction returns `Unsupported` |
//! | `distributed` | no | `openjiuwen-agent-teams/distributed` | the distributed `PyZmqMessager` seam, which rejects at construction |
//!
//! ## Observability, sandbox, test gates
//!
//! | Feature | Default | Forwards to | Effect |
//! | --- | --- | --- | --- |
//! | `otel` | no | `openjiuwen-extensions/otel`, `openjiuwen-core/otel`, `openjiuwen-session/otel`, `openjiuwen-agent-teams/otel` | live OpenTelemetry exporters and sink, the kernel trace-sink hook (`core::session::install_trace_sink` really installs the sink), and the team `OTel` span sink (`agent_teams::observability::OtelSpanSink`) |
//! | `sandbox` | no | `openjiuwen-sysop/sandbox` | the sandbox provider transport (`core::sys_operation::sandbox`) |
//! | `live-tests` | no | `openjiuwen-foundation/live-tests`, `openjiuwen-harness/live-tests`, `openjiuwen-harness-providers/live-tests`, `openjiuwen-agent-teams/live-tests`, `openjiuwen-extensions/live-tests` | opt-in network/vendor test gates |
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
