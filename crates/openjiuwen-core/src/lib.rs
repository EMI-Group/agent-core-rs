//! Kernel umbrella: the Rust counterpart of the Python `openjiuwen.core` package tree.
//!
//! This crate is a thin re-export facade. It owns no logic of its own: every module
//! listed here is a public alias for a concrete kernel crate, so downstream code can
//! address the kernel with the same dotted paths it uses in Python:
//!
//! | Python | Rust |
//! | --- | --- |
//! | `openjiuwen.core.common` | [`common`] → `openjiuwen_common` |
//! | `openjiuwen.core.foundation` | [`foundation`] → `openjiuwen_foundation` |
//! | `openjiuwen.core.session` | [`session`] → `openjiuwen_session` |
//! | `openjiuwen.core.graph` | [`graph`] → `openjiuwen_graph` |
//! | `openjiuwen.core.workflow` | [`workflow`] → `openjiuwen_workflow` |
//! | `openjiuwen.core.context_engine` | [`context_engine`] → `openjiuwen_context` |
//! | `openjiuwen.core.memory` | [`memory`] → `openjiuwen_memory` |
//! | `openjiuwen.core.retrieval` | [`retrieval`] → `openjiuwen_retrieval` |
//! | `openjiuwen.core.sys_operation` | [`sys_operation`] → `openjiuwen_sysop` |
//! | `openjiuwen.core.security` | [`security`] → `openjiuwen_security` |
//! | `openjiuwen.core.single_agent` | [`single_agent`] → `openjiuwen_agent` |
//! | `openjiuwen.core.application` | [`application`] → the application agents in `openjiuwen_agent` |
//! | `openjiuwen.core.controller` | [`controller`] → `openjiuwen_agent::controller` |
//! | `openjiuwen.core.kv_cache` | [`kv_cache`] → `openjiuwen_agent::kv_cache` |
//! | `openjiuwen.core.operator` | [`operator`] → `openjiuwen_agent::operator` |
//! | `openjiuwen.core.multi_agent` | [`multi_agent`] → `openjiuwen_multiagent` |
//! | `openjiuwen.core.runner` | [`runner`] → `openjiuwen_runner` |
//!
//! `controller`, `kv_cache`, and `operator` are Python top-level packages hosted by
//! `openjiuwen-agent`'s modules of the same name; this umbrella re-exports those
//! modules wholesale, so nested paths such as `controller::schema::Task`,
//! `kv_cache::runtime`, and `operator::base` stay reachable. Because
//! [`single_agent`] is a blanket glob of the same crate, those items are also
//! exposed there as an alternative spelling.
//!
//! `context_engine`, `memory`, `retrieval`, `workflow`, `session`, `foundation`,
//! `graph`, `sys_operation`, `security`, `common`, `multi_agent`, and `runner` are
//! whole-crate aliases (and [`single_agent`] is a blanket glob of `openjiuwen-agent`),
//! so *every* nested public module of those kernel crates is reachable under its
//! Python-shaped path without being re-listed here. The deeper trees the kernel
//! exposes today include:
//!
//! - `single_agent::{memory_provider, usage, prompt_pack, abilities, controller,
//!   kv_cache, operator, policy, rails}` — the agent seam, ability manager, rails,
//!   interrupts, and the controller/cache/operator stacks shared with the shims above.
//! - `context_engine::{usage, telemetry, policy, tokenizers,
//!   processor::rule_compression, recall, reload, offload, window}` — token
//!   accounting, window policies, the compressor and offload processors, and recall
//!   archives.
//! - `memory::{lite::{facts, embed, session, local, markdown, fts}, dreaming, team,
//!   external, store::migration, graph, analyzer}` — long-term, entity/graph, and team
//!   memory plus the lite coding memory; `lite::fts` and the items of
//!   `store::migration::runner` are gated on the `sqlite` feature.
//! - `retrieval::{loaders::{pptx, odt, docx, xlsx, pdf, html, csv, json, url,
//!   fixtures}, parsing::{zip, pptx, odt, docx, pdf, xlsx, inflate, xml}, chunking,
//!   embedding, vector_store, retriever, reranker, rewrite}` — document loading and
//!   parsing, chunking, embeddings, vector stores, retrievers, rerankers, and query
//!   rewriting.
//! - `workflow::{single, components, engine, render, registry, interrupt}` — the
//!   builder and run engine, built-in components, rendering, and HITL interrupts.
//! - `runner::{pool, resources, callback, hitl, monitor, spawn}` — resource managers,
//!   the callback framework, the HITL hub, the monitor, and the spawners.
//!
//! # Usage
//!
//! The umbrella exists for consumers that want Python-shaped paths without depending
//! on each layer-1/2/3 crate individually:
//!
//! ```
//! use openjiuwen_core::foundation::prompt::{BaseMessage, Role};
//! use openjiuwen_core::session::SessionConfig;
//!
//! let message = BaseMessage::user("hello");
//! assert!(matches!(message.role, Role::User));
//! assert!(SessionConfig::default().agent_config.is_none());
//! ```
//!
//! The concrete crates remain the canonical home of these items: documentation,
//! semver guarantees, and tests live there, and code internal to the workspace should
//! depend on the concrete crate (`openjiuwen-foundation`, `openjiuwen-agent`, …) rather
//! than routing through this umbrella. Layers that already depend on the kernel
//! crates directly (for example `openjiuwen-core`'s siblings) must not depend on this
//! crate, which keeps the dependency graph acyclic.
//!
//! # Feature forwarding
//!
//! The default feature set is empty: with no feature enabled the umbrella builds the
//! kernel exactly as the concrete crates' defaults do. Four opt-in features forward to
//! the crate that owns the gated surface, so a consumer can enable it here instead of
//! adding a direct dependency on that crate:
//!
//! - `otel` → `openjiuwen-session/otel`, which makes `session::install_trace_sink`
//!   really install the supplied sink (without it the helper is a documented no-op).
//! - `sqlite` → `openjiuwen-foundation/sqlite`, `openjiuwen-session/sqlite`, and
//!   `openjiuwen-memory/sqlite`, which add the SQLite key-value and document stores,
//!   the SQLite checkpointer, and memory's FTS5 coding index (`memory::lite::fts`) plus
//!   its `store::migration::runner` migration items.
//! - `sandbox` → `openjiuwen-sysop/sandbox`, which compiles the sandbox HTTP gateway
//!   client in `sys_operation::sandbox::http`; its remote calls still report
//!   `openjiuwen_common::error::Error::Unsupported`.
//! - `live-tests` → `openjiuwen-foundation/live-tests`, the concrete crate's opt-in
//!   network tests.
//!
//! These four forwarders cover the whole gated kernel surface, so the set is complete:
//! the kernel crates declare exactly `mcp`, `tiktoken`, `sqlite`, and `live-tests`
//! (`openjiuwen-foundation`), `sqlite` and `otel` (`openjiuwen-session`), `sqlite`
//! (`openjiuwen-memory`), and `sandbox` (`openjiuwen-sysop`). Every other kernel crate
//! this umbrella aliases — `openjiuwen-common`, `graph`, `workflow`, `context_engine`,
//! `retrieval`, `security`, `single_agent` (the `openjiuwen-agent` crate), `multi_agent`,
//! and `runner` — declares
//! `default = []` with no other feature and no optional dependency, so there is nothing
//! further to forward; `openjiuwen-runner` in particular declares no feature at all. Note
//! that the `sqlite` forwarder also turns on `openjiuwen-foundation/sqlite` through
//! `openjiuwen-memory/sqlite`.
//!
//! `mcp` and `tiktoken` are deliberately **not** exposed: they are
//! `openjiuwen-foundation`'s default features and the dependency is taken with default
//! features, so Cargo's additive feature model means an umbrella feature could never
//! turn them off. Harness, teams, and extension features (`distributed`, `vendor`,
//! `notify`, `browser`/`cua`/`mobile`) belong to higher layers and are out of scope
//! here.
#![forbid(unsafe_code)]
// The `core`/`common`, `foundation`/`…_foundation` naming of the aliases below
// intentionally mirrors the Python package tree.
#![allow(clippy::module_name_repetitions)]

// --- layer 0/1/2 primitives -------------------------------------------------
pub use openjiuwen_common as common;
pub use openjiuwen_context as context_engine;
pub use openjiuwen_foundation as foundation;
pub use openjiuwen_graph as graph;
pub use openjiuwen_memory as memory;
pub use openjiuwen_retrieval as retrieval;
pub use openjiuwen_security as security;
pub use openjiuwen_session as session;
pub use openjiuwen_sysop as sys_operation;
pub use openjiuwen_workflow as workflow;

// --- layer 3/4 agents, teams, runner ---------------------------------------
pub use openjiuwen_multiagent as multi_agent;
pub use openjiuwen_runner as runner;

/// The Python `openjiuwen.core.single_agent` package: the agent card, the `ReAct`
/// agent loop and its configuration, the ability manager, rails, and interrupts.
///
/// This is a blanket re-export of [`openjiuwen_agent`]'s public surface; the
/// application agents ([`LLMAgent`](application::LLMAgent),
/// [`WorkflowAgent`](application::WorkflowAgent)) live in [`application`]
/// to match the Python split.
pub mod single_agent {
    pub use openjiuwen_agent::*;
}

/// The Python `openjiuwen.core.application` package: agents composed from a fixed
/// model/workflow configuration rather than driven by a caller-supplied loop.
///
/// Only the application-agent surface is re-exported here — not the whole
/// `openjiuwen-agent` crate — so this module stays aligned with the Python package.
pub mod application {
    pub use openjiuwen_agent::{
        create_llm_agent_config, ConstrainConfig, DefaultResponse, LlmAgent as LLMAgent,
        LlmAgentConfig as LLMAgentConfig, ModelConfig, ModelInfo, WorkflowAgent,
        WorkflowAgentConfig,
    };
}

/// The Python `openjiuwen.core.controller` package: events, intents, tasks,
/// executors, and the turn loop that routes them.
///
/// This is a module glob of `openjiuwen_agent::controller` (the controller stack
/// is hosted by `openjiuwen-agent`), so nested paths such as
/// [`schema`](controller::schema) and [`executors`](controller::executors) remain
/// reachable alongside the flattened
/// [`ControllerAgent`](controller::ControllerAgent),
/// [`Task`](controller::Task), and [`TaskScheduler`](controller::TaskScheduler)
/// re-exports.
pub mod controller {
    pub use openjiuwen_agent::controller::*;
}

/// The Python `openjiuwen.core.kv_cache` package: cache identity/runtime types,
/// the affinity hook, and lineage metadata.
///
/// This is a module glob of `openjiuwen_agent::kv_cache`, so nested paths such as
/// [`runtime`](kv_cache::runtime) and [`types`](kv_cache::types) are reachable
/// alongside the flattened re-exports.
pub mod kv_cache {
    pub use openjiuwen_agent::kv_cache::*;
}

/// The Python `openjiuwen.core.operator` package: tunable parameter handles for
/// LLM, tool, memory, skill, and context-evolution call paths.
///
/// This is a module glob of `openjiuwen_agent::operator`, so nested paths such as
/// [`base`](operator::base) are reachable alongside the flattened operator
/// re-exports.
pub mod operator {
    pub use openjiuwen_agent::operator::*;
}
