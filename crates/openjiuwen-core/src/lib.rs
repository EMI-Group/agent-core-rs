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
//! | `openjiuwen.core.multi_agent` | [`multi_agent`] → `openjiuwen_multiagent` |
//! | `openjiuwen.core.runner` | [`runner`] → `openjiuwen_runner` |
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
/// application agents ([`LlmAgent`], [`WorkflowAgent`]) live in [`application`]
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
