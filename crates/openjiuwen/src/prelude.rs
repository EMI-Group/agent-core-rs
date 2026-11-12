//! Curated convenience re-exports: `use openjiuwen::prelude::*;`.
//!
//! This module is convenience only. It introduces no new types and no behavior, and
//! carries no stability promise beyond the items it forwards; the canonical paths
//! (and their documentation) live in the concrete crates listed below.
//!
//! # Deviations from Python
//!
//! - There is no `InMemorySession` type in the Rust API: [`AgentSession`] (from
//!   `openjiuwen-session`) *is* the in-memory session implementation.

// --- errors -----------------------------------------------------------------
pub use openjiuwen_core::common::error::{Error, Result};
// --- single agents ----------------------------------------------------------
// The agent-construction surface: the `ReAct` loop and its config, the
// caller-facing input/output types, and the two application agents.
pub use openjiuwen_core::single_agent::{
    Agent, AgentCard, AgentChunk, AgentInput, AgentOutput, LlmAgent, ReActAgent, ReActAgentConfig,
    ResultType, WorkflowAgent,
};
// --- sessions ---------------------------------------------------------------
pub use openjiuwen_core::session::{AgentSession, Session};
// --- models, messages, tools -----------------------------------------------
pub use openjiuwen_core::foundation::{BaseMessage, FunctionTool, Model, Tool};
// --- graph / workflow / context --------------------------------------------
pub use openjiuwen_core::context_engine::ContextEngine;
pub use openjiuwen_core::graph::CompiledGraph;
pub use openjiuwen_core::workflow::Workflow;
// --- memory / runner --------------------------------------------------------
pub use openjiuwen_core::memory::LongTermMemory;
pub use openjiuwen_core::runner::Runner;
// --- harness ----------------------------------------------------------------
pub use openjiuwen_harness::{create_deep_agent, DeepAgent, DeepAgentBuilder, DeepAgentConfig};
pub use openjiuwen_harness_protocol::{HarnessCard, HarnessProtocol};
pub use openjiuwen_harness_providers::{create_harness, HarnessIOAdapter};
// --- agent teams ------------------------------------------------------------
// `TeamAgent` is defined in `openjiuwen_agent_teams::runtime::team_agent` and
// re-exported from both `agent_teams::runtime` and the crate root.
pub use openjiuwen_agent_teams::{TeamAgent, TeamAgentSpec};
// --- optional extension backends -------------------------------------------
#[cfg(feature = "a2a")]
pub use openjiuwen_extensions::a2a::client::A2aClient;
#[cfg(feature = "otel")]
pub use openjiuwen_extensions::otel::OtelTracerConfig;
