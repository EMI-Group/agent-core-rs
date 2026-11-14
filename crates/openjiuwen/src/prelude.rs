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
//!
//! # Scope
//!
//! Python's top-level `openjiuwen/__init__.py` exports only `__version__`, mirrored by
//! [`crate::VERSION`], so there is no further top-level import surface to forward here.
//! Feature-gated trees (`harness_providers::providers`, `agent_teams::observability`,
//! `extensions::store`, …) stay reachable by module path instead of through the prelude,
//! which keeps it feature-free and capped at the agent-construction entry points.
//! The newer kernel trees (`single_agent::{usage, memory_provider, policy, prompt_pack}`,
//! `context_engine::{usage, telemetry, policy, tokenizers}`,
//! `foundation::llm::{external_provider, responses}`, `runner::dreaming`) and the team
//! trees (`agent_teams::{memory, observability}`) are deliberately omitted for the same
//! reason: they are diagnostics and configuration surfaces rather than agent-construction
//! entry points, so they are reached by module path and the cap stays at the ~40
//! agent-construction items.

// --- errors -----------------------------------------------------------------
pub use openjiuwen_core::common::error::{Error, Result};
// --- single agents ----------------------------------------------------------
// The agent-construction surface: the `ReAct` loop and its config, the
// caller-facing input/output types, and the two application agents.
pub use openjiuwen_core::single_agent::{
    Agent, AgentCard, AgentChunk, AgentInput, AgentOutput, LlmAgent, Observation, ReActAgent,
    ReActAgentConfig, ResultType, WorkflowAgent,
};
// --- controller stack -------------------------------------------------------
// The turn-loop executor that routes tasks to agents and workflows.
pub use openjiuwen_core::controller::ControllerAgent;
// --- sessions ---------------------------------------------------------------
// The session lifecycle: the in-memory session plus the disk-backed product
// session controller.
pub use openjiuwen_core::session::{AgentSession, Session, SessionController};
// --- models, messages, tools -----------------------------------------------
pub use openjiuwen_core::foundation::{BaseMessage, FunctionTool, Model, Tool};
// --- graph / workflow / context --------------------------------------------
pub use openjiuwen_core::context_engine::ContextEngine;
pub use openjiuwen_core::graph::CompiledGraph;
pub use openjiuwen_core::workflow::Workflow;
// --- memory / runner --------------------------------------------------------
// Memory plus the run orchestration surface: the process-global runner, the
// team runtime pool, and run observability.
pub use openjiuwen_core::memory::LongTermMemory;
pub use openjiuwen_core::runner::{RunMonitor, Runner, TeamPool};
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
