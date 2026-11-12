//! Agent runtime: the [`Agent`] seam, the `ReAct` loop, abilities, rails,
//! human-in-the-loop interrupts, and the application agents.
//!
//! Maps the Python packages `openjiuwen.core.single_agent` (`base.py`,
//! `agents/react_agent.py`, `ability_manager.py`, `rail/`, `interrupt/`,
//! `schema/`) and `openjiuwen.core.application` (`llm_agent/`,
//! `workflow_agent/`) onto idiomatic Rust.
//!
//! The crate owns the agent control flow and nothing else: tools and models are
//! injected by the caller (the runner passes them through the ability manager
//! and `set_model`), sessions come from `openjiuwen-session`, conversation
//! windows from `openjiuwen-context`, and workflows from `openjiuwen-workflow`.
//!
//! Modules:
//!
//! - [`agent`] — the [`Agent`] trait every runnable agent implements.
//! - [`card`] — [`AgentCard`], the registry-safe identity of an agent.
//! - [`input`] — [`AgentInput`] parsing of the caller's JSON payload.
//! - [`output`] — [`AgentOutput`], [`AgentChunk`], and the interrupt frame model.
//! - [`prompt`] — [`SystemPromptBuilder`] and its ordered [`PromptSection`]s.
//! - [`run`] — [`resolve_session`] and [`AgentRun`] session ownership handling.
//! - [`interrupt`] — interrupt requests, answers, approval parsing, and the
//!   persisted interruption state used to resume a paused turn.
//! - [`abilities`] — tool mounting and execution.
//! - [`rails`] — model/tool/loop hooks.
//! - [`react`] — the `ReActAgent` loop.
//! - [`llm_agent`] — `LLMAgent`, a configured `ReActAgent` composition.
//! - [`workflow_agent`] — `WorkflowAgent` over one or more workflows.
#![forbid(unsafe_code)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]
pub mod abilities;
pub mod agent;
pub mod card;
pub mod input;
pub mod interrupt;
pub mod llm_agent;
pub mod output;
pub mod prompt;
pub mod rails;
pub mod react;
pub mod run;
pub mod workflow_agent;
pub use agent::Agent;
pub use card::AgentCard;
pub use input::{AgentInput, InputQuery};
pub use output::{AgentChunk, AgentOutput, Observation, OutputFrame, ResultType};
pub use prompt::{PromptSection, SystemPromptBuilder};
pub use run::{resolve_session, AgentRun};
/// Returns the crate name as reported by `cargo`.
#[must_use]
pub const fn crate_name() -> &'static str {
    "openjiuwen-agent"
}
