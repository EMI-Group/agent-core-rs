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
//! # Seams
//!
//! - [`Agent`] — the trait every runnable agent implements ([`ReActAgent`],
//!   [`LlmAgent`], [`WorkflowAgent`]); it is what `openjiuwen-runner` drives.
//! - [`AgentOutput`] / [`AgentChunk`] — the invoke result and the streaming
//!   frame, both serialized with the Python wire names (`result_type`,
//!   `workflow_execution_state`, `type`).
//! - [`AgentInput`] — the caller's JSON payload, including interactive resume
//!   input.
//! - [`AbilityManager`] — the per-agent tool registry plus tool execution.
//! - [`AgentRail`] / [`RailContext`] / [`RailRegistry`] — the hook seam around
//!   invoke, model, tool, and iteration steps.
//! - [`InterruptRequest`] / [`InterruptionState`] — a paused run is an output,
//!   not an error; resuming is a normal invoke with interactive input.
//!
//! # Modules
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
//! - [`rails`] — model/tool/loop hooks and the rails the kernel hosts itself.
//! - [`react`] — the [`ReActAgent`] loop and its configuration.
//! - [`llm_agent`] — [`LlmAgent`], a configured `ReActAgent` composition.
//! - [`memory_provider`] — the external long-term-memory provider bridge: a rail
//!   and the provider's model-facing tools.
//! - [`workflow_agent`] — [`WorkflowAgent`] over one or more workflows.
//! - [`usage`] — the context usage telemetry seam: analyzer/recorder ownership,
//!   KV-cache scope keys, and the drained [`UsageTelemetry`] buffer.
#![forbid(unsafe_code)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]
pub mod abilities;
pub mod agent;
pub mod callbacks;
pub mod card;
pub mod controller;
pub mod evolve;
pub mod input;
pub mod interrupt;
pub mod kv_cache;
pub mod llm_agent;
pub mod memory_provider;
pub mod operator;
pub mod output;
pub mod policy;
pub mod prompt;
pub mod prompt_pack;
pub mod rails;
pub mod react;
pub mod run;
pub mod skills;
pub mod steering;
pub mod usage;
pub mod workflow_agent;
pub use abilities::{
    parse_tool_arguments, AbilityManager, AddAbilityResult, ToolCallResult, ToolExecutionOptions,
};
pub use agent::Agent;
pub use card::AgentCard;
pub use input::{AgentInput, InputQuery};
pub use interrupt::{
    approval_value, match_answers, parse_approval, reject_feedback, InterruptAnswer, InterruptKind,
    InterruptRequest, InterruptionState, MatchedAnswers, AUTO_CONFIRM_STATE_KEY,
    DEFAULT_REJECT_FEEDBACK, INTERRUPTION_STATE_KEY,
};
pub use llm_agent::{
    create_llm_agent_config, ConstrainConfig, LlmAgent, LlmAgentConfig, ModelConfig, ModelInfo,
};
pub use memory_provider::{provider_tool_cards, provider_tools, ProviderMemoryRail, ProviderTool};
pub use output::{AgentChunk, AgentOutput, Observation, OutputFrame, ResultType};
pub use prompt::{PromptSection, SystemPromptBuilder};
pub use rails::{
    AgentRail, AskUserRail, BaseInterruptRail, ConfirmInterruptRail, GuardrailRail, MemoryRail,
    ModelBackupRail, RailContext, RailEvent, RailRegistry, RetryRequest, StageFlags, ToolFilter,
    DEFAULT_RAIL_PRIORITY,
};
pub use react::{MaxIterationsPolicy, ReActAgent, ReActAgentConfig};
pub use run::{resolve_session, AgentRun};
pub use usage::{default_usage_analyzer, UsageTelemetry};
pub use workflow_agent::{DefaultResponse, WorkflowAgent, WorkflowAgentConfig};
/// Returns the crate name as reported by `cargo`.
#[must_use]
pub const fn crate_name() -> &'static str {
    "openjiuwen-agent"
}
