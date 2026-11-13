//! Self-evolution seam: the `ReActAgentEvolve` operator wrapper.
//!
//! Ports `openjiuwen/core/single_agent/agents/react_agent_evolve.py`: the same
//! `ReActAgent` execution path plus tunable operators whose parameter updates
//! are propagated back into the live agent configuration.
//!
//! [`ReActAgentEvolve`] owns an [`Arc<ReActAgent>`](crate::react::ReActAgent)
//! and implements [`Agent`](crate::agent::Agent) by delegation, so execution
//! logic is fully inherited. The operators it exposes are parameter handles,
//! not a second call path:
//!
//! - [`LLM_OPERATOR_ID`] (`"react_llm"`) — a
//!   [`LLMCallOperator`](crate::operator::LLMCallOperator) whose
//!   `system_prompt` is the agent's `prompt_template`; a change rewrites the
//!   configuration through [`ReActAgent::configure`](crate::react::ReActAgent)
//!   without touching the other configuration fields.
//! - [`TOOL_OPERATOR_ID`] (`"react_tool"`) — a
//!   [`ToolCallOperator`](crate::operator::ToolCallOperator) whose
//!   `tool_description` map rewrites the mounted tools' cards. Because
//!   `ReActAgent` has no card mutation API, each changed tool is re-wrapped in
//!   a [`DescriptionOverrideTool`] and mounted back in its original position.
//!
//! Both operators are rebuilt by [`ReActAgentEvolve::configure`] and
//! [`ReActAgentEvolve::new`], mirroring Python's `_init_operators`.
pub mod agent;
pub mod description_override;

pub use agent::{
    ReActAgentEvolve, SharedLlmOperator, SharedOperator, SharedToolOperator, LLM_OPERATOR_ID,
    TOOL_OPERATOR_ID,
};
pub use description_override::DescriptionOverrideTool;
