//! Agent specifications, the platform DSL, and the bridge into `openjiuwen-agent`.
//!
//! Ports `openjiuwen/dev_tools/agent_builder/`: programmatic specs, the
//! serialized LLM-agent/workflow DSL, session history, and the model-driven
//! builder that either yields DSL or asks a clarifying question.
//!
//! Module layout:
//!
//! - `dsl` — the serialized LLM-agent DSL ([`LlmAgentDsl`]) and its model,
//!   plugin, workflow, and dependency blocks.
//! - `workflow_dsl` — the serialized workflow document ([`WorkflowDsl`]) and its
//!   structural validation.
//! - `history` — the bounded builder dialogue history ([`HistoryManager`]).
//! - `spec` — programmatic agent and workflow specifications
//!   ([`LlmAgentSpec`], [`WorkflowSpec`]) that convert into the DSL.
//! - `bridge` — the DSL-to-kernel bridge ([`AgentCard`], [`LlmAgentConfig`], and
//!   [`LlmAgent`]) implemented as inherent methods on [`LlmAgentDsl`].
//! - `builder` is added in a later step.
//!
//! [`AgentCard`]: openjiuwen_agent::AgentCard
//! [`LlmAgentConfig`]: openjiuwen_agent::LlmAgentConfig
//! [`LlmAgent`]: openjiuwen_agent::LlmAgent

pub mod bridge;
pub mod dsl;
pub mod history;
pub mod spec;
pub mod workflow_dsl;

pub use dsl::{
    DslConfigs, DslDependencies, DslModel, DslModelInfo, LlmAgentDsl, PluginRef, WorkflowRef,
};
pub use history::{DialogueMessage, HistoryManager, DEFAULT_MAX_HISTORY_SIZE};
pub use spec::{LlmAgentSpec, WorkflowSpec};
pub use workflow_dsl::{DslEdge, DslNode, WorkflowDsl};
