//! Agent specifications, the platform DSL, and the bridge into `openjiuwen-agent`.
//!
//! Ports `openjiuwen/dev_tools/agent_builder/` across seven files: programmatic
//! specs, the serialized LLM-agent DSL, the serialized workflow DSL, the bounded
//! dialogue history, the model-driven builder, its result vocabulary, and the
//! bridge into `openjiuwen-agent`.
//!
//! The Python builder runs a multi-stage state machine — intention detection,
//! an LLM clarifier, an agent/workflow designer, a DSL generator and reflector,
//! an LLM cycle checker, and Mermaid rendering. This port keeps Python's
//! observable response shape (`status`, `session_id`, `agent_type`, plus one of
//! `dsl`/`workflow`/`response`) but makes **one** model round trip per request:
//! the model answers with a single fenced JSON document, and any answer that
//! does not parse becomes a plain-text clarifying response (a documented
//! deviation).
//!
//! Module layout:
//!
//! - `dsl` — the serialized LLM-agent DSL ([`LlmAgentDsl`]) and its model,
//!   plugin, workflow, and dependency blocks.
//! - `workflow_dsl` — the serialized workflow document ([`WorkflowDsl`]) and its
//!   structural validation.
//! - `spec` — programmatic agent and workflow specifications
//!   ([`LlmAgentSpec`], [`WorkflowSpec`]) that convert into the DSL.
//! - `bridge` — the DSL-to-kernel bridge ([`AgentCard`], [`LlmAgentConfig`], and
//!   [`LlmAgent`]) implemented as inherent methods on [`LlmAgentDsl`].
//! - `history` — the bounded builder dialogue history ([`HistoryManager`]).
//! - `builder` — the session-based builder ([`AgentBuilder`]), the artifact enum
//!   ([`AgentType`]), and the `now_millis` clock entry point.
//! - `outcome` — the result and state vocabulary ([`BuildResult`],
//!   [`BuildStatus`], [`BuildOutput`], [`BuildState`], [`map_state_to_status`]).
//!
//! [`AgentCard`]: openjiuwen_agent::AgentCard
//! [`LlmAgentConfig`]: openjiuwen_agent::LlmAgentConfig
//! [`LlmAgent`]: openjiuwen_agent::LlmAgent

pub mod bridge;
pub mod builder;
pub mod dsl;
pub mod history;
pub mod outcome;
pub mod spec;
pub mod workflow_dsl;

pub use builder::{now_millis, AgentBuilder, AgentType};
pub use dsl::{
    DslConfigs, DslDependencies, DslModel, DslModelInfo, LlmAgentDsl, PluginRef, WorkflowRef,
};
pub use history::{DialogueMessage, HistoryManager, DEFAULT_MAX_HISTORY_SIZE};
pub use outcome::{map_state_to_status, BuildOutput, BuildResult, BuildState, BuildStatus};
pub use spec::{LlmAgentSpec, WorkflowSpec};
pub use workflow_dsl::{DslEdge, DslNode, WorkflowDsl};
