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
//! - `spec`, `bridge`, and `builder` are added in a later step.

pub mod dsl;
pub mod history;
pub mod workflow_dsl;

pub use dsl::{
    DslConfigs, DslDependencies, DslModel, DslModelInfo, LlmAgentDsl, PluginRef, WorkflowRef,
};
pub use history::{DialogueMessage, HistoryManager, DEFAULT_MAX_HISTORY_SIZE};
pub use workflow_dsl::{DslEdge, DslNode, WorkflowDsl};
