//! Agent specifications, build stages, and the bridge into `openjiuwen-agent`.
//!
//! Ports `openjiuwen/dev_tools/agent_builder/` across thirteen modules: the
//! programmatic specs, the serialized LLM-agent and workflow documents, the
//! bounded dialogue history, the staged builder pipeline (requirement clarifier,
//! workflow designer, DL generator, DL reflector, cycle checker, Mermaid
//! rendering), the session-based builder, its result vocabulary, and the kernel
//! bridge.
//!
//! Module layout:
//!
//! - `bridge` — the DSL-to-kernel bridge ([`AgentCard`], [`LlmAgentConfig`], and
//!   [`LlmAgent`]) implemented as inherent methods on [`LlmAgentDsl`].
//! - `builder` — the session-based builder ([`AgentBuilder`]), the artifact enum
//!   ([`AgentType`]), and the `now_millis` clock entry point.
//! - `clarifier` — the requirement clarifier ([`Clarifier`]) deciding whether a
//!   request is complete enough to design from.
//! - `cycle_checker` — cycle detection over a workflow document
//!   ([`find_cycle`], [`has_cycle`]).
//! - `designer` — the workflow designer ([`WorkflowDesigner`]) producing a
//!   [`WorkflowDesign`] of [`DesignNode`]s and [`DesignEdge`]s.
//! - `dl_generator` — the designer-to-DSL generator ([`DlGenerator`]).
//! - `dl_reflector` — the rule-based document repairer ([`DlReflector`]) and its
//!   [`Reflection`] report.
//! - `dsl` — the serialized LLM-agent DSL ([`LlmAgentDsl`]) and its model,
//!   plugin, workflow, and dependency blocks.
//! - `history` — the bounded builder dialogue history ([`HistoryManager`]).
//! - `mermaid` — Mermaid rendering of designs and documents
//!   ([`render_design`], [`render_mermaid`]).
//! - `outcome` — the result and state vocabulary ([`BuildResult`],
//!   [`BuildStatus`], [`BuildOutput`], [`BuildState`], [`map_state_to_status`]).
//! - `spec` — programmatic agent and workflow specifications
//!   ([`LlmAgentSpec`], [`WorkflowSpec`]) that convert into the DSL.
//! - `workflow_dsl` — the serialized workflow document ([`WorkflowDsl`]) and its
//!   structural validation.
//!
//! # Pipeline
//!
//! [`AgentBuilder::build_agent`] and [`AgentBuilder::build_agent_at`] keep the
//! single-round-trip behavior: one fenced JSON answer completes the session, and
//! any other answer becomes a plain-text clarification.
//!
//! [`AgentBuilder::build_with_pipeline`] runs clarification -> (workflow:
//! designer -> generator -> reflector) or (llm agent: generation + deterministic
//! repair), falls back to the deterministic stage implementations on any
//! unusable model answer, and only errors on a blank query.
//!
//! [`AgentCard`]: openjiuwen_agent::AgentCard
//! [`LlmAgentConfig`]: openjiuwen_agent::LlmAgentConfig
//! [`LlmAgent`]: openjiuwen_agent::LlmAgent

pub mod bridge;
pub mod builder;
pub mod clarifier;
pub mod cycle_checker;
pub mod designer;
pub mod dl_generator;
pub mod dl_reflector;
pub mod dsl;
pub mod history;
pub mod mermaid;
pub mod outcome;
pub mod spec;
pub mod workflow_dsl;

pub use builder::{now_millis, AgentBuilder, AgentType};
pub use clarifier::{Clarification, Clarifier};
pub use cycle_checker::{find_cycle, has_cycle};
pub use designer::{DesignEdge, DesignNode, DesignNodeKind, WorkflowDesign, WorkflowDesigner};
pub use dl_generator::DlGenerator;
pub use dl_reflector::{DlReflector, Reflection, ReflectionIssue, ReflectionIssueKind};
pub use dsl::{
    DslConfigs, DslDependencies, DslModel, DslModelInfo, LlmAgentDsl, PluginRef, WorkflowRef,
};
pub use history::{DialogueMessage, HistoryManager, DEFAULT_MAX_HISTORY_SIZE};
pub use mermaid::{render_design, render_mermaid};
pub use outcome::{map_state_to_status, BuildOutput, BuildResult, BuildState, BuildStatus};
pub use spec::{LlmAgentSpec, WorkflowSpec};
pub use workflow_dsl::{DslEdge, DslNode, WorkflowDsl};
