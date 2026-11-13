//! Tunable operators: parameter handles for LLM calls, tool calls, memory
//! calls, skill selection, and context evolution.
//!
//! Ports `openjiuwen/core/operator/`: every operator exposes typed tunables, a
//! parameter set/get surface, and a serializable state so an external optimiser
//! can read, update, and restore the live call path. `ReActAgentEvolve` builds
//! on this module, so the [`ParameterCallback`] seam is the contract: whenever
//! a parameter changes, the handle pushes `(target, value)` to the callback and
//! the consumer re-reads the affected configuration.
//!
//! # Modules
//!
//! - [`base`] — [`Operator`], [`PreviewableOperator`], [`TunableSpec`],
//!   [`UpdateValue`], [`ApplyResult`], and the [`apply_merge`] helper.
//! - [`llm_call`] — [`LLMCallOperator`]: system/user prompts, model name,
//!   temperature, plus the real prompt-rendering call path.
//! - [`tool_call`] — [`ToolCallOperator`]: tool descriptions projected onto
//!   [`openjiuwen_foundation::ToolCard`]s.
//! - [`memory_call`] — [`MemoryCallOperator`]: recall knobs and the
//!   [`MemoryRecall`] seam over `openjiuwen_memory::LongTermMemory`.
//! - [`skill_call`] — [`SkillCallOperator`]: curated skill names and a show cap.
//! - [`context_evolve`] — [`ContextEvolveOperator`] /
//!   [`MetisContextEvolveOperator`]: deterministic context-window policies.
//!
//! # Update entry points
//!
//! - [`Operator::set_parameter`] applies an evolution update; frozen tunables
//!   are skipped with [`ApplyResult::skipped`].
//! - [`Operator::apply_update`] applies a structured [`UpdateValue`] (replace,
//!   merge, or append); preview-only operators route it to
//!   [`PreviewableOperator::preview_update`] instead of mutating.
//! - [`Operator::load_state`] restores a checkpoint, bypassing freeze markers
//!   and firing the callback for every restored field.
pub mod base;
pub mod context_evolve;
pub mod llm_call;
pub mod memory_call;
pub mod skill_call;
pub mod tool_call;

pub use base::{
    apply_merge, ApplyResult, Operator, ParameterCallback, PreviewableOperator, TunableKind,
    TunableSpec, UpdateValue,
};
pub use context_evolve::{
    ContextEvolveOperator, MetisContextEvolveOperator, UpdatePolicy, MAX_MESSAGES,
    METIS_SUMMARY_HEADER,
};
pub use llm_call::{LLMCallOperator, DEFAULT_USER_PROMPT, QUERY_PLACEHOLDER};
pub use memory_call::{
    MemoryCallOperator, MemoryRecall, DEFAULT_THRESHOLD, DEFAULT_TOP_K, MAX_TOP_K,
};
pub use skill_call::{SkillCallOperator, DEFAULT_MAX_SKILLS, MAX_MAX_SKILLS};
pub use tool_call::{ToolCallOperator, TOOL_DESCRIPTION_PREFIX, TOOL_DESCRIPTION_TARGET};
