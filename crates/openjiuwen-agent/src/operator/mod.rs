//! Tunable operators: parameter handles for LLM calls, tool calls, memory
//! calls, skill selection and experience staging, and context evolution.
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
//! - [`protocol`] — the shared evolution literals plus update normalisation and
//!   result summaries.
//! - [`llm_call`] — [`LLMCallOperator`]: system/user prompts, model name,
//!   temperature, plus the real prompt-rendering call path.
//! - [`tool_call`] — [`ToolCallOperator`]: tool descriptions projected onto
//!   [`openjiuwen_foundation::ToolCard`]s.
//! - [`memory_call`] — [`MemoryCallOperator`]: recall knobs, a bounded retry
//!   budget, and the [`MemoryRecall`] seam over `openjiuwen_memory::LongTermMemory`.
//! - [`skill_call`] — [`SkillCallOperator`]: curated skill names and a show cap.
//! - [`skill_experience`] — [`SkillExperienceOperator`]: preview-only staging of
//!   skill-experience records.
//! - [`staging`] — [`PendingChangeStore`] and the caller-owned pending-change
//!   lifecycle (approve/reject/retry) over a [`SkillExperienceStore`] sink.
//! - [`legacy`] — [`LLMCall`]: the legacy single-call wrapper with an optimizer
//!   trace callback.
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
pub mod legacy;
pub mod llm_call;
pub mod memory_call;
pub mod protocol;
pub mod skill_call;
pub mod skill_experience;
pub mod staging;
pub mod tool_call;

pub use base::{
    apply_merge, ApplyResult, Operator, ParameterCallback, PreviewableOperator, TunableKind,
    TunableSpec, UpdateEffect, UpdateMode, UpdateValue,
};
pub use context_evolve::{
    ContextEvolveOperator, MetisContextEvolveOperator, UpdatePolicy, MAX_MESSAGES,
    METIS_SUMMARY_HEADER,
};
pub use legacy::{LLMCall, OptimizerCallback, OptimizerTrace, OptimizerTraceFuture};
pub use llm_call::{LLMCallOperator, DEFAULT_USER_PROMPT, QUERY_PLACEHOLDER};
pub use memory_call::{
    MemoryCallOperator, MemoryRecall, DEFAULT_MAX_RETRIES, DEFAULT_THRESHOLD, DEFAULT_TOP_K,
    MAX_RETRIES, MAX_TOP_K,
};
pub use protocol::{
    normalize_update_value, normalize_updates, summarize_apply_results, update_key, ApplySummary,
    UpdateKey, APPEND_MODE, EXPERIENCES_TARGET, EXPERIENCE_ENTRY, LOCAL_APPLY_COMPLETED,
    MERGE_MODE, PENDING_CHANGE_EFFECT, REPLACE_MODE, SKILL_EXPERIENCE_ENTRY, STATE_EFFECT,
    TASK_MEMORY_ENTRY, TASK_MEMORY_TARGET,
};
pub use skill_call::{SkillCallOperator, DEFAULT_MAX_SKILLS, MAX_MAX_SKILLS};
pub use skill_experience::SkillExperienceOperator;
pub use staging::{
    build_local_apply_preview, InMemorySkillExperienceStore, LocalApplyPreview, PendingChange,
    PendingChangeOutcome, PendingChangeStatus, PendingChangeStore, SkillExperienceStore,
};
pub use tool_call::{ToolCallOperator, TOOL_DESCRIPTION_PREFIX, TOOL_DESCRIPTION_TARGET};
