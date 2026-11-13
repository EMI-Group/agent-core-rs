//! Team-member optimisation: target selection, action planning, execution
//! waves, and verification.
//!
//! This module ports the portable core of Python's
//! `openjiuwen.rsi.harness_rsi.member_optimizer` package — one
//! plan → execute → verify → publish pass over a team member's harness:
//!
//! | Python module | Rust module |
//! |---|---|
//! | `member_selector.py` (deterministic formula) | [`selection`] |
//! | `schema.py` plan/action/verification types | [`plan`], [`verify`] |
//! | `action_groups/scheduling.py` (dependency waves) | [`plan::plan_waves`] |
//! | `action_planner.py` (LLM planner) | [`plan::ActionPlanner`] seam + [`plan::DeterministicActionPlanner`] |
//! | `action_executor.py` (one `DeepAgent` per action) | [`optimizer::ActionExecutor`] seam + [`optimizer::DeterministicActionExecutor`] |
//! | `verification.py` (worktree + LLM repair) | [`verify::ActionVerifier`] seam + [`verify::StaticActionVerifier`] |
//! | `optimizer.py` (the pass) | [`optimizer::MemberOptimizer::optimize`] |
//!
//! The LLM attributors, the planner agent, the per-action executor, and the
//! repair agent all stay behind traits or caller-supplied data: nothing here
//! builds a model, spawns a subprocess, or touches git.
pub mod config;
pub mod optimizer;
pub mod plan;
pub mod selection;
pub mod verify;
pub use config::{
    MemberOptimizerConfig, DEFAULT_MAX_ACTIONS_PER_PLAN, DEFAULT_MAX_REPAIR_ROUNDS,
    DEFAULT_MAX_ROLES_PER_RUN,
};
pub use optimizer::{
    ActionExecutor, DeterministicActionExecutor, ExecutionRequest, MemberOptimizationArtifact,
    MemberOptimizationRequest, MemberOptimizationStatus, MemberOptimizer, PromotionStatus,
    EXECUTOR_NAME,
};
pub use plan::{
    plan_waves, ActionDefinition, ActionPlanner, DeterministicActionPlanner, OptimizationAction,
    OptimizationPlan, PlanRequest, RunIf, ACTION_PLANNER_NAME,
};
pub use selection::{
    rank_targets, select_targets, selection_score, AttributionInput, AttributionResult,
    IssueObservation, MemberTarget, SelectedTarget, Severity, SynthesizedAttributor,
    UnassignedIssue, MEMBER_METADATA_KEY, ROLE_METADATA_KEY,
};
pub use verify::{
    ActionOutcome, ActionVerifier, CheckStatus, FsWorkspaceProbe, InMemoryWorkspaceProbe,
    MemberVerificationResult, RoleVerification, StaticActionVerifier, VerificationCheck,
    VerificationRequest, VerificationStatus, WorkspaceProbe, CHECK_ACTION_SUCCEEDED,
    CHECK_DECLARED_WRITE_PATHS, CHECK_NO_DECLARED_WRITE_PATHS, CHECK_TARGET_EXISTS,
};
