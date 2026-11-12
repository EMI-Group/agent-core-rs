//! Config, state, task/todo, stop conditions, interaction, loop events, and agent mode schemas.
//!
//! Pure data vocabulary of the harness: serde types, pure helpers and
//! Python-compatible JSON round-trips. No IO, no live handles, no harness-module
//! imports beyond the leaf references `Workspace`, `PermissionsSection` and
//! `PromptMode` (defined in `mode`, re-exported by `crate::prompts`).
//! See `CONTEXT.md` for the full public contract.

mod config;
mod interaction;
mod loop_event;
mod mode;
mod state;
mod stop_condition;
mod task;

pub use config::{
    resolve_inner_react_max_iterations, DeepAgentConfig, OnMaxIterations, SubAgentConfig,
    TaskLoopNoProgressGuardConfig,
};
pub use interaction::{
    ActiveInteractionRound, InputDispatchMode, InteractionEvent, InteractionEventType,
    InteractionPhase, RoundOutcome, RoundWorkItem, RoundWorkKind, SendInputRequest,
};
pub use loop_event::{create_loop_event, default_event_priority, DeepLoopEvent, DeepLoopEventType};
pub use mode::{AgentMode, PromptMode};
pub use state::{DeepAgentState, PlanModeState, SESSION_RUNTIME_ATTR, SESSION_STATE_KEY};
pub use stop_condition::{
    CompletionPromise, CustomPredicate, MaxRounds, NoProgressAnswer, StopConditionEvaluator,
    StopConditionState, StopEvaluationContext, StopVerdict, Timeout, TokenBudget,
};
pub use task::{status_icon, ModelUsageRecord, TaskPlan, TodoItem, TodoStatus};
