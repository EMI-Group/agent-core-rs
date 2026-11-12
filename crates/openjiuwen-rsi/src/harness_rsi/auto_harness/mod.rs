//! Bounded self-optimisation session loop.
//!
//! Ported from Python `openjiuwen.rsi.harness_rsi.auto_harness`. The loop proposes
//! candidate configurations, evaluates each through a harness run, keeps the
//! elite, checkpoints every iteration, and stops on convergence, budget, or
//! cancellation.
//!
//! Modules are layered so later pieces can be added without touching the existing
//! ones:
//!
//! - [`config`] — serialisable [`AutoHarnessConfig`] and [`PipelinePreference`];
//! - [`budget`] — wall-clock and cost accounting through [`Budget`];
//! - [`task`] — task, iteration, session, and stop-condition records.
pub mod budget;
pub mod config;
pub mod task;

pub use budget::{Budget, BudgetLimits, BudgetStatus};
pub use config::{AutoHarnessConfig, PipelinePreference};
pub use task::{
    Gap, IterationRecord, OptimizationTask, SessionSummary, StageResult, StageStatus,
    StopConditions, StopDecision, StopReason, TaskStatus,
};
