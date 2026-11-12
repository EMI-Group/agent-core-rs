//! Bounded self-optimisation session loop.
//!
//! Ported from Python `openjiuwen.rsi.harness_rsi.auto_harness`. The loop
//! proposes candidate configurations, evaluates each through a harness run,
//! keeps the elite, checkpoints every iteration, and stops on convergence,
//! budget, or cancellation.
//!
//! Modules are layered so later pieces can be added without touching the existing
//! ones:
//!
//! - [`config`] — serialisable [`AutoHarnessConfig`] and [`PipelinePreference`];
//! - [`budget`] — wall-clock and cost accounting through [`Budget`];
//! - [`task`] — task, iteration, session, and stop-condition records;
//! - [`event`] — [`AutoHarnessEvent`], the `type`-tagged session event stream;
//! - [`runner`] — [`HarnessRunner`], the only seam to `openjiuwen-harness`, plus
//!   the [`HarnessEvaluator`] adapter over the shared evaluation protocol;
//! - [`checkpoint`] — [`SessionState`] and the [`CheckpointStore`] backends that
//!   make a session resumable;
//! - [`builder`] — [`AutoHarnessOrchestratorBuilder`], which validates the config
//!   and defaults every collaborator;
//! - [`orchestrator`] — [`AutoHarnessOrchestrator`], the session loop itself.
pub mod budget;
pub mod builder;
pub mod checkpoint;
pub mod config;
pub mod event;
pub mod orchestrator;
pub mod runner;
pub mod task;

pub use budget::{Budget, BudgetLimits, BudgetStatus};
pub use builder::AutoHarnessOrchestratorBuilder;
pub use checkpoint::{
    CheckpointStore, InMemoryCheckpointStore, KvCheckpointStore, SessionCheckpointStore,
    SessionState,
};
pub use config::{AutoHarnessConfig, PipelinePreference};
pub use event::AutoHarnessEvent;
pub use orchestrator::{AutoHarnessOrchestrator, DEFAULT_SEED_BODY, SEED_CANDIDATE_ID};
pub use runner::{HarnessEvaluator, HarnessRunOutcome, HarnessRunRequest, HarnessRunner};
pub use task::{
    Gap, IterationRecord, OptimizationTask, SessionSummary, StageResult, StageStatus,
    StopConditions, StopDecision, StopReason, TaskStatus,
};
