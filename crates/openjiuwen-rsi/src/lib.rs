//! Recursive self-improvement (RSI) for the openJiuwen runtime.
//!
//! Maps to the Python package `openjiuwen.rsi` in the openJiuwen reference
//! implementation. The crate owns three things:
//!
//! - [`harness_rsi::auto_harness`] — a bounded, resumable self-optimisation
//!   session loop over candidate configurations.
//! - [`artifact_rsi::program_opt`] — the portable `PUCT` tree search ranking
//!   candidate programs.
//! - [`evolution`] — the shared candidate / mutation / evaluation / selection
//!   protocol used by both halves.
//!
//! Surfaces that need a Python ML runtime, a vendor harness SDK, subprocess or
//! git execution, or distributed evolution live in [`deferred`] as
//! API-shaped stubs returning `Error::Unsupported`.
//!
//! ## Module map
//!
//! | Module | Contents |
//! |---|---|
//! | [`evolution`] | `Candidate`, `MutationStrategy`, `Evaluator`, `SelectionStrategy`, `ElitistSelection`, `DeterministicMutator` |
//! | [`harness_rsi::auto_harness`] | `AutoHarnessOrchestrator`, `AutoHarnessConfig`, `Budget`, `SessionState`, `HarnessRunner`, `AutoHarnessEvent` |
//! | [`artifact_rsi::program_opt`] | `ProgramOptConfig`, `PuctTree`, `Program`, `Node` |
//! | [`deferred`] | API-shaped stubs for the ML/vendor surfaces |
//!
//! ## Flat surface
//!
//! Python's `openjiuwen.rsi` exposes its protocol and session types through the
//! package root, so every public item is re-exported flat from here as well:
//! `openjiuwen_rsi::AutoHarnessOrchestrator`, `openjiuwen_rsi::PuctTree`,
//! `openjiuwen_rsi::Candidate`, `openjiuwen_rsi::Error`, … The module paths above
//! remain the canonical ones.
#![forbid(unsafe_code)]
// Type names deliberately repeat their module name to mirror the Python `rsi` module
// paths (`auto_harness::AutoHarnessConfig`, `program_opt::ProgramOptConfig`, …).
#![allow(clippy::module_name_repetitions)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]

pub mod artifact_rsi;
pub mod deferred;
pub mod evolution;
pub mod harness_rsi;

pub use artifact_rsi::{
    program_opt,
    program_opt::{
        Node, Program, ProgramOptConfig, PuctTree, ReplyFormat, DEFAULT_WORKERS, MAX_WORKERS,
    },
};
pub use deferred::{
    CiGate, EvolutionEngine, EvolutionPolicy, GitBackend, LlmJudgeEvaluator, MemberOptimizer,
    PaperOptimizer, RuntimeExtensionLoader, SubprocessEvaluator, SweBenchEvaluator,
    TerminalBenchEvaluator,
};
pub use evolution::{
    rank_score, ranked, Candidate, CandidateId, CandidateStatus, Case, CaseResult,
    DeterministicMutator, ElitistSelection, EvaluationReport, Evaluator, MutationRequest,
    MutationStrategy, SelectionStrategy,
};
pub use harness_rsi::auto_harness;
pub use harness_rsi::auto_harness::{
    AutoHarnessConfig, AutoHarnessEvent, AutoHarnessOrchestrator, AutoHarnessOrchestratorBuilder,
    Budget, BudgetLimits, BudgetStatus, CheckpointStore, Gap, HarnessEvaluator, HarnessRunOutcome,
    HarnessRunRequest, HarnessRunner, InMemoryCheckpointStore, IterationRecord, KvCheckpointStore,
    OptimizationTask, PipelinePreference, SessionCheckpointStore, SessionState, SessionSummary,
    StageResult, StageStatus, StopConditions, StopDecision, StopReason, TaskStatus,
    DEFAULT_SEED_BODY, SEED_CANDIDATE_ID,
};
pub use openjiuwen_common::error::{Error, ErrorKind, Result};

/// Returns the crate name.
#[must_use]
pub fn crate_name() -> &'static str {
    "openjiuwen-rsi"
}
