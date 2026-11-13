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
//! Around that core the crate also ships the process-level infrastructure
//! ([`runtime`]: command seam, git, CI gates, Python toolchain, subprocess
//! evaluation), run usage accounting ([`ledger`]), the evaluation package
//! ([`harness_rsi::evaluator`]: case records, judge scoring, the SWE-bench and
//! terminal-bench runtimes, metrics), team-member optimisation
//! ([`harness_rsi::member_optimizer`]), the single-harness promotion loops
//! ([`harness_rsi::single_harness`]), progress events
//! ([`harness_rsi::progress`]), and paper optimisation
//! ([`artifact_rsi::paper_opt`]).
//!
//! What genuinely needs a Python ML runtime or Python import machinery stays in
//! [`deferred`] as API-shaped stubs returning `Error::Unsupported`: the
//! `agentdescent` evolution engine, the LLM-as-judge grader, and the runtime
//! extension loader.
//!
//! ## Module map
//!
//! | Module | Contents |
//! |---|---|
//! | [`evolution`] | `Candidate`, `MutationStrategy`, `Evaluator`, `SelectionStrategy`, `ElitistSelection`, `DeterministicMutator` |
//! | [`harness_rsi::auto_harness`] | `AutoHarnessOrchestrator`, `AutoHarnessConfig`, `Budget`, `SessionState`, `HarnessRunner`, `AutoHarnessEvent` |
//! | [`harness_rsi::single_harness`] | `IterativeSingleHarness`, `ContinuousSingleHarness`, `SingleHarnessConfig`, `SingleHarnessState`, the promotion rules |
//! | [`harness_rsi::evaluator`] | `Dataset`, `EvaluationCase`, `ScoringContract`, `CaseRunner`, `SweBenchEvaluator`, `TerminalBenchEvaluator`, `MetricsSummary` |
//! | [`harness_rsi::member_optimizer`] | `MemberOptimizer`, target selection, action planning, verification |
//! | [`harness_rsi::progress`] | `OptimizationProgressEvent`, `ProgressCallback`, `emit_progress` |
//! | [`artifact_rsi::program_opt`] | `ProgramOptConfig`, `PuctTree`, `Program`, `Node` |
//! | [`artifact_rsi::paper_opt`] | `PaperOptimizer`, `PaperOptConfig`, `PaperTree`, rubric scoring |
//! | [`runtime`] | `CommandRunner`, `GitOperations`, `CiGateRunner`, `PythonToolchain`, `SubprocessEvaluator` |
//! | [`ledger`] | `RunLedger`, `UsageRecord`, `UsageTotals`, `LedgerStore` backends |
//! | [`deferred`] | API-shaped stubs for the remaining Python-only surfaces |
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
pub mod ledger;
pub mod runtime;

pub use artifact_rsi::{
    paper_opt,
    paper_opt::PaperOptimizer,
    program_opt,
    program_opt::{
        Node, Program, ProgramOptConfig, PuctTree, ReplyFormat, DEFAULT_WORKERS, MAX_WORKERS,
    },
};
pub use deferred::{EvolutionEngine, EvolutionPolicy, LlmJudgeEvaluator, RuntimeExtensionLoader};
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
pub use harness_rsi::evaluator::{
    collect_metrics, parse_judge_verdict, score_verdict, write_summary, CaseExecutionBackend,
    CaseExecutionResult, CaseRunner, CaseTrace, Dataset, EvaluationCase, JudgeResult, JudgeVerdict,
    MetricsSummary, ScoringContract, SweBenchConfig, SweBenchEvaluator, TerminalBenchConfig,
    TerminalBenchEvaluator,
};
pub use harness_rsi::member_optimizer::{
    MemberOptimizationArtifact, MemberOptimizationRequest, MemberOptimizationStatus,
    MemberOptimizer, MemberOptimizerConfig, MemberTarget, OptimizationAction, OptimizationPlan,
    Severity,
};
pub use harness_rsi::single_harness::{
    ContinuousResult, ContinuousSingleHarness, IterativeResult, IterativeSingleHarness,
    PromotionPolicyMode, SingleHarnessConfig, SingleHarnessState, SingleHarnessStatus,
};
pub use ledger::{
    FileLedgerStore, InMemoryLedgerStore, LedgerMetrics, LedgerStore, ModelCall, ModelCallStatus,
    RunEvent, RunLedger, UsageRecord, UsageSnapshot, UsageTokens, UsageTotals,
};
pub use openjiuwen_common::error::{Error, ErrorKind, Result};
pub use runtime::{
    discover_python, parse_gates, CiGateReport, CiGateRunner, CiGateSpec, CommandOutcome,
    CommandRunner, CommandSpec, FakeCommandRunner, GateKind, GateOutcome, GitOperations, GitStatus,
    ProcessCommandRunner, PytestOutcome, PytestReport, PythonToolchain, ReportFormat,
    SubprocessConfig, SubprocessEvaluator, SubprocessResult,
};

/// Returns the crate name.
#[must_use]
pub fn crate_name() -> &'static str {
    "openjiuwen-rsi"
}
