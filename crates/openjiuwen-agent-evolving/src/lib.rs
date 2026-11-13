//! Port of the offline core of Python `openjiuwen.agent_evolving`.
//!
//! This crate covers the portable, non-ML parts of the Python area: case datasets,
//! deterministic evaluation and scoring, the LLM-as-judge evaluator, optimizer and
//! checkpoint management, the bounded trainer loop, the research optimizers, the
//! evolution subsystems (signals, rewards, attribution, trajectories, experience,
//! sharing, TTSE), the Docker + git evaluation harness, the RL batching/training
//! seams, evolution rails, and the RL service skeleton.
//!
//! Only the Python-only ML/vendor stack keeps typed stubs: GPU/distributed RL
//! engines (`torch`/`verl`/`ray`), the Redis/vLLM online gateway, vendor reward and
//! judge services, and the remote sharing hub. Those paths return
//! [`openjiuwen_common::error::Error::Unsupported`] with a reason and are listed in
//! `docs/PARITY.md` and [`deferred`].
//!
//! # Module map
//!
//! - [`dataset`] — `Case`, `EvaluatedCase`, `CaseLoader`, JSON/JSONL loading.
//! - [`evaluator`] — evaluator/metric/runner seams, judge rubric + verdict parsing, reports.
//! - [`optimizer`] — candidates, grid-search baseline, research optimizers, checkpoints.
//! - [`trainer`] — bounded train/evaluate loop, progress, callbacks, reports.
//! - [`rails`] — pure evolution events, observer trait, harness rail adapter.
//! - [`service`] — axum RL service skeleton (health/evaluate, RL endpoints).
//! - [`rl`] — RL config, batch/rollout bookkeeping and the pluggable trainer loop.
//! - [`signal`] — the shared evolution-signal vocabulary and tracker.
//! - [`reward`] — reward records, aggregation and normalisation.
//! - [`attribution`] — dependency graph, credit assignment and keyword detection.
//! - [`trajectory`] — run recording, grouping, windows and digests.
//! - [`experience`] — experience records, archive, dedupe and extraction.
//! - [`sharing`] — shared-experience packaging and the backend seam.
//! - [`ttse`] — test-time skill experience: tip bank, induction, consultation.
//! - [`evaluator_pipeline`] — Docker + git evaluation harness over a command seam.
//! - [`deferred`] — deferred-feature table and `Error::Unsupported` helper.
//!
//! Everything except the stubs is offline and deterministic: no model calls, no
//! environment reads, and no wall-clock dependence in tests.

#![forbid(unsafe_code)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]
#![allow(
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::module_name_repetitions,
    clippy::must_use_candidate,
    clippy::too_many_arguments,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::items_after_statements,
    clippy::unused_async,
    clippy::significant_drop_tightening,
    clippy::manual_let_else,
    clippy::needless_pass_by_value,
    clippy::missing_const_for_fn
)]

/// Case datasets: `Case`, `EvaluatedCase`, `CaseLoader`, JSON/JSONL loading.
pub mod dataset;
/// Deferred-feature table and `Error::Unsupported` helper.
pub mod deferred;
/// Evaluator/metric/runner seams, reports, and offline scoring.
pub mod evaluator;
/// Docker + git evaluation harness over an injectable command runner.
pub mod evaluator_pipeline;
/// Candidates, the optimizer baseline, and checkpoints/stores/manager.
pub mod optimizer;
/// Pure evolution events, observer trait, and harness rail adapter.
pub mod rails;
/// RL records and API-shaped ML stubs.
pub mod rl;
/// Axum RL service skeleton (health/evaluate, RL endpoints).
pub mod service;
/// Bounded train/evaluate loop, progress, callbacks, and reports.
pub mod trainer;

/// Diagnostic signals, dependency graph and credit assignment.
pub mod attribution;
/// Experience records, archive, dedupe and extraction.
pub mod experience;
/// Reward records, aggregation and normalisation.
pub mod reward;
/// Shared-experience packaging and the backend seam.
pub mod sharing;
/// The shared evolution-signal vocabulary and tracker.
pub mod signal;
/// Run recording, grouping, windows and digests.
pub mod trajectory;
/// Test-time skill experience: tip bank, induction, consultation.
pub mod ttse;

/// The crate name as reported by Cargo.
pub const CRATE_NAME: &str = env!("CARGO_PKG_NAME");
/// The crate version as reported by Cargo.
pub const CRATE_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Diagnostic signals, attribution graphs, credit assignment, and keyword rules.
pub use attribution::{
    reward_tier, AttributionEdge, AttributionGraph, AttributionNode, CreditAssigner, CreditRank,
    CreditReport, DiagnosticSignal, KeywordAttributor, KeywordMatch, KeywordRule,
    RewardStageAdapter, SignalSource,
};
/// Case datasets: `Case`, `EvaluatedCase`, `CaseLoader`, JSON/JSONL loading.
pub use dataset::{
    load_cases_from_path, parse_cases_json, parse_cases_jsonl, parse_cases_str, shuffle_cases,
    split_cases, Case, CaseFormat, CaseLoader, EvaluatedCase,
};
/// Evaluator/metric/runner seams, judge prompts, and offline scoring.
pub use evaluator::{
    build_judge_messages, evaluate_cases, AgentCaseRunner, Aggregate, CaseRunner, DefaultEvaluator,
    EvaluationConfig, EvaluationReport, Evaluator, ExactMatchMetric, JudgeCriterion, JudgeRubric,
    JudgeVerdict, LlmAsJudgeMetric, LlmJudge, LlmJudgeEvaluator, MapCaseRunner, Metric,
    MetricEvaluator, MetricValue,
};
/// Docker + git evaluation harness over an injectable command runner.
pub use evaluator_pipeline::{
    container_name, find_program, parse_last_json, parse_score, parse_test_counts, CommandRunner,
    CommandSpec, DockerEnvironment, EvaluationPipeline, EvaluationTask, ExecResult,
    FakeCommandRunner, GitWorkspaceSpec, PipelineConfig, PipelineSummary, PreparedWorkspace,
    ProcessCommandRunner, ScoreSpec, ScoredRun, TaskReport, TestCounts, WorkspacePreparer,
};
/// Experience records, archives, extraction, and lifecycle rules.
pub use experience::{
    ExperienceArchive, ExperienceExtractor, ExperienceLifecycle, ExperienceMatch, ExperienceQuery,
    ExperienceRecord, ExperienceSection, ExperienceStats, LifecycleConfig, LifecycleReport,
};
/// Candidates, the optimizer baseline, research optimizers, and checkpoints.
pub use optimizer::{
    parse_tagged, select_best, BestSoFar, Candidate, CheckpointManager, CheckpointSnapshot,
    CheckpointStore, DefaultCheckpointManager, EvolveCheckpoint, FileCheckpointStore,
    GridSearchOptimizer, InstructionOptimizer, LlmInvokePolicy, MemoryCheckpointStore,
    MemoryOptimizer, MetisOptimizer, OperatorKind, OperatorState, Optimizer, ParameterSpace,
    ProposalRecord, RestoredState, SkillExperienceOptimizer, TextualGradientOptimizer,
    TextualParameter, TextualStrategy, ToolOptimizer, TtseOptimizer,
};
/// Pure evolution events, observer trait, and harness rail adapter.
pub use rails::{
    registerable_rail, EvolutionEvent, EvolutionObserver, EvolutionRailSpec, HarnessEvolutionRail,
    EVOLUTION_PRIORITY,
};
/// Reward records, aggregation, normalisation, and reward packs.
pub use reward::{
    aggregate_scores, default_pack, reward_from_score, score_evaluated_cases,
    MissingDimensionPolicy, RewardAggregator, RewardDimension, RewardPack, RewardSignal,
    RewardSummary, ScoreNormalisation, ScoreStats,
};
/// RL batches, rollouts, policy seams, checkpoints, and the trainer loop.
pub use rl::{
    token_estimate, BatchPacker, DeterministicPolicy, FakeRolloutEngine, Policy, RewardFn,
    RewardNormalisation, RewardNormaliser, RewardRegistry, RlBatch, RlCheckpoint, RlEpochReport,
    RlSample, RlStopReason, RlTask, RlTrainReport, RlTrainer, RlTrainerConfig, RlUpdate,
    RolloutEngine, RolloutLog,
};
/// Axum RL service skeleton (health/evaluate, 501 RL endpoints).
pub use service::{
    build_router, serve, ApiError, EvaluateRequest, EvaluateResponse, RlServiceConfig,
};
/// Shared-experience packaging, keyword extraction, and the backend seam.
pub use sharing::{
    KeywordExtractor, LocalFileSharingBackend, MemorySharingBackend, QueryKeywords, ShareStager,
    SharedExperience, SharedSkillBundle, SharingBackend, SharingMeta, SkillPackageMeta,
    SkillSearchResult, StagedBundle,
};
/// Evolution signals, feedback extraction, and the signal tracker.
pub use signal::{
    signals_from_conversation, signals_from_evaluated_cases, signals_from_failure,
    signals_from_review_feedback, signals_from_team_feedback, EvolutionCategory, EvolutionSignal,
    EvolutionTarget, SignalTracker, TrackedSignal,
};
/// Bounded train/evaluate loop, progress, callbacks, and reports.
pub use trainer::{
    Callbacks, EpochReport, NoopCallbacks, Progress, StopReason, TrainableTarget, Trainer,
    TrainerConfig, TrainerReport,
};
/// Trajectory recording, spans, windows, grouping, and digests.
pub use trajectory::{
    first_error, from_messages, last_n, longest_span, render_digest, span_by_name, spans,
    window_around, windows, StepKind, StepStatus, Trajectory, TrajectoryBuilder, TrajectoryDigest,
    TrajectoryGroupSummary, TrajectoryOutcome, TrajectorySpan, TrajectoryStats, TrajectoryStep,
    TrajectoryStore, TrajectoryWindow,
};
/// Test-time skill experience: banks, induction, dreaming, and consultation.
pub use ttse::{
    consult, render_tips, AddOutcome, ClassificationRule, ConsultResult, DreamReport, Dreamer,
    SuccessAssessment, SuccessDetector, TipBank, TipBankStats, TipCategory, TipClassifier,
    TipInducer, TipParser, TipRecord, TtseConfig, TtseEngine,
};
