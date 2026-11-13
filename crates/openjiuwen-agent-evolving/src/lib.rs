//! Port of the offline core of Python `openjiuwen.agent_evolving`.
//!
//! This crate keeps the portable, non-ML parts of the Python area — case datasets,
//! deterministic evaluation and scoring, optimizer/checkpoint management, a bounded
//! trainer loop, evolution rails, and the RL service skeleton — and exposes
//! API-shaped stubs (`Error::Unsupported`) for the ML/vendor/RL stack, which is
//! documented as deferred in `docs/PARITY.md`.
//!
//! # Module map
//!
//! - [`dataset`] — `Case`, `EvaluatedCase`, `CaseLoader`, JSON/JSONL loading.
//! - [`evaluator`] — `Evaluator`/`CaseRunner` seams, metrics, reports, offline scoring.
//! - [`optimizer`] — candidates, grid-search baseline, checkpoints and stores.
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

/// Case datasets: `Case`, `EvaluatedCase`, `CaseLoader`, JSON/JSONL loading.
pub use dataset::{
    load_cases_from_path, parse_cases_json, parse_cases_jsonl, parse_cases_str, shuffle_cases,
    split_cases, Case, CaseFormat, CaseLoader, EvaluatedCase,
};
/// Evaluator/metric/runner seams, reports, and offline scoring.
pub use evaluator::{
    evaluate_cases, AgentCaseRunner, Aggregate, CaseRunner, DefaultEvaluator, EvaluationConfig,
    EvaluationReport, Evaluator, ExactMatchMetric, LlmAsJudgeMetric, MapCaseRunner, Metric,
    MetricEvaluator, MetricValue,
};
/// Candidates, the optimizer baseline, and checkpoints/stores/manager.
pub use optimizer::{
    select_best, BestSoFar, Candidate, CheckpointManager, CheckpointSnapshot, CheckpointStore,
    DefaultCheckpointManager, EvolveCheckpoint, FileCheckpointStore, GridSearchOptimizer,
    LlmInvokePolicy, MemoryCheckpointStore, Optimizer, ParameterSpace, RestoredState,
};
/// Pure evolution events, observer trait, and harness rail adapter.
pub use rails::{
    registerable_rail, EvolutionEvent, EvolutionObserver, EvolutionRailSpec, HarnessEvolutionRail,
    EVOLUTION_PRIORITY,
};
/// Axum RL service skeleton (health/evaluate, 501 RL endpoints).
pub use service::{
    build_router, serve, ApiError, EvaluateRequest, EvaluateResponse, RlServiceConfig,
};
/// Bounded train/evaluate loop, progress, callbacks, and reports.
pub use trainer::{
    Callbacks, EpochReport, NoopCallbacks, Progress, StopReason, TrainableTarget, Trainer,
    TrainerConfig, TrainerReport,
};
