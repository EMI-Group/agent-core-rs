//! Evaluation, optimisation, and training pipelines for openJiuwen.
//!
//! This crate ports the portable core of the Python package
//! `openjiuwen.agent_evolving` (see `docs/PARITY.md`, Tier 2):
//!
//! - [`dataset`] — `Case`, `EvaluatedCase`, `CaseLoader`, and JSON/JSONL
//!   dataset loading.
//! - [`evaluator`] — the [`evaluator::Evaluator`] seam, deterministic metrics
//!   (`ExactMatchMetric`, `MetricEvaluator`), the offline `CaseRunner` seam,
//!   and [`evaluator::EvaluationReport`].
//! - [`optimizer`] — the [`optimizer::Optimizer`] seam, candidate records, a
//!   deterministic grid-search baseline, best-so-far selection, evolve
//!   checkpoints, and the file/memory checkpoint stores and manager.
//! - [`trainer`] — the bounded train/evaluate loop with cancellation, budgets,
//!   callbacks, checkpoint resume, and per-epoch reports.
//! - [`rails`] — pure-data evolution events and an `EvolutionObserver` adapter
//!   that plugs into the harness `EvolutionRail` seam without an upward
//!   dependency from `openjiuwen-harness`.
//! - [`service`] — an axum skeleton of the Python `agent_rl.online.service`
//!   (health + deterministic evaluate); RL/RLOps endpoints are documented
//!   `Error::Unsupported` responses.
//! - [`rl`] — API-shaped records and stubs for the RL training stack.
//! - [`deferred`] — the machine-readable deferred-feature table.
//!
//! Everything outside the RL stubs is offline and deterministic: no model calls,
//! no environment reads, and no wall-clock dependence in tests.
//!
//! # Design notes
//!
//! The Python package has two independent `Case` models (`dev_tools.tune` and
//! `agent_evolving.dataset`); this crate mirrors the `agent_evolving` shape,
//! including the `per_metric` field on `EvaluatedCase`, and converts through
//! JSON if a caller needs the `dev_tools` projection.
//!
//! Agent execution enters through traits ([`evaluator::CaseRunner`],
//! [`trainer::TrainableTarget`]) so the evaluation and training loops are
//! testable with fake agents and never depend on a live model.
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

pub mod dataset;
pub mod deferred;
pub mod evaluator;
pub mod optimizer;
pub mod rails;
pub mod rl;
pub mod service;
pub mod trainer;

/// The crate name, useful in diagnostics and documentation.
pub const CRATE_NAME: &str = env!("CARGO_PKG_NAME");
/// The crate version.
pub const CRATE_VERSION: &str = env!("CARGO_PKG_VERSION");
