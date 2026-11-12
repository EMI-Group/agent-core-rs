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
//! - [`service`] — axum RL service skeleton (health/evaluate, 501 RL endpoints).
//! - [`rl`] — RL records and API-shaped ML stubs.
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
/// Candidates, the optimizer baseline, and checkpoints/stores/manager.
pub mod optimizer;
/// Pure evolution events, observer trait, and harness rail adapter.
pub mod rails;
/// RL records and API-shaped ML stubs.
pub mod rl;
/// Axum RL service skeleton (health/evaluate, 501 RL endpoints).
pub mod service;
/// Bounded train/evaluate loop, progress, callbacks, and reports.
pub mod trainer;
