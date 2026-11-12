//! Standalone facade for the auto-harness optimisation session loop.
//!
//! The implementation lives in [`openjiuwen_rsi::auto_harness`] (module path
//! `openjiuwen_rsi::harness_rsi::auto_harness`), ported from Python
//! `openjiuwen.rsi.harness_rsi.auto_harness`. This crate is a thin, zero-cost
//! re-export shim: it preserves the Python import path
//! `openjiuwen.rsi.auto_harness` for consumers that treat `auto_harness` as a
//! standalone package (`openjiuwen-auto-harness`), so they can depend on it
//! without pulling the rest of `openjiuwen-rsi`'s public surface into scope.
//!
//! Nothing is declared here beyond re-exports — there is no second definition
//! and no behaviour of its own, so types obtained through this crate are the
//! *same* types as those from `openjiuwen_rsi::auto_harness` (the integration
//! test in `tests/reexports.rs` pins that with [`std::any::TypeId`]).
//!
//! # Surface
//!
//! - [`AutoHarnessConfig`] / [`PipelinePreference`] — serialisable session config.
//! - [`AutoHarnessOrchestrator`] / [`AutoHarnessOrchestratorBuilder`] — the loop
//!   and its builder.
//! - [`HarnessRunner`] / [`HarnessEvaluator`] / [`HarnessRunRequest`] /
//!   [`HarnessRunOutcome`] — the seam to `openjiuwen-harness`.
//! - [`Budget`] / [`BudgetLimits`] / [`BudgetStatus`] — wall-clock and cost accounting.
//! - [`CheckpointStore`] / [`InMemoryCheckpointStore`] / [`KvCheckpointStore`] /
//!   [`SessionCheckpointStore`] / [`SessionState`] — resumable sessions.
//! - [`AutoHarnessEvent`] — the `type`-tagged session event stream.
//! - [`OptimizationTask`], [`IterationRecord`], [`SessionSummary`], [`Gap`],
//!   [`StageResult`], [`StageStatus`], [`TaskStatus`], [`StopConditions`],
//!   [`StopDecision`], [`StopReason`] — task and stop-condition records.
//! - [`budget`], [`builder`], [`checkpoint`], [`config`], [`event`],
//!   [`orchestrator`], [`runner`], [`task`] — the canonical submodules.
//!
//! [`openjiuwen_rsi::auto_harness`]: openjiuwen_rsi::auto_harness
#![forbid(unsafe_code)]
// The crate root intentionally repeats the module name of its source module:
// the whole point is to mirror the Python package path `rsi.auto_harness`.
#![allow(clippy::module_name_repetitions)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]

// The canonical implementation, re-exported wholesale. This glob is what
// guarantees new items added to `openjiuwen_rsi::auto_harness` reach this
// facade, and it is how the `budget`, `builder`, `checkpoint`, `config`,
// `event`, `orchestrator`, `runner`, and `task` submodules stay reachable.
pub use openjiuwen_rsi::auto_harness::*;

// Explicit re-exports of the primary names so the public surface is greppable
// and documented in one place. These shadow the glob for the same names, which
// is allowed and keeps both spellings trivially identical.
pub use openjiuwen_rsi::auto_harness::{
    AutoHarnessConfig, AutoHarnessEvent, AutoHarnessOrchestrator, AutoHarnessOrchestratorBuilder,
    Budget, BudgetLimits, BudgetStatus, CheckpointStore, Gap, HarnessEvaluator, HarnessRunOutcome,
    HarnessRunRequest, HarnessRunner, InMemoryCheckpointStore, IterationRecord, KvCheckpointStore,
    OptimizationTask, PipelinePreference, SessionCheckpointStore, SessionState, SessionSummary,
    StageResult, StageStatus, StopConditions, StopDecision, StopReason, TaskStatus,
    DEFAULT_SEED_BODY, SEED_CANDIDATE_ID,
};

/// Returns the crate name, `"openjiuwen-auto-harness"`.
///
/// Useful for diagnostics and for binaries that report which packaging of the
/// shared `auto_harness` surface they were linked against.
#[must_use]
pub const fn crate_name() -> &'static str {
    "openjiuwen-auto-harness"
}
