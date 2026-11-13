//! The continuous (two-stage, attempt-based) single-harness optimisation loop.
//!
//! Ported from Python
//! `openjiuwen/rsi/harness_rsi/single_harness/continuous.py`, reduced to its
//! deterministic orchestration core: *per-target improvement followed by
//! exactly one independent full evaluation*.
//!
//! The continuous policy differs from the epoch-checkpointed
//! [`super::iterative::IterativeSingleHarness`] in three ways:
//!
//! - no epoch replay: a batch is retried up to
//!   [`SingleHarnessConfig::max_attempts_per_batch`] times inside the same
//!   epoch, and at most one candidate per batch is promoted;
//! - a run-wide trial cap ([`SingleHarnessConfig::max_trials`]) bounds the
//!   number of improved candidates, persisted as `continuous_trials`;
//! - regression is measured against the *current* incumbent per-case scores
//!   (updated on every promotion) with [`SingleHarnessConfig::protected_cases`]
//!   honoured by the full-evaluation decision, and the legacy
//!   "promote if it beats the incumbent" shortcut is never applied.
//!
//! Stop reasons are the shared ones: [`StopReason::MaxIterations`] (iteration or
//! trial cap), [`StopReason::Converged`], [`StopReason::TargetScore`],
//! [`StopReason::BudgetExhausted`] and [`StopReason::Cancelled`].

use std::sync::Arc;

use openjiuwen_common::error::Result;

use crate::evolution::{Case, Evaluator, MutationStrategy};
use crate::harness_rsi::auto_harness::{BudgetLimits, StopReason};

use super::config::SingleHarnessConfig;
use super::iterative::{IterativeResult, SessionCore};
use super::promotion::REASON_THRESHOLD_SATURATED;
use super::state::{batch_key, SingleHarnessState};

/// Result of a finished continuous run (same shape as the iterative result).
pub type ContinuousResult = IterativeResult;

/// Continuous two-stage single-harness optimiser.
///
/// See the module documentation for the policy; the API mirrors
/// [`super::iterative::IterativeSingleHarness`] so a caller can switch policies
/// without touching its wiring. This type always runs the continuous policy:
/// the configuration's `mode` is recorded in the state (and therefore checked
/// on resume) but never switches the loop.
pub struct ContinuousSingleHarness {
    /// Shared state and collaborators.
    core: SessionCore,
}

impl ContinuousSingleHarness {
    /// Builds a fresh continuous run.
    ///
    /// An empty `dataset` means "every case in `cases`".
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::Error::Config`] when the configuration is invalid, and
    /// [`openjiuwen_common::error::Error::Validation`] when there is no case or the dataset names a case
    /// that is not part of `cases`.
    pub fn new(
        config: SingleHarnessConfig,
        evaluator: Arc<dyn Evaluator>,
        improver: Arc<dyn MutationStrategy>,
        task_id: impl Into<String>,
        dataset: Vec<String>,
        cases: Vec<Case>,
        seed_body: impl Into<String>,
    ) -> Result<Self> {
        let core = SessionCore::new(
            config, evaluator, improver, task_id, dataset, cases, seed_body,
        )?;
        Ok(Self { core })
    }

    /// Resumes a persisted run instead of starting from scratch.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::Error::Validation`] when the state belongs to a different
    /// fingerprint, task id, iteration cap, protocol signature or mode.
    pub fn with_state(mut self, previous: SingleHarnessState) -> Result<Self> {
        self.core = self.core.resume(previous)?;
        Ok(self)
    }

    /// Replaces the budget limits.
    #[must_use]
    pub fn with_budget(mut self, limits: BudgetLimits) -> Self {
        self.core.set_budget(limits);
        self
    }

    /// Returns the current state.
    #[must_use]
    pub fn state(&self) -> &SingleHarnessState {
        &self.core.state
    }

    /// Returns the body of the incumbent.
    #[must_use]
    pub fn best_body(&self) -> &str {
        &self.core.best_body
    }

    /// Returns whether [`Self::stop`] was called.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.core.is_cancelled()
    }

    /// Runs every epoch until a stop reason applies.
    ///
    /// # Errors
    ///
    /// Returns the evaluator error when the baseline cannot be graded (the run
    /// is then marked failed); per-candidate failures are recorded instead.
    pub async fn run(&mut self) -> Result<ContinuousResult> {
        let max_epochs = self.core.config.max_epochs;
        for epoch in 1..=max_epochs {
            if !self.run_epoch(epoch).await? {
                break;
            }
        }
        self.core.freeze();
        Ok(self.core.result())
    }

    /// Runs one epoch (one batch, up to `max_attempts_per_batch` candidates).
    ///
    /// Returns `false` when the run stopped. A batch that already completed is
    /// skipped without consuming an iteration or a trial.
    ///
    /// # Errors
    ///
    /// Returns the evaluator error when the baseline cannot be graded.
    pub async fn run_epoch(&mut self, epoch: u32) -> Result<bool> {
        if self.core.is_cancelled() {
            self.core.record_stop(StopReason::Cancelled);
            return Ok(false);
        }
        if let Some(reason) = self.core.budget.exhausted() {
            self.core.record_stop(reason);
            return Ok(false);
        }
        self.core.ensure_baseline().await?;
        if let Some(reason) = self.core.stop_reason() {
            self.core.record_stop(reason);
            return Ok(false);
        }
        if self.trials_exhausted() {
            self.core.record_stop(StopReason::MaxIterations);
            return Ok(false);
        }
        let key = batch_key(epoch, 0);
        if self.core.state.is_batch_completed(&key) {
            return Ok(true);
        }
        let batch = self.core.select_batch(epoch);
        if batch.is_empty() {
            self.core.record_outcome(
                epoch,
                0,
                REASON_THRESHOLD_SATURATED,
                self.core.state.best_score,
                false,
            );
            return Ok(self.core.check_after_iteration());
        }

        // A batch with zero attempts could never be evaluated, so one attempt
        // always runs; the documented default is two.
        let attempts = self.core.config.max_attempts_per_batch.max(1);
        let mut promoted = false;
        let mut attempted = 0_u32;
        let mut last_reason = String::new();
        let mut last_score = None;
        for attempt in 1..=attempts {
            if self.trials_exhausted() {
                self.core.record_stop(StopReason::MaxIterations);
                break;
            }
            attempted = attempt;
            let generation = self.core.state.continuous_trials.saturating_add(1);
            self.core.state.continuous_attempts = attempt;
            self.core.state.continuous_trials = generation;
            let outcome = self
                .core
                .run_attempt(epoch, 0, attempt, generation, &batch, false)
                .await?;
            last_reason = outcome.reason;
            last_score = outcome.score;
            if outcome.promoted {
                promoted = true;
                break;
            }
            if let Some(reason) = self.core.stop_reason() {
                self.core.record_stop(reason);
                break;
            }
        }
        self.core.state.continuous_attempts = 0;
        if attempted > 0 {
            self.core.state.mark_batch_completed(&key);
            self.core
                .record_outcome(epoch, 0, last_reason, last_score, promoted);
        }
        if self.core.state.stop_reason.is_some() {
            return Ok(false);
        }
        Ok(self.core.check_after_iteration())
    }

    /// Cancels the run; the in-flight attempt observes it at its next await.
    pub fn stop(&self) {
        self.core.cancel();
    }

    /// Returns whether the run-wide trial cap is reached.
    fn trials_exhausted(&self) -> bool {
        let cap = self.core.config.max_trials;
        cap > 0 && self.core.state.continuous_trials >= cap
    }
}
