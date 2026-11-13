//! The iterative (epoch-checkpointed) single-harness optimisation loop.
//!
//! Ported from Python
//! `openjiuwen/rsi/harness_rsi/single_harness/iterative.py`, reduced to its
//! deterministic orchestration core. The Python loop runs one batch per epoch,
//! evaluates the candidate on that batch, then re-evaluates every case before
//! promoting; this port keeps exactly that shape and moves the LLM analyser and
//! improver behind the crate's existing seams:
//!
//! - the improver is any [`MutationStrategy`] (the default is
//!   [`crate::evolution::DeterministicMutator`], so tests stay offline);
//! - the evaluator is any [`Evaluator`] (a harness driver in production, a
//!   scripted fake in tests);
//! - the budget is an injected [`BudgetLimits`].
//!
//! ## Loop
//!
//! 1. the seed body is evaluated over **all** cases and becomes the baseline;
//! 2. per epoch the batch is selected from the cases that still have a
//!    satisfaction gap ([`batch_filter`]), cycling by epoch offset so the
//!    batches are deterministic and cover the dataset;
//! 3. already completed batches ([`super::state::batch_key`]) are skipped, so a
//!    resumed run never repeats work;
//! 4. the candidate runs the batch gate ([`stage_one_gate`]) and, when it
//!    passes, a full re-evaluation ([`full_evaluation_decision`]);
//! 5. a promotion replaces the incumbent and marks the batch completed; a
//!    rejection appends a journal entry and feeds the patience counter;
//! 6. the run stops on cancellation, budget exhaustion, the iteration or epoch
//!    cap, the target score, or convergence.
//!
//! The legacy policy ([`PromotionPolicyMode::LegacyEpochV1`]) keeps Python's
//! more forgiving rule: when the strict full-evaluation decision rejects a
//! candidate that nonetheless beats the incumbent's overall score, the
//! candidate is still promoted (`promote_on_best_found`). The continuous policy
//! never takes that shortcut.
//!
//! Everything is deterministic: ids come from the mutation strategy, the batch
//! selection depends only on the epoch and the case ids, and no wall clock
//! enters a score.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use serde_json::json;
use tokio_util::sync::CancellationToken;
use tracing::{debug, warn};

use openjiuwen_common::error::{Error, Result};

use crate::evolution::{
    Candidate, CandidateStatus, Case, CaseResult, EvaluationReport, Evaluator, MutationRequest,
    MutationStrategy,
};
use crate::harness_rsi::auto_harness::{Budget, BudgetLimits, StopReason};

use super::config::SingleHarnessConfig;
use super::promotion::{
    batch_filter, case_scores, full_evaluation_decision, stage_one_gate, CaseScores, EPSILON,
    REASON_ENGINEERING_FAILED, REASON_SELECTED, REASON_THRESHOLD_SATURATED,
};
use super::state::{batch_key, PromotionPolicyMode, SingleHarnessState, SingleHarnessStatus};

/// `tracing` target of the single-harness loops.
const LOG_TARGET: &str = "openjiuwen_rsi::single_harness";

/// Metadata key holding the incumbent per-case scores inside the state.
///
/// Per-case scores are not a field of the Python checkpoint, so the Rust port
/// parks them in `metadata`: a resumed run then keeps its regression context
/// instead of comparing against the seed again.
pub(crate) const CASE_SCORES_KEY: &str = "best_case_scores";

/// Wall-clock budget of a run that did not inject one (24 hours).
const DEFAULT_WALL_CLOCK_SECS: u64 = 24 * 60 * 60;

/// Timeout of a single evaluation when the caller did not inject one (1 hour).
const DEFAULT_TASK_TIMEOUT_SECS: u64 = 60 * 60;

/// Result of a finished single-harness run.
#[derive(Debug, Clone, PartialEq)]
pub struct IterativeResult {
    /// Final persisted state (incumbent, journal, gates, usage, stop reason).
    pub state: SingleHarnessState,
    /// Best overall score found by the run.
    pub best_score: Option<f64>,
    /// Body of the incumbent.
    pub best_body: String,
    /// Number of promotions in this run (resumed runs count their own).
    pub promotions: usize,
    /// Number of evaluator invocations in this run (baseline included).
    pub evaluations: usize,
}

/// Outcome of one candidate pushed through both promotion stages.
///
/// The stage-level detail (batch report, gate decision, full-evaluation
/// decision) is not carried here: it is already persisted as a
/// [`super::state::CandidateGateRecord`] and summarised in the journal entry the
/// caller appends.
#[derive(Debug, Clone)]
pub(crate) struct AttemptOutcome {
    /// Whether the candidate replaced the incumbent.
    pub(crate) promoted: bool,
    /// Decision reason literal.
    pub(crate) reason: String,
    /// Candidate score (full evaluation when it ran, batch score otherwise).
    pub(crate) score: Option<f64>,
}

/// Returns whether `score` strictly beats the incumbent `best`.
pub(crate) fn beats_incumbent(best: Option<f64>, score: Option<f64>) -> bool {
    match (best, score) {
        (Some(best), Some(score)) => score > best + EPSILON,
        (None, Some(_)) => true,
        _ => false,
    }
}

/// Default budget of a run that does not inject [`BudgetLimits`].
fn default_budget_limits() -> BudgetLimits {
    BudgetLimits {
        wall_clock: Duration::from_secs(DEFAULT_WALL_CLOCK_SECS),
        cost_limit_usd: f64::MAX,
        task_timeout: Duration::from_secs(DEFAULT_TASK_TIMEOUT_SECS),
    }
}

/// State, collaborators and progress shared by both loops.
///
/// This is the crate-internal half of [`IterativeSingleHarness`] and
/// [`super::continuous::ContinuousSingleHarness`]: the policies differ only in
/// how they sequence [`SessionCore::run_attempt`].
pub(crate) struct SessionCore {
    /// Run configuration.
    pub(crate) config: SingleHarnessConfig,
    /// Grader of candidate bodies.
    pub(crate) evaluator: Arc<dyn Evaluator>,
    /// Proposer of candidate bodies (the LLM improver in production).
    pub(crate) improver: Arc<dyn MutationStrategy>,
    /// Persisted run state.
    pub(crate) state: SingleHarnessState,
    /// Body of the incumbent.
    pub(crate) best_body: String,
    /// Per-case scores of the incumbent.
    pub(crate) best_scores: CaseScores,
    /// Wall-clock and cost budget.
    pub(crate) budget: Budget,
    /// Cancellation token flipped by `stop`.
    pub(crate) cancelled: CancellationToken,
    /// Every evaluation case, in the caller's order.
    pub(crate) cases: Vec<Case>,
    /// Seed body the baseline is measured on.
    pub(crate) seed_body: String,
    /// Promotions observed by this process.
    pub(crate) promotions: usize,
    /// Evaluator invocations observed by this process.
    pub(crate) evaluations: usize,
}

impl SessionCore {
    /// Builds the core of a fresh run.
    pub(crate) fn new(
        config: SingleHarnessConfig,
        evaluator: Arc<dyn Evaluator>,
        improver: Arc<dyn MutationStrategy>,
        task_id: impl Into<String>,
        dataset: Vec<String>,
        cases: Vec<Case>,
        seed_body: impl Into<String>,
    ) -> Result<Self> {
        config.validate()?;
        if cases.is_empty() {
            return Err(Error::validation(
                "single-harness runs need at least one evaluation case",
            ));
        }
        let dataset = if dataset.is_empty() {
            cases.iter().map(|case| case.id.clone()).collect()
        } else {
            dataset
        };
        let known: std::collections::BTreeSet<&str> =
            cases.iter().map(|case| case.id.as_str()).collect();
        if let Some(missing) = dataset
            .iter()
            .find(|case_id| !known.contains(case_id.as_str()))
        {
            return Err(Error::validation(format!(
                "dataset case `{missing}` is not part of the evaluation case set"
            )));
        }
        let mut seen = std::collections::BTreeSet::new();
        let dataset: Vec<String> = dataset
            .into_iter()
            .filter(|case_id| seen.insert(case_id.clone()))
            .collect();
        let fingerprint = config.fingerprint(&dataset);
        let mut state = SingleHarnessState::new(
            task_id,
            fingerprint,
            dataset,
            config.max_iterations,
            config.mode,
        );
        state
            .evaluation_protocol_signature
            .clone_from(&config.evaluation_protocol_signature);
        Ok(Self {
            config,
            evaluator,
            improver,
            state,
            best_body: String::new(),
            best_scores: CaseScores::new(),
            budget: Budget::start(default_budget_limits()),
            cancelled: CancellationToken::new(),
            cases,
            seed_body: seed_body.into(),
            promotions: 0,
            evaluations: 0,
        })
    }

    /// Restores `previous` as this run's state.
    pub(crate) fn resume(mut self, previous: SingleHarnessState) -> Result<Self> {
        if previous.mode != self.config.mode {
            return Err(Error::validation(format!(
                "single-harness state mode mismatch: expected `{}`, found `{}`",
                self.config.mode.token(),
                previous.mode.token()
            )));
        }
        let fingerprint = self.fingerprint();
        let signature = self.config.evaluation_protocol_signature.clone();
        let state = SingleHarnessState::resume_from(
            previous,
            &fingerprint,
            &self.state.task_id.clone(),
            self.config.max_iterations,
            &signature,
        )?;
        self.best_body.clone_from(&state.best_body);
        self.best_scores = state
            .metadata
            .get(CASE_SCORES_KEY)
            .and_then(|value| serde_json::from_value(value.clone()).ok())
            .unwrap_or_default();
        self.state = state;
        Ok(self)
    }

    /// Replaces the budget limits, starting a fresh wall clock.
    pub(crate) fn set_budget(&mut self, limits: BudgetLimits) {
        self.budget = Budget::start(limits);
    }

    /// Returns the configuration fingerprint of this run.
    pub(crate) fn fingerprint(&self) -> String {
        self.config.fingerprint(&self.state.dataset)
    }

    /// Returns whether [`Self::cancel`] was called.
    pub(crate) fn is_cancelled(&self) -> bool {
        self.cancelled.is_cancelled()
    }

    /// Cancels the run; in-flight iterations observe it at their next await.
    pub(crate) fn cancel(&self) {
        self.cancelled.cancel();
    }

    /// Stores the incumbent per-case scores in the state metadata.
    pub(crate) fn write_case_scores(&mut self) {
        match serde_json::to_value(&self.best_scores) {
            Ok(value) => {
                self.state
                    .metadata
                    .insert(CASE_SCORES_KEY.to_owned(), value);
            }
            Err(err) => {
                warn!(target: LOG_TARGET, error = %err, "cannot persist incumbent case scores");
            }
        }
    }

    /// Builds the incumbent as a mutation parent.
    pub(crate) fn parent_candidate(&self) -> Candidate {
        let mut parent = Candidate::new(
            format!("{}/best", self.state.task_id),
            self.best_body.clone(),
        );
        if !self.best_scores.is_empty() {
            let results = self
                .best_scores
                .iter()
                .map(|(case_id, score)| CaseResult {
                    case_id: case_id.clone(),
                    passed: *score >= 1.0 - EPSILON,
                    score: Some(*score),
                    weight: 1.0,
                    error: None,
                    details: serde_json::Value::Null,
                })
                .collect();
            parent.evaluation = Some(EvaluationReport::from_results(results));
            parent.status = CandidateStatus::Evaluated;
        }
        parent
    }

    /// Selects the batch of `epoch`.
    ///
    /// The open cases are those whose satisfaction gap still exceeds
    /// `min_improvement` ([`batch_filter`]); the batch is `batch_size` of them
    /// starting at the epoch's offset, wrapping around, so successive epochs
    /// cover the dataset deterministically.
    pub(crate) fn select_batch(&self, epoch: u32) -> Vec<String> {
        let scores: CaseScores = self
            .state
            .dataset
            .iter()
            .map(|case_id| {
                (
                    case_id.clone(),
                    self.best_scores.get(case_id).copied().unwrap_or(0.0),
                )
            })
            .collect();
        let open = batch_filter(&scores, self.config.min_improvement);
        if open.is_empty() {
            return Vec::new();
        }
        let size = self.config.batch_size.min(open.len());
        let offset = ((epoch.saturating_sub(1) as usize) * self.config.batch_size) % open.len();
        (0..size)
            .map(|step| open[(offset + step) % open.len()].clone())
            .collect()
    }

    /// Returns the [`Case`] definitions of `case_ids`, in evaluation order.
    pub(crate) fn batch_cases(&self, case_ids: &[String]) -> Vec<Case> {
        self.cases
            .iter()
            .filter(|case| case_ids.iter().any(|case_id| case_id == &case.id))
            .cloned()
            .collect()
    }

    /// Returns the incumbent scores restricted to `case_ids`.
    pub(crate) fn scores_for(&self, case_ids: &[String]) -> CaseScores {
        case_ids
            .iter()
            .filter_map(|case_id| {
                self.best_scores
                    .get(case_id)
                    .map(|score| (case_id.clone(), *score))
            })
            .collect()
    }

    /// Evaluates `candidate` over `cases` and accounts for the cost.
    pub(crate) async fn evaluate(
        &mut self,
        candidate: &Candidate,
        cases: &[Case],
    ) -> Result<EvaluationReport> {
        let evaluator = Arc::clone(&self.evaluator);
        let report = evaluator.evaluate(candidate, cases).await?;
        self.account(&report);
        Ok(report)
    }

    /// Records one evaluator invocation and its cost.
    fn account(&mut self, report: &EvaluationReport) {
        self.evaluations += 1;
        self.state.usage.calls = self.state.usage.calls.saturating_add(1);
        let cost = report.cost_usd;
        if cost.is_finite() && cost > 0.0 {
            self.state.usage.cost_usd += cost;
            self.budget.record_cost(cost);
        }
    }

    /// Proposes one candidate from the incumbent.
    pub(crate) async fn propose(
        &self,
        generation: u32,
        batch_key: &str,
        attempt: u32,
    ) -> Result<Candidate> {
        let mut metadata = BTreeMap::new();
        metadata.insert("batch_key".to_owned(), json!(batch_key));
        metadata.insert("attempt".to_owned(), json!(attempt));
        if let Some(score) = self.state.best_score {
            metadata.insert("incumbent_score".to_owned(), json!(score));
        }
        let proposals = self
            .improver
            .propose(MutationRequest {
                parents: vec![self.parent_candidate()],
                generation,
                count: 1,
                metadata,
            })
            .await?;
        proposals
            .into_iter()
            .next()
            .ok_or_else(|| Error::validation("single-harness improver returned no candidate"))
    }

    /// Appends a gate record for one evaluated candidate.
    pub(crate) fn record_gate(
        &mut self,
        batch_key: &str,
        candidate_id: &str,
        passed: bool,
        reason: &str,
        round: u32,
    ) {
        self.state
            .candidate_gates
            .push(super::state::CandidateGateRecord {
                batch_key: batch_key.to_owned(),
                candidate_id: candidate_id.to_owned(),
                passed,
                reason: reason.to_owned(),
                round,
            });
    }

    /// Appends a journal entry for one batch attempt.
    pub(crate) fn record_outcome(
        &mut self,
        epoch: u32,
        batch_index: u32,
        reason: impl Into<String>,
        score: Option<f64>,
        promoted: bool,
    ) {
        if promoted {
            self.promotions += 1;
        }
        self.state
            .optimization_journal
            .push(super::state::JournalEntry {
                epoch,
                batch_index,
                reason: reason.into(),
                score,
                promoted,
            });
    }

    /// Replaces the incumbent with `candidate`.
    pub(crate) fn apply_promotion(
        &mut self,
        candidate: &Candidate,
        report: &EvaluationReport,
        scores: CaseScores,
    ) {
        let score = report.score();
        self.best_body.clone_from(&candidate.body);
        self.state.best_body.clone_from(&candidate.body);
        self.state.best_score = score;
        self.state.best_harness_refs_path = None;
        match (self.state.historical_peak_score, score) {
            (Some(peak), Some(score)) if score > peak => {
                self.state.historical_peak_score = Some(score);
            }
            (None, Some(score)) => self.state.historical_peak_score = Some(score),
            _ => {}
        }
        self.best_scores = scores;
        self.write_case_scores();
        debug!(
            target: LOG_TARGET,
            candidate = candidate.id.as_str(),
            score,
            "single-harness promotion"
        );
    }

    /// Returns the stop reason that applies right now, if any.
    ///
    /// Order: cancellation, budget, iteration cap, target score, convergence.
    pub(crate) fn stop_reason(&self) -> Option<StopReason> {
        if self.cancelled.is_cancelled() {
            return Some(StopReason::Cancelled);
        }
        if let Some(reason) = self.budget.exhausted() {
            return Some(reason);
        }
        if self.state.iterations() >= self.config.max_iterations {
            return Some(StopReason::MaxIterations);
        }
        if let (Some(target), Some(best)) = (self.config.target_score, self.state.best_score) {
            if best >= target {
                return Some(StopReason::TargetScore);
            }
        }
        if self.config.patience > 0
            && u32::try_from(self.state.consecutive_non_promotions()).unwrap_or(u32::MAX)
                >= self.config.patience
        {
            return Some(StopReason::Converged);
        }
        None
    }

    /// Records `reason` as the reason the run stopped.
    pub(crate) fn record_stop(&mut self, reason: StopReason) {
        if self.state.stop_reason.is_some() {
            return;
        }
        self.state.stop_reason = Some(reason);
        self.state.status = match reason {
            StopReason::Cancelled => SingleHarnessStatus::Cancelled,
            StopReason::Failed => SingleHarnessStatus::Failed,
            _ => SingleHarnessStatus::Completed,
        };
        debug!(target: LOG_TARGET, ?reason, "single-harness run stopped");
    }

    /// Stops the run when a stop reason applies; returns `false` when it did.
    pub(crate) fn check_after_iteration(&mut self) -> bool {
        match self.stop_reason() {
            Some(reason) => {
                self.record_stop(reason);
                false
            }
            None => true,
        }
    }

    /// Fails the run and returns `err`.
    pub(crate) fn abort(&mut self, err: Error) -> Error {
        self.record_stop(StopReason::Failed);
        err
    }

    /// Evaluates the seed over every case, once per run.
    ///
    /// A resumed run keeps the persisted baseline and incumbent case scores;
    /// only what is missing is recomputed. An evaluator error aborts the run
    /// (there is no baseline to compare against).
    pub(crate) async fn ensure_baseline(&mut self) -> Result<()> {
        let baseline_needed = self.state.baseline_score.is_none();
        let scores_needed = self.best_scores.is_empty();
        if !baseline_needed && !scores_needed {
            return Ok(());
        }
        let seed = Candidate::new(
            format!("{}/seed", self.state.task_id),
            self.seed_body.clone(),
        );
        let all_cases = self.cases.clone();
        let report = match self.evaluate(&seed, &all_cases).await {
            Ok(report) => report,
            Err(err) => return Err(self.abort(err)),
        };
        let scores = case_scores(&report);
        if scores.is_empty() {
            return Err(self.abort(Error::validation(
                "single-harness baseline evaluation graded no case",
            )));
        }
        if baseline_needed {
            self.state.baseline_score = report.score();
            if self.state.best_score.is_none() {
                self.state.best_score = report.score();
                self.state.historical_peak_score = report.score();
            }
        }
        if scores_needed {
            self.best_scores = scores;
            self.write_case_scores();
        }
        if self.state.best_body.is_empty() {
            self.state.best_body.clone_from(&self.seed_body);
            self.best_body.clone_from(&self.seed_body);
        }
        Ok(())
    }

    /// Pushes one candidate through the batch gate and the full evaluation.
    ///
    /// `legacy_override` adds the legacy policy's shortcut: a candidate that
    /// beats the incumbent's overall score is promoted even when the strict
    /// decision rejects it.
    pub(crate) async fn run_attempt(
        &mut self,
        epoch: u32,
        batch_index: u32,
        attempt: u32,
        generation: u32,
        batch: &[String],
        legacy_override: bool,
    ) -> Result<AttemptOutcome> {
        let key = batch_key(epoch, batch_index);
        let parent = self.parent_candidate();
        let candidate = match self.propose(generation, &key, attempt).await {
            Ok(candidate) => candidate,
            Err(err) => {
                warn!(target: LOG_TARGET, error = %err, "single-harness improver failed");
                return Ok(AttemptOutcome {
                    promoted: false,
                    reason: format!("{REASON_ENGINEERING_FAILED}: {err}"),
                    score: None,
                });
            }
        };
        let content_changed = candidate.body != parent.body;
        let batch_cases = self.batch_cases(batch);
        if batch_cases.is_empty() {
            return Ok(AttemptOutcome {
                promoted: false,
                reason: REASON_THRESHOLD_SATURATED.to_owned(),
                score: None,
            });
        }
        let batch_before = self.scores_for(batch);
        let batch_report = match self.evaluate(&candidate, &batch_cases).await {
            Ok(report) => report,
            Err(err) => {
                warn!(target: LOG_TARGET, error = %err, "single-harness batch evaluation failed");
                let reason = format!("{REASON_ENGINEERING_FAILED}: {err}");
                self.record_gate(&key, candidate.id.as_str(), false, &reason, attempt);
                return Ok(AttemptOutcome {
                    promoted: false,
                    reason,
                    score: None,
                });
            }
        };
        let batch_after: CaseScores = case_scores(&batch_report)
            .into_iter()
            .filter(|(case_id, _)| batch.iter().any(|id| id == case_id))
            .collect();
        let stage_one = stage_one_gate(&batch_before, &batch_after, content_changed, None);
        self.record_gate(
            &key,
            candidate.id.as_str(),
            stage_one.passed,
            &stage_one.reason,
            attempt,
        );
        if !stage_one.passed {
            return Ok(AttemptOutcome {
                promoted: false,
                reason: stage_one.reason.clone(),
                score: batch_report.score(),
            });
        }

        let all_cases = self.cases.clone();
        let incumbent = self.best_scores.clone();
        let full_report = match self.evaluate(&candidate, &all_cases).await {
            Ok(report) => report,
            Err(err) => {
                warn!(target: LOG_TARGET, error = %err, "single-harness full evaluation failed");
                let reason = format!("{REASON_ENGINEERING_FAILED}: {err}");
                self.record_gate(&key, candidate.id.as_str(), false, &reason, attempt);
                return Ok(AttemptOutcome {
                    promoted: false,
                    reason,
                    score: batch_report.score(),
                });
            }
        };
        let full_scores = case_scores(&full_report);
        let decision = full_evaluation_decision(
            &incumbent,
            &full_scores,
            &stage_one.improved,
            &self.config.protected_cases,
            self.config.min_improvement,
            self.config.max_regression,
        );
        let mut promoted = decision.promoted;
        let mut reason = decision.reason.clone();
        if !promoted
            && legacy_override
            && self.config.promote_on_best_found
            && beats_incumbent(self.state.best_score, full_report.score())
        {
            promoted = true;
            REASON_SELECTED.clone_into(&mut reason);
        }
        if promoted {
            self.apply_promotion(&candidate, &full_report, full_scores);
        }
        Ok(AttemptOutcome {
            promoted,
            reason,
            score: full_report.score(),
        })
    }

    /// Freezes the state of a finished run.
    pub(crate) fn freeze(&mut self) {
        self.write_case_scores();
        if self.state.stop_reason.is_none() {
            self.state.stop_reason = Some(StopReason::MaxIterations);
        }
        if !self.state.status.is_terminal() {
            self.state.status = SingleHarnessStatus::Completed;
        }
    }

    /// Snapshots the run into an [`IterativeResult`].
    pub(crate) fn result(&self) -> IterativeResult {
        IterativeResult {
            state: self.state.clone(),
            best_score: self.state.best_score,
            best_body: self.best_body.clone(),
            promotions: self.promotions,
            evaluations: self.evaluations,
        }
    }
}

/// Epoch-checkpointed single-harness optimiser.
///
/// See the module documentation for the loop; the constructor takes the
/// collaborators, the run identity, the dataset, the full case set and the seed
/// body, so the type is usable offline and in tests.
pub struct IterativeSingleHarness {
    /// Shared state and collaborators.
    core: SessionCore,
}

impl IterativeSingleHarness {
    /// Builds a fresh run.
    ///
    /// An empty `dataset` means "every case in `cases`". The dataset must be a
    /// subset of the case ids, and the configuration must pass
    /// [`SingleHarnessConfig::validate`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] when the configuration is invalid, and
    /// [`Error::Validation`] when there is no case or the dataset names a case
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
    /// The previous state has to belong to the same configuration: the
    /// fingerprint (dataset, protocol signature, mode), task id, iteration cap
    /// and protocol signature must match, otherwise the persisted scores would
    /// be compared against a different protocol.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] naming the field that differs.
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
    /// is then marked failed); evaluation errors on individual candidates never
    /// abort the loop.
    pub async fn run(&mut self) -> Result<IterativeResult> {
        let max_epochs = self.core.config.max_epochs;
        for epoch in 1..=max_epochs {
            if !self.run_epoch(epoch).await? {
                break;
            }
        }
        self.core.freeze();
        Ok(self.core.result())
    }

    /// Runs one epoch; returns `false` when the run stopped.
    ///
    /// A batch that is already completed is skipped without consuming an
    /// iteration, which is what makes a resumed run continue where the previous
    /// one stopped.
    ///
    /// # Errors
    ///
    /// Returns the evaluator error when the baseline cannot be graded; every
    /// per-candidate failure is recorded instead.
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
        let legacy = self.core.config.mode == PromotionPolicyMode::LegacyEpochV1;
        let generation = self.core.state.iterations().saturating_add(1);
        let outcome = self
            .core
            .run_attempt(epoch, 0, 1, generation, &batch, legacy)
            .await?;
        if outcome.promoted {
            self.core.state.mark_batch_completed(&key);
        }
        self.core
            .record_outcome(epoch, 0, outcome.reason, outcome.score, outcome.promoted);
        Ok(self.core.check_after_iteration())
    }

    /// Cancels the run.
    ///
    /// The current iteration observes the cancellation at its next await point
    /// and the next iteration never starts; the stop reason becomes
    /// [`StopReason::Cancelled`].
    pub fn stop(&self) {
        self.core.cancel();
    }
}
