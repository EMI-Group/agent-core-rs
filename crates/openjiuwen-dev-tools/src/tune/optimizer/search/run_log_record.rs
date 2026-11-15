//! Building and emitting the run-log record of one prompt-search round.
//!
//! Split out of the loop so `mod.rs` stays the algorithm: this module turns a
//! finished round into one [`RunLogRecord`] and hands it to the caller's
//! [`RunLog`], and it is the only place that decides what a record contains.
//!
//! The contract is one record per completed round, written after the round's
//! convergence check, so a record always carries its round's stopping verdict.
//! Logging can never fail or perturb a run: a sink error or a poisoned
//! [`Mutex`] is logged with `tracing::warn!` and the loop continues, and the
//! record itself is built from values the loop already computed.

use std::collections::BTreeMap;
use std::sync::Mutex;

use crate::tune::optimizer::adopt::{AdoptDecision, AdoptPolicy};
use crate::tune::prompt_search::{
    ConvergenceState, CorrectnessReward, Evaluation, RunLog, RunLogRecord,
};

/// Everything one finished round contributes to its run-log record.
pub(super) struct RoundRecord<'a> {
    /// The 1-based round number.
    pub(super) iteration: usize,
    /// Every evaluated candidate of the round, in evaluation order.
    pub(super) evaluations: &'a [Evaluation],
    /// The index of the round's winning candidate inside `evaluations`.
    pub(super) best_index: usize,
    /// The run's best reward after this round.
    pub(super) best_score: f64,
    /// Whether this round changed the run's elite.
    pub(super) selected: bool,
    /// The convergence verdict of the round, when the detector fired.
    pub(super) convergence: Option<&'a ConvergenceState>,
    /// Why the loop stops after this round, if it does.
    pub(super) stop_reason: Option<&'a str>,
    /// The adoption decision, when a policy is attached.
    pub(super) adopt: Option<AdoptDecision>,
    /// The caller-supplied timestamp, in seconds.
    pub(super) timestamp: f64,
    /// The run identifier the round belongs to.
    pub(super) run_id: &'a str,
    /// The winner's observations; they are joined with `"; "`.
    pub(super) observations: &'a [String],
}

impl RoundRecord<'_> {
    /// Renders the round as one [`RunLogRecord`].
    ///
    /// The winner supplies the candidate, the reward, its raw components, the
    /// notes, and the drift deviation; the other candidates of the round supply
    /// `candidate_prompts` (evaluation order) and every candidate contributes to
    /// `scores`. `observations` is joined with `"; "` and left `None` when the
    /// winner had none, and `converged`/`convergence_reason` mirror the round's
    /// stopping verdict rather than the convergence state's own flags.
    pub(super) fn build(&self) -> RunLogRecord {
        let winner = &self.evaluations[self.best_index];
        let mut scores = BTreeMap::new();
        let mut others = Vec::new();
        for (index, evaluation) in self.evaluations.iter().enumerate() {
            let candidate = evaluation.candidate();
            scores.insert(candidate.candidate_id.clone(), evaluation.score());
            if index != self.best_index {
                others.push(candidate.prompt.clone());
            }
        }
        let reward = &winner.reward;
        let mut record = RunLogRecord::new(self.iteration, winner.candidate().clone())
            .with_candidate_prompts(others)
            .with_scores(scores)
            .with_reward(reward.score)
            .with_reward_components(reward.raw_components.clone())
            .with_reward_notes(reward.notes.clone())
            .with_best_score(self.best_score)
            .with_drift(reward.drift)
            .with_selected(self.selected)
            .with_timestamp(self.timestamp)
            .with_run_id(self.run_id);
        if let Some(convergence) = self.convergence {
            record = record.with_convergence(convergence.clone());
        }
        record = match self.stop_reason {
            Some(reason) => record.with_converged(true).with_convergence_reason(reason),
            None => record.with_converged(false),
        };
        if let Some(adopt) = self.adopt {
            record = record.with_adopt(adopt);
        }
        if !self.observations.is_empty() {
            record = record.with_observation(self.observations.join("; "));
        }
        record
    }
}

/// The adoption decision for a round's winner, or `None` when no policy is
/// attached.
///
/// The correctness the policy gates on is the winner execution's visible-case
/// mean score (the same mean [`CorrectnessReward`] scores), and `history` is the
/// round-best rewards of the *preceding* rounds, so a round is never compared
/// against itself.
pub(super) fn adopt_decision(
    policy: Option<&AdoptPolicy>,
    winner: &Evaluation,
    history: &[f64],
) -> Option<AdoptDecision> {
    policy.map(|policy| {
        policy.decide(
            winner.score(),
            visible_correctness(winner),
            winner.reward.drift,
            history,
        )
    })
}

/// The mean score of the winner execution's visible cases.
fn visible_correctness(winner: &Evaluation) -> f64 {
    let visible = winner
        .execution
        .visible_results()
        .into_iter()
        .cloned()
        .collect::<Vec<_>>();
    CorrectnessReward::new().correctness_on(&visible)
}

/// Writes one record to `log`, never failing the run.
///
/// A poisoned mutex and a sink error are both logged with `tracing::warn!` and
/// swallowed, so a broken run log cannot change a search result.
pub(super) fn emit(log: Option<&Mutex<dyn RunLog>>, record: &RunLogRecord) {
    let Some(log) = log else {
        return;
    };
    match log.lock() {
        Ok(mut sink) => {
            if let Err(error) = sink.record(record) {
                tracing::warn!(
                    iteration = record.iteration,
                    %error,
                    "run log sink rejected the record; the search result is unaffected"
                );
            }
        }
        Err(_poisoned) => {
            tracing::warn!(
                iteration = record.iteration,
                "run log sink mutex is poisoned; the record was dropped"
            );
        }
    }
}
