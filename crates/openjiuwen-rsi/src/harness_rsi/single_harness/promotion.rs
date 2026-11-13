//! Promotion mathematics shared by the single-harness optimisation loops.
//!
//! Ported from Python
//! `openjiuwen/rsi/harness_rsi/single_harness/promotion_policy.py`, reduced to
//! the deterministic core: every rule here is pure, works on plain per-case
//! scores, and never averages away a case failure. The Python original loads
//! evaluation references from disk and hashes harness packages; the Rust port
//! takes the already-materialised [`CaseScores`] so the evidence plumbing stays
//! with the caller (see the crate's deferred-scope table).
//!
//! Three decisions live here:
//!
//! - [`batch_filter`] picks the cases a batch still has to satisfy at all
//!   (`1.0 - score > min_improvement + EPSILON`);
//! - [`stage_one_gate`] decides whether a candidate is worth a full evaluation
//!   (it must change content and strictly improve at least one target);
//! - [`full_evaluation_decision`] decides whether the full re-evaluation of
//!   every case authorises a promotion (no target may fall back, no protected
//!   case may regress, and the aggregate score must actually improve).
//!
//! All comparisons use [`EPSILON`] so floating-point noise never turns an exact
//! boundary into a promotion.

use std::collections::BTreeMap;
use std::collections::BTreeSet;

use tracing::debug;

use crate::evolution::EvaluationReport;

/// Numeric tolerance of every promotion comparison (Python `EPSILON`).
pub const EPSILON: f64 = 1e-9;

/// `tracing` target of the promotion diagnostics.
const LOG_TARGET: &str = "openjiuwen_rsi::single_harness::promotion";

/// Reason: the improver returned the parent body unchanged.
pub const REASON_CONTENT_UNCHANGED: &str = "candidate_content_unchanged";
/// Reason prefix: the batch evaluation failed for engineering reasons.
pub const REASON_ENGINEERING_FAILED: &str = "engineering_failed";
/// Reason: no case of the batch can be improved because every one is already
/// fully satisfied.
pub const REASON_THRESHOLD_SATURATED: &str = "target_threshold_saturated";
/// Reason: a target did not improve (stage one) or did not hold its improvement
/// (full evaluation replay).
pub const REASON_TARGET_REPLAY_FAILED: &str = "candidate_failed_target_replay_checkpoint";
/// Reason: the batch gate passed; the full evaluation is still pending.
pub const REASON_BATCH_GATE_PASSED: &str = "candidate_passed_batch_gate_pending_epoch_checkpoint";
/// Reason: the full evaluation promoted the candidate.
pub const REASON_SELECTED: &str = "selected_global_non_regression";
/// Reason: the full evaluation rejected the candidate on score.
pub const REASON_SELECTED_SCORE_REGRESSED: &str = "selected_score_regressed";
/// Reason: a protected case regressed beyond `max_regression`.
pub const REASON_PROTECTED_CASE_REGRESSED: &str = "protected_case_regressed";

/// Per-case scores keyed by case id.
///
/// Only graded cases appear: an ungraded or errored case is simply absent, so a
/// caller can never mistake "not graded" for "scored zero".
pub type CaseScores = BTreeMap<String, f64>;

/// Returns the score of `scores` for `case_id`, treating a missing case as
/// `fallback`.
fn score_of(scores: &CaseScores, case_id: &str, fallback: f64) -> f64 {
    scores.get(case_id).copied().unwrap_or(fallback)
}

/// Extracts the graded per-case scores of a report.
///
/// A result contributes only when the evaluator both graded it (`score`) and
/// did not report a hard error, so failed cases stay out of every aggregate.
#[must_use]
pub fn case_scores(report: &EvaluationReport) -> CaseScores {
    report
        .results
        .iter()
        .filter(|result| result.error.is_none())
        .filter_map(|result| result.score.map(|score| (result.case_id.clone(), score)))
        .collect()
}

/// Returns whether `after_case` strictly improves on `before_case`.
///
/// The gain must exceed [`EPSILON`], so an unchanged score is never an
/// improvement.
#[must_use]
pub fn is_batch_improvement(before_case: f64, after_case: f64) -> bool {
    after_case > before_case + EPSILON
}

/// Returns the case ids whose satisfaction gap still exceeds
/// `min_improvement + EPSILON`, in ascending id order.
///
/// A case is fully satisfied at `1.0`; the gap `1.0 - score` is what a batch
/// has left to close, so a case at `0.9` with `min_improvement = 0.05` (gap
/// `0.1`) is selected while a case at `0.96` (gap `0.04`) is not.
#[must_use]
pub fn batch_filter(scores: &CaseScores, min_improvement: f64) -> Vec<String> {
    scores
        .iter()
        .filter(|(_, score)| 1.0 - **score > min_improvement + EPSILON)
        .map(|(case_id, _)| case_id.clone())
        .collect()
}

/// Returns whether `scores` is non-empty and every case is already fully
/// satisfied (`>= 1.0 - EPSILON`).
///
/// An empty map is *not* saturated: there is no evidence either way, and a
/// batch with no cases is a different problem.
#[must_use]
pub fn threshold_saturated(scores: &CaseScores) -> bool {
    !scores.is_empty() && scores.values().all(|score| *score >= 1.0 - EPSILON)
}

/// Returns the case ids that strictly improved, in ascending id order.
///
/// A case missing from `before` counts as improved (there is no frozen baseline
/// to beat); a case missing from `after` is not improved, because an ungraded
/// case is never evidence of progress.
#[must_use]
pub fn targets_improved(before: &CaseScores, after: &CaseScores) -> Vec<String> {
    after
        .iter()
        .filter(|(case_id, score)| match before.get(*case_id) {
            Some(previous) => **score > *previous + EPSILON,
            None => true,
        })
        .map(|(case_id, _)| case_id.clone())
        .collect()
}

/// Returns the mean per-target improvement between `before` and `after`.
///
/// The mean runs over [`targets_improved`], so it answers "by how much did the
/// cases that improved improve?". A case without a baseline contributes its new
/// score as the delta; an empty target set returns `0.0`.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn mean_improvement(before: &CaseScores, after: &CaseScores) -> f64 {
    let targets = targets_improved(before, after);
    if targets.is_empty() {
        return 0.0;
    }
    let total: f64 = targets
        .iter()
        .map(|case_id| score_of(after, case_id, 0.0) - score_of(before, case_id, 0.0))
        .sum();
    total / targets.len() as f64
}

/// Outcome of the stage-one (batch) gate.
#[derive(Debug, Clone, PartialEq)]
pub struct StageOneDecision {
    /// Whether the candidate deserves a full evaluation.
    pub passed: bool,
    /// The batch cases the candidate strictly improved, ascending.
    pub improved: Vec<String>,
    /// Why the gate decided this way (see the `REASON_*` literals).
    pub reason: String,
}

/// Evaluates the stage-one gate.
///
/// The rules run in this order, and the first match wins:
///
/// 1. `content_changed == false` →
///    [`REASON_CONTENT_UNCHANGED`] (an unchanged body is not a candidate);
/// 2. `engineering_failed = Some(detail)` →
///    `"engineering_failed: <detail>"` (the batch run itself failed, so no
///    score is trustworthy);
/// 3. [`threshold_saturated`] over `before` → [`REASON_THRESHOLD_SATURATED`]
///    (every batch case is already at `1.0`, so no candidate can improve);
/// 4. [`targets_improved`] is empty → [`REASON_TARGET_REPLAY_FAILED`];
/// 5. otherwise passed, with [`REASON_BATCH_GATE_PASSED`].
///
/// `before` holds the incumbent per-case scores of the batch and `after` the
/// candidate's batch scores; both may omit cases, and an omitted case follows
/// the [`targets_improved`] rules.
#[must_use]
pub fn stage_one_gate(
    before: &CaseScores,
    after: &CaseScores,
    content_changed: bool,
    engineering_failed: Option<&str>,
) -> StageOneDecision {
    if !content_changed {
        return StageOneDecision {
            passed: false,
            improved: Vec::new(),
            reason: REASON_CONTENT_UNCHANGED.to_owned(),
        };
    }
    if let Some(detail) = engineering_failed {
        return StageOneDecision {
            passed: false,
            improved: Vec::new(),
            reason: format!("{REASON_ENGINEERING_FAILED}: {detail}"),
        };
    }
    if threshold_saturated(before) {
        return StageOneDecision {
            passed: false,
            improved: Vec::new(),
            reason: REASON_THRESHOLD_SATURATED.to_owned(),
        };
    }
    let improved = targets_improved(before, after);
    if improved.is_empty() {
        return StageOneDecision {
            passed: false,
            improved,
            reason: REASON_TARGET_REPLAY_FAILED.to_owned(),
        };
    }
    StageOneDecision {
        passed: true,
        improved,
        reason: REASON_BATCH_GATE_PASSED.to_owned(),
    }
}

/// Outcome of the full (all-cases) evaluation decision.
#[derive(Debug, Clone, PartialEq)]
pub struct FullEvaluationDecision {
    /// Whether the candidate may replace the incumbent.
    pub promoted: bool,
    /// Why the decision was taken (see the `REASON_*` literals).
    pub reason: String,
    /// Improved targets that did not hold their gain, ascending.
    pub failed_targets: Vec<String>,
    /// Non-target, non-protected cases that lost more than `max_regression`,
    /// ascending.
    pub regressed_others: Vec<String>,
    /// Mean per-case delta over the union of the before/after case ids.
    ///
    /// Missing scores count as `0.0`, so a case that disappeared from the
    /// evaluation drags the mean down rather than being ignored.
    pub mean_improvement: f64,
}

/// Mean per-case delta over the union of both maps; missing scores count `0.0`.
#[must_use]
#[allow(clippy::cast_precision_loss)]
fn overall_mean_improvement(before: &CaseScores, after: &CaseScores) -> f64 {
    let ids: BTreeSet<&String> = before.keys().chain(after.keys()).collect();
    if ids.is_empty() {
        return 0.0;
    }
    let total: f64 = ids
        .iter()
        .map(|case_id| score_of(after, case_id, 0.0) - score_of(before, case_id, 0.0))
        .sum();
    total / ids.len() as f64
}

/// Mean per-case delta over `cases` (ids absent from both maps are ignored).
#[must_use]
#[allow(clippy::cast_precision_loss)]
fn mean_delta_over(cases: &BTreeSet<String>, before: &CaseScores, after: &CaseScores) -> f64 {
    if cases.is_empty() {
        return 0.0;
    }
    let total: f64 = cases
        .iter()
        .map(|case_id| score_of(after, case_id, 0.0) - score_of(before, case_id, 0.0))
        .sum();
    total / cases.len() as f64
}

/// Evaluates the second, full-evaluation gate.
///
/// `before` is the incumbent's frozen per-case scores, `after` the candidate's
/// full re-evaluation, `improved_targets` the stage-one target set,
/// `protected_cases` the cases that must never regress, `max_regression` the
/// allowed per-case loss and `min_improvement` the batch margin (recorded as a
/// diagnostic only — the batch gate already enforced it, and the full
/// evaluation requires only a strictly positive aggregate gain).
///
/// The rules run in this order, and the first match wins:
///
/// 1. **protected regression** — some `protected_cases` entry loses more than
///    `max_regression + EPSILON` (a protected case missing from `after` counts
///    as `0.0`, i.e. a total loss; a case missing from `before` has nothing to
///    lose) → [`REASON_PROTECTED_CASE_REGRESSED`];
/// 2. **target replay** — `improved_targets` is empty, or some improved target
///    now scores `<= before + EPSILON`, or vanished from `after` →
///    [`REASON_TARGET_REPLAY_FAILED`];
/// 3. **score regression** — the mean delta of the non-target, non-protected
///    cases is not strictly greater than `-max_regression`, or the overall mean
///    delta (rule 4 of the objective, the union of both maps) is not strictly
///    greater than [`EPSILON`] → [`REASON_SELECTED_SCORE_REGRESSED`];
/// 4. otherwise promoted with [`REASON_SELECTED`].
///
/// `regressed_others` reports rule 1's non-protected siblings: non-target,
/// non-protected cases that individually lost more than
/// `max_regression + EPSILON`. It is diagnostic; rule 3 is what decides.
#[must_use]
pub fn full_evaluation_decision(
    before: &CaseScores,
    after: &CaseScores,
    improved_targets: &[String],
    protected_cases: &[String],
    min_improvement: f64,
    max_regression: f64,
) -> FullEvaluationDecision {
    let targets: BTreeSet<&String> = improved_targets.iter().collect();
    let protected: BTreeSet<&String> = protected_cases.iter().collect();

    let regressed_protected: Vec<String> = protected
        .iter()
        .filter(|case_id| {
            score_of(before, case_id, 0.0) - score_of(after, case_id, 0.0)
                > max_regression + EPSILON
        })
        .map(|case_id| (*case_id).clone())
        .collect();

    let mut failed_targets: Vec<String> = improved_targets
        .iter()
        .filter(|case_id| match after.get(*case_id) {
            Some(score) => *score <= score_of(before, case_id, 0.0) + EPSILON,
            None => true,
        })
        .cloned()
        .collect();
    failed_targets.sort();
    failed_targets.dedup();

    let others: BTreeSet<String> = before
        .keys()
        .chain(after.keys())
        .filter(|case_id| !targets.contains(case_id) && !protected.contains(case_id))
        .cloned()
        .collect();
    let mut regressed_others: Vec<String> = others
        .iter()
        .filter(|case_id| {
            score_of(before, case_id, 0.0) - score_of(after, case_id, 0.0)
                > max_regression + EPSILON
        })
        .cloned()
        .collect();
    regressed_others.sort();

    let retention_mean = mean_delta_over(&others, before, after);
    let mean = overall_mean_improvement(before, after);
    debug!(
        target: LOG_TARGET,
        targets = improved_targets.len(),
        protected = protected_cases.len(),
        min_improvement,
        max_regression,
        retention_mean,
        mean_improvement = mean,
        failed_targets = failed_targets.len(),
        regressed_protected = regressed_protected.len(),
        regressed_others = regressed_others.len(),
        "full evaluation decision"
    );

    let reason = if !regressed_protected.is_empty() {
        REASON_PROTECTED_CASE_REGRESSED
    } else if failed_targets.is_empty() && !improved_targets.is_empty() {
        if retention_mean > -max_regression && mean > EPSILON {
            REASON_SELECTED
        } else {
            REASON_SELECTED_SCORE_REGRESSED
        }
    } else {
        REASON_TARGET_REPLAY_FAILED
    };

    FullEvaluationDecision {
        promoted: reason == REASON_SELECTED,
        reason: reason.to_owned(),
        failed_targets,
        regressed_others,
        mean_improvement: mean,
    }
}
