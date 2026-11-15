//! [`AdoptService`]: drift, convergence, the policy, and memory in one call.
//!
//! Python's `AdoptOptimizer` runs a whole pipeline (node conclusion → global
//! gradient → per-node gradient → rewrite) and leaves the "should this be kept?"
//! question unanswered. [`AdoptService::decide`] answers exactly that question
//! for one already-tuned candidate: it measures, applies every gate, and — when
//! the verdict is *adopt* — archives the prompt through
//! [`AdoptMemory`](super::AdoptMemory).
//!
//! The composition is deliberately thin, because every part already exists in
//! the crate's prompt-search layer: the deviation comes from an injected
//! [`DriftJudge`], the convergence verdict from an injected
//! [`ConvergenceDetector`], and the archive from an injected [`PromptMemory`].
//! Nothing is re-implemented here, and no stage reads the clock — `now` is a
//! parameter.

use std::collections::BTreeMap;
use std::sync::Arc;

use openjiuwen_common::error::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::tune::prompt_search::{
    candidate_id, clamp01, ConvergenceDetector, ConvergenceState, DriftJudge, NullDriftJudge,
    PromptMemory, PromptRecord,
};
use crate::tune::{PromptCandidate, PromptTaskSpec};

use super::{AdoptConfig, AdoptDecision, AdoptMemory, AdoptPolicy};

/// The result of one adoption check.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct AdoptOutcome {
    /// The policy's verdict and its reason.
    pub decision: AdoptDecision,
    /// The deviation the drift judge reported, in `0.0..=1.0`.
    pub drift: f64,
    /// The mean of the finite case scores, in `0.0..=1.0`.
    pub correctness: f64,
    /// What the convergence detector made of the reward history, when one is
    /// configured and it converged.
    pub convergence: Option<ConvergenceState>,
    /// The archived record, present exactly when the candidate was adopted.
    pub record: Option<PromptRecord>,
}

impl AdoptOutcome {
    /// Whether the candidate was adopted.
    #[must_use]
    pub fn adopted(&self) -> bool {
        self.decision.adopted()
    }
}

/// One adoption check: measure, gate, decide, and archive.
pub struct AdoptService {
    /// The pure decision layer.
    policy: AdoptPolicy,
    /// The semantic-drift measurement.
    drift_judge: Arc<dyn DriftJudge>,
    /// The optional convergence detector.
    convergence: Option<ConvergenceDetector>,
    /// The run's bookkeeping plus the persistent store behind it.
    memory: AdoptMemory,
    /// A caller-supplied characteristics string for archived records.
    characteristics: Option<String>,
}

impl std::fmt::Debug for AdoptService {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The judge is a trait object without `Debug`, so it is reported by name.
        formatter
            .debug_struct("AdoptService")
            .field("policy", &self.policy)
            .field("drift_judge", &"<dyn DriftJudge>")
            .field("convergence", &self.convergence)
            .field("memory", &self.memory)
            .field("characteristics", &self.characteristics)
            .finish()
    }
}

impl Default for AdoptService {
    /// A service with the default policy, no drift judge, no convergence
    /// detector, and no persistent memory.
    fn default() -> Self {
        Self::new(AdoptPolicy::new())
    }
}

impl AdoptService {
    /// Creates a service applying `policy` with every seam at its default.
    #[must_use]
    pub fn new(policy: AdoptPolicy) -> Self {
        Self {
            policy,
            drift_judge: Arc::new(NullDriftJudge::new()),
            convergence: None,
            memory: AdoptMemory::new(),
            characteristics: None,
        }
    }

    /// Replaces the drift judge (default [`NullDriftJudge`]).
    #[must_use]
    pub fn with_drift_judge(mut self, drift_judge: Arc<dyn DriftJudge>) -> Self {
        self.drift_judge = drift_judge;
        self
    }

    /// Enables convergence reporting over the reward history.
    #[must_use]
    pub fn with_convergence(mut self, convergence: ConvergenceDetector) -> Self {
        self.convergence = Some(convergence);
        self
    }

    /// Replaces the persistent prompt memory (default [`NullPromptMemory`]).
    ///
    /// [`NullPromptMemory`]: crate::tune::prompt_search::NullPromptMemory
    #[must_use]
    pub fn with_memory(mut self, memory: Arc<dyn PromptMemory>) -> Self {
        self.memory = AdoptMemory::with_memory(memory);
        self
    }

    /// Replaces the characteristics string archived records carry.
    ///
    /// When unset, [`PromptTaskSpec::characteristics`] supplies it.
    #[must_use]
    pub fn with_characteristics(mut self, characteristics: impl Into<String>) -> Self {
        self.characteristics = Some(characteristics.into());
        self
    }

    /// The policy in force.
    #[must_use]
    pub fn policy(&self) -> &AdoptPolicy {
        &self.policy
    }

    /// The thresholds in force.
    #[must_use]
    pub fn config(&self) -> &AdoptConfig {
        self.policy.config()
    }

    /// The drift judge in force.
    #[must_use]
    pub fn drift_judge(&self) -> &Arc<dyn DriftJudge> {
        &self.drift_judge
    }

    /// The convergence detector in force, if any.
    #[must_use]
    pub fn convergence(&self) -> Option<&ConvergenceDetector> {
        self.convergence.as_ref()
    }

    /// The run's adoption bookkeeping.
    #[must_use]
    pub fn memory(&self) -> &AdoptMemory {
        &self.memory
    }

    /// The characteristics override, if one was supplied.
    #[must_use]
    pub fn characteristics(&self) -> Option<&str> {
        self.characteristics.as_deref()
    }

    /// Decides whether `candidate` should replace the best of `history` for
    /// `task`, archiving it when the verdict is *adopt*.
    ///
    /// The measurements are taken in this order:
    ///
    /// 1. **correctness** — the mean of the finite values in `case_scores`, or
    ///    `0.0` when the slice holds none.
    /// 2. **drift** — the injected [`DriftJudge`]'s deviation from
    ///    [`PromptTaskSpec::objective`]. The judge is skipped (and the deviation
    ///    reported as `0.0`) when [`AdoptConfig::max_drift`] is `1.0`, because
    ///    then no deviation can fail the gate and no measurement is needed.
    /// 3. **convergence** — [`ConvergenceDetector::check`] over `history`, when a
    ///    detector is configured; it is reporting only and never influences the
    ///    verdict.
    /// 4. **decision** — [`AdoptPolicy::decide`].
    ///
    /// Only an adopted candidate is archived: a [`PromptRecord`] is built with
    /// `reward`, `drift`, and `correctness` in its metrics, `baseline_reward`
    /// from [`AdoptMemory::previous_best`], `created_at = now`, and the task's
    /// characteristics (or the [`AdoptService::with_characteristics`] override),
    /// then passed to [`AdoptMemory::record_adopted`] so it is marked applied at
    /// `now`. A rejected candidate persists nothing.
    ///
    /// # Errors
    ///
    /// Nothing on the deterministic path: a memory store failure is logged and
    /// swallowed by [`AdoptMemory`], and the drift judge reports a value rather
    /// than an error. The `Result` is part of the contract so a caller can later
    /// inject a measuring seam that fails.
    pub async fn decide(
        &self,
        task: &PromptTaskSpec,
        candidate: &PromptCandidate,
        reward: f64,
        case_scores: &[f64],
        history: &[f64],
        now: f64,
    ) -> Result<AdoptOutcome> {
        let correctness = mean_of(case_scores);
        let drift = if self.config().max_drift >= 1.0 {
            0.0
        } else {
            clamp01(self.drift_judge.deviation(&task.objective, candidate).await)
        };
        let convergence = self
            .convergence
            .as_ref()
            .and_then(|detector| detector.check(history));
        let decision = self.policy.decide(reward, correctness, drift, history);
        let record = if decision.adopted() {
            Some(self.archive(task, candidate, reward, correctness, drift, &decision, now))
        } else {
            None
        };
        Ok(AdoptOutcome {
            decision,
            drift,
            correctness,
            convergence,
            record,
        })
    }

    /// Builds and stores the record of an adopted candidate.
    fn archive(
        &self,
        task: &PromptTaskSpec,
        candidate: &PromptCandidate,
        reward: f64,
        correctness: f64,
        drift: f64,
        decision: &AdoptDecision,
        now: f64,
    ) -> PromptRecord {
        let baseline = self.memory.previous_best(&task.objective);
        let characteristics = self
            .characteristics
            .clone()
            .unwrap_or_else(|| task.characteristics());
        let mut record = PromptRecord::new(
            record_id(task, candidate),
            candidate.prompt.clone(),
            clamp01(reward),
            task.objective.clone(),
        )
        .with_characteristics(characteristics)
        .with_metrics(metrics(reward, correctness, drift, decision))
        .with_observations(vec![decision.describe()])
        .with_metadata(metadata(task, candidate))
        .with_created_at(now);
        if let Some(baseline) = baseline {
            record = record.with_baseline_reward(baseline);
        }
        self.memory.record_adopted(record, now)
    }
}

/// The mean of the finite values in `scores` (`0.0` when there are none).
fn mean_of(scores: &[f64]) -> f64 {
    let mut count = 0usize;
    let mut sum = 0.0;
    for score in scores {
        if score.is_finite() {
            sum += score;
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        clamp01(sum / count as f64)
    }
}

/// A deterministic record identifier derived from the candidate.
///
/// The candidate's own identifier when it has one, otherwise
/// [`candidate_id`] of its prompt; either way prefixed with `adopt-` so an
/// archived record cannot collide with a prompt-search record.
fn record_id(task: &PromptTaskSpec, candidate: &PromptCandidate) -> String {
    let name = if candidate.candidate_id.trim().is_empty() {
        let prompt = if candidate.prompt.is_empty() {
            &task.base_prompt
        } else {
            &candidate.prompt
        };
        candidate_id(prompt, 0)
    } else {
        candidate.candidate_id.clone()
    };
    format!("adopt-{name}")
}

/// The reward components written into the archived record.
fn metrics(
    reward: f64,
    correctness: f64,
    drift: f64,
    decision: &AdoptDecision,
) -> BTreeMap<String, Value> {
    BTreeMap::from([
        ("reward".to_owned(), Value::from(clamp01(reward))),
        ("correctness".to_owned(), Value::from(correctness)),
        ("drift".to_owned(), Value::from(drift)),
        ("gain".to_owned(), Value::from(decision.gain())),
    ])
}

/// The task metadata plus the candidate provenance written into the record.
fn metadata(task: &PromptTaskSpec, candidate: &PromptCandidate) -> BTreeMap<String, Value> {
    let mut metadata = task.metadata.clone();
    metadata.insert(
        "candidate_id".to_owned(),
        Value::from(candidate.candidate_id.clone()),
    );
    if !candidate.rationale.is_empty() {
        metadata.insert(
            "rationale".to_owned(),
            Value::from(candidate.rationale.clone()),
        );
    }
    metadata
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn correctness_is_the_mean_of_the_finite_scores() {
        assert!(mean_of(&[]).abs() < f64::EPSILON);
        assert!((mean_of(&[1.0, 0.0]) - 0.5).abs() < 1e-12);
        assert!((mean_of(&[0.5, f64::NAN, f64::INFINITY]) - 0.5).abs() < 1e-12);
        assert!(mean_of(&[f64::NAN]).abs() < f64::EPSILON);
        // Out-of-range values cannot push the mean outside the reward range.
        assert!((mean_of(&[5.0, 5.0]) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn record_ids_are_deterministic_and_prefixed() {
        let task = PromptTaskSpec::new("objective").with_base_prompt("base");
        let named = PromptCandidate::new("prompt", "seed").with_id("abc");
        assert_eq!(record_id(&task, &named), "adopt-abc");
        let derived = record_id(&task, &PromptCandidate::new("prompt", "seed"));
        assert_eq!(
            derived,
            record_id(&task, &PromptCandidate::new("prompt", "seed"))
        );
        assert!(derived.starts_with("adopt-"), "{derived}");
        let blank = record_id(&task, &PromptCandidate::default());
        assert_eq!(blank, format!("adopt-{}", candidate_id("base", 0)));
    }

    #[test]
    fn outcome_defaults_to_a_rejected_first_candidate() {
        let outcome = AdoptOutcome::default();
        assert!(!outcome.adopted());
        assert_eq!(outcome.drift, 0.0);
        assert_eq!(outcome.correctness, 0.0);
        assert!(outcome.convergence.is_none());
        assert!(outcome.record.is_none());
    }
}
