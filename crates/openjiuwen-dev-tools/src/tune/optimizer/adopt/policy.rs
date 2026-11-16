//! The adoption policy: whether a tuned prompt is worth installing.
//!
//! Ports the *decision* half of Python's
//! `optimizer/adopt/adopt_optimizer.py` — the moment where a tuned prompt is
//! either adopted (installed) or left alone — as a pure, synchronous value.
//! [`AdoptConfig`] holds the thresholds, [`AdoptPolicy::decide`] compares one
//! candidate against the recorded reward history, and [`AdoptReason`] explains
//! the verdict.
//!
//! Python has no such policy: `AdoptOptimizer` rewrites every `LLMCall` in
//! place and never asks whether the rewrite is worth keeping. The Rust port
//! keeps the deterministic part (a candidate only replaces a prompt when it is
//! measurably better, faithful to the objective, and correct enough) and leaves
//! the model-driven gradient work to the crate's prompt-search seams.
//!
//! Everything here is deterministic: no clock, no I/O, no randomness, no
//! `async`. The same inputs always produce the same decision, which makes a run
//! reproducible and the policy testable without fixtures.

use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};

use crate::tune::prompt_search::clamp01;

/// Python `DEFAULT_BAD_CASES_SAMPLE_NUM`'s neighbourhood: the default minimum
/// gain a candidate must beat to be adopted.
pub const DEFAULT_ADOPT_MIN_GAIN: f64 = 0.01;

/// The default maximum semantic drift a candidate may show.
pub const DEFAULT_ADOPT_MAX_DRIFT: f64 = 0.25;

/// The default correctness floor a candidate must clear.
pub const DEFAULT_ADOPT_MIN_CORRECTNESS: f64 = 0.5;

/// The default number of historical rewards needed to compare at all.
pub const DEFAULT_ADOPT_MIN_HISTORY: usize = 1;

/// The thresholds of the adoption policy.
///
/// Every field defaults to the value documented on its constant, so
/// [`AdoptConfig::default`] is the recommended configuration.
/// Deserialization is permissive (missing fields default, unknown fields are
/// ignored); only [`AdoptConfig::validate`] enforces the bounds.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct AdoptConfig {
    /// The minimum gain over the best recorded reward that counts as an
    /// improvement.
    pub min_gain: f64,
    /// The maximum semantic deviation from the objective that is tolerated.
    pub max_drift: f64,
    /// The correctness floor: a candidate below it is never adopted.
    pub min_correctness: f64,
    /// How many historical rewards are needed before a comparison is trusted.
    pub min_history: usize,
    /// Whether an improvement is required at all.
    ///
    /// When `false`, a candidate that clears the correctness and drift gates is
    /// adopted even without a gain; the [`AdoptReason`] still reports the
    /// measured gain so the caller can see that nothing improved.
    pub require_improvement: bool,
}

impl Default for AdoptConfig {
    fn default() -> Self {
        Self {
            min_gain: DEFAULT_ADOPT_MIN_GAIN,
            max_drift: DEFAULT_ADOPT_MAX_DRIFT,
            min_correctness: DEFAULT_ADOPT_MIN_CORRECTNESS,
            min_history: DEFAULT_ADOPT_MIN_HISTORY,
            require_improvement: true,
        }
    }
}

impl AdoptConfig {
    /// Creates the default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Replaces the minimum gain.
    #[must_use]
    pub fn with_min_gain(mut self, min_gain: f64) -> Self {
        self.min_gain = min_gain;
        self
    }

    /// Replaces the maximum tolerated drift.
    #[must_use]
    pub fn with_max_drift(mut self, max_drift: f64) -> Self {
        self.max_drift = max_drift;
        self
    }

    /// Replaces the correctness floor.
    #[must_use]
    pub fn with_min_correctness(mut self, min_correctness: f64) -> Self {
        self.min_correctness = min_correctness;
        self
    }

    /// Replaces the required history depth.
    #[must_use]
    pub fn with_min_history(mut self, min_history: usize) -> Self {
        self.min_history = min_history;
        self
    }

    /// Sets whether an improvement is required for adoption.
    #[must_use]
    pub fn with_require_improvement(mut self, require_improvement: bool) -> Self {
        self.require_improvement = require_improvement;
        self
    }

    /// Checks every threshold.
    ///
    /// `min_gain`, `max_drift`, and `min_correctness` must be finite fractions
    /// in `0.0..=1.0`. A `min_gain` or `max_drift` above `1.0` is rejected with
    /// its own message because it can never be met: a gain is the difference of
    /// two `0.0..=1.0` rewards and a deviation is itself a fraction, so the
    /// threshold would silently disable the check it configures.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] naming the offending field, its value, and the
    /// allowed range.
    pub fn validate(&self) -> Result<()> {
        for (field, value) in self.fractions() {
            if !value.is_finite() {
                return Err(Error::validation(format!(
                    "AdoptConfig.{field} must be finite, got {value}"
                )));
            }
        }
        if self.min_gain > 1.0 {
            return Err(Error::validation(format!(
                "AdoptConfig.min_gain must not exceed 1.0, got {}: a gain is the difference of two \
                 rewards in 0.0..=1.0",
                self.min_gain
            )));
        }
        if self.max_drift > 1.0 {
            return Err(Error::validation(format!(
                "AdoptConfig.max_drift must not exceed 1.0, got {}: a deviation is a fraction in \
                 0.0..=1.0",
                self.max_drift
            )));
        }
        for (field, value) in self.fractions() {
            if !(0.0..=1.0).contains(&value) {
                return Err(Error::validation(format!(
                    "AdoptConfig.{field} must be within 0.0..=1.0, got {value}"
                )));
            }
        }
        Ok(())
    }

    /// The three fractional thresholds paired with their field names.
    fn fractions(&self) -> [(&'static str, f64); 3] {
        [
            ("min_gain", self.min_gain),
            ("max_drift", self.max_drift),
            ("min_correctness", self.min_correctness),
        ]
    }
}

/// Why the policy reached its verdict.
///
/// The variants cover the whole evaluation order, so a rejected candidate says
/// which gate rejected it and an adopted one says what justified it.
/// `Display` renders the same sentence [`AdoptDecision::describe`] uses.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AdoptReason {
    /// There was no comparable history, so the candidate is the first one.
    FirstCandidate,
    /// The candidate beat the best recorded reward by more than `min_gain`.
    Improved {
        /// The gain over the best recorded reward.
        gain: f64,
    },
    /// The candidate did not beat the best recorded reward by `min_gain`.
    NoImprovement {
        /// The (possibly negative) gain over the best recorded reward.
        gain: f64,
    },
    /// The candidate drifted too far from the objective.
    Drift {
        /// The measured deviation, in `0.0..=1.0`.
        deviation: f64,
    },
    /// The candidate was not correct enough to be considered.
    InsufficientCorrectness {
        /// The measured correctness, in `0.0..=1.0`.
        score: f64,
    },
    /// Too few historical rewards to trust a comparison.
    InsufficientHistory {
        /// How many comparable rewards the history held.
        rounds: usize,
    },
}

impl Default for AdoptReason {
    /// The reason a decision defaults to: nothing has been compared yet.
    fn default() -> Self {
        Self::FirstCandidate
    }
}

impl std::fmt::Display for AdoptReason {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FirstCandidate => write!(formatter, "no comparable history: first candidate"),
            Self::Improved { gain } => {
                write!(
                    formatter,
                    "improved by {gain:.3} over the best recorded reward"
                )
            }
            Self::NoImprovement { gain } => {
                write!(
                    formatter,
                    "gain {gain:.3} over the best recorded reward is too small"
                )
            }
            Self::Drift { deviation } => {
                write!(
                    formatter,
                    "semantic drift {deviation:.3} exceeds the allowance"
                )
            }
            Self::InsufficientCorrectness { score } => {
                write!(formatter, "correctness {score:.3} is below the floor")
            }
            Self::InsufficientHistory { rounds } => {
                write!(formatter, "only {rounds} historical rewards are comparable")
            }
        }
    }
}

/// The verdict of an adoption check.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct AdoptDecision {
    /// Whether the candidate should be installed.
    pub adopted: bool,
    /// Why.
    pub reason: AdoptReason,
}

impl Default for AdoptDecision {
    /// A default decision adopts nothing and compares nothing.
    fn default() -> Self {
        Self {
            adopted: false,
            reason: AdoptReason::FirstCandidate,
        }
    }
}

impl AdoptDecision {
    /// Creates a decision from its two parts.
    #[must_use]
    pub fn new(adopted: bool, reason: AdoptReason) -> Self {
        Self { adopted, reason }
    }

    /// Whether the candidate should be installed.
    #[must_use]
    pub fn adopted(&self) -> bool {
        self.adopted
    }

    /// Why the policy reached its verdict.
    #[must_use]
    pub fn reason(&self) -> &AdoptReason {
        &self.reason
    }

    /// The measured gain, for the two comparison reasons (`0.0` otherwise).
    #[must_use]
    pub fn gain(&self) -> f64 {
        match self.reason {
            AdoptReason::Improved { gain } | AdoptReason::NoImprovement { gain } => gain,
            _ => 0.0,
        }
    }

    /// A one-line human-readable rendering, e.g. `adopted: improved by 0.050 …`.
    #[must_use]
    pub fn describe(&self) -> String {
        let verdict = if self.adopted { "adopted" } else { "rejected" };
        format!("{verdict}: {}", self.reason)
    }
}

/// The adoption policy: a pure function from measurements to a verdict.
///
/// The comparison baseline is the largest finite value in the caller's history;
/// a history with no finite value counts as *no history at all*.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct AdoptPolicy {
    /// The thresholds this policy applies.
    config: AdoptConfig,
}

impl AdoptPolicy {
    /// Creates the policy with the default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Replaces the configuration.
    #[must_use]
    pub fn with_config(mut self, config: AdoptConfig) -> Self {
        self.config = config;
        self
    }

    /// The configuration in force.
    #[must_use]
    pub fn config(&self) -> &AdoptConfig {
        &self.config
    }

    /// Decides whether a candidate measured by `candidate_reward`,
    /// `correctness`, and `drift` should replace the best of `history`.
    ///
    /// The checks run in this order, and the first one that fires decides:
    ///
    /// 1. **correctness gate** — `correctness < min_correctness` is rejected as
    ///    [`AdoptReason::InsufficientCorrectness`]; an incorrect prompt cannot
    ///    be adopted at any reward.
    /// 2. **drift gate** — `drift > max_drift` is rejected as
    ///    [`AdoptReason::Drift`], so a candidate that repurposed the task never
    ///    wins on score alone.
    /// 3. **history** — a history without any finite value is adopted as
    ///    [`AdoptReason::FirstCandidate`] (there is nothing to compare against,
    ///    so the first candidate is the baseline); a shorter-than-`min_history`
    ///    history is rejected as [`AdoptReason::InsufficientHistory`].
    /// 4. **improvement** — `candidate_reward - best(history) > min_gain` is
    ///    adopted as [`AdoptReason::Improved`], otherwise rejected as
    ///    [`AdoptReason::NoImprovement`] unless `require_improvement` is off, in
    ///    which case the candidate is adopted and the small gain is still
    ///    reported.
    ///
    /// Non-finite inputs are normalised so the function is total: a non-finite
    /// reward or correctness counts as `0.0` (it cannot win) and a non-finite
    /// drift counts as `1.0` (it is rejected). All values are clamped into
    /// `0.0..=1.0` first, so a caller-supplied out-of-range score cannot make
    /// the comparison depend on garbage.
    #[must_use]
    pub fn decide(
        &self,
        candidate_reward: f64,
        correctness: f64,
        drift: f64,
        history: &[f64],
    ) -> AdoptDecision {
        let reward = clamp01(candidate_reward);
        let correctness = clamp01(correctness);
        let drift = if drift.is_finite() {
            clamp01(drift)
        } else {
            1.0
        };
        if correctness < self.config.min_correctness {
            return AdoptDecision::new(
                false,
                AdoptReason::InsufficientCorrectness { score: correctness },
            );
        }
        if drift > self.config.max_drift {
            return AdoptDecision::new(false, AdoptReason::Drift { deviation: drift });
        }
        let baseline = best_of(history);
        let Some(baseline) = baseline else {
            return AdoptDecision::new(true, AdoptReason::FirstCandidate);
        };
        if history.len() < self.config.min_history {
            return AdoptDecision::new(
                false,
                AdoptReason::InsufficientHistory {
                    rounds: history.len(),
                },
            );
        }
        let gain = reward - baseline;
        if gain > self.config.min_gain {
            return AdoptDecision::new(true, AdoptReason::Improved { gain });
        }
        AdoptDecision::new(
            !self.config.require_improvement,
            AdoptReason::NoImprovement { gain },
        )
    }
}

/// The largest finite value in `history`, or `None` when there is none.
fn best_of(history: &[f64]) -> Option<f64> {
    let mut best: Option<f64> = None;
    for value in history {
        if !value.is_finite() {
            continue;
        }
        best = Some(match best {
            Some(current) => current.max(*value),
            None => *value,
        });
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use openjiuwen_common::error::ErrorKind;

    #[test]
    fn defaults_match_the_documented_constants() {
        let config = AdoptConfig::default();
        assert!((config.min_gain - DEFAULT_ADOPT_MIN_GAIN).abs() < f64::EPSILON);
        assert!((config.max_drift - DEFAULT_ADOPT_MAX_DRIFT).abs() < f64::EPSILON);
        assert!((config.min_correctness - DEFAULT_ADOPT_MIN_CORRECTNESS).abs() < f64::EPSILON);
        assert_eq!(config.min_history, DEFAULT_ADOPT_MIN_HISTORY);
        assert!(config.require_improvement);
        assert!(config.validate().is_ok());
        assert_eq!(AdoptConfig::new(), config);
        assert_eq!(AdoptConfig::default(), config);
    }

    #[test]
    fn builders_replace_every_field() {
        let config = AdoptConfig::new()
            .with_min_gain(0.0)
            .with_max_drift(1.0)
            .with_min_correctness(0.0)
            .with_min_history(0)
            .with_require_improvement(false);
        assert!(config.min_gain.abs() < f64::EPSILON);
        assert!((config.max_drift - 1.0).abs() < f64::EPSILON);
        assert!(config.min_correctness.abs() < f64::EPSILON);
        assert_eq!(config.min_history, 0);
        assert!(!config.require_improvement);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn validate_rejects_non_finite_and_out_of_range_fractions() {
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            for config in [
                AdoptConfig::new().with_min_gain(value),
                AdoptConfig::new().with_max_drift(value),
                AdoptConfig::new().with_min_correctness(value),
            ] {
                let error = config.validate().unwrap_err();
                assert_eq!(error.kind(), ErrorKind::Validation);
                assert!(error.to_string().contains("must be finite"), "{}", error);
            }
        }
        for (field, config) in [
            ("min_gain", AdoptConfig::new().with_min_gain(-0.1)),
            ("max_drift", AdoptConfig::new().with_max_drift(-0.1)),
            (
                "min_correctness",
                AdoptConfig::new().with_min_correctness(1.5),
            ),
        ] {
            let error = config.validate().unwrap_err();
            assert_eq!(error.kind(), ErrorKind::Validation);
            assert!(error.to_string().contains(field), "{field}: {error}");
            assert!(
                error.to_string().contains("must be within 0.0..=1.0"),
                "{field}: {error}"
            );
        }
    }

    #[test]
    fn validate_names_the_field_for_impossible_thresholds() {
        for (field, config) in [
            ("min_gain", AdoptConfig::new().with_min_gain(1.5)),
            ("max_drift", AdoptConfig::new().with_max_drift(2.0)),
        ] {
            let error = config.validate().unwrap_err();
            assert_eq!(error.kind(), ErrorKind::Validation);
            assert!(error.to_string().contains(field), "{field}: {error}");
            assert!(
                error.to_string().contains("must not exceed 1.0"),
                "{field}: {error}"
            );
        }
    }

    #[test]
    fn config_round_trips_through_json_and_tolerates_gaps() {
        let config = AdoptConfig::new()
            .with_min_gain(0.05)
            .with_max_drift(0.5)
            .with_min_correctness(0.75)
            .with_min_history(2)
            .with_require_improvement(false);
        let value = serde_json::to_value(config).unwrap();
        for key in [
            "min_gain",
            "max_drift",
            "min_correctness",
            "min_history",
            "require_improvement",
        ] {
            assert!(value.get(key).is_some(), "missing field {key}");
        }
        let restored: AdoptConfig = serde_json::from_value(value).unwrap();
        assert_eq!(restored, config);
        let partial: AdoptConfig =
            serde_json::from_str(r#"{"min_gain": 0.2, "unknown": true}"#).unwrap();
        assert!((partial.min_gain - 0.2).abs() < f64::EPSILON);
        assert!((partial.max_drift - DEFAULT_ADOPT_MAX_DRIFT).abs() < f64::EPSILON);
        assert!(partial.require_improvement);
        assert_eq!(AdoptConfig::default(), AdoptConfig::new());
    }

    #[test]
    fn gains_above_the_threshold_are_adopted() {
        let decision = AdoptPolicy::new().decide(0.8, 0.9, 0.0, &[0.5, 0.7]);
        assert!(decision.adopted());
        assert_eq!(
            decision.reason(),
            &AdoptReason::Improved { gain: 0.8 - 0.7 }
        );
        assert!((decision.gain() - 0.1).abs() < 1e-12);
        assert!(decision.describe().starts_with("adopted: improved by"));
    }

    #[test]
    fn a_gain_at_the_threshold_is_not_an_improvement() {
        let policy = AdoptPolicy::new().with_config(AdoptConfig::new().with_min_gain(0.1));
        let decision = policy.decide(0.6, 1.0, 0.0, &[0.5]);
        assert!(!decision.adopted());
        // Floating-point subtraction keeps this just under the threshold.
        match decision.reason() {
            AdoptReason::NoImprovement { gain } => {
                assert!((*gain - 0.1).abs() < 1e-12, "{gain}");
                assert!(*gain <= 0.1);
            }
            other => panic!("unexpected reason: {other}"),
        }
        assert!(decision.describe().starts_with("rejected: gain"));
    }

    #[test]
    fn a_lower_reward_is_rejected_with_a_negative_gain() {
        let decision = AdoptPolicy::new().decide(0.2, 1.0, 0.0, &[0.9]);
        assert!(!decision.adopted());
        match decision.reason() {
            AdoptReason::NoImprovement { gain } => assert!(*gain < 0.0),
            other => panic!("unexpected reason: {other}"),
        }
    }

    #[test]
    fn first_candidate_without_comparable_history() {
        let policy = AdoptPolicy::new();
        for history in [&[][..], &[f64::NAN][..], &[f64::NEG_INFINITY][..]] {
            let decision = policy.decide(0.1, 0.9, 0.0, history);
            assert!(decision.adopted());
            assert_eq!(decision.reason(), &AdoptReason::FirstCandidate);
            assert!(decision.gain().abs() < f64::EPSILON);
            assert_eq!(
                decision.describe(),
                "adopted: no comparable history: first candidate"
            );
        }
    }

    #[test]
    fn correctness_gate_beats_drift_and_improvement() {
        // Below the floor, drifting, and a huge gain: still rejected.
        let decision = AdoptPolicy::new().decide(1.0, 0.1, 1.0, &[0.0]);
        assert!(!decision.adopted());
        assert_eq!(
            decision.reason(),
            &AdoptReason::InsufficientCorrectness { score: 0.1 }
        );
        assert!(decision
            .describe()
            .contains("correctness 0.100 is below the floor"));
    }

    #[test]
    fn drift_gate_beats_improvement() {
        let decision = AdoptPolicy::new().decide(1.0, 1.0, 0.75, &[0.0]);
        assert!(!decision.adopted());
        assert_eq!(decision.reason(), &AdoptReason::Drift { deviation: 0.75 });
        assert!(decision.describe().contains("semantic drift"));
        // A deviation exactly at the allowance passes the gate.
        let allowed = AdoptPolicy::new().with_config(AdoptConfig::new().with_max_drift(0.75));
        assert!(allowed.decide(1.0, 1.0, 0.75, &[0.0]).adopted());
    }

    #[test]
    fn history_gate_fires_between_first_candidate_and_improvement() {
        let policy = AdoptPolicy::new().with_config(AdoptConfig::new().with_min_history(2));
        let decision = policy.decide(1.0, 1.0, 0.0, &[0.1]);
        assert!(!decision.adopted());
        assert_eq!(
            decision.reason(),
            &AdoptReason::InsufficientHistory { rounds: 1 }
        );
        assert!(decision.describe().contains("only 1 historical rewards"));
        // With enough history the same candidate is adopted.
        let decision = policy.decide(1.0, 1.0, 0.0, &[0.1, 0.2]);
        assert!(decision.adopted());
    }

    #[test]
    fn require_improvement_off_adopts_without_a_gain() {
        let policy =
            AdoptPolicy::new().with_config(AdoptConfig::new().with_require_improvement(false));
        let decision = policy.decide(0.5, 1.0, 0.0, &[0.5]);
        assert!(decision.adopted());
        assert_eq!(decision.reason(), &AdoptReason::NoImprovement { gain: 0.0 });
        // The gates still apply.
        let gated = policy.decide(0.5, 0.1, 0.0, &[0.5]);
        assert!(!gated.adopted());
    }

    #[test]
    fn decide_is_deterministic() {
        let policy = AdoptPolicy::new();
        let history = [0.1, 0.9, 0.3];
        for (reward, correctness, drift) in [
            (0.95, 0.9, 0.0),
            (0.2, 0.9, 0.0),
            (0.95, 0.1, 0.0),
            (0.95, 0.9, 0.9),
        ] {
            let first = policy.decide(reward, correctness, drift, &history);
            let second = policy.decide(reward, correctness, drift, &history);
            assert_eq!(first, second);
        }
    }

    #[test]
    fn non_finite_inputs_are_normalised() {
        let policy = AdoptPolicy::new();
        // A non-finite reward cannot win.
        let decision = policy.decide(f64::NAN, 1.0, 0.0, &[0.0]);
        assert!(!decision.adopted());
        // A non-finite drift counts as maximal drift.
        let decision = policy.decide(1.0, 1.0, f64::NAN, &[0.0]);
        assert_eq!(decision.reason(), &AdoptReason::Drift { deviation: 1.0 });
        // A non-finite correctness counts as zero correctness, so it cannot pass
        // the correctness gate.
        let decision = policy.decide(1.0, f64::INFINITY, 0.0, &[0.0]);
        assert_eq!(
            decision.reason(),
            &AdoptReason::InsufficientCorrectness { score: 0.0 }
        );
        // Out-of-range values are clamped before comparison.
        let clamped = policy.decide(5.0, 2.0, -1.0, &[0.0]);
        assert_eq!(clamped.reason(), &AdoptReason::Improved { gain: 1.0 });
    }

    #[test]
    fn decision_round_trips_through_json() {
        for decision in [
            AdoptDecision::new(true, AdoptReason::FirstCandidate),
            AdoptDecision::new(true, AdoptReason::Improved { gain: 0.25 }),
            AdoptDecision::new(false, AdoptReason::NoImprovement { gain: -0.1 }),
            AdoptDecision::new(false, AdoptReason::Drift { deviation: 0.9 }),
            AdoptDecision::new(false, AdoptReason::InsufficientCorrectness { score: 0.2 }),
            AdoptDecision::new(false, AdoptReason::InsufficientHistory { rounds: 2 }),
        ] {
            let value = serde_json::to_value(decision).unwrap();
            assert!(value.get("adopted").is_some());
            assert!(value.get("reason").is_some());
            let restored: AdoptDecision = serde_json::from_value(value).unwrap();
            assert_eq!(restored, decision);
        }
        let defaulted: AdoptDecision = serde_json::from_str("{}").unwrap();
        assert_eq!(defaulted, AdoptDecision::default());
        assert!(!defaulted.adopted());
        assert_eq!(defaulted.reason(), &AdoptReason::FirstCandidate);
    }

    #[test]
    fn policy_exposes_its_configuration() {
        let config = AdoptConfig::new().with_min_gain(0.0);
        let policy = AdoptPolicy::new().with_config(config);
        assert_eq!(policy.config(), &config);
        assert_eq!(AdoptPolicy::default(), AdoptPolicy::new());
        assert_eq!(AdoptPolicy::new().config(), &AdoptConfig::default());
    }
}
