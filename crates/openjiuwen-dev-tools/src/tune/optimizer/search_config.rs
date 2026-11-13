//! [`PromptSearchConfig`]: the tunables of the prompt-search optimizer.
//!
//! Ports the keyword defaults of Python's
//! `optimizer/prompt_search/optimizer.py::PromptSearchOptimizer.__init__` into
//! one validated value: the round and candidate budgets, the convergence
//! detector's threshold/window/target, the drift penalty, the correctness gate,
//! and the policy temperature.
//!
//! [`PromptSearchConfig::validate`] enforces the Python bounds without
//! panicking, naming the offending field, its value, and the allowed range.
//! `policy_temperature` is the one tunable with no offline effect: it is
//! forwarded to model rewrite requests when a model is attached, and is
//! otherwise recorded on the config (never silently dropped or ignored).

use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};

use crate::tune::optimizer::{
    MAX_ITERATION_NUM, MAX_PARALLEL_NUM, MIN_ITERATION_NUM, MIN_PARALLEL_NUM,
};

/// Python `PromptSearchOptimizer`'s default `max_iterations`.
pub const DEFAULT_SEARCH_MAX_ITERATIONS: usize = 6;
/// Python `PromptSearchOptimizer`'s default `candidate_prompts`.
pub const DEFAULT_SEARCH_CANDIDATE_PROMPTS: usize = 5;
/// Python `PromptSearchOptimizer`'s default `convergence_threshold`.
pub const DEFAULT_CONVERGENCE_THRESHOLD: f64 = 0.01;
/// Python `PromptSearchOptimizer`'s default `convergence_window`.
pub const DEFAULT_CONVERGENCE_WINDOW: usize = 3;
/// Python `PromptSearchOptimizer`'s default `drift_penalty`.
pub const DEFAULT_DRIFT_PENALTY: f64 = 0.5;
/// Python `PromptSearchOptimizer`'s default `min_correctness`.
pub const DEFAULT_MIN_CORRECTNESS: f64 = 0.5;
/// Python `PromptSearchOptimizer`'s default `policy_temperature`.
pub const DEFAULT_POLICY_TEMPERATURE: f64 = 0.9;

/// The search budget and stopping rules of the prompt-search optimizer.
///
/// Every field defaults to the matching Python default, so
/// [`PromptSearchConfig::default`] is the Python default configuration.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct PromptSearchConfig {
    /// How many optimisation rounds to run at most.
    pub max_iterations: usize,
    /// How many candidate prompts the policy proposes per round.
    pub candidate_prompts: usize,
    /// The improvement (or variance) floor below which the search stops.
    pub convergence_threshold: f64,
    /// How many recent rewards form the plateau window.
    pub convergence_window: usize,
    /// The reward subtracted for drifting away from the base prompt.
    pub drift_penalty: f64,
    /// The correctness gate below which a candidate's reward is capped.
    pub min_correctness: f64,
    /// The sampling temperature of the model-backed prompt policy.
    pub policy_temperature: f64,
    /// An optional reward at which the search stops immediately.
    pub target_score: Option<f64>,
}

impl Default for PromptSearchConfig {
    fn default() -> Self {
        Self {
            max_iterations: DEFAULT_SEARCH_MAX_ITERATIONS,
            candidate_prompts: DEFAULT_SEARCH_CANDIDATE_PROMPTS,
            convergence_threshold: DEFAULT_CONVERGENCE_THRESHOLD,
            convergence_window: DEFAULT_CONVERGENCE_WINDOW,
            drift_penalty: DEFAULT_DRIFT_PENALTY,
            min_correctness: DEFAULT_MIN_CORRECTNESS,
            policy_temperature: DEFAULT_POLICY_TEMPERATURE,
            target_score: None,
        }
    }
}

impl PromptSearchConfig {
    /// Creates the default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Replaces the round budget.
    #[must_use]
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Replaces the per-round candidate count.
    #[must_use]
    pub fn with_candidate_prompts(mut self, candidate_prompts: usize) -> Self {
        self.candidate_prompts = candidate_prompts;
        self
    }

    /// Replaces the convergence threshold.
    #[must_use]
    pub fn with_convergence_threshold(mut self, convergence_threshold: f64) -> Self {
        self.convergence_threshold = convergence_threshold;
        self
    }

    /// Replaces the convergence window.
    #[must_use]
    pub fn with_convergence_window(mut self, convergence_window: usize) -> Self {
        self.convergence_window = convergence_window;
        self
    }

    /// Replaces the drift penalty.
    #[must_use]
    pub fn with_drift_penalty(mut self, drift_penalty: f64) -> Self {
        self.drift_penalty = drift_penalty;
        self
    }

    /// Replaces the correctness gate.
    #[must_use]
    pub fn with_min_correctness(mut self, min_correctness: f64) -> Self {
        self.min_correctness = min_correctness;
        self
    }

    /// Replaces the policy temperature.
    #[must_use]
    pub fn with_policy_temperature(mut self, policy_temperature: f64) -> Self {
        self.policy_temperature = policy_temperature;
        self
    }

    /// Sets (or clears) the target reward that stops the search immediately.
    #[must_use]
    pub fn with_target_score(mut self, target_score: Option<f64>) -> Self {
        self.target_score = target_score;
        self
    }

    /// Checks every field against its Python bounds.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] naming the field, its value, and the allowed range:
    /// `max_iterations` in `MIN_ITERATION_NUM..=MAX_ITERATION_NUM`,
    /// `candidate_prompts` in `MIN_PARALLEL_NUM..=MAX_PARALLEL_NUM`,
    /// `convergence_threshold` finite and positive, `convergence_window` at
    /// least 1, `drift_penalty` finite and non-negative, `min_correctness` and
    /// `policy_temperature` (the latter in `0.0..=2.0`), and `target_score`
    /// finite and in `0.0..=1.0` when set.
    pub fn validate(&self) -> Result<()> {
        check_usize(
            "max_iterations",
            self.max_iterations,
            MIN_ITERATION_NUM,
            MAX_ITERATION_NUM,
        )?;
        check_usize(
            "candidate_prompts",
            self.candidate_prompts,
            MIN_PARALLEL_NUM,
            MAX_PARALLEL_NUM,
        )?;
        if !self.convergence_threshold.is_finite() || self.convergence_threshold <= 0.0 {
            return Err(Error::validation(format!(
                "PromptSearchConfig.convergence_threshold must be finite and greater than 0.0, got {}",
                self.convergence_threshold
            )));
        }
        if self.convergence_window == 0 {
            return Err(Error::validation(format!(
                "PromptSearchConfig.convergence_window must be at least 1, got {}",
                self.convergence_window
            )));
        }
        if !self.drift_penalty.is_finite() || self.drift_penalty < 0.0 {
            return Err(Error::validation(format!(
                "PromptSearchConfig.drift_penalty must be finite and non-negative, got {}",
                self.drift_penalty
            )));
        }
        if !self.min_correctness.is_finite() || !(0.0..=1.0).contains(&self.min_correctness) {
            return Err(Error::validation(format!(
                "PromptSearchConfig.min_correctness must be finite and between 0.0 and 1.0, got {}",
                self.min_correctness
            )));
        }
        if !self.policy_temperature.is_finite() || !(0.0..=2.0).contains(&self.policy_temperature) {
            return Err(Error::validation(format!(
                "PromptSearchConfig.policy_temperature must be finite and between 0.0 and 2.0, got {}",
                self.policy_temperature
            )));
        }
        if let Some(target) = self.target_score {
            if !target.is_finite() || !(0.0..=1.0).contains(&target) {
                return Err(Error::validation(format!(
                    "PromptSearchConfig.target_score must be finite and between 0.0 and 1.0 when set, got {target}"
                )));
            }
        }
        Ok(())
    }
}

/// Rejects `value` when it sits outside `min..=max`, naming the field.
fn check_usize(field: &str, value: usize, min: usize, max: usize) -> Result<()> {
    if value < min || value > max {
        return Err(Error::validation(format!(
            "PromptSearchConfig.{field} should be between {min} and {max}, got {value}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use openjiuwen_common::error::{Error, ErrorKind};

    use super::*;

    #[test]
    fn defaults_are_the_python_prompt_search_defaults() {
        let config = PromptSearchConfig::default();
        assert_eq!(config.max_iterations, 6);
        assert_eq!(config.candidate_prompts, 5);
        assert!((config.convergence_threshold - 0.01).abs() < f64::EPSILON);
        assert_eq!(config.convergence_window, 3);
        assert!((config.drift_penalty - 0.5).abs() < f64::EPSILON);
        assert!((config.min_correctness - 0.5).abs() < f64::EPSILON);
        assert!((config.policy_temperature - 0.9).abs() < f64::EPSILON);
        assert_eq!(config.target_score, None);
        assert_eq!(PromptSearchConfig::new(), config);
        assert!(config.validate().is_ok());

        assert_eq!(DEFAULT_SEARCH_MAX_ITERATIONS, 6);
        assert_eq!(DEFAULT_SEARCH_CANDIDATE_PROMPTS, 5);
        assert!((DEFAULT_CONVERGENCE_THRESHOLD - 0.01).abs() < f64::EPSILON);
        assert_eq!(DEFAULT_CONVERGENCE_WINDOW, 3);
        assert!((DEFAULT_DRIFT_PENALTY - 0.5).abs() < f64::EPSILON);
        assert!((DEFAULT_MIN_CORRECTNESS - 0.5).abs() < f64::EPSILON);
        assert!((DEFAULT_POLICY_TEMPERATURE - 0.9).abs() < f64::EPSILON);
    }

    #[test]
    fn builders_replace_every_field() {
        let config = PromptSearchConfig::new()
            .with_max_iterations(10)
            .with_candidate_prompts(7)
            .with_convergence_threshold(0.05)
            .with_convergence_window(4)
            .with_drift_penalty(0.25)
            .with_min_correctness(0.8)
            .with_policy_temperature(1.5)
            .with_target_score(Some(0.95));
        assert_eq!(config.max_iterations, 10);
        assert_eq!(config.candidate_prompts, 7);
        assert!((config.convergence_threshold - 0.05).abs() < 1e-12);
        assert_eq!(config.convergence_window, 4);
        assert!((config.drift_penalty - 0.25).abs() < 1e-12);
        assert!((config.min_correctness - 0.8).abs() < 1e-12);
        assert!((config.policy_temperature - 1.5).abs() < 1e-12);
        assert_eq!(config.target_score, Some(0.95));
        assert!(config.validate().is_ok());

        let cleared = config.with_target_score(None);
        assert_eq!(cleared.target_score, None);
    }

    #[test]
    fn every_bound_is_accepted() {
        assert!(PromptSearchConfig::default()
            .with_max_iterations(MIN_ITERATION_NUM)
            .validate()
            .is_ok());
        assert!(PromptSearchConfig::default()
            .with_max_iterations(MAX_ITERATION_NUM)
            .validate()
            .is_ok());
        assert!(PromptSearchConfig::default()
            .with_candidate_prompts(MIN_PARALLEL_NUM)
            .validate()
            .is_ok());
        assert!(PromptSearchConfig::default()
            .with_candidate_prompts(MAX_PARALLEL_NUM)
            .validate()
            .is_ok());
        for threshold in [f64::MIN_POSITIVE, 1.0] {
            assert!(PromptSearchConfig::default()
                .with_convergence_threshold(threshold)
                .validate()
                .is_ok());
        }
        assert!(PromptSearchConfig::default()
            .with_convergence_window(1)
            .validate()
            .is_ok());
        for penalty in [0.0, 1.0] {
            assert!(PromptSearchConfig::default()
                .with_drift_penalty(penalty)
                .validate()
                .is_ok());
        }
        for correctness in [0.0, 1.0] {
            assert!(PromptSearchConfig::default()
                .with_min_correctness(correctness)
                .validate()
                .is_ok());
        }
        for temperature in [0.0, 2.0] {
            assert!(PromptSearchConfig::default()
                .with_policy_temperature(temperature)
                .validate()
                .is_ok());
        }
        for target in [0.0, 1.0] {
            assert!(PromptSearchConfig::default()
                .with_target_score(Some(target))
                .validate()
                .is_ok());
        }
    }

    #[test]
    fn out_of_range_fields_name_themselves() {
        let rejected = [
            (
                PromptSearchConfig::default().with_max_iterations(0),
                "max_iterations",
            ),
            (
                PromptSearchConfig::default().with_max_iterations(MAX_ITERATION_NUM + 1),
                "max_iterations",
            ),
            (
                PromptSearchConfig::default().with_candidate_prompts(0),
                "candidate_prompts",
            ),
            (
                PromptSearchConfig::default().with_candidate_prompts(MAX_PARALLEL_NUM + 1),
                "candidate_prompts",
            ),
            (
                PromptSearchConfig::default().with_convergence_threshold(0.0),
                "convergence_threshold",
            ),
            (
                PromptSearchConfig::default().with_convergence_threshold(-0.1),
                "convergence_threshold",
            ),
            (
                PromptSearchConfig::default().with_convergence_threshold(f64::NAN),
                "convergence_threshold",
            ),
            (
                PromptSearchConfig::default().with_convergence_window(0),
                "convergence_window",
            ),
            (
                PromptSearchConfig::default().with_drift_penalty(-0.1),
                "drift_penalty",
            ),
            (
                PromptSearchConfig::default().with_drift_penalty(f64::NAN),
                "drift_penalty",
            ),
            (
                PromptSearchConfig::default().with_min_correctness(-0.1),
                "min_correctness",
            ),
            (
                PromptSearchConfig::default().with_min_correctness(1.1),
                "min_correctness",
            ),
            (
                PromptSearchConfig::default().with_policy_temperature(-0.1),
                "policy_temperature",
            ),
            (
                PromptSearchConfig::default().with_policy_temperature(2.1),
                "policy_temperature",
            ),
            (
                PromptSearchConfig::default().with_target_score(Some(1.1)),
                "target_score",
            ),
            (
                PromptSearchConfig::default().with_target_score(Some(f64::NAN)),
                "target_score",
            ),
        ];
        for (config, field) in rejected {
            let error = config.validate().unwrap_err();
            assert_eq!(error.kind(), ErrorKind::Validation, "{field}: {error}");
            assert!(matches!(error, Error::Validation(_)), "{field}: {error}");
            assert!(error.to_string().contains(field), "{field}: {error}");
        }
    }

    #[test]
    fn json_round_trip_uses_snake_case_and_defaults_missing_fields() {
        let config = PromptSearchConfig::default()
            .with_max_iterations(8)
            .with_candidate_prompts(3)
            .with_convergence_threshold(0.02)
            .with_convergence_window(2)
            .with_drift_penalty(0.3)
            .with_min_correctness(0.6)
            .with_policy_temperature(1.1)
            .with_target_score(Some(0.9));
        let value = serde_json::to_value(config).unwrap();
        let object = value.as_object().unwrap();
        for key in [
            "max_iterations",
            "candidate_prompts",
            "convergence_threshold",
            "convergence_window",
            "drift_penalty",
            "min_correctness",
            "policy_temperature",
            "target_score",
        ] {
            assert!(object.contains_key(key), "missing {key}");
        }
        let restored: PromptSearchConfig = serde_json::from_value(value).unwrap();
        assert_eq!(restored, config);

        let partial: PromptSearchConfig =
            serde_json::from_str(r#"{"candidate_prompts": 4}"#).unwrap();
        assert_eq!(partial.candidate_prompts, 4);
        assert_eq!(partial.max_iterations, DEFAULT_SEARCH_MAX_ITERATIONS);
        assert_eq!(partial.target_score, None);
        assert!(partial.validate().is_ok());
    }
}
