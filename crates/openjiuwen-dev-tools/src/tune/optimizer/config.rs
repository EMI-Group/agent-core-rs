//! [`OptimizerConfig`]: the search budget every optimizer shares.
//!
//! Ports the tunables behind Python's `TuneConstant` into one validated value:
//! the round budget, the beam width, the evaluation parallelism, the score that
//! stops the search early, and the few-shot example count.
//! [`OptimizerConfig::validate`] enforces the same bounds as Python without
//! panicking, naming the offending field and its allowed range.

use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};

use crate::tune::optimizer::{
    DEFAULT_EARLY_STOP_SCORE, DEFAULT_EXAMPLE_NUM, DEFAULT_ITERATION_NUM, DEFAULT_PARALLEL_NUM,
    MAX_EXAMPLE_NUM, MAX_ITERATION_NUM, MAX_PARALLEL_NUM, MIN_EXAMPLE_NUM, MIN_ITERATION_NUM,
    MIN_PARALLEL_NUM,
};

/// The search budget of an optimizer.
///
/// Every field defaults to the matching tuning constant, so
/// `OptimizerConfig::default()` is the Python default configuration.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct OptimizerConfig {
    /// How many optimisation rounds (beam expansions, epochs) to run at most.
    pub iterations: usize,
    /// How many best candidates a beam keeps per round.
    pub beam_width: usize,
    /// How many candidate prompts are evaluated concurrently.
    pub num_parallel: usize,
    /// The score at which the search stops early, in `0.0..=1.0`.
    pub early_stop_score: f64,
    /// How many few-shot examples to select.
    pub num_examples: usize,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            iterations: DEFAULT_ITERATION_NUM,
            beam_width: DEFAULT_PARALLEL_NUM,
            num_parallel: DEFAULT_PARALLEL_NUM,
            early_stop_score: DEFAULT_EARLY_STOP_SCORE,
            num_examples: DEFAULT_EXAMPLE_NUM,
        }
    }
}

impl OptimizerConfig {
    /// Creates the default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Replaces the round budget.
    #[must_use]
    pub fn with_iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }

    /// Replaces the beam width.
    #[must_use]
    pub fn with_beam_width(mut self, beam_width: usize) -> Self {
        self.beam_width = beam_width;
        self
    }

    /// Replaces the evaluation parallelism.
    #[must_use]
    pub fn with_num_parallel(mut self, num_parallel: usize) -> Self {
        self.num_parallel = num_parallel;
        self
    }

    /// Replaces the early-stop score.
    #[must_use]
    pub fn with_early_stop_score(mut self, early_stop_score: f64) -> Self {
        self.early_stop_score = early_stop_score;
        self
    }

    /// Replaces the few-shot example count.
    #[must_use]
    pub fn with_num_examples(mut self, num_examples: usize) -> Self {
        self.num_examples = num_examples;
        self
    }

    /// Checks every field against its tuning-constant bounds.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] naming the field, its value, and the allowed range:
    /// `iterations` in `MIN_ITERATION_NUM..=MAX_ITERATION_NUM`, `beam_width` and
    /// `num_parallel` in `MIN_PARALLEL_NUM..=MAX_PARALLEL_NUM`, `num_examples` in
    /// `MIN_EXAMPLE_NUM..=MAX_EXAMPLE_NUM`, and `early_stop_score` finite and in
    /// `0.0..=1.0`.
    pub fn validate(&self) -> Result<()> {
        check_usize(
            "iterations",
            self.iterations,
            MIN_ITERATION_NUM,
            MAX_ITERATION_NUM,
        )?;
        check_usize(
            "beam_width",
            self.beam_width,
            MIN_PARALLEL_NUM,
            MAX_PARALLEL_NUM,
        )?;
        check_usize(
            "num_parallel",
            self.num_parallel,
            MIN_PARALLEL_NUM,
            MAX_PARALLEL_NUM,
        )?;
        check_usize(
            "num_examples",
            self.num_examples,
            MIN_EXAMPLE_NUM,
            MAX_EXAMPLE_NUM,
        )?;
        if !self.early_stop_score.is_finite() || !(0.0..=1.0).contains(&self.early_stop_score) {
            return Err(Error::validation(format!(
                "OptimizerConfig.early_stop_score must be finite and between 0.0 and 1.0, got {}",
                self.early_stop_score
            )));
        }
        Ok(())
    }
}

/// Rejects `value` when it sits outside `min..=max`, naming the field.
fn check_usize(field: &str, value: usize, min: usize, max: usize) -> Result<()> {
    if value < min || value > max {
        return Err(Error::validation(format!(
            "OptimizerConfig.{field} should be between {min} and {max}, got {value}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_the_tuning_constants() {
        let config = OptimizerConfig::default();
        assert_eq!(config.iterations, DEFAULT_ITERATION_NUM);
        assert_eq!(config.beam_width, DEFAULT_PARALLEL_NUM);
        assert_eq!(config.num_parallel, DEFAULT_PARALLEL_NUM);
        assert!((config.early_stop_score - DEFAULT_EARLY_STOP_SCORE).abs() < f64::EPSILON);
        assert_eq!(config.num_examples, DEFAULT_EXAMPLE_NUM);
        assert_eq!(OptimizerConfig::new(), config);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn every_bound_is_accepted() {
        for iterations in [MIN_ITERATION_NUM, MAX_ITERATION_NUM] {
            let config = OptimizerConfig::default().with_iterations(iterations);
            assert!(config.validate().is_ok(), "iterations {iterations}");
        }
        for parallel in [MIN_PARALLEL_NUM, MAX_PARALLEL_NUM] {
            assert!(OptimizerConfig::default()
                .with_beam_width(parallel)
                .validate()
                .is_ok());
            assert!(OptimizerConfig::default()
                .with_num_parallel(parallel)
                .validate()
                .is_ok());
        }
        for examples in [MIN_EXAMPLE_NUM, MAX_EXAMPLE_NUM] {
            assert!(OptimizerConfig::default()
                .with_num_examples(examples)
                .validate()
                .is_ok());
        }
        for score in [0.0, 0.5, 1.0] {
            assert!(OptimizerConfig::default()
                .with_early_stop_score(score)
                .validate()
                .is_ok());
        }
    }

    #[test]
    fn out_of_range_fields_name_themselves() {
        let rejected = [
            (OptimizerConfig::default().with_iterations(0), "iterations"),
            (
                OptimizerConfig::default().with_iterations(MAX_ITERATION_NUM + 1),
                "iterations",
            ),
            (OptimizerConfig::default().with_beam_width(0), "beam_width"),
            (
                OptimizerConfig::default().with_beam_width(MAX_PARALLEL_NUM + 1),
                "beam_width",
            ),
            (
                OptimizerConfig::default().with_num_parallel(0),
                "num_parallel",
            ),
            (
                OptimizerConfig::default().with_num_examples(MAX_EXAMPLE_NUM + 1),
                "num_examples",
            ),
            (
                OptimizerConfig::default().with_early_stop_score(-0.1),
                "early_stop_score",
            ),
            (
                OptimizerConfig::default().with_early_stop_score(1.1),
                "early_stop_score",
            ),
            (
                OptimizerConfig::default().with_early_stop_score(f64::NAN),
                "early_stop_score",
            ),
        ];
        for (config, field) in rejected {
            let error = config.validate().unwrap_err();
            let message = error.to_string();
            assert!(
                matches!(error, Error::Validation(_)),
                "{field} produces a validation error, got {error}"
            );
            assert!(message.contains(field), "{field}: {message}");
        }
    }

    #[test]
    fn json_round_trip_uses_snake_case_and_defaults_missing_fields() {
        let config = OptimizerConfig::default()
            .with_iterations(5)
            .with_beam_width(2)
            .with_num_examples(4)
            .with_early_stop_score(0.9)
            .with_num_parallel(3);
        let value = serde_json::to_value(config).unwrap();
        let object = value.as_object().unwrap();
        for key in [
            "iterations",
            "beam_width",
            "num_parallel",
            "early_stop_score",
            "num_examples",
        ] {
            assert!(object.contains_key(key), "missing {key}");
        }
        let restored: OptimizerConfig = serde_json::from_value(value).unwrap();
        assert_eq!(restored, config);

        let partial: OptimizerConfig = serde_json::from_str(r#"{"iterations": 7}"#).unwrap();
        assert_eq!(partial.iterations, 7);
        assert_eq!(partial.beam_width, DEFAULT_PARALLEL_NUM);
        assert!(partial.validate().is_ok());
    }
}
