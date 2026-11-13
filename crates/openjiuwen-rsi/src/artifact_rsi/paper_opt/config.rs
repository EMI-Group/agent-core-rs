//! Weights and knobs of the paper-optimisation loop.
//!
//! Python parity: [`GeneralWeights`], [`ExperimentWeights`] and the composite
//! weights mirror `GeneralWeights`, `ExperimentWeights` and `CompositeWeights`
//! in `auto_research/modules/paper_scoring/schemas.py` (v0.1.20). Their
//! validators sum each weight set to `1.0` within `1e-6` and reject negative
//! weights.
//!
//! [`PaperOptConfig`] is the data-only port of the knobs the tree provider
//! reads from `configs/pipeline.default.yaml` and from its constructor
//! (`auto_research/tree_provider/orchestrator.py`: `max_iterations`,
//! `max_rounds`, `rubric_samples`, the dispersion thresholds). The `DeepAgent`
//! stage budgets, model selection and LaTeX toolchain settings are deliberately
//! absent: they belong to the deferred agent/toolchain half.
//!
//! Unknown YAML/JSON keys are tolerated and collected into
//! [`PaperOptConfig::extra`], so a newer Python pipeline configuration still
//! loads.
use super::scoring::{DEFAULT_MAD_THRESHOLD, DEFAULT_RANGE_THRESHOLD};
use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
/// Tolerance every weight set is required to sum within (Python `1e-6`).
pub const WEIGHT_TOLERANCE: f64 = 1e-6;
/// Weights of the general (NeurIPS-style) rubric dimensions.
///
/// Python parity: `schemas.GeneralWeights`; the defaults are the Python
/// defaults and the sum is exactly `1.0`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct GeneralWeights {
    /// Weight of the `soundness` dimension.
    pub soundness: f64,
    /// Weight of the `clarity` dimension.
    pub clarity: f64,
    /// Weight of the `contribution` dimension.
    pub contribution: f64,
    /// Weight of the general `overall` dimension.
    pub overall: f64,
}
impl Default for GeneralWeights {
    fn default() -> Self {
        Self {
            soundness: 0.30,
            clarity: 0.20,
            contribution: 0.25,
            overall: 0.25,
        }
    }
}
impl GeneralWeights {
    /// Returns the sum of the four weights.
    #[must_use]
    pub fn total(&self) -> f64 {
        self.soundness + self.clarity + self.contribution + self.overall
    }
    /// Rejects weights that do not form a probability distribution.
    ///
    /// # Errors
    ///
    /// Returns [`Error::config`] naming the `general` weight set when a weight
    /// is negative or non-finite, or when the four weights do not sum to `1.0`
    /// within [`WEIGHT_TOLERANCE`].
    pub fn validate(&self) -> Result<()> {
        for (name, value) in [
            ("soundness", self.soundness),
            ("clarity", self.clarity),
            ("contribution", self.contribution),
            ("overall", self.overall),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(Error::config(format!(
                    "general weights: {name} must be a finite non-negative number, got {value}"
                )));
            }
        }
        let total = self.total();
        if (total - 1.0).abs() > WEIGHT_TOLERANCE {
            return Err(Error::config(format!(
                "general weights must sum to 1.0, got {total}"
            )));
        }
        Ok(())
    }
}
/// Weights of the experiment-rigor rubric dimensions.
///
/// Python parity: `schemas.ExperimentWeights`; the defaults are the Python
/// defaults and the sum is exactly `1.0`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ExperimentWeights {
    /// Weight of the `question_alignment` dimension.
    pub question_alignment: f64,
    /// Weight of the `design_and_controls` dimension.
    pub design_and_controls: f64,
    /// Weight of the `measurement_and_statistics` dimension.
    pub measurement_and_statistics: f64,
    /// Weight of the `reporting_and_reproducibility` dimension.
    pub reporting_and_reproducibility: f64,
    /// Weight of the `claim_evidence_alignment` dimension.
    pub claim_evidence_alignment: f64,
    /// Weight of the `overall_experimental_rigor` dimension.
    pub overall_experimental_rigor: f64,
}
impl Default for ExperimentWeights {
    fn default() -> Self {
        Self {
            question_alignment: 0.20,
            design_and_controls: 0.20,
            measurement_and_statistics: 0.20,
            reporting_and_reproducibility: 0.15,
            claim_evidence_alignment: 0.15,
            overall_experimental_rigor: 0.10,
        }
    }
}
impl ExperimentWeights {
    /// Returns the sum of the six weights.
    #[must_use]
    pub fn total(&self) -> f64 {
        self.question_alignment
            + self.design_and_controls
            + self.measurement_and_statistics
            + self.reporting_and_reproducibility
            + self.claim_evidence_alignment
            + self.overall_experimental_rigor
    }
    /// Rejects weights that do not form a probability distribution.
    ///
    /// # Errors
    ///
    /// Returns [`Error::config`] naming the `experiment` weight set when a
    /// weight is negative or non-finite, or when the six weights do not sum to
    /// `1.0` within [`WEIGHT_TOLERANCE`].
    pub fn validate(&self) -> Result<()> {
        for (name, value) in [
            ("question_alignment", self.question_alignment),
            ("design_and_controls", self.design_and_controls),
            (
                "measurement_and_statistics",
                self.measurement_and_statistics,
            ),
            (
                "reporting_and_reproducibility",
                self.reporting_and_reproducibility,
            ),
            ("claim_evidence_alignment", self.claim_evidence_alignment),
            (
                "overall_experimental_rigor",
                self.overall_experimental_rigor,
            ),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(Error::config(format!(
                    "experiment weights: {name} must be a finite non-negative number, got {value}"
                )));
            }
        }
        let total = self.total();
        if (total - 1.0).abs() > WEIGHT_TOLERANCE {
            return Err(Error::config(format!(
                "experiment weights must sum to 1.0, got {total}"
            )));
        }
        Ok(())
    }
}
/// Every weight the composite score is built from.
///
/// Python parity: the two rubric weight models plus `CompositeWeights`
/// (`schemas.py`). Unlike Python these live in one value so a config document
/// cannot carry half a scoring policy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct PaperWeights {
    /// Weights of the general rubric dimensions.
    pub general: GeneralWeights,
    /// Weights of the experiment-rigor dimensions.
    pub experiment: ExperimentWeights,
    /// Weight of the general score inside the composite score (`0.5`).
    pub composite_general: f64,
    /// Weight of the experiment score inside the composite score (`0.5`).
    pub composite_experiment: f64,
}
impl Default for PaperWeights {
    fn default() -> Self {
        Self {
            general: GeneralWeights::default(),
            experiment: ExperimentWeights::default(),
            composite_general: 0.5,
            composite_experiment: 0.5,
        }
    }
}
impl PaperWeights {
    /// Validates every weight set.
    ///
    /// # Errors
    ///
    /// Returns [`Error::config`] naming the offending set — `general`,
    /// `experiment`, or `composite` — for a negative/non-finite weight or a
    /// sum outside `1.0 ± `[`WEIGHT_TOLERANCE`].
    pub fn validate(&self) -> Result<()> {
        self.general.validate()?;
        self.experiment.validate()?;
        for (name, value) in [
            ("composite_general", self.composite_general),
            ("composite_experiment", self.composite_experiment),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(Error::config(format!(
                    "composite weights: {name} must be a finite non-negative number, got {value}"
                )));
            }
        }
        let total = self.composite_general + self.composite_experiment;
        if (total - 1.0).abs() > WEIGHT_TOLERANCE {
            return Err(Error::config(format!(
                "composite weights must sum to 1.0, got {total}"
            )));
        }
        Ok(())
    }
}
/// Knobs of one paper-optimisation run.
///
/// Defaults are the Python pipeline defaults. The struct is `serde` and
/// tolerant of unknown keys, which are preserved in [`Self::extra`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct PaperOptConfig {
    /// Maximum optimisation iterations (rounds) one run may perform.
    pub max_iterations: u32,
    /// Maximum number of inner paper-writing rounds per iteration.
    ///
    /// Kept for Python interface parity; the portable loop drives one candidate
    /// set per iteration, so this is recorded but not consumed here.
    pub max_rounds: u32,
    /// Number of rubric samples the Python scorer draws per paper.
    ///
    /// Recorded for parity; the injected [`PaperScorer`](super::orchestrator::PaperScorer)
    /// owns sampling.
    pub rubric_samples: usize,
    /// Candidates proposed per iteration (`behaviour` of the round beam).
    pub beam_width: usize,
    /// Weights of every rubric dimension and of the composite score.
    pub weights: PaperWeights,
    /// Minimum per-dimension score range that flags dispersion (Python `3`).
    pub min_score_range: f64,
    /// Minimum mean absolute deviation that flags dispersion (Python `1.0`).
    pub mad_threshold: f64,
    /// Composite score at which the run stops early; `None` disables the target.
    pub min_composite: Option<f64>,
    /// Whether a candidate without a compiled paper artifact is pruned.
    pub require_artifact: bool,
    /// Unknown keys from a newer configuration document, preserved verbatim.
    #[serde(flatten)]
    pub extra: Value,
}
impl Default for PaperOptConfig {
    fn default() -> Self {
        Self {
            max_iterations: 5,
            max_rounds: 5,
            rubric_samples: 5,
            beam_width: 2,
            weights: PaperWeights::default(),
            min_score_range: DEFAULT_RANGE_THRESHOLD,
            mad_threshold: DEFAULT_MAD_THRESHOLD,
            min_composite: None,
            require_artifact: true,
            extra: Value::Object(serde_json::Map::new()),
        }
    }
}
impl PaperOptConfig {
    /// Parses a configuration from a JSON value.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`] when the value does not match the expected
    /// shape.
    pub fn from_json_value(value: Value) -> Result<Self> {
        serde_json::from_value(value)
            .map_err(|error| Error::validation(format!("invalid paper-opt config: {error}")))
    }
    /// Parses a configuration from a YAML document.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`] when the YAML cannot be parsed or does not
    /// match the expected shape.
    pub fn from_yaml_str(yaml: &str) -> Result<Self> {
        serde_yaml_ng::from_str(yaml)
            .map_err(|error| Error::validation(format!("invalid paper-opt config: {error}")))
    }
    /// Rejects configurations the loop cannot run with.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`] when `max_iterations` or `beam_width` is
    /// zero, or when a dispersion threshold is negative/non-finite; propagates
    /// the [`Error::config`] of [`PaperWeights::validate`] for a bad weight set.
    pub fn validate(&self) -> Result<()> {
        if self.max_iterations == 0 {
            return Err(Error::validation(
                "paper_opt: max_iterations must be at least 1",
            ));
        }
        if self.beam_width == 0 {
            return Err(Error::validation(
                "paper_opt: beam_width must be at least 1",
            ));
        }
        self.weights.validate()?;
        for (name, value) in [
            ("min_score_range", self.min_score_range),
            ("mad_threshold", self.mad_threshold),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(Error::validation(format!(
                    "paper_opt: {name} must be a finite non-negative number, got {value}"
                )));
            }
        }
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use openjiuwen_common::error::ErrorKind;
    use serde_json::json;
    #[test]
    fn defaults_match_python() {
        let config = PaperOptConfig::default();
        assert_eq!(config.max_iterations, 5);
        assert_eq!(config.max_rounds, 5);
        assert_eq!(config.rubric_samples, 5);
        assert_eq!(config.beam_width, 2);
        assert!(config.min_composite.is_none());
        assert!(config.require_artifact);
        assert!((config.min_score_range - 3.0).abs() < f64::EPSILON);
        assert!((config.mad_threshold - 1.0).abs() < f64::EPSILON);
        assert_eq!(config.weights, PaperWeights::default());
        assert!(config.validate().is_ok());
    }
    #[test]
    fn weight_validation_names_the_offending_set() {
        let cases: Vec<(PaperWeights, &str)> = vec![
            (
                PaperWeights {
                    general: GeneralWeights {
                        soundness: 0.5,
                        ..GeneralWeights::default()
                    },
                    ..PaperWeights::default()
                },
                "general",
            ),
            (
                PaperWeights {
                    experiment: ExperimentWeights {
                        question_alignment: 0.5,
                        ..ExperimentWeights::default()
                    },
                    ..PaperWeights::default()
                },
                "experiment",
            ),
            (
                PaperWeights {
                    composite_general: 0.4,
                    composite_experiment: 0.4,
                    ..PaperWeights::default()
                },
                "composite",
            ),
        ];
        for (weights, set) in cases {
            let error = weights.validate().expect_err("weights should be rejected");
            assert_eq!(error.kind(), ErrorKind::Config, "{set}");
            assert!(error.to_string().contains(set), "{error}");
        }
    }
    #[test]
    fn negative_weights_are_rejected() {
        let weights = GeneralWeights {
            soundness: -0.1,
            clarity: 0.5,
            ..GeneralWeights::default()
        };
        let error = weights.validate().expect_err("negative weight");
        assert_eq!(error.kind(), ErrorKind::Config);
    }
    #[test]
    fn validation_rejects_bad_knobs() {
        let cases: Vec<(PaperOptConfig, &str)> = vec![
            (
                PaperOptConfig {
                    max_iterations: 0,
                    ..PaperOptConfig::default()
                },
                "max_iterations",
            ),
            (
                PaperOptConfig {
                    beam_width: 0,
                    ..PaperOptConfig::default()
                },
                "beam_width",
            ),
            (
                PaperOptConfig {
                    min_score_range: -1.0,
                    ..PaperOptConfig::default()
                },
                "min_score_range",
            ),
            (
                PaperOptConfig {
                    mad_threshold: f64::NAN,
                    ..PaperOptConfig::default()
                },
                "mad_threshold",
            ),
        ];
        for (config, field) in cases {
            let error = config.validate().expect_err("config should be rejected");
            assert_eq!(error.kind(), ErrorKind::Validation, "{field}");
            assert!(error.to_string().contains(field), "{error}");
        }
    }
    #[test]
    fn parsing_tolerates_unknown_and_missing_keys() {
        let config = PaperOptConfig::from_yaml_str(
            "max_iterations: 3\nbeam_width: 4\nmin_composite: 7.5\nfuture_knob: yes\n",
        )
        .expect("yaml parses");
        assert_eq!(config.max_iterations, 3);
        assert_eq!(config.beam_width, 4);
        assert_eq!(config.min_composite, Some(7.5));
        assert_eq!(config.rubric_samples, 5);
        assert_eq!(config.extra["future_knob"], json!("yes"));
    }
    #[test]
    fn json_parsing_rejects_a_wrong_shape() {
        let error = PaperOptConfig::from_json_value(json!({"max_iterations": "many"}))
            .expect_err("wrong type");
        assert_eq!(error.kind(), ErrorKind::Validation);
    }
}
