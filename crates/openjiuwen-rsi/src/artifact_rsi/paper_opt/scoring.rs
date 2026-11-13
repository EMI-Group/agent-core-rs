//! Rubric samples, deterministic aggregation, and the compact scoresheet.
//!
//! Python parity: the two rubric review outputs
//! (`GeneralRubricReviewOutput`, `ExperimentRigorReviewOutput`), their
//! aggregation (`aggregate_dimension`, `general_score`, `experiment_score`,
//! `composite_score`, `build_scoresheet`, `compact_scoresheet`) in
//! `auto_research/modules/paper_scoring/aggregation.py` and the weight models in
//! `schemas.py` (v0.1.20).
//!
//! A [`RubricSample`] is one reviewer's verdict, reduced to the numbers the
//! portable core needs: Python carries a justification, strengths, weaknesses
//! and cited sections per dimension, but only the score participates in the
//! composite. [`aggregate`] averages each dimension over the samples, applies
//! the configured dimension weights, and composes the two family scores.
//!
//! Two deliberate deviations from Python, both required by the Rust contract:
//!
//! - [`flags_dispersion`] measures `mad` as the **mean** absolute deviation.
//!   Python's `aggregation.py::_mad` uses the **median** absolute deviation;
//!   the threshold semantics (flag when `>= threshold`) are identical, and the
//!   mean is what this crate's configuration names.
//! - [`compact_scoresheet`] receives a [`ScoreBreakdown`] (family and composite
//!   scores) rather than Python's per-dimension `Scoresheet`, so the compact map
//!   carries `general`, `experiment`, `composite` and `sample_count`.
use super::config::{ExperimentWeights, GeneralWeights, PaperWeights};
use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
/// Python `range_flag` default (`aggregation.py`): a score range at or above it
/// flags dispersion.
pub const DEFAULT_RANGE_THRESHOLD: f64 = 3.0;
/// Python `mad_flag` default (`aggregation.py`): a mean absolute deviation at or
/// above it flags dispersion.
pub const DEFAULT_MAD_THRESHOLD: f64 = 1.0;
/// One reviewer's general-rubric scores, each on the Python `1..=10` scale.
///
/// Python parity: `GeneralRubricReviewOutput`, with the per-dimension review
/// text dropped and the dimension score kept as `f64`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct GeneralScores {
    /// Soundness of the claims and method.
    pub soundness: f64,
    /// Clarity of the writing and presentation.
    pub clarity: f64,
    /// Significance and novelty of the contribution.
    pub contribution: f64,
    /// The reviewer's holistic general score.
    pub overall: f64,
}
/// One reviewer's experiment-rigor scores, each on the Python `1..=10` scale.
///
/// Python parity: `ExperimentRigorReviewOutput`, same reduction as
/// [`GeneralScores`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ExperimentScores {
    /// How well the experiments answer the stated question.
    pub question_alignment: f64,
    /// Quality of the experimental design and its controls.
    pub design_and_controls: f64,
    /// Quality of the measurements and statistical treatment.
    pub measurement_and_statistics: f64,
    /// Reporting completeness and reproducibility.
    pub reporting_and_reproducibility: f64,
    /// Whether the claims follow from the evidence.
    pub claim_evidence_alignment: f64,
    /// The reviewer's holistic rigor score.
    pub overall_experimental_rigor: f64,
}
/// One reviewer's full rubric: the general and experiment families together.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct RubricSample {
    /// General-rubric scores.
    pub general: GeneralScores,
    /// Experiment-rigor scores.
    pub experiment: ExperimentScores,
}
impl RubricSample {
    /// Builds a sample whose every dimension is `score`.
    ///
    /// Useful for the offline template proposer and for tests where one number
    /// stands for the whole rubric.
    #[must_use]
    pub fn uniform(score: f64) -> Self {
        Self {
            general: GeneralScores {
                soundness: score,
                clarity: score,
                contribution: score,
                overall: score,
            },
            experiment: ExperimentScores {
                question_alignment: score,
                design_and_controls: score,
                measurement_and_statistics: score,
                reporting_and_reproducibility: score,
                claim_evidence_alignment: score,
                overall_experimental_rigor: score,
            },
        }
    }
}
/// The scores of one paper after aggregation.
///
/// Python parity: the `general_score`, `experiment_score` and `composite_score`
/// values of a `Scoresheet`. `sample_count` records how many rubric samples were
/// averaged, so an empty or single-sample aggregate is visible downstream.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ScoreBreakdown {
    /// Weighted general-rubric score.
    pub general: f64,
    /// Weighted experiment-rigor score.
    pub experiment: f64,
    /// Weighted composite score (the ranking value).
    pub composite: f64,
    /// Number of rubric samples averaged.
    pub sample_count: usize,
}
impl ScoreBreakdown {
    /// Reports whether this score strictly improves on `other`.
    ///
    /// Python's adoption gate compares composites with `>`, so a tie is not an
    /// improvement.
    #[must_use]
    pub fn is_improvement_over(&self, other: &Self) -> bool {
        self.composite > other.composite
    }
}
/// Spread of the general `overall` dimension across a sample set.
///
/// `score_range` is `max - min`; `mad` is the mean absolute deviation from the
/// mean. Both are zero for fewer than one sample.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct RubricDispersion {
    /// `max - min` of the `overall` dimension.
    pub score_range: f64,
    /// Mean absolute deviation of the `overall` dimension.
    pub mad: f64,
}
/// Computes the weighted general-rubric score.
///
/// Python parity: `aggregation.py::general_score`.
#[must_use]
pub fn general_score(scores: &GeneralScores, weights: &GeneralWeights) -> f64 {
    weights.soundness * scores.soundness
        + weights.clarity * scores.clarity
        + weights.contribution * scores.contribution
        + weights.overall * scores.overall
}
/// Computes the weighted experiment-rigor score.
///
/// Python parity: `aggregation.py::experiment_score`.
#[must_use]
pub fn experiment_score(scores: &ExperimentScores, weights: &ExperimentWeights) -> f64 {
    weights.question_alignment * scores.question_alignment
        + weights.design_and_controls * scores.design_and_controls
        + weights.measurement_and_statistics * scores.measurement_and_statistics
        + weights.reporting_and_reproducibility * scores.reporting_and_reproducibility
        + weights.claim_evidence_alignment * scores.claim_evidence_alignment
        + weights.overall_experimental_rigor * scores.overall_experimental_rigor
}
/// Composes the two family scores into the composite score.
///
/// Python parity: `aggregation.py::composite_score`.
#[must_use]
pub fn composite_score(general: f64, experiment: f64, weights: &PaperWeights) -> f64 {
    weights.composite_general * general + weights.composite_experiment * experiment
}
/// Averages a non-empty sample set with the default [`PaperWeights`].
///
/// # Errors
///
/// Returns [`Error::validation`] when `samples` is empty.
pub fn aggregate(samples: &[RubricSample]) -> Result<ScoreBreakdown> {
    aggregate_with_weights(samples, &PaperWeights::default())
}
/// Averages each dimension over `samples` and applies `weights`.
///
/// Sample values are averaged per dimension first (Python
/// `aggregate_dimension`), then the dimension weights are applied to the means.
/// Order and duplicates do not affect the result.
///
/// # Errors
///
/// Returns [`Error::validation`] when `samples` is empty, and propagates the
/// [`Error::config`] of [`PaperWeights::validate`] for an invalid weight set.
#[allow(clippy::cast_precision_loss)] // sample counts are exact integers in f64 practice
pub fn aggregate_with_weights(
    samples: &[RubricSample],
    weights: &PaperWeights,
) -> Result<ScoreBreakdown> {
    if samples.is_empty() {
        return Err(Error::validation(
            "paper_opt: rubric samples must not be empty",
        ));
    }
    weights.validate()?;
    let count = samples.len() as f64;
    let inverse = 1.0 / count;
    let mut general = GeneralScores::default();
    let mut experiment = ExperimentScores::default();
    for sample in samples {
        general.soundness += sample.general.soundness;
        general.clarity += sample.general.clarity;
        general.contribution += sample.general.contribution;
        general.overall += sample.general.overall;
        experiment.question_alignment += sample.experiment.question_alignment;
        experiment.design_and_controls += sample.experiment.design_and_controls;
        experiment.measurement_and_statistics += sample.experiment.measurement_and_statistics;
        experiment.reporting_and_reproducibility += sample.experiment.reporting_and_reproducibility;
        experiment.claim_evidence_alignment += sample.experiment.claim_evidence_alignment;
        experiment.overall_experimental_rigor += sample.experiment.overall_experimental_rigor;
    }
    for value in [
        &mut general.soundness,
        &mut general.clarity,
        &mut general.contribution,
        &mut general.overall,
        &mut experiment.question_alignment,
        &mut experiment.design_and_controls,
        &mut experiment.measurement_and_statistics,
        &mut experiment.reporting_and_reproducibility,
        &mut experiment.claim_evidence_alignment,
        &mut experiment.overall_experimental_rigor,
    ] {
        *value *= inverse;
    }
    let general = general_score(&general, &weights.general);
    let experiment = experiment_score(&experiment, &weights.experiment);
    Ok(ScoreBreakdown {
        general,
        experiment,
        composite: composite_score(general, experiment, weights),
        sample_count: samples.len(),
    })
}
/// Measures the spread of the general `overall` dimension across `samples`.
///
/// An empty input yields zeroes; values are taken as given (no clamping to the
/// `1..=10` rubric scale).
#[must_use]
#[allow(clippy::cast_precision_loss)] // sample counts are exact integers in f64 practice
pub fn dispersion(samples: &[RubricSample]) -> RubricDispersion {
    if samples.is_empty() {
        return RubricDispersion::default();
    }
    let values: Vec<f64> = samples
        .iter()
        .map(|sample| sample.general.overall)
        .collect();
    let minimum = values.iter().copied().fold(f64::INFINITY, f64::min);
    let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let mad = values.iter().map(|value| (value - mean).abs()).sum::<f64>() / values.len() as f64;
    RubricDispersion {
        score_range: maximum - minimum,
        mad,
    }
}
/// Reports whether a sample set is too dispersed to trust.
///
/// Python parity: `DimensionAggregate.flagged_dispersion`
/// (`score_range >= range_flag or mad >= mad_flag`), applied to the general
/// `overall` dimension here rather than per dimension. An empty input is never
/// flagged. Callers pass [`DEFAULT_RANGE_THRESHOLD`] and
/// [`DEFAULT_MAD_THRESHOLD`] (or a [`PaperOptConfig`](super::config::PaperOptConfig)'s
/// `min_score_range` / `mad_threshold`).
#[must_use]
pub fn flags_dispersion(
    samples: &[RubricSample],
    range_threshold: f64,
    mad_threshold: f64,
) -> bool {
    if samples.is_empty() {
        return false;
    }
    let dispersion = dispersion(samples);
    dispersion.score_range >= range_threshold || dispersion.mad >= mad_threshold
}
/// Rounds `value` to `decimals` decimal places, ties to the even neighbour.
///
/// This is the rule Python's `round` intends for `compact_scoresheet`
/// (`aggregation.py`), but the two differ exactly at a binary-representable
/// midpoint: this implementation scales the value by `10^decimals` and rounds
/// that product half-to-even, so `0.12345` — whose product is exactly
/// `1234.5` — becomes `0.1234` (`1234` is even), while Python consults the exact
/// decimal value and returns `0.1235`. Both rules agree everywhere else:
/// `2.5 -> 2.0`, `3.5 -> 4.0`, `0.125 -> 0.12`, `0.375 -> 0.38`.
///
/// Non-finite values and absurd precisions are returned unchanged.
#[must_use]
#[allow(clippy::float_cmp)] // the tie is a deliberate exact-midpoint test
pub fn round_half_even(value: f64, decimals: u32) -> f64 {
    if !value.is_finite() {
        return value;
    }
    let Ok(exponent) = i32::try_from(decimals) else {
        return value;
    };
    if exponent > 300 {
        return value;
    }
    let factor = 10_f64.powi(exponent);
    let scaled = value * factor;
    if !scaled.is_finite() {
        return value;
    }
    let floor = scaled.floor();
    let fraction = scaled - floor;
    let rounded = if fraction == 0.5 {
        if floor % 2.0 == 0.0 {
            floor
        } else {
            floor + 1.0
        }
    } else {
        scaled.round()
    };
    rounded / factor
}
/// Rounds every value of `breakdown` to four decimals.
///
/// Python parity: `aggregation.py::compact_scoresheet` (values rounded with
/// `round(..., 4)`); the keys follow this crate's [`ScoreBreakdown`] because the
/// per-dimension `Scoresheet` is not part of the portable core.
#[must_use]
#[allow(clippy::cast_precision_loss)] // sample counts are exact integers in f64 practice
pub fn compact_scoresheet(breakdown: &ScoreBreakdown) -> BTreeMap<String, f64> {
    let mut compact = BTreeMap::new();
    compact.insert("general".to_owned(), round_half_even(breakdown.general, 4));
    compact.insert(
        "experiment".to_owned(),
        round_half_even(breakdown.experiment, 4),
    );
    compact.insert(
        "composite".to_owned(),
        round_half_even(breakdown.composite, 4),
    );
    compact.insert("sample_count".to_owned(), breakdown.sample_count as f64);
    compact
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn round_half_even_matches_the_documented_rule() {
        assert!((round_half_even(2.5, 0) - 2.0).abs() < f64::EPSILON);
        assert!((round_half_even(3.5, 0) - 4.0).abs() < f64::EPSILON);
        assert!((round_half_even(0.125, 2) - 0.12).abs() < 1e-12);
        assert!((round_half_even(0.375, 2) - 0.38).abs() < 1e-12);
        assert!((round_half_even(-2.5, 0) + 2.0).abs() < f64::EPSILON);
        // The scaled product is exactly 1234.5, whose even neighbour is 1234.
        assert!((round_half_even(0.12345, 4) - 0.1234).abs() < 1e-12);
        assert!(round_half_even(f64::NAN, 4).is_nan());
        assert_eq!(round_half_even(f64::INFINITY, 4), f64::INFINITY);
    }
    #[test]
    fn aggregate_rejects_an_empty_sample_set() {
        let error = aggregate(&[]).expect_err("no samples");
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Validation
        );
    }
    #[test]
    fn dispersion_of_nothing_is_zero() {
        assert_eq!(dispersion(&[]), RubricDispersion::default());
        assert!(!flags_dispersion(
            &[],
            DEFAULT_RANGE_THRESHOLD,
            DEFAULT_MAD_THRESHOLD
        ));
    }
}
