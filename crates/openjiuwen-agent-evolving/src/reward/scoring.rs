//! Normalisation, missing-dimension policies, weighted aggregation, and score
//! statistics.
//!
//! Python: the arithmetic of `openjiuwen.agent_evolving.reward.scoring`, which
//! turns claim statuses into `0..100` values in code. The Rust port keeps the
//! same shape — a raw value per dimension, a normalisation step, and a weighted
//! mean — but takes the values as supplied [`RewardSignal`]s instead of
//! extracting them from model replies, so the whole module is deterministic.

use std::collections::BTreeMap;

use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};

use super::models::{DimensionScore, RewardDimension, RewardSignal, RewardSummary};

/// How a raw dimension value becomes a comparable `0.0..=1.0` score.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScoreNormalisation {
    /// Divide by [`RewardDimension::max_score`] and invert when the dimension is
    /// not `higher_is_better`, but keep the result unclamped.
    ///
    /// Use this when out-of-range raw values are meaningful (for example a
    /// penalty that should push the total below zero); the result is still
    /// always finite.
    Raw,
    /// Divide by [`RewardDimension::max_score`], invert when needed, and clamp
    /// to `0.0..=1.0`.
    ///
    /// This is the default and the policy [`RewardPack`](crate::reward::RewardPack)
    /// uses.
    #[default]
    Clamp,
    /// Rescale `value` from `min..=max` to `0.0..=1.0`, then invert when needed
    /// and clamp to `0.0..=1.0`.
    ///
    /// The declared `max` replaces [`RewardDimension::max_score`] as the upper
    /// bound; a degenerate or non-finite range yields `0.0`.
    MinMax {
        /// The raw value that normalises to `0.0`.
        min: f64,
        /// The raw value that normalises to `1.0`.
        max: f64,
    },
}

impl ScoreNormalisation {
    /// Normalises one raw dimension value.
    ///
    /// The result is always finite. Non-finite input, a non-finite or
    /// non-positive [`max_score`](RewardDimension::max_score), and a degenerate
    /// `MinMax` range (`max <= min`, or a non-finite bound) all yield `0.0`
    /// instead of `NaN`/`inf`.
    #[must_use]
    pub fn apply(&self, dimension: &RewardDimension, value: f64) -> f64 {
        let max_score = dimension.max_score;
        if !value.is_finite() || !max_score.is_finite() || max_score <= 0.0 {
            return 0.0;
        }
        let base = match *self {
            Self::Raw | Self::Clamp => value / max_score,
            Self::MinMax { min, max } => {
                if !min.is_finite() || !max.is_finite() || max - min <= 0.0 {
                    return 0.0;
                }
                (value - min) / (max - min)
            }
        };
        let base = if dimension.higher_is_better {
            base
        } else {
            1.0 - base
        };
        match *self {
            Self::Raw => {
                if base.is_finite() {
                    base
                } else {
                    0.0
                }
            }
            Self::Clamp | Self::MinMax { .. } => base.clamp(0.0, 1.0),
        }
    }
}

/// What to do with a declared dimension that produced no signal.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MissingDimensionPolicy {
    /// Score the dimension as `0.0` and keep its weight in the mean, so an
    /// unmeasured dimension drags the total down.
    Zero,
    /// Leave the dimension out of the mean and list it in
    /// [`RewardSummary::missing`].
    #[default]
    Skip,
    /// Reject the aggregation with [`Error::validation`] naming every missing
    /// dimension.
    Fail,
}

/// One dimension's accumulated signals.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Accumulator {
    /// `sum(value * confidence)`.
    weighted_sum: f64,
    /// `sum(confidence)`.
    confidence_sum: f64,
    /// The first signal's value, used when every confidence is zero.
    first_value: f64,
}

/// Weighted aggregation of [`RewardSignal`]s over a declared set of
/// [`RewardDimension`]s.
///
/// The aggregator owns the scoring policy: how raw values are normalised, what
/// happens to declared dimensions without a signal, and the pass threshold.
///
/// # Example
///
/// ```
/// use openjiuwen_agent_evolving::reward::{RewardAggregator, RewardDimension, RewardSignal};
///
/// let aggregator = RewardAggregator::new([
///     RewardDimension::new("correctness").with_weight(0.75),
///     RewardDimension::new("efficiency").with_weight(0.25),
/// ])
/// .with_threshold(0.5);
/// let summary = aggregator.aggregate(&[RewardSignal::new("correctness", 0.8)])?;
/// // `efficiency` produced no signal, so `Skip` drops it from the weighted mean
/// // and reports it instead of dragging the total down.
/// assert!((summary.total() - 0.8).abs() < 1e-9);
/// assert_eq!(summary.missing(), ["efficiency".to_owned()]);
/// assert!(summary.passed());
/// # Ok::<(), openjiuwen_common::error::Error>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct RewardAggregator {
    /// The declared dimensions, keyed by name (the order of the mean).
    dimensions: BTreeMap<String, RewardDimension>,
    /// How raw values become normalised scores.
    normalisation: ScoreNormalisation,
    /// What happens to a declared dimension without a signal.
    policy: MissingDimensionPolicy,
    /// The total at or above which the run counts as passed.
    threshold: f64,
}

impl RewardAggregator {
    /// Declares the dimensions this aggregator scores.
    ///
    /// Duplicate names keep the last occurrence, so a pack that repeats a
    /// dimension is scored with the final definition. The aggregator otherwise
    /// starts with [`ScoreNormalisation::Clamp`],
    /// [`MissingDimensionPolicy::Skip`], and a threshold of `0.6`.
    pub fn new(dimensions: impl IntoIterator<Item = RewardDimension>) -> Self {
        Self {
            dimensions: dimensions
                .into_iter()
                .map(|dimension| (dimension.name.clone(), dimension))
                .collect(),
            normalisation: ScoreNormalisation::default(),
            policy: MissingDimensionPolicy::default(),
            threshold: 0.6,
        }
    }

    /// Sets how raw values become normalised scores.
    #[must_use]
    pub fn with_normalisation(mut self, normalisation: ScoreNormalisation) -> Self {
        self.normalisation = normalisation;
        self
    }

    /// Sets what happens to a declared dimension without a signal.
    #[must_use]
    pub fn with_policy(mut self, policy: MissingDimensionPolicy) -> Self {
        self.policy = policy;
        self
    }

    /// Sets the total at or above which the run counts as passed.
    #[must_use]
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold;
        self
    }

    /// The declared dimensions, keyed by name.
    #[must_use]
    pub fn dimensions(&self) -> &BTreeMap<String, RewardDimension> {
        &self.dimensions
    }

    /// How raw values become normalised scores.
    #[must_use]
    pub fn normalisation(&self) -> ScoreNormalisation {
        self.normalisation
    }

    /// What happens to a declared dimension without a signal.
    #[must_use]
    pub fn policy(&self) -> MissingDimensionPolicy {
        self.policy
    }

    /// The total at or above which the run counts as passed.
    #[must_use]
    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    /// Scores `signals` against the declared dimensions.
    ///
    /// Every signal is validated first. Signals naming a dimension that was not
    /// declared are ignored for scoring but still counted in
    /// [`RewardSummary::num_signals`], so the count always reflects the input.
    /// Duplicate signals for one dimension combine by confidence-weighted mean
    /// `sum(value * confidence) / sum(confidence)`; when every confidence is
    /// zero the first signal's value is used. A declared dimension without a
    /// signal follows the [`MissingDimensionPolicy`].
    ///
    /// # Errors
    ///
    /// [`Error::validation`] naming the signal index and dimension for an
    /// invalid signal, and — under [`MissingDimensionPolicy::Fail`] — naming
    /// every declared dimension that produced no signal.
    pub fn aggregate(&self, signals: &[RewardSignal]) -> Result<RewardSummary> {
        for (index, signal) in signals.iter().enumerate() {
            signal.validate().map_err(|error| {
                Error::validation(format!(
                    "invalid reward signal at index {index} for dimension {:?}: {error}",
                    signal.dimension
                ))
            })?;
        }

        let mut grouped: BTreeMap<String, Accumulator> = BTreeMap::new();
        for signal in signals {
            if !self.dimensions.contains_key(&signal.dimension) {
                continue;
            }
            let accumulator = grouped
                .entry(signal.dimension.clone())
                .or_insert(Accumulator {
                    weighted_sum: 0.0,
                    confidence_sum: 0.0,
                    first_value: signal.value,
                });
            accumulator.weighted_sum += signal.value * signal.confidence;
            accumulator.confidence_sum += signal.confidence;
        }

        let mut per_dimension = BTreeMap::new();
        let mut missing = Vec::new();
        // A `BTreeMap` iterates in name order, so `missing` is sorted.
        for (name, dimension) in &self.dimensions {
            if let Some(accumulated) = grouped.get(name) {
                let raw = if accumulated.confidence_sum > 0.0 {
                    accumulated.weighted_sum / accumulated.confidence_sum
                } else {
                    accumulated.first_value
                };
                let normalised = self.normalisation.apply(dimension, raw);
                per_dimension.insert(
                    name.clone(),
                    DimensionScore::new(name.clone(), raw, normalised, dimension.weight),
                );
            } else {
                missing.push(name.clone());
                if self.policy == MissingDimensionPolicy::Zero {
                    per_dimension.insert(
                        name.clone(),
                        DimensionScore::new(name.clone(), 0.0, 0.0, dimension.weight),
                    );
                }
            }
        }

        if self.policy == MissingDimensionPolicy::Fail && !missing.is_empty() {
            return Err(Error::validation(format!(
                "missing reward dimensions: {}",
                missing.join(", ")
            )));
        }

        Ok(RewardSummary::from_signals(
            signals,
            per_dimension,
            missing,
            self.threshold,
        ))
    }
}

/// Summary statistics over a batch of scores.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ScoreStats {
    /// How many scores were aggregated.
    #[serde(default)]
    pub count: usize,
    /// The arithmetic mean; `0.0` for an empty batch.
    #[serde(default)]
    pub mean: f64,
    /// The smallest score; `0.0` for an empty batch.
    #[serde(default)]
    pub min: f64,
    /// The largest score; `0.0` for an empty batch.
    #[serde(default)]
    pub max: f64,
    /// The population standard deviation; `0.0` for an empty batch.
    #[serde(default)]
    pub std_dev: f64,
}

/// Aggregates a batch of scores into [`ScoreStats`].
///
/// Non-finite values become `0.0` before the statistics are computed, so the
/// result is always deterministic and never `NaN`. `std_dev` is the population
/// standard deviation (dividing by `count`), matching
/// `statistics.pstdev`.
#[must_use]
pub fn aggregate_scores(scores: &[f64]) -> ScoreStats {
    if scores.is_empty() {
        return ScoreStats::default();
    }
    let values: Vec<f64> = scores
        .iter()
        .map(|score| if score.is_finite() { *score } else { 0.0 })
        .collect();
    let count = values.len();
    let mean = values.iter().sum::<f64>() / count as f64;
    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let variance = values
        .iter()
        .map(|value| (value - mean) * (value - mean))
        .sum::<f64>()
        / count as f64;
    ScoreStats {
        count,
        mean,
        min,
        max,
        std_dev: variance.sqrt(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_clamp_and_skip() {
        assert_eq!(ScoreNormalisation::default(), ScoreNormalisation::Clamp);
        assert_eq!(
            MissingDimensionPolicy::default(),
            MissingDimensionPolicy::Skip
        );
    }

    #[test]
    fn normalisation_variants_rescale_and_clamp() {
        let dimension = RewardDimension::new("quality").with_max_score(2.0);
        assert!((ScoreNormalisation::Raw.apply(&dimension, 1.5) - 0.75).abs() < 1e-9);
        assert!((ScoreNormalisation::Raw.apply(&dimension, 3.0) - 1.5).abs() < 1e-9);
        assert!((ScoreNormalisation::Clamp.apply(&dimension, 3.0) - 1.0).abs() < 1e-9);
        assert!((ScoreNormalisation::Clamp.apply(&dimension, -1.0)).abs() < 1e-9);

        let min_max = ScoreNormalisation::MinMax {
            min: 10.0,
            max: 30.0,
        };
        assert!((min_max.apply(&dimension, 20.0) - 0.5).abs() < 1e-9);
        assert!((min_max.apply(&dimension, 40.0) - 1.0).abs() < 1e-9);
        assert!((min_max.apply(&dimension, 0.0)).abs() < 1e-9);

        let lower_is_better = RewardDimension::new("latency")
            .with_max_score(2.0)
            .with_higher_is_better(false);
        assert!((ScoreNormalisation::Clamp.apply(&lower_is_better, 0.5) - 0.75).abs() < 1e-9);
        assert!((ScoreNormalisation::Raw.apply(&lower_is_better, 0.5) - 0.75).abs() < 1e-9);
        assert!((min_max.apply(&lower_is_better, 10.0) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn normalisation_never_produces_nan_or_infinity() {
        let dimension = RewardDimension::new("quality");
        for normalisation in [
            ScoreNormalisation::Raw,
            ScoreNormalisation::Clamp,
            ScoreNormalisation::MinMax { min: 0.0, max: 1.0 },
        ] {
            for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                assert!(normalisation.apply(&dimension, value).abs() < f64::EPSILON);
            }
            let zero_max = RewardDimension::new("quality").with_max_score(0.0);
            assert!(normalisation.apply(&zero_max, 0.5).abs() < f64::EPSILON);
            let nan_max = RewardDimension::new("quality").with_max_score(f64::NAN);
            assert!(normalisation.apply(&nan_max, 0.5).abs() < f64::EPSILON);
        }
        let tiny_max = RewardDimension::new("quality").with_max_score(f64::MIN_POSITIVE);
        assert!(ScoreNormalisation::Raw
            .apply(&tiny_max, f64::MAX)
            .is_finite());
        let degenerate = ScoreNormalisation::MinMax { min: 5.0, max: 5.0 };
        assert!(degenerate.apply(&dimension, 5.0).abs() < f64::EPSILON);
        let non_finite = ScoreNormalisation::MinMax {
            min: f64::NAN,
            max: 1.0,
        };
        assert!(non_finite.apply(&dimension, 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn aggregator_builders_and_accessors() {
        let aggregator = RewardAggregator::new([
            RewardDimension::new("correctness").with_weight(0.6),
            RewardDimension::new("correctness").with_weight(1.0),
        ]);
        assert_eq!(aggregator.dimensions().len(), 1, "last duplicate wins");
        assert_eq!(
            aggregator.dimensions()["correctness"].weight,
            1.0,
            "last duplicate wins"
        );
        assert_eq!(aggregator.normalisation(), ScoreNormalisation::Clamp);
        assert_eq!(aggregator.policy(), MissingDimensionPolicy::Skip);
        assert!((aggregator.threshold() - 0.6).abs() < f64::EPSILON);

        let aggregator = aggregator
            .with_normalisation(ScoreNormalisation::Raw)
            .with_policy(MissingDimensionPolicy::Zero)
            .with_threshold(0.9);
        assert_eq!(aggregator.normalisation(), ScoreNormalisation::Raw);
        assert_eq!(aggregator.policy(), MissingDimensionPolicy::Zero);
        assert!((aggregator.threshold() - 0.9).abs() < f64::EPSILON);
    }

    #[test]
    fn weighted_total_is_hand_computable() {
        let aggregator = RewardAggregator::new([
            RewardDimension::new("a").with_weight(0.5),
            RewardDimension::new("b").with_weight(0.3),
            RewardDimension::new("c").with_weight(0.2),
        ])
        .with_threshold(0.6);
        let signals = [
            RewardSignal::new("a", 0.9),
            RewardSignal::new("b", 0.5),
            RewardSignal::new("c", 0.1),
        ];
        let summary = aggregator.aggregate(&signals).unwrap();
        // (0.5*0.9 + 0.3*0.5 + 0.2*0.1) / 1.0 = 0.62
        assert!((summary.total() - 0.62).abs() < 1e-9);
        assert!(summary.passed());
        assert!(
            summary.missing().is_empty(),
            "expected no missing dimensions"
        );
        assert_eq!(summary.num_signals(), 3);

        let strict = aggregator.with_threshold(0.7);
        assert!(!strict.aggregate(&signals).unwrap().passed());
    }

    #[test]
    fn duplicate_signals_combine_by_confidence() {
        let aggregator = RewardAggregator::new([RewardDimension::new("correctness")]);
        let signals = [
            RewardSignal::new("correctness", 1.0).with_confidence(0.2),
            RewardSignal::new("correctness", 0.5).with_confidence(0.6),
        ];
        let summary = aggregator.aggregate(&signals).unwrap();
        // (1.0*0.2 + 0.5*0.6) / 0.8 = 0.625
        assert!((summary.dimension("correctness").unwrap().raw - 0.625).abs() < 1e-9);
        assert!((summary.total() - 0.625).abs() < 1e-9);

        let zero_confidence = [
            RewardSignal::new("correctness", 0.9).with_confidence(0.0),
            RewardSignal::new("correctness", 0.1).with_confidence(0.0),
        ];
        let summary = aggregator.aggregate(&zero_confidence).unwrap();
        assert!((summary.dimension("correctness").unwrap().raw - 0.9).abs() < 1e-9);
    }

    #[test]
    fn undeclared_dimensions_are_counted_but_ignored() {
        let aggregator = RewardAggregator::new([RewardDimension::new("correctness")]);
        let signals = [
            RewardSignal::new("correctness", 0.5),
            RewardSignal::new("bogus", 1.0),
        ];
        let summary = aggregator.aggregate(&signals).unwrap();
        assert!((summary.total() - 0.5).abs() < 1e-9);
        assert_eq!(summary.num_signals(), 2);
        assert!(summary.dimension("bogus").is_none());
        assert!(
            summary.missing().is_empty(),
            "expected no missing dimensions"
        );
    }

    #[test]
    fn missing_dimension_policies() {
        let dimensions = [
            RewardDimension::new("correctness").with_weight(0.6),
            RewardDimension::new("clarity").with_weight(0.4),
        ];
        let signals = [RewardSignal::new("correctness", 0.8)];

        let fail =
            RewardAggregator::new(dimensions.clone()).with_policy(MissingDimensionPolicy::Fail);
        let error = fail.aggregate(&signals).unwrap_err();
        let message = error.to_string();
        assert!(
            message.contains("missing reward dimensions: clarity"),
            "unexpected message: {message}"
        );

        let fail_many = RewardAggregator::new([
            RewardDimension::new("correctness"),
            RewardDimension::new("efficiency"),
            RewardDimension::new("clarity"),
        ])
        .with_policy(MissingDimensionPolicy::Fail);
        let message = fail_many.aggregate(&[]).unwrap_err().to_string();
        assert!(
            message.contains("missing reward dimensions: clarity, correctness, efficiency"),
            "unexpected message: {message}"
        );

        let zero =
            RewardAggregator::new(dimensions.clone()).with_policy(MissingDimensionPolicy::Zero);
        let summary = zero.aggregate(&signals).unwrap();
        // (0.6*0.8 + 0.4*0.0) / 1.0 = 0.48
        assert!((summary.total() - 0.48).abs() < 1e-9);
        assert_eq!(summary.missing(), ["clarity".to_owned()]);
        assert!(
            summary.dimension("clarity").unwrap().normalised().abs() < f64::EPSILON,
            "zero policy scores the missing dimension as 0.0"
        );

        let skip = RewardAggregator::new(dimensions).with_policy(MissingDimensionPolicy::Skip);
        let summary = skip.aggregate(&signals).unwrap();
        assert!((summary.total() - 0.8).abs() < 1e-9);
        assert_eq!(summary.missing(), ["clarity".to_owned()]);
        assert!(summary.dimension("clarity").is_none());
    }

    #[test]
    fn invalid_signal_names_index_and_dimension() {
        let aggregator = RewardAggregator::new([RewardDimension::new("correctness")]);
        let signals = [
            RewardSignal::new("correctness", 0.5),
            RewardSignal::new("correctness", f64::NAN),
        ];
        let message = aggregator.aggregate(&signals).unwrap_err().to_string();
        assert!(message.contains("index 1"), "unexpected message: {message}");
        assert!(
            message.contains("correctness"),
            "unexpected message: {message}"
        );
        assert!(message.contains("value"), "unexpected message: {message}");
    }

    #[test]
    fn aggregate_scores_statistics() {
        let stats = aggregate_scores(&[0.2, 0.4, 0.6]);
        assert_eq!(stats.count, 3);
        assert!((stats.mean - 0.4).abs() < 1e-9);
        assert!((stats.min - 0.2).abs() < 1e-9);
        assert!((stats.max - 0.6).abs() < 1e-9);
        assert!((stats.std_dev - 0.163_299_316_185_545_2).abs() < 1e-9);

        let empty = aggregate_scores(&[]);
        assert_eq!(empty.count, 0);
        assert!(empty.mean.abs() < f64::EPSILON);
        assert!(empty.min.abs() < f64::EPSILON);
        assert!(empty.max.abs() < f64::EPSILON);
        assert!(empty.std_dev.abs() < f64::EPSILON);

        let sanitised = aggregate_scores(&[f64::NAN, 1.0, f64::NEG_INFINITY]);
        assert_eq!(sanitised.count, 3);
        assert!((sanitised.mean - 1.0 / 3.0).abs() < 1e-9);
        assert!(sanitised.min.abs() < f64::EPSILON);
        assert!((sanitised.max - 1.0).abs() < 1e-9);
        assert!(sanitised.std_dev.is_finite());
    }

    #[test]
    fn normalisation_policies_round_trip_through_json() {
        let value =
            serde_json::to_value(ScoreNormalisation::MinMax { min: 1.0, max: 3.0 }).unwrap();
        assert_eq!(value["min_max"]["min"], 1.0);
        let restored: ScoreNormalisation = serde_json::from_value(value).unwrap();
        assert_eq!(restored, ScoreNormalisation::MinMax { min: 1.0, max: 3.0 });
        let value = serde_json::to_value(MissingDimensionPolicy::Skip).unwrap();
        assert_eq!(value, serde_json::json!("skip"));
        let restored: MissingDimensionPolicy = serde_json::from_value(value).unwrap();
        assert_eq!(restored, MissingDimensionPolicy::Skip);
    }
}
