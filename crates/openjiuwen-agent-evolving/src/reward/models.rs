//! Reward records: dimensions, signals, per-dimension scores, and summaries.
//!
//! These are pure data types. The constructors and builders never reject a
//! value — they exist to keep call sites short — while
//! [`RewardDimension::validate`] and [`RewardSignal::validate`] enforce the
//! Python `pydantic` constraints and are applied by
//! [`RewardAggregator`](crate::reward::RewardAggregator) before any arithmetic.

use std::collections::BTreeMap;

use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// Default [`RewardDimension::weight`]: every dimension counts the same.
fn default_weight() -> f64 {
    1.0
}

/// Default [`RewardDimension::max_score`]: a normalised `0.0..=1.0` dimension.
fn default_max_score() -> f64 {
    1.0
}

/// Default [`RewardDimension::higher_is_better`]: a larger raw value is better.
fn default_higher_is_better() -> bool {
    true
}

/// Default [`RewardSignal::confidence`]: the signal is fully trusted.
fn default_confidence() -> f64 {
    1.0
}

/// One dimension of quality a reward pack scores, with its weight and scale.
///
/// Python: the per-criterion weight/scale records of
/// `openjiuwen.agent_evolving.reward`. A dimension is declared by a
/// [`RewardPack`](crate::reward::RewardPack) or directly by
/// [`RewardAggregator::new`](crate::reward::RewardAggregator::new).
///
/// # Example
///
/// ```
/// use openjiuwen_agent_evolving::reward::RewardDimension;
///
/// let dimension = RewardDimension::new("correctness")
///     .with_weight(0.6)
///     .with_max_score(10.0)
///     .with_higher_is_better(false);
/// assert_eq!(dimension.weight, 0.6);
/// assert_eq!(dimension.max_score, 10.0);
/// assert!(!dimension.higher_is_better);
/// assert!(dimension.validate().is_ok());
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct RewardDimension {
    /// The dimension name, unique within a pack; must not be blank.
    pub name: String,
    /// The relative weight in the weighted mean; must be finite and positive.
    #[serde(default = "default_weight")]
    pub weight: f64,
    /// The raw value that normalises to `1.0`; must be finite and positive.
    #[serde(default = "default_max_score")]
    pub max_score: f64,
    /// Whether a larger raw value is better; inverted during normalisation.
    #[serde(default = "default_higher_is_better")]
    pub higher_is_better: bool,
}

impl RewardDimension {
    /// Creates a dimension with the default weight `1.0`, `max_score` `1.0`,
    /// and `higher_is_better` `true`.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            weight: default_weight(),
            max_score: default_max_score(),
            higher_is_better: default_higher_is_better(),
        }
    }

    /// Sets the relative weight of this dimension in the weighted mean.
    #[must_use]
    pub fn with_weight(mut self, weight: f64) -> Self {
        self.weight = weight;
        self
    }

    /// Sets the raw value that normalises to `1.0`.
    #[must_use]
    pub fn with_max_score(mut self, max_score: f64) -> Self {
        self.max_score = max_score;
        self
    }

    /// Sets whether a larger raw value is better.
    #[must_use]
    pub fn with_higher_is_better(mut self, higher_is_better: bool) -> Self {
        self.higher_is_better = higher_is_better;
        self
    }

    /// Checks the constraints the aggregation relies on.
    ///
    /// The builders accept any value so that a builder chain reads without
    /// `?`; this method (and
    /// [`RewardAggregator::aggregate`](crate::reward::RewardAggregator::aggregate))
    /// is where a bad dimension surfaces.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] naming the rejected field when `name` is blank,
    /// `weight` is non-finite or not positive, or `max_score` is non-finite or
    /// not positive.
    pub fn validate(&self) -> Result<()> {
        if self.name.trim().is_empty() {
            return Err(Error::validation("RewardDimension.name must not be empty"));
        }
        if !self.weight.is_finite() || self.weight <= 0.0 {
            return Err(Error::validation(format!(
                "RewardDimension.weight must be finite and positive, got {}",
                self.weight
            )));
        }
        if !self.max_score.is_finite() || self.max_score <= 0.0 {
            return Err(Error::validation(format!(
                "RewardDimension.max_score must be finite and positive, got {}",
                self.max_score
            )));
        }
        Ok(())
    }
}

/// One measured value for a declared dimension.
///
/// Python: one item of the `reward/scoring.py` claim-status stream, reduced to
/// the numeric value plus provenance. Duplicate signals for the same dimension
/// are combined by confidence-weighted mean in the aggregator.
///
/// # Example
///
/// ```
/// use openjiuwen_agent_evolving::reward::RewardSignal;
///
/// let signal = RewardSignal::new("correctness", 0.75)
///     .with_source("evaluated_case")
///     .with_evidence("case-1")
///     .with_confidence(0.5);
/// assert_eq!(signal.dimension, "correctness");
/// assert_eq!(signal.confidence, 0.5);
/// assert!(signal.validate().is_ok());
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct RewardSignal {
    /// The declared dimension this signal measures; must not be blank.
    pub dimension: String,
    /// The raw measured value; must be finite.
    pub value: f64,
    /// Where the signal came from (a producer name); free-form.
    #[serde(default)]
    pub source: Option<String>,
    /// The evidence backing the value, typically a case or claim identifier.
    #[serde(default)]
    pub evidence: Option<String>,
    /// How much the value is trusted, in `0.0..=1.0`; used as the merge weight.
    #[serde(default = "default_confidence")]
    pub confidence: f64,
}

impl RewardSignal {
    /// Creates a signal for `dimension` measuring `value`, with full
    /// confidence and no provenance.
    pub fn new(dimension: impl Into<String>, value: f64) -> Self {
        Self {
            dimension: dimension.into(),
            value,
            source: None,
            evidence: None,
            confidence: default_confidence(),
        }
    }

    /// Records where the signal came from.
    #[must_use]
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    /// Records the evidence backing the value.
    #[must_use]
    pub fn with_evidence(mut self, evidence: impl Into<String>) -> Self {
        self.evidence = Some(evidence.into());
        self
    }

    /// Sets the trust placed in the value.
    #[must_use]
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence;
        self
    }

    /// Checks the constraints the aggregation relies on.
    ///
    /// The builders accept any value so that a builder chain reads without
    /// `?`; this method (and
    /// [`RewardAggregator::aggregate`](crate::reward::RewardAggregator::aggregate))
    /// is where a bad signal surfaces.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] naming the rejected field when `dimension` is
    /// blank, `value` is non-finite, or `confidence` is non-finite or outside
    /// `0.0..=1.0`.
    pub fn validate(&self) -> Result<()> {
        if self.dimension.trim().is_empty() {
            return Err(Error::validation(
                "RewardSignal.dimension must not be empty",
            ));
        }
        if !self.value.is_finite() {
            return Err(Error::validation(format!(
                "RewardSignal.value must be finite, got {}",
                self.value
            )));
        }
        if !self.confidence.is_finite() || !(0.0..=1.0).contains(&self.confidence) {
            return Err(Error::validation(format!(
                "RewardSignal.confidence must be finite and between 0.0 and 1.0, got {}",
                self.confidence
            )));
        }
        Ok(())
    }
}

/// The scored result of one dimension: its raw value, its normalised value, and
/// the dimension weight used in the weighted mean.
///
/// Python: the per-criterion rows of `reward/scoring.py`'s `to_result`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct DimensionScore {
    /// The dimension name this row scores.
    pub name: String,
    /// The signal value after duplicate combination, before normalisation.
    pub raw: f64,
    /// The normalised value in `0.0..=1.0` (unclamped only for `Raw`).
    pub normalised: f64,
    /// The dimension weight applied in the weighted mean.
    pub weight: f64,
}

impl DimensionScore {
    /// Creates one scored dimension row.
    #[must_use]
    pub fn new(name: impl Into<String>, raw: f64, normalised: f64, weight: f64) -> Self {
        Self {
            name: name.into(),
            raw,
            normalised,
            weight,
        }
    }

    /// The dimension name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The combined raw value.
    #[must_use]
    pub fn raw(&self) -> f64 {
        self.raw
    }

    /// The normalised value.
    #[must_use]
    pub fn normalised(&self) -> f64 {
        self.normalised
    }

    /// The dimension weight.
    #[must_use]
    pub fn weight(&self) -> f64 {
        self.weight
    }
}

/// The aggregated outcome of one reward run: the weighted total, the
/// per-dimension scores, the dimensions that produced no signal, and whether
/// the total reached the threshold.
///
/// Python: the `total`/`score` summary of `reward/scoring.py`. The total is the
/// weight-averaged normalised score of the dimensions that contributed, so a
/// score is comparable across packs that declare different dimensions.
///
/// # Example
///
/// ```
/// use openjiuwen_agent_evolving::reward::RewardSummary;
///
/// let summary = RewardSummary::zero();
/// assert!(summary.total().abs() < f64::EPSILON);
/// assert!(!summary.passed());
/// assert_eq!(summary.num_signals(), 0);
/// ```
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct RewardSummary {
    /// The weighted mean of the normalised dimension scores in `0.0..=1.0`.
    #[serde(default)]
    pub total: f64,
    /// The scored dimensions, keyed by name; missing-and-skipped dimensions are
    /// absent.
    #[serde(default)]
    pub per_dimension: BTreeMap<String, DimensionScore>,
    /// The declared dimensions that produced no signal, sorted by name.
    #[serde(default)]
    pub missing: Vec<String>,
    /// How many signals were supplied, including signals for undeclared
    /// dimensions.
    #[serde(default)]
    pub num_signals: usize,
    /// Whether `total` reached the aggregator threshold.
    #[serde(default)]
    pub passed: bool,
}

impl RewardSummary {
    /// An empty summary: zero total, no dimensions, nothing passed.
    ///
    /// Used for empty input where no dimension could be measured.
    #[must_use]
    pub fn zero() -> Self {
        Self::default()
    }

    /// Builds a summary from the scored dimensions.
    ///
    /// The total is `sum(weight * normalised) / sum(weight)` over the rows of
    /// `per_dimension`, and `0.0` when there are none or the weights do not sum
    /// to a positive, finite value. `passed` is `total >= threshold`.
    #[must_use]
    pub fn from_signals(
        signals: &[RewardSignal],
        per_dimension: BTreeMap<String, DimensionScore>,
        missing: Vec<String>,
        threshold: f64,
    ) -> Self {
        let total_weight: f64 = per_dimension.values().map(|score| score.weight).sum();
        let weighted: f64 = per_dimension
            .values()
            .map(|score| score.weight * score.normalised)
            .sum();
        let total = if total_weight.is_finite() && total_weight > 0.0 {
            weighted / total_weight
        } else {
            0.0
        };
        let total = if total.is_finite() { total } else { 0.0 };
        Self {
            total,
            per_dimension,
            missing,
            num_signals: signals.len(),
            passed: total >= threshold,
        }
    }

    /// The weighted total in `0.0..=1.0` (`Raw` normalisation may exceed it).
    #[must_use]
    pub fn total(&self) -> f64 {
        self.total
    }

    /// Whether the total reached the aggregator threshold.
    #[must_use]
    pub fn passed(&self) -> bool {
        self.passed
    }

    /// The declared dimensions that produced no signal, sorted by name.
    #[must_use]
    pub fn missing(&self) -> &[String] {
        &self.missing
    }

    /// How many signals were supplied to the aggregation.
    #[must_use]
    pub fn num_signals(&self) -> usize {
        self.num_signals
    }

    /// The scored dimensions, keyed by name.
    #[must_use]
    pub fn per_dimension(&self) -> &BTreeMap<String, DimensionScore> {
        &self.per_dimension
    }

    /// The score of one dimension, when it contributed to the total.
    #[must_use]
    pub fn dimension(&self, name: &str) -> Option<&DimensionScore> {
        self.per_dimension.get(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dimension_builders_and_defaults() {
        let dimension = RewardDimension::new("accuracy");
        assert_eq!(dimension.name, "accuracy");
        assert!((dimension.weight - 1.0).abs() < f64::EPSILON);
        assert!((dimension.max_score - 1.0).abs() < f64::EPSILON);
        assert!(dimension.higher_is_better);
        assert!(dimension.validate().is_ok());

        let dimension = dimension
            .with_weight(0.25)
            .with_max_score(4.0)
            .with_higher_is_better(false);
        assert!((dimension.weight - 0.25).abs() < f64::EPSILON);
        assert!((dimension.max_score - 4.0).abs() < f64::EPSILON);
        assert!(!dimension.higher_is_better);
    }

    #[test]
    fn dimension_validate_rejects_blank_name_and_bad_numbers() {
        assert!(RewardDimension::new("  ").validate().is_err());
        assert!(RewardDimension::new("x")
            .with_weight(0.0)
            .validate()
            .is_err());
        assert!(RewardDimension::new("x")
            .with_weight(-1.0)
            .validate()
            .is_err());
        assert!(RewardDimension::new("x")
            .with_weight(f64::NAN)
            .validate()
            .is_err());
        assert!(RewardDimension::new("x")
            .with_max_score(0.0)
            .validate()
            .is_err());
        assert!(RewardDimension::new("x")
            .with_max_score(f64::INFINITY)
            .validate()
            .is_err());
    }

    #[test]
    fn signal_builders_accept_bad_values_but_validate_rejects_them() {
        let signal = RewardSignal::new("correctness", 0.5);
        assert!((signal.confidence - 1.0).abs() < f64::EPSILON);
        assert!(signal.source.is_none());
        assert!(signal.evidence.is_none());
        assert!(signal.validate().is_ok());

        let built = signal
            .clone()
            .with_source("metric")
            .with_evidence("case-1")
            .with_confidence(-0.5);
        assert_eq!(built.source.as_deref(), Some("metric"));
        assert_eq!(built.evidence.as_deref(), Some("case-1"));
        assert!(built.validate().is_err(), "builder accepted a bad value");

        assert!(RewardSignal::new(" ", 0.5).validate().is_err());
        assert!(RewardSignal::new("x", f64::NAN).validate().is_err());
        assert!(RewardSignal::new("x", 0.5)
            .with_confidence(f64::NAN)
            .validate()
            .is_err());
        assert!(RewardSignal::new("x", 0.5)
            .with_confidence(1.5)
            .validate()
            .is_err());
        assert!(RewardSignal::new("x", 0.5)
            .with_confidence(0.0)
            .validate()
            .is_ok());
        assert!(RewardSignal::new("x", 0.5)
            .with_confidence(1.0)
            .validate()
            .is_ok());
    }

    #[test]
    fn dimension_and_signal_json_round_trip_with_defaults() {
        let dimension: RewardDimension =
            serde_json::from_str(r#"{"name": "correctness"}"#).unwrap();
        assert!((dimension.weight - 1.0).abs() < f64::EPSILON);
        assert!((dimension.max_score - 1.0).abs() < f64::EPSILON);
        assert!(dimension.higher_is_better);
        let value = serde_json::to_value(&dimension).unwrap();
        assert_eq!(value["name"], "correctness");
        let restored: RewardDimension = serde_json::from_value(value).unwrap();
        assert_eq!(restored, dimension);

        let signal: RewardSignal =
            serde_json::from_str(r#"{"dimension": "clarity", "value": 0.25}"#).unwrap();
        assert!((signal.confidence - 1.0).abs() < f64::EPSILON);
        assert!(signal.source.is_none());
        let value = serde_json::to_value(&signal).unwrap();
        assert_eq!(value["confidence"], 1.0);
        let restored: RewardSignal = serde_json::from_value(value).unwrap();
        assert_eq!(restored, signal);

        // Unknown fields are tolerated.
        let permissive: RewardDimension =
            serde_json::from_str(r#"{"name": "x", "bogus": true}"#).unwrap();
        assert_eq!(permissive.name, "x");
    }

    #[test]
    fn summary_from_signals_weights_and_threshold() {
        let mut per_dimension = BTreeMap::new();
        per_dimension.insert(
            "correctness".to_owned(),
            DimensionScore::new("correctness", 0.9, 0.9, 0.6),
        );
        per_dimension.insert(
            "clarity".to_owned(),
            DimensionScore::new("clarity", 0.2, 0.2, 0.4),
        );
        let signals = [RewardSignal::new("correctness", 0.9)];
        let summary = RewardSummary::from_signals(
            &signals,
            per_dimension,
            vec!["efficiency".to_owned()],
            0.6,
        );
        // (0.6 * 0.9 + 0.4 * 0.2) / 1.0 = 0.62
        assert!((summary.total() - 0.62).abs() < 1e-9);
        assert!(summary.passed());
        assert_eq!(summary.num_signals(), 1);
        assert_eq!(summary.missing(), ["efficiency".to_owned()]);
        assert!((summary.dimension("clarity").unwrap().raw - 0.2).abs() < f64::EPSILON);

        let summary = RewardSummary::from_signals(&[], BTreeMap::new(), Vec::new(), 0.6);
        assert!(summary.total().abs() < f64::EPSILON);
        assert!(!summary.passed());
    }

    #[test]
    fn summary_zero_and_serde_round_trip() {
        let zero = RewardSummary::zero();
        assert!(zero.total().abs() < f64::EPSILON);
        assert!(zero.per_dimension().is_empty(), "expected no dimensions");
        assert!(zero.missing().is_empty(), "expected no missing dimensions");
        assert_eq!(zero.num_signals(), 0);
        assert!(!zero.passed());

        let from_empty_object: RewardSummary = serde_json::from_str("{}").unwrap();
        assert_eq!(from_empty_object, zero);

        let mut summary = zero.clone();
        summary.total = 0.75;
        summary.passed = true;
        summary.num_signals = 2;
        summary.per_dimension.insert(
            "correctness".to_owned(),
            DimensionScore::new("c", 0.75, 0.75, 1.0),
        );
        summary.missing.push("clarity".to_owned());
        let value = serde_json::to_value(&summary).unwrap();
        for key in ["total", "per_dimension", "missing", "num_signals", "passed"] {
            assert!(value.get(key).is_some(), "missing field {key}");
        }
        let restored: RewardSummary = serde_json::from_value(value).unwrap();
        assert_eq!(restored, summary);
    }
}
