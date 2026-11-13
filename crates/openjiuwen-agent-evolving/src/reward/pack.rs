//! Reward packs: the named dimension set a run scores against, plus the helpers
//! that turn evaluated cases into signals.
//!
//! Python: the batch-level parts of `openjiuwen.agent_evolving.reward`
//! (`rewardpack.py`) reduced to data. A pack is a versioned list of
//! [`RewardDimension`]s with a passing score; scoring it is delegated to the
//! [`RewardAggregator`] the pack builds, so packs and ad-hoc aggregators share
//! one implementation.

use std::collections::BTreeSet;

use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};

use super::models::{RewardDimension, RewardSignal, RewardSummary};
use super::scoring::{MissingDimensionPolicy, RewardAggregator, ScoreNormalisation};
use crate::dataset::EvaluatedCase;

/// The dimension [`score_evaluated_cases`] measures by default.
const CORRECTNESS: &str = "correctness";

/// The dimension of the batch-level pass-rate signal.
const PASS_RATE: &str = "pass_rate";

/// Default [`RewardPack::passing_score`].
fn default_passing_score() -> f64 {
    0.6
}

/// A named, versioned set of reward dimensions and the score a run must reach.
///
/// Python: the pack description of `reward/rewardpack.py`. The JSON shape is
/// the Rust record itself rather than Python's directory layout, so a pack can
/// travel with a run's results.
///
/// # Example
///
/// ```
/// use openjiuwen_agent_evolving::reward::RewardPack;
///
/// let pack = RewardPack::default_pack();
/// assert_eq!(pack.name, "default");
/// assert_eq!(pack.dimensions.len(), 3);
/// assert!(pack.validate().is_ok());
/// assert!((pack.aggregator().threshold() - 0.6).abs() < f64::EPSILON);
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct RewardPack {
    /// The pack name; must not be blank.
    pub name: String,
    /// The pack version; must not be blank.
    pub version: String,
    /// The scored dimensions, with unique names.
    pub dimensions: Vec<RewardDimension>,
    /// The total at or above which the run counts as passed, in `0.0..=1.0`.
    #[serde(default = "default_passing_score")]
    pub passing_score: f64,
}

impl RewardPack {
    /// The built-in pack: correctness `0.6`, efficiency `0.2`, clarity `0.2`,
    /// passing score `0.6`.
    ///
    /// The weights sum to `1.0`, so each normalised dimension contributes its
    /// weight directly to the total.
    #[must_use]
    pub fn default_pack() -> Self {
        Self {
            name: "default".to_owned(),
            version: "1".to_owned(),
            dimensions: vec![
                RewardDimension::new(CORRECTNESS).with_weight(0.6),
                RewardDimension::new("efficiency").with_weight(0.2),
                RewardDimension::new("clarity").with_weight(0.2),
            ],
            passing_score: default_passing_score(),
        }
    }

    /// Checks the pack constraints.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] when `name` or `version` is blank, `dimensions` is
    /// empty, a dimension name repeats, a dimension is
    /// [invalid](RewardDimension::validate), or `passing_score` is not finite or
    /// outside `0.0..=1.0`.
    pub fn validate(&self) -> Result<()> {
        if self.name.trim().is_empty() {
            return Err(Error::validation("RewardPack.name must not be empty"));
        }
        if self.version.trim().is_empty() {
            return Err(Error::validation("RewardPack.version must not be empty"));
        }
        if self.dimensions.is_empty() {
            return Err(Error::validation("RewardPack.dimensions must not be empty"));
        }
        let mut seen = BTreeSet::new();
        for dimension in &self.dimensions {
            dimension.validate()?;
            if !seen.insert(dimension.name.as_str()) {
                return Err(Error::validation(format!(
                    "RewardPack has a duplicate dimension: {}",
                    dimension.name
                )));
            }
        }
        if !self.passing_score.is_finite() || !(0.0..=1.0).contains(&self.passing_score) {
            return Err(Error::validation(format!(
                "RewardPack.passing_score must be finite and between 0.0 and 1.0, got {}",
                self.passing_score
            )));
        }
        Ok(())
    }

    /// The aggregator this pack describes: [`ScoreNormalisation::Clamp`],
    /// [`MissingDimensionPolicy::Skip`], and the pack's
    /// [`passing_score`](RewardPack::passing_score) as the threshold.
    ///
    /// Duplicate dimension names keep the last occurrence, matching
    /// [`RewardAggregator::new`].
    #[must_use]
    pub fn aggregator(&self) -> RewardAggregator {
        RewardAggregator::new(self.dimensions.clone())
            .with_normalisation(ScoreNormalisation::Clamp)
            .with_policy(MissingDimensionPolicy::Skip)
            .with_threshold(self.passing_score)
    }

    /// Scores `signals` with this pack's aggregator.
    ///
    /// # Errors
    ///
    /// The same validation errors as [`RewardAggregator::aggregate`]: the pack
    /// itself is not re-validated here, so call
    /// [`validate`](RewardPack::validate) first when the pack came from
    /// untrusted input.
    pub fn aggregate(&self, signals: &[RewardSignal]) -> Result<RewardSummary> {
        self.aggregator().aggregate(signals)
    }
}

impl Default for RewardPack {
    /// The built-in pack; see [`RewardPack::default_pack`].
    fn default() -> Self {
        Self::default_pack()
    }
}

/// The built-in pack; a free-function spelling of [`RewardPack::default_pack`].
#[must_use]
pub fn default_pack() -> RewardPack {
    RewardPack::default_pack()
}

/// Scores evaluated cases with `pack`.
///
/// Each case contributes one signal for the pack's correctness dimension — or
/// the pack's first dimension when it declares none — with the case's
/// [`score`](crate::dataset::EvaluatedCase::score) as the
/// value and its case id as the evidence. One batch-level `pass_rate` signal
/// (the fraction of cases at or above the pack's passing score) is added too;
/// the default pack does not declare that dimension, so the aggregator ignores
/// it for scoring while still counting it.
///
/// The aggregation uses [`MissingDimensionPolicy::Skip`]: a batch that cannot
/// measure every declared dimension is scored on what it did measure and lists
/// the rest in [`RewardSummary::missing`] rather than failing. Empty input
/// produces an all-zero summary that lists the pack's dimensions.
///
/// Non-finite case scores count as `0.0`.
#[must_use]
pub fn score_evaluated_cases(cases: &[EvaluatedCase], pack: &RewardPack) -> RewardSummary {
    let mut signals = Vec::with_capacity(cases.len() + 1);
    let mut passing = 0_usize;
    for case in cases {
        let mut signal = reward_from_score(case.score, pack);
        signal.evidence = Some(case.case_id().to_owned());
        signals.push(signal);
        if case.score >= pack.passing_score {
            passing += 1;
        }
    }
    if !cases.is_empty() {
        let pass_rate = passing as f64 / cases.len() as f64;
        signals.push(RewardSignal::new(PASS_RATE, pass_rate).with_source("evaluated_cases"));
    }
    match pack.aggregator().aggregate(&signals) {
        Ok(summary) => summary,
        Err(error) => {
            // The synthesised signals are valid and the pack aggregator skips
            // missing dimensions, so this is only reachable for a pack that was
            // never validated (for example a blank dimension name).
            tracing::warn!(%error, "reward pack aggregation failed; reporting an empty summary");
            RewardSummary::zero()
        }
    }
}

/// Builds one signal from a case score.
///
/// The signal measures the pack's `correctness` dimension when the pack declares
/// it, otherwise the pack's first dimension, and falls back to `correctness`
/// when the pack declares none. The source is `evaluated_case` and the
/// confidence is `1.0`; a non-finite score becomes `0.0`.
#[must_use]
pub fn reward_from_score(score: f64, pack: &RewardPack) -> RewardSignal {
    let dimension = pack
        .dimensions
        .iter()
        .find(|dimension| dimension.name == CORRECTNESS)
        .or_else(|| pack.dimensions.first())
        .map_or(CORRECTNESS, |dimension| dimension.name.as_str());
    let value = if score.is_finite() { score } else { 0.0 };
    RewardSignal::new(dimension, value).with_source("evaluated_case")
}
