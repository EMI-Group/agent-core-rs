//! Judge rubrics: the scoring criteria and the scale a verdict is read on.

use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};

use super::DEFAULT_MAX_SCORE;

/// The normalised pass threshold used by [`JudgeRubric::default_rubric`].
const DEFAULT_PASSING_SCORE: f64 = 0.6;

/// One scoring criterion of a [`JudgeRubric`].
///
/// Python encodes its rubric in prose inside `LLM_METRIC_TEMPLATE`; here it is
/// data, so a caller can reweight or extend the judge without touching prompts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct JudgeCriterion {
    /// Stable criterion name; also the key used in
    /// [`JudgeVerdict::per_criterion`](super::JudgeVerdict::per_criterion) and in
    /// the prompt.
    pub name: String,
    /// What the criterion measures, rendered into the judge prompt.
    pub description: String,
    /// Relative weight used when a verdict only reports per-criterion scores;
    /// `1.0` by default.
    pub weight: f64,
}

impl Default for JudgeCriterion {
    /// An unnamed criterion with weight `1.0`, the Python template's default.
    fn default() -> Self {
        Self {
            name: String::new(),
            description: String::new(),
            weight: 1.0,
        }
    }
}

impl JudgeCriterion {
    /// Creates a weight-`1.0` criterion.
    #[must_use]
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            weight: 1.0,
        }
    }

    /// Replaces [`JudgeCriterion::weight`].
    #[must_use]
    pub fn with_weight(mut self, weight: f64) -> Self {
        self.weight = weight;
        self
    }

    /// Checks that the criterion can be used in a prompt.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] when the name is blank or the weight is not a
    /// finite, strictly positive number.
    pub fn validate(&self) -> Result<()> {
        if self.name.trim().is_empty() {
            return Err(Error::validation("judge criterion name must not be empty"));
        }
        if !self.weight.is_finite() || self.weight <= 0.0 {
            return Err(Error::validation(format!(
                "judge criterion `{}` weight must be finite and greater than zero, got {}",
                self.name, self.weight
            )));
        }
        Ok(())
    }
}

/// The scoring rubric handed to the judge model.
///
/// Scores from the model are read on the `0..=max_score` scale and normalised to
/// `0.0..=1.0`; [`passing_score`](Self::passing_score) is compared against that
/// normalised score, which mirrors Python's 0/1 pass/fail metric.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct JudgeRubric {
    /// The criteria the judge must score, in prompt order.
    pub criteria: Vec<JudgeCriterion>,
    /// Upper bound of the raw score scale; must be finite and positive.
    pub max_score: f64,
    /// Normalised score at or above which a verdict counts as passed.
    pub passing_score: f64,
}

impl Default for JudgeRubric {
    /// The Python-equivalent default rubric, see
    /// [`JudgeRubric::default_rubric`].
    fn default() -> Self {
        Self::default_rubric()
    }
}

impl JudgeRubric {
    /// Builds the default rubric: `correctness` (1.0), `completeness` (0.8), and
    /// `clarity` (0.5) on a `0..10` scale with a `0.6` pass threshold.
    #[must_use]
    pub fn default_rubric() -> Self {
        Self {
            criteria: vec![
                JudgeCriterion::new(
                    "correctness",
                    "The answer agrees with the expected label in meaning and conclusion.",
                )
                .with_weight(1.0),
                JudgeCriterion::new(
                    "completeness",
                    "The answer covers every part of the question that the expected label covers.",
                )
                .with_weight(0.8),
                JudgeCriterion::new(
                    "clarity",
                    "The answer is unambiguous and does not contradict itself.",
                )
                .with_weight(0.5),
            ],
            max_score: DEFAULT_MAX_SCORE,
            passing_score: DEFAULT_PASSING_SCORE,
        }
    }

    /// Builds a rubric with a single weight-`1.0` criterion on the default scale.
    #[must_use]
    pub fn single(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            criteria: vec![JudgeCriterion::new(name, description)],
            max_score: DEFAULT_MAX_SCORE,
            passing_score: DEFAULT_PASSING_SCORE,
        }
    }

    /// Builds a rubric from explicit parts; call [`JudgeRubric::validate`]
    /// before using it with a model.
    #[must_use]
    pub fn new(criteria: Vec<JudgeCriterion>, max_score: f64, passing_score: f64) -> Self {
        Self {
            criteria,
            max_score,
            passing_score,
        }
    }

    /// The criteria the judge must score, in prompt order.
    #[must_use]
    pub fn criteria(&self) -> &[JudgeCriterion] {
        &self.criteria
    }

    /// The upper bound of the raw score scale.
    #[must_use]
    pub fn max_score(&self) -> f64 {
        self.max_score
    }

    /// The normalised pass threshold.
    #[must_use]
    pub fn passing_score(&self) -> f64 {
        self.passing_score
    }

    /// Whether the rubric declares no criteria.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.criteria.is_empty()
    }

    /// Checks that the rubric can be rendered and used for scaling.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] when there are no criteria, when a criterion is
    /// invalid, when `max_score` is not finite and positive, or when
    /// `passing_score` is not finite and inside `0.0..=1.0`.
    pub fn validate(&self) -> Result<()> {
        if self.criteria.is_empty() {
            return Err(Error::validation(
                "judge rubric must declare at least one criterion",
            ));
        }
        for criterion in &self.criteria {
            criterion.validate()?;
        }
        if !self.max_score.is_finite() || self.max_score <= 0.0 {
            return Err(Error::validation(format!(
                "judge rubric max_score must be finite and greater than zero, got {}",
                self.max_score
            )));
        }
        if !self.passing_score.is_finite() || !(0.0..=1.0).contains(&self.passing_score) {
            return Err(Error::validation(format!(
                "judge rubric passing_score must be finite and between 0.0 and 1.0, got {}",
                self.passing_score
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_support::close;
    use super::*;

    #[test]
    fn default_rubric_is_valid_and_ordered() {
        let rubric = JudgeRubric::default();
        assert!(rubric.validate().is_ok());
        let names = rubric
            .criteria()
            .iter()
            .map(|criterion| criterion.name.as_str())
            .collect::<Vec<_>>();
        assert_eq!(names, ["correctness", "completeness", "clarity"]);
        assert!(close(rubric.max_score(), 10.0));
        assert!(close(rubric.passing_score(), 0.6));
        assert!(!rubric.is_empty());
        assert_eq!(JudgeCriterion::default().weight, 1.0);
        assert_eq!(
            JudgeRubric::single("accuracy", "is it right")
                .criteria()
                .len(),
            1
        );
    }

    #[test]
    fn rubric_validation_rejects_bad_scales() {
        let empty = JudgeRubric::new(Vec::new(), 10.0, 0.6)
            .validate()
            .unwrap_err();
        assert!(
            matches!(&empty, Error::Validation(message) if message.contains("at least one criterion"))
        );

        let weight = JudgeRubric::new(
            vec![JudgeCriterion::new("correctness", "x").with_weight(0.0)],
            10.0,
            0.6,
        )
        .validate()
        .unwrap_err();
        assert!(matches!(&weight, Error::Validation(message) if message.contains("weight")));

        let scale = JudgeRubric::single("correctness", "x");
        assert!(JudgeRubric::new(scale.criteria.clone(), 0.0, 0.6)
            .validate()
            .is_err());
        assert!(JudgeRubric::new(scale.criteria.clone(), 10.0, 1.5)
            .validate()
            .is_err());
        assert!(JudgeRubric::new(scale.criteria, 10.0, f64::NAN)
            .validate()
            .is_err());
        assert!(JudgeCriterion::new("", "x").validate().is_err());
    }
}
