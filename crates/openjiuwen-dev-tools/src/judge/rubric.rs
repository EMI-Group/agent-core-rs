//! Rubrics: the dimensions a judge scores and the weighting it applies.
//!
//! A [`Rubric`] is a named, described set of [`RubricDimension`]s plus the
//! numeric [`scale`](Rubric::scale) every dimension is scored on. Scores arrive
//! as raw values on that scale; [`Rubric::normalize`] turns them into one
//! `0.0..=1.0` judgement by clamping each dimension to its scale, multiplying by
//! the dimension weight, and dividing by the total weight.
//!
//! Nothing here contacts a model: the rubric is pure data, validated by
//! [`Rubric::validate`] before any judge call so a malformed rubric fails fast
//! with [`openjiuwen_common::error::Error::Validation`] instead of producing a
//! meaningless score.

use std::collections::{BTreeMap, BTreeSet};

use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// The normalized score at or above which a judgement passes (`0.6`).
pub const DEFAULT_PASS_THRESHOLD: f64 = 0.6;

/// One scoring dimension of a [`Rubric`].
///
/// `weight` is a positive multiplier applied to the normalized dimension score;
/// `guidance` is optional extra text handed to the model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct RubricDimension {
    /// The dimension name, unique within its rubric.
    pub name: String,
    /// What the dimension measures.
    pub description: String,
    /// The weighting multiplier; must be finite and greater than zero.
    pub weight: f64,
    /// Optional scoring guidance appended to the rendered prompt.
    pub guidance: String,
}

impl Default for RubricDimension {
    /// A blank dimension weighing `1.0`; only the fields are defaulted, so
    /// validation still rejects the blank name.
    fn default() -> Self {
        Self {
            name: String::new(),
            description: String::new(),
            weight: 1.0,
            guidance: String::new(),
        }
    }
}

impl RubricDimension {
    /// Creates a dimension with weight `1.0` and no guidance.
    #[must_use]
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            weight: 1.0,
            guidance: String::new(),
        }
    }

    /// Sets the weighting multiplier.
    #[must_use]
    pub fn with_weight(mut self, weight: f64) -> Self {
        self.weight = weight;
        self
    }

    /// Sets the scoring guidance.
    #[must_use]
    pub fn with_guidance(mut self, guidance: impl Into<String>) -> Self {
        self.guidance = guidance.into();
        self
    }
}

/// A named scoring rubric.
///
/// The [`scale`](Self::scale) bounds each dimension score (`0.0` to `scale`),
/// while the result of [`normalize`](Self::normalize) is always `0.0..=1.0`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct Rubric {
    /// The rubric name, shown to the model.
    pub name: String,
    /// What the rubric evaluates.
    pub description: String,
    /// The maximum raw score a single dimension may receive.
    pub scale: f64,
    /// The scored dimensions, in presentation order.
    pub dimensions: Vec<RubricDimension>,
}

impl Default for Rubric {
    /// A blank rubric on scale `1.0` with no dimensions.
    fn default() -> Self {
        Self {
            name: String::new(),
            description: String::new(),
            scale: 1.0,
            dimensions: Vec::new(),
        }
    }
}

impl Rubric {
    /// Creates an empty rubric on scale `1.0`.
    #[must_use]
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            ..Self::default()
        }
    }

    /// Appends a dimension.
    #[must_use]
    pub fn with_dimension(mut self, dimension: RubricDimension) -> Self {
        self.dimensions.push(dimension);
        self
    }

    /// Sets the raw maximum each dimension is scored against.
    #[must_use]
    pub fn with_scale(mut self, scale: f64) -> Self {
        self.scale = scale;
        self
    }

    /// Creates a rubric with one weight-`1` dimension per name.
    ///
    /// Each dimension's description mirrors its name, so a uniform rubric renders
    /// sensibly without further authoring.
    #[must_use]
    pub fn uniform(
        name: impl Into<String>,
        description: impl Into<String>,
        dimension_names: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        let mut rubric = Self::new(name, description);
        rubric.dimensions = dimension_names
            .into_iter()
            .map(|dimension_name| {
                let dimension_name = dimension_name.into();
                RubricDimension::new(dimension_name.clone(), dimension_name)
            })
            .collect();
        rubric
    }

    /// Checks that the rubric is usable, naming the first violation.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the name or description is blank, the
    /// scale is not finite and positive, there is no dimension, or a dimension
    /// has a blank/duplicate name or a non-positive weight.
    pub fn validate(&self) -> Result<()> {
        if self.name.trim().is_empty() {
            return Err(Error::validation("rubric name must not be blank"));
        }
        if self.description.trim().is_empty() {
            return Err(Error::validation("rubric description must not be blank"));
        }
        if !self.scale.is_finite() || self.scale <= 0.0 {
            return Err(Error::validation(format!(
                "rubric scale must be a finite number greater than zero, got {}",
                self.scale
            )));
        }
        if self.dimensions.is_empty() {
            return Err(Error::validation(
                "rubric must declare at least one dimension",
            ));
        }

        let mut seen = BTreeSet::new();
        for dimension in &self.dimensions {
            if dimension.name.trim().is_empty() {
                return Err(Error::validation("rubric dimension name must not be blank"));
            }
            if !seen.insert(dimension.name.as_str()) {
                return Err(Error::validation(format!(
                    "duplicate rubric dimension '{}'",
                    dimension.name
                )));
            }
            if !dimension.weight.is_finite() || dimension.weight <= 0.0 {
                return Err(Error::validation(format!(
                    "rubric dimension '{}' must have a finite weight greater than zero, got {}",
                    dimension.name, dimension.weight
                )));
            }
        }
        Ok(())
    }

    /// Looks a dimension up by exact name.
    #[must_use]
    pub fn dimension(&self, name: &str) -> Option<&RubricDimension> {
        self.dimensions
            .iter()
            .find(|dimension| dimension.name == name)
    }

    /// The sum of every dimension weight.
    #[must_use]
    pub fn max_total_weight(&self) -> f64 {
        self.dimensions
            .iter()
            .map(|dimension| dimension.weight)
            .sum()
    }

    /// Normalizes raw per-dimension scores into one `0.0..=1.0` score.
    ///
    /// Each present dimension contributes `clamp(score / scale, 0, 1) * weight`;
    /// missing dimensions contribute nothing, and the sum is divided by the total
    /// weight, so skipping a dimension is never rewarded.
    /// Non-finite input scores contribute `0.0`, and a rubric with no dimensions
    /// (or a total weight of zero) normalizes to `0.0`.
    #[must_use]
    pub fn normalize(&self, dimension_scores: &BTreeMap<String, f64>) -> f64 {
        let total_weight = self.max_total_weight();
        if self.dimensions.is_empty() || total_weight <= 0.0 {
            return 0.0;
        }
        let mut weighted = 0.0f64;
        for dimension in &self.dimensions {
            let raw = dimension_scores.get(&dimension.name).copied();
            let normalized = match raw {
                Some(score) if score.is_finite() => (score / self.scale).clamp(0.0, 1.0),
                _ => 0.0,
            };
            weighted += normalized * dimension.weight;
        }
        (weighted / total_weight).clamp(0.0, 1.0)
    }

    /// Whether `score` reaches [`DEFAULT_PASS_THRESHOLD`].
    ///
    /// The comparison is exactly `score >= 0.6`, with no epsilon.
    #[must_use]
    pub fn is_passing(&self, score: f64) -> bool {
        score >= DEFAULT_PASS_THRESHOLD
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scores(pairs: &[(&str, f64)]) -> BTreeMap<String, f64> {
        pairs
            .iter()
            .map(|(name, value)| ((*name).to_owned(), *value))
            .collect()
    }

    fn rubric() -> Rubric {
        Rubric::new("skill quality", "Does the skill do its job?")
            .with_dimension(RubricDimension::new("clarity", "Is it clear?").with_weight(1.0))
            .with_dimension(RubricDimension::new("depth", "Is it deep?").with_weight(3.0))
    }

    #[test]
    fn builders_apply_defaults() {
        let dimension = RubricDimension::new("clarity", "Is it clear?");
        assert_eq!(dimension.weight, 1.0);
        assert_eq!(dimension.guidance, "");

        let dimension = dimension
            .with_weight(2.0)
            .with_guidance("Prefer concrete examples.");
        assert_eq!(dimension.weight, 2.0);
        assert_eq!(dimension.guidance, "Prefer concrete examples.");

        let rubric = Rubric::new("demo", "A demo").with_scale(10.0);
        assert_eq!(rubric.scale, 10.0);
        assert_eq!(rubric.dimensions, Vec::new());
    }

    #[test]
    fn uniform_creates_one_weighted_dimension_per_name() {
        let rubric = Rubric::uniform("demo", "A demo", ["clarity", "depth"]);
        assert_eq!(rubric.dimensions.len(), 2);
        assert_eq!(rubric.dimensions[0].name, "clarity");
        assert_eq!(rubric.dimensions[0].description, "clarity");
        assert_eq!(rubric.dimensions[0].weight, 1.0);
        assert_eq!(rubric.max_total_weight(), 2.0);
        rubric.validate().expect("uniform rubric is valid");
    }

    #[test]
    fn validate_accepts_a_well_formed_rubric() {
        rubric().validate().expect("valid rubric");
        assert_eq!(rubric().max_total_weight(), 4.0);
    }

    #[test]
    fn validate_rejects_blank_name() {
        let rubric = Rubric::uniform("  ", "A demo", ["clarity"]);
        let error = rubric.validate().expect_err("blank name");
        assert!(error.to_string().contains("name"), "{error}");
    }

    #[test]
    fn validate_rejects_blank_description() {
        let rubric = Rubric::uniform("demo", "\n", ["clarity"]);
        let error = rubric.validate().expect_err("blank description");
        assert!(error.to_string().contains("description"), "{error}");
    }

    #[test]
    fn validate_rejects_non_positive_or_non_finite_scale() {
        for scale in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let rubric = Rubric::uniform("demo", "A demo", ["clarity"]).with_scale(scale);
            let error = rubric.validate().expect_err("bad scale");
            assert!(error.to_string().contains("scale"), "{error}");
        }
    }

    #[test]
    fn validate_rejects_empty_dimensions() {
        let error = Rubric::new("demo", "A demo")
            .validate()
            .expect_err("no dimensions");
        assert!(error.to_string().contains("dimension"), "{error}");
    }

    #[test]
    fn validate_rejects_blank_and_duplicate_dimension_names() {
        let blank =
            Rubric::new("demo", "A demo").with_dimension(RubricDimension::new(" ", "blank"));
        let error = blank.validate().expect_err("blank dimension");
        assert!(error.to_string().contains("dimension name"), "{error}");

        let duplicate = Rubric::new("demo", "A demo")
            .with_dimension(RubricDimension::new("clarity", "one"))
            .with_dimension(RubricDimension::new("clarity", "two"));
        let error = duplicate.validate().expect_err("duplicate");
        assert!(error.to_string().contains("duplicate"), "{error}");
    }

    #[test]
    fn validate_rejects_non_positive_or_non_finite_weight() {
        for weight in [0.0, -2.0, f64::NAN, f64::NEG_INFINITY] {
            let rubric = Rubric::new("demo", "A demo")
                .with_dimension(RubricDimension::new("clarity", "clear").with_weight(weight));
            let error = rubric.validate().expect_err("bad weight");
            assert!(error.to_string().contains("weight"), "{error}");
        }
    }

    #[test]
    fn dimension_lookup_is_exact() {
        let rubric = rubric();
        assert_eq!(
            rubric.dimension("depth").map(|found| found.weight),
            Some(3.0)
        );
        assert!(rubric.dimension("Depth").is_none());
        assert!(rubric.dimension("").is_none());
    }

    #[test]
    fn normalize_weights_dimensions_by_their_share() {
        let rubric = rubric();
        let full = rubric.normalize(&scores(&[("clarity", 1.0), ("depth", 1.0)]));
        assert!((full - 1.0).abs() < 1e-9);

        // Only the light dimension scores: one quarter of the total weight.
        let light = rubric.normalize(&scores(&[("clarity", 1.0)]));
        assert!((light - 0.25).abs() < 1e-9);

        // Only the heavy dimension scores: three quarters of the total weight.
        let heavy = rubric.normalize(&scores(&[("depth", 1.0)]));
        assert!((heavy - 0.75).abs() < 1e-9);
    }

    #[test]
    fn normalize_clamps_and_ignores_missing_or_unknown() {
        let rubric = rubric();
        let normalized = rubric.normalize(&scores(&[
            ("clarity", 5.0),
            ("depth", -3.0),
            ("unknown", 1.0),
        ]));
        // clarity clamps to 1.0 (weight 1), depth clamps to 0.0 (weight 3).
        assert!((normalized - 0.25).abs() < 1e-9);
        assert!((rubric.normalize(&BTreeMap::new()) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn normalize_uses_the_rubric_scale() {
        let rubric = Rubric::uniform("demo", "A demo", ["clarity"]).with_scale(10.0);
        assert!((rubric.normalize(&scores(&[("clarity", 8.0)])) - 0.8).abs() < 1e-9);
        assert!((rubric.normalize(&scores(&[("clarity", 20.0)])) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn normalize_handles_empty_and_zero_weight_rubrics() {
        assert!((Rubric::default().normalize(&scores(&[("clarity", 1.0)])) - 0.0).abs() < 1e-9);

        let mut rubric = Rubric::uniform("demo", "A demo", ["clarity"]);
        rubric.dimensions[0].weight = 0.0;
        assert!((rubric.normalize(&scores(&[("clarity", 1.0)])) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn is_passing_boundary_is_exact() {
        let rubric = rubric();
        assert!(rubric.is_passing(DEFAULT_PASS_THRESHOLD));
        assert!(rubric.is_passing(0.6));
        assert!(!rubric.is_passing(0.599_999_9));
        assert!(!rubric.is_passing(0.0));
    }

    #[test]
    fn serde_round_trip_keeps_every_field() {
        let rubric = rubric().with_scale(10.0);
        let decoded: Rubric =
            serde_json::from_value(serde_json::to_value(&rubric).expect("serializes to JSON"))
                .expect("deserializes from JSON");
        assert_eq!(decoded, rubric);

        let decoded_dimension: RubricDimension = serde_json::from_value(
            serde_json::to_value(&rubric.dimensions[0]).expect("serializes"),
        )
        .expect("deserializes");
        assert_eq!(decoded_dimension, rubric.dimensions[0]);
    }

    #[test]
    fn serde_defaults_missing_fields() {
        let decoded: Rubric =
            serde_json::from_value(serde_json::json!({"name": "demo"})).expect("deserializes");
        assert_eq!(decoded.scale, 1.0);
        assert_eq!(decoded.dimensions, Vec::new());

        let dimension: RubricDimension =
            serde_json::from_value(serde_json::json!({"name": "clarity"})).expect("deserializes");
        assert_eq!(dimension.weight, 1.0);
        assert_eq!(dimension.guidance, "");
    }
}
