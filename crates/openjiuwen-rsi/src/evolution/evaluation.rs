//! Case-based evaluation of candidates.
//!
//! The protocol mirrors Python's `EvaluationReport` aggregation: a run produces
//! one [`CaseResult`] per [`Case`], and a score of `None` means "ungraded".
//! Ungraded results contribute to neither the average nor the weighted score,
//! so an evaluation that only ran half its cases never looks artificially good
//! (Python `test_evaluator.py` behavior).
//!
//! The actual grading is behind the [`Evaluator`] trait: the harness
//! self-optimisation loop implements it over a [`HarnessRunner`], tests
//! implement it with a scripted fake. No scoring formula lives in this module
//! beyond the aggregation.
//!
//! [`HarnessRunner`]: crate::harness_rsi::auto_harness::runner::HarnessRunner

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use openjiuwen_common::error::Result;

use super::candidate::Candidate;

/// One evaluation case.
///
/// `weight` defaults to `1.0`, matching Python's per-case weight. `input` and
/// `expectations` are opaque JSON: the [`Evaluator`] decides their meaning.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct Case {
    /// Stable case id, referenced by [`CaseResult::case_id`].
    pub id: String,
    /// Optional display name.
    pub name: Option<String>,
    /// Relative weight in the weighted score; `0.0` excludes the case.
    pub weight: f64,
    /// The evaluator-specific input document.
    pub input: Value,
    /// The evaluator-specific expectation document.
    pub expectations: Value,
    /// Caller-specific annotations; never interpreted here.
    pub metadata: std::collections::BTreeMap<String, Value>,
}

impl Default for Case {
    /// Returns an empty case with Python's unit weight.
    fn default() -> Self {
        Self {
            id: String::new(),
            name: None,
            weight: 1.0,
            input: Value::Null,
            expectations: Value::Null,
            metadata: std::collections::BTreeMap::new(),
        }
    }
}

/// The outcome of evaluating one [`Case`].
///
/// `score: None` marks the case as ungraded (the evaluator could not grade it
/// but did not fail); `error: Some(_)` marks a hard failure.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct CaseResult {
    /// The id of the [`Case`] this result belongs to.
    pub case_id: String,
    /// Whether the case passed its expectations.
    pub passed: bool,
    /// The numeric score; `None` means ungraded.
    pub score: Option<f64>,
    /// The case weight copied from the [`Case`].
    pub weight: f64,
    /// The failure message, when the case could not be graded at all.
    pub error: Option<String>,
    /// Evaluator-specific diagnostics.
    pub details: Value,
}

impl Default for CaseResult {
    /// Returns an empty ungraded result with unit weight.
    fn default() -> Self {
        Self {
            case_id: String::new(),
            passed: false,
            score: None,
            weight: 1.0,
            error: None,
            details: Value::Null,
        }
    }
}

/// The aggregated outcome of evaluating one candidate over a case set.
///
/// Build it with [`EvaluationReport::from_results`] so the counts and scores
/// stay consistent; the fields are public only so the report round-trips
/// through serde for checkpoints.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct EvaluationReport {
    /// One result per evaluated case, in evaluation order.
    pub results: Vec<CaseResult>,
    /// The number of results.
    pub total: usize,
    /// The number of results with `passed = true`.
    pub passed: usize,
    /// The number of graded results (those with a score).
    pub graded: usize,
    /// The mean of the graded scores, or `None` when nothing was graded.
    pub average_score: Option<f64>,
    /// The weight-normalised mean over graded results with positive weight.
    pub weighted_score: Option<f64>,
    /// The evaluation cost in US dollars, when the evaluator reports one.
    pub cost_usd: f64,
}

impl EvaluationReport {
    /// Aggregates per-case results into a report.
    ///
    /// `total` and `passed` count every result; `graded`, `average_score`, and
    /// `weighted_score` consider only results carrying a score, so ungraded
    /// cases are never folded into an average. The weighted score sums
    /// `score * weight` over graded results with `weight > 0.0` and divides by
    /// the sum of those weights. Both scores are `None` when nothing is graded.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn from_results(results: Vec<CaseResult>) -> Self {
        let total = results.len();
        let passed = results.iter().filter(|result| result.passed).count();
        let graded = results
            .iter()
            .filter(|result| result.score.is_some())
            .count();
        let average_score = if graded == 0 {
            None
        } else {
            let sum: f64 = results.iter().filter_map(|result| result.score).sum();
            Some(sum / graded as f64)
        };

        let mut weighted_sum = 0.0;
        let mut weight_total = 0.0;
        for result in &results {
            if result.weight <= 0.0 {
                continue;
            }
            if let Some(score) = result.score {
                weighted_sum += score * result.weight;
                weight_total += result.weight;
            }
        }
        let weighted_score = if weight_total > 0.0 {
            Some(weighted_sum / weight_total)
        } else {
            None
        };

        Self {
            results,
            total,
            passed,
            graded,
            average_score,
            weighted_score,
            cost_usd: 0.0,
        }
    }

    /// Returns the headline score: the weighted score when one exists, else the
    /// plain average, else `None` when nothing was graded.
    #[must_use]
    pub fn score(&self) -> Option<f64> {
        self.weighted_score.or(self.average_score)
    }

    /// Returns whether every result is graded and error-free.
    ///
    /// An empty report is valid: there is nothing that failed.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.results
            .iter()
            .all(|result| result.score.is_some() && result.error.is_none())
    }
}

/// Grades a candidate against a case set.
///
/// Implementations own the run (a harness session, a subprocess, a fake) and
/// must report per-case results rather than a single number, so ungraded cases
/// stay visible in the report.
#[async_trait]
pub trait Evaluator: Send + Sync {
    /// Returns the evaluator's human-readable name.
    fn name(&self) -> &str;

    /// Evaluates `candidate` over `cases`.
    ///
    /// # Errors
    ///
    /// Returns an error when the evaluation itself cannot run; per-case
    /// failures belong in [`CaseResult::error`] instead.
    async fn evaluate(&self, candidate: &Candidate, cases: &[Case]) -> Result<EvaluationReport>;
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn result(id: &str, score: Option<f64>, weight: f64, error: Option<&str>) -> CaseResult {
        CaseResult {
            case_id: id.to_owned(),
            passed: error.is_none(),
            score,
            weight,
            error: error.map(str::to_owned),
            details: Value::Null,
        }
    }

    fn approx(actual: Option<f64>, expected: f64) -> bool {
        actual.is_some_and(|value| (value - expected).abs() < 1e-9)
    }

    #[test]
    fn empty_report_is_valid_and_unscored() {
        let report = EvaluationReport::from_results(Vec::new());
        assert_eq!(report.total, 0);
        assert_eq!(report.passed, 0);
        assert_eq!(report.graded, 0);
        assert!(report.average_score.is_none());
        assert!(report.weighted_score.is_none());
        assert!(report.score().is_none());
        assert!(report.is_valid());
    }

    #[test]
    fn ungraded_results_never_enter_the_average() {
        let report = EvaluationReport::from_results(vec![
            result("a", Some(1.0), 1.0, None),
            result("b", None, 1.0, None),
            result("c", Some(0.0), 1.0, None),
        ]);
        assert_eq!(report.total, 3);
        assert_eq!(report.passed, 3);
        assert_eq!(report.graded, 2);
        assert!(approx(report.average_score, 0.5));
        assert!(approx(report.weighted_score, 0.5));
        assert!(approx(report.score(), 0.5));
    }

    #[test]
    fn weighted_score_uses_case_weights() {
        let report = EvaluationReport::from_results(vec![
            result("a", Some(1.0), 3.0, None),
            result("b", Some(0.0), 1.0, None),
        ]);
        assert!(approx(report.average_score, 0.5));
        assert!(approx(report.weighted_score, 0.75));
        assert!(approx(report.score(), 0.75));
    }

    #[test]
    fn zero_weight_results_are_excluded_from_the_weighted_score() {
        let report = EvaluationReport::from_results(vec![
            result("a", Some(1.0), 1.0, None),
            result("b", Some(0.0), 0.0, None),
        ]);
        assert!(approx(report.weighted_score, 1.0));
        assert!(approx(report.average_score, 0.5));
    }

    #[test]
    fn all_zero_weights_fall_back_to_the_average_score() {
        let report = EvaluationReport::from_results(vec![
            result("a", Some(1.0), 0.0, None),
            result("b", Some(0.0), 0.0, None),
        ]);
        assert!(report.weighted_score.is_none());
        assert!(approx(report.score(), 0.5));
    }

    #[test]
    fn errored_results_make_the_report_invalid() {
        let report = EvaluationReport::from_results(vec![
            result("a", Some(1.0), 1.0, None),
            result("b", None, 1.0, Some("grader crashed")),
        ]);
        assert!(!report.is_valid());
        assert!(approx(report.score(), 1.0));
    }

    #[test]
    fn ungraded_results_make_the_report_invalid() {
        let report = EvaluationReport::from_results(vec![result("a", None, 1.0, None)]);
        assert!(!report.is_valid());
    }

    #[test]
    fn case_weight_defaults_to_one() {
        let case: Case = serde_json::from_value(json!({"id": "c"})).expect("case parses");
        assert!((case.weight - 1.0).abs() < f64::EPSILON);
        assert!(case.name.is_none());
    }

    #[test]
    fn report_round_trips_through_json() {
        let report = EvaluationReport::from_results(vec![result("a", Some(0.25), 2.0, None)]);
        let encoded = serde_json::to_value(&report).expect("report serializes");
        let decoded: EvaluationReport =
            serde_json::from_value(encoded).expect("report deserializes");
        assert_eq!(decoded, report);
    }

    #[test]
    fn report_tolerates_unknown_fields() {
        let report: EvaluationReport = serde_json::from_value(json!({
            "graded": 1,
            "average_score": 0.5,
            "future_field": [1, 2, 3]
        }))
        .expect("report tolerates unknown fields");
        assert_eq!(report.graded, 1);
        assert!(approx(report.score(), 0.5));
    }
}
