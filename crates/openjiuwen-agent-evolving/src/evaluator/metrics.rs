//! Deterministic metrics and the [`MetricEvaluator`] that aggregates them.
//!
//! Ports Python `openjiuwen.agent_evolving.evaluator.metrics`:
//!
//! - [`Metric`] and [`MetricValue`] — the scoring seam and its result shape.
//! - [`ExactMatchMetric`] — offline string/JSON equality, optionally normalized.
//! - [`LlmAsJudgeMetric`] — API-shaped stub for the deferred model judge.
//! - [`MetricEvaluator`] — runs every metric and aggregates the ordered values
//!   with [`Aggregate::Mean`] or [`Aggregate::First`], exposing the per-metric
//!   breakdown.

use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::Evaluator;
use crate::dataset::{Case, EvaluatedCase};

/// The value produced by one [`Metric`].
///
/// Python metrics return either a plain float or a mapping of named scores; the
/// untagged representation keeps that wire shape intact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MetricValue {
    /// One score for the whole metric.
    Scalar(f64),
    /// Named sub-scores, iterated in their natural (sorted) order.
    Named(BTreeMap<String, f64>),
}

/// A deterministic scoring rule over a prediction and the case label.
///
/// Python: `openjiuwen.agent_evolving.evaluator.metrics.base.Metric`.
pub trait Metric: Send + Sync {
    /// The metric name used as the `per_metric` key.
    fn name(&self) -> &str;

    /// Whether a larger value is better; defaults to `true` as in Python.
    fn higher_is_better(&self) -> bool {
        true
    }

    /// Scores `prediction` against `label`.
    ///
    /// # Errors
    ///
    /// Implementations may return an error when scoring cannot proceed (for
    /// example the deferred model-judge metric).
    fn compute(&self, prediction: &Value, label: &BTreeMap<String, Value>) -> Result<MetricValue>;
}

/// Exact-match scoring with optional whitespace/case normalization.
///
/// Python: `openjiuwen.agent_evolving.evaluator.metrics.exact_match.ExactMatchMetric`.
///
/// The label answer is projected from the first available of: the `answer` key,
/// a single-key label map, or the full label object. Predictions are projected
/// the same way (`answer` key, single-entry object, or the raw value), so agent
/// outputs and dataset files need no adapter.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ExactMatchMetric {
    /// Whether string comparisons trim, lowercase, and collapse whitespace.
    normalize: bool,
}

impl Default for ExactMatchMetric {
    fn default() -> Self {
        Self::new()
    }
}

impl ExactMatchMetric {
    /// Creates a normalizing metric.
    #[must_use]
    pub const fn new() -> Self {
        Self { normalize: true }
    }

    /// Disables normalization, comparing strings exactly.
    #[must_use]
    pub const fn raw(mut self) -> Self {
        self.normalize = false;
        self
    }

    /// Whether string comparison is normalized.
    #[must_use]
    pub const fn normalize(&self) -> bool {
        self.normalize
    }
}

/// Case-sensitive/normalized text comparison.
///
/// Normalization trims, lowercases, and collapses internal whitespace runs to a
/// single space. Unicode case folding is out of scope; only `str::to_lowercase`
/// is applied.
fn normalize_text(text: &str) -> String {
    text.trim()
        .to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Projects the expected answer from a case label.
fn project_expected(label: &BTreeMap<String, Value>) -> Value {
    if let Some(answer) = label.get("answer") {
        answer.clone()
    } else if label.len() == 1 {
        label.values().next().cloned().unwrap_or(Value::Null)
    } else {
        Value::Object(label.clone().into_iter().collect())
    }
}

/// Projects the answer from a prediction.
fn project_prediction(prediction: &Value) -> Value {
    match prediction {
        Value::Object(map) => {
            if let Some(answer) = map.get("answer") {
                answer.clone()
            } else if map.len() == 1 {
                map.values().next().cloned().unwrap_or(Value::Null)
            } else {
                prediction.clone()
            }
        }
        _ => prediction.clone(),
    }
}

impl Metric for ExactMatchMetric {
    fn name(&self) -> &'static str {
        "exact_match"
    }

    fn compute(&self, prediction: &Value, label: &BTreeMap<String, Value>) -> Result<MetricValue> {
        let expected = project_expected(label);
        let predicted = project_prediction(prediction);
        let matched = match (&expected, &predicted) {
            (Value::String(expected), Value::String(predicted)) => {
                if self.normalize {
                    normalize_text(expected) == normalize_text(predicted)
                } else {
                    expected == predicted
                }
            }
            _ => expected == predicted,
        };
        Ok(MetricValue::Scalar(if matched { 1.0 } else { 0.0 }))
    }
}

/// The deferred LLM-as-judge metric.
///
/// Python: `openjiuwen.agent_evolving.evaluator.metrics.llm_as_judge.LlmAsJudgeMetric`.
/// The Python implementation calls a live model with a judge prompt, which is
/// non-deterministic and needs provider credentials; this crate keeps the type
/// and returns [`Error::Unsupported`] instead.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LlmAsJudgeMetric;

impl LlmAsJudgeMetric {
    /// Creates the stub metric.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl Metric for LlmAsJudgeMetric {
    fn name(&self) -> &'static str {
        "llm_as_judge"
    }

    fn compute(
        &self,
        _prediction: &Value,
        _label: &BTreeMap<String, Value>,
    ) -> Result<MetricValue> {
        Err(Error::unsupported(
            "LlmAsJudgeMetric is deferred: it requires a live model and is non-deterministic; see docs/PARITY.md",
        ))
    }
}

/// The deferred default (model-judge) evaluator.
///
/// Python: `openjiuwen.agent_evolving.evaluator.evaluator.DefaultEvaluator`.
/// Python scores cases by calling a live model with a judge prompt, which needs
/// provider credentials and is non-deterministic; this crate keeps the type and
/// returns [`Error::Unsupported`] instead.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DefaultEvaluator {
    metric: String,
}

impl DefaultEvaluator {
    /// Creates the stub evaluator for the named metric.
    #[must_use]
    pub fn new(metric: impl Into<String>) -> Self {
        Self {
            metric: metric.into(),
        }
    }

    /// The metric name the Python evaluator would have scored with.
    #[must_use]
    pub fn metric(&self) -> &str {
        &self.metric
    }
}

#[async_trait]
impl Evaluator for DefaultEvaluator {
    fn name(&self) -> &'static str {
        "default"
    }

    async fn evaluate(&self, case: &Case, prediction: &Value) -> Result<EvaluatedCase> {
        let _ = (case, prediction);
        Err(Error::unsupported(
            "DefaultEvaluator is deferred: it requires a live model judge and is non-deterministic; see docs/PARITY.md",
        ))
    }
}

/// How [`MetricEvaluator`] turns the ordered metric values into one score.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Aggregate {
    /// Average every metric value.
    #[default]
    Mean,
    /// Use only the first metric value.
    First,
}

/// Evaluator that scores with a list of [`Metric`]s and aggregates the results.
///
/// Python: `MetricEvaluator`. Every metric value is recorded in `per_metric`
/// (scalars under the metric name, named values under their own keys) and, in
/// that same order, folded into the final score.
pub struct MetricEvaluator {
    metrics: Vec<Arc<dyn Metric>>,
    aggregate: Aggregate,
}

impl MetricEvaluator {
    /// Creates an evaluator over `metrics`, aggregating by [`Aggregate::Mean`].
    #[must_use]
    pub const fn new(metrics: Vec<Arc<dyn Metric>>) -> Self {
        Self {
            metrics,
            aggregate: Aggregate::Mean,
        }
    }

    /// Replaces the aggregation strategy.
    #[must_use]
    pub const fn with_aggregate(mut self, aggregate: Aggregate) -> Self {
        self.aggregate = aggregate;
        self
    }

    /// The configured metrics, in evaluation order.
    #[must_use]
    pub fn metrics(&self) -> &[Arc<dyn Metric>] {
        &self.metrics
    }

    /// The configured aggregation strategy.
    #[must_use]
    pub const fn aggregate(&self) -> Aggregate {
        self.aggregate
    }
}

/// Coerces a non-finite score to `0.0` so aggregation never propagates `NaN`.
fn finite_or_zero(value: f64) -> f64 {
    if value.is_finite() {
        value
    } else {
        0.0
    }
}

#[async_trait]
impl Evaluator for MetricEvaluator {
    fn name(&self) -> &'static str {
        "metric"
    }

    async fn evaluate(&self, case: &Case, prediction: &Value) -> Result<EvaluatedCase> {
        let mut per_metric: BTreeMap<String, f64> = BTreeMap::new();
        let mut values: Vec<f64> = Vec::new();

        for metric in &self.metrics {
            match metric.compute(prediction, &case.label)? {
                MetricValue::Scalar(value) => {
                    let value = finite_or_zero(value);
                    per_metric.insert(metric.name().to_owned(), value);
                    values.push(value);
                }
                MetricValue::Named(named) => {
                    for (name, value) in named {
                        let value = finite_or_zero(value);
                        per_metric.insert(name, value);
                        values.push(value);
                    }
                }
            }
        }

        let score = match self.aggregate {
            Aggregate::Mean => {
                if values.is_empty() {
                    0.0
                } else {
                    values.iter().sum::<f64>() / values.len() as f64
                }
            }
            Aggregate::First => values.first().copied().unwrap_or(0.0),
        };

        let mut evaluated = EvaluatedCase::new(
            case.clone(),
            None,
            score,
            format!("{} metric(s)", self.metrics.len()),
        )?;
        if !per_metric.is_empty() {
            evaluated.per_metric = Some(per_metric);
        }
        Ok(evaluated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map(key: &str, value: Value) -> BTreeMap<String, Value> {
        let mut map = BTreeMap::new();
        map.insert(key.to_owned(), value);
        map
    }

    fn case(label: BTreeMap<String, Value>) -> Case {
        let inputs = map("question", Value::String("q".to_owned()));
        Case::new(inputs, label).unwrap()
    }

    fn close(actual: f64, expected: f64) -> bool {
        (actual - expected).abs() < 1e-9
    }

    struct FixedMetric {
        name: &'static str,
        value: MetricValue,
    }

    impl Metric for FixedMetric {
        fn name(&self) -> &str {
            self.name
        }

        fn compute(
            &self,
            _prediction: &Value,
            _label: &BTreeMap<String, Value>,
        ) -> Result<MetricValue> {
            Ok(self.value.clone())
        }
    }

    fn named_scalar(name: &'static str, value: f64) -> Arc<dyn Metric> {
        Arc::new(FixedMetric {
            name,
            value: MetricValue::Scalar(value),
        })
    }

    fn scalar(value: f64) -> Arc<dyn Metric> {
        named_scalar("scalar", value)
    }

    #[test]
    fn constructor_defaults_to_normalizing() {
        assert!(ExactMatchMetric::new().normalize());
        assert!(ExactMatchMetric::default().normalize());
        assert!(!ExactMatchMetric::new().raw().normalize());
        assert_eq!(ExactMatchMetric::new().name(), "exact_match");
    }

    #[test]
    fn normalizes_whitespace_and_case() {
        let metric = ExactMatchMetric::new();
        let label = map("answer", Value::String("Hello   World".to_owned()));
        let value = metric
            .compute(&Value::String("  hello world ".to_owned()), &label)
            .unwrap();
        assert_eq!(value, MetricValue::Scalar(1.0));
    }

    #[test]
    fn raw_mode_compares_exactly() {
        let metric = ExactMatchMetric::new().raw();
        let label = map("answer", Value::String("Hello".to_owned()));
        assert_eq!(
            metric
                .compute(&Value::String("Hello".to_owned()), &label)
                .unwrap(),
            MetricValue::Scalar(1.0)
        );
        assert_eq!(
            metric
                .compute(&Value::String("hello".to_owned()), &label)
                .unwrap(),
            MetricValue::Scalar(0.0)
        );
    }

    #[test]
    fn compares_non_strings_with_json_equality() {
        let metric = ExactMatchMetric::new();
        let label = map("answer", serde_json::json!(4));
        assert_eq!(
            metric.compute(&serde_json::json!(4), &label).unwrap(),
            MetricValue::Scalar(1.0)
        );
        assert_eq!(
            metric.compute(&serde_json::json!("4"), &label).unwrap(),
            MetricValue::Scalar(0.0)
        );
    }

    #[test]
    fn projects_answer_key_single_key_and_full_object_labels() {
        let metric = ExactMatchMetric::new();

        let answer_label = map("answer", Value::String("hello".to_owned()));
        assert_eq!(
            metric
                .compute(&serde_json::json!({"answer": "hello"}), &answer_label)
                .unwrap(),
            MetricValue::Scalar(1.0)
        );

        let single_label = map("expected", Value::String("hello".to_owned()));
        assert_eq!(
            metric
                .compute(&serde_json::json!({"expected": "hello"}), &single_label)
                .unwrap(),
            MetricValue::Scalar(1.0)
        );
        assert_eq!(
            metric
                .compute(&Value::String("hello".to_owned()), &single_label)
                .unwrap(),
            MetricValue::Scalar(1.0)
        );

        let full_label = BTreeMap::from([
            ("a".to_owned(), serde_json::json!(1)),
            ("b".to_owned(), serde_json::json!(2)),
        ]);
        assert_eq!(
            metric
                .compute(&serde_json::json!({"a": 1, "b": 2}), &full_label)
                .unwrap(),
            MetricValue::Scalar(1.0)
        );
        assert_eq!(
            metric
                .compute(&serde_json::json!({"a": 1, "b": 3}), &full_label)
                .unwrap(),
            MetricValue::Scalar(0.0)
        );
    }

    #[test]
    fn stub_metric_returns_unsupported() {
        let error = LlmAsJudgeMetric::new()
            .compute(&Value::Null, &map("answer", Value::String("x".to_owned())))
            .unwrap_err();
        assert!(
            matches!(&error, Error::Unsupported(message) if message.contains("LlmAsJudgeMetric")),
            "unexpected error: {error}"
        );
        assert_eq!(LlmAsJudgeMetric::new().name(), "llm_as_judge");
        assert!(LlmAsJudgeMetric::new().higher_is_better());
    }

    #[tokio::test]
    async fn default_evaluator_is_an_unsupported_stub() {
        let evaluator = DefaultEvaluator::new("accuracy");
        assert_eq!(evaluator.name(), "default");
        assert_eq!(evaluator.metric(), "accuracy");
        assert_eq!(DefaultEvaluator::default().metric(), "");

        let error = evaluator
            .evaluate(
                &case(map("answer", Value::String("x".to_owned()))),
                &Value::Null,
            )
            .await
            .unwrap_err();
        assert!(
            matches!(&error, Error::Unsupported(message)
                if message.contains("DefaultEvaluator") && message.contains("docs/PARITY.md")),
            "unexpected error: {error}"
        );
    }

    #[tokio::test]
    async fn mean_aggregates_scalars_and_named_values_in_order() {
        let named = Arc::new(FixedMetric {
            name: "named",
            value: MetricValue::Named(BTreeMap::from([
                ("x".to_owned(), 0.6),
                ("y".to_owned(), 1.0),
            ])),
        });
        let evaluator = MetricEvaluator::new(vec![scalar(0.2), named]);
        assert_eq!(evaluator.name(), "metric");
        assert_eq!(evaluator.aggregate(), Aggregate::Mean);
        assert_eq!(evaluator.metrics().len(), 2);

        let evaluated = evaluator
            .evaluate(
                &case(map("answer", Value::String("a".to_owned()))),
                &Value::Null,
            )
            .await
            .unwrap();

        // Values in order: 0.2 (scalar), then named BTreeMap order x=0.6, y=1.0.
        assert!(close(evaluated.score, 0.6), "score: {}", evaluated.score);
        let per_metric = evaluated.per_metric.expect("per-metric breakdown");
        assert_eq!(per_metric.len(), 3);
        assert!(close(per_metric["scalar"], 0.2));
        assert!(close(per_metric["x"], 0.6));
        assert!(close(per_metric["y"], 1.0));
        assert_eq!(evaluated.reason, "2 metric(s)");
        assert!(evaluated.answer.is_none());
    }

    #[tokio::test]
    async fn first_aggregation_takes_the_leading_value() {
        let evaluator =
            MetricEvaluator::new(vec![scalar(0.25), scalar(0.75)]).with_aggregate(Aggregate::First);
        let evaluated = evaluator
            .evaluate(
                &case(map("answer", Value::String("a".to_owned()))),
                &Value::Null,
            )
            .await
            .unwrap();
        assert!(close(evaluated.score, 0.25));
        let per_metric = evaluated.per_metric.expect("per-metric breakdown");
        assert_eq!(per_metric.len(), 1, "both metrics share the `scalar` name");
        assert!(close(per_metric["scalar"], 0.75));
    }

    #[tokio::test]
    async fn non_finite_values_are_coerced_to_zero() {
        let evaluator = MetricEvaluator::new(vec![
            named_scalar("nan", f64::NAN),
            named_scalar("two", 2.0),
        ]);
        let evaluated = evaluator
            .evaluate(
                &case(map("answer", Value::String("a".to_owned()))),
                &Value::Null,
            )
            .await
            .unwrap();
        assert!(close(evaluated.score, 1.0), "score: {}", evaluated.score);
        let per_metric = evaluated.per_metric.expect("per-metric breakdown");
        assert!(close(per_metric["nan"], 0.0));
        assert!(close(per_metric["two"], 2.0));
    }

    #[tokio::test]
    async fn empty_metrics_yield_zero_and_no_breakdown() {
        let evaluator = MetricEvaluator::new(Vec::new());
        let evaluated = evaluator
            .evaluate(
                &case(map("answer", Value::String("a".to_owned()))),
                &Value::Null,
            )
            .await
            .unwrap();
        assert!(close(evaluated.score, 0.0));
        assert!(evaluated.per_metric.is_none());
        assert_eq!(evaluated.reason, "0 metric(s)");
    }

    #[test]
    fn aggregate_serde_round_trips() {
        assert_eq!(
            serde_json::to_value(Aggregate::Mean).unwrap(),
            serde_json::json!("mean")
        );
        assert_eq!(
            serde_json::to_value(Aggregate::First).unwrap(),
            serde_json::json!("first")
        );
        let restored: Aggregate = serde_json::from_value(serde_json::json!("first")).unwrap();
        assert_eq!(restored, Aggregate::First);
    }
}
