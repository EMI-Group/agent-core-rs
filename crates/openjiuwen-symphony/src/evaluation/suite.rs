//! The evaluation suite: metric aggregation and fail-soft report orchestration.
//!
//! Ports `EvaluationSuite` from Python `openjiuwen.symphony.evaluation.suite`: the metric
//! list, `cache_signature()`, the fail-soft `evaluate`, the per-metric `aggregate`, and
//! the static/trace split (`evaluate_static`, `evaluate_case`).
//!
//! Aggregation is per metric: samples are grouped by metric name (sorted, so the result is
//! deterministic), each group is reduced with [`aggregate_metric`], and the
//! [`QualityResult`](crate::evaluation::QualityResult) reports the resulting confidence
//! tier instead of a composite score (`score` stays `None`, `aggregation` stays
//! `"per_metric"`).
use std::collections::BTreeMap;
use std::sync::Arc;

use serde_json::Value;

use crate::error::Result;
use crate::flow::model::sha256_hex;
use crate::llm::CompletionModel;

use super::metrics::{CompletenessMetric, LlmJudgedMetric, SchemaValidityMetric};
use super::trace::{EvaluationCase, EvaluationTrace, EvaluationWindow};
use super::{
    EvaluationInput, EvaluationReport, Metric, MetricScope, MetricScore, MetricStatus,
    QualityConfidence, QualityResult, PER_METRIC_AGGREGATION, SYMPHONY_EVALUATION_PROTOCOL,
};

/// Aggregates per-case samples of one metric, using Python's status precedence.
///
/// - A single sample is returned unchanged (Python merges only the diagnostic counts,
///   which this port carries in `detail`).
/// - Several samples: the score is the mean over samples that carry a score, and the
///   status is `Pass` only when every scored sample passed, else `Fail`.
/// - Without any scored sample: `Error` when a sample errored, else `Observed` when a
///   raw observation was recorded, else `NotApplicable`.
/// - No samples at all: `NotApplicable`.
#[must_use]
#[allow(clippy::cast_precision_loss)] // sample counts are far below 2^53
pub fn aggregate_metric(metric: &str, samples: &[MetricScore]) -> MetricScore {
    if samples.is_empty() {
        return MetricScore::not_applicable(metric, "no samples were supplied");
    }
    if let [sample] = samples {
        let mut single = sample.clone();
        if single.metric != metric {
            metric.clone_into(&mut single.metric);
        }
        return single;
    }
    let mut pass = 0_usize;
    let mut fail = 0_usize;
    let mut not_applicable = 0_usize;
    let mut error = 0_usize;
    let mut observed = 0_usize;
    for sample in samples {
        match sample.status {
            MetricStatus::Pass => pass += 1,
            MetricStatus::Fail => fail += 1,
            MetricStatus::NotApplicable => not_applicable += 1,
            MetricStatus::Error => error += 1,
            MetricStatus::Observed => observed += 1,
        }
    }
    let counts = format!(
        "{} samples ({pass} pass, {fail} fail, {not_applicable} not_applicable, \
         {error} error, {observed} observed)",
        samples.len()
    );
    let scored: Vec<&MetricScore> = samples
        .iter()
        .filter(|sample| sample.effective_score().is_some())
        .collect();
    if !scored.is_empty() {
        let total: f64 = scored
            .iter()
            .filter_map(|sample| sample.effective_score())
            .sum();
        let score = total / scored.len() as f64;
        let all_passed = scored
            .iter()
            .all(|sample| sample.status == MetricStatus::Pass);
        return MetricScore::scored(metric, score, all_passed)
            .with_detail(format!("aggregated across the supplied window: {counts}"));
    }
    if error > 0 {
        return MetricScore::error(
            metric,
            format!("at least one metric evaluation failed in the aggregation window: {counts}"),
        );
    }
    if observed > 0 {
        return MetricScore::observed(
            metric,
            format!(
                "raw observations were aggregated without a target or admission threshold: \
                 {counts}"
            ),
        );
    }
    MetricScore::not_applicable(
        metric,
        format!("no applicable metric result was available in the aggregation window: {counts}"),
    )
}

/// An ordered set of metrics evaluated fail-soft against one [`EvaluationInput`].
pub struct EvaluationSuite {
    /// The metrics, in evaluation (and report) order.
    metrics: Vec<Arc<dyn Metric>>,
}

impl EvaluationSuite {
    /// Builds a suite from the given metrics, evaluated in order.
    #[must_use]
    pub fn new(metrics: Vec<Arc<dyn Metric>>) -> Self {
        Self { metrics }
    }

    /// Builds the offline default suite: [`CompletenessMetric`] then
    /// [`SchemaValidityMetric`].
    #[must_use]
    pub fn default_suite() -> Self {
        Self::new(vec![
            Arc::new(CompletenessMetric),
            Arc::new(SchemaValidityMetric),
        ])
    }

    /// Appends the model-free [`LlmJudgedMetric`].
    ///
    /// Every evaluation then scores the card with the judge's deterministic fallback
    /// rubric (description, described input, described output), so the report has one
    /// extra score and — unlike a deferred stub — no error. Use
    /// [`with_llm_judge_model`](crate::evaluation::EvaluationSuite::with_llm_judge_model)
    /// to judge with a model instead.
    #[must_use]
    pub fn with_llm_judge(mut self) -> Self {
        self.metrics.push(Arc::new(LlmJudgedMetric::new()));
        self
    }

    /// Appends an [`LlmJudgedMetric`] that judges with `model`.
    ///
    /// The metric keeps the `"llm_judged"` name and the default rubric; a model verdict
    /// passes only on `score == 1`, one unusable reply is repaired, and a second failure
    /// falls back to the deterministic rubric.
    #[must_use]
    pub fn with_llm_judge_model(mut self, model: Arc<dyn CompletionModel>) -> Self {
        self.metrics
            .push(Arc::new(LlmJudgedMetric::new().with_model(model)));
        self
    }

    /// The configured metrics, in evaluation order.
    #[must_use]
    pub fn metrics(&self) -> &[Arc<dyn Metric>] {
        &self.metrics
    }

    /// The stable cache key for this suite's metric configuration.
    ///
    /// The digest is the SHA-256 of the canonical, sorted-key JSON document built from
    /// [`SYMPHONY_EVALUATION_PROTOCOL`] and each metric's
    /// `"{name}:{scope}:{cache_key|builtin:name}"` entry, joined in registration order.
    /// Two suites with the same metrics produce the same key on every run, and any
    /// metric change produces a different one.
    #[must_use]
    pub fn cache_signature(&self) -> String {
        let metrics: Vec<Value> = self
            .metrics
            .iter()
            .map(|metric| {
                let key = metric
                    .cache_key()
                    .unwrap_or_else(|| format!("builtin:{}", metric.name()));
                Value::String(format!(
                    "{}:{}:{}",
                    metric.name(),
                    metric.scope().as_str(),
                    key
                ))
            })
            .collect();
        let mut document: BTreeMap<&str, Value> = BTreeMap::new();
        document.insert(
            "protocol",
            Value::String(SYMPHONY_EVALUATION_PROTOCOL.to_string()),
        );
        document.insert("metrics", Value::Array(metrics));
        match serde_json::to_string(&document) {
            Ok(json) => sha256_hex(&json),
            // Strings only: serialization cannot fail, so this branch is unreachable.
            Err(_) => sha256_hex(SYMPHONY_EVALUATION_PROTOCOL),
        }
    }

    /// Evaluates `input` with every metric, collecting scores and failures.
    ///
    /// A metric error is recorded in [`EvaluationReport::errors`] as
    /// `"<metric>: <message>"` and does not stop the run. `overall` is the mean of the
    /// scored samples (trace metrics already aggregate their cases), or `0.0` when no
    /// metric produced a score.
    pub async fn evaluate(&self, input: &EvaluationInput) -> EvaluationReport {
        let mut scores = Vec::with_capacity(self.metrics.len());
        let mut errors = Vec::new();
        for metric in &self.metrics {
            match metric.evaluate(input).await {
                Ok(score) => scores.push(score),
                Err(error) => errors.push(format!("{}: {}", metric.name(), error.message())),
            }
        }
        EvaluationReport {
            capability_id: input.capability.capability_id.clone(),
            overall: mean_score(&scores),
            scores,
            errors,
        }
    }

    /// Evaluates `input` and aggregates the samples into a [`QualityResult`].
    ///
    /// A metric failure becomes a `status = error` sample (Python materialises evaluator
    /// exceptions the same way) instead of an entry in a separate error list, so every
    /// configured metric appears in [`QualityResult::per_metric`]. The sample count is
    /// the number of cases selected by [`EvaluationInput::selected_cases`], which also
    /// drives [`QualityConfidence`].
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::Error::Validation`] when the input's window is invalid.
    pub async fn evaluate_quality(&self, input: &EvaluationInput) -> Result<QualityResult> {
        let mut samples = Vec::with_capacity(self.metrics.len());
        for metric in &self.metrics {
            match metric.evaluate(input).await {
                Ok(score) => samples.push(score),
                Err(error) => samples.push(MetricScore::error(
                    metric.name(),
                    format!("evaluator failed: {}", error.message()),
                )),
            }
        }
        let sample_count = input.sample_count()?;
        self.aggregate(&samples, input.window.as_ref(), Some(sample_count))
    }

    /// Evaluates only the registered static metrics, in suite order.
    ///
    /// A failing metric yields a `status = error` sample, mirroring Python's
    /// `evaluate_static`, which converts evaluator exceptions into error results.
    pub async fn evaluate_static(&self, input: &EvaluationInput) -> Vec<MetricScore> {
        let mut scores = Vec::new();
        for metric in &self.metrics {
            if metric.scope() != MetricScope::Static {
                continue;
            }
            match metric.evaluate(input).await {
                Ok(score) => scores.push(score),
                Err(error) => scores.push(MetricScore::error(
                    metric.name(),
                    format!("evaluator failed: {}", error.message()),
                )),
            }
        }
        scores
    }

    /// Evaluates only the registered trace metrics for one supplied case.
    ///
    /// The case is wrapped in its own single-case trace, so a metric that aggregates
    /// internally returns exactly this case's verdict. The input's own window is
    /// ignored: the caller already chose the case.
    pub async fn evaluate_case(
        &self,
        input: &EvaluationInput,
        case: &EvaluationCase,
    ) -> Vec<MetricScore> {
        let scoped = EvaluationInput {
            capability: input.capability.clone(),
            target: input.target.clone(),
            trace: Some(EvaluationTrace::from_cases(vec![case.clone()])),
            window: None,
        };
        let mut scores = Vec::new();
        for metric in &self.metrics {
            if metric.scope() != MetricScope::Trace {
                continue;
            }
            match metric.evaluate(&scoped).await {
                Ok(score) => scores.push(score),
                Err(error) => scores.push(MetricScore::error(
                    metric.name(),
                    format!("evaluator failed: {}", error.message()),
                )),
            }
        }
        scores
    }

    /// Aggregates already-collected samples into a [`QualityResult`].
    ///
    /// Samples are grouped by [`MetricScore::metric`] in sorted (`BTreeMap`) order and
    /// each group is reduced with [`aggregate_metric`]. `sample_count` defaults to the
    /// largest group size, matching Python's inferred count.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::Error::Validation`] when `window` is invalid.
    pub fn aggregate(
        &self,
        samples: &[MetricScore],
        window: Option<&EvaluationWindow>,
        sample_count: Option<usize>,
    ) -> Result<QualityResult> {
        if let Some(window) = window {
            window.validate()?;
        }
        let mut grouped: BTreeMap<String, Vec<MetricScore>> = BTreeMap::new();
        for sample in samples {
            grouped
                .entry(sample.metric.clone())
                .or_default()
                .push(sample.clone());
        }
        let inferred = grouped.values().map(Vec::len).max().unwrap_or(0);
        let count = sample_count.unwrap_or(inferred);
        let per_metric = grouped
            .iter()
            .map(|(metric, group)| (metric.clone(), aggregate_metric(metric, group)))
            .collect();
        Ok(QualityResult {
            confidence: QualityConfidence::from_sample_count(count),
            score: None,
            aggregation: PER_METRIC_AGGREGATION.to_string(),
            sample_count: count,
            window: window.cloned(),
            per_metric,
        })
    }
}

/// The mean of the scored samples; `0.0` when no sample carries a score.
#[allow(clippy::cast_precision_loss)] // metric counts are far below 2^53
fn mean_score(scores: &[MetricScore]) -> f64 {
    let mut total = 0.0;
    let mut count = 0_usize;
    for score in scores {
        if let Some(value) = score.effective_score() {
            total += value;
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        total / count as f64
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::assert_is_empty
)]
mod tests {
    use super::*;
    use crate::evaluation::{CapabilityCall, SuccessRateMetric};
    use crate::model::{CapabilityDescriptor, CapabilityIo};

    /// A card that satisfies every static check.
    fn complete_capability() -> CapabilityDescriptor {
        let card = CapabilityDescriptor::new("tool", "sum", "Sum")
            .with_description("Adds two numbers.")
            .with_tags(["math", "arithmetic"])
            .with_inputs(vec![
                CapabilityIo::new("a", "number")
                    .with_description("first addend")
                    .with_required(true),
                CapabilityIo::new("b", "number")
                    .with_description("second addend")
                    .with_required(true),
            ])
            .with_outputs(vec![
                CapabilityIo::new("total", "number").with_description("the sum")
            ]);
        let mut card = card;
        card.content_hash = card.compute_content_hash();
        card
    }

    /// The same card, stripped of its vocabulary and with duplicate input names.
    fn broken_capability() -> CapabilityDescriptor {
        CapabilityDescriptor::new("tool", "broken", "Broken")
            .with_description("   ")
            .with_inputs(vec![
                CapabilityIo::new("dup", "number"),
                CapabilityIo::new("dup", "number"),
                CapabilityIo::new("  ", "string"),
            ])
            .with_outputs(vec![CapabilityIo::new("", "string")])
    }

    fn input(capability: CapabilityDescriptor) -> EvaluationInput {
        EvaluationInput::new(capability)
    }

    #[tokio::test]
    async fn default_suite_scores_a_complete_capability_above_threshold() {
        let report = EvaluationSuite::default_suite()
            .evaluate(&input(complete_capability()))
            .await;
        assert_eq!(report.capability_id, "sum");
        assert!(report.errors.is_empty());
        let names: Vec<&str> = report.scores.iter().map(|s| s.metric.as_str()).collect();
        assert_eq!(names, vec!["completeness", "schema_validity"]);
        for score in &report.scores {
            assert_eq!(
                score.score,
                Some(1.0),
                "{}: {:?}",
                score.metric,
                score.detail
            );
            assert_eq!(score.status, MetricStatus::Pass);
            assert!(score.passed);
            assert!(score.detail.is_none());
        }
        assert_eq!(report.overall, 1.0);
    }

    #[tokio::test]
    async fn blank_description_and_duplicate_names_fail_the_static_metrics() {
        let report = EvaluationSuite::default_suite()
            .evaluate(&input(broken_capability()))
            .await;
        assert_eq!(report.scores[0].status, MetricStatus::Fail);
        assert_eq!(report.scores[0].score, Some(0.0));
        assert_eq!(report.scores[1].score, Some(1.0 / 3.0));
        assert_eq!(report.overall, f64::midpoint(0.0, 1.0 / 3.0));
        assert!(report.errors.is_empty());
    }

    #[tokio::test]
    async fn overall_is_the_mean_of_the_successful_scores() {
        let report = EvaluationSuite::default_suite()
            .evaluate(&input(broken_capability().with_tags(["tools"])))
            .await;
        assert_eq!(report.scores.len(), 2);
        assert_eq!(report.scores[0].score, Some(0.25));
        assert_eq!(report.scores[1].score, Some(1.0 / 3.0));
        assert_eq!(report.overall, f64::midpoint(0.25, 1.0 / 3.0));
    }

    #[tokio::test]
    async fn an_empty_suite_reports_a_zero_overall_and_no_scores() {
        let report = EvaluationSuite::new(Vec::new())
            .evaluate(&input(complete_capability()))
            .await;
        assert!(report.scores.is_empty());
        assert!(report.errors.is_empty());
        assert_eq!(report.overall, 0.0);
        assert_eq!(report.capability_id, "sum");
    }

    #[tokio::test]
    async fn evaluate_static_skips_trace_metrics_and_reports_errors_as_samples() {
        let suite = EvaluationSuite::new(vec![
            Arc::new(CompletenessMetric),
            Arc::new(SuccessRateMetric),
        ]);
        let scores = suite.evaluate_static(&input(complete_capability())).await;
        assert_eq!(scores.len(), 1);
        assert_eq!(scores[0].metric, "completeness");
        assert_eq!(scores[0].status, MetricStatus::Pass);
    }

    #[tokio::test]
    async fn evaluate_case_runs_only_the_trace_metrics_on_the_case() {
        let case = EvaluationCase::new("case-1", "q")
            .with_calls(vec![CapabilityCall::new("sum").with_success(true)]);
        let suite = EvaluationSuite::new(vec![
            Arc::new(CompletenessMetric),
            Arc::new(SuccessRateMetric),
        ]);
        let scores = suite
            .evaluate_case(&input(complete_capability()), &case)
            .await;
        assert_eq!(scores.len(), 1);
        assert_eq!(scores[0].metric, "success_rate");
        assert_eq!(scores[0].score, Some(1.0));
        assert_eq!(scores[0].status, MetricStatus::Pass);
    }

    #[test]
    fn aggregate_metric_applies_python_status_precedence() {
        let pass = MetricScore::scored("success_rate", 1.0, true);
        let fail = MetricScore::scored("success_rate", 0.0, false);
        let mixed = aggregate_metric("success_rate", &[pass.clone(), fail.clone()]);
        assert_eq!(mixed.score, Some(0.5));
        assert_eq!(mixed.status, MetricStatus::Fail);
        assert!(!mixed.passed);
        let detail = mixed.detail.as_deref().unwrap();
        assert!(detail.contains("2 samples"), "{detail}");

        let single = aggregate_metric("success_rate", &[pass]);
        assert_eq!(single.score, Some(1.0));
        assert_eq!(single.status, MetricStatus::Pass);
        assert_eq!(single.detail, None);

        let skipped = MetricScore::not_applicable("latency", "no case");
        let error = MetricScore::error("latency", "boom");
        let observed = MetricScore::observed("latency", "raw");
        assert_eq!(
            aggregate_metric("latency", &[skipped, error]).status,
            MetricStatus::Error
        );
        assert_eq!(
            aggregate_metric("latency", &[observed]).status,
            MetricStatus::Observed
        );
        assert_eq!(
            aggregate_metric("latency", &[]).status,
            MetricStatus::NotApplicable
        );
    }

    #[test]
    fn aggregate_derives_confidence_from_the_sample_count() {
        let suite = EvaluationSuite::default_suite();
        let empty = suite.aggregate(&[], None, Some(0)).unwrap();
        assert_eq!(empty.confidence, QualityConfidence::None);
        assert_eq!(empty.aggregation, PER_METRIC_AGGREGATION);
        assert_eq!(empty.score, None);
        assert!(empty.per_metric.is_empty());
        for (count, expected) in [
            (1, QualityConfidence::Low),
            (2, QualityConfidence::Normal),
            (9, QualityConfidence::Normal),
            (10, QualityConfidence::High),
        ] {
            let result = suite.aggregate(&[], None, Some(count)).unwrap();
            assert_eq!(result.confidence, expected, "{count}");
            assert_eq!(result.sample_count, count);
        }
        let inferred = suite
            .aggregate(
                &[MetricScore::scored("completeness", 1.0, true)],
                None,
                None,
            )
            .unwrap();
        assert_eq!(inferred.sample_count, 1);
        assert_eq!(inferred.confidence, QualityConfidence::Low);
    }

    #[tokio::test]
    async fn evaluate_quality_aggregates_per_metric() {
        let suite = EvaluationSuite::default_suite();
        let quality = suite
            .evaluate_quality(&input(complete_capability()))
            .await
            .unwrap();
        assert_eq!(quality.confidence, QualityConfidence::None);
        assert_eq!(quality.score, None);
        assert_eq!(quality.aggregation, "per_metric");
        assert_eq!(quality.per_metric.len(), 2);
        let completeness = quality.per_metric.get("completeness").unwrap();
        assert_eq!(completeness.score, Some(1.0));
        let schema = quality.per_metric.get("schema_validity").unwrap();
        assert_eq!(schema.status, MetricStatus::Pass);
    }

    #[test]
    fn cache_signature_is_stable_and_content_sensitive() {
        let first = EvaluationSuite::default_suite().cache_signature();
        let second = EvaluationSuite::default_suite().cache_signature();
        assert_eq!(first, second);
        assert_eq!(first.len(), 64);
        assert!(first.chars().all(|c| c.is_ascii_hexdigit()));
        let extended = EvaluationSuite::default_suite()
            .with_llm_judge()
            .cache_signature();
        assert_ne!(first, extended);
        let reordered = EvaluationSuite::new(vec![
            Arc::new(SchemaValidityMetric),
            Arc::new(CompletenessMetric),
        ])
        .cache_signature();
        assert_ne!(first, reordered);
    }

    #[test]
    fn reports_round_trip_through_json() {
        let report = EvaluationReport {
            capability_id: "tool:sum".to_string(),
            scores: vec![
                MetricScore::scored("completeness", 0.75, true).with_detail("missing: tags"),
                MetricScore::observed("latency", "avg_ms=1.000"),
            ],
            overall: 0.75,
            errors: vec!["llm_judged: model unavailable".to_string()],
        };
        let encoded = serde_json::to_value(&report).unwrap();
        assert_eq!(encoded["capability_id"], "tool:sum");
        assert_eq!(encoded["scores"][0]["metric"], "completeness");
        assert_eq!(encoded["scores"][0]["status"], "pass");
        assert_eq!(encoded["scores"][1]["status"], "observed");
        assert_eq!(encoded["scores"][1]["score"], Value::Null);
        let decoded: EvaluationReport = serde_json::from_value(encoded).unwrap();
        assert_eq!(decoded, report);
    }
}
