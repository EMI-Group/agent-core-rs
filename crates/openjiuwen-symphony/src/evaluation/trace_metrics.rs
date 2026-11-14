//! The trace metrics: success rate, latency, accuracy, selection, and composition.
//!
//! Ports the five evidence-driven evaluators of Python
//! `openjiuwen.symphony.evaluation.evaluators`:
//!
//! | Rust metric | Python evaluator | Verdict |
//! | --- | --- | --- |
//! | [`SuccessRateMetric`] | `SuccessRateEvaluator` | scored from the recorded call outcomes |
//! | [`LatencyMetric`] | `LatencyEvaluator` | `observed`, never scored |
//! | [`AccuracyMetric`] | `AccuracyEvaluator` | exact-match short-circuit, then the judge |
//! | [`CapabilitySelectionMetric`] | `CapabilitySelectionEvaluator` | no-calls guard, then the judge |
//! | [`CompositionEffectivenessMetric`] | `CompositionEffectivenessEvaluator` | fewer-than-two-calls guard, then the judge |
//!
//! Every metric reads the cases selected by
//! [`EvaluationInput::selected_cases`](crate::evaluation::EvaluationInput::selected_cases) and
//! combines the per-case verdicts with
//! [`aggregate_metric`], so one case and many cases take
//! the same code path. The judge-backed metrics run with
//! [`FallbackMode::Deterministic`](crate::llm::FallbackMode), so an offline suite always
//! produces a verdict instead of an error.
use std::sync::Arc;

use async_trait::async_trait;

use crate::error::Result;
use crate::llm::judge::LlmJudgeMetric;
use crate::llm::{CompletionModel, FallbackMode};

use super::metrics::normalize_status;
use super::trace::{latency_statistics, CaseOutcome, EvaluationCase};
use super::{aggregate_metric, EvaluationInput, Metric, MetricScope, MetricScore};

/// The metric name of [`SuccessRateMetric`].
const SUCCESS_RATE: &str = "success_rate";
/// The metric name of [`LatencyMetric`].
const LATENCY: &str = "latency";
/// The metric name of [`AccuracyMetric`].
const ACCURACY: &str = "accuracy";
/// The metric name of [`CapabilitySelectionMetric`].
const CAPABILITY_SELECTION: &str = "capability_selection";
/// The metric name of [`CompositionEffectivenessMetric`].
const COMPOSITION_EFFECTIVENESS: &str = "composition_effectiveness";
/// The rubric of [`AccuracyMetric`].
///
/// Python carries a long Chinese three-valued rubric here; this port keeps its decision
/// boundary (factual error versus no usable answer) in English.
const ACCURACY_RUBRIC: &str = "Judge whether the answer is factually correct for the query. \
Return score 1 when it is accurate, 0 when it contains a factual, logical, or professional \
error. Treat missing confirmation as insufficient evidence rather than as an error.";
/// The rubric of [`CapabilitySelectionMetric`].
const CAPABILITY_SELECTION_RUBRIC: &str = "Judge whether the capabilities selected in the trace \
are relevant and sufficient for the query, without relying on execution outside the trace.";
/// The rubric of [`CompositionEffectivenessMetric`].
const COMPOSITION_RUBRIC: &str = "Judge whether the order, inputs, outputs, and hand-offs among \
the selected capabilities form an effective composition for the query.";

/// Reports whether the supplied trace cases succeeded.
///
/// Each case contributes `1.0` (`Success`), `0.0` (`Failure`), or a `not_applicable`
/// sample when no call recorded an outcome (`missing_success_outcome` in Python). With
/// no case at all the metric is `not_applicable` with no score.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SuccessRateMetric;

impl SuccessRateMetric {
    /// Builds the metric; it is stateless.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Metric for SuccessRateMetric {
    fn name(&self) -> &str {
        SUCCESS_RATE
    }

    fn scope(&self) -> MetricScope {
        MetricScope::Trace
    }

    async fn evaluate(&self, input: &EvaluationInput) -> Result<MetricScore> {
        let cases = input.selected_cases()?;
        if cases.is_empty() {
            return Ok(MetricScore::not_applicable(
                SUCCESS_RATE,
                "no evaluation trace case was supplied",
            ));
        }
        let samples: Vec<MetricScore> = cases
            .iter()
            .copied()
            .map(|case| match case.outcome() {
                Some(CaseOutcome::Success) => MetricScore::scored(SUCCESS_RATE, 1.0, true),
                Some(CaseOutcome::Failure) => MetricScore::scored(SUCCESS_RATE, 0.0, false)
                    .with_detail("the execution did not succeed"),
                None => MetricScore::not_applicable(
                    SUCCESS_RATE,
                    "the trace does not contain an execution outcome",
                ),
            })
            .collect();
        Ok(aggregate_metric(SUCCESS_RATE, &samples))
    }
}

/// Observes caller-supplied call latencies without scoring them.
///
/// Mirrors Python's `LatencyEvaluator`: the result is `Observed` with `score = None`,
/// and the `detail` string carries the [`LatencyStatistics`](crate::evaluation::LatencyStatistics)
/// summary (average, p50, p95, max, count). Without a case, or without any non-negative
/// finite `duration_ms`, the metric is `not_applicable`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LatencyMetric;

impl LatencyMetric {
    /// Builds the metric; it is stateless.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Metric for LatencyMetric {
    fn name(&self) -> &str {
        LATENCY
    }

    fn scope(&self) -> MetricScope {
        MetricScope::Trace
    }

    async fn evaluate(&self, input: &EvaluationInput) -> Result<MetricScore> {
        let cases = input.selected_cases()?;
        if cases.is_empty() {
            return Ok(MetricScore::not_applicable(
                LATENCY,
                "no evaluation trace case was supplied",
            ));
        }
        let samples: Vec<f64> = cases
            .iter()
            .flat_map(|case| case.calls.iter())
            .filter_map(|call| call.duration_ms)
            .collect();
        let Some(statistics) = latency_statistics(samples) else {
            return Ok(MetricScore::not_applicable(
                LATENCY,
                "the trace has no latency observations",
            ));
        };
        Ok(MetricScore::observed(LATENCY, statistics.summary()))
    }
}

/// Judges answer accuracy, with Python's exact-match short-circuit.
///
/// A case whose `actual_answer` equals its `expected_answer` passes immediately with
/// `detail = "evaluation_method: exact_match"`-style wording and never calls a model.
/// A case with no query or no answer is `not_applicable` (`missing_output`). Everything
/// else goes to [`LlmJudgeMetric`] with `ACCURACY_RUBRIC` and the deterministic
/// fallback.
#[derive(Debug, Clone)]
pub struct AccuracyMetric {
    /// The shared judge implementation.
    judge: LlmJudgeMetric,
}

impl Default for AccuracyMetric {
    fn default() -> Self {
        Self::new()
    }
}

impl AccuracyMetric {
    /// Builds the metric with no model (deterministic fallback only).
    #[must_use]
    pub fn new() -> Self {
        Self {
            judge: LlmJudgeMetric::new(ACCURACY, ACCURACY_RUBRIC)
                .with_fallback(FallbackMode::Deterministic),
        }
    }

    /// Sets the completion model used for judgement.
    #[must_use]
    pub fn with_model(mut self, model: Arc<dyn CompletionModel>) -> Self {
        self.judge = self.judge.with_model(model);
        self
    }

    /// Whether a model is configured.
    #[must_use]
    pub fn has_model(&self) -> bool {
        self.judge.has_model()
    }
}

#[async_trait]
impl Metric for AccuracyMetric {
    fn name(&self) -> &str {
        ACCURACY
    }

    fn scope(&self) -> MetricScope {
        MetricScope::Trace
    }

    async fn evaluate(&self, input: &EvaluationInput) -> Result<MetricScore> {
        evaluate_trace_cases(ACCURACY, &self.judge, input, accuracy_guard).await
    }
}

/// The deterministic accuracy verdict for one case, or `None` when the judge is needed.
fn accuracy_guard(case: &EvaluationCase) -> Option<MetricScore> {
    let exact_match = case.expected_answer.is_some() && case.expected_answer == case.actual_answer;
    if exact_match {
        return Some(
            MetricScore::scored(ACCURACY, 1.0, true)
                .with_detail("output exactly matches the supplied expected answer"),
        );
    }
    let has_query = !case.query.trim().is_empty();
    let has_answer = case
        .actual_answer
        .as_deref()
        .is_some_and(|answer| !answer.trim().is_empty());
    if !has_query || !has_answer {
        return Some(MetricScore::not_applicable(
            ACCURACY,
            "the trace has no usable answer",
        ));
    }
    None
}

/// Judges whether the trace selected suitable capabilities.
///
/// Python's guard is "no capability calls in the trace": such a case is
/// `not_applicable` (`missing_capability_calls`); every other case goes to
/// [`LlmJudgeMetric`] with `CAPABILITY_SELECTION_RUBRIC`.
#[derive(Debug, Clone)]
pub struct CapabilitySelectionMetric {
    /// The shared judge implementation.
    judge: LlmJudgeMetric,
}

impl Default for CapabilitySelectionMetric {
    fn default() -> Self {
        Self::new()
    }
}

impl CapabilitySelectionMetric {
    /// Builds the metric with no model (deterministic fallback only).
    #[must_use]
    pub fn new() -> Self {
        Self {
            judge: LlmJudgeMetric::new(CAPABILITY_SELECTION, CAPABILITY_SELECTION_RUBRIC)
                .with_fallback(FallbackMode::Deterministic),
        }
    }

    /// Sets the completion model used for judgement.
    #[must_use]
    pub fn with_model(mut self, model: Arc<dyn CompletionModel>) -> Self {
        self.judge = self.judge.with_model(model);
        self
    }

    /// Whether a model is configured.
    #[must_use]
    pub fn has_model(&self) -> bool {
        self.judge.has_model()
    }
}

#[async_trait]
impl Metric for CapabilitySelectionMetric {
    fn name(&self) -> &str {
        CAPABILITY_SELECTION
    }

    fn scope(&self) -> MetricScope {
        MetricScope::Trace
    }

    async fn evaluate(&self, input: &EvaluationInput) -> Result<MetricScore> {
        evaluate_trace_cases(
            CAPABILITY_SELECTION,
            &self.judge,
            input,
            |case: &EvaluationCase| {
                if case.calls.is_empty() {
                    return Some(MetricScore::not_applicable(
                        CAPABILITY_SELECTION,
                        "the trace contains no capability selections",
                    ));
                }
                None
            },
        )
        .await
    }
}

/// Judges the ordering and hand-offs of a multi-capability trace.
///
/// Python requires at least two observed calls: a case with fewer is `not_applicable`
/// (`insufficient_composition_calls`); every other case goes to [`LlmJudgeMetric`] with
/// `COMPOSITION_RUBRIC`.
#[derive(Debug, Clone)]
pub struct CompositionEffectivenessMetric {
    /// The shared judge implementation.
    judge: LlmJudgeMetric,
}

impl Default for CompositionEffectivenessMetric {
    fn default() -> Self {
        Self::new()
    }
}

impl CompositionEffectivenessMetric {
    /// Builds the metric with no model (deterministic fallback only).
    #[must_use]
    pub fn new() -> Self {
        Self {
            judge: LlmJudgeMetric::new(COMPOSITION_EFFECTIVENESS, COMPOSITION_RUBRIC)
                .with_fallback(FallbackMode::Deterministic),
        }
    }

    /// Sets the completion model used for judgement.
    #[must_use]
    pub fn with_model(mut self, model: Arc<dyn CompletionModel>) -> Self {
        self.judge = self.judge.with_model(model);
        self
    }

    /// Whether a model is configured.
    #[must_use]
    pub fn has_model(&self) -> bool {
        self.judge.has_model()
    }
}

#[async_trait]
impl Metric for CompositionEffectivenessMetric {
    fn name(&self) -> &str {
        COMPOSITION_EFFECTIVENESS
    }

    fn scope(&self) -> MetricScope {
        MetricScope::Trace
    }

    async fn evaluate(&self, input: &EvaluationInput) -> Result<MetricScore> {
        evaluate_trace_cases(
            COMPOSITION_EFFECTIVENESS,
            &self.judge,
            input,
            |case: &EvaluationCase| {
                if case.calls.len() < 2 {
                    return Some(MetricScore::not_applicable(
                        COMPOSITION_EFFECTIVENESS,
                        "composition evaluation requires at least two observed capability calls",
                    ));
                }
                None
            },
        )
        .await
    }
}

/// Runs `judge` once per selected case, using `guard` for the deterministic verdicts.
///
/// A case for which `guard` returns a verdict is recorded verbatim; otherwise the judge
/// scores the input. The per-case verdicts are combined with
/// [`aggregate_metric`], so a single case is returned unchanged.
async fn evaluate_trace_cases(
    name: &str,
    judge: &LlmJudgeMetric,
    input: &EvaluationInput,
    mut guard: impl FnMut(&EvaluationCase) -> Option<MetricScore>,
) -> Result<MetricScore> {
    let cases = input.selected_cases()?;
    if cases.is_empty() {
        return Ok(MetricScore::not_applicable(
            name,
            "no evaluation trace case was supplied",
        ));
    }
    let mut samples = Vec::with_capacity(cases.len());
    for case in cases.iter().copied() {
        match guard(case) {
            Some(verdict) => samples.push(verdict),
            None => samples.push(judge.evaluate(input).await.map(normalize_status)?),
        }
    }
    Ok(aggregate_metric(name, &samples))
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
    use crate::evaluation::metrics::complete_capability;
    use crate::evaluation::trace::CapabilityCall;
    use crate::evaluation::MetricStatus;

    fn input(capability: crate::model::CapabilityDescriptor) -> EvaluationInput {
        EvaluationInput::new(capability)
    }

    fn trace_input(
        capability: crate::model::CapabilityDescriptor,
        cases: Vec<EvaluationCase>,
    ) -> EvaluationInput {
        EvaluationInput::new(capability)
            .with_trace(crate::evaluation::trace::EvaluationTrace::from_cases(cases))
    }

    #[tokio::test]
    async fn success_rate_scores_each_case_and_skips_unknown_outcomes() {
        let cases = vec![
            EvaluationCase::new("ok", "q")
                .with_calls(vec![CapabilityCall::new("sum").with_success(true)]),
            EvaluationCase::new("failed", "q")
                .with_calls(vec![CapabilityCall::new("sum").with_success(false)]),
            EvaluationCase::new("unknown", "q").with_calls(vec![CapabilityCall::new("sum")]),
        ];
        let score = SuccessRateMetric::new()
            .evaluate(&trace_input(complete_capability(), cases))
            .await
            .unwrap();
        assert_eq!(score.score, Some(0.5));
        assert_eq!(score.status, MetricStatus::Fail);
        assert!(!score.passed);
        let detail = score.detail.as_deref().unwrap();
        assert!(detail.contains("3 samples"), "{detail}");
        assert!(detail.contains("1 pass"), "{detail}");
        assert!(detail.contains("1 fail"), "{detail}");
        assert!(detail.contains("1 not_applicable"), "{detail}");
    }

    #[tokio::test]
    async fn success_rate_is_not_applicable_without_a_case_or_an_outcome() {
        let score = SuccessRateMetric::new()
            .evaluate(&input(complete_capability()))
            .await
            .unwrap();
        assert_eq!(score.status, MetricStatus::NotApplicable);
        assert_eq!(score.score, None);
        assert_eq!(
            score.detail.as_deref(),
            Some("no evaluation trace case was supplied")
        );

        let undecided = EvaluationCase::new("c", "q").with_calls(vec![CapabilityCall::new("sum")]);
        let score = SuccessRateMetric::new()
            .evaluate(&trace_input(complete_capability(), vec![undecided]))
            .await
            .unwrap();
        assert_eq!(score.status, MetricStatus::NotApplicable);
        assert_eq!(score.score, None);
        assert_eq!(
            score.detail.as_deref(),
            Some("the trace does not contain an execution outcome")
        );
    }

    #[tokio::test]
    async fn latency_is_observed_without_a_score() {
        let cases = vec![
            EvaluationCase::new("one", "q").with_calls(vec![
                CapabilityCall::new("sum").with_duration_ms(3.0),
                CapabilityCall::new("sum").with_duration_ms(1.0),
            ]),
            EvaluationCase::new("two", "q")
                .with_calls(vec![CapabilityCall::new("sum").with_duration_ms(5.0)]),
        ];
        let score = LatencyMetric::new()
            .evaluate(&trace_input(complete_capability(), cases))
            .await
            .unwrap();
        assert_eq!(score.status, MetricStatus::Observed);
        assert_eq!(score.score, None);
        let detail = score.detail.as_deref().unwrap();
        assert!(detail.contains("avg_ms=3.000"), "{detail}");
        assert!(detail.contains("p50_ms=3.000"), "{detail}");
        assert!(detail.contains("p95_ms=4.800"), "{detail}");
        assert!(detail.contains("max_ms=5.000"), "{detail}");
        assert!(detail.contains("count=3"), "{detail}");
    }

    #[tokio::test]
    async fn latency_without_a_duration_is_not_applicable() {
        let cases =
            vec![EvaluationCase::new("one", "q").with_calls(vec![CapabilityCall::new("sum")])];
        let score = LatencyMetric::new()
            .evaluate(&trace_input(complete_capability(), cases))
            .await
            .unwrap();
        assert_eq!(score.status, MetricStatus::NotApplicable);
        assert_eq!(score.score, None);
        assert_eq!(
            score.detail.as_deref(),
            Some("the trace has no latency observations")
        );
    }

    #[tokio::test]
    async fn accuracy_short_circuits_on_an_exact_match() {
        let cases = vec![EvaluationCase::new("exact", "2+2").with_answers("4", "4")];
        let score = AccuracyMetric::new()
            .evaluate(&trace_input(complete_capability(), cases))
            .await
            .unwrap();
        assert_eq!(score.score, Some(1.0));
        assert_eq!(score.status, MetricStatus::Pass);
        assert_eq!(
            score.detail.as_deref(),
            Some("output exactly matches the supplied expected answer")
        );
    }

    #[tokio::test]
    async fn accuracy_is_not_applicable_without_a_usable_answer() {
        let cases = vec![EvaluationCase::new("empty", "")];
        let score = AccuracyMetric::new()
            .evaluate(&trace_input(complete_capability(), cases))
            .await
            .unwrap();
        assert_eq!(score.status, MetricStatus::NotApplicable);
        assert_eq!(score.score, None);
        assert_eq!(
            score.detail.as_deref(),
            Some("the trace has no usable answer")
        );
    }

    #[tokio::test]
    async fn accuracy_judges_with_the_deterministic_fallback() {
        let cases = vec![EvaluationCase::new("mismatch", "2+2").with_answers("4", "5")];
        let score = AccuracyMetric::new()
            .evaluate(&trace_input(complete_capability(), cases))
            .await
            .unwrap();
        assert_eq!(score.score, Some(1.0));
        assert_eq!(score.status, MetricStatus::Pass);
        let detail = score.detail.as_deref().unwrap();
        assert!(detail.starts_with("deterministic_fallback:"), "{detail}");
    }

    #[tokio::test]
    async fn selection_and_composition_guards_are_not_applicable() {
        let empty = vec![EvaluationCase::new("empty", "q")];
        let score = CapabilitySelectionMetric::new()
            .evaluate(&trace_input(complete_capability(), empty.clone()))
            .await
            .unwrap();
        assert_eq!(score.status, MetricStatus::NotApplicable);
        assert_eq!(score.score, None);
        assert_eq!(
            score.detail.as_deref(),
            Some("the trace contains no capability selections")
        );

        let single = vec![EvaluationCase::new("single", "q")
            .with_calls(vec![CapabilityCall::new("sum").with_success(true)])];
        let score = CompositionEffectivenessMetric::new()
            .evaluate(&trace_input(complete_capability(), single))
            .await
            .unwrap();
        assert_eq!(score.status, MetricStatus::NotApplicable);
        assert_eq!(score.score, None);
        assert_eq!(
            score.detail.as_deref(),
            Some("composition evaluation requires at least two observed capability calls")
        );
    }

    #[tokio::test]
    async fn selection_and_composition_judge_when_evidence_exists() {
        let two_calls = vec![EvaluationCase::new("two", "q").with_calls(vec![
            CapabilityCall::new("sum").with_success(true),
            CapabilityCall::new("mul").with_success(true),
        ])];
        let selection = CapabilitySelectionMetric::new()
            .evaluate(&trace_input(complete_capability(), two_calls.clone()))
            .await
            .unwrap();
        assert_eq!(selection.status, MetricStatus::Pass);
        assert!(selection
            .detail
            .as_deref()
            .unwrap_or_default()
            .starts_with("deterministic_fallback:"));
        let composition = CompositionEffectivenessMetric::new()
            .evaluate(&trace_input(complete_capability(), two_calls))
            .await
            .unwrap();
        assert_eq!(composition.status, MetricStatus::Pass);
    }
    #[test]
    fn judge_backed_metrics_expose_the_model_builder() {
        assert!(!AccuracyMetric::new().has_model());
        assert!(!crate::evaluation::DescriptionQualityMetric::new().has_model());
        assert!(!crate::evaluation::ClassificationConsistencyMetric::new().has_model());
        assert!(!CapabilitySelectionMetric::new().has_model());
        assert!(!CompositionEffectivenessMetric::new().has_model());
        assert_eq!(AccuracyMetric::default().name(), ACCURACY);
    }
}
