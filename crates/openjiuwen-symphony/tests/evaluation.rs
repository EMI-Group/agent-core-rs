//! Behaviour tests for [`openjiuwen_symphony::evaluation`].
//!
//! Four layers are covered:
//!
//! 1. The static suite — completeness, schema validity, structure conformance, and the
//!    deterministic description/classification guards.
//! 2. The trace metrics — success rate, latency (observed, never scored), accuracy's
//!    exact-match short-circuit, and the selection/composition `not_applicable` guards.
//! 3. Window selection and per-metric aggregation, including Python's confidence
//!    thresholds and the `cache_signature` cache key.
//! 4. The LLM-judged metric, a thin delegating wrapper over
//!    [`openjiuwen_symphony::llm::LlmJudgeMetric`] — exercised offline, with a scripted
//!    verdict, and with a model that answers garbage twice (both fallback modes).
//!
//! Nothing touches the network: the only model is a local fake, no credentials are read,
//! and no assertion depends on the clock.
#![allow(clippy::unwrap_used, clippy::expect_used)]
#![allow(clippy::float_cmp)]
#![allow(clippy::assert_is_empty)]
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use openjiuwen_symphony::evaluation::{
    aggregate_metric, latency_statistics, percentile, CapabilityCall, CaseOutcome,
    ClassificationConsistencyMetric, CompletenessMetric, DescriptionQualityMetric, EvaluationCase,
    EvaluationInput, EvaluationReport, EvaluationSuite, EvaluationTrace, EvaluationWindow,
    LatencyMetric, LatencyStatistics, LlmJudgedMetric, Metric, MetricScope, MetricScore,
    MetricStatus, QualityConfidence, QualityResult, SchemaValidityMetric,
    StructureConformanceMetric, SuccessRateMetric, SYMPHONY_EVALUATION_PROTOCOL,
};
use openjiuwen_symphony::llm::judge::DEFAULT_RUBRIC;
use openjiuwen_symphony::llm::{CompletionModel, FallbackMode};
use openjiuwen_symphony::model::{CapabilityDescriptor, CapabilityIo};
use openjiuwen_symphony::{Error, Result};
use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::{json, Map, Value};

/// A [`CompletionModel`] that replays scripted replies in order and records every prompt.
///
/// Once the script is exhausted the last reply repeats, so a two-call judgement (first
/// attempt plus repair) can be scripted with a single entry.
#[derive(Debug, Default)]
struct ScriptedJudge {
    /// The scripted replies.
    replies: Vec<String>,
    /// Number of completed calls; also the index into `replies`.
    calls: AtomicUsize,
    /// Prompts received so far, in call order.
    prompts: Mutex<Vec<String>>,
}

impl ScriptedJudge {
    /// Builds a model replaying `replies` in order.
    fn new(replies: &[&str]) -> Arc<Self> {
        Arc::new(Self {
            replies: replies.iter().map(|reply| (*reply).to_string()).collect(),
            calls: AtomicUsize::new(0),
            prompts: Mutex::new(Vec::new()),
        })
    }

    /// Builds a model replaying one reply, however often it is called.
    fn single(reply: &str) -> Arc<Self> {
        Self::new(&[reply])
    }

    /// Number of calls made so far.
    fn call_count(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }

    /// Prompts received so far, in call order.
    fn prompts(&self) -> Vec<String> {
        self.prompts.lock().unwrap().clone()
    }
}

#[async_trait]
impl CompletionModel for ScriptedJudge {
    fn name(&self) -> &'static str {
        "scripted-judge"
    }

    async fn complete(&self, prompt: &str) -> Result<String> {
        self.prompts.lock().unwrap().push(prompt.to_string());
        let index = self.calls.fetch_add(1, Ordering::SeqCst);
        let Some(last) = self.replies.len().checked_sub(1) else {
            return Err(Error::llm("ScriptedJudge has no scripted replies"));
        };
        let index = index.min(last);
        Ok(self.replies[index].clone())
    }
}

/// A card that satisfies every metric, including the judge's deterministic rubric.
fn complete_capability() -> CapabilityDescriptor {
    let card = CapabilityDescriptor::new("tool", "sum", "Sum")
        .with_description("Adds two numbers.")
        .with_classification("math")
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
    with_hash(card)
}

/// Fills the cached content hash, as `CapabilityDescriptor::fingerprint` would.
fn with_hash(mut card: CapabilityDescriptor) -> CapabilityDescriptor {
    card.content_hash = card.compute_content_hash();
    card
}

/// A card with no description and no described IO: the judge scores it `0.0`.
fn thin_capability() -> CapabilityDescriptor {
    CapabilityDescriptor::new("tool", "thin", "Thin")
}

/// Wraps a capability in the input a static metric inspects.
fn input(capability: CapabilityDescriptor) -> EvaluationInput {
    EvaluationInput::new(capability)
}

/// Wraps a capability and cases in a trace-bearing input.
fn trace_input(capability: CapabilityDescriptor, cases: Vec<EvaluationCase>) -> EvaluationInput {
    EvaluationInput::new(capability).with_trace(EvaluationTrace::from_cases(cases))
}

/// Serializes and deserializes `value`, asserting the round-trip is lossless.
fn round_trip<T>(value: &T) -> T
where
    T: Serialize + DeserializeOwned,
{
    let encoded = serde_json::to_value(value).unwrap();
    serde_json::from_value(encoded).unwrap()
}

// --- the static suite -----------------------------------------------------
#[tokio::test]
async fn default_suite_scores_a_well_formed_card() {
    let report = EvaluationSuite::default_suite()
        .evaluate(&input(complete_capability()))
        .await;
    assert!(report.errors.is_empty(), "{:?}", report.errors);
    assert_eq!(report.capability_id, "sum");
    assert_eq!(report.scores.len(), 2);
    assert_eq!(report.scores[0].metric, "completeness");
    assert_eq!(report.scores[0].score, Some(1.0));
    assert_eq!(report.scores[0].status, MetricStatus::Pass);
    assert!(report.scores[0].passed);
    assert_eq!(report.scores[0].detail, None);
    assert_eq!(report.scores[1].metric, "schema_validity");
    assert_eq!(report.scores[1].score, Some(1.0));
    assert!(report.scores[1].passed);
    assert_eq!(report.scores[1].detail, None);
    assert_eq!(report.overall, 1.0);
}

#[tokio::test]
async fn the_static_metrics_are_reusable_as_arcs() {
    let suite = EvaluationSuite::new(vec![
        Arc::new(CompletenessMetric::new()),
        Arc::new(SchemaValidityMetric::new()),
        Arc::new(StructureConformanceMetric::new()),
        Arc::new(DescriptionQualityMetric::new()),
        Arc::new(ClassificationConsistencyMetric::new()),
    ]);
    assert_eq!(suite.metrics().len(), 5);
    for metric in suite.metrics() {
        assert_eq!(metric.scope(), MetricScope::Static, "{}", metric.name());
        let score = metric
            .evaluate(&input(complete_capability()))
            .await
            .unwrap();
        assert!(score.passed, "{}: {:?}", metric.name(), score.detail);
        assert_eq!(score.status, MetricStatus::Pass, "{}", metric.name());
        assert_eq!(score.score, Some(1.0), "{}", metric.name());
    }
}

#[tokio::test]
async fn structure_conformance_accepts_a_conforming_card() {
    let score = StructureConformanceMetric::new()
        .evaluate(&input(complete_capability()))
        .await
        .unwrap();
    assert_eq!(score.metric, "structure_conformance");
    assert_eq!(score.score, Some(1.0));
    assert_eq!(score.status, MetricStatus::Pass);
    let detail = score.detail.as_deref().unwrap();
    assert!(detail.starts_with("checked: "), "{detail}");
    for field in [
        "capability_id",
        "capability_type",
        "name",
        "content_hash",
        "inputs",
        "outputs",
        "tags",
    ] {
        assert!(detail.contains(field), "{field}: {detail}");
    }
}

#[tokio::test]
async fn structure_conformance_reports_every_bad_field() {
    let card = CapabilityDescriptor::new("tool", "", "  ").with_inputs(vec![
        CapabilityIo::new("dup", "number"),
        CapabilityIo::new("dup", "number"),
    ]);
    let score = StructureConformanceMetric::new()
        .evaluate(&input(card))
        .await
        .unwrap();
    assert_eq!(score.score, Some(0.0));
    assert_eq!(score.status, MetricStatus::Fail);
    assert!(!score.passed);
    let detail = score.detail.as_deref().unwrap();
    assert!(
        detail.contains("capability_id: must be a non-empty string"),
        "{detail}"
    );
    assert!(
        detail.contains("name: must be a non-empty string"),
        "{detail}"
    );
    assert!(
        detail.contains("content_hash: must be a non-empty string"),
        "{detail}"
    );
    assert!(detail.contains("inputs: names must be unique"), "{detail}");
}

#[tokio::test]
async fn the_deterministic_guards_fail_without_calling_a_model() {
    let description = DescriptionQualityMetric::new()
        .evaluate(&input(thin_capability()))
        .await
        .unwrap();
    assert_eq!(description.status, MetricStatus::Fail);
    assert_eq!(description.score, Some(0.0));
    assert_eq!(
        description.detail.as_deref(),
        Some("the capability description is empty")
    );

    let mut unclassified = complete_capability().with_tags(Vec::<String>::new());
    unclassified.classification.clear();
    let classification = ClassificationConsistencyMetric::new()
        .evaluate(&input(unclassified))
        .await
        .unwrap();
    assert_eq!(classification.status, MetricStatus::Fail);
    assert_eq!(
        classification.detail.as_deref(),
        Some("the capability has neither a classification nor tags")
    );
}

// --- the trace metrics ----------------------------------------------------
#[tokio::test]
async fn success_rate_means_mixed_outcomes_and_skips_undecided_cases() {
    let cases = vec![
        EvaluationCase::new("ok-1", "q")
            .with_calls(vec![CapabilityCall::new("sum").with_success(true)]),
        EvaluationCase::new("ok-2", "q").with_calls(vec![
            CapabilityCall::new("sum").with_success(true),
            CapabilityCall::new("mul").with_success(true),
        ]),
        EvaluationCase::new("failed", "q")
            .with_calls(vec![CapabilityCall::new("sum").with_success(false)]),
        EvaluationCase::new("undecided", "q").with_calls(vec![CapabilityCall::new("sum")]),
    ];
    let score = SuccessRateMetric::new()
        .evaluate(&trace_input(complete_capability(), cases))
        .await
        .unwrap();
    assert_eq!(score.metric, "success_rate");
    assert_eq!(score.score, Some(2.0 / 3.0));
    assert_eq!(score.status, MetricStatus::Fail);
    let detail = score.detail.as_deref().unwrap();
    assert!(detail.contains("4 samples"), "{detail}");
    assert!(detail.contains("2 pass"), "{detail}");
    assert!(detail.contains("1 fail"), "{detail}");
    assert!(detail.contains("1 not_applicable"), "{detail}");
}

#[tokio::test]
async fn success_rate_is_not_applicable_without_a_trace_or_an_outcome() {
    let no_trace = SuccessRateMetric::new()
        .evaluate(&input(complete_capability()))
        .await
        .unwrap();
    assert_eq!(no_trace.status, MetricStatus::NotApplicable);
    assert_eq!(no_trace.score, None);
    assert!(!no_trace.passed);

    let undecided =
        vec![EvaluationCase::new("c", "q").with_calls(vec![CapabilityCall::new("sum")])];
    let score = SuccessRateMetric::new()
        .evaluate(&trace_input(complete_capability(), undecided))
        .await
        .unwrap();
    assert_eq!(score.status, MetricStatus::NotApplicable);
    assert_eq!(score.score, None);
}

#[tokio::test]
async fn latency_is_observed_with_statistics_and_no_score() {
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
    assert_eq!(score.metric, "latency");
    assert_eq!(score.status, MetricStatus::Observed);
    assert_eq!(score.score, None);
    assert!(!score.passed);
    let detail = score.detail.as_deref().unwrap();
    assert!(detail.contains("avg_ms=3.000"), "{detail}");
    assert!(detail.contains("p50_ms=3.000"), "{detail}");
    assert!(detail.contains("p95_ms=4.800"), "{detail}");
    assert!(detail.contains("max_ms=5.000"), "{detail}");
    assert!(detail.contains("count=3"), "{detail}");
}

#[tokio::test]
async fn latency_is_not_applicable_without_observations() {
    let no_trace = LatencyMetric::new()
        .evaluate(&input(complete_capability()))
        .await
        .unwrap();
    assert_eq!(no_trace.status, MetricStatus::NotApplicable);
    assert_eq!(no_trace.score, None);

    let no_duration =
        vec![EvaluationCase::new("one", "q").with_calls(vec![CapabilityCall::new("sum")])];
    let score = LatencyMetric::new()
        .evaluate(&trace_input(complete_capability(), no_duration))
        .await
        .unwrap();
    assert_eq!(score.status, MetricStatus::NotApplicable);
    assert_eq!(score.score, None);
    assert_eq!(
        score.detail.as_deref(),
        Some("the trace has no latency observations")
    );
}

// --- latency statistics ---------------------------------------------------
/// Compares two interpolated percentiles, which are not bit-exact for 0.95.
fn assert_close(actual: Option<f64>, expected: f64) {
    let actual = actual.unwrap();
    assert!(
        (actual - expected).abs() < 1e-9,
        "{actual} is not within 1e-9 of {expected}"
    );
}

#[test]
fn percentile_matches_hand_computed_interpolation() {
    assert_close(percentile(&[7.0], 0.5), 7.0);
    assert_close(percentile(&[7.0], 0.95), 7.0);
    assert_close(percentile(&[1.0, 2.0], 0.5), 1.5);
    assert_close(percentile(&[1.0, 2.0], 0.95), 1.95);
    assert_close(percentile(&[1.0, 2.0, 3.0, 4.0], 0.5), 2.5);
    assert_close(percentile(&[1.0, 2.0, 3.0, 4.0], 0.95), 3.85);
    assert_close(percentile(&[1.0, 2.0, 3.0, 4.0, 5.0], 0.5), 3.0);
    assert_close(percentile(&[1.0, 2.0, 3.0, 4.0, 5.0], 0.95), 4.8);
    let ten: Vec<f64> = (1..=10).map(f64::from).collect();
    assert_close(percentile(&ten, 0.5), 5.5);
    assert_close(percentile(&ten, 0.95), 9.55);
    assert_eq!(percentile(&[], 0.5), None);
    // Unsorted input is sorted internally.
    assert_close(percentile(&[4.0, 1.0, 3.0, 2.0], 0.5), 2.5);
}

#[test]
fn latency_statistics_are_ordered_and_drop_unusable_samples() {
    let stats = latency_statistics([5.0, -1.0, f64::NAN, 1.0, f64::INFINITY, 3.0]).unwrap();
    assert_eq!(stats.count, 3);
    assert_eq!(stats.samples_ms, vec![5.0, 1.0, 3.0]);
    assert_eq!(stats.avg_ms, 3.0);
    assert_eq!(stats.p50_ms, 3.0);
    assert_close(Some(stats.p95_ms), 4.8);
    assert_eq!(stats.max_ms, 5.0);
    assert!(stats.p50_ms <= stats.p95_ms);
    assert!(stats.p95_ms <= stats.max_ms);
    assert!(stats.avg_ms <= stats.max_ms);
    assert!(latency_statistics(Vec::<f64>::new()).is_none());
    assert!(latency_statistics([-1.0, f64::NAN]).is_none());
    let summary = stats.summary();
    assert!(summary.contains("count=3"), "{summary}");
}

// --- window selection -----------------------------------------------------
fn dated_cases() -> Vec<EvaluationCase> {
    vec![
        EvaluationCase::new("early", "q").with_event_time("2026-01-01T00:00:00Z"),
        EvaluationCase::new("start", "q").with_event_time("2026-01-02T00:00:00Z"),
        EvaluationCase::new("middle", "q").with_event_time("2026-01-02T12:00:00+08:00"),
        EvaluationCase::new("end", "q").with_event_time("2026-01-03T00:00:00Z"),
        EvaluationCase::new("undated", "q"),
    ]
}

#[test]
fn window_selection_is_start_inclusive_and_end_exclusive() {
    let cases = dated_cases();
    let start = EvaluationWindow::between(
        Some("2026-01-02T00:00:00Z".to_string()),
        Some("2026-01-03T00:00:00Z".to_string()),
    );
    let input = EvaluationInput::new(complete_capability())
        .with_trace(EvaluationTrace::from_cases(cases.clone()))
        .with_window(start);
    let selected = input.selected_cases().unwrap();
    let ids: Vec<&str> = selected.iter().map(|case| case.case_id.as_str()).collect();
    // `start` is included, `end` is excluded, `undated` is dropped, and the
    // `+08:00` timestamp (`2026-01-02T04:00:00Z`) falls inside the window.
    assert_eq!(ids, vec!["start", "middle"]);
    assert_eq!(input.sample_count().unwrap(), 2);

    let unbounded = EvaluationInput::new(complete_capability())
        .with_trace(EvaluationTrace::from_cases(cases.clone()))
        .with_window(EvaluationWindow::new());
    assert_eq!(unbounded.selected_cases().unwrap().len(), 5);
}

#[tokio::test]
async fn a_trace_metric_sees_only_windowed_cases() {
    let cases = vec![
        EvaluationCase::new("before", "q")
            .with_event_time("2025-12-31T00:00:00Z")
            .with_calls(vec![CapabilityCall::new("sum").with_success(false)]),
        EvaluationCase::new("inside", "q")
            .with_event_time("2026-01-02T00:00:00Z")
            .with_calls(vec![CapabilityCall::new("sum").with_success(true)]),
    ];
    let window = EvaluationWindow::between(Some("2026-01-01T00:00:00Z".to_string()), None);
    let score = SuccessRateMetric::new()
        .evaluate(&trace_input(complete_capability(), cases).with_window(window))
        .await
        .unwrap();
    assert_eq!(score.score, Some(1.0));
    assert_eq!(score.status, MetricStatus::Pass);
}

#[tokio::test]
async fn an_invalid_window_is_a_validation_error() {
    let inverted = EvaluationWindow::between(
        Some("2026-01-03T00:00:00Z".to_string()),
        Some("2026-01-02T00:00:00Z".to_string()),
    );
    let error = SuccessRateMetric::new()
        .evaluate(&trace_input(complete_capability(), dated_cases()).with_window(inverted))
        .await
        .unwrap_err();
    assert!(error.message().starts_with("window.end:"), "{error}");

    let bad_case = vec![EvaluationCase::new("bad", "q").with_event_time("yesterday")];
    let window = EvaluationWindow::between(Some("2026-01-01T00:00:00Z".to_string()), None);
    let error = SuccessRateMetric::new()
        .evaluate(&trace_input(complete_capability(), bad_case).with_window(window))
        .await
        .unwrap_err();
    assert!(error.message().starts_with("event_time:"), "{error}");
}

// --- aggregation and confidence ------------------------------------------
#[test]
fn aggregate_metric_means_scores_and_keeps_status_precedence() {
    let pass = MetricScore::scored("accuracy", 1.0, true);
    let fail = MetricScore::scored("accuracy", 0.0, false);
    let mixed = aggregate_metric("accuracy", &[pass.clone(), fail.clone(), pass.clone()]);
    assert_eq!(mixed.score, Some(2.0 / 3.0));
    assert_eq!(mixed.status, MetricStatus::Fail);
    assert!(!mixed.passed);
    let detail = mixed.detail.as_deref().unwrap();
    assert!(detail.contains("3 samples"), "{detail}");
    assert!(detail.contains("2 pass"), "{detail}");
    assert!(detail.contains("1 fail"), "{detail}");

    let single = aggregate_metric("accuracy", &[pass]);
    assert_eq!(single, MetricScore::scored("accuracy", 1.0, true));

    let error = MetricScore::error("latency", "boom");
    let observed = MetricScore::observed("latency", "raw");
    let skipped = MetricScore::not_applicable("latency", "nothing");
    assert_eq!(
        aggregate_metric(
            "latency",
            &[skipped.clone(), observed.clone(), error.clone()]
        )
        .status,
        MetricStatus::Error
    );
    assert_eq!(
        aggregate_metric("latency", &[skipped.clone(), observed.clone()]).status,
        MetricStatus::Observed
    );
    assert_eq!(
        aggregate_metric("latency", &[skipped]).status,
        MetricStatus::NotApplicable
    );
    assert_eq!(
        aggregate_metric("latency", &[]).status,
        MetricStatus::NotApplicable
    );
}

#[test]
fn confidence_follows_pythons_sample_count_thresholds() {
    assert_eq!(
        QualityConfidence::from_sample_count(0),
        QualityConfidence::None
    );
    assert_eq!(
        QualityConfidence::from_sample_count(1),
        QualityConfidence::Low
    );
    assert_eq!(
        QualityConfidence::from_sample_count(2),
        QualityConfidence::Normal
    );
    assert_eq!(
        QualityConfidence::from_sample_count(9),
        QualityConfidence::Normal
    );
    assert_eq!(
        QualityConfidence::from_sample_count(10),
        QualityConfidence::High
    );
    assert_eq!(QualityConfidence::default(), QualityConfidence::None);

    let suite = EvaluationSuite::new(Vec::new());
    for (count, expected) in [
        (0, QualityConfidence::None),
        (1, QualityConfidence::Low),
        (2, QualityConfidence::Normal),
        (9, QualityConfidence::Normal),
        (10, QualityConfidence::High),
    ] {
        let quality = suite.aggregate(&[], None, Some(count)).unwrap();
        assert_eq!(quality.confidence, expected, "{count}");
        assert_eq!(quality.sample_count, count);
        assert_eq!(quality.aggregation, "per_metric");
        assert_eq!(quality.score, None);
        assert!(quality.per_metric.is_empty());
    }
}

#[tokio::test]
async fn evaluate_quality_aggregates_per_metric_and_counts_cases() {
    let cases = vec![
        EvaluationCase::new("one", "q")
            .with_calls(vec![CapabilityCall::new("sum").with_success(true)]),
        EvaluationCase::new("two", "q")
            .with_calls(vec![CapabilityCall::new("sum").with_success(false)]),
    ];
    let suite = EvaluationSuite::new(vec![Arc::new(SuccessRateMetric)]);
    let quality = suite
        .evaluate_quality(&trace_input(complete_capability(), cases))
        .await
        .unwrap();
    assert_eq!(quality.sample_count, 2);
    assert_eq!(quality.confidence, QualityConfidence::Normal);
    assert_eq!(quality.aggregation, "per_metric");
    assert_eq!(quality.score, None);
    assert_eq!(quality.per_metric.len(), 1);
    let success = quality.per_metric.get("success_rate").unwrap();
    assert_eq!(success.score, Some(0.5));
    assert_eq!(success.status, MetricStatus::Fail);
}

#[tokio::test]
async fn a_failing_metric_becomes_an_error_sample_in_a_quality_result() {
    let metric = LlmJudgedMetric::new()
        .with_model(ScriptedJudge::single("not json"))
        .with_fallback(FallbackMode::Error);
    let suite = EvaluationSuite::new(vec![Arc::new(metric)]);
    let quality = suite
        .evaluate_quality(&input(complete_capability()))
        .await
        .unwrap();
    let judged = quality.per_metric.get("llm_judged").unwrap();
    assert_eq!(judged.status, MetricStatus::Error);
    assert_eq!(judged.score, None);
    assert!(judged
        .detail
        .as_deref()
        .unwrap_or_default()
        .starts_with("evaluator failed"));
}

#[test]
fn cache_signature_is_stable_and_content_sensitive() {
    let default = EvaluationSuite::default_suite().cache_signature();
    assert_eq!(default, EvaluationSuite::default_suite().cache_signature());
    assert_eq!(default.len(), 64);
    assert!(default.chars().all(|c| c.is_ascii_hexdigit()));
    assert_eq!(SYMPHONY_EVALUATION_PROTOCOL, "symphony-evaluation-suite-v5");

    let with_judge = EvaluationSuite::default_suite()
        .with_llm_judge()
        .cache_signature();
    assert_ne!(default, with_judge);

    let reordered = EvaluationSuite::new(vec![
        Arc::new(SchemaValidityMetric),
        Arc::new(CompletenessMetric),
    ])
    .cache_signature();
    assert_ne!(default, reordered);
    let empty = EvaluationSuite::new(Vec::new()).cache_signature();
    assert_ne!(default, empty);
}

// --- the judge, offline ---------------------------------------------------
#[tokio::test]
async fn with_llm_judge_scores_offline_without_errors() {
    let suite = EvaluationSuite::default_suite().with_llm_judge();
    assert_eq!(suite.metrics().len(), 3);
    let report = suite.evaluate(&input(complete_capability())).await;
    assert!(report.errors.is_empty(), "{:?}", report.errors);
    assert_eq!(report.scores.len(), 3);
    assert_eq!(report.scores[0].metric, "completeness");
    assert_eq!(report.scores[1].metric, "schema_validity");
    let judge = &report.scores[2];
    assert_eq!(judge.metric, "llm_judged");
    assert_eq!(judge.score, Some(1.0));
    assert_eq!(judge.status, MetricStatus::Pass);
    assert!(judge.passed);
    let detail = judge.detail.as_deref().unwrap();
    assert!(detail.starts_with("deterministic_fallback:"), "{detail}");
    assert!(detail.contains("no model is configured"), "{detail}");
    assert_eq!(report.overall, 1.0);
}

#[tokio::test]
async fn the_offline_judge_scores_a_thin_card_zero() {
    let report = EvaluationSuite::default_suite()
        .with_llm_judge()
        .evaluate(&input(thin_capability()))
        .await;
    assert!(report.errors.is_empty(), "{:?}", report.errors);
    assert_eq!(report.scores.len(), 3);
    let judge = &report.scores[2];
    assert_eq!(judge.metric, "llm_judged");
    assert_eq!(judge.score, Some(0.0));
    assert_eq!(judge.status, MetricStatus::Fail);
    assert!(!judge.passed);
    let detail = judge.detail.as_deref().unwrap();
    assert!(detail.starts_with("deterministic_fallback:"), "{detail}");
    assert!(detail.contains("missing: description"), "{detail}");
    // Completeness is 0.0, schema validity 1.0 (no declarations to be invalid), judge 0.0.
    assert_eq!(report.overall, 1.0 / 3.0);
}

// --- the judge, model-backed ----------------------------------------------
#[tokio::test]
async fn a_model_verdict_is_used_verbatim() {
    let model = ScriptedJudge::single(r#"{"score":1,"reason":"ok"}"#);
    let metric = LlmJudgedMetric::new().with_model(model.clone());
    let score = metric
        .evaluate(&input(complete_capability()))
        .await
        .unwrap();
    assert_eq!(score.metric, "llm_judged");
    assert_eq!(score.score, Some(1.0));
    assert_eq!(score.status, MetricStatus::Pass);
    assert!(score.passed);
    assert_eq!(score.detail.as_deref(), Some("ok"));
    assert_eq!(model.call_count(), 1);
    let prompt = &model.prompts()[0];
    assert!(prompt.contains("Capability card"), "{prompt}");
    assert!(prompt.contains(DEFAULT_RUBRIC), "{prompt}");
}

#[tokio::test]
async fn with_llm_judge_model_judges_inside_a_suite() {
    let model = ScriptedJudge::single(r#"{"score":0,"reason":"too terse"}"#);
    let suite = EvaluationSuite::default_suite().with_llm_judge_model(model.clone());
    assert_eq!(suite.metrics().len(), 3);
    let report = suite.evaluate(&input(complete_capability())).await;
    assert!(report.errors.is_empty(), "{:?}", report.errors);
    let judge = &report.scores[2];
    assert_eq!(judge.metric, "llm_judged");
    assert_eq!(judge.score, Some(0.0));
    assert_eq!(judge.status, MetricStatus::Fail);
    assert!(!judge.passed);
    assert_eq!(judge.detail.as_deref(), Some("too terse"));
    // The static metrics still score, so the report keeps their verdicts.
    assert_eq!(report.overall, 2.0 / 3.0);
    assert_eq!(model.call_count(), 1);
}

#[tokio::test]
async fn two_garbage_replies_are_recorded_as_one_error_in_error_mode() {
    let model = ScriptedJudge::single("not json");
    let metric = LlmJudgedMetric::new()
        .with_model(model.clone())
        .with_fallback(FallbackMode::Error);
    let report = EvaluationSuite::new(vec![Arc::new(metric)])
        .evaluate(&input(complete_capability()))
        .await;
    assert_eq!(report.errors.len(), 1, "{:?}", report.errors);
    let error = &report.errors[0];
    assert!(error.starts_with("llm_judged: "), "{error}");
    assert!(error.contains("rejected the model reply twice"), "{error}");
    assert!(report.scores.is_empty());
    assert_eq!(report.overall, 0.0);
    // Exactly one repair attempt: two calls, the second carrying the rejection.
    assert_eq!(model.call_count(), 2);
    assert!(
        model.prompts()[1].contains("rejected"),
        "{:?}",
        model.prompts()
    );
}

#[tokio::test]
async fn two_garbage_replies_fall_back_in_deterministic_mode() {
    let metric = LlmJudgedMetric::new()
        .with_model(ScriptedJudge::single("not json"))
        .with_fallback(FallbackMode::Deterministic);
    let report = EvaluationSuite::new(vec![Arc::new(metric)])
        .evaluate(&input(complete_capability()))
        .await;
    assert!(report.errors.is_empty(), "{:?}", report.errors);
    assert_eq!(report.scores.len(), 1);
    let judge = &report.scores[0];
    assert_eq!(judge.metric, "llm_judged");
    assert_eq!(judge.score, Some(1.0));
    assert!(judge.passed);
    let detail = judge.detail.as_deref().unwrap();
    assert!(detail.starts_with("deterministic_fallback:"), "{detail}");
    assert!(detail.contains("model judgement failed"), "{detail}");
}

// --- the wrapper surface --------------------------------------------------
#[test]
fn the_wrapper_keeps_its_name_and_builders() {
    let metric = LlmJudgedMetric::new();
    assert_eq!(metric.name(), "llm_judged");
    assert_eq!(metric.scope(), MetricScope::Static);
    assert_eq!(metric.rubric(), DEFAULT_RUBRIC);
    assert!(!metric.has_model());
    assert_eq!(LlmJudgedMetric::default().rubric(), DEFAULT_RUBRIC);

    let custom = LlmJudgedMetric::new()
        .with_rubric("must cite an example")
        .with_model(ScriptedJudge::single(r#"{"score":1,"reason":"ok"}"#))
        .with_fallback(FallbackMode::Error);
    assert_eq!(custom.name(), "llm_judged");
    assert_eq!(custom.rubric(), "must cite an example");
    assert!(custom.has_model());
}

#[test]
fn the_wrapper_is_debug_and_clone() {
    let metric = LlmJudgedMetric::new().with_rubric("custom");
    let cloned = metric.clone();
    assert_eq!(cloned.rubric(), "custom");
    let rendered = format!("{cloned:?}");
    assert!(rendered.contains("LlmJudgeMetric"), "{rendered}");
}

// --- the report -----------------------------------------------------------
#[test]
fn reports_round_trip_through_json() {
    let report = EvaluationReport {
        capability_id: "tool:sum".to_string(),
        scores: vec![
            MetricScore::scored("completeness", 0.75, true).with_detail("missing: tags"),
            MetricScore::observed("latency", "avg_ms=1.000"),
            MetricScore::not_applicable("accuracy", "no case"),
        ],
        overall: 0.75,
        errors: vec!["schema_validity: backend unavailable".to_string()],
    };
    let encoded = serde_json::to_value(&report).unwrap();
    assert_eq!(encoded["capability_id"], "tool:sum");
    assert_eq!(encoded["scores"][0]["metric"], "completeness");
    assert_eq!(encoded["scores"][0]["status"], "pass");
    assert_eq!(encoded["scores"][1]["status"], "observed");
    assert_eq!(encoded["scores"][1]["score"], Value::Null);
    assert_eq!(encoded["scores"][2]["status"], "not_applicable");
    assert_eq!(encoded["errors"][0], "schema_validity: backend unavailable");
    let decoded: EvaluationReport = serde_json::from_value(encoded).unwrap();
    assert_eq!(decoded, report);
}

#[tokio::test]
async fn an_empty_suite_reports_a_zero_overall() {
    let report = EvaluationSuite::new(Vec::new())
        .evaluate(&input(complete_capability()))
        .await;
    assert!(report.scores.is_empty());
    assert!(report.errors.is_empty());
    assert_eq!(report.overall, 0.0);
    assert_eq!(report.capability_id, "sum");
}

#[tokio::test]
async fn evaluate_static_and_evaluate_case_split_the_suite_by_scope() {
    let case = EvaluationCase::new("case-1", "q")
        .with_calls(vec![
            CapabilityCall::new("sum")
                .with_success(true)
                .with_duration_ms(2.0),
            CapabilityCall::new("mul")
                .with_success(true)
                .with_duration_ms(4.0),
        ])
        .with_answers("4", "4");
    let suite = EvaluationSuite::new(vec![
        Arc::new(CompletenessMetric),
        Arc::new(SuccessRateMetric),
        Arc::new(LatencyMetric),
    ]);
    let input = trace_input(complete_capability(), vec![case.clone()]);
    let static_scores = suite.evaluate_static(&input).await;
    assert_eq!(static_scores.len(), 1);
    assert_eq!(static_scores[0].metric, "completeness");
    let case_scores = suite.evaluate_case(&input, &case).await;
    let names: Vec<&str> = case_scores
        .iter()
        .map(|score| score.metric.as_str())
        .collect();
    assert_eq!(names, vec!["success_rate", "latency"]);
    assert_eq!(case_scores[0].score, Some(1.0));
    assert_eq!(case_scores[1].status, MetricStatus::Observed);
}

// --- serialization of every data type -------------------------------------
#[test]
fn status_and_confidence_enums_round_trip() {
    for status in [
        MetricStatus::Pass,
        MetricStatus::Fail,
        MetricStatus::NotApplicable,
        MetricStatus::Error,
        MetricStatus::Observed,
    ] {
        assert_eq!(round_trip(&status), status);
        let encoded = serde_json::to_value(status).unwrap();
        assert_eq!(encoded, Value::String(status.as_str().to_string()));
        assert_eq!(
            status.is_scored(),
            matches!(status, MetricStatus::Pass | MetricStatus::Fail)
        );
    }
    for scope in [MetricScope::Static, MetricScope::Trace] {
        assert_eq!(round_trip(&scope), scope);
        assert_eq!(
            serde_json::to_value(scope).unwrap(),
            Value::String(scope.as_str().to_string())
        );
    }
    for confidence in [
        QualityConfidence::None,
        QualityConfidence::Low,
        QualityConfidence::Normal,
        QualityConfidence::High,
    ] {
        assert_eq!(round_trip(&confidence), confidence);
        assert_eq!(
            serde_json::to_value(confidence).unwrap(),
            Value::String(confidence.as_str().to_string())
        );
    }
    for outcome in [CaseOutcome::Success, CaseOutcome::Failure] {
        assert_eq!(round_trip(&outcome), outcome);
    }
}

#[test]
fn trace_types_round_trip_through_json() {
    let call = CapabilityCall::new("sum")
        .with_success(true)
        .with_duration_ms(12.5)
        .with_event_time("2026-01-02T03:04:05Z")
        .with_arguments(json!({"a": 1, "b": "two"}));
    assert_eq!(round_trip(&call), call);
    assert_eq!(serde_json::to_value(&call).unwrap()["name"], "sum");

    let mut metadata = Map::new();
    metadata.insert("origin".to_string(), json!("test"));
    let case = EvaluationCase::new("case-1", "add 1 and 2")
        .with_event_time("2026-01-02T03:04:05Z")
        .with_calls(vec![call.clone()])
        .with_answers("3", "3")
        .with_metadata(metadata.clone());
    assert_eq!(round_trip(&case), case);
    assert_eq!(case.outcome(), Some(CaseOutcome::Success));

    let trace = EvaluationTrace::from_cases(vec![case.clone()]);
    assert_eq!(round_trip(&trace), trace);
    assert_eq!(trace.len(), 1);
    assert!(!trace.is_empty());
    assert_eq!(EvaluationTrace::new().len(), 0);

    let window = EvaluationWindow::between(
        Some("2026-01-01T00:00:00Z".to_string()),
        Some("2026-02-01T00:00:00Z".to_string()),
    );
    assert_eq!(round_trip(&window), window);
    assert!(window.is_bounded());
    assert!(!EvaluationWindow::new().is_bounded());

    let stats = latency_statistics([1.0, 2.0, 3.0]).unwrap();
    assert_eq!(round_trip(&stats), stats);
    assert_eq!(LatencyStatistics::default().count, 0);
}

#[test]
fn metric_and_quality_results_round_trip_through_json() {
    let scored = MetricScore::scored("accuracy", 0.5, false).with_detail("half right");
    assert_eq!(round_trip(&scored), scored);
    let observed = MetricScore::observed("latency", "raw");
    assert_eq!(round_trip(&observed), observed);
    assert_eq!(observed.effective_score(), None);
    assert_eq!(scored.effective_score(), Some(0.5));

    // A legacy payload without `status` reads as a scored sample.
    let legacy: MetricScore = serde_json::from_value(json!({
        "metric": "completeness",
        "score": 0.75,
        "passed": true,
        "detail": null
    }))
    .unwrap();
    assert_eq!(legacy.status, MetricStatus::Pass);
    assert_eq!(legacy.effective_score(), Some(0.75));

    let quality = QualityResult {
        confidence: QualityConfidence::Normal,
        score: None,
        aggregation: "per_metric".to_string(),
        sample_count: 2,
        window: Some(EvaluationWindow::between(
            Some("2026-01-01T00:00:00Z".to_string()),
            None,
        )),
        per_metric: BTreeMap::from([("accuracy".to_string(), scored.clone())]),
    };
    assert_eq!(round_trip(&quality), quality);
    assert_eq!(QualityResult::default().aggregation, "per_metric");

    let input = EvaluationInput::new(complete_capability())
        .with_target("judge-model")
        .with_trace(EvaluationTrace::from_cases(vec![EvaluationCase::new(
            "c", "q",
        )]))
        .with_window(EvaluationWindow::between(
            Some("2026-01-01T00:00:00Z".to_string()),
            None,
        ));
    assert_eq!(round_trip(&input), input);

    // The pre-trace JSON shape still deserializes.
    let legacy_input: EvaluationInput = serde_json::from_value(json!({
        "capability": {"capability_id": "sum", "capability_type": "tool", "name": "Sum"},
        "target": null
    }))
    .unwrap();
    assert_eq!(legacy_input.capability.capability_id, "sum");
    assert_eq!(legacy_input.trace, None);
    assert_eq!(legacy_input.window, None);
    assert_eq!(legacy_input.target, None);
}
