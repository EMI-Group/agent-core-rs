//! Behaviour tests for [`openjiuwen_symphony::evaluation`].
//!
//! Two layers are covered: the deterministic static suite (completeness and schema
//! validity) and the LLM-judged metric, which is a thin delegating wrapper over
//! [`openjiuwen_symphony::llm::LlmJudgeMetric`]. The judge is exercised three times —
//! offline (the documented deterministic rubric), with a scripted model that answers a
//! valid verdict, and with a model that answers garbage twice — so the delegation, the
//! single repair retry and both [`FallbackMode`] branches are all asserted here.
//!
//! Nothing touches the network: the only model is a local fake, no credentials are read,
//! and no assertion depends on the clock.
#![allow(clippy::unwrap_used, clippy::expect_used)]
#![allow(clippy::float_cmp)]
#![allow(clippy::assert_is_empty)]
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use openjiuwen_symphony::evaluation::{
    CompletenessMetric, EvaluationInput, EvaluationReport, EvaluationSuite, LlmJudgedMetric,
    Metric, MetricScore, SchemaValidityMetric,
};
use openjiuwen_symphony::llm::judge::DEFAULT_RUBRIC;
use openjiuwen_symphony::llm::{CompletionModel, FallbackMode};
use openjiuwen_symphony::model::{CapabilityDescriptor, CapabilityIo};
use openjiuwen_symphony::{Error, Result};

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
    CapabilityDescriptor::new("tool", "sum", "Sum")
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
        ])
}

/// A card with no description and no described IO: the judge scores it `0.0`.
fn thin_capability() -> CapabilityDescriptor {
    CapabilityDescriptor::new("tool", "thin", "Thin")
}

/// Wraps a capability in the input a metric inspects.
fn input(capability: CapabilityDescriptor) -> EvaluationInput {
    EvaluationInput {
        capability,
        target: None,
    }
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
    assert_eq!(report.scores[0].score, 1.0);
    assert!(report.scores[0].passed);
    assert_eq!(report.scores[0].detail, None);
    assert_eq!(report.scores[1].metric, "schema_validity");
    assert_eq!(report.scores[1].score, 1.0);
    assert!(report.scores[1].passed);
    assert_eq!(report.scores[1].detail, None);
    assert_eq!(report.overall, 1.0);
}

#[tokio::test]
async fn the_static_metrics_are_reusable_as_arcs() {
    let suite = EvaluationSuite::new(vec![
        Arc::new(CompletenessMetric::new()),
        Arc::new(SchemaValidityMetric::new()),
    ]);
    assert_eq!(suite.metrics().len(), 2);
    for metric in suite.metrics() {
        let score = metric
            .evaluate(&input(complete_capability()))
            .await
            .unwrap();
        assert!(score.passed, "{}", metric.name());
    }
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
    assert_eq!(judge.score, 1.0);
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
    assert_eq!(judge.score, 0.0);
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
    assert_eq!(score.score, 1.0);
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
    assert_eq!(judge.score, 0.0);
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
    assert_eq!(judge.score, 1.0);
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
        scores: vec![MetricScore {
            metric: "llm_judged".to_string(),
            score: 0.5,
            passed: true,
            detail: Some("deterministic_fallback: partial".to_string()),
        }],
        overall: 0.5,
        errors: vec!["schema_validity: backend unavailable".to_string()],
    };
    let encoded = serde_json::to_value(&report).unwrap();
    assert_eq!(encoded["capability_id"], "tool:sum");
    assert_eq!(encoded["scores"][0]["metric"], "llm_judged");
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
