//! Fail-soft capability evaluation with deterministic static metrics.
//!
//! Ports the offline subset of Python `openjiuwen.symphony.evaluation`. Python's suite
//! mixes deterministic static checks with LLM-judged and trace-driven metrics; the
//! Rust port keeps the static metrics and defers the model-backed ones to
//! [`LlmJudgedMetric`], which reports [`openjiuwen_common::error::Error::Unsupported`].
//!
//! # Shapes
//!
//! [`EvaluationInput`] is what a metric inspects — one [`CapabilityDescriptor`] plus
//! an optional judge target. A [`Metric`] produces one [`MetricScore`] in `[0.0, 1.0]`,
//! and [`EvaluationSuite`] runs every metric in order, collecting the scores into an
//! [`EvaluationReport`].
//!
//! # Fail-soft contract
//!
//! A metric that fails does not abort the suite: the error is rendered into
//! [`EvaluationReport::errors`] (prefixed with the metric name) and the remaining
//! metrics still run. That mirrors Python's per-evaluator isolation, and it keeps a
//! deferred or misconfigured metric from hiding the static results. `overall` is the
//! mean of the *successful* scores, so `0.0` means "nothing could be scored", not
//! "everything failed".
//!
//! # Determinism
//!
//! Both built-in metrics are pure functions of the card: no clock, no LLM, no
//! hash-map iteration order. Scores stay in suite order, and `detail` strings list the
//! offending fields in card order.
#![allow(clippy::unused_async_trait_impl)] // the `Metric` contract is async for LLM-backed metrics
use crate::error::{unsupported, Result};
use crate::model::{CapabilityDescriptor, CapabilityIo};
use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
/// The feature name reported by [`LlmJudgedMetric`].
const LLM_JUDGE_FEATURE: &str = "LLM-judged capability evaluation";
/// How many checks [`CompletenessMetric`] performs (equal weight each).
const COMPLETENESS_CHECKS: u32 = 4;
/// The [`CompletenessMetric`] score at or above which a capability counts as complete.
const COMPLETENESS_PASS_THRESHOLD: f64 = 0.75;
/// How many checks [`SchemaValidityMetric`] performs (equal weight each).
const SCHEMA_CHECKS: u32 = 3;
/// The metric name of [`CompletenessMetric`].
const COMPLETENESS: &str = "completeness";
/// The metric name of [`SchemaValidityMetric`].
const SCHEMA_VALIDITY: &str = "schema_validity";
/// The metric name of [`LlmJudgedMetric`].
const LLM_JUDGED: &str = "llm_judged";
/// The subject of one metric evaluation.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct EvaluationInput {
    /// The capability card under evaluation.
    pub capability: CapabilityDescriptor,
    /// Optional judge target, for example the model a deferred metric would call.
    pub target: Option<String>,
}
/// One metric's verdict on a capability.
///
/// `score` is normalized to `[0.0, 1.0]`, `passed` is the metric's own threshold
/// decision, and `detail` explains a partial or failing result (it is `None` when the
/// metric is fully satisfied).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct MetricScore {
    /// The metric name, matching [`Metric::name`].
    pub metric: String,
    /// The normalized score in `[0.0, 1.0]`.
    pub score: f64,
    /// Whether the score meets the metric's pass threshold.
    pub passed: bool,
    /// Human-readable explanation of missing pieces or offending names.
    pub detail: Option<String>,
}
/// The fail-soft result of evaluating one capability.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct EvaluationReport {
    /// The id of the evaluated capability.
    pub capability_id: String,
    /// The successful metric scores, in suite order.
    pub scores: Vec<MetricScore>,
    /// The mean of the successful scores; `0.0` when there are none.
    pub overall: f64,
    /// Metric failures, rendered as `"<metric>: <error>"`, in suite order.
    pub errors: Vec<String>,
}
/// One deterministic or model-backed quality check over a capability card.
///
/// Implementations must be cheap to share (`Send + Sync`, kept behind an [`Arc`]) and
/// must not panic on malformed cards: an unusable card is a low score or an
/// [`openjiuwen_common::error::Error`], never a crash.
#[async_trait]
pub trait Metric: Send + Sync {
    /// The metric name recorded in [`MetricScore::metric`].
    fn name(&self) -> &str;
    /// Scores `input`, returning the verdict or the reason it could not be computed.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::Error`] when the metric is unavailable, for
    /// example [`openjiuwen_common::error::Error::Unsupported`] from a deferred
    /// model-backed metric. The suite records the error instead of aborting.
    async fn evaluate(&self, input: &EvaluationInput) -> Result<MetricScore>;
}
/// Scores vocabulary completeness: description, tags, and described inputs/outputs.
///
/// The rubric is four equally weighted checks (`0.25` each), evaluated in this order:
///
/// 1. `description` is present and non-blank.
/// 2. `tags` contains at least one non-blank tag.
/// 3. At least one input carries a non-blank description.
/// 4. At least one output carries a non-blank description.
///
/// The score is `checks_passed / 4`, and the capability passes when the score is at
/// least `0.75` (three of four checks). `detail` lists the missing pieces.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CompletenessMetric;
impl CompletenessMetric {
    /// Builds the metric; it is stateless.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}
#[async_trait]
impl Metric for CompletenessMetric {
    fn name(&self) -> &str {
        COMPLETENESS
    }
    async fn evaluate(&self, input: &EvaluationInput) -> Result<MetricScore> {
        let capability = &input.capability;
        let mut missing: Vec<&str> = Vec::new();
        if capability.description.trim().is_empty() {
            missing.push("description");
        }
        if capability.tags.iter().all(|tag| tag.trim().is_empty()) {
            missing.push("tags");
        }
        if !has_described_io(&capability.inputs) {
            missing.push("input descriptions");
        }
        if !has_described_io(&capability.outputs) {
            missing.push("output descriptions");
        }
        let failed = u32::try_from(missing.len()).unwrap_or(COMPLETENESS_CHECKS);
        let score = f64::from(COMPLETENESS_CHECKS - failed) / f64::from(COMPLETENESS_CHECKS);
        let detail = if missing.is_empty() {
            None
        } else {
            Some(format!("missing: {}", missing.join(", ")))
        };
        Ok(MetricScore {
            metric: COMPLETENESS.to_string(),
            score,
            passed: score >= COMPLETENESS_PASS_THRESHOLD,
            detail,
        })
    }
}
/// Scores input/output schema validity with three equally weighted checks.
///
/// The checks are:
///
/// 1. Every IO name is non-blank.
/// 2. Names are unique within the inputs and within the outputs.
/// 3. `required` flags are coherent (a required IO must not declare a default).
///
/// The score is `checks_passed / 3`; the metric passes only when all three hold, since
/// a malformed schema cannot be "mostly valid". `detail` lists the offending IO names
/// (and their position, for blank names, which have no name to quote).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SchemaValidityMetric;
impl SchemaValidityMetric {
    /// Builds the metric; it is stateless.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}
#[async_trait]
impl Metric for SchemaValidityMetric {
    fn name(&self) -> &str {
        SCHEMA_VALIDITY
    }
    async fn evaluate(&self, input: &EvaluationInput) -> Result<MetricScore> {
        let capability = &input.capability;
        let mut issues: Vec<String> = Vec::new();
        let mut failed = 0_u32;
        let blank_issues = blank_name_issues(&capability.inputs, "input");
        let blank_output_issues = blank_name_issues(&capability.outputs, "output");
        let blank_ok = blank_issues.is_empty() && blank_output_issues.is_empty();
        if !blank_ok {
            failed += 1;
        }
        issues.extend(blank_issues);
        issues.extend(blank_output_issues);
        let mut duplicate_issues = duplicate_name_issues(&capability.inputs, "input");
        duplicate_issues.extend(duplicate_name_issues(&capability.outputs, "output"));
        if !duplicate_issues.is_empty() {
            failed += 1;
        }
        issues.extend(duplicate_issues);
        let mut required_issues = incoherent_required_issues(&capability.inputs, "input");
        required_issues.extend(incoherent_required_issues(&capability.outputs, "output"));
        if !required_issues.is_empty() {
            failed += 1;
        }
        issues.extend(required_issues);
        let passed_checks = SCHEMA_CHECKS - failed;
        let score = f64::from(passed_checks) / f64::from(SCHEMA_CHECKS);
        let detail = if issues.is_empty() {
            None
        } else {
            Some(issues.join("; "))
        };
        Ok(MetricScore {
            metric: SCHEMA_VALIDITY.to_string(),
            score,
            passed: passed_checks == SCHEMA_CHECKS,
            detail,
        })
    }
}
/// LLM-judged quality metric, deferred to a model-backed implementation.
///
/// Mirrors Python's `LLMJudgeEvaluator` in `openjiuwen.symphony.evaluation`, which asks
/// a model to judge capability quality. That needs an LLM stack, so this metric always
/// returns [`openjiuwen_common::error::Error::Unsupported`]; the suite records the
/// failure and still reports every static score.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LlmJudgedMetric;
impl LlmJudgedMetric {
    /// Builds the metric; it is stateless.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}
#[async_trait]
impl Metric for LlmJudgedMetric {
    fn name(&self) -> &str {
        LLM_JUDGED
    }
    /// Always fails with [`openjiuwen_common::error::Error::Unsupported`].
    ///
    /// # Errors
    ///
    /// Always returns [`openjiuwen_common::error::Error::Unsupported`]: LLM judgement is
    /// part of the crate's deferred scope.
    async fn evaluate(&self, _input: &EvaluationInput) -> Result<MetricScore> {
        Err(unsupported(LLM_JUDGE_FEATURE))
    }
}
/// An ordered set of metrics evaluated fail-soft against one capability.
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
    /// Appends the deferred [`LlmJudgedMetric`].
    ///
    /// Every evaluation then records exactly one fail-soft error; the static metrics
    /// still score the capability.
    #[must_use]
    pub fn with_llm_judge(mut self) -> Self {
        self.metrics.push(Arc::new(LlmJudgedMetric));
        self
    }
    /// The configured metrics, in evaluation order.
    #[must_use]
    pub fn metrics(&self) -> &[Arc<dyn Metric>] {
        &self.metrics
    }
    /// Evaluates `input` with every metric, collecting scores and failures.
    ///
    /// A metric error is recorded in [`EvaluationReport::errors`] as
    /// `"<metric>: <message>"` and does not stop the run. `overall` is the mean of the
    /// successful scores, or `0.0` when no metric succeeded.
    #[allow(clippy::cast_precision_loss)] // metric counts are far below 2^53
    pub async fn evaluate(&self, input: &EvaluationInput) -> EvaluationReport {
        let mut scores = Vec::with_capacity(self.metrics.len());
        let mut errors = Vec::new();
        for metric in &self.metrics {
            match metric.evaluate(input).await {
                Ok(score) => scores.push(score),
                Err(error) => errors.push(format!("{}: {}", metric.name(), error.message())),
            }
        }
        let overall = if scores.is_empty() {
            0.0
        } else {
            let total: f64 = scores.iter().map(|score| score.score).sum();
            total / scores.len() as f64
        };
        EvaluationReport {
            capability_id: input.capability.capability_id.clone(),
            scores,
            overall,
            errors,
        }
    }
}
/// Reports whether at least one IO carries a non-blank description.
fn has_described_io(ios: &[CapabilityIo]) -> bool {
    ios.iter().any(|io| !io.description.trim().is_empty())
}

/// Describes every blank IO name, naming its direction and position.
fn blank_name_issues(ios: &[CapabilityIo], direction: &str) -> Vec<String> {
    ios.iter()
        .enumerate()
        .filter(|(_, io)| io.name.trim().is_empty())
        .map(|(index, _)| format!("blank {direction} name at index {index}"))
        .collect()
}

/// Describes every IO name that appears more than once in `ios`, sorted by name.
fn duplicate_name_issues(ios: &[CapabilityIo], direction: &str) -> Vec<String> {
    use std::collections::{BTreeSet, HashSet};
    let mut seen: HashSet<&str> = HashSet::new();
    let mut duplicates: BTreeSet<&str> = BTreeSet::new();
    for io in ios {
        let name = io.name.trim();
        if name.is_empty() {
            continue;
        }
        if !seen.insert(name) {
            duplicates.insert(name);
        }
    }
    duplicates
        .into_iter()
        .map(|name| format!("duplicate {direction} name `{name}`"))
        .collect()
}

/// Describes every IO marked required that also declares a default.
fn incoherent_required_issues(ios: &[CapabilityIo], direction: &str) -> Vec<String> {
    ios.iter()
        .filter(|io| io.required == Some(true) && io.default.is_some())
        .map(|io| {
            format!(
                "{direction} `{}` is required but declares a default",
                io.name.trim()
            )
        })
        .collect()
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
    use serde_json::json;
    /// A card that satisfies every static check.
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
        EvaluationInput {
            capability,
            target: None,
        }
    }

    #[tokio::test]
    async fn default_suite_scores_a_complete_capability_above_threshold() {
        let report = EvaluationSuite::default_suite()
            .evaluate(&input(complete_capability()))
            .await;
        assert_eq!(report.capability_id, "sum");
        assert!(report.errors.is_empty());
        let names: Vec<&str> = report.scores.iter().map(|s| s.metric.as_str()).collect();
        assert_eq!(names, vec![COMPLETENESS, SCHEMA_VALIDITY]);
        for score in &report.scores {
            assert_eq!(score.score, 1.0, "{}: {:?}", score.metric, score.detail);
            assert!(score.passed);
            assert!(score.detail.is_none());
        }
        assert_eq!(report.overall, 1.0);
    }

    #[tokio::test]
    async fn completeness_rubric_weighs_four_checks_equally() {
        let metric = CompletenessMetric::new();
        let score = metric
            .evaluate(&input(complete_capability()))
            .await
            .unwrap();
        assert_eq!(score.score, 1.0);
        let no_tags = complete_capability().with_tags(Vec::<String>::new());
        let score = metric.evaluate(&input(no_tags)).await.unwrap();
        assert_eq!(score.score, 0.75);
        assert!(score.passed);
        assert_eq!(score.detail.as_deref(), Some("missing: tags"));
        let broken = metric.evaluate(&input(broken_capability())).await.unwrap();
        assert_eq!(broken.score, 0.0);
        assert!(!broken.passed);
        assert_eq!(
            broken.detail.as_deref(),
            Some("missing: description, tags, input descriptions, output descriptions")
        );
    }

    #[tokio::test]
    async fn blank_description_and_duplicate_names_fail_the_static_metrics() {
        let report = EvaluationSuite::default_suite()
            .evaluate(&input(broken_capability()))
            .await;
        let completeness = &report.scores[0];
        assert!(!completeness.passed);
        assert_eq!(completeness.score, 0.0);
        let schema = &report.scores[1];
        assert!(!schema.passed);
        // Blank and duplicate names fail two of the three checks; the required flags
        // are coherent in this card, so that check still passes.
        assert_eq!(schema.score, 1.0 / 3.0);
        let detail = schema.detail.as_deref().unwrap();
        assert!(detail.contains("blank input name at index 2"), "{detail}");
        assert!(detail.contains("blank output name at index 0"), "{detail}");
        assert!(detail.contains("duplicate input name `dup`"), "{detail}");
        assert_eq!(report.overall, f64::midpoint(0.0, 1.0 / 3.0));
        assert!(report.errors.is_empty());
    }

    #[tokio::test]
    async fn required_io_with_a_default_is_incoherent() {
        let capability = complete_capability().with_inputs(vec![CapabilityIo::new("a", "number")
            .with_description("first addend")
            .with_required(true)
            .with_default(json!(0))]);
        let score = SchemaValidityMetric::new()
            .evaluate(&input(capability))
            .await
            .unwrap();
        assert_eq!(score.score, 2.0 / 3.0);
        assert!(!score.passed);
        assert_eq!(
            score.detail.as_deref(),
            Some("input `a` is required but declares a default")
        );
    }

    #[tokio::test]
    async fn with_llm_judge_records_one_fail_soft_error() {
        let suite = EvaluationSuite::default_suite().with_llm_judge();
        assert_eq!(suite.metrics().len(), 3);
        let report = suite.evaluate(&input(complete_capability())).await;
        assert_eq!(report.errors.len(), 1);
        let error = &report.errors[0];
        assert!(error.starts_with(LLM_JUDGED), "{error}");
        assert!(error.contains(LLM_JUDGE_FEATURE), "{error}");
        assert_eq!(report.scores.len(), 2);
        assert!(report.scores.iter().all(|score| score.passed));
        assert_eq!(report.overall, 1.0);
    }

    #[tokio::test]
    async fn overall_is_the_mean_of_the_successful_scores() {
        let report = EvaluationSuite::default_suite()
            .evaluate(&input(broken_capability().with_tags(["tools"])))
            .await;
        assert_eq!(report.scores.len(), 2);
        // Completeness: only `tags` survives, so one of four checks passes.
        assert_eq!(report.scores[0].score, 0.25);
        // Schema validity: blank and duplicate names fail, coherent required flags pass.
        assert_eq!(report.scores[1].score, 1.0 / 3.0);
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

    #[test]
    fn reports_round_trip_through_json() {
        let report = EvaluationReport {
            capability_id: "tool:sum".to_string(),
            scores: vec![MetricScore {
                metric: COMPLETENESS.to_string(),
                score: 0.75,
                passed: true,
                detail: Some("missing: tags".to_string()),
            }],
            overall: 0.75,
            errors: vec!["llm_judged: unsupported".to_string()],
        };
        let encoded = serde_json::to_value(&report).unwrap();
        assert_eq!(encoded["capability_id"], "tool:sum");
        assert_eq!(encoded["scores"][0]["metric"], COMPLETENESS);
        let decoded: EvaluationReport = serde_json::from_value(encoded).unwrap();
        assert_eq!(decoded, report);
    }
}
