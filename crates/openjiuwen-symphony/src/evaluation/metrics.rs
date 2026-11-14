//! The card metrics: static conformance checks and judge-backed quality checks.
//!
//! Ports the static evaluators of Python `openjiuwen.symphony.evaluation.evaluators`
//! (`CompletenessEvaluator`, `StructureConformanceEvaluator`,
//! `DescriptionQualityEvaluator`, `ClassificationConsistencyEvaluator`) plus the
//! `LLMJudgeEvaluator` seam from `evaluation/base.py`. The evidence-driven evaluators
//! live in [`trace_metrics`](crate::evaluation::trace_metrics).
//!
//! | Rust metric | Python evaluator | Scope | Behaviour |
//! | --- | --- | --- | --- |
//! | [`CompletenessMetric`](crate::evaluation::CompletenessMetric) | `CompletenessEvaluator` | static | Rust static substitute: description, tags, described IO |
//! | [`SchemaValidityMetric`](crate::evaluation::SchemaValidityMetric) | — | static | blank/duplicate names and incoherent `required` flags |
//! | [`StructureConformanceMetric`](crate::evaluation::StructureConformanceMetric) | `StructureConformanceEvaluator` | static | the seven-field fingerprint contract |
//! | [`DescriptionQualityMetric`](crate::evaluation::DescriptionQualityMetric) | `DescriptionQualityEvaluator` | static | empty-description guard, then the judge |
//! | [`ClassificationConsistencyMetric`](crate::evaluation::ClassificationConsistencyMetric) | `ClassificationConsistencyEvaluator` | static | empty classification/tags guard, then the judge |
//! | [`LlmJudgedMetric`](crate::evaluation::LlmJudgedMetric) | `LLMJudgeEvaluator` | static | card quality, delegated to [`LlmJudgeMetric`](crate::llm::LlmJudgeMetric) |
//!
//! Every judge-backed metric runs with
//! [`FallbackMode::Deterministic`](crate::llm::FallbackMode), so an offline suite always
//! produces a verdict instead of an error.
use std::sync::Arc;

use async_trait::async_trait;

use crate::error::Result;
use crate::llm::judge::{LlmJudgeMetric, DEFAULT_RUBRIC};
use crate::llm::{CompletionModel, FallbackMode};
use crate::model::CapabilityIo;

use super::{EvaluationInput, Metric, MetricScope, MetricScore, MetricStatus};

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
/// The metric name of [`StructureConformanceMetric`].
const STRUCTURE_CONFORMANCE: &str = "structure_conformance";
/// The metric name of [`DescriptionQualityMetric`].
const DESCRIPTION_QUALITY: &str = "description_quality";
/// The metric name of [`ClassificationConsistencyMetric`].
const CLASSIFICATION_CONSISTENCY: &str = "classification_consistency";
/// The fields Python's `StructureConformanceEvaluator` inspects.
const STRUCTURE_CHECKS: [&str; 7] = [
    "capability_id",
    "capability_type",
    "name",
    "content_hash",
    "inputs",
    "outputs",
    "tags",
];
/// The rubric of [`DescriptionQualityMetric`] (Python's `DescriptionQualityEvaluator`).
const DESCRIPTION_RUBRIC: &str = "Judge whether the capability description clearly states what \
the capability does, when it should be used, and material limits without unsupported claims.";
/// The rubric of [`ClassificationConsistencyMetric`].
const CLASSIFICATION_RUBRIC: &str = "Judge whether classification and tags are semantically \
consistent with the capability description, profile, inputs, and outputs.";

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
        Ok(
            MetricScore::scored(COMPLETENESS, score, score >= COMPLETENESS_PASS_THRESHOLD)
                .with_optional_detail(detail),
        )
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
        Ok(
            MetricScore::scored(SCHEMA_VALIDITY, score, passed_checks == SCHEMA_CHECKS)
                .with_optional_detail(detail),
        )
    }
}

/// Checks Python's `StructureConformanceEvaluator` contract on a capability card.
///
/// The seven inspected fields are `capability_id`, `capability_type`, `name`,
/// `content_hash`, `inputs`, `outputs`, and `tags`:
///
/// - the four identity strings must be non-blank;
/// - `inputs`, `outputs`, and `tags` are sequences by construction in Rust, so those
///   checks always hold (they stay in the reported list for parity);
/// - every input and output name must be non-blank, and names must be unique within a
///   direction (Python's `elif`: uniqueness is only checked when no name is blank).
///
/// The score is `1.0` (`Pass`) or `0.0` (`Fail`) — Python treats the contract as
/// all-or-nothing. A descriptor built by hand should call
/// [`compute_content_hash`](crate::model::CapabilityDescriptor::compute_content_hash)
/// first, otherwise the `content_hash` check fails.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StructureConformanceMetric;

impl StructureConformanceMetric {
    /// Builds the metric; it is stateless.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Metric for StructureConformanceMetric {
    fn name(&self) -> &str {
        STRUCTURE_CONFORMANCE
    }

    fn scope(&self) -> MetricScope {
        MetricScope::Static
    }

    async fn evaluate(&self, input: &EvaluationInput) -> Result<MetricScore> {
        let capability = &input.capability;
        let mut issues: Vec<String> = Vec::new();
        let identity = [
            ("capability_id", capability.capability_id.as_str()),
            ("capability_type", capability.capability_type.as_str()),
            ("name", capability.name.as_str()),
            ("content_hash", capability.content_hash.as_str()),
        ];
        for (field, value) in identity {
            if value.trim().is_empty() {
                issues.push(format!("{field}: must be a non-empty string"));
            }
        }
        let directions = [
            ("inputs", capability.inputs.as_slice()),
            ("outputs", capability.outputs.as_slice()),
        ];
        for (direction, ios) in directions {
            if ios.iter().any(|io| io.name.trim().is_empty()) {
                issues.push(format!("{direction}: entries must have non-empty names"));
                continue;
            }
            let mut seen: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
            if ios.iter().any(|io| !seen.insert(io.name.trim())) {
                issues.push(format!("{direction}: names must be unique"));
            }
        }
        if issues.is_empty() {
            return Ok(MetricScore::scored(STRUCTURE_CONFORMANCE, 1.0, true)
                .with_detail(format!("checked: {}", STRUCTURE_CHECKS.join(", "))));
        }
        Ok(
            MetricScore::scored(STRUCTURE_CONFORMANCE, 0.0, false).with_detail(format!(
                "structural issues ({}): {}",
                issues.len(),
                issues.join("; ")
            )),
        )
    }
}

/// Judges description quality, with Python's empty-description fail guard.
///
/// A blank description is a deterministic `Fail` with score `0.0`; otherwise the card
/// goes to [`LlmJudgeMetric`], which uses [`DESCRIPTION_RUBRIC`] and falls back to the
/// static rubric when no model is configured.
#[derive(Debug, Clone)]
pub struct DescriptionQualityMetric {
    /// The shared judge implementation.
    judge: LlmJudgeMetric,
}

impl Default for DescriptionQualityMetric {
    fn default() -> Self {
        Self::new()
    }
}

impl DescriptionQualityMetric {
    /// Builds the metric with no model (deterministic fallback only).
    #[must_use]
    pub fn new() -> Self {
        Self {
            judge: LlmJudgeMetric::new(DESCRIPTION_QUALITY, DESCRIPTION_RUBRIC)
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
impl Metric for DescriptionQualityMetric {
    fn name(&self) -> &str {
        DESCRIPTION_QUALITY
    }

    fn scope(&self) -> MetricScope {
        MetricScope::Static
    }

    async fn evaluate(&self, input: &EvaluationInput) -> Result<MetricScore> {
        if input.capability.description.trim().is_empty() {
            return Ok(MetricScore::failed(
                DESCRIPTION_QUALITY,
                "the capability description is empty",
            ));
        }
        self.judge.evaluate(input).await.map(normalize_status)
    }
}

/// Judges classification/tag consistency, with Python's empty-metadata fail guard.
///
/// A card with neither a non-blank classification nor any tag is a deterministic
/// `Fail`; otherwise the card goes to [`LlmJudgeMetric`] with
/// [`CLASSIFICATION_RUBRIC`].
#[derive(Debug, Clone)]
pub struct ClassificationConsistencyMetric {
    /// The shared judge implementation.
    judge: LlmJudgeMetric,
}

impl Default for ClassificationConsistencyMetric {
    fn default() -> Self {
        Self::new()
    }
}

impl ClassificationConsistencyMetric {
    /// Builds the metric with no model (deterministic fallback only).
    #[must_use]
    pub fn new() -> Self {
        Self {
            judge: LlmJudgeMetric::new(CLASSIFICATION_CONSISTENCY, CLASSIFICATION_RUBRIC)
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
impl Metric for ClassificationConsistencyMetric {
    fn name(&self) -> &str {
        CLASSIFICATION_CONSISTENCY
    }

    fn scope(&self) -> MetricScope {
        MetricScope::Static
    }

    async fn evaluate(&self, input: &EvaluationInput) -> Result<MetricScore> {
        let capability = &input.capability;
        if capability.classification.trim().is_empty() && capability.tags.is_empty() {
            return Ok(MetricScore::failed(
                CLASSIFICATION_CONSISTENCY,
                "the capability has neither a classification nor tags",
            ));
        }
        self.judge.evaluate(input).await.map(normalize_status)
    }
}

/// LLM-judged quality metric, delegated to [`LlmJudgeMetric`].
///
/// Mirrors Python's `LLMJudgeEvaluator` in `openjiuwen.symphony.evaluation`, which asks a
/// model for a binary verdict plus a reason. This type is the evaluation surface's name
/// for that judgement: it wraps the crate's single judge implementation, keeping the
/// metric name `"llm_judged"` and
/// [`DEFAULT_RUBRIC`].
///
/// With no model it scores the card with the deterministic fallback rubric (description,
/// described input, described output — equal weight, passing at `0.5`, `detail` prefixed
/// `deterministic_fallback:`), so an offline suite still gets a verdict. With a model it
/// delegates the request, the single repair retry, and the
/// [`FallbackMode`] handling to the inner metric.
///
/// ```
/// use openjiuwen_symphony::evaluation::{EvaluationSuite, LlmJudgedMetric, Metric};
///
/// let metric = LlmJudgedMetric::new();
/// assert_eq!(metric.name(), "llm_judged");
/// assert!(!metric.has_model());
/// assert!(metric.rubric().starts_with("The capability card must be self-describing"));
/// let _suite = EvaluationSuite::default_suite().with_llm_judge();
/// ```
#[derive(Debug, Clone)]
pub struct LlmJudgedMetric {
    /// The shared judge implementation this metric delegates to.
    inner: LlmJudgeMetric,
}

impl Default for LlmJudgedMetric {
    fn default() -> Self {
        Self::new()
    }
}

impl LlmJudgedMetric {
    /// Builds the metric with the `"llm_judged"` name and the default rubric, no model.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: LlmJudgeMetric::new(LLM_JUDGED, DEFAULT_RUBRIC),
        }
    }

    /// Sets the completion model used for judgement.
    #[must_use]
    pub fn with_model(mut self, model: Arc<dyn CompletionModel>) -> Self {
        self.inner = self.inner.with_model(model);
        self
    }

    /// Replaces the rubric the model is asked to apply.
    #[must_use]
    pub fn with_rubric(mut self, rubric: impl Into<String>) -> Self {
        self.inner = self.inner.with_rubric(rubric);
        self
    }

    /// Sets what happens when the model fails or keeps answering unusably.
    #[must_use]
    pub fn with_fallback(mut self, fallback: FallbackMode) -> Self {
        self.inner = self.inner.with_fallback(fallback);
        self
    }

    /// The rubric the model is asked to apply.
    #[must_use]
    pub fn rubric(&self) -> &str {
        self.inner.rubric()
    }

    /// Whether a model is configured.
    #[must_use]
    pub fn has_model(&self) -> bool {
        self.inner.has_model()
    }
}

#[async_trait]
impl Metric for LlmJudgedMetric {
    fn name(&self) -> &str {
        self.inner.name()
    }

    /// Delegates to [`LlmJudgeMetric`].
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::Error::Llm`] only when a model is configured,
    /// the judgement fails twice, and the fallback mode is
    /// [`FallbackMode::Error`]. Without a model the
    /// deterministic rubric always answers.
    async fn evaluate(&self, input: &EvaluationInput) -> Result<MetricScore> {
        self.inner.evaluate(input).await.map(normalize_status)
    }
}

/// Forces a judge-produced score to carry the status its `passed` flag implies.
pub(super) fn normalize_status(mut score: MetricScore) -> MetricScore {
    score.status = if score.passed {
        MetricStatus::Pass
    } else {
        MetricStatus::Fail
    };
    score
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

/// The capability card every metric test starts from.
#[cfg(test)]
pub(super) fn complete_capability() -> crate::model::CapabilityDescriptor {
    let card = crate::model::CapabilityDescriptor::new("tool", "sum", "Sum")
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

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::assert_is_empty
)]
mod tests {
    use super::*;

    /// The same card, stripped of its vocabulary and with duplicate input names.
    fn broken_capability() -> crate::model::CapabilityDescriptor {
        crate::model::CapabilityDescriptor::new("tool", "broken", "Broken")
            .with_description("   ")
            .with_inputs(vec![
                CapabilityIo::new("dup", "number"),
                CapabilityIo::new("dup", "number"),
                CapabilityIo::new("  ", "string"),
            ])
            .with_outputs(vec![CapabilityIo::new("", "string")])
    }

    fn input(capability: crate::model::CapabilityDescriptor) -> EvaluationInput {
        EvaluationInput::new(capability)
    }

    #[tokio::test]
    async fn completeness_rubric_weighs_four_checks_equally() {
        let metric = CompletenessMetric::new();
        let score = metric
            .evaluate(&input(complete_capability()))
            .await
            .unwrap();
        assert_eq!(score.score, Some(1.0));
        assert_eq!(score.status, MetricStatus::Pass);
        let no_tags = complete_capability().with_tags(Vec::<String>::new());
        let score = metric.evaluate(&input(no_tags)).await.unwrap();
        assert_eq!(score.score, Some(0.75));
        assert!(score.passed);
        assert_eq!(score.detail.as_deref(), Some("missing: tags"));
        let broken = metric.evaluate(&input(broken_capability())).await.unwrap();
        assert_eq!(broken.score, Some(0.0));
        assert_eq!(broken.status, MetricStatus::Fail);
        assert_eq!(
            broken.detail.as_deref(),
            Some("missing: description, tags, input descriptions, output descriptions")
        );
    }

    #[tokio::test]
    async fn schema_validity_reports_blank_duplicate_and_incoherent_io() {
        let metric = SchemaValidityMetric::new();
        let broken = metric.evaluate(&input(broken_capability())).await.unwrap();
        assert_eq!(broken.score, Some(1.0 / 3.0));
        assert!(!broken.passed);
        assert_eq!(broken.status, MetricStatus::Fail);
        let detail = broken.detail.as_deref().unwrap();
        assert!(detail.contains("blank input name at index 2"), "{detail}");
        assert!(detail.contains("blank output name at index 0"), "{detail}");
        assert!(detail.contains("duplicate input name `dup`"), "{detail}");

        let capability = complete_capability().with_inputs(vec![CapabilityIo::new("a", "number")
            .with_description("first addend")
            .with_required(true)
            .with_default(serde_json::json!(0))]);
        let score = metric.evaluate(&input(capability)).await.unwrap();
        assert_eq!(score.score, Some(2.0 / 3.0));
        assert_eq!(
            score.detail.as_deref(),
            Some("input `a` is required but declares a default")
        );
    }

    #[tokio::test]
    async fn structure_conformance_accepts_a_complete_card() {
        let score = StructureConformanceMetric::new()
            .evaluate(&input(complete_capability()))
            .await
            .unwrap();
        assert_eq!(score.score, Some(1.0));
        assert_eq!(score.status, MetricStatus::Pass);
        let detail = score.detail.as_deref().unwrap();
        assert!(detail.starts_with("checked: "), "{detail}");
        assert!(detail.contains("content_hash"), "{detail}");
        assert!(detail.contains("tags"), "{detail}");
    }

    #[tokio::test]
    async fn structure_conformance_names_every_structural_issue() {
        let score = StructureConformanceMetric::new()
            .evaluate(&input(broken_capability()))
            .await
            .unwrap();
        assert_eq!(score.score, Some(0.0));
        assert_eq!(score.status, MetricStatus::Fail);
        let detail = score.detail.as_deref().unwrap();
        assert!(
            detail.contains("content_hash: must be a non-empty string"),
            "{detail}"
        );
        assert!(
            detail.contains("inputs: entries must have non-empty names"),
            "{detail}"
        );
        assert!(
            detail.contains("outputs: entries must have non-empty names"),
            "{detail}"
        );
        assert!(!detail.contains("names must be unique"), "{detail}");
    }

    #[tokio::test]
    async fn structure_conformance_rejects_duplicate_names() {
        let card = complete_capability().with_outputs(vec![
            CapabilityIo::new("total", "number"),
            CapabilityIo::new("total", "number"),
        ]);
        let score = StructureConformanceMetric::new()
            .evaluate(&input(card))
            .await
            .unwrap();
        assert_eq!(score.status, MetricStatus::Fail);
        assert_eq!(
            score.detail.as_deref(),
            Some("structural issues (1): outputs: names must be unique")
        );
    }

    #[tokio::test]
    async fn description_and_classification_guards_fail_without_a_model() {
        let score = DescriptionQualityMetric::new()
            .evaluate(&input(broken_capability()))
            .await
            .unwrap();
        assert_eq!(score.score, Some(0.0));
        assert_eq!(score.status, MetricStatus::Fail);
        assert_eq!(
            score.detail.as_deref(),
            Some("the capability description is empty")
        );

        let classified = complete_capability().with_tags(Vec::<String>::new());
        let score = ClassificationConsistencyMetric::new()
            .evaluate(&input(classified))
            .await
            .unwrap();
        assert_eq!(score.score, Some(0.0));
        assert_eq!(score.status, MetricStatus::Fail);
        assert_eq!(
            score.detail.as_deref(),
            Some("the capability has neither a classification nor tags")
        );

        let score = ClassificationConsistencyMetric::new()
            .evaluate(&input(complete_capability()))
            .await
            .unwrap();
        assert!(score.passed);
        assert_eq!(score.status, MetricStatus::Pass);
    }

    #[test]
    fn the_wrapper_keeps_its_name_and_builders() {
        let metric = LlmJudgedMetric::new();
        assert_eq!(metric.name(), "llm_judged");
        assert_eq!(metric.rubric(), DEFAULT_RUBRIC);
        assert!(!metric.has_model());
        assert_eq!(LlmJudgedMetric::default().rubric(), DEFAULT_RUBRIC);

        let custom = LlmJudgedMetric::new()
            .with_rubric("must cite an example")
            .with_fallback(FallbackMode::Error);
        assert_eq!(custom.name(), "llm_judged");
        assert_eq!(custom.rubric(), "must cite an example");
        assert!(!custom.has_model());
    }
}
