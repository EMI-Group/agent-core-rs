//! LLM-judged evaluation metrics.
//!
//! Python's `evaluation/base.py` LLM judge asks a model for a binary verdict plus a
//! reason. [`LlmJudgeMetric`] ports that metric onto the crate's
//! [`Metric`](crate::evaluation::Metric) trait and the [`CompletionModel`] seam, so it can
//! be dropped into an [`EvaluationSuite`](crate::evaluation::EvaluationSuite) — and, with
//! no model configured, into an offline suite that still produces a verdict.
//!
//! # Request and reply
//!
//! The prompt carries the metric's rubric plus the capability card as JSON. The reply
//! must be `{"score":0|1,"reason":"<non-empty>"}`: `score` has to be the *number* `0` or
//! `1` (a boolean or any other value is invalid) and `reason` has to be a non-empty
//! string. An invalid reply gets exactly one repair attempt, which re-sends the request
//! with the validation error appended; a second failure is handled by the configured
//! [`FallbackMode`].
//!
//! # The deterministic fallback
//!
//! Without a model — or with [`FallbackMode::Deterministic`] after a failed judgement —
//! the metric scores the card itself with three equal checks: a non-blank description,
//! at least one described input, and at least one described output. All three give
//! `1.0`, a description alone gives `0.5`, and a card without a description gives `0.0`.
//! The metric passes at `0.5` or above, and `detail` starts with `deterministic_fallback:`
//! so a report reader can tell the two paths apart. A model verdict passes only on
//! `score == 1`.
//!
//! # Provenance
//!
//! A bare [`MetricScore`](crate::evaluation::MetricScore) cannot tell "no model was
//! configured" from "the model was unusable twice": both carry the
//! `deterministic_fallback:` `detail`. [`LlmJudgeMetric::evaluate_with_provenance`] therefore
//! returns a [`JudgeVerdict`](crate::llm::judge::JudgeVerdict) — the score plus the
//! [`StageProvenance`] it came from:
//!
//! | Path | Provenance | Log record |
//! | --- | --- | --- |
//! | no model configured | `StageProvenance::NoModel` | one `tracing::debug!` |
//! | the model's verdict, or the single repair attempt's verdict, was usable | `StageProvenance::Model` | none |
//! | the model call failed or both replies were rejected under [`FallbackMode::Deterministic`] | `StageProvenance::DeterministicFallback` | exactly one `tracing::warn!` naming the stage and the static rubric |
//! | the model call failed or both replies were rejected under [`FallbackMode::Error`] | — | the typed `Error::Llm` is returned instead; no fallback record |
//!
//! The [`Metric`](crate::evaluation::Metric) impl is unchanged in behaviour: it answers
//! `evaluate_with_provenance(input).await.map(|verdict| verdict.score)`, so
//! [`Metric::evaluate`](crate::evaluation::Metric::evaluate) keeps returning the bare
//! [`MetricScore`](crate::evaluation::MetricScore) that
//! [`crate::evaluation::LlmJudgedMetric`] and the suites consuming it already expect —
//! they keep reading the marker through `detail`, while a caller that needs the exact
//! provenance reads it off the [`JudgeVerdict`](crate::llm::judge::JudgeVerdict).
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{Error, Result};
use crate::evaluation::{EvaluationInput, Metric, MetricScore};
use crate::model::CapabilityIo;

use super::{
    complete_json, note_fallback, note_no_model, CompletionModel, FallbackMode, StageProvenance,
};

/// The metric name a [`LlmJudgeMetric`] gets by default.
pub const DEFAULT_JUDGE_NAME: &str = "llm_judge";
/// The rubric a [`LlmJudgeMetric`] gets by default.
pub const DEFAULT_RUBRIC: &str = "The capability card must be self-describing: a non-empty \
description, and at least one input and one output that explain what they carry.";
/// The score a fallback needs to pass.
const FALLBACK_PASS_THRESHOLD: f64 = 0.5;
/// The score a model verdict needs to pass.
const LLM_PASS_THRESHOLD: f64 = 1.0;
/// The stage name carried by this metric's provenance log records.
const STAGE: &str = "judge metric";
/// The fallback rule named by this metric's provenance log records.
const FALLBACK_RULE: &str = "the static description/IO rubric";
/// The `detail` explanation recorded when no model is configured.
const NO_MODEL_EXPLANATION: &str = "no model is configured";
/// The system instruction sent with every judgement request.
const SYSTEM_PROMPT: &str = "Judge the capability card against the rubric. \
Return JSON only: {\"score\":0,\"reason\":\"…\"}. \
`score` must be the number 1 when the card satisfies the rubric and 0 when it does not; \
`reason` must be a non-empty explanation.";

/// One judge verdict plus where it came from.
///
/// [`LlmJudgeMetric::evaluate_with_provenance`] returns this; the
/// [`Metric`] trait keeps answering the bare
/// [`MetricScore`] held in [`score`](crate::llm::judge::JudgeVerdict::score).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct JudgeVerdict {
    /// The metric score, exactly as `Metric::evaluate` returns it.
    pub score: MetricScore,
    /// Where the verdict came from.
    pub provenance: StageProvenance,
}

impl JudgeVerdict {
    /// Builds a verdict with its provenance.
    #[must_use]
    pub fn new(score: MetricScore, provenance: StageProvenance) -> Self {
        Self { score, provenance }
    }

    /// Whether the verdict passes the metric's threshold.
    #[must_use]
    pub fn passed(&self) -> bool {
        self.score.passed
    }

    /// Whether a model produced the verdict.
    #[must_use]
    pub fn is_model_backed(&self) -> bool {
        self.provenance.is_model()
    }
}

/// A capability-quality metric judged by a model, with a static fallback rubric.
#[derive(Clone)]
pub struct LlmJudgeMetric {
    /// The metric name recorded in [`MetricScore::metric`].
    name: String,
    /// The rubric the model is asked to apply.
    rubric: String,
    /// The completion model; `None` keeps the static rubric.
    model: Option<Arc<dyn CompletionModel>>,
    /// What to do when the model fails or keeps answering unusably.
    fallback: FallbackMode,
}

impl Default for LlmJudgeMetric {
    fn default() -> Self {
        Self::new(DEFAULT_JUDGE_NAME, DEFAULT_RUBRIC)
    }
}

impl LlmJudgeMetric {
    /// Builds a metric with `name` and `rubric` and no model.
    #[must_use]
    pub fn new(name: impl Into<String>, rubric: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            rubric: rubric.into(),
            model: None,
            fallback: FallbackMode::Deterministic,
        }
    }

    /// Sets the completion model.
    #[must_use]
    pub fn with_model(mut self, model: Arc<dyn CompletionModel>) -> Self {
        self.model = Some(model);
        self
    }

    /// Replaces the rubric.
    #[must_use]
    pub fn with_rubric(mut self, rubric: impl Into<String>) -> Self {
        self.rubric = rubric.into();
        self
    }

    /// Sets the fallback mode.
    #[must_use]
    pub fn with_fallback(mut self, fallback: FallbackMode) -> Self {
        self.fallback = fallback;
        self
    }

    /// The rubric the model is asked to apply.
    #[must_use]
    pub fn rubric(&self) -> &str {
        &self.rubric
    }

    /// Whether a model is configured.
    #[must_use]
    pub fn has_model(&self) -> bool {
        self.model.is_some()
    }

    /// Judges `input`, reporting whether a model produced the verdict.
    ///
    /// With no model configured the static rubric answers with
    /// [`StageProvenance::NoModel`] and exactly one `debug!` is recorded. A usable verdict
    /// from the model — or from the single repair attempt — is
    /// [`StageProvenance::Model`] and logs nothing. A failed model call, or two rejected
    /// replies, under [`FallbackMode::Deterministic`] answers the static rubric with
    /// [`StageProvenance::DeterministicFallback`] and exactly one `warn!` naming the stage
    /// and the fallback rule. Under [`FallbackMode::Error`] the typed error is returned
    /// unchanged and no fallback record is written.
    ///
    /// The `detail` strings are identical to the
    /// [`Metric::evaluate`] path: the no-model path
    /// starts with `deterministic_fallback: no model is configured`, and the failed-model
    /// path with `deterministic_fallback: model judgement failed: {error}`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Llm`] when the capability card cannot be serialized, when
    /// [`FallbackMode::Error`] is configured and the judgement fails, and the
    /// [`CompletionModel::complete`] error unchanged in that case.
    pub async fn evaluate_with_provenance(&self, input: &EvaluationInput) -> Result<JudgeVerdict> {
        if self.model.is_none() {
            note_no_model(STAGE, FALLBACK_RULE);
            return Ok(JudgeVerdict::new(
                self.deterministic_score(input, NO_MODEL_EXPLANATION),
                StageProvenance::NoModel,
            ));
        }
        let card = match serde_json::to_string(&input.capability) {
            Ok(card) => card,
            Err(error) => {
                return Err(Error::llm(format!(
                    "judge metric `{}`: capability card could not be serialized: {error}",
                    self.name
                )))
            }
        };
        match self.judge(&card).await {
            Ok(score) => Ok(JudgeVerdict::new(score, StageProvenance::Model)),
            Err(error) => match self.fallback {
                FallbackMode::Deterministic => {
                    note_fallback(STAGE, FALLBACK_RULE, &error);
                    Ok(JudgeVerdict::new(
                        self.deterministic_score(
                            input,
                            &format!("model judgement failed: {error}"),
                        ),
                        StageProvenance::DeterministicFallback,
                    ))
                }
                FallbackMode::Error => Err(error),
            },
        }
    }

    /// Judges `input` with the model, retrying once after an invalid reply.
    async fn judge(&self, card: &str) -> Result<MetricScore> {
        let Some(model) = self.model.as_deref() else {
            return Err(Error::llm("judge metric: no model is configured"));
        };
        let system = self.system_prompt();
        let user = format!("Capability card:\n{card}");
        let first = complete_json(Some(model), &system, &user).await?;
        let problem = match parse_judgement(first.as_ref()) {
            Ok((score, reason)) => return Ok(self.llm_score(score, reason)),
            Err(problem) => problem,
        };
        let repair = format!(
            "{user}\n\nYour previous reply was rejected: {problem}. Return only the JSON object."
        );
        let second = complete_json(Some(model), &system, &repair).await?;
        match parse_judgement(second.as_ref()) {
            Ok((score, reason)) => Ok(self.llm_score(score, reason)),
            Err(second_problem) => Err(Error::llm(format!(
                "judge metric `{}` rejected the model reply twice: {problem}; then: {second_problem}",
                self.name
            ))),
        }
    }

    /// The system instruction for this metric.
    fn system_prompt(&self) -> String {
        format!("{SYSTEM_PROMPT}\n\nRubric: {}", self.rubric.trim())
    }

    /// The score of a valid model verdict.
    fn llm_score(&self, score: f64, reason: String) -> MetricScore {
        MetricScore::scored(self.name.clone(), score, score >= LLM_PASS_THRESHOLD)
            .with_detail(reason)
    }

    /// The static rubric verdict, with `explanation` recorded in `detail`.
    fn deterministic_score(&self, input: &EvaluationInput, explanation: &str) -> MetricScore {
        let capability = &input.capability;
        let has_description = !capability.description.trim().is_empty();
        let has_inputs = has_described_io(&capability.inputs);
        let has_outputs = has_described_io(&capability.outputs);
        let mut missing: Vec<&str> = Vec::new();
        if !has_description {
            missing.push("description");
        }
        if !has_inputs {
            missing.push("input descriptions");
        }
        if !has_outputs {
            missing.push("output descriptions");
        }
        let score = if missing.is_empty() {
            1.0
        } else if has_description {
            0.5
        } else {
            0.0
        };
        let mut detail = format!("deterministic_fallback: {explanation}");
        if !missing.is_empty() {
            detail.push_str("; missing: ");
            detail.push_str(&missing.join(", "));
        }
        MetricScore::scored(self.name.clone(), score, score >= FALLBACK_PASS_THRESHOLD)
            .with_detail(detail)
    }
}

impl fmt::Debug for LlmJudgeMetric {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("LlmJudgeMetric")
            .field("name", &self.name)
            .field("rubric", &self.rubric)
            .field("model", &self.model.as_deref().map(CompletionModel::name))
            .field("fallback", &self.fallback)
            .finish()
    }
}

#[async_trait]
impl Metric for LlmJudgeMetric {
    fn name(&self) -> &str {
        &self.name
    }

    async fn evaluate(&self, input: &EvaluationInput) -> Result<MetricScore> {
        self.evaluate_with_provenance(input)
            .await
            .map(|verdict| verdict.score)
    }
}

/// Parses one judgement reply into `(score, reason)`, or explains what is wrong.
///
/// The `Err` side is the validation message re-sent to the model by the one repair
/// attempt, so it names the offending field.
#[allow(clippy::float_cmp)] // a binary verdict is exactly 0.0 or 1.0
fn parse_judgement(reply: Option<&Value>) -> std::result::Result<(f64, String), String> {
    let Some(value) = reply else {
        return Err("the reply contained no JSON object".to_string());
    };
    let Some(score) = value.get("score") else {
        return Err("the JSON object has no `score`".to_string());
    };
    let Some(score) = score.as_f64() else {
        return Err("`score` must be the number 0 or 1".to_string());
    };
    if !score.is_finite() || (score != 0.0 && score != 1.0) {
        return Err("`score` must be the number 0 or 1".to_string());
    }
    let Some(reason) = value.get("reason").and_then(Value::as_str) else {
        return Err("`reason` must be a non-empty string".to_string());
    };
    let reason = reason.trim();
    if reason.is_empty() {
        return Err("`reason` must be a non-empty string".to_string());
    }
    Ok((score, reason.to_string()))
}

/// Whether at least one declaration in `ios` carries a non-blank description.
fn has_described_io(ios: &[CapabilityIo]) -> bool {
    ios.iter().any(|io| !io.description.trim().is_empty())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::model::CapabilityDescriptor;
    use openjiuwen_retrieval::StaticCompletionModel;
    use serde_json::json;

    /// A fully described card, so the static rubric scores it `1.0`.
    fn complete_card() -> CapabilityDescriptor {
        CapabilityDescriptor::new("tool", "a", "Name")
            .with_description("Does a thing")
            .with_inputs(vec![
                CapabilityIo::new("in", "string").with_description("input")
            ])
            .with_outputs(vec![
                CapabilityIo::new("out", "string").with_description("output")
            ])
    }

    /// An evaluation input around `card`.
    fn input_for(card: CapabilityDescriptor) -> EvaluationInput {
        EvaluationInput {
            capability: card,
            ..EvaluationInput::default()
        }
    }

    /// An input with the complete card.
    fn complete_input() -> EvaluationInput {
        input_for(complete_card())
    }

    #[test]
    fn parse_judgement_accepts_binary_verdicts_only() {
        assert_eq!(
            parse_judgement(Some(&json!({"score": 1, "reason": " ok "}))),
            Ok((1.0, "ok".to_string()))
        );
        assert_eq!(
            parse_judgement(Some(&json!({"score": 0.0, "reason": "missing io"}))),
            Ok((0.0, "missing io".to_string()))
        );
        assert!(parse_judgement(None).is_err());
        assert!(parse_judgement(Some(&json!({"score": true, "reason": "x"}))).is_err());
        assert!(parse_judgement(Some(&json!({"score": 0.5, "reason": "x"}))).is_err());
        assert!(parse_judgement(Some(&json!({"score": 1, "reason": "  "}))).is_err());
        assert!(parse_judgement(Some(&json!({"reason": "x"}))).is_err());
    }

    #[test]
    fn deterministic_rubric_scores_description_and_io() {
        let metric = LlmJudgeMetric::new("custom", "rubric");
        let score = metric.deterministic_score(
            &EvaluationInput {
                capability: complete_card(),
                ..EvaluationInput::default()
            },
            "test",
        );
        assert_eq!(score.score, Some(1.0));
        assert!(score.passed);
        assert_eq!(score.metric, "custom");
        assert!(score
            .detail
            .unwrap()
            .starts_with("deterministic_fallback: test"));
        let bare = CapabilityDescriptor::new("tool", "b", "Name").with_description("Does it");
        let score = metric.deterministic_score(
            &EvaluationInput {
                capability: bare,
                ..EvaluationInput::default()
            },
            "test",
        );
        assert_eq!(score.score, Some(0.5));
        assert!(score.passed);
        assert!(score
            .detail
            .unwrap()
            .contains("missing: input descriptions"));
        let empty = CapabilityDescriptor::new("tool", "c", "Name");
        let score = metric.deterministic_score(
            &EvaluationInput {
                capability: empty,
                ..EvaluationInput::default()
            },
            "test",
        );
        assert_eq!(score.score, Some(0.0));
        assert!(!score.passed);
    }

    #[tokio::test]
    async fn no_model_reports_no_model_and_the_deterministic_rubric() {
        let metric = LlmJudgeMetric::new("llm_judge", "rubric");
        assert!(!metric.has_model());
        let verdict = metric
            .evaluate_with_provenance(&complete_input())
            .await
            .unwrap();
        assert_eq!(verdict.provenance, StageProvenance::NoModel);
        assert!(!verdict.is_model_backed());
        assert!(verdict.provenance.is_deterministic());
        assert_eq!(verdict.score.score, Some(1.0));
        assert!(verdict.passed());
        let detail = verdict.score.detail.expect("detail");
        assert!(
            detail.starts_with("deterministic_fallback: no model is configured"),
            "{detail}"
        );
    }

    #[tokio::test]
    async fn a_scripted_model_verdict_reports_model() {
        let model = Arc::new(StaticCompletionModel::single(
            "{\"score\":1,\"reason\":\"ok\"}",
        ));
        let metric = LlmJudgeMetric::new("llm_judge", "rubric").with_model(model.clone());
        let verdict = metric
            .evaluate_with_provenance(&complete_input())
            .await
            .unwrap();
        assert_eq!(verdict.provenance, StageProvenance::Model);
        assert!(verdict.is_model_backed());
        assert!(verdict.passed());
        assert_eq!(verdict.score.score, Some(1.0));
        assert_eq!(verdict.score.detail.as_deref(), Some("ok"));
        assert_eq!(model.call_count(), 1);
    }

    #[tokio::test]
    async fn a_scripted_rejection_reports_model_and_does_not_pass() {
        let model = Arc::new(StaticCompletionModel::single(
            "{\"score\":0,\"reason\":\"no examples\"}",
        ));
        let metric = LlmJudgeMetric::new("llm_judge", "rubric").with_model(model.clone());
        let verdict = metric
            .evaluate_with_provenance(&complete_input())
            .await
            .unwrap();
        assert_eq!(verdict.provenance, StageProvenance::Model);
        assert!(!verdict.passed());
        assert_eq!(verdict.score.score, Some(0.0));
        assert_eq!(verdict.score.detail.as_deref(), Some("no examples"));
        assert_eq!(model.call_count(), 1);
    }

    #[tokio::test]
    async fn the_repair_attempt_reports_model_when_it_succeeds() {
        let model = Arc::new(StaticCompletionModel::new(vec![
            "not json".to_string(),
            "{\"score\":1,\"reason\":\"fixed\"}".to_string(),
        ]));
        let metric = LlmJudgeMetric::new("llm_judge", "rubric").with_model(model.clone());
        let verdict = metric
            .evaluate_with_provenance(&complete_input())
            .await
            .unwrap();
        assert_eq!(verdict.provenance, StageProvenance::Model);
        assert_eq!(verdict.score.detail.as_deref(), Some("fixed"));
        assert_eq!(model.call_count(), 2);
        let prompts = model.prompts();
        assert!(prompts[1].contains("Your previous reply was rejected"));
    }

    /// A model whose every call fails, for the fallback paths.
    struct Failing;

    #[async_trait::async_trait]
    impl CompletionModel for Failing {
        fn name(&self) -> &'static str {
            "failing"
        }

        async fn complete(&self, _prompt: &str) -> Result<String> {
            Err(crate::error::Error::llm("provider unavailable"))
        }
    }

    #[tokio::test]
    async fn a_failing_model_reports_the_deterministic_fallback() {
        let metric = LlmJudgeMetric::new("llm_judge", "rubric")
            .with_model(Arc::new(Failing))
            .with_fallback(FallbackMode::Deterministic);
        let verdict = metric
            .evaluate_with_provenance(&complete_input())
            .await
            .unwrap();
        assert_eq!(verdict.provenance, StageProvenance::DeterministicFallback);
        assert!(verdict.provenance.is_warning());
        assert!(!verdict.is_model_backed());
        assert_eq!(verdict.score.score, Some(1.0));
        let detail = verdict.score.detail.expect("detail");
        assert!(
            detail.contains("deterministic_fallback: model judgement failed"),
            "{detail}"
        );
        assert!(detail.contains("provider unavailable"), "{detail}");
    }

    #[tokio::test]
    async fn a_failing_model_surfaces_the_error_when_configured() {
        let metric = LlmJudgeMetric::new("llm_judge", "rubric")
            .with_model(Arc::new(Failing))
            .with_fallback(FallbackMode::Error);
        let error = metric
            .evaluate_with_provenance(&complete_input())
            .await
            .unwrap_err();
        assert!(
            error.to_string().contains("provider unavailable"),
            "{error}"
        );
    }

    #[tokio::test]
    async fn metric_evaluate_returns_the_verdict_score() {
        let metric = LlmJudgeMetric::new("llm_judge", "rubric");
        let input = complete_input();
        let score = metric.evaluate(&input).await.unwrap();
        let verdict = metric.evaluate_with_provenance(&input).await.unwrap();
        assert_eq!(score, verdict.score);
        let model = Arc::new(StaticCompletionModel::single(
            "{\"score\":1,\"reason\":\"ok\"}",
        ));
        let judged = LlmJudgeMetric::new("llm_judge", "rubric").with_model(model);
        let score = judged.evaluate(&input).await.unwrap();
        let verdict = judged.evaluate_with_provenance(&input).await.unwrap();
        assert_eq!(score, verdict.score);
    }

    #[test]
    fn judge_verdict_round_trips_and_defaults_the_provenance() {
        let verdict = JudgeVerdict::new(
            MetricScore::scored("llm_judge", 1.0, true).with_detail("ok"),
            StageProvenance::DeterministicFallback,
        );
        let json = serde_json::to_value(&verdict).unwrap();
        assert_eq!(json["provenance"], json!("deterministic_fallback"));
        assert_eq!(
            serde_json::from_value::<JudgeVerdict>(json).unwrap(),
            verdict
        );
        let legacy: JudgeVerdict = serde_json::from_value(json!({
            "score": {"metric": "llm_judge", "score": 1.0, "passed": true}
        }))
        .unwrap();
        assert_eq!(legacy.provenance, StageProvenance::NoModel);
        assert!(legacy.passed());
        assert_eq!(JudgeVerdict::default().provenance, StageProvenance::NoModel);
    }
}
