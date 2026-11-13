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
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use crate::error::{Error, Result};
use crate::evaluation::{EvaluationInput, Metric, MetricScore};
use crate::model::CapabilityIo;

use super::{complete_json, CompletionModel, FallbackMode};

/// The metric name a [`LlmJudgeMetric`] gets by default.
pub const DEFAULT_JUDGE_NAME: &str = "llm_judge";
/// The rubric a [`LlmJudgeMetric`] gets by default.
pub const DEFAULT_RUBRIC: &str = "The capability card must be self-describing: a non-empty \
description, and at least one input and one output that explain what they carry.";
/// The score a fallback needs to pass.
const FALLBACK_PASS_THRESHOLD: f64 = 0.5;
/// The score a model verdict needs to pass.
const LLM_PASS_THRESHOLD: f64 = 1.0;
/// The system instruction sent with every judgement request.
const SYSTEM_PROMPT: &str = "Judge the capability card against the rubric. \
Return JSON only: {\"score\":0,\"reason\":\"…\"}. \
`score` must be the number 1 when the card satisfies the rubric and 0 when it does not; \
`reason` must be a non-empty explanation.";

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
        MetricScore {
            metric: self.name.clone(),
            score,
            passed: score >= LLM_PASS_THRESHOLD,
            detail: Some(reason),
        }
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
        MetricScore {
            metric: self.name.clone(),
            score,
            passed: score >= FALLBACK_PASS_THRESHOLD,
            detail: Some(detail),
        }
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
        if self.model.is_none() {
            return Ok(self.deterministic_score(input, "no model is configured"));
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
            Ok(score) => Ok(score),
            Err(error) => {
                match self.fallback {
                    FallbackMode::Deterministic => {
                        tracing::warn!(
                            metric = self.name.as_str(),
                            %error,
                            "judge metric: model judgement failed; using the deterministic rubric"
                        );
                        Ok(self.deterministic_score(
                            input,
                            &format!("model judgement failed: {error}"),
                        ))
                    }
                    FallbackMode::Error => Err(error),
                }
            }
        }
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
    use serde_json::json;

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
        let complete = CapabilityDescriptor::new("tool", "a", "Name")
            .with_description("Does a thing")
            .with_inputs(vec![
                CapabilityIo::new("in", "string").with_description("input")
            ])
            .with_outputs(vec![
                CapabilityIo::new("out", "string").with_description("output")
            ]);
        let score = metric.deterministic_score(
            &EvaluationInput {
                capability: complete,
                target: None,
            },
            "test",
        );
        assert_eq!(score.score, 1.0);
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
                target: None,
            },
            "test",
        );
        assert_eq!(score.score, 0.5);
        assert!(score.passed);
        assert!(score
            .detail
            .unwrap()
            .contains("missing: input descriptions"));
        let empty = CapabilityDescriptor::new("tool", "c", "Name");
        let score = metric.deterministic_score(
            &EvaluationInput {
                capability: empty,
                target: None,
            },
            "test",
        );
        assert_eq!(score.score, 0.0);
        assert!(!score.passed);
    }
}
