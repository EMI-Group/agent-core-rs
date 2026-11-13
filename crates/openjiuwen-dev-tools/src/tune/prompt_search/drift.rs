//! Semantic drift protection: the judge seam and its two implementations.
//!
//! Ports `drift.py`. Semantic drift approximates a KL constraint with a judge:
//! given the original objective and a candidate prompt, return a deviation in
//! `0.0..=1.0` (`0.0` = faithful, `1.0` = the prompt has repurposed the task).
//! The composite reward subtracts `drift_penalty * deviation`.
//!
//! Python's only real implementation is an LLM judge; here
//! [`LexicalDriftJudge`] is the deterministic default (token-overlap based) and
//! [`ModelDriftJudge`] ports the LLM judge over the [`Model`] seam. Both fail
//! safe: a judge outage never penalizes a candidate.

use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_foundation::llm::{Model, ModelRequestConfig};
use openjiuwen_foundation::prompt::BaseMessage;
use serde_json::Value;

use crate::util::extract_json_fence;

use super::memory::token_set;
use super::models::PromptCandidate;
use super::reward::clamp01;

/// Measures how far a candidate prompt has drifted from the original objective.
#[async_trait]
pub trait DriftJudge: Send + Sync {
    /// The semantic deviation of `candidate` from `objective`, in `0.0..=1.0`.
    ///
    /// Implementations return a clamped value; the caller clamps again, so an
    /// out-of-range answer cannot escape the reward range.
    async fn deviation(&self, objective: &str, candidate: &PromptCandidate) -> f64;
}

/// A no-op judge that always reports no drift.
///
/// Useful for tests and for tasks where drift protection is not wanted.
#[derive(Debug, Clone, Copy, Default)]
pub struct NullDriftJudge;

impl NullDriftJudge {
    /// Creates the judge.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl DriftJudge for NullDriftJudge {
    async fn deviation(&self, _objective: &str, _candidate: &PromptCandidate) -> f64 {
        0.0
    }
}

/// A deterministic lexical approximation of semantic drift.
///
/// Both texts are lowercased into word-token sets; the deviation is
/// `1 - |objective ∩ prompt| / |objective|`. A blank prompt has drifted
/// completely (`1.0`), and a blank objective cannot be drifted from (`0.0`).
/// This is the default judge: it needs no model, and it is reproducible.
#[derive(Debug, Clone, Copy, Default)]
pub struct LexicalDriftJudge;

impl LexicalDriftJudge {
    /// Creates the judge.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl DriftJudge for LexicalDriftJudge {
    async fn deviation(&self, objective: &str, candidate: &PromptCandidate) -> f64 {
        if candidate.prompt.trim().is_empty() {
            return 1.0;
        }
        let objective_tokens = token_set(objective);
        if objective_tokens.is_empty() {
            return 0.0;
        }
        let prompt_tokens = token_set(&candidate.prompt);
        let overlap = objective_tokens.intersection(&prompt_tokens).count();
        clamp01(1.0 - overlap as f64 / objective_tokens.len() as f64)
    }
}

/// The system instruction of the LLM drift judge.
const SYSTEM_PROMPT: &str = "You measure whether an optimized system prompt still serves the \
original task objective. Output only valid JSON wrapped in ```json fences.";

/// A judge that asks a [`Model`] how far a candidate drifted.
///
/// One non-streaming call returns `{"deviation": <float>, "reason": "..."}` in a
/// fenced JSON block (a bare JSON object is accepted too). Any failure — a
/// provider error or an unusable answer — is logged and reported as `0.0`
/// drift, so a judge outage never penalizes a candidate.
#[derive(Clone)]
pub struct ModelDriftJudge {
    /// The model consulted for every measurement.
    model: Arc<dyn Model>,
}

impl std::fmt::Debug for ModelDriftJudge {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ModelDriftJudge")
            .field("model_name", &self.model.model_name())
            .finish()
    }
}

impl ModelDriftJudge {
    /// Creates the judge over `model`.
    #[must_use]
    pub fn new(model: Arc<dyn Model>) -> Self {
        Self { model }
    }

    /// The model consulted by the judge.
    #[must_use]
    pub fn model(&self) -> &Arc<dyn Model> {
        &self.model
    }
}

/// Renders the judge's user message.
fn render_prompt(objective: &str, prompt: &str) -> String {
    format!(
        "Original task objective:\n{objective}\nCandidate system prompt:\n{prompt}\n\
         Does the candidate system prompt still faithfully pursue the SAME objective, or has it \
         drifted toward a different task, added unrelated goals, or narrowed/broadened the scope?\n\
         Return ONLY valid JSON wrapped in ```json fences:\n\
         ```json\n{{\"deviation\": <float 0..1>, \"reason\": \"<short reason>\"}}\n```\n\
         where 0.0 means identical intent and 1.0 means a completely different task.\n"
    )
}

/// Parses a deviation from a model answer, accepting a fence or a bare object.
fn parse_deviation(answer: &str) -> Option<f64> {
    let value =
        extract_json_fence(answer).or_else(|| serde_json::from_str::<Value>(answer.trim()).ok())?;
    value.get("deviation")?.as_f64().map(clamp01)
}

#[async_trait]
impl DriftJudge for ModelDriftJudge {
    async fn deviation(&self, objective: &str, candidate: &PromptCandidate) -> f64 {
        if candidate.prompt.trim().is_empty() {
            return 1.0;
        }
        let messages = vec![
            BaseMessage::system(SYSTEM_PROMPT),
            BaseMessage::user(render_prompt(objective, &candidate.prompt)),
        ];
        let request = ModelRequestConfig::new(self.model.model_name());
        let completion = match self.model.invoke(&messages, &request).await {
            Ok(completion) => completion,
            Err(error) => {
                tracing::warn!(%error, "drift judge failed; assuming 0 drift");
                return 0.0;
            }
        };
        let answer = completion
            .message()
            .map(|message| message.text().into_owned())
            .unwrap_or_default();
        if let Some(deviation) = parse_deviation(&answer) {
            deviation
        } else {
            tracing::warn!(
                answer = %answer.chars().take(120).collect::<String>(),
                "drift judge answer contained no deviation; assuming 0 drift"
            );
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use openjiuwen_test_support::{ScriptedError, ScriptedModel};

    use super::*;

    fn candidate(prompt: &str) -> PromptCandidate {
        PromptCandidate::new(prompt, "why")
    }

    #[tokio::test]
    async fn null_judge_is_a_no_op() {
        let judge = NullDriftJudge::new();
        assert_eq!(
            judge.deviation("objective", &candidate("prompt")).await,
            0.0
        );
        assert_eq!(judge.deviation("", &candidate("")).await, 0.0);
    }

    #[tokio::test]
    async fn lexical_judge_bands() {
        let judge = LexicalDriftJudge::new();
        let objective = "write a haiku about autumn";
        assert_eq!(
            judge
                .deviation(objective, &candidate("write a haiku about autumn"))
                .await,
            0.0
        );
        let partial = judge
            .deviation(objective, &candidate("write a haiku about spring"))
            .await;
        assert!((partial - 0.2).abs() < 1e-12, "{partial}");
        let complete = judge
            .deviation(objective, &candidate("analyze the stock market"))
            .await;
        assert!((complete - 1.0).abs() < 1e-12);
        assert_eq!(judge.deviation(objective, &candidate("   ")).await, 1.0);
        assert_eq!(judge.deviation("   ", &candidate("anything")).await, 0.0);
        assert!((judge.deviation(objective, &candidate("Autumn HAIKU")).await - 0.6).abs() < 1e-12);
    }

    #[tokio::test]
    async fn model_judge_parses_a_fenced_answer() {
        let model =
            ScriptedModel::text("```json\n{\"deviation\": 0.42, \"reason\": \"narrowed\"}\n```");
        let judge = ModelDriftJudge::new(Arc::new(model.clone()));
        let value = judge
            .deviation("write a haiku", &candidate("write a haiku"))
            .await;
        assert!((value - 0.42).abs() < 1e-12);
        assert_eq!(model.call_count(), 1);
        let call = model.last_call().unwrap();
        let text = call.last_user_text().unwrap();
        assert!(text.contains("write a haiku"));
        assert!(text.contains("Candidate system prompt"));
        assert_eq!(judge.model().model_name(), "scripted-model");
    }

    #[tokio::test]
    async fn model_judge_accepts_a_bare_object_and_clamps() {
        let model = ScriptedModel::text("{\"deviation\": 4}");
        let judge = ModelDriftJudge::new(Arc::new(model));
        assert_eq!(judge.deviation("o", &candidate("p")).await, 1.0);
    }

    #[tokio::test]
    async fn model_judge_degrades_to_zero() {
        let unusable = ScriptedModel::text("no json here");
        let judge = ModelDriftJudge::new(Arc::new(unusable));
        assert_eq!(judge.deviation("o", &candidate("p")).await, 0.0);

        let failing = ScriptedModel::builder()
            .fail(ScriptedError::llm("provider down"))
            .build();
        let judge = ModelDriftJudge::new(Arc::new(failing));
        assert_eq!(judge.deviation("o", &candidate("p")).await, 0.0);

        let missing = ScriptedModel::text("```json\n{\"reason\": \"none\"}\n```");
        let judge = ModelDriftJudge::new(Arc::new(missing));
        assert_eq!(judge.deviation("o", &candidate("p")).await, 0.0);
    }

    #[tokio::test]
    async fn model_judge_skips_the_call_for_a_blank_prompt() {
        let model = ScriptedModel::text("```json\n{\"deviation\": 0.9}\n```");
        let judge = ModelDriftJudge::new(Arc::new(model.clone()));
        assert_eq!(judge.deviation("o", &candidate("  ")).await, 1.0);
        assert_eq!(model.call_count(), 0);
    }
}
