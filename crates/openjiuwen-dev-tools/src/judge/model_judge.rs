//! The judge seam and its rubric-driven model implementation.
//!
//! [`Judge`] is the one-method trait the crate's evaluators consume;
//! [`ModelJudge`] implements it over the [`Model`] seam. One judgement is one
//! non-streaming model call whose answer is read with
//! [`Verdict::parse`](crate::judge::Verdict::parse), so the judge is tolerant of
//! the shape a model happens to choose.
//!
//! [`ModelJudge::judge_with_fallback`] is the resilient entry point used by
//! scoring pipelines: any failure — an invalid rubric, a provider error, an
//! unparsable answer, or a verdict that fails validation — degrades to a
//! caller-supplied deterministic verdict instead of failing the run. The reason
//! is logged with [`tracing::warn!`], so degradation is visible but never fatal.

use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::llm::{Model, ModelRequestConfig};
use openjiuwen_foundation::prompt::BaseMessage;

use crate::judge::prompt;
use crate::judge::rubric::Rubric;
use crate::judge::verdict::{Verdict, VerdictSource};

/// One judging request: the rubric, the subject, the criteria, and a reference.
///
/// The type borrows its inputs so a pipeline can judge the same owned strings
/// many times without copying them; it is therefore not deserializable.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JudgeRequest<'a> {
    /// The rubric the subject is scored against.
    pub rubric: &'a Rubric,
    /// The material under evaluation, e.g. a skill document or a model answer.
    pub subject: &'a str,
    /// What the judge must check, in the caller's words.
    pub instructions: &'a str,
    /// An optional exemplar or ground truth to compare against.
    pub reference: Option<&'a str>,
}

impl<'a> JudgeRequest<'a> {
    /// Creates a request without a reference.
    #[must_use]
    pub fn new(rubric: &'a Rubric, subject: &'a str, instructions: &'a str) -> Self {
        Self {
            rubric,
            subject,
            instructions,
            reference: None,
        }
    }

    /// Attaches a reference exemplar or ground truth.
    #[must_use]
    pub fn with_reference(mut self, reference: &'a str) -> Self {
        self.reference = Some(reference);
        self
    }
}

/// Scores a request against its rubric.
#[async_trait]
pub trait Judge: Send + Sync {
    /// Produces one verdict for `request`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the rubric is invalid or the answer
    /// carries no parsable verdict, and the provider's own error when the model
    /// call fails.
    async fn judge(&self, request: &JudgeRequest<'_>) -> Result<Verdict>;
}

/// A [`Judge`] that asks a [`Model`] to apply the rubric.
pub struct ModelJudge {
    /// The model consulted for every judgement.
    model: Arc<dyn Model>,
    /// Replaces the built-in system prompt when set.
    system_prompt: Option<String>,
}

impl ModelJudge {
    /// Creates a judge over `model` using the built-in system prompt.
    #[must_use]
    pub fn new(model: Arc<dyn Model>) -> Self {
        Self {
            model,
            system_prompt: None,
        }
    }

    /// Replaces the built-in system prompt.
    #[must_use]
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// The model this judge consults.
    #[must_use]
    pub fn model(&self) -> &Arc<dyn Model> {
        &self.model
    }

    /// Judges `request`, returning `fallback()` on any failure.
    ///
    /// The fallback is marked [`VerdictSource::Deterministic`] so callers can
    /// tell a degraded judgement from a model one. Every failure mode logs a
    /// [`tracing::warn!`] naming the reason.
    pub async fn judge_with_fallback<F>(&self, request: &JudgeRequest<'_>, fallback: F) -> Verdict
    where
        F: FnOnce() -> Verdict,
    {
        match self.judge(request).await {
            Ok(verdict) => match verdict.validate() {
                Ok(()) => verdict,
                Err(error) => {
                    tracing::warn!(
                        rubric = %request.rubric.name,
                        %error,
                        "model judge returned an invalid verdict; using the deterministic fallback"
                    );
                    deterministic(fallback())
                }
            },
            Err(error) => {
                tracing::warn!(
                    rubric = %request.rubric.name,
                    %error,
                    "model judge failed; using the deterministic fallback"
                );
                deterministic(fallback())
            }
        }
    }
}

/// Marks a fallback verdict as deterministic.
fn deterministic(mut verdict: Verdict) -> Verdict {
    verdict.source = VerdictSource::Deterministic;
    verdict
}

impl std::fmt::Debug for ModelJudge {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ModelJudge")
            .field("provider", &self.model.provider())
            .field("model_name", &self.model.model_name())
            .field("system_prompt_override", &self.system_prompt.is_some())
            .finish()
    }
}

#[async_trait]
impl Judge for ModelJudge {
    async fn judge(&self, request: &JudgeRequest<'_>) -> Result<Verdict> {
        request.rubric.validate()?;

        let mut messages = prompt::rubric_messages(request);
        if let Some(system_prompt) = &self.system_prompt {
            if let Some(first) = messages.first_mut() {
                *first = BaseMessage::system(system_prompt.clone());
            }
        }

        let config = ModelRequestConfig::new(self.model.model_name());
        let completion = self.model.invoke(&messages, &config).await?;
        if completion.choices.is_empty() {
            return Err(Error::llm(
                "judge model returned no choices; expected a single structured verdict",
            ));
        }

        let answer = completion
            .message()
            .map(|message| message.text().into_owned())
            .unwrap_or_default();
        Verdict::parse(&answer, request.rubric).ok_or_else(|| {
            Error::validation(format!(
                "judge model answer contained no structured verdict, expected <score>…</score> \
                 sections or a fenced JSON object; answer began with {:?}",
                answer.chars().take(120).collect::<String>()
            ))
        })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use openjiuwen_test_support::{ScriptedModel, ScriptedTurn};

    use super::*;
    use crate::judge::rubric::RubricDimension;

    fn rubric() -> Rubric {
        Rubric::new("skill quality", "Does the skill do its job?")
            .with_dimension(RubricDimension::new("clarity", "Is it clear?"))
            .with_dimension(RubricDimension::new("depth", "Is it deep?"))
    }

    fn request(rubric: &Rubric) -> JudgeRequest<'_> {
        JudgeRequest::new(rubric, "## Skill\n\nDo the thing.", "Judge the skill.")
    }

    fn fenced_json() -> String {
        "Here is the verdict.\n```json\n\
         {\"score\": 0.75, \"passed\": true, \"reasoning\": \"clear enough\", \
         \"dimension_scores\": {\"clarity\": 0.9, \"depth\": 0.6}}\n```\n"
            .to_owned()
    }

    #[test]
    fn request_carries_the_optional_reference() {
        let rubric = rubric();
        let bare = request(&rubric);
        assert!(bare.reference.is_none());
        let with_reference = bare.with_reference("the exemplar");
        assert_eq!(with_reference.reference, Some("the exemplar"));
    }

    #[tokio::test]
    async fn judges_a_fenced_json_answer_in_one_call() {
        let model = ScriptedModel::text(fenced_json());
        let judge = ModelJudge::new(Arc::new(model.clone()));
        let rubric = rubric();

        let verdict = judge.judge(&request(&rubric)).await.expect("judges");
        assert_eq!(verdict.source, VerdictSource::Model);
        assert!((verdict.score - 0.75).abs() < 1e-9);
        assert!(verdict.passed);
        assert_eq!(verdict.reasoning, "clear enough");
        assert_eq!(verdict.dimension_scores.len(), 2);
        assert_eq!(model.call_count(), 1);
        assert_eq!(model.remaining(), 0);

        let call = model.last_call().expect("one call");
        assert_eq!(call.messages.len(), 2);
        assert!(call.messages[0].role.is_system());
        assert_eq!(
            call.messages[0].text().into_owned(),
            prompt::system_prompt()
        );
        assert_eq!(call.request.model_name, model.model_name());
    }

    #[tokio::test]
    async fn judges_a_tag_answer() {
        let model = ScriptedModel::text(
            "<score>0.8</score>\n<reasoning>strong</reasoning>\n\
             <dimension name=\"clarity\">0.9</dimension>",
        );
        let judge = ModelJudge::new(Arc::new(model));
        let rubric = rubric();

        let verdict = judge.judge(&request(&rubric)).await.expect("judges");
        assert_eq!(verdict.reasoning, "strong");
        assert_eq!(verdict.dimension_scores["clarity"], 0.9);
        // One known dimension at full weight out of two equal weights.
        assert!((verdict.score - 0.45).abs() < 1e-9);
        assert!(!verdict.passed);
    }

    #[tokio::test]
    async fn rejects_an_unparsable_answer() {
        let model = ScriptedModel::text("I think it is pretty good overall.");
        let judge = ModelJudge::new(Arc::new(model.clone()));
        let rubric = rubric();

        let error = judge
            .judge(&request(&rubric))
            .await
            .expect_err("no verdict");
        assert!(matches!(error, Error::Validation(_)), "{error:?}");
        assert!(error.to_string().contains("structured verdict"), "{error}");
        assert_eq!(model.call_count(), 1);
    }

    #[tokio::test]
    async fn rejects_an_invalid_rubric_without_calling_the_model() {
        let model = ScriptedModel::text(fenced_json());
        let judge = ModelJudge::new(Arc::new(model.clone()));
        let rubric = Rubric::new("demo", "A demo");

        let error = judge
            .judge(&request(&rubric))
            .await
            .expect_err("no dimensions");
        assert!(matches!(error, Error::Validation(_)), "{error:?}");
        assert!(error.to_string().contains("dimension"), "{error}");
        assert_eq!(model.call_count(), 0);
    }

    #[tokio::test]
    async fn propagates_model_failures() {
        let model = ScriptedModel::new([ScriptedTurn::fail("provider down")]);
        let judge = ModelJudge::new(Arc::new(model));
        let rubric = rubric();

        let error = judge.judge(&request(&rubric)).await.expect_err("fails");
        assert!(error.to_string().contains("provider down"), "{error}");
    }

    #[tokio::test]
    async fn rejects_a_response_without_choices() {
        let mut response =
            openjiuwen_test_support::MockCompletion::text("ignored").to_chat_response("");
        response.choices.clear();
        let model = ScriptedModel::new([ScriptedTurn::Response(response)]);
        let judge = ModelJudge::new(Arc::new(model));
        let rubric = rubric();

        let error = judge
            .judge(&request(&rubric))
            .await
            .expect_err("no choices");
        assert!(matches!(error, Error::Llm(_)), "{error:?}");
        assert!(error.to_string().contains("no choices"), "{error}");
    }

    #[tokio::test]
    async fn a_system_prompt_override_replaces_the_built_in_one() {
        let model = ScriptedModel::text(fenced_json());
        let judge =
            ModelJudge::new(Arc::new(model.clone())).with_system_prompt("You are a lenient judge.");
        let rubric = rubric();

        judge.judge(&request(&rubric)).await.expect("judges");
        let call = model.last_call().expect("one call");
        assert_eq!(
            call.messages[0].text().into_owned(),
            "You are a lenient judge."
        );
    }

    #[tokio::test]
    async fn judge_with_fallback_keeps_a_successful_verdict() {
        let model = ScriptedModel::text(fenced_json());
        let judge = ModelJudge::new(Arc::new(model));
        let rubric = rubric();

        let verdict = judge
            .judge_with_fallback(&request(&rubric), || {
                Verdict::deterministic(0.1, "fallback")
            })
            .await;
        assert_eq!(verdict.source, VerdictSource::Model);
        assert_eq!(verdict.reasoning, "clear enough");
    }

    #[tokio::test]
    async fn judge_with_fallback_uses_the_deterministic_verdict_on_an_invalid_rubric() {
        let model = ScriptedModel::text(fenced_json());
        let judge = ModelJudge::new(Arc::new(model.clone()));
        let rubric = Rubric::new("", "A demo");

        let verdict = judge
            .judge_with_fallback(&request(&rubric), || {
                Verdict::deterministic(0.2, "rubric missing")
            })
            .await;
        assert_eq!(verdict.source, VerdictSource::Deterministic);
        assert!((verdict.score - 0.2).abs() < 1e-9);
        assert_eq!(verdict.reasoning, "rubric missing");
        assert_eq!(model.call_count(), 0);
    }

    #[tokio::test]
    async fn judge_with_fallback_uses_the_deterministic_verdict_on_model_failure() {
        let model = ScriptedModel::new([ScriptedTurn::fail("rate limited")]);
        let judge = ModelJudge::new(Arc::new(model));
        let rubric = rubric();

        let verdict = judge
            .judge_with_fallback(&request(&rubric), || {
                Verdict::deterministic(0.3, "model unavailable")
            })
            .await;
        assert_eq!(verdict.source, VerdictSource::Deterministic);
        assert!((verdict.score - 0.3).abs() < 1e-9);
    }

    #[tokio::test]
    async fn judge_with_fallback_uses_the_deterministic_verdict_on_an_unparsable_answer() {
        let model = ScriptedModel::text("no structure at all");
        let judge = ModelJudge::new(Arc::new(model));
        let rubric = rubric();

        let verdict = judge
            .judge_with_fallback(&request(&rubric), || {
                Verdict::deterministic(0.4, "unparsable")
            })
            .await;
        assert_eq!(verdict.source, VerdictSource::Deterministic);
        assert!((verdict.score - 0.4).abs() < 1e-9);
        assert_eq!(verdict.reasoning, "unparsable");
    }

    #[test]
    fn debug_names_the_provider_model_and_override_state() {
        let model = ScriptedModel::builder()
            .provider("acme")
            .model_name("acme-judge")
            .build();
        let judge = ModelJudge::new(Arc::new(model.clone()));
        let rendered = format!("{judge:?}");
        assert!(rendered.contains("provider: \"acme\""), "{rendered}");
        assert!(rendered.contains("acme-judge"), "{rendered}");
        assert!(
            rendered.contains("system_prompt_override: false"),
            "{rendered}"
        );
        assert_eq!(judge.model().model_name(), model.model_name());

        let overridden = ModelJudge::new(Arc::new(model)).with_system_prompt("be strict");
        assert!(format!("{overridden:?}").contains("system_prompt_override: true"));
    }

    #[test]
    fn fallback_verdicts_forget_any_model_dimensions() {
        let mut scores = BTreeMap::new();
        scores.insert("clarity".to_owned(), 0.5);
        let verdict = deterministic(Verdict::model(0.5, true, "model", scores));
        assert_eq!(verdict.source, VerdictSource::Deterministic);
        assert_eq!(verdict.dimension_scores.len(), 1);
    }
}
