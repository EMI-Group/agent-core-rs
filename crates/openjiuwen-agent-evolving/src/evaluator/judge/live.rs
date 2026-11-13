//! The live-model judge: `LlmJudge` and its fallback-aware evaluator.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_core::foundation::llm::{Model, ModelRequestConfig};
use serde_json::Value;

use super::prompt::build_judge_messages;
use super::rubric::JudgeRubric;
use super::verdict::JudgeVerdict;
use crate::dataset::{Case, EvaluatedCase};
use crate::evaluator::metrics::{finite_or_zero, ExactMatchMetric, Metric, MetricValue};
use crate::evaluator::Evaluator;

/// The name every judge evaluator reports, matching Python's metric name.
const JUDGE_EVALUATOR_NAME: &str = "llm_as_judge";

/// How many judge calls [`LlmJudge`] attempts before reporting a parse failure.
const DEFAULT_JUDGE_ATTEMPTS: usize = 2;

/// The upper bound [`LlmJudge::with_attempts`] clamps to.
const MAX_JUDGE_ATTEMPTS: usize = 3;

/// How many prediction characters [`LlmJudge`] sends by default.
const DEFAULT_MAX_PREDICTION_CHARS: usize = 8000;

/// Scores predictions by asking a model to judge them against the case label.
///
/// This is the live-model path; the synchronous, verdict-parsing counterpart is
/// [`LlmAsJudgeMetric`](crate::evaluator::metrics::LlmAsJudgeMetric).
#[derive(Clone)]
pub struct LlmJudge {
    model: Arc<dyn Model>,
    rubric: JudgeRubric,
    request: ModelRequestConfig,
    attempts: usize,
    max_prediction_chars: usize,
}

impl fmt::Debug for LlmJudge {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("LlmJudge")
            .field("model", &self.model.model_name())
            .field("rubric", &self.rubric)
            .field("request", &self.request)
            .field("attempts", &self.attempts)
            .field("max_prediction_chars", &self.max_prediction_chars)
            .finish()
    }
}

impl LlmJudge {
    /// Creates a judge over `model`, using `rubric` and `request`.
    ///
    /// Defaults: 2 attempts and an 8000-character prediction cap.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] when `rubric` is invalid, so an unusable rubric
    /// fails at construction instead of on every case.
    pub fn new(
        model: Arc<dyn Model>,
        rubric: JudgeRubric,
        request: ModelRequestConfig,
    ) -> Result<Self> {
        rubric.validate()?;
        Ok(Self {
            model,
            rubric,
            request,
            attempts: DEFAULT_JUDGE_ATTEMPTS,
            max_prediction_chars: DEFAULT_MAX_PREDICTION_CHARS,
        })
    }

    /// Sets how many model calls one verdict may cost, clamped to `1..=3`.
    ///
    /// A retry only happens when the response cannot be parsed; model errors are
    /// never retried.
    #[must_use]
    pub fn with_attempts(mut self, attempts: usize) -> Self {
        self.attempts = attempts.clamp(1, MAX_JUDGE_ATTEMPTS);
        self
    }

    /// Sets the prediction character cap; `0` disables truncation.
    #[must_use]
    pub fn with_max_prediction_chars(mut self, max_prediction_chars: usize) -> Self {
        self.max_prediction_chars = max_prediction_chars;
        self
    }

    /// The rubric scores are read against.
    #[must_use]
    pub fn rubric(&self) -> &JudgeRubric {
        &self.rubric
    }

    /// The request configuration sent with every judge call.
    #[must_use]
    pub fn request(&self) -> &ModelRequestConfig {
        &self.request
    }

    /// The model calls a single verdict may cost.
    #[must_use]
    pub fn attempts(&self) -> usize {
        self.attempts
    }

    /// The prediction character cap; `0` means no truncation.
    #[must_use]
    pub fn max_prediction_chars(&self) -> usize {
        self.max_prediction_chars
    }

    /// The model under test.
    #[must_use]
    pub fn model(&self) -> &dyn Model {
        self.model.as_ref()
    }

    /// Asks the model to judge `prediction` for `case`.
    ///
    /// Each attempt reuses the same messages; a response that does not parse is
    /// retried up to [`LlmJudge::attempts`] times and the last parse error is
    /// returned. There is no sleep or backoff, so the call is deterministic given
    /// a deterministic model.
    ///
    /// # Errors
    ///
    /// The model's own error, or [`Error::validation`] when every response was
    /// unparseable.
    pub async fn judge(&self, case: &Case, prediction: &Value) -> Result<JudgeVerdict> {
        let messages =
            build_judge_messages(case, prediction, &self.rubric, self.max_prediction_chars);
        let mut last_error: Option<Error> = None;
        for _ in 0..self.attempts.max(1) {
            let response = self.model.invoke(&messages, &self.request).await?;
            let content = response
                .message()
                .map_or_else(String::new, |message| message.text().into_owned());
            match JudgeVerdict::parse(&content, &self.rubric) {
                Ok(verdict) => return Ok(verdict),
                Err(error) => last_error = Some(error),
            }
        }
        Err(last_error
            .unwrap_or_else(|| Error::validation("the judge model returned an empty verdict")))
    }
}

/// Scores with an [`LlmJudge`] and falls back to a deterministic [`Metric`].
///
/// The fallback is not a silent degradation: whenever it is used, the reason
/// says whether no judge was configured or the judge failed, and the fallback
/// metric's name is recorded in `per_metric`.
#[derive(Clone)]
pub struct LlmJudgeEvaluator {
    judge: Option<Arc<LlmJudge>>,
    fallback: Arc<dyn Metric>,
    rubric: JudgeRubric,
}

impl fmt::Debug for LlmJudgeEvaluator {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("LlmJudgeEvaluator")
            .field("judge", &self.judge)
            .field("fallback", &self.fallback.name())
            .field("rubric", &self.rubric)
            .finish()
    }
}

impl Default for LlmJudgeEvaluator {
    /// Equivalent to [`LlmJudgeEvaluator::new`].
    fn default() -> Self {
        Self::new()
    }
}

impl LlmJudgeEvaluator {
    /// Creates an evaluator without a judge, scoring with exact match until a
    /// judge is installed.
    #[must_use]
    pub fn new() -> Self {
        Self {
            judge: None,
            fallback: Arc::new(ExactMatchMetric::new()),
            rubric: JudgeRubric::default_rubric(),
        }
    }

    /// Installs the judge and adopts a copy of its rubric.
    #[must_use]
    pub fn with_judge(mut self, judge: Arc<LlmJudge>) -> Self {
        self.rubric = judge.rubric().clone();
        self.judge = Some(judge);
        self
    }

    /// Replaces the rubric.
    #[must_use]
    pub fn with_rubric(mut self, rubric: JudgeRubric) -> Self {
        self.rubric = rubric;
        self
    }

    /// Replaces the deterministic metric used when the judge cannot score.
    #[must_use]
    pub fn with_fallback(mut self, fallback: Arc<dyn Metric>) -> Self {
        self.fallback = fallback;
        self
    }

    /// The installed judge, if any.
    #[must_use]
    pub fn judge(&self) -> Option<&LlmJudge> {
        self.judge.as_deref()
    }

    /// The deterministic metric used when the judge cannot score.
    #[must_use]
    pub fn fallback(&self) -> &dyn Metric {
        self.fallback.as_ref()
    }

    /// The rubric a judge built from this evaluator would use.
    #[must_use]
    pub fn rubric(&self) -> &JudgeRubric {
        &self.rubric
    }
}

#[async_trait]
impl Evaluator for LlmJudgeEvaluator {
    fn name(&self) -> &'static str {
        JUDGE_EVALUATOR_NAME
    }

    async fn evaluate(&self, case: &Case, prediction: &Value) -> Result<EvaluatedCase> {
        let Some(judge) = &self.judge else {
            return fallback_evaluated(
                case,
                prediction,
                self.fallback.as_ref(),
                None,
                "no judge model is configured",
            );
        };
        match judge.judge(case, prediction).await {
            Ok(verdict) => judge_evaluated(case, &verdict),
            Err(error) => fallback_evaluated(
                case,
                prediction,
                self.fallback.as_ref(),
                Some(&error),
                "the judge call failed",
            ),
        }
    }
}

/// Builds an [`EvaluatedCase`] from a verdict.
///
/// The reason is the verdict's reasoning followed by a deterministic
/// `[criterion=score, ...]` breakdown, and `per_metric` carries every criterion
/// plus the overall score under `llm_as_judge`.
pub(crate) fn judge_evaluated(case: &Case, verdict: &JudgeVerdict) -> Result<EvaluatedCase> {
    let breakdown = verdict
        .per_criterion
        .iter()
        .map(|(name, value)| format!("{name}={value}"))
        .collect::<Vec<_>>()
        .join(", ");
    let reasoning = verdict.reasoning.trim();
    let reason = match (reasoning.is_empty(), breakdown.is_empty()) {
        (true, true) => String::from("judged by the model"),
        (true, false) => format!("[{breakdown}]"),
        (false, true) => reasoning.to_owned(),
        (false, false) => format!("{reasoning} [{breakdown}]"),
    };

    let mut per_metric = verdict.per_criterion.clone();
    per_metric.insert(JUDGE_EVALUATOR_NAME.to_owned(), verdict.score);

    let mut evaluated = EvaluatedCase::new(case.clone(), None, verdict.score, reason)?;
    evaluated.per_metric = Some(per_metric);
    Ok(evaluated)
}

/// Builds an [`EvaluatedCase`] by scoring with `metric`.
///
/// `failure` is the judge error, when there was one; the reason names it and the
/// fallback metric, so a fallback score is never mistaken for a judge score. A
/// named metric value is averaged (the same aggregation
/// [`MetricEvaluator`](crate::evaluator::metrics::MetricEvaluator) uses by default), and
/// non-finite values are coerced to `0.0` rather than poisoning the report.
///
/// # Errors
///
/// Whatever `metric.compute` returns, or [`Error::validation`] when the score
/// cannot be turned into an [`EvaluatedCase`].
pub(crate) fn fallback_evaluated(
    case: &Case,
    prediction: &Value,
    metric: &dyn Metric,
    failure: Option<&Error>,
    context: &str,
) -> Result<EvaluatedCase> {
    let value = metric.compute(prediction, &case.label)?;

    let (score, mut per_metric) = match value {
        MetricValue::Scalar(value) => (finite_or_zero(value), BTreeMap::new()),
        MetricValue::Named(named) => {
            let values = named
                .values()
                .copied()
                .map(finite_or_zero)
                .collect::<Vec<_>>();
            let score = if values.is_empty() {
                0.0
            } else {
                values.iter().sum::<f64>() / values.len() as f64
            };
            (score, named)
        }
    };

    per_metric
        .entry(metric.name().to_owned())
        .or_insert(finite_or_zero(score));
    // Keep the map purely finite so `EvaluationReport` aggregation stays sane.
    for value in per_metric.values_mut() {
        *value = finite_or_zero(*value);
    }

    let reason = match failure {
        Some(error) => format!(
            "{context} ({error}); scored with fallback metric `{}`",
            metric.name()
        ),
        None => format!("{context}; scored with fallback metric `{}`", metric.name()),
    };

    let mut evaluated = EvaluatedCase::new(case.clone(), None, score, reason)?;
    evaluated.per_metric = Some(per_metric);
    Ok(evaluated)
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    use futures::stream::BoxStream;
    use futures::StreamExt;
    use openjiuwen_core::foundation::llm::{ChatResponse, Choice, StreamChunk};
    use openjiuwen_core::foundation::prompt::BaseMessage;
    use serde_json::json;

    use super::super::test_support::{case, close};
    use super::*;

    /// A model that serves queued responses and records how often it was called.
    struct FakeModel {
        responses: Mutex<VecDeque<Result<String>>>,
        calls: AtomicUsize,
    }

    impl FakeModel {
        fn new(responses: Vec<Result<String>>) -> Arc<Self> {
            Arc::new(Self {
                responses: Mutex::new(responses.into()),
                calls: AtomicUsize::new(0),
            })
        }

        fn calls(&self) -> usize {
            self.calls.load(Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl Model for FakeModel {
        fn provider(&self) -> &'static str {
            "fake"
        }

        fn model_name(&self) -> &'static str {
            "fake-judge"
        }

        async fn invoke(
            &self,
            _messages: &[BaseMessage],
            _request: &ModelRequestConfig,
        ) -> Result<ChatResponse> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let next = self.responses.lock().unwrap().pop_front();
            match next {
                Some(Ok(content)) => Ok(ChatResponse {
                    choices: vec![Choice {
                        message: BaseMessage::assistant(content),
                        ..Choice::default()
                    }],
                    ..ChatResponse::default()
                }),
                Some(Err(error)) => Err(error),
                None => Err(Error::validation("the fake judge ran out of responses")),
            }
        }

        fn stream(
            &self,
            _messages: Vec<BaseMessage>,
            _request: ModelRequestConfig,
        ) -> BoxStream<'static, Result<StreamChunk>> {
            futures::stream::empty().boxed()
        }
    }
    #[tokio::test]
    async fn judge_retries_unparseable_responses_then_succeeds() {
        let model = FakeModel::new(vec![
            Ok(String::from("I cannot answer that.")),
            Ok(String::from(r#"{"score": 9, "reasoning": "correct"}"#)),
        ]);
        let judge = LlmJudge::new(
            model.clone(),
            JudgeRubric::default_rubric(),
            ModelRequestConfig::default(),
        )
        .unwrap();
        assert_eq!(judge.attempts(), DEFAULT_JUDGE_ATTEMPTS);
        assert_eq!(judge.max_prediction_chars(), DEFAULT_MAX_PREDICTION_CHARS);
        assert_eq!(judge.model().model_name(), "fake-judge");
        assert_eq!(judge.request().model_name, "");

        let verdict = judge.judge(&case(), &json!("4")).await.unwrap();
        assert!(close(verdict.score, 0.9));
        assert_eq!(model.calls(), 2);
    }

    #[tokio::test]
    async fn judge_reports_the_last_parse_error_and_propagates_model_errors() {
        let model = FakeModel::new(vec![Ok(String::from("nope")), Ok(String::from("nope"))]);
        let judge = LlmJudge::new(
            model.clone(),
            JudgeRubric::default_rubric(),
            ModelRequestConfig::default(),
        )
        .unwrap();
        let error = judge.judge(&case(), &json!("4")).await.unwrap_err();
        assert!(
            matches!(&error, Error::Validation(_)),
            "unexpected error: {error}"
        );
        assert_eq!(model.calls(), 2);

        let failing = FakeModel::new(vec![Err(Error::llm("provider down"))]);
        let judge = LlmJudge::new(
            failing.clone(),
            JudgeRubric::default_rubric(),
            ModelRequestConfig::default(),
        )
        .unwrap();
        let error = judge.judge(&case(), &json!("4")).await.unwrap_err();
        assert!(matches!(&error, Error::Llm(_)), "unexpected error: {error}");
        assert_eq!(failing.calls(), 1);
    }

    #[test]
    fn judge_builders_clamp_and_replace() {
        let model = FakeModel::new(Vec::new());
        let judge = LlmJudge::new(
            model,
            JudgeRubric::default_rubric(),
            ModelRequestConfig::new("judge-model"),
        )
        .unwrap();
        assert_eq!(judge.clone().with_attempts(0).attempts(), 1);
        assert_eq!(
            judge.clone().with_attempts(9).attempts(),
            MAX_JUDGE_ATTEMPTS
        );
        assert_eq!(
            judge
                .clone()
                .with_max_prediction_chars(12)
                .max_prediction_chars(),
            12
        );
        let error = LlmJudge::new(
            FakeModel::new(Vec::new()),
            JudgeRubric::new(Vec::new(), 10.0, 0.6),
            ModelRequestConfig::default(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("criterion"), "error: {error}");
    }

    #[tokio::test]
    async fn evaluator_scores_with_the_judge_and_breaks_down_criteria() {
        let model = FakeModel::new(vec![
            Ok(String::from(r#"{"score": 6, "reasoning": "partial"}"#)),
            Ok(String::from(
                r#"{"per_criterion": {"correctness": 10, "clarity": 5}, "reasoning": "fine"}"#,
            )),
        ]);
        let judge = Arc::new(
            LlmJudge::new(
                model,
                JudgeRubric::default_rubric(),
                ModelRequestConfig::default(),
            )
            .unwrap(),
        );
        let evaluator = LlmJudgeEvaluator::new().with_judge(judge);
        assert_eq!(evaluator.name(), "llm_as_judge");
        assert!(evaluator.judge().is_some());
        assert_eq!(evaluator.fallback().name(), "exact_match");
        assert_eq!(evaluator.rubric().criteria().len(), 3);

        let evaluated = evaluator.evaluate(&case(), &json!("4")).await.unwrap();
        assert!(close(evaluated.score, 0.6));
        assert_eq!(evaluated.reason, "partial");
        let per_metric = evaluated.per_metric.unwrap();
        assert!(close(per_metric["llm_as_judge"], 0.6));

        let evaluated = evaluator.evaluate(&case(), &json!("4")).await.unwrap();
        assert!(close(evaluated.score, 1.25 / 1.5));
        // `per_criterion` is a `BTreeMap`, so the breakdown is key-sorted.
        assert_eq!(evaluated.reason, "fine [clarity=0.5, correctness=1]");
        let per_metric = evaluated.per_metric.unwrap();
        assert!(close(per_metric["correctness"], 1.0));
        assert!(close(per_metric["llm_as_judge"], 1.25 / 1.5));
    }

    #[tokio::test]
    async fn evaluator_falls_back_explicitly_when_the_judge_fails() {
        let model = FakeModel::new(vec![
            Ok(String::from("no verdict")),
            Ok(String::from("still no verdict")),
        ]);
        let judge = Arc::new(
            LlmJudge::new(
                model,
                JudgeRubric::default_rubric(),
                ModelRequestConfig::default(),
            )
            .unwrap(),
        );
        let evaluator = LlmJudgeEvaluator::new().with_judge(judge);

        let evaluated = evaluator.evaluate(&case(), &json!("4")).await.unwrap();
        assert!(close(evaluated.score, 1.0));
        assert!(
            evaluated.reason.contains("the judge call failed")
                && evaluated.reason.contains("exact_match"),
            "reason: {}",
            evaluated.reason
        );
        let per_metric = evaluated.per_metric.unwrap();
        assert!(close(per_metric["exact_match"], 1.0));
    }

    #[tokio::test]
    async fn evaluator_without_a_judge_says_so() {
        let evaluator = LlmJudgeEvaluator::default();
        assert!(evaluator.judge().is_none());
        let evaluated = evaluator.evaluate(&case(), &json!("4")).await.unwrap();
        assert!(close(evaluated.score, 1.0));
        assert!(
            evaluated.reason.contains("no judge model is configured"),
            "reason: {}",
            evaluated.reason
        );

        let evaluated = evaluator.evaluate(&case(), &json!("5")).await.unwrap();
        assert!(close(evaluated.score, 0.0));
        assert_eq!(evaluator.fallback().name(), "exact_match");
        // The debug rendering names the model rather than an opaque `dyn`.
        let judge = LlmJudge::new(
            FakeModel::new(Vec::new()),
            JudgeRubric::default_rubric(),
            ModelRequestConfig::default(),
        )
        .unwrap();
        assert!(format!("{judge:?}").contains("fake-judge"));
        assert!(format!("{evaluator:?}").contains("exact_match"));
    }
}
