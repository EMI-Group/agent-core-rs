//! Bounded concurrent proposer fan-out over the configured reference models.
//!
//! Ports the fan-out half of `openjiuwen/harness/moa/consultation.py`: every configured
//! reference model receives the same read-only transcript and answers independently, so
//! their proposals do not influence each other. Responses are collected in *configured*
//! order regardless of completion order, capped by the per-response and total advice
//! budgets, and a partial failure never sinks the batch — only a batch in which every
//! reference model failed is an error.
//!
//! The fan-out is bounded by [`MoaConfig::max_concurrency`], every request is raced against
//! [`MoaConfig::reference_timeout`] and the caller's [`CancellationToken`], and a cancelled
//! consultation surfaces as [`Error::Cancelled`] instead of an empty result. No network
//! path lives here: the models are injected handles, so the module stays offline-testable
//! with scripted models.
// The public names mirror the Python contract and intentionally repeat the module name.
#![allow(clippy::module_name_repetitions)]

use std::sync::Arc;
use std::time::Duration;

use futures::stream::{self, StreamExt};
use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::llm::ModelRequestConfig;
use openjiuwen_foundation::{BaseMessage, Model, Role};
use tokio_util::sync::CancellationToken;

use super::config::MoaConfig;
use super::prompts::{consultation_transcript, reference_messages};
use super::state::{ProposalSource, ProposerFailure};

/// Name of the tool/action that exposed the consultation to the model.
///
/// It is the Python `consult_reference_models` name. [`consultation_transcript`] strips
/// the calls and results carrying this name so a consultation never quotes itself.
pub const CONSULT_TOOL_NAME: &str = "consult_reference_models";

/// One consultation batch: the retained proposals and the per-reference diagnostics.
///
/// `sources` holds only the proposals that survived both character budgets, in configured
/// order; `success_count` counts every reference that answered non-empty, including the
/// ones dropped by the total advice budget. `failures` records the references that timed
/// out, errored or answered with nothing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Consultation {
    /// Retained proposals in configured order.
    pub sources: Vec<ProposalSource>,
    /// How many reference models produced a usable proposal.
    pub success_count: usize,
    /// How many reference models were actually asked.
    pub total_models: usize,
    /// References that produced no usable proposal, in configured order.
    pub failures: Vec<ProposerFailure>,
}

impl Consultation {
    /// Builds a batch from its parts.
    #[must_use]
    pub fn new(
        sources: Vec<ProposalSource>,
        success_count: usize,
        total_models: usize,
        failures: Vec<ProposerFailure>,
    ) -> Self {
        Self {
            sources,
            success_count,
            total_models,
            failures,
        }
    }

    /// `true` when no proposal survived the character budgets.
    ///
    /// A nonempty batch can have a zero `sources` length only with a zero advice budget,
    /// which [`MoaConfig::validate`] rules out.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.sources.is_empty()
    }
}

/// What one proposer produced.
#[derive(Debug)]
enum ProposerOutcome {
    /// A non-empty proposal for the reference at `index`.
    Proposed {
        /// One-based position of the reference model in the configuration.
        index: usize,
        /// The raw response text, before the character budgets are applied.
        text: String,
    },
    /// The reference produced nothing usable.
    Failed(ProposerFailure),
    /// The caller's token fired while the request was in flight.
    Cancelled,
}

/// Asks every configured reference model for an independent proposal.
///
/// The transcript is the read-only flattening of `messages` produced by
/// [`consultation_transcript`]. When that flattening is empty, the last user message is
/// used instead so a conversation that only carries system messages still gets a request.
///
/// Proposers run concurrently with at most [`MoaConfig::max_concurrency`] requests in
/// flight; at most [`MoaConfig::proposal_count`] models are asked.
///
/// # Errors
///
/// - [`Error::Config`] when [`MoaConfig::reference_models`] is empty.
/// - [`Error::Cancelled`] when `cancel` fires before every proposer finished.
/// - [`Error::Llm`] when no reference model produced a proposal; the message starts with
///   `All reference models failed` and carries the first failure detail.
pub async fn consult(
    config: &MoaConfig,
    messages: &[BaseMessage],
    cancel: &CancellationToken,
) -> Result<Consultation> {
    let mut transcript = consultation_transcript(messages);
    if transcript.trim().is_empty() {
        transcript = last_user_text(messages).unwrap_or_default();
    }
    consult_transcript(config, &transcript, cancel).await
}

/// Asks every configured reference model for an independent proposal for `problem`.
///
/// Identical to [`consult`] except that a transcript without any user/assistant/tool text
/// falls back to `problem` rather than to the last user message. This is the entry point
/// for hosts that consult references on a stated problem instead of a live conversation.
///
/// # Errors
///
/// The same cases as [`consult`].
pub async fn consult_problem(
    config: &MoaConfig,
    problem: &str,
    messages: &[BaseMessage],
    cancel: &CancellationToken,
) -> Result<Consultation> {
    let mut transcript = consultation_transcript(messages);
    if transcript.trim().is_empty() {
        transcript = problem.to_owned();
    }
    consult_transcript(config, &transcript, cancel).await
}

/// Runs the bounded fan-out for an already flattened `transcript`.
async fn consult_transcript(
    config: &MoaConfig,
    transcript: &str,
    cancel: &CancellationToken,
) -> Result<Consultation> {
    if config.reference_models.is_empty() {
        return Err(Error::config("MoA requires at least one reference model"));
    }

    let request_messages = reference_messages(transcript);
    let total_models = config.proposal_count();
    let temperature = config.proposer_temperature;
    let timeout = config.reference_timeout;
    let concurrency = config.max_concurrency.max(1);

    let outcomes: Vec<ProposerOutcome> = stream::iter(
        config
            .reference_models
            .iter()
            .take(total_models)
            .cloned()
            .enumerate()
            .map(|(offset, model)| {
                let request_messages = request_messages.clone();
                async move {
                    ask(
                        offset + 1,
                        model,
                        &request_messages,
                        temperature,
                        timeout,
                        cancel,
                    )
                    .await
                }
            }),
    )
    .buffer_unordered(concurrency)
    .collect()
    .await;

    let mut successes: Vec<(usize, String)> = Vec::with_capacity(outcomes.len());
    let mut failures: Vec<ProposerFailure> = Vec::new();
    let mut cancelled = false;
    for outcome in outcomes {
        match outcome {
            ProposerOutcome::Proposed { index, text } => successes.push((index, text)),
            ProposerOutcome::Failed(failure) => failures.push(failure),
            ProposerOutcome::Cancelled => cancelled = true,
        }
    }
    if cancelled {
        return Err(Error::Cancelled);
    }

    // Sort after collection: `buffer_unordered` yields completion order, while the Python
    // contract (and every consumer of the batch) expects configured order.
    successes.sort_by_key(|(index, _)| *index);
    failures.sort_by_key(|failure| failure.index);

    let success_count = successes.len();
    let mut remaining = config.max_advice_chars;
    let mut sources = Vec::with_capacity(successes.len());
    for (index, text) in successes {
        let capped: String = text.chars().take(config.max_reference_chars).collect();
        if remaining == 0 {
            continue;
        }
        let kept: String = capped.chars().take(remaining).collect();
        remaining -= kept.chars().count();
        sources.push(ProposalSource::new(index, kept));
    }

    if success_count == 0 {
        let detail = failures
            .first()
            .map_or("no reference model produced a proposal", |failure| {
                failure.error.as_str()
            });
        return Err(Error::llm(format!("All reference models failed: {detail}")));
    }

    Ok(Consultation::new(
        sources,
        success_count,
        total_models,
        failures,
    ))
}

/// Asks one reference model, folding timeout, provider error, empty text and cancellation
/// into a [`ProposerOutcome`].
async fn ask(
    index: usize,
    model: Arc<dyn Model>,
    messages: &[BaseMessage],
    temperature: Option<f32>,
    timeout: Duration,
    cancel: &CancellationToken,
) -> ProposerOutcome {
    let mut request = ModelRequestConfig::new(model.model_name());
    if let Some(temperature) = temperature {
        request = request.with_temperature(temperature);
    }

    // `biased` makes the cancellation branch win over an already-ready response, so a
    // stopped consultation never reports proposals that raced the stop.
    let attempt = tokio::select! {
        biased;
        () = cancel.cancelled() => return ProposerOutcome::Cancelled,
        result = tokio::time::timeout(timeout, model.invoke(messages, &request)) => result,
    };

    match attempt {
        Err(_elapsed) => {
            tracing::debug!(
                index,
                model = model.model_name(),
                "reference proposer timed out"
            );
            ProposerOutcome::Failed(ProposerFailure::new(index, "timeout"))
        }
        Ok(Err(error)) => {
            tracing::debug!(
                index,
                model = model.model_name(),
                error = %error,
                "reference proposer failed"
            );
            ProposerOutcome::Failed(ProposerFailure::new(index, error.to_string()))
        }
        Ok(Ok(response)) => {
            let text = response
                .message()
                .map_or_else(String::new, |message| message.text().into_owned());
            if text.trim().is_empty() {
                tracing::debug!(
                    index,
                    model = model.model_name(),
                    "reference proposer returned an empty response"
                );
                ProposerOutcome::Failed(ProposerFailure::new(index, "empty response"))
            } else {
                tracing::debug!(
                    index,
                    model = model.model_name(),
                    chars = text.chars().count(),
                    "reference proposer answered"
                );
                ProposerOutcome::Proposed { index, text }
            }
        }
    }
}

/// Text of the last user message, used when the flattened transcript is empty.
fn last_user_text(messages: &[BaseMessage]) -> Option<String> {
    messages
        .iter()
        .rev()
        .find(|message| message.role == Role::User)
        .map(|message| message.text().into_owned())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use futures::stream::BoxStream;
    use openjiuwen_common::error::ErrorKind;
    use openjiuwen_foundation::llm::{ChatResponse, StreamChunk};
    use openjiuwen_test_support::{ScriptedModel, ScriptedTurn};

    use super::*;

    fn reference(model: ScriptedModel) -> Arc<dyn Model> {
        Arc::new(model)
    }

    fn user(text: &str) -> BaseMessage {
        BaseMessage::user(text)
    }

    #[tokio::test]
    async fn fan_out_preserves_configured_order() {
        let config = MoaConfig::new(vec![
            reference(ScriptedModel::text("first")),
            reference(ScriptedModel::text("second")),
        ])
        .with_max_concurrency(2);

        let result = consult(
            &config,
            &[user("plan the change")],
            &CancellationToken::new(),
        )
        .await
        .unwrap();

        assert_eq!(result.total_models, 2);
        assert_eq!(result.success_count, 2);
        assert_eq!(result.failures, Vec::new());
        assert!(!result.is_empty());
        assert_eq!(
            result.sources,
            vec![
                ProposalSource::new(1, "first"),
                ProposalSource::new(2, "second"),
            ]
        );
    }

    #[tokio::test]
    async fn partial_failure_is_recorded_without_sinking_the_batch() {
        let config = MoaConfig::new(vec![
            reference(ScriptedModel::new([ScriptedTurn::fail("boom")])),
            reference(ScriptedModel::text("recovered")),
        ]);

        let result = consult(&config, &[user("q")], &CancellationToken::new())
            .await
            .unwrap();

        assert_eq!(result.total_models, 2);
        assert_eq!(result.success_count, 1);
        assert_eq!(result.failures.len(), 1);
        assert_eq!(result.failures[0].index, 1);
        assert!(result.failures[0].error.contains("boom"));
        assert_eq!(result.sources, vec![ProposalSource::new(2, "recovered")]);
    }

    #[tokio::test]
    async fn every_proposer_failing_is_an_llm_error() {
        let config = MoaConfig::new(vec![
            reference(ScriptedModel::new([ScriptedTurn::fail("first boom")])),
            reference(ScriptedModel::new([ScriptedTurn::fail("second boom")])),
        ]);

        let error = consult(&config, &[user("q")], &CancellationToken::new())
            .await
            .unwrap_err();

        assert_eq!(error.kind(), ErrorKind::Llm);
        let message = error.to_string();
        assert!(message.contains("All reference models failed"));
        assert!(message.contains("first boom"));
    }

    #[tokio::test]
    async fn reference_and_advice_budgets_truncate_in_configured_order() {
        let config = MoaConfig::new(vec![
            reference(ScriptedModel::text("abcdefgh")),
            reference(ScriptedModel::text("ABCDEFGH")),
        ])
        .with_max_reference_chars(4)
        .with_max_advice_chars(6);

        let result = consult(&config, &[user("q")], &CancellationToken::new())
            .await
            .unwrap();

        // Each proposal is capped at four characters, then the six-character advice
        // budget stops the second retained proposal at two.
        assert_eq!(
            result.sources,
            vec![ProposalSource::new(1, "abcd"), ProposalSource::new(2, "AB")]
        );
        // Both references still answered successfully; only the retention was trimmed.
        assert_eq!(result.success_count, 2);
    }

    #[tokio::test]
    async fn cancelled_token_aborts_the_fan_out() {
        let cancel = CancellationToken::new();
        cancel.cancel();
        let config = MoaConfig::new(vec![reference(ScriptedModel::text("unused"))]);

        let error = consult(&config, &[user("q")], &cancel).await.unwrap_err();

        assert_eq!(error.kind(), ErrorKind::Cancelled);
    }

    #[tokio::test]
    async fn empty_reference_list_is_a_config_error() {
        let error = consult(
            &MoaConfig::new(Vec::new()),
            &[user("q")],
            &CancellationToken::new(),
        )
        .await
        .unwrap_err();

        assert_eq!(error.kind(), ErrorKind::Config);
        assert!(error
            .to_string()
            .contains("MoA requires at least one reference model"));
    }

    #[tokio::test]
    async fn consult_problem_falls_back_to_the_problem_text() {
        let model = ScriptedModel::text("proposal");
        let config = MoaConfig::new(vec![reference(model.clone())]);

        let result = consult_problem(&config, "why is CI red?", &[], &CancellationToken::new())
            .await
            .unwrap();

        assert_eq!(result.success_count, 1);
        let calls = model.calls();
        assert_eq!(calls.len(), 1);
        // [system(REFERENCE_SYSTEM_PROMPT), user(transcript-or-problem)]
        assert_eq!(calls[0].messages.len(), 2);
        assert_eq!(calls[0].messages[1].text(), "why is CI red?");
    }

    #[tokio::test]
    async fn slow_reference_is_recorded_as_a_timeout_failure() {
        let config = MoaConfig::new(vec![
            Arc::new(SlowModel {
                name: "slow".to_owned(),
            }),
            reference(ScriptedModel::text("quick")),
        ])
        .with_reference_timeout(Duration::from_millis(1));

        let result = consult(&config, &[user("q")], &CancellationToken::new())
            .await
            .unwrap();

        assert_eq!(result.success_count, 1);
        assert_eq!(result.failures.len(), 1);
        assert_eq!(result.failures[0].index, 1);
        assert_eq!(result.failures[0].error, "timeout");
        assert_eq!(result.sources, vec![ProposalSource::new(2, "quick")]);
    }

    /// A reference model that answers only after the configured deadline.
    struct SlowModel {
        name: String,
    }

    #[async_trait::async_trait]
    impl Model for SlowModel {
        fn provider(&self) -> &'static str {
            "scripted"
        }

        fn model_name(&self) -> &str {
            &self.name
        }

        async fn invoke(
            &self,
            _messages: &[BaseMessage],
            _request: &ModelRequestConfig,
        ) -> Result<ChatResponse> {
            tokio::time::sleep(Duration::from_millis(200)).await;
            Ok(ChatResponse::default())
        }

        fn stream(
            &self,
            _messages: Vec<BaseMessage>,
            _request: ModelRequestConfig,
        ) -> BoxStream<'static, Result<StreamChunk>> {
            futures::stream::empty().boxed()
        }
    }
}
