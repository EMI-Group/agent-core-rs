//! Model-backed fact extraction with a deterministic fallback.
//!
//! [`LlmExtractor`] sends one bounded conversation to a [`Model`] and parses the answer
//! with [`crate::analyzer::parse::parse_fact_response`]. Every failure mode — a provider
//! error, an empty answer, an unparseable answer — is retried up to
//! [`AnalyzerConfig::model_retries`] times; once the budget is spent the extractor falls
//! back to [`RuleExtractor`] and records that degradation on
//! [`ExtractionResult::degraded`] instead of failing the caller or pretending the model
//! worked. Cancellation is honored before the first call and around every call.
use crate::analyzer::config::AnalyzerConfig;
use crate::analyzer::fact::{ExtractionInput, ExtractionResult, FactExtractor};
use crate::analyzer::parse::parse_fact_response;
use crate::analyzer::prompt::{render_conversation, EXTRACTION_SYSTEM_PROMPT};
use crate::analyzer::rule::RuleExtractor;
use crate::record::CallOptions;
use async_trait::async_trait;
use openjiuwen_common::error::{ErrorKind, Result};
use openjiuwen_foundation::llm::{Model, ModelRequestConfig};
use openjiuwen_foundation::prompt::BaseMessage;
use std::sync::Arc;

/// Model-backed fact extractor.
pub struct LlmExtractor {
    /// The model that answers the extraction prompt.
    model: Arc<dyn Model>,
    /// Analyzer settings (limits, retries, importance tiers).
    config: AnalyzerConfig,
    /// Deterministic extractor used once the retry budget is spent.
    fallback: RuleExtractor,
}

impl LlmExtractor {
    /// Builds an extractor around `model` using `config`.
    ///
    /// The fallback is a [`RuleExtractor`] built from the same config;
    /// [`LlmExtractor::with_fallback`] replaces it when a caller tuned the rules
    /// separately.
    #[must_use]
    pub fn new(model: Arc<dyn Model>, config: AnalyzerConfig) -> Self {
        let fallback = RuleExtractor::new(config.clone());
        Self {
            model,
            config,
            fallback,
        }
    }

    /// Returns the extractor with `fallback` as its deterministic backstop.
    #[must_use]
    pub fn with_fallback(mut self, fallback: RuleExtractor) -> Self {
        self.fallback = fallback;
        self
    }

    /// Returns the configuration in use.
    #[must_use]
    pub fn config(&self) -> &AnalyzerConfig {
        &self.config
    }
}

#[async_trait]
impl FactExtractor for LlmExtractor {
    #[allow(clippy::unnecessary_literal_bound)] // the trait signature ties `&str` to `self`
    fn name(&self) -> &str {
        "llm"
    }

    async fn extract(
        &self,
        input: &ExtractionInput,
        options: &CallOptions,
    ) -> Result<ExtractionResult> {
        options.check()?;
        self.config.validate()?;
        input.validate()?;
        if !self.config.enabled {
            return Ok(ExtractionResult::empty(self.name()));
        }
        let bounded = bounded_input(input, &self.config);
        let messages = vec![
            BaseMessage::system(EXTRACTION_SYSTEM_PROMPT),
            BaseMessage::user(render_conversation(&bounded, self.config.max_input_chars)),
        ];
        let request = ModelRequestConfig::new(self.model.model_name().to_owned());
        let attempts = self.config.model_retries.saturating_add(1);
        // The first failure is kept: it is the root cause the retries started from.
        let mut failure: Option<String> = None;
        for attempt in 1..=attempts {
            match options.run(self.model.invoke(&messages, &request)).await {
                Ok(Ok(response)) => {
                    let raw = response
                        .message()
                        .map(|message| message.text().into_owned())
                        .unwrap_or_default();
                    if let Some(mut facts) = parse_fact_response(&raw, &self.config) {
                        facts.truncate(self.config.max_facts);
                        return Ok(ExtractionResult::new(facts, self.name()));
                    }
                    failure
                        .get_or_insert_with(|| "the model answer had no usable facts".to_owned());
                    tracing::warn!(
                        target: "openjiuwen_memory::analyzer",
                        attempt,
                        attempts,
                        "LLM fact extraction returned unusable output; retrying"
                    );
                }
                Ok(Err(error)) | Err(error) => {
                    if error.kind() == ErrorKind::Cancelled {
                        return Err(error);
                    }
                    failure.get_or_insert_with(|| format!("the model call failed: {error}"));
                    tracing::warn!(
                        target: "openjiuwen_memory::analyzer",
                        attempt,
                        attempts,
                        %error,
                        "LLM fact extraction call failed; retrying"
                    );
                }
            }
            options.check()?;
        }
        let failure = failure.unwrap_or_else(|| "no attempt was made".to_owned());
        tracing::warn!(
            target: "openjiuwen_memory::analyzer",
            attempts,
            %failure,
            "LLM fact extraction exhausted its retries; falling back to rules"
        );
        let mut result = self.fallback.extract(input, options).await?;
        self.fallback.name().clone_into(&mut result.extractor);
        result.degraded = Some(format!("llm extraction fell back to rules: {failure}"));
        Ok(result)
    }
}

/// Bounds the turn lengths and keeps only the newest history turns.
fn bounded_input(input: &ExtractionInput, config: &AnalyzerConfig) -> ExtractionInput {
    let mut bounded = input.bounded(config.max_input_chars);
    if bounded.history.len() > config.max_history_turns {
        let excess = bounded.history.len() - config.max_history_turns;
        bounded.history.drain(..excess);
    }
    bounded
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analyzer::fact::ConversationTurn;
    use crate::record::MemoryType;
    use openjiuwen_common::error::Error;
    use openjiuwen_test_support::{ScriptedError, ScriptedModel};
    use tokio_util::sync::CancellationToken;

    const FENCED_FACTS: &str = "Here you go:\n```json\n{\"facts\": [\
        {\"content\": \"Alice prefers dark mode\", \"type\": \"user_profile\", \"importance\": 0.8},\
        {\"content\": \"the service uses postgres\", \"type\": \"semantic_memory\"}]}\n```";

    fn extractor(model: ScriptedModel, config: AnalyzerConfig) -> LlmExtractor {
        LlmExtractor::new(Arc::new(model), config)
    }

    #[tokio::test]
    async fn parses_fenced_json_facts() {
        let model = ScriptedModel::builder().respond_text(FENCED_FACTS).build();
        let handle = model.clone();
        let result = extractor(model, AnalyzerConfig::default())
            .extract(
                &ExtractionInput::from_text("my name is Alice"),
                &CallOptions::new(),
            )
            .await
            .unwrap();
        assert!(!result.is_degraded());
        assert_eq!(result.extractor, "llm");
        assert_eq!(result.facts.len(), 2);
        assert_eq!(result.facts[0].content, "Alice prefers dark mode");
        assert_eq!(result.facts[0].kind, MemoryType::UserProfile);
        assert!((result.facts[0].importance - 0.8).abs() < f32::EPSILON);
        assert_eq!(handle.call_count(), 1);
        let call = handle.last_call().expect("the model was called");
        assert!(call.messages[0].text().contains("facts"));
        assert!(call
            .last_user_text()
            .is_some_and(|text| text.contains("my name is Alice")));
    }

    #[tokio::test]
    async fn caps_facts_at_max_facts() {
        let model = ScriptedModel::builder()
            .respond_text(
                r#"{"facts": [{"content": "one"}, {"content": "two"}, {"content": "three"}]}"#,
            )
            .build();
        let config = AnalyzerConfig {
            max_facts: 2,
            ..Default::default()
        };
        let result = extractor(model, config)
            .extract(&ExtractionInput::from_text("input"), &CallOptions::new())
            .await
            .unwrap();
        assert_eq!(result.facts.len(), 2);
        assert_eq!(result.facts[0].content, "one");
    }

    #[tokio::test]
    async fn malformed_output_falls_back_to_rules() {
        let model = ScriptedModel::builder()
            .respond_text("I think the user likes dark mode.")
            .build();
        let handle = model.clone();
        let result = extractor(model, AnalyzerConfig::default())
            .extract(
                &ExtractionInput::from_text("I prefer dark mode"),
                &CallOptions::new(),
            )
            .await
            .unwrap();
        assert!(result.is_degraded());
        assert_eq!(result.extractor, "rule");
        assert!(result
            .degraded
            .as_deref()
            .is_some_and(|reason| reason.contains("fell back to rules")));
        assert_eq!(result.facts.len(), 1);
        assert_eq!(result.facts[0].content, "I prefer dark mode");
        assert_eq!(handle.call_count(), 4);
    }

    #[tokio::test]
    async fn empty_script_falls_back_to_rules() {
        let model = ScriptedModel::builder().build();
        let handle = model.clone();
        let result = extractor(model, AnalyzerConfig::default())
            .extract(
                &ExtractionInput::from_text("my name is Alice"),
                &CallOptions::new(),
            )
            .await
            .unwrap();
        assert!(result.is_degraded());
        assert_eq!(result.extractor, "rule");
        assert_eq!(result.facts[0].content, "my name is Alice");
        assert_eq!(handle.call_count(), 4);
    }

    #[tokio::test]
    async fn model_errors_are_retried_and_then_degrade() {
        let model = ScriptedModel::builder()
            .fail(ScriptedError::llm("provider is down"))
            .build();
        let handle = model.clone();
        let config = AnalyzerConfig {
            model_retries: 1,
            ..Default::default()
        };
        let result = extractor(model, config)
            .extract(
                &ExtractionInput::from_text("I prefer dark mode"),
                &CallOptions::new(),
            )
            .await
            .unwrap();
        assert!(result.is_degraded());
        assert_eq!(result.extractor, "rule");
        assert!(result
            .degraded
            .as_deref()
            .is_some_and(|reason| reason.contains("provider is down")));
        assert_eq!(handle.call_count(), 2);
    }

    #[tokio::test]
    async fn a_retry_can_still_succeed() {
        let model = ScriptedModel::builder()
            .fail(ScriptedError::llm("transient"))
            .respond_text(FENCED_FACTS)
            .build();
        let handle = model.clone();
        let result = extractor(model, AnalyzerConfig::default())
            .extract(
                &ExtractionInput::from_text("my name is Alice"),
                &CallOptions::new(),
            )
            .await
            .unwrap();
        assert!(!result.is_degraded());
        assert_eq!(result.extractor, "llm");
        assert_eq!(handle.call_count(), 2);
    }

    #[tokio::test]
    async fn cancellation_returns_early_without_calling_the_model() {
        let model = ScriptedModel::builder().respond_text(FENCED_FACTS).build();
        let handle = model.clone();
        let cancellation = CancellationToken::new();
        cancellation.cancel();
        let options = CallOptions::new().with_cancellation(cancellation);
        let error = extractor(model, AnalyzerConfig::default())
            .extract(&ExtractionInput::from_text("my name is Alice"), &options)
            .await
            .unwrap_err();
        assert!(matches!(error, Error::Cancelled));
        assert_eq!(handle.call_count(), 0);
    }

    #[tokio::test]
    async fn disabled_config_skips_the_model() {
        let model = ScriptedModel::builder().respond_text(FENCED_FACTS).build();
        let handle = model.clone();
        let config = AnalyzerConfig {
            enabled: false,
            ..Default::default()
        };
        let result = extractor(model, config)
            .extract(
                &ExtractionInput::from_text("my name is Alice"),
                &CallOptions::new(),
            )
            .await
            .unwrap();
        assert_eq!(result.facts.len(), 0);
        assert!(!result.is_degraded());
        assert_eq!(handle.call_count(), 0);
    }

    #[tokio::test]
    async fn invalid_config_and_input_are_validation_errors() {
        let model = ScriptedModel::builder().respond_text(FENCED_FACTS).build();
        let handle = model.clone();
        let config = AnalyzerConfig {
            max_facts: 0,
            ..Default::default()
        };
        let error = extractor(model.clone(), config)
            .extract(&ExtractionInput::from_text("text"), &CallOptions::new())
            .await
            .unwrap_err();
        assert!(matches!(error, Error::Validation(_)));

        let input = ExtractionInput {
            turns: vec![ConversationTurn::new("", "text")],
            history: Vec::new(),
        };
        let error = extractor(model, AnalyzerConfig::default())
            .extract(&input, &CallOptions::new())
            .await
            .unwrap_err();
        assert!(matches!(error, Error::Validation(_)));
        assert_eq!(handle.call_count(), 0);
    }

    #[test]
    fn bounded_input_keeps_the_newest_history_turns() {
        let config = AnalyzerConfig {
            max_history_turns: 1,
            max_input_chars: 16,
            ..Default::default()
        };
        let input = ExtractionInput {
            turns: vec![ConversationTurn::new("user", "a".repeat(100))],
            history: vec![
                ConversationTurn::new("user", "first"),
                ConversationTurn::new("user", "second"),
            ],
        };
        let bounded = bounded_input(&input, &config);
        assert_eq!(bounded.history.len(), 1);
        assert_eq!(bounded.history[0].text, "second");
        assert_eq!(bounded.turns[0].text.chars().count(), 16);
    }

    #[tokio::test]
    async fn the_fallback_can_be_replaced() {
        let model = ScriptedModel::builder().build();
        let fallback_config = AnalyzerConfig {
            max_facts: 1,
            ..Default::default()
        };
        let llm = extractor(model, AnalyzerConfig::default())
            .with_fallback(RuleExtractor::new(fallback_config));
        assert_eq!(llm.config().max_facts, 8);
        let result = llm
            .extract(
                &ExtractionInput::from_text("my name is Alice. I prefer dark mode."),
                &CallOptions::new(),
            )
            .await
            .unwrap();
        assert_eq!(result.facts.len(), 1);
        assert_eq!(FactExtractor::name(&llm), "llm");
    }
}
