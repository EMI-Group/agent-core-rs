//! Model-backed conflict classification with a deterministic fallback.
//!
//! [`LlmClassifier`] is the Rust counterpart of Python's `MemUpdateChecker`: it sends
//! the incoming memory plus up to [`AnalyzerConfig::max_candidates`] existing records to
//! a [`Model`] and parses the strict-JSON answer with
//! [`crate::analyzer::parse::parse_conflict_response`]. A provider error or an
//! unusable answer is retried; once the retry budget is spent the classifier runs
//! [`RuleClassifier`] and stamps `degraded` on every verdict, so a caller can tell a
//! rule-based decision from a model-based one instead of silently getting the fallback.
use crate::analyzer::config::AnalyzerConfig;
use crate::analyzer::conflict::{ConflictClassifier, ConflictVerdict, RuleClassifier};
use crate::analyzer::parse::parse_conflict_response;
use crate::analyzer::prompt::{render_conflict_input, CONFLICT_SYSTEM_PROMPT};
use crate::config::DedupeConfig;
use crate::record::{MemoryRecord, NewMemory};
use async_trait::async_trait;
use openjiuwen_common::error::{ErrorKind, Result};
use openjiuwen_foundation::llm::{Model, ModelRequestConfig};
use openjiuwen_foundation::prompt::BaseMessage;
use std::sync::Arc;

/// Model-backed conflict classifier.
pub struct LlmClassifier {
    /// The model that answers the classification prompt.
    model: Arc<dyn Model>,
    /// Analyzer settings (candidate cap, retries, input limit).
    config: AnalyzerConfig,
    /// Deterministic classifier used once the retry budget is spent.
    fallback: RuleClassifier,
}

impl LlmClassifier {
    /// Builds a classifier around `model` using `config`.
    ///
    /// The fallback is a [`RuleClassifier`] built from the same config;
    /// [`LlmClassifier::with_fallback`] replaces it when a caller tuned the rules
    /// separately.
    #[must_use]
    pub fn new(model: Arc<dyn Model>, config: AnalyzerConfig) -> Self {
        let fallback = RuleClassifier::new(config.clone());
        Self {
            model,
            config,
            fallback,
        }
    }

    /// Returns the classifier with `fallback` as its deterministic backstop.
    #[must_use]
    pub fn with_fallback(mut self, fallback: RuleClassifier) -> Self {
        self.fallback = fallback;
        self
    }

    /// Returns the configuration in use.
    #[must_use]
    pub fn config(&self) -> &AnalyzerConfig {
        &self.config
    }

    /// Runs the deterministic classifier and marks every verdict as degraded.
    async fn fall_back(
        &self,
        incoming: &NewMemory,
        existing: &[MemoryRecord],
        dedupe: &DedupeConfig,
        failure: &str,
    ) -> Result<Vec<ConflictVerdict>> {
        let mut verdicts = self.fallback.classify(incoming, existing, dedupe).await?;
        let reason = format!("llm classification fell back to rules: {failure}");
        for verdict in &mut verdicts {
            verdict.degraded = Some(reason.clone());
        }
        Ok(verdicts)
    }
}

#[async_trait]
impl ConflictClassifier for LlmClassifier {
    #[allow(clippy::unnecessary_literal_bound)] // the trait signature ties `&str` to `self`
    fn name(&self) -> &str {
        "llm"
    }

    async fn classify(
        &self,
        incoming: &NewMemory,
        existing: &[MemoryRecord],
        config: &DedupeConfig,
    ) -> Result<Vec<ConflictVerdict>> {
        self.config.validate()?;
        incoming.validate()?;
        if existing.is_empty() {
            return Ok(Vec::new());
        }
        let candidates: Vec<MemoryRecord> = existing
            .iter()
            .take(self.config.max_candidates)
            .cloned()
            .collect();
        let messages = vec![
            BaseMessage::system(CONFLICT_SYSTEM_PROMPT),
            BaseMessage::user(render_conflict_input(
                incoming,
                &candidates,
                self.config.max_input_chars,
            )),
        ];
        let request = ModelRequestConfig::new(self.model.model_name().to_owned());
        let attempts = self.config.model_retries.saturating_add(1);
        // The first failure is kept: it is the root cause the retries started from.
        let mut failure: Option<String> = None;
        for attempt in 1..=attempts {
            match self.model.invoke(&messages, &request).await {
                Ok(response) => {
                    let raw = response
                        .message()
                        .map(|message| message.text().into_owned())
                        .unwrap_or_default();
                    if let Some(mut verdicts) = parse_conflict_response(&raw, &candidates) {
                        // Candidates beyond the cap keep a default (unrelated) verdict so
                        // the caller still gets one verdict per supplied record.
                        verdicts.resize(existing.len(), ConflictVerdict::default());
                        return Ok(verdicts);
                    }
                    failure.get_or_insert_with(|| {
                        "the model answer had no usable verdicts".to_owned()
                    });
                    tracing::warn!(
                        target: "openjiuwen_memory::analyzer",
                        attempt,
                        attempts,
                        "LLM conflict classification returned unusable output; retrying"
                    );
                }
                Err(error) => {
                    if error.kind() == ErrorKind::Cancelled {
                        return Err(error);
                    }
                    failure.get_or_insert_with(|| format!("the model call failed: {error}"));
                    tracing::warn!(
                        target: "openjiuwen_memory::analyzer",
                        attempt,
                        attempts,
                        %error,
                        "LLM conflict classification call failed; retrying"
                    );
                }
            }
        }
        let failure = failure.unwrap_or_else(|| "no attempt was made".to_owned());
        tracing::warn!(
            target: "openjiuwen_memory::analyzer",
            attempts,
            %failure,
            "LLM conflict classification exhausted its retries; falling back to rules"
        );
        self.fall_back(incoming, existing, config, &failure).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analyzer::conflict::ConflictKind;
    use crate::record::{MemoryScope, MemoryType};
    use openjiuwen_common::error::Error;
    use openjiuwen_test_support::{ScriptedError, ScriptedModel};

    fn record(id: &str, content: &str) -> MemoryRecord {
        MemoryRecord::new(
            MemoryScope::new("alice", "ws"),
            content,
            MemoryType::SemanticMemory,
        )
        .with_id(id)
    }

    fn classifier(model: ScriptedModel, config: AnalyzerConfig) -> LlmClassifier {
        LlmClassifier::new(Arc::new(model), config)
    }

    #[tokio::test]
    async fn parses_scripted_verdicts() {
        let model = ScriptedModel::builder()
            .respond_text(
                r#"```json
                {"verdicts": [
                    {"info_id": "id-1", "result": "redundant", "reason": "same fact"},
                    {"info_text": "Alice works at Globex", "result": "conflicting",
                     "related_infos": {"id-2": "Alice works at Globex"},
                     "merged_content": "Alice works at Acme"}
                ]}
                ```"#,
            )
            .build();
        let handle = model.clone();
        let existing = vec![
            record("id-1", "Alice prefers dark mode"),
            record("id-2", "Alice works at Globex"),
        ];
        let verdicts = classifier(model, AnalyzerConfig::default())
            .classify(
                &NewMemory::semantic("Alice prefers dark mode"),
                &existing,
                &DedupeConfig::default(),
            )
            .await
            .unwrap();
        assert_eq!(verdicts.len(), 2);
        assert_eq!(verdicts[0].kind, ConflictKind::Redundant);
        assert_eq!(verdicts[0].reason.as_deref(), Some("same fact"));
        assert!(verdicts[0].degraded.is_none());
        assert_eq!(verdicts[1].kind, ConflictKind::Conflicting);
        assert_eq!(verdicts[1].target_id.as_deref(), Some("id-2"));
        assert_eq!(handle.call_count(), 1);
        let call = handle.last_call().expect("the model was called");
        assert!(call.messages[0].text().contains("redundant"));
        let user = call.last_user_text().expect("a user message");
        assert!(user.contains("id-1"));
        assert!(user.contains("id-2"));
    }

    #[tokio::test]
    async fn malformed_output_falls_back_with_degraded_verdicts() {
        let model = ScriptedModel::builder()
            .respond_text("The memories look fine to me.")
            .build();
        let handle = model.clone();
        let existing = vec![record("id-1", "Alice prefers dark mode")];
        let verdicts = classifier(model, AnalyzerConfig::default())
            .classify(
                &NewMemory::semantic("alice prefers DARK mode"),
                &existing,
                &DedupeConfig::default(),
            )
            .await
            .unwrap();
        assert_eq!(verdicts.len(), 1);
        assert_eq!(verdicts[0].kind, ConflictKind::Redundant);
        let degraded = verdicts[0].degraded.as_deref().unwrap_or_default();
        assert!(degraded.contains("fell back to rules"), "{degraded}");
        assert_eq!(handle.call_count(), 4);
    }

    #[tokio::test]
    async fn empty_script_falls_back_to_rules() {
        let model = ScriptedModel::builder().build();
        let handle = model.clone();
        let existing = vec![record("id-1", "the service uses postgres")];
        let verdicts = classifier(model, AnalyzerConfig::default())
            .classify(
                &NewMemory::semantic("the service uses mysql"),
                &existing,
                &DedupeConfig::default(),
            )
            .await
            .unwrap();
        assert_eq!(verdicts.len(), 1);
        assert_eq!(verdicts[0].kind, ConflictKind::Unrelated);
        assert!(verdicts[0].degraded.is_some());
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
        let verdicts = classifier(model, config)
            .classify(
                &NewMemory::semantic("x"),
                &[record("id-1", "y")],
                &DedupeConfig::default(),
            )
            .await
            .unwrap();
        assert_eq!(handle.call_count(), 2);
        let degraded = verdicts[0].degraded.as_deref().unwrap_or_default();
        assert!(degraded.contains("provider is down"), "{degraded}");
    }

    #[tokio::test]
    async fn a_retry_can_still_succeed() {
        let model = ScriptedModel::builder()
            .fail(ScriptedError::llm("transient"))
            .respond_text(r#"[{"info_id": "id-1", "result": "none"}]"#)
            .build();
        let handle = model.clone();
        let verdicts = classifier(model, AnalyzerConfig::default())
            .classify(
                &NewMemory::semantic("x"),
                &[record("id-1", "y")],
                &DedupeConfig::default(),
            )
            .await
            .unwrap();
        assert_eq!(handle.call_count(), 2);
        assert_eq!(verdicts[0].kind, ConflictKind::Unrelated);
        assert!(verdicts[0].degraded.is_none());
    }

    #[tokio::test]
    async fn caps_the_candidates_sent_to_the_model() {
        let model = ScriptedModel::builder()
            .respond_text(r#"[{"info_id": "id-1", "result": "none"}]"#)
            .build();
        let handle = model.clone();
        let config = AnalyzerConfig {
            max_candidates: 1,
            ..Default::default()
        };
        let existing = vec![
            record("id-1", "first"),
            record("id-2", "second"),
            record("id-3", "third"),
        ];
        let verdicts = classifier(model, config)
            .classify(
                &NewMemory::semantic("x"),
                &existing,
                &DedupeConfig::default(),
            )
            .await
            .unwrap();
        assert_eq!(verdicts.len(), 3);
        assert_eq!(verdicts[0].kind, ConflictKind::Unrelated);
        assert_eq!(verdicts[1].kind, ConflictKind::Unrelated);
        assert_eq!(verdicts[2].kind, ConflictKind::Unrelated);
        let user = handle
            .last_call()
            .and_then(|call| call.last_user_text())
            .unwrap_or_default();
        assert!(user.contains("id-1"));
        assert!(!user.contains("id-2"));
        assert!(!user.contains("id-3"));
    }

    #[tokio::test]
    async fn no_existing_records_means_no_model_call() {
        let model = ScriptedModel::builder().build();
        let handle = model.clone();
        let verdicts = classifier(model, AnalyzerConfig::default())
            .classify(&NewMemory::semantic("x"), &[], &DedupeConfig::default())
            .await
            .unwrap();
        assert_eq!(verdicts.len(), 0);
        assert_eq!(handle.call_count(), 0);
    }

    #[tokio::test]
    async fn invalid_input_and_config_are_validation_errors() {
        let model = ScriptedModel::builder().build();
        let handle = model.clone();
        let error = classifier(model.clone(), AnalyzerConfig::default())
            .classify(
                &NewMemory::semantic("  "),
                &[record("id-1", "y")],
                &DedupeConfig::default(),
            )
            .await
            .unwrap_err();
        assert!(matches!(error, Error::Validation(_)));

        let config = AnalyzerConfig {
            max_candidates: 0,
            ..Default::default()
        };
        let error = classifier(model, config)
            .classify(
                &NewMemory::semantic("x"),
                &[record("id-1", "y")],
                &DedupeConfig::default(),
            )
            .await
            .unwrap_err();
        assert!(matches!(error, Error::Validation(_)));
        assert_eq!(handle.call_count(), 0);
    }

    #[test]
    fn the_classifier_names_are_distinct() {
        let model = ScriptedModel::builder().build();
        let llm = classifier(model, AnalyzerConfig::default());
        assert_eq!(ConflictClassifier::name(&llm), "llm");
        assert_eq!(
            ConflictClassifier::name(&RuleClassifier::new(AnalyzerConfig::default())),
            "rule"
        );
        assert_eq!(llm.config().max_candidates, 8);
    }
}
