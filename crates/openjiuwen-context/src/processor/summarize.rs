//! Summarisation processor: replace old rounds with one model-written summary.
//!
//! The processor summarises every *complete* dialogue round except the newest
//! [`SummarizationConfig::keep_last_rounds`] rounds, and leaves system messages
//! plus the newest [`SummarizationConfig::keep_recent_messages`] messages alone.
//! The selected messages are rendered as a plain transcript, summarised by the
//! configured model (or the per-request model when one is supplied), and replaced
//! by a single `[CONTEXT_SUMMARY]` user message.
//!
//! The replacement is built in a temporary vector, so a failing model call never
//! corrupts the history.
use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::llm::{Model, ModelRequestConfig};
use openjiuwen_foundation::prompt::{BaseMessage, Role};
use serde::{Deserialize, Serialize};

use crate::events::{ContextEvent, ContextEventKind};
use crate::processor::{ContextProcessor, ProcessorContext};
use crate::prompts::{render_summary_prompt, SUMMARY_SYSTEM_PROMPT};
use crate::tokens::count_messages;
use crate::window::dialogue_round_ranges;

/// Prefix of the message that replaces a summarised span.
pub const CONTEXT_SUMMARY_PREFIX: &str = "[CONTEXT_SUMMARY]";

/// Configuration of the summarisation processor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct SummarizationConfig {
    /// Token threshold above which summarisation runs; `None` uses the request
    /// trigger.
    pub tokens_threshold: Option<usize>,
    /// Newest messages that are never summarised.
    pub keep_recent_messages: usize,
    /// Newest rounds that are never summarised.
    pub keep_last_rounds: usize,
    /// Minimum number of messages a summary must cover.
    pub min_messages_to_summarize: usize,
    /// Token budget for the summary request.
    pub max_summary_tokens: u32,
}

impl Default for SummarizationConfig {
    fn default() -> Self {
        Self {
            tokens_threshold: None,
            keep_recent_messages: 6,
            keep_last_rounds: 1,
            min_messages_to_summarize: 2,
            max_summary_tokens: 900,
        }
    }
}

/// Replaces old dialogue rounds with a model-written summary.
#[derive(Clone)]
pub struct SummarizationProcessor {
    /// Processor configuration.
    pub config: SummarizationConfig,
    /// Model used when the request does not supply one.
    pub model: Arc<dyn Model>,
}

impl std::fmt::Debug for SummarizationProcessor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SummarizationProcessor")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl SummarizationProcessor {
    /// Creates a processor with `config` and a fallback `model`.
    #[must_use]
    pub fn new(config: SummarizationConfig, model: Arc<dyn Model>) -> Self {
        Self { config, model }
    }

    /// Renders the selected messages as `role: text` transcript lines.
    fn render_transcript(messages: &[BaseMessage], selected: &[usize]) -> String {
        let mut lines = Vec::with_capacity(selected.len());
        for &index in selected {
            let message = &messages[index];
            let mut line = format!("{}: {}", message.role.as_str(), message.text());
            for call in &message.tool_calls {
                line = format!(
                    "{line}\n  tool_call {}: name={} arguments={}",
                    call.id.as_deref().unwrap_or("<unknown>"),
                    call.name,
                    call.arguments
                );
            }
            lines.push(line);
        }
        lines.join("\n")
    }

    /// Indices of the messages the summary should cover, oldest first.
    ///
    /// Only complete rounds are summarised: the last
    /// [`SummarizationConfig::keep_last_rounds`] rounds of the history are
    /// excluded, and inside the remaining rounds system messages and the newest
    /// [`SummarizationConfig::keep_recent_messages`] positions are skipped.
    fn select(&self, messages: &[BaseMessage]) -> Vec<usize> {
        let rounds = dialogue_round_ranges(messages);
        let candidate_count = rounds.len().saturating_sub(self.config.keep_last_rounds);
        let protected_from = messages
            .len()
            .saturating_sub(self.config.keep_recent_messages);

        let mut selected = Vec::new();
        for range in rounds.iter().take(candidate_count) {
            let closes_with_assistant =
                messages
                    .get(range.end.saturating_sub(1))
                    .is_some_and(|message| {
                        message.role == Role::Assistant && message.tool_calls.is_empty()
                    });
            if !closes_with_assistant {
                continue;
            }
            for index in range.clone() {
                let message = &messages[index];
                if message.role == Role::System || index >= protected_from {
                    continue;
                }
                selected.push(index);
            }
        }
        selected
    }
}

#[async_trait]
impl ContextProcessor for SummarizationProcessor {
    fn name(&self) -> &'static str {
        "summarization"
    }

    async fn process(
        &self,
        messages: &mut Vec<BaseMessage>,
        cx: &ProcessorContext<'_>,
    ) -> Result<Option<ContextEvent>> {
        let selected = self.select(messages);
        if selected.len() < self.config.min_messages_to_summarize {
            return Ok(None);
        }
        let selected_messages: Vec<BaseMessage> = selected
            .iter()
            .map(|&index| messages[index].clone())
            .collect();
        let threshold = self.config.tokens_threshold.unwrap_or(cx.trigger_tokens);
        if !cx.manual && count_messages(cx.counter, &selected_messages) <= threshold {
            return Ok(None);
        }
        if cx.cancellation.is_cancelled() {
            return Err(Error::Cancelled);
        }

        let model: Arc<dyn Model> = match cx.model {
            Some(model) => Arc::clone(model),
            None => Arc::clone(&self.model),
        };
        let transcript = Self::render_transcript(messages, &selected);
        let request = ModelRequestConfig::default()
            .with_model_name(model.model_name())
            .with_max_tokens(self.config.max_summary_tokens);
        let response = model
            .invoke(
                &[
                    BaseMessage::system(SUMMARY_SYSTEM_PROMPT),
                    BaseMessage::user(render_summary_prompt(&transcript)),
                ],
                &request,
            )
            .await
            .map_err(|error| Error::Llm(format!("summarization model call failed: {error}")))?;
        let summary = response
            .message()
            .map(|message| message.text().into_owned())
            .filter(|text| !text.trim().is_empty())
            .ok_or_else(|| {
                Error::Llm("summarization model returned an empty summary".to_owned())
            })?;

        // Build the replacement off to the side: a failure above never mutated
        // the history, and this loop cannot fail.
        let tokens_before = count_messages(cx.counter, messages);
        let selected_set: std::collections::BTreeSet<usize> = selected.iter().copied().collect();
        let replacement = BaseMessage::user(format!("{CONTEXT_SUMMARY_PREFIX}\n{summary}"));
        let mut rebuilt: Vec<BaseMessage> = Vec::with_capacity(messages.len() - selected.len() + 1);
        let mut inserted = false;
        for (index, message) in messages.drain(..).enumerate() {
            if selected_set.contains(&index) {
                if !inserted {
                    rebuilt.push(replacement.clone());
                    inserted = true;
                }
                continue;
            }
            rebuilt.push(message);
        }

        *messages = rebuilt;
        let tokens_after = count_messages(cx.counter, messages);
        Ok(Some(ContextEvent {
            processor: self.name().to_owned(),
            kind: ContextEventKind::Summarized,
            messages_changed: selected,
            summary: Some(summary),
            offload_handle: None,
            tokens_before,
            tokens_after,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ContextConfig;
    use crate::offload::{InMemoryOffloadStore, OffloadScope, OffloadStore};
    use openjiuwen_foundation::prompt::{ApproxTokenCounter, ToolCall};
    use openjiuwen_test_support::{ScriptedError, ScriptedModel};
    use tokio_util::sync::CancellationToken;

    fn counter() -> ApproxTokenCounter {
        ApproxTokenCounter::with_chars_per_token(1)
    }

    fn history() -> Vec<BaseMessage> {
        vec![
            BaseMessage::system("persona"),
            BaseMessage::user("q1"),
            BaseMessage::assistant("a1"),
            BaseMessage::user("q2"),
            BaseMessage::assistant("a2"),
            BaseMessage::user("q3"),
            BaseMessage::assistant("a3"),
            BaseMessage::user("q4"),
        ]
    }

    struct Fixture {
        config: ContextConfig,
        counter: ApproxTokenCounter,
        store: InMemoryOffloadStore,
        scope: OffloadScope,
        cancellation: CancellationToken,
    }

    impl Fixture {
        fn new() -> Self {
            Self {
                config: ContextConfig::default(),
                counter: counter(),
                store: InMemoryOffloadStore::new(),
                scope: OffloadScope::new("s", "main"),
                cancellation: CancellationToken::new(),
            }
        }

        fn context(&self, trigger: usize, manual: bool) -> ProcessorContext<'_> {
            ProcessorContext {
                config: &self.config,
                counter: &self.counter,
                model: None,
                offload: &self.store as &dyn OffloadStore,
                scope: &self.scope,
                budget: 100_000,
                trigger_tokens: trigger,
                manual,
                cancellation: &self.cancellation,
                recall: None,
                session_memory: None,
            }
        }
    }

    fn scripted_model(text: &str) -> (Arc<dyn Model>, ScriptedModel) {
        let scripted = ScriptedModel::builder().respond_text(text).build();
        let model: Arc<dyn Model> = Arc::new(scripted.clone());
        (model, scripted)
    }

    #[tokio::test]
    async fn replaces_old_rounds_with_one_summary_message() {
        let fixture = Fixture::new();
        let (model, scripted) = scripted_model("older work discussed");
        let processor = SummarizationProcessor::new(
            SummarizationConfig {
                keep_recent_messages: 2,
                keep_last_rounds: 1,
                ..SummarizationConfig::default()
            },
            model,
        );
        let mut messages = history();
        let event = processor
            .process(&mut messages, &fixture.context(1_000, true))
            .await
            .unwrap()
            .unwrap();

        assert_eq!(event.kind, ContextEventKind::Summarized);
        assert_eq!(event.summary.as_deref(), Some("older work discussed"));
        assert_eq!(event.messages_changed, vec![1, 2, 3, 4, 5]);
        assert_eq!(scripted.call_count(), 1);

        assert_eq!(messages.len(), 4);
        assert_eq!(messages[0].role, Role::System);
        assert_eq!(
            messages[1].text(),
            "[CONTEXT_SUMMARY]\nolder work discussed"
        );
        assert_eq!(messages[2].text(), "a3");
        assert_eq!(messages[3].text(), "q4");

        let call = scripted.last_call().unwrap();
        let sent = &call.messages;
        assert_eq!(sent[0].role, Role::System);
        assert!(sent[1].text().contains("user: q1"));
        assert!(sent[1].text().contains("assistant: a1"));
        assert_eq!(call.request.model_name, scripted.model_name().to_owned());
        assert_eq!(call.request.max_tokens, Some(900));
    }

    #[tokio::test]
    async fn keeps_system_messages_inside_the_summarised_span() {
        let fixture = Fixture::new();
        let (model, _) = scripted_model("sum");
        let processor = SummarizationProcessor::new(
            SummarizationConfig {
                keep_recent_messages: 0,
                keep_last_rounds: 0,
                ..SummarizationConfig::default()
            },
            model,
        );
        let mut messages = vec![
            BaseMessage::system("persona"),
            BaseMessage::user("q1"),
            BaseMessage::assistant("a1"),
        ];
        let event = processor
            .process(&mut messages, &fixture.context(1_000, true))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(event.messages_changed, vec![1, 2]);
        assert_eq!(messages[0].role, Role::System);
        assert_eq!(messages[1].text(), "[CONTEXT_SUMMARY]\nsum");
    }

    #[tokio::test]
    async fn threshold_and_minimum_size_gate_the_call() {
        let fixture = Fixture::new();
        let (model, scripted) = scripted_model("sum");
        let processor = SummarizationProcessor::new(
            SummarizationConfig {
                keep_recent_messages: 0,
                keep_last_rounds: 0,
                tokens_threshold: Some(10_000),
                ..SummarizationConfig::default()
            },
            model,
        );
        let mut messages = history();
        // Not manual and far below the threshold: no call.
        assert!(processor
            .process(&mut messages, &fixture.context(10_000, false))
            .await
            .unwrap()
            .is_none());
        assert_eq!(scripted.call_count(), 0);

        // A raised minimum keeps a single candidate untouched.
        let (second, _) = scripted_model("sum");
        let processor = SummarizationProcessor::new(
            SummarizationConfig {
                keep_recent_messages: 0,
                keep_last_rounds: 0,
                min_messages_to_summarize: 99,
                ..SummarizationConfig::default()
            },
            second,
        );
        assert!(processor
            .process(&mut messages, &fixture.context(10_000, true))
            .await
            .unwrap()
            .is_none());
    }

    #[tokio::test]
    async fn transport_failure_and_empty_summary_are_llm_errors() {
        let fixture = Fixture::new();
        let failing: Arc<dyn Model> = Arc::new(
            ScriptedModel::builder()
                .fail(ScriptedError::llm("provider down"))
                .build(),
        );
        let processor = SummarizationProcessor::new(
            SummarizationConfig {
                keep_recent_messages: 0,
                keep_last_rounds: 0,
                ..SummarizationConfig::default()
            },
            failing,
        );
        let mut messages = history();
        let before = messages.clone();
        let error = processor
            .process(&mut messages, &fixture.context(1_000, true))
            .await
            .unwrap_err();
        assert!(matches!(error, Error::Llm(_)));
        assert_eq!(
            messages, before,
            "a failed call must not change the history"
        );

        let (empty, _) = scripted_model("   ");
        let processor = SummarizationProcessor::new(
            SummarizationConfig {
                keep_recent_messages: 0,
                keep_last_rounds: 0,
                ..SummarizationConfig::default()
            },
            empty,
        );
        let error = processor
            .process(&mut messages, &fixture.context(1_000, true))
            .await
            .unwrap_err();
        assert!(matches!(error, Error::Llm(_)));
        assert_eq!(messages, before);
    }

    #[tokio::test]
    async fn cancellation_stops_before_the_model_call() {
        let fixture = Fixture::new();
        let (model, scripted) = scripted_model("sum");
        let processor = SummarizationProcessor::new(
            SummarizationConfig {
                keep_recent_messages: 0,
                keep_last_rounds: 0,
                ..SummarizationConfig::default()
            },
            model,
        );
        fixture.cancellation.cancel();
        let mut messages = history();
        let error = processor
            .process(&mut messages, &fixture.context(1_000, true))
            .await
            .unwrap_err();
        assert!(matches!(error, Error::Cancelled));
        assert_eq!(scripted.call_count(), 0);
    }

    #[tokio::test]
    async fn prefers_the_per_request_model() {
        let fixture = Fixture::new();
        let (fallback, fallback_scripted) = scripted_model("fallback");
        let (request, request_scripted) = scripted_model("request");
        let processor = SummarizationProcessor::new(
            SummarizationConfig {
                keep_recent_messages: 0,
                keep_last_rounds: 1,
                ..SummarizationConfig::default()
            },
            fallback,
        );
        let mut messages = history();
        let cx = ProcessorContext {
            config: &fixture.config,
            counter: &fixture.counter,
            model: Some(&request),
            offload: &fixture.store,
            scope: &fixture.scope,
            budget: 100_000,
            trigger_tokens: 1_000,
            manual: true,
            cancellation: &fixture.cancellation,
            recall: None,
            session_memory: None,
        };
        processor
            .process(&mut messages, &cx)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(request_scripted.call_count(), 1);
        assert_eq!(fallback_scripted.call_count(), 0);
    }

    #[test]
    fn transcript_includes_tool_calls() {
        let messages = vec![
            BaseMessage::user("q"),
            BaseMessage::assistant("")
                .with_tool_calls(vec![ToolCall::new("read_file").with_id("call_1")]),
            BaseMessage::tool("call_1", "contents"),
        ];
        let text = SummarizationProcessor::render_transcript(&messages, &[0, 1, 2]);
        assert!(text.contains("user: q"));
        assert!(text.contains("tool_call call_1: name=read_file"));
        assert!(text.contains("tool: contents"));
    }
}
