//! Round-aware truncation processor.
//!
//! Drops the oldest complete dialogue rounds until the history fits the budget.
//! The newest messages are protected, system messages survive unless the config
//! opts out, and only whole rounds are removed so tool-call/tool-result pairs are
//! never split.
use async_trait::async_trait;
use openjiuwen_common::error::Result;
use openjiuwen_foundation::prompt::{BaseMessage, TokenCounter};
use serde::{Deserialize, Serialize};

use crate::events::{ContextEvent, ContextEventKind};
use crate::processor::{ContextProcessor, ProcessorContext};
use crate::tokens::count_messages;
use crate::window::truncate_keeping_rounds;

/// Configuration of the truncation processor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct TruncationConfig {
    /// Explicit token budget; `None` uses the budget of the current request.
    pub max_tokens: Option<usize>,
    /// Newest messages that are never removed.
    pub keep_recent_messages: usize,
    /// Whether system messages are exempt from eviction.
    pub keep_system: bool,
}

impl Default for TruncationConfig {
    fn default() -> Self {
        Self {
            max_tokens: None,
            keep_recent_messages: 4,
            keep_system: true,
        }
    }
}

/// Drops whole dialogue rounds from the oldest end of the history.
#[derive(Debug, Clone)]
pub struct TruncationProcessor {
    /// Processor configuration.
    pub config: TruncationConfig,
}

impl TruncationProcessor {
    /// Creates a processor with `config`.
    #[must_use]
    pub fn new(config: TruncationConfig) -> Self {
        Self { config }
    }
}

/// Indexes of the messages in `before` that are missing from `after`.
///
/// Both histories keep their relative order, so a two-pointer walk recovers the
/// dropped positions.
fn removed_indices(before: &[BaseMessage], after: &[BaseMessage]) -> Vec<usize> {
    let mut removed = Vec::new();
    let mut after_index = 0;
    for (before_index, message) in before.iter().enumerate() {
        if after.get(after_index) == Some(message) {
            after_index += 1;
        } else {
            removed.push(before_index);
        }
    }
    removed
}

/// Drops the oldest system messages while the history is over budget.
///
/// Only called when [`TruncationConfig::keep_system`] is `false`; the protected
/// tail is still respected.
fn drop_old_systems(
    messages: &mut Vec<BaseMessage>,
    budget: usize,
    keep_recent: usize,
    counter: &dyn TokenCounter,
) -> usize {
    let mut removed = 0;
    while count_messages(counter, messages) > budget {
        let protected_start = messages.len().saturating_sub(keep_recent);
        let Some(index) = messages
            .iter()
            .take(protected_start)
            .position(|message| message.role == openjiuwen_foundation::prompt::Role::System)
        else {
            break;
        };
        messages.remove(index);
        removed += 1;
    }
    removed
}

#[async_trait]
impl ContextProcessor for TruncationProcessor {
    fn name(&self) -> &'static str {
        "truncation"
    }

    async fn process(
        &self,
        messages: &mut Vec<BaseMessage>,
        cx: &ProcessorContext<'_>,
    ) -> Result<Option<ContextEvent>> {
        let budget = self.config.max_tokens.unwrap_or(cx.budget);
        let tokens_before = count_messages(cx.counter, messages);
        let triggered = cx.manual || tokens_before > cx.trigger_tokens || tokens_before > budget;
        if !triggered {
            return Ok(None);
        }

        let before = messages.clone();
        let mut removed = 0;
        if !self.config.keep_system {
            removed += drop_old_systems(
                messages,
                budget,
                self.config.keep_recent_messages,
                cx.counter,
            );
        }
        removed += truncate_keeping_rounds(
            messages,
            budget,
            cx.counter,
            self.config.keep_recent_messages,
        );
        if removed == 0 {
            return Ok(None);
        }

        let tokens_after = count_messages(cx.counter, messages);
        Ok(Some(ContextEvent {
            processor: self.name().to_owned(),
            kind: ContextEventKind::Truncated,
            messages_changed: removed_indices(&before, messages),
            summary: None,
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
    use openjiuwen_foundation::prompt::{ApproxTokenCounter, Role};
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

        fn context(&self, budget: usize, trigger: usize, manual: bool) -> ProcessorContext<'_> {
            ProcessorContext {
                config: &self.config,
                counter: &self.counter,
                model: None,
                offload: &self.store as &dyn OffloadStore,
                scope: &self.scope,
                budget,
                trigger_tokens: trigger,
                manual,
                cancellation: &self.cancellation,
                recall: None,
                session_memory: None,
            }
        }
    }

    #[tokio::test]
    async fn truncates_to_the_budget_and_keeps_system_and_recent() {
        let fixture = Fixture::new();
        let mut messages = history();
        let budget = count_messages(&fixture.counter, &messages) - 12;
        let processor = TruncationProcessor::new(TruncationConfig {
            keep_recent_messages: 4,
            ..TruncationConfig::default()
        });
        let cx = fixture.context(budget, budget, false);

        let event = processor
            .process(&mut messages, &cx)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(event.processor, "truncation");
        assert_eq!(event.kind, ContextEventKind::Truncated);
        assert_eq!(event.messages_changed, vec![1, 2]);
        assert!(count_messages(&fixture.counter, &messages) <= budget);
        assert_eq!(messages[0].role, Role::System);
        assert_eq!(
            event.tokens_before,
            count_messages(&fixture.counter, &messages) + 12
        );
        assert_eq!(
            event.tokens_after,
            count_messages(&fixture.counter, &messages)
        );
    }

    #[tokio::test]
    async fn does_nothing_when_the_history_fits() {
        let fixture = Fixture::new();
        let mut messages = history();
        let processor = TruncationProcessor::new(TruncationConfig::default());
        let cx = fixture.context(10_000, 10_000, false);
        assert!(processor
            .process(&mut messages, &cx)
            .await
            .unwrap()
            .is_none());
        assert_eq!(messages.len(), 7);

        // Manual mode forces a pass, but a history that already fits reports
        // nothing.
        let cx = fixture.context(10_000, 10_000, true);
        assert!(processor
            .process(&mut messages, &cx)
            .await
            .unwrap()
            .is_none());
    }

    #[tokio::test]
    async fn manual_mode_bypasses_the_trigger_threshold() {
        let fixture = Fixture::new();
        let mut messages = history();
        let budget = count_messages(&fixture.counter, &messages[3..]) - 1;
        let processor = TruncationProcessor::new(TruncationConfig {
            max_tokens: Some(budget),
            keep_recent_messages: 2,
            ..TruncationConfig::default()
        });
        // Trigger and request budget well above the current size: without
        // `manual` the pass cannot start even though the processor has its own
        // tighter budget.
        let idle = TruncationProcessor::new(TruncationConfig {
            max_tokens: None,
            keep_recent_messages: 2,
            ..TruncationConfig::default()
        });
        let cx = fixture.context(100_000, 100_000, false);
        assert!(idle.process(&mut messages, &cx).await.unwrap().is_none());

        let cx = fixture.context(100_000, 100_000, true);
        let event = processor
            .process(&mut messages, &cx)
            .await
            .unwrap()
            .unwrap();
        assert!(event.messages_changed.len() >= 2);
        assert!(count_messages(&fixture.counter, &messages) <= budget);
    }

    #[tokio::test]
    async fn drops_whole_rounds_only() {
        let fixture = Fixture::new();
        let mut messages = history();
        let processor = TruncationProcessor::new(TruncationConfig {
            max_tokens: Some(10),
            keep_recent_messages: 2,
            ..TruncationConfig::default()
        });
        let cx = fixture.context(1000, 1000, true);
        processor
            .process(&mut messages, &cx)
            .await
            .unwrap()
            .unwrap();
        // Rounds are (q1,a1), (q2,a2), (q3,a3); the last one is protected and
        // the two older ones go as pairs.
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[1].text(), "q3");
        assert_eq!(messages[2].text(), "a3");
    }

    #[tokio::test]
    async fn can_drop_old_system_messages_when_configured() {
        let fixture = Fixture::new();
        let mut messages = history();
        let processor = TruncationProcessor::new(TruncationConfig {
            max_tokens: Some(24),
            keep_recent_messages: 2,
            keep_system: false,
        });
        let cx = fixture.context(1000, 1000, true);
        let event = processor
            .process(&mut messages, &cx)
            .await
            .unwrap()
            .unwrap();
        assert_ne!(event.messages_changed.len(), 0);
        assert!(messages.iter().all(|message| message.role != Role::System));
        assert!(count_messages(&fixture.counter, &messages) <= 24);
    }

    #[test]
    fn removed_indices_reports_dropped_positions() {
        let before = vec![
            BaseMessage::system("s"),
            BaseMessage::user("a"),
            BaseMessage::user("b"),
        ];
        let after = vec![BaseMessage::system("s"), BaseMessage::user("b")];
        assert_eq!(removed_indices(&before, &after), vec![1]);
        assert_eq!(removed_indices(&before, &before), Vec::<usize>::new());
    }
}
