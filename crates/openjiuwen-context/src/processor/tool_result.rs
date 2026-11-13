//! Tool-result processor: shorten or offload old tool output.
//!
//! Tool results are the largest and least reusable part of a coding agent's
//! history. The processor keeps the newest [`ToolResultConfig::keep_recent_results`]
//! results and every result produced by a protected tool untouched, then shrinks
//! older ones: content above [`ToolResultConfig::offload_over_chars`] is written
//! to the offload store and replaced by a head excerpt plus a marker, and content
//! above [`ToolResultConfig::max_result_tokens`] is truncated head-and-tail.
//!
//! Messages are never removed and never change role, name, or `tool_call_id`, so
//! every pairing survives. When the total tool-output budget is set, the oldest
//! unprotected results are shrunk further until it fits.
use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_common::error::Result;
use openjiuwen_foundation::prompt::{BaseMessage, MessageContent, Role, TokenCounter};
use serde::{Deserialize, Serialize};

use crate::events::{ContextEvent, ContextEventKind};
use crate::offload::{OffloadMarker, OffloadRecord, OffloadStore, OFFLOAD_MARKER_PREFIX};
use crate::processor::{ContextProcessor, ProcessorContext};
use crate::tokens::{count_message, count_messages};

/// Marker inserted where a tool result was truncated in place.
pub const TRUNCATED_MARKER: &str =
    "... placeholder truncated; original content is preserved in offload storage ...";

/// Tag that wraps offloaded tool output.
pub const PERSISTED_OUTPUT_TAG: &str = "persisted-output";

/// Configuration of the tool-result processor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ToolResultConfig {
    /// Token budget above which a result is trimmed in place.
    pub max_result_tokens: usize,
    /// Optional budget for all tool results together.
    pub max_total_tool_tokens: Option<usize>,
    /// Characters of the original kept at the front of a trimmed result.
    pub head_chars: usize,
    /// Characters of the original kept at the end of a trimmed result.
    pub tail_chars: usize,
    /// Character length above which the original is offloaded instead of trimmed.
    pub offload_over_chars: Option<usize>,
    /// Tool names whose results are never touched.
    pub protected_tools: Vec<String>,
    /// Newest tool results that are never touched.
    pub keep_recent_results: usize,
}

impl Default for ToolResultConfig {
    fn default() -> Self {
        Self {
            max_result_tokens: 1000,
            max_total_tool_tokens: None,
            head_chars: 1000,
            tail_chars: 500,
            offload_over_chars: Some(8000),
            protected_tools: vec!["read_file".to_owned()],
            keep_recent_results: 3,
        }
    }
}

/// What one shrink attempt did.
#[derive(Debug, PartialEq, Eq)]
enum Shrink {
    /// The message was left alone.
    Unchanged,
    /// The content was truncated in place.
    Trimmed,
    /// The content was offloaded under this handle.
    Offloaded(String),
}

/// Shrinks old tool output in place.
#[derive(Clone)]
pub struct ToolResultProcessor {
    /// Processor configuration.
    pub config: ToolResultConfig,
    /// Where offloaded originals are written.
    pub store: Arc<dyn OffloadStore>,
}

impl std::fmt::Debug for ToolResultProcessor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolResultProcessor")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl ToolResultProcessor {
    /// Creates a processor with `config` writing to `store`.
    #[must_use]
    pub fn new(config: ToolResultConfig, store: Arc<dyn OffloadStore>) -> Self {
        Self { config, store }
    }

    /// Shrinks one message, offloading when it is large enough.
    async fn shrink(
        &self,
        message: &mut BaseMessage,
        cx: &ProcessorContext<'_>,
        force: bool,
    ) -> Result<Shrink> {
        let Some(text) = message.content.as_text() else {
            return Ok(Shrink::Unchanged);
        };
        // Never process an already rewritten message twice.
        if text.contains(OFFLOAD_MARKER_PREFIX) || text.contains(TRUNCATED_MARKER) {
            return Ok(Shrink::Unchanged);
        }
        let characters = text.chars().count();

        if let Some(limit) = self.config.offload_over_chars {
            if limit > 0 && characters >= limit {
                let original = message.clone();
                let handle = uuid::Uuid::new_v4().simple().to_string();
                let record = OffloadRecord {
                    handle: handle.clone(),
                    kind: self.store.kind(),
                    messages: vec![original],
                    metadata: serde_json::Map::new(),
                };
                self.store.put(cx.scope, record).await?;
                let marker = OffloadMarker {
                    handle: handle.clone(),
                    kind: self.store.kind(),
                    path: None,
                }
                .to_marker_string();
                let head: String = text.chars().take(self.config.head_chars).collect();
                message.content = MessageContent::Text(format!(
                    "<{PERSISTED_OUTPUT_TAG}>{head}\n...\n{marker}</{PERSISTED_OUTPUT_TAG}>"
                ));
                return Ok(Shrink::Offloaded(handle));
            }
        }

        if force || count_message(cx.counter, message) > self.config.max_result_tokens {
            let truncated = truncate_text(
                text,
                self.config.head_chars,
                self.config.tail_chars,
                TRUNCATED_MARKER,
            );
            if truncated == text {
                return Ok(Shrink::Unchanged);
            }
            message.content = MessageContent::Text(truncated);
            return Ok(Shrink::Trimmed);
        }
        Ok(Shrink::Unchanged)
    }
}

/// Head-and-tail truncation with `marker` between the excerpts.
fn truncate_text(text: &str, head_chars: usize, tail_chars: usize, marker: &str) -> String {
    let characters = text.chars().count();
    if characters <= head_chars.saturating_add(tail_chars) {
        return text.to_owned();
    }
    let head: String = text.chars().take(head_chars).collect();
    let tail: String = text
        .chars()
        .skip(characters.saturating_sub(tail_chars))
        .collect();
    if tail.is_empty() {
        format!("{head}\n{marker}")
    } else {
        format!("{head}\n{marker}\n{tail}")
    }
}

/// Maps every assistant tool-call id to the tool it names.
fn call_names(messages: &[BaseMessage]) -> HashMap<String, String> {
    let mut names = HashMap::new();
    for message in messages {
        if message.role != Role::Assistant {
            continue;
        }
        for call in &message.tool_calls {
            if let Some(id) = call.id.as_deref() {
                names.insert(id.to_owned(), call.name.clone());
            }
        }
    }
    names
}

/// Total tokens of every tool message.
fn tool_tokens(messages: &[BaseMessage], counter: &dyn TokenCounter) -> usize {
    messages
        .iter()
        .filter(|message| message.role == Role::Tool)
        .map(|message| count_message(counter, message))
        .sum()
}

#[async_trait]
impl ContextProcessor for ToolResultProcessor {
    fn name(&self) -> &'static str {
        "tool_result"
    }

    async fn process(
        &self,
        messages: &mut Vec<BaseMessage>,
        cx: &ProcessorContext<'_>,
    ) -> Result<Option<ContextEvent>> {
        let tokens_before = count_messages(cx.counter, messages);
        let tool_indices: Vec<usize> = messages
            .iter()
            .enumerate()
            .filter(|(_, message)| message.role == Role::Tool)
            .map(|(index, _)| index)
            .collect();
        if tool_indices.is_empty() {
            return Ok(None);
        }

        let names = call_names(messages);
        let protected_tools: BTreeSet<&str> = self
            .config
            .protected_tools
            .iter()
            .map(String::as_str)
            .collect();
        let recent_from = tool_indices
            .len()
            .saturating_sub(self.config.keep_recent_results);
        let is_protected = |position: usize, message: &BaseMessage| {
            if position >= recent_from {
                return true;
            }
            message
                .tool_call_id
                .as_deref()
                .and_then(|id| names.get(id))
                .is_some_and(|name| protected_tools.contains(name.as_str()))
        };

        let mut changed: BTreeSet<usize> = BTreeSet::new();
        let mut offloaded: Option<String> = None;
        for (position, &index) in tool_indices.iter().enumerate() {
            if is_protected(position, &messages[index]) {
                continue;
            }
            match self.shrink(&mut messages[index], cx, false).await? {
                Shrink::Unchanged => {}
                Shrink::Trimmed => {
                    changed.insert(index);
                }
                Shrink::Offloaded(handle) => {
                    changed.insert(index);
                    offloaded.get_or_insert(handle);
                }
            }
        }

        if let Some(max_total) = self.config.max_total_tool_tokens {
            let mut total = tool_tokens(messages, cx.counter);
            for (position, &index) in tool_indices.iter().enumerate() {
                if total <= max_total {
                    break;
                }
                if is_protected(position, &messages[index]) {
                    continue;
                }
                match self.shrink(&mut messages[index], cx, true).await? {
                    Shrink::Unchanged => {}
                    Shrink::Trimmed => {
                        changed.insert(index);
                    }
                    Shrink::Offloaded(handle) => {
                        changed.insert(index);
                        offloaded.get_or_insert(handle);
                    }
                }
                total = tool_tokens(messages, cx.counter);
            }
        }

        if changed.is_empty() {
            return Ok(None);
        }
        let tokens_after = count_messages(cx.counter, messages);
        let (kind, offload_handle) = match offloaded {
            Some(handle) => (ContextEventKind::Offloaded, Some(handle)),
            None => (ContextEventKind::ToolResultTrimmed, None),
        };
        Ok(Some(ContextEvent {
            processor: self.name().to_owned(),
            kind,
            messages_changed: changed.into_iter().collect(),
            summary: None,
            offload_handle,
            tokens_before,
            tokens_after,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ContextConfig;
    use crate::offload::{rehydrate, InMemoryOffloadStore, OffloadKind, OffloadScope};
    use openjiuwen_foundation::prompt::{ApproxTokenCounter, ToolCall};
    use tokio_util::sync::CancellationToken;

    fn counter() -> ApproxTokenCounter {
        ApproxTokenCounter::with_chars_per_token(1)
    }

    struct Fixture {
        config: ContextConfig,
        counter: ApproxTokenCounter,
        store: Arc<dyn OffloadStore>,
        scope: OffloadScope,
        cancellation: CancellationToken,
    }

    impl Fixture {
        fn new() -> Self {
            Self {
                config: ContextConfig::default(),
                counter: counter(),
                store: Arc::new(InMemoryOffloadStore::new()),
                scope: OffloadScope::new("s", "main"),
                cancellation: CancellationToken::new(),
            }
        }

        fn context(&self) -> ProcessorContext<'_> {
            ProcessorContext {
                config: &self.config,
                counter: &self.counter,
                model: None,
                offload: &*self.store,
                scope: &self.scope,
                budget: 100_000,
                trigger_tokens: 80_000,
                manual: false,
                cancellation: &self.cancellation,
                recall: None,
                session_memory: None,
            }
        }
    }

    fn tool_exchange(id: &str, tool: &str, output: &str) -> Vec<BaseMessage> {
        vec![
            BaseMessage::user("run"),
            BaseMessage::assistant("").with_tool_calls(vec![ToolCall::new(tool).with_id(id)]),
            BaseMessage::tool(id, output),
        ]
    }

    #[tokio::test]
    async fn trims_old_results_and_keeps_the_newest() {
        let fixture = Fixture::new();
        let long = "x".repeat(400);
        let mut messages = Vec::new();
        for index in 0..5 {
            messages.extend(tool_exchange(
                &format!("call_{index}"),
                "grep",
                &format!("{long}{index}"),
            ));
        }
        let processor = ToolResultProcessor::new(
            ToolResultConfig {
                max_result_tokens: 50,
                head_chars: 40,
                tail_chars: 20,
                offload_over_chars: None,
                keep_recent_results: 2,
                ..ToolResultConfig::default()
            },
            Arc::clone(&fixture.store),
        );
        let event = processor
            .process(&mut messages, &fixture.context())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(event.kind, ContextEventKind::ToolResultTrimmed);
        assert_eq!(event.offload_handle, None);
        assert_eq!(event.messages_changed, vec![2, 5, 8]);
        assert!(event.tokens_after < event.tokens_before);

        // Newest two tool results are intact; the older three carry the marker.
        assert_eq!(messages[14].text(), format!("{long}4"));
        assert_eq!(messages[11].text(), format!("{long}3"));
        assert!(messages[2].text().contains(TRUNCATED_MARKER));
        assert!(messages[2].text().starts_with(&"x".repeat(40)));
        assert!(messages[2].text().ends_with('0'));
        assert_eq!(messages[2].tool_call_id.as_deref(), Some("call_0"));
    }

    #[tokio::test]
    async fn protects_configured_tools() {
        let fixture = Fixture::new();
        let long = "y".repeat(400);
        let mut messages = Vec::new();
        messages.extend(tool_exchange("call_read", "read_file", &long));
        messages.extend(tool_exchange("call_grep", "grep", &long));
        let processor = ToolResultProcessor::new(
            ToolResultConfig {
                max_result_tokens: 10,
                head_chars: 40,
                tail_chars: 20,
                offload_over_chars: None,
                keep_recent_results: 0,
                ..ToolResultConfig::default()
            },
            Arc::clone(&fixture.store),
        );
        let event = processor
            .process(&mut messages, &fixture.context())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(event.messages_changed, vec![5]);
        assert_eq!(messages[2].text(), long);
        assert!(messages[5].text().contains(TRUNCATED_MARKER));
    }

    #[tokio::test]
    async fn offloads_large_output_and_rehydrates_it() {
        let fixture = Fixture::new();
        let original = "z".repeat(9_000);
        let mut messages = tool_exchange("call_big", "shell", &original);
        let processor = ToolResultProcessor::new(
            ToolResultConfig {
                offload_over_chars: Some(8_000),
                head_chars: 50,
                max_result_tokens: 10,
                keep_recent_results: 0,
                ..ToolResultConfig::default()
            },
            Arc::clone(&fixture.store),
        );
        let event = processor
            .process(&mut messages, &fixture.context())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(event.kind, ContextEventKind::Offloaded);
        let handle = event.offload_handle.clone().unwrap();
        assert_eq!(event.messages_changed, vec![2]);

        let text = messages[2].text().to_string();
        assert!(text.starts_with(&format!("<{PERSISTED_OUTPUT_TAG}>")));
        assert!(text.ends_with(&format!("</{PERSISTED_OUTPUT_TAG}>")));
        assert!(text.contains(&"z".repeat(50)));
        let marker = OffloadMarker::parse(&text).unwrap();
        assert_eq!(marker.handle, handle);
        assert_eq!(marker.kind, OffloadKind::InMemory);
        assert_eq!(messages[2].tool_call_id.as_deref(), Some("call_big"));

        let restored = rehydrate(&*fixture.store, &fixture.scope, &text)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(restored[0].text(), original);

        // Running again leaves the placeholder untouched.
        assert!(processor
            .process(&mut messages, &fixture.context())
            .await
            .unwrap()
            .is_none());
    }

    #[tokio::test]
    async fn enforces_the_total_tool_budget_oldest_first() {
        let fixture = Fixture::new();
        let mut messages = Vec::new();
        for index in 0..4 {
            messages.extend(tool_exchange(
                &format!("call_{index}"),
                "grep",
                &format!("{}{index}", "q".repeat(200)),
            ));
        }
        let processor = ToolResultProcessor::new(
            ToolResultConfig {
                max_result_tokens: 1_000,
                max_total_tool_tokens: Some(600),
                head_chars: 20,
                tail_chars: 10,
                offload_over_chars: None,
                keep_recent_results: 1,
                ..ToolResultConfig::default()
            },
            Arc::clone(&fixture.store),
        );
        let event = processor
            .process(&mut messages, &fixture.context())
            .await
            .unwrap()
            .unwrap();
        assert!(event.messages_changed.len() >= 2);
        assert!(tool_tokens(&messages, &fixture.counter) <= 600);
        // The newest result stays whole.
        assert_eq!(messages[11].text(), format!("{}{}", "q".repeat(200), 3));
        assert_eq!(event.messages_changed.last(), Some(&8));
    }

    #[tokio::test]
    async fn no_tool_messages_is_a_noop() {
        let fixture = Fixture::new();
        let processor =
            ToolResultProcessor::new(ToolResultConfig::default(), Arc::clone(&fixture.store));
        let mut messages = vec![BaseMessage::user("hi"), BaseMessage::assistant("hello")];
        assert!(processor
            .process(&mut messages, &fixture.context())
            .await
            .unwrap()
            .is_none());
    }

    #[test]
    fn truncate_text_keeps_both_ends() {
        let text = "abcdefghij";
        assert_eq!(truncate_text(text, 3, 2, "M"), "abc\nM\nij");
        assert_eq!(truncate_text(text, 8, 8, "M"), text);
        assert_eq!(truncate_text(text, 3, 0, "M"), "abc\nM");
    }
}
