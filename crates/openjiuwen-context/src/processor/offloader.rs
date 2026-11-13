//! Offloader processors: move oversized history out of the context window.
//!
//! Ports the three offloading processors of Python's
//! `openjiuwen.core.context_engine.processor.offloader` package that replace a
//! message body with a short preview plus an [`OffloadMarker`] pointing at the
//! stored original:
//!
//! - [`MessageOffloader`] — offloads large messages past a keep boundary once
//!   the history holds too many messages or too many tokens.
//! - [`ToolResultBudgetProcessor`] — enforces a per-round token budget on tool
//!   results, offloading the largest of a round first.
//! - [`ToolResultWindowProcessor`] — keeps only the newest `keep_last_k` results
//!   of the configured tools in context and shrinks the rest.
//!
//! The fourth processor, `MessageSummaryOffloader`, lives in
//! [`summary_offloader`](crate::processor::summary_offloader).
//!
//! All three share the same safety rules: only plain text bodies are rewritten,
//! an assistant message that requests tool calls and a message whose text
//! already carries an offload marker are never touched (so a second pass is a
//! no-op), role/`name`/`metadata`/`tool_call_id` are preserved so every
//! tool-call pairing survives, the original is stored under the caller's
//! [`OffloadScope`] before the history changes, and cancellation is honoured
//! before the pass and before every single offload.
use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::prompt::{BaseMessage, MessageContent, Role, ToolCall};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

use crate::events::{ContextEvent, ContextEventKind};
use crate::offload::{
    OffloadMarker, OffloadRecord, OffloadScope, OffloadStore, OFFLOAD_MARKER_PREFIX,
};
use crate::processor::impl_processor_state;
use crate::processor::tool_result::PERSISTED_OUTPUT_TAG;
use crate::processor::{ContextProcessor, ProcessorContext, ProcessorState};
use crate::tokens::{count_message, count_messages};
use crate::window::dialogue_round_ranges;

pub use crate::processor::tool_result_window::{
    ToolResultWindowProcessor, ToolResultWindowProcessorConfig,
};

/// Appended after the retained head of an offloaded message.
pub const OMIT_STRING: &str = "...";

/// Tag that wraps a compacted tool result kept only as a preview.
pub const WINDOWED_OUTPUT_TAG: &str = "windowed-output";

/// Metadata key marking a tool result the window processor already handled.
pub const WINDOW_MANAGED_METADATA_KEY: &str = "tool_result_window_managed";

/// Roles a message may have to be eligible for offloading.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MessageOffloadRole {
    /// User turns.
    User,
    /// Assistant turns.
    Assistant,
    /// Tool results.
    Tool,
}

impl MessageOffloadRole {
    /// Whether a message with `role` is covered by this entry.
    #[must_use]
    pub fn matches(self, role: &Role) -> bool {
        matches!(
            (self, role),
            (Self::User, Role::User)
                | (Self::Assistant, Role::Assistant)
                | (Self::Tool, Role::Tool)
        )
    }
}

/// Whether `roles` covers a message with `role`.
pub(crate) fn role_allowed(roles: &[MessageOffloadRole], role: &Role) -> bool {
    roles.iter().any(|entry| entry.matches(role))
}

/// Matches `text` against a `*`/`?` wildcard `pattern`.
///
/// `*` matches any run of characters (including an empty one) and `?` matches
/// exactly one character; every other character matches itself. This is the
/// subset of `fnmatch` that Python's protected-tool patterns use.
#[must_use]
pub fn wildcard_match(pattern: &str, text: &str) -> bool {
    let pattern: Vec<char> = pattern.chars().collect();
    let text: Vec<char> = text.chars().collect();
    let (mut pattern_index, mut text_index) = (0usize, 0usize);
    let mut star: Option<usize> = None;
    let mut resume = 0usize;

    while text_index < text.len() {
        if pattern_index < pattern.len()
            && (pattern[pattern_index] == '?' || pattern[pattern_index] == text[text_index])
        {
            pattern_index += 1;
            text_index += 1;
        } else if pattern_index < pattern.len() && pattern[pattern_index] == '*' {
            star = Some(pattern_index);
            pattern_index += 1;
            resume = text_index;
        } else if let Some(star_index) = star {
            pattern_index = star_index + 1;
            resume += 1;
            text_index = resume;
        } else {
            return false;
        }
    }
    while pattern_index < pattern.len() && pattern[pattern_index] == '*' {
        pattern_index += 1;
    }
    pattern_index == pattern.len()
}

/// The assistant tool call that produced the tool message at `index`.
///
/// The lookup walks backwards from the message, so only a preceding assistant
/// message can own a tool result.
pub(crate) fn resolve_tool_call(messages: &[BaseMessage], index: usize) -> Option<&ToolCall> {
    let id = messages.get(index)?.tool_call_id.as_deref()?;
    messages[..index]
        .iter()
        .rev()
        .filter(|message| message.role == Role::Assistant)
        .flat_map(|message| message.tool_calls.iter())
        .find(|call| call.id.as_deref() == Some(id))
}

/// Name of the tool that produced the tool message at `index`.
///
/// The owning assistant's `tool_calls` win; a tool message that carries its own
/// `name` field falls back to that.
pub(crate) fn resolve_tool_name(messages: &[BaseMessage], index: usize) -> Option<&str> {
    if let Some(call) = resolve_tool_call(messages, index) {
        return Some(call.name.as_str());
    }
    messages
        .get(index)?
        .name
        .as_deref()
        .filter(|name| !name.is_empty())
}

/// Whether any string argument of a `name:pattern` protected entry matches.
fn arguments_match(arguments: &str, pattern: &str) -> bool {
    let Ok(value) = serde_json::from_str::<Value>(arguments) else {
        // Malformed JSON arguments cannot be matched against a pattern.
        return false;
    };
    let Some(object) = value.as_object() else {
        return false;
    };
    object.values().any(|value| {
        value
            .as_str()
            .is_some_and(|text| wildcard_match(pattern, text))
    })
}

/// Whether the tool result at `index` is produced by a protected tool.
///
/// A plain entry matches the tool name exactly. An entry containing a colon is
/// `name:pattern`: the tool name must match exactly and one of the call's
/// string arguments must match the `*`/`?` wildcard pattern.
pub(crate) fn is_protected_tool_message(
    messages: &[BaseMessage],
    index: usize,
    protected_tool_names: &[String],
) -> bool {
    let Some(message) = messages.get(index) else {
        return false;
    };
    if message.role != Role::Tool {
        return false;
    }
    let Some(call) = resolve_tool_call(messages, index) else {
        return false;
    };
    for entry in protected_tool_names {
        if let Some((tool, pattern)) = entry.split_once(':') {
            if call.name == tool && arguments_match(&call.arguments, pattern) {
                return true;
            }
        } else if call.name == *entry {
            return true;
        }
    }
    false
}

/// Whether `text` already carries an offload marker.
pub(crate) fn is_already_offloaded(text: &str) -> bool {
    text.contains(OFFLOAD_MARKER_PREFIX)
}

/// Writes `message` to `store` and returns its handle and marker text.
pub(crate) async fn persist_offload(
    store: &dyn OffloadStore,
    scope: &OffloadScope,
    processor: &str,
    message: &BaseMessage,
    original_chars: usize,
) -> Result<(String, String)> {
    let handle = uuid::Uuid::new_v4().simple().to_string();
    let kind = store.kind();
    let marker = OffloadMarker {
        handle: handle.clone(),
        kind,
        path: None,
    }
    .to_marker_string();
    let mut metadata = Map::new();
    metadata.insert("processor".to_owned(), Value::String(processor.to_owned()));
    metadata.insert("original_chars".to_owned(), json!(original_chars));
    store
        .put(
            scope,
            OffloadRecord {
                handle: handle.clone(),
                kind,
                messages: vec![message.clone()],
                metadata,
            },
        )
        .await?;
    Ok((handle, marker))
}

/// The `<persisted-output>` placeholder that replaces an offloaded tool result.
pub(crate) fn persisted_placeholder(
    original_chars: usize,
    preview: &str,
    preview_chars: usize,
    marker: &str,
) -> String {
    format!(
        "<{PERSISTED_OUTPUT_TAG}>\n\
         Output too large ({original_chars} chars).\n\
         {marker}\n\
         Preview (first {preview_chars} chars):\n\
         {preview}\n\
         </{PERSISTED_OUTPUT_TAG}>"
    )
}

/// Offloads the tool result in `message` and replaces it with the placeholder.
///
/// `managed` additionally marks the message with
/// [`WINDOW_MANAGED_METADATA_KEY`], which is how the window processor recognises
/// its own output.
pub(crate) async fn offload_tool_result(
    store: &dyn OffloadStore,
    scope: &OffloadScope,
    processor: &str,
    message: &mut BaseMessage,
    trim_size: usize,
    managed: bool,
) -> Result<String> {
    let original = message.clone();
    let text = original.content.as_text().unwrap_or_default();
    let characters = text.chars().count();
    let preview: String = text.chars().take(trim_size).collect();
    let (handle, marker) = persist_offload(store, scope, processor, &original, characters).await?;
    message.content = MessageContent::Text(persisted_placeholder(
        characters, &preview, trim_size, &marker,
    ));
    if managed {
        message
            .metadata
            .insert(WINDOW_MANAGED_METADATA_KEY.to_owned(), Value::Bool(true));
    }
    Ok(handle)
}

/// Builds the event the offloaders report.
pub(crate) fn offloaded_event(
    processor: &str,
    changed: &mut Vec<usize>,
    handle: Option<String>,
    tokens_before: usize,
    tokens_after: usize,
) -> ContextEvent {
    changed.sort_unstable();
    changed.dedup();
    ContextEvent {
        processor: processor.to_owned(),
        kind: ContextEventKind::Offloaded,
        messages_changed: std::mem::take(changed),
        summary: None,
        offload_handle: handle,
        tokens_before,
        tokens_after,
    }
}

/// Configuration of [`MessageOffloader`].
///
/// Rules are evaluated in this order: the newest `messages_to_keep` messages
/// are never offloaded, then a message-count threshold, then a token threshold.
/// Only messages whose role is listed in `offload_message_types` and whose text
/// is longer than `large_message_threshold` are eligible.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct MessageOffloaderConfig {
    /// Message count above which the offloader fires; `None` disables it.
    pub messages_threshold: Option<usize>,
    /// Token count above which the offloader fires.
    pub tokens_threshold: usize,
    /// Character length above which a message counts as large, in characters.
    pub large_message_threshold: usize,
    /// Roles eligible for offloading; other roles are always kept.
    pub offload_message_types: Vec<MessageOffloadRole>,
    /// Tools whose results are never offloaded, `name` or `name:pattern`.
    pub protected_tool_names: Vec<String>,
    /// Characters of the original kept in front of the marker.
    pub trim_size: usize,
    /// Newest messages that are never offloaded; `None` keeps no fixed tail.
    pub messages_to_keep: Option<usize>,
    /// Whether the newest assistant turn without tool calls and everything
    /// after it are protected.
    pub keep_last_round: bool,
}

impl Default for MessageOffloaderConfig {
    fn default() -> Self {
        Self {
            messages_threshold: None,
            tokens_threshold: 20_000,
            large_message_threshold: 1_000,
            offload_message_types: vec![MessageOffloadRole::Tool],
            protected_tool_names: vec!["read_file".to_owned()],
            trim_size: 100,
            messages_to_keep: None,
            keep_last_round: true,
        }
    }
}

impl MessageOffloaderConfig {
    /// Checks the trim size and the keep window against their thresholds.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] when the trim size is not smaller than the
    /// large-message threshold — a preview that keeps everything would not
    /// shrink the message — and when the keep window is not smaller than the
    /// message threshold while both are set.
    pub fn validate(&self) -> Result<()> {
        if self.trim_size >= self.large_message_threshold {
            return Err(Error::Config(format!(
                "message offloader: trim_size ({}) must be smaller than large_message_threshold \
                 ({})",
                self.trim_size, self.large_message_threshold
            )));
        }
        if let (Some(keep), Some(threshold)) = (self.messages_to_keep, self.messages_threshold) {
            if keep >= threshold {
                return Err(Error::Config(format!(
                    "message offloader: messages_to_keep ({keep}) must be smaller than \
                     messages_threshold ({threshold})"
                )));
            }
        }
        Ok(())
    }
}

/// Offloads large messages once the history outgrows the configured limits.
#[derive(Clone)]
pub struct MessageOffloader {
    /// Processor configuration.
    pub config: MessageOffloaderConfig,
    /// Where offloaded originals are written.
    pub store: Arc<dyn OffloadStore>,
    /// Cross-run counters, persisted through the context checkpoint.
    pub state: ProcessorState,
}

impl std::fmt::Debug for MessageOffloader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MessageOffloader")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl MessageOffloader {
    /// Creates an offloader with `config` writing to `store`.
    #[must_use]
    pub fn new(config: MessageOffloaderConfig, store: Arc<dyn OffloadStore>) -> Self {
        Self {
            config,
            store,
            state: ProcessorState::default(),
        }
    }

    /// The exclusive end of the span that may be offloaded.
    fn offload_end(&self, messages: &[BaseMessage]) -> usize {
        let keep_boundary = match self.config.messages_to_keep {
            Some(keep) => messages.len().saturating_sub(keep),
            None => messages.len(),
        };
        if !self.config.keep_last_round {
            return keep_boundary;
        }
        match last_assistant_without_tool_calls(messages) {
            Some(index) => keep_boundary.min(index),
            None => keep_boundary,
        }
    }

    /// Whether the thresholds fire for this history.
    fn triggered(&self, messages: &[BaseMessage], cx: &ProcessorContext<'_>) -> bool {
        if let Some(keep) = self.config.messages_to_keep {
            if messages.len() <= keep {
                return false;
            }
        }
        if let Some(threshold) = self.config.messages_threshold {
            if messages.len() > threshold {
                return self.has_candidate(messages);
            }
        }
        count_messages(cx.counter, messages) > self.config.tokens_threshold
            && self.has_candidate(messages)
    }

    /// Whether any message below the keep boundary is eligible.
    fn has_candidate(&self, messages: &[BaseMessage]) -> bool {
        (0..self.offload_end(messages))
            .rev()
            .any(|index| self.is_candidate(messages, index))
    }

    /// Whether the message at `index` may be offloaded.
    fn is_candidate(&self, messages: &[BaseMessage], index: usize) -> bool {
        let message = &messages[index];
        if !role_allowed(&self.config.offload_message_types, &message.role) {
            return false;
        }
        if message.role == Role::Assistant && !message.tool_calls.is_empty() {
            return false;
        }
        let Some(text) = message.content.as_text() else {
            return false;
        };
        if is_already_offloaded(text) {
            return false;
        }
        if text.chars().count() <= self.config.large_message_threshold {
            return false;
        }
        !is_protected_tool_message(messages, index, &self.config.protected_tool_names)
    }

    /// Offloads one message and returns its handle.
    async fn offload_one(
        &self,
        messages: &mut [BaseMessage],
        index: usize,
        cx: &ProcessorContext<'_>,
    ) -> Result<String> {
        let original = messages[index].clone();
        let text = original.content.as_text().unwrap_or_default();
        let characters = text.chars().count();
        let head: String = text.chars().take(self.config.trim_size).collect();
        let (handle, marker) =
            persist_offload(&*self.store, cx.scope, self.name(), &original, characters).await?;
        messages[index].content = MessageContent::Text(format!("{head}{OMIT_STRING}{marker}"));
        Ok(handle)
    }
}

/// Index of the newest assistant message that requests no tool calls.
fn last_assistant_without_tool_calls(messages: &[BaseMessage]) -> Option<usize> {
    messages
        .iter()
        .rposition(|message| message.role == Role::Assistant && message.tool_calls.is_empty())
}

#[async_trait]
impl ContextProcessor for MessageOffloader {
    fn name(&self) -> &'static str {
        "message_offloader"
    }

    impl_processor_state!(MessageOffloader);

    async fn process(
        &self,
        messages: &mut Vec<BaseMessage>,
        cx: &ProcessorContext<'_>,
    ) -> Result<Option<ContextEvent>> {
        if cx.cancellation.is_cancelled() {
            return Err(Error::Cancelled);
        }
        // `manual` bypasses both thresholds: the candidate rules (role, size,
        // protection, already offloaded) still decide what may be rewritten.
        if !cx.manual && !self.triggered(messages, cx) {
            return Ok(None);
        }

        let tokens_before = count_messages(cx.counter, messages);
        let end = self.offload_end(messages);
        let mut changed: Vec<usize> = Vec::new();
        let mut last_handle: Option<String> = None;
        // Newest first, so the retained tail is decided before anything older
        // is rewritten.
        for index in (0..end).rev() {
            if cx.cancellation.is_cancelled() {
                return Err(Error::Cancelled);
            }
            if !self.is_candidate(messages, index) {
                continue;
            }
            last_handle = Some(self.offload_one(messages, index, cx).await?);
            changed.push(index);
        }
        if changed.is_empty() {
            return Ok(None);
        }
        let tokens_after = count_messages(cx.counter, messages);
        self.state.add("offloaded", changed.len() as u64);
        self.state
            .record_tokens("last_tokens_before", tokens_before);
        self.state.record_tokens("last_tokens_after", tokens_after);
        Ok(Some(offloaded_event(
            self.name(),
            &mut changed,
            last_handle,
            tokens_before,
            tokens_after,
        )))
    }
}

/// Configuration of [`ToolResultBudgetProcessor`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ToolResultBudgetProcessorConfig {
    /// Token budget for the tool results of one dialogue round.
    pub tokens_threshold: usize,
    /// Token size above which a single tool result may be offloaded.
    pub large_message_threshold: usize,
    /// Characters kept as the preview of an offloaded result.
    pub trim_size: usize,
    /// Tools never offloaded regardless of size; `None` protects nothing.
    pub tool_name_allowlist: Option<Vec<String>>,
}

impl Default for ToolResultBudgetProcessorConfig {
    fn default() -> Self {
        Self {
            tokens_threshold: 50_000,
            large_message_threshold: 10_000,
            trim_size: 3_000,
            tool_name_allowlist: None,
        }
    }
}

impl ToolResultBudgetProcessorConfig {
    /// Checks the preview against the eligibility threshold.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] when
    /// [`ToolResultBudgetProcessorConfig::trim_size`] is not smaller than
    /// [`ToolResultBudgetProcessorConfig::large_message_threshold`].
    pub fn validate(&self) -> Result<()> {
        if self.trim_size >= self.large_message_threshold {
            return Err(Error::Config(format!(
                "tool result budget: trim_size ({}) must be smaller than \
                 large_message_threshold ({})",
                self.trim_size, self.large_message_threshold
            )));
        }
        Ok(())
    }
}

/// Offloads the largest tool results of an over-budget round until it fits.
#[derive(Clone)]
pub struct ToolResultBudgetProcessor {
    /// Processor configuration.
    pub config: ToolResultBudgetProcessorConfig,
    /// Where offloaded originals are written.
    pub store: Arc<dyn OffloadStore>,
    /// Cross-run counters, persisted through the context checkpoint.
    pub state: ProcessorState,
}

impl std::fmt::Debug for ToolResultBudgetProcessor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolResultBudgetProcessor")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl ToolResultBudgetProcessor {
    /// Creates a processor with `config` writing to `store`.
    #[must_use]
    pub fn new(config: ToolResultBudgetProcessorConfig, store: Arc<dyn OffloadStore>) -> Self {
        Self {
            config,
            store,
            state: ProcessorState::default(),
        }
    }

    /// Total tokens of the tool results inside `round`.
    fn round_tool_tokens(
        messages: &[BaseMessage],
        round: &std::ops::Range<usize>,
        cx: &ProcessorContext<'_>,
    ) -> usize {
        round
            .clone()
            .filter(|&index| messages[index].role == Role::Tool)
            .map(|index| count_message(cx.counter, &messages[index]))
            .sum()
    }

    /// Whether the round is over budget and has at least one candidate.
    fn round_exceeded(
        &self,
        messages: &[BaseMessage],
        round: &std::ops::Range<usize>,
        cx: &ProcessorContext<'_>,
    ) -> bool {
        Self::round_tool_tokens(messages, round, cx) > self.config.tokens_threshold
            && self.largest_candidate(messages, round, cx).is_some()
    }

    /// Whether the tool result at `index` may be offloaded.
    fn is_candidate(
        &self,
        messages: &[BaseMessage],
        index: usize,
        cx: &ProcessorContext<'_>,
    ) -> bool {
        let message = &messages[index];
        if message.role != Role::Tool {
            return false;
        }
        let Some(text) = message.content.as_text() else {
            return false;
        };
        if is_already_offloaded(text) {
            return false;
        }
        if self.is_allowlisted(messages, index) {
            return false;
        }
        count_message(cx.counter, message) > self.config.large_message_threshold
    }

    /// Whether the tool that produced `index` is on the allowlist.
    fn is_allowlisted(&self, messages: &[BaseMessage], index: usize) -> bool {
        let Some(allowlist) = self.config.tool_name_allowlist.as_ref() else {
            return false;
        };
        if allowlist.is_empty() {
            return false;
        }
        resolve_tool_name(messages, index)
            .is_some_and(|name| allowlist.iter().any(|allowed| allowed == name))
    }

    /// The largest eligible result of `round`, oldest one on a tie.
    fn largest_candidate(
        &self,
        messages: &[BaseMessage],
        round: &std::ops::Range<usize>,
        cx: &ProcessorContext<'_>,
    ) -> Option<usize> {
        let mut best: Option<(usize, usize)> = None;
        for index in round.clone() {
            if !self.is_candidate(messages, index, cx) {
                continue;
            }
            let size = count_message(cx.counter, &messages[index]);
            if best.is_none_or(|(_, best_size)| size > best_size) {
                best = Some((index, size));
            }
        }
        best.map(|(index, _)| index)
    }
}

#[async_trait]
impl ContextProcessor for ToolResultBudgetProcessor {
    fn name(&self) -> &'static str {
        "tool_result_budget"
    }

    impl_processor_state!(ToolResultBudgetProcessor);

    async fn process(
        &self,
        messages: &mut Vec<BaseMessage>,
        cx: &ProcessorContext<'_>,
    ) -> Result<Option<ContextEvent>> {
        if cx.cancellation.is_cancelled() {
            return Err(Error::Cancelled);
        }
        let rounds = dialogue_round_ranges(messages);
        // `manual` bypasses this pre-check only; the per-round budget is the
        // rule itself, so a history that already fits stays untouched either
        // way.
        if !cx.manual
            && !rounds
                .iter()
                .any(|round| self.round_exceeded(messages, round, cx))
        {
            return Ok(None);
        }

        let tokens_before = count_messages(cx.counter, messages);
        let mut changed: Vec<usize> = Vec::new();
        let mut last_handle: Option<String> = None;
        for round in rounds {
            loop {
                if cx.cancellation.is_cancelled() {
                    return Err(Error::Cancelled);
                }
                if Self::round_tool_tokens(messages, &round, cx) <= self.config.tokens_threshold {
                    break;
                }
                let Some(index) = self.largest_candidate(messages, &round, cx) else {
                    break;
                };
                last_handle = Some(
                    offload_tool_result(
                        &*self.store,
                        cx.scope,
                        self.name(),
                        &mut messages[index],
                        self.config.trim_size,
                        false,
                    )
                    .await?,
                );
                changed.push(index);
            }
        }
        if changed.is_empty() {
            return Ok(None);
        }
        let tokens_after = count_messages(cx.counter, messages);
        self.state.add("offloaded", changed.len() as u64);
        self.state
            .record_tokens("last_tokens_before", tokens_before);
        self.state.record_tokens("last_tokens_after", tokens_after);
        Ok(Some(offloaded_event(
            self.name(),
            &mut changed,
            last_handle,
            tokens_before,
            tokens_after,
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use openjiuwen_foundation::prompt::ToolCall;

    #[test]
    fn tool_names_resolve_through_the_owning_assistant() {
        let messages = vec![
            BaseMessage::user("run"),
            BaseMessage::assistant("").with_tool_calls(vec![ToolCall::new("grep")
                .with_id("call_1")
                .with_arguments(serde_json::json!({"path": "docs/guide.md"}))]),
            BaseMessage::tool("call_1", "body"),
        ];
        assert_eq!(resolve_tool_name(&messages, 2), Some("grep"));
        assert!(is_protected_tool_message(
            &messages,
            2,
            &["grep:*.md".to_owned()]
        ));
        assert!(!is_protected_tool_message(
            &messages,
            2,
            &["grep:*.rs".to_owned()]
        ));
        assert!(is_protected_tool_message(
            &messages,
            2,
            &["grep".to_owned()]
        ));
        assert!(!is_protected_tool_message(
            &messages,
            2,
            &["read".to_owned()]
        ));

        let orphan = vec![
            BaseMessage::assistant("").with_tool_calls(vec![ToolCall::new("shell").with_id("c9")]),
            BaseMessage::tool("c9", "out").with_name("shell"),
        ];
        assert_eq!(resolve_tool_name(&orphan, 1), Some("shell"));
    }

    #[test]
    fn malformed_arguments_never_match_a_protected_pattern() {
        let call = ToolCall {
            arguments: "{not json".to_owned(),
            ..ToolCall::new("grep").with_id("c1")
        };
        let messages = vec![
            BaseMessage::assistant("").with_tool_calls(vec![call]),
            BaseMessage::tool("c1", "body"),
        ];
        assert!(!is_protected_tool_message(
            &messages,
            1,
            &["grep:*".to_owned()]
        ));
    }
}
