//! Full compact processor: replace an exhausted conversation with a summary.
//!
//! Ports the deterministic core of Python's
//! `processor/compressor/full_compact_processor.py`. Where the other compressors
//! try to keep as much raw history as possible, full compaction is the escape
//! hatch: once the history passes
//! [`FullCompactProcessorConfig::trigger_total_tokens`] **and** the history ends
//! on a completed API round, the active part of the conversation is replaced by
//!
//! ```text
//! [system] [FULL_COMPACT_BOUNDARY]\nConversation compacted
//! [user]   This session is being continued …\n\n<summary>\n\n<notice>
//! ```
//!
//! followed by the newest messages verbatim. A previously written boundary
//! ([`FULL_COMPACT_BOUNDARY`] or
//! [`SESSION_MEMORY_BOUNDARY`](crate::session_memory::SESSION_MEMORY_BOUNDARY))
//! splits the history into a prefix that is always preserved and the active part
//! that gets compacted, so repeated compactions never stack summaries.
//!
//! Two summarisation paths exist:
//!
//! - **Session memory** — when a [`SessionMemoryStore`] holds non-empty notes
//!   and its anchor resolves, the notes replace the prefix they cover and the
//!   messages after the anchor stay verbatim. This costs no model call.
//! - **Model summary** — otherwise the active part is trimmed to
//!   [`FullCompactProcessorConfig::compression_call_max_tokens`] by dropping whole
//!   leading API rounds, summarised by the configured model, and replaced by the
//!   boundary plus summary. Without a model the deterministic offline transcript
//!   summary is used, so the pass works offline.
//!
//! A candidate is accepted only when the history really shrinks; nothing is
//! dropped from the preserved prefix, system messages included.
use std::collections::BTreeSet;
use std::ops::Range;
use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::llm::Model;
use openjiuwen_foundation::prompt::{BaseMessage, Role};
use serde::{Deserialize, Serialize};

use crate::events::{ContextEvent, ContextEventKind};
use crate::processor::compressor::{summarize, summarize_offline};
use crate::processor::{ContextProcessor, ProcessorContext};
use crate::session_memory::{reinject_session_memory, SessionMemoryStore, RECENT_MESSAGES_NOTICE};
use crate::tokens::count_messages;

/// The marker that opens a full-compaction boundary message.
pub const FULL_COMPACT_BOUNDARY: &str = "[FULL_COMPACT_BOUNDARY]";

/// The full text of the boundary message full compaction writes.
pub const FULL_COMPACT_BOUNDARY_TEXT: &str = "[FULL_COMPACT_BOUNDARY]\nConversation compacted";

/// The intro line of the summary message full compaction writes.
pub const FULL_COMPACT_SUMMARY_INTRO: &str = "This session is being continued from a previous \
conversation that ran out of context. The summary below covers the earlier portion of the \
conversation.";

/// Prepended to a trimmed remainder that starts with an assistant message.
pub const TRUNCATED_CONVERSATION_MARKER: &str =
    "[earlier conversation truncated for compaction retry]";

/// Instruction used for the compaction summary call.
const COMPACTION_PROMPT: &str = "Summarise this conversation so work can continue after a \
compaction. Preserve decisions, constraints, file paths, identifiers, open questions and next \
steps. Do not invent details.";

/// Maximum number of messages the offline fallback summary covers.
const FALLBACK_SUMMARY_MESSAGES: usize = 20;

/// Maximum characters of the offline fallback summary.
const FALLBACK_SUMMARY_CHARS: usize = 8_000;

/// Characters per token the prompt budget assumes.
const CHARS_PER_TOKEN_ESTIMATE: usize = 4;

/// Configuration of [`FullCompactProcessor`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct FullCompactProcessorConfig {
    /// Compact once the history holds more than this many tokens.
    pub trigger_total_tokens: usize,
    /// Maximum token budget of the internal summary call.
    pub compression_call_max_tokens: usize,
    /// Newest active messages kept verbatim after compaction.
    pub messages_to_keep: usize,
    /// Prefer committed session-memory notes over a model summary.
    pub session_memory_enabled: bool,
    /// Keep the assistant tool-call message that owns a kept tool result.
    pub keep_tool_message_pairs: bool,
}

impl Default for FullCompactProcessorConfig {
    fn default() -> Self {
        Self {
            trigger_total_tokens: 180_000,
            compression_call_max_tokens: 200_000,
            messages_to_keep: 10,
            session_memory_enabled: true,
            keep_tool_message_pairs: true,
        }
    }
}

/// Groups `messages` into API rounds: a round opens at the first message or at a
/// user message that follows a completed round, and closes at an assistant
/// without tool calls or at the tool result that answers the last pending call.
///
/// A trailing round that never closes is returned too, so trimming never drops
/// the newest messages.
pub(crate) fn api_round_ranges(messages: &[BaseMessage]) -> Vec<Range<usize>> {
    let mut rounds: Vec<Range<usize>> = Vec::new();
    let mut current_start: Option<usize> = None;
    let mut pending: Option<BTreeSet<String>> = None;

    for (index, message) in messages.iter().enumerate() {
        if current_start.is_none() || (message.role == Role::User && pending.is_none()) {
            // The first message opens the first round; a user turn after a
            // completed round opens the next one.
            current_start = Some(index);
        }

        match message.role {
            Role::Assistant => {
                if message.tool_calls.is_empty() {
                    let start = current_start.take().unwrap_or(index);
                    rounds.push(start..index + 1);
                    pending = None;
                } else {
                    let ids: BTreeSet<String> = message
                        .tool_calls
                        .iter()
                        .filter_map(|call| call.id.clone())
                        .collect();
                    if ids.is_empty() {
                        let start = current_start.take().unwrap_or(index);
                        rounds.push(start..index + 1);
                    } else {
                        pending = Some(ids);
                    }
                }
            }
            Role::Tool => {
                let Some(pending_ids) = pending.as_mut() else {
                    continue;
                };
                if let Some(id) = message.tool_call_id.as_deref() {
                    pending_ids.remove(id);
                }
                if pending_ids.is_empty() {
                    let start = current_start.take().unwrap_or(index);
                    rounds.push(start..index + 1);
                    pending = None;
                }
            }
            Role::System | Role::User | Role::Other(_) => {}
        }
    }
    if let Some(start) = current_start {
        rounds.push(start..messages.len());
    }
    rounds
}

/// Whether `messages` ends on a completed API round.
///
/// The last message must be an assistant that requests no tool calls, or the
/// tool result that answers the final pending call of its owning assistant.
/// Anything else — a trailing user turn, an unanswered tool call, a trailing
/// system message — means the round is still in flight and compaction must wait.
pub(crate) fn ends_on_completed_api_round(messages: &[BaseMessage]) -> bool {
    let Some(last) = messages.last() else {
        return false;
    };
    match &last.role {
        Role::Assistant => last.tool_calls.is_empty(),
        Role::Tool => {
            let Some(id) = last.tool_call_id.as_deref() else {
                return false;
            };
            let owner = messages[..messages.len() - 1].iter().rposition(|message| {
                message.role == Role::Assistant
                    && message
                        .tool_calls
                        .iter()
                        .any(|call| call.id.as_deref() == Some(id))
            });
            let Some(owner_index) = owner else {
                return false;
            };
            let pending: BTreeSet<&str> = messages[owner_index]
                .tool_calls
                .iter()
                .filter_map(|call| call.id.as_deref())
                .collect();
            let answered: BTreeSet<&str> = messages[owner_index + 1..]
                .iter()
                .filter(|message| message.role == Role::Tool)
                .filter_map(|message| message.tool_call_id.as_deref())
                .collect();
            !pending.is_empty() && pending.is_subset(&answered)
        }
        Role::System | Role::User | Role::Other(_) => false,
    }
}

/// Moves `start` back so every kept tool result keeps its owning assistant.
///
/// Only preceded assistants are considered, and the walk continues until every
/// missing tool call has an owner, so no kept tool result is orphaned.
fn extend_start_for_tool_pairs(messages: &[BaseMessage], start: usize) -> usize {
    if start == 0 || start >= messages.len() {
        return start;
    }
    let needed: BTreeSet<&str> = messages[start..]
        .iter()
        .filter(|message| message.role == Role::Tool)
        .filter_map(|message| message.tool_call_id.as_deref())
        .collect();
    if needed.is_empty() {
        return start;
    }
    let present: BTreeSet<&str> = messages[start..]
        .iter()
        .filter(|message| message.role == Role::Assistant)
        .flat_map(|message| message.tool_calls.iter())
        .filter_map(|call| call.id.as_deref())
        .collect();
    let mut missing: BTreeSet<&str> = needed.difference(&present).copied().collect();
    if missing.is_empty() {
        return start;
    }
    let mut adjusted = start;
    for index in (0..start).rev() {
        if messages[index].role != Role::Assistant {
            continue;
        }
        let mut matched = false;
        for call in &messages[index].tool_calls {
            if let Some(id) = call.id.as_deref() {
                if missing.remove(id) {
                    matched = true;
                }
            }
        }
        if matched {
            adjusted = index;
        }
        if missing.is_empty() {
            break;
        }
    }
    adjusted
}

/// Replaces an exhausted conversation with one boundary and summary.
#[derive(Clone)]
pub struct FullCompactProcessor {
    /// Processor configuration.
    pub config: FullCompactProcessorConfig,
    /// Model used when the request does not supply one.
    pub model: Option<Arc<dyn Model>>,
    /// Session-memory notes preferred over a model summary.
    pub session_memory: Option<Arc<dyn SessionMemoryStore>>,
}

impl std::fmt::Debug for FullCompactProcessor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FullCompactProcessor")
            .field("config", &self.config)
            .field("has_model", &self.model.is_some())
            .field("has_session_memory", &self.session_memory.is_some())
            .finish()
    }
}

impl FullCompactProcessor {
    /// Creates a processor with `config`, an optional `model`, and an optional
    /// session-memory `store`.
    #[must_use]
    pub fn new(
        config: FullCompactProcessorConfig,
        model: Option<Arc<dyn Model>>,
        session_memory: Option<Arc<dyn SessionMemoryStore>>,
    ) -> Self {
        Self {
            config,
            model,
            session_memory,
        }
    }

    /// Sets the session-memory store used before falling back to a summary.
    #[must_use]
    pub fn with_session_memory(mut self, store: Arc<dyn SessionMemoryStore>) -> Self {
        self.session_memory = Some(store);
        self
    }

    /// The index of the last compaction boundary, when the history has one.
    fn boundary_index(messages: &[BaseMessage]) -> Option<usize> {
        messages.iter().rposition(|message| {
            message.role == Role::System
                && (message.text().starts_with(FULL_COMPACT_BOUNDARY)
                    || message
                        .text()
                        .starts_with(crate::session_memory::SESSION_MEMORY_BOUNDARY))
        })
    }

    /// The session-memory candidate, when the notes cover part of `active`.
    async fn session_memory_candidate(
        &self,
        cx: &ProcessorContext<'_>,
        prefix: &[BaseMessage],
        active: &[BaseMessage],
    ) -> Result<Option<(Vec<BaseMessage>, String)>> {
        if !self.config.session_memory_enabled {
            return Ok(None);
        }
        let Some(store) = self.session_memory.as_ref() else {
            return Ok(None);
        };
        let Some(notes) = store.get(cx.scope).await? else {
            return Ok(None);
        };
        if notes.is_empty() {
            return Ok(None);
        }
        if cx.cancellation.is_cancelled() {
            return Err(Error::Cancelled);
        }
        let mut replaced = active.to_vec();
        if reinject_session_memory(&mut replaced, &notes).is_none() {
            tracing::debug!(
                "full compaction skipped the session-memory candidate: the anchor does not resolve"
            );
            return Ok(None);
        }
        let mut candidate = prefix.to_vec();
        candidate.extend(replaced);
        if count_messages(cx.counter, &candidate) > self.config.trigger_total_tokens {
            tracing::debug!(
                "full compaction rejected the session-memory candidate: over the trigger budget"
            );
            return Ok(None);
        }
        Ok(Some((candidate, notes.notes)))
    }

    /// Trims `active` to the summary call budget by dropping leading API rounds.
    fn trim_to_call_budget(
        &self,
        active: &[BaseMessage],
        cx: &ProcessorContext<'_>,
    ) -> Vec<BaseMessage> {
        let groups = api_round_ranges(active);
        if groups.is_empty() {
            return Vec::new();
        }
        let mut first = 0usize;
        loop {
            let mut candidate: Vec<BaseMessage> = Vec::new();
            if active[groups[first].start].role == Role::Assistant {
                candidate.push(BaseMessage::user(TRUNCATED_CONVERSATION_MARKER));
            }
            for range in &groups[first..] {
                candidate.extend(active[range.clone()].iter().cloned());
            }
            let fits =
                count_messages(cx.counter, &candidate) <= self.config.compression_call_max_tokens;
            if fits || first + 1 >= groups.len() {
                return candidate;
            }
            first += 1;
        }
    }

    /// The newest messages kept verbatim, including their owning tool calls.
    fn messages_to_keep(&self, active: &[BaseMessage]) -> Vec<BaseMessage> {
        let keep = self.config.messages_to_keep;
        if keep == 0 || active.is_empty() {
            return Vec::new();
        }
        let mut start = active.len().saturating_sub(keep);
        if self.config.keep_tool_message_pairs {
            start = extend_start_for_tool_pairs(active, start);
        }
        active[start..].to_vec()
    }

    /// The model-summary candidate.
    async fn summary_candidate(
        &self,
        cx: &ProcessorContext<'_>,
        prefix: &[BaseMessage],
        active: &[BaseMessage],
    ) -> (Vec<BaseMessage>, String) {
        let trimmed = self.trim_to_call_budget(active, cx);
        let fallback_source = &trimmed[trimmed.len().saturating_sub(FALLBACK_SUMMARY_MESSAGES)..];
        let fallback = format!(
            "Summary:\n{}",
            summarize_offline(fallback_source, FALLBACK_SUMMARY_CHARS)
        );
        let transcript = summarize_offline(
            &trimmed,
            self.config
                .compression_call_max_tokens
                .saturating_mul(CHARS_PER_TOKEN_ESTIMATE),
        );
        let model = cx.model.or(self.model.as_ref());
        let max_tokens = u32::try_from(self.config.compression_call_max_tokens).unwrap_or(u32::MAX);
        let summary = summarize(model, COMPACTION_PROMPT, &transcript, max_tokens, || {
            fallback.clone()
        })
        .await;

        let kept = self.messages_to_keep(active);
        let mut candidate = prefix.to_vec();
        candidate.push(BaseMessage::system(FULL_COMPACT_BOUNDARY_TEXT));
        let mut content = format!("{FULL_COMPACT_SUMMARY_INTRO}\n\n{summary}");
        if !kept.is_empty() {
            content.push_str("\n\n");
            content.push_str(RECENT_MESSAGES_NOTICE);
        }
        candidate.push(BaseMessage::user(content));
        candidate.extend(kept);
        (candidate, summary)
    }
}

#[async_trait]
impl ContextProcessor for FullCompactProcessor {
    fn name(&self) -> &'static str {
        "full_compact"
    }

    async fn process(
        &self,
        messages: &mut Vec<BaseMessage>,
        cx: &ProcessorContext<'_>,
    ) -> Result<Option<ContextEvent>> {
        let tokens_before = count_messages(cx.counter, messages);
        if !cx.manual
            && (tokens_before <= self.config.trigger_total_tokens
                || !ends_on_completed_api_round(messages))
        {
            return Ok(None);
        }
        let original_len = messages.len();
        let (prefix, active) = match Self::boundary_index(messages) {
            Some(index) => (messages[..index].to_vec(), messages[index + 1..].to_vec()),
            None => (Vec::new(), messages.clone()),
        };
        if active.is_empty() {
            return Ok(None);
        }

        // Session memory first; a model summary only when it cannot apply.
        let (candidate, summary) = match self.session_memory_candidate(cx, &prefix, &active).await?
        {
            Some(value) => value,
            None => self.summary_candidate(cx, &prefix, &active).await,
        };
        let tokens_after = count_messages(cx.counter, &candidate);
        if tokens_after >= tokens_before {
            tracing::debug!(
                tokens_before,
                tokens_after,
                "full compaction left the history in place: no compression benefit"
            );
            return Ok(None);
        }
        *messages = candidate;
        Ok(Some(ContextEvent {
            processor: self.name().to_owned(),
            kind: ContextEventKind::Summarized,
            messages_changed: (0..original_len).collect(),
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
    use openjiuwen_foundation::prompt::ToolCall;

    fn tool_call_assistant(id: &str, tool: &str) -> BaseMessage {
        BaseMessage::assistant("").with_tool_calls(vec![ToolCall::new(tool).with_id(id)])
    }

    #[test]
    fn config_round_trips_and_defaults_are_stable() {
        let config = FullCompactProcessorConfig::default();
        assert_eq!(
            serde_json::from_str::<FullCompactProcessorConfig>("{}").unwrap(),
            config
        );
        let json = serde_json::to_string(&config).unwrap();
        assert_eq!(
            serde_json::from_str::<FullCompactProcessorConfig>(&json).unwrap(),
            config
        );
    }

    #[test]
    fn api_rounds_close_on_answers_and_tool_results() {
        let messages = vec![
            BaseMessage::user("q1"),
            BaseMessage::assistant("a1"),
            BaseMessage::user("q2"),
            tool_call_assistant("c1", "read_file"),
            BaseMessage::tool("c1", "ok"),
            BaseMessage::assistant("a2"),
            BaseMessage::user("q3"),
        ];
        assert_eq!(api_round_ranges(&messages), vec![0..2, 2..5, 5..6, 6..7]);
        assert!(ends_on_completed_api_round(&messages[..6]));
        assert!(!ends_on_completed_api_round(&messages));
        // An unanswered tool call is still in flight.
        assert!(!ends_on_completed_api_round(&messages[..4]));
        assert!(!ends_on_completed_api_round(&[]));
    }

    #[test]
    fn a_kept_tool_result_pulls_its_owning_assistant_along() {
        let messages = vec![
            BaseMessage::user("q"),
            tool_call_assistant("c1", "read_file"),
            BaseMessage::tool("c1", "body"),
            BaseMessage::assistant("a"),
        ];
        assert_eq!(extend_start_for_tool_pairs(&messages, 2), 1);
        assert_eq!(extend_start_for_tool_pairs(&messages, 3), 3);
        assert_eq!(extend_start_for_tool_pairs(&messages, 0), 0);
    }
}
