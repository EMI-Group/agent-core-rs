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
//!   messages after the anchor stay verbatim. When the store holds no usable
//!   notes — none at all, an empty set, or notes anchored on a message that has
//!   left the history — the configured
//!   [`SessionMemoryAgent`]
//!   extracts a note set over the active history, commits it to the store, and
//!   the branch continues with it. The model call this costs is the reason the
//!   branch is preferred at all: it replaces the whole active history with a
//!   short notes block. A branch that cannot run at all (no store, no agent, no
//!   replaceable prefix) reports
//!   `FullCompactProcessor::refuse_session_memory`'s reason and the summary
//!   path below takes over.
//! - **Model summary** — otherwise the active part is trimmed to
//!   [`FullCompactProcessorConfig::compression_call_max_tokens`] by dropping whole
//!   leading API rounds, summarised by the configured model, and replaced by the
//!   boundary plus summary. Without a model the deterministic offline transcript
//!   summary is used, so the pass works offline.
//!
//! A candidate is accepted only when the history really shrinks; nothing is
//! dropped from the preserved prefix, system messages included.
//!
//! Once a candidate is accepted, an optional state snapshot may be reinjected
//! between the summary and the kept tail: with
//! [`FullCompactProcessorConfig::reinject_state`] on and a registry wired
//! through [`FullCompactProcessor::with_reinjection`], the host renders one
//! `<recovered_context>` message from the sections it can still read, so
//! behavioural state (the active plan, loaded skills, plan-mode restrictions,
//! team task status) survives the summary. The message is written only for an
//! accepted pass, and a stale snapshot left in the kept tail is replaced rather
//! than stacked a second time.
use std::collections::BTreeSet;
use std::ops::Range;
use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::llm::Model;
use openjiuwen_foundation::prompt::{BaseMessage, MessageContent, Role};
use serde::{Deserialize, Serialize};

use crate::events::{ContextEvent, ContextEventKind};
use crate::processor::compressor::reinjection::{
    is_reinjected_state_message, SnapshotContext, SnapshotRegistry, DEFAULT_STATE_MARKER,
    DEFAULT_STATE_SNAPSHOT_MAX_CHARS, REINJECTED_SECTIONS_METADATA_KEY,
};
use crate::processor::compressor::{
    archive_replaced_messages, rollback_recall_archive, summarize, summarize_offline,
};
use crate::processor::impl_processor_state;
use crate::processor::{ContextProcessor, ProcessorContext, ProcessorState};
use crate::session_memory::{
    notes_anchor_index, reinject_session_memory, SessionMemoryStore, RECENT_MESSAGES_NOTICE,
};
use crate::session_memory_agent::{SessionMemoryAgent, SessionMemoryOutcome};
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

/// Why the session-memory branch could not run: nothing to read the notes from.
const NO_SESSION_MEMORY_STORE: &str =
    "no session-memory store is mounted on the full-compaction processor";

/// Why the branch could not run: the notes are missing and nothing can write them.
const NO_SESSION_MEMORY_AGENT: &str = "the session-memory notes are missing and no \
session-memory agent is configured to extract them";

/// Why the branch could not run: the notes cover too little history.
const NO_REPLACEABLE_PREFIX: &str = "the session-memory notes do not cover a replaceable prefix";

/// Checkpoint key carrying the last reason the session-memory branch could not run.
const STATE_REFUSAL: &str = "session_memory_refusal";
/// Checkpoint counter of the refusals the session-memory branch reported.
const STATE_REFUSALS: &str = "session_memory_refusals";
/// Checkpoint key naming the producer of the notes the last pass used.
const STATE_SOURCE: &str = "session_memory_source";

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
    /// Reinject a host-rendered state snapshot after compaction.
    ///
    /// Off by default: a host enables it once it has registered
    /// [`ContextSnapshotProvider`](crate::processor::compressor::reinjection::ContextSnapshotProvider)s
    /// on the registry handed to
    /// [`FullCompactProcessor::with_reinjection`]. Without a wired registry the
    /// field has no effect.
    pub reinject_state: bool,
    /// Character budget of one reinjected section.
    pub state_snapshot_max_chars: usize,
    /// Sections to reinject by name; empty means every registered provider.
    pub reinject_sections: Vec<String>,
    /// Marker that opens a reinjected section header.
    ///
    /// `None` (or an empty string) means [`DEFAULT_STATE_MARKER`].
    pub state_marker: Option<String>,
}

impl Default for FullCompactProcessorConfig {
    fn default() -> Self {
        Self {
            trigger_total_tokens: 180_000,
            compression_call_max_tokens: 200_000,
            messages_to_keep: 10,
            session_memory_enabled: true,
            keep_tool_message_pairs: true,
            reinject_state: false,
            state_snapshot_max_chars: DEFAULT_STATE_SNAPSHOT_MAX_CHARS,
            reinject_sections: Vec::new(),
            state_marker: Some(DEFAULT_STATE_MARKER.to_owned()),
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
    /// Agent that extracts the notes the store does not hold yet.
    ///
    /// `None` means the session-memory branch can only reinject committed notes;
    /// when the store is empty it then reports
    /// `NO_SESSION_MEMORY_AGENT` instead of doing nothing.
    pub session_memory_agent: Option<Arc<SessionMemoryAgent>>,
    /// Host-rendered state sections reinjected after an accepted compaction.
    pub reinjection: Option<Arc<SnapshotRegistry>>,
    /// Cross-run counters, persisted through the context checkpoint.
    pub state: ProcessorState,
}

impl std::fmt::Debug for FullCompactProcessor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FullCompactProcessor")
            .field("config", &self.config)
            .field("has_model", &self.model.is_some())
            .field("has_session_memory", &self.session_memory.is_some())
            .field(
                "has_session_memory_agent",
                &self.session_memory_agent.is_some(),
            )
            .field("has_reinjection", &self.reinjection.is_some())
            .finish_non_exhaustive()
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
            session_memory_agent: None,
            reinjection: None,
            state: ProcessorState::default(),
        }
    }

    /// Sets the session-memory store used before falling back to a summary.
    #[must_use]
    pub fn with_session_memory(mut self, store: Arc<dyn SessionMemoryStore>) -> Self {
        self.session_memory = Some(store);
        self
    }

    /// Sets the agent that extracts session-memory notes the store lacks.
    ///
    /// The agent is only consulted when the store holds no usable notes — they
    /// are missing, empty, or anchored on a message that has left the history —
    /// and only while [`FullCompactProcessorConfig::session_memory_enabled`] is
    /// on. A [`ProcessorContext::session_memory_agent`] supplied for one request
    /// wins over it.
    #[must_use]
    pub fn with_session_memory_agent(mut self, agent: Arc<SessionMemoryAgent>) -> Self {
        self.session_memory_agent = Some(agent);
        self
    }

    /// The agent this pass may extract notes with.
    ///
    /// Precedence: the per-request [`ProcessorContext::session_memory_agent`],
    /// then the processor's own agent, then none.
    fn effective_agent<'a>(
        &'a self,
        cx: &'a ProcessorContext<'_>,
    ) -> Option<&'a SessionMemoryAgent> {
        cx.session_memory_agent
            .as_deref()
            .or(self.session_memory_agent.as_deref())
    }

    /// Reports why the session-memory branch could not run.
    ///
    /// The reason reaches a host twice: as a structured [`tracing::warn!`] and
    /// through the processor checkpoint ([`STATE_REFUSAL`] plus the
    /// [`STATE_REFUSALS`] counter), because a processor that returns no event
    /// has no other reporting channel. The branch itself falls through to the
    /// model or offline summary.
    fn refuse_session_memory(&self, reason: &str) {
        tracing::warn!(
            reason,
            "full compaction could not use session memory; falling back to the summary path"
        );
        self.state.set_text(STATE_REFUSAL, reason);
        self.state.bump(STATE_REFUSALS);
    }

    /// Sets the registry of state sections reinjected after an accepted
    /// compaction.
    ///
    /// The registry is only consulted when
    /// [`FullCompactProcessorConfig::reinject_state`] is on, and only after the
    /// candidate has passed the benefit gate.
    #[must_use]
    pub fn with_reinjection(mut self, registry: Arc<SnapshotRegistry>) -> Self {
        self.reinjection = Some(registry);
        self
    }

    /// The marker that opens a reinjected section header.
    fn state_marker(&self) -> &str {
        self.config
            .state_marker
            .as_deref()
            .filter(|marker| !marker.is_empty())
            .unwrap_or(DEFAULT_STATE_MARKER)
    }

    /// The recovered-context message this pass injects, when reinjection is on.
    ///
    /// `messages` is the pre-compaction history the snapshot may inspect;
    /// `candidate` is the accepted compacted history whose tail after
    /// `summary_index` survives verbatim, which is exactly the tail providers
    /// de-duplicate against on both the summary and the session-memory path.
    ///
    /// This crate holds no session handle and no workspace root, so
    /// [`SnapshotContext::session_state`] is [`serde_json::Value::Null`] and
    /// [`SnapshotContext::workspace_root`] is `None` here; a provider that needs
    /// either captures it in its own fields.
    fn reinjection_message(
        &self,
        messages: &[BaseMessage],
        candidate: &[BaseMessage],
        summary_index: usize,
    ) -> Option<BaseMessage> {
        if !self.config.reinject_state {
            return None;
        }
        let registry = self.reinjection.as_ref()?;
        let kept: Vec<BaseMessage> = candidate[summary_index + 1..].to_vec();
        let only: Vec<&str> = self
            .config
            .reinject_sections
            .iter()
            .map(String::as_str)
            .collect();
        let state = serde_json::Value::Null;
        let cx = SnapshotContext {
            session_state: &state,
            source_messages: messages,
            kept_messages: &kept,
            workspace_root: None,
            state_marker: self.state_marker(),
            max_chars: self.config.state_snapshot_max_chars,
        };
        let message =
            registry.build_single_message(&cx, (!only.is_empty()).then_some(only.as_slice()))?;
        let sections = message
            .metadata
            .get(REINJECTED_SECTIONS_METADATA_KEY)
            .and_then(serde_json::Value::as_array)
            .map_or(0, Vec::len);
        tracing::debug!(sections, "full compaction reinjected a state snapshot");
        Some(message)
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
    ///
    /// The notes are taken from the store when they are usable — non-empty and
    /// anchored on a message still in `active`. Otherwise the branch refreshes
    /// them: an agent (see [`FullCompactProcessor::effective_agent`]) extracts a
    /// new note set over `active` and the result is committed to the store before
    /// the reinjection runs.
    ///
    /// The second element of the result is the reason the branch could not run,
    /// `None` when it did. A caller reports it through
    /// [`FullCompactProcessor::refuse_session_memory`]; the summary path then
    /// takes over.
    async fn session_memory_candidate(
        &self,
        cx: &ProcessorContext<'_>,
        prefix: &[BaseMessage],
        active: &mut [BaseMessage],
    ) -> Result<(Option<(Vec<BaseMessage>, String)>, Option<String>)> {
        if !self.config.session_memory_enabled {
            return Ok((None, None));
        }
        let Some(store) = self.session_memory.as_ref() else {
            return Ok((None, Some(NO_SESSION_MEMORY_STORE.to_owned())));
        };
        let stored = store.get(cx.scope).await?;
        let usable = stored
            .as_ref()
            .filter(|notes| !notes.is_empty() && notes_anchor_index(active, notes).is_some());
        let notes = if let Some(usable) = usable {
            self.state.set_text(STATE_SOURCE, "store");
            usable.clone()
        } else {
            // The notes are missing, empty, or anchored on a message that
            // has left the history: only an extraction can make them usable.
            let Some(agent) = self.effective_agent(cx) else {
                return Ok((None, Some(NO_SESSION_MEMORY_AGENT.to_owned())));
            };
            let current = stored.unwrap_or_default();
            let model = cx.model.or(self.model.as_ref());
            let outcome = agent.update(active, &current, model).await;
            let written = match outcome {
                SessionMemoryOutcome::Written {
                    source,
                    notes,
                    directives_applied,
                    model_failure,
                } => {
                    if let Some(failure) = model_failure {
                        tracing::warn!(
                            failure,
                            "session-memory extraction used the offline extractor"
                        );
                    }
                    self.state.set_text(STATE_SOURCE, source.as_str());
                    self.state.set_u64(
                        "session_memory_directives",
                        u64::try_from(directives_applied).unwrap_or(u64::MAX),
                    );
                    self.state.bump("session_memory_extractions");
                    tracing::debug!(
                        source = source.as_str(),
                        directives_applied,
                        "full compaction extracted new session-memory notes"
                    );
                    // A write failure costs durability, not this pass: the
                    // notes in hand are still the ones this compaction uses.
                    if let Err(error) = store.put(cx.scope, notes.clone()).await {
                        tracing::warn!(
                            error = %error,
                            "could not persist the extracted session-memory notes"
                        );
                    }
                    notes
                }
                outcome => {
                    let reason = outcome
                        .reason()
                        .unwrap_or("the session-memory agent produced no notes and no reason");
                    return Ok((
                        None,
                        Some(format!("session memory could not be extracted: {reason}")),
                    ));
                }
            };
            written
        };
        if cx.cancellation.is_cancelled() {
            return Err(Error::Cancelled);
        }
        let mut replaced = active.to_vec();
        if reinject_session_memory(&mut replaced, &notes).is_none() {
            return Ok((None, Some(NO_REPLACEABLE_PREFIX.to_owned())));
        }
        let mut candidate = prefix.to_vec();
        candidate.extend(replaced);
        if count_messages(cx.counter, &candidate) > self.config.trigger_total_tokens {
            tracing::debug!(
                "full compaction rejected the session-memory candidate: over the trigger budget"
            );
            return Ok((None, None));
        }
        Ok((Some((candidate, notes.notes)), None))
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

    impl_processor_state!(FullCompactProcessor);

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
        let (prefix, mut active) = match Self::boundary_index(messages) {
            Some(index) => (messages[..index].to_vec(), messages[index + 1..].to_vec()),
            None => (Vec::new(), messages.clone()),
        };
        if active.is_empty() {
            return Ok(None);
        }

        // Session memory first; a model summary only when it cannot apply. A
        // branch that cannot apply says why before the summary takes over.
        let (session_memory, refusal) = self
            .session_memory_candidate(cx, &prefix, active.as_mut_slice())
            .await?;
        if let Some(reason) = refusal.as_deref() {
            self.refuse_session_memory(reason);
        }
        let (mut candidate, summary, from_session_memory) =
            if let Some((candidate, notes)) = session_memory {
                (candidate, notes, true)
            } else {
                let (candidate, summary) = self.summary_candidate(cx, &prefix, &active).await;
                (candidate, summary, false)
            };
        // Both paths put the summary user message right after the boundary, so
        // the recall marker of the replaced `active` history lands at its end.
        let marker = archive_replaced_messages(cx, &active).await;
        if let Some(marker) = &marker {
            let summary_index = prefix.len() + 1;
            if let Some(message) = candidate.get_mut(summary_index) {
                if message.role == Role::User {
                    let text = message.text().into_owned();
                    message.content = MessageContent::Text(marker.append(text));
                }
            } else {
                tracing::warn!(
                    summary_index,
                    "full compaction could not locate its summary message; no marker was written"
                );
                rollback_recall_archive(cx, marker).await;
            }
        }
        let tokens_after = count_messages(cx.counter, &candidate);
        if tokens_after >= tokens_before {
            tracing::debug!(
                tokens_before,
                tokens_after,
                "full compaction left the history in place: no compression benefit"
            );
            if let Some(marker) = &marker {
                rollback_recall_archive(cx, marker).await;
            }
            return Ok(None);
        }
        // The pass is accepted: a state snapshot may now be reinjected after the
        // summary and before the kept tail. A stale snapshot that survived in
        // the tail is replaced rather than stacked a second time.
        let summary_index = prefix.len() + 1;
        let tokens_after = match self.reinjection_message(messages, &candidate, summary_index) {
            Some(message) => {
                let mut tail: Vec<BaseMessage> = candidate.split_off(summary_index + 1);
                tail.retain(|kept| !is_reinjected_state_message(kept));
                tail.insert(0, message);
                candidate.extend(tail);
                count_messages(cx.counter, &candidate)
            }
            None => tokens_after,
        };
        *messages = candidate;
        self.state.bump("compactions");
        if from_session_memory {
            self.state.bump("session_memory_compactions");
        }
        self.state
            .record_tokens("last_tokens_before", tokens_before);
        self.state.record_tokens("last_tokens_after", tokens_after);
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
