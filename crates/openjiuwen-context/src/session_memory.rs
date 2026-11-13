//! The session-memory seam: notes a compressed context can be rebuilt from.
//!
//! Ports the session-memory half of Python's
//! `core/context_engine/context/session_memory_manager.py` and
//! `processor/compressor/full_compact_processor.py`. When a conversation grows
//! past the window, the harness writes a notes file summarising the prior work
//! and remembers the id of the last message those notes cover (the *anchor*).
//! On a later request the whole prefix up to and including the anchor can be
//! replaced by a short boundary + summary pair, optionally keeping the messages
//! that follow the anchor verbatim:
//!
//! ```text
//! [system]  [SESSION_MEMORY_BOUNDARY]\nEarlier conversation replaced with session memory
//! [user]    Earlier conversation has been replaced with the session memory file.…
//! ```
//!
//! The messages themselves are pure functions ([`reinject_session_memory`]
//! mutates only the slice it is given); durability is provided by
//! [`SessionMemoryStore`] implementations scoped to `(session_id, context_id)`.
use std::collections::HashMap;
use std::sync::{Arc, PoisonError, RwLock};

use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::prompt::{BaseMessage, Role};
use openjiuwen_foundation::store::{KvStore, KvValue};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::offload::{sanitize_key, OffloadScope};
use crate::window::dialogue_round_ranges;

/// The marker that opens the replaced-prefix boundary message.
pub const SESSION_MEMORY_BOUNDARY: &str = "[SESSION_MEMORY_BOUNDARY]";

/// The intro line of the summary message that carries the session-memory notes.
pub const SESSION_MEMORY_SUMMARY_INTRO: &str =
    "Earlier conversation has been replaced with the session memory file. Use it as the canonical summary of prior work.";

/// Appended to the summary message when later messages are kept verbatim.
pub const RECENT_MESSAGES_NOTICE: &str = "Recent messages are preserved verbatim.";

/// The message metadata key carrying the stable per-message id.
///
/// [`stamp_message_id`] writes it; [`reinject_session_memory`] resolves the
/// session-memory anchor through it.
pub const CONTEXT_MESSAGE_ID_KEY: &str = "context_message_id";

/// The persisted session-memory notes for one scope.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct SessionMemoryNotes {
    /// The Markdown notes written by the session-memory extractor.
    pub notes: String,
    /// The `context_message_id` of the last message the notes cover.
    ///
    /// `None` means the notes cannot be anchored yet; without an anchor the
    /// prefix is never replaced.
    pub anchor_message_id: Option<String>,
}

impl SessionMemoryNotes {
    /// Whether the notes carry no text (an empty or whitespace-only file).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.notes.trim().is_empty()
    }
}

/// Durable storage for the session-memory notes of one context.
#[async_trait]
pub trait SessionMemoryStore: Send + Sync {
    /// Stores `notes` under `scope`, replacing any previous notes.
    ///
    /// # Errors
    ///
    /// Returns a backend error (serialization, I/O, or session persistence).
    async fn put(&self, scope: &OffloadScope, notes: SessionMemoryNotes) -> Result<()>;

    /// Reads the notes for `scope`, or `None` when none were written.
    ///
    /// # Errors
    ///
    /// Returns a backend error when an entry exists but cannot be decoded.
    async fn get(&self, scope: &OffloadScope) -> Result<Option<SessionMemoryNotes>>;

    /// Removes the notes for `scope`; removing missing notes is a no-op.
    ///
    /// # Errors
    ///
    /// Returns a backend error when the removal cannot be persisted.
    async fn remove(&self, scope: &OffloadScope) -> Result<()>;
}

/// The isolation key for one scope.
fn scope_key(scope: &OffloadScope) -> String {
    format!(
        "{}_{}",
        sanitize_key(&scope.session_id),
        sanitize_key(&scope.context_id)
    )
}

/// Process-local session-memory storage, used by tests and short-lived engines.
#[derive(Debug, Default)]
pub struct InMemorySessionMemoryStore {
    records: RwLock<HashMap<String, SessionMemoryNotes>>,
}

impl InMemorySessionMemoryStore {
    /// Creates an empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of stored note sets.
    #[must_use]
    pub fn len(&self) -> usize {
        self.records
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .len()
    }

    /// Whether the store holds no notes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[async_trait]
impl SessionMemoryStore for InMemorySessionMemoryStore {
    async fn put(&self, scope: &OffloadScope, notes: SessionMemoryNotes) -> Result<()> {
        self.records
            .write()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(scope_key(scope), notes);
        Ok(())
    }

    async fn get(&self, scope: &OffloadScope) -> Result<Option<SessionMemoryNotes>> {
        Ok(self
            .records
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .get(&scope_key(scope))
            .cloned())
    }

    async fn remove(&self, scope: &OffloadScope) -> Result<()> {
        self.records
            .write()
            .unwrap_or_else(PoisonError::into_inner)
            .remove(&scope_key(scope));
        Ok(())
    }
}

/// Session-memory storage backed by foundation's [`KvStore`].
#[derive(Clone)]
pub struct KvSessionMemoryStore {
    store: Arc<dyn KvStore>,
    prefix: String,
}

impl std::fmt::Debug for KvSessionMemoryStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KvSessionMemoryStore")
            .field("prefix", &self.prefix)
            .finish_non_exhaustive()
    }
}

impl KvSessionMemoryStore {
    /// The default key prefix.
    pub const DEFAULT_PREFIX: &'static str = "context:session_memory:";

    /// Wraps `store` with the default key prefix.
    #[must_use]
    pub fn new(store: Arc<dyn KvStore>) -> Self {
        Self::with_prefix(store, Self::DEFAULT_PREFIX)
    }

    /// Wraps `store` with an explicit key prefix.
    #[must_use]
    pub fn with_prefix(store: Arc<dyn KvStore>, prefix: impl Into<String>) -> Self {
        Self {
            store,
            prefix: prefix.into(),
        }
    }

    /// The storage key for one scope.
    #[must_use]
    pub fn key(&self, scope: &OffloadScope) -> String {
        format!("{}{}", self.prefix, scope_key(scope))
    }
}

#[async_trait]
impl SessionMemoryStore for KvSessionMemoryStore {
    async fn put(&self, scope: &OffloadScope, notes: SessionMemoryNotes) -> Result<()> {
        let bytes = serde_json::to_vec(&notes).map_err(|error| {
            Error::Validation(format!("cannot serialize session memory notes: {error}"))
        })?;
        self.store
            .set(&self.key(scope), KvValue::bytes(bytes))
            .await
    }

    async fn get(&self, scope: &OffloadScope) -> Result<Option<SessionMemoryNotes>> {
        let Some(value) = self.store.get(&self.key(scope)).await? else {
            return Ok(None);
        };
        let notes = serde_json::from_slice(value.as_bytes()).map_err(|error| {
            Error::Validation(format!("cannot decode session memory notes: {error}"))
        })?;
        Ok(Some(notes))
    }

    async fn remove(&self, scope: &OffloadScope) -> Result<()> {
        self.store.delete(&self.key(scope)).await
    }
}

/// Builds the system message that opens a replaced prefix.
#[must_use]
pub fn boundary_message() -> BaseMessage {
    BaseMessage::system(format!(
        "{SESSION_MEMORY_BOUNDARY}\nEarlier conversation replaced with session memory"
    ))
}

/// Builds the user message that carries the session-memory notes.
///
/// `preserve_recent` appends [`RECENT_MESSAGES_NOTICE`] when messages after the
/// anchor are kept verbatim, matching Python's full-compaction wording.
#[must_use]
pub fn summary_message(notes: &SessionMemoryNotes, preserve_recent: bool) -> BaseMessage {
    let mut content = format!("{SESSION_MEMORY_SUMMARY_INTRO}\n\n{}", notes.notes);
    if preserve_recent {
        content.push_str("\n\n");
        content.push_str(RECENT_MESSAGES_NOTICE);
    }
    BaseMessage::user(content)
}

/// Whether `message` is a [`boundary_message`] (system role, marker prefix).
#[must_use]
pub fn is_boundary_message(message: &BaseMessage) -> bool {
    message.role == Role::System && message.text().starts_with(SESSION_MEMORY_BOUNDARY)
}

/// Index the anchor moves to so no tool-call/tool-result pair is split.
///
/// When `index` falls inside a dialogue round that contains a tool exchange,
/// the covered prefix grows to the end of that round; otherwise the anchor is
/// returned unchanged.
fn tool_round_end(messages: &[BaseMessage], index: usize) -> Option<usize> {
    let round = dialogue_round_ranges(messages)
        .into_iter()
        .find(|range| range.contains(&index))?;
    let has_tool_exchange = messages[round.clone()].iter().any(|message| {
        message.role == Role::Tool
            || (message.role == Role::Assistant && !message.tool_calls.is_empty())
    });
    if has_tool_exchange {
        Some(round.end.saturating_sub(1))
    } else {
        None
    }
}

/// Replaces the notes-covered prefix of `messages` with a boundary + summary.
///
/// The anchor is resolved through [`CONTEXT_MESSAGE_ID_KEY`]; everything up to
/// and including it is replaced, and the messages after it are kept untouched.
/// When the anchor sits inside a round containing a tool exchange it moves
/// forward to the end of that round, so a tool result is never orphaned.
///
/// Returns `Some(n)` with the number of original messages removed from the
/// front (the synthesized boundary and summary messages are not counted), or
/// `None` when the notes are empty, the anchor is missing, or the replacement
/// would not make `messages` strictly shorter — in which case `messages` is
/// left unchanged.
pub fn reinject_session_memory(
    messages: &mut Vec<BaseMessage>,
    notes: &SessionMemoryNotes,
) -> Option<usize> {
    if notes.is_empty() {
        return None;
    }
    let anchor_id = notes
        .anchor_message_id
        .as_deref()
        .filter(|anchor| !anchor.is_empty())?;
    let anchor = messages.iter().position(|message| {
        message
            .metadata
            .get(CONTEXT_MESSAGE_ID_KEY)
            .and_then(Value::as_str)
            == Some(anchor_id)
    })?;
    let anchor = tool_round_end(messages, anchor).unwrap_or(anchor);

    // Two synthesized messages replace the covered prefix; a one- or two-message
    // prefix would keep the history the same length or grow it.
    let covered = anchor + 1;
    if covered <= 2 {
        return None;
    }
    let preserve_recent = messages.len() > covered;

    let mut replaced = Vec::with_capacity(messages.len() - covered + 2);
    replaced.push(boundary_message());
    replaced.push(summary_message(notes, preserve_recent));
    replaced.extend(messages.drain(covered..));
    *messages = replaced;
    Some(covered)
}

/// Stamps a stable id into [`BaseMessage::metadata`] and returns it.
///
/// The id is only written when the key is absent (or empty), so repeated calls
/// are stable. Ids are random UUIDs in simple hex form, which is what the
/// session-memory anchor references.
pub fn stamp_message_id(message: &mut BaseMessage) -> String {
    if let Some(Value::String(existing)) = message.metadata.get(CONTEXT_MESSAGE_ID_KEY) {
        if !existing.is_empty() {
            return existing.clone();
        }
    }
    let id = uuid::Uuid::new_v4().simple().to_string();
    message
        .metadata
        .insert(CONTEXT_MESSAGE_ID_KEY.to_owned(), Value::String(id.clone()));
    id
}
