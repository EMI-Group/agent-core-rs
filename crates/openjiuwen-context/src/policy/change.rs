//! KV-cache diffing over LLM-bound windows.
//!
//! Ports the `ContextWindowChange` / `detect_context_window_change` primitives
//! from `base.py` and `context/context.py`, plus Python's `model_revision` /
//! `position` diffing. A [`ContextWindowSnapshot`] is the message/tool payload a
//! provider actually receives; comparing the previous snapshot with the current
//! one answers the only question a prompt cache cares about: *how much of the
//! reused prefix survives?*
//!
//! The answer is deliberately richer than Python's
//! `ContextWindowChange` (which only reports `msg_start`/`tools_start` ranges
//! into the old window): the Rust enum keeps the old/new message and tool lists
//! out of the report and instead names the shape of the change, so a caller can
//! decide whether to evict, reuse, or ignore.
//!
//! Every function here is pure CPU: no I/O, no clock, no model call.
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use openjiuwen_foundation::prompt::BaseMessage;

use crate::usage::context_tokens::canonical_json;

/// Separator between hash fields; a control byte no chat payload sends.
const FIELD_SEPARATOR: u8 = 0x1f;
/// Separator between repeated tool-call records.
const RECORD_SEPARATOR: u8 = 0x1e;

/// Renders bytes as lowercase hex.
fn hex_digest(bytes: &[u8]) -> String {
    use std::fmt::Write as _;

    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        // Writing to a `String` is infallible.
        let _ = write!(out, "{byte:02x}");
    }
    out
}

/// A stable content hash of one message as it appears on the wire.
///
/// The digest covers the role, the flattened content text, `name`, the
/// message's own `tool_call_id`, and each tool call's id, name and arguments.
/// Those are exactly the fields a provider serializes; metadata, reasoning text
/// and other local annotations are ignored, so re-stamping a bookkeeping key
/// never looks like a prompt change.
///
/// The digest is a *content* fingerprint: two messages with equal digests are
/// byte-identical for prompt-cache purposes.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(default)]
pub struct MessageFingerprint {
    /// Lowercase hex SHA-256 of the wire-relevant message fields.
    pub digest: String,
}

impl MessageFingerprint {
    /// Fingerprints `message`.
    #[must_use]
    pub fn of(message: &BaseMessage) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(message.role.as_str().as_bytes());
        hasher.update([FIELD_SEPARATOR]);
        hasher.update(message.text().as_bytes());
        hasher.update([FIELD_SEPARATOR]);
        hasher.update(message.name.as_deref().unwrap_or_default().as_bytes());
        hasher.update([FIELD_SEPARATOR]);
        hasher.update(
            message
                .tool_call_id
                .as_deref()
                .unwrap_or_default()
                .as_bytes(),
        );
        for call in &message.tool_calls {
            hasher.update([RECORD_SEPARATOR]);
            hasher.update(call.id.as_deref().unwrap_or_default().as_bytes());
            hasher.update([FIELD_SEPARATOR]);
            hasher.update(call.name.as_bytes());
            hasher.update([FIELD_SEPARATOR]);
            hasher.update(call.arguments.as_bytes());
        }
        Self {
            digest: hex_digest(&hasher.finalize()),
        }
    }

    /// The digest as a hex string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.digest
    }
}

/// Content signature of a tool-definition list.
///
/// Tool definitions are serialized as one JSON array, so the signature hashes
/// each definition through [`canonical_json`] (object keys sorted) and keeps the
/// array order significant: reordering tools changes the prompt.
#[must_use]
pub fn tools_signature(tools: &[Value]) -> String {
    let mut hasher = Sha256::new();
    for tool in tools {
        hasher.update(canonical_json(tool).as_bytes());
        hasher.update([RECORD_SEPARATOR]);
    }
    hex_digest(&hasher.finalize())
}

/// One LLM-bound window: the system prompt, the conversation, and the tools.
///
/// `system_messages` and `context_messages` mirror Python's `ContextWindow`;
/// `tools` is the provider-neutral JSON form of the tool definitions (the same
/// `Value`s `PrepareOptions::tools` carries). `tools_signature` and
/// `message_fingerprints` are derived: they are recomputed by
/// [`ContextWindowSnapshot::recompute_derived`], so a snapshot assembled by
/// hand or loaded from JSON should call it before it is diffed.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ContextWindowSnapshot {
    /// System-level directives, always sent first.
    pub system_messages: Vec<BaseMessage>,
    /// Conversation messages after the system prompt.
    pub context_messages: Vec<BaseMessage>,
    /// Tool definitions available for this call.
    pub tools: Vec<Value>,
    /// Opaque model revision label; a different non-empty value invalidates the
    /// whole cache (see [`detect_context_window_change`]).
    pub model_revision: Option<String>,
    /// Monotonic window sequence label, used by
    /// [`diff_positions`] to report how far the window moved.
    pub position: Option<u64>,
    /// Derived signature of [`ContextWindowSnapshot::tools`].
    pub tools_signature: String,
    /// Derived fingerprints of `system_messages` followed by `context_messages`,
    /// i.e. in the exact order the model receives them.
    pub message_fingerprints: Vec<MessageFingerprint>,
}

impl ContextWindowSnapshot {
    /// Builds a snapshot from owned parts and derives the signatures.
    #[must_use]
    pub fn new(
        system_messages: Vec<BaseMessage>,
        context_messages: Vec<BaseMessage>,
        tools: Vec<Value>,
    ) -> Self {
        let mut snapshot = Self {
            system_messages,
            context_messages,
            tools,
            ..Self::default()
        };
        snapshot.recompute_derived();
        snapshot
    }

    /// Builds a snapshot by cloning borrowed parts and deriving the signatures.
    #[must_use]
    pub fn from_messages(
        system_messages: &[BaseMessage],
        context_messages: &[BaseMessage],
        tools: &[Value],
    ) -> Self {
        Self::new(
            system_messages.to_vec(),
            context_messages.to_vec(),
            tools.to_vec(),
        )
    }

    /// Sets the model revision label.
    #[must_use]
    pub fn with_model_revision(mut self, revision: impl Into<String>) -> Self {
        self.model_revision = Some(revision.into());
        self
    }

    /// Sets the window position label.
    #[must_use]
    pub fn with_position(mut self, position: u64) -> Self {
        self.position = Some(position);
        self
    }

    /// Replaces the tool definitions and refreshes the signature.
    pub fn set_tools(&mut self, tools: Vec<Value>) {
        self.tools = tools;
        self.recompute_derived();
    }

    /// Replaces the context messages and refreshes the fingerprints.
    pub fn set_context_messages(&mut self, messages: Vec<BaseMessage>) {
        self.context_messages = messages;
        self.recompute_derived();
    }

    /// Replaces the system messages and refreshes the fingerprints.
    pub fn set_system_messages(&mut self, messages: Vec<BaseMessage>) {
        self.system_messages = messages;
        self.recompute_derived();
    }

    /// Recomputes [`ContextWindowSnapshot::tools_signature`] and
    /// [`ContextWindowSnapshot::message_fingerprints`] from the current parts.
    pub fn recompute_derived(&mut self) {
        self.tools_signature = tools_signature(&self.tools);
        self.message_fingerprints = self
            .system_messages
            .iter()
            .chain(self.context_messages.iter())
            .map(MessageFingerprint::of)
            .collect();
    }

    /// The messages in the order the model receives them.
    pub fn messages(&self) -> impl Iterator<Item = &BaseMessage> {
        self.system_messages
            .iter()
            .chain(self.context_messages.iter())
    }

    /// Total number of system plus context messages.
    #[must_use]
    pub fn message_count(&self) -> usize {
        self.system_messages.len() + self.context_messages.len()
    }

    /// Number of leading system messages.
    #[must_use]
    pub fn system_message_count(&self) -> usize {
        self.system_messages.len()
    }
}

/// How a window changed relative to the previously tracked one.
///
/// The variants are ordered by descending information: a caller that only needs
/// "may I reuse the prefix?" can call
/// [`ContextWindowChange::invalidates_cache`] instead of matching.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ContextWindowChange {
    /// The window is byte-identical; every message stays reusable.
    Unchanged {
        /// Number of messages in the (identical) window.
        #[serde(default)]
        messages: usize,
    },
    /// The window only grew: the old messages are an exact prefix of the new
    /// ones.
    Appended {
        /// Number of leading messages that are unchanged, i.e. the old length.
        prefix_messages: usize,
    },
    /// The first divergence sits inside the system prompt.
    SystemChanged {
        /// Index of the first system message that changed.
        index: usize,
    },
    /// The tool definitions changed.
    ///
    /// Tool serialization sits **right after the system prompt**, so *any* tool
    /// change — including an append at the end of the list — shifts every
    /// following conversation token. Message-suffix invalidation is therefore
    /// mandatory. Detection always reports
    /// `requires_suffix_invalidation: true`; the field stays explicit so a
    /// caller can construct `false` for a window that carries no context
    /// messages at all.
    ToolsChanged {
        /// Whether the messages after the system prompt must be invalidated.
        requires_suffix_invalidation: bool,
    },
    /// The reused prefix was rewritten (messages removed, reordered, or edited).
    Rewritten {
        /// Index of the first message that differs from the old window.
        first_divergence: usize,
    },
    /// The model revision changed; no cached entry survives.
    ModelChanged {
        /// Revision the previous window was bound to.
        from_revision: Option<String>,
        /// Revision the current window is bound to.
        to_revision: Option<String>,
    },
    /// The window content is identical but its position label moved.
    PositionChanged {
        /// Previous position label.
        from: Option<u64>,
        /// Current position label.
        to: Option<u64>,
    },
}

impl ContextWindowChange {
    /// The stable `snake_case` variant name (`"unchanged"`, `"appended"`, ...).
    #[must_use]
    pub const fn kind(&self) -> &'static str {
        match self {
            Self::Unchanged { .. } => "unchanged",
            Self::Appended { .. } => "appended",
            Self::SystemChanged { .. } => "system_changed",
            Self::ToolsChanged { .. } => "tools_changed",
            Self::Rewritten { .. } => "rewritten",
            Self::ModelChanged { .. } => "model_changed",
            Self::PositionChanged { .. } => "position_changed",
        }
    }

    /// Number of leading messages whose cached prefix survives.
    ///
    /// `SystemChanged` reports its divergence index (the system messages before
    /// it still match); a `ToolsChanged` window reports `0` because the tool
    /// change invalidates every message after the system prompt and the diff
    /// alone does not know how many system messages there were.
    #[must_use]
    pub const fn reusable_prefix_len(&self) -> usize {
        match self {
            Self::Unchanged { messages } => *messages,
            Self::Appended { prefix_messages } => *prefix_messages,
            Self::SystemChanged { index } => *index,
            Self::Rewritten { first_divergence } => *first_divergence,
            Self::ToolsChanged { .. }
            | Self::ModelChanged { .. }
            | Self::PositionChanged { .. } => 0,
        }
    }

    /// Whether any prompt-cache entry must be dropped.
    ///
    /// Appending keeps the whole prefix warm, and a position-only change leaves
    /// the prompt byte-identical (the label is telemetry, not payload), so both
    /// report `false`.
    #[must_use]
    pub const fn invalidates_cache(&self) -> bool {
        !matches!(
            self,
            Self::Unchanged { .. } | Self::Appended { .. } | Self::PositionChanged { .. }
        )
    }

    /// A compact, serde-serializable summary of the prompt-cache outcome.
    ///
    /// `reason` is a stable code: `unchanged`, `appended`, `system_changed`,
    /// `tools_changed`, `rewritten`, `model_changed` or `position_changed`.
    #[must_use]
    pub fn prompt_cache_impact(&self) -> PromptCacheImpact {
        let reused_messages = self.reusable_prefix_len();
        match self {
            Self::Unchanged { .. } => PromptCacheImpact {
                reused_messages,
                invalidated_messages: Some(0),
                reason: "unchanged".to_string(),
            },
            Self::Appended { .. } => PromptCacheImpact {
                reused_messages,
                invalidated_messages: Some(0),
                reason: "appended".to_string(),
            },
            Self::SystemChanged { index } => PromptCacheImpact {
                reused_messages,
                invalidated_messages: None,
                reason: format!("system_changed at index {index}"),
            },
            Self::ToolsChanged { .. } => PromptCacheImpact {
                reused_messages,
                invalidated_messages: None,
                reason: "tools_changed invalidates the message suffix".to_string(),
            },
            Self::Rewritten { first_divergence } => PromptCacheImpact {
                reused_messages,
                invalidated_messages: None,
                reason: format!("rewritten from index {first_divergence}"),
            },
            Self::ModelChanged {
                from_revision,
                to_revision,
            } => PromptCacheImpact {
                reused_messages,
                invalidated_messages: None,
                reason: format!(
                    "model_changed from {} to {}",
                    from_revision.as_deref().unwrap_or("<none>"),
                    to_revision.as_deref().unwrap_or("<none>")
                ),
            },
            Self::PositionChanged { from, to } => PromptCacheImpact {
                reused_messages,
                invalidated_messages: Some(0),
                reason: format!(
                    "position_changed from {} to {}",
                    format_position(*from),
                    format_position(*to)
                ),
            },
        }
    }
}

/// Renders a position label for a log/summary string.
fn format_position(position: Option<u64>) -> String {
    position.map_or_else(|| "<none>".to_string(), |value| value.to_string())
}

/// The prompt-cache outcome of one [`ContextWindowChange`].
///
/// `invalidated_messages` is `None` when the change does not reveal how long the
/// old window was (a system or tool rewrite only names the divergence point), so
/// `Some(0)` — "definitely nothing" — is never confused with "unknown".
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct PromptCacheImpact {
    /// Messages whose cached prefix survives the change.
    pub reused_messages: usize,
    /// Old-window messages that must be invalidated, when the diff knows the old
    /// window length.
    pub invalidated_messages: Option<usize>,
    /// Short, human-readable reason naming the change.
    pub reason: String,
}

/// Index of the first differing fingerprint, or `None` when `current` only
/// appends to (or equals) `previous`.
///
/// This is Python's `first_changed_index`: an append-only list is "unchanged"
/// because every token up to the old length stays cache-warm. A shorter
/// `current` whose shared prefix matches reports `current.len()` as the first
/// change.
fn first_divergence(
    previous: &[MessageFingerprint],
    current: &[MessageFingerprint],
) -> Option<usize> {
    let shared = previous.len().min(current.len());
    for index in 0..shared {
        if previous[index] != current[index] {
            return Some(index);
        }
    }
    if current.len() < previous.len() {
        Some(current.len())
    } else {
        None
    }
}

/// Compares the previously tracked window with the current one.
///
/// Precedence of the reported variant, most severe first:
///
/// 1. [`ContextWindowChange::ModelChanged`] when both snapshots carry a
///    non-empty, different `model_revision` (no cached entry can survive a model
///    switch);
/// 2. a message-level change — [`ContextWindowChange::SystemChanged`] when the
///    divergence sits inside the system prompt, otherwise
///    [`ContextWindowChange::Rewritten`];
/// 3. [`ContextWindowChange::ToolsChanged`] when the message prefix is reused
///    but the tool signature moved (an append is enough: tools serialize
///    directly after the system prompt);
/// 4. [`ContextWindowChange::Appended`] when the window only grew;
/// 5. [`ContextWindowChange::PositionChanged`] when the payload is identical but
///    the position label moved;
/// 6. [`ContextWindowChange::Unchanged`] otherwise.
///
/// Detection is order-insensitive for tool objects (keys are canonicalized) but
/// order-sensitive for the tool list and for messages.
#[must_use]
pub fn detect_context_window_change(
    previous: &ContextWindowSnapshot,
    current: &ContextWindowSnapshot,
) -> ContextWindowChange {
    let previous_revision = previous
        .model_revision
        .as_deref()
        .filter(|revision| !revision.is_empty());
    let current_revision = current
        .model_revision
        .as_deref()
        .filter(|revision| !revision.is_empty());
    if let (Some(from), Some(to)) = (previous_revision, current_revision) {
        if from != to {
            return ContextWindowChange::ModelChanged {
                from_revision: Some(from.to_string()),
                to_revision: Some(to.to_string()),
            };
        }
    }

    if let Some(index) = first_divergence(
        &previous.message_fingerprints,
        &current.message_fingerprints,
    ) {
        return if index < previous.system_messages.len() {
            ContextWindowChange::SystemChanged { index }
        } else {
            ContextWindowChange::Rewritten {
                first_divergence: index,
            }
        };
    }
    if previous.tools_signature != current.tools_signature {
        return ContextWindowChange::ToolsChanged {
            requires_suffix_invalidation: true,
        };
    }
    if previous.message_fingerprints.len() < current.message_fingerprints.len() {
        return ContextWindowChange::Appended {
            prefix_messages: previous.message_fingerprints.len(),
        };
    }
    if previous.position != current.position {
        return ContextWindowChange::PositionChanged {
            from: previous.position,
            to: current.position,
        };
    }
    ContextWindowChange::Unchanged {
        messages: current.message_fingerprints.len(),
    }
}

/// One change in the `model_revision`/`position` labels of a window.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PositionChange {
    /// The model revision moved; every cached KV entry is invalid.
    ModelRevision {
        /// Previous revision.
        from: Option<String>,
        /// Current revision.
        to: Option<String>,
    },
    /// The window moved to a different sequence position.
    Position {
        /// Previous position label.
        from: Option<u64>,
        /// Current position label.
        to: Option<u64>,
        /// Signed `to - from` when both labels are known and the difference
        /// fits an `i64`; `None` otherwise.
        #[serde(default)]
        delta: Option<i64>,
    },
}

impl PositionChange {
    /// Whether this change invalidates every cached entry (a revision move
    /// does; a position move alone does not).
    #[must_use]
    pub const fn invalidates_everything(&self) -> bool {
        matches!(self, Self::ModelRevision { .. })
    }
}

/// Diffs the `model_revision` and `position` labels of two windows.
///
/// A revision change is reported first and invalidates everything; a position
/// shift alone reports the label delta. An absent label on either side is not a
/// change (there is nothing to compare), so an empty result means "no label
/// moved". Message/tool content is deliberately *not* inspected here — use
/// [`detect_context_window_change`] for that.
#[must_use]
pub fn diff_positions(
    previous: &ContextWindowSnapshot,
    current: &ContextWindowSnapshot,
) -> Vec<PositionChange> {
    let mut changes = Vec::new();
    let previous_revision = previous
        .model_revision
        .as_deref()
        .filter(|revision| !revision.is_empty());
    let current_revision = current
        .model_revision
        .as_deref()
        .filter(|revision| !revision.is_empty());
    if let (Some(from), Some(to)) = (previous_revision, current_revision) {
        if from != to {
            changes.push(PositionChange::ModelRevision {
                from: Some(from.to_string()),
                to: Some(to.to_string()),
            });
        }
    }
    if previous.position != current.position {
        let delta = match (previous.position, current.position) {
            (Some(from), Some(to)) => i64::try_from(i128::from(to) - i128::from(from)).ok(),
            _ => None,
        };
        changes.push(PositionChange::Position {
            from: previous.position,
            to: current.position,
            delta,
        });
    }
    changes
}
