//! The incremental skill-notice session layer over [`crate::discovery::notice`].
//!
//! Ports Python `openjiuwen.symphony.discovery.toolkit`: a per-session-scope tool id
//! ([`tool_id`]), card normalization ([`normalize_cards`]), the bounded notice-cursor
//! registry ([`IncrementalNoticeRegistry`]) and the [`IncrementalNoticeSession`] that
//! only acknowledges a reminder once it fits the delivered output budget.
//!
//! Python keeps one process-global `OrderedDict` of notice states keyed by
//! `"<tool_name>__<sha256>"`, evicts the least-recently-used entry past 1 024 sessions,
//! and acknowledges a reminder in a `try`/`finally` block, rolling the cursor back when
//! the delivered result would exceed its budget. [`IncrementalNoticeRegistry`] is the
//! owned, thread-safe replacement for that global map: one instance per host, or one per
//! test, with no process-wide state and no `await` anywhere in this module.
//!
//! The reminder text is shared with [`crate::discovery::notice`] through its
//! `render_reminder_rows` helper, so both notice paths emit byte-identical
//! `<system-reminder>` blocks.
//!
//! ```
//! # use openjiuwen_symphony::discovery::toolkit::{
//! #     IncrementalNoticeRegistry, SkillCard, SkillCards, SKILL_INDEX_TOOL_NAME,
//! # };
//! # fn card(name: &str) -> SkillCard {
//! #     SkillCard { name: Some(name.to_string()), description: None }
//! # }
//! # fn run() -> Result<(), openjiuwen_symphony::Error> {
//! let baseline: SkillCards = [("alpha".to_string(), card("Alpha"))].into_iter().collect();
//! let grown: SkillCards = [
//!     ("alpha".to_string(), card("Alpha")),
//!     ("beta".to_string(), card("Beta")),
//! ]
//! .into_iter()
//! .collect();
//! let registry = IncrementalNoticeRegistry::new();
//! registry.initialize("session-1", &baseline, SKILL_INDEX_TOOL_NAME)?;
//! let reminder = registry.consume("session-1", &grown, 4_000, SKILL_INDEX_TOOL_NAME)?;
//! assert!(reminder.is_some_and(|text| text.contains("- Beta: Beta")));
//! assert!(registry
//!     .consume("session-1", &grown, 4_000, SKILL_INDEX_TOOL_NAME)?
//!     .is_none());
//! # Ok(())
//! # }
//! ```
//!
//! Documented differences from Python:
//!
//! - Python's `casefold()` ordering becomes [`str::to_lowercase`], which is identical
//!   for ASCII names and the practical choice for the rest;
//! - a card field that is blank after trimming falls back to the next source, where
//!   Python only falls back on an empty (falsy) value;
//! - `max_chars == 0` is a typed [`Error::Validation`](crate::error::Error) naming
//!   `max_chars` instead of a `ValueError`.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Mutex, MutexGuard, PoisonError};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::discovery::models::SkillInventory;
use crate::discovery::notice::{render_reminder_rows, safe_text, shorten_description};
use crate::error::{validation, Result};

pub use crate::discovery::notice::{
    DEFAULT_NOTICE_MAX_CHARS, NOTICE_DESCRIPTION_CHARS, SKILL_INDEX_TOOL_NAME,
};

/// The maximum number of session cursors a registry keeps (Python's
/// `_MAX_SESSION_STATES`); the least-recently-used cursor is evicted past it.
pub const MAX_SESSION_STATES: usize = 1_024;

/// The tool-id hash domain, Python's `_TOOL_ID_DOMAIN` without its trailing NUL.
///
/// [`tool_id`] hashes `TOOL_ID_DOMAIN ++ "\0" ++ tool_name ++ "\0" ++ session_scope`,
/// so the domain separator is inserted explicitly between the three parts.
pub const TOOL_ID_DOMAIN: &str = "openjiuwen.skill-index.v1";

/// One raw skill card as a caller supplies it.
///
/// Mirrors a value of Python's `Mapping[str, Mapping[str, str]]`: `name` and
/// `description` are optional, and [`normalize_cards`] derives the rest.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct SkillCard {
    /// The display name; the worker id is used when it is unset or blank.
    pub name: Option<String>,
    /// The human-readable description; the resolved name is used when it is unset or
    /// blank.
    pub description: Option<String>,
}

/// The raw skill cards of one session, keyed by worker id.
pub type SkillCards = BTreeMap<String, SkillCard>;

/// One normalized card: Python's `(name, description)` snapshot plus its worker id.
///
/// Both text fields are whitespace-collapsed, angle-bracket escaped, and the
/// description is shortened to [`NOTICE_DESCRIPTION_CHARS`].
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct SkillCardSnapshot {
    /// The trimmed worker id the card was registered under.
    pub worker_id: String,
    /// The collapsed and escaped display name.
    pub name: String,
    /// The collapsed, escaped and shortened description.
    pub description: String,
}

/// Builds the per-session tool id Python's `_tool_id` derives.
///
/// The id is `"<tool_name>__<sha256 hex>"`, where the digest covers
/// [`TOOL_ID_DOMAIN`], a NUL, the trimmed `tool_name`, a NUL, and the trimmed
/// `session_scope` — the same bytes Python hashes, so an id computed here matches one
/// computed by the Python SDK.
///
/// # Errors
///
/// [`Error::Validation`](crate::error::Error) naming `session_scope` or
/// `discovery_tool_name` when the corresponding argument is blank after trimming.
pub fn tool_id(session_scope: &str, tool_name: &str) -> Result<String> {
    let scope = session_scope.trim();
    if scope.is_empty() {
        return Err(validation("session_scope", "must not be blank"));
    }
    let name = tool_name.trim();
    if name.is_empty() {
        return Err(validation("discovery_tool_name", "must not be blank"));
    }
    let mut hasher = Sha256::new();
    hasher.update(TOOL_ID_DOMAIN.as_bytes());
    hasher.update([0u8]);
    hasher.update(name.as_bytes());
    hasher.update([0u8]);
    hasher.update(scope.as_bytes());
    Ok(format!("{name}__{:x}", hasher.finalize()))
}

/// Normalizes raw cards exactly as Python's `_normalize_cards` does.
///
/// A card whose worker id is blank is skipped; the name falls back to the worker id and
/// the description to the name, each collapsed and angle-bracket escaped through the
/// shared `notice::safe_text`, and the description is shortened to
/// [`NOTICE_DESCRIPTION_CHARS`] with `...` on a word boundary. The result is keyed and
/// ordered by worker id, so iteration order never depends on the input map.
#[must_use]
pub fn normalize_cards(cards: &SkillCards) -> BTreeMap<String, SkillCardSnapshot> {
    let mut normalized = BTreeMap::new();
    for (raw_id, card) in cards {
        let worker_id = raw_id.trim();
        if worker_id.is_empty() {
            continue;
        }
        let name_source = card
            .name
            .as_deref()
            .map(str::trim)
            .filter(|name| !name.is_empty())
            .unwrap_or(worker_id);
        let name = safe_text(name_source);
        let description_source = card
            .description
            .as_deref()
            .map(str::trim)
            .filter(|description| !description.is_empty())
            .unwrap_or(&name);
        let description =
            shorten_description(&safe_text(description_source), NOTICE_DESCRIPTION_CHARS);
        normalized.insert(
            worker_id.to_string(),
            SkillCardSnapshot {
                worker_id: worker_id.to_string(),
                name,
                description,
            },
        );
    }
    normalized
}

/// Projects a [`SkillInventory`] onto the raw cards the notice layer consumes.
///
/// Every record becomes `worker_id -> {name, description}`, the shape Python's discovery
/// tool hands to `initialize_incremental_skill_notice_state` and
/// `consume_incremental_skill_reminder`.
#[must_use]
pub fn cards_from_inventory(inventory: &SkillInventory) -> SkillCards {
    inventory
        .items
        .iter()
        .map(|record| {
            (
                record.worker_id.clone(),
                SkillCard {
                    name: Some(record.name.clone()),
                    description: Some(record.description.clone()),
                },
            )
        })
        .collect()
}

/// Renders Python's `_render_reminder` for the added cards.
///
/// The rows are `- <name>: <description>`, ordered by `name.to_lowercase()`, joined by
/// newlines, and wrapped in the exact `<system-reminder>` block; when that block exceeds
/// `max_chars` (or `max_chars` is `0`) the compact form is rendered instead: one line
/// naming the number of new skills and the backticked `tool_name`. `None` when `added`
/// is empty.
#[must_use]
pub fn render_reminder(
    added: &[SkillCardSnapshot],
    max_chars: usize,
    tool_name: &str,
) -> Option<String> {
    if added.is_empty() {
        return None;
    }
    let rows: Vec<(String, String)> = added
        .iter()
        .map(|card| (card.name.clone(), card.description.clone()))
        .collect();
    render_reminder_rows(&rows, max_chars, tool_name)
}

/// One tracked session cursor inside a [`IncrementalNoticeRegistry`].
#[derive(Debug, Clone)]
struct NoticeEntry {
    /// The normalized cards seen by the last acknowledged consume.
    cards: BTreeMap<String, SkillCardSnapshot>,
    /// The monotonic recency stamp of the last lookup.
    sequence: u64,
}

/// The registry's guarded state: cursors plus the recency counter.
#[derive(Debug, Default)]
struct RegistryInner {
    /// One cursor per tool id, ordered by id (not by recency).
    entries: BTreeMap<String, NoticeEntry>,
    /// The next recency stamp; larger means more recently used.
    next_sequence: u64,
}

impl RegistryInner {
    /// Looks `tool_id` up, inserting `snapshot` as the baseline when it is unknown, and
    /// stamps it as the most recently used cursor.
    ///
    /// Past `capacity` cursors the least-recently-used one is dropped; the cursor just
    /// touched always carries the largest stamp and is therefore never the victim.
    fn touch(
        &mut self,
        tool_id: &str,
        capacity: usize,
        snapshot: &BTreeMap<String, SkillCardSnapshot>,
    ) -> &mut NoticeEntry {
        let sequence = self.next_sequence;
        self.next_sequence += 1;
        {
            let entry = self
                .entries
                .entry(tool_id.to_string())
                .or_insert_with(|| NoticeEntry {
                    cards: snapshot.clone(),
                    sequence,
                });
            entry.sequence = sequence;
        }
        while self.entries.len() > capacity {
            let oldest = self
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.sequence)
                .map(|(id, _)| id.clone());
            let Some(oldest) = oldest else {
                break;
            };
            if oldest == tool_id {
                break;
            }
            self.entries.remove(&oldest);
        }
        self.entries
            .entry(tool_id.to_string())
            .or_insert_with(|| NoticeEntry {
                cards: snapshot.clone(),
                sequence,
            })
    }
}

/// Python's process-global `_STATES`, as an owned, thread-safe registry.
///
/// One cursor per `(session_scope, tool_name)` tool id, bounded by
/// [`MAX_SESSION_STATES`] (or an explicit capacity) with least-recently-used eviction.
/// The lock is an ordinary [`Mutex`] held only for the map work — this module has no
/// `await`, so a guard never crosses an async suspension point — and a poisoned lock is
/// recovered from instead of panicking, because a notice cursor is not worth taking a
/// session down for.
#[derive(Debug)]
pub struct IncrementalNoticeRegistry {
    /// The guarded cursors; see [`RegistryInner`].
    inner: Mutex<RegistryInner>,
    /// The maximum number of cursors kept.
    capacity: usize,
}

impl Default for IncrementalNoticeRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl IncrementalNoticeRegistry {
    /// Builds a registry holding at most [`MAX_SESSION_STATES`] cursors.
    #[must_use]
    pub fn new() -> Self {
        Self::with_capacity(MAX_SESSION_STATES)
    }

    /// Builds a registry holding at most `capacity` cursors; `0` is clamped to `1`, so a
    /// registry always keeps at least the cursor it just created.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            inner: Mutex::new(RegistryInner::default()),
            capacity: capacity.max(1),
        }
    }

    /// The number of cursors currently tracked.
    #[must_use]
    pub fn len(&self) -> usize {
        self.lock().entries.len()
    }

    /// Whether no cursor is tracked.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The maximum number of cursors this registry keeps.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Drops every cursor, Python's `clear_incremental_skill_notice_states`.
    pub fn clear(&self) {
        self.lock().entries.clear();
    }

    /// Seeds the baseline for `session_scope` without consuming a delta.
    ///
    /// Python's `initialize_incremental_skill_notice_state`. A later
    /// [`consume`](Self::consume) reports only the skills that were not in this card map.
    /// An existing cursor is *not* reset: only [`consume`](Self::consume) moves it, so a
    /// re-initialization with a grown catalog still leaves the new skills pending.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`](crate::error::Error) when `session_scope` or `tool_name` is
    /// blank after trimming.
    pub fn initialize(
        &self,
        session_scope: &str,
        cards: &SkillCards,
        tool_name: &str,
    ) -> Result<()> {
        let id = tool_id(session_scope, tool_name)?;
        let current = normalize_cards(cards);
        let mut inner = self.lock();
        inner.touch(&id, self.capacity, &current);
        Ok(())
    }

    /// Consumes the one-shot reminder for skills added since this session's baseline.
    ///
    /// Python's `consume_incremental_skill_reminder`: the delta is computed against the
    /// cursor, the reminder is rendered at most once per added set, and the cursor then
    /// acknowledges the new cards — so an identical second call returns `None`. An
    /// unknown session is baselined with `cards` and therefore reports nothing on its
    /// first call.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`](crate::error::Error) naming `max_chars` when it is `0` (this
    /// is checked before any cursor work, exactly like Python), or naming
    /// `session_scope`/`discovery_tool_name` when either is blank.
    pub fn consume(
        &self,
        session_scope: &str,
        cards: &SkillCards,
        max_chars: usize,
        tool_name: &str,
    ) -> Result<Option<String>> {
        if max_chars == 0 {
            return Err(validation("max_chars", "must be positive"));
        }
        let id = tool_id(session_scope, tool_name)?;
        let current = normalize_cards(cards);
        let mut inner = self.lock();
        let entry = inner.touch(&id, self.capacity, &current);
        let added: Vec<SkillCardSnapshot> = current
            .values()
            .filter(|card| !entry.cards.contains_key(&card.worker_id))
            .cloned()
            .collect();
        let reminder = render_reminder(&added, max_chars, tool_name);
        entry.cards = current;
        Ok(reminder)
    }

    /// The tracked tool ids, sorted; diagnostics only.
    #[must_use]
    pub fn tool_ids(&self) -> Vec<String> {
        self.lock().entries.keys().cloned().collect()
    }

    /// Locks the guarded cursors, recovering from a poisoned lock.
    fn lock(&self) -> MutexGuard<'_, RegistryInner> {
        self.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// One session's notice cursor: Python's `IncrementalNoticeSession` minus the global map.
///
/// Unlike the registry, an owned session renders a reminder and lets the caller decide
/// whether the delivered result had room for it: [`append`](Self::append) only
/// acknowledges a reminder once it fits the `output_budget`, so a reminder that did not
/// fit is offered again on the next call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IncrementalNoticeSession {
    /// The character budget of the rendered reminder.
    max_chars: usize,
    /// The tool id this session's cursor is keyed by.
    tool_id: String,
    /// The normalized cards already announced (or, after a rollback, the acknowledged
    /// subset of them).
    seen: BTreeMap<String, SkillCardSnapshot>,
}

impl IncrementalNoticeSession {
    /// Opens a session baselined with `cards`, consuming no delta.
    ///
    /// Python's `IncrementalNoticeSession.__init__`, which always keys the cursor by
    /// [`SKILL_INDEX_TOOL_NAME`].
    ///
    /// # Errors
    ///
    /// [`Error::Validation`](crate::error::Error) naming `max_chars` when it is `0`, or
    /// `session_scope` when it is blank after trimming.
    pub fn new(session_scope: &str, cards: &SkillCards, max_chars: usize) -> Result<Self> {
        if max_chars == 0 {
            return Err(validation("max_chars", "must be positive"));
        }
        let tool_id = tool_id(session_scope, SKILL_INDEX_TOOL_NAME)?;
        Ok(Self {
            max_chars,
            tool_id,
            seen: normalize_cards(cards),
        })
    }

    /// The tool id this session's cursor is keyed by.
    #[must_use]
    pub fn tool_id(&self) -> &str {
        &self.tool_id
    }

    /// The character budget of the rendered reminder.
    #[must_use]
    pub const fn max_chars(&self) -> usize {
        self.max_chars
    }

    /// The cards already acknowledged by this session.
    #[must_use]
    pub fn seen(&self) -> &BTreeMap<String, SkillCardSnapshot> {
        &self.seen
    }

    /// Appends the pending reminder to `output`, acknowledging it only if it fits.
    ///
    /// Python's `IncrementalNoticeSession.append`: the delta is computed against
    /// [`seen`](Self::seen), the reminder rendered at [`max_chars`](Self::max_chars), and
    /// the combination is `format!("{output}\n\n{reminder}")` (the reminder alone when
    /// `output` is empty). When `output_budget` is `Some(budget)` and that combination is
    /// longer than `budget`, the unchanged `output` and `None` are returned *without*
    /// acknowledging, so the same reminder is offered again on the next call. A call with
    /// nothing added still acknowledges the new cards, matching Python's `finally` block.
    #[must_use]
    pub fn append(
        &mut self,
        output: &str,
        cards: &SkillCards,
        output_budget: Option<usize>,
    ) -> (String, Option<String>) {
        let current = normalize_cards(cards);
        let added: Vec<SkillCardSnapshot> = current
            .values()
            .filter(|card| !self.seen.contains_key(&card.worker_id))
            .cloned()
            .collect();
        let Some(reminder) = render_reminder(&added, self.max_chars, SKILL_INDEX_TOOL_NAME) else {
            self.seen = current;
            return (output.to_string(), None);
        };
        let combined = if output.is_empty() {
            reminder.clone()
        } else {
            format!("{output}\n\n{reminder}")
        };
        if output_budget.is_some_and(|budget| combined.chars().count() > budget) {
            let pending: BTreeSet<&str> =
                added.iter().map(|card| card.worker_id.as_str()).collect();
            self.seen = current
                .into_iter()
                .filter(|(worker_id, _)| !pending.contains(worker_id.as_str()))
                .collect();
            return (output.to_string(), None);
        }
        self.seen = current;
        (combined, Some(reminder))
    }
}
