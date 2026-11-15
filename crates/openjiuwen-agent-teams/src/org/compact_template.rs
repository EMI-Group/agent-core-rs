//! Template-driven compaction: Python's `compact_context` split/direction contract, offline.
//!
//! Python's `openjiuwen/agent_teams/fork_compact.py` compacts *one side* of a forked history at a
//! `split_at` index: `direction="before"` (the default) compresses the head and keeps the tail
//! verbatim, `direction="after"` keeps the head verbatim and compresses the tail. Every
//! `SystemMessage` is removed before the split is computed, and the compressed side is replaced by
//! one summary message inserted on its own side of the kept block.
//!
//! This module is that contract without the live agent: [`compact_with_template`] is a pure
//! function of the transcript and a [`CompactTemplate`], so a fork can be planned offline and
//! replayed byte-for-byte. The summary is a deterministic template line
//! instead of an LLM answer: it reports the same counts as [`super::fork_compact::compact_transcript`]
//! plus the compressed side, the split point and the facts Python's `_COMPACTION_PROMPT` asks the
//! model to retain (file paths, decisions, errors, user feedback).
//!
//! Deliberate deviations from Python, each documented at its item:
//!
//! - Python *skips* compaction when `split_at <= 0 || split_at >= len`; this port reports a typed
//!   [`Error::validation`] instead, because a caller that asked for a compaction wants to know the
//!   request was impossible;
//! - the summary is deterministic (no model call), so the same input always renders the same line;
//! - a total byte budget ([`CompactTemplate::total_budget_bytes`]) lets a fork fit a hard token
//!   ceiling; Python has no such knob;
//! - compaction never grows a transcript: a summary that would cost more than the messages it
//!   replaces is omitted, and [`TemplateCompaction::summary_omitted`] records why.
//!
//! Everything here is pure: no I/O, no clock, no randomness, no process-global state.
//!
//! Length note: this file is a little over the 1000-line guide on purpose. Direction, side, budget
//! tiers, retention and the deterministic fact extractor are one policy pipeline — splitting them
//! across modules would scatter the rules that only read correctly together.

use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};

use super::fork_compact::{
    role_count_parts, shortened_tool_content, CompactRole, TranscriptMessage,
    DEFAULT_MAX_TOOL_RESULT_BYTES, DEFAULT_TRUNCATED_SUFFIX, SUMMARY_PREFIX,
};

/// Per-category excerpt cap the template summary applies by default.
pub const DEFAULT_TEMPLATE_MAX_FACTS: usize = 4;

/// Maximum characters of one excerpt the template summary keeps by default.
pub const DEFAULT_EXCERPT_CHARS: usize = 48;

/// Highest-priority-first budget order a default [`CompactTemplate`] uses.
///
/// The summary is protected first, then the verbatim side, then the tool results of that side.
pub const DEFAULT_BUDGET_PRIORITY: [BudgetTier; 3] = [
    BudgetTier::Summary,
    BudgetTier::KeptSide,
    BudgetTier::ToolResults,
];

/// Words that mark a transcript line as an error report (case-insensitive).
const ERROR_MARKERS: [&str; 5] = ["error", "failed", "failure", "panic", "exception"];

/// Words that mark a transcript line as a design decision or a chosen pattern (case-insensitive).
const DECISION_MARKERS: [&str; 6] = [
    "decision",
    "decided",
    "architecture",
    "pattern",
    "approach",
    "chose",
];

/// Characters that end a path-like token inside free text.
const PATH_DELIMITERS: [char; 18] = [
    ',', ';', ':', '"', '\'', '`', '(', ')', '[', ']', '{', '}', '<', '>', '|', '=', '!', '?',
];

// ---------------------------------------------------------------------------------------------
// Direction and side
// ---------------------------------------------------------------------------------------------

/// Which side of `split_at` a template compresses.
///
/// The wire strings are Python's `direction` values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum CompactDirection {
    /// Compress `messages[..split_at]` and keep the tail verbatim (Python's default).
    #[default]
    Before,
    /// Keep `messages[..split_at]` verbatim and compress the tail.
    After,
}

impl CompactDirection {
    /// Returns the Python wire string (`before`, `after`).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Before => "before",
            Self::After => "after",
        }
    }

    /// Parses a Python `direction` value.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`] for anything other than `before` or `after`.
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "before" => Ok(Self::Before),
            "after" => Ok(Self::After),
            other => Err(Error::validation(format!(
                "invalid compact direction '{other}': expected 'before' or 'after'"
            ))),
        }
    }

    /// Returns the side this direction compresses.
    #[must_use]
    pub const fn compresses(self) -> CompactionSide {
        match self {
            Self::Before => CompactionSide::Head,
            Self::After => CompactionSide::Tail,
        }
    }
}

/// The side of a transcript that a compaction replaces with its summary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum CompactionSide {
    /// The messages before the split (compressed by [`CompactDirection::Before`]).
    #[default]
    Head,
    /// The messages after the split (compressed by [`CompactDirection::After`]).
    Tail,
}

impl CompactionSide {
    /// Returns the lowercase name used in the summary line (`head`, `tail`).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Head => "head",
            Self::Tail => "tail",
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Budget tiers and summary retention
// ---------------------------------------------------------------------------------------------

/// One tier of a [`CompactTemplate`]'s total byte budget.
///
/// [`CompactTemplate::budget_priority`] lists the tiers highest-priority first: a tier that comes
/// later is reduced *before* an earlier one when the transcript does not fit the total budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BudgetTier {
    /// The summary message. Reducing this tier omits the summary.
    Summary,
    /// The verbatim side's non-tool messages. Reducing this tier drops them farthest-from-split
    /// first.
    KeptSide,
    /// The verbatim side's tool results, beyond their own per-result byte budget. Reducing this
    /// tier drops them farthest-from-split first.
    ToolResults,
}

impl BudgetTier {
    /// Returns the snake-case tier name used in validation messages.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Summary => "summary",
            Self::KeptSide => "kept_side",
            Self::ToolResults => "tool_results",
        }
    }
}

/// Reason a compaction produced no summary line even though it compressed messages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SummaryOmission {
    /// The summary line would have cost more than the messages it replaces, so emitting it would
    /// have grown the transcript.
    Regressive,
    /// The total byte budget had no room for the summary once lower-priority tiers were protected.
    Budget,
}

impl SummaryOmission {
    /// Returns the snake-case reason name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Regressive => "regressive",
            Self::Budget => "budget",
        }
    }
}

/// One fact category Python's `_COMPACTION_PROMPT` asks the summary to retain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SummaryFact {
    /// Every file path that was read or written.
    Paths,
    /// Key code patterns and architectural decisions.
    Decisions,
    /// Errors encountered and how they were resolved.
    Errors,
    /// User feedback and changes of direction.
    Feedback,
}

impl SummaryFact {
    /// Returns the snake-case category name used in the summary line.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Paths => "paths",
            Self::Decisions => "decisions",
            Self::Errors => "errors",
            Self::Feedback => "feedback",
        }
    }
}

/// Which facts a template summary retains, and how verbose each category is.
///
/// Categories render in the fixed order `paths`, `decisions`, `errors`, `feedback` regardless of
/// the order of [`SummaryRetention::facts`], so the summary line is stable.
/// [`SummaryRetention::max_facts`] caps how many excerpts one category lists; `0` makes every
/// category report a count instead of excerpts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct SummaryRetention {
    /// Fact categories to report; an empty list emits no fact segment at all.
    pub facts: Vec<SummaryFact>,
    /// Maximum excerpts listed per category (`0` = count only).
    pub max_facts: usize,
    /// Maximum characters of one excerpt before it is cut with `…`.
    pub max_excerpt_chars: usize,
}

impl Default for SummaryRetention {
    fn default() -> Self {
        Self {
            facts: vec![
                SummaryFact::Paths,
                SummaryFact::Decisions,
                SummaryFact::Errors,
                SummaryFact::Feedback,
            ],
            max_facts: DEFAULT_TEMPLATE_MAX_FACTS,
            max_excerpt_chars: DEFAULT_EXCERPT_CHARS,
        }
    }
}

impl SummaryRetention {
    /// Creates the default retention: every category, four excerpts, 48 characters each.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Restricts the retained categories to `facts`.
    #[must_use]
    pub fn with_facts<I: IntoIterator<Item = SummaryFact>>(mut self, facts: I) -> Self {
        self.facts = facts.into_iter().collect();
        self
    }

    /// Drops one category from [`SummaryRetention::facts`].
    #[must_use]
    pub fn without(mut self, fact: SummaryFact) -> Self {
        self.facts.retain(|retained| *retained != fact);
        self
    }

    /// Sets [`SummaryRetention::max_facts`].
    #[must_use]
    pub fn with_max_facts(mut self, facts: usize) -> Self {
        self.max_facts = facts;
        self
    }

    /// Sets [`SummaryRetention::max_excerpt_chars`].
    #[must_use]
    pub fn with_excerpt_chars(mut self, chars: usize) -> Self {
        self.max_excerpt_chars = chars;
        self
    }

    /// Returns `true` when `fact` is retained.
    #[must_use]
    pub fn retains(&self, fact: SummaryFact) -> bool {
        self.facts.contains(&fact)
    }
}

// ---------------------------------------------------------------------------------------------
// The template
// ---------------------------------------------------------------------------------------------

/// The knobs of one template-driven compaction.
///
/// Defaults reproduce Python's `compact_context` arguments: `direction` is
/// [`CompactDirection::Before`], system messages are stripped and never re-emitted
/// ([`CompactTemplate::keep_system`]), the head of an `after` compaction is widened through
/// trailing tool results ([`CompactTemplate::extend_through_tool_results`]), and there is no total
/// byte budget. [`CompactTemplate::split_at`] has no default: it must be set with
/// [`CompactTemplate::new`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct CompactTemplate {
    /// Index in the system-message-stripped transcript that separates the kept and compressed
    /// sides. Must satisfy `1 <= split_at < <non-system messages>`.
    pub split_at: usize,
    /// Which side of [`CompactTemplate::split_at`] is compressed.
    pub direction: CompactDirection,
    /// Whether system messages are re-emitted at the front of the result.
    ///
    /// They are always stripped before the split is computed; Python then never puts them back,
    /// which is the default here.
    pub keep_system: bool,
    /// Hard ceiling, in byte weight, on the compacted transcript. `None` (the default) leaves only
    /// the "never grow" invariant.
    pub total_budget_bytes: Option<usize>,
    /// Budget tiers, highest priority first. Must be non-empty and duplicate-free.
    pub budget_priority: Vec<BudgetTier>,
    /// Byte budget of one tool result of the kept side.
    pub max_tool_result_bytes: usize,
    /// Appended to a tool result cut to [`CompactTemplate::max_tool_result_bytes`].
    pub truncated_suffix: String,
    /// Whether an [`CompactDirection::After`] compaction widens its verbatim head through trailing
    /// tool results.
    ///
    /// Python does this unconditionally so the kept head never ends with a dangling tool call whose
    /// result sits in the compressed tail.
    pub extend_through_tool_results: bool,
    /// Facts the template summary retains.
    pub retention: SummaryRetention,
}

impl Default for CompactTemplate {
    fn default() -> Self {
        Self {
            split_at: 0,
            direction: CompactDirection::Before,
            keep_system: false,
            total_budget_bytes: None,
            budget_priority: DEFAULT_BUDGET_PRIORITY.to_vec(),
            max_tool_result_bytes: DEFAULT_MAX_TOOL_RESULT_BYTES,
            truncated_suffix: DEFAULT_TRUNCATED_SUFFIX.to_owned(),
            extend_through_tool_results: true,
            retention: SummaryRetention::default(),
        }
    }
}

impl CompactTemplate {
    /// Creates a template at `split_at` with every other knob at its default.
    #[must_use]
    pub fn new(split_at: usize) -> Self {
        Self {
            split_at,
            ..Self::default()
        }
    }

    /// Compresses the head instead of the tail ([`CompactDirection::Before`]).
    #[must_use]
    pub fn before(mut self) -> Self {
        self.direction = CompactDirection::Before;
        self
    }

    /// Keeps the head verbatim and compresses the tail ([`CompactDirection::After`]).
    #[must_use]
    pub fn after(mut self) -> Self {
        self.direction = CompactDirection::After;
        self
    }

    /// Re-emits system messages at the front of the result.
    #[must_use]
    pub fn keeping_system(mut self) -> Self {
        self.keep_system = true;
        self
    }

    /// Sets [`CompactTemplate::total_budget_bytes`].
    #[must_use]
    pub fn with_budget(mut self, bytes: usize) -> Self {
        self.total_budget_bytes = Some(bytes);
        self
    }

    /// Replaces [`CompactTemplate::budget_priority`].
    #[must_use]
    pub fn with_priority(mut self, priority: Vec<BudgetTier>) -> Self {
        self.budget_priority = priority;
        self
    }

    /// Sets [`CompactTemplate::max_tool_result_bytes`].
    #[must_use]
    pub fn with_tool_budget(mut self, bytes: usize) -> Self {
        self.max_tool_result_bytes = bytes;
        self
    }

    /// Stops widening the head of an `after` compaction through trailing tool results.
    #[must_use]
    pub fn without_tool_extension(mut self) -> Self {
        self.extend_through_tool_results = false;
        self
    }

    /// Replaces [`CompactTemplate::retention`].
    #[must_use]
    pub fn with_retention(mut self, retention: SummaryRetention) -> Self {
        self.retention = retention;
        self
    }

    /// Validates the template against a transcript that has `non_system_messages` non-system
    /// messages.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`] when [`CompactTemplate::split_at`] cannot split that
    /// transcript (it must be at least `1` and leave at least one message on each side), or when
    /// [`CompactTemplate::budget_priority`] is empty or repeats a tier.
    pub fn validate(&self, non_system_messages: usize) -> Result<()> {
        if self.split_at == 0 || non_system_messages < 2 || self.split_at >= non_system_messages {
            return Err(Error::validation(format!(
                "compact split_at {} is out of range for {} non-system messages (expected 1..{})",
                self.split_at, non_system_messages, non_system_messages
            )));
        }
        self.squeeze_order().map(|_| ())
    }

    /// Returns the budget tiers lowest priority first, which is the order they are reduced in.
    fn squeeze_order(&self) -> Result<Vec<BudgetTier>> {
        if self.budget_priority.is_empty() {
            return Err(Error::validation(
                "compact template budget_priority must name at least one budget tier",
            ));
        }
        let mut seen: Vec<BudgetTier> = Vec::with_capacity(self.budget_priority.len());
        for tier in &self.budget_priority {
            if seen.contains(tier) {
                return Err(Error::validation(format!(
                    "compact template budget_priority repeats tier '{}'",
                    tier.as_str()
                )));
            }
            seen.push(*tier);
        }
        seen.reverse();
        Ok(seen)
    }
}

// ---------------------------------------------------------------------------------------------
// The result
// ---------------------------------------------------------------------------------------------

/// The result of one template-driven compaction.
///
/// `original_bytes` and `compacted_bytes` are the summed [`TranscriptMessage::byte_len`] of the
/// input and of [`TemplateCompaction::messages`], which makes
/// `compacted_bytes <= original_bytes` a checkable invariant rather than a promise.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case", default)]
pub struct TemplateCompaction {
    /// The compacted transcript: the summary on its own side of the verbatim block.
    pub messages: Vec<TranscriptMessage>,
    /// The summary line that was emitted, when one was.
    pub summary: Option<String>,
    /// Why no summary was emitted even though messages were compressed.
    pub summary_omitted: Option<SummaryOmission>,
    /// Direction the template asked for.
    pub direction: CompactDirection,
    /// Side that was actually compressed (`head` for `before`, `tail` for `after`).
    pub compressed_side: CompactionSide,
    /// Split index the compaction used in the stripped transcript, after any tool-result
    /// widening.
    pub split_at: usize,
    /// Split index the template asked for, before any tool-result widening.
    pub requested_split_at: usize,
    /// Number of messages replaced by the summary.
    pub compressed_messages: usize,
    /// Number of system messages stripped before the split was computed.
    pub stripped_system_messages: usize,
    /// Number of kept-side messages that survived verbatim.
    pub kept_messages: usize,
    /// Number of kept-side and system messages removed by the total byte budget.
    ///
    /// A summary the budget removed is reported by [`TemplateCompaction::summary_omitted`]
    /// instead.
    pub squeezed_messages: usize,
    /// Number of surviving tool results shortened to the template's tool budget.
    pub truncated_tool_results: usize,
    /// Summed byte weight of the input transcript.
    pub original_bytes: usize,
    /// Summed byte weight of [`TemplateCompaction::messages`].
    pub compacted_bytes: usize,
}

impl TemplateCompaction {
    /// Returns `true` when the result drops nothing, truncates nothing and squeezes nothing.
    #[must_use]
    pub fn is_lossless(&self) -> bool {
        self.compressed_messages == 0
            && self.stripped_system_messages == 0
            && self.squeezed_messages == 0
            && self.truncated_tool_results == 0
    }

    /// Renders the compacted transcript as newline-separated `[role] content` lines.
    #[must_use]
    pub fn text(&self) -> String {
        self.messages
            .iter()
            .map(TranscriptMessage::render)
            .collect::<Vec<_>>()
            .join("\n")
    }
}

// ---------------------------------------------------------------------------------------------
// The entry point
// ---------------------------------------------------------------------------------------------

/// Compact one side of a transcript at [`CompactTemplate::split_at`], keeping the other verbatim.
///
/// Rules, in order:
///
/// 1. Every [`CompactRole::System`] message is stripped before the split is computed (Python's
///    order), and counted in [`TemplateCompaction::stripped_system_messages`]. They are re-emitted
///    at the front only when [`CompactTemplate::keep_system`].
/// 2. The split is validated against the stripped transcript (`1 <= split_at < len`).
/// 3. [`CompactDirection::Before`] compresses `stripped[..split_at]` and keeps `stripped[split_at..]`
///    verbatim; [`CompactDirection::After`] keeps `stripped[..head_end]` and compresses
///    `stripped[head_end..]`, where `head_end` starts at `split_at` and — when
///    [`CompactTemplate::extend_through_tool_results`] — advances through trailing tool results.
/// 4. The compressed side becomes exactly one summary message ([`SUMMARY_PREFIX`], the side, the
///    split, the role counts and the retained facts), placed first for `before` and last for
///    `after`; it is omitted when the compressed side is empty.
/// 5. Surviving tool results longer than [`CompactTemplate::max_tool_result_bytes`] are cut on a
///    `char` boundary and [`CompactTemplate::truncated_suffix`] is appended.
/// 6. Compaction never grows the transcript: a summary that would cost more than the messages it
///    replaces is omitted with [`SummaryOmission::Regressive`].
/// 7. When [`CompactTemplate::total_budget_bytes`] is set, the tiers of
///    [`CompactTemplate::budget_priority`] are reduced lowest priority first until the result fits
///    `min(budget, original_bytes)`: tool results and kept-side messages are dropped
///    farthest-from-split first, and the summary is omitted with [`SummaryOmission::Budget`]. Kept
///    system messages are the last resort, so the budget always holds.
///
/// # Errors
///
/// Returns [`Error::validation`] when the template is invalid for this transcript: the split is out
/// of range, or the budget priority list is empty or repeats a tier.
pub fn compact_with_template(
    messages: &[TranscriptMessage],
    template: &CompactTemplate,
) -> Result<TemplateCompaction> {
    let original_bytes: usize = messages.iter().map(TranscriptMessage::byte_len).sum();

    let mut system_messages: Vec<&TranscriptMessage> = Vec::new();
    let mut stripped: Vec<&TranscriptMessage> = Vec::with_capacity(messages.len());
    for message in messages {
        if message.role == CompactRole::System {
            system_messages.push(message);
        } else {
            stripped.push(message);
        }
    }

    template.validate(stripped.len())?;
    let squeeze_order = template.squeeze_order()?;

    let total_messages = stripped.len();
    let split_at = match template.direction {
        CompactDirection::Before => template.split_at,
        CompactDirection::After => {
            let mut head_end = template.split_at;
            if template.extend_through_tool_results {
                while head_end < total_messages && stripped[head_end].role == CompactRole::Tool {
                    head_end += 1;
                }
            }
            head_end
        }
    };
    let (kept_source, compressed): (&[&TranscriptMessage], &[&TranscriptMessage]) =
        match template.direction {
            CompactDirection::Before => (
                &stripped[template.split_at..],
                &stripped[..template.split_at],
            ),
            CompactDirection::After => (&stripped[..split_at], &stripped[split_at..]),
        };

    let mut kept: Vec<TranscriptMessage> = kept_source
        .iter()
        .map(|message| (*message).clone())
        .collect();
    let truncated_tool_results = truncate_tool_results(&mut kept, template);

    let facts = collect_facts(compressed, &template.retention);
    let summary_line = if compressed.is_empty() {
        None
    } else {
        Some(template_summary_line(
            template,
            split_at,
            total_messages,
            compressed,
            &facts,
        ))
    };

    let systems: Vec<TranscriptMessage> = if template.keep_system {
        system_messages
            .iter()
            .map(|message| (*message).clone())
            .collect()
    } else {
        Vec::new()
    };
    let summary_bytes = summary_line
        .as_ref()
        .map_or(0, |line| TranscriptMessage::system(line.clone()).byte_len());
    let mut budget = BudgetState::new(systems, kept, summary_bytes, summary_line.is_some());
    budget.keep_monotone(original_bytes);
    if let Some(limit) = template.total_budget_bytes {
        budget.enforce(
            limit.min(original_bytes),
            &squeeze_order,
            template.direction,
        );
    }

    let kept_messages = budget.surviving_messages();
    let squeezed_messages = budget.squeezed;
    let summary_omitted = budget.omission;
    let summary = if budget.summary_included {
        summary_line
    } else {
        None
    };
    let compacted = budget.assemble(summary.as_deref(), template.direction);

    let compacted_bytes = compacted.iter().map(TranscriptMessage::byte_len).sum();
    Ok(TemplateCompaction {
        messages: compacted,
        summary,
        summary_omitted,
        direction: template.direction,
        compressed_side: template.direction.compresses(),
        split_at,
        requested_split_at: template.split_at,
        compressed_messages: compressed.len(),
        stripped_system_messages: system_messages.len(),
        kept_messages,
        squeezed_messages,
        truncated_tool_results,
        original_bytes,
        compacted_bytes,
    })
}

/// Cuts kept-side tool results to [`CompactTemplate::max_tool_result_bytes`].
///
/// Returns the number of tool results that were shortened.
fn truncate_tool_results(kept: &mut [TranscriptMessage], template: &CompactTemplate) -> usize {
    let mut truncated = 0;
    for message in kept {
        if message.role != CompactRole::Tool {
            continue;
        }
        if let Some(content) = shortened_tool_content(
            &message.content,
            template.max_tool_result_bytes,
            &template.truncated_suffix,
        ) {
            message.content = content;
            truncated += 1;
        }
    }
    truncated
}

/// Byte-budget accounting for one compaction pass.
///
/// The state is created with every message included, then reduced by
/// [`BudgetState::keep_monotone`] and [`BudgetState::enforce`] until
/// [`BudgetState::assemble`] can render the result.
struct BudgetState {
    /// Kept system messages; empty unless the template keeps them.
    systems: Vec<TranscriptMessage>,
    /// How many of `systems` still survive.
    system_kept: usize,
    /// The verbatim side, already cut to the template's tool budget.
    kept: Vec<TranscriptMessage>,
    /// Which `kept` entries survive.
    included: Vec<bool>,
    /// Whether the summary message counts towards the total.
    summary_included: bool,
    /// Byte weight of the summary message.
    summary_bytes: usize,
    /// Current byte weight of the would-be output.
    total: usize,
    /// Messages removed by the budget, excluding the summary.
    squeezed: usize,
    /// Why the summary is missing, when it is.
    omission: Option<SummaryOmission>,
}

impl BudgetState {
    /// Creates the fully-included state and weighs it.
    fn new(
        systems: Vec<TranscriptMessage>,
        kept: Vec<TranscriptMessage>,
        summary_bytes: usize,
        summary_included: bool,
    ) -> Self {
        let included = vec![true; kept.len()];
        let system_kept = systems.len();
        let total = systems
            .iter()
            .chain(kept.iter())
            .map(TranscriptMessage::byte_len)
            .sum::<usize>()
            + if summary_included { summary_bytes } else { 0 };
        Self {
            systems,
            system_kept,
            kept,
            included,
            summary_included,
            summary_bytes,
            total,
            squeezed: 0,
            omission: None,
        }
    }

    /// Drops the summary when emitting it would grow the transcript past `original_bytes`.
    fn keep_monotone(&mut self, original_bytes: usize) {
        if self.summary_included && self.total > original_bytes {
            self.summary_included = false;
            self.total -= self.summary_bytes;
            self.omission = Some(SummaryOmission::Regressive);
        }
    }

    /// Reduces `squeeze_order` (lowest priority first) until the total fits `limit`.
    ///
    /// Kept system messages are the last resort, so a limit that can be met always is.
    fn enforce(&mut self, limit: usize, squeeze_order: &[BudgetTier], direction: CompactDirection) {
        // Within one tier the messages farthest from the split go first, so the side nearest the
        // split survives longest.
        let mut farthest_first: Vec<usize> = (0..self.kept.len()).collect();
        if direction == CompactDirection::Before {
            farthest_first.reverse();
        }
        for tier in squeeze_order {
            if self.total <= limit {
                break;
            }
            match tier {
                BudgetTier::Summary => {
                    if self.summary_included {
                        self.summary_included = false;
                        self.total -= self.summary_bytes;
                        self.omission = Some(SummaryOmission::Budget);
                    }
                }
                BudgetTier::ToolResults => self.drop_tier(&farthest_first, limit, true),
                BudgetTier::KeptSide => self.drop_tier(&farthest_first, limit, false),
            }
        }
        while self.total > limit && self.system_kept > 0 {
            self.system_kept -= 1;
            self.total -= self.systems[self.system_kept].byte_len();
            self.squeezed += 1;
        }
    }

    /// Drops one tier's messages farthest-from-split first until the total fits `limit`.
    ///
    /// `tools` selects the tier: `true` reduces tool results, `false` reduces non-tool messages.
    fn drop_tier(&mut self, farthest_first: &[usize], limit: usize, tools: bool) {
        for &index in farthest_first {
            if self.total <= limit {
                break;
            }
            if self.included[index] && (self.kept[index].role == CompactRole::Tool) == tools {
                self.included[index] = false;
                self.total -= self.kept[index].byte_len();
                self.squeezed += 1;
            }
        }
    }

    /// Returns the number of kept-side messages that survive verbatim.
    fn surviving_messages(&self) -> usize {
        self.included.iter().filter(|keep| **keep).count()
    }

    /// Renders the surviving messages with `summary` on its side of the verbatim block.
    fn assemble(
        self,
        summary: Option<&str>,
        direction: CompactDirection,
    ) -> Vec<TranscriptMessage> {
        let mut compacted: Vec<TranscriptMessage> =
            Vec::with_capacity(self.system_kept + self.kept.len() + usize::from(summary.is_some()));
        compacted.extend(self.systems.into_iter().take(self.system_kept));
        let surviving = self
            .kept
            .into_iter()
            .zip(self.included.iter())
            .filter_map(|(message, keep)| (*keep).then_some(message));
        match direction {
            CompactDirection::Before => {
                if let Some(line) = summary {
                    compacted.push(TranscriptMessage::system(line));
                }
                compacted.extend(surviving);
            }
            CompactDirection::After => {
                compacted.extend(surviving);
                if let Some(line) = summary {
                    compacted.push(TranscriptMessage::system(line));
                }
            }
        }
        compacted
    }
}

// ---------------------------------------------------------------------------------------------
// The deterministic summary
// ---------------------------------------------------------------------------------------------

/// The facts extracted from a compressed side, in first-seen order and without duplicates.
#[derive(Debug, Default)]
struct SummaryFacts {
    paths: Vec<String>,
    decisions: Vec<String>,
    errors: Vec<String>,
    feedback: Vec<String>,
}

impl SummaryFacts {
    /// Returns the values of one category.
    fn values(&self, fact: SummaryFact) -> &[String] {
        match fact {
            SummaryFact::Paths => &self.paths,
            SummaryFact::Decisions => &self.decisions,
            SummaryFact::Errors => &self.errors,
            SummaryFact::Feedback => &self.feedback,
        }
    }
}

/// Builds the deterministic summary line for one compressed side.
///
/// Shape: `<SUMMARY_PREFIX><side>@<split>/<total>: <n> message(s) compressed (<role counts>); <fact
/// segments>`. The fact segments appear in the fixed order `paths`, `decisions`, `errors`,
/// `feedback`; an empty or disabled category contributes nothing, so no byte is spent on it.
fn template_summary_line(
    template: &CompactTemplate,
    split_at: usize,
    total_messages: usize,
    compressed: &[&TranscriptMessage],
    facts: &SummaryFacts,
) -> String {
    let count = compressed.len();
    let noun = if count == 1 { "message" } else { "messages" };
    let mut line = format!(
        "{SUMMARY_PREFIX}{}@{split_at}/{total_messages}: {count} {noun} compressed ({})",
        template.direction.compresses().as_str(),
        role_count_parts(compressed)
    );

    let retention = &template.retention;
    let mut segments: Vec<String> = Vec::with_capacity(4);
    for fact in [
        SummaryFact::Paths,
        SummaryFact::Decisions,
        SummaryFact::Errors,
        SummaryFact::Feedback,
    ] {
        if !retention.retains(fact) {
            continue;
        }
        let values = facts.values(fact);
        let excerpt = fact != SummaryFact::Paths;
        if let Some(segment) = fact_segment(
            fact.as_str(),
            values,
            retention.max_facts,
            retention.max_excerpt_chars,
            excerpt,
        ) {
            segments.push(segment);
        }
    }
    if !segments.is_empty() {
        line.push_str("; ");
        line.push_str(&segments.join("; "));
    }
    line
}

/// Renders one fact category: `name: a, b (+2)`, or `name: <count>` when `max_facts` is `0`.
fn fact_segment(
    name: &str,
    values: &[String],
    max_facts: usize,
    max_excerpt_chars: usize,
    excerpt: bool,
) -> Option<String> {
    if values.is_empty() {
        return None;
    }
    if max_facts == 0 {
        return Some(format!("{name}: {}", values.len()));
    }
    let shown: Vec<String> = values
        .iter()
        .take(max_facts)
        .map(|value| {
            if excerpt {
                excerpt_text(value, max_excerpt_chars)
            } else {
                value.clone()
            }
        })
        .collect();
    let hidden = values.len() - shown.len();
    let suffix = if hidden > 0 {
        format!(" (+{hidden})")
    } else {
        String::new()
    };
    Some(format!("{name}: {}{suffix}", shown.join(", ")))
}

/// Cuts `text` to `max_chars` characters on a `char` boundary and marks the cut with `…`.
fn excerpt_text(text: &str, max_chars: usize) -> String {
    match text.char_indices().nth(max_chars) {
        Some((end, _)) => format!("{}…", &text[..end]),
        None => text.to_owned(),
    }
}

/// Extracts the retained facts from the compressed side, deterministically.
fn collect_facts(compressed: &[&TranscriptMessage], retention: &SummaryRetention) -> SummaryFacts {
    let mut facts = SummaryFacts::default();
    if retention.facts.is_empty() {
        return facts;
    }
    let scan_lines = retention.retains(SummaryFact::Errors)
        || retention.retains(SummaryFact::Decisions)
        || retention.retains(SummaryFact::Feedback);

    for message in compressed {
        if retention.retains(SummaryFact::Paths) {
            for path in path_tokens(&message.content) {
                push_unique(&mut facts.paths, path);
            }
        }
        if !scan_lines {
            continue;
        }
        for line in message.content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if retention.retains(SummaryFact::Errors) && contains_marker(line, &ERROR_MARKERS) {
                push_unique(&mut facts.errors, line.to_owned());
            }
            if retention.retains(SummaryFact::Decisions) && contains_marker(line, &DECISION_MARKERS)
            {
                push_unique(&mut facts.decisions, line.to_owned());
            }
            if retention.retains(SummaryFact::Feedback) && message.role == CompactRole::User {
                push_unique(&mut facts.feedback, line.to_owned());
            }
        }
    }
    facts
}

/// Appends `value` unless an identical value is already present, preserving first-seen order.
fn push_unique(values: &mut Vec<String>, value: String) {
    if !values.iter().any(|existing| existing == &value) {
        values.push(value);
    }
}

/// Returns `true` when `line` contains any of `markers`, case-insensitively.
fn contains_marker(line: &str, markers: &[&str]) -> bool {
    let lowered = line.to_lowercase();
    markers.iter().any(|marker| lowered.contains(marker))
}

/// Extracts path-like tokens from free text, in first-seen order and without duplicates.
///
/// A token qualifies when it contains a `/`, is built only from `[A-Za-z0-9_./-~]`, and either
/// starts with `/` (an absolute path) or has a file extension on its last segment. That keeps
/// `src/parser.rs` and `/work/main.rs` while rejecting prose such as `and/or`.
fn path_tokens(content: &str) -> Vec<String> {
    let mut tokens: Vec<String> = Vec::new();
    for raw in content.split(|c: char| c.is_whitespace() || PATH_DELIMITERS.contains(&c)) {
        let token = raw.trim_end_matches(['.', ',', '-']);
        if is_path_token(token) && !tokens.iter().any(|seen| seen == token) {
            tokens.push(token.to_owned());
        }
    }
    tokens
}

/// Returns `true` when `token` looks like a file path rather than a piece of prose.
fn is_path_token(token: &str) -> bool {
    if token.len() < 3 || !token.contains('/') || token.starts_with("//") || token.ends_with('/') {
        return false;
    }
    if !token.chars().any(|c| c.is_ascii_alphanumeric()) {
        return false;
    }
    if !token
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '/' | '-' | '~'))
    {
        return false;
    }
    let looks_like_file = token
        .rsplit('/')
        .next()
        .is_some_and(|segment| segment.contains('.'));
    token.starts_with('/') || looks_like_file
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
#[path = "compact_template_tests.rs"]
mod compact_template_tests;
