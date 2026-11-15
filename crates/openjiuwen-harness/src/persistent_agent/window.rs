//! Token-budgeted window assembly and rollover planning.
//!
//! A persistent execution is replayed from its journal ([`ExecutionEntry`] values); this module
//! turns such a replay into the model-facing window: an ordered list of [`BaseMessage`]s that fits
//! a token budget. Assembly is pure and synchronous — no clock, no randomness, no ambient IO — so
//! the same entries and the same summariser always produce byte-identical messages. Any model call
//! goes through the injected [`WindowSummarizer`] seam, never through the assembler itself.
//!
//! The rendering policy mirrors what the journal can prove:
//!
//! | entry kind | rendered as | elided when |
//! | --- | --- | --- |
//! | `execution_started` | one user message `【本次执行】{goal}` | it no longer fits |
//! | `model_turn` | one assistant message (`(no content)` when empty) | it no longer fits |
//! | `tool_intent` run | one assistant message carrying every [`ToolCall`] | together with its results |
//! | `tool_completed` run | one tool message per call (`tool_call_id`, `result_summary`) | together with its intents |
//! | `reply_delta` | never rendered | always ([`ElisionReason::CountedOnly`]) |
//! | lifecycle kinds | system message `[{kind}] …` only when
//!   [`WindowPolicy::include_lifecycle_markers`] is set | otherwise ([`ElisionReason::LifecycleMarker`]) |
//!
//! Raw tool arguments are deliberately not journalled, so a reconstructed [`ToolCall`] carries the
//! persisted `args_summary` as its `arguments`. A run of tool intents and the results that
//! immediately follow them form one indivisible group: a group is either kept whole or elided
//! whole, so a window never contains a tool result without its call (or the reverse).
//!
//! Elided entries are replaced by one summary message placed *before* the retained messages. The
//! default [`ExtractiveSummarizer`] produces that summary without a model; a host that wants a
//! model-written summary injects a [`CompletionSummarizer`] — a bounded prompt plus a
//! caller-supplied synchronous completion function — through [`WindowAssembler::with_summarizer`].
//!
//! Assembly never degrades silently. When the injected summariser fails, [`WindowAssembler`] still
//! renders one deterministic count-only fallback so a caller that cannot act on an error keeps a
//! usable window, but that text is marked `摘要不可用` and the failure is recorded in
//! [`WindowAssembly::summary_error`]; [`WindowAssembler::try_assemble`] and
//! [`WindowAssembler::try_assemble_with_budget`] return the error to the caller instead.
// Size note: ~1010 lines of implementation plus an inline test module; the rendering
// table, budget accounting and rollover planner are one cohesive unit.
use std::collections::HashSet;
use std::fmt;
use std::sync::Arc;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::prompt::{ApproxTokenCounter, BaseMessage, TokenCounter, ToolCall};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use super::entry::{EntryKind, EntryPayload, ExecutionEntry};
use super::util::truncate_chars;

/// Content of a model turn that produced no text.
const NO_CONTENT: &str = "(no content)";

/// Prefix of every extractive summary.
const SUMMARY_PREFIX: &str = "【窗口摘要】";

/// Marker carried by the fallback summary, so a degraded summary is never mistaken for a real one.
const SUMMARY_UNAVAILABLE: &str = "摘要不可用";

/// Maximum number of characters of one elided entry quoted in the summary hint.
const SUMMARY_HINT_CHARS: usize = 120;

/// Prefix of every [`CompletionSummarizer`] prompt.
const COMPLETION_PROMPT_PREFIX: &str = "【摘要请求】";

/// Instruction appended to every [`CompletionSummarizer`] prompt.
const COMPLETION_INSTRUCTION: &str =
    "请用一段话概括上面被省略的执行记录，保留任务目标、关键动作和仍未完成的事项；\
                                      不要引用记录中没有的信息，只输出摘要正文。";

/// Maximum number of excerpts quoted in one [`CompletionSummarizer`] prompt.
const COMPLETION_MAX_HINTS: usize = 3;

/// Default bound on the answer [`CompletionSummarizer`] accepts, in Unicode scalar values.
const DEFAULT_COMPLETION_MAX_CHARS: usize = 4_096;

/// Every [`EntryKind`], in declaration order.
///
/// Summaries list their per-kind counts in this order, so the rendering does not depend on the
/// order the entries happen to be elided in.
const ENTRY_KINDS: [EntryKind; 9] = [
    EntryKind::ExecutionStarted,
    EntryKind::ModelTurn,
    EntryKind::ToolIntent,
    EntryKind::ToolCompleted,
    EntryKind::ReplyDelta,
    EntryKind::ExecutionPaused,
    EntryKind::ExecutionFinal,
    EntryKind::WindowSwitched,
    EntryKind::Note,
];

/// Token budget of one model window.
///
/// `context_window` is the model's full context; `output_reserve` and `safety_margin` are kept
/// free for the reply and for estimation error, and `input_limit` is the provider's (or the host's)
/// hard cap on the prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct Capacity {
    /// Tokens the model can hold in total, input and output together.
    pub context_window: usize,
    /// Tokens kept free for the model's reply.
    pub output_reserve: usize,
    /// Hard upper bound on the input tokens, independent of the context window.
    pub input_limit: usize,
    /// Tokens kept free as a cushion against estimation error.
    pub safety_margin: usize,
}

impl Default for Capacity {
    /// A `128_000`-token window with a `4_096`-token reply reserve and a `1_024`-token margin.
    fn default() -> Self {
        Self {
            context_window: 128_000,
            output_reserve: 4_096,
            input_limit: 128_000,
            safety_margin: 1_024,
        }
    }
}

impl Capacity {
    /// Validates the capacity.
    ///
    /// # Errors
    /// `Error::Validation` naming the field when any value is zero, when `input_limit` exceeds
    /// `context_window`, or when `output_reserve` plus `safety_margin` does not leave room for
    /// input inside `context_window`.
    pub fn validate(&self) -> Result<()> {
        for (field, value) in [
            ("context_window", self.context_window),
            ("output_reserve", self.output_reserve),
            ("input_limit", self.input_limit),
            ("safety_margin", self.safety_margin),
        ] {
            if value == 0 {
                return Err(Error::validation(format!(
                    "capacity {field} must be a positive integer"
                )));
            }
        }
        if self.input_limit > self.context_window {
            return Err(Error::validation(
                "capacity input_limit must not exceed context_window",
            ));
        }
        if self.output_reserve.saturating_add(self.safety_margin) >= self.context_window {
            return Err(Error::validation(
                "capacity output_reserve plus safety_margin must be smaller than context_window",
            ));
        }
        Ok(())
    }

    /// Returns the number of input tokens one window may use.
    ///
    /// This is the smaller of [`Capacity::input_limit`] and the context window minus the reply
    /// reserve and the safety margin. The arithmetic saturates, so an unvalidated capacity cannot
    /// wrap around into a huge budget.
    #[must_use]
    pub fn available_input(&self) -> usize {
        let reserved = self
            .context_window
            .saturating_sub(self.output_reserve.saturating_add(self.safety_margin));
        self.input_limit.min(reserved)
    }

    /// Returns the token count above which the window is rolled over: 80% of the context window.
    #[must_use]
    pub fn rollover_threshold(&self) -> usize {
        self.context_window.saturating_mul(4) / 5
    }

    /// Returns the input budget of a freshly rolled-over window: 40% of the context window.
    #[must_use]
    pub fn new_window_limit(&self) -> usize {
        self.context_window.saturating_mul(2) / 5
    }
}

/// Shape of the assembled window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct WindowPolicy {
    /// Maximum number of message groups kept after the retention pass.
    pub max_recent_groups: usize,
    /// Upper bound on the summary length, in Unicode scalar values.
    pub summary_max_chars: usize,
    /// Number of groups the budget pass never drops below.
    pub min_recent_groups: usize,
    /// Whether lifecycle entries are rendered as system messages instead of being elided.
    pub include_lifecycle_markers: bool,
}

impl Default for WindowPolicy {
    /// Eight recent groups, a `4096`-character summary, at least one group, no lifecycle markers.
    fn default() -> Self {
        Self {
            max_recent_groups: 8,
            summary_max_chars: 4_096,
            min_recent_groups: 1,
            include_lifecycle_markers: false,
        }
    }
}

impl WindowPolicy {
    /// Validates the policy.
    ///
    /// # Errors
    /// `Error::Validation` when `max_recent_groups` or `min_recent_groups` is zero, when
    /// `min_recent_groups` exceeds `max_recent_groups`, or when `summary_max_chars` is below `16`.
    pub fn validate(&self) -> Result<()> {
        if self.max_recent_groups == 0 {
            return Err(Error::validation(
                "window policy max_recent_groups must be a positive integer",
            ));
        }
        if self.min_recent_groups == 0 {
            return Err(Error::validation(
                "window policy min_recent_groups must be a positive integer",
            ));
        }
        if self.min_recent_groups > self.max_recent_groups {
            return Err(Error::validation(
                "window policy min_recent_groups must not exceed max_recent_groups",
            ));
        }
        if self.summary_max_chars < 16 {
            return Err(Error::validation(
                "window policy summary_max_chars must be at least 16",
            ));
        }
        Ok(())
    }
}

/// Builds the replacement text for the entries a window had to drop.
pub trait WindowSummarizer: Send + Sync {
    /// Summarises the elided `entries`, in journal order.
    ///
    /// # Errors
    /// `Error::Unsupported` when the summariser needs a model that was never injected, or any
    /// error the host implementation reports.
    fn summarize(&self, elided: &[ExecutionEntry]) -> Result<String>;
}

/// The default summariser: a deterministic Chinese sentence built from the elided entries.
///
/// The sentence names the number of dropped entries, their per-kind counts in [`EntryKind`]
/// declaration order, the oldest and newest dropped sequence number, and one hint taken from the
/// first dropped model turn or tool call. Nothing is quoted beyond
/// `SUMMARY_HINT_CHARS`, so the text stays bounded and reproducible.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ExtractiveSummarizer {
    /// Upper bound on the summary length, in Unicode scalar values.
    pub max_chars: usize,
}

impl ExtractiveSummarizer {
    /// Creates an extractive summariser bounded to `max_chars` Unicode scalar values.
    #[must_use]
    pub fn new(max_chars: usize) -> Self {
        Self { max_chars }
    }
}

impl Default for ExtractiveSummarizer {
    /// A summary bounded to `4096` characters.
    fn default() -> Self {
        Self { max_chars: 4_096 }
    }
}

impl WindowSummarizer for ExtractiveSummarizer {
    fn summarize(&self, elided: &[ExecutionEntry]) -> Result<String> {
        if elided.is_empty() {
            return Ok(empty_summary());
        }
        let (first, last) = seq_bounds(elided);
        let mut text = format!(
            "{SUMMARY_PREFIX}省略 {} 条记录（{}）；最早 seq {first}，最晚 seq {last}。",
            elided.len(),
            kind_counts(elided)
        );
        if let Some(hint) = first_hint(elided) {
            text.push_str(&hint);
            text.push('。');
        }
        Ok(truncate_chars(&text, self.max_chars))
    }
}

/// Returns the summary of an empty elision set: nothing was dropped, so there is nothing to
/// summarise and no completion is worth a call.
fn empty_summary() -> String {
    format!("{SUMMARY_PREFIX}省略 0 条记录。")
}

/// Renders the per-kind counts of `elided` in [`EntryKind`] declaration order.
fn kind_counts(elided: &[ExecutionEntry]) -> String {
    ENTRY_KINDS
        .iter()
        .map(|kind| {
            let count = elided.iter().filter(|entry| entry.kind() == *kind).count();
            (*kind, count)
        })
        .filter(|(_, count)| *count > 0)
        .map(|(kind, count)| format!("{}×{count}", kind.as_str()))
        .collect::<Vec<String>>()
        .join(", ")
}

/// Returns the oldest and newest journal sequence number of `elided`.
fn seq_bounds(elided: &[ExecutionEntry]) -> (u64, u64) {
    let first = elided
        .iter()
        .map(|entry| entry.seq)
        .min()
        .unwrap_or_default();
    let last = elided
        .iter()
        .map(|entry| entry.seq)
        .max()
        .unwrap_or_default();
    (first, last)
}

/// One quotable detail of an elided entry.
enum Hint<'a> {
    /// Text of an elided model turn.
    ModelTurn(&'a str),
    /// Name of a tool called by an elided entry.
    Tool(&'a str),
}

impl Hint<'_> {
    /// Renders the hint as `{prefix}模型回复：{bounded text}` or `{prefix}工具：{name}`.
    fn render(&self, prefix: &str, excerpt_chars: usize) -> String {
        match self {
            Self::ModelTurn(response) => {
                let text = if response.is_empty() {
                    NO_CONTENT
                } else {
                    response
                };
                format!(
                    "{prefix}模型回复：{}",
                    truncate_chars(text, excerpt_chars.max(1))
                )
            }
            Self::Tool(name) => format!("{prefix}工具：{name}"),
        }
    }
}

/// Returns up to `limit` hints of `elided`, in journal order.
fn hints(elided: &[ExecutionEntry], limit: usize) -> Vec<Hint<'_>> {
    let mut found = Vec::new();
    for entry in elided {
        match &entry.payload {
            EntryPayload::ModelTurn { response, .. } => found.push(Hint::ModelTurn(response)),
            EntryPayload::ToolIntent { tool_name, .. }
            | EntryPayload::ToolCompleted { tool_name, .. } => found.push(Hint::Tool(tool_name)),
            _ => continue,
        }
        if found.len() >= limit {
            break;
        }
    }
    found
}

/// Returns the first hint of the elided entries: the first model turn's text, or the first tool's
/// name, whichever comes first in journal order.
fn first_hint(elided: &[ExecutionEntry]) -> Option<String> {
    hints(elided, 1)
        .first()
        .map(|hint| hint.render("首个", SUMMARY_HINT_CHARS))
}

/// The completion seam a [`CompletionSummarizer`] calls: one bounded prompt in, one summary out.
///
/// Every `Fn(&str) -> Result<String> + Send + Sync + 'static` injected through
/// [`CompletionSummarizer::new`] or [`CompletionSummarizer::with_max_chars`] becomes an
/// `Arc<CompletionFn>`, so a host can also hold and share its seam directly.
pub type CompletionFn = dyn Fn(&str) -> Result<String> + Send + Sync;

/// A summariser backed by a caller-supplied synchronous completion function.
///
/// [`WindowAssembler`] is pure and synchronous, so it cannot await a model call. This summariser
/// bridges that: the host injects the completion seam as
/// `Fn(&str) -> Result<String> + Send + Sync` — a model client behind a blocking bridge, a cache,
/// a pre-computed answer or a KV-backed queue — and the summariser turns the elided entries into
/// one bounded prompt and returns the answer.
///
/// The prompt is bounded by construction (the per-kind counts span at most the nine
/// [`EntryKind`]s, the sequence numbers are integers, and at most three
/// excerpts of at most `SUMMARY_HINT_CHARS` characters each are quoted), so a prompt stays within
/// roughly one kilobyte whatever the journal contains. [`CompletionSummarizer::prompt`] returns
/// exactly what a call sends, and [`CompletionSummarizer::last_prompt`] returns what the last call
/// actually sent.
///
/// The answer is trimmed, bounded to [`CompletionSummarizer::max_chars`], and returned; an empty or
/// whitespace-only answer is a typed [`Error::Validation`] naming the completion seam instead of a
/// silently blank summary, and an error from the seam is returned as-is. A summariser that fails is
/// never retried inside one assembly, and the host pays one completion call per budget-pass
/// iteration, so an expensive seam should memoise inside its function.
pub struct CompletionSummarizer {
    /// The injected completion seam.
    complete: Arc<CompletionFn>,
    /// Upper bound on the returned summary, in Unicode scalar values.
    max_chars: usize,
    /// Prompt sent by the most recent [`WindowSummarizer::summarize`] call.
    last_prompt: Mutex<Option<String>>,
}

impl CompletionSummarizer {
    /// Creates a summariser over `complete`, bounding the answer to `4096` Unicode scalar values
    /// (the [`WindowPolicy::summary_max_chars`] default).
    #[must_use]
    pub fn new<F>(complete: F) -> Self
    where
        F: Fn(&str) -> Result<String> + Send + Sync + 'static,
    {
        Self::with_max_chars(complete, DEFAULT_COMPLETION_MAX_CHARS)
    }

    /// Creates a summariser over `complete`, bounding the answer to `max_chars` Unicode scalar
    /// values.
    #[must_use]
    pub fn with_max_chars<F>(complete: F, max_chars: usize) -> Self
    where
        F: Fn(&str) -> Result<String> + Send + Sync + 'static,
    {
        Self {
            complete: Arc::new(complete),
            max_chars,
            last_prompt: Mutex::new(None),
        }
    }

    /// Returns the bound on the returned summary, in Unicode scalar values.
    #[must_use]
    pub fn max_chars(&self) -> usize {
        self.max_chars
    }

    /// Returns the prompt [`WindowSummarizer::summarize`] sends for `elided`.
    ///
    /// The prompt is deterministic and bounded, so a host (or a test) can assert its content
    /// without calling the completion seam.
    #[must_use]
    pub fn prompt(&self, elided: &[ExecutionEntry]) -> String {
        let excerpt_chars = SUMMARY_HINT_CHARS.min(self.max_chars.max(1));
        let mut text = if elided.is_empty() {
            format!("{COMPLETION_PROMPT_PREFIX}省略 0 条记录。")
        } else {
            let (first, last) = seq_bounds(elided);
            format!(
                "{COMPLETION_PROMPT_PREFIX}省略 {} 条记录（{}）；最早 seq {first}，最晚 seq {last}。",
                elided.len(),
                kind_counts(elided)
            )
        };
        let excerpts = hints(elided, COMPLETION_MAX_HINTS)
            .iter()
            .map(|hint| hint.render("", excerpt_chars))
            .collect::<Vec<String>>();
        if !excerpts.is_empty() {
            text.push_str("记录片段：");
            text.push_str(&excerpts.join("；"));
            text.push('。');
        }
        text.push_str(COMPLETION_INSTRUCTION);
        text
    }

    /// Returns the prompt of the most recent call, or `None` before the first one.
    #[must_use]
    pub fn last_prompt(&self) -> Option<String> {
        self.last_prompt.lock().clone()
    }
}

impl fmt::Debug for CompletionSummarizer {
    /// Renders the configured bounds and the size of the last prompt; neither the prompt body nor
    /// the seam is printed, so the rendering stays bounded and credential-free.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let last_prompt_chars = self
            .last_prompt
            .lock()
            .as_ref()
            .map(|prompt| prompt.chars().count());
        f.debug_struct("CompletionSummarizer")
            .field("max_chars", &self.max_chars)
            .field("last_prompt_chars", &last_prompt_chars)
            .finish_non_exhaustive()
    }
}

impl WindowSummarizer for CompletionSummarizer {
    fn summarize(&self, elided: &[ExecutionEntry]) -> Result<String> {
        if elided.is_empty() {
            return Ok(empty_summary());
        }
        let prompt = self.prompt(elided);
        *self.last_prompt.lock() = Some(prompt.clone());
        let answer = (self.complete)(&prompt)?;
        let trimmed = answer.trim();
        if trimmed.is_empty() {
            return Err(Error::validation(
                "persistent window summarisation completion seam returned an empty answer; the \
                 Fn(&str) -> Result<String> injected through WindowAssembler::with_summarizer \
                 must return the summary text",
            ));
        }
        Ok(truncate_chars(trimmed, self.max_chars))
    }
}

/// A summariser that has no model to call, so it refuses to summarise.
///
/// It exists to prove the seam is typed, not to be used: inject a real summariser with
/// [`WindowAssembler::with_summarizer`] — a [`CompletionSummarizer`] over a model client (or a
/// cache, or a KV-backed queue) for a model-written summary, or an [`ExtractiveSummarizer`] for
/// the deterministic offline default. Neither [`WindowAssembler::assemble`] nor
/// [`WindowAssembler::try_assemble`] pretends the refusal is a summary: the infallible path records
/// it in [`WindowAssembly::summary_error`] and marks the fallback text, the fallible path returns
/// the error.
#[derive(Debug, Clone, Copy, Default)]
pub struct UnsupportedModelSummarizer;

impl WindowSummarizer for UnsupportedModelSummarizer {
    fn summarize(&self, _elided: &[ExecutionEntry]) -> Result<String> {
        Err(Error::unsupported(
            "persistent window summarisation has no completion seam to call; inject one with \
             WindowAssembler::with_summarizer — CompletionSummarizer for a model-backed summary, \
             or ExtractiveSummarizer for the deterministic offline default",
        ))
    }
}

/// Why an entry is not part of the rendered window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ElisionReason {
    /// The entry did not fit the token budget.
    OverBudget,
    /// The entry is accounted for in the summary but was never rendered (reply fragments).
    CountedOnly,
    /// The entry is a lifecycle marker that this policy does not render.
    LifecycleMarker,
    /// The entry fell outside [`WindowPolicy::max_recent_groups`].
    RetentionLimit,
}

/// One entry that the assembled window does not contain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ElidedEntry {
    /// Sequence number of the entry in the execution journal.
    pub seq: u64,
    /// Kind of the elided entry.
    pub kind: EntryKind,
    /// Why the entry was elided.
    pub reason: ElisionReason,
}

impl ElidedEntry {
    /// Creates an elision record.
    const fn new(seq: u64, kind: EntryKind, reason: ElisionReason) -> Self {
        Self { seq, kind, reason }
    }
}

/// The model-facing window rebuilt from an execution journal.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct WindowAssembly {
    /// Messages in model order: the summary message (if any) first, then the retained groups.
    pub messages: Vec<BaseMessage>,
    /// Token count of [`WindowAssembly::messages`], as reported by the assembler's counter.
    pub tokens: usize,
    /// Token budget the assembly was fitted to.
    pub budget: usize,
    /// Summary of the elided entries, present whenever anything was elided.
    ///
    /// When [`WindowAssembly::summary_error`] is set this is the documented count-only fallback,
    /// not the summariser's text.
    pub summary: Option<String>,
    /// Why the summariser failed, when it did.
    ///
    /// `None` means the summary (if any) came from the summariser itself. When the summariser
    /// returned an error, [`WindowAssembler::assemble`] still produces a window — it is infallible
    /// — but [`WindowAssembly::summary`] carries the deterministic count-only fallback marked
    /// `摘要不可用` and this field carries the failure, so a host can never mistake a degraded
    /// window for a summarised one. [`WindowAssembler::try_assemble`] and
    /// [`WindowAssembler::try_assemble_with_budget`] return that error instead of the assembly.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary_error: Option<String>,
    /// Every elided entry with its reason, ordered by sequence number.
    pub elided: Vec<ElidedEntry>,
    /// Sequence numbers of the entries rendered into [`WindowAssembly::messages`], in order.
    pub retained_seqs: Vec<u64>,
    /// Whether even the minimum retained set plus summary exceeds the budget.
    pub over_budget: bool,
}

impl WindowAssembly {
    /// Returns the number of elided entries.
    #[must_use]
    pub fn elided_count(&self) -> usize {
        self.elided.len()
    }
}

/// One renderable group of journal entries and the messages it becomes.
#[derive(Debug, Clone, Default)]
struct Group {
    /// Sequence numbers and kinds of the entries the group consumes, in journal order.
    entries: Vec<(u64, EntryKind)>,
    /// Messages rendered for the group.
    messages: Vec<BaseMessage>,
}

impl Group {
    /// Creates a group holding one entry and the message it renders to.
    fn single(entry: &ExecutionEntry, message: BaseMessage) -> Self {
        Self {
            entries: vec![(entry.seq, entry.kind())],
            messages: vec![message],
        }
    }

    /// Records that the group consumes `entry`.
    fn push(&mut self, entry: &ExecutionEntry) {
        self.entries.push((entry.seq, entry.kind()));
    }

    /// Returns one elision record per consumed entry.
    fn elisions(&self, reason: ElisionReason) -> Vec<ElidedEntry> {
        self.entries
            .iter()
            .map(|(seq, kind)| ElidedEntry::new(*seq, *kind, reason))
            .collect()
    }

    /// Iterates the sequence numbers of the consumed entries.
    fn seqs(&self) -> impl Iterator<Item = u64> + '_ {
        self.entries.iter().map(|(seq, _)| *seq)
    }
}

/// Renders journal `entries` into groups, reporting the entries that are never rendered.
fn build_groups(
    entries: &[ExecutionEntry],
    include_lifecycle_markers: bool,
) -> (Vec<Group>, Vec<ElidedEntry>) {
    let mut groups = Vec::new();
    let mut elided = Vec::new();
    let mut index = 0;
    while let Some(entry) = entries.get(index) {
        match &entry.payload {
            EntryPayload::ExecutionStarted { goal } => {
                groups.push(Group::single(
                    entry,
                    BaseMessage::user(format!("【本次执行】{goal}")),
                ));
                index += 1;
            }
            EntryPayload::ModelTurn { response, .. } => {
                groups.push(Group::single(entry, model_turn_message(response)));
                index += 1;
            }
            EntryPayload::ToolIntent { .. } => groups.push(tool_group(entries, &mut index)),
            EntryPayload::ToolCompleted {
                tool_call_id,
                result_summary,
                ..
            } => {
                // No intent was journalled for this call, so it stands on its own.
                groups.push(Group::single(
                    entry,
                    BaseMessage::tool(tool_call_id.clone(), result_summary.clone()),
                ));
                index += 1;
            }
            EntryPayload::ReplyDelta { .. } => {
                elided.push(ElidedEntry::new(
                    entry.seq,
                    EntryKind::ReplyDelta,
                    ElisionReason::CountedOnly,
                ));
                index += 1;
            }
            EntryPayload::ExecutionPaused { .. }
            | EntryPayload::ExecutionFinal { .. }
            | EntryPayload::WindowSwitched { .. }
            | EntryPayload::Note { .. } => {
                if include_lifecycle_markers {
                    groups.push(Group::single(
                        entry,
                        BaseMessage::system(lifecycle_text(entry)),
                    ));
                } else {
                    elided.push(ElidedEntry::new(
                        entry.seq,
                        entry.kind(),
                        ElisionReason::LifecycleMarker,
                    ));
                }
                index += 1;
            }
        }
    }
    (groups, elided)
}

/// Renders one model turn: the recorded response, or [`NO_CONTENT`] when it produced no text.
fn model_turn_message(response: &str) -> BaseMessage {
    if response.is_empty() {
        BaseMessage::assistant(NO_CONTENT)
    } else {
        BaseMessage::assistant(response)
    }
}

/// Renders a run of tool intents and the results that follow it as one indivisible group.
///
/// The group starts with one assistant message carrying every call and continues with one tool
/// message per result. Raw arguments are not journalled, so each call carries the persisted
/// `args_summary` as its `arguments`.
fn tool_group(entries: &[ExecutionEntry], index: &mut usize) -> Group {
    let mut group = Group::default();
    let mut calls = Vec::new();
    while let Some(entry) = entries.get(*index) {
        let EntryPayload::ToolIntent {
            tool_call_id,
            tool_name,
            args_summary,
            ..
        } = &entry.payload
        else {
            break;
        };
        let mut call = ToolCall::new(tool_name.clone()).with_id(tool_call_id.clone());
        call.arguments.clone_from(args_summary);
        calls.push(call);
        group.push(entry);
        *index += 1;
    }
    group
        .messages
        .push(BaseMessage::assistant("").with_tool_calls(calls));
    while let Some(entry) = entries.get(*index) {
        let EntryPayload::ToolCompleted {
            tool_call_id,
            result_summary,
            ..
        } = &entry.payload
        else {
            break;
        };
        group.push(entry);
        group.messages.push(BaseMessage::tool(
            tool_call_id.clone(),
            result_summary.clone(),
        ));
        *index += 1;
    }
    group
}

/// Renders a lifecycle entry as `[{kind}] {detail}`.
fn lifecycle_text(entry: &ExecutionEntry) -> String {
    let kind = entry.kind().as_str();
    let detail = match &entry.payload {
        EntryPayload::ExecutionPaused { reason, detail } => match detail {
            Some(detail) => format!("{} {detail}", pause_reason_text(*reason)),
            None => pause_reason_text(*reason).to_string(),
        },
        EntryPayload::ExecutionFinal { status, reason } => match reason {
            Some(reason) => format!("{} {reason}", final_status_text(*status)),
            None => final_status_text(*status).to_string(),
        },
        EntryPayload::WindowSwitched {
            archive_ref,
            window_seq,
        } => format!("archive {archive_ref} (window_seq {window_seq})"),
        EntryPayload::Note { text } => text.clone(),
        _ => String::new(),
    };
    if detail.is_empty() {
        format!("[{kind}]")
    } else {
        format!("[{kind}] {detail}")
    }
}

/// Returns the persisted snake-case spelling of a pause reason.
fn pause_reason_text(reason: super::entry::PauseReason) -> &'static str {
    match reason {
        super::entry::PauseReason::AwaitingConfirmation => "awaiting_confirmation",
        super::entry::PauseReason::Stopped => "stopped",
        super::entry::PauseReason::BudgetDenied => "budget_denied",
        super::entry::PauseReason::ContextUnfit => "context_unfit",
        super::entry::PauseReason::Error => "error",
    }
}

/// Returns the persisted snake-case spelling of a terminal status.
fn final_status_text(status: super::entry::FinalStatus) -> &'static str {
    match status {
        super::entry::FinalStatus::Completed => "completed",
        super::entry::FinalStatus::Ignored => "ignored",
        super::entry::FinalStatus::Cancelled => "cancelled",
        super::entry::FinalStatus::Failed => "failed",
    }
}

/// Places the summary message (when there is one) before the retained messages.
fn compose(summary: Option<&str>, groups: &[Group]) -> Vec<BaseMessage> {
    let mut messages = Vec::new();
    if let Some(text) = summary {
        messages.push(BaseMessage::user(text));
    }
    for group in groups {
        messages.extend(group.messages.iter().cloned());
    }
    messages
}

/// Fits an execution journal into a token budget.
///
/// The assembler is pure: [`WindowAssembler::assemble`] and
/// [`WindowAssembler::assemble_with_budget`] take `&self` and never observe a clock, a random
/// source or the model. The only outbound call is the injected [`WindowSummarizer`].
#[derive(Clone)]
pub struct WindowAssembler {
    capacity: Capacity,
    counter: Arc<dyn TokenCounter + Send + Sync>,
    policy: WindowPolicy,
    summarizer: Arc<dyn WindowSummarizer>,
}

impl fmt::Debug for WindowAssembler {
    /// Renders the capacity and the policy; the counter and summariser are opaque handles.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WindowAssembler")
            .field("capacity", &self.capacity)
            .field("policy", &self.policy)
            .finish_non_exhaustive()
    }
}

impl WindowAssembler {
    /// Creates an assembler over `capacity` with the default policy, an [`ApproxTokenCounter`] and
    /// an [`ExtractiveSummarizer`].
    #[must_use]
    pub fn new(capacity: Capacity) -> Self {
        Self {
            capacity,
            counter: Arc::new(ApproxTokenCounter::new()),
            policy: WindowPolicy::default(),
            summarizer: Arc::new(ExtractiveSummarizer::default()),
        }
    }

    /// Replaces the token counter.
    #[must_use]
    pub fn with_counter(mut self, counter: Arc<dyn TokenCounter + Send + Sync>) -> Self {
        self.counter = counter;
        self
    }

    /// Replaces the window policy.
    #[must_use]
    pub fn with_policy(mut self, policy: WindowPolicy) -> Self {
        self.policy = policy;
        self
    }

    /// Replaces the summariser.
    #[must_use]
    pub fn with_summarizer(mut self, summarizer: Arc<dyn WindowSummarizer>) -> Self {
        self.summarizer = summarizer;
        self
    }

    /// Returns the capacity the assembler was built with.
    #[must_use]
    pub fn capacity(&self) -> Capacity {
        self.capacity
    }

    /// Returns the policy the assembler was built with.
    #[must_use]
    pub fn policy(&self) -> WindowPolicy {
        self.policy
    }

    /// Assembles the window for `entries`, using [`Capacity::available_input`] as the budget.
    ///
    /// `entries` are expected in journal order (increasing sequence numbers). This call is
    /// infallible; see [`WindowAssembler::try_assemble`] for the fallible form.
    #[must_use]
    pub fn assemble(&self, entries: &[ExecutionEntry]) -> WindowAssembly {
        self.assemble_with_budget(entries, self.capacity.available_input())
    }

    /// Assembles the window under [`Capacity::available_input`], returning a failing summariser's
    /// error to the caller.
    ///
    /// # Errors
    /// Any error [`WindowSummarizer::summarize`] reports, unchanged. The assembly itself is
    /// identical to [`WindowAssembler::assemble`]'s; on failure it is dropped instead of returned.
    pub fn try_assemble(&self, entries: &[ExecutionEntry]) -> Result<WindowAssembly> {
        self.try_assemble_with_budget(entries, self.capacity.available_input())
    }

    /// Assembles the window for `entries` under an explicit `budget`.
    ///
    /// The passes are deterministic:
    ///
    /// 1. entries are rendered into groups; reply fragments and (by default) lifecycle markers are
    ///    recorded as elided instead of rendered;
    /// 2. the oldest groups beyond [`WindowPolicy::max_recent_groups`] are elided
    ///    ([`ElisionReason::RetentionLimit`]);
    /// 3. the oldest retained group is elided ([`ElisionReason::OverBudget`]) until the retained
    ///    messages plus the summary fit the budget, never dropping below
    ///    [`WindowPolicy::min_recent_groups`];
    /// 4. the summary of everything elided becomes the first message and is counted in the total;
    /// 5. if the minimum retained set still does not fit, the summary is truncated to
    ///    [`WindowPolicy::summary_max_chars`] and [`WindowAssembly::over_budget`] is set.
    ///
    /// The summary is recomputed whenever the elided set grows, so the summariser is called once
    /// per budget-pass iteration and a failing summariser is not retried within one assembly. A
    /// summariser failure does not fail this call: the window carries the documented count-only
    /// fallback marked `摘要不可用` plus the failure in [`WindowAssembly::summary_error`].
    /// [`WindowAssembler::try_assemble_with_budget`] returns the error instead.
    #[must_use]
    pub fn assemble_with_budget(
        &self,
        entries: &[ExecutionEntry],
        budget: usize,
    ) -> WindowAssembly {
        self.assemble_parts(entries, budget).0
    }

    /// Assembles the window for `entries` under an explicit `budget`, returning a failing
    /// summariser's error to the caller.
    ///
    /// # Errors
    /// Any error [`WindowSummarizer::summarize`] reports, unchanged. The assembly of the same
    /// `entries` and `budget` is otherwise identical to
    /// [`WindowAssembler::assemble_with_budget`]'s; on failure it is dropped instead of returned.
    pub fn try_assemble_with_budget(
        &self,
        entries: &[ExecutionEntry],
        budget: usize,
    ) -> Result<WindowAssembly> {
        match self.assemble_parts(entries, budget) {
            (_, Some(error)) => Err(error),
            (assembly, None) => Ok(assembly),
        }
    }

    /// Fits `entries` into `budget`, reporting a summariser failure alongside the assembly.
    ///
    /// The returned assembly is always usable: when the summariser failed, its
    /// [`WindowAssembly::summary_error`] carries the first failure and the summary text is the
    /// marked count-only fallback.
    fn assemble_parts(
        &self,
        entries: &[ExecutionEntry],
        budget: usize,
    ) -> (WindowAssembly, Option<Error>) {
        let max_groups = self.policy.max_recent_groups.max(1);
        let min_groups = self.policy.min_recent_groups.clamp(1, max_groups);
        let (mut groups, mut elided) = build_groups(entries, self.policy.include_lifecycle_markers);

        while groups.len() > max_groups {
            let oldest = groups.remove(0);
            elided.extend(oldest.elisions(ElisionReason::RetentionLimit));
        }

        let mut summary_error: Option<Error> = None;
        loop {
            let mut summary = self.summarise(entries, &elided, &mut summary_error);
            let mut messages = compose(summary.as_deref(), &groups);
            let mut tokens = self.counter.count_messages(&messages);
            if tokens > budget && groups.len() > min_groups {
                let oldest = groups.remove(0);
                elided.extend(oldest.elisions(ElisionReason::OverBudget));
                continue;
            }
            let over_budget = tokens > budget;
            if over_budget {
                if let Some(text) = summary.as_mut() {
                    *text = truncate_chars(text, self.policy.summary_max_chars);
                }
                messages = compose(summary.as_deref(), &groups);
                tokens = self.counter.count_messages(&messages);
            }
            let retained_seqs = groups.iter().flat_map(Group::seqs).collect::<Vec<u64>>();
            elided.sort_by_key(|entry| entry.seq);
            let assembly = WindowAssembly {
                messages,
                tokens,
                budget,
                summary,
                summary_error: summary_error.as_ref().map(ToString::to_string),
                elided,
                retained_seqs,
                over_budget,
            };
            return (assembly, summary_error);
        }
    }

    /// Summarises everything elided so far, or returns `None` when nothing was elided.
    ///
    /// The first summariser failure is recorded in `summary_error` and replaced by the one
    /// documented deterministic fallback ([`fallback_summary`]), which is marked so it can never
    /// pass for a real summary; the failing seam is not called again within this assembly, so a
    /// failing model is charged at most one call per assembly.
    fn summarise(
        &self,
        entries: &[ExecutionEntry],
        elided: &[ElidedEntry],
        summary_error: &mut Option<Error>,
    ) -> Option<String> {
        if elided.is_empty() {
            return None;
        }
        if summary_error.is_some() {
            return Some(fallback_summary(elided.len()));
        }
        let seqs: HashSet<u64> = elided.iter().map(|entry| entry.seq).collect();
        let dropped = entries
            .iter()
            .filter(|entry| seqs.contains(&entry.seq))
            .cloned()
            .collect::<Vec<ExecutionEntry>>();
        match self.summarizer.summarize(&dropped) {
            Ok(text) => Some(text),
            Err(error) => {
                tracing::warn!(
                    error = %error,
                    dropped = dropped.len(),
                    "persistent window summariser failed; the assembly carries a flagged \
                     count-only fallback summary"
                );
                *summary_error = Some(error);
                Some(fallback_summary(elided.len()))
            }
        }
    }
}

/// The one deterministic fallback summary, used only when the summariser failed.
///
/// [`SUMMARY_UNAVAILABLE`] marks it, and the failure itself travels in
/// [`WindowAssembly::summary_error`], so a degraded window is never indistinguishable from a
/// summarised one.
fn fallback_summary(count: usize) -> String {
    format!("{SUMMARY_PREFIX}省略 {count} 条记录（{SUMMARY_UNAVAILABLE}）。")
}

/// Plans window rollovers for a session.
///
/// The manager owns the capacity and the assembler; it never archives anything itself. Archival
/// belongs to the host `SessionStateStore`, so [`WindowManager::plan`] only reports whether the
/// assembled window crossed [`Capacity::rollover_threshold`].
#[derive(Debug, Clone)]
pub struct WindowManager {
    assembler: WindowAssembler,
    capacity: Capacity,
}

impl WindowManager {
    /// Creates a manager over `capacity` with a default-policy assembler.
    #[must_use]
    pub fn new(capacity: Capacity) -> Self {
        Self {
            assembler: WindowAssembler::new(capacity),
            capacity,
        }
    }

    /// Assembles the window for `entries` under the full available input budget.
    #[must_use]
    pub fn assemble(&self, entries: &[ExecutionEntry]) -> WindowAssembly {
        self.assembler.assemble(entries)
    }

    /// Returns whether `tokens` crosses [`Capacity::rollover_threshold`].
    #[must_use]
    pub fn should_rollover(&self, tokens: usize) -> bool {
        tokens > self.capacity.rollover_threshold()
    }

    /// Returns the input budget of a freshly rolled-over window.
    #[must_use]
    pub fn new_window_limit(&self) -> usize {
        self.capacity.new_window_limit()
    }

    /// Assembles the window and decides whether it should be rolled over.
    ///
    /// A summariser failure does not fail the plan: it travels in
    /// [`WindowAssembly::summary_error`] (alongside the marked fallback summary), so a host that
    /// needs the error itself builds its own [`WindowAssembler`] and calls
    /// [`WindowAssembler::try_assemble`].
    #[must_use]
    pub fn plan(&self, entries: &[ExecutionEntry]) -> WindowPlan {
        let assembly = self.assemble(entries);
        let rollover = self.should_rollover(assembly.tokens);
        WindowPlan { assembly, rollover }
    }
}

/// An assembled window together with the rollover decision for it.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct WindowPlan {
    /// The assembled window.
    ///
    /// [`WindowAssembly::summary_error`] reports a degraded summary; the plan itself never hides
    /// it.
    pub assembly: WindowAssembly,
    /// Whether the assembled window should be archived and replaced.
    pub rollover: bool,
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::{
        Capacity, CompletionSummarizer, ElisionReason, EntryKind, ExecutionEntry,
        ExtractiveSummarizer, UnsupportedModelSummarizer, WindowAssembler, WindowManager,
        WindowPolicy, WindowSummarizer, COMPLETION_INSTRUCTION,
    };
    use crate::persistent_agent::entry::{
        EntryPayload, FinalStatus, ModelUsage, PauseReason, ToolStatus,
    };
    use openjiuwen_common::error::Error;
    use openjiuwen_foundation::prompt::{ApproxTokenCounter, BaseMessage, TokenCounter};
    use serde_json::json;

    const TS: &str = "2026-01-01T00:00:00.000Z";

    fn entry(seq: u64, payload: EntryPayload) -> ExecutionEntry {
        ExecutionEntry::new(seq, TS, payload)
    }

    fn model_turn(seq: u64, response: &str) -> ExecutionEntry {
        entry(
            seq,
            EntryPayload::model_turn(format!("mc_{seq}"), response, ModelUsage::known(1, 1)),
        )
    }

    fn tool_intent(seq: u64, call_id: &str) -> ExecutionEntry {
        entry(
            seq,
            EntryPayload::tool_intent(call_id, "bash", r#"{"command":"ls"}"#, "digest", "op_1"),
        )
    }

    fn tool_completed(seq: u64, call_id: &str) -> ExecutionEntry {
        entry(
            seq,
            EntryPayload::tool_completed(call_id, "bash", ToolStatus::Succeeded, "{}", "ok"),
        )
    }

    fn loose_assembler() -> WindowAssembler {
        WindowAssembler::new(Capacity::default())
    }

    fn capacity(context_window: usize) -> Capacity {
        Capacity {
            context_window,
            output_reserve: 100,
            input_limit: context_window,
            safety_margin: 100,
        }
    }

    #[test]
    fn capacity_defaults_and_thresholds() {
        let default = Capacity::default();
        assert_eq!(default.context_window, 128_000);
        assert_eq!(default.output_reserve, 4_096);
        assert_eq!(default.input_limit, 128_000);
        assert_eq!(default.safety_margin, 1_024);
        assert_eq!(default.available_input(), 128_000 - 4_096 - 1_024);
        assert_eq!(default.rollover_threshold(), 102_400);
        assert_eq!(default.new_window_limit(), 51_200);

        let small = capacity(1_000);
        assert_eq!(small.available_input(), 800);
        assert_eq!(small.rollover_threshold(), 800);
        assert_eq!(small.new_window_limit(), 400);

        // Integer floor, never a float: 999 * 4 / 5 = 799 and 999 * 2 / 5 = 399.
        let odd = capacity(999);
        assert_eq!(odd.rollover_threshold(), 799);
        assert_eq!(odd.new_window_limit(), 399);

        // A tighter input_limit wins over the window minus reserves.
        let capped = Capacity {
            input_limit: 200,
            ..capacity(1_000)
        };
        assert_eq!(capped.available_input(), 200);
    }

    #[test]
    fn capacity_validation_names_the_field() {
        assert!(Capacity::default().validate().is_ok());
        assert!(capacity(1_000).validate().is_ok());
        let cases = [
            (
                Capacity {
                    context_window: 0,
                    ..Capacity::default()
                },
                "capacity context_window must be a positive integer",
            ),
            (
                Capacity {
                    output_reserve: 0,
                    ..Capacity::default()
                },
                "capacity output_reserve must be a positive integer",
            ),
            (
                Capacity {
                    input_limit: 0,
                    ..Capacity::default()
                },
                "capacity input_limit must be a positive integer",
            ),
            (
                Capacity {
                    safety_margin: 0,
                    ..Capacity::default()
                },
                "capacity safety_margin must be a positive integer",
            ),
            (
                Capacity {
                    context_window: 1_000,
                    output_reserve: 100,
                    input_limit: 1_001,
                    safety_margin: 100,
                },
                "capacity input_limit must not exceed context_window",
            ),
            (
                Capacity {
                    context_window: 1_000,
                    output_reserve: 600,
                    input_limit: 400,
                    safety_margin: 400,
                },
                "capacity output_reserve plus safety_margin must be smaller than context_window",
            ),
        ];
        for (capacity, message) in cases {
            match capacity.validate().unwrap_err() {
                Error::Validation(actual) => assert_eq!(actual, message),
                other => panic!("unexpected error: {other}"),
            }
        }
    }

    #[test]
    fn window_policy_validation() {
        assert!(WindowPolicy::default().validate().is_ok());
        assert_eq!(WindowPolicy::default().max_recent_groups, 8);
        assert_eq!(WindowPolicy::default().summary_max_chars, 4_096);
        assert_eq!(WindowPolicy::default().min_recent_groups, 1);
        assert!(!WindowPolicy::default().include_lifecycle_markers);
        let cases = [
            (
                WindowPolicy {
                    max_recent_groups: 0,
                    ..WindowPolicy::default()
                },
                "window policy max_recent_groups must be a positive integer",
            ),
            (
                WindowPolicy {
                    min_recent_groups: 0,
                    ..WindowPolicy::default()
                },
                "window policy min_recent_groups must be a positive integer",
            ),
            (
                WindowPolicy {
                    max_recent_groups: 2,
                    min_recent_groups: 3,
                    ..WindowPolicy::default()
                },
                "window policy min_recent_groups must not exceed max_recent_groups",
            ),
            (
                WindowPolicy {
                    summary_max_chars: 15,
                    ..WindowPolicy::default()
                },
                "window policy summary_max_chars must be at least 16",
            ),
        ];
        for (policy, message) in cases {
            match policy.validate().unwrap_err() {
                Error::Validation(actual) => assert_eq!(actual, message),
                other => panic!("unexpected error: {other}"),
            }
        }
    }

    #[test]
    fn loose_budget_keeps_every_entry_and_emits_no_summary() {
        let entries = vec![
            entry(1, EntryPayload::execution_started("ship the fix")),
            tool_intent(2, "call_1"),
            tool_completed(3, "call_1"),
            model_turn(4, "done"),
        ];
        let assembly = loose_assembler().assemble(&entries);
        assert_eq!(assembly.retained_seqs, vec![1, 2, 3, 4]);
        assert_eq!(assembly.elided, Vec::new());
        assert!(assembly.summary.is_none());
        assert!(!assembly.over_budget);
        assert_eq!(assembly.budget, Capacity::default().available_input());
        assert_eq!(assembly.messages.len(), 4);
        assert_eq!(
            assembly.messages[0].text().as_ref(),
            "【本次执行】ship the fix"
        );
        assert_eq!(assembly.messages[1].tool_calls.len(), 1);
        assert_eq!(assembly.messages[1].tool_calls[0].name, "bash");
        // Raw arguments are not journalled, so the persisted summary stands in for them.
        assert_eq!(
            assembly.messages[1].tool_calls[0].arguments,
            r#"{"command":"ls"}"#
        );
        assert_eq!(assembly.messages[2].tool_call_id.as_deref(), Some("call_1"));
        assert_eq!(assembly.messages[3].text().as_ref(), "done");
        let counter = ApproxTokenCounter::new();
        assert_eq!(assembly.tokens, counter.count_messages(&assembly.messages));
        assert_eq!(assembly.elided_count(), 0);
    }

    #[test]
    fn empty_model_turn_renders_a_placeholder() {
        let entries = vec![model_turn(1, "")];
        let assembly = loose_assembler().assemble(&entries);
        assert_eq!(assembly.messages.len(), 1);
        assert_eq!(assembly.messages[0].text().as_ref(), "(no content)");
    }

    #[test]
    fn tight_budget_elides_the_oldest_groups_and_summarises() {
        let entries = vec![
            model_turn(1, &"a".repeat(400)),
            model_turn(2, &"b".repeat(400)),
            model_turn(3, &"c".repeat(40)),
        ];
        let assembly = loose_assembler().assemble_with_budget(&entries, 120);
        assert_eq!(assembly.retained_seqs, vec![3]);
        assert_eq!(assembly.elided_count(), 2);
        assert_eq!(assembly.elided[0].seq, 1);
        assert_eq!(assembly.elided[0].kind, EntryKind::ModelTurn);
        assert_eq!(assembly.elided[0].reason, ElisionReason::OverBudget);
        assert_eq!(assembly.elided[1].seq, 2);
        assert_eq!(assembly.elided[1].reason, ElisionReason::OverBudget);
        let summary = assembly.summary.clone().unwrap();
        assert!(summary.starts_with("【窗口摘要】省略 2 条记录"));
        assert!(summary.contains("model_turn×2"));
        // The summary message comes before the retained messages and counts towards the total.
        assert_eq!(assembly.messages.len(), 2);
        assert_eq!(assembly.messages[0].text().as_ref(), summary);
        assert_eq!(assembly.messages[1].text().as_ref(), "c".repeat(40));
        assert!(!assembly.over_budget);
        assert!(assembly.tokens <= assembly.budget);
        let counter = ApproxTokenCounter::new();
        assert_eq!(assembly.tokens, counter.count_messages(&assembly.messages));
    }

    #[test]
    fn budget_of_one_keeps_the_minimum_and_flags_over_budget() {
        let entries = vec![model_turn(1, "old"), model_turn(2, "new")];
        let assembly = loose_assembler().assemble_with_budget(&entries, 1);
        assert_eq!(assembly.retained_seqs, vec![2]);
        assert_eq!(assembly.elided_count(), 1);
        assert_eq!(assembly.elided[0].seq, 1);
        assert_eq!(assembly.elided[0].reason, ElisionReason::OverBudget);
        assert!(assembly.over_budget);
        assert!(assembly.tokens > assembly.budget);
        assert!(assembly.summary.is_some());
        assert_eq!(assembly.messages.len(), 2);
    }

    #[test]
    fn a_tool_group_is_elided_as_a_whole() {
        let entries = vec![
            tool_intent(1, "call_1"),
            tool_completed(2, "call_1"),
            model_turn(3, "done"),
        ];
        let assembly = loose_assembler().assemble_with_budget(&entries, 6);
        assert_eq!(assembly.retained_seqs, vec![3]);
        assert_eq!(assembly.elided_count(), 2);
        assert_eq!(assembly.elided[0].seq, 1);
        assert_eq!(assembly.elided[0].kind, EntryKind::ToolIntent);
        assert_eq!(assembly.elided[0].reason, ElisionReason::OverBudget);
        assert_eq!(assembly.elided[1].seq, 2);
        assert_eq!(assembly.elided[1].kind, EntryKind::ToolCompleted);
        assert_eq!(assembly.elided[1].reason, ElisionReason::OverBudget);
        // Neither the call nor its result survives on its own.
        assert!(!assembly
            .messages
            .iter()
            .any(|message| !message.tool_calls.is_empty() || message.tool_call_id.is_some()));
    }

    #[test]
    fn a_tool_result_without_an_intent_is_a_standalone_message() {
        let entries = vec![tool_completed(1, "call_1"), model_turn(2, "done")];
        let assembly = loose_assembler().assemble(&entries);
        assert_eq!(assembly.retained_seqs, vec![1, 2]);
        assert_eq!(assembly.messages.len(), 2);
        assert_eq!(assembly.messages[0].tool_call_id.as_deref(), Some("call_1"));
        assert_eq!(assembly.messages[0].text().as_ref(), "ok");
    }

    #[test]
    fn reply_deltas_are_counted_only_and_never_rendered() {
        let entries = vec![
            entry(1, EntryPayload::reply_delta("hel")),
            entry(2, EntryPayload::reply_delta("lo")),
            model_turn(3, "final"),
        ];
        let assembly = loose_assembler().assemble(&entries);
        assert_eq!(assembly.retained_seqs, vec![3]);
        assert_eq!(assembly.elided_count(), 2);
        for elided in &assembly.elided {
            assert_eq!(elided.kind, EntryKind::ReplyDelta);
            assert_eq!(elided.reason, ElisionReason::CountedOnly);
        }
        assert_eq!(assembly.messages.len(), 2);
        assert!(assembly.summary.is_some());
        assert!(!assembly
            .messages
            .iter()
            .any(|message| message.text().contains("hel")));
    }

    #[test]
    fn lifecycle_markers_are_elided_by_default_and_rendered_when_enabled() {
        let entries = vec![
            entry(
                1,
                EntryPayload::execution_paused(PauseReason::AwaitingConfirmation, None),
            ),
            entry(
                2,
                EntryPayload::execution_final(FinalStatus::Completed, Some("all done".into())),
            ),
            entry(3, EntryPayload::note("checkpoint")),
        ];
        let assembly = loose_assembler().assemble(&entries);
        assert_eq!(assembly.retained_seqs, Vec::<u64>::new());
        assert_eq!(assembly.elided_count(), 3);
        for elided in &assembly.elided {
            assert_eq!(elided.reason, ElisionReason::LifecycleMarker);
        }
        assert_eq!(assembly.messages.len(), 1);
        assert!(assembly
            .summary
            .unwrap()
            .starts_with("【窗口摘要】省略 3 条记录"));

        let assembler = loose_assembler().with_policy(WindowPolicy {
            include_lifecycle_markers: true,
            ..WindowPolicy::default()
        });
        let assembly = assembler.assemble(&entries);
        assert_eq!(assembly.elided, Vec::new());
        assert!(assembly.summary.is_none());
        assert_eq!(assembly.retained_seqs, vec![1, 2, 3]);
        let rendered = assembly
            .messages
            .iter()
            .map(BaseMessage::text)
            .collect::<Vec<_>>();
        assert_eq!(rendered[0], "[execution_paused] awaiting_confirmation");
        assert_eq!(rendered[1], "[execution_final] completed all done");
        assert_eq!(rendered[2], "[note] checkpoint");
    }

    #[test]
    fn max_recent_groups_drops_older_groups_with_retention_limit() {
        let entries = vec![
            model_turn(1, "one"),
            model_turn(2, "two"),
            model_turn(3, "three"),
            model_turn(4, "four"),
        ];
        let assembler = loose_assembler().with_policy(WindowPolicy {
            max_recent_groups: 2,
            min_recent_groups: 1,
            ..WindowPolicy::default()
        });
        let assembly = assembler.assemble(&entries);
        assert_eq!(assembly.retained_seqs, vec![3, 4]);
        assert_eq!(assembly.elided_count(), 2);
        assert_eq!(assembly.elided[0].seq, 1);
        assert_eq!(assembly.elided[0].reason, ElisionReason::RetentionLimit);
        assert_eq!(assembly.elided[1].seq, 2);
        assert_eq!(assembly.elided[1].reason, ElisionReason::RetentionLimit);
        assert!(assembly.summary.is_some());
        assert!(!assembly.over_budget);
        assert_eq!(assembly.messages.len(), 3);
    }

    #[test]
    fn identical_input_yields_identical_assemblies() {
        let entries = vec![
            entry(1, EntryPayload::execution_started("goal")),
            tool_intent(2, "call_1"),
            tool_completed(3, "call_1"),
            entry(4, EntryPayload::reply_delta("par")),
            model_turn(5, &"x".repeat(200)),
        ];
        let assembler = loose_assembler();
        let first = assembler.assemble_with_budget(&entries, 60);
        let second = assembler.assemble_with_budget(&entries, 60);
        assert_eq!(first, second);
        assert_eq!(
            serde_json::to_value(&first).unwrap(),
            serde_json::to_value(&second).unwrap()
        );
    }

    #[test]
    fn unsupported_model_summariser_reports_the_documented_message() {
        let error = UnsupportedModelSummarizer.summarize(&[]).unwrap_err();
        match error {
            Error::Unsupported(message) => assert_eq!(
                message,
                "persistent window summarisation has no completion seam to call; inject one with \
                 WindowAssembler::with_summarizer — CompletionSummarizer for a model-backed \
                 summary, or ExtractiveSummarizer for the deterministic offline default"
            ),
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn a_failing_summariser_is_flagged_alongside_the_fallback() {
        let entries = vec![model_turn(1, "old"), model_turn(2, "new")];
        let assembler = loose_assembler()
            .with_policy(WindowPolicy {
                max_recent_groups: 1,
                ..WindowPolicy::default()
            })
            .with_summarizer(std::sync::Arc::new(UnsupportedModelSummarizer));
        let assembly = assembler.assemble(&entries);
        assert_eq!(assembly.retained_seqs, vec![2]);
        assert_eq!(
            assembly.summary.as_deref(),
            Some("【窗口摘要】省略 1 条记录（摘要不可用）。"),
            "the fallback stays byte-identical, but it names itself as unavailable"
        );
        let reported = assembly
            .summary_error
            .as_deref()
            .expect("the failure is reported on the assembly");
        assert!(reported.starts_with("unsupported: persistent window summarisation"));
        assert!(!assembly.over_budget);

        // The fallible path returns the summariser's own error instead of the degraded window.
        match assembler.try_assemble(&entries).unwrap_err() {
            Error::Unsupported(message) => {
                assert!(message.contains("WindowAssembler::with_summarizer"));
            }
            other => panic!("expected the summariser error, got {other:?}"),
        }
        assert!(
            assembler
                .try_assemble_with_budget(&entries, 10_000)
                .is_err(),
            "the fallible path is budget-independent"
        );
    }

    #[test]
    fn a_successful_summariser_leaves_no_summary_error() {
        let entries = vec![model_turn(1, "old"), model_turn(2, "new")];
        let assembly = loose_assembler()
            .with_policy(WindowPolicy {
                max_recent_groups: 1,
                ..WindowPolicy::default()
            })
            .assemble(&entries);
        assert!(assembly.summary.is_some());
        assert_eq!(assembly.summary_error, None);
        assert_eq!(
            loose_assembler()
                .try_assemble(&entries)
                .unwrap()
                .summary_error,
            None
        );
    }

    #[test]
    fn completion_summarizer_sends_a_bounded_prompt_and_returns_the_answer() {
        let seen = std::sync::Arc::new(parking_lot::Mutex::new(Vec::<String>::new()));
        let sink = std::sync::Arc::clone(&seen);
        let summarizer = CompletionSummarizer::new(move |prompt: &str| {
            sink.lock().push(prompt.to_owned());
            Ok("  模型写的摘要  ".to_owned())
        });
        let entries = vec![model_turn(3, "hello"), tool_intent(7, "call_1")];

        assert_eq!(summarizer.max_chars(), 4_096);
        assert_eq!(summarizer.last_prompt(), None);
        let summary = summarizer.summarize(&entries).unwrap();
        assert_eq!(summary, "模型写的摘要", "the answer is trimmed");

        let sent = seen.lock().clone();
        assert_eq!(sent.len(), 1);
        assert_eq!(sent[0], summarizer.prompt(&entries));
        assert_eq!(summarizer.last_prompt().as_deref(), Some(sent[0].as_str()));
        for expected in [
            "【摘要请求】省略 2 条记录",
            "model_turn×1, tool_intent×1",
            "最早 seq 3",
            "最晚 seq 7",
            "模型回复：hello",
            "工具：bash",
            COMPLETION_INSTRUCTION,
        ] {
            assert!(
                sent[0].contains(expected),
                "the prompt must carry {expected:?}: {}",
                sent[0]
            );
        }
        assert!(
            !sent[0].contains("【窗口摘要】"),
            "the prompt must not reuse the summary prefix"
        );
    }

    #[test]
    fn completion_summarizer_prompt_is_bounded_and_handles_no_entries() {
        let summarizer = CompletionSummarizer::new(|_| Ok("ok".to_owned()));
        let entries = (1..=50)
            .map(|seq| model_turn(seq, &"q".repeat(400)))
            .collect::<Vec<ExecutionEntry>>();
        let prompt = summarizer.prompt(&entries);
        assert!(
            prompt.chars().count() < 1_000,
            "the prompt stays bounded whatever the journal holds: {} chars",
            prompt.chars().count()
        );
        assert_eq!(
            prompt.matches("模型回复：").count(),
            3,
            "three excerpts at most"
        );
        assert!(
            !prompt.contains(&"q".repeat(121)),
            "every excerpt is bounded"
        );
        assert!(summarizer.prompt(&[]).contains("省略 0 条记录"));
        assert_eq!(
            summarizer.summarize(&[]).unwrap(),
            "【窗口摘要】省略 0 条记录。",
            "an empty elision set needs no completion call"
        );
        assert_eq!(summarizer.last_prompt(), None);

        let debug = format!("{summarizer:?}");
        assert!(debug.contains("max_chars: 4096"), "bounded debug: {debug}");
        assert!(
            !debug.contains('q'),
            "the debug output never quotes the prompt"
        );
    }

    #[test]
    fn completion_summarizer_truncates_and_rejects_empty_answers() {
        let bounded = CompletionSummarizer::with_max_chars(|_| Ok("z".repeat(50)), 10);
        let summary = bounded.summarize(&[model_turn(1, "x")]).unwrap();
        assert_eq!(summary.chars().count(), 10, "the answer is bounded");
        assert!(summary.ends_with('…'));

        let empty = CompletionSummarizer::new(|_| Ok("  \n\t ".to_owned()));
        match empty.summarize(&[model_turn(1, "x")]).unwrap_err() {
            Error::Validation(message) => {
                assert!(
                    message.contains("completion seam") && message.contains("with_summarizer"),
                    "the refusal names the seam: {message}"
                );
            }
            other => panic!("expected a typed refusal, got {other:?}"),
        }

        let failing = CompletionSummarizer::new(|_| Err(Error::llm("endpoint unreachable")));
        assert_eq!(
            failing
                .summarize(&[model_turn(1, "x")])
                .unwrap_err()
                .to_string(),
            "llm: endpoint unreachable",
            "a seam error is propagated unchanged"
        );
    }

    #[test]
    fn a_completion_summariser_drives_the_assembly_summary() {
        let entries = vec![model_turn(1, "old"), model_turn(2, "new")];
        let assembler = loose_assembler()
            .with_policy(WindowPolicy {
                max_recent_groups: 1,
                ..WindowPolicy::default()
            })
            .with_summarizer(std::sync::Arc::new(CompletionSummarizer::new(
                |prompt: &str| Ok(format!("模型摘要（{} 字符）", prompt.chars().count())),
            )));
        let assembly = assembler.assemble(&entries);
        assert_eq!(assembly.retained_seqs, vec![2]);
        let summary = assembly.summary.clone().unwrap();
        assert!(summary.starts_with("模型摘要"), "got {summary}");
        assert_eq!(assembly.summary_error, None);
        assert_eq!(assembly.messages[0].text().as_ref(), summary);
    }

    #[test]
    fn extractive_summary_lists_counts_in_declaration_order_and_a_hint() {
        let entries = vec![model_turn(3, "hello"), tool_intent(7, "call_1")];
        let summary = ExtractiveSummarizer::default().summarize(&entries).unwrap();
        assert_eq!(
            summary,
            "【窗口摘要】省略 2 条记录（model_turn×1, tool_intent×1）；最早 seq 3，最晚 seq 7。\
             首个模型回复：hello。"
        );
        let tool_only = ExtractiveSummarizer::default()
            .summarize(&entries[1..])
            .unwrap();
        assert_eq!(
            tool_only,
            "【窗口摘要】省略 1 条记录（tool_intent×1）；最早 seq 7，最晚 seq 7。首个工具：bash。"
        );
    }

    #[test]
    fn extractive_summary_is_bounded_and_handles_no_entries() {
        assert_eq!(
            ExtractiveSummarizer::default().summarize(&[]).unwrap(),
            "【窗口摘要】省略 0 条记录。"
        );
        let entries = vec![model_turn(1, &"y".repeat(400))];
        let summary = ExtractiveSummarizer::new(20).summarize(&entries).unwrap();
        assert!(summary.chars().count() <= 20);
        assert_eq!(
            summary.chars().count(),
            20,
            "the ellipsis is part of the budget"
        );
        assert!(summary.ends_with('…'));
        assert!(summary.starts_with("【窗口摘要】"));
    }

    #[test]
    fn manager_plans_rollovers_around_the_threshold() {
        let manager = WindowManager::new(capacity(1_000));
        assert!(!manager.should_rollover(800));
        assert!(manager.should_rollover(801));
        assert_eq!(manager.new_window_limit(), 400);
        let entries = vec![model_turn(1, "short")];
        let plan = manager.plan(&entries);
        assert!(!plan.rollover);
        assert_eq!(plan.assembly.retained_seqs, vec![1]);
        assert_eq!(plan.assembly.budget, 800);
        assert_eq!(plan.assembly, manager.assemble(&entries));

        let long = vec![model_turn(1, &"z".repeat(4_000))];
        let plan = manager.plan(&long);
        assert_eq!(plan.assembly.tokens, 1_004);
        assert!(plan.rollover);
    }

    #[test]
    fn assembly_is_serializable() {
        let entries = vec![model_turn(1, "hello")];
        let assembly = loose_assembler().assemble(&entries);
        let value = serde_json::to_value(&assembly).unwrap();
        assert_eq!(value["retained_seqs"], json!([1]));
        assert_eq!(value["elided"], json!([]));
        assert_eq!(value["over_budget"], json!(false));
        assert!(
            value.get("summary_error").is_none(),
            "a healthy assembly omits the failure field"
        );
        let restored: super::WindowAssembly = serde_json::from_value(value).unwrap();
        assert_eq!(restored, assembly);

        // A degraded assembly keeps the failure and the marked fallback through a round trip.
        let degraded = loose_assembler()
            .with_policy(WindowPolicy {
                max_recent_groups: 1,
                ..WindowPolicy::default()
            })
            .with_summarizer(std::sync::Arc::new(UnsupportedModelSummarizer))
            .assemble(&[model_turn(1, "old"), model_turn(2, "new")]);
        let value = serde_json::to_value(&degraded).unwrap();
        assert!(value["summary_error"].is_string());
        assert!(value["summary"].as_str().unwrap().contains("摘要不可用"));
        let restored: super::WindowAssembly = serde_json::from_value(value).unwrap();
        assert_eq!(restored, degraded);
    }
}
