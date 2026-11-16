//! Fork and compact of member context: child-spec derivation and deterministic compaction.
//!
//! Python parity sources: `openjiuwen/agent_teams/fork.py` (`ForkContext`) and
//! `openjiuwen/agent_teams/fork_compact.py` (`compact_context`). Python snapshots a live agent's
//! conversation, strips every `SystemMessage` so the source member's role identity cannot leak
//! into the target, and later compresses one side of that history with a model-generated summary.
//!
//! This port keeps the intent and drops the live-agent dependency:
//!
//! - [`fork_member_context`] is a pure function of
//!   the parent's `DeepAgentSpec` plus the child's `TeamMemberSpec`. It produces the child's
//!   blueprint (the "forked member"): session id, workspace layout, working directory, project
//!   root and model are inherited, while the parent's `system_prompt` and `card` are cleared —
//!   exactly Python's `SystemMessage` stripping, because the child's role must come from its own
//!   `TeamMemberSpec::base::prompt`, never from the fork source.
//! - [`compact_transcript`] replaces Python's
//!   `_run_compaction` (one LLM call) with a deterministic template line that reports what was
//!   dropped, so a fork can be planned offline and replayed byte-for-byte in a test.
//!
//! Deliberate deviations from Python, each documented at its item:
//!
//! - the fork carries a *blueprint*, not messages: Python's checkpoint/split machinery
//!   (`compact_split`, `compact_direction`, `checkpoint` truncation) has no counterpart here;
//! - compaction never grows a transcript: a summary that would cost more than the messages it
//!   replaces is omitted, and a tool truncation that would not shorten the content is skipped;
//! - the tool-result budget applies to [`CompactRole::Tool`] messages only, so the summary line
//!   (a system message) can never be truncated by it.
//!
//! Everything here is pure: no I/O, no clock, no randomness, no process-global state.

use std::collections::{BTreeMap, BTreeSet};

use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::constants::is_reserved_name;
use crate::schema::{
    DeepAgentSpec, ModelRef, TeamAgentSpec, TeamMemberSpec, ToolSpec, WorkspaceSpec,
};

// The template-driven entry point lives in the sibling [`super::compact_template`] module and is
// re-exported here so `fork_compact` keeps presenting the whole compaction surface.
pub use super::compact_template::{
    compact_with_template, BudgetTier, CompactDirection, CompactTemplate, CompactionSide,
    SummaryFact, SummaryOmission, SummaryRetention, TemplateCompaction, DEFAULT_BUDGET_PRIORITY,
    DEFAULT_EXCERPT_CHARS, DEFAULT_TEMPLATE_MAX_FACTS,
};

/// Prefix of the single summary message compaction emits.
pub const SUMMARY_PREFIX: &str = "[compacted] ";

/// Suffix appended to a tool result that was cut to the budget.
pub const DEFAULT_TRUNCATED_SUFFIX: &str = "… [truncated]";

/// Number of trailing non-system messages [`CompactPolicy`] keeps by default.
pub const DEFAULT_KEEP_LAST_MESSAGES: usize = 6;

/// Per-tool-result byte budget [`CompactPolicy`] applies by default.
pub const DEFAULT_MAX_TOOL_RESULT_BYTES: usize = 2_048;

// ---------------------------------------------------------------------------------------------
// Fork: what the child inherits from the parent member
// ---------------------------------------------------------------------------------------------

/// A task the forked member starts on, replacing the parent's current task.
///
/// The fork itself never writes the task board: this is the *intent* handed to the child's
/// runtime, which creates the record through [`crate::taskboard`]. `parent_task_id` keeps a fork
/// that continues an existing task linked to it, so a fork chain stays traceable on the board.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case", default)]
pub struct ForkTaskOverride {
    /// Short task title.
    pub title: String,
    /// Task body: the concrete instruction the child must carry out.
    pub content: String,
    /// Board task this fork continues, when it is a continuation rather than a fresh task.
    pub parent_task_id: Option<String>,
    /// Free-form metadata copied onto the child's task record.
    pub metadata: BTreeMap<String, Value>,
}

impl ForkTaskOverride {
    /// Creates an unparented task override with empty metadata.
    #[must_use]
    pub fn new(title: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            content: content.into(),
            parent_task_id: None,
            metadata: BTreeMap::new(),
        }
    }

    /// Sets [`ForkTaskOverride::parent_task_id`].
    #[must_use]
    pub fn with_parent_task(mut self, task_id: impl Into<String>) -> Self {
        self.parent_task_id = Some(task_id.into());
        self
    }
}

/// How a fork changes the child's tool set relative to its parent's.
///
/// Three knobs, applied in a fixed order:
///
/// 1. [`ForkToolPolicy::denied_tools`] is a hard refusal: a tool with that name is dropped
///    whatever its origin (inherited from the parent or listed in
///    [`ForkToolPolicy::additive_tools`]).
/// 2. [`ForkToolPolicy::allowed_tools`], when `Some`, keeps only inherited tools whose name is in
///    the set. `None` means "inherit the parent's list as-is". The additive list is *not* filtered
///    by it: those tools are explicitly requested by the fork's author.
/// 3. [`ForkToolPolicy::additive_tools`] is appended after the inherited tools. A tool whose name
///    is already present is skipped, so the result never contains a duplicate name.
///
/// [`ForkToolPolicy::apply`] preserves the parent's order first and the additive order second, so
/// the result is deterministic without relying on hashing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case", default)]
pub struct ForkToolPolicy {
    /// Tool names the child may never use.
    pub denied_tools: BTreeSet<String>,
    /// When `Some`, the only inherited tool names the child keeps.
    pub allowed_tools: Option<BTreeSet<String>>,
    /// Extra tools the child gets on top of the inherited ones.
    pub additive_tools: Vec<ToolSpec>,
}

impl ForkToolPolicy {
    /// Creates the empty policy, which passes the parent's tool list through unchanged.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds `tool` to [`ForkToolPolicy::denied_tools`].
    #[must_use]
    pub fn denying(mut self, tool: impl Into<String>) -> Self {
        self.denied_tools.insert(tool.into());
        self
    }

    /// Restricts the inherited tools to `tools`.
    #[must_use]
    pub fn allowing_only<I: IntoIterator<Item = String>>(mut self, tools: I) -> Self {
        self.allowed_tools = Some(tools.into_iter().collect());
        self
    }

    /// Appends `tool` to [`ForkToolPolicy::additive_tools`].
    #[must_use]
    pub fn with_tool(mut self, tool: ToolSpec) -> Self {
        self.additive_tools.push(tool);
        self
    }

    /// Applies this policy to `parent_tools`; deterministic order, no duplicate names.
    ///
    /// See the type-level docs for the precedence between denial, the allow-list and the additive
    /// list.
    #[must_use]
    pub fn apply(&self, parent_tools: &[ToolSpec]) -> Vec<ToolSpec> {
        let mut applied: Vec<ToolSpec> =
            Vec::with_capacity(parent_tools.len() + self.additive_tools.len());
        let mut seen: BTreeSet<&str> = BTreeSet::new();

        for tool in parent_tools {
            if self.denied_tools.contains(&tool.name) {
                continue;
            }
            if let Some(allowed) = &self.allowed_tools {
                if !allowed.contains(&tool.name) {
                    continue;
                }
            }
            if seen.insert(tool.name.as_str()) {
                applied.push(tool.clone());
            }
        }

        for tool in &self.additive_tools {
            if self.denied_tools.contains(&tool.name) {
                continue;
            }
            if seen.insert(tool.name.as_str()) {
                applied.push(tool.clone());
            }
        }

        applied
    }
}

/// The parent member a fork starts from.
///
/// Python reads the parent's live conversation; this port reads the parent's *effective*
/// blueprint instead, so `spec` is expected to already carry the resolved workspace, working
/// directory, project root and model (the runtime copies them onto the member spec when it
/// materializes the member).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case", default)]
pub struct ForkSource {
    /// Team the parent member belongs to.
    pub team_name: String,
    /// Member name of the parent.
    pub member_name: String,
    /// Session id of the parent, inherited verbatim by the child.
    pub session_id: String,
    /// Roster entry of the parent, useful for display and for traceability.
    pub member: TeamMemberSpec,
    /// Effective blueprint of the parent; the child's blueprint is derived from it.
    pub spec: DeepAgentSpec,
    /// Ancestors of the parent, nearest last; empty when the parent is a root member.
    pub lineage: Vec<String>,
}

impl ForkSource {
    /// Creates a root source: `lineage` is empty because the parent is not itself a fork.
    #[must_use]
    pub fn new(
        team: impl Into<String>,
        member_name: impl Into<String>,
        session_id: impl Into<String>,
        member: TeamMemberSpec,
        spec: DeepAgentSpec,
    ) -> Self {
        Self {
            team_name: team.into(),
            member_name: member_name.into(),
            session_id: session_id.into(),
            member,
            spec,
            lineage: Vec::new(),
        }
    }
}

/// The child blueprint a fork produces.
///
/// `spec` is the child's `DeepAgentSpec`; [`ForkedMemberSpec::workspace`], [`ForkedMemberSpec::cwd`]
/// and [`ForkedMemberSpec::project_root`] repeat the values inherited from the parent so a caller
/// can read them without re-deriving them from `spec`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case", default)]
pub struct ForkedMemberSpec {
    /// Roster entry of the child, as supplied to the fork.
    pub member: TeamMemberSpec,
    /// Blueprint of the child: parent tools filtered by the policy, identity cleared.
    pub spec: DeepAgentSpec,
    /// Team the fork source belonged to.
    pub parent_team: String,
    /// Member name of the fork source.
    pub parent_member: String,
    /// Session id inherited from the parent.
    pub session_id: String,
    /// Workspace inherited from the parent, when the parent had one.
    pub workspace: Option<WorkspaceSpec>,
    /// Working directory inherited from the parent.
    pub cwd: Option<String>,
    /// Project root inherited from the parent.
    pub project_root: Option<String>,
    /// Task the child starts on, when the fork overrides the parent's task.
    pub task: Option<ForkTaskOverride>,
    /// Ancestors of the child, nearest last: the parent chain with the parent appended.
    pub lineage: Vec<String>,
    /// Number of fork generations above the child; `1` for a fork of a root member.
    ///
    /// Always equal to `lineage.len()`: each generation contributes exactly one ancestor.
    pub fork_depth: usize,
}

impl ForkedMemberSpec {
    /// Returns the child's member name.
    #[must_use]
    pub fn member_name(&self) -> &str {
        &self.member.base.member_name
    }

    /// Returns `true` when the child's parent was a root member (`fork_depth <= 1`).
    #[must_use]
    pub fn is_root(&self) -> bool {
        self.fork_depth <= 1
    }
}

/// Builds the child member's blueprint from the parent's context, task override and tool policy.
///
/// Rules, in order:
///
/// 1. `child.base.member_name` must match `^[a-z][a-z0-9_-]*$`; otherwise the fork is rejected.
/// 2. It must not be a reserved name ([`crate::constants::is_reserved_name`]).
/// 3. It must differ from the parent's name — a fork is a *new* member, never a rename.
/// 4. The child's `spec` is the parent's `spec` with:
///    - `tools = tool_policy.apply(&source.spec.tools)`;
///    - `system_prompt = None` (the parent's role identity must not leak into the child; the
///      child's role comes from [`TeamMemberSpec::base`]'s `prompt`);
///    - `card = None` (the parent card names the parent; [`TeamMemberSpec`] carries no card, so
///      the child's identity is built downstream by its runtime);
///    - `model` = the parent's model, unless `child.base.model_name` names a model, in which case
///      it becomes a [`ModelRef`] with that name and the parent's provider. A blank
///      `model_name` counts as absent.
///
/// Everything else (MCP servers, sub-agents, rails, skills, language, approval list, timeouts,
/// unmodelled Python fields in `extra`) is inherited untouched. `session_id`, `workspace`, `cwd`
/// and `project_root` come from the parent, so the child shares the parent's session identity and
/// workspace layout. `lineage` is the parent's lineage plus the parent, and `fork_depth` is its
/// length.
///
/// # Errors
///
/// Returns [`Error::Validation`] for a malformed child name, a reserved child name, or a child
/// that is the parent itself.
pub fn fork_member_context(
    source: &ForkSource,
    child: TeamMemberSpec,
    task: Option<ForkTaskOverride>,
    tool_policy: &ForkToolPolicy,
) -> Result<ForkedMemberSpec> {
    let member_name = child.base.member_name.as_str();
    if !TeamAgentSpec::is_member_name_valid(member_name) {
        return Err(Error::validation(format!(
            "invalid member name '{member_name}': expected ^[a-z][a-z0-9_-]*$"
        )));
    }
    if is_reserved_name(member_name) {
        return Err(Error::validation(format!(
            "member name '{member_name}' is reserved"
        )));
    }
    if member_name == source.member_name {
        return Err(Error::validation(format!(
            "cannot fork member '{member_name}': a fork must name a new member"
        )));
    }

    let mut spec = source.spec.clone();
    spec.tools = tool_policy.apply(&source.spec.tools);
    // Python strips every `SystemMessage` from a captured context so the source's role identity
    // never reaches the target; the blueprint counterparts are the system prompt and the card.
    spec.system_prompt = None;
    spec.card = None;
    if let Some(model_name) = child
        .base
        .model_name
        .as_deref()
        .filter(|name| !name.is_empty())
    {
        spec.model = Some(ModelRef {
            name: None,
            model_name: Some(model_name.to_owned()),
            provider: source
                .spec
                .model
                .as_ref()
                .and_then(|model| model.provider.clone()),
            params: BTreeMap::new(),
        });
    }

    let mut lineage = source.lineage.clone();
    lineage.push(source.member_name.clone());
    let fork_depth = lineage.len();

    Ok(ForkedMemberSpec {
        member: child,
        workspace: spec.workspace.clone(),
        cwd: spec.cwd.clone(),
        project_root: spec.project_root.clone(),
        spec,
        parent_team: source.team_name.clone(),
        parent_member: source.member_name.clone(),
        session_id: source.session_id.clone(),
        task,
        lineage,
        fork_depth,
    })
}

// ---------------------------------------------------------------------------------------------
// Compact: a deterministic, model-free reduction of an inherited transcript
// ---------------------------------------------------------------------------------------------

/// Speaker of a [`TranscriptMessage`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum CompactRole {
    /// System prompt; kept at the front by default and never subject to the tool budget.
    #[default]
    System,
    /// End user (or the parent member's task instruction).
    User,
    /// Model turn.
    Assistant,
    /// Tool result.
    Tool,
}

impl CompactRole {
    /// Returns the Python wire string of this role (`system`, `user`, `assistant`, `tool`).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Tool => "tool",
        }
    }
}

/// One message of a transcript that is about to be inherited by a fork.
///
/// This is the JSON-native counterpart of Python's `BaseMessage`: role, content and the tool
/// bookkeeping a tool result needs to stay interpretable. Unknown fields are ignored on input and
/// every missing field defaults, so a transcript persisted by another release still loads.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case", default)]
pub struct TranscriptMessage {
    /// Who produced the message.
    pub role: CompactRole,
    /// Message text.
    pub content: String,
    /// Tool the message belongs to; only set on [`CompactRole::Tool`] messages.
    pub tool_name: Option<String>,
    /// Call id the tool result answers, when the producer recorded one.
    pub tool_call_id: Option<String>,
}

impl TranscriptMessage {
    /// Creates a system message.
    #[must_use]
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: CompactRole::System,
            content: content.into(),
            ..Self::default()
        }
    }

    /// Creates a user message.
    #[must_use]
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: CompactRole::User,
            content: content.into(),
            ..Self::default()
        }
    }

    /// Creates an assistant message.
    #[must_use]
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: CompactRole::Assistant,
            content: content.into(),
            ..Self::default()
        }
    }

    /// Creates a tool result message attributed to `tool_name`.
    #[must_use]
    pub fn tool(tool_name: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: CompactRole::Tool,
            content: content.into(),
            tool_name: Some(tool_name.into()),
            tool_call_id: None,
        }
    }

    /// Byte weight of this message: its content plus the role label and the tool bookkeeping.
    ///
    /// Weighing the labels too keeps two messages with equal content but different roles from
    /// cancelling out in the byte accounting that drives the "never grow" guard.
    #[must_use]
    pub fn byte_len(&self) -> usize {
        message_weight(
            self.role,
            &self.content,
            self.tool_name.as_deref(),
            self.tool_call_id.as_deref(),
        )
    }

    /// Renders the message as one line: `[role] content`, or `[tool:<name>] content`.
    pub(crate) fn render(&self) -> String {
        match (self.role, self.tool_name.as_deref()) {
            (CompactRole::Tool, Some(name)) if !name.is_empty() => {
                format!("[tool:{name}] {}", self.content)
            }
            (role, _) => format!("[{}] {}", role.as_str(), self.content),
        }
    }
}

/// Byte weight of one message, shared by [`TranscriptMessage::byte_len`] and the summary guard.
fn message_weight(
    role: CompactRole,
    content: &str,
    tool_name: Option<&str>,
    tool_call_id: Option<&str>,
) -> usize {
    content.len()
        + role.as_str().len()
        + tool_name.map_or(0, str::len)
        + tool_call_id.map_or(0, str::len)
}

/// How much of an inherited transcript [`compact_transcript`] keeps verbatim.
///
/// Defaults match the contract: system messages are kept, the last six non-system messages are
/// kept, a tool result longer than 2 048 bytes is cut, and a cut result ends with
/// [`DEFAULT_TRUNCATED_SUFFIX`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct CompactPolicy {
    /// Whether system messages survive at the front of the transcript.
    pub keep_system: bool,
    /// How many trailing non-system messages are kept verbatim.
    pub keep_last_messages: usize,
    /// Byte budget of one kept tool result. Applies to [`CompactRole::Tool`] only.
    pub max_tool_result_bytes: usize,
    /// Appended to a tool result that was cut to [`CompactPolicy::max_tool_result_bytes`].
    pub truncated_suffix: String,
}

impl Default for CompactPolicy {
    fn default() -> Self {
        Self {
            keep_system: true,
            keep_last_messages: DEFAULT_KEEP_LAST_MESSAGES,
            max_tool_result_bytes: DEFAULT_MAX_TOOL_RESULT_BYTES,
            truncated_suffix: DEFAULT_TRUNCATED_SUFFIX.to_owned(),
        }
    }
}

impl CompactPolicy {
    /// Creates the default policy.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets [`CompactPolicy::keep_last_messages`].
    #[must_use]
    pub fn keeping_last(mut self, messages: usize) -> Self {
        self.keep_last_messages = messages;
        self
    }

    /// Sets [`CompactPolicy::max_tool_result_bytes`].
    #[must_use]
    pub fn with_tool_budget(mut self, bytes: usize) -> Self {
        self.max_tool_result_bytes = bytes;
        self
    }

    /// Turns [`CompactPolicy::keep_system`] off, so system messages are dropped and counted.
    #[must_use]
    pub fn without_system_messages(mut self) -> Self {
        self.keep_system = false;
        self
    }
}

/// The result of one compaction pass.
///
/// `original_bytes` and `compacted_bytes` are the summed [`TranscriptMessage::byte_len`] of the
/// input and of the produced list (summary included), which makes
/// `compacted_bytes <= original_bytes` a checkable invariant rather than a promise.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case", default)]
pub struct CompactedTranscript {
    /// Messages kept verbatim, plus the single summary message when one was emitted.
    pub messages: Vec<TranscriptMessage>,
    /// The summary line inserted into [`CompactedTranscript::messages`], when one was emitted.
    pub summary: Option<String>,
    /// Number of input messages that did not survive (system messages included when
    /// [`CompactPolicy::keep_system`] is off).
    pub dropped_messages: usize,
    /// Number of kept tool results that were shortened to the budget.
    pub truncated_tool_results: usize,
    /// Summed byte weight of the input transcript.
    pub original_bytes: usize,
    /// Summed byte weight of [`CompactedTranscript::messages`].
    pub compacted_bytes: usize,
}

impl CompactedTranscript {
    /// Returns `true` when nothing was dropped and nothing was truncated.
    #[must_use]
    pub fn is_lossless(&self) -> bool {
        self.dropped_messages == 0 && self.truncated_tool_results == 0
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

/// Compacts an inherited transcript into what a forked member should actually receive.
///
/// This is the deterministic, model-free approximation of Python's `compact_context`: instead of
/// asking an LLM to summarise the head of the conversation, it drops the head and reports what was
/// dropped, so the result is reproducible on any machine.
///
/// Rules, in order:
///
/// 1. System messages are kept verbatim, in order, at the front when
///    [`CompactPolicy::keep_system`]; otherwise they are dropped and counted in
///    [`CompactedTranscript::dropped_messages`].
/// 2. The last [`CompactPolicy::keep_last_messages`] non-system messages are kept verbatim, in
///    order. `0` keeps none of them, so only the summary (plus the kept system messages) remains.
/// 3. Every earlier non-system message is dropped and replaced by exactly one system message
///    whose content starts with [`SUMMARY_PREFIX`] and reports counts, e.g.
///    `[compacted] 12 earlier messages dropped (3 user, 5 assistant, 4 tool)`. Categories with a
///    zero count are omitted and the noun is singular for one message. There is no timestamp and
///    no prose, so the line is stable across runs. The summary goes *after* the kept system
///    messages and before the kept tail.
/// 4. A kept tool result longer than [`CompactPolicy::max_tool_result_bytes`] is cut on a
///    `char` boundary to at most that many bytes, and [`CompactPolicy::truncated_suffix`] is
///    appended. Only [`CompactRole::Tool`] messages are affected, so neither a long user message
///    nor the summary line can be truncated.
/// 5. Compaction never grows a transcript: a tool cut that would not shorten the content is
///    skipped, and the summary is omitted when it would cost more than the messages it replaces
///    (possible only when the dropped messages together weigh less than the summary line). The
///    dropped messages are still dropped and still counted in
///    [`CompactedTranscript::dropped_messages`] — only the summary line is missing, because
///    emitting it would have made the transcript longer than the input.
///
/// Compaction is idempotent under the same policy: the summary is a system message, so a second
/// pass finds nothing to drop and emits no second summary.
#[must_use]
pub fn compact_transcript(
    messages: &[TranscriptMessage],
    policy: &CompactPolicy,
) -> CompactedTranscript {
    let original_bytes = messages.iter().map(TranscriptMessage::byte_len).sum();

    let mut system_messages: Vec<&TranscriptMessage> = Vec::new();
    let mut other_messages: Vec<&TranscriptMessage> = Vec::new();
    for message in messages {
        if message.role == CompactRole::System {
            system_messages.push(message);
        } else {
            other_messages.push(message);
        }
    }

    let dropped_count = other_messages
        .len()
        .saturating_sub(policy.keep_last_messages);
    let (dropped, kept) = other_messages.split_at(dropped_count);

    let mut dropped_messages = dropped.len();
    if !policy.keep_system {
        dropped_messages += system_messages.len();
    }

    let mut compacted: Vec<TranscriptMessage> =
        Vec::with_capacity(system_messages.len() + kept.len() + usize::from(!dropped.is_empty()));
    if policy.keep_system {
        compacted.extend(system_messages.iter().map(|message| (*message).clone()));
    }
    // The summary belongs between the inherited system messages and the kept tail.
    let summary_index = compacted.len();
    compacted.extend(kept.iter().map(|message| (*message).clone()));

    let mut truncated_tool_results = 0;
    for message in &mut compacted {
        if message.role != CompactRole::Tool {
            continue;
        }
        if let Some(content) = shortened_tool_content(
            &message.content,
            policy.max_tool_result_bytes,
            &policy.truncated_suffix,
        ) {
            message.content = content;
            truncated_tool_results += 1;
        }
    }

    let kept_bytes: usize = compacted.iter().map(TranscriptMessage::byte_len).sum();
    let mut summary = if dropped.is_empty() {
        None
    } else {
        Some(summary_line(dropped))
    };
    if let Some(line) = &summary {
        let summary_bytes = message_weight(CompactRole::System, line, None, None);
        if kept_bytes + summary_bytes > original_bytes {
            summary = None;
        }
    }
    if let Some(line) = &summary {
        compacted.insert(summary_index, TranscriptMessage::system(line.clone()));
    }

    let compacted_bytes = compacted.iter().map(TranscriptMessage::byte_len).sum();
    CompactedTranscript {
        messages: compacted,
        summary,
        dropped_messages,
        truncated_tool_results,
        original_bytes,
        compacted_bytes,
    }
}

/// Builds the deterministic summary line for the dropped non-system messages.
///
/// Counts are fixed-order (`user`, `assistant`, `tool`) and zero categories are omitted, so the
/// same input always renders the same line.
fn summary_line(dropped: &[&TranscriptMessage]) -> String {
    let total = dropped.len();
    let noun = if total == 1 { "message" } else { "messages" };
    format!(
        "{SUMMARY_PREFIX}{total} earlier {noun} dropped ({})",
        role_count_parts(dropped)
    )
}

/// Renders the role counts of `messages` as `"3 user, 5 assistant"`.
///
/// The order is fixed (`user`, `assistant`, `tool`) and zero categories are omitted, so the same
/// input always renders the same text. Shared by [`summary_line`] and the template-driven summary
/// of [`super::compact_template`].
pub(crate) fn role_count_parts(messages: &[&TranscriptMessage]) -> String {
    let mut user = 0;
    let mut assistant = 0;
    let mut tool = 0;
    for message in messages {
        match message.role {
            CompactRole::User => user += 1,
            CompactRole::Assistant => assistant += 1,
            CompactRole::Tool => tool += 1,
            CompactRole::System => {}
        }
    }

    let mut parts: Vec<String> = Vec::with_capacity(3);
    for (count, label) in [(user, "user"), (assistant, "assistant"), (tool, "tool")] {
        if count > 0 {
            parts.push(format!("{count} {label}"));
        }
    }
    parts.join(", ")
}

/// Cuts `content` to `max_bytes` on a `char` boundary and appends `suffix`.
///
/// Returns `None` when nothing should change: the content already fits, or the cut would not make
/// it shorter (a tiny budget with a long suffix), which keeps compaction monotone.
pub(crate) fn shortened_tool_content(
    content: &str,
    max_bytes: usize,
    suffix: &str,
) -> Option<String> {
    if content.len() <= max_bytes {
        return None;
    }
    let mut end = max_bytes.min(content.len());
    while end > 0 && !content.is_char_boundary(end) {
        end -= 1;
    }
    let mut shortened = String::with_capacity(end + suffix.len());
    shortened.push_str(&content[..end]);
    shortened.push_str(suffix);
    (shortened.len() < content.len()).then_some(shortened)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
#[path = "fork_compact_tests.rs"]
mod fork_compact_tests;
