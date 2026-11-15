//! LLM-driven session-memory extraction over the shared model seam.
//!
//! Ports the intent of Python's
//! `processor/forked/compressor/session_memory_agent.py` and
//! `context/session_memory_manager.py::SessionMemoryUpdateAgent`: before a full
//! compaction can reinject session memory, the notes have to come from
//! somewhere. This module turns a message history into
//! [`SessionMemoryNotes`] either through a model call or, when no model is
//! reachable, through a deterministic offline extractor — and it always says
//! which of the two produced the notes.
//!
//! **Protocol.** This crate holds no tool runtime, so Python's `edit_file` tool
//! loop is replaced by a line protocol the model answers in plain text:
//!
//! ```text
//! ADD: <one line to append to the notes>
//! REPLACE: <text already in the notes> => <replacement text>
//! REMOVE: <text already in the notes>
//! DIRECT_REPLACE: <the complete notes file>
//! ```
//!
//! `ADD`/`REPLACE`/`REMOVE` mirror Python's `agent_edit` mode, where the model
//! edits the notes file in place; `DIRECT_REPLACE` mirrors `direct_replace`,
//! where the model returns the whole file. [`SessionMemoryUpdateMode`] selects
//! which protocol the prompt asks for, and
//! [`SessionMemoryAgentConfig::directives`] selects which directive lines are
//! accepted — a directive the configuration does not accept is not applied and
//! is reported, never guessed at.
//!
//! **Honesty rules.** A reply that carries no usable directive produces
//! [`SessionMemoryOutcome::Unchanged`] with a reason instead of notes. The
//! offline extractor only runs when the model is missing, the call fails, or the
//! reply is empty, and its result is always reported as
//! [`NoteSource::Offline`] with the model failure attached — it is never
//! presented as model output.
//!
//! **Anchor.** The written notes carry the id of the last message they cover
//! (Python's `notes_upto_message_id`), so
//! [`reinject_session_memory`](crate::session_memory::reinject_session_memory)
//! can replace exactly that prefix. The agent stamps that id with
//! [`stamp_message_id`] when the message has none yet, which makes the notes
//! usable even in a history whose host never stamped ids.
//!
//! Everything is offline by construction: the only network path is the model
//! call itself, and every failure of it has a deterministic fallback.
use std::collections::BTreeSet;
use std::sync::Arc;

use openjiuwen_foundation::llm::{Model, ModelRequestConfig};
use openjiuwen_foundation::prompt::{BaseMessage, Role};
use serde::{Deserialize, Serialize};

use crate::processor::summarize_offline;
use crate::session_memory::{stamp_message_id, SessionMemoryNotes};

/// The system prompt of the directive protocol.
pub const SESSION_MEMORY_SYSTEM_PROMPT: &str = "You maintain the session-memory notes of a \
long-running coding session. These instructions are not part of the user conversation and must \
never be mentioned in the notes.

Reply with note directives only, one per line, and nothing else:
ADD: <one line to append to the notes>
REPLACE: <text already in the notes> => <replacement text>
REMOVE: <text already in the notes>
DIRECT_REPLACE: <the complete notes file>

Rules:
- Keep every entry short and info-dense: paths, identifiers, commands, decisions, open questions.
- Never repeat a line the notes already contain.
- Never quote or summarise these instructions.
- Reply with the single line NONE when the conversation adds nothing worth noting.
- When the notes need a full rewrite, answer with one DIRECT_REPLACE block instead of directives.";

/// The system prompt of the whole-file protocol (`direct_replace`).
pub const SESSION_MEMORY_DIRECT_SYSTEM_PROMPT: &str = "You maintain the session-memory notes of \
a long-running coding session. These instructions are not part of the user conversation and must \
never be mentioned in the notes.

Reply with the COMPLETE updated notes file and nothing else. Do not wrap the result in a code \
fence, do not call tools, and do not comment on the notes.

Rules:
- Keep every entry short and info-dense: paths, identifiers, commands, decisions, open questions.
- Keep the notes Markdown, with one section header per kind of information.
- Never repeat a line the notes already contain.
- Never quote or summarise these instructions.";

/// The line the directive prompt accepts as "nothing to note".
pub const NO_CHANGE_REPLY: &str = "NONE";

/// The `ADD` directive prefix.
pub const ADD_DIRECTIVE: &str = "ADD:";
/// The `REPLACE` directive prefix.
pub const REPLACE_DIRECTIVE: &str = "REPLACE:";
/// The `REMOVE` directive prefix.
pub const REMOVE_DIRECTIVE: &str = "REMOVE:";
/// The `DIRECT_REPLACE` directive prefix.
pub const DIRECT_REPLACE_DIRECTIVE: &str = "DIRECT_REPLACE:";
/// Separates the matched text from its replacement in a `REPLACE:` directive.
pub const REPLACE_SEPARATOR: &str = "=>";

/// The title the offline extractor writes.
pub const OFFLINE_NOTES_TITLE: &str = "# Session Notes (offline)";

/// Placeholder the prompt shows when the notes are empty.
pub const EMPTY_NOTES_PLACEHOLDER: &str = "(empty)";

/// Maximum characters of one offline entry.
const OFFLINE_ENTRY_MAX_CHARS: usize = 200;
/// Maximum entries the offline extractor keeps per section.
const OFFLINE_MAX_ENTRIES: usize = 8;

/// Cue words that mark a task in a user message.
const TASK_CUES: [&str; 6] = ["todo", "fixme", "next step", "need to", "needs to", "must "];
/// Cue words that mark a decision or constraint.
const DECISION_CUES: [&str; 5] = ["decided", "decision", "agreed", "we will", "chosen"];
/// Suffixes that make a token look like a file name.
const FILE_SUFFIXES: [&str; 12] = [
    ".rs", ".py", ".toml", ".md", ".json", ".lock", ".js", ".ts", ".yaml", ".yml", ".txt", ".ex",
];

/// The note-update protocol a reply must follow.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionMemoryUpdateMode {
    /// `ADD:`/`REPLACE:`/`REMOVE:` directives applied to the existing notes
    /// (Python's `agent_edit`).
    #[default]
    Directives,
    /// The reply is the complete notes file (Python's `direct_replace`).
    DirectReplace,
}

/// One note-edit directive this agent understands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NoteDirective {
    /// Append a line the notes do not contain yet.
    Add,
    /// Replace the text matched by the directive's left-hand side.
    Replace,
    /// Drop the line containing the directive's target text.
    Remove,
    /// Replace the whole notes file with the directive's body.
    DirectReplace,
}

impl NoteDirective {
    /// The directive's reply prefix.
    #[must_use]
    pub const fn prefix(self) -> &'static str {
        match self {
            Self::Add => ADD_DIRECTIVE,
            Self::Replace => REPLACE_DIRECTIVE,
            Self::Remove => REMOVE_DIRECTIVE,
            Self::DirectReplace => DIRECT_REPLACE_DIRECTIVE,
        }
    }
}

/// Configuration of [`SessionMemoryAgent`].
///
/// Every field is `serde`-defaulted, so a configuration written by an older
/// build still loads.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct SessionMemoryAgentConfig {
    /// Whether the agent may run at all; a disabled agent reports
    /// [`SessionMemoryOutcome::Skipped`].
    pub enabled: bool,
    /// Character budget of the notes the agent writes; longer notes are
    /// truncated from the tail.
    pub max_notes_chars: usize,
    /// Maximum output tokens of one extraction call.
    pub model_max_tokens: u32,
    /// Shortest history the agent will extract from; a shorter one is skipped,
    /// because it cannot hold a session worth remembering.
    pub min_messages: usize,
    /// The protocol the prompt asks for.
    pub update_mode: SessionMemoryUpdateMode,
    /// Directives a reply may use; a directive outside this list is not applied
    /// and counts as unrecognised.
    pub directives: Vec<NoteDirective>,
    /// Whether a missing or failing model falls back to the offline extractor.
    ///
    /// With `false` such a case reports [`SessionMemoryOutcome::ModelFailed`]
    /// instead of writing offline notes.
    pub offline_fallback: bool,
    /// Character budget of the transcript the prompt carries.
    pub max_transcript_chars: usize,
}

impl Default for SessionMemoryAgentConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            // Python's `SessionMemoryCompressorConfig::max_notes_chars`.
            max_notes_chars: 120_000,
            model_max_tokens: 4_096,
            min_messages: 2,
            update_mode: SessionMemoryUpdateMode::Directives,
            directives: vec![
                NoteDirective::Add,
                NoteDirective::Replace,
                NoteDirective::Remove,
                NoteDirective::DirectReplace,
            ],
            offline_fallback: true,
            max_transcript_chars: 60_000,
        }
    }
}

impl SessionMemoryAgentConfig {
    /// Whether a reply may use `directive`.
    #[must_use]
    pub fn accepts(&self, directive: NoteDirective) -> bool {
        self.directives.contains(&directive)
    }
}

/// Which producer wrote a note set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NoteSource {
    /// A model reply parsed into directives.
    Model,
    /// The deterministic offline extractor.
    Offline,
}

impl NoteSource {
    /// The source's stable name, for logs and processor checkpoints.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Model => "model",
            Self::Offline => "offline",
        }
    }
}

/// What one extraction attempt produced.
///
/// The outcome is the honest report the caller (and the compaction branch) acts
/// on: only [`SessionMemoryOutcome::Written`] carries notes, and its
/// [`NoteSource`] says whether a model or the offline extractor wrote them.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum SessionMemoryOutcome {
    /// New notes were produced.
    Written {
        /// Which producer wrote them.
        source: NoteSource,
        /// The notes, anchored at the last message they cover.
        notes: SessionMemoryNotes,
        /// Number of directives that changed the notes.
        directives_applied: usize,
        /// The model failure that forced the offline fallback, when one did.
        model_failure: Option<String>,
    },
    /// The attempt ran but left the notes as they were.
    Unchanged {
        /// Why nothing was written.
        reason: String,
    },
    /// The model call failed and the offline fallback is disabled.
    ModelFailed {
        /// The model's error.
        reason: String,
    },
    /// The agent did not run.
    Skipped {
        /// Why it did not run.
        reason: String,
    },
}

impl SessionMemoryOutcome {
    /// Whether the attempt produced notes.
    #[must_use]
    pub fn is_written(&self) -> bool {
        matches!(self, Self::Written { .. })
    }

    /// The producer of the notes, `None` when no notes were written.
    #[must_use]
    pub fn source(&self) -> Option<NoteSource> {
        match self {
            Self::Written { source, .. } => Some(*source),
            Self::Unchanged { .. } | Self::ModelFailed { .. } | Self::Skipped { .. } => None,
        }
    }

    /// The notes, `None` when no notes were written.
    #[must_use]
    pub fn notes(&self) -> Option<&SessionMemoryNotes> {
        match self {
            Self::Written { notes, .. } => Some(notes),
            Self::Unchanged { .. } | Self::ModelFailed { .. } | Self::Skipped { .. } => None,
        }
    }

    /// Why no notes were written, `None` for a written outcome.
    #[must_use]
    pub fn reason(&self) -> Option<&str> {
        match self {
            Self::Written { .. } => None,
            Self::Unchanged { reason }
            | Self::ModelFailed { reason }
            | Self::Skipped { reason } => Some(reason),
        }
    }
}

/// What one reply did to the notes.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NoteEdit {
    /// The resulting notes, `None` when the reply carried no usable directive.
    pub notes: Option<String>,
    /// Directives that changed the notes.
    pub applied: usize,
    /// Directives that parsed but matched nothing in the current notes.
    pub unapplied: usize,
    /// Lines that were not directives this agent accepts.
    pub unrecognised: usize,
    /// Directives skipped because the notes already contained their text.
    pub duplicates: usize,
}

/// Truncates `text` to `max_chars` characters.
///
/// The head is kept and the cut respects character boundaries, so a multi-byte
/// character is never split.
#[must_use]
fn truncate_chars(text: &str, max_chars: usize) -> String {
    match text.char_indices().nth(max_chars) {
        Some((index, _)) => text[..index].to_owned(),
        None => text.to_owned(),
    }
}

/// Clips one offline entry to [`OFFLINE_ENTRY_MAX_CHARS`].
fn clip_entry(text: &str) -> String {
    let text = text.trim();
    if text.chars().count() <= OFFLINE_ENTRY_MAX_CHARS {
        return text.to_owned();
    }
    let mut clipped = truncate_chars(text, OFFLINE_ENTRY_MAX_CHARS);
    clipped.push('…');
    clipped
}

/// Appends `entry` to `entries` unless the section already holds it.
fn push_unique(entries: &mut Vec<String>, entry: String) {
    if entry.is_empty() || entries.iter().any(|seen| seen == &entry) {
        return;
    }
    if entries.len() < OFFLINE_MAX_ENTRIES {
        entries.push(entry);
    }
}

/// Whether `line` carries any of `cues`, case-insensitively.
fn has_cue(line: &str, cues: &[&str]) -> bool {
    let lowered = line.to_lowercase();
    cues.iter().any(|cue| lowered.contains(cue))
}

/// The file-looking tokens of one line.
fn file_tokens(line: &str) -> Vec<String> {
    line.split(|c: char| c.is_whitespace() || "()[]{}<>,;\"'`*!?".contains(c))
        .filter_map(|token| {
            let token = token.trim_matches(|c: char| matches!(c, '.' | ':' | ',' | ';' | '\''));
            if token.chars().count() < 3 || token.chars().count() > OFFLINE_ENTRY_MAX_CHARS {
                return None;
            }
            let lowered = token.to_lowercase();
            let looks_like_path = token.contains('/') || token.contains('\\');
            let looks_like_file = FILE_SUFFIXES.iter().any(|suffix| lowered.ends_with(suffix));
            (looks_like_path || looks_like_file).then(|| token.to_owned())
        })
        .collect()
}

/// Deterministically derives session notes from `messages`.
///
/// The extractor reads the transcript the way a reader would skim it: task cues
/// (`todo`, `fixme`, `next step`, `need to`, `needs to`, `must`) in user and
/// unknown-role messages become `## Tasks` entries, decision cues (`decided`,
/// `decision`, `agreed`, `we will`, `chosen`) in assistant messages become
/// `## Decisions` entries, a user line that ends with `?` becomes an
/// `## Open Questions` entry, and every file-looking token becomes a `## Files`
/// entry.
///
/// Every entry is one trimmed line clipped to `OFFLINE_ENTRY_MAX_CHARS`, each
/// section keeps at most `OFFLINE_MAX_ENTRIES` entries, and duplicates are
/// dropped. The result never exceeds `max_chars` characters and starts with
/// [`OFFLINE_NOTES_TITLE`]; a transcript with nothing to note yields an empty
/// string rather than a title-only document.
#[must_use]
pub fn extract_offline_notes(messages: &[BaseMessage], max_chars: usize) -> String {
    let mut tasks: Vec<String> = Vec::new();
    let mut files: Vec<String> = Vec::new();
    let mut decisions: Vec<String> = Vec::new();
    let mut questions: Vec<String> = Vec::new();

    for message in messages {
        let text = message.text().into_owned();
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            for token in file_tokens(line) {
                push_unique(&mut files, token);
            }
            match message.role {
                Role::User | Role::Other(_) => {
                    if has_cue(line, &TASK_CUES) {
                        push_unique(&mut tasks, clip_entry(line));
                    }
                    if line.ends_with('?') {
                        push_unique(&mut questions, clip_entry(line));
                    }
                }
                Role::Assistant => {
                    if has_cue(line, &DECISION_CUES) {
                        push_unique(&mut decisions, clip_entry(line));
                    }
                }
                Role::System | Role::Tool => {}
            }
        }
    }

    let sections: [(&str, &Vec<String>); 4] = [
        ("## Tasks", &tasks),
        ("## Files", &files),
        ("## Decisions", &decisions),
        ("## Open Questions", &questions),
    ];
    let mut out = String::new();
    for (heading, entries) in sections {
        if entries.is_empty() {
            continue;
        }
        if out.is_empty() {
            out.push_str(OFFLINE_NOTES_TITLE);
        }
        out.push_str("\n\n");
        out.push_str(heading);
        for entry in entries {
            out.push_str("\n- ");
            out.push_str(entry);
        }
    }
    truncate_chars(&out, max_chars)
}

/// Normalizes `lines` into one notes body.
///
/// Trailing whitespace is trimmed, blank lines are dropped, exact duplicates are
/// removed in first-appearance order, and the body is truncated to `max_chars`
/// characters.
fn normalize_notes(lines: Vec<String>, max_chars: usize) -> String {
    let mut seen: BTreeSet<String> = BTreeSet::new();
    let mut kept: Vec<String> = Vec::new();
    for line in lines {
        let line = line.trim_end();
        if line.trim().is_empty() {
            continue;
        }
        if seen.insert(line.to_owned()) {
            kept.push(line.to_owned());
        }
    }
    truncate_chars(&kept.join("\n"), max_chars)
}

/// Strips one surrounding Markdown code fence from `text`.
fn strip_code_fence(text: &str) -> String {
    let trimmed = text.trim();
    if !trimmed.starts_with("```") {
        return trimmed.to_owned();
    }
    let body = trimmed.trim_start_matches("```");
    let body = body.split_once('\n').map_or(body, |(_, rest)| rest);
    let body = body.trim_end();
    let body = body.strip_suffix("```").unwrap_or(body);
    body.trim_end().to_owned()
}

/// Strips a leading Markdown bullet marker, as a model may add one.
fn strip_bullet(line: &str) -> &str {
    let line = line.trim_start_matches(['-', '*']).trim_start();
    line
}

/// Splits `line` into its directive and body, when it starts with a prefix.
fn directive_kind(line: &str) -> Option<(NoteDirective, &str)> {
    for directive in [
        NoteDirective::DirectReplace,
        NoteDirective::Replace,
        NoteDirective::Remove,
        NoteDirective::Add,
    ] {
        let prefix = directive.prefix();
        if let Some(body) = strip_prefix_ascii_case(line, prefix) {
            return Some((directive, body.trim()));
        }
    }
    None
}

/// Case-insensitive ASCII prefix strip that never splits a character.
fn strip_prefix_ascii_case<'a>(line: &'a str, prefix: &str) -> Option<&'a str> {
    let head = line.as_bytes().get(..prefix.len())?;
    if !head.eq_ignore_ascii_case(prefix.as_bytes()) {
        return None;
    }
    line.get(prefix.len()..)
}

/// Applies a model `reply` to the current `notes`.
///
/// The reply is read line by line, after stripping a leading Markdown bullet and
/// surrounding whitespace:
///
/// - `ADD: <text>` appends `<text>` as one line unless the notes already contain
///   that exact line;
/// - `REPLACE: <old> => <new>` replaces every occurrence of `<old>` inside the
///   first line that contains it, and drops the line when the replacement leaves
///   it blank;
/// - `REMOVE: <text>` drops the first line that contains `<text>`;
/// - `DIRECT_REPLACE: <body>` replaces the whole notes body with `<body>` plus
///   every following line of the reply, so a multi-line notes file works.
///
/// In [`SessionMemoryUpdateMode::DirectReplace`] the whole reply is the notes
/// file, and the directives are not parsed at all. A surrounding Markdown code
/// fence and the [`NO_CHANGE_REPLY`] line are accepted in both modes.
///
/// A directive the configuration does not accept counts as unrecognised, an
/// empty body counts as unrecognised, and a directive whose target is absent
/// counts as unapplied. When nothing changed,
/// [`NoteEdit::notes`] is `None`: an unparsable reply never yields notes.
#[must_use]
pub fn apply_note_directives(
    notes: &str,
    reply: &str,
    config: &SessionMemoryAgentConfig,
) -> NoteEdit {
    let mut edit = NoteEdit::default();
    let stripped = strip_code_fence(reply);
    if stripped.eq_ignore_ascii_case(NO_CHANGE_REPLY) {
        return edit;
    }
    if config.update_mode == SessionMemoryUpdateMode::DirectReplace {
        if stripped.is_empty() {
            return edit;
        }
        edit.notes = Some(truncate_chars(&stripped, config.max_notes_chars));
        edit.applied = 1;
        return edit;
    }

    let mut lines: Vec<String> = notes
        .lines()
        .map(str::trim_end)
        .filter(|line| !line.trim().is_empty())
        .map(str::to_owned)
        .collect();
    let raw: Vec<&str> = stripped.lines().collect();
    let mut index = 0usize;
    while index < raw.len() {
        let line = strip_bullet(raw[index].trim());
        index += 1;
        if line.is_empty() {
            continue;
        }
        let Some((directive, body)) = directive_kind(line) else {
            edit.unrecognised += 1;
            continue;
        };
        if !config.accepts(directive) {
            edit.unrecognised += 1;
            continue;
        }
        match directive {
            NoteDirective::DirectReplace => {
                let body = direct_replace_body(body, &raw[index..]);
                if body.is_empty() {
                    edit.unrecognised += 1;
                    continue;
                }
                edit.notes = Some(truncate_chars(&body, config.max_notes_chars));
                edit.applied += 1;
                return edit;
            }
            NoteDirective::Add => {
                if body.is_empty() {
                    edit.unrecognised += 1;
                    continue;
                }
                if lines.iter().any(|existing| existing.trim() == body) {
                    edit.duplicates += 1;
                    continue;
                }
                lines.push(body.to_owned());
                edit.applied += 1;
            }
            NoteDirective::Replace => {
                let Some((old, new)) = body.split_once(REPLACE_SEPARATOR) else {
                    edit.unrecognised += 1;
                    continue;
                };
                let old = old.trim();
                if old.is_empty() {
                    edit.unrecognised += 1;
                    continue;
                }
                let Some(position) = lines.iter().position(|line| line.contains(old)) else {
                    edit.unapplied += 1;
                    continue;
                };
                let replaced = lines[position].replace(old, new.trim());
                if replaced.trim().is_empty() {
                    lines.remove(position);
                } else {
                    lines[position] = replaced;
                }
                edit.applied += 1;
            }
            NoteDirective::Remove => {
                if body.is_empty() {
                    edit.unrecognised += 1;
                    continue;
                }
                let Some(position) = lines.iter().position(|line| line.contains(body)) else {
                    edit.unapplied += 1;
                    continue;
                };
                lines.remove(position);
                edit.applied += 1;
            }
        }
    }
    if edit.applied > 0 {
        edit.notes = Some(normalize_notes(lines, config.max_notes_chars));
    }
    edit
}

/// Merges the `derived` entries into the `existing` notes.///
/// Lines of `existing` come first in their original order, then every `derived`
/// line the result does not contain yet, so an offline refresh never discards
/// what an earlier pass wrote.
fn merge_notes(existing: &str, derived: &str, max_chars: usize) -> String {
    let mut lines: Vec<String> = existing.lines().map(str::to_owned).collect();
    lines.extend(derived.lines().map(str::to_owned));
    normalize_notes(lines, max_chars)
}

/// Renders the user prompt: the current notes plus the transcript.
///
/// The layout is fixed so a deterministic model (and a test) can parse it:
/// a one-line instruction, a `<current_notes>` block, a `<conversation>` block,
/// then the closing instruction.
#[must_use]
pub fn render_session_memory_prompt(notes: &str, transcript: &str) -> String {
    let notes = if notes.trim().is_empty() {
        EMPTY_NOTES_PLACEHOLDER
    } else {
        notes.trim()
    };
    format!(
        "These instructions are not part of the user conversation.\n\n\
<current_notes>\n{notes}\n</current_notes>\n\n\
<conversation>\n{transcript}\n</conversation>\n\n\
Update the notes from the conversation above and reply as instructed."
    )
}

/// Extracts session-memory notes over the shared model seam.
///
/// The agent holds no model of its own: [`SessionMemoryAgent::update`] takes the
/// model the caller resolved (the compaction processor passes the per-request
/// model before its own), so one agent instance serves every context.
///
/// ```text
/// update(messages, &current_notes, Some(&model))
///   ├─ model configured  → invoke → parse directives → Written { source: Model }
///   ├─ model failed      → offline extractor        → Written { source: Offline }
///   ├─ reply unusable    → Unchanged { reason }
///   └─ history too short → Skipped { reason }
/// ```
#[derive(Debug, Clone, Default)]
pub struct SessionMemoryAgent {
    /// The agent configuration.
    config: SessionMemoryAgentConfig,
}

impl SessionMemoryAgent {
    /// Creates an agent with `config`.
    #[must_use]
    pub fn new(config: SessionMemoryAgentConfig) -> Self {
        Self { config }
    }

    /// The agent configuration.
    #[must_use]
    pub fn config(&self) -> &SessionMemoryAgentConfig {
        &self.config
    }

    /// The `[system, user]` prompt pair for one extraction.
    ///
    /// The transcript is the deterministic `[i] role: text` rendering of the
    /// history, truncated to
    /// [`SessionMemoryAgentConfig::max_transcript_chars`], so the prompt is
    /// bounded and reproducible.
    #[must_use]
    pub fn render_prompt(
        &self,
        current: &SessionMemoryNotes,
        messages: &[BaseMessage],
    ) -> Vec<BaseMessage> {
        let transcript = summarize_offline(messages, self.config.max_transcript_chars);
        let system = match self.config.update_mode {
            SessionMemoryUpdateMode::Directives => SESSION_MEMORY_SYSTEM_PROMPT,
            SessionMemoryUpdateMode::DirectReplace => SESSION_MEMORY_DIRECT_SYSTEM_PROMPT,
        };
        vec![
            BaseMessage::system(system),
            BaseMessage::user(render_session_memory_prompt(&current.notes, &transcript)),
        ]
    }

    /// Extracts the notes for `messages`, updating `current`.
    ///
    /// The history is read-only apart from one side effect: the last message
    /// gets a stable [`CONTEXT_MESSAGE_ID_KEY`](crate::session_memory::CONTEXT_MESSAGE_ID_KEY)
    /// when it has none, because the written notes anchor on it. Storing the
    /// result is the caller's job — this method performs no I/O beyond the model
    /// call.
    pub async fn update(
        &self,
        messages: &mut [BaseMessage],
        current: &SessionMemoryNotes,
        model: Option<&Arc<dyn Model>>,
    ) -> SessionMemoryOutcome {
        if !self.config.enabled {
            return SessionMemoryOutcome::Skipped {
                reason: "the session-memory agent is disabled by configuration".to_owned(),
            };
        }
        if messages.len() < self.config.min_messages {
            return SessionMemoryOutcome::Skipped {
                reason: format!(
                    "the history holds {} message(s), fewer than the {} needed to extract notes",
                    messages.len(),
                    self.config.min_messages
                ),
            };
        }
        // The notes cover everything the transcript shows, so the anchor is the
        // last message; stamping it here keeps the notes usable in a history
        // whose host never wrote message ids.
        let anchor = messages.last_mut().map(stamp_message_id);
        let Some(model) = model else {
            let failure = "no model is configured for session-memory extraction".to_owned();
            return self.offline_outcome(messages, current, anchor, Some(failure));
        };

        let prompt = self.render_prompt(current, messages);
        let mut request = ModelRequestConfig::default()
            .with_model_name(model.model_name())
            .with_max_tokens(self.config.model_max_tokens);
        // No-op outside a `with_operation` scope; inside one it tags the request
        // with the purpose and operation id that correlate the provider call.
        crate::telemetry::stamp_current_operation(&mut request);
        let response = match model.invoke(&prompt, &request).await {
            Ok(response) => response,
            Err(error) => {
                let failure = format!("session-memory model call failed: {error}");
                tracing::warn!(error = %error, "session-memory extraction failed; trying the offline extractor");
                return self.offline_outcome(messages, current, anchor, Some(failure));
            }
        };
        // Usage is observability only: report it before looking at the text,
        // because a call that answered nothing still cost tokens.
        crate::processor::record_model_usage(&response.usage, model.model_name());
        let reply = response
            .message()
            .map(|message| message.text().into_owned())
            .unwrap_or_default();
        if reply.trim().is_empty() {
            let failure = "the session-memory model returned an empty reply".to_owned();
            tracing::warn!(
                "session-memory extraction returned no text; trying the offline extractor"
            );
            return self.offline_outcome(messages, current, anchor, Some(failure));
        }

        let edit = apply_note_directives(&current.notes, &reply, &self.config);
        let Some(notes) = edit.notes else {
            let reason = if edit.unrecognised > 0 || edit.unapplied > 0 || edit.duplicates > 0 {
                format!(
                    "the session-memory reply carried no applicable directive \
                     ({} unrecognised, {} unmatched, {} duplicate)",
                    edit.unrecognised, edit.unapplied, edit.duplicates
                )
            } else {
                "the session-memory reply carried no usable note directive".to_owned()
            };
            tracing::warn!(reason, "session-memory extraction produced no notes");
            return SessionMemoryOutcome::Unchanged { reason };
        };
        // The directive path already normalized its lines; a whole-file reply is
        // kept as the model wrote it, so blank lines and headings survive.
        let notes = notes.trim_end().to_owned();
        if notes.trim() == current.notes.trim() {
            return SessionMemoryOutcome::Unchanged {
                reason: "the session-memory reply left the notes unchanged".to_owned(),
            };
        }
        SessionMemoryOutcome::Written {
            source: NoteSource::Model,
            notes: SessionMemoryNotes {
                notes,
                anchor_message_id: anchor,
            },
            directives_applied: edit.applied,
            model_failure: None,
        }
    }

    /// The offline outcome for a missing or failed model.
    ///
    /// The derived entries are merged into `current`, so the fallback refreshes
    /// the notes instead of discarding them. `model_failure` is carried on a
    /// written outcome, and folded into the reason when nothing new was derived —
    /// either way the failure is never hidden.
    fn offline_outcome(
        &self,
        messages: &[BaseMessage],
        current: &SessionMemoryNotes,
        anchor: Option<String>,
        model_failure: Option<String>,
    ) -> SessionMemoryOutcome {
        if !self.config.offline_fallback {
            return SessionMemoryOutcome::ModelFailed {
                reason: model_failure.unwrap_or_else(|| {
                    "the offline session-memory fallback is disabled".to_owned()
                }),
            };
        }
        let derived = extract_offline_notes(messages, self.config.max_notes_chars);
        let merged = merge_notes(&current.notes, &derived, self.config.max_notes_chars);
        let existing = normalize_notes(
            current.notes.lines().map(str::to_owned).collect(),
            usize::MAX,
        );
        if merged == existing {
            let detail = model_failure.unwrap_or_else(|| "no model was used".to_owned());
            return SessionMemoryOutcome::Unchanged {
                reason: format!("{detail}; the offline extractor found nothing new to note"),
            };
        }
        SessionMemoryOutcome::Written {
            source: NoteSource::Offline,
            notes: SessionMemoryNotes {
                notes: merged,
                anchor_message_id: anchor,
            },
            directives_applied: 0,
            model_failure,
        }
    }
}

/// Builds a whole-file reply from a directive body and the reply lines after it.
fn direct_replace_body(body: &str, rest: &[&str]) -> String {
    let mut out = body.to_owned();
    for line in rest {
        out.push('\n');
        out.push_str(line);
    }
    strip_code_fence(&out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use openjiuwen_foundation::prompt::ToolCall;

    fn notes(text: &str) -> SessionMemoryNotes {
        SessionMemoryNotes {
            notes: text.to_owned(),
            anchor_message_id: None,
        }
    }

    #[test]
    fn config_round_trips_and_defaults_are_stable() {
        let config = SessionMemoryAgentConfig::default();
        assert_eq!(
            serde_json::from_str::<SessionMemoryAgentConfig>("{}").unwrap(),
            config
        );
        let json = serde_json::to_string(&config).unwrap();
        assert_eq!(
            serde_json::from_str::<SessionMemoryAgentConfig>(&json).unwrap(),
            config
        );
        assert!(config.accepts(NoteDirective::Remove));
        assert_eq!(
            serde_json::to_string(&SessionMemoryUpdateMode::DirectReplace).unwrap(),
            "\"direct_replace\""
        );
    }

    #[test]
    fn add_replace_remove_apply_in_reply_order() {
        let config = SessionMemoryAgentConfig::default();
        let edit = apply_note_directives(
            "# Notes\n- keep me\n- drop me",
            "ADD: - new task\nREPLACE: keep me => keep this\nREMOVE: drop me\nADD: - keep this",
            &config,
        );
        assert_eq!(edit.applied, 3);
        assert_eq!(edit.duplicates, 1);
        assert_eq!(
            edit.notes.as_deref(),
            Some("# Notes\n- keep this\n- new task")
        );
    }

    #[test]
    fn direct_replace_swallows_the_rest_of_the_reply() {
        let config = SessionMemoryAgentConfig::default();
        let edit = apply_note_directives(
            "- old",
            "DIRECT_REPLACE: # Notes\n\n## Tasks\n- rewrite everything",
            &config,
        );
        assert_eq!(edit.applied, 1);
        assert_eq!(
            edit.notes.as_deref(),
            Some("# Notes\n\n## Tasks\n- rewrite everything")
        );
    }

    #[test]
    fn direct_replace_mode_takes_the_whole_reply() {
        let config = SessionMemoryAgentConfig {
            update_mode: SessionMemoryUpdateMode::DirectReplace,
            ..SessionMemoryAgentConfig::default()
        };
        let edit = apply_note_directives("- old", "```\n# Notes\n- new\n```", &config);
        assert_eq!(edit.notes.as_deref(), Some("# Notes\n- new"));
        assert_eq!(edit.applied, 1);
        // An empty reply never yields notes.
        assert_eq!(apply_note_directives("- old", "   ", &config).notes, None);
    }

    #[test]
    fn unusable_replies_never_produce_notes() {
        let config = SessionMemoryAgentConfig::default();
        let edit =
            apply_note_directives("- old", "Sure! Here is my summary of the session.", &config);
        assert_eq!(edit.notes, None);
        assert_eq!(edit.unrecognised, 1);
        // A directive whose target is absent is reported, not invented.
        let edit = apply_note_directives("- old", "REMOVE: nothing alike", &config);
        assert_eq!(edit.notes, None);
        assert_eq!(edit.unapplied, 1);
        // A directive the configuration does not accept is unrecognised.
        let strict = SessionMemoryAgentConfig {
            directives: vec![NoteDirective::Add],
            ..SessionMemoryAgentConfig::default()
        };
        let edit = apply_note_directives("- old", "REMOVE: old", &strict);
        assert_eq!(edit.notes, None);
        assert_eq!(edit.unrecognised, 1);
        // The no-change reply is accepted and writes nothing.
        let edit = apply_note_directives("- old", "NONE", &config);
        assert_eq!(edit.notes, None);
        assert_eq!(edit.unrecognised, 0);
        // A bullet in front of a directive is tolerated.
        let edit = apply_note_directives("- old", "- ADD: - new", &config);
        assert_eq!(edit.notes.as_deref(), Some("- old\n- new"));
    }

    #[test]
    fn notes_are_deduplicated_and_truncated() {
        let config = SessionMemoryAgentConfig {
            max_notes_chars: 12,
            ..SessionMemoryAgentConfig::default()
        };
        let edit = apply_note_directives("- a\n- a", "ADD: - b\nADD: - c", &config);
        let notes = edit.notes.expect("the reply changed the notes");
        // The duplicate line is dropped, so the body is 11 characters and fits.
        assert_eq!(notes, "- a\n- b\n- c");
        // A tighter budget truncates the normalized body.
        let tight = SessionMemoryAgentConfig {
            max_notes_chars: 8,
            ..SessionMemoryAgentConfig::default()
        };
        let edit = apply_note_directives("- a", "ADD: - bcd", &tight);
        assert_eq!(edit.notes.as_deref(), Some("- a\n- bc"));
    }

    #[test]
    fn the_offline_extractor_reads_tasks_files_decisions_and_questions() {
        let messages = vec![
            BaseMessage::user("TODO: port the parser to src/parser.rs"),
            BaseMessage::user("should we keep the legacy path?"),
            BaseMessage::assistant("We decided to keep the legacy path in src/legacy.rs."),
            BaseMessage::tool("call_1", "docs/README.md updated"),
        ];
        let notes = extract_offline_notes(&messages, 10_000);
        assert!(notes.starts_with(OFFLINE_NOTES_TITLE));
        assert!(notes.contains("## Tasks\n- TODO: port the parser to src/parser.rs"));
        assert!(notes.contains("## Files\n- src/parser.rs\n- src/legacy.rs\n- docs/README.md"));
        assert!(
            notes.contains("## Decisions\n- We decided to keep the legacy path in src/legacy.rs.")
        );
        assert!(notes.contains("## Open Questions\n- should we keep the legacy path?"));
        // A transcript with nothing to note yields no notes at all.
        assert_eq!(
            extract_offline_notes(&[BaseMessage::assistant("ok")], 10_000),
            ""
        );
        // Tool-call arguments are part of the transcript, not of the offline notes.
        let call = BaseMessage::assistant("").with_tool_calls(vec![ToolCall::new("grep")]);
        assert_eq!(extract_offline_notes(&[call], 10_000), "");
    }

    #[test]
    fn merge_keeps_existing_notes_ahead_of_derived_ones() {
        let merged = merge_notes("- keep\n- a", "- b\n- keep", 10_000);
        assert_eq!(merged, "- keep\n- a\n- b");
        // A derived section that repeats an existing line adds nothing.
        assert_eq!(merge_notes("- a", "- a", 10_000), "- a");
    }

    #[test]
    fn the_prompt_carries_the_notes_and_the_transcript() {
        let agent = SessionMemoryAgent::default();
        let prompt = agent.render_prompt(
            &notes("- a"),
            &[
                BaseMessage::system("persona"),
                BaseMessage::user("do the thing"),
            ],
        );
        assert_eq!(prompt.len(), 2);
        assert_eq!(prompt[0].role, Role::System);
        assert_eq!(prompt[0].text(), SESSION_MEMORY_SYSTEM_PROMPT);
        let user = prompt[1].text().into_owned();
        assert!(user.contains("<current_notes>\n- a\n</current_notes>"));
        assert!(user.contains("<conversation>\n[1] system: persona\n[2] user: do the thing\n"));
        // Empty notes are shown through the placeholder.
        let prompt = agent.render_prompt(&notes(""), &[BaseMessage::user("hi")]);
        assert!(prompt[1].text().contains(EMPTY_NOTES_PLACEHOLDER));
        // The direct protocol uses the other system prompt.
        let direct = SessionMemoryAgent::new(SessionMemoryAgentConfig {
            update_mode: SessionMemoryUpdateMode::DirectReplace,
            ..SessionMemoryAgentConfig::default()
        });
        let prompt = direct.render_prompt(&notes(""), &[BaseMessage::user("hi")]);
        assert_eq!(prompt[0].text(), SESSION_MEMORY_DIRECT_SYSTEM_PROMPT);
    }
}
