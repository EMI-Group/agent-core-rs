//! Task extraction and the JSON-schema memory-block plan.
//!
//! A memory block used to carry only a free-text summary. The adaptive path
//! additionally extracts the durable state of the span — pending tasks,
//! decisions, constraints, touched files and open questions — so a resumed
//! session can recover *what is still to do* without re-reading the transcript.
//!
//! [`MemoryBlockPlan`] is that state. It is produced either from a model reply
//! (see [`MemoryBlockPlan::from_model_reply`], driven by
//! [`MemoryBlockPlan::json_schema`]) or deterministically from the span (see
//! [`MemoryBlockPlan::from_messages`], which scans TODO/next-step/decision/
//! constraint markers and tool-result file paths and falls back to the offline
//! transcript summary when nothing is found).
//!
//! Rendering ([`MemoryBlockPlan::render_body`]) keeps the existing memory-block
//! envelope: the plan becomes the block's `Summary:` body, followed by one
//! `label:` section per non-empty list, one `- item` bullet per entry. Sections
//! are emitted in the fixed order `tasks`, `decisions`, `constraints`, `files`,
//! `open_questions`, so the block is stable across runs and a Python session can
//! read it as plain text.
use std::collections::BTreeSet;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::prompt::BaseMessage;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::prompt::extract_json_object;
use crate::processor::compressor::summarize_offline;

/// Maximum number of entries kept per plan section.
const MAX_PLAN_ITEMS: usize = 8;

/// Maximum characters of one rendered plan entry.
const MAX_PLAN_ITEM_CHARS: usize = 240;

/// Characters of the offline transcript summary used as the plan fallback.
const PLAN_SUMMARY_CHARS: usize = 2_000;

/// Lines carrying one of these markers become a task.
const TASK_MARKERS: &[&str] = &[
    "todo",
    "to-do",
    "next step",
    "next-step",
    "remaining",
    "still need",
    "follow-up",
    "followup",
    "action item",
];

/// Lines carrying one of these markers become a decision.
const DECISION_MARKERS: &[&str] = &[
    "decided",
    "decision",
    "we will",
    "we chose",
    "chose ",
    "settled on",
];

/// Lines carrying one of these markers become a constraint.
const CONSTRAINT_MARKERS: &[&str] = &[
    "must ",
    "do not ",
    "don't ",
    "never ",
    "constraint",
    "requirement",
    "only ",
];

/// Lines carrying one of these markers become an open question.
const QUESTION_MARKERS: &[&str] = &[
    "open question",
    "unclear",
    "not sure",
    "unknown whether",
    "needs confirmation",
];

/// File extensions a path token must carry to count as a touched file.
const FILE_EXTENSIONS: &[&str] = &[
    ".rs", ".py", ".md", ".json", ".toml", ".yaml", ".yml", ".txt", ".ts", ".tsx", ".js", ".jsx",
    ".go", ".java", ".c", ".h", ".cc", ".cpp", ".hpp", ".sh", ".sql", ".csv", ".html", ".css",
    ".ex", ".exs", ".rb", ".php",
];

/// The durable state a memory block carries.
///
/// Every field is optional on the wire: [`MemoryBlockPlan::default`] is an empty
/// plan, and missing fields deserialize to empty lists, so a plan written by an
/// older or newer producer still loads.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct MemoryBlockPlan {
    /// The free-text summary of the replaced span.
    pub summary: String,
    /// Work that is still to do.
    pub tasks: Vec<String>,
    /// Decisions already taken.
    pub decisions: Vec<String>,
    /// Constraints the work must respect.
    pub constraints: Vec<String>,
    /// Files, directories or artifacts the span touched.
    pub files: Vec<String>,
    /// Questions that remain unanswered.
    pub open_questions: Vec<String>,
}

impl MemoryBlockPlan {
    /// Whether the plan carries no information at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.summary.trim().is_empty()
            && self.tasks.is_empty()
            && self.decisions.is_empty()
            && self.constraints.is_empty()
            && self.files.is_empty()
            && self.open_questions.is_empty()
    }

    /// The JSON schema of a memory-block plan, as sent in the model prompt.
    ///
    /// Every property is optional; `additionalProperties` stays open so a model
    /// reply may carry extra keys without failing validation.
    #[must_use]
    pub fn json_schema() -> Value {
        let string_array = |description: &str| {
            serde_json::json!({
                "type": "array",
                "description": description,
                "items": {"type": "string"},
            })
        };
        serde_json::json!({
            "type": "object",
            "description": "Durable state extracted from the compressed span.",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Compact summary of the span; keeps identifiers, paths and open questions.",
                },
                "tasks": string_array("Work that is still to do."),
                "decisions": string_array("Decisions already taken."),
                "constraints": string_array("Constraints the work must respect."),
                "files": string_array("Files, directories or artifacts the span touched."),
                "open_questions": string_array("Questions that remain unanswered."),
            },
            "required": [],
            "additionalProperties": true,
        })
    }

    /// Parses a memory-block plan out of `text`, which may be a model reply.
    ///
    /// Parsing is lenient: fenced code blocks, leading prose and unknown keys are
    /// tolerated, arrays of strings and arrays of objects (read through `text`,
    /// `summary`, `description`, `task`, `decision`, `path`, `file` or `name`)
    /// are accepted, and duplicate entries are dropped. A reply with no JSON
    /// object at all is read as the plain-text summary.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for an empty reply, for malformed JSON, and
    /// for a JSON object that carries none of the plan fields.
    pub fn from_model_reply(text: &str) -> Result<Self> {
        let text = text.trim();
        if text.is_empty() {
            return Err(Error::Validation(
                "cannot build a memory-block plan from an empty model reply".to_owned(),
            ));
        }
        let Some(object) = extract_json_object(text) else {
            return Ok(Self {
                summary: text.to_owned(),
                ..Self::default()
            });
        };
        let value: Value = serde_json::from_str(object).map_err(|error| {
            Error::Validation(format!("the memory-block plan is not valid JSON: {error}"))
        })?;
        let object = value.as_object().ok_or_else(|| {
            Error::Validation("the memory-block plan is not a JSON object".to_owned())
        })?;
        let plan = Self::from_object(object);
        if plan.is_empty() {
            return Err(Error::Validation(
                "the memory-block plan carries none of summary, tasks, decisions, constraints, \
                 files or open_questions"
                    .to_owned(),
            ));
        }
        Ok(plan)
    }

    /// Reads a plan out of a JSON object, ignoring every unrecognised key.
    ///
    /// Unlike [`MemoryBlockPlan::from_model_reply`] this never fails: an object
    /// without plan fields simply yields an empty plan, which is how a strategy
    /// reply may omit the task state.
    #[must_use]
    pub(crate) fn from_object(object: &serde_json::Map<String, Value>) -> Self {
        let summary = ["summary", "state_snapshot", "overview"]
            .iter()
            .find_map(|key| object.get(*key).and_then(Value::as_str))
            .map(str::trim)
            .filter(|text| !text.is_empty())
            .map(str::to_owned)
            .unwrap_or_default();
        Self {
            summary,
            tasks: list_from_object(object, &["tasks", "todo", "todos", "next_steps"]),
            decisions: list_from_object(object, &["decisions", "key_decisions"]),
            constraints: list_from_object(object, &["constraints", "requirements"]),
            files: list_from_object(object, &["files", "paths", "artifacts"]),
            open_questions: list_from_object(object, &["open_questions", "questions", "blockers"]),
        }
    }

    /// Derives a plan deterministically from the messages of a span.
    ///
    /// Each non-empty line of every message is classified by its markers: a task
    /// line mentions a TODO/next step/remaining work, a decision line mentions a
    /// decision, a constraint line mentions a must/never/requirement, and a
    /// question line ends with `?` or mentions an open question. Path-like tokens
    /// become `files` entries. Entries are trimmed to one line, capped at
    /// `MAX_PLAN_ITEMS` per section and deduplicated.
    ///
    /// `task_extraction = false` yields an empty plan. When nothing is found the
    /// plan's `summary` becomes the offline transcript summary (capped at
    /// `PLAN_SUMMARY_CHARS`), so the caller always has a usable body.
    #[must_use]
    pub fn from_messages(messages: &[BaseMessage], task_extraction: bool) -> Self {
        if !task_extraction {
            return Self::default();
        }
        let mut plan = Self::default();
        let mut files: BTreeSet<String> = BTreeSet::new();
        for message in messages {
            let text = message.text();
            for line in text.lines() {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                collect_files(line, &mut files);
                classify_line(line, &mut plan);
            }
        }
        for file in files {
            push_item(&mut plan.files, &file);
        }
        if plan.is_empty() {
            plan.summary = summarize_offline(messages, PLAN_SUMMARY_CHARS);
        }
        plan
    }

    /// Copies every non-empty field of `other` into the matching empty field.
    ///
    /// Used to complete a model-supplied plan with deterministic extraction:
    /// whatever the model already reported wins, and gaps are filled from the
    /// span.
    pub fn fill_missing(&mut self, other: &Self) {
        if self.summary.trim().is_empty() {
            self.summary.clone_from(&other.summary);
        }
        for (mine, theirs) in [
            (&mut self.tasks, &other.tasks),
            (&mut self.decisions, &other.decisions),
            (&mut self.constraints, &other.constraints),
            (&mut self.files, &other.files),
            (&mut self.open_questions, &other.open_questions),
        ] {
            if mine.is_empty() {
                mine.clone_from(theirs);
            }
        }
    }

    /// Renders the structured sections of the plan, without the summary.
    ///
    /// One `label:` line per non-empty section (in the fixed order `tasks`,
    /// `decisions`, `constraints`, `files`, `open_questions`) followed by one
    /// `- item` line per entry. Sections are separated by a blank line; an empty
    /// plan renders as an empty string.
    #[must_use]
    pub fn render_sections(&self) -> String {
        let mut sections: Vec<String> = Vec::new();
        for (label, items) in [
            ("tasks", &self.tasks),
            ("decisions", &self.decisions),
            ("constraints", &self.constraints),
            ("files", &self.files),
            ("open_questions", &self.open_questions),
        ] {
            let bullets: Vec<String> = items
                .iter()
                .filter_map(|item| clean_item(item))
                .map(|item| format!("- {item}"))
                .collect();
            if bullets.is_empty() {
                continue;
            }
            sections.push(format!("{label}:\n{}", bullets.join("\n")));
        }
        sections.join("\n\n")
    }

    /// Renders the plan body: the summary, then the structured sections.
    ///
    /// The body is what [`MemoryBlockKind::render`] wraps in the memory-block
    /// envelope, so a block reads `Summary:` followed by the summary and the
    /// sections. Both parts are optional; an empty plan renders as an empty body.
    ///
    /// [`MemoryBlockKind::render`]: crate::processor::compressor::MemoryBlockKind::render
    #[must_use]
    pub fn render_body(&self) -> String {
        let mut parts: Vec<String> = Vec::new();
        let summary = self.summary.trim();
        if !summary.is_empty() {
            parts.push(summary.to_owned());
        }
        let sections = self.render_sections();
        if !sections.is_empty() {
            parts.push(sections);
        }
        parts.join("\n\n")
    }
}

/// Reads the first recognised list key of `object` into plan entries.
fn list_from_object(object: &serde_json::Map<String, Value>, keys: &[&str]) -> Vec<String> {
    let mut items: Vec<String> = Vec::new();
    for key in keys {
        let Some(value) = object.get(*key) else {
            continue;
        };
        for entry in entries(value) {
            push_item(&mut items, &entry);
        }
    }
    items
}

/// Flattens one JSON value into entry strings.
fn entries(value: &Value) -> Vec<String> {
    match value {
        Value::String(text) => vec![text.clone()],
        Value::Array(items) => items.iter().flat_map(entries).collect(),
        Value::Object(object) => [
            "text",
            "summary",
            "description",
            "task",
            "decision",
            "path",
            "file",
            "name",
        ]
        .iter()
        .find_map(|key| object.get(*key).and_then(Value::as_str))
        .map(|text| vec![text.to_owned()])
        .unwrap_or_default(),
        Value::Null | Value::Bool(_) | Value::Number(_) => Vec::new(),
    }
}

/// Pushes a cleaned entry, dropping duplicates and honouring the caps.
fn push_item(items: &mut Vec<String>, text: &str) {
    if items.len() >= MAX_PLAN_ITEMS {
        return;
    }
    let Some(item) = clean_item(text) else {
        return;
    };
    if items.iter().any(|seen| seen == &item) {
        return;
    }
    items.push(item);
}

/// Normalises one entry to a single line and truncates it.
fn clean_item(text: &str) -> Option<String> {
    let single = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let trimmed = single
        .trim()
        .trim_matches(|ch: char| ch == '-' || ch == '*' || ch == '#' || ch == '+')
        .trim();
    let trimmed = trimmed.trim_end_matches([':', ';', '.']).trim();
    if trimmed.chars().count() < 3 {
        return None;
    }
    Some(truncate(trimmed, MAX_PLAN_ITEM_CHARS))
}

/// Truncates `text` to `max_chars` characters, marking the cut with `...`.
fn truncate(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_owned();
    }
    let mut out: String = text.chars().take(max_chars.saturating_sub(3)).collect();
    out.push_str("...");
    out
}

/// Collects path-like tokens of one line into `files`.
fn collect_files(line: &str, files: &mut BTreeSet<String>) {
    for token in line.split(|ch: char| {
        ch.is_whitespace()
            || matches!(
                ch,
                ',' | ';'
                    | '"'
                    | '\''
                    | '`'
                    | '('
                    | ')'
                    | '['
                    | ']'
                    | '{'
                    | '}'
                    | ':'
                    | '='
                    | '<'
                    | '>'
            )
    }) {
        let token = token.trim_matches(|ch: char| ch == '.' || ch == ',' || ch == ':' || ch == '-');
        if !looks_like_path(token) {
            continue;
        }
        let token = truncate(token, MAX_PLAN_ITEM_CHARS);
        files.insert(token);
    }
}

/// Whether `token` looks like a path the span touched.
fn looks_like_path(token: &str) -> bool {
    if token.len() < 4 || token.starts_with("http://") || token.starts_with("https://") {
        return false;
    }
    let lower = token.to_ascii_lowercase();
    if other_extension_matches(&lower) {
        return true;
    }
    lower.contains('/') && lower.chars().any(|ch| ch.is_ascii_alphabetic())
}

/// Whether `token` ends with a known source or document extension.
fn other_extension_matches(lower: &str) -> bool {
    FILE_EXTENSIONS
        .iter()
        .any(|extension| lower.ends_with(extension))
}

/// Classifies one line into the plan's narrative sections.
fn classify_line(line: &str, plan: &mut MemoryBlockPlan) {
    let lower = line.to_ascii_lowercase();
    if TASK_MARKERS.iter().any(|marker| lower.contains(marker)) {
        push_item(&mut plan.tasks, line);
        return;
    }
    if DECISION_MARKERS.iter().any(|marker| lower.contains(marker)) {
        push_item(&mut plan.decisions, line);
        return;
    }
    if CONSTRAINT_MARKERS
        .iter()
        .any(|marker| lower.contains(marker))
    {
        push_item(&mut plan.constraints, line);
        return;
    }
    if lower.trim_end().ends_with('?')
        || QUESTION_MARKERS.iter().any(|marker| lower.contains(marker))
    {
        push_item(&mut plan.open_questions, line);
    }
}
