//! Row rendering, summary lines, and output budgeting for the `skill_index` tool.
//!
//! Ports the rendering half of Python `openjiuwen.symphony.discovery.directory_toolkit`:
//! `_list_row`, `_search_row`, `_search_directory_row`, `_skill_path`, `_inline_text`,
//! `_safe_match_snippet`, `_directory_description`, `_metadata_card`, `_row_count`,
//! `_summary_line`, `_bounded_summary_line`, `_compact`, `_append_note`,
//! `_candidate_tokens`, `_empty_message`, and `_fit_rows`.
//!
//! Every pinned string keeps Python's exact spacing, including the two spaces that
//! separate `_list_row`/`_search_row` fields and the `.rstrip()` that trims a category
//! row whose description is empty.
//!
//! Budgets are counted in [`char`]s (`chars().count()`), never bytes, so a multi-byte
//! label is never cut through a code point — Python's `len()` counts code points too.
//!
//! # Deviation from Python
//!
//! Python's `sanitize_model_text` prefixes `[untrusted Skill metadata] ` whenever its
//! instruction-injection heuristic fires or the text contains `<`/`>`. The port keeps
//! the escaping (`<` → `&lt;`, `>` → `&gt;`) and the control-character scrubbing but
//! drops the prefix heuristic: escaping already stops a metadata line from forging
//! markup or another row, and the prefix would change every pinned row text.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::discovery::models::SkillRecord;

use super::matcher::case_fold;
use super::view::{DirectoryEntry, SkillDirectoryView};

/// The characters of a Skill description a row shows.
pub const SKILL_DESCRIPTION_CHARS: usize = 180;
/// The characters of a category description a list row shows.
pub const CATEGORY_DESCRIPTION_CHARS: usize = 240;
/// The characters of a category description a search row shows.
pub const SEARCH_CATEGORY_DESCRIPTION_CHARS: usize = 300;
/// The characters of narrative routing text a category description keeps.
pub const CATEGORY_ROUTING_CHARS: usize = 180;
/// The characters of `Select when:` text a category description keeps.
pub const SELECT_WHEN_CHARS: usize = 96;
/// The marker appended when the output budget left rows or characters out.
pub const SHORTENED_MARKER: &str = "[skill_index output shortened to fit output budget]";
/// The category-row prefix a summary counts.
pub const CATEGORY_ROW_PREFIX: &str = "- [category]";

/// Which list view a row is rendered for.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ListView {
    /// Categories as names only, Skills as one identifying line.
    #[default]
    Names,
    /// Names plus the description each entry carries.
    Details,
    /// The tree, indented by depth.
    Tree,
}

impl ListView {
    /// Every view, in Python's `_LIST_VIEWS` order.
    pub const ALL: [Self; 3] = [Self::Names, Self::Details, Self::Tree];

    /// The wire name of this view.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Names => "names",
            Self::Details => "details",
            Self::Tree => "tree",
        }
    }

    /// Parses a wire name, returning `None` for an unknown value.
    #[must_use]
    pub fn parse(value: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|view| view.as_str() == value)
    }
}

impl std::fmt::Display for ListView {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// One rendered output row: its text plus the Skill it belongs to, when any.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct Row {
    /// The rendered line(s).
    pub text: String,
    /// The Skill this row observes; empty for a category row.
    pub worker_id: String,
}

impl Row {
    /// Builds a row that observes `worker_id`.
    #[must_use]
    pub fn new(text: impl Into<String>, worker_id: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            worker_id: worker_id.into(),
        }
    }

    /// Builds a row that observes no Skill (a category or a count row).
    #[must_use]
    pub fn text_only(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            worker_id: String::new(),
        }
    }

    /// Whether this row is a category row.
    #[must_use]
    pub fn is_category_row(&self) -> bool {
        self.text.trim_start().starts_with(CATEGORY_ROW_PREFIX)
    }
}

/// The counts a summary line is rendered from.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct ResponseSummary {
    /// The operation the response answers (`list`, `search`, or `read`).
    pub operation: String,
    /// How many distinct Skills the response returns.
    pub returned_skill_count: usize,
    /// How many category rows the response returns.
    pub returned_category_count: usize,
    /// Whether every returned Skill was already shown to this session.
    pub previously_shown: bool,
}

/// The result of fitting rows into an output budget.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct FittedRows {
    /// The body rows (without the header), including the shortened marker when shown.
    pub body: String,
    /// The model-facing text: header, blank line, body.
    pub model_content: String,
    /// The Skill ids the returned rows observe, de-duplicated in row order.
    pub observed: Vec<String>,
    /// Whether the budget left at least one row (or character) out.
    pub shortened: bool,
    /// How many entries are shown, or the pipeline count when one was supplied.
    pub shown_count: usize,
}

/// Renders one list row.
///
/// A category row reads `- [category] <chain>  desc: Contains <n> skill(s). <description>`
/// (the `desc:` part is dropped in the names view and trailing whitespace is trimmed),
/// and a Skill row reads
/// `- [skill] <worker_id>  category: <chain>  path: <SKILL.md>` plus the same optional
/// `desc:` tail. Tree rows are indented two spaces per depth.
///
/// Rendering is total: an entry whose worker id or path the view does not know renders
/// an empty path (or `Contains 0 skills.`) instead of failing, because the pinned
/// signature returns a [`Row`]. Python indexes the view and raises.
#[must_use]
pub fn list_row(entry: &DirectoryEntry, view: ListView, directory: &SkillDirectoryView) -> Row {
    let indent = if view == ListView::Tree {
        "  ".repeat(entry.depth)
    } else {
        String::new()
    };
    if entry.is_directory() {
        let description = directory_description(&entry.description, CATEGORY_DESCRIPTION_CHARS);
        let count = directory.skill_count_or_zero(&entry.path);
        let mut detail = format!(
            "  desc: Contains {count} {}. {description}",
            plural_skills(count)
        );
        detail = if view == ListView::Names {
            String::new()
        } else {
            detail.trim_end().to_string()
        };
        return Row::text_only(format!(
            "{indent}- [category] {}{detail}",
            directory.category_from_path(&entry.path)
        ));
    }
    let category = directory.skill_category_or_empty(&entry.worker_id);
    let detail = if view != ListView::Names && !entry.description.is_empty() {
        format!(
            "  desc: {}",
            compact(&entry.description, SKILL_DESCRIPTION_CHARS)
        )
    } else {
        String::new()
    };
    let path = directory
        .record(&entry.worker_id)
        .map(skill_path)
        .unwrap_or_default();
    Row::new(
        format!(
            "{indent}- [skill] {}  category: {category}  path: {path}{detail}",
            entry.worker_id
        ),
        entry.worker_id.clone(),
    )
}

/// Renders one Skill search row.
///
/// The fields are joined by two spaces:
/// `- [skill] <worker_id>  category: <chain>[  matches: 1, 3][  match: <snippet>]  desc: <description>  path: <SKILL.md>`.
#[must_use]
pub fn search_row(
    record: &SkillRecord,
    directory: &SkillDirectoryView,
    indexes: &[usize],
    snippet: Option<&str>,
) -> Row {
    let description = compact(
        if record.description.is_empty() {
            record.name.as_str()
        } else {
            record.description.as_str()
        },
        SKILL_DESCRIPTION_CHARS,
    );
    let mut fields = vec![
        format!("- [skill] {}", record.worker_id),
        format!(
            "category: {}",
            directory.skill_category_or_empty(&record.worker_id)
        ),
    ];
    if !indexes.is_empty() {
        fields.push(format!("matches: {}", join_indexes(indexes)));
    }
    if let Some(snippet) = snippet {
        if !snippet.is_empty() {
            fields.push(format!("match: {snippet}"));
        }
    }
    fields.push(format!("desc: {description}"));
    fields.push(format!("path: {}", skill_path(record)));
    Row::new(fields.join("  "), record.worker_id.clone())
}

/// Renders one category search row.
///
/// Reads `- [category] <chain>[  matches: 1, 3]  desc: Contains <n> skill(s). <description>`
/// with trailing whitespace trimmed.
#[must_use]
pub fn search_directory_row(
    entry: &DirectoryEntry,
    directory: &SkillDirectoryView,
    indexes: &[usize],
) -> Row {
    let mapping = if indexes.is_empty() {
        String::new()
    } else {
        format!("  matches: {}", join_indexes(indexes))
    };
    let description = directory_description(&entry.description, SEARCH_CATEGORY_DESCRIPTION_CHARS);
    let count = directory.skill_count_or_zero(&entry.path);
    let detail = format!(
        "  desc: Contains {count} {}. {description}",
        plural_skills(count)
    );
    let detail = detail.trim_end();
    Row::text_only(format!(
        "- [category] {}{mapping}{detail}",
        directory.category_from_path(&entry.path)
    ))
}

/// The rendered `SKILL.md` path of one record (`inline_text(record.skill_file)`).
#[must_use]
pub fn skill_path(record: &SkillRecord) -> String {
    inline_text(&record.skill_file)
}

/// Renders one field so it cannot create another Markdown row.
///
/// The value is sanitized with [`sanitize_model_text`] and then JSON-string-escaped
/// without the surrounding quotes, exactly like Python's
/// `json.dumps(sanitize_model_text(value), ensure_ascii=False)[1:-1]`: quotes,
/// backslashes, and newlines become escapes and non-ASCII text stays verbatim.
///
/// ```
/// use openjiuwen_symphony::discovery::directory_toolkit::render::inline_text;
///
/// assert_eq!(inline_text("a \"b\""), "a \\\"b\\\"");
/// assert_eq!(inline_text("a\nb"), "a\\nb");
/// ```
#[must_use]
pub fn inline_text(value: &str) -> String {
    let sanitized = sanitize_model_text(value);
    let Ok(rendered) = serde_json::to_string(&sanitized) else {
        return sanitized;
    };
    if rendered.len() < 2 {
        return String::new();
    }
    rendered
        .get(1..rendered.len() - 1)
        .unwrap_or_default()
        .to_string()
}

/// Escapes and scrubs model-visible metadata.
///
/// Leading and trailing whitespace is trimmed, `<` and `>` become `&lt;`/`&gt;`, then
/// the result is trimmed again. Control characters are dropped, except `'\n'` and
/// `'\t'`; the Unicode line/paragraph separators (`U+2028`/`U+2029`) become a space so a
/// value cannot open a line of its own.
///
/// Python replaces the dropped control characters with a space and prefixes
/// `[untrusted Skill metadata] ` when it suspects instruction injection; see the module
/// docs for why the port drops them and skips the prefix.
#[must_use]
pub fn sanitize_model_text(value: &str) -> String {
    let mut cleaned = String::with_capacity(value.len());
    for character in value.trim().chars() {
        if matches!(character, '\u{2028}' | '\u{2029}') {
            cleaned.push(' ');
            continue;
        }
        if character.is_control() && !matches!(character, '\n' | '\t') {
            continue;
        }
        match character {
            '<' => cleaned.push_str("&lt;"),
            '>' => cleaned.push_str("&gt;"),
            other => cleaned.push(other),
        }
    }
    cleaned.trim().to_string()
}

/// Scrubs and re-labels one search snippet.
///
/// A `name: `, `alias: `, `description: `, `body: `, or `category: ` prefixed snippet
/// keeps its label and sanitizes only the evidence, so the label itself cannot be
/// forged; every other value is sanitized whole. Python's `partition(": ")` semantics are
/// preserved, so a `body:12: text` snippet (whose label is not one of the five) is
/// sanitized whole.
#[must_use]
pub fn safe_match_snippet(value: &str) -> String {
    match value.split_once(": ") {
        Some((label, evidence)) if is_snippet_label(label) => {
            format!("{label}: {}", sanitize_model_text(evidence))
        }
        _ => sanitize_model_text(value),
    }
}

/// Renders a category description: the semantic paragraph plus any `Select when:` line.
///
/// Mirrors Python's `_directory_description`: index-construction statistics (`covers …`,
/// `representative …`, `don't select when: …`) are dropped, the first non-empty paragraph
/// becomes the semantic summary (cut to `min(limit, 180)` characters), the first
/// `Select when:` line becomes routing text (cut to 96 characters), and both are joined
/// and cut to `limit`.
#[must_use]
pub fn directory_description(value: &str, limit: usize) -> String {
    let text = sanitize_model_text(value);
    let mut select_when = String::new();
    let mut semantic_lines: Vec<String> = Vec::new();
    for paragraph in paragraphs(&text) {
        let mut current: Vec<String> = Vec::new();
        for raw_line in paragraph {
            let line = raw_line.trim();
            let lowered = case_fold(line);
            if lowered.starts_with("select when:") {
                if select_when.is_empty() {
                    select_when = line
                        .split_once(':')
                        .map(|(_, rest)| rest.trim().to_string())
                        .unwrap_or_default();
                }
                continue;
            }
            if lowered.starts_with("covers ")
                || lowered.starts_with("representative ")
                || lowered.starts_with("don't select when:")
            {
                continue;
            }
            if !line.is_empty() {
                current.push(line.to_string());
            }
        }
        if !current.is_empty() && semantic_lines.is_empty() {
            semantic_lines = current;
        }
    }
    let semantic = if semantic_lines.is_empty() {
        String::new()
    } else {
        compact(&semantic_lines.join(" "), limit.min(CATEGORY_ROUTING_CHARS))
    };
    let routing = if select_when.is_empty() {
        String::new()
    } else {
        format!("Select when: {}", compact(&select_when, SELECT_WHEN_CHARS))
    };
    let joined: Vec<String> = [semantic, routing]
        .into_iter()
        .filter(|part| !part.is_empty())
        .collect();
    compact(&joined.join(" "), limit)
}

/// Collapses whitespace and cuts `value` to `limit` characters.
///
/// Every run of whitespace becomes one space and the result is trimmed; a value longer
/// than `limit` is cut to `limit - 3` characters (trimmed) plus `...`. A `limit` below
/// three therefore renders `...`.
#[must_use]
pub fn compact(value: &str, limit: usize) -> String {
    let text = value.split_whitespace().collect::<Vec<_>>().join(" ");
    if text.chars().count() <= limit {
        return text;
    }
    let keep = limit.saturating_sub(3);
    let prefix: String = text.chars().take(keep).collect();
    format!("{}...", prefix.trim_end())
}

/// Appends `note` to `content` as a blank-line separated paragraph within `budget`.
///
/// An empty `content` becomes just the note (compacted to `budget`); with no budget the
/// note is appended unchanged; otherwise the note is kept only when at least four
/// characters remain after `content` and the separator, and is compacted to the room
/// that is left.
#[must_use]
pub fn append_note(content: &str, note: &str, budget: Option<usize>) -> String {
    if content.is_empty() {
        return match budget {
            None => note.to_string(),
            Some(budget) => compact(note, budget),
        };
    }
    let Some(budget) = budget else {
        return format!("{content}\n\n{note}");
    };
    let remaining = budget
        .saturating_sub(content.chars().count())
        .saturating_sub(2);
    if remaining <= 3 {
        return content.to_string();
    }
    let rendered = if note.chars().count() <= remaining {
        note.to_string()
    } else {
        compact(note, remaining)
    };
    format!("{content}\n\n{rendered}")
}

/// Renders the metadata card one `read` row shows.
///
/// The card is `# <name>`, a blank line, `- Description: …`, `- Source: …`, and the
/// optional `- Version:`/`- Author:` lines. A blank name falls back to the worker id, a
/// blank description to the name, and a blank source to `local`.
#[must_use]
pub fn metadata_card(record: &SkillRecord) -> String {
    let name = if record.name.is_empty() {
        record.worker_id.as_str()
    } else {
        record.name.as_str()
    };
    let description = if record.description.is_empty() {
        name
    } else {
        record.description.as_str()
    };
    let source = if record.source.is_empty() {
        "local"
    } else {
        record.source.as_str()
    };
    let mut lines = vec![
        format!("# {name}"),
        String::new(),
        format!("- Description: {description}"),
        format!("- Source: {source}"),
    ];
    if !record.version.is_empty() {
        lines.push(format!("- Version: {}", record.version));
    }
    if !record.author.is_empty() {
        lines.push(format!("- Author: {}", record.author));
    }
    lines.join("\n")
}

/// Renders the summary header line for one response shape.
///
/// | skills | categories | operation | header |
/// | --- | --- | --- | --- |
/// | yes | no | any | `## Skills`, or `## Skills — all names previously shown` |
/// | no | yes | search | `## Category hints — matching groups, not Skills` |
/// | no | yes | other | `## Categories` |
/// | yes | yes | search | `## Skills and category hints` |
/// | yes | yes | other | `## Entries` |
/// | no | no | search | `## Skills` |
/// | no | no | other | `## Results` |
#[must_use]
pub fn summary_line(summary: &ResponseSummary) -> String {
    if summary.returned_skill_count > 0 && summary.returned_category_count == 0 {
        return if summary.previously_shown {
            "## Skills — all names previously shown".to_string()
        } else {
            "## Skills".to_string()
        };
    }
    if summary.returned_category_count > 0 && summary.returned_skill_count == 0 {
        return if summary.operation == "search" {
            "## Category hints — matching groups, not Skills".to_string()
        } else {
            "## Categories".to_string()
        };
    }
    if summary.returned_category_count > 0 {
        return if summary.operation == "search" {
            "## Skills and category hints".to_string()
        } else {
            "## Entries".to_string()
        };
    }
    if summary.operation == "search" {
        "## Skills".to_string()
    } else {
        "## Results".to_string()
    }
}

/// Renders [`summary_line`], compacted to `budget` when one is given.
#[must_use]
pub fn bounded_summary_line(summary: &ResponseSummary, budget: Option<usize>) -> String {
    let line = summary_line(summary);
    match budget {
        None => line,
        Some(budget) => compact(&line, budget),
    }
}

/// Fits `rows` into `budget` characters and renders the model-facing text.
///
/// Reproduces Python's two-phase selection:
///
/// 1. append rows while `header + "\n\n" + body` still fits;
/// 2. pop rows while the summary recomputed from the selected rows still overflows;
/// 3. when nothing fits, render a single compacted copy of the first row;
/// 4. append [`SHORTENED_MARKER`] when rows were dropped, the caller allows the marker,
///    and it still fits.
///
/// `count_value` replaces the shown count (a pipeline `count` stage), and
/// `show_shortened_marker` is `false` for a paged simple request, whose footer already
/// says the result was shortened.
#[must_use]
pub fn fit_rows(
    rows: &[Row],
    summary: &ResponseSummary,
    budget: Option<usize>,
    count_value: Option<usize>,
    show_shortened_marker: bool,
) -> FittedRows {
    let mut header = bounded_summary_line(summary, budget);
    let mut selected: Vec<Row> = Vec::new();
    let mut shortened = false;
    for row in rows {
        let mut parts: Vec<&str> = selected.iter().map(|item| item.text.as_str()).collect();
        parts.push(row.text.as_str());
        let body = parts.join("\n");
        let candidate = header.chars().count() + 2 + body.chars().count();
        if let Some(budget) = budget {
            if candidate > budget {
                shortened = true;
                break;
            }
        }
        selected.push(row.clone());
    }
    loop {
        header = bounded_summary_line(&selection_summary(summary, &selected), budget);
        let body = selected
            .iter()
            .map(|item| item.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        let candidate = if body.is_empty() {
            header.chars().count()
        } else {
            header.chars().count() + 2 + body.chars().count()
        };
        if budget.is_none_or(|budget| candidate <= budget) || selected.is_empty() {
            break;
        }
        selected.pop();
        shortened = true;
    }
    if shortened && selected.is_empty() {
        if let (Some(first), Some(budget)) = (rows.first(), budget) {
            header = bounded_summary_line(
                &selection_summary(summary, std::slice::from_ref(first)),
                Some(budget),
            );
            let available = budget.saturating_sub(header.chars().count() + 2);
            if available > 3 {
                selected.push(Row::new(
                    compact(&first.text, available),
                    first.worker_id.clone(),
                ));
            }
        }
    }
    let mut body_lines: Vec<String> = selected.iter().map(|item| item.text.clone()).collect();
    if shortened && show_shortened_marker {
        let mut parts: Vec<&str> = vec![header.as_str(), ""];
        parts.extend(body_lines.iter().map(String::as_str));
        parts.push(SHORTENED_MARKER);
        let candidate: usize = parts.iter().map(|part| part.chars().count()).sum::<usize>()
            + parts.len().saturating_sub(1);
        if budget.is_none_or(|budget| candidate <= budget) {
            body_lines.push(SHORTENED_MARKER.to_string());
        }
    }
    let body = body_lines.join("\n");
    let model_content = if body.is_empty() {
        header
    } else {
        format!("{header}\n\n{body}")
    };
    let mut observed: Vec<String> = Vec::new();
    for row in &selected {
        if row.worker_id.is_empty() || observed.contains(&row.worker_id) {
            continue;
        }
        observed.push(row.worker_id.clone());
    }
    let shown_count = match count_value {
        Some(count) if !selected.is_empty() => count,
        _ => selected.len(),
    };
    FittedRows {
        body,
        model_content,
        observed,
        shortened,
        shown_count,
    }
}

/// The number of lines `rows` render, counting an empty row as one line.
#[must_use]
pub fn row_count(rows: &[Row]) -> usize {
    rows.iter().map(|row| row.text.lines().count().max(1)).sum()
}

/// The estimated model tokens of the candidate skill lines.
///
/// Python's `(len(rendered) + 3) // 4` over `- <worker_id>: <collapsed description>`
/// lines; the port counts characters rather than bytes so non-ASCII text is not
/// overestimated.
#[must_use]
pub fn candidate_tokens(records: &[SkillRecord]) -> usize {
    let rendered = records
        .iter()
        .map(|record| {
            format!(
                "- {}: {}",
                record.worker_id,
                record
                    .description
                    .split_whitespace()
                    .collect::<Vec<_>>()
                    .join(" ")
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    rendered.chars().count().div_ceil(4)
}

/// The text a response renders when it has no rows at all.
///
/// A `search` with no match names the category it searched when one is given; every
/// other operation renders `No entries.`
#[must_use]
pub fn empty_message(operation: &str, category: Option<&str>) -> String {
    if operation != "search" {
        return "No entries.".to_string();
    }
    match category {
        Some(category) if !category.is_empty() => {
            format!(
                "No matching Skills found in `{}`.",
                sanitize_model_text(category)
            )
        }
        _ => "No matching installed Skills found.".to_string(),
    }
}

/// Renders `count` as `skill`/`skills`.
fn plural_skills(count: usize) -> &'static str {
    if count == 1 {
        "skill"
    } else {
        "skills"
    }
}

/// Joins 1-based query indexes the way Python's `', '.join(map(str, indexes))` does.
fn join_indexes(indexes: &[usize]) -> String {
    indexes
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(", ")
}

/// Whether `label` is one of the snippet labels `safe_match_snippet` preserves.
fn is_snippet_label(label: &str) -> bool {
    matches!(
        label,
        "name" | "alias" | "description" | "body" | "category"
    )
}

/// Splits `text` into paragraphs on blank lines, preserving each line.
fn paragraphs(text: &str) -> Vec<Vec<&str>> {
    let mut result: Vec<Vec<&str>> = Vec::new();
    let mut current: Vec<&str> = Vec::new();
    for line in text.split('\n') {
        if line.trim().is_empty() {
            if !current.is_empty() {
                result.push(std::mem::take(&mut current));
            }
            continue;
        }
        current.push(line);
    }
    if !current.is_empty() {
        result.push(current);
    }
    result
}

/// Recomputes a summary from the rows actually selected.
fn selection_summary(summary: &ResponseSummary, rows: &[Row]) -> ResponseSummary {
    let mut skills: Vec<&str> = Vec::new();
    for row in rows {
        if row.worker_id.is_empty() || skills.contains(&row.worker_id.as_str()) {
            continue;
        }
        skills.push(row.worker_id.as_str());
    }
    ResponseSummary {
        operation: summary.operation.clone(),
        returned_skill_count: skills.len(),
        returned_category_count: rows.iter().filter(|row| row.is_category_row()).count(),
        previously_shown: summary.previously_shown,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn record() -> SkillRecord {
        SkillRecord::new("alpha", "Alpha").with_description("Adds numbers")
    }

    #[test]
    fn compact_collapses_and_cuts() {
        assert_eq!(compact("  a \n b\tc ", 20), "a b c");
        assert_eq!(compact("abcdef", 6), "abcdef");
        assert_eq!(compact("abcdefg", 6), "abc...");
        assert_eq!(compact("abcdefg", 2), "...");
        assert_eq!(compact("日本語のテキスト", 4), "日...");
    }

    #[test]
    fn inline_text_escapes_without_quotes() {
        assert_eq!(inline_text("plain"), "plain");
        assert_eq!(inline_text("a \"b\""), "a \\\"b\\\"");
        assert_eq!(inline_text("a\nb"), "a\\nb");
        assert_eq!(inline_text("<b>"), "&lt;b&gt;");
        assert_eq!(inline_text("  padded  "), "padded");
        assert_eq!(inline_text("bel\0l"), "bell");
    }

    #[test]
    fn sanitize_keeps_newlines_and_tabs() {
        assert_eq!(sanitize_model_text("a\tb\nc"), "a\tb\nc");
        assert_eq!(sanitize_model_text("a\u{2028}b"), "a b");
        assert_eq!(sanitize_model_text("a\rb"), "ab");
    }

    #[test]
    fn notes_are_compacted_to_the_remaining_room() {
        assert_eq!(append_note("body", "note", None), "body\n\nnote");
        assert_eq!(append_note("", "note", None), "note");
        assert_eq!(append_note("", "a long note", Some(8)), "a lon...");
        assert_eq!(append_note("body", "note", Some(4)), "body");
        assert_eq!(append_note("body", "note", Some(12)), "body\n\nnote");
        assert_eq!(
            append_note("body", "a long note", Some(13)),
            "body\n\na lo..."
        );
    }

    #[test]
    fn summary_lines_cover_every_shape() {
        let summary = |skills, categories, operation: &str, shown| ResponseSummary {
            operation: operation.to_string(),
            returned_skill_count: skills,
            returned_category_count: categories,
            previously_shown: shown,
        };
        assert_eq!(summary_line(&summary(1, 0, "list", false)), "## Skills");
        assert_eq!(
            summary_line(&summary(1, 0, "search", true)),
            "## Skills — all names previously shown"
        );
        assert_eq!(
            summary_line(&summary(0, 1, "search", false)),
            "## Category hints — matching groups, not Skills"
        );
        assert_eq!(summary_line(&summary(0, 1, "list", false)), "## Categories");
        assert_eq!(
            summary_line(&summary(1, 1, "search", false)),
            "## Skills and category hints"
        );
        assert_eq!(summary_line(&summary(1, 1, "list", false)), "## Entries");
        assert_eq!(summary_line(&summary(0, 0, "search", false)), "## Skills");
        assert_eq!(summary_line(&summary(0, 0, "read", false)), "## Results");
        assert_eq!(
            bounded_summary_line(&summary(0, 0, "search", false), Some(4)),
            "#..."
        );
    }

    #[test]
    fn metadata_cards_render_optional_lines() {
        assert_eq!(
            metadata_card(&record()),
            "# Alpha\n\n- Description: Adds numbers\n- Source: local"
        );
        let full = record().with_version("2.0.0").with_author("Ada");
        assert_eq!(
            metadata_card(&full),
            "# Alpha\n\n- Description: Adds numbers\n- Source: local\n- Version: 2.0.0\n- Author: Ada"
        );
        let bare = SkillRecord::new("id", "");
        assert_eq!(
            metadata_card(&bare),
            "# id\n\n- Description: id\n- Source: local"
        );
    }

    #[test]
    fn candidate_tokens_round_up_to_quarters() {
        let record = SkillRecord::new("alpha", "Alpha").with_description("one   two");
        // "- alpha: one two" is 16 characters -> (16 + 3) / 4 == 4.
        assert_eq!(candidate_tokens(&[record]), 4);
        assert_eq!(candidate_tokens(&[]), 0);
    }

    #[test]
    fn empty_messages_name_the_category() {
        assert_eq!(empty_message("list", None), "No entries.");
        assert_eq!(empty_message("read", Some("Data")), "No entries.");
        assert_eq!(
            empty_message("search", None),
            "No matching installed Skills found."
        );
        assert_eq!(
            empty_message("search", Some("Data")),
            "No matching Skills found in `Data`."
        );
    }
}
