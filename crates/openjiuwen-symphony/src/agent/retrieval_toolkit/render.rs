//! Rendering: page bodies, the model summary header, the prompt snapshot and the
//! character-budget helpers.
//!
//! All text here is deterministic and byte-identical for identical inputs. Every
//! shortening operation is character based, so a multi-byte character is never split.
//! The page budget is enforced before the `[next]` footer is appended: the caller reserves
//! [`super::CURSOR_FOOTER_CHARS`] and this module fits the rows into what remains.
use std::fmt::Write as _;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::ops::{node_stats, ReadItem};
use super::{RetrievalEntry, RetrievalOperation, ABSTAIN_BLANK_QUERY, ABSTAIN_NO_RESULTS};
use crate::retrieval::types::{LoadedIndex, RetrieverNode};

/// The characters one rendered row may use before the body compactor shortens it.
pub(crate) const ROW_MAX_CHARS: usize = 420;
/// The characters a row's description may use.
pub(crate) const ROW_DESCRIPTION_CHARS: usize = 240;
/// The characters a node header's description may use.
pub(crate) const NODE_DESCRIPTION_CHARS: usize = 120;
/// The characters a search query may use in the title and summary.
pub(crate) const QUERY_DISPLAY_CHARS: usize = 80;
/// How many first-level categories the prompt lists before summarising the rest.
pub(crate) const MAX_PROMPT_CATEGORIES: usize = 30;
/// The characters a prompt category's description may use.
pub(crate) const PROMPT_CATEGORY_DESCRIPTION_CHARS: usize = 120;

/// The fixed prompt instructions, ending with the category-section heading.
const PROMPT_HEAD: &str = "\
# Agentic Skill Retrieval

`skill_retrieval` is a read-only browser over the indexed skill tree. It discloses
categories and installed skills; read a returned skill only after it looks likely needed.
For a task that clearly needs installed skills, use the directory before answering that no
skill is relevant.

- `{\"operation\":\"list\"}` lists the first-level categories; `{\"operation\":\"list\",\"cid\":\"<cid>\"}`
  lists one branch's direct sub-branches with skill and descendant counts.
- `{\"operation\":\"browse\",\"cid\":\"<cid>\"}` shows one branch's description, sub-branches and
  its own skills.
- `{\"operation\":\"search\",\"query\":\"<task phrase>\"}` ranks indexed skills for a phrase.
  Search never invents a skill: a blank or unmatched query abstains.
- `{\"operation\":\"read\",\"cid\":\"<cid or worker id>\"}` returns one item's routing description.

Cids are the backticked tree ids in the results (`Skills.Data`), never display numbers and
never filesystem paths. Prioritize the main capability branch, then explore a supporting
branch only when a task constraint changes skill selection, execution or acceptance. Prefer
a precise task skill over a broad overview skill. Copy a `[next]` cursor unchanged to page;
omit the cursor to change the request.

## First-Level Categories
";

/// One rendered prompt: the text plus what it could and could not include.
///
/// `included` holds the first-level category cids the text shows (in tree order) and
/// `omitted` the cids it does not (the `MAX_PROMPT_CATEGORIES` cap or the character
/// budget). `truncated` is set only when the budget, not the category cap, shortened the
/// text.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct RetrievalPromptSnapshot {
    /// The rendered prompt, at most `max_chars` characters.
    pub text: String,
    /// The first-level category cids the text shows.
    pub included: Vec<String>,
    /// The first-level category cids the text does not show.
    pub omitted: Vec<String>,
    /// Whether the character budget shortened the text.
    pub truncated: bool,
    /// The character budget the text was fitted into.
    pub max_chars: usize,
}

/// Renders the retrieval prompt over `index`, fitted into `max_chars` characters.
///
/// The instructions are followed by the first `MAX_PROMPT_CATEGORIES` first-level
/// branches as ``- `cid` - description (N skills)``; any further branch is summarised as
/// `- ... N more categories` when the budget still has room. A `max_chars` of `0` yields
/// an empty text with every category omitted.
#[must_use]
pub fn render_retrieval_prompt(index: &LoadedIndex, max_chars: usize) -> RetrievalPromptSnapshot {
    let children = index.root.children();
    let mut text = PROMPT_HEAD.to_string();
    let mut included: Vec<String> = Vec::new();
    let mut omitted: Vec<String> = Vec::new();
    let mut truncated = false;
    if max_chars < text.chars().count() {
        let (fitted, _) = truncate_chars(&text, max_chars);
        return RetrievalPromptSnapshot {
            text: fitted,
            included,
            omitted: children.iter().map(|child| child.node_id.clone()).collect(),
            truncated: true,
            max_chars,
        };
    }
    for child in &children {
        if included.len() >= MAX_PROMPT_CATEGORIES {
            omitted.push(child.node_id.clone());
            continue;
        }
        let line = category_line(child);
        let next_len = text.chars().count() + line.chars().count() + 1;
        if next_len > max_chars {
            truncated = true;
            omitted.push(child.node_id.clone());
            continue;
        }
        text.push_str(&line);
        text.push('\n');
        included.push(child.node_id.clone());
    }
    if included.is_empty() && omitted.is_empty() {
        let line = "No first-level branches are available.".to_string();
        if text.chars().count() + line.chars().count() <= max_chars {
            text.push_str(&line);
        } else {
            truncated = true;
        }
    }
    if !omitted.is_empty() {
        let note = format!("- ... {} more categories", omitted.len());
        if text.chars().count() + 1 + note.chars().count() <= max_chars {
            if !text.ends_with('\n') {
                text.push('\n');
            }
            text.push_str(&note);
        } else {
            truncated = true;
        }
    }
    RetrievalPromptSnapshot {
        text: text.trim_end().to_string(),
        included,
        omitted,
        truncated,
        max_chars,
    }
}

/// One prompt category line.
fn category_line(node: &RetrieverNode) -> String {
    let description = compact(&node.description, PROMPT_CATEGORY_DESCRIPTION_CHARS);
    let suffix = if description.is_empty() {
        String::new()
    } else {
        format!(" - {description}")
    };
    format!(
        "- `{}`{} ({} skills)",
        node.node_id,
        suffix,
        node_stats(node).item_count
    )
}

/// The context one page is rendered in.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RenderContext<'a> {
    /// The operation being rendered.
    pub operation: RetrievalOperation,
    /// The resolved scope cid (`ROOT`, a branch cid, or empty for search).
    pub cid: &'a str,
    /// The scope node's label.
    pub label: &'a str,
    /// The scope node's description.
    pub description: &'a str,
    /// The search query; empty for the other operations.
    pub query: &'a str,
    /// The backend that answered.
    pub method: &'a str,
    /// How many rows the operation matched, before paging.
    pub total: usize,
    /// The offset this page starts at.
    pub offset: usize,
}

/// One fitted page.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct RenderedPage {
    /// The concise body: title, rows and no footer.
    pub content: String,
    /// `content` behind the summary header.
    pub model_content: String,
    /// How many rows fitted.
    pub shown_count: usize,
    /// Whether the budget left a row out.
    pub shortened: bool,
}

/// Renders one page of `window` rows into `budget` characters, reserving `footer_reserve`
/// characters for the caller's `[next]` footer.
///
/// The first row always fits (compacted to the room left) when the budget allows at least
/// a few characters; otherwise `shown_count` is `0` and `shortened` is set, so the caller
/// can refuse to mint a cursor that would not advance.
pub(crate) fn render_page(
    context: &RenderContext<'_>,
    window: &[RetrievalEntry],
    budget: usize,
    footer_reserve: usize,
) -> RenderedPage {
    let title = title(context);
    let header = summary_line(context, window.len());
    let body_budget = budget
        .saturating_sub(header.chars().count() + 2)
        .saturating_sub(footer_reserve);
    let mut body = title.clone();
    let mut shown = 0_usize;
    let mut shortened = false;
    for entry in window {
        let row = compact(&render_row(context.operation, entry), ROW_MAX_CHARS);
        let extra = if shown == 0 {
            row.chars().count()
        } else {
            row.chars().count() + 1
        };
        if body.chars().count() + 2 + extra > body_budget && shown > 0 {
            shortened = true;
            break;
        }
        if body.chars().count() + 2 + extra > body_budget {
            let available = body_budget.saturating_sub(body.chars().count() + 2);
            if available >= 8 {
                body.push_str("\n\n");
                body.push_str(&compact(&row, available));
                shown = 1;
            }
            shortened = true;
            break;
        }
        if shown == 0 {
            body.push_str("\n\n");
        } else {
            body.push('\n');
        }
        body.push_str(&row);
        shown += 1;
    }
    if window.is_empty() {
        let message = empty_message(context.operation);
        let next_len = body.chars().count() + 2 + message.chars().count();
        if next_len <= body_budget {
            body.push_str("\n\n");
            body.push_str(message);
        } else {
            let available = body_budget.saturating_sub(body.chars().count() + 2);
            if available >= 8 {
                body.push_str("\n\n");
                body.push_str(&compact(message, available));
            }
            shortened = true;
        }
    }
    let (body, budget_cut) = truncate_chars(&body, body_budget);
    if budget_cut {
        shortened = true;
    }
    let complete = context.offset + shown >= context.total && !shortened;
    let header = summary_line_with_complete(context, shown, complete);
    let mut model_content = header;
    model_content.push_str("\n\n");
    model_content.push_str(&body);
    RenderedPage {
        content: body,
        model_content,
        shown_count: shown,
        shortened,
    }
}
/// Renders one item's full routing description.
pub(crate) fn render_read(item: &ReadItem, budget: usize) -> RenderedPage {
    let entry = &item.entry;
    let mut body = format!("# Skill\n\n## Item: `{}`", entry.label);
    if !entry.worker_id.is_empty() {
        let _ = write!(body, "\n- worker_id: `{}`", entry.worker_id);
    }
    let _ = write!(body, "\n- cid: `{}`", entry.cid);
    if !entry.path.is_empty() {
        let path = entry
            .path
            .iter()
            .map(|segment| format!("`{segment}`"))
            .collect::<Vec<_>>()
            .join(" > ");
        let _ = write!(body, "\n- path: {path}");
    }
    if !item.skill_path.is_empty() {
        let _ = write!(body, "\n- SKILL.md: `{}`", item.skill_path);
    }
    body.push_str("\n\n");
    body.push_str(item.body.trim());
    let header = format!(
        "[skill_retrieval] operation=read cid={} total=1 shown=1 offset=0 complete=true",
        entry.cid
    );
    let body_budget = budget.saturating_sub(header.chars().count() + 2);
    let (body, shortened) = truncate_chars(&body, body_budget);
    let mut model_content = header;
    model_content.push_str("\n\n");
    model_content.push_str(&body);
    RenderedPage {
        content: body,
        model_content,
        shown_count: 1,
        shortened,
    }
}
/// Renders one abstain answer: the reason, the suggestion and the summary header.
pub(crate) fn render_abstain(reason: &str, suggestion: &str, method: &str) -> (String, String) {
    let message = match reason {
        ABSTAIN_BLANK_QUERY => "The search query is empty, so no search was run.",
        ABSTAIN_NO_RESULTS => "No indexed skill matched the query.",
        other => other,
    };
    let content = format!("# Skill Search\n\n{message}\n\nSuggestion: {suggestion}");
    let header = format!(
        "[skill_retrieval] operation=search cid=- total=0 shown=0 offset=0 complete=true method={method}"
    );
    let model_content = format!("{header}\n\n{content}");
    (content, model_content)
}

/// The title block of one page.
fn title(context: &RenderContext<'_>) -> String {
    match context.operation {
        RetrievalOperation::List | RetrievalOperation::Browse => {
            let name = if context.operation == RetrievalOperation::List {
                "List"
            } else {
                "Browse"
            };
            let cid = if context.cid.is_empty() {
                "ROOT"
            } else {
                context.cid
            };
            let mut title = format!("# Skill Branch {name}\n\n## Node: `{cid}`");
            let description = compact(context.description, NODE_DESCRIPTION_CHARS);
            if !description.is_empty() {
                let _ = write!(title, " - {description}");
            }
            title
        }
        RetrievalOperation::Search => format!(
            "# Skill Search\n\nQuery: \"{}\"",
            compact(context.query, QUERY_DISPLAY_CHARS)
        ),
        RetrievalOperation::Read => format!("# Skill\n\n## Item: `{}`", context.label),
    }
}

/// The one-line `[skill_retrieval] …` summary.
fn summary_line(context: &RenderContext<'_>, shown: usize) -> String {
    let complete = context.offset + shown >= context.total;
    summary_line_with_complete(context, shown, complete)
}

/// The summary line with an explicit completion flag.
fn summary_line_with_complete(context: &RenderContext<'_>, shown: usize, complete: bool) -> String {
    let cid = if context.cid.is_empty() {
        "-"
    } else {
        context.cid
    };
    let query = if context.query.is_empty() {
        String::new()
    } else {
        format!(" query=\"{}\"", compact(context.query, QUERY_DISPLAY_CHARS))
    };
    format!(
        "[skill_retrieval] operation={} cid={cid}{query} total={} shown={shown} offset={} \
         complete={} method={}",
        context.operation.as_str(),
        context.total,
        context.offset,
        if complete { "true" } else { "false" },
        context.method
    )
}

/// Renders one row.
fn render_row(operation: RetrievalOperation, entry: &RetrievalEntry) -> String {
    let description = first_line(&entry.description);
    let suffix = if description.is_empty() {
        String::new()
    } else {
        format!(" - {}", compact(description, ROW_DESCRIPTION_CHARS))
    };
    match operation {
        RetrievalOperation::Search => {
            let name = if entry.worker_id.is_empty() {
                entry.label.as_str()
            } else {
                entry.worker_id.as_str()
            };
            format!(
                "{}. `{}` (cid: `{}`){}",
                entry.rank, name, entry.cid, suffix
            )
        }
        RetrievalOperation::Browse => {
            if entry.kind == "branch" {
                format!(
                    "- `{}`{} ({} skills, {} branches)",
                    entry.cid, suffix, entry.item_count, entry.branch_count
                )
            } else {
                let name = if entry.worker_id.is_empty() {
                    entry.label.as_str()
                } else {
                    entry.worker_id.as_str()
                };
                format!("- `{}` (cid: `{}`){}", name, entry.cid, suffix)
            }
        }
        RetrievalOperation::List | RetrievalOperation::Read => format!(
            "- `{}`{} ({} skills, {} branches)",
            entry.cid, suffix, entry.item_count, entry.branch_count
        ),
    }
}

/// The message a page with no rows shows.
fn empty_message(operation: RetrievalOperation) -> &'static str {
    match operation {
        RetrievalOperation::List => "No sub-branches are available.",
        RetrievalOperation::Browse => "No sub-branches or skills are available in this node.",
        RetrievalOperation::Search => "No matching skills.",
        RetrievalOperation::Read => "No routing description is available.",
    }
}

/// Collapses whitespace and truncates `value` to `limit` characters, adding `...`.
#[must_use]
pub(crate) fn compact(value: &str, limit: usize) -> String {
    let text = value.split_whitespace().collect::<Vec<_>>().join(" ");
    if text.chars().count() <= limit {
        return text;
    }
    if limit == 0 {
        return String::new();
    }
    let keep = limit.saturating_sub(3);
    let kept: String = text.chars().take(keep).collect();
    let mut kept = kept.trim_end().to_string();
    if limit > 3 {
        kept.push_str("...");
    }
    kept
}

/// Truncates `text` to `limit` characters, adding `...` when there is room for it.
///
/// Returns the text and whether it was shortened; the cut is always on a character
/// boundary.
#[must_use]
pub(crate) fn truncate_chars(text: &str, limit: usize) -> (String, bool) {
    if text.chars().count() <= limit {
        return (text.to_string(), false);
    }
    if limit == 0 {
        return (String::new(), true);
    }
    let keep = limit.saturating_sub(3);
    let kept: String = text.chars().take(keep).collect();
    let mut kept = kept.trim_end().to_string();
    if limit > 3 {
        kept.push_str("...");
    }
    (kept, true)
}

/// Appends `note` to `content` as a blank-line separated paragraph within `budget`.
///
/// An empty `content` becomes just the note (compacted to `budget`); with no budget the
/// note is appended unchanged; otherwise the note is kept only when at least four
/// characters remain after `content` and the separator, and is compacted to the room left.
#[must_use]
pub(crate) fn append_note(content: &str, note: &str, budget: Option<usize>) -> String {
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

/// The first non-empty line of `text`, trimmed.
fn first_line(text: &str) -> &str {
    text.split('\n')
        .map(str::trim)
        .find(|line| !line.is_empty())
        .unwrap_or("")
}
