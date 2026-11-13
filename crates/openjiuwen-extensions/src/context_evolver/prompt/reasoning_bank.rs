//! `ReasoningBank` prompt construction and reply parsing.
//!
//! Ports Python's
//! `openjiuwen/extensions/context_evolver/summary/task/reasoning_bank/{prompt.py,update.py}`
//! and `retrieve/task/reasoning_bank/run.py`.
//!
//! Python's retrieval is an embedding search followed by an LLM rerank; this
//! port keeps only the rerank half, so [`retrieve_prompt`] asks the model to
//! order the candidate memories it is handed and [`parse_retrieve`] reads the
//! usual `{"ranked_ids": [...]}` ranking through the shared
//! [`parse_ranked_ids`](super::parse_ranked_ids).
//!
//! Summarization is Python's `SummarizeMemoryOp`: one trajectory (or a batch)
//! in, up to three memory items out, with the system prompt selected by the
//! trajectory outcome — success (`EXTRACT_SUCCESS_TRAJ_SYSTEM_PROMPT`), failure
//! (`EXTRACT_FAIL_TRAJ_SYSTEM_PROMPT`), or the comparative wording for a mixed
//! batch.
//!
//! # Reply formats
//!
//! Python's `MemoryItemParser` accepts only the markdown layout
//! (`# Memory Item i` / `## Title …` / `## Description …` / `## Content …`);
//! this port accepts that layout *and* a JSON array of
//! `{"title", "description", "content", "source_type"?}` objects, because
//! deployments that force JSON output still speak the same schema. Both paths
//! require all three of `title`, `description`, and `content`, matching Python's
//! "incomplete item" rule.
//!
//! # Deviations from Python
//!
//! * Python labels a trajectory with a separate LLM-as-judge call
//!   (`LabelDeterminator`). Here the label comes from
//!   [`TrajectoryFeedback`](super::TrajectoryFeedback), so the judge stays with
//!   the caller.
//! * Python wraps the parsed items in a `ReasoningBankMemory` (`query`, `label`)
//!   and stores the whole batch as one record. This port returns one
//!   [`SummaryDraft`] per item and carries the verdict as `source_type`
//!   (`success`/`failure`) when the parser can infer it.
//! * Python's parallel-scaling prompt (up to 5 items from many trajectories) is
//!   folded into the comparative system prompt.
//!
//! Python counterparts: `tests/unit_tests/extensions/context_evolver/` — see
//! `#[cfg(test)] mod tests` below.
use openjiuwen_common::error::{Error, Result};
use serde_json::Value;
use tracing::debug;

use super::{
    batch_outcome, fill, find_record, parse_json_reply, parse_ranked_ids, present,
    render_candidates, retrieve_phase, summary_phase, BatchOutcome, PromptPair, RetrievedHit,
    SummaryDraft, Trajectory, LOG_TARGET,
};
use crate::context_evolver::completion::strip_code_fence;
use crate::context_evolver::record::{MemoryDraft, MemoryRecord};

/// The `ReasoningBank` retrieval system prompt.
pub const RETRIEVE_SYSTEM: &str = r#"You are an expert at selecting the memory items that will be most useful for solving a task.
You will be given a user query and a numbered list of candidate memory items extracted from earlier attempts.
Rank the candidates by how useful they are for the query, considering:
- DIRECT RELEVANCE: how directly the memory applies to the query
- GENERALIZABILITY: whether the insight transfers beyond the exact wording of one task
- ACTIONABILITY: how specific and executable the guidance is
Output ONLY a JSON object: {"ranked_ids": ["<candidate id>", ...]} listing each candidate id at most once, most useful first."#;

/// The success system prompt (Python's `EXTRACT_SUCCESS_TRAJ_SYSTEM_PROMPT`).
pub const SUMMARIZE_SYSTEM: &str = r"You are an expert in generating reusable memories. You will be given a user query, the corresponding trajectory that represents how an agent successfully accomplished the task.
## Guidelines
You need to extract and summarize useful insights in the format of memory items based on the agent's successful trajectory.
The goal of summarized memory items is to be helpful and generalizable for future similar tasks.
## Important notes
- You must first think why the trajectory is successful, and then summarize the insights.
- You can extract at most 3 memory items from the trajectory.
- You must not repeat similar or overlapping items.
- Do not mention specific queries, or string contents, but rather focus on the generalizable insights.
## Output Format
Your output must strictly follow the Markdown format shown below:
```
# Memory Item i
## Title <the title of the memory item>
## Description <one sentence summary of the memory item>
## Content <1-3 sentences describing the insights learned to successfully accomplishing the task>
```";

/// The failure system prompt (Python's `EXTRACT_FAIL_TRAJ_SYSTEM_PROMPT`).
pub const SUMMARIZE_FAILURE_SYSTEM: &str = r"You are an expert in generating reusable memories. You will be given a user query, the corresponding trajectory that represents how an agent attempted to resolve the task but failed.
## Guidelines
You need to extract and summarize useful insights in the format of memory items based on the agent's failed trajectory.
The goal of summarized memory items is to be helpful and generalizable for future similar tasks.
## Important notes
- You must first reflect and think why the trajectory failed, and then summarize what lessons you have learned or strategies to prevent the failure in the future.
- You can extract at most 3 memory items from the trajectory.
- You must not repeat similar or overlapping items.
- Do not mention specific websites, queries, or string contents, but rather focus on the generalizable insights.
## Output Format
Your output must strictly follow the Markdown format shown below:
```
# Memory Item i
## Title <the title of the memory item>
## Description <one sentence summary of the memory item>
## Content <1-3 sentences describing the insights learned to successfully accomplishing the task>
```";

/// The mixed-batch system prompt (Python's `PARALLEL_SCALING_SYSTEM_PROMPT`).
pub const SUMMARIZE_COMPARATIVE_SYSTEM: &str = r"You are an expert in generating reusable memories. You will be given a user query and multiple trajectories showing how an agent attempted the task. Some trajectories may be successful, and others may have failed.
## Guidelines
Your goal is to compare and contrast these trajectories to identify the most useful and generalizable strategies as memory items.
Use self-contrast reasoning:
- Identify patterns and strategies that consistently led to success.
- Identify mistakes or inefficiencies from failed trajectories and formulate preventative strategies.
- Prefer strategies that generalize beyond specific pages or exact wording.
## Important notes
- Think first: Why did some trajectories succeed while others failed?
- You can extract at most 5 memory items from all trajectories combined.
- Do not repeat similar or overlapping items.
- Do not mention specific websites, queries, or string contents — focus on generalizable behaviors and reasoning patterns.
- Make sure each memory item captures actionable and transferable insights.
## Output Format
Your output must strictly follow the Markdown format shown below:
```
# Memory Item i
## Title <the title of the memory item>
## Description <one sentence summary of the memory item>
## Content <1-5 sentences describing the insights learned to successfully accomplishing the task>
```";

/// Builds the rerank prompt for the supplied candidates.
///
/// The candidate block is [`render_candidates`](super::render_candidates)
/// (`Candidate i:` with the record id, title, `When to use`, and content); the
/// reply asks for `{"ranked_ids": [...]}` so the parser can resolve ids directly.
#[must_use]
pub fn retrieve_prompt(query: &str, memories: &[MemoryRecord]) -> Option<PromptPair> {
    let candidates = render_candidates(memories);
    let candidates = if candidates.is_empty() {
        "(none)".to_string()
    } else {
        candidates
    };
    let user = fill(
        r#"Query: {query}

Candidate Memories:
{candidates}

Return the candidate ids ranked from most to least useful as JSON:
{"ranked_ids": ["<candidate id>", ...]}"#,
        &[("query", query.trim()), ("candidates", &candidates)],
    );
    Some(PromptPair::new(RETRIEVE_SYSTEM, user))
}

/// Builds the extraction prompt, selecting the system prompt by outcome.
///
/// All-helpful batches use [`SUMMARIZE_SYSTEM`], all-harmful batches use
/// [`SUMMARIZE_FAILURE_SYSTEM`], and anything else (mixed, all-neutral, empty)
/// uses [`SUMMARIZE_COMPARATIVE_SYSTEM`].
#[must_use]
pub fn summarize_prompt(trajectories: &[Trajectory], memories: &[MemoryRecord]) -> PromptPair {
    let system = match batch_outcome(trajectories) {
        BatchOutcome::Success => SUMMARIZE_SYSTEM,
        BatchOutcome::Failure => SUMMARIZE_FAILURE_SYSTEM,
        BatchOutcome::Comparative => SUMMARIZE_COMPARATIVE_SYSTEM,
    };
    let mut blocks = Vec::with_capacity(trajectories.len());
    for trajectory in trajectories {
        let query = trajectory.query.trim();
        let response = trajectory.response.trim();
        blocks.push(format!("Query: {query}\nTrajectory: {response}"));
    }
    let mut user = blocks.join("\n\n");
    let existing = existing_titles(memories);
    if !existing.is_empty() {
        user.push_str("\n\nDo not repeat these existing memory items:\n");
        user.push_str(&existing);
    }
    PromptPair::new(system, user)
}

/// Lists the titles (falling back to content) of the stored memories.
fn existing_titles(memories: &[MemoryRecord]) -> String {
    memories
        .iter()
        .filter_map(|record| {
            present(record.title.as_deref()).or_else(|| present(Some(record.content.as_str())))
        })
        .map(|title| format!("- {title}"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Parses a rerank reply into ordered hits.
///
/// # Errors
///
/// Returns [`Error::Validation`] naming the `reasoningbank retrieve` phase when
/// the reply is not JSON, carries no ranking array, or names no supplied memory.
pub fn parse_retrieve(reply: &str, memories: &[MemoryRecord]) -> Result<Vec<RetrievedHit>> {
    parse_ranked_ids(&retrieve_phase("reasoningbank"), reply, memories)
}

/// Parses an extraction reply into memory drafts.
///
/// Order of attempts:
///
/// 1. JSON — a bare array of memory objects, or an object holding an `items` or
///    `memory` array;
/// 2. the markdown layout Python's `MemoryItemParser` accepts.
///
/// # Errors
///
/// Returns [`Error::Validation`] naming the `reasoningbank summarize` phase when
/// the reply is neither shape or holds no complete item.
pub fn parse_summarize(reply: &str) -> Result<Vec<SummaryDraft>> {
    let phase = summary_phase("reasoningbank");
    if let Ok(value) = parse_json_reply(&phase, reply) {
        let drafts = json_items(&value).map(drafts_from_json).unwrap_or_default();
        if !drafts.is_empty() {
            return Ok(drafts);
        }
    }
    let drafts = drafts_from_markdown(reply);
    if drafts.is_empty() {
        return Err(Error::Validation(format!(
            "context_evolver {phase} pipeline: reply holds no complete memory item"
        )));
    }
    Ok(drafts)
}

/// Collects the memory-object array from a JSON reply.
fn json_items(value: &Value) -> Option<&[Value]> {
    match value {
        Value::Array(items) => Some(items.as_slice()),
        Value::Object(map) => ["items", "memory"]
            .iter()
            .find_map(|key| map.get(*key).and_then(Value::as_array))
            .map(Vec::as_slice),
        _ => None,
    }
}

/// Maps JSON memory objects onto drafts, skipping incomplete items.
fn drafts_from_json(items: &[Value]) -> Vec<SummaryDraft> {
    let mut drafts = Vec::with_capacity(items.len());
    for item in items {
        let Some(map) = item.as_object() else {
            debug!(target: LOG_TARGET, "reasoningbank memory item is not an object; skipping");
            continue;
        };
        let title = map
            .get("title")
            .and_then(Value::as_str)
            .and_then(|text| present(Some(text)));
        let description = map
            .get("description")
            .and_then(Value::as_str)
            .and_then(|text| present(Some(text)));
        let content = map
            .get("content")
            .and_then(Value::as_str)
            .and_then(|text| present(Some(text)));
        let (Some(title), Some(description), Some(content)) = (title, description, content) else {
            debug!(
                target: LOG_TARGET,
                "reasoningbank memory item is missing title, description, or content; skipping"
            );
            continue;
        };
        let draft = MemoryDraft {
            content: content.to_string(),
            title: Some(title.to_string()),
            description: Some(description.to_string()),
            source_type: map
                .get("source_type")
                .and_then(Value::as_str)
                .and_then(|text| present(Some(text)))
                .map(ToOwned::to_owned),
            ..MemoryDraft::default()
        };
        drafts.push(SummaryDraft::add(draft));
    }
    drafts
}

/// Parses Python's `# Memory Item i` markdown layout into drafts.
fn drafts_from_markdown(reply: &str) -> Vec<SummaryDraft> {
    let body = strip_code_fence(reply);
    let mut drafts = Vec::new();
    for section in markdown_sections(body) {
        let (title, description, content) = extract_fields(&section);
        let (Some(title), Some(description), Some(content)) = (title, description, content) else {
            debug!(
                target: LOG_TARGET,
                "reasoningbank markdown section is missing title, description, or content; skipping"
            );
            continue;
        };
        let draft = MemoryDraft {
            content,
            title: Some(title),
            description: Some(description),
            ..MemoryDraft::default()
        };
        drafts.push(SummaryDraft::add(draft));
    }
    drafts
}

/// Splits a markdown reply on its `# Memory Item i` headers.
fn markdown_sections(body: &str) -> Vec<String> {
    let mut sections = Vec::new();
    let mut current: Vec<&str> = Vec::new();
    for line in body.lines() {
        if is_memory_item_header(line) {
            let section = current.join("\n");
            if !section.trim().is_empty() {
                sections.push(section);
            }
            current.clear();
            continue;
        }
        current.push(line);
    }
    let section = current.join("\n");
    if !section.trim().is_empty() {
        sections.push(section);
    }
    sections
}

/// Reports whether a line opens a memory item, matching `#\s*Memory\s+Item\s+\d+`.
fn is_memory_item_header(line: &str) -> bool {
    let Some(rest) = line.trim_start().strip_prefix('#') else {
        return false;
    };
    let rest = rest.trim_start();
    let Some(rest) = strip_prefix_ignore_case(rest, "memory") else {
        return false;
    };
    let rest = rest.trim_start();
    let Some(rest) = strip_prefix_ignore_case(rest, "item") else {
        return false;
    };
    rest.trim_start()
        .chars()
        .next()
        .is_some_and(|character| character.is_ascii_digit())
}

/// Extracts the `Title`/`Description`/`Content` fields of one section.
fn extract_fields(section: &str) -> (Option<String>, Option<String>, Option<String>) {
    const FIELD_COUNT: usize = 3;
    let mut values: [Option<String>; FIELD_COUNT] = [None, None, None];
    let lines: Vec<&str> = section.lines().collect();
    let mut index = 0;
    while index < lines.len() {
        let Some((slot, rest)) = field_header(lines[index].trim()) else {
            index += 1;
            continue;
        };
        let mut value = rest.trim().to_string();
        index += 1;
        while index < lines.len() {
            let next = lines[index].trim();
            if next.is_empty() || next == "```" {
                index += 1;
                continue;
            }
            if next.starts_with("##") || is_memory_item_header(next) {
                break;
            }
            if !value.is_empty() {
                value.push(' ');
            }
            value.push_str(next);
            index += 1;
        }
        if let Some(slot) = values.get_mut(slot) {
            if slot.is_none() && !value.trim().is_empty() {
                *slot = Some(value.trim().to_string());
            }
        }
    }
    let [title, description, content] = values;
    (title, description, content)
}

/// Matches a `## Title …`/`## Description …`/`## Content …` header line.
///
/// Python's regexes demand at least one space after the field name; this port
/// also accepts a bare header and an optional `:` separator, so
/// `## Title`, `## Title: x`, and `## Title x` all work.
fn field_header(line: &str) -> Option<(usize, &str)> {
    let rest = line.strip_prefix("##")?.trim_start();
    for (slot, name) in ["title", "description", "content"].iter().enumerate() {
        let Some(after) = strip_prefix_ignore_case(rest, name) else {
            continue;
        };
        match after.chars().next() {
            None => return Some((slot, "")),
            Some(character) if character.is_whitespace() || character == ':' => {}
            Some(_) => continue,
        }
        let value = after.trim_start();
        let value = value.strip_prefix(':').unwrap_or(value);
        return Some((slot, value));
    }
    None
}

/// Strips an ASCII prefix case-insensitively without splitting a character.
fn strip_prefix_ignore_case<'a>(text: &'a str, prefix: &str) -> Option<&'a str> {
    if !text.is_char_boundary(prefix.len()) {
        return None;
    }
    let (head, tail) = text.split_at(prefix.len());
    head.eq_ignore_ascii_case(prefix).then_some(tail)
}

/// Renders the retained memories in Python's `ReasoningBank` format.
///
/// `Title:` / `Description:` / `Content:` per item, blocks separated by a blank
/// line. Hits are rendered in their own order; an id that names no supplied
/// record is skipped with a `tracing::debug!`.
#[must_use]
pub fn render_memory_string(hits: &[RetrievedHit], memories: &[MemoryRecord]) -> String {
    let mut blocks = Vec::with_capacity(hits.len());
    for hit in hits {
        let Some(record) = find_record(memories, &hit.id) else {
            debug!(
                target: LOG_TARGET,
                id = hit.id.as_str(),
                "reasoningbank hit names no supplied memory; skipping"
            );
            continue;
        };
        let mut lines = Vec::new();
        if let Some(title) = present(record.title.as_deref()) {
            lines.push(format!("Title: {title}"));
        }
        if let Some(description) = present(record.description.as_deref()) {
            lines.push(format!("Description: {description}"));
        }
        if let Some(content) = present(Some(record.content.as_str())) {
            lines.push(format!("Content: {content}"));
        }
        if !lines.is_empty() {
            blocks.push(lines.join("\n"));
        }
    }
    blocks.join("\n\n")
}

#[cfg(test)]
mod tests {
    // Python counterpart: `openjiuwen/extensions/context_evolver/summary/task/reasoning_bank/`
    // (`prompt.py`, `update.py::MemoryItemParser`) and `retrieve/task/reasoning_bank/run.py`.
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::float_cmp)]
    use super::*;
    use crate::context_evolver::prompt::{SummaryOperation, TrajectoryFeedback};
    use openjiuwen_common::error::ErrorKind;
    use serde_json::json;

    fn record(id: &str, title: &str, content: &str) -> MemoryRecord {
        MemoryRecord {
            id: id.to_string(),
            user_id: "user".into(),
            content: content.to_string(),
            title: Some(title.to_string()),
            description: Some(format!("{title} description")),
            ..MemoryRecord::default()
        }
    }

    fn trajectory(query: &str, response: &str, feedback: TrajectoryFeedback) -> Trajectory {
        Trajectory {
            query: query.into(),
            response: response.into(),
            feedback,
            ..Trajectory::default()
        }
    }

    const MARKDOWN: &str = r"# Memory Item 1
## Title Use pagination loops
## Description Iterate until the API returns no items.
## Content Use a while True loop over the page index instead of a fixed range.
# Memory Item 2
## Title Verify authentication first
## Description Read the passwords before calling an app.
## Content Call the supervisor password API instead of inventing a password.
";

    #[test]
    fn retrieve_prompt_ranks_candidates_and_names_the_query() {
        let memories = vec![record("rb-1", "Pagination", "loop while True")];
        let prompt = retrieve_prompt("count the playlists", &memories).unwrap();
        assert_eq!(prompt.system, RETRIEVE_SYSTEM);
        assert!(prompt.user.contains("count the playlists"));
        assert!(prompt.user.contains("Candidate 0:"));
        assert!(prompt.user.contains("id: rb-1"));
        assert!(prompt.user.contains("Title: Pagination"));
        assert!(prompt.user.contains("Content: loop while True"));
        assert!(prompt
            .user
            .contains(r#"{"ranked_ids": ["<candidate id>", ...]}"#));
    }

    #[test]
    fn retrieve_prompt_without_candidates_says_none() {
        let prompt = retrieve_prompt("query", &[]).unwrap();
        assert!(prompt.user.contains("(none)"));
    }

    #[test]
    fn parse_retrieve_reads_the_ranked_ids_reply() {
        let memories = vec![
            record("rb-1", "first", "one"),
            record("rb-2", "second", "two"),
        ];
        let hits = parse_retrieve(r#"{"ranked_ids": ["rb-2", "rb-1"]}"#, &memories).unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].id, "rb-2");
        assert!(hits[0].score > hits[1].score);
    }

    #[test]
    fn summarize_prompt_selects_the_system_prompt_by_outcome() {
        let success = vec![trajectory("q", "r", TrajectoryFeedback::Helpful)];
        let failure = vec![trajectory("q", "r", TrajectoryFeedback::Harmful)];
        let mixed = vec![
            trajectory("q", "r", TrajectoryFeedback::Helpful),
            trajectory("q", "r", TrajectoryFeedback::Harmful),
        ];
        assert_eq!(summarize_prompt(&success, &[]).system, SUMMARIZE_SYSTEM);
        assert_eq!(
            summarize_prompt(&failure, &[]).system,
            SUMMARIZE_FAILURE_SYSTEM
        );
        assert_eq!(
            summarize_prompt(&mixed, &[]).system,
            SUMMARIZE_COMPARATIVE_SYSTEM
        );
        assert_eq!(
            summarize_prompt(&[], &[]).system,
            SUMMARIZE_COMPARATIVE_SYSTEM
        );
    }

    #[test]
    fn summarize_prompt_carries_trajectories_and_existing_titles() {
        let trajectories = vec![trajectory(
            "count playlists",
            "used a fixed range loop",
            TrajectoryFeedback::Harmful,
        )];
        let memories = vec![record("rb-1", "Pagination", "loop while True")];
        let prompt = summarize_prompt(&trajectories, &memories);
        assert!(prompt.user.contains("Query: count playlists"));
        assert!(prompt.user.contains("Trajectory: used a fixed range loop"));
        assert!(prompt.user.contains("- Pagination"));
        assert!(prompt
            .user
            .contains("Do not repeat these existing memory items"));
    }

    #[test]
    fn parse_summarize_reads_the_markdown_layout() {
        let drafts = parse_summarize(MARKDOWN).unwrap();
        assert_eq!(drafts.len(), 2);
        assert_eq!(drafts[0].operation, SummaryOperation::Add);
        assert_eq!(
            drafts[0].draft.title.as_deref(),
            Some("Use pagination loops")
        );
        assert_eq!(
            drafts[0].draft.description.as_deref(),
            Some("Iterate until the API returns no items.")
        );
        assert_eq!(
            drafts[0].draft.content,
            "Use a while True loop over the page index instead of a fixed range."
        );
        assert_eq!(
            drafts[1].draft.title.as_deref(),
            Some("Verify authentication first")
        );
        assert_eq!(drafts[0].draft.section, "general");
    }

    #[test]
    fn parse_summarize_tolerates_fences_and_multiline_fields() {
        let reply = "```\n# Memory Item 1\n## Title: Paginate properly\n## Description: Avoid fixed ranges\n## Content: Use a while True loop\nover the page index.\n```";
        let drafts = parse_summarize(reply).unwrap();
        assert_eq!(drafts.len(), 1);
        assert_eq!(drafts[0].draft.title.as_deref(), Some("Paginate properly"));
        assert_eq!(
            drafts[0].draft.content,
            "Use a while True loop over the page index."
        );
    }

    #[test]
    fn parse_summarize_reads_the_json_layout() {
        let reply = json!([
            {
                "title": "Pagination",
                "description": "Avoid fixed ranges",
                "content": "Use a while True loop",
                "source_type": "failure"
            },
            {"title": "second", "description": "d", "content": "c"}
        ])
        .to_string();
        let drafts = parse_summarize(&reply).unwrap();
        assert_eq!(drafts.len(), 2);
        assert_eq!(drafts[0].draft.title.as_deref(), Some("Pagination"));
        assert_eq!(drafts[0].draft.source_type.as_deref(), Some("failure"));
        assert_eq!(drafts[1].draft.source_type, None);
    }

    #[test]
    fn parse_summarize_reads_fenced_json() {
        let reply =
            "```json\n[{\"title\": \"t\", \"description\": \"d\", \"content\": \"c\"}]\n```";
        let drafts = parse_summarize(reply).unwrap();
        assert_eq!(drafts.len(), 1);
        assert_eq!(drafts[0].draft.content, "c");
    }

    #[test]
    fn parse_summarize_skips_incomplete_items_and_reports_empty_replies() {
        let error = parse_summarize("## Title only a title\n").unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(
            error.message().contains("reasoningbank summarize"),
            "{}",
            error.message()
        );

        let error = parse_summarize("no memory here").unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.message().contains("no complete memory item"));
    }

    #[test]
    fn parse_summarize_falls_back_to_markdown_when_json_lacks_items() {
        let drafts = parse_summarize(MARKDOWN).unwrap();
        assert_eq!(drafts.len(), 2);
        let drafts = parse_summarize(&format!("{{}}\n{MARKDOWN}")).unwrap();
        assert_eq!(drafts.len(), 2);
    }

    #[test]
    fn render_memory_string_uses_the_python_format() {
        let memories = vec![
            record("rb-1", "First", "one"),
            record("rb-2", "Second", "two"),
        ];
        let hits = vec![
            RetrievedHit::new("rb-2", 1.0),
            RetrievedHit::new("missing", 0.5),
            RetrievedHit::new("rb-1", 0.4),
        ];
        assert_eq!(
            render_memory_string(&hits, &memories),
            "Title: Second\nDescription: Second description\nContent: two\n\n\
             Title: First\nDescription: First description\nContent: one"
        );
        assert_eq!(render_memory_string(&[], &memories), "");
    }

    #[test]
    fn render_memory_string_skips_blank_fields() {
        let memories = vec![MemoryRecord {
            id: "rb-3".into(),
            user_id: "user".into(),
            content: "only content".into(),
            ..MemoryRecord::default()
        }];
        let hits = vec![RetrievedHit::new("rb-3", 1.0)];
        assert_eq!(
            render_memory_string(&hits, &memories),
            "Content: only content"
        );
    }

    #[test]
    fn header_and_field_matching_is_case_insensitive() {
        let reply =
            "# memory item 3\n## tItLe Mixed Case\n## DESCRIPTION Upper\n## content Lower\n";
        let drafts = parse_summarize(reply).unwrap();
        assert_eq!(drafts.len(), 1);
        assert_eq!(drafts[0].draft.title.as_deref(), Some("Mixed Case"));
        assert_eq!(drafts[0].draft.description.as_deref(), Some("Upper"));
        assert_eq!(drafts[0].draft.content, "Lower");
    }
}
