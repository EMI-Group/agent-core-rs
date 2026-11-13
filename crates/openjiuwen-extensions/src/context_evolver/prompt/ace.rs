//! ACE prompt construction and reply parsing.
//!
//! Ports the model-facing half of Python's
//! `openjiuwen/extensions/context_evolver/summary/task/ace/` (`prompt.py`'s
//! `ACE_CURATOR_PROMPT`, `playbook.py`'s `DeltaOperation`/`DeltaBatch` JSON and
//! bullet rendering) and `retrieve/task/ace/run.py`.
//!
//! ACE differs from every other family in one structural way: retrieval is
//! unranked. Python's `RecallMemoryOp` loads the whole playbook (up to 50
//! bullets, filtered by workspace metadata) and never asks the model to rank, so
//! [`retrieve_prompt`] returns `None` and [`parse_retrieve`] returns every
//! supplied record in input order with a score of `1.0`.
//!
//! The summarize side is a curator: the model sees the current playbook and the
//! latest attempts, then emits a delta of `ADD`/`UPDATE`/`TAG`/`REMOVE`
//! operations ([`SummaryOperation`]) instead of a regenerated playbook.
//!
//! # Deviations from Python
//!
//! * Python keeps the playbook in memory as `Playbook` with per-section id
//!   counters (`section-0001`). This port emits and parses the same operation
//!   JSON but leaves id allocation and section bookkeeping to the persister.
//! * Python's `Bullet` tracks a `neutral` counter that [`MemoryRecord`] does not
//!   have; the renderer reads a `neutral` attribute when present and prints `0`
//!   otherwise.
//! * Per-operation `metadata` is projected into [`MemoryDraft::attributes`] for
//!   `ADD`/`UPDATE`/`TAG`, which is where Python applies it too
//!   (`Playbook.add_bullet`/`apply_metadata`).
//! * Python builds the prompt from a separate reflector call plus ground truth
//!   and test-report inputs; this port folds the reflection into the trajectory
//!   block and never requests ground truth, because the caller owns it.
//!
//! Python counterparts: `tests/unit_tests/extensions/context_evolver/`
//! (summary/retrieve flow tests) — see `#[cfg(test)] mod tests` below.
use std::collections::BTreeMap;

use openjiuwen_common::error::{Error, Result};
use serde_json::{Map, Value};
use tracing::debug;

use super::{
    fill, find_record, parse_json_reply, present, render_trajectories, summary_phase, PromptPair,
    RetrievedHit, SummaryDraft, SummaryOperation, Trajectory, LOG_TARGET,
};
use crate::context_evolver::record::{MemoryDraft, MemoryRecord};

/// The ACE curator system prompt (Python's `ACE_CURATOR_PROMPT`).
///
/// Condensed from Python: the worked examples are dropped, every instruction
/// that determines the reply shape is kept (delta-only output, the 50-bullet
/// ceiling, the operation table, and the exact JSON envelope).
pub const SUMMARIZE_SYSTEM: &str = r#"You are a master curator of knowledge. Your job is to identify what new insights should be added to an existing playbook based on a reflection from a previous attempt.
Context:
- The playbook you created will be used to help answering similar questions.
- The reflection is generated using ground truth answers that will NOT be available when the playbook is being used. So you need to come up with content that can aid the playbook user to create predictions that likely align with ground truth.
Instructions:
- Review the existing playbook and the reflection from the previous attempt.
- ADD ONLY the NEW insights, strategies, or mistakes that are MISSING from the current playbook.
- Avoid redundancy: if similar advice already exists, only add content that is a perfect complement to the existing playbook.
- The MAXIMUM number of insights in the playbook is 50. If the playbook is full, consider REMOVE or UPDATE of existing insight.
- Do NOT regenerate the entire playbook, only provide the additions needed or update an existing insight with a refined one.
- Focus on quality over quantity; a focused well-organized playbook is better than an exhaustive one. You can REMOVE redundant or unnecessary insight.
- Be concise and specific: each operation should be actionable.
- For coding tasks, explicitly curate from the reflections the output format/schema of APIs used when unclear or mismatched with expectations (e.g., apis.blah.show_contents() returns a list of content_ids (strings), not content objects).
Your Task: output ONLY a valid JSON object with these exact fields:
- reasoning: your chain of thought / reasoning / thinking process, detailed analysis and calculations
- operations: a list of operations to be performed on the playbook
Available Operations:
1. ADD: create new bullet points with fresh IDs.
   - section: the section to add the new bullet to.
   - content: the new content of the bullet. No need to include the bullet_id in the content like '[ctx-00263] helpful=1 harmful=0 ::'; the bullet_id is added by the system.
2. UPDATE: modify the content or metadata of an existing bullet.
   - bullet_id: (string, required) the ID of the bullet to modify.
   - content: (string, optional) the new text content for the bullet.
   - metadata: (dict, optional) a dictionary of new metadata tags to apply.
3. TAG: add or increment a numerical metadata tag on an existing bullet.
   - bullet_id: (string, required) the ID of the bullet to tag.
   - metadata: (dict, optional) a dictionary of metadata tags to increment (e.g., {"helpful": 1, "harmful": 0}).
4. REMOVE: delete an existing bullet point.
   - bullet_id: (string, required) the ID of the bullet to delete.
RESPONSE FORMAT - output ONLY this JSON structure (no markdown, no code blocks):
{
  "reasoning": "<how you decided on the updates>",
  "operations": [
    {
      "type": "ADD|UPDATE|TAG|REMOVE",
      "section": "<section name>",
      "content": "<insight/strategy/mistake>",
      "bullet_id": "<optional existing id>",
      "metadata": {"helpful": 1, "harmful": 0}
    }
  ]
}
If no updates are required, return an empty list for "operations"."#;

/// Renders one playbook bullet the way Python's presentation helpers do.
///
/// `[<id>] helpful=<h> harmful=<hh> neutral=<n>` on the first line, then
/// `Section:` and `Content:` lines when the record carries them.
#[must_use]
pub fn render_bullet(record: &MemoryRecord) -> String {
    let id = present(Some(record.id.as_str())).unwrap_or("<unassigned>");
    let helpful = record.helpful;
    let harmful = record.harmful;
    let neutral = neutral_count(record);
    let mut lines = vec![format!(
        "[{id}] helpful={helpful} harmful={harmful} neutral={neutral}"
    )];
    if let Some(section) = present(Some(record.section.as_str())) {
        lines.push(format!("Section: {section}"));
    }
    if let Some(content) = present(Some(record.content.as_str())) {
        lines.push(format!("Content: {content}"));
    }
    lines.join("\n")
}

/// Renders the whole playbook, one bullet block per record.
///
/// This is the `Current Playbook` section of the curator prompt; an empty input
/// renders as `"(empty)"` so the model still sees a section.
#[must_use]
pub fn render_playbook(memories: &[MemoryRecord]) -> String {
    if memories.is_empty() {
        return "(empty)".to_string();
    }
    memories
        .iter()
        .map(render_bullet)
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// ACE retrieval is unranked, so there is no retrieve prompt.
///
/// Python's `RecallMemoryOp` loads the whole playbook without a model call; the
/// caller injects those bullets directly (see [`parse_retrieve`]).
#[must_use]
pub fn retrieve_prompt(_query: &str, _memories: &[MemoryRecord]) -> Option<PromptPair> {
    None
}

/// Builds the curator prompt from the latest attempts and the current playbook.
#[must_use]
pub fn summarize_prompt(trajectories: &[Trajectory], memories: &[MemoryRecord]) -> PromptPair {
    let task_context = task_context(trajectories);
    let playbook = render_playbook(memories);
    let attempts = render_trajectories(trajectories);
    let user = fill(
        r"Task Context (the actual task instruction):
{question_context}

Current Playbook:
{playbook}

Attempts and Reflections (latest attempt with reasoning, planning and feedback):
{trajectories}",
        &[
            ("question_context", &task_context),
            ("playbook", &playbook),
            ("trajectories", &attempts),
        ],
    );
    PromptPair::new(SUMMARIZE_SYSTEM, user)
}

/// Joins the distinct trajectory queries into the task-context block.
fn task_context(trajectories: &[Trajectory]) -> String {
    let mut queries: Vec<&str> = Vec::new();
    for trajectory in trajectories {
        if let Some(query) = present(Some(trajectory.query.as_str())) {
            if !queries.contains(&query) {
                queries.push(query);
            }
        }
    }
    if queries.is_empty() {
        return "(not provided)".to_string();
    }
    queries.join("\n")
}

/// Returns every supplied record as a hit, in input order.
///
/// Ports `retrieve/task/ace/run.py`: ACE retrieval performs no ranking, so the
/// reply is ignored and every hit scores `1.0`. A record is returned even when a
/// caller supplies an empty or unrelated reply.
///
/// # Errors
///
/// Never fails; the signature matches the sibling families so the dispatch can
/// stay uniform.
pub fn parse_retrieve(_reply: &str, memories: &[MemoryRecord]) -> Result<Vec<RetrievedHit>> {
    Ok(memories
        .iter()
        .map(|record| RetrievedHit::new(record.id.clone(), 1.0))
        .collect())
}

/// Parses a curator reply into playbook operations.
///
/// Reads `{"operations": [...]}` (a bare array of operations is also accepted) and
/// maps each entry onto a [`SummaryDraft`]:
///
/// | type     | result                                                     |
/// |----------|------------------------------------------------------------|
/// | `ADD`    | [`SummaryDraft::add`] with the operation's section         |
/// | `UPDATE` | `ADD` shape plus `.with_id(bullet_id)`                     |
/// | `TAG`    | [`SummaryOperation::Tag`] plus the metadata attributes     |
/// | `REMOVE` | [`SummaryOperation::Remove`] with an empty draft           |
///
/// An operation of unknown type, or one missing the content/id it needs, is
/// skipped with a `tracing::debug!`; a missing `operations` field is an empty
/// delta, exactly like Python's empty `DeltaBatch`.
///
/// # Errors
///
/// Returns [`Error::Validation`] from [`parse_json_reply`] when the reply is not
/// JSON, and when `operations` is present but not an array.
pub fn parse_summarize(reply: &str) -> Result<Vec<SummaryDraft>> {
    let phase = summary_phase("ace");
    let value = parse_json_reply(&phase, reply)?;
    let operations: &[Value] = match &value {
        Value::Array(items) => items.as_slice(),
        Value::Object(map) => match map.get("operations") {
            None | Some(Value::Null) => return Ok(Vec::new()),
            Some(Value::Array(items)) => items.as_slice(),
            Some(_) => {
                return Err(Error::Validation(format!(
                    "context_evolver {phase} pipeline: 'operations' must be a JSON array"
                )))
            }
        },
        _ => {
            return Err(Error::Validation(format!(
            "context_evolver {phase} pipeline: expected a JSON object with an 'operations' array"
        )))
        }
    };
    let mut drafts = Vec::with_capacity(operations.len());
    for operation in operations {
        if let Some(draft) = parse_operation(operation) {
            drafts.push(draft);
        }
    }
    Ok(drafts)
}

/// Maps one `DeltaOperation` onto a [`SummaryDraft`].
fn parse_operation(operation: &Value) -> Option<SummaryDraft> {
    let Some(map) = operation.as_object() else {
        debug!(target: LOG_TARGET, "ace operation is not a JSON object; skipping");
        return None;
    };
    let kind = map
        .get("type")
        .and_then(Value::as_str)
        .map(|kind| kind.trim().to_ascii_uppercase());
    let Some(kind) = kind else {
        debug!(target: LOG_TARGET, "ace operation has no string 'type'; skipping");
        return None;
    };
    let bullet_id = map
        .get("bullet_id")
        .and_then(Value::as_str)
        .and_then(|id| present(Some(id)))
        .map(ToOwned::to_owned);
    let content = map
        .get("content")
        .and_then(Value::as_str)
        .and_then(|text| present(Some(text)))
        .map(ToOwned::to_owned);
    let section = map
        .get("section")
        .and_then(Value::as_str)
        .and_then(|name| present(Some(name)))
        .map(ToOwned::to_owned);
    let attributes = metadata_attributes(map);
    match kind.as_str() {
        "ADD" => {
            let Some(content) = content else {
                debug!(target: LOG_TARGET, "ace ADD without content; skipping");
                return None;
            };
            let mut draft = MemoryDraft::new(content);
            if let Some(section) = section {
                draft.section = section;
            }
            draft.attributes = attributes;
            Some(SummaryDraft::add(draft))
        }
        "UPDATE" => {
            let (Some(id), Some(content)) = (bullet_id, content) else {
                debug!(
                    target: LOG_TARGET,
                    "ace UPDATE without bullet_id or content; skipping"
                );
                return None;
            };
            let mut draft = MemoryDraft::new(content);
            if let Some(section) = section {
                draft.section = section;
            }
            draft.attributes = attributes;
            Some(SummaryDraft::add(draft).with_id(id))
        }
        "TAG" => {
            let Some(id) = bullet_id else {
                debug!(target: LOG_TARGET, "ace TAG without bullet_id; skipping");
                return None;
            };
            let draft = MemoryDraft {
                attributes,
                ..MemoryDraft::default()
            };
            Some(SummaryDraft {
                id: Some(id),
                operation: SummaryOperation::Tag,
                draft,
            })
        }
        "REMOVE" => {
            let Some(id) = bullet_id else {
                debug!(target: LOG_TARGET, "ace REMOVE without bullet_id; skipping");
                return None;
            };
            Some(SummaryDraft {
                id: Some(id),
                operation: SummaryOperation::Remove,
                draft: MemoryDraft::default(),
            })
        }
        other => {
            debug!(target: LOG_TARGET, operation = other, "unknown ace operation type; skipping");
            None
        }
    }
}

/// Projects an operation's `metadata` counters onto draft attributes.
fn metadata_attributes(map: &Map<String, Value>) -> BTreeMap<String, Value> {
    let mut attributes = BTreeMap::new();
    let Some(Value::Object(metadata)) = map.get("metadata") else {
        return attributes;
    };
    for (key, value) in metadata {
        if value.is_number() {
            attributes.insert(key.clone(), value.clone());
        } else {
            debug!(target: LOG_TARGET, key, "ace metadata value is not a number; skipping");
        }
    }
    attributes
}

/// Reads the playbook bullet's `neutral` counter from its attributes.
fn neutral_count(record: &MemoryRecord) -> u64 {
    record
        .attributes
        .get("neutral")
        .and_then(Value::as_u64)
        .unwrap_or(0)
}

/// Renders the retained bullets in the Python ACE playbook format.
///
/// Python counterparts: `summary/task/ace/playbook.py::as_prompt` — one
/// `[id] helpful/harmful/neutral` line, then `Section:` and `Content:` per
/// bullet, blocks separated by a blank line. Hits are rendered in their own
/// order; an id that names no supplied record is skipped with a `tracing::debug!`.
#[must_use]
pub fn render_memory_string(hits: &[RetrievedHit], memories: &[MemoryRecord]) -> String {
    let mut blocks = Vec::with_capacity(hits.len());
    for hit in hits {
        if let Some(record) = find_record(memories, &hit.id) {
            blocks.push(render_bullet(record));
        } else {
            debug!(
                target: LOG_TARGET,
                id = hit.id.as_str(),
                "ace hit names no supplied memory; skipping"
            );
        }
    }
    blocks.join("\n\n")
}

#[cfg(test)]
mod tests {
    // Python counterpart: `openjiuwen/extensions/context_evolver/summary/task/ace/`
    // (`prompt.py`, `playbook.py`, `update.py`) and `retrieve/task/ace/run.py`.
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::float_cmp)]
    use super::*;
    use crate::context_evolver::prompt::{Trajectory, TrajectoryFeedback};
    use openjiuwen_common::error::ErrorKind;
    use serde_json::json;

    fn record(id: &str, section: &str, content: &str) -> MemoryRecord {
        MemoryRecord {
            id: id.to_string(),
            user_id: "user".into(),
            content: content.to_string(),
            section: section.to_string(),
            helpful: 2,
            harmful: 1,
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

    #[test]
    fn retrieval_is_unranked_like_python() {
        let memories = vec![
            record("a-1", "general", "first"),
            record("a-2", "api", "second"),
        ];
        assert_eq!(retrieve_prompt("query", &memories), None);
        let hits = parse_retrieve("not even json", &memories).unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].id, "a-1");
        assert_eq!(hits[1].id, "a-2");
        assert!(hits.iter().all(|hit| hit.score == 1.0));
        assert_eq!(parse_retrieve("", &[]).unwrap().len(), 0);
    }

    #[test]
    fn summarize_prompt_carries_context_playbook_and_attempts() {
        let memories = vec![record(
            "general-0001",
            "strategies",
            "always authenticate first",
        )];
        let trajectories = vec![trajectory(
            "count all playlists",
            "used a fixed range loop",
            TrajectoryFeedback::Harmful,
        )];
        let prompt = summarize_prompt(&trajectories, &memories);
        assert_eq!(prompt.system, SUMMARIZE_SYSTEM);
        assert!(prompt.user.contains("count all playlists"));
        assert!(prompt
            .user
            .contains("[general-0001] helpful=2 harmful=1 neutral=0"));
        assert!(prompt.user.contains("Section: strategies"));
        assert!(prompt.user.contains("Content: always authenticate first"));
        assert!(prompt.user.contains("used a fixed range loop"));
        assert!(prompt.user.contains("Feedback: harmful"));
        assert!(prompt.user.contains("Current Playbook:"));
        assert!(prompt
            .system
            .contains("MAXIMUM number of insights in the playbook is 50"));
        assert!(prompt.system.contains("ADD ONLY the NEW insights"));
    }

    #[test]
    fn summarize_prompt_without_memories_or_trajectories_stays_well_formed() {
        let prompt = summarize_prompt(&[], &[]);
        assert!(prompt.user.contains("(not provided)"));
        assert!(prompt.user.contains("(empty)"));
    }

    #[test]
    fn parse_summarize_maps_all_four_operation_types() {
        let reply = json!({
            "reasoning": "add, update, tag and remove",
            "operations": [
                {"type": "ADD", "section": "api_usage", "content": "paginate with while True"},
                {"type": "UPDATE", "bullet_id": "api_usage-0001", "content": "refined pagination rule"},
                {"type": "TAG", "bullet_id": "api_usage-0001", "metadata": {"helpful": 1, "harmful": 0}},
                {"type": "REMOVE", "bullet_id": "api_usage-0002"}
            ]
        })
        .to_string();
        let drafts = parse_summarize(&reply).unwrap();
        assert_eq!(drafts.len(), 4);

        assert_eq!(drafts[0].operation, SummaryOperation::Add);
        assert_eq!(drafts[0].id, None);
        assert_eq!(drafts[0].draft.content, "paginate with while True");
        assert_eq!(drafts[0].draft.section, "api_usage");

        assert_eq!(drafts[1].operation, SummaryOperation::Add);
        assert_eq!(drafts[1].id.as_deref(), Some("api_usage-0001"));
        assert_eq!(drafts[1].draft.content, "refined pagination rule");

        assert_eq!(drafts[2].operation, SummaryOperation::Tag);
        assert_eq!(drafts[2].id.as_deref(), Some("api_usage-0001"));
        assert_eq!(drafts[2].draft.attributes.get("helpful"), Some(&json!(1)));
        assert_eq!(drafts[2].draft.attributes.get("harmful"), Some(&json!(0)));
        assert_eq!(drafts[2].draft.content.len(), 0);

        assert_eq!(drafts[3].operation, SummaryOperation::Remove);
        assert_eq!(drafts[3].id.as_deref(), Some("api_usage-0002"));
        assert_eq!(drafts[3].draft.content.len(), 0);
    }

    #[test]
    fn parse_summarize_accepts_fenced_json_and_lowercase_types() {
        let reply = "Here is the delta:\n```json\n{\"operations\": [{\"type\": \"add\", \"section\": \"general\", \"content\": \"be careful\"}]}\n```";
        let drafts = parse_summarize(reply).unwrap();
        assert_eq!(drafts.len(), 1);
        assert_eq!(drafts[0].draft.content, "be careful");
    }

    #[test]
    fn parse_summarize_accepts_a_bare_operation_array() {
        let drafts =
            parse_summarize(r#"[{"type": "ADD", "section": "general", "content": "x"}]"#).unwrap();
        assert_eq!(drafts.len(), 1);
        assert_eq!(drafts[0].draft.content, "x");
    }

    #[test]
    fn parse_summarize_without_operations_is_an_empty_delta() {
        assert_eq!(
            parse_summarize(r#"{"reasoning": "nothing new"}"#)
                .unwrap()
                .len(),
            0
        );
        assert_eq!(parse_summarize(r#"{"operations": []}"#).unwrap().len(), 0);
        assert_eq!(parse_summarize(r#"{"operations": null}"#).unwrap().len(), 0);
    }

    #[test]
    fn parse_summarize_skips_incomplete_and_unknown_operations() {
        let reply = json!({
            "operations": [
                {"type": "ADD"},
                {"type": "UPDATE", "bullet_id": "b-1"},
                {"type": "TAG"},
                {"type": "REMOVE"},
                {"type": "RENAME", "bullet_id": "b-1"},
                {"no_type": true},
                "not an object"
            ]
        })
        .to_string();
        assert_eq!(parse_summarize(&reply).unwrap().len(), 0);
    }

    #[test]
    fn parse_summarize_rejects_malformed_replies_naming_the_phase() {
        let error = parse_summarize("I could not decide.").unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(
            error.message().contains("ace summarize"),
            "{}",
            error.message()
        );

        let error = parse_summarize(r#"{"operations": "ADD"}"#).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(
            error.message().contains("ace summarize"),
            "{}",
            error.message()
        );
    }

    #[test]
    fn render_memory_string_uses_the_python_playbook_format() {
        let memories = vec![
            record("a-1", "general", "first"),
            record("a-2", "api", "second"),
        ];
        let hits = vec![
            RetrievedHit::new("a-2", 1.0),
            RetrievedHit::new("missing", 0.5),
            RetrievedHit::new("a-1", 0.4),
        ];
        let rendered = render_memory_string(&hits, &memories);
        assert_eq!(
            rendered,
            "[a-2] helpful=2 harmful=1 neutral=0\nSection: api\nContent: second\n\n\
             [a-1] helpful=2 harmful=1 neutral=0\nSection: general\nContent: first"
        );
        assert_eq!(render_memory_string(&[], &memories), "");
    }

    #[test]
    fn render_bullet_reads_the_neutral_counter_from_attributes() {
        let mut record = record("a-1", "general", "content");
        record.attributes.insert("neutral".into(), json!(3));
        assert!(render_bullet(&record).starts_with("[a-1] helpful=2 harmful=1 neutral=3"));
        assert_eq!(render_playbook(&[]), "(empty)");
    }
}
