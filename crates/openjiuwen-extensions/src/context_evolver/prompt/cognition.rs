//! Cognition prompt construction and reply parsing.
//!
//! Ports Python's
//! `openjiuwen/extensions/context_evolver/retrieve/task/cognition/{run.py,prompt.py}`
//! and `summary/task/cognition/{update.py,prompt.py}`.
//!
//! A cognition memory is a *description* plus a list of actionable rules
//! (`experience`), optionally classified into attribute keys. Python reaches that
//! shape in three model calls per phase: classify the query, rerank the
//! candidates, then classify the solution and write the cognition. This port
//! folds each phase into one prompt and documents both folds.
//!
//! # Deviations from Python
//!
//! * Retrieval fold: Python's `CLASSIFY_QUERY_TEMPLATES` and
//!   `RERANK_COGNITION_TEMPLATES` are one prompt here. It shows the query, the
//!   candidate memories with their attributes, and the distinct attribute keys,
//!   and asks for `{"attributes": {...}, "ranked_ids": [...]}`.
//!   [`parse_retrieve`] keeps only `ranked_ids`, so attribute predictions are
//!   advisory; a caller that wants them can parse the same reply itself.
//! * Summarize fold: Python's `CLASSIFY_SOLUTION_TEMPLATES` and
//!   `WRITE_COGNITION_TEMPLATES` are one prompt here, which is why the reply may
//!   carry an optional `attributes` object next to the required `description` and
//!   `experience`.
//! * Python emits one `CognitionMemory` per task with a list of experiences; this
//!   port emits one [`SummaryDraft`] per experience string so a persister can
//!   deduplicate them independently.
//! * Python's topology-schema discovery (`get_query_attributes`, shared key sets
//!   across users) is not part of this layer: the prompt receives the keys present
//!   in the supplied candidates.
//!
//! Python counterparts: `tests/unit_tests/extensions/context_evolver/` — see
//! `#[cfg(test)] mod tests` below.
use std::collections::{BTreeMap, BTreeSet};

use openjiuwen_common::error::{Error, Result};
use serde_json::{Map, Value};
use tracing::debug;

use super::{
    attribute_text, batch_outcome, fill, find_record, parse_json_reply, parse_ranked_ids, present,
    render_trajectories, retrieve_phase, summary_phase, BatchOutcome, PromptPair, RetrievedHit,
    SummaryDraft, Trajectory, LOG_TARGET,
};
use crate::context_evolver::record::{MemoryDraft, MemoryRecord};

/// The section every cognition draft is filed under.
pub const COGNITION_SECTION: &str = "cognition";

/// The folded classify-and-rerank retrieval system prompt.
pub const RETRIEVE_SYSTEM: &str = r#"You are a strict, highly accurate JSON classification and ranking engine.
You will be given a user query, a list of candidate past experiences with their attributes, and the attribute keys those experiences use.
### REQUIRED OUTPUT FORMAT
You MUST output ONLY a valid JSON object. Do NOT wrap the response in markdown blocks (e.g., no ```json).
The JSON object MUST contain exactly these two keys:
1. "attributes": an object that classifies the query. Use the attribute keys listed in the prompt as its keys. Prefer an existing value observed on a candidate; only invent a concise (1-2 word) value with underscores instead of spaces (e.g. "Time_Management") when nothing existing fits. Every key must be present; assign null when the key is irrelevant.
2. "ranked_ids": a list of candidate ids, most relevant first.
### STRICT RULES
1. CLASSIFY THE QUERY: base the attributes on the query itself, not on a candidate.
2. EXACT MATCH FOR RANKING: the ids in "ranked_ids" MUST be exact ids from the candidate list. Do NOT invent or hallucinate ids.
3. RELEVANCE CRITERIA: judge semantic similarity between the query and the candidate's description, content, and attributes.
4. LIMIT: include only candidates that are genuinely relevant; never more than the candidate count."#;

/// The cognition-writing system prompt (Python's `WRITE_COGNITION_TEMPLATES`).
pub const SUMMARIZE_SYSTEM: &str = r#"You are an expert AI behavior analyst and knowledge extractor.
Your task is to analyze an agent's recent task execution trajectory and extract high-value cognitive memory.
### REQUIRED OUTPUT FORMAT
You MUST output ONLY a valid JSON object. Do NOT wrap the response in markdown blocks (e.g., no ```json).
The JSON object MUST contain exactly these keys:
1. "description": A concise string (1-2 sentences) summarizing what this memory is about and what specific problem it solves.
2. "experience": A list of strings. Each string is a highly actionable rule, insight, or "lesson learned" derived from the trajectory. Emit at least one entry; never emit an empty list.
3. "attributes": An optional object that classifies the task. Keys are short category names (for example "domain", "tool", "other"); values are concise semantic tags with underscores instead of spaces (e.g. "API_Integration" instead of "API Integration").
### REFLECTION RULES
- If "Task Successful" is true: focus the "experience" list on WHY it succeeded. What specific commands, logic, or steps were effective? Extract reusable patterns.
- If "Task Successful" is false: focus the "experience" list on WHY it failed. What were the dead ends? What errors occurred? Formulate "Do NOT do X, instead try Y" style rules to prevent future agents from repeating the exact same mistakes.
- If "Task Successful" is null: focus the "experience" list on objective observations and verified mechanics. Since the final outcome is unverified or ambiguous, extract neutral facts about how the environment responded to specific actions, in the form "Action X reliably results in Observation Y" or "Tool Z requires parameters in format W".
- Be concrete. Do NOT say "Use the right tool". DO say "When querying the database, ensure the date format is YYYY-MM-DD to avoid syntax errors"."#;

/// Builds the folded classify-and-rerank prompt.
///
/// The candidate block carries each memory's description, content, and
/// attributes, followed by the distinct attribute keys the candidates use (the
/// schema the reply's `attributes` object is asked to fill).
#[must_use]
pub fn retrieve_prompt(query: &str, memories: &[MemoryRecord]) -> Option<PromptPair> {
    let candidates = render_attributes_candidates(memories);
    let attribute_keys = distinct_attribute_keys(memories);
    let total = memories.len().to_string();
    let user = fill(
        r#"# User Query
{query}
# Candidate Experiences (Total: {num_candidates})
{candidates}
# Attribute Keys Observed On The Candidates
{attribute_keys}
Return ONLY valid JSON with these two keys:
{"attributes": {"<attribute key>": "<value or null>"}, "ranked_ids": ["<candidate id>", ...]}"#,
        &[
            ("query", query.trim()),
            ("num_candidates", &total),
            ("candidates", &candidates),
            ("attribute_keys", &attribute_keys),
        ],
    );
    Some(PromptPair::new(RETRIEVE_SYSTEM, user))
}

/// Renders the candidates with their attributes.
fn render_attributes_candidates(memories: &[MemoryRecord]) -> String {
    if memories.is_empty() {
        return "(none)".to_string();
    }
    let mut blocks = Vec::with_capacity(memories.len());
    for (index, record) in memories.iter().enumerate() {
        let mut lines = vec![format!("Candidate {index}:")];
        if let Some(id) = present(Some(record.id.as_str())) {
            lines.push(format!("  id: {id}"));
        }
        if let Some(description) = present(record.description.as_deref()) {
            lines.push(format!("  Description: {description}"));
        }
        if let Some(content) = present(Some(record.content.as_str())) {
            lines.push(format!("  Content: {content}"));
        }
        if !record.attributes.is_empty() {
            let attributes: Map<String, Value> = record
                .attributes
                .iter()
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect();
            lines.push(format!("  Attributes: {}", Value::Object(attributes)));
        }
        blocks.push(lines.join("\n"));
    }
    blocks.join("\n\n")
}

/// Lists the distinct attribute keys found across the candidates.
fn distinct_attribute_keys(memories: &[MemoryRecord]) -> String {
    let keys: BTreeSet<&str> = memories
        .iter()
        .flat_map(|record| record.attributes.keys().map(String::as_str))
        .collect();
    if keys.is_empty() {
        return "(none)".to_string();
    }
    keys.into_iter().collect::<Vec<_>>().join(", ")
}

/// Builds the cognition-writing prompt.
///
/// The execution context carries the joined queries, the batch verdict
/// (`true`, `false`, or `null` when the batch is mixed or neutral), and the
/// rendered trajectories.
#[must_use]
pub fn summarize_prompt(trajectories: &[Trajectory], memories: &[MemoryRecord]) -> PromptPair {
    let query = joined_queries(trajectories);
    let is_correct = match batch_outcome(trajectories) {
        BatchOutcome::Success => "true",
        BatchOutcome::Failure => "false",
        BatchOutcome::Comparative => "null",
    };
    let trajectory = render_trajectories(trajectories);
    let trajectory = if trajectory.is_empty() {
        "(none)".to_string()
    } else {
        trajectory
    };
    let user = fill(
        r#"### EXECUTION CONTEXT
- User Query (Task): "{query}"
- Task Successful: {is_correct}
- Execution Trajectory (Steps and Observations):
{trajectory}"#,
        &[
            ("query", &query),
            ("is_correct", is_correct),
            ("trajectory", &trajectory),
        ],
    );
    let existing = existing_cognition(memories);
    let user = if existing.is_empty() {
        user
    } else {
        format!("{user}\n\nAvoid duplicating these existing cognition units:\n{existing}")
    };
    PromptPair::new(SUMMARIZE_SYSTEM, user)
}

/// Joins the distinct trajectory queries.
fn joined_queries(trajectories: &[Trajectory]) -> String {
    let mut queries: Vec<&str> = Vec::new();
    for trajectory in trajectories {
        if let Some(query) = present(Some(trajectory.query.as_str())) {
            if !queries.contains(&query) {
                queries.push(query);
            }
        }
    }
    if queries.is_empty() {
        "(not provided)".to_string()
    } else {
        queries.join("\n")
    }
}

/// Lists the stored cognition descriptions as `- <description>` lines.
fn existing_cognition(memories: &[MemoryRecord]) -> String {
    memories
        .iter()
        .filter_map(|record| present(record.description.as_deref()))
        .map(|description| format!("- {description}"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Parses the folded retrieval reply, keeping only the ranking.
///
/// The `attributes` object is ignored here: it is advisory classification, and
/// [`RetrievedHit`] carries no field for it.
///
/// # Errors
///
/// Returns [`Error::Validation`] naming the `cognition retrieve` phase when the
/// reply is not JSON, carries no ranking array, or names no supplied memory.
pub fn parse_retrieve(reply: &str, memories: &[MemoryRecord]) -> Result<Vec<RetrievedHit>> {
    parse_ranked_ids(&retrieve_phase("cognition"), reply, memories)
}

/// Parses a cognition reply into one draft per experience string.
///
/// The reply must be a JSON object with a non-empty `experience` array (Python
/// retries until it gets one). Each non-blank string becomes a
/// [`SummaryDraft::add`] whose content is the rule, whose description is the
/// reply's `description`, and whose section is [`COGNITION_SECTION`]; the optional
/// `attributes` object is copied onto every draft.
///
/// # Errors
///
/// Returns [`Error::Validation`] naming the `cognition summarize` phase when the
/// reply is not JSON, is not an object, or has a missing/empty `experience` list.
/// A list of only blank strings yields an empty result.
pub fn parse_summarize(reply: &str) -> Result<Vec<SummaryDraft>> {
    let phase = summary_phase("cognition");
    let value = parse_json_reply(&phase, reply)?;
    let Some(map) = value.as_object() else {
        return Err(Error::Validation(format!(
            "context_evolver {phase} pipeline: expected a JSON object with 'description' and \
             'experience'"
        )));
    };
    let description = map
        .get("description")
        .and_then(Value::as_str)
        .and_then(|text| present(Some(text)))
        .map(ToOwned::to_owned);
    let attributes = attributes_from(map);
    let experiences = map
        .get("experience")
        .and_then(Value::as_array)
        .filter(|items| !items.is_empty())
        .ok_or_else(|| {
            Error::Validation(format!(
                "context_evolver {phase} pipeline: 'experience' must be a non-empty JSON array"
            ))
        })?;
    let mut drafts = Vec::with_capacity(experiences.len());
    for experience in experiences {
        let Some(text) = experience.as_str() else {
            debug!(target: LOG_TARGET, "cognition experience entry is not a string; skipping");
            continue;
        };
        let Some(text) = present(Some(text)) else {
            debug!(target: LOG_TARGET, "cognition experience entry is blank; skipping");
            continue;
        };
        drafts.push(SummaryDraft::add(MemoryDraft {
            content: text.to_string(),
            description: description.clone(),
            section: COGNITION_SECTION.to_string(),
            attributes: attributes.clone(),
            ..MemoryDraft::default()
        }));
    }
    Ok(drafts)
}

/// Copies the reply's optional `attributes` object onto draft attributes.
fn attributes_from(map: &Map<String, Value>) -> BTreeMap<String, Value> {
    match map.get("attributes") {
        Some(Value::Object(attributes)) => attributes
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect(),
        _ => BTreeMap::new(),
    }
}

/// Renders the retained cognition units.
///
/// `Description:` and `Content:` lines per record, plus a comma-separated
/// `Attributes: key=value` line when the record carries attributes; blocks are
/// separated by a blank line. Hits are rendered in their own order; an id that
/// names no supplied record is skipped with a `tracing::debug!`.
#[must_use]
pub fn render_memory_string(hits: &[RetrievedHit], memories: &[MemoryRecord]) -> String {
    let mut blocks = Vec::with_capacity(hits.len());
    for hit in hits {
        let Some(record) = find_record(memories, &hit.id) else {
            debug!(
                target: LOG_TARGET,
                id = hit.id.as_str(),
                "cognition hit names no supplied memory; skipping"
            );
            continue;
        };
        let mut lines = Vec::new();
        if let Some(description) = present(record.description.as_deref()) {
            lines.push(format!("Description: {description}"));
        }
        if let Some(content) = present(Some(record.content.as_str())) {
            lines.push(format!("Content: {content}"));
        }
        if !record.attributes.is_empty() {
            let pairs = record
                .attributes
                .iter()
                .map(|(key, value)| format!("{key}={}", attribute_text(value)))
                .collect::<Vec<_>>()
                .join(", ");
            lines.push(format!("Attributes: {pairs}"));
        }
        if !lines.is_empty() {
            blocks.push(lines.join("\n"));
        }
    }
    blocks.join("\n\n")
}

#[cfg(test)]
mod tests {
    // Python counterpart: `openjiuwen/extensions/context_evolver/retrieve/task/cognition/`
    // (`run.py`, `prompt.py`) and `summary/task/cognition/` (`update.py`, `prompt.py`).
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::float_cmp)]
    use super::*;
    use crate::context_evolver::prompt::TrajectoryFeedback;
    use openjiuwen_common::error::ErrorKind;
    use serde_json::json;

    fn record(id: &str, description: &str, content: &str) -> MemoryRecord {
        MemoryRecord {
            id: id.to_string(),
            user_id: "user".into(),
            description: Some(description.to_string()),
            content: content.to_string(),
            section: COGNITION_SECTION.to_string(),
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
    fn retrieve_prompt_folds_classification_and_ranking() {
        let mut first = record("cog-1", "Pagination handling", "loop while True");
        first.attributes.insert("domain".into(), json!("web"));
        let mut second = record("cog-2", "Authentication", "read the password");
        second.attributes.insert("tool".into(), json!("supervisor"));
        second.attributes.insert("domain".into(), json!("api"));
        let prompt = retrieve_prompt("count playlists", &[first, second]).unwrap();

        assert_eq!(prompt.system, RETRIEVE_SYSTEM);
        assert!(prompt.user.contains("# User Query\ncount playlists"));
        assert!(prompt.user.contains("Total: 2"));
        assert!(prompt.user.contains("Candidate 0:"));
        assert!(prompt.user.contains("id: cog-1"));
        assert!(prompt.user.contains("Description: Pagination handling"));
        assert!(prompt.user.contains(r#"Attributes: {"domain":"web"}"#));
        assert!(prompt
            .user
            .contains("# Attribute Keys Observed On The Candidates\ndomain, tool"));
        assert!(prompt.user.contains(r#"{"attributes": {"<attribute key>": "<value or null>"}, "ranked_ids": ["<candidate id>", ...]}"#));
    }

    #[test]
    fn retrieve_prompt_without_candidates_lists_none() {
        let prompt = retrieve_prompt("query", &[]).unwrap();
        assert!(prompt.user.contains("(none)"));
        assert!(prompt.user.contains("Total: 0"));
    }

    #[test]
    fn parse_retrieve_keeps_the_ranking_and_ignores_attributes() {
        let memories = vec![
            record("cog-1", "first", "one"),
            record("cog-2", "second", "two"),
        ];
        let reply = json!({
            "attributes": {"domain": "web"},
            "ranked_ids": ["cog-2", "cog-1"]
        })
        .to_string();
        let hits = parse_retrieve(&reply, &memories).unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].id, "cog-2");
        assert!(hits[0].score > hits[1].score);

        let error = parse_retrieve("nope", &memories).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(
            error.message().contains("cognition retrieve"),
            "{}",
            error.message()
        );
    }

    #[test]
    fn summarize_prompt_carries_the_execution_context() {
        let trajectories = vec![trajectory(
            "count all playlists",
            "used a fixed range loop",
            TrajectoryFeedback::Harmful,
        )];
        let memories = vec![record("cog-1", "Pagination handling", "loop while True")];
        let prompt = summarize_prompt(&trajectories, &memories);
        assert_eq!(prompt.system, SUMMARIZE_SYSTEM);
        assert!(prompt
            .user
            .contains(r#"- User Query (Task): "count all playlists""#));
        assert!(prompt.user.contains("- Task Successful: false"));
        assert!(prompt.user.contains("used a fixed range loop"));
        assert!(prompt.user.contains("Feedback: harmful"));
        assert!(prompt.user.contains("- Pagination handling"));
    }

    #[test]
    fn summarize_prompt_reports_the_batch_verdict() {
        let success = vec![trajectory("q", "r", TrajectoryFeedback::Helpful)];
        let mixed = vec![
            trajectory("q", "r", TrajectoryFeedback::Helpful),
            trajectory("q", "r", TrajectoryFeedback::Harmful),
        ];
        assert!(summarize_prompt(&success, &[])
            .user
            .contains("- Task Successful: true"));
        assert!(summarize_prompt(&mixed, &[])
            .user
            .contains("- Task Successful: null"));
        assert!(summarize_prompt(&[], &[]).user.contains("(not provided)"));
        assert!(summarize_prompt(&[], &[]).user.contains("(none)"));
    }

    #[test]
    fn parse_summarize_emits_one_draft_per_experience() {
        let reply = json!({
            "description": "Handling paginated APIs",
            "experience": [
                "Use a while True loop over the page index.",
                "  ",
                "Stop when the API returns an empty page."
            ],
            "attributes": {"domain": "web", "tool": "http"}
        })
        .to_string();
        let drafts = parse_summarize(&reply).unwrap();
        assert_eq!(drafts.len(), 2);
        assert_eq!(
            drafts[0].draft.content,
            "Use a while True loop over the page index."
        );
        assert_eq!(
            drafts[0].draft.description.as_deref(),
            Some("Handling paginated APIs")
        );
        assert_eq!(drafts[0].draft.section, COGNITION_SECTION);
        assert_eq!(
            drafts[0].draft.attributes.get("domain"),
            Some(&json!("web"))
        );
        assert_eq!(drafts[0].draft.attributes.get("tool"), Some(&json!("http")));
        assert_eq!(
            drafts[1].draft.attributes.get("domain"),
            Some(&json!("web"))
        );
        assert_eq!(drafts[0].operation, super::super::SummaryOperation::Add);
    }

    #[test]
    fn parse_summarize_accepts_fenced_json_without_attributes() {
        let reply = "```json\n{\"description\": \"d\", \"experience\": [\"rule\"]}\n```";
        let drafts = parse_summarize(reply).unwrap();
        assert_eq!(drafts.len(), 1);
        assert!(drafts[0].draft.attributes.is_empty());
        assert_eq!(drafts[0].draft.description.as_deref(), Some("d"));
    }

    #[test]
    fn parse_summarize_rejects_a_missing_or_empty_experience_list() {
        for reply in [
            r#"{"description": "d"}"#,
            r#"{"description": "d", "experience": []}"#,
            r#"{"description": "d", "experience": "rule"}"#,
        ] {
            let error = parse_summarize(reply).unwrap_err();
            assert_eq!(error.kind(), ErrorKind::Validation, "reply: {reply}");
            assert!(
                error.message().contains("cognition summarize"),
                "reply: {reply} message: {}",
                error.message()
            );
        }
        let error = parse_summarize(r#"["not", "an", "object"]"#).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.message().contains("cognition summarize"));
    }

    #[test]
    fn parse_summarize_reports_malformed_json_with_the_phase() {
        let error = parse_summarize("I could not write anything.").unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(
            error.message().contains("cognition summarize"),
            "{}",
            error.message()
        );
    }

    #[test]
    fn parse_summarize_returns_empty_when_every_experience_is_blank() {
        assert_eq!(
            parse_summarize(r#"{"experience": ["", "  "]}"#)
                .unwrap()
                .len(),
            0
        );
    }

    #[test]
    fn render_memory_string_lists_description_content_and_attributes() {
        let mut first = record("cog-1", "Pagination handling", "loop while True");
        first.attributes.insert("domain".into(), json!("web"));
        let second = record("cog-2", "Authentication", "read the password");
        let memories = vec![first, second];
        let hits = vec![
            RetrievedHit::new("cog-2", 1.0),
            RetrievedHit::new("missing", 0.5),
            RetrievedHit::new("cog-1", 0.4),
        ];
        assert_eq!(
            render_memory_string(&hits, &memories),
            "Description: Authentication\nContent: read the password\n\n\
             Description: Pagination handling\nContent: loop while True\nAttributes: domain=web"
        );
        assert_eq!(render_memory_string(&[], &memories), "");
    }
}
