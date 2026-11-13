//! `ReMe` prompt construction and reply parsing.
//!
//! Ports Python's `openjiuwen/extensions/context_evolver/summary/task/reme/{prompt.py,update.py}`
//! and `retrieve/task/reme/{prompt.py,run.py,utils.py}`. This module serves the
//! whole `ReMe` family: [`Algo::Reme`], [`Algo::Refcon`], and [`Algo::Divcon`].
//!
//! Retrieval is Python's `RerankMemoryOp`: the embedding search stays with the
//! caller, and the model reranks the candidates it is handed.
//! [`retrieve_prompt`] therefore asks for `{"ranked_indices": [...]}` over the
//! numbered `Condition:`/`Experience:` blocks and
//! [`parse_retrieve`](super::parse_ranked_ids) resolves the indices.
//!
//! Summarization extracts step-level experiences. The three Python extraction
//! prompts are selected by the trajectory batch (success, failure, comparative)
//! exactly like the `ReasoningBank` family, and the reply is the JSON array
//! Python emits:
//! `{"when_to_use", "experience", "tags", "confidence", "step_type", "tools_used"}`.
//!
//! # Per-algorithm accents
//!
//! `RefCon` and `DivCon` share the `ReMe` extraction prompts but add their own
//! instruction block. [`summarize_prompt_for`] takes the [`Algo`] so the caller's
//! choice reaches the prompt; [`summarize_prompt`] is the `ReMe` default and
//! adds no accent.
//!
//! # Deviations from Python
//!
//! * Python's `MEMORY_VALIDATION_PROMPT` (a second LLM round-trip that scores an
//!   extracted memory) is folded into the extraction guidelines instead of being
//!   a separate call, because a caller that wants the gate can run it against the
//!   parsed drafts.
//! * Python's `MEMORY_REWRITE_PROMPT` rewrites the retained memories into a
//!   cohesive context; this port renders them, leaving rewriting to the caller.
//! * Python accepts either `when_to_use` or `condition` as the trigger, and a
//!   bare object reply instead of an array; both tolerances are kept.
//! * Python silently returns an empty list when the reply is not parseable; this
//!   port reports [`Error::Validation`] so a caller cannot mistake a broken model
//!   for "no new memories". An explicit empty array still yields an empty result.
//!
//! Python counterparts: `tests/unit_tests/extensions/context_evolver/` — see
//! `#[cfg(test)] mod tests` below.
use std::collections::BTreeMap;

use openjiuwen_common::error::{Error, Result};
use serde_json::{Map, Value};
use tracing::debug;

use super::{
    batch_outcome, fill, find_record, parse_json_reply, parse_ranked_ids, present, retrieve_phase,
    summary_phase, BatchOutcome, PromptPair, RetrievedHit, SummaryDraft, Trajectory, LOG_TARGET,
};
use crate::context_evolver::record::{MemoryDraft, MemoryRecord};
use crate::context_evolver::Algo;

/// The rerank system prompt (Python's `MEMORY_RERANK_PROMPT`).
pub const RETRIEVE_SYSTEM: &str = r#"You are an expert AI analyst tasked with reranking retrieved experiences based on their relevance to a specific query.
Your task is to analyze the candidates and rank them by relevance, considering:
● DIRECT RELEVANCE: How directly applicable the experience is to the current query
● SITUATION SIMILARITY: How similar the experience context is to the current situation
● ACTIONABILITY: How actionable and specific the experience is
● QUALITY: The overall quality and clarity of the experience
OUTPUT FORMAT:
Provide a ranked list of candidate indices (0-based) from most relevant to least relevant:
{"ranked_indices": [2, 0, 4, 1, 3], "reasoning": "Brief explanation of ranking rationale"}
Note: Include ALL candidate indices in the ranking, even if some are less relevant."#;

/// The success extraction prompt (Python's `SUCCESS_MEMORY_PROMPT`).
pub const SUMMARIZE_SYSTEM: &str = r#"You are an expert AI analyst reviewing successful step sequences from an AI agent execution.
Your task is to extract reusable, actionable step-level task memories that can guide future agent executions.
Focus on identifying specific patterns, techniques, and decision points that contributed to success.
ANALYSIS FRAMEWORK:
● STEP PATTERN ANALYSIS: Identify the specific sequence of actions that led to success
● DECISION POINTS: Highlight critical decisions made during these steps
● TECHNIQUE EFFECTIVENESS: Analyze why specific approaches worked well
● REUSABILITY: Extract patterns that can be applied to similar scenarios
EXTRACTION PRINCIPLES:
● Focus on TRANSFERABLE TECHNIQUES and decision frameworks
● Frame insights as actionable guidelines and best practices
OUTPUT FORMAT:
Generate 1-3 step-level success insights as JSON objects:
[
{
    "when_to_use": "Specific conditions when this step pattern should be applied",
    "experience": "Detailed description of the successful step pattern and why it works",
    "tags": ["relevant", "keywords", "for", "categorization"],
    "confidence": 0.8,
    "step_type": "reasoning|action|observation|decision",
    "tools_used": ["list", "of", "tools"]
}
]"#;

/// The failure extraction prompt (Python's `FAILURE_MEMORY_PROMPT`).
pub const SUMMARIZE_FAILURE_SYSTEM: &str = r#"You are an expert AI analyst reviewing failed step sequences from an AI agent execution.
Your task is to extract learning task memories from failures to prevent similar mistakes in future executions.
Focus on identifying error patterns, missed opportunities, and alternative approaches.
ANALYSIS FRAMEWORK:
● FAILURE POINT IDENTIFICATION: Pinpoint where and why the steps went wrong
● ERROR PATTERN ANALYSIS: Identify recurring mistakes or problematic approaches
● ALTERNATIVE APPROACHES: Suggest what could have been done differently
● PREVENTION STRATEGIES: Extract actionable insights to avoid similar failures
EXTRACTION PRINCIPLES:
● Extract GENERAL PRINCIPLES as well as SPECIFIC INSTRUCTIONS
● Focus on PATTERNS and RULES as well as particular instances
OUTPUT FORMAT:
Generate 1-3 step-level failure prevention insights as JSON objects:
[
{
    "when_to_use": "Specific situations where this lesson should be remembered",
    "experience": "Universal principle or rule extracted from the failure pattern",
    "tags": ["error_prevention", "failure_analysis", "relevant_keywords"],
    "confidence": 0.7,
    "step_type": "reasoning|action|observation|decision",
    "tools_used": ["list", "of", "tools"]
}
]"#;

/// The comparative extraction prompt (Python's `COMPARATIVE_MEMORY_PROMPT`).
pub const SUMMARIZE_COMPARATIVE_SYSTEM: &str = r#"You are an expert AI analyst comparing higher-scoring and lower-scoring step sequences to extract performance insights.
Your task is to identify the key differences between higher and lower performing approaches at the step level.
Focus on what made the higher-scoring approach more effective, even when both approaches may have had partial success.
SOFT COMPARATIVE ANALYSIS FRAMEWORK:
● PERFORMANCE FACTORS: Identify what specifically contributed to the higher score
● APPROACH DIFFERENCES: Compare methodologies and execution strategies
● EFFICIENCY ANALYSIS: Analyze why one approach was more efficient or effective
● OPTIMIZATION INSIGHTS: Extract lessons for improving performance
EXTRACTION PRINCIPLES:
● Focus on INCREMENTAL IMPROVEMENTS and performance optimization
● Extract QUALITY INDICATORS that differentiate better vs good approaches
● Identify REFINEMENT STRATEGIES that lead to higher scores
● Frame insights as PERFORMANCE ENHANCEMENT guidelines
OUTPUT FORMAT:
Generate up to 5 performance improvement insights as JSON objects:
[
{
    "when_to_use": "Specific scenarios where this performance insight applies",
    "experience": "Detailed analysis of what made the higher-scoring approach more effective",
    "tags": ["performance_optimization", "score_improvement", "relevant_keywords"],
    "confidence": 0.7,
    "step_type": "reasoning|action|observation|decision",
    "tools_used": ["list", "of", "tools"]
}
]"#;

/// The `RefCon` accent block appended to the extraction prompt.
pub const REF_CON_ACCENT: &str = r"REF_CON CONTRAST:
- Contrast each successful approach with the failed approach it replaces.
- State the failing behaviour, the successful behaviour, and the signal that distinguishes them, so a future agent can pick the right one from the same observation.";

/// The `DivCon` accent block appended to the extraction prompt.
pub const DIV_CON_ACCENT: &str = r"DIV_CON DECOMPOSITION:
- First decompose the task into sub-questions, then record which decomposition produced usable sub-answers and which one led to a dead end.
- Make each insight name the sub-question it answers, so a future agent can reuse the decomposition that worked.";

/// Builds the rerank prompt, numbering the candidates as Python does.
///
/// Each candidate is `Candidate <i>:` followed by `Condition:` (the record's
/// `when_to_use`) and `Experience:` (its content), blocks joined with `---`.
#[must_use]
pub fn retrieve_prompt(query: &str, memories: &[MemoryRecord]) -> Option<PromptPair> {
    let mut candidates = Vec::with_capacity(memories.len());
    for (index, record) in memories.iter().enumerate() {
        let condition = present(record.when_to_use.as_deref()).unwrap_or("<unspecified>");
        let experience = present(Some(record.content.as_str())).unwrap_or_default();
        candidates.push(format!(
            "Candidate {index}:\nCondition: {condition}\nExperience: {experience}"
        ));
    }
    let candidates = if candidates.is_empty() {
        "(none)".to_string()
    } else {
        candidates.join("\n---\n")
    };
    let total = memories.len().to_string();
    let user = fill(
        r"# Current Query
{query}
# Candidate Experiences (Total: {num_candidates})
{candidates}",
        &[
            ("query", query.trim()),
            ("num_candidates", &total),
            ("candidates", &candidates),
        ],
    );
    Some(PromptPair::new(RETRIEVE_SYSTEM, user))
}

/// Builds the `ReMe` extraction prompt, with no per-algorithm accent.
#[must_use]
pub fn summarize_prompt(trajectories: &[Trajectory], memories: &[MemoryRecord]) -> PromptPair {
    summarize_prompt_for(Algo::Reme, trajectories, memories)
}

/// Builds the extraction prompt for one `ReMe`-family algorithm.
///
/// The system prompt is selected by the batch outcome (success, failure,
/// comparative) and then extended with [`REF_CON_ACCENT`] for
/// [`Algo::Refcon`] or [`DIV_CON_ACCENT`] for [`Algo::Divcon`].
#[must_use]
pub fn summarize_prompt_for(
    algo: Algo,
    trajectories: &[Trajectory],
    memories: &[MemoryRecord],
) -> PromptPair {
    let base = match batch_outcome(trajectories) {
        BatchOutcome::Success => SUMMARIZE_SYSTEM,
        BatchOutcome::Failure => SUMMARIZE_FAILURE_SYSTEM,
        BatchOutcome::Comparative => SUMMARIZE_COMPARATIVE_SYSTEM,
    };
    let system = match algo {
        Algo::Refcon => format!("{base}\n\n{REF_CON_ACCENT}"),
        Algo::Divcon => format!("{base}\n\n{DIV_CON_ACCENT}"),
        _ => base.to_string(),
    };
    PromptPair::new(system, summarize_user_prompt(trajectories, memories))
}

/// Builds the shared extraction user prompt.
fn summarize_user_prompt(trajectories: &[Trajectory], memories: &[MemoryRecord]) -> String {
    let query = batch_query(trajectories);
    let outcome = outcome_label(trajectories);
    let mut sequences = Vec::with_capacity(trajectories.len());
    for (index, trajectory) in trajectories.iter().enumerate() {
        let response = trajectory.response.trim();
        let feedback = trajectory.feedback.as_str();
        sequences.push(format!(
            "Step sequence {index} (feedback: {feedback}):\n{response}"
        ));
    }
    let step_sequence = if sequences.is_empty() {
        "(none)".to_string()
    } else {
        sequences.join("\n\n")
    };
    let existing = existing_memories(memories);
    let mut user = format!(
        r"# Original Query
{query}
# Step Sequence Analysis
{step_sequence}
# Outcome
This step sequence was part of a {outcome} trajectory."
    );
    if !existing.is_empty() {
        user.push_str("\n\nDo not repeat these existing memories:\n");
        user.push_str(&existing);
    }
    user
}

/// Joins the distinct trajectory queries, or reports that none was supplied.
fn batch_query(trajectories: &[Trajectory]) -> String {
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

/// Labels the batch the way Python's `{outcome}` placeholder does.
fn outcome_label(trajectories: &[Trajectory]) -> &'static str {
    match batch_outcome(trajectories) {
        BatchOutcome::Success => "successful",
        BatchOutcome::Failure => "failed",
        BatchOutcome::Comparative => "mixed",
    }
}

/// Lists the stored memories as `When to use`/`Content` blocks.
fn existing_memories(memories: &[MemoryRecord]) -> String {
    let mut blocks = Vec::new();
    for record in memories {
        let when_to_use = present(record.when_to_use.as_deref()).unwrap_or("<unspecified>");
        let content = present(Some(record.content.as_str())).unwrap_or_default();
        blocks.push(format!(
            "- When to use: {when_to_use}\n  Content: {content}"
        ));
    }
    blocks.join("\n")
}

/// Parses a rerank reply into ordered hits.
///
/// # Errors
///
/// Returns [`Error::Validation`] naming the `reme retrieve` phase when the reply
/// is not JSON, carries no ranking array, or names no supplied memory.
pub fn parse_retrieve(reply: &str, memories: &[MemoryRecord]) -> Result<Vec<RetrievedHit>> {
    parse_ranked_ids(&retrieve_phase("reme"), reply, memories)
}

/// Parses an extraction reply into memory drafts.
///
/// Accepts a JSON array of experience objects, or a single experience object
/// (Python's fallback). Each item needs a non-blank `experience`; an item without
/// one is skipped with a `tracing::debug!`, so an all-invalid array yields an
/// empty result. `when_to_use` (or Python's alias `condition`) becomes
/// [`MemoryDraft::when_to_use`], `experience` becomes
/// [`MemoryDraft::content`], and `tags`/`confidence`/`step_type`/`tools_used` are
/// carried verbatim in [`MemoryDraft::attributes`].
///
/// # Errors
///
/// Returns [`Error::Validation`] naming the `reme summarize` phase when the reply
/// is not JSON, or is a JSON value that is neither an array nor an object.
pub fn parse_summarize(reply: &str) -> Result<Vec<SummaryDraft>> {
    let phase = summary_phase("reme");
    let value = parse_json_reply(&phase, reply)?;
    let items: Vec<&Value> = match &value {
        Value::Array(items) => items.iter().collect(),
        Value::Object(_) => vec![&value],
        _ => {
            return Err(Error::Validation(format!(
                "context_evolver {phase} pipeline: expected a JSON array of experience objects"
            )))
        }
    };
    let mut drafts = Vec::with_capacity(items.len());
    for item in items {
        if let Some(draft) = draft_from_item(item) {
            drafts.push(draft);
        }
    }
    Ok(drafts)
}

/// Maps one experience object onto a draft, skipping items without experience.
fn draft_from_item(item: &Value) -> Option<SummaryDraft> {
    let Some(map) = item.as_object() else {
        debug!(target: LOG_TARGET, "reme experience item is not a JSON object; skipping");
        return None;
    };
    let experience = map
        .get("experience")
        .and_then(Value::as_str)
        .and_then(|text| present(Some(text)));
    let Some(experience) = experience else {
        debug!(target: LOG_TARGET, "reme experience item has no experience text; skipping");
        return None;
    };
    let when_to_use = ["when_to_use", "condition"]
        .iter()
        .find_map(|key| map.get(*key).and_then(Value::as_str))
        .and_then(|text| present(Some(text)))
        .map(ToOwned::to_owned);
    Some(SummaryDraft::add(MemoryDraft {
        content: experience.to_string(),
        when_to_use,
        attributes: attributes_from(map),
        ..MemoryDraft::default()
    }))
}

/// Copies the optional `ReMe` metadata fields onto draft attributes.
fn attributes_from(map: &Map<String, Value>) -> BTreeMap<String, Value> {
    let mut attributes = BTreeMap::new();
    for key in ["tags", "confidence", "step_type", "tools_used"] {
        if let Some(value) = map.get(key) {
            if value.is_null() {
                continue;
            }
            attributes.insert(key.to_string(), value.clone());
        }
    }
    attributes
}

/// Renders the retained memories in Python's `ReMe` context format.
///
/// `Memory <i>:` (1-based, matching `_format_memories_for_context`) with
/// two-space-indented `When to use:` and `Content:` lines, blocks separated by a
/// blank line. Hits are rendered in their own order; an id that names no supplied
/// record is skipped with a `tracing::debug!`.
#[must_use]
pub fn render_memory_string(hits: &[RetrievedHit], memories: &[MemoryRecord]) -> String {
    let mut blocks = Vec::with_capacity(hits.len());
    for hit in hits {
        let Some(record) = find_record(memories, &hit.id) else {
            debug!(
                target: LOG_TARGET,
                id = hit.id.as_str(),
                "reme hit names no supplied memory; skipping"
            );
            continue;
        };
        let when_to_use = present(record.when_to_use.as_deref()).unwrap_or("<unspecified>");
        let content = present(Some(record.content.as_str())).unwrap_or_default();
        blocks.push(format!(
            "Memory {}:\n  When to use: {when_to_use}\n  Content: {content}",
            blocks.len() + 1
        ));
    }
    blocks.join("\n\n")
}

#[cfg(test)]
mod tests {
    // Python counterpart: `openjiuwen/extensions/context_evolver/summary/task/reme/`
    // (`prompt.py`, `update.py`, `utils.py`) and `retrieve/task/reme/`
    // (`prompt.py`, `run.py::RerankMemoryOp`).
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::float_cmp)]
    use super::*;
    use crate::context_evolver::prompt::{SummaryOperation, TrajectoryFeedback};
    use openjiuwen_common::error::ErrorKind;
    use serde_json::json;

    fn record(id: &str, when_to_use: &str, content: &str) -> MemoryRecord {
        MemoryRecord {
            id: id.to_string(),
            user_id: "user".into(),
            when_to_use: Some(when_to_use.to_string()),
            content: content.to_string(),
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
    fn retrieve_prompt_numbers_conditions_and_experiences() {
        let memories = vec![
            record("reme-1", "when listing playlists", "loop while True"),
            record("reme-2", "when logging in", "read the password first"),
        ];
        let prompt = retrieve_prompt("count all playlists", &memories).unwrap();
        assert_eq!(prompt.system, RETRIEVE_SYSTEM);
        assert!(prompt.user.contains("# Current Query\ncount all playlists"));
        assert!(prompt.user.contains("Total: 2"));
        assert!(prompt.user.contains(
            "Candidate 0:\nCondition: when listing playlists\nExperience: loop while True"
        ));
        assert!(prompt.user.contains(
            "Candidate 1:\nCondition: when logging in\nExperience: read the password first"
        ));
        assert!(prompt.user.contains("---"));
        assert!(prompt.system.contains("ranked_indices"));
    }

    #[test]
    fn retrieve_prompt_without_candidates_says_none() {
        let prompt = retrieve_prompt("query", &[]).unwrap();
        assert!(prompt.user.contains("(none)"));
        assert!(prompt.user.contains("Total: 0"));
    }

    #[test]
    fn parse_retrieve_resolves_ranked_indices() {
        let memories = vec![
            record("reme-1", "a", "first"),
            record("reme-2", "b", "second"),
            record("reme-3", "c", "third"),
        ];
        let hits = parse_retrieve(r#"{"ranked_indices": [2, 0]}"#, &memories).unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].id, "reme-3");
        assert_eq!(hits[1].id, "reme-1");
        assert!(hits[0].score > hits[1].score);

        let error = parse_retrieve("not json", &memories).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(
            error.message().contains("reme retrieve"),
            "{}",
            error.message()
        );
    }

    #[test]
    fn summarize_prompt_selects_the_prompt_by_batch_outcome() {
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
        assert!(summarize_prompt(&success, &[]).user.contains("successful"));
        assert!(summarize_prompt(&failure, &[]).user.contains("failed"));
        assert!(summarize_prompt(&mixed, &[]).user.contains("mixed"));
    }

    #[test]
    fn summarize_prompt_carries_the_trajectory_and_existing_memories() {
        let trajectories = vec![trajectory(
            "count all playlists",
            "used a fixed range loop",
            TrajectoryFeedback::Harmful,
        )];
        let memories = vec![record(
            "reme-1",
            "when listing playlists",
            "loop while True",
        )];
        let prompt = summarize_prompt(&trajectories, &memories);
        assert!(prompt
            .user
            .contains("# Original Query\ncount all playlists"));
        assert!(prompt.user.contains("used a fixed range loop"));
        assert!(prompt.user.contains("Step sequence 0 (feedback: harmful)"));
        assert!(prompt
            .user
            .contains("- When to use: when listing playlists\n  Content: loop while True"));
    }

    #[test]
    fn summarize_prompt_for_adds_the_refcon_and_divcon_accents() {
        let trajectories = vec![trajectory("q", "r", TrajectoryFeedback::Helpful)];
        let reme = summarize_prompt_for(Algo::Reme, &trajectories, &[]);
        let refcon = summarize_prompt_for(Algo::Refcon, &trajectories, &[]);
        let divcon = summarize_prompt_for(Algo::Divcon, &trajectories, &[]);
        assert_eq!(reme.system, SUMMARIZE_SYSTEM);
        assert!(refcon.system.starts_with(SUMMARIZE_SYSTEM));
        assert!(refcon.system.contains("REF_CON CONTRAST"));
        assert!(refcon.system.contains("Contrast each successful approach"));
        assert!(divcon.system.starts_with(SUMMARIZE_SYSTEM));
        assert!(divcon.system.contains("DIV_CON DECOMPOSITION"));
        assert!(divcon
            .system
            .contains("decompose the task into sub-questions"));
    }

    #[test]
    fn parse_summarize_maps_experiences_and_attributes() {
        let reply = json!([
            {
                "when_to_use": "when paginating",
                "experience": "Use a while True loop",
                "tags": ["pagination"],
                "confidence": 0.8,
                "step_type": "action",
                "tools_used": ["http"]
            },
            {"condition": "when logging in", "experience": "read the password first"},
            {"experience": "   "},
            {"when_to_use": "no experience"}
        ])
        .to_string();
        let drafts = parse_summarize(&reply).unwrap();
        assert_eq!(drafts.len(), 2);
        assert_eq!(drafts[0].operation, SummaryOperation::Add);
        assert_eq!(drafts[0].draft.content, "Use a while True loop");
        assert_eq!(
            drafts[0].draft.when_to_use.as_deref(),
            Some("when paginating")
        );
        assert_eq!(
            drafts[0].draft.attributes.get("tags"),
            Some(&json!(["pagination"]))
        );
        assert_eq!(
            drafts[0].draft.attributes.get("confidence"),
            Some(&json!(0.8))
        );
        assert_eq!(
            drafts[0].draft.attributes.get("step_type"),
            Some(&json!("action"))
        );
        assert_eq!(
            drafts[0].draft.attributes.get("tools_used"),
            Some(&json!(["http"]))
        );
        assert_eq!(
            drafts[1].draft.when_to_use.as_deref(),
            Some("when logging in")
        );
        assert!(drafts[1].draft.attributes.is_empty());
    }

    #[test]
    fn parse_summarize_accepts_fenced_json_and_a_single_object() {
        let fenced = "```json\n[{\"experience\": \"one\", \"when_to_use\": \"w\"}]\n```";
        assert_eq!(parse_summarize(fenced).unwrap().len(), 1);
        let single = r#"{"experience": "only one", "when_to_use": "w"}"#;
        let drafts = parse_summarize(single).unwrap();
        assert_eq!(drafts.len(), 1);
        assert_eq!(drafts[0].draft.content, "only one");
    }

    #[test]
    fn parse_summarize_returns_empty_for_an_explicit_empty_array() {
        assert_eq!(parse_summarize("[]").unwrap().len(), 0);
    }

    #[test]
    fn parse_summarize_rejects_malformed_replies_naming_the_phase() {
        let error = parse_summarize("no experiences").unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(
            error.message().contains("reme summarize"),
            "{}",
            error.message()
        );

        let error = parse_summarize("42").unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(
            error.message().contains("reme summarize"),
            "{}",
            error.message()
        );
    }

    #[test]
    fn render_memory_string_uses_the_python_context_format() {
        let memories = vec![
            record("reme-1", "when listing", "loop while True"),
            record("reme-2", "when logging in", "read the password"),
        ];
        let hits = vec![
            RetrievedHit::new("reme-2", 1.0),
            RetrievedHit::new("missing", 0.5),
            RetrievedHit::new("reme-1", 0.4),
        ];
        assert_eq!(
            render_memory_string(&hits, &memories),
            "Memory 1:\n  When to use: when logging in\n  Content: read the password\n\n\
             Memory 2:\n  When to use: when listing\n  Content: loop while True"
        );
        assert_eq!(render_memory_string(&[], &memories), "");
    }
}
