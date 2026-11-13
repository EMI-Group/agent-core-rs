//! Prompt construction and reply parsing for the context-evolver pipelines.
//!
//! Ports the model-facing half of Python's
//! `openjiuwen/extensions/context_evolver/`: one module per algorithm family
//! ([`ace`], [`reasoning_bank`], [`reme`], [`cognition`]) plus the shared
//! contract in this module — [`PromptPair`], [`Trajectory`], [`SummaryDraft`],
//! the ranked-id parser, and the dispatch functions [`retrieve_prompt`],
//! [`summarize_prompt`], [`parse_retrieve`], [`parse_summarize`], and
//! [`render_memory_string`].
//!
//! Python drives retrieval and summarization through `BaseOp` pipelines bound to
//! a `ServiceContext` (LLM, embedding model, vector store). This port keeps only
//! the prompt text and the reply schema, so a host can drive any completion
//! backend; embeddings, ranking by vector similarity, and persistence stay with
//! the caller. Reply tolerance is delegated to
//! [`super::completion::parse_json_reply`] everywhere, so there is exactly one
//! JSON extractor in the crate.
//!
//! Python counterparts: `retrieve/task/*/{run.py,prompt.py}`,
//! `summary/task/*/{update.py,prompt.py}`, `schema/trajectory.py`, and
//! `schema/memory.py`.
//!
//! # Deviations from Python
//!
//! * Python holds prompts in `@dataclass` prompt tables (`ReMePrompts`,
//!   `ReasoningBankPrompts`, `CognitionPrompts`, `ACEPrompt`); this port exposes
//!   the same text as `SUMMARIZE_SYSTEM`-style constants plus builder functions,
//!   which is what the call sites actually need.
//! * Python formats prompts with `str.format` and escapes JSON braces as `{{`/
//!   `}}`; this port substitutes only the exact `{name}` placeholders a template
//!   declares ([`fill`]), so the JSON examples stay literal.
//! * Python's retrieval steps are two or three separate LLM round-trips
//!   (classify, rerank, rewrite). The family modules fold them into a single
//!   prompt and document each fold.
//! * Ranking scores are derived from the reply's rank position
//!   (see [`parse_ranked_ids`]) instead of a vector similarity, because the
//!   vector store is not part of this layer.
//!
//! # Example
//!
//! ```
//! use openjiuwen_extensions::context_evolver::prompt::{parse_trajectories, TrajectoryFeedback};
//! use serde_json::json;
//!
//! let trajectories = parse_trajectories(&json!([{"query": "q", "response": "r", "feedback": "success"}]))
//!     .expect("a JSON array of objects parses");
//! assert_eq!(trajectories[0].feedback, TrajectoryFeedback::Helpful);
//! ```
use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tracing::debug;

use super::completion::parse_json_reply;
use super::record::{MemoryDraft, MemoryRecord};
use super::{Algo, AlgoFamily};

pub mod ace;
pub mod cognition;
pub mod reasoning_bank;
pub mod reme;

/// The tracing target shared by this module tree.
pub(crate) const LOG_TARGET: &str = "openjiuwen_extensions::context_evolver::prompt";

/// A system/user prompt pair ready for a single completion call.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct PromptPair {
    /// The system prompt: the algorithm's role and its output contract.
    pub system: String,
    /// The user prompt: task context, candidate memories, and the reply shape.
    pub user: String,
}

impl PromptPair {
    /// Builds a prompt pair from a system and a user prompt.
    #[must_use]
    pub fn new(system: impl Into<String>, user: impl Into<String>) -> Self {
        Self {
            system: system.into(),
            user: user.into(),
        }
    }
}

/// One memory retained by a retrieve pipeline, in ranked order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct RetrievedHit {
    /// The [`MemoryRecord::id`] of the retained memory.
    pub id: String,
    /// Rank-derived relevance score, highest first within one reply.
    pub score: f32,
    /// Optional justification; Python's rerank replies may carry one.
    #[serde(default)]
    pub reason: Option<String>,
}

impl RetrievedHit {
    /// Builds a hit with a score and no justification.
    #[must_use]
    pub fn new(id: impl Into<String>, score: f32) -> Self {
        Self {
            id: id.into(),
            score,
            reason: None,
        }
    }

    /// Attaches a justification to the hit.
    #[must_use]
    pub fn with_reason(mut self, reason: impl Into<String>) -> Self {
        self.reason = Some(reason.into());
        self
    }
}

/// The outcome label a trajectory carries.
///
/// Ports `schema/trajectory.py::FeedbackType`; serde uses `snake_case` names, so
/// `"helpful"`, `"harmful"`, and `"neutral"` round-trip. [`Trajectory`] parsing
/// is more tolerant than the enum itself and also accepts the Python aliases
/// `success`, `failure`, and `unknown`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrajectoryFeedback {
    /// The attempt succeeded (`FeedbackType.HELPFUL`).
    Helpful,
    /// The attempt failed (`FeedbackType.HARMFUL`).
    Harmful,
    /// No verdict; Python's default (`FeedbackType.NEUTRAL`).
    #[default]
    Neutral,
}

impl TrajectoryFeedback {
    /// The Python enum value as a lower-case string.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Helpful => "helpful",
            Self::Harmful => "harmful",
            Self::Neutral => "neutral",
        }
    }

    /// Whether the attempt succeeded (`FeedbackType.HELPFUL`).
    #[must_use]
    pub const fn is_success(self) -> bool {
        matches!(self, Self::Helpful)
    }

    /// Whether the attempt failed (`FeedbackType.HARMFUL`).
    #[must_use]
    pub const fn is_failure(self) -> bool {
        matches!(self, Self::Harmful)
    }
}

/// One recorded task attempt with its outcome.
///
/// Ports `schema/trajectory.py::Trajectory`. [`Trajectory::context`] keeps
/// Python's free-form `context` dictionary; a missing or `null` value becomes an
/// empty object.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct Trajectory {
    /// The query or task that was executed.
    pub query: String,
    /// The response or output the agent produced.
    pub response: String,
    /// The outcome label; Python's default is [`TrajectoryFeedback::Neutral`].
    pub feedback: TrajectoryFeedback,
    /// Additional execution context, carried verbatim.
    pub context: Value,
}

/// Parses a trajectory batch, accepting the two shapes Python callers pass.
///
/// Supported input: a JSON array of trajectory objects, or an object with a
/// `trajectories` array (Python's `TrajectoryBatch`). Each entry must be an
/// object; `query` and `response` default to `""` when absent, `null`, or not a
/// string, and `feedback` accepts `helpful`/`success`, `harmful`/`failure`, and
/// `neutral`/`unknown` (case-insensitive) — anything else, including an absent
/// value, is [`TrajectoryFeedback::Neutral`].
///
/// # Errors
///
/// Returns [`Error::Validation`] when the payload is neither an array nor an
/// object holding a `trajectories` array, or when an entry is not a JSON object
/// (the message names its index).
pub fn parse_trajectories(value: &Value) -> Result<Vec<Trajectory>> {
    let entries = match value {
        Value::Array(items) => items.as_slice(),
        Value::Object(map) => match map.get("trajectories") {
            Some(Value::Array(items)) => items.as_slice(),
            _ => {
                return Err(Error::Validation(
                    "context_evolver trajectories: expected a JSON array or an object with a \
                     'trajectories' array"
                        .to_string(),
                ))
            }
        },
        _ => {
            return Err(Error::Validation(
                "context_evolver trajectories: expected a JSON array or an object with a \
                 'trajectories' array"
                    .to_string(),
            ))
        }
    };
    let mut trajectories = Vec::with_capacity(entries.len());
    for (index, entry) in entries.iter().enumerate() {
        let Some(map) = entry.as_object() else {
            return Err(Error::Validation(format!(
                "context_evolver trajectories: entry at index {index} is not a JSON object"
            )));
        };
        trajectories.push(Trajectory {
            query: string_field(map.get("query"), index, "query"),
            response: string_field(map.get("response"), index, "response"),
            feedback: feedback_field(map.get("feedback"), index),
            context: match map.get("context") {
                None | Some(Value::Null) => Value::Object(Map::new()),
                Some(context) => context.clone(),
            },
        });
    }
    Ok(trajectories)
}

/// Reads a string field, defaulting to `""` and logging the coercion.
fn string_field(value: Option<&Value>, index: usize, field: &str) -> String {
    match value {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(text)) => text.clone(),
        Some(_) => {
            debug!(
                target: LOG_TARGET,
                index, field, "trajectory field is not a string; using an empty string"
            );
            String::new()
        }
    }
}

/// Reads the feedback label, tolerating Python's aliases.
fn feedback_field(value: Option<&Value>, index: usize) -> TrajectoryFeedback {
    let Some(raw) = value.and_then(Value::as_str) else {
        return TrajectoryFeedback::Neutral;
    };
    match raw.trim().to_ascii_lowercase().as_str() {
        "helpful" | "success" => TrajectoryFeedback::Helpful,
        "harmful" | "failure" => TrajectoryFeedback::Harmful,
        "neutral" | "unknown" => TrajectoryFeedback::Neutral,
        _ => {
            debug!(
                target: LOG_TARGET,
                index, "unrecognised trajectory feedback; using neutral"
            );
            TrajectoryFeedback::Neutral
        }
    }
}

/// The mutation a [`SummaryDraft`] asks a persister to perform.
///
/// Ports `summary/task/ace/playbook.py::OperationType` (`ADD`, `UPDATE`, `TAG`,
/// `REMOVE`); serde represents the variants in `snake_case`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SummaryOperation {
    /// Create a new memory.
    #[default]
    Add,
    /// Replace the content of the memory named by [`SummaryDraft::id`].
    Update,
    /// Adjust the feedback counters of the memory named by [`SummaryDraft::id`].
    Tag,
    /// Delete the memory named by [`SummaryDraft::id`].
    Remove,
}

/// One parsed summarize operation.
///
/// [`SummaryDraft::draft`] carries the payload a persister needs; for
/// [`SummaryOperation::Remove`] it is the default (empty) draft because nothing
/// is written.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct SummaryDraft {
    /// The target memory id; `None` means "assign a fresh id" (Python's `ADD`).
    pub id: Option<String>,
    /// The mutation to apply.
    pub operation: SummaryOperation,
    /// The payload to store or merge.
    pub draft: MemoryDraft,
}

impl Default for SummaryDraft {
    fn default() -> Self {
        Self {
            id: None,
            operation: SummaryOperation::Add,
            draft: MemoryDraft::default(),
        }
    }
}

impl SummaryDraft {
    /// Builds an `ADD` operation for `draft`; the persister assigns the id.
    #[must_use]
    pub fn add(draft: MemoryDraft) -> Self {
        Self {
            id: None,
            operation: SummaryOperation::Add,
            draft,
        }
    }

    /// Targets an existing memory id.
    #[must_use]
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// The targeted memory id, if any.
    #[must_use]
    pub fn target_id(&self) -> Option<&str> {
        self.id.as_deref()
    }
}

/// Builds the phase label used in summarize-reply validation errors.
///
/// The result feeds [`super::completion::parse_json_reply`], so a malformed
/// reply names both the algorithm and the phase, for example
/// `context_evolver ace summarize pipeline: …`.
#[must_use]
pub fn summary_phase(name: &str) -> String {
    format!("{name} summarize")
}

/// Builds the phase label used in retrieve-reply validation errors.
#[must_use]
pub fn retrieve_phase(name: &str) -> String {
    format!("{name} retrieve")
}

/// Renders the numbered candidate block the ranking prompts show the model.
///
/// One block per record, index `0`-based (matching the `ranked_indices` reply
/// Python asks for), with only the fields the record actually carries:
///
/// ```text
/// Candidate 0:
///   id: rb-1
///   Title: Pagination
///   When to use: paginated APIs
///   Content: …
/// ```
///
/// Blocks are separated by a blank line. An empty input renders as `""`.
#[must_use]
pub fn render_candidates(memories: &[MemoryRecord]) -> String {
    let mut blocks = Vec::with_capacity(memories.len());
    for (index, record) in memories.iter().enumerate() {
        let mut lines = vec![format!("Candidate {index}:")];
        if let Some(id) = present(Some(record.id.as_str())) {
            lines.push(format!("  id: {id}"));
        }
        if let Some(title) = present(record.title.as_deref()) {
            lines.push(format!("  Title: {title}"));
        }
        if let Some(when_to_use) = present(record.when_to_use.as_deref()) {
            lines.push(format!("  When to use: {when_to_use}"));
        }
        if let Some(content) = present(Some(record.content.as_str())) {
            lines.push(format!("  Content: {content}"));
        }
        blocks.push(lines.join("\n"));
    }
    blocks.join("\n\n")
}

/// Parses a ranked-id reply into ordered [`RetrievedHit`]s.
///
/// Accepts every shape the Python rerank prompts produce:
///
/// * `{"ranked_ids": ["id", …]}` and `{"ids": […]}` — ids, or numeric strings
///   that name a candidate index;
/// * `{"ranked_indices": [0, 2, …]}` and numbers in a bare array — 0-based
///   candidate indices;
/// * a bare JSON array of ids, numeric strings, or indices.
///
/// Unknown ids and out-of-range indices are skipped with a `tracing::debug!`,
/// duplicates are dropped keeping the first occurrence, and the surviving hits
/// are scored by rank: `score = 1.0 - rank / max(1, n)` for `n` hits, which is
/// strictly descending, so callers can rely on the order.
///
/// # Errors
///
/// Returns [`Error::Validation`] naming `phase` when the reply is not JSON (via
/// [`super::completion::parse_json_reply`]), carries none of the three arrays, or
/// matches no supplied memory.
pub fn parse_ranked_ids(
    phase: &str,
    reply: &str,
    memories: &[MemoryRecord],
) -> Result<Vec<RetrievedHit>> {
    let value = parse_json_reply(phase, reply)?;
    let entries = ranked_entries(&value).ok_or_else(|| {
        Error::Validation(format!(
            "context_evolver {phase} pipeline: reply carries no ranked_ids, ids, or ranked_indices \
             array"
        ))
    })?;
    let mut hits: Vec<RetrievedHit> = Vec::with_capacity(entries.len());
    let mut taken = vec![false; memories.len()];
    for entry in entries {
        let Some(index) = resolve_entry(entry, memories) else {
            debug!(
                target: LOG_TARGET,
                phase, "ranked entry does not name a supplied memory; skipping"
            );
            continue;
        };
        if std::mem::replace(&mut taken[index], true) {
            debug!(
                target: LOG_TARGET,
                phase, "duplicate ranked entry; keeping the first occurrence"
            );
            continue;
        }
        hits.push(RetrievedHit::new(memories[index].id.clone(), 0.0));
    }
    if hits.is_empty() {
        return Err(Error::Validation(format!(
            "context_evolver {phase} pipeline: reply ranked none of the {} supplied memories",
            memories.len()
        )));
    }
    assign_rank_scores(&mut hits);
    Ok(hits)
}

/// Collects the entry list from whichever ranking key the reply used.
fn ranked_entries(value: &Value) -> Option<&[Value]> {
    match value {
        Value::Array(items) => Some(items.as_slice()),
        Value::Object(map) => ["ranked_ids", "ids", "ranked_indices"]
            .iter()
            .find_map(|key| map.get(*key).and_then(Value::as_array))
            .map(Vec::as_slice),
        _ => None,
    }
}

/// Maps one ranked entry onto a candidate index.
fn resolve_entry(entry: &Value, memories: &[MemoryRecord]) -> Option<usize> {
    match entry {
        Value::String(raw) => {
            let name = raw.trim();
            if let Some(index) = memories.iter().position(|record| record.id == name) {
                return Some(index);
            }
            name.parse::<usize>()
                .ok()
                .filter(|index| *index < memories.len())
        }
        Value::Number(number) => number
            .as_u64()
            .and_then(|index| usize::try_from(index).ok())
            .filter(|index| *index < memories.len()),
        _ => None,
    }
}

/// Applies the rank-derived, strictly descending score.
// The rank is bounded by the candidate count, which is far below `2^24`.
#[allow(clippy::cast_precision_loss)]
fn assign_rank_scores(hits: &mut [RetrievedHit]) {
    let count = hits.len() as f32;
    for (rank, hit) in hits.iter_mut().enumerate() {
        hit.score = 1.0 - (rank as f32) / count;
    }
}

/// Substitutes every `{name}` placeholder a prompt template declares.
///
/// Only the exact `{name}` forms in `values` are replaced, so the literal JSON
/// braces of an output example survive untouched — which is why the templates do
/// not need Python's `{{`/`}}` doubling.
pub(crate) fn fill(template: &str, values: &[(&str, &str)]) -> String {
    let mut rendered = template.to_owned();
    for (name, value) in values {
        rendered = rendered.replace(&format!("{{{name}}}"), value);
    }
    rendered
}

/// How a trajectory batch splits between successful and failed attempts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BatchOutcome {
    /// Every trajectory is [`TrajectoryFeedback::Helpful`].
    Success,
    /// Every trajectory is [`TrajectoryFeedback::Harmful`].
    Failure,
    /// A mix of outcomes, or no decisive trajectory at all.
    Comparative,
}

/// Classifies a batch for prompt selection.
///
/// An empty batch, an all-neutral batch, and any mixed batch are
/// [`BatchOutcome::Comparative`]: Python's single-trajectory prompts assume one
/// verdict, and a neutral trajectory has none.
pub(crate) fn batch_outcome(trajectories: &[Trajectory]) -> BatchOutcome {
    if !trajectories.is_empty() && trajectories.iter().all(|item| item.feedback.is_success()) {
        return BatchOutcome::Success;
    }
    if !trajectories.is_empty() && trajectories.iter().all(|item| item.feedback.is_failure()) {
        return BatchOutcome::Failure;
    }
    BatchOutcome::Comparative
}

/// Renders a trajectory batch as `Query`/`Response`/`Feedback` blocks.
pub(crate) fn render_trajectories(trajectories: &[Trajectory]) -> String {
    let mut blocks = Vec::with_capacity(trajectories.len());
    for trajectory in trajectories {
        let query = trajectory.query.trim();
        let response = trajectory.response.trim();
        let feedback = trajectory.feedback.as_str();
        blocks.push(format!(
            "Query: {query}\nResponse: {response}\nFeedback: {feedback}"
        ));
    }
    blocks.join("\n\n")
}

/// Trims `value` and returns it only when something is left.
pub(crate) fn present(value: Option<&str>) -> Option<&str> {
    value.map(str::trim).filter(|text| !text.is_empty())
}

/// Renders a JSON attribute value for a prompt line.
pub(crate) fn attribute_text(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => other.to_string(),
    }
}

/// Looks a record up by id.
pub(crate) fn find_record<'a>(memories: &'a [MemoryRecord], id: &str) -> Option<&'a MemoryRecord> {
    let name = id.trim();
    memories.iter().find(|record| record.id == name)
}

/// Builds the retrieve prompt for `algo`, or `None` when retrieval is unranked.
///
/// [`Algo::Ace`] loads the whole playbook without a model call in Python, so it
/// has no retrieve prompt; every other algorithm returns a ranking prompt.
#[must_use]
pub fn retrieve_prompt(algo: Algo, query: &str, memories: &[MemoryRecord]) -> Option<PromptPair> {
    match algo {
        Algo::Ace => ace::retrieve_prompt(query, memories),
        Algo::ReasoningBank => reasoning_bank::retrieve_prompt(query, memories),
        Algo::Reme | Algo::Refcon | Algo::Divcon => reme::retrieve_prompt(query, memories),
        Algo::Cognition => cognition::retrieve_prompt(query, memories),
    }
}

/// Builds the summarize prompt for `algo`.
///
/// `RefCon` and `DivCon` share the `ReMe` family prompt but keep their own
/// accent paragraph, selected through
/// [`reme::summarize_prompt_for`].
#[must_use]
pub fn summarize_prompt(
    algo: Algo,
    trajectories: &[Trajectory],
    memories: &[MemoryRecord],
) -> PromptPair {
    match algo.family() {
        AlgoFamily::Ace => ace::summarize_prompt(trajectories, memories),
        AlgoFamily::ReasoningBank => reasoning_bank::summarize_prompt(trajectories, memories),
        AlgoFamily::Reme => reme::summarize_prompt_for(algo, trajectories, memories),
        AlgoFamily::Cognition => cognition::summarize_prompt(trajectories, memories),
    }
}

/// Parses a retrieve reply for `algo`.
///
/// # Errors
///
/// Returns [`Error::Validation`] naming the algorithm's retrieve phase when the
/// reply cannot be interpreted.
pub fn parse_retrieve(
    algo: Algo,
    reply: &str,
    memories: &[MemoryRecord],
) -> Result<Vec<RetrievedHit>> {
    match algo.family() {
        AlgoFamily::Ace => ace::parse_retrieve(reply, memories),
        AlgoFamily::ReasoningBank => reasoning_bank::parse_retrieve(reply, memories),
        AlgoFamily::Reme => reme::parse_retrieve(reply, memories),
        AlgoFamily::Cognition => cognition::parse_retrieve(reply, memories),
    }
}

/// Parses a summarize reply for `algo`.
///
/// # Errors
///
/// Returns [`Error::Validation`] naming the algorithm's summarize phase when the
/// reply cannot be interpreted.
pub fn parse_summarize(algo: Algo, reply: &str) -> Result<Vec<SummaryDraft>> {
    match algo.family() {
        AlgoFamily::Ace => ace::parse_summarize(reply),
        AlgoFamily::ReasoningBank => reasoning_bank::parse_summarize(reply),
        AlgoFamily::Reme => reme::parse_summarize(reply),
        AlgoFamily::Cognition => cognition::parse_summarize(reply),
    }
}

/// Renders the retained memories as the prompt block `algo` uses.
///
/// The hits select and order the records; the format is the algorithm family's
/// Python memory string (see each family module's `render_memory_string`).
#[must_use]
pub fn render_memory_string(
    algo: Algo,
    hits: &[RetrievedHit],
    memories: &[MemoryRecord],
) -> String {
    match algo.family() {
        AlgoFamily::Ace => ace::render_memory_string(hits, memories),
        AlgoFamily::ReasoningBank => reasoning_bank::render_memory_string(hits, memories),
        AlgoFamily::Reme => reme::render_memory_string(hits, memories),
        AlgoFamily::Cognition => cognition::render_memory_string(hits, memories),
    }
}

#[cfg(test)]
mod tests {
    // Python counterparts: `openjiuwen/extensions/context_evolver/schema/trajectory.py`,
    // `retrieve/task/*/run.py`, and `summary/task/*/update.py` (the family tests
    // live beside each module; these cover the shared contract and dispatch).
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::float_cmp)]
    use super::*;
    use openjiuwen_common::error::ErrorKind;
    use serde_json::json;

    fn record(id: &str, content: &str) -> MemoryRecord {
        MemoryRecord {
            id: id.to_string(),
            user_id: "user".into(),
            content: content.to_string(),
            ..MemoryRecord::default()
        }
    }

    fn full_record(id: &str) -> MemoryRecord {
        MemoryRecord {
            id: id.to_string(),
            user_id: "user".into(),
            content: "body".into(),
            title: Some("T".into()),
            description: Some("D".into()),
            when_to_use: Some("W".into()),
            ..MemoryRecord::default()
        }
    }

    fn trajectory(query: &str, feedback: TrajectoryFeedback) -> Trajectory {
        Trajectory {
            query: query.into(),
            response: "response".into(),
            feedback,
            ..Trajectory::default()
        }
    }

    const RANKED_ALGOS: [Algo; 5] = [
        Algo::ReasoningBank,
        Algo::Reme,
        Algo::Refcon,
        Algo::Divcon,
        Algo::Cognition,
    ];

    const ALL_ALGOS: [Algo; 6] = [
        Algo::Ace,
        Algo::ReasoningBank,
        Algo::Reme,
        Algo::Refcon,
        Algo::Divcon,
        Algo::Cognition,
    ];

    #[test]
    fn prompt_pair_and_hit_builders_keep_their_inputs() {
        let pair = PromptPair::new("system", String::from("user"));
        assert_eq!(pair.system, "system");
        assert_eq!(pair.user, "user");

        let hit = RetrievedHit::new("m-1", 0.5).with_reason("because");
        assert_eq!(hit.id, "m-1");
        assert_eq!(hit.score, 0.5);
        assert_eq!(hit.reason.as_deref(), Some("because"));
        assert_eq!(RetrievedHit::new("m-2", 1.0).reason, None);
    }

    #[test]
    fn trajectory_feedback_reports_the_python_predicates() {
        assert_eq!(TrajectoryFeedback::default(), TrajectoryFeedback::Neutral);
        assert_eq!(TrajectoryFeedback::Helpful.as_str(), "helpful");
        assert_eq!(TrajectoryFeedback::Harmful.as_str(), "harmful");
        assert_eq!(TrajectoryFeedback::Neutral.as_str(), "neutral");
        assert!(TrajectoryFeedback::Helpful.is_success());
        assert!(!TrajectoryFeedback::Helpful.is_failure());
        assert!(TrajectoryFeedback::Harmful.is_failure());
        assert!(!TrajectoryFeedback::Harmful.is_success());
        assert!(!TrajectoryFeedback::Neutral.is_success());
        assert!(!TrajectoryFeedback::Neutral.is_failure());
    }

    #[test]
    fn dispatch_offers_a_retrieve_prompt_for_every_ranked_algo() {
        let memories = vec![record("m-1", "use a while True loop")];
        assert!(retrieve_prompt(Algo::Ace, "query", &memories).is_none());
        for algo in RANKED_ALGOS {
            let prompt = retrieve_prompt(algo, "count the playlists", &memories)
                .unwrap_or_else(|| panic!("{algo:?} must have a retrieve prompt"));
            assert!(!prompt.system.is_empty(), "{algo:?}");
            assert!(prompt.user.contains("count the playlists"), "{algo:?}");
            assert!(prompt.user.contains("use a while True loop"), "{algo:?}");
        }
    }

    #[test]
    fn dispatch_builds_a_summarize_prompt_for_every_algo() {
        let trajectories = vec![trajectory(
            "count the playlists",
            TrajectoryFeedback::Helpful,
        )];
        for algo in ALL_ALGOS {
            let prompt = summarize_prompt(algo, &trajectories, &[]);
            assert!(!prompt.system.is_empty(), "{algo:?}");
            assert!(prompt.user.contains("count the playlists"), "{algo:?}");
        }
        assert!(summarize_prompt(Algo::Refcon, &trajectories, &[])
            .system
            .contains("REF_CON"));
        assert!(summarize_prompt(Algo::Divcon, &trajectories, &[])
            .system
            .contains("DIV_CON"));
        assert!(!summarize_prompt(Algo::Reme, &trajectories, &[])
            .system
            .contains("REF_CON"));
        assert!(!summarize_prompt(Algo::Reme, &trajectories, &[])
            .system
            .contains("DIV_CON"));
    }

    #[test]
    fn dispatch_parses_each_family_reply_shape() {
        let memories = vec![record("m-1", "first"), record("m-2", "second")];

        assert_eq!(
            parse_retrieve(Algo::Ace, "ignored", &memories)
                .unwrap()
                .len(),
            2
        );
        for algo in [Algo::ReasoningBank, Algo::Cognition] {
            let hits = parse_retrieve(algo, r#"{"ranked_ids": ["m-2"]}"#, &memories).unwrap();
            assert_eq!(hits.len(), 1, "{algo:?}");
            assert_eq!(hits[0].id, "m-2", "{algo:?}");
        }
        for algo in [Algo::Reme, Algo::Refcon, Algo::Divcon] {
            let hits = parse_retrieve(algo, r#"{"ranked_indices": [1]}"#, &memories).unwrap();
            assert_eq!(hits[0].id, "m-2", "{algo:?}");
        }

        let ace = parse_summarize(
            Algo::Ace,
            r#"{"operations": [{"type": "ADD", "section": "general", "content": "ace insight"}]}"#,
        )
        .unwrap();
        assert_eq!(ace.len(), 1);
        assert_eq!(ace[0].draft.content, "ace insight");

        let bank = parse_summarize(
            Algo::ReasoningBank,
            r#"[{"title": "t", "description": "d", "content": "bank insight"}]"#,
        )
        .unwrap();
        assert_eq!(bank[0].draft.content, "bank insight");

        for algo in [Algo::Reme, Algo::Refcon, Algo::Divcon] {
            let drafts = parse_summarize(algo, r#"[{"experience": "reme insight"}]"#).unwrap();
            assert_eq!(drafts[0].draft.content, "reme insight", "{algo:?}");
        }

        let cognition = parse_summarize(
            Algo::Cognition,
            r#"{"description": "d", "experience": ["cognition insight"]}"#,
        )
        .unwrap();
        assert_eq!(cognition[0].draft.content, "cognition insight");
        assert_eq!(cognition[0].draft.section, "cognition");
    }

    #[test]
    fn dispatch_renders_each_family_memory_format() {
        let memories = vec![full_record("m-1")];
        let hits = vec![RetrievedHit::new("m-1", 1.0)];

        assert_eq!(
            render_memory_string(Algo::Ace, &hits, &memories),
            "[m-1] helpful=0 harmful=0 neutral=0\nSection: general\nContent: body"
        );
        assert_eq!(
            render_memory_string(Algo::ReasoningBank, &hits, &memories),
            "Title: T\nDescription: D\nContent: body"
        );
        assert_eq!(
            render_memory_string(Algo::Reme, &hits, &memories),
            "Memory 1:\n  When to use: W\n  Content: body"
        );
        assert_eq!(
            render_memory_string(Algo::Refcon, &hits, &memories),
            render_memory_string(Algo::Reme, &hits, &memories)
        );
        assert_eq!(
            render_memory_string(Algo::Divcon, &hits, &memories),
            render_memory_string(Algo::Reme, &hits, &memories)
        );
        assert_eq!(
            render_memory_string(Algo::Cognition, &hits, &memories),
            "Description: D\nContent: body"
        );
    }

    #[test]
    fn dispatch_unknown_hits_render_nothing() {
        let memories = vec![full_record("m-1")];
        let hits = vec![RetrievedHit::new("missing", 1.0)];
        for algo in ALL_ALGOS {
            assert!(
                render_memory_string(algo, &hits, &memories).is_empty(),
                "{algo:?}"
            );
            assert!(
                render_memory_string(algo, &[], &memories).is_empty(),
                "{algo:?}"
            );
        }
    }

    #[test]
    fn parse_trajectories_reads_arrays_and_wrapped_objects() {
        let array = json!([{"query": "q", "response": "r", "feedback": "helpful"}]);
        let trajectories = parse_trajectories(&array).unwrap();
        assert_eq!(trajectories.len(), 1);
        assert_eq!(trajectories[0].query, "q");
        assert_eq!(trajectories[0].response, "r");
        assert_eq!(trajectories[0].feedback, TrajectoryFeedback::Helpful);
        assert_eq!(trajectories[0].context, json!({}));

        let wrapped = json!({"trajectories": [{"query": "q"}], "user_id": "u"});
        assert_eq!(parse_trajectories(&wrapped).unwrap().len(), 1);
    }

    #[test]
    fn parse_trajectories_maps_the_python_feedback_aliases() {
        let cases = [
            ("helpful", TrajectoryFeedback::Helpful),
            ("HELPFUL", TrajectoryFeedback::Helpful),
            ("success", TrajectoryFeedback::Helpful),
            ("harmful", TrajectoryFeedback::Harmful),
            ("failure", TrajectoryFeedback::Harmful),
            ("neutral", TrajectoryFeedback::Neutral),
            ("unknown", TrajectoryFeedback::Neutral),
            ("something else", TrajectoryFeedback::Neutral),
        ];
        for (raw, expected) in cases {
            let value = json!([{"query": "q", "feedback": raw}]);
            let parsed = parse_trajectories(&value).unwrap();
            assert_eq!(parsed[0].feedback, expected, "raw: {raw}");
        }
        let absent = json!([{"query": "q"}]);
        assert_eq!(
            parse_trajectories(&absent).unwrap()[0].feedback,
            TrajectoryFeedback::Neutral
        );
    }

    #[test]
    fn parse_trajectories_defaults_missing_query_and_response() {
        let value = json!([{"response": "r"}, {"query": 42, "response": null}]);
        let parsed = parse_trajectories(&value).unwrap();
        assert_eq!(parsed[0].query, "");
        assert_eq!(parsed[0].response, "r");
        assert_eq!(parsed[1].query, "");
        assert_eq!(parsed[1].response, "");
    }

    #[test]
    fn parse_trajectories_keeps_the_context_payload() {
        let value = json!([{"query": "q", "context": {"steps": 3}}]);
        assert_eq!(
            parse_trajectories(&value).unwrap()[0].context,
            json!({"steps": 3})
        );
    }

    #[test]
    fn parse_trajectories_rejects_non_object_entries_naming_the_index() {
        let error = parse_trajectories(&json!([{"query": "q"}, "not an object"])).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.message().contains("index 1"), "{}", error.message());
    }

    #[test]
    fn parse_trajectories_handles_empty_and_unrecognised_payloads() {
        assert_eq!(parse_trajectories(&json!([])).unwrap().len(), 0);
        assert_eq!(
            parse_trajectories(&json!({"trajectories": []}))
                .unwrap()
                .len(),
            0
        );

        for payload in [json!({"user_id": "u"}), json!("text"), json!(3)] {
            let error = parse_trajectories(&payload).unwrap_err();
            assert_eq!(error.kind(), ErrorKind::Validation, "payload: {payload}");
        }
    }

    #[test]
    fn render_candidates_numbers_and_renders_present_fields() {
        let mut first = record("m-1", "first content");
        first.title = Some("First".into());
        first.when_to_use = Some("when first".into());
        let second = record("m-2", "second content");
        let rendered = render_candidates(&[first, second]);
        assert_eq!(
            rendered,
            "Candidate 0:\n  id: m-1\n  Title: First\n  When to use: when first\n  Content: first content\n\n\
             Candidate 1:\n  id: m-2\n  Content: second content"
        );
        assert_eq!(render_candidates(&[]), "");
    }

    #[test]
    fn parse_ranked_ids_reads_ids_indices_and_bare_arrays() {
        let memories = vec![
            record("m-1", "first"),
            record("m-2", "second"),
            record("m-3", "third"),
        ];
        let from_ids =
            parse_ranked_ids("test", r#"{"ranked_ids": ["m-3", "m-1"]}"#, &memories).unwrap();
        assert_eq!(
            from_ids
                .iter()
                .map(|hit| hit.id.as_str())
                .collect::<Vec<_>>(),
            vec!["m-3", "m-1"]
        );

        let from_alias = parse_ranked_ids("test", r#"{"ids": ["m-2"]}"#, &memories).unwrap();
        assert_eq!(from_alias[0].id, "m-2");

        let from_indices =
            parse_ranked_ids("test", r#"{"ranked_indices": [2, 0]}"#, &memories).unwrap();
        assert_eq!(
            from_indices
                .iter()
                .map(|hit| hit.id.as_str())
                .collect::<Vec<_>>(),
            vec!["m-3", "m-1"]
        );

        let bare = parse_ranked_ids("test", r#"["m-2", "m-1"]"#, &memories).unwrap();
        assert_eq!(bare.len(), 2);
        assert_eq!(bare[0].id, "m-2");

        let numeric_strings = parse_ranked_ids("test", r#"["1", "0"]"#, &memories).unwrap();
        assert_eq!(numeric_strings[0].id, "m-2");
        assert_eq!(numeric_strings[1].id, "m-1");
    }

    #[test]
    fn parse_ranked_ids_skips_unknown_entries_and_drops_duplicates() {
        let memories = vec![record("m-1", "first"), record("m-2", "second")];
        let hits = parse_ranked_ids(
            "test",
            r#"{"ranked_ids": ["missing", "m-2", "m-2", 5, "m-1"]}"#,
            &memories,
        )
        .unwrap();
        assert_eq!(
            hits.iter().map(|hit| hit.id.as_str()).collect::<Vec<_>>(),
            vec!["m-2", "m-1"]
        );
        assert_eq!(hits[0].score, 1.0);
        assert_eq!(hits[1].score, 0.5);
    }

    #[test]
    fn parse_ranked_ids_scores_are_strictly_descending() {
        let memories = vec![
            record("m-1", "first"),
            record("m-2", "second"),
            record("m-3", "third"),
        ];
        let hits = parse_ranked_ids("test", r#"["m-1", "m-2", "m-3"]"#, &memories).unwrap();
        for pair in hits.windows(2) {
            assert!(pair[0].score > pair[1].score, "scores: {hits:?}");
        }
        assert_eq!(hits[0].score, 1.0);
        let single = parse_ranked_ids("test", r#"["m-2"]"#, &memories).unwrap();
        assert_eq!(single[0].score, 1.0);
    }

    #[test]
    fn parse_ranked_ids_tolerates_fences() {
        let memories = vec![record("m-1", "first")];
        let reply = "```json\n{\"ranked_ids\": [\"m-1\"]}\n```";
        assert_eq!(parse_ranked_ids("test", reply, &memories).unwrap().len(), 1);
    }

    #[test]
    fn parse_ranked_ids_rejects_unusable_replies_naming_the_phase() {
        let memories = vec![record("m-1", "first")];
        for reply in [
            "not json at all",
            r#"{"reasoning": "no ranking here"}"#,
            r#"{"ranked_ids": []}"#,
            r#"{"ranked_ids": ["unknown"]}"#,
            "[]",
        ] {
            let error = parse_ranked_ids("reasoningbank retrieve", reply, &memories).unwrap_err();
            assert_eq!(error.kind(), ErrorKind::Validation, "reply: {reply}");
            assert!(
                error.message().contains("reasoningbank retrieve"),
                "reply: {reply} message: {}",
                error.message()
            );
        }
    }

    #[test]
    fn parse_ranked_ids_without_memories_reports_validation() {
        let error = parse_ranked_ids("test", r#"["m-1"]"#, &[]).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
    }

    #[test]
    fn summary_operation_round_trips_through_serde() {
        for (operation, name) in [
            (SummaryOperation::Add, "add"),
            (SummaryOperation::Update, "update"),
            (SummaryOperation::Tag, "tag"),
            (SummaryOperation::Remove, "remove"),
        ] {
            let encoded = serde_json::to_value(operation).unwrap();
            assert_eq!(encoded, json!(name));
            assert_eq!(
                serde_json::from_value::<SummaryOperation>(json!(name)).unwrap(),
                operation
            );
        }
        assert_eq!(SummaryOperation::default(), SummaryOperation::Add);
    }

    #[test]
    fn summary_draft_builders_set_id_and_operation() {
        let draft = SummaryDraft::add(MemoryDraft::new("insight"));
        assert_eq!(draft.operation, SummaryOperation::Add);
        assert_eq!(draft.target_id(), None);
        let targeted = draft.with_id("m-1");
        assert_eq!(targeted.target_id(), Some("m-1"));
        assert_eq!(targeted.draft.content, "insight");
        assert_eq!(SummaryDraft::default().operation, SummaryOperation::Add);

        let encoded = serde_json::to_value(&targeted).unwrap();
        let decoded: SummaryDraft = serde_json::from_value(encoded).unwrap();
        assert_eq!(decoded, targeted);
    }

    #[test]
    fn summary_and_retrieve_phases_name_the_algorithm() {
        assert_eq!(summary_phase("ace"), "ace summarize");
        assert_eq!(retrieve_phase("reme"), "reme retrieve");
    }
}
