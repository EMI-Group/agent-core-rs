//! Core data models of the prompt-search (RLAF-P) domain layer.
//!
//! Ports the dataclasses of
//! `openjiuwen/dev_tools/tune/optimizer/prompt_search/models.py` onto serde
//! types that keep the Python JSON field names. The vocabulary is shared by the
//! reward model, the drift judge, the prompt memory, the history compressor, and
//! (above this crate) the optimization loop:
//!
//! - [`PromptTaskCase`] — one evaluation case plus whether it is held out.
//!   Python wraps a `Case`; here the extracted `input_text`/`expected_text` are
//!   the stored shape and [`PromptTaskCase::to_case`] converts back, so a task
//!   still composes with the rest of `tune`.
//! - [`PromptTaskSpec`] — what is being optimized for: objective, cases,
//!   constraints, base prompt, metadata.
//! - [`PromptCandidate`], [`CaseResult`], [`Execution`] — one candidate prompt,
//!   one case outcome, and one full run of a candidate.
//! - [`RewardBreakdown`], [`Evaluation`], [`IterationRecord`] — the reward of a
//!   run, a candidate paired with its reward, and one round's candidates.
//! - [`PromptRecord`] — a persisted optimization outcome for the prompt memory.
//!
//! Every identifier and timestamp is supplied by the caller or derived
//! deterministically: nothing here reads a clock, the environment, or a random
//! number generator, so a run and its tests are reproducible.

use std::collections::BTreeMap;

use openjiuwen_common::error::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::tune::base::Case;
use crate::tune::tool::RunTelemetry;

/// Offset basis of the 64-bit FNV-1a hash.
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
/// Prime of the 64-bit FNV-1a hash.
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Renders one JSON value the way Python's `str()` renders the common shapes.
///
/// A string is returned bare; every other value is rendered as compact JSON
/// (`4` stays `4`, an object stays `{"a":1}`).
fn render_value(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => other.to_string(),
    }
}

/// The first value of `data` under one of `preferred` keys, else its first value.
///
/// Mirrors Python's `_first_value`: the preferred keys are tried in order, then
/// the map's iteration order (deterministic here, because the maps are
/// `BTreeMap`s), and an empty map yields `""`.
fn first_value(data: &BTreeMap<String, Value>, preferred: &[&str]) -> String {
    for key in preferred {
        if let Some(value) = data.get(*key) {
            return render_value(value);
        }
    }
    data.values().next().map_or_else(String::new, render_value)
}

/// A deterministic replacement for Python's `uuid4().hex[:8]` candidate id.
///
/// Eight lowercase hex characters of the 64-bit FNV-1a hash over
/// `"{index}\x1f{prompt}"`. The index keeps two candidates with an identical
/// prompt distinct, and hashing keeps ids stable across processes, so a run —
/// and a test — produces the same ids every time.
///
/// # Examples
///
/// ```
/// # use openjiuwen_dev_tools::tune::candidate_id;
/// let id = candidate_id("Answer carefully.", 0);
/// assert_eq!(id.len(), 8);
/// assert_eq!(id, candidate_id("Answer carefully.", 0));
/// assert_ne!(id, candidate_id("Answer carefully.", 1));
/// ```
#[must_use]
pub fn candidate_id(prompt: &str, index: usize) -> String {
    let mut hash = FNV_OFFSET;
    for byte in format!("{index}\u{1f}{prompt}").as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    format!("{:08x}", hash as u32)
}

/// One evaluation case, plus whether it is held out from the visible set.
///
/// Hidden cases are never shown to the policy; the reward model uses them to
/// detect a candidate that overfits the visible cases (reward hacking).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct PromptTaskCase {
    /// The text handed to the candidate prompt.
    pub input_text: String,
    /// The expected answer text used for grading.
    pub expected_text: String,
    /// Whether the case is held out from the visible set.
    pub hidden: bool,
}

impl PromptTaskCase {
    /// Builds the common single-string case, as Python's
    /// `PromptTaskCase.from_text(text, expected)` does.
    ///
    /// Delegates to [`Case::from_text`], so a blank `text` is
    /// [`Error::validation`](openjiuwen_common::error::Error::validation).
    ///
    /// # Errors
    ///
    /// [`Error::validation`](openjiuwen_common::error::Error::validation) when
    /// `text` is blank.
    pub fn from_text(text: impl Into<String>, expected: impl Into<String>) -> Result<Self> {
        let case = Case::from_text(text, expected)?;
        Ok(Self::from_case(&case, false))
    }

    /// Marks the case as held out (or visible).
    #[must_use]
    pub fn with_hidden(mut self, hidden: bool) -> Self {
        self.hidden = hidden;
        self
    }

    /// Extracts the case's input and expected text from a [`Case`].
    ///
    /// The input prefers `inputs["query"]`, then `inputs["input"]`, then the
    /// first input value; the expected text prefers `label["output"]`, then
    /// `label["expected"]`, then the first label value. String values are
    /// returned bare, other JSON values compactly (Python's `str(value)`).
    #[must_use]
    pub fn from_case(case: &Case, hidden: bool) -> Self {
        Self {
            input_text: first_value(&case.inputs, &["query", "input"]),
            expected_text: first_value(&case.label, &["output", "expected"]),
            hidden,
        }
    }

    /// Converts back into the shared tuning [`Case`] (`{"query": …}` /
    /// `{"output": …}`).
    ///
    /// # Errors
    ///
    /// [`Error::validation`](openjiuwen_common::error::Error::validation) when
    /// [`PromptTaskCase::input_text`] is blank.
    pub fn to_case(&self) -> Result<Case> {
        Case::from_text(self.input_text.clone(), self.expected_text.clone())
    }
}

/// What the optimizer is optimizing a prompt *for*.
///
/// `objective` is the immutable intent used for drift protection; `cases` are
/// the evaluation inputs, those flagged `hidden` forming the held-out set.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct PromptTaskSpec {
    /// The immutable task intent.
    pub objective: String,
    /// The evaluation cases, visible and hidden.
    pub cases: Vec<PromptTaskCase>,
    /// Additional constraints the candidate prompt must respect.
    pub constraints: Vec<String>,
    /// The prompt the optimization starts from.
    pub base_prompt: String,
    /// Free-form metadata carried with the task.
    pub metadata: BTreeMap<String, Value>,
}

impl PromptTaskSpec {
    /// Creates a task spec for `objective` with no cases, constraints, base
    /// prompt, or metadata.
    #[must_use]
    pub fn new(objective: impl Into<String>) -> Self {
        Self {
            objective: objective.into(),
            ..Self::default()
        }
    }

    /// Replaces the evaluation cases.
    #[must_use]
    pub fn with_cases(mut self, cases: Vec<PromptTaskCase>) -> Self {
        self.cases = cases;
        self
    }

    /// Replaces the constraints.
    #[must_use]
    pub fn with_constraints(mut self, constraints: Vec<String>) -> Self {
        self.constraints = constraints;
        self
    }

    /// Replaces the base prompt.
    #[must_use]
    pub fn with_base_prompt(mut self, base_prompt: impl Into<String>) -> Self {
        self.base_prompt = base_prompt.into();
        self
    }

    /// Replaces the metadata.
    #[must_use]
    pub fn with_metadata(mut self, metadata: BTreeMap<String, Value>) -> Self {
        self.metadata = metadata;
        self
    }

    /// A compact text signature used for memory retrieval.
    ///
    /// The objective, plus `constraints: a; b` when any constraint exists.
    #[must_use]
    pub fn characteristics(&self) -> String {
        let mut parts = vec![self.objective.clone()];
        if !self.constraints.is_empty() {
            parts.push(format!("constraints: {}", self.constraints.join("; ")));
        }
        parts.join("\n")
    }

    /// The cases shown to the policy.
    #[must_use]
    pub fn visible_cases(&self) -> Vec<&PromptTaskCase> {
        self.cases.iter().filter(|case| !case.hidden).collect()
    }

    /// The held-out cases used for overfitting detection.
    #[must_use]
    pub fn hidden_cases(&self) -> Vec<&PromptTaskCase> {
        self.cases.iter().filter(|case| case.hidden).collect()
    }
}

/// One candidate system prompt produced by the policy.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct PromptCandidate {
    /// The candidate system prompt.
    pub prompt: String,
    /// A deterministic short identifier (see [`candidate_id`]).
    pub candidate_id: String,
    /// Why the policy produced this candidate.
    pub rationale: String,
}

impl PromptCandidate {
    /// Creates a candidate whose id is [`candidate_id`] with index `0`.
    #[must_use]
    pub fn new(prompt: impl Into<String>, rationale: impl Into<String>) -> Self {
        let prompt = prompt.into();
        let candidate_id = candidate_id(&prompt, 0);
        Self {
            prompt,
            candidate_id,
            rationale: rationale.into(),
        }
    }

    /// Replaces the deterministic identifier.
    #[must_use]
    pub fn with_id(mut self, candidate_id: impl Into<String>) -> Self {
        self.candidate_id = candidate_id.into();
        self
    }

    /// The candidate system prompt.
    #[must_use]
    pub fn prompt(&self) -> &str {
        &self.prompt
    }

    /// The deterministic identifier.
    #[must_use]
    pub fn candidate_id(&self) -> &str {
        &self.candidate_id
    }

    /// Why the policy produced this candidate.
    #[must_use]
    pub fn rationale(&self) -> &str {
        &self.rationale
    }
}

/// The outcome of running a candidate prompt against one case.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct CaseResult {
    /// The identifier of the case the candidate ran on.
    pub case_id: String,
    /// The candidate's output for the case.
    pub output: String,
    /// The per-case score the caller's evaluator assigned, in `0.0..=1.0`.
    pub score: f64,
    /// The failure, when the run failed for this case.
    pub error: Option<String>,
    /// Whether the case was part of the held-out set.
    pub hidden: bool,
    /// What this case's run cost, as measured by the caller's evaluator.
    ///
    /// Absent fields mean *not reported* rather than zero, so a case that
    /// reported nothing keeps [`RunTelemetry::default`].
    pub telemetry: RunTelemetry,
}

impl CaseResult {
    /// Creates a result with score `0.0`, no error, and `hidden = false`.
    #[must_use]
    pub fn new(case_id: impl Into<String>, output: impl Into<String>) -> Self {
        Self {
            case_id: case_id.into(),
            output: output.into(),
            ..Self::default()
        }
    }

    /// Sets the per-case score.
    #[must_use]
    pub fn with_score(mut self, score: f64) -> Self {
        self.score = score;
        self
    }

    /// Marks the result as held out (or visible).
    #[must_use]
    pub fn with_hidden(mut self, hidden: bool) -> Self {
        self.hidden = hidden;
        self
    }

    /// Sets what the case's run cost.
    #[must_use]
    pub fn with_telemetry(mut self, telemetry: RunTelemetry) -> Self {
        self.telemetry = telemetry;
        self
    }

    /// Whether the case ran without an error.
    #[must_use]
    pub fn ok(&self) -> bool {
        self.error.is_none()
    }

    /// An empty placeholder result, useful for aligning results with cases.
    #[must_use]
    pub fn blank() -> Self {
        Self::default()
    }
}

/// The outcome of executing a candidate prompt in the environment.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct Execution {
    /// The candidate that produced this execution.
    pub candidate: PromptCandidate,
    /// The run's wall-clock latency, as supplied by the caller.
    ///
    /// A legacy field kept for wire compatibility: it is only consulted when
    /// [`Execution::telemetry`] reports no latency (see
    /// [`Execution::measured_latency_s`]).
    pub latency_ms: f64,
    /// The run's total token usage, as supplied by the caller.
    ///
    /// A legacy field kept for wire compatibility: it is only consulted when
    /// [`Execution::telemetry`] reports no token counts (see
    /// [`Execution::measured_total_tokens`]).
    pub total_tokens: u64,
    /// The failure, when the whole run failed.
    pub error: Option<String>,
    /// The per-case results, visible and hidden.
    pub results: Vec<CaseResult>,
    /// What the run cost, as measured by the caller's evaluator.
    ///
    /// Absent fields mean *not reported* rather than zero, so an evaluator that
    /// measures nothing leaves this at [`RunTelemetry::default`] and the legacy
    /// `latency_ms`/`total_tokens` fields stay authoritative.
    pub telemetry: RunTelemetry,
}

impl Execution {
    /// Creates an execution of `candidate` with no results, zero latency and
    /// tokens, and no error.
    #[must_use]
    pub fn new(candidate: PromptCandidate) -> Self {
        Self {
            candidate,
            ..Self::default()
        }
    }

    /// Appends one case result.
    pub fn push_result(&mut self, result: CaseResult) {
        self.results.push(result);
    }

    /// The latency in seconds, from the legacy caller-supplied field only.
    #[must_use]
    pub fn latency_s(&self) -> f64 {
        self.latency_ms / 1000.0
    }

    /// The latency in seconds, preferring the measured telemetry.
    ///
    /// Uses [`Execution::telemetry`]'s latency when it was reported, otherwise
    /// falls back to the legacy caller-supplied `latency_ms`. Both are zero when
    /// nothing was reported at all, matching [`Execution::latency_s`].
    #[must_use]
    pub fn measured_latency_s(&self) -> f64 {
        self.telemetry.latency_ms.unwrap_or(self.latency_ms) / 1000.0
    }

    /// The total tokens, preferring the measured telemetry.
    ///
    /// Uses [`RunTelemetry::total_tokens`] when either token side was reported,
    /// otherwise falls back to the legacy caller-supplied `total_tokens`.
    #[must_use]
    pub fn measured_total_tokens(&self) -> u64 {
        self.telemetry.total_tokens().unwrap_or(self.total_tokens)
    }

    /// The measured monetary cost of the run, when the caller reported one.
    ///
    /// There is no legacy cost field, so an absent measurement stays absent
    /// rather than being inferred from tokens.
    #[must_use]
    pub fn measured_cost(&self) -> Option<f64> {
        self.telemetry.cost
    }

    /// The non-held-out results, in run order.
    #[must_use]
    pub fn visible_results(&self) -> Vec<&CaseResult> {
        self.results
            .iter()
            .filter(|result| !result.hidden)
            .collect()
    }

    /// The held-out results, in run order.
    #[must_use]
    pub fn hidden_results(&self) -> Vec<&CaseResult> {
        self.results.iter().filter(|result| result.hidden).collect()
    }

    /// The visible outputs, empty ones skipped, joined by `\n\n---\n\n`.
    #[must_use]
    pub fn combined_output(&self) -> String {
        join_outputs(self.visible_results())
    }

    /// The held-out outputs, empty ones skipped, joined by `\n\n---\n\n`.
    #[must_use]
    pub fn combined_output_hidden(&self) -> String {
        join_outputs(self.hidden_results())
    }
}

/// Joins the non-empty outputs of `results`.
fn join_outputs(results: Vec<&CaseResult>) -> String {
    results
        .into_iter()
        .map(|result| result.output.as_str())
        .filter(|output| !output.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n---\n\n")
}

/// Scalar reward plus the per-component contributions that produced it.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct RewardBreakdown {
    /// The final reward, after gating and penalties, in `0.0..=1.0`.
    pub score: f64,
    /// The (possibly normalized) per-component values that were summed.
    pub components: BTreeMap<String, f64>,
    /// The per-component values before normalization.
    pub raw_components: BTreeMap<String, f64>,
    /// The raw correctness value (`1.0` when no correctness component exists).
    pub correctness: f64,
    /// The semantic deviation from the objective, in `0.0..=1.0`.
    pub drift: f64,
    /// How much the drift penalty subtracted.
    pub drift_penalty_applied: f64,
    /// Whether the min-correctness gate capped the reward.
    pub gated: bool,
    /// Human-readable notes describing every adjustment that was applied.
    pub notes: Vec<String>,
}

impl RewardBreakdown {
    /// Whether the min-correctness gate capped the reward.
    #[must_use]
    pub fn is_gated(&self) -> bool {
        self.gated
    }
}

/// A candidate paired with its execution and reward — the loop's unit of
/// comparison.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct Evaluation {
    /// The execution of the candidate.
    pub execution: Execution,
    /// The reward computed for that execution.
    pub reward: RewardBreakdown,
}

impl Evaluation {
    /// The evaluated candidate.
    #[must_use]
    pub fn candidate(&self) -> &PromptCandidate {
        &self.execution.candidate
    }

    /// The final reward score.
    #[must_use]
    pub fn score(&self) -> f64 {
        self.reward.score
    }
}

/// Everything that happened in one optimization round, for logging.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct IterationRecord {
    /// The 1-based round number (Python's `iteration`).
    pub index: usize,
    /// The best score seen in this round.
    pub best_score: f64,
    /// The candidates tried in this round, in evaluation order.
    pub candidates: Vec<PromptCandidate>,
    /// Their scores, in the same order.
    pub scores: Vec<f64>,
}

impl IterationRecord {
    /// Creates an empty record for round `index`.
    #[must_use]
    pub fn new(index: usize) -> Self {
        Self {
            index,
            ..Self::default()
        }
    }

    /// Records one evaluated candidate and tracks the round's best score.
    pub fn push(&mut self, candidate: PromptCandidate, score: f64) {
        self.best_score = self.best_score.max(score);
        self.candidates.push(candidate);
        self.scores.push(score);
    }

    /// The best-scoring candidate of the round; ties keep the first one.
    #[must_use]
    pub fn best_candidate(&self) -> Option<&PromptCandidate> {
        let mut best: Option<(usize, f64)> = None;
        for index in 0..self.candidates.len() {
            let score = self.scores.get(index).copied().unwrap_or(0.0);
            if best.is_none_or(|(_, best_score)| score > best_score) {
                best = Some((index, score));
            }
        }
        best.map(|(index, _)| &self.candidates[index])
    }
}

/// A persisted optimization outcome for the prompt memory.
///
/// `baseline_reward` is the best reward previously known for the same
/// objective, so [`PromptRecord::gain`] shows the actual improvement.
/// `applied` tracks whether a human installed this prompt somewhere; the
/// optimizer never sets it itself, and `applied_at` is supplied by the caller
/// (nothing here reads a clock).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct PromptRecord {
    /// The identifier of the record.
    pub record_id: String,
    /// The optimized prompt.
    pub prompt: String,
    /// The reward the prompt earned.
    pub reward: f64,
    /// The objective the prompt was optimized for.
    pub objective: String,
    /// The task characteristics used for similarity retrieval.
    pub task_characteristics: String,
    /// The reward components and other metrics of the run.
    pub metrics: BTreeMap<String, Value>,
    /// Observations recorded during the run.
    pub observations: Vec<String>,
    /// Free-form metadata carried with the record.
    pub metadata: BTreeMap<String, Value>,
    /// The creation timestamp, supplied by the caller.
    pub created_at: f64,
    /// The best reward previously known for the same objective.
    pub baseline_reward: Option<f64>,
    /// Whether a human installed this prompt.
    pub applied: bool,
    /// When the prompt was installed, supplied by the caller.
    pub applied_at: Option<f64>,
}

impl PromptRecord {
    /// Creates a record with the caller-supplied identifier and timestamp `0.0`.
    #[must_use]
    pub fn new(
        record_id: impl Into<String>,
        prompt: impl Into<String>,
        reward: f64,
        objective: impl Into<String>,
    ) -> Self {
        Self {
            record_id: record_id.into(),
            prompt: prompt.into(),
            reward,
            objective: objective.into(),
            ..Self::default()
        }
    }

    /// Replaces the task characteristics.
    #[must_use]
    pub fn with_characteristics(mut self, characteristics: impl Into<String>) -> Self {
        self.task_characteristics = characteristics.into();
        self
    }

    /// Replaces the metrics.
    #[must_use]
    pub fn with_metrics(mut self, metrics: BTreeMap<String, Value>) -> Self {
        self.metrics = metrics;
        self
    }

    /// Replaces the observations.
    #[must_use]
    pub fn with_observations(mut self, observations: Vec<String>) -> Self {
        self.observations = observations;
        self
    }

    /// Replaces the metadata.
    #[must_use]
    pub fn with_metadata(mut self, metadata: BTreeMap<String, Value>) -> Self {
        self.metadata = metadata;
        self
    }

    /// Sets the creation timestamp.
    #[must_use]
    pub fn with_created_at(mut self, created_at: f64) -> Self {
        self.created_at = created_at;
        self
    }

    /// Sets the baseline reward the gain is measured against.
    #[must_use]
    pub fn with_baseline_reward(mut self, baseline_reward: f64) -> Self {
        self.baseline_reward = Some(baseline_reward);
        self
    }

    /// Sets the applied flag and its timestamp.
    #[must_use]
    pub fn with_applied(mut self, applied: bool, applied_at: Option<f64>) -> Self {
        self.applied = applied;
        self.applied_at = applied_at;
        self
    }

    /// The improvement over [`PromptRecord::baseline_reward`] (`0.0` when none).
    #[must_use]
    pub fn gain(&self) -> f64 {
        self.reward - self.baseline_reward.unwrap_or(0.0)
    }

    /// Whether the record is unapplied and its gain exceeds `threshold`.
    #[must_use]
    pub fn is_pending(&self, threshold: f64) -> bool {
        !self.applied && self.gain() > threshold
    }
}

#[cfg(test)]
#[path = "models_tests.rs"]
mod tests;
