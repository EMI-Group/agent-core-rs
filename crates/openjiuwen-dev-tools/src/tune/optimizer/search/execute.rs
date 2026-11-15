//! The prompt-search round mechanics: execution, drift, memory, context.
//!
//! Split out of the loop in the parent module so the search itself stays
//! readable. Everything here is a pure helper over the prompt-search vocabulary:
//! it executes one candidate through the evaluator seam, batches candidates for
//! parallel execution, gathers drift deviations, warms the run from prompt
//! memory, renders the round's policy context, and derives the per-round record
//! extras (observations, metrics, metadata).

use std::collections::BTreeMap;

use serde_json::Value;

use super::WARM_START_LIMIT;
use crate::tune::base::{Case, EvaluatedCase};
use crate::tune::evaluator::CaseEvaluator;
use crate::tune::prompt_search::{
    CaseResult, DriftJudge, Execution, PromptCandidate, PromptMemory, PromptRecord, PromptTaskSpec,
    RewardBreakdown,
};

/// One candidate's run plus the cases that evaluated successfully.
///
/// The evaluated cases are the evidence a model rewrite is shown (the worst
/// ones); a case whose evaluation failed appears only as an errored
/// [`CaseResult`] on the execution.
pub(super) struct CandidateOutcome {
    /// The execution scored by the composite reward.
    pub(super) execution: Execution,
    /// The successfully evaluated cases, in run order.
    pub(super) evaluated: Vec<EvaluatedCase>,
}

/// The round-1 seed: the task's base prompt, or its objective when that is
/// blank.
pub(super) fn base_prompt(task: &PromptTaskSpec) -> String {
    let base = task.base_prompt.trim();
    if base.is_empty() {
        task.objective.trim().to_owned()
    } else {
        base.to_owned()
    }
}

/// The `top_k` most similar prior records, highest reward first.
pub(super) fn warm_start(memory: &dyn PromptMemory, task: &PromptTaskSpec) -> Vec<PromptRecord> {
    let mut records = memory.search_similar(task, WARM_START_LIMIT);
    records.truncate(WARM_START_LIMIT);
    records.sort_by(|left, right| {
        right
            .reward
            .partial_cmp(&left.reward)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.prompt.cmp(&right.prompt))
    });
    records
}

/// The textual policy update handed to the next round's model rewrite.
///
/// The compressed lessons plus the warm-start prompts, or `None` when there is
/// nothing to say (which keeps the rewrite request byte-identical to the other
/// optimizers').
pub(super) fn policy_context(lessons: &str, warm: &[PromptRecord]) -> Option<String> {
    let mut parts = Vec::new();
    if !lessons.trim().is_empty() {
        parts.push(format!("## Distilled lessons\n{}", lessons.trim()));
    }
    if !warm.is_empty() {
        let prompts = warm
            .iter()
            .map(|record| format!("- {} (reward {:.3})", record.prompt.trim(), record.reward))
            .collect::<Vec<_>>()
            .join("\n");
        parts.push(format!("## Prior successful prompts\n{prompts}"));
    }
    if parts.is_empty() {
        None
    } else {
        Some(parts.join("\n\n"))
    }
}

/// The drift of every candidate against the task objective.
///
/// A zero drift penalty skips the judge entirely and reports all zeros, which is
/// Python's early return.
pub(super) async fn drift_scores(
    judge: &dyn DriftJudge,
    executions: &[Execution],
    task: &PromptTaskSpec,
    drift_penalty: f64,
) -> BTreeMap<String, f64> {
    let mut scores = BTreeMap::new();
    for execution in executions {
        let candidate_id = execution.candidate.candidate_id.clone();
        let deviation = if drift_penalty == 0.0 {
            0.0
        } else {
            judge.deviation(&task.objective, &execution.candidate).await
        };
        scores.insert(candidate_id, deviation);
    }
    scores
}

/// Executes `candidates` over every case, `num_parallel` candidates at a time.
///
/// Results come back in input order regardless of completion order, so a round
/// stays deterministic.
pub(super) async fn execute_candidates(
    evaluator: &dyn CaseEvaluator,
    candidates: &[PromptCandidate],
    cases: &[Case],
    hidden: &[bool],
    num_parallel: usize,
) -> Vec<CandidateOutcome> {
    let mut outcomes = Vec::with_capacity(candidates.len());
    for batch in candidates.chunks(num_parallel.max(1)) {
        let futures = batch
            .iter()
            .map(|candidate| execute_candidate(evaluator, candidate.clone(), cases, hidden));
        outcomes.extend(futures::future::join_all(futures).await);
    }
    outcomes
}

/// Executes one candidate over every case.
///
/// One [`CaseResult`] per case, keyed `case-{index}` (a [`PromptTaskCase`] has no
/// id of its own). Each case is evaluated through
/// [`CaseEvaluator::evaluate_run`], so the run's measured
/// [`RunTelemetry`](crate::tune::RunTelemetry) is kept on that case's
/// [`CaseResult`] and merged into [`Execution::telemetry`]: the execution's
/// latency and token totals are the sum over its cases. The legacy
/// `latency_ms`/`total_tokens` fields are filled from that merged telemetry only
/// when it was measured, so absent telemetry leaves them at `0.0`/`0` rather than
/// fabricating a measurement.
///
/// A failing evaluation becomes that result's `error` and the run continues; the
/// execution is marked failed only when every case failed.
pub(super) async fn execute_candidate(
    evaluator: &dyn CaseEvaluator,
    candidate: PromptCandidate,
    cases: &[Case],
    hidden: &[bool],
) -> CandidateOutcome {
    let prompt = candidate.prompt().to_owned();
    let mut execution = Execution::new(candidate);
    let mut evaluated = Vec::new();
    for (index, case) in cases.iter().enumerate() {
        let case_id = format!("case-{index}");
        let is_hidden = hidden.get(index).copied().unwrap_or(false);
        match evaluator.evaluate_run(&prompt, case).await {
            Ok(run) => {
                execution.push_result(
                    CaseResult::new(&case_id, answer_text(&run.evaluated.answer))
                        .with_score(run.evaluated.score)
                        .with_hidden(is_hidden)
                        .with_telemetry(run.telemetry),
                );
                execution.telemetry.merge(run.telemetry);
                evaluated.push(run.evaluated);
            }
            Err(error) => {
                let mut failure = CaseResult::new(&case_id, String::new()).with_hidden(is_hidden);
                failure.error = Some(error.to_string());
                execution.push_result(failure);
            }
        }
    }
    if !execution.telemetry.is_empty() {
        execution.latency_ms = execution.telemetry.latency_ms.unwrap_or(0.0);
        execution.total_tokens = execution.telemetry.total_tokens().unwrap_or(0);
    }
    if !execution.results.is_empty() && execution.results.iter().all(|result| !result.ok()) {
        execution.error = Some(format!("all {} case(s) failed", execution.results.len()));
    }
    CandidateOutcome {
        execution,
        evaluated,
    }
}

/// The text form of an answer map: `answer["output"]` verbatim when it is a
/// string, that value compactly rendered otherwise, the whole map compactly
/// rendered when there is no `output`, and `""` for an empty map.
pub(super) fn answer_text(answer: &BTreeMap<String, Value>) -> String {
    match answer.get("output") {
        Some(Value::String(text)) => text.clone(),
        Some(other) => other.to_string(),
        None if answer.is_empty() => String::new(),
        None => serde_json::to_string(answer).unwrap_or_default(),
    }
}

/// The index of the highest score, ties keeping the earlier candidate.
pub(super) fn best_index(scores: &[f64]) -> Option<usize> {
    let mut best: Option<usize> = None;
    for (index, score) in scores.iter().enumerate() {
        match best {
            None => best = Some(index),
            Some(current) if *score > scores[current] => best = Some(index),
            Some(_) => {}
        }
    }
    best
}

/// Python's `_observations`: the rationale, the reward notes, and the strongest
/// and weakest raw metric.
pub(super) fn observations(candidate: &PromptCandidate, reward: &RewardBreakdown) -> Vec<String> {
    let mut observations = Vec::new();
    if !candidate.rationale.trim().is_empty() {
        observations.push(candidate.rationale.clone());
    }
    observations.extend(reward.notes.iter().cloned());
    if reward.raw_components.len() >= 2 {
        let mut entries: Vec<(&String, f64)> = reward
            .raw_components
            .iter()
            .map(|(name, value)| (name, *value))
            .collect();
        entries.sort_by(|left, right| {
            right
                .1
                .partial_cmp(&left.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.0.cmp(right.0))
        });
        let (strongest, strongest_value) = entries[0];
        let (weakest, weakest_value) = entries[entries.len() - 1];
        observations.push(format!(
            "strongest metric: {strongest} ({strongest_value:.2}); weakest: {weakest} ({weakest_value:.2})"
        ));
    }
    observations
}

/// Renders reward components as the record's JSON metrics.
pub(super) fn metrics_of(raw_components: &BTreeMap<String, f64>) -> BTreeMap<String, Value> {
    raw_components
        .iter()
        .map(|(name, value)| (name.clone(), Value::from(*value)))
        .collect()
}

/// The record metadata Python writes: the round count plus the task metadata.
pub(super) fn record_metadata(task: &PromptTaskSpec, iterations: usize) -> BTreeMap<String, Value> {
    let mut metadata = BTreeMap::new();
    metadata.insert("iterations".to_owned(), Value::from(iterations));
    for (key, value) in &task.metadata {
        metadata.insert(key.clone(), value.clone());
    }
    metadata
}
