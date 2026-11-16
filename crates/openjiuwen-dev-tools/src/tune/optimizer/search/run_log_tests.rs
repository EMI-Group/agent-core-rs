//! The prompt-search run-log suite: one record per round, determinism, the
//! adoption verdict, and the never-fail contract.
//!
//! Everything here is offline: the same fixed-score evaluator, fixed cases, and
//! a caller-supplied timestamp, so a record is a pure function of the run.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde_json::Value;

use super::*;
use crate::tune::base::EvaluatedCase;
use crate::tune::optimizer::adopt::{AdoptConfig, AdoptReason};
use crate::tune::prompt_search::{InMemoryRunLog, RunLog, RunLogRecord};

fn case(id: &str, question: &str, answer: &str) -> Case {
    let mut inputs = BTreeMap::new();
    inputs.insert("question".to_owned(), Value::String(question.to_owned()));
    let mut label = BTreeMap::new();
    label.insert("answer".to_owned(), Value::String(answer.to_owned()));
    Case::new(inputs, label).unwrap().with_case_id(id)
}

fn cases() -> Vec<Case> {
    vec![
        case("c1", "capital of France", "Paris"),
        case("c2", "2 + 2", "four"),
    ]
}

const SEED: &str = "Answer the question.";

/// The timestamp every logged run in this suite stamps its records with.
const NOW: f64 = 1_700_000_000.25;

/// A shared in-memory log plus the handle an optimizer takes.
///
/// The concrete handle stays with the test so it can read back what the run
/// wrote through the `dyn RunLog` handle.
fn shared_log() -> (Arc<Mutex<InMemoryRunLog>>, Arc<Mutex<dyn RunLog>>) {
    let concrete = Arc::new(Mutex::new(InMemoryRunLog::new()));
    let handle: Arc<Mutex<dyn RunLog>> = concrete.clone();
    (concrete, handle)
}

/// The records the log holds, in write order.
fn records(concrete: &Arc<Mutex<InMemoryRunLog>>) -> Vec<RunLogRecord> {
    concrete
        .lock()
        .expect("the run log is not poisoned")
        .records()
        .to_vec()
}

/// Scores every prompt and case the same, so elitism keeps the seed.
struct Fixed {
    score: f64,
}

#[async_trait]
impl CaseEvaluator for Fixed {
    async fn evaluate(&self, _prompt: &str, case: &Case) -> Result<EvaluatedCase> {
        let mut answer = BTreeMap::new();
        answer.insert(
            "output".to_owned(),
            Value::String("complete answer".to_owned()),
        );
        EvaluatedCase::new(case.clone(), answer, self.score, "fixed")
    }
}

/// Scores a prompt by how many deterministic mutation markers it carries, capped
/// at two, so one round improves the elite and the next only matches it.
struct MarkerCount;

/// The marker phrases of the deterministic mutation bank.
const MARKERS: &[&str] = &[
    "think step by step",
    "output format",
    "never invent facts",
    "before finishing, verify",
    "follow the constraints",
    "use the provided context only",
];

#[async_trait]
impl CaseEvaluator for MarkerCount {
    async fn evaluate(&self, prompt: &str, case: &Case) -> Result<EvaluatedCase> {
        let lower = prompt.to_lowercase();
        let markers = MARKERS
            .iter()
            .filter(|marker| lower.contains(**marker))
            .count();
        let score = 0.1 + 0.2 * markers.min(2) as f64;
        let mut answer = BTreeMap::new();
        answer.insert(
            "output".to_owned(),
            Value::String("complete answer".to_owned()),
        );
        EvaluatedCase::new(case.clone(), answer, score, "marker count")
    }
}

/// A sink that rejects every record.
struct Rejecting;

impl RunLog for Rejecting {
    fn record(&mut self, _record: &RunLogRecord) -> Result<()> {
        Err(Error::validation("this sink rejects every record"))
    }
}

/// The unlogged reference run every comparison in this suite uses.
async fn reference() -> OptimizationResult {
    PromptSearchOptimizer::new()
        .with_evaluator(Arc::new(Fixed { score: 0.8 }))
        .optimize(SEED, &cases())
        .await
        .unwrap()
}

#[tokio::test]
async fn one_record_per_round_with_ascending_indices() {
    let (concrete, handle) = shared_log();
    let result = PromptSearchOptimizer::new()
        .with_evaluator(Arc::new(Fixed { score: 0.8 }))
        .with_run_log(handle)
        .with_timestamp(NOW)
        .optimize(SEED, &cases())
        .await
        .unwrap();

    let logged = records(&concrete);
    assert_eq!(logged.len(), result.iterations);
    assert!(logged.len() >= 2, "a plateau takes several rounds");
    assert_eq!(
        logged
            .iter()
            .map(|record| record.iteration)
            .collect::<Vec<_>>(),
        (1..=result.iterations).collect::<Vec<_>>()
    );
    assert!(logged
        .iter()
        .all(|record| !record.candidate.prompt.trim().is_empty()));
    assert!(logged.iter().all(|record| record.timestamp == NOW));
    assert!(logged.iter().all(|record| record.run_id == result.run_id));
    // The last record reports the run's final best reward.
    assert_eq!(logged.last().unwrap().best_score, result.score);
}

#[tokio::test]
async fn every_record_mirrors_its_round() {
    let (concrete, handle) = shared_log();
    let result = PromptSearchOptimizer::new()
        .with_evaluator(Arc::new(Fixed { score: 0.8 }))
        .with_run_log(handle)
        .optimize(SEED, &cases())
        .await
        .unwrap();

    let logged = records(&concrete);
    for (record, round) in logged.iter().zip(&result.records) {
        assert_eq!(record.iteration, round.index);
        assert_eq!(record.scores.len(), round.candidates.len());
        assert!(record.scores.contains_key(&record.candidate.candidate_id));
        assert_eq!(record.candidate, round.candidates[0]);
        assert_eq!(
            record.candidate_prompts,
            round
                .candidates
                .iter()
                .skip(1)
                .map(|candidate| candidate.prompt.clone())
                .collect::<Vec<_>>()
        );
        // The reward is the winner's score, and its components are its raw ones.
        assert_eq!(record.reward, record.scores[&record.candidate.candidate_id]);
        assert_eq!(
            record.reward_components.get("correctness"),
            Some(&0.8),
            "{:?}",
            record.reward_components
        );
        assert!(record.observation.is_some());
    }
}

#[tokio::test]
async fn selected_marks_the_rounds_that_moved_the_elite() {
    let (concrete, handle) = shared_log();
    PromptSearchOptimizer::new()
        .with_evaluator(Arc::new(Fixed { score: 0.8 }))
        .with_run_log(handle)
        .optimize(SEED, &cases())
        .await
        .unwrap();

    let logged = records(&concrete);
    // A flat run establishes the elite in round 1 and never moves it again.
    assert!(logged[0].selected);
    assert!(logged[1..].iter().all(|record| !record.selected));
}

#[tokio::test]
async fn only_the_stopping_round_reports_convergence() {
    let (concrete, handle) = shared_log();
    let result = PromptSearchOptimizer::new()
        .with_evaluator(Arc::new(Fixed { score: 0.8 }))
        .with_run_log(handle)
        .optimize(SEED, &cases())
        .await
        .unwrap();

    let logged = records(&concrete);
    let last = logged.last().unwrap();
    assert!(result.converged);
    assert!(last.converged);
    assert_eq!(last.convergence_reason, result.convergence_reason);
    assert!(last.convergence.is_some(), "the detector fired");
    assert!(logged[..logged.len() - 1]
        .iter()
        .all(|record| !record.converged && record.convergence.is_none()));
}

#[tokio::test]
async fn repeated_runs_produce_equal_records() {
    let (first_concrete, first_handle) = shared_log();
    let first = PromptSearchOptimizer::new()
        .with_evaluator(Arc::new(Fixed { score: 0.8 }))
        .with_run_log(first_handle)
        .with_timestamp(NOW)
        .optimize(SEED, &cases())
        .await
        .unwrap();
    let (second_concrete, second_handle) = shared_log();
    let second = PromptSearchOptimizer::new()
        .with_evaluator(Arc::new(Fixed { score: 0.8 }))
        .with_run_log(second_handle)
        .with_timestamp(NOW)
        .optimize(SEED, &cases())
        .await
        .unwrap();

    assert_eq!(first, second);
    assert_eq!(records(&first_concrete), records(&second_concrete));
    assert_ne!(records(&first_concrete).len(), 0);
}

#[tokio::test]
async fn the_adoption_verdict_is_absent_without_a_policy() {
    let (concrete, handle) = shared_log();
    PromptSearchOptimizer::new()
        .with_evaluator(Arc::new(Fixed { score: 0.8 }))
        .with_run_log(handle)
        .optimize(SEED, &cases())
        .await
        .unwrap();

    let logged = records(&concrete);
    assert_ne!(logged.len(), 0);
    assert!(logged.iter().all(|record| record.adopt.is_none()));
    assert!(logged.iter().all(|record| !record.adopted()));
}

#[tokio::test]
async fn a_policy_that_demands_a_gain_rejects_every_later_round() {
    let (concrete, handle) = shared_log();
    let policy = AdoptPolicy::new().with_config(AdoptConfig::new().with_min_gain(1.0));
    let result = PromptSearchOptimizer::new()
        .with_evaluator(Arc::new(Fixed { score: 0.8 }))
        .with_adopt_policy(policy)
        .with_run_log(handle)
        .optimize(SEED, &cases())
        .await
        .unwrap();

    let logged = records(&concrete);
    assert_eq!(logged.len(), result.iterations);
    // No history in round 1, so the first candidate is the baseline.
    let first = logged[0].adopt.expect("a decision per round");
    assert!(first.adopted());
    assert_eq!(first.reason(), &AdoptReason::FirstCandidate);
    // A gain of at most 1.0 can never clear a threshold of 1.0.
    for record in &logged[1..] {
        let decision = record.adopt.expect("a decision per round");
        assert!(!decision.adopted(), "{:?}", decision.reason());
        assert!(matches!(
            decision.reason(),
            AdoptReason::NoImprovement { .. }
        ));
        assert!(!record.adopted());
    }
    // The verdict is reporting only: the search itself is unchanged.
    assert_eq!(result, reference().await);
}

#[tokio::test]
async fn is_improvement_marks_the_round_that_advances_the_elite() {
    let (concrete, handle) = shared_log();
    // No gate and no drift penalty, so the reward is a fixed monotone function of
    // the evaluator's per-case score and a round's effect on the elite is
    // exactly its winning score's effect.
    let search = PromptSearchConfig::new()
        .with_min_correctness(0.0)
        .with_drift_penalty(0.0);
    let result = PromptSearchOptimizer::new()
        .with_evaluator(Arc::new(MarkerCount))
        .with_search_config(search)
        .with_run_log(handle)
        .with_timestamp(NOW)
        .optimize(SEED, &cases())
        .await
        .unwrap();

    let logged = records(&concrete);
    assert_eq!(logged.len(), result.iterations);
    assert_eq!(logged.len(), 3, "the default budget is three rounds");

    // Round 1 has no elite to advance: it establishes the first one, which is
    // what `selected` reports.
    assert_eq!(logged[0].prior_best_score, None);
    assert!(!logged[0].is_improvement());
    assert!(logged[0].selected);

    // Every later round reports the elite it started from, so the predicate is
    // exactly that comparison.
    for pair in logged.windows(2) {
        let (previous, next) = (&pair[0], &pair[1]);
        assert_eq!(next.prior_best_score, Some(previous.best_score));
        assert_eq!(
            next.is_improvement(),
            next.reward > previous.best_score,
            "iteration {} disagrees with its own record",
            next.iteration
        );
    }

    // A mutation adds a marker, so round 2 beats round 1's elite ...
    assert!(logged[1].is_improvement(), "{}", logged[1].best_score);
    assert!(logged[1].selected);
    // ... and the marker count is capped, so round 3 only matches it.
    assert_eq!(logged[2].prior_best_score, Some(logged[1].best_score));
    assert_eq!(logged[2].reward, logged[2].best_score);
    assert!(!logged[2].is_improvement());
    assert!(!logged[2].selected);
}

#[test]
fn builders_and_readers_cover_the_pre_round_elite() {
    let record = RunLogRecord::new(2, PromptCandidate::new("bare prompt", "carried over"));
    assert_eq!(record.prior_best_score, None);
    assert!(!record.is_improvement());
    let record = record.with_prior_best_score(Some(0.25)).with_reward(0.5);
    assert_eq!(record.prior_best_score, Some(0.25));
    assert!(record.is_improvement());
}

#[test]
fn the_builders_and_readers_report_their_configuration() {
    let optimizer = PromptSearchOptimizer::new();
    assert!(!optimizer.has_run_log());
    assert!(!optimizer.has_adopt_policy());
    assert_eq!(optimizer.timestamp(), 0.0);

    let (_, handle) = shared_log();
    let optimizer = optimizer
        .with_run_log(handle)
        .with_adopt_policy(AdoptPolicy::new())
        .with_timestamp(NOW);
    assert!(optimizer.has_run_log());
    assert!(optimizer.has_adopt_policy());
    assert_eq!(optimizer.timestamp(), NOW);
    let debug = format!("{optimizer:?}");
    assert!(debug.contains("has_run_log: true"), "{debug}");
    assert!(debug.contains("has_adopt_policy: true"), "{debug}");
}

#[tokio::test]
async fn a_rejecting_sink_does_not_fail_the_run() {
    let handle: Arc<Mutex<dyn RunLog>> = Arc::new(Mutex::new(Rejecting));
    let result = PromptSearchOptimizer::new()
        .with_evaluator(Arc::new(Fixed { score: 0.8 }))
        .with_run_log(handle)
        .optimize(SEED, &cases())
        .await
        .unwrap();
    assert_eq!(result, reference().await);
}

#[tokio::test]
async fn a_poisoned_run_log_does_not_fail_the_run() {
    let (concrete, handle) = shared_log();
    // Poison the shared mutex: panic while its guard is held, catching the
    // unwind so the test can carry on with the poisoned log.
    let poison = Arc::clone(&concrete);
    let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        let _guard = poison.lock().expect("the log starts unpoisoned");
        panic!("poison the run-log mutex");
    }));
    assert!(unwind.is_err());
    assert!(concrete.lock().is_err());

    let result = PromptSearchOptimizer::new()
        .with_evaluator(Arc::new(Fixed { score: 0.8 }))
        .with_run_log(handle)
        .optimize(SEED, &cases())
        .await
        .unwrap();
    assert!(concrete.lock().is_err(), "the log stays poisoned");
    assert_eq!(result, reference().await);
}

#[tokio::test]
async fn a_logged_run_matches_a_logless_run() {
    let plain = reference().await;
    let (concrete, handle) = shared_log();
    let logged = PromptSearchOptimizer::new()
        .with_evaluator(Arc::new(Fixed { score: 0.8 }))
        .with_run_log(handle)
        .with_adopt_policy(AdoptPolicy::new())
        .with_timestamp(NOW)
        .optimize(SEED, &cases())
        .await
        .unwrap();
    assert_eq!(logged, plain);
    assert_eq!(records(&concrete).len(), plain.iterations);
}
