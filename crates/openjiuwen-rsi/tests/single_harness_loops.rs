//! Integration tests for the single-harness optimisation loops.
//!
//! A scripted [`Evaluator`] and the crate's `DeterministicMutator` drive both
//! policies offline: no model, no harness, no network, no clock. The scenarios
//! mirror the behaviours Python's `iterative.py` / `continuous.py` encode —
//! promotion on a real improvement, rejection on a regression, the stop reasons
//! (iteration cap, target score, cancellation, trials), resume without replay,
//! determinism, and protected-case retention.

// The emptiness assertions read better than `assert_eq!(value, empty_value)`.
#![allow(clippy::assert_is_empty)]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use openjiuwen_rsi::harness_rsi::single_harness::{
    batch_key, CandidateGateRecord, CaseScores, ContinuousSingleHarness, IterativeSingleHarness,
    JournalEntry, PromotionPolicyMode, SingleHarnessConfig, SingleHarnessStatus,
    REASON_PROTECTED_CASE_REGRESSED, REASON_SELECTED, REASON_TARGET_REPLAY_FAILED,
};
use openjiuwen_rsi::{
    Candidate, Case, CaseResult, DeterministicMutator, Error, ErrorKind, EvaluationReport,
    Evaluator, Result, StopReason,
};

/// Builds a score map from literal pairs.
fn scores(entries: &[(&str, f64)]) -> CaseScores {
    entries
        .iter()
        .map(|(case_id, score)| ((*case_id).to_owned(), *score))
        .collect()
}

/// Builds case ids from literals.
fn ids(entries: &[&str]) -> Vec<String> {
    entries.iter().map(|entry| (*entry).to_owned()).collect()
}

/// Builds case definitions from literals.
fn cases(entries: &[&str]) -> Vec<Case> {
    entries
        .iter()
        .map(|case_id| Case {
            id: (*case_id).to_owned(),
            ..Case::default()
        })
        .collect()
}

/// Mirrors the `DeterministicMutator` body derivation, so a test can predict
/// the candidate body of one generation.
fn variant(body: &str, generation: u32) -> String {
    format!("{body}\n# rsi variant g{generation}-0")
}

/// Compares floats with the promotion tolerance.
fn approx(actual: f64, expected: f64) -> bool {
    (actual - expected).abs() < 1e-9
}

/// An evaluator that returns scripted per-case scores keyed by candidate body.
///
/// The script is filtered to the cases the loop asks for, so the batch
/// evaluation and the full evaluation of the same body stay consistent; the
/// optional fallback serves any body that was not scripted. An unscripted body
/// without a fallback fails the evaluation, which is how the "engineering
/// failure" path is exercised.
struct ScriptedEvaluator {
    /// Per-body scores.
    script: Mutex<BTreeMap<String, CaseScores>>,
    /// Scores used for bodies without a script.
    fallback: Option<CaseScores>,
    /// Number of `evaluate` calls.
    calls: AtomicUsize,
}

impl ScriptedEvaluator {
    /// Builds an evaluator with no script.
    fn new() -> Self {
        Self {
            script: Mutex::new(BTreeMap::new()),
            fallback: None,
            calls: AtomicUsize::new(0),
        }
    }

    /// Adds the scripted score map of `body`.
    fn with_script(mut self, body: &str, scores: CaseScores) -> Self {
        self.script
            .get_mut()
            .expect("script lock")
            .insert(body.to_owned(), scores);
        self
    }

    /// Sets the fallback score map of unscripted bodies.
    fn with_fallback(mut self, scores: CaseScores) -> Self {
        self.fallback = Some(scores);
        self
    }

    /// Returns how many times the evaluator ran.
    fn calls(&self) -> usize {
        self.calls.load(Ordering::Relaxed)
    }
}

#[async_trait]
impl Evaluator for ScriptedEvaluator {
    fn name(&self) -> &'static str {
        "scripted"
    }

    async fn evaluate(&self, candidate: &Candidate, cases: &[Case]) -> Result<EvaluationReport> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        let selected = {
            let script = self.script.lock().expect("script lock");
            script
                .get(candidate.body.as_str())
                .cloned()
                .or_else(|| self.fallback.clone())
        };
        let selected = selected.ok_or_else(|| {
            Error::validation(format!("unscripted candidate body `{}`", candidate.body))
        })?;
        let results = cases
            .iter()
            .map(|case| CaseResult {
                case_id: case.id.clone(),
                passed: true,
                score: selected.get(&case.id).copied(),
                weight: case.weight,
                error: None,
                details: serde_json::json!(null),
            })
            .collect();
        Ok(EvaluationReport::from_results(results))
    }
}

/// Iterative test configuration with a deterministic single-case batch.
fn iterative_config() -> SingleHarnessConfig {
    SingleHarnessConfig {
        max_epochs: 1,
        max_iterations: 1,
        batch_size: 1,
        patience: 0,
        ..SingleHarnessConfig::default()
    }
}

/// Continuous test configuration with two attempts per batch.
fn continuous_config() -> SingleHarnessConfig {
    SingleHarnessConfig {
        max_epochs: 1,
        max_iterations: 2,
        batch_size: 1,
        patience: 0,
        max_attempts_per_batch: 2,
        max_trials: 8,
        ..SingleHarnessConfig::default()
    }
}

/// Builds an iterative harness over the given evaluator.
///
/// The case ids are taken by value so a call site reads as one expression.
#[allow(clippy::needless_pass_by_value)]
fn iterative(
    config: SingleHarnessConfig,
    evaluator: Arc<ScriptedEvaluator>,
    dataset: Vec<String>,
    case_ids: Vec<String>,
    seed: &str,
) -> Result<IterativeSingleHarness> {
    let case_ids: Vec<&str> = case_ids.iter().map(String::as_str).collect();
    IterativeSingleHarness::new(
        config,
        evaluator,
        Arc::new(DeterministicMutator::new()),
        "task",
        dataset,
        cases(&case_ids),
        seed,
    )
}

/// Unwraps the error of a result whose success type is not `Debug`.
fn expect_error<T>(result: Result<T>, what: &str) -> Error {
    result
        .err()
        .unwrap_or_else(|| panic!("expected `{what}` to be rejected"))
}

#[tokio::test]
async fn iterative_promotes_an_improving_candidate() {
    let seed = "model: seed";
    let evaluator = Arc::new(
        ScriptedEvaluator::new()
            .with_script(seed, scores(&[("a", 0.5), ("b", 0.5)]))
            .with_fallback(scores(&[("a", 0.7), ("b", 0.5)])),
    );
    let mut harness = iterative(
        iterative_config(),
        evaluator.clone(),
        ids(&["a", "b"]),
        ids(&["a", "b"]),
        seed,
    )
    .expect("harness");

    let result = harness.run().await.expect("run");
    assert_eq!(result.promotions, 1);
    assert_eq!(result.evaluations, 3, "baseline, batch, full evaluation");
    assert_eq!(result.state.optimization_journal.len(), 1);
    let entry = &result.state.optimization_journal[0];
    assert!(entry.promoted);
    assert_eq!(entry.epoch, 1);
    assert_eq!(entry.reason, REASON_SELECTED);
    assert_eq!(result.best_body, variant(seed, 1));
    assert_eq!(result.state.best_body, variant(seed, 1));
    assert!(approx(result.best_score.expect("score"), 0.6));
    assert!(approx(result.state.baseline_score.expect("baseline"), 0.5));
    assert_eq!(result.state.completed_batches, vec![batch_key(1, 0)]);
    assert!(result.state.candidate_gates[0].passed);
    assert_eq!(result.state.best_harness_refs_path, None);
    assert_eq!(result.state.stop_reason, Some(StopReason::MaxIterations));
    assert_eq!(evaluator.calls(), result.evaluations);
}

#[tokio::test]
async fn iterative_rejects_a_regressing_candidate() {
    let seed = "model: seed";
    let evaluator = Arc::new(
        ScriptedEvaluator::new()
            .with_script(seed, scores(&[("a", 0.5), ("b", 0.5)]))
            .with_fallback(scores(&[("a", 0.3), ("b", 0.5)])),
    );
    let mut harness = iterative(
        iterative_config(),
        evaluator.clone(),
        ids(&["a", "b"]),
        ids(&["a", "b"]),
        seed,
    )
    .expect("harness");

    let result = harness.run().await.expect("run");
    assert_eq!(result.promotions, 0);
    assert_eq!(result.best_body, seed);
    assert!(approx(result.best_score.expect("score"), 0.5));
    let entry = &result.state.optimization_journal[0];
    assert!(!entry.promoted);
    assert_eq!(entry.reason, REASON_TARGET_REPLAY_FAILED);
    assert!(!result.state.candidate_gates[0].passed);
    assert!(result.state.completed_batches.is_empty());
    assert_eq!(result.evaluations, 2, "baseline and the batch only");
}

#[tokio::test]
async fn iterative_stops_at_max_iterations() {
    let seed = "model: seed";
    let evaluator = Arc::new(
        ScriptedEvaluator::new()
            .with_script(seed, scores(&[("a", 0.5), ("b", 0.5)]))
            .with_fallback(scores(&[("a", 0.5), ("b", 0.5)])),
    );
    let config = SingleHarnessConfig {
        max_epochs: 5,
        max_iterations: 2,
        ..iterative_config()
    };
    let mut harness =
        iterative(config, evaluator, ids(&["a", "b"]), ids(&["a", "b"]), seed).expect("harness");

    let result = harness.run().await.expect("run");
    assert_eq!(result.state.optimization_journal.len(), 2);
    assert_eq!(result.state.optimization_journal[0].epoch, 1);
    assert_eq!(result.state.optimization_journal[1].epoch, 2);
    assert_eq!(result.state.stop_reason, Some(StopReason::MaxIterations));
    assert_eq!(result.state.status, SingleHarnessStatus::Completed);
    assert_eq!(result.promotions, 0);
}

#[tokio::test]
async fn iterative_stops_when_the_target_score_is_reached() {
    let seed = "model: seed";
    let evaluator = Arc::new(
        ScriptedEvaluator::new()
            .with_script(seed, scores(&[("a", 0.5), ("b", 0.5)]))
            .with_fallback(scores(&[("a", 0.6), ("b", 0.6)])),
    );
    let config = SingleHarnessConfig {
        max_epochs: 5,
        max_iterations: 5,
        batch_size: 2,
        patience: 0,
        target_score: Some(0.55),
        ..SingleHarnessConfig::default()
    };
    let mut harness =
        iterative(config, evaluator, ids(&["a", "b"]), ids(&["a", "b"]), seed).expect("harness");

    let result = harness.run().await.expect("run");
    assert_eq!(result.promotions, 1);
    assert_eq!(result.state.optimization_journal.len(), 1);
    assert!(approx(result.best_score.expect("score"), 0.6));
    assert_eq!(result.state.stop_reason, Some(StopReason::TargetScore));
}

#[tokio::test]
async fn iterative_stop_cancels_the_run() {
    let seed = "model: seed";
    let evaluator = Arc::new(ScriptedEvaluator::new().with_fallback(scores(&[("a", 0.5)])));
    let mut harness = iterative(
        SingleHarnessConfig {
            max_epochs: 3,
            ..iterative_config()
        },
        evaluator.clone(),
        ids(&["a"]),
        ids(&["a"]),
        seed,
    )
    .expect("harness");

    harness.stop();
    assert!(harness.is_cancelled());
    let result = harness.run().await.expect("run");
    assert!(result.state.optimization_journal.is_empty());
    assert_eq!(result.state.stop_reason, Some(StopReason::Cancelled));
    assert_eq!(result.state.status, SingleHarnessStatus::Cancelled);
    assert_eq!(evaluator.calls(), 0, "a cancelled run never evaluates");
}

#[tokio::test]
async fn iterative_records_an_engineering_failure_without_aborting() {
    let seed = "model: seed";
    // No fallback: the improver's candidate body is unscripted, so every
    // candidate evaluation errors.
    let evaluator = Arc::new(ScriptedEvaluator::new().with_script(seed, scores(&[("a", 0.5)])));
    let config = SingleHarnessConfig {
        max_epochs: 3,
        max_iterations: 2,
        ..iterative_config()
    };
    let mut harness =
        iterative(config, evaluator.clone(), ids(&["a"]), ids(&["a"]), seed).expect("harness");

    let result = harness.run().await.expect("run");
    assert_eq!(result.state.optimization_journal.len(), 2);
    for entry in &result.state.optimization_journal {
        assert!(!entry.promoted);
        assert!(
            entry.reason.starts_with("engineering_failed"),
            "unexpected reason `{}`",
            entry.reason
        );
    }
    assert_eq!(result.state.stop_reason, Some(StopReason::MaxIterations));
    assert_eq!(result.best_body, seed);
}

#[tokio::test]
async fn iterative_resume_continues_without_replaying_a_completed_batch() {
    let seed = "model: seed";
    let first = Arc::new(
        ScriptedEvaluator::new()
            .with_script(seed, scores(&[("a", 0.5), ("b", 0.5)]))
            .with_fallback(scores(&[("a", 0.7), ("b", 0.5)])),
    );
    let mut harness = iterative(
        SingleHarnessConfig {
            max_epochs: 1,
            max_iterations: 3,
            ..iterative_config()
        },
        Arc::clone(&first),
        ids(&["a", "b"]),
        ids(&["a", "b"]),
        seed,
    )
    .expect("harness");
    let result = harness.run().await.expect("run");
    assert_eq!(result.state.completed_batches, vec![batch_key(1, 0)]);
    assert_eq!(result.state.optimization_journal.len(), 1);
    let promoted_body = result.best_body.clone();

    // The resumed run has no baseline script: the state already carries the
    // baseline and the incumbent case scores.
    let second =
        Arc::new(ScriptedEvaluator::new().with_fallback(scores(&[("a", 0.7), ("b", 0.7)])));
    let mut resumed = iterative(
        SingleHarnessConfig {
            max_epochs: 2,
            max_iterations: 3,
            ..iterative_config()
        },
        Arc::clone(&second),
        ids(&["a", "b"]),
        ids(&["a", "b"]),
        seed,
    )
    .expect("harness")
    .with_state(result.state.clone())
    .expect("state resumes");

    let resumed_result = resumed.run().await.expect("resumed run");
    assert_eq!(
        resumed_result.state.optimization_journal.len(),
        2,
        "epoch 1 was skipped, epoch 2 appended one entry"
    );
    assert_eq!(
        resumed_result.state.optimization_journal[0].epoch, 1,
        "the completed batch keeps its original entry"
    );
    assert_eq!(second.calls(), 2, "one batch and one full evaluation");
    assert_eq!(resumed_result.best_body, variant(&promoted_body, 2));
}

#[tokio::test]
async fn iterative_runs_are_deterministic() {
    async fn run_once() -> (
        SingleHarnessConfig,
        Vec<JournalEntry>,
        Vec<CandidateGateRecord>,
        String,
    ) {
        let seed = "model: seed";
        let evaluator = Arc::new(
            ScriptedEvaluator::new()
                .with_script(seed, scores(&[("a", 0.5), ("b", 0.5)]))
                .with_fallback(scores(&[("a", 0.7), ("b", 0.7)])),
        );
        let config = SingleHarnessConfig {
            max_epochs: 3,
            max_iterations: 3,
            ..iterative_config()
        };
        let mut harness = iterative(
            config.clone(),
            evaluator,
            ids(&["a", "b"]),
            ids(&["a", "b"]),
            seed,
        )
        .expect("harness");
        let result = harness.run().await.expect("run");
        (
            config,
            result.state.optimization_journal,
            result.state.candidate_gates,
            result.best_body,
        )
    }

    let (config, first_journal, first_gates, first_body) = run_once().await;
    let (_, second_journal, second_gates, second_body) = run_once().await;
    assert_eq!(first_journal, second_journal);
    assert_eq!(first_gates, second_gates);
    assert_eq!(first_body, second_body);
    assert_eq!(first_journal.len(), 3);
    assert!(config.validate().is_ok());
}

#[tokio::test]
async fn iterative_legacy_mode_promotes_a_candidate_that_beats_the_incumbent() {
    let seed = "model: seed";
    let scripted = || {
        Arc::new(
            ScriptedEvaluator::new()
                .with_script(seed, scores(&[("a", 0.5), ("p", 0.9)]))
                .with_fallback(scores(&[("a", 0.9), ("p", 0.6)])),
        )
    };
    let base = SingleHarnessConfig {
        protected_cases: ids(&["p"]),
        ..iterative_config()
    };

    // Continuous policy: the protected case regressed, so nothing is promoted.
    let mut strict = iterative(
        base.clone(),
        scripted(),
        ids(&["a"]),
        ids(&["a", "p"]),
        seed,
    )
    .expect("harness");
    let strict_result = strict.run().await.expect("run");
    assert_eq!(strict_result.promotions, 0);
    assert_eq!(
        strict_result.state.optimization_journal[0].reason,
        REASON_PROTECTED_CASE_REGRESSED
    );

    // Legacy policy: the same candidate beats the incumbent's overall score
    // (0.75 against 0.70), so the documented shortcut promotes it.
    let mut legacy = iterative(
        SingleHarnessConfig {
            mode: PromotionPolicyMode::LegacyEpochV1,
            ..base
        },
        scripted(),
        ids(&["a"]),
        ids(&["a", "p"]),
        seed,
    )
    .expect("harness");
    let legacy_result = legacy.run().await.expect("run");
    assert_eq!(legacy_result.promotions, 1);
    assert_eq!(
        legacy_result.state.optimization_journal[0].reason,
        REASON_SELECTED
    );
    assert_eq!(legacy_result.best_body, variant(seed, 1));
}

#[tokio::test]
async fn continuous_promotes_one_candidate_per_batch() {
    let seed = "model: seed";
    let first = variant(seed, 1);
    let second = variant(&first, 2);
    let evaluator = Arc::new(
        ScriptedEvaluator::new()
            .with_script(seed, scores(&[("a", 0.5), ("b", 0.5)]))
            .with_script(&first, scores(&[("a", 0.7), ("b", 0.5)]))
            .with_script(&second, scores(&[("a", 0.7), ("b", 0.7)])),
    );
    let config = SingleHarnessConfig {
        max_epochs: 2,
        ..continuous_config()
    };
    let mut harness = ContinuousSingleHarness::new(
        config,
        evaluator.clone(),
        Arc::new(DeterministicMutator::new()),
        "task",
        ids(&["a", "b"]),
        cases(&["a", "b"]),
        seed,
    )
    .expect("harness");

    let result = harness.run().await.expect("run");
    assert_eq!(result.promotions, 2);
    assert_eq!(result.state.optimization_journal.len(), 2);
    assert!(result
        .state
        .optimization_journal
        .iter()
        .all(|entry| entry.promoted));
    assert_eq!(result.state.continuous_trials, 2);
    assert_eq!(result.state.continuous_attempts, 0);
    assert_eq!(result.best_body, second);
    assert!(approx(result.best_score.expect("score"), 0.7));
    assert_eq!(result.state.stop_reason, Some(StopReason::MaxIterations));
}

#[tokio::test]
async fn continuous_respects_the_trial_cap() {
    let seed = "model: seed";
    let evaluator = Arc::new(
        ScriptedEvaluator::new()
            .with_script(seed, scores(&[("a", 0.5), ("b", 0.5)]))
            .with_fallback(scores(&[("a", 0.5), ("b", 0.5)])),
    );
    let config = SingleHarnessConfig {
        max_epochs: 3,
        max_iterations: 5,
        max_trials: 1,
        ..continuous_config()
    };
    let mut harness = ContinuousSingleHarness::new(
        config,
        evaluator.clone(),
        Arc::new(DeterministicMutator::new()),
        "task",
        ids(&["a", "b"]),
        cases(&["a", "b"]),
        seed,
    )
    .expect("harness");

    let result = harness.run().await.expect("run");
    assert_eq!(result.state.continuous_trials, 1);
    assert_eq!(result.promotions, 0);
    assert_eq!(result.state.optimization_journal.len(), 1);
    assert!(!result.state.optimization_journal[0].promoted);
    assert_eq!(result.state.stop_reason, Some(StopReason::MaxIterations));
    assert_eq!(result.evaluations, 2, "baseline and the single attempt");
    assert_eq!(evaluator.calls(), 2);
}

#[tokio::test]
async fn continuous_blocks_a_protected_case_regression() {
    let seed = "model: seed";
    let evaluator = Arc::new(
        ScriptedEvaluator::new()
            .with_script(seed, scores(&[("a", 0.5), ("p", 0.9)]))
            .with_fallback(scores(&[("a", 0.8), ("p", 0.7)])),
    );
    let config = SingleHarnessConfig {
        protected_cases: ids(&["p"]),
        max_attempts_per_batch: 1,
        ..continuous_config()
    };
    let mut harness = ContinuousSingleHarness::new(
        config,
        evaluator,
        Arc::new(DeterministicMutator::new()),
        "task",
        ids(&["a"]),
        cases(&["a", "p"]),
        seed,
    )
    .expect("harness");

    let result = harness.run().await.expect("run");
    assert_eq!(result.promotions, 0);
    assert_eq!(result.best_body, seed);
    assert_eq!(
        result.state.optimization_journal[0].reason,
        REASON_PROTECTED_CASE_REGRESSED
    );
    assert!(
        result.state.candidate_gates[0].passed,
        "the batch gate passes; only the full evaluation rejects"
    );
    assert!(approx(result.best_score.expect("score"), 0.7));
}

#[tokio::test]
async fn continuous_retries_a_batch_up_to_the_attempt_cap() {
    let seed = "model: seed";
    let candidate_one = variant(seed, 1);
    let candidate_two = variant(seed, 2);
    let evaluator = Arc::new(
        ScriptedEvaluator::new()
            .with_script(seed, scores(&[("a", 0.5)]))
            .with_script(&candidate_one, scores(&[("a", 0.5)]))
            .with_script(&candidate_two, scores(&[("a", 0.8)])),
    );
    let config = SingleHarnessConfig {
        max_attempts_per_batch: 2,
        max_epochs: 1,
        max_trials: 4,
        ..continuous_config()
    };
    let mut harness = ContinuousSingleHarness::new(
        config,
        evaluator.clone(),
        Arc::new(DeterministicMutator::new()),
        "task",
        ids(&["a"]),
        cases(&["a"]),
        seed,
    )
    .expect("harness");

    let result = harness.run().await.expect("run");
    assert_eq!(result.state.continuous_trials, 2);
    assert_eq!(result.promotions, 1);
    assert_eq!(result.state.optimization_journal.len(), 1);
    assert!(result.state.optimization_journal[0].promoted);
    assert_eq!(result.state.candidate_gates.len(), 2);
    assert!(!result.state.candidate_gates[0].passed);
    assert!(result.state.candidate_gates[1].passed);
    assert_eq!(result.best_body, candidate_two);
}

#[tokio::test]
async fn constructors_reject_unusable_inputs() {
    let seed = "model: seed";
    let evaluator = Arc::new(ScriptedEvaluator::new().with_fallback(scores(&[("a", 0.5)])));

    let unknown = expect_error(
        iterative(
            iterative_config(),
            evaluator.clone(),
            ids(&["missing"]),
            ids(&["a"]),
            seed,
        ),
        "unknown dataset case",
    );
    assert_eq!(unknown.kind(), ErrorKind::Validation);

    let empty = expect_error(
        IterativeSingleHarness::new(
            iterative_config(),
            evaluator.clone(),
            Arc::new(DeterministicMutator::new()),
            "task",
            Vec::new(),
            Vec::new(),
            seed,
        ),
        "empty case set",
    );
    assert_eq!(empty.kind(), ErrorKind::Validation);

    let invalid = expect_error(
        IterativeSingleHarness::new(
            SingleHarnessConfig {
                batch_size: 0,
                ..iterative_config()
            },
            evaluator,
            Arc::new(DeterministicMutator::new()),
            "task",
            ids(&["a"]),
            cases(&["a"]),
            seed,
        ),
        "invalid config",
    );
    assert_eq!(invalid.kind(), ErrorKind::Config);
}
