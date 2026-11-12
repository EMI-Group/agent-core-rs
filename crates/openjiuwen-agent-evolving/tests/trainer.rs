#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! Integration tests for `openjiuwen_agent_evolving::trainer`.
//!
//! Covers baseline validation plus epochs, checkpoint saves on improvement, resume
//! from a stored checkpoint, cancellation, evaluation/wall-clock budgets, candidate
//! exhaustion, early stopping, callback ordering, and the serde/default contract.
//!
//! Everything here is offline and deterministic: the fakes describe validation
//! scores up front, the checkpoint store is in-process, no environment is read, and
//! the only clock use is inside the explicitly requested `time_budget_ms` test,
//! which asserts the stop reason alone.
use std::collections::{BTreeMap, VecDeque};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use openjiuwen_agent_evolving::dataset::{Case, EvaluatedCase};
use openjiuwen_agent_evolving::evaluator::{EvaluationReport, Evaluator};
use openjiuwen_agent_evolving::optimizer::{
    Candidate, CheckpointStore, DefaultCheckpointManager, MemoryCheckpointStore, Optimizer,
};
use openjiuwen_agent_evolving::trainer::{
    Callbacks, EpochReport, NoopCallbacks, Progress, StopReason, TrainableTarget, Trainer,
    TrainerConfig, TrainerReport,
};
use openjiuwen_common::error::{Error, Result};
use serde_json::{json, Value};
use tokio_util::sync::CancellationToken;

/// A valid case with fixed inputs/label and an explicit identifier.
fn case(id: &str) -> Case {
    Case::new(
        BTreeMap::from([("question".to_owned(), json!(id))]),
        BTreeMap::from([("answer".to_owned(), json!(id))]),
    )
    .expect("valid case")
    .with_case_id(id)
}

/// An unscored candidate with no parameters; the fake target keys scores by id.
fn candidate(id: &str) -> Candidate {
    Candidate::new(id, BTreeMap::new())
}

/// A config with `num_iterations` epochs and every other field at its default.
fn config(num_iterations: usize) -> TrainerConfig {
    TrainerConfig {
        num_iterations,
        ..TrainerConfig::default()
    }
}

fn close(actual: f64, expected: f64) -> bool {
    (actual - expected).abs() < 1e-9
}

/// Runs one training pass with a fresh cancellation token.
async fn run(
    trainer: &Trainer,
    target: &FakeTarget,
    optimizer: &mut ScriptedOptimizer,
    evaluator: &MapEvaluator,
    train_cases: &[Case],
    val_cases: &[Case],
) -> TrainerReport {
    trainer
        .train(
            target,
            optimizer,
            evaluator,
            train_cases,
            val_cases,
            &CancellationToken::new(),
        )
        .await
        .expect("train")
}

/// State shared between [`FakeTarget`] and [`MapEvaluator`].
#[derive(Debug, Default)]
struct TargetState {
    /// Candidate ids in application order; the last one drives scoring.
    applied: Vec<String>,
    /// Validation score per candidate id.
    candidate_scores: BTreeMap<String, f64>,
    /// Last payload handed to `restore`.
    restored: Option<Value>,
    /// Value returned from `snapshot`.
    snapshot_value: Value,
}

/// A target whose validation score is the applied candidate's table entry.
struct FakeTarget {
    state: Arc<Mutex<TargetState>>,
}

impl FakeTarget {
    /// Builds a target scoring the given `candidate id -> validation score` pairs.
    fn new(candidate_scores: &[(&str, f64)]) -> Self {
        let candidate_scores = candidate_scores
            .iter()
            .map(|(id, score)| ((*id).to_owned(), *score))
            .collect();
        Self {
            state: Arc::new(Mutex::new(TargetState {
                applied: Vec::new(),
                candidate_scores,
                restored: None,
                snapshot_value: json!({ "temperature": 0.5 }),
            })),
        }
    }

    fn state(&self) -> Arc<Mutex<TargetState>> {
        Arc::clone(&self.state)
    }

    fn applied(&self) -> Vec<String> {
        self.state.lock().expect("state").applied.clone()
    }
}

#[async_trait]
impl TrainableTarget for FakeTarget {
    fn id(&self) -> &'static str {
        "target"
    }

    async fn run_case(&self, case: &Case, cancel: &CancellationToken) -> Result<Value> {
        if cancel.is_cancelled() {
            return Err(Error::cancelled());
        }
        Ok(json!({ "output": case.case_id }))
    }

    async fn apply(&self, candidate: &Candidate) -> Result<()> {
        let mut state = self.state.lock().expect("state");
        state.applied.push(candidate.id.clone());
        Ok(())
    }

    async fn snapshot(&self) -> Result<Value> {
        Ok(self.state.lock().expect("state").snapshot_value.clone())
    }

    async fn restore(&self, state: Value) -> Result<()> {
        self.state.lock().expect("state").restored = Some(state);
        Ok(())
    }
}

/// Scores every case with the currently applied candidate's score, or `fallback`
/// when no candidate has been applied yet (the baseline pass).
struct MapEvaluator {
    state: Arc<Mutex<TargetState>>,
    fallback: f64,
}

impl MapEvaluator {
    fn new(target: &FakeTarget, fallback: f64) -> Self {
        Self {
            state: target.state(),
            fallback,
        }
    }

    fn score(&self) -> f64 {
        let state = self.state.lock().expect("state");
        state
            .applied
            .last()
            .and_then(|id| state.candidate_scores.get(id).copied())
            .unwrap_or(self.fallback)
    }
}

#[async_trait]
impl Evaluator for MapEvaluator {
    fn name(&self) -> &'static str {
        "map"
    }

    async fn evaluate(&self, case: &Case, _prediction: &Value) -> Result<EvaluatedCase> {
        EvaluatedCase::new(case.clone(), None, self.score(), "fake")
    }
}

/// Proposes pre-scripted candidate chunks and tracks the best observed candidate.
struct ScriptedOptimizer {
    chunks: VecDeque<Vec<Candidate>>,
    observed: Vec<(Candidate, f64)>,
}

impl ScriptedOptimizer {
    fn new(chunks: Vec<Vec<Candidate>>) -> Self {
        Self {
            chunks: chunks.into(),
            observed: Vec::new(),
        }
    }

    fn observed_ids(&self) -> Vec<&str> {
        self.observed
            .iter()
            .map(|(candidate, _)| candidate.id.as_str())
            .collect()
    }
}

#[async_trait]
impl Optimizer for ScriptedOptimizer {
    fn name(&self) -> &'static str {
        "scripted"
    }

    async fn propose(&mut self, limit: usize) -> Result<Vec<Candidate>> {
        let mut chunk = self.chunks.pop_front().unwrap_or_default();
        chunk.truncate(limit);
        Ok(chunk)
    }

    fn observe(&mut self, candidate: &Candidate, score: f64) {
        let mut observed = candidate.clone();
        observed.score = Some(score);
        self.observed.push((observed, score));
    }

    fn best(&self) -> Option<&Candidate> {
        self.observed
            .iter()
            .max_by(|(_, left), (_, right)| left.total_cmp(right))
            .map(|(candidate, _)| candidate)
    }
}

/// Records callback order and whether each call saw a baseline report.
#[derive(Clone)]
struct RecordingCallbacks {
    events: Arc<Mutex<Vec<String>>>,
    baselines: Arc<Mutex<Vec<bool>>>,
}

impl RecordingCallbacks {
    fn new() -> Self {
        Self {
            events: Arc::new(Mutex::new(Vec::new())),
            baselines: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn events(&self) -> Vec<String> {
        self.events.lock().expect("events").clone()
    }

    fn baselines(&self) -> Vec<bool> {
        self.baselines.lock().expect("baselines").clone()
    }
}

impl Callbacks for RecordingCallbacks {
    fn on_train_begin(&self, _progress: &Progress, baseline: Option<&EvaluationReport>) {
        self.events.lock().expect("events").push("begin".to_owned());
        let seen = baseline.is_some();
        self.baselines.lock().expect("baselines").push(seen);
    }

    fn on_train_epoch_begin(&self, _progress: &Progress) {
        let mut events = self.events.lock().expect("events");
        events.push("epoch_begin".to_owned());
    }

    fn on_train_epoch_end(&self, _progress: &Progress, _report: Option<&EvaluationReport>) {
        let mut events = self.events.lock().expect("events");
        events.push("epoch_end".to_owned());
    }

    fn on_train_end(&self, _progress: &Progress, _report: Option<&EvaluationReport>) {
        let mut events = self.events.lock().expect("events");
        events.push("end".to_owned());
    }
}

#[tokio::test]
async fn baseline_then_three_epochs_completes() {
    let target = FakeTarget::new(&[("cand-1", 0.4), ("cand-2", 0.4), ("cand-3", 0.4)]);
    let evaluator = MapEvaluator::new(&target, 0.4);
    let mut optimizer = ScriptedOptimizer::new(vec![
        vec![candidate("cand-1")],
        vec![candidate("cand-2")],
        vec![candidate("cand-3")],
    ]);
    let trainer = Trainer::new(config(3));

    let report = run(
        &trainer,
        &target,
        &mut optimizer,
        &evaluator,
        &[],
        &[case("val-1")],
    )
    .await;

    assert_eq!(report.stopped_reason, StopReason::Completed);
    assert!(!report.cancelled);
    assert_eq!(report.epochs_run, 3);
    assert_eq!(report.steps.len(), 3);
    let epochs: Vec<usize> = report.steps.iter().map(|step| step.epoch).collect();
    assert_eq!(epochs, vec![0, 1, 2]);
    assert!(close(report.best_score, 0.4));
    let best = report
        .best_candidate
        .as_ref()
        .map(|candidate| candidate.id.as_str());
    assert_eq!(best, Some("cand-1"));
    // One baseline evaluation plus one validation case per candidate; no train cases.
    assert_eq!(report.evaluations, 4);
    assert_eq!(optimizer.observed_ids(), vec!["cand-1", "cand-2", "cand-3"]);
    assert_eq!(target.applied(), vec!["cand-1", "cand-2", "cand-3"]);
}

#[tokio::test]
async fn improvement_triggers_a_checkpoint_save() {
    let store = Arc::new(MemoryCheckpointStore::new());
    let manager = Arc::new(DefaultCheckpointManager::new().with_run_id("run-improve"));
    let target = FakeTarget::new(&[("cand-a", 0.2), ("cand-b", 0.8)]);
    let evaluator = MapEvaluator::new(&target, 0.0);
    let mut optimizer =
        ScriptedOptimizer::new(vec![vec![candidate("cand-a")], vec![candidate("cand-b")]]);
    let trainer = Trainer::new(TrainerConfig {
        num_iterations: 2,
        checkpoint_filename: "run.json".to_owned(),
        ..TrainerConfig::default()
    })
    .with_checkpoint(store.clone(), manager);

    let report = run(
        &trainer,
        &target,
        &mut optimizer,
        &evaluator,
        &[],
        &[case("val-1")],
    )
    .await;

    assert!(report.steps[0].checkpoint_saved);
    assert!(report.steps[1].improved);
    assert!(report.steps[1].checkpoint_saved);
    let checkpoint = store.load_checkpoint("run.json").await.expect("load");
    let checkpoint = checkpoint.expect("checkpoint present");
    assert!(close(checkpoint.best.best_score, 0.8));
    assert_eq!(checkpoint.step.epoch, 1);
    assert_eq!(checkpoint.run_id, "run-improve");
}

#[tokio::test]
async fn resume_restores_best_score_and_reruns_stored_epoch() {
    let store = Arc::new(MemoryCheckpointStore::new());
    let manager = Arc::new(DefaultCheckpointManager::new().with_run_id("run-resume"));
    let save_config = TrainerConfig {
        num_iterations: 1,
        checkpoint_filename: "run.json".to_owned(),
        ..TrainerConfig::default()
    };

    let first_target = FakeTarget::new(&[("cand-a", 0.9)]);
    let first_evaluator = MapEvaluator::new(&first_target, 0.0);
    let mut first_optimizer = ScriptedOptimizer::new(vec![vec![candidate("cand-a")]]);
    let first_trainer =
        Trainer::new(save_config.clone()).with_checkpoint(store.clone(), manager.clone());
    let first = run(
        &first_trainer,
        &first_target,
        &mut first_optimizer,
        &first_evaluator,
        &[],
        &[case("val-1")],
    )
    .await;
    assert!(first.steps[0].checkpoint_saved);
    assert!(close(first.best_score, 0.9));

    // A worse candidate must neither regress the restored best nor be adopted, and
    // the stored epoch (0) is re-run before the second epoch finds no candidates.
    let target = FakeTarget::new(&[("cand-worse", 0.3)]);
    let evaluator = MapEvaluator::new(&target, 0.0);
    let mut optimizer = ScriptedOptimizer::new(vec![vec![candidate("cand-worse")]]);
    let trainer = Trainer::new(TrainerConfig {
        num_iterations: 2,
        ..save_config
    })
    .with_checkpoint(store, manager)
    .with_resume_from(Some("run.json".to_owned()));

    let report = run(
        &trainer,
        &target,
        &mut optimizer,
        &evaluator,
        &[],
        &[case("val-1")],
    )
    .await;

    assert_eq!(report.steps[0].epoch, 0);
    assert!(close(report.best_score, 0.9));
    assert_eq!(report.best_candidate, None);
    assert!(!report.steps[0].improved);
    let restored = target.state().lock().expect("state").restored.clone();
    let restored = restored.expect("restored state");
    assert_eq!(restored["target"]["temperature"], json!(0.5));
}

#[tokio::test]
async fn cancellation_stops_before_any_epoch() {
    let target = FakeTarget::new(&[]);
    let evaluator = MapEvaluator::new(&target, 1.0);
    let mut optimizer = ScriptedOptimizer::new(vec![vec![candidate("cand-1")]]);
    let callbacks = RecordingCallbacks::new();
    let trainer = Trainer::new(config(2))
        .with_callbacks(vec![Arc::new(callbacks.clone()) as Arc<dyn Callbacks>]);
    let cancel = CancellationToken::new();
    cancel.cancel();

    let report = trainer
        .train(&target, &mut optimizer, &evaluator, &[], &[], &cancel)
        .await
        .expect("train");

    assert_eq!(report.stopped_reason, StopReason::Cancelled);
    assert!(report.cancelled);
    assert_eq!(report.epochs_run, 0);
    assert_eq!(report.steps.len(), 0);
    assert_eq!(optimizer.observed_ids(), Vec::<&str>::new());
    let events = callbacks.events();
    assert_eq!(events.last().map(String::as_str), Some("end"));
    assert_eq!(events, vec!["begin", "end"]);
    assert_eq!(callbacks.baselines(), vec![false]);
}

#[tokio::test]
async fn evaluation_budget_exhausts_before_the_forward_pass() {
    let target = FakeTarget::new(&[("cand-1", 0.5)]);
    let evaluator = MapEvaluator::new(&target, 0.4);
    let mut optimizer = ScriptedOptimizer::new(vec![vec![candidate("cand-1")]]);
    let trainer = Trainer::new(TrainerConfig {
        max_case_evaluations: Some(1),
        ..config(2)
    });

    let report = run(
        &trainer,
        &target,
        &mut optimizer,
        &evaluator,
        &[case("train-1"), case("train-2")],
        &[case("val-1")],
    )
    .await;

    assert_eq!(report.stopped_reason, StopReason::BudgetExhausted);
    assert_eq!(report.epochs_run, 0);
    // The baseline consumed the whole budget; no candidate was ever applied.
    assert_eq!(report.evaluations, 1);
    assert_eq!(target.applied(), Vec::<String>::new());
}

#[tokio::test]
async fn wall_clock_budget_stops_the_loop() {
    let target = FakeTarget::new(&[("cand-1", 0.5)]);
    let evaluator = MapEvaluator::new(&target, 0.4);
    let mut optimizer = ScriptedOptimizer::new(vec![vec![candidate("cand-1")]]);
    let trainer = Trainer::new(TrainerConfig {
        time_budget_ms: Some(0),
        ..config(2)
    });

    let report = run(
        &trainer,
        &target,
        &mut optimizer,
        &evaluator,
        &[],
        &[case("val-1")],
    )
    .await;

    // A zero budget is always exhausted, so only the stop reason is deterministic.
    assert_eq!(report.stopped_reason, StopReason::BudgetExhausted);
}

#[tokio::test]
async fn empty_proposal_stops_with_no_candidates() {
    let target = FakeTarget::new(&[]);
    let evaluator = MapEvaluator::new(&target, 0.0);
    let mut optimizer = ScriptedOptimizer::new(vec![Vec::new()]);
    let trainer = Trainer::new(config(3));

    let report = run(
        &trainer,
        &target,
        &mut optimizer,
        &evaluator,
        &[],
        &[case("val-1")],
    )
    .await;

    assert_eq!(report.stopped_reason, StopReason::NoCandidates);
    assert_eq!(report.epochs_run, 0);
    assert_eq!(report.steps.len(), 0);
}

#[tokio::test]
async fn baseline_at_early_stop_score_stops_immediately() {
    let target = FakeTarget::new(&[]);
    let evaluator = MapEvaluator::new(&target, 0.0);
    let mut optimizer = ScriptedOptimizer::new(vec![vec![candidate("cand-1")]]);
    let trainer = Trainer::new(TrainerConfig {
        early_stop_score: 0.0,
        ..config(3)
    });

    let report = run(
        &trainer,
        &target,
        &mut optimizer,
        &evaluator,
        &[],
        &[case("val-1")],
    )
    .await;

    assert_eq!(report.stopped_reason, StopReason::EarlyStop);
    assert_eq!(report.epochs_run, 0);
    assert_eq!(optimizer.observed_ids(), Vec::<&str>::new());
}

#[tokio::test]
async fn epoch_reaching_early_stop_score_stops_after_one_step() {
    let target = FakeTarget::new(&[("cand-1", 0.8)]);
    let evaluator = MapEvaluator::new(&target, 0.0);
    let mut optimizer =
        ScriptedOptimizer::new(vec![vec![candidate("cand-1")], vec![candidate("cand-2")]]);
    let trainer = Trainer::new(TrainerConfig {
        early_stop_score: 0.8,
        ..config(3)
    });

    let report = run(
        &trainer,
        &target,
        &mut optimizer,
        &evaluator,
        &[],
        &[case("val-1")],
    )
    .await;

    assert_eq!(report.stopped_reason, StopReason::EarlyStop);
    assert_eq!(report.epochs_run, 1);
    assert_eq!(report.steps[0].epoch, 0);
    assert!(close(report.best_score, 0.8));
}

#[tokio::test]
async fn callbacks_fire_in_order_for_one_epoch() {
    let target = FakeTarget::new(&[("cand-1", 0.5)]);
    let evaluator = MapEvaluator::new(&target, 0.0);
    let mut optimizer = ScriptedOptimizer::new(vec![vec![candidate("cand-1")]]);
    let recording = RecordingCallbacks::new();
    let callbacks: Vec<Arc<dyn Callbacks>> =
        vec![Arc::new(NoopCallbacks), Arc::new(recording.clone())];
    let trainer = Trainer::new(config(1)).with_callbacks(callbacks);

    let report = run(
        &trainer,
        &target,
        &mut optimizer,
        &evaluator,
        &[],
        &[case("val-1")],
    )
    .await;

    assert_eq!(report.epochs_run, 1);
    let expected = vec!["begin", "epoch_begin", "epoch_end", "end"];
    assert_eq!(recording.events(), expected);
    assert_eq!(recording.baselines(), vec![true]);
}

#[tokio::test]
async fn callbacks_fire_begin_and_end_on_baseline_early_stop() {
    let target = FakeTarget::new(&[]);
    let evaluator = MapEvaluator::new(&target, 0.0);
    let mut optimizer = ScriptedOptimizer::new(vec![vec![candidate("cand-1")]]);
    let recording = RecordingCallbacks::new();
    let trainer = Trainer::new(TrainerConfig {
        early_stop_score: 0.0,
        ..config(2)
    })
    .with_callbacks(vec![Arc::new(recording.clone()) as Arc<dyn Callbacks>]);

    let report = run(
        &trainer,
        &target,
        &mut optimizer,
        &evaluator,
        &[],
        &[case("val-1")],
    )
    .await;

    assert_eq!(report.stopped_reason, StopReason::EarlyStop);
    assert_eq!(recording.events(), vec!["begin", "end"]);
    assert_eq!(recording.baselines(), vec![true]);
}

#[test]
fn report_round_trips_and_renders_snake_case_stop_reason() {
    let report = TrainerReport {
        run_id: "run-1".to_owned(),
        epochs_run: 1,
        best_score: 0.5,
        best_candidate: Some(candidate("cand-1")),
        steps: vec![EpochReport {
            epoch: 0,
            score: 0.5,
            best_score: 0.5,
            improved: true,
            candidates: 1,
            evaluations: 2,
            checkpoint_saved: true,
        }],
        evaluations: 2,
        cancelled: false,
        stopped_reason: StopReason::Completed,
    };

    let value = serde_json::to_value(&report).expect("serialize");
    assert_eq!(value["stopped_reason"], json!("completed"));
    assert_eq!(value["steps"][0]["checkpoint_saved"], json!(true));
    let decoded: TrainerReport = serde_json::from_value(value).expect("deserialize");
    assert_eq!(decoded, report);
}

#[test]
fn config_defaults_match_python() {
    assert_eq!(
        TrainerConfig::default(),
        TrainerConfig {
            num_iterations: 3,
            num_parallel: 1,
            early_stop_score: 1.0,
            candidates_per_epoch: 1,
            max_case_evaluations: None,
            time_budget_ms: None,
            checkpoint_filename: "latest.json".to_owned(),
            resume_from: None,
        }
    );
}

#[test]
fn progress_runs_epochs_up_to_max() {
    let mut progress = Progress::new(2);
    assert_eq!(progress.run_epoch(), Some(1));
    assert_eq!(progress.run_epoch(), Some(2));
    assert_eq!(progress.run_epoch(), None);
}
