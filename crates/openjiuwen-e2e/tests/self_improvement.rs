#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Integration scenario: RSI (`openjiuwen::rsi`) self-improvement orchestration and
//! `openjiuwen::agent_evolving` trainer checkpoints, driven entirely by fakes.
//! Offline and deterministic: scripted runners/targets, temp dirs, no model, no network.
//!
//! Two seams are exercised end to end:
//!
//! 1. `rsi::AutoHarnessOrchestrator` graded by a scripted [`HarnessRunner`]: flat scores
//!    converge after `patience`, and a second orchestrator resumes a persisted session
//!    from a shared [`CheckpointStore`] instead of replaying it.
//! 2. `agent_evolving::trainer::Trainer` over a fake [`TrainableTarget`] and a grid
//!    search: an improving epoch checkpoints itself, and a resumed trainer restores the
//!    best score/run id from a real `FileCheckpointStore` before re-running the stored
//!    epoch.
//!
//! Every assertion reads recorded evidence — runner requests, iteration records, epoch
//! reports, checkpoint files — and never a timestamp or a generated identifier.
mod support;

use std::collections::{BTreeMap, VecDeque};
use std::path::Path;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use openjiuwen::agent_evolving::dataset::{Case, EvaluatedCase};
use openjiuwen::agent_evolving::evaluator::Evaluator;
use openjiuwen::agent_evolving::optimizer::{
    Candidate, CheckpointStore as TrainerCheckpointStore, DefaultCheckpointManager,
    FileCheckpointStore, GridSearchOptimizer, ParameterSpace,
};
use openjiuwen::agent_evolving::trainer::{TrainableTarget, Trainer, TrainerConfig};
use openjiuwen::core::common::error::Result;
use openjiuwen::rsi::{
    AutoHarnessConfig, AutoHarnessOrchestrator, Case as RsiCase, CaseResult,
    CheckpointStore as RsiCheckpointStore, EvaluationReport, HarnessRunOutcome, HarnessRunRequest,
    HarnessRunner, InMemoryCheckpointStore, StopReason,
};
use serde_json::{json, Value};
use support::within;
use tempfile::TempDir;
use tokio_util::sync::CancellationToken;

/// Session id the resume scenario writes and reads; the checkpoint key derives from it.
const RESUME_SESSION_ID: &str = "e2e-rsi-resume";

/// Target identifier shared by both trainer runs, and therefore the key the trainer
/// uses inside `EvolveCheckpoint.operators_state`.
const TARGET_ID: &str = "fake-target";

/// Checkpoint file name the trainer scenarios persist under their [`TempDir`].
const CHECKPOINT_FILE: &str = "latest.json";

// ---------------------------------------------------------------------------
// RSI fakes
// ---------------------------------------------------------------------------

/// A [`HarnessRunner`] that answers with queued scores and records every request.
///
/// The orchestrator only ever reads the produced [`EvaluationReport`], so a scripted
/// score queue is the whole seam. The request log is the evidence the scenarios assert
/// on: one request per iteration, carrying the continued iteration number.
struct ScriptedRunner {
    /// Scores handed out by successive runs, oldest first.
    scores: Mutex<VecDeque<f64>>,
    /// Every request the orchestrator issued, in call order.
    requests: Mutex<Vec<HarnessRunRequest>>,
}

impl ScriptedRunner {
    /// Creates a runner answering with `scores` in order.
    fn new(scores: Vec<f64>) -> Arc<Self> {
        Arc::new(Self {
            scores: Mutex::new(scores.into_iter().collect()),
            requests: Mutex::new(Vec::new()),
        })
    }

    /// Returns a copy of the recorded requests.
    fn requests(&self) -> Vec<HarnessRunRequest> {
        self.requests
            .lock()
            .expect("request log lock")
            .iter()
            .cloned()
            .collect()
    }
}

#[async_trait]
impl HarnessRunner for ScriptedRunner {
    fn name(&self) -> &'static str {
        "scripted-e2e"
    }

    async fn run(&self, request: HarnessRunRequest) -> Result<HarnessRunOutcome> {
        let score = {
            let mut scores = self.scores.lock().expect("score queue lock");
            scores
                .pop_front()
                .expect("scripted runner needs one score per run")
        };
        self.requests
            .lock()
            .expect("request log lock")
            .push(request);
        Ok(HarnessRunOutcome {
            report: graded_report(score),
            ..HarnessRunOutcome::default()
        })
    }
}

/// A one-case report scoring `score`.
///
/// A graded report is what makes a candidate eligible to become the incumbent, so the
/// fake must always provide a score.
fn graded_report(score: f64) -> EvaluationReport {
    EvaluationReport::from_results(vec![CaseResult {
        case_id: "case-1".to_owned(),
        passed: true,
        score: Some(score),
        weight: 1.0,
        error: None,
        details: json!({"note": "graded"}),
    }])
}

/// A configuration rooted at the scenario's temp directory, every other field defaulted.
///
/// `data_dir` is the only field `AutoHarnessConfig::validate` requires; the fully
/// scripted runner means no session artifact is ever written.
fn rsi_config(data_dir: &Path) -> AutoHarnessConfig {
    AutoHarnessConfig {
        data_dir: data_dir.to_path_buf(),
        ..AutoHarnessConfig::default()
    }
}

/// The single case every candidate is graded against.
fn rsi_case() -> RsiCase {
    RsiCase {
        id: "case-1".to_owned(),
        ..RsiCase::default()
    }
}

// ---------------------------------------------------------------------------
// Trainer fakes
// ---------------------------------------------------------------------------

/// State shared by [`FakeTarget`] and the scenario assertions.
#[derive(Debug, Default)]
struct TargetState {
    /// Candidate ids in application order.
    applied: Vec<String>,
    /// Score the currently applied candidate's parameter implies (`0.0` before any).
    current_quality: f64,
    /// Payload the last `restore` observed, when a checkpoint was resumed.
    restored: Option<Value>,
}

/// A [`TrainableTarget`] whose validation score is the applied candidate's `quality`.
///
/// It owns no model: `run_case` echoes the quality of the last applied candidate, so the
/// evaluator can turn a prediction into a deterministic score without hidden clocks or
/// randomness.
struct FakeTarget {
    /// Mutable target state, shared with the scenario's assertions.
    state: Arc<Mutex<TargetState>>,
}

impl FakeTarget {
    /// Creates a target with no candidate applied.
    fn new() -> Self {
        Self {
            state: Arc::new(Mutex::new(TargetState::default())),
        }
    }

    /// Returns a handle on the target state for assertions.
    fn state(&self) -> Arc<Mutex<TargetState>> {
        Arc::clone(&self.state)
    }
}

#[async_trait]
impl TrainableTarget for FakeTarget {
    fn id(&self) -> &'static str {
        TARGET_ID
    }

    async fn run_case(&self, case: &Case, cancel: &CancellationToken) -> Result<Value> {
        if cancel.is_cancelled() {
            return Err(openjiuwen::core::common::error::Error::cancelled());
        }
        let quality = self
            .state
            .lock()
            .expect("target state lock")
            .current_quality;
        Ok(json!({ "quality": quality, "case": case.case_id }))
    }

    async fn apply(&self, candidate: &Candidate) -> Result<()> {
        let quality = candidate
            .parameters
            .get("quality")
            .and_then(Value::as_f64)
            .unwrap_or(0.0);
        let mut state = self.state.lock().expect("target state lock");
        state.applied.push(candidate.id.clone());
        state.current_quality = quality;
        Ok(())
    }

    async fn snapshot(&self) -> Result<Value> {
        let state = self.state.lock().expect("target state lock");
        Ok(json!({ "version": 1, "applied": state.applied }))
    }

    async fn restore(&self, state: Value) -> Result<()> {
        self.state.lock().expect("target state lock").restored = Some(state);
        Ok(())
    }
}

/// An [`Evaluator`] that reads the score straight out of the target's prediction.
struct PredictionScoreEvaluator;

impl PredictionScoreEvaluator {
    /// The only prediction field the fake target produces.
    fn score(prediction: &Value) -> f64 {
        prediction
            .get("quality")
            .and_then(Value::as_f64)
            .unwrap_or(0.0)
    }
}

#[async_trait]
impl Evaluator for PredictionScoreEvaluator {
    fn name(&self) -> &'static str {
        "prediction-score"
    }

    async fn evaluate(&self, case: &Case, prediction: &Value) -> Result<EvaluatedCase> {
        EvaluatedCase::new(case.clone(), None, Self::score(prediction), "fake")
    }
}

/// A one-dimension grid: three candidates of increasing `quality`, in grid order.
fn quality_space() -> ParameterSpace {
    ParameterSpace::new().with_dimension("quality", vec![json!(0.2), json!(0.4), json!(0.6)])
}

/// A valid validation case with a stable id.
fn trainer_case(id: &str) -> Case {
    Case::new(
        BTreeMap::from([("question".to_owned(), json!(id))]),
        BTreeMap::from([("answer".to_owned(), json!(id))]),
    )
    .expect("valid case")
    .with_case_id(id)
}

/// `true` when both scores are within floating-point noise of each other.
fn close(actual: f64, expected: f64) -> bool {
    (actual - expected).abs() < 1e-9
}

// ---------------------------------------------------------------------------
// RSI scenarios
// ---------------------------------------------------------------------------

/// Flat scores exhaust `patience` and converge; the summary reports the session.
#[tokio::test]
async fn rsi_orchestrator_converges_in_two_iterations() {
    let dir = TempDir::new().expect("scenario temp dir");
    let runner = ScriptedRunner::new(vec![0.5, 0.5]);
    let orchestrator = AutoHarnessOrchestrator::builder(rsi_config(dir.path()))
        .with_runner(runner.clone())
        .with_cases(vec![rsi_case()])
        .with_session_id("e2e-rsi-converge")
        .build()
        .expect("orchestrator builds");

    let first = within("iteration 1", orchestrator.run_iteration())
        .await
        .expect("iteration 1 runs");
    // The first graded candidate has no incumbent to beat, so it is adopted and reports
    // no improvement.
    assert_eq!(first.iteration, 1);
    assert!(
        first.accepted,
        "the first graded candidate sets the incumbent"
    );
    assert!(close(first.improvement, 0.0));
    assert_eq!(first.stop, None, "patience needs a second flat iteration");

    let second = within("iteration 2", orchestrator.run_iteration())
        .await
        .expect("iteration 2 runs");
    assert_eq!(second.iteration, 2);
    assert!(
        !second.accepted,
        "an equal score does not replace the incumbent"
    );
    assert!(
        close(second.improvement, 0.0),
        "the best score did not move"
    );
    assert_eq!(second.stop, Some(StopReason::Converged));

    assert_eq!(
        orchestrator.state().stop_reason,
        Some(StopReason::Converged)
    );
    assert!(!orchestrator.is_running());

    let summary = within("finish", orchestrator.finish())
        .await
        .expect("finish closes the session");
    assert_eq!(summary.iterations, 2);
    assert_eq!(summary.evaluated, 2);
    assert_eq!(summary.accepted, 1, "only the baseline adoption counted");
    let best = summary.best_score.expect("the session graded a candidate");
    assert!(close(best, 0.5));
    assert_eq!(summary.stop_reason, Some(StopReason::Converged));

    let requests = runner.requests();
    assert_eq!(requests.len(), 2, "one harness run per iteration");
    assert_eq!(requests[0].session_id, "e2e-rsi-converge");
    assert_eq!(requests[0].iteration, 1);
    assert_eq!(requests[1].iteration, 2);
}

/// A second orchestrator resumes the persisted session and continues its numbering.
#[tokio::test]
async fn rsi_resume_continues_at_the_persisted_iteration() {
    let dir = TempDir::new().expect("scenario temp dir");
    let memory: Arc<InMemoryCheckpointStore> = Arc::new(InMemoryCheckpointStore::new());
    let store: Arc<dyn RsiCheckpointStore> = Arc::clone(&memory) as Arc<dyn RsiCheckpointStore>;

    let first = AutoHarnessOrchestrator::builder(rsi_config(dir.path()))
        .with_runner(ScriptedRunner::new(vec![0.1, 0.2]))
        .with_cases(vec![rsi_case()])
        .with_session_id(RESUME_SESSION_ID)
        .with_checkpoint_store(Arc::clone(&store))
        .build()
        .expect("orchestrator A builds");

    let one = within("iteration 1", first.run_iteration())
        .await
        .expect("iteration 1 runs");
    assert_eq!(one.iteration, 1);
    assert!(one.accepted, "the first scored candidate becomes incumbent");
    let two = within("iteration 2", first.run_iteration())
        .await
        .expect("iteration 2 runs");
    assert_eq!(two.iteration, 2);
    assert!(two.accepted, "0.2 improves on 0.1");
    assert!(
        first.state().stop_reason.is_none(),
        "the session can continue"
    );

    let persisted = memory
        .load(RESUME_SESSION_ID)
        .await
        .expect("checkpoint loads")
        .expect("checkpoint was written");
    assert_eq!(persisted.iteration, 2);
    assert_eq!(persisted.history.len(), 2);
    let persisted_best = persisted.best_score.expect("persisted incumbent score");
    assert!(close(persisted_best, 0.2));

    // A long score script keeps the resumed session from converging on its first
    // iteration, so the asserted continuation is about iteration numbering, not stopping.
    let resumed_runner = ScriptedRunner::new(vec![0.3, 0.35, 0.4]);
    let second = AutoHarnessOrchestrator::builder(rsi_config(dir.path()))
        .with_runner(resumed_runner.clone())
        .with_cases(vec![rsi_case()])
        .with_session_id(RESUME_SESSION_ID)
        .with_checkpoint_store(Arc::clone(&store))
        .build()
        .expect("orchestrator B builds");

    let restored = within("resume", second.resume())
        .await
        .expect("resume reads the store")
        .expect("a checkpoint exists");
    assert_eq!(restored.iteration, 2);
    assert_eq!(restored.history.len(), 2);
    let restored_best = restored.best_score.expect("restored incumbent score");
    assert!(close(restored_best, 0.2));
    assert_eq!(second.state().iteration, 2);
    assert!(second.is_running(), "resume marks the session running");

    let three = within("iteration 3", second.run_iteration())
        .await
        .expect("iteration 3 runs after resume");
    assert_eq!(three.iteration, 3, "numbering continues after the resume");
    let previous = three.previous_best_score.expect("resumed baseline");
    assert!(close(previous, 0.2));
    assert!(three.accepted, "0.3 improves on the restored 0.2");

    let calls = resumed_runner.requests();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].session_id, RESUME_SESSION_ID);
    assert_eq!(calls[0].iteration, 3);

    let recheckpointed = memory
        .load(RESUME_SESSION_ID)
        .await
        .expect("checkpoint loads")
        .expect("checkpoint exists after continuation");
    assert_eq!(recheckpointed.iteration, 3);
    assert_eq!(recheckpointed.history.len(), 3);
}

// ---------------------------------------------------------------------------
// Trainer scenarios
// ---------------------------------------------------------------------------

/// A grid search over a fake target improves the best score and writes checkpoints.
#[tokio::test]
async fn trainer_improves_a_fake_target_and_saves_checkpoints() {
    let dir = TempDir::new().expect("scenario temp dir");
    let store: Arc<dyn TrainerCheckpointStore> = Arc::new(FileCheckpointStore::new(dir.path()));
    let manager = Arc::new(
        DefaultCheckpointManager::new()
            .with_run_id("run-improve-e2e")
            // Periodic saves are pushed out of reach so only improvements checkpoint;
            // this is what makes `checkpoint_saved` evidence of the save-on-improve rule.
            .with_save_policy(100, true),
    );
    let trainer = Trainer::new(TrainerConfig {
        num_iterations: 3,
        checkpoint_filename: CHECKPOINT_FILE.to_owned(),
        ..TrainerConfig::default()
    })
    .with_checkpoint(Arc::clone(&store), manager);

    let target = FakeTarget::new();
    let state = target.state();
    let mut optimizer = GridSearchOptimizer::new(quality_space());
    let report = within(
        "train",
        trainer.train(
            &target,
            &mut optimizer,
            &PredictionScoreEvaluator,
            &[],
            &[trainer_case("val-1")],
            &CancellationToken::new(),
        ),
    )
    .await
    .expect("training completes");

    assert_eq!(report.run_id, "run-improve-e2e");
    assert_eq!(
        report.stopped_reason,
        openjiuwen::agent_evolving::trainer::StopReason::Completed
    );
    assert_eq!(report.epochs_run, 3);
    assert!(!report.cancelled);
    assert!(close(report.best_score, 0.6), "the last grid point wins");

    let best_candidate = report.best_candidate.expect("a candidate improved the run");
    assert_eq!(best_candidate.id, "cand-2");
    let applied = state.lock().expect("target state lock").applied.clone();
    assert_eq!(applied, vec!["cand-0", "cand-1", "cand-2"]);

    assert_eq!(report.steps.len(), 3);
    let improving: Vec<_> = report.steps.iter().filter(|step| step.improved).collect();
    assert_eq!(
        improving.len(),
        3,
        "every grid point improves on the previous best"
    );
    assert!(
        improving.iter().all(|step| step.checkpoint_saved),
        "save_on_improve checkpoints each improving epoch"
    );
    let last = report.steps.last().expect("three epoch reports");
    assert!(close(last.score, 0.6));
    assert!(close(last.best_score, 0.6));

    let checkpoint_path = dir.path().join(CHECKPOINT_FILE);
    assert!(
        checkpoint_path.is_file(),
        "the checkpoint file exists under the scenario temp dir"
    );
    let written: Value = serde_json::from_str(
        &std::fs::read_to_string(&checkpoint_path).expect("checkpoint is readable"),
    )
    .expect("checkpoint is valid JSON");
    assert_eq!(written["run_id"], json!("run-improve-e2e"));
    assert_eq!(written["best"]["best_score"], json!(0.6));
    assert_eq!(written["step"]["epoch"], json!(2));
}

/// A resumed trainer restores the best score/run id and re-runs the stored epoch.
#[tokio::test]
async fn trainer_resume_restores_best_score_and_reruns_the_stored_epoch() {
    let dir = TempDir::new().expect("scenario temp dir");
    let store: Arc<dyn TrainerCheckpointStore> = Arc::new(FileCheckpointStore::new(dir.path()));
    let save_config = TrainerConfig {
        num_iterations: 2,
        checkpoint_filename: CHECKPOINT_FILE.to_owned(),
        ..TrainerConfig::default()
    };

    let first_trainer = Trainer::new(save_config.clone()).with_checkpoint(
        Arc::clone(&store),
        Arc::new(DefaultCheckpointManager::new().with_run_id("run-resume-e2e")),
    );
    let first_report = within(
        "first train",
        first_trainer.train(
            &FakeTarget::new(),
            &mut GridSearchOptimizer::new(quality_space()),
            &PredictionScoreEvaluator,
            &[],
            &[trainer_case("val-1")],
            &CancellationToken::new(),
        ),
    )
    .await
    .expect("first training completes");
    assert_eq!(first_report.run_id, "run-resume-e2e");
    assert!(close(first_report.best_score, 0.4));

    let checkpoint_path = dir.path().join(CHECKPOINT_FILE);
    let stored: Value = serde_json::from_str(
        &std::fs::read_to_string(&checkpoint_path).expect("checkpoint is readable"),
    )
    .expect("checkpoint is valid JSON");
    let stored_epoch = usize::try_from(stored["step"]["epoch"].as_u64().expect("stored epoch"))
        .expect("epoch fits in usize");
    assert_eq!(stored_epoch, 1, "the last completed epoch was persisted");
    assert_eq!(stored["run_id"], json!("run-resume-e2e"));
    assert_eq!(stored["best"]["best_score"], json!(0.4));

    // A fresh target and optimizer in the same directory: only the checkpoint can carry
    // the run's history across. The manager's own run id must lose to the stored one.
    let target = FakeTarget::new();
    let state = target.state();
    let resume_trainer = Trainer::new(TrainerConfig {
        num_iterations: 3,
        ..save_config
    })
    .with_checkpoint(
        Arc::clone(&store),
        Arc::new(DefaultCheckpointManager::new().with_run_id("run-ignored")),
    )
    .with_resume_from(Some(CHECKPOINT_FILE.to_owned()));
    let report = within(
        "resumed train",
        resume_trainer.train(
            &target,
            &mut GridSearchOptimizer::new(quality_space()),
            &PredictionScoreEvaluator,
            &[],
            &[trainer_case("val-1")],
            &CancellationToken::new(),
        ),
    )
    .await
    .expect("resumed training completes");

    assert_eq!(
        report.steps[0].epoch, stored_epoch,
        "the stored epoch is re-run, not skipped"
    );
    assert_eq!(report.run_id, "run-resume-e2e", "the stored run id wins");
    assert!(close(report.best_score, 0.4), "the restored best is kept");
    assert!(
        report.steps.len() >= 2,
        "training continued past the stored epoch"
    );

    let restored = state
        .lock()
        .expect("target state lock")
        .restored
        .clone()
        .expect("the checkpoint payload reached the target");
    assert_eq!(restored[TARGET_ID]["version"], json!(1));
    assert_eq!(
        restored[TARGET_ID]["applied"],
        json!(["cand-0", "cand-1"]),
        "the snapshot of the improving epoch is what was persisted"
    );
}
