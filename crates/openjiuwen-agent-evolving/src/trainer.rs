//! Bounded train/evaluate loop, progress tracking, callbacks, and reports.
//!
//! Ports the trainer half of Python `openjiuwen.agent_evolving`: the loop is
//! candidate-driven, validates every candidate on the validation split, and stops
//! on early-stop score, cancellation, evaluation budget, wall-clock budget, or
//! candidate exhaustion.
//!
//! The loop is deliberately model-free: [`TrainableTarget`] produces predictions,
//! [`Evaluator`] scores them, and [`Optimizer`] proposes candidates, so the whole
//! loop runs offline and deterministically.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use openjiuwen_common::error::Result;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio_util::sync::CancellationToken;

use crate::dataset::Case;
use crate::evaluator::{evaluate_cases, CaseRunner, EvaluationConfig, EvaluationReport, Evaluator};
use crate::optimizer::{
    Candidate, CheckpointManager, CheckpointSnapshot, CheckpointStore, DefaultCheckpointManager,
    FileCheckpointStore, Optimizer,
};

/// Bounds and stopping conditions for [`Trainer::train`].
///
/// Defaults mirror Python's trainer configuration: three epochs, one validation
/// case evaluated at a time, one candidate per epoch, an early-stop score of
/// `1.0`, no evaluation/wall-clock budget, and `latest.json` as checkpoint file.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct TrainerConfig {
    /// Number of epochs (inclusive upper bound of the epoch range).
    pub num_iterations: usize,
    /// Number of cases evaluated concurrently; must be in `1..=20`.
    pub num_parallel: usize,
    /// Stop as soon as the best validation score reaches this value.
    pub early_stop_score: f64,
    /// Candidates proposed (and evaluated) per epoch.
    pub candidates_per_epoch: usize,
    /// Optional cap on the total number of case evaluations.
    pub max_case_evaluations: Option<usize>,
    /// Optional wall-clock budget for the whole run, in milliseconds.
    pub time_budget_ms: Option<u64>,
    /// Checkpoint filename passed to the configured [`CheckpointStore`].
    pub checkpoint_filename: String,
    /// Optional checkpoint name to resume from before training starts.
    pub resume_from: Option<String>,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            num_iterations: 3,
            num_parallel: 1,
            early_stop_score: 1.0,
            candidates_per_epoch: 1,
            max_case_evaluations: None,
            time_budget_ms: None,
            checkpoint_filename: "latest.json".to_owned(),
            resume_from: None,
        }
    }
}

/// The thing being tuned: runs cases, accepts candidates, and snapshots state.
///
/// This is the trainer's only handle on the tuned subject, replacing Python's
/// duck-typed operator chain. Implementations own their own synchronisation;
/// the trainer never holds a lock across `.await`.
#[async_trait]
pub trait TrainableTarget: Send + Sync {
    /// Stable identifier, used as the key inside `EvolveCheckpoint.operators_state`.
    fn id(&self) -> &str;

    /// Runs `case` and returns its prediction.
    ///
    /// # Errors
    ///
    /// [`openjiuwen_common::error::Error::Cancelled`] when `cancel` is cancelled,
    /// or the target's own failure.
    async fn run_case(&self, case: &Case, cancel: &CancellationToken) -> Result<Value>;

    /// Applies `candidate`'s parameters to the target.
    ///
    /// # Errors
    ///
    /// Returns an error when the candidate cannot be applied.
    async fn apply(&self, candidate: &Candidate) -> Result<()>;

    /// Captures the target's current state for checkpointing.
    ///
    /// # Errors
    ///
    /// Returns an error when the state cannot be serialized.
    async fn snapshot(&self) -> Result<Value> {
        Ok(Value::Null)
    }

    /// Restores previously snapshotted state (checkpoint resume).
    ///
    /// # Errors
    ///
    /// Returns an error when the state cannot be applied.
    async fn restore(&self, state: Value) -> Result<()> {
        let _ = state;
        Ok(())
    }
}

/// Mutable training-loop position and score bookkeeping.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub struct Progress {
    /// Epoch the run resumed from (zero when starting fresh).
    pub start_epoch: usize,
    /// Current epoch index.
    pub current_epoch: usize,
    /// Last epoch index, inclusive.
    pub max_epoch: usize,
    /// Current batch iteration inside the epoch.
    pub current_batch_iter: usize,
    /// Last batch iteration inside the epoch, inclusive.
    pub max_batch_iter: usize,
    /// Best validation score observed so far.
    pub best_score: f64,
    /// Best validation score observed in the current batch.
    pub best_batch_score: f64,
    /// Validation score of the current epoch.
    pub current_epoch_score: f64,
}

impl Progress {
    /// Creates zeroed progress bounded by `max_epoch`.
    #[must_use]
    pub fn new(max_epoch: usize) -> Self {
        Self {
            start_epoch: 0,
            current_epoch: 0,
            max_epoch,
            current_batch_iter: 0,
            max_batch_iter: 0,
            best_score: 0.0,
            best_batch_score: 0.0,
            current_epoch_score: 0.0,
        }
    }

    /// Advances to the next epoch, clipping at `max_epoch`.
    ///
    /// Starting from `start_epoch`, this yields `start_epoch + 1 ..= max_epoch`
    /// and then `None`.
    pub fn run_epoch(&mut self) -> Option<usize> {
        let next = self.current_epoch.saturating_add(1);
        if next > self.max_epoch {
            return None;
        }
        self.current_epoch = next;
        Some(next)
    }
}

/// Observation hooks for the training loop; every method defaults to a no-op.
pub trait Callbacks: Send + Sync {
    /// Called once before the baseline evaluation is reported.
    fn on_train_begin(&self, progress: &Progress, baseline: Option<&EvaluationReport>) {
        let _ = (progress, baseline);
    }

    /// Called at the start of every epoch.
    fn on_train_epoch_begin(&self, progress: &Progress) {
        let _ = progress;
    }

    /// Called at the end of every epoch with the winning validation report, if any.
    fn on_train_epoch_end(&self, progress: &Progress, report: Option<&EvaluationReport>) {
        let _ = (progress, report);
    }

    /// Called once when the loop stops.
    fn on_train_end(&self, progress: &Progress, report: Option<&EvaluationReport>) {
        let _ = (progress, report);
    }
}

/// A [`Callbacks`] implementation that does nothing.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoopCallbacks;

impl Callbacks for NoopCallbacks {}

/// Why [`Trainer::train`] returned.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    /// Every configured epoch ran.
    Completed,
    /// The best validation score reached `early_stop_score`.
    EarlyStop,
    /// The cancellation token fired.
    Cancelled,
    /// The evaluation or wall-clock budget would be exceeded.
    BudgetExhausted,
    /// The optimizer proposed no candidates.
    NoCandidates,
}

/// Per-epoch summary inside [`TrainerReport`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct EpochReport {
    /// Epoch index.
    pub epoch: usize,
    /// Best validation score after this epoch.
    pub score: f64,
    /// Best validation score over the whole run after this epoch.
    pub best_score: f64,
    /// Whether this epoch strictly improved the best score.
    pub improved: bool,
    /// Number of candidates proposed in this epoch.
    pub candidates: usize,
    /// Case evaluations spent in this epoch.
    pub evaluations: usize,
    /// Whether a checkpoint was written for this epoch.
    pub checkpoint_saved: bool,
}

/// Outcome of a training run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct TrainerReport {
    /// Run identifier, from the checkpoint manager or a restored checkpoint.
    pub run_id: String,
    /// Number of epochs that produced an [`EpochReport`].
    pub epochs_run: usize,
    /// Best validation score over the run.
    pub best_score: f64,
    /// Best candidate applied so far, if any improved the baseline.
    pub best_candidate: Option<Candidate>,
    /// Per-epoch summaries, in epoch order.
    pub steps: Vec<EpochReport>,
    /// Total case evaluations spent.
    pub evaluations: usize,
    /// Whether the run stopped because it was cancelled.
    pub cancelled: bool,
    /// Why the run stopped.
    pub stopped_reason: StopReason,
}

/// Runs the bounded train/evaluate loop.
pub struct Trainer {
    config: TrainerConfig,
    store: Arc<dyn CheckpointStore>,
    manager: Arc<dyn CheckpointManager>,
    callbacks: Vec<Arc<dyn Callbacks>>,
}

impl Trainer {
    /// Creates a trainer with a disabled checkpoint store and a default manager.
    #[must_use]
    pub fn new(config: TrainerConfig) -> Self {
        Self {
            config,
            store: Arc::new(FileCheckpointStore::disabled()),
            manager: Arc::new(DefaultCheckpointManager::new()),
            callbacks: Vec::new(),
        }
    }

    /// Replaces the checkpoint store and manager.
    #[must_use]
    pub fn with_checkpoint(
        mut self,
        store: Arc<dyn CheckpointStore>,
        manager: Arc<dyn CheckpointManager>,
    ) -> Self {
        self.store = store;
        self.manager = manager;
        self
    }

    /// Registers callbacks, called in vector order.
    #[must_use]
    pub fn with_callbacks(mut self, callbacks: Vec<Arc<dyn Callbacks>>) -> Self {
        self.callbacks = callbacks;
        self
    }

    /// Sets (or clears) the checkpoint to resume from.
    #[must_use]
    pub fn with_resume_from(mut self, name: Option<String>) -> Self {
        self.config.resume_from = name;
        self
    }

    /// The manager's run identifier.
    #[must_use]
    pub fn run_id(&self) -> &str {
        self.manager.run_id()
    }

    /// The trainer configuration.
    #[must_use]
    pub fn config(&self) -> &TrainerConfig {
        &self.config
    }

    /// Evaluates `cases` through `target` and scores them with `evaluator`.
    ///
    /// # Errors
    ///
    /// Propagates runner, evaluator, and configuration errors (for example an
    /// out-of-range `num_parallel`).
    pub async fn evaluate(
        &self,
        target: &dyn TrainableTarget,
        evaluator: &dyn Evaluator,
        cases: &[Case],
        cancel: &CancellationToken,
    ) -> Result<EvaluationReport> {
        let runner = TargetRunner { target };
        evaluate_cases(
            self.manager.run_id(),
            &runner,
            evaluator,
            cases,
            cancel,
            &EvaluationConfig {
                num_parallel: self.config.num_parallel,
                max_cases: None,
            },
        )
        .await
    }

    /// Runs baseline validation, then up to `num_iterations` candidate epochs.
    ///
    /// Resume semantics: a restored checkpoint's epoch is re-run, so the loop
    /// starts at `start_epoch` inclusive.
    ///
    /// # Errors
    ///
    /// Propagates checkpoint store, target `apply`/`snapshot`/`restore`, optimizer,
    /// evaluator, and configuration errors. Cancellation is reported in the
    /// returned [`TrainerReport`] (never as an error).
    #[allow(clippy::too_many_lines)]
    pub async fn train(
        &self,
        target: &dyn TrainableTarget,
        optimizer: &mut dyn Optimizer,
        evaluator: &dyn Evaluator,
        train_cases: &[Case],
        val_cases: &[Case],
        cancel: &CancellationToken,
    ) -> Result<TrainerReport> {
        let mut evaluations = 0_usize;
        let mut best_score = 0.0_f64;
        let mut best_candidate: Option<Candidate> = None;
        let mut run_id = self.manager.run_id().to_owned();
        let mut start_epoch = 0_usize;

        // Resume: a missing checkpoint is a silent no-op. The checkpointed epoch
        // is re-run because the checkpoint records the last completed epoch.
        if let Some(name) = &self.config.resume_from {
            if let Some(checkpoint) = self.store.load_checkpoint(name).await? {
                let restored = self.manager.restore(&checkpoint);
                run_id = restored.run_id;
                start_epoch = restored.start_epoch;
                best_score = restored.best_score;
                target
                    .restore(Value::Object(
                        restored.operators_state.into_iter().collect(),
                    ))
                    .await?;
            }
        }

        let started: Option<Instant> = self.config.time_budget_ms.map(|_| Instant::now());

        let mut progress = Progress::new(self.config.num_iterations);
        progress.start_epoch = start_epoch;
        progress.current_epoch = start_epoch;
        progress.best_score = best_score;

        // Baseline validation.
        let mut baseline: Option<EvaluationReport> = None;
        if !val_cases.is_empty() {
            if !self.can_afford(evaluations, val_cases.len()) {
                self.call_begin(&progress, None);
                return Ok(self.finish(
                    &progress,
                    None,
                    run_id,
                    best_score,
                    best_candidate,
                    Vec::new(),
                    evaluations,
                    StopReason::BudgetExhausted,
                ));
            }
            match self.evaluate(target, evaluator, val_cases, cancel).await {
                Ok(report) => {
                    evaluations += val_cases.len();
                    best_score = best_score.max(report.mean_score);
                    progress.best_score = best_score;
                    baseline = Some(report);
                }
                Err(error) if error.is_cancelled() => {
                    self.call_begin(&progress, None);
                    return Ok(self.finish(
                        &progress,
                        None,
                        run_id,
                        best_score,
                        best_candidate,
                        Vec::new(),
                        evaluations,
                        StopReason::Cancelled,
                    ));
                }
                Err(error) => return Err(error),
            }
        }
        self.call_begin(&progress, baseline.as_ref());
        if best_score >= self.config.early_stop_score {
            return Ok(self.finish(
                &progress,
                baseline.as_ref(),
                run_id,
                best_score,
                best_candidate,
                Vec::new(),
                evaluations,
                StopReason::EarlyStop,
            ));
        }

        let mut steps: Vec<EpochReport> = Vec::new();
        let mut last_report: Option<EvaluationReport> = None;
        let mut stop_reason = StopReason::Completed;
        let mut best_snapshot = Value::Null;

        for epoch in start_epoch..self.config.num_iterations {
            if cancel.is_cancelled() {
                stop_reason = StopReason::Cancelled;
                break;
            }
            if budget_exhausted(&started, self.config.time_budget_ms) {
                stop_reason = StopReason::BudgetExhausted;
                break;
            }

            progress.current_epoch = epoch;
            progress.best_score = best_score;
            for callback in &self.callbacks {
                callback.on_train_epoch_begin(&progress);
            }

            let score_at_epoch_start = best_score;
            let evaluations_at_epoch_start = evaluations;

            // Warm-up forward pass over the training split. Errors are recorded as
            // predictions and otherwise discarded, matching Python.
            if !train_cases.is_empty() {
                if !self.can_afford(evaluations, train_cases.len()) {
                    stop_reason = StopReason::BudgetExhausted;
                    break;
                }
                let mut forward_cancelled = false;
                for case in train_cases {
                    if cancel.is_cancelled() {
                        forward_cancelled = true;
                        break;
                    }
                    let _prediction = match target.run_case(case, cancel).await {
                        Ok(prediction) => prediction,
                        Err(error) if error.is_cancelled() => {
                            forward_cancelled = true;
                            break;
                        }
                        Err(error) => json!({ "error": error.to_string() }),
                    };
                    evaluations += 1;
                }
                if forward_cancelled {
                    stop_reason = StopReason::Cancelled;
                    break;
                }
            }

            let candidates = optimizer.propose(self.config.candidates_per_epoch).await?;
            if candidates.is_empty() {
                stop_reason = StopReason::NoCandidates;
                break;
            }
            let candidate_count = candidates.len();

            let mut epoch_best_candidate_score: Option<f64> = None;
            let mut epoch_report: Option<EvaluationReport> = None;
            let mut candidate_stop: Option<StopReason> = None;
            for candidate in &candidates {
                if cancel.is_cancelled() {
                    candidate_stop = Some(StopReason::Cancelled);
                    break;
                }
                if budget_exhausted(&started, self.config.time_budget_ms) {
                    candidate_stop = Some(StopReason::BudgetExhausted);
                    break;
                }
                if !val_cases.is_empty() && !self.can_afford(evaluations, val_cases.len()) {
                    candidate_stop = Some(StopReason::BudgetExhausted);
                    break;
                }

                target.apply(candidate).await?;
                let report = match self.evaluate(target, evaluator, val_cases, cancel).await {
                    Ok(report) => report,
                    Err(error) if error.is_cancelled() => {
                        candidate_stop = Some(StopReason::Cancelled);
                        break;
                    }
                    Err(error) => return Err(error),
                };
                evaluations += val_cases.len();
                let score = report.mean_score;
                optimizer.observe(candidate, score);
                // Every candidate counts towards the epoch's score, but only an
                // adoption changes the run's best.
                epoch_best_candidate_score = Some(match epoch_best_candidate_score {
                    Some(best) => best.max(score),
                    None => score,
                });
                // The first recorded candidate may tie the baseline (so a
                // candidate is reported even when nothing strictly improved);
                // later ones must strictly beat it. This keeps `best_score`
                // monotone: a candidate worse than the baseline is never adopted,
                // so the baseline stays the best score.
                let should_adopt = match &best_candidate {
                    None => score >= best_score,
                    Some(_) => score > best_score,
                };
                if should_adopt {
                    best_score = score;
                    best_candidate = Some(candidate.clone());
                    best_snapshot = target.snapshot().await?;
                    epoch_report = Some(report);
                }
            }
            if let Some(reason) = candidate_stop {
                stop_reason = reason;
                break;
            }

            let epoch_score = epoch_best_candidate_score.unwrap_or(best_score);
            let improved = epoch_score > score_at_epoch_start;
            let mut operators_state = BTreeMap::new();
            operators_state.insert(target.id().to_owned(), best_snapshot.clone());
            let snapshot = CheckpointSnapshot {
                epoch,
                batch: 0,
                best_score,
                current_epoch_score: epoch_score,
                seed: None,
                operators_state,
                updater_state: None,
                searcher_state: serde_json::to_value(optimizer.best()).ok(),
            };
            let checkpoint = self.manager.build_checkpoint(&snapshot);
            let checkpoint_saved = if self.manager.should_save(epoch, improved) {
                self.store
                    .save_checkpoint(&checkpoint, &self.config.checkpoint_filename)
                    .await?
                    .is_some()
            } else {
                false
            };

            steps.push(EpochReport {
                epoch,
                score: epoch_score,
                best_score,
                improved,
                candidates: candidate_count,
                evaluations: evaluations - evaluations_at_epoch_start,
                checkpoint_saved,
            });
            progress.current_epoch = epoch;
            progress.current_epoch_score = epoch_score;
            progress.best_score = best_score;
            for callback in &self.callbacks {
                callback.on_train_epoch_end(&progress, epoch_report.as_ref());
            }
            if epoch_report.is_some() {
                last_report = epoch_report;
            }

            if best_score >= self.config.early_stop_score {
                stop_reason = StopReason::EarlyStop;
                break;
            }
        }

        Ok(self.finish(
            &progress,
            last_report.as_ref(),
            run_id,
            best_score,
            best_candidate,
            steps,
            evaluations,
            stop_reason,
        ))
    }

    /// Whether `additional` more evaluations fit in the configured budget.
    fn can_afford(&self, evaluations: usize, additional: usize) -> bool {
        match self.config.max_case_evaluations {
            Some(max) => evaluations.saturating_add(additional) <= max,
            None => true,
        }
    }

    /// Fires `on_train_begin` on every callback.
    fn call_begin(&self, progress: &Progress, baseline: Option<&EvaluationReport>) {
        for callback in &self.callbacks {
            callback.on_train_begin(progress, baseline);
        }
    }

    /// Fires `on_train_end` and builds the final report.
    #[allow(clippy::too_many_arguments)]
    fn finish(
        &self,
        progress: &Progress,
        last_report: Option<&EvaluationReport>,
        run_id: String,
        best_score: f64,
        best_candidate: Option<Candidate>,
        steps: Vec<EpochReport>,
        evaluations: usize,
        stopped_reason: StopReason,
    ) -> TrainerReport {
        for callback in &self.callbacks {
            callback.on_train_end(progress, last_report);
        }
        stop_report(
            run_id,
            best_score,
            best_candidate,
            steps,
            evaluations,
            stopped_reason,
        )
    }
}

/// Adapts a [`TrainableTarget`] to the [`CaseRunner`] seam used by evaluation.
struct TargetRunner<'a> {
    target: &'a dyn TrainableTarget,
}

#[async_trait]
impl CaseRunner for TargetRunner<'_> {
    fn name(&self) -> &'static str {
        "trainable_target"
    }

    async fn run_case(&self, case: &Case, cancel: &CancellationToken) -> Result<Value> {
        self.target.run_case(case, cancel).await
    }
}

/// Builds the final [`TrainerReport`] from the accumulated loop state.
fn stop_report(
    run_id: String,
    best_score: f64,
    best_candidate: Option<Candidate>,
    steps: Vec<EpochReport>,
    evaluations: usize,
    stopped_reason: StopReason,
) -> TrainerReport {
    TrainerReport {
        run_id,
        epochs_run: steps.len(),
        best_score,
        best_candidate,
        steps,
        evaluations,
        cancelled: stopped_reason == StopReason::Cancelled,
        stopped_reason,
    }
}

/// Whether the optional wall-clock budget has been used up.
#[allow(clippy::ref_option)]
fn budget_exhausted(started: &Option<Instant>, budget_ms: Option<u64>) -> bool {
    match (started, budget_ms) {
        (Some(started), Some(limit)) => started.elapsed().as_millis() >= u128::from(limit),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use openjiuwen_common::error::Error;

    use super::*;
    use crate::dataset::EvaluatedCase;
    use crate::optimizer::{
        CheckpointBest, CheckpointStep, EvolveCheckpoint, MemoryCheckpointStore,
    };

    fn case(id: &str) -> Case {
        Case::new(
            BTreeMap::from([("question".to_owned(), json!("q"))]),
            BTreeMap::from([("answer".to_owned(), json!("a"))]),
        )
        .expect("valid case")
        .with_case_id(id)
    }

    fn cand(id: &str, score: f64) -> Candidate {
        Candidate::new(id, BTreeMap::from([("score".to_owned(), json!(score))]))
    }

    fn close(actual: f64, expected: f64) -> bool {
        (actual - expected).abs() < 1e-9
    }

    /// State shared between the fake target and the fake evaluator: candidates
    /// rewrite `score_table`, `apply` records ids, and `restore` records payloads.
    struct TargetState {
        applied: Vec<String>,
        score_table: BTreeMap<String, f64>,
        snapshot_value: Value,
        restored: Vec<Value>,
    }

    struct FakeTarget {
        state: Arc<Mutex<TargetState>>,
    }

    impl FakeTarget {
        fn new(scores: &[(&str, f64)]) -> Self {
            Self {
                state: Arc::new(Mutex::new(TargetState {
                    applied: Vec::new(),
                    score_table: scores
                        .iter()
                        .map(|(id, score)| ((*id).to_owned(), *score))
                        .collect(),
                    snapshot_value: json!({ "temperature": 0.5 }),
                    restored: Vec::new(),
                })),
            }
        }

        fn state(&self) -> Arc<Mutex<TargetState>> {
            Arc::clone(&self.state)
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
            Ok(json!({ "case_id": case.case_id }))
        }

        async fn apply(&self, candidate: &Candidate) -> Result<()> {
            let mut state = self.state.lock().expect("state");
            state.applied.push(candidate.id.clone());
            if let Some(score) = candidate.parameters.get("score").and_then(Value::as_f64) {
                for value in state.score_table.values_mut() {
                    *value = score;
                }
            }
            Ok(())
        }

        async fn snapshot(&self) -> Result<Value> {
            Ok(self.state.lock().expect("state").snapshot_value.clone())
        }

        async fn restore(&self, state: Value) -> Result<()> {
            self.state.lock().expect("state").restored.push(state);
            Ok(())
        }
    }

    /// Scores each case from the target's `case_id -> score` table and records the
    /// ids it evaluated.
    struct MapEvaluator {
        state: Arc<Mutex<TargetState>>,
        seen: Mutex<Vec<String>>,
    }

    impl MapEvaluator {
        fn new(state: Arc<Mutex<TargetState>>) -> Self {
            Self {
                state,
                seen: Mutex::new(Vec::new()),
            }
        }

        fn seen(&self) -> Vec<String> {
            self.seen.lock().expect("seen").clone()
        }
    }

    #[async_trait]
    impl Evaluator for MapEvaluator {
        fn name(&self) -> &'static str {
            "map"
        }

        async fn evaluate(&self, case: &Case, _prediction: &Value) -> Result<EvaluatedCase> {
            self.seen.lock().expect("seen").push(case.case_id.clone());
            let score = self
                .state
                .lock()
                .expect("state")
                .score_table
                .get(&case.case_id)
                .copied()
                .unwrap_or(0.0);
            EvaluatedCase::new(case.clone(), None, score, "fake")
        }
    }

    struct FakeOptimizer {
        pending: std::vec::IntoIter<Candidate>,
        observed: usize,
    }

    impl FakeOptimizer {
        fn new(candidates: Vec<Candidate>) -> Self {
            Self {
                pending: candidates.into_iter(),
                observed: 0,
            }
        }
    }

    #[async_trait]
    impl Optimizer for FakeOptimizer {
        fn name(&self) -> &'static str {
            "fake"
        }

        async fn propose(&mut self, limit: usize) -> Result<Vec<Candidate>> {
            let mut proposed = Vec::new();
            for _ in 0..limit {
                match self.pending.next() {
                    Some(candidate) => proposed.push(candidate),
                    None => break,
                }
            }
            Ok(proposed)
        }

        fn observe(&mut self, _candidate: &Candidate, _score: f64) {
            self.observed += 1;
        }
    }

    struct RecordingCallbacks {
        events: Arc<Mutex<Vec<&'static str>>>,
    }

    impl Callbacks for RecordingCallbacks {
        fn on_train_begin(&self, _progress: &Progress, _baseline: Option<&EvaluationReport>) {
            self.events.lock().expect("events").push("begin");
        }

        fn on_train_epoch_begin(&self, _progress: &Progress) {
            self.events.lock().expect("events").push("epoch_begin");
        }

        fn on_train_epoch_end(&self, _progress: &Progress, _report: Option<&EvaluationReport>) {
            self.events.lock().expect("events").push("epoch_end");
        }

        fn on_train_end(&self, _progress: &Progress, _report: Option<&EvaluationReport>) {
            self.events.lock().expect("events").push("end");
        }
    }

    #[tokio::test]
    async fn evaluate_returns_a_report() {
        let target = FakeTarget::new(&[("case-1", 0.75)]);
        let evaluator = MapEvaluator::new(target.state());
        let trainer = Trainer::new(TrainerConfig::default());

        let report = trainer
            .evaluate(
                &target,
                &evaluator,
                &[case("case-1")],
                &CancellationToken::new(),
            )
            .await
            .expect("evaluate");

        assert_eq!(report.num_cases, 1);
        assert!(close(report.mean_score, 0.75));
        assert_eq!(report.run_id, trainer.run_id());
        assert_eq!(evaluator.seen(), vec!["case-1".to_owned()]);
    }

    #[tokio::test]
    async fn train_runs_baseline_then_each_epoch() {
        let target = FakeTarget::new(&[("val-1", 0.4)]);
        let evaluator = MapEvaluator::new(target.state());
        let mut optimizer = FakeOptimizer::new(vec![cand("cand-1", 0.4), cand("cand-2", 0.4)]);
        let trainer = Trainer::new(TrainerConfig {
            num_iterations: 2,
            ..TrainerConfig::default()
        });

        let report = trainer
            .train(
                &target,
                &mut optimizer,
                &evaluator,
                &[case("train-1")],
                &[case("val-1")],
                &CancellationToken::new(),
            )
            .await
            .expect("train");

        assert_eq!(report.stopped_reason, StopReason::Completed);
        assert_eq!(report.epochs_run, 2);
        assert_eq!(report.steps.len(), 2);
        assert_eq!(report.steps[0].epoch, 0);
        assert!(!report.steps[0].improved);
        assert!(close(report.best_score, 0.4));
        // The first candidate becomes the incumbent even when it only ties the
        // baseline; later ties do not replace it.
        assert_eq!(
            report
                .best_candidate
                .as_ref()
                .map(|candidate| candidate.id.as_str()),
            Some("cand-1")
        );
        // Baseline (1) + two forward passes (2) + two candidate evaluations (2).
        assert_eq!(report.evaluations, 5);
        assert_eq!(
            target.state().lock().expect("state").applied,
            vec!["cand-1".to_owned(), "cand-2".to_owned()]
        );
        assert_eq!(optimizer.observed, 2);
    }

    #[tokio::test]
    async fn a_candidate_below_the_baseline_never_regresses_the_best_score() {
        let target = FakeTarget::new(&[("val-1", 0.5)]);
        let evaluator = MapEvaluator::new(target.state());
        let mut optimizer = FakeOptimizer::new(vec![cand("cand-1", 0.2)]);
        let store = Arc::new(MemoryCheckpointStore::new());
        let manager = Arc::new(DefaultCheckpointManager::new().with_run_id("run-baseline"));
        let trainer = Trainer::new(TrainerConfig {
            num_iterations: 1,
            ..TrainerConfig::default()
        })
        .with_checkpoint(Arc::clone(&store) as Arc<dyn CheckpointStore>, manager);

        let report = trainer
            .train(
                &target,
                &mut optimizer,
                &evaluator,
                &[],
                &[case("val-1")],
                &CancellationToken::new(),
            )
            .await
            .expect("train");

        assert_eq!(report.stopped_reason, StopReason::Completed);
        assert_eq!(report.epochs_run, 1);
        // The baseline stays the best score and the candidate is not adopted.
        assert!(close(report.best_score, 0.5));
        assert!(report.best_candidate.is_none());
        // The epoch still reports the candidate it saw, and did not improve.
        assert!(close(report.steps[0].score, 0.2));
        assert!(close(report.steps[0].best_score, 0.5));
        assert!(!report.steps[0].improved);
        // Epoch 0 always saves, and the checkpoint records the baseline score.
        assert!(report.steps[0].checkpoint_saved);
        let checkpoint = store
            .load_checkpoint("latest.json")
            .await
            .expect("load")
            .expect("checkpoint written");
        assert!(close(checkpoint.best.best_score, 0.5));
    }

    #[tokio::test]
    async fn a_better_candidate_is_adopted_after_a_worse_one() {
        let target = FakeTarget::new(&[("val-1", 0.5)]);
        let evaluator = MapEvaluator::new(target.state());
        let mut optimizer = FakeOptimizer::new(vec![cand("cand-1", 0.2), cand("cand-2", 0.8)]);
        let trainer = Trainer::new(TrainerConfig {
            num_iterations: 1,
            candidates_per_epoch: 2,
            ..TrainerConfig::default()
        });

        let report = trainer
            .train(
                &target,
                &mut optimizer,
                &evaluator,
                &[],
                &[case("val-1")],
                &CancellationToken::new(),
            )
            .await
            .expect("train");

        assert_eq!(report.stopped_reason, StopReason::Completed);
        assert!(close(report.best_score, 0.8));
        assert_eq!(
            report
                .best_candidate
                .as_ref()
                .map(|candidate| candidate.id.as_str()),
            Some("cand-2")
        );
        assert!(close(report.steps[0].score, 0.8));
        assert!(close(report.steps[0].best_score, 0.8));
        assert!(report.steps[0].improved);
        assert_eq!(
            target.state().lock().expect("state").applied,
            vec!["cand-1".to_owned(), "cand-2".to_owned()]
        );
    }

    #[tokio::test]
    async fn early_stop_paths() {
        // The baseline alone can trip the threshold.
        let target = FakeTarget::new(&[("val-1", 0.4)]);
        let evaluator = MapEvaluator::new(target.state());
        let mut optimizer = FakeOptimizer::new(vec![cand("cand-1", 0.9)]);
        let trainer = Trainer::new(TrainerConfig {
            early_stop_score: 0.0,
            ..TrainerConfig::default()
        });
        let report = trainer
            .train(
                &target,
                &mut optimizer,
                &evaluator,
                &[],
                &[case("val-1")],
                &CancellationToken::new(),
            )
            .await
            .expect("train");
        assert_eq!(report.stopped_reason, StopReason::EarlyStop);
        assert_eq!(report.epochs_run, 0);
        assert_eq!(report.steps, Vec::<EpochReport>::new());
        assert_eq!(report.evaluations, 1);
        assert!(close(report.best_score, 0.4));
        assert!(!report.cancelled);

        // An epoch can trip it too.
        let target = FakeTarget::new(&[("val-1", 0.2)]);
        let evaluator = MapEvaluator::new(target.state());
        let mut optimizer = FakeOptimizer::new(vec![cand("cand-1", 0.9)]);
        let trainer = Trainer::new(TrainerConfig {
            num_iterations: 3,
            early_stop_score: 0.9,
            ..TrainerConfig::default()
        });
        let report = trainer
            .train(
                &target,
                &mut optimizer,
                &evaluator,
                &[],
                &[case("val-1")],
                &CancellationToken::new(),
            )
            .await
            .expect("train");
        assert_eq!(report.stopped_reason, StopReason::EarlyStop);
        assert_eq!(report.epochs_run, 1);
        assert!(report.steps[0].improved);
        assert!(close(report.best_score, 0.9));
        assert_eq!(
            report
                .best_candidate
                .as_ref()
                .map(|candidate| candidate.id.as_str()),
            Some("cand-1")
        );
    }

    #[tokio::test]
    async fn termination_paths() {
        let empty_cancel = CancellationToken::new();

        // No candidates ends the run before any epoch report is produced.
        let target = FakeTarget::new(&[("val-1", 0.4)]);
        let evaluator = MapEvaluator::new(target.state());
        let mut optimizer = FakeOptimizer::new(Vec::new());
        let trainer = Trainer::new(TrainerConfig::default());
        let report = trainer
            .train(
                &target,
                &mut optimizer,
                &evaluator,
                &[case("train-1")],
                &[case("val-1")],
                &empty_cancel,
            )
            .await
            .expect("train");
        assert_eq!(report.stopped_reason, StopReason::NoCandidates);
        assert_eq!(report.epochs_run, 0);
        assert_eq!(report.steps, Vec::<EpochReport>::new());
        // Baseline (1) + forward pass (1).
        assert_eq!(report.evaluations, 2);

        // The evaluation budget can stop after the baseline...
        let target = FakeTarget::new(&[("val-1", 0.4)]);
        let evaluator = MapEvaluator::new(target.state());
        let mut optimizer = FakeOptimizer::new(vec![cand("cand-1", 0.9)]);
        let trainer = Trainer::new(TrainerConfig {
            max_case_evaluations: Some(1),
            ..TrainerConfig::default()
        });
        let report = trainer
            .train(
                &target,
                &mut optimizer,
                &evaluator,
                &[],
                &[case("val-1")],
                &empty_cancel,
            )
            .await
            .expect("train");
        assert_eq!(report.stopped_reason, StopReason::BudgetExhausted);
        assert_eq!(report.epochs_run, 0);
        assert_eq!(report.evaluations, 1);

        // ...or before it (a zero budget cannot even afford the baseline).
        let target = FakeTarget::new(&[("val-1", 0.4)]);
        let evaluator = MapEvaluator::new(target.state());
        let mut optimizer = FakeOptimizer::new(vec![cand("cand-1", 0.9)]);
        let trainer = Trainer::new(TrainerConfig {
            max_case_evaluations: Some(0),
            ..TrainerConfig::default()
        });
        let report = trainer
            .train(
                &target,
                &mut optimizer,
                &evaluator,
                &[],
                &[case("val-1")],
                &empty_cancel,
            )
            .await
            .expect("train");
        assert_eq!(report.stopped_reason, StopReason::BudgetExhausted);
        assert_eq!(report.evaluations, 0);

        // The wall-clock budget is checked before the first epoch.
        let target = FakeTarget::new(&[]);
        let evaluator = MapEvaluator::new(target.state());
        let mut optimizer = FakeOptimizer::new(vec![cand("cand-1", 0.9)]);
        let trainer = Trainer::new(TrainerConfig {
            time_budget_ms: Some(0),
            ..TrainerConfig::default()
        });
        let report = trainer
            .train(&target, &mut optimizer, &evaluator, &[], &[], &empty_cancel)
            .await
            .expect("train");
        assert_eq!(report.stopped_reason, StopReason::BudgetExhausted);
        assert_eq!(report.epochs_run, 0);

        // A pre-cancelled token stops before any work.
        let target = FakeTarget::new(&[]);
        let evaluator = MapEvaluator::new(target.state());
        let mut optimizer = FakeOptimizer::new(vec![cand("cand-1", 0.9)]);
        let trainer = Trainer::new(TrainerConfig::default());
        let cancel = CancellationToken::new();
        cancel.cancel();
        let report = trainer
            .train(&target, &mut optimizer, &evaluator, &[], &[], &cancel)
            .await
            .expect("train");
        assert_eq!(report.stopped_reason, StopReason::Cancelled);
        assert!(report.cancelled);
        assert_eq!(report.epochs_run, 0);
    }

    #[tokio::test]
    async fn callbacks_fire_in_order() {
        let target = FakeTarget::new(&[("val-1", 0.2)]);
        let evaluator = MapEvaluator::new(target.state());
        let mut optimizer = FakeOptimizer::new(vec![cand("cand-1", 0.9), cand("cand-2", 0.9)]);
        let events: Arc<Mutex<Vec<&'static str>>> = Arc::new(Mutex::new(Vec::new()));
        let recorder = Arc::new(RecordingCallbacks {
            events: Arc::clone(&events),
        });
        let trainer = Trainer::new(TrainerConfig {
            num_iterations: 2,
            ..TrainerConfig::default()
        })
        .with_callbacks(vec![recorder]);

        let report = trainer
            .train(
                &target,
                &mut optimizer,
                &evaluator,
                &[],
                &[case("val-1")],
                &CancellationToken::new(),
            )
            .await
            .expect("train");

        assert_eq!(report.epochs_run, 2);
        assert_eq!(
            *events.lock().expect("events"),
            vec![
                "begin",
                "epoch_begin",
                "epoch_end",
                "epoch_begin",
                "epoch_end",
                "end",
            ]
        );
    }

    #[tokio::test]
    async fn checkpoint_is_saved_when_an_epoch_improves() {
        let target = FakeTarget::new(&[("val-1", 0.2)]);
        let evaluator = MapEvaluator::new(target.state());
        let mut optimizer = FakeOptimizer::new(vec![cand("cand-1", 0.9)]);
        let store = Arc::new(MemoryCheckpointStore::new());
        let manager = Arc::new(DefaultCheckpointManager::new().with_run_id("run-checkpoint"));
        let trainer = Trainer::new(TrainerConfig {
            num_iterations: 1,
            ..TrainerConfig::default()
        })
        .with_checkpoint(Arc::clone(&store) as Arc<dyn CheckpointStore>, manager);

        let report = trainer
            .train(
                &target,
                &mut optimizer,
                &evaluator,
                &[],
                &[case("val-1")],
                &CancellationToken::new(),
            )
            .await
            .expect("train");

        assert!(report.steps[0].improved);
        assert!(report.steps[0].checkpoint_saved);

        let checkpoint = store
            .load_checkpoint("latest.json")
            .await
            .expect("load")
            .expect("checkpoint written");
        assert_eq!(checkpoint.run_id, "run-checkpoint");
        assert_eq!(checkpoint.step, CheckpointStep { epoch: 0, batch: 0 });
        assert!(close(checkpoint.best.best_score, 0.9));
        assert_eq!(
            checkpoint.operators_state.get("target"),
            Some(&json!({ "temperature": 0.5 }))
        );
    }

    #[tokio::test]
    async fn resume_restores_epoch_and_best_score() {
        let store = MemoryCheckpointStore::new();
        let checkpoint = EvolveCheckpoint {
            version: "v1".to_owned(),
            run_id: "run-restored".to_owned(),
            step: CheckpointStep { epoch: 1, batch: 0 },
            best: CheckpointBest { best_score: 0.5 },
            seed: None,
            operators_state: BTreeMap::from([("target".to_owned(), json!({ "temperature": 0.7 }))]),
            updater_state: None,
            searcher_state: None,
            last_metrics: BTreeMap::new(),
        };
        store
            .save_checkpoint(&checkpoint, "latest.json")
            .await
            .expect("save");

        let target = FakeTarget::new(&[("val-1", 0.2)]);
        let evaluator = MapEvaluator::new(target.state());
        let mut optimizer = FakeOptimizer::new(vec![cand("cand-1", 0.6)]);
        let trainer = Trainer::new(TrainerConfig {
            num_iterations: 2,
            ..TrainerConfig::default()
        })
        .with_checkpoint(
            Arc::new(store),
            Arc::new(DefaultCheckpointManager::new().with_run_id("run-new")),
        )
        .with_resume_from(Some("latest.json".to_owned()));

        assert_eq!(trainer.run_id(), "run-new");

        let report = trainer
            .train(
                &target,
                &mut optimizer,
                &evaluator,
                &[],
                &[case("val-1")],
                &CancellationToken::new(),
            )
            .await
            .expect("train");

        assert_eq!(report.run_id, "run-restored");
        assert_eq!(report.steps.len(), 1);
        assert_eq!(report.steps[0].epoch, 1);
        assert!(close(report.best_score, 0.6));
        // `restore` receives the whole `operators_state` object, keyed by target id.
        assert_eq!(
            target
                .state()
                .lock()
                .expect("state")
                .restored
                .first()
                .cloned(),
            Some(json!({ "target": { "temperature": 0.7 } }))
        );
    }

    #[test]
    fn progress_config_and_serde() {
        let mut progress = Progress::new(2);
        assert_eq!(progress.run_epoch(), Some(1));
        assert_eq!(progress.run_epoch(), Some(2));
        assert_eq!(progress.run_epoch(), None);
        progress.current_epoch = 7;
        assert_eq!(progress.run_epoch(), None);

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
        let restored: TrainerConfig =
            serde_json::from_value(json!({ "num_iterations": 5, "unknown": true }))
                .expect("deserialize");
        assert_eq!(restored.num_iterations, 5);
        assert_eq!(restored.num_parallel, 1);
        assert_eq!(restored.checkpoint_filename, "latest.json");

        assert_eq!(
            serde_json::to_value(StopReason::NoCandidates).expect("serialize"),
            json!("no_candidates")
        );
        assert_eq!(
            serde_json::from_value::<StopReason>(json!("budget_exhausted")).expect("deserialize"),
            StopReason::BudgetExhausted
        );

        let report = TrainerReport {
            run_id: "run-1".to_owned(),
            epochs_run: 0,
            best_score: 0.0,
            best_candidate: None,
            steps: Vec::new(),
            evaluations: 0,
            cancelled: false,
            stopped_reason: StopReason::Completed,
        };
        let value = serde_json::to_value(&report).expect("serialize");
        assert_eq!(value["stopped_reason"], json!("completed"));
        assert_eq!(value["cancelled"], json!(false));
        assert_eq!(
            serde_json::from_value::<TrainerReport>(value).expect("deserialize"),
            report
        );
    }
}
