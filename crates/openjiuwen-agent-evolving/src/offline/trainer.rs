//! The offline trainer (Python `agent_rl/offline/main_trainer.py`).
//!
//! [`OfflineTrainer`] is the `MainTrainer` port: it walks the dataset in batches,
//! runs one [`TrainingCoordinator`] round loop per batch, applies the resulting
//! [`RlBatch`](crate::rl::RlBatch) through the [`Policy`] seam, records metrics,
//! writes step summaries, validates periodically and checkpoints at a cadence.
//! A run resumes from the newest checkpoint in its directory.
//!
//! Python's trainer hands the batch to verl, which owns the PPO update, the
//! distributed workers and the GPU. That half is not portable: the policy seam
//! here is a plain Rust trait, and a caller that wants the verl path must supply
//! a [`GpuTrainingExecutor`]. [`PpoTrainingExecutor`] is the honest stand-in — it
//! reports [`openjiuwen_common::error::Error::Unsupported`] naming `torch`,
//! `transformers`, `verl`, `ray` and vLLM, so `check_health` fails up front
//! instead of pretending to be ready.
//!
//! Error handling follows Python: when the coordinator or the policy fails, the
//! trainer writes a checkpoint (so the work up to that step is not lost) and then
//! re-raises the original error. A failure to *write* the checkpoint is reported
//! through `tracing` and never masks the original error.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use openjiuwen_common::error::{Error, Result};

use crate::rl::{Policy, RlBatch, RlCheckpoint, RlTask, RlUpdate};

use super::coordinator::{CoordinatorConfig, TrainingCoordinator, ValidationMetrics};
use super::experience::ExperienceCollector;
use super::metrics::{BatchDiagnostics, MetricsTracker, TrainingStepMetrics};
use super::store::RolloutPersistence;

/// The deferred area named by the GPU training seam.
pub const GPU_TRAINING_AREA: &str = "agent_rl/offline PPO training (verl/ray/vLLM)";

/// The Python ML stack the GPU path would need.
const GPU_REQUIREMENTS: &str = "torch, transformers, verl, ray and vLLM";

/// The error every GPU training entry point returns.
///
/// It carries [`crate::deferred::unsupported`]'s message so the deferral table
/// stays the single source of truth, plus the concrete dependency list so an
/// operator reading a log knows exactly what is missing.
#[must_use]
pub fn gpu_training_unavailable() -> Error {
    let base = crate::deferred::unsupported(GPU_TRAINING_AREA);
    Error::unsupported(format!(
        "{base}; it needs {GPU_REQUIREMENTS}, none of which are available in a GPU-free build"
    ))
}

/// The seam a GPU-backed PPO step plugs into.
///
/// The offline trainer calls [`Self::train_step`] instead of [`Policy::update`]
/// when an executor is installed, so a verl/ray implementation can be dropped in
/// without touching the loop.
#[async_trait]
pub trait GpuTrainingExecutor: Send + Sync {
    /// Stable name used in reports and diagnostics.
    fn name(&self) -> &str;

    /// Applies one planned batch.
    ///
    /// # Errors
    ///
    /// Whatever the backend cannot do; [`PpoTrainingExecutor`] always reports
    /// [`openjiuwen_common::error::Error::Unsupported`].
    async fn train_step(&self, batch: &RlBatch) -> Result<RlUpdate>;

    /// Reports whether the backend can run at all.
    ///
    /// # Errors
    ///
    /// [`openjiuwen_common::error::Error::Unsupported`] when the backend is not
    /// available, so a caller can fail before collecting rollouts.
    fn check_health(&self) -> Result<()>;
}

/// The PPO/verl executor: every entry point is typed `Unsupported`.
///
/// This is the seam Python's `MainTrainer` fills with verl. It exists so a
/// deployment that expects a GPU path fails loudly and early (health check)
/// instead of silently training nothing.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PpoTrainingExecutor;

impl PpoTrainingExecutor {
    /// Creates the executor.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

#[async_trait]
impl GpuTrainingExecutor for PpoTrainingExecutor {
    fn name(&self) -> &'static str {
        "ppo"
    }

    async fn train_step(&self, _batch: &RlBatch) -> Result<RlUpdate> {
        Err(gpu_training_unavailable())
    }

    fn check_health(&self) -> Result<()> {
        Err(gpu_training_unavailable())
    }
}

/// The trainer's configuration.
///
/// `validate_every` and `checkpoint_every` count *steps*; `0` disables the
/// corresponding action. `resume` reads the newest checkpoint in
/// `checkpoint_dir` before training, and `max_steps` bounds the run (Python's
/// `total_training_steps`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct OfflineTrainerConfig {
    /// Number of passes over the dataset.
    pub total_epochs: usize,
    /// Dataset rows per coordinator round loop.
    pub train_batch_size: usize,
    /// The coordinator configuration.
    pub rollout: CoordinatorConfig,
    /// Validate every N steps (`0` disables).
    pub validate_every: usize,
    /// Checkpoint every N steps (`0` disables).
    pub checkpoint_every: usize,
    /// Directory for checkpoints; `None` disables checkpointing entirely.
    pub checkpoint_dir: Option<PathBuf>,
    /// Identifier written into every checkpoint.
    pub run_id: String,
    /// Resume from the newest checkpoint in `checkpoint_dir`.
    pub resume: bool,
    /// Hard step budget (`None` = unbounded).
    pub max_steps: Option<usize>,
}

impl Default for OfflineTrainerConfig {
    fn default() -> Self {
        Self {
            total_epochs: 1,
            train_batch_size: 1,
            rollout: CoordinatorConfig::default(),
            validate_every: 0,
            checkpoint_every: 0,
            checkpoint_dir: None,
            run_id: "offline-trainer".to_owned(),
            resume: true,
            max_steps: None,
        }
    }
}

impl OfflineTrainerConfig {
    /// Validates the counts, the coordinator configuration and the run id.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] when an epoch or batch count is zero, the run id is
    /// blank, or [`CoordinatorConfig::validate`] fails.
    pub fn validate(&self) -> Result<()> {
        if self.total_epochs == 0 {
            return Err(Error::validation(
                "OfflineTrainerConfig total_epochs must be at least 1",
            ));
        }
        if self.train_batch_size == 0 {
            return Err(Error::validation(
                "OfflineTrainerConfig train_batch_size must be at least 1",
            ));
        }
        if self.run_id.trim().is_empty() {
            return Err(Error::validation(
                "OfflineTrainerConfig run_id must not be empty",
            ));
        }
        if self.max_steps == Some(0) {
            return Err(Error::validation(
                "OfflineTrainerConfig max_steps must be at least 1 when set",
            ));
        }
        self.rollout.validate()
    }
}

/// Why a run stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OfflineStopReason {
    /// Every requested epoch was trained.
    EpochsComplete,
    /// `max_steps` was reached.
    StepBudgetReached,
}

impl OfflineStopReason {
    /// The serde/wire label.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::EpochsComplete => "epochs_complete",
            Self::StepBudgetReached => "step_budget_reached",
        }
    }
}

/// One recorded validation pass.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ValidationSummary {
    /// The training step the pass ran after.
    pub step: usize,
    /// The epoch the step belongs to.
    pub epoch: usize,
    /// The `val/*` metrics.
    pub metrics: ValidationMetrics,
}

impl Default for ValidationSummary {
    fn default() -> Self {
        Self {
            step: 0,
            epoch: 0,
            metrics: ValidationMetrics::default(),
        }
    }
}

/// What one [`OfflineTrainer::fit`] did.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct OfflineTrainReport {
    /// The run identifier.
    pub run_id: String,
    /// Epochs trained (excluding the resumed ones).
    pub epochs: usize,
    /// Training steps executed.
    pub steps: usize,
    /// Samples consumed by the policy updates.
    pub samples: usize,
    /// Mean reward reported by the policy updates.
    pub mean_reward: f64,
    /// Best reward reported by the policy updates.
    pub best_reward: f64,
    /// Why the run stopped.
    pub stop_reason: Option<OfflineStopReason>,
    /// The epoch a resumed run started from.
    pub resumed_from: Option<usize>,
    /// Validation passes in execution order.
    pub validations: Vec<ValidationSummary>,
    /// Checkpoint paths written by this run.
    pub checkpoints: Vec<PathBuf>,
    /// The per-step metrics.
    pub metrics: Vec<TrainingStepMetrics>,
}

/// The offline trainer.
///
/// Construction validates the configuration and the coordinator, so a bad
/// configuration fails before any rollout is collected.
pub struct OfflineTrainer {
    config: OfflineTrainerConfig,
    coordinator: TrainingCoordinator,
    policy: Arc<dyn Policy>,
    tracker: MetricsTracker,
    collector: Option<Arc<dyn ExperienceCollector>>,
    gpu_executor: Option<Arc<dyn GpuTrainingExecutor>>,
}

impl std::fmt::Debug for OfflineTrainer {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OfflineTrainer")
            .field("config", &self.config)
            .field("policy", &self.policy.name())
            .finish_non_exhaustive()
    }
}

impl OfflineTrainer {
    /// Creates a trainer around a coordinator and a policy.
    ///
    /// # Errors
    ///
    /// The validation error of [`OfflineTrainerConfig::validate`].
    pub fn new(
        config: OfflineTrainerConfig,
        coordinator: TrainingCoordinator,
        policy: Arc<dyn Policy>,
    ) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            config,
            coordinator,
            policy,
            tracker: MetricsTracker::new(),
            collector: None,
            gpu_executor: None,
        })
    }

    /// Installs an experience collector, called for every collected rollout.
    #[must_use]
    pub fn with_experience_collector(mut self, collector: Arc<dyn ExperienceCollector>) -> Self {
        self.collector = Some(collector);
        self
    }

    /// Installs a GPU training executor, replacing [`Policy::update`].
    #[must_use]
    pub fn with_gpu_executor(mut self, executor: Arc<dyn GpuTrainingExecutor>) -> Self {
        self.gpu_executor = Some(executor);
        self
    }

    /// The configuration.
    #[must_use]
    pub fn config(&self) -> &OfflineTrainerConfig {
        &self.config
    }

    /// The coordinator.
    #[must_use]
    pub fn coordinator(&self) -> &TrainingCoordinator {
        &self.coordinator
    }

    /// The metrics tracker.
    #[must_use]
    pub fn tracker(&self) -> &MetricsTracker {
        &self.tracker
    }

    /// Trains over `rows`, validating on `validation_rows` at the cadence.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] for an empty dataset; a coordinator, policy,
    /// persistence or experience-collection error, re-raised after a checkpoint
    /// is written when one is configured.
    pub async fn fit(
        &mut self,
        rows: &[Value],
        validation_rows: &[Value],
    ) -> Result<OfflineTrainReport> {
        if rows.is_empty() {
            return Err(Error::validation(
                "OfflineTrainer::fit needs at least one dataset row",
            ));
        }
        let mut report = OfflineTrainReport {
            run_id: self.config.run_id.clone(),
            ..OfflineTrainReport::default()
        };
        let mut start_epoch = 0usize;
        let mut best_reward = 0.0f64;
        let mut step = 0usize;

        if self.config.resume {
            if let Some(dir) = &self.config.checkpoint_dir {
                if let Some(checkpoint) = RlCheckpoint::latest_in(dir).await? {
                    self.policy.restore(&checkpoint.policy)?;
                    best_reward = checkpoint.best_reward;
                    step = checkpoint.epoch;
                    start_epoch = checkpoint.epoch.saturating_add(1);
                    report.resumed_from = Some(checkpoint.epoch);
                }
            }
        }

        let mut stop_reason = OfflineStopReason::EpochsComplete;
        'epochs: for epoch in start_epoch..self.config.total_epochs {
            report.epochs += 1;
            let mut batch_index = 0usize;
            for chunk in rows.chunks(self.config.train_batch_size) {
                step = step.saturating_add(1);
                batch_index += 1;
                let outcome = self.train_step(step, epoch, batch_index, chunk).await;
                match outcome {
                    Ok((metrics, samples, update)) => {
                        report.steps += 1;
                        report.samples += samples;
                        report.mean_reward += update.mean_reward;
                        best_reward = best_reward.max(update.best_reward);
                        report.metrics.push(metrics);
                    }
                    Err(error) => {
                        self.save_checkpoint(epoch, step, best_reward, &report)
                            .await;
                        return Err(error);
                    }
                }
                if let Some(validation) = self
                    .maybe_validate(step, epoch, validation_rows)
                    .await?
                {
                    report.validations.push(validation);
                }
                if self.should_checkpoint(step) {
                    if let Some(path) = self
                        .write_checkpoint(epoch, step, best_reward, &report)
                        .await?
                    {
                        report.checkpoints.push(path);
                    }
                }
                if self.config.max_steps.is_some_and(|budget| step >= budget) {
                    stop_reason = OfflineStopReason::StepBudgetReached;
                    break 'epochs;
                }
            }
        }

        if self.should_checkpoint(step) {
            if let Some(path) = self
                .write_checkpoint(
                    self.config.total_epochs.saturating_sub(1),
                    step,
                    best_reward,
                    &report,
                )
                .await?
            {
                report.checkpoints.push(path);
            }
        }
        if report.steps > 0 {
            report.mean_reward /= report.steps as f64;
        }
        report.best_reward = best_reward;
        report.stop_reason = Some(stop_reason);
        Ok(report)
    }

    /// Runs one training step: the coordinator round loop and the policy update.
    async fn train_step(
        &mut self,
        step: usize,
        epoch: usize,
        batch_index: usize,
        rows: &[Value],
    ) -> Result<(TrainingStepMetrics, usize, RlUpdate)> {
        let round = self.coordinator.run_rounds(rows, step).await?;
        let batch = round.batch.to_rl_batch(format!("step-{step}-batch-{batch_index}"));
        let update = match &self.gpu_executor {
            Some(executor) => executor.train_step(&batch).await?,
            None => self.policy.update(&batch).await?,
        };

        self.collect_experience()?;

        let mut verl_metrics: BTreeMap<String, Value> = BTreeMap::new();
        for (key, value) in round.diagnostics.to_metrics() {
            verl_metrics.insert(key, Value::from(value));
        }
        for (key, value) in [
            ("rollout/collected", round.collected_rollouts as f64),
            ("rollout/training_samples", round.training_samples as f64),
            ("rollout/unique_prompts", round.unique_prompts as f64),
            ("rollout/positive_samples", round.positive_samples as f64),
            ("rollout/negative_samples", round.negative_samples as f64),
            ("rollout/avg_turns", round.avg_turns),
            ("rollout/reward_mean", round.reward_mean),
            ("rollout/dropped", round.dropped_rollouts.len() as f64),
            ("policy/samples", update.samples as f64),
            ("policy/mean_reward", update.mean_reward),
            ("policy/best_reward", update.best_reward),
        ] {
            verl_metrics.insert(key.to_owned(), Value::from(value));
        }
        let metrics = self.tracker.record_step(
            step,
            epoch,
            round.avg_turns,
            round.reward_mean,
            verl_metrics,
        );
        if metrics.consecutive_zero_reward_steps >= self.tracker.collapse_threshold() {
            tracing::warn!(
                step,
                consecutive = metrics.consecutive_zero_reward_steps,
                "rollout reward has been zero for several steps"
            );
        }
        if let Some(persistence) = self.coordinator.persistence() {
            let summary = step_summary(&metrics, &round.diagnostics);
            persistence.save_step_summary(step, &summary).await?;
        }
        Ok((metrics, update.samples, update))
    }

    async fn maybe_validate(
        &mut self,
        step: usize,
        epoch: usize,
        validation_rows: &[Value],
    ) -> Result<Option<ValidationSummary>> {
        let every = self.config.validate_every;
        if every == 0 || step % every != 0 || validation_rows.is_empty() {
            return Ok(None);
        }
        let metrics = self.coordinator.validate(validation_rows, step).await?;
        self.tracker.log_validation(step, metrics.to_metrics());
        Ok(Some(ValidationSummary {
            step,
            epoch,
            metrics,
        }))
    }

    fn should_checkpoint(&self, step: usize) -> bool {
        let every = self.config.checkpoint_every;
        every > 0 && step > 0 && step % every == 0 && self.config.checkpoint_dir.is_some()
    }

    async fn write_checkpoint(
        &self,
        epoch: usize,
        step: usize,
        best_reward: f64,
        report: &OfflineTrainReport,
    ) -> Result<Option<PathBuf>> {
        let Some(dir) = &self.config.checkpoint_dir else {
            return Ok(None);
        };
        let mut metrics = BTreeMap::new();
        if let Some(last) = report.metrics.last() {
            metrics = last.to_metrics();
        }
        metrics.insert("training/global_step".to_owned(), step as f64);
        metrics.insert("training/epoch".to_owned(), epoch as f64);
        metrics.insert("training/best_reward".to_owned(), best_reward);
        let checkpoint = RlCheckpoint::new(
            self.config.run_id.clone(),
            epoch,
            best_reward,
            self.policy.parameters(),
            metrics,
        );
        let path = checkpoint.save_to_dir(dir).await?;
        Ok(Some(path))
    }

    /// Best-effort checkpoint write used on the error path.
    ///
    /// A write failure is logged, never returned: the caller is about to
    /// re-raise the original training error and masking it with an I/O error
    /// would hide the cause.
    async fn save_checkpoint(
        &self,
        epoch: usize,
        step: usize,
        best_reward: f64,
        report: &OfflineTrainReport,
    ) {
        if self.config.checkpoint_dir.is_none() {
            return;
        }
        if let Err(error) = self.write_checkpoint(epoch, step, best_reward, report).await {
            tracing::warn!(%error, step, "could not write the failure checkpoint");
        }
    }

    fn collect_experience(&self) -> Result<()> {
        let Some(collector) = &self.collector else {
            return Ok(());
        };
        for message in self.coordinator.last_rollouts().values() {
            // The message carries the identifiers but not the dataset row, so the
            // collector receives a task shape without a sample payload.
            let task = RlTask {
                task_id: message.task_id.clone(),
                origin_task_id: message.origin_task_id.clone(),
                task_sample: Value::Null,
                round_num: message.round_num,
            };
            collector.collect(&task, message)?;
        }
        Ok(())
    }
}

/// The JSONL step summary persisted for one step.
fn step_summary(
    metrics: &TrainingStepMetrics,
    diagnostics: &BatchDiagnostics,
) -> BTreeMap<String, Value> {
    let mut summary: BTreeMap<String, Value> = BTreeMap::new();
    for (key, value) in metrics.to_metrics() {
        summary.insert(key, Value::from(value));
    }
    for (key, value) in diagnostics.to_metrics() {
        summary.insert(key, Value::from(value));
    }
    summary
}
