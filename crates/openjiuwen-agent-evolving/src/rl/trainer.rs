//! The offline RL orchestration loop.
//!
//! This is the portable replacement for Python's `rl_trainer` package: it rolls
//! tasks out through a [`RolloutEngine`], normalises rewards, packs samples,
//! applies policy updates, tracks the best epoch, and writes checkpoints. The
//! GPU engines themselves (`verl`, `ray`, vLLM) stay deferred behind the seams.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

use openjiuwen_common::error::{Error, Result};

use super::batch::{BatchPacker, RlBatch};
use super::checkpoint::RlCheckpoint;
use super::policy::{Policy, RlUpdate, RolloutEngine};
use super::rollout::{RewardNormaliser, RewardStats, RolloutLog};
use super::task::{RlTask, RolloutWithReward};

/// Configuration of [`RlTrainer`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RlTrainerConfig {
    /// Explicit run identifier; when `None` it is derived from the first task id.
    pub run_id: Option<String>,
    /// Number of epochs to run.
    pub epochs: usize,
    /// Advisory number of rollouts per task; rollout engines may override it
    /// (the fake engine reads its own setting or the task's `num_samples`).
    pub rollout_samples: usize,
    /// Stop once the best reward reaches this value.
    pub early_stop_reward: f64,
    /// Directory checkpoints are written to; `None` disables checkpointing.
    pub checkpoint_dir: Option<PathBuf>,
    /// Write a checkpoint every N epochs (`0` means every epoch).
    pub checkpoint_every: usize,
}

impl Default for RlTrainerConfig {
    fn default() -> Self {
        Self {
            run_id: None,
            epochs: 3,
            rollout_samples: 4,
            early_stop_reward: 1.0,
            checkpoint_dir: None,
            checkpoint_every: 1,
        }
    }
}

/// Why the training loop stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RlStopReason {
    /// All configured epochs ran.
    Completed,
    /// The best reward reached [`RlTrainerConfig::early_stop_reward`].
    EarlyStop,
    /// The cancellation token fired.
    Cancelled,
    /// No tasks were supplied.
    NoTasks,
}

/// Report for one epoch.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RlEpochReport {
    /// Zero-based epoch index.
    pub epoch: usize,
    /// Number of samples consumed by the policy updates.
    pub samples: usize,
    /// Mean normalised reward of the epoch.
    pub mean_reward: f64,
    /// Best normalised reward of the epoch.
    pub best_reward: f64,
    /// Whether at least one batch update was applied.
    pub updated: bool,
    /// File name of the checkpoint written for the epoch, when configured.
    pub checkpoint: Option<String>,
}

/// Report for a full training run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RlTrainReport {
    /// Identifier of the run.
    pub run_id: String,
    /// Number of completed epochs.
    pub epochs_run: usize,
    /// Best normalised reward across the completed epochs (`0.0` when none).
    pub best_reward: f64,
    /// Batch with the highest mean reward, if any batch was produced.
    pub best_batch: Option<RlBatch>,
    /// Per-epoch reports in completion order.
    pub epochs: Vec<RlEpochReport>,
    /// Why the loop stopped.
    pub stopped_reason: RlStopReason,
}

#[derive(Debug, Default)]
struct UpdateSummary {
    batches: usize,
    samples: usize,
    reward_sum: f64,
    best_reward: f64,
}

impl UpdateSummary {
    fn record(&mut self, update: &RlUpdate) {
        self.batches += 1;
        self.samples += update.samples;
        self.reward_sum += update.mean_reward * update.samples as f64;
        self.best_reward = self.best_reward.max(update.best_reward);
    }

    fn mean_reward(&self) -> f64 {
        if self.samples == 0 {
            0.0
        } else {
            self.reward_sum / self.samples as f64
        }
    }
}

/// The offline RL trainer: rollout, normalise, pack, update, checkpoint.
#[derive(Debug, Clone)]
pub struct RlTrainer {
    config: RlTrainerConfig,
    packer: BatchPacker,
    normaliser: RewardNormaliser,
}

impl RlTrainer {
    /// Creates a trainer with the default packer and group-centred normaliser.
    #[must_use]
    pub fn new(config: RlTrainerConfig) -> Self {
        Self {
            config,
            packer: BatchPacker::default(),
            normaliser: RewardNormaliser::default(),
        }
    }

    /// Replaces the batch packer.
    #[must_use]
    pub fn with_packer(mut self, packer: BatchPacker) -> Self {
        self.packer = packer;
        self
    }

    /// Replaces the reward normaliser.
    #[must_use]
    pub fn with_normaliser(mut self, normaliser: RewardNormaliser) -> Self {
        self.normaliser = normaliser;
        self
    }

    /// The configuration the trainer was built with.
    #[must_use]
    pub fn config(&self) -> &RlTrainerConfig {
        &self.config
    }

    /// Runs the training loop.
    ///
    /// Every epoch rolls all tasks out, normalises the pooled rewards once,
    /// packs the resulting samples, and applies one policy update per batch.
    /// Cancellation is honoured at every await point and produces a partial
    /// report with [`RlStopReason::Cancelled`]. The method is deterministic: it
    /// never reads a clock and derives its run id from the configuration or the
    /// first task id.
    pub async fn train(
        &self,
        policy: Arc<dyn Policy>,
        engine: &dyn RolloutEngine,
        tasks: &[RlTask],
        cancel: &CancellationToken,
    ) -> Result<RlTrainReport> {
        let run_id = self.run_id(tasks);
        let mut report = RlTrainReport {
            run_id: run_id.clone(),
            epochs_run: 0,
            best_reward: 0.0,
            best_batch: None,
            epochs: Vec::new(),
            stopped_reason: RlStopReason::Completed,
        };
        if tasks.is_empty() {
            report.stopped_reason = RlStopReason::NoTasks;
            return Ok(report);
        }
        for epoch in 0..self.config.epochs {
            if cancel.is_cancelled() {
                report.stopped_reason = RlStopReason::Cancelled;
                break;
            }
            let Some(mut pooled) = self.roll_epoch(&policy, engine, tasks, cancel).await? else {
                report.stopped_reason = RlStopReason::Cancelled;
                break;
            };
            let stats = self.normaliser.normalise(&mut pooled);
            let samples = RolloutLog::with_rollouts(run_id.clone(), pooled).to_samples();
            let batches = self.packer.pack(&samples);
            if cancel.is_cancelled() {
                report.stopped_reason = RlStopReason::Cancelled;
                break;
            }
            let Some(summary) = self
                .apply_batches(policy.as_ref(), &batches, cancel)
                .await?
            else {
                report.stopped_reason = RlStopReason::Cancelled;
                break;
            };
            let epoch_report = self
                .epoch_report(epoch, &run_id, &stats, &summary, policy.as_ref())
                .await?;
            report.best_reward = report.best_reward.max(epoch_report.best_reward);
            if let Some(best) = best_batch(&batches) {
                if report
                    .best_batch
                    .as_ref()
                    .is_none_or(|current| best.mean_reward() > current.mean_reward())
                {
                    report.best_batch = Some(best.clone());
                }
            }
            report.epochs.push(epoch_report);
            if report.best_reward >= self.config.early_stop_reward {
                report.stopped_reason = RlStopReason::EarlyStop;
                break;
            }
        }
        report.epochs_run = report.epochs.len();
        Ok(report)
    }

    fn run_id(&self, tasks: &[RlTask]) -> String {
        self.config.run_id.clone().unwrap_or_else(|| {
            tasks.first().map_or_else(
                || "rl-run".to_owned(),
                |task| format!("rl-{}", task.task_id),
            )
        })
    }

    async fn roll_epoch(
        &self,
        policy: &Arc<dyn Policy>,
        engine: &dyn RolloutEngine,
        tasks: &[RlTask],
        cancel: &CancellationToken,
    ) -> Result<Option<Vec<RolloutWithReward>>> {
        let mut pooled = Vec::new();
        for task in tasks {
            if cancel.is_cancelled() {
                return Ok(None);
            }
            match engine.roll(task, Arc::clone(policy), cancel).await {
                Ok(rollouts) => pooled.extend(rollouts),
                Err(Error::Cancelled) => return Ok(None),
                Err(error) => return Err(error),
            }
        }
        Ok(Some(pooled))
    }

    async fn apply_batches(
        &self,
        policy: &dyn Policy,
        batches: &[RlBatch],
        cancel: &CancellationToken,
    ) -> Result<Option<UpdateSummary>> {
        let mut summary = UpdateSummary::default();
        for batch in batches {
            if cancel.is_cancelled() {
                return Ok(None);
            }
            match policy.update(batch).await {
                Ok(update) => summary.record(&update),
                Err(Error::Cancelled) => return Ok(None),
                Err(error) => return Err(error),
            }
        }
        Ok(Some(summary))
    }

    async fn epoch_report(
        &self,
        epoch: usize,
        run_id: &str,
        stats: &RewardStats,
        summary: &UpdateSummary,
        policy: &dyn Policy,
    ) -> Result<RlEpochReport> {
        let updated = summary.batches > 0;
        let best_reward = summary.best_reward.max(stats.max);
        let checkpoint = self
            .maybe_checkpoint(epoch, run_id, best_reward, stats, policy)
            .await?;
        Ok(RlEpochReport {
            epoch,
            samples: if updated {
                summary.samples
            } else {
                stats.count
            },
            mean_reward: if updated {
                summary.mean_reward()
            } else {
                stats.mean
            },
            best_reward,
            updated,
            checkpoint,
        })
    }

    async fn maybe_checkpoint(
        &self,
        epoch: usize,
        run_id: &str,
        best_reward: f64,
        stats: &RewardStats,
        policy: &dyn Policy,
    ) -> Result<Option<String>> {
        let Some(dir) = self.config.checkpoint_dir.as_ref() else {
            return Ok(None);
        };
        if self.config.checkpoint_every != 0 && (epoch + 1) % self.config.checkpoint_every != 0 {
            return Ok(None);
        }
        let metrics = BTreeMap::from([
            ("mean_reward".to_owned(), stats.mean),
            ("best_reward".to_owned(), best_reward),
            ("samples".to_owned(), stats.count as f64),
        ]);
        let checkpoint =
            RlCheckpoint::new(run_id, epoch, best_reward, policy.parameters(), metrics);
        let path = checkpoint.save_to_dir(dir).await?;
        let file_name = path.file_name().map_or_else(
            || checkpoint.file_name(),
            |name| name.to_string_lossy().into_owned(),
        );
        Ok(Some(file_name))
    }
}

fn best_batch(batches: &[RlBatch]) -> Option<&RlBatch> {
    let mut best: Option<&RlBatch> = None;
    for batch in batches {
        if best.is_none_or(|current| batch.mean_reward() > current.mean_reward()) {
            best = Some(batch);
        }
    }
    best
}
