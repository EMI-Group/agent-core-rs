//! The training coordinator (Python `coordinator/training_coordinator.py`).
//!
//! [`TrainingCoordinator`] owns the rollout-feedback-update cycle:
//!
//! 1. expand the dataset rows into `rollout_n` tasks per row, all sharing one
//!    `origin_task_id` (the GRPO group);
//! 2. submit them to a [`TaskQueue`] served by [`ParallelRuntimeExecutor`] worker
//!    loops and wait (bounded) for the queue to drain;
//! 3. encode the collected rollouts, classify them per group, apply the stop
//!    validator and re-submit the unfinished groups for the next round;
//! 4. sample the caches, merge them and plan one [`BatchPlan`].
//!
//! Validation reuses steps 1–2 with one task per row and reports the Python
//! `val/*` metrics.
//!
//! Every identifier is derived from the step and the row index
//! (`origin-{step}-{row}`, `task-{step}-{row}-{k}`) instead of `uuid4`, so a run
//! is reproducible. `rollout_n`, `runtime_parallel_num` and the length limits are
//! validated at construction: a zero would silently produce an empty round.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use openjiuwen_common::error::{Error, Result};

use crate::rl::{RlTask, RolloutMessage, RolloutWithReward};

use super::batch::{BatchBuilder, BatchConfig, BatchPlan};
use super::encoding::TurnEncoder;
use super::metrics::BatchDiagnostics;
use super::processors::{
    ProcessorsRegistry, DEFAULT_CLASSIFIER, DEFAULT_SAMPLER, DEFAULT_VALIDATOR,
};
use super::queue::TaskQueue;
use super::runtime::{ParallelRuntimeExecutor, RewardComposer, RolloutSource, RuntimeExecutor};
use super::store::{RolloutPersistence, RolloutPhase};
use super::{mean, rollout_turns};

/// The per-group progress the coordinator tracks across rounds.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct RolloutState {
    /// Positive samples collected for the group.
    pub pos: usize,
    /// Negative samples collected for the group.
    pub neg: usize,
    /// Whether the stop validator accepted the group.
    pub finished: bool,
}

/// The reward of one collected rollout.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RolloutRewards {
    /// The rollout's global reward.
    pub global: f64,
    /// The rollout's per-turn rewards.
    pub per_turn: Vec<f64>,
}

impl Default for RolloutRewards {
    fn default() -> Self {
        Self {
            global: 0.0,
            per_turn: Vec::new(),
        }
    }
}

/// The coordinator's configuration.
///
/// The field names follow Python's `JiuwenRL`/`trainer`/`data` sections. The
/// processor names select entries of the [`ProcessorsRegistry`], and
/// `round_timeout_ms` bounds one round's collection so a stuck rollout surfaces
/// as [`Error::timeout`] instead of hanging the trainer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct CoordinatorConfig {
    /// Number of rollout worker loops.
    pub runtime_parallel_num: usize,
    /// Rollouts produced per dataset row.
    pub rollout_n: usize,
    /// Maximum number of collection rounds per step.
    pub rollout_max_round: usize,
    /// Whether a whole rollout becomes one sample with a per-token loss mask.
    pub whole_trajectory: bool,
    /// Per-prompt sample target used by the validators and samplers.
    pub final_keep_per_prompt: usize,
    /// Maximum prompt length of a batch row.
    pub max_prompt_length: usize,
    /// Maximum response length of a batch row.
    pub max_response_length: usize,
    /// Pad token id of a batch row.
    pub pad_token_id: u32,
    /// Name of the classifier in the registry.
    pub classifier: String,
    /// Name of the stop validator in the registry.
    pub validator: String,
    /// Name of the sampler in the registry.
    pub sampler: String,
    /// Upper bound, in milliseconds, for one round's collection.
    pub round_timeout_ms: u64,
}

impl Default for CoordinatorConfig {
    fn default() -> Self {
        Self {
            runtime_parallel_num: 1,
            rollout_n: 1,
            rollout_max_round: 1,
            whole_trajectory: false,
            final_keep_per_prompt: 8,
            max_prompt_length: 1024,
            max_response_length: 1024,
            pad_token_id: 0,
            classifier: DEFAULT_CLASSIFIER.to_owned(),
            validator: DEFAULT_VALIDATOR.to_owned(),
            sampler: DEFAULT_SAMPLER.to_owned(),
            round_timeout_ms: 30_000,
        }
    }
}

impl CoordinatorConfig {
    /// Validates the counts and lengths.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] when a worker count, rollout count, round count or
    /// length is zero, or when `final_keep_per_prompt` is zero (the samplers
    /// would select nothing).
    pub fn validate(&self) -> Result<()> {
        if self.runtime_parallel_num == 0 {
            return Err(Error::validation(
                "CoordinatorConfig runtime_parallel_num must be at least 1",
            ));
        }
        if self.rollout_n == 0 {
            return Err(Error::validation(
                "CoordinatorConfig rollout_n must be at least 1",
            ));
        }
        if self.rollout_max_round == 0 {
            return Err(Error::validation(
                "CoordinatorConfig rollout_max_round must be at least 1",
            ));
        }
        if self.final_keep_per_prompt == 0 {
            return Err(Error::validation(
                "CoordinatorConfig final_keep_per_prompt must be at least 1",
            ));
        }
        BatchConfig::new(
            self.max_prompt_length,
            self.max_response_length,
            self.pad_token_id,
        )
        .validate()
    }

    /// The batch configuration implied by the coordinator's lengths.
    #[must_use]
    pub const fn batch_config(&self) -> BatchConfig {
        BatchConfig::new(
            self.max_prompt_length,
            self.max_response_length,
            self.pad_token_id,
        )
    }

    fn round_timeout(&self) -> Duration {
        Duration::from_millis(self.round_timeout_ms.max(1))
    }
}

/// What one collection round did.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RoundReport {
    /// Zero-based round index.
    pub round_id: usize,
    /// Tasks submitted at the start of the round.
    pub submitted: usize,
    /// Rollouts collected in the round.
    pub collected: usize,
    /// Rollouts whose turns could not be encoded.
    pub dropped_rollouts: usize,
    /// Groups the stop validator accepted in this round.
    pub finished_tasks: usize,
    /// Groups re-submitted for the next round.
    pub remaining_tasks: usize,
}

/// The result of one [`TrainingCoordinator::run_rounds`].
#[derive(Debug, Clone, PartialEq)]
pub struct CoordinatorReport {
    /// The training step the round loop ran for.
    pub step: usize,
    /// The planned training batch.
    pub batch: BatchPlan,
    /// One entry per executed round.
    pub rounds: Vec<RoundReport>,
    /// Rollouts collected over all rounds.
    pub collected_rollouts: usize,
    /// Samples in the final batch (post-split granularity).
    pub training_samples: usize,
    /// Distinct prompt groups in the batch.
    pub unique_prompts: usize,
    /// Positive samples accumulated over all rounds.
    pub positive_samples: usize,
    /// Negative samples accumulated over all rounds.
    pub negative_samples: usize,
    /// Why rollouts were dropped, in collection order.
    pub dropped_rollouts: Vec<String>,
    /// Mean turns per collected rollout.
    pub avg_turns: f64,
    /// Mean global reward over the collected rollouts.
    pub reward_mean: f64,
    /// Per-group rewards of the collected rollouts.
    pub rewards_by_uid: BTreeMap<String, Vec<RolloutRewards>>,
    /// Deterministic diagnostics of the batch.
    pub diagnostics: BatchDiagnostics,
}

/// The `val/*` metrics of one validation pass.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ValidationMetrics {
    /// Mean global reward.
    pub global_reward_mean: f64,
    /// Share of rollouts with `global_reward >= 0.9`.
    pub accuracy: f64,
    /// Number of such rollouts.
    pub correct_count: usize,
    /// Number of validation rollouts.
    pub sample_count: usize,
    /// Mean number of turns per rollout.
    pub average_turn_num: f64,
    /// Per-rollout turn counts.
    pub turn_num: Vec<usize>,
    /// Per-rollout reward lists.
    pub reward_list: Vec<Vec<f64>>,
}

impl ValidationMetrics {
    /// Renders the metrics under Python's `val/` keys.
    #[must_use]
    pub fn to_metrics(&self) -> BTreeMap<String, Value> {
        let mut metrics = BTreeMap::new();
        metrics.insert(
            "val/global_reward_mean".to_owned(),
            Value::from(self.global_reward_mean),
        );
        metrics.insert("val/accuracy".to_owned(), Value::from(self.accuracy));
        metrics.insert(
            "val/correct_count".to_owned(),
            Value::from(self.correct_count),
        );
        metrics.insert(
            "val/sample_count".to_owned(),
            Value::from(self.sample_count),
        );
        metrics.insert(
            "val/average_turn_num".to_owned(),
            Value::from(self.average_turn_num),
        );
        metrics.insert(
            "val/turn_num".to_owned(),
            Value::from(self.turn_num.clone()),
        );
        metrics.insert(
            "val/reward_list".to_owned(),
            Value::from(self.reward_list.clone()),
        );
        metrics
    }
}

/// Orchestrates rollout collection, classification, sampling and batching.
///
/// One coordinator serves one trainer: [`Self::run_rounds`] and
/// [`Self::validate`] both reset the accumulated state first, so a step never
/// sees the previous step's caches.
pub struct TrainingCoordinator {
    config: CoordinatorConfig,
    source: Arc<dyn RolloutSource>,
    composer: Option<Arc<dyn RewardComposer>>,
    persistence: Option<Arc<dyn RolloutPersistence>>,
    processors: ProcessorsRegistry,
    encoder: TurnEncoder,
    queue: Arc<TaskQueue>,
    positive_cache: BTreeMap<String, Vec<RolloutWithReward>>,
    negative_cache: BTreeMap<String, Vec<RolloutWithReward>>,
    rollout_state: BTreeMap<String, RolloutState>,
    turn_counts: Vec<usize>,
    rewards_by_uid: BTreeMap<String, Vec<RolloutRewards>>,
    last_rollouts: BTreeMap<String, RolloutMessage>,
    total_positive: usize,
    total_negative: usize,
    last_avg_turn_count: f64,
    last_training_sample_count: usize,
}

impl std::fmt::Debug for TrainingCoordinator {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("TrainingCoordinator")
            .field("config", &self.config)
            .field("source", &self.source.name())
            .finish_non_exhaustive()
    }
}

impl TrainingCoordinator {
    /// Creates a coordinator over `source`.
    ///
    /// # Errors
    ///
    /// The validation error of [`CoordinatorConfig::validate`], or
    /// [`Error::validation`] when a configured processor name is not registered.
    pub fn new(config: CoordinatorConfig, source: Arc<dyn RolloutSource>) -> Result<Self> {
        config.validate()?;
        let processors = ProcessorsRegistry::new();
        // Fail at construction rather than in the middle of a training step.
        processors.classifier(&config.classifier)?;
        processors.validator(&config.validator)?;
        processors.sampler(&config.sampler)?;
        Ok(Self {
            config,
            source,
            composer: None,
            persistence: None,
            processors,
            encoder: TurnEncoder::new(),
            queue: Arc::new(TaskQueue::new()),
            positive_cache: BTreeMap::new(),
            negative_cache: BTreeMap::new(),
            rollout_state: BTreeMap::new(),
            turn_counts: Vec::new(),
            rewards_by_uid: BTreeMap::new(),
            last_rollouts: BTreeMap::new(),
            total_positive: 0,
            total_negative: 0,
            last_avg_turn_count: 0.0,
            last_training_sample_count: 0,
        })
    }

    /// Adds reward composition to the rollout executor.
    #[must_use]
    pub fn with_composer(mut self, composer: Arc<dyn RewardComposer>) -> Self {
        self.composer = Some(composer);
        self
    }

    /// Adds rollout persistence.
    #[must_use]
    pub fn with_persistence(mut self, persistence: Arc<dyn RolloutPersistence>) -> Self {
        self.persistence = Some(persistence);
        self
    }

    /// Replaces the processor registry (for custom classifier/validator/sampler).
    ///
    /// The configured names are re-checked against the new registry.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] when a configured name is missing from `processors`.
    pub fn with_processors(mut self, processors: ProcessorsRegistry) -> Result<Self> {
        processors.classifier(&self.config.classifier)?;
        processors.validator(&self.config.validator)?;
        processors.sampler(&self.config.sampler)?;
        self.processors = processors;
        Ok(self)
    }

    /// The configuration.
    #[must_use]
    pub fn config(&self) -> &CoordinatorConfig {
        &self.config
    }

    /// The task queue the rounds submit to.
    #[must_use]
    pub fn queue(&self) -> &Arc<TaskQueue> {
        &self.queue
    }

    /// The persistence backend, when one is configured.
    #[must_use]
    pub fn persistence(&self) -> Option<&Arc<dyn RolloutPersistence>> {
        self.persistence.as_ref()
    }

    /// The accumulated positive cache.
    #[must_use]
    pub fn positive_cache(&self) -> &BTreeMap<String, Vec<RolloutWithReward>> {
        &self.positive_cache
    }

    /// The accumulated negative cache.
    #[must_use]
    pub fn negative_cache(&self) -> &BTreeMap<String, Vec<RolloutWithReward>> {
        &self.negative_cache
    }

    /// The per-group progress of the last run.
    #[must_use]
    pub fn rollout_state(&self) -> &BTreeMap<String, RolloutState> {
        &self.rollout_state
    }

    /// The per-group rewards of the last run.
    #[must_use]
    pub fn rewards_by_uid(&self) -> &BTreeMap<String, Vec<RolloutRewards>> {
        &self.rewards_by_uid
    }

    /// The rollouts collected by the last [`Self::run_rounds`] call.
    ///
    /// Keyed by rollout id and kept so a trainer can feed its experience
    /// collector without re-reading the persistence layer.
    #[must_use]
    pub fn last_rollouts(&self) -> &BTreeMap<String, RolloutMessage> {
        &self.last_rollouts
    }

    /// Mean turns per collected rollout of the last run.
    #[must_use]
    pub fn last_avg_turn_count(&self) -> f64 {
        self.last_avg_turn_count
    }

    /// Samples in the last planned batch.
    #[must_use]
    pub fn last_training_sample_count(&self) -> usize {
        self.last_training_sample_count
    }

    /// Resets the caches and counters for a fresh cycle.
    pub async fn clear_up_data(&mut self) {
        self.positive_cache.clear();
        self.negative_cache.clear();
        self.rollout_state.clear();
        self.turn_counts.clear();
        self.rewards_by_uid.clear();
        self.last_rollouts.clear();
        self.total_positive = 0;
        self.total_negative = 0;
        self.last_avg_turn_count = 0.0;
        self.last_training_sample_count = 0;
        self.queue.clear().await;
    }

    /// Merges a positive and a negative cache into one rollout map.
    #[must_use]
    pub fn merge_caches(
        pos_cache: &BTreeMap<String, Vec<RolloutWithReward>>,
        neg_cache: &BTreeMap<String, Vec<RolloutWithReward>>,
    ) -> BTreeMap<String, Vec<RolloutWithReward>> {
        let mut merged: BTreeMap<String, Vec<RolloutWithReward>> = BTreeMap::new();
        for (uid, list) in pos_cache {
            merged.insert(uid.clone(), list.clone());
        }
        for (uid, list) in neg_cache {
            merged
                .entry(uid.clone())
                .or_default()
                .extend(list.iter().cloned());
        }
        merged
    }

    /// Runs the round loop for `rows` and plans one training batch.
    ///
    /// Each row is expanded into `rollout_n` tasks; the loop submits the
    /// unfinished tasks, collects and encodes the rollouts, classifies them,
    /// applies the stop validator and re-submits the unfinished groups until
    /// `rollout_max_round` is reached. The caches are then sampled and merged
    /// into a [`BatchPlan`].
    ///
    /// # Errors
    ///
    /// [`Error::validation`] for an empty row set, an unknown processor name or
    /// an empty final batch; [`Error::timeout`] when a round does not drain
    /// within `round_timeout_ms`; a persistence error from the configured
    /// backend.
    pub async fn run_rounds(&mut self, rows: &[Value], step: usize) -> Result<CoordinatorReport> {
        if rows.is_empty() {
            return Err(Error::validation(
                "TrainingCoordinator::run_rounds needs at least one dataset row",
            ));
        }
        self.clear_up_data().await;
        let mut tasks = self.build_initial_tasks(rows, step);
        let mut rounds = Vec::with_capacity(self.config.rollout_max_round);
        let mut collected_rollouts = 0usize;
        let mut dropped = Vec::new();

        for round_id in 0..self.config.rollout_max_round {
            if tasks.is_empty() {
                break;
            }
            let submitted = tasks.len();
            for task in tasks.values() {
                self.queue.queue_task(task.clone()).await;
                self.rollout_state
                    .entry(task.origin_task_id.clone().unwrap_or_default())
                    .or_default();
            }

            let executor = self.build_executor();
            executor.start().await?;
            let finish = executor.wait_for_finish(self.config.round_timeout()).await;
            let collected = self.queue.get_rollouts().await;
            executor.stop().await?;
            finish?;

            self.persist_rollouts(step, &collected, RolloutPhase::Train)
                .await?;
            collected_rollouts += collected.len();

            let grouped = self.collect_round(&collected, &mut dropped);
            let finished = self.update_rollout_state(&grouped)?;
            let (remaining, finished_count) =
                Self::filter_unfinished_tasks(&self.rollout_state, tasks);
            tasks = remaining;
            rounds.push(RoundReport {
                round_id,
                submitted,
                collected: collected.len(),
                dropped_rollouts: dropped.len(),
                finished_tasks: finished.max(finished_count),
                remaining_tasks: tasks.len(),
            });
            self.last_rollouts.extend(collected);
        }

        let sampler = self.processors.sampler(&self.config.sampler)?;
        let keep = Some(self.config.final_keep_per_prompt);
        let (sampled_pos, sampled_neg) = sampler(&self.positive_cache, &self.negative_cache, keep);
        let merged = Self::merge_caches(&sampled_pos, &sampled_neg);
        let builder = BatchBuilder::new(self.config.batch_config());
        let batch = builder.assemble_batch(&merged)?;

        self.last_training_sample_count = batch.sample_count();
        self.last_avg_turn_count = mean(self.turn_counts.iter().map(|count| *count as f64));
        let reward_mean = mean(
            self.rewards_by_uid
                .values()
                .flatten()
                .map(|rewards| rewards.global),
        );
        let diagnostics = BatchDiagnostics::from_plan(&batch);

        Ok(CoordinatorReport {
            step,
            batch,
            rounds,
            collected_rollouts,
            training_samples: self.last_training_sample_count,
            unique_prompts: self.rewards_by_uid.len(),
            positive_samples: self.total_positive,
            negative_samples: self.total_negative,
            dropped_rollouts: dropped,
            avg_turns: self.last_avg_turn_count,
            reward_mean,
            rewards_by_uid: self.rewards_by_uid.clone(),
            diagnostics,
        })
    }

    /// Runs one validation pass over `rows`.
    ///
    /// Validation submits one task per row (no `rollout_n` multiplication), never
    /// retries and never classifies: the metrics describe the rollouts as they
    /// came. Rollouts are persisted under the `val` phase.
    ///
    /// # Errors
    ///
    /// [`Error::timeout`] when the pass does not drain in time, or a persistence
    /// error from the configured backend. An empty row set yields zero metrics
    /// (Python's `len(next(iter(rl_data.values())))` would raise there).
    pub async fn validate(&mut self, rows: &[Value], step: usize) -> Result<ValidationMetrics> {
        self.clear_up_data().await;
        if rows.is_empty() {
            return Ok(ValidationMetrics::default());
        }
        let tasks: Vec<RlTask> = rows
            .iter()
            .enumerate()
            .map(|(index, row)| {
                let task_id = format!("val-{step}-{index}");
                RlTask {
                    task_id: task_id.clone(),
                    origin_task_id: Some(task_id),
                    task_sample: row.clone(),
                    round_num: 0,
                }
            })
            .collect();
        for task in &tasks {
            self.queue.queue_task(task.clone()).await;
        }

        let executor = self.build_executor();
        executor.start().await?;
        let finish = executor.wait_for_finish(self.config.round_timeout()).await;
        let collected = self.queue.get_rollouts().await;
        executor.stop().await?;
        finish?;
        self.persist_rollouts(step, &collected, RolloutPhase::Val)
            .await?;

        let mut global_rewards = Vec::with_capacity(collected.len());
        let mut turn_num = Vec::with_capacity(collected.len());
        let mut reward_list = Vec::with_capacity(collected.len());
        for message in collected.values() {
            let turns = match rollout_turns(message) {
                Ok(turns) => turns,
                Err(error) => {
                    tracing::warn!(rollout_id = %message.rollout_id, %error, "skipping a malformed validation rollout");
                    continue;
                }
            };
            turn_num.push(turns.len());
            reward_list.push(message.reward_list.clone());
            global_rewards.push(if message.global_reward.is_finite() {
                message.global_reward
            } else {
                0.0
            });
        }

        let sample_count = global_rewards.len();
        let correct_count = global_rewards
            .iter()
            .filter(|reward| **reward >= 0.9)
            .count();
        Ok(ValidationMetrics {
            global_reward_mean: mean(global_rewards.iter().copied()),
            accuracy: if sample_count == 0 {
                0.0
            } else {
                correct_count as f64 / sample_count as f64
            },
            correct_count,
            sample_count,
            average_turn_num: mean(turn_num.iter().map(|count| *count as f64)),
            turn_num,
            reward_list,
        })
    }

    fn build_executor(&self) -> ParallelRuntimeExecutor {
        let mut executor = RuntimeExecutor::new(Arc::clone(&self.source));
        if let Some(composer) = &self.composer {
            executor = executor.with_composer(Arc::clone(composer));
        }
        ParallelRuntimeExecutor::new(
            Arc::clone(&self.queue),
            self.config.runtime_parallel_num,
            Arc::new(executor),
        )
    }

    fn build_initial_tasks(&self, rows: &[Value], step: usize) -> BTreeMap<String, RlTask> {
        let mut tasks = BTreeMap::new();
        for (row_index, row) in rows.iter().enumerate() {
            let origin = format!("origin-{step}-{row_index}");
            for sample in 0..self.config.rollout_n {
                let task_id = format!("task-{step}-{row_index}-{sample}");
                tasks.insert(
                    task_id.clone(),
                    RlTask {
                        task_id,
                        origin_task_id: Some(origin.clone()),
                        task_sample: row.clone(),
                        round_num: 0,
                    },
                );
            }
        }
        tasks
    }

    async fn persist_rollouts(
        &self,
        step: usize,
        collected: &BTreeMap<String, RolloutMessage>,
        phase: RolloutPhase,
    ) -> Result<()> {
        if let Some(persistence) = &self.persistence {
            for message in collected.values() {
                persistence
                    .save_rollout(step, &message.task_id, message, phase)
                    .await?;
            }
        }
        Ok(())
    }

    /// Encodes the collected rollouts and groups them by prompt.
    fn collect_round(
        &mut self,
        collected: &BTreeMap<String, RolloutMessage>,
        dropped: &mut Vec<String>,
    ) -> BTreeMap<String, Vec<RolloutWithReward>> {
        let mut grouped: BTreeMap<String, Vec<RolloutWithReward>> = BTreeMap::new();
        for message in collected.values() {
            let turns = match rollout_turns(message) {
                Ok(turns) => turns,
                Err(error) => {
                    dropped.push(error.to_string());
                    continue;
                }
            };
            if turns.is_empty() {
                continue;
            }
            self.turn_counts.push(turns.len());
            let uid = message
                .origin_task_id
                .clone()
                .unwrap_or_else(|| message.task_id.clone());
            self.rewards_by_uid
                .entry(uid.clone())
                .or_default()
                .push(RolloutRewards {
                    global: message.global_reward,
                    per_turn: message.reward_list.clone(),
                });

            let outcome = if self.config.whole_trajectory {
                self.encoder.build_whole_trajectory(message)
            } else {
                self.encoder.build(message)
            };
            if let Some(reason) = &outcome.dropped_reason {
                dropped.push(reason.clone());
                continue;
            }
            grouped.entry(uid).or_default().extend(outcome.samples);
        }
        grouped
    }

    /// Classifies one round's samples and applies the stop validator.
    ///
    /// Returns how many groups the validator accepted in this round.
    fn update_rollout_state(
        &mut self,
        grouped: &BTreeMap<String, Vec<RolloutWithReward>>,
    ) -> Result<usize> {
        let classifier = self.processors.classifier(&self.config.classifier)?;
        let validator = self.processors.validator(&self.config.validator)?;
        let keep = Some(self.config.final_keep_per_prompt);
        let mut finished = 0usize;

        for (uid, samples) in grouped {
            let (positive, negative) = classifier(samples);
            self.total_positive += positive.len();
            self.total_negative += negative.len();
            if !negative.is_empty() {
                self.negative_cache
                    .entry(uid.clone())
                    .or_default()
                    .extend(negative);
            }
            if !positive.is_empty() {
                self.positive_cache
                    .entry(uid.clone())
                    .or_default()
                    .extend(positive);
            }
            let empty = Vec::new();
            let pos = self.positive_cache.get(uid).unwrap_or(&empty);
            let neg = self.negative_cache.get(uid).unwrap_or(&empty);
            let state_entry = self.rollout_state.entry(uid.clone()).or_default();
            state_entry.pos = pos.len();
            state_entry.neg = neg.len();
            state_entry.finished = validator(pos, neg, keep);
            finished += usize::from(state_entry.finished);
        }
        Ok(finished)
    }

    /// Drops the accepted groups, bumping the round of the rest.
    ///
    /// Returns the unfinished tasks and how many were finished.
    fn filter_unfinished_tasks(
        rollout_state: &BTreeMap<String, RolloutState>,
        tasks: BTreeMap<String, RlTask>,
    ) -> (BTreeMap<String, RlTask>, usize) {
        let total = tasks.len();
        let mut unfinished = BTreeMap::new();
        for (task_id, mut task) in tasks {
            let origin = task.origin_task_id.clone().unwrap_or_default();
            let state = rollout_state.get(&origin);
            if state.is_some_and(|state| !state.finished) {
                task.round_num += 1;
                unfinished.insert(task_id, task);
            }
        }
        let finished = total - unfinished.len();
        (unfinished, finished)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use serde_json::json;

    use crate::rl::Rollout;

    use super::super::runtime::ExactMatchComposer;

    /// A source whose answer depends on the task id, so a round can mix
    /// successes and failures deterministically.
    struct ScriptedSource {
        correct_every: usize,
        fail_task: Option<&'static str>,
        turns: usize,
    }

    #[async_trait]
    impl RolloutSource for ScriptedSource {
        fn name(&self) -> &'static str {
            "scripted"
        }

        async fn rollout(&self, task: &RlTask) -> Result<Vec<Rollout>> {
            if self.fail_task == Some(task.task_id.as_str()) {
                return Err(Error::validation("boom"));
            }
            let expected = task
                .task_sample
                .get("expected")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let answer = task
                .task_id
                .rsplit('-')
                .next()
                .and_then(|index| index.parse::<usize>().ok())
                .filter(|index| index % self.correct_every == 0)
                .map_or("wrong", |_| expected);
            Ok((0..self.turns)
                .map(|turn| Rollout {
                    turn_id: turn as u64,
                    input_prompt: json!({"message": [{"role": "user", "content": "q"}], "ground_truth": expected}),
                    output_response: json!(answer),
                    llm_config: json!({}),
                    input_prompt_ids: vec![1, 2],
                    output_response_ids: vec![3],
                })
                .collect())
        }
    }

    fn source(correct_every: usize, turns: usize) -> Arc<dyn RolloutSource> {
        Arc::new(ScriptedSource {
            correct_every,
            fail_task: None,
            turns,
        })
    }

    fn rows(count: usize) -> Vec<Value> {
        (0..count)
            .map(|i| json!({"query": format!("q{i}"), "expected": "42"}))
            .collect()
    }

    fn config() -> CoordinatorConfig {
        CoordinatorConfig {
            runtime_parallel_num: 2,
            rollout_n: 2,
            rollout_max_round: 2,
            max_prompt_length: 8,
            max_response_length: 4,
            round_timeout_ms: 5_000,
            ..CoordinatorConfig::default()
        }
    }

    async fn coordinator() -> TrainingCoordinator {
        TrainingCoordinator::new(config(), source(2, 2))
            .expect("coordinator")
            .with_composer(Arc::new(ExactMatchComposer::new()))
    }

    #[test]
    fn configuration_defaults_and_validation() {
        let defaults = CoordinatorConfig::default();
        assert_eq!(defaults.runtime_parallel_num, 1);
        assert_eq!(defaults.rollout_n, 1);
        assert_eq!(defaults.rollout_max_round, 1);
        assert!(!defaults.whole_trajectory);
        assert_eq!(defaults.final_keep_per_prompt, 8);
        assert_eq!(defaults.max_prompt_length, 1024);
        assert_eq!(defaults.max_response_length, 1024);
        assert_eq!(defaults.pad_token_id, 0);
        assert_eq!(defaults.classifier, DEFAULT_CLASSIFIER);
        assert_eq!(defaults.validator, DEFAULT_VALIDATOR);
        assert_eq!(defaults.sampler, DEFAULT_SAMPLER);
        assert_eq!(defaults.round_timeout_ms, 30_000);
        assert_eq!(defaults.batch_config().max_prompt_length, 1024);
        defaults.validate().expect("valid");

        for broken in [
            CoordinatorConfig {
                runtime_parallel_num: 0,
                ..CoordinatorConfig::default()
            },
            CoordinatorConfig {
                rollout_n: 0,
                ..CoordinatorConfig::default()
            },
            CoordinatorConfig {
                rollout_max_round: 0,
                ..CoordinatorConfig::default()
            },
            CoordinatorConfig {
                final_keep_per_prompt: 0,
                ..CoordinatorConfig::default()
            },
            CoordinatorConfig {
                max_prompt_length: 0,
                ..CoordinatorConfig::default()
            },
            CoordinatorConfig {
                max_response_length: 0,
                ..CoordinatorConfig::default()
            },
        ] {
            assert!(broken.validate().is_err(), "{broken:?} should be rejected");
        }
    }

    #[tokio::test]
    async fn construction_rejects_unknown_processors() {
        let error = TrainingCoordinator::new(
            CoordinatorConfig {
                classifier: "nope".to_owned(),
                ..CoordinatorConfig::default()
            },
            source(1, 1),
        )
        .expect_err("unknown classifier");
        assert!(error.to_string().contains("unknown classifier"));

        let registry = ProcessorsRegistry::new();
        let coordinator = TrainingCoordinator::new(CoordinatorConfig::default(), source(1, 1))
            .expect("coordinator");
        let coordinator = coordinator
            .with_processors(registry)
            .expect("default registry");
        let error = coordinator
            .with_processors(ProcessorsRegistry::default())
            .expect_err("empty registry");
        assert!(error.to_string().contains("unknown classifier"));
    }

    #[tokio::test]
    async fn a_round_loop_collects_encodes_and_plans() {
        let mut coordinator = coordinator().await;
        let report = coordinator.run_rounds(&rows(2), 1).await.expect("rounds");

        assert_eq!(report.step, 1);
        // 2 rows x 2 rollouts x 2 turns = 8 samples; one rollout of each row
        // answers correctly and one does not.
        assert_eq!(report.collected_rollouts, 4);
        assert_eq!(report.training_samples, 8);
        assert_eq!(report.unique_prompts, 2);
        assert_eq!(report.positive_samples, 4);
        assert_eq!(report.negative_samples, 4);
        assert_eq!(report.avg_turns, 2.0);
        assert!((report.reward_mean - 0.5).abs() < 1e-12);
        // Both groups reached the stop rule in the first round, so the loop
        // never submits a second one.
        assert_eq!(report.rounds.len(), 1);
        assert_eq!(report.rounds[0].submitted, 4);
        assert_eq!(report.rounds[0].collected, 4);
        assert_eq!(report.rounds[0].remaining_tasks, 0);
        assert!(report.dropped_rollouts.is_empty(), "nothing should drop");
        assert_eq!(report.diagnostics.sample_count, 8);
        assert_eq!(report.batch.sample_count(), 8);
        assert_eq!(report.batch.config.max_prompt_length, 8);
        assert_eq!(report.rewards_by_uid.len(), 2);
        assert_eq!(report.rewards_by_uid["origin-1-0"].len(), 2);
        assert_eq!(report.rewards_by_uid["origin-1-0"][0].per_turn.len(), 2);

        // The coordinator exposes the same state after the run.
        assert_eq!(coordinator.last_training_sample_count(), 8);
        assert_eq!(coordinator.last_avg_turn_count(), 2.0);
        assert_eq!(coordinator.rewards_by_uid().len(), 2);
        assert!(!coordinator.positive_cache().is_empty());
        assert!(coordinator.queue().is_finished().await);
    }

    #[tokio::test]
    async fn whole_trajectory_mode_produces_one_sample_per_rollout() {
        let mut coordinator = TrainingCoordinator::new(
            CoordinatorConfig {
                whole_trajectory: true,
                ..config()
            },
            source(1, 3),
        )
        .expect("coordinator")
        .with_composer(Arc::new(ExactMatchComposer::new()));
        let report = coordinator.run_rounds(&rows(1), 2).await.expect("rounds");
        // 1 row x 2 rollouts, one sample each.
        assert_eq!(report.collected_rollouts, 2);
        assert_eq!(report.training_samples, 2);
        let batch = report.batch.to_rl_batch("b");
        assert!(batch
            .samples
            .iter()
            .all(|sample| sample.metadata["n_turns"] == json!(3)));
    }

    #[tokio::test]
    async fn empty_rows_are_rejected_and_state_is_reset() {
        let mut coordinator = coordinator().await;
        let error = coordinator
            .run_rounds(&[], 1)
            .await
            .expect_err("empty rows");
        assert!(error.to_string().contains("at least one dataset row"));

        coordinator.run_rounds(&rows(1), 1).await.expect("rounds");
        // A second run starts from a clean state.
        let report = coordinator.run_rounds(&rows(1), 2).await.expect("rounds");
        assert_eq!(report.collected_rollouts, 2);
        assert_eq!(report.rewards_by_uid.len(), 1);
        assert!(report.rewards_by_uid.contains_key("origin-2-0"));
    }

    #[tokio::test]
    async fn a_failing_task_yields_a_dropped_empty_rollout() {
        let mut coordinator = TrainingCoordinator::new(
            CoordinatorConfig {
                rollout_max_round: 1,
                ..config()
            },
            Arc::new(ScriptedSource {
                correct_every: 1,
                fail_task: Some("task-1-0-0"),
                turns: 1,
            }),
        )
        .expect("coordinator")
        .with_composer(Arc::new(ExactMatchComposer::new()));
        let report = coordinator.run_rounds(&rows(1), 1).await.expect("rounds");
        // The failed task still reports a rollout, but an empty one has no turn
        // to encode, so only the successful task's turn becomes a sample.
        assert_eq!(report.collected_rollouts, 2);
        assert_eq!(report.training_samples, 1);
        assert_eq!(report.rewards_by_uid["origin-1-0"].len(), 1);
        assert_eq!(report.rewards_by_uid["origin-1-0"][0].global, 1.0);
    }

    #[tokio::test]
    async fn malformed_rollout_info_is_reported_not_swallowed() {
        struct BadSource;
        #[async_trait]
        impl RolloutSource for BadSource {
            async fn rollout(&self, _task: &RlTask) -> Result<Vec<Rollout>> {
                Ok(Vec::new())
            }
        }
        let mut coordinator = TrainingCoordinator::new(
            CoordinatorConfig {
                rollout_n: 1,
                ..config()
            },
            Arc::new(BadSource),
        )
        .expect("coordinator");
        // Empty turns are collected but produce no samples, so the batch is empty.
        let error = coordinator
            .run_rounds(&rows(1), 1)
            .await
            .expect_err("empty batch");
        assert!(error
            .to_string()
            .contains("0 samples collected after rollout"));
    }

    #[tokio::test]
    async fn validation_reports_the_python_metric_keys() {
        let mut coordinator = coordinator().await;
        let metrics = coordinator.validate(&rows(2), 3).await.expect("validate");
        // Four... two rows, two rollouts: the first row answers correctly.
        assert_eq!(metrics.sample_count, 2);
        assert_eq!(metrics.correct_count, 1);
        assert!((metrics.accuracy - 0.5).abs() < 1e-12);
        assert!((metrics.global_reward_mean - 0.5).abs() < 1e-12);
        assert_eq!(metrics.average_turn_num, 2.0);
        assert_eq!(metrics.turn_num, vec![2, 2]);
        assert_eq!(metrics.reward_list.len(), 2);

        let rendered = metrics.to_metrics();
        for key in [
            "val/global_reward_mean",
            "val/accuracy",
            "val/correct_count",
            "val/sample_count",
            "val/average_turn_num",
            "val/turn_num",
            "val/reward_list",
        ] {
            assert!(rendered.contains_key(key), "missing {key}");
        }
        assert_eq!(rendered["val/sample_count"], json!(2));

        // An empty validation set yields zero metrics instead of failing.
        let empty = coordinator
            .validate(&[], 4)
            .await
            .expect("empty validation");
        assert_eq!(empty, ValidationMetrics::default());
    }

    #[tokio::test]
    async fn caches_merge_positives_first() {
        fn rollout(reward: f64) -> RolloutWithReward {
            RolloutWithReward {
                turn_id: 0,
                task_id: "t".to_owned(),
                rollout_id: "r".to_owned(),
                input_prompt_ids: vec![1],
                output_response_ids: vec![2],
                reward,
                n_turns: 1,
                loss_mask: Vec::new(),
                group_id: None,
            }
        }
        let pos = BTreeMap::from([("a".to_owned(), vec![rollout(1.0)])]);
        let neg = BTreeMap::from([
            ("a".to_owned(), vec![rollout(0.0)]),
            ("b".to_owned(), vec![rollout(0.0)]),
        ]);
        let merged = TrainingCoordinator::merge_caches(&pos, &neg);
        assert_eq!(merged.len(), 2);
        assert_eq!(merged["a"].len(), 2);
        assert_eq!(merged["a"][0].reward, 1.0);
        assert_eq!(merged["a"][1].reward, 0.0);
        assert_eq!(merged["b"].len(), 1);
    }

    #[tokio::test]
    async fn persistence_receives_every_collected_rollout() {
        use super::super::store::FileRolloutStore;
        let temp = tempfile::TempDir::new().expect("temp dir");
        let store = Arc::new(
            FileRolloutStore::open(temp.path(), 10)
                .await
                .expect("store"),
        );
        let mut coordinator = TrainingCoordinator::new(
            CoordinatorConfig {
                rollout_n: 1,
                rollout_max_round: 1,
                ..config()
            },
            source(1, 1),
        )
        .expect("coordinator")
        .with_composer(Arc::new(ExactMatchComposer::new()))
        .with_persistence(store.clone());

        let report = coordinator.run_rounds(&rows(2), 5).await.expect("rounds");
        assert_eq!(report.collected_rollouts, 2);
        let stored = store
            .query_rollouts(&BTreeMap::new(), 0)
            .await
            .expect("query");
        assert_eq!(stored.len(), 2);
        assert!(stored.iter().all(|doc| doc["step"] == json!(5)));

        coordinator.validate(&rows(1), 6).await.expect("validate");
        let rolls = store
            .query_rollouts(&BTreeMap::new(), 0)
            .await
            .expect("query");
        // The validation rollout landed in the val directory.
        assert_eq!(rolls.len(), 3);
        let val = store.rollout_path(RolloutPhase::Val, 6);
        assert!(val.exists());
    }
}
