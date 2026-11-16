//! Training metrics and batch diagnostics (Python `store/metrics_tracker.py`).
//!
//! Python wraps verl's `Tracking` to log to TensorBoard/W&B and hides a pile of
//! tensor dumps behind `TrainingDiagnostics`. The backends and the tensors are
//! not portable, so this module keeps the *content*: typed per-step records, a
//! per-step history, the reward-collapse counter, and deterministic scalar
//! summaries of a batch.

use std::collections::BTreeMap;
use std::sync::Mutex;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::batch::BatchPlan;
use super::mean;

/// How many consecutive zero-reward steps trigger a collapse warning.
pub const COLLAPSE_WARNING_STEPS: usize = 3;

/// Count, mean, minimum and maximum of one numeric series.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SeriesSummary {
    /// Number of values summarised.
    pub count: usize,
    /// Minimum value; `0.0` when empty.
    pub min: f64,
    /// Maximum value; `0.0` when empty.
    pub max: f64,
    /// Arithmetic mean; `0.0` when empty.
    pub mean: f64,
}

impl SeriesSummary {
    /// Summarises `values`, ignoring non-finite entries.
    #[must_use]
    pub fn from_values(values: impl IntoIterator<Item = f64>) -> Self {
        let mut summary = Self::default();
        let mut total = 0.0;
        for value in values {
            if !value.is_finite() {
                continue;
            }
            if summary.count == 0 {
                summary.min = value;
                summary.max = value;
            } else {
                summary.min = summary.min.min(value);
                summary.max = summary.max.max(value);
            }
            summary.count += 1;
            total += value;
        }
        if summary.count > 0 {
            summary.mean = total / summary.count as f64;
        }
        summary
    }

    /// Adds the summary's fields to `metrics` under `prefix`.
    pub fn insert_into(&self, metrics: &mut BTreeMap<String, f64>, prefix: &str) {
        metrics.insert(format!("{prefix}_count"), self.count as f64);
        metrics.insert(format!("{prefix}_min"), self.min);
        metrics.insert(format!("{prefix}_max"), self.max);
        metrics.insert(format!("{prefix}_mean"), self.mean);
    }
}

/// Deterministic scalar diagnostics of one [`BatchPlan`].
///
/// Everything here is derived from the plan's integer and float fields: no
/// tensor is materialised, so the values are identical on every run and can be
/// asserted in tests.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct BatchDiagnostics {
    /// Number of samples in the batch.
    pub sample_count: usize,
    /// Number of distinct rollout groups (`data_ids`).
    pub unique_data_ids: usize,
    /// Prompt lengths (attended prompt tokens per sample).
    pub prompt_tokens: SeriesSummary,
    /// Response lengths (attended response tokens per sample).
    pub response_tokens: SeriesSummary,
    /// Sample rewards.
    pub rewards: SeriesSummary,
    /// Samples whose prompt exceeded `max_prompt_length`.
    pub drop_count: usize,
    /// Samples whose response was truncated.
    pub truncated_count: usize,
    /// Trainable response tokens (loss mask `1`, or every attended token when
    /// the sample carries no mask).
    pub trainable_token_count: usize,
    /// Masked response tokens (loss mask `0`).
    pub masked_token_count: usize,
    /// Mean `n_turns` over the samples.
    pub mean_n_turns: f64,
}

impl BatchDiagnostics {
    /// Summarises `plan`.
    #[must_use]
    pub fn from_plan(plan: &BatchPlan) -> Self {
        let components = &plan.components;
        let mut groups = std::collections::BTreeSet::new();
        let mut trainable = 0usize;
        let mut masked = 0usize;
        for (index, _mask) in plan.attention_mask.iter().enumerate() {
            let response_len = components
                .response_attention_mask
                .get(index)
                .map_or(0, |row| row.iter().filter(|value| **value != 0).count());
            let loss_mask = components.loss_masks.get(index).and_then(Option::as_ref);
            match loss_mask {
                Some(losses) => {
                    let attended = &components.response_attention_mask[index];
                    for (position, loss) in losses.iter().enumerate() {
                        if attended.get(position).copied().unwrap_or(0) == 0 {
                            continue;
                        }
                        if *loss == 0 {
                            masked += 1;
                        } else {
                            trainable += 1;
                        }
                    }
                }
                None => trainable += response_len,
            }
            if let Some(data_id) = components.data_ids.get(index) {
                groups.insert(data_id.clone());
            }
        }

        Self {
            sample_count: plan.sample_count(),
            unique_data_ids: groups.len(),
            prompt_tokens: SeriesSummary::from_values(
                components
                    .input_attention_mask
                    .iter()
                    .map(|row| row.iter().filter(|value| **value != 0).count() as f64),
            ),
            response_tokens: SeriesSummary::from_values(
                components
                    .response_attention_mask
                    .iter()
                    .map(|row| row.iter().filter(|value| **value != 0).count() as f64),
            ),
            rewards: SeriesSummary::from_values(components.rewards.iter().copied()),
            drop_count: components.drop_count(),
            truncated_count: components.truncation_count,
            trainable_token_count: trainable,
            masked_token_count: masked,
            mean_n_turns: mean(components.n_turns_list.iter().map(|turns| *turns as f64)),
        }
    }

    /// Renders the diagnostics as scalar metrics with a `batch/` prefix.
    #[must_use]
    pub fn to_metrics(&self) -> BTreeMap<String, f64> {
        let mut metrics = BTreeMap::new();
        metrics.insert("batch/sample_count".to_owned(), self.sample_count as f64);
        metrics.insert(
            "batch/unique_data_ids".to_owned(),
            self.unique_data_ids as f64,
        );
        metrics.insert("batch/drop_count".to_owned(), self.drop_count as f64);
        metrics.insert(
            "batch/truncated_count".to_owned(),
            self.truncated_count as f64,
        );
        metrics.insert(
            "batch/trainable_token_count".to_owned(),
            self.trainable_token_count as f64,
        );
        metrics.insert(
            "batch/masked_token_count".to_owned(),
            self.masked_token_count as f64,
        );
        metrics.insert("batch/mean_n_turns".to_owned(), self.mean_n_turns);
        self.prompt_tokens
            .insert_into(&mut metrics, "batch/prompt_tokens");
        self.response_tokens
            .insert_into(&mut metrics, "batch/response_tokens");
        self.rewards.insert_into(&mut metrics, "batch/reward");
        metrics
    }
}

/// One logged training step.
///
/// Field names follow Python's `TrainingStepMetrics`; `verl_metrics` stays a
/// free-form map so callers can forward whatever the policy reported without
/// this crate depending on verl's key set.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TrainingStepMetrics {
    /// The global training step.
    pub step: usize,
    /// The epoch the step belongs to (zero-based).
    pub epoch: usize,
    /// The scalar metrics of the step.
    pub verl_metrics: BTreeMap<String, Value>,
    /// Mean number of turns per conversation.
    pub avg_turns: f64,
    /// Mean rollout reward.
    pub reward_mean: f64,
    /// Consecutive steps with `reward_mean <= 0.0`, as counted by the tracker.
    pub consecutive_zero_reward_steps: usize,
}

impl TrainingStepMetrics {
    /// Renders the record as scalar metrics, prefixed like Python's logger.
    ///
    /// `verl_metrics` entries that are numbers are included; nested values are
    /// skipped because a metrics store takes scalars.
    #[must_use]
    pub fn to_metrics(&self) -> BTreeMap<String, f64> {
        let mut metrics = BTreeMap::new();
        for (key, value) in &self.verl_metrics {
            if let Some(number) = value.as_f64() {
                metrics.insert(key.clone(), number);
            }
        }
        metrics.insert("training/global_step".to_owned(), self.step as f64);
        metrics.insert("training/epoch".to_owned(), self.epoch as f64);
        metrics.insert(
            "training/avg_conversation_turns".to_owned(),
            self.avg_turns,
        );
        metrics.insert("training/rollout_reward_mean".to_owned(), self.reward_mean);
        metrics.insert(
            "training/consecutive_zero_reward_steps".to_owned(),
            self.consecutive_zero_reward_steps as f64,
        );
        metrics
    }
}

/// Collects training-step metrics in memory.
///
/// Python forwards to verl's `Tracking` (TensorBoard, W&B, MLflow, ...). Those
/// backends need network access and vendor SDKs, so this tracker *is* the
/// backend: it keeps the history and raises the reward-collapse warning Python
/// only logs. Read the history with [`Self::training_history`] and
/// [`Self::validation_history`].
#[derive(Debug)]
pub struct MetricsTracker {
    history: Mutex<Vec<TrainingStepMetrics>>,
    validations: Mutex<Vec<(usize, BTreeMap<String, Value>)>>,
    consecutive_zero_reward_steps: Mutex<usize>,
    collapse_warnings: Mutex<usize>,
    collapse_threshold: usize,
}

impl Default for MetricsTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricsTracker {
    /// Creates a tracker warning after [`COLLAPSE_WARNING_STEPS`] zero-reward steps.
    #[must_use]
    pub fn new() -> Self {
        Self::with_collapse_threshold(COLLAPSE_WARNING_STEPS)
    }

    /// Creates a tracker with an explicit collapse threshold (at least one).
    #[must_use]
    pub fn with_collapse_threshold(threshold: usize) -> Self {
        Self {
            history: Mutex::new(Vec::new()),
            validations: Mutex::new(Vec::new()),
            consecutive_zero_reward_steps: Mutex::new(0),
            collapse_warnings: Mutex::new(0),
            collapse_threshold: threshold.max(1),
        }
    }

    /// The collapse threshold.
    #[must_use]
    pub fn collapse_threshold(&self) -> usize {
        self.collapse_threshold
    }

    /// Records and returns one step.
    ///
    /// The consecutive zero-reward counter is owned by the tracker: it is
    /// incremented when `reward_mean <= 0.0`, reset otherwise, written back into
    /// the record, and the number of raised warnings is incremented once per
    /// step at or beyond the threshold (Python warns on every such step).
    pub fn record_step(
        &self,
        step: usize,
        epoch: usize,
        avg_turns: f64,
        reward_mean: f64,
        verl_metrics: BTreeMap<String, Value>,
    ) -> TrainingStepMetrics {
        self.log_training_step(TrainingStepMetrics {
            step,
            epoch,
            verl_metrics,
            avg_turns,
            reward_mean,
            consecutive_zero_reward_steps: 0,
        })
    }

    /// Records a pre-built step, normalising its collapse counter.
    ///
    /// Returns the stored record, whose `consecutive_zero_reward_steps` is the
    /// tracker's count rather than the caller's.
    pub fn log_training_step(&self, mut metrics: TrainingStepMetrics) -> TrainingStepMetrics {
        let mut consecutive = lock(&self.consecutive_zero_reward_steps);
        if metrics.reward_mean <= 0.0 || !metrics.reward_mean.is_finite() {
            *consecutive += 1;
        } else {
            *consecutive = 0;
        }
        metrics.consecutive_zero_reward_steps = *consecutive;
        if *consecutive >= self.collapse_threshold {
            *lock(&self.collapse_warnings) += 1;
        }
        drop(consecutive);
        lock(&self.history).push(metrics.clone());
        metrics
    }

    /// Records one validation pass.
    pub fn log_validation(&self, step: usize, metrics: BTreeMap<String, Value>) {
        lock(&self.validations).push((step, metrics));
    }

    /// The recorded training steps, in recording order.
    #[must_use]
    pub fn training_history(&self) -> Vec<TrainingStepMetrics> {
        lock(&self.history).clone()
    }

    /// The recorded validation passes, in recording order.
    #[must_use]
    pub fn validation_history(&self) -> Vec<(usize, BTreeMap<String, Value>)> {
        lock(&self.validations).clone()
    }

    /// The current consecutive zero-reward count.
    #[must_use]
    pub fn consecutive_zero_reward_steps(&self) -> usize {
        *lock(&self.consecutive_zero_reward_steps)
    }

    /// How many collapse warnings were raised.
    #[must_use]
    pub fn collapse_warnings(&self) -> usize {
        *lock(&self.collapse_warnings)
    }

    /// The last recorded step, if any.
    #[must_use]
    pub fn last_step(&self) -> Option<TrainingStepMetrics> {
        lock(&self.history).last().cloned()
    }

    /// Number of recorded training steps.
    #[must_use]
    pub fn step_count(&self) -> usize {
        lock(&self.history).len()
    }
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    use crate::rl::RolloutWithReward;

    fn plan() -> BatchPlan {
        let builder = super::super::BatchBuilder::new(super::super::BatchConfig::new(3, 2, 0));
        let masked = RolloutWithReward {
            turn_id: 0,
            task_id: "t".to_owned(),
            rollout_id: "r".to_owned(),
            input_prompt_ids: vec![1, 2],
            output_response_ids: vec![3],
            reward: 1.0,
            n_turns: 2,
            loss_mask: vec![1, 0],
            group_id: None,
        };
        let rollouts = BTreeMap::from([
            ("a".to_owned(), vec![masked]),
            (
                "b".to_owned(),
                vec![RolloutWithReward {
                    turn_id: 1,
                    task_id: "t".to_owned(),
                    rollout_id: "r2".to_owned(),
                    input_prompt_ids: vec![1, 2, 3, 4],
                    output_response_ids: vec![5],
                    reward: 0.0,
                    n_turns: 1,
                    loss_mask: Vec::new(),
                    group_id: None,
                }],
            ),
        ]);
        builder.assemble_batch(&rollouts).expect("plan")
    }

    #[test]
    fn series_summaries_ignore_non_finite_values() {
        let summary = SeriesSummary::from_values([1.0, 3.0, f64::NAN]);
        assert_eq!(summary.count, 2);
        assert_eq!(summary.min, 1.0);
        assert_eq!(summary.max, 3.0);
        assert_eq!(summary.mean, 2.0);

        let empty = SeriesSummary::from_values(std::iter::empty());
        assert_eq!(empty, SeriesSummary::default());

        let mut metrics = BTreeMap::new();
        summary.insert_into(&mut metrics, "x");
        assert_eq!(metrics["x_count"], 2.0);
        assert_eq!(metrics["x_mean"], 2.0);
    }

    #[test]
    fn batch_diagnostics_are_derived_from_the_plan() {
        let diagnostics = BatchDiagnostics::from_plan(&plan());
        assert_eq!(diagnostics.sample_count, 2);
        assert_eq!(diagnostics.unique_data_ids, 2);
        assert_eq!(diagnostics.drop_count, 1);
        assert_eq!(diagnostics.truncated_count, 0);
        assert_eq!(diagnostics.rewards.count, 2);
        assert_eq!(diagnostics.rewards.max, 1.0);
        assert_eq!(diagnostics.rewards.mean, 0.5);
        // Sample a: one attended response token, mask [1, 0] -> 1 trainable.
        // Sample b: one attended response token without a mask -> 1 trainable.
        assert_eq!(diagnostics.trainable_token_count, 2);
        assert_eq!(diagnostics.masked_token_count, 1);
        assert_eq!(diagnostics.mean_n_turns, 1.5);
        assert_eq!(diagnostics.prompt_tokens.max, 3.0);
        assert_eq!(diagnostics.response_tokens.max, 1.0);

        let metrics = diagnostics.to_metrics();
        assert_eq!(metrics["batch/sample_count"], 2.0);
        assert_eq!(metrics["batch/drop_count"], 1.0);
        assert_eq!(metrics["batch/reward_mean"], 0.5);
        assert_eq!(metrics["batch/prompt_tokens_max"], 3.0);
    }

    #[test]
    fn recordings_keep_history_and_the_collapse_counter() {
        let tracker = MetricsTracker::new();
        assert_eq!(tracker.collapse_threshold(), 3);
        assert_eq!(tracker.step_count(), 0);
        assert!(tracker.last_step().is_none());

        let first = tracker.record_step(1, 0, 2.0, 0.0, BTreeMap::new());
        assert_eq!(first.consecutive_zero_reward_steps, 1);
        assert_eq!(tracker.collapse_warnings(), 0);

        let second = tracker.record_step(2, 0, 2.5, -0.5, BTreeMap::new());
        assert_eq!(second.consecutive_zero_reward_steps, 2);
        assert_eq!(tracker.collapse_warnings(), 0);

        let mut metrics = BTreeMap::new();
        metrics.insert("rollout/reward_mean".to_owned(), json!(0.0));
        let third = tracker.record_step(3, 0, 3.0, 0.0, metrics);
        assert_eq!(third.consecutive_zero_reward_steps, 3);
        assert_eq!(tracker.collapse_warnings(), 1);

        // A positive reward resets the counter but the warning count stays.
        let fourth = tracker.record_step(4, 1, 1.0, 0.5, BTreeMap::new());
        assert_eq!(fourth.consecutive_zero_reward_steps, 0);
        assert_eq!(tracker.collapse_warnings(), 1);
        assert_eq!(tracker.consecutive_zero_reward_steps(), 0);
        assert_eq!(tracker.step_count(), 4);
        assert_eq!(tracker.last_step().expect("last").step, 4);

        let history = tracker.training_history();
        assert_eq!(history.len(), 4);
        assert_eq!(history[2].verl_metrics["rollout/reward_mean"], json!(0.0));
    }

    #[test]
    fn a_custom_threshold_and_manual_records_are_respected() {
        let tracker = MetricsTracker::with_collapse_threshold(1);
        assert_eq!(tracker.collapse_threshold(), 1);
        let record = tracker.log_training_step(TrainingStepMetrics {
            step: 9,
            epoch: 2,
            verl_metrics: BTreeMap::from([("a".to_owned(), json!(1.5))]),
            avg_turns: 1.25,
            reward_mean: 0.0,
            consecutive_zero_reward_steps: 99,
        });
        assert_eq!(record.consecutive_zero_reward_steps, 1);
        assert_eq!(tracker.collapse_warnings(), 1);

        // A tracker built with zero still warns on the first step.
        assert_eq!(
            MetricsTracker::with_collapse_threshold(0).collapse_threshold(),
            1
        );
    }

    #[test]
    fn validation_history_and_metric_rendering() {
        let tracker = MetricsTracker::new();
        let mut validation = BTreeMap::new();
        validation.insert("val/accuracy".to_owned(), json!(0.5));
        tracker.log_validation(3, validation);
        let history = tracker.validation_history();
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].0, 3);
        assert_eq!(history[0].1["val/accuracy"], json!(0.5));

        let step = TrainingStepMetrics {
            step: 7,
            epoch: 1,
            verl_metrics: BTreeMap::from([
                ("policy/samples".to_owned(), json!(4)),
                ("policy/best".to_owned(), json!(1.0)),
                ("nested".to_owned(), json!({"a": 1})),
            ]),
            avg_turns: 2.0,
            reward_mean: 0.25,
            consecutive_zero_reward_steps: 0,
        };
        let metrics = step.to_metrics();
        assert_eq!(metrics["training/global_step"], 7.0);
        assert_eq!(metrics["training/epoch"], 1.0);
        assert_eq!(metrics["training/avg_conversation_turns"], 2.0);
        assert_eq!(metrics["training/rollout_reward_mean"], 0.25);
        assert_eq!(metrics["policy/samples"], 4.0);
        assert_eq!(metrics["policy/best"], 1.0);
        assert!(!metrics.contains_key("nested"));
    }

    #[test]
    fn records_round_trip_through_json() {
        let step = TrainingStepMetrics {
            step: 1,
            epoch: 0,
            verl_metrics: BTreeMap::from([("a".to_owned(), json!(1))]),
            avg_turns: 1.5,
            reward_mean: 0.5,
            consecutive_zero_reward_steps: 2,
        };
        let json = serde_json::to_string(&step).expect("serialize");
        let decoded: TrainingStepMetrics = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(decoded, step);
        // Unknown and missing fields are tolerated.
        let partial: TrainingStepMetrics =
            serde_json::from_str(r#"{"step": 3, "future": true}"#).expect("partial");
        assert_eq!(partial.step, 3);
        assert_eq!(partial.epoch, 0);
    }
}
