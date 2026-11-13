//! Reward normalisation and rollout bookkeeping.
//!
//! GRPO-style training compares rollouts of the same prompt, so rewards are
//! normalised per group before they reach the policy update. The code here is
//! pure arithmetic and data: deterministic, NaN-free, and independent of any
//! training stack.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::batch::RlSample;
use super::task::RolloutWithReward;

/// Group-relative reward normalisation applied before a policy update.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RewardNormalisation {
    /// Leave rewards unchanged.
    None,
    /// Centre each group on its mean and divide by its population standard deviation.
    GroupZScore,
    /// Subtract each group's mean from its rewards.
    #[default]
    GroupCentered,
    /// Rescale each group linearly into `0.0..=1.0`.
    MinMax,
}

/// Summary statistics of the rewards in one group.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RewardGroupStats {
    /// Number of rewards in the group.
    pub count: usize,
    /// Mean of the normalised rewards.
    pub mean: f64,
    /// Minimum normalised reward.
    pub min: f64,
    /// Maximum normalised reward.
    pub max: f64,
}

/// Summary statistics over all normalised rewards of a epoch.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RewardStats {
    /// Number of normalised rewards.
    pub count: usize,
    /// Mean of the normalised rewards.
    pub mean: f64,
    /// Minimum normalised reward (`0.0` when empty).
    pub min: f64,
    /// Maximum normalised reward (`0.0` when empty).
    pub max: f64,
    /// Population standard deviation of the normalised rewards.
    pub std_dev: f64,
    /// Per-group statistics keyed by group id; rollouts without a group id share the empty-string group.
    pub groups: BTreeMap<String, RewardGroupStats>,
}

/// Count, mean, min, max, and population standard deviation of `values`.
fn summarise(values: &[f64]) -> (usize, f64, f64, f64, f64) {
    if values.is_empty() {
        return (0, 0.0, 0.0, 0.0, 0.0);
    }
    let count = values.len();
    let mean = values.iter().sum::<f64>() / count as f64;
    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let variance = values
        .iter()
        .map(|value| {
            let diff = value - mean;
            diff * diff
        })
        .sum::<f64>()
        / count as f64;
    (count, mean, min, max, variance.sqrt())
}

/// Group-relative reward normaliser with optional clipping.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RewardNormaliser {
    /// Normalisation mode.
    pub mode: RewardNormalisation,
    /// Optional inclusive `(min, max)` clip applied after normalisation.
    pub clip: Option<(f64, f64)>,
}

impl RewardNormaliser {
    /// Creates a normaliser for `mode` without clipping.
    #[must_use]
    pub fn new(mode: RewardNormalisation) -> Self {
        Self { mode, clip: None }
    }

    /// Clips rewards into the inclusive range after normalisation.
    ///
    /// The bounds are ordered defensively, so passing them in either order
    /// produces the same clip.
    #[must_use]
    pub fn with_clip(mut self, low: f64, high: f64) -> Self {
        let (low, high) = if low <= high {
            (low, high)
        } else {
            (high, low)
        };
        self.clip = Some((low, high));
        self
    }

    /// Normalises `rollouts` in place and returns the resulting statistics.
    ///
    /// Rollouts are grouped by [`RolloutWithReward::group_id`]; absent ids share
    /// the empty-string group. Non-finite rewards are treated as `0.0` before
    /// normalisation so the returned statistics stay finite. Statistics are
    /// computed over the normalised (and clipped) rewards.
    pub fn normalise(&self, rollouts: &mut [RolloutWithReward]) -> RewardStats {
        for rollout in rollouts.iter_mut() {
            if !rollout.reward.is_finite() {
                rollout.reward = 0.0;
            }
        }
        let groups = collect_groups(rollouts);
        for indices in groups.values() {
            self.normalise_group(rollouts, indices);
        }
        if let Some((low, high)) = self.clip {
            for rollout in rollouts.iter_mut() {
                rollout.reward = rollout.reward.clamp(low, high);
            }
        }
        Self::stats(rollouts)
    }

    fn normalise_group(&self, rollouts: &mut [RolloutWithReward], indices: &[usize]) {
        let rewards: Vec<f64> = indices
            .iter()
            .map(|&index| rollouts[index].reward)
            .collect();
        let (_, mean, min, max, std_dev) = summarise(&rewards);
        match self.mode {
            RewardNormalisation::None => {}
            RewardNormalisation::GroupZScore => {
                for &index in indices {
                    rollouts[index].reward = if std_dev > 0.0 {
                        (rollouts[index].reward - mean) / std_dev
                    } else {
                        0.0
                    };
                }
            }
            RewardNormalisation::GroupCentered => {
                for &index in indices {
                    rollouts[index].reward -= mean;
                }
            }
            RewardNormalisation::MinMax => {
                let range = max - min;
                for &index in indices {
                    rollouts[index].reward = if range > 0.0 {
                        (rollouts[index].reward - min) / range
                    } else {
                        0.0
                    };
                }
            }
        }
    }

    fn stats(rollouts: &[RolloutWithReward]) -> RewardStats {
        let rewards: Vec<f64> = rollouts.iter().map(|rollout| rollout.reward).collect();
        let (count, mean, min, max, std_dev) = summarise(&rewards);
        let mut groups: BTreeMap<String, RewardGroupStats> = BTreeMap::new();
        for (key, indices) in collect_groups(rollouts) {
            let values: Vec<f64> = indices
                .iter()
                .map(|&index| rollouts[index].reward)
                .collect();
            let (count, mean, min, max, _) = summarise(&values);
            groups.insert(
                key,
                RewardGroupStats {
                    count,
                    mean,
                    min,
                    max,
                },
            );
        }
        RewardStats {
            count,
            mean,
            min,
            max,
            std_dev,
            groups,
        }
    }
}

fn collect_groups(rollouts: &[RolloutWithReward]) -> BTreeMap<String, Vec<usize>> {
    let mut groups: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (index, rollout) in rollouts.iter().enumerate() {
        let key = rollout.group_id.clone().unwrap_or_default();
        groups.entry(key).or_default().push(index);
    }
    groups
}

fn render_tokens(ids: &[u32]) -> String {
    ids.iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(" ")
}

/// The rollouts produced for one task.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RolloutLog {
    /// Identifier of the task the rollouts belong to.
    pub task_id: String,
    /// Reward-annotated turns in deterministic order.
    pub rollouts: Vec<RolloutWithReward>,
}

impl RolloutLog {
    /// Creates an empty log for `task_id`.
    #[must_use]
    pub fn new(task_id: impl Into<String>) -> Self {
        Self {
            task_id: task_id.into(),
            rollouts: Vec::new(),
        }
    }

    /// Creates a log over `rollouts` for `task_id`.
    #[must_use]
    pub fn with_rollouts(task_id: impl Into<String>, rollouts: Vec<RolloutWithReward>) -> Self {
        Self {
            task_id: task_id.into(),
            rollouts,
        }
    }

    /// Appends one rollout.
    pub fn push(&mut self, rollout: RolloutWithReward) {
        self.rollouts.push(rollout);
    }

    /// Number of rollouts in the log.
    #[must_use]
    pub fn len(&self) -> usize {
        self.rollouts.len()
    }

    /// Whether the log carries no rollouts.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rollouts.is_empty()
    }

    /// Mean reward over the rollouts; `0.0` for an empty log.
    #[must_use]
    pub fn mean_reward(&self) -> f64 {
        if self.rollouts.is_empty() {
            return 0.0;
        }
        let total: f64 = self
            .rollouts
            .iter()
            .map(|rollout| {
                if rollout.reward.is_finite() {
                    rollout.reward
                } else {
                    0.0
                }
            })
            .sum();
        total / self.rollouts.len() as f64
    }

    /// Projects the rollouts onto packable samples.
    ///
    /// [`RolloutWithReward`] carries token ids rather than text (Python parity),
    /// so prompts and responses are rendered as space-separated decimal ids. An
    /// empty prompt falls back to the log's task id so the produced sample stays
    /// valid. Group ids and rewards are preserved, and non-finite rewards become
    /// `0.0`.
    #[must_use]
    pub fn to_samples(&self) -> Vec<RlSample> {
        self.rollouts
            .iter()
            .enumerate()
            .map(|(index, rollout)| {
                let mut prompt = render_tokens(&rollout.input_prompt_ids);
                if prompt.trim().is_empty() {
                    prompt.clone_from(&self.task_id);
                }
                let reward = if rollout.reward.is_finite() {
                    rollout.reward
                } else {
                    0.0
                };
                let metadata = BTreeMap::from([
                    ("turn_id".to_owned(), Value::from(rollout.turn_id)),
                    ("n_turns".to_owned(), Value::from(rollout.n_turns)),
                    (
                        "rollout_id".to_owned(),
                        Value::String(rollout.rollout_id.clone()),
                    ),
                ]);
                RlSample {
                    sample_id: format!("{}-{}-{index}", rollout.rollout_id, rollout.turn_id),
                    prompt,
                    response: render_tokens(&rollout.output_response_ids),
                    reward,
                    group_id: rollout.group_id.clone(),
                    metadata,
                }
            })
            .collect()
    }
}
