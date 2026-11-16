//! The RL parameter-update pipeline: advantages, gradient application, checkpoints.
//!
//! Python applies PPO/`verl` gradients inside the training stack. That stack is
//! deferred (see [`crate::deferred`]), so this module keeps the portable half of
//! the loop — the half a caller can test offline:
//!
//! - [`compute_advantages`] turns reward-annotated rollouts into group-relative
//!   advantages (GRPO's `adv` column), with a scalar-normalisation fallback for
//!   rollouts that carry no group.
//! - [`GradientOptimizer`] is the gradient seam: [`OfflinePolicyOptimizer`] drives a
//!   [`Policy`] deterministically, while [`GpuGradientOptimizer`] is the typed
//!   [`openjiuwen_common::error::Error::Unsupported`] for the `torch`/`verl`/`ray`
//!   optimiser.
//! - [`RlUpdatePipeline`] packs and validates a batch, asks the optimizer for
//!   updates, applies them through [`crate::updater::UpdateTarget`] in the order
//!   the optimizer wrote them (`None` values last), and emits an
//!   [`RlCheckpoint`] describing the applied updates and advantages.
//!
//! Everything is deterministic: no clock, no randomness, and no network.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::json;

use openjiuwen_common::error::{Error, Result};

use crate::rl::{
    BatchPacker, Policy, RewardNormalisation, RewardNormaliser, RlBatch, RlCheckpoint,
    RolloutWithReward,
};

use super::apply::{execute_updates, summarize_apply_results, OperatorMap, UpdateTarget};
use super::types::{
    ApplyResult, ApplySummary, Update, UpdateValue, UpdateKey, Updates,
};

/// The target a policy update is written to.
pub const POLICY_TARGET: &str = "policy";

/// Per-group advantage statistics.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AdvantageGroupStats {
    /// Number of samples in the group.
    pub count: usize,
    /// Mean reward of the group before the advantage replaced it.
    pub mean_reward: f64,
    /// Minimum reward of the group.
    pub min_reward: f64,
    /// Maximum reward of the group.
    pub max_reward: f64,
    /// Mean advantage of the group.
    pub mean_advantage: f64,
    /// Minimum advantage of the group.
    pub min_advantage: f64,
    /// Maximum advantage of the group.
    pub max_advantage: f64,
    /// Whether the group fell back to scalar normalisation (no group id).
    pub fallback: bool,
}

/// The advantages of one rollout batch, aligned with the input order.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AdvantageReport {
    /// The raw rewards as observed, before the advantage replaced them.
    pub rewards: Vec<f64>,
    /// The advantages, aligned with [`Self::rewards`].
    pub advantages: Vec<f64>,
    /// Per-group statistics, keyed by group id; ungrouped samples share the empty key.
    pub groups: BTreeMap<String, AdvantageGroupStats>,
    /// The group ids that used the scalar-normalisation fallback, sorted.
    pub fallback_groups: Vec<String>,
}

impl AdvantageReport {
    /// Number of samples the report covers.
    #[must_use]
    pub fn len(&self) -> usize {
        self.advantages.len()
    }

    /// Whether the report covers no sample.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.advantages.is_empty()
    }

    /// Mean advantage; `0.0` when empty.
    #[must_use]
    pub fn mean_advantage(&self) -> f64 {
        mean(&self.advantages)
    }

    /// Mean raw reward; `0.0` when empty.
    #[must_use]
    pub fn mean_reward(&self) -> f64 {
        mean(&self.rewards)
    }

    /// The largest raw reward; `0.0` when empty.
    #[must_use]
    pub fn best_reward(&self) -> f64 {
        max(&self.rewards)
    }

    /// The advantage of one sample, by index.
    #[must_use]
    pub fn advantage(&self, index: usize) -> Option<f64> {
        self.advantages.get(index).copied()
    }

    /// The statistics of one group.
    #[must_use]
    pub fn group(&self, group_id: &str) -> Option<&AdvantageGroupStats> {
        self.groups.get(group_id)
    }
}

/// Computes group-relative advantages, replacing each rollout's reward.
///
/// Rewards are grouped by [`RolloutWithReward::group_id`]; within a group the
/// advantage is `reward - mean(group)`, which is what GRPO's advantage column
/// holds. Rollouts without a group id form the empty-string group and go through
/// [`RewardNormaliser`] with [`RewardNormalisation::GroupCentered`] as the
/// documented scalar fallback, so one lone ungrouped rollout yields an advantage
/// of `0.0` instead of an error. Non-finite rewards are treated as `0.0` first.
///
/// The rollouts' rewards are replaced by their advantages; [`AdvantageReport`]
/// keeps the raw rewards and the per-group statistics, so the transformation is
/// reversible for reporting. Ordering is the input order and nothing reads a
/// clock or a random source.
pub fn compute_advantages(rollouts: &mut [RolloutWithReward]) -> AdvantageReport {
    let observed: Vec<(f64, Option<String>)> = rollouts
        .iter()
        .map(|rollout| {
            (
                sanitise(rollout.reward),
                rollout.group_id.clone().filter(|group| !group.is_empty()),
            )
        })
        .collect();
    let report = scalar_advantages(&observed);
    for (rollout, advantage) in rollouts.iter_mut().zip(&report.advantages) {
        rollout.reward = *advantage;
    }
    report
}

/// The shared advantage arithmetic over `(reward, group_id)` samples.
fn scalar_advantages(observed: &[(f64, Option<String>)]) -> AdvantageReport {
    let rewards: Vec<f64> = observed.iter().map(|(reward, _)| *reward).collect();
    let mut buckets: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (index, (_, group)) in observed.iter().enumerate() {
        buckets
            .entry(group.clone().unwrap_or_default())
            .or_default()
            .push(index);
    }
    let mut advantages = vec![0.0; observed.len()];
    let mut groups: BTreeMap<String, AdvantageGroupStats> = BTreeMap::new();
    let mut fallback_groups = Vec::new();
    for (group_id, indices) in &buckets {
        let group_rewards: Vec<f64> = indices.iter().map(|&index| rewards[index]).collect();
        let group_advantages = if group_id.is_empty() {
            fallback_groups.push(group_id.clone());
            normalise_scalar(&group_rewards)
        } else {
            let centre = mean(&group_rewards);
            group_rewards.iter().map(|reward| reward - centre).collect()
        };
        for (&index, advantage) in indices.iter().zip(&group_advantages) {
            advantages[index] = *advantage;
        }
        groups.insert(
            group_id.clone(),
            AdvantageGroupStats {
                count: indices.len(),
                mean_reward: mean(&group_rewards),
                min_reward: min(&group_rewards),
                max_reward: max(&group_rewards),
                mean_advantage: mean(&group_advantages),
                min_advantage: min(&group_advantages),
                max_advantage: max(&group_advantages),
                fallback: group_id.is_empty(),
            },
        );
    }
    AdvantageReport {
        rewards,
        advantages,
        groups,
        fallback_groups,
    }
}

/// Applies the crate's scalar normaliser to one ungrouped bucket.
fn normalise_scalar(rewards: &[f64]) -> Vec<f64> {
    let mut bucket: Vec<RolloutWithReward> = rewards
        .iter()
        .enumerate()
        .map(|(index, reward)| RolloutWithReward {
            turn_id: index as u64,
            task_id: String::new(),
            rollout_id: index.to_string(),
            input_prompt_ids: Vec::new(),
            output_response_ids: Vec::new(),
            reward: *reward,
            n_turns: 1,
            loss_mask: Vec::new(),
            group_id: None,
        })
        .collect();
    RewardNormaliser::new(RewardNormalisation::GroupCentered).normalise(&mut bucket);
    bucket.iter().map(|rollout| rollout.reward).collect()
}

/// Replaces a non-finite reward with `0.0`.
fn sanitise(reward: f64) -> f64 {
    if reward.is_finite() {
        reward
    } else {
        0.0
    }
}

/// Mean of `values`; `0.0` when empty.
fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

/// Minimum of `values`; `0.0` when empty.
fn min(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().copied().fold(f64::INFINITY, f64::min)
}

/// Maximum of `values`; `0.0` when empty.
fn max(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().copied().fold(f64::NEG_INFINITY, f64::max)
}

/// The gradient seam: the training stack, reduced to what the pipeline needs.
#[async_trait]
pub trait GradientOptimizer: Send + Sync {
    /// Stable name used in reports and checkpoints.
    fn name(&self) -> &str;

    /// Derives the parameter updates for one packed batch and its advantages.
    ///
    /// # Errors
    ///
    /// Any failure of the underlying optimizer, including the typed
    /// [`Error::Unsupported`] of [`GpuGradientOptimizer`].
    async fn gradients(&self, batch: &RlBatch, advantages: &AdvantageReport) -> Result<Updates>;

    /// Whether the optimizer needs a GPU stack this crate does not ship.
    fn requires_gpu(&self) -> bool;
}

/// The portable gradient optimizer: one policy update per sample.
///
/// Wraps a [`Policy`] and applies each sample's advantage as its reward weight,
/// one single-sample batch at a time and in batch order, then reports
/// [`Policy::parameters`] as the update payload of the policy's `policy` target.
/// The order matters: policies accumulate state per update, so the produced
/// parameters are a function of the input order. It never needs a GPU, and the
/// whole path is offline and deterministic for the crate's deterministic policies.
pub struct OfflinePolicyOptimizer {
    policy: Arc<dyn Policy>,
}

impl OfflinePolicyOptimizer {
    /// Creates the optimizer over `policy`.
    #[must_use]
    pub fn new(policy: Arc<dyn Policy>) -> Self {
        Self { policy }
    }

    /// The wrapped policy.
    #[must_use]
    pub fn policy(&self) -> &Arc<dyn Policy> {
        &self.policy
    }
}

impl std::fmt::Debug for OfflinePolicyOptimizer {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OfflinePolicyOptimizer")
            .field("policy", &self.policy.name())
            .finish()
    }
}

#[async_trait]
impl GradientOptimizer for OfflinePolicyOptimizer {
    fn name(&self) -> &'static str {
        "offline-policy"
    }

    async fn gradients(&self, batch: &RlBatch, advantages: &AdvantageReport) -> Result<Updates> {
        if advantages.len() != batch.samples.len() {
            return Err(Error::validation(format!(
                "OfflinePolicyOptimizer::gradients: {} advantage(s) for {} sample(s)",
                advantages.len(),
                batch.samples.len()
            )));
        }
        let mut updates = Updates::new();
        if batch.samples.is_empty() {
            return Ok(updates);
        }
        let mut weighted = 0.0;
        for (index, sample) in batch.samples.iter().enumerate() {
            let advantage = advantages.advantages[index];
            weighted += advantage;
            let mut derived = sample.clone();
            derived.reward = advantage;
            derived.metadata.insert(
                "reward".to_owned(),
                serde_json::Number::from_f64(sample.reward).map_or(serde_json::Value::Null, serde_json::Value::Number),
            );
            derived.metadata.insert(
                "advantage".to_owned(),
                serde_json::Number::from_f64(advantage).map_or(serde_json::Value::Null, serde_json::Value::Number),
            );
            let single = RlBatch::from_samples(format!("{}-s{index}", batch.batch_id), vec![derived]);
            self.policy.update(&single).await?;
        }
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "optimizer".to_owned(),
            serde_json::Value::String(self.name().to_owned()),
        );
        metadata.insert("samples".to_owned(), json!(batch.samples.len()));
        metadata.insert("mean_advantage".to_owned(), json!(weighted / batch.samples.len() as f64));
        updates.insert_update(
            self.policy.name().to_owned(),
            POLICY_TARGET,
            Update::Structured(UpdateValue {
                payload: self.policy.parameters(),
                metadata,
                ..UpdateValue::default()
            }),
        );
        Ok(updates)
    }

    fn requires_gpu(&self) -> bool {
        false
    }
}

/// The deferred GPU gradient optimizer.
///
/// Python's PPO/`verl`/`ray` optimiser needs `torch`, `transformers`, `verl`,
/// `ray` and a CUDA device. None of that is ported, so
/// [`GradientOptimizer::gradients`] returns the crate's typed
/// [`Error::Unsupported`] naming the missing stack instead of pretending to train.
#[derive(Debug, Clone, Copy, Default)]
pub struct GpuGradientOptimizer;

#[async_trait]
impl GradientOptimizer for GpuGradientOptimizer {
    fn name(&self) -> &'static str {
        "gpu"
    }

    async fn gradients(&self, _batch: &RlBatch, _advantages: &AdvantageReport) -> Result<Updates> {
        Err(crate::deferred::unsupported(
            "GpuGradientOptimizer::gradients (torch/verl/ray GPU optimiser)",
        ))
    }

    fn requires_gpu(&self) -> bool {
        true
    }
}

/// What one [`RlUpdatePipeline::apply`] produced.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParameterUpdate {
    /// Identifier of the input batch.
    pub batch_id: String,
    /// Name of the gradient optimizer that produced the updates.
    pub optimizer: String,
    /// The pipeline epoch the checkpoint was written for.
    pub epoch: usize,
    /// Number of samples the gradient pass consumed.
    pub sample_count: usize,
    /// Number of packed chunks the input formed.
    pub packed_batches: usize,
    /// Number of samples the packer truncated to satisfy its token budget.
    pub truncated: usize,
    /// The advantages the gradient pass consumed.
    pub advantages: AdvantageReport,
    /// The updates exactly as derived, in write order.
    pub updates: Updates,
    /// The per-update apply results, non-`None` updates first and `None` ones last.
    pub results: Vec<ApplyResult>,
    /// Aggregate counts over [`Self::results`].
    pub summary: ApplySummary,
    /// The checkpoint describing this step.
    pub checkpoint: RlCheckpoint,
}

/// Batching, gradient application and checkpoint emission over the `rl` seams.
pub struct RlUpdatePipeline {
    optimizer: Arc<dyn GradientOptimizer>,
    target: Arc<dyn UpdateTarget>,
    packer: BatchPacker,
    run_id: String,
    epoch: usize,
}

impl RlUpdatePipeline {
    /// Creates a pipeline over `optimizer` that writes to `target`.
    ///
    /// The run id defaults to `"rl-update"`, the packer to [`BatchPacker::default`],
    /// and the epoch counter starts at `0`; the first [`Self::apply`] reports epoch
    /// `1`.
    #[must_use]
    pub fn new(optimizer: Arc<dyn GradientOptimizer>, target: Arc<dyn UpdateTarget>) -> Self {
        Self {
            optimizer,
            target,
            packer: BatchPacker::default(),
            run_id: "rl-update".to_owned(),
            epoch: 0,
        }
    }

    /// Replaces the batch packer used to validate and truncate the input.
    #[must_use]
    pub fn with_packer(mut self, packer: BatchPacker) -> Self {
        self.packer = packer;
        self
    }

    /// Replaces the run id recorded in the checkpoint.
    #[must_use]
    pub fn with_run_id(mut self, run_id: impl Into<String>) -> Self {
        self.run_id = run_id.into();
        self
    }

    /// The run id recorded in checkpoints.
    #[must_use]
    pub fn run_id(&self) -> &str {
        &self.run_id
    }

    /// The epoch of the last applied batch (`0` before the first [`Self::apply`]).
    #[must_use]
    pub fn epoch(&self) -> usize {
        self.epoch
    }

    /// The gradient optimizer the pipeline drives.
    #[must_use]
    pub fn optimizer(&self) -> &Arc<dyn GradientOptimizer> {
        &self.optimizer
    }

    /// Whether the gradient optimizer needs a GPU stack this crate does not ship.
    #[must_use]
    pub fn requires_gpu(&self) -> bool {
        self.optimizer.requires_gpu()
    }

    /// Runs one parameter-update step over `batch`.
    ///
    /// The batch's samples are validated and packed with
    /// [`BatchPacker`] (truncating to the configured token budget), the packed
    /// samples are flattened back into one gradient batch — a gradient pass is one
    /// optimizer step, so splitting it would multiply the steps silently — the
    /// advantages are computed, the gradient optimizer is asked for the updates,
    /// and those are applied through the target in the optimizer's write order
    /// (`None`-valued updates last). The result carries a checkpoint describing the
    /// applied updates and the advantages; persist it with [`Self::save_checkpoint`].
    ///
    /// # Errors
    ///
    /// - [`Error::validation`] for an empty batch, an invalid sample, or a gradient
    ///   optimizer that answers with a mismatched advantage/sample count;
    /// - the gradient optimizer's own error, including the typed
    ///   [`Error::Unsupported`] from [`GpuGradientOptimizer`];
    /// - [`Error::workflow`] when any update fails to apply. The message names the
    ///   count and the first failure, so a partially applied batch is never
    ///   reported as success.
    pub async fn apply(&mut self, batch: &RlBatch) -> Result<ParameterUpdate> {
        if batch.samples.is_empty() {
            return Err(Error::validation(format!(
                "RlUpdatePipeline::apply: batch {:?} carries no samples",
                batch.batch_id
            )));
        }
        for sample in &batch.samples {
            sample.validate()?;
        }
        let packed = self.packer.pack(&batch.samples);
        let mut samples = Vec::new();
        let mut truncated = 0usize;
        for chunk in &packed {
            truncated += chunk.truncated;
            samples.extend(chunk.samples.iter().cloned());
        }
        let mut validated = RlBatch::from_samples(batch.batch_id.clone(), samples);
        validated.truncated = truncated;

        let observed: Vec<(f64, Option<String>)> = validated
            .samples
            .iter()
            .map(|sample| (sanitise(sample.reward), sample.group_id.clone()))
            .collect();
        let advantages = scalar_advantages(&observed);
        for (sample, advantage) in validated
            .samples
            .iter_mut()
            .zip(&advantages.advantages)
        {
            sample.reward = *advantage;
        }

        let updates = self.optimizer.gradients(&validated, &advantages).await?;
        let mut operators = OperatorMap::new();
        for (operator_id, _, value) in updates.iter() {
            if value.is_some() {
                operators.insert(operator_id.to_owned(), Arc::clone(&self.target));
            }
        }
        let results = execute_updates(&operators, &updates);
        let summary = summarize_apply_results(&results);
        if summary.failed > 0 {
            let reason = results
                .iter()
                .find(|result| !result.ok())
                .map(|result| result.errors.join("; "))
                .unwrap_or_default();
            return Err(Error::workflow(format!(
                "RlUpdatePipeline::apply: {} of {} update(s) failed: {reason}",
                summary.failed, summary.total
            )));
        }

        self.epoch += 1;
        let mut metrics = BTreeMap::new();
        metrics.insert("samples".to_owned(), validated.samples.len() as f64);
        metrics.insert("packed_batches".to_owned(), packed.len() as f64);
        metrics.insert("truncated".to_owned(), truncated as f64);
        metrics.insert("applied".to_owned(), summary.applied as f64);
        metrics.insert("failed".to_owned(), summary.failed as f64);
        metrics.insert("mean_reward".to_owned(), advantages.mean_reward());
        metrics.insert("mean_advantage".to_owned(), advantages.mean_advantage());
        let checkpoint = RlCheckpoint::new(
            self.run_id.clone(),
            self.epoch,
            advantages.best_reward(),
            json!({
                "optimizer": self.optimizer.name(),
                "updates": updates,
                "advantages": advantages,
            }),
            metrics,
        );
        checkpoint.validate()?;

        Ok(ParameterUpdate {
            batch_id: batch.batch_id.clone(),
            optimizer: self.optimizer.name().to_owned(),
            epoch: self.epoch,
            sample_count: validated.samples.len(),
            packed_batches: packed.len(),
            truncated,
            advantages,
            updates,
            results,
            summary,
            checkpoint,
        })
    }

    /// Persists `update.checkpoint` into `dir` through [`RlCheckpoint::save_to_dir`].
    ///
    /// Returns the epoch file path. Persistence is deliberately not part of
    /// [`Self::apply`], so applying stays a pure in-memory step.
    ///
    /// # Errors
    ///
    /// Any filesystem error from [`RlCheckpoint::save_to_dir`].
    pub async fn save_checkpoint(update: &ParameterUpdate, dir: &Path) -> Result<PathBuf> {
        update.checkpoint.save_to_dir(dir).await
    }
}

impl std::fmt::Debug for RlUpdatePipeline {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RlUpdatePipeline")
            .field("optimizer", &self.optimizer.name())
            .field("run_id", &self.run_id)
            .field("epoch", &self.epoch)
            .finish_non_exhaustive()
    }
}

/// The update key a policy update is written to, for callers inspecting a result.
#[must_use]
pub fn policy_update_key(policy_name: &str) -> UpdateKey {
    (policy_name.to_owned(), POLICY_TARGET.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rollout(reward: f64, group: Option<&str>) -> RolloutWithReward {
        RolloutWithReward {
            turn_id: 0,
            task_id: "task".to_owned(),
            rollout_id: "rollout".to_owned(),
            input_prompt_ids: vec![1],
            output_response_ids: vec![2],
            reward,
            n_turns: 1,
            loss_mask: vec![1],
            group_id: group.map(str::to_owned),
        }
    }

    #[test]
    fn grouped_advantages_are_group_centred() {
        let mut rollouts = vec![
            rollout(1.0, Some("a")),
            rollout(0.0, Some("a")),
            rollout(2.0, Some("b")),
        ];
        let report = compute_advantages(&mut rollouts);
        assert_eq!(report.advantages, [0.5, -0.5, 0.0]);
        assert_eq!(report.rewards, [1.0, 0.0, 2.0]);
        assert_eq!(rollouts[0].reward, 0.5);
        assert_eq!(rollouts[2].reward, 0.0);
        let group = report.group("a").expect("group a");
        assert_eq!(group.count, 2);
        assert_eq!(group.mean_reward, 0.5);
        assert_eq!(group.max_advantage, 0.5);
        assert!(!group.fallback);
        assert!(report.fallback_groups.is_empty());
        assert_eq!(report.best_reward(), 2.0);
        assert_eq!(report.mean_reward(), 1.0);
        assert_eq!(report.advantage(1), Some(-0.5));
    }

    #[test]
    fn ungrouped_advantages_fall_back_to_scalar_normalisation() {
        let mut rollouts = vec![rollout(3.0, None), rollout(1.0, None), rollout(0.0, Some(""))];
        let report = compute_advantages(&mut rollouts);
        assert_eq!(report.advantages, [5.0 / 3.0, -1.0 / 3.0, -4.0 / 3.0]);
        assert_eq!(report.fallback_groups, [String::new()]);
        let group = report.group("").expect("fallback group");
        assert!(group.fallback);
        assert_eq!(group.count, 3);
    }

    #[test]
    fn non_finite_rewards_are_sanitised() {
        let mut rollouts = vec![rollout(f64::NAN, Some("a")), rollout(2.0, Some("a"))];
        let report = compute_advantages(&mut rollouts);
        assert_eq!(report.rewards, [0.0, 2.0]);
        assert_eq!(report.advantages, [-1.0, 1.0]);
        assert!(report
            .advantages
            .iter()
            .all(|advantage| advantage.is_finite()));
    }

    #[test]
    fn empty_input_reports_zeroes() {
        let report = compute_advantages(&mut []);
        assert!(report.is_empty());
        assert_eq!(report.mean_advantage(), 0.0);
        assert_eq!(report.best_reward(), 0.0);
        assert_eq!(min(&[]), 0.0);
        assert_eq!(max(&[]), 0.0);
        assert_eq!(mean(&[]), 0.0);
    }

    #[test]
    fn policy_update_key_is_the_documented_pair() {
        assert_eq!(
            policy_update_key("deterministic"),
            ("deterministic".to_owned(), "policy".to_owned())
        );
    }
}
