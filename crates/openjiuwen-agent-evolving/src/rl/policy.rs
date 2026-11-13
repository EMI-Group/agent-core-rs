//! Pluggable policy and rollout-engine seams for offline RL.
//!
//! The Python trainer drives a `verl` actor and a vLLM rollout service. Those
//! GPU services are deferred, so this module defines the narrow seams the
//! portable loop needs — [`Policy`] (act/update/parameters) and
//! [`RolloutEngine`] (roll a task) — and ships deterministic offline
//! implementations used by tests and examples.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, MutexGuard};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio_util::sync::CancellationToken;

use openjiuwen_common::error::{Error, Result};

use super::batch::RlBatch;
use super::task::{RlTask, RolloutWithReward};

/// The statistics reported by one [`Policy::update`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RlUpdate {
    /// Number of samples the update consumed.
    pub samples: usize,
    /// Mean reward of the consumed samples.
    pub mean_reward: f64,
    /// Best reward among the consumed samples.
    pub best_reward: f64,
    /// Opaque policy parameters after the update.
    pub parameters: Value,
}

/// A trainable policy.
///
/// Implementations own their parameters; the trainer only needs to sample
/// actions, apply a batch update, and read/restore the opaque parameter blob for
/// checkpointing.
#[async_trait]
pub trait Policy: Send + Sync {
    /// Stable name used in reports and parameter blobs.
    fn name(&self) -> &str;

    /// Produces a response for `prompt`.
    async fn act(&self, prompt: &str, cancel: &CancellationToken) -> Result<String>;

    /// Applies one packed batch and reports what was consumed.
    async fn update(&self, batch: &RlBatch) -> Result<RlUpdate>;

    /// Current opaque parameters, suitable for [`RlCheckpoint`](super::RlCheckpoint).
    fn parameters(&self) -> Value;

    /// Restores parameters produced by [`Self::parameters`].
    ///
    /// The default implementation accepts and ignores any state, which is the
    /// right behaviour for policies without trainable parameters.
    fn restore(&self, _state: &Value) -> Result<()> {
        Ok(())
    }
}

/// Produces rollouts for a task with a given policy.
#[async_trait]
pub trait RolloutEngine: Send + Sync {
    /// Stable name used in diagnostics.
    fn name(&self) -> &str;

    /// Rolls `task` out with `policy`, returning reward-annotated turns in
    /// deterministic order.
    async fn roll(
        &self,
        task: &RlTask,
        policy: Arc<dyn Policy>,
        cancel: &CancellationToken,
    ) -> Result<Vec<RolloutWithReward>>;
}

/// The injected response-scoring function: `(expected, response) -> reward`.
pub type RewardFn = dyn Fn(&str, &str) -> f64 + Send + Sync;

fn exact_match_reward(expected: &str, response: &str) -> f64 {
    if expected.trim() == response.trim() {
        1.0
    } else {
        0.0
    }
}

fn object_field(task: &RlTask, keys: &[&str]) -> Option<String> {
    let Value::Object(map) = &task.task_sample else {
        return None;
    };
    for key in keys {
        if let Some(Value::String(text)) = map.get(*key) {
            if !text.trim().is_empty() {
                return Some(text.clone());
            }
        }
    }
    None
}

fn task_prompt(task: &RlTask) -> String {
    if let Value::String(text) = &task.task_sample {
        if !text.trim().is_empty() {
            return text.clone();
        }
    }
    if let Some(prompt) = object_field(task, &["prompt", "query", "input", "question"]) {
        return prompt;
    }
    if task.task_sample.is_null() {
        return task.task_id.clone();
    }
    task.task_sample.to_string()
}

fn expected_answer(task: &RlTask) -> String {
    if let Some(expected) = object_field(task, &["expected", "answer", "label"]) {
        return expected;
    }
    if let Value::Object(map) = &task.task_sample {
        if let Some(Value::Object(metadata)) = map.get("metadata") {
            if let Some(Value::String(expected)) = metadata.get("expected") {
                return expected.clone();
            }
        }
    }
    String::new()
}

fn encode_tokens(text: &str) -> Vec<u32> {
    text.chars().map(u32::from).collect()
}

fn samples_per_task(task: &RlTask, override_count: Option<usize>) -> usize {
    if let Some(count) = override_count {
        return count.max(1);
    }
    let configured = task
        .task_sample
        .get("num_samples")
        .and_then(Value::as_u64)
        .unwrap_or(1);
    configured.max(1) as usize
}

/// A deterministic rollout engine for offline tests and baselines.
///
/// Rolls `policy` out `num_samples` times per task (configured on the engine or
/// in the task sample's `num_samples` field) and scores each response with the
/// injected [`RewardFn`]. The default reward function is exact match against the
/// task's `expected`/`answer`/`label` field. Responses of the same prompt share
/// the prompt as their group id, so group-relative normalisation behaves like
/// GRPO. Rollout ids and ordering are deterministic.
pub struct FakeRolloutEngine {
    reward_fn: Arc<RewardFn>,
    sample_count: Option<usize>,
}

impl std::fmt::Debug for FakeRolloutEngine {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("FakeRolloutEngine")
            .field("sample_count", &self.sample_count)
            .finish_non_exhaustive()
    }
}

impl Default for FakeRolloutEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl FakeRolloutEngine {
    /// Creates an engine with exact-match scoring and no sample-count override.
    #[must_use]
    pub fn new() -> Self {
        Self {
            reward_fn: Arc::new(exact_match_reward),
            sample_count: None,
        }
    }

    /// Replaces the scoring function.
    #[must_use]
    pub fn with_reward_fn<F>(mut self, reward_fn: F) -> Self
    where
        F: Fn(&str, &str) -> f64 + Send + Sync + 'static,
    {
        self.reward_fn = Arc::new(reward_fn);
        self
    }

    /// Overrides how many rollouts are produced per task.
    #[must_use]
    pub fn with_sample_count(mut self, count: usize) -> Self {
        self.sample_count = Some(count.max(1));
        self
    }
}

#[async_trait]
impl RolloutEngine for FakeRolloutEngine {
    fn name(&self) -> &'static str {
        "fake"
    }

    async fn roll(
        &self,
        task: &RlTask,
        policy: Arc<dyn Policy>,
        cancel: &CancellationToken,
    ) -> Result<Vec<RolloutWithReward>> {
        if cancel.is_cancelled() {
            return Err(Error::cancelled());
        }
        let prompt = task_prompt(task);
        let expected = expected_answer(task);
        let count = samples_per_task(task, self.sample_count);
        let mut rollouts = Vec::with_capacity(count);
        for index in 0..count {
            if cancel.is_cancelled() {
                return Err(Error::cancelled());
            }
            let response = policy.act(&prompt, cancel).await?;
            let scored = (self.reward_fn)(&expected, &response);
            let response_ids = encode_tokens(&response);
            let loss_mask = vec![1; response_ids.len()];
            rollouts.push(RolloutWithReward {
                turn_id: index as u64,
                task_id: task.task_id.clone(),
                rollout_id: format!("{}-{index}", task.task_id),
                input_prompt_ids: encode_tokens(&prompt),
                output_response_ids: response_ids,
                reward: if scored.is_finite() { scored } else { 0.0 },
                n_turns: 1,
                loss_mask,
                group_id: Some(prompt.clone()),
            });
        }
        Ok(rollouts)
    }
}

/// A deterministic policy backed by a scripted response table.
///
/// A stable FNV-1a hash of the prompt selects a scripted response, or
/// synthesises `"{fallback_prefix}:{hash}"` when the prompt is not scripted.
/// [`Policy::update`] records sample statistics in the policy state, and
/// [`Policy::parameters`]/[`Policy::restore`] round-trip that state through JSON.
/// This is the documented offline stand-in for a real model policy.
pub struct DeterministicPolicy {
    responses: BTreeMap<String, String>,
    fallback_prefix: String,
    state: Mutex<Value>,
}

impl std::fmt::Debug for DeterministicPolicy {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DeterministicPolicy")
            .field("responses", &self.responses)
            .field("fallback_prefix", &self.fallback_prefix)
            .finish_non_exhaustive()
    }
}

impl DeterministicPolicy {
    /// Creates a policy over the scripted `responses` table.
    #[must_use]
    pub fn new(responses: BTreeMap<String, String>) -> Self {
        Self {
            responses,
            fallback_prefix: "policy".to_owned(),
            state: Mutex::new(Value::Null),
        }
    }

    /// Sets the prefix of synthesised fallback responses.
    #[must_use]
    pub fn with_fallback_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.fallback_prefix = prefix.into();
        self
    }

    fn lock_state(&self) -> MutexGuard<'_, Value> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

/// Stable FNV-1a hash of a prompt, used to pick scripted responses.
fn prompt_hash(prompt: &str) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in prompt.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

#[async_trait]
impl Policy for DeterministicPolicy {
    fn name(&self) -> &'static str {
        "deterministic"
    }

    async fn act(&self, prompt: &str, cancel: &CancellationToken) -> Result<String> {
        if cancel.is_cancelled() {
            return Err(Error::cancelled());
        }
        if let Some(response) = self.responses.get(prompt) {
            return Ok(response.clone());
        }
        Ok(format!(
            "{}:{:016x}",
            self.fallback_prefix,
            prompt_hash(prompt)
        ))
    }

    async fn update(&self, batch: &RlBatch) -> Result<RlUpdate> {
        let mut sum = 0.0;
        let mut best = 0.0;
        let mut seen = false;
        for sample in &batch.samples {
            if sample.reward.is_finite() {
                sum += sample.reward;
                if !seen || sample.reward > best {
                    best = sample.reward;
                    seen = true;
                }
            }
        }
        let samples = batch.samples.len();
        let mean_reward = if samples == 0 {
            0.0
        } else {
            sum / samples as f64
        };
        let parameters = {
            let mut state = self.lock_state();
            let updates = state
                .get("updates")
                .and_then(Value::as_u64)
                .unwrap_or(0)
                .saturating_add(1);
            let parameters = json!({
                "name": self.name(),
                "updates": updates,
                "samples": samples,
                "mean_reward": mean_reward,
                "best_reward": best,
            });
            *state = parameters.clone();
            parameters
        };
        Ok(RlUpdate {
            samples,
            mean_reward,
            best_reward: best,
            parameters,
        })
    }

    fn parameters(&self) -> Value {
        self.lock_state().clone()
    }

    fn restore(&self, state: &Value) -> Result<()> {
        *self.lock_state() = state.clone();
        Ok(())
    }
}
