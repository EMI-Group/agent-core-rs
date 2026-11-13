//! RL configuration records mirroring Python `agent_rl.config`.
//!
//! The Python area splits configuration into offline, online, and service
//! schemas. All of them are pure data; this module keeps the merged shape the
//! rest of the crate and the RL service consume. Defaults match the Python
//! dataclasses, and every record tolerates partial and unknown JSON fields.

use serde::{Deserialize, Serialize};

/// The full RL configuration: training, rollout, runtime, and persistence.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RlConfig {
    /// Training-loop configuration.
    pub training: RlTrainingConfig,
    /// Rollout/actor configuration.
    pub rollout: RlRolloutConfig,
    /// Generation-time runtime configuration.
    pub runtime: RlRuntimeConfig,
    /// Rollout persistence configuration.
    pub persistence: RlPersistenceConfig,
}

/// Training-loop configuration (Python `RlTrainingConfig`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RlTrainingConfig {
    /// Path of the training dataset.
    pub train_data_path: Option<String>,
    /// Path of the validation dataset.
    pub val_data_path: Option<String>,
    /// Training data files, when the dataset is split over several files.
    pub train_files: Vec<String>,
    /// Validation data files, when the dataset is split over several files.
    pub val_files: Vec<String>,
    /// Path of the policy model to train.
    pub model_path: Option<String>,
    /// Directory the trainer writes checkpoints to.
    pub save_path: Option<String>,
    /// Advantage estimator name (`grpo` in Python).
    pub algorithm_adv_estimator: String,
    /// Number of training epochs.
    pub epoch_num: usize,
    /// Save a checkpoint every `save_freq` steps.
    pub save_freq: usize,
    /// Run the validation split every `test_freq` steps.
    pub test_freq: usize,
    /// Training batch size.
    pub train_batch_size: usize,
    /// Number of concurrent rollout workers.
    pub rollout_concurrency: usize,
    /// Micro-batch size per GPU.
    pub micro_batch_size_per_gpu: usize,
    /// Maximum prompt length in tokens.
    pub max_prompt_length: usize,
    /// Maximum response length in tokens.
    pub max_response_length: usize,
    /// Whether to evaluate the validation split before training starts.
    pub val_before_train: bool,
}

fn default_algorithm_adv_estimator() -> String {
    "grpo".to_owned()
}

const fn default_one() -> usize {
    1
}

const fn default_prompt_length() -> usize {
    1024
}

const fn default_true() -> bool {
    true
}

impl Default for RlTrainingConfig {
    fn default() -> Self {
        Self {
            train_data_path: None,
            val_data_path: None,
            train_files: Vec::new(),
            val_files: Vec::new(),
            model_path: None,
            save_path: None,
            algorithm_adv_estimator: default_algorithm_adv_estimator(),
            epoch_num: default_one(),
            save_freq: default_one(),
            test_freq: default_one(),
            train_batch_size: default_one(),
            rollout_concurrency: default_one(),
            micro_batch_size_per_gpu: default_one(),
            max_prompt_length: default_prompt_length(),
            max_response_length: default_prompt_length(),
            val_before_train: default_true(),
        }
    }
}

/// Rollout/actor configuration (Python `RlRolloutConfig`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RlRolloutConfig {
    /// Learning rate of the actor optimizer.
    pub actor_optimizer_lr: f64,
    /// Whether the actor uses a KL penalty against the reference policy.
    pub actor_use_kl_loss: bool,
    /// KL-penalty coefficient.
    pub actor_kl_loss_coef: f64,
    /// Entropy-bonus coefficient.
    pub actor_entropy_coef: f64,
    /// Lower clipping ratio of the policy ratio.
    pub actor_clip_ratio_low: f64,
    /// Upper clipping ratio of the policy ratio.
    pub actor_clip_ratio_high: f64,
    /// Number of sampled responses per prompt.
    pub rollout_n: usize,
}

const fn default_actor_optimizer_lr() -> f64 {
    1e-6
}

const fn default_actor_kl_loss_coef() -> f64 {
    0.001
}

const fn default_actor_entropy_coef() -> f64 {
    0.001
}

const fn default_actor_clip_ratio_low() -> f64 {
    0.2
}

const fn default_actor_clip_ratio_high() -> f64 {
    0.28
}

impl Default for RlRolloutConfig {
    fn default() -> Self {
        Self {
            actor_optimizer_lr: default_actor_optimizer_lr(),
            actor_use_kl_loss: default_true(),
            actor_kl_loss_coef: default_actor_kl_loss_coef(),
            actor_entropy_coef: default_actor_entropy_coef(),
            actor_clip_ratio_low: default_actor_clip_ratio_low(),
            actor_clip_ratio_high: default_actor_clip_ratio_high(),
            rollout_n: default_one(),
        }
    }
}

/// Generation-time runtime configuration (Python `RlRuntimeConfig`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RlRuntimeConfig {
    /// System prompt prepended to every rollout prompt.
    pub system_prompt: String,
    /// Sampling temperature.
    pub temperature: f64,
    /// Nucleus-sampling probability mass.
    pub top_p: f64,
    /// Maximum number of generated tokens per turn.
    pub max_new_tokens: usize,
    /// Presence penalty applied to generated tokens.
    pub presence_penalty: f64,
    /// Frequency penalty applied to generated tokens.
    pub frequency_penalty: f64,
}

const fn default_temperature() -> f64 {
    1.0
}

const fn default_top_p() -> f64 {
    1.0
}

impl Default for RlRuntimeConfig {
    fn default() -> Self {
        Self {
            system_prompt: String::new(),
            temperature: default_temperature(),
            top_p: default_top_p(),
            max_new_tokens: default_prompt_length(),
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
        }
    }
}

/// Rollout persistence configuration (Python `RlPersistenceConfig`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RlPersistenceConfig {
    /// Whether rollout persistence is enabled.
    pub enabled: bool,
    /// Directory rollouts are written to.
    pub save_path: Option<String>,
    /// Flush interval in steps; `0` flushes every step.
    pub flush_interval: u64,
    /// Whether the raw rollouts are persisted.
    pub save_rollouts: bool,
    /// Whether the per-step summaries are persisted.
    pub save_step_summaries: bool,
}
