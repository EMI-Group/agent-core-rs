//! RL records and API-shaped stubs for the Python RL stack.
//!
//! Python `openjiuwen.agent_evolving.agent_rl` splits into two layers:
//!
//! - pure configuration and record models (`RlConfig`, `RlTask`, `Rollout`,
//!   `RolloutMessage`, `RolloutWithReward`, `RewardRegistry`), which have no ML
//!   dependency and are ported here as faithful `serde` shapes;
//! - the training/rollout engines (`OfflineRlOptimizer`, `OnlineRlOptimizer`),
//!   which need `torch`/`verl`/`ray`, Redis and a vLLM gateway, and are therefore
//!   deferred (see [`crate::deferred`] and `docs/PARITY.md`).
//!
//! The stub methods keep their Python signatures so callers, tests, and the RL
//! service router can be written against the final API; every one of them returns
//! [`openjiuwen_common::error::Error::Unsupported`] naming the operation, never a
//! panic. `RewardRegistry` is real: it is a pure name → reward map with
//! insert/replace semantics.
// Stub entry points keep their Python `async` shape although they never await.
#![allow(clippy::unused_async_trait_impl)]
use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use openjiuwen_common::error::{Error, Result};

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

/// One task handed to the RL environment (Python `RlTask`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RlTask {
    /// Identifier of the task.
    pub task_id: String,
    /// Identifier of the task this one was derived from, if any.
    pub origin_task_id: Option<String>,
    /// The task sample (prompt, metadata) as JSON.
    pub task_sample: Value,
    /// Round number the task belongs to.
    pub round_num: u64,
}

/// One model turn of a rollout (Python `Rollout`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Rollout {
    /// Turn index inside the rollout.
    pub turn_id: u64,
    /// Prompt fed to the model, as JSON.
    pub input_prompt: Value,
    /// Raw model response, as JSON.
    pub output_response: Value,
    /// LLM configuration used for the turn, as JSON.
    pub llm_config: Value,
    /// Token ids of the prompt.
    pub input_prompt_ids: Vec<u32>,
    /// Token ids of the response.
    pub output_response_ids: Vec<u32>,
}

/// A finished rollout with its reward, as published to the trainer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RolloutMessage {
    /// Identifier of the task the rollout belongs to.
    pub task_id: String,
    /// Identifier of the task this one was derived from, if any.
    pub origin_task_id: Option<String>,
    /// Identifier of the rollout.
    pub rollout_id: String,
    /// ISO-8601 start time, when the producer recorded one.
    pub start_time: Option<String>,
    /// ISO-8601 end time, when the producer recorded one.
    pub end_time: Option<String>,
    /// Rollout metadata, as JSON.
    pub rollout_info: Value,
    /// Per-turn rewards.
    pub reward_list: Vec<f64>,
    /// Aggregated reward of the rollout.
    pub global_reward: f64,
    /// Number of turns in the rollout.
    pub turn_count: u64,
    /// Round number the rollout belongs to.
    pub round_num: u64,
}

/// A reward-annotated turn used as a training sample (Python `RolloutWithReward`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RolloutWithReward {
    /// Turn index inside the rollout.
    pub turn_id: u64,
    /// Identifier of the task the turn belongs to.
    pub task_id: String,
    /// Identifier of the rollout the turn belongs to.
    pub rollout_id: String,
    /// Token ids of the prompt.
    pub input_prompt_ids: Vec<u32>,
    /// Token ids of the response.
    pub output_response_ids: Vec<u32>,
    /// Reward assigned to the turn.
    pub reward: f64,
    /// Number of turns in the rollout.
    pub n_turns: u64,
    /// Loss mask over the response tokens (`1` = train, `0` = ignore).
    pub loss_mask: Vec<u8>,
}

/// A name → reward-payload registry, mirroring Python's `RewardRegistry`.
///
/// Registration is a plain map insert: registering a new name returns `true`,
/// registering an existing name replaces the payload and returns `false`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RewardRegistry {
    /// Reward payloads keyed by reward name.
    pub rewards: BTreeMap<String, Value>,
}

impl RewardRegistry {
    /// Creates an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers `reward` under `name`.
    ///
    /// Returns `true` when the name was newly inserted and `false` when an
    /// existing payload was replaced.
    pub fn register(&mut self, name: impl Into<String>, reward: Value) -> bool {
        self.rewards.insert(name.into(), reward).is_none()
    }

    /// Returns the payload registered under `name`.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&Value> {
        self.rewards.get(name)
    }

    /// Lists the registered reward names in sorted order.
    #[must_use]
    pub fn list(&self) -> Vec<&str> {
        self.rewards.keys().map(String::as_str).collect()
    }

    /// Number of registered rewards.
    #[must_use]
    pub fn len(&self) -> usize {
        self.rewards.len()
    }

    /// Whether no reward is registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rewards.is_empty()
    }
}

/// The offline (batch) RL optimizer, stubbed.
///
/// Python drives this through `verl` + `ray` with a local PPO trainer. That stack
/// is Python-only and deferred; the type keeps the constructor, the configuration
/// accessor, and the reward registry so callers can be written today.
#[derive(Debug, Clone)]
pub struct OfflineRlOptimizer {
    config: RlConfig,
    rewards: RewardRegistry,
}

impl OfflineRlOptimizer {
    /// Creates an optimizer for `config` with an empty reward registry.
    #[must_use]
    pub fn new(config: RlConfig) -> Self {
        Self {
            config,
            rewards: RewardRegistry::new(),
        }
    }

    /// The configuration the optimizer was built with.
    #[must_use]
    pub fn config(&self) -> &RlConfig {
        &self.config
    }

    /// Runs the full offline training loop.
    ///
    /// # Errors
    ///
    /// Always returns [`Error::Unsupported`]: the distributed training stack is
    /// deferred (see `docs/PARITY.md`).
    pub async fn train(&mut self) -> Result<()> {
        Err(Error::unsupported(
            "OfflineRlOptimizer::train is deferred: the Python PPO/verl/ray training stack is \
             not ported; see docs/PARITY.md",
        ))
    }

    /// Starts offline training without waiting for it to finish.
    ///
    /// # Errors
    ///
    /// Always returns [`Error::Unsupported`]: the distributed training stack is
    /// deferred (see `docs/PARITY.md`).
    pub async fn start_training(&mut self) -> Result<()> {
        Err(Error::unsupported(
            "OfflineRlOptimizer::start_training is deferred: the Python PPO/verl/ray training \
             stack is not ported; see docs/PARITY.md",
        ))
    }

    /// Registers a reward payload under `name`; see [`RewardRegistry::register`].
    pub fn register_reward(&mut self, name: impl Into<String>, reward: Value) -> bool {
        self.rewards.register(name, reward)
    }
}

/// The online RL optimizer, stubbed.
///
/// Python declares this class abstract: it talks to a Redis-backed task queue, a
/// vLLM generation service, and the `verl` training gateway. None of those
/// dependencies are ported, so every entry point returns
/// [`Error::Unsupported`] naming the operation.
#[derive(Debug, Clone)]
pub struct OnlineRlOptimizer {
    config: RlConfig,
}

impl OnlineRlOptimizer {
    /// Creates an online optimizer for `config`.
    #[must_use]
    pub fn new(config: RlConfig) -> Self {
        Self { config }
    }

    /// The configuration the optimizer was built with.
    #[must_use]
    pub fn config(&self) -> &RlConfig {
        &self.config
    }

    /// Starts the online training loop.
    ///
    /// # Errors
    ///
    /// Always returns [`Error::Unsupported`]: the Redis/vLLM/verl gateway is
    /// deferred (see `docs/PARITY.md`).
    pub async fn start_training(&mut self) -> Result<()> {
        Err(Error::unsupported(
            "OnlineRlOptimizer::start_training is deferred: the Redis/vLLM/verl online RL \
             gateway is not ported; see docs/PARITY.md",
        ))
    }

    /// Trains one step on `batch`.
    ///
    /// # Errors
    ///
    /// Always returns [`Error::Unsupported`]: the Redis/vLLM/verl gateway is
    /// deferred (see `docs/PARITY.md`).
    pub async fn train_on_batch(&self, _batch: Value) -> Result<()> {
        Err(Error::unsupported(
            "OnlineRlOptimizer::train_on_batch is deferred: the Redis/vLLM/verl online RL \
             gateway is not ported; see docs/PARITY.md",
        ))
    }

    /// Exports the trained `LoRA` adapter and returns its path.
    ///
    /// # Errors
    ///
    /// Always returns [`Error::Unsupported`]: the Redis/vLLM/verl gateway and its
    /// artifact store are deferred (see `docs/PARITY.md`).
    pub fn export_lora(&self) -> Result<String> {
        Err(Error::unsupported(
            "OnlineRlOptimizer::export_lora is deferred: the Redis/vLLM/verl online RL \
             gateway is not ported; see docs/PARITY.md",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn config_defaults_match_the_python_values() {
        let config = RlConfig::default();
        assert_eq!(config.training.algorithm_adv_estimator, "grpo");
        assert_eq!(config.training.epoch_num, 1);
        assert_eq!(config.training.max_prompt_length, 1024);
        assert_eq!(config.training.max_response_length, 1024);
        assert!(config.training.val_before_train);
        assert!((config.rollout.actor_optimizer_lr - 1e-6).abs() < f64::EPSILON);
        assert!(config.rollout.actor_use_kl_loss);
        assert!((config.rollout.actor_clip_ratio_high - 0.28).abs() < f64::EPSILON);
        assert_eq!(config.rollout.rollout_n, 1);
        assert!((config.runtime.temperature - 1.0).abs() < f64::EPSILON);
        assert!((config.runtime.top_p - 1.0).abs() < f64::EPSILON);
        assert_eq!(config.runtime.max_new_tokens, 1024);
        assert!(!config.persistence.enabled);
        assert_eq!(config.persistence.flush_interval, 0);
    }

    #[test]
    fn config_tolerates_partial_and_unknown_fields() {
        let config: RlConfig =
            serde_json::from_str(r#"{"training":{"epoch_num":5,"future":true},"future":{}}"#)
                .expect("deserialize");
        assert_eq!(config.training.epoch_num, 5);
        assert_eq!(config.training.save_freq, 1);
        assert_eq!(config.rollout.rollout_n, 1);
    }

    #[test]
    fn registry_insert_and_replace_semantics() {
        let mut registry = RewardRegistry::new();
        assert!(registry.is_empty());
        assert!(registry.register("rule", json!({"weight": 1.0})));
        assert_eq!(registry.len(), 1);
        assert!(!registry.register("rule", json!({"weight": 2.0})));
        assert_eq!(registry.len(), 1);
        assert_eq!(registry.get("rule"), Some(&json!({"weight": 2.0})));
        assert_eq!(registry.get("missing"), None);
    }

    #[test]
    fn registry_lists_names_in_sorted_order() {
        let mut registry = RewardRegistry::new();
        registry.register("zeta", json!(3));
        registry.register("alpha", json!(1));
        registry.register("mu", json!(2));
        assert_eq!(registry.list(), ["alpha", "mu", "zeta"]);
    }

    #[test]
    fn registry_round_trips_through_json() {
        let mut registry = RewardRegistry::new();
        registry.register("exact", json!({"kind": "exact_match"}));
        let json = serde_json::to_value(&registry).expect("serialize");
        assert_eq!(json, json!({"rewards": {"exact": {"kind": "exact_match"}}}));
        let decoded: RewardRegistry = serde_json::from_value(json).expect("deserialize");
        assert_eq!(decoded, registry);
    }

    #[tokio::test]
    async fn offline_optimizer_stubs_report_unsupported() {
        let mut optimizer = OfflineRlOptimizer::new(RlConfig::default());
        assert_eq!(optimizer.config().training.epoch_num, 1);
        assert!(optimizer.register_reward("rule", json!(1)));
        let error = optimizer.train().await.expect_err("train is deferred");
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Unsupported
        );
        assert!(error.to_string().contains("OfflineRlOptimizer::train"));
        let error = optimizer.start_training().await.expect_err("deferred");
        assert!(error
            .to_string()
            .contains("OfflineRlOptimizer::start_training"));
    }

    #[tokio::test]
    async fn online_optimizer_stubs_report_unsupported() {
        let mut optimizer = OnlineRlOptimizer::new(RlConfig::default());
        assert_eq!(
            optimizer.config().training.train_files,
            Vec::<String>::new()
        );
        let error = optimizer.start_training().await.expect_err("deferred");
        assert!(error
            .to_string()
            .contains("OnlineRlOptimizer::start_training"));
        let error = optimizer
            .train_on_batch(json!({"batch": 1}))
            .await
            .expect_err("deferred");
        assert!(error
            .to_string()
            .contains("OnlineRlOptimizer::train_on_batch"));
        let error = optimizer.export_lora().expect_err("deferred");
        assert!(error.to_string().contains("OnlineRlOptimizer::export_lora"));
    }

    #[test]
    fn records_use_python_field_names() {
        let task = RlTask {
            task_id: "t-1".to_owned(),
            origin_task_id: None,
            task_sample: json!({"query": "hi"}),
            round_num: 2,
        };
        let json = serde_json::to_value(&task).expect("serialize");
        assert_eq!(
            json,
            json!({
                "task_id": "t-1",
                "origin_task_id": null,
                "task_sample": {"query": "hi"},
                "round_num": 2,
            })
        );

        let rollout = Rollout {
            turn_id: 0,
            input_prompt: json!("p"),
            output_response: json!("r"),
            llm_config: json!({}),
            input_prompt_ids: vec![1, 2],
            output_response_ids: vec![3],
        };
        let decoded: Rollout =
            serde_json::from_value(serde_json::to_value(&rollout).expect("serialize"))
                .expect("deserialize");
        assert_eq!(decoded, rollout);

        let message = RolloutMessage {
            task_id: "t-1".to_owned(),
            origin_task_id: Some("t-0".to_owned()),
            rollout_id: "r-1".to_owned(),
            start_time: None,
            end_time: None,
            rollout_info: json!({}),
            reward_list: vec![0.5],
            global_reward: 0.5,
            turn_count: 1,
            round_num: 0,
        };
        let json = serde_json::to_value(&message).expect("serialize");
        assert_eq!(json["origin_task_id"], json!("t-0"));
        assert_eq!(json["reward_list"], json!([0.5]));

        let sample = RolloutWithReward {
            turn_id: 1,
            task_id: "t-1".to_owned(),
            rollout_id: "r-1".to_owned(),
            input_prompt_ids: vec![1],
            output_response_ids: vec![2, 3],
            reward: 0.25,
            n_turns: 1,
            loss_mask: vec![1, 1],
        };
        let decoded: RolloutWithReward =
            serde_json::from_value(serde_json::to_value(&sample).expect("serialize"))
                .expect("deserialize");
        assert_eq!(decoded, sample);
    }
}
