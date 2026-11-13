//! RL configuration, batching, rollout bookkeeping, and the pluggable trainer loop.
//!
//! Python `openjiuwen.agent_evolving.agent_rl` splits into two layers:
//!
//! - pure configuration and record models (`RlConfig`, `RlTask`, `Rollout`,
//!   `RolloutMessage`, `RolloutWithReward`, `RewardRegistry`) plus the offline
//!   machinery that does not need a GPU — sample packing, reward
//!   normalisation, rollout logs, checkpoints, and the training loop over
//!   [`Policy`]/[`RolloutEngine`] seams;
//! - the GPU training/rollout engines (`OfflineRlOptimizer`'s PPO/verl/ray
//!   stack and `OnlineRlOptimizer`'s Redis/vLLM gateway), which remain deferred
//!   (see [`crate::deferred`] and `docs/PARITY.md`).
//!
//! The deferred methods keep their Python signatures and return
//! [`openjiuwen_common::error::Error::Unsupported`] naming the operation, never
//! a panic. The orchestration itself is real: [`RlTrainer`] drives pluggable
//! policies and rollout engines offline and deterministically, and
//! [`OfflineRlOptimizer::train_with`] wires it to the same configuration the
//! deferred `train` entry point uses.
// Stub entry points keep their Python `async` shape although they never await.
#![allow(clippy::unused_async_trait_impl)]

use std::path::PathBuf;
use std::sync::Arc;

use serde_json::Value;
use tokio_util::sync::CancellationToken;

use openjiuwen_common::error::{Error, Result};

mod batch;
mod checkpoint;
mod config;
mod policy;
mod rollout;
mod task;
mod trainer;

pub use batch::{token_estimate, BatchPacker, RlBatch, RlSample, MESSAGE_TOKEN_OVERHEAD};
pub use checkpoint::{RlCheckpoint, CHECKPOINT_VERSION};
pub use config::{
    RlConfig, RlPersistenceConfig, RlRolloutConfig, RlRuntimeConfig, RlTrainingConfig,
};
pub use policy::{
    DeterministicPolicy, FakeRolloutEngine, Policy, RewardFn, RlUpdate, RolloutEngine,
};
pub use rollout::{
    RewardGroupStats, RewardNormalisation, RewardNormaliser, RewardStats, RolloutLog,
};
pub use task::{RewardRegistry, RlTask, Rollout, RolloutMessage, RolloutWithReward};
pub use trainer::{RlEpochReport, RlStopReason, RlTrainReport, RlTrainer, RlTrainerConfig};

/// The offline (batch) RL optimizer.
///
/// Python drives this through `verl` + `ray` with a local PPO trainer. That
/// stack is Python-only and deferred; the type keeps the constructor, the
/// configuration accessor, and the reward registry so callers can be written
/// today, and exposes the portable orchestration through
/// [`Self::train_with`].
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
    /// deferred (see `docs/PARITY.md`). Use [`Self::train_with`] for the
    /// portable loop.
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

    /// Runs the portable offline loop over a pluggable policy and rollout engine.
    ///
    /// The epoch count, rollout count, checkpoint frequency, and checkpoint
    /// directory come from [`RlConfig`]: `training.epoch_num`,
    /// `rollout.rollout_n`, `training.save_freq`, and `training.save_path`.
    /// Unlike [`Self::train`], this method needs no GPU stack and is fully
    /// offline and deterministic.
    pub async fn train_with(
        &mut self,
        policy: Arc<dyn Policy>,
        engine: &dyn RolloutEngine,
        tasks: &[RlTask],
        cancel: &CancellationToken,
    ) -> Result<RlTrainReport> {
        let config = RlTrainerConfig {
            run_id: None,
            epochs: self.config.training.epoch_num,
            rollout_samples: self.config.rollout.rollout_n,
            early_stop_reward: RlTrainerConfig::default().early_stop_reward,
            checkpoint_dir: self.config.training.save_path.as_ref().map(PathBuf::from),
            checkpoint_every: self.config.training.save_freq.max(1),
        };
        RlTrainer::new(config)
            .train(policy, engine, tasks, cancel)
            .await
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
            group_id: None,
        };
        let decoded: RolloutWithReward =
            serde_json::from_value(serde_json::to_value(&sample).expect("serialize"))
                .expect("deserialize");
        assert_eq!(decoded, sample);
    }
}
