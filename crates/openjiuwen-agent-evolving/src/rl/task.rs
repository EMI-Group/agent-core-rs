//! Task, rollout, and reward-registry records from Python `agent_rl.schemas`.
//!
//! Every record is a faithful `serde` shape: field names, optionals, and
//! defaults match the Python models so rollout files and wire payloads can be
//! exchanged unchanged. The records carry no behaviour beyond the registry's
//! insert/replace semantics.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

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
    /// Group the turn was rolled out with, used for group-relative rewards.
    ///
    /// Rollouts produced by the same prompt share a group id. Absent group ids
    /// are treated as one shared group by [`super::RewardNormaliser`].
    #[serde(default)]
    pub group_id: Option<String>,
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
