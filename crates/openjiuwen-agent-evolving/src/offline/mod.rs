//! The GPU-free offline RL runtime (Python `agent_rl/offline`).
//!
//! Python's offline PPO pipeline combines a portable half — a task queue,
//! parallel rollout workers, rollout encoders, batch planning, processors,
//! persistence, metrics and the training loop — with a Python-only half
//! (`verl`, `ray`, `torch`, `transformers`, vLLM and the YAML configuration
//! ecosystem). This module ports the portable half to idiomatic Rust and keeps
//! every GPU entry point typed as [`openjiuwen_common::error::Error::Unsupported`]
//! (see `trainer::PpoTrainingExecutor`).
//!
//! # Module map
//!
//! - `queue` — `TaskQueue`: FIFO task queue, in-processing tracking and a
//!   rollout result buffer (`coordinator/task_queue.py`).
//! - `store` — `RolloutPersistence`, `FileRolloutStore` (JSONL grouped by step
//!   ranges) and `NullRolloutStore` (`store/`).
//! - `processors` — rollout classifiers, stop validators and samplers plus the
//!   explicitly registered `ProcessorsRegistry` (`coordinator/processors.py`).
//! - `encoding` — `TurnEncoder`: per-turn and whole-trajectory samples
//!   (`coordinator/encoding.py`).
//! - `batch` — the portable replacement for Python's `TensorDict` builder:
//!   padding, attention masks, token-level scores and `BatchPlan`
//!   (`coordinator/batch_builder.py`).
//! - `runtime` — the agent seam (`RolloutSource`), reward composers,
//!   `RuntimeExecutor` and `ParallelRuntimeExecutor` (`runtime/`).
//! - `coordinator` — `TrainingCoordinator`: round scheduling, rollout
//!   collection, classification/sampling and validation
//!   (`coordinator/training_coordinator.py`).
//! - `metrics` — `TrainingStepMetrics`, `MetricsTracker` and
//!   `BatchDiagnostics` (`store/metrics_tracker.py`, without verl/W&B).
//! - [`trainer`] — `OfflineTrainer` (the `MainTrainer` port) plus the typed GPU
//!   training seam (`main_trainer.py`).
//! - [`experience`] — the experience-collection hook that turns rollouts into
//!   [`crate::trajectory`] recordings and [`crate::experience`] records.
//!
//! # Determinism
//!
//! No module reads the wall clock, a random number generator, a process id or a
//! filesystem modification time. Identifiers are derived from their inputs with
//! the FNV-1a hash in `stable_id`, token ids come from character codes (see
//! `encoding`), records are ordered through `BTreeMap`/`BTreeSet`, and the
//! persisted JSONL documents omit Python's wall-clock `timestamp` field.
//! Polling intervals only affect *when* work completes, never *what* is
//! produced.

mod batch;
mod coordinator;
mod encoding;
mod experience;
mod metrics;
mod processors;
mod queue;
mod runtime;
mod store;
mod trainer;

use serde::Deserialize;
use serde_json::Value;

use openjiuwen_common::error::{Error, Result};

use crate::rl::Rollout;

pub use batch::{
    assemble_batch, left_padded_ids_and_attention_mask, right_padded_ids_and_attention_mask,
    token_level_scores, truncate_prompt_and_response, BatchBuilder, BatchComponents, BatchConfig,
    BatchPlan,
};
pub use coordinator::{
    CoordinatorConfig, CoordinatorReport, RolloutRewards, RoundReport, TrainingCoordinator,
    ValidationMetrics,
};
pub use encoding::{EncodingOutcome, TurnEncoder};
pub use experience::{ExperienceCollector, TrajectoryExperienceCollector};
pub use metrics::{BatchDiagnostics, MetricsTracker, SeriesSummary, TrainingStepMetrics};
pub use processors::{
    classify_rollouts, downsample_one_uid, sampling, sampling_ada, validate_stop,
    validate_stop_balanced, ClassifiedRollouts, ClassifierFn, ProcessorsRegistry, RolloutCache,
    SampledCaches, SamplerFn, ValidatorFn, ADAPTIVE_SAMPLER, BALANCED_VALIDATOR,
    DEFAULT_CLASSIFIER, DEFAULT_SAMPLER, DEFAULT_VALIDATOR,
};
pub use queue::TaskQueue;
pub use runtime::{
    ExactMatchComposer, ParallelRuntimeExecutor, RewardComposer, RolloutSource, RuntimeExecutor,
    RuntimeOutcome, ScriptedRolloutSource, WeightedRewardComposer,
};
pub use store::{FileRolloutStore, NullRolloutStore, RolloutPersistence, RolloutPhase};
pub use trainer::{
    GpuTrainingExecutor, OfflineStopReason, OfflineTrainReport, OfflineTrainer,
    OfflineTrainerConfig, PpoTrainingExecutor, ValidationSummary, GPU_TRAINING_AREA,
};

/// One FNV-1a hash of a byte slice.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// Builds a deterministic identifier: `"{prefix}-{16 hex digits}"`.
///
/// Python mints `uuid4` strings for tasks, rollouts and trajectories; a random
/// identifier cannot be reproduced, so both the identifier and every record
/// derived from it would change on each run. Hashing the identifying parts
/// instead keeps runs comparable while staying collision-resistant enough for
/// in-process bookkeeping.
fn stable_id(prefix: &str, parts: &[&str]) -> String {
    let mut bytes: Vec<u8> = Vec::new();
    for part in parts {
        bytes.extend_from_slice(part.as_bytes());
        // A separator that cannot occur inside a UTF-8 byte sequence keeps
        // `["ab", "c"]` and `["a", "bc"]` distinct.
        bytes.push(0xff);
    }
    format!("{prefix}-{:016x}", fnv1a(&bytes))
}

/// Tolerant decoding shape for the turns stored in `RolloutMessage.rollout_info`.
#[derive(Debug, Deserialize)]
struct TurnRecord {
    #[serde(default)]
    turn_id: u64,
    #[serde(default)]
    input_prompt: Value,
    #[serde(default)]
    output_response: Value,
    #[serde(default)]
    llm_config: Value,
    #[serde(default)]
    input_prompt_ids: Vec<u32>,
    #[serde(default)]
    output_response_ids: Vec<u32>,
}

/// Decodes the turns of a rollout message.
///
/// Python types `RolloutMessage.rollout_info` as a list of `Rollout`; this crate
/// keeps it as JSON ([`Value`]) because the same field carries the wire payload.
/// Every field of a turn is defaulted, so a producer that omits `turn_id` or the
/// token ids still decodes, and unknown fields are ignored.
///
/// A `null` field and an empty array both mean "no turns".
///
/// # Errors
///
/// [`Error::validation`] when `rollout_info` is not an array of turn objects.
pub fn rollout_turns(message: &crate::rl::RolloutMessage) -> Result<Vec<Rollout>> {
    if message.rollout_info.is_null() {
        return Ok(Vec::new());
    }
    let records: Vec<TurnRecord> =
        serde_json::from_value(message.rollout_info.clone()).map_err(|error| {
            Error::validation(format!(
                "rollout {} has a malformed rollout_info: {error}",
                message.rollout_id
            ))
        })?;
    Ok(records
        .into_iter()
        .map(|record| Rollout {
            turn_id: record.turn_id,
            input_prompt: record.input_prompt,
            output_response: record.output_response,
            llm_config: record.llm_config,
            input_prompt_ids: record.input_prompt_ids,
            output_response_ids: record.output_response_ids,
        })
        .collect())
}

/// Renders a prompt payload as text.
///
/// Python prompts are dicts of the form `{"message": [...], "tools": [...]}`;
/// strings and plain JSON payloads also appear. The rendering prefers the
/// `message`/`messages` list, then a `prompt`/`query` string, and finally the
/// compact JSON encoding so no information is dropped silently.
fn render_prompt_text(prompt: &Value) -> String {
    match prompt {
        Value::Null => String::new(),
        Value::String(text) => text.clone(),
        Value::Object(map) => {
            if let Some(messages) = map.get("message").or_else(|| map.get("messages")) {
                let rendered = render_messages(messages);
                if !rendered.trim().is_empty() {
                    return rendered;
                }
            }
            for key in ["prompt", "query", "input", "question"] {
                if let Some(Value::String(text)) = map.get(key) {
                    if !text.trim().is_empty() {
                        return text.clone();
                    }
                }
            }
            prompt.to_string()
        }
        other => other.to_string(),
    }
}

fn render_messages(messages: &Value) -> String {
    match messages {
        Value::String(text) => text.clone(),
        Value::Array(items) => items
            .iter()
            .map(render_message)
            .filter(|text| !text.is_empty())
            .collect::<Vec<_>>()
            .join("\n"),
        other => other.to_string(),
    }
}

fn render_message(message: &Value) -> String {
    match message {
        Value::String(text) => text.clone(),
        Value::Object(map) => map.get("content").map(render_content).unwrap_or_default(),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

fn render_content(content: &Value) -> String {
    match content {
        Value::Null => String::new(),
        Value::String(text) => text.clone(),
        Value::Array(parts) => parts
            .iter()
            .map(|part| match part {
                Value::String(text) => text.clone(),
                Value::Object(map) => map
                    .get("text")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned(),
                other => other.to_string(),
            })
            .filter(|text| !text.is_empty())
            .collect::<Vec<_>>()
            .join(" "),
        other => other.to_string(),
    }
}

/// Renders a model response payload as text.
fn render_response_text(response: &Value) -> String {
    match response {
        Value::Null => String::new(),
        Value::String(text) => text.clone(),
        Value::Object(map) => {
            if let Some(content) = map.get("content") {
                return render_content(content);
            }
            if let Some(Value::String(text)) = map.get("text") {
                return text.clone();
            }
            response.to_string()
        }
        other => other.to_string(),
    }
}

/// Reads the `ground_truth` field Python stores on a turn's prompt payload.
fn prompt_ground_truth(prompt: &Value) -> String {
    if let Value::Object(map) = prompt {
        for key in ["ground_truth", "expected", "answer", "label"] {
            if let Some(Value::String(text)) = map.get(key) {
                if !text.trim().is_empty() {
                    return text.clone();
                }
            }
        }
    }
    String::new()
}

/// The deterministic token encoding used by this module.
///
/// One token per Unicode scalar value (its `u32` code point). The crate's
/// tokenizer-free [`crate::rl::token_estimate`] is a *length* heuristic, not an
/// encoding, so it cannot produce ids; Python's fallback path calls a real
/// tokenizer, which this workspace does not ship. Character codes keep the
/// pipeline reproducible, are UTF-8 safe and match the ids
/// [`crate::rl::FakeRolloutEngine`] produces, so samples from both paths mix.
fn encode_text(text: &str) -> Vec<u32> {
    text.chars().map(u32::from).collect()
}

/// The prompt ids of a turn: precomputed ids when present, else a text encoding.
fn turn_prompt_ids(turn: &Rollout) -> Vec<u32> {
    if !turn.input_prompt_ids.is_empty() {
        return turn.input_prompt_ids.clone();
    }
    let text = render_prompt_text(&turn.input_prompt);
    encode_text(&text)
}

/// The response ids of a turn: precomputed ids when present, else a text encoding.
fn turn_response_ids(turn: &Rollout) -> Vec<u32> {
    if !turn.output_response_ids.is_empty() {
        return turn.output_response_ids.clone();
    }
    let text = render_response_text(&turn.output_response);
    encode_text(&text)
}

/// The task identity a sample under `origin_task_id` carries.
///
/// Python passes `rolloutmsg.origin_task_id` straight through, which can be
/// `None`; [`crate::rl::RolloutWithReward::task_id`] is a `String`, so an absent
/// origin falls back to the task id of the message.
fn sample_task_id(message: &crate::rl::RolloutMessage) -> String {
    message
        .origin_task_id
        .clone()
        .unwrap_or_else(|| message.task_id.clone())
}

/// The reward of a rollout, coerced to a finite number.
fn finite_reward(message: &crate::rl::RolloutMessage) -> f64 {
    if message.global_reward.is_finite() {
        message.global_reward
    } else {
        0.0
    }
}

/// The arithmetic mean of `values`, or `0.0` when empty.
fn mean(values: impl Iterator<Item = f64>) -> f64 {
    let mut total = 0.0;
    let mut count = 0usize;
    for value in values {
        if value.is_finite() {
            total += value;
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        total / count as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    use crate::rl::RolloutMessage;

    fn message(rollout_info: Value) -> RolloutMessage {
        RolloutMessage {
            task_id: "t-1".to_owned(),
            origin_task_id: None,
            rollout_id: "r-1".to_owned(),
            start_time: None,
            end_time: None,
            rollout_info,
            reward_list: Vec::new(),
            global_reward: 0.0,
            turn_count: 0,
            round_num: 0,
        }
    }

    #[test]
    fn stable_ids_are_reproducible_and_part_sensitive() {
        assert_eq!(
            stable_id("task", &["1", "2"]),
            stable_id("task", &["1", "2"])
        );
        assert_ne!(
            stable_id("task", &["1", "2"]),
            stable_id("task", &["12", ""])
        );
        assert_ne!(
            stable_id("task", &["1", "2"]),
            stable_id("task", &["1", "3"])
        );
        assert!(stable_id("rollout", &["x"]).starts_with("rollout-"));
        assert_eq!(stable_id("x", &[]).len(), "x-".len() + 16);
    }

    #[test]
    fn turns_decode_with_defaults_and_reject_malformed_payloads() {
        let decoded = rollout_turns(&message(json!([]))).expect("empty");
        assert!(decoded.is_empty(), "an empty array has no turns");
        let decoded = rollout_turns(&message(Value::Null)).expect("null");
        assert!(decoded.is_empty(), "a null payload has no turns");

        let decoded = rollout_turns(&message(json!([
            {"input_prompt_ids": [1, 2], "output_response_ids": [3]},
            {"turn_id": 7, "input_prompt": {"message": []}, "future": true}
        ])))
        .expect("decode");
        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded[0].turn_id, 0);
        assert_eq!(decoded[0].input_prompt_ids, vec![1, 2]);
        assert_eq!(decoded[1].turn_id, 7);
        assert!(
            decoded[1].output_response_ids.is_empty(),
            "the response is empty"
        );

        let error = rollout_turns(&message(json!({"turns": []}))).expect_err("not an array");
        assert!(error.to_string().contains("malformed rollout_info"));
    }

    #[test]
    fn prompt_rendering_prefers_message_content() {
        let prompt = json!({"message": [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": [{"text": "hi"}, {"text": "there"}]}
        ]});
        assert_eq!(render_prompt_text(&prompt), "be brief\nhi there");
        assert_eq!(render_prompt_text(&json!("plain")), "plain");
        assert_eq!(render_prompt_text(&json!({"query": "q"})), "q");
        assert_eq!(render_prompt_text(&Value::Null), "");
        assert_eq!(render_prompt_text(&json!({"other": 1})), r#"{"other":1}"#);
    }

    #[test]
    fn response_and_ground_truth_rendering() {
        assert_eq!(render_response_text(&json!("answer")), "answer");
        assert_eq!(
            render_response_text(&json!({"content": "answer"})),
            "answer"
        );
        assert_eq!(render_response_text(&json!({"text": "answer"})), "answer");
        assert_eq!(
            render_response_text(&json!({"tool": "x"})),
            r#"{"tool":"x"}"#
        );
        assert_eq!(render_response_text(&Value::Null), "");

        assert_eq!(prompt_ground_truth(&json!({"ground_truth": "42"})), "42");
        assert_eq!(prompt_ground_truth(&json!({"metadata": {}})), "");
        assert_eq!(prompt_ground_truth(&json!("42")), "");
    }

    #[test]
    fn token_encoding_uses_code_points() {
        assert_eq!(encode_text("ab"), vec![97, 98]);
        assert_eq!(encode_text("ü"), vec![252]);
        assert!(encode_text("").is_empty(), "no text, no tokens");
    }

    #[test]
    fn ids_and_rewards_fall_back_deterministically() {
        let mut msg = message(json!([]));
        assert_eq!(sample_task_id(&msg), "t-1");
        msg.origin_task_id = Some("o-0".to_owned());
        assert_eq!(sample_task_id(&msg), "o-0");

        msg.global_reward = f64::NAN;
        assert_eq!(finite_reward(&msg), 0.0);
        msg.global_reward = 0.5;
        assert_eq!(finite_reward(&msg), 0.5);

        assert_eq!(mean(std::iter::empty()), 0.0);
        assert_eq!(mean([1.0, 2.0].into_iter()), 1.5);
        assert_eq!(mean([1.0, f64::NAN].into_iter()), 1.0);
    }
}
