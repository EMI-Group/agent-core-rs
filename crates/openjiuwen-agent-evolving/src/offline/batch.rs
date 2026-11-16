//! Batch planning: the portable replacement for Python's `TensorDict` builder
//! (Python `coordinator/batch_builder.py`).
//!
//! Python builds padded `torch` tensors, a `TensorDict` and a bf16 score tensor.
//! None of that is portable, so this module keeps the *arithmetic* of batch
//! construction and drops the tensors: [`BatchComponents`] holds the padded rows
//! in a flat, serializable shape, and [`BatchPlan`] adds the derived
//! concatenated sequences, position ids and token-level scores a trainer needs.
//!
//! Three rules are copied verbatim from Python because a silent change would
//! alter the training signal:
//!
//! - prompts are **left** padded and truncated to their last `max_prompt_length`
//!   tokens (the end of the prompt is what the model continues from);
//! - responses are **right** padded and truncated to their first
//!   `max_response_length` tokens;
//! - a sample whose prompt exceeds `max_prompt_length` is flagged in `is_drop`
//!   instead of being silently shortened (verl drops it downstream).
//!
//! Token ids are `u32` (the encoding in [`super::encoding`] is code points),
//! attention and loss masks are `u8` (0/1), and rewards are `f64`.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use openjiuwen_common::error::{Error, Result};

use crate::rl::{RlBatch, RlSample, RolloutWithReward};

/// The lengths and pad token a batch is built with.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct BatchConfig {
    /// Maximum prompt length; longer prompts are truncated and flagged.
    pub max_prompt_length: usize,
    /// Maximum response length; longer responses are truncated.
    pub max_response_length: usize,
    /// The token id used for padding.
    pub pad_token_id: u32,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_prompt_length: 1024,
            max_response_length: 1024,
            pad_token_id: 0,
        }
    }
}

impl BatchConfig {
    /// Creates a configuration.
    #[must_use]
    pub const fn new(max_prompt_length: usize, max_response_length: usize, pad_token_id: u32) -> Self {
        Self {
            max_prompt_length,
            max_response_length,
            pad_token_id,
        }
    }

    /// Validates that both lengths are positive.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] when a length is zero: a zero-length prompt or
    /// response would produce an empty tensor row, which Python crashes on
    /// (`ids[-0:]` is the whole list, not an empty one).
    pub fn validate(&self) -> Result<()> {
        if self.max_prompt_length == 0 {
            return Err(Error::validation(
                "BatchConfig max_prompt_length must be at least 1",
            ));
        }
        if self.max_response_length == 0 {
            return Err(Error::validation(
                "BatchConfig max_response_length must be at least 1",
            ));
        }
        Ok(())
    }
}

/// The padded rows of a batch, in insertion order.
///
/// Every vector has one entry per sample. `loss_masks` is `None` for samples
/// without a mask (the per-turn mode), which is Python's `None`; an empty
/// `RolloutWithReward::loss_mask` is read as "no mask".
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct BatchComponents {
    /// Left-padded prompt ids, `max_prompt_length` wide.
    pub input_ids: Vec<Vec<u32>>,
    /// Prompt attention masks, `max_prompt_length` wide.
    pub input_attention_mask: Vec<Vec<u8>>,
    /// Right-padded response ids, `max_response_length` wide.
    pub response_ids: Vec<Vec<u32>>,
    /// Response attention masks, `max_response_length` wide.
    pub response_attention_mask: Vec<Vec<u8>>,
    /// One reward per sample.
    pub rewards: Vec<f64>,
    /// The turn index of each sample inside its rollout.
    pub turn_indices: Vec<u64>,
    /// Whether the sample's prompt exceeded `max_prompt_length`.
    pub is_drop: Vec<bool>,
    /// The rollout group (`origin_task_id`) each sample belongs to.
    pub data_ids: Vec<String>,
    /// Optional per-token loss masks (whole-trajectory mode).
    pub loss_masks: Vec<Option<Vec<u8>>>,
    /// The number of turns in each sample's rollout.
    pub n_turns_list: Vec<u64>,
    /// How many samples had their response truncated.
    pub truncation_count: usize,
}

impl BatchComponents {
    /// Number of samples in the batch.
    #[must_use]
    pub fn len(&self) -> usize {
        self.input_ids.len()
    }

    /// Whether the batch holds no samples.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.input_ids.is_empty()
    }

    /// Number of samples flagged for dropping.
    #[must_use]
    pub fn drop_count(&self) -> usize {
        self.is_drop.iter().filter(|drop| **drop).count()
    }
}

/// A fully planned batch: the components plus the derived per-sample rows.
///
/// `sequence_ids`, `attention_mask` and `position_ids` are the prompt and
/// response rows concatenated (Python's `seq_ids`/`attention_mask`/
/// `position_ids`), `token_level_scores` is Python's `token_level_scores`
/// (the reward placed at the last attended position and sliced to the response
/// window), and `actor_loss_mask` exists only when at least one sample carries a
/// loss mask — Python's `actor_loss_mask` is added under the same condition.
#[derive(Debug, Clone, PartialEq)]
pub struct BatchPlan {
    /// The configuration the plan was built with.
    pub config: BatchConfig,
    /// The padded components.
    pub components: BatchComponents,
    /// Prompt plus response ids, one row per sample.
    pub sequence_ids: Vec<Vec<u32>>,
    /// Prompt plus response attention masks.
    pub attention_mask: Vec<Vec<u8>>,
    /// Positions of the attended tokens, zero-based and clamped at zero.
    pub position_ids: Vec<Vec<u32>>,
    /// The reward at the last attended position of each row, response window only.
    pub token_level_scores: Vec<Vec<f64>>,
    /// Per-response-token trainable mask, present only when a sample has one.
    pub actor_loss_mask: Option<Vec<Vec<u8>>>,
}

impl BatchPlan {
    /// Number of samples in the plan.
    #[must_use]
    pub fn sample_count(&self) -> usize {
        self.components.len()
    }

    /// Whether the plan holds no samples.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.components.is_empty()
    }

    /// Mean reward over the samples; `0.0` when empty.
    #[must_use]
    pub fn reward_mean(&self) -> f64 {
        super::mean(self.components.rewards.iter().copied())
    }

    /// The `is_drop` flags of the plan.
    #[must_use]
    pub fn drop_mask(&self) -> Vec<bool> {
        self.components.is_drop.clone()
    }

    /// Number of samples whose prompt exceeded `max_prompt_length`.
    #[must_use]
    pub fn drop_count(&self) -> usize {
        self.components.drop_count()
    }

    /// Projects the plan onto the crate's [`RlBatch`] so a [`crate::rl::Policy`]
    /// can consume it.
    ///
    /// Prompts and responses are rendered as space-separated decimal token ids,
    /// the same projection [`crate::rl::RolloutLog::to_samples`] uses; the
    /// rollout group id and the reward are preserved. Dropped samples are
    /// excluded, because a downstream policy must not train on a truncated
    /// prompt.
    #[must_use]
    pub fn to_rl_batch(&self, batch_id: impl Into<String>) -> RlBatch {
        let mut samples = Vec::with_capacity(self.sample_count());
        for (index, reward) in self.components.rewards.iter().enumerate() {
            if self.components.is_drop.get(index).copied().unwrap_or(false) {
                continue;
            }
            let prompt = render_tokens(&self.components.input_ids[index]);
            let response = render_tokens(&self.components.response_ids[index]);
            let mut metadata = BTreeMap::new();
            metadata.insert(
                "turn_id".to_owned(),
                Value::from(self.components.turn_indices[index]),
            );
            metadata.insert(
                "n_turns".to_owned(),
                Value::from(self.components.n_turns_list[index]),
            );
            let sample_id = format!(
                "{}-{index}",
                self.components.data_ids[index]
            );
            samples.push(RlSample {
                sample_id,
                prompt: if prompt.is_empty() {
                    self.components.data_ids[index].clone()
                } else {
                    prompt
                },
                response,
                reward: if reward.is_finite() { *reward } else { 0.0 },
                group_id: Some(self.components.data_ids[index].clone()),
                metadata,
            });
        }
        RlBatch::from_samples(batch_id, samples)
    }
}

fn render_tokens(ids: &[u32]) -> String {
    ids.iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(" ")
}

/// Left-pads `ids` to `max_length` and returns the attention mask.
///
/// When `ids` is longer it is truncated to its **last** `max_length` tokens (the
/// end of a prompt is what the model continues from) and the whole mask is `1`.
/// A `max_length` of zero yields two empty vectors; Python's `ids[-0:]` would
/// return the whole list there, which is not a usable batch row.
#[must_use]
pub fn left_padded_ids_and_attention_mask(
    ids: &[u32],
    max_length: usize,
    pad_token_id: u32,
) -> (Vec<u32>, Vec<u8>) {
    if max_length == 0 {
        return (Vec::new(), Vec::new());
    }
    if ids.len() >= max_length {
        return (ids[ids.len() - max_length..].to_vec(), vec![1; max_length]);
    }
    let pad_len = max_length - ids.len();
    let mut padded = vec![pad_token_id; pad_len];
    padded.extend_from_slice(ids);
    let mut mask = vec![0; pad_len];
    mask.extend(std::iter::repeat_n(1, ids.len()));
    (padded, mask)
}

/// Right-pads `ids` to `max_length` and returns the attention mask.
///
/// When `ids` is longer it is truncated to its **first** `max_length` tokens.
/// A `max_length` of zero yields two empty vectors.
#[must_use]
pub fn right_padded_ids_and_attention_mask(
    ids: &[u32],
    max_length: usize,
    pad_token_id: u32,
) -> (Vec<u32>, Vec<u8>) {
    if max_length == 0 {
        return (Vec::new(), Vec::new());
    }
    if ids.len() >= max_length {
        return (ids[..max_length].to_vec(), vec![1; max_length]);
    }
    let mut padded = ids.to_vec();
    padded.resize(max_length, pad_token_id);
    let mut mask = vec![1; ids.len()];
    mask.resize(max_length, 0);
    (padded, mask)
}

/// Truncates a prompt/response pair and reports whether the prompt must be dropped.
///
/// A prompt longer than `max_prompt_length` is cut to its first
/// `max_prompt_length` tokens **and** flagged, because losing prompt text changes
/// the task; a long response is cut silently because the reward already covers
/// the answer. Python truncates both from the front; the response truncation is
/// the same, and the prompt keeps the same rule so the flag stays comparable.
#[must_use]
pub fn truncate_prompt_and_response(
    prompt_ids: &[u32],
    response_ids: &[u32],
    max_prompt_length: usize,
    max_response_length: usize,
) -> (Vec<u32>, Vec<u32>, bool) {
    let is_drop = prompt_ids.len() > max_prompt_length;
    let prompt = if is_drop {
        prompt_ids[..max_prompt_length].to_vec()
    } else {
        prompt_ids.to_vec()
    };
    let response = if response_ids.len() > max_response_length {
        response_ids[..max_response_length].to_vec()
    } else {
        response_ids.to_vec()
    };
    (prompt, response, is_drop)
}

/// Places each reward at the last attended position and keeps the response window.
///
/// `attention_mask` and `position_ids` are the concatenated prompt+response rows;
/// the end-of-sequence position is the last index whose mask is `1` (Python's
/// `argmax(position_ids * attention_mask)`, which peaks at the final attended
/// token). The result is one row per sample, sliced to the last
/// `response_length` entries.
///
/// # Errors
///
/// [`Error::validation`] when the three slices disagree on row count or a row's
/// masks disagree on length.
pub fn token_level_scores(
    attention_mask: &[Vec<u8>],
    position_ids: &[Vec<u32>],
    scores: &[f64],
    response_length: usize,
) -> Result<Vec<Vec<f64>>> {
    if attention_mask.len() != scores.len() || position_ids.len() != scores.len() {
        return Err(Error::validation(format!(
            "token_level_scores needs one row per score, got {} masks, {} position rows and {} scores",
            attention_mask.len(),
            position_ids.len(),
            scores.len()
        )));
    }
    let mut rows = Vec::with_capacity(scores.len());
    for (index, score) in scores.iter().enumerate() {
        let mask = &attention_mask[index];
        let positions = &position_ids[index];
        if mask.len() != positions.len() {
            return Err(Error::validation(format!(
                "token_level_scores row {index} has {} attention entries but {} position entries",
                mask.len(),
                positions.len()
            )));
        }
        let mut row = vec![0.0; mask.len()];
        if let Some(eos) = mask.iter().rposition(|value| *value != 0) {
            row[eos] = if score.is_finite() { *score } else { 0.0 };
        }
        let start = row.len().saturating_sub(response_length);
        rows.push(row[start..].to_vec());
    }
    Ok(rows)
}

/// Builds batch plans from reward-annotated rollouts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchBuilder {
    config: BatchConfig,
}

impl BatchBuilder {
    /// Creates a builder for `config`.
    #[must_use]
    pub const fn new(config: BatchConfig) -> Self {
        Self { config }
    }

    /// The configuration the builder uses.
    #[must_use]
    pub const fn config(&self) -> BatchConfig {
        self.config
    }

    /// Pads, truncates and groups the rollouts into components.
    ///
    /// Iteration is by [`BTreeMap`] key, so the sample order does not depend on
    /// the order workers finished in.
    #[must_use]
    pub fn generate_components(
        &self,
        rollouts: &BTreeMap<String, Vec<RolloutWithReward>>,
    ) -> BatchComponents {
        let mut components = BatchComponents::default();
        for (data_id, list) in rollouts {
            for rollout in list {
                self.push_sample(&mut components, data_id, rollout);
            }
        }
        components
    }

    fn push_sample(
        &self,
        components: &mut BatchComponents,
        data_id: &str,
        rollout: &RolloutWithReward,
    ) {
        let (prompt, response, is_drop) = truncate_prompt_and_response(
            &rollout.input_prompt_ids,
            &rollout.output_response_ids,
            self.config.max_prompt_length,
            self.config.max_response_length,
        );
        let truncated = rollout.output_response_ids.len() > self.config.max_response_length;

        let (input_ids, input_mask) = left_padded_ids_and_attention_mask(
            &prompt,
            self.config.max_prompt_length,
            self.config.pad_token_id,
        );
        let (response_ids, response_mask) = right_padded_ids_and_attention_mask(
            &response,
            self.config.max_response_length,
            self.config.pad_token_id,
        );

        let loss_mask = if rollout.loss_mask.is_empty() {
            None
        } else {
            let mut mask = rollout
                .loss_mask
                .iter()
                .take(self.config.max_response_length)
                .copied()
                .collect::<Vec<u8>>();
            mask.resize(self.config.max_response_length, 0);
            Some(mask)
        };

        components.input_ids.push(input_ids);
        components.input_attention_mask.push(input_mask);
        components.response_ids.push(response_ids);
        components
            .response_attention_mask
            .push(response_mask);
        components.rewards.push(if rollout.reward.is_finite() {
            rollout.reward
        } else {
            0.0
        });
        components.turn_indices.push(rollout.turn_id);
        components.is_drop.push(is_drop);
        components.data_ids.push(data_id.to_owned());
        components.loss_masks.push(loss_mask);
        components.n_turns_list.push(rollout.n_turns);
        components.truncation_count += usize::from(truncated);
    }

    /// Generates the components and assembles them.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] for an empty batch (Python raises
    /// `AGENT_RL_ROLLOUT_BATCH_EXECUTION_ERROR` there) or a configuration with
    /// zero-length rows.
    pub fn assemble_batch(
        &self,
        rollouts: &BTreeMap<String, Vec<RolloutWithReward>>,
    ) -> Result<BatchPlan> {
        self.assemble(self.generate_components(rollouts))
    }

    /// Derives the plan rows from prepared components.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] when there are no samples, the configuration has a
    /// zero length, or the component vectors disagree on length.
    pub fn assemble(&self, components: BatchComponents) -> Result<BatchPlan> {
        self.config.validate()?;
        if components.is_empty() {
            return Err(Error::validation(
                "0 samples collected after rollout; refusing to assemble an empty batch",
            ));
        }
        let expected = components.len();
        for (name, actual) in [
            ("input_attention_mask", components.input_attention_mask.len()),
            ("response_ids", components.response_ids.len()),
            (
                "response_attention_mask",
                components.response_attention_mask.len(),
            ),
            ("rewards", components.rewards.len()),
            ("turn_indices", components.turn_indices.len()),
            ("is_drop", components.is_drop.len()),
            ("data_ids", components.data_ids.len()),
            ("loss_masks", components.loss_masks.len()),
            ("n_turns_list", components.n_turns_list.len()),
        ] {
            if actual != expected {
                return Err(Error::validation(format!(
                    "batch component {name} has {actual} entries but input_ids has {expected}"
                )));
            }
        }

        let mut sequence_ids = Vec::with_capacity(expected);
        let mut attention_mask = Vec::with_capacity(expected);
        let mut position_ids = Vec::with_capacity(expected);
        for index in 0..expected {
            let mut ids = components.input_ids[index].clone();
            ids.extend_from_slice(&components.response_ids[index]);
            let mut mask = components.input_attention_mask[index].clone();
            mask.extend_from_slice(&components.response_attention_mask[index]);
            let mut positions = Vec::with_capacity(mask.len());
            let mut running: u32 = 0;
            for value in &mask {
                if *value != 0 {
                    running = running.saturating_add(1);
                }
                positions.push(running.saturating_sub(1));
            }
            sequence_ids.push(ids);
            attention_mask.push(mask);
            position_ids.push(positions);
        }

        let token_level_scores = token_level_scores(
            &attention_mask,
            &position_ids,
            &components.rewards,
            self.config.max_response_length,
        )?;

        let has_loss_mask = components.loss_masks.iter().any(Option::is_some);
        let actor_loss_mask = has_loss_mask.then(|| {
            components
                .loss_masks
                .iter()
                .map(|mask| {
                    mask.clone()
                        .unwrap_or_else(|| vec![1; self.config.max_response_length])
                })
                .collect()
        });

        Ok(BatchPlan {
            config: self.config,
            components,
            sequence_ids,
            attention_mask,
            position_ids,
            token_level_scores,
            actor_loss_mask,
        })
    }
}

/// Assembles a plan from rollouts with an explicit configuration.
///
/// # Errors
///
/// See [`BatchBuilder::assemble_batch`].
pub fn assemble_batch(
    config: &BatchConfig,
    rollouts: &BTreeMap<String, Vec<RolloutWithReward>>,
) -> Result<BatchPlan> {
    BatchBuilder::new(*config).assemble_batch(rollouts)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rollout(prompt: &[u32], response: &[u32], reward: f64) -> RolloutWithReward {
        RolloutWithReward {
            turn_id: 0,
            task_id: "t-1".to_owned(),
            rollout_id: "r-1".to_owned(),
            input_prompt_ids: prompt.to_vec(),
            output_response_ids: response.to_vec(),
            reward,
            n_turns: 1,
            loss_mask: Vec::new(),
            group_id: Some("g".to_owned()),
        }
    }

    fn cache(entries: &[(&str, Vec<RolloutWithReward>)]) -> BTreeMap<String, Vec<RolloutWithReward>> {
        entries
            .iter()
            .map(|(key, value)| ((*key).to_owned(), value.clone()))
            .collect()
    }

    #[test]
    fn left_padding_keeps_the_prompt_tail() {
        let (ids, mask) = left_padded_ids_and_attention_mask(&[7, 8], 4, 0);
        assert_eq!(ids, vec![0, 0, 7, 8]);
        assert_eq!(mask, vec![0, 0, 1, 1]);

        let (ids, mask) = left_padded_ids_and_attention_mask(&[1, 2, 3, 4, 5], 3, 0);
        assert_eq!(ids, vec![3, 4, 5]);
        assert_eq!(mask, vec![1, 1, 1]);

        let (ids, mask) = left_padded_ids_and_attention_mask(&[1], 1, 9);
        assert_eq!(ids, vec![1]);
        assert_eq!(mask, vec![1]);

        let (ids, mask) = left_padded_ids_and_attention_mask(&[1, 2], 0, 9);
        assert!(ids.is_empty());
        assert!(mask.is_empty());
    }

    #[test]
    fn right_padding_keeps_the_response_head() {
        let (ids, mask) = right_padded_ids_and_attention_mask(&[7, 8], 4, 5);
        assert_eq!(ids, vec![7, 8, 5, 5]);
        assert_eq!(mask, vec![1, 1, 0, 0]);

        let (ids, mask) = right_padded_ids_and_attention_mask(&[1, 2, 3, 4], 2, 0);
        assert_eq!(ids, vec![1, 2]);
        assert_eq!(mask, vec![1, 1]);

        let (ids, mask) = right_padded_ids_and_attention_mask(&[], 2, 9);
        assert_eq!(ids, vec![9, 9]);
        assert_eq!(mask, vec![0, 0]);

        assert!(right_padded_ids_and_attention_mask(&[1], 0, 0).0.is_empty());
    }

    #[test]
    fn truncation_flags_only_over_long_prompts() {
        let (prompt, response, is_drop) =
            truncate_prompt_and_response(&[1, 2, 3], &[4, 5, 6], 2, 2);
        assert_eq!(prompt, vec![1, 2]);
        assert_eq!(response, vec![4, 5]);
        assert!(is_drop);

        let (prompt, response, is_drop) =
            truncate_prompt_and_response(&[1], &[4, 5, 6], 2, 2);
        assert_eq!(prompt, vec![1]);
        assert_eq!(response, vec![4, 5]);
        assert!(!is_drop);

        let (prompt, response, is_drop) =
            truncate_prompt_and_response(&[1, 2], &[3], 2, 2);
        assert_eq!(prompt, vec![1, 2]);
        assert_eq!(response, vec![3]);
        assert!(!is_drop);
    }

    #[test]
    fn token_level_scores_land_on_the_last_attended_position() {
        let mask = vec![vec![0, 0, 1, 1, 1], vec![1, 1, 0, 0, 0]];
        let positions = vec![vec![0, 0, 0, 1, 2], vec![0, 1, 1, 1, 1]];
        let scores = token_level_scores(&mask, &positions, &[1.0, 0.5], 3).expect("rows");
        // Row 1: eos at index 4 -> score in the last of the 3 kept positions.
        assert_eq!(scores[0], vec![0.0, 0.0, 1.0]);
        // Row 2: eos at index 1 -> outside the response window.
        assert_eq!(scores[1], vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn token_level_scores_reject_mismatched_rows() {
        let error = token_level_scores(&[vec![1]], &[vec![0], vec![0]], &[1.0], 1)
            .expect_err("row mismatch");
        assert!(error.to_string().contains("one row per score"));

        let error = token_level_scores(&[vec![1, 0]], &[vec![0]], &[1.0], 1)
            .expect_err("width mismatch");
        assert!(error.to_string().contains("position entries"));
    }

    #[test]
    fn components_carry_padded_rows_and_flags() {
        let builder = BatchBuilder::new(BatchConfig::new(3, 2, 0));
        let rollouts = cache(&[
            ("a", vec![rollout(&[1, 2], &[3, 4, 5], 1.0)]),
            ("b", vec![rollout(&[1, 2, 3, 4], &[6], 0.0)]),
        ]);
        let components = builder.generate_components(&rollouts);
        assert_eq!(components.len(), 2);
        assert_eq!(components.input_ids[0], vec![0, 1, 2]);
        assert_eq!(components.input_attention_mask[0], vec![0, 1, 1]);
        assert_eq!(components.response_ids[0], vec![3, 4]);
        assert_eq!(components.response_attention_mask[0], vec![1, 1]);
        assert_eq!(components.truncation_count, 1);
        assert!(!components.is_drop[0]);
        assert_eq!(components.data_ids, vec!["a", "b"]);
        assert_eq!(components.rewards, vec![1.0, 0.0]);
        // The second prompt is longer than the limit: truncated and flagged.
        assert_eq!(components.input_ids[1], vec![1, 2, 3]);
        assert!(components.is_drop[1]);
        assert_eq!(components.drop_count(), 1);
        assert!(components.loss_masks.iter().all(Option::is_none));
    }

    #[test]
    fn assembling_derives_sequences_and_scores() {
        let builder = BatchBuilder::new(BatchConfig::new(2, 2, 0));
        let rollouts = cache(&[("a", vec![rollout(&[1, 2], &[3], 1.0)])]);
        let plan = builder.assemble_batch(&rollouts).expect("plan");
        assert_eq!(plan.sample_count(), 1);
        assert!(!plan.is_empty());
        assert_eq!(plan.sequence_ids[0], vec![1, 2, 3, 0]);
        assert_eq!(plan.attention_mask[0], vec![1, 1, 1, 0]);
        assert_eq!(plan.position_ids[0], vec![0, 1, 2, 2]);
        assert_eq!(plan.token_level_scores[0], vec![1.0, 0.0]);
        assert_eq!(plan.reward_mean(), 1.0);
        assert_eq!(plan.drop_mask(), vec![false]);
        assert!(plan.actor_loss_mask.is_none());

        let batch = plan.to_rl_batch("batch-0");
        assert_eq!(batch.num_samples(), 1);
        assert_eq!(batch.samples[0].prompt, "1 2");
        assert_eq!(batch.samples[0].response, "3 0");
        assert_eq!(batch.samples[0].reward, 1.0);
        assert_eq!(batch.samples[0].group_id.as_deref(), Some("a"));
        assert_eq!(batch.samples[0].metadata["n_turns"], Value::from(1));
    }

    #[test]
    fn actor_loss_mask_appears_only_with_a_mask() {
        let builder = BatchBuilder::new(BatchConfig::new(2, 3, 0));
        let mut masked = rollout(&[1], &[2, 3], 1.0);
        masked.loss_mask = vec![1, 0];
        let plan = builder
            .assemble_batch(&cache(&[("a", vec![masked])]))
            .expect("plan");
        let masks = plan.actor_loss_mask.expect("mask present");
        assert_eq!(masks[0], vec![1, 0, 0]);
    }

    #[test]
    fn empty_batches_are_rejected() {
        let builder = BatchBuilder::new(BatchConfig::default());
        let error = builder
            .assemble_batch(&BTreeMap::new())
            .expect_err("empty batch");
        assert!(error.to_string().contains("0 samples collected after rollout"));

        let error = assemble_batch(&BatchConfig::default(), &BTreeMap::new())
            .expect_err("empty batch");
        assert_eq!(error.kind(), openjiuwen_common::error::ErrorKind::Validation);
    }

    #[test]
    fn a_zero_length_configuration_is_rejected() {
        let builder = BatchBuilder::new(BatchConfig::new(0, 4, 0));
        let error = builder
            .assemble_batch(&cache(&[("a", vec![rollout(&[1], &[2], 1.0)])]))
            .expect_err("zero prompt length");
        assert!(error.to_string().contains("max_prompt_length"));

        let builder = BatchBuilder::new(BatchConfig::new(4, 0, 0));
        let error = builder
            .assemble_batch(&cache(&[("a", vec![rollout(&[1], &[2], 1.0)])]))
            .expect_err("zero response length");
        assert!(error.to_string().contains("max_response_length"));
    }

    #[test]
    fn mismatched_components_are_rejected() {
        let mut components = BatchComponents {
            input_ids: vec![vec![1]],
            input_attention_mask: vec![vec![1]],
            response_ids: vec![vec![2]],
            response_attention_mask: vec![vec![1]],
            rewards: vec![1.0],
            turn_indices: vec![0],
            is_drop: vec![false],
            data_ids: vec!["a".to_owned()],
            loss_masks: Vec::new(),
            n_turns_list: vec![1],
            truncation_count: 0,
        };
        let builder = BatchBuilder::new(BatchConfig::new(1, 1, 0));
        let error = builder
            .assemble(components.clone())
            .expect_err("loss_masks mismatch");
        assert!(error.to_string().contains("loss_masks"));

        components.loss_masks.push(None);
        let plan = builder.assemble(components).expect("plan");
        assert_eq!(plan.sample_count(), 1);
    }

    #[test]
    fn non_finite_rewards_are_coerced() {
        let builder = BatchBuilder::new(BatchConfig::new(2, 2, 0));
        let mut bad = rollout(&[1], &[2], f64::NAN);
        bad.turn_id = 3;
        bad.n_turns = 3;
        let plan = builder
            .assemble_batch(&cache(&[("a", vec![bad])]))
            .expect("plan");
        assert_eq!(plan.components.rewards[0], 0.0);
        assert_eq!(plan.components.turn_indices[0], 3);
        assert_eq!(plan.components.n_turns_list[0], 3);
        assert_eq!(plan.reward_mean(), 0.0);
    }

    #[test]
    fn dropped_samples_are_excluded_from_the_policy_batch() {
        let builder = BatchBuilder::new(BatchConfig::new(1, 1, 0));
        let plan = builder
            .assemble_batch(&cache(&[
                ("a", vec![rollout(&[1], &[2], 1.0)]),
                ("b", vec![rollout(&[1, 2], &[3], 1.0)]),
            ]))
            .expect("plan");
        assert_eq!(plan.drop_count(), 1);
        let batch = plan.to_rl_batch("b");
        assert_eq!(batch.num_samples(), 1);
        assert_eq!(batch.samples[0].sample_id, "a-0");
    }
}
