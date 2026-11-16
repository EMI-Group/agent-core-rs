//! Rollout encoders (Python `coordinator/encoding.py`).
//!
//! [`TurnEncoder`] turns a [`RolloutMessage`] into the training samples the batch
//! builder consumes. Two modes exist, exactly as in Python:
//!
//! - **per-turn** ([`TurnEncoder::build`]): every turn is one sample, and every
//!   sample of a rollout carries the *same* global reward, so GRPO sees reward
//!   variance across rollouts instead of artificial variance across turns;
//! - **whole-trajectory** ([`TurnEncoder::build_whole_trajectory`]): the whole
//!   conversation is one sample whose `loss_mask` marks model tokens (`1`) and
//!   environment tokens (`0`).
//!
//! Python encodes text with `tokenizer.apply_chat_template` and falls back to
//! `tokenizer.encode`. No tokenizer ships with this workspace, so the fallback
//! uses the deterministic character-code encoding documented on
//! `crate::offline::encode_text`; precomputed `input_prompt_ids` /
//! `output_response_ids` are used unchanged when both are present, which is the
//! fast path Python takes first.
//!
//! A turn that cannot be encoded drops the whole rollout (Python logs the
//! failure and returns an empty list). Dropping is reported in
//! [`EncodingOutcome::dropped_reason`] instead of being swallowed.

use crate::rl::{Rollout, RolloutMessage, RolloutWithReward};

use super::{finite_reward, rollout_turns, sample_task_id, turn_prompt_ids, turn_response_ids};

/// The outcome of encoding one rollout message.
///
/// `samples` is empty when the rollout was dropped; `dropped_reason` then names
/// the turn and the problem. An empty rollout (no turns) yields no samples and
/// no reason: there is nothing to encode and nothing failed.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct EncodingOutcome {
    /// The samples the rollout produced.
    pub samples: Vec<RolloutWithReward>,
    /// Why the rollout was dropped, when it was.
    pub dropped_reason: Option<String>,
}

impl EncodingOutcome {
    /// Whether the rollout was dropped.
    #[must_use]
    pub fn is_dropped(&self) -> bool {
        self.dropped_reason.is_some()
    }

    /// Number of produced samples.
    #[must_use]
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Whether no sample was produced.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    fn dropped(reason: impl Into<String>) -> Self {
        Self {
            samples: Vec::new(),
            dropped_reason: Some(reason.into()),
        }
    }
}

/// Encodes rollout messages into reward-annotated training samples.
///
/// The encoder is stateless and cheap to clone; it holds no tokenizer.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TurnEncoder;

impl TurnEncoder {
    /// Creates the encoder.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Builds one sample per turn, all sharing the rollout's global reward.
    ///
    /// `turn_id` is the position inside the rollout, `n_turns` its length, and
    /// `group_id` the rollout's `origin_task_id` — every rollout of one prompt
    /// therefore lands in one GRPO group. The per-turn mode sets an empty
    /// `loss_mask`, which is Python's `None`; the batch builder reads an empty
    /// mask as "no mask" and the whole response becomes trainable.
    #[must_use]
    pub fn build(&self, message: &RolloutMessage) -> EncodingOutcome {
        let turns = match rollout_turns(message) {
            Ok(turns) => turns,
            Err(error) => return EncodingOutcome::dropped(error.to_string()),
        };
        if turns.is_empty() {
            return EncodingOutcome::default();
        }
        let reward = finite_reward(message);
        let task_id = sample_task_id(message);
        let n_turns = turns.len() as u64;
        let mut samples = Vec::with_capacity(turns.len());

        for (index, turn) in turns.iter().enumerate() {
            let prompt_ids = turn_prompt_ids(turn);
            let response_ids = turn_response_ids(turn);
            if prompt_ids.is_empty() || response_ids.is_empty() {
                return EncodingOutcome::dropped(format!(
                    "turn {index} of rollout {} has no encodable {} ids",
                    message.rollout_id,
                    if prompt_ids.is_empty() {
                        "prompt"
                    } else {
                        "response"
                    }
                ));
            }
            samples.push(RolloutWithReward {
                turn_id: index as u64,
                task_id: task_id.clone(),
                rollout_id: message.rollout_id.clone(),
                input_prompt_ids: prompt_ids,
                output_response_ids: response_ids,
                reward,
                n_turns,
                loss_mask: Vec::new(),
                group_id: message.origin_task_id.clone(),
            });
        }
        EncodingOutcome {
            samples,
            dropped_reason: None,
        }
    }

    /// Builds one whole-trajectory sample, or falls back to [`Self::build`].
    ///
    /// Fallbacks mirror Python: a rollout with no turns yields nothing, a
    /// single-turn rollout is encoded per turn, and an encoding failure drops
    /// the rollout (Python falls back to the per-turn build there; with one
    /// turn both paths are the same work, and with several the per-turn path
    /// would silently change the sample shape, so dropping is the honest
    /// outcome and the reason is reported).
    #[must_use]
    pub fn build_whole_trajectory(&self, message: &RolloutMessage) -> EncodingOutcome {
        let turns = match rollout_turns(message) {
            Ok(turns) => turns,
            Err(error) => return EncodingOutcome::dropped(error.to_string()),
        };
        if turns.is_empty() {
            return EncodingOutcome::default();
        }
        if turns.len() == 1 {
            return self.build(message);
        }

        let prompt_ids = turn_prompt_ids(&turns[0]);
        if prompt_ids.is_empty() {
            return EncodingOutcome::dropped(format!(
                "rollout {} has no encodable initial prompt",
                message.rollout_id
            ));
        }

        let mut response_ids: Vec<u32> = Vec::new();
        let mut loss_mask: Vec<u8> = Vec::new();
        let base = prompt_ids.len();

        for (index, turn) in turns.iter().enumerate() {
            let turn_prompt = turn_prompt_ids(turn);
            let turn_response = turn_response_ids(turn);
            if turn_response.is_empty() {
                return EncodingOutcome::dropped(format!(
                    "turn {index} of rollout {} has no encodable response ids",
                    message.rollout_id
                ));
            }
            // Python derives the boundaries from chat-template lengths:
            // `start = n_before - n_prompt`, `end = n_after - n_prompt`, where
            // `n_before`/`n_after` count the conversation before/after the turn.
            // The token-id projection of that arithmetic is "the turn's prompt
            // beyond the initial prompt", which is the environment text emitted
            // between two model turns.
            let start = turn_prompt.len().saturating_sub(base);
            if response_ids.len() < start {
                // Emit the environment tokens the turn's prompt added beyond
                // the part of the conversation already projected: they sit
                // right after the initial prompt plus what was emitted so far.
                let emitted = response_ids.len();
                let from = (base + emitted).min(turn_prompt.len());
                if from < turn_prompt.len() {
                    response_ids.extend_from_slice(&turn_prompt[from..]);
                    loss_mask.resize(response_ids.len(), 0);
                }
            }
            // Defensive: a producer whose prompt arrays do not extend each other
            // leaves a hole; the pad token fills it and stays masked out instead
            // of shifting the sample.
            if response_ids.len() < start {
                response_ids.resize(start, 0);
                loss_mask.resize(start, 0);
            }
            let end = start + turn_response.len();
            if response_ids.len() < end {
                response_ids.resize(end, 0);
                loss_mask.resize(end, 0);
            }
            for (offset, token) in turn_response.iter().enumerate() {
                response_ids[start + offset] = *token;
                loss_mask[start + offset] = 1;
            }
        }

        if response_ids.is_empty() {
            return EncodingOutcome::dropped(format!(
                "rollout {} has no encodable response tokens",
                message.rollout_id
            ));
        }

        EncodingOutcome {
            samples: vec![RolloutWithReward {
                turn_id: 0,
                task_id: sample_task_id(message),
                rollout_id: message.rollout_id.clone(),
                input_prompt_ids: prompt_ids,
                output_response_ids: response_ids,
                reward: finite_reward(message),
                n_turns: turns.len() as u64,
                loss_mask,
                group_id: message.origin_task_id.clone(),
            }],
            dropped_reason: None,
        }
    }

    /// The number of turns a message carries, without encoding it.
    ///
    /// # Errors
    ///
    /// [`openjiuwen_common::error::Error::validation`] when `rollout_info` is
    /// malformed.
    pub fn turn_count(&self, message: &RolloutMessage) -> crate::Result<usize> {
        Ok(rollout_turns(message)?.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn message(turns: Value, reward: f64) -> RolloutMessage {
        RolloutMessage {
            task_id: "t-1".to_owned(),
            origin_task_id: Some("origin-1".to_owned()),
            rollout_id: "r-1".to_owned(),
            start_time: None,
            end_time: None,
            rollout_info: turns,
            reward_list: vec![reward],
            global_reward: reward,
            turn_count: 3,
            round_num: 0,
        }
    }

    fn turn(prompt: &[u32], response: &[u32]) -> Value {
        json!({
            "input_prompt_ids": prompt,
            "output_response_ids": response,
        })
    }

    #[test]
    fn per_turn_samples_share_the_global_reward() {
        let msg = message(
            json!([
                turn(&[1, 2, 3], &[4, 5]),
                turn(&[1, 2, 3, 4, 5], &[6]),
                turn(&[1, 2, 3, 4, 5, 6], &[7, 8]),
            ]),
            0.75,
        );
        let outcome = TurnEncoder::new().build(&msg);
        assert!(!outcome.is_dropped());
        assert_eq!(outcome.len(), 3);
        for (index, sample) in outcome.samples.iter().enumerate() {
            assert_eq!(sample.turn_id, index as u64);
            assert_eq!(sample.reward, 0.75);
            assert_eq!(sample.n_turns, 3);
            assert_eq!(sample.task_id, "origin-1");
            assert_eq!(sample.rollout_id, "r-1");
            assert_eq!(sample.group_id.as_deref(), Some("origin-1"));
            assert!(sample.loss_mask.is_empty());
        }
        assert_eq!(outcome.samples[1].input_prompt_ids, vec![1, 2, 3, 4, 5]);
        assert_eq!(outcome.samples[2].output_response_ids, vec![7, 8]);
    }

    #[test]
    fn text_payloads_fall_back_to_character_codes() {
        let msg = message(
            json!([{
                "input_prompt": {"message": [{"role": "user", "content": "hi"}]},
                "output_response": {"content": "ok"},
            }]),
            1.0,
        );
        let outcome = TurnEncoder::new().build(&msg);
        assert_eq!(outcome.len(), 1);
        assert_eq!(outcome.samples[0].input_prompt_ids, vec![104, 105]);
        assert_eq!(outcome.samples[0].output_response_ids, vec![111, 107]);
    }

    #[test]
    fn empty_rollouts_produce_nothing_without_a_reason() {
        let outcome = TurnEncoder::new().build(&message(json!([]), 1.0));
        assert!(outcome.is_empty());
        assert!(!outcome.is_dropped());

        let outcome = TurnEncoder::new().build_whole_trajectory(&message(json!([]), 1.0));
        assert!(outcome.is_empty());
        assert!(!outcome.is_dropped());
    }

    #[test]
    fn a_turn_that_cannot_be_encoded_drops_the_whole_rollout() {
        let msg = message(
            json!([turn(&[1], &[2]), {"input_prompt_ids": [1], "output_response_ids": []}]),
            1.0,
        );
        let outcome = TurnEncoder::new().build(&msg);
        assert!(outcome.is_dropped());
        assert!(outcome.is_empty());
        let reason = outcome.dropped_reason.expect("reason");
        assert!(reason.contains("turn 1 of rollout r-1"));
        assert!(reason.contains("response"));

        let msg = message(json!([{"input_prompt_ids": [], "output_response_ids": [2]}]), 1.0);
        let outcome = TurnEncoder::new().build(&msg);
        assert!(outcome.dropped_reason.expect("reason").contains("prompt"));

        let error = TurnEncoder::new().build(&message(json!({"turns": []}), 1.0));
        assert!(error.is_dropped());
    }

    #[test]
    fn non_finite_rewards_become_zero() {
        let msg = message(json!([turn(&[1], &[2])]), f64::INFINITY);
        let outcome = TurnEncoder::new().build(&msg);
        assert_eq!(outcome.samples[0].reward, 0.0);
    }

    #[test]
    fn a_single_turn_whole_trajectory_equals_the_per_turn_build() {
        let msg = message(json!([turn(&[1, 2], &[3])]), 0.5);
        let encoder = TurnEncoder::new();
        assert_eq!(
            encoder.build_whole_trajectory(&msg),
            encoder.build(&msg)
        );
    }

    #[test]
    fn whole_trajectory_marks_environment_tokens_with_zero() {
        let msg = message(
            json!([
                turn(&[1, 2], &[3, 4]),
                // The second prompt repeats the conversation and adds two
                // environment tokens (5, 6).
                turn(&[1, 2, 3, 4, 5, 6], &[7]),
                // The third prompt extends by one more environment token (8).
                turn(&[1, 2, 3, 4, 5, 6, 7, 8], &[9, 10]),
            ]),
            1.0,
        );
        let outcome = TurnEncoder::new().build_whole_trajectory(&msg);
        assert!(!outcome.is_dropped());
        assert_eq!(outcome.len(), 1);
        let sample = &outcome.samples[0];
        assert_eq!(sample.input_prompt_ids, vec![1, 2]);
        assert_eq!(sample.output_response_ids, vec![3, 4, 5, 6, 7, 8, 9, 10]);
        assert_eq!(sample.loss_mask, vec![1, 1, 0, 0, 1, 0, 1, 1]);
        assert_eq!(sample.n_turns, 3);
        assert_eq!(sample.reward, 1.0);
        assert_eq!(sample.turn_id, 0);
    }

    #[test]
    fn whole_trajectory_rewrites_a_window_whose_prompt_does_not_extend() {
        // The second prompt carries no ids, so it does not extend the
        // conversation: its turn window is recomputed from the prompt length
        // (zero offset) and overwrites the first response token, exactly like
        // Python's `loss_mask[start..end]` arithmetic.
        let msg = message(
            json!([
                turn(&[1], &[2]),
                {"input_prompt_ids": [], "output_response_ids": [3]},
            ]),
            1.0,
        );
        let outcome = TurnEncoder::new().build_whole_trajectory(&msg);
        let sample = &outcome.samples[0];
        assert_eq!(sample.input_prompt_ids, vec![1]);
        assert_eq!(sample.output_response_ids, vec![3]);
        assert_eq!(sample.loss_mask, vec![1]);
        assert_eq!(sample.n_turns, 2);
    }

    #[test]
    fn whole_trajectory_drops_a_rollout_without_a_response() {
        let msg = message(
            json!([turn(&[1], &[2]), {"input_prompt_ids": [1, 2], "output_response_ids": []}]),
            1.0,
        );
        let outcome = TurnEncoder::new().build_whole_trajectory(&msg);
        assert!(outcome.is_dropped());
        assert!(outcome
            .dropped_reason
            .expect("reason")
            .contains("turn 1 of rollout r-1"));
    }

    #[test]
    fn turn_count_reports_the_decoded_length() {
        let encoder = TurnEncoder::new();
        let msg = message(json!([turn(&[1], &[2]), turn(&[1, 2], &[3])]), 1.0);
        assert_eq!(encoder.turn_count(&msg).expect("count"), 2);
        assert!(encoder.turn_count(&message(json!(5), 1.0)).is_err());
    }

}
