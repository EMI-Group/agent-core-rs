//! Deterministic sample and batch packing for the offline RL loop.
//!
//! Python builds `verl` `TensorDict` batches with a tokenizer and padding. That
//! stack is deferred, so this module keeps the portable part: validated samples,
//! a tokenizer-independent token estimate, and a deterministic packer that
//! chunks samples in input order and optionally truncates oversized ones.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use openjiuwen_common::error::{Error, Result};

/// Per-message overhead, in tokens, added by [`token_estimate`].
pub const MESSAGE_TOKEN_OVERHEAD: usize = 4;

/// Deterministic token-count heuristic.
///
/// One token per four Unicode scalar values, rounded up, minimum one for any
/// non-empty text, plus [`MESSAGE_TOKEN_OVERHEAD`]. Empty text counts as zero
/// tokens. The heuristic is tokenizer-independent so batch packing is
/// reproducible offline, without a model or the Python tokenizer stack.
#[must_use]
pub fn token_estimate(text: &str) -> usize {
    let chars = text.chars().count();
    if chars == 0 {
        return 0;
    }
    chars.div_ceil(4) + MESSAGE_TOKEN_OVERHEAD
}

fn truncate_to_budget(text: &str, budget: usize) -> String {
    if token_estimate(text) <= budget {
        return text.to_owned();
    }
    let chars = budget
        .saturating_sub(MESSAGE_TOKEN_OVERHEAD)
        .saturating_mul(4);
    text.chars().take(chars).collect()
}

fn fit_sample(sample: &RlSample, budget: usize) -> (RlSample, bool) {
    if sample.token_count() <= budget {
        return (sample.clone(), false);
    }
    let mut fitted = sample.clone();
    let response_budget = budget.saturating_sub(token_estimate(&fitted.prompt));
    fitted.response = truncate_to_budget(&fitted.response, response_budget);
    let prompt_budget = budget.saturating_sub(token_estimate(&fitted.response));
    fitted.prompt = truncate_to_budget(&fitted.prompt, prompt_budget);
    (fitted, true)
}

/// One reward-annotated prompt/response pair ready for packing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RlSample {
    /// Identifier of the sample; unique inside a rollout log.
    pub sample_id: String,
    /// Prompt text shown to the policy.
    pub prompt: String,
    /// Response text produced by the policy.
    pub response: String,
    /// Reward assigned to the response.
    pub reward: f64,
    /// Group the sample was rolled out with, when group-relative rewards apply.
    #[serde(default)]
    pub group_id: Option<String>,
    /// Free-form metadata carried through packing.
    #[serde(default)]
    pub metadata: BTreeMap<String, Value>,
}

impl RlSample {
    /// Creates a validated sample.
    pub fn new(
        sample_id: impl Into<String>,
        prompt: impl Into<String>,
        response: impl Into<String>,
        reward: f64,
    ) -> Result<Self> {
        let sample = Self {
            sample_id: sample_id.into(),
            prompt: prompt.into(),
            response: response.into(),
            reward,
            group_id: None,
            metadata: BTreeMap::new(),
        };
        sample.validate()?;
        Ok(sample)
    }

    /// Validates that the identifier and prompt are non-empty and the reward is finite.
    pub fn validate(&self) -> Result<()> {
        if self.sample_id.trim().is_empty() {
            return Err(Error::validation("RlSample sample_id must not be empty"));
        }
        if self.prompt.trim().is_empty() {
            return Err(Error::validation("RlSample prompt must not be empty"));
        }
        if !self.reward.is_finite() {
            return Err(Error::validation(format!(
                "RlSample reward must be finite, got {}",
                self.reward
            )));
        }
        Ok(())
    }

    /// Sets the group id used for group-relative reward normalisation.
    #[must_use]
    pub fn with_group_id(mut self, group_id: impl Into<String>) -> Self {
        self.group_id = Some(group_id.into());
        self
    }

    /// Adds one metadata entry.
    #[must_use]
    pub fn with_metadata(mut self, key: impl Into<String>, value: Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Estimated tokens of the prompt plus the response.
    #[must_use]
    pub fn token_count(&self) -> usize {
        token_estimate(&self.prompt) + token_estimate(&self.response)
    }
}

/// A packed batch of samples.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RlBatch {
    /// Identifier of the batch (`"{prefix}-{index}"` when produced by [`BatchPacker`]).
    pub batch_id: String,
    /// Samples in deterministic input order.
    #[serde(default)]
    pub samples: Vec<RlSample>,
    /// Sum of [`RlSample::token_count`] over the samples.
    #[serde(default)]
    pub total_tokens: usize,
    /// Number of samples that were truncated to satisfy the token budget.
    #[serde(default)]
    pub truncated: usize,
}

impl RlBatch {
    /// Builds a batch from samples, computing the token total.
    ///
    /// `truncated` starts at zero; packers set it from their own bookkeeping.
    #[must_use]
    pub fn from_samples(batch_id: impl Into<String>, samples: Vec<RlSample>) -> Self {
        let total_tokens = samples.iter().map(RlSample::token_count).sum();
        Self {
            batch_id: batch_id.into(),
            samples,
            total_tokens,
            truncated: 0,
        }
    }

    /// Number of samples in the batch.
    #[must_use]
    pub fn num_samples(&self) -> usize {
        self.samples.len()
    }

    /// Mean reward over the samples; `0.0` for an empty batch.
    #[must_use]
    pub fn mean_reward(&self) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }
        let total: f64 = self
            .samples
            .iter()
            .map(|sample| {
                if sample.reward.is_finite() {
                    sample.reward
                } else {
                    0.0
                }
            })
            .sum();
        total / self.samples.len() as f64
    }

    /// Whether the batch carries no samples.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}

/// Deterministic chunking of samples into batches.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct BatchPacker {
    /// Maximum samples per batch; values below one are clamped to one.
    pub max_samples: usize,
    /// Per-sample token budget; `None` disables truncation and splitting.
    pub max_tokens_per_sample: Option<usize>,
    /// Whether oversized samples are truncated instead of split into their own batch.
    pub truncate_long_samples: bool,
    /// Prefix of generated batch ids.
    pub batch_prefix: String,
}

impl Default for BatchPacker {
    fn default() -> Self {
        Self {
            max_samples: 8,
            max_tokens_per_sample: None,
            truncate_long_samples: true,
            batch_prefix: "batch".to_owned(),
        }
    }
}

impl BatchPacker {
    /// Creates the default packer.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the maximum number of samples per batch.
    #[must_use]
    pub fn with_max_samples(mut self, max_samples: usize) -> Self {
        self.max_samples = max_samples;
        self
    }

    /// Sets the per-sample token budget.
    #[must_use]
    pub fn with_max_tokens_per_sample(mut self, max_tokens: usize) -> Self {
        self.max_tokens_per_sample = Some(max_tokens);
        self
    }

    /// Chooses whether oversized samples are truncated or split into their own batch.
    #[must_use]
    pub fn with_truncate_long_samples(mut self, truncate: bool) -> Self {
        self.truncate_long_samples = truncate;
        self
    }

    /// Sets the batch-id prefix.
    #[must_use]
    pub fn with_batch_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.batch_prefix = prefix.into();
        self
    }

    /// Packs `samples` into batches of at most `max_samples` samples.
    ///
    /// Ordering is the input order and the result never contains an empty
    /// batch. With a token budget set, an oversized sample is either truncated
    /// (response first, then prompt, both UTF-8 safe) or, when
    /// [`Self::truncate_long_samples`] is `false`, emitted alone in its own
    /// batch. Batch ids are `"{prefix}-{index}"` in emission order.
    #[must_use]
    pub fn pack(&self, samples: &[RlSample]) -> Vec<RlBatch> {
        let max_samples = self.max_samples.max(1);
        let mut batches: Vec<RlBatch> = Vec::new();
        let mut current: Vec<RlSample> = Vec::new();
        let mut current_truncated = 0usize;

        for sample in samples {
            let (fitted, truncated) = match self.max_tokens_per_sample {
                Some(budget) => fit_sample(sample, budget),
                None => (sample.clone(), false),
            };
            if truncated && !self.truncate_long_samples {
                self.flush(&mut batches, &mut current, &mut current_truncated);
                let batch_id = self.batch_id(batches.len());
                batches.push(RlBatch::from_samples(batch_id, vec![sample.clone()]));
                continue;
            }
            if current.len() >= max_samples {
                self.flush(&mut batches, &mut current, &mut current_truncated);
            }
            current.push(fitted);
            current_truncated += usize::from(truncated);
        }
        self.flush(&mut batches, &mut current, &mut current_truncated);
        batches
    }

    fn batch_id(&self, index: usize) -> String {
        format!("{}-{index}", self.batch_prefix)
    }

    fn flush(
        &self,
        batches: &mut Vec<RlBatch>,
        current: &mut Vec<RlSample>,
        truncated: &mut usize,
    ) {
        if current.is_empty() {
            return;
        }
        let batch_id = self.batch_id(batches.len());
        let mut batch = RlBatch::from_samples(batch_id, std::mem::take(current));
        batch.truncated = *truncated;
        *truncated = 0;
        batches.push(batch);
    }
}
