//! Token accounting shared by the model clients.
//!
//! Ports the `Usage` shape of `openjiuwen/core/foundation/llm/schema/`
//! (`usage.py`): the token counters every provider reports, plus a flattened
//! `extra` map for vendor-specific fields so unknown numbers survive a round
//! trip instead of being dropped.
//!
//! ```
//! use openjiuwen_foundation::prompt::usage::Usage;
//!
//! let mut total = Usage::new(10, 5);
//! total.add(&Usage::new(2, 3));
//! assert_eq!(total.total_tokens, 20);
//! ```

use std::collections::BTreeMap;
use std::ops::AddAssign;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Token counts reported for one model invocation.
///
/// Every field defaults to zero, so any subset of counters deserializes; keys
/// the struct does not model are captured in [`Usage::extra`] and re-serialized
/// at the same level.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct Usage {
    /// Tokens consumed by the prompt sent to the model.
    pub input_tokens: u64,
    /// Tokens produced by the model.
    pub output_tokens: u64,
    /// Total tokens as reported by the provider, or the input/output sum.
    pub total_tokens: u64,
    /// Prompt tokens served from the provider's prompt cache.
    pub cache_tokens: u64,
    /// Output tokens spent on reasoning/thinking.
    pub reasoning_tokens: u64,
    /// Provider-specific counters that this struct does not model.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

impl Usage {
    /// Creates a usage from the two mandatory counters; `total_tokens` is their
    /// sum and every other counter starts at zero.
    #[must_use]
    pub fn new(input_tokens: u64, output_tokens: u64) -> Self {
        Self {
            input_tokens,
            output_tokens,
            total_tokens: input_tokens.saturating_add(output_tokens),
            ..Self::default()
        }
    }

    /// Adds another usage field-wise; `extra` keys from `other` win collisions.
    pub fn add(&mut self, other: &Usage) {
        self.input_tokens = self.input_tokens.saturating_add(other.input_tokens);
        self.output_tokens = self.output_tokens.saturating_add(other.output_tokens);
        self.total_tokens = self.total_tokens.saturating_add(other.total_tokens);
        self.cache_tokens = self.cache_tokens.saturating_add(other.cache_tokens);
        self.reasoning_tokens = self.reasoning_tokens.saturating_add(other.reasoning_tokens);
        for (key, value) in &other.extra {
            self.extra.insert(key.clone(), value.clone());
        }
    }

    /// Returns `true` when no counter and no extra field carries a value.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.input_tokens == 0
            && self.output_tokens == 0
            && self.total_tokens == 0
            && self.cache_tokens == 0
            && self.reasoning_tokens == 0
            && self.extra.is_empty()
    }

    /// Fills in `total_tokens` when a provider reported the parts but not the
    /// total; a reported total is never recomputed.
    #[must_use]
    pub fn normalized(&self) -> Usage {
        let mut usage = self.clone();
        if usage.total_tokens == 0 && (usage.input_tokens != 0 || usage.output_tokens != 0) {
            usage.total_tokens = usage.input_tokens.saturating_add(usage.output_tokens);
        }
        usage
    }
}

impl AddAssign<&Usage> for Usage {
    fn add_assign(&mut self, other: &Usage) {
        self.add(other);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn new_sums_the_mandatory_counters() {
        let usage = Usage::new(12, 7);
        assert_eq!(usage.input_tokens, 12);
        assert_eq!(usage.output_tokens, 7);
        assert_eq!(usage.total_tokens, 19);
        assert_eq!(usage.cache_tokens, 0);
        assert_eq!(usage.reasoning_tokens, 0);
        assert!(usage.extra.is_empty());
    }

    #[test]
    fn add_accumulates_counters_and_merges_extra() {
        let mut left = Usage::new(1, 2);
        left.cache_tokens = 3;
        left.extra.insert("requests".to_owned(), json!(1));
        left.extra.insert("shared".to_owned(), json!("left"));

        let mut right = Usage::new(4, 5);
        right.reasoning_tokens = 6;
        right.extra.insert("shared".to_owned(), json!("right"));
        right.extra.insert("audio".to_owned(), json!(9));

        left.add(&right);
        assert_eq!(left.input_tokens, 5);
        assert_eq!(left.output_tokens, 7);
        assert_eq!(left.total_tokens, 12);
        assert_eq!(left.cache_tokens, 3);
        assert_eq!(left.reasoning_tokens, 6);
        assert_eq!(left.extra.get("requests"), Some(&json!(1)));
        assert_eq!(left.extra.get("shared"), Some(&json!("right")));
        assert_eq!(left.extra.get("audio"), Some(&json!(9)));

        left += &Usage::new(1, 1);
        assert_eq!(left.input_tokens, 6);
        assert_eq!(left.total_tokens, 14);
    }

    #[test]
    fn is_empty_covers_default_new_and_accumulated_usages() {
        assert!(Usage::default().is_empty());
        assert!(!Usage::new(0, 1).is_empty());

        let mut extra_only = Usage::default();
        extra_only.extra.insert("cached".to_owned(), json!(0));
        assert!(!extra_only.is_empty());

        let mut accumulated = Usage::default();
        accumulated.add(&Usage::new(0, 0));
        assert!(accumulated.is_empty());
    }

    #[test]
    fn normalized_only_recomputes_a_missing_total() {
        let mut usage = Usage::new(10, 5);
        usage.total_tokens = 0;
        let normalized = usage.normalized();
        assert_eq!(normalized.total_tokens, 15);
        assert_eq!(normalized.input_tokens, 10);
        assert_eq!(normalized.output_tokens, 5);

        usage.total_tokens = 99;
        assert_eq!(usage.normalized().total_tokens, 99);

        let one_sided = Usage {
            output_tokens: 4,
            ..Usage::default()
        };
        assert_eq!(one_sided.normalized().total_tokens, 4);

        assert_eq!(Usage::default().normalized(), Usage::default());
    }

    #[test]
    fn serde_reads_a_full_payload() {
        let usage: Usage = serde_json::from_value(json!({
            "input_tokens": 11,
            "output_tokens": 5,
            "total_tokens": 16,
            "cache_tokens": 3,
            "reasoning_tokens": 2
        }))
        .unwrap();
        assert_eq!(
            usage,
            Usage {
                input_tokens: 11,
                output_tokens: 5,
                total_tokens: 16,
                cache_tokens: 3,
                reasoning_tokens: 2,
                extra: BTreeMap::new(),
            }
        );
    }

    #[test]
    fn serde_defaults_every_missing_field() {
        let usage: Usage = serde_json::from_value(json!({})).unwrap();
        assert_eq!(usage, Usage::default());
        assert!(usage.is_empty());
    }

    #[test]
    fn serde_keeps_unknown_keys_flat_under_extra() {
        let usage: Usage = serde_json::from_value(json!({
            "input_tokens": 1,
            "output_tokens": 2,
            "prompt_tokens_details": {"cached_tokens": 3},
            "vendor_score": 0.5
        }))
        .unwrap();
        assert_eq!(usage.input_tokens, 1);
        assert_eq!(usage.output_tokens, 2);
        assert_eq!(
            usage.extra.get("prompt_tokens_details"),
            Some(&json!({"cached_tokens": 3}))
        );
        assert_eq!(usage.extra.get("vendor_score"), Some(&json!(0.5)));

        let value = serde_json::to_value(&usage).unwrap();
        assert_eq!(
            value,
            json!({
                "input_tokens": 1,
                "output_tokens": 2,
                "total_tokens": 0,
                "cache_tokens": 0,
                "reasoning_tokens": 0,
                "prompt_tokens_details": {"cached_tokens": 3},
                "vendor_score": 0.5
            })
        );
        assert!(value.get("extra").is_none());
    }

    #[test]
    fn serde_round_trips_a_usage_with_extras() {
        let mut usage = Usage::new(3, 4);
        usage.extra.insert("vendor_score".to_owned(), json!(0.25));
        let value = serde_json::to_value(&usage).unwrap();
        let back: Usage = serde_json::from_value(value).unwrap();
        assert_eq!(back, usage);
    }
}
