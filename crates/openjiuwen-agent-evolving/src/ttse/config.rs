//! Tunables for the TTSE pipeline.
//!
//! Ports `ttse/config.py` to a serde record with `snake_case` field names, so a
//! Python-produced config file stays readable. Every field has a default, and
//! [`TtseConfig::validate`] is the explicit entry point for the bounds the
//! constructors of the rest of the module assume.
//!
//! The config is pure data: no clocks, no environment reads, no model calls.

use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// The default store path, relative to the working directory.
const DEFAULT_STORE_PATH: &str = ".ttse/bank.json";

/// Tunables for tip induction, ranking, consultation and dreaming.
///
/// Unknown JSON fields are ignored and missing fields fall back to the
/// [`Default`] implementation, so newer or partial config files load cleanly.
// The flag-shaped tunables mirror the Python config one-to-one; folding them
// into enums would break the field names the Python config files use.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TtseConfig {
    /// Where the tip bank is persisted, for example `.ttse/bank.json`.
    pub store_path: String,
    /// Maximum number of tips kept in the bank; lowest-scoring tips are evicted.
    pub capacity: usize,
    /// Jaccard similarity at or above which two tips are considered duplicates.
    pub dedup_threshold: f64,
    /// Whether induced tips are injected into the agent's prompt.
    pub inject_enabled: bool,
    /// Whether the bank keeps evolving (induction and dreaming) after bootstrapping.
    pub evolve_enabled: bool,
    /// Minimum trajectory score for a run to count as a success signal.
    pub success_threshold: f64,
    /// Minimum number of tool calls before a run is considered for tip induction.
    pub detect_min_tool_calls: usize,
    /// Maximum characters of rendered tips injected in one consultation.
    pub consult_max_chars: usize,
    /// Maximum number of tips considered for one consultation.
    pub consult_max_rules: usize,
    /// Top-`k` cut applied when ranking tips for consultation.
    pub consult_top_k: usize,
    /// Whether the periodic dream pass (merge/prune) runs.
    pub dream_enabled: bool,
    /// How many bank ticks pass between dream passes.
    pub dream_interval: u64,
    /// Minimum number of tips before a dream pass does any work.
    pub dream_min_rules: usize,
    /// Ticks without an update after which a tip is prunable; `0` disables TTL pruning.
    pub dream_ttl_ticks: u64,
    /// Minimum cluster size the dreamer merges.
    pub dream_cluster_min_size: usize,
    /// Similarity threshold used when the dreamer merges a cluster of tips.
    pub dream_merge_threshold: f64,
    /// Whether the dreamer prunes stale or low-weight tips.
    pub dream_prune_enabled: bool,
    /// Maximum number of tips attached to a single observation.
    pub max_tips_per_observation: usize,
}

impl Default for TtseConfig {
    fn default() -> Self {
        Self {
            store_path: DEFAULT_STORE_PATH.to_owned(),
            capacity: 512,
            dedup_threshold: 0.88,
            inject_enabled: true,
            evolve_enabled: true,
            success_threshold: 0.999,
            detect_min_tool_calls: 5,
            consult_max_chars: 8_000,
            consult_max_rules: 40,
            consult_top_k: 8,
            dream_enabled: true,
            dream_interval: 20,
            dream_min_rules: 8,
            dream_ttl_ticks: 90,
            dream_cluster_min_size: 2,
            dream_merge_threshold: 0.72,
            dream_prune_enabled: true,
            max_tips_per_observation: 8,
        }
    }
}

impl TtseConfig {
    /// Checks the bounds the rest of the module assumes.
    ///
    /// Error messages name the offending field so a bad config file is
    /// actionable. `dream_ttl_ticks` may be `0`, which disables TTL pruning.
    pub fn validate(&self) -> Result<()> {
        validate_unit_interval(self.dedup_threshold, "dedup_threshold")?;
        validate_unit_interval(self.success_threshold, "success_threshold")?;
        validate_unit_interval(self.dream_merge_threshold, "dream_merge_threshold")?;
        validate_positive(self.capacity, "capacity")?;
        validate_positive(self.consult_max_chars, "consult_max_chars")?;
        validate_positive(self.consult_max_rules, "consult_max_rules")?;
        validate_positive(self.consult_top_k, "consult_top_k")?;
        validate_positive_u64(self.dream_interval, "dream_interval")?;
        validate_positive(self.dream_min_rules, "dream_min_rules")?;
        validate_positive(self.dream_cluster_min_size, "dream_cluster_min_size")?;
        validate_positive(self.max_tips_per_observation, "max_tips_per_observation")?;
        if self.store_path.trim().is_empty() {
            return Err(Error::validation(
                "ttse config field `store_path` must not be empty",
            ));
        }
        Ok(())
    }

    /// The effective number of tips one consultation may consider.
    ///
    /// Python applies both limits; the tighter of [`Self::consult_top_k`] and
    /// [`Self::consult_max_rules`] wins.
    #[must_use]
    pub fn consult_limit(&self) -> usize {
        self.consult_top_k.min(self.consult_max_rules)
    }
}

/// Rejects values outside the closed unit interval, naming the field.
fn validate_unit_interval(value: f64, field: &str) -> Result<()> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(Error::validation(format!(
            "ttse config field `{field}` must be finite and within 0.0..=1.0, got {value}"
        )));
    }
    Ok(())
}

/// Rejects non-positive counts, naming the field.
fn validate_positive(value: usize, field: &str) -> Result<()> {
    if value == 0 {
        return Err(Error::validation(format!(
            "ttse config field `{field}` must be greater than 0"
        )));
    }
    Ok(())
}

/// Rejects non-positive tick counts, naming the field.
fn validate_positive_u64(value: u64, field: &str) -> Result<()> {
    if value == 0 {
        return Err(Error::validation(format!(
            "ttse config field `{field}` must be greater than 0"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn defaults_match_the_python_config() {
        let config = TtseConfig::default();
        assert_eq!(config.store_path, ".ttse/bank.json");
        assert_eq!(config.capacity, 512);
        assert!((config.dedup_threshold - 0.88).abs() < f64::EPSILON);
        assert!(config.inject_enabled && config.evolve_enabled);
        assert!((config.success_threshold - 0.999).abs() < f64::EPSILON);
        assert_eq!(config.detect_min_tool_calls, 5);
        assert_eq!(
            (
                config.consult_max_chars,
                config.consult_max_rules,
                config.consult_top_k
            ),
            (8_000, 40, 8)
        );
        assert!(config.dream_enabled && config.dream_prune_enabled);
        assert_eq!(
            (
                config.dream_interval,
                config.dream_min_rules,
                config.dream_ttl_ticks,
                config.dream_cluster_min_size,
                config.max_tips_per_observation,
            ),
            (20, 8, 90, 2, 8)
        );
        assert!((config.dream_merge_threshold - 0.72).abs() < f64::EPSILON);
        assert!(config.validate().is_ok());
        assert_eq!(config.consult_limit(), 8);
    }

    #[test]
    fn consult_limit_is_the_tighter_bound() {
        let config = TtseConfig {
            consult_max_rules: 3,
            ..TtseConfig::default()
        };
        assert_eq!(config.consult_limit(), 3);
        let config = TtseConfig {
            consult_max_rules: 3,
            consult_top_k: 2,
            ..TtseConfig::default()
        };
        assert_eq!(config.consult_limit(), 2);
    }

    #[test]
    fn validate_rejects_out_of_range_thresholds() {
        for field in [
            "dedup_threshold",
            "success_threshold",
            "dream_merge_threshold",
        ] {
            let config = match field {
                "dedup_threshold" => TtseConfig {
                    dedup_threshold: 1.5,
                    ..TtseConfig::default()
                },
                "success_threshold" => TtseConfig {
                    success_threshold: f64::NAN,
                    ..TtseConfig::default()
                },
                _ => TtseConfig {
                    dream_merge_threshold: -0.1,
                    ..TtseConfig::default()
                },
            };
            let error = config.validate().expect_err("out-of-range threshold");
            match error {
                Error::Validation(message) => assert!(
                    message.contains(field),
                    "message `{message}` must name `{field}`"
                ),
                other => panic!("unexpected error: {other:?}"),
            }
        }
    }

    #[test]
    fn validate_rejects_zero_counts_and_blank_store_path() {
        let config = TtseConfig {
            capacity: 0,
            ..TtseConfig::default()
        };
        assert!(config.validate().is_err());

        let config = TtseConfig {
            consult_top_k: 0,
            ..TtseConfig::default()
        };
        assert!(config.validate().is_err());

        let config = TtseConfig {
            dream_cluster_min_size: 0,
            ..TtseConfig::default()
        };
        assert!(config.validate().is_err());

        let config = TtseConfig {
            max_tips_per_observation: 0,
            ..TtseConfig::default()
        };
        assert!(config.validate().is_err());

        let config = TtseConfig {
            store_path: "   ".to_owned(),
            ..TtseConfig::default()
        };
        assert!(config.validate().is_err());

        let config = TtseConfig {
            dream_ttl_ticks: 0,
            ..TtseConfig::default()
        };
        assert!(config.validate().is_ok());
    }

    #[test]
    fn serde_fills_missing_fields_and_ignores_unknown_ones() {
        let value = json!({ "capacity": 16, "unknown_field": [1, 2, 3] });
        let config: TtseConfig = serde_json::from_value(value).expect("partial config loads");
        assert_eq!(config.capacity, 16);
        assert_eq!(config.store_path, TtseConfig::default().store_path);
        assert!((config.dedup_threshold - 0.88).abs() < f64::EPSILON);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn serde_round_trip() {
        let config = TtseConfig {
            capacity: 7,
            dream_ttl_ticks: 0,
            ..TtseConfig::default()
        };
        let json = serde_json::to_string(&config).expect("serializable");
        let restored: TtseConfig = serde_json::from_str(&json).expect("deserializable");
        assert_eq!(restored, config);
    }
}
