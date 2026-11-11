//! [`DedupeConfig`]: deterministic duplicate detection and merge settings.
//!
//! [`DedupePolicy`] is defined in `crate::record::options` because per-call
//! `AddOptions` needs it; this module re-exports it so configuration code can name it
//! as `config::DedupePolicy` without a second enum existing.
use super::{ensure_positive, ensure_unit_interval};
use openjiuwen_common::error::Result;
use serde::{Deserialize, Serialize};

pub use crate::record::DedupePolicy;

/// Deterministic dedupe/merge settings for the engine's write path.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct DedupeConfig {
    /// Whether duplicate detection runs at all (`true`).
    pub enabled: bool,
    /// What to do when a duplicate is found (`DedupePolicy::Merge`).
    pub policy: DedupePolicy,
    /// Normalized cosine similarity at or above which two records are duplicates
    /// (`0.92`).
    pub similarity_threshold: f32,
    /// Token Jaccard similarity at or above which two records are duplicates (`0.75`).
    pub jaccard_threshold: f32,
    /// How many recent candidates to compare against (`5`).
    pub candidate_top_k: usize,
    /// Minimum content length for fuzzy comparison; shorter text is only exact-deduped
    /// (`4`).
    pub min_content_len: usize,
}

impl Default for DedupeConfig {
    /// Returns the documented defaults (enabled, merge, `0.92`/`0.75`, `5`, `4`).
    fn default() -> Self {
        Self {
            enabled: true,
            policy: DedupePolicy::Merge,
            similarity_threshold: 0.92,
            jaccard_threshold: 0.75,
            candidate_top_k: 5,
            min_content_len: 4,
        }
    }
}

impl DedupeConfig {
    /// Validates the thresholds and candidate budget.
    ///
    /// Called by `MemoryEngineConfig::validate`; each failure is `Error::Config`
    /// naming the field and the offending value.
    pub(crate) fn validate(&self) -> Result<()> {
        ensure_unit_interval("similarity_threshold", self.similarity_threshold)?;
        ensure_unit_interval("jaccard_threshold", self.jaccard_threshold)?;
        ensure_positive("candidate_top_k", self.candidate_top_k)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use openjiuwen_common::error::Error;

    fn error_message(config: &DedupeConfig) -> String {
        match config.validate() {
            Err(Error::Config(message)) => message,
            other => panic!("expected a config error, got {other:?}"),
        }
    }

    #[test]
    fn defaults_match_the_specification() {
        let config = DedupeConfig::default();
        assert!(config.enabled);
        assert_eq!(config.policy, DedupePolicy::Merge);
        assert!((config.similarity_threshold - 0.92).abs() < f32::EPSILON);
        assert!((config.jaccard_threshold - 0.75).abs() < f32::EPSILON);
        assert_eq!(config.candidate_top_k, 5);
        assert_eq!(config.min_content_len, 4);
    }

    #[test]
    fn defaults_validate() {
        assert!(DedupeConfig::default().validate().is_ok());
    }

    #[test]
    fn policy_is_the_record_type() {
        // Compiles only when `config::DedupePolicy` is the canonical `record` type.
        let policy: crate::record::DedupePolicy = DedupeConfig::default().policy;
        assert_eq!(policy, crate::record::DedupePolicy::Merge);
        assert_eq!(
            serde_json::to_value(policy).unwrap(),
            serde_json::json!("merge")
        );
    }

    #[test]
    fn json_round_trip_preserves_every_field() {
        let config = DedupeConfig {
            enabled: false,
            policy: DedupePolicy::Replace,
            similarity_threshold: 0.5,
            jaccard_threshold: 0.25,
            candidate_top_k: 1,
            min_content_len: 0,
        };
        let json = serde_json::to_value(&config).unwrap();
        for key in [
            "enabled",
            "policy",
            "similarity_threshold",
            "jaccard_threshold",
            "candidate_top_k",
            "min_content_len",
        ] {
            assert!(json.get(key).is_some(), "missing {key}");
        }
        assert_eq!(json["policy"], serde_json::json!("replace"));
        let restored: DedupeConfig = serde_json::from_value(json).unwrap();
        assert_eq!(restored, config);
    }

    #[test]
    fn unknown_fields_are_ignored() {
        let config: DedupeConfig = serde_json::from_value(serde_json::json!({
            "candidate_top_k": 3,
            "not_a_knob": "x",
        }))
        .unwrap();
        assert_eq!(config.candidate_top_k, 3);
        assert!(config.enabled);
        assert_eq!(config.policy, DedupePolicy::Merge);
    }

    #[test]
    fn rejects_out_of_range_similarity_threshold() {
        for value in [-0.01, 1.01, f32::NAN] {
            let config = DedupeConfig {
                similarity_threshold: value,
                ..Default::default()
            };
            assert!(
                error_message(&config).contains("similarity_threshold"),
                "value {value} must be rejected"
            );
        }
    }

    #[test]
    fn rejects_out_of_range_jaccard_threshold() {
        for value in [-0.01, 1.01, f32::NAN] {
            let config = DedupeConfig {
                jaccard_threshold: value,
                ..Default::default()
            };
            assert!(
                error_message(&config).contains("jaccard_threshold"),
                "value {value} must be rejected"
            );
        }
    }

    #[test]
    fn rejects_zero_candidate_top_k() {
        let config = DedupeConfig {
            candidate_top_k: 0,
            ..Default::default()
        };
        assert!(error_message(&config).contains("candidate_top_k"));
    }
}
