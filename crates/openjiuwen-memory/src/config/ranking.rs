//! [`RankingConfig`]: the weights and knobs of the documented recall scoring blend.
//!
//! The blend is fixed by the crate documentation:
//! `semantic = (w_sim * similarity + w_kw * keyword) / (w_sim + w_kw)` and
//! `final = (w_sem * semantic + w_imp * importance + w_rec * recency)
//! / (w_sem + w_imp + w_rec)`.
//! Weights are renormalized per group, so setting a weight to zero disables that term
//! as long as the group's sum stays positive.
use super::{ensure_non_negative, ensure_positive, ensure_unit_interval};
use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// Weights and knobs of the recall ranking blend.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct RankingConfig {
    /// Weight of the vector-similarity term inside the semantic score (`0.70`).
    pub similarity_weight: f32,
    /// Weight of the keyword (`BM25`) term inside the semantic score (`0.30`).
    pub keyword_weight: f32,
    /// Weight of the blended semantic score in the final score (`0.85`).
    pub semantic_weight: f32,
    /// Weight of record importance in the final score (`0.10`).
    pub importance_weight: f32,
    /// Weight of recency in the final score (`0.05`).
    pub recency_weight: f32,
    /// Half-life of the recency decay in seconds (`2_592_000`, i.e. 30 days).
    pub recency_half_life_secs: u64,
    /// Minimum normalized keyword score for a keyword-only candidate (`0.10`).
    pub keyword_min_score: f32,
    /// Candidate pool size as a multiple of `top_k` (`2.0`).
    pub candidate_multiplier: f32,
    /// Upper bound on the candidate pool regardless of `top_k` (`200`).
    pub max_candidates: usize,
}

impl Default for RankingConfig {
    /// Returns the documented defaults (`0.70`, `0.30`, `0.85`, `0.10`, `0.05`, ...).
    fn default() -> Self {
        Self {
            similarity_weight: 0.70,
            keyword_weight: 0.30,
            semantic_weight: 0.85,
            importance_weight: 0.10,
            recency_weight: 0.05,
            recency_half_life_secs: 2_592_000,
            keyword_min_score: 0.10,
            candidate_multiplier: 2.0,
            max_candidates: 200,
        }
    }
}

impl RankingConfig {
    /// Validates the weights and knobs.
    ///
    /// Called by `MemoryEngineConfig::validate`; each failure is `Error::Config`
    /// naming the field and the offending value.
    pub(crate) fn validate(&self) -> Result<()> {
        for (field, value) in [
            ("similarity_weight", self.similarity_weight),
            ("keyword_weight", self.keyword_weight),
            ("semantic_weight", self.semantic_weight),
            ("importance_weight", self.importance_weight),
            ("recency_weight", self.recency_weight),
        ] {
            ensure_non_negative(field, value)?;
        }
        let similarity_group = self.similarity_weight + self.keyword_weight;
        if similarity_group <= 0.0 {
            return Err(Error::config(format!(
                "similarity_weight + keyword_weight must be greater than zero (got \
                 {similarity_group})"
            )));
        }
        let final_group = self.semantic_weight + self.importance_weight + self.recency_weight;
        if final_group <= 0.0 {
            return Err(Error::config(format!(
                "semantic_weight + importance_weight + recency_weight must be greater than \
                 zero (got {final_group})"
            )));
        }
        if self.recency_half_life_secs == 0 {
            return Err(Error::config(
                "recency_half_life_secs must be greater than zero (got 0)",
            ));
        }
        ensure_unit_interval("keyword_min_score", self.keyword_min_score)?;
        ensure_positive("max_candidates", self.max_candidates)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn error_message(config: &RankingConfig) -> String {
        match config.validate() {
            Err(Error::Config(message)) => message,
            other => panic!("expected a config error, got {other:?}"),
        }
    }

    #[test]
    fn defaults_match_the_specification() {
        let config = RankingConfig::default();
        assert!((config.similarity_weight - 0.70).abs() < f32::EPSILON);
        assert!((config.keyword_weight - 0.30).abs() < f32::EPSILON);
        assert!((config.semantic_weight - 0.85).abs() < f32::EPSILON);
        assert!((config.importance_weight - 0.10).abs() < f32::EPSILON);
        assert!((config.recency_weight - 0.05).abs() < f32::EPSILON);
        assert_eq!(config.recency_half_life_secs, 2_592_000);
        assert!((config.keyword_min_score - 0.10).abs() < f32::EPSILON);
        assert!((config.candidate_multiplier - 2.0).abs() < f32::EPSILON);
        assert_eq!(config.max_candidates, 200);
    }

    #[test]
    fn documented_weight_sums_are_pinned() {
        let config = RankingConfig::default();
        assert!((config.similarity_weight + config.keyword_weight - 1.0).abs() < 1e-6);
        assert!(
            (config.semantic_weight + config.importance_weight + config.recency_weight - 1.0).abs()
                < 1e-6
        );
    }

    #[test]
    fn defaults_validate() {
        assert!(RankingConfig::default().validate().is_ok());
    }

    #[test]
    fn json_round_trip_preserves_every_field() {
        let config = RankingConfig {
            similarity_weight: 0.5,
            keyword_weight: 0.5,
            semantic_weight: 1.0,
            importance_weight: 0.0,
            recency_weight: 0.0,
            recency_half_life_secs: 60,
            keyword_min_score: 0.2,
            candidate_multiplier: 3.0,
            max_candidates: 10,
        };
        let json = serde_json::to_value(&config).unwrap();
        for key in [
            "similarity_weight",
            "keyword_weight",
            "semantic_weight",
            "importance_weight",
            "recency_weight",
            "recency_half_life_secs",
            "keyword_min_score",
            "candidate_multiplier",
            "max_candidates",
        ] {
            assert!(json.get(key).is_some(), "missing {key}");
        }
        let restored: RankingConfig = serde_json::from_value(json).unwrap();
        assert_eq!(restored, config);
    }

    #[test]
    fn unknown_fields_are_ignored() {
        let config: RankingConfig = serde_json::from_value(serde_json::json!({
            "max_candidates": 7,
            "not_a_weight": 0.5,
        }))
        .unwrap();
        assert_eq!(config.max_candidates, 7);
        assert!((config.similarity_weight - 0.70).abs() < f32::EPSILON);
    }

    #[test]
    fn rejects_negative_and_non_finite_weights() {
        for value in [-0.1, f32::NAN, f32::NEG_INFINITY] {
            let config = RankingConfig {
                similarity_weight: value,
                ..Default::default()
            };
            assert!(
                error_message(&config).contains("similarity_weight"),
                "value {value} must be rejected"
            );
        }
        let config = RankingConfig {
            recency_weight: f32::INFINITY,
            ..Default::default()
        };
        assert!(error_message(&config).contains("recency_weight"));
    }

    #[test]
    fn rejects_empty_similarity_group() {
        let config = RankingConfig {
            similarity_weight: 0.0,
            keyword_weight: 0.0,
            ..Default::default()
        };
        let message = error_message(&config);
        assert!(message.contains("similarity_weight"));
        assert!(message.contains("keyword_weight"));
    }

    #[test]
    fn rejects_empty_final_group() {
        let config = RankingConfig {
            semantic_weight: 0.0,
            importance_weight: 0.0,
            recency_weight: 0.0,
            ..Default::default()
        };
        let message = error_message(&config);
        assert!(message.contains("semantic_weight"));
        assert!(message.contains("recency_weight"));
    }

    #[test]
    fn rejects_zero_recency_half_life() {
        let config = RankingConfig {
            recency_half_life_secs: 0,
            ..Default::default()
        };
        assert!(error_message(&config).contains("recency_half_life_secs"));
    }

    #[test]
    fn rejects_out_of_range_keyword_min_score() {
        for value in [-0.1, 1.1] {
            let config = RankingConfig {
                keyword_min_score: value,
                ..Default::default()
            };
            assert!(
                error_message(&config).contains("keyword_min_score"),
                "value {value} must be rejected"
            );
        }
    }

    #[test]
    fn rejects_zero_max_candidates() {
        let config = RankingConfig {
            max_candidates: 0,
            ..Default::default()
        };
        assert!(error_message(&config).contains("max_candidates"));
    }
}
