//! [`MemoryEngineConfig`]: the top-level configuration of a memory engine instance.
//!
//! The defaults mirror the Python `MemoryEngineConfig` fields where they exist
//! (`top_k = 5`, `threshold = 0.3`, `input_msg_max_len = 8192`,
//! `summary_max_token = 128`, `get_recent_messages(num = 10)`); the ranking, dedupe,
//! extraction, graph, and coding sections are Rust-only knobs documented in their own
//! submodules.
use super::{
    coding::CodingMemoryConfig, dedupe::DedupeConfig, ensure_positive, ensure_unit_interval,
    extraction::ExtractionConfig, graph::GraphMemoryConfig, ranking::RankingConfig,
};
use crate::record::MemoryType;
use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// The configuration of one memory engine instance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct MemoryEngineConfig {
    /// Default number of recall hits (`5`).
    pub default_top_k: usize,
    /// Default minimum blended score a hit must reach (`0.3`).
    pub default_threshold: f32,
    /// Optional default token budget for recalled snippets (`None`).
    pub token_budget: Option<usize>,
    /// Kinds recall searches unless a call overrides them (`MemoryType::SEARCHABLE`).
    pub searchable_kinds: Vec<MemoryType>,
    /// Ranking blend weights and knobs.
    pub ranking: RankingConfig,
    /// Duplicate detection and merge settings.
    pub dedupe: DedupeConfig,
    /// Message-history extraction settings.
    pub extraction: ExtractionConfig,
    /// Entity/graph memory settings.
    pub graph: GraphMemoryConfig,
    /// Lite coding memory settings.
    pub coding: CodingMemoryConfig,
    /// Variable names that must never be persisted (Python's comma-split string, as a
    /// list).
    pub forbidden_variables: Vec<String>,
    /// Maximum input message length in characters before truncation (`8192`).
    pub input_msg_max_len: usize,
    /// Token budget of a single-turn history summary (`128`).
    pub single_turn_history_summary_max_token: usize,
    /// Number of recent messages returned by default (`10`).
    pub message_history_limit: usize,
    /// Importance assigned when a record does not specify one (`0.5`).
    pub importance_default: f32,
}

impl Default for MemoryEngineConfig {
    /// Returns the documented defaults.
    fn default() -> Self {
        Self {
            default_top_k: 5,
            default_threshold: 0.3,
            token_budget: None,
            searchable_kinds: MemoryType::SEARCHABLE.to_vec(),
            ranking: RankingConfig::default(),
            dedupe: DedupeConfig::default(),
            extraction: ExtractionConfig::default(),
            graph: GraphMemoryConfig::default(),
            coding: CodingMemoryConfig::default(),
            forbidden_variables: Vec::new(),
            input_msg_max_len: 8192,
            single_turn_history_summary_max_token: 128,
            message_history_limit: 10,
            importance_default: 0.5,
        }
    }
}

impl MemoryEngineConfig {
    /// Validates every invariant of the config and its nested sections.
    ///
    /// Each failure is `Error::Config` naming the field and the offending value.
    pub fn validate(&self) -> Result<()> {
        ensure_positive("default_top_k", self.default_top_k)?;
        ensure_unit_interval("default_threshold", self.default_threshold)?;
        ensure_unit_interval("importance_default", self.importance_default)?;
        self.validate_searchable_kinds()?;
        self.ranking.validate()?;
        self.dedupe.validate()?;
        self.extraction.validate()?;
        self.graph.validate()?;
        self.coding.validate()?;
        ensure_positive("input_msg_max_len", self.input_msg_max_len)?;
        ensure_positive(
            "single_turn_history_summary_max_token",
            self.single_turn_history_summary_max_token,
        )?;
        ensure_positive("message_history_limit", self.message_history_limit)?;
        Ok(())
    }

    /// Validates that the searchable kinds are non-empty, exclude `Variable`, and are
    /// unique.
    fn validate_searchable_kinds(&self) -> Result<()> {
        if self.searchable_kinds.is_empty() {
            return Err(Error::config("searchable_kinds must not be empty"));
        }
        for (index, kind) in self.searchable_kinds.iter().enumerate() {
            if *kind == MemoryType::Variable {
                return Err(Error::config(
                    "searchable_kinds must not contain variable (variables are not searchable \
                     records)",
                ));
            }
            if self.searchable_kinds[..index].contains(kind) {
                return Err(Error::config(format!(
                    "searchable_kinds must not contain duplicate {}",
                    kind.as_str()
                )));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn error_message(config: &MemoryEngineConfig) -> String {
        match config.validate() {
            Err(Error::Config(message)) => message,
            other => panic!("expected a config error, got {other:?}"),
        }
    }

    #[test]
    fn defaults_match_the_specification() {
        let config = MemoryEngineConfig::default();
        assert_eq!(config.default_top_k, 5);
        assert!((config.default_threshold - 0.3).abs() < f32::EPSILON);
        assert!(config.token_budget.is_none());
        assert_eq!(config.searchable_kinds, MemoryType::SEARCHABLE);
        assert_eq!(config.ranking, RankingConfig::default());
        assert_eq!(config.dedupe, DedupeConfig::default());
        assert_eq!(config.extraction, ExtractionConfig::default());
        assert_eq!(config.graph, GraphMemoryConfig::default());
        assert_eq!(config.coding, CodingMemoryConfig::default());
        assert_eq!(config.forbidden_variables, Vec::<String>::new());
        assert_eq!(config.input_msg_max_len, 8192);
        assert_eq!(config.single_turn_history_summary_max_token, 128);
        assert_eq!(config.message_history_limit, 10);
        assert!((config.importance_default - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn defaults_validate() {
        assert!(MemoryEngineConfig::default().validate().is_ok());
    }

    #[test]
    fn json_round_trip_preserves_every_field() {
        let config = MemoryEngineConfig {
            token_budget: Some(512),
            forbidden_variables: vec!["api_key".to_owned()],
            ..Default::default()
        };
        let json = serde_json::to_value(&config).unwrap();
        for key in [
            "default_top_k",
            "default_threshold",
            "token_budget",
            "searchable_kinds",
            "ranking",
            "dedupe",
            "extraction",
            "graph",
            "coding",
            "forbidden_variables",
            "input_msg_max_len",
            "single_turn_history_summary_max_token",
            "message_history_limit",
            "importance_default",
        ] {
            assert!(json.get(key).is_some(), "missing {key}");
        }
        assert_eq!(
            json["searchable_kinds"],
            serde_json::json!([
                "user_profile",
                "semantic_memory",
                "episodic_memory",
                "summary",
                "coding"
            ])
        );
        assert_eq!(json["token_budget"], serde_json::json!(512));
        let restored: MemoryEngineConfig = serde_json::from_value(json).unwrap();
        assert_eq!(restored, config);
    }

    #[test]
    fn unknown_fields_are_ignored() {
        let config: MemoryEngineConfig = serde_json::from_value(serde_json::json!({
            "default_top_k": 9,
            "not_a_section": {"a": 1},
        }))
        .unwrap();
        assert_eq!(config.default_top_k, 9);
        assert!((config.default_threshold - 0.3).abs() < f32::EPSILON);
        assert_eq!(config.ranking, RankingConfig::default());
    }

    #[test]
    fn rejects_zero_default_top_k() {
        let config = MemoryEngineConfig {
            default_top_k: 0,
            ..Default::default()
        };
        let message = error_message(&config);
        assert!(message.contains("default_top_k"));
        assert!(message.contains('0'));
    }

    #[test]
    fn rejects_out_of_range_thresholds() {
        for value in [-0.1, 1.1, f32::NAN, f32::INFINITY] {
            let config = MemoryEngineConfig {
                default_threshold: value,
                ..Default::default()
            };
            assert!(
                error_message(&config).contains("default_threshold"),
                "value {value} must be rejected"
            );
            let config = MemoryEngineConfig {
                importance_default: value,
                ..Default::default()
            };
            assert!(
                error_message(&config).contains("importance_default"),
                "value {value} must be rejected"
            );
        }
    }

    #[test]
    fn rejects_empty_searchable_kinds() {
        let config = MemoryEngineConfig {
            searchable_kinds: Vec::new(),
            ..Default::default()
        };
        assert!(error_message(&config).contains("searchable_kinds"));
    }

    #[test]
    fn rejects_variable_in_searchable_kinds() {
        let config = MemoryEngineConfig {
            searchable_kinds: vec![MemoryType::Summary, MemoryType::Variable],
            ..Default::default()
        };
        let message = error_message(&config);
        assert!(message.contains("searchable_kinds"));
        assert!(message.contains("variable"));
    }

    #[test]
    fn rejects_duplicate_searchable_kinds() {
        let config = MemoryEngineConfig {
            searchable_kinds: vec![MemoryType::Summary, MemoryType::Summary],
            ..Default::default()
        };
        let message = error_message(&config);
        assert!(message.contains("searchable_kinds"));
        assert!(message.contains("duplicate"));
        assert!(message.contains("summary"));
    }

    #[test]
    fn rejects_zero_message_limits() {
        let config = MemoryEngineConfig {
            input_msg_max_len: 0,
            ..Default::default()
        };
        assert!(error_message(&config).contains("input_msg_max_len"));

        let config = MemoryEngineConfig {
            single_turn_history_summary_max_token: 0,
            ..Default::default()
        };
        assert!(error_message(&config).contains("single_turn_history_summary_max_token"));

        let config = MemoryEngineConfig {
            message_history_limit: 0,
            ..Default::default()
        };
        assert!(error_message(&config).contains("message_history_limit"));
    }

    #[test]
    fn surfaces_nested_section_errors() {
        let config = MemoryEngineConfig {
            dedupe: DedupeConfig {
                candidate_top_k: 0,
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(error_message(&config).contains("candidate_top_k"));

        let config = MemoryEngineConfig {
            graph: GraphMemoryConfig {
                max_depth: 5,
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(error_message(&config).contains("max_depth"));

        let config = MemoryEngineConfig {
            extraction: ExtractionConfig {
                max_facts: 0,
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(error_message(&config).contains("max_facts"));

        let config = MemoryEngineConfig {
            coding: CodingMemoryConfig {
                max_index_lines: 0,
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(error_message(&config).contains("max_index_lines"));

        let config = MemoryEngineConfig {
            ranking: RankingConfig {
                max_candidates: 0,
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(error_message(&config).contains("max_candidates"));
    }
}
