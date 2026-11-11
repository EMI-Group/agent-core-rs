//! [`ExtractionConfig`], [`AgentMemoryConfig`], and [`MemoryScopeConfig`].
//!
//! These mirror the Python `core/memory/config/config.py` triples: the extraction
//! knobs of the message path, the per-agent memory toggles, and the per-scope memory
//! definition text.
use super::ensure_positive;
use crate::record::MemoryType;
use openjiuwen_common::error::Result;
use serde::{Deserialize, Serialize};

/// Message-history extraction and summarization settings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ExtractionConfig {
    /// Whether `add_messages` extracts memory facts (`true`).
    pub enabled: bool,
    /// Maximum number of facts extracted from one message batch (`8`).
    pub max_facts: usize,
    /// How many times the extraction model call is retried (`3`).
    pub model_retries: u32,
    /// How many earlier messages are passed to extraction as history (`2`).
    pub use_history_messages: usize,
    /// Messages shorter than this many characters are skipped (`8`).
    pub min_message_len: usize,
}

impl Default for ExtractionConfig {
    /// Returns the documented defaults (enabled, `8`, `3`, `2`, `8`).
    fn default() -> Self {
        Self {
            enabled: true,
            max_facts: 8,
            model_retries: 3,
            use_history_messages: 2,
            min_message_len: 8,
        }
    }
}

impl ExtractionConfig {
    /// Validates the extraction limits.
    ///
    /// Called by `MemoryEngineConfig::validate`; each failure is `Error::Config`
    /// naming the field and the offending value.
    pub(crate) fn validate(&self) -> Result<()> {
        ensure_positive("max_facts", self.max_facts)?;
        ensure_positive("min_message_len", self.min_message_len)?;
        Ok(())
    }
}

/// Per-agent memory toggles, mirroring Python's `AgentMemoryConfig`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct AgentMemoryConfig {
    /// Master switch for long-term memory (`true`).
    pub enable_long_term_mem: bool,
    /// Whether user-profile memories are written and recalled (`true`).
    pub enable_user_profile: bool,
    /// Whether semantic memories are written and recalled (`true`).
    pub enable_semantic_memory: bool,
    /// Whether episodic memories are written and recalled (`true`).
    pub enable_episodic_memory: bool,
    /// Whether summary memories are written and recalled (`true`).
    pub enable_summary_memory: bool,
    /// Variable names the agent exposes to memory extraction (`[]`).
    pub mem_variables: Vec<String>,
}

impl Default for AgentMemoryConfig {
    /// Returns the documented defaults (every toggle enabled, no variables).
    fn default() -> Self {
        Self {
            enable_long_term_mem: true,
            enable_user_profile: true,
            enable_semantic_memory: true,
            enable_episodic_memory: true,
            enable_summary_memory: true,
            mem_variables: Vec::new(),
        }
    }
}

impl AgentMemoryConfig {
    /// Returns whether `kind` is enabled for this agent.
    ///
    /// [`MemoryType::Variable`] and [`MemoryType::Unknown`] are never enabled.
    /// [`MemoryType::Coding`] has no dedicated flag in this struct and follows
    /// `enable_long_term_mem`, matching Python's treatment of coding memory as part of
    /// long-term memory.
    #[must_use]
    pub const fn kind_enabled(&self, kind: MemoryType) -> bool {
        match kind {
            MemoryType::UserProfile => self.enable_user_profile,
            MemoryType::SemanticMemory => self.enable_semantic_memory,
            MemoryType::EpisodicMemory => self.enable_episodic_memory,
            MemoryType::Summary => self.enable_summary_memory,
            MemoryType::Coding => self.enable_long_term_mem,
            MemoryType::Variable | MemoryType::Unknown => false,
        }
    }
}

/// Per-scope memory definition text, mirroring Python's `MemoryScopeConfig`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct MemoryScopeConfig {
    /// Custom definition of what belongs in user-profile memory.
    pub user_profile_definition: Option<String>,
    /// Custom definition of what belongs in semantic memory.
    pub semantic_memory_definition: Option<String>,
    /// Custom definition of what belongs in episodic memory.
    pub episodic_memory_definition: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use openjiuwen_common::error::Error;

    fn error_message(config: &ExtractionConfig) -> String {
        match config.validate() {
            Err(Error::Config(message)) => message,
            other => panic!("expected a config error, got {other:?}"),
        }
    }

    #[test]
    fn extraction_defaults_match_the_specification() {
        let config = ExtractionConfig::default();
        assert!(config.enabled);
        assert_eq!(config.max_facts, 8);
        assert_eq!(config.model_retries, 3);
        assert_eq!(config.use_history_messages, 2);
        assert_eq!(config.min_message_len, 8);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn extraction_json_round_trip_preserves_every_field() {
        let config = ExtractionConfig {
            enabled: false,
            max_facts: 1,
            model_retries: 0,
            use_history_messages: 0,
            min_message_len: 1,
        };
        let json = serde_json::to_value(&config).unwrap();
        for key in [
            "enabled",
            "max_facts",
            "model_retries",
            "use_history_messages",
            "min_message_len",
        ] {
            assert!(json.get(key).is_some(), "missing {key}");
        }
        let restored: ExtractionConfig = serde_json::from_value(json).unwrap();
        assert_eq!(restored, config);
    }

    #[test]
    fn extraction_unknown_fields_are_ignored() {
        let config: ExtractionConfig = serde_json::from_value(serde_json::json!({
            "max_facts": 2,
            "not_a_knob": true,
        }))
        .unwrap();
        assert_eq!(config.max_facts, 2);
        assert!(config.enabled);
    }

    #[test]
    fn extraction_rejects_zero_limits() {
        let config = ExtractionConfig {
            max_facts: 0,
            ..Default::default()
        };
        assert!(error_message(&config).contains("max_facts"));

        let config = ExtractionConfig {
            min_message_len: 0,
            ..Default::default()
        };
        assert!(error_message(&config).contains("min_message_len"));
    }

    #[test]
    fn agent_defaults_and_kind_enabled() {
        let config = AgentMemoryConfig::default();
        assert!(config.enable_long_term_mem);
        assert!(config.enable_user_profile);
        assert!(config.enable_semantic_memory);
        assert!(config.enable_episodic_memory);
        assert!(config.enable_summary_memory);
        assert_eq!(config.mem_variables, Vec::<String>::new());
        for kind in MemoryType::SEARCHABLE {
            assert!(config.kind_enabled(kind), "{kind} must be enabled");
        }
        assert!(!config.kind_enabled(MemoryType::Variable));
        assert!(!config.kind_enabled(MemoryType::Unknown));
    }

    #[test]
    fn agent_kind_enabled_maps_each_flag() {
        for (kind, config) in [
            (
                MemoryType::UserProfile,
                AgentMemoryConfig {
                    enable_user_profile: false,
                    ..Default::default()
                },
            ),
            (
                MemoryType::SemanticMemory,
                AgentMemoryConfig {
                    enable_semantic_memory: false,
                    ..Default::default()
                },
            ),
            (
                MemoryType::EpisodicMemory,
                AgentMemoryConfig {
                    enable_episodic_memory: false,
                    ..Default::default()
                },
            ),
            (
                MemoryType::Summary,
                AgentMemoryConfig {
                    enable_summary_memory: false,
                    ..Default::default()
                },
            ),
        ] {
            assert!(!config.kind_enabled(kind), "{kind} must be disabled");
        }
        let config = AgentMemoryConfig {
            enable_long_term_mem: false,
            ..Default::default()
        };
        assert!(!config.kind_enabled(MemoryType::Coding));
        assert!(config.kind_enabled(MemoryType::Summary));
    }

    #[test]
    fn agent_json_round_trip_preserves_every_field() {
        let config = AgentMemoryConfig {
            enable_long_term_mem: false,
            enable_user_profile: false,
            enable_semantic_memory: false,
            enable_episodic_memory: false,
            enable_summary_memory: false,
            mem_variables: vec!["city".to_owned()],
        };
        let json = serde_json::to_value(&config).unwrap();
        for key in [
            "enable_long_term_mem",
            "enable_user_profile",
            "enable_semantic_memory",
            "enable_episodic_memory",
            "enable_summary_memory",
            "mem_variables",
        ] {
            assert!(json.get(key).is_some(), "missing {key}");
        }
        let restored: AgentMemoryConfig = serde_json::from_value(json).unwrap();
        assert_eq!(restored, config);
        let tolerant: AgentMemoryConfig =
            serde_json::from_value(serde_json::json!({"enable_user_profile": false, "x": 1}))
                .unwrap();
        assert!(!tolerant.enable_user_profile);
        assert!(tolerant.enable_long_term_mem);
    }

    #[test]
    fn scope_config_defaults_are_empty() {
        let config = MemoryScopeConfig::default();
        assert!(config.user_profile_definition.is_none());
        assert!(config.semantic_memory_definition.is_none());
        assert!(config.episodic_memory_definition.is_none());
    }

    #[test]
    fn scope_config_json_round_trip_and_unknown_fields() {
        let config = MemoryScopeConfig {
            user_profile_definition: Some("identity".to_owned()),
            semantic_memory_definition: Some("facts".to_owned()),
            episodic_memory_definition: None,
        };
        let json = serde_json::to_value(&config).unwrap();
        assert_eq!(
            json["user_profile_definition"],
            serde_json::json!("identity")
        );
        assert_eq!(json["episodic_memory_definition"], serde_json::Value::Null);
        let restored: MemoryScopeConfig = serde_json::from_value(json).unwrap();
        assert_eq!(restored, config);
        let tolerant: MemoryScopeConfig =
            serde_json::from_value(serde_json::json!({"not_a_field": 1})).unwrap();
        assert!(tolerant.episodic_memory_definition.is_none());
    }
}
