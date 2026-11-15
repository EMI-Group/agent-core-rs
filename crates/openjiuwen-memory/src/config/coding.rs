//! [`CodingMemoryConfig`]: settings of the lite coding memory.
//!
//! Coding facts are ordinary memory records; these knobs control whether the subsystem
//! is active, how many lines its rendered Markdown index may have, and which fact
//! types are accepted.
use super::ensure_positive;
use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// Settings of the lite coding memory (`crate::lite`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct CodingMemoryConfig {
    /// Whether coding facts are written and their index is maintained (`true`).
    pub enabled: bool,
    /// Maximum number of lines in the rendered Markdown index (`200`, Python's
    /// `MAX_INDEX_LINES`).
    pub max_index_lines: usize,
    /// Accepted fact types (`["user", "feedback", "project", "reference"]`).
    pub fact_types: Vec<String>,
    /// Fact type used when a caller does not specify one (`"project"`).
    pub default_fact_type: String,
}

impl Default for CodingMemoryConfig {
    /// Returns the documented defaults (enabled, `200`, the four fact types,
    /// `"project"`).
    fn default() -> Self {
        Self {
            enabled: true,
            max_index_lines: 200,
            fact_types: ["user", "feedback", "project", "reference"]
                .iter()
                .map(|fact_type| (*fact_type).to_owned())
                .collect(),
            default_fact_type: "project".to_owned(),
        }
    }
}

impl CodingMemoryConfig {
    /// Validates the index budget, the accepted fact types, and the default fact type.
    ///
    /// Called by `MemoryEngineConfig::validate` regardless of [`Self::enabled`], so a
    /// config that would silently reject every fact once enabled fails up front: an empty
    /// `fact_types` list, a blank entry, and a `default_fact_type` the list does not
    /// contain are all `Error::Config` naming the field and the offending value.
    pub(crate) fn validate(&self) -> Result<()> {
        ensure_positive("max_index_lines", self.max_index_lines)?;
        if self.fact_types.is_empty() {
            return Err(Error::config(
                "fact_types must not be empty; at least one fact type must be accepted",
            ));
        }
        for fact_type in &self.fact_types {
            if fact_type.trim().is_empty() {
                return Err(Error::config(
                    "fact_types must not contain an empty fact type",
                ));
            }
        }
        if self.default_fact_type.trim().is_empty() {
            return Err(Error::config(format!(
                "default_fact_type must not be empty (got {:?})",
                self.default_fact_type
            )));
        }
        if !self.fact_types.contains(&self.default_fact_type) {
            return Err(Error::config(format!(
                "default_fact_type `{}` is not in fact_types ({})",
                self.default_fact_type,
                self.fact_types.join(", ")
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn error_message(config: &CodingMemoryConfig) -> String {
        match config.validate() {
            Err(Error::Config(message)) => message,
            other => panic!("expected a config error, got {other:?}"),
        }
    }

    #[test]
    fn defaults_match_the_specification() {
        let config = CodingMemoryConfig::default();
        assert!(config.enabled);
        assert_eq!(config.max_index_lines, 200);
        assert_eq!(
            config.fact_types,
            vec![
                "user".to_owned(),
                "feedback".to_owned(),
                "project".to_owned(),
                "reference".to_owned()
            ]
        );
        assert_eq!(config.default_fact_type, "project");
        assert!(config.validate().is_ok());
    }

    #[test]
    fn json_round_trip_preserves_every_field() {
        let config = CodingMemoryConfig {
            enabled: false,
            max_index_lines: 1,
            fact_types: vec!["note".to_owned()],
            default_fact_type: "note".to_owned(),
        };
        let json = serde_json::to_value(&config).unwrap();
        for key in [
            "enabled",
            "max_index_lines",
            "fact_types",
            "default_fact_type",
        ] {
            assert!(json.get(key).is_some(), "missing {key}");
        }
        let restored: CodingMemoryConfig = serde_json::from_value(json).unwrap();
        assert_eq!(restored, config);
    }

    #[test]
    fn unknown_fields_are_ignored() {
        let config: CodingMemoryConfig = serde_json::from_value(serde_json::json!({
            "max_index_lines": 5,
            "not_a_knob": "x",
        }))
        .unwrap();
        assert_eq!(config.max_index_lines, 5);
        assert!(config.enabled);
        assert_eq!(config.default_fact_type, "project");
    }

    #[test]
    fn rejects_zero_max_index_lines() {
        let config = CodingMemoryConfig {
            max_index_lines: 0,
            ..Default::default()
        };
        assert!(error_message(&config).contains("max_index_lines"));
    }

    #[test]
    fn rejects_empty_default_fact_type() {
        let config = CodingMemoryConfig {
            default_fact_type: String::new(),
            ..Default::default()
        };
        assert!(error_message(&config).contains("default_fact_type"));
        let config = CodingMemoryConfig {
            default_fact_type: "   ".to_owned(),
            ..Default::default()
        };
        assert!(error_message(&config).contains("default_fact_type"));
    }

    #[test]
    fn rejects_empty_fact_types_and_a_blank_entry() {
        let config = CodingMemoryConfig {
            fact_types: Vec::new(),
            ..Default::default()
        };
        let message = error_message(&config);
        assert!(message.contains("fact_types"), "{message}");
        let config = CodingMemoryConfig {
            fact_types: vec!["user".to_owned(), "  ".to_owned()],
            default_fact_type: "user".to_owned(),
            ..Default::default()
        };
        let message = error_message(&config);
        assert!(message.contains("fact_types"), "{message}");
        assert!(message.contains("empty"), "{message}");
    }

    #[test]
    fn rejects_a_default_fact_type_outside_fact_types() {
        let config = CodingMemoryConfig {
            fact_types: vec!["user".to_owned(), "project".to_owned()],
            default_fact_type: "reference".to_owned(),
            ..Default::default()
        };
        let message = error_message(&config);
        assert!(message.contains("default_fact_type"), "{message}");
        assert!(message.contains("reference"), "{message}");
        assert!(message.contains("fact_types"), "{message}");
        assert!(CodingMemoryConfig {
            fact_types: vec!["reference".to_owned()],
            default_fact_type: "reference".to_owned(),
            ..Default::default()
        }
        .validate()
        .is_ok());
    }
}
