//! [`GraphMemoryConfig`]: entity/graph memory settings.
//!
//! Graph memory is disabled by default; enabling it turns on entity extraction, graph
//! persistence, and neighbor expansion during `search_all`.
use super::{ensure_positive, ensure_unit_interval};
use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// The deepest neighbor expansion a single search may request.
///
/// It is the bound `GraphMemoryConfig::validate` enforces on
/// [`GraphMemoryConfig::max_depth`] and the ceiling a per-call
/// [`GraphSearchOptions::max_depth`](crate::graph::GraphSearchOptions::max_depth) is
/// clamped to, so neither a config nor a call can ask for an unbounded walk.
pub const MAX_SEARCH_DEPTH: usize = 4;

/// The largest per-parent neighbor budget a single search may request.
///
/// `GraphMemoryConfig::validate` only requires a positive
/// [`GraphMemoryConfig::max_neighbors`]; this constant is the absolute ceiling a per-call
/// [`GraphSearchOptions::max_neighbors`](crate::graph::GraphSearchOptions::max_neighbors)
/// is clamped to, so one call cannot ask to read thousands of edges per entity.
pub const MAX_SEARCH_NEIGHBORS: usize = 256;

/// Settings of the optional entity/graph memory subsystem.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct GraphMemoryConfig {
    /// Whether graph memory is built and searched (`false`).
    pub enabled: bool,
    /// Neighbor-expansion depth from the seed entities; `0` returns seeds only (`1`).
    pub max_depth: usize,
    /// Maximum number of neighbors expanded per seed entity (`8`).
    pub max_neighbors: usize,
    /// Minimum score an entity or relation must reach to be returned (`0.3`).
    pub min_score: f32,
    /// Whether relations are extracted alongside entities (`true`).
    pub extract_relations: bool,
    /// Maximum number of entities extracted from one text (`16`).
    pub max_entities: usize,
}

impl Default for GraphMemoryConfig {
    /// Returns the documented defaults (disabled, depth `1`, `8`, `0.3`, relations
    /// on, `16`).
    fn default() -> Self {
        Self {
            enabled: false,
            max_depth: 1,
            max_neighbors: 8,
            min_score: 0.3,
            extract_relations: true,
            max_entities: 16,
        }
    }
}

impl GraphMemoryConfig {
    /// Validates the graph limits.
    ///
    /// Called by `MemoryEngineConfig::validate`; each failure is `Error::Config`
    /// naming the field and the offending value.
    pub(crate) fn validate(&self) -> Result<()> {
        if self.max_depth > MAX_SEARCH_DEPTH {
            return Err(Error::config(format!(
                "max_depth must be at most {MAX_SEARCH_DEPTH} (got {})",
                self.max_depth
            )));
        }
        ensure_positive("max_neighbors", self.max_neighbors)?;
        ensure_unit_interval("min_score", self.min_score)?;
        ensure_positive("max_entities", self.max_entities)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn error_message(config: &GraphMemoryConfig) -> String {
        match config.validate() {
            Err(Error::Config(message)) => message,
            other => panic!("expected a config error, got {other:?}"),
        }
    }

    #[test]
    fn defaults_match_the_specification() {
        let config = GraphMemoryConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.max_depth, 1);
        assert_eq!(config.max_neighbors, 8);
        assert!((config.min_score - 0.3).abs() < f32::EPSILON);
        assert!(config.extract_relations);
        assert_eq!(config.max_entities, 16);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn json_round_trip_preserves_every_field() {
        let config = GraphMemoryConfig {
            enabled: true,
            max_depth: 0,
            max_neighbors: 1,
            min_score: 1.0,
            extract_relations: false,
            max_entities: 1,
        };
        let json = serde_json::to_value(&config).unwrap();
        for key in [
            "enabled",
            "max_depth",
            "max_neighbors",
            "min_score",
            "extract_relations",
            "max_entities",
        ] {
            assert!(json.get(key).is_some(), "missing {key}");
        }
        let restored: GraphMemoryConfig = serde_json::from_value(json).unwrap();
        assert_eq!(restored, config);
    }

    #[test]
    fn unknown_fields_are_ignored() {
        let config: GraphMemoryConfig = serde_json::from_value(serde_json::json!({
            "enabled": true,
            "not_a_knob": [1, 2],
        }))
        .unwrap();
        assert!(config.enabled);
        assert_eq!(config.max_depth, 1);
    }

    #[test]
    fn rejects_depth_above_four() {
        let config = GraphMemoryConfig {
            max_depth: 5,
            ..Default::default()
        };
        let message = error_message(&config);
        assert!(message.contains("max_depth"));
        assert!(message.contains('5'));
        assert!(GraphMemoryConfig {
            max_depth: 4,
            ..Default::default()
        }
        .validate()
        .is_ok());
    }

    #[test]
    fn rejects_zero_max_neighbors() {
        let config = GraphMemoryConfig {
            max_neighbors: 0,
            ..Default::default()
        };
        assert!(error_message(&config).contains("max_neighbors"));
    }

    #[test]
    fn rejects_zero_max_entities() {
        let config = GraphMemoryConfig {
            max_entities: 0,
            ..Default::default()
        };
        let message = error_message(&config);
        assert!(message.contains("max_entities"), "{message}");
        assert!(message.contains('0'), "{message}");
    }

    #[test]
    fn rejects_out_of_range_min_score() {
        for value in [-0.1, 1.1, f32::NAN] {
            let config = GraphMemoryConfig {
                min_score: value,
                ..Default::default()
            };
            assert!(
                error_message(&config).contains("min_score"),
                "value {value} must be rejected"
            );
        }
    }
}
