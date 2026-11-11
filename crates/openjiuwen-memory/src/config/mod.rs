//! [`MemoryEngineConfig`] and the nested configuration types of the memory crate.
//!
//! [`MemoryEngineConfig`] is the single top-level config; each section it contains
//! lives in its own submodule. All types are plain `serde` data with `snake_case` JSON
//! names, tolerate unknown fields, and validate their invariants with
//! `openjiuwen_common::error::Error::Config`.
pub mod coding;
mod dedupe;
pub mod engine;
pub mod extraction;
pub mod graph;
pub mod ranking;
pub mod storage;

pub use coding::CodingMemoryConfig;
pub use dedupe::{DedupeConfig, DedupePolicy};
pub use engine::MemoryEngineConfig;
pub use extraction::{AgentMemoryConfig, ExtractionConfig, MemoryScopeConfig};
pub use graph::GraphMemoryConfig;
pub use ranking::RankingConfig;
pub use storage::StorageBackend;

use openjiuwen_common::error::{Error, Result};

/// Returns `Error::Config` unless `value` is greater than zero.
///
/// Shared by the nested config sections so every size/count knob reports the same
/// message shape: `"<field> must be greater than zero (got <value>)"`.
pub(crate) fn ensure_positive(field: &str, value: usize) -> Result<()> {
    if value == 0 {
        return Err(Error::config(format!(
            "{field} must be greater than zero (got {value})"
        )));
    }
    Ok(())
}

/// Returns `Error::Config` unless `value` is a finite number in `0.0..=1.0`.
///
/// Shared by the nested config sections so every threshold-like knob reports the same
/// message shape: `"<field> must be between 0.0 and 1.0 (got <value>)"`.
pub(crate) fn ensure_unit_interval(field: &str, value: f32) -> Result<()> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(Error::config(format!(
            "{field} must be between 0.0 and 1.0 (got {value})"
        )));
    }
    Ok(())
}

/// Returns `Error::Config` unless `value` is a finite, non-negative weight.
///
/// Shared by the ranking section so every weight reports the same message shape:
/// `"<field> must be finite and non-negative (got <value>)"`.
pub(crate) fn ensure_non_negative(field: &str, value: f32) -> Result<()> {
    if !value.is_finite() || value < 0.0 {
        return Err(Error::config(format!(
            "{field} must be finite and non-negative (got {value})"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positive_helper_names_the_field_and_value() {
        assert!(ensure_positive("default_top_k", 5).is_ok());
        let message = match ensure_positive("default_top_k", 0) {
            Err(Error::Config(message)) => message,
            other => panic!("expected a config error, got {other:?}"),
        };
        assert_eq!(message, "default_top_k must be greater than zero (got 0)");
    }

    #[test]
    fn unit_interval_helper_rejects_non_finite_and_out_of_range() {
        assert!(ensure_unit_interval("default_threshold", 0.0).is_ok());
        assert!(ensure_unit_interval("default_threshold", 1.0).is_ok());
        for value in [-0.01, 1.01, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(
                ensure_unit_interval("default_threshold", value).is_err(),
                "value {value} must be rejected"
            );
        }
    }

    #[test]
    fn non_negative_helper_rejects_negative_and_non_finite() {
        assert!(ensure_non_negative("similarity_weight", 0.0).is_ok());
        for value in [-0.5, f32::NAN, f32::NEG_INFINITY] {
            assert!(
                ensure_non_negative("similarity_weight", value).is_err(),
                "value {value} must be rejected"
            );
        }
    }
}
