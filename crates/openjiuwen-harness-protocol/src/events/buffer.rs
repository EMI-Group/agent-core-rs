//! Bounded observation-buffer contract advertised by a harness implementation.
//!
//! Mirrors the Python `EventBufferConfig` dataclass (v0.1.20): a positive `capacity` and the
//! overflow [`EventOverflowPolicy`] the harness applies when the buffer is full.
//!
//! Python's `capacity <= 0` check is expressed against `u64`, so only zero is rejectable;
//! negative wire values are already rejected by serde.

use serde::{Deserialize, Serialize};

use crate::errors::{HarnessError, HarnessResult};
use crate::events::enums::EventOverflowPolicy;
use crate::validation::Validate;

/// Bounded observation-buffer contract advertised by an implementation.
///
/// Wire keys are always present; `capacity` defaults to `1024` and `overflow` to
/// `coalesce_or_block`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub struct EventBufferConfig {
    /// Maximum number of retained events; must be positive.
    #[serde(default = "default_capacity")]
    pub capacity: u64,
    /// Behavior applied when the buffer is full.
    #[serde(default)]
    pub overflow: EventOverflowPolicy,
}

/// Default `capacity` of a buffer, shared by `Default` and the serde default.
///
/// A plain `#[serde(default)]` on a `u64` would decode a missing key as `0`,
/// which [`EventBufferConfig::validate`] then rejects — so Python's
/// `{"overflow": "coalesce_or_block"}` (no `capacity`) would fail to validate,
/// and a consumer that skips validation would run a one-slot buffer.
fn default_capacity() -> u64 {
    1024
}

impl Default for EventBufferConfig {
    fn default() -> Self {
        Self {
            capacity: default_capacity(),
            overflow: EventOverflowPolicy::CoalesceOrBlock,
        }
    }
}

impl Validate for EventBufferConfig {
    /// Checks that `capacity` is positive, in Python's wording.
    fn validate(&self) -> HarnessResult<()> {
        if self.capacity == 0 {
            return Err(HarnessError::validation(
                "event buffer capacity must be positive",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    // Tests assert by panicking on failure, which is the intent of a test.
    #![allow(clippy::expect_used, clippy::unwrap_used)]

    use serde_json::json;

    use super::EventBufferConfig;
    use crate::events::EventOverflowPolicy;
    use crate::Validate;

    #[test]
    fn default_matches_python() {
        let config = EventBufferConfig::default();
        assert_eq!(config.capacity, 1024);
        assert_eq!(config.overflow, EventOverflowPolicy::CoalesceOrBlock);
        assert!(config.validate().is_ok());
        assert_eq!(
            serde_json::to_value(config).expect("serializes"),
            json!({"capacity": 1024, "overflow": "coalesce_or_block"})
        );
    }

    #[test]
    fn keys_are_always_present_and_unknown_keys_tolerated() {
        let decoded: EventBufferConfig = serde_json::from_value(json!({
            "capacity": 8,
            "overflow": "drop_best_effort_or_block",
            "future_key": true,
        }))
        .expect("unknown keys are tolerated");
        assert_eq!(decoded.capacity, 8);
        assert_eq!(decoded.overflow, EventOverflowPolicy::DropBestEffortOrBlock);

        let encoded = serde_json::to_value(decoded).expect("serializes");
        let keys: Vec<&str> = encoded
            .as_object()
            .expect("object")
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(keys, ["capacity", "overflow"]);
    }

    #[test]
    fn a_missing_capacity_decodes_and_validates_as_1024() {
        // Python's shape: the overflow policy alone, no `capacity` key.
        let decoded: EventBufferConfig = serde_json::from_value(json!({
            "overflow": "coalesce_or_block",
        }))
        .expect("a missing capacity is a default, not a decode error");
        assert_eq!(decoded.capacity, 1024);
        assert_eq!(decoded, EventBufferConfig::default());
        assert!(
            decoded.validate().is_ok(),
            "the serde default must agree with `Default`: {decoded:?}"
        );

        let empty: EventBufferConfig = serde_json::from_value(json!({})).expect("all keys default");
        assert_eq!(empty, EventBufferConfig::default());
    }

    #[test]
    fn an_explicit_zero_capacity_still_fails_validation_after_decoding() {
        let decoded: EventBufferConfig = serde_json::from_value(json!({
            "capacity": 0,
            "overflow": "coalesce_or_block",
        }))
        .expect("zero is a representable u64");
        assert_eq!(decoded.capacity, 0, "an explicit 0 is not defaulted");
        let error = decoded.validate().expect_err("capacity must be positive");
        assert_eq!(error.message(), "event buffer capacity must be positive");
    }

    #[test]
    fn zero_capacity_is_rejected_with_python_message() {
        let config = EventBufferConfig {
            capacity: 0,
            overflow: EventOverflowPolicy::Block,
        };
        let error = config.validate().expect_err("capacity must be positive");
        assert_eq!(error.message(), "event buffer capacity must be positive");

        let negative = serde_json::from_value::<EventBufferConfig>(json!({
            "capacity": -1,
            "overflow": "block",
        }));
        assert!(negative.is_err());
    }
}
