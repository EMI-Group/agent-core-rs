//! [`DreamConfig`]: the scheduling, retry, retention, and trigger knobs for dreaming.
//!
//! Python's `core.memory.dreaming` package hardcodes almost all of this: an interval
//! floor of 60 seconds, a 120-second initial settle delay, and no config object at all.
//! The schema below is this crate's own; it preserves those two hardcoded Python values
//! as defaults and adds the bounds a Rust caller needs to run the queue explicitly.
use chrono::Duration;
use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// Configuration for the dreaming queue and its (deferred) scheduler.
///
/// Every field has a default, JSON is `snake_case`, and missing fields take their
/// default value. [`DreamConfig::validate`] rejects values that would make a sweep loop
/// busy-spin or never finish.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct DreamConfig {
    /// Whether dreaming is enabled at all; default `true`.
    ///
    /// A disabled configuration still allows explicit jobs to be enqueued; the deferred
    /// orchestrator is the component that honours this switch.
    pub enabled: bool,
    /// Minimum seconds between sweeps; default `60`.
    ///
    /// Python's `DreamingOrchestrator` clamps its interval to a 60-second floor, so this
    /// keeps parity while letting tests shorten nothing below that floor.
    pub min_interval_secs: u64,
    /// Seconds to wait after startup before the first sweep; default `120`.
    ///
    /// Ports Python's hardcoded settle delay, which lets an application finish opening
    /// its storage before consolidation starts.
    pub initial_delay_secs: u64,
    /// Maximum jobs claimed by one sweep; default `10`.
    ///
    /// Caps the work done by a single [`DreamRepository::claim_next`](crate::dreaming::DreamRepository::claim_next)
    /// loop so a large backlog cannot starve the foreground.
    pub batch_size: usize,
    /// Attempt budget for newly enqueued jobs; default `3`.
    ///
    /// The queue itself reads the per-job `max_attempts`; this value seeds new jobs.
    pub max_attempts: u32,
    /// Seconds finished jobs are kept before [`DreamRepository::purge`](crate::dreaming::DreamRepository::purge)
    /// may remove them; default `604_800` (7 days).
    ///
    /// Keeping finished rows for a week makes failures diagnosable without growing the
    /// queue forever.
    pub retention_secs: u64,
    /// Message count that makes a [`DreamTrigger::MessageThreshold`](crate::dreaming::DreamTrigger::MessageThreshold)
    /// dream due; default `100`.
    pub message_threshold: usize,
    /// Idle seconds that make a [`DreamTrigger::Idle`](crate::dreaming::DreamTrigger::Idle)
    /// dream due; default `300`.
    pub idle_secs: u64,
    /// Cosine similarity at or above which two memories count as duplicates; default
    /// `0.92`.
    ///
    /// Matches the engine's deterministic duplicate threshold so consolidation and the
    /// write path agree on what "the same memory" means.
    pub dedupe_similarity: f32,
}

impl Default for DreamConfig {
    /// The documented defaults: enabled, 60 s interval, 120 s initial delay, 10 jobs per
    /// batch, 3 attempts, 7-day retention, 100-message / 300-second triggers, and a
    /// 0.92 dedupe threshold.
    fn default() -> Self {
        Self {
            enabled: true,
            min_interval_secs: 60,
            initial_delay_secs: 120,
            batch_size: 10,
            max_attempts: 3,
            retention_secs: 604_800,
            message_threshold: 100,
            idle_secs: 300,
            dedupe_similarity: 0.92,
        }
    }
}

impl DreamConfig {
    /// Returns the configuration with dreaming switched on or off.
    #[must_use]
    pub const fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// Returns the configuration with `batch_size` jobs claimed per sweep.
    #[must_use]
    pub const fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Returns the configuration with `max_attempts` as the default attempt budget.
    #[must_use]
    pub const fn with_max_attempts(mut self, max_attempts: u32) -> Self {
        self.max_attempts = max_attempts;
        self
    }

    /// Returns [`DreamConfig::retention_secs`] as a [`chrono::Duration`].
    ///
    /// Values chrono cannot represent saturate at [`Duration::MAX`] instead of wrapping
    /// or panicking.
    #[must_use]
    pub fn retention(&self) -> Duration {
        i64::try_from(self.retention_secs)
            .ok()
            .and_then(Duration::try_seconds)
            .unwrap_or(Duration::MAX)
    }

    /// Validates the configuration, returning `Error::Validation` naming the field.
    ///
    /// Requirements: a non-zero interval, initial delay, retention, message threshold,
    /// and idle threshold; a non-zero batch size; a `max_attempts` in `1..=100`; and a
    /// finite `dedupe_similarity` in `0.0..=1.0` (NaN is rejected).
    pub fn validate(&self) -> Result<()> {
        if self.min_interval_secs == 0 {
            return Err(Error::validation(
                "dream config `min_interval_secs` must be greater than 0",
            ));
        }
        if self.batch_size == 0 {
            return Err(Error::validation(
                "dream config `batch_size` must be greater than 0",
            ));
        }
        if !(1..=100).contains(&self.max_attempts) {
            return Err(Error::validation(format!(
                "dream config `max_attempts` must be in 1..=100 (got {})",
                self.max_attempts
            )));
        }
        if self.retention_secs == 0 {
            return Err(Error::validation(
                "dream config `retention_secs` must be greater than 0",
            ));
        }
        if self.message_threshold == 0 {
            return Err(Error::validation(
                "dream config `message_threshold` must be greater than 0",
            ));
        }
        if self.idle_secs == 0 {
            return Err(Error::validation(
                "dream config `idle_secs` must be greater than 0",
            ));
        }
        if !self.dedupe_similarity.is_finite() || !(0.0..=1.0).contains(&self.dedupe_similarity) {
            return Err(Error::validation(format!(
                "dream config `dedupe_similarity` must be finite and in 0.0..=1.0 (got {})",
                self.dedupe_similarity
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn defaults_are_documented_values() {
        let config = DreamConfig::default();
        assert!(config.enabled);
        assert_eq!(config.min_interval_secs, 60);
        assert_eq!(config.initial_delay_secs, 120);
        assert_eq!(config.batch_size, 10);
        assert_eq!(config.max_attempts, 3);
        assert_eq!(config.retention_secs, 604_800);
        assert_eq!(config.message_threshold, 100);
        assert_eq!(config.idle_secs, 300);
        assert!((config.dedupe_similarity - 0.92).abs() < f32::EPSILON);
        assert!(config.validate().is_ok());
        assert_eq!(config.retention(), Duration::days(7));
    }

    #[test]
    fn builders_and_retention_saturation() {
        let config = DreamConfig::default()
            .with_enabled(false)
            .with_batch_size(4)
            .with_max_attempts(2);
        assert!(!config.enabled);
        assert_eq!(config.batch_size, 4);
        assert_eq!(config.max_attempts, 2);
        let huge = DreamConfig {
            retention_secs: u64::MAX,
            ..DreamConfig::default()
        };
        assert_eq!(huge.retention(), Duration::MAX);
    }

    #[test]
    fn validate_rejects_zero_interval_and_batch() {
        let config = DreamConfig {
            min_interval_secs: 0,
            ..DreamConfig::default()
        };
        assert!(config
            .validate()
            .unwrap_err()
            .to_string()
            .contains("min_interval_secs"));
        let config = DreamConfig {
            batch_size: 0,
            ..DreamConfig::default()
        };
        assert!(config
            .validate()
            .unwrap_err()
            .to_string()
            .contains("batch_size"));
    }

    #[test]
    fn validate_rejects_out_of_range_attempts() {
        assert!(DreamConfig::default()
            .with_max_attempts(0)
            .validate()
            .unwrap_err()
            .to_string()
            .contains("max_attempts"));
        assert!(DreamConfig::default()
            .with_max_attempts(101)
            .validate()
            .unwrap_err()
            .to_string()
            .contains("max_attempts"));
        assert!(DreamConfig::default()
            .with_max_attempts(1)
            .validate()
            .is_ok());
        assert!(DreamConfig::default()
            .with_max_attempts(100)
            .validate()
            .is_ok());
    }

    #[test]
    fn validate_rejects_zero_retention_and_thresholds() {
        for (mutate, field) in [
            (
                (|config: &mut DreamConfig| config.retention_secs = 0) as fn(&mut DreamConfig),
                "retention_secs",
            ),
            (
                |config: &mut DreamConfig| config.message_threshold = 0,
                "message_threshold",
            ),
            (|config: &mut DreamConfig| config.idle_secs = 0, "idle_secs"),
        ] {
            let mut config = DreamConfig::default();
            mutate(&mut config);
            let err = config.validate().unwrap_err();
            assert!(matches!(err, Error::Validation(_)));
            assert!(err.to_string().contains(field), "field = {field}");
        }
    }

    #[test]
    fn validate_rejects_non_finite_or_out_of_range_similarity() {
        for value in [f32::NAN, f32::INFINITY, -0.1, 1.1] {
            let config = DreamConfig {
                dedupe_similarity: value,
                ..DreamConfig::default()
            };
            let err = config.validate().unwrap_err();
            assert!(
                err.to_string().contains("dedupe_similarity"),
                "value = {value}"
            );
        }
        for value in [0.0_f32, 0.5, 1.0] {
            let config = DreamConfig {
                dedupe_similarity: value,
                ..DreamConfig::default()
            };
            assert!(config.validate().is_ok());
        }
    }

    #[test]
    fn partial_json_fills_defaults_and_ignores_unknown_fields() {
        let config: DreamConfig =
            serde_json::from_value(json!({"batch_size": 20, "unknown": true})).unwrap();
        assert_eq!(config.batch_size, 20);
        assert_eq!(config.min_interval_secs, 60);
        assert!(config.enabled);
        let config: DreamConfig = serde_json::from_value(json!({})).unwrap();
        assert_eq!(config, DreamConfig::default());
    }

    #[test]
    fn serde_round_trip_uses_snake_case() {
        let config = DreamConfig::default();
        let value = serde_json::to_value(&config).unwrap();
        assert_eq!(value["min_interval_secs"], json!(60));
        assert_eq!(value["initial_delay_secs"], json!(120));
        assert!(value.get("enabled").is_some());
        assert_eq!(
            serde_json::from_value::<DreamConfig>(value).unwrap(),
            config
        );
    }
}
