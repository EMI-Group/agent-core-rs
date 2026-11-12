//! Deferred Apache Pulsar backend.
//!
//! Ports the public surface of `openjiuwen/extensions/message_queue/pulsar/`
//! (v0.1.20): the configuration, the connection entry points, and the constants
//! Python exposed. No broker client is compiled in and no network call is made;
//! every connection entry point validates its configuration and then answers
//! [`Error::Unsupported`] naming the in-process queue, so a host discovers the
//! deferral at build/connect time instead of silently using a local queue.
#![allow(clippy::module_name_repetitions)]

use std::sync::Arc;
use std::time::Duration;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_core::runner::mq::MessageQueue;
use serde::{Deserialize, Serialize};

use super::PULSAR_DEFERRED;

/// `tracing` target of every event this module emits.
const TRACING_TARGET: &str = "openjiuwen_extensions::message_queue";

/// Upper bound on producers per queue (Python `MAX_PRODUCERS`).
pub const MAX_PRODUCERS: usize = 10_000;

/// Subscription name used when the caller does not pick one (Python
/// `DEFAULT_SUBSCRIPTION_NAME`).
pub const DEFAULT_SUBSCRIPTION_NAME: &str = "default";

/// Default connect retry budget (Python `DEFAULT_MAX_RETRIES`).
pub const DEFAULT_MAX_RETRIES: u32 = 3;

/// Default delay between connect attempts, in seconds (Python
/// `DEFAULT_RETRY_DELAY_SECS`).
pub const DEFAULT_RETRY_DELAY_SECS: f64 = 1.0;

/// Settings for the deferred Pulsar backend (Python `PulsarConfig`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct PulsarConfig {
    /// Broker URL; `None` means the client default.
    pub url: Option<String>,
    /// Worker threads the backend would start (default 8).
    pub max_workers: usize,
}

impl Default for PulsarConfig {
    fn default() -> Self {
        Self {
            url: None,
            max_workers: 8,
        }
    }
}

impl PulsarConfig {
    /// Builds a configuration for `url`, keeping the documented defaults.
    #[must_use]
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: Some(url.into()),
            ..Self::default()
        }
    }
}

/// The deferred Pulsar queue handle.
///
/// Constructing one validates the configuration, which is all this port can do;
/// the broker connection itself is deferred.
pub struct PulsarMessageQueue {
    /// Validated settings.
    config: PulsarConfig,
}

impl PulsarMessageQueue {
    /// Validates `config` and builds the handle.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] when `max_workers` is zero, naming the field.
    pub fn new(config: PulsarConfig) -> Result<Self> {
        if config.max_workers == 0 {
            return Err(Error::Validation(
                "pulsar message queue requires max_workers > 0".to_owned(),
            ));
        }
        Ok(Self { config })
    }

    /// Returns the validated configuration.
    #[must_use]
    pub const fn config(&self) -> &PulsarConfig {
        &self.config
    }

    /// Returns the producer bound this backend would apply
    /// ([`MAX_PRODUCERS`]).
    #[must_use]
    pub const fn max_producers(&self) -> usize {
        MAX_PRODUCERS
    }

    /// Returns the subscription name this backend would use
    /// ([`DEFAULT_SUBSCRIPTION_NAME`]).
    #[must_use]
    pub const fn subscription_name(&self) -> &'static str {
        DEFAULT_SUBSCRIPTION_NAME
    }
}

/// Connects to Pulsar and returns the queue — deferred.
///
/// The configuration is validated first, so an invalid config is
/// [`Error::Validation`] rather than a deferral error.
///
/// # Errors
///
/// [`Error::Validation`] for an invalid configuration;
/// [`Error::Unsupported`] otherwise, because the transport is deferred.
// The async signature mirrors Python's `await`able connect and is part of the
// public contract; no I/O happens here.
#[allow(clippy::unused_async)]
pub async fn connect(config: PulsarConfig) -> Result<Arc<dyn MessageQueue>> {
    let queue = PulsarMessageQueue::new(config)?;
    tracing::debug!(
        target: TRACING_TARGET,
        url = ?queue.config().url,
        max_workers = queue.config().max_workers,
        "pulsar message queue connection is deferred"
    );
    Err(Error::Unsupported(PULSAR_DEFERRED.to_owned()))
}

/// Connects with a retry budget and returns the queue — deferred.
///
/// The retry budget is reported at `debug` but never consumed: this port does
/// not sleep, reconnect, or fall back to another transport.
///
/// # Errors
///
/// [`Error::Validation`] for an invalid configuration;
/// [`Error::Unsupported`] otherwise, because the transport is deferred.
// The async signature mirrors Python's `await`able connect-with-retry.
#[allow(clippy::unused_async)]
pub async fn start_with_retry(
    config: PulsarConfig,
    max_retries: u32,
    retry_delay: Duration,
) -> Result<Arc<dyn MessageQueue>> {
    let queue = PulsarMessageQueue::new(config)?;
    tracing::debug!(
        target: TRACING_TARGET,
        url = ?queue.config().url,
        max_workers = queue.config().max_workers,
        max_retries,
        retry_delay_ms = retry_delay.as_millis(),
        "pulsar message queue connect-with-retry is deferred"
    );
    Err(Error::Unsupported(PULSAR_DEFERRED.to_owned()))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]

    use std::time::Duration;

    use openjiuwen_common::error::ErrorKind;
    use serde_json::json;

    use super::*;

    #[test]
    fn default_config_matches_python() {
        let config = PulsarConfig::default();
        assert_eq!(config.url, None);
        assert_eq!(config.max_workers, 8);

        let encoded = serde_json::to_value(&config).expect("serialize");
        assert_eq!(encoded, json!({"url": null, "max_workers": 8}));
        let decoded: PulsarConfig = serde_json::from_value(encoded).expect("deserialize");
        assert_eq!(decoded, config);
    }

    #[test]
    fn config_constructor_sets_the_url() {
        let config = PulsarConfig::new("pulsar://broker:6650");
        assert_eq!(config.url.as_deref(), Some("pulsar://broker:6650"));
        assert_eq!(config.max_workers, 8);
    }

    #[test]
    fn new_rejects_zero_workers() {
        let config = PulsarConfig {
            max_workers: 0,
            ..PulsarConfig::default()
        };
        let error = PulsarMessageQueue::new(config)
            .err()
            .expect("zero workers is invalid");
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("max_workers"));
    }

    #[test]
    fn new_accepts_valid_workers_and_exposes_the_python_constants() {
        let queue = PulsarMessageQueue::new(PulsarConfig::default()).expect("valid config");
        assert_eq!(queue.config().max_workers, 8);
        assert_eq!(queue.max_producers(), MAX_PRODUCERS);
        assert_eq!(queue.subscription_name(), DEFAULT_SUBSCRIPTION_NAME);
        assert_eq!(DEFAULT_MAX_RETRIES, 3);
        assert!((DEFAULT_RETRY_DELAY_SECS - 1.0).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn connect_is_deferred() {
        let error = connect(PulsarConfig::new("pulsar://localhost:6650"))
            .await
            .err()
            .expect("deferred");
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        let message = error.to_string();
        assert!(message.contains("pulsar"), "message was {message}");
        assert!(message.contains("in-process"), "message was {message}");
    }

    #[tokio::test]
    async fn connect_validates_before_deferring() {
        let config = PulsarConfig {
            max_workers: 0,
            ..PulsarConfig::default()
        };
        let error = connect(config).await.err().expect("invalid config");
        assert_eq!(error.kind(), ErrorKind::Validation);
    }

    #[tokio::test]
    async fn start_with_retry_is_deferred() {
        let error = start_with_retry(
            PulsarConfig::default(),
            DEFAULT_MAX_RETRIES,
            Duration::from_secs_f64(DEFAULT_RETRY_DELAY_SECS),
        )
        .await
        .err()
        .expect("deferred");
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        let message = error.to_string();
        assert!(message.contains("pulsar"), "message was {message}");
        assert!(message.contains("in-process"), "message was {message}");
    }
}
