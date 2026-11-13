//! Deferred Apache Pulsar backend.
//!
//! Ports the public surface of `openjiuwen/extensions/message_queue/pulsar/`
//! (v0.1.20): the configuration ([`PulsarConfig`]), the provider entry points
//! (`connect`, `start_with_retry`), and the constants Python exposed. No broker
//! client is compiled in and no network call is made.
//!
//! # Deferred scope
//!
//! Both connection entry points validate their configuration and then answer
//! [`Error::Unsupported`] with `PULSAR_DEFERRED_REASON`, so a host discovers the
//! deferral at connect time instead of silently running on a local queue.
//! [`PulsarMessageQueue`] is an introspection handle only — it exposes the
//! validated config and the Python constants but has no `start`/`send`/
//! `subscribe`, so no path can fabricate a live transport. Nothing is retried,
//! slept on, or replaced by another backend.
//!
//! # Why this is deferred
//!
//! Pulsar is not an HTTP service with a plain-JSON wire. A client opens a
//! length-prefixed, protobuf-framed binary connection to the broker's own TCP
//! ports — `pulsar://` (6650) for plaintext and `pulsar+ssl://` (6651) for TLS —
//! and drives a separate HTTP admin API for topics, subscriptions and
//! namespaces. Python delegates all of that to `pulsar-client`, a pybind11
//! wrapper around the C++ `libpulsar` client, so the framing and the generated
//! protobuf types live in native code. This workspace has no Pulsar client
//! crate, the offline registry carries no `prost`/`tonic` protobuf code
//! generation to write one against, and there is no broker for the test suite
//! to talk to. A hand-rolled framing layer would be unverifiable, so the
//! capability stays typed as [`Error::Unsupported`] with an actionable reason.
//!
//! # Migration path
//!
//! Use the in-process queue through `create_queue` with
//! `QueueConfig::in_memory()`, i.e. the kernel's `InMemoryMessageQueue` behind
//! the same [`MessageQueue`] trait; `QueueBridge` then adds the JSON
//! request/reply, publish/subscribe and topic-stream seams a host would use over
//! Pulsar. A live backend would need a protobuf-framed client over
//! `tokio::net::TcpStream` (or a new, offline-vendored client dependency) plus a
//! broker to test against.
#![allow(clippy::module_name_repetitions)]

use std::sync::Arc;
use std::time::Duration;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_core::runner::mq::MessageQueue;
use serde::{Deserialize, Serialize};

/// `tracing` target of every event this module emits.
const TRACING_TARGET: &str = "openjiuwen_extensions::message_queue";

/// Reason carried by every deferred Pulsar entry point in this module.
///
/// Names the backend, what is missing (the protobuf-framed `pulsar-client`
/// protocol and the codegen a port would need), and the supported alternative
/// (the in-process queue). The long rationale lives in the module docs above.
pub const PULSAR_DEFERRED_REASON: &str = "pulsar message queue backend is deferred: Pulsar's binary protocol is protobuf-framed over `pulsar://` (6650) / `pulsar+ssl://` (6651) plus an HTTP admin API, so it needs a `pulsar-client`-equivalent crate and `prost`/`tonic` protobuf codegen that this offline workspace does not have (and there is no broker for the test suite); use the in-process queue instead (`openjiuwen_runner::mq::InMemoryMessageQueue`, i.e. `create_queue(&QueueConfig::in_memory())`).";

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
/// the broker connection itself is deferred. The handle deliberately exposes no
/// `start`/`send`/`subscribe`, so it cannot stand in for a live transport.
pub struct PulsarMessageQueue {
    /// Validated settings.
    config: PulsarConfig,
}

impl PulsarMessageQueue {
    /// Validates `config` and builds the handle.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] when `max_workers` is zero, naming the field. The
    /// transport deferral ([`PULSAR_DEFERRED_REASON`]) is reported by
    /// [`connect`] and [`start_with_retry`], never here, so configuration
    /// mistakes are always diagnosed before the deferral.
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
/// [`Error::Validation`] rather than a deferral error. A valid config answers
/// [`Error::Unsupported`] carrying [`PULSAR_DEFERRED_REASON`]: no socket is
/// opened and no substitute queue is returned.
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
    Err(Error::Unsupported(PULSAR_DEFERRED_REASON.to_owned()))
}

/// Connects with a retry budget and returns the queue — deferred.
///
/// The retry budget is reported at `debug` but never consumed: this port does
/// not sleep, reconnect, or fall back to another transport. A valid config
/// answers [`Error::Unsupported`] carrying [`PULSAR_DEFERRED_REASON`].
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
    Err(Error::Unsupported(PULSAR_DEFERRED_REASON.to_owned()))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]

    use std::time::Duration;

    use openjiuwen_common::error::ErrorKind;
    use serde_json::json;

    use super::*;

    /// Asserts the deferral text is precise and actionable.
    ///
    /// `tests/adapters.rs` (a crate-boundary suite) only checks `pulsar` and
    /// `in-process`; the protocol and dependency phrases are asserted here so
    /// the reason cannot silently regress to a vague one-liner.
    fn assert_deferred_reason(message: &str) {
        assert!(message.contains("pulsar"), "message was {message}");
        assert!(message.contains("in-process"), "message was {message}");
        assert!(message.contains("protobuf"), "message was {message}");
        assert!(
            message.contains("pulsar-client") || message.contains("broker"),
            "message was {message}"
        );
        assert!(
            message.contains("openjiuwen_runner::mq::InMemoryMessageQueue"),
            "message was {message}"
        );
        assert!(
            message.contains("QueueConfig::in_memory()"),
            "message was {message}"
        );
    }

    #[test]
    fn deferred_reason_names_the_protocol_and_the_alternative() {
        assert_deferred_reason(PULSAR_DEFERRED_REASON);
        // Both ports and the admin API are what a live client would have to speak.
        assert!(PULSAR_DEFERRED_REASON.contains("6650"));
        assert!(PULSAR_DEFERRED_REASON.contains("6651"));
        assert!(PULSAR_DEFERRED_REASON.contains("admin API"));
        // The codegen gap is named, not just the crate gap.
        assert!(PULSAR_DEFERRED_REASON.contains("prost"));
        // Python counterpart: openjiuwen/extensions/message_queue/pulsar/.
    }

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
    fn config_tolerates_unknown_fields() {
        let decoded: PulsarConfig =
            serde_json::from_value(json!({"url": "pulsar://broker:6650", "future_field": 1}))
                .expect("unknown fields are tolerated");
        assert_eq!(decoded.url.as_deref(), Some("pulsar://broker:6650"));
        assert_eq!(decoded.max_workers, 8);
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
        assert_eq!(MAX_PRODUCERS, 10_000);
        assert_eq!(DEFAULT_SUBSCRIPTION_NAME, "default");
        assert_eq!(DEFAULT_MAX_RETRIES, 3);
        assert!((DEFAULT_RETRY_DELAY_SECS - 1.0).abs() < f64::EPSILON);
    }

    // Port of tests/unit_tests/extensions/message_queue/test_start_with_retry.py
    // (the transport is deferred with an actionable reason).
    #[tokio::test]
    async fn connect_is_deferred_with_an_actionable_reason() {
        let error = connect(PulsarConfig::new("pulsar://localhost:6650"))
            .await
            .err()
            .expect("deferred");
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        assert_eq!(error.message(), PULSAR_DEFERRED_REASON);
        assert_deferred_reason(&error.to_string());
    }

    #[tokio::test]
    async fn connect_validates_before_deferring() {
        let config = PulsarConfig {
            max_workers: 0,
            ..PulsarConfig::default()
        };
        let error = connect(config).await.err().expect("invalid config");
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("max_workers"));
    }

    #[tokio::test]
    async fn start_with_retry_is_deferred_with_an_actionable_reason() {
        let error = start_with_retry(
            PulsarConfig::default(),
            DEFAULT_MAX_RETRIES,
            Duration::from_secs_f64(DEFAULT_RETRY_DELAY_SECS),
        )
        .await
        .err()
        .expect("deferred");
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        assert_eq!(error.message(), PULSAR_DEFERRED_REASON);
        assert_deferred_reason(&error.to_string());
    }

    #[tokio::test]
    async fn start_with_retry_validates_before_deferring() {
        let config = PulsarConfig {
            max_workers: 0,
            ..PulsarConfig::default()
        };
        let error = start_with_retry(config, DEFAULT_MAX_RETRIES, Duration::from_secs(1))
            .await
            .err()
            .expect("invalid config");
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("max_workers"));
    }
}
