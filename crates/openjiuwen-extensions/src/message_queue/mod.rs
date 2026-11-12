//! Message-queue factory, JSON bridge, and the deferred Pulsar backend.
//!
//! Ports `openjiuwen/extensions/message_queue/` (v0.1.20). The kernel already
//! owns the queue itself ([`InMemoryMessageQueue`] behind the [`MessageQueue`]
//! trait); this module adds the three things Python's `extensions` package
//! contributes:
//!
//! - [`QueueConfig`] / [`create_queue`] — a config-driven factory that maps the
//!   Python `MessageQueueConfig.type` discriminators onto the kernel queue;
//! - [`bridge::QueueBridge`] — a [`serde_json::Value`](serde_json::Value)
//!   adapter with request/reply, publish/subscribe, and topic streams;
//! - `pulsar` — the Apache Pulsar surface, preserved name-for-name but
//!   deferred with [`openjiuwen_common::error::Error::Unsupported`].
//!
//! Like the runner's own [`create_message_queue`](openjiuwen_core::runner::mq::create_message_queue),
//! [`create_queue`] **builds but does not start** the queue, so the caller (or
//! `Runner`) owns the lifecycle and tests can inspect it first.
//!
//! # Examples
//!
//! ```
//! use std::sync::Arc;
//!
//! use async_trait::async_trait;
//! use openjiuwen_common::error::Result;
//! use openjiuwen_extensions::message_queue::{
//!     create_queue, JsonMessageHandler, MessageQueue, QueueBridge, QueueConfig,
//! };
//! use serde_json::{json, Value};
//!
//! struct Doubler;
//!
//! #[async_trait]
//! impl JsonMessageHandler for Doubler {
//!     async fn handle_json(&self, payload: Value) -> Result<Value> {
//!         Ok(json!(payload.as_i64().unwrap_or_default() * 2))
//!     }
//! }
//!
//! # async fn example() -> Result<()> {
//! let queue = create_queue(&QueueConfig::in_memory()).await?;
//! queue.start().await?;
//! let bridge = QueueBridge::new(queue);
//! bridge.subscribe_json("math", Arc::new(Doubler)).await?;
//! bridge.publish_json("math", json!(3)).await?;
//! let reply = bridge.request_json("math", json!(21), None).await?;
//! assert_eq!(reply, json!(42));
//! bridge.queue().stop().await?;
//! # Ok(())
//! # }
//! ```
use std::sync::Arc;
use std::time::Duration;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_core::runner::mq::{MessageQueueConfig, MessageQueueKind};
use openjiuwen_core::runner::{DEFAULT_REQUEST_TIMEOUT_SECS, DEFAULT_TOPIC_CAPACITY};
use serde::{Deserialize, Serialize};

pub mod bridge;
#[cfg(feature = "pulsar")]
pub mod pulsar;

#[cfg(feature = "pulsar")]
pub use pulsar::{PulsarConfig, PulsarMessageQueue};

pub use bridge::{JsonMessageHandler, QueueBridge};
pub use openjiuwen_core::runner::mq::{
    InMemoryMessageQueue, MessageHandler, MessageQueue, QueueMessage, SUCCESS,
};

/// Message returned for every deferred Pulsar entry point.
///
/// Shared with the `pulsar` module so the factory and the backend answer with
/// the same text; the in-process queue is named as the supported alternative.
pub(crate) const PULSAR_DEFERRED: &str =
    "pulsar message queue backend is deferred; use the in-process queue";

/// Which transport [`create_queue`] builds.
///
/// Python's `MessageQueueType` also carried `fake`, which the kernel spells
/// [`InMemory`](Self::InMemory).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QueueKind {
    /// The tokio-native in-process queue ([`InMemoryMessageQueue`]).
    #[default]
    InMemory,
    /// Apache Pulsar (deferred).
    Pulsar,
}

impl QueueKind {
    /// The `snake_case` name used in configuration and serde output.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::InMemory => "in_memory",
            Self::Pulsar => "pulsar",
        }
    }
}

/// Pulsar wire-format version (Python `WireVersion`, default `V1`).
///
/// Preserved for configuration parity; the transport that would interpret it is
/// deferred.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WireVersion {
    /// The original framing (Python default).
    #[default]
    V1,
    /// The newer framing.
    V2,
}

/// Pulsar subscription-key encoding (Python `KeyEncoding`, default `Literal`).
///
/// Preserved for configuration parity; the transport that would interpret it is
/// deferred.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KeyEncoding {
    /// Use the topic string as-is (Python default).
    #[default]
    Literal,
    /// Escape reserved characters in the topic.
    Escaped,
}

/// Settings for [`create_queue`].
///
/// Mirrors the kernel's [`MessageQueueConfig`] and adds the two Pulsar-only
/// knobs Python kept on the extensions side. Defaults match the kernel:
/// in-process, capacity
/// [`DEFAULT_TOPIC_CAPACITY`](openjiuwen_core::runner::DEFAULT_TOPIC_CAPACITY),
/// a 30-second request deadline, and eight workers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct QueueConfig {
    /// Which transport to build.
    pub kind: QueueKind,
    /// Bounded per-topic capacity (in-flight deliveries).
    pub capacity: usize,
    /// Deadline applied by `request` when the caller passes none; `None` waits
    /// forever.
    pub request_timeout: Option<Duration>,
    /// Broker URL for `pulsar` (ignored by `in_memory`).
    pub url: Option<String>,
    /// Broker worker threads (a Pulsar concern; preserved in-memory).
    pub max_workers: usize,
    /// Pulsar wire format (Pulsar only).
    pub wire_version: WireVersion,
    /// Pulsar subscription-key encoding (Pulsar only).
    pub legacy_key_encoding: KeyEncoding,
}

impl Default for QueueConfig {
    fn default() -> Self {
        Self {
            kind: QueueKind::InMemory,
            capacity: DEFAULT_TOPIC_CAPACITY,
            request_timeout: Some(Duration::from_secs_f64(DEFAULT_REQUEST_TIMEOUT_SECS)),
            url: None,
            max_workers: 8,
            wire_version: WireVersion::V1,
            legacy_key_encoding: KeyEncoding::Literal,
        }
    }
}

impl QueueConfig {
    /// Builds the documented in-process configuration.
    #[must_use]
    pub fn in_memory() -> Self {
        Self::default()
    }

    /// Builds a Pulsar configuration for `url`, keeping every other default.
    #[must_use]
    pub fn pulsar(url: impl Into<String>) -> Self {
        Self {
            kind: QueueKind::Pulsar,
            url: Some(url.into()),
            ..Self::default()
        }
    }

    /// Returns the request deadline in seconds, as the kernel config expects.
    ///
    /// `None` stays `None`; a non-positive duration is passed through and the
    /// kernel reads it as "no default deadline", exactly like Python's `0.0`.
    #[must_use]
    pub fn request_timeout_secs(&self) -> Option<f64> {
        self.request_timeout.as_ref().map(Duration::as_secs_f64)
    }

    /// Projects this configuration onto the kernel's queue config.
    fn to_kernel_config(&self) -> MessageQueueConfig {
        MessageQueueConfig {
            kind: MessageQueueKind::InMemory,
            capacity: self.capacity,
            request_timeout_secs: self.request_timeout_secs(),
            url: self.url.clone(),
            max_workers: self.max_workers,
        }
    }
}

/// Builds the queue described by `config`, **without starting it**.
///
/// The in-process variant applies `capacity` and `request_timeout` to the
/// kernel [`InMemoryMessageQueue`]; starting it stays the caller's job, exactly
/// like `Runner` expects. The Pulsar variant fails because the transport is
/// deferred — there is deliberately no silent fallback to the in-memory queue.
///
/// # Errors
///
/// [`Error::Unsupported`] when `config.kind` is [`QueueKind::Pulsar`]; the
/// message names the backend and the in-process alternative.
///
/// # Examples
///
/// ```
/// # async fn example() -> openjiuwen_common::error::Result<()> {
/// use openjiuwen_extensions::message_queue::{create_queue, QueueConfig};
///
/// let queue = create_queue(&QueueConfig::in_memory()).await?;
/// assert!(!queue.is_started());
/// queue.start().await?;
/// assert!(queue.is_started());
/// queue.stop().await?;
/// # Ok(())
/// # }
/// ```
// The async signature is part of the seam: hosts call it from the same place
// they await `Runner::start`, and a future broker backend will need to await its
// connection. The in-process path performs no I/O.
#[allow(clippy::unused_async)]
pub async fn create_queue(config: &QueueConfig) -> Result<Arc<dyn MessageQueue>> {
    match config.kind {
        QueueKind::InMemory => {
            let queue: Arc<dyn MessageQueue> = InMemoryMessageQueue::new(config.to_kernel_config());
            Ok(queue)
        }
        QueueKind::Pulsar => Err(Error::Unsupported(PULSAR_DEFERRED.to_owned())),
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]

    use std::time::Duration;

    use openjiuwen_common::error::ErrorKind;
    use serde_json::json;

    use super::*;
    use crate::message_queue::bridge::QueueBridge;

    #[test]
    fn defaults_match_the_kernel() {
        let config = QueueConfig::default();
        assert_eq!(config.kind, QueueKind::InMemory);
        assert_eq!(config.capacity, DEFAULT_TOPIC_CAPACITY);
        assert_eq!(
            config.request_timeout_secs(),
            Some(DEFAULT_REQUEST_TIMEOUT_SECS)
        );
        assert_eq!(config.max_workers, 8);
        assert_eq!(config.url, None);
        assert_eq!(config.wire_version, WireVersion::V1);
        assert_eq!(config.legacy_key_encoding, KeyEncoding::Literal);
        assert_eq!(config.wire_version, WireVersion::default());
        assert_eq!(config.legacy_key_encoding, KeyEncoding::default());
        assert_eq!(QueueKind::default().as_str(), "in_memory");
        assert_eq!(QueueKind::Pulsar.as_str(), "pulsar");
    }

    #[test]
    fn constructors_keep_the_documented_defaults() {
        assert_eq!(QueueConfig::in_memory(), QueueConfig::default());

        let pulsar = QueueConfig::pulsar("pulsar://broker:6650");
        assert_eq!(pulsar.kind, QueueKind::Pulsar);
        assert_eq!(pulsar.url.as_deref(), Some("pulsar://broker:6650"));
        assert_eq!(pulsar.capacity, QueueConfig::default().capacity);
        assert_eq!(pulsar.max_workers, 8);
        assert_eq!(pulsar.wire_version, WireVersion::V1);
    }

    #[test]
    fn request_timeout_maps_to_seconds_for_the_kernel() {
        let mut config = QueueConfig::in_memory();
        config.request_timeout = None;
        assert_eq!(config.request_timeout_secs(), None);
        config.request_timeout = Some(Duration::from_millis(1500));
        assert_eq!(config.request_timeout_secs(), Some(1.5));
        assert_eq!(config.to_kernel_config().request_timeout_secs, Some(1.5));
        assert_eq!(config.to_kernel_config().capacity, DEFAULT_TOPIC_CAPACITY);
    }

    #[test]
    fn kernel_config_carries_every_queue_field() {
        let config = QueueConfig {
            kind: QueueKind::InMemory,
            capacity: 3,
            request_timeout: Some(Duration::from_secs(2)),
            url: Some("pulsar://broker:6650".to_owned()),
            max_workers: 5,
            wire_version: WireVersion::V2,
            legacy_key_encoding: KeyEncoding::Escaped,
        };
        let kernel = config.to_kernel_config();
        assert_eq!(kernel.kind, MessageQueueKind::InMemory);
        assert_eq!(kernel.capacity, 3);
        assert_eq!(kernel.request_timeout_secs, Some(2.0));
        assert_eq!(kernel.url.as_deref(), Some("pulsar://broker:6650"));
        assert_eq!(kernel.max_workers, 5);
    }

    #[tokio::test]
    async fn create_queue_returns_an_unstarted_in_memory_queue() {
        let queue = create_queue(&QueueConfig::in_memory())
            .await
            .expect("queue");
        assert!(!queue.is_started());
        queue.start().await.expect("start");
        assert!(queue.is_started());
        queue.stop().await.expect("stop");
    }

    #[tokio::test]
    async fn create_queue_applies_the_configured_request_deadline() {
        let mut config = QueueConfig::in_memory();
        config.request_timeout = Some(Duration::from_millis(25));
        let queue = create_queue(&config).await.expect("queue");
        queue.start().await.expect("start");
        let bridge = QueueBridge::new(queue);
        bridge
            .subscribe_json("slow", std::sync::Arc::new(SlowHandler))
            .await
            .expect("handler");

        // No explicit timeout: the queue applies the configured 25 ms deadline.
        let error = tokio::time::timeout(
            Duration::from_secs(1),
            bridge.request_json("slow", json!(1), None),
        )
        .await
        .expect("the configured deadline fires instead of hanging")
        .expect_err("the slow handler exceeds the deadline");
        assert_eq!(error.kind(), ErrorKind::Timeout);
    }

    #[tokio::test]
    async fn create_queue_applies_the_configured_capacity() {
        let mut config = QueueConfig::in_memory();
        config.capacity = 1;
        config.request_timeout = Some(Duration::from_secs(5));
        let queue = create_queue(&config).await.expect("queue");
        queue.start().await.expect("start");
        let bridge = QueueBridge::new(queue);

        let (seen_tx, mut seen_rx) = tokio::sync::mpsc::unbounded_channel();
        let gate = std::sync::Arc::new(tokio::sync::Semaphore::new(0));
        bridge
            .subscribe_json(
                "work",
                std::sync::Arc::new(BlockingHandler {
                    seen: seen_tx,
                    gate: std::sync::Arc::clone(&gate),
                }),
            )
            .await
            .expect("handler");

        bridge
            .publish_json("work", json!("first"))
            .await
            .expect("first");
        let first = tokio::time::timeout(Duration::from_secs(1), seen_rx.recv())
            .await
            .expect("the worker starts the first delivery")
            .expect("payload");
        assert_eq!(first, json!("first"));

        // The worker is parked in the handler and the topic holds one slot, so
        // the second publish fits and the third must wait for capacity.
        bridge
            .publish_json("work", json!("second"))
            .await
            .expect("second");
        let third = tokio::time::timeout(
            Duration::from_millis(100),
            bridge.publish_json("work", json!("third")),
        )
        .await;
        assert!(
            third.is_err(),
            "a capacity-1 topic must backpressure the third publish"
        );

        gate.add_permits(1);
        tokio::time::timeout(
            Duration::from_secs(1),
            bridge.publish_json("work", json!("third")),
        )
        .await
        .expect("the third publish fits once the handler drains")
        .expect("publish");
        gate.add_permits(8);
    }

    #[tokio::test]
    async fn pulsar_queue_is_deferred() {
        let config = QueueConfig::pulsar("pulsar://localhost:6650");
        let error = create_queue(&config).await.err().expect("deferred backend");
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        let message = error.to_string();
        assert!(message.contains("pulsar"), "message was {message}");
        assert!(message.contains("in-process"), "message was {message}");
    }

    #[test]
    fn queue_config_round_trips_through_snake_case_json() {
        let config = QueueConfig {
            kind: QueueKind::Pulsar,
            capacity: 4,
            request_timeout: Some(Duration::from_millis(1500)),
            url: Some("pulsar://broker:6650".to_owned()),
            max_workers: 3,
            wire_version: WireVersion::V2,
            legacy_key_encoding: KeyEncoding::Escaped,
        };
        let encoded = serde_json::to_value(&config).expect("serialize");
        assert_eq!(encoded["kind"], json!("pulsar"));
        assert_eq!(encoded["wire_version"], json!("v2"));
        assert_eq!(encoded["legacy_key_encoding"], json!("escaped"));
        assert!(encoded.get("request_timeout").is_some());

        let decoded: QueueConfig = serde_json::from_value(encoded).expect("deserialize");
        assert_eq!(decoded, config);
    }

    #[test]
    fn queue_config_tolerates_unknown_fields_and_missing_fields() {
        let decoded: QueueConfig = serde_json::from_value(json!({
            "kind": "in_memory",
            "capacity": 16,
            "request_timeout": null,
            "url": null,
            "max_workers": 4,
            "wire_version": "v1",
            "legacy_key_encoding": "literal",
            "future_field": true,
        }))
        .expect("unknown fields are tolerated");
        assert_eq!(decoded.capacity, 16);
        assert_eq!(decoded.request_timeout, None);

        let partial: QueueConfig =
            serde_json::from_value(json!({"kind": "pulsar"})).expect("missing fields default");
        assert_eq!(partial.kind, QueueKind::Pulsar);
        assert_eq!(partial.capacity, DEFAULT_TOPIC_CAPACITY);
        assert_eq!(partial.max_workers, 8);
    }

    /// A handler that never answers within the test's deadline.
    struct SlowHandler;

    #[async_trait::async_trait]
    impl crate::message_queue::JsonMessageHandler for SlowHandler {
        async fn handle_json(
            &self,
            payload: serde_json::Value,
        ) -> openjiuwen_common::error::Result<serde_json::Value> {
            tokio::time::sleep(Duration::from_secs(10)).await;
            Ok(payload)
        }
    }

    /// A handler that parks until released, for backpressure assertions.
    struct BlockingHandler {
        /// Receives every payload the handler observes.
        seen: tokio::sync::mpsc::UnboundedSender<serde_json::Value>,
        /// Zero permits: each delivery parks until the test adds one.
        gate: std::sync::Arc<tokio::sync::Semaphore>,
    }

    #[async_trait::async_trait]
    impl crate::message_queue::JsonMessageHandler for BlockingHandler {
        async fn handle_json(
            &self,
            payload: serde_json::Value,
        ) -> openjiuwen_common::error::Result<serde_json::Value> {
            let _ = self.seen.send(payload.clone());
            let _permit = self.gate.acquire().await;
            Ok(payload)
        }
    }
}
