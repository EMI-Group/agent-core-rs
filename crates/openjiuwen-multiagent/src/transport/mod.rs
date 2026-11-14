//! The team transport seam: where one team's messages physically travel.
//!
//! Python parity: `core/multi_agent/team_runtime/message_bus.py` hides its
//! transport behind `MessageQueueInMemory`. This module makes that seam
//! explicit so the same team, cards, and strategies can run over different
//! backends:
//!
//! - [`InProcessTransport`] — 1:1 delegation to the crate's [`MessageBus`](crate::MessageBus)
//!   (per-topic FIFO workers, awaited pub/sub fan-out, synchronous replies).
//! - [`RedisTransport`] — envelopes are JSON documents held in Redis lists, so
//!   several processes can serve one team.
//! - [`TransportKind::Pulsar`] — declared but not implemented; building it
//!   returns [`Error::Unsupported`] with a reason instead of silently degrading
//!   to another backend.
//!
//! Every backend keeps the contract the strategies rely on: `send` returns the
//! recipient's reply value (or the recipient's error unchanged), `publish` names
//! its subscribers through the logical registry, and `subscribers` returns ids
//! sorted, so routing and tests stay deterministic.
//!
//! # Redis key scheme
//!
//! [`RedisTransport`] derives every key from `{key_prefix}:{team_id}`:
//!
//! - `{prefix}:{team}:p2p:{recipient}` — the point-to-point inbox of one agent;
//!   `send` pushes the envelope JSON with `RPUSH` and a worker of the receiving
//!   transport polls it with `RPOP`.
//! - `{prefix}:{team}:reply:{message_id}` — the reply list of one request, keyed
//!   by [`MessageEnvelope::message_id`]; the worker pushes
//!   `{"value": ...}` or `{"error": "..."}` and the waiting `send` polls until
//!   the deadline.
//! - `{prefix}:{team}:pubsub:{subscriber}` — the pub/sub inbox of one agent; a
//!   publisher pushes one copy per sorted subscriber.
//!
//! Only agents registered with [`RedisTransport::watch_agent`] get inbox
//! workers, because `RPOP` removes a message: a transport must never poll a
//! queue that belongs to another process.
//!
//! # Divergence from the in-process transport
//!
//! Redis delivery is asynchronous: `publish` returns once every `RPUSH`
//! succeeded, which is before (or even without) any handler running.
//! `InProcessTransport::publish` instead returns after the fan-out completed.
//! `send` stays request/response on both backends.
//!
//! # `Pulsar`
//!
//! The Pulsar backend is deferred, not stubbed: [`build_transport`] returns
//! [`ErrorKind::Unsupported`](openjiuwen_common::error::ErrorKind::Unsupported)
//! naming Pulsar, so a misconfiguration fails loudly at construction.

mod inproc;
mod redis;

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub use inproc::InProcessTransport;
pub use redis::{RedisTransport, RedisTransportConfig, DEFAULT_POLL_INTERVAL_MS};

use crate::envelope::MessageEnvelope;
use crate::message_bus::{MessageBusConfig, MessageHandler};

/// Prefix every Redis key of a transport starts with.
pub const DEFAULT_KEY_PREFIX: &str = "openjiuwen:team";

/// The backend a [`TeamTransport`] uses.
///
/// The wire spelling is `snake_case` (`"in_process"`, `"redis"`, `"pulsar"`), and
/// the default is [`TransportKind::InProcess`], so a config object without a
/// `kind` keeps today's behaviour.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum TransportKind {
    /// In-process delivery through the crate's message bus.
    #[default]
    InProcess,
    /// Shared delivery through Redis lists (RESP2, no vendor SDK).
    Redis,
    /// Pulsar delivery; declared but not implemented.
    Pulsar,
}

impl TransportKind {
    /// Returns the wire name of the kind.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::InProcess => "in_process",
            Self::Redis => "redis",
            Self::Pulsar => "pulsar",
        }
    }
}

impl fmt::Display for TransportKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Configuration of one team transport.
///
/// Unknown JSON fields are tolerated and every field is defaulted, so an empty
/// object yields the in-process transport with [`DEFAULT_KEY_PREFIX`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct TransportConfig {
    /// The backend to build.
    pub kind: TransportKind,
    /// Prefix of every key the Redis backend creates.
    pub key_prefix: String,
    /// Redis connection settings; required when
    /// [`kind`](Self::kind) is [`TransportKind::Redis`].
    pub redis: Option<RedisTransportConfig>,
}

impl Default for TransportConfig {
    fn default() -> Self {
        Self {
            kind: TransportKind::InProcess,
            key_prefix: DEFAULT_KEY_PREFIX.to_owned(),
            redis: None,
        }
    }
}

impl TransportConfig {
    /// Creates the configuration of the default in-process transport.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Selects the backend.
    #[must_use]
    pub const fn with_kind(mut self, kind: TransportKind) -> Self {
        self.kind = kind;
        self
    }

    /// Sets the prefix of every Redis key.
    #[must_use]
    pub fn with_key_prefix(mut self, key_prefix: impl Into<String>) -> Self {
        self.key_prefix = key_prefix.into();
        self
    }

    /// Sets the Redis connection settings.
    #[must_use]
    pub fn with_redis(mut self, redis: RedisTransportConfig) -> Self {
        self.redis = Some(redis);
        self
    }
}

/// A team's message transport.
///
/// The trait is the seam a later phase uses to route [`TeamRuntime`] through a
/// backend other than the in-process bus; it deliberately mirrors the
/// [`MessageBus`] surface (`send`, `publish`, subscriptions, `start`/`stop`) so
/// the team runtime keeps one dispatch path.
///
/// [`TeamRuntime`]: crate::TeamRuntime
/// [`MessageBus`]: crate::MessageBus
#[async_trait]
pub trait TeamTransport: Send + Sync + fmt::Debug {
    /// Returns the backend kind this transport implements.
    fn kind(&self) -> TransportKind;

    /// Returns the effective team id embedded in transport key or topic names.
    fn team_id(&self) -> &str;

    /// Starts the transport. Idempotent.
    ///
    /// # Errors
    ///
    /// Returns a transport error when the backend cannot be reached (a Redis
    /// handshake failure, for example).
    async fn start(&self) -> Result<()>;

    /// Stops the transport and joins every worker. Idempotent.
    ///
    /// # Errors
    ///
    /// Reserved for backend shutdown failures; both shipped backends return
    /// `Ok(())` after logging a worker that did not stop cleanly.
    async fn stop(&self) -> Result<()>;

    /// Delivers `envelope` point-to-point and waits for the reply.
    ///
    /// `timeout` overrides the backend default deadline; `None` keeps it.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] for an envelope without a recipient,
    /// [`Error::Timeout`] when the deadline elapses, and otherwise the
    /// recipient's error unchanged.
    async fn send(&self, envelope: MessageEnvelope, timeout: Option<Duration>) -> Result<Value>;

    /// Publishes `envelope` to every subscriber of its topic.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] for an envelope without a topic id, and a
    /// transport error when a delivery could not be handed to the backend.
    async fn publish(&self, envelope: MessageEnvelope) -> Result<()>;

    /// Registers `agent_id` as a subscriber of `topic` (an exact topic or a
    /// `*`/`?` pattern).
    fn add_subscription(&self, agent_id: &str, topic: &str);

    /// Removes the `(agent_id, topic)` subscription; unknown pairs are a no-op.
    fn remove_subscription(&self, agent_id: &str, topic: &str);

    /// Returns the sorted ids subscribed to `topic_id`.
    fn subscribers(&self, topic_id: &str) -> Vec<String>;

    /// Returns the whole subscription registry as `pattern -> sorted agent ids`.
    ///
    /// Python parity: `MessageBus.list_subscriptions()` without an agent filter.
    /// The default is empty, so a backend without a logical registry still
    /// supports introspection; the shipped backends override it.
    #[must_use]
    fn list_subscriptions(&self) -> BTreeMap<String, Vec<String>> {
        BTreeMap::new()
    }

    /// Returns the sorted patterns `agent_id` is subscribed to.
    ///
    /// Python parity: the `topics` list of
    /// `MessageBus.list_subscriptions(agent_id)`. The default is empty.
    #[must_use]
    fn agent_subscriptions(&self, _agent_id: &str) -> Vec<String> {
        Vec::new()
    }

    /// Returns the total number of `(agent, pattern)` pairs.
    ///
    /// Python parity: `MessageBus.get_subscription_count`. The default is zero.
    #[must_use]
    fn subscription_count(&self) -> usize {
        0
    }
}

/// Builds the transport `config` selects for the team `team_id`.
///
/// `bus_config` only applies to the in-process backend (it seeds the message
/// bus queue size and process timeout) and is stamped with `team_id`, so topic
/// names never fall back to `default` for a team with its own id.
///
/// # Errors
///
/// - [`ErrorKind::Unsupported`](openjiuwen_common::error::ErrorKind::Unsupported)
///   for [`TransportKind::Pulsar`], whose message names Pulsar and states that
///   it is not implemented.
/// - [`Error::Config`] when [`TransportKind::Redis`] is selected without
///   [`TransportConfig::redis`].
pub fn build_transport(
    config: &TransportConfig,
    team_id: &str,
    bus_config: MessageBusConfig,
    handler: Arc<dyn MessageHandler>,
) -> Result<Arc<dyn TeamTransport>> {
    match config.kind {
        TransportKind::InProcess => Ok(InProcessTransport::new(team_id, bus_config, handler)),
        TransportKind::Redis => {
            let redis = config.redis.clone().ok_or_else(|| {
                Error::config(
                    "the 'redis' team transport requires `redis` connection settings: \
                     set `transport.redis` (addr, optional username/password/db)",
                )
            })?;
            Ok(RedisTransport::new(
                redis,
                team_id,
                config.key_prefix.clone(),
                handler,
            ))
        }
        TransportKind::Pulsar => Err(Error::unsupported(
            "Pulsar team transport is not implemented: openjiuwen-multiagent ships the \
             'in_process' and 'redis' transports only",
        )),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn default_config_builds_the_in_process_transport() {
        let config = TransportConfig::default();
        assert_eq!(config.kind, TransportKind::InProcess);
        assert_eq!(config.key_prefix, DEFAULT_KEY_PREFIX);
        assert_eq!(config.redis, None);
        assert_eq!(TransportConfig::new(), config);
    }

    #[test]
    fn builders_set_fields() {
        let redis = RedisTransportConfig::new().with_addr("redis:6380");
        let config = TransportConfig::new()
            .with_kind(TransportKind::Redis)
            .with_key_prefix("team")
            .with_redis(redis.clone());
        assert_eq!(config.kind, TransportKind::Redis);
        assert_eq!(config.key_prefix, "team");
        assert_eq!(config.redis, Some(redis));
    }

    #[test]
    fn json_uses_snake_case_and_tolerates_unknown_fields() {
        let config: TransportConfig =
            serde_json::from_value(serde_json::json!({"kind": "redis", "future": 1})).unwrap();
        assert_eq!(config.kind, TransportKind::Redis);
        assert_eq!(config.key_prefix, DEFAULT_KEY_PREFIX);
        assert!(config.redis.is_none());

        let encoded = serde_json::to_value(TransportConfig::new()).unwrap();
        assert_eq!(encoded["kind"], "in_process");

        let pulsar: TransportKind = serde_json::from_value(serde_json::json!("pulsar")).unwrap();
        assert_eq!(pulsar, TransportKind::Pulsar);
        assert_eq!(serde_json::to_value(pulsar).unwrap(), "pulsar");
    }

    #[test]
    fn kind_names_are_stable() {
        assert_eq!(TransportKind::InProcess.as_str(), "in_process");
        assert_eq!(TransportKind::Redis.as_str(), "redis");
        assert_eq!(TransportKind::Pulsar.as_str(), "pulsar");
        assert_eq!(TransportKind::Redis.to_string(), "redis");
        assert_eq!(TransportKind::default(), TransportKind::InProcess);
    }
}
