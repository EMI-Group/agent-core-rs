//! Configuration and topology of the Redis team transport.
//!
//! [`RedisTopology`] names the deployment a transport targets; only a single
//! node is implemented, because the workspace ships no cluster- or
//! Sentinel-aware RESP client. [`RedisTransportConfig`] carries the connection,
//! polling, consumer-group and claim settings every worker reads.

use std::fmt;
use std::time::Duration;

use openjiuwen_common::error::Error;
use openjiuwen_common::security::SecretString;
use openjiuwen_foundation::store::external::redis::DEFAULT_ADDR;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Default idle delay between two polls of one stream, in milliseconds.
pub const DEFAULT_POLL_INTERVAL_MS: u64 = 10;

/// Default idle time after which a pending entry may be claimed, in milliseconds.
pub const DEFAULT_CLAIM_IDLE_MS: u64 = 30_000;

/// Default idle delay between two refreshes of the shared subscription registry,
/// in milliseconds.
pub const DEFAULT_REGISTRY_INTERVAL_MS: u64 = 250;

/// The Redis deployment topology a transport targets.
///
/// Only [`RedisTopology::Single`] is implemented. The other variants are
/// declared so a deployment cannot silently fall back to a single node: building
/// or starting such a transport returns
/// [`Error::Unsupported`](openjiuwen_common::error::ErrorKind::Unsupported)
/// naming the missing client.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RedisTopology {
    /// One Redis node addressed by [`RedisTransportConfig::addr`].
    #[default]
    Single,
    /// A Redis Cluster.
    Cluster,
    /// Redis nodes behind Sentinel.
    Sentinel,
}

impl RedisTopology {
    /// Returns the wire name of the topology.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Single => "single",
            Self::Cluster => "cluster",
            Self::Sentinel => "sentinel",
        }
    }

    /// Whether this topology is a single node, the only implemented one.
    #[must_use]
    pub const fn is_single(self) -> bool {
        matches!(self, Self::Single)
    }
}

impl fmt::Display for RedisTopology {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// The unsupported-topology error, naming the client this workspace lacks.
pub(super) fn topology_unsupported(topology: RedisTopology) -> Error {
    let client = match topology {
        RedisTopology::Single => "single-node RESP2 client",
        RedisTopology::Cluster => "Redis Cluster client",
        RedisTopology::Sentinel => "Redis Sentinel client",
    };
    Error::unsupported(format!(
        "the Redis '{}' team transport is not implemented: openjiuwen-multiagent ships no \
         {client} (only the single-node RESP2 client is available)",
        topology.as_str()
    ))
}

/// Connection settings of the Redis team transport.
///
/// Every field is defaulted and unknown JSON fields are tolerated, so an empty
/// object points at the local Redis on [`DEFAULT_ADDR`]. The password is held in
/// a [`SecretString`], which keeps the plaintext for serialization (a config
/// round-trip preserves it) but redacts it in `Debug` and `Display`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct RedisTransportConfig {
    /// `host:port` of the Redis server.
    pub addr: String,
    /// Username sent by `AUTH`, when the deployment requires one.
    pub username: Option<String>,
    /// Password sent by `AUTH`; redacted in `Debug` output.
    #[schemars(with = "Option<String>")]
    pub password: Option<SecretString>,
    /// Database selected after connecting.
    pub db: Option<u32>,
    /// Idle delay between two polls of one stream, in milliseconds.
    pub poll_interval_ms: u64,
    /// Consumer group shared by every transport of one deployment.
    ///
    /// `None` (or an empty name) derives `{prefix}:{team}:p2p:cg` and
    /// `{prefix}:{team}:pubsub:cg`, one group per inbox kind.
    pub consumer_group: Option<String>,
    /// Idle time after which a worker claims another consumer's pending
    /// entries, in milliseconds.
    pub claim_idle_ms: u64,
    /// Idle delay between two refreshes of the shared subscription registry,
    /// in milliseconds.
    pub registry_interval_ms: u64,
    /// Deployment topology; only [`RedisTopology::Single`] is implemented.
    pub topology: RedisTopology,
}

impl Default for RedisTransportConfig {
    fn default() -> Self {
        Self {
            addr: DEFAULT_ADDR.to_owned(),
            username: None,
            password: None,
            db: None,
            poll_interval_ms: DEFAULT_POLL_INTERVAL_MS,
            consumer_group: None,
            claim_idle_ms: DEFAULT_CLAIM_IDLE_MS,
            registry_interval_ms: DEFAULT_REGISTRY_INTERVAL_MS,
            topology: RedisTopology::Single,
        }
    }
}

impl RedisTransportConfig {
    /// Creates the configuration of a local, unauthenticated Redis.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the `host:port` of the server.
    #[must_use]
    pub fn with_addr(mut self, addr: impl Into<String>) -> Self {
        self.addr = addr.into();
        self
    }

    /// Sets the `AUTH` username.
    #[must_use]
    pub fn with_username(mut self, username: impl Into<String>) -> Self {
        self.username = Some(username.into());
        self
    }

    /// Sets the `AUTH` password.
    #[must_use]
    pub fn with_password(mut self, password: impl Into<String>) -> Self {
        self.password = Some(SecretString::new(password));
        self
    }

    /// Sets the database selected after connecting.
    #[must_use]
    pub const fn with_db(mut self, db: u32) -> Self {
        self.db = Some(db);
        self
    }

    /// Sets the idle delay between two polls of one stream.
    #[must_use]
    pub const fn with_poll_interval_ms(mut self, poll_interval_ms: u64) -> Self {
        self.poll_interval_ms = poll_interval_ms;
        self
    }

    /// Sets the consumer group shared by every transport of one deployment.
    #[must_use]
    pub fn with_consumer_group(mut self, consumer_group: impl Into<String>) -> Self {
        self.consumer_group = Some(consumer_group.into());
        self
    }

    /// Sets the idle time after which a pending entry may be claimed.
    #[must_use]
    pub const fn with_claim_idle_ms(mut self, claim_idle_ms: u64) -> Self {
        self.claim_idle_ms = claim_idle_ms;
        self
    }

    /// Sets the idle delay between two refreshes of the shared subscription
    /// registry.
    #[must_use]
    pub const fn with_registry_interval_ms(mut self, registry_interval_ms: u64) -> Self {
        self.registry_interval_ms = registry_interval_ms;
        self
    }

    /// Sets the deployment topology.
    #[must_use]
    pub const fn with_topology(mut self, topology: RedisTopology) -> Self {
        self.topology = topology;
        self
    }

    /// Returns the poll delay, never smaller than one millisecond.
    #[must_use]
    pub fn poll_interval(&self) -> Duration {
        Duration::from_millis(self.poll_interval_ms.max(1))
    }

    /// Returns the registry refresh delay, never smaller than one millisecond.
    #[must_use]
    pub fn registry_interval(&self) -> Duration {
        Duration::from_millis(self.registry_interval_ms.max(1))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn config_defaults() {
        let config = RedisTransportConfig::new();
        assert_eq!(config.addr, DEFAULT_ADDR);
        assert_eq!(config.poll_interval_ms, DEFAULT_POLL_INTERVAL_MS);
        assert!(config.username.is_none());
        assert!(config.password.is_none());
        assert_eq!(config.db, None);
        assert_eq!(config.consumer_group, None);
        assert_eq!(config.claim_idle_ms, DEFAULT_CLAIM_IDLE_MS);
        assert_eq!(config.registry_interval_ms, DEFAULT_REGISTRY_INTERVAL_MS);
        assert_eq!(config.topology, RedisTopology::Single);
        assert_eq!(config.poll_interval(), Duration::from_millis(10));
        assert_eq!(config.registry_interval(), Duration::from_millis(250));
    }

    #[test]
    fn builders_set_fields() {
        let config = RedisTransportConfig::new()
            .with_addr("redis:6380")
            .with_username("agent")
            .with_password("s3cret")
            .with_db(2)
            .with_poll_interval_ms(0)
            .with_consumer_group("crew")
            .with_claim_idle_ms(5)
            .with_registry_interval_ms(0)
            .with_topology(RedisTopology::Cluster);
        assert_eq!(config.addr, "redis:6380");
        assert_eq!(config.username.as_deref(), Some("agent"));
        assert_eq!(config.password.clone().unwrap().expose_secret(), "s3cret");
        assert_eq!(config.db, Some(2));
        // A zero interval would spin; the accessor clamps it to one millisecond.
        assert_eq!(config.poll_interval(), Duration::from_millis(1));
        assert_eq!(config.consumer_group.as_deref(), Some("crew"));
        assert_eq!(config.claim_idle_ms, 5);
        // The registry interval is clamped the same way the poll interval is.
        assert_eq!(config.registry_interval(), Duration::from_millis(1));
        assert_eq!(config.topology, RedisTopology::Cluster);
    }

    #[test]
    fn debug_redacts_the_password() {
        let config = RedisTransportConfig::new().with_password("s3cret");
        let rendered = format!("{config:?}");
        assert!(!rendered.contains("s3cret"), "{rendered}");
        assert!(rendered.contains("<REDACTED>"), "{rendered}");
    }

    #[test]
    fn json_is_snake_case_and_tolerates_unknown_fields() {
        let config: RedisTransportConfig = serde_json::from_value(serde_json::json!({
            "addr": "127.0.0.1:7000",
            "poll_interval_ms": 25,
            "consumer_group": "crew",
            "claim_idle_ms": 250,
            "topology": "sentinel",
            "future": true,
        }))
        .unwrap();
        assert_eq!(config.addr, "127.0.0.1:7000");
        assert_eq!(config.poll_interval_ms, 25);
        assert_eq!(config.consumer_group.as_deref(), Some("crew"));
        assert_eq!(config.claim_idle_ms, 250);
        assert_eq!(config.topology, RedisTopology::Sentinel);
        assert_eq!(config.db, None);

        let encoded = serde_json::to_value(RedisTransportConfig::new()).unwrap();
        assert_eq!(encoded["addr"], DEFAULT_ADDR);
        assert_eq!(encoded["poll_interval_ms"], 10);
        assert_eq!(encoded["claim_idle_ms"], DEFAULT_CLAIM_IDLE_MS);
        assert_eq!(encoded["topology"], "single");
        assert_eq!(
            serde_json::to_value(RedisTopology::Cluster).unwrap(),
            "cluster"
        );
    }

    #[test]
    fn topology_names_are_stable() {
        assert_eq!(RedisTopology::default(), RedisTopology::Single);
        assert_eq!(RedisTopology::Single.as_str(), "single");
        assert_eq!(RedisTopology::Cluster.to_string(), "cluster");
        assert_eq!(RedisTopology::Sentinel.as_str(), "sentinel");
        assert!(RedisTopology::Single.is_single());
        assert!(!RedisTopology::Cluster.is_single());
        assert!(!RedisTopology::Sentinel.is_single());
    }

    #[test]
    fn topology_guard_names_the_missing_client() {
        for (topology, needle) in [
            (RedisTopology::Cluster, "Redis Cluster client"),
            (RedisTopology::Sentinel, "Redis Sentinel client"),
        ] {
            let error = topology_unsupported(topology);
            assert_eq!(
                error.kind(),
                openjiuwen_common::error::ErrorKind::Unsupported
            );
            assert!(error.message().contains(needle), "{}", error.message());
        }
    }
}
