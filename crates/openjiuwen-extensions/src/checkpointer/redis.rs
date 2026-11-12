//! Redis checkpointer surface (deferred).
//!
//! Ports the configuration and provider shape of
//! `openjiuwen/extensions/checkpointer/redis/` (v0.1.20) without a Redis client:
//! Python's `RedisCheckpointerProvider` registers the `redis` checkpointer kind
//! and `RedisCheckpointer` stores JSON blobs under
//! `{session_id}:agent:{agent_id}:agent_state_blobs` (key layout documented
//! only).
//!
//! The backend itself is deliberately not implemented — pickled payloads are a
//! non-goal (`docs/PARITY.md`) — so [`RedisCheckpointer::connect`] and every
//! [`Checkpointer`] method return
//! [`openjiuwen_common::error::ErrorKind::Unsupported`] naming the supported
//! alternatives. Configuration types still validate exactly as Python does, so
//! callers fail on a bad URL before they hit the deferral.
//!
//! Every operation keeps Python's `async` signature while its body is
//! synchronous, and clippy reports part of the family as `unused_async` and
//! part as `unused_async_trait_impl`, so both lints are allowed here.
#![allow(clippy::unused_async, clippy::unused_async_trait_impl)]

use std::collections::BTreeMap;
use std::sync::Arc;

use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::{Checkpoint, CheckpointMeta, Checkpointer};

/// The message every deferred operation returns.
const DEFERRED: &str = "redis checkpointer backend is deferred; use the in-memory, file, \
                        or sqlite backend";

/// TTL configuration for data stored by the Redis checkpointer.
///
/// Python field name: `default_ttl` (minutes). Rust exposes it as
/// [`RedisTTLConfig::default_ttl_minutes`] while serializing the Python name.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RedisTTLConfig {
    /// Default time-to-live in minutes; `None` means entries never expire.
    #[serde(rename = "default_ttl")]
    pub default_ttl_minutes: Option<f64>,
    /// Whether reading an entry refreshes its TTL.
    pub refresh_on_read: bool,
}

impl RedisTTLConfig {
    /// Creates a TTL configuration; `default_ttl_minutes` is optional.
    #[must_use]
    pub fn new(default_ttl_minutes: Option<f64>, refresh_on_read: bool) -> Self {
        Self {
            default_ttl_minutes,
            refresh_on_read,
        }
    }
}

/// Redis connection configuration (Python `RedisConnectionConfig`).
///
/// Python additionally accepts a pre-configured `redis_client`; Rust has no
/// Redis client dependency, so a `url` is required. Accepted schemes are
/// `redis://`, `rediss://`, `redis+cluster://`, and `rediss+cluster://`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RedisConnectionConfig {
    /// Redis connection URL; required before [`RedisConnectionConfig::validate`]
    /// succeeds.
    pub url: Option<String>,
    /// Explicit cluster mode; when `None` it is inferred from the URL scheme.
    pub cluster_mode: Option<bool>,
    /// Additional connection arguments, carried through untouched.
    pub connection_args: BTreeMap<String, Value>,
}

impl RedisConnectionConfig {
    /// Creates a connection configuration for `url`; validation is deferred to
    /// [`RedisConnectionConfig::validate`].
    #[must_use]
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: Some(url.into()),
            ..Self::default()
        }
    }

    /// Validates the connection configuration.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::ErrorKind::Validation`] when `url` is
    /// missing or does not start with one of the four accepted schemes.
    pub fn validate(&self) -> Result<()> {
        let url = self.url.as_deref().ok_or_else(|| {
            Error::validation("a redis `url` is required in RedisConnectionConfig")
        })?;
        if !ACCEPTED_SCHEMES
            .iter()
            .any(|scheme| url.starts_with(scheme))
        {
            return Err(Error::validation(format!(
                "invalid Redis URL format: {url}. URL must start with redis://, rediss://, \
                 redis+cluster://, or rediss+cluster://"
            )));
        }
        Ok(())
    }

    /// Returns whether cluster mode should be used.
    ///
    /// An explicit `cluster_mode` wins; otherwise the `+cluster` URL scheme is
    /// inferred; otherwise standalone mode is assumed.
    #[must_use]
    pub fn is_cluster_mode(&self) -> bool {
        if let Some(cluster_mode) = self.cluster_mode {
            return cluster_mode;
        }
        let Some(url) = self.url.as_deref() else {
            return false;
        };
        url.starts_with(REDIS_CLUSTER_SCHEME) || url.starts_with(REDISS_CLUSTER_SCHEME)
    }

    /// Returns the URL to hand to a Redis client, with the `+cluster` marker
    /// removed (Python `get_connection_url`).
    #[must_use]
    pub fn connection_url(&self) -> Option<String> {
        let url = self.url.as_deref()?;
        if let Some(rest) = url.strip_prefix(REDIS_CLUSTER_SCHEME) {
            return Some(format!("{REDIS_SCHEME}{rest}"));
        }
        if let Some(rest) = url.strip_prefix(REDISS_CLUSTER_SCHEME) {
            return Some(format!("{REDISS_SCHEME}{rest}"));
        }
        Some(url.to_string())
    }
}

/// Accepted Redis URL schemes, in Python's validation order.
const ACCEPTED_SCHEMES: [&str; 4] = [
    REDIS_SCHEME,
    REDISS_SCHEME,
    REDIS_CLUSTER_SCHEME,
    REDISS_CLUSTER_SCHEME,
];
const REDIS_SCHEME: &str = "redis://";
const REDISS_SCHEME: &str = "rediss://";
const REDIS_CLUSTER_SCHEME: &str = "redis+cluster://";
const REDISS_CLUSTER_SCHEME: &str = "rediss+cluster://";

/// Complete configuration for [`RedisCheckpointer`] (Python
/// `RedisCheckpointerConfig`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RedisCheckpointerConfig {
    /// Redis connection configuration; required.
    pub connection: RedisConnectionConfig,
    /// Optional TTL configuration for stored data.
    #[serde(default)]
    pub ttl: Option<RedisTTLConfig>,
}

impl RedisCheckpointerConfig {
    /// Creates a configuration without TTL settings.
    #[must_use]
    pub fn new(connection: RedisConnectionConfig) -> Self {
        Self {
            connection,
            ttl: None,
        }
    }

    /// Sets (or clears) the TTL configuration.
    #[must_use]
    pub fn with_ttl(mut self, ttl: Option<RedisTTLConfig>) -> Self {
        self.ttl = ttl;
        self
    }
}

/// A Redis-backed checkpointer that is API-compatible but deferred.
///
/// Construction validates the connection configuration; every storage
/// operation then returns [`openjiuwen_common::error::ErrorKind::Unsupported`].
/// `session_exists` errors rather than returning `false`, so callers cannot
/// mistake deferral for an empty session.
#[derive(Debug, Clone)]
pub struct RedisCheckpointer {
    config: RedisCheckpointerConfig,
}

impl RedisCheckpointer {
    /// Creates a Redis checkpointer handle after validating `config`.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::ErrorKind::Validation`] when the
    /// connection URL is missing or uses an unsupported scheme.
    pub fn new(config: RedisCheckpointerConfig) -> Result<Self> {
        config.connection.validate()?;
        Ok(Self { config })
    }

    /// Returns the configuration this handle was built from.
    #[must_use]
    pub const fn config(&self) -> &RedisCheckpointerConfig {
        &self.config
    }

    /// Python's provider entry point (`CheckpointerFactory.register("redis")`).
    ///
    /// # Errors
    ///
    /// Always returns [`openjiuwen_common::error::ErrorKind::Unsupported`]: the
    /// backend is deferred.
    pub async fn connect(_config: RedisCheckpointerConfig) -> Result<Arc<dyn Checkpointer>> {
        Err(deferred())
    }

    /// Alias of [`RedisCheckpointer::connect`].
    ///
    /// # Errors
    ///
    /// Always returns [`openjiuwen_common::error::ErrorKind::Unsupported`]: the
    /// backend is deferred.
    pub async fn create(_config: &RedisCheckpointerConfig) -> Result<Arc<dyn Checkpointer>> {
        Err(deferred())
    }
}

/// The deferral error shared by every Redis operation.
fn deferred() -> Error {
    Error::unsupported(DEFERRED)
}

#[async_trait::async_trait]
impl Checkpointer for RedisCheckpointer {
    async fn save(&self, _checkpoint: Checkpoint) -> Result<String> {
        Err(deferred())
    }

    async fn load(&self, _session_id: &str, _checkpoint_id: &str) -> Result<Option<Checkpoint>> {
        Err(deferred())
    }

    async fn list(&self, _session_id: &str) -> Result<Vec<CheckpointMeta>> {
        Err(deferred())
    }

    async fn delete(&self, _session_id: &str, _checkpoint_id: &str) -> Result<()> {
        Err(deferred())
    }

    async fn release(&self, _session_id: &str) -> Result<()> {
        Err(deferred())
    }

    async fn session_exists(&self, _session_id: &str) -> Result<bool> {
        Err(deferred())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::checkpointer::CheckpointNamespace;
    use openjiuwen_common::error::ErrorKind;
    use serde_json::json;

    // port of tests/unit_tests/extensions/checkpointer/test_redis_checkpointer.py and
    // test_redis_checkpointer_provider.py (configuration/validation cases)

    fn valid_config() -> RedisCheckpointerConfig {
        RedisCheckpointerConfig::new(RedisConnectionConfig::new("redis://127.0.0.1:6379"))
    }

    #[test]
    fn accepts_every_python_scheme() {
        for url in [
            "redis://localhost:6379",
            "rediss://localhost:6379",
            "redis+cluster://localhost:7000",
            "rediss+cluster://localhost:7000",
        ] {
            let config = RedisConnectionConfig::new(url);
            assert!(config.validate().is_ok(), "{url} must be accepted");
        }
    }

    #[test]
    fn rejects_missing_or_foreign_urls() {
        let missing = RedisConnectionConfig::default();
        let error = missing.validate().unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("url"), "unexpected: {error}");

        let foreign = RedisConnectionConfig::new("http://localhost:6379");
        let error = foreign.validate().unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(
            error.to_string().contains("invalid Redis URL format"),
            "unexpected: {error}"
        );
    }

    #[test]
    fn cluster_mode_is_inferred_from_the_scheme_when_unset() {
        assert!(!RedisConnectionConfig::new("redis://localhost:6379").is_cluster_mode());
        assert!(!RedisConnectionConfig::new("rediss://localhost:6379").is_cluster_mode());
        assert!(RedisConnectionConfig::new("redis+cluster://localhost:7000").is_cluster_mode());
        assert!(RedisConnectionConfig::new("rediss+cluster://localhost:7000").is_cluster_mode());
    }

    #[test]
    fn explicit_cluster_mode_overrides_the_scheme() {
        let mut config = RedisConnectionConfig::new("redis://localhost:6379");
        config.cluster_mode = Some(true);
        assert!(config.is_cluster_mode());

        let mut config = RedisConnectionConfig::new("redis+cluster://localhost:7000");
        config.cluster_mode = Some(false);
        assert!(!config.is_cluster_mode());
    }

    #[test]
    fn connection_url_strips_the_cluster_marker() {
        assert_eq!(
            RedisConnectionConfig::new("redis+cluster://localhost:7000").connection_url(),
            Some("redis://localhost:7000".to_string())
        );
        assert_eq!(
            RedisConnectionConfig::new("rediss+cluster://localhost:7000").connection_url(),
            Some("rediss://localhost:7000".to_string())
        );
        assert_eq!(
            RedisConnectionConfig::new("redis://localhost:6379").connection_url(),
            Some("redis://localhost:6379".to_string())
        );
        assert_eq!(RedisConnectionConfig::default().connection_url(), None);
    }

    #[test]
    fn connection_args_are_carried_through() {
        let config: RedisConnectionConfig = serde_json::from_value(json!({
            "url": "redis://localhost:6379",
            "connection_args": {"db": 1, "timeout": 5}
        }))
        .unwrap();
        assert_eq!(config.connection_args["db"], json!(1));
        assert_eq!(config.connection_args["timeout"], json!(5));
        assert!(!config.is_cluster_mode());
    }

    #[test]
    fn checkpointer_config_round_trips_with_python_field_names() {
        let value = json!({
            "connection": {"url": "redis://localhost:6379", "cluster_mode": true},
            "ttl": {"default_ttl": 5.0, "refresh_on_read": true}
        });
        let config: RedisCheckpointerConfig = serde_json::from_value(value).unwrap();
        assert_eq!(
            config.connection.url.as_deref(),
            Some("redis://localhost:6379")
        );
        assert!(config.connection.is_cluster_mode());
        let ttl = config.ttl.as_ref().unwrap();
        assert_eq!(ttl.default_ttl_minutes, Some(5.0));
        assert!(ttl.refresh_on_read);

        let encoded = serde_json::to_value(&config).unwrap();
        assert_eq!(
            encoded["connection"]["url"],
            json!("redis://localhost:6379")
        );
        assert_eq!(encoded["ttl"]["default_ttl"], json!(5.0));
        assert_eq!(encoded["ttl"]["refresh_on_read"], json!(true));
    }

    #[test]
    fn new_validates_the_connection_url() {
        let error = RedisCheckpointer::new(RedisCheckpointerConfig::new(
            RedisConnectionConfig::new("noscheme"),
        ))
        .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);

        let checkpointer = RedisCheckpointer::new(valid_config()).unwrap();
        assert_eq!(
            checkpointer.config().connection.url.as_deref(),
            Some("redis://127.0.0.1:6379")
        );
    }

    #[tokio::test]
    async fn connect_and_create_are_unsupported() {
        let error = RedisCheckpointer::connect(valid_config())
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        assert!(
            error.to_string().contains("deferred"),
            "unexpected: {error}"
        );

        let error = RedisCheckpointer::create(&valid_config())
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Unsupported);
    }

    #[tokio::test]
    async fn every_checkpointer_method_is_unsupported() {
        let checkpointer = RedisCheckpointer::new(valid_config()).unwrap();
        let checkpoint = Checkpoint::new("session-1", CheckpointNamespace::Agent, "agent-1");

        let error = checkpointer.save(checkpoint).await.unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        assert!(
            error.to_string().contains("deferred"),
            "unexpected: {error}"
        );

        let error = checkpointer.load("session-1", "cp-1").await.unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Unsupported);

        let error = checkpointer.list("session-1").await.unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Unsupported);

        let error = checkpointer.latest("session-1").await.unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Unsupported);

        let error = checkpointer.delete("session-1", "cp-1").await.unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Unsupported);

        let error = checkpointer.release("session-1").await.unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Unsupported);

        // Deferral must never masquerade as an empty session.
        let error = checkpointer.session_exists("session-1").await.unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Unsupported);
    }
}
