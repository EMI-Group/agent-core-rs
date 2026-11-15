//! Redis checkpointer (single-node, live backend).
//!
//! Ports `openjiuwen/extensions/checkpointer/redis/` (v0.1.20): Python's
//! `RedisCheckpointerProvider` registers the `redis` checkpointer kind, and
//! `RedisCheckpointer` persists session checkpoints. Rust cannot depend on a
//! Redis driver (the workspace builds offline and vendors none), so the backend
//! speaks `RESP` through this crate's in-tree client
//! ([`crate::redis::Client`]).
//!
//! Key layout (one key per checkpoint plus one index per session):
//!
//! - `openjiuwen:checkpoint:{session_id}:{checkpoint_id}` — the payload,
//!   holding the [`crate::checkpointer::Checkpoint`] as JSON.
//! - `openjiuwen:checkpoint-index:{session_id}` — a sorted set of that
//!   session's checkpoint ids, each scored with its save time in epoch
//!   milliseconds.
//!
//! [`crate::checkpointer::Checkpointer::list`] reads the index, fetches the
//! payloads, drops index members whose payload disappeared, and returns the
//! summaries newest first by `(created_at, checkpoint_id)` descending;
//! `latest` therefore picks the newest checkpoint of a session.
//!
//! When `ttl.default_ttl_minutes` is configured, `save` sets an expiry on both
//! the payload and the index, and `ttl.refresh_on_read` re-arms both expiries
//! after a successful `load`.
//!
//! # Entry points
//!
//! The kernel's `CheckpointerKind` has no Redis variant, so
//! [`crate::checkpointer::build_checkpointer`] cannot select this backend:
//! construct it with [`RedisCheckpointer::new`] (lazy connect) or
//! [`RedisCheckpointer::connect`] / [`RedisCheckpointer::create`] (eager connect,
//! which also verifies the server answers `PING`).
//!
//! # Transport
//!
//! `rediss://` (TLS, verified against the web PKI) and `redis[s]+cluster://`
//! (a seed verified as a cluster node, with `MOVED`/`ASK` redirections followed)
//! are handled by [`crate::redis`], so both URL families work here unchanged.
use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::utils::id::{now_rfc3339, uuid_hex};
use openjiuwen_core::session::checkpoint::{validate_checkpoint_id, validate_session_id};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::OnceCell;

use super::{Checkpoint, CheckpointMeta, Checkpointer};
use crate::redis::{command, Client, ConnectionOptions, Reply};

/// Prefix of a checkpoint payload key (`openjiuwen:checkpoint:{session}:{id}`).
const PAYLOAD_PREFIX: &str = "openjiuwen:checkpoint:";
/// Prefix of a session's checkpoint index (`openjiuwen:checkpoint-index:{session}`).
const INDEX_PREFIX: &str = "openjiuwen:checkpoint-index:";
/// Page size used by the `SCAN` loop in [`Checkpointer::release`].
const SCAN_COUNT: u32 = 500;
/// Maximum number of keys per `MGET`/`DEL`/`ZREM` round trip.
const BATCH_KEYS: usize = 500;

/// Builds a checkpoint payload key.
fn payload_key(session_id: &str, checkpoint_id: &str) -> String {
    format!("{PAYLOAD_PREFIX}{session_id}:{checkpoint_id}")
}

/// Builds a session's checkpoint index key.
fn index_key(session_id: &str) -> String {
    format!("{INDEX_PREFIX}{session_id}")
}

/// Escapes `SCAN MATCH` glob metacharacters so a session id matches literally.
///
/// Ids cannot contain `/`, `\`, a NUL byte, or `..`, but they may contain `*`,
/// `?`, `[` and `]`, which would otherwise widen the release pattern.
fn escape_glob(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for character in value.chars() {
        if matches!(character, '*' | '?' | '[' | ']' | '\\') {
            escaped.push('\\');
        }
        escaped.push(character);
    }
    escaped
}

/// The current time in epoch milliseconds, used as the index sort score.
fn now_epoch_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_millis())
}

/// Converts a TTL in minutes into whole seconds, rounding up and never below 1.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn ttl_seconds(minutes: f64) -> u64 {
    if !minutes.is_finite() || minutes <= 0.0 {
        return 1;
    }
    // A positive, finite product saturates instead of wrapping in the cast.
    (minutes * 60.0).ceil().max(1.0) as u64
}

/// Decodes a stored checkpoint, naming the ids but never the payload.
fn decode_checkpoint(session_id: &str, checkpoint_id: &str, payload: &[u8]) -> Result<Checkpoint> {
    serde_json::from_slice(payload).map_err(|error| {
        Error::session(format!(
            "failed to decode checkpoint '{checkpoint_id}' of session '{session_id}': {error}"
        ))
    })
}

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
/// Python additionally accepts a pre-configured `redis_client`; Rust's client is
/// built from a URL, so a `url` is required. Accepted schemes are `redis://`,
/// `rediss://`, `redis+cluster://`, and `rediss+cluster://`.
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

/// A single-node Redis-backed [`Checkpointer`].
///
/// [`RedisCheckpointer::new`] validates the configuration without touching the
/// network; the connection is opened by the first storage operation and reused
/// afterwards. [`RedisCheckpointer::connect`] connects eagerly instead, so an
/// unreachable or misconfigured server fails at construction time.
pub struct RedisCheckpointer {
    config: RedisCheckpointerConfig,
    client: OnceCell<Client>,
}

/// A redacting `Debug`: neither `connection_args` (which may carry a password)
/// nor the URL (which may embed credentials) is printed.
impl fmt::Debug for RedisCheckpointer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RedisCheckpointer")
            .field("cluster_mode", &self.config.connection.is_cluster_mode())
            .field("ttl", &self.config.ttl)
            .field("connected", &self.client.initialized())
            .finish_non_exhaustive()
    }
}

impl RedisCheckpointer {
    /// Creates a checkpointer handle after validating `config`.
    ///
    /// No connection is opened; the first storage operation connects lazily.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::ErrorKind::Validation`] when the
    /// connection URL is missing or uses a scheme outside the four accepted
    /// ones.
    pub fn new(config: RedisCheckpointerConfig) -> Result<Self> {
        config.connection.validate()?;
        Ok(Self {
            config,
            client: OnceCell::new(),
        })
    }

    /// Returns the configuration this handle was built from.
    #[must_use]
    pub const fn config(&self) -> &RedisCheckpointerConfig {
        &self.config
    }

    /// Python's provider entry point (`CheckpointerFactory.register("redis")`).
    ///
    /// Connects eagerly and verifies the server answers `PING`, so a bad
    /// endpoint, credentials, or TLS/cluster request fails here rather than on
    /// the first checkpoint operation. The kernel `CheckpointerKind` has no
    /// Redis variant, so this (and [`RedisCheckpointer::create`]) is how a
    /// Redis checkpointer is obtained.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::ErrorKind::Validation`] for an
    /// invalid configuration (including cluster mode against a node that is not a
    /// cluster node), and `Transport`/`Timeout` when the server cannot be reached
    /// or a `rediss://` TLS handshake fails.
    pub async fn connect(config: RedisCheckpointerConfig) -> Result<Arc<dyn Checkpointer>> {
        let checkpointer = Self::new(config)?;
        checkpointer.client().await?.ping().await?;
        Ok(Arc::new(checkpointer))
    }

    /// Alias of [`RedisCheckpointer::connect`].
    ///
    /// # Errors
    ///
    /// Same as [`RedisCheckpointer::connect`].
    pub async fn create(config: &RedisCheckpointerConfig) -> Result<Arc<dyn Checkpointer>> {
        Self::connect(config.clone()).await
    }

    /// Returns the shared client, connecting on first use.
    async fn client(&self) -> Result<&Client> {
        self.client
            .get_or_try_init(|| async {
                let url = self.config.connection.connection_url().ok_or_else(|| {
                    Error::validation("a redis `url` is required in RedisConnectionConfig")
                })?;
                let options = ConnectionOptions::from_url(&url)?
                    .with_connection_args(&self.config.connection.connection_args)?
                    .with_cluster_mode(self.config.connection.is_cluster_mode());
                Client::connect(options).await
            })
            .await
    }

    /// The configured default TTL in whole seconds, when one is set.
    fn default_ttl_seconds(&self) -> Option<u64> {
        self.config
            .ttl
            .as_ref()?
            .default_ttl_minutes
            .map(ttl_seconds)
    }

    /// Whether a successful read must re-arm the expiries.
    fn refresh_on_read(&self) -> bool {
        self.default_ttl_seconds().is_some()
            && self
                .config
                .ttl
                .as_ref()
                .is_some_and(|ttl| ttl.refresh_on_read)
    }

    /// Re-arms the payload and index expiries after a read.
    ///
    /// A refresh failure never turns a successful read into an error; transport
    /// failures on the read itself are still propagated by the caller.
    async fn refresh_ttl(&self, session_id: &str, key: &str) {
        let Some(seconds) = self.default_ttl_seconds() else {
            return;
        };
        let seconds = seconds.to_string();
        let index_key = index_key(session_id);
        let batch = [
            command(&[b"EXPIRE".as_slice(), key.as_bytes(), seconds.as_bytes()]),
            command(&[
                b"EXPIRE".as_slice(),
                index_key.as_bytes(),
                seconds.as_bytes(),
            ]),
        ];
        match self.client().await {
            Ok(client) => {
                if let Err(error) = client.pipeline(&batch).await {
                    tracing::debug!(
                        %error,
                        session_id,
                        "redis checkpointer could not refresh the checkpoint ttl"
                    );
                }
            }
            Err(error) => tracing::debug!(
                %error,
                session_id,
                "redis checkpointer could not refresh the checkpoint ttl"
            ),
        }
    }

    /// Drops stale members from a session's checkpoint index.
    async fn remove_from_index(&self, index_key: &str, checkpoint_ids: &[String]) -> Result<()> {
        let client = self.client().await?;
        for chunk in checkpoint_ids.chunks(BATCH_KEYS) {
            let mut args: Vec<&[u8]> = Vec::with_capacity(chunk.len() + 2);
            args.push(b"ZREM");
            args.push(index_key.as_bytes());
            args.extend(chunk.iter().map(String::as_bytes));
            client.command(&args).await?;
        }
        Ok(())
    }
}

#[async_trait::async_trait]
impl Checkpointer for RedisCheckpointer {
    async fn save(&self, mut checkpoint: Checkpoint) -> Result<String> {
        validate_session_id(&checkpoint.session_id)?;
        if checkpoint.checkpoint_id.is_empty() {
            checkpoint.checkpoint_id = uuid_hex();
        } else {
            validate_checkpoint_id(&checkpoint.checkpoint_id)?;
        }
        if checkpoint.created_at.is_empty() {
            checkpoint.created_at = now_rfc3339();
        }
        let checkpoint_id = checkpoint.checkpoint_id.clone();
        let key = payload_key(&checkpoint.session_id, &checkpoint_id);
        let index_key = index_key(&checkpoint.session_id);
        let payload = serde_json::to_vec(&checkpoint).map_err(|error| {
            Error::session(format!(
                "failed to encode checkpoint '{checkpoint_id}' of session '{}': {error}",
                checkpoint.session_id
            ))
        })?;
        let score = now_epoch_millis().to_string();
        let mut batch = vec![
            command(&[b"SET".as_slice(), key.as_bytes(), payload.as_slice()]),
            command(&[
                b"ZADD".as_slice(),
                index_key.as_bytes(),
                score.as_bytes(),
                checkpoint_id.as_bytes(),
            ]),
        ];
        if let Some(seconds) = self.default_ttl_seconds() {
            let seconds = seconds.to_string();
            batch.push(command(&[
                b"EXPIRE".as_slice(),
                key.as_bytes(),
                seconds.as_bytes(),
            ]));
            batch.push(command(&[
                b"EXPIRE".as_slice(),
                index_key.as_bytes(),
                seconds.as_bytes(),
            ]));
        }
        self.client().await?.pipeline(&batch).await?;
        Ok(checkpoint_id)
    }

    async fn load(&self, session_id: &str, checkpoint_id: &str) -> Result<Option<Checkpoint>> {
        validate_session_id(session_id)?;
        validate_checkpoint_id(checkpoint_id)?;
        let key = payload_key(session_id, checkpoint_id);
        let reply = self.client().await?.command_str(&["GET", &key]).await?;
        let Some(payload) = reply.as_opt_bytes()? else {
            return Ok(None);
        };
        let checkpoint = decode_checkpoint(session_id, checkpoint_id, payload)?;
        if self.refresh_on_read() {
            self.refresh_ttl(session_id, &key).await;
        }
        Ok(Some(checkpoint))
    }

    async fn list(&self, session_id: &str) -> Result<Vec<CheckpointMeta>> {
        validate_session_id(session_id)?;
        let index_key = index_key(session_id);
        let client = self.client().await?;
        let reply = client
            .command_str(&["ZRANGE", &index_key, "0", "-1"])
            .await?;
        let ids: Vec<String> = reply
            .as_array()?
            .iter()
            .map(Reply::as_string_owned)
            .collect::<Result<_>>()?;

        let mut checkpoints: Vec<Checkpoint> = Vec::with_capacity(ids.len());
        let mut stale: Vec<String> = Vec::new();
        for chunk in ids.chunks(BATCH_KEYS) {
            let keys: Vec<String> = chunk.iter().map(|id| payload_key(session_id, id)).collect();
            let mut args: Vec<&[u8]> = Vec::with_capacity(keys.len() + 1);
            args.push(b"MGET");
            args.extend(keys.iter().map(String::as_bytes));
            let reply = client.command(&args).await?;
            let values = reply.as_array()?;
            if values.len() != chunk.len() {
                return Err(Error::transport(
                    "redis MGET returned an unexpected number of replies",
                ));
            }
            for (checkpoint_id, value) in chunk.iter().zip(values.iter()) {
                match value.as_opt_bytes()? {
                    Some(payload) => {
                        checkpoints.push(decode_checkpoint(session_id, checkpoint_id, payload)?);
                    }
                    None => stale.push(checkpoint_id.clone()),
                }
            }
        }
        if !stale.is_empty() {
            self.remove_from_index(&index_key, &stale).await?;
        }
        checkpoints.sort_by(|left, right| {
            (&right.created_at, &right.checkpoint_id).cmp(&(&left.created_at, &left.checkpoint_id))
        });
        Ok(checkpoints.iter().map(CheckpointMeta::from).collect())
    }

    async fn delete(&self, session_id: &str, checkpoint_id: &str) -> Result<()> {
        validate_session_id(session_id)?;
        validate_checkpoint_id(checkpoint_id)?;
        let key = payload_key(session_id, checkpoint_id);
        let index_key = index_key(session_id);
        let batch = [
            command(&[b"DEL".as_slice(), key.as_bytes()]),
            command(&[
                b"ZREM".as_slice(),
                index_key.as_bytes(),
                checkpoint_id.as_bytes(),
            ]),
        ];
        self.client().await?.pipeline(&batch).await?;
        Ok(())
    }

    async fn release(&self, session_id: &str) -> Result<()> {
        validate_session_id(session_id)?;
        let index_key = index_key(session_id);
        let pattern = format!("{PAYLOAD_PREFIX}{}:*", escape_glob(session_id));
        let count = SCAN_COUNT.to_string();
        let client = self.client().await?;
        let mut cursor = "0".to_owned();
        loop {
            let reply = client
                .command_str(&["SCAN", &cursor, "MATCH", &pattern, "COUNT", &count])
                .await?;
            let items = reply.as_array()?;
            let [next_cursor, keys] = items else {
                return Err(Error::transport("redis SCAN returned a malformed reply"));
            };
            cursor = next_cursor.as_string_owned()?;
            let payloads: Vec<Vec<u8>> = keys
                .as_array()?
                .iter()
                .map(|key| key.as_bytes().map(ToOwned::to_owned))
                .collect::<Result<_>>()?;
            for chunk in payloads.chunks(BATCH_KEYS) {
                let mut args: Vec<&[u8]> = Vec::with_capacity(chunk.len() + 1);
                args.push(b"DEL");
                args.extend(chunk.iter().map(Vec::as_slice));
                client.command(&args).await?;
            }
            if cursor == "0" {
                break;
            }
        }
        client.command_str(&["DEL", &index_key]).await?;
        Ok(())
    }

    async fn session_exists(&self, session_id: &str) -> Result<bool> {
        validate_session_id(session_id)?;
        let index_key = index_key(session_id);
        let reply = self
            .client()
            .await?
            .command_str(&["ZCARD", &index_key])
            .await?;
        Ok(reply.as_integer()? > 0)
    }
}

#[cfg(test)]
mod tests;
