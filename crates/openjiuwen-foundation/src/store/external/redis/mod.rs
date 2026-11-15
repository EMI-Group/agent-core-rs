//! Redis key-value store over a RESP2 connection.
//!
//! [`RedisKvStore`] implements the [`KvStore`](crate::store::kv::KvStore) contract
//! of [`crate::store::kv`] on top of a Redis server, speaking the wire protocol
//! through the in-crate codec ([`codec`]) and connection ([`client`]) — no vendor
//! SDK is involved, so the backend compiles in an offline build.
//!
//! # Configuration
//!
//! [`RedisConfig`] mirrors the convention of the other external backends: it is
//! `serde`-deserializable with a default for every field, so a store can be built
//! straight from a JSON object, and exposes builder setters for code.
//! A `redis://[user:password@]host:port/db` URL is parsed when given; an explicit
//! `addr`, `username`, `password` or `db` overrides the matching URL component.
//! `rediss://` selects TLS (see [`RedisConfig::with_tls`]) and keeps the same
//! parsing rules, including the `6379` default port; every other scheme is an
//! [`Error::Config`](openjiuwen_common::error::Error), never silently downgraded.
//!
//! # TLS (`rediss://`)
//!
//! A `rediss://` URL (or [`RedisConfig::with_tls`]) wraps the TCP socket in a
//! `tokio-rustls` client whose trust anchors are the bundled
//! [`webpki_roots::TLS_SERVER_ROOTS`] only. The TLS handshake is covered by the
//! configuration's `timeout`, exactly like the TCP connect.
//! There is no custom-CA bundle, no certificate pinning, and no
//! `insecure-skip-verify` switch: a deployment that terminates TLS with a private
//! CA cannot be reached from this backend yet.
//! Because a real TLS round trip needs either a trusted certificate chain or a
//! PEM/certificate-generation crate (neither of which the workspace ships), the
//! offline tests prove the path is taken (the first bytes on the wire are a TLS
//! handshake record, never cleartext RESP) and that a failed handshake surfaces
//! as a typed error; they do not complete a handshake.
//!
//! # Semantics
//!
//! The Python `BaseKVStore` contract is preserved: a missing key reads as `None`,
//! `mget` keeps input order, prefix scans are ordered by key, and both delete
//! variants return a live-entry count.
//! `key_prefix` is applied transparently to every key written or read (and
//! stripped from scan results), so two stores with different prefixes share one
//! server without colliding.
//! `set` issues a plain `SET`, which also clears any TTL the key had.
//! `exclusive_set` issues `SET ... NX`; see
//! [`RedisKvStore::exclusive_set`](crate::store::kv::KvStore::exclusive_set) for the
//! TTL details.
//!
//! # Value encoding
//!
//! `KvValue::Text` is stored as raw UTF-8 and read back as text. `KvValue::Bytes`
//! is stored as the ASCII marker `\0ojb:` followed by standard base64, so a text
//! value can never be mistaken for bytes. A stored payload that is not valid UTF-8
//! and lacks the marker decodes as [`KvValue::Bytes`](crate::store::kv::KvValue)
//! rather than being lossily replaced.

pub mod client;
pub mod codec;
pub mod store;

use std::time::Duration;

use serde::{Deserialize, Serialize};
use url::Url;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::security::SecretString;

pub use client::{RedisConnection, RedisServerError};
pub use codec::{Decoder, RespValue};
pub use store::RedisKvStore;

/// Address used when neither `url` nor `addr` names a server.
pub const DEFAULT_ADDR: &str = "127.0.0.1:6379";

/// Port used when a `redis://` or `rediss://` URL omits it.
const DEFAULT_PORT: u16 = 6379;

/// Timeout used when the configuration does not set one.
const fn default_timeout() -> Duration {
    Duration::from_secs(5)
}

/// Configuration of a [`RedisKvStore`].
///
/// ```
/// use openjiuwen_foundation::store::external::redis::RedisConfig;
///
/// let config = RedisConfig::new("127.0.0.1:6379")
///     .with_db(1)
///     .with_key_prefix("app:");
///
/// assert_eq!(config.addr.as_deref(), Some("127.0.0.1:6379"));
/// assert_eq!(config.db, Some(1));
/// assert_eq!(config.key_prefix, "app:");
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, rename_all = "snake_case")]
pub struct RedisConfig {
    /// A `redis://[user:password@]host:port/db` URL.
    pub url: Option<String>,
    /// Explicit `host:port`, overriding the URL host when set.
    pub addr: Option<String>,
    /// Explicit username for `AUTH`, overriding the URL user when set.
    pub username: Option<String>,
    /// Explicit password for `AUTH`, overriding the URL password when set.
    pub password: Option<SecretString>,
    /// Explicit database index for `SELECT`, overriding the URL path when set.
    pub db: Option<u32>,
    /// Whether to speak TLS to the server.
    ///
    /// A `rediss://` URL enables this automatically; set it explicitly to reach a
    /// TLS endpoint through [`RedisConfig::addr`]. It can never be turned off for a
    /// `rediss://` URL, so a TLS URL is not silently downgraded to cleartext.
    pub tls: bool,
    /// Timeout of a connect (including the TLS handshake), a write, and a reply read.
    pub timeout: Duration,
    /// Prefix prepended to every key written and stripped from every key read.
    pub key_prefix: String,
}

impl Default for RedisConfig {
    fn default() -> Self {
        Self {
            url: None,
            addr: None,
            username: None,
            password: None,
            db: None,
            tls: false,
            timeout: default_timeout(),
            key_prefix: String::new(),
        }
    }
}

impl RedisConfig {
    /// Creates a configuration for an explicit `host:port`.
    pub fn new(addr: impl Into<String>) -> Self {
        Self {
            addr: Some(addr.into()),
            ..Self::default()
        }
    }

    /// Creates a configuration from a `redis://` or `rediss://` URL.
    pub fn from_url(url: impl Into<String>) -> Self {
        Self {
            url: Some(url.into()),
            ..Self::default()
        }
    }

    /// Sets the username sent by `AUTH`.
    #[must_use]
    pub fn with_username(mut self, username: impl Into<String>) -> Self {
        self.username = Some(username.into());
        self
    }

    /// Sets the password sent by `AUTH`.
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

    /// Sets the connect/read/write timeout.
    #[must_use]
    pub const fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Enables TLS (the `rediss://` transport).
    ///
    /// Only `true` is meaningful: a `rediss://` URL already enables TLS and cannot
    /// be downgraded to cleartext, so this setter is how an explicit
    /// [`RedisConfig::addr`] reaches a TLS endpoint.
    #[must_use]
    pub const fn with_tls(mut self, tls: bool) -> Self {
        self.tls = tls;
        self
    }

    /// Sets the prefix applied to every key.
    #[must_use]
    pub fn with_key_prefix(mut self, key_prefix: impl Into<String>) -> Self {
        self.key_prefix = key_prefix.into();
        self
    }

    /// Builds a configuration from the environment.
    ///
    /// The server URL comes from `OPENJIUWEN_REDIS_URL`, falling back to
    /// `REDIS_URL`; `OPENJIUWEN_REDIS_USERNAME` and `OPENJIUWEN_REDIS_PASSWORD`
    /// override what the URL carries. Empty variables count as unset. The
    /// resulting configuration is validated here, so an invalid URL fails at
    /// configuration time rather than on connect.
    pub fn from_env() -> Result<Self> {
        let url = non_empty_env("OPENJIUWEN_REDIS_URL").or_else(|| non_empty_env("REDIS_URL"));
        let mut config = match url {
            Some(url) => Self::from_url(url),
            None => Self::default(),
        };
        if let Some(username) = non_empty_env("OPENJIUWEN_REDIS_USERNAME") {
            config.username = Some(username);
        }
        if let Some(password) = non_empty_env("OPENJIUWEN_REDIS_PASSWORD") {
            config.password = Some(SecretString::new(password));
        }
        config.resolved()?;
        Ok(config)
    }

    /// Connects a [`RedisKvStore`] with this configuration.
    pub async fn connect(&self) -> Result<RedisKvStore> {
        let resolved = self.resolved()?;
        let connection = if resolved.tls {
            RedisConnection::connect_tls_with_timeout(
                &resolved.addr,
                resolved.auth(),
                resolved.db,
                self.timeout,
            )
            .await?
        } else {
            RedisConnection::connect_with_timeout(
                &resolved.addr,
                resolved.auth(),
                resolved.db,
                self.timeout,
            )
            .await?
        };
        Ok(RedisKvStore::from_connection(connection, self.clone()))
    }

    /// Resolves this configuration into concrete connection parameters.
    ///
    /// URL components supply the defaults; an explicitly set `addr`, `username`,
    /// `password` or `db` wins over them.
    fn resolved(&self) -> Result<ResolvedRedis> {
        let mut resolved = match &self.url {
            Some(url) => parse_redis_url(url)?,
            None => ResolvedRedis {
                addr: DEFAULT_ADDR.to_string(),
                username: None,
                password: None,
                db: None,
                tls: false,
            },
        };
        if let Some(addr) = self.addr.as_deref().filter(|addr| !addr.is_empty()) {
            resolved.addr = addr.to_string();
        }
        if let Some(username) = self.username.as_deref().filter(|user| !user.is_empty()) {
            resolved.username = Some(username.to_string());
        }
        if let Some(password) = self.password.as_ref() {
            resolved.password = Some(password.expose_secret().to_string());
        }
        if let Some(db) = self.db {
            resolved.db = Some(db);
        }
        // A `rediss://` URL can only widen this flag, never clear it.
        resolved.tls |= self.tls;
        Ok(resolved)
    }
}

/// Concrete connection parameters after URL parsing and field overrides.
struct ResolvedRedis {
    /// `host:port` to connect to.
    addr: String,
    /// Username for `AUTH`, if any.
    username: Option<String>,
    /// Password for `AUTH`, if any.
    password: Option<String>,
    /// Database index for `SELECT`, if any.
    db: Option<u32>,
    /// Whether the connection speaks TLS (`rediss://`).
    tls: bool,
}

impl ResolvedRedis {
    /// The `AUTH` credentials, if a password is configured.
    ///
    /// A username without a password is ignored: Redis has no meaningful
    /// username-only authentication.
    fn auth(&self) -> client::RedisAuth {
        self.password
            .as_ref()
            .map(|password| (self.username.clone(), password.clone()))
    }
}

/// Reads an environment variable, treating an empty value as unset.
fn non_empty_env(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|value| !value.is_empty())
}

/// Parses a `redis://` or `rediss://` URL into connection parameters.
fn parse_redis_url(raw: &str) -> Result<ResolvedRedis> {
    let url = Url::parse(raw)
        .map_err(|error| Error::config(format!("invalid redis url {raw:?}: {error}")))?;
    let tls = match url.scheme() {
        "redis" => false,
        "rediss" => true,
        other => {
            return Err(Error::config(format!(
                "unsupported redis url scheme {other:?}; expected `redis://` or `rediss://`"
            )))
        }
    };
    let host = url
        .host_str()
        .filter(|host| !host.is_empty())
        .ok_or_else(|| Error::config(format!("redis url {raw:?} has no host")))?;
    let addr = format!("{host}:{}", url.port().unwrap_or(DEFAULT_PORT));
    let username = Some(url.username())
        .filter(|user| !user.is_empty())
        .map(str::to_string);
    let password = url.password().map(str::to_string);
    let db = match url.path().trim_matches('/') {
        "" => None,
        path => Some(path.parse::<u32>().map_err(|_| {
            Error::config(format!(
                "redis url {raw:?} has an invalid database index {path:?}"
            ))
        })?),
    };
    Ok(ResolvedRedis {
        addr,
        username,
        password,
        db,
        tls,
    })
}

#[cfg(test)]
mod tests;
