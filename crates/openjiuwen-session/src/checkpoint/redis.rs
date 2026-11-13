//! The Redis checkpoint backend.
//!
//! Ports the storage role of Python's Redis checkpointer
//! (`openjiuwen/core/session/checkpointer/`, v0.1.20, which delegates to the
//! `redis` client library) onto the crate's own minimal RESP2 client
//! ([`super::resp`]). Checkpoints are JSON documents — the same document shape
//! the file backend writes — so a checkpoint written by one process is readable
//! by any other process pointed at the same server and prefix.
//!
//! Key layout, all rooted at [`RedisOptions::prefix`]
//! (default [`DEFAULT_REDIS_KEY_PREFIX`]):
//!
//! | Key | Type | Contents |
//! | --- | --- | --- |
//! | `{prefix}:session:{session_id}:checkpoint:{checkpoint_id}` | string | one JSON document |
//! | `{prefix}:session:{session_id}:checkpoints` | set | that session's checkpoint ids |
//! | `{prefix}:sessions` | set | every session that has checkpoints |
//!
//! Two checkpointers with different prefixes never see each other's data, which
//! is what tests and side-by-side deployments rely on.
#![allow(clippy::module_name_repetitions)]

use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::utils::id::{now_rfc3339, uuid_hex};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;

use super::resp::{RespConnection, RespValue};
use super::{
    validate_checkpoint_id, validate_session_id, Checkpoint, CheckpointMeta, Checkpointer,
};

/// The default key prefix, Python's `openjiuwen:session` namespace.
pub const DEFAULT_REDIS_KEY_PREFIX: &str = "openjiuwen:session";

/// Configuration for [`RedisCheckpointer`].
///
/// `url` uses the `redis://[user[:password]@]host[:port][/db]` form; `prefix`
/// namespaces every key; `ttl` (when set) expires each checkpoint document;
/// `database` selects a Redis database and overrides the database in the URL.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct RedisOptions {
    /// The server URL; empty is a validation error.
    pub url: String,
    /// The key prefix every layout key is rooted at.
    pub prefix: String,
    /// The expiry applied to each checkpoint document, in seconds (rounded up to
    /// at least one second).
    pub ttl: Option<std::time::Duration>,
    /// The Redis database index, sent as `SELECT` after connecting.
    pub database: Option<u32>,
}

impl Default for RedisOptions {
    fn default() -> Self {
        Self {
            url: String::new(),
            prefix: DEFAULT_REDIS_KEY_PREFIX.to_string(),
            ttl: None,
            database: None,
        }
    }
}

/// A checkpointer that stores checkpoints in Redis.
///
/// The backend holds one serialized connection ([`RespConnection`]) behind a
/// [`tokio::sync::Mutex`]; a transport failure marks the connection broken and
/// the next operation dials a fresh one, so a server restart or an idle
/// connection does not wedge the backend. Application-level errors (a rejected
/// command) do not reconnect.
#[derive(Default)]
pub struct RedisCheckpointer {
    options: RedisOptions,
    connection: Mutex<Option<RespConnection>>,
}

impl std::fmt::Debug for RedisCheckpointer {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RedisCheckpointer")
            .field("url", &redact_url(&self.options.url))
            .field("prefix", &self.options.prefix)
            .field("ttl", &self.options.ttl)
            .field("database", &self.options.database)
            .finish_non_exhaustive()
    }
}

impl RedisCheckpointer {
    /// Connects to `url` with default options, validating the connection
    /// eagerly.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::ErrorKind::Validation`] for an empty
    /// or malformed URL, [`openjiuwen_common::error::ErrorKind::Unsupported`]
    /// for `rediss://`, and [`openjiuwen_common::error::ErrorKind::Session`]
    /// when the connection or its handshake fails.
    pub async fn connect(url: impl Into<String>) -> Result<Self> {
        Self::connect_with_options(RedisOptions {
            url: url.into(),
            ..RedisOptions::default()
        })
        .await
    }

    /// Connects with explicit `options`, validating the connection eagerly so a
    /// misconfigured URL fails at construction rather than on first use.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::ErrorKind::Validation`] for an empty
    /// URL or prefix, [`openjiuwen_common::error::ErrorKind::Unsupported`] for
    /// `rediss://`, and [`openjiuwen_common::error::ErrorKind::Session`] when
    /// the connection, the `SELECT`, or the handshake fails.
    pub async fn connect_with_options(options: RedisOptions) -> Result<Self> {
        if options.url.trim().is_empty() {
            return Err(Error::validation(
                "redis checkpointer requires a non-empty `url`",
            ));
        }
        if options.prefix.trim().is_empty() {
            return Err(Error::validation(
                "redis checkpointer requires a non-empty `prefix`",
            ));
        }
        let connection = dial(&options).await?;
        Ok(Self {
            options,
            connection: Mutex::new(Some(connection)),
        })
    }

    /// Returns the options this backend was built with.
    #[must_use]
    pub fn options(&self) -> &RedisOptions {
        &self.options
    }

    /// Returns the key of one checkpoint document.
    fn document_key(&self, session_id: &str, checkpoint_id: &str) -> String {
        format!(
            "{}:session:{session_id}:checkpoint:{checkpoint_id}",
            self.options.prefix
        )
    }

    /// Returns the key of a session's checkpoint-id index set.
    fn index_key(&self, session_id: &str) -> String {
        format!("{}:session:{session_id}:checkpoints", self.options.prefix)
    }

    /// Returns the key of the global session index set.
    fn sessions_key(&self) -> String {
        format!("{}:sessions", self.options.prefix)
    }

    /// Opens a fresh connection and applies the configured database.
    async fn dial(&self) -> Result<RespConnection> {
        dial(&self.options).await
    }

    /// Sends one text command, reconnecting once when the connection is broken.
    async fn run(&self, args: &[&str]) -> Result<RespValue> {
        let parts: Vec<&[u8]> = args.iter().map(|arg| arg.as_bytes()).collect();
        self.exchange(&parts).await
    }

    /// Sends one command with a binary payload appended to the text arguments.
    async fn run_with_payload(&self, args: &[&str], payload: &[u8]) -> Result<RespValue> {
        let mut parts: Vec<&[u8]> = args.iter().map(|arg| arg.as_bytes()).collect();
        parts.push(payload);
        self.exchange(&parts).await
    }

    /// Sends one text command and unwraps a server error reply.
    async fn checked(&self, args: &[&str]) -> Result<RespValue> {
        self.run(args).await?.into_result()
    }

    /// Runs one command over the shared connection, retrying once on a fresh
    /// connection after a transport failure.
    async fn exchange(&self, args: &[&[u8]]) -> Result<RespValue> {
        let mut guard = self.connection.lock().await;
        let mut connection = match guard.take() {
            Some(connection) if !connection.is_broken() => connection,
            _ => self.dial().await?,
        };
        let error = match connection.command_bytes(args).await {
            Ok(reply) => {
                *guard = Some(connection);
                return Ok(reply);
            }
            Err(error) => error,
        };
        if !connection.is_broken() {
            *guard = Some(connection);
            return Err(error);
        }
        // The transport failed: retry the (idempotent) command once on a fresh
        // connection so a dropped or restarted server does not fail the call.
        let mut fresh = match self.dial().await {
            Ok(connection) => connection,
            Err(dial_error) => {
                *guard = Some(connection);
                return Err(dial_error);
            }
        };
        let outcome = fresh.command_bytes(args).await;
        *guard = Some(fresh);
        outcome
    }

    /// Returns the members of an index set, in a deterministic order.
    async fn index_members(&self, index: &str) -> Result<Vec<String>> {
        let reply = self.checked(&["SMEMBERS", index]).await?;
        let items = match reply {
            RespValue::Array(Some(items)) => items,
            RespValue::Array(None) => Vec::new(),
            other => {
                return Err(Error::session(format!(
                    "unexpected SMEMBERS reply for {index}: {other:?}"
                )));
            }
        };
        let mut members = Vec::with_capacity(items.len());
        for item in items {
            let member = item.as_str().ok_or_else(|| {
                Error::session(format!(
                    "non-string member in the index set {index}: {item:?}"
                ))
            })?;
            members.push(member.to_string());
        }
        members.sort();
        Ok(members)
    }

    /// Fetches and decodes one checkpoint document.
    ///
    /// Returns `Ok(None)` for the nil reply so callers can tell "missing" from
    /// "corrupt".
    async fn get_document(&self, document: &str) -> Result<Option<Vec<u8>>> {
        let reply = self.checked(&["GET", document]).await?;
        match reply {
            RespValue::Bulk(Some(payload)) => Ok(Some(payload)),
            RespValue::Bulk(None) => Ok(None),
            other => Err(Error::session(format!(
                "unexpected GET reply for {document}: {other:?}"
            ))),
        }
    }

    /// Decodes a checkpoint document.
    fn decode(document: &str, payload: &[u8]) -> Result<Checkpoint> {
        serde_json::from_slice(payload).map_err(|error| {
            Error::session(format!("corrupt checkpoint document {document}: {error}"))
        })
    }
}

#[async_trait]
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
        let payload = serde_json::to_vec(&checkpoint).map_err(|error| {
            Error::session(format!(
                "failed to serialize checkpoint {checkpoint_id}: {error}"
            ))
        })?;
        let document = self.document_key(&checkpoint.session_id, &checkpoint_id);
        let index = self.index_key(&checkpoint.session_id);
        let sessions = self.sessions_key();
        self.run_with_payload(&["SET", &document], &payload)
            .await?
            .into_result()?;
        if let Some(ttl) = self.options.ttl {
            // `EXPIRE 0` would delete the document, so a sub-second TTL is
            // rounded up to one second.
            let seconds = ttl.as_secs().max(1).to_string();
            self.checked(&["EXPIRE", &document, &seconds]).await?;
        }
        self.checked(&["SADD", &index, &checkpoint_id]).await?;
        self.checked(&["SADD", &sessions, &checkpoint.session_id])
            .await?;
        Ok(checkpoint_id)
    }

    async fn load(&self, session_id: &str, checkpoint_id: &str) -> Result<Option<Checkpoint>> {
        validate_session_id(session_id)?;
        validate_checkpoint_id(checkpoint_id)?;
        let document = self.document_key(session_id, checkpoint_id);
        match self.get_document(&document).await? {
            Some(payload) => Ok(Some(Self::decode(&document, &payload)?)),
            None => Ok(None),
        }
    }

    async fn list(&self, session_id: &str) -> Result<Vec<CheckpointMeta>> {
        validate_session_id(session_id)?;
        let index = self.index_key(session_id);
        let mut metas = Vec::new();
        for checkpoint_id in self.index_members(&index).await? {
            let document = self.document_key(session_id, &checkpoint_id);
            match self.get_document(&document).await? {
                Some(payload) => {
                    let checkpoint = Self::decode(&document, &payload)?;
                    metas.push(CheckpointMeta::from(&checkpoint));
                }
                None => {
                    // Expired or externally deleted document: prune the stale
                    // index member so the session index stays accurate.
                    self.checked(&["SREM", &index, &checkpoint_id]).await?;
                }
            }
        }
        metas.sort_by(|left, right| {
            (&left.created_at, &left.checkpoint_id).cmp(&(&right.created_at, &right.checkpoint_id))
        });
        Ok(metas)
    }

    async fn delete(&self, session_id: &str, checkpoint_id: &str) -> Result<()> {
        validate_session_id(session_id)?;
        validate_checkpoint_id(checkpoint_id)?;
        let document = self.document_key(session_id, checkpoint_id);
        let index = self.index_key(session_id);
        self.checked(&["DEL", &document]).await?;
        self.checked(&["SREM", &index, checkpoint_id]).await?;
        Ok(())
    }

    async fn release(&self, session_id: &str) -> Result<()> {
        validate_session_id(session_id)?;
        let index = self.index_key(session_id);
        let sessions = self.sessions_key();
        let members = self.index_members(&index).await?;
        if !members.is_empty() {
            let documents: Vec<String> = members
                .iter()
                .map(|checkpoint_id| self.document_key(session_id, checkpoint_id))
                .collect();
            let mut args: Vec<&str> = Vec::with_capacity(documents.len() + 1);
            args.push("DEL");
            args.extend(documents.iter().map(String::as_str));
            self.checked(&args).await?;
        }
        self.checked(&["DEL", &index]).await?;
        self.checked(&["SREM", &sessions, session_id]).await?;
        Ok(())
    }

    async fn session_exists(&self, session_id: &str) -> Result<bool> {
        validate_session_id(session_id)?;
        let index = self.index_key(session_id);
        if self.checked(&["EXISTS", &index]).await?.as_i64() == Some(0) {
            return Ok(false);
        }
        // The index key exists, so the session has at least one member; members
        // whose documents were deleted out-of-band are pruned by `list`.
        Ok(!self.index_members(&index).await?.is_empty())
    }
}

/// Connects to `options.url` and selects `options.database` when set.
async fn dial(options: &RedisOptions) -> Result<RespConnection> {
    let mut connection = RespConnection::connect(&options.url).await?;
    if let Some(database) = options.database {
        connection.select_database(database).await?;
    }
    Ok(connection)
}

/// Masks the password in a `redis://` URL for logs and `Debug` output.
fn redact_url(url: &str) -> String {
    let Some((scheme, rest)) = url.split_once("://") else {
        return url.to_string();
    };
    let Some((credentials, host)) = rest.rsplit_once('@') else {
        return url.to_string();
    };
    match credentials.split_once(':') {
        Some((username, _)) => format!("{scheme}://{username}:***@{host}"),
        None => format!("{scheme}://***@{host}"),
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use openjiuwen_common::error::ErrorKind;

    use super::{redact_url, RedisCheckpointer, RedisOptions, DEFAULT_REDIS_KEY_PREFIX};

    #[test]
    fn options_default_to_the_python_namespace() {
        let options = RedisOptions::default();
        assert_eq!(options.url, "");
        assert_eq!(options.prefix, DEFAULT_REDIS_KEY_PREFIX);
        assert_eq!(options.ttl, None);
        assert_eq!(options.database, None);
    }

    #[test]
    fn options_round_trip_through_json() {
        let options = RedisOptions {
            url: "redis://cache:6379/1".to_string(),
            prefix: "tests".to_string(),
            ttl: Some(std::time::Duration::from_secs(30)),
            database: Some(1),
        };
        let encoded = serde_json::to_value(&options).unwrap();
        assert_eq!(encoded["url"], "redis://cache:6379/1");
        assert_eq!(encoded["prefix"], "tests");
        assert_eq!(encoded["database"], 1);
        assert_eq!(encoded["ttl"]["secs"], 30);
        let decoded: RedisOptions = serde_json::from_value(encoded).unwrap();
        assert_eq!(decoded, options);
        // Unknown and missing keys are tolerated.
        let tolerant: RedisOptions =
            serde_json::from_value(serde_json::json!({ "extra": 1 })).unwrap();
        assert_eq!(tolerant, RedisOptions::default());
    }

    #[tokio::test]
    async fn rejects_empty_configuration_without_dialing() {
        let error = RedisCheckpointer::connect("")
            .await
            .expect_err("an empty url must be rejected");
        assert_eq!(error.kind(), ErrorKind::Validation);

        let error = RedisCheckpointer::connect_with_options(RedisOptions {
            url: "redis://127.0.0.1:1".to_string(),
            prefix: String::new(),
            ..RedisOptions::default()
        })
        .await
        .expect_err("an empty prefix must be rejected");
        assert_eq!(error.kind(), ErrorKind::Validation);

        let error = RedisCheckpointer::connect("rediss://127.0.0.1:6379")
            .await
            .expect_err("TLS is not supported");
        assert_eq!(error.kind(), ErrorKind::Unsupported);
    }

    #[test]
    fn redacts_credentials_in_debug_output() {
        assert_eq!(
            redact_url("redis://user:secret@cache:6379/2"),
            "redis://user:***@cache:6379/2"
        );
        assert_eq!(
            redact_url("redis://secret@cache:6379"),
            "redis://***@cache:6379"
        );
        assert_eq!(redact_url("redis://cache:6379"), "redis://cache:6379");
        assert_eq!(redact_url("cache:6379"), "cache:6379");
        let checkpointer = RedisCheckpointer {
            options: RedisOptions {
                url: "redis://user:secret@cache:6379".to_string(),
                ..RedisOptions::default()
            },
            ..RedisCheckpointer::default()
        };
        let debug = format!("{checkpointer:?}");
        assert!(debug.contains("user:***@cache:6379"), "{debug}");
        assert!(!debug.contains("secret"), "{debug}");
    }
}
