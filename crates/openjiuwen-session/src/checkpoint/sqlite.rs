//! The `SQLite` checkpoint backend (feature `sqlite`).
//!
//! Ports the storage role of `openjiuwen/core/session/checkpointer/persistence.py`
//! (v0.1.20) to one `session_checkpoints` table instead of a key-value store:
//! each row holds the checkpoint's identifiers, its creation time, and the full
//! JSON payload. All I/O goes through `sqlx`'s runtime API, so the backend is
//! safe to call from async tasks.
//!
//! A save is a single `INSERT … ON CONFLICT … DO UPDATE` statement, so a
//! checkpoint is never partially written. [`SqliteCheckpointer::list`] returns a
//! session's checkpoints newest first (`created_at DESC, checkpoint_id DESC`);
//! the memory and file backends return them oldest first, so callers that care
//! about order should sort themselves ([`Checkpointer::latest`] does).
use std::path::Path;

use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::utils::id::{now_rfc3339, uuid_hex};
use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions};
use sqlx::{Row, SqlitePool};

use super::{
    validate_checkpoint_id, validate_session_id, Checkpoint, CheckpointMeta, Checkpointer,
};

/// Table and index DDL, applied idempotently by every constructor.
const SCHEMA: &str = "\
CREATE TABLE IF NOT EXISTS session_checkpoints (
    session_id TEXT NOT NULL,
    checkpoint_id TEXT NOT NULL,
    namespace TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY (session_id, checkpoint_id)
)";

/// Index backing the per-session listing and existence checks.
const SESSION_INDEX: &str = "\
CREATE INDEX IF NOT EXISTS idx_session_checkpoints_session
    ON session_checkpoints (session_id, created_at)";

/// Upsert of one checkpoint, atomic in a single statement.
const UPSERT_SQL: &str = "\
INSERT INTO session_checkpoints
    (session_id, checkpoint_id, namespace, entity_id, created_at, payload)
VALUES (?, ?, ?, ?, ?, ?)
ON CONFLICT(session_id, checkpoint_id) DO UPDATE SET
    namespace = excluded.namespace,
    entity_id = excluded.entity_id,
    created_at = excluded.created_at,
    payload = excluded.payload";

/// Loads one checkpoint's payload.
const LOAD_SQL: &str =
    "SELECT payload FROM session_checkpoints WHERE session_id = ? AND checkpoint_id = ?";

/// Lists a session's payloads newest first.
const LIST_SQL: &str = "\
SELECT payload FROM session_checkpoints
WHERE session_id = ?
ORDER BY created_at DESC, checkpoint_id DESC";

/// Deletes one checkpoint.
const DELETE_SQL: &str =
    "DELETE FROM session_checkpoints WHERE session_id = ? AND checkpoint_id = ?";

/// Deletes every checkpoint of a session.
const RELEASE_SQL: &str = "DELETE FROM session_checkpoints WHERE session_id = ?";

/// Reports whether a session has at least one checkpoint.
const EXISTS_SQL: &str =
    "SELECT EXISTS(SELECT 1 FROM session_checkpoints WHERE session_id = ?) AS present";

/// A checkpointer backed by a `SQLite` database.
///
/// Clones share the same connection pool. Use [`SqliteCheckpointer::open`] for a
/// database file and [`SqliteCheckpointer::in_memory`] for tests and short-lived
/// runs.
#[derive(Debug, Clone)]
pub struct SqliteCheckpointer {
    /// The connection pool every operation borrows from.
    pool: SqlitePool,
}

impl SqliteCheckpointer {
    /// Opens (creating if missing) the database at `path` and applies the schema.
    ///
    /// The parent directory is created when it is non-empty, so a path such as
    /// `data/sessions/checkpoints.db` works on a fresh checkout.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::ErrorKind::Session`] when the
    /// directory cannot be created, the database cannot be opened, or the schema
    /// cannot be applied.
    pub async fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            tokio::fs::create_dir_all(parent).await.map_err(|error| {
                Error::session(format!(
                    "sqlite checkpointer: failed to create directory {}: {error}",
                    parent.display()
                ))
            })?;
        }
        let options = SqliteConnectOptions::new()
            .filename(path)
            .create_if_missing(true);
        Self::connect(options, 4).await
    }

    /// Creates a fresh in-memory database and applies the schema.
    ///
    /// The pool is pinned to a single connection: an in-memory `SQLite` database
    /// is private to the connection that created it, so a second pooled
    /// connection would see an empty database.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::ErrorKind::Session`] when the
    /// database cannot be opened or the schema cannot be applied.
    pub async fn in_memory() -> Result<Self> {
        let options = SqliteConnectOptions::new().filename(":memory:");
        Self::connect(options, 1).await
    }

    /// Connects with `options` and applies the schema.
    async fn connect(options: SqliteConnectOptions, max_connections: u32) -> Result<Self> {
        let pool = SqlitePoolOptions::new()
            .max_connections(max_connections)
            .connect_with(options)
            .await
            .map_err(|error| sqlite_error(&error))?;
        sqlx::query(SCHEMA)
            .execute(&pool)
            .await
            .map_err(|error| sqlite_error(&error))?;
        sqlx::query(SESSION_INDEX)
            .execute(&pool)
            .await
            .map_err(|error| sqlite_error(&error))?;
        Ok(Self { pool })
    }
}

#[async_trait]
impl Checkpointer for SqliteCheckpointer {
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
        let payload = encode(&checkpoint)?;
        let checkpoint_id = checkpoint.checkpoint_id.clone();
        sqlx::query(UPSERT_SQL)
            .bind(&checkpoint.session_id)
            .bind(&checkpoint_id)
            .bind(checkpoint.namespace.as_str())
            .bind(&checkpoint.entity_id)
            .bind(&checkpoint.created_at)
            .bind(payload)
            .execute(&self.pool)
            .await
            .map_err(|error| sqlite_error(&error))?;
        Ok(checkpoint_id)
    }

    async fn load(&self, session_id: &str, checkpoint_id: &str) -> Result<Option<Checkpoint>> {
        validate_session_id(session_id)?;
        validate_checkpoint_id(checkpoint_id)?;
        let row = sqlx::query(LOAD_SQL)
            .bind(session_id)
            .bind(checkpoint_id)
            .fetch_optional(&self.pool)
            .await
            .map_err(|error| sqlite_error(&error))?;
        let Some(row) = row else {
            return Ok(None);
        };
        let payload: String = row
            .try_get("payload")
            .map_err(|error| sqlite_error(&error))?;
        Ok(Some(decode(&payload)?))
    }

    async fn list(&self, session_id: &str) -> Result<Vec<CheckpointMeta>> {
        validate_session_id(session_id)?;
        let rows = sqlx::query(LIST_SQL)
            .bind(session_id)
            .fetch_all(&self.pool)
            .await
            .map_err(|error| sqlite_error(&error))?;
        let mut metas = Vec::with_capacity(rows.len());
        for row in rows {
            let payload: String = row
                .try_get("payload")
                .map_err(|error| sqlite_error(&error))?;
            let checkpoint = decode(&payload)?;
            metas.push(CheckpointMeta::from(&checkpoint));
        }
        Ok(metas)
    }

    async fn delete(&self, session_id: &str, checkpoint_id: &str) -> Result<()> {
        validate_session_id(session_id)?;
        validate_checkpoint_id(checkpoint_id)?;
        sqlx::query(DELETE_SQL)
            .bind(session_id)
            .bind(checkpoint_id)
            .execute(&self.pool)
            .await
            .map_err(|error| sqlite_error(&error))?;
        Ok(())
    }

    async fn release(&self, session_id: &str) -> Result<()> {
        validate_session_id(session_id)?;
        sqlx::query(RELEASE_SQL)
            .bind(session_id)
            .execute(&self.pool)
            .await
            .map_err(|error| sqlite_error(&error))?;
        Ok(())
    }

    async fn session_exists(&self, session_id: &str) -> Result<bool> {
        validate_session_id(session_id)?;
        let row = sqlx::query(EXISTS_SQL)
            .bind(session_id)
            .fetch_one(&self.pool)
            .await
            .map_err(|error| sqlite_error(&error))?;
        let present: i64 = row
            .try_get("present")
            .map_err(|error| sqlite_error(&error))?;
        Ok(present != 0)
    }
}

/// Maps a `sqlx` failure to a [`openjiuwen_common::error::ErrorKind::Session`]
/// error.
fn sqlite_error(error: &sqlx::Error) -> Error {
    Error::session(format!("sqlite checkpointer: {error}"))
}

/// Serializes a checkpoint into the `payload` column.
fn encode(checkpoint: &Checkpoint) -> Result<String> {
    serde_json::to_string(checkpoint).map_err(|error| {
        Error::session(format!(
            "sqlite checkpointer: failed to serialize checkpoint {}: {error}",
            checkpoint.checkpoint_id
        ))
    })
}

/// Decodes a stored `payload` column back into a checkpoint.
fn decode(payload: &str) -> Result<Checkpoint> {
    serde_json::from_str(payload).map_err(|error| {
        Error::session(format!(
            "sqlite checkpointer: corrupt checkpoint payload: {error}"
        ))
    })
}
