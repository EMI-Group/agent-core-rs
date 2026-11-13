//! The SQLite [`TeamStore`] backend (feature `sqlite`).
//!
//! Python parity source: `openjiuwen/agent_teams/tools/database/` — the `SQLAlchemy` DAOs
//! (`team_dao`, `member_dao`, `task_dao`, `message_dao`) and the SQLite `TeamDatabase` they run on.
//! This port keeps the observable answers of [`InMemoryTeamStore`](crate::storage::InMemoryTeamStore)
//! — row shapes, ordering, compare-and-set transitions, stamping and the broadcast watermark — and
//! replaces the process-local maps with tables, so a team's state outlives the process.
//!
//! Two properties shape the implementation:
//!
//! * `open_store(&StorageSpec)` is synchronous but `sqlx` connects asynchronously, so the pool lives
//!   in a `tokio::sync::OnceCell` and every trait method starts with `self.pool().await?`.
//!   [`SqliteTeamStore::open`](crate::storage::SqliteTeamStore::open) and
//!   [`SqliteTeamStore::in_memory`](crate::storage::SqliteTeamStore::in_memory) connect eagerly
//!   instead, so a bad path or a broken database surfaces at construction time.
//! * `:memory:` is private to the connection that created it, so the in-memory construction pins its
//!   pool to a single connection; a file pool keeps four connections for concurrent readers.
//!
//! The whole DDL is idempotent and is applied on every connect (see the `schema` submodule), so a
//! database written by an older build of this crate keeps working. Every `sqlx` error is mapped
//! through one helper, and no lock or transaction is held across an `.await` that is not part of the
//! database call itself.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::Duration;

use async_trait::async_trait;
use serde_json::Value;
use sqlx::sqlite::{SqliteConnectOptions, SqliteJournalMode, SqlitePoolOptions, SqliteSynchronous};
use sqlx::SqlitePool;
use tokio::sync::OnceCell;

use openjiuwen_common::error::{Error, Result};

use crate::schema::{
    is_valid_member_transition, MemberStatus, ReviewVote, TaskFilter, TaskPatch, TaskRecord,
    TaskStatus,
};

use statements::{
    ADVANCE_WATERMARK, MARK_MESSAGE_READ, MEMBER_COLUMNS, MESSAGE_COLUMNS, TASK_COLUMNS,
    TEAM_COLUMNS, UNREAD_PREDICATE, UPDATE_MEMBER_ROW, UPDATE_TASK_ROW, UPSERT_MEMBER,
    UPSERT_MESSAGE, UPSERT_TASK, UPSERT_TEAM, UPSERT_VOTE, VOTE_COLUMNS,
};

use super::records::{MemberPatch, MemberRecord, MessageRecord, TeamInfoRecord};
use super::stamps::{refresh, require_key, stamp};
use super::TeamStore;

mod rows;
mod schema;
mod statements;

#[cfg(test)]
mod tests;

/// Storage kind of this backend; the `StorageSpec` value [`open_store`](super::open_store) dispatches
/// on.
pub const SQLITE_STORAGE_KIND: &str = "sqlite";

/// `StorageSpec::params` keys consulted by [`SqliteTeamStore::lazy_from_params`], in precedence order.
const PARAM_KEYS: [&str; 3] = ["path", "url", "database"];

/// Marker any of the accepted values may carry to select the in-memory database.
const MEMORY_MARKER: &str = ":memory:";

/// Filename the in-memory construction opens; `SQLite` treats `:memory:` as a private database.
const MEMORY_FILENAME: &str = ":memory:";

/// Maximum number of pooled connections of a file database.
const MAX_FILE_CONNECTIONS: u32 = 4;

/// How long a caller waits for a pooled connection or for a lock held by another writer.
const BUSY_TIMEOUT: Duration = Duration::from_secs(5);

/// Error message of a value that selects no database at all.
const EMPTY_TARGET: &str = "sqlite storage requires a non-empty 'path' or 'url'";

/// Error message of a `StorageSpec` that names no path, url or database.
const PARAM_REQUIRED: &str = "storage backend 'sqlite' requires a 'path' or 'url' parameter";

/// Where a store connects to.
#[derive(Debug)]
enum Target {
    /// A private single-connection `:memory:` database: tests and short-lived runs.
    Memory,
    /// A database file on disk.
    Path(PathBuf),
    /// A `sqlx` connection URL (`sqlite:`/`sqlite://…`), validated when the store is built.
    Url(String),
}

/// The SQLite-backed [`TeamStore`].
///
/// Construct it with [`SqliteTeamStore::open`] (a database file) or [`SqliteTeamStore::in_memory`]
/// (a private `:memory:` database), or let [`open_store`](super::open_store) build a lazily
/// connected one from a [`StorageSpec`](crate::schema::StorageSpec):
///
/// ```text
/// [storage]
/// kind = "sqlite"
/// [storage.params]
/// path = "data/team.db"
/// ```
///
/// Accepted `path`/`url`/`database` forms:
///
/// * `data/team.db` — a filesystem path, relative to the working directory.
/// * `sqlite://data/team.db` — the same relative path as a `sqlx` URL.
/// * `sqlite:///abs/team.db` — an absolute path as a `sqlx` URL.
/// * `:memory:` (or any value containing it) — a private in-memory database.
///
/// The store is `Send + Sync` and cheap to share; clone the surrounding `Arc<dyn TeamStore>` rather
/// than the struct. A lazily built store connects on its first trait call, so a bad path is reported
/// then, not at construction.
#[derive(Debug)]
pub struct SqliteTeamStore {
    /// The database the store connects to.
    target: Target,
    /// The connection pool, opened on first use (or eagerly by the constructors).
    pool: OnceCell<SqlitePool>,
}

impl SqliteTeamStore {
    /// Opens (creating if missing) the database file at `path`, applies the schema and returns a
    /// store that is ready to serve.
    ///
    /// The parent directory is created when it is missing, so `data/team.db` works on a fresh
    /// checkout. The file runs in WAL mode with a normal synchronous level and a five-second busy
    /// timeout, which keeps the leader and the member loops from failing on a short lock.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::Validation`](openjiuwen_common::error::ErrorKind::Validation) for an
    /// empty path, and [`ErrorKind::External`](openjiuwen_common::error::ErrorKind::External) when
    /// the directory cannot be created, the database cannot be opened, or the schema cannot be
    /// applied.
    pub async fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        if path.as_os_str().is_empty() {
            return Err(Error::validation(EMPTY_TARGET));
        }
        let store = Self {
            target: Target::Path(path.to_path_buf()),
            pool: OnceCell::new(),
        };
        store.pool().await?;
        Ok(store)
    }

    /// Creates a fresh in-memory database with the schema applied.
    ///
    /// The pool is pinned to a single connection: a `:memory:` database is private to the connection
    /// that created it, so a second pooled connection would see an empty database.
    ///
    /// # Errors
    ///
    /// Returns whatever the driver reports when the database cannot be opened or the schema cannot
    /// be applied.
    pub async fn in_memory() -> Result<Self> {
        let store = Self {
            target: Target::Memory,
            pool: OnceCell::new(),
        };
        store.pool().await?;
        Ok(store)
    }

    /// Builds a lazily connected store from `StorageSpec::params`.
    ///
    /// The `path` key wins over `url`, which wins over `database`; each must be a string, and a
    /// non-string value is [`ErrorKind::Validation`](openjiuwen_common::error::ErrorKind::Validation).
    /// A value containing `:memory:` selects the private in-memory database, a value starting with
    /// `sqlite:` is parsed as a `sqlx` URL (so bad syntax is reported here), and anything else is a
    /// filesystem path.
    ///
    /// The connection is opened on the first trait call, because [`open_store`](super::open_store)
    /// is synchronous; call [`SqliteTeamStore::open`] directly when construction errors should
    /// surface immediately.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::Config`](openjiuwen_common::error::ErrorKind::Config) when neither
    /// `path` nor `url` nor `database` is set, and
    /// [`ErrorKind::Validation`](openjiuwen_common::error::ErrorKind::Validation) for a non-string
    /// parameter, an empty target, or a malformed `sqlite:` URL.
    pub(crate) fn lazy_from_params(params: &BTreeMap<String, Value>) -> Result<Self> {
        let mut selected: Option<&str> = None;
        for key in PARAM_KEYS {
            let Some(value) = params.get(key) else {
                continue;
            };
            let Some(text) = value.as_str() else {
                return Err(Error::validation(format!(
                    "storage param '{key}' must be a string"
                )));
            };
            selected = Some(text);
            break;
        }
        let Some(raw) = selected else {
            return Err(Error::config(PARAM_REQUIRED));
        };
        Ok(Self {
            target: Self::parse_target(raw)?,
            pool: OnceCell::new(),
        })
    }

    /// Turns one accepted parameter value into the target it names.
    fn parse_target(raw: &str) -> Result<Target> {
        if raw.is_empty() {
            return Err(Error::validation(EMPTY_TARGET));
        }
        if raw.contains(MEMORY_MARKER) {
            return Ok(Target::Memory);
        }
        if raw.starts_with("sqlite:") {
            SqliteConnectOptions::from_str(raw).map_err(|error| {
                Error::validation(format!("invalid sqlite url '{raw}': {error}"))
            })?;
            return Ok(Target::Url(raw.to_owned()));
        }
        Ok(Target::Path(PathBuf::from(raw)))
    }

    /// Returns the connection pool, connecting on first use.
    async fn pool(&self) -> Result<&SqlitePool> {
        self.pool
            .get_or_try_init(|| async { self.connect().await })
            .await
    }

    /// Opens the pool for the configured target and applies the schema.
    async fn connect(&self) -> Result<SqlitePool> {
        match &self.target {
            Target::Memory => {
                let options = SqliteConnectOptions::new()
                    .filename(MEMORY_FILENAME)
                    .busy_timeout(BUSY_TIMEOUT)
                    .foreign_keys(true);
                let pool = SqlitePoolOptions::new()
                    .max_connections(1)
                    .min_connections(1)
                    .acquire_timeout(BUSY_TIMEOUT)
                    .idle_timeout(None)
                    .max_lifetime(None)
                    .connect_with(options)
                    .await
                    .map_err(rows::sqlite_error)?;
                Self::finish(pool).await
            }
            Target::Path(path) => {
                Self::create_parent(path).await?;
                let options = file_options(SqliteConnectOptions::new().filename(path));
                Self::file_pool(options).await
            }
            Target::Url(url) => {
                let options = SqliteConnectOptions::from_str(url).map_err(|error| {
                    Error::validation(format!("invalid sqlite url '{url}': {error}"))
                })?;
                Self::create_parent(options.get_filename()).await?;
                Self::file_pool(file_options(options)).await
            }
        }
    }

    /// Opens a pool over a file database and applies the schema.
    async fn file_pool(options: SqliteConnectOptions) -> Result<SqlitePool> {
        let pool = SqlitePoolOptions::new()
            .max_connections(MAX_FILE_CONNECTIONS)
            .acquire_timeout(BUSY_TIMEOUT)
            .connect_with(options)
            .await
            .map_err(rows::sqlite_error)?;
        Self::finish(pool).await
    }

    /// Applies the schema to a freshly connected pool and reports the ready database once.
    async fn finish(pool: SqlitePool) -> Result<SqlitePool> {
        schema::ensure_schema(&pool).await?;
        tracing::debug!(
            schema_version = schema::SCHEMA_VERSION,
            "sqlite team store connected"
        );
        Ok(pool)
    }

    /// Creates the parent directory of `path` when it is missing.
    async fn create_parent(path: &Path) -> Result<()> {
        let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        else {
            return Ok(());
        };
        tokio::fs::create_dir_all(parent).await.map_err(|error| {
            Error::external(error).with_context(format!(
                "sqlite team store: create directory {}",
                parent.display()
            ))
        })
    }
}

/// Applies the file-database connection settings shared by both path forms.
fn file_options(options: SqliteConnectOptions) -> SqliteConnectOptions {
    options
        .create_if_missing(true)
        .journal_mode(SqliteJournalMode::Wal)
        .synchronous(SqliteSynchronous::Normal)
        .busy_timeout(BUSY_TIMEOUT)
        .foreign_keys(true)
}

#[async_trait]
impl TeamStore for SqliteTeamStore {
    fn name(&self) -> &'static str {
        SQLITE_STORAGE_KIND
    }

    async fn upsert_team(&self, record: TeamInfoRecord) -> Result<()> {
        require_key(&record.team_name, "team_name")?;
        let pool = self.pool().await?;
        let mut tx = pool
            .begin_with("BEGIN IMMEDIATE")
            .await
            .map_err(rows::sqlite_error)?;
        let previous: Option<(i64, i64)> =
            sqlx::query_as("SELECT created, updated_at FROM team_info WHERE team_name = ?")
                .bind(&record.team_name)
                .fetch_optional(&mut *tx)
                .await
                .map_err(rows::sqlite_error)?;
        let mut record = record;
        if let Some((created, updated_at)) = previous {
            if record.created == 0 {
                record.created = u64::try_from(created).unwrap_or_default();
            }
            record.updated_at = stamp(
                record.updated_at,
                u64::try_from(updated_at).unwrap_or_default(),
            );
        } else {
            record.created = stamp(record.created, 0);
            record.updated_at = stamp(record.updated_at, record.created);
        }
        sqlx::query(UPSERT_TEAM)
            .bind(&record.team_name)
            .bind(&record.display_name)
            .bind(&record.leader_member_name)
            .bind(&record.desc)
            .bind(&record.prompt)
            .bind(rows::encode_enum(record.dispatch_mode)?)
            .bind(record.enable_task_verification)
            .bind(rows::to_i64(record.created))
            .bind(rows::to_i64(record.updated_at))
            .execute(&mut *tx)
            .await
            .map_err(rows::sqlite_error)?;
        tx.commit().await.map_err(rows::sqlite_error)
    }

    async fn team(&self, team_name: &str) -> Result<Option<TeamInfoRecord>> {
        let pool = self.pool().await?;
        let statement = format!("SELECT {TEAM_COLUMNS} FROM team_info WHERE team_name = ?");
        let row = sqlx::query(&statement)
            .bind(team_name)
            .fetch_optional(pool)
            .await
            .map_err(rows::sqlite_error)?;
        row.as_ref().map(rows::decode_team).transpose()
    }

    async fn insert_member(&self, record: MemberRecord) -> Result<()> {
        require_key(&record.team_name, "team_name")?;
        require_key(&record.member_name, "member_name")?;
        let pool = self.pool().await?;
        let mut tx = pool
            .begin_with("BEGIN IMMEDIATE")
            .await
            .map_err(rows::sqlite_error)?;
        let previous: Option<i64> = sqlx::query_scalar(
            "SELECT updated_at FROM team_member WHERE team_name = ? AND member_name = ?",
        )
        .bind(&record.team_name)
        .bind(&record.member_name)
        .fetch_optional(&mut *tx)
        .await
        .map_err(rows::sqlite_error)?;
        let mut record = record;
        record.updated_at = stamp(
            record.updated_at,
            previous.map_or(0, |value| u64::try_from(value).unwrap_or_default()),
        );
        sqlx::query(UPSERT_MEMBER)
            .bind(&record.team_name)
            .bind(&record.member_name)
            .bind(&record.display_name)
            .bind(&record.desc)
            .bind(rows::encode_enum(record.role)?)
            .bind(rows::encode_enum(record.status)?)
            .bind(rows::encode_enum(record.execution)?)
            .bind(rows::encode_enum(record.mode)?)
            .bind(&record.prompt)
            .bind(record.model_ref.as_deref())
            .bind(record.worktree.as_deref())
            .bind(rows::to_i64(record.updated_at))
            .execute(&mut *tx)
            .await
            .map_err(rows::sqlite_error)?;
        tx.commit().await.map_err(rows::sqlite_error)
    }

    async fn member(&self, team_name: &str, member_name: &str) -> Result<Option<MemberRecord>> {
        let pool = self.pool().await?;
        let statement = format!(
            "SELECT {MEMBER_COLUMNS} FROM team_member WHERE team_name = ? AND member_name = ?"
        );
        let row = sqlx::query(&statement)
            .bind(team_name)
            .bind(member_name)
            .fetch_optional(pool)
            .await
            .map_err(rows::sqlite_error)?;
        row.as_ref().map(rows::decode_member).transpose()
    }

    async fn list_members(&self, team_name: &str) -> Result<Vec<MemberRecord>> {
        let pool = self.pool().await?;
        let statement = format!(
            "SELECT {MEMBER_COLUMNS} FROM team_member WHERE team_name = ? ORDER BY member_name"
        );
        let rows = sqlx::query(&statement)
            .bind(team_name)
            .fetch_all(pool)
            .await
            .map_err(rows::sqlite_error)?;
        rows.iter().map(rows::decode_member).collect()
    }

    async fn update_member(
        &self,
        team_name: &str,
        member_name: &str,
        patch: &MemberPatch,
    ) -> Result<bool> {
        let pool = self.pool().await?;
        let mut tx = pool
            .begin_with("BEGIN IMMEDIATE")
            .await
            .map_err(rows::sqlite_error)?;
        let statement = format!(
            "SELECT {MEMBER_COLUMNS} FROM team_member WHERE team_name = ? AND member_name = ?"
        );
        let row = sqlx::query(&statement)
            .bind(team_name)
            .bind(member_name)
            .fetch_optional(&mut *tx)
            .await
            .map_err(rows::sqlite_error)?;
        let Some(row) = row else {
            tx.rollback().await.map_err(rows::sqlite_error)?;
            return Ok(false);
        };
        let mut record = rows::decode_member(&row)?;
        if !patch.apply(&mut record) {
            tx.rollback().await.map_err(rows::sqlite_error)?;
            return Ok(true);
        }
        record.updated_at = refresh(record.updated_at);
        sqlx::query(UPDATE_MEMBER_ROW)
            .bind(&record.display_name)
            .bind(&record.desc)
            .bind(rows::encode_enum(record.role)?)
            .bind(rows::encode_enum(record.status)?)
            .bind(rows::encode_enum(record.execution)?)
            .bind(rows::encode_enum(record.mode)?)
            .bind(&record.prompt)
            .bind(record.model_ref.as_deref())
            .bind(record.worktree.as_deref())
            .bind(rows::to_i64(record.updated_at))
            .bind(team_name)
            .bind(member_name)
            .execute(&mut *tx)
            .await
            .map_err(rows::sqlite_error)?;
        tx.commit().await.map_err(rows::sqlite_error)?;
        Ok(true)
    }

    async fn transition_member(
        &self,
        team_name: &str,
        member_name: &str,
        from: MemberStatus,
        to: MemberStatus,
    ) -> Result<bool> {
        if !is_valid_member_transition(from, to) {
            return Ok(false);
        }
        let pool = self.pool().await?;
        let result = sqlx::query(
            "UPDATE team_member SET status = ?, updated_at = MAX(?, updated_at + 1) \
             WHERE team_name = ? AND member_name = ? AND status = ?",
        )
        .bind(rows::encode_enum(to)?)
        .bind(rows::to_i64(refresh(0)))
        .bind(team_name)
        .bind(member_name)
        .bind(rows::encode_enum(from)?)
        .execute(pool)
        .await
        .map_err(rows::sqlite_error)?;
        Ok(result.rows_affected() == 1)
    }

    async fn insert_task(&self, record: TaskRecord) -> Result<()> {
        require_key(&record.team_name, "team_name")?;
        require_key(&record.task_id, "task_id")?;
        let pool = self.pool().await?;
        let mut tx = pool
            .begin_with("BEGIN IMMEDIATE")
            .await
            .map_err(rows::sqlite_error)?;
        let previous: Option<i64> = sqlx::query_scalar(
            "SELECT updated_at FROM team_task WHERE team_name = ? AND task_id = ?",
        )
        .bind(&record.team_name)
        .bind(&record.task_id)
        .fetch_optional(&mut *tx)
        .await
        .map_err(rows::sqlite_error)?;
        let mut record = record;
        record.updated_at = stamp(
            record.updated_at,
            previous.map_or(0, |value| u64::try_from(value).unwrap_or_default()),
        );
        let reviewers = rows::encode_list(&record.reviewers)?;
        let change_notes = rows::encode_list(&record.change_notes)?;
        sqlx::query(UPSERT_TASK)
            .bind(&record.team_name)
            .bind(&record.task_id)
            .bind(&record.title)
            .bind(&record.content)
            .bind(rows::encode_enum(record.status)?)
            .bind(record.assignee.as_deref())
            .bind(&reviewers)
            .bind(i64::from(record.review_round))
            .bind(i64::from(record.max_review_rounds))
            .bind(record.summary.as_deref())
            .bind(record.note.as_deref())
            .bind(record.parent_task_id.as_deref())
            .bind(record.parent_revision.map(i64::from))
            .bind(record.created_by.as_deref())
            .bind(i64::from(record.revision))
            .bind(i64::from(record.observed_revision))
            .bind(&change_notes)
            .bind(rows::to_i64(record.updated_at))
            .execute(&mut *tx)
            .await
            .map_err(rows::sqlite_error)?;
        tx.commit().await.map_err(rows::sqlite_error)
    }

    async fn task(&self, team_name: &str, task_id: &str) -> Result<Option<TaskRecord>> {
        let pool = self.pool().await?;
        let statement =
            format!("SELECT {TASK_COLUMNS} FROM team_task WHERE team_name = ? AND task_id = ?");
        let row = sqlx::query(&statement)
            .bind(team_name)
            .bind(task_id)
            .fetch_optional(pool)
            .await
            .map_err(rows::sqlite_error)?;
        row.as_ref().map(rows::decode_task).transpose()
    }

    async fn list_tasks(&self, team_name: &str, filter: &TaskFilter) -> Result<Vec<TaskRecord>> {
        let pool = self.pool().await?;
        let mut statement = format!("SELECT {TASK_COLUMNS} FROM team_task WHERE team_name = ?");
        if filter.status.is_some() {
            statement.push_str(" AND status = ?");
        }
        if filter.assignee.is_some() {
            statement.push_str(" AND assignee = ?");
        }
        if filter.parent.is_some() {
            statement.push_str(" AND parent_task_id = ?");
        }
        statement.push_str(" ORDER BY updated_at DESC, task_id ASC");
        if filter.limit.is_some() {
            statement.push_str(" LIMIT ?");
        }
        let mut query = sqlx::query(&statement).bind(team_name);
        if let Some(status) = filter.status {
            query = query.bind(rows::encode_enum(status)?);
        }
        if let Some(assignee) = filter.assignee.as_deref() {
            query = query.bind(assignee);
        }
        if let Some(parent) = filter.parent.as_deref() {
            query = query.bind(parent);
        }
        if let Some(limit) = filter.limit {
            query = query.bind(i64::try_from(limit).unwrap_or(i64::MAX));
        }
        let rows = query.fetch_all(pool).await.map_err(rows::sqlite_error)?;
        rows.iter().map(rows::decode_task).collect()
    }

    async fn update_task(&self, team_name: &str, task_id: &str, patch: &TaskPatch) -> Result<bool> {
        let pool = self.pool().await?;
        let mut tx = pool
            .begin_with("BEGIN IMMEDIATE")
            .await
            .map_err(rows::sqlite_error)?;
        let statement =
            format!("SELECT {TASK_COLUMNS} FROM team_task WHERE team_name = ? AND task_id = ?");
        let row = sqlx::query(&statement)
            .bind(team_name)
            .bind(task_id)
            .fetch_optional(&mut *tx)
            .await
            .map_err(rows::sqlite_error)?;
        let Some(row) = row else {
            tx.rollback().await.map_err(rows::sqlite_error)?;
            return Ok(false);
        };
        let mut record = rows::decode_task(&row)?;
        rows::apply_task_patch(&mut record, patch);
        let reviewers = rows::encode_list(&record.reviewers)?;
        let change_notes = rows::encode_list(&record.change_notes)?;
        sqlx::query(UPDATE_TASK_ROW)
            .bind(&record.title)
            .bind(&record.content)
            .bind(rows::encode_enum(record.status)?)
            .bind(record.assignee.as_deref())
            .bind(&reviewers)
            .bind(i64::from(record.review_round))
            .bind(i64::from(record.max_review_rounds))
            .bind(record.summary.as_deref())
            .bind(record.note.as_deref())
            .bind(record.parent_task_id.as_deref())
            .bind(record.parent_revision.map(i64::from))
            .bind(record.created_by.as_deref())
            .bind(i64::from(record.revision))
            .bind(i64::from(record.observed_revision))
            .bind(&change_notes)
            .bind(rows::to_i64(record.updated_at))
            .bind(team_name)
            .bind(task_id)
            .execute(&mut *tx)
            .await
            .map_err(rows::sqlite_error)?;
        tx.commit().await.map_err(rows::sqlite_error)?;
        Ok(true)
    }

    async fn transition_task(
        &self,
        team_name: &str,
        task_id: &str,
        from: TaskStatus,
        to: TaskStatus,
    ) -> Result<bool> {
        if !TaskStatus::is_valid_transition(from, to) {
            return Ok(false);
        }
        let pool = self.pool().await?;
        let result = sqlx::query(
            "UPDATE team_task SET status = ?, updated_at = MAX(?, updated_at + 1) \
             WHERE team_name = ? AND task_id = ? AND status = ?",
        )
        .bind(rows::encode_enum(to)?)
        .bind(rows::to_i64(refresh(0)))
        .bind(team_name)
        .bind(task_id)
        .bind(rows::encode_enum(from)?)
        .execute(pool)
        .await
        .map_err(rows::sqlite_error)?;
        Ok(result.rows_affected() == 1)
    }

    async fn add_dependency(&self, team_name: &str, task_id: &str, depends_on: &str) -> Result<()> {
        require_key(team_name, "team_name")?;
        require_key(task_id, "task_id")?;
        require_key(depends_on, "depends_on")?;
        if task_id == depends_on {
            return Err(Error::validation(format!(
                "task '{task_id}' cannot depend on itself"
            )));
        }
        let pool = self.pool().await?;
        sqlx::query(
            "INSERT OR IGNORE INTO team_task_dependency (team_name, task_id, depends_on) \
             VALUES (?, ?, ?)",
        )
        .bind(team_name)
        .bind(task_id)
        .bind(depends_on)
        .execute(pool)
        .await
        .map_err(rows::sqlite_error)?;
        Ok(())
    }

    async fn dependencies(&self, team_name: &str, task_id: &str) -> Result<Vec<String>> {
        let pool = self.pool().await?;
        sqlx::query_scalar(
            "SELECT depends_on FROM team_task_dependency \
             WHERE team_name = ? AND task_id = ? ORDER BY depends_on",
        )
        .bind(team_name)
        .bind(task_id)
        .fetch_all(pool)
        .await
        .map_err(rows::sqlite_error)
    }

    async fn dependents(&self, team_name: &str, task_id: &str) -> Result<Vec<String>> {
        let pool = self.pool().await?;
        sqlx::query_scalar(
            "SELECT task_id FROM team_task_dependency \
             WHERE team_name = ? AND depends_on = ? ORDER BY task_id",
        )
        .bind(team_name)
        .bind(task_id)
        .fetch_all(pool)
        .await
        .map_err(rows::sqlite_error)
    }

    async fn insert_vote(&self, record: ReviewVote) -> Result<()> {
        let pool = self.pool().await?;
        sqlx::query(UPSERT_VOTE)
            .bind(&record.team_name)
            .bind(&record.task_id)
            .bind(i64::from(record.round))
            .bind(&record.reviewer)
            .bind(rows::encode_enum(record.verdict)?)
            .execute(pool)
            .await
            .map_err(rows::sqlite_error)?;
        Ok(())
    }

    async fn votes(&self, team_name: &str, task_id: &str, round: u32) -> Result<Vec<ReviewVote>> {
        let pool = self.pool().await?;
        let statement = format!(
            "SELECT {VOTE_COLUMNS} FROM team_review_vote \
             WHERE team_name = ? AND task_id = ? AND review_round = ? ORDER BY reviewer"
        );
        let rows = sqlx::query(&statement)
            .bind(team_name)
            .bind(task_id)
            .bind(i64::from(round))
            .fetch_all(pool)
            .await
            .map_err(rows::sqlite_error)?;
        rows.iter().map(rows::decode_vote).collect()
    }

    async fn insert_message(&self, record: MessageRecord) -> Result<()> {
        require_key(&record.team_name, "team_name")?;
        require_key(&record.message_id, "message_id")?;
        let pool = self.pool().await?;
        sqlx::query(UPSERT_MESSAGE)
            .bind(&record.team_name)
            .bind(&record.message_id)
            .bind(&record.from_member_name)
            .bind(record.to_member_name.as_deref())
            .bind(&record.content)
            .bind(rows::to_i64(record.timestamp))
            .bind(record.broadcast)
            .bind(&record.protocol)
            .bind(record.is_read)
            .execute(pool)
            .await
            .map_err(rows::sqlite_error)?;
        Ok(())
    }

    async fn unread_messages(
        &self,
        team_name: &str,
        member_name: &str,
    ) -> Result<Vec<MessageRecord>> {
        let watermark = self.watermark_as_i64(team_name, member_name).await?;
        let pool = self.pool().await?;
        let statement = format!(
            "SELECT {MESSAGE_COLUMNS} FROM team_message WHERE team_name = ? AND ({UNREAD_PREDICATE}) \
             ORDER BY timestamp, message_id"
        );
        let rows = sqlx::query(&statement)
            .bind(team_name)
            .bind(member_name)
            .bind(member_name)
            .bind(watermark)
            .bind(watermark)
            .fetch_all(pool)
            .await
            .map_err(rows::sqlite_error)?;
        rows.iter().map(rows::decode_message).collect()
    }

    async fn mark_read(
        &self,
        team_name: &str,
        member_name: &str,
        message_ids: &[String],
    ) -> Result<()> {
        let pool = self.pool().await?;
        let mut tx = pool.begin().await.map_err(rows::sqlite_error)?;
        for message_id in message_ids {
            sqlx::query(MARK_MESSAGE_READ)
                .bind(team_name)
                .bind(message_id)
                .bind(member_name)
                .execute(&mut *tx)
                .await
                .map_err(rows::sqlite_error)?;
        }
        tx.commit().await.map_err(rows::sqlite_error)
    }

    async fn read_watermark(&self, team_name: &str, member_name: &str) -> Result<Option<u64>> {
        let pool = self.pool().await?;
        let stored: Option<i64> = sqlx::query_scalar(
            "SELECT watermark FROM team_read_watermark WHERE team_name = ? AND member_name = ?",
        )
        .bind(team_name)
        .bind(member_name)
        .fetch_optional(pool)
        .await
        .map_err(rows::sqlite_error)?;
        Ok(stored.map(|value| u64::try_from(value).unwrap_or_default()))
    }

    async fn set_read_watermark(
        &self,
        team_name: &str,
        member_name: &str,
        timestamp: u64,
    ) -> Result<()> {
        require_key(team_name, "team_name")?;
        require_key(member_name, "member_name")?;
        let pool = self.pool().await?;
        sqlx::query(ADVANCE_WATERMARK)
            .bind(team_name)
            .bind(member_name)
            .bind(rows::to_i64(timestamp))
            .execute(pool)
            .await
            .map_err(rows::sqlite_error)?;
        Ok(())
    }

    async fn message_count(&self, team_name: &str, member_name: &str) -> Result<usize> {
        let watermark = self.watermark_as_i64(team_name, member_name).await?;
        let pool = self.pool().await?;
        let statement = format!(
            "SELECT COUNT(*) FROM team_message WHERE team_name = ? AND ({UNREAD_PREDICATE})"
        );
        let count: i64 = sqlx::query_scalar(&statement)
            .bind(team_name)
            .bind(member_name)
            .bind(member_name)
            .bind(watermark)
            .bind(watermark)
            .fetch_one(pool)
            .await
            .map_err(rows::sqlite_error)?;
        Ok(usize::try_from(count).unwrap_or_default())
    }
}

impl SqliteTeamStore {
    /// Reads the member's broadcast watermark in the signed form the mailbox predicate binds.
    async fn watermark_as_i64(&self, team_name: &str, member_name: &str) -> Result<Option<i64>> {
        Ok(self
            .read_watermark(team_name, member_name)
            .await?
            .map(rows::to_i64))
    }
}
