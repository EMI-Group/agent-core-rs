//! `SQLite`-backed team memory (feature `sqlite`).
//!
//! [`SqliteTeamMemoryStore`] is the persistent counterpart of
//! [`InMemoryTeamMemoryStore`](super::InMemoryTeamMemoryStore): it implements
//! [`TeamMemoryStore`] over one `team_memories` table (see `sqlite::schema`) so team memory
//! survives a process restart. The schema is applied idempotently by every constructor,
//! so opening an existing database or a fresh path both work.
//!
//! # Columns
//!
//! One row per [`TeamMemoryRecord`]: the scalar fields map to columns of the same name,
//! `visibility`/`tags`/`metadata` are stored as JSON text (`visibility_json`,
//! `tags_json`, `metadata_json`, always present), and `expires_at` is `NULL` when the
//! record never expires. Timestamps are RFC 3339 UTC strings with millisecond precision
//! and a literal `Z`, which makes them lexicographically ordered exactly like the
//! instants they denote — so `ORDER BY`, `MIN`/`MAX`, and the expiry comparisons are plain
//! text comparisons (see `sqlite::schema`).
//!
//! # Parity with the in-memory backend
//!
//! Behaviour matches [`InMemoryTeamMemoryStore`](super::InMemoryTeamMemoryStore)
//! method for method: `upsert` preserves an existing row's `created_at` and returns the
//! stored record, `get` returns expired rows too, `list` orders by `updated_at`
//! descending then `id` ascending and applies `offset`/`limit`, `search` delegates to the
//! shared `rank_candidates` (so the ranking formula cannot drift), `delete`/`set_expiry`
//! report whether a row of *this* team changed, and `prune_expired` removes the rows
//! whose expiry is at or before `now`. Deleted and expired-row handling is identical
//! because expiry stays a read-view concern here too.
//!
//! # Deliberate differences
//!
//! - **Case folding.** `text_contains` is pushed into SQL as
//!   `lower(content) LIKE ? OR lower(title) LIKE ?`, and `SQLite`'s `lower` folds ASCII
//!   only; the in-memory backend folds full Unicode in Rust. A non-ASCII needle
//!   therefore matches case-insensitively in memory but only case-sensitively here.
//! - **Timestamp precision.** Rows are written and read at millisecond precision, so
//!   `upsert` returns the record as it was persisted (sub-millisecond digits of
//!   `created_at`/`updated_at`/`expires_at` are truncated), which is exactly what a
//!   following `get` reads back. The in-memory backend stores the input verbatim.
//! - **Pagination in SQL.** `limit`/`offset` are applied by `LIMIT`/`OFFSET`, not by
//!   skipping in Rust; the ordering is identical, so the same pages come back.
//! - **Malformed rows degrade, they do not fail the listing.** JSON columns fall back to
//!   fail-closed/empty defaults and unparseable timestamps skip the row; every fallback
//!   is logged with `tracing` (see `sqlite::decode`).
//!
//! # Query construction
//!
//! Statements are built with numbered placeholders (`?1`, `?2`, ...) through the private
//! `sqlite::query::Where` helper, and every value is bound — no caller input is ever formatted
//! into SQL. A bare `?` is never mixed with `?N` in one statement: the helper numbers
//! dynamically built predicates, and fixed suffixes (`LIMIT ?12 OFFSET ?13`) reuse the
//! indices it hands out.
mod decode;
mod query;
mod schema;
#[cfg(test)]
#[allow(clippy::assert_is_empty, clippy::unwrap_used, clippy::expect_used)]
mod tests;
use super::model::{
    TeamMemoryFilter, TeamMemoryRecord, TeamMemoryScope, TeamMemoryStats, TeamSearchHit,
};
use super::{rank_candidates, TeamMemoryStore, TeamSearchOptions};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use openjiuwen_common::error::{Error, Result};
use query::{filter_where, int_param, push_kinds, push_live, Param, Where};
use schema::{parse_ts, render_ts, truncate_ts, COLUMNS, DDL};
use sqlx::query::Query;
use sqlx::sqlite::{SqliteArguments, SqliteConnectOptions, SqlitePoolOptions};
use sqlx::{Row, Sqlite, SqlitePool};
use std::path::Path;
use std::str::FromStr;
use tracing::{debug, warn};
/// The tracing target of every log line this module emits.
const LOG_TARGET: &str = "openjiuwen_memory::team";
/// The upsert of one record.
///
/// `ON CONFLICT` overwrites every column except `created_at`, so an update never rewrites
/// history; `RETURNING created_at` hands back the persisted value (the input's on insert,
/// the existing row's on update) without a second round trip.
const UPSERT_SQL: &str = "\
INSERT INTO team_memories
    (id, team_id, member_id, session_id, kind, visibility_json, title, content, importance,
     tags_json, metadata_json, source, created_at, updated_at, expires_at)
VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15)
ON CONFLICT(id) DO UPDATE SET
    team_id = excluded.team_id,
    member_id = excluded.member_id,
    session_id = excluded.session_id,
    kind = excluded.kind,
    visibility_json = excluded.visibility_json,
    title = excluded.title,
    content = excluded.content,
    importance = excluded.importance,
    tags_json = excluded.tags_json,
    metadata_json = excluded.metadata_json,
    source = excluded.source,
    updated_at = excluded.updated_at,
    expires_at = excluded.expires_at
RETURNING created_at";
/// The read ordering of every listing: newest update first, then id.
const LIST_ORDER: &str = " ORDER BY updated_at DESC, id ASC";
/// A `SQLite`-backed team memory store.
///
/// Clones share one connection pool, so a `SqliteTeamMemoryStore` can be cloned into
/// several facades without duplicating connections or writes.
#[derive(Debug, Clone)]
pub struct SqliteTeamMemoryStore {
    /// The connection pool every operation borrows from.
    pool: SqlitePool,
}
impl SqliteTeamMemoryStore {
    /// Opens (creating if missing) the database at `path` and applies the schema.
    ///
    /// The parent directory is created when it is non-empty, so `data/teams/memory.db`
    /// works on a fresh checkout. Missing files are created.
    ///
    /// # Errors
    ///
    /// [`Error::Memory`] when the parent directory cannot be created, and
    /// [`Error::External`] when the database cannot be opened or the schema cannot be
    /// applied.
    pub async fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            tokio::fs::create_dir_all(parent).await.map_err(|error| {
                Error::memory(format!(
                    "team memory sqlite: cannot create directory {}: {error}",
                    parent.display()
                ))
            })?;
        }
        let options = SqliteConnectOptions::new()
            .filename(path)
            .create_if_missing(true);
        Self::connect_with(options, 4).await
    }
    /// Connects to a `SQLite` URL such as `sqlite::memory:` or `sqlite:team-memory.db`
    /// and applies the schema.
    ///
    /// An in-memory URL is pinned to a single pooled connection: an in-memory database is
    /// private to the connection that created it, so a second pooled connection would see
    /// an empty database.
    ///
    /// # Errors
    ///
    /// [`Error::External`] when the URL cannot be parsed, the database cannot be opened,
    /// or the schema cannot be applied.
    pub async fn connect(url: &str) -> Result<Self> {
        let options = SqliteConnectOptions::from_str(url)
            .map_err(Error::external)?
            .create_if_missing(true);
        let max_connections = if url.contains("memory") { 1 } else { 4 };
        Self::connect_with(options, max_connections).await
    }
    /// Wraps an existing pool and makes sure the schema exists.
    ///
    /// # Errors
    ///
    /// [`Error::External`] when the schema cannot be applied.
    pub async fn from_pool(pool: SqlitePool) -> Result<Self> {
        for statement in DDL {
            sqlx::query(statement)
                .execute(&pool)
                .await
                .map_err(Error::external)?;
        }
        Ok(Self { pool })
    }
    /// The underlying pool, for callers that need to run companion queries.
    #[must_use]
    pub fn pool(&self) -> &SqlitePool {
        &self.pool
    }
    /// Connects with `options`, builds a pool of `max_connections`, and applies the
    /// schema.
    async fn connect_with(options: SqliteConnectOptions, max_connections: u32) -> Result<Self> {
        let pool = SqlitePoolOptions::new()
            .max_connections(max_connections)
            .connect_with(options)
            .await
            .map_err(Error::external)?;
        Self::from_pool(pool).await
    }
    /// Runs one `SELECT` of the record columns and decodes every returned row.
    ///
    /// `suffix` carries the ordering and pagination (or nothing); `where_clause` carries
    /// the predicate and its bind values.
    async fn fetch_records(
        &self,
        where_clause: &Where,
        suffix: &str,
    ) -> Result<Vec<TeamMemoryRecord>> {
        let sql = format!(
            "SELECT {COLUMNS} FROM team_memories WHERE {}{suffix}",
            where_clause.sql()
        );
        let query = bind_params(sqlx::query(&sql), where_clause.params());
        let rows = query.fetch_all(&self.pool).await.map_err(Error::external)?;
        let mut records = Vec::with_capacity(rows.len());
        for row in &rows {
            if let Some(record) = decode::decode_row(row)? {
                records.push(record);
            }
        }
        Ok(records)
    }
    /// Fills `total`, `expired`, `oldest`, and `newest` for `where_clause`.
    async fn fill_totals(
        &self,
        where_clause: &Where,
        now: DateTime<Utc>,
        stats: &mut TeamMemoryStats,
    ) -> Result<()> {
        let mut scoped = where_clause.clone();
        let expired_index = scoped.push(Param::Text(render_ts(now)));
        let sql = format!(
            "SELECT COUNT(*) AS total, MIN(created_at) AS oldest, MAX(created_at) AS newest, \
             SUM(CASE WHEN expires_at IS NOT NULL AND expires_at <= ?{expired_index} \
             THEN 1 ELSE 0 END) AS expired \
             FROM team_memories WHERE {}",
            scoped.sql()
        );
        let query = bind_params(sqlx::query(&sql), scoped.params());
        let row = query.fetch_one(&self.pool).await.map_err(Error::external)?;
        let total: i64 = row.try_get("total").map_err(Error::external)?;
        let expired: Option<i64> = row.try_get("expired").map_err(Error::external)?;
        stats.total = usize::try_from(total).unwrap_or(0);
        stats.expired = usize::try_from(expired.unwrap_or(0)).unwrap_or(0);
        stats.oldest = decode_bound(&row, "oldest")?;
        stats.newest = decode_bound(&row, "newest")?;
        Ok(())
    }
    /// Fills `by_kind` for `where_clause`.
    async fn fill_by_kind(&self, where_clause: &Where, stats: &mut TeamMemoryStats) -> Result<()> {
        let sql = format!(
            "SELECT kind AS bucket, COUNT(*) AS n FROM team_memories WHERE {} GROUP BY kind",
            where_clause.sql()
        );
        let query = bind_params(sqlx::query(&sql), where_clause.params());
        let rows = query.fetch_all(&self.pool).await.map_err(Error::external)?;
        for row in &rows {
            let kind: String = row.try_get("bucket").map_err(Error::external)?;
            let count: i64 = row.try_get("n").map_err(Error::external)?;
            stats
                .by_kind
                .insert(kind, usize::try_from(count).unwrap_or(0));
        }
        Ok(())
    }
    /// Fills `by_member` for `where_clause`, bucketing member-less rows under
    /// [`TEAM_WIDE_MEMBER_KEY`](super::model::TEAM_WIDE_MEMBER_KEY).
    async fn fill_by_member(
        &self,
        where_clause: &Where,
        stats: &mut TeamMemoryStats,
    ) -> Result<()> {
        let mut scoped = where_clause.clone();
        let key_index = scoped.push(Param::Text(super::model::TEAM_WIDE_MEMBER_KEY.to_owned()));
        let sql = format!(
            "SELECT COALESCE(member_id, ?{key_index}) AS bucket, COUNT(*) AS n \
             FROM team_memories WHERE {} GROUP BY bucket",
            scoped.sql()
        );
        let query = bind_params(sqlx::query(&sql), scoped.params());
        let rows = query.fetch_all(&self.pool).await.map_err(Error::external)?;
        for row in &rows {
            let member: String = row.try_get("bucket").map_err(Error::external)?;
            let count: i64 = row.try_get("n").map_err(Error::external)?;
            stats
                .by_member
                .insert(member, usize::try_from(count).unwrap_or(0));
        }
        Ok(())
    }
}
/// Binds accumulated parameters to `query`, in numbered-placeholder order.
///
/// Values are cloned into the query, so no borrow of the parameter slice outlives this
/// call.
fn bind_params<'q>(
    mut query: Query<'q, Sqlite, SqliteArguments<'q>>,
    params: &[Param],
) -> Query<'q, Sqlite, SqliteArguments<'q>> {
    for param in params {
        query = match param {
            Param::Text(value) => query.bind(value.clone()),
            Param::Int(value) => query.bind(*value),
            Param::Real(value) => query.bind(*value),
        };
    }
    query
}
/// Decodes an aggregate timestamp column (`oldest`/`newest`).
///
/// A `NULL` (no matching row) is `None`; an unparseable value is `None` plus a warning,
/// which keeps a statistics call from failing over one odd row.
fn decode_bound(row: &sqlx::sqlite::SqliteRow, column: &str) -> Result<Option<DateTime<Utc>>> {
    let raw: Option<String> = row.try_get(column).map_err(Error::external)?;
    let Some(raw) = raw else {
        return Ok(None);
    };
    let parsed = parse_ts(&raw);
    if parsed.is_none() {
        warn!(
            target: LOG_TARGET,
            column = column,
            value = %raw,
            "unparseable team memory timestamp aggregate; reporting none"
        );
    }
    Ok(parsed)
}
#[async_trait]
impl TeamMemoryStore for SqliteTeamMemoryStore {
    async fn upsert(&self, record: &TeamMemoryRecord) -> Result<TeamMemoryRecord> {
        record.validate()?;
        let visibility_json = serde_json::to_string(&record.visibility).map_err(Error::external)?;
        let tags_json = serde_json::to_string(&record.tags).map_err(Error::external)?;
        let metadata_json = serde_json::to_string(&record.metadata).map_err(Error::external)?;
        let created_at = render_ts(record.created_at);
        let updated_at = render_ts(record.updated_at);
        let expires_at = record.expires_at.map(render_ts);
        let row = sqlx::query(UPSERT_SQL)
            .bind(&record.id)
            .bind(&record.team_id)
            .bind(record.member_id.as_deref())
            .bind(record.session_id.as_deref())
            .bind(record.kind.as_str())
            .bind(&visibility_json)
            .bind(record.title.as_deref())
            .bind(&record.content)
            .bind(f64::from(record.importance))
            .bind(&tags_json)
            .bind(&metadata_json)
            .bind(record.source.as_deref())
            .bind(&created_at)
            .bind(&updated_at)
            .bind(expires_at.as_deref())
            .fetch_one(&self.pool)
            .await
            .map_err(Error::external)?;
        let persisted: String = row.try_get("created_at").map_err(Error::external)?;
        let created_at = parse_ts(&persisted).ok_or_else(|| {
            Error::memory(format!(
                "team memory `{}`: unparseable `created_at` `{persisted}` returned by SQLite",
                record.id
            ))
        })?;
        let stored = TeamMemoryRecord {
            created_at,
            updated_at: truncate_ts(record.updated_at),
            expires_at: record.expires_at.map(truncate_ts),
            ..record.clone()
        };
        debug!(
            target: LOG_TARGET,
            team_id = %stored.team_id,
            id = %stored.id,
            "stored team memory record in SQLite"
        );
        Ok(stored)
    }
    async fn get(&self, team_id: &str, id: &str) -> Result<Option<TeamMemoryRecord>> {
        let mut where_clause = Where::new();
        where_clause.bind("team_id = ", Param::Text(team_id.to_owned()));
        where_clause.bind("id = ", Param::Text(id.to_owned()));
        let records = self.fetch_records(&where_clause, "").await?;
        Ok(records.into_iter().next())
    }
    async fn list(&self, filter: &TeamMemoryFilter) -> Result<Vec<TeamMemoryRecord>> {
        filter.validate()?;
        let mut where_clause = filter_where(filter, Utc::now())?;
        let limit = filter
            .limit
            .map(|limit| where_clause.push(Param::Int(int_param(limit))));
        let offset = filter
            .offset
            .map(|offset| where_clause.push(Param::Int(int_param(offset))));
        let suffix = match (limit, offset) {
            (Some(limit), Some(offset)) => {
                format!("{LIST_ORDER} LIMIT ?{limit} OFFSET ?{offset}")
            }
            (Some(limit), None) => format!("{LIST_ORDER} LIMIT ?{limit}"),
            // `OFFSET` without `LIMIT` is a syntax error in SQLite; `LIMIT -1` means
            // "no limit".
            (None, Some(offset)) => format!("{LIST_ORDER} LIMIT -1 OFFSET ?{offset}"),
            (None, None) => LIST_ORDER.to_owned(),
        };
        self.fetch_records(&where_clause, &suffix).await
    }
    async fn search(
        &self,
        scope: &TeamMemoryScope,
        query: &str,
        options: &TeamSearchOptions,
    ) -> Result<Vec<TeamSearchHit>> {
        scope.validate()?;
        options.validate()?;
        let mut where_clause = Where::new();
        where_clause.bind("team_id = ", Param::Text(scope.team_id.clone()));
        push_kinds(&mut where_clause, &options.kinds);
        if !options.include_expired {
            push_live(&mut where_clause, Utc::now());
        }
        let mut records = self.fetch_records(&where_clause, "").await?;
        records.retain(|record| scope.allows(record));
        Ok(rank_candidates(records, query, options, Utc::now()))
    }
    async fn delete(&self, team_id: &str, id: &str) -> Result<bool> {
        let result = sqlx::query("DELETE FROM team_memories WHERE team_id = ?1 AND id = ?2")
            .bind(team_id)
            .bind(id)
            .execute(&self.pool)
            .await
            .map_err(Error::external)?;
        let removed = result.rows_affected() > 0;
        if removed {
            debug!(target: LOG_TARGET, team_id = %team_id, id = %id, "deleted team memory record");
        }
        Ok(removed)
    }
    async fn clear(&self, filter: &TeamMemoryFilter) -> Result<usize> {
        filter.validate()?;
        let where_clause = filter_where(filter, Utc::now())?;
        let sql = format!("DELETE FROM team_memories WHERE {}", where_clause.sql());
        let query = bind_params(sqlx::query(&sql), where_clause.params());
        let result = query.execute(&self.pool).await.map_err(Error::external)?;
        let removed = usize::try_from(result.rows_affected()).unwrap_or(usize::MAX);
        debug!(
            target: LOG_TARGET,
            team_id = %filter.team_id,
            removed,
            "cleared team memory records"
        );
        Ok(removed)
    }
    async fn stats(&self, filter: &TeamMemoryFilter) -> Result<TeamMemoryStats> {
        filter.validate()?;
        let now = Utc::now();
        let where_clause = filter_where(filter, now)?;
        let mut stats = TeamMemoryStats::default();
        self.fill_totals(&where_clause, now, &mut stats).await?;
        self.fill_by_kind(&where_clause, &mut stats).await?;
        self.fill_by_member(&where_clause, &mut stats).await?;
        Ok(stats)
    }
    async fn set_expiry(
        &self,
        team_id: &str,
        id: &str,
        expires_at: Option<DateTime<Utc>>,
    ) -> Result<bool> {
        let result =
            sqlx::query("UPDATE team_memories SET expires_at = ?1 WHERE team_id = ?2 AND id = ?3")
                .bind(expires_at.map(render_ts))
                .bind(team_id)
                .bind(id)
                .execute(&self.pool)
                .await
                .map_err(Error::external)?;
        Ok(result.rows_affected() > 0)
    }
    async fn prune_expired(&self, now: DateTime<Utc>) -> Result<usize> {
        let result = sqlx::query(
            "DELETE FROM team_memories WHERE expires_at IS NOT NULL AND expires_at <= ?1",
        )
        .bind(render_ts(now))
        .execute(&self.pool)
        .await
        .map_err(Error::external)?;
        let removed = usize::try_from(result.rows_affected()).unwrap_or(usize::MAX);
        debug!(target: LOG_TARGET, removed, "pruned expired team memory records");
        Ok(removed)
    }
}
