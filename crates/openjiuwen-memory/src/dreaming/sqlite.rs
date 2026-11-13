//! [`SqliteDreamRepository`]: the durable, feature-gated dream-job backend.
//!
//! One `dream_jobs` table holds every job, with the free-form payload/metadata and the
//! metrics stored as JSON text. Timestamps are RFC 3339 UTC text with fixed microsecond
//! precision ([`SecondsFormat::Micros`], `Z` suffix), so lexicographic TEXT ordering is
//! chronological ordering: the retry predicate and the retention cutoff are plain string
//! comparisons, and the claim index can use them without a functional index.
//!
//! Every query is parameter-bound — the only interpolated text is the module-level
//! column list const, never user data. Claiming is a single atomic
//! `UPDATE ... RETURNING` statement, so two workers racing on the same row cannot both
//! win: `SQLite` serializes writers, the loser re-evaluates the `state = 'pending'`
//! predicate against the committed row and matches nothing.
use std::collections::BTreeMap;
use std::fmt;
use std::str::FromStr;
use std::time::Duration as StdDuration;

use async_trait::async_trait;
use chrono::{DateTime, SecondsFormat, Utc};
use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions, SqliteRow};
use sqlx::{Row, SqlitePool};

use openjiuwen_common::error::{Error, Result};

use crate::record::MemoryScope;

use super::model::{DreamJob, DreamKind, DreamMetrics, DreamState, DreamTrigger};
use super::outcome::DreamOutcome;
use super::repository::{DreamFilter, DreamRepository};

/// Job table DDL, applied idempotently by every constructor.
const SCHEMA_JOBS: &str = "
CREATE TABLE IF NOT EXISTS dream_jobs (
    id TEXT PRIMARY KEY NOT NULL,
    user_id TEXT NOT NULL,
    scope_id TEXT NOT NULL,
    session_id TEXT,
    agent_id TEXT,
    kind TEXT NOT NULL,
    state TEXT NOT NULL,
    trigger TEXT NOT NULL,
    priority INTEGER NOT NULL DEFAULT 0,
    attempts INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 3,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    last_error TEXT,
    metrics_json TEXT NOT NULL DEFAULT '{}',
    payload_json TEXT NOT NULL DEFAULT '{}',
    metadata_json TEXT NOT NULL DEFAULT '{}'
)";

/// Index backing the claim head and the list ordering.
const CLAIM_INDEX: &str = "CREATE INDEX IF NOT EXISTS dream_jobs_claim_idx \
     ON dream_jobs (state, priority DESC, created_at ASC, id ASC)";

/// Index backing scope-scoped listings.
const SCOPE_INDEX: &str =
    "CREATE INDEX IF NOT EXISTS dream_jobs_scope_idx ON dream_jobs (user_id, scope_id)";

/// The shared column list of every read [`DreamJob`] performs.
///
/// Spelled once so the claim's `RETURNING` clause and the plain `SELECT`s decode through
/// the same [`row_to_job`] helper and can never drift apart.
const COLUMNS: &str = "id, user_id, scope_id, session_id, agent_id, kind, state, trigger, \
     priority, attempts, max_attempts, created_at, updated_at, started_at, finished_at, \
     last_error, metrics_json, payload_json, metadata_json";

/// The terminal states a retention sweep may delete; mirrors
/// [`DreamState::is_terminal`].
const TERMINAL_STATES: &str = "'succeeded', 'failed', 'skipped', 'cancelled'";

/// Inserts a job, keeping an existing row untouched so the caller can detect a duplicate.
const INSERT_JOB: &str = "
INSERT INTO dream_jobs (
    id, user_id, scope_id, session_id, agent_id, kind, state, trigger, priority, attempts,
    max_attempts, created_at, updated_at, started_at, finished_at, last_error,
    metrics_json, payload_json, metadata_json
) VALUES (
    ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19
) ON CONFLICT(id) DO NOTHING";

/// The single-statement claim: pick the head, move it to `running`, and return it.
const CLAIM_JOB: &str = "
UPDATE dream_jobs
   SET state = 'running', attempts = attempts + 1, started_at = ?1, updated_at = ?1
 WHERE id = (
        SELECT id FROM dream_jobs
         WHERE state = 'pending' AND attempts < max_attempts
           AND (started_at IS NULL OR started_at < ?1)
         ORDER BY priority DESC, created_at ASC, id ASC
         LIMIT 1
       )
   AND state = 'pending'
RETURNING id, user_id, scope_id, session_id, agent_id, kind, state, trigger, priority, \
          attempts, max_attempts, created_at, updated_at, started_at, finished_at, \
          last_error, metrics_json, payload_json, metadata_json";

/// A [`DreamRepository`] backed by a `SQLite` database.
///
/// Clones share the same connection pool. The backend implements the full
/// [`DreamRepository`] contract, including the retry marker in `started_at` and the
/// attempt running out into terminal [`DreamState::Failed`]; `purge` uses the trait's
/// default implementation over [`DreamRepository::delete_finished`].
#[derive(Debug, Clone)]
pub struct SqliteDreamRepository {
    /// The connection pool every operation borrows from.
    pool: SqlitePool,
}

impl SqliteDreamRepository {
    /// Connects to `url` (for example `sqlite::memory:` or `sqlite:dreams.db`).
    ///
    /// The database file is created when missing, in-memory URLs are pinned to a single
    /// pooled connection (otherwise every checkout would open a fresh, empty database),
    /// and lock contention waits up to five seconds before failing.
    pub async fn connect(url: &str) -> Result<Self> {
        let options = SqliteConnectOptions::from_str(url)
            .map_err(Error::external)?
            .create_if_missing(true)
            .busy_timeout(StdDuration::from_secs(5));
        let pool_options = SqlitePoolOptions::new();
        let pool_options = if url.contains("memory") {
            pool_options.max_connections(1)
        } else {
            pool_options
        };
        let pool = pool_options
            .connect_with(options)
            .await
            .map_err(Error::external)?;
        Self::from_pool(pool).await
    }

    /// Wraps an existing pool (created by a caller or another subsystem) and makes
    /// sure the schema exists.
    pub async fn from_pool(pool: SqlitePool) -> Result<Self> {
        for statement in [SCHEMA_JOBS, CLAIM_INDEX, SCOPE_INDEX] {
            sqlx::query(statement)
                .execute(&pool)
                .await
                .map_err(Error::external)?;
        }
        Ok(Self { pool })
    }

    /// The underlying pool, for callers that need to run companion queries.
    pub fn pool(&self) -> &SqlitePool {
        &self.pool
    }
}

#[async_trait]
impl DreamRepository for SqliteDreamRepository {
    async fn enqueue(&self, job: DreamJob) -> Result<DreamJob> {
        job.validate()?;
        let metrics_json = encode_json("metrics_json", &job.metrics)?;
        let payload_json = encode_json("payload_json", &job.payload)?;
        let metadata_json = encode_json("metadata_json", &job.metadata)?;
        let result = sqlx::query(INSERT_JOB)
            .bind(&job.id)
            .bind(&job.scope.user_id)
            .bind(&job.scope.scope_id)
            .bind(job.scope.session_id.as_deref())
            .bind(job.scope.agent_id.as_deref())
            .bind(job.kind.as_str())
            .bind(job.state.as_str())
            .bind(job.trigger.as_str())
            .bind(job.priority)
            .bind(job.attempts)
            .bind(job.max_attempts)
            .bind(format_time(job.created_at))
            .bind(format_time(job.updated_at))
            .bind(job.started_at.map(format_time))
            .bind(job.finished_at.map(format_time))
            .bind(job.last_error.as_deref())
            .bind(metrics_json)
            .bind(payload_json)
            .bind(metadata_json)
            .execute(&self.pool)
            .await
            .map_err(Error::external)?;
        if result.rows_affected() == 0 {
            return Err(Error::validation(format!(
                "dream job `{}` already exists",
                job.id
            )));
        }
        tracing::debug!(
            target: "openjiuwen_memory::dreaming::sqlite",
            job_id = %job.id,
            kind = job.kind.as_str(),
            "enqueued dream job"
        );
        Ok(job)
    }

    async fn claim_next(&self, now: DateTime<Utc>) -> Result<Option<DreamJob>> {
        let row = sqlx::query(CLAIM_JOB)
            .bind(format_time(now))
            .fetch_optional(&self.pool)
            .await
            .map_err(Error::external)?;
        let claimed = row.as_ref().map(row_to_job).transpose()?;
        if let Some(job) = &claimed {
            tracing::debug!(
                target: "openjiuwen_memory::dreaming::sqlite",
                job_id = %job.id,
                attempts = job.attempts,
                "claimed dream job"
            );
        }
        Ok(claimed)
    }

    async fn complete(&self, job_id: &str, outcome: DreamOutcome) -> Result<DreamJob> {
        outcome.validate()?;
        if outcome.job_id != job_id {
            return Err(Error::validation(format!(
                "dream outcome job id `{}` does not match `{job_id}`",
                outcome.job_id
            )));
        }
        let metrics_json = encode_json("metrics_json", &outcome.metrics)?;
        let now = format_time(Utc::now());
        let mut transaction = self.pool.begin().await.map_err(Error::external)?;
        let result = sqlx::query(
            "UPDATE dream_jobs
                SET state = ?2, metrics_json = ?3, finished_at = ?4, updated_at = ?4,
                    last_error = NULL
              WHERE id = ?1 AND state = 'running'",
        )
        .bind(job_id)
        .bind(outcome.state.as_str())
        .bind(metrics_json)
        .bind(&now)
        .execute(&mut *transaction)
        .await
        .map_err(Error::external)?;
        if result.rows_affected() == 0 {
            transaction.rollback().await.map_err(Error::external)?;
            return Err(self.not_completable(job_id).await?);
        }
        let row = sqlx::query(&format!("SELECT {COLUMNS} FROM dream_jobs WHERE id = ?1"))
            .bind(job_id)
            .fetch_one(&mut *transaction)
            .await
            .map_err(Error::external)?;
        let job = row_to_job(&row)?;
        transaction.commit().await.map_err(Error::external)?;
        tracing::debug!(
            target: "openjiuwen_memory::dreaming::sqlite",
            job_id = %job.id,
            state = job.state.as_str(),
            "completed dream job"
        );
        Ok(job)
    }

    async fn fail(&self, job_id: &str, error: &str) -> Result<DreamJob> {
        let now = format_time(Utc::now());
        let mut transaction = self.pool.begin().await.map_err(Error::external)?;
        // `started_at` is deliberately preserved: it is the retry marker, so the next
        // claim only succeeds once the caller's clock has moved past it.
        let result = sqlx::query(
            "UPDATE dream_jobs
                SET state = CASE WHEN attempts >= max_attempts THEN 'failed' ELSE 'pending' END,
                    last_error = ?2,
                    finished_at = CASE WHEN attempts >= max_attempts THEN ?3 ELSE NULL END,
                    updated_at = ?3
              WHERE id = ?1 AND state = 'running'",
        )
        .bind(job_id)
        .bind(error)
        .bind(&now)
        .execute(&mut *transaction)
        .await
        .map_err(Error::external)?;
        if result.rows_affected() == 0 {
            transaction.rollback().await.map_err(Error::external)?;
            return Err(self.not_failable(job_id).await?);
        }
        let row = sqlx::query(&format!("SELECT {COLUMNS} FROM dream_jobs WHERE id = ?1"))
            .bind(job_id)
            .fetch_one(&mut *transaction)
            .await
            .map_err(Error::external)?;
        let job = row_to_job(&row)?;
        transaction.commit().await.map_err(Error::external)?;
        tracing::debug!(
            target: "openjiuwen_memory::dreaming::sqlite",
            job_id = %job.id,
            state = job.state.as_str(),
            attempts = job.attempts,
            "recorded a failed dream attempt"
        );
        Ok(job)
    }

    async fn get(&self, job_id: &str) -> Result<Option<DreamJob>> {
        let row = sqlx::query(&format!("SELECT {COLUMNS} FROM dream_jobs WHERE id = ?1"))
            .bind(job_id)
            .fetch_optional(&self.pool)
            .await
            .map_err(Error::external)?;
        row.as_ref().map(row_to_job).transpose()
    }

    async fn list(&self, filter: &DreamFilter) -> Result<Vec<DreamJob>> {
        // The tenant predicates are narrowed in SQL; the remaining clauses (ids, states,
        // kinds, triggers, the exclusive time bounds) are residual and stay in Rust, so
        // both backends rank and truncate identically through `DreamFilter::matches`.
        let created_after = filter.created_after.map(format_time);
        let created_before = filter.created_before.map(format_time);
        let rows = sqlx::query(&format!(
            "SELECT {COLUMNS} FROM dream_jobs
              WHERE (?1 IS NULL OR user_id = ?1)
                AND (?2 IS NULL OR scope_id = ?2)
                AND (?3 IS NULL OR session_id = ?3)
                AND (?4 IS NULL OR agent_id = ?4)
                AND (?5 IS NULL OR created_at > ?5)
                AND (?6 IS NULL OR created_at < ?6)
              ORDER BY priority DESC, created_at ASC, id ASC"
        ))
        .bind(filter.user_id.as_deref())
        .bind(filter.scope_id.as_deref())
        .bind(filter.session_id.as_deref())
        .bind(filter.agent_id.as_deref())
        .bind(created_after)
        .bind(created_before)
        .fetch_all(&self.pool)
        .await
        .map_err(Error::external)?;
        let mut matched: Vec<DreamJob> = rows
            .iter()
            .map(row_to_job)
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .filter(|job| filter.matches(job))
            .collect();
        if let Some(limit) = filter.limit {
            matched.truncate(limit);
        }
        Ok(matched)
    }

    async fn count_by_state(&self) -> Result<BTreeMap<DreamState, usize>> {
        let rows = sqlx::query("SELECT state, COUNT(*) AS n FROM dream_jobs GROUP BY state")
            .fetch_all(&self.pool)
            .await
            .map_err(Error::external)?;
        let mut counts: BTreeMap<DreamState, usize> = DreamState::ALL
            .into_iter()
            .map(|state| (state, 0))
            .collect();
        for row in &rows {
            let raw: String = column(row, "state")?;
            let total: i64 = column(row, "n")?;
            if let Some(state) = DreamState::parse(&raw) {
                counts.insert(state, usize::try_from(total).unwrap_or(0));
            } else {
                tracing::warn!(
                    target: "openjiuwen_memory::dreaming::sqlite",
                    state = %raw,
                    "ignoring a dream job row with an unknown state"
                );
            }
        }
        Ok(counts)
    }

    async fn delete_finished(&self, older_than: DateTime<Utc>) -> Result<usize> {
        let result = sqlx::query(&format!(
            "DELETE FROM dream_jobs
              WHERE state IN ({TERMINAL_STATES})
                AND COALESCE(finished_at, updated_at) < ?1"
        ))
        .bind(format_time(older_than))
        .execute(&self.pool)
        .await
        .map_err(Error::external)?;
        Ok(usize::try_from(result.rows_affected()).unwrap_or(usize::MAX))
    }
}

impl SqliteDreamRepository {
    /// Explains why a `complete` matched no running row: absent or not running.
    async fn not_completable(&self, job_id: &str) -> Result<Error> {
        Ok(match self.get(job_id).await? {
            None => Error::not_found(format!("dream job `{job_id}` does not exist")),
            Some(job) => Error::validation(format!(
                "dream job `{job_id}` is `{}` but must be `running` to complete",
                job.state
            )),
        })
    }

    /// Explains why a `fail` matched no running row: absent or not running.
    async fn not_failable(&self, job_id: &str) -> Result<Error> {
        Ok(match self.get(job_id).await? {
            None => Error::not_found(format!("dream job `{job_id}` does not exist")),
            Some(job) => Error::validation(format!(
                "dream job `{job_id}` is `{}` but must be `running` to fail",
                job.state
            )),
        })
    }
}

/// Formats `value` in the single canonical storage format: RFC 3339, microseconds, `Z`.
///
/// The fixed width and precision are what make TEXT ordering equal chronological
/// ordering, so every write goes through this helper.
fn format_time(value: DateTime<Utc>) -> String {
    value.to_rfc3339_opts(SecondsFormat::Micros, true)
}

/// A stored-value decoding failure, naming the column it came from.
#[derive(Debug)]
struct DecodeError {
    /// Column whose stored representation could not be decoded.
    column: &'static str,
    /// The underlying parser error.
    cause: Box<dyn std::error::Error + Send + Sync>,
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "dream job column `{}` could not be decoded: {}",
            self.column, self.cause
        )
    }
}

impl std::error::Error for DecodeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.cause.as_ref())
    }
}

/// Builds a [`DecodeError`] for `column`, wrapping the parser failure.
fn decode_error(
    column: &'static str,
    cause: impl std::error::Error + Send + Sync + 'static,
) -> Error {
    Error::external(DecodeError {
        column,
        cause: Box::new(cause),
    })
}

/// Reads one non-null column, mapping SQLite and type errors to [`Error::External`].
fn column<'r, T>(row: &'r SqliteRow, name: &str) -> Result<T>
where
    T: sqlx::Decode<'r, sqlx::Sqlite> + sqlx::Type<sqlx::Sqlite>,
{
    row.try_get(name).map_err(Error::external)
}

/// Decodes a stored RFC 3339 timestamp back into UTC.
fn decode_time(column_name: &'static str, raw: &str) -> Result<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(raw)
        .map(|value| value.with_timezone(&Utc))
        .map_err(|error| decode_error(column_name, error))
}

/// Decodes a stored JSON object column.
fn decode_json<T>(column_name: &'static str, raw: &str) -> Result<T>
where
    T: serde::de::DeserializeOwned,
{
    serde_json::from_str(raw).map_err(|error| decode_error(column_name, error))
}

/// Encodes a value as the JSON text of a column.
fn encode_json<T: serde::Serialize>(column_name: &'static str, value: &T) -> Result<String> {
    serde_json::to_string(value).map_err(|error| decode_error(column_name, error))
}

/// Decodes the `state` discriminator, tolerating an unknown value as `Pending`.
fn decode_state(raw: &str) -> DreamState {
    DreamState::parse(raw).unwrap_or_else(|| {
        tracing::warn!(
            target: "openjiuwen_memory::dreaming::sqlite",
            state = %raw,
            "unknown dream job state in storage; treating the job as pending"
        );
        DreamState::Pending
    })
}

/// Decodes the `trigger` discriminator, tolerating an unknown value as `Manual`.
fn decode_trigger(raw: &str) -> DreamTrigger {
    DreamTrigger::parse(raw).unwrap_or_else(|| {
        tracing::warn!(
            target: "openjiuwen_memory::dreaming::sqlite",
            trigger = %raw,
            "unknown dream job trigger in storage; treating it as manual"
        );
        DreamTrigger::Manual
    })
}

/// Decodes one `dream_jobs` row (selected with [`COLUMNS`]) into a [`DreamJob`].
fn row_to_job(row: &SqliteRow) -> Result<DreamJob> {
    let kind_raw: String = column(row, "kind")?;
    let state_raw: String = column(row, "state")?;
    let trigger_raw: String = column(row, "trigger")?;
    let created_at: String = column(row, "created_at")?;
    let updated_at: String = column(row, "updated_at")?;
    let started_at: Option<String> = column(row, "started_at")?;
    let finished_at: Option<String> = column(row, "finished_at")?;
    let metrics_json: String = column(row, "metrics_json")?;
    let payload_json: String = column(row, "payload_json")?;
    let metadata_json: String = column(row, "metadata_json")?;
    let attempts: i64 = column(row, "attempts")?;
    let max_attempts: i64 = column(row, "max_attempts")?;
    let priority: i64 = column(row, "priority")?;

    Ok(DreamJob {
        id: column(row, "id")?,
        scope: MemoryScope {
            user_id: column(row, "user_id")?,
            scope_id: column(row, "scope_id")?,
            session_id: column(row, "session_id")?,
            agent_id: column(row, "agent_id")?,
        },
        kind: DreamKind::parse(&kind_raw),
        state: decode_state(&state_raw),
        trigger: decode_trigger(&trigger_raw),
        priority: i32::try_from(priority).unwrap_or(0),
        created_at: decode_time("created_at", &created_at)?,
        updated_at: decode_time("updated_at", &updated_at)?,
        started_at: started_at
            .as_deref()
            .map(|raw| decode_time("started_at", raw))
            .transpose()?,
        finished_at: finished_at
            .as_deref()
            .map(|raw| decode_time("finished_at", raw))
            .transpose()?,
        attempts: u32::try_from(attempts).unwrap_or(0),
        max_attempts: u32::try_from(max_attempts).unwrap_or(0),
        last_error: column(row, "last_error")?,
        metrics: decode_json::<DreamMetrics>("metrics_json", &metrics_json)?,
        payload: decode_json("payload_json", &payload_json)?,
        metadata: decode_json("metadata_json", &metadata_json)?,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use chrono::Duration;
    use tempfile::TempDir;

    /// A fixed clock base; every test timestamp is an offset from it.
    fn ts(secs: i64) -> DateTime<Utc> {
        DateTime::from_timestamp(1_700_000_000, 0).unwrap() + Duration::seconds(secs)
    }

    fn job(id: &str, priority: i32, created_at: i64) -> DreamJob {
        DreamJob::new(
            MemoryScope::new("alice", "ws"),
            DreamKind::Consolidation,
            DreamTrigger::Manual,
            ts(created_at),
        )
        .with_id(id)
        .with_priority(priority)
    }

    fn database_url(dir: &TempDir) -> String {
        format!("sqlite:{}/dreams.db", dir.path().display())
    }

    async fn open(dir: &TempDir) -> SqliteDreamRepository {
        SqliteDreamRepository::connect(&database_url(dir))
            .await
            .unwrap()
    }

    async fn claimed(repo: &SqliteDreamRepository, now: DateTime<Utc>) -> DreamJob {
        repo.claim_next(now)
            .await
            .unwrap()
            .expect("a claimable job")
    }

    /// Enqueues a job that is already terminal at `finished_at`.
    fn finished_job(id: &str, state: DreamState, finished_at: i64) -> DreamJob {
        DreamJob {
            state,
            attempts: 1,
            started_at: Some(ts(finished_at)),
            finished_at: Some(ts(finished_at)),
            updated_at: ts(finished_at),
            ..job(id, 0, finished_at)
        }
    }

    #[tokio::test]
    async fn round_trip_survives_a_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let metrics = DreamMetrics {
            records_scanned: 7,
            records_merged: 3,
            records_deleted: 2,
            records_created: 1,
            tokens_saved: 128,
            duration_ms: 42,
        };
        {
            let repo = open(&dir).await;
            repo.enqueue(job("done", 0, 0)).await.unwrap();
            let mut retried = job("retried", 0, 1).with_max_attempts(1);
            retried.payload =
                serde_json::Map::from_iter([("pass".to_owned(), serde_json::json!("all"))]);
            retried.metadata =
                serde_json::Map::from_iter([("owner".to_owned(), serde_json::json!("alice"))]);
            repo.enqueue(retried).await.unwrap();

            claimed(&repo, ts(10)).await;
            repo.complete("done", DreamOutcome::succeeded("done", metrics))
                .await
                .unwrap();

            let second = claimed(&repo, ts(11)).await;
            assert_eq!(second.id, "retried");
            repo.fail("retried", "boom").await.unwrap();
        }

        let reopened = open(&dir).await;
        let done = reopened.get("done").await.unwrap().expect("done");
        assert_eq!(done.state, DreamState::Succeeded);
        assert_eq!(done.metrics, metrics);
        assert_eq!(done.attempts, 1);
        assert!(done.finished_at.is_some());
        assert!(done.last_error.is_none());

        let retried = reopened.get("retried").await.unwrap().expect("retried");
        assert_eq!(retried.state, DreamState::Failed);
        assert_eq!(retried.attempts, 1);
        assert_eq!(retried.last_error.as_deref(), Some("boom"));
        assert!(retried.finished_at.is_some());
        assert_eq!(retried.payload.get("pass"), Some(&serde_json::json!("all")));
        assert_eq!(
            retried.metadata.get("owner"),
            Some(&serde_json::json!("alice"))
        );
    }

    #[tokio::test]
    async fn claim_orders_by_priority_then_creation_then_id() {
        let dir = tempfile::tempdir().unwrap();
        let repo = open(&dir).await;
        repo.enqueue(job("low", 0, 1)).await.unwrap();
        repo.enqueue(job("high", 5, 5)).await.unwrap();
        repo.enqueue(job("high-older", 5, 2)).await.unwrap();
        repo.enqueue(job("high-tie-b", 5, 2)).await.unwrap();

        let mut order = Vec::new();
        for step in 0..4 {
            order.push(claimed(&repo, ts(10 + step)).await.id);
        }
        assert_eq!(order, vec!["high-older", "high-tie-b", "high", "low"]);
    }

    #[tokio::test]
    async fn concurrent_claims_yield_exactly_one_winner() {
        let dir = tempfile::tempdir().unwrap();
        let repo = open(&dir).await;
        repo.enqueue(job("only", 0, 0)).await.unwrap();

        let left = repo.clone();
        let right = repo.clone();
        let now = ts(10);
        let (first, second) = tokio::join!(left.claim_next(now), right.claim_next(now));
        let winners = [first.unwrap(), second.unwrap()];
        assert_eq!(winners.iter().filter(|winner| winner.is_some()).count(), 1);
        let winner = winners.into_iter().flatten().next().unwrap();
        assert_eq!(winner.id, "only");
        assert_eq!(winner.attempts, 1);

        // The claim stamped `started_at = now` as the retry marker: the same clock is
        // not eligible, a later one is.
        sqlx::query(
            "UPDATE dream_jobs SET state = 'pending', attempts = 0, started_at = ?2, \
             finished_at = NULL WHERE id = ?1",
        )
        .bind("only")
        .bind(format_time(now))
        .execute(repo.pool())
        .await
        .unwrap();
        assert!(repo.claim_next(now).await.unwrap().is_none());
        assert!(repo
            .claim_next(now + Duration::seconds(1))
            .await
            .unwrap()
            .is_some());
    }

    #[tokio::test]
    async fn delete_finished_and_purge_only_remove_old_terminal_jobs() {
        let dir = tempfile::tempdir().unwrap();
        let repo = open(&dir).await;
        repo.enqueue(finished_job("old", DreamState::Succeeded, 10))
            .await
            .unwrap();
        repo.enqueue(finished_job("recent", DreamState::Failed, 900))
            .await
            .unwrap();
        repo.enqueue(finished_job("cancelled", DreamState::Cancelled, 5))
            .await
            .unwrap();
        repo.enqueue(job("pending", 0, 0)).await.unwrap();

        assert_eq!(
            repo.delete_finished(ts(5)).await.unwrap(),
            0,
            "strictly earlier"
        );
        assert_eq!(repo.delete_finished(ts(11)).await.unwrap(), 2);
        assert_eq!(repo.delete_finished(ts(1000)).await.unwrap(), 1);
        assert!(repo.get("pending").await.unwrap().is_some());
        assert!(repo.get("old").await.unwrap().is_none());

        // `purge` is the trait default over `delete_finished`.
        repo.enqueue(finished_job("purged", DreamState::Skipped, 20))
            .await
            .unwrap();
        let now = ts(30);
        assert_eq!(repo.purge(now, Duration::days(7)).await.unwrap(), 0);
        assert_eq!(
            repo.purge(now, Duration::seconds(5)).await.unwrap(),
            1,
            "skipped row finished at t=20 is older than the 5 s cutoff"
        );
        assert!(repo.get("purged").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn json_columns_round_trip_exactly() {
        let dir = tempfile::tempdir().unwrap();
        let repo = open(&dir).await;
        let payload = serde_json::Map::from_iter([
            ("nested".to_owned(), serde_json::json!({"list": [1, 2, 3]})),
            ("null".to_owned(), serde_json::Value::Null),
            ("unicode".to_owned(), serde_json::json!("héllo ☃")),
        ]);
        let metadata = serde_json::Map::from_iter([("tag".to_owned(), serde_json::json!(true))]);
        repo.enqueue(
            job("json", 0, 0)
                .with_payload(payload.clone())
                .with_metadata(metadata.clone()),
        )
        .await
        .unwrap();
        claimed(&repo, ts(10)).await;
        let metrics = DreamMetrics {
            records_scanned: 11,
            records_merged: 4,
            records_deleted: 1,
            records_created: 9,
            tokens_saved: 4096,
            duration_ms: 7,
        };
        repo.complete("json", DreamOutcome::succeeded("json", metrics))
            .await
            .unwrap();

        let stored = repo.get("json").await.unwrap().expect("stored");
        assert_eq!(stored.payload, payload);
        assert_eq!(stored.metadata, metadata);
        assert_eq!(stored.metrics, metrics);
        assert!(!stored.metrics.is_empty());
    }

    #[tokio::test]
    async fn count_by_state_reports_all_six_states() {
        let dir = tempfile::tempdir().unwrap();
        let repo = open(&dir).await;
        repo.enqueue(job("pending", 0, 0)).await.unwrap();
        repo.enqueue(job("running", 0, 1)).await.unwrap();
        repo.enqueue(finished_job("succeeded", DreamState::Succeeded, 2))
            .await
            .unwrap();
        repo.enqueue(finished_job("failed", DreamState::Failed, 3))
            .await
            .unwrap();
        claimed(&repo, ts(10)).await;

        let counts = repo.count_by_state().await.unwrap();
        assert_eq!(counts.len(), DreamState::ALL.len());
        assert_eq!(
            counts,
            BTreeMap::from_iter([
                (DreamState::Pending, 1),
                (DreamState::Running, 1),
                (DreamState::Succeeded, 1),
                (DreamState::Failed, 1),
                (DreamState::Skipped, 0),
                (DreamState::Cancelled, 0),
            ])
        );
    }

    #[tokio::test]
    async fn enqueue_complete_and_fail_errors_match_the_contract() {
        let dir = tempfile::tempdir().unwrap();
        let repo = open(&dir).await;
        repo.enqueue(job("dup", 0, 0)).await.unwrap();
        let duplicate = repo.enqueue(job("dup", 0, 0)).await.unwrap_err();
        assert!(matches!(duplicate, Error::Validation(_)), "{duplicate:?}");
        assert!(duplicate.to_string().contains("already exists"));

        let missing = repo
            .complete(
                "ghost",
                DreamOutcome::succeeded("ghost", DreamMetrics::default()),
            )
            .await
            .unwrap_err();
        assert!(matches!(missing, Error::NotFound(_)), "{missing:?}");

        let pending = repo
            .complete(
                "dup",
                DreamOutcome::succeeded("dup", DreamMetrics::default()),
            )
            .await
            .unwrap_err();
        assert!(matches!(pending, Error::Validation(_)), "{pending:?}");
        assert!(pending.to_string().contains("must be `running`"));

        let mismatch = repo
            .complete(
                "dup",
                DreamOutcome::succeeded("other", DreamMetrics::default()),
            )
            .await
            .unwrap_err();
        assert!(matches!(mismatch, Error::Validation(_)), "{mismatch:?}");

        let fail_pending = repo.fail("dup", "nope").await.unwrap_err();
        assert!(
            matches!(fail_pending, Error::Validation(_)),
            "{fail_pending:?}"
        );
        let fail_missing = repo.fail("ghost", "nope").await.unwrap_err();
        assert!(
            matches!(fail_missing, Error::NotFound(_)),
            "{fail_missing:?}"
        );
    }

    #[tokio::test]
    async fn list_filters_by_state_kind_and_scope_with_a_limit() {
        let dir = tempfile::tempdir().unwrap();
        let repo = open(&dir).await;
        repo.enqueue(job("low", 0, 1)).await.unwrap();
        repo.enqueue(job("high", 5, 5)).await.unwrap();
        let mut dedupe = job("dedupe", 0, 2);
        dedupe.kind = DreamKind::Dedupe;
        repo.enqueue(dedupe).await.unwrap();
        let mut other_scope = job("bob", 0, 3);
        other_scope.scope = MemoryScope::new("bob", "ws");
        repo.enqueue(other_scope).await.unwrap();

        let all = repo.list(&DreamFilter::new()).await.unwrap();
        assert_eq!(
            all.iter().map(|job| job.id.as_str()).collect::<Vec<_>>(),
            vec!["high", "low", "dedupe", "bob"]
        );

        let dedupe = repo
            .list(&DreamFilter::new().with_kind(DreamKind::Dedupe))
            .await
            .unwrap();
        assert_eq!(dedupe.len(), 1);
        assert_eq!(dedupe[0].id, "dedupe");

        let scoped = repo
            .list(&DreamFilter::new().for_scope(&MemoryScope::new("bob", "ws")))
            .await
            .unwrap();
        assert_eq!(scoped.len(), 1);
        assert_eq!(scoped[0].id, "bob");

        let limited = repo.list(&DreamFilter::new().with_limit(2)).await.unwrap();
        assert_eq!(
            limited
                .iter()
                .map(|job| job.id.as_str())
                .collect::<Vec<_>>(),
            vec!["high", "low"]
        );

        claimed(&repo, ts(10)).await;
        let running = repo
            .list(&DreamFilter::new().with_state(DreamState::Running))
            .await
            .unwrap();
        assert_eq!(running.len(), 1);
        assert_eq!(running[0].id, "high");

        let bounded = repo
            .list(&DreamFilter {
                created_after: Some(ts(1)),
                created_before: Some(ts(4)),
                ..DreamFilter::new()
            })
            .await
            .unwrap();
        assert_eq!(
            bounded
                .iter()
                .map(|job| job.id.as_str())
                .collect::<Vec<_>>(),
            vec!["dedupe", "bob"]
        );
    }
}
