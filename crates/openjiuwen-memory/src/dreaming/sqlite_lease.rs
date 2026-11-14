//! SQLite persistence for dream-job leases.
//!
//! This is a child module of [`sqlite`](super), so it shares the parent's canonical
//! timestamp format, column decoding, and claim SQL. A lease lives in its own
//! `dream_job_leases` table (one row per job) rather than as extra `dream_jobs` columns:
//! the job table's shape and the claim's `RETURNING` decoding stay untouched, and a lease
//! row is deleted together with its job.
//!
//! The takeover path — reclaiming a `Running` job whose lease expired — is one
//! `UPDATE ... RETURNING` statement inside the same transaction that replaces the lease,
//! so two workers cannot both reclaim the same crashed job.
use chrono::{DateTime, Utc};
use sqlx::{Sqlite, Transaction};

use openjiuwen_common::error::{Error, Result};

use crate::dreaming::lease::{DreamLease, LeaseLostReason, LeaseRequest, LeaseStatus, LeasedJob};

use super::{
    claim_query, column, decode_time, format_time, row_to_job, SqliteDreamRepository, COLUMNS,
};

/// Lease table DDL, applied idempotently by the parent's constructors.
pub(super) const SCHEMA_LEASES: &str = "
CREATE TABLE IF NOT EXISTS dream_job_leases (
    job_id TEXT PRIMARY KEY NOT NULL,
    owner TEXT NOT NULL,
    acquired_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
)";

/// Index used by the takeover scan (expired leases on running jobs).
pub(super) const LEASE_INDEX: &str = "CREATE INDEX IF NOT EXISTS dream_job_leases_expiry_idx \
     ON dream_job_leases (expires_at)";

/// Removes lease rows whose job no longer exists; run after job deletions.
pub(super) const DELETE_ORPHANS: &str =
    "DELETE FROM dream_job_leases WHERE job_id NOT IN (SELECT id FROM dream_jobs)";

/// The lease column list every read uses.
const LEASE_COLUMNS: &str = "job_id, owner, acquired_at, expires_at";

/// Inserts or replaces the lease row for one job.
const UPSERT_LEASE: &str = "
INSERT INTO dream_job_leases (job_id, owner, acquired_at, expires_at)
VALUES (?1, ?2, ?3, ?4)
ON CONFLICT(job_id) DO UPDATE SET
    owner = excluded.owner,
    acquired_at = excluded.acquired_at,
    expires_at = excluded.expires_at";

/// Reclaims the oldest `Running` job whose lease expired at or before `?1`.
///
/// The `JOIN` (not a left join) is deliberate: a `Running` job with no lease row is
/// never taken over, because it may have been claimed without leasing.
fn takeover_query() -> String {
    format!(
        "UPDATE dream_jobs
   SET state = 'running', attempts = attempts + 1, started_at = ?1, updated_at = ?1
 WHERE id = (
        SELECT j.id FROM dream_jobs j
          JOIN dream_job_leases l ON l.job_id = j.id
         WHERE j.state = 'running' AND j.attempts < j.max_attempts
           AND l.expires_at <= ?1
         ORDER BY j.priority DESC, j.created_at ASC, j.id ASC
         LIMIT 1
       )
   AND state = 'running'
RETURNING {COLUMNS}"
    )
}

impl SqliteDreamRepository {
    /// Claims a pending job or reclaims an expired leased job, then stores the lease.
    pub(super) async fn claim_leased(
        &self,
        now: DateTime<Utc>,
        request: &LeaseRequest,
    ) -> Result<Option<LeasedJob>> {
        request.validate()?;
        let mut transaction = self.pool.begin().await.map_err(Error::external)?;
        let claimed = self.claim_and_lease(&mut transaction, now, request).await?;
        if let Some(leased) = &claimed {
            transaction.commit().await.map_err(Error::external)?;
            tracing::debug!(
                target: "openjiuwen_memory::dreaming::sqlite",
                job_id = %leased.job.id,
                owner = %leased.lease.owner,
                attempts = leased.job.attempts,
                "claimed dream job under a lease"
            );
            return Ok(claimed);
        }
        transaction.commit().await.map_err(Error::external)?;
        Ok(None)
    }

    /// Runs the pending claim and, on a miss, the expired-lease takeover.
    async fn claim_and_lease(
        &self,
        transaction: &mut Transaction<'_, Sqlite>,
        now: DateTime<Utc>,
        request: &LeaseRequest,
    ) -> Result<Option<LeasedJob>> {
        for query in [claim_query(), takeover_query()] {
            let row = sqlx::query(&query)
                .bind(format_time(now))
                .fetch_optional(&mut **transaction)
                .await
                .map_err(Error::external)?;
            let Some(row) = row else { continue };
            let job = row_to_job(&row)?;
            self.store_lease(transaction, &job.id, request, now).await?;
            let lease = DreamLease::new(job.id.clone(), request.owner.clone(), now, request.ttl);
            return Ok(Some(LeasedJob { job, lease }));
        }
        Ok(None)
    }

    /// Upserts the lease for `job_id` from `request`, stamped with `now`.
    async fn store_lease(
        &self,
        transaction: &mut Transaction<'_, Sqlite>,
        job_id: &str,
        request: &LeaseRequest,
        now: DateTime<Utc>,
    ) -> Result<()> {
        let expires_at = now
            .checked_add_signed(request.ttl)
            .unwrap_or(DateTime::<Utc>::MAX_UTC);
        sqlx::query(UPSERT_LEASE)
            .bind(job_id)
            .bind(&request.owner)
            .bind(format_time(now))
            .bind(format_time(expires_at))
            .execute(&mut **transaction)
            .await
            .map_err(Error::external)?;
        Ok(())
    }

    /// Extends a live lease owned by `owner`.
    pub(super) async fn renew_leased(
        &self,
        job_id: &str,
        owner: &str,
        now: DateTime<Utc>,
        ttl: chrono::Duration,
    ) -> Result<LeaseStatus> {
        if ttl <= chrono::Duration::zero() {
            return Err(Error::validation(format!(
                "dream lease `ttl` must be positive (got {}s)",
                ttl.num_seconds()
            )));
        }
        let expires_at = now
            .checked_add_signed(ttl)
            .unwrap_or(DateTime::<Utc>::MAX_UTC);
        let mut transaction = self.pool.begin().await.map_err(Error::external)?;
        let updated = sqlx::query(
            "UPDATE dream_job_leases
                SET expires_at = ?3
              WHERE job_id = ?1 AND owner = ?2 AND expires_at > ?4
                AND EXISTS (
                    SELECT 1 FROM dream_jobs WHERE id = ?1 AND state = 'running'
                )",
        )
        .bind(job_id)
        .bind(owner)
        .bind(format_time(expires_at))
        .bind(format_time(now))
        .execute(&mut *transaction)
        .await
        .map_err(Error::external)?;
        if updated.rows_affected() == 1 {
            let lease = self.read_lease(&mut transaction, job_id).await?;
            transaction.commit().await.map_err(Error::external)?;
            return match lease {
                Some(lease) => Ok(LeaseStatus::Renewed(lease)),
                None => Ok(LeaseStatus::Lost(LeaseLostReason::Missing)),
            };
        }
        let reason = self.diagnose_loss(&mut transaction, job_id, owner).await?;
        transaction.rollback().await.map_err(Error::external)?;
        Ok(LeaseStatus::Lost(reason))
    }

    /// Explains a failed renewal by reading the lease row and the job state.
    async fn diagnose_loss(
        &self,
        transaction: &mut Transaction<'_, Sqlite>,
        job_id: &str,
        owner: &str,
    ) -> Result<LeaseLostReason> {
        let lease = self.read_lease(transaction, job_id).await?;
        let state = sqlx::query("SELECT state FROM dream_jobs WHERE id = ?1")
            .bind(job_id)
            .fetch_optional(&mut **transaction)
            .await
            .map_err(Error::external)?
            .map(|row| column::<String>(&row, "state"))
            .transpose()?;
        Ok(match (lease, state) {
            (None, _) | (Some(_), None) => LeaseLostReason::Missing,
            (Some(_), Some(state)) if state != "running" => LeaseLostReason::NotRunning,
            (Some(lease), Some(_)) if lease.owner != owner => LeaseLostReason::OwnerMismatch,
            (Some(_), Some(_)) => LeaseLostReason::Expired,
        })
    }

    /// Deletes the lease when `owner` still holds it.
    pub(super) async fn release_leased(&self, job_id: &str, owner: &str) -> Result<bool> {
        let result = sqlx::query("DELETE FROM dream_job_leases WHERE job_id = ?1 AND owner = ?2")
            .bind(job_id)
            .bind(owner)
            .execute(&self.pool)
            .await
            .map_err(Error::external)?;
        Ok(result.rows_affected() > 0)
    }

    /// Reads the lease row for `job_id`.
    pub(super) async fn stored_lease(&self, job_id: &str) -> Result<Option<DreamLease>> {
        let row = sqlx::query(&format!(
            "SELECT {LEASE_COLUMNS} FROM dream_job_leases WHERE job_id = ?1"
        ))
        .bind(job_id)
        .fetch_optional(&self.pool)
        .await
        .map_err(Error::external)?;
        row.as_ref().map(decode_lease).transpose()
    }

    /// Deletes one job and its lease; `false` when the job did not exist.
    pub(super) async fn delete_leased_job(&self, job_id: &str) -> Result<bool> {
        let mut transaction = self.pool.begin().await.map_err(Error::external)?;
        sqlx::query("DELETE FROM dream_job_leases WHERE job_id = ?1")
            .bind(job_id)
            .execute(&mut *transaction)
            .await
            .map_err(Error::external)?;
        let deleted = sqlx::query("DELETE FROM dream_jobs WHERE id = ?1")
            .bind(job_id)
            .execute(&mut *transaction)
            .await
            .map_err(Error::external)?;
        transaction.commit().await.map_err(Error::external)?;
        Ok(deleted.rows_affected() > 0)
    }

    /// Reads the lease row inside a transaction.
    async fn read_lease(
        &self,
        transaction: &mut Transaction<'_, Sqlite>,
        job_id: &str,
    ) -> Result<Option<DreamLease>> {
        let row = sqlx::query(&format!(
            "SELECT {LEASE_COLUMNS} FROM dream_job_leases WHERE job_id = ?1"
        ))
        .bind(job_id)
        .fetch_optional(&mut **transaction)
        .await
        .map_err(Error::external)?;
        row.as_ref().map(decode_lease).transpose()
    }
}

/// Decodes one `dream_job_leases` row.
fn decode_lease(row: &sqlx::sqlite::SqliteRow) -> Result<DreamLease> {
    let acquired_at: String = column(row, "acquired_at")?;
    let expires_at: String = column(row, "expires_at")?;
    Ok(DreamLease {
        job_id: column(row, "job_id")?,
        owner: column(row, "owner")?,
        acquired_at: decode_time("acquired_at", &acquired_at)?,
        expires_at: decode_time("expires_at", &expires_at)?,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::dreaming::{
        DreamJob, DreamKind, DreamOutcome, DreamRepository, DreamState, DreamTrigger,
    };
    use crate::record::MemoryScope;
    use chrono::Duration;
    use tempfile::TempDir;

    fn ts(offset_secs: i64) -> DateTime<Utc> {
        DateTime::from_timestamp(1_700_000_000 + offset_secs, 0).unwrap()
    }

    fn job(id: &str, created: DateTime<Utc>) -> DreamJob {
        DreamJob::new(
            MemoryScope::new("alice", "ws"),
            DreamKind::Consolidation,
            DreamTrigger::Manual,
            created,
        )
        .with_id(id)
    }

    async fn repo(temp: &TempDir) -> SqliteDreamRepository {
        let url = format!("sqlite:{}/dreams.db", temp.path().display());
        SqliteDreamRepository::connect(&url).await.unwrap()
    }

    fn request(owner: &str) -> LeaseRequest {
        LeaseRequest::new(owner, Duration::seconds(30))
    }

    #[tokio::test]
    async fn leased_claim_creates_a_lease_and_blocks_a_second_worker() {
        let temp = TempDir::new().unwrap();
        let repo = repo(&temp).await;
        repo.enqueue(job("job-1", ts(0))).await.unwrap();

        let first = repo
            .claim_next_leased(ts(10), &request("worker-a"))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(first.job.state, DreamState::Running);
        assert_eq!(first.job.attempts, 1);
        assert_eq!(first.lease.owner, "worker-a");
        assert_eq!(first.lease.expires_at, ts(40));
        assert_eq!(repo.lease("job-1").await.unwrap(), Some(first.lease));

        let second = repo
            .claim_next_leased(ts(11), &request("worker-b"))
            .await
            .unwrap();
        assert!(second.is_none(), "a live lease is not stealable");
    }

    #[tokio::test]
    async fn expired_lease_is_reclaimed_and_attempts_advance() {
        let temp = TempDir::new().unwrap();
        let repo = repo(&temp).await;
        repo.enqueue(job("job-1", ts(0))).await.unwrap();
        repo.claim_next_leased(ts(10), &request("worker-a"))
            .await
            .unwrap()
            .unwrap();

        assert!(repo
            .claim_next_leased(ts(39), &request("worker-b"))
            .await
            .unwrap()
            .is_none());
        let takeover = repo
            .claim_next_leased(ts(40), &request("worker-b"))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(takeover.job.attempts, 2);
        assert_eq!(takeover.job.started_at, Some(ts(40)));
        assert_eq!(takeover.lease.owner, "worker-b");
        assert_eq!(repo.lease("job-1").await.unwrap(), Some(takeover.lease));

        let status = repo
            .renew_lease("job-1", "worker-a", ts(41), Duration::seconds(30))
            .await
            .unwrap();
        assert_eq!(status, LeaseStatus::Lost(LeaseLostReason::OwnerMismatch));
    }

    #[tokio::test]
    async fn renew_extends_release_removes_and_wrong_owner_loses() {
        let temp = TempDir::new().unwrap();
        let repo = repo(&temp).await;
        repo.enqueue(job("job-1", ts(0))).await.unwrap();
        repo.claim_next_leased(ts(10), &request("worker-a"))
            .await
            .unwrap()
            .unwrap();

        let renewed = repo
            .renew_lease("job-1", "worker-a", ts(20), Duration::seconds(60))
            .await
            .unwrap();
        let lease = renewed.lease().cloned().unwrap();
        assert_eq!(lease.expires_at, ts(80));
        assert_eq!(lease.acquired_at, ts(10));

        let wrong = repo
            .renew_lease("job-1", "worker-b", ts(21), Duration::seconds(60))
            .await
            .unwrap();
        assert_eq!(wrong, LeaseStatus::Lost(LeaseLostReason::OwnerMismatch));
        assert!(!repo.release_lease("job-1", "worker-b").await.unwrap());
        assert!(repo.release_lease("job-1", "worker-a").await.unwrap());
        assert_eq!(repo.lease("job-1").await.unwrap(), None);
        assert!(!repo.release_lease("job-1", "worker-a").await.unwrap());
    }

    #[tokio::test]
    async fn renew_after_expiry_or_completion_is_lost() {
        let temp = TempDir::new().unwrap();
        let repo = repo(&temp).await;
        repo.enqueue(job("job-1", ts(0))).await.unwrap();
        repo.claim_next_leased(ts(10), &request("worker-a"))
            .await
            .unwrap()
            .unwrap();
        let expired = repo
            .renew_lease("job-1", "worker-a", ts(40), Duration::seconds(30))
            .await
            .unwrap();
        assert_eq!(expired, LeaseStatus::Lost(LeaseLostReason::Expired));

        repo.complete(
            "job-1",
            DreamOutcome::succeeded("job-1", crate::dreaming::DreamMetrics::default()),
        )
        .await
        .unwrap();
        let finished = repo
            .renew_lease("job-1", "worker-a", ts(41), Duration::seconds(30))
            .await
            .unwrap();
        assert_eq!(finished, LeaseStatus::Lost(LeaseLostReason::NotRunning));
        assert_eq!(
            repo.renew_lease("missing", "worker-a", ts(41), Duration::seconds(30))
                .await
                .unwrap(),
            LeaseStatus::Lost(LeaseLostReason::Missing)
        );
    }

    #[tokio::test]
    async fn leases_survive_a_reopen_and_release_with_the_job() {
        let temp = TempDir::new().unwrap();
        let url = format!("sqlite:{}/dreams.db", temp.path().display());
        {
            let repo = SqliteDreamRepository::connect(&url).await.unwrap();
            repo.enqueue(job("job-1", ts(0))).await.unwrap();
            repo.claim_next_leased(ts(10), &request("worker-a"))
                .await
                .unwrap()
                .unwrap();
        }
        let reopened = SqliteDreamRepository::connect(&url).await.unwrap();
        let lease = reopened.lease("job-1").await.unwrap().unwrap();
        assert_eq!(lease.owner, "worker-a");
        assert_eq!(lease.expires_at, ts(40));

        assert!(reopened.delete_job("job-1").await.unwrap());
        assert!(!reopened.delete_job("job-1").await.unwrap());
        assert_eq!(reopened.lease("job-1").await.unwrap(), None);
        assert_eq!(reopened.get("job-1").await.unwrap(), None);
    }

    #[tokio::test]
    async fn deleting_finished_jobs_drops_their_leases() {
        let temp = TempDir::new().unwrap();
        let repo = repo(&temp).await;
        repo.enqueue(job("job-1", ts(0))).await.unwrap();
        repo.claim_next_leased(ts(10), &request("worker-a"))
            .await
            .unwrap()
            .unwrap();
        repo.complete(
            "job-1",
            DreamOutcome::succeeded("job-1", crate::dreaming::DreamMetrics::default()),
        )
        .await
        .unwrap();
        let cutoff = Utc::now() + Duration::hours(1);
        assert!(repo.delete_finished(cutoff).await.unwrap() >= 1);
        assert_eq!(repo.lease("job-1").await.unwrap(), None);
    }

    #[tokio::test]
    async fn a_running_job_without_a_lease_is_never_stolen() {
        let temp = TempDir::new().unwrap();
        let repo = repo(&temp).await;
        repo.enqueue(job("job-1", ts(0))).await.unwrap();
        repo.claim_next(ts(10)).await.unwrap().unwrap();
        assert!(repo
            .claim_next_leased(ts(10_000), &request("worker-b"))
            .await
            .unwrap()
            .is_none());
        assert_eq!(repo.lease("job-1").await.unwrap(), None);
    }
}
