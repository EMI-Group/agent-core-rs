//! Leases: cross-process ownership of a claimed [`DreamJob`](crate::dreaming::DreamJob).
//!
//! A plain [`DreamRepository::claim_next`](crate::dreaming::DreamRepository::claim_next)
//! moves a pending job to `Running`, but nothing stops a second process from acting on a
//! job whose worker crashed before recording the transition. A lease closes that gap:
//! the claim records an owner plus an expiry, and only that owner may renew, complete, or
//! fail the job while the lease is live.
//!
//! The rules:
//! - a lease is created by
//!   [`DreamRepository::claim_next_leased`](crate::dreaming::DreamRepository::claim_next_leased)
//!   with the claim's `now` and the request's TTL;
//! - a lease is valid while `now < expires_at`, and is expired at or after `expires_at`;
//! - a `Running` job whose lease expired is reclaimable by another worker (the takeover
//!   path of `claim_next_leased`); a `Running` job *without* a lease record is never
//!   stolen, because it may have been claimed without leasing;
//! - the owner detects a stolen or expired lease through [`DreamLeaseGuard`], which
//!   reports [`LeaseStatus::Lost`] and latches [`DreamLeaseGuard::is_lost`] so a worker
//!   stops instead of recording a transition on somebody else's job.
//!
//! Lease timestamps are real timestamps (they describe other processes), while the
//! scheduler keeps its injectable `now` for sweeps; tests drive both explicitly.
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration as StdDuration;

use chrono::{DateTime, Duration as ChronoDuration, Utc};
use openjiuwen_common::error::{Error, Result};
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};

use super::model::{validate_non_empty, DreamJob};
use super::repository::DreamRepository;

/// Ownership of one claimed job for a bounded window.
///
/// `job_id` and `owner` are non-empty, NUL-free strings; `expires_at` is strictly after
/// `acquired_at`. A lease is data only: the repository persists it and the guard renews
/// it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct DreamLease {
    /// The leased job.
    pub job_id: String,
    /// Identifier of the worker that holds the lease.
    pub owner: String,
    /// When the lease was (re)created.
    pub acquired_at: DateTime<Utc>,
    /// When the lease stops being valid.
    pub expires_at: DateTime<Utc>,
}

impl DreamLease {
    /// Builds a lease for `job_id` owned by `owner`, valid until `now + ttl`.
    ///
    /// A TTL that chrono cannot add saturates at [`DateTime::<Utc>::MAX_UTC`] instead
    /// of wrapping; [`Self::validate`] still rejects a non-positive TTL because the
    /// resulting expiry would not follow the acquisition time.
    #[must_use]
    pub fn new(
        job_id: impl Into<String>,
        owner: impl Into<String>,
        now: DateTime<Utc>,
        ttl: ChronoDuration,
    ) -> Self {
        Self {
            job_id: job_id.into(),
            owner: owner.into(),
            acquired_at: now,
            expires_at: now
                .checked_add_signed(ttl)
                .unwrap_or(DateTime::<Utc>::MAX_UTC),
        }
    }

    /// Returns `true` when the lease is no longer valid at `now`.
    ///
    /// The boundary is exclusive: a lease expiring exactly at `now` is expired, which
    /// makes takeover deterministic when two workers share a clock.
    #[must_use]
    pub fn is_expired(&self, now: DateTime<Utc>) -> bool {
        now >= self.expires_at
    }

    /// Checks that the lease is non-empty, NUL-free, and has a positive window.
    pub fn validate(&self) -> Result<()> {
        validate_non_empty("dream lease `job_id`", &self.job_id)?;
        validate_non_empty("dream lease `owner`", &self.owner)?;
        if self.expires_at <= self.acquired_at {
            return Err(Error::validation(format!(
                "dream lease `expires_at` ({}) must follow `acquired_at` ({})",
                self.expires_at, self.acquired_at
            )));
        }
        Ok(())
    }
}

/// What a leased claim asks for: who owns the job and for how long.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct LeaseRequest {
    /// Identifier of the claiming worker.
    pub owner: String,
    /// Lease lifetime from the claim's clock.
    pub ttl: ChronoDuration,
}

impl LeaseRequest {
    /// Builds a request owned by `owner` with a `ttl` lifetime.
    #[must_use]
    pub fn new(owner: impl Into<String>, ttl: ChronoDuration) -> Self {
        Self {
            owner: owner.into(),
            ttl,
        }
    }

    /// Checks that `owner` is non-empty and NUL-free and `ttl` is positive.
    pub fn validate(&self) -> Result<()> {
        validate_non_empty("dream lease request `owner`", &self.owner)?;
        if self.ttl <= ChronoDuration::zero() {
            return Err(Error::validation(format!(
                "dream lease `ttl` must be positive (got {}s)",
                self.ttl.num_seconds()
            )));
        }
        Ok(())
    }
}

/// A leased claim: the claimed job plus the lease that backs the claim.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct LeasedJob {
    /// The job, already moved to `Running` and stamped with the claim clock.
    pub job: DreamJob,
    /// The lease the claimant now owns.
    pub lease: DreamLease,
}

/// Why a lease is no longer usable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LeaseLostReason {
    /// No lease row exists for the job (released, deleted, or never leased).
    Missing,
    /// The stored lease expired before the renewal.
    Expired,
    /// Another worker now owns the lease.
    OwnerMismatch,
    /// The job is no longer `Running` (completed or failed elsewhere).
    NotRunning,
}

impl LeaseLostReason {
    /// The `snake_case` name used on the wire and in logs.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Missing => "missing",
            Self::Expired => "expired",
            Self::OwnerMismatch => "owner_mismatch",
            Self::NotRunning => "not_running",
        }
    }
}

impl fmt::Display for LeaseLostReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The outcome of a renewal: still held, or lost with a reason.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LeaseStatus {
    /// The lease was extended and is now valid until the returned expiry.
    Renewed(DreamLease),
    /// The lease is gone; the worker must stop.
    Lost(LeaseLostReason),
}

impl LeaseStatus {
    /// Returns `true` for [`LeaseStatus::Renewed`].
    #[must_use]
    pub const fn is_renewed(&self) -> bool {
        matches!(self, Self::Renewed(_))
    }

    /// Returns `true` for [`LeaseStatus::Lost`].
    #[must_use]
    pub const fn is_lost(&self) -> bool {
        matches!(self, Self::Lost(_))
    }

    /// The renewed lease, when still held.
    #[must_use]
    pub fn lease(&self) -> Option<&DreamLease> {
        match self {
            Self::Renewed(lease) => Some(lease),
            Self::Lost(_) => None,
        }
    }

    /// The loss reason, when the lease is gone.
    #[must_use]
    pub const fn reason(&self) -> Option<LeaseLostReason> {
        match self {
            Self::Renewed(_) => None,
            Self::Lost(reason) => Some(*reason),
        }
    }
}

/// Everything a leased sweep needs: owner, TTL, and heartbeat cadence.
///
/// The heartbeat must be positive and shorter than the TTL; otherwise a worker could
/// lose its lease between two renewals even while making progress.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DreamLeaseOptions {
    /// Identifier of the worker running the sweep.
    pub owner: String,
    /// Lease lifetime handed to every claim and renewal.
    pub ttl: ChronoDuration,
    /// How often an in-flight attempt renews its lease.
    pub heartbeat: StdDuration,
}

impl DreamLeaseOptions {
    /// Builds options for `owner` with `ttl` and `heartbeat`.
    #[must_use]
    pub fn new(owner: impl Into<String>, ttl: ChronoDuration, heartbeat: StdDuration) -> Self {
        Self {
            owner: owner.into(),
            ttl,
            heartbeat,
        }
    }

    /// The lease-time half of these options.
    #[must_use]
    pub fn request(&self) -> LeaseRequest {
        LeaseRequest::new(self.owner.clone(), self.ttl)
    }

    /// Checks the owner, TTL, and heartbeat bounds.
    pub fn validate(&self) -> Result<()> {
        self.request().validate()?;
        if self.heartbeat.is_zero() {
            return Err(Error::validation(
                "dream lease `heartbeat` must be greater than zero",
            ));
        }
        let heartbeat = ChronoDuration::from_std(self.heartbeat)
            .map_err(|_| Error::validation("dream lease `heartbeat` is too large to represent"))?;
        if heartbeat >= self.ttl {
            return Err(Error::validation(format!(
                "dream lease `heartbeat` ({:?}) must be shorter than the lease `ttl` ({}s)",
                self.heartbeat,
                self.ttl.num_seconds()
            )));
        }
        Ok(())
    }
}

/// A worker's handle on one lease.
///
/// The guard latches loss: once [`Self::renew`] reports [`LeaseStatus::Lost`] (or
/// [`Self::ensure_valid`] observes the local expiry) every later check fails, so a
/// worker cannot accidentally resume after its job was taken over.
pub struct DreamLeaseGuard {
    /// Repository the lease lives in.
    repository: Arc<dyn DreamRepository>,
    /// The most recently known lease; renewal updates it.
    lease: RwLock<DreamLease>,
    /// Latched loss flag, readable without taking the lease lock.
    lost: AtomicBool,
    /// Why the lease was lost, set together with `lost`.
    lost_reason: Mutex<Option<LeaseLostReason>>,
}

impl fmt::Debug for DreamLeaseGuard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DreamLeaseGuard")
            .field("lease", &*self.lease.read())
            .field("lost", &self.is_lost())
            .field("lost_reason", &*self.lost_reason.lock())
            .finish()
    }
}

impl DreamLeaseGuard {
    /// Wraps `lease`, which was just claimed through `repository`.
    #[must_use]
    pub fn new(repository: Arc<dyn DreamRepository>, lease: DreamLease) -> Self {
        Self {
            repository,
            lease: RwLock::new(lease),
            lost: AtomicBool::new(false),
            lost_reason: Mutex::new(None),
        }
    }

    /// A snapshot of the most recently known lease.
    #[must_use]
    pub fn lease(&self) -> DreamLease {
        self.lease.read().clone()
    }

    /// The leased job id.
    #[must_use]
    pub fn job_id(&self) -> String {
        self.lease.read().job_id.clone()
    }

    /// The owning worker id.
    #[must_use]
    pub fn owner(&self) -> String {
        self.lease.read().owner.clone()
    }

    /// Returns `true` once the lease was reported lost.
    #[must_use]
    pub fn is_lost(&self) -> bool {
        self.lost.load(Ordering::Acquire)
    }

    /// The reason recorded when the lease was lost, if any.
    #[must_use]
    pub fn lost_reason(&self) -> Option<LeaseLostReason> {
        *self.lost_reason.lock()
    }

    /// Fails when the lease is lost locally (latched or expired at `now`).
    ///
    /// This is the cheap synchronous check for loops that are between awaits; it does
    /// not contact the repository, so a takeover is only observed once the process's
    /// own expiry passes or [`Self::renew`] runs.
    pub fn ensure_valid(&self, now: DateTime<Utc>) -> Result<()> {
        if self.is_lost() {
            return Err(self.lost_error());
        }
        if self.lease.read().is_expired(now) {
            self.mark_lost(LeaseLostReason::Expired);
            return Err(self.lost_error());
        }
        Ok(())
    }

    /// Renews the lease through the repository, extending it to `now + ttl`.
    ///
    /// A locally expired lease is reported as lost without a repository round-trip. A
    /// renewal that the repository reports as lost latches the guard; the caller should
    /// stop working on the job.
    pub async fn renew(&self, now: DateTime<Utc>, ttl: ChronoDuration) -> Result<LeaseStatus> {
        if self.is_lost() {
            return Ok(LeaseStatus::Lost(
                self.lost_reason().unwrap_or(LeaseLostReason::Missing),
            ));
        }
        if self.lease.read().is_expired(now) {
            self.mark_lost(LeaseLostReason::Expired);
            return Ok(LeaseStatus::Lost(LeaseLostReason::Expired));
        }
        let lease = self.lease.read().clone();
        let status = self
            .repository
            .renew_lease(&lease.job_id, &lease.owner, now, ttl)
            .await?;
        if let LeaseStatus::Renewed(renewed) = &status {
            *self.lease.write() = renewed.clone();
        } else if let LeaseStatus::Lost(reason) = &status {
            self.mark_lost(*reason);
        }
        Ok(status)
    }

    /// Deletes the lease when this worker still owns it.
    ///
    /// Returns `false` when the lease was missing or owned by somebody else; that is
    /// not an error, it only means there is nothing to release.
    pub async fn release(&self) -> Result<bool> {
        let lease = self.lease.read().clone();
        self.repository
            .release_lease(&lease.job_id, &lease.owner)
            .await
    }

    /// Latches the loss and logs it once.
    fn mark_lost(&self, reason: LeaseLostReason) {
        let first = !self.lost.swap(true, Ordering::AcqRel);
        let mut stored = self.lost_reason.lock();
        if stored.is_none() {
            *stored = Some(reason);
        }
        if first {
            let lease = self.lease.read();
            tracing::warn!(
                target: "openjiuwen_memory::dreaming::lease",
                job_id = %lease.job_id,
                owner = %lease.owner,
                reason = reason.as_str(),
                "dream lease lost; the worker must stop"
            );
        }
    }

    /// The error a lost lease surfaces to a worker.
    fn lost_error(&self) -> Error {
        let lease = self.lease.read();
        let reason = self.lost_reason().unwrap_or(LeaseLostReason::Missing);
        Error::memory(format!(
            "dream lease for job `{}` owned by `{}` was lost: {reason}",
            lease.job_id, lease.owner
        ))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::dreaming::{
        DreamJob, DreamKind, DreamRepository, DreamState, DreamTrigger, InMemoryDreamRepository,
    };
    use crate::record::MemoryScope;
    use std::sync::Arc;

    fn ts(offset_secs: i64) -> DateTime<Utc> {
        DateTime::from_timestamp(1_700_000_000 + offset_secs, 0).unwrap()
    }

    fn job(id: &str) -> DreamJob {
        DreamJob::new(
            MemoryScope::new("alice", "ws"),
            DreamKind::Consolidation,
            DreamTrigger::Manual,
            ts(0),
        )
        .with_id(id)
    }

    async fn claimed(repo: &Arc<InMemoryDreamRepository>, owner: &str) -> LeasedJob {
        repo.claim_next_leased(
            ts(10),
            &LeaseRequest::new(owner, ChronoDuration::seconds(30)),
        )
        .await
        .unwrap()
        .unwrap()
    }

    #[test]
    fn lease_window_and_expiry_boundary() {
        let lease = DreamLease::new("job-1", "worker-a", ts(0), ChronoDuration::seconds(30));
        assert_eq!(lease.expires_at, ts(30));
        assert!(!lease.is_expired(ts(29)));
        assert!(lease.is_expired(ts(30)));
        assert!(lease.is_expired(ts(31)));
        assert!(lease.validate().is_ok());
    }

    #[test]
    fn lease_validation_rejects_bad_owner_job_and_window() {
        let mut lease = DreamLease::new("job-1", "worker-a", ts(0), ChronoDuration::seconds(30));
        lease.job_id = String::new();
        assert!(lease.validate().unwrap_err().to_string().contains("job_id"));
        let mut lease = DreamLease::new("job-1", "worker-a", ts(0), ChronoDuration::seconds(30));
        lease.owner = "bad\0owner".to_owned();
        assert!(lease.validate().unwrap_err().to_string().contains("NUL"));
        let mut lease = DreamLease::new("job-1", "worker-a", ts(0), ChronoDuration::seconds(30));
        lease.expires_at = lease.acquired_at;
        assert!(lease
            .validate()
            .unwrap_err()
            .to_string()
            .contains("expires_at"));
    }

    #[test]
    fn request_and_options_validation() {
        assert!(LeaseRequest::new("worker-a", ChronoDuration::seconds(5))
            .validate()
            .is_ok());
        assert!(LeaseRequest::new("", ChronoDuration::seconds(5))
            .validate()
            .unwrap_err()
            .to_string()
            .contains("owner"));
        assert!(LeaseRequest::new("worker-a", ChronoDuration::zero())
            .validate()
            .unwrap_err()
            .to_string()
            .contains("ttl"));
        let ok = DreamLeaseOptions::new(
            "worker-a",
            ChronoDuration::seconds(30),
            StdDuration::from_secs(10),
        );
        assert_eq!(ok.request().owner, "worker-a");
        assert!(ok.validate().is_ok());
        let zero =
            DreamLeaseOptions::new("worker-a", ChronoDuration::seconds(30), StdDuration::ZERO);
        assert!(zero
            .validate()
            .unwrap_err()
            .to_string()
            .contains("heartbeat"));
        let long = DreamLeaseOptions::new(
            "worker-a",
            ChronoDuration::seconds(30),
            StdDuration::from_secs(30),
        );
        assert!(long.validate().unwrap_err().to_string().contains("shorter"));
    }

    #[tokio::test]
    async fn guard_renews_and_releases_a_live_lease() {
        let repo = Arc::new(InMemoryDreamRepository::new());
        repo.enqueue(job("job-1")).await.unwrap();
        let claimed = claimed(&repo, "worker-a").await;
        let guard = DreamLeaseGuard::new(repo.clone(), claimed.lease.clone());
        assert!(!guard.is_lost());

        let status = guard
            .renew(ts(20), ChronoDuration::seconds(30))
            .await
            .unwrap();
        assert!(status.is_renewed());
        assert_eq!(guard.lease().expires_at, ts(50));
        assert!(!guard.is_lost());
        assert!(guard.ensure_valid(ts(49)).is_ok());

        assert!(guard.release().await.unwrap());
        assert_eq!(repo.lease("job-1").await.unwrap(), None);
        assert!(!guard.release().await.unwrap(), "second release is a no-op");
    }

    #[tokio::test]
    async fn guard_reports_loss_after_takeover() {
        let repo = Arc::new(InMemoryDreamRepository::new());
        repo.enqueue(job("job-1")).await.unwrap();
        let first = claimed(&repo, "worker-a").await;
        let stale_guard = DreamLeaseGuard::new(repo.clone(), first.lease);

        // The lease expires; a second worker takes the job over.
        let second = repo
            .claim_next_leased(
                ts(41),
                &LeaseRequest::new("worker-b", ChronoDuration::seconds(30)),
            )
            .await
            .unwrap()
            .unwrap();
        assert_eq!(second.lease.owner, "worker-b");

        let status = stale_guard
            .renew(ts(42), ChronoDuration::seconds(30))
            .await
            .unwrap();
        // The guard notices its own expiry before contacting the repository, which is
        // the cheap and correct diagnosis: the lease is gone either way.
        assert_eq!(status, LeaseStatus::Lost(LeaseLostReason::Expired));
        assert!(stale_guard.is_lost());
        assert_eq!(stale_guard.lost_reason(), Some(LeaseLostReason::Expired));
        let error = stale_guard.ensure_valid(ts(42)).unwrap_err();
        assert!(matches!(error, Error::Memory(_)));
        assert!(error.to_string().contains("lost: expired"));
        assert!(!stale_guard.release().await.unwrap(), "not the owner");
        assert_eq!(
            repo.lease("job-1").await.unwrap().unwrap().owner,
            "worker-b"
        );
    }

    #[tokio::test]
    async fn guard_reports_local_expiry_without_a_round_trip() {
        let repo = Arc::new(InMemoryDreamRepository::new());
        repo.enqueue(job("job-1")).await.unwrap();
        let claimed = claimed(&repo, "worker-a").await;
        let guard = DreamLeaseGuard::new(repo, claimed.lease);
        assert!(guard.ensure_valid(ts(39)).is_ok());
        let error = guard.ensure_valid(ts(40)).unwrap_err();
        assert!(error.to_string().contains("expired"));
        assert!(guard.is_lost());
        assert_eq!(guard.lost_reason(), Some(LeaseLostReason::Expired));
        // A later renewal stays lost even if the clock moves backwards.
        let status = guard
            .renew(ts(10), ChronoDuration::seconds(30))
            .await
            .unwrap();
        assert_eq!(status, LeaseStatus::Lost(LeaseLostReason::Expired));
    }

    #[tokio::test]
    async fn guard_reports_loss_when_the_job_finished() {
        let repo = Arc::new(InMemoryDreamRepository::new());
        repo.enqueue(job("job-1")).await.unwrap();
        let claimed = claimed(&repo, "worker-a").await;
        let guard = DreamLeaseGuard::new(repo.clone(), claimed.lease);
        repo.complete(
            "job-1",
            crate::dreaming::DreamOutcome::succeeded(
                "job-1",
                crate::dreaming::DreamMetrics::default(),
            ),
        )
        .await
        .unwrap();
        let status = guard
            .renew(ts(20), ChronoDuration::seconds(30))
            .await
            .unwrap();
        assert_eq!(status, LeaseStatus::Lost(LeaseLostReason::NotRunning));
    }

    #[test]
    fn lease_status_helpers_and_serde() {
        let lease = DreamLease::new("job-1", "worker-a", ts(0), ChronoDuration::seconds(30));
        let renewed = LeaseStatus::Renewed(lease.clone());
        assert!(renewed.is_renewed());
        assert_eq!(renewed.lease(), Some(&lease));
        assert_eq!(renewed.reason(), None);
        let lost = LeaseStatus::Lost(LeaseLostReason::Missing);
        assert!(lost.is_lost());
        assert_eq!(lost.reason(), Some(LeaseLostReason::Missing));
        let encoded = serde_json::to_value(&lease).unwrap();
        assert_eq!(encoded["owner"], serde_json::json!("worker-a"));
        assert_eq!(
            serde_json::from_value::<DreamLease>(encoded).unwrap(),
            lease
        );
        for reason in [
            LeaseLostReason::Missing,
            LeaseLostReason::Expired,
            LeaseLostReason::OwnerMismatch,
            LeaseLostReason::NotRunning,
        ] {
            assert_eq!(LeaseStatus::Lost(reason).reason(), Some(reason));
            assert_eq!(reason.to_string(), reason.as_str());
        }
    }

    #[tokio::test]
    async fn leased_claim_moves_the_job_to_running() {
        let repo = Arc::new(InMemoryDreamRepository::new());
        repo.enqueue(job("job-1")).await.unwrap();
        let claimed = claimed(&repo, "worker-a").await;
        assert_eq!(claimed.job.state, DreamState::Running);
        assert_eq!(claimed.job.attempts, 1);
        assert_eq!(claimed.job.started_at, Some(ts(10)));
        assert_eq!(claimed.lease.job_id, "job-1");
        assert_eq!(repo.lease("job-1").await.unwrap(), Some(claimed.lease));
    }
}
