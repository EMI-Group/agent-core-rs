//! [`DreamRetention`]: timer-driven (but never self-starting) dream-job retention.
//!
//! The queue keeps finished jobs so failures stay diagnosable, but a busy application
//! accumulates them forever unless something removes the old rows. This module is that
//! policy: two independent rules, both measured against a caller-supplied `now`, so tests
//! are deterministic and nothing here reads a clock of its own.
//!
//! - **Age**: terminal jobs that finished strictly before `now - retention_secs` are
//!   removed through [`DreamRepository::delete_finished`].
//! - **Count cap**: when [`DreamRetention::max_per_state`] is set, each of the four
//!   terminal states (`succeeded`, `failed`, `skipped`, `cancelled`) keeps at most that
//!   many jobs — newest first — and every older row is removed one at a time through
//!   [`DreamRepository::delete_job`].
//!
//! The policy is idempotent: a second call with the same `now` finds nothing to remove and
//! reports an empty [`RetentionReport`]. Like
//! [`DreamPlanner`](crate::dreaming::DreamPlanner), it starts no timer, spawns no task,
//! and owns no background loop; the caller decides when a sweep happens and passes the
//! clock in. A policy with a zero window and no cap purges nothing at all.
use chrono::{DateTime, Duration, Utc};
use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};

use super::{DreamConfig, DreamFilter, DreamRepository, DreamState};

/// The terminal states a per-state count cap applies to, in the documented order.
const CAPPED_STATES: [DreamState; 4] = [
    DreamState::Succeeded,
    DreamState::Failed,
    DreamState::Skipped,
    DreamState::Cancelled,
];

/// The retention policy for finished dream jobs.
///
/// Both rules are optional and independent: a zero [`DreamRetention::retention_secs`]
/// disables the age purge, an unset [`DreamRetention::max_per_state`] disables the count
/// cap, and a policy with both disabled purges nothing. JSON is `snake_case`, every field
/// is optional on input (missing fields take their [`DreamRetention::default`] value), and
/// unknown fields are ignored.
///
/// The default is the [`DreamConfig::default`] window (7 days) with no count cap, which
/// makes the policy a safe drop-in for an application that already calls
/// [`DreamRepository::purge`](crate::dreaming::DreamRepository::purge) on a timer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct DreamRetention {
    /// Seconds a finished job is kept before the age purge removes it; `0` disables the
    /// age rule. Default `604_800` (7 days), matching [`DreamConfig::retention_secs`].
    pub retention_secs: u64,
    /// Maximum jobs kept per terminal state, newest first; `None` disables the cap.
    ///
    /// `Some(0)` is rejected by [`DreamRetention::validate`]: a zero cap would delete
    /// every finished job, which the age rule cannot express and callers should not ask
    /// for by accident.
    pub max_per_state: Option<usize>,
}

impl Default for DreamRetention {
    /// The documented defaults: a 7-day age window and no per-state count cap.
    fn default() -> Self {
        Self {
            retention_secs: 604_800,
            max_per_state: None,
        }
    }
}

impl DreamRetention {
    /// Builds an age-only policy that keeps finished jobs for `retention_secs` seconds.
    ///
    /// `retention_secs = 0` disables the age rule, so the returned policy purges nothing
    /// until a count cap is added with [`DreamRetention::with_max_per_state`].
    #[must_use]
    pub const fn new(retention_secs: u64) -> Self {
        Self {
            retention_secs,
            max_per_state: None,
        }
    }

    /// Builds a policy that purges nothing: a zero age window and no count cap.
    #[must_use]
    pub const fn disabled() -> Self {
        Self::new(0)
    }

    /// Builds a policy from `config`, copying its retention window and leaving the count
    /// cap unset.
    #[must_use]
    pub fn from_config(config: &DreamConfig) -> Self {
        Self::new(config.retention_secs)
    }

    /// Returns the policy with `retention_secs` as its age window.
    #[must_use]
    pub const fn with_retention_secs(mut self, retention_secs: u64) -> Self {
        self.retention_secs = retention_secs;
        self
    }

    /// Returns the policy with `max_per_state` as its per-state count cap.
    #[must_use]
    pub const fn with_max_per_state(mut self, max_per_state: usize) -> Self {
        self.max_per_state = Some(max_per_state);
        self
    }

    /// Returns [`DreamRetention::retention_secs`] as a [`chrono::Duration`].
    ///
    /// Values chrono cannot represent saturate at [`Duration::MAX`] instead of wrapping or
    /// panicking, mirroring [`DreamConfig::retention`].
    #[must_use]
    pub fn retention(&self) -> Duration {
        i64::try_from(self.retention_secs)
            .ok()
            .and_then(Duration::try_seconds)
            .unwrap_or(Duration::MAX)
    }

    /// Returns `true` when at least one rule is active.
    ///
    /// A policy with a zero window and no count cap is enabled nowhere, so
    /// [`DreamRetention::purge_expired`] returns an empty report without touching the
    /// repository.
    #[must_use]
    pub const fn is_enabled(&self) -> bool {
        self.retention_secs > 0 || self.max_per_state.is_some()
    }

    /// Validates the policy, returning `Error::Validation` naming the offending field.
    ///
    /// A zero `retention_secs` is allowed (the age rule is off); `max_per_state = Some(0)`
    /// is rejected because it would delete every finished job.
    pub fn validate(&self) -> Result<()> {
        if self.max_per_state == Some(0) {
            return Err(Error::validation(
                "dream retention `max_per_state` must be greater than 0",
            ));
        }
        Ok(())
    }

    /// Applies the policy to `repository` at `now` and reports what was removed.
    ///
    /// The age purge runs first (when the window is non-zero), then the per-state cap is
    /// applied to the rows that remain. Both rules are idempotent, so a second call with
    /// the same `now` removes nothing and returns an empty [`RetentionReport`].
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] when the policy itself is invalid, and whatever error the
    /// repository returns — repository failures propagate unchanged.
    pub async fn purge_expired(
        &self,
        repository: &dyn DreamRepository,
        now: DateTime<Utc>,
    ) -> Result<RetentionReport> {
        self.validate()?;
        let mut report = RetentionReport::default();
        if !self.is_enabled() {
            return Ok(report);
        }
        if self.retention_secs > 0 {
            let cutoff = now
                .checked_sub_signed(self.retention())
                .unwrap_or(DateTime::<Utc>::MIN_UTC);
            report.deleted_expired = repository.delete_finished(cutoff).await?;
        }
        if let Some(cap) = self.max_per_state {
            for state in CAPPED_STATES {
                report.deleted_over_cap = report
                    .deleted_over_cap
                    .saturating_add(trim_state(repository, state, cap).await?);
            }
        }
        Ok(report)
    }
}

/// Removes every `state` job beyond the newest `cap` rows.
///
/// Jobs are ordered newest first by `finished_at` (falling back to `updated_at`) and then
/// by id ascending, so the verdict is deterministic even when timestamps tie. A row the
/// repository reports as already gone is not an error and is not counted.
async fn trim_state(
    repository: &dyn DreamRepository,
    state: DreamState,
    cap: usize,
) -> Result<usize> {
    let mut jobs = repository
        .list(&DreamFilter::new().with_state(state))
        .await?;
    jobs.sort_by(|left, right| {
        let left_finished = left.finished_at.unwrap_or(left.updated_at);
        let right_finished = right.finished_at.unwrap_or(right.updated_at);
        right_finished
            .cmp(&left_finished)
            .then_with(|| left.id.cmp(&right.id))
    });
    let mut deleted = 0_usize;
    for job in jobs.into_iter().skip(cap) {
        if repository.delete_job(&job.id).await? {
            deleted = deleted.saturating_add(1);
        }
    }
    Ok(deleted)
}

/// What one [`DreamRetention::purge_expired`] sweep removed.
///
/// The counters are split by rule so an operator can tell an age cutoff from a count cap;
/// [`RetentionReport::total`] is their sum. JSON is `snake_case`, every field is optional
/// on input (missing fields default to `0`), and unknown fields are ignored.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct RetentionReport {
    /// Jobs removed by the age cutoff.
    pub deleted_expired: usize,
    /// Jobs removed by the per-state count cap.
    pub deleted_over_cap: usize,
}

impl RetentionReport {
    /// Returns the total number of jobs removed by both rules, saturating on overflow.
    #[must_use]
    pub const fn total(&self) -> usize {
        self.deleted_expired.saturating_add(self.deleted_over_cap)
    }

    /// Returns `true` when the sweep removed nothing.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.deleted_expired == 0 && self.deleted_over_cap == 0
    }

    /// Adds `other` into `self`, saturating instead of wrapping on overflow.
    pub const fn merge(&mut self, other: Self) {
        self.deleted_expired = self.deleted_expired.saturating_add(other.deleted_expired);
        self.deleted_over_cap = self.deleted_over_cap.saturating_add(other.deleted_over_cap);
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
    use serde_json::json;

    fn ts(offset: i64) -> DateTime<Utc> {
        DateTime::from_timestamp(1_700_000_000 + offset, 0).unwrap()
    }

    fn scope() -> MemoryScope {
        MemoryScope::new("alice", "workspace")
    }

    fn terminal_job(
        id: &str,
        state: DreamState,
        created: DateTime<Utc>,
        finished: DateTime<Utc>,
    ) -> DreamJob {
        DreamJob {
            state,
            started_at: Some(finished),
            finished_at: Some(finished),
            attempts: 1,
            updated_at: finished,
            ..DreamJob::new(
                scope(),
                DreamKind::Consolidation,
                DreamTrigger::Manual,
                created,
            )
        }
        .with_id(id)
    }

    fn pending_job(id: &str) -> DreamJob {
        DreamJob::new(
            scope(),
            DreamKind::Consolidation,
            DreamTrigger::Manual,
            ts(0),
        )
        .with_id(id)
    }

    fn running_job(id: &str) -> DreamJob {
        DreamJob {
            state: DreamState::Running,
            started_at: Some(ts(1)),
            attempts: 1,
            ..pending_job(id)
        }
    }

    async fn ids(repo: &InMemoryDreamRepository) -> Vec<String> {
        let mut ids = repo
            .list(&DreamFilter::new())
            .await
            .unwrap()
            .into_iter()
            .map(|job| job.id)
            .collect::<Vec<_>>();
        ids.sort();
        ids
    }

    #[tokio::test]
    async fn age_purge_removes_only_jobs_finished_before_the_cutoff() {
        let repo = InMemoryDreamRepository::new();
        repo.enqueue(terminal_job(
            "expired",
            DreamState::Succeeded,
            ts(0),
            ts(10),
        ))
        .await
        .unwrap();
        repo.enqueue(terminal_job("fresh", DreamState::Failed, ts(0), ts(950)))
            .await
            .unwrap();
        repo.enqueue(terminal_job(
            "boundary",
            DreamState::Cancelled,
            ts(0),
            ts(900),
        ))
        .await
        .unwrap();
        repo.enqueue(pending_job("pending")).await.unwrap();
        repo.enqueue(running_job("running")).await.unwrap();

        let report = DreamRetention::new(100)
            .purge_expired(&repo, ts(1000))
            .await
            .unwrap();
        assert_eq!(
            report,
            RetentionReport {
                deleted_expired: 1,
                deleted_over_cap: 0,
            }
        );
        assert_eq!(report.total(), 1);
        assert!(!report.is_empty());
        assert!(repo.get("expired").await.unwrap().is_none());
        assert!(repo.get("boundary").await.unwrap().is_some());
        assert_eq!(
            ids(&repo).await,
            vec!["boundary", "fresh", "pending", "running"]
        );
    }

    #[tokio::test]
    async fn age_purge_is_idempotent() {
        let repo = InMemoryDreamRepository::new();
        repo.enqueue(terminal_job(
            "expired",
            DreamState::Succeeded,
            ts(0),
            ts(10),
        ))
        .await
        .unwrap();
        repo.enqueue(terminal_job("fresh", DreamState::Skipped, ts(0), ts(950)))
            .await
            .unwrap();
        let retention = DreamRetention::new(100);
        assert_eq!(
            retention.purge_expired(&repo, ts(1000)).await.unwrap(),
            RetentionReport {
                deleted_expired: 1,
                deleted_over_cap: 0,
            }
        );

        let second = retention.purge_expired(&repo, ts(1000)).await.unwrap();
        assert!(second.is_empty());
        assert_eq!(repo.len(), 1);
        assert_eq!(ids(&repo).await, vec!["fresh"]);
    }

    #[tokio::test]
    async fn cap_keeps_the_newest_jobs_per_state() {
        let repo = InMemoryDreamRepository::new();
        for (id, finished) in [("s-10", 10), ("s-20", 20), ("s-30", 30), ("s-40", 40)] {
            repo.enqueue(terminal_job(id, DreamState::Succeeded, ts(0), ts(finished)))
                .await
                .unwrap();
        }
        for (id, finished) in [("f-5", 5), ("f-15", 15), ("f-25", 25)] {
            repo.enqueue(terminal_job(id, DreamState::Failed, ts(0), ts(finished)))
                .await
                .unwrap();
        }
        repo.enqueue(terminal_job("k-1", DreamState::Skipped, ts(0), ts(1)))
            .await
            .unwrap();
        repo.enqueue(pending_job("pending")).await.unwrap();

        let retention = DreamRetention::disabled().with_max_per_state(2);
        assert!(retention.is_enabled());
        let report = retention.purge_expired(&repo, ts(1000)).await.unwrap();
        assert_eq!(
            report,
            RetentionReport {
                deleted_expired: 0,
                deleted_over_cap: 3,
            }
        );
        assert_eq!(
            ids(&repo).await,
            vec!["f-15", "f-25", "k-1", "pending", "s-30", "s-40"]
        );
        for id in ["s-10", "s-20", "f-5"] {
            assert!(repo.get(id).await.unwrap().is_none(), "kept {id}");
        }
    }

    #[tokio::test]
    async fn age_purge_runs_before_the_cap() {
        let repo = InMemoryDreamRepository::new();
        for (id, finished) in [("old", 10), ("middle", 950), ("newest", 960)] {
            repo.enqueue(terminal_job(id, DreamState::Succeeded, ts(0), ts(finished)))
                .await
                .unwrap();
        }

        let report = DreamRetention::new(100)
            .with_max_per_state(1)
            .purge_expired(&repo, ts(1000))
            .await
            .unwrap();
        assert_eq!(
            report,
            RetentionReport {
                deleted_expired: 1,
                deleted_over_cap: 1,
            }
        );
        assert_eq!(ids(&repo).await, vec!["newest"]);
    }

    /// A repository whose single-row deletes report "already gone" without removing
    /// anything, modelling a row another worker deleted first.
    struct VanishRepository {
        inner: InMemoryDreamRepository,
    }

    #[async_trait::async_trait]
    impl DreamRepository for VanishRepository {
        async fn enqueue(&self, job: DreamJob) -> Result<DreamJob> {
            self.inner.enqueue(job).await
        }

        async fn claim_next(&self, now: DateTime<Utc>) -> Result<Option<DreamJob>> {
            self.inner.claim_next(now).await
        }

        async fn delete_job(&self, _job_id: &str) -> Result<bool> {
            Ok(false)
        }

        async fn complete(
            &self,
            job_id: &str,
            outcome: crate::dreaming::DreamOutcome,
        ) -> Result<DreamJob> {
            self.inner.complete(job_id, outcome).await
        }

        async fn fail(&self, job_id: &str, error: &str) -> Result<DreamJob> {
            self.inner.fail(job_id, error).await
        }

        async fn get(&self, job_id: &str) -> Result<Option<DreamJob>> {
            self.inner.get(job_id).await
        }

        async fn list(&self, filter: &DreamFilter) -> Result<Vec<DreamJob>> {
            self.inner.list(filter).await
        }

        async fn count_by_state(&self) -> Result<std::collections::BTreeMap<DreamState, usize>> {
            self.inner.count_by_state().await
        }

        async fn delete_finished(&self, older_than: DateTime<Utc>) -> Result<usize> {
            self.inner.delete_finished(older_than).await
        }
    }

    #[tokio::test]
    async fn cap_counts_only_rows_the_repository_actually_deleted() {
        let repo = VanishRepository {
            inner: InMemoryDreamRepository::new(),
        };
        for (id, finished) in [("a", 10), ("b", 20), ("c", 30)] {
            repo.enqueue(terminal_job(id, DreamState::Succeeded, ts(0), ts(finished)))
                .await
                .unwrap();
        }

        let report = DreamRetention::disabled()
            .with_max_per_state(1)
            .purge_expired(&repo, ts(1000))
            .await
            .unwrap();
        assert!(report.is_empty());
        assert_eq!(repo.inner.len(), 3);
    }

    #[tokio::test]
    async fn disabled_policy_deletes_nothing() {
        let repo = InMemoryDreamRepository::new();
        repo.enqueue(terminal_job("old", DreamState::Succeeded, ts(0), ts(0)))
            .await
            .unwrap();

        let retention = DreamRetention::disabled();
        assert!(!retention.is_enabled());
        let report = retention.purge_expired(&repo, ts(1_000_000)).await.unwrap();
        assert!(report.is_empty());
        assert_eq!(repo.len(), 1);
    }

    #[test]
    fn defaults_builders_and_config_mapping() {
        let retention = DreamRetention::default();
        assert_eq!(retention.retention_secs, 604_800);
        assert_eq!(retention.max_per_state, None);
        assert_eq!(retention, DreamRetention::new(604_800));
        assert!(retention.is_enabled());
        assert_eq!(retention.retention(), Duration::days(7));

        let retention = DreamRetention::disabled()
            .with_retention_secs(120)
            .with_max_per_state(5);
        assert_eq!(retention.retention_secs, 120);
        assert_eq!(retention.max_per_state, Some(5));
        assert_eq!(retention.retention(), Duration::seconds(120));
        assert_eq!(DreamRetention::new(u64::MAX).retention(), Duration::MAX);

        let config = DreamConfig {
            retention_secs: 3600,
            ..DreamConfig::default()
        };
        let retention = DreamRetention::from_config(&config);
        assert_eq!(retention, DreamRetention::new(3600));
        assert_eq!(retention.max_per_state, None);
        assert_eq!(retention.retention(), Duration::seconds(3600));
    }

    #[test]
    fn validate_allows_zero_window_but_rejects_zero_cap() {
        assert!(DreamRetention::disabled().validate().is_ok());
        assert!(DreamRetention::new(0).validate().is_ok());
        assert!(DreamRetention::new(60).validate().is_ok());
        assert!(DreamRetention::disabled()
            .with_max_per_state(1)
            .validate()
            .is_ok());

        let err = DreamRetention::disabled()
            .with_max_per_state(0)
            .validate()
            .unwrap_err();
        assert!(matches!(err, Error::Validation(_)));
        assert!(err.to_string().contains("max_per_state"));
    }

    #[tokio::test]
    async fn purge_rejects_an_invalid_policy_before_touching_the_repository() {
        let repo = InMemoryDreamRepository::new();
        repo.enqueue(terminal_job("old", DreamState::Succeeded, ts(0), ts(0)))
            .await
            .unwrap();

        let err = DreamRetention::disabled()
            .with_max_per_state(0)
            .purge_expired(&repo, ts(1000))
            .await
            .unwrap_err();
        assert!(matches!(err, Error::Validation(_)));
        assert_eq!(repo.len(), 1);
    }

    #[test]
    fn report_merges_and_totals_with_saturation() {
        let mut report = RetentionReport::default();
        assert!(report.is_empty());
        assert_eq!(report.total(), 0);

        report.merge(RetentionReport {
            deleted_expired: 2,
            deleted_over_cap: 3,
        });
        assert!(!report.is_empty());
        assert_eq!(report.total(), 5);
        report.merge(RetentionReport {
            deleted_expired: 1,
            deleted_over_cap: 0,
        });
        assert_eq!(
            report,
            RetentionReport {
                deleted_expired: 3,
                deleted_over_cap: 3,
            }
        );

        let mut saturated = RetentionReport {
            deleted_expired: usize::MAX,
            deleted_over_cap: usize::MAX,
        };
        saturated.merge(RetentionReport {
            deleted_expired: 1,
            deleted_over_cap: 1,
        });
        assert_eq!(
            saturated,
            RetentionReport {
                deleted_expired: usize::MAX,
                deleted_over_cap: usize::MAX,
            }
        );
        assert_eq!(saturated.total(), usize::MAX);
    }

    #[test]
    fn serde_defaults_and_round_trip() {
        let retention: DreamRetention = serde_json::from_value(json!({})).unwrap();
        assert_eq!(retention, DreamRetention::default());
        let retention: DreamRetention =
            serde_json::from_value(json!({"retention_secs": 120, "unknown": true})).unwrap();
        assert_eq!(retention, DreamRetention::new(120));

        let retention = DreamRetention::default().with_max_per_state(7);
        let value = serde_json::to_value(&retention).unwrap();
        assert_eq!(value["retention_secs"], json!(604_800));
        assert_eq!(value["max_per_state"], json!(7));
        assert_eq!(
            serde_json::from_value::<DreamRetention>(value).unwrap(),
            retention
        );

        let report: RetentionReport = serde_json::from_value(json!({})).unwrap();
        assert_eq!(report, RetentionReport::default());
        let report: RetentionReport =
            serde_json::from_value(json!({"deleted_expired": 2, "deleted_over_cap": 4})).unwrap();
        assert_eq!(report.total(), 6);
        let value = serde_json::to_value(report).unwrap();
        assert_eq!(value["deleted_expired"], json!(2));
        assert_eq!(value["deleted_over_cap"], json!(4));
        assert_eq!(
            serde_json::from_value::<RetentionReport>(value).unwrap(),
            report
        );
    }
}
