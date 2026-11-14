//! Lease-aware sweeps: [`DreamPlanner::run_leased`] and
//! [`DreamPlanner::run_due_leased`].
//!
//! [`DreamPlanner::run`] trusts that a claimed job still belongs to it when the transition
//! is recorded. That is true inside one process, where the claim and the transition are
//! separated only by the consolidator call, but not across processes: a worker that stalls
//! long enough loses its lease, and another worker may take the job over and finish it.
//!
//! The leased sweep closes that gap. It claims through
//! [`DreamRepository::claim_next_leased`](crate::dreaming::DreamRepository::claim_next_leased),
//! wraps the resulting lease in a [`DreamLeaseGuard`], consolidates, renews the lease, and
//! only then records the transition. When the renewal reports
//! [`LeaseStatus::Lost`] the job may already belong to somebody else, so the sweep records
//! nothing and stops.
//!
//! Everything else is shared with [`DreamPlanner::run`], not reimplemented: the sweep uses
//! the same `consolidate_step`/`record_step` pair (see `DreamPlanner::attempt`) that the
//! trusting sweep composes, so
//! the cancellation policy, the outcome routing, the forced job id, and the metrics are
//! identical. The split exists precisely so the lease renewal can happen between the two.
//!
//! The heartbeat in [`DreamLeaseOptions::heartbeat`] is advisory here: one job's
//! consolidation is expected to be short relative to the TTL, and the sweep renews exactly
//! once per job, so it spawns no background renewal task. A consolidator that can run
//! longer than its TTL has to renew through its own [`DreamLeaseGuard`] (the guard is
//! cheap to clone into the consolidator if needed) or shorten its work below the TTL;
//! otherwise this module reports the loss and stops.
use std::sync::Arc;

use chrono::{DateTime, Utc};
use openjiuwen_common::error::Result;

use super::DreamPlanner;
use crate::dreaming::{
    DreamJob, DreamLeaseGuard, DreamLeaseOptions, DreamRunSummary, LeaseStatus, LeasedJob,
};
use crate::record::CallOptions;

impl DreamPlanner {
    /// Runs one lease-aware sweep: at most `limit` leased claims at the caller's `now`.
    ///
    /// The sweep mirrors [`DreamPlanner::run`] except where ownership has to be proven:
    ///
    /// - [`DreamLeaseOptions::validate`] runs first, so a bad owner, a non-positive TTL, or
    ///   a heartbeat that is not shorter than the TTL is an [`Error::Validation`] *before*
    ///   anything is claimed.
    /// - A disabled [`DreamConfig::enabled`](crate::dreaming::DreamConfig::enabled) or
    ///   `limit == 0` returns an empty [`DreamRunSummary`] without claiming anything.
    /// - Cancellation is checked *before* every claim, exactly as in `run`.
    /// - Each iteration claims through
    ///   [`DreamRepository::claim_next_leased`](crate::dreaming::DreamRepository::claim_next_leased)
    ///   with the options' request (owner and TTL) and stops when that returns `None`. The
    ///   claim either takes the pending head or takes over a `Running` job whose lease
    ///   expired, never both.
    /// - The claimed lease becomes a [`DreamLeaseGuard`] over the same repository.
    /// - The job is consolidated through the shared `DreamPlanner::attempt` steps, so the
    ///   outcome routing and
    ///   metrics are identical to `run`. The one deviation is ordering: **before** the
    ///   transition is written, the lease is renewed with
    ///   [`DreamLeaseGuard::renew`]`(now, ttl)`.
    ///   [`LeaseStatus::Lost`] means the job may already belong to another worker, so
    ///   nothing is recorded (a completion or a requeue would overwrite the new owner's
    ///   row), the loss is logged with the job id and the reason, and the sweep returns the
    ///   summary collected so far.
    /// - Afterwards — including after a lost lease — the lease is released with
    ///   [`DreamLeaseGuard::release`]. `false` is not an error: it only means the lease was
    ///   already gone (released, expired and taken over, or deleted), so it is logged at
    ///   debug level.
    ///
    /// A terminally completed, skipped, cancelled, requeued, or terminally failed job all
    /// leave the job's lease released. A cancelled consolidator still completes the job as
    /// [`DreamState::Cancelled`](crate::dreaming::DreamState::Cancelled) and stops the
    /// sweep, and the release still runs. [`Error::Cancelled`] raised by `renew` or
    /// `release` propagates rather than being degraded; a repository that does not
    /// implement leasing fails the claim with [`Error::Unsupported`], which propagates too.
    ///
    /// [`DreamRunSummary`] counts recorded outcomes, so a job whose lease was lost adds
    /// nothing: `scanned` counts claims that reached a transition, and a lost lease
    /// deliberately produces no transition.
    ///
    /// [`Error::Validation`]: openjiuwen_common::error::Error::Validation
    /// [`Error::Cancelled`]: openjiuwen_common::error::Error::Cancelled
    /// [`Error::Unsupported`]: openjiuwen_common::error::Error::Unsupported
    pub async fn run_leased(
        &self,
        now: DateTime<Utc>,
        limit: usize,
        ctx: &CallOptions,
        options: &DreamLeaseOptions,
    ) -> Result<DreamRunSummary> {
        options.validate()?;
        let mut summary = DreamRunSummary::new();
        if !self.config().enabled {
            tracing::debug!(
                target: "openjiuwen_memory::dreaming::planner::lease",
                "dreaming is disabled; the leased sweep claims nothing"
            );
            return Ok(summary);
        }
        if limit == 0 {
            tracing::debug!(
                target: "openjiuwen_memory::dreaming::planner::lease",
                "sweep limit is zero; the leased sweep claims nothing"
            );
            return Ok(summary);
        }
        for claimed in 0..limit {
            if ctx.check().is_err() {
                tracing::debug!(
                    target: "openjiuwen_memory::dreaming::planner::lease",
                    claimed,
                    "the leased sweep was cancelled before the next claim"
                );
                break;
            }
            let Some(leased) = self
                .repository()
                .claim_next_leased(now, &options.request())
                .await?
            else {
                break;
            };
            let LeasedJob { job, lease } = leased;
            let guard = DreamLeaseGuard::new(Arc::clone(self.repository()), lease);
            // Release in every path, including a failed attempt and a lost lease.
            let attempt = self
                .leased_attempt(&job, &guard, now, ctx, options, &mut summary)
                .await;
            let released = guard.release().await;
            let stop = attempt?;
            match released {
                Ok(true) => tracing::debug!(
                    target: "openjiuwen_memory::dreaming::planner::lease",
                    job_id = %job.id,
                    "released the dream lease"
                ),
                Ok(false) => tracing::debug!(
                    target: "openjiuwen_memory::dreaming::planner::lease",
                    job_id = %job.id,
                    "the dream lease was already gone at release time"
                ),
                Err(error) => return Err(error),
            }
            if stop {
                break;
            }
        }
        Ok(summary)
    }

    /// Runs one lease-aware sweep with
    /// [`DreamConfig::batch_size`](crate::dreaming::DreamConfig::batch_size) as the limit.
    ///
    /// The leased counterpart of [`DreamPlanner::run_due`]: a scheduler that wants
    /// cross-process ownership only has to own the timer and pass its options through.
    pub async fn run_due_leased(
        &self,
        now: DateTime<Utc>,
        ctx: &CallOptions,
        options: &DreamLeaseOptions,
    ) -> Result<DreamRunSummary> {
        self.run_leased(now, self.config().batch_size, ctx, options)
            .await
    }

    /// Consolidates one leased job, renews the lease, and records the transition.
    ///
    /// Returns `true` when the sweep must stop: either the run was cancelled (the shared
    /// recording path) or the lease was lost before the write. A lost lease writes nothing
    /// and is logged here; the caller releases the (already gone) lease either way.
    async fn leased_attempt(
        &self,
        job: &DreamJob,
        guard: &DreamLeaseGuard,
        now: DateTime<Utc>,
        ctx: &CallOptions,
        options: &DreamLeaseOptions,
        summary: &mut DreamRunSummary,
    ) -> Result<bool> {
        let result = self.consolidate_step(job, ctx).await;
        match guard.renew(now, options.ttl).await? {
            LeaseStatus::Renewed(lease) => tracing::debug!(
                target: "openjiuwen_memory::dreaming::planner::lease",
                job_id = %job.id,
                owner = %lease.owner,
                expires_at = %lease.expires_at,
                "renewed the dream lease before recording the transition"
            ),
            LeaseStatus::Lost(reason) => {
                tracing::warn!(
                    target: "openjiuwen_memory::dreaming::planner::lease",
                    job_id = %job.id,
                    owner = %options.owner,
                    reason = reason.as_str(),
                    "dream lease lost before the transition; recording nothing and stopping"
                );
                return Ok(true);
            }
        }
        self.record_step(job, result, summary).await
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::dreaming::{
        DreamConfig, DreamConsolidator, DreamFilter, DreamJob, DreamKind, DreamMetrics,
        DreamOutcome, DreamRepository, DreamState, DreamTrigger, InMemoryDreamRepository,
    };
    use crate::record::MemoryScope;
    use async_trait::async_trait;
    use chrono::Duration;
    use openjiuwen_common::error::Error;
    use parking_lot::Mutex;
    use std::collections::{BTreeMap, VecDeque};
    use std::time::Duration as StdDuration;
    use tokio_util::sync::CancellationToken;

    /// One scripted consolidator behaviour, mirroring the `planner` test module.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Step {
        /// Succeed with zero metrics.
        Succeeded,
        /// Fail with a generic memory error.
        Failed,
        /// Fail with [`Error::Cancelled`].
        Cancelled,
        /// Block until released, so a test can race or cancel mid-consolidation.
        Block,
    }

    /// A consolidator that replays a script of [`Step`]s, then repeats `Succeeded`.
    #[derive(Debug)]
    struct ScriptedConsolidator {
        steps: Mutex<VecDeque<Step>>,
        calls: Mutex<Vec<String>>,
        entered: tokio::sync::Notify,
        release: tokio::sync::Notify,
    }

    impl ScriptedConsolidator {
        fn new(steps: impl IntoIterator<Item = Step>) -> Self {
            Self {
                steps: Mutex::new(steps.into_iter().collect()),
                calls: Mutex::new(Vec::new()),
                entered: tokio::sync::Notify::new(),
                release: tokio::sync::Notify::new(),
            }
        }

        fn calls(&self) -> Vec<String> {
            self.calls.lock().clone()
        }
    }

    #[async_trait]
    impl DreamConsolidator for ScriptedConsolidator {
        async fn consolidate(&self, job: &DreamJob, _ctx: &CallOptions) -> Result<DreamOutcome> {
            self.calls.lock().push(job.id.clone());
            let step = self.steps.lock().pop_front().unwrap_or(Step::Succeeded);
            match step {
                Step::Succeeded => Ok(DreamOutcome::succeeded(
                    job.id.clone(),
                    DreamMetrics::default(),
                )),
                Step::Failed => Err(Error::Memory("scripted failure".to_owned())),
                Step::Cancelled => Err(Error::Cancelled),
                Step::Block => {
                    self.entered.notify_one();
                    self.release.notified().await;
                    Ok(DreamOutcome::succeeded(
                        job.id.clone(),
                        DreamMetrics::default(),
                    ))
                }
            }
        }
    }

    /// A repository that implements only the required methods, so the trait's default
    /// (typed `Unsupported`) lease bodies are exercised.
    #[derive(Debug, Default)]
    struct PlainRepository {
        inner: InMemoryDreamRepository,
    }

    #[async_trait]
    impl DreamRepository for PlainRepository {
        async fn enqueue(&self, job: DreamJob) -> Result<DreamJob> {
            self.inner.enqueue(job).await
        }

        async fn claim_next(&self, now: DateTime<Utc>) -> Result<Option<DreamJob>> {
            self.inner.claim_next(now).await
        }

        async fn complete(&self, job_id: &str, outcome: DreamOutcome) -> Result<DreamJob> {
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

        async fn count_by_state(&self) -> Result<BTreeMap<DreamState, usize>> {
            self.inner.count_by_state().await
        }

        async fn delete_finished(&self, older_than: DateTime<Utc>) -> Result<usize> {
            self.inner.delete_finished(older_than).await
        }
    }

    fn ts(secs: i64) -> DateTime<Utc> {
        DateTime::from_timestamp(1_700_000_000, 0).unwrap() + Duration::seconds(secs)
    }

    fn job(id: &str, max_attempts: u32) -> DreamJob {
        DreamJob::new(
            MemoryScope::new("alice", "ws"),
            DreamKind::Consolidation,
            DreamTrigger::Manual,
            ts(0),
        )
        .with_id(id)
        .with_max_attempts(max_attempts)
    }

    fn options(owner: &str, ttl_secs: i64, heartbeat_secs: u64) -> DreamLeaseOptions {
        DreamLeaseOptions::new(
            owner,
            Duration::seconds(ttl_secs),
            StdDuration::from_secs(heartbeat_secs),
        )
    }

    fn planner(
        repo: &Arc<InMemoryDreamRepository>,
        consolidator: &Arc<ScriptedConsolidator>,
        config: DreamConfig,
    ) -> DreamPlanner {
        DreamPlanner::new(repo.clone(), consolidator.clone(), config).unwrap()
    }

    async fn stored(planner: &DreamPlanner, id: &str) -> DreamJob {
        planner
            .repository()
            .get(id)
            .await
            .unwrap()
            .expect("the job is stored")
    }

    async fn lease_of(planner: &DreamPlanner, id: &str) -> Option<crate::dreaming::DreamLease> {
        planner.repository().lease(id).await.unwrap()
    }

    #[tokio::test]
    async fn leased_sweep_completes_and_releases_the_lease() {
        let repo = Arc::new(InMemoryDreamRepository::new());
        let consolidator = Arc::new(ScriptedConsolidator::new([Step::Succeeded]));
        let planner = planner(&repo, &consolidator, DreamConfig::default());
        repo.enqueue(job("job-1", 3)).await.unwrap();

        let summary = planner
            .run_leased(
                ts(10),
                10,
                &CallOptions::new(),
                &options("worker-a", 30, 10),
            )
            .await
            .unwrap();
        assert_eq!(
            (
                summary.scanned,
                summary.succeeded,
                summary.failed,
                summary.cancelled
            ),
            (1, 1, 0, 0)
        );
        let stored = stored(&planner, "job-1").await;
        assert_eq!(stored.state, DreamState::Succeeded);
        assert_eq!(stored.attempts, 1);
        assert!(stored.finished_at.is_some());
        assert!(
            lease_of(&planner, "job-1").await.is_none(),
            "the lease must be released"
        );
        assert_eq!(consolidator.calls(), vec!["job-1".to_owned()]);
    }

    #[tokio::test]
    async fn racing_sweeps_claim_one_job_once() {
        let repo = Arc::new(InMemoryDreamRepository::new());
        let consolidator = Arc::new(ScriptedConsolidator::new([Step::Block]));
        let planner = Arc::new(planner(&repo, &consolidator, DreamConfig::default()));
        repo.enqueue(job("job-1", 3)).await.unwrap();

        let first_options = options("worker-a", 30, 10);
        let second_options = options("worker-b", 30, 10);
        let running = Arc::clone(&planner);
        let handle = tokio::spawn(async move {
            running
                .run_leased(ts(10), 10, &CallOptions::new(), &first_options)
                .await
        });
        consolidator.entered.notified().await;

        let second = planner
            .run_leased(ts(10), 10, &CallOptions::new(), &second_options)
            .await
            .unwrap();
        assert!(
            second.is_empty(),
            "the running job is leased by the first sweep at the same clock"
        );

        consolidator.release.notify_one();
        let first = handle.await.unwrap().unwrap();
        assert_eq!((first.scanned, first.succeeded), (1, 1));
        assert_eq!(
            consolidator.calls(),
            vec!["job-1".to_owned()],
            "exactly one consolidation"
        );
        assert_eq!(stored(&planner, "job-1").await.state, DreamState::Succeeded);
        assert!(lease_of(&planner, "job-1").await.is_none());
    }

    #[tokio::test]
    async fn an_expired_lease_is_taken_over_and_the_old_owner_records_nothing() {
        let repo = Arc::new(InMemoryDreamRepository::new());
        let consolidator = Arc::new(ScriptedConsolidator::new([Step::Block, Step::Succeeded]));
        let planner = Arc::new(planner(&repo, &consolidator, DreamConfig::default()));
        repo.enqueue(job("job-1", 3)).await.unwrap();

        let stale_options = options("worker-a", 2, 1);
        let fresh_options = options("worker-b", 30, 10);
        let stale = Arc::clone(&planner);
        let handle = tokio::spawn(async move {
            stale
                .run_leased(ts(10), 10, &CallOptions::new(), &stale_options)
                .await
        });
        consolidator.entered.notified().await;
        let stale_owner = lease_of(&planner, "job-1").await.expect("leased");
        assert_eq!(stale_owner.owner, "worker-a");
        assert_eq!(stale_owner.expires_at, ts(12));

        // `now` is past the first lease's expiry, so the takeover path applies.
        let second = planner
            .run_leased(ts(13), 10, &CallOptions::new(), &fresh_options)
            .await
            .unwrap();
        assert_eq!((second.scanned, second.succeeded), (1, 1));
        let taken = stored(&planner, "job-1").await;
        assert_eq!(taken.state, DreamState::Succeeded);
        assert_eq!(
            taken.attempts, 2,
            "the takeover counted as a second attempt"
        );

        // The stale owner wakes up, fails its renewal, and records no transition.
        consolidator.release.notify_one();
        let stale_summary = handle.await.unwrap().unwrap();
        assert!(
            stale_summary.is_empty(),
            "a lost lease must not record a transition"
        );
        let after = stored(&planner, "job-1").await;
        assert_eq!(after.state, DreamState::Succeeded);
        assert_eq!(after.attempts, 2, "the stale worker overwrote nothing");
        assert!(lease_of(&planner, "job-1").await.is_none());
    }

    #[tokio::test]
    async fn pre_cancelled_leased_sweep_claims_nothing() {
        let repo = Arc::new(InMemoryDreamRepository::new());
        let consolidator = Arc::new(ScriptedConsolidator::new([]));
        let planner = planner(&repo, &consolidator, DreamConfig::default());
        repo.enqueue(job("job-1", 3)).await.unwrap();

        let token = CancellationToken::new();
        token.cancel();
        let ctx = CallOptions::new().with_cancellation(token);
        let summary = planner
            .run_leased(ts(10), 10, &ctx, &options("worker-a", 30, 10))
            .await
            .unwrap();
        assert!(summary.is_empty());
        assert_eq!(consolidator.calls(), Vec::<String>::new());
        let untouched = stored(&planner, "job-1").await;
        assert_eq!(untouched.state, DreamState::Pending);
        assert_eq!(untouched.attempts, 0);
        assert!(lease_of(&planner, "job-1").await.is_none());
    }

    #[tokio::test]
    async fn cancellation_mid_consolidation_completes_cancelled_and_releases() {
        let repo = Arc::new(InMemoryDreamRepository::new());
        let consolidator = Arc::new(ScriptedConsolidator::new([Step::Block]));
        let planner = Arc::new(planner(&repo, &consolidator, DreamConfig::default()));
        repo.enqueue(job("first", 3)).await.unwrap();
        repo.enqueue(job("second", 3)).await.unwrap();

        let token = CancellationToken::new();
        let ctx = CallOptions::new().with_cancellation(token.clone());
        let leased_options = options("worker-a", 30, 10);
        let running = Arc::clone(&planner);
        let handle =
            tokio::spawn(
                async move { running.run_leased(ts(10), 10, &ctx, &leased_options).await },
            );

        consolidator.entered.notified().await;
        token.cancel();
        let summary = handle.await.unwrap().unwrap();
        assert_eq!(
            (summary.scanned, summary.cancelled, summary.succeeded),
            (1, 1, 0)
        );
        let cancelled = stored(&planner, "first").await;
        assert_eq!(cancelled.state, DreamState::Cancelled);
        assert!(cancelled.finished_at.is_some());
        assert!(
            lease_of(&planner, "first").await.is_none(),
            "the lease is released on the cancellation path"
        );
        let untouched = stored(&planner, "second").await;
        assert_eq!(untouched.state, DreamState::Pending);
        assert_eq!(untouched.attempts, 0);
        assert_eq!(consolidator.calls(), vec!["first".to_owned()]);
    }

    #[tokio::test]
    async fn a_consolidator_cancellation_error_stops_and_releases() {
        let repo = Arc::new(InMemoryDreamRepository::new());
        let consolidator = Arc::new(ScriptedConsolidator::new([Step::Cancelled]));
        let planner = planner(&repo, &consolidator, DreamConfig::default());
        repo.enqueue(job("first", 3)).await.unwrap();
        repo.enqueue(job("second", 3)).await.unwrap();

        let summary = planner
            .run_leased(
                ts(10),
                10,
                &CallOptions::new(),
                &options("worker-a", 30, 10),
            )
            .await
            .unwrap();
        assert_eq!((summary.scanned, summary.cancelled), (1, 1));
        assert_eq!(stored(&planner, "first").await.state, DreamState::Cancelled);
        assert!(
            lease_of(&planner, "first").await.is_none(),
            "the lease is released when the consolidator cancels"
        );
        assert_eq!(stored(&planner, "second").await.state, DreamState::Pending);
        assert_eq!(consolidator.calls(), vec!["first".to_owned()]);
    }

    #[tokio::test]
    async fn a_terminal_failure_releases_the_lease() {
        let repo = Arc::new(InMemoryDreamRepository::new());
        let consolidator = Arc::new(ScriptedConsolidator::new([Step::Failed]));
        let planner = planner(
            &repo,
            &consolidator,
            DreamConfig::default().with_max_attempts(1),
        );
        repo.enqueue(job("job-1", 1)).await.unwrap();

        let summary = planner
            .run_leased(
                ts(10),
                10,
                &CallOptions::new(),
                &options("worker-a", 30, 10),
            )
            .await
            .unwrap();
        assert_eq!((summary.scanned, summary.failed), (1, 1));
        let failed = stored(&planner, "job-1").await;
        assert_eq!(failed.state, DreamState::Failed);
        assert_eq!(failed.attempts, 1);
        assert!(failed.finished_at.is_some());
        assert!(failed
            .last_error
            .as_deref()
            .unwrap_or_default()
            .contains("scripted failure"));
        assert!(lease_of(&planner, "job-1").await.is_none());
    }

    #[tokio::test]
    async fn a_requeued_failure_releases_the_lease() {
        let repo = Arc::new(InMemoryDreamRepository::new());
        let consolidator = Arc::new(ScriptedConsolidator::new([Step::Failed]));
        let planner = planner(
            &repo,
            &consolidator,
            DreamConfig::default().with_max_attempts(2),
        );
        repo.enqueue(job("job-1", 2)).await.unwrap();

        let summary = planner
            .run_leased(
                ts(10),
                10,
                &CallOptions::new(),
                &options("worker-a", 30, 10),
            )
            .await
            .unwrap();
        assert_eq!((summary.scanned, summary.failed), (1, 1));
        let requeued = stored(&planner, "job-1").await;
        assert_eq!(requeued.state, DreamState::Pending);
        assert_eq!(requeued.attempts, 1);
        assert!(requeued.finished_at.is_none());
        assert!(lease_of(&planner, "job-1").await.is_none());
    }

    #[tokio::test]
    async fn a_non_leasing_repository_reports_unsupported() {
        let repo = Arc::new(PlainRepository::default());
        let consolidator = Arc::new(ScriptedConsolidator::new([]));
        let planner =
            DreamPlanner::new(repo.clone(), consolidator.clone(), DreamConfig::default()).unwrap();
        repo.enqueue(job("job-1", 3)).await.unwrap();

        let error = planner
            .run_leased(
                ts(10),
                10,
                &CallOptions::new(),
                &options("worker-a", 30, 10),
            )
            .await
            .unwrap_err();
        assert!(matches!(error, Error::Unsupported(_)), "{error:?}");
        assert!(error.to_string().contains("leased"));
        assert_eq!(consolidator.calls(), Vec::<String>::new());
        assert_eq!(stored(&planner, "job-1").await.state, DreamState::Pending);
    }

    #[tokio::test]
    async fn invalid_options_are_rejected_before_claiming() {
        let repo = Arc::new(InMemoryDreamRepository::new());
        let consolidator = Arc::new(ScriptedConsolidator::new([]));
        let planner = planner(&repo, &consolidator, DreamConfig::default());
        repo.enqueue(job("job-1", 3)).await.unwrap();

        let bad_heartbeat = options("worker-a", 30, 30);
        let error = planner
            .run_leased(ts(10), 10, &CallOptions::new(), &bad_heartbeat)
            .await
            .unwrap_err();
        assert!(matches!(error, Error::Validation(_)), "{error:?}");
        assert!(error.to_string().contains("heartbeat"));

        let empty_owner = options("", 30, 10);
        let error = planner
            .run_leased(ts(10), 10, &CallOptions::new(), &empty_owner)
            .await
            .unwrap_err();
        assert!(matches!(error, Error::Validation(_)), "{error:?}");

        let untouched = stored(&planner, "job-1").await;
        assert_eq!(untouched.state, DreamState::Pending);
        assert_eq!(untouched.attempts, 0);
        assert_eq!(consolidator.calls(), Vec::<String>::new());
    }

    #[tokio::test]
    async fn run_due_leased_uses_the_configured_batch_size() {
        let repo = Arc::new(InMemoryDreamRepository::new());
        let consolidator = Arc::new(ScriptedConsolidator::new([]));
        let planner = planner(
            &repo,
            &consolidator,
            DreamConfig::default().with_batch_size(2),
        );
        for index in 0..3 {
            repo.enqueue(job(&format!("job-{index}"), 3)).await.unwrap();
        }

        let summary = planner
            .run_due_leased(ts(10), &CallOptions::new(), &options("worker-a", 30, 10))
            .await
            .unwrap();
        assert_eq!(summary.scanned, 2);
        assert_eq!(consolidator.calls().len(), 2);
        assert_eq!(
            planner
                .repository()
                .list(&DreamFilter::new().with_state(DreamState::Pending))
                .await
                .unwrap()
                .len(),
            1
        );
    }
}
