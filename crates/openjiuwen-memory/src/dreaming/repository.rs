//! [`DreamRepository`]: the queue seam plus the in-memory backend.
//!
//! The repository is the contract the deferred scheduler and the stage-2 SQLite backend
//! implement. It keeps the claim transition atomic and deterministic: a single
//! `claim_next` call picks the head of the queue by `(priority DESC, created_at ASC,
//! id ASC)` and moves it from `Pending` to `Running` (or reports that there is nothing
//! eligible). The caller passes the clock in, so retries and backoff are reproducible in
//! tests instead of depending on wall-clock timing.
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::btree_map::Entry;
use std::collections::BTreeMap;

use crate::record::MemoryScope;

use super::model::{DreamJob, DreamKind, DreamOutcome, DreamState, DreamTrigger};

/// A conjunction of job predicates used by [`DreamRepository::list`].
///
/// Empty `ids`/`states`/`kinds`/`triggers` vectors mean "any", exactly like
/// [`MemoryFilter`](crate::record::MemoryFilter). Scope fields must equal the job's
/// value when set. Timestamp bounds are exclusive and compare against the job's
/// creation time.
///
/// Ordering and the `limit` are applied by the repository *after*
/// [`DreamFilter::matches`], so every backend returns the same order (`priority DESC,
/// created_at ASC, id ASC`) and honours the same truncation.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct DreamFilter {
    /// Accepted job ids; empty means every id.
    pub ids: Vec<String>,
    /// Accepted states; empty means every state.
    pub states: Vec<DreamState>,
    /// Accepted kinds; empty means every kind.
    pub kinds: Vec<DreamKind>,
    /// Accepted triggers; empty means every trigger.
    pub triggers: Vec<DreamTrigger>,
    /// Required owner user id.
    pub user_id: Option<String>,
    /// Required owner scope id.
    pub scope_id: Option<String>,
    /// Required session id (jobs without a session do not match).
    pub session_id: Option<String>,
    /// Required agent id (jobs without an agent do not match).
    pub agent_id: Option<String>,
    /// Exclusive lower bound on `created_at`.
    pub created_after: Option<DateTime<Utc>>,
    /// Exclusive upper bound on `created_at`.
    pub created_before: Option<DateTime<Utc>>,
    /// Maximum number of jobs to return after filtering and ordering.
    pub limit: Option<usize>,
}

impl DreamFilter {
    /// Builds an empty filter that matches every job.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the filter narrowed to `scope`, copying all four ids.
    ///
    /// Other clauses already set on `self` (ids, states, kinds, triggers, time bounds,
    /// limit) are preserved.
    #[must_use]
    pub fn for_scope(&self, scope: &MemoryScope) -> Self {
        Self {
            user_id: Some(scope.user_id.clone()),
            scope_id: Some(scope.scope_id.clone()),
            session_id: scope.session_id.clone(),
            agent_id: scope.agent_id.clone(),
            ..self.clone()
        }
    }

    /// Returns the filter with one more accepted state.
    #[must_use]
    pub fn with_state(mut self, state: DreamState) -> Self {
        self.states.push(state);
        self
    }

    /// Returns the filter with one more accepted kind.
    #[must_use]
    pub fn with_kind(mut self, kind: DreamKind) -> Self {
        self.kinds.push(kind);
        self
    }

    /// Returns the filter with one more accepted trigger.
    #[must_use]
    pub fn with_trigger(mut self, trigger: DreamTrigger) -> Self {
        self.triggers.push(trigger);
        self
    }

    /// Returns the filter limited to `limit` returned jobs.
    #[must_use]
    pub const fn with_limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Returns `true` when `job` satisfies every set clause.
    ///
    /// Only the predicates are applied here; ordering and the limit belong to the
    /// repository.
    #[must_use]
    pub fn matches(&self, job: &DreamJob) -> bool {
        if !self.ids.is_empty() && !self.ids.contains(&job.id) {
            return false;
        }
        if !self.states.is_empty() && !self.states.contains(&job.state) {
            return false;
        }
        if !self.kinds.is_empty() && !self.kinds.contains(&job.kind) {
            return false;
        }
        if !self.triggers.is_empty() && !self.triggers.contains(&job.trigger) {
            return false;
        }
        if let Some(user_id) = &self.user_id {
            if &job.scope.user_id != user_id {
                return false;
            }
        }
        if let Some(scope_id) = &self.scope_id {
            if &job.scope.scope_id != scope_id {
                return false;
            }
        }
        if let Some(session_id) = &self.session_id {
            if job.scope.session_id.as_ref() != Some(session_id) {
                return false;
            }
        }
        if let Some(agent_id) = &self.agent_id {
            if job.scope.agent_id.as_ref() != Some(agent_id) {
                return false;
            }
        }
        if let Some(created_after) = self.created_after {
            if job.created_at <= created_after {
                return false;
            }
        }
        if let Some(created_before) = self.created_before {
            if job.created_at >= created_before {
                return false;
            }
        }
        true
    }
}

/// Durable storage for dream jobs.
///
/// The contract every backend must satisfy:
///
/// - **`enqueue`** validates the job and inserts it verbatim. An id that already exists
///   is an [`Error::Validation`], never a silent overwrite.
/// - **`claim_next(now)`** performs one atomic `Pending` → `Running` transition for the
///   deterministic head of the queue, ordered `priority DESC, created_at ASC, id ASC`.
///   The claim sets `state = Running`, increments `attempts`, sets `started_at = now`,
///   and sets `updated_at = now`. A job is eligible when `state == Pending`,
///   `attempts < max_attempts`, and `started_at` is absent or strictly before `now`.
///   The clock is passed in (never read here), which makes runs reproducible and turns
///   the `started_at` timestamp into the retry backoff: a job retried at `now` cannot be
///   claimed by the same run. Concurrent callers must never both win; exactly one gets
///   `Some`.
/// - **`complete(job_id, outcome)`** validates the outcome, requires the stored job to
///   be `Running` (anything else is an [`Error::Validation`]; an unknown id is
///   [`Error::NotFound`]), and requires `outcome.job_id == job_id`. It stores the
///   outcome's state and metrics, sets `finished_at = updated_at = now`, and clears
///   `last_error` — that field only ever describes a failed attempt.
/// - **`fail(job_id, error)`** requires the stored job to be `Running`. It records
///   `last_error`. When `attempts >= max_attempts` the job becomes terminal `Failed`
///   with `finished_at = updated_at = now`; otherwise it returns to `Pending` with
///   `finished_at = None`, keeps `started_at` as the retry marker, and sets
///   `updated_at = now`. Attempts are counted by the claim, not by the failure.
/// - **`get`** returns `None` for an unknown id.
/// - **`list`** applies [`DreamFilter::matches`], then the shared ordering, then the
///   limit.
/// - **`count_by_state`** reports all six states, including zeros.
/// - **`delete_finished(older_than)`** removes terminal jobs whose finish time
///   (`finished_at`, falling back to `updated_at`) is strictly earlier than
///   `older_than`, and returns how many rows were removed.
///
/// Implementations must not hold a lock guard across `.await`.
#[async_trait]
pub trait DreamRepository: Send + Sync {
    /// Persists a new job and returns the stored value.
    async fn enqueue(&self, job: DreamJob) -> Result<DreamJob>;

    /// Claims the next eligible job, stamping it with `now`.
    async fn claim_next(&self, now: DateTime<Utc>) -> Result<Option<DreamJob>>;

    /// Marks a running job finished with `outcome`'s state and metrics.
    async fn complete(&self, job_id: &str, outcome: DreamOutcome) -> Result<DreamJob>;

    /// Records a failed attempt for a running job, requeueing or failing it.
    async fn fail(&self, job_id: &str, error: &str) -> Result<DreamJob>;

    /// Returns the job with `job_id`, if it exists.
    async fn get(&self, job_id: &str) -> Result<Option<DreamJob>>;

    /// Returns the jobs matching `filter`, ordered and truncated as documented.
    async fn list(&self, filter: &DreamFilter) -> Result<Vec<DreamJob>>;

    /// Counts jobs per state; every state is present.
    async fn count_by_state(&self) -> Result<BTreeMap<DreamState, usize>>;

    /// Removes terminal jobs finished strictly before `older_than`.
    async fn delete_finished(&self, older_than: DateTime<Utc>) -> Result<usize>;

    /// Removes terminal jobs finished more than `retention` before `now`.
    ///
    /// The default implementation computes the cutoff (saturating at
    /// [`DateTime::<Utc>::MIN_UTC`]) and delegates to [`Self::delete_finished`].
    async fn purge(&self, now: DateTime<Utc>, retention: chrono::Duration) -> Result<usize> {
        let cutoff = now
            .checked_sub_signed(retention)
            .unwrap_or(DateTime::<Utc>::MIN_UTC);
        self.delete_finished(cutoff).await
    }
}

/// The shared queue ordering: higher priority first, then oldest, then id.
fn queue_order(left: &DreamJob, right: &DreamJob) -> std::cmp::Ordering {
    right
        .priority
        .cmp(&left.priority)
        .then_with(|| left.created_at.cmp(&right.created_at))
        .then_with(|| left.id.cmp(&right.id))
}

/// Returns `true` when `job` may be claimed at `now`.
fn is_claimable(job: &DreamJob, now: DateTime<Utc>) -> bool {
    job.state == DreamState::Pending
        && job.attempts < job.max_attempts
        && job.started_at.is_none_or(|started| started < now)
}

/// A process-local [`DreamRepository`] backed by a `BTreeMap`.
///
/// Suitable for tests, single-process applications, and the default build (which has no
/// SQLite feature). Jobs are lost on process exit; the stage-2 backend makes them
/// durable.
#[derive(Debug, Default)]
pub struct InMemoryDreamRepository {
    jobs: parking_lot::RwLock<BTreeMap<String, DreamJob>>,
}

impl InMemoryDreamRepository {
    /// Builds an empty repository.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the number of stored jobs.
    #[must_use]
    pub fn len(&self) -> usize {
        self.jobs.read().len()
    }

    /// Returns `true` when no job is stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.jobs.read().is_empty()
    }
}

/// Looks up a job by id, mapping absence to [`Error::NotFound`].
fn stored_job<'a>(
    jobs: &'a mut BTreeMap<String, DreamJob>,
    job_id: &str,
) -> Result<&'a mut DreamJob> {
    jobs.get_mut(job_id)
        .ok_or_else(|| Error::not_found(format!("dream job `{job_id}` does not exist")))
}

#[async_trait]
impl DreamRepository for InMemoryDreamRepository {
    async fn enqueue(&self, job: DreamJob) -> Result<DreamJob> {
        job.validate()?;
        let mut jobs = self.jobs.write();
        match jobs.entry(job.id.clone()) {
            Entry::Occupied(_) => Err(Error::validation(format!(
                "dream job `{}` already exists",
                job.id
            ))),
            Entry::Vacant(slot) => {
                slot.insert(job.clone());
                tracing::debug!(
                    target: "openjiuwen_memory::dreaming::repository",
                    job_id = %job.id,
                    kind = job.kind.as_str(),
                    "enqueued dream job"
                );
                Ok(job)
            }
        }
    }

    async fn claim_next(&self, now: DateTime<Utc>) -> Result<Option<DreamJob>> {
        let mut jobs = self.jobs.write();
        let mut eligible: Vec<&DreamJob> =
            jobs.values().filter(|job| is_claimable(job, now)).collect();
        if eligible.is_empty() {
            return Ok(None);
        }
        eligible.sort_by(|left, right| queue_order(left, right));
        let job_id = eligible[0].id.clone();
        drop(eligible);
        let job = stored_job(&mut jobs, &job_id)?;
        job.state = DreamState::Running;
        job.attempts = job.attempts.saturating_add(1);
        job.started_at = Some(now);
        job.updated_at = now;
        let claimed = job.clone();
        tracing::debug!(
            target: "openjiuwen_memory::dreaming::repository",
            job_id = %claimed.id,
            attempts = claimed.attempts,
            "claimed dream job"
        );
        Ok(Some(claimed))
    }

    async fn complete(&self, job_id: &str, outcome: DreamOutcome) -> Result<DreamJob> {
        outcome.validate()?;
        let mut jobs = self.jobs.write();
        let job = stored_job(&mut jobs, job_id)?;
        if job.state != DreamState::Running {
            return Err(Error::validation(format!(
                "dream job `{job_id}` is `{}` but must be `running` to complete",
                job.state
            )));
        }
        if outcome.job_id != job_id {
            return Err(Error::validation(format!(
                "dream outcome job id `{}` does not match `{job_id}`",
                outcome.job_id
            )));
        }
        let now = Utc::now();
        job.state = outcome.state;
        job.metrics = outcome.metrics;
        job.finished_at = Some(now);
        job.updated_at = now;
        job.last_error = None;
        Ok(job.clone())
    }

    async fn fail(&self, job_id: &str, error: &str) -> Result<DreamJob> {
        let mut jobs = self.jobs.write();
        let job = stored_job(&mut jobs, job_id)?;
        if job.state != DreamState::Running {
            return Err(Error::validation(format!(
                "dream job `{job_id}` is `{}` but must be `running` to fail",
                job.state
            )));
        }
        let now = Utc::now();
        job.last_error = Some(error.to_owned());
        job.updated_at = now;
        if job.attempts >= job.max_attempts {
            job.state = DreamState::Failed;
            job.finished_at = Some(now);
        } else {
            job.state = DreamState::Pending;
            job.finished_at = None;
            // `started_at` is preserved: it is the retry marker that backs off the next
            // claim until the caller's clock has moved past it.
        }
        Ok(job.clone())
    }

    async fn get(&self, job_id: &str) -> Result<Option<DreamJob>> {
        Ok(self.jobs.read().get(job_id).cloned())
    }

    async fn list(&self, filter: &DreamFilter) -> Result<Vec<DreamJob>> {
        let jobs = self.jobs.read();
        let mut matched: Vec<DreamJob> = jobs
            .values()
            .filter(|job| filter.matches(job))
            .cloned()
            .collect();
        matched.sort_by(queue_order);
        if let Some(limit) = filter.limit {
            matched.truncate(limit);
        }
        Ok(matched)
    }

    async fn count_by_state(&self) -> Result<BTreeMap<DreamState, usize>> {
        let jobs = self.jobs.read();
        let mut counts: BTreeMap<DreamState, usize> = DreamState::ALL
            .into_iter()
            .map(|state| (state, 0))
            .collect();
        for job in jobs.values() {
            if let Some(count) = counts.get_mut(&job.state) {
                *count += 1;
            }
        }
        Ok(counts)
    }

    async fn delete_finished(&self, older_than: DateTime<Utc>) -> Result<usize> {
        let mut jobs = self.jobs.write();
        let before = jobs.len();
        jobs.retain(|_, job| {
            if !job.state.is_terminal() {
                return true;
            }
            let finished = job.finished_at.unwrap_or(job.updated_at);
            finished >= older_than
        });
        Ok(before - jobs.len())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::dreaming::DreamMetrics;
    use chrono::Duration;
    use serde_json::json;

    fn ts(offset_secs: i64) -> DateTime<Utc> {
        DateTime::from_timestamp(1_700_000_000 + offset_secs, 0).unwrap()
    }

    fn scope() -> MemoryScope {
        MemoryScope::new("alice", "ws")
    }

    fn job(id: &str, priority: i32, created_secs: i64) -> DreamJob {
        DreamJob::new(
            scope(),
            DreamKind::Consolidation,
            DreamTrigger::Manual,
            ts(created_secs),
        )
        .with_id(id)
        .with_priority(priority)
    }

    async fn claimed(repo: &InMemoryDreamRepository, now: DateTime<Utc>) -> DreamJob {
        repo.claim_next(now).await.unwrap().unwrap()
    }

    #[test]
    fn filter_matches_scope_kind_state_and_bounds() {
        let base = job("job-1", 0, 0);
        assert!(DreamFilter::new().matches(&base));
        assert!(DreamFilter::new().for_scope(&scope()).matches(&base));
        assert!(!DreamFilter::new()
            .for_scope(&MemoryScope::new("bob", "ws"))
            .matches(&base));
        assert!(!DreamFilter::new()
            .for_scope(&MemoryScope::new("alice", "other"))
            .matches(&base));
        assert!(!DreamFilter::new()
            .for_scope(&MemoryScope::new("alice", "ws").session("s1"))
            .matches(&base));
        assert!(DreamFilter::new()
            .with_state(DreamState::Pending)
            .with_kind(DreamKind::Consolidation)
            .with_trigger(DreamTrigger::Manual)
            .matches(&base));
        assert!(!DreamFilter::new()
            .with_state(DreamState::Running)
            .matches(&base));
        assert!(!DreamFilter::new()
            .with_kind(DreamKind::Dedupe)
            .matches(&base));
        assert!(!DreamFilter::new()
            .with_trigger(DreamTrigger::Idle)
            .matches(&base));

        let mut filter = DreamFilter::new();
        filter.ids = vec!["job-2".to_owned()];
        assert!(!filter.matches(&base));

        let mut filter = DreamFilter::new();
        filter.created_after = Some(ts(0));
        assert!(!filter.matches(&base));
        filter.created_after = Some(ts(-1));
        assert!(filter.matches(&base));

        let mut filter = DreamFilter::new();
        filter.created_before = Some(ts(0));
        assert!(!filter.matches(&base));
        filter.created_before = Some(ts(1));
        assert!(filter.matches(&base));
    }

    #[test]
    fn filter_for_scope_preserves_other_clauses() {
        let filter = DreamFilter::new()
            .with_state(DreamState::Succeeded)
            .with_limit(3)
            .for_scope(&MemoryScope::new("alice", "ws").session("s1"));
        assert_eq!(filter.user_id.as_deref(), Some("alice"));
        assert_eq!(filter.scope_id.as_deref(), Some("ws"));
        assert_eq!(filter.session_id.as_deref(), Some("s1"));
        assert_eq!(filter.agent_id, None);
        assert_eq!(filter.states, vec![DreamState::Succeeded]);
        assert_eq!(filter.limit, Some(3));
    }

    #[test]
    fn filter_serde_defaults() {
        let filter: DreamFilter = serde_json::from_value(json!({})).unwrap();
        assert_eq!(filter, DreamFilter::default());
        let value =
            serde_json::to_value(DreamFilter::new().with_state(DreamState::Failed)).unwrap();
        assert_eq!(value["states"], json!(["failed"]));
        assert_eq!(
            serde_json::from_value::<DreamFilter>(value).unwrap(),
            DreamFilter::new().with_state(DreamState::Failed)
        );
    }

    #[tokio::test]
    async fn enqueue_get_round_trip_and_duplicate_rejected() {
        let repo = InMemoryDreamRepository::new();
        assert!(repo.is_empty());
        let stored = repo.enqueue(job("job-1", 0, 0)).await.unwrap();
        assert_eq!(stored.id, "job-1");
        assert_eq!(repo.len(), 1);
        assert_eq!(repo.get("job-1").await.unwrap(), Some(stored.clone()));
        assert_eq!(repo.get("missing").await.unwrap(), None);

        let err = repo.enqueue(job("job-1", 9, 9)).await.unwrap_err();
        assert!(matches!(err, Error::Validation(_)));
        assert!(err.to_string().contains("already exists"));
        assert_eq!(repo.len(), 1);
    }

    #[tokio::test]
    async fn enqueue_validates_the_job() {
        let repo = InMemoryDreamRepository::new();
        let err = repo
            .enqueue(job("job-1", 0, 0).with_max_attempts(0))
            .await
            .unwrap_err();
        assert!(matches!(err, Error::Validation(_)));
        assert!(repo.is_empty());
    }

    #[tokio::test]
    async fn claim_returns_none_when_empty() {
        let repo = InMemoryDreamRepository::new();
        assert_eq!(repo.claim_next(ts(0)).await.unwrap(), None);
    }

    #[tokio::test]
    async fn claim_order_is_priority_then_created_at_then_id() {
        let repo = InMemoryDreamRepository::new();
        repo.enqueue(job("b", 0, 5)).await.unwrap();
        repo.enqueue(job("a", 0, 5)).await.unwrap();
        repo.enqueue(job("early", 0, 1)).await.unwrap();
        repo.enqueue(job("urgent", 5, 9)).await.unwrap();

        let order: Vec<String> = {
            let mut ids = Vec::new();
            while let Some(job) = repo.claim_next(ts(10)).await.unwrap() {
                ids.push(job.id);
            }
            ids
        };
        assert_eq!(order, vec!["urgent", "early", "a", "b"]);
    }

    #[tokio::test]
    async fn claim_sets_state_attempts_and_started_at() {
        let repo = InMemoryDreamRepository::new();
        repo.enqueue(job("job-1", 0, 0)).await.unwrap();
        let claimed = claimed(&repo, ts(10)).await;
        assert_eq!(claimed.state, DreamState::Running);
        assert_eq!(claimed.attempts, 1);
        assert_eq!(claimed.started_at, Some(ts(10)));
        assert_eq!(claimed.updated_at, ts(10));
        assert_eq!(claimed.finished_at, None);
        assert_eq!(
            repo.get("job-1").await.unwrap().unwrap(),
            claimed,
            "the stored job reflects the claim"
        );
    }

    #[tokio::test]
    async fn claim_skips_non_pending_and_exhausted_jobs() {
        let repo = InMemoryDreamRepository::new();
        let mut succeeded = job("done", 9, 0);
        succeeded.state = DreamState::Succeeded;
        succeeded.started_at = Some(ts(1));
        succeeded.finished_at = Some(ts(2));
        succeeded.attempts = 1;
        repo.enqueue(succeeded).await.unwrap();
        repo.enqueue(job("spent", 9, 0).with_max_attempts(1))
            .await
            .unwrap();
        let exhausted = claimed(&repo, ts(10)).await;
        assert_eq!(exhausted.id, "spent");
        assert_eq!(exhausted.attempts, 1);
        // "spent" is now Running with attempts == max_attempts; "done" is terminal.
        assert_eq!(repo.claim_next(ts(11)).await.unwrap(), None);
    }

    #[tokio::test]
    async fn retry_eligibility_uses_started_at_as_backoff() {
        let repo = InMemoryDreamRepository::new();
        repo.enqueue(job("job-1", 0, 0)).await.unwrap();
        let first = claimed(&repo, ts(10)).await;
        repo.fail(&first.id, "boom").await.unwrap();

        // A retried job cannot be re-claimed at the same instant.
        assert_eq!(repo.claim_next(ts(10)).await.unwrap(), None);
        // One tick later the retry marker no longer blocks it.
        let retried = claimed(&repo, ts(11)).await;
        assert_eq!(retried.attempts, 2);
        assert_eq!(retried.started_at, Some(ts(11)));
    }

    #[tokio::test]
    async fn complete_unknown_is_not_found() {
        let repo = InMemoryDreamRepository::new();
        let err = repo
            .complete(
                "missing",
                DreamOutcome::succeeded("missing", DreamMetrics::default()),
            )
            .await
            .unwrap_err();
        assert!(matches!(err, Error::NotFound(_)));
    }

    #[tokio::test]
    async fn complete_on_pending_is_validation() {
        let repo = InMemoryDreamRepository::new();
        repo.enqueue(job("job-1", 0, 0)).await.unwrap();
        let err = repo
            .complete(
                "job-1",
                DreamOutcome::succeeded("job-1", DreamMetrics::default()),
            )
            .await
            .unwrap_err();
        assert!(matches!(err, Error::Validation(_)));
        assert!(err.to_string().contains("must be `running`"));
    }

    #[tokio::test]
    async fn complete_rejects_outcome_for_another_job() {
        let repo = InMemoryDreamRepository::new();
        repo.enqueue(job("job-1", 0, 0)).await.unwrap();
        claimed(&repo, ts(1)).await;
        let err = repo
            .complete("job-1", DreamOutcome::cancelled("other"))
            .await
            .unwrap_err();
        assert!(matches!(err, Error::Validation(_)));
        assert!(err.to_string().contains("does not match"));
    }

    #[tokio::test]
    async fn complete_sets_finished_metrics_and_clears_last_error() {
        let repo = InMemoryDreamRepository::new();
        let mut job = job("job-1", 0, 0);
        job.state = DreamState::Running;
        job.attempts = 1;
        job.started_at = Some(ts(0));
        job.last_error = Some("previous failure".to_owned());
        repo.enqueue(job).await.unwrap();

        let metrics = DreamMetrics {
            records_scanned: 4,
            records_merged: 2,
            ..DreamMetrics::default()
        };
        let completed = repo
            .complete(
                "job-1",
                DreamOutcome::succeeded("job-1", metrics).with_note("done"),
            )
            .await
            .unwrap();
        assert_eq!(completed.state, DreamState::Succeeded);
        assert_eq!(completed.metrics, metrics);
        assert!(completed.finished_at.is_some());
        assert!(completed.updated_at >= completed.created_at);
        assert_eq!(completed.last_error, None);
        assert!(completed.validate().is_ok());
        assert_eq!(completed.attempts, 1);
    }

    #[tokio::test]
    async fn fail_requeues_with_last_error_and_preserved_started_at() {
        let repo = InMemoryDreamRepository::new();
        repo.enqueue(job("job-1", 0, 0)).await.unwrap();
        let claimed = claimed(&repo, ts(1)).await;
        let failed = repo.fail(&claimed.id, "boom").await.unwrap();
        assert_eq!(failed.state, DreamState::Pending);
        assert_eq!(failed.attempts, 1);
        assert_eq!(failed.started_at, Some(ts(1)));
        assert_eq!(failed.finished_at, None);
        assert_eq!(failed.last_error.as_deref(), Some("boom"));
        assert!(failed.validate().is_ok());
    }

    #[tokio::test]
    async fn fail_at_max_attempts_becomes_terminal() {
        let repo = InMemoryDreamRepository::new();
        repo.enqueue(job("job-1", 0, 0).with_max_attempts(1))
            .await
            .unwrap();
        let claimed = claimed(&repo, ts(1)).await;
        let failed = repo.fail(&claimed.id, "boom").await.unwrap();
        assert_eq!(failed.state, DreamState::Failed);
        assert!(failed.finished_at.is_some());
        assert!(failed.validate().is_ok());
        assert_eq!(repo.claim_next(ts(99)).await.unwrap(), None);
    }

    #[tokio::test]
    async fn fail_on_non_running_job_is_rejected() {
        let repo = InMemoryDreamRepository::new();
        repo.enqueue(job("job-1", 0, 0)).await.unwrap();
        assert!(matches!(
            repo.fail("job-1", "boom").await.unwrap_err(),
            Error::Validation(_)
        ));
        assert!(matches!(
            repo.fail("missing", "boom").await.unwrap_err(),
            Error::NotFound(_)
        ));
    }

    #[tokio::test]
    async fn count_by_state_includes_all_six_states() {
        let repo = InMemoryDreamRepository::new();
        repo.enqueue(job("pending", 0, 0)).await.unwrap();
        repo.enqueue(job("running", 0, 0)).await.unwrap();
        claimed(&repo, ts(1)).await;
        let counts = repo.count_by_state().await.unwrap();
        assert_eq!(counts.len(), DreamState::ALL.len());
        for state in DreamState::ALL {
            assert!(counts.contains_key(&state), "missing {state}");
        }
        assert_eq!(counts[&DreamState::Pending], 1);
        assert_eq!(counts[&DreamState::Running], 1);
        assert_eq!(counts[&DreamState::Succeeded], 0);
    }

    #[tokio::test]
    async fn list_filters_orders_and_limits() {
        let repo = InMemoryDreamRepository::new();
        repo.enqueue(job("low", 0, 1)).await.unwrap();
        repo.enqueue(job("high", 5, 5)).await.unwrap();
        let mut other_kind = job("dedupe", 0, 2);
        other_kind.kind = DreamKind::Dedupe;
        repo.enqueue(other_kind).await.unwrap();
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
        assert_eq!(limited.len(), 2);
        assert_eq!(limited[0].id, "high");

        let after_claim = {
            claimed(&repo, ts(10)).await;
            repo.list(&DreamFilter::new().with_state(DreamState::Running))
                .await
                .unwrap()
        };
        assert_eq!(after_claim.len(), 1);
        assert_eq!(after_claim[0].id, "high");
    }

    #[tokio::test]
    async fn delete_finished_removes_only_old_terminal_jobs() {
        let repo = InMemoryDreamRepository::new();
        let mut old = job("old", 0, 0);
        old.state = DreamState::Succeeded;
        old.attempts = 1;
        old.started_at = Some(ts(0));
        old.finished_at = Some(ts(10));
        old.updated_at = ts(10);
        repo.enqueue(old).await.unwrap();
        repo.enqueue(job("pending", 0, 0)).await.unwrap();

        assert_eq!(repo.delete_finished(ts(5)).await.unwrap(), 0);
        assert_eq!(repo.delete_finished(ts(11)).await.unwrap(), 1);
        assert_eq!(repo.len(), 1);
        assert!(repo.get("old").await.unwrap().is_none());
        assert!(repo.get("pending").await.unwrap().is_some());
    }

    #[tokio::test]
    async fn purge_uses_retention_window() {
        let repo = InMemoryDreamRepository::new();
        repo.enqueue(job("job-1", 0, 0)).await.unwrap();
        claimed(&repo, ts(1)).await;
        repo.complete("job-1", DreamOutcome::cancelled("job-1"))
            .await
            .unwrap();
        let now = Utc::now();
        assert_eq!(repo.purge(now, Duration::days(7)).await.unwrap(), 0);
        assert_eq!(
            repo.purge(now + Duration::days(8), Duration::days(7))
                .await
                .unwrap(),
            1
        );
        assert!(repo.is_empty());
    }
}
