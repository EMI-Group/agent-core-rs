//! [`TaskBoard`]: every task-board operation over one team's persisted state.
//!
//! Python parity source: `TeamTaskManager` of `openjiuwen/agent_teams/tools/task_manager.py` and
//! the atomic mutations of `tools/database/task_dao.py`. The Python class is also the tool surface
//! (`create_task`, `claim_task`, `complete_task`, …); here the board is the domain layer and the
//! tool wrappers live in [`crate::runtime`] — the board takes no LLM-facing arguments and renders no
//! prose, it only reads and writes rows and reports what changed.
//!
//! # Write order and atomicity
//!
//! * **Creation** validates everything first (existing ids, `depended_by` targets, cycles), then
//!   writes the dependency edges, then the row. A failing row write therefore leaves only edges that
//!   point at a task that does not exist — which the cycle detector and the blocking rule treat as
//!   an unresolved dependency, never as a half-created task. A cycle is rejected *before* any write,
//!   so a rejected graph leaves no rows and no edges.
//! * **Mutations of an existing task** write exactly one row ([`TaskBoard::persist`]), so a status
//!   change, an assignee change, a note and the revision marker land together. Unblocking
//!   dependents happens after that row write and is idempotent: a partial pass can be re-run without
//!   changing the outcome.
//! * **Review votes** are inserted before the status row, because the vote is a fact about the round
//!   and stays valid even if the row write fails (the task is still `in_review`, the vote is simply
//!   counted again by the next tally).
//!
//! Status legality is the board's decision, not the store's: the board writes whole rows through
//! [`crate::storage::TeamStore::insert_task`] so every field of a transition lands at once, and it
//! consults [`TaskStatus::can_transition_to`] (the same [`crate::schema::TASK_TRANSITIONS`] table the
//! store's compare-and-set path uses) before doing so. A plan-mode completion is the only path that
//! needs two table steps: `planning -> in_progress -> completed` is applied as one row write, so the
//! intermediate state is never observable.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::Arc;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::utils::id::uuid_hex;

use crate::schema::{
    EventPayload, MemberMode, MemberStatus, ReviewVote, TaskCreatedEvent, TaskDetail, TaskFilter,
    TaskGraphSpec, TaskRecord, TaskReviewVoteEvent, TaskRevision, TaskStatus,
    TaskStatusChangedEvent, TaskSummary, TaskVerifiedEvent, Verdict,
};
use crate::storage::TeamStore;

use super::graph::{depth_of, detect_cycle};
use super::review::{tally_with, ReviewTally};
use super::{missing_task, TaskBoardLimits, TaskMutation, TaskOpResult};

/// Reason returned when the assignee has not seen the task's current content revision.
///
/// Stable string: it is shown to the model, which is expected to re-read the task and retry.
pub const REASON_STALE_CONTENT: &str = "content changed; re-read the task before completing";

/// The task board of one team.
///
/// The board is a thin, synchronous domain layer over [`TeamStore`]: construct it per runtime (it
/// holds no cache and no lock) and share it behind an `Arc`. Two boards over the same
/// `(store, team_name)` observe the same state — the store is the single source of truth.
#[derive(Clone)]
pub struct TaskBoard {
    store: Arc<dyn TeamStore>,
    team_name: String,
    limits: TaskBoardLimits,
}

/// Renders the board's identity and limits; the store is shown by its backend kind because
/// `dyn TeamStore` is not `Debug`.
impl fmt::Debug for TaskBoard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TaskBoard")
            .field("store", &self.store.name())
            .field("team_name", &self.team_name)
            .field("limits", &self.limits)
            .finish()
    }
}

impl TaskBoard {
    /// Builds a board for `team_name` over `store`.
    #[must_use]
    pub fn new(
        store: Arc<dyn TeamStore>,
        team_name: impl Into<String>,
        limits: TaskBoardLimits,
    ) -> Self {
        Self {
            store,
            team_name: team_name.into(),
            limits,
        }
    }

    /// Builds a board with the default [`TaskBoardLimits`].
    #[must_use]
    pub fn with_defaults(store: Arc<dyn TeamStore>, team_name: impl Into<String>) -> Self {
        Self::new(store, team_name, TaskBoardLimits::default())
    }

    /// Returns the store the board reads and writes.
    #[must_use]
    pub fn store(&self) -> &Arc<dyn TeamStore> {
        &self.store
    }

    /// Returns the team the board belongs to.
    #[must_use]
    pub fn team_name(&self) -> &str {
        &self.team_name
    }

    /// Returns the limits the board applies to new tasks.
    #[must_use]
    pub const fn limits(&self) -> TaskBoardLimits {
        self.limits
    }

    // -------------------------------------------------------------------------------------------
    // Creation
    // -------------------------------------------------------------------------------------------

    /// Creates one top-level task, its dependency edges and its events.
    ///
    /// `spec.task_id` is used verbatim when set, otherwise an id is minted as
    /// `task_<32 hex chars>`. A task with an unresolved dependency is created `blocked` (an id that
    /// does not exist yet counts as unresolved, so a forward reference blocks exactly like a
    /// pending task); a task whose dependencies are all terminal is created `pending`.
    ///
    /// # Errors
    ///
    /// * [`Error::Validation`] when `spec.task_id` is already on the board, when a `depended_by`
    ///   target does not exist, or when the resulting graph would contain a cycle
    ///   (`Circular dependency: a -> b -> a`). Nothing is written in any of those cases.
    /// * Whatever the store fails with.
    pub async fn create_task(&self, spec: TaskGraphSpec) -> Result<TaskMutation> {
        self.create_spec(spec, &[]).await
    }

    /// Creates a subtask of `parent_id` and parks the parent while the child is open.
    ///
    /// The new task carries `parent_task_id = parent_id` and, unless the spec sets one, the parent's
    /// current revision as `parent_revision` (the derivations a stale subtask is detected against).
    /// A `depends_on` edge is added from the parent to the child, and the parent is parked at
    /// `blocked` — the same rule as Python's `suspend_parent` (`F_128`): the parent is not done, its
    /// work is delegated, so it must not be resumed until every subtask is terminal. The ordinary
    /// unblock pass moves it back to `pending` once the child completes or is cancelled.
    ///
    /// # Errors
    ///
    /// * [`Error::NotFound`] when the parent task does not exist.
    /// * [`Error::Validation`] when the subtask would nest deeper than
    ///   [`TaskBoardLimits::max_subtask_depth`] (`subtask depth 2 exceeds max_subtask_depth 1`), or
    ///   for every reason [`TaskBoard::create_task`] rejects a spec.
    /// * Whatever the store fails with.
    pub async fn create_subtask(
        &self,
        parent_id: &str,
        spec: TaskGraphSpec,
    ) -> Result<TaskMutation> {
        let parent = self.require(parent_id).await?;
        let depth = self.subtask_depth(&parent).await?;
        let max_depth = self.limits.max_subtask_depth;
        if depth > max_depth {
            return Err(Error::validation(format!(
                "subtask depth {depth} exceeds max_subtask_depth {max_depth}"
            )));
        }
        let mut spec = spec;
        spec.parent_task_id = Some(parent.task_id.clone());
        spec.parent_revision = spec.parent_revision.or(Some(parent.revision));
        let edge = (parent.task_id.clone(), String::new());
        let mut mutation = self.create_spec(spec, &[edge]).await?;
        // `create_spec` cannot know the child's id up front, so the parent edge is written here.
        self.store
            .add_dependency(&self.team_name, &parent.task_id, &mutation.task.task_id)
            .await?;
        if !parent.is_terminal() && parent.status != TaskStatus::Blocked {
            let mut parked = parent;
            parked.status = TaskStatus::Blocked;
            let parked = self.persist(parked).await?;
            mutation.events.push(self.status_event(&parked));
        }
        Ok(mutation)
    }

    /// Creates one task from `spec` plus the caller's extra `(dependent, depends_on)` edges.
    async fn create_spec(
        &self,
        spec: TaskGraphSpec,
        extra_edges: &[(String, String)],
    ) -> Result<TaskMutation> {
        let task_id = match spec.task_id.as_deref().map(str::trim) {
            Some(id) if !id.is_empty() => id.to_owned(),
            _ => format!("task_{}", uuid_hex()),
        };
        let snapshot = self.snapshot().await?;
        if snapshot.tasks.contains_key(&task_id) {
            return Err(Error::validation(format!(
                "task '{task_id}' already exists in team '{}'",
                self.team_name
            )));
        }
        for dependent in &spec.depended_by {
            if !snapshot.tasks.contains_key(dependent) {
                return Err(Error::validation(format!(
                    "depended_by task '{dependent}' does not exist"
                )));
            }
        }
        let mut edges = snapshot.edges.clone();
        let mut dependencies: BTreeSet<String> = BTreeSet::new();
        for dependency in &spec.depends_on {
            let dependency = dependency.trim();
            if dependency.is_empty() {
                return Err(Error::validation("depends_on must not contain an empty id"));
            }
            dependencies.insert(dependency.to_owned());
        }
        edges.insert(task_id.clone(), dependencies.clone());
        for dependent in &spec.depended_by {
            edges
                .entry(dependent.clone())
                .or_default()
                .insert(task_id.clone());
        }
        for (dependent, depends_on) in extra_edges {
            if depends_on.is_empty() {
                edges.entry(dependent.clone()).or_default();
            } else {
                edges
                    .entry(dependent.clone())
                    .or_default()
                    .insert(depends_on.clone());
            }
        }
        if let Some(cycle) = detect_cycle(&edges) {
            return Err(Error::validation(format!(
                "Circular dependency: {}",
                cycle.join(" -> ")
            )));
        }
        let blocked = dependencies
            .iter()
            .any(|dependency| !snapshot.is_terminal(dependency));
        let mut record = TaskRecord::new(&self.team_name, &task_id, &spec.title, &spec.content);
        record.status = if blocked {
            TaskStatus::Blocked
        } else {
            TaskStatus::Pending
        };
        record.assignee = spec
            .assignee
            .as_deref()
            .map(str::trim)
            .filter(|name| !name.is_empty())
            .map(str::to_owned);
        record.reviewers = distinct(&spec.reviewers);
        record.max_review_rounds = spec
            .max_review_rounds
            .unwrap_or(self.limits.default_max_review_rounds);
        record.parent_task_id = spec.parent_task_id.clone();
        record.parent_revision = spec.parent_revision;
        record.created_by = spec.created_by.clone();
        for dependency in &dependencies {
            self.store
                .add_dependency(&self.team_name, &task_id, dependency)
                .await?;
        }
        for dependent in &spec.depended_by {
            self.store
                .add_dependency(&self.team_name, dependent, &task_id)
                .await?;
        }
        for (dependent, depends_on) in extra_edges {
            if !depends_on.is_empty() {
                self.store
                    .add_dependency(&self.team_name, dependent, depends_on)
                    .await?;
            }
        }
        let record = self.persist(record).await?;
        let events = vec![self.created_event(&record), self.status_event(&record)];
        Ok(TaskMutation::new(record, events))
    }

    // -------------------------------------------------------------------------------------------
    // Ownership: claim, assign, reassign, plan
    // -------------------------------------------------------------------------------------------

    /// Claims `task_id` for `member`, moving it `pending -> in_progress`.
    ///
    /// A re-claim by the current owner is a no-op success that also refreshes
    /// [`TaskRecord::observed_revision`]: claiming is how a member gets the content, so it counts as
    /// having read the current revision.
    ///
    /// # Errors
    ///
    /// [`Error::NotFound`] when the task does not exist, and whatever the store fails with. A refusal
    /// (already claimed, not `pending`) is `Ok` with `ok = false` and a stable reason.
    pub async fn claim(&self, task_id: &str, member: &str) -> Result<TaskOpResult> {
        let mut record = self.require(task_id).await?;
        if record.assignee.as_deref() == Some(member) {
            if record.status == TaskStatus::InProgress {
                record.observed_revision = record.revision;
                self.persist(record).await?;
                return Ok(TaskOpResult::ok());
            }
            if record.status != TaskStatus::Pending {
                return Ok(TaskOpResult::failed(format!(
                    "task '{task_id}' cannot be claimed from status '{}' (only pending tasks are claimable)",
                    record.status.as_str()
                )));
            }
        } else if let Some(owner) = record.assignee.as_deref() {
            return Ok(TaskOpResult::failed(format!(
                "task '{task_id}' is already claimed by '{owner}'"
            )));
        }
        if record.status != TaskStatus::Pending {
            return Ok(TaskOpResult::failed(format!(
                "task '{task_id}' cannot be claimed from status '{}' (only pending tasks are claimable)",
                record.status.as_str()
            )));
        }
        record.status = TaskStatus::InProgress;
        record.assignee = Some(member.to_owned());
        record.observed_revision = record.revision;
        self.persist(record).await?;
        Ok(TaskOpResult::ok())
    }

    /// Assigns `task_id` to `assignee` on the leader's behalf and appends a change note.
    ///
    /// The task keeps its status except for `pending`, which enters the owner's entry gate: a
    /// `build_mode` member starts executing (`in_progress`), a `plan_mode` member lands in
    /// `planning`. A `blocked` task keeps its status (it is still waiting on a dependency) and only
    /// gains the assignee.
    ///
    /// # Errors
    ///
    /// [`Error::NotFound`] when the task does not exist. Refusals — terminal task, a different
    /// current owner, an unknown member — are `Ok` with `ok = false`.
    pub async fn assign(
        &self,
        task_id: &str,
        assignee: &str,
        reason: Option<&str>,
    ) -> Result<TaskOpResult> {
        let mut record = self.require(task_id).await?;
        if record.is_terminal() {
            return Ok(TaskOpResult::failed(format!(
                "task '{task_id}' cannot be assigned from status '{}'",
                record.status.as_str()
            )));
        }
        if let Some(owner) = record.assignee.as_deref() {
            if owner != assignee {
                return Ok(TaskOpResult::failed(format!(
                    "task '{task_id}' is already claimed by '{owner}'; reset the task before reassigning to '{assignee}'"
                )));
            }
        }
        let entry_status = match self.member_mode(assignee).await? {
            Some(MemberMode::PlanMode) => TaskStatus::Planning,
            Some(MemberMode::BuildMode) => TaskStatus::InProgress,
            None => {
                return Ok(TaskOpResult::failed(format!(
                    "member '{assignee}' not found in team '{}'",
                    self.team_name
                )))
            }
        };
        let already_there = record.assignee.as_deref() == Some(assignee)
            && (record.status == entry_status || record.status == TaskStatus::Blocked);
        record.assignee = Some(assignee.to_owned());
        record.observed_revision = record.revision;
        if !already_there && record.status == TaskStatus::Pending {
            record.status = entry_status;
        }
        if let Some(reason) = reason.map(str::trim).filter(|note| !note.is_empty()) {
            record.push_change_note(
                TaskRevision {
                    revision: record.revision,
                    note: format!("assigned to {assignee}: {reason}"),
                    by: None,
                }
                .to_note(),
            );
        }
        self.persist(record).await?;
        Ok(TaskOpResult::ok())
    }

    /// Hands `task_id` from `from` to `to` without bouncing through `pending`.
    ///
    /// Only the assignee changes: the status is preserved, so an in-flight task does not flash through
    /// the claimable pool and no idle member is woken by a spurious release.
    ///
    /// # Errors
    ///
    /// [`Error::NotFound`] when the task does not exist. Refusals — not held by `from`, a status that
    /// does not allow handing over, an unknown member — are `Ok` with `ok = false`.
    pub async fn reassign(&self, task_id: &str, from: &str, to: &str) -> Result<TaskOpResult> {
        let mut record = self.require(task_id).await?;
        let Some(owner) = record.assignee.as_deref() else {
            return Ok(TaskOpResult::failed(format!(
                "task '{task_id}' has no current assignee to reassign from"
            )));
        };
        if owner != from {
            return Ok(TaskOpResult::failed(format!(
                "task '{task_id}' is not assigned to '{from}'"
            )));
        }
        if !record.status.is_reassignable() {
            return Ok(TaskOpResult::failed(format!(
                "task '{task_id}' cannot be reassigned from status '{}'",
                record.status.as_str()
            )));
        }
        if self.member_mode(to).await?.is_none() {
            return Ok(TaskOpResult::failed(format!(
                "member '{to}' not found in team '{}'",
                self.team_name
            )));
        }
        record.assignee = Some(to.to_owned());
        record.observed_revision = record.revision;
        record.push_change_note(
            TaskRevision {
                revision: record.revision,
                note: format!("reassigned from {from} to {to}"),
                by: Some(from.to_owned()),
            }
            .to_note(),
        );
        self.persist(record).await?;
        Ok(TaskOpResult::ok())
    }

    /// Submits `member`'s plan for `task_id`, moving it into `planning`.
    ///
    /// The plan text is stored in [`TaskRecord::note`] so the leader's approval step reads it back
    /// with the task; a plan-mode member may not claim past the gate before that approval, which is
    /// the runtime's rule — the board only records the submission.
    ///
    /// # Errors
    ///
    /// [`Error::NotFound`] when the task does not exist. Refusals — not the assignee, a status that
    /// cannot enter planning — are `Ok` with `ok = false`.
    pub async fn set_plan(&self, task_id: &str, member: &str, plan: &str) -> Result<TaskOpResult> {
        let mut record = self.require(task_id).await?;
        if record.assignee.as_deref() != Some(member) {
            return Ok(TaskOpResult::failed(not_assignee_reason(
                record.assignee.as_deref(),
            )));
        }
        if !matches!(record.status, TaskStatus::Pending | TaskStatus::InProgress) {
            return Ok(TaskOpResult::failed(format!(
                "task '{task_id}' cannot enter planning from status '{}'",
                record.status.as_str()
            )));
        }
        record.status = TaskStatus::Planning;
        record.observed_revision = record.revision;
        record.note = Some(plan.to_owned());
        record.push_change_note(
            TaskRevision {
                revision: record.revision,
                note: "plan submitted".to_owned(),
                by: Some(member.to_owned()),
            }
            .to_note(),
        );
        self.persist(record).await?;
        Ok(TaskOpResult::ok())
    }

    // -------------------------------------------------------------------------------------------
    // Completion, review, cancellation, reset
    // -------------------------------------------------------------------------------------------

    /// Completes `task_id` on `member`'s behalf, gated by the task's reviewers.
    ///
    /// Without reviewers the task reaches `completed` directly; with reviewers it moves to
    /// `in_review`, increments [`TaskRecord::review_round`] (rounds are per task and monotonic, which
    /// is what makes the previous round's votes unreadable and therefore "reset") and waits for a
    /// verdict. A `planning` task resolves through `in_progress` in the same row write.
    ///
    /// A `completed` or `cancelled` upstream task unblocks its direct dependents here: every dependent
    /// still `blocked` whose dependencies are all terminal becomes `pending` and is reported in
    /// [`TaskMutation::unblocked`].
    ///
    /// # Errors
    ///
    /// [`Error::NotFound`] when the task does not exist; [`Error::Validation`] when the caller is not
    /// the assignee (`task is assigned to 'bob'`), when the assignee has not observed the current
    /// content revision ([`REASON_STALE_CONTENT`]), or when the status cannot be completed.
    pub async fn complete(
        &self,
        task_id: &str,
        member: &str,
        summary: &str,
    ) -> Result<TaskMutation> {
        let mut record = self.require(task_id).await?;
        if record.assignee.as_deref() != Some(member) {
            return Err(Error::validation(not_assignee_reason(
                record.assignee.as_deref(),
            )));
        }
        if record.revision > record.observed_revision {
            return Err(Error::validation(REASON_STALE_CONTENT));
        }
        if !matches!(record.status, TaskStatus::InProgress | TaskStatus::Planning) {
            return Err(Error::validation(format!(
                "task '{task_id}' cannot be completed from status '{}'",
                record.status.as_str()
            )));
        }
        if !summary.trim().is_empty() {
            record.summary = Some(summary.to_owned());
        }
        let mut unblocked = Vec::new();
        let completed = record.reviewers.is_empty();
        if completed {
            if record.status == TaskStatus::Planning {
                // The table has no `planning -> completed` edge; resolve through `in_progress` in
                // the same row write so the intermediate state is never observable.
                record.status = TaskStatus::InProgress;
            }
            record.status = TaskStatus::Completed;
        } else {
            record.status = TaskStatus::InReview;
            record.review_round = record.review_round.saturating_add(1);
        }
        let record = self.persist(record).await?;
        let events = vec![self.status_event(&record)];
        if completed {
            unblocked = self.unblock_dependents(task_id).await?;
        }
        Ok(TaskMutation {
            task: record,
            events,
            unblocked,
        })
    }

    /// Applies one reviewer verdict to `task_id` (autonomous dispatch: the first verdict wins).
    ///
    /// A `pass` completes the task and unblocks its dependents; a `fail` sends it back to
    /// `in_progress` for rework, releases the assignee back to the claimable pool (only when the
    /// reviewer is somebody else — a self-review keeps the author on the task) and appends the
    /// feedback as a change note.
    ///
    /// The vote is persisted before the status row, so a failed row write cannot lose the fact that
    /// the reviewer voted; the task simply stays `in_review` and the next verdict replaces the vote.
    ///
    /// # Errors
    ///
    /// [`Error::NotFound`] when the task does not exist; [`Error::Validation`] when the task is not
    /// `in_review` or `reviewer` is not one of its reviewers.
    pub async fn verify(
        &self,
        task_id: &str,
        reviewer: &str,
        verdict: Verdict,
        feedback: &str,
    ) -> Result<TaskMutation> {
        let mut record = self.require(task_id).await?;
        if record.status != TaskStatus::InReview {
            return Err(Error::validation(format!(
                "task '{task_id}' is not under review (status '{}'); nothing to verify",
                record.status.as_str()
            )));
        }
        if !record.reviewers.iter().any(|name| name == reviewer) {
            return Err(Error::validation(format!(
                "'{reviewer}' is not a reviewer of task '{task_id}'; cannot verify it"
            )));
        }
        let round = record.review_round;
        let author = record.assignee.clone().unwrap_or_default();
        self.store
            .insert_vote(ReviewVote {
                team_name: self.team_name.clone(),
                task_id: task_id.to_owned(),
                round,
                reviewer: reviewer.to_owned(),
                verdict,
            })
            .await?;
        let mut events = vec![EventPayload::TaskReviewVote(TaskReviewVoteEvent {
            team_name: self.team_name.clone(),
            task_id: task_id.to_owned(),
            round,
            reviewer: reviewer.to_owned(),
            verdict,
        })];
        let mut unblocked = Vec::new();
        match verdict {
            Verdict::Pass => {
                record.status = TaskStatus::Completed;
                let record = self.persist(record).await?;
                events.push(self.status_event(&record));
                events.push(EventPayload::TaskVerified(TaskVerifiedEvent {
                    team_name: self.team_name.clone(),
                    task_id: task_id.to_owned(),
                    member_name: author,
                    verdict,
                }));
                unblocked = self.unblock_dependents(task_id).await?;
                Ok(TaskMutation {
                    task: record,
                    events,
                    unblocked,
                })
            }
            Verdict::Fail => {
                record.status = TaskStatus::InProgress;
                if record.assignee.as_deref() == Some(reviewer) {
                    // Self-review: the author is the reviewer, so it keeps the task to rework it.
                } else {
                    record.assignee = None;
                }
                let feedback = feedback.trim();
                if !feedback.is_empty() {
                    record.note = Some(feedback.to_owned());
                    record.push_change_note(
                        TaskRevision {
                            revision: record.revision,
                            note: format!("review failed: {feedback}"),
                            by: Some(reviewer.to_owned()),
                        }
                        .to_note(),
                    );
                }
                let record = self.persist(record).await?;
                events.push(self.status_event(&record));
                events.push(EventPayload::TaskVerified(TaskVerifiedEvent {
                    team_name: self.team_name.clone(),
                    task_id: task_id.to_owned(),
                    member_name: author,
                    verdict,
                }));
                Ok(TaskMutation {
                    task: record,
                    events,
                    unblocked,
                })
            }
        }
    }

    /// Records one reviewer vote under scheduled dispatch; the status does not change.
    ///
    /// The round keeps collecting votes until [`TaskBoard::settle_review`] applies the tally. The
    /// emitted [`EventPayload::TaskReviewVote`] carries the vote itself; the running tally is read
    /// with [`TaskBoard::review_tally`] (the crate's vote payload has no count fields, unlike
    /// Python's `TaskReviewVoteEvent`).
    ///
    /// # Errors
    ///
    /// [`Error::NotFound`] when the task does not exist; [`Error::Validation`] when the task is not
    /// `in_review` or `reviewer` is not one of its reviewers.
    pub async fn record_vote(
        &self,
        task_id: &str,
        reviewer: &str,
        verdict: Verdict,
    ) -> Result<TaskMutation> {
        let record = self.require(task_id).await?;
        if record.status != TaskStatus::InReview {
            return Err(Error::validation(format!(
                "task '{task_id}' is not under review (status '{}'); nothing to vote on",
                record.status.as_str()
            )));
        }
        if !record.reviewers.iter().any(|name| name == reviewer) {
            return Err(Error::validation(format!(
                "'{reviewer}' is not a reviewer of task '{task_id}'; cannot vote on it"
            )));
        }
        let round = record.review_round;
        self.store
            .insert_vote(ReviewVote {
                team_name: self.team_name.clone(),
                task_id: task_id.to_owned(),
                round,
                reviewer: reviewer.to_owned(),
                verdict,
            })
            .await?;
        let events = vec![EventPayload::TaskReviewVote(TaskReviewVoteEvent {
            team_name: self.team_name.clone(),
            task_id: task_id.to_owned(),
            round,
            reviewer: reviewer.to_owned(),
            verdict,
        })];
        Ok(TaskMutation::new(record, events))
    }

    /// Returns the vote tally of `task_id`'s current review round.
    ///
    /// # Errors
    ///
    /// [`Error::NotFound`] when the task does not exist, and whatever the store fails with.
    pub async fn review_tally(&self, task_id: &str) -> Result<ReviewTally> {
        let record = self.require(task_id).await?;
        let votes = self
            .store
            .votes(&self.team_name, task_id, record.review_round)
            .await?;
        Ok(tally_with(&votes, &record.reviewers))
    }

    /// Settles `task_id`'s review round from its tally (scheduled dispatch).
    ///
    /// A passing tally completes the task and unblocks dependents; a failing tally returns it to
    /// `in_progress` with the author still assigned (the author is expected to rework it, unlike a
    /// `fail` verdict which releases the assignee). The failed round is recorded in the note so the
    /// next reader sees why the task came back.
    ///
    /// # Errors
    ///
    /// [`Error::NotFound`] when the task does not exist; [`Error::Validation`] when the task is not
    /// `in_review` or the round is not decided yet (`… is not decided yet (1 of 3 votes)`).
    pub async fn settle_review(&self, task_id: &str) -> Result<TaskMutation> {
        let mut record = self.require(task_id).await?;
        if record.status != TaskStatus::InReview {
            return Err(Error::validation(format!(
                "task '{task_id}' is not under review (status '{}'); nothing to settle",
                record.status.as_str()
            )));
        }
        let tally = self.review_tally(task_id).await?;
        if !tally.decided() {
            return Err(Error::validation(format!(
                "review tally of task '{task_id}' is not decided yet ({} of {} votes)",
                tally.voted(),
                tally.reviewer_count
            )));
        }
        let verdict = if tally.passed {
            Verdict::Pass
        } else {
            Verdict::Fail
        };
        let author = record.assignee.clone().unwrap_or_default();
        let mut unblocked = Vec::new();
        if tally.passed {
            record.status = TaskStatus::Completed;
        } else {
            record.status = TaskStatus::InProgress;
            record.note = Some(format!(
                "review round {} failed ({} pass / {} fail)",
                tally.round, tally.pass_count, tally.fail_count
            ));
        }
        let record = self.persist(record).await?;
        let mut events = vec![
            self.status_event(&record),
            EventPayload::TaskVerified(TaskVerifiedEvent {
                team_name: self.team_name.clone(),
                task_id: task_id.to_owned(),
                member_name: author,
                verdict,
            }),
        ];
        if tally.passed {
            unblocked = self.unblock_dependents(task_id).await?;
        }
        events.push(EventPayload::TaskReviewVote(TaskReviewVoteEvent {
            team_name: self.team_name.clone(),
            task_id: task_id.to_owned(),
            round: tally.round,
            reviewer: String::new(),
            verdict,
        }));
        Ok(TaskMutation {
            task: record,
            events,
            unblocked,
        })
    }

    /// Cancels `task_id`, unblocking its dependents exactly as a completion does.
    ///
    /// The assignee is kept on the row so the runtime can steer it off the cancelled task. Open
    /// subtasks are *not* cascaded here: cancellation is one task, and the runtime cancels a subtree
    /// depth-first because a child cancelled first would let the parked parent resume.
    ///
    /// # Errors
    ///
    /// [`Error::NotFound`] when the task does not exist; [`Error::Validation`] when it is already
    /// terminal.
    pub async fn cancel(&self, task_id: &str) -> Result<TaskMutation> {
        let mut record = self.require(task_id).await?;
        if record.is_terminal() {
            return Err(Error::validation(format!(
                "task '{task_id}' cannot be cancelled from status '{}' (already terminal)",
                record.status.as_str()
            )));
        }
        record.status = TaskStatus::Cancelled;
        let record = self.persist(record).await?;
        let events = vec![self.status_event(&record)];
        let unblocked = self.unblock_dependents(task_id).await?;
        Ok(TaskMutation {
            task: record,
            events,
            unblocked,
        })
    }

    /// Returns a `completed` / `cancelled` / `in_review` task to `pending`, releasing its owner.
    ///
    /// The owner, the closing summary and the note are cleared so the task is claimable again from a
    /// clean slate; reviewers and the review ceiling are kept. [`TaskRecord::review_round`] is
    /// deliberately *not* reset: rounds are monotonic per task and the store keeps votes forever, so
    /// rewinding to round `1` would let the next round re-read the previous cycle's verdicts.
    ///
    /// # Errors
    ///
    /// [`Error::NotFound`] when the task does not exist; [`Error::Validation`] for any other status.
    pub async fn reset(&self, task_id: &str) -> Result<TaskMutation> {
        let mut record = self.require(task_id).await?;
        if !matches!(
            record.status,
            TaskStatus::Completed | TaskStatus::Cancelled | TaskStatus::InReview
        ) {
            return Err(Error::validation(format!(
                "task '{task_id}' cannot be reset from status '{}'; only completed / cancelled / in-review tasks can be reset",
                record.status.as_str()
            )));
        }
        record.status = TaskStatus::Pending;
        record.assignee = None;
        record.summary = None;
        record.note = None;
        record.observed_revision = record.revision;
        let record = self.persist(record).await?;
        let events = vec![self.status_event(&record)];
        Ok(TaskMutation::new(record, events))
    }

    // -------------------------------------------------------------------------------------------
    // Reads
    // -------------------------------------------------------------------------------------------

    /// Lists tasks matching `filter`, ordered by `updated_at` descending then `task_id`.
    ///
    /// The graph-derived fields of every summary are filled from the whole board, so a filtered list
    /// still shows a task's `blocked_by` even when the blockers themselves are filtered out.
    ///
    /// # Errors
    ///
    /// Whatever the store fails with.
    pub async fn view(&self, filter: &TaskFilter) -> Result<Vec<TaskSummary>> {
        let snapshot = self.snapshot().await?;
        let records = self.store.list_tasks(&self.team_name, filter).await?;
        Ok(records
            .iter()
            .map(|record| snapshot.summary(record))
            .collect())
    }

    /// Returns the full detail of `task_id`, or `None` when it is not on the board.
    ///
    /// # Errors
    ///
    /// Whatever the store fails with.
    pub async fn detail(&self, task_id: &str) -> Result<Option<TaskDetail>> {
        let snapshot = self.snapshot().await?;
        Ok(snapshot
            .tasks
            .get(task_id)
            .map(|record| snapshot.detail(record)))
    }

    /// Returns the tasks a member may pick up: `pending` and unassigned, newest first.
    ///
    /// # Errors
    ///
    /// Whatever the store fails with.
    pub async fn claimable(&self) -> Result<Vec<TaskSummary>> {
        let filter = TaskFilter {
            status: Some(TaskStatus::Pending),
            ..TaskFilter::default()
        };
        Ok(self
            .view(&filter)
            .await?
            .into_iter()
            .filter(|summary| summary.assignee.is_none())
            .collect())
    }

    /// Returns the direct subtasks of `parent_id`, newest first.
    ///
    /// # Errors
    ///
    /// Whatever the store fails with.
    pub async fn subtasks(&self, parent_id: &str) -> Result<Vec<TaskSummary>> {
        let filter = TaskFilter {
            parent: Some(parent_id.to_owned()),
            ..TaskFilter::default()
        };
        self.view(&filter).await
    }

    /// Returns `true` when no task is left open — every task is `completed` or `cancelled`.
    ///
    /// An empty board is settled: there is nothing to wait for. The runtime uses this as its
    /// "drained" signal, which is safe before the first task exists and after the last one finishes.
    ///
    /// # Errors
    ///
    /// Whatever the store fails with.
    pub async fn is_settled(&self) -> Result<bool> {
        let snapshot = self.snapshot().await?;
        Ok(snapshot.tasks.values().all(TaskRecord::is_terminal))
    }

    /// Returns `true` when the board has tasks and every one of them is terminal.
    ///
    /// Stricter than [`TaskBoard::is_settled`] by requiring at least one task, so an untouched board
    /// never reads as a completed team: the Python `task_list_drained` signal only fires once tasks
    /// existed, and a team that produced nothing is not "done".
    ///
    /// # Errors
    ///
    /// Whatever the store fails with.
    pub async fn is_completed(&self) -> Result<bool> {
        let snapshot = self.snapshot().await?;
        Ok(!snapshot.tasks.is_empty() && snapshot.tasks.values().all(TaskRecord::is_terminal))
    }

    /// Returns the tasks that look abandoned, sorted by `task_id`.
    ///
    /// A `planning` / `in_progress` task is stale once it has not been written for `claim_idle_ms`; an
    /// unassigned `pending` task is stale once the board has not changed it for `pending_idle_ms`
    /// (nobody claimed it). The boundary is inclusive — a task exactly `idle_ms` old is stale — and a
    /// row without a timestamp (`0`, never written by a store that stamps) counts as infinitely old.
    ///
    /// # Errors
    ///
    /// Whatever the store fails with.
    pub async fn stale_tasks(
        &self,
        now_ms: u64,
        claim_idle_ms: u64,
        pending_idle_ms: u64,
    ) -> Result<Vec<TaskSummary>> {
        let snapshot = self.snapshot().await?;
        let mut stale: Vec<TaskSummary> = snapshot
            .tasks
            .values()
            .filter(|record| {
                let age_ms = now_ms.saturating_sub(record.updated_at);
                match record.status {
                    TaskStatus::Planning | TaskStatus::InProgress => age_ms >= claim_idle_ms,
                    TaskStatus::Pending => record.assignee.is_none() && age_ms >= pending_idle_ms,
                    TaskStatus::Blocked
                    | TaskStatus::InReview
                    | TaskStatus::Completed
                    | TaskStatus::Cancelled => false,
                }
            })
            .map(|record| snapshot.summary(record))
            .collect();
        stale.sort_by(|left, right| left.task_id.cmp(&right.task_id));
        Ok(stale)
    }

    // -------------------------------------------------------------------------------------------
    // Internals
    // -------------------------------------------------------------------------------------------

    /// Loads one row, mapping a missing task to [`Error::NotFound`].
    async fn require(&self, task_id: &str) -> Result<TaskRecord> {
        self.load(task_id)
            .await?
            .ok_or_else(|| missing_task(task_id))
    }

    /// Loads one row, or `None` when it is not on the board.
    async fn load(&self, task_id: &str) -> Result<Option<TaskRecord>> {
        self.store.task(&self.team_name, task_id).await
    }

    /// Writes `record` as the task's new row and returns the store's authoritative copy.
    ///
    /// [`TaskRecord::updated_at`] is zeroed first so the store stamps this write; a caller-supplied
    /// timestamp would otherwise pin the row to the old age and defeat the stale sweep.
    async fn persist(&self, mut record: TaskRecord) -> Result<TaskRecord> {
        record.updated_at = 0;
        let task_id = record.task_id.clone();
        self.store.insert_task(record).await?;
        self.require(&task_id).await
    }

    /// Reads the whole board: every task, its dependency edges and the roster statuses.
    async fn snapshot(&self) -> Result<Snapshot> {
        let records = self
            .store
            .list_tasks(&self.team_name, &TaskFilter::default())
            .await?;
        let mut tasks = BTreeMap::new();
        let mut edges = BTreeMap::new();
        for record in records {
            let dependencies = self
                .store
                .dependencies(&self.team_name, &record.task_id)
                .await?;
            edges.insert(
                record.task_id.clone(),
                dependencies.into_iter().collect::<BTreeSet<_>>(),
            );
            tasks.insert(record.task_id.clone(), record);
        }
        let members = self
            .store
            .list_members(&self.team_name)
            .await?
            .into_iter()
            .map(|member| (member.member_name, member.status))
            .collect();
        Ok(Snapshot {
            tasks,
            edges,
            members,
        })
    }

    /// Returns the roster mode of `member`, or `None` when it is not on the roster.
    async fn member_mode(&self, member: &str) -> Result<Option<MemberMode>> {
        Ok(self
            .store
            .member(&self.team_name, member)
            .await?
            .map(|record| record.mode))
    }

    /// Move every `blocked` dependent of `task_id` to `pending` whose dependencies are all terminal.
    async fn unblock_dependents(&self, task_id: &str) -> Result<Vec<String>> {
        let snapshot = self.snapshot().await?;
        let mut unblocked = Vec::new();
        for dependent_id in self.store.dependents(&self.team_name, task_id).await? {
            let Some(record) = snapshot.tasks.get(&dependent_id).cloned() else {
                continue;
            };
            if record.status != TaskStatus::Blocked
                || !snapshot.blocked_by(&dependent_id).is_empty()
            {
                continue;
            }
            let mut record = record;
            record.status = TaskStatus::Pending;
            self.persist(record).await?;
            unblocked.push(dependent_id);
        }
        unblocked.sort();
        Ok(unblocked)
    }

    /// Returns the nesting depth `task` would hand to its own subtask.
    async fn subtask_depth(&self, task: &TaskRecord) -> Result<usize> {
        let mut chain = vec![task.task_id.clone()];
        let mut seen: BTreeSet<String> = chain.iter().cloned().collect();
        let mut parent_id = task.parent_task_id.clone();
        while let Some(parent) = parent_id {
            if !seen.insert(parent.clone()) {
                break;
            }
            chain.push(parent.clone());
            parent_id = self
                .load(&parent)
                .await?
                .and_then(|record| record.parent_task_id);
        }
        Ok(depth_of(&chain))
    }

    /// Builds the `task_created` payload of `record`.
    fn created_event(&self, record: &TaskRecord) -> EventPayload {
        EventPayload::TaskCreated(TaskCreatedEvent {
            team_name: self.team_name.clone(),
            task_id: record.task_id.clone(),
            title: record.title.clone(),
            assignee: record.assignee.clone(),
        })
    }

    /// Builds the `task_status_changed` payload of `record`.
    fn status_event(&self, record: &TaskRecord) -> EventPayload {
        EventPayload::TaskStatusChanged(TaskStatusChangedEvent {
            team_name: self.team_name.clone(),
            task_id: record.task_id.clone(),
            status: record.status,
            assignee: record.assignee.clone(),
        })
    }
}

/// The stable refusal used whenever the caller is not the task's assignee.
fn not_assignee_reason(assignee: Option<&str>) -> String {
    format!(
        "task is assigned to '{}'",
        assignee.unwrap_or("<unassigned>")
    )
}

/// Returns `values` in order, dropping blanks and duplicates.
fn distinct(values: &[String]) -> Vec<String> {
    let mut seen = BTreeSet::new();
    values
        .iter()
        .map(|value| value.trim())
        .filter(|value| !value.is_empty())
        .filter(|value| seen.insert((*value).to_owned()))
        .map(str::to_owned)
        .collect()
}

/// One consistent read of the whole board, used to answer every query without re-reading rows.
#[derive(Debug)]
struct Snapshot {
    /// Every task row, keyed by id.
    tasks: BTreeMap<String, TaskRecord>,
    /// `task_id` → ids it depends on.
    edges: BTreeMap<String, BTreeSet<String>>,
    /// Member name → roster status.
    members: BTreeMap<String, MemberStatus>,
}

impl Snapshot {
    /// Returns `true` when `task_id` is unknown or not terminal.
    fn is_terminal(&self, task_id: &str) -> bool {
        self.tasks.get(task_id).is_some_and(TaskRecord::is_terminal)
    }

    /// Returns the dependencies of `task_id` that are still unresolved, sorted.
    fn blocked_by(&self, task_id: &str) -> Vec<String> {
        self.edges
            .get(task_id)
            .map(|dependencies| {
                dependencies
                    .iter()
                    .filter(|dependency| !self.is_terminal(dependency))
                    .cloned()
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Returns the ids that directly depend on `task_id`, sorted.
    fn blocks(&self, task_id: &str) -> Vec<String> {
        self.edges
            .iter()
            .filter(|(_, dependencies)| dependencies.contains(task_id))
            .map(|(dependent, _)| dependent.clone())
            .collect()
    }

    /// Returns the direct subtasks of `parent_id`, sorted.
    fn children(&self, parent_id: &str) -> Vec<String> {
        self.tasks
            .values()
            .filter(|record| record.parent_task_id.as_deref() == Some(parent_id))
            .map(|record| record.task_id.clone())
            .collect()
    }

    /// Projects one row into its list view with the graph-derived fields filled.
    fn summary(&self, record: &TaskRecord) -> TaskSummary {
        let mut summary = TaskSummary::from_record(record);
        summary.blocked_by = self.blocked_by(&record.task_id);
        summary.assignee_status = self.open_assignee_status(record);
        summary.stale_against = self.stale_marker(record);
        summary
    }

    /// Projects one row into its detail view with the graph-derived fields filled.
    fn detail(&self, record: &TaskRecord) -> TaskDetail {
        let mut detail = TaskDetail::from_record(record);
        detail.assignee_status = self.open_assignee_status(record);
        detail.blocked_by = self.blocked_by(&record.task_id);
        detail.blocks = self.blocks(&record.task_id);
        detail.subtasks = self.children(&record.task_id);
        detail
    }

    /// Returns the owner's roster status while the task is open and the owner is on the roster.
    fn open_assignee_status(&self, record: &TaskRecord) -> Option<MemberStatus> {
        if record.is_terminal() {
            return None;
        }
        record
            .assignee
            .as_deref()
            .and_then(|name| self.members.get(name))
            .copied()
    }

    /// Renders the `(parent revision now, revision it was based on)` stale marker of a subtask.
    fn stale_marker(&self, record: &TaskRecord) -> Option<String> {
        let parent = record.parent_task_id.as_deref()?;
        let current = self.tasks.get(parent)?.revision;
        let base = record.parent_revision?;
        (current > base && record.status != TaskStatus::Completed)
            .then(|| format!("r{current} vs r{base}"))
    }
}
