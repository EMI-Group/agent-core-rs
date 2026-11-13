//! The organisation task pool: cross-team tasks persisted through the team store.
//!
//! This module is legitimately long (~1.1k lines) and holds the pool policy only: the value
//! objects (row projection, sidecar, operation result, event vocabulary) live in
//! [`task_pool_types`]. The pool itself stays in one file because its twenty compare-and-set
//! guarded lifecycle operations share one store namespace and one audit clock, and the rules only
//! read as a whole when create, delegate, claim, complete, review and settle sit together.
//!
//! Python parity: `openjiuwen/agent_teams/organization/schema.py` (`OrgTask`, `OrgTaskFailureCode`)
//! and `organization/task_pool.py` (`OrgTaskManager`). Python wraps each operation in one SQL
//! transaction; this port writes through [`crate::storage::TeamStore`] with the same conditional-
//! update discipline.
//!
//! ## Storage model
//!
//! Rows live in the existing store under `crate::org::org_namespace(organization_id)` (for example
//! `org::acme`) — never in a second database. One row is one task:
//!
//! * `team_name` is the namespace; `task_id`, `title`, `content`, `status`, `summary` and
//!   `updated_at` are the task's own fields.
//! * `assignee`, `reviewers` and `created_by` are team-qualified member ids
//!   (`crate::org::qualify_member`); `parent_task_id` is the decomposition parent.
//! * `review_round` / `max_review_rounds` are the review gate, as on the task board, and
//!   `change_notes` holds the rendered audit notes.
//! * `note` is the [`OrgTaskSidecar`] JSON (`owner_team`, `target_team`, `depth`, `failure`,
//!   `metadata`). The pool is the only writer of organisation rows, so a human note never lands
//!   there; a row whose note does not parse is read with the sidecar defaults.
//!
//! Dependency edges mirror the task board's "parent depends on child" rule: creating a child adds
//! `add_dependency(namespace, parent_id, child_id)`, so a parent stays `blocked` until every child
//! is terminal. A terminal write moves every direct dependent whose dependencies are all terminal
//! `blocked -> pending`.
//!
//! ## Lifecycle
//!
//! | Operation | Allowed from | Result |
//! | --- | --- | --- |
//! | `create` | — | `pending`; a child gets `depth + 1` and parks its parent `blocked` |
//! | `delegate` | `pending`, `blocked` | `target_team` (and assignee) rewritten after the router allows the hop |
//! | `claim` | `pending`, unassigned | `in_progress` plus the qualified assignee (one CAS winner) |
//! | `start` | `pending`, pre-assigned to the caller | `in_progress` |
//! | `complete` | `planning`, `in_progress` | `in_review` + round `+ 1` with reviewers, else `completed` |
//! | `fail` | `planning`, `in_progress` | `cancelled` with the failure in the sidecar (no failed status) |
//! | `review` / `vote` | `in_review`, listed reviewer | first verdict (autonomous) / one round-tagged vote (scheduled) |
//! | `settle_review` | `in_review`, decided tally | `pass` → `completed`, otherwise rework in `in_progress` |
//!
//! Every status change is a compare-and-set through
//! [`crate::storage::TeamStore::transition_task`]: a `false` answer means somebody else won the
//! race, never that the write failed. Fields the status change does not own are patched afterwards
//! through [`crate::storage::TeamStore::update_task`], so a status is never written read-then-write.
//! The single full-row write appends `change_notes` (which has no partial-update path) from a fresh
//! read, after the status CAS.
//!
//! A policy refusal — a rejected delegation, a task claimed by somebody else, an undecided review
//! round — is `Ok` with `ok = false` and a stable reason, because those sentences are model-visible.
//! An unknown task is `Error::NotFound`; bad caller input is `Error::Validation`.
//!
//! Every mutation builds its [`OrgTaskEvent`]s: [`OrgTaskPool::create`] returns them in
//! [`OrgTaskMutation`], and every operation forwards them to the optional [`OrgTaskEventSink`]
//! installed with [`OrgTaskPool::with_event_sink`] — the runtime's team-bus hook — and logs them.
//!
//! No LLM call, no clock of its own (the store stamps `updated_at`), no global state.

#[path = "task_pool_types.rs"]
pub mod task_pool_types;

use std::fmt;
use std::sync::Arc;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::utils::id::uuid_hex;

use super::delegation::{DelegationRequest, DelegationRouter};
use super::{org_namespace, qualify_member, split_qualified, validate_org_id};
use crate::schema::{ReviewVote, TaskFilter, TaskPatch, TaskRecord, TaskStatus, Verdict};
use crate::storage::TeamStore;
use crate::taskboard::{missing_task, tally_with};

pub use task_pool_types::*;

/// The cross-team task pool of one organisation.
///
/// Built once per organisation over the shared [`TeamStore`] and the chart-backed
/// [`DelegationRouter`], and shared behind an `Arc`: every method is `&self`, and the only state is
/// the two seams plus the derived namespace.
#[derive(Clone)]
pub struct OrgTaskPool {
    store: Arc<dyn TeamStore>,
    router: Arc<DelegationRouter>,
    organization_id: String,
    namespace: String,
    events: Option<OrgTaskEventSink>,
}

impl fmt::Debug for OrgTaskPool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OrgTaskPool")
            .field("organization_id", &self.organization_id)
            .field("namespace", &self.namespace)
            .field("store", &self.store.name())
            .field("router", &self.router)
            .field("event_sink", &self.events.is_some())
            .finish()
    }
}

impl OrgTaskPool {
    /// Builds the pool of the organisation the router decides for.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the chart's organisation id is not a valid organisation
    /// id (a hand-built chart can bypass `OrganizationSpec::validate`).
    pub fn new(store: Arc<dyn TeamStore>, router: Arc<DelegationRouter>) -> Result<Self> {
        let organization_id = router.chart().organization_id().to_owned();
        validate_org_id(&organization_id)?;
        Ok(Self {
            namespace: org_namespace(&organization_id),
            store,
            router,
            organization_id,
            events: None,
        })
    }

    /// Returns this pool with `sink` installed as its event listener.
    ///
    /// The sink is called synchronously, after the write landed, once per event.
    #[must_use]
    pub fn with_event_sink(mut self, sink: OrgTaskEventSink) -> Self {
        self.events = Some(sink);
        self
    }

    /// Returns the organisation id this pool belongs to.
    #[must_use]
    pub fn organization_id(&self) -> &str {
        &self.organization_id
    }

    /// Returns the store namespace holding this organisation's rows.
    #[must_use]
    pub fn namespace(&self) -> &str {
        &self.namespace
    }

    /// Returns the store the pool writes through.
    #[must_use]
    pub fn store(&self) -> &Arc<dyn TeamStore> {
        &self.store
    }

    /// Returns the delegation router the pool asks before it delegates.
    #[must_use]
    pub fn router(&self) -> &Arc<DelegationRouter> {
        &self.router
    }

    /// Creates a task, and parks its parent while the new child is open.
    ///
    /// A root task starts at depth `0`; a child carries `depth = parent.depth + 1`, is linked to
    /// its parent through a dependency edge (`parent depends on child`) and parks the parent
    /// `blocked` until the child settles.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] for a blank title, an unknown owner/target team, a reviewer outside
    /// the target team, a creator outside the owner team or a terminal parent;
    /// [`Error::NotFound`] for an unknown `parent_task_id`; whatever the store fails with.
    pub async fn create(&self, spec: OrgTaskSpec) -> Result<OrgTaskMutation> {
        let title = spec.title.trim().to_owned();
        if title.is_empty() {
            return Err(Error::validation("task title must not be empty"));
        }
        let owner_team = spec.owner_team.trim().to_owned();
        self.require_team(&owner_team)?;
        let target = spec.target_team.as_deref().map(str::trim);
        let target_team = target
            .filter(|team| !team.is_empty())
            .map_or_else(|| owner_team.clone(), str::to_owned);
        self.require_team(&target_team)?;
        let reviewers = self.resolve_reviewers(&target_team, &spec.reviewers)?;
        let creator = spec.created_by.as_deref().map(str::trim);
        let created_by = match creator.filter(|member| !member.is_empty()) {
            Some(member) => self.resolve_member(&owner_team, member, "creator")?,
            None => qualify_member(&owner_team, self.router.chart().leader_of(&owner_team)?),
        };
        let parent_id = spec.parent_task_id.as_deref().map(str::trim);
        let (parent, depth) = match parent_id.filter(|parent| !parent.is_empty()) {
            Some(parent_id) => {
                let parent = self.require_task(parent_id).await?;
                if parent.is_terminal() {
                    return Err(Error::validation(format!(
                        "parent task '{parent_id}' is terminal and cannot be decomposed"
                    )));
                }
                let depth = parent.depth.saturating_add(1);
                (Some(parent), depth)
            }
            None => (None, 0),
        };

        let task_id = format!("task_{}", uuid_hex());
        let sidecar = OrgTaskSidecar {
            owner_team: owner_team.clone(),
            target_team: target_team.clone(),
            depth,
            failure: None,
            metadata: spec.metadata,
        };
        let mut record = TaskRecord::new(&self.namespace, &task_id, title, spec.content.trim());
        record.status = TaskStatus::Pending;
        record.reviewers = reviewers;
        record.created_by = Some(created_by);
        record.parent_task_id = parent.as_ref().map(|parent| parent.task_id.clone());
        record.note = Some(Self::encode_sidecar(&sidecar)?);
        self.store.insert_task(record).await?;
        if let Some(parent) = &parent {
            let edge = (&parent.task_id, &task_id);
            self.store
                .add_dependency(&self.namespace, edge.0, edge.1)
                .await?;
            self.park_parent(&parent.task_id).await?;
        }
        // A fresh task has no incoming edge (a child is a dependency of its *parent*).
        if self.has_open_dependencies(&task_id).await? {
            self.cas(&task_id, TaskStatus::Pending, TaskStatus::Blocked)
                .await?;
        }
        let task = self.require_task(&task_id).await?;
        let events = vec![OrgTaskEvent::Created {
            task_id: task_id.clone(),
            owner_team,
            target_team,
        }];
        self.emit(&events);
        tracing::debug!(organization = %self.organization_id, task_id = %task_id, "org task created");
        Ok(OrgTaskMutation {
            task,
            events,
            unblocked: Vec::new(),
        })
    }

    /// Delegates a task to another team (and optionally to one member of that team).
    ///
    /// The delegator is resolved from the chart and the hop is checked by the [`DelegationRouter`]
    /// with the task's depth and the target team's open-task count.
    ///
    /// # Errors
    ///
    /// [`Error::NotFound`] for an unknown task and whatever the store fails with. A refused
    /// delegation is `Ok` with `ok = false` and the router's own sentence.
    pub async fn delegate(
        &self,
        task_id: &str,
        from_member: &str,
        to_team: &str,
        to_member: Option<&str>,
    ) -> Result<OrgTaskOpResult> {
        let record = self.require(task_id).await?;
        let task = self.decode(record);
        if !matches!(task.status, TaskStatus::Pending | TaskStatus::Blocked) {
            return Ok(OrgTaskOpResult::failed(format!(
                "task '{task_id}' cannot be delegated from status '{}'",
                task.status.as_str()
            )));
        }
        let to_team = to_team.trim();
        let to_member = to_member.map(str::trim).filter(|member| !member.is_empty());
        let from_member = from_member.trim();
        let (from_team, delegator) = self.resolve_delegator(&task.owner_team, from_member);
        if !self.may_delegate(&task.owner_team, &from_team) {
            return Ok(OrgTaskOpResult::failed(format!(
                "task '{task_id}' is owned by team '{}'; '{from_member}' may not delegate it",
                task.owner_team
            )));
        }
        let open = self.open_task_count(to_team).await?;
        let request = DelegationRequest {
            from_team,
            from_member: delegator,
            to_team: to_team.to_owned(),
            to_member: to_member.map(str::to_owned),
            depth: task.depth,
            open_tasks_in_target: Some(open),
        };
        if let Some(refusal) = self.router.decide(&request).refusal() {
            return Ok(OrgTaskOpResult::failed(refusal.message()));
        }
        let mut sidecar = Self::sidecar(&task);
        sidecar.target_team = to_team.to_owned();
        let mut patch = TaskPatch::new().with_note(Self::encode_sidecar(&sidecar)?);
        let note = match to_member {
            Some(member) => {
                patch = patch.with_assignee(qualify_member(to_team, member));
                format!("delegated to {to_team}/{member}")
            }
            None => format!("delegated to {to_team}"),
        };
        self.patch_and_note(task_id, patch, note).await?;
        let task = self.require_task(task_id).await?;
        let events = vec![OrgTaskEvent::Delegated {
            task_id: task_id.to_owned(),
            from_team: request.from_team,
            to_team: to_team.to_owned(),
            to_member: to_member.map(|member| qualify_member(to_team, member)),
        }];
        Ok(self.applied(task, &events))
    }

    /// Claims a `pending` task for `member` of the task's target team.
    ///
    /// The claim is a single `pending -> in_progress` compare-and-set, so exactly one caller wins;
    /// only the winner writes the qualified assignee. Re-claiming a task the caller holds is a
    /// no-op success.
    ///
    /// # Errors
    ///
    /// [`Error::NotFound`] for an unknown task and whatever the store fails with. A refusal is
    /// `Ok` with `ok = false`.
    pub async fn claim(&self, task_id: &str, team: &str, member: &str) -> Result<OrgTaskOpResult> {
        let team = team.trim();
        let member = member.trim();
        let qualified = qualify_member(team, member);
        let task = self.require_task(task_id).await?;
        if task.target_team != team || self.router.chart().member(team, member).is_none() {
            return Ok(OrgTaskOpResult::failed(format!(
                "member '{qualified}' is not a member of team '{}'",
                task.target_team
            )));
        }
        if task.assignee.as_deref() == Some(qualified.as_str()) {
            if task.status == TaskStatus::InProgress {
                return Ok(OrgTaskOpResult::ok(task));
            }
            if task.status != TaskStatus::Pending {
                return Ok(OrgTaskOpResult::failed(not_claimable(task_id, task.status)));
            }
        } else if let Some(owner) = task.assignee.as_deref() {
            return Ok(OrgTaskOpResult::failed(already_claimed(task_id, owner)));
        }
        if task.status != TaskStatus::Pending {
            return Ok(OrgTaskOpResult::failed(not_claimable(task_id, task.status)));
        }
        let won = self
            .cas(task_id, TaskStatus::Pending, TaskStatus::InProgress)
            .await?;
        if !won {
            return self.lost_claim(task_id).await;
        }
        let patch = TaskPatch::new().with_assignee(qualified.clone());
        self.apply(task_id, &patch).await?;
        let task = self.require_task(task_id).await?;
        let events = vec![OrgTaskEvent::Claimed {
            task_id: task_id.to_owned(),
            assignee: qualified,
        }];
        Ok(self.applied(task, &events))
    }

    /// Starts a `pending` task that was pre-assigned to the caller.
    ///
    /// # Errors
    ///
    /// [`Error::NotFound`] for an unknown task and whatever the store fails with. A refusal is
    /// `Ok` with `ok = false`.
    pub async fn start(&self, task_id: &str, team: &str, member: &str) -> Result<OrgTaskOpResult> {
        let qualified = qualify_member(team.trim(), member.trim());
        let task = self.require_task(task_id).await?;
        if task.assignee.as_deref() != Some(qualified.as_str()) {
            return Ok(OrgTaskOpResult::failed(assigned_to(
                task_id,
                task.assignee.as_deref(),
            )));
        }
        if task.status == TaskStatus::InProgress {
            return Ok(OrgTaskOpResult::ok(task));
        }
        if task.status != TaskStatus::Pending {
            return Ok(OrgTaskOpResult::failed(format!(
                "task '{task_id}' cannot be started from status '{}'",
                task.status.as_str()
            )));
        }
        let won = self
            .cas(task_id, TaskStatus::Pending, TaskStatus::InProgress)
            .await?;
        if !won {
            let fresh = self.require_task(task_id).await?;
            if fresh.status == TaskStatus::InProgress
                && fresh.assignee.as_deref() == Some(qualified.as_str())
            {
                return Ok(OrgTaskOpResult::ok(fresh));
            }
            return self.lost_claim(task_id).await;
        }
        let task = self.require_task(task_id).await?;
        let events = vec![OrgTaskEvent::Started {
            task_id: task_id.to_owned(),
            assignee: qualified,
        }];
        Ok(self.applied(task, &events))
    }

    /// Completes the caller's work on `task_id`.
    ///
    /// With reviewers the task enters `in_review` and the round advances; without them it is
    /// `completed` and its dependents unblock. A `planning` task resolves through `in_progress`,
    /// the only completion edge the transition table has.
    ///
    /// # Errors
    ///
    /// [`Error::NotFound`] for an unknown task and whatever the store fails with. A refusal is
    /// `Ok` with `ok = false`.
    pub async fn complete(
        &self,
        task_id: &str,
        team: &str,
        member: &str,
        summary: &str,
    ) -> Result<OrgTaskOpResult> {
        let qualified = qualify_member(team.trim(), member.trim());
        let task = self.require_task(task_id).await?;
        if task.assignee.as_deref() != Some(qualified.as_str()) {
            return Ok(OrgTaskOpResult::failed(assigned_to(
                task_id,
                task.assignee.as_deref(),
            )));
        }
        if !matches!(task.status, TaskStatus::Planning | TaskStatus::InProgress) {
            return Ok(OrgTaskOpResult::failed(format!(
                "task '{task_id}' cannot be completed from status '{}'",
                task.status.as_str()
            )));
        }
        let review = !task.reviewers.is_empty();
        let target = if review {
            TaskStatus::InReview
        } else {
            TaskStatus::Completed
        };
        if !self.cas(task_id, task.status, target).await? {
            let status = self.status_of(task_id).await?;
            return Ok(OrgTaskOpResult::failed(format!(
                "task '{task_id}' cannot be completed from status '{}'",
                status.as_str()
            )));
        }
        let mut patch = TaskPatch::new();
        let summary = summary.trim();
        if !summary.is_empty() {
            patch = patch.with_summary(summary);
        }
        if review {
            patch = patch.with_review_round(task.review_round.saturating_add(1));
        }
        self.apply(task_id, &patch).await?;
        let mut events = vec![OrgTaskEvent::Completed {
            task_id: task_id.to_owned(),
            assignee: qualified,
        }];
        if !review {
            self.push_unblocked(task_id, &mut events).await?;
        }
        let task = self.require_task(task_id).await?;
        Ok(self.applied(task, &events))
    }

    /// Abandons `task_id` with `failure`.
    ///
    /// [`TaskStatus`] has no failed state, so the row becomes `cancelled`, the failure is persisted
    /// in the sidecar and a change note records it; dependents unblock like a completion.
    ///
    /// # Errors
    ///
    /// [`Error::NotFound`] for an unknown task and whatever the store fails with. A refusal is
    /// `Ok` with `ok = false`.
    pub async fn fail(
        &self,
        task_id: &str,
        team: &str,
        member: &str,
        failure: OrgTaskFailure,
    ) -> Result<OrgTaskOpResult> {
        let qualified = qualify_member(team.trim(), member.trim());
        let task = self.require_task(task_id).await?;
        if task.assignee.as_deref() != Some(qualified.as_str()) {
            return Ok(OrgTaskOpResult::failed(assigned_to(
                task_id,
                task.assignee.as_deref(),
            )));
        }
        if !matches!(task.status, TaskStatus::Planning | TaskStatus::InProgress) {
            return Ok(OrgTaskOpResult::failed(format!(
                "task '{task_id}' cannot be failed from status '{}'",
                task.status.as_str()
            )));
        }
        if !self
            .cas(task_id, task.status, TaskStatus::Cancelled)
            .await?
        {
            let status = self.status_of(task_id).await?;
            return Ok(OrgTaskOpResult::failed(format!(
                "task '{task_id}' cannot be failed from status '{}'",
                status.as_str()
            )));
        }
        let mut sidecar = Self::sidecar(&task);
        sidecar.failure = Some(failure.clone());
        let patch = TaskPatch::new().with_note(Self::encode_sidecar(&sidecar)?);
        let note = format!("failed: {} — {}", failure.code, failure.message);
        self.patch_and_note(task_id, patch, note).await?;
        let mut events = vec![OrgTaskEvent::Failed {
            task_id: task_id.to_owned(),
            code: failure.code,
        }];
        self.push_unblocked(task_id, &mut events).await?;
        let task = self.require_task(task_id).await?;
        Ok(self.applied(task, &events))
    }

    /// Applies one reviewer verdict to an `in_review` task (autonomous dispatch).
    ///
    /// A `pass` completes the task and unblocks its dependents; a `fail` returns it to
    /// `in_progress`, releases the assignee when the reviewer is not the author and records the
    /// feedback as a change note. The verdict is persisted as a round-tagged vote either way.
    ///
    /// # Errors
    ///
    /// [`Error::NotFound`] for an unknown task and whatever the store fails with. A refusal is
    /// `Ok` with `ok = false`.
    pub async fn review(
        &self,
        task_id: &str,
        reviewer_team: &str,
        reviewer: &str,
        verdict: Verdict,
        feedback: &str,
    ) -> Result<OrgTaskOpResult> {
        let qualified = qualify_member(reviewer_team.trim(), reviewer.trim());
        let task = self.require_task(task_id).await?;
        if task.status != TaskStatus::InReview {
            return Ok(OrgTaskOpResult::failed(format!(
                "task '{task_id}' is not under review (status '{}'); nothing to review",
                task.status.as_str()
            )));
        }
        if !task.reviewers.iter().any(|name| name == &qualified) {
            return Ok(OrgTaskOpResult::failed(format!(
                "'{qualified}' is not a reviewer of task '{task_id}'; cannot review it"
            )));
        }
        self.record_vote(&task, &qualified, verdict).await?;
        let keep = task.assignee.as_deref() == Some(qualified.as_str());
        let events = vec![OrgTaskEvent::Reviewed {
            task_id: task_id.to_owned(),
            reviewer: qualified,
            passed: verdict == Verdict::Pass,
        }];
        self.settle_verdict(&task, verdict, keep, feedback, events)
            .await
    }

    /// Records one reviewer's vote for the current round (scheduled dispatch).
    ///
    /// Nothing but the vote changes; the round is settled by [`OrgTaskPool::settle_review`] once
    /// every reviewer has voted.
    ///
    /// # Errors
    ///
    /// [`Error::NotFound`] for an unknown task and whatever the store fails with. A refusal is
    /// `Ok` with `ok = false`.
    pub async fn vote(
        &self,
        task_id: &str,
        reviewer_team: &str,
        reviewer: &str,
        verdict: Verdict,
    ) -> Result<OrgTaskOpResult> {
        let qualified = qualify_member(reviewer_team.trim(), reviewer.trim());
        let task = self.require_task(task_id).await?;
        if task.status != TaskStatus::InReview {
            return Ok(OrgTaskOpResult::failed(format!(
                "task '{task_id}' is not under review (status '{}'); nothing to vote on",
                task.status.as_str()
            )));
        }
        if !task.reviewers.iter().any(|name| name == &qualified) {
            return Ok(OrgTaskOpResult::failed(format!(
                "'{qualified}' is not a reviewer of task '{task_id}'; cannot vote on it"
            )));
        }
        self.record_vote(&task, &qualified, verdict).await?;
        let events = vec![OrgTaskEvent::Reviewed {
            task_id: task_id.to_owned(),
            reviewer: qualified,
            passed: verdict == Verdict::Pass,
        }];
        Ok(self.applied(task, &events))
    }

    /// Settles the current review round from the votes cast so far (scheduled dispatch).
    ///
    /// # Errors
    ///
    /// [`Error::NotFound`] for an unknown task and whatever the store fails with. An undecided
    /// round, a task outside the gate or a lost status race is `Ok` with `ok = false`.
    pub async fn settle_review(&self, task_id: &str) -> Result<OrgTaskOpResult> {
        let task = self.require_task(task_id).await?;
        if task.status != TaskStatus::InReview {
            return Ok(OrgTaskOpResult::failed(format!(
                "task '{task_id}' is not under review (status '{}'); nothing to settle",
                task.status.as_str()
            )));
        }
        let votes = self
            .store
            .votes(&self.namespace, task_id, task.review_round)
            .await?;
        let tally = tally_with(&votes, &task.reviewers);
        if !tally.decided() {
            return Ok(OrgTaskOpResult::failed(format!(
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
        let keep = votes.iter().any(|vote| {
            vote.verdict == Verdict::Fail
                && task.assignee.as_deref() == Some(vote.reviewer.as_str())
        });
        let events = votes
            .iter()
            .map(|vote| OrgTaskEvent::Reviewed {
                task_id: task_id.to_owned(),
                reviewer: vote.reviewer.clone(),
                passed: vote.verdict == Verdict::Pass,
            })
            .collect();
        self.settle_verdict(&task, verdict, keep, "", events).await
    }

    /// Returns one organisation task, or `None` when it is unknown.
    ///
    /// # Errors
    ///
    /// Returns whatever the store fails with.
    pub async fn get(&self, task_id: &str) -> Result<Option<OrgTask>> {
        let record = self.store.task(&self.namespace, task_id.trim()).await?;
        Ok(record.map(|record| self.decode(record)))
    }

    /// Returns the tasks matching `filter`, in the store's order (`updated_at` descending, then
    /// `task_id` ascending).
    ///
    /// # Errors
    ///
    /// Returns whatever the store fails with.
    pub async fn list(&self, filter: &OrgTaskFilter) -> Result<Vec<OrgTask>> {
        let team = filter.team.as_deref().map(str::trim);
        let team = team.filter(|team| !team.is_empty());
        let wanted = filter.assignee.as_deref().map(str::trim);
        let wanted = wanted.filter(|assignee| !assignee.is_empty());
        let qualified = wanted.filter(|value| split_qualified(value).is_some());
        let parent = filter.parent_task_id.as_deref().map(str::trim);
        let parent = parent.filter(|parent| !parent.is_empty());
        // `team` and a bare assignee are not store columns, so the store cannot apply the limit
        // before those filters run: fetch first, filter, then truncate.
        let post_filtered = team.is_some() || (wanted.is_some() && qualified.is_none());
        let store_filter = TaskFilter {
            status: filter.status,
            assignee: qualified.map(str::to_owned),
            parent: parent.map(str::to_owned),
            limit: if post_filtered { None } else { filter.limit },
        };
        let rows = self
            .store
            .list_tasks(&self.namespace, &store_filter)
            .await?;
        let mut tasks: Vec<OrgTask> = rows
            .into_iter()
            .map(|record| self.decode(record))
            .filter(|task| team.is_none_or(|team| task.target_team == team))
            .filter(|task| wanted.is_none_or(|wanted| assignee_matches(task, wanted, team)))
            .collect();
        if post_filtered {
            if let Some(limit) = filter.limit {
                tasks.truncate(limit);
            }
        }
        Ok(tasks)
    }

    /// Returns the direct children of `parent_task_id`.
    ///
    /// # Errors
    ///
    /// Returns whatever the store fails with.
    pub async fn children(&self, parent_task_id: &str) -> Result<Vec<OrgTask>> {
        let filter = OrgTaskFilter {
            parent_task_id: Some(parent_task_id.to_owned()),
            ..OrgTaskFilter::default()
        };
        self.list(&filter).await
    }

    /// Counts the non-terminal tasks addressed to `team` — the delegation budget input.
    ///
    /// # Errors
    ///
    /// Returns whatever the store fails with.
    pub async fn open_task_count(&self, team: &str) -> Result<usize> {
        let filter = OrgTaskFilter {
            team: Some(team.trim().to_owned()),
            ..OrgTaskFilter::default()
        };
        Ok(self
            .list(&filter)
            .await?
            .iter()
            .filter(|task| task.is_open())
            .count())
    }

    /// Loads one row, mapping a missing task to [`Error::NotFound`].
    async fn require(&self, task_id: &str) -> Result<TaskRecord> {
        self.store
            .task(&self.namespace, task_id.trim())
            .await?
            .ok_or_else(|| missing_task(task_id))
    }

    /// Loads one row and decodes it.
    async fn require_task(&self, task_id: &str) -> Result<OrgTask> {
        Ok(self.decode(self.require(task_id).await?))
    }

    /// Returns the stored status of `task_id`.
    async fn status_of(&self, task_id: &str) -> Result<TaskStatus> {
        Ok(self.require(task_id).await?.status)
    }

    /// Projects a stored row into an [`OrgTask`].
    fn decode(&self, record: TaskRecord) -> OrgTask {
        let sidecar = Self::sidecar_of(&record);
        OrgTask {
            task_id: record.task_id,
            organization_id: self.organization_id.clone(),
            title: record.title,
            content: record.content,
            status: record.status,
            owner_team: sidecar.owner_team,
            target_team: sidecar.target_team,
            assignee: record.assignee,
            reviewers: record.reviewers,
            review_round: record.review_round,
            max_review_rounds: record.max_review_rounds,
            parent_task_id: record.parent_task_id,
            depth: sidecar.depth,
            created_by: record.created_by.unwrap_or_default(),
            summary: record.summary,
            failure: sidecar.failure,
            metadata: sidecar.metadata,
            updated_at: record.updated_at,
        }
    }

    /// Reads the sidecar of a stored row, falling back to the defaults.
    fn sidecar_of(record: &TaskRecord) -> OrgTaskSidecar {
        record
            .note
            .as_deref()
            .and_then(|note| serde_json::from_str(note).ok())
            .unwrap_or_default()
    }

    /// Reads the sidecar of a decoded task.
    fn sidecar(task: &OrgTask) -> OrgTaskSidecar {
        OrgTaskSidecar {
            owner_team: task.owner_team.clone(),
            target_team: task.target_team.clone(),
            depth: task.depth,
            failure: task.failure.clone(),
            metadata: task.metadata.clone(),
        }
    }

    /// Serializes `sidecar` for the row's `note` column.
    fn encode_sidecar(sidecar: &OrgTaskSidecar) -> Result<String> {
        serde_json::to_string(sidecar).map_err(Error::external)
    }

    /// Fails unless `team` is a declared chart team.
    fn require_team(&self, team: &str) -> Result<()> {
        if self.router.chart().team(team).is_some() {
            return Ok(());
        }
        Err(Error::validation(format!(
            "unknown organization team '{team}'"
        )))
    }

    /// Resolves one member reference inside `team` to its qualified id.
    fn resolve_member(&self, team: &str, member: &str, role: &str) -> Result<String> {
        let chart = self.router.chart();
        let (member_team, member_id) = split_qualified(member).unwrap_or((team, member));
        if member_team != team || chart.member(member_team, member_id).is_none() {
            return Err(Error::validation(format!(
                "{role} '{member}' is not a member of team '{team}'"
            )));
        }
        Ok(qualify_member(member_team, member_id))
    }

    /// Resolves and de-duplicates the reviewer set of `target_team`.
    fn resolve_reviewers(&self, target_team: &str, reviewers: &[String]) -> Result<Vec<String>> {
        let mut resolved: Vec<String> = Vec::new();
        for reviewer in reviewers {
            let reviewer = reviewer.trim();
            if reviewer.is_empty() {
                continue;
            }
            let qualified = self.resolve_member(target_team, reviewer, "reviewer")?;
            if !resolved.contains(&qualified) {
                resolved.push(qualified);
            }
        }
        Ok(resolved)
    }

    /// Splits a delegator reference into its team and member id.
    fn resolve_delegator(&self, owner_team: &str, member: &str) -> (String, String) {
        let chart = self.router.chart();
        if let Some((team, id)) = split_qualified(member) {
            if chart.member(team, id).is_some() {
                return (team.to_owned(), id.to_owned());
            }
        } else if chart.member(owner_team, member).is_some() {
            return (owner_team.to_owned(), member.to_owned());
        }
        // An unresolved delegator keeps its owning team, so the router reports `UnknownDelegator`.
        (owner_team.to_owned(), member.to_owned())
    }

    /// Returns `true` when `from_team` may speak for `owner_team` (itself or an ancestor).
    fn may_delegate(&self, owner_team: &str, from_team: &str) -> bool {
        from_team == owner_team || self.router.chart().is_ancestor(from_team, owner_team)
    }

    /// Moves `task_id` from `from` to `to`, through `in_progress` when no direct edge exists.
    async fn cas(&self, task_id: &str, from: TaskStatus, to: TaskStatus) -> Result<bool> {
        if from.can_transition_to(to) {
            return self
                .store
                .transition_task(&self.namespace, task_id, from, to)
                .await;
        }
        if !from.can_transition_to(TaskStatus::InProgress) {
            return Ok(false);
        }
        let stepped = self
            .store
            .transition_task(&self.namespace, task_id, from, TaskStatus::InProgress)
            .await?;
        if !stepped {
            return Ok(false);
        }
        self.store
            .transition_task(&self.namespace, task_id, TaskStatus::InProgress, to)
            .await
    }

    /// Parks `parent_id` at `blocked` while one of its children is open.
    ///
    /// `pending -> blocked` is a table transition; `planning` / `in_progress` have no edge to
    /// `blocked` (the member is suspended, not transitioned), so the row is parked directly, the
    /// same repair the task board performs for `create_subtask`.
    async fn park_parent(&self, parent_id: &str) -> Result<()> {
        let parent = self.require(parent_id).await?;
        if parent.status.is_terminal() || parent.status == TaskStatus::Blocked {
            return Ok(());
        }
        if self
            .cas(parent_id, parent.status, TaskStatus::Blocked)
            .await?
        {
            return Ok(());
        }
        let mut parked = self.require(parent_id).await?;
        if parked.status.is_terminal() || parked.status == TaskStatus::Blocked {
            return Ok(());
        }
        parked.status = TaskStatus::Blocked;
        parked.updated_at = 0;
        self.store.insert_task(parked).await
    }

    /// Returns `true` when `task_id` has a dependency that is not terminal.
    async fn has_open_dependencies(&self, task_id: &str) -> Result<bool> {
        for dependency in self.store.dependencies(&self.namespace, task_id).await? {
            let record = self.store.task(&self.namespace, &dependency).await?;
            if !record.is_some_and(|record| record.status.is_terminal()) {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Moves every `blocked` dependent of `task_id` to `pending` whose dependencies are terminal.
    async fn unblock_dependents(&self, task_id: &str) -> Result<Vec<String>> {
        let mut unblocked = Vec::new();
        for dependent_id in self.store.dependents(&self.namespace, task_id).await? {
            let Some(record) = self.store.task(&self.namespace, &dependent_id).await? else {
                continue;
            };
            if record.status != TaskStatus::Blocked {
                continue;
            }
            if self.has_open_dependencies(&dependent_id).await? {
                continue;
            }
            let moved = self
                .store
                .transition_task(
                    &self.namespace,
                    &dependent_id,
                    TaskStatus::Blocked,
                    TaskStatus::Pending,
                )
                .await?;
            if moved {
                unblocked.push(dependent_id);
            }
        }
        unblocked.sort();
        Ok(unblocked)
    }

    /// Unblocks the dependents of `task_id` and appends one event per unblocked task.
    async fn push_unblocked(
        &self,
        task_id: &str,
        events: &mut Vec<OrgTaskEvent>,
    ) -> Result<Vec<String>> {
        let unblocked = self.unblock_dependents(task_id).await?;
        events.extend(unblocked.iter().map(|id| OrgTaskEvent::Unblocked {
            task_id: id.clone(),
        }));
        Ok(unblocked)
    }

    /// Records one round-tagged vote of `task`.
    async fn record_vote(&self, task: &OrgTask, reviewer: &str, verdict: Verdict) -> Result<()> {
        let vote = ReviewVote {
            team_name: self.namespace.clone(),
            task_id: task.task_id.clone(),
            round: task.review_round,
            reviewer: reviewer.to_owned(),
            verdict,
        };
        self.store.insert_vote(vote).await
    }

    /// Applies a settled `verdict` to an `in_review` task.
    async fn settle_verdict(
        &self,
        task: &OrgTask,
        verdict: Verdict,
        keep_assignee: bool,
        feedback: &str,
        mut events: Vec<OrgTaskEvent>,
    ) -> Result<OrgTaskOpResult> {
        let task_id = task.task_id.as_str();
        match verdict {
            Verdict::Pass => {
                let passed = self
                    .cas(task_id, TaskStatus::InReview, TaskStatus::Completed)
                    .await?;
                if !passed {
                    let status = self.status_of(task_id).await?;
                    return Ok(OrgTaskOpResult::failed(format!(
                        "task '{task_id}' cannot be completed from status '{}'",
                        status.as_str()
                    )));
                }
                self.push_unblocked(task_id, &mut events).await?;
            }
            Verdict::Fail => {
                let reopened = self
                    .cas(task_id, TaskStatus::InReview, TaskStatus::InProgress)
                    .await?;
                if !reopened {
                    let status = self.status_of(task_id).await?;
                    return Ok(OrgTaskOpResult::failed(format!(
                        "task '{task_id}' cannot be reopened from status '{}'",
                        status.as_str()
                    )));
                }
                let patch = if keep_assignee {
                    TaskPatch::new()
                } else {
                    TaskPatch::new().clearing_assignee()
                };
                self.apply(task_id, &patch).await?;
                let feedback = feedback.trim();
                if !feedback.is_empty() {
                    let note = format!("review failed: {feedback}");
                    self.append_notes(task_id, &[note]).await?;
                }
            }
        }
        let task = self.require_task(task_id).await?;
        Ok(self.applied(task, &events))
    }

    /// Applies `patch` to a row the caller already checked, mapping a vanished row to not-found.
    async fn apply(&self, task_id: &str, patch: &TaskPatch) -> Result<()> {
        if patch.is_empty() {
            return Ok(());
        }
        let applied = self
            .store
            .update_task(&self.namespace, task_id, patch)
            .await?;
        if !applied {
            return Err(missing_task(task_id));
        }
        Ok(())
    }

    /// Appends rendered audit `notes` to the row's `change_notes`.
    ///
    /// [`TaskPatch`] has no change-note field, so this is the one full-row write: the status comes
    /// from the fresh read, never from a stale one, and the status CAS has already happened.
    async fn append_notes(&self, task_id: &str, notes: &[String]) -> Result<()> {
        if notes.is_empty() {
            return Ok(());
        }
        let mut record = self.require(task_id).await?;
        for note in notes {
            record.push_change_note(note.clone());
        }
        record.updated_at = 0;
        self.store.insert_task(record).await
    }

    /// Applies `patch`, then appends one audit `note`.
    async fn patch_and_note(
        &self,
        task_id: &str,
        patch: TaskPatch,
        note: impl Into<String>,
    ) -> Result<()> {
        self.apply(task_id, &patch).await?;
        self.append_notes(task_id, &[note.into()]).await
    }

    /// Builds the success result of an operation that produced `events`.
    fn applied(&self, task: OrgTask, events: &[OrgTaskEvent]) -> OrgTaskOpResult {
        self.emit(events);
        tracing::debug!(
            organization = %self.organization_id,
            task_id = %task.task_id,
            status = task.status.as_str(),
            "org task mutated"
        );
        OrgTaskOpResult {
            ok: true,
            reason: None,
            task: Some(task),
        }
    }

    /// Forwards `events` to the installed sink and logs them.
    fn emit(&self, events: &[OrgTaskEvent]) {
        for event in events {
            tracing::debug!(organization = %self.organization_id, event = ?event, "org task event");
            if let Some(sink) = &self.events {
                sink(event.clone());
            }
        }
    }

    /// Reports a claim lost to another writer from the row's current state.
    async fn lost_claim(&self, task_id: &str) -> Result<OrgTaskOpResult> {
        let fresh = self.require_task(task_id).await?;
        Ok(match fresh.assignee.as_deref() {
            Some(owner) => OrgTaskOpResult::failed(already_claimed(task_id, owner)),
            None => OrgTaskOpResult::failed(not_claimable(task_id, fresh.status)),
        })
    }
}

/// The stable refusal used when a task is already held by another member.
fn already_claimed(task_id: &str, owner: &str) -> String {
    format!("task '{task_id}' is already claimed by '{owner}'")
}

/// The stable refusal used when the caller is not the task's assignee.
fn assigned_to(task_id: &str, assignee: Option<&str>) -> String {
    format!(
        "task '{task_id}' is assigned to '{}'",
        assignee.unwrap_or("<unassigned>")
    )
}

/// The stable refusal used when a task cannot be claimed from its current status.
fn not_claimable(task_id: &str, status: TaskStatus) -> String {
    format!(
        "task '{task_id}' cannot be claimed from status '{}' (only pending tasks are claimable)",
        status.as_str()
    )
}

/// Returns `true` when `task`'s qualified assignee matches `wanted`.
fn assignee_matches(task: &OrgTask, wanted: &str, team: Option<&str>) -> bool {
    match split_qualified(wanted) {
        Some((wanted_team, wanted_member)) => {
            task.assignee_parts().is_some_and(|(own_team, own_member)| {
                own_team == wanted_team && own_member == wanted_member
            })
        }
        None => task.assignee_parts().is_some_and(|(own_team, own_member)| {
            own_member == wanted && team.is_none_or(|team| own_team == team)
        }),
    }
}

#[cfg(test)]
#[path = "task_pool_tests.rs"]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod task_pool_tests;
