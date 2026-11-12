//! [`TeamMember`]: the status machine of one roster row over the [`TeamStore`] seam.
//!
//! Python parity source: `openjiuwen/agent_teams/agent/member.py` (`TeamMember.update_status` /
//! `update_execution_status`) plus the member-side settle path in `agent/team_agent.py`
//! (`shutdown_self` / `resolve_retire_target`). Python keeps three sources of truth — the
//! `TeamMember` handle, the `member` table and the event stream — and publishes a
//! `MemberStatusChangedEvent` on every write. The Rust port keeps the row authoritative and
//! mirrors it into a cached projection for the synchronous getters; event publication belongs to
//! the runtime (`TeamBackend`), which owns the messager and the topic.
//!
//! Every lifecycle write goes through [`TeamStore::transition_member`], the compare-and-set path,
//! so two writers (leader sweep and member harness) never clobber each other: a lost CAS refreshes
//! the cached row and reports `Ok(false)` rather than failing the caller.
//!
//! ```
//! use std::sync::Arc;
//! use openjiuwen_agent_teams::member::TeamMember;
//! use openjiuwen_agent_teams::schema::{MemberStatus, StorageSpec, TeamRole};
//! use openjiuwen_agent_teams::storage::{open_store, MemberRecord};
//!
//! # async fn demo() -> openjiuwen_common::error::Result<()> {
//! let store = open_store(&StorageSpec::in_memory())?;
//! store
//!     .insert_member(MemberRecord::new("alpha", "dev-1", TeamRole::Teammate))
//!     .await?;
//! let member = TeamMember::new(store, MemberRecord::new("alpha", "dev-1", TeamRole::Teammate));
//! assert_eq!(member.status(), MemberStatus::Unstarted);
//! assert!(member.mark(MemberStatus::Starting).await?);
//! assert_eq!(member.status(), MemberStatus::Starting);
//! # Ok(())
//! # }
//! ```

use std::fmt;
use std::sync::Arc;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_session::Session;
use parking_lot::{Mutex, RwLock};
use tracing::{debug, warn};

use crate::schema::{
    is_valid_member_transition, retire_target, round_transitions, ExecutionStatus, MemberLifetime,
    MemberStatus, RoundOutcome, TeamRole,
};
use crate::storage::{MemberPatch, MemberRecord, TeamStore};

use super::runtime::{MemberObserver, MemberRuntime};

/// One member of a team: its persisted row, its status machine and — once the runtime has built
/// it — its [`MemberRuntime`].
///
/// The handle is constructed eagerly for every role, before the member row may even exist: every
/// lifecycle write therefore tolerates a missing row (`Ok(false)`), mirroring Python's
/// "not registered yet; skipping status update" no-op. Identity (`team_name`, `member_name`) is
/// immutable; the record behind the getters is a cache refreshed on every successful write.
pub struct TeamMember {
    store: Arc<dyn TeamStore>,
    team_name: String,
    member_name: String,
    record: RwLock<MemberRecord>,
    runtime: RwLock<Option<Arc<dyn MemberRuntime>>>,
    observers: Mutex<Vec<Arc<dyn MemberObserver>>>,
}

impl TeamMember {
    /// Wraps `record` in a lifecycle handle over `store`.
    ///
    /// The record is not written: the caller inserts the row (`TeamStore::insert_member`) when it
    /// is ready for the roster, exactly like Python's eager handle before `build_team` runs.
    #[must_use]
    pub fn new(store: Arc<dyn TeamStore>, record: MemberRecord) -> Self {
        Self {
            store,
            team_name: record.team_name.clone(),
            member_name: record.member_name.clone(),
            record: RwLock::new(record),
            runtime: RwLock::new(None),
            observers: Mutex::new(Vec::new()),
        }
    }

    /// Returns a copy of the member's cached row.
    #[must_use]
    pub fn record(&self) -> MemberRecord {
        self.record.read().clone()
    }

    /// Returns the team the member belongs to.
    #[must_use]
    pub fn team_name(&self) -> &str {
        &self.team_name
    }

    /// Returns the member's wire name.
    #[must_use]
    pub fn member_name(&self) -> &str {
        &self.member_name
    }

    /// Returns the member's role.
    #[must_use]
    pub fn role(&self) -> TeamRole {
        self.record.read().role
    }

    /// Returns the cached lifecycle status.
    #[must_use]
    pub fn status(&self) -> MemberStatus {
        self.record.read().status
    }

    /// Returns the cached execution status.
    #[must_use]
    pub fn execution(&self) -> ExecutionStatus {
        self.record.read().execution
    }

    /// Returns the runtime attached to this member, if any.
    #[must_use]
    pub fn runtime(&self) -> Option<Arc<dyn MemberRuntime>> {
        self.runtime.read().clone()
    }

    /// Attaches the member's runtime and forwards every observer registered so far.
    ///
    /// Attaching twice replaces the runtime: the new one receives the observer set, the old one is
    /// simply dropped (a runtime per run cycle — the harness closes the previous one itself).
    pub fn attach_runtime(&self, runtime: Arc<dyn MemberRuntime>) {
        let pending: Vec<Arc<dyn MemberObserver>> = std::mem::take(&mut *self.observers.lock());
        for observer in &pending {
            runtime.subscribe(Arc::clone(observer));
        }
        *self.runtime.write() = Some(runtime);
        debug!(
            team = %self.team_name,
            member = %self.member_name,
            observers = pending.len(),
            "attached member runtime"
        );
    }

    /// Registers an observer for the member's runtime.
    ///
    /// Before [`Self::attach_runtime`] the observer is queued and handed over on attach; afterwards
    /// it is registered immediately.
    pub fn observe(&self, observer: Arc<dyn MemberObserver>) {
        let runtime = self.runtime.read().clone();
        match runtime {
            Some(runtime) => runtime.subscribe(observer),
            None => self.observers.lock().push(observer),
        }
    }

    /// Re-reads the member row and returns it, refreshing the cached projection.
    ///
    /// Use this when another writer may have moved the row (the leader sweep, a peer's retire):
    /// the synchronous getters only know about writes made through this handle.
    ///
    /// # Errors
    ///
    /// Returns [`Error::not_found`] when the row does not exist and propagates store failures.
    pub async fn refresh(&self) -> Result<MemberRecord> {
        let stored = self
            .store
            .member(&self.team_name, &self.member_name)
            .await?
            .ok_or_else(|| {
                Error::not_found(format!(
                    "member '{}' not found in team '{}'",
                    self.member_name, self.team_name
                ))
            })?;
        *self.record.write() = stored.clone();
        Ok(stored)
    }

    /// Moves the member to `to` through the validated compare-and-set path.
    ///
    /// Re-asserting the current status is an idempotent `Ok(true)` with no write and no event,
    /// which is what lets recovery paths pull a member back to `ready` unconditionally. An edge
    /// the transition table rejects is `Ok(false)` with the store untouched. A lost compare-and-set
    /// is also `Ok(false)`, after refreshing the cache so the caller can re-read the winner's
    /// status.
    ///
    /// # Errors
    ///
    /// Propagates a store failure.
    pub async fn mark(&self, to: MemberStatus) -> Result<bool> {
        let from = self.status();
        if from == to {
            return Ok(true);
        }
        if !is_valid_member_transition(from, to) {
            debug!(
                team = %self.team_name,
                member = %self.member_name,
                from = from.as_str(),
                to = to.as_str(),
                "rejecting illegal member status transition"
            );
            return Ok(false);
        }
        let applied = self
            .store
            .transition_member(&self.team_name, &self.member_name, from, to)
            .await?;
        if applied {
            self.record.write().status = to;
            debug!(
                team = %self.team_name,
                member = %self.member_name,
                from = from.as_str(),
                to = to.as_str(),
                "member status changed"
            );
            Ok(true)
        } else {
            // Lost race: another writer moved the row first. Report that, and make the cache
            // agree with the store so the next decision starts from the truth.
            debug!(
                team = %self.team_name,
                member = %self.member_name,
                from = from.as_str(),
                to = to.as_str(),
                "member status transition lost its compare-and-set"
            );
            self.reload_cache().await?;
            Ok(false)
        }
    }

    /// Sets the execution status through a patch write.
    ///
    /// Execution status has no compare-and-set in the store (Python validates it in the member
    /// DAO); the legality check therefore happens here, against the same transition table.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`] for an edge the execution table rejects, and
    /// [`Error::not_found`] when the member row does not exist.
    pub async fn mark_execution(&self, to: ExecutionStatus) -> Result<()> {
        let from = self.execution();
        if from == to {
            return Ok(());
        }
        if !from.can_transition_to(to) {
            return Err(Error::validation(format!(
                "member '{}' cannot move execution status from '{}' to '{}'",
                self.member_name,
                from.as_str(),
                to.as_str()
            )));
        }
        let patch = MemberPatch::new().with_execution(to);
        if self
            .store
            .update_member(&self.team_name, &self.member_name, &patch)
            .await?
        {
            self.record.write().execution = to;
            debug!(
                team = %self.team_name,
                member = %self.member_name,
                from = from.as_str(),
                to = to.as_str(),
                "member execution status changed"
            );
            Ok(())
        } else {
            Err(Error::not_found(format!(
                "member '{}' not found in team '{}'",
                self.member_name, self.team_name
            )))
        }
    }

    /// Applies one round outcome as the [`round_transitions`] walk, then mirrors it onto the
    /// lifecycle status.
    ///
    /// Steps the current execution status cannot reach are skipped (a `finished` note arriving
    /// while already `idle` has nothing to replay), which makes the call idempotent for repeated
    /// or late observer callbacks. A walk that ends `running` marks the member `busy`; one that
    /// ends `idle` marks it `ready`. The lifecycle edge is applied only where the member table
    /// allows it, and only when the walk actually advanced (or the member is still `busy` /
    /// `starting`) — a duplicate terminal note must not revive a `paused` or `stopped` member.
    ///
    /// # Errors
    ///
    /// Propagates store failures from either the execution or the status write.
    pub async fn note_round(&self, outcome: RoundOutcome) -> Result<()> {
        let mut advanced = false;
        for step in round_transitions(outcome) {
            let current = self.execution();
            if current == step {
                continue;
            }
            if !current.can_transition_to(step) {
                debug!(
                    team = %self.team_name,
                    member = %self.member_name,
                    from = current.as_str(),
                    to = step.as_str(),
                    outcome = ?outcome,
                    "skipping unreachable execution transition in round walk"
                );
                continue;
            }
            self.mark_execution(step).await?;
            advanced = true;
        }

        let releases =
            advanced || matches!(self.status(), MemberStatus::Busy | MemberStatus::Starting);
        if !releases {
            debug!(
                team = %self.team_name,
                member = %self.member_name,
                outcome = ?outcome,
                status = self.status().as_str(),
                "round note changed nothing; leaving the member status alone"
            );
            return Ok(());
        }
        let settled = match self.execution() {
            ExecutionStatus::Running => Some(MemberStatus::Busy),
            ExecutionStatus::Idle => Some(MemberStatus::Ready),
            _ => None,
        };
        if let Some(status) = settled {
            if !self.mark(status).await? {
                debug!(
                    team = %self.team_name,
                    member = %self.member_name,
                    status = status.as_str(),
                    "round outcome has no legal lifecycle edge; leaving status unchanged"
                );
            }
        }
        Ok(())
    }

    /// Starts the member: CAS `unstarted -> starting`, run the runtime, settle to `ready`.
    ///
    /// A failed [`MemberRuntime::start`] rolls the row back to `unstarted` (best effort — a failed
    /// rollback is logged, never masking the original error) and returns the runtime's error
    /// unchanged. The member must be `unstarted` and must have a runtime attached; anything else is
    /// a caller bug, not a race.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`] when the member is not `unstarted`,
    /// [`Error::config`] when no runtime is attached, the runtime's own error on start failure,
    /// and propagates store failures.
    pub async fn start(&self, team_session: Option<Arc<dyn Session>>) -> Result<()> {
        let Some(runtime) = self.runtime() else {
            return Err(Error::config(format!(
                "member '{}' has no runtime attached",
                self.member_name
            )));
        };
        if self.status() != MemberStatus::Unstarted {
            return Err(Error::validation(format!(
                "member '{}' is not in unstarted state",
                self.member_name
            )));
        }
        if !self.mark(MemberStatus::Starting).await? {
            return Err(Error::validation(format!(
                "member '{}' is not in unstarted state",
                self.member_name
            )));
        }
        if let Err(error) = runtime.start(team_session).await {
            if let Err(rollback) = self.rollback_start().await {
                warn!(
                    team = %self.team_name,
                    member = %self.member_name,
                    error = %rollback,
                    "failed to roll the member status back to unstarted"
                );
            }
            return Err(error);
        }
        if !self.mark(MemberStatus::Ready).await? {
            debug!(
                team = %self.team_name,
                member = %self.member_name,
                "member left starting before the ready write; keeping the winner's status"
            );
        }
        Ok(())
    }

    /// Settles a retire request into the status [`retire_target`] prescribes.
    ///
    /// The leader-side request writes `shutdown_requested`; this member-side settle works from any
    /// live status: it first moves to `shutdown_requested` (when the table allows it), then to the
    /// terminal target — `shutdown` for resident members and every forced retire, `dormant` for a
    /// task-scoped member. When `shutdown_requested` is unreachable from the current status the
    /// target is written directly (a never-started member can only reach `shut_down`). The
    /// returned status is what the row actually holds after the settle, so an already-retired
    /// member reports its terminal status instead of an error.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`] when neither `shutdown_requested` nor the target is reachable
    /// from the member's current status (Python rejects the same request), and propagates store
    /// failures.
    pub async fn shutdown(&self, lifetime: MemberLifetime, force: bool) -> Result<MemberStatus> {
        let target = retire_target(Some(lifetime), force);
        if self.status() == target {
            return Ok(target);
        }
        let mut status = self.status();
        if status != MemberStatus::ShutdownRequested
            && self.mark(MemberStatus::ShutdownRequested).await?
        {
            status = MemberStatus::ShutdownRequested;
        }
        if status == MemberStatus::ShutdownRequested {
            if self.mark(target).await? {
                return Ok(target);
            }
        } else if self.mark(target).await? {
            return Ok(self.status());
        }

        let record = self.refresh().await?;
        if record.status == target {
            return Ok(target);
        }
        Err(Error::validation(format!(
            "member '{}' cannot shut down from status '{}'",
            self.member_name,
            record.status.as_str()
        )))
    }

    /// Records that the member stopped with an error.
    ///
    /// The message is logged (tracing) and the row moves to `error` where the table allows it. A
    /// member that is already gone (`shut_down`, `dormant`) keeps its status: reporting a crash
    /// must never fail the crash path.
    ///
    /// # Errors
    ///
    /// Propagates store failures.
    pub async fn error(&self, message: &str) -> Result<()> {
        warn!(
            team = %self.team_name,
            member = %self.member_name,
            message,
            "member reported an error"
        );
        if !self.mark(MemberStatus::Error).await? {
            debug!(
                team = %self.team_name,
                member = %self.member_name,
                status = self.status().as_str(),
                "member cannot enter error from its current status"
            );
        }
        Ok(())
    }

    /// Rolls a failed start back from `starting` to `unstarted`.
    async fn rollback_start(&self) -> Result<()> {
        let applied = self
            .store
            .transition_member(
                &self.team_name,
                &self.member_name,
                MemberStatus::Starting,
                MemberStatus::Unstarted,
            )
            .await?;
        if applied {
            self.record.write().status = MemberStatus::Unstarted;
            debug!(
                team = %self.team_name,
                member = %self.member_name,
                "rolled member status back to unstarted"
            );
        } else {
            self.reload_cache().await?;
        }
        Ok(())
    }

    /// Copies the stored row into the cache when it still exists.
    async fn reload_cache(&self) -> Result<()> {
        if let Some(stored) = self
            .store
            .member(&self.team_name, &self.member_name)
            .await?
        {
            *self.record.write() = stored;
        }
        Ok(())
    }
}

impl fmt::Debug for TeamMember {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TeamMember")
            .field("team_name", &self.team_name)
            .field("member_name", &self.member_name)
            .field("status", &self.status())
            .field("execution", &self.execution())
            .field("has_runtime", &self.runtime.read().is_some())
            .finish_non_exhaustive()
    }
}
