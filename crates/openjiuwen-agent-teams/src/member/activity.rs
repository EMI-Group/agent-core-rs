//! [`MemberActivityRegistry`]: the leader-side projection of every member's activity, plus the
//! pure [`IdleDebounce`] window that decides when the team's idle marker may fire.
//!
//! Python parity source: `openjiuwen/agent_teams/agent/member_activity.py`. Python's registry is a
//! name → status map that also answers "what does the pending `team.idle` marker owe?"
//! (`IdleSignal`, armed on the first movement). The Rust port replaces the signal enum with an
//! explicit registry (which members are moving, which have settled, since when they have been
//! idle) because `TeamBackend` owns the marker itself and only needs the questions, not the
//! tallying. The debounce window lives here as a **pure** predicate so no timer, task or sleep is
//! involved in this node — the caller owns the clock.
//!
//! The registry holds `Arc<TeamMember>` handles, so its status answers follow the handles' cached
//! rows: the owner refreshes them (or records status events through `mark_busy` / `mark_idle`) as
//! the roster moves.
//!
//! ```
//! use std::sync::Arc;
//! use std::time::{Duration, Instant};
//! use openjiuwen_agent_teams::member::{IdleDebounce, MemberActivityRegistry, TeamMember};
//! use openjiuwen_agent_teams::schema::{StorageSpec, TeamRole};
//! use openjiuwen_agent_teams::storage::{open_store, MemberRecord};
//!
//! # async fn demo() -> openjiuwen_common::error::Result<()> {
//! let store = open_store(&StorageSpec::in_memory())?;
//! let member = Arc::new(TeamMember::new(
//!     store,
//!     MemberRecord::new("alpha", "dev-1", TeamRole::Teammate),
//! ));
//! let registry = MemberActivityRegistry::new();
//! registry.register(member);
//! assert!(!registry.is_moving());
//! assert_eq!(registry.settled_members(), Vec::<String>::new()); // `unstarted` is quiescent, not settled
//!
//! let debounce = IdleDebounce { window: Duration::from_secs(2) };
//! let started = Instant::now();
//! assert!(!debounce.should_emit_idle(started + Duration::from_millis(1999), Some(started)));
//! assert!(debounce.should_emit_idle(started + Duration::from_secs(2), Some(started)));
//! # Ok(())
//! # }
//! ```

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use tracing::trace;

use crate::schema::{ExecutionStatus, MemberSnapshot, MemberStatus};

use super::lifecycle::TeamMember;

/// Default window the team must hold still before the idle marker is emitted.
///
/// Python `_TEAM_IDLE_DEBOUNCE_SECONDS` in `agent/stream_controller.py`.
pub const DEFAULT_IDLE_DEBOUNCE: Duration = Duration::from_secs(2);

/// Returns `true` when the member is currently doing something.
///
/// A member moves while its lifecycle status is `starting`, `busy` or `restarting`, or while its
/// execution status is `running`, `completing` or `cancelling` — the exact pair Python's
/// `is_moving` check folds over the roster.
fn member_is_moving(status: MemberStatus, execution: ExecutionStatus) -> bool {
    matches!(
        status,
        MemberStatus::Starting | MemberStatus::Busy | MemberStatus::Restarting
    ) || matches!(
        execution,
        ExecutionStatus::Running | ExecutionStatus::Completing | ExecutionStatus::Cancelling
    )
}

/// One tracked member: its handle, when it went idle, and when it last moved.
#[derive(Debug)]
struct ActivityEntry {
    member: Arc<TeamMember>,
    idle_since: Option<Instant>,
    last_activity: Option<Instant>,
}

/// The leader-side view of who is working.
///
/// Not persisted and not shared across processes: it is a projection the leader rebuilds at the
/// start of every run cycle and keeps current from status events. Members are keyed by
/// `member_name`, and every listing is returned in name order so callers (and tests) see a stable
/// roster.
#[derive(Debug, Default)]
pub struct MemberActivityRegistry {
    entries: Mutex<BTreeMap<String, ActivityEntry>>,
}

impl MemberActivityRegistry {
    /// Creates an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds — or re-homes — a member in the registry.
    ///
    /// Re-registering preserves the existing idle timestamp (the member did not move because its
    /// handle was replaced). A member that is already moving is registered as active; a resting
    /// one starts its idle clock now.
    pub fn register(&self, member: Arc<TeamMember>) {
        let member_name = member.member_name().to_string();
        let moving = member_is_moving(member.status(), member.execution());
        let now = Instant::now();
        let mut entries = self.entries.lock();
        if let Some(entry) = entries.get_mut(&member_name) {
            entry.member = member;
            return;
        }
        entries.insert(
            member_name,
            ActivityEntry {
                member,
                idle_since: if moving { None } else { Some(now) },
                last_activity: if moving { Some(now) } else { None },
            },
        );
    }

    /// Records that `member_name` started working.
    ///
    /// Unknown names are ignored: the registry only reports on members the owner registered.
    pub fn mark_busy(&self, member_name: &str) {
        let mut entries = self.entries.lock();
        if let Some(entry) = entries.get_mut(member_name) {
            entry.idle_since = None;
            entry.last_activity = Some(Instant::now());
        } else {
            trace!(
                member = member_name,
                "ignoring activity for unregistered member"
            );
        }
    }

    /// Records that `member_name` came to rest.
    ///
    /// Idempotent: the first idle timestamp is kept, so a repeated status event cannot postpone
    /// the idle marker forever.
    pub fn mark_idle(&self, member_name: &str) {
        let now = Instant::now();
        let mut entries = self.entries.lock();
        if let Some(entry) = entries.get_mut(member_name) {
            entry.idle_since.get_or_insert(now);
            entry.last_activity = Some(now);
        } else {
            trace!(
                member = member_name,
                "ignoring activity for unregistered member"
            );
        }
    }

    /// Returns `true` while any registered member is moving.
    ///
    /// The leader includes itself in the roster, so a leader mid-dispatch keeps the team moving.
    #[must_use]
    pub fn is_moving(&self) -> bool {
        self.entries
            .lock()
            .values()
            .any(|entry| member_is_moving(entry.member.status(), entry.member.execution()))
    }

    /// Returns when `member_name` last came to rest, if it is at rest and known.
    ///
    /// `None` means either "unknown member" or "currently moving" — callers that must tell the two
    /// apart check [`Self::is_moving`] first.
    #[must_use]
    pub fn idle_since(&self, member_name: &str) -> Option<Instant> {
        self.entries
            .lock()
            .get(member_name)
            .and_then(|entry| entry.idle_since)
    }

    /// Returns the most recent movement across the roster, if anything has moved at all.
    ///
    /// Paired with [`IdleDebounce::should_emit_idle`] this answers "may the idle marker fire?".
    #[must_use]
    pub fn last_activity(&self) -> Option<Instant> {
        self.entries
            .lock()
            .values()
            .filter_map(|entry| entry.last_activity)
            .max()
    }

    /// Returns the members that have settled, sorted by name.
    ///
    /// Settled means the lifecycle status is one of
    /// [`MemberStatus::is_settled`](MemberStatus::is_settled) *and* nothing is executing — a
    /// `paused` member still finalising a round is not settled.
    #[must_use]
    pub fn settled_members(&self) -> Vec<String> {
        self.entries
            .lock()
            .iter()
            .filter(|(_, entry)| {
                entry.member.status().is_settled()
                    && !member_is_moving(entry.member.status(), entry.member.execution())
            })
            .map(|(name, _)| name.clone())
            .collect()
    }

    /// Returns the roster projection, sorted by member name.
    #[must_use]
    pub fn snapshot(&self) -> Vec<MemberSnapshot> {
        self.entries
            .lock()
            .values()
            .map(|entry| {
                let record = entry.member.record();
                let display_name = record.display_name_or_default().to_string();
                MemberSnapshot {
                    member_name: record.member_name,
                    display_name,
                    role: record.role,
                    status: record.status,
                    execution: record.execution,
                }
            })
            .collect()
    }

    /// Removes a member from the registry and returns its handle.
    pub fn remove(&self, member_name: &str) -> Option<Arc<TeamMember>> {
        self.entries
            .lock()
            .remove(member_name)
            .map(|entry| entry.member)
    }

    /// Returns the number of registered members.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.lock().len()
    }

    /// Returns `true` when no member is registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.lock().is_empty()
    }
}

/// A debounce window for the team-idle marker.
///
/// The roster crosses the quiescent set constantly in the seams between rounds — a teammate
/// settling to `ready` a beat before the leader picks its next move looks exactly like a finished
/// team for a few milliseconds. The marker is only meaningful if the quiet holds, so the caller
/// arms a timer with this window and re-checks [`Self::should_emit_idle`] when it fires.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IdleDebounce {
    /// How long the roster must hold still before the marker may fire.
    pub window: Duration,
}

impl IdleDebounce {
    /// Creates a debounce with an explicit window.
    #[must_use]
    pub const fn new(window: Duration) -> Self {
        Self { window }
    }

    /// Returns `true` when the quiet has lasted at least [`Self::window`].
    ///
    /// `last_activity == None` is `false` by design: a team that never moved owes no idle marker
    /// (Python's registry starts disarmed for the same reason). The window boundary is inclusive,
    /// and a clock that moved backwards saturates to "not yet".
    #[must_use]
    pub fn should_emit_idle(&self, now: Instant, last_activity: Option<Instant>) -> bool {
        last_activity.is_some_and(|last| now.saturating_duration_since(last) >= self.window)
    }
}

impl Default for IdleDebounce {
    /// Python's `_TEAM_IDLE_DEBOUNCE_SECONDS`: two seconds.
    fn default() -> Self {
        Self::new(DEFAULT_IDLE_DEBOUNCE)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::schema::StorageSpec;
    use crate::storage::{open_store, MemberRecord};

    fn member(name: &str, status: MemberStatus, execution: ExecutionStatus) -> Arc<TeamMember> {
        let store = open_store(&StorageSpec::in_memory()).unwrap();
        let mut record = MemberRecord::new("alpha", name, crate::schema::TeamRole::Teammate);
        record.display_name = format!("{name} display");
        record.status = status;
        record.execution = execution;
        Arc::new(TeamMember::new(store, record))
    }

    #[test]
    fn a_resting_member_does_not_move_and_starts_its_idle_clock() {
        let registry = MemberActivityRegistry::new();
        assert!(registry.is_empty());
        registry.register(member("dev-1", MemberStatus::Ready, ExecutionStatus::Idle));

        assert_eq!(registry.len(), 1);
        assert!(!registry.is_moving());
        assert!(registry.idle_since("dev-1").is_some());
        assert_eq!(registry.settled_members(), vec!["dev-1".to_string()]);
    }

    #[test]
    fn an_unstarted_member_is_quiescent_but_not_settled() {
        let registry = MemberActivityRegistry::new();
        registry.register(member(
            "dev-1",
            MemberStatus::Unstarted,
            ExecutionStatus::Idle,
        ));
        assert!(!registry.is_moving());
        assert_eq!(registry.settled_members(), Vec::<String>::new());
    }

    #[test]
    fn a_busy_status_makes_the_team_move() {
        let registry = MemberActivityRegistry::new();
        registry.register(member(
            "dev-1",
            MemberStatus::Busy,
            ExecutionStatus::Running,
        ));
        assert!(registry.is_moving());
        assert!(registry.idle_since("dev-1").is_none());
        assert_eq!(registry.settled_members(), Vec::<String>::new());
    }

    #[test]
    fn a_running_execution_makes_the_team_move_even_when_ready() {
        let registry = MemberActivityRegistry::new();
        registry.register(member(
            "dev-1",
            MemberStatus::Ready,
            ExecutionStatus::Running,
        ));
        assert!(registry.is_moving());
        assert_eq!(registry.settled_members(), Vec::<String>::new());
    }

    #[test]
    fn any_moving_member_keeps_the_whole_team_moving() {
        let registry = MemberActivityRegistry::new();
        registry.register(member("dev-1", MemberStatus::Ready, ExecutionStatus::Idle));
        registry.register(member(
            "dev-2",
            MemberStatus::Restarting,
            ExecutionStatus::Idle,
        ));
        assert!(registry.is_moving());
        assert_eq!(registry.settled_members(), vec!["dev-1".to_string()]);
    }

    #[test]
    fn mark_busy_and_mark_idle_drive_the_idle_clock() {
        let registry = MemberActivityRegistry::new();
        registry.register(member("dev-1", MemberStatus::Ready, ExecutionStatus::Idle));
        let first = registry.idle_since("dev-1").unwrap();

        registry.mark_busy("dev-1");
        assert!(registry.idle_since("dev-1").is_none());

        registry.mark_idle("dev-1");
        let second = registry.idle_since("dev-1").unwrap();
        assert!(second >= first);
        registry.mark_idle("dev-1");
        assert_eq!(registry.idle_since("dev-1"), Some(second));
    }

    #[test]
    fn unknown_names_are_ignored_by_the_activity_hooks() {
        let registry = MemberActivityRegistry::new();
        registry.mark_busy("ghost");
        registry.mark_idle("ghost");
        assert!(registry.is_empty());
        assert!(registry.idle_since("ghost").is_none());
    }

    #[test]
    fn last_activity_follows_the_most_recent_movement() {
        let registry = MemberActivityRegistry::new();
        assert!(registry.last_activity().is_none());
        registry.register(member("dev-1", MemberStatus::Ready, ExecutionStatus::Idle));
        assert!(registry.last_activity().is_none());
        registry.mark_busy("dev-1");
        assert!(registry.last_activity().is_some());
    }

    #[test]
    fn snapshot_is_sorted_and_carries_the_roster_projection() {
        let registry = MemberActivityRegistry::new();
        registry.register(member("zeta", MemberStatus::Ready, ExecutionStatus::Idle));
        registry.register(member(
            "alpha",
            MemberStatus::Busy,
            ExecutionStatus::Running,
        ));

        let snapshot = registry.snapshot();
        assert_eq!(snapshot.len(), 2);
        assert_eq!(snapshot[0].member_name, "alpha");
        assert_eq!(snapshot[0].status, MemberStatus::Busy);
        assert_eq!(snapshot[0].execution, ExecutionStatus::Running);
        assert_eq!(snapshot[0].display_name, "alpha display");
        assert_eq!(snapshot[1].member_name, "zeta");
    }

    #[test]
    fn remove_forgets_a_member() {
        let registry = MemberActivityRegistry::new();
        registry.register(member(
            "dev-1",
            MemberStatus::Busy,
            ExecutionStatus::Running,
        ));
        assert!(registry.is_moving());

        let removed = registry.remove("dev-1").expect("registered member");
        assert_eq!(removed.member_name(), "dev-1");
        assert!(!registry.is_moving());
        assert_eq!(registry.snapshot(), Vec::new());
        assert!(registry.remove("dev-1").is_none());
    }

    #[test]
    fn re_registering_preserves_the_idle_clock() {
        let registry = MemberActivityRegistry::new();
        registry.register(member("dev-1", MemberStatus::Ready, ExecutionStatus::Idle));
        let idle_since = registry.idle_since("dev-1").unwrap();

        registry.register(member("dev-1", MemberStatus::Ready, ExecutionStatus::Idle));

        assert_eq!(registry.len(), 1);
        assert_eq!(registry.idle_since("dev-1"), Some(idle_since));
    }

    #[test]
    fn idle_debounce_window_boundary() {
        let debounce = IdleDebounce::default();
        assert_eq!(debounce.window, Duration::from_secs(2));
        let started = Instant::now();

        assert!(!debounce.should_emit_idle(started, None));
        assert!(!debounce.should_emit_idle(started + Duration::from_millis(1999), Some(started)));
        assert!(debounce.should_emit_idle(started + Duration::from_secs(2), Some(started)));
        assert!(debounce.should_emit_idle(started + Duration::from_secs(30), Some(started)));
    }

    #[test]
    fn idle_debounce_tolerates_a_backwards_clock() {
        let debounce = IdleDebounce::new(Duration::from_secs(1));
        let later = Instant::now();
        let earlier = later.checked_sub(Duration::from_millis(50)).unwrap();
        assert!(!debounce.should_emit_idle(earlier, Some(later)));
    }

    #[test]
    fn a_zero_window_emits_immediately() {
        let debounce = IdleDebounce::new(Duration::ZERO);
        let now = Instant::now();
        assert!(debounce.should_emit_idle(now, Some(now)));
    }
}
