//! [`TeamHealth`]: the member-liveness projection of a running team.
//!
//! The roster already answers "which members exist" and each member handles already answer "which
//! status, which execution phase, is a brain attached". This module folds those three answers into
//! one flat, serializable projection so a host (the CLI, the MCP server, a monitor) can answer
//! "who is alive right now" with a single synchronous call, without spawning a task, probing a
//! transport or reading the store.
//!
//! Python parity: the Python runtime computes the same view on demand from its live member
//! objects (`openjiuwen/agent_teams/agent/team_agent.py`, `get_team_status`). The Rust port keeps
//! it a pure projection over [`crate::member::TeamMember`], so it can never disagree with the
//! handles the coordination loop drives.
//!
//! Nothing here stores state, awaits or allocates a task: the crate-internal roster constructor
//! reads the handles it is given and returns an owned snapshot.
//!
//! ```text
//! alive = a runtime is attached
//!         AND that runtime's state is not `Terminated`
//!         AND the row's status is not `Unstarted`
//! ```
//!
//! The status clause matters because the build path (`TeamAgentSpec::build_with`) attaches the
//! leader's harness before the first start: the runtime is already `Idle` while the row is still
//! `Unstarted`. Without the clause a built-but-never-started team would report its leader alive.
use crate::member::TeamMember;
use crate::schema::{ExecutionStatus, HarnessState, MemberStatus, TeamRole};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
/// Liveness of one member.
///
/// A pure snapshot: the identity and both statuses come from the member handle's cached row, and
/// `alive` comes from the attached runtime's phase. Serializes with `snake_case` keys and tolerates
/// unknown fields, so a Python-side consumer can read it as-is.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct MemberHealth {
    /// Member identity, unique within the team.
    pub name: String,
    /// Roster role (`leader`, `teammate`, …).
    pub role: TeamRole,
    /// Lifecycle status of the member row (`unstarted`, `ready`, `shut_down`, …).
    pub status: MemberStatus,
    /// Detail status of the member's current task execution (`idle`, `running`, …).
    pub execution: ExecutionStatus,
    /// `true` when a runtime is attached, that runtime is not terminated, and the row started.
    pub alive: bool,
}
/// Liveness of a whole team: the team identity, its run flag and its whole roster.
///
/// `members` covers every live roster handle, leader included, sorted by name; `leader` repeats
/// the leader's entry so a consumer need not know the leader's name.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct TeamHealth {
    /// Team name every member row is scoped to.
    pub team_name: String,
    /// `true` between a completed team start and the next stop.
    pub started: bool,
    /// The leader's entry, when the roster holds a row for the leader.
    pub leader: Option<MemberHealth>,
    /// Every roster member (leader included), sorted by name.
    pub members: Vec<MemberHealth>,
}
impl TeamHealth {
    /// Projects a roster of live member handles into a team health snapshot.
    ///
    /// `leader_name` selects which entry is repeated in [`Self::leader`]; a blank or unknown name
    /// leaves it `None`. `started` stays `false` here because the backend does not track the run
    /// lifecycle — a handle that does (`crate::runtime::team_agent::TeamAgent`) fills it in.
    #[must_use]
    pub(crate) fn from_roster(
        team_name: &str,
        leader_name: Option<&str>,
        members: &[Arc<TeamMember>],
    ) -> Self {
        let mut entries: Vec<MemberHealth> = members
            .iter()
            .map(|member| member_health(member.as_ref()))
            .collect();
        entries.sort_by(|left, right| left.name.cmp(&right.name));
        let leader = leader_name
            .filter(|name| !name.is_empty())
            .and_then(|name| entries.iter().find(|entry| entry.name == name).cloned());
        Self {
            team_name: team_name.to_owned(),
            started: false,
            leader,
            members: entries,
        }
    }
}
/// Projects one member handle into its health entry.
///
/// `alive` requires all three parts of the rule: a runtime is attached, that runtime's phase is not
/// [`HarnessState::Terminated`], and the row's status is not [`MemberStatus::Unstarted`]. The last
/// clause is what keeps a built-but-not-yet-started team honest: `TeamAgentSpec`'s build path
/// attaches the leader's harness, so its runtime is already `Idle` while the row has never
/// started — only the status says the run has not begun. A row that was never spawned reads
/// `false` (no runtime), and a torn-down member reads `false` while its handle stays in the roster.
pub(crate) fn member_health(member: &TeamMember) -> MemberHealth {
    let status = member.status();
    let alive = status != MemberStatus::Unstarted
        && member
            .runtime()
            .is_some_and(|runtime| runtime.state() != HarnessState::Terminated);
    MemberHealth {
        name: member.member_name().to_owned(),
        role: member.role(),
        status,
        execution: member.execution(),
        alive,
    }
}
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::member::{MemberObserver, MemberRuntime};
    use crate::schema::TeamRole;
    use crate::storage::{InMemoryTeamStore, MemberRecord, TeamStore};
    use async_trait::async_trait;
    use openjiuwen_common::error::Result;
    use openjiuwen_session::Session;
    use parking_lot::Mutex;
    /// A runtime whose phase a test sets directly; it never starts a round.
    struct FakeRuntime {
        /// Phase reported by [`MemberRuntime::state`].
        state: Mutex<HarnessState>,
    }
    impl FakeRuntime {
        /// Builds a runtime parked in `state`.
        fn new(state: HarnessState) -> Arc<Self> {
            Arc::new(Self {
                state: Mutex::new(state),
            })
        }
    }
    #[async_trait]
    #[allow(clippy::needless_pass_by_value)]
    impl MemberRuntime for FakeRuntime {
        fn session_id(&self) -> &'static str {
            "session-a"
        }
        fn state(&self) -> HarnessState {
            *self.state.lock()
        }
        async fn start(&self, _team_session: Option<Arc<dyn Session>>) -> Result<()> {
            Ok(())
        }
        async fn stop(&self) -> Result<()> {
            Ok(())
        }
        async fn send(&self, _content: String, _immediate: bool) -> Result<()> {
            Ok(())
        }
        async fn abort(&self, _immediate: bool) -> Result<()> {
            Ok(())
        }
        async fn pause(&self) -> Result<()> {
            Ok(())
        }
        async fn resume(&self, _query: Option<String>) -> Result<()> {
            Ok(())
        }
        fn subscribe(&self, _observer: Arc<dyn MemberObserver>) {}
    }
    /// Builds one member handle over a fresh in-memory store.
    fn member(name: &str, role: TeamRole) -> Arc<TeamMember> {
        let store: Arc<InMemoryTeamStore> = Arc::new(InMemoryTeamStore::new());
        Arc::new(TeamMember::new(
            store as Arc<dyn TeamStore>,
            MemberRecord::new("alpha", name, role),
        ))
    }
    /// Builds one member handle whose row has already started, through the same store compare-and-set
    /// the runtime uses (`insert_member` then `TeamMember::mark`).
    async fn started_member(name: &str, role: TeamRole) -> Arc<TeamMember> {
        let store: Arc<InMemoryTeamStore> = Arc::new(InMemoryTeamStore::new());
        let record = MemberRecord::new("alpha", name, role);
        store.insert_member(record.clone()).await.unwrap();
        let member = Arc::new(TeamMember::new(store as Arc<dyn TeamStore>, record));
        assert!(
            member.mark(MemberStatus::Ready).await.unwrap(),
            "the store accepts unstarted -> ready"
        );
        assert_eq!(member.status(), MemberStatus::Ready);
        member
    }
    #[tokio::test]
    async fn alive_needs_a_runtime_a_live_phase_and_a_started_row() {
        let unstarted = member("dev-1", TeamRole::Teammate);
        assert!(
            !member_health(&unstarted).alive,
            "a row with no runtime is not alive"
        );
        unstarted.attach_runtime(FakeRuntime::new(HarnessState::Idle) as Arc<dyn MemberRuntime>);
        assert!(
            !member_health(&unstarted).alive,
            "an idle harness attached before the row started is not alive"
        );
        let started = started_member("dev-1", TeamRole::Teammate).await;
        started.attach_runtime(FakeRuntime::new(HarnessState::Idle) as Arc<dyn MemberRuntime>);
        assert!(
            member_health(&started).alive,
            "a started row with an attached idle runtime is alive"
        );
        started
            .attach_runtime(FakeRuntime::new(HarnessState::Terminated) as Arc<dyn MemberRuntime>);
        assert!(
            !member_health(&started).alive,
            "a terminated runtime is no longer alive"
        );
    }
    #[test]
    fn member_health_reports_identity_and_statuses() {
        let member = member("dev-1", TeamRole::Teammate);
        let health = member_health(&member);
        assert_eq!(health.name, "dev-1");
        assert_eq!(health.role, TeamRole::Teammate);
        assert_eq!(health.status, MemberStatus::Unstarted);
        assert_eq!(health.execution, ExecutionStatus::Idle);
    }
    #[test]
    fn from_roster_sorts_members_and_copies_the_leader() {
        let roster = vec![
            member("dev-1", TeamRole::Teammate),
            member("boss", TeamRole::Leader),
            member("dev-2", TeamRole::Teammate),
        ];
        let health = TeamHealth::from_roster("alpha", Some("boss"), &roster);
        assert_eq!(health.team_name, "alpha");
        assert!(!health.started, "the backend never reports a start");
        let names: Vec<&str> = health.members.iter().map(|m| m.name.as_str()).collect();
        assert_eq!(names, ["boss", "dev-1", "dev-2"], "sorted by name");
        let leader = health.leader.expect("the leader row is present");
        assert_eq!(leader.name, "boss");
        assert_eq!(leader.role, TeamRole::Leader);
        assert_eq!(Some(&leader), health.members.first());
    }
    #[test]
    fn from_roster_without_a_known_leader_leaves_it_none() {
        let roster = vec![member("dev-1", TeamRole::Teammate)];
        assert!(TeamHealth::from_roster("alpha", None, &roster)
            .leader
            .is_none());
        assert!(TeamHealth::from_roster("alpha", Some(""), &roster)
            .leader
            .is_none());
        assert!(
            TeamHealth::from_roster("alpha", Some("boss"), &roster)
                .leader
                .is_none(),
            "an unknown leader name selects nothing"
        );
        let empty = TeamHealth::from_roster("alpha", Some("boss"), &[]);
        assert_eq!(empty.members, Vec::<MemberHealth>::new());
        assert!(empty.leader.is_none());
    }
    #[test]
    fn health_serializes_with_snake_case_keys() {
        let roster = vec![member("boss", TeamRole::Leader)];
        let value =
            serde_json::to_value(TeamHealth::from_roster("alpha", Some("boss"), &roster)).unwrap();
        let object = value.as_object().unwrap();
        let mut keys: Vec<&str> = object.keys().map(String::as_str).collect();
        keys.sort_unstable();
        assert_eq!(keys, ["leader", "members", "started", "team_name"]);
        assert_eq!(object["team_name"], serde_json::json!("alpha"));
        let member = object["members"][0].as_object().unwrap();
        let mut member_keys: Vec<&str> = member.keys().map(String::as_str).collect();
        member_keys.sort_unstable();
        assert_eq!(
            member_keys,
            ["alive", "execution", "name", "role", "status"]
        );
        assert_eq!(member["status"], serde_json::json!("unstarted"));
        assert_eq!(member["execution"], serde_json::json!("idle"));
        assert_eq!(member["role"], serde_json::json!("leader"));
        assert_eq!(member["alive"], serde_json::json!(false));
    }
    #[test]
    fn health_tolerates_unknown_and_missing_fields() {
        let health: TeamHealth = serde_json::from_value(serde_json::json!({
            "team_name": "alpha",
            "members": [{"name": "dev-1", "status": "ready", "extra": 1}],
            "surprise": true,
        }))
        .unwrap();
        assert_eq!(health.team_name, "alpha");
        assert!(!health.started);
        assert!(health.leader.is_none());
        assert_eq!(health.members.len(), 1);
        assert_eq!(health.members[0].status, MemberStatus::Ready);
        assert_eq!(health.members[0].role, TeamRole::Leader, "defaulted role");
        assert!(!health.members[0].alive);
    }
}
