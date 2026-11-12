//! Lifecycle vocabulary: member, execution and task statuses.
//!
//! This module is pure data — the status enums, their transition tables, the status-set constants
//! and the pure helpers that answer transition questions. Every other module of the crate asks
//! these questions *here* instead of repeating the rules, so the tables are the single source of
//! truth (see `src/schema/CONTEXT.md`).
//!
//! Wire-visible: every enum serializes to the Python string (`shut_down`, `in_progress`, …) and
//! `Default` matches the Python default. Python parity source:
//! `openjiuwen/agent_teams/schema/status.py`.

use serde::{Deserialize, Serialize};

/// Lifecycle status of a team member.
///
/// Wire values are the Python strings; `ShutDown` serializes as `shut_down`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
#[serde(rename_all = "snake_case")]
pub enum MemberStatus {
    /// Created but not yet started.
    Unstarted,
    /// The member agent is being spawned (transitional, guards duplicate spawn).
    Starting,
    /// Ready to receive tasks.
    Ready,
    /// Currently processing a task.
    Busy,
    /// Idle at the end of a round; revivable via `resume`.
    Paused,
    /// Runtime torn down by an external stop while the team stays live; revivable by `recover_team`.
    Stopped,
    /// Being restarted (after failure, or as a dormant member's wake path).
    Restarting,
    /// A shutdown request has been written but the member may still be finishing a last round.
    ShutdownRequested,
    /// Terminal: the member is gone.
    ShutDown,
    /// Retired after its last task with `options.lifetime == "dormant"`; the roster row stays and
    /// a single-recipient wake moves it `dormant -> restarting -> ready`.
    Dormant,
    /// The member stopped without finishing; recoverable through `restarting` or `ready`.
    Error,
}

impl Default for MemberStatus {
    /// Python default: `unstarted`.
    fn default() -> Self {
        Self::Unstarted
    }
}

impl MemberStatus {
    /// Returns the Python wire string of this status.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Unstarted => "unstarted",
            Self::Starting => "starting",
            Self::Ready => "ready",
            Self::Busy => "busy",
            Self::Paused => "paused",
            Self::Stopped => "stopped",
            Self::Restarting => "restarting",
            Self::ShutdownRequested => "shutdown_requested",
            Self::ShutDown => "shut_down",
            Self::Dormant => "dormant",
            Self::Error => "error",
        }
    }

    /// Returns `true` when the leader has released the member (shutdown requested or already gone).
    ///
    /// Guards that protect a member's in-flight work lift here, at the *request*, not at the
    /// terminal state.
    #[must_use]
    pub const fn is_departed(self) -> bool {
        member_status_in(&MEMBER_DEPARTED_STATUSES, self)
    }

    /// Returns `true` when the member is actually gone, so message delivery must stop.
    #[must_use]
    pub const fn is_unreachable(self) -> bool {
        member_status_in(&MEMBER_UNREACHABLE_STATUSES, self)
    }

    /// Returns `true` when the member finished its share of the work (team-completion verdict).
    ///
    /// Deliberately excludes `unstarted` and `error`; see [`Self::is_quiescent`].
    #[must_use]
    pub const fn is_settled(self) -> bool {
        member_status_in(&MEMBER_SETTLED_STATUSES, self)
    }

    /// Returns `true` when nothing is running on the member's behalf right now (team-idle signal).
    ///
    /// This is *not* the same question as [`Self::is_settled`]: a team with an errored or never
    /// started member is idle, but not complete.
    #[must_use]
    pub const fn is_quiescent(self) -> bool {
        member_status_in(&MEMBER_QUIESCENT_STATUSES, self)
    }

    /// Returns `true` when `self -> to` is listed in [`MEMBER_TRANSITIONS`].
    #[must_use]
    pub fn can_transition_to(self, to: Self) -> bool {
        table_allows(MEMBER_TRANSITIONS, self, to)
    }

    /// Returns `true` when `from -> to` is listed in [`MEMBER_TRANSITIONS`].
    #[must_use]
    pub fn is_valid_transition(from: Self, to: Self) -> bool {
        from.can_transition_to(to)
    }
}

/// Detail status of one member task execution.
///
/// Wire values are the Python strings; `TimedOut` serializes as `timed_out`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionStatus {
    /// Not executing any task.
    Idle,
    /// Task execution is starting.
    Starting,
    /// Task is actively running.
    Running,
    /// Cancellation has been requested.
    CancelRequested,
    /// Task is being cancelled.
    Cancelling,
    /// Task was cancelled.
    Cancelled,
    /// Task is completing.
    Completing,
    /// Task completed successfully.
    Completed,
    /// Task failed.
    Failed,
    /// Task timed out.
    TimedOut,
}

impl Default for ExecutionStatus {
    /// Python default: `idle`.
    fn default() -> Self {
        Self::Idle
    }
}

impl ExecutionStatus {
    /// Returns the Python wire string of this status.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Starting => "starting",
            Self::Running => "running",
            Self::CancelRequested => "cancel_requested",
            Self::Cancelling => "cancelling",
            Self::Cancelled => "cancelled",
            Self::Completing => "completing",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::TimedOut => "timed_out",
        }
    }

    /// Returns `true` when `self -> to` is listed in [`EXECUTION_TRANSITIONS`].
    #[must_use]
    pub fn can_transition_to(self, to: Self) -> bool {
        table_allows(EXECUTION_TRANSITIONS, self, to)
    }

    /// Returns `true` when `from -> to` is listed in [`EXECUTION_TRANSITIONS`].
    #[must_use]
    pub fn is_valid_transition(from: Self, to: Self) -> bool {
        from.can_transition_to(to)
    }
}

/// Status of a task on the team task board.
///
/// Wire values are the Python strings; `InProgress` serializes as `in_progress`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
#[serde(rename_all = "snake_case")]
pub enum TaskStatus {
    /// Not started; the only claimable status.
    Pending,
    /// Waiting on a dependency edge.
    Blocked,
    /// Plan mode: reserved for planning, a plan is being drafted or revised.
    Planning,
    /// Owned and executing.
    InProgress,
    /// Submitted for verification by a reviewer.
    InReview,
    /// Terminal: verified or accepted.
    Completed,
    /// Terminal.
    Cancelled,
}

impl Default for TaskStatus {
    /// Python default: `pending`.
    fn default() -> Self {
        Self::Pending
    }
}

impl TaskStatus {
    /// Returns the Python wire string of this status.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Blocked => "blocked",
            Self::Planning => "planning",
            Self::InProgress => "in_progress",
            Self::InReview => "in_review",
            Self::Completed => "completed",
            Self::Cancelled => "cancelled",
        }
    }

    /// Returns `true` for `completed` / `cancelled`.
    #[must_use]
    pub const fn is_terminal(self) -> bool {
        task_status_in(&TASK_TERMINAL_STATUSES, self)
    }

    /// Returns `true` when a member may claim the task — only `pending`.
    #[must_use]
    pub const fn is_claimable(self) -> bool {
        matches!(self, Self::Pending)
    }

    /// Returns `true` when the task's owner can be swapped in place without touching the status.
    #[must_use]
    pub const fn is_reassignable(self) -> bool {
        task_status_in(&TASK_REASSIGNABLE_STATUSES, self)
    }

    /// Returns `true` when `self -> to` is listed in [`TASK_TRANSITIONS`].
    #[must_use]
    pub fn can_transition_to(self, to: Self) -> bool {
        table_allows(TASK_TRANSITIONS, self, to)
    }

    /// Returns `true` when `from -> to` is listed in [`TASK_TRANSITIONS`].
    #[must_use]
    pub fn is_valid_transition(from: Self, to: Self) -> bool {
        from.can_transition_to(to)
    }
}

/// Lifetime policy a member retires under, from `options.lifetime`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
#[serde(rename_all = "snake_case")]
pub enum MemberLifetime {
    /// Retire to `shut_down` (the default for members without a policy).
    Shutdown,
    /// Retire to `dormant` so the roster row survives and a direct message can wake it.
    Dormant,
}

impl Default for MemberLifetime {
    /// Python default: `shutdown`.
    fn default() -> Self {
        Self::Shutdown
    }
}

/// Outcome of one member round, as reported to observers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
#[serde(rename_all = "snake_case")]
pub enum RoundOutcome {
    /// The round started; execution moves `idle -> starting -> running`.
    Started,
    /// The round finished normally.
    Finished,
    /// The round was aborted (cancellation requested, then honoured).
    Aborted,
    /// The round failed.
    Failed,
}

impl Default for RoundOutcome {
    /// Python default: `started`.
    fn default() -> Self {
        Self::Started
    }
}

/// State transition table for [`MemberStatus`]; the only place the member rules live.
///
/// Python `MEMBER_TRANSITIONS`, row and target order preserved.
pub static MEMBER_TRANSITIONS: &[(MemberStatus, &[MemberStatus])] = &[
    (
        MemberStatus::Unstarted,
        &[
            MemberStatus::Starting,
            MemberStatus::Ready,
            MemberStatus::ShutDown,
            MemberStatus::Error,
        ],
    ),
    (
        MemberStatus::Starting,
        &[
            MemberStatus::Ready,
            MemberStatus::Unstarted,
            MemberStatus::ShutDown,
            MemberStatus::Error,
        ],
    ),
    (
        MemberStatus::Ready,
        &[
            MemberStatus::Ready,
            MemberStatus::Busy,
            MemberStatus::Paused,
            MemberStatus::Stopped,
            MemberStatus::ShutdownRequested,
            MemberStatus::ShutDown,
            MemberStatus::Error,
        ],
    ),
    (
        MemberStatus::Busy,
        &[
            MemberStatus::Ready,
            MemberStatus::Paused,
            MemberStatus::Stopped,
            MemberStatus::ShutdownRequested,
            MemberStatus::Error,
        ],
    ),
    (
        MemberStatus::Paused,
        &[
            MemberStatus::Ready,
            MemberStatus::Restarting,
            MemberStatus::Stopped,
            MemberStatus::ShutdownRequested,
            MemberStatus::ShutDown,
            MemberStatus::Error,
        ],
    ),
    (
        MemberStatus::Stopped,
        &[
            MemberStatus::Ready,
            MemberStatus::Restarting,
            MemberStatus::ShutdownRequested,
            MemberStatus::ShutDown,
            MemberStatus::Error,
        ],
    ),
    (
        MemberStatus::Restarting,
        &[
            MemberStatus::Ready,
            MemberStatus::Stopped,
            MemberStatus::Error,
            MemberStatus::ShutDown,
        ],
    ),
    (
        MemberStatus::ShutdownRequested,
        &[
            MemberStatus::ShutDown,
            MemberStatus::Dormant,
            MemberStatus::Error,
        ],
    ),
    (MemberStatus::ShutDown, &[MemberStatus::Restarting]),
    (
        MemberStatus::Dormant,
        &[MemberStatus::Restarting, MemberStatus::ShutDown],
    ),
    (
        MemberStatus::Error,
        &[
            MemberStatus::Restarting,
            MemberStatus::Ready,
            MemberStatus::Stopped,
            MemberStatus::ShutdownRequested,
            MemberStatus::ShutDown,
        ],
    ),
];

/// State transition table for [`ExecutionStatus`]; the only place the execution rules live.
///
/// Python `EXECUTION_TRANSITIONS`, row and target order preserved.
pub static EXECUTION_TRANSITIONS: &[(ExecutionStatus, &[ExecutionStatus])] = &[
    (ExecutionStatus::Idle, &[ExecutionStatus::Starting]),
    (
        ExecutionStatus::Starting,
        &[
            ExecutionStatus::Running,
            ExecutionStatus::CancelRequested,
            ExecutionStatus::Cancelling,
            ExecutionStatus::Failed,
            ExecutionStatus::TimedOut,
        ],
    ),
    (
        ExecutionStatus::Running,
        &[
            ExecutionStatus::CancelRequested,
            ExecutionStatus::Cancelling,
            ExecutionStatus::Completing,
            ExecutionStatus::Failed,
            ExecutionStatus::TimedOut,
        ],
    ),
    (
        ExecutionStatus::CancelRequested,
        &[
            ExecutionStatus::Cancelling,
            ExecutionStatus::Cancelled,
            ExecutionStatus::Failed,
            ExecutionStatus::TimedOut,
        ],
    ),
    (
        ExecutionStatus::Cancelling,
        &[
            ExecutionStatus::Cancelled,
            ExecutionStatus::Failed,
            ExecutionStatus::TimedOut,
        ],
    ),
    (ExecutionStatus::Cancelled, &[ExecutionStatus::Idle]),
    (
        ExecutionStatus::Completing,
        &[
            ExecutionStatus::Completed,
            ExecutionStatus::Failed,
            ExecutionStatus::TimedOut,
        ],
    ),
    (ExecutionStatus::Completed, &[ExecutionStatus::Idle]),
    (ExecutionStatus::Failed, &[ExecutionStatus::Idle]),
    (ExecutionStatus::TimedOut, &[ExecutionStatus::Idle]),
];

/// State transition table for [`TaskStatus`]; the only place the task rules live.
///
/// Deliberately narrower than Python's superset (see `src/schema/CONTEXT.md`): the task board
/// codes against this table, so `planning` / `in_progress` / `in_review` have no back-edge to
/// `pending`, `blocked` or `planning`.
pub static TASK_TRANSITIONS: &[(TaskStatus, &[TaskStatus])] = &[
    (
        TaskStatus::Pending,
        &[
            TaskStatus::Blocked,
            TaskStatus::Planning,
            TaskStatus::InProgress,
            TaskStatus::Cancelled,
        ],
    ),
    (
        TaskStatus::Blocked,
        &[TaskStatus::Pending, TaskStatus::Cancelled],
    ),
    (
        TaskStatus::Planning,
        &[TaskStatus::InProgress, TaskStatus::Cancelled],
    ),
    (
        TaskStatus::InProgress,
        &[
            TaskStatus::InReview,
            TaskStatus::Completed,
            TaskStatus::Cancelled,
            TaskStatus::Planning,
        ],
    ),
    (
        TaskStatus::InReview,
        &[
            TaskStatus::Completed,
            TaskStatus::InProgress,
            TaskStatus::Cancelled,
        ],
    ),
    (TaskStatus::Completed, &[]),
    (TaskStatus::Cancelled, &[]),
];

/// Statuses in which the leader has released the member: shutdown requested, or already gone.
pub const MEMBER_DEPARTED_STATUSES: [MemberStatus; 2] =
    [MemberStatus::ShutdownRequested, MemberStatus::ShutDown];

/// Statuses in which the member is actually gone, so message delivery stops.
pub const MEMBER_UNREACHABLE_STATUSES: [MemberStatus; 1] = [MemberStatus::ShutDown];

/// Statuses a member can rest in once it has finished its share of the work.
pub const MEMBER_SETTLED_STATUSES: [MemberStatus; 5] = [
    MemberStatus::Ready,
    MemberStatus::Paused,
    MemberStatus::Stopped,
    MemberStatus::ShutDown,
    MemberStatus::Dormant,
];

/// Statuses in which nothing is running on the member's behalf right now.
///
/// Settled plus `unstarted` and `error`; do not merge the two sets.
pub const MEMBER_QUIESCENT_STATUSES: [MemberStatus; 7] = [
    MemberStatus::Unstarted,
    MemberStatus::Ready,
    MemberStatus::Paused,
    MemberStatus::Stopped,
    MemberStatus::ShutDown,
    MemberStatus::Dormant,
    MemberStatus::Error,
];

/// Task statuses whose owner can be swapped in place, leaving the status untouched.
pub const TASK_REASSIGNABLE_STATUSES: [TaskStatus; 3] = [
    TaskStatus::Pending,
    TaskStatus::Blocked,
    TaskStatus::InProgress,
];

/// Terminal task statuses: `completed` and `cancelled`.
pub const TASK_TERMINAL_STATUSES: [TaskStatus; 2] = [TaskStatus::Completed, TaskStatus::Cancelled];

/// Returns `true` when `from -> to` is a valid [`MemberStatus`] transition.
#[must_use]
pub fn is_valid_member_transition(from: MemberStatus, to: MemberStatus) -> bool {
    from.can_transition_to(to)
}

/// Returns the status a retire request settles in.
///
/// `shutdown_member` writes `shutdown_requested`; the member picks the terminal state from the
/// lifetime policy persisted on its own row. Members without a `dormant` policy — and every
/// `force` retire — settle to `ShutDown`.
#[must_use]
pub const fn retire_target(lifetime: Option<MemberLifetime>, force: bool) -> MemberStatus {
    if matches!(lifetime, Some(MemberLifetime::Dormant)) && !force {
        MemberStatus::Dormant
    } else {
        MemberStatus::ShutDown
    }
}

/// Returns the execution statuses a member walks through for one round outcome.
#[must_use]
pub fn round_transitions(outcome: RoundOutcome) -> Vec<ExecutionStatus> {
    match outcome {
        RoundOutcome::Started => vec![ExecutionStatus::Starting, ExecutionStatus::Running],
        RoundOutcome::Finished => vec![
            ExecutionStatus::Completing,
            ExecutionStatus::Completed,
            ExecutionStatus::Idle,
        ],
        RoundOutcome::Aborted => vec![
            ExecutionStatus::CancelRequested,
            ExecutionStatus::Cancelling,
            ExecutionStatus::Cancelled,
            ExecutionStatus::Idle,
        ],
        RoundOutcome::Failed => vec![ExecutionStatus::Failed, ExecutionStatus::Idle],
    }
}

/// Looks `from -> to` up in a transition table.
fn table_allows<T: Copy + PartialEq>(table: &[(T, &[T])], from: T, to: T) -> bool {
    table
        .iter()
        .find(|(state, _)| *state == from)
        .is_some_and(|(_, targets)| targets.contains(&to))
}

/// Returns `true` when `status` is one of the [`MemberStatus`] values listed in `set`.
const fn member_status_in(set: &[MemberStatus], status: MemberStatus) -> bool {
    let mut index = 0;
    while index < set.len() {
        if set[index] as u8 == status as u8 {
            return true;
        }
        index += 1;
    }
    false
}

/// Returns `true` when `status` is one of the [`TaskStatus`] values listed in `set`.
const fn task_status_in(set: &[TaskStatus], status: TaskStatus) -> bool {
    let mut index = 0;
    while index < set.len() {
        if set[index] as u8 == status as u8 {
            return true;
        }
        index += 1;
    }
    false
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    const ALL_MEMBER_STATUSES: [MemberStatus; 11] = [
        MemberStatus::Unstarted,
        MemberStatus::Starting,
        MemberStatus::Ready,
        MemberStatus::Busy,
        MemberStatus::Paused,
        MemberStatus::Stopped,
        MemberStatus::Restarting,
        MemberStatus::ShutdownRequested,
        MemberStatus::ShutDown,
        MemberStatus::Dormant,
        MemberStatus::Error,
    ];

    const ALL_EXECUTION_STATUSES: [ExecutionStatus; 10] = [
        ExecutionStatus::Idle,
        ExecutionStatus::Starting,
        ExecutionStatus::Running,
        ExecutionStatus::CancelRequested,
        ExecutionStatus::Cancelling,
        ExecutionStatus::Cancelled,
        ExecutionStatus::Completing,
        ExecutionStatus::Completed,
        ExecutionStatus::Failed,
        ExecutionStatus::TimedOut,
    ];

    const ALL_TASK_STATUSES: [TaskStatus; 7] = [
        TaskStatus::Pending,
        TaskStatus::Blocked,
        TaskStatus::Planning,
        TaskStatus::InProgress,
        TaskStatus::InReview,
        TaskStatus::Completed,
        TaskStatus::Cancelled,
    ];

    fn json(status: impl Serialize) -> String {
        serde_json::to_string(&status).unwrap()
    }

    fn from_json<T: serde::de::DeserializeOwned>(wire: &str) -> T {
        serde_json::from_str(&format!("\"{wire}\"")).unwrap()
    }

    #[test]
    fn member_status_wire_strings() {
        let cases = [
            (MemberStatus::Unstarted, "unstarted"),
            (MemberStatus::Starting, "starting"),
            (MemberStatus::Ready, "ready"),
            (MemberStatus::Busy, "busy"),
            (MemberStatus::Paused, "paused"),
            (MemberStatus::Stopped, "stopped"),
            (MemberStatus::Restarting, "restarting"),
            (MemberStatus::ShutdownRequested, "shutdown_requested"),
            (MemberStatus::ShutDown, "shut_down"),
            (MemberStatus::Dormant, "dormant"),
            (MemberStatus::Error, "error"),
        ];
        for (status, wire) in cases {
            assert_eq!(json(status), format!("\"{wire}\""));
            assert_eq!(from_json::<MemberStatus>(wire), status);
            assert_eq!(status.as_str(), wire);
        }
        assert_eq!(
            serde_json::from_str::<MemberStatus>("\"shut_down\"").unwrap(),
            MemberStatus::ShutDown
        );
    }

    #[test]
    fn execution_status_wire_strings() {
        let cases = [
            (ExecutionStatus::Idle, "idle"),
            (ExecutionStatus::Starting, "starting"),
            (ExecutionStatus::Running, "running"),
            (ExecutionStatus::CancelRequested, "cancel_requested"),
            (ExecutionStatus::Cancelling, "cancelling"),
            (ExecutionStatus::Cancelled, "cancelled"),
            (ExecutionStatus::Completing, "completing"),
            (ExecutionStatus::Completed, "completed"),
            (ExecutionStatus::Failed, "failed"),
            (ExecutionStatus::TimedOut, "timed_out"),
        ];
        for (status, wire) in cases {
            assert_eq!(json(status), format!("\"{wire}\""));
            assert_eq!(from_json::<ExecutionStatus>(wire), status);
            assert_eq!(status.as_str(), wire);
        }
    }

    #[test]
    fn task_status_wire_strings() {
        let cases = [
            (TaskStatus::Pending, "pending"),
            (TaskStatus::Blocked, "blocked"),
            (TaskStatus::Planning, "planning"),
            (TaskStatus::InProgress, "in_progress"),
            (TaskStatus::InReview, "in_review"),
            (TaskStatus::Completed, "completed"),
            (TaskStatus::Cancelled, "cancelled"),
        ];
        for (status, wire) in cases {
            assert_eq!(json(status), format!("\"{wire}\""));
            assert_eq!(from_json::<TaskStatus>(wire), status);
            assert_eq!(status.as_str(), wire);
        }
    }

    #[test]
    fn lifetime_and_round_outcome_wire_strings() {
        let lifetimes = [
            (MemberLifetime::Shutdown, "shutdown"),
            (MemberLifetime::Dormant, "dormant"),
        ];
        for (value, wire) in lifetimes {
            assert_eq!(json(value), format!("\"{wire}\""));
            assert_eq!(from_json::<MemberLifetime>(wire), value);
        }

        let outcomes = [
            (RoundOutcome::Started, "started"),
            (RoundOutcome::Finished, "finished"),
            (RoundOutcome::Aborted, "aborted"),
            (RoundOutcome::Failed, "failed"),
        ];
        for (value, wire) in outcomes {
            assert_eq!(json(value), format!("\"{wire}\""));
            assert_eq!(from_json::<RoundOutcome>(wire), value);
        }
    }

    #[test]
    fn defaults_match_python() {
        assert_eq!(MemberStatus::default(), MemberStatus::Unstarted);
        assert_eq!(ExecutionStatus::default(), ExecutionStatus::Idle);
        assert_eq!(TaskStatus::default(), TaskStatus::Pending);
        assert_eq!(MemberLifetime::default(), MemberLifetime::Shutdown);
        assert_eq!(RoundOutcome::default(), RoundOutcome::Started);
    }

    #[test]
    fn member_transition_table_is_pinned() {
        use MemberStatus::{
            Busy, Dormant, Error, Paused, Ready, Restarting, ShutDown, ShutdownRequested, Starting,
            Stopped, Unstarted,
        };
        let expected: &[(MemberStatus, &[MemberStatus])] = &[
            (Unstarted, &[Starting, Ready, ShutDown, Error]),
            (Starting, &[Ready, Unstarted, ShutDown, Error]),
            (
                Ready,
                &[
                    Ready,
                    Busy,
                    Paused,
                    Stopped,
                    ShutdownRequested,
                    ShutDown,
                    Error,
                ],
            ),
            (Busy, &[Ready, Paused, Stopped, ShutdownRequested, Error]),
            (
                Paused,
                &[
                    Ready,
                    Restarting,
                    Stopped,
                    ShutdownRequested,
                    ShutDown,
                    Error,
                ],
            ),
            (
                Stopped,
                &[Ready, Restarting, ShutdownRequested, ShutDown, Error],
            ),
            (Restarting, &[Ready, Stopped, Error, ShutDown]),
            (ShutdownRequested, &[ShutDown, Dormant, Error]),
            (ShutDown, &[Restarting]),
            (Dormant, &[Restarting, ShutDown]),
            (
                Error,
                &[Restarting, Ready, Stopped, ShutdownRequested, ShutDown],
            ),
        ];
        assert_eq!(MEMBER_TRANSITIONS.len(), expected.len());
        for (row, want) in MEMBER_TRANSITIONS.iter().zip(expected) {
            assert_eq!(row.0, want.0);
            assert_eq!(row.1, want.1, "targets for {:?}", row.0);
        }
    }

    #[test]
    fn execution_transition_table_is_pinned() {
        use ExecutionStatus::{
            CancelRequested, Cancelled, Cancelling, Completed, Completing, Failed, Idle, Running,
            Starting, TimedOut,
        };
        let expected: &[(ExecutionStatus, &[ExecutionStatus])] = &[
            (Idle, &[Starting]),
            (
                Starting,
                &[Running, CancelRequested, Cancelling, Failed, TimedOut],
            ),
            (
                Running,
                &[CancelRequested, Cancelling, Completing, Failed, TimedOut],
            ),
            (CancelRequested, &[Cancelling, Cancelled, Failed, TimedOut]),
            (Cancelling, &[Cancelled, Failed, TimedOut]),
            (Cancelled, &[Idle]),
            (Completing, &[Completed, Failed, TimedOut]),
            (Completed, &[Idle]),
            (Failed, &[Idle]),
            (TimedOut, &[Idle]),
        ];
        assert_eq!(EXECUTION_TRANSITIONS.len(), expected.len());
        for (row, want) in EXECUTION_TRANSITIONS.iter().zip(expected) {
            assert_eq!(row.0, want.0);
            assert_eq!(row.1, want.1, "targets for {:?}", row.0);
        }
    }

    #[test]
    fn task_transition_table_is_pinned() {
        use TaskStatus::{Blocked, Cancelled, Completed, InProgress, InReview, Pending, Planning};
        let expected: &[(TaskStatus, &[TaskStatus])] = &[
            (Pending, &[Blocked, Planning, InProgress, Cancelled]),
            (Blocked, &[Pending, Cancelled]),
            (Planning, &[InProgress, Cancelled]),
            (InProgress, &[InReview, Completed, Cancelled, Planning]),
            (InReview, &[Completed, InProgress, Cancelled]),
            (Completed, &[]),
            (Cancelled, &[]),
        ];
        assert_eq!(TASK_TRANSITIONS.len(), expected.len());
        for (row, want) in TASK_TRANSITIONS.iter().zip(expected) {
            assert_eq!(row.0, want.0);
            assert_eq!(row.1, want.1, "targets for {:?}", row.0);
        }
    }

    #[test]
    fn member_transitions_agree_with_table() {
        assert_eq!(MEMBER_TRANSITIONS.len(), ALL_MEMBER_STATUSES.len());
        for (from, targets) in MEMBER_TRANSITIONS {
            for to in ALL_MEMBER_STATUSES {
                let expected = targets.contains(&to);
                assert_eq!(from.can_transition_to(to), expected, "{from:?} -> {to:?}");
                assert_eq!(
                    MemberStatus::is_valid_transition(*from, to),
                    expected,
                    "{from:?} -> {to:?}"
                );
                assert_eq!(
                    is_valid_member_transition(*from, to),
                    expected,
                    "{from:?} -> {to:?}"
                );
            }
        }
    }

    #[test]
    fn execution_transitions_agree_with_table() {
        assert_eq!(EXECUTION_TRANSITIONS.len(), ALL_EXECUTION_STATUSES.len());
        for (from, targets) in EXECUTION_TRANSITIONS {
            for to in ALL_EXECUTION_STATUSES {
                assert_eq!(
                    from.can_transition_to(to),
                    targets.contains(&to),
                    "{from:?} -> {to:?}"
                );
                assert_eq!(
                    ExecutionStatus::is_valid_transition(*from, to),
                    from.can_transition_to(to)
                );
            }
        }
    }

    #[test]
    fn task_transitions_agree_with_table() {
        assert_eq!(TASK_TRANSITIONS.len(), ALL_TASK_STATUSES.len());
        for (from, targets) in TASK_TRANSITIONS {
            for to in ALL_TASK_STATUSES {
                assert_eq!(
                    from.can_transition_to(to),
                    targets.contains(&to),
                    "{from:?} -> {to:?}"
                );
                assert_eq!(
                    TaskStatus::is_valid_transition(*from, to),
                    from.can_transition_to(to)
                );
            }
        }
    }

    #[test]
    fn member_transition_parity_facts() {
        use MemberStatus::{Dormant, Ready, Restarting, ShutDown, ShutdownRequested};
        assert!(ShutdownRequested.can_transition_to(Dormant));
        assert!(ShutdownRequested.can_transition_to(ShutDown));
        assert!(Dormant.can_transition_to(Restarting));
        assert!(Dormant.can_transition_to(ShutDown));
        assert!(ShutDown.can_transition_to(Restarting));
        assert!(Ready.can_transition_to(Ready));
        assert!(!Ready.can_transition_to(Dormant));
        assert!(!Dormant.can_transition_to(Ready));
        assert!(!ShutDown.can_transition_to(Ready));
    }

    #[test]
    fn completed_tasks_are_dead_ends() {
        for to in ALL_TASK_STATUSES {
            assert!(!TaskStatus::Completed.can_transition_to(to), "-> {to:?}");
            assert!(!TaskStatus::Cancelled.can_transition_to(to), "-> {to:?}");
        }
        assert!(TaskStatus::Completed.is_terminal());
        assert!(TaskStatus::Cancelled.is_terminal());
    }

    #[test]
    fn retire_target_parity() {
        assert_eq!(retire_target(None, false), MemberStatus::ShutDown);
        assert_eq!(
            retire_target(Some(MemberLifetime::Dormant), false),
            MemberStatus::Dormant
        );
        assert_eq!(
            retire_target(Some(MemberLifetime::Dormant), true),
            MemberStatus::ShutDown
        );
        assert_eq!(
            retire_target(Some(MemberLifetime::Shutdown), false),
            MemberStatus::ShutDown
        );
        assert_eq!(
            retire_target(Some(MemberLifetime::Shutdown), true),
            MemberStatus::ShutDown
        );
    }

    #[test]
    fn round_transitions_parity() {
        assert_eq!(
            round_transitions(RoundOutcome::Started),
            vec![ExecutionStatus::Starting, ExecutionStatus::Running]
        );
        assert_eq!(
            round_transitions(RoundOutcome::Finished),
            vec![
                ExecutionStatus::Completing,
                ExecutionStatus::Completed,
                ExecutionStatus::Idle
            ]
        );
        assert_eq!(
            round_transitions(RoundOutcome::Aborted),
            vec![
                ExecutionStatus::CancelRequested,
                ExecutionStatus::Cancelling,
                ExecutionStatus::Cancelled,
                ExecutionStatus::Idle
            ]
        );
        assert_eq!(
            round_transitions(RoundOutcome::Failed),
            vec![ExecutionStatus::Failed, ExecutionStatus::Idle]
        );
    }

    #[test]
    fn member_status_sets_match_python() {
        let departed = [MemberStatus::ShutdownRequested, MemberStatus::ShutDown];
        let unreachable = [MemberStatus::ShutDown];
        let settled = [
            MemberStatus::Ready,
            MemberStatus::Paused,
            MemberStatus::Stopped,
            MemberStatus::ShutDown,
            MemberStatus::Dormant,
        ];
        let quiescent = [
            MemberStatus::Unstarted,
            MemberStatus::Ready,
            MemberStatus::Paused,
            MemberStatus::Stopped,
            MemberStatus::ShutDown,
            MemberStatus::Dormant,
            MemberStatus::Error,
        ];
        assert_eq!(MEMBER_DEPARTED_STATUSES, departed);
        assert_eq!(MEMBER_UNREACHABLE_STATUSES, unreachable);
        assert_eq!(MEMBER_SETTLED_STATUSES, settled);
        assert_eq!(MEMBER_QUIESCENT_STATUSES, quiescent);

        for status in ALL_MEMBER_STATUSES {
            assert_eq!(
                status.is_departed(),
                departed.contains(&status),
                "{status:?}"
            );
            assert_eq!(
                status.is_unreachable(),
                unreachable.contains(&status),
                "{status:?}"
            );
            assert_eq!(status.is_settled(), settled.contains(&status), "{status:?}");
            assert_eq!(
                status.is_quiescent(),
                quiescent.contains(&status),
                "{status:?}"
            );
        }
    }

    #[test]
    fn settled_and_quiescent_are_distinct_questions() {
        assert!(!MemberStatus::Unstarted.is_settled());
        assert!(MemberStatus::Unstarted.is_quiescent());
        assert!(!MemberStatus::Error.is_settled());
        assert!(MemberStatus::Error.is_quiescent());
        for status in [
            MemberStatus::Starting,
            MemberStatus::Busy,
            MemberStatus::Restarting,
            MemberStatus::ShutdownRequested,
        ] {
            assert!(!status.is_settled(), "{status:?}");
            assert!(!status.is_quiescent(), "{status:?}");
        }
    }

    #[test]
    fn task_status_sets_match_python() {
        assert_eq!(
            TASK_REASSIGNABLE_STATUSES,
            [
                TaskStatus::Pending,
                TaskStatus::Blocked,
                TaskStatus::InProgress
            ]
        );
        assert_eq!(
            TASK_TERMINAL_STATUSES,
            [TaskStatus::Completed, TaskStatus::Cancelled]
        );
        for status in ALL_TASK_STATUSES {
            assert_eq!(
                status.is_claimable(),
                status == TaskStatus::Pending,
                "{status:?}"
            );
            assert_eq!(
                status.is_terminal(),
                status == TaskStatus::Completed || status == TaskStatus::Cancelled
            );
            assert_eq!(
                status.is_reassignable(),
                [
                    TaskStatus::Pending,
                    TaskStatus::Blocked,
                    TaskStatus::InProgress
                ]
                .contains(&status),
                "{status:?}"
            );
        }
    }
}
