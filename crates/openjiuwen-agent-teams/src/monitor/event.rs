//! The renderable event model of the team monitor.
//!
//! Ports the status-stream half of `openjiuwen/agent_teams/monitor/` (Python v0.1.20): the record
//! a monitor appends, its timestamped envelope, and the projection of a team snapshot into the
//! same records.
//!
//! [`MonitorEvent`] is a *renderable event model*, **not** a mirror of the Python `MonitorEvent`
//! Pydantic model or of the monitor's database rows
//! (`openjiuwen/agent_teams/monitor/models.py`): it carries exactly the fields a one-line text
//! record needs, so history, queries and persistence can change without touching the renderer.
//! [`MonitorEvent::from_member_status`] therefore ignores `MemberStatusChangedEvent::previous` —
//! a status line shows the new status only — and every role/status reaches the text through its
//! `as_str()` wire name.
//!
//! Everything here is pure data plus pure conversions: no clocks (except [`StampedEvent::now`]),
//! no I/O, no global state. The text shape is owned by [`render`](crate::monitor::render).

use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::schema::{
    HarnessState, MemberSnapshot, MemberStatus, MemberStatusChangedEvent, RoundOutcome,
    TaskCreatedEvent, TaskStatus, TaskStatusChangedEvent, TaskSummary, TeamRole, TeamSnapshot,
};

/// One renderable monitor event.
///
/// The internally tagged wire shape (`{"kind": …}`) is what a stream consumer parses; the
/// renderer turns each variant into a single text line. Variants are additive — a reader must
/// tolerate an unknown `kind`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MonitorEvent {
    /// The team's lifecycle state changed.
    TeamStatus {
        /// The state the team now holds.
        state: HarnessState,
    },
    /// The team stopped for good.
    TeamStopped,
    /// A member was spawned onto the roster.
    MemberSpawned {
        /// Wire name of the member.
        member_name: String,
        /// Role the member plays.
        role: TeamRole,
    },
    /// A member's lifecycle status changed.
    MemberStatus {
        /// Wire name of the member.
        member_name: String,
        /// The status the member now holds.
        status: MemberStatus,
        /// Task the member is working on, when the publisher knew it.
        task_id: Option<String>,
    },
    /// A member finished one execution round.
    MemberRound {
        /// Wire name of the member.
        member_name: String,
        /// One-based round number.
        round: u32,
        /// How the round ended.
        outcome: RoundOutcome,
    },
    /// A member was stopped or went down.
    MemberStopped {
        /// Wire name of the member.
        member_name: String,
        /// Why the member stopped, when known.
        reason: Option<String>,
    },
    /// A task was created on the board.
    TaskCreated {
        /// Identity of the task.
        task_id: String,
        /// Short title.
        title: String,
    },
    /// A task changed status.
    TaskStatus {
        /// Identity of the task.
        task_id: String,
        /// The status the task now holds.
        status: TaskStatus,
        /// Owner after the transition, if any.
        assignee: Option<String>,
    },
}

impl MonitorEvent {
    /// Builds a [`MonitorEvent::MemberStatus`] from a published status-changed payload.
    ///
    /// `MemberStatusChangedEvent::previous` is deliberately unused: the status line shows the new
    /// status only, and a reader wanting a diff keeps the previous line.
    #[must_use]
    pub fn from_member_status(event: &MemberStatusChangedEvent, task_id: Option<String>) -> Self {
        Self::MemberStatus {
            member_name: event.member_name.clone(),
            status: event.status,
            task_id,
        }
    }

    /// Builds a [`MonitorEvent::TaskCreated`] from a published creation payload.
    ///
    /// The payload's `assignee` is not rendered by the creation line; it shows up on the later
    /// [`MonitorEvent::TaskStatus`] line that carries the same task id.
    #[must_use]
    pub fn from_task_created(event: &TaskCreatedEvent) -> Self {
        Self::TaskCreated {
            task_id: event.task_id.clone(),
            title: event.title.clone(),
        }
    }

    /// Builds a [`MonitorEvent::TaskStatus`] from a published status-changed payload.
    #[must_use]
    pub fn from_task_status(event: &TaskStatusChangedEvent) -> Self {
        Self::TaskStatus {
            task_id: event.task_id.clone(),
            status: event.status,
            assignee: event.assignee.clone(),
        }
    }

    /// Returns the wire name of this variant (`team_status`, `member_round`, …).
    #[must_use]
    pub const fn kind(&self) -> &'static str {
        match self {
            Self::TeamStatus { .. } => "team_status",
            Self::TeamStopped => "team_stopped",
            Self::MemberSpawned { .. } => "member_spawned",
            Self::MemberStatus { .. } => "member_status",
            Self::MemberRound { .. } => "member_round",
            Self::MemberStopped { .. } => "member_stopped",
            Self::TaskCreated { .. } => "task_created",
            Self::TaskStatus { .. } => "task_status",
        }
    }

    /// Returns the member name or task id this event is about, or `""` for the team variants.
    #[must_use]
    pub fn subject(&self) -> &str {
        match self {
            Self::TeamStatus { .. } | Self::TeamStopped => "",
            Self::MemberSpawned { member_name, .. }
            | Self::MemberStatus { member_name, .. }
            | Self::MemberRound { member_name, .. }
            | Self::MemberStopped { member_name, .. } => member_name,
            Self::TaskCreated { task_id, .. } | Self::TaskStatus { task_id, .. } => task_id,
        }
    }
}

/// A [`MonitorEvent`] stamped with the wall-clock time it was observed.
///
/// `timestamp` is milliseconds since the UNIX epoch, so a rendered line is reproducible from a
/// golden test and a stream can be replayed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StampedEvent {
    /// Milliseconds since the UNIX epoch.
    pub timestamp: u64,
    /// The event itself.
    pub event: MonitorEvent,
}

impl StampedEvent {
    /// Stamps `event` with an explicit `timestamp` (milliseconds since the UNIX epoch).
    #[must_use]
    pub const fn new(timestamp: u64, event: MonitorEvent) -> Self {
        Self { timestamp, event }
    }

    /// Stamps `event` with the current wall-clock time.
    ///
    /// A clock before the epoch yields `0` rather than an error; the millisecond count saturates
    /// at `u64::MAX` instead of panicking or wrapping.
    #[must_use]
    pub fn now(event: MonitorEvent) -> Self {
        let elapsed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default();
        let timestamp = u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX);
        Self::new(timestamp, event)
    }
}

/// Projects a [`TeamSnapshot`] into the monitor's event records.
///
/// One [`MonitorEvent::MemberStatus`] per roster row (with no task id — a snapshot does not know
/// what a member is working on) sorted by member name, followed by one [`MonitorEvent::TaskStatus`]
/// per task sorted by task id. Every record is stamped with `snapshot.updated_at`, so a rendered
/// snapshot is deterministic. The input is only read, never reordered or mutated.
#[must_use]
pub fn snapshot_events(snapshot: &TeamSnapshot) -> Vec<StampedEvent> {
    let timestamp = snapshot.updated_at;

    let mut members: Vec<&MemberSnapshot> = snapshot.members.iter().collect();
    members.sort_by(|left, right| left.member_name.cmp(&right.member_name));

    let mut tasks: Vec<&TaskSummary> = snapshot.tasks.iter().collect();
    tasks.sort_by(|left, right| left.task_id.cmp(&right.task_id));

    let mut events = Vec::with_capacity(members.len() + tasks.len());
    events.extend(members.into_iter().map(|member| {
        StampedEvent::new(
            timestamp,
            MonitorEvent::MemberStatus {
                member_name: member.member_name.clone(),
                status: member.status,
                task_id: None,
            },
        )
    }));
    events.extend(tasks.into_iter().map(|task| {
        StampedEvent::new(
            timestamp,
            MonitorEvent::TaskStatus {
                task_id: task.task_id.clone(),
                status: task.status,
                assignee: task.assignee.clone(),
            },
        )
    }));
    events
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use serde_json::{json, Value};

    use super::*;

    /// Every variant paired with its pinned wire `kind`.
    fn all_variants() -> Vec<(MonitorEvent, &'static str, &'static str)> {
        vec![
            (
                MonitorEvent::TeamStatus {
                    state: HarnessState::Running,
                },
                "team_status",
                "",
            ),
            (MonitorEvent::TeamStopped, "team_stopped", ""),
            (
                MonitorEvent::MemberSpawned {
                    member_name: "dev-1".to_owned(),
                    role: TeamRole::Teammate,
                },
                "member_spawned",
                "dev-1",
            ),
            (
                MonitorEvent::MemberStatus {
                    member_name: "dev-1".to_owned(),
                    status: MemberStatus::Busy,
                    task_id: Some("t-3".to_owned()),
                },
                "member_status",
                "dev-1",
            ),
            (
                MonitorEvent::MemberRound {
                    member_name: "dev-1".to_owned(),
                    round: 2,
                    outcome: RoundOutcome::Finished,
                },
                "member_round",
                "dev-1",
            ),
            (
                MonitorEvent::MemberStopped {
                    member_name: "dev-1".to_owned(),
                    reason: Some("done".to_owned()),
                },
                "member_stopped",
                "dev-1",
            ),
            (
                MonitorEvent::TaskCreated {
                    task_id: "t-3".to_owned(),
                    title: "write docs".to_owned(),
                },
                "task_created",
                "t-3",
            ),
            (
                MonitorEvent::TaskStatus {
                    task_id: "t-3".to_owned(),
                    status: TaskStatus::InReview,
                    assignee: Some("dev-1".to_owned()),
                },
                "task_status",
                "t-3",
            ),
        ]
    }

    fn fixture_snapshot() -> TeamSnapshot {
        TeamSnapshot {
            team_name: "alpha".to_owned(),
            state: HarnessState::Running,
            members: vec![
                MemberSnapshot {
                    member_name: "team_leader".to_owned(),
                    role: TeamRole::Leader,
                    status: MemberStatus::Ready,
                    ..MemberSnapshot::default()
                },
                MemberSnapshot {
                    member_name: "dev-1".to_owned(),
                    role: TeamRole::Teammate,
                    status: MemberStatus::Busy,
                    ..MemberSnapshot::default()
                },
            ],
            tasks: vec![
                TaskSummary {
                    task_id: "t-2".to_owned(),
                    title: "write docs".to_owned(),
                    status: TaskStatus::InReview,
                    assignee: Some("dev-1".to_owned()),
                    ..TaskSummary::default()
                },
                TaskSummary {
                    task_id: "t-1".to_owned(),
                    title: "scaffold".to_owned(),
                    status: TaskStatus::Completed,
                    assignee: None,
                    ..TaskSummary::default()
                },
            ],
            updated_at: 43_200_000,
        }
    }

    #[test]
    fn kind_pins_every_variant() {
        for (event, kind, _) in all_variants() {
            assert_eq!(event.kind(), kind, "{event:?}");
        }
    }

    #[test]
    fn subject_pins_every_variant() {
        for (event, _, subject) in all_variants() {
            assert_eq!(event.subject(), subject, "{event:?}");
        }
    }

    #[test]
    fn wire_shape_pins_every_variant() {
        let expected = vec![
            json!({"kind": "team_status", "state": "running"}),
            json!({"kind": "team_stopped"}),
            json!({"kind": "member_spawned", "member_name": "dev-1", "role": "teammate"}),
            json!({"kind": "member_status", "member_name": "dev-1", "status": "busy", "task_id": "t-3"}),
            json!({"kind": "member_round", "member_name": "dev-1", "round": 2, "outcome": "finished"}),
            json!({"kind": "member_stopped", "member_name": "dev-1", "reason": "done"}),
            json!({"kind": "task_created", "task_id": "t-3", "title": "write docs"}),
            json!({"kind": "task_status", "task_id": "t-3", "status": "in_review", "assignee": "dev-1"}),
        ];
        let actual: Vec<Value> = all_variants()
            .into_iter()
            .map(|(event, _, _)| serde_json::to_value(&event).unwrap())
            .collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn json_round_trip_preserves_every_variant() {
        for (event, _, _) in all_variants() {
            let text = serde_json::to_string(&event).unwrap();
            let parsed: MonitorEvent = serde_json::from_str(&text).unwrap();
            assert_eq!(parsed, event, "{text}");
        }
    }

    #[test]
    fn deserialization_tolerates_unknown_fields() {
        let parsed: MonitorEvent =
            serde_json::from_value(json!({"kind": "team_stopped", "future": 1})).unwrap();
        assert_eq!(parsed, MonitorEvent::TeamStopped);
        let parsed: MonitorEvent = serde_json::from_value(json!({
            "kind": "task_status",
            "task_id": "t-3",
            "status": "completed",
            "future": {"nested": true},
        }))
        .unwrap();
        assert_eq!(
            parsed,
            MonitorEvent::TaskStatus {
                task_id: "t-3".to_owned(),
                status: TaskStatus::Completed,
                assignee: None,
            }
        );
    }

    #[test]
    fn stamped_event_new_keeps_timestamp_and_event() {
        let event = MonitorEvent::TeamStopped;
        let stamped = StampedEvent::new(42, event.clone());
        assert_eq!(stamped.timestamp, 42);
        assert_eq!(stamped.event, event);
    }

    #[test]
    fn stamped_event_now_uses_wall_clock() {
        let event = MonitorEvent::TeamStatus {
            state: HarnessState::Idle,
        };
        let stamped = StampedEvent::now(event.clone());
        assert!(stamped.timestamp > 0, "{stamped:?}");
        assert_eq!(stamped.event, event);
    }

    #[test]
    fn stamped_event_round_trips_through_json() {
        let stamped = StampedEvent::new(43_200_250, MonitorEvent::TeamStopped);
        let value = serde_json::to_value(&stamped).unwrap();
        assert_eq!(
            value,
            json!({"timestamp": 43_200_250, "event": {"kind": "team_stopped"}})
        );
        let parsed: StampedEvent = serde_json::from_value(value).unwrap();
        assert_eq!(parsed, stamped);
    }

    #[test]
    fn member_status_conversion_maps_fields_and_ignores_previous() {
        let event = MemberStatusChangedEvent {
            team_name: "alpha".to_owned(),
            member_name: "dev-1".to_owned(),
            status: MemberStatus::Busy,
            previous: Some(MemberStatus::Ready),
        };
        assert_eq!(
            MonitorEvent::from_member_status(&event, Some("t-3".to_owned())),
            MonitorEvent::MemberStatus {
                member_name: "dev-1".to_owned(),
                status: MemberStatus::Busy,
                task_id: Some("t-3".to_owned()),
            }
        );
        assert_eq!(
            MonitorEvent::from_member_status(&event, None),
            MonitorEvent::MemberStatus {
                member_name: "dev-1".to_owned(),
                status: MemberStatus::Busy,
                task_id: None,
            }
        );
    }

    #[test]
    fn task_created_conversion_drops_assignee() {
        let event = TaskCreatedEvent {
            team_name: "alpha".to_owned(),
            task_id: "t-7".to_owned(),
            title: "write docs".to_owned(),
            assignee: Some("dev-1".to_owned()),
        };
        let converted = MonitorEvent::from_task_created(&event);
        assert_eq!(
            converted,
            MonitorEvent::TaskCreated {
                task_id: "t-7".to_owned(),
                title: "write docs".to_owned(),
            }
        );
        let value = serde_json::to_value(&converted).unwrap();
        assert_eq!(
            value,
            json!({"kind": "task_created", "task_id": "t-7", "title": "write docs"})
        );
    }

    #[test]
    fn task_status_conversion_keeps_assignee() {
        let event = TaskStatusChangedEvent {
            team_name: "alpha".to_owned(),
            task_id: "t-7".to_owned(),
            status: TaskStatus::InReview,
            assignee: Some("dev-1".to_owned()),
        };
        assert_eq!(
            MonitorEvent::from_task_status(&event),
            MonitorEvent::TaskStatus {
                task_id: "t-7".to_owned(),
                status: TaskStatus::InReview,
                assignee: Some("dev-1".to_owned()),
            }
        );
        let unowned = TaskStatusChangedEvent {
            assignee: None,
            ..event
        };
        assert_eq!(
            MonitorEvent::from_task_status(&unowned),
            MonitorEvent::TaskStatus {
                task_id: "t-7".to_owned(),
                status: TaskStatus::InReview,
                assignee: None,
            }
        );
    }

    #[test]
    fn snapshot_events_sorts_members_before_tasks() {
        let snapshot = fixture_snapshot();
        let before = snapshot.clone();
        let events = snapshot_events(&snapshot);

        let subjects: Vec<&str> = events.iter().map(|e| e.event.subject()).collect();
        assert_eq!(subjects, vec!["dev-1", "team_leader", "t-1", "t-2"]);
        let kinds: Vec<&str> = events.iter().map(|e| e.event.kind()).collect();
        assert_eq!(
            kinds,
            vec![
                "member_status",
                "member_status",
                "task_status",
                "task_status"
            ]
        );
        assert!(events.iter().all(|e| e.timestamp == 43_200_000));
        assert_eq!(
            events[0].event,
            MonitorEvent::MemberStatus {
                member_name: "dev-1".to_owned(),
                status: MemberStatus::Busy,
                task_id: None,
            }
        );
        assert_eq!(
            events[3].event,
            MonitorEvent::TaskStatus {
                task_id: "t-2".to_owned(),
                status: TaskStatus::InReview,
                assignee: Some("dev-1".to_owned()),
            }
        );
        assert_eq!(snapshot, before, "the input snapshot must not be mutated");
        assert_eq!(
            snapshot_events(&snapshot),
            events,
            "projection is deterministic"
        );
    }

    #[test]
    fn snapshot_events_of_empty_snapshot_is_empty() {
        let snapshot = TeamSnapshot::new("alpha", HarnessState::Idle);
        let events = snapshot_events(&snapshot);
        assert!(events.is_empty(), "{events:?}");
    }
}
