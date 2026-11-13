//! Plain-text rendering of monitor events and team snapshots.
//!
//! The monitor's output is one record per line, fixed word order, no colour, no cursor control and
//! no box drawing: it is meant to be `grep`ped, `tail -f`ed and diffed by a human or a script.
//! There is deliberately no TUI here — the Python monitor's `Textual` app is deferred — and no
//! database access: Python's monitor reads and writes event rows in its own store, while this node
//! only renders events it is given.
//!
//! # Line shape
//!
//! An optional `[HH:MM:SS] ` / `[HH:MM:SS.mmm] ` prefix (wall-clock time in UTC shifted by
//! [`TeamMonitorConfig::utc_offset_seconds`], taken from [`StampedEvent::timestamp`]) followed by
//! the body, e.g. `[12:00:00] member dev-1 busy task=t-3`. Optional suffixes (`task=`, `reason=`,
//! `assignee=`) are appended only when the field is present, so absent information never renders
//! as `None`.
//!
//! # Time of day
//!
//! The maths runs in `i128`, so **any** `u64` timestamp and **any** `i32` offset renders without
//! panicking: out-of-range values simply wrap modulo 24 hours, and `rem_euclid` also handles a
//! negative offset crossing midnight backwards. There is no error path, so the renderer has no
//! fallback branch to keep consistent.
//!
//! `HarnessState` comes from `openjiuwen_harness_protocol` and exposes no `as_str`/`Display`, so
//! `state_str` carries the mapping here and a test locks it to the serde wire values.

use serde::{Deserialize, Serialize};

use crate::monitor::event::{snapshot_events, MonitorEvent, StampedEvent};
use crate::schema::{HarnessState, RoundOutcome, TeamSnapshot};

/// Milliseconds in one day; the modulus of the time-of-day computation.
const MILLIS_PER_DAY: i128 = 86_400_000;
/// Milliseconds in one hour.
const MILLIS_PER_HOUR: i128 = 3_600_000;
/// Milliseconds in one minute.
const MILLIS_PER_MINUTE: i128 = 60_000;
/// Milliseconds in one second.
const MILLIS_PER_SECOND: i128 = 1_000;

/// How much wall-clock time a rendered line carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TimestampPolicy {
    /// No time prefix at all; the line starts with the body.
    Hidden,
    /// `[HH:MM:SS] ` — whole seconds.
    Seconds,
    /// `[HH:MM:SS.mmm] ` — milliseconds.
    Millis,
}

impl Default for TimestampPolicy {
    /// Python's default text-monitor shape: whole-second timestamps.
    fn default() -> Self {
        Self::Seconds
    }
}

/// Presentation knobs of a [`TeamMonitor`].
///
/// `Default` is `timestamp = Seconds`, `utc_offset_seconds = 0` (UTC) and `include_header = true`;
/// every field is optional on the wire, so a sparse config parses to those defaults.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct TeamMonitorConfig {
    /// Whether a line carries a time prefix, and to what resolution.
    pub timestamp: TimestampPolicy,
    /// Whole-second shift from UTC applied when formatting the time of day.
    pub utc_offset_seconds: i32,
    /// Whether [`TeamMonitor::render_snapshot`] prints the team header line first.
    pub include_header: bool,
}

impl Default for TeamMonitorConfig {
    /// Seconds resolution, UTC, header on.
    fn default() -> Self {
        Self {
            timestamp: TimestampPolicy::Seconds,
            utc_offset_seconds: 0,
            include_header: true,
        }
    }
}

/// Renders monitor events and team snapshots as plain text lines.
///
/// The renderer holds no state beyond its configuration: any number of lines can be rendered from
/// any number of events, in any order, with identical results.
#[derive(Debug, Clone, Default)]
pub struct TeamMonitor {
    config: TeamMonitorConfig,
}

impl TeamMonitor {
    /// Creates a monitor with the default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a monitor with an explicit configuration.
    #[must_use]
    pub fn with_config(config: TeamMonitorConfig) -> Self {
        Self { config }
    }

    /// Returns the configuration this monitor renders with.
    #[must_use]
    pub const fn config(&self) -> &TeamMonitorConfig {
        &self.config
    }

    /// Renders one event as a single line, without a trailing newline.
    #[must_use]
    pub fn render_event(&self, event: &StampedEvent) -> String {
        let mut line = self.prefix(event.timestamp);
        line.push_str(&Self::body(&event.event));
        line
    }

    /// Renders a snapshot as lines joined with `'\n'`, without a trailing newline.
    ///
    /// The roster comes first, then the task board — both sorted by identity by
    /// [`snapshot_events`] — and every line is stamped with
    /// [`TeamSnapshot::updated_at`]. With [`TeamMonitorConfig::include_header`] the first line is
    /// `team {name} {state} members={n} tasks={m}`; a snapshot with no rows and the header off
    /// renders as the empty string. The snapshot is only read, never mutated.
    #[must_use]
    pub fn render_snapshot(&self, snapshot: &TeamSnapshot) -> String {
        let mut lines: Vec<String> = Vec::new();
        if self.config.include_header {
            let team_name = &snapshot.team_name;
            let state = state_str(snapshot.state);
            let members = snapshot.members.len();
            let tasks = snapshot.tasks.len();
            lines.push(format!(
                "team {team_name} {state} members={members} tasks={tasks}"
            ));
        }
        lines.extend(
            snapshot_events(snapshot)
                .iter()
                .map(|event| self.render_event(event)),
        );
        lines.join("\n")
    }

    /// Returns the time prefix for `timestamp`, including its trailing space (empty when hidden).
    fn prefix(&self, timestamp: u64) -> String {
        if self.config.timestamp == TimestampPolicy::Hidden {
            return String::new();
        }
        let (hours, minutes, seconds, millis) =
            time_of_day(timestamp, self.config.utc_offset_seconds);
        if self.config.timestamp == TimestampPolicy::Millis {
            format!("[{hours:02}:{minutes:02}:{seconds:02}.{millis:03}] ")
        } else {
            format!("[{hours:02}:{minutes:02}:{seconds:02}] ")
        }
    }

    /// Renders the body of one event, without the time prefix.
    fn body(event: &MonitorEvent) -> String {
        match event {
            MonitorEvent::TeamStatus { state } => format!("team {}", state_str(*state)),
            MonitorEvent::TeamStopped => "team stopped".to_owned(),
            MonitorEvent::MemberSpawned { member_name, role } => {
                format!("member {member_name} spawned role={}", role.as_str())
            }
            MonitorEvent::MemberStatus {
                member_name,
                status,
                task_id,
            } => match task_id {
                Some(task_id) => format!("member {member_name} {} task={task_id}", status.as_str()),
                None => format!("member {member_name} {}", status.as_str()),
            },
            MonitorEvent::MemberRound {
                member_name,
                round,
                outcome,
            } => format!(
                "round {member_name} round={round} outcome={}",
                round_outcome_str(*outcome)
            ),
            MonitorEvent::MemberStopped {
                member_name,
                reason,
            } => match reason {
                Some(reason) => format!("member {member_name} stopped reason={reason}"),
                None => format!("member {member_name} stopped"),
            },
            MonitorEvent::TaskCreated { task_id, title } => {
                format!("task {task_id} created title={title}")
            }
            MonitorEvent::TaskStatus {
                task_id,
                status,
                assignee,
            } => match assignee {
                Some(assignee) => format!("task {task_id} {} assignee={assignee}", status.as_str()),
                None => format!("task {task_id} {}", status.as_str()),
            },
        }
    }
}

/// Returns the wire name of `state`.
///
/// `openjiuwen_harness_protocol` exposes no `as_str`/`Display` for `HarnessState`, so the map
/// lives here; `state_str_matches_serde_wire_values` locks it to the serialized names.
const fn state_str(state: HarnessState) -> &'static str {
    match state {
        HarnessState::Idle => "idle",
        HarnessState::Running => "running",
        HarnessState::Pausing => "pausing",
        HarnessState::Paused => "paused",
        HarnessState::Terminated => "terminated",
    }
}

/// Returns the wire name of `outcome`.
///
/// `RoundOutcome` in `crate::schema` exposes no `as_str`, so the map lives here;
/// `round_outcome_str_matches_serde_wire_values` locks it to the serialized names.
const fn round_outcome_str(outcome: RoundOutcome) -> &'static str {
    match outcome {
        RoundOutcome::Started => "started",
        RoundOutcome::Finished => "finished",
        RoundOutcome::Aborted => "aborted",
        RoundOutcome::Failed => "failed",
    }
}

/// Splits `timestamp` (millis since the UNIX epoch, shifted by `utc_offset_seconds`) into
/// `(hours, minutes, seconds, millis)` of the wall-clock day.
///
/// The arithmetic is done in `i128` and reduced with `rem_euclid`, so no `u64`/`i32` input can
/// overflow, underflow or panic; out-of-range values wrap modulo 24 hours.
fn time_of_day(timestamp: u64, utc_offset_seconds: i32) -> (u64, u64, u64, u64) {
    let total_ms = i128::from(timestamp) + i128::from(utc_offset_seconds) * MILLIS_PER_SECOND;
    let ms_of_day = total_ms.rem_euclid(MILLIS_PER_DAY);
    (
        u64::try_from(ms_of_day / MILLIS_PER_HOUR).unwrap_or(0),
        u64::try_from((ms_of_day % MILLIS_PER_HOUR) / MILLIS_PER_MINUTE).unwrap_or(0),
        u64::try_from((ms_of_day % MILLIS_PER_MINUTE) / MILLIS_PER_SECOND).unwrap_or(0),
        u64::try_from(ms_of_day % MILLIS_PER_SECOND).unwrap_or(0),
    )
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::schema::{
        MemberSnapshot, MemberStatus, RoundOutcome, TaskStatus, TaskSummary, TeamRole,
    };

    /// 12:00:00 UTC in milliseconds.
    const NOON_MS: u64 = 43_200_000;
    /// 12:00:00.250 UTC in milliseconds.
    const NOON_250_MS: u64 = 43_200_250;

    fn stamped(timestamp: u64, event: MonitorEvent) -> StampedEvent {
        StampedEvent::new(timestamp, event)
    }

    /// A snapshot whose members and tasks are deliberately in the wrong order.
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
            updated_at: NOON_MS,
        }
    }

    #[test]
    fn renders_every_variant_with_second_prefix() {
        let monitor = TeamMonitor::new();
        let cases = vec![
            (
                MonitorEvent::TeamStatus {
                    state: HarnessState::Running,
                },
                "[12:00:00] team running",
            ),
            (MonitorEvent::TeamStopped, "[12:00:00] team stopped"),
            (
                MonitorEvent::MemberSpawned {
                    member_name: "dev-1".to_owned(),
                    role: TeamRole::Teammate,
                },
                "[12:00:00] member dev-1 spawned role=teammate",
            ),
            (
                MonitorEvent::MemberStatus {
                    member_name: "dev-1".to_owned(),
                    status: MemberStatus::Busy,
                    task_id: Some("t-3".to_owned()),
                },
                "[12:00:00] member dev-1 busy task=t-3",
            ),
            (
                MonitorEvent::MemberRound {
                    member_name: "dev-1".to_owned(),
                    round: 2,
                    outcome: RoundOutcome::Finished,
                },
                "[12:00:00] round dev-1 round=2 outcome=finished",
            ),
            (
                MonitorEvent::MemberStopped {
                    member_name: "dev-1".to_owned(),
                    reason: Some("done".to_owned()),
                },
                "[12:00:00] member dev-1 stopped reason=done",
            ),
            (
                MonitorEvent::TaskCreated {
                    task_id: "t-3".to_owned(),
                    title: "write docs".to_owned(),
                },
                "[12:00:00] task t-3 created title=write docs",
            ),
            (
                MonitorEvent::TaskStatus {
                    task_id: "t-3".to_owned(),
                    status: TaskStatus::InReview,
                    assignee: Some("dev-1".to_owned()),
                },
                "[12:00:00] task t-3 in_review assignee=dev-1",
            ),
        ];
        for (event, expected) in cases {
            assert_eq!(monitor.render_event(&stamped(NOON_MS, event)), expected);
        }
    }

    #[test]
    fn millis_policy_renders_three_digit_fraction() {
        let monitor = TeamMonitor::with_config(TeamMonitorConfig {
            timestamp: TimestampPolicy::Millis,
            ..TeamMonitorConfig::default()
        });
        let event = MonitorEvent::MemberStatus {
            member_name: "dev-1".to_owned(),
            status: MemberStatus::Busy,
            task_id: Some("t-3".to_owned()),
        };
        assert_eq!(
            monitor.render_event(&stamped(NOON_250_MS, event)),
            "[12:00:00.250] member dev-1 busy task=t-3"
        );
    }

    #[test]
    fn millis_policy_truncates_sub_second_noise() {
        let monitor = TeamMonitor::with_config(TeamMonitorConfig {
            timestamp: TimestampPolicy::Millis,
            ..TeamMonitorConfig::default()
        });
        assert_eq!(
            monitor.render_event(&stamped(1_234, MonitorEvent::TeamStopped)),
            "[00:00:01.234] team stopped"
        );
    }

    #[test]
    fn hidden_policy_omits_prefix_and_leading_space() {
        let monitor = TeamMonitor::with_config(TeamMonitorConfig {
            timestamp: TimestampPolicy::Hidden,
            ..TeamMonitorConfig::default()
        });
        let event = MonitorEvent::MemberStatus {
            member_name: "dev-1".to_owned(),
            status: MemberStatus::Busy,
            task_id: Some("t-3".to_owned()),
        };
        assert_eq!(
            monitor.render_event(&stamped(NOON_MS, event)),
            "member dev-1 busy task=t-3"
        );
    }

    #[test]
    fn optional_suffixes_are_omitted_when_absent() {
        let monitor = TeamMonitor::new();
        let cases = vec![
            (
                MonitorEvent::MemberStatus {
                    member_name: "dev-1".to_owned(),
                    status: MemberStatus::Ready,
                    task_id: None,
                },
                "[12:00:00] member dev-1 ready",
            ),
            (
                MonitorEvent::MemberStopped {
                    member_name: "dev-1".to_owned(),
                    reason: None,
                },
                "[12:00:00] member dev-1 stopped",
            ),
            (
                MonitorEvent::TaskStatus {
                    task_id: "t-3".to_owned(),
                    status: TaskStatus::Pending,
                    assignee: None,
                },
                "[12:00:00] task t-3 pending",
            ),
        ];
        for (event, expected) in cases {
            assert_eq!(monitor.render_event(&stamped(NOON_MS, event)), expected);
        }
    }

    #[test]
    fn new_matches_default_config() {
        assert_eq!(*TeamMonitor::new().config(), TeamMonitorConfig::default());
        assert_eq!(
            TeamMonitor::new().config().timestamp,
            TimestampPolicy::Seconds
        );
        assert_eq!(TeamMonitor::new().config().utc_offset_seconds, 0);
        assert!(TeamMonitor::new().config().include_header);
    }

    #[test]
    fn with_config_exposes_the_config() {
        let config = TeamMonitorConfig {
            timestamp: TimestampPolicy::Hidden,
            utc_offset_seconds: 3_600,
            include_header: false,
        };
        let monitor = TeamMonitor::with_config(config);
        assert_eq!(*monitor.config(), config);
    }

    #[test]
    fn renders_full_snapshot_sorted_with_header() {
        let monitor = TeamMonitor::new();
        let expected = [
            "team alpha running members=2 tasks=2",
            "[12:00:00] member dev-1 busy",
            "[12:00:00] member team_leader ready",
            "[12:00:00] task t-1 completed",
            "[12:00:00] task t-2 in_review assignee=dev-1",
        ]
        .join("\n");
        assert_eq!(monitor.render_snapshot(&fixture_snapshot()), expected);
    }

    #[test]
    fn snapshot_without_header_starts_with_the_roster() {
        let monitor = TeamMonitor::with_config(TeamMonitorConfig {
            include_header: false,
            ..TeamMonitorConfig::default()
        });
        let rendered = monitor.render_snapshot(&fixture_snapshot());
        assert!(!rendered.contains("members=2"), "{rendered}");
        assert_eq!(
            rendered.lines().next(),
            Some("[12:00:00] member dev-1 busy")
        );
        assert_eq!(rendered.lines().count(), 4);
    }

    #[test]
    fn empty_snapshot_renders_header_only_or_nothing() {
        let snapshot = TeamSnapshot::new("alpha", HarnessState::Idle);
        assert_eq!(
            TeamMonitor::new().render_snapshot(&snapshot),
            "team alpha idle members=0 tasks=0"
        );
        let headerless = TeamMonitor::with_config(TeamMonitorConfig {
            include_header: false,
            ..TeamMonitorConfig::default()
        });
        assert_eq!(headerless.render_snapshot(&snapshot), "");
    }

    #[test]
    fn rendering_is_deterministic_and_does_not_mutate_input() {
        let monitor = TeamMonitor::new();
        let snapshot = fixture_snapshot();
        let before = snapshot.clone();
        let first = monitor.render_snapshot(&snapshot);
        let second = monitor.render_snapshot(&snapshot);
        assert_eq!(first, second);
        assert_eq!(snapshot, before);
    }

    #[test]
    fn utc_offset_shifts_and_wraps_backwards_across_midnight() {
        // 00:30:00 UTC minus one hour is 23:30:00 on the previous day.
        let monitor = TeamMonitor::with_config(TeamMonitorConfig {
            utc_offset_seconds: -3_600,
            ..TeamMonitorConfig::default()
        });
        assert_eq!(
            monitor.render_event(&stamped(1_800_000, MonitorEvent::TeamStopped)),
            "[23:30:00] team stopped"
        );
        // 23:30:00 UTC plus one hour is 00:30:00 on the next day.
        let monitor = TeamMonitor::with_config(TeamMonitorConfig {
            utc_offset_seconds: 3_600,
            ..TeamMonitorConfig::default()
        });
        assert_eq!(
            monitor.render_event(&stamped(84_600_000, MonitorEvent::TeamStopped)),
            "[00:30:00] team stopped"
        );
    }

    #[test]
    fn extreme_timestamps_and_offsets_do_not_panic() {
        let event = MonitorEvent::TeamStopped;
        for offset in [0, 86_400, -86_400, i32::MIN, i32::MAX] {
            let monitor = TeamMonitor::with_config(TeamMonitorConfig {
                timestamp: TimestampPolicy::Millis,
                utc_offset_seconds: offset,
                include_header: true,
            });
            for timestamp in [0, u64::MAX, 1, u64::MAX - 1] {
                let line = monitor.render_event(&stamped(timestamp, event.clone()));
                assert!(
                    line.starts_with('[') && line.ends_with("team stopped"),
                    "{line}"
                );
            }
        }
    }

    #[test]
    fn state_str_matches_serde_wire_values() {
        for state in [
            HarnessState::Idle,
            HarnessState::Running,
            HarnessState::Pausing,
            HarnessState::Paused,
            HarnessState::Terminated,
        ] {
            let wire = serde_json::to_value(state).unwrap();
            assert_eq!(wire, json!(state_str(state)));
        }
    }

    #[test]
    fn round_outcome_str_matches_serde_wire_values() {
        for outcome in [
            RoundOutcome::Started,
            RoundOutcome::Finished,
            RoundOutcome::Aborted,
            RoundOutcome::Failed,
        ] {
            let wire = serde_json::to_value(outcome).unwrap();
            assert_eq!(wire, json!(round_outcome_str(outcome)));
        }
    }

    #[test]
    fn config_serde_round_trips_with_defaults_and_unknown_fields() {
        let value =
            json!({"timestamp": "millis", "utc_offset_seconds": -3_600, "include_header": false});
        let config: TeamMonitorConfig = serde_json::from_value(value.clone()).unwrap();
        assert_eq!(
            config,
            TeamMonitorConfig {
                timestamp: TimestampPolicy::Millis,
                utc_offset_seconds: -3_600,
                include_header: false,
            }
        );
        assert_eq!(serde_json::to_value(config).unwrap(), value);

        let sparse: TeamMonitorConfig = serde_json::from_value(json!({"future": 1})).unwrap();
        assert_eq!(sparse, TeamMonitorConfig::default());
        assert_eq!(sparse.timestamp, TimestampPolicy::Seconds);
        assert_eq!(sparse.utc_offset_seconds, 0);
        assert!(sparse.include_header);
        assert_eq!(TimestampPolicy::default(), TimestampPolicy::Seconds);
    }
}
