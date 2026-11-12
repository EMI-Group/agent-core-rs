//! Team-layer stream frames: the chunk envelope that attributes output to a member and the
//! framework-emitted team event markers.
//!
//! Python parity source: `openjiuwen/agent_teams/schema/stream.py`. Python subclasses
//! `OutputSchema` and adds `source_member`/`role`; the Rust port keeps the team attribution in
//! [`TeamOutputSchema`]'s own `chunk` field so the core stream layer stays free of team-specific
//! fields and a non-team producer keeps yielding plain frames.
//!
//! Team markers ([`TEAM_IDLE`], [`TEAM_COMPLETED`], [`TEAM_ERROR`]) ride the same stream as model
//! output so a streaming consumer sees them in order, but they carry no agent content. A
//! non-streaming caller that reduces a stream to "the last thing produced" uses
//! [`is_team_event_marker`] to skip them.

use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

use super::team::TeamRole;

/// The event-type value of the idle marker: the team is up but has nothing to run.
pub const TEAM_IDLE: &str = "team.idle";

/// The event-type value of the completed marker: the team finished its work.
pub const TEAM_COMPLETED: &str = "team.completed";

/// The event-type value of the error marker: the team stopped on an unrecoverable error.
pub const TEAM_ERROR: &str = "team.error";

/// The event-type value of the retrying marker: a member round failed and is being retried.
pub const RETRYING: &str = "retrying";

/// One member-attributed stream frame.
///
/// `source_member` carries the `member_name` of the team member that produced `chunk`, and `role`
/// is that member's [`TeamRole`]. Both default to empty / [`TeamRole::Leader`] for a producer that
/// has no team context, mirroring Python's `None` defaults as empty values.
///
/// JSON shape: `{"source_member": "...", "role": "teammate", "chunk": {...}}`. Use [`tag_chunk`]
/// to build one rather than assembling the object by hand.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct TeamOutputSchema {
    /// `member_name` of the member that produced `chunk`.
    pub source_member: String,
    /// Role of the producing member.
    pub role: TeamRole,
    /// The opaque chunk produced by the member (model output or a framework marker).
    pub chunk: Value,
}

/// Builds the idle-team marker frame.
///
/// The frame carries no member and no task, matching Python's `team.idle` payload.
#[must_use]
pub fn team_idle_frame() -> Value {
    json!({
        "event_type": TEAM_IDLE,
        "member_count": 0,
        "members": {},
    })
}

/// Builds the team-completed marker frame.
///
/// The frame carries no member and no task, matching Python's `team.completed` payload.
#[must_use]
pub fn team_completed_frame() -> Value {
    json!({
        "event_type": TEAM_COMPLETED,
        "member_count": 0,
        "task_count": 0,
    })
}

/// Tags `chunk` with the member that produced it, yielding a [`TeamOutputSchema`]-shaped frame.
///
/// Python: `TeamOutputSchema.from_output` — a new frame is returned and the original chunk is not
/// mutated, so the producing member keeps its object identity.
#[must_use]
pub fn tag_chunk(source_member: impl Into<String>, role: TeamRole, chunk: Value) -> Value {
    let mut frame = Map::with_capacity(3);
    frame.insert(
        "source_member".to_owned(),
        Value::String(source_member.into()),
    );
    frame.insert("role".to_owned(), Value::String(role.as_str().to_owned()));
    frame.insert("chunk".to_owned(), chunk);
    Value::Object(frame)
}

/// Returns `true` when `chunk` is a framework-emitted team marker rather than agent output.
///
/// `chunk` is a [`TeamOutputSchema`]-shaped frame (for example the value returned by
/// [`tag_chunk`]); the marker test reads the event type out of the tagged payload at
/// `chunk.chunk.payload.event_type` and accepts any string starting with `"team."`. A frame whose
/// payload is missing or not an object, or whose event type is not a `team.` string, is agent
/// output and answers `false`.
#[must_use]
pub fn is_team_event_marker(chunk: &Value) -> bool {
    chunk
        .get("chunk")
        .and_then(|tagged| tagged.get("payload"))
        .and_then(|payload| payload.get("event_type"))
        .and_then(Value::as_str)
        .is_some_and(|event_type| event_type.starts_with("team."))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    /// The idle frame's exact JSON.
    #[test]
    fn idle_frame_has_pinned_json() {
        assert_eq!(
            team_idle_frame(),
            json!({"event_type": "team.idle", "member_count": 0, "members": {}})
        );
    }

    /// The completed frame's exact JSON.
    #[test]
    fn completed_frame_has_pinned_json() {
        assert_eq!(
            team_completed_frame(),
            json!({"event_type": "team.completed", "member_count": 0, "task_count": 0})
        );
    }

    /// `tag_chunk` yields exactly the `TeamOutputSchema` shape.
    #[test]
    fn tag_chunk_has_pinned_json() {
        assert_eq!(
            tag_chunk("teammate_1", TeamRole::Teammate, json!({"type": "answer"})),
            json!({
                "source_member": "teammate_1",
                "role": "teammate",
                "chunk": {"type": "answer"},
            })
        );
    }

    /// Roles inside a tagged frame serialize as their `snake_case` wire strings.
    #[test]
    fn tag_chunk_uses_role_wire_values() {
        let cases = [
            (TeamRole::Leader, "leader"),
            (TeamRole::Teammate, "teammate"),
            (TeamRole::HumanAgent, "human_agent"),
            (TeamRole::PassiveHuman, "passive_human"),
            (TeamRole::BridgeAgent, "bridge_agent"),
            (TeamRole::Worker, "worker"),
            (TeamRole::ExternalCli, "external_cli"),
        ];
        for (role, wire) in cases {
            let frame = tag_chunk("member", role, Value::Null);
            assert_eq!(frame["role"], json!(wire), "role {role:?}");
        }
    }

    /// `TeamOutputSchema` round-trips through JSON and ignores unknown fields.
    #[test]
    fn team_output_schema_round_trips() {
        let frame = TeamOutputSchema {
            source_member: "teammate_1".to_owned(),
            role: TeamRole::Teammate,
            chunk: json!({"type": "answer", "payload": {"content": "hi"}}),
        };
        let text = serde_json::to_string(&frame).unwrap();
        assert_eq!(
            serde_json::from_str::<Value>(&text).unwrap(),
            json!({
                "source_member": "teammate_1",
                "role": "teammate",
                "chunk": {"type": "answer", "payload": {"content": "hi"}},
            })
        );
        assert_eq!(
            serde_json::from_str::<TeamOutputSchema>(&text).unwrap(),
            frame
        );

        let tolerant = serde_json::from_value::<TeamOutputSchema>(json!({
            "source_member": "leader",
            "role": "leader",
            "chunk": null,
            "unknown_field": 1,
        }))
        .unwrap();
        assert_eq!(tolerant.source_member, "leader");
        assert_eq!(tolerant.role, TeamRole::Leader);
    }

    /// A default frame is empty, leader-role and null-chunk.
    #[test]
    fn team_output_schema_defaults() {
        assert_eq!(
            serde_json::to_value(TeamOutputSchema::default()).unwrap(),
            json!({"source_member": "", "role": "leader", "chunk": null})
        );
    }

    /// Both framework markers are recognised after tagging.
    #[test]
    fn team_event_markers_are_recognised() {
        let idle = tag_chunk(
            "team_leader",
            TeamRole::Leader,
            json!({"payload": team_idle_frame()}),
        );
        let completed = tag_chunk(
            "team_leader",
            TeamRole::Leader,
            json!({"payload": team_completed_frame()}),
        );
        assert!(is_team_event_marker(&idle));
        assert!(is_team_event_marker(&completed));
        assert!(is_team_event_marker(&tag_chunk(
            "member",
            TeamRole::Teammate,
            json!({"payload": {"event_type": "team.error"}}),
        )));
    }

    /// Agent output, non-object payloads and non-`team.` event types are not markers.
    #[test]
    fn non_markers_are_rejected() {
        // A normal model-output chunk.
        let output = tag_chunk(
            "teammate_1",
            TeamRole::Teammate,
            json!({"type": "answer", "payload": {"content": "hi"}}),
        );
        assert!(!is_team_event_marker(&output));

        // A payload that is not an object.
        assert!(!is_team_event_marker(&tag_chunk(
            "teammate_1",
            TeamRole::Teammate,
            json!({"payload": "team.idle"}),
        )));
        assert!(!is_team_event_marker(&tag_chunk(
            "teammate_1",
            TeamRole::Teammate,
            json!({"payload": ["team.idle"]}),
        )));

        // A bus event type that is not a team marker.
        assert!(!is_team_event_marker(&tag_chunk(
            "teammate_1",
            TeamRole::Teammate,
            json!({"payload": {"event_type": "message"}}),
        )));

        // Not a `TeamOutputSchema`-shaped frame at all: the marker lives under `chunk.chunk`.
        assert!(!is_team_event_marker(&json!({"event_type": "team.idle"})));
        assert!(!is_team_event_marker(&Value::Null));
    }
}
