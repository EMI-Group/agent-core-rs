//! Event-bus envelope, topic grammar and typed event payloads.
//!
//! Python parity source: `openjiuwen/agent_teams/schema/events.py`. The messager moves opaque
//! [`EventMessage`] envelopes; this module gives them their topic grammar and their typed payloads
//! so every producer and consumer of the bus agrees on the wire shape.
//!
//! Topics are `session:<sid>:team:<team>:<kind>`, and every component is escaped by
//! [`topic_part`] (`%` → `%25` first, then `:` → `%3A`) because the session and team ids are
//! human-supplied: an id containing the literal `:team:` would otherwise render a topic that two
//! different (session, team) pairs share, and each could receive the other's events.
//!
//! [`EventPayload`] is deliberately not `serde`-derived: it carries the event-type string in the
//! payload position where Python has a separate `TeamEvent` key, so conversion goes through
//! [`EventPayload::to_message`] / [`EventPayload::from_message`]. An unrecognised event type is
//! preserved as [`EventPayload::Unknown`] rather than rejected — a newer peer's event must not
//! break delivery to an older member.

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use openjiuwen_common::error::{Error, Result};

use super::status::{ExecutionStatus, MemberStatus, TaskStatus};
use super::task::Verdict;

/// Event type of a direct (point-to-point) message.
const EVENT_MESSAGE: &str = "message";
/// Event type of a broadcast message.
const EVENT_BROADCAST: &str = "broadcast";
/// Event type of a member lifecycle-status change.
const EVENT_MEMBER_STATUS_CHANGED: &str = "member_status_changed";
/// Event type of a member execution-status change.
const EVENT_MEMBER_EXECUTION_CHANGED: &str = "member_execution_changed";
/// Event type of a task being created.
const EVENT_TASK_CREATED: &str = "task_created";
/// Event type of a task status transition.
const EVENT_TASK_STATUS_CHANGED: &str = "task_status_changed";
/// Event type of a reviewer passing a task.
const EVENT_TASK_VERIFIED: &str = "task_verified";
/// Event type of a reviewer recording a vote under scheduled dispatch.
const EVENT_TASK_REVIEW_VOTE: &str = "task_review_vote";
/// Event type of a leader approving or rejecting one interrupted tool call.
const EVENT_TOOL_APPROVAL_RESULT: &str = "tool_approval_result";
/// Event type of the whole team reaching a completed state.
const EVENT_TEAM_COMPLETED: &str = "team_completed";

/// One envelope on the event bus: an event type, its payload and the sender that published it.
///
/// The payload stays a raw [`Value`] here — typing it is [`EventPayload`]'s job — because the
/// messager only needs the envelope to route, and a member may legitimately not understand a
/// payload yet.
///
/// `sender_id` defaults to the empty string (Python's default) and is stamped by the messager with
/// the publishing member's name when the producer left it blank.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct EventMessage {
    /// Event type string (see [`EventPayload::event_type`] for the known ones).
    pub event_type: String,
    /// Raw event payload object.
    pub payload: Value,
    /// Identifier of the publishing member, used to filter self-published messages.
    pub sender_id: String,
}

impl EventMessage {
    /// Builds an envelope with an empty [`sender_id`](Self::sender_id).
    #[must_use]
    pub fn new(event_type: impl Into<String>, payload: Value) -> Self {
        Self {
            event_type: event_type.into(),
            payload,
            sender_id: String::new(),
        }
    }

    /// Returns this envelope with [`sender_id`](Self::sender_id) set.
    #[must_use]
    pub fn with_sender(mut self, sender_id: impl Into<String>) -> Self {
        self.sender_id = sender_id.into();
        self
    }

    /// Serializes the envelope to JSON.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] if the envelope cannot be serialized to JSON.
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string(self).map_err(|err| {
            Error::validation(format!(
                "failed to serialize event message '{}': {err}",
                self.event_type
            ))
        })
    }

    /// Deserializes an envelope from JSON.
    ///
    /// Unknown keys are ignored, so an envelope written by a newer release still loads.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] if `data` is not a JSON object matching this envelope.
    pub fn from_json(data: &str) -> Result<Self> {
        serde_json::from_str(data)
            .map_err(|err| Error::validation(format!("invalid event message JSON: {err}")))
    }
}

/// Escapes one topic component so it cannot forge a separator.
///
/// `%` is escaped first, then `:`; the reverse order would turn a literal `%3A` in the input into a
/// `:` the receiver could not distinguish from a real separator.
#[must_use]
pub fn topic_part(value: &str) -> String {
    value.replace('%', "%25").replace(':', "%3A")
}

/// Un-escapes one topic component (the inverse of [`topic_part`]).
fn topic_part_reverse(value: &str) -> String {
    value.replace("%3A", ":").replace("%25", "%")
}

/// Topic categories for team event routing.
///
/// Wire values are the Python `TeamTopic` strings (`team`, `task`, `message`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TeamTopic {
    /// Team-scoped events: lifecycle, status and completion signals.
    #[default]
    Team,
    /// Task-board events.
    Task,
    /// Direct and broadcast message events.
    Message,
}

impl TeamTopic {
    /// Returns the Python wire string of this topic category.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Team => "team",
            Self::Task => "task",
            Self::Message => "message",
        }
    }

    /// Builds the final topic string, `session:<sid>:team:<team>:<category>`.
    ///
    /// Both the session id and the team name are escaped with [`topic_part`].
    #[must_use]
    pub fn topic(self, session_id: &str, team_name: &str) -> String {
        format!(
            "session:{}:team:{}:{}",
            topic_part(session_id),
            topic_part(team_name),
            self.as_str()
        )
    }

    /// Parses a topic string back into `(session_id, team_name, category)`.
    ///
    /// Returns `None` for a topic with the wrong number of segments, the wrong literal
    /// `session` / `team` segments, or an unknown trailing category.
    #[must_use]
    pub fn parse(topic: &str) -> Option<(String, String, TeamTopic)> {
        let parts: Vec<&str> = topic.split(':').collect();
        let [session_literal, session_id, team_literal, team_name, category] = parts.as_slice()
        else {
            return None;
        };
        if *session_literal != "session" || *team_literal != "team" {
            return None;
        }
        let category = match *category {
            "team" => Self::Team,
            "task" => Self::Task,
            "message" => Self::Message,
            _ => return None,
        };
        Some((
            topic_part_reverse(session_id),
            topic_part_reverse(team_name),
            category,
        ))
    }
}

/// Payload of an event published when a point-to-point message is sent.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct MessageEvent {
    /// Message identity.
    pub message_id: String,
    /// Member that sent the message.
    pub from_member_name: String,
    /// Member the message is addressed to (`None` when the target is resolved at delivery).
    pub to_member_name: Option<String>,
}

/// Payload of an event published when a broadcast message is sent.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct BroadcastEvent {
    /// Message identity.
    pub message_id: String,
    /// Member that sent the broadcast.
    pub from_member_name: String,
}

/// Payload of an event published when a member's lifecycle status changes.
///
/// Rust names the transition `status` / `previous` where Python uses `new_status` /
/// `old_status`; the shape is pinned by `src/schema/CONTEXT.md`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct MemberStatusChangedEvent {
    /// Team the member belongs to.
    pub team_name: String,
    /// Member whose status changed.
    pub member_name: String,
    /// Status the member now holds.
    pub status: MemberStatus,
    /// Status the member held before, when the publisher knew it.
    pub previous: Option<MemberStatus>,
}

/// Payload of an event published when a member's execution status changes.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct MemberExecutionChangedEvent {
    /// Team the member belongs to.
    pub team_name: String,
    /// Member whose execution status changed.
    pub member_name: String,
    /// Execution status the member now holds.
    pub status: ExecutionStatus,
}

/// Payload of an event published when a task is created.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct TaskCreatedEvent {
    /// Team owning the task list.
    pub team_name: String,
    /// Identity of the created task.
    pub task_id: String,
    /// Title the task was created with.
    pub title: String,
    /// Member the task was pre-assigned to, if any.
    pub assignee: Option<String>,
}

/// Payload of an event published when a task changes status.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct TaskStatusChangedEvent {
    /// Team owning the task list.
    pub team_name: String,
    /// Identity of the task that moved.
    pub task_id: String,
    /// Status the task now holds.
    pub status: TaskStatus,
    /// Member owning the task after the transition, if any.
    pub assignee: Option<String>,
}

/// Payload of an event published when a reviewer passes a task.
///
/// `member_name` is the task's author, so the completion unblocks downstream tasks exactly as a
/// direct completion does.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct TaskVerifiedEvent {
    /// Team owning the task list.
    pub team_name: String,
    /// Identity of the verified task.
    pub task_id: String,
    /// Author of the task (the task's assignee).
    pub member_name: String,
    /// Verdict that settled the review gate.
    pub verdict: Verdict,
}

/// Payload of an event published when a reviewer records a vote under scheduled dispatch.
///
/// Under scheduled dispatch a verdict only persists a vote; the leader-side scheduler tallies the
/// votes and settles the task. `member_name` is not carried here — the author is read from the task.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct TaskReviewVoteEvent {
    /// Team owning the task list.
    pub team_name: String,
    /// Identity of the task under review.
    pub task_id: String,
    /// Review round the vote belongs to.
    pub round: u32,
    /// Member that cast the vote.
    pub reviewer: String,
    /// The vote itself.
    pub verdict: Verdict,
}

/// Payload of an event published when the leader approves or rejects one interrupted tool call.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ToolApprovalResultEvent {
    /// Team the member belongs to.
    pub team_name: String,
    /// Member that requested the approval.
    pub member_name: String,
    /// Identity of the interrupted tool call.
    pub tool_call_id: String,
    /// Whether the tool call was approved.
    pub approved: bool,
}

/// Payload of an event published when the whole team reaches a completed state.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct TeamCompletedEvent {
    /// Team that completed.
    pub team_name: String,
    /// Why the team was declared complete, when the publisher knows.
    pub reason: Option<String>,
}

/// A typed team event.
///
/// Conversion to and from the wire envelope is manual ([`Self::to_message`] /
/// [`Self::from_message`]) because the envelope stores the event type beside the payload, not
/// inside it. `Unknown` keeps an unrecognised event intact so a newer peer's event can still be
/// routed and logged.
#[derive(Debug, Clone, PartialEq)]
pub enum EventPayload {
    /// A point-to-point message was sent.
    Message(MessageEvent),
    /// A broadcast message was sent.
    Broadcast(BroadcastEvent),
    /// A member's lifecycle status changed.
    MemberStatusChanged(MemberStatusChangedEvent),
    /// A member's execution status changed.
    MemberExecutionChanged(MemberExecutionChangedEvent),
    /// A task was created.
    TaskCreated(TaskCreatedEvent),
    /// A task changed status.
    TaskStatusChanged(TaskStatusChangedEvent),
    /// A reviewer passed a task.
    TaskVerified(TaskVerifiedEvent),
    /// A reviewer recorded a vote.
    TaskReviewVote(TaskReviewVoteEvent),
    /// The leader answered a tool-approval request.
    ToolApprovalResult(ToolApprovalResultEvent),
    /// The team reached a completed state.
    TeamCompleted(TeamCompletedEvent),
    /// An event type this build does not know; the payload is preserved verbatim.
    Unknown {
        /// The raw event-type string.
        event_type: String,
        /// The raw payload object.
        payload: Value,
    },
}

impl EventPayload {
    /// Returns the Python `TeamEvent` wire string of this payload.
    ///
    /// For [`EventPayload::Unknown`] this is the unrecognised string that arrived.
    #[must_use]
    pub fn event_type(&self) -> &str {
        match self {
            Self::Message(_) => EVENT_MESSAGE,
            Self::Broadcast(_) => EVENT_BROADCAST,
            Self::MemberStatusChanged(_) => EVENT_MEMBER_STATUS_CHANGED,
            Self::MemberExecutionChanged(_) => EVENT_MEMBER_EXECUTION_CHANGED,
            Self::TaskCreated(_) => EVENT_TASK_CREATED,
            Self::TaskStatusChanged(_) => EVENT_TASK_STATUS_CHANGED,
            Self::TaskVerified(_) => EVENT_TASK_VERIFIED,
            Self::TaskReviewVote(_) => EVENT_TASK_REVIEW_VOTE,
            Self::ToolApprovalResult(_) => EVENT_TOOL_APPROVAL_RESULT,
            Self::TeamCompleted(_) => EVENT_TEAM_COMPLETED,
            Self::Unknown { event_type, .. } => event_type,
        }
    }

    /// Serializes the payload to a JSON object.
    ///
    /// For [`EventPayload::Unknown`] this is the payload that arrived, unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] if the payload cannot be serialized to JSON.
    pub fn payload_value(&self) -> Result<Value> {
        match self {
            Self::Message(payload) => payload_value(payload, EVENT_MESSAGE),
            Self::Broadcast(payload) => payload_value(payload, EVENT_BROADCAST),
            Self::MemberStatusChanged(payload) => {
                payload_value(payload, EVENT_MEMBER_STATUS_CHANGED)
            }
            Self::MemberExecutionChanged(payload) => {
                payload_value(payload, EVENT_MEMBER_EXECUTION_CHANGED)
            }
            Self::TaskCreated(payload) => payload_value(payload, EVENT_TASK_CREATED),
            Self::TaskStatusChanged(payload) => payload_value(payload, EVENT_TASK_STATUS_CHANGED),
            Self::TaskVerified(payload) => payload_value(payload, EVENT_TASK_VERIFIED),
            Self::TaskReviewVote(payload) => payload_value(payload, EVENT_TASK_REVIEW_VOTE),
            Self::ToolApprovalResult(payload) => payload_value(payload, EVENT_TOOL_APPROVAL_RESULT),
            Self::TeamCompleted(payload) => payload_value(payload, EVENT_TEAM_COMPLETED),
            Self::Unknown { payload, .. } => Ok(payload.clone()),
        }
    }

    /// Builds a payload from an event-type string and its raw JSON payload.
    ///
    /// An unrecognised `event_type` yields [`EventPayload::Unknown`] — never an error — so an
    /// unknown event still reaches subscribers.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when `event_type` is known but `payload` does not match the
    /// payload struct of that event (a corrupted envelope, not a newer peer).
    pub fn from_value(event_type: &str, payload: &Value) -> Result<Self> {
        match event_type {
            EVENT_MESSAGE => Ok(Self::Message(from_payload(event_type, payload)?)),
            EVENT_BROADCAST => Ok(Self::Broadcast(from_payload(event_type, payload)?)),
            EVENT_MEMBER_STATUS_CHANGED => Ok(Self::MemberStatusChanged(from_payload(
                event_type, payload,
            )?)),
            EVENT_MEMBER_EXECUTION_CHANGED => Ok(Self::MemberExecutionChanged(from_payload(
                event_type, payload,
            )?)),
            EVENT_TASK_CREATED => Ok(Self::TaskCreated(from_payload(event_type, payload)?)),
            EVENT_TASK_STATUS_CHANGED => {
                Ok(Self::TaskStatusChanged(from_payload(event_type, payload)?))
            }
            EVENT_TASK_VERIFIED => Ok(Self::TaskVerified(from_payload(event_type, payload)?)),
            EVENT_TASK_REVIEW_VOTE => Ok(Self::TaskReviewVote(from_payload(event_type, payload)?)),
            EVENT_TOOL_APPROVAL_RESULT => {
                Ok(Self::ToolApprovalResult(from_payload(event_type, payload)?))
            }
            EVENT_TEAM_COMPLETED => Ok(Self::TeamCompleted(from_payload(event_type, payload)?)),
            other => Ok(Self::Unknown {
                event_type: other.to_owned(),
                payload: payload.clone(),
            }),
        }
    }

    /// Wraps this payload in an [`EventMessage`] stamped with `sender_id`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] if the payload cannot be serialized to JSON.
    pub fn to_message(&self, sender_id: impl Into<String>) -> Result<EventMessage> {
        Ok(EventMessage {
            event_type: self.event_type().to_owned(),
            payload: self.payload_value()?,
            sender_id: sender_id.into(),
        })
    }

    /// Rebuilds a payload from an envelope, using its `event_type` to pick the variant.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the envelope's `event_type` is known but its payload
    /// does not match that event's struct.
    pub fn from_message(message: &EventMessage) -> Result<Self> {
        Self::from_value(&message.event_type, &message.payload)
    }
}

/// Serializes one payload struct, naming the event in the error message.
fn payload_value<T: Serialize>(payload: &T, event_type: &str) -> Result<Value> {
    serde_json::to_value(payload).map_err(|err| {
        Error::validation(format!(
            "failed to serialize '{event_type}' event payload: {err}"
        ))
    })
}

/// Deserializes one payload struct, naming the event in the error message.
fn from_payload<T: DeserializeOwned>(event_type: &str, payload: &Value) -> Result<T> {
    serde_json::from_value(payload.clone())
        .map_err(|err| Error::validation(format!("invalid '{event_type}' event payload: {err}")))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use serde_json::json;

    use super::*;

    /// Every known event type with one populated payload, in `event_type()` order.
    fn all_payloads() -> Vec<EventPayload> {
        vec![
            EventPayload::Message(MessageEvent {
                message_id: "msg-1".to_owned(),
                from_member_name: "leader".to_owned(),
                to_member_name: Some("bob".to_owned()),
            }),
            EventPayload::Broadcast(BroadcastEvent {
                message_id: "msg-2".to_owned(),
                from_member_name: "leader".to_owned(),
            }),
            EventPayload::MemberStatusChanged(MemberStatusChangedEvent {
                team_name: "team".to_owned(),
                member_name: "bob".to_owned(),
                status: MemberStatus::Ready,
                previous: Some(MemberStatus::Starting),
            }),
            EventPayload::MemberExecutionChanged(MemberExecutionChangedEvent {
                team_name: "team".to_owned(),
                member_name: "bob".to_owned(),
                status: ExecutionStatus::Running,
            }),
            EventPayload::TaskCreated(TaskCreatedEvent {
                team_name: "team".to_owned(),
                task_id: "task-1".to_owned(),
                title: "Port the event bus".to_owned(),
                assignee: Some("bob".to_owned()),
            }),
            EventPayload::TaskStatusChanged(TaskStatusChangedEvent {
                team_name: "team".to_owned(),
                task_id: "task-1".to_owned(),
                status: TaskStatus::InReview,
                assignee: Some("bob".to_owned()),
            }),
            EventPayload::TaskVerified(TaskVerifiedEvent {
                team_name: "team".to_owned(),
                task_id: "task-1".to_owned(),
                member_name: "bob".to_owned(),
                verdict: Verdict::Pass,
            }),
            EventPayload::TaskReviewVote(TaskReviewVoteEvent {
                team_name: "team".to_owned(),
                task_id: "task-1".to_owned(),
                round: 2,
                reviewer: "carol".to_owned(),
                verdict: Verdict::Fail,
            }),
            EventPayload::ToolApprovalResult(ToolApprovalResultEvent {
                team_name: "team".to_owned(),
                member_name: "bob".to_owned(),
                tool_call_id: "call-1".to_owned(),
                approved: true,
            }),
            EventPayload::TeamCompleted(TeamCompletedEvent {
                team_name: "team".to_owned(),
                reason: Some("all tasks terminal".to_owned()),
            }),
        ]
    }

    #[test]
    fn topic_part_escapes_percent_before_colon() {
        assert_eq!(topic_part("plain_name-1"), "plain_name-1");
        assert_eq!(topic_part("%"), "%25");
        assert_eq!(topic_part(":"), "%3A");
        // `%` is escaped first, so a literal `%3A` can never be mistaken for an escaped colon.
        assert_eq!(topic_part("%:"), "%25%3A");
        assert_eq!(topic_part("%3A"), "%253A");
        assert_eq!(topic_part("a:b%c"), "a%3Ab%25c");
        assert_eq!(topic_part(":team:"), "%3Ateam%3A");
    }

    #[test]
    fn topic_renders_every_category() {
        assert_eq!(
            TeamTopic::Team.topic("sess-1", "team-1"),
            "session:sess-1:team:team-1:team"
        );
        assert_eq!(
            TeamTopic::Task.topic("sess-1", "team-1"),
            "session:sess-1:team:team-1:task"
        );
        assert_eq!(
            TeamTopic::Message.topic("sess-1", "team-1"),
            "session:sess-1:team:team-1:message"
        );
    }

    #[test]
    fn topic_escapes_both_components() {
        assert_eq!(
            TeamTopic::Task.topic("a:b", "c%d"),
            "session:a%3Ab:team:c%25d:task"
        );
    }

    #[test]
    fn parse_round_trips_plain_and_escaped_topics() {
        for category in [TeamTopic::Team, TeamTopic::Task, TeamTopic::Message] {
            let topic = category.topic("sess-1", "team-1");
            assert_eq!(
                TeamTopic::parse(&topic),
                Some(("sess-1".to_owned(), "team-1".to_owned(), category)),
                "{topic}"
            );
        }
        let topic = TeamTopic::Message.topic("a:b", "c%d%3A");
        assert_eq!(
            TeamTopic::parse(&topic),
            Some(("a:b".to_owned(), "c%d%3A".to_owned(), TeamTopic::Message))
        );
    }

    #[test]
    fn an_escaped_name_cannot_forge_a_separator() {
        // `x:team:y` would otherwise render a topic indistinguishable from team `y` in session
        // `x`; after escaping the two topics differ and both parse back to their own inputs.
        let forged = TeamTopic::Team.topic("x:team:y", "real");
        let honest = TeamTopic::Team.topic("x", "y:team:real");
        assert_ne!(forged, honest);
        assert_eq!(
            TeamTopic::parse(&forged),
            Some(("x:team:y".to_owned(), "real".to_owned(), TeamTopic::Team))
        );
        assert_eq!(
            TeamTopic::parse(&honest),
            Some(("x".to_owned(), "y:team:real".to_owned(), TeamTopic::Team))
        );
    }

    #[test]
    fn parse_rejects_malformed_topics() {
        for topic in [
            "",
            "team",
            "session:s:team:t",
            "session:s:team:t:team:extra",
            "chan:s:team:t:team",
            "session:s:group:t:team",
            "session:s:team:t:unknown",
            "session::team::",
        ] {
            assert_eq!(TeamTopic::parse(topic), None, "{topic}");
        }
    }

    #[test]
    fn event_message_defaults_and_round_trip() {
        let message = EventMessage::new(EVENT_TEAM_COMPLETED, json!({"team_name": "team"}));
        assert_eq!(message.sender_id, "");
        assert_eq!(message.event_type, "team_completed");
        assert_eq!(message.payload, json!({"team_name": "team"}));

        let stamped = message.clone().with_sender("leader");
        assert_eq!(stamped.sender_id, "leader");

        let json = stamped.to_json().unwrap();
        assert_eq!(EventMessage::from_json(&json).unwrap(), stamped);
    }

    #[test]
    fn event_message_tolerates_unknown_fields_and_defaults() {
        let parsed = EventMessage::from_json(r#"{"event_type":"message","extra":1}"#).unwrap();
        assert_eq!(parsed.payload, Value::Null);
        assert_eq!(parsed.sender_id, "");
        assert_eq!(EventMessage::default().event_type, "");
    }

    #[test]
    fn event_message_from_json_rejects_malformed_input() {
        // `[]` is deliberately absent: serde accepts a struct from a positional sequence, so the
        // empty sequence deserializes to the all-default envelope instead of failing.
        for data in ["", "not json", "{", "[1]", "42", r#"{"sender_id": 3}"#] {
            let err = EventMessage::from_json(data).unwrap_err();
            assert!(
                matches!(err, Error::Validation(_)),
                "{data} produced {err:?}"
            );
        }
    }

    #[test]
    fn every_event_type_round_trips_through_a_message() {
        let payloads = all_payloads();
        assert_eq!(
            payloads
                .iter()
                .map(EventPayload::event_type)
                .collect::<Vec<_>>(),
            vec![
                "message",
                "broadcast",
                "member_status_changed",
                "member_execution_changed",
                "task_created",
                "task_status_changed",
                "task_verified",
                "task_review_vote",
                "tool_approval_result",
                "team_completed",
            ]
        );

        for payload in payloads {
            let message = payload.to_message("leader").unwrap();
            assert_eq!(message.event_type, payload.event_type());
            assert_eq!(message.sender_id, "leader");
            assert_eq!(EventPayload::from_message(&message).unwrap(), payload);

            // The envelope also survives a real JSON trip.
            let reparsed = EventMessage::from_json(&message.to_json().unwrap()).unwrap();
            assert_eq!(EventPayload::from_message(&reparsed).unwrap(), payload);
        }
    }

    #[test]
    fn payload_json_names_follow_the_contract() {
        let vote = EventPayload::TaskReviewVote(TaskReviewVoteEvent {
            team_name: "team".to_owned(),
            task_id: "task-1".to_owned(),
            round: 2,
            reviewer: "carol".to_owned(),
            verdict: Verdict::Fail,
        });
        assert_eq!(
            vote.payload_value().unwrap(),
            json!({
                "team_name": "team",
                "task_id": "task-1",
                "round": 2,
                "reviewer": "carol",
                "verdict": "fail",
            })
        );

        let changed = EventPayload::MemberStatusChanged(MemberStatusChangedEvent {
            team_name: "team".to_owned(),
            member_name: "bob".to_owned(),
            status: MemberStatus::ShutDown,
            previous: Some(MemberStatus::ShutdownRequested),
        });
        assert_eq!(
            changed.payload_value().unwrap(),
            json!({
                "team_name": "team",
                "member_name": "bob",
                "status": "shut_down",
                "previous": "shutdown_requested",
            })
        );

        assert_eq!(
            EventPayload::Broadcast(BroadcastEvent {
                message_id: "msg-2".to_owned(),
                from_member_name: "leader".to_owned(),
            })
            .payload_value()
            .unwrap(),
            json!({"message_id": "msg-2", "from_member_name": "leader"})
        );

        assert_eq!(
            EventPayload::TeamCompleted(TeamCompletedEvent::default())
                .payload_value()
                .unwrap(),
            json!({"team_name": "", "reason": null})
        );
    }

    #[test]
    fn payload_structs_tolerate_unknown_fields_and_defaults() {
        let created: TaskCreatedEvent = serde_json::from_value(json!({
            "team_name": "team",
            "task_id": "task-1",
            "unknown": {"nested": true},
        }))
        .unwrap();
        assert_eq!(created.title, "");
        assert_eq!(created.assignee, None);

        let message: MessageEvent = serde_json::from_value(json!({})).unwrap();
        assert_eq!(message, MessageEvent::default());
        assert_eq!(message.to_member_name, None);

        let tool: ToolApprovalResultEvent =
            serde_json::from_value(json!({"approved": true})).unwrap();
        assert!(tool.approved);
        assert_eq!(tool.team_name, "");
    }

    #[test]
    fn unknown_event_type_is_preserved_not_rejected() {
        let raw = json!({"some": ["future", 1]});
        let payload = EventPayload::from_value("future_event", &raw).unwrap();
        assert_eq!(payload.event_type(), "future_event");
        assert_eq!(
            payload,
            EventPayload::Unknown {
                event_type: "future_event".to_owned(),
                payload: raw.clone(),
            }
        );
        assert_eq!(payload.payload_value().unwrap(), raw);

        let message = payload.to_message("bob").unwrap();
        assert_eq!(message.event_type, "future_event");
        assert_eq!(message.payload, raw);
        assert_eq!(EventPayload::from_message(&message).unwrap(), payload);
    }

    #[test]
    fn known_event_type_with_a_bad_payload_is_an_error() {
        let err = EventPayload::from_value("task_review_vote", &json!({"task_id": 5})).unwrap_err();
        assert!(matches!(err, Error::Validation(_)), "{err:?}");

        let err =
            EventPayload::from_value("member_status_changed", &json!("not an object")).unwrap_err();
        assert!(matches!(err, Error::Validation(_)), "{err:?}");
    }
}
