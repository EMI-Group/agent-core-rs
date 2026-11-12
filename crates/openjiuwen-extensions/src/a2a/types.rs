//! A2A wire models.
//!
//! Port of the protobuf-shaped types used by `openjiuwen.extensions.a2a`
//! (v0.1.20) through the Python `a2a-sdk`. These types cross process
//! boundaries and are also spoken to non-Rust agents, so their JSON shape —
//! not their Rust ergonomics — is what matters here.
//!
//! # Encoding rules
//!
//! - Field names are `camelCase` on the wire (`messageId`, `supportedInterfaces`,
//!   `lastChunk`, ...) and `snake_case` in Rust.
//! - `Option` fields are **omitted** when `None` instead of being sent as
//!   `null`. That matches protobuf JSON, which never emits an unset message or
//!   scalar field; peers that read either form still understand the card and
//!   event payloads.
//! - Repeated fields, maps, and plain strings always serialize (an empty list
//!   becomes `[]`). protobuf JSON readers accept both, and emitting them keeps
//!   the decoders on this side straightforward.
//! - Every field decodes with a default, so a partial payload — for example a
//!   bare `{"state": "working"}` status — never fails to parse.
//! - Unknown fields are ignored (there is no `deny_unknown_fields`), so a peer
//!   that adds a field does not break this side.
//! - [`Role`] and [`TaskState`] serialize to the proto identifiers
//!   (`ROLE_USER`, `TASK_STATE_WORKING`) and decode tolerantly from the
//!   canonical value, the bare suffix, and `snake_case`/`camelCase` spellings;
//!   an unrecognized value decodes to `Unspecified` instead of failing.
use serde::de::{self, Deserializer};
use serde::{Deserialize, Serialize, Serializer};
use serde_json::Value;
use std::collections::BTreeMap;
use std::fmt;
/// Uppercases, trims, and strips a leading `ROLE_` / `TASK_STATE_` prefix.
///
/// This is the single normalisation step shared by [`Role`] and [`TaskState`]:
/// `" role_user "` and `"Role_User"` both become `"USER"`.
fn normalize(raw: &str) -> String {
    let upper = raw.trim().to_ascii_uppercase();
    upper
        .strip_prefix("ROLE_")
        .or_else(|| upper.strip_prefix("TASK_STATE_"))
        .unwrap_or(upper.as_str())
        .to_owned()
}
/// The author of a [`Message`].
///
/// Serializes as `ROLE_UNSPECIFIED` / `ROLE_USER` / `ROLE_AGENT`, and decodes
/// from those values as well as `"user"`, `"ROLE_USER"`, `"role_user"`, and
/// `"agent"`. Anything unrecognized (and `null`) decodes to
/// [`Role::Unspecified`] rather than failing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Role {
    /// The role was not set on the wire.
    #[default]
    Unspecified,
    /// A message authored by the end user.
    User,
    /// A message authored by the agent.
    Agent,
}
impl Role {
    /// The proto identifier of this role (`ROLE_USER`, ...).
    #[must_use]
    pub const fn as_wire_str(self) -> &'static str {
        match self {
            Self::Unspecified => "ROLE_UNSPECIFIED",
            Self::User => "ROLE_USER",
            Self::Agent => "ROLE_AGENT",
        }
    }
    /// Decodes a role tolerantly from any of its accepted spellings.
    fn from_wire_str(raw: &str) -> Self {
        match normalize(raw).as_str() {
            "USER" => Self::User,
            "AGENT" => Self::Agent,
            _ => Self::Unspecified,
        }
    }
}
impl fmt::Display for Role {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_wire_str())
    }
}
impl Serialize for Role {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.as_wire_str())
    }
}
impl<'de> Deserialize<'de> for Role {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        match Value::deserialize(deserializer)? {
            Value::String(raw) => Ok(Self::from_wire_str(&raw)),
            // protobuf JSON never emits `null` for an enum, but a tolerant
            // reader is preferable to a decode failure across process borders.
            Value::Null => Ok(Self::Unspecified),
            other => Err(de::Error::custom(format!(
                "expected a role string such as \"ROLE_USER\", found {other}"
            ))),
        }
    }
}
/// The lifecycle state of a [`Task`].
///
/// Serializes as `TASK_STATE_*` and decodes from the canonical value, the bare
/// `snake_case` suffix, the `camelCase` suffix, and the two-L spelling
/// `cancelled`. Unrecognized values and `null` decode to
/// [`TaskState::Unspecified`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum TaskState {
    /// The state was not set on the wire.
    #[default]
    Unspecified,
    /// The task was accepted but has not started.
    Submitted,
    /// The task is running.
    Working,
    /// The task finished successfully.
    Completed,
    /// The task failed.
    Failed,
    /// The task was cancelled.
    Canceled,
    /// The task is waiting for more input from the user.
    InputRequired,
    /// The agent rejected the task.
    Rejected,
    /// The task needs authentication before it can proceed.
    AuthRequired,
}
impl TaskState {
    /// The proto identifier of this state (`TASK_STATE_WORKING`, ...).
    #[must_use]
    pub const fn as_wire_str(self) -> &'static str {
        match self {
            Self::Unspecified => "TASK_STATE_UNSPECIFIED",
            Self::Submitted => "TASK_STATE_SUBMITTED",
            Self::Working => "TASK_STATE_WORKING",
            Self::Completed => "TASK_STATE_COMPLETED",
            Self::Failed => "TASK_STATE_FAILED",
            Self::Canceled => "TASK_STATE_CANCELED",
            Self::InputRequired => "TASK_STATE_INPUT_REQUIRED",
            Self::Rejected => "TASK_STATE_REJECTED",
            Self::AuthRequired => "TASK_STATE_AUTH_REQUIRED",
        }
    }
    /// Whether this is a terminal state (`COMPLETED`, `FAILED`, `CANCELED`,
    /// `REJECTED`).
    #[must_use]
    pub const fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Completed | Self::Failed | Self::Canceled | Self::Rejected
        )
    }
    /// Decodes a state tolerantly; unknown values become
    /// [`TaskState::Unspecified`].
    ///
    /// Underscores are dropped before matching, so `"input_required"` and
    /// `"inputRequired"` both land on [`TaskState::InputRequired`].
    fn from_wire_str(raw: &str) -> Self {
        match normalize(raw).replace('_', "").as_str() {
            "SUBMITTED" => Self::Submitted,
            "WORKING" => Self::Working,
            "COMPLETED" => Self::Completed,
            "FAILED" => Self::Failed,
            "CANCELED" | "CANCELLED" => Self::Canceled,
            "INPUTREQUIRED" => Self::InputRequired,
            "REJECTED" => Self::Rejected,
            "AUTHREQUIRED" => Self::AuthRequired,
            _ => Self::Unspecified,
        }
    }
}
impl fmt::Display for TaskState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_wire_str())
    }
}
impl Serialize for TaskState {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.as_wire_str())
    }
}
impl<'de> Deserialize<'de> for TaskState {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        match Value::deserialize(deserializer)? {
            Value::String(raw) => Ok(Self::from_wire_str(&raw)),
            // protobuf JSON never emits `null` for an enum, but a tolerant
            // reader is preferable to a decode failure across process borders.
            Value::Null => Ok(Self::Unspecified),
            other => Err(de::Error::custom(format!(
                "expected a task-state string such as \"TASK_STATE_WORKING\", found {other}"
            ))),
        }
    }
}
/// The capability flags advertised on an [`AgentCard`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentCapabilities {
    /// Whether the agent supports `SendStreamingMessage` (`streaming`).
    #[serde(default)]
    pub streaming: bool,
    /// Whether the agent supports push notifications (`pushNotifications`).
    #[serde(default)]
    pub push_notifications: bool,
}
/// One transport endpoint of an [`AgentCard`].
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentInterface {
    /// The endpoint URL.
    #[serde(default)]
    pub url: String,
    /// The transport binding, for example `JSONRPC`.
    #[serde(default)]
    pub protocol_binding: String,
    /// The protocol version spoken at this endpoint, for example `1.0`.
    #[serde(default)]
    pub protocol_version: String,
    /// An optional tenant selector.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tenant: Option<String>,
}
/// A single advertised skill of an [`AgentCard`].
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentSkill {
    /// Stable skill id.
    #[serde(default)]
    pub id: String,
    /// Human-readable skill name.
    #[serde(default)]
    pub name: String,
    /// What the skill does.
    #[serde(default)]
    pub description: String,
    /// Free-form tags used for discovery.
    #[serde(default)]
    pub tags: Vec<String>,
    /// Example prompts that exercise the skill.
    #[serde(default)]
    pub examples: Vec<String>,
    /// Media types the skill accepts.
    #[serde(default)]
    pub input_modes: Vec<String>,
    /// Media types the skill produces.
    #[serde(default)]
    pub output_modes: Vec<String>,
}
/// The organisation that publishes an [`AgentCard`].
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentProvider {
    /// The organisation name.
    #[serde(default)]
    pub organization: String,
    /// A URL for the organisation.
    #[serde(default)]
    pub url: String,
}
/// The A2A agent card served at `/.well-known/agent-card.json`.
///
/// Built locally by `card::AgentCardAdapter::to_a2a` and consumed by
/// `server::A2aServerConfig`.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentCard {
    /// The agent name.
    #[serde(default)]
    pub name: String,
    /// The agent description, including the embedded parameter JSON sections.
    #[serde(default)]
    pub description: String,
    /// The agent version.
    #[serde(default)]
    pub version: String,
    /// Every endpoint the agent is reachable at.
    #[serde(default)]
    pub supported_interfaces: Vec<AgentInterface>,
    /// The advertised capability flags.
    #[serde(default)]
    pub capabilities: AgentCapabilities,
    /// Media types the agent accepts by default.
    #[serde(default)]
    pub default_input_modes: Vec<String>,
    /// Media types the agent produces by default.
    #[serde(default)]
    pub default_output_modes: Vec<String>,
    /// The advertised skills.
    #[serde(default)]
    pub skills: Vec<AgentSkill>,
    /// The publishing organisation, when advertised.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<AgentProvider>,
    /// Named security schemes, keyed by scheme name.
    #[serde(default)]
    pub security_schemes: serde_json::Map<String, Value>,
    /// Security requirements: each entry maps a scheme name to its scopes.
    ///
    /// The wire shape is a list of objects mapping a scheme name to a list of
    /// scope strings. A [`BTreeMap`] is used rather than `serde_json::Map`
    /// because `serde_json` implements `Serialize`, `Deserialize`, `Debug`,
    /// `Clone`, and `PartialEq` only for `Map<String, Value>`; the JSON output
    /// is identical (object keys are emitted in sorted order either way).
    #[serde(default)]
    pub security: Vec<BTreeMap<String, Vec<String>>>,
}
/// A single content part of a [`Message`] or [`Artifact`].
///
/// A part carries text, raw bytes, a URL, or structured data; exactly which
/// fields are set is decided by the producer. Use [`Part::text`] for the common
/// text case.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Part {
    /// The textual content.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// Base64-encoded raw bytes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw: Option<String>,
    /// A URL pointing at the content.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    /// Structured JSON content.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
    /// The file name, for file parts.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filename: Option<String>,
    /// The media type of the content.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub media_type: Option<String>,
    /// Part-level metadata.
    #[serde(default)]
    pub metadata: serde_json::Map<String, Value>,
}
impl Part {
    /// A part holding only text.
    #[must_use]
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            text: Some(text.into()),
            ..Self::default()
        }
    }
}
/// A message exchanged between a client and an agent.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Message {
    /// A unique message id (a fresh `uuid4().hex` on the client side).
    #[serde(default)]
    pub message_id: String,
    /// The conversation this message belongs to.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_id: Option<String>,
    /// The task this message is a follow-up to.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task_id: Option<String>,
    /// Who authored the message.
    #[serde(default)]
    pub role: Role,
    /// The content parts.
    #[serde(default)]
    pub parts: Vec<Part>,
    /// Message-level metadata (carries the non-`query` input fields).
    #[serde(default)]
    pub metadata: serde_json::Map<String, Value>,
    /// URIs of extensions the sender used.
    #[serde(default)]
    pub extensions: Vec<String>,
    /// Ids of the tasks this message references.
    #[serde(default)]
    pub reference_task_ids: Vec<String>,
}
/// The current status of a [`Task`].
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TaskStatus {
    /// The lifecycle state; a bare status payload decodes to
    /// [`TaskState::Unspecified`].
    #[serde(default)]
    pub state: TaskState,
    /// An optional message accompanying the state.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<Message>,
    /// An optional timestamp supplied by the producer.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<String>,
}
/// A named output produced by a [`Task`].
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Artifact {
    /// The artifact id, unique within its task.
    #[serde(default)]
    pub artifact_id: String,
    /// An optional human-readable name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// An optional description.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// The artifact content.
    #[serde(default)]
    pub parts: Vec<Part>,
    /// Artifact-level metadata.
    #[serde(default)]
    pub metadata: serde_json::Map<String, Value>,
    /// URIs of extensions the producer used.
    #[serde(default)]
    pub extensions: Vec<String>,
}
/// A unit of work tracked by an A2A agent.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Task {
    /// The task id.
    #[serde(default)]
    pub id: String,
    /// The conversation this task belongs to.
    #[serde(default)]
    pub context_id: String,
    /// The current status.
    #[serde(default)]
    pub status: TaskStatus,
    /// The artifacts produced so far.
    #[serde(default)]
    pub artifacts: Vec<Artifact>,
    /// The messages exchanged so far.
    #[serde(default)]
    pub history: Vec<Message>,
    /// Task-level metadata.
    #[serde(default)]
    pub metadata: serde_json::Map<String, Value>,
}
/// A streamed status transition of a [`Task`].
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TaskStatusUpdateEvent {
    /// The task the update belongs to.
    #[serde(default)]
    pub task_id: String,
    /// The conversation the task belongs to.
    #[serde(default)]
    pub context_id: String,
    /// The new status.
    #[serde(default)]
    pub status: TaskStatus,
    /// Event-level metadata.
    #[serde(default)]
    pub metadata: serde_json::Map<String, Value>,
}
/// A streamed artifact chunk of a [`Task`].
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TaskArtifactUpdateEvent {
    /// The task the artifact belongs to.
    #[serde(default)]
    pub task_id: String,
    /// The conversation the task belongs to.
    #[serde(default)]
    pub context_id: String,
    /// The artifact (or artifact chunk) itself.
    #[serde(default)]
    pub artifact: Artifact,
    /// Whether the chunk appends to the artifact delivered so far.
    #[serde(default)]
    pub append: bool,
    /// Whether this is the last chunk of the artifact.
    #[serde(default)]
    pub last_chunk: bool,
    /// Event-level metadata.
    #[serde(default)]
    pub metadata: serde_json::Map<String, Value>,
}
/// The params of a `SendMessage` / `SendStreamingMessage` call.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SendMessageRequest {
    /// An optional tenant selector.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tenant: Option<String>,
    /// The message to send.
    #[serde(default)]
    pub message: Message,
    /// Optional send configuration, passed through untouched.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub configuration: Option<Value>,
    /// Request-level metadata.
    #[serde(default)]
    pub metadata: serde_json::Map<String, Value>,
}
/// The params of a `CancelTask` call.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CancelTaskRequest {
    /// An optional tenant selector.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tenant: Option<String>,
    /// The id of the task to cancel.
    #[serde(default)]
    pub id: String,
    /// Request-level metadata.
    #[serde(default)]
    pub metadata: serde_json::Map<String, Value>,
}
#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use serde_json::json;
    #[test]
    fn message_json_uses_camel_case_and_omits_unset_options() {
        // ports `tests/unit_tests/extensions/a2a/test_a2a_transformer.py::test_to_a2a_request`.
        let message = Message {
            message_id: "abc".to_owned(),
            role: Role::User,
            parts: vec![Part::text("hello")],
            ..Message::default()
        };
        let value = serde_json::to_value(&message).unwrap();
        let object = value.as_object().unwrap();
        assert_eq!(object.get("messageId"), Some(&json!("abc")));
        assert_eq!(object.get("role"), Some(&json!("ROLE_USER")));
        assert_eq!(
            object.get("parts"),
            Some(&json!([{ "text": "hello", "metadata": {} }]))
        );
        assert_eq!(object.get("metadata"), Some(&json!({})));
        assert!(!object.contains_key("contextId"));
        assert!(!object.contains_key("taskId"));
        assert!(!object.contains_key("context_id"));
        assert!(!object.contains_key("task_id"));
    }
    #[test]
    fn part_json_uses_camel_case_and_omits_unset_options() {
        let part = Part::text("hi");
        assert_eq!(
            serde_json::to_value(&part).unwrap(),
            json!({ "text": "hi", "metadata": {} })
        );
        let file = Part {
            url: Some("https://example.test/a.bin".to_owned()),
            filename: Some("a.bin".to_owned()),
            media_type: Some("application/octet-stream".to_owned()),
            ..Part::default()
        };
        let value = serde_json::to_value(&file).unwrap();
        assert_eq!(
            value,
            json!({
                "url": "https://example.test/a.bin",
                "filename": "a.bin",
                "mediaType": "application/octet-stream",
                "metadata": {},
            })
        );
        assert!(value.get("raw").is_none());
        assert!(value.get("data").is_none());
    }
    #[test]
    fn task_artifact_update_event_json_uses_camel_case() {
        let event = TaskArtifactUpdateEvent {
            task_id: "t1".to_owned(),
            context_id: "c1".to_owned(),
            artifact: Artifact {
                artifact_id: "a1".to_owned(),
                parts: vec![Part::text("chunk")],
                ..Artifact::default()
            },
            append: true,
            last_chunk: true,
            ..TaskArtifactUpdateEvent::default()
        };
        let value = serde_json::to_value(&event).unwrap();
        assert_eq!(value["taskId"], json!("t1"));
        assert_eq!(value["contextId"], json!("c1"));
        assert_eq!(value["lastChunk"], json!(true));
        assert_eq!(value["append"], json!(true));
        assert_eq!(value["artifact"]["artifactId"], json!("a1"));
        assert!(value["artifact"].get("name").is_none());
    }
    #[test]
    fn agent_card_round_trips_with_camel_case() {
        let card = AgentCard {
            name: "agent".to_owned(),
            description: "does things".to_owned(),
            version: "1.0".to_owned(),
            supported_interfaces: vec![AgentInterface {
                url: "http://127.0.0.1:1/a2a/jsonrpc/".to_owned(),
                protocol_binding: "JSONRPC".to_owned(),
                protocol_version: "1.0".to_owned(),
                tenant: None,
            }],
            capabilities: AgentCapabilities {
                streaming: true,
                push_notifications: false,
            },
            default_input_modes: vec!["text/plain".to_owned()],
            default_output_modes: vec!["text/plain".to_owned()],
            skills: vec![AgentSkill {
                id: "s1".to_owned(),
                input_modes: vec!["text/plain".to_owned()],
                ..AgentSkill::default()
            }],
            provider: Some(AgentProvider {
                organization: "openJiuwen".to_owned(),
                url: "https://example.test".to_owned(),
            }),
            ..AgentCard::default()
        };
        let value = serde_json::to_value(&card).unwrap();
        assert_eq!(
            value["supportedInterfaces"][0]["protocolBinding"],
            "JSONRPC"
        );
        assert_eq!(value["capabilities"]["pushNotifications"], json!(false));
        assert_eq!(value["defaultInputModes"], json!(["text/plain"]));
        assert_eq!(value["skills"][0]["inputModes"], json!(["text/plain"]));
        assert_eq!(value["provider"]["organization"], "openJiuwen");
        assert_eq!(value["securitySchemes"], json!({}));
        assert_eq!(value["security"], json!([]));
        let back: AgentCard = serde_json::from_value(value).unwrap();
        assert_eq!(back, card);
    }
    #[test]
    fn role_serializes_as_proto_strings() {
        assert_eq!(
            serde_json::to_value(Role::Unspecified).unwrap(),
            json!("ROLE_UNSPECIFIED")
        );
        assert_eq!(
            serde_json::to_value(Role::User).unwrap(),
            json!("ROLE_USER")
        );
        assert_eq!(
            serde_json::to_value(Role::Agent).unwrap(),
            json!("ROLE_AGENT")
        );
        assert_eq!(Role::User.to_string(), "ROLE_USER");
        assert_eq!(Role::default(), Role::Unspecified);
    }
    #[test]
    fn role_deserializes_tolerantly() {
        for raw in ["ROLE_USER", "role_user", " user ", "user", "USER", "User"] {
            let role: Role = serde_json::from_value(json!(raw)).unwrap();
            assert_eq!(role, Role::User, "`{raw}` should decode to ROLE_USER");
        }
        for raw in ["ROLE_AGENT", "agent", "Agent"] {
            let role: Role = serde_json::from_value(json!(raw)).unwrap();
            assert_eq!(role, Role::Agent);
        }
        for raw in ["ROLE_UNSPECIFIED", "unspecified", "", "ROLE_ROBOT"] {
            let role: Role = serde_json::from_value(json!(raw)).unwrap();
            assert_eq!(role, Role::Unspecified, "`{raw}` is not a known role");
        }
        let null_role: Role = serde_json::from_value(json!(null)).unwrap();
        assert_eq!(null_role, Role::Unspecified);
    }
    #[test]
    fn task_state_serializes_as_proto_strings() {
        let expected = [
            (TaskState::Unspecified, "TASK_STATE_UNSPECIFIED"),
            (TaskState::Submitted, "TASK_STATE_SUBMITTED"),
            (TaskState::Working, "TASK_STATE_WORKING"),
            (TaskState::Completed, "TASK_STATE_COMPLETED"),
            (TaskState::Failed, "TASK_STATE_FAILED"),
            (TaskState::Canceled, "TASK_STATE_CANCELED"),
            (TaskState::InputRequired, "TASK_STATE_INPUT_REQUIRED"),
            (TaskState::Rejected, "TASK_STATE_REJECTED"),
            (TaskState::AuthRequired, "TASK_STATE_AUTH_REQUIRED"),
        ];
        for (state, wire) in expected {
            assert_eq!(state.as_wire_str(), wire);
            assert_eq!(state.to_string(), wire);
            assert_eq!(serde_json::to_value(state).unwrap(), json!(wire));
        }
        assert_eq!(TaskState::default(), TaskState::Unspecified);
    }
    #[test]
    fn task_state_deserializes_tolerantly() {
        // `inputRequired` is the camelCase spelling, `cancelled` the two-L one.
        let accepted = [
            ("TASK_STATE_WORKING", TaskState::Working),
            ("working", TaskState::Working),
            ("WORKING", TaskState::Working),
            ("TASK_STATE_CANCELED", TaskState::Canceled),
            ("canceled", TaskState::Canceled),
            ("cancelled", TaskState::Canceled),
            ("TASK_STATE_CANCELLED", TaskState::Canceled),
            ("inputRequired", TaskState::InputRequired),
            ("input_required", TaskState::InputRequired),
            ("TASK_STATE_INPUT_REQUIRED", TaskState::InputRequired),
            ("TASK_STATE_AUTH_REQUIRED", TaskState::AuthRequired),
            ("authRequired", TaskState::AuthRequired),
            ("submitted", TaskState::Submitted),
            ("completed", TaskState::Completed),
            ("failed", TaskState::Failed),
            ("rejected", TaskState::Rejected),
            ("unspecified", TaskState::Unspecified),
            ("  TASK_STATE_WORKING  ", TaskState::Working),
        ];
        for (raw, expected) in accepted {
            let state: TaskState = serde_json::from_value(json!(raw)).unwrap();
            assert_eq!(state, expected, "`{raw}` decoded to {state}");
        }
        for raw in ["nonsense", "", "TASK_STATE_TELEPORTING"] {
            let state: TaskState = serde_json::from_value(json!(raw)).unwrap();
            assert_eq!(state, TaskState::Unspecified, "`{raw}` must be tolerant");
        }
        let null_state: TaskState = serde_json::from_value(json!(null)).unwrap();
        assert_eq!(null_state, TaskState::Unspecified);
        assert!(serde_json::from_value::<TaskState>(json!(42)).is_err());
    }
    #[test]
    fn task_state_terminal_states() {
        for state in [
            TaskState::Completed,
            TaskState::Failed,
            TaskState::Canceled,
            TaskState::Rejected,
        ] {
            assert!(state.is_terminal(), "{state} is terminal");
        }
        for state in [
            TaskState::Unspecified,
            TaskState::Submitted,
            TaskState::Working,
            TaskState::InputRequired,
            TaskState::AuthRequired,
        ] {
            assert!(!state.is_terminal(), "{state} is not terminal");
        }
    }
    #[test]
    fn unknown_json_keys_are_ignored() {
        let message: Message = serde_json::from_value(json!({
            "messageId": "m1",
            "role": "ROLE_AGENT",
            "parts": [{ "text": "hi", "kind": "text", "extra": 1 }],
            "brandNewField": { "nested": true },
        }))
        .unwrap();
        assert_eq!(message.message_id, "m1");
        assert_eq!(message.role, Role::Agent);
        assert_eq!(message.parts, vec![Part::text("hi")]);
        let card: AgentCard = serde_json::from_value(json!({
            "name": "a",
            "protocolVersion": "0.3",
            "unknown": [1, 2, 3],
        }))
        .unwrap();
        assert_eq!(card.name, "a");
        assert_eq!(card.version, "");
    }
    #[test]
    fn partial_payloads_decode_with_defaults() {
        let status: TaskStatus = serde_json::from_value(json!({ "state": "working" })).unwrap();
        assert_eq!(status.state, TaskState::Working);
        assert!(status.message.is_none());
        assert!(status.timestamp.is_none());
        let empty_status: TaskStatus = serde_json::from_value(json!({})).unwrap();
        assert_eq!(empty_status.state, TaskState::Unspecified);
        let message: Message = serde_json::from_value(json!({})).unwrap();
        assert_eq!(message, Message::default());
        let request: SendMessageRequest =
            serde_json::from_value(json!({ "message": { "parts": [{ "text": "q" }] } })).unwrap();
        assert!(request.tenant.is_none());
        assert!(request.configuration.is_none());
        assert_eq!(request.message.parts, vec![Part::text("q")]);
        let cancel: CancelTaskRequest = serde_json::from_value(json!({ "id": "t9" })).unwrap();
        assert_eq!(cancel.id, "t9");
    }
    #[test]
    fn part_text_constructor_sets_only_text() {
        let part = Part::text(String::from("built"));
        assert_eq!(part.text.as_deref(), Some("built"));
        assert!(part.raw.is_none());
        assert!(part.url.is_none());
        assert!(part.data.is_none());
        assert!(part.filename.is_none());
        assert!(part.media_type.is_none());
        assert_eq!(part.metadata, serde_json::Map::new());
    }
    #[test]
    fn defaults_are_empty() {
        let card = AgentCard::default();
        assert_eq!(card.name, "");
        assert_eq!(card.supported_interfaces, Vec::new());
        assert_eq!(card.skills, Vec::new());
        assert!(card.provider.is_none());
        assert_eq!(card.security_schemes, serde_json::Map::new());
        assert_eq!(card.security, Vec::new());
        let capabilities = AgentCapabilities::default();
        assert!(!capabilities.streaming);
        assert!(!capabilities.push_notifications);
        assert_eq!(AgentSkill::default().id, "");
        assert_eq!(AgentProvider::default().url, "");
        assert!(Part::default().text.is_none());
        assert_eq!(Task::default().status, TaskStatus::default());
        assert!(!TaskArtifactUpdateEvent::default().last_chunk);
        assert_eq!(Message::default().role, Role::Unspecified);
        assert!(Artifact::default().name.is_none());
        assert!(CancelTaskRequest::default().tenant.is_none());
        assert!(SendMessageRequest::default().configuration.is_none());
    }
}
