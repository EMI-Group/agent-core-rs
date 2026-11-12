//! Mapping between openJiuwen payloads and A2A payloads.
//!
//! Port of `openjiuwen/extensions/a2a/a2a_transformer.py` (v0.1.20), plus the
//! client-side result helpers that the Python implementation keeps as private
//! methods of `A2AAgentClient` in `a2a_client.py` (`_resolve_session_id`,
//! `_with_session_id`, `_merge_agent_results`). The openJiuwen-side result model
//! mirrors `openjiuwen/core/single_agent/schema/agent_result.py`, and
//! [`AgentStatus`] carries the values of
//! `openjiuwen/core/controller/schema/task.py::TaskStatus`.
//!
//! # Request mapping (`to_request`)
//!
//! A plain payload object becomes a [`SendMessageRequest`]: a fresh
//! `uuid4().hex`-shaped message id (32 lowercase hex characters),
//! `role = ROLE_USER`, `context_id` from `conversation_id` (falling back to
//! `sessionId`), the `query` value as the single text part, and every other
//! non-null input key copied into the message metadata. `query`, `sessionId`,
//! and `conversation_id` are never copied into the metadata.
//!
//! # Response mapping (`from_response`)
//!
//! The JSON-RPC `result` value is classified in the Python order, so a task
//! object (which has `id`, `status`, and `artifacts`) is never mistaken for a
//! status update, and an artifact update is never mistaken for a task:
//!
//! | Shape | Status | Artifacts |
//! |---|---|---|
//! | `artifact` object (artifact update) | `working` | the event's artifact |
//! | `status` object without `id` (status update) | mapped from `status.state` | none |
//! | `role` + `parts` (message) | `completed` | one `message` artifact |
//! | `id` + `status` object (task) | mapped from `status.state` | the task's artifacts |
//! | anything else | `completed` | none |
//!
//! The Python protobuf wrapper keys `artifact_update` / `artifactUpdate` and
//! `status_update` / `statusUpdate` are unwrapped before classification.
//!
//! # A2A state → openJiuwen status
//!
//! | `TaskState` | `AgentStatus` |
//! |---|---|
//! | `Unspecified` | `Unknown` |
//! | `Submitted` | `Submitted` |
//! | `Working` | `Working` |
//! | `Completed` | `Completed` |
//! | `Failed` | `Failed` |
//! | `Canceled` | `Canceled` |
//! | `InputRequired` | `InputRequired` |
//! | `Rejected` | `Failed` |
//! | `AuthRequired` | `InputRequired` |
//!
//! # Metadata merge rule
//!
//! Metadata is merged last and never overrides the payload: a metadata entry is
//! inserted only when its value is non-null **and** the payload does not already
//! have that key. This is Python's `_merge_metadata` behaviour and is what keeps
//! a `message_id`-derived `task_id`/`sessionId` authoritative over stale
//! metadata.
//!
//! # Documented deviations from Python
//!
//! 1. `data` parts keep their structured JSON value. Python stringifies
//!    non-dict data with `str()` (`_a2a_part_to_part`); here the JSON value
//!    round-trips unchanged so rich payloads stay machine-readable.
//! 2. Python defaults `AgentResult.status` to `None`; [`AgentResult`] defaults to
//!    [`AgentStatus::Unknown`] so the field is total in Rust. The status tests
//!    pin the same meaning: "unset" is `unknown`.
use openjiuwen_common::error::{Error, Result};
use serde::de::{DeserializeOwned, Deserializer};
use serde::{Deserialize, Serialize, Serializer};
use serde_json::{Map, Value};

use super::types::{
    Artifact, Message, Part, Role, SendMessageRequest, Task, TaskState, TaskStatus,
};

/// The input keys that are consumed by the request mapping and therefore never
/// copied into the message metadata (Python's exclusion set).
const RESERVED_INPUT_KEYS: [&str; 3] = ["query", "sessionId", "conversation_id"];

/// The status of an openJiuwen agent result.
///
/// Serializes to the Python `TaskStatus` values: `submitted`, `working`,
/// `paused`, `input-required` (with a hyphen), `completed`, `canceled`,
/// `failed`, `waiting`, and `unknown`.
///
/// Decoding is tolerant: `input_required`, `inputRequired`, and all-uppercase
/// spellings are accepted, `cancelled` decodes to [`AgentStatus::Canceled`], and
/// any unrecognized value — including a non-string — decodes to
/// [`AgentStatus::Unknown`] instead of failing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum AgentStatus {
    /// The task was accepted but has not started.
    Submitted,
    /// The task is running.
    Working,
    /// The task was paused.
    Paused,
    /// The task is waiting for more input from the user.
    InputRequired,
    /// The task finished successfully.
    Completed,
    /// The task was cancelled.
    Canceled,
    /// The task failed.
    Failed,
    /// The task is waiting for a dependency to complete.
    Waiting,
    /// The status is unset or unrecognized; stands in for Python's `None`.
    #[default]
    Unknown,
}

impl AgentStatus {
    /// The Python `TaskStatus` value of this status.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Submitted => "submitted",
            Self::Working => "working",
            Self::Paused => "paused",
            Self::InputRequired => "input-required",
            Self::Completed => "completed",
            Self::Canceled => "canceled",
            Self::Failed => "failed",
            Self::Waiting => "waiting",
            Self::Unknown => "unknown",
        }
    }

    /// Whether the agent has stopped working (Python `_HALT_STATUSES`):
    /// `completed`, `canceled`, `failed`, and `input-required`.
    #[must_use]
    pub const fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Completed | Self::Canceled | Self::Failed | Self::InputRequired
        )
    }

    /// Whether the status is unset or unrecognized.
    #[must_use]
    pub const fn is_unknown(self) -> bool {
        matches!(self, Self::Unknown)
    }

    /// Decodes a status tolerantly; unknown values become [`Self::Unknown`].
    fn from_wire_str(raw: &str) -> Self {
        let normalized = raw.trim().to_ascii_lowercase().replace(['-', '_'], "");
        match normalized.as_str() {
            "submitted" => Self::Submitted,
            "working" => Self::Working,
            "paused" => Self::Paused,
            "inputrequired" => Self::InputRequired,
            "completed" => Self::Completed,
            "canceled" | "cancelled" => Self::Canceled,
            "failed" => Self::Failed,
            "waiting" => Self::Waiting,
            _ => Self::Unknown,
        }
    }
}

impl Serialize for AgentStatus {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for AgentStatus {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        match Value::deserialize(deserializer)? {
            Value::String(raw) => Ok(Self::from_wire_str(&raw)),
            // A peer that emits `null`, a number, or an object for the status is
            // unsetting it; a tolerant reader beats a decode failure.
            _ => Ok(Self::Unknown),
        }
    }
}

/// One content part of an openJiuwen [`AgentArtifact`].
///
/// Mirrors the Python `Part` model: `media_type` is `mediaType` on the wire and
/// every optional field is omitted when unset.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentPart {
    /// The textual content.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// Raw bytes, as carried by the producer.
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
    /// The media type of the content (`mediaType` on the wire).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub media_type: Option<String>,
    /// Part-level metadata.
    #[serde(default)]
    pub metadata: Map<String, Value>,
}

/// A named output produced by an agent.
///
/// Mirrors the Python `Artifact` model: the id is `artifactId` on the wire.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentArtifact {
    /// The artifact id (`artifactId` on the wire).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifact_id: Option<String>,
    /// A semantic name, for example `summary`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// A human-readable description.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// The artifact content.
    #[serde(default)]
    pub parts: Vec<AgentPart>,
    /// Artifact-level metadata.
    #[serde(default)]
    pub metadata: Map<String, Value>,
}

/// The result of one agent turn.
///
/// The JSON shape is Python's `model_dump()`: `task_id` stays snake case while
/// `sessionId` is camel case, and unset options are omitted.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct AgentResult {
    /// The task this result belongs to.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task_id: Option<String>,
    /// The conversation (`sessionId` on the wire).
    #[serde(rename = "sessionId", default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    /// The status of the turn.
    #[serde(default)]
    pub status: AgentStatus,
    /// The artifacts produced by the turn.
    #[serde(default)]
    pub artifacts: Vec<AgentArtifact>,
    /// Result-level metadata.
    #[serde(default)]
    pub metadata: Map<String, Value>,
}

/// The stateless request/response transformer.
///
/// Python's `A2ATransformer` only exists as a namespace for class methods; here
/// it is a unit struct with associated functions, so callers write
/// `A2aTransformer::to_request(&inputs)`.
#[derive(Debug, Default, Clone, Copy)]
pub struct A2aTransformer;

impl A2aTransformer {
    /// Builds the `SendMessage` request for a plain openJiuwen payload.
    ///
    /// The payload must be a JSON object. See the [module docs](self) for the
    /// field mapping.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] when `inputs` is not a JSON object.
    pub fn to_request(inputs: &Value) -> Result<SendMessageRequest> {
        let input = inputs.as_object().ok_or_else(|| {
            Error::Validation(format!(
                "request must be an object, got {}",
                json_type_name(inputs)
            ))
        })?;

        let mut request = SendMessageRequest {
            message: Message {
                message_id: uuid::Uuid::new_v4().simple().to_string(),
                role: Role::User,
                ..Message::default()
            },
            ..SendMessageRequest::default()
        };

        request.message.context_id = resolve_session_id(inputs);

        let query = input.get("query").filter(|value| !value.is_null());
        if let Some(query) = query {
            request.message.parts.push(Part::text(stringify(query)));
        }

        for (key, value) in input {
            if value.is_null() || RESERVED_INPUT_KEYS.contains(&key.as_str()) {
                continue;
            }
            request.message.metadata.insert(key.clone(), value.clone());
        }

        Ok(request)
    }

    /// Converts JSON-RPC `params` into an openJiuwen payload.
    ///
    /// Three shapes are accepted, in the Python precedence order:
    ///
    /// 1. a `SendMessageRequest` object (a `message` object member) — the inner
    ///    message is mapped, then the request-level `metadata` is merged in;
    /// 2. a bare `Message` (it has `parts` or `role`);
    /// 3. any other object — copied key by key, with its `metadata` member
    ///    removed and merged back.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] when `params` is not a JSON object or when a
    /// recognised message shape is malformed.
    pub fn from_request(params: &Value) -> Result<Value> {
        let params_object = params.as_object().ok_or_else(|| {
            Error::Validation(format!(
                "a2a request params must be an object, got {}",
                json_type_name(params)
            ))
        })?;

        if let Some(message_value) = params_object
            .get("message")
            .filter(|value| value.is_object())
        {
            let message = decode_message(message_value)?;
            let mut payload = message_to_payload(&message);
            if let Some(metadata) = params_object.get("metadata").and_then(Value::as_object) {
                merge_metadata(&mut payload, metadata);
            }
            return Ok(Value::Object(payload));
        }

        if params_object.contains_key("parts") || params_object.contains_key("role") {
            let message = decode_message(params)?;
            return Ok(Value::Object(message_to_payload(&message)));
        }

        let mut payload = params_object.clone();
        let metadata = payload.remove("metadata");
        if let Some(metadata) = metadata.as_ref().and_then(Value::as_object) {
            merge_metadata(&mut payload, metadata);
        }
        Ok(Value::Object(payload))
    }

    /// Converts a JSON-RPC `result` value into an [`AgentResult`].
    ///
    /// This is the Python `from_a2a_response` classification (see the
    /// [module docs](self) for the precedence table). It never fails and never
    /// panics: an unrecognized shape yields a `completed` result with no
    /// artifacts, and a malformed member degrades to its default value.
    #[must_use]
    pub fn from_response(result: &Value) -> AgentResult {
        let Some(result_object) = result.as_object() else {
            return completed_result();
        };
        let object = unwrap_stream_response(result_object);

        if let Some(artifact) = object.get("artifact").filter(|value| value.is_object()) {
            return AgentResult {
                task_id: id_field(object, "taskId"),
                session_id: id_field(object, "contextId"),
                status: AgentStatus::Working,
                artifacts: vec![from_a2a_artifact(&decode_or_default(artifact))],
                metadata: metadata_field(object),
            };
        }

        // A status update has no task `id`; a task object always does, so a task
        // is never read as a status update.
        if let Some(status) = object
            .get("status")
            .filter(|value| value.is_object())
            .filter(|_| !object.contains_key("id"))
        {
            return AgentResult {
                task_id: id_field(object, "taskId"),
                session_id: id_field(object, "contextId"),
                status: map_task_state(decode_or_default::<TaskStatus>(status).state),
                artifacts: Vec::new(),
                metadata: metadata_field(object),
            };
        }

        if object.contains_key("role") && object.contains_key("parts") {
            let message: Message = decode_or_default(&Value::Object(object.clone()));
            return message_result(&message);
        }

        if object.contains_key("id") && object.get("status").is_some_and(Value::is_object) {
            let task: Task = decode_or_default(&Value::Object(object.clone()));
            return AgentResult {
                task_id: non_empty(&task.id),
                session_id: non_empty(&task.context_id),
                status: map_task_state(task.status.state),
                artifacts: task.artifacts.iter().map(from_a2a_artifact).collect(),
                metadata: task.metadata,
            };
        }

        completed_result()
    }
}

/// Maps an A2A task state onto an openJiuwen status (Python
/// `_A2A_STATUS_TO_OJW_STATUS`).
///
/// `Rejected` maps to [`AgentStatus::Failed`] and `AuthRequired` to
/// [`AgentStatus::InputRequired`]; an unspecified state maps to
/// [`AgentStatus::Unknown`].
#[must_use]
pub fn map_task_state(state: TaskState) -> AgentStatus {
    match state {
        TaskState::Unspecified => AgentStatus::Unknown,
        TaskState::Submitted => AgentStatus::Submitted,
        TaskState::Working => AgentStatus::Working,
        TaskState::Completed => AgentStatus::Completed,
        // Python folds both failure states into `failed`.
        TaskState::Failed | TaskState::Rejected => AgentStatus::Failed,
        TaskState::Canceled => AgentStatus::Canceled,
        // Python needs user interaction for both.
        TaskState::InputRequired | TaskState::AuthRequired => AgentStatus::InputRequired,
    }
}

/// Folds a streamed result into the aggregate (Python `_merge_agent_results`).
///
/// Artifacts append, metadata is overwritten key by key, ids fall back to the
/// base when the update has none, and the status is only taken from the update
/// when it is not [`AgentStatus::Unknown`].
#[must_use]
pub fn merge_results(base: AgentResult, next: AgentResult) -> AgentResult {
    let AgentResult {
        task_id,
        session_id,
        status,
        artifacts,
        metadata,
    } = base;
    let mut merged_metadata = metadata;
    merged_metadata.extend(next.metadata);
    let mut merged_artifacts = artifacts;
    merged_artifacts.extend(next.artifacts);
    AgentResult {
        task_id: next.task_id.or(task_id),
        session_id: next.session_id.or(session_id),
        status: if next.status.is_unknown() {
            status
        } else {
            next.status
        },
        artifacts: merged_artifacts,
        metadata: merged_metadata,
    }
}

/// The session id carried by a request payload.
///
/// `conversation_id` wins over `sessionId`; the value is stringified (strings as
/// they are, other scalars as their JSON text, objects and arrays compactly).
/// Empty strings and `null` count as absent, matching Python's `or` chaining.
/// Returns `None` when `inputs` is not an object.
#[must_use]
pub fn resolve_session_id(inputs: &Value) -> Option<String> {
    let input = inputs.as_object()?;
    ["conversation_id", "sessionId"].iter().find_map(|key| {
        input
            .get(*key)
            .filter(|value| is_present(value))
            .map(stringify)
    })
}

/// Stamps the request's session id onto a result.
///
/// `Some(id)` overwrites [`AgentResult::session_id`]; `None` leaves the result
/// untouched (Python `_with_session_id`).
#[must_use]
pub fn with_session_id(mut result: AgentResult, session_id: Option<&str>) -> AgentResult {
    if let Some(session_id) = session_id {
        result.session_id = Some(session_id.to_owned());
    }
    result
}

/// Converts an openJiuwen part into an A2A part.
///
/// Every unset option stays unset and `data` is carried over as the JSON value
/// it already is.
#[must_use]
pub fn to_a2a_part(part: &AgentPart) -> Part {
    Part {
        text: part.text.clone(),
        raw: part.raw.clone(),
        url: part.url.clone(),
        data: part.data.clone(),
        filename: part.filename.clone(),
        media_type: part.media_type.clone(),
        metadata: part.metadata.clone(),
    }
}

/// Converts an A2A part into an openJiuwen part.
///
/// Empty strings become `None` (Python's `value or None`). `data` is the one
/// documented deviation: Python stringifies it, this side keeps the structured
/// JSON value.
#[must_use]
pub fn from_a2a_part(part: &Part) -> AgentPart {
    AgentPart {
        text: part.text.as_deref().and_then(non_empty),
        raw: part.raw.as_deref().and_then(non_empty),
        url: part.url.as_deref().and_then(non_empty),
        data: part.data.clone(),
        filename: part.filename.as_deref().and_then(non_empty),
        media_type: part.media_type.as_deref().and_then(non_empty),
        metadata: part.metadata.clone(),
    }
}

/// Converts an A2A artifact into an openJiuwen artifact.
///
/// Empty ids, names, and descriptions become `None` (Python's `value or None`);
/// parts are mapped and the metadata is copied.
#[must_use]
pub fn from_a2a_artifact(artifact: &Artifact) -> AgentArtifact {
    AgentArtifact {
        artifact_id: non_empty(&artifact.artifact_id),
        name: artifact.name.as_deref().and_then(non_empty),
        description: artifact.description.as_deref().and_then(non_empty),
        parts: artifact.parts.iter().map(from_a2a_part).collect(),
        metadata: artifact.metadata.clone(),
    }
}

/// Converts an openJiuwen artifact into an A2A artifact.
///
/// `artifact_id` overrides the artifact's own id when given; otherwise the id
/// carried by the artifact is used. A2A extension URIs have no openJiuwen
/// counterpart and start out empty.
#[must_use]
pub fn to_a2a_artifact(artifact: &AgentArtifact, artifact_id: Option<&str>) -> Artifact {
    Artifact {
        artifact_id: artifact_id
            .map(str::to_owned)
            .or_else(|| artifact.artifact_id.clone())
            .unwrap_or_default(),
        name: artifact.name.clone(),
        description: artifact.description.clone(),
        parts: artifact.parts.iter().map(to_a2a_part).collect(),
        metadata: artifact.metadata.clone(),
        extensions: Vec::new(),
    }
}

/// The JSON type name used in validation messages, mirroring Python's
/// `type(value).__name__` for the builtin container types.
fn json_type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

/// Whether a payload value counts as present (Python truthiness for the session
/// id chain): `null` and empty strings are absent.
fn is_present(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::String(raw) => !raw.is_empty(),
        Value::Number(_) | Value::Bool(_) | Value::Array(_) | Value::Object(_) => true,
    }
}

/// Stringifies a JSON value: strings verbatim, everything else as JSON text
/// (numbers and booleans included).
fn stringify(value: &Value) -> String {
    match value {
        Value::String(raw) => raw.clone(),
        other => other.to_string(),
    }
}

/// `Some` for a non-empty string, matching Python's `value or None`.
fn non_empty(value: &str) -> Option<String> {
    if value.is_empty() {
        None
    } else {
        Some(value.to_owned())
    }
}

/// Decodes a wire member tolerantly; a malformed value degrades to its default.
fn decode_or_default<T>(value: &Value) -> T
where
    T: DeserializeOwned + Default,
{
    serde_json::from_value(value.clone()).unwrap_or_default()
}

/// Decodes an A2A `Message`, reporting a validation error for a bad shape.
fn decode_message(value: &Value) -> Result<Message> {
    serde_json::from_value(value.clone())
        .map_err(|error| Error::Validation(format!("invalid a2a message: {error}")))
}

/// Maps a `Message` onto the openJiuwen payload (Python `_message_to_payload`).
///
/// `query` is the first non-empty text part, `task_id`/`sessionId` come from the
/// message ids, and the message metadata is merged last.
fn message_to_payload(message: &Message) -> Map<String, Value> {
    let mut payload = Map::new();

    let text = message
        .parts
        .iter()
        .filter_map(|part| part.text.as_deref())
        .find(|text| !text.is_empty());
    if let Some(text) = text {
        payload.insert(String::from("query"), Value::String(text.to_owned()));
    }
    if let Some(task_id) = message.task_id.as_deref().filter(|id| !id.is_empty()) {
        payload.insert(String::from("task_id"), Value::String(task_id.to_owned()));
    }
    if let Some(context_id) = message.context_id.as_deref().filter(|id| !id.is_empty()) {
        payload.insert(
            String::from("sessionId"),
            Value::String(context_id.to_owned()),
        );
    }

    merge_metadata(&mut payload, &message.metadata);
    payload
}

/// Merges metadata into a payload without overriding existing keys or inserting
/// nulls (Python `_merge_metadata`).
fn merge_metadata(payload: &mut Map<String, Value>, metadata: &Map<String, Value>) {
    for (key, value) in metadata {
        if value.is_null() || payload.contains_key(key) {
            continue;
        }
        payload.insert(key.clone(), value.clone());
    }
}

/// Unwraps the Python protobuf stream-response wrapper keys, if present.
fn unwrap_stream_response(object: &Map<String, Value>) -> &Map<String, Value> {
    for key in [
        "artifact_update",
        "artifactUpdate",
        "status_update",
        "statusUpdate",
    ] {
        if let Some(inner) = object.get(key).and_then(Value::as_object) {
            return inner;
        }
    }
    object
}

/// The non-empty string member `key`, if any.
fn id_field(object: &Map<String, Value>, key: &str) -> Option<String> {
    object.get(key).and_then(Value::as_str).and_then(non_empty)
}

/// The object `metadata` member, or an empty map.
fn metadata_field(object: &Map<String, Value>) -> Map<String, Value> {
    object
        .get("metadata")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default()
}

/// The result of a message event: `completed` with one `message` artifact.
fn message_result(message: &Message) -> AgentResult {
    AgentResult {
        task_id: message.task_id.as_deref().and_then(non_empty),
        session_id: message.context_id.as_deref().and_then(non_empty),
        status: AgentStatus::Completed,
        artifacts: vec![AgentArtifact {
            artifact_id: Some(String::from("message")),
            name: None,
            description: None,
            parts: message.parts.iter().map(from_a2a_part).collect(),
            metadata: Map::new(),
        }],
        metadata: message.metadata.clone(),
    }
}

/// The fallback result for an unrecognized event: `completed`, no artifacts.
fn completed_result() -> AgentResult {
    AgentResult {
        status: AgentStatus::Completed,
        ..AgentResult::default()
    }
}

#[cfg(test)]
mod tests;
