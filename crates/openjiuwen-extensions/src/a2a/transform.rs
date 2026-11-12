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
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use openjiuwen_common::error::ErrorKind;
    use serde_json::json;

    /// A metadata map from a JSON literal.
    fn metadata(entries: Value) -> Map<String, Value> {
        match entries {
            Value::Object(map) => map,
            _ => Map::new(),
        }
    }

    /// A text-only agent part.
    fn text_part(text: &str) -> AgentPart {
        AgentPart {
            text: Some(text.to_owned()),
            ..AgentPart::default()
        }
    }

    #[test]
    fn message_ids_are_fresh_lowercase_hex() {
        // ports `test_a2a_transformer.py::test_to_a2a_request` (message id part).
        let first = A2aTransformer::to_request(&json!({ "query": "a" }))
            .unwrap()
            .message
            .message_id;
        let second = A2aTransformer::to_request(&json!({ "query": "a" }))
            .unwrap()
            .message
            .message_id;
        assert_eq!(first.len(), 32, "uuid4().hex is 32 characters: {first}");
        assert!(
            first
                .chars()
                .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
            "expected lowercase hex, got {first}"
        );
        assert_ne!(first, second, "every request gets a fresh message id");
    }

    #[test]
    fn to_request_builds_the_python_message() {
        // ports `test_a2a_transformer.py::test_to_a2a_request`.
        let request = A2aTransformer::to_request(&json!({
            "query": "what is up",
            "conversation_id": "conv-1",
            "sessionId": "ignored",
            "query_null": null,
            "attempts": 2,
            "extra": { "nested": true },
        }))
        .unwrap();

        assert_eq!(request.message.role, Role::User);
        assert_eq!(request.message.context_id.as_deref(), Some("conv-1"));
        assert_eq!(request.message.parts, vec![Part::text("what is up")]);
        assert_eq!(
            request.message.metadata,
            metadata(json!({ "attempts": 2, "extra": { "nested": true } }))
        );
        assert!(request.tenant.is_none());
        assert!(request.configuration.is_none());
        assert_eq!(
            serde_json::to_value(&request).unwrap(),
            json!({
                "message": {
                    "messageId": request.message.message_id,
                    "contextId": "conv-1",
                    "role": "ROLE_USER",
                    "parts": [{ "text": "what is up", "metadata": {} }],
                    "metadata": { "attempts": 2, "extra": { "nested": true } },
                    "extensions": [],
                    "referenceTaskIds": [],
                },
                "metadata": {},
            })
        );
    }

    #[test]
    fn to_request_stringifies_non_string_queries() {
        assert_eq!(
            A2aTransformer::to_request(&json!({ "query": 42 }))
                .unwrap()
                .message
                .parts,
            vec![Part::text("42")]
        );
        assert_eq!(
            A2aTransformer::to_request(&json!({ "query": { "a": 1 } }))
                .unwrap()
                .message
                .parts,
            vec![Part::text(r#"{"a":1}"#)]
        );
        // A null query produces no parts at all.
        assert_eq!(
            A2aTransformer::to_request(&json!({ "query": null }))
                .unwrap()
                .message
                .parts,
            Vec::new()
        );
        // So does a missing one.
        assert_eq!(
            A2aTransformer::to_request(&json!({}))
                .unwrap()
                .message
                .parts,
            Vec::new()
        );
    }

    #[test]
    fn to_request_omits_the_context_id_when_no_session_is_given() {
        let request = A2aTransformer::to_request(&json!({ "query": "hi" })).unwrap();
        assert!(request.message.context_id.is_none());
        assert_eq!(
            serde_json::to_value(&request).unwrap()["message"]
                .get("contextId")
                .cloned(),
            None
        );
    }

    #[test]
    fn conversation_id_wins_over_session_id() {
        assert_eq!(
            A2aTransformer::to_request(&json!({
                "conversation_id": "conv-1",
                "sessionId": "session-1",
            }))
            .unwrap()
            .message
            .context_id
            .as_deref(),
            Some("conv-1")
        );
        assert_eq!(
            A2aTransformer::to_request(&json!({ "sessionId": "session-1" }))
                .unwrap()
                .message
                .context_id
                .as_deref(),
            Some("session-1")
        );
        // An empty `conversation_id` is falsy in Python, so `sessionId` is used.
        assert_eq!(
            A2aTransformer::to_request(&json!({
                "conversation_id": "",
                "sessionId": "session-1",
            }))
            .unwrap()
            .message
            .context_id
            .as_deref(),
            Some("session-1")
        );
        assert_eq!(
            A2aTransformer::to_request(&json!({ "conversation_id": 7 }))
                .unwrap()
                .message
                .context_id
                .as_deref(),
            Some("7")
        );
        assert!(A2aTransformer::to_request(&json!({ "sessionId": null }))
            .unwrap()
            .message
            .context_id
            .is_none());
    }

    #[test]
    fn to_request_rejects_non_object_inputs() {
        // ports the Python `TypeError` branch of `to_a2a_request`.
        for (value, name) in [
            (Value::Null, "null"),
            (json!([1, 2]), "array"),
            (json!("query"), "string"),
            (json!(1.5), "number"),
            (json!(true), "boolean"),
        ] {
            let error = A2aTransformer::to_request(&value).unwrap_err();
            assert_eq!(error.kind(), ErrorKind::Validation);
            assert_eq!(
                error.to_string(),
                format!("validation: request must be an object, got {name}")
            );
        }
    }

    #[test]
    fn from_request_reads_a_send_message_request() {
        // ports `test_a2a_transformer.py::test_from_a2a_request`.
        let payload = A2aTransformer::from_request(&json!({
            "message": {
                "messageId": "m1",
                "role": "ROLE_USER",
                "contextId": "ctx-1",
                "taskId": "task-1",
                "parts": [{ "text": "" }, { "text": "the query" }],
                "metadata": { "query": "overridden", "task_id": "stale", "extra": 2, "none": null },
            },
            "metadata": { "sessionId": "stale", "request_level": true },
        }))
        .unwrap();

        assert_eq!(
            payload,
            json!({
                "query": "the query",
                "task_id": "task-1",
                "sessionId": "ctx-1",
                "extra": 2,
                "request_level": true,
            })
        );
    }

    #[test]
    fn from_request_reads_a_bare_message() {
        let payload = A2aTransformer::from_request(&json!({
            "role": "ROLE_AGENT",
            "contextId": "ctx-2",
            "parts": [{ "text": "hello" }],
            "metadata": { "k": 1 },
        }))
        .unwrap();
        assert_eq!(
            payload,
            json!({ "query": "hello", "sessionId": "ctx-2", "k": 1 })
        );

        // A message without text still yields its ids and metadata.
        let payload = A2aTransformer::from_request(&json!({
            "parts": [{ "data": { "a": 1 } }],
            "taskId": "t3",
        }))
        .unwrap();
        assert_eq!(payload, json!({ "task_id": "t3" }));
    }

    #[test]
    fn from_request_reads_a_plain_payload() {
        let payload = A2aTransformer::from_request(&json!({
            "query": "raw",
            "metadata": { "query": "ignored", "a": 1, "b": null },
            "other": [1, 2],
        }))
        .unwrap();
        assert_eq!(payload, json!({ "query": "raw", "a": 1, "other": [1, 2] }));
    }

    #[test]
    fn from_request_rejects_non_objects() {
        for (value, name) in [
            (Value::Null, "null"),
            (json!([1]), "array"),
            (json!("SendMessage"), "string"),
            (json!(7), "number"),
            (json!(false), "boolean"),
        ] {
            let error = A2aTransformer::from_request(&value).unwrap_err();
            assert_eq!(error.kind(), ErrorKind::Validation);
            assert_eq!(
                error.to_string(),
                format!("validation: a2a request params must be an object, got {name}")
            );
        }
    }

    #[test]
    fn from_request_reports_a_malformed_message() {
        let error = A2aTransformer::from_request(&json!({
            "message": { "parts": "not-a-list" },
        }))
        .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error
            .to_string()
            .starts_with("validation: invalid a2a message: "));
    }

    #[test]
    fn metadata_merge_never_overrides_and_skips_nulls() {
        // Pins Python `_merge_metadata`.
        let mut payload = Map::new();
        payload.insert(String::from("kept"), json!("original"));
        merge_metadata(
            &mut payload,
            &metadata(json!({ "kept": "overridden", "fresh": 1, "none": null })),
        );
        assert_eq!(payload, metadata(json!({ "kept": "original", "fresh": 1 })));
    }

    #[test]
    fn from_response_reads_a_message() {
        // ports `test_a2a_transformer.py::test_from_a2a_response_message`.
        let result = A2aTransformer::from_response(&json!({
            "messageId": "m1",
            "role": "ROLE_AGENT",
            "taskId": "task-1",
            "contextId": "ctx-1",
            "parts": [{ "text": "hello" }, { "data": { "chart": [1, 2] } }],
            "metadata": { "model": "fake" },
        }));

        assert_eq!(
            result,
            AgentResult {
                task_id: Some(String::from("task-1")),
                session_id: Some(String::from("ctx-1")),
                status: AgentStatus::Completed,
                artifacts: vec![AgentArtifact {
                    artifact_id: Some(String::from("message")),
                    name: None,
                    description: None,
                    parts: vec![
                        text_part("hello"),
                        AgentPart {
                            data: Some(json!({ "chart": [1, 2] })),
                            ..AgentPart::default()
                        },
                    ],
                    metadata: Map::new(),
                }],
                metadata: metadata(json!({ "model": "fake" })),
            }
        );
    }

    #[test]
    fn from_response_reads_a_task() {
        // ports `test_a2a_transformer.py::test_from_a2a_response_task`; a task
        // has both `id` and `status`, so it must not be read as a status update.
        let result = A2aTransformer::from_response(&json!({
            "id": "task-1",
            "contextId": "ctx-1",
            "status": { "state": "TASK_STATE_WORKING" },
            "artifacts": [{
                "artifactId": "a1",
                "name": "summary",
                "parts": [{ "text": "body" }],
            }],
            "history": [],
            "metadata": { "k": "v" },
        }));

        assert_eq!(result.task_id.as_deref(), Some("task-1"));
        assert_eq!(result.session_id.as_deref(), Some("ctx-1"));
        assert_eq!(result.status, AgentStatus::Working);
        assert_eq!(
            result.artifacts,
            vec![AgentArtifact {
                artifact_id: Some(String::from("a1")),
                name: Some(String::from("summary")),
                description: None,
                parts: vec![text_part("body")],
                metadata: Map::new(),
            }]
        );
        assert_eq!(result.metadata, metadata(json!({ "k": "v" })));
    }

    #[test]
    fn from_response_reads_a_status_update() {
        let result = A2aTransformer::from_response(&json!({
            "taskId": "task-1",
            "contextId": "ctx-1",
            "status": { "state": "TASK_STATE_INPUT_REQUIRED" },
            "metadata": { "m": 1 },
        }));
        assert_eq!(result.task_id.as_deref(), Some("task-1"));
        assert_eq!(result.session_id.as_deref(), Some("ctx-1"));
        assert_eq!(result.status, AgentStatus::InputRequired);
        assert_eq!(result.artifacts, Vec::new());
        assert_eq!(result.metadata, metadata(json!({ "m": 1 })));
    }

    #[test]
    fn from_response_reads_an_artifact_update() {
        let result = A2aTransformer::from_response(&json!({
            "taskId": "task-1",
            "contextId": "ctx-1",
            "artifact": { "artifactId": "a1", "parts": [{ "text": "chunk" }] },
            "append": true,
            "lastChunk": false,
        }));
        assert_eq!(result.task_id.as_deref(), Some("task-1"));
        assert_eq!(result.session_id.as_deref(), Some("ctx-1"));
        assert_eq!(result.status, AgentStatus::Working);
        assert_eq!(
            result.artifacts,
            vec![AgentArtifact {
                artifact_id: Some(String::from("a1")),
                name: None,
                description: None,
                parts: vec![text_part("chunk")],
                metadata: Map::new(),
            }]
        );
    }

    #[test]
    fn from_response_unwraps_stream_response_keys() {
        // The Python proto wrapper (`{"artifact_update": {...}}`) is unwrapped
        // before classification; both the snake and camel spellings work.
        let artifact_update = A2aTransformer::from_response(&json!({
            "artifact_update": {
                "taskId": "task-1",
                "artifact": { "artifactId": "a1", "parts": [] },
            },
        }));
        assert_eq!(artifact_update.status, AgentStatus::Working);
        assert_eq!(artifact_update.artifacts.len(), 1);

        let status_update = A2aTransformer::from_response(&json!({
            "statusUpdate": {
                "taskId": "task-2",
                "status": { "state": "TASK_STATE_COMPLETED" },
            },
        }));
        assert_eq!(status_update.task_id.as_deref(), Some("task-2"));
        assert_eq!(status_update.status, AgentStatus::Completed);
    }

    #[test]
    fn from_response_of_an_unknown_event_is_completed() {
        for value in [
            json!({}),
            json!({ "somethingElse": true }),
            json!({ "status": "not-an-object", "id": "task-1" }),
            Value::Null,
            json!("a string"),
            json!([1, 2]),
            json!(42),
        ] {
            assert_eq!(
                A2aTransformer::from_response(&value),
                AgentResult {
                    status: AgentStatus::Completed,
                    ..AgentResult::default()
                },
                "{value} yields the fallback result"
            );
        }
    }

    #[test]
    fn every_task_state_maps_to_its_python_status() {
        for (state, expected) in [
            (TaskState::Unspecified, AgentStatus::Unknown),
            (TaskState::Submitted, AgentStatus::Submitted),
            (TaskState::Working, AgentStatus::Working),
            (TaskState::Completed, AgentStatus::Completed),
            (TaskState::Failed, AgentStatus::Failed),
            (TaskState::Canceled, AgentStatus::Canceled),
            (TaskState::InputRequired, AgentStatus::InputRequired),
            (TaskState::Rejected, AgentStatus::Failed),
            (TaskState::AuthRequired, AgentStatus::InputRequired),
        ] {
            assert_eq!(
                map_task_state(state),
                expected,
                "{state} maps to {expected:?}"
            );
        }
    }

    #[test]
    fn agent_status_serializes_to_the_python_task_status_values() {
        for (status, wire) in [
            (AgentStatus::Submitted, "submitted"),
            (AgentStatus::Working, "working"),
            (AgentStatus::Paused, "paused"),
            (AgentStatus::InputRequired, "input-required"),
            (AgentStatus::Completed, "completed"),
            (AgentStatus::Canceled, "canceled"),
            (AgentStatus::Failed, "failed"),
            (AgentStatus::Waiting, "waiting"),
            (AgentStatus::Unknown, "unknown"),
        ] {
            assert_eq!(status.as_str(), wire);
            assert_eq!(serde_json::to_value(status).unwrap(), json!(wire));
            assert_eq!(
                serde_json::from_value::<AgentStatus>(json!(wire)).unwrap(),
                status
            );
        }
        assert_eq!(AgentStatus::default(), AgentStatus::Unknown);
    }

    #[test]
    fn agent_status_decodes_tolerantly() {
        for wire in [
            "input-required",
            "input_required",
            "inputRequired",
            "INPUT_REQUIRED",
        ] {
            assert_eq!(
                serde_json::from_value::<AgentStatus>(json!(wire)).unwrap(),
                AgentStatus::InputRequired,
                "{wire} decodes to input-required"
            );
        }
        for wire in ["cancelled", "cancelled  ", "CANCELED"] {
            assert_eq!(
                serde_json::from_value::<AgentStatus>(json!(wire)).unwrap(),
                AgentStatus::Canceled,
                "{wire} decodes to canceled"
            );
        }
        for value in [json!("mystery"), json!(""), Value::Null, json!(7)] {
            assert_eq!(
                serde_json::from_value::<AgentStatus>(value.clone()).unwrap(),
                AgentStatus::Unknown,
                "{value} decodes to unknown"
            );
        }
    }

    #[test]
    fn terminal_and_unknown_status_predicates() {
        for status in [
            AgentStatus::Completed,
            AgentStatus::Canceled,
            AgentStatus::Failed,
            AgentStatus::InputRequired,
        ] {
            assert!(status.is_terminal(), "{status:?} is terminal");
        }
        for status in [
            AgentStatus::Submitted,
            AgentStatus::Working,
            AgentStatus::Paused,
            AgentStatus::Waiting,
            AgentStatus::Unknown,
        ] {
            assert!(!status.is_terminal(), "{status:?} is not terminal");
        }
        assert!(AgentStatus::Unknown.is_unknown());
        assert!(!AgentStatus::Working.is_unknown());
    }

    #[test]
    fn agent_result_json_shape_matches_python() {
        let result = AgentResult {
            task_id: Some(String::from("task-1")),
            session_id: Some(String::from("ctx-1")),
            status: AgentStatus::Working,
            artifacts: vec![AgentArtifact {
                artifact_id: Some(String::from("a1")),
                parts: vec![text_part("body")],
                ..AgentArtifact::default()
            }],
            metadata: metadata(json!({ "k": "v" })),
        };
        let serialized = serde_json::to_value(&result).unwrap();
        assert_eq!(
            serialized,
            json!({
                "task_id": "task-1",
                "sessionId": "ctx-1",
                "status": "working",
                "artifacts": [{
                    "artifactId": "a1",
                    "parts": [{ "text": "body", "metadata": {} }],
                    "metadata": {},
                }],
                "metadata": { "k": "v" },
            })
        );
        assert_eq!(
            serde_json::from_value::<AgentResult>(serialized).unwrap(),
            result
        );

        // Unset options are omitted rather than emitted as `null`.
        assert_eq!(
            serde_json::to_value(AgentResult::default()).unwrap(),
            json!({ "status": "unknown", "artifacts": [], "metadata": {} })
        );
    }

    #[test]
    fn merge_results_concatenates_and_overrides() {
        // ports `a2a_client.py::_merge_agent_results`.
        let base = AgentResult {
            task_id: Some(String::from("base-task")),
            session_id: Some(String::from("base-session")),
            status: AgentStatus::Working,
            artifacts: vec![AgentArtifact {
                artifact_id: Some(String::from("a1")),
                ..AgentArtifact::default()
            }],
            metadata: metadata(json!({ "shared": 1, "base": true })),
        };

        // An `unknown` status and missing ids do not override the base.
        let merged = merge_results(
            base.clone(),
            AgentResult {
                status: AgentStatus::Unknown,
                artifacts: vec![AgentArtifact {
                    artifact_id: Some(String::from("a2")),
                    ..AgentArtifact::default()
                }],
                metadata: metadata(json!({ "shared": 2, "next": true })),
                ..AgentResult::default()
            },
        );
        assert_eq!(merged.task_id.as_deref(), Some("base-task"));
        assert_eq!(merged.session_id.as_deref(), Some("base-session"));
        assert_eq!(merged.status, AgentStatus::Working);
        assert_eq!(
            merged
                .artifacts
                .iter()
                .map(|artifact| artifact.artifact_id.clone().unwrap_or_default())
                .collect::<Vec<_>>(),
            vec![String::from("a1"), String::from("a2")]
        );
        assert_eq!(
            merged.metadata,
            metadata(json!({ "shared": 2, "base": true, "next": true }))
        );

        // Known statuses and set ids win.
        let merged = merge_results(
            merged,
            AgentResult {
                task_id: Some(String::from("next-task")),
                session_id: Some(String::from("next-session")),
                status: AgentStatus::Completed,
                ..AgentResult::default()
            },
        );
        assert_eq!(merged.task_id.as_deref(), Some("next-task"));
        assert_eq!(merged.session_id.as_deref(), Some("next-session"));
        assert_eq!(merged.status, AgentStatus::Completed);
        assert_eq!(merged.artifacts.len(), 2, "an empty update adds nothing");
    }

    #[test]
    fn resolve_session_id_follows_python_precedence() {
        assert_eq!(
            resolve_session_id(&json!({ "conversation_id": "c", "sessionId": "s" })),
            Some(String::from("c"))
        );
        assert_eq!(
            resolve_session_id(&json!({ "sessionId": "s" })),
            Some(String::from("s"))
        );
        assert_eq!(
            resolve_session_id(&json!({ "conversation_id": 7 })),
            Some(String::from("7"))
        );
        assert_eq!(
            resolve_session_id(&json!({ "conversation_id": { "a": 1 } })),
            Some(String::from(r#"{"a":1}"#))
        );
        assert_eq!(
            resolve_session_id(&json!({ "conversation_id": "", "sessionId": "s" })),
            Some(String::from("s"))
        );
        assert_eq!(resolve_session_id(&json!({})), None);
        assert_eq!(
            resolve_session_id(&json!({ "conversation_id": null })),
            None
        );
        assert_eq!(resolve_session_id(&json!([1])), None);
        assert_eq!(resolve_session_id(&Value::Null), None);
    }

    #[test]
    fn with_session_id_overrides_only_when_given() {
        let result = with_session_id(AgentResult::default(), Some("ctx-1"));
        assert_eq!(result.session_id.as_deref(), Some("ctx-1"));

        let untouched = with_session_id(result.clone(), None);
        assert_eq!(untouched, result);

        let overwritten = with_session_id(result, Some("ctx-2"));
        assert_eq!(overwritten.session_id.as_deref(), Some("ctx-2"));
    }

    #[test]
    fn parts_round_trip_through_the_wire_types() {
        let agent_part = AgentPart {
            text: Some(String::from("hi")),
            raw: None,
            url: Some(String::from("https://example.test/a.txt")),
            data: None,
            filename: Some(String::from("a.txt")),
            media_type: Some(String::from("text/plain")),
            metadata: metadata(json!({ "k": 1 })),
        };
        let wire = to_a2a_part(&agent_part);
        assert_eq!(
            serde_json::to_value(&wire).unwrap(),
            json!({
                "text": "hi",
                "url": "https://example.test/a.txt",
                "filename": "a.txt",
                "mediaType": "text/plain",
                "metadata": { "k": 1 },
            })
        );
        assert_eq!(from_a2a_part(&wire), agent_part);

        let text_only = text_part("bare");
        assert_eq!(
            serde_json::to_value(to_a2a_part(&text_only)).unwrap(),
            json!({ "text": "bare", "metadata": {} })
        );
        assert_eq!(from_a2a_part(&Part::text("bare")), text_only);
    }

    #[test]
    fn structured_data_survives_the_round_trip() {
        // Documented deviation: Python stringifies `data`, this side keeps JSON.
        let agent_part = AgentPart {
            data: Some(json!({ "chart": [1, 2], "ok": true })),
            ..AgentPart::default()
        };
        let wire = to_a2a_part(&agent_part);
        assert_eq!(wire.data, agent_part.data);
        assert_eq!(from_a2a_part(&wire), agent_part);
    }

    #[test]
    fn empty_wire_strings_decode_to_none() {
        let wire = Part {
            text: Some(String::new()),
            raw: Some(String::new()),
            url: Some(String::new()),
            data: None,
            filename: Some(String::new()),
            media_type: Some(String::new()),
            metadata: Map::new(),
        };
        assert_eq!(from_a2a_part(&wire), AgentPart::default());

        let artifact = Artifact {
            artifact_id: String::new(),
            name: Some(String::new()),
            description: Some(String::new()),
            parts: Vec::new(),
            metadata: Map::new(),
            extensions: Vec::new(),
        };
        assert_eq!(from_a2a_artifact(&artifact), AgentArtifact::default());
    }

    #[test]
    fn artifacts_convert_in_both_directions() {
        let wire = Artifact {
            artifact_id: String::from("a9"),
            name: Some(String::from("summary")),
            description: None,
            parts: vec![Part::text("body")],
            metadata: metadata(json!({ "m": 2 })),
            extensions: vec![String::from("urn:example:ext")],
        };
        let agent_artifact = from_a2a_artifact(&wire);
        assert_eq!(
            agent_artifact,
            AgentArtifact {
                artifact_id: Some(String::from("a9")),
                name: Some(String::from("summary")),
                description: None,
                parts: vec![text_part("body")],
                metadata: metadata(json!({ "m": 2 })),
            }
        );

        assert_eq!(
            to_a2a_artifact(&agent_artifact, None),
            Artifact {
                extensions: Vec::new(),
                ..wire
            }
        );
        assert_eq!(
            to_a2a_artifact(&agent_artifact, Some("generated")).artifact_id,
            "generated"
        );
    }
}
