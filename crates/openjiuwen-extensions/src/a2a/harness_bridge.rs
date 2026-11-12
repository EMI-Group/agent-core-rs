//! Conversion between A2A wire payloads and the harness protocol.
//!
//! The harness protocol (`openjiuwen_harness_protocol`) is the provider-neutral
//! seam between the openJiuwen runtime and an external harness provider; the A2A
//! module speaks JSON-RPC to remote agents. This bridge lets one side drive the
//! other: an inbound A2A [`Message`] becomes a [`HarnessInput`], a terminal
//! [`TurnResult`] becomes an [`AgentResult`] (the model the A2A client and
//! server already exchange, see [`super::transform`]), and an [`AgentResult`]
//! becomes an A2A [`Task`].
//!
//! There is no Python counterpart — `openjiuwen/extensions/a2a` (v0.1.20) ships
//! no harness bridge — so the tables below *are* the contract.
//!
//! # Input mapping (`Message` → `HarnessInput`)
//!
//! | Message parts | `HarnessInput.content` |
//! |---|---|
//! | exactly one part with `text` set and no `data`/`url`/`raw` | the text as a JSON string |
//! | any other non-empty part list | a JSON array with one `serde_json::to_value(part)` per part (camelCase, unset fields omitted) |
//! | no parts | JSON `null` |
//!
//! `HarnessInput.metadata` is a verbatim clone of `Message.metadata`, so
//! provider-specific extensions ride along untouched. `role`, `taskId`, and
//! `contextId` are **not** embedded: the harness receives the caller's context
//! through the harness protocol itself, so callers that need those values put
//! them into the A2A message metadata.
//!
//! # Output mapping (`HarnessInput` → `Message`)
//!
//! The [`harness_input_to_message`] inverse of that table:
//!
//! | `HarnessInput.content` | Message parts |
//! |---|---|
//! | JSON string | one text part |
//! | JSON array | each element decoded as a [`Part`]; elements that fail to decode are skipped |
//! | JSON object with a `parts` array | the same decoding applied to that array |
//! | anything else (object, number, boolean, `null`) | one part carrying the value in `data` |
//!
//! The produced message has `role = ROLE_AGENT`, the caller-supplied
//! `message_id`, and `metadata` cloned from the input. The conversion is
//! infallible, which is why an undecodable array element is skipped rather than
//! surfaced as an error.
//!
//! # Part mapping (`Part` → `ContentBlock`)
//!
//! The first set field wins, in the order `text` > `data` > `url` > `raw`:
//!
//! | Part fields | `ContentBlock.kind` | `ContentBlock.content` |
//! |---|---|---|
//! | `text` | `text` | `{"text": <text>}` |
//! | `data` | `data` | the JSON value |
//! | `url` | `file` | `{"url": <url>, "media_type"?, "filename"?}` |
//! | `raw` | `binary` | `{"raw": <raw>, "media_type"?, "filename"?}` |
//! | none | `empty` | `null` |
//!
//! The block id is the caller-supplied `block_id`, and `ContentBlock.data` (the
//! block's provider extension object) is the part metadata.
//!
//! # Part mapping (`ContentBlock` → `Part`)
//!
//! | `ContentBlock.kind` | Part produced |
//! |---|---|
//! | `text` | `text` from `content["text"]`, or from `content` itself when it is a JSON string; every other content field stays unset |
//! | `data` | `data` = the block content |
//! | `file` | `url` / `media_type` / `filename` read from the `content` object |
//! | `binary` | `raw` / `media_type` / `filename` read from the `content` object |
//! | `empty` | no content fields |
//! | any other kind | `data` = the block content (lossless fallback, so a provider-specific block still reaches the caller) |
//!
//! In every case the part metadata is the block's `data` object.
//! `content_block_to_part(part_to_content_block(part, id))` returns `part`
//! unchanged for text, data, url, and raw parts, with empty or non-empty
//! metadata. Documented lossy spots:
//!
//! 1. a `text` or `data` part's `filename`/`media_type` is not carried by the
//!    block kind (only `file` and `binary` blocks carry them), so it is dropped;
//! 2. a `file`/`binary` block whose `content` is not a JSON object loses it and
//!    produces an empty part;
//! 3. `block_id` is not carried back — A2A parts have no id field.
//!
//! Artifact parts additionally pass through
//! [`super::transform::from_a2a_part`], which normalises empty strings to unset
//! fields (Python's `value or None`).
//!
//! # Result mapping (`TurnResult` → `AgentResult`)
//!
//! | `TurnStatus` | `AgentStatus` |
//! |---|---|
//! | `Completed` | `Completed` |
//! | `Failed` | `Failed` |
//! | `Interrupted` | `InputRequired` (an interrupted turn resumes only after the caller supplies input) |
//!
//! `task_id` and `session_id` stay `None`: a [`TurnResult`] carries neither, so
//! the caller attaches them (for example through [`agent_result_to_task`]).
//! Each [`TurnMessage`] becomes one [`AgentArtifact`]: `artifact_id` is the
//! message id, `name` is the role name (see [`message_role_name`]),
//! `description` is unset, `parts` are the message's blocks converted with
//! [`content_block_to_part`], and `metadata` is the message's `data` object.
//!
//! `AgentResult.metadata` starts as a clone of `TurnResult.provider_data` and
//! then gains the derived keys below. A derived key is inserted only when it is
//! present and non-null **and** `provider_data` does not already define it —
//! provider data always wins.
//!
//! | Metadata key | Source | Shape |
//! |---|---|---|
//! | `stopReason` | `stop_reason` | string |
//! | `finalOutput` | `final_output` | JSON (skipped when `null`) |
//! | `structuredOutput` | `structured_output` | JSON (skipped when `null`) |
//! | `termination` | `termination` | JSON object |
//! | `error` | `error` | JSON object |
//! | `usage` | `usage` | JSON object |
//! | `cost` | `cost` | JSON object |
//! | `durationMs` | `duration_ms` | number |
//! | `startedAt` | `started_at` | number |
//! | `completedAt` | `completed_at` | number |
//!
//! # Task mapping (`AgentResult` → `Task`)
//!
//! [`agent_result_to_task`] honours an explicit task id first, then the result's
//! own `task_id`, and finally a fresh 32-character lowercase hex id (the shape of
//! Python's `uuid4().hex`). The context id is the caller's; the status carries
//! only the mapped state (no message, no timestamp); the history is empty; the
//! task metadata is the result metadata. Each artifact goes through
//! [`super::transform::to_a2a_artifact`], using its own id or a fresh hex id when
//! it has none.
//!
//! [`AgentStatus`] → [`TaskState`] (there is no A2A equivalent for the
//! openJiuwen-only `paused` and `waiting` states, so both report `WORKING`; the
//! inverse of that table lives in [`super::transform::map_task_state`]):
//!
//! | `AgentStatus` | `TaskState` |
//! |---|---|
//! | `Submitted` | `Submitted` |
//! | `Working` | `Working` |
//! | `Paused` | `Working` |
//! | `InputRequired` | `InputRequired` |
//! | `Completed` | `Completed` |
//! | `Canceled` | `Canceled` |
//! | `Failed` | `Failed` |
//! | `Waiting` | `Working` |
//! | `Unknown` | `Unspecified` |
use openjiuwen_harness_protocol::{
    ContentBlock, HarnessInput, JsonObject, MessageRole, TurnMessage, TurnResult, TurnStatus,
};
use serde::Serialize;
use serde_json::{Map, Value};

use super::transform::{self, AgentArtifact, AgentResult, AgentStatus};
use super::types::{Message, Part, Role, Task, TaskState, TaskStatus};

/// Converts an inbound A2A message into harness input.
///
/// A message with exactly one plain text part becomes a string `content`;
/// anything else becomes a JSON array of parts (or `null` when the message has
/// no parts). The message metadata is carried over verbatim. See the
/// [module docs](self) for the full table.
#[must_use]
pub fn message_to_harness_input(message: &Message) -> HarnessInput {
    let content = match message.parts.as_slice() {
        [] => Value::Null,
        [part] if is_plain_text(part) => part.text.clone().map_or(Value::Null, Value::String),
        parts => Value::Array(parts.iter().map(part_to_json).collect()),
    };
    HarnessInput {
        content,
        metadata: to_json_object(&message.metadata),
    }
}

/// Converts harness input into an A2A message authored by the agent.
///
/// The conversion is infallible: an array element that is not a [`Part`] is
/// skipped, and any content that is not a string, a part array, or an object
/// with a `parts` array becomes a single `data` part. See the
/// [module docs](self) for the full table.
#[must_use]
pub fn harness_input_to_message(input: &HarnessInput, message_id: &str) -> Message {
    let parts = match &input.content {
        Value::String(text) => vec![Part::text(text.clone())],
        Value::Array(items) => parts_from_array(items),
        Value::Object(object) => match object.get("parts") {
            Some(Value::Array(items)) => parts_from_array(items),
            _ => vec![data_part(&input.content)],
        },
        other => vec![data_part(other)],
    };
    Message {
        message_id: message_id.to_owned(),
        role: Role::Agent,
        parts,
        metadata: to_json_map(&input.metadata),
        ..Message::default()
    }
}

/// Converts an A2A part into a harness content block.
///
/// The block `kind` follows the `text` > `data` > `url` > `raw` precedence, the
/// part metadata becomes the block's provider `data`, and `block_id` is copied
/// as given. See the [module docs](self) for the table.
#[must_use]
pub fn part_to_content_block(part: &Part, block_id: &str) -> ContentBlock {
    let (kind, content) = if let Some(text) = &part.text {
        ("text", serde_json::json!({ "text": text }))
    } else if let Some(data) = &part.data {
        ("data", data.clone())
    } else if let Some(url) = &part.url {
        ("file", referenced_content("url", url, part))
    } else if let Some(raw) = &part.raw {
        ("binary", referenced_content("raw", raw, part))
    } else {
        ("empty", Value::Null)
    };
    ContentBlock {
        block_id: block_id.to_owned(),
        kind: kind.to_owned(),
        content,
        data: to_json_object(&part.metadata),
    }
}

/// Converts a harness content block into an A2A part.
///
/// The part is keyed on the block `kind`; an unknown kind falls back to a `data`
/// part so no provider-specific content is lost. The block's provider `data`
/// becomes the part metadata. See the [module docs](self) for the table and the
/// documented lossy spots.
#[must_use]
pub fn content_block_to_part(block: &ContentBlock) -> Part {
    let metadata = to_json_map(&block.data);
    match block.kind.as_str() {
        "text" => Part {
            text: text_of(&block.content),
            metadata,
            ..Part::default()
        },
        "file" => Part {
            url: object_field(&block.content, "url"),
            media_type: object_field(&block.content, "media_type"),
            filename: object_field(&block.content, "filename"),
            metadata,
            ..Part::default()
        },
        "binary" => Part {
            raw: object_field(&block.content, "raw"),
            media_type: object_field(&block.content, "media_type"),
            filename: object_field(&block.content, "filename"),
            metadata,
            ..Part::default()
        },
        "empty" => Part {
            metadata,
            ..Part::default()
        },
        // `data` and every provider-specific kind.
        _ => Part {
            data: Some(block.content.clone()),
            metadata,
            ..Part::default()
        },
    }
}

/// Converts one retained turn message into an agent artifact.
///
/// The artifact id is the message id and the name is the role name, so a caller
/// reading the task artifacts can still tell user, assistant, system, and tool
/// output apart. Each block becomes an [`super::transform::AgentPart`] through
/// [`content_block_to_part`] and [`super::transform::from_a2a_part`] (which
/// normalises empty strings to unset fields), and the message's `data` object
/// becomes the artifact metadata.
#[must_use]
pub fn turn_message_to_artifact(message: &TurnMessage) -> AgentArtifact {
    AgentArtifact {
        artifact_id: Some(message.message_id.clone()),
        name: Some(message_role_name(message.role).to_owned()),
        description: None,
        parts: message
            .content
            .iter()
            .map(|block| transform::from_a2a_part(&content_block_to_part(block)))
            .collect(),
        metadata: to_json_map(&message.data),
    }
}

/// Converts a terminal turn result into an openJiuwen agent result.
///
/// `task_id` and `session_id` stay `None` (a [`TurnResult`] carries neither),
/// the status is mapped through the table in the [module docs](self), every
/// message becomes one artifact, and the turn metadata is the provider data plus
/// the derived keys — provider data always wins over a derived key.
#[must_use]
pub fn turn_result_to_agent_result(result: &TurnResult) -> AgentResult {
    let mut metadata = to_json_map(&result.provider_data);
    insert_derived(
        &mut metadata,
        "stopReason",
        result
            .stop_reason
            .as_deref()
            .map_or(Value::Null, |reason| Value::String(reason.to_owned())),
    );
    insert_derived(&mut metadata, "finalOutput", result.final_output.clone());
    insert_derived(
        &mut metadata,
        "structuredOutput",
        result.structured_output.clone(),
    );
    insert_derived(
        &mut metadata,
        "termination",
        result.termination.as_ref().map_or(Value::Null, json_value),
    );
    insert_derived(
        &mut metadata,
        "error",
        result.error.as_ref().map_or(Value::Null, json_value),
    );
    insert_derived(
        &mut metadata,
        "usage",
        result.usage.as_ref().map_or(Value::Null, json_value),
    );
    insert_derived(
        &mut metadata,
        "cost",
        result.cost.as_ref().map_or(Value::Null, json_value),
    );
    insert_derived(
        &mut metadata,
        "durationMs",
        result.duration_ms.map_or(Value::Null, Value::from),
    );
    insert_derived(
        &mut metadata,
        "startedAt",
        result.started_at.map_or(Value::Null, Value::from),
    );
    insert_derived(
        &mut metadata,
        "completedAt",
        result.completed_at.map_or(Value::Null, Value::from),
    );

    AgentResult {
        task_id: None,
        session_id: None,
        status: turn_status_to_agent_status(result.status),
        artifacts: result
            .messages
            .iter()
            .map(turn_message_to_artifact)
            .collect(),
        metadata,
    }
}

/// Converts an openJiuwen agent result into an A2A task.
///
/// The task id is the explicit argument, then the result's own `task_id`, then a
/// fresh 32-character hex id. The status carries the mapped state only, the
/// history is empty, and the artifacts are converted with
/// [`super::transform::to_a2a_artifact`], each with its own id or a fresh hex id
/// when it has none.
#[must_use]
pub fn agent_result_to_task(result: &AgentResult, task_id: Option<&str>, context_id: &str) -> Task {
    Task {
        id: task_id
            .map(ToOwned::to_owned)
            .or_else(|| result.task_id.clone())
            .unwrap_or_else(fresh_hex_id),
        context_id: context_id.to_owned(),
        status: TaskStatus {
            state: agent_status_to_task_state(result.status),
            ..TaskStatus::default()
        },
        artifacts: result
            .artifacts
            .iter()
            .map(|artifact| {
                let artifact_id = artifact.artifact_id.clone().unwrap_or_else(fresh_hex_id);
                transform::to_a2a_artifact(artifact, Some(&artifact_id))
            })
            .collect(),
        history: Vec::new(),
        metadata: result.metadata.clone(),
    }
}

/// Maps an openJiuwen agent status onto the A2A task state.
///
/// `paused` and `waiting` are openJiuwen-only states with no A2A equivalent, so
/// both report `WORKING`; `unknown` reports `UNSPECIFIED`. See the
/// [module docs](self) for the full table.
#[must_use]
pub const fn agent_status_to_task_state(status: AgentStatus) -> TaskState {
    match status {
        AgentStatus::Submitted => TaskState::Submitted,
        AgentStatus::Working | AgentStatus::Paused | AgentStatus::Waiting => TaskState::Working,
        AgentStatus::InputRequired => TaskState::InputRequired,
        AgentStatus::Completed => TaskState::Completed,
        AgentStatus::Canceled => TaskState::Canceled,
        AgentStatus::Failed => TaskState::Failed,
        AgentStatus::Unknown => TaskState::Unspecified,
    }
}

/// Returns the portable name of a harness message role
/// (`user`, `assistant`, `system`, `tool`).
///
/// The name carries the message role across the bridge, since an A2A artifact
/// has no role of its own.
#[must_use]
pub const fn message_role_name(role: MessageRole) -> &'static str {
    match role {
        MessageRole::User => "user",
        MessageRole::Assistant => "assistant",
        MessageRole::System => "system",
        MessageRole::Tool => "tool",
    }
}

/// Maps a terminal turn status onto the openJiuwen agent status.
///
/// An interrupted turn still needs caller input to resume, hence
/// [`AgentStatus::InputRequired`].
const fn turn_status_to_agent_status(status: TurnStatus) -> AgentStatus {
    match status {
        TurnStatus::Completed => AgentStatus::Completed,
        TurnStatus::Failed => AgentStatus::Failed,
        TurnStatus::Interrupted => AgentStatus::InputRequired,
    }
}

/// Whether the part is a lone text part (the only shape that collapses to a
/// string content).
fn is_plain_text(part: &Part) -> bool {
    part.text.is_some() && part.data.is_none() && part.url.is_none() && part.raw.is_none()
}

/// Serializes one part for the array form of the input content.
///
/// A [`Part`] always serializes, so the fallback only keeps this infallible
/// helper panic-free; it degenerates to `null` instead of failing.
fn part_to_json(part: &Part) -> Value {
    serde_json::to_value(part).unwrap_or(Value::Null)
}

/// Decodes the parts of a JSON array, skipping elements that are not parts.
fn parts_from_array(items: &[Value]) -> Vec<Part> {
    items
        .iter()
        .filter_map(|item| serde_json::from_value(item.clone()).ok())
        .collect()
}

/// Wraps a JSON value as a single `data` part.
fn data_part(content: &Value) -> Part {
    Part {
        data: Some(content.clone()),
        ..Part::default()
    }
}

/// Builds the `content` object of a `file`/`binary` block: the reference under
/// `key`, plus the optional media type and filename when the part carries them.
fn referenced_content(key: &str, value: &str, part: &Part) -> Value {
    let mut content = Map::new();
    content.insert(key.to_owned(), Value::String(value.to_owned()));
    if let Some(media_type) = &part.media_type {
        content.insert(
            "media_type".to_owned(),
            Value::String(media_type.to_owned()),
        );
    }
    if let Some(filename) = &part.filename {
        content.insert("filename".to_owned(), Value::String(filename.to_owned()));
    }
    Value::Object(content)
}

/// Reads the text of a `text` block: the `text` member of an object, or the
/// content itself when it is a JSON string.
fn text_of(content: &Value) -> Option<String> {
    match content {
        Value::String(text) => Some(text.clone()),
        _ => object_field(content, "text"),
    }
}

/// Reads a string member of a JSON object (any other shape yields `None`).
fn object_field(content: &Value, key: &str) -> Option<String> {
    content
        .get(key)
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
}

/// Inserts a derived metadata key unless it is null or the provider data already
/// defines it (provider data wins).
fn insert_derived(metadata: &mut Map<String, Value>, key: &str, value: Value) {
    if value.is_null() || metadata.contains_key(key) {
        return;
    }
    metadata.insert(key.to_owned(), value);
}

/// Serializes a derived metadata value; an unrepresentable value becomes `null`
/// and is skipped by [`insert_derived`].
fn json_value<T: Serialize>(value: &T) -> Value {
    serde_json::to_value(value).unwrap_or(Value::Null)
}

/// Converts A2A message metadata into a harness JSON object.
fn to_json_object(metadata: &Map<String, Value>) -> JsonObject {
    metadata
        .iter()
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect()
}

/// Converts a harness JSON object into A2A message metadata.
fn to_json_map(object: &JsonObject) -> Map<String, Value> {
    object
        .iter()
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect()
}

/// A fresh 32-character lowercase hex identifier (the shape of Python's
/// `uuid4().hex`).
fn fresh_hex_id() -> String {
    uuid::Uuid::new_v4().simple().to_string()
}

#[cfg(test)]
mod tests;
