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
mod tests {
    // Tests assert by panicking on failure, which is the intent of a test.
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use serde_json::json;

    use super::super::transform::AgentPart;
    use super::super::types::Artifact;
    use super::*;
    use openjiuwen_harness_protocol::{
        MonetaryAmount, TurnError, TurnTermination, TurnTerminationKind, TurnUsage,
    };

    /// Builds A2A message metadata.
    fn metadata(pairs: &[(&str, Value)]) -> Map<String, Value> {
        pairs
            .iter()
            .map(|(key, value)| ((*key).to_owned(), value.clone()))
            .collect()
    }

    /// Builds a harness JSON object.
    fn json_object(pairs: &[(&str, Value)]) -> JsonObject {
        pairs
            .iter()
            .map(|(key, value)| ((*key).to_owned(), value.clone()))
            .collect()
    }

    /// Builds a content block with no provider data.
    fn block(block_id: &str, kind: &str, content: Value) -> ContentBlock {
        ContentBlock {
            block_id: block_id.to_owned(),
            kind: kind.to_owned(),
            content,
            data: JsonObject::new(),
        }
    }

    /// Builds an otherwise-empty turn result with the given status.
    fn turn_result(status: TurnStatus) -> TurnResult {
        TurnResult {
            status,
            messages: Vec::new(),
            final_output: Value::Null,
            structured_output: Value::Null,
            stop_reason: None,
            termination: None,
            error: None,
            usage: None,
            cost: None,
            started_at: None,
            completed_at: None,
            duration_ms: None,
            provider_data: JsonObject::new(),
        }
    }

    /// Asserts that the value looks like a fresh `uuid4().hex` identifier.
    fn assert_hex_id(id: &str) {
        assert_eq!(id.len(), 32, "{id} has 32 characters");
        assert!(id.chars().all(|c| c.is_ascii_hexdigit()), "{id} is hex");
    }

    #[test]
    fn single_text_message_becomes_string_content() {
        let message = Message {
            message_id: "m1".to_owned(),
            role: Role::User,
            parts: vec![Part::text("hello")],
            metadata: metadata(&[("model", json!("m"))]),
            ..Message::default()
        };
        let input = message_to_harness_input(&message);
        assert_eq!(input.content, json!("hello"));
        assert_eq!(input.metadata.get("model"), Some(&json!("m")));
    }

    #[test]
    fn multi_part_message_becomes_a_part_array() {
        let message = Message {
            parts: vec![
                Part::text("a"),
                Part {
                    data: Some(json!({ "k": 1 })),
                    ..Part::default()
                },
            ],
            ..Message::default()
        };
        let input = message_to_harness_input(&message);
        let items = input.content.as_array().expect("array content");
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].get("text"), Some(&json!("a")));
        assert_eq!(items[1].get("data"), Some(&json!({ "k": 1 })));
        assert!(input.metadata.is_empty());
    }

    #[test]
    fn text_with_another_content_field_stays_a_part_array() {
        let message = Message {
            parts: vec![Part {
                text: Some("a".to_owned()),
                data: Some(json!(1)),
                ..Part::default()
            }],
            ..Message::default()
        };
        let input = message_to_harness_input(&message);
        assert!(input.content.is_array(), "{}", input.content);
    }

    #[test]
    fn message_without_parts_becomes_null_content() {
        let input = message_to_harness_input(&Message::default());
        assert_eq!(input.content, Value::Null);
        assert!(input.metadata.is_empty());
    }

    #[test]
    fn text_input_becomes_one_text_part() {
        let message = harness_input_to_message(&HarnessInput::text("hello"), "m7");
        assert_eq!(message.message_id, "m7");
        assert_eq!(message.role, Role::Agent);
        assert_eq!(message.parts, vec![Part::text("hello")]);
        assert!(message.metadata.is_empty());
        assert!(message.context_id.is_none());
        assert!(message.task_id.is_none());
    }

    #[test]
    fn part_array_input_round_trips_through_the_bridge() {
        let message = Message {
            parts: vec![Part::text("a"), Part::text("b")],
            metadata: metadata(&[("k", json!("v"))]),
            ..Message::default()
        };
        let input = message_to_harness_input(&message);
        let decoded = harness_input_to_message(&input, "m8");
        assert_eq!(decoded.parts, message.parts);
        assert_eq!(decoded.metadata, message.metadata);
        assert_eq!(decoded.role, Role::Agent);
        assert_eq!(decoded.message_id, "m8");
    }

    #[test]
    fn undecodable_array_elements_are_skipped() {
        let input = HarnessInput {
            content: json!([{ "text": "keep" }, "not-a-part", 7, null]),
            metadata: JsonObject::new(),
        };
        let message = harness_input_to_message(&input, "m9");
        assert_eq!(message.parts, vec![Part::text("keep")]);
    }

    #[test]
    fn parts_member_of_an_object_is_decoded() {
        let input = HarnessInput {
            content: json!({
                "parts": [
                    { "text": "x" },
                    { "url": "https://example.test/a.txt", "mediaType": "text/plain" },
                ],
                "ignored": true,
            }),
            metadata: JsonObject::new(),
        };
        let message = harness_input_to_message(&input, "m10");
        assert_eq!(
            message.parts,
            vec![
                Part::text("x"),
                Part {
                    url: Some("https://example.test/a.txt".to_owned()),
                    media_type: Some("text/plain".to_owned()),
                    ..Part::default()
                },
            ]
        );
    }

    #[test]
    fn non_part_content_becomes_a_data_part() {
        for content in [
            json!({ "kind": "custom", "payload": [1, 2] }),
            json!(42),
            json!(true),
            Value::Null,
        ] {
            let input = HarnessInput {
                content: content.clone(),
                metadata: json_object(&[("a", json!(1))]),
            };
            let message = harness_input_to_message(&input, "m11");
            assert_eq!(
                message.parts,
                vec![Part {
                    data: Some(content.clone()),
                    ..Part::default()
                }],
                "{content}"
            );
            assert_eq!(message.metadata.get("a"), Some(&json!(1)));
        }
    }

    #[test]
    fn part_block_kind_precedence_is_text_then_data_then_url_then_raw() {
        let all_fields = Part {
            text: Some("t".to_owned()),
            raw: Some("r".to_owned()),
            url: Some("u".to_owned()),
            data: Some(json!(1)),
            filename: Some("f".to_owned()),
            media_type: Some("m".to_owned()),
            metadata: metadata(&[("k", json!(2))]),
        };
        let block = part_to_content_block(&all_fields, "b1");
        assert_eq!(block.block_id, "b1");
        assert_eq!(block.kind, "text");
        assert_eq!(block.content, json!({ "text": "t" }));
        assert_eq!(block.data.get("k"), Some(&json!(2)));

        let data_part = Part {
            raw: Some("r".to_owned()),
            url: Some("u".to_owned()),
            data: Some(json!({ "answer": 41 })),
            ..Part::default()
        };
        let block = part_to_content_block(&data_part, "b2");
        assert_eq!(block.kind, "data");
        assert_eq!(block.content, json!({ "answer": 41 }));

        let url_part = Part {
            raw: Some("r".to_owned()),
            url: Some("https://example.test/f.bin".to_owned()),
            ..Part::default()
        };
        let block = part_to_content_block(&url_part, "b3");
        assert_eq!(block.kind, "file");
        assert_eq!(
            block.content,
            json!({ "url": "https://example.test/f.bin" })
        );

        let raw_part = Part {
            raw: Some("AAEC".to_owned()),
            ..Part::default()
        };
        let block = part_to_content_block(&raw_part, "b4");
        assert_eq!(block.kind, "binary");
        assert_eq!(block.content, json!({ "raw": "AAEC" }));

        let block = part_to_content_block(&Part::default(), "b5");
        assert_eq!(block.kind, "empty");
        assert_eq!(block.content, Value::Null);
        assert!(block.data.is_empty());
    }

    #[test]
    fn file_and_binary_blocks_carry_media_type_and_filename() {
        let url_part = Part {
            url: Some("https://example.test/f.pdf".to_owned()),
            filename: Some("f.pdf".to_owned()),
            media_type: Some("application/pdf".to_owned()),
            ..Part::default()
        };
        let block = part_to_content_block(&url_part, "b1");
        assert_eq!(
            block.content,
            json!({
                "url": "https://example.test/f.pdf",
                "media_type": "application/pdf",
                "filename": "f.pdf",
            })
        );
        let raw_part = Part {
            raw: Some("AAEC".to_owned()),
            filename: Some("f.pdf".to_owned()),
            media_type: Some("application/pdf".to_owned()),
            ..Part::default()
        };
        let block = part_to_content_block(&raw_part, "b2");
        assert_eq!(
            block.content,
            json!({
                "raw": "AAEC",
                "media_type": "application/pdf",
                "filename": "f.pdf",
            })
        );
    }

    #[test]
    fn parts_round_trip_through_content_blocks() {
        let parts = [
            Part::text("plain"),
            Part {
                text: Some("with metadata".to_owned()),
                metadata: metadata(&[("k", json!("v"))]),
                ..Part::default()
            },
            Part {
                data: Some(json!({ "answer": 42 })),
                metadata: metadata(&[("k", json!(1))]),
                ..Part::default()
            },
            Part {
                url: Some("https://example.test/a.txt".to_owned()),
                media_type: Some("text/plain".to_owned()),
                filename: Some("a.txt".to_owned()),
                metadata: metadata(&[("k", json!(true))]),
                ..Part::default()
            },
            Part {
                raw: Some("AAEC".to_owned()),
                media_type: Some("application/octet-stream".to_owned()),
                metadata: metadata(&[("k", json!([1, 2]))]),
                ..Part::default()
            },
        ];
        for part in parts {
            let block = part_to_content_block(&part, "b1");
            assert_eq!(content_block_to_part(&block), part, "{part:?}");
        }
    }

    #[test]
    fn text_block_reads_a_string_or_a_text_member_only() {
        let string_block = block("b1", "text", json!("plain"));
        assert_eq!(content_block_to_part(&string_block), Part::text("plain"));

        let member_block = block("b2", "text", json!({ "text": "member" }));
        assert_eq!(content_block_to_part(&member_block), Part::text("member"));

        let lossy_block = block("b3", "text", json!({ "text": 5 }));
        assert_eq!(content_block_to_part(&lossy_block), Part::default());
    }

    #[test]
    fn empty_block_and_metadata_map_both_ways() {
        let empty = ContentBlock {
            block_id: "b1".to_owned(),
            kind: "empty".to_owned(),
            content: Value::Null,
            data: json_object(&[("signature", json!("s"))]),
        };
        let part = content_block_to_part(&empty);
        assert!(part.text.is_none());
        assert!(part.raw.is_none());
        assert!(part.url.is_none());
        assert!(part.data.is_none());
        assert_eq!(part.metadata.get("signature"), Some(&json!("s")));
    }

    #[test]
    fn unknown_block_kind_falls_back_to_a_data_part() {
        let unknown = ContentBlock {
            block_id: "b1".to_owned(),
            kind: "reasoning".to_owned(),
            content: json!({ "text": "thinking", "signature": "s" }),
            data: json_object(&[("provider", json!("anthropic"))]),
        };
        let part = content_block_to_part(&unknown);
        assert_eq!(
            part.data,
            Some(json!({ "text": "thinking", "signature": "s" }))
        );
        assert_eq!(part.metadata.get("provider"), Some(&json!("anthropic")));
    }

    #[test]
    fn file_block_with_a_non_object_content_yields_an_empty_part() {
        let file = block("b1", "file", json!("https://example.test/a.txt"));
        assert_eq!(content_block_to_part(&file), Part::default());
    }

    #[test]
    fn turn_message_becomes_a_named_artifact() {
        let message = TurnMessage {
            message_id: "m1".to_owned(),
            role: MessageRole::Tool,
            content: vec![
                block("b1", "text", json!({ "text": "done" })),
                block("b2", "reasoning", json!({ "text": "why" })),
            ],
            data: json_object(&[("name", json!("search"))]),
        };
        let artifact = turn_message_to_artifact(&message);
        assert_eq!(artifact.artifact_id.as_deref(), Some("m1"));
        assert_eq!(artifact.name.as_deref(), Some("tool"));
        assert!(artifact.description.is_none());
        assert_eq!(
            artifact.parts,
            vec![
                AgentPart {
                    text: Some("done".to_owned()),
                    ..AgentPart::default()
                },
                AgentPart {
                    data: Some(json!({ "text": "why" })),
                    ..AgentPart::default()
                },
            ]
        );
        assert_eq!(artifact.metadata.get("name"), Some(&json!("search")));
    }

    #[test]
    fn message_role_names_are_portable() {
        for (role, name) in [
            (MessageRole::User, "user"),
            (MessageRole::Assistant, "assistant"),
            (MessageRole::System, "system"),
            (MessageRole::Tool, "tool"),
        ] {
            assert_eq!(message_role_name(role), name);
        }
    }

    #[test]
    fn completed_turn_becomes_a_completed_result_with_artifacts() {
        let result = TurnResult {
            status: TurnStatus::Completed,
            messages: vec![
                TurnMessage {
                    message_id: "m1".to_owned(),
                    role: MessageRole::User,
                    content: vec![block("b1", "text", json!({ "text": "hi" }))],
                    data: json_object(&[("trace", json!("t1"))]),
                },
                TurnMessage {
                    message_id: "m2".to_owned(),
                    role: MessageRole::Assistant,
                    content: vec![block("b2", "text", json!({ "text": "hello" }))],
                    data: JsonObject::new(),
                },
            ],
            stop_reason: Some("stop".to_owned()),
            final_output: json!("hello"),
            structured_output: json!({ "answer": "hello" }),
            duration_ms: Some(12),
            started_at: Some(1.5),
            completed_at: Some(1.75),
            ..turn_result(TurnStatus::Completed)
        };
        let agent = turn_result_to_agent_result(&result);

        assert_eq!(agent.status, AgentStatus::Completed);
        assert!(agent.task_id.is_none());
        assert!(agent.session_id.is_none());
        assert_eq!(agent.artifacts.len(), 2);
        assert_eq!(agent.artifacts[0].artifact_id.as_deref(), Some("m1"));
        assert_eq!(agent.artifacts[0].name.as_deref(), Some("user"));
        assert_eq!(agent.artifacts[0].metadata.get("trace"), Some(&json!("t1")));
        assert_eq!(agent.artifacts[1].name.as_deref(), Some("assistant"));
        assert_eq!(
            agent.artifacts[1].parts,
            vec![AgentPart {
                text: Some("hello".to_owned()),
                ..AgentPart::default()
            }]
        );

        assert_eq!(agent.metadata.get("stopReason"), Some(&json!("stop")));
        assert_eq!(agent.metadata.get("finalOutput"), Some(&json!("hello")));
        assert_eq!(
            agent.metadata.get("structuredOutput"),
            Some(&json!({ "answer": "hello" }))
        );
        assert_eq!(agent.metadata.get("durationMs"), Some(&json!(12)));
        assert_eq!(agent.metadata.get("startedAt"), Some(&json!(1.5)));
        assert_eq!(agent.metadata.get("completedAt"), Some(&json!(1.75)));
        assert!(!agent.metadata.contains_key("error"));
        assert!(!agent.metadata.contains_key("termination"));
        assert!(!agent.metadata.contains_key("usage"));
        assert!(!agent.metadata.contains_key("cost"));
    }

    #[test]
    fn absent_and_null_turn_fields_are_not_copied() {
        let agent = turn_result_to_agent_result(&turn_result(TurnStatus::Completed));
        assert_eq!(agent.status, AgentStatus::Completed);
        assert_eq!(agent.artifacts, Vec::<AgentArtifact>::new());
        assert!(agent.metadata.is_empty(), "{:?}", agent.metadata);
    }

    #[test]
    fn failed_turn_becomes_failed_with_the_error_object() {
        let result = TurnResult {
            status: TurnStatus::Failed,
            error: Some(TurnError {
                message: "boom".to_owned(),
                code: Some("E_BOOM".to_owned()),
                category: None,
                retryable: Some(true),
                provider_data: JsonObject::new(),
            }),
            ..turn_result(TurnStatus::Failed)
        };
        let agent = turn_result_to_agent_result(&result);
        assert_eq!(agent.status, AgentStatus::Failed);
        assert_eq!(agent.metadata["error"]["message"], "boom");
        assert_eq!(agent.metadata["error"]["code"], "E_BOOM");
        assert_eq!(agent.metadata["error"]["retryable"], json!(true));
        assert!(!agent.metadata.contains_key("termination"));
    }

    #[test]
    fn interrupted_turn_becomes_input_required_with_the_termination() {
        let result = TurnResult {
            status: TurnStatus::Interrupted,
            termination: Some(TurnTermination {
                kind: TurnTerminationKind::Timeout,
                message: Some("too slow".to_owned()),
                code: None,
                provider_data: JsonObject::new(),
            }),
            ..turn_result(TurnStatus::Interrupted)
        };
        let agent = turn_result_to_agent_result(&result);
        assert_eq!(agent.status, AgentStatus::InputRequired);
        assert_eq!(agent.metadata["termination"]["kind"], "timeout");
        assert_eq!(agent.metadata["termination"]["message"], "too slow");
        assert!(!agent.metadata.contains_key("error"));
    }

    #[test]
    fn usage_and_cost_are_copied_as_objects() {
        let result = TurnResult {
            status: TurnStatus::Completed,
            usage: Some(TurnUsage {
                input_tokens: Some(10),
                output_tokens: Some(2),
                ..TurnUsage::default()
            }),
            cost: Some(MonetaryAmount::new(1_500, "usd")),
            ..turn_result(TurnStatus::Completed)
        };
        let agent = turn_result_to_agent_result(&result);
        assert_eq!(agent.metadata["usage"]["input_tokens"], json!(10));
        assert_eq!(agent.metadata["usage"]["output_tokens"], json!(2));
        assert_eq!(agent.metadata["cost"]["micros"], json!(1_500));
        assert_eq!(agent.metadata["cost"]["currency"], "USD");
    }

    #[test]
    fn provider_data_wins_over_derived_metadata() {
        let result = TurnResult {
            status: TurnStatus::Completed,
            stop_reason: Some("stop".to_owned()),
            final_output: json!("derived"),
            duration_ms: Some(7),
            provider_data: json_object(&[
                ("stopReason", json!("provider")),
                ("finalOutput", json!("provider output")),
                ("vendor", json!({ "id": "v1" })),
            ]),
            ..turn_result(TurnStatus::Completed)
        };
        let agent = turn_result_to_agent_result(&result);
        assert_eq!(agent.metadata.get("stopReason"), Some(&json!("provider")));
        assert_eq!(
            agent.metadata.get("finalOutput"),
            Some(&json!("provider output"))
        );
        assert_eq!(agent.metadata.get("vendor"), Some(&json!({ "id": "v1" })));
        assert_eq!(agent.metadata.get("durationMs"), Some(&json!(7)));
    }

    #[test]
    fn agent_status_maps_onto_the_task_state_table() {
        for (status, state) in [
            (AgentStatus::Submitted, TaskState::Submitted),
            (AgentStatus::Working, TaskState::Working),
            (AgentStatus::Paused, TaskState::Working),
            (AgentStatus::InputRequired, TaskState::InputRequired),
            (AgentStatus::Completed, TaskState::Completed),
            (AgentStatus::Canceled, TaskState::Canceled),
            (AgentStatus::Failed, TaskState::Failed),
            (AgentStatus::Waiting, TaskState::Working),
            (AgentStatus::Unknown, TaskState::Unspecified),
        ] {
            assert_eq!(agent_status_to_task_state(status), state, "{status:?}");
        }
    }

    #[test]
    fn task_id_precedence_is_argument_then_result_then_fresh() {
        let with_id = AgentResult {
            task_id: Some("result-task".to_owned()),
            ..AgentResult::default()
        };
        assert_eq!(
            agent_result_to_task(&with_id, Some("explicit"), "ctx").id,
            "explicit"
        );
        assert_eq!(
            agent_result_to_task(&with_id, None, "ctx").id,
            "result-task"
        );

        let without_id = AgentResult::default();
        let task = agent_result_to_task(&without_id, None, "ctx");
        assert_hex_id(&task.id);
    }

    #[test]
    fn task_carries_status_context_artifacts_and_metadata() {
        let result = AgentResult {
            task_id: Some("result-task".to_owned()),
            session_id: Some("s1".to_owned()),
            status: AgentStatus::Working,
            artifacts: vec![
                AgentArtifact {
                    artifact_id: Some("a1".to_owned()),
                    name: Some("summary".to_owned()),
                    description: Some("the summary".to_owned()),
                    parts: vec![AgentPart {
                        text: Some("x".to_owned()),
                        ..AgentPart::default()
                    }],
                    metadata: metadata(&[("k", json!("v"))]),
                },
                AgentArtifact {
                    parts: vec![AgentPart {
                        data: Some(json!({ "answer": 1 })),
                        ..AgentPart::default()
                    }],
                    ..AgentArtifact::default()
                },
            ],
            metadata: metadata(&[("turn", json!(3))]),
        };
        let task = agent_result_to_task(&result, Some("task-1"), "ctx-1");

        assert_eq!(task.id, "task-1");
        assert_eq!(task.context_id, "ctx-1");
        assert_eq!(task.status.state, TaskState::Working);
        assert!(task.status.message.is_none());
        assert!(task.status.timestamp.is_none());
        assert_eq!(task.history, Vec::<Message>::new());
        assert_eq!(task.metadata.get("turn"), Some(&json!(3)));

        assert_eq!(task.artifacts.len(), 2);
        assert_eq!(task.artifacts[0].artifact_id, "a1");
        assert_eq!(task.artifacts[0].name.as_deref(), Some("summary"));
        assert_eq!(task.artifacts[0].parts, vec![Part::text("x")]);
        assert_eq!(task.artifacts[0].metadata.get("k"), Some(&json!("v")));
        assert_hex_id(&task.artifacts[1].artifact_id);
        assert_eq!(
            task.artifacts[1].parts,
            vec![Part {
                data: Some(json!({ "answer": 1 })),
                ..Part::default()
            }]
        );
    }

    #[test]
    fn unknown_status_produces_an_unspecified_task_state() {
        let task = agent_result_to_task(&AgentResult::default(), None, "ctx");
        assert_eq!(task.status.state, TaskState::Unspecified);
        assert_eq!(task.artifacts, Vec::<Artifact>::new());
        assert!(task.metadata.is_empty());
    }
}
