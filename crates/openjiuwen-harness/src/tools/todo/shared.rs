//! Item validation and message helpers shared by more than one todo tool.

use serde_json::Value;

use super::schema::JsonObject;
use crate::schema::{TodoItem, TodoStatus};
use crate::tools::support::compact_json;

/// A fallible operation whose error is a ready-to-report message.
pub(super) type Outcome<T> = std::result::Result<T, String>;

/// The required fields of a new todo item, in Python validation order.
const REQUIRED_ITEM_FIELDS: [&str; 5] = ["content", "activeForm", "description", "status", "id"];
/// Python's rendering of the valid status values in a validation message.
const VALID_STATUSES: &str = r#"["pending", "in_progress", "completed", "cancelled"]"#;

/// Reads a text field of a `todo_create` task, reporting Python's per-index message.
pub(super) fn required_create_field(
    object: Option<&JsonObject>,
    index: usize,
    field: &str,
) -> Outcome<String> {
    match object
        .and_then(|map| map.get(field))
        .and_then(Value::as_str)
    {
        Some(text) if !text.is_empty() => Ok(text.to_owned()),
        _ => Err(format!(
            "Task at index {index} is missing a '{field}' field"
        )),
    }
}

/// Returns the model-supplied id, or a fresh uuid when absent.
pub(super) fn task_id(object: Option<&JsonObject>) -> String {
    let id = object
        .and_then(|map| map.get("id"))
        .map_or_else(String::new, scalar_text);
    if id.is_empty() {
        uuid::Uuid::new_v4().to_string()
    } else {
        id
    }
}

/// Renders a scalar (`id`) the way Python's `str(value)` does.
fn scalar_text(value: &Value) -> String {
    match value {
        Value::String(text) => text.trim().to_owned(),
        Value::Null => String::new(),
        other => compact_json(other),
    }
}

/// Reads an optional string field, treating every non-string value as absent.
pub(super) fn optional_string(value: Option<&Value>) -> Option<String> {
    value.and_then(Value::as_str).map(str::to_owned)
}

/// Reads a required string field of an update/append/insert item.
pub(super) fn required_string(object: &JsonObject, field: &str) -> Outcome<String> {
    object
        .get(field)
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| format!("Invalid input: '{field}' must be a string"))
}

/// Parses a raw status value, accepting only the four Python enum values.
pub(super) fn parse_status(value: &Value) -> Outcome<TodoStatus> {
    let text = value
        .as_str()
        .map_or_else(|| compact_json(value), str::to_owned);
    TodoStatus::parse(&text)
        .ok_or_else(|| format!("Invalid status '{text}'. Valid values: {VALID_STATUSES}"))
}

/// Validates one raw task object used by `append` and the insert actions.
fn validate_todo_item(raw: &Value) -> Outcome<TodoItem> {
    let Some(object) = raw.as_object() else {
        // Python indexes a mapping here; a non-object is reported as a missing required field.
        return Err("Todo data validation error: Missing required field: 'content'".to_owned());
    };
    let missing: Vec<String> = REQUIRED_ITEM_FIELDS
        .iter()
        .filter(|field| !object.contains_key(**field))
        .map(|field| format!("Missing required field: '{field}'"))
        .collect();
    if !missing.is_empty() {
        return Err(format!(
            "Todo data validation error: {}",
            missing.join("; ")
        ));
    }
    let status = object
        .get("status")
        .map_or(Ok(TodoStatus::Pending), parse_status)?;
    Ok(TodoItem {
        id: task_id(Some(object)),
        content: required_string(object, "content")?,
        active_form: required_string(object, "activeForm")?,
        description: required_string(object, "description")?,
        status,
        selected_model_id: optional_string(object.get("selected_model_id")),
        ..TodoItem::default()
    })
}

/// Validates a raw task array used by `append` and the insert actions.
pub(super) fn validate_todo_items(raw_items: &[Value]) -> Outcome<Vec<TodoItem>> {
    raw_items.iter().map(validate_todo_item).collect()
}

/// Renders the `message` of a create/modify payload, falling back to compact JSON.
pub(super) fn render_message(output: &Value) -> String {
    output
        .get("data")
        .and_then(|data| data.get("message"))
        .and_then(Value::as_str)
        .map_or_else(|| compact_json(output), str::to_owned)
}
