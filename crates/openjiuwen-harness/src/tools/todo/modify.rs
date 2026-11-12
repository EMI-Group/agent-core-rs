//! The `todo_modify` tool: the six mutation actions and their validation messages.

use std::collections::HashSet;
use std::sync::Arc;

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use openjiuwen_common::error::Result;
use openjiuwen_foundation::tool::{Tool, ToolCard, ToolContext};

use super::schema::todo_modify_schema;
use super::shared::{
    optional_string, parse_status, render_message, required_string, validate_todo_items, Outcome,
};
use super::store::{session_key, TodoStore};
use crate::schema::{TodoItem, TodoStatus};
use crate::tools::descriptions;
use crate::tools::support::{compact_json, tool_card, ToolPayload};

/// Argument example of the `update` action.
const UPDATE_EXAMPLE: &str = r#"{"todos": [{"id": "task_id", "status": "completed"}]}"#;
/// Argument example of the `append` action.
const APPEND_EXAMPLE: &str = r#"{"todos": [{"id": "task_id", "content": "...", "activeForm": "...", "description": "...", "status": "pending"}]}"#;

/// The action of a `todo_modify` call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum TodoAction {
    /// Patch fields of existing tasks.
    Update,
    /// Remove tasks.
    Delete,
    /// Mark tasks as cancelled.
    Cancel,
    /// Add tasks at the end of the list.
    Append,
    /// Insert tasks after a target task.
    InsertAfter,
    /// Insert tasks before a target task.
    InsertBefore,
}

impl TodoAction {
    /// Every action, in the order Python's `todo_modify` card lists them.
    pub const ALL: [Self; 6] = [
        Self::Update,
        Self::Delete,
        Self::Cancel,
        Self::Append,
        Self::InsertAfter,
        Self::InsertBefore,
    ];

    /// Returns the Python action name, e.g. `insert_after`.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Update => "update",
            Self::Delete => "delete",
            Self::Cancel => "cancel",
            Self::Append => "append",
            Self::InsertAfter => "insert_after",
            Self::InsertBefore => "insert_before",
        }
    }

    /// Parses an exact Python action name; unknown input yields `None`.
    #[must_use]
    pub fn parse(action: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|candidate| candidate.as_str() == action)
    }
}

/// The `todo_data` argument of the insert actions.
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct TodoData {
    /// ID of the task the new items are inserted next to.
    pub target_id: String,
    /// The items to insert.
    pub items: Vec<TodoItem>,
}

/// Arguments of the `todo_modify` tool.
///
/// The invocation reads the raw JSON so every validation message stays identical to Python's;
/// these typed fields drive the model-facing schema.
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct TodoModifyArgs {
    /// One of `update`, `delete`, `cancel`, `append`, `insert_after`, `insert_before`.
    pub action: Option<String>,
    /// Task IDs for `delete` and `cancel`.
    pub ids: Option<Vec<String>>,
    /// Task patches (`update`) or new tasks (`append`).
    pub todos: Option<Vec<Value>>,
    /// Insert payload for `insert_after` and `insert_before`.
    pub todo_data: Option<TodoData>,
}

todo_tool!(
    TodoModifyTool,
    "todo_modify",
    modify,
    render_message,
    todo_modify_schema
);

impl TodoModifyTool {
    /// Validates the arguments, applies the action and persists the result.
    async fn modify(&self, inputs: &Value, ctx: &ToolContext) -> ToolPayload {
        let session = session_key(ctx);
        let _guard = self.store.lock(&session).await;
        let mut todos = match self.store.load(&session).await {
            Ok(todos) => todos,
            Err(error) => return ToolPayload::fail(error.to_string()),
        };
        let message = match apply_action(inputs, &mut todos) {
            Ok(message) => message,
            Err(message) => return ToolPayload::fail(message),
        };
        if let Err(error) = self.store.save(&session, &todos).await {
            return ToolPayload::fail(error.to_string());
        }
        ToolPayload::ok(json!({"message": message}))
    }
}

/// Validates the arguments and applies the requested action to `todos`.
fn apply_action(inputs: &Value, todos: &mut Vec<TodoItem>) -> Outcome<String> {
    let action = parse_action(inputs)?;
    match action {
        TodoAction::Delete => Ok(remove_tasks(todos, &require_ids(inputs, action)?, false)),
        TodoAction::Cancel => Ok(remove_tasks(todos, &require_ids(inputs, action)?, true)),
        TodoAction::Update => {
            update_tasks(todos, &require_todo_array(inputs, action, UPDATE_EXAMPLE)?)
        }
        TodoAction::Append => {
            let raw = require_todo_array(inputs, action, APPEND_EXAMPLE)?;
            append_tasks(todos, validate_todo_items(&raw)?)
        }
        TodoAction::InsertAfter | TodoAction::InsertBefore => {
            let (target_id, items) = require_insert(inputs)?;
            insert_tasks(todos, &target_id, items, action == TodoAction::InsertAfter)
        }
    }
}

/// Deletes (`cancel = false`) or cancels every task whose id is in `ids`.
fn remove_tasks(todos: &mut Vec<TodoItem>, ids: &[String], cancel: bool) -> String {
    let requested: HashSet<&str> = ids.iter().map(String::as_str).collect();
    let mut matched = Vec::new();
    todos.retain_mut(|entry| {
        if !requested.contains(entry.id.as_str()) {
            return true;
        }
        if cancel {
            entry.status = TodoStatus::Cancelled;
        }
        matched.push(entry.id.clone());
        // Cancelled tasks stay in the list; deleted ones are dropped.
        cancel
    });
    let verb = if cancel { "cancelled" } else { "deleted" };
    if matched.is_empty() {
        format!(
            "No tasks {verb}: None of the provided IDs ({}) were found",
            ids.join(", ")
        )
    } else {
        format!(
            "Successfully {verb} {} task(s) (IDs: {})",
            matched.len(),
            matched.join(", ")
        )
    }
}

/// Patches the fields present in each raw item.
fn update_tasks(todos: &mut [TodoItem], items: &[Value]) -> Outcome<String> {
    let mut updated = 0_usize;
    for raw in items {
        let Some(object) = raw.as_object() else {
            return Err("Batch update failed: Missing required field: 'id'".to_owned());
        };
        let id = match object.get("id") {
            Some(Value::String(text)) if !text.is_empty() => text.clone(),
            _ => return Err("Batch update failed: Missing required field: 'id'".to_owned()),
        };
        let Some(entry) = todos.iter_mut().find(|entry| entry.id == id) else {
            return Err(format!(
                "Batch update failed: Task with ID '{id}' not found"
            ));
        };
        if object.contains_key("content") {
            entry.content = required_string(object, "content")?;
        }
        if object.contains_key("activeForm") {
            entry.active_form = required_string(object, "activeForm")?;
        }
        if object.contains_key("description") {
            entry.description = required_string(object, "description")?;
        }
        if let Some(value) = object.get("status") {
            entry.status = parse_status(value)?;
        }
        if let Some(value) = object.get("selected_model_id") {
            entry.selected_model_id = optional_string(Some(value));
        }
        updated += 1;
    }
    ensure_single_in_progress(todos)?;
    Ok(format!("Successfully updated {updated} task(s)"))
}

/// Appends the new items after the current list.
fn append_tasks(todos: &mut Vec<TodoItem>, items: Vec<TodoItem>) -> Outcome<String> {
    let mut existing: HashSet<String> = todos.iter().map(|entry| entry.id.clone()).collect();
    let count = items.len();
    for item in items {
        if !existing.insert(item.id.clone()) {
            return Err(format!(
                "Batch append failed: Task with ID '{}' is duplicated",
                item.id
            ));
        }
        todos.push(item);
    }
    ensure_single_in_progress(todos)?;
    Ok(format!("Successfully appended {count} task(s)"))
}

/// Inserts the new items after (`after = true`) or before the target task.
fn insert_tasks(
    todos: &mut Vec<TodoItem>,
    target_id: &str,
    items: Vec<TodoItem>,
    after: bool,
) -> Outcome<String> {
    let Some(index) = todos.iter().position(|entry| entry.id == target_id) else {
        return Err(format!(
            "Target task with ID '{target_id}' not found in current todo list"
        ));
    };
    let status = todos[index].status;
    let allowed = status == TodoStatus::Pending || (after && status == TodoStatus::InProgress);
    if !allowed {
        return Err(format!(
            "Target task status '{}' doesn't allow insertion.",
            status.as_str()
        ));
    }
    let mut existing: HashSet<String> = todos.iter().map(|entry| entry.id.clone()).collect();
    let count = items.len();
    let mut built = Vec::with_capacity(count);
    for item in items {
        if !existing.insert(item.id.clone()) {
            return Err(format!(
                "Insert failed: Task with ID '{}' already exists",
                item.id
            ));
        }
        built.push(item);
    }
    let at = if after { index + 1 } else { index };
    todos.splice(at..at, built);
    ensure_single_in_progress(todos)?;
    let position = if after { "after" } else { "before" };
    Ok(format!(
        "Successfully inserted {count} task(s) {position} target task, id: '{target_id}'"
    ))
}

/// Rejects a list carrying more than one in-progress task.
fn ensure_single_in_progress(todos: &[TodoItem]) -> Outcome<()> {
    let in_progress = todos
        .iter()
        .filter(|entry| entry.status == TodoStatus::InProgress)
        .count();
    if in_progress > 1 {
        return Err("More than one task is marked as 'in_progress' (only one allowed)".to_owned());
    }
    Ok(())
}

/// Parses the raw `action` field.
fn parse_action(inputs: &Value) -> Outcome<TodoAction> {
    match inputs.get("action") {
        None | Some(Value::Null) => Err("Invalid input: 'action' field is required".to_owned()),
        Some(Value::String(action)) if action.is_empty() => {
            Err("Invalid input: 'action' field is required".to_owned())
        }
        Some(Value::String(action)) => {
            TodoAction::parse(action).ok_or_else(|| format!("Invalid action: {action}"))
        }
        Some(other) => Err(format!("Invalid action: {}", compact_json(other))),
    }
}

/// Requires a non-empty `ids` array of strings.
fn require_ids(inputs: &Value, action: TodoAction) -> Outcome<Vec<String>> {
    let message = || {
        format!(
            "Invalid input for {} action: 'ids' must be a non-empty list of task IDs",
            action.as_str()
        )
    };
    let ids = inputs
        .get("ids")
        .and_then(Value::as_array)
        .filter(|ids| !ids.is_empty())
        .ok_or_else(message)?;
    let mut parsed = Vec::with_capacity(ids.len());
    for id in ids {
        match id {
            Value::String(text) => parsed.push(text.clone()),
            _ => return Err(message()),
        }
    }
    Ok(parsed)
}

/// Requires a non-empty `todos` array of task objects.
fn require_todo_array(inputs: &Value, action: TodoAction, example: &str) -> Outcome<Vec<Value>> {
    match inputs.get("todos") {
        Some(Value::Array(items)) if !items.is_empty() => Ok(items.clone()),
        _ => Err(format!(
            "Invalid input for {}: 'todos' must be a non-empty array of task objects. Example: {example}",
            action.as_str()
        )),
    }
}

/// Validates the `todo_data` object of the insert actions.
fn require_insert(inputs: &Value) -> Outcome<(String, Vec<TodoItem>)> {
    let Some(todo_data) = inputs
        .get("todo_data")
        .filter(|value| value.as_object().is_some_and(|map| !map.is_empty()))
    else {
        return Err(
            "Invalid input for insert action: 'todo_data' must be an object with 'target_id' and 'items'"
                .to_owned(),
        );
    };
    let Some(target_id) = todo_data
        .get("target_id")
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty())
    else {
        return Err("Invalid input: todo_data 'target_id' must be a non-empty string".to_owned());
    };
    let Some(items) = todo_data
        .get("items")
        .and_then(Value::as_array)
        .filter(|items| !items.is_empty())
    else {
        return Err(
            "Invalid input: todo_data 'items' must be a non-empty list of todo objects".to_owned(),
        );
    };
    Ok((target_id.to_owned(), validate_todo_items(items)?))
}
