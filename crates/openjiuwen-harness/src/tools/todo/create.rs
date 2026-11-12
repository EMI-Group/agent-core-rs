//! The `todo_create` tool: argument parsing, validation and message rendering.

use std::collections::HashSet;
use std::fmt::Write as _;
use std::sync::Arc;

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use openjiuwen_common::error::Result;
use openjiuwen_foundation::tool::{Tool, ToolCard, ToolContext};

use super::schema::todo_create_schema;
use super::shared::{optional_string, render_message, required_create_field, task_id, Outcome};
use super::store::{session_key, TodoStore};
use crate::schema::{status_icon, TodoItem, TodoStatus};
use crate::tools::descriptions;
use crate::tools::support::{tool_card, ToolPayload};

/// Arguments of the `todo_create` tool.
///
/// The schema uses the typed [`TodoItem`]; the invocation parses the raw objects so a bogus
/// `status` is ignored instead of failing the call, as Python does.
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct TodoCreateArgs {
    /// The tasks to create; the first becomes `in_progress`, the rest `pending`.
    pub todos: Vec<TodoItem>,
}

todo_tool!(
    TodoCreateTool,
    "todo_create",
    create,
    render_message,
    todo_create_schema
);

impl TodoCreateTool {
    /// Validates the raw `todos` array and builds the items to persist.
    fn build_todos(inputs: &Value) -> Outcome<Vec<TodoItem>> {
        let Some(Value::Array(raw_items)) = inputs.get("todos") else {
            return Err("'todos' parameter is required and must be a JSON array".to_owned());
        };
        if raw_items.is_empty() {
            return Err("Task list cannot be empty".to_owned());
        }
        let mut seen: HashSet<String> = HashSet::new();
        let mut todos = Vec::with_capacity(raw_items.len());
        for (index, raw) in raw_items.iter().enumerate() {
            let object = raw.as_object();
            let content = required_create_field(object, index, "content")?;
            let active_form = required_create_field(object, index, "activeForm")?;
            let description = required_create_field(object, index, "description")?;
            let id = task_id(object);
            if !seen.insert(id.clone()) {
                return Err(format!("Duplicate task id '{id}' at index {index}"));
            }
            let status = if index == 0 {
                TodoStatus::InProgress
            } else {
                TodoStatus::Pending
            };
            todos.push(TodoItem {
                id,
                content,
                active_form,
                description,
                status,
                selected_model_id: optional_string(
                    object.and_then(|map| map.get("selected_model_id")),
                ),
                ..TodoItem::default()
            });
        }
        Ok(todos)
    }

    /// Replaces the session todo list with a freshly created one.
    async fn create(&self, inputs: &Value, ctx: &ToolContext) -> ToolPayload {
        let todos = match Self::build_todos(inputs) {
            Ok(todos) => todos,
            Err(message) => return ToolPayload::fail(message),
        };
        let session = session_key(ctx);
        let _guard = self.store.lock(&session).await;
        if let Err(error) = self.store.save(&session, &todos).await {
            return ToolPayload::fail(error.to_string());
        }
        ToolPayload::ok(json!({"message": format_create_result(&todos)}))
    }
}

/// Formats the `todo_create` message (`TodoCreateTool._format_create_result`).
fn format_create_result(todos: &[TodoItem]) -> String {
    let mut rendered = format!("Successfully created {} task(s):\n", todos.len());
    for entry in todos {
        let model_info = entry
            .selected_model_id
            .as_deref()
            .map_or_else(String::new, |model| format!(" (model: {model})"));
        let _ = writeln!(
            rendered,
            "  {} task_id: {} , content: {}{model_info}",
            status_icon(entry.status),
            entry.id,
            entry.content
        );
    }
    let first = todos.first().map_or("", |entry| entry.content.as_str());
    let _ = write!(rendered, "\nNext step: Immediately execute task '{first}'");
    rendered.trim().to_owned()
}
