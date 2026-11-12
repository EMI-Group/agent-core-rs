//! The `todo_list` tool: the active items and their rendering.

use std::sync::Arc;

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use openjiuwen_common::error::Result;
use openjiuwen_foundation::tool::{Tool, ToolCard, ToolContext};

use super::schema::todo_list_schema;
use super::store::{session_key, TodoStore};
use crate::schema::TodoStatus;
use crate::tools::descriptions;
use crate::tools::support::{compact_json, tool_card, ToolPayload};

/// Arguments of the `todo_list` tool (no parameters).
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub(super) struct TodoListArgs {}

todo_tool!(
    TodoListTool,
    "todo_list",
    list,
    render_active_list,
    todo_list_schema
);

impl TodoListTool {
    /// Loads the active (neither completed nor cancelled) items.
    async fn list(&self, _inputs: &Value, ctx: &ToolContext) -> ToolPayload {
        let session = session_key(ctx);
        let todos = match self.store.load(&session).await {
            Ok(todos) => todos,
            Err(error) => return ToolPayload::fail(error.to_string()),
        };
        let active: Vec<Value> = todos
            .iter()
            .filter(|item| !matches!(item.status, TodoStatus::Completed | TodoStatus::Cancelled))
            .map(|item| {
                json!({"id": item.id, "content": item.content,
                       "status": item.status.as_str(), "depends_on": item.depends_on})
            })
            .collect();
        ToolPayload::ok(json!({"todos": active}))
    }
}

/// Renders the active `todo_list` entries (`TodoListTool.render_for_llm`).
fn render_active_list(output: &Value) -> String {
    let Some(items) = output
        .get("data")
        .and_then(|data| data.get("todos"))
        .and_then(Value::as_array)
    else {
        return compact_json(output);
    };
    let lines: Vec<String> = items
        .iter()
        .map(|item| {
            let field = |key: &str| item.get(key).and_then(Value::as_str).unwrap_or_default();
            let depends: Vec<&str> = item
                .get("depends_on")
                .and_then(Value::as_array)
                .map_or_else(Vec::new, |values| {
                    values.iter().filter_map(Value::as_str).collect()
                });
            let suffix = if depends.is_empty() {
                String::new()
            } else {
                format!(" (depends on: {})", depends.join(", "))
            };
            format!(
                "- [{}] {}: {}{suffix}",
                field("status"),
                field("id"),
                field("content")
            )
        })
        .collect();
    if lines.is_empty() {
        "No active tasks.".to_owned()
    } else {
        lines.join("\n")
    }
}
