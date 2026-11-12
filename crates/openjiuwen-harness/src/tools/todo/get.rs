//! The `todo_get` tool: one task by id and its rendering.

use std::sync::Arc;

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use openjiuwen_common::error::Result;
use openjiuwen_foundation::tool::{Tool, ToolCard, ToolContext};

use super::schema::todo_get_schema;
use super::store::{session_key, TodoStore};
use crate::tools::descriptions;
use crate::tools::support::{compact_json, parse_args, render_fields, tool_card, ToolPayload};

/// Arguments of the `todo_get` tool.
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct TodoGetArgs {
    /// ID of the task to fetch.
    pub id: String,
}

todo_tool!(TodoGetTool, "todo_get", get, render_todo, todo_get_schema);

impl TodoGetTool {
    /// Looks the requested task up by id.
    async fn get(&self, inputs: &Value, ctx: &ToolContext) -> ToolPayload {
        let Ok(args) = parse_args::<TodoGetArgs>(inputs) else {
            return ToolPayload::fail("Task ID is required");
        };
        if args.id.is_empty() {
            return ToolPayload::fail("Task ID is required");
        }
        let session = session_key(ctx);
        match self.store.load(&session).await {
            Ok(todos) => todos.iter().find(|item| item.id == args.id).map_or_else(
                || ToolPayload::fail(format!("Task with id '{}' not found", args.id)),
                |item| ToolPayload::ok(json!({"todo": item.to_dict()})),
            ),
            Err(error) => ToolPayload::fail(error.to_string()),
        }
    }
}

/// Renders a `todo_get` item through [`render_fields`] in its Python key order.
fn render_todo(output: &Value) -> String {
    let Some(todo) = output
        .get("data")
        .and_then(|data| data.get("todo"))
        .and_then(Value::as_object)
    else {
        return compact_json(output);
    };
    let fields: Vec<(&str, Value)> = todo
        .iter()
        .map(|(key, value)| (key.as_str(), value.clone()))
        .collect();
    render_fields(&fields)
}
