//! Tool dispatch for both MCP scopes.
//!
//! The operator scope is a fixed catalogue ([`crate::skill::operator_tools`]) whose names this
//! module dispatches over [`crate::skill::TeamClient`] operations; every result is the compact JSON
//! text the Python server produces (`json.dumps`, no spaces). The member scope delegates every
//! non-inbox name to [`crate::skill::TeamClient::invoke_tool`] and returns its text verbatim.
//!
//! Failure text is part of the wire contract: unknown tools answer `Unknown tool: <name>`, client
//! errors answer `Internal error: <error>` (and are logged), a malformed required argument answers
//! `Tool <name> missing argument '<arg>'`. Nothing here panics on user input.
use crate::mcp::READ_INBOX;
use crate::skill::TeamClient;
use openjiuwen_common::error::Error;
use serde::Serialize;
use serde_json::{json, Value};
/// Which task-board write a `claim_task` / `complete_task` call performs.
enum TaskAction {
    /// Claim a pending task.
    Claim,
    /// Complete a claimed task.
    Complete,
}
/// Dispatches one operator-scope tool call and renders its result as plain text.
pub async fn dispatch_operator(client: &dyn TeamClient, name: &str, args: &Value) -> String {
    match name {
        READ_INBOX => read_inbox(client).await,
        "send_message" => send_message(client, name, args).await,
        "create_task" => create_task(client, name, args).await,
        "update_task" => update_task(client, name, args).await,
        "list_tasks" => list_tasks(client, args).await,
        "claimable_tasks" => claimable_tasks(client).await,
        "get_task" => get_task(client, name, args).await,
        "claim_task" => task_action(client, name, args, TaskAction::Claim).await,
        "complete_task" => task_action(client, name, args, TaskAction::Complete).await,
        "list_members" => list_members(client).await,
        _ => format!("Unknown tool: {name}"),
    }
}
/// Dispatches one member-scope tool call and returns the client's rendered text.
///
/// `read_inbox` is operator-only: a member is pushed its messages by the parent process, so the
/// name answers `Unknown tool: read_inbox` here as well (belt and braces with `call_tool`).
pub async fn dispatch_member(client: &dyn TeamClient, name: &str, args: &Value) -> String {
    if name == READ_INBOX {
        return format!("Unknown tool: {name}");
    }
    match client.invoke_tool(name, args.clone()).await {
        Ok(text) => text,
        Err(error) => internal(name, &error),
    }
}
async fn read_inbox(client: &dyn TeamClient) -> String {
    match client.read_inbox().await {
        Ok(text) => render(&text),
        Err(error) => internal(READ_INBOX, &error),
    }
}
async fn send_message(client: &dyn TeamClient, name: &str, args: &Value) -> String {
    let to = match required_str(name, args, "to") {
        Ok(value) => value,
        Err(text) => return text,
    };
    let content = match required_str(name, args, "content") {
        Ok(value) => value,
        Err(text) => return text,
    };
    match client.send_message(to, content).await {
        Ok(outcome) => render(&json!({ "ok": outcome.ok, "message_id": outcome.message_id })),
        Err(error) => internal(name, &error),
    }
}
async fn create_task(client: &dyn TeamClient, name: &str, args: &Value) -> String {
    let title = match required_str(name, args, "title") {
        Ok(value) => value,
        Err(text) => return text,
    };
    let content = match required_str(name, args, "content") {
        Ok(value) => value,
        Err(text) => return text,
    };
    let task_id = optional_str(args, "task_id");
    let dependencies = optional_string_array(args, "dependencies");
    match client
        .create_task(title, content, task_id, dependencies.as_deref())
        .await
    {
        Ok(outcome) => render(&json!({
            "ok": outcome.ok,
            "reason": outcome.reason.unwrap_or_default(),
            "task_id": outcome.task_id,
        })),
        Err(error) => internal(name, &error),
    }
}
async fn update_task(client: &dyn TeamClient, name: &str, args: &Value) -> String {
    let task_id = match required_str(name, args, "task_id") {
        Ok(value) => value,
        Err(text) => return text,
    };
    let title = optional_str(args, "title");
    let content = optional_str(args, "content");
    match client.update_task(task_id, title, content).await {
        Ok(outcome) => render(&json!({
            "ok": outcome.ok,
            "reason": outcome.reason.unwrap_or_default(),
        })),
        Err(error) => internal(name, &error),
    }
}
async fn list_tasks(client: &dyn TeamClient, args: &Value) -> String {
    match client.list_tasks(optional_str(args, "status")).await {
        Ok(tasks) => render(
            &tasks
                .iter()
                .map(|task| {
                    json!({
                        "task_id": task.task_id,
                        "title": task.title,
                        "status": task.status,
                        "assignee": task.assignee,
                    })
                })
                .collect::<Vec<_>>(),
        ),
        Err(error) => internal("list_tasks", &error),
    }
}
async fn claimable_tasks(client: &dyn TeamClient) -> String {
    match client.claimable_tasks().await {
        Ok(tasks) => render(
            &tasks
                .iter()
                .map(|task| json!({ "task_id": task.task_id, "title": task.title }))
                .collect::<Vec<_>>(),
        ),
        Err(error) => internal("claimable_tasks", &error),
    }
}
async fn get_task(client: &dyn TeamClient, name: &str, args: &Value) -> String {
    let task_id = match required_str(name, args, "task_id") {
        Ok(value) => value,
        Err(text) => return text,
    };
    match client.get_task(task_id).await {
        // `None` renders as the bare `null` the Python server emits.
        Ok(detail) => render(&detail),
        Err(error) => internal(name, &error),
    }
}
async fn task_action(
    client: &dyn TeamClient,
    name: &str,
    args: &Value,
    action: TaskAction,
) -> String {
    let task_id = match required_str(name, args, "task_id") {
        Ok(value) => value,
        Err(text) => return text,
    };
    let result = match action {
        TaskAction::Claim => client.claim_task(task_id).await,
        TaskAction::Complete => client.complete_task(task_id).await,
    };
    match result {
        Ok(outcome) => render(&json!({
            "ok": outcome.ok,
            "reason": outcome.reason.unwrap_or_default(),
        })),
        Err(error) => internal(name, &error),
    }
}
async fn list_members(client: &dyn TeamClient) -> String {
    match client.list_members().await {
        Ok(members) => render(
            &members
                .iter()
                .map(|member| {
                    json!({
                        "member_name": member.member_name,
                        "role": member.role,
                        "status": member.status,
                    })
                })
                .collect::<Vec<_>>(),
        ),
        Err(error) => internal("list_members", &error),
    }
}
/// Reads a required string argument, or returns the wire error text for a missing/mistyped one.
fn required_str<'a>(
    name: &str,
    args: &'a Value,
    key: &str,
) -> std::result::Result<&'a str, String> {
    args.get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("Tool {name} missing argument '{key}'"))
}
/// Reads an optional string argument; absent or mistyped values are `None`.
fn optional_str<'a>(args: &'a Value, key: &str) -> Option<&'a str> {
    args.get(key).and_then(Value::as_str)
}
/// Reads an optional string array, dropping non-string elements; absent or mistyped values are
/// `None`.
fn optional_string_array(args: &Value, key: &str) -> Option<Vec<String>> {
    let items = args.get(key)?.as_array()?;
    Some(
        items
            .iter()
            .filter_map(Value::as_str)
            .map(str::to_owned)
            .collect(),
    )
}
/// Serializes a result into the compact JSON text the server returns.
///
/// Serialization of these shapes cannot fail; the fallback keeps the function panic-free anyway.
fn render<T: Serialize + ?Sized>(value: &T) -> String {
    serde_json::to_string(value).unwrap_or_else(|error| format!("Internal error: {error}"))
}
/// Renders (and logs) a client failure as ordinary result text.
fn internal(name: &str, error: &Error) -> String {
    tracing::error!(tool = name, error = %error, "team MCP tool failed");
    format!("Internal error: {error}")
}
