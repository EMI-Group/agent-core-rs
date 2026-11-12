//! Port of `openjiuwen/harness/tools/todo.py`: the todo store and the four todo tools.
//!
//! [`TodoStore`] persists one JSON array per session at `<dir>/<session>/todo.json`. The tools
//! (`todo_create`, `todo_list`, `todo_get`, `todo_modify`) keep Python's names, argument shapes,
//! message text and payload envelopes. Every create/modify action holds the per-session
//! [`TodoStore::lock`] from load to save.
//!
//! Size note: the store plus the four tools are ~890 lines; the rest is the Python-message
//! parity test table required by the module spec. The pieces cannot be split because the
//! private `mod todo;` wiring in `mod.rs` is owned by the later registry wave, so this file
//! stays one cohesive unit (the same shape as the test-heavy `search.rs`).
//!
//! Deviations from Python: the session id comes from [`ToolContext`] and falls back to `default`
//! (Python raises `Session ID is required`); the file lives under the workspace `todo` node;
//! expected failures (including IO errors and the validation errors Python raises) return a
//! `success=false` payload instead of an `Err`; and a save writes a temporary file that is
//! renamed over `todo.json`, so no reader sees a partial list.

// The module stays private until `ToolRegistry::build` wires the tool set; keeping the
// intended public surface (`pub`) documents the eventual export and avoids a churn diff.
#![allow(unreachable_pub)]

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use tokio::sync::{Mutex, OwnedMutexGuard};

use openjiuwen_common::error::Result;
use openjiuwen_foundation::tool::{Tool, ToolCard, ToolContext};

use super::descriptions;
use super::support::{compact_json, parse_args, render_fields, schema_for, tool_card, ToolPayload};
use crate::schema::{status_icon, TodoItem, TodoStatus};
use crate::workspace::{Workspace, WorkspaceNode};

/// Persisted file name inside each session directory.
const TODO_FILE: &str = "todo.json";
/// Session key used when the invocation has no usable session id.
const DEFAULT_SESSION: &str = "default";
/// The required fields of a new todo item, in Python validation order.
const REQUIRED_ITEM_FIELDS: [&str; 5] = ["content", "activeForm", "description", "status", "id"];
/// Python's rendering of the valid status values in a validation message.
const VALID_STATUSES: &str = r#"["pending", "in_progress", "completed", "cancelled"]"#;
/// Argument example of the `update` action.
const UPDATE_EXAMPLE: &str = r#"{"todos": [{"id": "task_id", "status": "completed"}]}"#;
/// Argument example of the `append` action.
const APPEND_EXAMPLE: &str = r#"{"todos": [{"id": "task_id", "content": "...", "activeForm": "...", "description": "...", "status": "pending"}]}"#;

/// A JSON object as produced by `serde_json`.
type JsonObject = Map<String, Value>;
/// A fallible operation whose error is a ready-to-report message.
type Outcome<T> = std::result::Result<T, String>;

/// Whether `session` is safe to use as a single path component.
fn is_safe_session(session: &str) -> bool {
    !session.is_empty() && session != ".." && !session.contains(['/', '\\', '\0'])
}

/// Maps a raw session id to a safe single path component.
fn normalize_session(session: &str) -> &str {
    if is_safe_session(session) {
        session
    } else {
        DEFAULT_SESSION
    }
}

/// Returns the session key of `ctx`, falling back to `default` for an absent or unsafe id.
#[must_use]
pub fn session_key(ctx: &ToolContext) -> String {
    ctx.session_id.as_deref().map_or_else(
        || DEFAULT_SESSION.to_owned(),
        |session| normalize_session(session).to_owned(),
    )
}

/// Per-session todo persistence with one async lock per session.
///
/// The store owns no process-global state: each registry (or test) builds its own instance,
/// and only the files under [`TodoStore::dir`] are shared.
pub struct TodoStore {
    dir: PathBuf,
    locks: Mutex<HashMap<String, Arc<Mutex<()>>>>,
}

impl TodoStore {
    /// Builds a store rooted at `dir`.
    #[must_use]
    pub fn new(dir: impl Into<PathBuf>) -> Arc<Self> {
        Arc::new(Self {
            dir: dir.into(),
            locks: Mutex::new(HashMap::new()),
        })
    }

    /// Builds a store over the workspace `todo` node, falling back to `<root_path>/todo`.
    #[must_use]
    pub fn for_workspace(workspace: &Workspace) -> Arc<Self> {
        let dir = workspace
            .get_node_path_for(WorkspaceNode::Todo)
            .unwrap_or_else(|| workspace.root_path.join(WorkspaceNode::Todo.as_str()));
        Self::new(dir)
    }

    /// Returns the root directory holding the per-session files.
    #[must_use]
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Returns `<dir>/<session>/todo.json`; an unsafe session id is mapped to `default`, so no
    /// caller can escape [`Self::dir`].
    #[must_use]
    pub fn path_for(&self, session: &str) -> PathBuf {
        self.dir().join(normalize_session(session)).join(TODO_FILE)
    }

    /// Loads the todo items of `session`; a missing file yields an empty list.
    ///
    /// # Errors
    ///
    /// Returns an error when the file exists but cannot be read, is not a JSON array, or holds
    /// an item that does not deserialize as a [`TodoItem`].
    pub async fn load(&self, session: &str) -> Result<Vec<TodoItem>> {
        let path = self.path_for(session);
        let bytes = match tokio::fs::read(&path).await {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => return Err(error.into()),
        };
        let items: Vec<Value> = serde_json::from_slice(&bytes)?;
        items.iter().map(TodoItem::from_dict).collect()
    }

    /// Replaces the persisted todo list of `session` through an atomic rename.
    ///
    /// # Errors
    ///
    /// Returns an error when the session directory cannot be created, the items cannot be
    /// serialized, or the temporary file cannot be written or renamed.
    pub async fn save(&self, session: &str, items: &[TodoItem]) -> Result<()> {
        let path = self.path_for(session);
        let dir = path
            .parent()
            .map_or_else(|| self.dir.clone(), Path::to_path_buf);
        tokio::fs::create_dir_all(&dir).await?;
        let payload = Value::Array(items.iter().map(TodoItem::to_dict).collect());
        let rendered = serde_json::to_string_pretty(&payload)?;
        let temporary = dir.join(format!("{TODO_FILE}.tmp-{}", uuid::Uuid::new_v4()));
        tokio::fs::write(&temporary, rendered.as_bytes()).await?;
        if let Err(error) = tokio::fs::rename(&temporary, &path).await {
            // Windows cannot replace the target in place; remove it and retry once.
            tracing::debug!(%error, path = %path.display(), "retrying todo rename");
            let _ = tokio::fs::remove_file(&path).await;
            if let Err(error) = tokio::fs::rename(&temporary, &path).await {
                let _ = tokio::fs::remove_file(&temporary).await;
                return Err(error.into());
            }
        }
        Ok(())
    }

    /// Acquires the operation lock of `session`; hold the guard for a whole read-modify-write
    /// span. The lock map is never held across an `.await`.
    pub async fn lock(&self, session: &str) -> OwnedMutexGuard<()> {
        let mutex = {
            let mut locks = self.locks.lock().await;
            Arc::clone(
                locks
                    .entry(normalize_session(session).to_owned())
                    .or_insert_with(|| Arc::new(Mutex::new(()))),
            )
        };
        mutex.lock_owned().await
    }
}

/// Generates the per-tool [`Tool`] glue: `$run` performs the action, `$render` renders it.
macro_rules! todo_tool {
    ($tool:ident, $args:ty, $name:literal, $run:ident, $render:path) => {
        /// A model-facing todo tool backed by a [`TodoStore`].
        pub struct $tool {
            store: Arc<TodoStore>,
            card: ToolCard,
        }

        impl $tool {
            /// Builds the tool over `store` with descriptions in `language`.
            #[must_use]
            pub fn new(store: Arc<TodoStore>, language: &str) -> Self {
                Self {
                    card: Self::card_for(language),
                    store,
                }
            }

            /// Builds the model-facing card in `language`; unknown languages fall back to `cn`.
            #[must_use]
            pub fn card_for(language: &str) -> ToolCard {
                tool_card(
                    $name,
                    descriptions::description($name, language),
                    schema_for::<$args>(),
                    true,
                )
            }
        }

        #[async_trait]
        impl Tool for $tool {
            fn card(&self) -> &ToolCard {
                &self.card
            }

            async fn invoke(&self, inputs: Value, ctx: &ToolContext) -> Result<Value> {
                ctx.check_cancelled()?;
                Ok(self.$run(&inputs, ctx).await.to_json())
            }

            fn render_for_llm(&self, output: &Value) -> String {
                $render(output)
            }
        }
    };
}

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
    TodoCreateArgs,
    "todo_create",
    create,
    render_message
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

/// Arguments of the `todo_list` tool (no parameters).
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
struct TodoListArgs {}

todo_tool!(
    TodoListTool,
    TodoListArgs,
    "todo_list",
    list,
    render_active_list
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

/// Arguments of the `todo_get` tool.
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct TodoGetArgs {
    /// ID of the task to fetch.
    pub id: String,
}

todo_tool!(TodoGetTool, TodoGetArgs, "todo_get", get, render_todo);

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
        [
            "update",
            "delete",
            "cancel",
            "append",
            "insert_after",
            "insert_before",
        ]
        .iter()
        .position(|name| *name == action)
        .map(|index| {
            [
                Self::Update,
                Self::Delete,
                Self::Cancel,
                Self::Append,
                Self::InsertAfter,
                Self::InsertBefore,
            ][index]
        })
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
    TodoModifyArgs,
    "todo_modify",
    modify,
    render_message
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

/// Reads a text field of a `todo_create` task, reporting Python's per-index message.
fn required_create_field(
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
fn task_id(object: Option<&JsonObject>) -> String {
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
fn optional_string(value: Option<&Value>) -> Option<String> {
    value.and_then(Value::as_str).map(str::to_owned)
}

/// Reads a required string field of an update/append/insert item.
fn required_string(object: &JsonObject, field: &str) -> Outcome<String> {
    object
        .get(field)
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| format!("Invalid input: '{field}' must be a string"))
}

/// Parses a raw status value, accepting only the four Python enum values.
fn parse_status(value: &Value) -> Outcome<TodoStatus> {
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
fn validate_todo_items(raw_items: &[Value]) -> Outcome<Vec<TodoItem>> {
    raw_items.iter().map(validate_todo_item).collect()
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

/// Renders the `message` of a create/modify payload, falling back to compact JSON.
fn render_message(output: &Value) -> String {
    output
        .get("data")
        .and_then(|data| data.get("message"))
        .and_then(Value::as_str)
        .map_or_else(|| compact_json(output), str::to_owned)
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

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    const SESSION: &str = "s1";
    const INVALID_STATUS: &str =
        "Invalid status 'bogus'. Valid values: [\"pending\", \"in_progress\", \"completed\", \"cancelled\"]";
    const INSERT_SHAPE: &str =
        "Invalid input for insert action: 'todo_data' must be an object with 'target_id' and 'items'";

    fn setup() -> (Arc<TodoStore>, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let store = TodoStore::new(dir.path());
        (store, dir)
    }

    fn ctx(session: &str) -> ToolContext {
        ToolContext::new().with_session(session)
    }

    async fn invoke(tool: &dyn Tool, inputs: Value, session: &str) -> Value {
        tool.invoke(inputs, &ctx(session)).await.unwrap()
    }

    fn create_inputs() -> Value {
        json!({"todos": [
            {"id": "a", "content": "do a", "activeForm": "doing a", "description": "first"},
            {"id": "b", "content": "do b", "activeForm": "doing b", "description": "second"}
        ]})
    }

    fn task(id: &str) -> Value {
        json!({"id": id, "content": format!("do {id}"), "activeForm": format!("doing {id}"),
               "description": "inserted", "status": "pending"})
    }

    /// Creates the two shared tasks and returns the store, a modify tool and the temp dir.
    async fn seeded() -> (Arc<TodoStore>, TodoModifyTool, tempfile::TempDir) {
        let (store, dir) = setup();
        let create = TodoCreateTool::new(Arc::clone(&store), "en");
        let out = invoke(&create, create_inputs(), SESSION).await;
        assert_eq!(out["success"], json!(true), "{out}");
        let modify = TodoModifyTool::new(Arc::clone(&store), "en");
        (store, modify, dir)
    }

    async fn ids(store: &TodoStore) -> Vec<String> {
        store
            .load(SESSION)
            .await
            .unwrap()
            .into_iter()
            .map(|item| item.id)
            .collect()
    }

    /// Runs table-driven `(inputs, expected error)` cases against `tool`.
    async fn assert_failures(tool: &dyn Tool, cases: Vec<(Value, &str)>) {
        for (inputs, expected) in cases {
            let out = invoke(tool, inputs, SESSION).await;
            assert_eq!(out["success"], json!(false), "{out}");
            assert_eq!(out["error"], json!(expected));
        }
    }

    #[test]
    fn cards_expose_python_names_and_parallel_safety() {
        let create = TodoCreateTool::card_for("en");
        assert_eq!(
            (create.name.as_str(), create.parallel_safe),
            ("todo_create", true)
        );
        assert_eq!(create.input_params["required"], json!(["todos"]));
        assert_eq!(
            create.input_params["properties"]["todos"]["type"],
            json!("array")
        );

        let list = TodoListTool::card_for("en");
        assert_eq!(
            (list.name.as_str(), list.parallel_safe),
            ("todo_list", true)
        );
        assert_eq!(
            list.input_params["properties"].as_object().unwrap().len(),
            0
        );

        let get = TodoGetTool::card_for("en");
        assert_eq!((get.name.as_str(), get.parallel_safe), ("todo_get", true));
        assert_eq!(get.input_params["required"], json!(["id"]));

        let modify = TodoModifyTool::card_for("en");
        assert_eq!(
            (modify.name.as_str(), modify.parallel_safe),
            ("todo_modify", true)
        );
        for key in ["action", "ids", "todos", "todo_data"] {
            assert!(
                modify.input_params["properties"].get(key).is_some(),
                "missing {key}"
            );
        }
    }

    #[tokio::test]
    async fn paths_sessions_and_store_round_trip_are_confined() {
        let (store, dir) = setup();
        assert_eq!(
            store.path_for(SESSION),
            dir.path().join("s1").join("todo.json")
        );
        assert_eq!(
            TodoStore::for_workspace(&Workspace::new(dir.path())).dir(),
            dir.path().join("todo")
        );
        assert_eq!(session_key(&ToolContext::new()), "default");
        assert_eq!(session_key(&ctx("normal")), "normal");
        for unsafe_id in ["", "..", "../x", "a/b", "a\\b"] {
            assert_eq!(session_key(&ctx(unsafe_id)), "default", "{unsafe_id:?}");
            assert_eq!(
                store.path_for(unsafe_id),
                dir.path().join("default").join("todo.json"),
                "unexpected path for {unsafe_id:?}"
            );
        }

        assert_eq!(store.load(SESSION).await.unwrap(), Vec::new());
        let item = TodoItem {
            id: "a".to_owned(),
            content: "do a".to_owned(),
            active_form: "doing a".to_owned(),
            description: "first".to_owned(),
            ..TodoItem::default()
        };
        store
            .save(SESSION, std::slice::from_ref(&item))
            .await
            .unwrap();
        assert_eq!(store.load(SESSION).await.unwrap(), vec![item]);
        let raw = std::fs::read_to_string(store.path_for(SESSION)).unwrap();
        assert!(raw.contains('\n'), "payload must be pretty printed");
        assert!(raw.contains("\"activeForm\": \"doing a\""));
    }

    #[tokio::test]
    async fn create_list_get_round_trip() {
        let (store, dir) = setup();
        let create = TodoCreateTool::new(Arc::clone(&store), "en");
        let out = invoke(&create, create_inputs(), SESSION).await;
        assert_eq!(out["success"], json!(true), "{out}");
        let message = out["data"]["message"].as_str().unwrap();
        assert_eq!(
            message,
            "Successfully created 2 task(s):\n  [→] task_id: a , content: do a\n  \
             [ ] task_id: b , content: do b\n\nNext step: Immediately execute task 'do a'"
        );
        assert_eq!(create.render_for_llm(&out), message);

        let raw = std::fs::read_to_string(dir.path().join(SESSION).join("todo.json")).unwrap();
        let items = serde_json::from_str::<Value>(&raw).unwrap();
        assert_eq!(
            (
                &items[0]["activeForm"],
                &items[0]["status"],
                &items[1]["status"]
            ),
            (&json!("doing a"), &json!("in_progress"), &json!("pending"))
        );

        let list = TodoListTool::new(Arc::clone(&store), "en");
        let out = invoke(&list, json!({}), SESSION).await;
        assert_eq!(
            out["data"]["todos"],
            json!([
                {"id": "a", "content": "do a", "status": "in_progress", "depends_on": []},
                {"id": "b", "content": "do b", "status": "pending", "depends_on": []}
            ])
        );
        assert_eq!(
            list.render_for_llm(&out),
            "- [in_progress] a: do a\n- [pending] b: do b"
        );

        let get = TodoGetTool::new(Arc::clone(&store), "en");
        let out = invoke(&get, json!({"id": "b"}), SESSION).await;
        assert_eq!(out["data"]["todo"]["activeForm"], json!("doing b"));
        assert_eq!(
            get.render_for_llm(&out),
            "id: b\ncontent: do b\nactiveForm: doing b\ndescription: second\nstatus: pending\ndepends_on: []"
        );
    }

    #[tokio::test]
    async fn create_validation_messages() {
        let (store, _dir) = setup();
        let create = TodoCreateTool::new(store, "en");
        assert_failures(
            &create,
            vec![
                (
                    json!({}),
                    "'todos' parameter is required and must be a JSON array",
                ),
                (
                    json!({"todos": "nope"}),
                    "'todos' parameter is required and must be a JSON array",
                ),
                (json!({"todos": []}), "Task list cannot be empty"),
                (
                    json!({"todos": [{"id": "a", "activeForm": "x", "description": "d"}]}),
                    "Task at index 0 is missing a 'content' field",
                ),
                (
                    json!({"todos": [{"id": "a", "content": "c", "description": "d"}]}),
                    "Task at index 0 is missing a 'activeForm' field",
                ),
                (
                    json!({"todos": [{"id": "a", "content": "c", "activeForm": "x"}]}),
                    "Task at index 0 is missing a 'description' field",
                ),
                (
                    json!({"todos": [
                        {"id": "a", "content": "c", "activeForm": "x", "description": "d"},
                        {"id": "a", "content": "c2", "activeForm": "x2", "description": "d2"}
                    ]}),
                    "Duplicate task id 'a' at index 1",
                ),
            ],
        )
        .await;

        // A missing id gets a uuid and a bogus status is ignored.
        let inputs = json!({"todos": [{"content": "c", "activeForm": "x", "description": "d",
                                       "status": "bogus"}]});
        let out = invoke(&create, inputs, SESSION).await;
        assert_eq!(out["success"], json!(true), "{out}");
        assert!(out["data"]["message"]
            .as_str()
            .unwrap()
            .contains("[→] task_id: "));
    }

    #[tokio::test]
    async fn list_hides_finished_items_and_get_reports_unknown_ids() {
        let (store, _dir) = setup();
        let list = TodoListTool::new(Arc::clone(&store), "en");
        let out = invoke(&list, json!({}), SESSION).await;
        assert_eq!(out["data"]["todos"], json!([]));
        assert_eq!(list.render_for_llm(&out), "No active tasks.");

        let create = TodoCreateTool::new(Arc::clone(&store), "en");
        invoke(&create, create_inputs(), SESSION).await;
        let modify = TodoModifyTool::new(Arc::clone(&store), "en");
        let inputs = json!({"action": "update", "todos": [
            {"id": "a", "status": "completed"}, {"id": "b", "status": "cancelled"}
        ]});
        assert_eq!(
            invoke(&modify, inputs, SESSION).await["success"],
            json!(true)
        );
        let out = invoke(&list, json!({}), SESSION).await;
        assert_eq!(out["data"]["todos"], json!([]));
        assert_eq!(list.render_for_llm(&out), "No active tasks.");

        let get = TodoGetTool::new(store, "en");
        assert_failures(
            &get,
            vec![
                (json!({}), "Task ID is required"),
                (json!({"id": ""}), "Task ID is required"),
                (json!({"id": "ghost"}), "Task with id 'ghost' not found"),
            ],
        )
        .await;
    }

    #[tokio::test]
    async fn modify_updates_fields_and_validates_the_patch() {
        let (store, modify, _dir) = seeded().await;
        let inputs = json!({"action": "update", "todos": [
            {"id": "a", "status": "completed"},
            {"id": "b", "status": "in_progress", "content": "do b now"}
        ]});
        let out = invoke(&modify, inputs, SESSION).await;
        assert_eq!(
            out["data"]["message"],
            json!("Successfully updated 2 task(s)")
        );
        assert_eq!(
            modify.render_for_llm(&out),
            "Successfully updated 2 task(s)"
        );
        let todos = store.load(SESSION).await.unwrap();
        assert_eq!(
            (todos[0].status, todos[1].status),
            (TodoStatus::Completed, TodoStatus::InProgress)
        );
        assert_eq!(
            (todos[1].content.as_str(), todos[1].active_form.as_str()),
            ("do b now", "doing b")
        );

        assert_failures(
            &modify,
            vec![
                (
                    json!({"action": "update"}),
                    "Invalid input for update: 'todos' must be a non-empty array of task objects. \
                     Example: {\"todos\": [{\"id\": \"task_id\", \"status\": \"completed\"}]}",
                ),
                (
                    json!({"action": "update", "todos": [{"status": "completed"}]}),
                    "Batch update failed: Missing required field: 'id'",
                ),
                (
                    json!({"action": "update", "todos": [{"id": "ghost"}]}),
                    "Batch update failed: Task with ID 'ghost' not found",
                ),
                (
                    json!({"action": "update", "todos": [{"id": "b", "status": "bogus"}]}),
                    INVALID_STATUS,
                ),
                (
                    json!({"action": "update", "todos": [{"id": "a", "status": "in_progress"}]}),
                    "More than one task is marked as 'in_progress' (only one allowed)",
                ),
            ],
        )
        .await;
    }

    #[tokio::test]
    async fn modify_appends_deletes_and_cancels() {
        let (store, modify, _dir) = seeded().await;
        assert_failures(
            &modify,
            vec![
                (
                    json!({"action": "append"}),
                    "Invalid input for append: 'todos' must be a non-empty array of task objects. \
                     Example: {\"todos\": [{\"id\": \"task_id\", \"content\": \"...\", \
                     \"activeForm\": \"...\", \"description\": \"...\", \"status\": \"pending\"}]}",
                ),
                (
                    json!({"action": "append", "todos": [{"id": "c", "content": "do c"}]}),
                    "Todo data validation error: Missing required field: 'activeForm'; \
                     Missing required field: 'description'; Missing required field: 'status'",
                ),
                (
                    json!({"action": "cancel", "ids": []}),
                    "Invalid input for cancel action: 'ids' must be a non-empty list of task IDs",
                ),
            ],
        )
        .await;

        let out = invoke(
            &modify,
            json!({"action": "append", "todos": [task("c")]}),
            SESSION,
        )
        .await;
        assert_eq!(
            out["data"]["message"],
            json!("Successfully appended 1 task(s)")
        );
        assert_eq!(ids(&store).await, vec!["a", "b", "c"]);
        let out = invoke(
            &modify,
            json!({"action": "append", "todos": [task("c")]}),
            SESSION,
        )
        .await;
        assert_eq!(
            out["error"],
            json!("Batch append failed: Task with ID 'c' is duplicated")
        );

        let out = invoke(
            &modify,
            json!({"action": "delete", "ids": ["b", "ghost"]}),
            SESSION,
        )
        .await;
        assert_eq!(
            out["data"]["message"],
            json!("Successfully deleted 1 task(s) (IDs: b)")
        );
        let out = invoke(
            &modify,
            json!({"action": "delete", "ids": ["ghost"]}),
            SESSION,
        )
        .await;
        assert_eq!(
            out["data"]["message"],
            json!("No tasks deleted: None of the provided IDs (ghost) were found")
        );
        let out = invoke(
            &modify,
            json!({"action": "cancel", "ids": ["a", "ghost"]}),
            SESSION,
        )
        .await;
        assert_eq!(
            out["data"]["message"],
            json!("Successfully cancelled 1 task(s) (IDs: a)")
        );
        assert_eq!(
            store.load(SESSION).await.unwrap()[0].status,
            TodoStatus::Cancelled
        );
    }

    #[tokio::test]
    async fn modify_inserts_around_pending_targets() {
        let (store, modify, _dir) = seeded().await;
        let insert = |action: &str, target: &str, id: &str| json!({"action": action, "todo_data": {"target_id": target, "items": [task(id)]}});
        let out = invoke(&modify, insert("insert_after", "a", "c"), SESSION).await;
        assert_eq!(
            out["data"]["message"],
            json!("Successfully inserted 1 task(s) after target task, id: 'a'")
        );
        assert_eq!(ids(&store).await, vec!["a", "c", "b"]);
        assert_failures(
            &modify,
            vec![
                (
                    insert("insert_before", "a", "z"),
                    "Target task status 'in_progress' doesn't allow insertion.",
                ),
                (
                    insert("insert_after", "a", "c"),
                    "Insert failed: Task with ID 'c' already exists",
                ),
                (
                    insert("insert_after", "ghost", "q"),
                    "Target task with ID 'ghost' not found in current todo list",
                ),
            ],
        )
        .await;
        let out = invoke(&modify, insert("insert_before", "b", "z"), SESSION).await;
        assert_eq!(
            out["data"]["message"],
            json!("Successfully inserted 1 task(s) before target task, id: 'b'")
        );
        assert_eq!(ids(&store).await, vec!["a", "c", "z", "b"]);
    }

    #[tokio::test]
    async fn modify_validates_insert_shape_and_actions() {
        let (_store, modify, _dir) = seeded().await;
        let with_data = |target: Value, items: Value| json!({"action": "insert_after", "todo_data": {"target_id": target, "items": items}});
        assert_failures(
            &modify,
            vec![
                (json!({}), "Invalid input: 'action' field is required"),
                (
                    json!({"action": ""}),
                    "Invalid input: 'action' field is required",
                ),
                (json!({"action": "explode"}), "Invalid action: explode"),
                (json!({"action": "insert_after"}), INSERT_SHAPE),
                (
                    json!({"action": "insert_before", "todo_data": {}}),
                    INSERT_SHAPE,
                ),
                (
                    with_data(json!(""), json!([{}])),
                    "Invalid input: todo_data 'target_id' must be a non-empty string",
                ),
                (
                    with_data(json!("a"), json!([])),
                    "Invalid input: todo_data 'items' must be a non-empty list of todo objects",
                ),
                (
                    with_data(json!("a"), json!([{"id": "x"}])),
                    "Todo data validation error: Missing required field: 'content'; \
                     Missing required field: 'activeForm'; Missing required field: 'description'; \
                     Missing required field: 'status'",
                ),
                (
                    with_data(
                        json!("a"),
                        json!([{"id": "x", "content": "c", "activeForm": "x",
                                "description": "d", "status": "bogus"}]),
                    ),
                    INVALID_STATUS,
                ),
            ],
        )
        .await;
    }

    #[tokio::test]
    async fn cancelled_context_returns_cancelled() {
        let (store, _dir) = setup();
        let create = TodoCreateTool::new(store, "en");
        let context = ToolContext::new().with_session(SESSION);
        context.cancellation.cancel();
        let error = create.invoke(create_inputs(), &context).await.unwrap_err();
        assert!(matches!(error, openjiuwen_common::error::Error::Cancelled));
    }
}
