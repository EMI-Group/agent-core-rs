//! Task-plan and todo schema for the `DeepAgent` task loop.
//!
//! Ports `openjiuwen/harness/schema/task.py`: [`TodoStatus`], [`TodoItem`],
//! [`TaskPlan`] and [`ModelUsageRecord`], plus the [`status_icon`] helper.
//! The JSON key set matches Python (`activeForm`, not `active_form`), nulls are
//! preserved on round-trip, and [`TaskPlan::next_task`] implements Python's
//! `get_next_task` dependency scan.

use std::collections::{BTreeMap, HashSet};
use std::fmt;

use openjiuwen_common::error::{Error, Result};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// Lifecycle status of a single todo item.
///
/// The wire values are the Python enum values (`pending`, `in_progress`,
/// `completed`, `cancelled`), so a serialized item round-trips through a
/// persisted Python task plan.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
    Default,
    Serialize,
    Deserialize,
    JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum TodoStatus {
    /// `pending` — not started yet (the default).
    #[default]
    Pending,
    /// `in_progress` — currently being executed.
    InProgress,
    /// `completed` — finished successfully.
    Completed,
    /// `cancelled` — abandoned; unlike `completed`, it does not count as progress.
    Cancelled,
}

impl TodoStatus {
    /// Returns the Python enum value, e.g. `"in_progress"`.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::InProgress => "in_progress",
            Self::Completed => "completed",
            Self::Cancelled => "cancelled",
        }
    }

    /// Parses an exact Python enum value; unknown or differently-cased input yields `None`.
    #[must_use]
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "pending" => Some(Self::Pending),
            "in_progress" => Some(Self::InProgress),
            "completed" => Some(Self::Completed),
            "cancelled" => Some(Self::Cancelled),
            _ => None,
        }
    }
}

/// Returns the checkbox icon Python uses for `status` (`STATUS_ICONS`).
#[must_use]
pub fn status_icon(status: TodoStatus) -> &'static str {
    match status {
        TodoStatus::Pending => "[ ]",
        TodoStatus::InProgress => "[→]",
        TodoStatus::Completed => "[√]",
        TodoStatus::Cancelled => "[×]",
    }
}

/// One item of a [`TaskPlan`] (Python `TodoItem`).
///
/// Deserialization is lenient: every field is optional on the wire (missing
/// fields fall back to [`Self::default`] values) and unknown keys are ignored,
/// so legacy or partially written Python payloads still load. Serialization uses
/// Python's key names and order, including the camel-case `activeForm` key.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(default)]
pub struct TodoItem {
    /// Unique task identifier — a short semantic string chosen by the model
    /// (e.g. `translate_doc`); a random UUID is used as the fallback.
    pub id: String,
    /// Task summary description.
    pub content: String,
    /// Present-tense form of `content` (e.g. "Translating document"); `activeForm` on the wire.
    #[serde(rename = "activeForm")]
    pub active_form: String,
    /// Detailed task content.
    pub description: String,
    /// Current lifecycle status.
    pub status: TodoStatus,
    /// IDs of prerequisite tasks; every one of them must be done before this task runs.
    pub depends_on: Vec<String>,
    /// Brief summary written after completion or cancellation.
    pub result_summary: Option<String>,
    /// Arbitrary task metadata, persisted as-is.
    pub meta_data: Option<BTreeMap<String, Value>>,
    /// Optional model ID to use when executing this task.
    pub selected_model_id: Option<String>,
}

impl Default for TodoItem {
    /// Builds an empty pending item with a fresh random (v4) identifier.
    fn default() -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            content: String::new(),
            active_form: String::new(),
            description: String::new(),
            status: TodoStatus::Pending,
            depends_on: Vec::new(),
            result_summary: None,
            meta_data: None,
            selected_model_id: None,
        }
    }
}

impl TodoItem {
    /// Converts to Python's JSON-serializable dict (`TodoItem.to_dict`).
    ///
    /// The key set and order are exactly Python's: `id`, `content`, `activeForm`,
    /// `description`, `status`, `depends_on`, `result_summary`, `meta_data`,
    /// `selected_model_id`; absent summaries and metadata stay as `null`.
    #[must_use]
    pub fn to_dict(&self) -> Value {
        let mut map = Map::new();
        map.insert("id".to_owned(), Value::String(self.id.clone()));
        map.insert("content".to_owned(), Value::String(self.content.clone()));
        map.insert(
            "activeForm".to_owned(),
            Value::String(self.active_form.clone()),
        );
        map.insert(
            "description".to_owned(),
            Value::String(self.description.clone()),
        );
        map.insert(
            "status".to_owned(),
            Value::String(self.status.as_str().to_owned()),
        );
        map.insert(
            "depends_on".to_owned(),
            Value::Array(self.depends_on.iter().cloned().map(Value::String).collect()),
        );
        map.insert(
            "result_summary".to_owned(),
            optional_string(self.result_summary.as_deref()),
        );
        map.insert(
            "meta_data".to_owned(),
            metadata_value(self.meta_data.as_ref()),
        );
        map.insert(
            "selected_model_id".to_owned(),
            optional_string(self.selected_model_id.as_deref()),
        );
        Value::Object(map)
    }

    /// Builds an item from a persisted dict (`TodoItem.from_dict`).
    ///
    /// Missing fields take their [`Self::default`] values and unknown keys are
    /// ignored; an invalid `status` or a non-object value is an error.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`] when `value` is not a JSON object that matches
    /// this type (for example an unknown `status`).
    pub fn from_dict(value: &Value) -> Result<Self> {
        serde_json::from_value(value.clone())
            .map_err(|error| Error::validation(format!("invalid TodoItem: {error}")))
    }
}

/// Structured task plan for the outer task loop (Python `TaskPlan`).
///
/// Serialization uses Python's key names (`goal`, `tasks`, `current_task_id`),
/// and deserialization tolerates missing keys and unknown fields.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(default)]
pub struct TaskPlan {
    /// High-level objective the plan pursues.
    pub goal: String,
    /// Ordered list of todo items.
    pub tasks: Vec<TodoItem>,
    /// ID of the task currently being executed, if any.
    pub current_task_id: Option<String>,
}

impl TaskPlan {
    /// Returns the task with `task_id`, or `None` (Python `get_task`).
    #[must_use]
    pub fn get_task(&self, task_id: &str) -> Option<&TodoItem> {
        self.tasks.iter().find(|task| task.id == task_id)
    }

    /// Returns the first `pending` task whose dependencies are all done
    /// (Python `get_next_task`).
    ///
    /// "Done" means `completed` *or* `cancelled`; tasks are scanned in list order;
    /// an empty `depends_on` is immediately eligible; a dependency that is missing
    /// from the plan or still pending blocks the task permanently for this scan.
    #[must_use]
    pub fn next_task(&self) -> Option<&TodoItem> {
        let done_ids: HashSet<&str> = self
            .tasks
            .iter()
            .filter(|task| matches!(task.status, TodoStatus::Completed | TodoStatus::Cancelled))
            .map(|task| task.id.as_str())
            .collect();
        self.tasks.iter().find(|task| {
            task.status == TodoStatus::Pending
                && task
                    .depends_on
                    .iter()
                    .all(|dependency| done_ids.contains(dependency.as_str()))
        })
    }

    /// Appends a task item.
    pub fn add_task(&mut self, task: TodoItem) {
        self.tasks.push(task);
    }

    /// Marks `task_id` as in progress and records it as the current task.
    ///
    /// Unknown IDs are ignored.
    pub fn mark_in_progress(&mut self, task_id: &str) {
        if let Some(task) = self.tasks.iter_mut().find(|task| task.id == task_id) {
            task.status = TodoStatus::InProgress;
            self.current_task_id = Some(task_id.to_owned());
        }
    }

    /// Marks `task_id` as completed and stores `summary` as its result summary.
    ///
    /// Clears `current_task_id` when it points at `task_id`; unknown IDs are ignored.
    pub fn mark_completed(&mut self, task_id: &str, summary: &str) {
        if let Some(task) = self.tasks.iter_mut().find(|task| task.id == task_id) {
            task.status = TodoStatus::Completed;
            task.result_summary = Some(summary.to_owned());
            if self.current_task_id.as_deref() == Some(task_id) {
                self.current_task_id = None;
            }
        }
    }

    /// Marks `task_id` as cancelled and stores `reason` as its result summary.
    ///
    /// Clears `current_task_id` when it points at `task_id`; unknown IDs are ignored.
    pub fn mark_cancelled(&mut self, task_id: &str, reason: &str) {
        if let Some(task) = self.tasks.iter_mut().find(|task| task.id == task_id) {
            task.status = TodoStatus::Cancelled;
            task.result_summary = Some(reason.to_owned());
            if self.current_task_id.as_deref() == Some(task_id) {
                self.current_task_id = None;
            }
        }
    }

    /// Returns `"{completed}/{total} completed"`; cancelled tasks are not counted
    /// as completed (Python `get_progress_summary`).
    #[must_use]
    pub fn progress_summary(&self) -> String {
        let completed = self
            .tasks
            .iter()
            .filter(|task| task.status == TodoStatus::Completed)
            .count();
        format!("{completed}/{} completed", self.tasks.len())
    }

    /// Renders the plan as a markdown checklist (Python `to_markdown`).
    ///
    /// The output starts with `## Goal: {goal}`, a blank line, then one
    /// `- [x] {content}` line per task in list order, with the em-dash summary
    /// suffix when a non-empty `result_summary` exists.
    ///
    /// Python parity: the lines are joined, so the string does not end with a
    /// newline once tasks exist; an empty plan still yields exactly
    /// `## Goal: {goal}\n` because of the trailing blank line element.
    #[must_use]
    pub fn to_markdown(&self) -> String {
        let mut lines = vec![format!("## Goal: {}", self.goal), String::new()];
        for task in &self.tasks {
            let suffix = match task.result_summary.as_deref() {
                Some(summary) if !summary.is_empty() => format!(" — {summary}"),
                _ => String::new(),
            };
            lines.push(format!(
                "- [{}] {}{suffix}",
                markdown_mark(task.status),
                task.content
            ));
        }
        lines.join("\n")
    }

    /// Converts to a JSON-friendly value for session persistence
    /// (Python `TaskPlan.to_dict`).
    #[must_use]
    pub fn to_dict(&self) -> Value {
        let mut map = Map::new();
        map.insert("goal".to_owned(), Value::String(self.goal.clone()));
        map.insert(
            "tasks".to_owned(),
            Value::Array(self.tasks.iter().map(TodoItem::to_dict).collect()),
        );
        map.insert(
            "current_task_id".to_owned(),
            optional_string(self.current_task_id.as_deref()),
        );
        Value::Object(map)
    }

    /// Builds a plan from a persisted dict (Python `TaskPlan.from_dict`).
    ///
    /// `null` and `{}` both yield [`Self::default`]; unknown keys are ignored.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`] when `value` is not a JSON object that matches
    /// this type (for example a non-array `tasks`).
    pub fn from_dict(value: &Value) -> Result<Self> {
        let empty_object = value.as_object().is_some_and(Map::is_empty);
        if value.is_null() || empty_object {
            return Ok(Self::default());
        }
        serde_json::from_value(value.clone())
            .map_err(|error| Error::validation(format!("invalid TaskPlan: {error}")))
    }
}

/// Accumulated token usage for one `model_id` within a single invoke
/// (Python `ModelUsageRecord`).
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(default)]
pub struct ModelUsageRecord {
    /// The model selection ID (a user-defined key).
    pub model_id: String,
    /// Total input tokens consumed.
    pub input_tokens: u64,
    /// Total output tokens consumed.
    pub output_tokens: u64,
}

impl ModelUsageRecord {
    /// Accumulates one round of token counts.
    pub fn add(&mut self, input_tokens: u64, output_tokens: u64) {
        self.input_tokens = self.input_tokens.saturating_add(input_tokens);
        self.output_tokens = self.output_tokens.saturating_add(output_tokens);
    }
}

impl fmt::Display for ModelUsageRecord {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "model_id={} input={} output={}",
            self.model_id, self.input_tokens, self.output_tokens
        )
    }
}

/// The checkbox mark Python's `to_markdown` uses for `status` (without brackets).
fn markdown_mark(status: TodoStatus) -> &'static str {
    match status {
        TodoStatus::Pending => " ",
        TodoStatus::InProgress => ">",
        TodoStatus::Completed => "√",
        TodoStatus::Cancelled => "×",
    }
}

/// Serializes an optional string, keeping `null` for absent values.
fn optional_string(value: Option<&str>) -> Value {
    value.map_or(Value::Null, |text| Value::String(text.to_owned()))
}

/// Serializes optional metadata, keeping `null` for absent values.
fn metadata_value(meta_data: Option<&BTreeMap<String, Value>>) -> Value {
    meta_data.map_or(Value::Null, |entries| {
        Value::Object(
            entries
                .iter()
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect(),
        )
    })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use serde_json::json;

    use super::*;

    fn item(id: &str, status: TodoStatus) -> TodoItem {
        TodoItem {
            id: id.to_owned(),
            content: format!("task {id}"),
            active_form: format!("doing {id}"),
            status,
            ..TodoItem::default()
        }
    }

    fn blocked_item(id: &str, depends_on: &[&str]) -> TodoItem {
        TodoItem {
            depends_on: depends_on.iter().map(|dep| (*dep).to_owned()).collect(),
            ..item(id, TodoStatus::Pending)
        }
    }

    #[test]
    fn status_wire_values_and_parse() {
        let cases = [
            (TodoStatus::Pending, "pending"),
            (TodoStatus::InProgress, "in_progress"),
            (TodoStatus::Completed, "completed"),
            (TodoStatus::Cancelled, "cancelled"),
        ];
        for (status, wire) in cases {
            assert_eq!(status.as_str(), wire);
            assert_eq!(TodoStatus::parse(wire), Some(status));
            assert_eq!(serde_json::to_value(status).unwrap(), json!(wire));
            let round_trip: TodoStatus = serde_json::from_value(json!(wire)).unwrap();
            assert_eq!(round_trip, status);
        }
        assert_eq!(TodoStatus::default(), TodoStatus::Pending);
        assert_eq!(TodoStatus::parse("Pending"), None);
        assert_eq!(TodoStatus::parse(""), None);
        assert_eq!(TodoStatus::parse("done"), None);
    }

    #[test]
    fn status_icon_per_status() {
        assert_eq!(status_icon(TodoStatus::Pending), "[ ]");
        assert_eq!(status_icon(TodoStatus::InProgress), "[→]");
        assert_eq!(status_icon(TodoStatus::Completed), "[√]");
        assert_eq!(status_icon(TodoStatus::Cancelled), "[×]");
    }

    #[test]
    fn todo_item_default_has_uuid_and_empty_fields() {
        let first = TodoItem::default();
        let second = TodoItem::default();
        assert_ne!(first.id, second.id);
        assert_eq!(first.id.len(), 36);
        assert_eq!(first.content, "");
        assert_eq!(first.active_form, "");
        assert_eq!(first.description, "");
        assert_eq!(first.status, TodoStatus::Pending);
        assert_eq!(first.depends_on.len(), 0);
        assert_eq!(first.result_summary, None);
        assert_eq!(first.meta_data, None);
        assert_eq!(first.selected_model_id, None);
    }

    #[test]
    fn active_form_is_camel_case_on_the_wire_both_directions() {
        let source = json!({
            "id": "translate_doc",
            "content": "Translate document",
            "activeForm": "Translating document",
            "description": "translate the doc",
            "status": "in_progress",
            "depends_on": ["read_doc"],
            "result_summary": null,
            "meta_data": null,
            "selected_model_id": "fast"
        });
        let parsed = TodoItem::from_dict(&source).unwrap();
        assert_eq!(parsed.active_form, "Translating document");
        assert_eq!(parsed.status, TodoStatus::InProgress);
        assert_eq!(parsed.depends_on, vec!["read_doc".to_owned()]);
        assert_eq!(parsed.selected_model_id.as_deref(), Some("fast"));
        assert_eq!(serde_json::to_value(&parsed).unwrap(), source);
        assert!(!parsed
            .to_dict()
            .as_object()
            .unwrap()
            .contains_key("active_form"));
    }

    #[test]
    fn to_dict_uses_python_key_set_and_order() {
        let parsed = TodoItem {
            id: "a".to_owned(),
            content: "content".to_owned(),
            active_form: "active".to_owned(),
            description: "description".to_owned(),
            status: TodoStatus::Cancelled,
            depends_on: vec!["b".to_owned()],
            result_summary: Some("not needed".to_owned()),
            meta_data: Some(BTreeMap::from([("k".to_owned(), json!(1))])),
            selected_model_id: Some("m1".to_owned()),
        };
        let dict = parsed.to_dict();
        let keys: Vec<&str> = dict
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(
            keys,
            vec![
                "id",
                "content",
                "activeForm",
                "description",
                "status",
                "depends_on",
                "result_summary",
                "meta_data",
                "selected_model_id",
            ]
        );
        assert_eq!(dict["status"], json!("cancelled"));
        assert_eq!(dict["depends_on"], json!(["b"]));
        assert_eq!(dict["meta_data"], json!({"k": 1}));
        assert_eq!(parsed.to_dict(), serde_json::to_value(&parsed).unwrap());
    }

    #[test]
    fn to_dict_preserves_nulls() {
        let dict = TodoItem::default().to_dict();
        assert_eq!(dict["result_summary"], Value::Null);
        assert_eq!(dict["meta_data"], Value::Null);
        assert_eq!(dict["selected_model_id"], Value::Null);
        assert_eq!(dict.as_object().unwrap().len(), 9);
    }

    #[test]
    fn from_dict_round_trips_and_rejects_junk() {
        let original = TodoItem {
            result_summary: Some("done".to_owned()),
            ..item("a", TodoStatus::Completed)
        };
        let restored = TodoItem::from_dict(&original.to_dict()).unwrap();
        assert_eq!(restored, original);

        assert!(TodoItem::from_dict(&json!("not an object")).is_err());
        assert!(TodoItem::from_dict(&json!({"status": "unknown"})).is_err());
        assert!(TodoItem::from_dict(&json!({"depends_on": "not-an-array"})).is_err());
    }

    #[test]
    fn from_dict_tolerates_missing_and_unknown_keys() {
        let parsed =
            TodoItem::from_dict(&json!({"content": "only content", "extra": true})).unwrap();
        assert_eq!(parsed.content, "only content");
        assert_eq!(parsed.status, TodoStatus::Pending);
        assert_ne!(parsed.id, "");
    }

    #[test]
    fn next_task_scans_in_list_order() {
        let plan = TaskPlan {
            goal: "goal".to_owned(),
            tasks: vec![
                item("first", TodoStatus::Pending),
                item("second", TodoStatus::Pending),
            ],
            current_task_id: None,
        };
        assert_eq!(plan.next_task().map(|task| task.id.as_str()), Some("first"));

        let skipping = TaskPlan {
            tasks: vec![
                item("running", TodoStatus::InProgress),
                item("done", TodoStatus::Completed),
                item("waiting", TodoStatus::Pending),
            ],
            ..TaskPlan::default()
        };
        assert_eq!(
            skipping.next_task().map(|task| task.id.as_str()),
            Some("waiting")
        );

        let exhausted = TaskPlan {
            tasks: vec![
                item("a", TodoStatus::Completed),
                item("b", TodoStatus::Cancelled),
            ],
            ..TaskPlan::default()
        };
        assert!(exhausted.next_task().is_none());
    }

    #[test]
    fn next_task_treats_cancelled_dependency_as_done() {
        let plan = TaskPlan {
            tasks: vec![
                item("setup", TodoStatus::Cancelled),
                blocked_item("build", &["setup"]),
            ],
            ..TaskPlan::default()
        };
        assert_eq!(plan.next_task().map(|task| task.id.as_str()), Some("build"));
    }

    #[test]
    fn next_task_blocks_on_pending_dependency() {
        let plan = TaskPlan {
            tasks: vec![
                item("setup", TodoStatus::Pending),
                blocked_item("build", &["setup"]),
            ],
            ..TaskPlan::default()
        };
        // `setup` itself is eligible; `build` is not until `setup` is done.
        assert_eq!(plan.next_task().map(|task| task.id.as_str()), Some("setup"));

        let mut progressed = plan;
        progressed.mark_completed("setup", "done");
        assert_eq!(
            progressed.next_task().map(|task| task.id.as_str()),
            Some("build")
        );
    }

    #[test]
    fn next_task_never_returns_task_with_unknown_dependency() {
        let plan = TaskPlan {
            tasks: vec![blocked_item("build", &["missing"])],
            ..TaskPlan::default()
        };
        assert!(plan.next_task().is_none());

        let empty_dependencies = TaskPlan {
            tasks: vec![blocked_item("free", &[])],
            ..TaskPlan::default()
        };
        assert_eq!(
            empty_dependencies.next_task().map(|task| task.id.as_str()),
            Some("free")
        );
    }

    #[test]
    fn get_task_and_add_task() {
        let mut plan = TaskPlan::default();
        assert!(plan.get_task("a").is_none());
        plan.add_task(item("a", TodoStatus::Pending));
        plan.add_task(item("b", TodoStatus::Pending));
        assert_eq!(plan.tasks.len(), 2);
        assert_eq!(
            plan.get_task("a").map(|task| task.content.as_str()),
            Some("task a")
        );
        assert_eq!(plan.get_task("zzz"), None);
    }

    #[test]
    fn mark_in_progress_sets_current_task() {
        let mut plan = TaskPlan {
            tasks: vec![item("a", TodoStatus::Pending)],
            ..TaskPlan::default()
        };
        plan.mark_in_progress("a");
        assert_eq!(plan.get_task("a").unwrap().status, TodoStatus::InProgress);
        assert_eq!(plan.current_task_id.as_deref(), Some("a"));

        plan.mark_in_progress("unknown");
        assert_eq!(plan.current_task_id.as_deref(), Some("a"));
    }

    #[test]
    fn mark_completed_sets_summary_and_clears_current_task() {
        let mut plan = TaskPlan {
            tasks: vec![
                item("a", TodoStatus::InProgress),
                item("b", TodoStatus::Pending),
            ],
            current_task_id: Some("a".to_owned()),
            ..TaskPlan::default()
        };
        plan.mark_completed("a", "all good");
        let task = plan.get_task("a").unwrap();
        assert_eq!(task.status, TodoStatus::Completed);
        assert_eq!(task.result_summary.as_deref(), Some("all good"));
        assert_eq!(plan.current_task_id, None);

        plan.mark_in_progress("b");
        plan.mark_completed("a", "again");
        assert_eq!(plan.current_task_id.as_deref(), Some("b"));

        plan.mark_completed("unknown", "nope");
        assert_eq!(plan.get_task("b").unwrap().status, TodoStatus::InProgress);
    }

    #[test]
    fn mark_cancelled_sets_reason_and_clears_current_task() {
        let mut plan = TaskPlan {
            tasks: vec![item("a", TodoStatus::InProgress)],
            current_task_id: Some("a".to_owned()),
            ..TaskPlan::default()
        };
        plan.mark_cancelled("a", "obsolete");
        let task = plan.get_task("a").unwrap();
        assert_eq!(task.status, TodoStatus::Cancelled);
        assert_eq!(task.result_summary.as_deref(), Some("obsolete"));
        assert_eq!(plan.current_task_id, None);

        plan.mark_cancelled("unknown", "whatever");
        assert!(plan.get_task("unknown").is_none());
    }

    #[test]
    fn progress_summary_excludes_cancelled() {
        assert_eq!(TaskPlan::default().progress_summary(), "0/0 completed");

        let plan = TaskPlan {
            tasks: vec![
                item("a", TodoStatus::Completed),
                item("b", TodoStatus::Completed),
                item("c", TodoStatus::Cancelled),
                item("d", TodoStatus::InProgress),
            ],
            ..TaskPlan::default()
        };
        assert_eq!(plan.progress_summary(), "2/4 completed");
    }

    #[test]
    fn to_markdown_renders_every_status() {
        let mut plan = TaskPlan {
            goal: "ship it".to_owned(),
            tasks: vec![
                item("a", TodoStatus::Pending),
                item("b", TodoStatus::InProgress),
                item("c", TodoStatus::Completed),
                item("d", TodoStatus::Cancelled),
            ],
            ..TaskPlan::default()
        };
        plan.mark_completed("c", "finished");
        plan.mark_cancelled("d", "dropped");

        assert_eq!(
            plan.to_markdown(),
            "## Goal: ship it\n\n\
             - [ ] task a\n\
             - [>] task b\n\
             - [√] task c — finished\n\
             - [×] task d — dropped"
        );
    }

    #[test]
    fn to_markdown_omits_empty_summary_suffix() {
        let plan = TaskPlan {
            goal: "g".to_owned(),
            tasks: vec![TodoItem {
                result_summary: Some(String::new()),
                ..item("a", TodoStatus::Completed)
            }],
            ..TaskPlan::default()
        };
        assert_eq!(plan.to_markdown(), "## Goal: g\n\n- [√] task a");
    }

    #[test]
    fn to_markdown_without_tasks() {
        let plan = TaskPlan {
            goal: "empty".to_owned(),
            ..TaskPlan::default()
        };
        assert_eq!(plan.to_markdown(), "## Goal: empty\n");
    }

    #[test]
    fn plan_dict_round_trip_and_defaults() {
        let plan = TaskPlan {
            goal: "goal".to_owned(),
            tasks: vec![item("a", TodoStatus::Pending)],
            current_task_id: None,
        };
        let dict = plan.to_dict();
        let keys: Vec<&str> = dict
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(keys, vec!["goal", "tasks", "current_task_id"]);
        assert_eq!(dict["current_task_id"], Value::Null);
        assert_eq!(dict["tasks"][0]["activeForm"], json!("doing a"));
        assert_eq!(TaskPlan::from_dict(&dict).unwrap(), plan);

        assert_eq!(
            TaskPlan::from_dict(&Value::Null).unwrap(),
            TaskPlan::default()
        );
        assert_eq!(
            TaskPlan::from_dict(&json!({})).unwrap(),
            TaskPlan::default()
        );
        assert_eq!(
            TaskPlan::from_dict(&json!({"goal": "only goal"}))
                .unwrap()
                .goal,
            "only goal"
        );
        assert!(TaskPlan::from_dict(&json!({"tasks": "nope"})).is_err());
        assert!(TaskPlan::from_dict(&json!([1, 2, 3])).is_err());
    }

    #[test]
    fn model_usage_record_accumulates_and_displays() {
        let mut record = ModelUsageRecord {
            model_id: "gpt-4o".to_owned(),
            ..ModelUsageRecord::default()
        };
        assert_eq!(record.input_tokens, 0);
        assert_eq!(record.output_tokens, 0);
        record.add(10, 5);
        record.add(1, 2);
        assert_eq!(record.input_tokens, 11);
        assert_eq!(record.output_tokens, 7);
        assert_eq!(record.to_string(), "model_id=gpt-4o input=11 output=7");
        assert_eq!(
            serde_json::to_value(&record).unwrap(),
            json!({"model_id": "gpt-4o", "input_tokens": 11, "output_tokens": 7})
        );
    }
}
