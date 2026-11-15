//! Controller tasks and their results.
//!
//! Ports `openjiuwen/core/controller/schema/task.py`: a [`Task`] is one unit
//! of work scheduled onto a task executor, and [`TaskResult`] is what an
//! executor reports back. [`TaskStatus`] keeps Python's wire strings,
//! including the hyphenated `input-required`.
use super::dataframe::DataFrame;
use super::event::Event;
use super::output::ControllerOutputChunk;
use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
/// Task lifecycle statuses.
///
/// Wire strings are `submitted`, `working`, `paused`, `input-required`,
/// `completed`, `canceled`, `failed`, `waiting`, and `unknown`.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum TaskStatus {
    /// Queued and waiting for a worker.
    Submitted,
    /// Currently executing.
    Working,
    /// Suspended and resumable.
    Paused,
    /// Waiting for the caller to supply required input.
    #[serde(rename = "input-required")]
    InputRequired,
    /// Finished successfully.
    Completed,
    /// Cancelled before finishing.
    Canceled,
    /// Finished with an error.
    Failed,
    /// Waiting on a dependent task.
    Waiting,
    /// Status not reported yet.
    #[default]
    Unknown,
}
impl TaskStatus {
    /// Returns the Python wire string of the status.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Submitted => "submitted",
            Self::Working => "working",
            Self::Paused => "paused",
            Self::InputRequired => "input-required",
            Self::Completed => "completed",
            Self::Canceled => "canceled",
            Self::Failed => "failed",
            Self::Waiting => "waiting",
            Self::Unknown => "unknown",
        }
    }
    /// Whether the status is final (`completed`, `canceled`, `failed`).
    #[must_use]
    pub const fn is_terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Canceled | Self::Failed)
    }
    /// Whether the task is running or about to run (`submitted`, `working`).
    #[must_use]
    pub const fn is_active(self) -> bool {
        matches!(self, Self::Submitted | Self::Working)
    }
}
/// Result of executing one task.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct TaskResult {
    /// Outcome status.
    pub status: TaskStatus,
    /// Structured result payload; `null` is omitted from the wire.
    #[serde(default, skip_serializing_if = "Value::is_null")]
    pub output: Value,
    /// Failure reason when the task failed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Free-form result metadata.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, Value>,
    /// Frames the result carries (interaction prompts, partial output).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub data: Vec<DataFrame>,
    /// The executor's reasoning text; surfaced only when the controller's
    /// `expose_reasoning` is on.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
}
impl TaskResult {
    /// Builds a completed result carrying `output`.
    #[must_use]
    pub fn completed(output: impl Into<Value>) -> Self {
        Self {
            status: TaskStatus::Completed,
            output: output.into(),
            ..Self::default()
        }
    }
    /// Builds a completed result carrying a text output.
    #[must_use]
    pub fn text(text: impl Into<String>) -> Self {
        Self::completed(Value::String(text.into()))
    }
    /// Builds a failed result carrying `error`.
    #[must_use]
    pub fn failed(error: impl Into<String>) -> Self {
        Self {
            status: TaskStatus::Failed,
            error: Some(error.into()),
            ..Self::default()
        }
    }
    /// Builds a result that waits for caller input, carrying `data`.
    #[must_use]
    pub fn input_required(data: Vec<DataFrame>) -> Self {
        Self {
            status: TaskStatus::InputRequired,
            data,
            ..Self::default()
        }
    }
    /// Builds an interrupted result.
    ///
    /// Python's legacy `interrupt` maps onto [`TaskStatus::InputRequired`]
    /// with the captured frames.
    #[must_use]
    pub fn interrupted(data: Vec<DataFrame>) -> Self {
        Self::input_required(data)
    }
    /// Renders the output as text.
    ///
    /// Strings are returned as-is, `null` becomes empty, and every other
    /// value is serialized as JSON.
    #[must_use]
    pub fn display(&self) -> String {
        match &self.output {
            Value::String(text) => text.clone(),
            Value::Null => String::new(),
            other => serde_json::to_string(other).unwrap_or_default(),
        }
    }
    /// Sets the executor's reasoning text.
    #[must_use]
    pub fn with_reasoning(mut self, reasoning: impl Into<String>) -> Self {
        self.reasoning = Some(reasoning.into());
        self
    }
}
/// One scheduled task.
///
/// [`Default`] mirrors Python's field defaults (`priority` 1,
/// [`TaskStatus::Unknown`]); [`Task::new`] starts a task in
/// [`TaskStatus::Submitted`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct Task {
    /// Session the task belongs to.
    pub session_id: String,
    /// Unique task id.
    pub task_id: String,
    /// Task type, which selects the executor.
    pub task_type: String,
    /// Human-readable task description.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Scheduling priority; larger numbers are handled first (Python default 1).
    #[serde(default = "default_priority")]
    pub priority: i32,
    /// Input events that fed the task.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub inputs: Vec<Event>,
    /// Output frames produced by the task.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub outputs: Vec<ControllerOutputChunk>,
    /// Current lifecycle status.
    pub status: TaskStatus,
    /// Parent task, when the task was spawned by another task.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_task_id: Option<String>,
    /// Context the task shares with its siblings.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_id: Option<String>,
    /// Field definitions the caller must fill while the task is
    /// [`TaskStatus::InputRequired`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_required_fields: Option<Value>,
    /// Failure reason while the task is [`TaskStatus::Failed`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
    /// Free-form task metadata.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, Value>,
    /// Protocol extensions carried through unchanged.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extensions: BTreeMap<String, Value>,
    /// Rust-only creation sequence used for stable ordering; never on the wire.
    #[serde(default, skip)]
    pub created_seq: u64,
}
/// Default task priority (Python `1`).
fn default_priority() -> i32 {
    1
}
impl Default for Task {
    fn default() -> Self {
        Self {
            session_id: String::new(),
            task_id: String::new(),
            task_type: String::new(),
            description: None,
            priority: default_priority(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            status: TaskStatus::Unknown,
            parent_task_id: None,
            context_id: None,
            input_required_fields: None,
            error_message: None,
            metadata: BTreeMap::new(),
            extensions: BTreeMap::new(),
            created_seq: 0,
        }
    }
}
impl Task {
    /// Builds a submitted task with the Python defaults.
    #[must_use]
    pub fn new(
        session_id: impl Into<String>,
        task_id: impl Into<String>,
        task_type: impl Into<String>,
    ) -> Self {
        Self {
            session_id: session_id.into(),
            task_id: task_id.into(),
            task_type: task_type.into(),
            status: TaskStatus::Submitted,
            ..Self::default()
        }
    }
    /// Builds a submitted task carrying a description.
    #[must_use]
    pub fn submitted(
        session_id: impl Into<String>,
        task_id: impl Into<String>,
        task_type: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        Self::new(session_id, task_id, task_type).with_description(description)
    }
    /// Validates the required fields and status-specific invariants.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`] when an id or the task type is blank,
    /// `priority` is negative, `parent_task_id` equals `task_id`,
    /// [`TaskStatus::Failed`] lacks `error_message`, or
    /// [`TaskStatus::InputRequired`] lacks `input_required_fields`.
    pub fn validate(&self) -> Result<()> {
        for (name, value) in [
            ("session_id", &self.session_id),
            ("task_id", &self.task_id),
            ("task_type", &self.task_type),
        ] {
            if value.trim().is_empty() {
                return Err(Error::validation(format!("{name} cannot be empty")));
            }
        }
        if self.priority < 0 {
            return Err(Error::validation("Priority must be a non-negative integer"));
        }
        if self.parent_task_id.as_deref() == Some(self.task_id.as_str()) {
            return Err(Error::validation(
                "task_id cannot be the same as parent_task_id (circular reference)",
            ));
        }
        if self.status == TaskStatus::Failed
            && self
                .error_message
                .as_deref()
                .is_none_or(|message| message.trim().is_empty())
        {
            return Err(Error::validation(
                "error_message is required when status is FAILED",
            ));
        }
        if self.status == TaskStatus::InputRequired && self.input_required_fields.is_none() {
            return Err(Error::validation(
                "input_required_fields is required when status is INPUT_REQUIRED",
            ));
        }
        Ok(())
    }
    /// Text of the last input event, falling back to the task description.
    #[must_use]
    pub fn input_text(&self) -> Option<String> {
        self.inputs
            .last()
            .and_then(Event::text)
            .or_else(|| self.description.clone())
    }
    /// Whether the task reached a terminal status.
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        self.status.is_terminal()
    }
    /// Whether the task is submitted or working.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.status.is_active()
    }
    /// Sets the task description.
    #[must_use]
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }
    /// Sets the lifecycle status.
    #[must_use]
    pub fn with_status(mut self, status: TaskStatus) -> Self {
        self.status = status;
        self
    }
    /// Sets the scheduling priority.
    #[must_use]
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }
    /// Appends one input event.
    #[must_use]
    pub fn with_input(mut self, event: Event) -> Self {
        self.inputs.push(event);
        self
    }
    /// Sets the parent task id.
    #[must_use]
    pub fn with_parent(mut self, parent_task_id: impl Into<String>) -> Self {
        self.parent_task_id = Some(parent_task_id.into());
        self
    }
    /// Sets the context id.
    #[must_use]
    pub fn with_context_id(mut self, context_id: impl Into<String>) -> Self {
        self.context_id = Some(context_id.into());
        self
    }
    /// Adds one metadata entry.
    #[must_use]
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<Value>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
    /// Sets the failure reason.
    #[must_use]
    pub fn with_error(mut self, error_message: impl Into<String>) -> Self {
        self.error_message = Some(error_message.into());
        self
    }
    /// Sets the creation sequence used for stable ordering.
    #[must_use]
    pub fn with_created_seq(mut self, created_seq: u64) -> Self {
        self.created_seq = created_seq;
        self
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn task_round_trips_with_hyphenated_status() {
        let mut task = Task::submitted("s1", "t1", "agent", "do it")
            .with_status(TaskStatus::InputRequired)
            .with_input(Event::user_input("hello"));
        task.input_required_fields = Some(json!({"fields": []}));
        let wire = serde_json::to_value(&task).unwrap();
        assert_eq!(wire["status"], json!("input-required"));
        assert_eq!(wire["priority"], json!(1));
        let parsed: Task = serde_json::from_value(wire).unwrap();
        assert_eq!(parsed, task);
        let completed = task.clone().with_status(TaskStatus::Completed);
        let wire = serde_json::to_value(&completed).unwrap();
        assert_eq!(wire["status"], json!("completed"));
        let parsed: Task = serde_json::from_value(wire).unwrap();
        assert_eq!(parsed.status, TaskStatus::Completed);
    }
    #[test]
    fn created_seq_is_never_serialized() {
        let task = Task::new("s1", "t1", "agent").with_created_seq(7);
        let wire = serde_json::to_value(&task).unwrap();
        assert!(wire.get("created_seq").is_none());
        let parsed: Task = serde_json::from_value(wire).unwrap();
        assert_eq!(parsed.created_seq, 0);
    }
    #[test]
    fn validation_enforces_required_fields() {
        assert!(Task::new("s1", "t1", "agent").validate().is_ok());
        assert!(Task::new(" ", "t1", "agent").validate().is_err());
        assert!(Task::new("s1", "", "agent").validate().is_err());
        assert!(Task::new("s1", "t1", "  ").validate().is_err());
        assert!(Task::new("s1", "t1", "agent")
            .with_priority(-1)
            .validate()
            .is_err());
        assert!(Task::new("s1", "t1", "agent")
            .with_parent("t1")
            .validate()
            .is_err());
        assert!(Task::new("s1", "t1", "agent")
            .with_status(TaskStatus::Failed)
            .validate()
            .is_err());
        assert!(Task::new("s1", "t1", "agent")
            .with_status(TaskStatus::Failed)
            .with_error("boom")
            .validate()
            .is_ok());
        assert!(Task::new("s1", "t1", "agent")
            .with_status(TaskStatus::InputRequired)
            .validate()
            .is_err());
    }
    #[test]
    fn input_text_prefers_the_last_event() {
        assert_eq!(
            Task::new("s1", "t1", "agent")
                .with_description("fallback")
                .input_text()
                .as_deref(),
            Some("fallback")
        );
        assert_eq!(
            Task::new("s1", "t1", "agent")
                .with_input(Event::user_input("first"))
                .with_input(Event::user_input("second"))
                .input_text()
                .as_deref(),
            Some("second")
        );
    }
    #[test]
    fn task_result_helpers() {
        let completed = TaskResult::text("done");
        assert_eq!(completed.status, TaskStatus::Completed);
        assert_eq!(completed.display(), "done");
        assert_eq!(TaskResult::default().display(), "");
        assert_eq!(
            TaskResult::completed(json!({"a": 1})).display(),
            "{\"a\":1}"
        );
        let failed = TaskResult::failed("boom");
        assert_eq!(failed.status, TaskStatus::Failed);
        assert_eq!(failed.error.as_deref(), Some("boom"));
        let interrupted = TaskResult::interrupted(vec![DataFrame::text("ask")]);
        assert_eq!(interrupted.status, TaskStatus::InputRequired);
        assert_eq!(interrupted.data.len(), 1);
        let wire = serde_json::to_value(&interrupted).unwrap();
        assert_eq!(wire["status"], json!("input-required"));
    }
    #[test]
    fn task_result_reasoning_builder() {
        let result = TaskResult::text("done").with_reasoning("why");
        assert_eq!(result.reasoning.as_deref(), Some("why"));
        assert_eq!(result.status, TaskStatus::Completed);
        assert_eq!(result.display(), "done");
        assert_eq!(TaskResult::text("done").reasoning, None);
        assert_eq!(TaskResult::default().reasoning, None);
    }
    #[test]
    fn task_result_reasoning_wire_shape() {
        let plain = serde_json::to_value(TaskResult::text("done")).unwrap();
        assert!(plain.get("reasoning").is_none());
        assert_eq!(plain, json!({"status": "completed", "output": "done"}));
        let reasoned = TaskResult::text("done").with_reasoning("why");
        let wire = serde_json::to_value(&reasoned).unwrap();
        assert_eq!(
            wire,
            json!({"status": "completed", "output": "done", "reasoning": "why"})
        );
        let parsed: TaskResult = serde_json::from_value(wire).unwrap();
        assert_eq!(parsed, reasoned);
        let tolerated: TaskResult =
            serde_json::from_value(json!({"status": "completed", "reasoning": "why"})).unwrap();
        assert_eq!(tolerated.reasoning.as_deref(), Some("why"));
    }
    #[test]
    fn status_helpers() {
        assert!(TaskStatus::Completed.is_terminal());
        assert!(TaskStatus::Canceled.is_terminal());
        assert!(TaskStatus::Failed.is_terminal());
        assert!(!TaskStatus::Working.is_terminal());
        assert!(TaskStatus::Submitted.is_active());
        assert!(TaskStatus::Working.is_active());
        assert!(!TaskStatus::Paused.is_active());
        assert_eq!(TaskStatus::InputRequired.as_str(), "input-required");
        assert_eq!(TaskStatus::Unknown.as_str(), "unknown");
    }
}
