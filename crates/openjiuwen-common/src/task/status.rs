//! Task identity, status, options, and snapshots.
//!
//! These types port the Python `openjiuwen.core.common.task_manager.types` and
//! `openjiuwen.core.common.task_manager.task` data models: `TaskStatus` becomes
//! [`TaskStatus`], the mutable `Task` dataclass becomes the immutable [`TaskInfo`]
//! snapshot plus [`TaskOptions`] for creation-time configuration.
//!
//! ```
//! use openjiuwen_common::task::{TaskId, TaskStatus};
//!
//! let id = TaskId::generate();
//! assert_eq!(id.as_str().len(), 36);
//! assert!(TaskStatus::Completed.is_terminal());
//! ```
#![allow(clippy::module_name_repetitions)]

use std::collections::BTreeMap;
use std::fmt;
use std::time::Duration;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::utils::id::uuid_v4;

/// Unique identifier of a tracked task.
///
/// The id is a UUID v4 in canonical hyphenated form (Python `str(uuid4())`).
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TaskId(String);

impl TaskId {
    /// Generates a random task id.
    #[must_use]
    pub fn generate() -> Self {
        Self(uuid_v4())
    }

    /// Creates a task id from an existing string.
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Returns the id as a string slice.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for TaskId {
    fn default() -> Self {
        Self::generate()
    }
}

impl fmt::Display for TaskId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<&str> for TaskId {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}

impl From<String> for TaskId {
    fn from(value: String) -> Self {
        Self::new(value)
    }
}

/// Lifecycle status of a tracked task.
///
/// The string values match the Python `TaskStatus` enum exactly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskStatus {
    /// Registered but not yet polled.
    #[default]
    Pending,
    /// The wrapped future is being polled.
    Running,
    /// The future returned successfully.
    Completed,
    /// The future panicked or returned an error.
    Failed,
    /// The task was cancelled, explicitly or through a cascade.
    Cancelled,
    /// The task exceeded its timeout.
    Timeout,
}

impl TaskStatus {
    /// The terminal statuses; a task in one of these never transitions again.
    pub const TERMINAL: [Self; 4] = [
        Self::Completed,
        Self::Failed,
        Self::Cancelled,
        Self::Timeout,
    ];

    /// Returns the Python-compatible `snake_case` name of this status.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Running => "running",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
            Self::Timeout => "timeout",
        }
    }

    /// Returns `true` when no further transition is possible.
    #[must_use]
    pub const fn is_terminal(&self) -> bool {
        matches!(
            self,
            Self::Completed | Self::Failed | Self::Cancelled | Self::Timeout
        )
    }
}

impl fmt::Display for TaskStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Creation-time configuration for [`crate::task::TaskManager::spawn_with`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TaskOptions {
    /// Human-readable task name; empty means the id prefix is displayed instead.
    pub name: String,
    /// Optional group used by group-scoped cancellation.
    pub group: Option<String>,
    /// Optional per-task timeout.
    pub timeout: Option<Duration>,
    /// Arbitrary caller metadata carried on the task snapshot.
    pub metadata: BTreeMap<String, Value>,
}

impl TaskOptions {
    /// Creates options with the given task name.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Self::default()
        }
    }

    /// Sets the task group.
    #[must_use]
    pub fn with_group(mut self, group: impl Into<String>) -> Self {
        self.group = Some(group.into());
        self
    }

    /// Sets the per-task timeout.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Adds one metadata entry.
    #[must_use]
    pub fn with_metadata(mut self, key: impl Into<String>, value: Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }
}

/// Immutable snapshot of a tracked task.
///
/// Snapshots are cheap values used by the listing API and the status event
/// broadcast; the mutable state lives inside the task manager.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskInfo {
    /// Unique task id.
    pub id: TaskId,
    /// Human-readable task name (may be empty).
    pub name: String,
    /// Optional group used by group-scoped cancellation.
    pub group: Option<String>,
    /// Current lifecycle status.
    pub status: TaskStatus,
    /// Id of the task that spawned this one, if any.
    pub parent_id: Option<TaskId>,
    /// Registration timestamp.
    pub created_at: DateTime<Utc>,
    /// Timestamp when the task started running.
    pub started_at: Option<DateTime<Utc>>,
    /// Timestamp when the task reached a terminal status.
    pub finished_at: Option<DateTime<Utc>>,
    /// Configured timeout, if any.
    pub timeout: Option<Duration>,
    /// Error message for failed or timed-out tasks.
    pub error: Option<String>,
    /// Reason recorded when the task was cancelled.
    pub cancel_reason: Option<String>,
    /// Id of the task or actor that requested cancellation.
    pub cancelled_by: Option<String>,
    /// Arbitrary caller metadata.
    pub metadata: BTreeMap<String, Value>,
}

impl TaskInfo {
    /// Returns the name, falling back to the first eight characters of the id.
    ///
    /// Port of Python `Task.display_name`.
    #[must_use]
    pub fn display_name(&self) -> String {
        if self.name.is_empty() {
            self.id.as_str().chars().take(8).collect()
        } else {
            self.name.clone()
        }
    }

    /// Returns the wall-clock duration between start and finish, in milliseconds.
    #[must_use]
    pub fn duration_ms(&self) -> Option<i64> {
        let started = self.started_at?;
        let finished = self.finished_at?;
        Some((finished - started).num_milliseconds())
    }
}

// port of tests/unit_tests/core/common/test_task_manager.py
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn sample_info() -> TaskInfo {
        TaskInfo {
            id: TaskId::new("0123456789abcdef"),
            name: "worker".into(),
            group: Some("pool".into()),
            status: TaskStatus::Pending,
            parent_id: None,
            created_at: Utc::now(),
            started_at: None,
            finished_at: None,
            timeout: None,
            error: None,
            cancel_reason: None,
            cancelled_by: None,
            metadata: BTreeMap::new(),
        }
    }

    #[test]
    fn status_serializes_to_python_values() {
        let cases = [
            (TaskStatus::Pending, "\"pending\""),
            (TaskStatus::Running, "\"running\""),
            (TaskStatus::Completed, "\"completed\""),
            (TaskStatus::Failed, "\"failed\""),
            (TaskStatus::Cancelled, "\"cancelled\""),
            (TaskStatus::Timeout, "\"timeout\""),
        ];
        for (status, expected) in cases {
            assert_eq!(serde_json::to_string(&status).unwrap(), expected);
            assert_eq!(status.as_str(), status.to_string());
            assert_eq!(
                serde_json::from_str::<TaskStatus>(expected).unwrap(),
                status
            );
        }
        assert_eq!(TaskStatus::default(), TaskStatus::Pending);
    }

    #[test]
    fn terminal_statuses_are_exactly_the_python_set() {
        let terminal: Vec<TaskStatus> = TaskStatus::TERMINAL.to_vec();
        for status in terminal {
            assert!(status.is_terminal());
        }
        assert!(!TaskStatus::Pending.is_terminal());
        assert!(!TaskStatus::Running.is_terminal());
    }

    #[test]
    fn task_id_round_trips_through_json() {
        let id = TaskId::generate();
        let json = serde_json::to_string(&id).unwrap();
        assert_eq!(json, format!("\"{id}\""));
        assert_eq!(serde_json::from_str::<TaskId>(&json).unwrap(), id);

        let restored = serde_json::from_str::<TaskId>("\"abc\"").unwrap();
        assert_eq!(restored, TaskId::from("abc"));
        assert_eq!(TaskId::from(String::from("abc")), restored);
        assert_eq!(TaskId::default().as_str().len(), 36);
        assert_eq!(id.to_string(), id.as_str());
    }

    #[test]
    fn display_name_falls_back_to_id_prefix() {
        let mut info = sample_info();
        assert_eq!(info.display_name(), "worker");
        info.name.clear();
        assert_eq!(info.display_name(), "01234567");
    }

    #[test]
    fn duration_is_only_known_after_finish() {
        let mut info = sample_info();
        assert_eq!(info.duration_ms(), None);
        info.started_at = Some(Utc::now());
        assert_eq!(info.duration_ms(), None);
        info.finished_at = Some(info.started_at.unwrap());
        assert_eq!(info.duration_ms(), Some(0));
    }

    #[test]
    fn options_builders_compose() {
        let options = TaskOptions::new("worker")
            .with_group("pool")
            .with_timeout(Duration::from_millis(10))
            .with_metadata("priority", serde_json::json!("high"));
        assert_eq!(options.name, "worker");
        assert_eq!(options.group.as_deref(), Some("pool"));
        assert_eq!(options.timeout, Some(Duration::from_millis(10)));
        assert_eq!(options.metadata.get("priority").unwrap(), "high");
        assert_eq!(TaskOptions::default().name, "");
    }
}
