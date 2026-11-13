//! Fixtures shared by the SQLite suite: row builders, id projections and `StorageSpec` helpers.

use std::path::Path;

use serde_json::Value;

use crate::schema::{StorageSpec, TaskRecord, TaskStatus};
use crate::storage::records::{MemberRecord, MessageRecord};

/// Builds one task row with the fields the listing and filtering assertions look at.
pub(super) fn task(
    task_id: &str,
    updated_at: u64,
    status: TaskStatus,
    assignee: Option<&str>,
    parent: Option<&str>,
) -> TaskRecord {
    TaskRecord {
        status,
        assignee: assignee.map(str::to_owned),
        parent_task_id: parent.map(str::to_owned),
        updated_at,
        ..TaskRecord::new("alpha", task_id, format!("title {task_id}"), "body")
    }
}

/// Projects a roster to its member names, in the order the store returned them.
pub(super) fn member_names(rows: &[MemberRecord]) -> Vec<&str> {
    rows.iter().map(|row| row.member_name.as_str()).collect()
}

/// Projects a mailbox to its message ids, in the order the store returned them.
pub(super) fn ids(messages: &[MessageRecord]) -> Vec<&str> {
    messages
        .iter()
        .map(|message| message.message_id.as_str())
        .collect()
}

/// Projects a task listing to its task ids, in the order the store returned them.
pub(super) fn task_ids(tasks: &[TaskRecord]) -> Vec<&str> {
    tasks.iter().map(|task| task.task_id.as_str()).collect()
}

/// Builds a [`StorageSpec`] of `kind` with the given `params` entries.
pub(super) fn spec(kind: &str, params: &[(&str, Value)]) -> StorageSpec {
    StorageSpec {
        kind: kind.to_owned(),
        params: params
            .iter()
            .map(|(key, value)| ((*key).to_owned(), value.clone()))
            .collect(),
    }
}

/// Wraps a filesystem path as the JSON string a `StorageSpec` parameter carries.
pub(super) fn path_value(path: &Path) -> Value {
    Value::String(path.to_string_lossy().into_owned())
}
