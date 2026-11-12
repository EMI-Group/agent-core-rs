//! The plain-text line renderers both CLI scopes write to stdout.
//!
//! Python parity source: the `print(f"…")` calls of `openjiuwen/agent_teams/skill/cli.py` plus the
//! `render_for_llm` views of the member tool set (`view_task`). Every renderer is a pure function
//! returning a [`String`], so the formats are unit-tested without a client, a store or a runtime.
//!
//! The formats are wire-visible: an external script parses these lines, so the spacing of the
//! `task get` block (which is not column-aligned in Python either) is reproduced byte for byte.

use serde_json::Value;

use openjiuwen_common::error::{Error, Result};

use crate::schema::{TaskDetail, TaskStatus, TaskSummary};

use super::super::client::MemberSummary;

/// Renders the placeholder for an absent value (`-`, the Python `or '-'`).
const EMPTY: &str = "-";

/// Renders one row of `task list`: `[<task_id>] <status> <title> (<assignee or ->)`.
#[must_use]
pub(crate) fn task_list_line(task: &TaskSummary) -> String {
    list_line(
        &task.task_id,
        task.status.as_str(),
        &task.title,
        task.assignee.as_deref(),
    )
}

/// Renders one row of `task claimable`: `[<task_id>] <title>`.
#[must_use]
pub(crate) fn claimable_line(task: &TaskSummary) -> String {
    format!("[{}] {}", task.task_id, task.title)
}

/// Renders the 7-line `task get` block, ending with the task content.
#[must_use]
pub(crate) fn task_detail_block(detail: &TaskDetail) -> String {
    detail_block(
        &detail.task_id,
        &detail.title,
        detail.status.as_str(),
        detail.assignee.as_deref(),
        &detail.blocked_by,
        &detail.blocks,
        &detail.content,
    )
}

/// Renders one roster row: `<member_name> (<role>) [<status>]`.
#[must_use]
pub(crate) fn member_line(member: &MemberSummary) -> String {
    format!(
        "{} ({}) [{}]",
        member.member_name,
        member.role.as_str(),
        member.status.as_str()
    )
}

/// Renders the acknowledgement of a direct message: `sent <id> -> <to>`.
#[must_use]
pub(crate) fn sent_line(message_id: &str, to: &str) -> String {
    format!("sent {message_id} -> {to}")
}

/// Renders the acknowledgement of a broadcast: `broadcast <id>`.
#[must_use]
pub(crate) fn broadcast_line(message_id: &str) -> String {
    format!("broadcast {message_id}")
}

/// Renders the acknowledgement of a created task: `created <task_id>`.
#[must_use]
pub(crate) fn created_line(task_id: &str) -> String {
    format!("created {task_id}")
}

/// Renders the acknowledgement of a claim: `claimed <task_id>`.
#[must_use]
pub(crate) fn claimed_line(task_id: &str) -> String {
    format!("claimed {task_id}")
}

/// Renders the acknowledgement of a completion: `completed <task_id>`.
#[must_use]
pub(crate) fn completed_line(task_id: &str) -> String {
    format!("completed {task_id}")
}

/// Renders the acknowledgement of an edit: `updated <task_id>`.
#[must_use]
pub(crate) fn updated_line(task_id: &str) -> String {
    format!("updated {task_id}")
}

/// Renders one `view_task` answer, keyed by the action that produced it.
///
/// The member scope invokes the same tool an in-process teammate invokes, so the answer is raw
/// JSON: `list` and `in_review` are arrays of task rows, `claimable` is an array of
/// `{task_id, title}`, `get` is the serialized task (or `null`), and `note` is a completion note
/// (or `null`).
///
/// # Errors
///
/// Returns [`Error::Validation`] when `action` is unknown or when `value` does not have the shape
/// the action promises.
pub(crate) fn render_view_task(action: &str, value: &Value) -> Result<String> {
    match action {
        "list" | "in_review" => {
            let rows = rows_of(value)?;
            let mut lines = Vec::with_capacity(rows.len());
            for row in rows {
                lines.push(list_line(
                    field(row, "task_id")?,
                    field(row, "status")?,
                    field(row, "title")?,
                    row.get("assignee").and_then(Value::as_str),
                ));
            }
            Ok(lines.join("\n"))
        }
        "claimable" => {
            let rows = rows_of(value)?;
            let mut lines = Vec::with_capacity(rows.len());
            for row in rows {
                lines.push(format!(
                    "[{}] {}",
                    field(row, "task_id")?,
                    field(row, "title")?
                ));
            }
            Ok(lines.join("\n"))
        }
        "get" => match value {
            Value::Null => Ok("null".to_string()),
            Value::Object(_) => json_detail_block(value),
            _ => Err(shape_error(action, "a task object or null")),
        },
        "note" => match value {
            Value::Null => Ok("null".to_string()),
            other => serde_json::to_string(other)
                .map_err(|err| Error::validation(format!("failed to render the note: {err}"))),
        },
        other => Err(Error::validation(format!(
            "unknown view_task action '{other}'"
        ))),
    }
}

/// Parses a task status wire string (`pending`, `in_progress`, `completed`, …).
///
/// # Errors
///
/// Returns [`Error::Validation`] naming `raw` when it is not a task status.
pub(crate) fn format_status_filter(raw: &str) -> Result<TaskStatus> {
    serde_json::from_str::<TaskStatus>(&format!("\"{raw}\""))
        .map_err(|_| Error::validation(format!("unknown task status '{raw}'")))
}

/// Renders `[<task_id>] <status> <title> (<assignee or ->)` from loose parts.
fn list_line(task_id: &str, status: &str, title: &str, assignee: Option<&str>) -> String {
    format!("[{task_id}] {status} {title} ({})", or_empty(assignee))
}

/// Renders the 7-line detail block from loose parts.
fn detail_block(
    task_id: &str,
    title: &str,
    status: &str,
    assignee: Option<&str>,
    blocked_by: &[String],
    blocks: &[String],
    content: &str,
) -> String {
    format!(
        "id:        {task_id}\n\
         title:     {title}\n\
         status:    {status}\n\
         assignee:  {}\n\
         blocked_by: {}\n\
         blocks:     {}\n\
         content:\n\
         {content}",
        or_empty(assignee),
        join_ids(blocked_by),
        join_ids(blocks),
    )
}

/// Renders the detail block of a `view_task` `get` answer.
fn json_detail_block(row: &Value) -> Result<String> {
    let blocked_by = string_list(row, "blocked_by")?;
    let blocks = string_list(row, "blocks")?;
    Ok(detail_block(
        field(row, "task_id")?,
        field(row, "title")?,
        field(row, "status")?,
        row.get("assignee").and_then(Value::as_str),
        &blocked_by,
        &blocks,
        row.get("content")
            .and_then(Value::as_str)
            .unwrap_or_default(),
    ))
}

/// Returns the array of task rows `value` must hold for a list-shaped answer.
fn rows_of(value: &Value) -> Result<&[Value]> {
    value.as_array().map(Vec::as_slice).ok_or_else(|| {
        Error::validation("view_task answer is not an array of task rows".to_string())
    })
}

/// Returns a required string field of one task row.
fn field<'a>(row: &'a Value, key: &str) -> Result<&'a str> {
    row.get(key).and_then(Value::as_str).ok_or_else(|| {
        Error::validation(format!("view_task task row is missing the '{key}' field"))
    })
}

/// Returns a string array field of one task row (absent or `null` means empty).
fn string_list(row: &Value, key: &str) -> Result<Vec<String>> {
    match row.get(key) {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(Value::Array(items)) => items
            .iter()
            .map(|item| {
                item.as_str().map(str::to_owned).ok_or_else(|| {
                    Error::validation(format!("task field '{key}' holds a non-string entry"))
                })
            })
            .collect(),
        Some(_) => Err(Error::validation(format!(
            "task field '{key}' is not a list of ids"
        ))),
    }
}

/// Reports a `view_task` answer whose shape does not match its action.
fn shape_error(action: &str, expected: &str) -> Error {
    Error::validation(format!(
        "view_task '{action}' expects {expected}, got a different JSON shape"
    ))
}

/// Renders `value`, or the `-` placeholder when it is absent or empty.
fn or_empty(value: Option<&str>) -> &str {
    match value {
        Some(text) if !text.is_empty() => text,
        _ => EMPTY,
    }
}

/// Joins task ids with `, `, or the `-` placeholder when the list is empty.
fn join_ids(ids: &[String]) -> String {
    if ids.is_empty() {
        EMPTY.to_string()
    } else {
        ids.join(", ")
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use serde_json::json;

    use crate::schema::{MemberStatus, TeamRole};

    use super::*;

    fn summary() -> TaskSummary {
        TaskSummary {
            task_id: "t-1".to_string(),
            title: "ship it".to_string(),
            status: TaskStatus::InProgress,
            assignee: Some("dev-1".to_string()),
            ..TaskSummary::default()
        }
    }

    fn detail() -> TaskDetail {
        TaskDetail {
            task_id: "t-1".to_string(),
            title: "ship it".to_string(),
            content: "body\nsecond line".to_string(),
            status: TaskStatus::InReview,
            assignee: None,
            blocked_by: vec!["t-0".to_string(), "t-2".to_string()],
            blocks: Vec::new(),
            ..TaskDetail::default()
        }
    }

    #[test]
    fn task_list_line_matches_the_python_format() {
        assert_eq!(
            task_list_line(&summary()),
            "[t-1] in_progress ship it (dev-1)"
        );
        let unassigned = TaskSummary {
            assignee: None,
            ..summary()
        };
        assert_eq!(task_list_line(&unassigned), "[t-1] in_progress ship it (-)");
    }

    #[test]
    fn claimable_line_lists_the_title_only() {
        assert_eq!(claimable_line(&summary()), "[t-1] ship it");
    }

    #[test]
    fn detail_block_uses_the_seven_python_labels() {
        assert_eq!(
            task_detail_block(&detail()),
            "id:        t-1\n\
             title:     ship it\n\
             status:    in_review\n\
             assignee:  -\n\
             blocked_by: t-0, t-2\n\
             blocks:     -\n\
             content:\n\
             body\nsecond line"
        );
    }

    #[test]
    fn member_line_uses_wire_spellings() {
        let member = MemberSummary::new("lead", TeamRole::Leader, MemberStatus::ShutDown);
        assert_eq!(member_line(&member), "lead (leader) [shut_down]");
    }

    #[test]
    fn acknowledgement_lines_are_pinned() {
        assert_eq!(sent_line("m1", "dev-1"), "sent m1 -> dev-1");
        assert_eq!(broadcast_line("m2"), "broadcast m2");
        assert_eq!(created_line("t-1"), "created t-1");
        assert_eq!(claimed_line("t-1"), "claimed t-1");
        assert_eq!(completed_line("t-1"), "completed t-1");
        assert_eq!(updated_line("t-1"), "updated t-1");
    }

    #[test]
    fn render_view_task_lists_rows() {
        let value = json!([
            { "task_id": "t-1", "title": "one", "status": "pending", "assignee": null },
            { "task_id": "t-2", "title": "two", "status": "completed", "assignee": "dev-1" },
        ]);
        assert_eq!(
            render_view_task("list", &value).unwrap(),
            "[t-1] pending one (-)\n[t-2] completed two (dev-1)"
        );
        assert_eq!(
            render_view_task("in_review", &value).unwrap(),
            "[t-1] pending one (-)\n[t-2] completed two (dev-1)"
        );
        assert_eq!(render_view_task("list", &json!([])).unwrap(), "");
        assert!(render_view_task("list", &json!({}))
            .unwrap_err()
            .to_string()
            .contains("not an array"));
    }

    #[test]
    fn render_view_task_claimable_rows() {
        let value = json!([{ "task_id": "t-1", "title": "one" }]);
        assert_eq!(render_view_task("claimable", &value).unwrap(), "[t-1] one");
        assert!(
            render_view_task("claimable", &json!([{ "task_id": "t-1" }]))
                .unwrap_err()
                .to_string()
                .contains("'title'")
        );
    }

    #[test]
    fn render_view_task_detail_block_round_trips_the_task_json() {
        let value = serde_json::to_value(detail()).unwrap();
        assert_eq!(
            render_view_task("get", &value).unwrap(),
            task_detail_block(&detail())
        );
        assert_eq!(render_view_task("get", &Value::Null).unwrap(), "null");
        assert!(render_view_task("get", &json!([1]))
            .unwrap_err()
            .to_string()
            .contains("task object or null"));
        assert!(render_view_task("get", &json!({ "task_id": "t-1" }))
            .unwrap_err()
            .to_string()
            .contains("'title'"));
    }

    #[test]
    fn render_view_task_notes_and_unknown_actions() {
        assert_eq!(
            render_view_task("note", &json!("closing note")).unwrap(),
            "\"closing note\""
        );
        assert_eq!(render_view_task("note", &Value::Null).unwrap(), "null");
        assert!(render_view_task("explode", &Value::Null)
            .unwrap_err()
            .to_string()
            .contains("unknown view_task action 'explode'"));
    }

    #[test]
    fn status_filter_parses_wire_strings() {
        assert_eq!(
            format_status_filter("in_progress").unwrap(),
            TaskStatus::InProgress
        );
        let err = format_status_filter("shipped").unwrap_err();
        assert!(
            err.to_string().contains("unknown task status 'shipped'"),
            "unexpected message: {err}"
        );
    }
}
