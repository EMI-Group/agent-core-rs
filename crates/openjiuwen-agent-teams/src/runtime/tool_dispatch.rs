//! The in-process team-tool dispatcher: the four member-facing tools answered against a
//! [`TeamBackend`](super::backend::TeamBackend).
//!
//! This is the in-process counterpart of `crate::skill`'s store-backed `invoke_tool`: it serves the
//! same four member-facing tools (`view_task`, `claim_task`, `verify_task`, `send_message`) with the
//! same wire answer shapes, but reads and writes the live backend (task board, mailbox and event
//! bus) instead of rebuilding a client from a store and a join descriptor. It deliberately does not
//! import `crate::skill`, so neither side depends on the other.
//!
//! Callers own roster and role validation: the dispatcher never checks that `member` is on the
//! roster, that the human it acts for may use a tool, or that the tool is exposed in that member's
//! scope. `member` is simply the acting identity, so assignee locks, reviewer guards and the sender
//! stamp all follow from it.
//!
//! Errors are reserved for malformed requests and infrastructure failures: a board refusal (already
//! claimed, not the assignee, not under review) is a successful answer carrying `ok = false` and the
//! board's stable reason, exactly like [`crate::skill`]'s client.
//!
//! The entry points carry `#[allow(dead_code)]` because their callers live in sibling leaves that
//! land separately: `TeamAgent::deliver` (a relayed passive-human tool call) and the attach
//! handler's `invoke_tool` op.

use serde_json::{json, Value};

use openjiuwen_common::error::{Error, Result};

use crate::schema::{TaskFilter, TaskStatus, TaskSummary, Verdict};

use super::backend::TeamBackend;

/// Member-facing team tool names this dispatcher serves.
#[allow(dead_code)] // read through `is_supported_tool`; the callers land in sibling leaves
pub(crate) const TEAM_TOOL_NAMES: [&str; 4] =
    ["view_task", "claim_task", "verify_task", "send_message"];

/// Action the `view_task` tool performs when the caller does not name one.
const DEFAULT_VIEW_ACTION: &str = "list";

/// Returns whether `name` is one of the team tools this dispatcher serves.
#[must_use]
#[allow(dead_code)] // the callers land in sibling leaves
pub(crate) fn is_supported_tool(name: &str) -> bool {
    TEAM_TOOL_NAMES.contains(&name)
}

/// Dispatches one member-facing team tool for `member` against `backend`.
///
/// `member` is the acting identity (the leader, or a passive human the caller already checked); it
/// is never validated against the roster here. `arguments` is the tool-call argument object; a
/// missing required key is [`Error::Validation`] naming that key.
///
/// # Errors
///
/// * [`Error::NotFound`] when `name` is not a team tool.
/// * [`Error::Validation`] for a malformed argument (missing key, unknown action/status/verdict, a
///   task detail that cannot be serialized).
/// * Whatever the board, the store or the transport fails with while applying the tool.
#[allow(dead_code)] // the callers land in sibling leaves
pub(crate) async fn dispatch_tool(
    backend: &TeamBackend,
    member: &str,
    name: &str,
    arguments: &Value,
) -> Result<Value> {
    match name {
        "view_task" => view_task(backend, arguments).await,
        "claim_task" => claim_task(backend, member, arguments).await,
        "verify_task" => verify_task(backend, member, arguments).await,
        "send_message" => send_message(backend, member, arguments).await,
        other => Err(Error::not_found(format!("unknown tool '{other}'"))),
    }
}

/// Renders a tool answer as the text a tool call returns.
///
/// A JSON string is passed through verbatim (the answer is already text); every other value is
/// rendered as compact JSON, the same shape the harness shows a model.
#[must_use]
#[allow(dead_code)] // the callers land in sibling leaves
pub(crate) fn render_tool_output(value: &Value) -> Value {
    match value {
        Value::String(text) => Value::String(text.clone()),
        other => match serde_json::to_string(other) {
            Ok(text) => Value::String(text),
            // A `Value` map always has string keys, so this is unreachable in practice; `Display`
            // renders the same compact JSON, which keeps the function total without a panic.
            Err(_) => Value::String(other.to_string()),
        },
    }
}

/// Implements `view_task` (`list`, `claimable`, `in_review`, `get`, `note`).
async fn view_task(backend: &TeamBackend, arguments: &Value) -> Result<Value> {
    let action = arguments
        .get("action")
        .and_then(Value::as_str)
        .unwrap_or(DEFAULT_VIEW_ACTION);
    match action {
        "list" => {
            let status = match arguments.get("status").and_then(Value::as_str) {
                Some(raw) => Some(parse_task_status(raw)?),
                None => None,
            };
            let filter = TaskFilter {
                status,
                ..TaskFilter::default()
            };
            Ok(task_rows(&backend.tasks().view(&filter).await?))
        }
        "in_review" => {
            let filter = TaskFilter {
                status: Some(TaskStatus::InReview),
                ..TaskFilter::default()
            };
            Ok(task_rows(&backend.tasks().view(&filter).await?))
        }
        "claimable" => {
            let tasks = backend.tasks().claimable().await?;
            Ok(Value::Array(
                tasks
                    .iter()
                    .map(|task| json!({ "task_id": task.task_id, "title": task.title }))
                    .collect(),
            ))
        }
        "get" => {
            let task_id = required_str(arguments, "task_id")?;
            match backend.tasks().detail(task_id).await? {
                Some(detail) => serde_json::to_value(&detail).map_err(|err| {
                    Error::validation(format!("failed to serialize task '{task_id}': {err}"))
                }),
                None => Ok(Value::Null),
            }
        }
        // Python's `note` view reads a completion note; this crate keeps notes on the task row
        // itself, so there is no separate read view to answer with.
        "note" => Ok(Value::Null),
        other => Err(Error::validation(format!(
            "unknown view_task action '{other}'"
        ))),
    }
}

/// Implements `claim_task` (`status = claimed | completed`).
async fn claim_task(backend: &TeamBackend, member: &str, arguments: &Value) -> Result<Value> {
    let task_id = required_str(arguments, "task_id")?;
    let status = required_str(arguments, "status")?;
    match status {
        "claimed" => {
            let outcome = backend.tasks().claim(task_id, member).await?;
            Ok(json!({ "ok": outcome.ok, "reason": outcome.reason }))
        }
        "completed" => {
            // `complete` refuses with an `Err` (not the assignee, stale content, wrong status),
            // unlike `claim`'s `Ok(ok: false)`: a refusal is a normal tool answer here too.
            match backend.tasks().complete(task_id, member, "").await {
                Ok(mutation) => {
                    backend.publish_mutation(&mutation).await?;
                    Ok(json!({ "ok": true, "reason": null }))
                }
                Err(err) => Ok(json!({ "ok": false, "reason": err.to_string() })),
            }
        }
        other => Err(Error::validation(format!(
            "unknown claim_task status '{other}'; expected 'claimed' or 'completed'"
        ))),
    }
}

/// Implements `verify_task` (one reviewer verdict on a task under review).
async fn verify_task(backend: &TeamBackend, member: &str, arguments: &Value) -> Result<Value> {
    let task_id = required_str(arguments, "task_id")?;
    let decision = required_str(arguments, "decision")?;
    let verdict = parse_verdict(decision)?;
    let feedback = arguments.get("feedback").and_then(Value::as_str);
    match backend
        .tasks()
        .verify(task_id, member, verdict, feedback.unwrap_or(""))
        .await
    {
        Ok(mutation) => {
            let task_id = mutation.task.task_id.clone();
            backend.publish_mutation(&mutation).await?;
            Ok(json!({
                "ok": true,
                "task_id": task_id,
                "decision": verdict.as_str(),
                "feedback": feedback,
            }))
        }
        Err(err) => Ok(json!({ "ok": false, "reason": err.to_string() })),
    }
}

/// Implements `send_message`: one mailbox row plus its message event.
async fn send_message(backend: &TeamBackend, member: &str, arguments: &Value) -> Result<Value> {
    let to = required_str(arguments, "to")?;
    let content = required_str(arguments, "content")?;
    let record = backend.send_message(member, Some(to), content).await?;
    Ok(json!({
        "ok": true,
        "message_id": record.message_id,
        "reason": null,
    }))
}

/// Projects one list row of the `view_task` tool.
#[must_use]
fn task_rows(tasks: &[TaskSummary]) -> Value {
    Value::Array(tasks.iter().map(task_row).collect())
}

/// Projects one list row of the `view_task` tool.
#[must_use]
fn task_row(task: &TaskSummary) -> Value {
    json!({
        "task_id": task.task_id,
        "title": task.title,
        "status": task.status.as_str(),
        "assignee": task.assignee,
    })
}

/// Reads a required string argument, naming the missing key exactly as Python does.
#[must_use = "the missing-argument error is the answer"]
fn required_str<'a>(arguments: &'a Value, key: &str) -> Result<&'a str> {
    arguments
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| Error::validation(format!("'{key}' is required")))
}

/// Parses a task status wire string (`pending`, `blocked`, `in_progress`, …).
#[must_use = "the validation error is the answer"]
fn parse_task_status(raw: &str) -> Result<TaskStatus> {
    serde_json::from_value::<TaskStatus>(Value::String(raw.to_owned()))
        .map_err(|_| Error::validation(format!("unknown task status '{raw}'")))
}

/// Parses a review verdict wire string (`pass` or `fail`).
#[must_use = "the validation error is the answer"]
fn parse_verdict(raw: &str) -> Result<Verdict> {
    serde_json::from_value::<Verdict>(Value::String(raw.to_owned()))
        .map_err(|_| Error::validation(format!("unknown review verdict '{raw}'")))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use std::sync::Arc;

    use serde_json::json;

    use openjiuwen_common::error::ErrorKind;

    use crate::runtime::team_agent::test_support::{blueprint, built};
    use crate::runtime::team_agent::TeamAgent;
    use crate::schema::TaskGraphSpec;

    use super::*;

    /// Builds one team (`alpha`, led by `boss`, one teammate `dev-1`) and its backend.
    async fn fixture() -> (Arc<TeamAgent>, Arc<TeamBackend>) {
        let (agent, _factory, _workspace) = built(&blueprint("alpha", "boss", &["dev-1"])).await;
        let backend = agent.backend();
        (agent, backend)
    }

    /// Creates task `id` with optional reviewers, unassigned.
    async fn create_task(backend: &TeamBackend, id: &str, reviewers: &[&str]) {
        backend
            .tasks()
            .create_task(TaskGraphSpec {
                title: format!("task {id}"),
                task_id: Some(id.to_owned()),
                reviewers: reviewers.iter().map(|name| (*name).to_owned()).collect(),
                created_by: Some("boss".to_owned()),
                ..TaskGraphSpec::default()
            })
            .await
            .unwrap();
    }

    /// Claims and then completes `id` as `member` through the dispatcher.
    async fn claim_and_complete(backend: &TeamBackend, id: &str, member: &str) {
        let claimed = dispatch_tool(
            backend,
            member,
            "claim_task",
            &json!({ "task_id": id, "status": "claimed" }),
        )
        .await
        .unwrap();
        assert_eq!(claimed["ok"], json!(true), "claim of '{id}' is applied");

        let completed = dispatch_tool(
            backend,
            member,
            "claim_task",
            &json!({ "task_id": id, "status": "completed" }),
        )
        .await
        .unwrap();
        assert_eq!(completed["ok"], json!(true), "completion of '{id}'");
    }

    /// Returns the `task_id` of every row of a `view_task` answer, sorted.
    fn ids(value: &Value) -> Vec<String> {
        let mut ids: Vec<String> = value
            .as_array()
            .unwrap()
            .iter()
            .map(|row| row["task_id"].as_str().unwrap().to_owned())
            .collect();
        ids.sort();
        ids
    }

    #[tokio::test]
    async fn view_task_lists_filters_and_projects_rows() {
        let (_agent, backend) = fixture().await;
        create_task(&backend, "t-1", &[]).await;

        let listed = dispatch_tool(&backend, "boss", "view_task", &json!({}))
            .await
            .unwrap();
        assert_eq!(
            listed,
            json!([{
                "task_id": "t-1",
                "title": "task t-1",
                "status": "pending",
                "assignee": null,
            }]),
            "the default action is `list` with the four projected fields"
        );

        create_task(&backend, "t-2", &[]).await;
        claim_and_complete(&backend, "t-2", "dev-1").await;
        let completed = dispatch_tool(
            &backend,
            "boss",
            "view_task",
            &json!({ "action": "list", "status": "completed" }),
        )
        .await
        .unwrap();
        assert_eq!(ids(&completed), vec!["t-2"]);
        assert_eq!(completed[0]["status"], json!("completed"));
        assert_eq!(completed[0]["assignee"], json!("dev-1"));

        create_task(&backend, "t-3", &["boss"]).await;
        claim_and_complete(&backend, "t-3", "dev-1").await;
        let in_review = dispatch_tool(
            &backend,
            "boss",
            "view_task",
            &json!({ "action": "in_review" }),
        )
        .await
        .unwrap();
        assert_eq!(ids(&in_review), vec!["t-3"]);
        assert_eq!(in_review[0]["status"], json!("in_review"));

        let claimable = dispatch_tool(
            &backend,
            "dev-1",
            "view_task",
            &json!({ "action": "claimable" }),
        )
        .await
        .unwrap();
        assert_eq!(
            claimable,
            json!([{ "task_id": "t-1", "title": "task t-1" }]),
            "claimable rows carry only id and title"
        );

        let detail = dispatch_tool(
            &backend,
            "boss",
            "view_task",
            &json!({ "action": "get", "task_id": "t-1" }),
        )
        .await
        .unwrap();
        assert_eq!(detail["task_id"], json!("t-1"));
        assert_eq!(detail["title"], json!("task t-1"));

        let unknown = dispatch_tool(
            &backend,
            "boss",
            "view_task",
            &json!({ "action": "get", "task_id": "zzz" }),
        )
        .await
        .unwrap();
        assert_eq!(unknown, Value::Null, "an unknown id answers null");

        let note = dispatch_tool(
            &backend,
            "boss",
            "view_task",
            &json!({ "action": "note", "task_id": "t-1" }),
        )
        .await
        .unwrap();
        assert_eq!(note, Value::Null);

        let bad_action = dispatch_tool(
            &backend,
            "boss",
            "view_task",
            &json!({ "action": "explode" }),
        )
        .await
        .unwrap_err();
        assert_eq!(bad_action.kind(), ErrorKind::Validation);
        assert!(bad_action.to_string().contains("unknown view_task action"));

        let bad_status = dispatch_tool(
            &backend,
            "boss",
            "view_task",
            &json!({ "action": "list", "status": "sleeping" }),
        )
        .await
        .unwrap_err();
        assert_eq!(bad_status.kind(), ErrorKind::Validation);
        assert!(bad_status
            .to_string()
            .contains("unknown task status 'sleeping'"));
    }

    #[tokio::test]
    async fn claim_task_claims_once_and_refuses_a_second_owner() {
        let (_agent, backend) = fixture().await;
        create_task(&backend, "t-1", &[]).await;

        let claimed = dispatch_tool(
            &backend,
            "dev-1",
            "claim_task",
            &json!({ "task_id": "t-1", "status": "claimed" }),
        )
        .await
        .unwrap();
        assert_eq!(claimed, json!({ "ok": true, "reason": null }));

        let refused = dispatch_tool(
            &backend,
            "boss",
            "claim_task",
            &json!({ "task_id": "t-1", "status": "claimed" }),
        )
        .await
        .unwrap();
        assert_eq!(refused["ok"], json!(false));
        assert!(
            refused["reason"]
                .as_str()
                .unwrap()
                .contains("already claimed by 'dev-1'"),
            "the refusal names the current owner: {refused}"
        );

        let bad_status = dispatch_tool(
            &backend,
            "dev-1",
            "claim_task",
            &json!({ "task_id": "t-1", "status": "reopened" }),
        )
        .await
        .unwrap_err();
        assert_eq!(bad_status.kind(), ErrorKind::Validation);
        assert!(bad_status
            .to_string()
            .contains("unknown claim_task status 'reopened'"));
    }

    #[tokio::test]
    async fn claim_task_completes_only_for_the_owner() {
        let (_agent, backend) = fixture().await;
        create_task(&backend, "t-1", &[]).await;

        let claimed = dispatch_tool(
            &backend,
            "dev-1",
            "claim_task",
            &json!({ "task_id": "t-1", "status": "claimed" }),
        )
        .await
        .unwrap();
        assert_eq!(claimed["ok"], json!(true));

        let completed = dispatch_tool(
            &backend,
            "dev-1",
            "claim_task",
            &json!({ "task_id": "t-1", "status": "completed" }),
        )
        .await
        .unwrap();
        assert_eq!(completed, json!({ "ok": true, "reason": null }));
        let detail = backend.tasks().detail("t-1").await.unwrap().unwrap();
        assert_eq!(detail.status, TaskStatus::Completed);

        create_task(&backend, "t-2", &[]).await;
        let claimed = dispatch_tool(
            &backend,
            "dev-1",
            "claim_task",
            &json!({ "task_id": "t-2", "status": "claimed" }),
        )
        .await
        .unwrap();
        assert_eq!(claimed["ok"], json!(true));

        let refused = dispatch_tool(
            &backend,
            "boss",
            "claim_task",
            &json!({ "task_id": "t-2", "status": "completed" }),
        )
        .await
        .unwrap();
        assert_eq!(
            refused["ok"],
            json!(false),
            "a board refusal is an answer, not an error"
        );
        assert!(refused["reason"]
            .as_str()
            .unwrap()
            .contains("task is assigned to 'dev-1'"));
        assert_eq!(
            backend.tasks().detail("t-2").await.unwrap().unwrap().status,
            TaskStatus::InProgress,
            "the refused completion left the row untouched"
        );
    }

    #[tokio::test]
    async fn verify_task_answers_refusals_and_applies_reviewer_verdicts() {
        let (_agent, backend) = fixture().await;
        create_task(&backend, "t-1", &["boss"]).await;

        let not_under_review = dispatch_tool(
            &backend,
            "boss",
            "verify_task",
            &json!({ "task_id": "t-1", "decision": "pass" }),
        )
        .await
        .unwrap();
        assert_eq!(not_under_review["ok"], json!(false));
        assert!(not_under_review["reason"]
            .as_str()
            .unwrap()
            .contains("not under review"));

        claim_and_complete(&backend, "t-1", "dev-1").await;

        let not_a_reviewer = dispatch_tool(
            &backend,
            "dev-1",
            "verify_task",
            &json!({ "task_id": "t-1", "decision": "pass" }),
        )
        .await
        .unwrap();
        assert_eq!(not_a_reviewer["ok"], json!(false));
        assert!(not_a_reviewer["reason"]
            .as_str()
            .unwrap()
            .contains("not a reviewer"));

        let bad_decision = dispatch_tool(
            &backend,
            "boss",
            "verify_task",
            &json!({ "task_id": "t-1", "decision": "maybe" }),
        )
        .await
        .unwrap_err();
        assert_eq!(bad_decision.kind(), ErrorKind::Validation);
        assert!(bad_decision
            .to_string()
            .contains("unknown review verdict 'maybe'"));

        let passed = dispatch_tool(
            &backend,
            "boss",
            "verify_task",
            &json!({ "task_id": "t-1", "decision": "pass", "feedback": "looks good" }),
        )
        .await
        .unwrap();
        assert_eq!(
            passed,
            json!({
                "ok": true,
                "task_id": "t-1",
                "decision": "pass",
                "feedback": "looks good",
            })
        );
        assert_eq!(
            backend.tasks().detail("t-1").await.unwrap().unwrap().status,
            TaskStatus::Completed
        );
    }

    #[tokio::test]
    async fn send_message_writes_one_readable_mailbox_row() {
        let (_agent, backend) = fixture().await;

        let sent = dispatch_tool(
            &backend,
            "boss",
            "send_message",
            &json!({ "to": "dev-1", "content": "please pick up t-1" }),
        )
        .await
        .unwrap();
        assert_eq!(sent["ok"], json!(true));
        assert_eq!(sent["reason"], Value::Null);
        let message_id = sent["message_id"].as_str().unwrap();
        assert_ne!(message_id, "");

        let unread = backend.unread("dev-1").await.unwrap();
        assert_eq!(unread.len(), 1);
        assert_eq!(unread[0].message_id, message_id);
        assert_eq!(unread[0].from_member_name, "boss");
        assert_eq!(unread[0].to_member_name.as_deref(), Some("dev-1"));
        assert_eq!(unread[0].content, "please pick up t-1");
    }

    #[tokio::test]
    async fn unknown_tools_and_missing_arguments_are_typed_errors() {
        let (_agent, backend) = fixture().await;

        let unknown = dispatch_tool(&backend, "boss", "shell", &json!({}))
            .await
            .unwrap_err();
        assert_eq!(unknown.kind(), ErrorKind::NotFound);
        assert!(unknown.to_string().contains("unknown tool 'shell'"));

        for name in TEAM_TOOL_NAMES {
            assert!(is_supported_tool(name), "'{name}' is a team tool");
        }
        assert!(!is_supported_tool("shell"));

        let missing = dispatch_tool(&backend, "boss", "claim_task", &json!({}))
            .await
            .unwrap_err();
        assert_eq!(missing.kind(), ErrorKind::Validation);
        assert!(missing.to_string().contains("'task_id' is required"));

        let missing_content =
            dispatch_tool(&backend, "boss", "send_message", &json!({ "to": "dev-1" }))
                .await
                .unwrap_err();
        assert_eq!(missing_content.kind(), ErrorKind::Validation);
        assert!(missing_content
            .to_string()
            .contains("'content' is required"));
    }

    #[test]
    fn render_tool_output_passes_strings_through_and_compacts_everything_else() {
        let text = json!("plain answer");
        assert_eq!(render_tool_output(&text), json!("plain answer"));

        let object = json!({ "ok": true, "task_id": "t-1" });
        assert_eq!(
            render_tool_output(&object),
            json!("{\"ok\":true,\"task_id\":\"t-1\"}")
        );
        assert_eq!(render_tool_output(&Value::Null), json!("null"));
    }
}
