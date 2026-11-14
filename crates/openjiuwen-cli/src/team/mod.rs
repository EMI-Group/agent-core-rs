//! `openjiuwen team` — operate a team hosted by another process.
//!
//! Cross-process teams meet over the `openjiuwen-agent-teams` attach protocol
//! (one JSON request/response per line on a Unix domain socket). This module is
//! a thin client: it resolves the socket, dials it with
//! [`openjiuwen_agent_teams::AttachClient`], maps each subcommand onto one
//! attach operation, and renders the answer. The team itself — its members, task
//! board, mailbox and leader — runs in the hosting process.
//!
//! The socket comes from `--socket`, else `OPENJIUWEN_TEAM_SOCKET`. The command
//! vocabulary follows the `team-member` binary's operator scope (`send`,
//! `broadcast`, `create-task`, `task list|claimable|get`, `claim`, `complete`,
//! `update`, `members`, `inbox`) and adds the two operations only the attach
//! protocol has: `snapshot` and `stop`.
//!
//! # Connection model
//!
//! One process opens one connection and performs exactly one operation: there is
//! no reconnect loop and no retry, so a failure is reported as the error it is.
//! [`AttachError::into_error`] preserves the wire kind, which is why a remote
//! `not_found` (unknown member or task) reaches the process exit path as
//! `error: not_found: …` instead of a generic transport failure.
//!
//! # Output
//!
//! `--json` (group-global) prints exactly one pretty JSON document on stdout:
//! the member array of `members`, one `MemberSnapshot` for `member`, the text of
//! `inbox`, the operation payload of `send` / `broadcast` / `create-task` /
//! `claim` / `complete` / `update` / `stop`, the task array of `task list` /
//! `task claimable`, one `TaskDetail` for `task get`, and the whole
//! `TeamSnapshot` for `snapshot`.
//!
//! Text mode prints one item per line:
//!
//! - `members` / `member`: `<member_name>\t<role>\t<status>\t<execution>`, plus
//!   a fifth `<display_name>` column when the roster carries one that differs
//!   from the member name; an empty roster prints `no team members`;
//! - `inbox`: the text exactly as the host returned it, or `inbox is empty`;
//! - `send` / `broadcast`: `<message_id> from <from> to <to|*>`, or the compact
//!   JSON payload when the host answered something else;
//! - `create-task` / `update`: `<task_id>\t<status>\t<title>` when the payload
//!   carries those three fields, else the compact JSON payload;
//! - `task list` / `task claimable`: `<task_id>\t<status>\t<title>` plus
//!   `\t<assignee>` when the task has an owner, or `no tasks` /
//!   `no claimable tasks` when the board is empty;
//! - `task get`: `task_id`, `title`, `status` and `assignee` rows, then the task
//!   content;
//! - `claim` / `complete`: `claimed <id>` / `completed <id>`;
//! - `snapshot`: `team <name> (<state>): N member(s), M task(s)` followed by one
//!   line per member and per task;
//! - `stop`: `stop requested`.
//!
//! A payload carrying `ok: false` — the task board's policy answer, e.g. a claim
//! of a task somebody else owns — is a `tool` error carrying the host's
//! `reason`, never a successful exit with JSON on stdout, in both output modes.
use std::path::PathBuf;

use openjiuwen::agent_teams::skill::descriptor::TEAM_SOCKET_ENV;
use openjiuwen::agent_teams::{
    AttachClient, AttachError, HarnessState, MemberSnapshot, TaskStatus, TaskSummary,
};
use openjiuwen_common::error::{Error, Result};
use serde_json::Value;

/// `openjiuwen team [--socket PATH] [--json] <COMMAND>`.
#[derive(Debug, Clone, clap::Args)]
pub(crate) struct TeamCli {
    /// Attach socket of the hosting team.
    ///
    /// Falls back to `OPENJIUWEN_TEAM_SOCKET` when absent.
    #[arg(long, global = true, value_name = "PATH")]
    pub(crate) socket: Option<PathBuf>,
    /// Print machine-readable JSON instead of text.
    #[arg(long, global = true)]
    pub(crate) json: bool,
    /// The team operation to run.
    #[command(subcommand)]
    pub(crate) command: TeamCommand,
}

/// Attach operations a terminal user needs.
#[derive(Debug, Clone, clap::Subcommand)]
pub(crate) enum TeamCommand {
    /// List the team's members.
    Members,
    /// Show one member.
    Member {
        /// Member name.
        name: String,
    },
    /// Read and drain the leader's unread inbox.
    Inbox,
    /// Send a message to one member.
    Send {
        /// Recipient member name.
        to: String,
        /// Message text.
        content: String,
    },
    /// Send a message to every member.
    Broadcast {
        /// Message text.
        content: String,
    },
    /// Create a task on the team's board.
    CreateTask {
        /// Task title.
        title: String,
        /// Task content.
        content: String,
        /// Explicit task id; the team assigns one when omitted.
        #[arg(long = "task-id")]
        task_id: Option<String>,
        /// Id of a task this one is blocked by; repeatable.
        #[arg(long = "dep")]
        dependencies: Vec<String>,
    },
    /// Task-board reads.
    Task {
        /// The read to perform.
        #[command(subcommand)]
        command: TeamTaskCommand,
    },
    /// Claim a task for this caller.
    Claim {
        /// Task id.
        task_id: String,
    },
    /// Mark a claimed task completed.
    Complete {
        /// Task id.
        task_id: String,
    },
    /// Patch a task's title or content.
    Update {
        /// Task id.
        task_id: String,
        /// New title.
        #[arg(long)]
        title: Option<String>,
        /// New content.
        #[arg(long)]
        content: Option<String>,
    },
    /// Print the whole team snapshot (members, tasks, state).
    Snapshot,
    /// Ask the hosting team to stop.
    Stop,
}

/// Task-board reads.
#[derive(Debug, Clone, clap::Subcommand)]
pub(crate) enum TeamTaskCommand {
    /// List tasks, optionally filtered by status.
    List {
        /// Status filter: `pending`, `blocked`, `planning`, `in_progress`,
        /// `in_review`, `completed` or `cancelled`.
        #[arg(long)]
        status: Option<String>,
    },
    /// List the tasks this caller may claim.
    Claimable,
    /// Show one task in full.
    Get {
        /// Task id.
        task_id: String,
    },
}

/// Runs `openjiuwen team`.
///
/// # Errors
///
/// Returns a validation error when no socket is configured, the attach
/// transport's typed errors, and remote failures with their wire kind (for
/// example `not_found` for an unknown member or task).
pub(crate) async fn run(args: &TeamCli) -> Result<()> {
    let socket = resolve_socket(args)?;
    let mut client = AttachClient::connect(&socket)
        .await
        .map_err(AttachError::into_error)?;
    let json = args.json;
    match &args.command {
        TeamCommand::Members => members(&mut client, json).await,
        TeamCommand::Member { name } => member(&mut client, name, json).await,
        TeamCommand::Inbox => inbox(&mut client, json).await,
        TeamCommand::Send { to, content } => send(&mut client, to, content, json).await,
        TeamCommand::Broadcast { content } => broadcast(&mut client, content, json).await,
        TeamCommand::CreateTask {
            title,
            content,
            task_id,
            dependencies,
        } => {
            create_task(
                &mut client,
                title,
                content,
                task_id.as_deref(),
                dependencies,
                json,
            )
            .await
        }
        TeamCommand::Task { command } => match command {
            TeamTaskCommand::List { status } => {
                list_tasks(&mut client, status.as_deref(), json).await
            }
            TeamTaskCommand::Claimable => claimable(&mut client, json).await,
            TeamTaskCommand::Get { task_id } => task_get(&mut client, task_id, json).await,
        },
        TeamCommand::Claim { task_id } => {
            let value = client
                .claim_task(task_id)
                .await
                .map_err(AttachError::into_error)?;
            render_task_op(&value, "claimed", task_id, json)
        }
        TeamCommand::Complete { task_id } => {
            let value = client
                .complete_task(task_id)
                .await
                .map_err(AttachError::into_error)?;
            render_task_op(&value, "completed", task_id, json)
        }
        TeamCommand::Update {
            task_id,
            title,
            content,
        } => {
            update(
                &mut client,
                task_id,
                title.as_deref(),
                content.as_deref(),
                json,
            )
            .await
        }
        TeamCommand::Snapshot => snapshot(&mut client, json).await,
        TeamCommand::Stop => stop(&mut client, json).await,
    }
}

/// Resolves the attach socket from `--socket`, else the documented env var.
///
/// # Errors
///
/// Returns a validation error naming both sources when neither is set to a
/// non-empty value.
fn resolve_socket(args: &TeamCli) -> Result<PathBuf> {
    if let Some(socket) = &args.socket {
        if !socket.as_os_str().is_empty() {
            return Ok(socket.clone());
        }
    }
    match std::env::var(TEAM_SOCKET_ENV) {
        Ok(value) if !value.trim().is_empty() => Ok(PathBuf::from(value.trim())),
        _ => Err(Error::validation(format!(
            "team socket is required: pass --socket <PATH> or set {TEAM_SOCKET_ENV}"
        ))),
    }
}

/// Prints the roster.
async fn members(client: &mut AttachClient, json: bool) -> Result<()> {
    let members = client
        .list_members()
        .await
        .map_err(AttachError::into_error)?;
    if json {
        return print_json(&members);
    }
    if members.is_empty() {
        println!("no team members");
        return Ok(());
    }
    for member in &members {
        println!("{}", member_line(member));
    }
    Ok(())
}

/// Prints one roster row.
async fn member(client: &mut AttachClient, name: &str, json: bool) -> Result<()> {
    let member = client
        .member_status(name)
        .await
        .map_err(AttachError::into_error)?;
    if json {
        return print_json(&member);
    }
    println!("{}", member_line(&member));
    Ok(())
}

/// Prints the leader's unread inbox.
async fn inbox(client: &mut AttachClient, json: bool) -> Result<()> {
    let text = client.read_inbox().await.map_err(AttachError::into_error)?;
    if json {
        return print_json(&text);
    }
    if text.trim().is_empty() {
        println!("inbox is empty");
        return Ok(());
    }
    if text.ends_with('\n') {
        print!("{text}");
    } else {
        println!("{text}");
    }
    Ok(())
}

/// Sends a direct message.
async fn send(client: &mut AttachClient, to: &str, content: &str, json: bool) -> Result<()> {
    let value = client
        .send_member(to, content)
        .await
        .map_err(AttachError::into_error)?;
    render_message(&value, json)
}

/// Sends a message to every member.
async fn broadcast(client: &mut AttachClient, content: &str, json: bool) -> Result<()> {
    let value = client
        .broadcast(content)
        .await
        .map_err(AttachError::into_error)?;
    render_message(&value, json)
}

/// Creates a task on the board.
async fn create_task(
    client: &mut AttachClient,
    title: &str,
    content: &str,
    task_id: Option<&str>,
    dependencies: &[String],
    json: bool,
) -> Result<()> {
    let value = client
        .create_task(title, content, task_id, dependencies)
        .await
        .map_err(AttachError::into_error)?;
    render_task_record(&value, json)
}

/// Lists tasks, optionally filtered by status.
async fn list_tasks(client: &mut AttachClient, status: Option<&str>, json: bool) -> Result<()> {
    let filter = status.map(parse_status).transpose()?;
    let tasks = client
        .list_tasks(filter)
        .await
        .map_err(AttachError::into_error)?;
    render_tasks(&tasks, "no tasks", json)
}

/// Lists the tasks this caller may claim.
async fn claimable(client: &mut AttachClient, json: bool) -> Result<()> {
    let tasks = client
        .claimable_tasks()
        .await
        .map_err(AttachError::into_error)?;
    render_tasks(&tasks, "no claimable tasks", json)
}

/// Prints one task in full.
async fn task_get(client: &mut AttachClient, task_id: &str, json: bool) -> Result<()> {
    let detail = client
        .task_detail(task_id)
        .await
        .map_err(AttachError::into_error)?
        .ok_or_else(|| Error::not_found(format!("task not found: {task_id}")))?;
    if json {
        return print_json(&detail);
    }
    println!("task_id\t{}", detail.task_id);
    println!("title\t{}", detail.title);
    println!("status\t{}", detail.status.as_str());
    println!("assignee\t{}", detail.assignee.as_deref().unwrap_or("-"));
    println!("{}", detail.content);
    Ok(())
}

/// Patches a task's title and/or content.
async fn update(
    client: &mut AttachClient,
    task_id: &str,
    title: Option<&str>,
    content: Option<&str>,
    json: bool,
) -> Result<()> {
    let value = client
        .update_task(task_id, title, content)
        .await
        .map_err(AttachError::into_error)?;
    render_task_record(&value, json)
}

/// Prints the whole team snapshot.
async fn snapshot(client: &mut AttachClient, json: bool) -> Result<()> {
    let snapshot = client.snapshot().await.map_err(AttachError::into_error)?;
    if json {
        return print_json(&snapshot);
    }
    println!(
        "team {} ({}): {} member(s), {} task(s)",
        snapshot.team_name,
        state_word(snapshot.state),
        snapshot.members.len(),
        snapshot.tasks.len()
    );
    for member in &snapshot.members {
        println!("{}", member_line(member));
    }
    for task in &snapshot.tasks {
        println!("{}", task_line(task));
    }
    Ok(())
}

/// Asks the hosting team to stop.
async fn stop(client: &mut AttachClient, json: bool) -> Result<()> {
    let value = client.stop_team().await.map_err(AttachError::into_error)?;
    if json {
        return print_json(&value);
    }
    println!("stop requested");
    Ok(())
}

/// Parses a `--status` value, which the schema type deliberately does not own.
///
/// # Errors
///
/// Returns a validation error naming the accepted statuses.
fn parse_status(raw: &str) -> Result<TaskStatus> {
    serde_json::from_str::<TaskStatus>(&format!("\"{raw}\"")).map_err(|_| {
        Error::validation(format!(
            "unknown task status '{raw}'; expected pending, blocked, planning, in_progress, in_review, completed or cancelled"
        ))
    })
}

/// Renders a message-send payload.
fn render_message(value: &Value, json: bool) -> Result<()> {
    refuse(value)?;
    if json {
        return print_json(value);
    }
    match message_line(value) {
        Some(line) => println!("{line}"),
        None => print_compact(value)?,
    }
    Ok(())
}

/// Renders a task payload that may be a full task record.
fn render_task_record(value: &Value, json: bool) -> Result<()> {
    refuse(value)?;
    if json {
        return print_json(value);
    }
    match task_record_line(value) {
        Some(line) => println!("{line}"),
        None => print_compact(value)?,
    }
    Ok(())
}

/// Renders a claim/complete payload.
fn render_task_op(value: &Value, done: &str, task_id: &str, json: bool) -> Result<()> {
    refuse(value)?;
    if json {
        return print_json(value);
    }
    if outcome(value) == Some(true) {
        println!("{done} {task_id}");
    } else {
        print_compact(value)?;
    }
    Ok(())
}

/// Prints a task list, or its empty marker.
fn render_tasks(tasks: &[TaskSummary], empty: &str, json: bool) -> Result<()> {
    if json {
        return print_json(&tasks);
    }
    if tasks.is_empty() {
        println!("{empty}");
        return Ok(());
    }
    for task in tasks {
        println!("{}", task_line(task));
    }
    Ok(())
}

/// Turns a policy refusal (`ok: false`) into a `tool` error.
fn refuse(value: &Value) -> Result<()> {
    if outcome(value) == Some(false) {
        return Err(Error::tool(refusal_reason(value)));
    }
    Ok(())
}

/// Reads the `ok` flag of an operation outcome.
fn outcome(value: &Value) -> Option<bool> {
    value.get("ok").and_then(Value::as_bool)
}

/// Reads the host's `reason`, falling back to a short explanation.
fn refusal_reason(value: &Value) -> String {
    value
        .get("reason")
        .and_then(Value::as_str)
        .filter(|reason| !reason.trim().is_empty())
        .map_or_else(
            || "the team refused the request without a reason".to_owned(),
            ToOwned::to_owned,
        )
}

/// Renders a persisted message record as `<id> from <from> to <to|*>`.
fn message_line(value: &Value) -> Option<String> {
    let message_id = value.get("message_id")?.as_str()?;
    let from = value.get("from_member_name")?.as_str()?;
    let to = value
        .get("to_member_name")
        .and_then(Value::as_str)
        .unwrap_or("*");
    Some(format!("{message_id} from {from} to {to}"))
}

/// Renders a task record as `<task_id>\t<status>\t<title>`.
fn task_record_line(value: &Value) -> Option<String> {
    let task_id = value.get("task_id")?.as_str()?;
    let title = value.get("title")?.as_str()?;
    let status = value.get("status")?.as_str()?;
    Some(format!("{task_id}\t{status}\t{title}"))
}

/// Renders one roster row as `<name>\t<role>\t<status>\t<execution>[\t<display>]`.
fn member_line(member: &MemberSnapshot) -> String {
    let mut line = format!(
        "{}\t{}\t{}\t{}",
        member.member_name,
        member.role.as_str(),
        member.status.as_str(),
        member.execution.as_str()
    );
    if !member.display_name.is_empty() && member.display_name != member.member_name {
        line.push('\t');
        line.push_str(&member.display_name);
    }
    line
}

/// Renders one task row as `<task_id>\t<status>\t<title>[\t<assignee>]`.
fn task_line(task: &TaskSummary) -> String {
    let mut line = format!("{}\t{}\t{}", task.task_id, task.status.as_str(), task.title);
    if let Some(assignee) = &task.assignee {
        line.push('\t');
        line.push_str(assignee);
    }
    line
}

/// Returns the wire word of the team lifecycle state.
fn state_word(state: HarnessState) -> &'static str {
    match state {
        HarnessState::Idle => "idle",
        HarnessState::Running => "running",
        HarnessState::Pausing => "pausing",
        HarnessState::Paused => "paused",
        HarnessState::Terminated => "terminated",
    }
}

/// Prints one pretty JSON document.
fn print_json(value: &impl serde::Serialize) -> Result<()> {
    println!("{}", serde_json::to_string_pretty(value)?);
    Ok(())
}

/// Prints one compact JSON document.
fn print_compact(value: &Value) -> Result<()> {
    println!("{}", serde_json::to_string(value)?);
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parse_status_accepts_every_wire_word() {
        for (raw, expected) in [
            ("pending", TaskStatus::Pending),
            ("blocked", TaskStatus::Blocked),
            ("planning", TaskStatus::Planning),
            ("in_progress", TaskStatus::InProgress),
            ("in_review", TaskStatus::InReview),
            ("completed", TaskStatus::Completed),
            ("cancelled", TaskStatus::Cancelled),
        ] {
            assert_eq!(parse_status(raw).unwrap(), expected);
        }
    }

    #[test]
    fn parse_status_names_the_known_statuses() {
        let error = parse_status("done").unwrap_err();
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Validation
        );
        let message = error.to_string();
        assert!(message.contains("unknown task status 'done'"), "{message}");
        assert!(message.contains("in_progress"), "{message}");
    }

    #[test]
    fn a_message_record_renders_from_and_to() {
        let record = json!({
            "message_id": "msg_1",
            "from_member_name": "leader",
            "to_member_name": "dev-1",
        });
        assert_eq!(
            message_line(&record).as_deref(),
            Some("msg_1 from leader to dev-1")
        );
        let broadcast = json!({
            "message_id": "msg_2",
            "from_member_name": "leader",
            "to_member_name": null,
        });
        assert_eq!(
            message_line(&broadcast).as_deref(),
            Some("msg_2 from leader to *")
        );
    }

    #[test]
    fn an_outcome_payload_without_names_has_no_message_line() {
        assert!(message_line(&json!({"ok": true, "message_id": "msg_1"})).is_none());
        assert!(task_record_line(&json!({"ok": true, "task_id": "t1"})).is_none());
        assert_eq!(
            task_record_line(&json!({"task_id": "t1", "title": "docs", "status": "pending"}))
                .as_deref(),
            Some("t1\tpending\tdocs")
        );
    }

    #[test]
    fn a_refusal_becomes_a_tool_error_with_the_host_reason() {
        let error = refuse(&json!({"ok": false, "reason": "task t1 is not pending"})).unwrap_err();
        assert_eq!(error.kind(), openjiuwen_common::error::ErrorKind::Tool);
        assert!(error.to_string().contains("task t1 is not pending"));
        let fallback = refuse(&json!({"ok": false})).unwrap_err();
        assert!(fallback.to_string().contains("without a reason"));
        assert!(refuse(&json!({"ok": true})).is_ok());
        assert!(refuse(&json!({"stopping": true})).is_ok());
    }

    #[test]
    fn member_lines_append_the_display_name_only_when_it_adds_information() {
        let mut member = MemberSnapshot {
            member_name: "leader".to_owned(),
            display_name: "Ada".to_owned(),
            ..MemberSnapshot::default()
        };
        assert_eq!(member_line(&member), "leader\tleader\tunstarted\tidle\tAda");
        member.display_name = "leader".to_owned();
        assert_eq!(member_line(&member), "leader\tleader\tunstarted\tidle");
        member.display_name.clear();
        assert_eq!(member_line(&member), "leader\tleader\tunstarted\tidle");
    }

    #[test]
    fn task_lines_carry_the_assignee_only_when_owned() {
        let owned = TaskSummary {
            task_id: "t1".to_owned(),
            title: "docs".to_owned(),
            status: TaskStatus::InProgress,
            assignee: Some("dev-1".to_owned()),
            ..TaskSummary::default()
        };
        assert_eq!(task_line(&owned), "t1\tin_progress\tdocs\tdev-1");
        let unowned = TaskSummary {
            assignee: None,
            ..owned
        };
        assert_eq!(task_line(&unowned), "t1\tin_progress\tdocs");
    }
}
