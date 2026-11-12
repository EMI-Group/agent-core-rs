//! Operator-scope dispatch: the outside team-control surface.
//!
//! Python parity source: `_dispatch_operator` and `_dispatch_operator_task` in
//! `openjiuwen/agent_teams/skill/cli.py`. An operator is not a team member: it has no claim loop
//! and no tool set, it controls the task board, the mailbox and the roster explicitly through
//! [`TeamClient`]. Every acknowledgement line and refusal is rendered by [`super::output`].
//!
//! `--watch` is accepted for parity but performs a single inbox read (streaming is deferred).

use std::io::Write;
use std::sync::Arc;

use openjiuwen_common::error::Result;

use crate::constants::BROADCAST_TARGET;

use super::super::client::{OpOutcome, TeamClient};
use super::output::{
    broadcast_line, claimable_line, claimed_line, completed_line, created_line,
    format_status_filter, member_line, sent_line, task_detail_block, task_list_line, updated_line,
};
use super::{
    emit_line, emit_lines, fail, refuse, CliCommand, OperatorCommand, SkillCli, TaskCommand,
};

/// Dispatches one operator-scope command, returning the process exit code.
pub(crate) async fn dispatch(
    cli: &SkillCli,
    client: Arc<dyn TeamClient>,
    out: &mut dyn Write,
    err: &mut dyn Write,
) -> i32 {
    let CliCommand::Operator(command) = cli.command() else {
        let _ = writeln!(err, "internal: the operator command tree was not selected");
        return 2;
    };
    match command {
        OperatorCommand::Inbox { watch } => inbox(client.as_ref(), out, err, *watch).await,
        OperatorCommand::Send { to, content } => send(client.as_ref(), out, err, to, content).await,
        OperatorCommand::Broadcast { content } => {
            broadcast(client.as_ref(), out, err, content).await
        }
        OperatorCommand::CreateTask {
            title,
            content,
            task_id,
        } => {
            create_task(
                client.as_ref(),
                out,
                err,
                title,
                content,
                task_id.as_deref(),
            )
            .await
        }
        OperatorCommand::Task { action } => task(client.as_ref(), out, err, action).await,
        OperatorCommand::Claim { task_id } => claim(client.as_ref(), out, err, task_id).await,
        OperatorCommand::Complete { task_id } => complete(client.as_ref(), out, err, task_id).await,
        OperatorCommand::Update {
            task_id,
            title,
            content,
        } => {
            update(
                client.as_ref(),
                out,
                err,
                task_id,
                title.as_deref(),
                content.as_deref(),
            )
            .await
        }
        OperatorCommand::Members => members(client.as_ref(), out, err).await,
    }
}

/// Renders the operator's unread mailbox (`--watch` performs the same single read).
async fn inbox(
    client: &dyn TeamClient,
    out: &mut dyn Write,
    err: &mut dyn Write,
    _watch: bool,
) -> i32 {
    match client.read_inbox().await {
        Ok(text) => emit_line(out, err, &text),
        Err(error) => fail(err, &error),
    }
}

/// Sends a direct message to `to`.
async fn send(
    client: &dyn TeamClient,
    out: &mut dyn Write,
    err: &mut dyn Write,
    to: &str,
    content: &str,
) -> i32 {
    match client.send_message(to, content).await {
        Ok(outcome) if outcome.ok => {
            let message_id = outcome.message_id.as_deref().unwrap_or_default();
            emit_line(out, err, &sent_line(message_id, to))
        }
        Ok(outcome) => refuse(err, outcome.reason.as_deref()),
        Err(error) => fail(err, &error),
    }
}

/// Broadcasts `content` to the whole team.
async fn broadcast(
    client: &dyn TeamClient,
    out: &mut dyn Write,
    err: &mut dyn Write,
    content: &str,
) -> i32 {
    match client.send_message(BROADCAST_TARGET, content).await {
        Ok(outcome) if outcome.ok => {
            let message_id = outcome.message_id.as_deref().unwrap_or_default();
            emit_line(out, err, &broadcast_line(message_id))
        }
        Ok(outcome) => refuse(err, outcome.reason.as_deref()),
        Err(error) => fail(err, &error),
    }
}

/// Creates a task, optionally pinned to `task_id`.
async fn create_task(
    client: &dyn TeamClient,
    out: &mut dyn Write,
    err: &mut dyn Write,
    title: &str,
    content: &str,
    task_id: Option<&str>,
) -> i32 {
    match client.create_task(title, content, task_id, &[]).await {
        Ok(outcome) if outcome.ok => {
            let created = outcome.task_id.as_deref().unwrap_or_default();
            emit_line(out, err, &created_line(created))
        }
        Ok(outcome) => refuse(err, outcome.reason.as_deref()),
        Err(error) => fail(err, &error),
    }
}

/// Runs one `task` subcommand (list, claimable or get).
async fn task(
    client: &dyn TeamClient,
    out: &mut dyn Write,
    err: &mut dyn Write,
    action: &TaskCommand,
) -> i32 {
    match action {
        TaskCommand::List { status } => task_list(client, out, err, status.as_deref()).await,
        TaskCommand::Claimable => match client.claimable_tasks().await {
            Ok(tasks) => {
                let lines: Vec<String> = tasks.iter().map(claimable_line).collect();
                emit_lines(out, err, &lines)
            }
            Err(error) => fail(err, &error),
        },
        TaskCommand::Get { task_id } => match client.get_task(task_id).await {
            Ok(Some(detail)) => emit_line(out, err, &task_detail_block(&detail)),
            Ok(None) => {
                let _ = writeln!(err, "Task '{task_id}' not found");
                1
            }
            Err(error) => fail(err, &error),
        },
    }
}

/// Lists the board, optionally filtered by a status wire string.
async fn task_list(
    client: &dyn TeamClient,
    out: &mut dyn Write,
    err: &mut dyn Write,
    status: Option<&str>,
) -> i32 {
    let filter = match status.map(format_status_filter).transpose() {
        Ok(filter) => filter,
        Err(error) => return fail(err, &error),
    };
    match client.list_tasks(filter).await {
        Ok(tasks) => {
            let lines: Vec<String> = tasks.iter().map(task_list_line).collect();
            emit_lines(out, err, &lines)
        }
        Err(error) => fail(err, &error),
    }
}

/// Claims a pending task for the operator's member identity.
async fn claim(
    client: &dyn TeamClient,
    out: &mut dyn Write,
    err: &mut dyn Write,
    task_id: &str,
) -> i32 {
    report_outcome(
        out,
        err,
        client.claim_task(task_id).await,
        &claimed_line(task_id),
    )
}

/// Completes a task owned by the operator's member identity.
async fn complete(
    client: &dyn TeamClient,
    out: &mut dyn Write,
    err: &mut dyn Write,
    task_id: &str,
) -> i32 {
    report_outcome(
        out,
        err,
        client.complete_task(task_id).await,
        &completed_line(task_id),
    )
}

/// Edits a task's title and/or content.
async fn update(
    client: &dyn TeamClient,
    out: &mut dyn Write,
    err: &mut dyn Write,
    task_id: &str,
    title: Option<&str>,
    content: Option<&str>,
) -> i32 {
    report_outcome(
        out,
        err,
        client.update_task(task_id, title, content).await,
        &updated_line(task_id),
    )
}

/// Lists the roster in the order the client returns it.
async fn members(client: &dyn TeamClient, out: &mut dyn Write, err: &mut dyn Write) -> i32 {
    match client.list_members().await {
        Ok(members) => {
            let lines: Vec<String> = members.iter().map(member_line).collect();
            emit_lines(out, err, &lines)
        }
        Err(error) => fail(err, &error),
    }
}

/// Writes `ok_line` when the operation applied, else the refusal reason and exit code `1`.
fn report_outcome(
    out: &mut dyn Write,
    err: &mut dyn Write,
    result: Result<OpOutcome>,
    ok_line: &str,
) -> i32 {
    match result {
        Ok(outcome) if outcome.ok => emit_line(out, err, ok_line),
        Ok(outcome) => refuse(err, outcome.reason.as_deref()),
        Err(error) => fail(err, &error),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use crate::constants::MAX_MESSAGE_CONTENT_CHARS;
    use crate::schema::{TaskStatus, TeamRole};
    use crate::skill::ClientScope;

    use super::super::test_support::CliFixture;
    use super::*;

    /// Builds an operator fixture joined as member `user`.
    fn fixture() -> CliFixture {
        CliFixture::new(ClientScope::Operator, "user")
    }

    /// Parses `words` in the operator tree.
    fn parse(words: &[&str]) -> SkillCli {
        let mut argv = vec!["team-member".to_string()];
        argv.extend(words.iter().map(|word| (*word).to_string()));
        SkillCli::try_parse_from(argv, ClientScope::Operator).unwrap()
    }

    /// Runs one operator-scope invocation and returns `(exit code, stdout, stderr)`.
    async fn run(fixture: &CliFixture, words: &[&str]) -> (i32, String, String) {
        let cli = parse(words);
        let mut out: Vec<u8> = Vec::new();
        let mut err: Vec<u8> = Vec::new();
        let code = cli
            .run_with(fixture.client.clone(), &mut out, &mut err)
            .await;
        (
            code,
            String::from_utf8(out).unwrap(),
            String::from_utf8(err).unwrap(),
        )
    }

    #[tokio::test]
    async fn inbox_reads_the_mailbox_once() {
        let fixture = fixture();
        assert_eq!(
            run(&fixture, &["inbox"]).await,
            (0, "(inbox empty)\n".to_string(), String::new())
        );

        fixture
            .client
            .send_message("user", "status?")
            .await
            .unwrap();
        let (code, out, _) = run(&fixture, &["inbox", "--watch"]).await;
        assert_eq!(code, 0);
        assert!(out.contains("-> user : status?"), "unexpected inbox: {out}");
        assert_eq!(
            run(&fixture, &["inbox"]).await.1,
            "(inbox empty)\n",
            "the read consumed the row, so --watch is one-shot"
        );
    }

    #[tokio::test]
    async fn send_and_broadcast_persist_and_acknowledge() {
        let fixture = fixture();
        let (code, out, err) = run(&fixture, &["send", "dev-1", "start now"]).await;
        assert_eq!(code, 0);
        assert!(
            out.starts_with("sent ") && out.ends_with(" -> dev-1\n"),
            "unexpected: {out}"
        );
        assert_eq!(err.len(), 0);

        let (code, out, _) = run(&fixture, &["broadcast", "all hands"]).await;
        assert_eq!(code, 0);
        assert!(out.starts_with("broadcast "), "unexpected: {out}");
        assert!(out.ends_with('\n'));

        // The broadcast row is unread for every member, so only the direct one is asserted here.
        let rows = fixture
            .store
            .unread_messages("alpha", "dev-1")
            .await
            .unwrap();
        let direct: Vec<_> = rows.iter().filter(|row| !row.broadcast).collect();
        assert_eq!(direct.len(), 1);
        assert_eq!(direct[0].content, "start now");
        assert_eq!(direct[0].from_member_name, "user");
        assert_eq!(direct[0].to_member_name.as_deref(), Some("dev-1"));
    }

    #[tokio::test]
    async fn send_over_the_limit_is_refused() {
        let fixture = fixture();
        let content = "x".repeat(MAX_MESSAGE_CONTENT_CHARS + 1);
        let (code, out, err) = run(&fixture, &["send", "dev-1", &content]).await;
        assert_eq!(code, 1);
        assert_eq!(out.len(), 0);
        assert!(err.contains("content too long"), "unexpected: {err}");
    }

    #[tokio::test]
    async fn create_task_acknowledges_and_persists() {
        let fixture = fixture();
        let (code, out, err) = run(&fixture, &["create_task", "title", "body"]).await;
        assert_eq!(code, 0);
        assert!(out.starts_with("created "), "unexpected: {out}");
        assert_eq!(err.len(), 0);

        let task_id = out.trim().trim_start_matches("created ").to_string();
        let row = fixture
            .store
            .task("alpha", &task_id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(row.title, "title");
        assert_eq!(row.content, "body");
        assert_eq!(row.status, TaskStatus::Pending);
        assert_eq!(row.created_by.as_deref(), Some("user"));

        let pinned = run(
            &fixture,
            &["create_task", "second", "body", "--task-id", "t-9"],
        )
        .await;
        assert_eq!(pinned, (0, "created t-9\n".to_string(), String::new()));
    }

    #[tokio::test]
    async fn create_task_refusal_is_reported() {
        let fixture = fixture();
        assert_eq!(
            run(&fixture, &["create_task", "a", "b", "--task-id", "t-1"])
                .await
                .0,
            0
        );
        let (code, out, err) = run(&fixture, &["create_task", "c", "d", "--task-id", "t-1"]).await;
        assert_eq!(code, 1);
        assert_eq!(out.len(), 0);
        assert!(err.contains("t-1"), "unexpected: {err}");
    }

    #[tokio::test]
    async fn task_list_renders_rows_and_filters_by_status() {
        let fixture = fixture();
        fixture.seed_task("t-1", "first").await;
        fixture.seed_task("t-2", "second").await;
        fixture.board.claim("t-2", "dev-1").await.unwrap();

        // The board orders by recency of the last transition, so `t-2` comes first.
        let (code, out, err) = run(&fixture, &["task", "list"]).await;
        assert_eq!(code, 0);
        assert_eq!(
            out,
            "[t-2] in_progress second (dev-1)\n[t-1] pending first (-)\n"
        );
        assert_eq!(err.len(), 0);

        let (code, out, _) = run(&fixture, &["task", "list", "--status", "in_progress"]).await;
        assert_eq!(code, 0);
        assert_eq!(out, "[t-2] in_progress second (dev-1)\n");

        let (code, out, _) = run(&fixture, &["task", "list", "--status", "completed"]).await;
        assert_eq!(code, 0);
        assert_eq!(out.len(), 0);
    }

    #[tokio::test]
    async fn task_list_rejects_an_unknown_status() {
        let fixture = fixture();
        let (code, out, err) = run(&fixture, &["task", "list", "--status", "shipped"]).await;
        assert_eq!(code, 1);
        assert_eq!(out.len(), 0);
        assert!(
            err.contains("unknown task status 'shipped'"),
            "unexpected: {err}"
        );
    }

    #[tokio::test]
    async fn task_claimable_lists_open_tasks() {
        let fixture = fixture();
        fixture.seed_task("t-1", "first").await;
        fixture.seed_task("t-2", "second").await;
        fixture.board.claim("t-2", "dev-1").await.unwrap();

        assert_eq!(
            run(&fixture, &["task", "claimable"]).await,
            (0, "[t-1] first\n".to_string(), String::new())
        );
    }

    #[tokio::test]
    async fn task_get_renders_the_seven_line_block() {
        let fixture = fixture();
        fixture.seed_task("t-1", "first").await;
        fixture.seed_task("t-0", "dependency").await;
        fixture
            .board
            .create_task(crate::schema::TaskGraphSpec {
                title: "dependent".to_string(),
                content: "body".to_string(),
                task_id: Some("t-2".to_string()),
                depends_on: vec!["t-0".to_string()],
                ..crate::schema::TaskGraphSpec::default()
            })
            .await
            .unwrap();

        let (code, out, err) = run(&fixture, &["task", "get", "t-2"]).await;
        assert_eq!(code, 0);
        assert!(out.starts_with("id:        t-2\n"), "unexpected: {out}");
        assert!(out.contains("title:     dependent\n"));
        assert!(out.contains("status:    blocked\n"));
        assert!(out.contains("assignee:  -\n"));
        assert!(out.contains("blocked_by: t-0\n"));
        assert!(out.contains("blocks:     -\n"));
        assert!(out.contains("content:\nbody"));
        assert_eq!(err.len(), 0);

        assert_eq!(
            run(&fixture, &["task", "get", "nope"]).await,
            (1, String::new(), "Task 'nope' not found\n".to_string())
        );
    }

    #[tokio::test]
    async fn claim_and_complete_walk_the_board() {
        let fixture = fixture();
        fixture.seed_task("t-1", "first").await;

        assert_eq!(
            run(&fixture, &["claim", "t-1"]).await,
            (0, "claimed t-1\n".to_string(), String::new())
        );
        let claimed = fixture.board.detail("t-1").await.unwrap().unwrap();
        assert_eq!(claimed.assignee.as_deref(), Some("user"));
        assert_eq!(claimed.status, TaskStatus::InProgress);

        assert_eq!(
            run(&fixture, &["complete", "t-1"]).await,
            (0, "completed t-1\n".to_string(), String::new())
        );
        let completed = fixture.board.detail("t-1").await.unwrap().unwrap();
        assert_eq!(completed.status, TaskStatus::Completed);
    }

    #[tokio::test]
    async fn claim_refusal_writes_the_reason_to_the_diagnostic_writer() {
        let fixture = fixture();
        fixture.seed_task("t-1", "first").await;
        fixture.board.claim("t-1", "dev-1").await.unwrap();

        let (code, out, err) = run(&fixture, &["claim", "t-1"]).await;
        assert_eq!(code, 1);
        assert_eq!(out.len(), 0);
        assert!(
            err.contains("already claimed by 'dev-1'"),
            "unexpected: {err}"
        );
    }

    #[tokio::test]
    async fn update_edits_a_pending_task_and_refuses_a_started_one() {
        let fixture = fixture();
        fixture.seed_task("t-1", "first").await;

        let (code, out, err) = run(&fixture, &["update", "t-1", "--title", "better"]).await;
        assert_eq!(
            (code, out, err),
            (0, "updated t-1\n".to_string(), String::new())
        );
        let row = fixture.store.task("alpha", "t-1").await.unwrap().unwrap();
        assert_eq!(row.title, "better");
        assert_eq!(row.content, "body");

        assert_eq!(
            run(&fixture, &["claim", "t-1"]).await,
            (0, "claimed t-1\n".to_string(), String::new())
        );
        let (code, out, err) = run(&fixture, &["update", "t-1", "--content", "later"]).await;
        assert_eq!(code, 1);
        assert_eq!(out.len(), 0);
        assert!(
            err.contains("cannot be edited from status 'in_progress'"),
            "unexpected: {err}"
        );
    }

    #[tokio::test]
    async fn members_render_in_roster_order() {
        let fixture = fixture();
        for (name, role) in [
            ("zeta", TeamRole::Teammate),
            ("alpha", TeamRole::Teammate),
            ("team_leader", TeamRole::Leader),
        ] {
            fixture.seed_member(name, role).await;
        }
        assert_eq!(
            run(&fixture, &["members"]).await,
            (
                0,
                "alpha (teammate) [unstarted]\n\
                 team_leader (leader) [unstarted]\n\
                 zeta (teammate) [unstarted]\n"
                    .to_string(),
                String::new()
            )
        );
    }

    #[tokio::test]
    async fn an_empty_roster_writes_nothing() {
        let fixture = fixture();
        assert_eq!(
            run(&fixture, &["members"]).await,
            (0, String::new(), String::new())
        );
    }

    #[tokio::test]
    async fn a_member_command_is_not_run_under_an_operator_client() {
        let fixture = fixture();
        let mut argv = vec!["team-member".to_string()];
        argv.extend(["inbox"].map(str::to_string));
        let cli = SkillCli::try_parse_from(argv, ClientScope::Member).unwrap();

        let mut out: Vec<u8> = Vec::new();
        let mut err: Vec<u8> = Vec::new();
        let code = cli
            .run_with(fixture.client.clone(), &mut out, &mut err)
            .await;
        assert_eq!(code, 2);
        assert!(String::from_utf8(err)
            .unwrap()
            .contains("operator command tree was not selected"));
    }
}
