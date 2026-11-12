//! Member-scope dispatch: a spawned CLI driving the real teammate tool set.
//!
//! Python parity source: `_dispatch_member` in `openjiuwen/agent_teams/skill/cli.py`. A member
//! process is a first-class team member, so its commands are the tools an in-process teammate
//! calls (`openjiuwen_agent_teams::skill::member_tools_for`): `view_task`, `claim_task` and
//! `send_message` are dispatched through [`TeamClient::invoke_tool`] and their answers are rendered
//! here, plus `inbox`, which is the external-only pull operation backed by the mailbox directly.
//!
//! `--watch` is accepted for parity but performs a single read: the event-driven re-render loop
//! needs the runtime's watch hook, which is not wired yet (see the crate's deferred list).

use std::io::Write;
use std::sync::Arc;

use serde_json::{json, Map, Value};

use crate::constants::BROADCAST_TARGET;

use super::super::client::TeamClient;
use super::output::{broadcast_line, claimed_line, completed_line, render_view_task, sent_line};
use super::{
    emit_line, fail, refuse, ClaimStatus, CliCommand, MemberCommand, SkillCli, ViewAction,
};

/// Dispatches one member-scope command, returning the process exit code.
pub(crate) async fn dispatch(
    cli: &SkillCli,
    client: Arc<dyn TeamClient>,
    out: &mut dyn Write,
    err: &mut dyn Write,
) -> i32 {
    let CliCommand::Member(command) = cli.command() else {
        let _ = writeln!(err, "internal: the member command tree was not selected");
        return 2;
    };
    match command {
        MemberCommand::Inbox { watch } => inbox(client.as_ref(), out, err, *watch).await,
        MemberCommand::ViewTask {
            action,
            task_id,
            status,
        } => {
            view_task(
                client.as_ref(),
                out,
                err,
                *action,
                task_id.as_deref(),
                status.as_deref(),
            )
            .await
        }
        MemberCommand::ClaimTask { task_id, status } => {
            claim_task(client.as_ref(), out, err, task_id, *status).await
        }
        MemberCommand::SendMessage { to, content } => {
            send_message(client.as_ref(), out, err, to, content).await
        }
    }
}

/// Renders the member's unread mailbox (`--watch` performs the same single read).
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

/// Invokes the `view_task` tool with the supplied action and filters.
///
/// `task_id` and `status` are only sent when the caller supplied them, exactly like Python builds
/// the tool input dictionary.
async fn view_task(
    client: &dyn TeamClient,
    out: &mut dyn Write,
    err: &mut dyn Write,
    action: ViewAction,
    task_id: Option<&str>,
    status: Option<&str>,
) -> i32 {
    let mut arguments = Map::new();
    arguments.insert(
        "action".to_string(),
        Value::String(action.as_str().to_string()),
    );
    if let Some(task_id) = task_id {
        arguments.insert("task_id".to_string(), Value::String(task_id.to_string()));
    }
    if let Some(status) = status {
        arguments.insert("status".to_string(), Value::String(status.to_string()));
    }
    match client
        .invoke_tool("view_task", Value::Object(arguments))
        .await
    {
        Ok(value) => match render_view_task(action.as_str(), &value) {
            Ok(rendered) if rendered.is_empty() => 0,
            Ok(rendered) => emit_line(out, err, &rendered),
            Err(error) => fail(err, &error),
        },
        Err(error) => fail(err, &error),
    }
}

/// Invokes the teammate `claim_task` tool and reports its outcome.
///
/// The tool answers `{ok, reason}`. An answer without `ok` is not a shape this CLI knows, so it is
/// written verbatim rather than guessed at.
async fn claim_task(
    client: &dyn TeamClient,
    out: &mut dyn Write,
    err: &mut dyn Write,
    task_id: &str,
    status: ClaimStatus,
) -> i32 {
    let arguments = json!({ "task_id": task_id, "status": status.as_str() });
    match client.invoke_tool("claim_task", arguments).await {
        Ok(value) => match value.get("ok").and_then(Value::as_bool) {
            Some(true) => {
                let line = match status {
                    ClaimStatus::Claimed => claimed_line(task_id),
                    ClaimStatus::Completed => completed_line(task_id),
                };
                emit_line(out, err, &line)
            }
            Some(false) => refuse(err, value.get("reason").and_then(Value::as_str)),
            None => emit_line(out, err, &value.to_string()),
        },
        Err(error) => fail(err, &error),
    }
}

/// Invokes the `send_message` tool and renders the acknowledgement.
async fn send_message(
    client: &dyn TeamClient,
    out: &mut dyn Write,
    err: &mut dyn Write,
    to: &str,
    content: &str,
) -> i32 {
    let arguments = json!({ "to": to, "content": content });
    match client.invoke_tool("send_message", arguments).await {
        Ok(value) => match value.get("ok").and_then(Value::as_bool) {
            Some(true) => {
                let message_id = value
                    .get("message_id")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let line = if to == BROADCAST_TARGET {
                    broadcast_line(message_id)
                } else {
                    sent_line(message_id, to)
                };
                emit_line(out, err, &line)
            }
            _ => refuse(err, value.get("reason").and_then(Value::as_str)),
        },
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

    /// Builds a fixture joined by `member_name`.
    fn fixture(member_name: &str) -> CliFixture {
        CliFixture::new(ClientScope::Member, member_name)
    }

    /// Runs one member-scope invocation and returns `(exit code, stdout, stderr)`.
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

    /// Parses `words` in the member tree.
    fn parse(words: &[&str]) -> SkillCli {
        let mut argv = vec!["team-member".to_string()];
        argv.extend(words.iter().map(|word| (*word).to_string()));
        SkillCli::try_parse_from(argv, ClientScope::Member).unwrap()
    }

    #[tokio::test]
    async fn inbox_renders_the_mailbox_once() {
        let fixture = fixture("dev-1");
        assert_eq!(
            run(&fixture, &["inbox"]).await,
            (0, "(inbox empty)\n".to_string(), String::new())
        );

        fixture.client.send_message("dev-1", "hello").await.unwrap();
        let (code, out, err) = run(&fixture, &["inbox"]).await;
        assert_eq!(code, 0);
        assert!(out.contains("-> dev-1 : hello"), "unexpected inbox: {out}");
        assert_eq!(err.len(), 0);

        // The read consumed the row: `--watch` is one-shot, not a re-render loop.
        assert_eq!(
            run(&fixture, &["inbox", "--watch"]).await,
            (0, "(inbox empty)\n".to_string(), String::new())
        );
    }

    #[tokio::test]
    async fn view_task_renders_list_get_and_claimable() {
        let fixture = fixture("dev-1");
        fixture.seed_task("t-1", "first").await;

        assert_eq!(
            run(&fixture, &["view_task"]).await,
            (0, "[t-1] pending first (-)\n".to_string(), String::new())
        );
        assert_eq!(
            run(
                &fixture,
                &["view_task", "--action", "list", "--status", "pending"]
            )
            .await,
            (0, "[t-1] pending first (-)\n".to_string(), String::new())
        );
        assert_eq!(
            run(&fixture, &["view_task", "--action", "claimable"]).await,
            (0, "[t-1] first\n".to_string(), String::new())
        );

        let (code, out, err) = run(
            &fixture,
            &["view_task", "--action", "get", "--task-id", "t-1"],
        )
        .await;
        assert_eq!(code, 0);
        assert!(
            out.starts_with("id:        t-1\ntitle:     first\n"),
            "unexpected block: {out}"
        );
        assert!(out.contains("content:\nbody"));
        assert_eq!(err.len(), 0);

        assert_eq!(
            run(
                &fixture,
                &["view_task", "--action", "get", "--task-id", "nope"]
            )
            .await,
            (0, "null\n".to_string(), String::new())
        );
    }

    #[tokio::test]
    async fn view_task_prints_nothing_for_an_empty_board() {
        let fixture = fixture("dev-1");
        assert_eq!(
            run(&fixture, &["view_task"]).await,
            (0, String::new(), String::new())
        );
    }

    #[tokio::test]
    async fn view_task_reports_an_unknown_status_filter() {
        let fixture = fixture("dev-1");
        fixture.seed_task("t-1", "first").await;
        let (code, out, err) = run(&fixture, &["view_task", "--status", "shipped"]).await;
        assert_eq!(code, 1);
        assert_eq!(out.len(), 0);
        assert!(
            err.contains("unknown task status 'shipped'"),
            "unexpected: {err}"
        );
    }

    #[tokio::test]
    async fn claim_task_claims_then_completes() {
        let fixture = fixture("dev-1");
        fixture.seed_task("t-1", "first").await;

        assert_eq!(
            run(&fixture, &["claim_task", "t-1", "claimed"]).await,
            (0, "claimed t-1\n".to_string(), String::new())
        );
        let claimed = fixture.board.detail("t-1").await.unwrap().unwrap();
        assert_eq!(claimed.status, TaskStatus::InProgress);
        assert_eq!(claimed.assignee.as_deref(), Some("dev-1"));

        assert_eq!(
            run(&fixture, &["claim_task", "t-1", "completed"]).await,
            (0, "completed t-1\n".to_string(), String::new())
        );
        let completed = fixture.board.detail("t-1").await.unwrap().unwrap();
        assert_eq!(completed.status, TaskStatus::Completed);
    }

    #[tokio::test]
    async fn claim_task_refusal_goes_to_the_diagnostic_writer() {
        let fixture = fixture("dev-1");
        fixture.seed_task("t-1", "first").await;
        // Someone else already owns the task, so the board refuses the claim.
        fixture.board.claim("t-1", "dev-2").await.unwrap();

        let (code, out, err) = run(&fixture, &["claim_task", "t-1", "claimed"]).await;
        assert_eq!(code, 1);
        assert_eq!(out.len(), 0);
        assert!(!err.is_empty(), "expected a refusal reason");
        assert!(
            err.contains("t-1"),
            "the reason should name the task: {err}"
        );
    }

    #[tokio::test]
    async fn send_message_direct_and_broadcast() {
        let fixture = fixture("dev-1");
        let (code, out, err) = run(&fixture, &["send_message", "team_leader", "starting"]).await;
        assert_eq!(code, 0);
        assert!(out.starts_with("sent "), "unexpected: {out}");
        assert!(out.ends_with(" -> team_leader\n"), "unexpected: {out}");
        assert_eq!(err.len(), 0);

        let (code, out, err) = run(&fixture, &["send_message", "*", "all hands"]).await;
        assert_eq!(code, 0);
        assert!(out.starts_with("broadcast "), "unexpected: {out}");
        assert_eq!(err.len(), 0);

        // The broadcast row is unread for every member, so only the direct one is asserted here.
        let rows = fixture
            .store
            .unread_messages("alpha", "team_leader")
            .await
            .unwrap();
        let direct: Vec<_> = rows.iter().filter(|row| !row.broadcast).collect();
        assert_eq!(direct.len(), 1);
        assert_eq!(direct[0].content, "starting");
        assert_eq!(direct[0].from_member_name, "dev-1");
        assert!(
            rows.iter().any(|row| row.broadcast),
            "the broadcast row is persisted"
        );
    }

    #[tokio::test]
    async fn over_limit_send_message_is_refused() {
        let fixture = fixture("dev-1");
        let content = "x".repeat(MAX_MESSAGE_CONTENT_CHARS + 1);
        let (code, out, err) = run(&fixture, &["send_message", "team_leader", &content]).await;
        assert_eq!(code, 1);
        assert_eq!(out.len(), 0);
        assert!(err.contains("content too long"), "unexpected: {err}");
        let rows = fixture
            .store
            .unread_messages("alpha", "team_leader")
            .await
            .unwrap();
        assert!(rows.is_empty(), "nothing may be persisted");
    }

    #[tokio::test]
    async fn an_operator_command_is_not_run_under_a_member_client() {
        let fixture = fixture("dev-1");
        let mut argv = vec!["team-member".to_string()];
        argv.extend(["members"].map(str::to_string));
        let cli = SkillCli::try_parse_from(argv, ClientScope::Operator).unwrap();

        let mut out: Vec<u8> = Vec::new();
        let mut err: Vec<u8> = Vec::new();
        let code = cli
            .run_with(fixture.client.clone(), &mut out, &mut err)
            .await;
        assert_eq!(code, 2);
        assert!(String::from_utf8(err)
            .unwrap()
            .contains("member command tree was not selected"));
    }

    #[tokio::test]
    async fn the_roster_is_readable_through_the_member_client() {
        let fixture = fixture("dev-1");
        fixture.seed_member("dev-1", TeamRole::Teammate).await;
        let members = fixture.client.list_members().await.unwrap();
        assert_eq!(members.len(), 1);
        assert_eq!(members[0].member_name, "dev-1");
        assert_eq!(members[0].role, TeamRole::Teammate);
    }
}
