//! Process-level suite of the `team` command group.
//!
//! `openjiuwen team` is a client of a team hosted by another process, so every
//! case here drives the real binary against a **stub host**: a Unix listener
//! bound inside the test that speaks the public attach protocol
//! ([`RequestFrame`] / [`ResponseFrame`]) and answers one canned payload per
//! op. Nothing is networked and no team runtime is built, which keeps the suite
//! offline and hermetic.
//!
//! The suite pins four things:
//!
//! 1. **The grammar** — `team --help` lists every operator subcommand.
//! 2. **The wire** — each command reaches the host as exactly one attach request
//!    with the documented arguments (`task list --status in_progress` forwards
//!    the parsed status, `task get` forwards the id, …), and the stub records
//!    every op it received so the assertion is on the request, not on the reply.
//! 3. **The rendering** — one row per member/task, the message and task-record
//!    line shapes, the empty markers, and one pretty JSON document per command
//!    in `--json` mode.
//! 4. **The failures** — a remote `not_found` reaches the process as
//!    `error: not_found:` with exit `1` and empty stdout, an unknown `--status`
//!    is rejected before any request is sent, and a missing socket is a
//!    `validation` error naming both sources.
//!
//! Every child runs through [`support::Sandbox`] (temp cwd, temp
//! `OPENJIUWEN_HOME`, ambient `OPENJIUWEN_TEAM_SOCKET` removed) and is bounded
//! by [`support::CHILD_WAIT`]; the stub's accept loop is aborted when the test
//! ends, so no case waits on a listener.
#![cfg(unix)]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod support;

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use openjiuwen::agent_teams::attach::protocol::{
    read_frame_line, write_all_line, RequestFrame, ResponseFrame,
};
use openjiuwen::agent_teams::{AttachError, AttachResult};
use openjiuwen_common::error::ErrorKind;
use serde_json::{json, Value};
use support::{assert_exit, assert_success, has_row, non_empty_lines, parse_json, stderr, Sandbox};

/// One canned answer: the op name it answers and the outcome sent back.
type Answer = (&'static str, AttachResult<Value>);

/// Distinguishes the sockets of several stubs bound in one sandbox.
///
/// A dropped stub never removes its socket file — the sandbox does that when the
/// test ends — so a second stub must not reuse the first one's path.
static NEXT_SOCKET: AtomicU32 = AtomicU32::new(0);

/// A raw attach-protocol host: one Unix socket and one canned answer per op.
struct Stub {
    /// Socket the child dials through `--socket`.
    socket: PathBuf,
    /// Request frames the host received, in arrival order.
    received: Arc<Mutex<Vec<Value>>>,
    /// The accept loop; aborted when the stub is dropped.
    accept_loop: tokio::task::JoinHandle<()>,
}

impl Stub {
    /// Binds a fresh socket inside `sandbox` and starts answering `answers`.
    fn start(sandbox: &Sandbox, answers: Vec<Answer>) -> Self {
        let index = NEXT_SOCKET.fetch_add(1, Ordering::Relaxed);
        let socket = sandbox.path(format!("team-{index}.sock"));
        let listener = tokio::net::UnixListener::bind(&socket)
            .unwrap_or_else(|err| panic!("binding the stub socket {}: {err}", socket.display()));
        let received = Arc::new(Mutex::new(Vec::new()));
        let accept_loop = tokio::spawn(serve(listener, Arc::new(answers), Arc::clone(&received)));
        Self {
            socket,
            received,
            accept_loop,
        }
    }

    /// The `--socket` value the child must be given.
    fn socket(&self) -> &Path {
        &self.socket
    }

    /// Every request frame the host decoded, as `{"op": …, "args": …}`.
    fn received(&self) -> Vec<Value> {
        self.received
            .lock()
            .expect("the recorded-op lock is never held across a panic")
            .clone()
    }

    /// The arguments of the single request carrying `op`.
    fn args_of(&self, op: &str) -> Value {
        let received = self.received();
        let frame = received
            .iter()
            .find(|frame| frame.get("op") == Some(&Value::String(op.to_owned())))
            .unwrap_or_else(|| panic!("no `{op}` request reached the host: {received:?}"));
        frame.get("args").cloned().unwrap_or(Value::Null)
    }
}

impl Drop for Stub {
    fn drop(&mut self) {
        // Bounded shutdown: aborting the accept loop closes the listener without
        // waiting for another connection, and the sandbox removes the socket file.
        self.accept_loop.abort();
    }
}

/// Accepts connections until aborted, answering every frame of each one.
async fn serve(
    listener: tokio::net::UnixListener,
    answers: Arc<Vec<Answer>>,
    received: Arc<Mutex<Vec<Value>>>,
) {
    loop {
        let Ok((stream, _addr)) = listener.accept().await else {
            return;
        };
        let (read_half, mut write_half) = stream.into_split();
        let mut reader = tokio::io::BufReader::new(read_half);
        loop {
            // EOF (the client is done) or a dead socket: back to accepting.
            let Ok(Some(line)) = read_frame_line(&mut reader).await else {
                break;
            };
            // A frame-level failure closes the connection, as the contract says.
            let Ok(frame) = RequestFrame::decode(&line) else {
                break;
            };
            let op = match frame.op() {
                Ok(op) => op.name(),
                Err(error) => {
                    if reply(&mut write_half, ResponseFrame::err(frame.id, error)).await {
                        continue;
                    }
                    break;
                }
            };
            received
                .lock()
                .expect("the recorded-op lock is never held across a panic")
                .push(json!({"op": op, "args": frame.args}));
            let response = match answers.iter().find(|(name, _)| *name == op) {
                Some((_, Ok(data))) => ResponseFrame::ok(frame.id, data.clone()),
                Some((_, Err(error))) => ResponseFrame::err(frame.id, error.clone()),
                None => ResponseFrame::err(
                    frame.id,
                    AttachError::remote(
                        ErrorKind::NotFound,
                        format!("the stub has no answer for op `{op}`"),
                    ),
                ),
            };
            if !reply(&mut write_half, response).await {
                break;
            }
        }
    }
}

/// Encodes and writes one response; returns whether the connection is still usable.
async fn reply(writer: &mut tokio::net::unix::OwnedWriteHalf, response: ResponseFrame) -> bool {
    match response.encode() {
        Ok(bytes) => write_all_line(writer, &bytes).await.is_ok(),
        Err(_) => false,
    }
}

/// Runs one `team` invocation, optionally pointed at `socket`.
async fn run_team(sandbox: &Sandbox, socket: Option<&Path>, tail: &[&str]) -> std::process::Output {
    let mut args: Vec<String> = vec!["team".to_owned()];
    if let Some(path) = socket {
        args.push("--socket".to_owned());
        args.push(path.display().to_string());
    }
    args.extend(tail.iter().map(|arg| (*arg).to_owned()));
    sandbox.run(&args).await
}

/// One roster row in the schema's wire shape.
fn member(name: &str, display: &str, role: &str, status: &str, execution: &str) -> Value {
    json!({
        "member_name": name,
        "display_name": display,
        "role": role,
        "status": status,
        "execution": execution,
    })
}

/// One task summary in the schema's wire shape.
fn task(id: &str, title: &str, status: &str, assignee: Option<&str>) -> Value {
    json!({
        "task_id": id,
        "title": title,
        "status": status,
        "assignee": assignee,
    })
}

#[tokio::test]
async fn help_lists_every_subcommand() {
    let sandbox = Sandbox::new();
    let output = run_team(&sandbox, None, &["--help"]).await;
    let text = assert_success(&output);
    for row in [
        "members",
        "member",
        "inbox",
        "send",
        "broadcast",
        "create-task",
        "task",
        "claim",
        "complete",
        "update",
        "snapshot",
        "stop",
    ] {
        assert!(has_row(&text, row), "missing `{row}` row in {text:?}");
    }
}

#[tokio::test]
async fn members_renders_one_row_per_member() {
    let sandbox = Sandbox::new();
    let stub = Stub::start(
        &sandbox,
        vec![(
            "list_members",
            Ok(json!([
                member("leader", "Ada", "leader", "ready", "idle"),
                member("dev-1", "", "teammate", "busy", "running"),
            ])),
        )],
    );
    let output = run_team(&sandbox, Some(stub.socket()), &["members"]).await;
    let text = assert_success(&output);
    assert_eq!(
        non_empty_lines(&text),
        vec![
            "leader\tleader\tready\tidle\tAda",
            "dev-1\tteammate\tbusy\trunning"
        ],
        "{text:?}"
    );
    assert!(stub.args_of("list_members").is_object());
}

#[tokio::test]
async fn members_json_decodes_to_the_roster() {
    let sandbox = Sandbox::new();
    let stub = Stub::start(
        &sandbox,
        vec![(
            "list_members",
            Ok(json!([
                member("leader", "Ada", "leader", "ready", "idle"),
                member("dev-1", "", "teammate", "busy", "running"),
            ])),
        )],
    );
    let output = run_team(&sandbox, Some(stub.socket()), &["members", "--json"]).await;
    let value = parse_json(&assert_success(&output));
    let members = value.as_array().expect("a JSON array of members");
    assert_eq!(members.len(), 2);
    assert_eq!(members[0]["member_name"], "leader");
    assert_eq!(members[0]["display_name"], "Ada");
    assert_eq!(members[1]["role"], "teammate");
    assert_eq!(members[1]["execution"], "running");
}

#[tokio::test]
async fn an_empty_roster_says_so() {
    let sandbox = Sandbox::new();
    let stub = Stub::start(&sandbox, vec![("list_members", Ok(json!([])))]);
    let output = run_team(&sandbox, Some(stub.socket()), &["members"]).await;
    assert_eq!(assert_success(&output).trim(), "no team members");
}

#[tokio::test]
async fn task_list_forwards_the_status_filter() {
    let sandbox = Sandbox::new();
    let stub = Stub::start(
        &sandbox,
        vec![(
            "list_tasks",
            Ok(json!([
                task("t1", "write docs", "in_progress", Some("dev-1")),
                task("t2", "review docs", "in_progress", None),
            ])),
        )],
    );
    let output = run_team(
        &sandbox,
        Some(stub.socket()),
        &["task", "list", "--status", "in_progress"],
    )
    .await;
    let text = assert_success(&output);
    assert_eq!(
        non_empty_lines(&text),
        vec![
            "t1\tin_progress\twrite docs\tdev-1",
            "t2\tin_progress\treview docs"
        ],
        "{text:?}"
    );
    assert_eq!(stub.args_of("list_tasks"), json!({"status": "in_progress"}));
}

#[tokio::test]
async fn an_empty_task_list_says_so() {
    let sandbox = Sandbox::new();
    let stub = Stub::start(
        &sandbox,
        vec![
            ("list_tasks", Ok(json!([]))),
            ("claimable_tasks", Ok(json!([]))),
        ],
    );
    let output = run_team(&sandbox, Some(stub.socket()), &["task", "list"]).await;
    assert_eq!(assert_success(&output).trim(), "no tasks");
    let output = run_team(&sandbox, Some(stub.socket()), &["task", "claimable"]).await;
    assert_eq!(assert_success(&output).trim(), "no claimable tasks");
    assert!(stub.args_of("list_tasks").get("status").is_some());
}

#[tokio::test]
async fn task_list_rejects_an_unknown_status_without_asking_the_host() {
    let sandbox = Sandbox::new();
    let stub = Stub::start(&sandbox, Vec::new());
    let output = run_team(
        &sandbox,
        Some(stub.socket()),
        &["task", "list", "--status", "done"],
    )
    .await;
    assert_exit(&output, 1);
    assert!(output.stdout.is_empty(), "{}", support::describe(&output));
    let text = stderr(&output);
    assert!(text.contains("error: validation:"), "{text:?}");
    assert!(text.contains("unknown task status 'done'"), "{text:?}");
    assert!(text.contains("in_progress"), "{text:?}");
    assert!(stub.received().is_empty(), "{:?}", stub.received());
}

#[tokio::test]
async fn task_get_reports_a_missing_task() {
    let sandbox = Sandbox::new();
    let stub = Stub::start(
        &sandbox,
        vec![(
            "task_detail",
            Err(AttachError::remote(
                ErrorKind::NotFound,
                "task not found: task-404",
            )),
        )],
    );
    let output = run_team(&sandbox, Some(stub.socket()), &["task", "get", "task-404"]).await;
    assert_exit(&output, 1);
    assert!(output.stdout.is_empty(), "{}", support::describe(&output));
    let text = stderr(&output);
    assert!(text.contains("error: not_found:"), "{text:?}");
    assert!(text.contains("task not found: task-404"), "{text:?}");
    assert_eq!(stub.args_of("task_detail"), json!({"task_id": "task-404"}));
}

#[tokio::test]
async fn task_get_renders_the_detail() {
    let sandbox = Sandbox::new();
    let mut detail = task("t1", "write docs", "in_review", Some("dev-1"));
    detail["content"] = json!("cover every subcommand");
    let stub = Stub::start(&sandbox, vec![("task_detail", Ok(detail))]);
    let output = run_team(&sandbox, Some(stub.socket()), &["task", "get", "t1"]).await;
    let text = assert_success(&output);
    let rows = non_empty_lines(&text);
    assert!(rows.contains(&"task_id\tt1"), "{rows:?}");
    assert!(rows.contains(&"title\twrite docs"), "{rows:?}");
    assert!(rows.contains(&"status\tin_review"), "{rows:?}");
    assert!(rows.contains(&"assignee\tdev-1"), "{rows:?}");
    assert!(rows.contains(&"cover every subcommand"), "{rows:?}");
}

#[tokio::test]
async fn member_prints_one_row() {
    let sandbox = Sandbox::new();
    let stub = Stub::start(
        &sandbox,
        vec![(
            "member_status",
            Ok(member("dev-1", "", "teammate", "busy", "running")),
        )],
    );
    let output = run_team(&sandbox, Some(stub.socket()), &["member", "dev-1"]).await;
    assert_eq!(
        assert_success(&output).trim(),
        "dev-1\tteammate\tbusy\trunning"
    );
    assert_eq!(
        stub.args_of("member_status"),
        json!({"member_name": "dev-1"})
    );
}

#[tokio::test]
async fn send_and_broadcast_report_the_message() {
    let sandbox = Sandbox::new();
    let stub = Stub::start(
        &sandbox,
        vec![
            (
                "send_member",
                Ok(json!({
                    "team_name": "core",
                    "message_id": "msg_1",
                    "from_member_name": "leader",
                    "to_member_name": "dev-1",
                    "content": "ping",
                })),
            ),
            (
                "broadcast",
                Ok(json!({
                    "team_name": "core",
                    "message_id": "msg_2",
                    "from_member_name": "leader",
                    "to_member_name": null,
                    "content": "standup",
                })),
            ),
        ],
    );
    let output = run_team(&sandbox, Some(stub.socket()), &["send", "dev-1", "ping"]).await;
    assert_eq!(assert_success(&output).trim(), "msg_1 from leader to dev-1");
    assert_eq!(
        stub.args_of("send_member"),
        json!({"to": "dev-1", "content": "ping"})
    );
    let output = run_team(&sandbox, Some(stub.socket()), &["broadcast", "standup"]).await;
    assert_eq!(assert_success(&output).trim(), "msg_2 from leader to *");
    assert_eq!(stub.args_of("broadcast"), json!({"content": "standup"}));
}

#[tokio::test]
async fn create_task_renders_a_task_record() {
    let sandbox = Sandbox::new();
    let stub = Stub::start(
        &sandbox,
        vec![(
            "create_task",
            Ok(json!({"task_id": "t9", "title": "write docs", "status": "pending"})),
        )],
    );
    let output = run_team(
        &sandbox,
        Some(stub.socket()),
        &[
            "create-task",
            "write docs",
            "cover the CLI",
            "--task-id",
            "t9",
        ],
    )
    .await;
    assert_eq!(assert_success(&output).trim(), "t9\tpending\twrite docs");
    assert_eq!(
        stub.args_of("create_task"),
        json!({
            "title": "write docs",
            "content": "cover the CLI",
            "task_id": "t9",
            "dependencies": [],
        })
    );
}

#[tokio::test]
async fn claim_and_complete_render_outcomes() {
    let sandbox = Sandbox::new();
    let stub = Stub::start(
        &sandbox,
        vec![
            (
                "claim_task",
                Ok(json!({"ok": false, "reason": "task t1 is not pending"})),
            ),
            ("complete_task", Ok(json!({"ok": true, "reason": null}))),
        ],
    );
    let output = run_team(&sandbox, Some(stub.socket()), &["claim", "t1"]).await;
    assert_exit(&output, 1);
    assert!(output.stdout.is_empty(), "{}", support::describe(&output));
    let text = stderr(&output);
    assert!(text.contains("error: tool:"), "{text:?}");
    assert!(text.contains("task t1 is not pending"), "{text:?}");
    let output = run_team(&sandbox, Some(stub.socket()), &["complete", "t2"]).await;
    assert_eq!(assert_success(&output).trim(), "completed t2");
    assert_eq!(stub.args_of("complete_task"), json!({"task_id": "t2"}));
}

#[tokio::test]
async fn snapshot_json_decodes_to_the_team_snapshot() {
    let sandbox = Sandbox::new();
    let stub = Stub::start(
        &sandbox,
        vec![(
            "snapshot",
            Ok(json!({
                "team_name": "core",
                "state": "running",
                "members": [
                    member("leader", "Ada", "leader", "ready", "idle"),
                    member("dev-1", "", "teammate", "busy", "running"),
                ],
                "tasks": [task("t1", "write docs", "in_progress", Some("dev-1"))],
                "updated_at": 1_700_000_000_000u64,
            })),
        )],
    );
    let output = run_team(&sandbox, Some(stub.socket()), &["snapshot", "--json"]).await;
    let value = parse_json(&assert_success(&output));
    assert_eq!(value["team_name"], "core");
    assert_eq!(value["state"], "running");
    assert_eq!(
        value["members"].as_array().map(Vec::len),
        Some(2),
        "{value}"
    );
    assert_eq!(value["tasks"][0]["task_id"], "t1");
    assert!(value["updated_at"].is_u64());
}

#[tokio::test]
async fn snapshot_text_summarises_the_team() {
    let sandbox = Sandbox::new();
    let stub = Stub::start(
        &sandbox,
        vec![(
            "snapshot",
            Ok(json!({
                "team_name": "core",
                "state": "running",
                "members": [
                    member("leader", "Ada", "leader", "ready", "idle"),
                    member("dev-1", "", "teammate", "busy", "running"),
                ],
                "tasks": [task("t1", "write docs", "in_progress", Some("dev-1"))],
                "updated_at": 0,
            })),
        )],
    );
    let output = run_team(&sandbox, Some(stub.socket()), &["snapshot"]).await;
    let text = assert_success(&output);
    let rows = non_empty_lines(&text);
    assert_eq!(rows[0], "team core (running): 2 member(s), 1 task(s)");
    assert_eq!(rows[1], "leader\tleader\tready\tidle\tAda");
    assert_eq!(rows[2], "dev-1\tteammate\tbusy\trunning");
    assert_eq!(rows[3], "t1\tin_progress\twrite docs\tdev-1");
}

#[tokio::test]
async fn inbox_prints_the_text_and_the_empty_marker() {
    let sandbox = Sandbox::new();
    let stub = Stub::start(
        &sandbox,
        vec![("read_inbox", Ok(json!("hello from ops\n")))],
    );
    let output = run_team(&sandbox, Some(stub.socket()), &["inbox"]).await;
    assert_eq!(assert_success(&output), "hello from ops\n");
    let empty = Stub::start(&sandbox, vec![("read_inbox", Ok(json!("  \n")))]);
    let output = run_team(&sandbox, Some(empty.socket()), &["inbox"]).await;
    assert_eq!(assert_success(&output).trim(), "inbox is empty");
}

#[tokio::test]
async fn stop_reports_the_request() {
    let sandbox = Sandbox::new();
    let stub = Stub::start(&sandbox, vec![("stop_team", Ok(json!({"stopping": true})))]);
    let output = run_team(&sandbox, Some(stub.socket()), &["stop"]).await;
    assert_eq!(assert_success(&output).trim(), "stop requested");
    let output = run_team(&sandbox, Some(stub.socket()), &["stop", "--json"]).await;
    assert_eq!(
        parse_json(&assert_success(&output)),
        json!({"stopping": true})
    );
}

#[tokio::test]
async fn a_missing_socket_is_a_validation_error() {
    let sandbox = Sandbox::new();
    // `Sandbox` removes every ambient `OPENJIUWEN_*` variable, so neither the
    // flag nor the environment variable supplies a socket here.
    let output = run_team(&sandbox, None, &["members"]).await;
    assert_exit(&output, 1);
    assert!(output.stdout.is_empty(), "{}", support::describe(&output));
    let text = stderr(&output);
    assert!(text.contains("error: validation:"), "{text:?}");
    assert!(text.contains("team socket is required"), "{text:?}");
    assert!(text.contains("OPENJIUWEN_TEAM_SOCKET"), "{text:?}");
}
