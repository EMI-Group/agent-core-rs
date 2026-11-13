#![cfg(unix)]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::pedantic)]
//! Tests for the attach server over real Unix domain sockets.
//!
//! Every test binds a real socket in a temporary directory (`<tmp>/s`, so the path stays short) and
//! drives it either through [`AttachClient`] or through a hand-written request line on a raw
//! `UnixStream` for the malformed cases.
//! The non-Unix variant of `AttachServer` is untested by construction: this whole file is
//! `#![cfg(unix)]`, because every test needs a socket to bind and a socket to connect to.
//! Nothing touches the network beyond the local socket and nothing sleeps unboundedly: the only
//! waits are `tokio::time::timeout(10s, ..)` guards, so a regression fails instead of hanging.

use std::os::unix::fs::{FileTypeExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use openjiuwen_common::error::ErrorKind;
use serde_json::{json, Value};
use tokio::io::{AsyncBufRead, AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};

use super::client::AttachClient;
use super::protocol::{
    AttachError, AttachHandler, AttachOp, AttachResult, ResponseFrame, MAX_FRAME_BYTES,
};
use super::server::AttachServer;
use crate::schema::{
    ExecutionStatus, MemberSnapshot, MemberStatus, TaskDetail, TaskStatus, TaskSummary, TeamRole,
    TeamSnapshot,
};

/// Upper bound for every wait in this file.
const BOUND: Duration = Duration::from_secs(10);

// --- fixtures --------------------------------------------------------------

/// One roster row with every field set.
fn member_row(name: &str) -> MemberSnapshot {
    MemberSnapshot {
        member_name: name.to_owned(),
        display_name: format!("{name} display"),
        role: TeamRole::Teammate,
        status: MemberStatus::Busy,
        execution: ExecutionStatus::Running,
    }
}

/// One task list row with every field set.
fn task_row(id: &str, status: TaskStatus) -> TaskSummary {
    TaskSummary {
        task_id: id.to_owned(),
        title: format!("Task {id}"),
        status,
        updated_at: 1_700,
        ..TaskSummary::default()
    }
}

/// One task detail with every field set.
fn task_detail(id: &str) -> TaskDetail {
    TaskDetail {
        task_id: id.to_owned(),
        title: "Write tests".to_owned(),
        content: "Cover the server".to_owned(),
        status: TaskStatus::InProgress,
        ..TaskDetail::default()
    }
}

/// A full team snapshot fixture.
fn snapshot_fixture() -> TeamSnapshot {
    TeamSnapshot {
        team_name: "agent_team".to_owned(),
        members: vec![member_row("team_leader"), member_row("dev-1")],
        tasks: vec![task_row("task-1", TaskStatus::Pending)],
        updated_at: 1_700,
        ..TeamSnapshot::default()
    }
}

/// One sample of every op variant, in the order the round-trip test sends them.
fn every_op() -> Vec<AttachOp> {
    vec![
        AttachOp::ListMembers,
        AttachOp::MemberStatus {
            member_name: "dev-1".to_owned(),
        },
        AttachOp::ReadInbox,
        AttachOp::SendMember {
            to: "dev-1".to_owned(),
            content: "hello".to_owned(),
        },
        AttachOp::Broadcast {
            content: "hello all".to_owned(),
        },
        AttachOp::ListTasks {
            status: Some(TaskStatus::Pending),
        },
        AttachOp::ClaimableTasks,
        AttachOp::TaskDetail {
            task_id: "task-1".to_owned(),
        },
        AttachOp::CreateTask {
            title: "Next".to_owned(),
            content: "Do the next thing".to_owned(),
            task_id: Some("task-2".to_owned()),
            dependencies: vec!["task-1".to_owned()],
        },
        AttachOp::UpdateTask {
            task_id: "task-1".to_owned(),
            title: Some("Renamed".to_owned()),
            content: None,
        },
        AttachOp::ClaimTask {
            task_id: "task-1".to_owned(),
        },
        AttachOp::CompleteTask {
            task_id: "task-1".to_owned(),
        },
        AttachOp::InvokeTool {
            name: "list_tasks".to_owned(),
            arguments: json!({"status": "pending"}),
        },
        AttachOp::Snapshot,
        AttachOp::StopTeam,
    ]
}

/// The `data` payload the fake handlers answer each op with.
fn payload(op: &AttachOp) -> Value {
    match op {
        AttachOp::ListMembers => {
            serde_json::to_value(vec![member_row("team_leader"), member_row("dev-1")]).unwrap()
        }
        AttachOp::MemberStatus { member_name } => {
            serde_json::to_value(member_row(member_name)).unwrap()
        }
        AttachOp::ReadInbox => json!("2 unread messages"),
        AttachOp::SendMember { .. } | AttachOp::Broadcast { .. } => {
            json!({"ok": true, "message_id": "m-1", "reason": null})
        }
        AttachOp::ListTasks { .. } => {
            serde_json::to_value(vec![task_row("task-1", TaskStatus::Pending)]).unwrap()
        }
        AttachOp::ClaimableTasks => serde_json::to_value(Vec::<TaskSummary>::new()).unwrap(),
        AttachOp::TaskDetail { task_id } if task_id == "missing" => Value::Null,
        AttachOp::TaskDetail { task_id } => serde_json::to_value(task_detail(task_id)).unwrap(),
        AttachOp::CreateTask { .. } => {
            json!({"ok": true, "task_id": "task-2", "reason": null})
        }
        AttachOp::UpdateTask { .. }
        | AttachOp::ClaimTask { .. }
        | AttachOp::CompleteTask { .. } => {
            json!({"ok": true, "reason": null})
        }
        AttachOp::InvokeTool { arguments, .. } => arguments.clone(),
        AttachOp::Snapshot => serde_json::to_value(snapshot_fixture()).unwrap(),
        AttachOp::StopTeam => json!({"stopped": true}),
    }
}

// --- fake handlers ---------------------------------------------------------

/// Handler that records every op it sees and answers with [`payload`].
#[derive(Default)]
struct EchoHandler {
    /// Ops in arrival order.
    seen: Mutex<Vec<AttachOp>>,
}

impl EchoHandler {
    /// Ops recorded so far, in arrival order.
    fn ops(&self) -> Vec<AttachOp> {
        self.seen.lock().unwrap().clone()
    }
}

#[async_trait]
impl AttachHandler for EchoHandler {
    async fn handle(&self, op: AttachOp) -> AttachResult<Value> {
        let data = payload(&op);
        self.seen.lock().unwrap().push(op);
        Ok(data)
    }
}

/// Handler that reports a `NotFound` error for the member named `ghost` and `null` for other ops.
struct GhostHandler;

#[async_trait]
impl AttachHandler for GhostHandler {
    async fn handle(&self, op: AttachOp) -> AttachResult<Value> {
        match op {
            AttachOp::MemberStatus { member_name } if member_name == "ghost" => {
                Err(AttachError::remote(
                    ErrorKind::NotFound,
                    format!("unknown member `{member_name}`"),
                ))
            }
            AttachOp::MemberStatus { member_name } => {
                Ok(serde_json::to_value(member_row(&member_name)).unwrap())
            }
            _ => Ok(Value::Null),
        }
    }
}

// --- harness ---------------------------------------------------------------

/// A bound server plus its serving task and temporary directory.
struct Harness {
    /// Keeps the socket directory alive for the lifetime of the server.
    _dir: tempfile::TempDir,
    /// The bound server, shut down on drop.
    server: Arc<AttachServer>,
    /// The serving task, aborted on drop.
    handle: tokio::task::JoinHandle<openjiuwen_common::error::Result<()>>,
}

impl Harness {
    /// Binds `<tmp>/s` and starts serving it with `handler`.
    async fn start(handler: Arc<dyn AttachHandler>) -> Self {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s");
        let server = Arc::new(AttachServer::bind(&path).await.unwrap());
        let serving = Arc::clone(&server);
        let handle = tokio::spawn(async move { serving.serve(handler).await });
        Self {
            _dir: dir,
            server,
            handle,
        }
    }

    /// The socket path this harness serves.
    fn path(&self) -> &Path {
        self.server.path()
    }

    /// A client connected to this harness.
    async fn client(&self) -> AttachClient {
        AttachClient::connect(self.path()).await.unwrap()
    }
}

impl Drop for Harness {
    fn drop(&mut self) {
        self.server.shutdown();
        self.handle.abort();
    }
}

/// A harness whose handler is the recording [`EchoHandler`].
async fn echo_harness() -> (Harness, Arc<EchoHandler>) {
    let handler = Arc::new(EchoHandler::default());
    let shared: Arc<dyn AttachHandler> = handler.clone();
    (Harness::start(shared).await, handler)
}

// --- raw socket helpers ----------------------------------------------------

/// Reads response lines until EOF (or the test bound expires).
async fn read_until_eof<R: AsyncBufRead + Unpin>(reader: &mut R) -> Vec<Vec<u8>> {
    let mut lines = Vec::new();
    loop {
        let mut line = Vec::new();
        let read = tokio::time::timeout(BOUND, reader.read_until(b'\n', &mut line))
            .await
            .unwrap()
            .unwrap();
        if read == 0 {
            return lines;
        }
        lines.push(line);
    }
}

/// Sends one raw request line on a fresh connection and returns every response line until EOF.
///
/// The write half is shut down after the request, so a request without its trailing newline still
/// reaches the server (which reads the last frame at EOF) and every response line is followed by
/// the server closing the connection.
async fn raw_exchange(path: &Path, request: &[u8]) -> Vec<Vec<u8>> {
    let stream = UnixStream::connect(path).await.unwrap();
    let (read_half, mut write_half) = stream.into_split();
    write_half.write_all(request).await.unwrap();
    write_half.flush().await.unwrap();
    write_half.shutdown().await.unwrap();
    let mut reader = BufReader::new(read_half);
    read_until_eof(&mut reader).await
}

/// Sends one raw request line and decodes the single error response that must answer it.
async fn raw_error(path: &Path, request: &[u8]) -> ResponseFrame {
    let lines = raw_exchange(path, request).await;
    assert_eq!(lines.len(), 1, "expected exactly one response line");
    let response = ResponseFrame::decode(&lines[0]).unwrap();
    assert!(!response.ok, "expected an error response");
    response
}

// --- binding ---------------------------------------------------------------

#[tokio::test]
async fn bind_creates_parent_directories_and_sets_mode() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("nested").join("dir").join("s");
    let server = AttachServer::bind(&path).await.unwrap();

    assert_eq!(server.path(), path);
    let metadata = std::fs::symlink_metadata(&path).unwrap();
    assert!(metadata.file_type().is_socket(), "path must be a socket");
    assert_eq!(
        metadata.permissions().mode() & 0o777,
        0o600,
        "socket must only be reachable by its owner"
    );
}

#[tokio::test]
async fn bind_refuses_a_live_socket() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("s");
    let _live = UnixListener::bind(&path).unwrap();

    let error = AttachServer::bind(&path).await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Config);
    assert!(
        error.to_string().contains("in use"),
        "message should say the socket is in use: {error}"
    );
}

#[tokio::test]
async fn bind_reclaims_a_stale_socket() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("s");
    let listener = UnixListener::bind(&path).unwrap();
    drop(listener);
    assert!(
        std::fs::symlink_metadata(&path)
            .unwrap()
            .file_type()
            .is_socket(),
        "the dead listener leaves its socket file behind"
    );

    let server = AttachServer::bind(&path).await.unwrap();
    assert_eq!(server.path(), path);
}

#[tokio::test]
async fn bind_rejects_an_empty_path() {
    let error = AttachServer::bind("").await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
}

#[tokio::test]
async fn bind_rejects_a_directory_path() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("a-directory");
    std::fs::create_dir(&path).unwrap();

    let error = AttachServer::bind(&path).await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("directory"), "{error}");
}

#[tokio::test]
async fn bind_refuses_a_regular_file_and_leaves_it_alone() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("s");
    std::fs::write(&path, b"precious").unwrap();

    let error = AttachServer::bind(&path).await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Config);
    assert_eq!(std::fs::read(&path).unwrap(), b"precious");
}

// --- request dispatch ------------------------------------------------------

#[tokio::test]
async fn round_trips_every_op() {
    let (harness, echo) = echo_harness().await;
    let mut client = harness.client().await;

    for op in every_op() {
        let data = client.call(op.clone()).await.unwrap();
        assert_eq!(data, payload(&op), "payload mismatch for `{}`", op.name());
    }

    assert_eq!(echo.ops(), every_op(), "server saw a different op sequence");
}

#[tokio::test]
async fn typed_wrappers_decode_payloads() {
    let (harness, _echo) = echo_harness().await;
    let mut client = harness.client().await;

    let members = client.list_members().await.unwrap();
    assert_eq!(members.len(), 2);
    assert_eq!(members[1].member_name, "dev-1");

    let snapshot = client.snapshot().await.unwrap();
    assert_eq!(snapshot.team_name, "agent_team");
    assert_eq!(snapshot.members.len(), 2);
    assert_eq!(snapshot.tasks.len(), 1);

    assert_eq!(client.read_inbox().await.unwrap(), "2 unread messages");

    assert!(
        client.task_detail("missing").await.unwrap().is_none(),
        "a null payload decodes to None"
    );
    let detail = client.task_detail("task-1").await.unwrap().unwrap();
    assert_eq!(detail.task_id, "task-1");

    let stopped = client.stop_team().await.unwrap();
    assert_eq!(stopped, json!({"stopped": true}));
}

#[tokio::test]
async fn handler_null_data_is_a_valid_payload() {
    let handler: Arc<dyn AttachHandler> = Arc::new(GhostHandler);
    let harness = Harness::start(handler).await;
    let mut client = harness.client().await;

    let data = client.call(AttachOp::Snapshot).await.unwrap();
    assert_eq!(data, Value::Null);
}

#[tokio::test]
async fn handler_error_keeps_the_connection_open() {
    let handler: Arc<dyn AttachHandler> = Arc::new(GhostHandler);
    let harness = Harness::start(handler).await;
    let mut client = harness.client().await;

    let error = client.member_status("ghost").await.unwrap_err();
    assert_eq!(error.kind, ErrorKind::NotFound);
    assert!(error.message.contains("ghost"), "{error}");

    let row = client.member_status("dev-1").await.unwrap();
    assert_eq!(row.member_name, "dev-1", "the same connection still serves");
}

#[tokio::test]
async fn two_concurrent_clients_both_round_trip() {
    let (harness, _echo) = echo_harness().await;
    let mut first = harness.client().await;
    let mut second = harness.client().await;

    let (members, snapshot) = tokio::join!(first.list_members(), second.snapshot());
    assert_eq!(members.unwrap().len(), 2);
    assert_eq!(snapshot.unwrap().team_name, "agent_team");

    let (left, right) = tokio::join!(first.read_inbox(), second.read_inbox());
    assert_eq!(left.unwrap(), "2 unread messages");
    assert_eq!(right.unwrap(), "2 unread messages");
}

// --- frame-level faults ----------------------------------------------------

#[tokio::test]
async fn malformed_json_is_answered_with_id_zero_then_closed() {
    let (harness, _echo) = echo_harness().await;
    let response = raw_error(harness.path(), b"not json\n").await;

    assert_eq!(response.id, 0);
    let error = response.error.unwrap();
    assert_eq!(error.kind, ErrorKind::Validation);

    let mut client = harness.client().await;
    assert_eq!(
        client.list_members().await.unwrap().len(),
        2,
        "the server survived the malformed frame"
    );
}

#[tokio::test]
async fn unknown_op_is_rejected_then_the_connection_closes() {
    let (harness, _echo) = echo_harness().await;
    let response = raw_error(harness.path(), br#"{"v":1,"id":9,"op":"nope","args":{}}"#).await;

    assert_eq!(response.id, 9);
    let error = response.error.unwrap();
    assert_eq!(error.kind, ErrorKind::Validation);
    assert!(error.message.contains("nope"), "{error}");
}

#[tokio::test]
async fn invalid_op_fields_are_rejected_then_the_connection_closes() {
    let (harness, _echo) = echo_harness().await;
    let response = raw_error(
        harness.path(),
        br#"{"v":1,"id":3,"op":"member_status","args":{"member_name":""}}"#,
    )
    .await;

    assert_eq!(response.id, 3);
    assert_eq!(response.error.unwrap().kind, ErrorKind::Validation);
}

#[tokio::test]
async fn unsupported_version_is_rejected_then_the_connection_closes() {
    let (harness, _echo) = echo_harness().await;
    let response = raw_error(
        harness.path(),
        br#"{"v":2,"id":4,"op":"list_members","args":{}}"#,
    )
    .await;

    assert_eq!(response.id, 4);
    let error = response.error.unwrap();
    assert_eq!(error.kind, ErrorKind::Validation);
    assert!(error.message.contains("version"), "{error}");
}

#[tokio::test]
async fn empty_line_is_rejected_then_the_connection_closes() {
    let (harness, _echo) = echo_harness().await;
    let response = raw_error(harness.path(), b"\n").await;

    assert_eq!(response.id, 0);
    assert_eq!(response.error.unwrap().kind, ErrorKind::Validation);
}

#[tokio::test]
async fn oversized_line_is_rejected_and_the_server_keeps_serving() {
    let (harness, _echo) = echo_harness().await;
    let stream = UnixStream::connect(harness.path()).await.unwrap();
    let (read_half, mut write_half) = stream.into_split();

    // One payload larger than a whole frame, with no newline: the server has to drain it and
    // answer the framing error once the write half is closed.
    let oversized = vec![b'x'; MAX_FRAME_BYTES + 1024];
    write_half.write_all(&oversized).await.unwrap();
    write_half.flush().await.unwrap();
    write_half.shutdown().await.unwrap();

    let mut reader = BufReader::new(read_half);
    let lines = read_until_eof(&mut reader).await;
    assert_eq!(lines.len(), 1, "expected exactly one response line");
    let response = ResponseFrame::decode(&lines[0]).unwrap();
    assert!(!response.ok);
    assert_eq!(response.id, 0);
    assert_eq!(response.error.unwrap().kind, ErrorKind::Validation);

    let mut client = harness.client().await;
    assert_eq!(client.list_members().await.unwrap().len(), 2);
}

#[tokio::test]
async fn an_empty_connection_does_not_disturb_the_server() {
    let (harness, _echo) = echo_harness().await;

    // `is_available` connects and closes immediately, which must read as a clean EOF.
    assert!(AttachClient::is_available(harness.path()).await);
    assert!(AttachClient::is_available(harness.path()).await);

    let mut client = harness.client().await;
    assert_eq!(client.list_members().await.unwrap().len(), 2);
}

// --- shutdown and drop -----------------------------------------------------

#[tokio::test]
async fn shutdown_stops_serve_and_removes_the_socket() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("s");
    let server = Arc::new(AttachServer::bind(&path).await.unwrap());
    let handler: Arc<dyn AttachHandler> = Arc::new(EchoHandler::default());
    let serving = Arc::clone(&server);
    let task = tokio::spawn(async move { serving.serve(handler).await });

    // Prove the loop is really running before stopping it.
    let mut client = AttachClient::connect(&path).await.unwrap();
    assert_eq!(client.list_members().await.unwrap().len(), 2);
    drop(client);

    server.shutdown();
    tokio::time::timeout(BOUND, task)
        .await
        .expect("serve did not return after shutdown")
        .unwrap()
        .unwrap();

    assert!(!path.exists(), "the socket file must be removed");

    let error = AttachClient::connect(&path).await.unwrap_err();
    assert_eq!(error.kind, ErrorKind::Transport);
}

#[tokio::test]
async fn cancelling_a_cloned_token_stops_serve() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("s");
    let server = Arc::new(AttachServer::bind(&path).await.unwrap());
    let token = server.cancellation_token();
    let handler: Arc<dyn AttachHandler> = Arc::new(EchoHandler::default());
    let serving = Arc::clone(&server);
    let task = tokio::spawn(async move { serving.serve(handler).await });

    let mut client = AttachClient::connect(&path).await.unwrap();
    assert_eq!(client.read_inbox().await.unwrap(), "2 unread messages");
    drop(client);

    token.cancel();
    tokio::time::timeout(BOUND, task)
        .await
        .expect("serve did not return after cancellation")
        .unwrap()
        .unwrap();
    assert!(!path.exists());
}

#[tokio::test]
async fn shutdown_closes_idle_connections() {
    let (harness, _echo) = echo_harness().await;
    let mut client = harness.client().await;
    assert_eq!(client.list_members().await.unwrap().len(), 2);

    harness.server.shutdown();

    // The server closes the idle connection, so the next call fails instead of hanging.
    let error = client.read_inbox().await.unwrap_err();
    assert!(
        matches!(
            error.kind,
            ErrorKind::Transport | ErrorKind::Timeout | ErrorKind::Cancelled
        ),
        "unexpected error kind after shutdown: {error}"
    );
}

#[tokio::test]
async fn drop_without_serving_removes_the_socket_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("s");
    {
        let server = AttachServer::bind(&path).await.unwrap();
        assert!(path.exists());
        drop(server);
    }
    assert!(!path.exists(), "drop must remove the socket file");

    // The path is reusable afterwards, without any stale-socket reclamation.
    let again = AttachServer::bind(&path).await.unwrap();
    assert_eq!(again.path(), path);
}

#[tokio::test]
async fn drop_leaves_a_replaced_regular_file_in_place() {
    let dir = tempfile::tempdir().unwrap();
    let path: PathBuf = dir.path().join("s");
    let server = AttachServer::bind(&path).await.unwrap();

    std::fs::remove_file(&path).unwrap();
    std::fs::write(&path, b"precious").unwrap();
    drop(server);

    assert_eq!(std::fs::read(&path).unwrap(), b"precious");
}
