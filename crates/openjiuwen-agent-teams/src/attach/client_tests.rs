#![cfg(unix)]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::pedantic)]
//! Tests for the attach client over real Unix domain sockets.
//!
//! Every test binds a fake server socket in a temporary directory and answers the client's request
//! lines with programmed response frames.
//! The fake server is deliberately tiny: an accept loop, one task per connection, and a responder
//! closure that maps a decoded request to the bytes to write back (or to "close now" / "stay
//! silent").
//! Nothing touches the network, nothing sleeps unboundedly: the only waits are
//! `tokio::time::timeout(10s, ..)` guards around bounded polling.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use openjiuwen_common::error::{Error, ErrorKind};
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};

use super::client::{AttachClient, CONNECT_TIMEOUT, DEFAULT_TIMEOUT};
use super::protocol::{AttachError, AttachOp, RequestFrame, ResponseFrame, MAX_FRAME_BYTES};
use crate::schema::{
    ExecutionStatus, MemberSnapshot, MemberStatus, TaskDetail, TaskStatus, TaskSummary, TeamRole,
    TeamSnapshot,
};

/// Upper bound for every wait in this file.
const BOUND: Duration = Duration::from_secs(10);

// --- fake server -----------------------------------------------------------

/// What the fake server does after reading one request.
#[derive(Debug)]
enum Reply {
    /// Write these byte chunks back, then keep reading (an empty list means "stay silent").
    Bytes(Vec<Vec<u8>>),
    /// Close the connection without answering, which the client sees as EOF.
    Close,
}

/// One responder per connection, shared across connections of the same server.
type Responder = Arc<dyn Fn(&RequestFrame) -> Reply + Send + Sync>;

/// Everything the fake server saw, in arrival order.
#[derive(Default)]
struct Records {
    /// Raw request lines, exactly as they arrived (with their trailing newline).
    raw: Vec<Vec<u8>>,
    /// The same lines decoded as request frames.
    frames: Vec<RequestFrame>,
}

/// A fake attach server bound to `<tempdir>/s`.
struct FakeServer {
    /// Keeps the socket directory alive for the lifetime of the server.
    _dir: tempfile::TempDir,
    /// Socket path the client connects to.
    path: PathBuf,
    /// Recorded requests, shared with the connection tasks.
    records: Arc<Mutex<Records>>,
    /// Accept-loop task, aborted on drop.
    handle: tokio::task::JoinHandle<()>,
}

impl Drop for FakeServer {
    fn drop(&mut self) {
        self.handle.abort();
    }
}

impl FakeServer {
    /// Binds the socket and starts accepting connections.
    fn start(responder: impl Fn(&RequestFrame) -> Reply + Send + Sync + 'static) -> Self {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s");
        let listener = UnixListener::bind(&path).unwrap();
        let records: Arc<Mutex<Records>> = Arc::new(Mutex::new(Records::default()));
        let responder: Responder = Arc::new(responder);
        let handle = tokio::spawn(accept_loop(
            listener,
            Arc::clone(&records),
            Arc::clone(&responder),
        ));
        Self {
            _dir: dir,
            path,
            records,
            handle,
        }
    }

    /// The socket path this server listens on.
    fn path(&self) -> &Path {
        &self.path
    }

    /// A path inside the same directory that nothing is listening on.
    fn missing_path(&self) -> PathBuf {
        self._dir.path().join("absent")
    }

    /// Requests recorded so far, in arrival order.
    fn frames(&self) -> Vec<RequestFrame> {
        self.records.lock().unwrap().frames.clone()
    }

    /// Raw request lines recorded so far, decoded as lossy text.
    fn raw(&self) -> Vec<String> {
        self.records
            .lock()
            .unwrap()
            .raw
            .iter()
            .map(|line| String::from_utf8_lossy(line).into_owned())
            .collect()
    }

    /// Waits (bounded) until at least `count` requests have arrived, then returns them.
    async fn wait_for(&self, count: usize) -> Vec<RequestFrame> {
        tokio::time::timeout(BOUND, async {
            loop {
                {
                    let records = self.records.lock().unwrap();
                    if records.frames.len() >= count {
                        return records.frames.clone();
                    }
                }
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .expect("fake server did not record the expected requests")
    }
}

/// Accepts connections forever, spawning one serving task per connection.
async fn accept_loop(listener: UnixListener, records: Arc<Mutex<Records>>, responder: Responder) {
    loop {
        let Ok((stream, _)) = listener.accept().await else {
            return;
        };
        let records = Arc::clone(&records);
        let responder = Arc::clone(&responder);
        tokio::spawn(serve_connection(stream, records, responder));
    }
}

/// Reads request lines from one connection and answers them with the responder's reply.
async fn serve_connection(stream: UnixStream, records: Arc<Mutex<Records>>, responder: Responder) {
    let (read_half, mut write_half) = stream.into_split();
    let mut reader = BufReader::new(read_half);
    let mut line = Vec::new();
    loop {
        line.clear();
        match reader.read_until(b'\n', &mut line).await {
            Ok(0) | Err(_) => return,
            Ok(_) => {}
        }
        let Ok(frame) = RequestFrame::decode(&line) else {
            return;
        };
        let reply = responder(&frame);
        {
            let mut recorded = records.lock().unwrap();
            recorded.raw.push(line.clone());
            recorded.frames.push(frame);
        }
        match reply {
            Reply::Close => return,
            Reply::Bytes(chunks) => {
                for chunk in chunks {
                    if write_half.write_all(&chunk).await.is_err() {
                        return;
                    }
                }
                if write_half.flush().await.is_err() {
                    return;
                }
            }
        }
    }
}

/// A responder that answers every request with a success frame carrying `data`.
fn answer_with(
    data: impl Fn(&RequestFrame) -> Value + Send + Sync + 'static,
) -> impl Fn(&RequestFrame) -> Reply + Send + Sync + 'static {
    move |frame| {
        Reply::Bytes(vec![ResponseFrame::ok(frame.id, data(frame))
            .encode()
            .unwrap()])
    }
}

/// A responder that answers every request with a fixed success payload.
fn fixed(data: Value) -> impl Fn(&RequestFrame) -> Reply + Send + Sync + 'static {
    answer_with(move |_| data.clone())
}

/// A responder that answers every request with a fixed op-level failure.
fn failing(error: AttachError) -> impl Fn(&RequestFrame) -> Reply + Send + Sync + 'static {
    move |frame| {
        Reply::Bytes(vec![ResponseFrame::err(frame.id, error.clone())
            .encode()
            .unwrap()])
    }
}

/// A responder that writes exactly these bytes for every request.
fn raw_bytes(chunks: Vec<Vec<u8>>) -> impl Fn(&RequestFrame) -> Reply + Send + Sync + 'static {
    move |_| Reply::Bytes(chunks.clone())
}

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
        blocked_by: Vec::new(),
        updated_at: 1_700,
        ..TaskSummary::default()
    }
}

/// One task detail with every field set.
fn task_detail(id: &str) -> TaskDetail {
    TaskDetail {
        task_id: id.to_owned(),
        title: "Write tests".to_owned(),
        content: "Cover the client".to_owned(),
        status: TaskStatus::InProgress,
        blocked_by: vec!["task-0".to_owned()],
        blocks: Vec::new(),
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

/// One sample of every op variant, used by the round-trip test.
fn every_op() -> Vec<AttachOp> {
    vec![
        AttachOp::ListMembers,
        AttachOp::MemberStatus {
            member_name: "dev-1".to_owned(),
        },
        AttachOp::ReadInbox,
        AttachOp::SendMember {
            to: "dev-2".to_owned(),
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
            title: "Write tests".to_owned(),
            content: "Cover the client".to_owned(),
            task_id: Some("task-2".to_owned()),
            dependencies: vec!["task-1".to_owned()],
        },
        AttachOp::UpdateTask {
            task_id: "task-1".to_owned(),
            title: Some("Rename".to_owned()),
            content: None,
        },
        AttachOp::ClaimTask {
            task_id: "task-1".to_owned(),
        },
        AttachOp::CompleteTask {
            task_id: "task-1".to_owned(),
        },
        AttachOp::InvokeTool {
            name: "read_file".to_owned(),
            arguments: json!({"path": "a.txt"}),
        },
        AttachOp::Snapshot,
        AttachOp::StopTeam,
    ]
}

// --- connect ---------------------------------------------------------------

#[tokio::test]
async fn default_deadlines_are_ten_seconds() {
    assert_eq!(DEFAULT_TIMEOUT, Duration::from_secs(10));
    assert_eq!(CONNECT_TIMEOUT, Duration::from_secs(10));
}

#[tokio::test]
async fn connect_exposes_path_and_timeout() {
    let server = FakeServer::start(fixed(json!(null)));
    let client = AttachClient::connect(server.path()).await.unwrap();
    assert_eq!(client.path(), server.path());
    assert_eq!(client.timeout(), DEFAULT_TIMEOUT);
    drop(client);

    let client = AttachClient::connect_with_timeout(server.path(), Duration::from_millis(500))
        .await
        .unwrap();
    assert_eq!(client.path(), server.path());
    assert_eq!(client.timeout(), Duration::from_millis(500));
    assert!(format!("{client:?}").contains("AttachClient"));
}

#[tokio::test]
async fn connect_to_missing_path_is_a_transport_error() {
    let server = FakeServer::start(fixed(json!(null)));
    let error = AttachClient::connect(server.missing_path())
        .await
        .expect_err("connecting to a missing socket must fail");
    assert_eq!(error.kind, ErrorKind::Transport);
    assert!(
        error.message.contains("attach connect to"),
        "unexpected message: {}",
        error.message
    );
    assert!(
        error.message.contains("failed"),
        "unexpected message: {}",
        error.message
    );
}

#[tokio::test]
async fn two_sequential_clients_both_work() {
    let server = FakeServer::start(answer_with(|frame| frame.args.clone()));

    let mut first = AttachClient::connect(server.path()).await.unwrap();
    assert_eq!(first.call(AttachOp::ReadInbox).await.unwrap(), json!({}));
    drop(first);

    let mut second = AttachClient::connect(server.path()).await.unwrap();
    assert_eq!(
        second
            .call(AttachOp::Broadcast {
                content: "hi".to_owned()
            })
            .await
            .unwrap(),
        json!({"content": "hi"})
    );
    assert_eq!(server.wait_for(2).await.len(), 2);
}

// --- framing and ids -------------------------------------------------------

#[tokio::test]
async fn requests_are_framed_with_incrementing_ids() {
    let server = FakeServer::start(fixed(json!(null)));
    let mut client = AttachClient::connect(server.path()).await.unwrap();

    client.call(AttachOp::ListMembers).await.unwrap();
    client.call(AttachOp::ReadInbox).await.unwrap();
    client.call(AttachOp::Snapshot).await.unwrap();

    let frames = server.wait_for(3).await;
    assert_eq!(
        frames.iter().map(|frame| frame.id).collect::<Vec<_>>(),
        vec![1, 2, 3]
    );
    let raw = server.raw();
    assert!(
        raw[0].starts_with(r#"{"v":1,"id":1,"op":"list_members""#),
        "{:?}",
        raw[0]
    );
    assert!(raw[0].contains(r#""args":{}"#), "{:?}", raw[0]);
    assert!(raw[0].ends_with("}\n"), "{:?}", raw[0]);
    assert!(
        raw[1].contains(r#""id":2,"op":"read_inbox""#),
        "{:?}",
        raw[1]
    );
    assert!(raw[2].contains(r#""id":3,"op":"snapshot""#), "{:?}", raw[2]);
}

#[tokio::test]
async fn call_round_trips_every_op_family() {
    let server = FakeServer::start(answer_with(|frame| frame.args.clone()));
    let mut client = AttachClient::connect(server.path()).await.unwrap();

    for op in every_op() {
        let expected = op.args().unwrap();
        let data = client.call(op.clone()).await.unwrap();
        assert_eq!(data, expected, "op `{}` did not round-trip", op.name());
    }
    assert_eq!(server.wait_for(15).await.len(), 15);
}

#[tokio::test]
async fn invalid_op_never_reaches_the_wire() {
    let server = FakeServer::start(fixed(json!(null)));
    let mut client = AttachClient::connect(server.path()).await.unwrap();

    let error = client
        .call(AttachOp::MemberStatus {
            member_name: "  ".to_owned(),
        })
        .await
        .expect_err("an empty member name must be rejected");
    assert_eq!(error.kind, ErrorKind::Validation);
    assert!(error.message.contains("member_name"), "{}", error.message);

    let error = client
        .call(AttachOp::TaskDetail {
            task_id: String::new(),
        })
        .await
        .expect_err("an empty task id must be rejected");
    assert_eq!(error.kind, ErrorKind::Validation);

    assert!(server.frames().is_empty(), "no frame may reach the wire");
}

// --- typed decoding --------------------------------------------------------

#[tokio::test]
async fn list_members_decodes_the_roster() {
    let expected = vec![member_row("team_leader"), member_row("dev-1")];
    let payload = serde_json::to_value(&expected).unwrap();
    let server = FakeServer::start(fixed(payload));
    let mut client = AttachClient::connect(server.path()).await.unwrap();

    assert_eq!(client.list_members().await.unwrap(), expected);
    let frames = server.wait_for(1).await;
    assert_eq!(frames[0].op, "list_members");
}

#[tokio::test]
async fn member_status_decodes_one_row() {
    let expected = member_row("dev-1");
    let payload = serde_json::to_value(&expected).unwrap();
    let server = FakeServer::start(fixed(payload));
    let mut client = AttachClient::connect(server.path()).await.unwrap();

    assert_eq!(client.member_status("dev-1").await.unwrap(), expected);
    let frames = server.wait_for(1).await;
    assert_eq!(frames[0].args, json!({"member_name": "dev-1"}));
}

#[tokio::test]
async fn read_inbox_decodes_text() {
    let server = FakeServer::start(fixed(json!("2 unread messages\n")));
    let mut client = AttachClient::connect(server.path()).await.unwrap();

    assert_eq!(client.read_inbox().await.unwrap(), "2 unread messages\n");
}

#[tokio::test]
async fn list_tasks_decodes_summaries() {
    let expected = vec![
        task_row("task-1", TaskStatus::Pending),
        task_row("task-2", TaskStatus::InProgress),
    ];
    let payload = serde_json::to_value(&expected).unwrap();
    let server = FakeServer::start(fixed(payload));
    let mut client = AttachClient::connect(server.path()).await.unwrap();

    assert_eq!(
        client.list_tasks(Some(TaskStatus::Pending)).await.unwrap(),
        expected
    );
    let frames = server.wait_for(1).await;
    assert_eq!(frames[0].args, json!({"status": "pending"}));

    assert_eq!(
        client.list_tasks(None).await.unwrap(),
        expected,
        "an unfiltered listing decodes the same way"
    );
    let frames = server.wait_for(2).await;
    assert_eq!(frames[1].args, json!({"status": null}));
}

#[tokio::test]
async fn claimable_tasks_decodes_summaries() {
    let expected = vec![task_row("task-7", TaskStatus::Pending)];
    let payload = serde_json::to_value(&expected).unwrap();
    let server = FakeServer::start(fixed(payload));
    let mut client = AttachClient::connect(server.path()).await.unwrap();

    assert_eq!(client.claimable_tasks().await.unwrap(), expected);
    let frames = server.wait_for(1).await;
    assert_eq!(frames[0].op, "claimable_tasks");
}

#[tokio::test]
async fn task_detail_decodes_some_and_none() {
    let server = FakeServer::start(answer_with(|frame| {
        if frame.args == json!({"task_id": "missing"}) {
            Value::Null
        } else {
            serde_json::to_value(task_detail("task-1")).unwrap()
        }
    }));
    let mut client = AttachClient::connect(server.path()).await.unwrap();

    assert_eq!(
        client.task_detail("task-1").await.unwrap(),
        Some(task_detail("task-1"))
    );
    assert_eq!(client.task_detail("missing").await.unwrap(), None);
}

#[tokio::test]
async fn snapshot_decodes_the_team_snapshot() {
    let expected = snapshot_fixture();
    let payload = serde_json::to_value(&expected).unwrap();
    let server = FakeServer::start(fixed(payload));
    let mut client = AttachClient::connect(server.path()).await.unwrap();

    assert_eq!(client.snapshot().await.unwrap(), expected);
    let frames = server.wait_for(1).await;
    assert_eq!(frames[0].op, "snapshot");
}

#[tokio::test]
async fn unexpected_payload_is_a_protocol_error() {
    let server = FakeServer::start(fixed(json!({"nope": true})));
    let mut client = AttachClient::connect(server.path()).await.unwrap();

    let error = client
        .list_members()
        .await
        .expect_err("a non-array payload must not decode");
    assert_eq!(error.kind, ErrorKind::Validation);
    assert!(
        error
            .message
            .contains("op 'list_members' returned an unexpected payload"),
        "unexpected message: {}",
        error.message
    );
}

#[tokio::test]
async fn outcome_ops_return_raw_json() {
    let payload = json!({"ok": true, "task_id": "task-9", "reason": null});
    let server = FakeServer::start(fixed(payload.clone()));
    let mut client = AttachClient::connect(server.path()).await.unwrap();
    let dependencies = vec!["task-1".to_owned()];

    assert_eq!(
        client
            .create_task("Title", "Body", Some("task-9"), &dependencies)
            .await
            .unwrap(),
        payload
    );
    assert_eq!(
        client
            .update_task("task-9", Some("New"), None)
            .await
            .unwrap(),
        payload
    );
    assert_eq!(client.claim_task("task-9").await.unwrap(), payload);
    assert_eq!(client.complete_task("task-9").await.unwrap(), payload);
    assert_eq!(client.send_member("dev-2", "ping").await.unwrap(), payload);
    assert_eq!(client.broadcast("ping all").await.unwrap(), payload);
    assert_eq!(client.stop_team().await.unwrap(), payload);

    let echo = FakeServer::start(answer_with(|frame| frame.args.clone()));
    let mut echo_client = AttachClient::connect(echo.path()).await.unwrap();
    assert_eq!(
        echo_client
            .call(AttachOp::SendMember {
                to: "dev-2".to_owned(),
                content: "ping".to_owned(),
            })
            .await
            .unwrap(),
        json!({"to": "dev-2", "content": "ping"})
    );

    let result = json!({"content": "file body"});
    let tool_server = FakeServer::start(fixed(result.clone()));
    let mut tool_client = AttachClient::connect(tool_server.path()).await.unwrap();
    assert_eq!(
        tool_client
            .invoke_tool("read_file", json!({"path": "a.txt"}))
            .await
            .unwrap(),
        result
    );
    let frames = tool_server.wait_for(1).await;
    assert_eq!(
        frames[0].args,
        json!({"name": "read_file", "arguments": {"path": "a.txt"}})
    );

    let frames = server.wait_for(7).await;
    assert_eq!(frames[0].op, "create_task");
    assert_eq!(
        frames[0].args,
        json!({
            "title": "Title",
            "content": "Body",
            "task_id": "task-9",
            "dependencies": ["task-1"],
        })
    );
    assert_eq!(
        frames[1].args,
        json!({"task_id": "task-9", "title": "New", "content": null})
    );
    assert_eq!(frames[6].op, "stop_team");
}

// --- remote failures -------------------------------------------------------

#[tokio::test]
async fn remote_not_found_error_survives() {
    let error = AttachError::new(ErrorKind::NotFound, "task missing");
    let server = FakeServer::start(failing(error));
    let mut client = AttachClient::connect(server.path()).await.unwrap();

    let error = client
        .task_detail("task-404")
        .await
        .expect_err("the remote not_found must surface");
    assert_eq!(error.kind, ErrorKind::NotFound);
    assert_eq!(error.message, "task missing");

    let shared: Error = error.clone().into();
    assert_eq!(shared.kind(), ErrorKind::NotFound);
    assert_eq!(shared.kind(), Error::from(error).kind());

    // The connection survives an op-level failure.
    let error = client
        .claim_task("task-404")
        .await
        .expect_err("the next op must still be answered");
    assert_eq!(error.kind, ErrorKind::NotFound);
    assert_eq!(server.wait_for(2).await.len(), 2);
}

#[tokio::test]
async fn remote_validation_error_kind_is_preserved() {
    let error = AttachError::new(ErrorKind::Validation, "title must not be empty");
    let server = FakeServer::start(failing(error));
    let mut client = AttachClient::connect(server.path()).await.unwrap();

    let error = client
        .send_member("dev-2", "ping")
        .await
        .expect_err("the remote validation must surface");
    assert_eq!(error.kind, ErrorKind::Validation);
    assert_eq!(error.message, "title must not be empty");
}

// --- framing failures ------------------------------------------------------

#[tokio::test]
async fn malformed_response_is_a_protocol_error() {
    let server = FakeServer::start(raw_bytes(vec![b"not json\n".to_vec()]));
    let mut client = AttachClient::connect(server.path()).await.unwrap();

    let error = client
        .call(AttachOp::ReadInbox)
        .await
        .expect_err("malformed JSON must be rejected");
    assert_eq!(error.kind, ErrorKind::Validation);
    assert!(
        error.message.contains("malformed response frame"),
        "unexpected message: {}",
        error.message
    );
}

#[tokio::test]
async fn response_id_mismatch_is_a_protocol_error() {
    let server = FakeServer::start(|frame| {
        Reply::Bytes(vec![ResponseFrame::ok(frame.id + 1, json!(null))
            .encode()
            .unwrap()])
    });
    let mut client = AttachClient::connect(server.path()).await.unwrap();

    let error = client
        .call(AttachOp::ReadInbox)
        .await
        .expect_err("a mismatched id must be rejected");
    assert_eq!(error.kind, ErrorKind::Validation);
    assert_eq!(error.message, "response id 2 did not match request id 1");
}

#[tokio::test]
async fn response_version_mismatch_is_a_protocol_error() {
    let server = FakeServer::start(raw_bytes(vec![
        b"{\"v\":2,\"id\":1,\"ok\":true,\"data\":null}\n".to_vec(),
    ]));
    let mut client = AttachClient::connect(server.path()).await.unwrap();

    let error = client
        .call(AttachOp::ReadInbox)
        .await
        .expect_err("another protocol version must be rejected");
    assert_eq!(error.kind, ErrorKind::Validation);
    assert!(
        error.message.contains("unsupported protocol version 2"),
        "unexpected message: {}",
        error.message
    );
}

#[tokio::test]
async fn oversized_response_is_a_protocol_error() {
    let mut body = vec![b'x'; MAX_FRAME_BYTES + 16];
    body.push(b'\n');
    let server = FakeServer::start(raw_bytes(vec![body]));
    let mut client = AttachClient::connect(server.path()).await.unwrap();

    let error = client
        .call(AttachOp::ReadInbox)
        .await
        .expect_err("an oversized frame must be rejected");
    assert_eq!(error.kind, ErrorKind::Validation);
    assert!(
        error.message.contains("exceeds"),
        "unexpected message: {}",
        error.message
    );
}

#[tokio::test]
async fn eof_before_response_is_a_transport_error() {
    let server = FakeServer::start(|_| Reply::Close);
    let mut client = AttachClient::connect(server.path()).await.unwrap();

    let error = client
        .call(AttachOp::ReadInbox)
        .await
        .expect_err("a closed connection must be reported");
    assert_eq!(error.kind, ErrorKind::Transport);
    assert_eq!(
        error.message,
        "attach connection closed before a response arrived"
    );
}

#[tokio::test]
async fn silent_server_times_out() {
    let answered = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&answered);
    let server = FakeServer::start(move |_| {
        counter.fetch_add(1, Ordering::SeqCst);
        Reply::Bytes(Vec::new())
    });
    let mut client = AttachClient::connect_with_timeout(server.path(), Duration::from_millis(80))
        .await
        .unwrap();

    let error = client
        .call(AttachOp::ReadInbox)
        .await
        .expect_err("a silent server must time out");
    assert_eq!(error.kind, ErrorKind::Timeout);
    assert!(
        error.message.contains("attach request 'read_inbox'"),
        "unexpected message: {}",
        error.message
    );
    assert!(
        error.message.contains("timed out after 80ms"),
        "unexpected message: {}",
        error.message
    );
    assert_eq!(
        answered.load(Ordering::SeqCst),
        1,
        "the request was received"
    );
    assert_eq!(server.wait_for(1).await.len(), 1);
}

// --- availability probe ----------------------------------------------------

#[tokio::test]
async fn is_available_tracks_the_socket_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("s");
    assert!(
        !AttachClient::is_available(&path).await,
        "a missing socket is not available"
    );

    let listener = UnixListener::bind(&path).unwrap();
    assert!(
        AttachClient::is_available(&path).await,
        "a listening socket is available"
    );
    drop(listener);
    std::fs::remove_file(&path).unwrap();
    assert!(
        !AttachClient::is_available(&path).await,
        "a removed socket is not available"
    );
}

#[tokio::test]
async fn is_available_probes_a_live_server_without_sending_a_frame() {
    let server = FakeServer::start(fixed(json!(null)));
    assert!(AttachClient::is_available(server.path()).await);
    assert!(
        server.frames().is_empty(),
        "the probe connects and immediately drops the socket"
    );
}

// --- misc ------------------------------------------------------------------

#[tokio::test]
async fn a_response_without_data_decodes_as_null() {
    let server = FakeServer::start(|frame| {
        Reply::Bytes(vec![format!(
            "{{\"v\":1,\"id\":{},\"ok\":true}}\n",
            frame.id
        )
        .into_bytes()])
    });
    let mut client = AttachClient::connect(server.path()).await.unwrap();

    assert_eq!(client.call(AttachOp::StopTeam).await.unwrap(), Value::Null);
    assert_eq!(client.task_detail("task-1").await.unwrap(), None);
}
