#![cfg(unix)]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::pedantic)]
//! Tests for [`SocketTeamClient`](crate::skill::SocketTeamClient) over real Unix domain sockets.
//!
//! Every case runs against a genuine `AttachServer` bound inside a temporary directory
//! (`<tmp>/s`, so the path stays well inside the platform's socket-path limit) and a hand-written
//! `AttachHandler` that answers every `AttachOp` variant with distinctive fixture data while
//! recording what it saw. That is what makes the assertions about the *wire* possible: the decode
//! tests prove the op each method builds, and the refusal tests prove an in-band `ok = false` never
//! turns into an `Err`.
//!
//! Nothing here touches the network beyond the local socket, reads the process environment or the
//! process-global runtime registry, and nothing sleeps unboundedly: every await is bounded either by
//! the client's own attach deadline (`DEFAULT_TIMEOUT`, 10 s) or by this file's `BOUND` cap, and the
//! only polling loop is the `wait_until` helper (10 ms steps, 10 s cap). A regression therefore
//! fails instead of hanging. The whole file is `#![cfg(unix)]`, because the attach transport
//! requires Unix domain sockets; the non-Unix stubs are covered where they live
//! (`crate::attach::server_tests`).
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use openjiuwen_common::error::ErrorKind;
use serde_json::{json, Value};
use tokio::task::JoinHandle;

use crate::attach::{AttachError, AttachHandler, AttachOp, AttachResult, AttachServer};
use crate::constants::MAX_MESSAGE_CONTENT_CHARS;
use crate::schema::{
    DispatchMode, ExecutionStatus, MemberMode, MemberSnapshot, MemberStatus, TaskDetail,
    TaskStatus, TaskSummary, TeamRole,
};

use super::client::{
    operator_tools, CreateOutcome, MemberSummary, OpOutcome, SendOutcome, TeamClient,
};
use super::descriptor::{ClientScope, TeamJoinDescriptor};
use super::live::connect_socket_client;
use super::socket_client::SocketTeamClient;

/// Upper bound for every wait in this file.
const BOUND: Duration = Duration::from_secs(10);

/// Inbox text the fake host renders, chosen so a socket client is recognisable by its output.
const INBOX: &str = "[1700] boss -> user : over the socket";

/// Reason the fake host refuses `update_task("locked", ..)` with.
const LOCKED_REASON: &str = "task 'locked' is no longer pending or blocked";

/// Reason the fake host refuses `claim_task("taken")` with.
const TAKEN_REASON: &str = "task 'taken' is already claimed by dev-1";

/// Reason the fake host refuses `create_task("dup", ..)` with.
const DUP_REASON: &str = "task id 'dup' already exists";

// --- fixtures --------------------------------------------------------------

/// One roster row with every wire field set.
fn member_row(name: &str, role: TeamRole, status: MemberStatus) -> MemberSnapshot {
    MemberSnapshot {
        member_name: name.to_owned(),
        display_name: format!("{name} display"),
        role,
        status,
        execution: ExecutionStatus::Idle,
    }
}

/// One task list row.
fn task_row(id: &str, status: TaskStatus) -> TaskSummary {
    TaskSummary {
        task_id: id.to_owned(),
        title: format!("Task {id}"),
        status,
        updated_at: 1_700,
        ..TaskSummary::default()
    }
}

/// The detail payload the fake host answers `task_detail` with.
fn task_detail(id: &str) -> TaskDetail {
    TaskDetail {
        task_id: id.to_owned(),
        title: "Write tests".to_owned(),
        content: "Cover the socket client".to_owned(),
        status: TaskStatus::InProgress,
        assignee: Some("dev-1".to_owned()),
        blocked_by: vec!["task-0".to_owned()],
        blocks: vec!["task-2".to_owned()],
        ..TaskDetail::default()
    }
}

/// The `data` payload the fake host answers `op` with.
///
/// Every variant is answered, and the ids `locked` / `taken` / `dup` / `missing` / `unknown`
/// select the refusal branches the tests pin.
fn answer(op: &AttachOp) -> AttachResult<Value> {
    Ok(match op {
        AttachOp::ListMembers => serde_json::to_value(vec![
            member_row("boss", TeamRole::Leader, MemberStatus::Busy),
            member_row("dev-1", TeamRole::Teammate, MemberStatus::Ready),
        ])
        .unwrap(),
        AttachOp::MemberStatus { member_name } => serde_json::to_value(member_row(
            member_name,
            TeamRole::Teammate,
            MemberStatus::Ready,
        ))
        .unwrap(),
        AttachOp::ReadInbox => json!(INBOX),
        AttachOp::SendMember { .. } => json!({"ok": true, "message_id": "m-1", "reason": null}),
        AttachOp::Broadcast { .. } => json!({"ok": true, "message_id": "m-2", "reason": null}),
        AttachOp::ListTasks { .. } => serde_json::to_value(vec![
            task_row("task-1", TaskStatus::Pending),
            task_row("task-2", TaskStatus::InProgress),
        ])
        .unwrap(),
        AttachOp::ClaimableTasks => {
            serde_json::to_value(vec![task_row("task-1", TaskStatus::Pending)]).unwrap()
        }
        AttachOp::TaskDetail { task_id } if task_id == "missing" => Value::Null,
        AttachOp::TaskDetail { task_id } => serde_json::to_value(task_detail(task_id)).unwrap(),
        AttachOp::CreateTask { title, .. } if title == "dup" => {
            json!({"ok": false, "task_id": null, "reason": DUP_REASON})
        }
        AttachOp::CreateTask { .. } => json!({"ok": true, "task_id": "task-9", "reason": null}),
        AttachOp::UpdateTask { task_id, .. } if task_id == "locked" => {
            json!({"ok": false, "reason": LOCKED_REASON})
        }
        AttachOp::UpdateTask { .. } => json!({"ok": true, "reason": null}),
        AttachOp::ClaimTask { task_id } if task_id == "taken" => {
            json!({"ok": false, "reason": TAKEN_REASON})
        }
        AttachOp::ClaimTask { .. } => json!({"ok": true, "reason": null}),
        AttachOp::CompleteTask { .. } => json!({"ok": true, "reason": null}),
        AttachOp::InvokeTool { name, .. } if name == "unknown" => {
            return Err(AttachError::remote(
                ErrorKind::NotFound,
                "unknown tool 'unknown'",
            ))
        }
        AttachOp::InvokeTool { name, arguments } => json!({"tool": name, "arguments": arguments}),
        AttachOp::Snapshot => Value::Null,
        AttachOp::StopTeam => json!({"stopped": true}),
    })
}

// --- fake host -------------------------------------------------------------

/// Handler that answers every op with [`answer`] and records what it saw.
#[derive(Default)]
struct RecordingHost {
    /// Ops in arrival order.
    seen: Mutex<Vec<AttachOp>>,
}

impl RecordingHost {
    /// Returns the ops recorded so far, in arrival order.
    fn ops(&self) -> Vec<AttachOp> {
        self.seen.lock().unwrap().clone()
    }
}

#[async_trait]
impl AttachHandler for RecordingHost {
    async fn handle(&self, op: AttachOp) -> AttachResult<Value> {
        self.seen.lock().unwrap().push(op.clone());
        answer(&op)
    }
}

// --- harness ---------------------------------------------------------------

/// A bound attach server plus its serving task, shut down on drop.
struct Harness {
    /// Directory holding the socket; `None` when the caller owns the directory instead.
    _dir: Option<tempfile::TempDir>,
    /// The bound server.
    server: Arc<AttachServer>,
    /// The serving task; taken by [`Harness::shutdown`] so it can be awaited.
    handle: Option<JoinHandle<openjiuwen_common::error::Result<()>>>,
}

impl Harness {
    /// Binds a fresh `<tmp>/s` and serves it with `handler`.
    async fn start(handler: Arc<dyn AttachHandler>) -> Self {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s");
        let mut harness = Self::at(&path, handler).await;
        harness._dir = Some(dir);
        harness
    }

    /// Binds `path` (creating its parents) and serves it with `handler`.
    async fn at(path: &Path, handler: Arc<dyn AttachHandler>) -> Self {
        let server = Arc::new(AttachServer::bind(path).await.unwrap());
        let serving = Arc::clone(&server);
        let handle = tokio::spawn(async move { serving.serve(handler).await });
        Self {
            _dir: None,
            server,
            handle: Some(handle),
        }
    }

    /// The socket this harness serves.
    fn path(&self) -> &Path {
        self.server.path()
    }

    /// Shuts the server down and waits for the accept loop to stop.
    ///
    /// Awaiting the loop is what makes the "the host is gone" cases deterministic: `serve` closes
    /// every connection it was serving before it returns, so the next client exchange cannot be
    /// answered by a straggler.
    async fn shutdown(mut self) {
        self.server.shutdown();
        let handle = self.handle.take().expect("the serving task is running");
        tokio::time::timeout(BOUND, handle)
            .await
            .expect("serve did not stop after shutdown")
            .expect("the serving task did not panic")
            .expect("serve reports a clean shutdown");
    }
}

impl Drop for Harness {
    fn drop(&mut self) {
        self.server.shutdown();
        if let Some(handle) = self.handle.take() {
            handle.abort();
        }
    }
}

/// A harness whose handler is the recording [`RecordingHost`], plus that handler.
async fn recording_harness() -> (Harness, Arc<RecordingHost>) {
    let host = Arc::new(RecordingHost::default());
    let shared: Arc<dyn AttachHandler> = host.clone();
    (Harness::start(shared).await, host)
}

/// Polls `predicate` every 10 ms until it holds, or until [`BOUND`] expires.
///
/// Returns the last observation, so a caller can assert on it with its own message.
async fn wait_until(predicate: impl Fn() -> bool) -> bool {
    let deadline = tokio::time::Instant::now() + BOUND;
    loop {
        if predicate() {
            return true;
        }
        if tokio::time::Instant::now() >= deadline {
            return false;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

// --- clients ---------------------------------------------------------------

/// An operator descriptor for `team_name`.
fn operator_descriptor(team_name: &str) -> TeamJoinDescriptor {
    TeamJoinDescriptor {
        session_id: "session-socket".to_owned(),
        team_name: team_name.to_owned(),
        member_name: "user".to_owned(),
        scope: ClientScope::Operator,
        ..TeamJoinDescriptor::default()
    }
}

/// A lazily connecting client for `path` acting under `descriptor`.
fn lazy_client(descriptor: TeamJoinDescriptor, path: &Path) -> SocketTeamClient {
    SocketTeamClient::new(descriptor, path)
}

// --- decode ----------------------------------------------------------------

#[tokio::test]
async fn every_team_client_method_decodes() {
    let (harness, host) = recording_harness().await;
    let client = lazy_client(operator_descriptor("skill-socket"), harness.path());

    // The descriptor decides the surface; `new` has opened nothing yet.
    assert_eq!(client.scope(), ClientScope::Operator);
    assert_eq!(client.member_name(), "user");
    assert_eq!(client.path(), harness.path());
    assert_eq!(client.descriptor().team_name, "skill-socket");
    assert_eq!(client.timeout(), crate::attach::client::DEFAULT_TIMEOUT);
    assert!(!client.is_connected().await, "new() does no I/O");

    assert_eq!(client.read_inbox().await.unwrap(), INBOX);
    assert!(
        client.is_connected().await,
        "the first call connected lazily"
    );

    assert_eq!(
        client.send_message("dev-1", "hello").await.unwrap(),
        SendOutcome::sent("m-1")
    );
    assert_eq!(
        client.send_message(" * ", "hello all").await.unwrap(),
        SendOutcome::sent("m-2"),
        "the target is trimmed before the broadcast check"
    );

    let created = client
        .create_task("Ship it", "body", Some("task-9"), &["task-1".to_owned()])
        .await
        .unwrap();
    assert_eq!(created, CreateOutcome::created("task-9"));

    assert_eq!(
        client
            .update_task("task-1", Some("Renamed"), Some("new body"))
            .await
            .unwrap(),
        OpOutcome::ok()
    );

    let listed = client.list_tasks(Some(TaskStatus::Pending)).await.unwrap();
    let ids: Vec<&str> = listed.iter().map(|task| task.task_id.as_str()).collect();
    assert_eq!(ids, vec!["task-1", "task-2"]);
    assert_eq!(listed[0].status, TaskStatus::Pending);
    assert_eq!(listed[1].status, TaskStatus::InProgress);

    let claimable = client.claimable_tasks().await.unwrap();
    assert_eq!(claimable.len(), 1);
    assert_eq!(claimable[0].task_id, "task-1");

    let detail = client
        .get_task("task-1")
        .await
        .unwrap()
        .expect("the host has task-1");
    assert_eq!(detail.task_id, "task-1");
    assert_eq!(detail.title, "Write tests");
    assert_eq!(detail.content, "Cover the socket client");
    assert_eq!(detail.status, TaskStatus::InProgress);
    assert_eq!(detail.assignee.as_deref(), Some("dev-1"));
    assert_eq!(detail.blocked_by, vec!["task-0".to_owned()]);
    assert_eq!(detail.blocks, vec!["task-2".to_owned()]);

    assert_eq!(client.claim_task("task-1").await.unwrap(), OpOutcome::ok());
    assert_eq!(
        client.complete_task("task-1").await.unwrap(),
        OpOutcome::ok()
    );

    let roster = client.list_members().await.unwrap();
    assert_eq!(
        roster,
        vec![
            MemberSummary::new("boss", TeamRole::Leader, MemberStatus::Busy),
            MemberSummary::new("dev-1", TeamRole::Teammate, MemberStatus::Ready),
        ],
        "the wire snapshot is projected onto the CLI row"
    );

    let echoed = client
        .invoke_tool("list_tasks", json!({"status": "pending"}))
        .await
        .unwrap();
    assert_eq!(
        echoed,
        json!({"tool": "list_tasks", "arguments": {"status": "pending"}})
    );

    assert_eq!(
        client.tools().len(),
        operator_tools().len(),
        "an operator scope advertises the operator catalogue"
    );

    // The assertions above only hold if the ops really crossed the socket.
    let ops = host.ops();
    assert!(
        ops.contains(&AttachOp::SendMember {
            to: "dev-1".to_owned(),
            content: "hello".to_owned(),
        }),
        "{ops:?}"
    );
    assert!(
        ops.contains(&AttachOp::Broadcast {
            content: "hello all".to_owned(),
        }),
        "{ops:?}"
    );
    assert!(
        ops.contains(&AttachOp::ListTasks {
            status: Some(TaskStatus::Pending),
        }),
        "the status filter reached the wire: {ops:?}"
    );
    assert!(
        ops.contains(&AttachOp::CreateTask {
            title: "Ship it".to_owned(),
            content: "body".to_owned(),
            task_id: Some("task-9".to_owned()),
            dependencies: vec!["task-1".to_owned()],
        }),
        "{ops:?}"
    );
}

#[tokio::test]
async fn task_detail_miss_decodes_to_none() {
    let (harness, _host) = recording_harness().await;
    let client = lazy_client(operator_descriptor("skill-socket"), harness.path());

    let missing = client.get_task("missing").await.unwrap();
    assert!(missing.is_none(), "a null payload is an absent task");
    assert!(
        client.is_connected().await,
        "a null payload is not a failure"
    );
}

#[tokio::test]
async fn member_scope_tools_follow_the_descriptor() {
    let descriptor = TeamJoinDescriptor {
        team_name: "skill-socket".to_owned(),
        member_name: "dev-1".to_owned(),
        scope: ClientScope::Member,
        teammate_mode: MemberMode::PlanMode,
        dispatch_mode: DispatchMode::Scheduled,
        max_subtask_depth: 2,
        ..TeamJoinDescriptor::default()
    };
    // `tools` is local metadata: no harness and no socket are needed.
    let client = lazy_client(descriptor, Path::new("unused.sock"));

    let tools = client.tools();
    let names: Vec<&str> = tools.iter().map(|tool| tool.name.as_str()).collect();
    assert_eq!(
        names,
        vec![
            "view_task",
            "send_message",
            "member_complete_task",
            "submit_plan",
            "create_task",
            "update_task",
        ],
        "scheduled dispatch completes instead of claiming, plan mode adds submit_plan, \
         subtasks need a positive depth"
    );
}

// --- refusals --------------------------------------------------------------

#[tokio::test]
async fn refusals_keep_the_outcome_shapes() {
    let (harness, _host) = recording_harness().await;
    let client = lazy_client(operator_descriptor("skill-socket"), harness.path());

    let updated = client
        .update_task("locked", None, Some("new body"))
        .await
        .unwrap();
    assert!(!updated.ok);
    assert_eq!(updated.reason.as_deref(), Some(LOCKED_REASON));

    let claimed = client.claim_task("taken").await.unwrap();
    assert!(!claimed.ok);
    assert_eq!(claimed.reason.as_deref(), Some(TAKEN_REASON));

    let duplicate = client.create_task("dup", "body", None, &[]).await.unwrap();
    assert!(!duplicate.ok);
    assert!(duplicate.task_id.is_none());
    assert_eq!(duplicate.reason.as_deref(), Some(DUP_REASON));
}

#[tokio::test]
async fn remote_errors_keep_their_kind_and_the_connection() {
    let (harness, _host) = recording_harness().await;
    let client = lazy_client(operator_descriptor("skill-socket"), harness.path());

    assert_eq!(client.read_inbox().await.unwrap(), INBOX);

    let error = client.invoke_tool("unknown", json!({})).await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::NotFound);
    assert!(
        error.to_string().contains("unknown tool 'unknown'"),
        "{error}"
    );
    assert!(
        client.is_connected().await,
        "an in-band error says nothing about the health of the connection"
    );
    assert_eq!(
        client.read_inbox().await.unwrap(),
        INBOX,
        "the same connection still serves"
    );
}

#[tokio::test]
async fn local_checks_run_before_the_wire() {
    let (harness, host) = recording_harness().await;
    let client = lazy_client(operator_descriptor("skill-socket"), harness.path());

    let error = client.send_message("   ", "hello").await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(
        error.to_string().contains("target member name is required"),
        "{error}"
    );

    let over_limit = "x".repeat(MAX_MESSAGE_CONTENT_CHARS + 1);
    let refused = client.send_message("dev-1", &over_limit).await.unwrap();
    assert!(!refused.ok);
    assert!(refused.message_id.is_none());
    let reason = refused.reason.unwrap_or_default();
    assert!(reason.contains("content too long"), "{reason}");
    assert!(
        reason.contains(&(MAX_MESSAGE_CONTENT_CHARS + 1).to_string()),
        "the reported length is the real one: {reason}"
    );

    assert!(
        !client.is_connected().await,
        "neither local check opens the socket"
    );
    let sends = host
        .ops()
        .into_iter()
        .filter(|op| matches!(op, AttachOp::SendMember { .. } | AttachOp::Broadcast { .. }))
        .count();
    assert_eq!(sends, 0, "neither local check reached the host");
}

// --- transport faults ------------------------------------------------------

#[tokio::test]
async fn missing_socket_is_a_typed_error() {
    let dir = tempfile::tempdir().unwrap();
    let path: PathBuf = dir.path().join("absent");

    let lazy = lazy_client(operator_descriptor("skill-socket"), &path);
    let error = lazy.read_inbox().await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Transport);
    assert!(
        !lazy.is_connected().await,
        "a failed connect caches nothing"
    );

    let error = SocketTeamClient::connect(operator_descriptor("skill-socket"), &path)
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Transport);

    let error = connect_socket_client(&path, operator_descriptor("skill-socket"))
        .await
        .err()
        .expect("a missing socket must not resolve");
    assert_eq!(error.kind(), ErrorKind::Transport);
}

#[tokio::test]
async fn concurrent_calls_are_safe_on_one_client() {
    let (harness, host) = recording_harness().await;
    let client = Arc::new(lazy_client(
        operator_descriptor("skill-socket"),
        harness.path(),
    ));

    let mut calls = Vec::new();
    for index in 0..8 {
        let client = Arc::clone(&client);
        calls.push(tokio::spawn(async move {
            if index % 2 == 0 {
                client.read_inbox().await
            } else {
                client
                    .list_tasks(None)
                    .await
                    .map(|tasks| format!("{} tasks", tasks.len()))
            }
        }));
    }
    for (index, call) in calls.into_iter().enumerate() {
        let value = call.await.expect("the call task did not panic").unwrap();
        let expected = if index % 2 == 0 {
            INBOX.to_owned()
        } else {
            "2 tasks".to_owned()
        };
        assert_eq!(value, expected, "call {index} decoded its own answer");
    }
    assert_eq!(
        host.ops().len(),
        8,
        "one serialised connection served every call"
    );
}

#[tokio::test]
async fn a_transport_failure_drops_the_connection_for_a_reconnect() {
    let (harness, _host) = recording_harness().await;
    let path = harness.path().to_path_buf();
    let client = lazy_client(operator_descriptor("skill-socket"), &path);

    assert_eq!(client.read_inbox().await.unwrap(), INBOX);
    assert!(client.is_connected().await);

    // Shutting the server down closes the connection it was serving, and awaiting the accept loop
    // removes the socket file before the harness is dropped.
    harness.shutdown().await;
    assert!(
        wait_until(|| !path.exists()).await,
        "the socket file must be gone: {}",
        path.display()
    );

    let error = client.read_inbox().await.unwrap_err();
    assert!(
        matches!(error.kind(), ErrorKind::Transport | ErrorKind::Timeout),
        "a host that is gone is a transport-level failure: {error}"
    );
    assert!(
        !client.is_connected().await,
        "the failed exchange dropped the stale connection"
    );

    // A fresh host on the same path proves the next call reconnects lazily.
    let _second = Harness::at(&path, Arc::new(RecordingHost::default())).await;
    assert_eq!(
        client.read_inbox().await.unwrap(),
        INBOX,
        "the next call reconnected"
    );
    assert!(client.is_connected().await);
}

#[tokio::test]
async fn clean_shutdown_unlinks_the_socket_and_fails_typed() {
    // The directory outlives the server here, so the missing file proves `shutdown` unlinked it.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("s");
    let harness = Harness::at(&path, Arc::new(RecordingHost::default())).await;
    let client = lazy_client(operator_descriptor("skill-socket"), &path);

    assert_eq!(client.read_inbox().await.unwrap(), INBOX);

    harness.shutdown().await;
    assert!(dir.path().exists(), "the test still owns the directory");
    assert!(!path.exists(), "shutdown removed the socket file");

    let error = client.read_inbox().await.unwrap_err();
    assert_eq!(
        error.kind(),
        ErrorKind::Transport,
        "the peer is gone, which reads as a transport failure: {error}"
    );

    let fresh = lazy_client(operator_descriptor("skill-socket"), &path);
    let error = fresh.read_inbox().await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Transport);
}
