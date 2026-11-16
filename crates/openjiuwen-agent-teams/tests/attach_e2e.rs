//! End-to-end cross-process attach: a team built with `metadata["attach"]` is drivable from outside.
//!
//! A team whose spec names an attach socket is a *host*: `start()` binds a Unix domain socket and
//! serves it with the team's backend, which is the seam a foreign `team-member` CLI or
//! `openjiuwen-team-mcp` process dials. This suite proves that seam from the outside, using only the
//! public API:
//!
//! 1. the socket file exists only after `start()`, for every documented metadata shape;
//! 2. a [`SocketTeamClient`] — the CLI's `TeamClient` over the wire — lists the roster, reads the
//!    leader's inbox, sends mail that lands in the recipient's stored mailbox, and snapshots the
//!    team;
//! 3. the same client creates, edits, claims and completes a task on the hosting team's board;
//! 4. `invoke_tool` round-trips a member-facing team tool (`view_task`) through the host's shared
//!    dispatcher and answers with that tool's own wire shape, and a rejected call — an unknown tool
//!    name, reported in band as `not_found` — leaves the very same connection serving;
//! 5. an op-level fault (`not_found`) travels in band and keeps the connection, while a frame-level
//!    fault is answered once and ends that connection — and neither disturbs the team;
//! 6. the `stop_team` op answers `{"stopping": true}`, cancels the open run, and the host tears the
//!    socket down; a second stop is a typed failure that returns immediately.
//!
//! Everything is offline and deterministic: the `support` fakes answer every member input with one
//! scripted observation triple, sockets live in their own `tempfile::TempDir` (short `sun_path`),
//! and every wait is bounded by [`support::BOUND`].
//!
//! Whole file is `#![cfg(unix)]`: on other targets the server reports `Unsupported` and there is no
//! transport to drive.
#![cfg(unix)]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod support;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use openjiuwen_agent_teams::schema::{
    ExecutionStatus, MemberSnapshot, MemberStatus, TaskStatus, TeamAgentSpec, TeamRole,
};
use openjiuwen_agent_teams::skill::{
    ClientScope, MemberSummary, SocketTeamClient, TeamClient, TeamJoinDescriptor,
};
use openjiuwen_agent_teams::storage::{InMemoryTeamStore, TeamStore};
use openjiuwen_agent_teams::{
    AttachClient, AttachOp, RequestFrame, ResponseFrame, SpecBuild, TeamAgent, TeamBuildDeps,
};
use openjiuwen_common::error::ErrorKind;
use serde_json::{json, Value};
use tempfile::TempDir;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::unix::{OwnedReadHalf, OwnedWriteHalf};
use tokio::time::{timeout, Instant};

use support::{build_deps, FakeHarnessFactory, FakeWorkspaceProvider, Fixture, BOUND};

/// Poll interval of [`connect_raw`].
const POLL: Duration = Duration::from_millis(5);

/// Session id every hosted team is activated under.
const SESSION: &str = "session-a";

/// The roster `support::sample_spec()` builds: leader, predefined teammate and the `teammate`
/// blueprint row, sorted the way the hosting backend reports them.
const ROSTER: [&str; 3] = ["dev-1", "team_leader", "teammate"];

/// The metadata shapes `metadata["attach"]` documents: a bare path or an object naming it.
#[derive(Debug, Clone, Copy)]
enum Shape {
    /// `"attach": "<path>"`.
    Bare,
    /// `"attach": {"socket": "<path>"}`.
    Socket,
    /// `"attach": {"path": "<path>"}`.
    Path,
    /// `"attach": {"socket": "<path>", "required": true}`.
    Required,
}

impl Shape {
    /// Renders the metadata value this shape publishes for `socket`.
    fn metadata(self, socket: &Path) -> Value {
        let path = socket.to_str().expect("a UTF-8 socket path");
        match self {
            Self::Bare => json!(path),
            Self::Socket => json!({"socket": path}),
            Self::Path => json!({"path": path}),
            Self::Required => json!({"socket": path, "required": true}),
        }
    }
}

/// One team blueprint, its build dependencies, its tempdir and its attach socket.
///
/// `new` prepares everything (no socket is bound yet); [`Prepared::build`] builds the handle and
/// [`Prepared::start`] starts it, which is what binds the socket.
struct Prepared {
    /// Keeps the socket's directory alive for the lifetime of the fixture.
    _dir: TempDir,
    /// The socket path named by the metadata.
    socket: PathBuf,
    /// The blueprint under test.
    spec: TeamAgentSpec,
    /// The injected fakes (harness factory, workspace provider, in-memory store).
    deps: TeamBuildDeps,
    /// The store the hosting team persists into, so the suite can read the rows the socket wrote.
    store: Arc<InMemoryTeamStore>,
    /// Workspace provider, kept alive with the dependencies.
    _workspace: Arc<FakeWorkspaceProvider>,
}

impl Prepared {
    /// Builds the blueprint with `attach` metadata of `shape` pointing at `<tempdir>/<name>`.
    fn new(shape: Shape, name: &str) -> Self {
        let dir = tempfile::Builder::new()
            .prefix("ojt")
            .tempdir()
            .expect("a temp directory");
        let socket = dir.path().join(name);
        let mut spec = support::sample_spec();
        spec.metadata
            .insert("attach".to_owned(), shape.metadata(&socket));
        let factory = FakeHarnessFactory::new();
        let workspace = FakeWorkspaceProvider::new();
        let store: Arc<InMemoryTeamStore> = Arc::new(InMemoryTeamStore::new());
        let deps = build_deps(&factory, &workspace, Some(Arc::clone(&store)));
        Self {
            _dir: dir,
            socket,
            spec,
            deps,
            store,
            _workspace: workspace,
        }
    }

    /// Builds the team handle; nothing is bound or started yet.
    async fn build(&self) -> Arc<TeamAgent> {
        self.spec
            .build_with(self.deps.clone())
            .await
            .expect("the team did not build")
    }

    /// Builds and starts the team, which binds the attach socket.
    async fn start(&self) -> Arc<TeamAgent> {
        let agent = self.build().await;
        agent.start().await.expect("the team did not start");
        agent
    }

    /// Builds the join descriptor an outside operator process would carry.
    fn descriptor(&self) -> TeamJoinDescriptor {
        TeamJoinDescriptor {
            session_id: SESSION.to_owned(),
            team_name: "alpha".to_owned(),
            member_name: "external".to_owned(),
            role: TeamRole::Teammate,
            scope: ClientScope::Operator,
            socket_path: Some(
                self.socket
                    .to_str()
                    .expect("a UTF-8 socket path")
                    .to_owned(),
            ),
            ..TeamJoinDescriptor::default()
        }
    }

    /// Connects the CLI-shaped client eagerly, the way an outside process resolves the socket.
    async fn client(&self) -> SocketTeamClient {
        SocketTeamClient::connect(self.descriptor(), &self.socket)
            .await
            .expect("the socket client did not connect")
    }
}

/// Connects a raw attach client, retrying until the accept loop answers or [`BOUND`] expires.
async fn connect_raw(socket: &Path) -> AttachClient {
    let deadline = Instant::now() + BOUND;
    loop {
        match AttachClient::connect(socket).await {
            Ok(client) => return client,
            Err(error) => {
                assert!(
                    Instant::now() < deadline,
                    "no attach server answered at {}: {}",
                    socket.display(),
                    error.message
                );
                tokio::time::sleep(POLL).await;
            }
        }
    }
}

/// Returns the roster names in the order a [`TeamClient`] reports them.
fn summary_names(rows: &[MemberSummary]) -> Vec<String> {
    rows.iter().map(|row| row.member_name.clone()).collect()
}

/// Returns the roster names in the order a snapshot reports them.
fn snapshot_names(rows: &[MemberSnapshot]) -> Vec<String> {
    rows.iter().map(|row| row.member_name.clone()).collect()
}

/// Asserts the projected roster the hosting backend reports for `support::sample_spec()`.
fn assert_roster(rows: &[MemberSummary]) {
    assert_eq!(
        summary_names(rows),
        ROSTER,
        "the roster is sorted by member name"
    );
    let row = |name: &str| {
        rows.iter()
            .find(|row| row.member_name == name)
            .unwrap_or_else(|| panic!("'{name}' is missing from the roster"))
    };
    assert_eq!(row("team_leader").role, TeamRole::Leader);
    assert_eq!(
        row("team_leader").status,
        MemberStatus::Ready,
        "start readied the leader"
    );
    assert_eq!(row("dev-1").role, TeamRole::Teammate);
    assert_eq!(
        row("dev-1").status,
        MemberStatus::Unstarted,
        "nobody spawned dev-1 in this run"
    );
    assert_eq!(row("teammate").role, TeamRole::Teammate);
}

/// Writes one encoded request frame to a raw connection.
async fn send_request(writer: &mut OwnedWriteHalf, id: u64, op: &AttachOp) {
    let frame = RequestFrame::new(id, op).expect("a request frame");
    writer
        .write_all(&frame.encode().expect("an encoded request frame"))
        .await
        .expect("the request write");
}

/// Reads one response frame from a raw connection, failing the test if the server stays silent.
async fn read_response(reader: &mut BufReader<OwnedReadHalf>) -> ResponseFrame {
    let mut line = String::new();
    let read = timeout(BOUND, reader.read_line(&mut line))
        .await
        .expect("the response did not arrive")
        .expect("the response read");
    assert!(
        read > 0,
        "the server closed the connection instead of answering"
    );
    ResponseFrame::decode(line.as_bytes()).expect("a well-formed response frame")
}

#[tokio::test]
async fn the_socket_file_exists_only_after_start() {
    let prepared = Prepared::new(Shape::Bare, "team.sock");

    assert!(
        !prepared.socket.exists(),
        "preparing a blueprint writes nothing"
    );
    let agent = prepared.build().await;
    assert!(!prepared.socket.exists(), "a build never binds the socket");
    assert_eq!(agent.attach_socket(), None, "a build publishes no host");
    assert!(
        !AttachClient::is_available(&prepared.socket).await,
        "nothing answers before start"
    );

    agent.start().await.expect("the team did not start");
    assert_eq!(
        agent.attach_socket(),
        Some(prepared.socket.clone()),
        "start publishes the bound path"
    );
    assert!(
        prepared.socket.exists(),
        "start must bind the configured socket"
    );
    assert!(
        AttachClient::is_available(&prepared.socket).await,
        "the bound socket accepts connections"
    );

    agent.start().await.expect("a second start is a no-op");
    assert_eq!(
        agent.attach_socket(),
        Some(prepared.socket.clone()),
        "one host per team"
    );

    agent.stop().await.expect("the team did not stop");
    assert_eq!(agent.attach_socket(), None, "stop drops the host");
    assert!(
        !prepared.socket.exists(),
        "stop must remove the socket file"
    );
    assert!(!AttachClient::is_available(&prepared.socket).await);
}

#[tokio::test]
async fn every_documented_attach_shape_binds_and_serves_a_socket() {
    let shapes = [
        ("bare", Shape::Bare),
        ("socket", Shape::Socket),
        ("path", Shape::Path),
        ("required", Shape::Required),
    ];
    for (name, shape) in shapes {
        let prepared = Prepared::new(shape, &format!("{name}.sock"));
        let agent = prepared.start().await;
        assert!(
            prepared.socket.exists(),
            "the {name} shape must bind its socket"
        );
        assert_eq!(agent.attach_socket(), Some(prepared.socket.clone()));

        let mut client = connect_raw(&prepared.socket).await;
        let mut names: Vec<String> = client
            .list_members()
            .await
            .expect("list_members")
            .into_iter()
            .map(|row| row.member_name)
            .collect();
        names.sort_unstable();
        assert_eq!(names, ROSTER, "the {name} shape serves the whole roster");

        agent.stop().await.expect("the team did not stop");
        assert!(
            !prepared.socket.exists(),
            "the {name} shape removes its socket on stop"
        );
    }
}

#[tokio::test]
async fn an_outside_client_drives_the_roster_inbox_mail_and_snapshot() {
    let prepared = Prepared::new(Shape::Socket, "team.sock");
    let agent = prepared.start().await;
    let client = prepared.client().await;

    assert!(client.is_connected().await, "connect() opens eagerly");
    assert_eq!(client.path(), prepared.socket.as_path());
    assert_eq!(client.descriptor().team_name, "alpha");
    assert_eq!(client.member_name(), "external");
    assert_eq!(client.scope(), ClientScope::Operator);

    let roster = client.list_members().await.expect("list_members");
    assert_roster(&roster);

    // `member_status` is not part of the CLI client; the raw attach client serves it, and the wire
    // payload carries the display name and execution status the CLI projection drops.
    let mut raw = connect_raw(&prepared.socket).await;
    let dev = raw.member_status("dev-1").await.expect("member_status");
    assert_eq!(dev.member_name, "dev-1");
    assert_eq!(dev.display_name, "Dev One");
    assert_eq!(dev.status, MemberStatus::Unstarted);
    assert_eq!(dev.execution, ExecutionStatus::Idle);

    // The inbox is the leader's unread mail, rendered by the hosting team.
    assert_eq!(
        client.read_inbox().await.expect("read_inbox"),
        "",
        "the leader has no mail yet"
    );

    // A direct message is sent as the team leader and persisted on the recipient's mailbox.
    let outcome = client
        .send_message("dev-1", "please pick up the attach task")
        .await
        .expect("send_message");
    assert_eq!(outcome.reason, None, "{outcome:?}");
    // The hosting handler answers `send_member` with the persisted `MessageRecord`, so the CLI
    // outcome's `ok` keeps its serde default; the row identity below is what proves the send landed.
    let message_id = outcome
        .message_id
        .expect("the reply carries the persisted row id");
    let rows = prepared
        .store
        .unread_messages("alpha", "dev-1")
        .await
        .expect("the recipient's mailbox");
    assert_eq!(rows.len(), 1, "{rows:?}");
    assert_eq!(rows[0].message_id, message_id);
    assert_eq!(
        rows[0].from_member_name, "team_leader",
        "the handler acts as the leader"
    );
    assert_eq!(rows[0].to_member_name.as_deref(), Some("dev-1"));
    assert_eq!(rows[0].content, "please pick up the attach task");
    assert!(!rows[0].broadcast, "a direct row, not a broadcast");
    assert!(!rows[0].is_read, "the recipient has not read it yet");

    // Mail addressed to the leader comes back through `read_inbox` on the next round trip.
    let replied = raw
        .send_member("team_leader", "status?")
        .await
        .expect("send_member");
    assert_eq!(replied["from_member_name"], "team_leader");
    assert_eq!(replied["to_member_name"], "team_leader");
    assert_eq!(
        client.read_inbox().await.expect("read_inbox"),
        "team_leader: status?"
    );
    assert!(
        prepared
            .store
            .unread_messages("alpha", "dev-1")
            .await
            .expect("the recipient's mailbox")
            .len()
            == 1,
        "reading the leader's inbox touches no other mailbox"
    );

    // `snapshot` is not part of the CLI client either; the raw attach client serves it.
    let snapshot = raw.snapshot().await.expect("snapshot");
    assert_eq!(snapshot.team_name, "alpha");
    assert_eq!(
        snapshot_names(&snapshot.members),
        ROSTER,
        "the snapshot roster is sorted"
    );
    assert!(snapshot.tasks.is_empty(), "the board is still empty");

    agent.stop().await.expect("the team did not stop");
}

#[tokio::test]
async fn an_outside_client_creates_edits_claims_and_completes_a_task() {
    let prepared = Prepared::new(Shape::Path, "team.sock");
    let agent = prepared.start().await;
    let client = prepared.client().await;
    let mut raw = connect_raw(&prepared.socket).await;

    assert!(
        client
            .claimable_tasks()
            .await
            .expect("claimable_tasks")
            .is_empty(),
        "the board starts empty"
    );

    let created = client
        .create_task(
            "Write the attach tests",
            "drive the socket from outside",
            Some("task-a"),
            &[],
        )
        .await
        .expect("create_task");
    // The reply is the persisted task row, not the CLI outcome object, so only the row identity is
    // asserted here; the board is what proves the creation.
    assert_eq!(created.task_id.as_deref(), Some("task-a"), "{created:?}");
    assert!(created.reason.is_none(), "{created:?}");

    let claimable = client.claimable_tasks().await.expect("claimable_tasks");
    assert_eq!(claimable.len(), 1, "{claimable:?}");
    assert_eq!(claimable[0].task_id, "task-a");
    assert_eq!(claimable[0].status, TaskStatus::Pending);
    assert!(claimable[0].assignee.is_none());
    assert_eq!(claimable[0].title, "Write the attach tests");

    let updated = client
        .update_task(
            "task-a",
            Some("Write the attach tests"),
            Some("drive the socket from a foreign process"),
        )
        .await
        .expect("update_task");
    assert!(updated.reason.is_none(), "{updated:?}");

    let claimed = client.claim_task("task-a").await.expect("claim_task");
    assert!(claimed.ok, "{claimed:?}");
    let open = client
        .list_tasks(Some(TaskStatus::InProgress))
        .await
        .expect("list_tasks");
    assert_eq!(open.len(), 1, "{open:?}");
    assert_eq!(open[0].task_id, "task-a");
    assert_eq!(
        open[0].assignee.as_deref(),
        Some("team_leader"),
        "the handler claims as the leader"
    );
    assert!(
        client
            .claimable_tasks()
            .await
            .expect("claimable_tasks")
            .is_empty(),
        "a claimed task is no longer claimable"
    );

    let detail = client
        .get_task("task-a")
        .await
        .expect("get_task")
        .expect("the task exists");
    assert_eq!(detail.title, "Write the attach tests");
    assert_eq!(detail.content, "drive the socket from a foreign process");
    assert_eq!(detail.assignee.as_deref(), Some("team_leader"));
    assert!(detail.blocked_by.is_empty(), "no dependencies: {detail:?}");
    assert_eq!(detail.subtasks.len(), 0, "no subtasks: {detail:?}");
    assert!(detail.reviewers.is_empty(), "no review gate on this team");

    let completed = client.complete_task("task-a").await.expect("complete_task");
    assert!(completed.reason.is_none(), "{completed:?}");
    let done = client
        .get_task("task-a")
        .await
        .expect("get_task")
        .expect("the completed task exists");
    assert_eq!(done.status, TaskStatus::Completed);
    assert_eq!(done.summary.as_deref(), Some("completed via attach"));
    assert!(
        client
            .list_tasks(Some(TaskStatus::Pending))
            .await
            .expect("list_tasks")
            .is_empty(),
        "the finished task left the pending queue"
    );

    let snapshot = raw.snapshot().await.expect("snapshot");
    assert_eq!(snapshot.tasks.len(), 1, "{snapshot:?}");
    assert_eq!(snapshot.tasks[0].task_id, "task-a");
    assert_eq!(snapshot.tasks[0].status, TaskStatus::Completed);
    assert_eq!(
        snapshot_names(&snapshot.members),
        ROSTER,
        "the snapshot carries members and tasks together"
    );

    agent.stop().await.expect("the team did not stop");
}

#[tokio::test]
async fn invoke_tool_round_trips_and_survives_an_in_band_rejection() {
    let prepared = Prepared::new(Shape::Socket, "team.sock");
    let agent = prepared.start().await;
    let client = prepared.client().await;

    // Seed the board so the `view_task` listing has exactly one row to project.
    client
        .create_task(
            "Write the attach tests",
            "drive the socket from outside",
            Some("task-a"),
            &[],
        )
        .await
        .expect("create_task");

    // A tool call crosses the socket and answers with the tool's own wire shape: one entry per
    // board task carrying the four `view_task` list fields, exactly as the in-process tool answers.
    let listed = timeout(
        BOUND,
        client.invoke_tool("view_task", json!({"action": "list"})),
    )
    .await
    .expect("invoke_tool did not answer")
    .expect("view_task");
    assert_eq!(
        listed,
        json!([{
            "task_id": "task-a",
            "title": "Write the attach tests",
            "status": "pending",
            "assignee": null,
        }]),
        "the host answers with the `view_task` list projection"
    );

    // An unknown tool is refused in band as `not_found`, not as a frame-level fault.
    let error = timeout(BOUND, client.invoke_tool("nope", json!({})))
        .await
        .expect("the rejected call did not answer")
        .expect_err("'nope' is not a team tool");
    assert_eq!(error.kind(), ErrorKind::NotFound, "{error}");
    assert!(error.message().contains("nope"), "{}", error.message());
    assert!(
        client.is_connected().await,
        "an in-band rejection keeps the connection"
    );

    // The rejected call cost nothing: the next round trip on that same connection is answered.
    let relisted = timeout(
        BOUND,
        client.invoke_tool("view_task", json!({"action": "list"})),
    )
    .await
    .expect("the surviving call did not answer")
    .expect("the connection still serves");
    assert_eq!(
        relisted, listed,
        "the surviving call answers the same listing"
    );
    assert!(client.is_connected().await, "the connection is still held");

    agent.stop().await.expect("the team did not stop");
}

#[tokio::test]
async fn op_level_faults_keep_the_connection_and_frame_level_faults_close_it() {
    let prepared = Prepared::new(Shape::Socket, "team.sock");
    let agent = prepared.start().await;

    let stream = tokio::net::UnixStream::connect(&prepared.socket)
        .await
        .expect("a raw connection to the hosted socket");
    let (read_half, mut write_half) = stream.into_split();
    let mut reader = BufReader::new(read_half);

    // An op-level fault (unknown member) travels in band …
    send_request(
        &mut write_half,
        1,
        &AttachOp::MemberStatus {
            member_name: "ghost".to_owned(),
        },
    )
    .await;
    let frame = read_response(&mut reader).await;
    assert_eq!(frame.id, 1, "the reply echoes the request id");
    assert!(!frame.ok, "a remote failure is `ok: false`");
    let error = frame.into_result().expect_err("ghost is not on the roster");
    assert_eq!(error.kind, ErrorKind::NotFound, "{}", error.message);
    assert!(error.message.contains("ghost"), "{}", error.message);

    // … and the next request on the very same connection is answered.
    send_request(&mut write_half, 2, &AttachOp::ListMembers).await;
    let frame = read_response(&mut reader).await;
    assert_eq!(frame.id, 2);
    assert!(frame.ok, "{frame:?}");
    let members: Vec<MemberSnapshot> = serde_json::from_value(frame.data.expect("roster data"))
        .expect("the roster payload is a member array");
    assert_eq!(snapshot_names(&members), ROSTER);

    // A frame-level fault is answered once and ends that connection: an unknown op is deliberate.
    write_half
        .write_all(b"{\"v\":1,\"id\":9,\"op\":\"nope\"}\n")
        .await
        .expect("the raw frame write");
    let frame = read_response(&mut reader).await;
    assert_eq!(frame.id, 9, "the envelope parsed, so the id survives");
    assert!(!frame.ok);
    let error = frame.into_result().expect_err("an unknown op is refused");
    assert_eq!(error.kind, ErrorKind::Validation, "{}", error.message);
    assert!(error.message.contains("nope"), "{}", error.message);
    let mut rest = String::new();
    let read = timeout(BOUND, reader.read_line(&mut rest))
        .await
        .expect("the server must close, not hang")
        .expect("the post-fault read");
    assert_eq!(read, 0, "a frame-level fault closes the connection");

    // The team is untouched and keeps serving new connections.
    let mut fresh = connect_raw(&prepared.socket).await;
    assert_eq!(
        fresh.list_members().await.expect("list_members").len(),
        ROSTER.len()
    );

    // The CLI-shaped client sees the same in-band `not_found` and keeps its connection usable.
    let client = prepared.client().await;
    let error = client
        .get_task("ghost-task")
        .await
        .expect_err("the task does not exist");
    assert_eq!(error.kind(), ErrorKind::NotFound, "{error}");
    assert!(
        client.is_connected().await,
        "an in-band error keeps the connection"
    );
    assert!(
        client
            .list_tasks(None)
            .await
            .expect("the connection still serves")
            .is_empty(),
        "the next op on the same connection is answered"
    );

    agent.stop().await.expect("the team did not stop");
}

#[tokio::test]
async fn stop_team_answers_stopping_cancels_the_run_and_the_host_removes_the_socket() {
    let fixture = Fixture::new();
    fixture.factory.hanging_for("team_leader");
    let dir = tempfile::Builder::new()
        .prefix("ojt")
        .tempdir()
        .expect("a temp directory");
    let socket = dir.path().join("team.sock");
    let mut spec = support::sample_spec();
    spec.metadata.insert(
        "attach".to_owned(),
        json!({"socket": socket.to_str().expect("a UTF-8 socket path")}),
    );
    let agent = fixture.activate(&spec, SESSION).await;
    assert!(
        socket.exists(),
        "activation starts the team, which binds the socket"
    );

    let client = SocketTeamClient::connect(
        TeamJoinDescriptor {
            session_id: SESSION.to_owned(),
            team_name: "alpha".to_owned(),
            member_name: "external".to_owned(),
            socket_path: Some(socket.to_str().expect("a UTF-8 socket path").to_owned()),
            ..TeamJoinDescriptor::default()
        },
        &socket,
    )
    .await
    .expect("the socket client did not connect");
    assert_eq!(
        client.list_members().await.expect("list_members").len(),
        ROSTER.len()
    );

    // One run in flight, so the stop request has something to cancel; the `join!` keeps both
    // futures on this task and the stop waits for the leader's first turn.
    let stop_request = async {
        support::wait_until("the leader's first turn", || {
            fixture
                .factory
                .harness("team_leader")
                .is_some_and(|harness| harness.sends().iter().any(|text| text == "start"))
        })
        .await;
        let mut raw = connect_raw(&socket).await;
        timeout(BOUND, raw.stop_team())
            .await
            .expect("stop_team did not answer")
            .expect("stop_team")
    };
    let (frames, reply) = tokio::join!(
        support::collect(agent.stream(json!("start"), None)),
        stop_request
    );
    assert_eq!(reply, json!({"stopping": true}));
    assert_eq!(frames.len(), 1, "the cancellation ends the run: {frames:?}");
    match &frames[0] {
        Err(error) => assert_eq!(error.kind(), ErrorKind::Cancelled, "{error}"),
        Ok(chunk) => panic!("expected the cancellation, got {chunk:?}"),
    }
    assert!(
        socket.exists(),
        "the op asks for the stop; the host owns the teardown"
    );

    assert!(
        fixture
            .manager
            .stop_team("alpha", SESSION)
            .await
            .expect("stop_team through the manager"),
        "the owning session may stop the team"
    );
    assert!(!agent.is_started(), "the handle was torn down");
    assert!(
        !socket.exists(),
        "teardown removes the socket file the host bound"
    );
    assert!(
        fixture.manager.live("alpha").is_none(),
        "the pool forgot the team"
    );
    assert!(
        !fixture
            .manager
            .stop_team("alpha", SESSION)
            .await
            .expect("a second manager stop"),
        "a stopped team is no longer active"
    );

    assert_team_is_gone(&socket, &client).await;
}

/// Asserts a stopped team answers nothing: a second stop attempt is a typed failure, not a hang.
async fn assert_team_is_gone(socket: &Path, client: &SocketTeamClient) {
    let error = timeout(BOUND, AttachClient::connect(socket))
        .await
        .expect("connect must not hang")
        .expect_err("the socket is gone");
    assert_eq!(error.kind, ErrorKind::Transport, "{}", error.message);
    let error = timeout(BOUND, client.list_members())
        .await
        .expect("the client must not hang")
        .expect_err("the hosted team is gone");
    assert_eq!(error.kind(), ErrorKind::Transport, "{error}");
    assert!(
        !client.is_connected().await,
        "the transport failure dropped the connection"
    );
    assert!(!AttachClient::is_available(socket).await);
}
