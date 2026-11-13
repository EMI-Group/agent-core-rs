//! `18_team_attach` — drive a team hosted by another process over a Unix domain socket.
//!
//! The attach layer is the cross-process seam of `openjiuwen::agent_teams`: the process that hosts
//! a team binds `AttachServer` on a socket path (what `TeamAgent::start` does when it is given an
//! `attach` path), and a second process — the `team-member` CLI or the `openjiuwen-team-mcp`
//! server — dials it with `AttachClient` and drives the roster, the task board, the mailbox and the
//! team lifecycle through one JSON object per line.
//! Python parity: `openjiuwen/agent_teams/external/` (join descriptor and client) plus
//! `openjiuwen/agent_teams/organization/transport_api.py`; Python reaches another process through
//! shared storage and `pyzmq`, while this port carries the same operations over a socket.
//!
//! The example is two-sided but single-process: the `AttachHandler` *is* the hosted team (a small
//! `DemoTeam` over the crate's schema types) and the clients share no handle with it — everything
//! they learn arrives as a response frame.
//! A second client proves that: it re-reads the board and mailbox the first client mutated and sees
//! the same state, because the state belongs to the hosted team, not to the connection.
//!
//! The walk covers `PROTOCOL_VERSION` and `MAX_FRAME_BYTES` at bind time; `list_members`,
//! `member_status` (an unknown member is an in-band `not_found` that leaves the connection open),
//! `create_task`, `list_tasks`, `claim_task`, `complete_task`, `send_member`, `read_inbox`,
//! `snapshot` and `stop_team`; a fresh client over the same socket; and three frame-level faults
//! (unknown op, wrong `v`, oversized line) whose typed kind is asserted, never printed blindly.
//! Nothing needs a model, a network or the clock — the "board clock" is a logical counter — so two
//! runs print the same narrative.
#![allow(
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    clippy::items_after_statements
)]
mod common;
use anyhow::{ensure, Context, Result};
use async_trait::async_trait;
use openjiuwen::agent_teams::attach::protocol::to_data;
use openjiuwen::agent_teams::{
    AttachClient, AttachError, AttachHandler, AttachOp, AttachResult, AttachServer,
    ExecutionStatus, HarnessState, MemberSnapshot, MemberStatus, ResponseFrame, TaskDetail,
    TaskStatus, TaskSummary, TeamRole, TeamSnapshot, MAX_FRAME_BYTES, PROTOCOL_VERSION,
};
use openjiuwen::core::common::error::ErrorKind;
use serde_json::{json, Value};
use std::future::Future;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
/// Team name the fake hosted team reports.
const TEAM_NAME: &str = "attach-demo";
/// Leader of the fake team; the member the attached client addresses messages to.
const LEADER: &str = "team_leader";
/// The member the handler speaks for, exactly as a `team-member` CLI joins as one identity.
const ATTACHED_MEMBER: &str = "dev-1";
/// Upper bound on every socket operation, so a regression fails instead of hanging the example.
const BOUND: Duration = Duration::from_secs(10);
// --- the hosted team -------------------------------------------------------
/// Everything behind the socket: roster, board, mailbox log `(recipient, line)`, and a clock.
struct DemoTeam {
    members: Vec<MemberSnapshot>,
    tasks: Vec<TaskDetail>,
    deliveries: Vec<(String, String)>,
    clock: u64,
    stopped: bool,
}
impl DemoTeam {
    /// A team with a leader, the attached member, one open task and one seeded message.
    fn seeded() -> Self {
        Self {
            members: vec![
                roster_row(LEADER, "Team Leader", TeamRole::Leader),
                roster_row(ATTACHED_MEMBER, "Developer One", TeamRole::Teammate),
            ],
            tasks: vec![task_record(
                "task-1",
                "Review the release notes",
                "Read `RELEASE.md` and flag anything the changelog missed.",
            )],
            deliveries: vec![(
                ATTACHED_MEMBER.to_owned(),
                format!("{LEADER}: claim a task from the board when you are ready"),
            )],
            clock: 1,
            stopped: false,
        }
    }
    /// The roster row of `name`, as a fault when the member is unknown.
    fn member_mut(&mut self, name: &str) -> AttachResult<&mut MemberSnapshot> {
        self.members
            .iter_mut()
            .find(|row| row.member_name == name)
            .ok_or_else(|| unknown_member(name))
    }
    /// The task of `task_id`, as a fault when it is unknown.
    fn task_mut(&mut self, task_id: &str) -> AttachResult<&mut TaskDetail> {
        self.tasks
            .iter_mut()
            .find(|task| task.task_id == task_id)
            .ok_or_else(|| unknown_task(task_id))
    }
    /// Applies one attach op and returns the payload the client decodes.
    ///
    /// Synchronous on purpose: the whole mutation happens under one lock, so no lock is ever held
    /// across an `.await`.
    fn apply(&mut self, op: AttachOp) -> AttachResult<Value> {
        match op {
            AttachOp::ListMembers => to_data(&self.members),
            AttachOp::MemberStatus { member_name } => self
                .members
                .iter()
                .find(|row| row.member_name == member_name)
                .map_or_else(|| Err(unknown_member(&member_name)), to_data),
            AttachOp::ReadInbox => Ok(json!(self.inbox())),
            AttachOp::SendMember { to, content } => self.deliver(&to, &content),
            AttachOp::ListTasks { status } => to_data(&self.summaries(status)),
            AttachOp::CreateTask {
                title,
                content,
                task_id,
                dependencies,
            } => self.create(task_id.as_deref(), &title, &content, &dependencies),
            AttachOp::ClaimTask { task_id } => {
                self.transition(&task_id, TaskStatus::Pending, TaskStatus::InProgress)
            }
            AttachOp::CompleteTask { task_id } => {
                self.transition(&task_id, TaskStatus::InProgress, TaskStatus::Completed)
            }
            AttachOp::Snapshot => to_data(&self.snapshot()),
            AttachOp::StopTeam => {
                self.stopped = true;
                self.clock += 1;
                Ok(json!({"stopped": true}))
            }
            // A real adapter maps the whole op table onto the runtime. This team has no tool
            // catalogue, no fan-out and no edits to model, and says so instead of pretending.
            AttachOp::Broadcast { .. }
            | AttachOp::ClaimableTasks
            | AttachOp::TaskDetail { .. }
            | AttachOp::UpdateTask { .. }
            | AttachOp::InvokeTool { .. } => Err(AttachError::remote(
                ErrorKind::Unsupported,
                "the demo team does not implement this op",
            )),
        }
    }
    /// The attached member's mailbox, as the `read_inbox` op answers it.
    fn inbox(&self) -> String {
        let lines: Vec<&str> = self
            .deliveries
            .iter()
            .filter(|(to, _)| to == ATTACHED_MEMBER)
            .map(|(_, line)| line.as_str())
            .collect();
        let body = if lines.is_empty() {
            "nothing new"
        } else {
            &lines.join(" | ")
        };
        format!("{} message(s) for {ATTACHED_MEMBER}: {body}", lines.len())
    }
    /// The task board projected into list rows, optionally filtered by status.
    fn summaries(&self, status: Option<TaskStatus>) -> Vec<TaskSummary> {
        self.tasks
            .iter()
            .filter(|task| status.is_none_or(|wanted| task.status == wanted))
            .map(|task| TaskSummary {
                task_id: task.task_id.clone(),
                title: task.title.clone(),
                status: task.status,
                assignee: task.assignee.clone(),
                blocked_by: task.blocked_by.clone(),
                updated_at: task.updated_at,
                ..TaskSummary::default()
            })
            .collect()
    }
    /// The whole team, as the `snapshot` op answers it.
    fn snapshot(&self) -> TeamSnapshot {
        TeamSnapshot {
            team_name: TEAM_NAME.to_owned(),
            state: if self.stopped {
                HarnessState::Terminated
            } else {
                HarnessState::Idle
            },
            members: self.members.clone(),
            tasks: self.summaries(None),
            updated_at: self.clock,
        }
    }
    /// Adds a pending task and records the dependencies it declares.
    ///
    /// The board is flat on purpose: it keeps the declared edges but does not run a dependency
    /// graph, and this walk creates a task without one.
    fn create(
        &mut self,
        task_id: Option<&str>,
        title: &str,
        content: &str,
        dependencies: &[String],
    ) -> AttachResult<Value> {
        let task_id =
            task_id.map_or_else(|| format!("task-{}", self.tasks.len() + 1), str::to_owned);
        if self.tasks.iter().any(|task| task.task_id == task_id) {
            return Err(AttachError::remote(
                ErrorKind::Validation,
                format!("task `{task_id}` already exists"),
            ));
        }
        for dependency in dependencies {
            if !self.tasks.iter().any(|task| &task.task_id == dependency) {
                return Err(unknown_task(dependency));
            }
        }
        self.clock += 1;
        let mut detail = task_record(&task_id, title, content);
        detail.blocked_by = dependencies.to_vec();
        detail.updated_at = self.clock;
        self.tasks.push(detail);
        Ok(json!({"ok": true, "task_id": task_id, "reason": null}))
    }
    /// Moves one task from `from` to `to`, keeping the attached member's roster row in step.
    fn transition(
        &mut self,
        task_id: &str,
        from: TaskStatus,
        to: TaskStatus,
    ) -> AttachResult<Value> {
        let current = self
            .tasks
            .iter()
            .find(|task| task.task_id == task_id)
            .ok_or_else(|| unknown_task(task_id))?
            .status;
        if current != from {
            return Err(AttachError::remote(
                ErrorKind::Validation,
                format!(
                    "task `{task_id}` is `{}`, expected `{}`",
                    current.as_str(),
                    from.as_str()
                ),
            ));
        }
        self.clock += 1;
        let clock = self.clock;
        let detail = self.task_mut(task_id)?;
        detail.status = to;
        detail.updated_at = clock;
        if to == TaskStatus::InProgress {
            detail.assignee = Some(ATTACHED_MEMBER.to_owned());
        }
        let (status, execution) = if to == TaskStatus::Completed {
            (MemberStatus::Ready, ExecutionStatus::Idle)
        } else {
            (MemberStatus::Busy, ExecutionStatus::Running)
        };
        let row = self.member_mut(ATTACHED_MEMBER)?;
        row.status = status;
        row.execution = execution;
        Ok(json!({"ok": true, "reason": null}))
    }
    /// Delivers one message from the attached member into `to`'s mailbox.
    fn deliver(&mut self, to: &str, content: &str) -> AttachResult<Value> {
        if !self.members.iter().any(|row| row.member_name == to) {
            return Err(unknown_member(to));
        }
        let message_id = format!("m-{}", self.deliveries.len() + 1);
        self.deliveries
            .push((to.to_owned(), format!("{ATTACHED_MEMBER}: {content}")));
        Ok(json!({"ok": true, "message_id": message_id, "reason": null}))
    }
}
/// One roster row: `ready` and `idle` until a claim moves it.
fn roster_row(name: &str, display_name: &str, role: TeamRole) -> MemberSnapshot {
    MemberSnapshot {
        member_name: name.to_owned(),
        display_name: display_name.to_owned(),
        role,
        status: MemberStatus::Ready,
        execution: ExecutionStatus::Idle,
    }
}
/// One task detail in the state a freshly created task has.
fn task_record(task_id: &str, title: &str, content: &str) -> TaskDetail {
    TaskDetail {
        task_id: task_id.to_owned(),
        title: title.to_owned(),
        content: content.to_owned(),
        status: TaskStatus::Pending,
        ..TaskDetail::default()
    }
}
/// The `not_found` fault the hosted team reports for an unknown member.
fn unknown_member(name: &str) -> AttachError {
    AttachError::remote(ErrorKind::NotFound, format!("unknown member `{name}`"))
}
/// The `not_found` fault the hosted team reports for an unknown task.
fn unknown_task(task_id: &str) -> AttachError {
    AttachError::remote(ErrorKind::NotFound, format!("unknown task `{task_id}`"))
}
// --- the socket side -------------------------------------------------------
/// The one `AttachHandler` impl this example needs: the hosted team behind a lock.
struct DemoHandler {
    team: Mutex<DemoTeam>,
}
#[async_trait]
impl AttachHandler for DemoHandler {
    async fn handle(&self, op: AttachOp) -> AttachResult<Value> {
        let mut team = self.team.lock().map_err(|_| {
            AttachError::remote(ErrorKind::External, "the demo team lock is poisoned")
        })?;
        team.apply(op)
    }
}
/// Runs one attach call under [`BOUND`], keeping the error typed.
///
/// A timeout becomes an `AttachError` of kind `Timeout`, so a caller can assert on the kind of every
/// failure without downcasting an `anyhow` error.
async fn bounded<T>(what: &str, call: impl Future<Output = AttachResult<T>>) -> AttachResult<T> {
    match tokio::time::timeout(BOUND, call).await {
        Ok(answer) => answer,
        Err(_) => Err(AttachError::timeout(format!(
            "{what} did not answer within {}s",
            BOUND.as_secs()
        ))),
    }
}
/// Sends one hand-written request line on its own connection and returns the single response.
///
/// A frame-level fault answers exactly once and then closes the connection, so a probe never needs
/// a second read; the write half is shut down after the frame so the server also sees EOF.
async fn raw_exchange(path: &Path, request: &[u8]) -> Result<ResponseFrame> {
    let stream = tokio::net::UnixStream::connect(path)
        .await
        .context("the raw probe could not connect")?;
    let (read_half, mut write_half) = stream.into_split();
    write_half
        .write_all(request)
        .await
        .context("the raw probe could not write its frame")?;
    write_half
        .shutdown()
        .await
        .context("the raw probe could not half-close the socket")?;
    let mut reader = BufReader::new(read_half);
    let mut line = Vec::new();
    let read = tokio::time::timeout(BOUND, reader.read_until(b'\n', &mut line))
        .await
        .context("the raw probe got no response in time")?
        .context("the raw probe could not read the response")?;
    ensure!(
        read > 0,
        "the server closed without answering the raw frame"
    );
    ResponseFrame::decode(&line).context("the raw response did not decode")
}
/// The frames only the protocol layer can reject, labelled for the narrative.
fn fault_frames() -> Vec<(&'static str, Vec<u8>)> {
    let mut oversized =
        json!({"v": PROTOCOL_VERSION, "id": 93, "op": "list_members", "args": {}}).to_string();
    oversized.push_str(&" ".repeat(MAX_FRAME_BYTES));
    oversized.push('\n');
    vec![
        (
            "unknown op",
            request_line(
                &json!({"v": PROTOCOL_VERSION, "id": 91, "op": "delete_everything", "args": {}}),
            ),
        ),
        (
            "wrong version",
            request_line(
                &json!({"v": PROTOCOL_VERSION + 1, "id": 92, "op": "list_members", "args": {}}),
            ),
        ),
        ("oversized frame", oversized.into_bytes()),
    ]
}
/// The wire form of one request: the JSON object plus its newline terminator.
fn request_line(value: &Value) -> Vec<u8> {
    let mut bytes = value.to_string().into_bytes();
    bytes.push(b'\n');
    bytes
}
// --- the walkthrough -------------------------------------------------------
/// What the first client changes, so the second client can be checked against it.
struct LeftBehind {
    task_id: String,
    inbox: String,
}
/// Connects to the hosted team and sends the ops a `team-member` CLI would send.
async fn drive_first_client(path: &Path) -> Result<LeftBehind> {
    common::banner::step("first client: roster, board, mailbox, snapshot");
    let mut client = bounded("connect", AttachClient::connect(path))
        .await
        .context("the first client could not connect")?;

    for row in bounded("list_members", client.list_members()).await? {
        common::banner::note(&format!(
            "member {} — {} — {} / {}",
            row.member_name,
            row.role.as_str(),
            row.status.as_str(),
            row.execution.as_str()
        ));
    }
    let row = bounded("member_status", client.member_status(ATTACHED_MEMBER)).await?;
    common::banner::note(&format!(
        "{ATTACHED_MEMBER}: {} / {}",
        row.status.as_str(),
        row.execution.as_str()
    ));
    match bounded("member_status(ghost)", client.member_status("ghost")).await {
        Ok(_) => anyhow::bail!("the hosted team reported a member it does not have"),
        Err(error) => {
            ensure!(
                error.kind == ErrorKind::NotFound,
                "an unknown member reported `{}` instead of not_found",
                error.kind
            );
            common::banner::note(&format!(
                "unknown member -> {} ({}) — in band, the connection stays open",
                error.kind, error.message
            ));
        }
    }

    let task_id = drive_task_board(&mut client).await?;
    let inbox = drive_mailbox(&mut client).await?;
    Ok(LeftBehind { task_id, inbox })
}

/// Creates, lists, claims and completes one task; returns the id it created.
async fn drive_task_board(client: &mut AttachClient) -> Result<String> {
    let created = bounded(
        "create_task",
        client.create_task(
            "Audit the changelog",
            "Group the entries by area and flag the breaking ones.",
            Some("task-2"),
            &[],
        ),
    )
    .await?;
    ensure!(
        created.get("ok").and_then(Value::as_bool) == Some(true),
        "create_task was not acknowledged"
    );
    let task_id = created
        .get("task_id")
        .and_then(Value::as_str)
        .context("create_task answered without a task id")?
        .to_owned();
    common::banner::note(&format!("create_task -> {task_id}"));

    for task in bounded("list_tasks", client.list_tasks(None)).await? {
        common::banner::note(&format!(
            "task {} — {} ({})",
            task.task_id,
            task.title,
            task.status.as_str()
        ));
    }
    bounded("claim_task", client.claim_task(&task_id)).await?;
    let row = bounded("member_status", client.member_status(ATTACHED_MEMBER)).await?;
    common::banner::note(&format!(
        "claimed {task_id}; the roster now reports {ATTACHED_MEMBER} {} / {}",
        row.status.as_str(),
        row.execution.as_str()
    ));
    bounded("complete_task", client.complete_task(&task_id)).await?;
    for task in bounded(
        "list_tasks(completed)",
        client.list_tasks(Some(TaskStatus::Completed)),
    )
    .await?
    {
        common::banner::note(&format!(
            "completed: {} — {} (assignee {})",
            task.task_id,
            task.title,
            task.assignee.as_deref().unwrap_or("nobody")
        ));
    }
    Ok(task_id)
}

/// Sends one message, then reads the attached member's inbox and the team snapshot.
async fn drive_mailbox(client: &mut AttachClient) -> Result<String> {
    let sent = bounded(
        "send_member",
        client.send_member(LEADER, "task-2 is done; starting task-1 next"),
    )
    .await?;
    common::banner::note(&format!("send_member -> {LEADER}: {sent}"));
    let inbox = bounded("read_inbox", client.read_inbox()).await?;
    common::banner::note(&format!("read_inbox: {inbox}"));

    let snapshot = bounded("snapshot", client.snapshot()).await?;
    common::banner::note(&format!(
        "snapshot: team `{}` in state {:?}, {} member(s), {} task(s), board clock {}",
        snapshot.team_name,
        snapshot.state,
        snapshot.members.len(),
        snapshot.tasks.len(),
        snapshot.updated_at
    ));
    Ok(inbox)
}

/// Connects a second client and checks it sees the state the first client left behind.
async fn drive_second_client(path: &Path, left: &LeftBehind) -> Result<()> {
    common::banner::step("second client: the state lives with the hosted team");
    let mut client = bounded("connect (second client)", AttachClient::connect(path)).await?;

    let completed = bounded(
        "list_tasks(completed) (second client)",
        client.list_tasks(Some(TaskStatus::Completed)),
    )
    .await?;
    ensure!(
        completed.iter().any(|task| task.task_id == left.task_id),
        "the second client does not see the task the first client completed"
    );
    let inbox = bounded("read_inbox (second client)", client.read_inbox()).await?;
    ensure!(
        inbox == left.inbox,
        "the two clients read different mailboxes"
    );
    common::banner::note(&format!(
        "a fresh connection sees `{}` completed and the same inbox; the client holds no team state",
        left.task_id
    ));

    let stopped = bounded("stop_team", client.stop_team()).await?;
    ensure!(
        stopped.get("stopped").and_then(Value::as_bool) == Some(true),
        "stop_team was not acknowledged"
    );
    common::banner::note(&format!("stop_team -> {stopped}"));
    let snapshot = bounded("snapshot (after stop)", client.snapshot()).await?;
    common::banner::note(&format!(
        "team state after the stop: {:?} — the socket stays open until the host shuts it down",
        snapshot.state
    ));
    Ok(())
}

/// Sends the frames the protocol layer itself rejects and asserts their typed kind.
async fn probe_protocol_faults(path: &Path) -> Result<()> {
    common::banner::step("frame-level faults are typed and cost one connection");
    for (label, frame) in fault_frames() {
        let response = raw_exchange(path, &frame).await?;
        ensure!(!response.ok, "the server accepted the {label} frame");
        let error = response
            .error
            .context("an error response carried no error object")?;
        ensure!(
            error.kind == ErrorKind::Validation,
            "the {label} frame produced `{}` instead of validation",
            error.kind
        );
        common::banner::note(&format!("{label} -> {} ({})", error.kind, error.message));
    }
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let mode = common::setup()?;
    common::banner::title("18 team attach");

    if mode.is_live() {
        common::banner::note(
            "live mode is set, but this example needs no model: the hosted team is a local fake \
             reached over a Unix socket",
        );
    }

    common::banner::step("host: bind the attach socket");
    let dir = common::tempdir().context("no temporary directory for the attach socket")?;
    let path = dir.path().join("s");
    let handler: Arc<dyn AttachHandler> = Arc::new(DemoHandler {
        team: Mutex::new(DemoTeam::seeded()),
    });
    let server = Arc::new(
        AttachServer::bind(&path)
            .await
            .context("binding the attach socket failed")?,
    );
    let serving = {
        let server = Arc::clone(&server);
        tokio::spawn(async move { server.serve(handler).await.map_err(anyhow::Error::from) })
    };
    // The raw path holds the temp dir, so the narrative prints the file name only.
    let socket_name = path.file_name().map_or_else(
        || String::from("<socket>"),
        |name| name.to_string_lossy().into_owned(),
    );
    common::banner::note(&format!(
        "unix socket `{socket_name}` in a temp dir (mode 0600) — protocol v{PROTOCOL_VERSION}, \
         frame cap {MAX_FRAME_BYTES} bytes"
    ));
    ensure!(
        AttachClient::is_available(&path).await,
        "the freshly bound socket does not accept connections"
    );

    let left = drive_first_client(&path).await?;
    drive_second_client(&path, &left).await?;
    probe_protocol_faults(&path).await?;

    common::banner::step("host: stop serving and remove the socket");
    server.shutdown();
    let joined = tokio::time::timeout(BOUND, serving)
        .await
        .context("the serve task did not stop within the bound")?;
    joined.context("the serve task panicked")??;
    ensure!(
        !AttachClient::is_available(&path).await,
        "the socket still accepts connections after shutdown"
    );
    match AttachClient::connect(&path).await {
        Ok(_) => anyhow::bail!("a stopped host still accepts attach clients"),
        Err(error) => {
            ensure!(
                error.kind == ErrorKind::Transport,
                "a stopped host reported `{}` instead of transport",
                error.kind
            );
            // The message names the socket path, which names the temp dir: print the kind only.
            common::banner::note(&format!(
                "connect after shutdown -> {} (the socket file is gone)",
                error.kind
            ));
        }
    }

    common::banner::answer(&format!(
        "protocol v{PROTOCOL_VERSION}: roster, board, mailbox, snapshot and stop over a Unix \
         socket, with typed faults at both the frame and the op level"
    ));
    Ok(())
}
