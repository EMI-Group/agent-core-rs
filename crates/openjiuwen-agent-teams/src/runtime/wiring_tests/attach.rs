#![cfg(unix)]
//! `metadata["attach"]` / `OPENJIUWEN_TEAM_SOCKET` → a bound, served and torn-down Unix socket.
//!
//! The build resolves an [`AttachConfig`](crate::runtime::spec_meta::AttachConfig) once; the team's
//! `start` binds an [`AttachServer`](crate::attach::AttachServer) and serves it with a
//! [`BackendAttachHandler`](crate::runtime::attach_host::BackendAttachHandler) over the team's
//! backend, and `stop`/`stop_team` remove the socket again. These cases drive that wiring over real
//! short-path sockets in a temp directory, through the real
//! [`AttachClient`](crate::attach::AttachClient): the roster, mail, the task board and the snapshot
//! all come back over the wire, `invoke_tool` dispatches the member-facing team tools through the
//! shared in-process dispatcher (an unknown tool or bad arguments answer in band), and the
//! bind-failure policy (`required` or not) is pinned.
//!
//! Whole file is `#![cfg(unix)]`: on other targets `AttachServer::bind`/`serve` return
//! `Error::Unsupported` and there is no transport to test, so the module compiles to nothing.
use super::{
    blueprint, deps, wire, with_metadata, RecordingWorkspaceProvider, ScriptedFactory, BOUND,
};
use crate::attach::{AttachClient, AttachHandler};
use crate::runtime::attach_host::{AttachHost, BackendAttachHandler};
use crate::runtime::manager::TeamRuntimeManager;
use crate::runtime::spec_meta::{attach_config, AttachConfig, ATTACH_SOCKET_ENV};
use crate::schema::{MemberStatus, TeamAgentSpec, TeamRole};
use openjiuwen_common::error::ErrorKind;
use serde_json::json;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::time::Instant;
/// Serialises the cases that read or write the process-global attach socket variable.
///
/// The variable is process-wide while the lib suite runs its tests in parallel, so no case here
/// starts a team through the environment fallback: the parsing cases take this lock, and the teams
/// that really bind take their path from metadata.
static ENV_LOCK: Mutex<()> = Mutex::new(());
/// Poll interval of [`connect`].
const POLL: Duration = Duration::from_millis(5);
/// Takes [`ENV_LOCK`], ignoring poisoning so one failing case does not mask the others.
fn env_guard() -> std::sync::MutexGuard<'static, ()> {
    ENV_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}
/// Creates a temp directory plus a short socket path inside it (`<tmp>/s`).
///
/// The `sun_path` limit is about 108 bytes, so every socket here stays well below it.
fn socket_in_tempdir() -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::Builder::new()
        .prefix("ojt")
        .tempdir()
        .expect("a temp directory");
    let socket = dir.path().join("s");
    (dir, socket)
}
/// Builds the one-teammate blueprint served over `socket`.
fn attached_team(socket: &Path) -> TeamAgentSpec {
    with_metadata(
        blueprint("alpha", "boss", &["dev-1"]),
        "attach",
        json!({"socket": socket}),
    )
}
/// Connects to `path`, retrying until the accept loop answers or [`BOUND`] expires.
async fn connect(path: &Path) -> AttachClient {
    let deadline = Instant::now() + BOUND;
    loop {
        match AttachClient::connect(path).await {
            Ok(client) => return client,
            Err(error) => {
                assert!(
                    Instant::now() < deadline,
                    "no attach server answered at {}: {}",
                    path.display(),
                    error.message
                );
                tokio::time::sleep(POLL).await;
            }
        }
    }
}
#[tokio::test]
async fn metadata_binds_and_serves_every_backend_op() {
    let (_dir, socket) = socket_in_tempdir();
    let wiring = wire(&attached_team(&socket), |deps| deps).await;
    wiring.agent.start().await.expect("the team did not start");
    assert_eq!(wiring.agent.attach_socket(), Some(socket.clone()));
    assert!(socket.exists(), "start must bind the configured socket");
    let mut client = connect(&socket).await;
    let members = client.list_members().await.expect("list_members");
    let mut names: Vec<&str> = members.iter().map(|row| row.member_name.as_str()).collect();
    names.sort_unstable();
    assert_eq!(names, vec!["boss", "dev-1"], "the leader plus the roster");
    assert_eq!(
        members
            .iter()
            .filter(|row| row.role == TeamRole::Teammate)
            .count(),
        1
    );
    let dev = client.member_status("dev-1").await.expect("member_status");
    assert_eq!(dev.member_name, "dev-1");
    assert_eq!(dev.status, MemberStatus::Unstarted, "nobody spawned dev-1");
    assert_eq!(client.read_inbox().await.expect("read_inbox"), "");
    let row = client
        .send_member("dev-1", "hi")
        .await
        .expect("send_member");
    assert_eq!(row["to_member_name"], "dev-1");
    assert_eq!(
        row["from_member_name"], "boss",
        "the handler acts as leader"
    );
    assert_eq!(row["content"], "hi");
    let created = client
        .create_task("Write", "the tests", None, &[])
        .await
        .expect("create_task");
    let task_id = created["task_id"].as_str().expect("a task id").to_owned();
    assert_eq!(created["title"], "Write");
    let claimed = client.claim_task(&task_id).await.expect("claim_task");
    assert_eq!(claimed["ok"], true);
    let completed = client.complete_task(&task_id).await.expect("complete_task");
    assert_eq!(completed["status"], "completed");
    let detail = client
        .task_detail(&task_id)
        .await
        .expect("task_detail")
        .expect("the task exists");
    assert_eq!(detail.task_id, task_id);
    let snapshot = client.snapshot().await.expect("snapshot");
    assert!(snapshot
        .members
        .iter()
        .any(|row| row.member_name == "dev-1"));
    assert_eq!(
        snapshot.tasks.len(),
        1,
        "the completed task is on the board"
    );
    let viewed = client
        .invoke_tool("view_task", json!({"status": "completed"}))
        .await
        .expect("view_task is a member-facing team tool");
    let rows = viewed.as_array().expect("view_task answers a task list");
    assert_eq!(rows.len(), 1, "the board holds exactly the completed task");
    assert_eq!(rows[0]["task_id"].as_str(), Some(task_id.as_str()));
    assert_eq!(rows[0]["title"], json!("Write"));
    let unknown = client
        .invoke_tool("shell", json!({"cmd": "true"}))
        .await
        .expect_err("shell is not a team tool");
    assert_eq!(unknown.kind, ErrorKind::NotFound, "{}", unknown.message);
    assert!(
        unknown.message.contains("shell"),
        "the message names the unknown tool: {}",
        unknown.message
    );
    let bad_arguments = client
        .invoke_tool("claim_task", json!({}))
        .await
        .expect_err("claim_task needs a task id");
    assert_eq!(
        bad_arguments.kind,
        ErrorKind::Validation,
        "{}",
        bad_arguments.message
    );
    let missing = client
        .member_status("ghost")
        .await
        .expect_err("ghost is not on the roster");
    assert_eq!(missing.kind, ErrorKind::NotFound, "{}", missing.message);
    wiring.agent.stop().await.expect("stop");
}
#[tokio::test]
async fn invoke_tool_sends_a_message_as_the_leader() {
    let (_dir, socket) = socket_in_tempdir();
    let wiring = wire(&attached_team(&socket), |deps| deps).await;
    wiring.agent.start().await.expect("the team did not start");
    let mut client = connect(&socket).await;
    let reply = client
        .invoke_tool("send_message", json!({"to": "dev-1", "content": "hi"}))
        .await
        .expect("send_message is a member-facing team tool");
    assert_eq!(reply["ok"], json!(true));
    assert!(
        reply["reason"].is_null(),
        "an applied send carries no reason: {reply}"
    );
    let message_id = reply["message_id"]
        .as_str()
        .expect("a message id")
        .to_owned();
    assert_ne!(message_id, "", "the reply names the stored row");
    assert_eq!(
        client.read_inbox().await.expect("read_inbox"),
        "",
        "the leader sends, so its own inbox stays empty"
    );
    let unread = wiring
        .agent
        .backend()
        .unread("dev-1")
        .await
        .expect("dev-1's mailbox");
    assert_eq!(unread.len(), 1, "the tool wrote exactly one row");
    assert_eq!(unread[0].message_id, message_id);
    assert_eq!(unread[0].from_member_name, "boss", "the op acts as leader");
    assert_eq!(unread[0].to_member_name.as_deref(), Some("dev-1"));
    assert_eq!(unread[0].content, "hi");
    wiring.agent.stop().await.expect("stop");
}
#[tokio::test]
async fn stopping_through_the_manager_tears_the_socket_down() {
    let (_dir, socket) = socket_in_tempdir();
    let factory = ScriptedFactory::new();
    let workspace = RecordingWorkspaceProvider::new();
    let manager = TeamRuntimeManager::new(deps(&factory, &workspace));
    let spec = attached_team(&socket);
    let agent = manager
        .activate(&spec, "session-a", &json!({}))
        .await
        .expect("the manager did not activate the team");
    assert_eq!(agent.attach_socket(), Some(socket.clone()));
    let mut client = connect(&socket).await;
    let members = client.list_members().await.expect("list_members");
    assert_eq!(
        members.len(),
        2,
        "the socket must serve the leader and the teammate"
    );
    assert!(
        manager.stop_team("alpha", "session-a").await.expect("stop"),
        "the owning session may stop the team"
    );
    assert!(!socket.exists(), "stop_team must remove the socket");
    assert!(manager.live("alpha").is_none(), "the pool forgot the team");
    let error = client
        .list_members()
        .await
        .expect_err("the server is gone after teardown");
    assert_eq!(error.kind, ErrorKind::Transport, "{}", error.message);
}
#[tokio::test]
async fn plain_stop_releases_the_socket_and_later_calls_fail() {
    let (_dir, socket) = socket_in_tempdir();
    let wiring = wire(&attached_team(&socket), |deps| deps).await;
    wiring.agent.start().await.expect("the team did not start");
    let mut client = connect(&socket).await;
    let members = client.list_members().await.expect("list_members");
    assert_eq!(
        members.len(),
        2,
        "the socket must serve the leader and the teammate"
    );
    wiring.agent.stop().await.expect("stop");
    assert_eq!(wiring.agent.attach_socket(), None, "the host is dropped");
    assert!(!socket.exists(), "stop must remove the socket");
    let error = client
        .snapshot()
        .await
        .expect_err("the connection is dead after teardown");
    assert_eq!(error.kind, ErrorKind::Transport, "{}", error.message);
}
#[tokio::test]
async fn the_stop_team_op_acknowledges_and_cancels_the_run_token() {
    let (_dir, socket) = socket_in_tempdir();
    let wiring = wire(&attached_team(&socket), |deps| deps).await;
    wiring.agent.start().await.expect("the team did not start");
    let run_token = wiring.agent.cancellation_token();
    assert!(!run_token.is_cancelled());
    let mut client = connect(&socket).await;
    let reply = client.stop_team().await.expect("stop_team");
    assert_eq!(reply, json!({"stopping": true}));
    assert!(
        run_token.is_cancelled(),
        "the op cancels the token the runtime runs under"
    );
    assert!(
        socket.exists(),
        "the op asks for a stop; the host tears down on stop"
    );
    wiring.agent.stop().await.expect("stop");
    assert!(!socket.exists());
    assert_eq!(wiring.agent.attach_socket(), None);
}
#[tokio::test]
async fn an_unbindable_socket_is_not_fatal() {
    let (dir, _socket) = socket_in_tempdir();
    let blocked = dir.path().join("blocked");
    std::fs::write(&blocked, b"a regular file, not a directory").expect("the blocker file");
    let socket = blocked.join("s");
    let wiring = wire(&attached_team(&socket), |deps| deps).await;
    wiring
        .agent
        .start()
        .await
        .expect("a best-effort socket must not fail the start");
    assert_eq!(wiring.agent.attach_socket(), None);
    assert!(!socket.exists());
    wiring
        .agent
        .spawn_member("dev-1")
        .await
        .expect("the team still runs without a socket");
    assert_eq!(
        wiring
            .agent
            .backend()
            .member("dev-1")
            .expect("the spawned row")
            .status(),
        MemberStatus::Ready
    );
    wiring.agent.stop().await.expect("stop");
}
#[tokio::test]
async fn a_required_socket_fails_the_start() {
    let (dir, _socket) = socket_in_tempdir();
    let blocked = dir.path().join("blocked");
    std::fs::write(&blocked, b"a regular file, not a directory").expect("the blocker file");
    let socket = blocked.join("s");
    let spec = with_metadata(
        blueprint("alpha", "boss", &["dev-1"]),
        "attach",
        json!({"socket": socket, "required": true}),
    );
    let wiring = wire(&spec, |deps| deps).await;
    let error = wiring
        .agent
        .start()
        .await
        .expect_err("a required socket must fail the start");
    assert_eq!(error.kind(), ErrorKind::SysOp, "{error}");
    assert!(
        error.to_string().contains(&blocked.display().to_string()),
        "the message names the unusable parent: {error}"
    );
    assert!(!wiring.agent.is_started(), "start rolled its flag back");
    assert_eq!(wiring.agent.attach_socket(), None);
    assert!(!socket.exists(), "the failed bind left nothing behind");
}
#[test]
fn metadata_shapes_parse_without_binding() {
    let _guard = env_guard();
    std::env::remove_var(ATTACH_SOCKET_ENV);
    let bare = with_metadata(
        blueprint("alpha", "boss", &["dev-1"]),
        "attach",
        json!("/tmp/bare.sock"),
    );
    let config = attach_config(&bare).expect("a bare string names the socket");
    assert_eq!(config.socket, PathBuf::from("/tmp/bare.sock"));
    assert!(!config.required, "best effort by default");
    let with_socket = with_metadata(
        blueprint("alpha", "boss", &["dev-1"]),
        "attach",
        json!({"socket": "/tmp/a.sock", "required": true}),
    );
    let config = attach_config(&with_socket).expect("the object form names the socket");
    assert_eq!(config.socket, PathBuf::from("/tmp/a.sock"));
    assert!(config.required);
    let with_path = with_metadata(
        blueprint("alpha", "boss", &["dev-1"]),
        "attach",
        json!({"path": "/tmp/b.sock"}),
    );
    assert_eq!(
        attach_config(&with_path)
            .expect("`path` is an alias")
            .socket,
        PathBuf::from("/tmp/b.sock")
    );
    let required_only = with_metadata(
        blueprint("alpha", "boss", &["dev-1"]),
        "attach",
        json!({"required": true}),
    );
    assert!(
        attach_config(&required_only).is_none(),
        "a flag alone names no socket"
    );
    for blank in [json!(""), json!("   "), json!({"socket": "  "})] {
        let spec = with_metadata(blueprint("alpha", "boss", &["dev-1"]), "attach", blank);
        assert!(
            attach_config(&spec).is_none(),
            "a blank socket counts as unset"
        );
    }
    let plain = blueprint("alpha", "boss", &["dev-1"]);
    assert!(
        attach_config(&plain).is_none(),
        "no metadata, no environment"
    );
}
#[test]
fn the_environment_is_a_fallback_and_metadata_wins() {
    let _guard = env_guard();
    let previous = std::env::var(ATTACH_SOCKET_ENV).ok();
    let (dir, blocked) = socket_in_tempdir();
    // The parent is a regular file, so even a team that picks the variable up cannot bind it.
    std::fs::write(&blocked, b"keeps the fallback unbindable").expect("the blocker file");
    let env_socket = blocked.join("s");
    std::env::set_var(ATTACH_SOCKET_ENV, &env_socket);
    let plain = blueprint("alpha", "boss", &["dev-1"]);
    let config = attach_config(&plain).expect("the environment supplies a socket");
    assert_eq!(config.socket, env_socket);
    assert!(!config.required, "the environment never sets `required`");
    let spec = with_metadata(
        blueprint("alpha", "boss", &["dev-1"]),
        "attach",
        json!({"socket": dir.path().join("meta.sock")}),
    );
    let config = attach_config(&spec).expect("metadata names a socket");
    assert_eq!(
        config.socket,
        dir.path().join("meta.sock"),
        "metadata wins over the environment"
    );
    match previous {
        Some(value) => std::env::set_var(ATTACH_SOCKET_ENV, value),
        None => std::env::remove_var(ATTACH_SOCKET_ENV),
    }
}
#[tokio::test]
async fn a_parsed_config_serves_a_built_team() {
    let (_dir, socket) = socket_in_tempdir();
    let spec = attached_team(&socket);
    let config: AttachConfig = attach_config(&spec).expect("metadata names a socket");
    let wiring = wire(&spec, |deps| deps).await;
    let handler: Arc<dyn AttachHandler> = Arc::new(BackendAttachHandler::new(
        wiring.agent.backend(),
        wiring.agent.leader_member_name(),
        wiring.agent.cancellation_token(),
    ));
    let host = AttachHost::start(&config, handler)
        .await
        .expect("the host did not bind");
    assert_eq!(host.path(), config.socket.as_path());
    let mut client = connect(host.path()).await;
    let members = client.list_members().await.expect("list_members");
    assert!(members.iter().any(|row| row.member_name == "boss"));
    assert!(socket.exists(), "the parsed path is the bound one");
    host.shutdown().await;
    assert!(!socket.exists(), "shutdown removes the socket file");
}
#[tokio::test]
async fn no_configuration_means_no_socket() {
    {
        let _guard = env_guard();
        std::env::remove_var(ATTACH_SOCKET_ENV);
        assert!(
            attach_config(&blueprint("alpha", "boss", &["dev-1"])).is_none(),
            "neither metadata nor the environment names a socket"
        );
    }
    // No `start` here: the variable is process-global and a team-level start would race the case
    // above. A build never binds, so its handle has no socket to report.
    let wiring = wire(&blueprint("alpha", "boss", &["dev-1"]), |deps| deps).await;
    assert!(
        wiring.agent.attach_socket().is_none(),
        "a build binds nothing"
    );
}
