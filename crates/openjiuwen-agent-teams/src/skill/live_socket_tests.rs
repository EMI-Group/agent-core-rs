#![cfg(unix)]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::pedantic)]
//! Tests for the socket half of live-team resolution: `connect_with_registry` and
//! `resolve_socket_client`.
//!
//! The resolution order the CLI and the MCP server depend on is pinned here: a team this process
//! hosts wins, a published attach socket is the fallback, and anything else is `None` (which keeps
//! the CLI's "no live team in this process" message and exit code `1`).
//!
//! Each case drives the real Unix-domain attach bridge: a temporary directory holds the socket
//! (`<tmp>/s`) and a tiny hand-written `AttachHandler` answers `read_inbox` with a distinctive
//! text, so a socket client can be told apart from the store client that answers `(inbox empty)`.
//! The split halves of the resolver are called with an explicit manager and an injected environment
//! lookup, so **no** case touches the process-global runtime registry or the process environment.
//! Everything is offline and bounded: no network, no `#[ignore]`, and every server is shut down and
//! aborted by `Harness`'s drop.
use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{json, Value};

use crate::attach::{AttachHandler, AttachOp, AttachResult, AttachServer};
use crate::runtime::team_agent::test_support::{
    blueprint, deps, RecordingWorkspaceProvider, ScriptedFactory,
};
use crate::runtime::TeamRuntimeManager;
use crate::schema::{ExecutionStatus, MemberSnapshot, MemberStatus, TeamRole};
use crate::storage::{InMemoryTeamStore, TeamStore};

use super::descriptor::{TeamJoinDescriptor, TEAM_SOCKET_ENV};
use super::live::{connect_with_registry, resolve_socket_client};

/// Inbox text the default socket host renders, so a socket client is recognisable.
const INBOX: &str = "[1] socket -> user : over the socket";

/// Inbox text of the first host in the precedence cases.
const INBOX_A: &str = "[1] socket-a -> user : from A";

/// Inbox text of the second host in the precedence cases.
const INBOX_B: &str = "[2] socket-b -> user : from B";

// --- fixtures --------------------------------------------------------------

/// The roster a socket host publishes: one row nothing else ever produces.
fn remote_row() -> MemberSnapshot {
    MemberSnapshot {
        member_name: "remote".to_owned(),
        display_name: "Remote".to_owned(),
        role: TeamRole::Teammate,
        status: MemberStatus::Busy,
        execution: ExecutionStatus::Idle,
    }
}

/// Handler that answers `read_inbox` with its own text and `list_members` with [`remote_row`].
struct SocketHost {
    /// Text `read_inbox` answers with.
    inbox: &'static str,
}

#[async_trait]
impl AttachHandler for SocketHost {
    async fn handle(&self, op: AttachOp) -> AttachResult<Value> {
        Ok(match op {
            AttachOp::ReadInbox => json!(self.inbox),
            AttachOp::ListMembers => serde_json::to_value(vec![remote_row()]).unwrap(),
            _ => Value::Null,
        })
    }
}

// --- harness ---------------------------------------------------------------

/// A bound attach server serving one [`SocketHost`], shut down on drop.
struct Harness {
    /// Keeps the socket directory alive for the lifetime of the server.
    _dir: tempfile::TempDir,
    /// The bound server.
    server: Arc<AttachServer>,
    /// The serving task, aborted on drop.
    handle: tokio::task::JoinHandle<openjiuwen_common::error::Result<()>>,
}

impl Harness {
    /// Binds `<tmp>/s` and serves it with a host answering `inbox`.
    async fn start(inbox: &'static str) -> Self {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s");
        let server = Arc::new(AttachServer::bind(&path).await.unwrap());
        let serving = Arc::clone(&server);
        let handler: Arc<dyn AttachHandler> = Arc::new(SocketHost { inbox });
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

    /// The socket path as a descriptor-ready string.
    fn path_string(&self) -> String {
        self.path().display().to_string()
    }
}

impl Drop for Harness {
    fn drop(&mut self) {
        self.server.shutdown();
        self.handle.abort();
    }
}

/// A descriptor for `team_name` acting as the operator `user`, optionally pinning `socket_path`.
fn descriptor(team_name: &str, socket_path: Option<&str>) -> TeamJoinDescriptor {
    TeamJoinDescriptor {
        team_name: team_name.to_owned(),
        member_name: "user".to_owned(),
        socket_path: socket_path.map(str::to_owned),
        ..TeamJoinDescriptor::default()
    }
}

// --- socket fallback -------------------------------------------------------

#[tokio::test]
async fn socket_fallback_resolves_when_no_team_is_hosted() {
    let harness = Harness::start(INBOX).await;
    let descriptor = descriptor("skill-live-socket", Some(&harness.path_string()));

    let client =
        connect_with_registry(None, &descriptor, |_| None).expect("the socket fallback resolves");
    assert_eq!(client.member_name(), "user");
    assert_eq!(
        client.read_inbox().await.expect("the socket answers"),
        INBOX
    );
}

#[tokio::test]
async fn descriptor_path_wins_over_the_environment() {
    let first = Harness::start(INBOX_A).await;
    let second = Harness::start(INBOX_B).await;
    let descriptor = descriptor("skill-live-socket", Some(&first.path_string()));
    let from_environment = second.path_string();

    let client = connect_with_registry(None, &descriptor, |key| {
        (key == TEAM_SOCKET_ENV).then(|| from_environment.clone())
    })
    .expect("the descriptor's own socket resolves");
    assert_eq!(
        client.read_inbox().await.expect("the socket answers"),
        INBOX_A,
        "the descriptor's socket_path is consulted before the environment"
    );
}

#[tokio::test]
async fn environment_path_is_the_fallback() {
    let harness = Harness::start(INBOX).await;
    let published = harness.path_string();
    let descriptor = descriptor("skill-live-env", None);

    let client = connect_with_registry(None, &descriptor, |key| {
        (key == TEAM_SOCKET_ENV).then(|| published.clone())
    })
    .expect("the environment's socket resolves");
    assert_eq!(
        client.read_inbox().await.expect("the socket answers"),
        INBOX
    );

    assert!(
        connect_with_registry(None, &descriptor, |_| Some("   ".to_owned())).is_none(),
        "a blank environment value publishes no socket"
    );
    assert!(
        connect_with_registry(None, &descriptor, |_| None).is_none(),
        "an unset environment publishes no socket"
    );
}

#[tokio::test]
async fn nothing_configured_or_not_a_socket_resolves_to_none() {
    let bare = descriptor("skill-live-none", None);
    assert!(
        resolve_socket_client(&bare, |_| None).is_none(),
        "no descriptor path and no environment path means no socket"
    );

    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("not-a-socket");
    std::fs::write(&file, b"precious").unwrap();
    let regular = descriptor("skill-live-none", Some(&file.display().to_string()));
    assert!(
        resolve_socket_client(&regular, |_| None).is_none(),
        "a regular file is never attached to"
    );
    assert_eq!(std::fs::read(&file).unwrap(), b"precious");

    let absent = dir.path().join("missing.sock");
    let missing = descriptor("skill-live-none", Some(&absent.display().to_string()));
    assert!(
        resolve_socket_client(&missing, |_| None).is_none(),
        "a path that does not exist publishes no socket"
    );
}

// --- resolution order ------------------------------------------------------

#[tokio::test]
async fn the_in_process_registry_is_preferred_over_the_socket() {
    // The socket answers a distinctive text and a single `remote` row, so whichever client wins is
    // unmistakable.
    let harness = Harness::start(INBOX).await;

    let factory = ScriptedFactory::new();
    let workspace = RecordingWorkspaceProvider::new();
    let store: Arc<InMemoryTeamStore> = Arc::new(InMemoryTeamStore::new());
    let build_deps =
        deps(&factory, &workspace).with_store(Arc::clone(&store) as Arc<dyn TeamStore>);
    let manager = TeamRuntimeManager::new(build_deps);
    manager
        .activate(
            &blueprint("skill-live-socket", "boss", &["dev-1"]),
            "session-live",
            &Value::Null,
        )
        .await
        .expect("the team activates over the scripted fakes");

    let descriptor = descriptor("skill-live-socket", Some(&harness.path_string()));
    let client = connect_with_registry(Some(&manager), &descriptor, |_| None)
        .expect("the in-process team resolves");

    let roster = client.list_members().await.expect("the roster is readable");
    let rows: Vec<(&str, TeamRole)> = roster
        .iter()
        .map(|member| (member.member_name.as_str(), member.role))
        .collect();
    assert_eq!(
        rows,
        vec![("boss", TeamRole::Leader), ("dev-1", TeamRole::Teammate)],
        "the live team's own roster, not the socket's single `remote` row"
    );
    assert_eq!(
        client.read_inbox().await.expect("the inbox is readable"),
        "(inbox empty)",
        "the store client answered, so the socket never served this call"
    );
}

#[tokio::test]
async fn unhosted_team_without_a_socket_stays_none() {
    let manager = TeamRuntimeManager::with_defaults();
    let descriptor = descriptor("skill-live-unhosted", None);

    assert!(
        connect_with_registry(Some(&manager), &descriptor, |_| None).is_none(),
        "a team this process does not host and no published socket resolves to None, \
         which keeps the CLI's exit code 1"
    );
}
