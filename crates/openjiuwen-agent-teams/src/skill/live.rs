//! Live-team resolution: the bridge from a join descriptor to the team a client drives.
//!
//! This is the single wiring point for **both** the `team-member` skill CLI and the
//! `openjiuwen-team-mcp` server: each of them holds a [`TeamJoinDescriptor`] naming a team and asks
//! [`connect_live_client`] for a client over that team.
//!
//! Two hosts can answer, in this order:
//!
//! 1. **This process.** The team's [`TeamBackend`](crate::runtime::TeamBackend) is handed out by the
//!    process-global runtime registry ([`crate::runtime::try_global_registry`]), and the client is a
//!    [`StoreTeamClient`] over that backend's own store, task board and messager. This is the
//!    cheapest and most direct route, so it always wins.
//! 2. **Another process.** A hosting process publishes its attach socket (the `socket_path` field of
//!    the descriptor, else the `OPENJIUWEN_TEAM_SOCKET` variable), and the client is a lazily
//!    connecting [`SocketTeamClient`] over the attach bridge.
//!
//! # Why the socket client is lazy here
//!
//! [`connect_live_client`] is synchronous and may be called from inside a runtime (the MCP resolver
//! runs on the server's task), so it must not block or await: it only probes the filesystem
//! (`socket_present`, a cheap `stat`) and hands back a client that connects on its first
//! operation. A socket that is unreachable therefore surfaces at the first operation as a typed
//! [`ErrorKind::Transport`](openjiuwen_common::error::ErrorKind::Transport) or
//! [`ErrorKind::Timeout`](openjiuwen_common::error::ErrorKind::Timeout) error, which the CLI maps
//! to exit code `1`.
//!
//! A descriptor that names no socket, or whose socket is not an existing Unix socket file, keeps
//! the in-process rule unchanged: resolution is `None`, the CLI reports
//! `no live team '<name>' in this process` and exits `1`. A host that needs the connect failure up
//! front uses [`connect_socket_client`] instead.

use std::path::Path;
use std::sync::Arc;

use crate::runtime::TeamRuntimeManager;

use super::client::TeamClient;
use super::descriptor::TeamJoinDescriptor;
use super::socket_client::SocketTeamClient;
use super::store_client::StoreTeamClient;

/// Returns a client over the team `descriptor` names, or `None` when no host can be found.
///
/// Resolution order: a team this process hosts (the global registry →
/// [`TeamRuntimeManager::live`] → the live [`TeamAgent`](crate::runtime::TeamAgent)'s backend → a
/// [`StoreTeamClient`]), else a [`SocketTeamClient`] over the socket the descriptor resolves from
/// its `socket_path` field or the `OPENJIUWEN_TEAM_SOCKET` variable.
///
/// The socket route does no I/O beyond a `stat`, so this function never blocks and is safe to call
/// from inside a runtime; the connection is established by the first operation.
///
/// `None` means "no such team here and no published socket": either nothing was installed behind
/// the registry, the installed manager hosts no team under that name, and no socket path resolves
/// (or the path is not an existing Unix socket file).
#[must_use]
pub fn connect_live_client(descriptor: &TeamJoinDescriptor) -> Option<Arc<dyn TeamClient>> {
    connect_with_registry(
        crate::runtime::try_global_registry().as_deref(),
        descriptor,
        |key| std::env::var(key).ok(),
    )
}

/// Resolves a client against an explicit registry slot and environment lookup.
///
/// Split from [`connect_live_client`] so both halves are unit-testable without touching the
/// process-global registry or the process environment: `manager` is the registry's manager (or
/// `None` when nothing is installed) and `get` is the environment lookup the socket fallback uses.
#[must_use]
pub(crate) fn connect_with_registry(
    manager: Option<&TeamRuntimeManager>,
    descriptor: &TeamJoinDescriptor,
    get: impl Fn(&str) -> Option<String>,
) -> Option<Arc<dyn TeamClient>> {
    if let Some(client) = manager.and_then(|manager| resolve_live_client(manager, descriptor)) {
        return Some(client);
    }
    resolve_socket_client(descriptor, get)
}

/// Builds the [`StoreTeamClient`] for `descriptor` out of the live team `manager` hosts, if any.
///
/// Split from [`connect_live_client`] so the resolution is unit-testable against a local manager
/// without touching the process-global registry.
#[must_use]
pub(crate) fn resolve_live_client(
    manager: &TeamRuntimeManager,
    descriptor: &TeamJoinDescriptor,
) -> Option<Arc<dyn TeamClient>> {
    let agent = manager.live(&descriptor.team_name)?;
    let backend = agent.backend();
    let client = StoreTeamClient::new(
        descriptor.clone(),
        backend.store(),
        Arc::clone(backend.tasks()),
        Arc::clone(backend.messager()),
    );
    Some(Arc::new(client))
}

/// Builds a lazily connecting [`SocketTeamClient`] for the socket `descriptor` resolves, if any.
///
/// The socket is taken from the descriptor's `socket_path` field first and from the `get` lookup
/// ([`TEAM_SOCKET_ENV`](super::descriptor::TEAM_SOCKET_ENV)) second. Nothing is connected here: the
/// path must exist *and* be a Unix socket file ([`socket_present`]), which keeps a stale or mistyped
/// path from turning into a phantom client, and the bounded connect happens on the first operation.
#[must_use]
pub(crate) fn resolve_socket_client(
    descriptor: &TeamJoinDescriptor,
    get: impl Fn(&str) -> Option<String>,
) -> Option<Arc<dyn TeamClient>> {
    let path = descriptor.resolve_socket_path_with(get)?;
    if !socket_present(&path) {
        tracing::debug!(
            path = %path.display(),
            team = %descriptor.team_name,
            "socket path is not an attach socket; not attaching"
        );
        return None;
    }
    Some(Arc::new(SocketTeamClient::new(descriptor.clone(), path)))
}

/// Connects to the attach socket at `path` and returns a client over the hosted team.
///
/// This is the eager counterpart of the socket fallback in [`connect_live_client`], for a host or a
/// test that wants the connect failure reported up front rather than on the first operation.
///
/// # Errors
///
/// Returns the attach bridge's typed error: [`ErrorKind::Transport`](openjiuwen_common::error::ErrorKind::Transport)
/// when the socket cannot be reached, [`ErrorKind::Timeout`](openjiuwen_common::error::ErrorKind::Timeout)
/// when the attach deadline elapses first, and
/// [`ErrorKind::Unsupported`](openjiuwen_common::error::ErrorKind::Unsupported) on a target without
/// Unix domain sockets.
pub async fn connect_socket_client(
    path: impl AsRef<Path>,
    descriptor: TeamJoinDescriptor,
) -> openjiuwen_common::error::Result<Arc<dyn TeamClient>> {
    let client = SocketTeamClient::connect(descriptor, path).await?;
    Ok(Arc::new(client))
}

/// Returns whether `path` is an existing Unix socket file.
///
/// Only a socket file can be connected to, so this is a cheap pre-check rather than a probe: a
/// directory, a regular file or a missing path all report `false` without ever opening anything.
/// Non-Unix targets have no attach transport, so the answer there is always `false`.
#[cfg(unix)]
fn socket_present(path: &Path) -> bool {
    use std::os::unix::fs::FileTypeExt;

    std::fs::metadata(path).is_ok_and(|meta| meta.file_type().is_socket())
}

/// Returns whether `path` is an existing Unix socket file; always `false` off Unix.
#[cfg(not(unix))]
fn socket_present(_path: &Path) -> bool {
    false
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use std::sync::{Arc, Mutex, MutexGuard};

    use serde_json::Value;

    use crate::runtime::manager::{
        clear_global_registry, set_global_registry, try_global_registry, TeamRuntimeManager,
    };
    use crate::runtime::team_agent::test_support::{
        blueprint, deps, RecordingWorkspaceProvider, ScriptedFactory,
    };
    use crate::schema::TeamRole;
    use crate::storage::{InMemoryTeamStore, TeamStore};

    use super::super::descriptor::ClientScope;
    use super::*;

    /// Serializes the cases that install into the process-global runtime registry.
    ///
    /// The registry is a process-global also exercised by `crate::runtime::tests`
    /// (`src/runtime/tests/end_to_end.rs`), which holds no lock of its own, so a lock here only
    /// orders *this* module's cases; [`RegistryRestore`] is what actually keeps the shared slot
    /// intact for everyone else (it puts back whatever was there before the case ran). Distinctive
    /// team names keep the remaining interleavings harmless.
    static REGISTRY_LOCK: Mutex<()> = Mutex::new(());

    /// Puts the process-global registry slot back the way the case found it, panics included.
    struct RegistryRestore {
        /// The manager installed before the case ran, `None` when the slot was empty.
        previous: Option<Arc<TeamRuntimeManager>>,
    }

    impl Drop for RegistryRestore {
        fn drop(&mut self) {
            match self.previous.take() {
                Some(manager) => set_global_registry(manager),
                None => clear_global_registry(),
            }
        }
    }

    /// Takes [`REGISTRY_LOCK`] and captures the slot to restore.
    ///
    /// The lock is poison-tolerant: a panicking case must not turn every later case into a panic.
    fn registry_guard() -> (MutexGuard<'static, ()>, RegistryRestore) {
        let lock = REGISTRY_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let previous = try_global_registry();
        (lock, RegistryRestore { previous })
    }

    #[test]
    fn unknown_team_does_not_resolve() {
        let manager = TeamRuntimeManager::with_defaults();
        let descriptor = TeamJoinDescriptor {
            team_name: "skill-live-missing".to_string(),
            ..TeamJoinDescriptor::default()
        };
        assert!(resolve_live_client(&manager, &descriptor).is_none());
    }

    #[test]
    fn global_registry_resolution_and_clear() {
        let (_lock, _restore) = registry_guard();
        set_global_registry(Arc::new(TeamRuntimeManager::with_defaults()));
        assert!(try_global_registry().is_some(), "the manager is installed");

        let descriptor = TeamJoinDescriptor {
            team_name: "skill-live-not-hosted".to_string(),
            socket_path: None,
            ..TeamJoinDescriptor::default()
        };
        assert!(
            connect_with_registry(try_global_registry().as_deref(), &descriptor, |_| None)
                .is_none(),
            "an unhosted team with no published socket does not resolve"
        );

        clear_global_registry();
        assert!(try_global_registry().is_none(), "the slot is cleared");
    }

    #[test]
    fn a_regular_file_is_not_an_attach_socket() {
        let file = tempfile::NamedTempFile::new().unwrap();
        assert!(!socket_present(file.path()));
        assert!(!socket_present(std::path::Path::new(
            "does-not-exist/attach.sock"
        )));
        let descriptor = TeamJoinDescriptor {
            socket_path: Some(file.path().display().to_string()),
            ..TeamJoinDescriptor::default()
        };
        assert!(
            resolve_socket_client(&descriptor, |_| None).is_none(),
            "only a socket file is attached to"
        );
    }

    // The registry lock is held for the whole case so its install and its restore cannot be
    // interleaved by another case of this module; no other holder exists, so the awaits below
    // cannot deadlock, and each `#[tokio::test]` runs its own current-thread runtime.
    #[allow(clippy::await_holding_lock)]
    #[tokio::test]
    async fn a_live_team_resolves_through_the_global_registry() {
        let (_lock, _restore) = registry_guard();
        let factory = ScriptedFactory::new();
        let workspace = RecordingWorkspaceProvider::new();
        let store: Arc<InMemoryTeamStore> = Arc::new(InMemoryTeamStore::new());
        let build_deps =
            deps(&factory, &workspace).with_store(Arc::clone(&store) as Arc<dyn TeamStore>);
        let manager = Arc::new(TeamRuntimeManager::new(build_deps));
        let team = manager
            .activate(
                &blueprint("skill-live", "boss", &["dev-1"]),
                "session-live",
                &Value::Null,
            )
            .await
            .expect("the team activates over the scripted fakes");
        set_global_registry(Arc::clone(&manager));

        let descriptor = TeamJoinDescriptor {
            team_name: "skill-live".to_string(),
            member_name: "user".to_string(),
            scope: ClientScope::Operator,
            ..TeamJoinDescriptor::default()
        };
        let client = connect_live_client(&descriptor).expect("the live team resolves");
        assert_eq!(client.scope(), ClientScope::Operator);

        let members = client.list_members().await.expect("the roster is readable");
        let rows: Vec<(&str, TeamRole)> = members
            .iter()
            .map(|member| (member.member_name.as_str(), member.role))
            .collect();
        assert_eq!(
            rows,
            vec![("boss", TeamRole::Leader), ("dev-1", TeamRole::Teammate)],
            "the live roster, sorted by name"
        );
        assert!(
            members
                .iter()
                .all(|member| member.status != crate::schema::MemberStatus::ShutDown),
            "{members:?}"
        );
        assert_eq!(
            client.read_inbox().await.expect("the inbox is readable"),
            "(inbox empty)"
        );

        let created = client
            .create_task("wire live client", "body", None, &[])
            .await
            .expect("the board accepts the task");
        assert!(created.ok, "{:?}", created.reason);
        let task_id = created.task_id.expect("the created task has an id");

        let claimable = client.claimable_tasks().await.expect("claimable tasks");
        assert!(
            claimable.iter().any(|task| task.task_id == task_id),
            "{claimable:?}"
        );

        // The client wrote through the live team's own board, not a copy of it.
        assert!(Arc::ptr_eq(&manager.live("skill-live").unwrap(), &team));
        let live_claimable = team
            .backend()
            .tasks()
            .claimable()
            .await
            .expect("the live board answers");
        assert!(
            live_claimable.iter().any(|task| task.task_id == task_id),
            "{live_claimable:?}"
        );
    }
}
