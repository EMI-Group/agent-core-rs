//! Spawn seam: [`MemberSpawner`], its request, and the in-process task handle.
//!
//! Python parity source: `openjiuwen/agent_teams/spawn/inprocess_spawn.py` and
//! `spawn/inprocess_handle.py`. Python's `inprocess_spawn` builds a teammate `TeamAgent` and
//! creates an `asyncio.Task` around `Runner.run_agent_team`; the returned `InProcessSpawnHandle`
//! wraps that task (`cancel` / `join` / `abort`). The Rust port keeps the same shape: a
//! [`MemberSpawner`] builds the member's runtime and a [`InProcessSpawnHandle`] owns the tokio task
//! that drives it.
//!
//! Only the in-process mode is wired up in this crate so far; the launch configuration for
//! OS-process members, external-CLI members and the SSH transport lives in
//! [`spawn_config`](crate::member::spawn_config) and travels on
//! [`MemberSpawnRequest::options`], but the child-process spawners built on it are still being
//! ported. A spawner asked for a mode it cannot serve fails with
//! [`openjiuwen_common::error::Error::Unsupported`] rather than pretending to work.
//!
//! Ownership rule: a [`InProcessSpawnHandle`] is the only owner of its task. Dropping a
//! `JoinHandle` detaches the task instead of aborting it, so every handle that leaves
//! [`SpawnRegistry`] must be joined or aborted by its owner — `TeamBackend` for a live team, the
//! spawner for a failed spawn. The registry never drops a live handle on the floor; see
//! [`SpawnRegistry::cancel_all`].

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tracing::debug;

use super::runtime::MemberRuntime;
use super::spawn_config::{MemberSpawnOptions, StdioCommand};
use crate::runtime::checkpoints::CHECKPOINT_DIR;
use crate::runtime::spec_meta::{CheckpointConfig, CheckpointStoreConfig};

/// Environment variable carrying the owning team name to a spawned member process.
pub const TEAM_NAME_ENV: &str = "OPENJIUWEN_TEAM_NAME";
/// Environment variable carrying the member's own name to a spawned member process.
pub const MEMBER_NAME_ENV: &str = "OPENJIUWEN_MEMBER_NAME";
/// Environment variable carrying the session id to a spawned member process.
pub const SESSION_ID_ENV: &str = "OPENJIUWEN_SESSION_ID";
/// Environment variable carrying the member's workspace root to a spawned member process.
pub const MEMBER_WORKSPACE_ENV: &str = "OPENJIUWEN_MEMBER_WORKSPACE";
/// Environment variable telling a spawned member process whether to capture checkpoints.
///
/// [`CHECKPOINT_OFF`] disables every capture and resume attempt; any other value — and an absent
/// variable — leaves checkpointing enabled, so a leader that exports nothing keeps the served
/// member on its own enabled default.
pub const CHECKPOINT_ENV: &str = "OPENJIUWEN_MEMBER_CHECKPOINTS";
/// Environment variable naming the directory a spawned member writes file-backed checkpoints into.
///
/// The leader exports the absolute directory it resolved for the member, so a file-backed store
/// with no pinned directory reaches the child as `<workspace root>/.openjiuwen/checkpoints`. An
/// absent or blank variable leaves the served member on its process-local in-memory store.
pub const CHECKPOINT_DIR_ENV: &str = "OPENJIUWEN_MEMBER_CHECKPOINT_DIR";
/// Value of [`CHECKPOINT_ENV`] that disables checkpointing in a spawned member.
pub const CHECKPOINT_OFF: &str = "off";
/// Value of [`CHECKPOINT_ENV`] that keeps checkpointing enabled in a spawned member.
pub const CHECKPOINT_ON: &str = "on";

/// Everything a spawner needs to bring one member up.
///
/// Python passes a `TeamRuntimeContext` plus the owning `TeamAgent`; Rust keeps the request to the
/// facts a spawner cannot derive: which team and member, the session id to propagate, where
/// the member's workspace is rooted, and how the member should be launched
/// ([`MemberSpawnOptions`], in-process by default).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct MemberSpawnRequest {
    /// Team the member belongs to.
    pub team_name: String,
    /// Member to spawn.
    pub member_name: String,
    /// Session id propagated into the spawned task.
    pub session_id: String,
    /// Root of the member's workspace.
    pub workspace_root: PathBuf,
    /// Spawn mode plus the per-kind launch configuration.
    ///
    /// Defaults to in-process, so a payload without `options` spawns the member the way this
    /// crate has always done it.
    pub options: MemberSpawnOptions,
    /// Checkpoint settings the leader resolved for this member.
    ///
    /// A child-process transport exports them as [`CHECKPOINT_ENV`]/[`CHECKPOINT_DIR_ENV`] (see
    /// [`Self::checkpoint_env`]); `None` — the default — exports nothing, so a served member keeps
    /// its own enabled default. The field stays out of the JSON form: it travels to a spawner
    /// in-process and never crosses a wire.
    #[serde(skip)]
    pub checkpoint: Option<CheckpointConfig>,
}

impl MemberSpawnRequest {
    /// Builds an in-process request.
    #[must_use]
    pub fn new(
        team_name: impl Into<String>,
        member_name: impl Into<String>,
        session_id: impl Into<String>,
        workspace_root: impl Into<PathBuf>,
    ) -> Self {
        Self {
            team_name: team_name.into(),
            member_name: member_name.into(),
            session_id: session_id.into(),
            workspace_root: workspace_root.into(),
            options: MemberSpawnOptions::default(),
            checkpoint: None,
        }
    }

    /// Replaces the spawn options, selecting another mode or a launch configuration.
    #[must_use]
    pub fn with_options(mut self, options: MemberSpawnOptions) -> Self {
        self.options = options;
        self
    }

    /// Attaches the checkpoint settings the leader resolved for this member.
    ///
    /// The build path reads them from `metadata["checkpoint"]`; a request that never sets them
    /// leaves [`Self::checkpoint`] `None` and therefore exports nothing.
    #[must_use]
    pub fn with_checkpoint_config(mut self, checkpoint: CheckpointConfig) -> Self {
        self.checkpoint = Some(checkpoint);
        self
    }

    /// Returns the identity environment a child-process spawner exports to the member.
    ///
    /// A spawned member process cannot inherit the request, and the JSON-lines member protocol
    /// carries only the session id in its `start` frame, so the team name, member name, session id
    /// and workspace root cross the process boundary as environment variables, together with the
    /// checkpoint settings of [`Self::checkpoint_env`]. Empty values are omitted so an unset field
    /// never blanks a variable the child already has.
    #[must_use]
    pub fn identity_env(&self) -> BTreeMap<String, String> {
        let workspace = self.workspace_root.to_string_lossy().into_owned();
        let mut env: BTreeMap<String, String> = [
            (TEAM_NAME_ENV, self.team_name.as_str()),
            (MEMBER_NAME_ENV, self.member_name.as_str()),
            (SESSION_ID_ENV, self.session_id.as_str()),
            (MEMBER_WORKSPACE_ENV, workspace.as_str()),
        ]
        .into_iter()
        .filter(|(_, value)| !value.is_empty())
        .map(|(key, value)| (key.to_string(), value.to_string()))
        .collect();
        env.extend(self.checkpoint_env());
        env
    }

    /// Returns the checkpoint environment a child-process spawner exports to the member.
    ///
    /// The served member cannot read the leader's configuration any other way, so two variables
    /// carry it: [`CHECKPOINT_ENV`] ([`CHECKPOINT_OFF`] for a disabled policy, [`CHECKPOINT_ON`]
    /// otherwise) and — for a file-backed store — [`CHECKPOINT_DIR_ENV`] with the absolute
    /// directory to write into, an unpinned file root resolving to
    /// `<workspace root>/.openjiuwen/checkpoints`.
    ///
    /// An absent [`checkpoint`](Self::checkpoint) configuration exports nothing at all, so the
    /// served member keeps its own enabled default on a process-local in-memory store.
    #[must_use]
    pub fn checkpoint_env(&self) -> BTreeMap<String, String> {
        let Some(config) = &self.checkpoint else {
            return BTreeMap::new();
        };
        let enabled = config.policy.is_enabled();
        let mut env = BTreeMap::new();
        env.insert(
            CHECKPOINT_ENV.to_owned(),
            if enabled { CHECKPOINT_ON } else { CHECKPOINT_OFF }.to_owned(),
        );
        if let Some(dir) = checkpoint_dir(config, &self.workspace_root) {
            env.insert(
                CHECKPOINT_DIR_ENV.to_owned(),
                dir.to_string_lossy().into_owned(),
            );
        }
        env
    }

    /// Applies [`Self::identity_env`] to `command` without overriding an explicitly configured value.
    ///
    /// A spec that pre-sets one of these variables is an intentional override and wins over the
    /// request-derived value; every other variable is inherited from the parent environment.
    pub fn apply_identity_env(&self, command: &mut StdioCommand) {
        for (key, value) in self.identity_env() {
            command.env.entry(key).or_insert(value);
        }
    }
}

/// Creates and tears down member runtimes.
///
/// The in-process implementation lives in the harness layer (`NativeMemberFactory` +
/// `TeamBackend`), which is also the owner of the spawned tasks. A spawner must not start the
/// runtime: [`MemberRuntime::start`] is called by the member's lifecycle so a failed start can be
/// rolled back in the roster.
#[async_trait]
pub trait MemberSpawner: Send + Sync {
    /// Builds a runtime for `request`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::unsupported`] for a spawn mode this crate does not implement, and
    /// propagates build failures (missing model, bad workspace, missing spec).
    async fn spawn(&self, request: MemberSpawnRequest) -> Result<Arc<dyn MemberRuntime>>;

    /// Tears down the runtime spawned for `member_name`.
    ///
    /// # Errors
    ///
    /// Propagates teardown failures; an unknown member is not an error (the spawn may never have
    /// completed).
    async fn despawn(&self, member_name: &str) -> Result<()>;
}

/// An in-process member task: its member name, cancellation token and tokio task.
///
/// `cancel` signals cooperatively (the task observes the token and unwinds); `abort` drops the
/// task immediately. Either way the task must still be joined — [`Self::join`] turns the join
/// error into [`Error::cancelled`] or a wrapped external error, so a cancelled member is not
/// mistaken for a crash.
#[derive(Debug)]
pub struct InProcessSpawnHandle {
    member_name: String,
    cancel: CancellationToken,
    task: JoinHandle<Result<()>>,
}

impl InProcessSpawnHandle {
    /// Wraps `task` with a fresh cancellation token.
    ///
    /// Use [`Self::with_cancellation`] when the spawned task also needs the token — the common
    /// case, since the task must observe it to unwind cooperatively.
    #[must_use]
    pub fn new(member_name: impl Into<String>, task: JoinHandle<Result<()>>) -> Self {
        Self {
            member_name: member_name.into(),
            cancel: CancellationToken::new(),
            task,
        }
    }

    /// Wraps `task` with an existing cancellation token.
    #[must_use]
    pub fn with_cancellation(
        member_name: impl Into<String>,
        cancel: CancellationToken,
        task: JoinHandle<Result<()>>,
    ) -> Self {
        Self {
            member_name: member_name.into(),
            cancel,
            task,
        }
    }

    /// Returns the member this task belongs to.
    #[must_use]
    pub fn member_name(&self) -> &str {
        &self.member_name
    }

    /// Returns a clone of the cancellation token (for the spawned task or a sibling).
    #[must_use]
    pub fn cancellation_token(&self) -> CancellationToken {
        self.cancel.clone()
    }

    /// Returns `true` once the task has completed (or been aborted).
    #[must_use]
    pub fn is_finished(&self) -> bool {
        self.task.is_finished()
    }

    /// Requests cooperative cancellation. The task keeps running until it observes the token.
    pub fn cancel(&self) {
        debug!(member = %self.member_name, "cancelling in-process member task");
        self.cancel.cancel();
    }

    /// Awaits the task's result.
    ///
    /// # Errors
    ///
    /// Returns the task's own error, [`Error::cancelled`] when the task was aborted, or a wrapped
    /// external error when the task panicked.
    pub async fn join(self) -> Result<()> {
        match self.task.await {
            Ok(result) => result,
            Err(error) if error.is_cancelled() => Err(Error::cancelled()),
            Err(error) => Err(Error::external(error)),
        }
    }

    /// Hard-aborts the task and cancels its token.
    pub fn abort(&self) {
        debug!(member = %self.member_name, "aborting in-process member task");
        self.cancel.cancel();
        self.task.abort();
    }
}

/// The live in-process tasks of one team, keyed by member name.
///
/// The registry only holds handles; it never joins them implicitly. Callers remove a handle
/// (`take` / `remove` / `cancel_all`) and join it on their own task, which keeps teardown order
/// under `TeamBackend`'s control. Inserting a second handle for the same member returns the first
/// one so it cannot be leaked by accident.
#[derive(Debug, Default)]
pub struct SpawnRegistry {
    handles: Mutex<HashMap<String, InProcessSpawnHandle>>,
}

impl SpawnRegistry {
    /// Creates an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Inserts `handle`, returning any handle already registered for that member.
    ///
    /// The caller owns the returned handle and must abort or join it.
    pub fn insert(&self, handle: InProcessSpawnHandle) -> Option<InProcessSpawnHandle> {
        self.handles
            .lock()
            .insert(handle.member_name.clone(), handle)
    }

    /// Returns `true` when a handle is registered for `member_name`.
    #[must_use]
    pub fn contains(&self, member_name: &str) -> bool {
        self.handles.lock().contains_key(member_name)
    }

    /// Removes the handle for `member_name` without signalling it.
    ///
    /// Use this for a task that already finished (or that the caller is about to join as-is).
    pub fn remove(&self, member_name: &str) -> Option<InProcessSpawnHandle> {
        self.handles.lock().remove(member_name)
    }

    /// Cancels and removes the handle for `member_name`.
    ///
    /// The returned handle still has to be joined; the token only asks the task to unwind.
    pub fn take(&self, member_name: &str) -> Option<InProcessSpawnHandle> {
        let handle = self.handles.lock().remove(member_name)?;
        handle.cancel();
        Some(handle)
    }

    /// Cancels and removes every handle, returning them for joining.
    ///
    /// This is the teardown path: no live task is left detached inside the registry.
    pub fn cancel_all(&self) -> Vec<InProcessSpawnHandle> {
        let handles: Vec<InProcessSpawnHandle> = self
            .handles
            .lock()
            .drain()
            .map(|(_, handle)| handle)
            .collect();
        for handle in &handles {
            handle.cancel();
        }
        handles
    }

    /// Returns the number of live handles.
    #[must_use]
    pub fn len(&self) -> usize {
        self.handles.lock().len()
    }

    /// Returns `true` when no handle is registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.handles.lock().is_empty()
    }

    /// Returns the registered member names, sorted.
    #[must_use]
    pub fn names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.handles.lock().keys().cloned().collect();
        names.sort();
        names
    }
}

/// Resolves the directory a child-process member writes file-backed checkpoints into.
///
/// `None` when the policy is disabled or the store stays in memory, the pinned directory when the
/// configuration names one, and `<workspace root>/.openjiuwen/checkpoints` for the file backend
/// without a pinned root — the same rule the in-process member build applies.
fn checkpoint_dir(config: &CheckpointConfig, workspace_root: &Path) -> Option<PathBuf> {
    if !config.policy.is_enabled() {
        return None;
    }
    match &config.store {
        CheckpointStoreConfig::Memory => None,
        CheckpointStoreConfig::File { dir: Some(dir) } => Some(dir.clone()),
        CheckpointStoreConfig::File { dir: None } => Some(workspace_root.join(CHECKPOINT_DIR)),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::time::Duration;

    use tokio::sync::oneshot;

    use super::*;
    use crate::harness::MemberCheckpointPolicy;
    use crate::member::spawn_config::{MemberSpawnKind, MemberSpawnOptions, SshTransportSpec};

    fn finished_handle(member_name: &str) -> InProcessSpawnHandle {
        InProcessSpawnHandle::new(member_name, tokio::spawn(async { Ok(()) }))
    }

    /// Spawns a task that signals readiness and then waits for the token.
    fn gated_handle(member_name: &str) -> (InProcessSpawnHandle, oneshot::Receiver<()>) {
        let cancel = CancellationToken::new();
        let (ready_tx, ready_rx) = oneshot::channel();
        let task_token = cancel.clone();
        let task = tokio::spawn(async move {
            let _ = ready_tx.send(());
            task_token.cancelled().await;
            Ok(())
        });
        (
            InProcessSpawnHandle::with_cancellation(member_name, cancel, task),
            ready_rx,
        )
    }

    #[tokio::test]
    async fn cancel_then_join_returns_the_tasks_result() {
        let (handle, ready) = gated_handle("dev-1");
        ready.await.unwrap();
        assert_eq!(handle.member_name(), "dev-1");
        assert!(!handle.is_finished());

        handle.cancel();
        handle.join().await.unwrap();
    }

    #[tokio::test]
    async fn aborting_a_task_reports_cancellation() {
        let (handle, ready) = gated_handle("dev-1");
        ready.await.unwrap();

        handle.abort();
        let error = handle.join().await.unwrap_err();
        assert!(error.is_cancelled());
    }

    #[tokio::test]
    async fn a_task_error_is_propagated_unchanged() {
        let handle = InProcessSpawnHandle::new(
            "dev-1",
            tokio::spawn(async { Err(Error::validation("round blew up")) }),
        );
        let error = handle.join().await.unwrap_err();
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Validation
        );
        assert!(error.message().contains("round blew up"));
    }

    #[tokio::test]
    async fn a_panicking_task_becomes_an_external_error() {
        let handle = InProcessSpawnHandle::new("dev-1", tokio::spawn(async { panic!("boom") }));
        let error = handle.join().await.unwrap_err();
        assert_eq!(error.kind(), openjiuwen_common::error::ErrorKind::External);
    }

    #[tokio::test]
    async fn is_finished_observes_a_completed_task() {
        let handle = finished_handle("dev-1");
        let mut finished = handle.is_finished();
        for _ in 0..100 {
            if finished {
                break;
            }
            tokio::task::yield_now().await;
            finished = handle.is_finished();
        }
        assert!(finished);
        handle.join().await.unwrap();
    }

    #[tokio::test]
    async fn registry_insert_returns_the_replaced_handle() {
        let registry = SpawnRegistry::new();
        assert!(registry.is_empty());
        assert!(registry.insert(finished_handle("dev-1")).is_none());
        assert!(registry.contains("dev-1"));
        assert_eq!(registry.len(), 1);

        let replaced = registry.insert(finished_handle("dev-1")).expect("previous");
        assert_eq!(replaced.member_name(), "dev-1");
        assert_eq!(registry.len(), 1);
        replaced.join().await.unwrap();

        let names = {
            registry.insert(finished_handle("dev-2"));
            registry.names()
        };
        assert_eq!(names, vec!["dev-1".to_string(), "dev-2".to_string()]);
    }

    #[tokio::test]
    async fn registry_remove_does_not_cancel() {
        let registry = SpawnRegistry::new();
        let (handle, ready) = gated_handle("dev-1");
        ready.await.unwrap();
        registry.insert(handle);

        let removed = registry.remove("dev-1").expect("registered");
        assert!(!removed.cancellation_token().is_cancelled());
        assert!(registry.remove("dev-1").is_none());

        removed.cancel();
        removed.join().await.unwrap();
    }

    #[tokio::test]
    async fn registry_take_cancels_and_removes() {
        let registry = SpawnRegistry::new();
        let (handle, ready) = gated_handle("dev-1");
        ready.await.unwrap();
        registry.insert(handle);

        let taken = registry.take("dev-1").expect("registered");
        assert!(taken.cancellation_token().is_cancelled());
        assert!(registry.is_empty());
        assert!(registry.take("dev-1").is_none());
        taken.join().await.unwrap();
    }

    #[tokio::test]
    async fn cancel_all_returns_every_handle_cancelled_and_empties_the_registry() {
        let registry = SpawnRegistry::new();
        let (first, first_ready) = gated_handle("dev-1");
        let (second, second_ready) = gated_handle("dev-2");
        first_ready.await.unwrap();
        second_ready.await.unwrap();
        registry.insert(first);
        registry.insert(second);
        assert_eq!(registry.len(), 2);

        let handles = registry.cancel_all();

        assert_eq!(handles.len(), 2);
        assert!(registry.is_empty());
        for handle in handles {
            assert!(handle.cancellation_token().is_cancelled());
            handle.join().await.unwrap();
        }
    }

    #[test]
    fn spawn_request_round_trips_through_json() {
        let request = MemberSpawnRequest::new("alpha", "dev-1", "sess-1", "/tmp/ws");
        let encoded = serde_json::to_string(&request).unwrap();
        assert!(encoded.contains("\"workspace_root\""));
        assert!(encoded.contains("\"options\""));
        let decoded: MemberSpawnRequest = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, request);
        let default: MemberSpawnRequest = serde_json::from_str("{}").unwrap();
        assert_eq!(default, MemberSpawnRequest::default());
    }

    #[test]
    fn identity_env_carries_the_request_facts() {
        let request = MemberSpawnRequest::new("alpha", "dev-1", "sess-1", "/tmp/ws");
        let env = request.identity_env();
        assert_eq!(env.get(TEAM_NAME_ENV).map(String::as_str), Some("alpha"));
        assert_eq!(env.get(MEMBER_NAME_ENV).map(String::as_str), Some("dev-1"));
        assert_eq!(env.get(SESSION_ID_ENV).map(String::as_str), Some("sess-1"));
        assert_eq!(
            env.get(MEMBER_WORKSPACE_ENV).map(String::as_str),
            Some("/tmp/ws")
        );
    }

    #[test]
    fn identity_env_omits_empty_values_and_respects_explicit_overrides() {
        let request = MemberSpawnRequest::new("", "", "", "");
        assert!(request.identity_env().is_empty());

        let request = MemberSpawnRequest::new("alpha", "dev-1", "sess-1", "/tmp/ws");
        let mut command = StdioCommand::new("team-member")
            .with_arg("serve")
            .with_env(MEMBER_NAME_ENV, "explicit");
        request.apply_identity_env(&mut command);
        assert_eq!(
            command.env().get(MEMBER_NAME_ENV).map(String::as_str),
            Some("explicit")
        );
        assert_eq!(
            command.env().get(TEAM_NAME_ENV).map(String::as_str),
            Some("alpha")
        );
    }

    #[test]
    fn checkpoint_env_is_absent_without_a_configuration() {
        let request = MemberSpawnRequest::new("alpha", "dev-1", "sess-1", "/tmp/ws");
        assert!(request.checkpoint_env().is_empty());
        let env = request.identity_env();
        assert!(!env.contains_key(CHECKPOINT_ENV));
        assert!(!env.contains_key(CHECKPOINT_DIR_ENV));
    }

    #[test]
    fn checkpoint_env_carries_the_resolved_directory() {
        let request = MemberSpawnRequest::new("alpha", "dev-1", "sess-1", "/tmp/ws")
            .with_checkpoint_config(CheckpointConfig {
                policy: MemberCheckpointPolicy::default(),
                store: CheckpointStoreConfig::File {
                    dir: Some(PathBuf::from("/tmp/cp")),
                },
            });
        let env = request.identity_env();
        assert_eq!(
            env.get(CHECKPOINT_ENV).map(String::as_str),
            Some(CHECKPOINT_ON)
        );
        assert_eq!(
            env.get(CHECKPOINT_DIR_ENV).map(String::as_str),
            Some("/tmp/cp")
        );
    }

    #[test]
    fn an_unpinned_file_root_is_exported_under_the_workspace() {
        let request = MemberSpawnRequest::new("alpha", "dev-1", "sess-1", "/tmp/ws")
            .with_checkpoint_config(CheckpointConfig {
                policy: MemberCheckpointPolicy::default(),
                store: CheckpointStoreConfig::File { dir: None },
            });
        assert_eq!(
            request.checkpoint_env().get(CHECKPOINT_DIR_ENV),
            Some(&"/tmp/ws/.openjiuwen/checkpoints".to_owned())
        );
    }

    #[test]
    fn a_memory_store_exports_enabled_without_a_directory() {
        let request = MemberSpawnRequest::new("alpha", "dev-1", "sess-1", "/tmp/ws")
            .with_checkpoint_config(CheckpointConfig::default());
        let env = request.checkpoint_env();
        assert_eq!(
            env.get(CHECKPOINT_ENV).map(String::as_str),
            Some(CHECKPOINT_ON)
        );
        assert!(!env.contains_key(CHECKPOINT_DIR_ENV));
    }

    #[test]
    fn a_disabled_policy_exports_off_and_no_directory() {
        let request = MemberSpawnRequest::new("alpha", "dev-1", "sess-1", "/tmp/ws")
            .with_checkpoint_config(CheckpointConfig {
                policy: MemberCheckpointPolicy::disabled(),
                store: CheckpointStoreConfig::File {
                    dir: Some(PathBuf::from("/tmp/cp")),
                },
            });
        let env = request.checkpoint_env();
        assert_eq!(
            env.get(CHECKPOINT_ENV).map(String::as_str),
            Some(CHECKPOINT_OFF)
        );
        assert!(!env.contains_key(CHECKPOINT_DIR_ENV));
    }

    #[test]
    fn checkpoint_env_respects_an_explicit_override() {
        let request = MemberSpawnRequest::new("alpha", "dev-1", "sess-1", "/tmp/ws")
            .with_checkpoint_config(CheckpointConfig {
                policy: MemberCheckpointPolicy::default(),
                store: CheckpointStoreConfig::File {
                    dir: Some(PathBuf::from("/tmp/cp")),
                },
            });
        let mut command =
            StdioCommand::new("team-member").with_env(CHECKPOINT_DIR_ENV, "/explicit/cp");
        request.apply_identity_env(&mut command);
        assert_eq!(
            command.env().get(CHECKPOINT_DIR_ENV).map(String::as_str),
            Some("/explicit/cp")
        );
        assert_eq!(
            command.env().get(CHECKPOINT_ENV).map(String::as_str),
            Some(CHECKPOINT_ON)
        );
    }

    #[test]
    fn the_checkpoint_configuration_stays_out_of_the_json_form() {
        let request = MemberSpawnRequest::new("alpha", "dev-1", "sess-1", "/tmp/ws")
            .with_checkpoint_config(CheckpointConfig {
                policy: MemberCheckpointPolicy::disabled(),
                store: CheckpointStoreConfig::Memory,
            });
        let encoded = serde_json::to_string(&request).unwrap();
        assert!(!encoded.contains("checkpoint"));
        let decoded: MemberSpawnRequest = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded.checkpoint, None);
    }

    #[test]
    fn a_request_without_options_decodes_to_in_process() {
        let decoded: MemberSpawnRequest = serde_json::from_str(
            r#"{"team_name":"a","member_name":"dev-1","session_id":"s","workspace_root":"/tmp/ws"}"#,
        )
        .unwrap();
        assert_eq!(decoded.options, MemberSpawnOptions::default());
        assert_eq!(decoded.options.kind, MemberSpawnKind::InProcess);
        assert!(decoded.options.is_in_process());
    }

    #[test]
    fn spawn_options_travel_with_the_request() {
        let request = MemberSpawnRequest::new("a", "dev-1", "s", "/tmp/ws");
        assert_eq!(request.options.kind, MemberSpawnKind::InProcess);

        let ssh = SshTransportSpec {
            host: "build-box".to_string(),
            remote_command: "team-member serve".to_string(),
            ..SshTransportSpec::default()
        };
        let request =
            request.with_options(MemberSpawnOptions::from_kind(MemberSpawnKind::Ssh).with_ssh(ssh));
        let encoded = serde_json::to_string(&request).unwrap();
        assert!(encoded.contains("\"kind\":\"ssh\""));
        let decoded: MemberSpawnRequest = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, request);
        assert_eq!(decoded.options.kind, MemberSpawnKind::Ssh);
        assert_eq!(
            decoded.options.ssh.as_ref().unwrap().host,
            "build-box".to_string()
        );
    }

    #[test]
    fn unknown_request_fields_are_tolerated() {
        let decoded: MemberSpawnRequest = serde_json::from_str(
            r#"{"team_name":"a","options":{"kind":"process","unknown_knob":1},"extra":true}"#,
        )
        .unwrap();
        assert_eq!(decoded.options.kind, MemberSpawnKind::Process);
        assert_eq!(decoded.member_name, String::new());
    }

    /// A spawner is object-safe and can be stored behind an `Arc`.
    struct StubSpawner;

    #[async_trait]
    impl MemberSpawner for StubSpawner {
        async fn spawn(&self, _request: MemberSpawnRequest) -> Result<Arc<dyn MemberRuntime>> {
            Err(Error::unsupported(
                "in-process spawn is provided by TeamBackend",
            ))
        }

        async fn despawn(&self, _member_name: &str) -> Result<()> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn the_spawner_trait_is_object_safe() {
        let spawner: Arc<dyn MemberSpawner> = Arc::new(StubSpawner);
        let Err(error) = spawner
            .spawn(MemberSpawnRequest::new("alpha", "dev-1", "sess", "/tmp/ws"))
            .await
        else {
            panic!("the stub spawner must not build a runtime");
        };
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Unsupported
        );
        spawner.despawn("dev-1").await.unwrap();
    }

    #[tokio::test]
    async fn an_idle_task_stays_live_until_cancelled() {
        // Guards `is_finished` against a false positive: a gated task must not finish on its own.
        let (handle, ready) = gated_handle("dev-1");
        ready.await.unwrap();
        tokio::time::timeout(
            Duration::from_millis(20),
            handle.cancellation_token().cancelled(),
        )
        .await
        .unwrap_err();
        assert!(!handle.is_finished());
        handle.cancel();
        handle.join().await.unwrap();
    }
}
