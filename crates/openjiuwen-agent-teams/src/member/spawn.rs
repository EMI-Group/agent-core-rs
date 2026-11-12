//! Spawn seam: [`MemberSpawner`], its request, and the in-process task handle.
//!
//! Python parity source: `openjiuwen/agent_teams/spawn/inprocess_spawn.py` and
//! `spawn/inprocess_handle.py`. Python's `inprocess_spawn` builds a teammate `TeamAgent` and
//! creates an `asyncio.Task` around `Runner.run_agent_team`; the returned `InProcessSpawnHandle`
//! wraps that task (`cancel` / `join` / `abort`). The Rust port keeps the same shape: a
//! [`MemberSpawner`] builds the member's runtime and a [`InProcessSpawnHandle`] owns the tokio task
//! that drives it.
//!
//! Only the in-process mode is implemented; OS-process members, external-CLI members and the SSH
//! transport are deferred (see `src/member/CONTEXT.md`), so a spawner for them fails with
//! [`openjiuwen_common::error::Error::Unsupported`] rather than pretending to work.
//!
//! Ownership rule: a [`InProcessSpawnHandle`] is the only owner of its task. Dropping a
//! `JoinHandle` detaches the task instead of aborting it, so every handle that leaves
//! [`SpawnRegistry`] must be joined or aborted by its owner — `TeamBackend` for a live team, the
//! spawner for a failed spawn. The registry never drops a live handle on the floor; see
//! [`SpawnRegistry::cancel_all`].

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tracing::debug;

use super::runtime::MemberRuntime;

/// Everything a spawner needs to bring one member up.
///
/// Python passes a `TeamRuntimeContext` plus the owning `TeamAgent`; Rust keeps the request to the
/// facts a spawner cannot derive: which team and member, the session id to propagate, and where
/// the member's workspace is rooted.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
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
}

impl MemberSpawnRequest {
    /// Builds a request.
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

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::time::Duration;

    use tokio::sync::oneshot;

    use super::*;

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
        let decoded: MemberSpawnRequest = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, request);
        let default: MemberSpawnRequest = serde_json::from_str("{}").unwrap();
        assert_eq!(default, MemberSpawnRequest::default());
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
