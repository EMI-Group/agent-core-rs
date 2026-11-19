//! The production [`SubAgentSpawner`]: the harness runtime behind the rail seam.
//!
//! [`crate::rails::subagent`] defines the provider-neutral [`SubAgentSpawner`] the subagent
//! rail, the `task_tool` and upper layers drive; [`super::SubAgentRuntime`] is the in-process
//! implementation of sub-agents. [`RuntimeSubAgentSpawner`] is the adapter between the two,
//! so `SubagentRail::with_spawner` has a real implementor instead of only test fakes.
//!
//! Two behaviours are worth knowing before using it:
//!
//! - **Capacity is reserved at `spawn`.** A bounded [`tokio::sync::Semaphore`] with
//!   [`RuntimeSubAgentSpawner::limit`] permits backs every spawn, and `spawn` takes its
//!   permit with `try_acquire_owned`, so an over-limit call fails immediately with
//!   [`Error::Validation`] (`subagent concurrency limit reached`) instead of queueing behind
//!   a live task. A background watcher holds the permit until the instance reaches a terminal
//!   runtime status, then releases it and drops its cancellation token, so a caller that
//!   never waits cannot leak capacity.
//! - **One cancellation token per instance.** [`RuntimeSubAgentSpawner::close`] cancels it
//!   before asking the runtime to close. The token is this adapter's bookkeeping only: the
//!   runtime still owns turn abortion, and it refuses to close an instance whose turn is
//!   running (`cannot close running subagent: <id>`), which `close` propagates unchanged.
//!   A caller that must stop a running instance uses the runtime directly
//!   ([`SubAgentRuntime::send_input`] with `interrupt`, or [`SubAgentRuntime::cancel_all`]).
use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use openjiuwen_common::error::{Error, Result};
use parking_lot::Mutex;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio_util::sync::CancellationToken;

use crate::rails::{
    SpawnRequest, SubAgentHandle, SubAgentSpawner, SubAgentState, SubAgentStatus as RailStatus,
};

use super::runtime::{SubAgentId, SubAgentOutcome, SubAgentRuntime, SubAgentSpawnRequest};
use super::{SubagentStatus as RuntimeStatus, GENERAL_PURPOSE_NAME};

/// How often the background watcher re-checks a non-final instance.
///
/// The runtime's `wait` is notify-driven, so this is an upper bound between checks, not a
/// sleep: a status transition (or a timeout) wakes the watcher at once.
const WATCH_POLL_INTERVAL: Duration = Duration::from_millis(100);

/// `wait` budget when the caller passes `None`.
///
/// Python's `TURN_TIMEOUT_S_DEFAULT` (2400 s) is the runtime's own per-turn timeout, so one
/// turn can never outlive a default wait by much.
const DEFAULT_WAIT_TIMEOUT: Duration = Duration::from_secs(2400);

/// The [`SubAgentSpawner`] implementation over the harness [`SubAgentRuntime`].
///
/// Share it behind an `Arc` (the rail does): the semaphore and the token map are the
/// instance-wide concurrency ledger, so two clones would double the limit.
pub struct RuntimeSubAgentSpawner {
    /// The in-process runtime that owns the child instances.
    runtime: Arc<SubAgentRuntime>,
    /// Capacity ledger: one permit per live spawn, [`Self::limit`] permits in total.
    semaphore: Arc<Semaphore>,
    /// Cancellation token of every instance this spawner started, keyed by runtime id.
    tokens: Arc<Mutex<BTreeMap<String, Arc<CancellationToken>>>>,
    /// Number of permits; mirrors the semaphore's capacity for the error message.
    limit: usize,
}

impl RuntimeSubAgentSpawner {
    /// Wraps `runtime` with the concurrency limit the runtime itself enforces.
    ///
    /// The limit is `runtime.quota().max_concurrent_running`, clamped to at least one so a
    /// degenerate quota cannot deadlock the seam.
    #[must_use]
    pub fn new(runtime: Arc<SubAgentRuntime>) -> Self {
        let limit = runtime.quota().max_concurrent_running.max(1);
        Self::with_limit(runtime, limit)
    }

    /// Wraps `runtime` with an explicit concurrency limit.
    ///
    /// A `max_concurrent` of `0` is clamped to `1` and logged: the seam has no fallible
    /// constructor, and a zero-permit spawner could never spawn anything.
    #[must_use]
    pub fn with_limit(runtime: Arc<SubAgentRuntime>, max_concurrent: usize) -> Self {
        if max_concurrent == 0 {
            tracing::warn!(
                target: "openjiuwen_harness::subagents",
                "subagent spawner concurrency limit 0 clamped to 1"
            );
        }
        let limit = max_concurrent.max(1);
        Self {
            runtime,
            semaphore: Arc::new(Semaphore::new(limit)),
            tokens: Arc::new(Mutex::new(BTreeMap::new())),
            limit,
        }
    }

    /// Returns the runtime this spawner delegates to.
    #[must_use]
    pub fn runtime(&self) -> &Arc<SubAgentRuntime> {
        &self.runtime
    }

    /// Returns the concurrency limit, i.e. the maximum number of live spawns.
    #[must_use]
    pub fn limit(&self) -> usize {
        self.limit
    }

    /// Returns how many permits are currently held by live spawns.
    ///
    /// This is the number of instances this spawner started that have not reached a terminal
    /// runtime status yet; a fully drained spawner reports `0` again.
    #[must_use]
    pub fn active_permits(&self) -> usize {
        self.limit
            .saturating_sub(self.semaphore.available_permits())
    }

    /// Returns the number of instances still tracked (token map size).
    #[must_use]
    pub fn tracked(&self) -> usize {
        self.tokens.lock().len()
    }

    /// Starts an instance through the shared concurrency ledger.
    ///
    /// This is the bounded spawn path every model-facing tool uses: the permit is taken with
    /// `try_acquire_owned` before the runtime reserves a slot, so an over-limit call fails
    /// immediately with [`Error::Validation`] instead of queueing behind a live task.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] when the concurrency limit is reached or the request is invalid,
    /// and every runtime quota/child-construction failure unchanged.
    pub async fn spawn_request(&self, request: SubAgentSpawnRequest) -> Result<SubAgentId> {
        let Ok(permit) = Arc::clone(&self.semaphore).try_acquire_owned() else {
            return Err(Error::validation(format!(
                "subagent concurrency limit reached: {} of {} slots in use",
                self.active_permits(),
                self.limit
            )));
        };
        let id = match self.runtime.spawn(request).await {
            Ok(id) => id,
            Err(error) => {
                drop(permit);
                return Err(error);
            }
        };
        let token = Arc::new(CancellationToken::new());
        self.tokens.lock().insert(id.clone(), Arc::clone(&token));
        self.spawn_watcher(id.clone(), permit, token);
        Ok(id)
    }

    /// Starts an instance and a background watcher holding its permit.
    async fn start(&self, request: SpawnRequest) -> Result<SubAgentHandle> {
        let subagent_type = request
            .preset
            .clone()
            .unwrap_or_else(|| GENERAL_PURPOSE_NAME.to_owned());
        // `parent_task_id` is a caller-side transcript grouping key; the runtime's
        // `parent_subagent_id` names a live parent *instance* (it refuses an unknown one), so
        // it is never derived from it.
        let runtime_request = SubAgentSpawnRequest {
            subagent_type,
            task_description: request.prompt,
            subagent_id: request.agent_id,
            ..SubAgentSpawnRequest::default()
        };
        let id = self.spawn_request(runtime_request).await?;
        Ok(SubAgentHandle::new(id, request.preset))
    }

    /// Spawns the task that releases `permit` once instance `id` is terminal.
    ///
    /// The watcher holds the permit, not the spawner, so dropping the spawner (or the rail)
    /// while a sub-agent runs still frees capacity when the run ends. It never holds the
    /// token lock across an `await`, and it drops the token entry only when the entry is
    /// still its own, so a later instance reusing the id keeps its own token.
    fn spawn_watcher(
        &self,
        id: String,
        permit: OwnedSemaphorePermit,
        token: Arc<CancellationToken>,
    ) {
        let runtime = Arc::clone(&self.runtime);
        let tokens = Arc::clone(&self.tokens);
        tokio::spawn(async move {
            let mut cancellation_seen = false;
            loop {
                let outcomes = tokio::select! {
                    outcomes = runtime.wait(std::slice::from_ref(&id), WATCH_POLL_INTERVAL) => {
                        outcomes
                    }
                    // A cancelled token only wakes the loop early to re-check; the permit is
                    // released on the terminal status, never on cancellation.
                    () = token.cancelled(), if !cancellation_seen => {
                        cancellation_seen = true;
                        continue;
                    }
                };
                if outcomes.first().is_some_and(SubAgentOutcome::is_final) {
                    break;
                }
            }
            drop(permit);
            let mut tracked = tokens.lock();
            if tracked
                .get(&id)
                .is_some_and(|stored| Arc::ptr_eq(stored, &token))
            {
                tracked.remove(&id);
            }
        });
    }

    /// Returns the tracked cancellation token of `id`, when this spawner started it.
    fn token_for(&self, id: &str) -> Option<Arc<CancellationToken>> {
        self.tokens.lock().get(id).map(Arc::clone)
    }

    /// Maps one runtime outcome onto the rail snapshot.
    fn rail_status(outcome: &SubAgentOutcome) -> RailStatus {
        RailStatus {
            id: outcome.subagent_id.clone(),
            state: rail_state(outcome.status),
            output: outcome.output.clone(),
            error: outcome.error.clone(),
        }
    }

    /// The typed unknown-id error both `wait` and `close` report.
    fn unknown(id: &str) -> Error {
        Error::not_found(format!("subagent not found: {id}"))
    }
}

impl fmt::Debug for RuntimeSubAgentSpawner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RuntimeSubAgentSpawner")
            .field("limit", &self.limit)
            .field("active_permits", &self.active_permits())
            .field("tracked", &self.tracked())
            .field("parent_session_id", &self.runtime.parent_session_id())
            .finish_non_exhaustive()
    }
}

/// Maps a runtime status onto the rail lifecycle state.
///
/// `PendingInit`/`Running` are still working, `Completed` succeeded, `Interrupted`/`Errored`
/// failed, and `Closed`/`NotFound` are no longer live.
fn rail_state(status: RuntimeStatus) -> SubAgentState {
    match status {
        RuntimeStatus::PendingInit | RuntimeStatus::Running => SubAgentState::Running,
        RuntimeStatus::Completed => SubAgentState::Completed,
        RuntimeStatus::Interrupted | RuntimeStatus::Errored => SubAgentState::Failed,
        RuntimeStatus::Closed | RuntimeStatus::NotFound => SubAgentState::Closed,
    }
}

#[async_trait::async_trait]
impl SubAgentSpawner for RuntimeSubAgentSpawner {
    /// Starts a subagent on the runtime, reserving one concurrency permit.
    ///
    /// `preset` addresses a registered sub-agent definition (`agent_card.name`) and defaults
    /// to [`GENERAL_PURPOSE_NAME`]; `agent_id` becomes the runtime instance id when the
    /// registry has no sticky id for the type.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] when the concurrency limit is reached or the type is unknown,
    /// [`Error::Config`] when a runtime quota is exhausted, and the child-construction error
    /// when the nested agent cannot be built.
    async fn spawn(&self, request: SpawnRequest) -> Result<SubAgentHandle> {
        self.start(request).await
    }

    /// Waits for `id` to finish, or for `timeout` (default 2400 s) to elapse.
    ///
    /// A timed-out wait returns the current snapshot, so `state` is still `Running`.
    ///
    /// # Errors
    ///
    /// [`Error::NotFound`] for an id the runtime never knew.
    async fn wait(&self, id: &str, timeout: Option<Duration>) -> Result<RailStatus> {
        let timeout = timeout.unwrap_or(DEFAULT_WAIT_TIMEOUT);
        // `wait` reports unknown ids as `NotFound` outcomes; distinguish them from a real
        // tombstone (a closed instance still has metadata) before waiting.
        if self.runtime.get_outcome(id).is_none() {
            return Err(Self::unknown(id));
        }
        let outcomes = self.runtime.wait(&[id.to_owned()], timeout).await;
        match outcomes.first() {
            Some(outcome) if outcome.status != RuntimeStatus::NotFound => {
                Ok(Self::rail_status(outcome))
            }
            _ => Err(Self::unknown(id)),
        }
    }

    /// Lists the live (non-closed) instances in spawn order, without their output.
    ///
    /// # Errors
    ///
    /// Currently infallible; the signature mirrors the seam.
    async fn list(&self) -> Result<Vec<RailStatus>> {
        Ok(self
            .runtime
            .list()
            .into_iter()
            .map(|metadata| RailStatus {
                id: metadata.subagent_id,
                state: rail_state(metadata.status),
                output: None,
                error: None,
            })
            .collect())
    }

    /// Queues `input` as a follow-up turn, without interrupting the running one.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] for an unknown or closed id, and for a restored instance that
    /// has no live child agent yet.
    async fn send_input(&self, id: &str, input: String) -> Result<()> {
        self.runtime.send_input(id, &input, false).await
    }

    /// Cancels the instance's token and closes it on the runtime.
    ///
    /// # Errors
    ///
    /// [`Error::NotFound`] for an unknown id, and the runtime's refusal
    /// (`cannot close running subagent: <id>`) while a turn is running.
    async fn close(&self, id: &str) -> Result<()> {
        if self.runtime.get_metadata(id).is_none() {
            return Err(Self::unknown(id));
        }
        if let Some(token) = self.token_for(id) {
            token.cancel();
        }
        self.runtime.close(id).await.map(|_previous| ())?;
        self.tokens.lock().remove(id);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]

    use std::time::Duration;

    use openjiuwen_common::error::ErrorKind;
    use tokio::sync::Notify;

    use super::*;
    use crate::subagents::test_support::{
        build_registry, build_runtime, echo_runtime, Behavior, FakeFactory,
    };
    use crate::subagents::{
        general_purpose_config, SubAgentRegistry, SubAgentRuntime, SubAgentSpec,
    };

    /// Bounded deadline for a wait a fake agent satisfies on its first turn.
    const BOUND: Duration = Duration::from_secs(10);

    /// How long the polling helpers below keep checking (never a fixed sleep).
    const POLL: Duration = Duration::from_secs(5);

    /// Polls `condition` until it holds or [`POLL`] elapses.
    async fn until(mut condition: impl FnMut() -> bool) -> bool {
        let deadline = tokio::time::Instant::now() + POLL;
        while tokio::time::Instant::now() < deadline {
            if condition() {
                return true;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        condition()
    }

    #[test]
    fn the_default_limit_is_the_runtime_quota() {
        let (runtime, _factory) = echo_runtime();
        let spawner = RuntimeSubAgentSpawner::new(runtime);
        assert_eq!(spawner.limit(), 5);
        assert_eq!(spawner.active_permits(), 0);
        assert_eq!(spawner.tracked(), 0);
        assert!(Arc::ptr_eq(spawner.runtime(), spawner.runtime()));
        assert!(format!("{spawner:?}").contains("limit: 5"));
    }

    #[test]
    fn the_default_limit_follows_a_reduced_quota() {
        let factory = FakeFactory::new(Behavior::Echo);
        let runtime = Arc::new(
            SubAgentRuntime::new(build_registry(), factory, "parent-session").with_limits(10, 3),
        );
        assert_eq!(RuntimeSubAgentSpawner::new(runtime).limit(), 3);
    }

    #[test]
    fn a_zero_limit_is_clamped_to_one() {
        let (runtime, _factory) = echo_runtime();
        let spawner = RuntimeSubAgentSpawner::with_limit(runtime, 0);
        assert_eq!(spawner.limit(), 1);
        assert_eq!(spawner.active_permits(), 0);
    }

    #[test]
    fn every_runtime_status_maps_onto_a_rail_state() {
        assert_eq!(
            rail_state(RuntimeStatus::PendingInit),
            SubAgentState::Running
        );
        assert_eq!(rail_state(RuntimeStatus::Running), SubAgentState::Running);
        assert_eq!(
            rail_state(RuntimeStatus::Completed),
            SubAgentState::Completed
        );
        assert_eq!(
            rail_state(RuntimeStatus::Interrupted),
            SubAgentState::Failed
        );
        assert_eq!(rail_state(RuntimeStatus::Errored), SubAgentState::Failed);
        assert_eq!(rail_state(RuntimeStatus::Closed), SubAgentState::Closed);
        assert_eq!(rail_state(RuntimeStatus::NotFound), SubAgentState::Closed);
    }

    #[tokio::test]
    async fn an_unknown_id_is_not_found_for_wait_and_close() {
        let (runtime, _factory) = echo_runtime();
        let spawner = RuntimeSubAgentSpawner::new(runtime);
        let error = spawner.wait("nope", Some(BOUND)).await.unwrap_err();
        assert_eq!(error.kind(), ErrorKind::NotFound);
        assert!(error.to_string().contains("subagent not found: nope"));
        let error = spawner.close("nope").await.unwrap_err();
        assert_eq!(error.kind(), ErrorKind::NotFound);
    }

    #[tokio::test]
    async fn a_spawn_runs_to_completion_and_releases_its_permit() {
        let (runtime, factory) = echo_runtime();
        let spawner = RuntimeSubAgentSpawner::with_limit(runtime, 2);

        let handle = spawner
            .spawn(SpawnRequest::new("dig").with_preset("code_agent"))
            .await
            .unwrap();
        assert_eq!(handle.preset.as_deref(), Some("code_agent"));
        assert_eq!(factory.recorded().len(), 1);

        let status = spawner.wait(&handle.id, Some(BOUND)).await.unwrap();
        assert_eq!(status.state, SubAgentState::Completed);
        assert_eq!(status.output.as_deref(), Some("echo:dig"));
        assert!(status.error.is_none());

        let listed = spawner.list().await.unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].id, handle.id);
        assert_eq!(listed[0].state, SubAgentState::Completed);
        assert!(listed[0].output.is_none());

        // The watcher frees the slot and forgets the token once the instance is final.
        assert!(until(|| spawner.active_permits() == 0).await);
        assert!(until(|| spawner.tracked() == 0).await);
    }

    #[tokio::test]
    async fn a_missing_preset_falls_back_to_general_purpose() {
        let (runtime, factory) = echo_runtime();
        let spawner = RuntimeSubAgentSpawner::new(runtime);
        let handle = spawner.spawn(SpawnRequest::new("dig")).await.unwrap();
        assert!(handle.preset.is_none());
        // `build_registry` registers the general-purpose fallback, as the builder does when
        // `add_general_purpose_agent` is set.
        assert_eq!(
            factory.recorded()[0].0.agent_card.name,
            GENERAL_PURPOSE_NAME.to_owned()
        );
        let status = spawner.wait(&handle.id, Some(BOUND)).await.unwrap();
        assert_eq!(status.state, SubAgentState::Completed);
    }

    #[tokio::test]
    async fn an_unregistered_preset_reports_the_runtime_error_and_frees_the_slot() {
        let (runtime, _factory) = echo_runtime();
        let spawner = RuntimeSubAgentSpawner::with_limit(runtime, 1);
        let error = spawner
            .spawn(SpawnRequest::new("dig").with_preset("ghost"))
            .await
            .unwrap_err();
        assert!(error.to_string().contains("subagent type not found: ghost"));
        assert_eq!(spawner.active_permits(), 0);
        assert_eq!(spawner.tracked(), 0);
    }

    #[tokio::test]
    async fn a_saturated_spawner_rejects_the_second_spawn_until_the_first_ends() {
        let gate = Arc::new(Notify::new());
        let factory = FakeFactory::new(Behavior::Blocked(Arc::clone(&gate)));
        let runtime = build_runtime(factory);
        let spawner = RuntimeSubAgentSpawner::with_limit(runtime, 1);

        let first = spawner.spawn(SpawnRequest::new("dig")).await.unwrap();
        assert_eq!(spawner.active_permits(), 1);

        let error = spawner
            .spawn(SpawnRequest::new("second"))
            .await
            .unwrap_err();
        assert!(error
            .to_string()
            .contains("subagent concurrency limit reached"));
        assert_eq!(spawner.active_permits(), 1);

        gate.notify_one();
        let status = spawner.wait(&first.id, Some(BOUND)).await.unwrap();
        assert_eq!(status.state, SubAgentState::Completed);
        assert!(until(|| spawner.active_permits() == 0).await);

        // Capacity is reusable once the first instance is final.
        let second = spawner.spawn(SpawnRequest::new("again")).await.unwrap();
        assert_eq!(spawner.active_permits(), 1);
        gate.notify_one();
        spawner.wait(&second.id, Some(BOUND)).await.unwrap();
        assert!(until(|| spawner.active_permits() == 0).await);
    }

    #[tokio::test]
    async fn send_input_queues_a_follow_up_turn() {
        let (runtime, factory) = echo_runtime();
        let spawner = RuntimeSubAgentSpawner::new(runtime);
        let handle = spawner.spawn(SpawnRequest::new("first")).await.unwrap();
        spawner
            .send_input(&handle.id, "follow up".to_string())
            .await
            .unwrap();
        // A completed instance parks idle, so the queued follow-up runs as its own turn.
        let status = spawner.wait(&handle.id, Some(BOUND)).await.unwrap();
        assert_eq!(status.state, SubAgentState::Completed);
        assert!(until(|| factory.recorded().len() == 1).await);
    }

    #[tokio::test]
    async fn close_of_a_finished_instance_removes_its_token() {
        let (runtime, _factory) = echo_runtime();
        let spawner = RuntimeSubAgentSpawner::new(runtime);
        let handle = spawner.spawn(SpawnRequest::new("dig")).await.unwrap();
        spawner.wait(&handle.id, Some(BOUND)).await.unwrap();

        spawner.close(&handle.id).await.unwrap();
        assert_eq!(spawner.tracked(), 0);
        // The tombstone still answers, now as a closed instance.
        let status = spawner.wait(&handle.id, Some(BOUND)).await.unwrap();
        assert_eq!(status.state, SubAgentState::Closed);
        assert!(spawner.list().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn close_of_a_running_instance_cancels_the_token_and_reports_the_refusal() {
        let gate = Arc::new(Notify::new());
        let factory = FakeFactory::new(Behavior::Blocked(Arc::clone(&gate)));
        let runtime = build_runtime(factory);
        let spawner = RuntimeSubAgentSpawner::new(runtime);
        let handle = spawner.spawn(SpawnRequest::new("dig")).await.unwrap();
        // `spawn` returns once the first turn is queued; wait for it to actually run, because
        // the runtime only refuses to close a `Running` instance.
        let deadline = tokio::time::Instant::now() + BOUND;
        while spawner.runtime().get_status(&handle.id) != RuntimeStatus::Running {
            assert!(
                tokio::time::Instant::now() < deadline,
                "the gated turn never started"
            );
            tokio::time::sleep(Duration::from_millis(5)).await;
        }

        let token = spawner.token_for(&handle.id).unwrap();
        assert!(!token.is_cancelled());
        let error = spawner.close(&handle.id).await.unwrap_err();
        assert!(error.to_string().contains("cannot close running subagent"));
        // The token is this adapter's bookkeeping; it is cancelled even though the runtime
        // refused the close, and the watcher still owns the permit.
        assert!(token.is_cancelled());
        assert_eq!(spawner.active_permits(), 1);

        gate.notify_one();
        let status = spawner.wait(&handle.id, Some(BOUND)).await.unwrap();
        assert_eq!(status.state, SubAgentState::Completed);
        // The watcher drops the cancelled token once the instance is final.
        assert!(until(|| spawner.tracked() == 0).await);
        assert!(until(|| spawner.active_permits() == 0).await);
    }

    #[test]
    fn the_general_purpose_fallback_is_registered_by_the_test_registry() {
        // Guards the assumption `a_missing_preset_falls_back_to_general_purpose` relies on.
        let registry = SubAgentRegistry::with_builtin_presets("en");
        assert!(!registry.contains(GENERAL_PURPOSE_NAME));
        registry.register(SubAgentSpec::new(general_purpose_config()));
        assert!(registry.contains(GENERAL_PURPOSE_NAME));
    }
}
