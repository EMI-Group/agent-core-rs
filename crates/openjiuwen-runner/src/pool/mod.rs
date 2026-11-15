//! The in-process team runtime pool: pooled team runs with bounded
//! concurrency, leases, idle eviction, and a HITL payload buffer.
//!
//! **Intent.** A [`TeamPool`] starts agent-team runs in the background and hands
//! the caller a cheap [`PoolRun`] handle (or a [`PoolLease`] that keeps the run
//! alive while it is held). One run per `(team_name, session_id)` is live at a
//! time; the pool bounds how many run concurrently, publishes one
//! [`PoolFrame`] stream per run, and evicts idle runs.
//!
//! **Python correspondence.** This ports the *in-process* part of
//! `openjiuwen/core/runner/team_runner.py` and
//! `openjiuwen/agent_teams/runtime/`: `_TeamRunnerMixin.stop_agent_team` /
//! `pause_agent_team` / `list_active_teams` /
//! `_maybe_release_team_session` (as [`TeamPool::stop`], [`TeamPool::pause`],
//! [`TeamPool::list_active`], and [`TeamPool::stop_all`]; `pause` keeps Python's
//! entry point but refuses, see *Pause/resume are refused* below), the
//! `TeamRuntimeManager.activate` / `finalize` run cycle (as [`TeamPool::start`]
//! and the worker's exactly-once finalisation), the `TeamRuntimePool`
//! ([`ActiveTeamInfo`] is a read-only snapshot, like Python's), the
//! `RunActionKind` rejections ([`PoolError`]), and `interact_agent_team`'s
//! inbound channel (as the per-run payload buffer).
//!
//! **Single live run per key.** A second [`TeamPool::start`] on a live key is
//! [`PoolError::Busy`]; [`TeamPool::acquire`] *joins* the live run instead and
//! ignores the joining request's inputs and options. Once the run is terminal a
//! new `start` replaces it, which is Python's cold-recover path without the
//! disk hop.
//!
//! **One owner for observation.** A pooled run registers itself on the run
//! monitor and on the HITL hub, then drives `crate::run::stream_team` with an
//! inner [`RunServices`](crate::run::RunServices) whose `monitor` and `hitl` are
//! cleared: the pool owns those two seams for a pooled run, so the run must not
//! create a second monitor entry or a second hub registration.
//!
//! **Payloads, not live gates.** A pooled run's inbound human payloads are
//! buffered ([`PoolRun::payloads`], [`TeamPool::take_payloads`]) and the host
//! re-invokes the strategy with them. Python's live `InteractGate` wake-up is
//! deferred, so no session is attached to the hub entry and
//! `InteractionHub::answer` cannot resolve an in-flight interaction request of a
//! pooled run.
//!
//! **Best-effort observation.** The frame channel is a `tokio::sync::broadcast`:
//! a slow subscriber sees `Lagged` and never a replay. Status mirroring and hub
//! unregistration are all logged and never turned into control flow.
//!
//! **Pause/resume are refused.** [`TeamPool::pause`] and [`TeamPool::resume`]
//! answer [`Error::Unsupported`](openjiuwen_common::error::Error::Unsupported) for
//! a live run: an in-process strategy owns no mid-flight suspension seam, so the
//! pool cannot suspend or restart the invocation and publishes no frame for it.
//! Stopping a pooled run is [`TeamPool::stop`] (or [`PoolRun::cancel`]), which
//! trips the run's cancellation token.

mod entry;
mod error;
mod key;
mod reaper;
mod request;
mod run;

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::{Mutex, MutexGuard};
use tokio::sync::Semaphore;
use tracing::{debug, warn};

use crate::config::{RunnerConfig, TeamPoolConfig};
use crate::monitor::RunPhase;

pub use entry::{ActiveTeamInfo, PoolFrame};
pub use error::{PoolError, PoolShutdown};
pub use key::PoolKey;
pub use request::PoolRequest;
pub use run::{PoolLease, PoolRun};

use entry::{select_pruned, Entry};
use reaper::ReaperHandle;

/// The `tracing` target every pool log line uses.
pub(crate) const LOG_TARGET: &str = "openjiuwen_runner::pool";

/// The shared state of one pool: the entries, the permit pool, and the reaper.
pub(crate) struct TeamPoolInner {
    /// The pool's own configuration.
    config: TeamPoolConfig,
    /// The concurrency bound; one permit per admitted run.
    semaphore: Arc<Semaphore>,
    /// The retained entries, ordered by `(team_name, session_id)`.
    entries: Mutex<BTreeMap<PoolKey, Arc<Entry>>>,
    /// Whether [`TeamPool::stop_all`] closed the pool.
    closed: AtomicBool,
    /// The finalisation sequence counter used to order pruning.
    seq: AtomicU64,
    /// The single idle-eviction task, when one is running.
    reaper: Mutex<Option<ReaperHandle>>,
}

impl TeamPoolInner {
    /// The pool configuration.
    pub(crate) fn config(&self) -> &TeamPoolConfig {
        &self.config
    }

    /// The concurrency semaphore.
    pub(crate) fn semaphore(&self) -> Arc<Semaphore> {
        Arc::clone(&self.semaphore)
    }

    /// Whether the pool was closed.
    pub(crate) fn is_closed(&self) -> bool {
        self.closed.load(Ordering::SeqCst)
    }

    /// Closes the pool so no further run is admitted.
    pub(crate) fn close(&self) {
        self.closed.store(true, Ordering::SeqCst);
    }

    /// Hands out the next finalisation sequence number.
    pub(crate) fn next_seq(&self) -> u64 {
        self.seq.fetch_add(1, Ordering::SeqCst)
    }

    /// Locks the entry map.
    pub(crate) fn entries(&self) -> MutexGuard<'_, BTreeMap<PoolKey, Arc<Entry>>> {
        self.entries.lock()
    }

    /// Returns the entry pooled under `key`, whatever its phase.
    pub(crate) fn entry(&self, key: &PoolKey) -> Option<Arc<Entry>> {
        self.entries.lock().get(key).map(Arc::clone)
    }

    /// Removes and returns the entry pooled under `key`.
    pub(crate) fn remove(&self, key: &PoolKey) -> Option<Arc<Entry>> {
        self.entries.lock().remove(key)
    }

    /// Returns every retained key, in reporting order.
    pub(crate) fn keys(&self) -> Vec<PoolKey> {
        self.entries.lock().keys().cloned().collect()
    }

    /// How many entries are retained (live plus retained finished ones).
    pub(crate) fn len(&self) -> usize {
        self.entries.lock().len()
    }

    /// Takes the running reaper, so its task can be joined or aborted.
    pub(crate) fn take_reaper(&self) -> Option<ReaperHandle> {
        self.reaper.lock().take()
    }

    /// Installs the reaper, unless one is already running.
    ///
    /// Returns `true` when a reaper was started.
    pub(crate) fn install_reaper(&self, reaper: ReaperHandle) -> bool {
        let mut slot = self.reaper.lock();
        if slot.is_some() {
            return false;
        }
        *slot = Some(reaper);
        true
    }

    /// Prunes finished entries beyond `retain_finished`, oldest first.
    ///
    /// Only a terminal entry with no live lease is a candidate, so the bound is
    /// best effort: when every finished entry is leased, nothing is pruned.
    pub(crate) fn prune_finished(&self) {
        let finished: Vec<(u64, PoolKey)> = {
            let entries = self.entries.lock();
            entries
                .iter()
                .filter(|(_, entry)| entry.leases() == 0)
                .filter_map(|(key, entry)| entry.finished_seq().map(|seq| (seq, key.clone())))
                .collect()
        };
        let prune = select_pruned(&finished, self.config.retain_finished);
        if prune.is_empty() {
            return;
        }
        {
            let mut entries = self.entries.lock();
            for key in &prune {
                entries.remove(key);
            }
        }
        debug!(
            target: LOG_TARGET,
            pruned = prune.len(),
            "pruned finished pooled team runs"
        );
    }
}

/// The in-process pool of agent-team runs.
///
/// A pool is a cheap, cloneable handle: every clone shares one entry map, one
/// concurrency bound, and one reaper. See the [module docs](self) for the
/// single-live-run rule, the observation ownership, and the payload contract.
#[derive(Clone)]
pub struct TeamPool {
    /// The shared pool state.
    inner: Arc<TeamPoolInner>,
}

impl TeamPool {
    /// Builds a pool with [`TeamPoolConfig::default`].
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(TeamPoolConfig::default())
    }

    /// Builds a pool from explicit settings.
    #[must_use]
    pub fn with_config(config: TeamPoolConfig) -> Self {
        let semaphore = Arc::new(Semaphore::new(config.max_runs()));
        Self {
            inner: Arc::new(TeamPoolInner {
                config,
                semaphore,
                entries: Mutex::new(BTreeMap::new()),
                closed: AtomicBool::new(false),
                seq: AtomicU64::new(0),
                reaper: Mutex::new(None),
            }),
        }
    }

    /// Builds a pool from the `team_pool` block of `config`.
    #[must_use]
    pub fn from_runner_config(config: &RunnerConfig) -> Self {
        Self::with_config(config.team_pool.clone())
    }

    /// The pool's configuration.
    #[must_use]
    pub fn config(&self) -> TeamPoolConfig {
        self.inner.config.clone()
    }

    /// The effective concurrency bound.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.inner.config.max_runs()
    }

    /// How many runs can still be admitted without waiting.
    #[must_use]
    pub fn available_permits(&self) -> usize {
        self.inner.semaphore.available_permits()
    }

    /// Returns the snapshot of the run pooled under `key`, if one is retained.
    ///
    /// A terminal entry answers until it is pruned
    /// (`TeamPoolConfig::retain_finished`) or evicted.
    #[must_use]
    pub fn get(&self, key: &PoolKey) -> Option<ActiveTeamInfo> {
        self.inner.entry(key).map(|entry| entry.info())
    }

    /// Lists the runs that are not finished, ordered by `(team_name,
    /// session_id)`.
    #[must_use]
    pub fn list_active(&self) -> Vec<ActiveTeamInfo> {
        self.inner
            .entries()
            .values()
            .filter(|entry| !entry.is_terminal())
            .map(|entry| entry.info())
            .collect()
    }

    /// Stops every run and closes the pool.
    ///
    /// The reaper is cancelled and joined (bounded by
    /// `TeamPoolConfig::stop_timeout`), then each retained run is stopped. Every
    /// entry is removed, live or already finished; [`PoolShutdown::stopped`]
    /// counts only the live ones (a retained finished entry is swept but not
    /// counted as a stopped run). After this call the pool is closed:
    /// `start`/`try_start`/`acquire` report [`PoolError::Closed`], and a host
    /// that wants to run teams again builds a fresh pool (the runner facade does
    /// exactly that when it restarts).
    pub async fn stop_all(&self) -> PoolShutdown {
        self.inner.close();
        self.stop_reaper().await;
        // Read the phases before the sweep removes the entries: `stopped` counts
        // the runs that were live when the sweep began, never a retained finished
        // one.
        let sweep: Vec<(PoolKey, bool)> = self
            .phases()
            .into_iter()
            .map(|(key, phase)| (key, phase.is_terminal()))
            .collect();
        let mut shutdown = PoolShutdown::default();
        for (key, finished) in sweep {
            match self.stop(&key).await {
                Ok(true) => {
                    if !finished {
                        shutdown.stopped += 1;
                    }
                }
                Ok(false) => {}
                Err(error) => {
                    warn!(
                        target: LOG_TARGET,
                        key = %key,
                        %error,
                        "stopping a pooled team run failed"
                    );
                    if shutdown.first_error.is_none() {
                        shutdown.first_error = Some(error);
                    }
                }
            }
        }
        debug!(
            target: LOG_TARGET,
            stopped = shutdown.stopped,
            "team pool closed"
        );
        shutdown
    }

    /// Returns how many entries are retained (live plus finished).
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns `true` when no entry is retained.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns `true` once [`TeamPool::stop_all`] closed the pool.
    #[must_use]
    pub fn is_closed(&self) -> bool {
        self.inner.is_closed()
    }

    /// Returns the phases of the retained entries, for diagnostics.
    ///
    /// A cheap read-only projection used by hosts that render a queue without
    /// paying for [`ActiveTeamInfo`] snapshots.
    #[must_use]
    pub fn phases(&self) -> Vec<(PoolKey, RunPhase)> {
        self.inner
            .entries()
            .iter()
            .map(|(key, entry)| (key.clone(), entry.phase()))
            .collect()
    }

    /// Cancels and joins the reaper, bounded by `stop_timeout`.
    async fn stop_reaper(&self) {
        let Some(reaper) = self.inner.take_reaper() else {
            return;
        };
        reaper.token.cancel();
        let timeout = self.inner.config.stop_timeout();
        let mut handle = reaper.handle;
        if tokio::time::timeout(timeout, &mut handle).await.is_err() {
            handle.abort();
        }
    }
}

impl Default for TeamPool {
    /// Same as [`TeamPool::new`].
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for TeamPool {
    /// Reports the configuration, the retained entries, and whether it is closed.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TeamPool")
            .field("capacity", &self.capacity())
            .field("available_permits", &self.available_permits())
            .field("entries", &self.len())
            .field("closed", &self.is_closed())
            .finish()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::run::RunServices;

    #[test]
    fn a_pool_starts_empty_and_open() {
        let pool = TeamPool::new();
        assert!(pool.is_empty());
        assert_eq!(pool.len(), 0);
        assert!(!pool.is_closed());
        assert_eq!(pool.capacity(), 8);
        assert_eq!(pool.available_permits(), 8);
        assert!(pool.get(&PoolKey::new("crew", "s-1")).is_none());
        assert_eq!(pool.list_active(), Vec::new());
        assert_eq!(pool.phases(), Vec::new());
        assert_eq!(pool.config(), TeamPoolConfig::default());
        let debug = format!("{pool:?}");
        assert!(debug.contains("capacity: 8"), "{debug}");
    }

    #[test]
    fn clones_share_one_pool_and_the_config_decides_the_bound() {
        let config = TeamPoolConfig::default().with_max_concurrent_runs(2);
        let pool = TeamPool::with_config(config.clone());
        let clone = pool.clone();
        assert_eq!(clone.capacity(), 2);
        assert_eq!(clone.available_permits(), 2);
        assert_eq!(clone.config(), config);
        assert!(Arc::ptr_eq(&pool.inner, &clone.inner));
    }

    #[test]
    fn the_runner_config_supplies_the_pool_config() {
        let config = RunnerConfig::default()
            .with_team_pool(TeamPoolConfig::default().with_max_concurrent_runs(3));
        let pool = TeamPool::from_runner_config(&config);
        assert_eq!(pool.capacity(), 3);
        assert_eq!(pool.config(), config.team_pool);
    }

    #[test]
    fn pruning_respects_the_retain_bound_and_live_leases() {
        let pool = TeamPool::with_config(TeamPoolConfig::default().with_retain_finished(1));
        let keys = [
            PoolKey::new("crew", "s-1"),
            PoolKey::new("crew", "s-2"),
            PoolKey::new("crew", "s-3"),
        ];
        let mut entries = Vec::new();
        for key in &keys {
            let entry = Arc::new(Entry::new(entry::EntryInit {
                run_id: format!("run-{}", key.session_id),
                key: key.clone(),
                label: "crew".to_owned(),
                deadline: None,
                payload_capacity: 2,
                frame_capacity: 2,
                hub: None,
                monitor: None,
            }));
            pool.inner.entries().insert(key.clone(), Arc::clone(&entry));
            entries.push(entry);
        }

        // The oldest finished entry holds a lease, so it is never a candidate.
        entries[0].add_lease();
        for (index, entry) in entries.iter().enumerate() {
            let seq = pool.inner.next_seq();
            assert!(entry.finalise(
                &entry::FinalOutcome::Completed(serde_json::json!(index)),
                seq
            ));
        }
        pool.inner.prune_finished();

        assert_eq!(pool.len(), 2, "one finished entry was pruned");
        assert!(pool.get(&keys[0]).is_some(), "a leased entry survives");
        assert!(pool.get(&keys[1]).is_none(), "the middle entry was pruned");
        assert!(pool.get(&keys[2]).is_some(), "the newest entry survives");
        assert!(pool.list_active().is_empty(), "every entry is terminal");
    }

    #[test]
    fn pruning_is_bounded_by_the_configured_retention() {
        let pool = TeamPool::with_config(TeamPoolConfig::default().with_retain_finished(2));
        for session in ["s-1", "s-2", "s-3", "s-4"] {
            let key = PoolKey::new("crew", session);
            let entry = Arc::new(Entry::new(entry::EntryInit {
                run_id: format!("run-{session}"),
                key: key.clone(),
                label: "crew".to_owned(),
                deadline: None,
                payload_capacity: 2,
                frame_capacity: 2,
                hub: None,
                monitor: None,
            }));
            pool.inner.entries().insert(key, Arc::clone(&entry));
            let seq = pool.inner.next_seq();
            entry.finalise(&entry::FinalOutcome::Cancelled("test".to_owned()), seq);
            pool.inner.prune_finished();
            assert!(pool.len() <= 2, "the retention bound holds after every run");
        }
        assert_eq!(pool.len(), 2);
    }

    #[tokio::test]
    async fn stop_all_counts_only_live_runs_and_sweeps_finished_ones() {
        let pool = TeamPool::new();
        let keys = [
            PoolKey::new("crew", "s-done"),
            PoolKey::new("crew", "s-live"),
        ];
        let entries: Vec<Arc<Entry>> = keys
            .iter()
            .map(|key| {
                Arc::new(Entry::new(entry::EntryInit {
                    run_id: format!("run-{}", key.session_id),
                    key: key.clone(),
                    label: "crew".to_owned(),
                    deadline: None,
                    payload_capacity: 2,
                    frame_capacity: 2,
                    hub: None,
                    monitor: None,
                }))
            })
            .collect();
        for (key, entry) in keys.iter().zip(&entries) {
            pool.inner.entries().insert(key.clone(), Arc::clone(entry));
        }
        // One entry finished before the sweep; the other is still live (queued).
        let seq = pool.inner.next_seq();
        assert!(entries[0].finalise(
            &entry::FinalOutcome::Completed(serde_json::json!("done")),
            seq
        ));

        let shutdown = pool.stop_all().await;
        assert!(shutdown.is_ok(), "the sweep reported no failure");
        assert_eq!(
            shutdown.stopped, 1,
            "only the live run counts; the finished entry is swept but not counted"
        );
        for key in &keys {
            assert!(
                pool.get(key).is_none(),
                "the sweep removed every entry, finished or live"
            );
        }
        assert_eq!(pool.len(), 0);
        assert!(pool.is_closed());
    }

    #[tokio::test]
    async fn a_reaper_is_installed_at_most_once() {
        let pool = TeamPool::new();
        let first = ReaperHandle::detached(tokio_util::sync::CancellationToken::new());
        assert!(pool.inner.install_reaper(first));
        assert!(!pool.inner.install_reaper(ReaperHandle::detached(
            tokio_util::sync::CancellationToken::new()
        )));
        assert!(pool.inner.take_reaper().is_some());
        assert!(pool.inner.take_reaper().is_none());
    }

    #[test]
    fn closing_the_pool_is_reported_and_stops_admission() {
        let pool = TeamPool::new();
        assert!(!pool.is_closed());
        pool.inner.close();
        assert!(pool.is_closed());
    }

    #[test]
    fn pooled_services_clear_only_the_observation_seams() {
        let monitor = Arc::new(crate::monitor::RunMonitor::new());
        let hub = Arc::new(crate::hitl::InteractionHub::new());
        let services = RunServices::new(
            crate::resources::ResourceMgr::new(),
            crate::callback::CallbackFramework::new(),
            RunnerConfig::default(),
            None,
        )
        .with_monitor(Arc::clone(&monitor))
        .with_hitl(Arc::clone(&hub));
        let resources = Arc::clone(&services.resources);
        let pooled = run::pooled_services(services);
        assert!(pooled.monitor.is_none(), "the pool owns the monitor");
        assert!(pooled.hitl.is_none(), "the pool owns the hub");
        assert!(
            Arc::ptr_eq(&pooled.resources, &resources),
            "the registries are shared"
        );
    }
}
